// NimbleCAS arbitrary-precision binary floating point (BigFloat; software multi-precision).
// @author Olumuyiwa Oluwasanmi
//
// BigFloat is the software floating-point companion to the arbitrary-precision integer
// BigInt (nimblecas.bigint): the analogue of MPFR's mpf / GMP's mpf_t. A value is a
// sign-magnitude BigInt mantissa m scaled by a binary exponent e,
//
//     value = m * 2^e,
//
// carried to a caller-chosen precision `prec` (significant bits). After every operation
// the mantissa is renormalised: trailing zero bits are absorbed into the exponent (so the
// stored mantissa is the unique minimal odd integer, or exactly 0), and if the raw result
// carries more than `prec` significant bits it is rounded to `prec` bits round-to-nearest,
// ties-to-even, using an exact guard/round/sticky discipline. Because the canonical form
// is unique per value, dyadic numbers (0.5, 0.75, 3.25, ...) are represented EXACTLY and
// equality is a structural compare.
//
// HONESTY — read this before trusting a result:
//   * This is SOFTWARE arbitrary precision. No hardware 128/256-bit binary float exists;
//     every bit beyond a machine double is produced by BigInt limb arithmetic in software.
//   * It is still FLOATING POINT, hence INEXACT: results are rounded to `prec` bits. For
//     EXACT arithmetic use nimblecas.bigint (integers) or Rational (nimblecas.ratpoly).
//   * Rounding is round-to-nearest, ties-to-even throughout. add / subtract / multiply are
//     CORRECTLY ROUNDED (the exact result is computed, then rounded once). divide and sqrt
//     are ALSO correctly rounded: the quotient / integer square root is computed to more
//     than `prec` bits with an exact remainder feeding the sticky bit, so the single final
//     rounding is to-nearest-even with no double rounding.
//   * AVX / AVX2 / AVX-512 do NOT change any single-operation result or its latency; they
//     only raise THROUGHPUT of the limb-level mantissa multiply and of the batched
//     reductions (bigfloat_sum / bigfloat_dot). AVX-512 IFMA's 52-bit multiply-add is the
//     natural primitive for the schoolbook limb multiply inside BigInt. The reductions here
//     are exposed with a correct SCALAR reference path (below); any SIMD acceleration must
//     live inside BigInt's limb kernels (nimblecas.simd waterfall: AVX-512 -> AVX2 ->
//     scalar) and produce bit-identical mantissas, so the BigFloat results are unchanged.
//
// Railway-oriented (Rule 32): fallible entry points return Result<BigFloat>. Division by
// zero -> division_by_zero; sqrt of a negative value -> domain_error; a malformed string
// -> syntax_error; a non-positive precision -> domain_error; an exponent that would
// overflow std::int64_t -> overflow.

module;
#include <cassert>

export module nimblecas.bigfloat;

import std;
import nimblecas.core;
import nimblecas.bigint;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// BigFloat — value = mantissa * 2^exponent, rounded to `precision` bits.
// ---------------------------------------------------------------------------
// Canonical form: either mantissa == 0 (with exponent 0, the unique zero), or the mantissa
// is odd (no trailing zero bits) and has at most `precision` significant bits. Equality and
// ordering therefore compare values, not representations.
class BigFloat {
public:
    // --- construction (all fallible on a non-positive precision) ---------------------

    // The integer v at `prec` bits (exact whenever v fits in `prec` significant bits).
    [[nodiscard]] static auto from_i64(std::int64_t v, std::int64_t prec) -> Result<BigFloat>;
    // The arbitrary-precision integer b at `prec` bits.
    [[nodiscard]] static auto from_bigint(const BigInt& b, std::int64_t prec) -> Result<BigFloat>;
    // The IEEE double d converted EXACTLY (a double is itself a dyadic rational) then
    // rounded to `prec` bits. NaN / infinity have no representation -> domain_error.
    [[nodiscard]] static auto from_double(double d, std::int64_t prec) -> Result<BigFloat>;
    // Parse a decimal literal: optional sign, integer and/or fraction digits, optional
    // e/E exponent (e.g. "-3.14", "1.25e2", ".5", "6E-3"). The decimal value is formed
    // exactly and then rounded to `prec` bits. A malformed literal -> syntax_error.
    [[nodiscard]] static auto from_string(std::string_view text, std::int64_t prec)
        -> Result<BigFloat>;

    // Re-round this value to a new precision (widen: exact; narrow: round-to-nearest-even).
    [[nodiscard]] auto with_precision(std::int64_t prec) const -> Result<BigFloat>;

    // --- accessors -------------------------------------------------------------------

    [[nodiscard]] auto is_zero() const noexcept -> bool { return mant_.is_zero(); }
    [[nodiscard]] auto precision() const noexcept -> std::int64_t { return prec_; }
    // -1, 0, or +1.
    [[nodiscard]] auto sign() const -> int {
        if (mant_.is_zero()) {
            return 0;
        }
        return mant_.is_negative() ? -1 : 1;
    }

    // Nearest double (may overflow to +-inf / underflow to 0 for out-of-range exponents).
    [[nodiscard]] auto to_double() const -> double;
    // Fixed-point decimal with exactly `decimal_digits` fractional digits, rounded to
    // nearest (ties to even). Shows the full carried precision, e.g. 1/3 -> "0.3333...".
    [[nodiscard]] auto to_string(std::size_t decimal_digits) const -> std::string;

    // --- arithmetic at an explicit result precision (round-to-nearest-even) -----------

    [[nodiscard]] auto add(const BigFloat& o, std::int64_t prec) const -> Result<BigFloat>;
    [[nodiscard]] auto subtract(const BigFloat& o, std::int64_t prec) const -> Result<BigFloat>;
    [[nodiscard]] auto multiply(const BigFloat& o, std::int64_t prec) const -> Result<BigFloat>;
    // Long division of the mantissas to prec + guard bits, then rounded. Divisor 0 -> error.
    [[nodiscard]] auto divide(const BigFloat& o, std::int64_t prec) const -> Result<BigFloat>;
    // Square root via integer-sqrt (Newton) on the scaled mantissa. Negative -> domain_error.
    [[nodiscard]] auto sqrt(std::int64_t prec) const -> Result<BigFloat>;

    // The additive inverse (a pure sign flip; the precision is unchanged).
    [[nodiscard]] auto negate() const -> BigFloat { return BigFloat{mant_.negate(), exp_, prec_}; }

    // --- comparison (by value; precision is irrelevant) -------------------------------

    [[nodiscard]] auto operator<=>(const BigFloat& o) const -> std::strong_ordering;
    [[nodiscard]] auto operator==(const BigFloat& o) const -> bool {
        return (*this <=> o) == std::strong_ordering::equal;
    }

private:
    BigFloat(BigInt mant, std::int64_t exp, std::int64_t prec)
        : mant_(std::move(mant)), exp_(exp), prec_(prec) {}

    // Round m * 2^e to `prec` significant bits (round-to-nearest, ties-to-even) and reduce
    // to canonical form. `sticky` records nonzero content below the mantissa's LSB (from a
    // division / sqrt remainder), breaking exact ties upward. prec must be > 0.
    [[nodiscard]] static auto normalize(BigInt m, std::int64_t e, std::int64_t prec,
                                        bool sticky) -> Result<BigFloat>;

    // Round (num / den) * 2^e_bias to `prec` bits. den must be non-zero, prec > 0.
    [[nodiscard]] static auto ratio(const BigInt& num, const BigInt& den, std::int64_t e_bias,
                                    std::int64_t prec) -> Result<BigFloat>;

    BigInt mant_;         // sign-magnitude mantissa; canonical == odd or exactly 0
    std::int64_t exp_{};  // binary exponent: value = mant_ * 2^exp_
    std::int64_t prec_{}; // significant-bit budget (> 0 for every constructed value)
};

// ---------------------------------------------------------------------------
// Batched reductions (documented SIMD payoff; correct scalar reference here).
// ---------------------------------------------------------------------------
// A left-fold accumulation carried at prec + guard bits then rounded once to prec. The
// throughput win from AVX/AVX2/AVX-512 lives inside BigInt's limb add/multiply kernels
// (dispatched through the nimblecas.simd waterfall); it does not alter these results.
[[nodiscard]] auto bigfloat_sum(std::span<const BigFloat> xs, std::int64_t prec)
    -> Result<BigFloat>;
[[nodiscard]] auto bigfloat_dot(std::span<const BigFloat> a, std::span<const BigFloat> b,
                                std::int64_t prec) -> Result<BigFloat>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// --- small BigInt constants (built from the documented from_i64 factory) --------------

[[nodiscard]] auto bi_zero() -> BigInt { return BigInt::from_i64(0); }
[[nodiscard]] auto bi_one() -> BigInt { return BigInt::from_i64(1); }
[[nodiscard]] auto bi_two() -> BigInt { return BigInt::from_i64(2); }
[[nodiscard]] auto bi_five() -> BigInt { return BigInt::from_i64(5); }
[[nodiscard]] auto bi_ten() -> BigInt { return BigInt::from_i64(10); }

// --- overflow-checked int64 exponent arithmetic --------------------------------------

[[nodiscard]] auto add_ov(std::int64_t a, std::int64_t b, std::int64_t& out) -> bool {
    return __builtin_add_overflow(a, b, &out);
}
[[nodiscard]] auto sub_ov(std::int64_t a, std::int64_t b, std::int64_t& out) -> bool {
    return __builtin_sub_overflow(a, b, &out);
}
[[nodiscard]] auto mul_ov(std::int64_t a, std::int64_t b, std::int64_t& out) -> bool {
    return __builtin_mul_overflow(a, b, &out);
}

// --- comparison / parity helpers (only the documented BigInt surface is used) ---------

// Three-way compare collapsed to strong_ordering irrespective of BigInt's category.
[[nodiscard]] auto bi_cmp(const BigInt& a, const BigInt& b) -> std::strong_ordering {
    const auto c = a <=> b;
    if (c < 0) {
        return std::strong_ordering::less;
    }
    if (c > 0) {
        return std::strong_ordering::greater;
    }
    return std::strong_ordering::equal;
}

// floor(x / 2) for x >= 0 (2 is never zero, so divmod always yields a value).
[[nodiscard]] auto bi_half(const BigInt& x) -> BigInt {
    auto dm = x.divmod(bi_two());
    return dm ? dm->first : bi_zero();
}

// Least-significant bit of |x| (works for any sign: divmod truncates toward zero).
[[nodiscard]] auto bi_is_odd(const BigInt& x) -> bool {
    auto dm = x.divmod(bi_two());
    return dm && !dm->second.is_zero();
}

// --- bit length, powers of two/five/ten, and shifts (all local to BigFloat) -----------

// Number of significant bits of |x| (0 for x == 0). Doubling scan: O(bits) big multiplies.
[[nodiscard]] auto bit_length(const BigInt& x) -> std::int64_t {
    BigInt v = x.abs();
    if (v.is_zero()) {
        return 0;
    }
    std::int64_t k = 0;
    BigInt p = bi_one();  // invariant: p == 2^k and p <= v
    for (;;) {
        BigInt twice = p.multiply(bi_two());  // 2^(k+1)
        if (bi_cmp(twice, v) == std::strong_ordering::greater) {
            break;  // 2^(k+1) > v  =>  2^k <= v < 2^(k+1)
        }
        p = std::move(twice);
        ++k;
    }
    return k + 1;
}

// base^exp by exponentiation-by-squaring, using only BigInt::multiply.
[[nodiscard]] auto bi_pow(BigInt base, std::uint64_t exp) -> BigInt {
    BigInt result = bi_one();
    while (exp > 0) {
        if ((exp & 1U) != 0U) {
            result = result.multiply(base);
        }
        exp >>= 1U;
        if (exp > 0) {
            base = base.multiply(base);
        }
    }
    return result;
}

[[nodiscard]] auto pow2(std::uint64_t k) -> BigInt { return bi_pow(bi_two(), k); }
[[nodiscard]] auto pow5(std::uint64_t k) -> BigInt { return bi_pow(bi_five(), k); }
[[nodiscard]] auto pow10(std::uint64_t k) -> BigInt { return bi_pow(bi_ten(), k); }

// x * 2^k for k >= 0 (multiply by 2^k, per the module contract's shift_left helper).
[[nodiscard]] auto shift_left(const BigInt& x, std::int64_t k) -> BigInt {
    assert(k >= 0 && "shift_left with negative amount");
    return x.multiply(pow2(static_cast<std::uint64_t>(k)));
}

// --- BigInt <-> int64 and integer square root -----------------------------------------

// Exact conversion for a BigInt already known to fit in an int64 (via to_string).
[[nodiscard]] auto bigint_to_i64(const BigInt& x) -> std::optional<std::int64_t> {
    const std::string s = x.to_string();
    std::int64_t out = 0;
    const auto* first = s.data();
    const auto* last = s.data() + s.size();
    const auto res = std::from_chars(first, last, out);
    if (res.ec != std::errc{} || res.ptr != last) {
        return std::nullopt;
    }
    return out;
}

// floor(sqrt(n)) for n >= 0 via integer Newton iteration (all BigInt arithmetic).
[[nodiscard]] auto isqrt(const BigInt& n) -> BigInt {
    if (n.is_zero()) {
        return bi_zero();
    }
    const std::int64_t b = bit_length(n);
    // x0 = 2^ceil(b/2) >= sqrt(n): the classic overshoot guaranteeing monotone descent.
    BigInt x = pow2(static_cast<std::uint64_t>((b + 1) / 2));
    for (;;) {
        auto dm = n.divmod(x);  // x > 0
        BigInt q = dm ? dm->first : bi_zero();
        BigInt y = bi_half(x.add(q));  // floor((x + n/x) / 2)
        if (bi_cmp(y, x) != std::strong_ordering::less) {
            break;  // converged: y >= x  =>  x == floor(sqrt(n))
        }
        x = std::move(y);
    }
    return x;
}

// Guard bits kept beyond `prec` when a quotient / sqrt is computed before rounding. The
// remainder feeds the sticky bit, so a single guard bit suffices for correct rounding;
// two are kept for headroom.
constexpr std::int64_t guard_bits = 2;

}  // namespace

// --- normalisation / rounding ---------------------------------------------------------

auto BigFloat::normalize(BigInt m, std::int64_t e, std::int64_t prec, bool sticky)
    -> Result<BigFloat> {
    assert(prec > 0 && "normalize requires a positive precision");
    if (m.is_zero()) {
        return BigFloat{bi_zero(), 0, prec};  // the canonical zero (sticky below LSB is < ulp)
    }
    const bool neg = m.is_negative();
    BigInt mag = m.abs();
    const std::int64_t b = bit_length(mag);
    if (b > prec) {
        // Discard the low s bits with round-to-nearest, ties-to-even. rem holds every
        // discarded bit exactly; `sticky` adds sub-LSB content strictly below the round bit.
        const std::int64_t s = b - prec;
        auto dm = mag.divmod(pow2(static_cast<std::uint64_t>(s)));
        BigInt q = dm ? dm->first : bi_zero();   // mag >> s
        const BigInt r = dm ? dm->second : bi_zero();  // low s bits
        const BigInt half = pow2(static_cast<std::uint64_t>(s - 1));
        const auto c = bi_cmp(r, half);
        bool round_up = false;
        if (c == std::strong_ordering::greater) {
            round_up = true;
        } else if (c == std::strong_ordering::equal) {
            round_up = sticky ? true : bi_is_odd(q);  // exact tie -> to even (or up if sticky)
        }
        if (round_up) {
            q = q.add(bi_one());  // may carry to 2^prec; the trailing-zero pass below fixes it
        }
        mag = std::move(q);
        if (add_ov(e, s, e)) {
            return make_error<BigFloat>(MathError::overflow);
        }
    }
    // Absorb trailing zero bits into the exponent so the mantissa is the minimal odd form
    // (a rounding carry to a power of two collapses here to mantissa 1 with a raised exp).
    for (;;) {
        auto dm = mag.divmod(bi_two());
        if (!dm || !dm->second.is_zero() || dm->first.is_zero()) {
            break;  // odd (or fully reduced): canonical
        }
        mag = dm->first;
        if (e == std::numeric_limits<std::int64_t>::max()) {
            break;  // refuse to overflow the exponent; leaving a trailing zero is still correct
        }
        ++e;
    }
    return BigFloat{neg ? mag.negate() : mag, e, prec};
}

auto BigFloat::ratio(const BigInt& num, const BigInt& den, std::int64_t e_bias,
                     std::int64_t prec) -> Result<BigFloat> {
    if (den.is_zero()) {
        return make_error<BigFloat>(MathError::division_by_zero);
    }
    if (num.is_zero()) {
        return BigFloat{bi_zero(), 0, prec};
    }
    const bool neg = num.is_negative() != den.is_negative();
    const BigInt a = num.abs();
    const BigInt d = den.abs();
    // Shift the numerator so the quotient carries > prec significant bits; the exact
    // remainder then supplies the sticky bit, giving a correctly-rounded single rounding.
    const std::int64_t shift = bit_length(d) + prec + guard_bits;
    auto dm = shift_left(a, shift).divmod(d);
    BigInt q = dm ? dm->first : bi_zero();
    const bool sticky = dm && !dm->second.is_zero();
    std::int64_t e = 0;
    if (sub_ov(e_bias, shift, e)) {  // value = (a/d) * 2^e_bias = q * 2^(e_bias - shift) + ...
        return make_error<BigFloat>(MathError::overflow);
    }
    return normalize(neg ? q.negate() : std::move(q), e, prec, sticky);
}

// --- construction ---------------------------------------------------------------------

auto BigFloat::from_i64(std::int64_t v, std::int64_t prec) -> Result<BigFloat> {
    if (prec <= 0) {
        return make_error<BigFloat>(MathError::domain_error);
    }
    return normalize(BigInt::from_i64(v), 0, prec, false);
}

auto BigFloat::from_bigint(const BigInt& b, std::int64_t prec) -> Result<BigFloat> {
    if (prec <= 0) {
        return make_error<BigFloat>(MathError::domain_error);
    }
    return normalize(b, 0, prec, false);
}

auto BigFloat::from_double(double d, std::int64_t prec) -> Result<BigFloat> {
    if (prec <= 0) {
        return make_error<BigFloat>(MathError::domain_error);
    }
    if (std::isnan(d) || std::isinf(d)) {
        return make_error<BigFloat>(MathError::domain_error);
    }
    if (d == 0.0) {
        return BigFloat{bi_zero(), 0, prec};
    }
    // A finite double is exactly frac * 2^exp with frac in [0.5, 1); scaling frac by 2^53
    // yields an exact 53-bit integer mantissa. So d == mantissa * 2^(exp - 53), exactly.
    int exp = 0;
    const double frac = std::frexp(d, &exp);
    const double scaled = std::ldexp(frac, 53);           // integer-valued, |.| in [2^52, 2^53)
    const auto mantissa = static_cast<std::int64_t>(scaled);  // exact: |mantissa| < 2^53
    return normalize(BigInt::from_i64(mantissa), static_cast<std::int64_t>(exp) - 53, prec, false);
}

auto BigFloat::from_string(std::string_view text, std::int64_t prec) -> Result<BigFloat> {
    if (prec <= 0) {
        return make_error<BigFloat>(MathError::domain_error);
    }
    const auto is_digit = [](char c) { return c >= '0' && c <= '9'; };
    std::size_t i = 0;
    const std::size_t n = text.size();
    if (n == 0) {
        return make_error<BigFloat>(MathError::syntax_error);
    }

    bool neg = false;
    if (text[i] == '+' || text[i] == '-') {
        neg = text[i] == '-';
        ++i;
    }

    // Integer and fraction digits fold into one big integer of digits; fraction length
    // records the implied division by 10^frac_len.
    std::string digits;
    std::int64_t frac_len = 0;
    bool any_digit = false;
    while (i < n && is_digit(text[i])) {
        digits.push_back(text[i]);
        ++i;
        any_digit = true;
    }
    if (i < n && text[i] == '.') {
        ++i;
        while (i < n && is_digit(text[i])) {
            digits.push_back(text[i]);
            ++frac_len;
            ++i;
            any_digit = true;
        }
    }
    if (!any_digit) {
        return make_error<BigFloat>(MathError::syntax_error);
    }

    std::int64_t decimal_exp = 0;
    if (i < n && (text[i] == 'e' || text[i] == 'E')) {
        ++i;
        bool exp_neg = false;
        if (i < n && (text[i] == '+' || text[i] == '-')) {
            exp_neg = text[i] == '-';
            ++i;
        }
        bool any_exp_digit = false;
        std::int64_t value = 0;
        while (i < n && is_digit(text[i])) {
            const std::int64_t dv = text[i] - '0';
            if (mul_ov(value, 10, value) || add_ov(value, dv, value)) {
                return make_error<BigFloat>(MathError::overflow);
            }
            ++i;
            any_exp_digit = true;
        }
        if (!any_exp_digit) {
            return make_error<BigFloat>(MathError::syntax_error);
        }
        decimal_exp = exp_neg ? -value : value;
    }
    if (i != n) {
        return make_error<BigFloat>(MathError::syntax_error);  // trailing garbage
    }

    // value = sign * digits * 10^(decimal_exp - frac_len).
    std::string int_text;
    if (neg) {
        int_text.push_back('-');
    }
    int_text += digits;
    auto parsed = BigInt::from_string(int_text);
    if (!parsed) {
        return make_error<BigFloat>(MathError::syntax_error);
    }
    const BigInt n_int = *parsed;
    if (n_int.is_zero()) {
        return BigFloat{bi_zero(), 0, prec};
    }
    std::int64_t p = 0;
    if (sub_ov(decimal_exp, frac_len, p)) {
        return make_error<BigFloat>(MathError::overflow);
    }
    if (p >= 0) {
        // 10^p = 2^p * 5^p, so value = (digits * 5^p) * 2^p — an exact dyadic, then rounded.
        BigInt scaled = n_int.multiply(pow5(static_cast<std::uint64_t>(p)));
        return normalize(std::move(scaled), p, prec, false);
    }
    // p < 0: value = n_int / 5^k * 2^p with k = -p; a rounded binary division.
    std::int64_t k = 0;
    if (sub_ov(0, p, k)) {  // guards p == INT64_MIN
        return make_error<BigFloat>(MathError::overflow);
    }
    return ratio(n_int, pow5(static_cast<std::uint64_t>(k)), p, prec);
}

auto BigFloat::with_precision(std::int64_t prec) const -> Result<BigFloat> {
    if (prec <= 0) {
        return make_error<BigFloat>(MathError::domain_error);
    }
    return normalize(mant_, exp_, prec, false);
}

// --- accessors ------------------------------------------------------------------------

auto BigFloat::to_double() const -> double {
    if (mant_.is_zero()) {
        return 0.0;
    }
    const bool neg = mant_.is_negative();
    const BigInt mag = mant_.abs();
    const std::int64_t b = bit_length(mag);
    const std::int64_t s = b > 53 ? b - 53 : 0;  // shift off all but the top <= 53 bits
    BigInt top;
    if (s == 0) {
        top = mag;  // already <= 53 bits: exact, nothing to round
    } else {
        // Round the discarded low s bits to NEAREST, ties to even (the "nearest double"
        // contract) rather than truncating toward zero.
        auto dm = mag.divmod(pow2(static_cast<std::uint64_t>(s)));
        top = dm ? dm->first : bi_zero();
        const BigInt rem = dm ? dm->second : bi_zero();
        const BigInt half = pow2(static_cast<std::uint64_t>(s - 1));
        const auto cmp = rem <=> half;
        bool round_up = cmp > 0;
        if (cmp == 0) {  // exact halfway: round to even (up iff the kept value is odd)
            auto parity = top.divmod(BigInt::from_i64(2));
            round_up = parity && !parity->second.is_zero();
        }
        if (round_up) {
            top = top.add(BigInt::from_i64(1));  // may carry to 2^53, still an exact double
        }
    }
    const auto ti = bigint_to_i64(top);           // <= 2^53 -> always fits i64 and double
    double value = ti ? static_cast<double>(*ti) : 0.0;

    std::int64_t expo = 0;
    if (add_ov(exp_, s, expo)) {
        value = std::numeric_limits<double>::infinity();
        return neg ? -value : value;
    }
    constexpr std::int64_t clamp = 1 << 20;  // ldexp takes int; clamp far past double's range
    if (expo > clamp) {
        value = std::numeric_limits<double>::infinity();
    } else if (expo < -clamp) {
        value = 0.0;
    } else {
        value = std::ldexp(value, static_cast<int>(expo));
    }
    return neg ? -value : value;
}

auto BigFloat::to_string(std::size_t decimal_digits) const -> std::string {
    const auto dd = static_cast<std::uint64_t>(decimal_digits);
    const bool neg = mant_.is_negative() && !mant_.is_zero();
    const BigInt mag = mant_.abs();

    // scaled = round(|value| * 10^dd) as a non-negative integer.
    BigInt scaled = bi_zero();
    const BigInt num = mag.multiply(pow10(dd));  // |mantissa| * 10^dd
    if (exp_ >= 0) {
        scaled = num.multiply(pow2(static_cast<std::uint64_t>(exp_)));  // exact
    } else {
        const auto k = static_cast<std::uint64_t>(-(exp_ + 1)) + 1U;  // = -exp_, overflow-safe
        auto dm = num.divmod(pow2(k));
        BigInt q = dm ? dm->first : bi_zero();
        const BigInt r = dm ? dm->second : bi_zero();
        const BigInt half = pow2(k - 1);
        const auto c = bi_cmp(r, half);
        if (c == std::strong_ordering::greater ||
            (c == std::strong_ordering::equal && bi_is_odd(q))) {
            q = q.add(bi_one());  // round-to-nearest, ties-to-even
        }
        scaled = std::move(q);
    }

    std::string ds = scaled.to_string();  // magnitude only, no sign
    const bool is_zero_value = scaled.is_zero();
    std::string out;
    if (neg && !is_zero_value) {
        out.push_back('-');
    }
    if (dd == 0) {
        out += ds;
        return out;
    }
    if (ds.size() <= decimal_digits) {
        ds.insert(0, decimal_digits - ds.size() + 1, '0');  // ensure an integer digit exists
    }
    const std::size_t point = ds.size() - decimal_digits;
    out += ds.substr(0, point);
    out.push_back('.');
    out += ds.substr(point);
    return out;
}

// --- arithmetic -----------------------------------------------------------------------

auto BigFloat::add(const BigFloat& o, std::int64_t prec) const -> Result<BigFloat> {
    if (prec <= 0) {
        return make_error<BigFloat>(MathError::domain_error);
    }
    if (is_zero()) {
        return o.with_precision(prec);
    }
    if (o.is_zero()) {
        return with_precision(prec);
    }
    // Align to the smaller exponent (an exact shift), add, then round once.
    const std::int64_t low = std::min(exp_, o.exp_);
    const BigInt a = shift_left(mant_, exp_ - low);
    const BigInt b = shift_left(o.mant_, o.exp_ - low);
    return normalize(a.add(b), low, prec, false);
}

auto BigFloat::subtract(const BigFloat& o, std::int64_t prec) const -> Result<BigFloat> {
    return add(o.negate(), prec);
}

auto BigFloat::multiply(const BigFloat& o, std::int64_t prec) const -> Result<BigFloat> {
    if (prec <= 0) {
        return make_error<BigFloat>(MathError::domain_error);
    }
    if (is_zero() || o.is_zero()) {
        return BigFloat{bi_zero(), 0, prec};
    }
    std::int64_t e = 0;
    if (add_ov(exp_, o.exp_, e)) {
        return make_error<BigFloat>(MathError::overflow);
    }
    return normalize(mant_.multiply(o.mant_), e, prec, false);
}

auto BigFloat::divide(const BigFloat& o, std::int64_t prec) const -> Result<BigFloat> {
    if (prec <= 0) {
        return make_error<BigFloat>(MathError::domain_error);
    }
    if (o.is_zero()) {
        return make_error<BigFloat>(MathError::division_by_zero);
    }
    if (is_zero()) {
        return BigFloat{bi_zero(), 0, prec};
    }
    std::int64_t bias = 0;
    if (sub_ov(exp_, o.exp_, bias)) {  // (m/o.m) * 2^(exp_ - o.exp_)
        return make_error<BigFloat>(MathError::overflow);
    }
    return ratio(mant_, o.mant_, bias, prec);
}

auto BigFloat::sqrt(std::int64_t prec) const -> Result<BigFloat> {
    if (prec <= 0) {
        return make_error<BigFloat>(MathError::domain_error);
    }
    if (!mant_.is_zero() && mant_.is_negative()) {
        return make_error<BigFloat>(MathError::domain_error);
    }
    if (is_zero()) {
        return BigFloat{bi_zero(), 0, prec};
    }
    // sqrt(mag * 2^exp_) = sqrt(mag * 2^shift) * 2^((exp_ - shift)/2). Choose an even
    // (exp_ - shift) and enough shift that the radicand has ~2*(prec+guard) bits, so its
    // integer square root carries > prec bits; the isqrt remainder is the sticky bit.
    const BigInt mag = mant_.abs();
    const std::int64_t bm = bit_length(mag);
    const std::int64_t want = 2 * (prec + guard_bits);
    std::int64_t shift = want > bm ? want - bm : 0;
    std::int64_t half_exp = 0;
    if (sub_ov(exp_, shift, half_exp)) {
        return make_error<BigFloat>(MathError::overflow);
    }
    if ((half_exp & 1LL) != 0) {  // make (exp_ - shift) even so the halving is exact
        ++shift;
        if (sub_ov(exp_, shift, half_exp)) {
            return make_error<BigFloat>(MathError::overflow);
        }
    }
    const BigInt radicand = shift_left(mag, shift);
    const BigInt root = isqrt(radicand);
    const bool sticky = bi_cmp(root.multiply(root), radicand) != std::strong_ordering::equal;
    return normalize(root, half_exp / 2, prec, sticky);
}

// --- comparison -----------------------------------------------------------------------

auto BigFloat::operator<=>(const BigFloat& o) const -> std::strong_ordering {
    // Align both mantissas to a common exponent (exact shifts) and compare signed integers.
    const std::int64_t low = std::min(exp_, o.exp_);
    const BigInt a = shift_left(mant_, exp_ - low);
    const BigInt b = shift_left(o.mant_, o.exp_ - low);
    return bi_cmp(a, b);
}

// --- batched reductions ---------------------------------------------------------------

auto bigfloat_sum(std::span<const BigFloat> xs, std::int64_t prec) -> Result<BigFloat> {
    if (prec <= 0) {
        return make_error<BigFloat>(MathError::domain_error);
    }
    if (xs.empty()) {
        return BigFloat::from_i64(0, prec);
    }
    // Accumulate at extra precision so the intermediate roundings do not accrete error,
    // then round once to the requested precision.
    std::int64_t work = 0;
    if (add_ov(prec, 8, work)) {
        return make_error<BigFloat>(MathError::overflow);
    }
    auto acc = xs[0].with_precision(work);
    if (!acc) {
        return acc;
    }
    for (std::size_t i = 1; i < xs.size(); ++i) {
        acc = acc->add(xs[i], work);
        if (!acc) {
            return acc;
        }
    }
    return acc->with_precision(prec);
}

auto bigfloat_dot(std::span<const BigFloat> a, std::span<const BigFloat> b, std::int64_t prec)
    -> Result<BigFloat> {
    if (prec <= 0) {
        return make_error<BigFloat>(MathError::domain_error);
    }
    if (a.size() != b.size()) {
        return make_error<BigFloat>(MathError::domain_error);
    }
    std::int64_t work = 0;
    if (add_ov(prec, 8, work)) {
        return make_error<BigFloat>(MathError::overflow);
    }
    auto acc = BigFloat::from_i64(0, work);
    if (!acc) {
        return acc;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        auto term = a[i].multiply(b[i], work);
        if (!term) {
            return term;
        }
        acc = acc->add(*term, work);
        if (!acc) {
            return acc;
        }
    }
    return acc->with_precision(prec);
}

}  // namespace nimblecas
