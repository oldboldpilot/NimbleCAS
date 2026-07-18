// NimbleCAS exact base-10 decimal number (BigDecimal, on arbitrary-precision BigInt).
// @author Olumuyiwa Oluwasanmi
//
// WHY THIS TYPE EXISTS (and why BigRational is not enough).
// BigRational already supplies every decimal *value* exactly. What it cannot carry is a
// decimal *quantity*: the pair (value, scale) together with an explicit rounding policy.
// BigRational::normalise() reduces to lowest terms on every operation, so 2.50 becomes
// 5/2 — indistinguishable from 2.5. For money and for spreadsheet semantics that is a
// loss of information, not a simplification: scale ("this amount is stated to the cent")
// and rounding mode are the *contract*, not a rendering detail. Excel's DB rounds a rate
// to three decimals mid-algorithm; CUMIPMT sums per-period roundings — perform the round
// at a different point or in a different mode and the answer is WRONG, not less precise.
// Base-2 types are disqualified outright: BigFloat cannot represent 0.1 at any precision,
// so a decimal literal would enter already wrong, violating the honesty invariant at the
// door. BigDecimal is therefore the money type and the boundary quantizer that turns an
// exact/numerical result into a stated decimal amount.
//
// REPRESENTATION: BigInt unscaled_ and std::int32_t scale_, with
//     value == unscaled_ * 10^(-scale_).
// A POSITIVE scale means fractional digits (150 at scale 2 is 1.50); a NEGATIVE scale
// means trailing implied zeros (15 at scale -2 is 1500). Trailing zeros in unscaled_ are
// PRESERVED — scale is semantic, so 1.50 (150, 2) and 1.5 (15, 1) are different
// representations of the same numeric value. Equality/ordering are NUMERIC (2.50 == 2.5),
// matching the BigFloat precedent and side-stepping the Java equals-vs-compareTo trap;
// same_representation() is provided for the rare scale-sensitive test.
//
// HONESTY: every finite decimal is exact here — add/subtract/multiply and widening are
// exact and total (fallible only when the int32 scale would overflow). Division is the
// honesty-critical operation: divide(scale, Rounding) rounds under an EXPLICIT policy the
// caller supplies, while divide_exact() refuses with MathError::inexact when the quotient
// does not terminate in base 10 — it never silently rounds. There is NO ambient/thread-
// local rounding context (Java MathContext / Python decimal context): every fallible
// rounding takes its (scale, Rounding) by argument, exactly as BigFloat takes prec. That
// keeps the type immutable, thread-safe under TBB, and bit-identical across hosts by
// construction (integer-backed — stronger than the -ffast-math parity Rule 55 requires).

module;
#include <cassert>  // assert macro (unavailable via `import std`); active when !NDEBUG

export module nimblecas.bigdecimal;

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// Rounding modes — the seven IEEE 754 / Java BigDecimal directed and half-way
// rules. Each fallible-rounding operation takes one explicitly; there is no
// default and no stored mode, so a rounding decision is never made behind the
// caller's back.
// ---------------------------------------------------------------------------
enum class Rounding : std::uint8_t {
    half_even,  // banker's rounding: to nearest, ties to even (the money default)
    half_up,    // to nearest, ties away from zero (Excel's DB uses this at 3 decimals)
    half_down,  // to nearest, ties toward zero
    down,       // toward zero (truncate)
    up,         // away from zero
    ceiling,    // toward +infinity
    floor,      // toward -infinity
};

// ---------------------------------------------------------------------------
// BigDecimal — a scaled arbitrary-precision base-10 number, unscaled_ * 10^(-scale_).
// ---------------------------------------------------------------------------
class BigDecimal {
public:
    BigDecimal() = default;  // the canonical zero, 0 at scale 0

    // --- Factories (total) -------------------------------------------------
    // unscaled * 10^(-scale). No normalisation of trailing zeros: scale is kept as given.
    [[nodiscard]] static auto from_bigint(BigInt unscaled, std::int32_t scale = 0) -> BigDecimal;
    [[nodiscard]] static auto from_i64(std::int64_t v, std::int32_t scale = 0) -> BigDecimal;
    [[nodiscard]] static auto zero() -> BigDecimal { return BigDecimal{}; }
    [[nodiscard]] static auto one() -> BigDecimal { return from_i64(1); }

    // --- Factories (fallible) ----------------------------------------------
    // Parse an exact decimal literal: optional sign, digits, optional '.', optional
    // 'e'/'E' signed exponent. The scale is (fraction digits) - (exponent), so "1.50"
    // -> (150, 2) and "1.5e3" -> (15, -2). Malformed input -> syntax_error; a scale that
    // will not fit int32 -> overflow (guards a hostile "1e-9999999999").
    [[nodiscard]] static auto from_string(std::string_view text) -> Result<BigDecimal>;
    // Quantize a double onto the given scale under an explicit mode. There is deliberately
    // NO scale-free overload, so the Java `new BigDecimal(0.1)` binary-artifact trap is
    // unrepresentable: the double's exact dyadic value is taken and then rounded to scale.
    // NaN/inf -> domain_error.
    [[nodiscard]] static auto from_double(double d, std::int32_t scale, Rounding mode)
        -> Result<BigDecimal>;
    // Quantize an exact rational onto scale under an explicit mode (total but for scale
    // overflow) — the reverse of to_bigrational().
    [[nodiscard]] static auto from_bigrational(const BigRational& r, std::int32_t scale,
                                               Rounding mode) -> BigDecimal;
    // Exact rational -> decimal WITHOUT rounding: succeeds iff the reduced denominator is
    // of the form 2^a * 5^b (the quotient terminates in base 10); otherwise inexact.
    [[nodiscard]] static auto from_bigrational_exact(const BigRational& r)
        -> Result<BigDecimal>;

    // --- Accessors ---------------------------------------------------------
    [[nodiscard]] auto unscaled() const noexcept -> const BigInt& { return unscaled_; }
    [[nodiscard]] auto scale() const noexcept -> std::int32_t { return scale_; }
    [[nodiscard]] auto is_zero() const noexcept -> bool { return unscaled_.is_zero(); }
    [[nodiscard]] auto sign() const noexcept -> int { return unscaled_.sign(); }
    [[nodiscard]] auto negate() const -> BigDecimal { return BigDecimal{unscaled_.negate(), scale_}; }
    [[nodiscard]] auto abs() const -> BigDecimal { return BigDecimal{unscaled_.abs(), scale_}; }
    // Plain decimal notation, trailing zeros preserved, never scientific (a deliberate
    // divergence from Java's toString): "1.50", "1500", "0.007", "-0.00".
    [[nodiscard]] auto to_string() const -> std::string;
    // Exact, total: unscaled_ / 10^scale_ as a reduced rational.
    [[nodiscard]] auto to_bigrational() const -> BigRational;

    // --- Arithmetic (exact) ------------------------------------------------
    // Align to max(scale, o.scale) by a power-of-ten limb shift (NO gcd — money amounts
    // share power-of-ten denominators), then add/subtract. Infallible: the aligned scale
    // is a max of two int32 scales, and BigInt cannot overflow.
    [[nodiscard]] auto add(const BigDecimal& o) const -> BigDecimal;
    [[nodiscard]] auto subtract(const BigDecimal& o) const -> BigDecimal;
    // unscaled_ * o.unscaled_ at scale_ + o.scale_. Mathematically exact; fallible only
    // because the summed scale can overflow int32 -> overflow.
    [[nodiscard]] auto multiply(const BigDecimal& o) const -> Result<BigDecimal>;
    // Non-negative integer power (exp >= 0), exact; exp < 0 -> domain_error (an exact
    // reciprocal power belongs in BigRational, which finance uses for (1+r)^n). Scale
    // overflow -> overflow.
    [[nodiscard]] auto pow(std::int64_t exp) const -> Result<BigDecimal>;

    // --- Division / rounding ----------------------------------------------
    // this / o rounded to result_scale under an explicit mode. Zero divisor ->
    // division_by_zero.
    [[nodiscard]] auto divide(const BigDecimal& o, std::int32_t result_scale, Rounding mode) const
        -> Result<BigDecimal>;
    // this / o WITHOUT rounding: succeeds iff the quotient terminates in base 10, else
    // inexact (the honesty invariant applied to division). Zero divisor ->
    // division_by_zero.
    [[nodiscard]] auto divide_exact(const BigDecimal& o) const -> Result<BigDecimal>;
    // Re-express at new_scale. Widening (new_scale >= scale_) is exact and appends zeros;
    // narrowing rounds under the explicit mode. Total: every value is representable at
    // every scale once a mode is supplied.
    [[nodiscard]] auto quantize(std::int32_t new_scale, Rounding mode) const -> BigDecimal;
    // Alias reading better at call sites that "round to N places".
    [[nodiscard]] auto round(std::int32_t new_scale, Rounding mode) const -> BigDecimal {
        return quantize(new_scale, mode);
    }

    // --- Comparison --------------------------------------------------------
    // NUMERIC three-way compare: 2.50 <=> 2.5 is equal. Aligns to the common scale by a
    // non-negative power-of-ten shift, so the sign of the shifted-integer compare is the
    // sign of the true difference (a total order).
    [[nodiscard]] auto operator<=>(const BigDecimal& o) const -> std::strong_ordering;
    [[nodiscard]] auto operator==(const BigDecimal& o) const -> bool {
        return (*this <=> o) == std::strong_ordering::equal;
    }
    // Representation identity (scale-sensitive): 2.50 and 2.5 are NOT same_representation.
    [[nodiscard]] auto same_representation(const BigDecimal& o) const -> bool {
        return scale_ == o.scale_ && unscaled_ == o.unscaled_;
    }

private:
    // Raw constructor: value == unscaled * 10^(-scale), no processing.
    BigDecimal(BigInt unscaled, std::int32_t scale)
        : unscaled_(std::move(unscaled)), scale_(scale) {}

    BigInt unscaled_{};        // 0 by default; carries the value's sign
    std::int32_t scale_{0};    // value == unscaled_ * 10^(-scale_)
};

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

[[nodiscard]] auto bi_ten() -> BigInt { return BigInt::from_u64(10); }
[[nodiscard]] auto bi_one() -> BigInt { return BigInt::from_u64(1); }
[[nodiscard]] auto bi_two() -> BigInt { return BigInt::from_u64(2); }

// Largest |scale| we will represent. A BigDecimal at scale S can require a 10^|S| BigInt
// (~|S| decimal digits) on quantize / divide / to_bigrational / to_string, so |S| is a hard
// DoS lever on any untrusted input (a ~15-byte "1e-2000000000" fits inside int32 but detonates
// on first use). 10^6 fractional digits (~sub-MB) is far beyond any real money/rate need;
// anything larger is rejected as overflow at the parse boundary rather than allocated.
inline constexpr std::int64_t kMaxScaleMagnitude = 1'000'000;

// 10^k for k >= 0.
[[nodiscard]] auto pow10(std::uint64_t k) -> BigInt { return bi_ten().pow(k); }

// Non-negative int64 -> uint64 magnitude, safe for INT64_MIN (never negated in signed
// space). Callers guard k >= 0 before use where only non-negative k is meaningful.
[[nodiscard]] auto abs_u64(std::int64_t k) -> std::uint64_t {
    return k < 0 ? (0ULL - static_cast<std::uint64_t>(k)) : static_cast<std::uint64_t>(k);
}

// BigInt division exact-or-assert (used where the caller has proven divisibility).
[[nodiscard]] auto divide_checked(const BigInt& a, const BigInt& b) -> BigInt {
    auto q = a.divide(b);
    assert(q.has_value() && "divide_checked requires a non-zero divisor");
    return *q;
}

// Odd-parity test on a BigInt magnitude via a divmod-by-two remainder.
[[nodiscard]] auto is_odd(const BigInt& x) -> bool {
    auto dm = x.divmod(bi_two());
    assert(dm.has_value() && "is_odd: divisor 2 is non-zero");
    return !dm->second.is_zero();
}

// Round N / D to an integer under the given mode. D must be non-zero (asserted). The
// result's sign is sign(N) * sign(D); the increment decision uses magnitudes only. This
// single helper backs divide(), quantize()-narrowing, from_double() and
// from_bigrational().
[[nodiscard]] auto round_div(const BigInt& N, const BigInt& D, Rounding mode) -> BigInt {
    assert(!D.is_zero() && "round_div requires a non-zero divisor");
    const bool negative = (N.sign() * D.sign()) < 0;
    const BigInt a = N.abs();
    const BigInt b = D.abs();
    auto dm = a.divmod(b);  // b > 0, so this never fails
    assert(dm.has_value() && "round_div: divisor magnitude is non-zero");
    const BigInt& q = dm->first;
    const BigInt& r = dm->second;

    bool increment = false;
    if (!r.is_zero()) {
        const std::strong_ordering twice = r.multiply(bi_two()) <=> b;  // 2r vs D
        switch (mode) {
            case Rounding::down:      increment = false; break;
            case Rounding::up:        increment = true; break;
            case Rounding::floor:     increment = negative; break;
            case Rounding::ceiling:   increment = !negative; break;
            case Rounding::half_up:   increment = (twice != std::strong_ordering::less); break;
            case Rounding::half_down: increment = (twice == std::strong_ordering::greater); break;
            case Rounding::half_even:
                increment = (twice == std::strong_ordering::greater) ||
                            (twice == std::strong_ordering::equal && is_odd(q));
                break;
        }
    }
    BigInt mag = increment ? q.add(bi_one()) : q;
    return negative ? mag.negate() : mag;
}

// Exact double -> BigRational (double is a dyadic rational). NaN/inf -> domain_error.
[[nodiscard]] auto double_to_bigrational(double d) -> Result<BigRational> {
    if (!std::isfinite(d)) {
        return make_error<BigRational>(MathError::domain_error);
    }
    if (d == 0.0) {
        return BigRational::from_int(0);
    }
    int exp = 0;
    const double frac = std::frexp(d, &exp);          // d = frac * 2^exp, |frac| in [0.5,1)
    const double scaled = std::ldexp(frac, 53);        // frac * 2^53 is an exact integer
    const std::int64_t mantissa = static_cast<std::int64_t>(scaled);
    const std::int64_t e2 = static_cast<std::int64_t>(exp) - 53;  // d = mantissa * 2^e2
    const BigInt m = BigInt::from_i64(mantissa);
    if (e2 >= 0) {
        return BigRational::from_bigint(m.multiply(bi_two().pow(static_cast<std::uint64_t>(e2))));
    }
    // d = mantissa / 2^(-e2); denominator is a power of two, so this never divides by zero.
    return BigRational::make(m, bi_two().pow(static_cast<std::uint64_t>(-e2)));
}

}  // namespace

// --- Factories (total) ------------------------------------------------------

auto BigDecimal::from_bigint(BigInt unscaled, std::int32_t scale) -> BigDecimal {
    return BigDecimal{std::move(unscaled), scale};
}

auto BigDecimal::from_i64(std::int64_t v, std::int32_t scale) -> BigDecimal {
    return BigDecimal{BigInt::from_i64(v), scale};
}

// --- Factories (fallible) ---------------------------------------------------

auto BigDecimal::from_string(std::string_view text) -> Result<BigDecimal> {
    if (text.empty()) {
        return make_error<BigDecimal>(MathError::syntax_error);
    }
    std::size_t i = 0;
    std::string digits;               // sign + all significant digits, fed to BigInt
    digits.reserve(text.size());
    if (text[i] == '+' || text[i] == '-') {
        if (text[i] == '-') { digits.push_back('-'); }
        ++i;
    }
    std::int64_t frac_digits = 0;     // digits seen after the '.'
    bool saw_digit = false;
    bool saw_dot = false;
    for (; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '.') {
            if (saw_dot) { return make_error<BigDecimal>(MathError::syntax_error); }
            saw_dot = true;
        } else if (c >= '0' && c <= '9') {
            digits.push_back(c);
            saw_digit = true;
            if (saw_dot) { ++frac_digits; }
        } else if (c == 'e' || c == 'E') {
            break;  // exponent handled below
        } else {
            return make_error<BigDecimal>(MathError::syntax_error);
        }
    }
    if (!saw_digit) {
        return make_error<BigDecimal>(MathError::syntax_error);
    }
    std::int64_t exponent = 0;
    if (i < text.size() && (text[i] == 'e' || text[i] == 'E')) {
        ++i;
        auto exp_str = text.substr(i);
        // std::from_chars never accepts a leading '+', but a signed exponent may carry one
        // ("1.5e+3", as std::format emits); skip it so scientific round-trips parse.
        if (!exp_str.empty() && exp_str.front() == '+') { exp_str.remove_prefix(1); }
        if (exp_str.empty()) { return make_error<BigDecimal>(MathError::syntax_error); }
        auto [ptr, ec] = std::from_chars(exp_str.data(), exp_str.data() + exp_str.size(), exponent);
        if (ec == std::errc::result_out_of_range) {
            return make_error<BigDecimal>(MathError::overflow);  // exponent too large for int64
        }
        if (ec != std::errc{} || ptr != exp_str.data() + exp_str.size()) {
            return make_error<BigDecimal>(MathError::syntax_error);
        }
        // Reject a wildly out-of-range exponent NOW, before it enters the `frac_digits -
        // exponent` subtraction below: from_chars accepts any int64, so e.g. "1e-9223372036854775808"
        // (exponent == INT64_MIN) would make that subtraction signed-overflow UB at the exact
        // untrusted-input boundary. |exponent| past the scale bound cannot yield a legal scale.
        if (exponent > kMaxScaleMagnitude || exponent < -kMaxScaleMagnitude) {
            return make_error<BigDecimal>(MathError::overflow);
        }
    }
    // A lone sign with no magnitude (e.g. "-") is caught by saw_digit above; digits may be
    // just "-" + nothing only if !saw_digit, already rejected. Empty numeric part -> "0".
    if (digits.empty() || digits == "-") {
        return make_error<BigDecimal>(MathError::syntax_error);
    }
    auto mant = BigInt::from_string(digits);
    if (!mant) {
        return make_error<BigDecimal>(mant.error());
    }
    // scale = frac_digits - exponent, checked against the DoS bound (not merely int32): a
    // scale inside int32 but beyond kMaxScaleMagnitude would materialize a 10^|scale| BigInt
    // on first use. Reject it here, at the untrusted-input boundary.
    const std::int64_t scale64 = frac_digits - exponent;
    if (scale64 > kMaxScaleMagnitude || scale64 < -kMaxScaleMagnitude) {
        return make_error<BigDecimal>(MathError::overflow);
    }
    return BigDecimal{std::move(*mant), static_cast<std::int32_t>(scale64)};
}

auto BigDecimal::from_double(double d, std::int32_t scale, Rounding mode) -> Result<BigDecimal> {
    auto r = double_to_bigrational(d);
    if (!r) {
        return make_error<BigDecimal>(r.error());
    }
    return from_bigrational(*r, scale, mode);
}

auto BigDecimal::from_bigrational(const BigRational& r, std::int32_t scale, Rounding mode)
    -> BigDecimal {
    // value * 10^scale = num * 10^scale / den, rounded to an integer at the given scale.
    const BigInt& num = r.numerator();
    const BigInt& den = r.denominator();  // > 0 by BigRational invariant
    BigInt N = num;
    BigInt D = den;
    if (scale >= 0) {
        N = num.multiply(pow10(static_cast<std::uint64_t>(scale)));
    } else {
        D = den.multiply(pow10(abs_u64(scale)));
    }
    return BigDecimal{round_div(N, D, mode), scale};
}

auto BigDecimal::from_bigrational_exact(const BigRational& r) -> Result<BigDecimal> {
    // The reduced fraction p/q terminates in base 10 iff q = 2^a * 5^b. Factor 2s and 5s
    // out of q; if anything remains, the decimal does not terminate.
    BigInt q = r.denominator();  // > 0, reduced against the numerator
    std::uint64_t twos = 0;
    std::uint64_t fives = 0;
    while (true) {
        auto dm = q.divmod(bi_two());
        if (!dm->second.is_zero()) { break; }
        q = dm->first;
        ++twos;
    }
    while (true) {
        auto dm = q.divmod(BigInt::from_u64(5));
        if (!dm->second.is_zero()) { break; }
        q = dm->first;
        ++fives;
    }
    if (!(q == bi_one())) {
        return make_error<BigDecimal>(MathError::inexact);  // a prime other than 2 or 5 remains
    }
    // p/q = p * (10^m / original_q) at scale m, where m = max(twos, fives) and
    // 10^m / q = 2^(m-twos) * 5^(m-fives) is an exact integer.
    const std::uint64_t m = std::max(twos, fives);
    if (m > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        return make_error<BigDecimal>(MathError::overflow);
    }
    const BigInt factor = bi_two().pow(m - twos).multiply(BigInt::from_u64(5).pow(m - fives));
    const BigInt unscaled = r.numerator().multiply(factor);
    return BigDecimal{unscaled, static_cast<std::int32_t>(m)};
}

// --- Accessors --------------------------------------------------------------

auto BigDecimal::to_string() const -> std::string {
    const std::string mag = unscaled_.abs().to_string();  // "0" when zero
    const bool neg = unscaled_.is_negative();
    std::string body;
    if (scale_ <= 0) {
        // Integer with |scale_| implied trailing zeros.
        body = mag;
        if (scale_ < 0 && !unscaled_.is_zero()) {
            body.append(static_cast<std::size_t>(-static_cast<std::int64_t>(scale_)), '0');
        }
    } else {
        const std::size_t s = static_cast<std::size_t>(scale_);
        if (mag.size() > s) {
            const std::size_t point = mag.size() - s;
            body = mag.substr(0, point);
            body.push_back('.');
            body.append(mag.substr(point));
        } else {
            body = "0.";
            body.append(s - mag.size(), '0');
            body.append(mag);
        }
    }
    return (neg && !unscaled_.is_zero()) ? "-" + body : body;
}

auto BigDecimal::to_bigrational() const -> BigRational {
    if (scale_ == 0) {
        return BigRational::from_bigint(unscaled_);
    }
    if (scale_ > 0) {
        // unscaled_ / 10^scale_ (make() reduces; denominator 10^scale_ > 0).
        auto r = BigRational::make(unscaled_, pow10(static_cast<std::uint64_t>(scale_)));
        assert(r.has_value() && "to_bigrational: 10^scale is non-zero");
        return *r;
    }
    // scale_ < 0: value = unscaled_ * 10^(-scale_), an integer.
    return BigRational::from_bigint(unscaled_.multiply(pow10(abs_u64(scale_))));
}

// --- Arithmetic -------------------------------------------------------------

auto BigDecimal::add(const BigDecimal& o) const -> BigDecimal {
    const std::int32_t s = std::max(scale_, o.scale_);
    // Lift each unscaled integer to the common scale s by a non-negative power-of-ten
    // shift (s - scale_ >= 0 and s - o.scale_ >= 0), then add. No gcd, O(limbs).
    const BigInt n1 = unscaled_.multiply(pow10(static_cast<std::uint64_t>(
        static_cast<std::int64_t>(s) - static_cast<std::int64_t>(scale_))));
    const BigInt n2 = o.unscaled_.multiply(pow10(static_cast<std::uint64_t>(
        static_cast<std::int64_t>(s) - static_cast<std::int64_t>(o.scale_))));
    return BigDecimal{n1.add(n2), s};
}

auto BigDecimal::subtract(const BigDecimal& o) const -> BigDecimal {
    return add(o.negate());
}

auto BigDecimal::multiply(const BigDecimal& o) const -> Result<BigDecimal> {
    const std::int64_t s = static_cast<std::int64_t>(scale_) + static_cast<std::int64_t>(o.scale_);
    if (s > std::numeric_limits<std::int32_t>::max() ||
        s < std::numeric_limits<std::int32_t>::min()) {
        return make_error<BigDecimal>(MathError::overflow);
    }
    return BigDecimal{unscaled_.multiply(o.unscaled_), static_cast<std::int32_t>(s)};
}

auto BigDecimal::pow(std::int64_t exp) const -> Result<BigDecimal> {
    if (exp < 0) {
        return make_error<BigDecimal>(MathError::domain_error);
    }
    // scale_ * exp can overflow int64 itself (UB) before a naive range check runs — compute
    // the product with overflow detection, then bound it to int32.
    std::int64_t s = 0;
    if (__builtin_mul_overflow(static_cast<std::int64_t>(scale_), exp, &s) ||
        s > std::numeric_limits<std::int32_t>::max() ||
        s < std::numeric_limits<std::int32_t>::min()) {
        return make_error<BigDecimal>(MathError::overflow);
    }
    return BigDecimal{unscaled_.pow(static_cast<std::uint64_t>(exp)), static_cast<std::int32_t>(s)};
}

// --- Division / rounding ----------------------------------------------------

auto BigDecimal::divide(const BigDecimal& o, std::int32_t result_scale, Rounding mode) const
    -> Result<BigDecimal> {
    if (o.is_zero()) {
        return make_error<BigDecimal>(MathError::division_by_zero);
    }
    // result_unscaled = round( value * 10^result_scale ), and
    // value * 10^result_scale = (u1 * 10^k) / u2  with  k = o.scale_ - scale_ + result_scale.
    const std::int64_t k = static_cast<std::int64_t>(o.scale_) -
                           static_cast<std::int64_t>(scale_) +
                           static_cast<std::int64_t>(result_scale);
    BigInt N = unscaled_;
    BigInt D = o.unscaled_;
    if (k >= 0) {
        N = unscaled_.multiply(pow10(static_cast<std::uint64_t>(k)));
    } else {
        D = o.unscaled_.multiply(pow10(abs_u64(k)));
    }
    return BigDecimal{round_div(N, D, mode), result_scale};
}

auto BigDecimal::divide_exact(const BigDecimal& o) const -> Result<BigDecimal> {
    if (o.is_zero()) {
        return make_error<BigDecimal>(MathError::division_by_zero);
    }
    auto quotient = to_bigrational().divide(o.to_bigrational());
    assert(quotient.has_value() && "divide_exact: divisor already checked non-zero");
    return from_bigrational_exact(*quotient);
}

auto BigDecimal::quantize(std::int32_t new_scale, Rounding mode) const -> BigDecimal {
    if (new_scale >= scale_) {
        // Widen: multiply by 10^(new_scale - scale_) >= 1. Exact; appends trailing zeros.
        const std::uint64_t shift = static_cast<std::uint64_t>(
            static_cast<std::int64_t>(new_scale) - static_cast<std::int64_t>(scale_));
        return BigDecimal{unscaled_.multiply(pow10(shift)), new_scale};
    }
    // Narrow: round unscaled_ / 10^(scale_ - new_scale) under the explicit mode.
    const std::uint64_t drop = static_cast<std::uint64_t>(
        static_cast<std::int64_t>(scale_) - static_cast<std::int64_t>(new_scale));
    return BigDecimal{round_div(unscaled_, pow10(drop), mode), new_scale};
}

// --- Comparison -------------------------------------------------------------

auto BigDecimal::operator<=>(const BigDecimal& o) const -> std::strong_ordering {
    if (scale_ == o.scale_) {
        return unscaled_ <=> o.unscaled_;
    }
    const std::int32_t s = std::max(scale_, o.scale_);
    const BigInt n1 = unscaled_.multiply(pow10(static_cast<std::uint64_t>(
        static_cast<std::int64_t>(s) - static_cast<std::int64_t>(scale_))));
    const BigInt n2 = o.unscaled_.multiply(pow10(static_cast<std::uint64_t>(
        static_cast<std::int64_t>(s) - static_cast<std::int64_t>(o.scale_))));
    return n1 <=> n2;
}

}  // namespace nimblecas
