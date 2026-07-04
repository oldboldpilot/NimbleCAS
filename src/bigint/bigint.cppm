// NimbleCAS arbitrary-precision signed integer (BigInt).
// @author Olumuyiwa Oluwasanmi
//
// The int64 rings elsewhere in the engine (Rational, Polynomial) report overflow as a
// MathError. BigInt removes that ceiling entirely: it is the unbounded exact integer
// that unblocks cryptography (modular exponentiation, primality, RSA) and unbounded CAS
// arithmetic. Because an arbitrary-precision add/subtract/multiply can always allocate
// enough limbs to hold the exact result, those operations CANNOT overflow and so are
// INFALLIBLE — they return a BigInt by value, not a Result. Only division/modulo by
// zero and a malformed from_string can fail, and those (alone) return Result (Rule 32).
//
// Representation: sign-magnitude. `negative_` carries the sign; `mag_` is the magnitude
// as little-endian base-2^32 limbs (mag_[0] is least significant). Intermediate limb
// products/sums use std::uint64_t, so no 128-bit integer type is required (portable).
// INVARIANTS, maintained by normalise() after every mutation:
//   * mag_ has no trailing (most-significant) zero limbs;
//   * zero is the empty magnitude with negative_ == false — there is never a "-0".
// Canonicalisation makes equality a field-wise compare and the ordering a total order.

module;
#include <cassert>

export module nimblecas.bigint;

import std;
import nimblecas.core;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// BigInt — an exact signed integer of unbounded magnitude (sign-magnitude).
// ---------------------------------------------------------------------------
class BigInt {
public:
    BigInt() = default;  // the canonical zero (empty magnitude, non-negative)

    // --- Factories ---------------------------------------------------------
    [[nodiscard]] static auto from_i64(std::int64_t v) -> BigInt;
    [[nodiscard]] static auto from_u64(std::uint64_t v) -> BigInt;
    // Parse an optionally signed decimal integer. Accepts a leading '+'/'-', then one
    // or more decimal digits ("0" and "-0" both denote zero). Empty input or any
    // non-digit character yields MathError::syntax_error.
    [[nodiscard]] static auto from_string(std::string_view s) -> Result<BigInt>;

    // Canonical decimal rendering: "-" only for negative values, "0" for zero, and no
    // superfluous leading zeros.
    [[nodiscard]] auto to_string() const -> std::string;

    // --- Sign / predicates -------------------------------------------------
    [[nodiscard]] auto is_zero() const noexcept -> bool { return mag_.empty(); }
    [[nodiscard]] auto is_negative() const noexcept -> bool { return negative_; }
    [[nodiscard]] auto sign() const noexcept -> int {
        return mag_.empty() ? 0 : (negative_ ? -1 : 1);
    }
    [[nodiscard]] auto abs() const -> BigInt { return from_mag(mag_, false); }
    [[nodiscard]] auto negate() const -> BigInt { return from_mag(mag_, !negative_); }

    // --- Comparison (total order) ------------------------------------------
    // Order by sign first, then by magnitude (length, then most-significant limb down).
    [[nodiscard]] auto operator<=>(const BigInt& o) const noexcept -> std::strong_ordering {
        if (negative_ != o.negative_) {
            return negative_ ? std::strong_ordering::less : std::strong_ordering::greater;
        }
        // Same sign: for negatives the larger magnitude is the smaller value, which is
        // exactly the ordering obtained by comparing the magnitudes in reverse.
        return negative_ ? compare_mag(o.mag_, mag_) : compare_mag(mag_, o.mag_);
    }
    [[nodiscard]] auto operator==(const BigInt& o) const noexcept -> bool {
        return negative_ == o.negative_ && mag_ == o.mag_;
    }

    // --- Arithmetic (INFALLIBLE — arbitrary precision cannot overflow) ------
    [[nodiscard]] auto add(const BigInt& o) const -> BigInt;
    [[nodiscard]] auto subtract(const BigInt& o) const -> BigInt;
    [[nodiscard]] auto multiply(const BigInt& o) const -> BigInt;

    [[nodiscard]] auto operator+(const BigInt& o) const -> BigInt { return add(o); }
    [[nodiscard]] auto operator-(const BigInt& o) const -> BigInt { return subtract(o); }
    [[nodiscard]] auto operator*(const BigInt& o) const -> BigInt { return multiply(o); }

    // --- Division (the correctness-critical operation) ---------------------
    // Truncated division toward zero: quotient truncates and the remainder takes the
    // DIVIDEND's sign, so that a == q*divisor + r with |r| < |divisor| exactly. A zero
    // divisor yields MathError::division_by_zero.
    [[nodiscard]] auto divmod(const BigInt& divisor) const
        -> Result<std::pair<BigInt, BigInt>>;
    [[nodiscard]] auto divide(const BigInt& divisor) const -> Result<BigInt>;
    [[nodiscard]] auto mod(const BigInt& divisor) const -> Result<BigInt>;

    // --- Number-theory primitives (cryptography foundation) ----------------
    // Exponentiation by squaring; pow(0) == 1 (so 0^0 == 1 by convention here).
    [[nodiscard]] auto pow(std::uint64_t exp) const -> BigInt;
    // Non-negative greatest common divisor via the Euclidean algorithm on magnitudes;
    // gcd(0, n) == |n| and gcd(0, 0) == 0.
    [[nodiscard]] static auto gcd(const BigInt& a, const BigInt& b) -> BigInt;
    // Modular exponentiation by squaring: (*this)^exp mod modulus, result in
    // [0, modulus). Requires exp >= 0 and modulus > 0, else MathError::domain_error.
    [[nodiscard]] auto modpow(const BigInt& exp, const BigInt& modulus) const
        -> Result<BigInt>;

private:
    bool negative_{false};              // sign; always false when mag_ is empty (zero)
    std::vector<std::uint32_t> mag_{};  // little-endian base-2^32 limbs, no trailing zeros

    // Strip trailing zero limbs and collapse a zero magnitude to the canonical +0.
    auto normalise() -> void {
        while (!mag_.empty() && mag_.back() == 0) {
            mag_.pop_back();
        }
        if (mag_.empty()) {
            negative_ = false;
        }
    }

    // Assemble a BigInt from a raw magnitude and a sign, then canonicalise it.
    [[nodiscard]] static auto from_mag(std::vector<std::uint32_t> mag, bool negative) -> BigInt {
        BigInt r;
        r.mag_ = std::move(mag);
        r.negative_ = negative;
        r.normalise();
        return r;
    }

    // Compare two normalised magnitudes as unsigned integers.
    [[nodiscard]] static auto compare_mag(const std::vector<std::uint32_t>& a,
                                          const std::vector<std::uint32_t>& b) noexcept
        -> std::strong_ordering;

    // Reduce a value into [0, modulus) where modulus > 0 (used by modpow).
    [[nodiscard]] static auto reduce_mod(const BigInt& x, const BigInt& modulus) -> BigInt;
};

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// Base of the limb representation: each limb is a digit in radix 2^32.
constexpr std::uint64_t limb_base = 1ULL << 32;
constexpr std::uint64_t limb_mask = 0xFFFFFFFFULL;

// Drop trailing zero limbs so a magnitude is in canonical (unnormalised-free) form.
auto trim(std::vector<std::uint32_t>& v) -> void {
    while (!v.empty() && v.back() == 0) {
        v.pop_back();
    }
}

// Unsigned magnitude comparison of two trimmed limb vectors.
[[nodiscard]] auto mag_cmp(const std::vector<std::uint32_t>& a,
                           const std::vector<std::uint32_t>& b) noexcept -> std::strong_ordering {
    if (a.size() != b.size()) {
        return a.size() <=> b.size();
    }
    for (std::size_t i = a.size(); i-- > 0;) {
        if (a[i] != b[i]) {
            return a[i] <=> b[i];
        }
    }
    return std::strong_ordering::equal;
}

// Magnitude sum a + b (little-endian). Result is trimmed by construction.
[[nodiscard]] auto mag_add(const std::vector<std::uint32_t>& a,
                           const std::vector<std::uint32_t>& b) -> std::vector<std::uint32_t> {
    const std::size_t n = std::max(a.size(), b.size());
    std::vector<std::uint32_t> r;
    r.reserve(n + 1);
    std::uint64_t carry = 0;
    for (std::size_t i = 0; i < n; ++i) {
        std::uint64_t sum = carry;
        if (i < a.size()) {
            sum += a[i];
        }
        if (i < b.size()) {
            sum += b[i];
        }
        r.push_back(static_cast<std::uint32_t>(sum & limb_mask));
        carry = sum >> 32;
    }
    if (carry != 0) {
        r.push_back(static_cast<std::uint32_t>(carry));
    }
    return r;
}

// Magnitude difference a - b (little-endian). PRECONDITION: a >= b (unsigned).
[[nodiscard]] auto mag_sub(const std::vector<std::uint32_t>& a,
                           const std::vector<std::uint32_t>& b) -> std::vector<std::uint32_t> {
    assert(mag_cmp(a, b) != std::strong_ordering::less && "mag_sub requires a >= b");
    std::vector<std::uint32_t> r;
    r.reserve(a.size());
    std::int64_t borrow = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        std::int64_t diff = static_cast<std::int64_t>(a[i]) - borrow -
                            (i < b.size() ? static_cast<std::int64_t>(b[i]) : 0);
        if (diff < 0) {
            diff += static_cast<std::int64_t>(limb_base);
            borrow = 1;
        } else {
            borrow = 0;
        }
        r.push_back(static_cast<std::uint32_t>(diff));
    }
    assert(borrow == 0 && "mag_sub underflow (precondition a >= b violated)");
    trim(r);
    return r;
}

// Schoolbook magnitude product a * b with uint64 intermediates. Result is trimmed.
[[nodiscard]] auto mag_mul(const std::vector<std::uint32_t>& a,
                           const std::vector<std::uint32_t>& b) -> std::vector<std::uint32_t> {
    if (a.empty() || b.empty()) {
        return {};
    }
    std::vector<std::uint32_t> r(a.size() + b.size(), 0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        const std::uint64_t ai = a[i];
        std::uint64_t carry = 0;
        for (std::size_t j = 0; j < b.size(); ++j) {
            const std::uint64_t cur =
                static_cast<std::uint64_t>(r[i + j]) + ai * b[j] + carry;
            r[i + j] = static_cast<std::uint32_t>(cur & limb_mask);
            carry = cur >> 32;
        }
        // Index i + b.size() is untouched until this outer iteration, so a plain store
        // of the final carry is correct (later outer iterations accumulate onto it).
        r[i + b.size()] = static_cast<std::uint32_t>(carry);
    }
    trim(r);
    return r;
}

// Multiply a magnitude by a small (< 2^32) factor in place-and-return form.
[[nodiscard]] auto mul_small(std::vector<std::uint32_t> v, std::uint32_t m)
    -> std::vector<std::uint32_t> {
    if (m == 0 || v.empty()) {
        return {};
    }
    std::uint64_t carry = 0;
    for (auto& limb : v) {
        const std::uint64_t cur = static_cast<std::uint64_t>(limb) * m + carry;
        limb = static_cast<std::uint32_t>(cur & limb_mask);
        carry = cur >> 32;
    }
    if (carry != 0) {
        v.push_back(static_cast<std::uint32_t>(carry));
    }
    return v;
}

// Add a small (< 2^32) addend to a magnitude in place-and-return form.
[[nodiscard]] auto add_small(std::vector<std::uint32_t> v, std::uint32_t a)
    -> std::vector<std::uint32_t> {
    std::uint64_t carry = a;
    for (std::size_t i = 0; i < v.size() && carry != 0; ++i) {
        const std::uint64_t cur = static_cast<std::uint64_t>(v[i]) + carry;
        v[i] = static_cast<std::uint32_t>(cur & limb_mask);
        carry = cur >> 32;
    }
    if (carry != 0) {
        v.push_back(static_cast<std::uint32_t>(carry));
    }
    return v;
}

// Single-limb long division: divide magnitude u by the scalar d (d != 0), returning
// (quotient, remainder). The remainder is a single limb (< d).
[[nodiscard]] auto divmod_small(const std::vector<std::uint32_t>& u, std::uint32_t d)
    -> std::pair<std::vector<std::uint32_t>, std::uint32_t> {
    std::vector<std::uint32_t> q(u.size(), 0);
    std::uint64_t rem = 0;
    for (std::size_t i = u.size(); i-- > 0;) {
        const std::uint64_t cur = (rem << 32) | u[i];
        q[i] = static_cast<std::uint32_t>(cur / d);
        rem = cur % d;
    }
    trim(q);
    return {std::move(q), static_cast<std::uint32_t>(rem)};
}

// Knuth TAOCP Vol.2 Algorithm D (normalised long division) for a multi-limb divisor,
// following Warren's "Hacker's Delight" divmnu formulation with 32-bit limbs and 64-bit
// temporaries. PRECONDITIONS: v is trimmed with v.size() >= 2, and u >= v (so the
// quotient has u.size() - v.size() + 1 limbs). Returns (quotient, remainder), trimmed.
[[nodiscard]] auto divmod_knuth(const std::vector<std::uint32_t>& u,
                                const std::vector<std::uint32_t>& v)
    -> std::pair<std::vector<std::uint32_t>, std::vector<std::uint32_t>> {
    const std::size_t n = v.size();          // divisor length (>= 2)
    const std::size_t m = u.size() - n;      // quotient has m + 1 limbs

    // D1. Normalise: left-shift so the divisor's top limb has its high bit set. This
    // shrinks the quotient-digit estimate error to at most 2 (Knuth's Theorem B).
    const int shift = std::countl_zero(v[n - 1]);

    std::vector<std::uint32_t> vn(n);
    if (shift == 0) {
        vn = v;
    } else {
        std::uint32_t carry = 0;
        for (std::size_t i = 0; i < n; ++i) {
            vn[i] = (v[i] << shift) | carry;
            carry = v[i] >> (32 - shift);
        }
        // carry is 0: shift == countl_zero(v[n-1]) shifts out only zero bits at the top.
    }

    std::vector<std::uint32_t> un(u.size() + 1);
    if (shift == 0) {
        for (std::size_t i = 0; i < u.size(); ++i) {
            un[i] = u[i];
        }
        un[u.size()] = 0;
    } else {
        std::uint32_t carry = 0;
        for (std::size_t i = 0; i < u.size(); ++i) {
            un[i] = (u[i] << shift) | carry;
            carry = u[i] >> (32 - shift);
        }
        un[u.size()] = carry;
    }

    std::vector<std::uint32_t> q(m + 1, 0);
    for (std::size_t jj = m + 1; jj-- > 0;) {
        const std::size_t j = jj;
        // D3. Estimate the quotient digit qhat (and running remainder rhat).
        const std::uint64_t num =
            (static_cast<std::uint64_t>(un[j + n]) << 32) | un[j + n - 1];
        std::uint64_t qhat = num / vn[n - 1];
        std::uint64_t rhat = num % vn[n - 1];
        while (qhat >= limb_base ||
               qhat * vn[n - 2] > ((rhat << 32) | un[j + n - 2])) {
            --qhat;
            rhat += vn[n - 1];
            if (rhat >= limb_base) {
                break;  // rhat no longer fits a limb: further correction impossible
            }
        }

        // D4. Multiply and subtract qhat * vn from the current window of un. The signed
        // 64-bit borrow k carries the exact deficit: the identity
        // S = (uint32)S + 2^32 * (S >> 32) holds for any S (arithmetic shift = floor),
        // so k stays correct even across multi-unit borrows.
        std::int64_t k = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint64_t p = qhat * vn[i];
            const std::int64_t sub = static_cast<std::int64_t>(un[j + i]) - k -
                                     static_cast<std::int64_t>(static_cast<std::uint32_t>(p));
            un[j + i] = static_cast<std::uint32_t>(sub);
            k = static_cast<std::int64_t>(p >> 32) - (sub >> 32);
        }
        const std::int64_t sub_top = static_cast<std::int64_t>(un[j + n]) - k;
        un[j + n] = static_cast<std::uint32_t>(sub_top);
        q[j] = static_cast<std::uint32_t>(qhat);

        // D5/D6. If qhat was one too large (the subtraction went negative), decrement
        // the quotient digit and add one divisor back to restore a non-negative window.
        if (sub_top < 0) {
            --q[j];
            std::uint64_t carry = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const std::uint64_t sum =
                    static_cast<std::uint64_t>(un[j + i]) + vn[i] + carry;
                un[j + i] = static_cast<std::uint32_t>(sum & limb_mask);
                carry = sum >> 32;
            }
            un[j + n] = static_cast<std::uint32_t>(static_cast<std::uint64_t>(un[j + n]) + carry);
        }
    }
    trim(q);

    // D8. Denormalise: the remainder occupies the low n limbs of un; shift it back right.
    std::vector<std::uint32_t> r(n);
    if (shift == 0) {
        for (std::size_t i = 0; i < n; ++i) {
            r[i] = un[i];
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            r[i] = (un[i] >> shift) | (un[i + 1] << (32 - shift));
        }
    }
    trim(r);
    return {std::move(q), std::move(r)};
}

// Unsigned magnitude division dispatching to the single-limb fast path, the trivial
// "dividend smaller than divisor" case, or Knuth Algorithm D. PRECONDITION: v != 0.
[[nodiscard]] auto mag_divmod(const std::vector<std::uint32_t>& u,
                              const std::vector<std::uint32_t>& v)
    -> std::pair<std::vector<std::uint32_t>, std::vector<std::uint32_t>> {
    assert(!v.empty() && "mag_divmod requires a non-zero divisor");
    if (mag_cmp(u, v) == std::strong_ordering::less) {
        return {std::vector<std::uint32_t>{}, u};  // quotient 0, remainder = dividend
    }
    if (v.size() == 1) {
        auto [q, rem] = divmod_small(u, v[0]);
        std::vector<std::uint32_t> r;
        if (rem != 0) {
            r.push_back(rem);
        }
        return {std::move(q), std::move(r)};
    }
    return divmod_knuth(u, v);
}

// Build the little-endian limb magnitude of an unsigned 64-bit value (trimmed).
[[nodiscard]] auto limbs_from_u64(std::uint64_t v) -> std::vector<std::uint32_t> {
    std::vector<std::uint32_t> m;
    const auto lo = static_cast<std::uint32_t>(v & limb_mask);
    const auto hi = static_cast<std::uint32_t>(v >> 32);
    if (hi != 0) {
        m = {lo, hi};
    } else if (lo != 0) {
        m = {lo};
    }
    return m;
}

}  // namespace

// --- Factories --------------------------------------------------------------

auto BigInt::from_u64(std::uint64_t v) -> BigInt {
    return from_mag(limbs_from_u64(v), false);
}

auto BigInt::from_i64(std::int64_t v) -> BigInt {
    // Compute the magnitude without ever negating in signed space: negating INT64_MIN
    // is UB, but the unsigned two's-complement round-trip 0 - (uint64)v is well defined.
    const bool neg = v < 0;
    const std::uint64_t mag = neg ? (0ULL - static_cast<std::uint64_t>(v))
                                  : static_cast<std::uint64_t>(v);
    return from_mag(limbs_from_u64(mag), neg);
}

auto BigInt::from_string(std::string_view s) -> Result<BigInt> {
    if (s.empty()) {
        return make_error<BigInt>(MathError::syntax_error);
    }
    std::size_t idx = 0;
    bool neg = false;
    if (s[0] == '+') {
        idx = 1;
    } else if (s[0] == '-') {
        neg = true;
        idx = 1;
    }
    if (idx == s.size()) {
        return make_error<BigInt>(MathError::syntax_error);  // a bare sign is malformed
    }

    // Accumulate the decimal digits into base-2^32 limbs in groups of nine (10^9 < 2^32
    // fits a single limb multiplier), validating each character as we go.
    constexpr std::uint32_t chunk_pow10 = 1000000000u;  // 10^9
    std::vector<std::uint32_t> mag;
    std::uint32_t chunk = 0;
    std::uint32_t chunk_mult = 1;
    int count = 0;
    for (std::size_t p = idx; p < s.size(); ++p) {
        const char c = s[p];
        if (c < '0' || c > '9') {
            return make_error<BigInt>(MathError::syntax_error);
        }
        chunk = chunk * 10 + static_cast<std::uint32_t>(c - '0');
        chunk_mult *= 10;
        if (++count == 9) {
            mag = add_small(mul_small(std::move(mag), chunk_pow10), chunk);
            chunk = 0;
            chunk_mult = 1;
            count = 0;
        }
    }
    if (count > 0) {  // flush the trailing partial group
        mag = add_small(mul_small(std::move(mag), chunk_mult), chunk);
    }
    return from_mag(std::move(mag), neg);
}

auto BigInt::to_string() const -> std::string {
    if (mag_.empty()) {
        return "0";
    }
    // Repeatedly divide the magnitude by 10^9, collecting nine-decimal-digit chunks.
    constexpr std::uint32_t chunk_pow10 = 1000000000u;  // 10^9
    std::vector<std::uint32_t> tmp = mag_;
    std::vector<std::uint32_t> chunks;
    while (!tmp.empty()) {
        auto [q, rem] = divmod_small(tmp, chunk_pow10);
        tmp = std::move(q);
        chunks.push_back(rem);
    }
    std::string out;
    if (negative_) {
        out += '-';
    }
    out += std::to_string(chunks.back());  // most-significant chunk: no zero padding
    for (std::size_t i = chunks.size() - 1; i-- > 0;) {
        out += std::format("{:09}", chunks[i]);
    }
    return out;
}

// --- Comparison -------------------------------------------------------------

auto BigInt::compare_mag(const std::vector<std::uint32_t>& a,
                         const std::vector<std::uint32_t>& b) noexcept -> std::strong_ordering {
    return mag_cmp(a, b);
}

// --- Arithmetic -------------------------------------------------------------

auto BigInt::add(const BigInt& o) const -> BigInt {
    if (negative_ == o.negative_) {
        // Like signs: the magnitudes add and the shared sign is preserved.
        return from_mag(mag_add(mag_, o.mag_), negative_);
    }
    // Unlike signs: subtract the smaller magnitude from the larger; the larger keeps
    // its sign (and equal magnitudes cancel to the canonical zero via normalise()).
    const std::strong_ordering c = mag_cmp(mag_, o.mag_);
    if (c == std::strong_ordering::equal) {
        return BigInt{};
    }
    if (c == std::strong_ordering::greater) {
        return from_mag(mag_sub(mag_, o.mag_), negative_);
    }
    return from_mag(mag_sub(o.mag_, mag_), o.negative_);
}

auto BigInt::subtract(const BigInt& o) const -> BigInt {
    return add(o.negate());
}

auto BigInt::multiply(const BigInt& o) const -> BigInt {
    if (mag_.empty() || o.mag_.empty()) {
        return BigInt{};  // exact zero (sign never "-0")
    }
    return from_mag(mag_mul(mag_, o.mag_), negative_ != o.negative_);
}

// --- Division ---------------------------------------------------------------

auto BigInt::divmod(const BigInt& divisor) const -> Result<std::pair<BigInt, BigInt>> {
    if (divisor.mag_.empty()) {
        return make_error<std::pair<BigInt, BigInt>>(MathError::division_by_zero);
    }
    auto [qmag, rmag] = mag_divmod(mag_, divisor.mag_);
    // Truncated toward zero: the quotient's sign is the XOR of the operand signs; the
    // remainder inherits the DIVIDEND's sign. normalise() drops the sign when zero.
    BigInt q = from_mag(std::move(qmag), negative_ != divisor.negative_);
    BigInt r = from_mag(std::move(rmag), negative_);
    return std::pair<BigInt, BigInt>{std::move(q), std::move(r)};
}

auto BigInt::divide(const BigInt& divisor) const -> Result<BigInt> {
    auto dm = divmod(divisor);
    if (!dm) {
        return make_error<BigInt>(dm.error());
    }
    return std::move(dm->first);
}

auto BigInt::mod(const BigInt& divisor) const -> Result<BigInt> {
    auto dm = divmod(divisor);
    if (!dm) {
        return make_error<BigInt>(dm.error());
    }
    return std::move(dm->second);
}

// --- Number theory ----------------------------------------------------------

auto BigInt::pow(std::uint64_t exp) const -> BigInt {
    BigInt result = from_u64(1);
    BigInt base = *this;
    while (exp > 0) {
        if ((exp & 1U) != 0) {
            result = result.multiply(base);
        }
        exp >>= 1;
        if (exp > 0) {
            base = base.multiply(base);
        }
    }
    return result;
}

auto BigInt::gcd(const BigInt& a, const BigInt& b) -> BigInt {
    // Euclid on magnitudes only, so the result is inherently non-negative.
    std::vector<std::uint32_t> x = a.mag_;
    std::vector<std::uint32_t> y = b.mag_;
    while (!y.empty()) {
        auto qr = mag_divmod(x, y);  // qr.second = x mod y (y != 0 within the loop)
        x = std::move(y);
        y = std::move(qr.second);
    }
    return from_mag(std::move(x), false);
}

auto BigInt::reduce_mod(const BigInt& x, const BigInt& modulus) -> BigInt {
    // modulus is > 0 (guaranteed by the caller). Reduce x into [0, modulus).
    auto qr = mag_divmod(x.mag_, modulus.mag_);
    auto& rmag = qr.second;
    if (rmag.empty()) {
        return BigInt{};  // exact multiple of the modulus
    }
    if (x.negative_) {
        // A negative dividend gives a negative truncated remainder; fold it back into
        // the canonical range by taking modulus - |remainder| (strictly in (0, modulus)).
        return from_mag(mag_sub(modulus.mag_, rmag), false);
    }
    return from_mag(std::move(rmag), false);
}

auto BigInt::modpow(const BigInt& exp, const BigInt& modulus) const -> Result<BigInt> {
    if (exp.negative_) {
        return make_error<BigInt>(MathError::domain_error);  // negative exponents undefined
    }
    if (modulus.sign() <= 0) {
        return make_error<BigInt>(MathError::domain_error);  // modulus must be > 0
    }
    // Left-to-right binary exponentiation, scanning the exponent bits from the most
    // significant down. Every intermediate is kept reduced into [0, modulus).
    BigInt base = reduce_mod(*this, modulus);
    BigInt result = reduce_mod(from_u64(1), modulus);  // 1 mod modulus (0 when modulus == 1)
    if (exp.mag_.empty()) {
        return result;  // exponent 0: result is 1 mod modulus
    }
    const std::vector<std::uint32_t>& em = exp.mag_;
    const int top_bits = 32 - std::countl_zero(em.back());
    for (std::size_t li = em.size(); li-- > 0;) {
        const int start = (li == em.size() - 1) ? top_bits - 1 : 31;
        for (int bit = start; bit >= 0; --bit) {
            result = reduce_mod(result.multiply(result), modulus);
            if (((em[li] >> bit) & 1U) != 0) {
                result = reduce_mod(result.multiply(base), modulus);
            }
        }
    }
    return result;
}

}  // namespace nimblecas
