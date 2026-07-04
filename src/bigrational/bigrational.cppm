// NimbleCAS exact unbounded rational number (BigRational, on arbitrary-precision BigInt).
// @author Olumuyiwa Oluwasanmi
//
// The arithmetic tower climbs from the int64 Rational (nimblecas.ratpoly) through the
// 128-bit Rational128 to this, its unbounded exact tier. The int64/int128 rationals
// report a saturated numerator or denominator as MathError::overflow; BigRational removes
// that ceiling entirely by carrying num_ and den_ as arbitrary-precision BigInt. Because
// a BigInt add/subtract/multiply can always allocate enough limbs to hold the exact
// result, the field operations that only combine magnitudes — add, subtract, multiply,
// negate — CANNOT overflow and are therefore INFALLIBLE (they return a BigRational by
// value, not a Result). Only make/divide/reciprocal/pow (which can divide by zero) and
// from_string (which can be malformed) return Result (Rule 32).
//
// HONESTY: BigRational is exact and unbounded over Q — no rounding, no overflow. The
// cost is heap-allocating BigInt arithmetic on every operation, so it is the slow-but-
// exact tier: values that comfortably fit int64 should use the int64 Rational, and
// to_double() below is an APPROXIMATION (the only lossy operation on this type).
//
// Representation: two BigInt members num_ and den_. INVARIANTS, restored by normalise()
// after every operation, make equality a field-wise compare and comparison a total order:
//   * den_ > 0 — never zero, never negative; the value's entire sign lives in num_;
//   * gcd(|num_|, den_) == 1 — always fully reduced (lowest terms);
//   * the canonical zero is 0/1.

module;
#include <cassert>

export module nimblecas.bigrational;

import std;
import nimblecas.core;
import nimblecas.bigint;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// BigRational — an exact fraction num_/den_ in lowest terms with den_ > 0.
// ---------------------------------------------------------------------------
class BigRational {
public:
    BigRational() = default;  // the canonical zero, 0/1

    // --- Factories ---------------------------------------------------------
    // Lift an integer n into Q as n/1 (already reduced, denominator 1).
    [[nodiscard]] static auto from_bigint(const BigInt& n) -> BigRational;
    [[nodiscard]] static auto from_int(std::int64_t v) -> BigRational;
    // Construct num/den in canonical form. A zero denominator yields
    // MathError::division_by_zero; otherwise the fraction is reduced and its sign
    // normalised onto the numerator.
    [[nodiscard]] static auto make(BigInt num, BigInt den) -> Result<BigRational>;
    // Parse "num/den" or a bare integer "num". Each side follows BigInt::from_string
    // (optional leading sign, then decimal digits). A malformed side yields
    // MathError::syntax_error; a denominator of zero yields MathError::division_by_zero.
    [[nodiscard]] static auto from_string(std::string_view s) -> Result<BigRational>;

    // --- Accessors ---------------------------------------------------------
    [[nodiscard]] auto numerator() const noexcept -> const BigInt& { return num_; }
    [[nodiscard]] auto denominator() const noexcept -> const BigInt& { return den_; }
    [[nodiscard]] auto is_zero() const noexcept -> bool { return num_.is_zero(); }
    [[nodiscard]] auto is_integer() const -> bool { return den_ == BigInt::from_u64(1); }
    // Sign of the value: -1, 0, or +1. Since den_ > 0, it is exactly the sign of num_.
    [[nodiscard]] auto sign() const noexcept -> int { return num_.sign(); }
    // "num/den", or just "num" when the value is integral (den_ == 1).
    [[nodiscard]] auto to_string() const -> std::string;
    // APPROXIMATE conversion to double (the only lossy operation on this type): the
    // nearest double to num_/den_ for in-range magnitudes, +/-inf when the integer part
    // exceeds the double range. Never use it where exactness matters.
    [[nodiscard]] auto to_double() const -> double;

    // --- Arithmetic (INFALLIBLE — arbitrary precision cannot overflow) ------
    [[nodiscard]] auto add(const BigRational& o) const -> BigRational;
    [[nodiscard]] auto subtract(const BigRational& o) const -> BigRational;
    [[nodiscard]] auto multiply(const BigRational& o) const -> BigRational;
    [[nodiscard]] auto negate() const -> BigRational;

    // --- Division / powers (fallible: only on division by zero) -------------
    [[nodiscard]] auto divide(const BigRational& o) const -> Result<BigRational>;
    [[nodiscard]] auto reciprocal() const -> Result<BigRational>;
    // (num/den)^exp. exp == 0 gives 1 (so 0^0 == 1, matching BigInt::pow). A negative
    // exponent inverts, so a negative power of zero yields MathError::division_by_zero.
    [[nodiscard]] auto pow(std::int64_t exp) const -> Result<BigRational>;

    // --- Comparison (total order) ------------------------------------------
    // Compare a/b against c/d by cross-multiplying with the POSITIVE denominators: with
    // b, d > 0 the sign of a*d - c*b is the sign of the true difference, so no direction
    // flip is ever needed and the ordering is total.
    [[nodiscard]] auto operator<=>(const BigRational& o) const -> std::strong_ordering {
        return num_.multiply(o.den_) <=> o.num_.multiply(den_);
    }
    // Canonical form (reduced, den_ > 0) makes equality a plain field-wise compare.
    [[nodiscard]] auto operator==(const BigRational& o) const -> bool {
        return num_ == o.num_ && den_ == o.den_;
    }

private:
    // Raw constructor: assumes num/den already satisfy the class invariants.
    BigRational(BigInt num, BigInt den) : num_(std::move(num)), den_(std::move(den)) {}

    // Reduce num/den to canonical form (den != 0 required): move any sign onto the
    // numerator, then divide both through by their gcd.
    [[nodiscard]] static auto normalise(BigInt num, BigInt den) -> BigRational;

    BigInt num_{};                      // 0 by default; carries the value's sign
    BigInt den_ = BigInt::from_u64(1);  // 1 by default; invariant: den_ > 0
};

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// Integer quotient a / b for a divisor known to be non-zero (asserts otherwise). Used
// for the gcd reduction in normalise() (where the division is exact) and for the scaled
// long division inside to_double() (where it truncates). BigInt::divide only fails on a
// zero divisor, which the callers rule out.
[[nodiscard]] auto divide_checked(const BigInt& a, const BigInt& b) -> BigInt {
    auto q = a.divide(b);
    assert(q.has_value() && "divide_checked requires a non-zero divisor");
    return *q;
}

// Best-effort BigInt -> double via its exact decimal rendering: std::from_chars rounds to
// the nearest double and yields +/-inf for magnitudes beyond the double range.
[[nodiscard]] auto bigint_to_double(const BigInt& x) -> double {
    const std::string s = x.to_string();
    double out = 0.0;
    std::from_chars(s.data(), s.data() + s.size(), out);
    return out;
}

}  // namespace

// --- Factories --------------------------------------------------------------

auto BigRational::from_bigint(const BigInt& n) -> BigRational {
    // n/1 is already canonical: gcd(|n|, 1) == 1 and the denominator is positive.
    return BigRational{n, BigInt::from_u64(1)};
}

auto BigRational::from_int(std::int64_t v) -> BigRational {
    return from_bigint(BigInt::from_i64(v));
}

auto BigRational::make(BigInt num, BigInt den) -> Result<BigRational> {
    if (den.is_zero()) {
        return make_error<BigRational>(MathError::division_by_zero);
    }
    return normalise(std::move(num), std::move(den));
}

auto BigRational::from_string(std::string_view s) -> Result<BigRational> {
    const auto slash = s.find('/');
    if (slash == std::string_view::npos) {
        auto n = BigInt::from_string(s);  // bare integer
        if (!n) {
            return make_error<BigRational>(n.error());  // syntax_error
        }
        return from_bigint(*n);
    }
    // "num/den": both sides must parse; a trailing/leading empty side or a second '/'
    // (which would land in the denominator substring) fails BigInt::from_string.
    auto n = BigInt::from_string(s.substr(0, slash));
    auto d = BigInt::from_string(s.substr(slash + 1));
    if (!n || !d) {
        return make_error<BigRational>(MathError::syntax_error);
    }
    return make(std::move(*n), std::move(*d));  // a "0" denominator -> division_by_zero
}

auto BigRational::normalise(BigInt num, BigInt den) -> BigRational {
    // Caller guarantees den != 0. Move the sign onto the numerator so den_ > 0.
    if (den.is_negative()) {
        num = num.negate();
        den = den.negate();
    }
    // BigInt::gcd is sign-independent and non-negative; den != 0 makes it >= 1, so both
    // divisions below are exact and cannot divide by zero.
    const BigInt g = BigInt::gcd(num, den);
    return BigRational{divide_checked(num, g), divide_checked(den, g)};
}

// --- Accessors --------------------------------------------------------------

auto BigRational::to_string() const -> std::string {
    if (is_integer()) {
        return num_.to_string();
    }
    return std::format("{}/{}", num_.to_string(), den_.to_string());
}

auto BigRational::to_double() const -> double {
    if (num_.is_zero()) {
        return 0.0;
    }
    // Split |value| into its integer part q and a fractional part rem/den_, then read the
    // fraction back through an 18-decimal-digit scaling. q converts directly (possibly to
    // +/-inf); the scaled fraction stays below 10^18 and so is always finite.
    const BigInt a = num_.abs();
    auto dm = a.divmod(den_);  // den_ > 0, so this never fails
    assert(dm.has_value() && "to_double: denominator must be non-zero");
    const BigInt& q = dm->first;
    const BigInt& rem = dm->second;
    constexpr std::uint64_t scale = 1000000000000000000ULL;  // 10^18
    const BigInt frac_int = divide_checked(rem.multiply(BigInt::from_u64(scale)), den_);
    const double mag =
        bigint_to_double(q) + bigint_to_double(frac_int) / static_cast<double>(scale);
    return num_.is_negative() ? -mag : mag;
}

// --- Arithmetic -------------------------------------------------------------

auto BigRational::add(const BigRational& o) const -> BigRational {
    // a/b + c/d = (a*d + c*b) / (b*d); b, d > 0 so the new denominator is positive.
    BigInt n = num_.multiply(o.den_).add(o.num_.multiply(den_));
    BigInt d = den_.multiply(o.den_);
    return normalise(std::move(n), std::move(d));
}

auto BigRational::subtract(const BigRational& o) const -> BigRational {
    BigInt n = num_.multiply(o.den_).subtract(o.num_.multiply(den_));
    BigInt d = den_.multiply(o.den_);
    return normalise(std::move(n), std::move(d));
}

auto BigRational::multiply(const BigRational& o) const -> BigRational {
    return normalise(num_.multiply(o.num_), den_.multiply(o.den_));
}

auto BigRational::negate() const -> BigRational {
    // Negating the numerator leaves den_ > 0 and the gcd unchanged: already canonical.
    return BigRational{num_.negate(), den_};
}

// --- Division / powers ------------------------------------------------------

auto BigRational::divide(const BigRational& o) const -> Result<BigRational> {
    if (o.is_zero()) {
        return make_error<BigRational>(MathError::division_by_zero);
    }
    // (a/b) / (c/d) = (a*d) / (b*c); b*c != 0 here, and normalise() fixes the sign when
    // c is negative.
    BigInt n = num_.multiply(o.den_);
    BigInt d = den_.multiply(o.num_);
    return normalise(std::move(n), std::move(d));
}

auto BigRational::reciprocal() const -> Result<BigRational> {
    if (is_zero()) {
        return make_error<BigRational>(MathError::division_by_zero);
    }
    // Swap numerator and denominator; normalise() moves the sign back onto the numerator
    // when num_ was negative.
    return normalise(den_, num_);
}

auto BigRational::pow(std::int64_t exp) const -> Result<BigRational> {
    if (exp == 0) {
        return from_int(1);  // x^0 == 1 for every x (0^0 == 1, as in BigInt::pow)
    }
    const bool negative_exp = exp < 0;
    // Take |exp| without ever negating in signed space (negating INT64_MIN is UB).
    const std::uint64_t e = negative_exp ? (0ULL - static_cast<std::uint64_t>(exp))
                                         : static_cast<std::uint64_t>(exp);
    if (negative_exp && is_zero()) {
        return make_error<BigRational>(MathError::division_by_zero);
    }
    BigInt np = num_.pow(e);  // BigInt::pow carries the numerator's sign for odd e
    BigInt dp = den_.pow(e);  // den_ > 0, so dp > 0
    if (negative_exp) {
        // (num/den)^-e = den^e / num^e; num^e may be negative, so normalise() fixes sign.
        return normalise(std::move(dp), std::move(np));
    }
    // Positive exponent: coprimality of num_/den_ lifts to their powers and dp > 0, so
    // the result is already canonical.
    return BigRational{std::move(np), std::move(dp)};
}

}  // namespace nimblecas
