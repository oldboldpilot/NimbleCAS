// NimbleCAS dense univariate polynomials over the unbounded rationals, Q[x] on BigRational.
// @author Olumuyiwa Oluwasanmi
//
// This is the arbitrary-precision analogue of nimblecas.ratpoly's RationalPoly. Where
// RationalPoly carries int64 Rational coefficients and reports MathError::overflow the
// moment a numerator or denominator saturates, BigRationalPoly carries BigRational
// coefficients and has NO such ceiling — it is the foundation for a bignum-backed
// splitting-field / Jordan path that must exceed the int64 overflow envelope.
//
// Because BigRational's magnitude-combining operations (add, subtract, multiply, negate)
// are themselves INFALLIBLE (arbitrary precision cannot overflow), the polynomial
// operations built purely on them — add, subtract, multiply, scale, negate, derivative —
// are likewise infallible and return a BigRationalPoly by value, not a Result. Only the
// operations that can genuinely fail return Result (Rule 32): divide/monic/gcd fail on a
// zero divisor (MathError::division_by_zero), and the to_ratpoly() bridge fails with
// MathError::overflow when a coefficient no longer fits the int64 Rational tier. The
// whole point of this module is that MathError::overflow is NEVER produced by the
// arithmetic itself. Coefficients are stored low-degree-first and trimmed (back() != 0)
// so the degree is unambiguous.

module;
#include <cassert>

export module nimblecas.bigratpoly;

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.ratpoly;  // for the from_ratpoly / to_ratpoly int64 <-> bignum bridge

export namespace nimblecas {

// Quotient/remainder pair from BigRationalPoly::divide (defined after BigRationalPoly,
// since it holds BigRationalPoly members).
struct BigRatPolyDivMod;

// ---------------------------------------------------------------------------
// BigRationalPoly — dense univariate polynomial over BigRational.
// ---------------------------------------------------------------------------
// coeffs[i] is the coefficient of x^i, stored trimmed (back() != 0, or empty for the
// zero polynomial). Over the exact field Q, divide() always yields an exact quotient and
// a remainder of strictly smaller degree; unlike the int64 tier it can never overflow, so
// the only failure is a zero divisor.
class BigRationalPoly {
public:
    BigRationalPoly() = default;  // the zero polynomial

    // Coefficients are trimmed on construction; a trailing zero BigRational is dropped.
    [[nodiscard]] static auto from_coeffs(std::vector<BigRational> coeffs) -> BigRationalPoly;
    [[nodiscard]] static auto constant(const BigRational& c) -> BigRationalPoly;
    [[nodiscard]] static auto monomial(const BigRational& coeff, std::size_t degree)
        -> BigRationalPoly;
    [[nodiscard]] static auto zero() -> BigRationalPoly { return BigRationalPoly{}; }

    // Lift an int64 RationalPoly into the bignum tier. Each int64 Rational coefficient is
    // already canonical (den > 0, reduced), so the widening is exact and cannot fail.
    [[nodiscard]] static auto from_ratpoly(const RationalPoly& p) -> BigRationalPoly;

    [[nodiscard]] auto is_zero() const noexcept -> bool { return coeffs_.empty(); }
    [[nodiscard]] auto degree() const noexcept -> std::int64_t {
        return coeffs_.empty() ? -1 : static_cast<std::int64_t>(coeffs_.size()) - 1;
    }
    [[nodiscard]] auto coefficient(std::size_t i) const -> BigRational {
        return i < coeffs_.size() ? coeffs_[i] : BigRational{};
    }
    [[nodiscard]] auto leading_coefficient() const -> BigRational {
        return coeffs_.empty() ? BigRational{} : coeffs_.back();
    }
    [[nodiscard]] auto coefficients() const noexcept -> std::span<const BigRational> {
        return coeffs_;
    }

    // --- Arithmetic (INFALLIBLE — arbitrary precision cannot overflow) ------
    [[nodiscard]] auto add(const BigRationalPoly& o) const -> BigRationalPoly;
    [[nodiscard]] auto subtract(const BigRationalPoly& o) const -> BigRationalPoly;
    [[nodiscard]] auto multiply(const BigRationalPoly& o) const -> BigRationalPoly;
    [[nodiscard]] auto scale(const BigRational& s) const -> BigRationalPoly;
    [[nodiscard]] auto negate() const -> BigRationalPoly;
    [[nodiscard]] auto derivative() const -> BigRationalPoly;  // formal d/dx

    // Horner evaluation at x. Returns Result for signature parity with the algebraic tier;
    // over BigRational it only combines magnitudes and therefore always succeeds.
    [[nodiscard]] auto evaluate(const BigRational& x) const -> Result<BigRational>;

    // --- Field operations (fallible only on a zero divisor) -----------------
    // Divide the leading coefficient out, producing a monic polynomial. Zero maps to zero.
    [[nodiscard]] auto monic() const -> Result<BigRationalPoly>;
    // Euclidean division: (quotient, remainder) with
    // *this == quotient * divisor + remainder and deg(remainder) < deg(divisor). Over the
    // field this is always exact; the only failure is a zero divisor.
    [[nodiscard]] auto divide(const BigRationalPoly& divisor) const -> Result<BigRatPolyDivMod>;
    // Monic greatest common divisor via the Euclidean algorithm (gcd(0,0) == 0).
    [[nodiscard]] auto gcd(const BigRationalPoly& o) const -> Result<BigRationalPoly>;

    // Narrow back down to the int64 RationalPoly tier. Fails with MathError::overflow when
    // any coefficient's numerator or denominator no longer fits an int64.
    [[nodiscard]] auto to_ratpoly() const -> Result<RationalPoly>;

    [[nodiscard]] auto is_equal(const BigRationalPoly& o) const -> bool {
        return coeffs_ == o.coeffs_;
    }
    [[nodiscard]] auto to_string(std::string_view var = "x") const -> std::string;

private:
    explicit BigRationalPoly(std::vector<BigRational> coeffs) : coeffs_(std::move(coeffs)) {}
    std::vector<BigRational> coeffs_;  // trimmed: back() != 0, or empty for zero
};

struct BigRatPolyDivMod {
    BigRationalPoly quotient;
    BigRationalPoly remainder;
};

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

[[nodiscard]] auto trim(std::vector<BigRational> c) -> std::vector<BigRational> {
    while (!c.empty() && c.back().is_zero()) {
        c.pop_back();
    }
    return c;
}

// Build a canonical BigRational from an int64 numerator/denominator pair whose denominator
// is already positive (as every int64 Rational's is). make() can only fail on a zero
// denominator, ruled out here, so the assert-guarded dereference never throws — mirroring
// bigrational.cppm's own divide_checked pattern.
[[nodiscard]] auto bigrat_from_i64(std::int64_t num, std::int64_t den) -> BigRational {
    auto r = BigRational::make(BigInt::from_i64(num), BigInt::from_i64(den));
    assert(r.has_value() && "canonical int64 Rational always has a non-zero denominator");
    return *r;
}

// Exact BigInt -> int64: render to decimal and re-parse. std::from_chars reports an
// out-of-range magnitude, which we surface as "does not fit the int64 tier".
[[nodiscard]] auto bigint_to_i64(const BigInt& x) -> std::optional<std::int64_t> {
    const std::string s = x.to_string();
    std::int64_t out = 0;
    const auto* end = s.data() + s.size();
    const auto res = std::from_chars(s.data(), end, out);
    if (res.ec != std::errc{} || res.ptr != end) {
        return std::nullopt;
    }
    return out;
}

}  // namespace

auto BigRationalPoly::from_coeffs(std::vector<BigRational> coeffs) -> BigRationalPoly {
    return BigRationalPoly{trim(std::move(coeffs))};
}

auto BigRationalPoly::constant(const BigRational& c) -> BigRationalPoly {
    if (c.is_zero()) {
        return BigRationalPoly{};
    }
    return BigRationalPoly{std::vector<BigRational>{c}};
}

auto BigRationalPoly::monomial(const BigRational& coeff, std::size_t degree) -> BigRationalPoly {
    if (coeff.is_zero()) {
        return BigRationalPoly{};
    }
    assert(degree < std::numeric_limits<std::size_t>::max() && "monomial degree too large");
    std::vector<BigRational> c(degree + 1);  // BigRational{} default is the canonical zero
    c[degree] = coeff;
    return BigRationalPoly{std::move(c)};
}

auto BigRationalPoly::from_ratpoly(const RationalPoly& p) -> BigRationalPoly {
    std::vector<BigRational> c;
    c.reserve(p.coefficients().size());
    for (const Rational& q : p.coefficients()) {
        c.push_back(bigrat_from_i64(q.numerator(), q.denominator()));
    }
    return BigRationalPoly{std::move(c)};  // already trimmed (p is trimmed)
}

auto BigRationalPoly::add(const BigRationalPoly& o) const -> BigRationalPoly {
    std::vector<BigRational> r(std::max(coeffs_.size(), o.coeffs_.size()));
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        r[i] = coeffs_[i];
    }
    for (std::size_t i = 0; i < o.coeffs_.size(); ++i) {
        r[i] = r[i].add(o.coeffs_[i]);
    }
    return from_coeffs(std::move(r));
}

auto BigRationalPoly::subtract(const BigRationalPoly& o) const -> BigRationalPoly {
    std::vector<BigRational> r(std::max(coeffs_.size(), o.coeffs_.size()));
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        r[i] = coeffs_[i];
    }
    for (std::size_t i = 0; i < o.coeffs_.size(); ++i) {
        r[i] = r[i].subtract(o.coeffs_[i]);
    }
    return from_coeffs(std::move(r));
}

auto BigRationalPoly::scale(const BigRational& s) const -> BigRationalPoly {
    if (s.is_zero()) {
        return BigRationalPoly{};
    }
    std::vector<BigRational> r(coeffs_.size());
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        r[i] = coeffs_[i].multiply(s);
    }
    return from_coeffs(std::move(r));  // s != 0 introduces no zeros, but trim is cheap
}

auto BigRationalPoly::multiply(const BigRationalPoly& o) const -> BigRationalPoly {
    if (is_zero() || o.is_zero()) {
        return BigRationalPoly{};
    }
    // Both operands are non-empty here; the result width cannot realistically wrap
    // std::size_t (it would demand >2^63 coefficients), so assert rather than fabricate a
    // MathError this infallible operation has no channel to return.
    assert(coeffs_.size() <= std::numeric_limits<std::size_t>::max() - o.coeffs_.size() &&
           "polynomial product width overflow");
    std::vector<BigRational> r(coeffs_.size() + o.coeffs_.size() - 1);
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        for (std::size_t j = 0; j < o.coeffs_.size(); ++j) {
            r[i + j] = r[i + j].add(coeffs_[i].multiply(o.coeffs_[j]));
        }
    }
    return from_coeffs(std::move(r));
}

auto BigRationalPoly::negate() const -> BigRationalPoly {
    std::vector<BigRational> r(coeffs_.size());
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        r[i] = coeffs_[i].negate();
    }
    return BigRationalPoly{std::move(r)};  // negation preserves the trimmed invariant
}

auto BigRationalPoly::derivative() const -> BigRationalPoly {
    if (coeffs_.size() <= 1) {
        return BigRationalPoly{};
    }
    // The exponent i is cast to int64 below; guard the (physically unreachable) case of a
    // degree beyond INT64_MAX so the cast can never wrap.
    assert(coeffs_.size() <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) &&
           "derivative degree beyond INT64_MAX");
    std::vector<BigRational> r(coeffs_.size() - 1);
    for (std::size_t i = 1; i < coeffs_.size(); ++i) {
        r[i - 1] = coeffs_[i].multiply(BigRational::from_int(static_cast<std::int64_t>(i)));
    }
    return from_coeffs(std::move(r));
}

auto BigRationalPoly::evaluate(const BigRational& x) const -> Result<BigRational> {
    // Horner from the highest-degree coefficient down: acc <- acc*x + c[i].
    BigRational acc;  // canonical zero
    for (std::size_t i = coeffs_.size(); i-- > 0;) {
        acc = acc.multiply(x).add(coeffs_[i]);
    }
    return acc;  // always succeeds; Result is for signature parity with the algebraic tier
}

auto BigRationalPoly::monic() const -> Result<BigRationalPoly> {
    if (is_zero()) {
        return BigRationalPoly{};
    }
    const BigRational lc = leading_coefficient();
    if (lc == BigRational::from_int(1)) {
        return *this;
    }
    auto inv = lc.reciprocal();  // lc != 0 (non-zero polynomial), so this never fails
    if (!inv) {
        return make_error<BigRationalPoly>(inv.error());
    }
    return scale(*inv);
}

auto BigRationalPoly::divide(const BigRationalPoly& divisor) const -> Result<BigRatPolyDivMod> {
    if (divisor.is_zero()) {
        return make_error<BigRatPolyDivMod>(MathError::division_by_zero);
    }
    if (degree() < divisor.degree()) {
        return BigRatPolyDivMod{.quotient = BigRationalPoly{}, .remainder = *this};
    }
    const BigRational lc_b = divisor.leading_coefficient();
    const std::int64_t nb = divisor.degree();
    std::vector<BigRational> quotient(static_cast<std::size_t>(degree() - nb) + 1);
    BigRationalPoly r = *this;
    while (!r.is_zero() && r.degree() >= nb) {
        // t = (lc(r) / lc(divisor)) * x^(deg r - nb); r <- r - t * divisor.
        auto q = r.leading_coefficient().divide(lc_b);  // lc_b != 0, so this never fails
        if (!q) {
            return make_error<BigRatPolyDivMod>(q.error());
        }
        const std::size_t k = static_cast<std::size_t>(r.degree() - nb);
        quotient[k] = *q;
        // Infallible: monomial * divisor and the subtraction only combine magnitudes; the
        // leading term cancels, so the degree strictly drops and the loop terminates.
        r = r.subtract(BigRationalPoly::monomial(*q, k).multiply(divisor));
    }
    return BigRatPolyDivMod{.quotient = from_coeffs(std::move(quotient)), .remainder = r};
}

auto BigRationalPoly::gcd(const BigRationalPoly& o) const -> Result<BigRationalPoly> {
    BigRationalPoly a = *this;
    BigRationalPoly b = o;
    while (!b.is_zero()) {
        auto dm = a.divide(b);
        if (!dm) {
            return make_error<BigRationalPoly>(dm.error());
        }
        a = b;
        b = std::move(dm->remainder);
    }
    return a.monic();  // canonical (monic) representative, or 0 when both inputs were 0
}

auto BigRationalPoly::to_ratpoly() const -> Result<RationalPoly> {
    std::vector<Rational> c;
    c.reserve(coeffs_.size());
    for (const BigRational& q : coeffs_) {
        const auto num = bigint_to_i64(q.numerator());
        const auto den = bigint_to_i64(q.denominator());
        if (!num || !den) {
            return make_error<RationalPoly>(MathError::overflow);  // exceeds the int64 tier
        }
        // q is canonical (den > 0, reduced), so make() re-derives the same reduced form.
        auto r = Rational::make(*num, *den);
        if (!r) {
            return make_error<RationalPoly>(r.error());
        }
        c.push_back(*r);
    }
    return RationalPoly::from_coeffs(std::move(c));
}

auto BigRationalPoly::to_string(std::string_view var) const -> std::string {
    if (coeffs_.empty()) {
        return "0";
    }
    std::string out;
    bool first = true;
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        if (coeffs_[i].is_zero()) {
            continue;
        }
        if (!first) {
            out += " + ";
        }
        first = false;
        const std::string c = coeffs_[i].to_string();
        if (i == 0) {
            out += c;
        } else if (i == 1) {
            out += std::format("({})*{}", c, var);
        } else {
            out += std::format("({})*{}^{}", c, var, i);
        }
    }
    return out;
}

}  // namespace nimblecas
