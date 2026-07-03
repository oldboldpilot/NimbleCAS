// NimbleCAS dense univariate polynomials over the rationals, Q[x] (ROADMAP 7.17/7.20).
// @author Olumuyiwa Oluwasanmi
//
// The integer ring Z[x] (nimblecas.polynomial) is not a field: a general division of
// one polynomial by another does not stay in Z[x]. Partial-fraction decomposition, true
// division-with-remainder, and a Euclidean GCD that produces a *monic* result all need
// the coefficient field Q. This module supplies that: Rational is an exact reduced
// int64 fraction, and RationalPoly is a dense polynomial over it.
//
// Following the rest of the engine, arithmetic is exact and overflow-checked: when an
// int64 numerator or denominator would overflow, the operation returns
// MathError::overflow rather than silently wrapping (Rule 32). Coefficients are stored
// trimmed (no trailing zeros) so the degree is unambiguous.

module;
#include <cassert>

export module nimblecas.ratpoly;

import std;
import nimblecas.core;
import nimblecas.polynomial;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// Rational — an exact fraction num/den in lowest terms with den > 0.
// ---------------------------------------------------------------------------
// The canonical form (den > 0, gcd(|num|, den) == 1, zero represented as 0/1) makes
// equality a field-wise compare and keeps intermediate magnitudes as small as int64
// allows. Every arithmetic operation is overflow-checked and returns Result.
class Rational {
public:
    Rational() = default;  // 0/1

    // Construct num/den in canonical form. Fails with division_by_zero (den == 0) or
    // overflow (an int64 boundary such as INT64_MIN that cannot be sign-normalised).
    [[nodiscard]] static auto make(std::int64_t num, std::int64_t den) -> Result<Rational>;
    [[nodiscard]] static auto from_int(std::int64_t v) -> Rational;

    [[nodiscard]] auto numerator() const noexcept -> std::int64_t { return num_; }
    [[nodiscard]] auto denominator() const noexcept -> std::int64_t { return den_; }
    [[nodiscard]] auto is_zero() const noexcept -> bool { return num_ == 0; }
    [[nodiscard]] auto is_integer() const noexcept -> bool { return den_ == 1; }

    [[nodiscard]] auto add(const Rational& o) const -> Result<Rational>;
    [[nodiscard]] auto subtract(const Rational& o) const -> Result<Rational>;
    [[nodiscard]] auto multiply(const Rational& o) const -> Result<Rational>;
    // Fails with division_by_zero when o == 0.
    [[nodiscard]] auto divide(const Rational& o) const -> Result<Rational>;
    [[nodiscard]] auto negate() const -> Result<Rational>;  // fails only on INT64_MIN

    [[nodiscard]] auto operator==(const Rational& o) const noexcept -> bool {
        return num_ == o.num_ && den_ == o.den_;
    }
    [[nodiscard]] auto to_string() const -> std::string;

private:
    Rational(std::int64_t num, std::int64_t den) : num_(num), den_(den) {}
    std::int64_t num_{0};
    std::int64_t den_{1};
};

// Quotient/remainder pair from RationalPoly::divide (defined after RationalPoly,
// since it holds RationalPoly members).
struct RationalDivMod;

// ---------------------------------------------------------------------------
// RationalPoly — dense univariate polynomial over Rational.
// ---------------------------------------------------------------------------
// coeffs[i] is the coefficient of x^i, stored trimmed (back() != 0). The field
// structure means divide() always yields an exact quotient and a remainder of
// strictly smaller degree — the operation only fails on overflow or a zero divisor.
class RationalPoly {
public:
    RationalPoly() = default;  // the zero polynomial
    // Coefficients are trimmed on construction; a zero Rational tail is dropped.
    [[nodiscard]] static auto from_coeffs(std::vector<Rational> coeffs) -> RationalPoly;
    // Lift an integer polynomial into Q[x] (each coefficient over denominator 1).
    [[nodiscard]] static auto from_polynomial(const Polynomial& p) -> RationalPoly;
    [[nodiscard]] static auto constant(const Rational& c) -> RationalPoly;
    [[nodiscard]] static auto monomial(const Rational& coeff, std::size_t degree) -> RationalPoly;

    [[nodiscard]] auto is_zero() const noexcept -> bool { return coeffs_.empty(); }
    [[nodiscard]] auto degree() const noexcept -> std::int64_t {
        return coeffs_.empty() ? -1 : static_cast<std::int64_t>(coeffs_.size()) - 1;
    }
    [[nodiscard]] auto coefficient(std::size_t i) const -> Rational {
        return i < coeffs_.size() ? coeffs_[i] : Rational{};
    }
    [[nodiscard]] auto leading_coefficient() const -> Rational {
        return coeffs_.empty() ? Rational{} : coeffs_.back();
    }
    [[nodiscard]] auto coefficients() const noexcept -> std::span<const Rational> {
        return coeffs_;
    }

    [[nodiscard]] auto add(const RationalPoly& o) const -> Result<RationalPoly>;
    [[nodiscard]] auto subtract(const RationalPoly& o) const -> Result<RationalPoly>;
    [[nodiscard]] auto scale(const Rational& s) const -> Result<RationalPoly>;
    [[nodiscard]] auto multiply(const RationalPoly& o) const -> Result<RationalPoly>;

    // Divide the leading coefficient out, producing a monic polynomial (leading
    // coefficient 1). The zero polynomial maps to itself.
    [[nodiscard]] auto monic() const -> Result<RationalPoly>;

    // Euclidean division: returns (quotient, remainder) with
    // *this == quotient * divisor + remainder and deg(remainder) < deg(divisor).
    // Over a field this is always exact; fails only on a zero divisor or overflow.
    [[nodiscard]] auto divide(const RationalPoly& divisor) const -> Result<RationalDivMod>;

    // Monic greatest common divisor via the Euclidean algorithm (gcd(0,0) == 0).
    [[nodiscard]] auto gcd(const RationalPoly& o) const -> Result<RationalPoly>;

    // Formal derivative d/dx.
    [[nodiscard]] auto derivative() const -> Result<RationalPoly>;

    // Convert back to an integer polynomial when every coefficient is integral;
    // fails with domain_error otherwise.
    [[nodiscard]] auto to_polynomial() const -> Result<Polynomial>;

    [[nodiscard]] auto is_equal(const RationalPoly& o) const noexcept -> bool {
        return coeffs_ == o.coeffs_;
    }
    [[nodiscard]] auto to_string(std::string_view var = "x") const -> std::string;

private:
    explicit RationalPoly(std::vector<Rational> coeffs) : coeffs_(std::move(coeffs)) {}
    std::vector<Rational> coeffs_;  // trimmed: back() != 0, or empty for zero
};

struct RationalDivMod {
    RationalPoly quotient;
    RationalPoly remainder;
};

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

[[nodiscard]] auto add_ov(std::int64_t a, std::int64_t b, std::int64_t& out) -> bool {
    return __builtin_add_overflow(a, b, &out);
}
[[nodiscard]] auto mul_ov(std::int64_t a, std::int64_t b, std::int64_t& out) -> bool {
    return __builtin_mul_overflow(a, b, &out);
}

constexpr std::int64_t int64_min = std::numeric_limits<std::int64_t>::min();

// std::gcd is UB when an argument is INT64_MIN (its magnitude is unrepresentable).
[[nodiscard]] auto checked_gcd(std::int64_t a, std::int64_t b) -> std::optional<std::int64_t> {
    if (a == int64_min || b == int64_min) {
        return std::nullopt;
    }
    return std::gcd(a, b);
}

}  // namespace

// --- Rational ---------------------------------------------------------------

auto Rational::make(std::int64_t num, std::int64_t den) -> Result<Rational> {
    if (den == 0) {
        return make_error<Rational>(MathError::division_by_zero);
    }
    // Normalise the sign onto the numerator so the denominator is positive. Negating
    // INT64_MIN overflows, so reject any INT64_MIN operand up front.
    if (num == int64_min || den == int64_min) {
        return make_error<Rational>(MathError::overflow);
    }
    if (den < 0) {
        num = -num;
        den = -den;
    }
    const auto g = checked_gcd(num, den);  // both magnitudes now representable
    if (!g) {
        return make_error<Rational>(MathError::overflow);
    }
    const std::int64_t d = *g;  // gcd(num, den) >= 1 here (den > 0)
    return Rational{num / d, den / d};
}

auto Rational::from_int(std::int64_t v) -> Rational {
    return Rational{v, 1};
}

auto Rational::add(const Rational& o) const -> Result<Rational> {
    // num_/den_ + o.num_/o.den_ = (num_*o.den_ + o.num_*den_) / (den_*o.den_).
    std::int64_t t1 = 0;
    std::int64_t t2 = 0;
    std::int64_t den = 0;
    std::int64_t num = 0;
    if (mul_ov(num_, o.den_, t1) || mul_ov(o.num_, den_, t2) || add_ov(t1, t2, num) ||
        mul_ov(den_, o.den_, den)) {
        return make_error<Rational>(MathError::overflow);
    }
    return make(num, den);
}

auto Rational::subtract(const Rational& o) const -> Result<Rational> {
    auto neg = o.negate();
    if (!neg) {
        return neg;
    }
    return add(*neg);
}

auto Rational::multiply(const Rational& o) const -> Result<Rational> {
    std::int64_t num = 0;
    std::int64_t den = 0;
    if (mul_ov(num_, o.num_, num) || mul_ov(den_, o.den_, den)) {
        return make_error<Rational>(MathError::overflow);
    }
    return make(num, den);
}

auto Rational::divide(const Rational& o) const -> Result<Rational> {
    if (o.num_ == 0) {
        return make_error<Rational>(MathError::division_by_zero);
    }
    std::int64_t num = 0;
    std::int64_t den = 0;
    if (mul_ov(num_, o.den_, num) || mul_ov(den_, o.num_, den)) {
        return make_error<Rational>(MathError::overflow);
    }
    return make(num, den);  // make() re-normalises the sign if o.num_ < 0
}

auto Rational::negate() const -> Result<Rational> {
    if (num_ == int64_min) {
        return make_error<Rational>(MathError::overflow);
    }
    return Rational{-num_, den_};  // already canonical (den_ > 0, gcd unchanged)
}

auto Rational::to_string() const -> std::string {
    return den_ == 1 ? std::format("{}", num_) : std::format("{}/{}", num_, den_);
}

// --- RationalPoly ------------------------------------------------------------

namespace {

[[nodiscard]] auto trim(std::vector<Rational> c) -> std::vector<Rational> {
    while (!c.empty() && c.back().is_zero()) {
        c.pop_back();
    }
    return c;
}

}  // namespace

auto RationalPoly::from_coeffs(std::vector<Rational> coeffs) -> RationalPoly {
    return RationalPoly{trim(std::move(coeffs))};
}

auto RationalPoly::from_polynomial(const Polynomial& p) -> RationalPoly {
    std::vector<Rational> c;
    c.reserve(p.coefficients().size());
    for (const std::int64_t v : p.coefficients()) {
        c.push_back(Rational::from_int(v));
    }
    return RationalPoly{std::move(c)};  // already trimmed (p is trimmed)
}

auto RationalPoly::constant(const Rational& c) -> RationalPoly {
    if (c.is_zero()) {
        return RationalPoly{};
    }
    return RationalPoly{std::vector<Rational>{c}};
}

auto RationalPoly::monomial(const Rational& coeff, std::size_t degree) -> RationalPoly {
    if (coeff.is_zero()) {
        return RationalPoly{};
    }
    assert(degree < std::numeric_limits<std::size_t>::max() && "monomial degree too large");
    std::vector<Rational> c(degree + 1);  // Rational{} default is 0/1
    c[degree] = coeff;
    return RationalPoly{std::move(c)};
}

auto RationalPoly::add(const RationalPoly& o) const -> Result<RationalPoly> {
    std::vector<Rational> r(std::max(coeffs_.size(), o.coeffs_.size()));
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        r[i] = coeffs_[i];
    }
    for (std::size_t i = 0; i < o.coeffs_.size(); ++i) {
        auto sum = r[i].add(o.coeffs_[i]);
        if (!sum) {
            return make_error<RationalPoly>(sum.error());
        }
        r[i] = *sum;
    }
    return from_coeffs(std::move(r));
}

auto RationalPoly::subtract(const RationalPoly& o) const -> Result<RationalPoly> {
    std::vector<Rational> r(std::max(coeffs_.size(), o.coeffs_.size()));
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        r[i] = coeffs_[i];
    }
    for (std::size_t i = 0; i < o.coeffs_.size(); ++i) {
        auto diff = r[i].subtract(o.coeffs_[i]);
        if (!diff) {
            return make_error<RationalPoly>(diff.error());
        }
        r[i] = *diff;
    }
    return from_coeffs(std::move(r));
}

auto RationalPoly::scale(const Rational& s) const -> Result<RationalPoly> {
    if (s.is_zero()) {
        return RationalPoly{};
    }
    std::vector<Rational> r(coeffs_.size());
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        auto p = coeffs_[i].multiply(s);
        if (!p) {
            return make_error<RationalPoly>(p.error());
        }
        r[i] = *p;
    }
    return from_coeffs(std::move(r));  // no zeros introduced (s != 0), but trim is cheap
}

auto RationalPoly::multiply(const RationalPoly& o) const -> Result<RationalPoly> {
    if (is_zero() || o.is_zero()) {
        return RationalPoly{};
    }
    // Guard the result-width computation itself against std::size_t wrap before
    // allocating (both operands are non-empty here).
    if (coeffs_.size() > std::numeric_limits<std::size_t>::max() - o.coeffs_.size()) {
        return make_error<RationalPoly>(MathError::overflow);
    }
    std::vector<Rational> r(coeffs_.size() + o.coeffs_.size() - 1);
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        for (std::size_t j = 0; j < o.coeffs_.size(); ++j) {
            auto prod = coeffs_[i].multiply(o.coeffs_[j]);
            if (!prod) {
                return make_error<RationalPoly>(prod.error());
            }
            auto sum = r[i + j].add(*prod);
            if (!sum) {
                return make_error<RationalPoly>(sum.error());
            }
            r[i + j] = *sum;
        }
    }
    return from_coeffs(std::move(r));
}

auto RationalPoly::monic() const -> Result<RationalPoly> {
    if (is_zero()) {
        return RationalPoly{};
    }
    const Rational lc = leading_coefficient();
    if (lc == Rational::from_int(1)) {
        return *this;
    }
    auto inv = Rational::from_int(1).divide(lc);  // lc != 0 (non-zero polynomial)
    if (!inv) {
        return make_error<RationalPoly>(inv.error());
    }
    return scale(*inv);
}

auto RationalPoly::divide(const RationalPoly& divisor) const -> Result<RationalDivMod> {
    if (divisor.is_zero()) {
        return make_error<RationalDivMod>(MathError::division_by_zero);
    }
    if (degree() < divisor.degree()) {
        return RationalDivMod{.quotient = RationalPoly{}, .remainder = *this};
    }
    const Rational lc_b = divisor.leading_coefficient();
    const std::int64_t nb = divisor.degree();
    std::vector<Rational> quotient(static_cast<std::size_t>(degree() - nb) + 1);
    RationalPoly r = *this;
    while (!r.is_zero() && r.degree() >= nb) {
        // t = (lc(r) / lc(divisor)) * x^(deg r - nb); r <- r - t * divisor.
        auto q = r.leading_coefficient().divide(lc_b);
        if (!q) {
            return make_error<RationalDivMod>(q.error());
        }
        const std::size_t k = static_cast<std::size_t>(r.degree() - nb);
        quotient[k] = *q;
        auto tb = RationalPoly::monomial(*q, k).multiply(divisor);
        if (!tb) {
            return make_error<RationalDivMod>(tb.error());
        }
        auto next = r.subtract(*tb);  // leading term cancels; degree strictly drops
        if (!next) {
            return make_error<RationalDivMod>(next.error());
        }
        r = *next;
    }
    return RationalDivMod{.quotient = from_coeffs(std::move(quotient)), .remainder = r};
}

auto RationalPoly::gcd(const RationalPoly& o) const -> Result<RationalPoly> {
    RationalPoly a = *this;
    RationalPoly b = o;
    while (!b.is_zero()) {
        auto dm = a.divide(b);
        if (!dm) {
            return make_error<RationalPoly>(dm.error());
        }
        a = b;
        b = std::move(dm->remainder);
    }
    return a.monic();  // canonical representative (monic), or 0 when both inputs were 0
}

auto RationalPoly::derivative() const -> Result<RationalPoly> {
    if (coeffs_.size() <= 1) {
        return RationalPoly{};
    }
    // The exponent i is cast to int64 below; guard the (physically unreachable) case
    // of a degree beyond INT64_MAX so the cast can never wrap.
    if (coeffs_.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        return make_error<RationalPoly>(MathError::overflow);
    }
    std::vector<Rational> r(coeffs_.size() - 1);
    for (std::size_t i = 1; i < coeffs_.size(); ++i) {
        auto term = coeffs_[i].multiply(Rational::from_int(static_cast<std::int64_t>(i)));
        if (!term) {
            return make_error<RationalPoly>(term.error());
        }
        r[i - 1] = *term;
    }
    return from_coeffs(std::move(r));
}

auto RationalPoly::to_polynomial() const -> Result<Polynomial> {
    std::vector<std::int64_t> c(coeffs_.size());
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        if (!coeffs_[i].is_integer()) {
            return make_error<Polynomial>(MathError::domain_error);
        }
        c[i] = coeffs_[i].numerator();
    }
    return Polynomial{std::move(c)};
}

auto RationalPoly::to_string(std::string_view var) const -> std::string {
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
