// NimbleCAS exact dense univariate polynomials over an algebraic number field Q(alpha)
// (§7 algebra substrate; the shared arithmetic layer for Rothstein-Trager residues over
// extension fields and for Jordan canonical forms over general splitting fields).
// @author Olumuyiwa Oluwasanmi
//
// An AlgebraicPoly is a polynomial in x whose coefficients are elements of ONE NumberField
// L = Q(alpha) = Q[t]/(m(t)). It is the natural home for the two computations that need
// polynomial arithmetic over an extension field:
//   * Rothstein-Trager over Q(alpha): the log argument v(x) = gcd_x(D, A - alpha*D') is a
//     gcd of two L[x] polynomials;
//   * Jordan over a splitting field: Trager's norm factorization and the eigenvalue
//     bookkeeping both manipulate L[x].
//
// REPRESENTATION. A dense coefficient vector, coeffs_[i] the coefficient of x^i, every
// coefficient an AlgebraicNumber in the SAME field (equal minimal polynomials). Trailing
// zero coefficients are trimmed, so the zero polynomial has an empty vector and degree() is
// -1 (matching RationalPoly's convention). Because AlgebraicNumber has no default
// constructor (its residue always names a field), coefficient vectors are always built by
// push_back / field.zero() fill, never resize().
//
// FIELD. L is a field: every nonzero AlgebraicNumber is invertible (extended Euclid in
// Q[t]). Hence general Euclidean division works with only the divisor's leading-coefficient
// inverse, and monic gcd is the plain Euclidean algorithm; remainders are re-normalised to
// monic every step to bound coefficient growth.
//
// HONESTY (Rule 32). Every fallible operation returns Result<T>; nothing throws and nothing
// returns a plausible-but-wrong value. Combining polynomials over different fields is
// MathError::domain_error; dividing (or taking monic / the leading inverse) by the zero
// polynomial is MathError::division_by_zero. All underlying int64 rational arithmetic is
// overflow-checked and surfaces MathError::overflow (inherited from AlgebraicNumber /
// RationalPoly) rather than wrapping.

export module nimblecas.algpoly;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.algnum;

export namespace nimblecas {

struct AlgPolyDivMod;  // { AlgebraicPoly quotient; AlgebraicPoly remainder; } — defined below.

// ---------------------------------------------------------------------------
// AlgebraicPoly — a dense polynomial in x over a single NumberField L.
// ---------------------------------------------------------------------------
class AlgebraicPoly {
public:
    // The zero polynomial over L (empty coefficient vector, degree -1).
    [[nodiscard]] static auto zero(const NumberField& field) -> AlgebraicPoly;

    // Build a polynomial from coefficients (index i is the coefficient of x^i). Every
    // coefficient must live in `field`; a mismatch is MathError::domain_error. Trailing
    // zeros are trimmed.
    [[nodiscard]] static auto from_coeffs(const NumberField& field,
                                          std::vector<AlgebraicNumber> coeffs)
        -> Result<AlgebraicPoly>;

    // Lift a polynomial over Q into L[x], embedding each rational coefficient via
    // NumberField::from_rational (which cannot fail). Trailing zeros are trimmed.
    [[nodiscard]] static auto embed(const NumberField& field, const RationalPoly& p)
        -> AlgebraicPoly;

    [[nodiscard]] auto field() const -> const NumberField& { return field_; }
    [[nodiscard]] auto is_zero() const -> bool { return coeffs_.empty(); }
    // Degree, with the -1 convention for the zero polynomial.
    [[nodiscard]] auto degree() const -> std::int64_t {
        return coeffs_.empty() ? -1 : static_cast<std::int64_t>(coeffs_.size()) - 1;
    }
    [[nodiscard]] auto coefficients() const -> std::span<const AlgebraicNumber> {
        return coeffs_;
    }
    // Coefficient of x^i; the field's zero beyond the stored degree.
    [[nodiscard]] auto coefficient(std::size_t i) const -> AlgebraicNumber;
    // Leading coefficient; the field's zero for the zero polynomial.
    [[nodiscard]] auto leading_coefficient() const -> AlgebraicNumber;
    // Equality: same field AND identical (trimmed) coefficient vectors.
    [[nodiscard]] auto is_equal(const AlgebraicPoly& o) const -> bool;

    // --- ring operations (all exact; overflow-checked; field-mismatch -> domain_error) ---
    [[nodiscard]] auto add(const AlgebraicPoly& o) const -> Result<AlgebraicPoly>;
    [[nodiscard]] auto subtract(const AlgebraicPoly& o) const -> Result<AlgebraicPoly>;
    [[nodiscard]] auto multiply(const AlgebraicPoly& o) const -> Result<AlgebraicPoly>;
    // Scale every coefficient by a field element c (c must live in the same field).
    [[nodiscard]] auto scale(const AlgebraicNumber& c) const -> Result<AlgebraicPoly>;
    // Formal derivative d/dx.
    [[nodiscard]] auto derivative() const -> Result<AlgebraicPoly>;
    // Horner evaluation at a field element x (x must live in the same field).
    [[nodiscard]] auto evaluate(const AlgebraicNumber& x) const -> Result<AlgebraicNumber>;

    // Euclidean division: *this == quotient*o + remainder, deg(remainder) < deg(o).
    // Dividing by the zero polynomial is MathError::division_by_zero.
    [[nodiscard]] auto divide(const AlgebraicPoly& o) const -> Result<AlgPolyDivMod>;
    // Normalise to monic (leading coefficient 1). The zero polynomial has no monic form:
    // MathError::division_by_zero.
    [[nodiscard]] auto monic() const -> Result<AlgebraicPoly>;
    // Monic greatest common divisor via the Euclidean algorithm; gcd(0, 0) == 0. The
    // result is monic (or the zero polynomial when both inputs are zero).
    [[nodiscard]] auto gcd(const AlgebraicPoly& o) const -> Result<AlgebraicPoly>;

    [[nodiscard]] auto to_string(std::string_view var = "x",
                                 std::string_view alpha_var = "a") const -> std::string;

private:
    // Precondition: every element of `coeffs` lives in `field` and trailing zeros are
    // trimmed. Only the factories / operations above (which enforce this) call it.
    AlgebraicPoly(NumberField field, std::vector<AlgebraicNumber> coeffs)
        : field_(std::move(field)), coeffs_(std::move(coeffs)) {}

    NumberField field_;
    std::vector<AlgebraicNumber> coeffs_;  // coeffs_[i] = coeff of x^i; trimmed; empty = 0
};

// Quotient/remainder pair returned by AlgebraicPoly::divide.
struct AlgPolyDivMod {
    AlgebraicPoly quotient;
    AlgebraicPoly remainder;
};

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// Drop trailing zero coefficients so the representation is canonical.
auto trim_alg(std::vector<AlgebraicNumber>& c) -> void {
    while (!c.empty() && c.back().is_zero()) {
        c.pop_back();
    }
}

}  // namespace

// --- factories --------------------------------------------------------------

auto AlgebraicPoly::zero(const NumberField& field) -> AlgebraicPoly {
    return AlgebraicPoly{field, {}};
}

auto AlgebraicPoly::from_coeffs(const NumberField& field, std::vector<AlgebraicNumber> coeffs)
    -> Result<AlgebraicPoly> {
    for (const AlgebraicNumber& c : coeffs) {
        if (!c.field().is_same(field)) {
            return make_error<AlgebraicPoly>(MathError::domain_error);
        }
    }
    trim_alg(coeffs);
    return AlgebraicPoly{field, std::move(coeffs)};
}

auto AlgebraicPoly::embed(const NumberField& field, const RationalPoly& p) -> AlgebraicPoly {
    std::vector<AlgebraicNumber> coeffs;
    const std::span<const Rational> rc = p.coefficients();
    coeffs.reserve(rc.size());
    for (const Rational& r : rc) {
        coeffs.push_back(field.from_rational(r));  // constant, always reduced; cannot fail
    }
    trim_alg(coeffs);
    return AlgebraicPoly{field, std::move(coeffs)};
}

// --- element access ---------------------------------------------------------

auto AlgebraicPoly::coefficient(std::size_t i) const -> AlgebraicNumber {
    return i < coeffs_.size() ? coeffs_[i] : field_.zero();
}

auto AlgebraicPoly::leading_coefficient() const -> AlgebraicNumber {
    return coeffs_.empty() ? field_.zero() : coeffs_.back();
}

auto AlgebraicPoly::is_equal(const AlgebraicPoly& o) const -> bool {
    if (!field_.is_same(o.field_) || coeffs_.size() != o.coeffs_.size()) {
        return false;
    }
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        if (!coeffs_[i].is_equal(o.coeffs_[i])) {
            return false;
        }
    }
    return true;
}

// --- ring operations --------------------------------------------------------

auto AlgebraicPoly::add(const AlgebraicPoly& o) const -> Result<AlgebraicPoly> {
    if (!field_.is_same(o.field_)) {
        return make_error<AlgebraicPoly>(MathError::domain_error);
    }
    const std::size_t n = std::max(coeffs_.size(), o.coeffs_.size());
    std::vector<AlgebraicNumber> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto s = coefficient(i).add(o.coefficient(i));
        if (!s) {
            return make_error<AlgebraicPoly>(s.error());
        }
        out.push_back(std::move(*s));
    }
    trim_alg(out);
    return AlgebraicPoly{field_, std::move(out)};
}

auto AlgebraicPoly::subtract(const AlgebraicPoly& o) const -> Result<AlgebraicPoly> {
    if (!field_.is_same(o.field_)) {
        return make_error<AlgebraicPoly>(MathError::domain_error);
    }
    const std::size_t n = std::max(coeffs_.size(), o.coeffs_.size());
    std::vector<AlgebraicNumber> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto d = coefficient(i).subtract(o.coefficient(i));
        if (!d) {
            return make_error<AlgebraicPoly>(d.error());
        }
        out.push_back(std::move(*d));
    }
    trim_alg(out);
    return AlgebraicPoly{field_, std::move(out)};
}

auto AlgebraicPoly::multiply(const AlgebraicPoly& o) const -> Result<AlgebraicPoly> {
    if (!field_.is_same(o.field_)) {
        return make_error<AlgebraicPoly>(MathError::domain_error);
    }
    if (coeffs_.empty() || o.coeffs_.empty()) {
        return AlgebraicPoly{field_, {}};
    }
    std::vector<AlgebraicNumber> acc(coeffs_.size() + o.coeffs_.size() - 1, field_.zero());
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        for (std::size_t j = 0; j < o.coeffs_.size(); ++j) {
            auto prod = coeffs_[i].multiply(o.coeffs_[j]);
            if (!prod) {
                return make_error<AlgebraicPoly>(prod.error());
            }
            auto sum = acc[i + j].add(*prod);
            if (!sum) {
                return make_error<AlgebraicPoly>(sum.error());
            }
            acc[i + j] = std::move(*sum);
        }
    }
    trim_alg(acc);
    return AlgebraicPoly{field_, std::move(acc)};
}

auto AlgebraicPoly::scale(const AlgebraicNumber& c) const -> Result<AlgebraicPoly> {
    if (!c.field().is_same(field_)) {
        return make_error<AlgebraicPoly>(MathError::domain_error);
    }
    std::vector<AlgebraicNumber> out;
    out.reserve(coeffs_.size());
    for (const AlgebraicNumber& a : coeffs_) {
        auto p = a.multiply(c);
        if (!p) {
            return make_error<AlgebraicPoly>(p.error());
        }
        out.push_back(std::move(*p));
    }
    trim_alg(out);  // c may be zero
    return AlgebraicPoly{field_, std::move(out)};
}

auto AlgebraicPoly::derivative() const -> Result<AlgebraicPoly> {
    if (coeffs_.size() < 2) {
        return AlgebraicPoly{field_, {}};  // constant or zero -> 0
    }
    std::vector<AlgebraicNumber> out;
    out.reserve(coeffs_.size() - 1);
    for (std::size_t i = 1; i < coeffs_.size(); ++i) {
        const AlgebraicNumber k = field_.from_rational(Rational::from_int(static_cast<std::int64_t>(i)));
        auto term = coeffs_[i].multiply(k);
        if (!term) {
            return make_error<AlgebraicPoly>(term.error());
        }
        out.push_back(std::move(*term));
    }
    trim_alg(out);
    return AlgebraicPoly{field_, std::move(out)};
}

auto AlgebraicPoly::evaluate(const AlgebraicNumber& x) const -> Result<AlgebraicNumber> {
    if (!x.field().is_same(field_)) {
        return make_error<AlgebraicNumber>(MathError::domain_error);
    }
    AlgebraicNumber acc = field_.zero();
    for (std::size_t i = coeffs_.size(); i-- > 0;) {
        auto mul = acc.multiply(x);
        if (!mul) {
            return make_error<AlgebraicNumber>(mul.error());
        }
        auto add = mul->add(coeffs_[i]);
        if (!add) {
            return make_error<AlgebraicNumber>(add.error());
        }
        acc = std::move(*add);
    }
    return acc;
}

auto AlgebraicPoly::divide(const AlgebraicPoly& o) const -> Result<AlgPolyDivMod> {
    if (!field_.is_same(o.field_)) {
        return make_error<AlgPolyDivMod>(MathError::domain_error);
    }
    if (o.coeffs_.empty()) {
        return make_error<AlgPolyDivMod>(MathError::division_by_zero);
    }
    auto lead_inv = o.leading_coefficient().inverse();  // o nonzero => leading nonzero
    if (!lead_inv) {
        return make_error<AlgPolyDivMod>(lead_inv.error());
    }
    const std::size_t d_o = o.coeffs_.size() - 1;
    std::vector<AlgebraicNumber> rem = coeffs_;  // working remainder
    std::vector<AlgebraicNumber> quo;
    if (coeffs_.size() > d_o) {
        quo.assign(coeffs_.size() - d_o, field_.zero());
    }
    while (!rem.empty() && rem.size() - 1 >= d_o) {
        const std::size_t shift = (rem.size() - 1) - d_o;
        auto factor = rem.back().multiply(*lead_inv);
        if (!factor) {
            return make_error<AlgPolyDivMod>(factor.error());
        }
        quo[shift] = *factor;
        for (std::size_t j = 0; j <= d_o; ++j) {
            auto term = factor->multiply(o.coeffs_[j]);
            if (!term) {
                return make_error<AlgPolyDivMod>(term.error());
            }
            auto diff = rem[shift + j].subtract(*term);
            if (!diff) {
                return make_error<AlgPolyDivMod>(diff.error());
            }
            rem[shift + j] = std::move(*diff);
        }
        trim_alg(rem);  // the top coefficient just cancelled
    }
    trim_alg(quo);
    return AlgPolyDivMod{AlgebraicPoly{field_, std::move(quo)},
                         AlgebraicPoly{field_, std::move(rem)}};
}

auto AlgebraicPoly::monic() const -> Result<AlgebraicPoly> {
    if (coeffs_.empty()) {
        return make_error<AlgebraicPoly>(MathError::division_by_zero);
    }
    auto inv = leading_coefficient().inverse();
    if (!inv) {
        return make_error<AlgebraicPoly>(inv.error());
    }
    return scale(*inv);
}

auto AlgebraicPoly::gcd(const AlgebraicPoly& o) const -> Result<AlgebraicPoly> {
    if (!field_.is_same(o.field_)) {
        return make_error<AlgebraicPoly>(MathError::domain_error);
    }
    AlgebraicPoly a = *this;
    AlgebraicPoly b = o;
    while (!b.is_zero()) {
        auto dm = a.divide(b);
        if (!dm) {
            return make_error<AlgebraicPoly>(dm.error());
        }
        a = std::move(b);
        if (dm->remainder.is_zero()) {
            b = std::move(dm->remainder);
        } else {
            // Re-normalise the remainder to monic each step to bound coefficient growth.
            auto rm = dm->remainder.monic();
            if (!rm) {
                return make_error<AlgebraicPoly>(rm.error());
            }
            b = std::move(*rm);
        }
    }
    if (a.is_zero()) {
        return a;  // gcd(0, 0) == 0
    }
    return a.monic();
}

auto AlgebraicPoly::to_string(std::string_view var, std::string_view alpha_var) const
    -> std::string {
    if (coeffs_.empty()) {
        return "0";
    }
    std::string out;
    bool first = true;
    for (std::size_t i = coeffs_.size(); i-- > 0;) {
        if (coeffs_[i].is_zero()) {
            continue;
        }
        if (!first) {
            out += " + ";
        }
        first = false;
        out += '(';
        out += coeffs_[i].to_string(alpha_var);
        out += ')';
        if (i == 1) {
            out += '*';
            out += var;
        } else if (i >= 2) {
            out += '*';
            out += var;
            out += '^';
            out += std::to_string(i);
        }
    }
    return first ? "0" : out;
}

}  // namespace nimblecas
