// NimbleCAS exact dense univariate polynomials over an algebraic number field Q(alpha) on the
// UNBOUNDED rationals (§7 algebra substrate; the bignum mirror of nimblecas.algpoly for
// Rothstein-Trager residues over extension fields and Jordan canonical forms over general
// splitting fields that must exceed the int64 overflow envelope).
// @author Olumuyiwa Oluwasanmi
//
// A BigAlgebraicPoly is a polynomial in x whose coefficients are elements of ONE
// BigNumberField L = Q(alpha) = Q[t]/(m(t)). It is the arbitrary-precision analogue of
// nimblecas.algpoly's AlgebraicPoly: where AlgebraicPoly carries int64 AlgebraicNumber
// coefficients and surfaces MathError::overflow the moment a residue coefficient saturates,
// BigAlgebraicPoly carries BigAlgebraicNumber coefficients (BigRationalPoly residues) and has
// NO such ceiling. It is the natural home for the two computations that need polynomial
// arithmetic over an extension field beyond the int64 envelope:
//   * Rothstein-Trager over Q(alpha): the log argument v(x) = gcd_x(D, A - alpha*D') is a
//     gcd of two L[x] polynomials;
//   * Jordan over a splitting field: Trager's norm factorization and the eigenvalue
//     bookkeeping both manipulate L[x].
//
// REPRESENTATION. A dense coefficient vector, coeffs_[i] the coefficient of x^i, every
// coefficient a BigAlgebraicNumber in the SAME field (equal minimal polynomials). Trailing
// zero coefficients are trimmed, so the zero polynomial has an empty vector and degree() is
// -1 (matching BigRationalPoly's convention). Because BigAlgebraicNumber has no default
// constructor (its residue always names a field), coefficient vectors are always built by
// push_back / field.zero() fill, never resize().
//
// FIELD. L is a field: every nonzero BigAlgebraicNumber is invertible (extended Euclid in
// Q[t]). Hence general Euclidean division works with only the divisor's leading-coefficient
// inverse, and monic gcd is the plain Euclidean algorithm; remainders are re-normalised to
// monic every step to bound coefficient growth. (Should the modulus be reducible, a residue
// sharing a factor with it surfaces MathError::division_by_zero from inverse() — inherited
// honestly from BigAlgebraicNumber rather than fabricating a wrong value.)
//
// HONESTY (Rule 32). Every fallible operation returns Result<T>; nothing throws and nothing
// returns a plausible-but-wrong value. Combining polynomials over different fields is
// MathError::domain_error; dividing (or taking monic / the leading inverse) by the zero
// polynomial is MathError::division_by_zero. Because the underlying BigRational /
// BigRationalPoly arithmetic only combines magnitudes, it CANNOT overflow — MathError::overflow
// can NEVER arise in this tier, which is the entire point of the bignum mirror.

export module nimblecas.bigalgpoly;

import std;
import nimblecas.core;
import nimblecas.bigrational;
import nimblecas.bigratpoly;
import nimblecas.bigalgnum;

export namespace nimblecas {

struct BigAlgPolyDivMod;  // { BigAlgebraicPoly quotient; BigAlgebraicPoly remainder; } — below.

// ---------------------------------------------------------------------------
// BigAlgebraicPoly — a dense polynomial in x over a single BigNumberField L.
// ---------------------------------------------------------------------------
class BigAlgebraicPoly {
public:
    // The zero polynomial over L (empty coefficient vector, degree -1).
    [[nodiscard]] static auto zero(const BigNumberField& field) -> BigAlgebraicPoly;

    // Build a polynomial from coefficients (index i is the coefficient of x^i). Every
    // coefficient must live in `field`; a mismatch is MathError::domain_error. Trailing
    // zeros are trimmed.
    [[nodiscard]] static auto from_coeffs(const BigNumberField& field,
                                          std::vector<BigAlgebraicNumber> coeffs)
        -> Result<BigAlgebraicPoly>;

    // Lift a polynomial over Q into L[x], embedding each BigRational coefficient via
    // BigNumberField::from_bigrational (which cannot fail). Trailing zeros are trimmed.
    [[nodiscard]] static auto embed(const BigNumberField& field, const BigRationalPoly& p)
        -> BigAlgebraicPoly;

    [[nodiscard]] auto field() const -> const BigNumberField& { return field_; }
    [[nodiscard]] auto is_zero() const -> bool { return coeffs_.empty(); }
    // Degree, with the -1 convention for the zero polynomial.
    [[nodiscard]] auto degree() const -> std::int64_t {
        return coeffs_.empty() ? -1 : static_cast<std::int64_t>(coeffs_.size()) - 1;
    }
    [[nodiscard]] auto coefficients() const -> std::span<const BigAlgebraicNumber> {
        return coeffs_;
    }
    // Coefficient of x^i; the field's zero beyond the stored degree.
    [[nodiscard]] auto coefficient(std::size_t i) const -> BigAlgebraicNumber;
    // Leading coefficient; the field's zero for the zero polynomial.
    [[nodiscard]] auto leading_coefficient() const -> BigAlgebraicNumber;
    // Equality: same field AND identical (trimmed) coefficient vectors.
    [[nodiscard]] auto is_equal(const BigAlgebraicPoly& o) const -> bool;

    // --- ring operations (all exact; NEVER overflow; field-mismatch -> domain_error) ---
    [[nodiscard]] auto add(const BigAlgebraicPoly& o) const -> Result<BigAlgebraicPoly>;
    [[nodiscard]] auto subtract(const BigAlgebraicPoly& o) const -> Result<BigAlgebraicPoly>;
    [[nodiscard]] auto multiply(const BigAlgebraicPoly& o) const -> Result<BigAlgebraicPoly>;
    // Scale every coefficient by a field element c (c must live in the same field).
    [[nodiscard]] auto scale(const BigAlgebraicNumber& c) const -> Result<BigAlgebraicPoly>;
    // Formal derivative d/dx.
    [[nodiscard]] auto derivative() const -> Result<BigAlgebraicPoly>;
    // Horner evaluation at a field element x (x must live in the same field).
    [[nodiscard]] auto evaluate(const BigAlgebraicNumber& x) const -> Result<BigAlgebraicNumber>;

    // Euclidean division: *this == quotient*o + remainder, deg(remainder) < deg(o).
    // Dividing by the zero polynomial is MathError::division_by_zero.
    [[nodiscard]] auto divide(const BigAlgebraicPoly& o) const -> Result<BigAlgPolyDivMod>;
    // Normalise to monic (leading coefficient 1). The zero polynomial has no monic form:
    // MathError::division_by_zero.
    [[nodiscard]] auto monic() const -> Result<BigAlgebraicPoly>;
    // Monic greatest common divisor via the Euclidean algorithm; gcd(0, 0) == 0. The
    // result is monic (or the zero polynomial when both inputs are zero).
    [[nodiscard]] auto gcd(const BigAlgebraicPoly& o) const -> Result<BigAlgebraicPoly>;

    [[nodiscard]] auto to_string(std::string_view var = "x",
                                 std::string_view alpha_var = "a") const -> std::string;

private:
    // Precondition: every element of `coeffs` lives in `field` and trailing zeros are
    // trimmed. Only the factories / operations above (which enforce this) call it.
    BigAlgebraicPoly(BigNumberField field, std::vector<BigAlgebraicNumber> coeffs)
        : field_(std::move(field)), coeffs_(std::move(coeffs)) {}

    BigNumberField field_;
    std::vector<BigAlgebraicNumber> coeffs_;  // coeffs_[i] = coeff of x^i; trimmed; empty = 0
};

// Quotient/remainder pair returned by BigAlgebraicPoly::divide.
struct BigAlgPolyDivMod {
    BigAlgebraicPoly quotient;
    BigAlgebraicPoly remainder;
};

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// Drop trailing zero coefficients so the representation is canonical.
auto trim_bigalg(std::vector<BigAlgebraicNumber>& c) -> void {
    while (!c.empty() && c.back().is_zero()) {
        c.pop_back();
    }
}

}  // namespace

// --- factories --------------------------------------------------------------

auto BigAlgebraicPoly::zero(const BigNumberField& field) -> BigAlgebraicPoly {
    return BigAlgebraicPoly{field, {}};
}

auto BigAlgebraicPoly::from_coeffs(const BigNumberField& field,
                                   std::vector<BigAlgebraicNumber> coeffs)
    -> Result<BigAlgebraicPoly> {
    for (const BigAlgebraicNumber& c : coeffs) {
        if (!c.field().is_same(field)) {
            return make_error<BigAlgebraicPoly>(MathError::domain_error);
        }
    }
    trim_bigalg(coeffs);
    return BigAlgebraicPoly{field, std::move(coeffs)};
}

auto BigAlgebraicPoly::embed(const BigNumberField& field, const BigRationalPoly& p)
    -> BigAlgebraicPoly {
    std::vector<BigAlgebraicNumber> coeffs;
    const std::span<const BigRational> rc = p.coefficients();
    coeffs.reserve(rc.size());
    for (const BigRational& r : rc) {
        coeffs.push_back(field.from_bigrational(r));  // constant, always reduced; cannot fail
    }
    trim_bigalg(coeffs);
    return BigAlgebraicPoly{field, std::move(coeffs)};
}

// --- element access ---------------------------------------------------------

auto BigAlgebraicPoly::coefficient(std::size_t i) const -> BigAlgebraicNumber {
    return i < coeffs_.size() ? coeffs_[i] : field_.zero();
}

auto BigAlgebraicPoly::leading_coefficient() const -> BigAlgebraicNumber {
    return coeffs_.empty() ? field_.zero() : coeffs_.back();
}

auto BigAlgebraicPoly::is_equal(const BigAlgebraicPoly& o) const -> bool {
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

auto BigAlgebraicPoly::add(const BigAlgebraicPoly& o) const -> Result<BigAlgebraicPoly> {
    if (!field_.is_same(o.field_)) {
        return make_error<BigAlgebraicPoly>(MathError::domain_error);
    }
    const std::size_t n = std::max(coeffs_.size(), o.coeffs_.size());
    std::vector<BigAlgebraicNumber> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto s = coefficient(i).add(o.coefficient(i));
        if (!s) {
            return make_error<BigAlgebraicPoly>(s.error());
        }
        out.push_back(std::move(*s));
    }
    trim_bigalg(out);
    return BigAlgebraicPoly{field_, std::move(out)};
}

auto BigAlgebraicPoly::subtract(const BigAlgebraicPoly& o) const -> Result<BigAlgebraicPoly> {
    if (!field_.is_same(o.field_)) {
        return make_error<BigAlgebraicPoly>(MathError::domain_error);
    }
    const std::size_t n = std::max(coeffs_.size(), o.coeffs_.size());
    std::vector<BigAlgebraicNumber> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto d = coefficient(i).subtract(o.coefficient(i));
        if (!d) {
            return make_error<BigAlgebraicPoly>(d.error());
        }
        out.push_back(std::move(*d));
    }
    trim_bigalg(out);
    return BigAlgebraicPoly{field_, std::move(out)};
}

auto BigAlgebraicPoly::multiply(const BigAlgebraicPoly& o) const -> Result<BigAlgebraicPoly> {
    if (!field_.is_same(o.field_)) {
        return make_error<BigAlgebraicPoly>(MathError::domain_error);
    }
    if (coeffs_.empty() || o.coeffs_.empty()) {
        return BigAlgebraicPoly{field_, {}};
    }
    std::vector<BigAlgebraicNumber> acc(coeffs_.size() + o.coeffs_.size() - 1, field_.zero());
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        for (std::size_t j = 0; j < o.coeffs_.size(); ++j) {
            auto prod = coeffs_[i].multiply(o.coeffs_[j]);
            if (!prod) {
                return make_error<BigAlgebraicPoly>(prod.error());
            }
            auto sum = acc[i + j].add(*prod);
            if (!sum) {
                return make_error<BigAlgebraicPoly>(sum.error());
            }
            acc[i + j] = std::move(*sum);
        }
    }
    trim_bigalg(acc);
    return BigAlgebraicPoly{field_, std::move(acc)};
}

auto BigAlgebraicPoly::scale(const BigAlgebraicNumber& c) const -> Result<BigAlgebraicPoly> {
    if (!c.field().is_same(field_)) {
        return make_error<BigAlgebraicPoly>(MathError::domain_error);
    }
    std::vector<BigAlgebraicNumber> out;
    out.reserve(coeffs_.size());
    for (const BigAlgebraicNumber& a : coeffs_) {
        auto p = a.multiply(c);
        if (!p) {
            return make_error<BigAlgebraicPoly>(p.error());
        }
        out.push_back(std::move(*p));
    }
    trim_bigalg(out);  // c may be zero
    return BigAlgebraicPoly{field_, std::move(out)};
}

auto BigAlgebraicPoly::derivative() const -> Result<BigAlgebraicPoly> {
    if (coeffs_.size() < 2) {
        return BigAlgebraicPoly{field_, {}};  // constant or zero -> 0
    }
    std::vector<BigAlgebraicNumber> out;
    out.reserve(coeffs_.size() - 1);
    for (std::size_t i = 1; i < coeffs_.size(); ++i) {
        const BigAlgebraicNumber k =
            field_.from_bigrational(BigRational::from_int(static_cast<std::int64_t>(i)));
        auto term = coeffs_[i].multiply(k);
        if (!term) {
            return make_error<BigAlgebraicPoly>(term.error());
        }
        out.push_back(std::move(*term));
    }
    trim_bigalg(out);
    return BigAlgebraicPoly{field_, std::move(out)};
}

auto BigAlgebraicPoly::evaluate(const BigAlgebraicNumber& x) const -> Result<BigAlgebraicNumber> {
    if (!x.field().is_same(field_)) {
        return make_error<BigAlgebraicNumber>(MathError::domain_error);
    }
    BigAlgebraicNumber acc = field_.zero();
    for (std::size_t i = coeffs_.size(); i-- > 0;) {
        auto mul = acc.multiply(x);
        if (!mul) {
            return make_error<BigAlgebraicNumber>(mul.error());
        }
        auto add = mul->add(coeffs_[i]);
        if (!add) {
            return make_error<BigAlgebraicNumber>(add.error());
        }
        acc = std::move(*add);
    }
    return acc;
}

auto BigAlgebraicPoly::divide(const BigAlgebraicPoly& o) const -> Result<BigAlgPolyDivMod> {
    if (!field_.is_same(o.field_)) {
        return make_error<BigAlgPolyDivMod>(MathError::domain_error);
    }
    if (o.coeffs_.empty()) {
        return make_error<BigAlgPolyDivMod>(MathError::division_by_zero);
    }
    auto lead_inv = o.leading_coefficient().inverse();  // o nonzero => leading nonzero
    if (!lead_inv) {
        return make_error<BigAlgPolyDivMod>(lead_inv.error());
    }
    const std::size_t d_o = o.coeffs_.size() - 1;
    std::vector<BigAlgebraicNumber> rem = coeffs_;  // working remainder
    std::vector<BigAlgebraicNumber> quo;
    if (coeffs_.size() > d_o) {
        quo.assign(coeffs_.size() - d_o, field_.zero());
    }
    while (!rem.empty() && rem.size() - 1 >= d_o) {
        const std::size_t shift = (rem.size() - 1) - d_o;
        auto factor = rem.back().multiply(*lead_inv);
        if (!factor) {
            return make_error<BigAlgPolyDivMod>(factor.error());
        }
        quo[shift] = *factor;
        for (std::size_t j = 0; j <= d_o; ++j) {
            auto term = factor->multiply(o.coeffs_[j]);
            if (!term) {
                return make_error<BigAlgPolyDivMod>(term.error());
            }
            auto diff = rem[shift + j].subtract(*term);
            if (!diff) {
                return make_error<BigAlgPolyDivMod>(diff.error());
            }
            rem[shift + j] = std::move(*diff);
        }
        trim_bigalg(rem);  // the top coefficient just cancelled
    }
    trim_bigalg(quo);
    return BigAlgPolyDivMod{BigAlgebraicPoly{field_, std::move(quo)},
                            BigAlgebraicPoly{field_, std::move(rem)}};
}

auto BigAlgebraicPoly::monic() const -> Result<BigAlgebraicPoly> {
    if (coeffs_.empty()) {
        return make_error<BigAlgebraicPoly>(MathError::division_by_zero);
    }
    auto inv = leading_coefficient().inverse();
    if (!inv) {
        return make_error<BigAlgebraicPoly>(inv.error());
    }
    return scale(*inv);
}

auto BigAlgebraicPoly::gcd(const BigAlgebraicPoly& o) const -> Result<BigAlgebraicPoly> {
    if (!field_.is_same(o.field_)) {
        return make_error<BigAlgebraicPoly>(MathError::domain_error);
    }
    BigAlgebraicPoly a = *this;
    BigAlgebraicPoly b = o;
    while (!b.is_zero()) {
        auto dm = a.divide(b);
        if (!dm) {
            return make_error<BigAlgebraicPoly>(dm.error());
        }
        a = std::move(b);
        if (dm->remainder.is_zero()) {
            b = std::move(dm->remainder);
        } else {
            // Re-normalise the remainder to monic each step to bound coefficient growth.
            auto rm = dm->remainder.monic();
            if (!rm) {
                return make_error<BigAlgebraicPoly>(rm.error());
            }
            b = std::move(*rm);
        }
    }
    if (a.is_zero()) {
        return a;  // gcd(0, 0) == 0
    }
    return a.monic();
}

auto BigAlgebraicPoly::to_string(std::string_view var, std::string_view alpha_var) const
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
