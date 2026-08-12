// NimbleCAS exact arithmetic in a simple algebraic number field Q(alpha) over the UNBOUNDED
// rationals (§7 algebra substrate; bignum groundwork for a splitting-field / Jordan Tier-3).
// @author Olumuyiwa Oluwasanmi
//
// This is the arbitrary-precision analogue of nimblecas.algnum. Where NumberField /
// AlgebraicNumber carry int64 RationalPoly residues and surface MathError::overflow the
// moment a residue coefficient saturates, BigNumberField / BigAlgebraicNumber carry
// BigRationalPoly residues and have NO such ceiling: it is the layer-2 field on which a
// bignum-backed splitting field can exceed the int64 overflow envelope. The public method
// surface mirrors algnum's exactly so this is a drop-in bignum replacement.
//
// A number field here is a simple extension Q(alpha) = Q[x]/(m(x)) for a monic minimal
// polynomial m of degree d >= 1 (a non-monic m is normalised to monic). An element is the
// unique BigRationalPoly residue of degree < d; {1, alpha, ..., alpha^(d-1)} is a Q-basis,
// so the residue is a normal form and equality is a coefficient-vector compare. Add /
// subtract / negate are componentwise; multiply is a polynomial product reduced mod m;
// inverse of a nonzero element uses the extended Euclidean algorithm over BigRationalPoly.
//
// DIVERGENCE FROM algnum (honest, documented). algnum verifies irreducibility of m via
// factor_over_Q, so gcd(a, m) is always a nonzero constant and every nonzero element is
// invertible. There is no bignum polynomial-factoring layer available here, so create()
// does NOT prove irreducibility — it only rejects the zero polynomial and constants and
// normalises to monic. The honesty is preserved at inversion time instead: if the modulus
// is reducible and the residue a shares a nontrivial factor with it, gcd(a, m) has degree
// >= 1 (not a unit) and inverse() reports MathError::division_by_zero rather than
// fabricating a wrong "inverse". Inverting the zero element is division_by_zero as well.
//
// HONESTY (Rule 32). Every fallible operation returns Result<T>; nothing throws. Because
// BigRational / BigRationalPoly arithmetic only combines magnitudes, it CANNOT overflow —
// MathError::overflow never arises from the field arithmetic, which is the entire point of
// this tier. Combining elements of different fields is MathError::domain_error; a negative
// power is domain_error. The modulus is held behind a CowPtr so copying a field (and every
// element that embeds it) is an O(1) refcount bump (Rule 22).

module;
#include <cassert>

export module nimblecas.bigalgnum;

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.bigratpoly;

export namespace nimblecas {

class BigAlgebraicNumber;  // element of a BigNumberField; defined below.

// ---------------------------------------------------------------------------
// BigNumberField — a simple algebraic extension Q(alpha) = Q[x]/(m(x)) over BigRational.
// ---------------------------------------------------------------------------
// Constructed from a minimal polynomial m of degree d >= 1 (a non-monic m is normalised to
// monic; a zero or constant m is rejected). Acts as the factory for its elements and carries
// the modulus their arithmetic reduces by.
class BigNumberField {
public:
    // Build Q[x]/(m). The minimal polynomial is normalised to monic (same ideal, same
    // field). Fails with MathError::domain_error when m is the zero polynomial or a nonzero
    // constant (degree < 1). Unlike algnum this does NOT verify irreducibility (no bignum
    // factoring layer); a reducible modulus is accepted and non-invertible residues surface
    // honestly as division_by_zero at inverse() time.
    [[nodiscard]] static auto create(const BigRationalPoly& minimal) -> Result<BigNumberField>;

    // The degree d = deg m of the extension ([Q(alpha):Q]).
    [[nodiscard]] auto degree() const -> std::int64_t { return modulus_->degree(); }
    // The monic minimal polynomial m.
    [[nodiscard]] auto modulus() const -> const BigRationalPoly& { return *modulus_; }
    // Same field iff the minimal polynomials are identical.
    [[nodiscard]] auto is_same(const BigNumberField& o) const -> bool {
        return modulus_->is_equal(*o.modulus_);
    }

    // --- element factories ---
    [[nodiscard]] auto zero() const -> BigAlgebraicNumber;   // 0
    [[nodiscard]] auto one() const -> BigAlgebraicNumber;    // 1
    // The constant c as an element (degree 0 < d, already reduced).
    [[nodiscard]] auto from_bigrational(const BigRational& c) const -> BigAlgebraicNumber;
    // The class of an arbitrary p in Q[x], i.e. p reduced mod m.
    [[nodiscard]] auto from_poly(const BigRationalPoly& p) const -> Result<BigAlgebraicNumber>;
    // The generator alpha = x mod m.
    [[nodiscard]] auto generator() const -> Result<BigAlgebraicNumber>;

    [[nodiscard]] auto to_string(std::string_view var = "a") const -> std::string {
        return std::format("Q[{}]/({})", var, modulus_->to_string(var));
    }

private:
    explicit BigNumberField(CowPtr<BigRationalPoly> m) : modulus_(std::move(m)) {}
    CowPtr<BigRationalPoly> modulus_;
};

// ---------------------------------------------------------------------------
// BigAlgebraicNumber — an element of a BigNumberField, in canonical residue form.
// ---------------------------------------------------------------------------
// Holds its field and the canonical residue (a BigRationalPoly of degree < d). Binary
// operations require both operands to live in the SAME field (equal minimal polynomials),
// else MathError::domain_error. No default constructor — build via the field factories.
class BigAlgebraicNumber {
public:
    [[nodiscard]] auto field() const -> const BigNumberField& { return field_; }
    // The canonical residue: coefficients of 1, alpha, ..., alpha^(d-1) (degree < d).
    [[nodiscard]] auto value() const -> const BigRationalPoly& { return value_; }

    [[nodiscard]] auto is_zero() const -> bool { return value_.is_zero(); }
    [[nodiscard]] auto is_one() const -> bool {
        return value_.is_equal(BigRationalPoly::constant(BigRational::from_int(1)));
    }
    // Equality: same field AND same residue.
    [[nodiscard]] auto is_equal(const BigAlgebraicNumber& o) const -> bool {
        return field_.is_same(o.field_) && value_.is_equal(o.value_);
    }

    // --- field operations (all exact; NEVER overflow) -----------------------
    [[nodiscard]] auto add(const BigAlgebraicNumber& o) const -> Result<BigAlgebraicNumber>;
    [[nodiscard]] auto subtract(const BigAlgebraicNumber& o) const -> Result<BigAlgebraicNumber>;
    [[nodiscard]] auto negate() const -> Result<BigAlgebraicNumber>;
    [[nodiscard]] auto scale(const BigRational& s) const -> Result<BigAlgebraicNumber>;
    [[nodiscard]] auto multiply(const BigAlgebraicNumber& o) const -> Result<BigAlgebraicNumber>;
    // Multiplicative inverse via the extended Euclidean algorithm over BigRationalPoly. The
    // zero element, or a residue sharing a nontrivial factor with a reducible modulus, has
    // no inverse: MathError::division_by_zero.
    [[nodiscard]] auto inverse() const -> Result<BigAlgebraicNumber>;
    // a / b = a * b^{-1}; division_by_zero when b has no inverse.
    [[nodiscard]] auto divide(const BigAlgebraicNumber& o) const -> Result<BigAlgebraicNumber>;
    // Non-negative integer power by repeated squaring; pow(0) == 1. A negative exponent is
    // MathError::domain_error (use inverse() first for negative powers).
    [[nodiscard]] auto pow(std::int64_t exponent) const -> Result<BigAlgebraicNumber>;

    // The field norm N(a) = det, and trace Tr(a) = trace, of the Q-linear
    // multiplication-by-a map on the basis {1, alpha, ..., alpha^(d-1)}. Both exact rationals.
    [[nodiscard]] auto norm() const -> Result<BigRational>;
    [[nodiscard]] auto trace() const -> Result<BigRational>;

    [[nodiscard]] auto to_string(std::string_view var = "a") const -> std::string {
        return value_.to_string(var);
    }

private:
    friend class BigNumberField;
    BigAlgebraicNumber(BigNumberField f, BigRationalPoly v)
        : field_(std::move(f)), value_(std::move(v)) {}

    // The d x d BigRational matrix of multiplication-by-*this on {1, ..., alpha^(d-1)}:
    // column j holds the residue coordinates of (*this) * alpha^j. Row-major.
    [[nodiscard]] auto mult_matrix() const
        -> Result<std::vector<std::vector<BigRational>>>;

    BigNumberField field_;
    BigRationalPoly value_;  // canonical residue, degree < d
};

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// Extended Euclidean algorithm over BigRationalPoly: returns (g, u, v) with u*a + v*b == g,
// g a greatest common divisor of a and b (the raw Bezout g, not normalised to monic — the
// inverse routine divides through by g's own leading value). Only divide is fallible; the
// magnitude-combining subtract/multiply are infallible over BigRational.
struct ExtGcd {
    BigRationalPoly g;
    BigRationalPoly u;
    BigRationalPoly v;
};

[[nodiscard]] auto ext_gcd(const BigRationalPoly& a, const BigRationalPoly& b)
    -> Result<ExtGcd> {
    BigRationalPoly r0 = a;
    BigRationalPoly r1 = b;
    BigRationalPoly s0 = BigRationalPoly::constant(BigRational::from_int(1));
    BigRationalPoly s1;  // zero
    BigRationalPoly t0;  // zero
    BigRationalPoly t1 = BigRationalPoly::constant(BigRational::from_int(1));
    while (!r1.is_zero()) {
        auto dm = r0.divide(r1);  // r1 != 0
        if (!dm) {
            return make_error<ExtGcd>(dm.error());
        }
        const BigRationalPoly& q = dm->quotient;
        // (s0, s1) <- (s1, s0 - q*s1); (t0, t1) <- (t1, t0 - q*t1); (r0, r1) <- (r1, rem).
        BigRationalPoly s_next = s0.subtract(q.multiply(s1));  // infallible
        BigRationalPoly t_next = t0.subtract(q.multiply(t1));  // infallible
        s0 = std::move(s1);
        s1 = std::move(s_next);
        t0 = std::move(t1);
        t1 = std::move(t_next);
        r0 = std::move(r1);
        r1 = std::move(dm->remainder);
    }
    return ExtGcd{.g = std::move(r0), .u = std::move(s0), .v = std::move(t0)};
}

// Exact determinant of a square BigRational matrix by Gaussian elimination over the field Q
// (BigRational is exact and unbounded, so this never rounds and never overflows). Row swaps
// flip the running sign; an all-zero pivot column yields a zero determinant.
[[nodiscard]] auto determinant(std::vector<std::vector<BigRational>> a) -> Result<BigRational> {
    const std::size_t n = a.size();
    BigRational det = BigRational::from_int(1);
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t piv = col;
        while (piv < n && a[piv][col].is_zero()) {
            ++piv;
        }
        if (piv == n) {
            return BigRational::from_int(0);  // singular: a full column of zeros below/at col
        }
        if (piv != col) {
            std::swap(a[piv], a[col]);
            det = det.negate();
        }
        const BigRational pivot = a[col][col];
        det = det.multiply(pivot);
        for (std::size_t r = col + 1; r < n; ++r) {
            if (a[r][col].is_zero()) {
                continue;
            }
            auto factor = a[r][col].divide(pivot);  // pivot != 0
            if (!factor) {
                return make_error<BigRational>(factor.error());
            }
            for (std::size_t c = col; c < n; ++c) {
                a[r][c] = a[r][c].subtract(factor->multiply(a[col][c]));
            }
        }
    }
    return det;
}

}  // namespace

// --- BigNumberField ---------------------------------------------------------

auto BigNumberField::create(const BigRationalPoly& minimal) -> Result<BigNumberField> {
    if (minimal.is_zero() || minimal.degree() < 1) {
        // The zero polynomial and nonzero constants define no proper extension.
        return make_error<BigNumberField>(MathError::domain_error);
    }
    // Normalise to monic (same ideal, hence same field). No irreducibility proof: the
    // bignum tier has no factoring layer, so honesty is enforced at inverse() instead.
    auto monic = minimal.monic();
    if (!monic) {
        return make_error<BigNumberField>(monic.error());
    }
    return BigNumberField{CowPtr<BigRationalPoly>::make(std::move(*monic))};
}

auto BigNumberField::zero() const -> BigAlgebraicNumber {
    return BigAlgebraicNumber{*this, BigRationalPoly{}};
}

auto BigNumberField::one() const -> BigAlgebraicNumber {
    return BigAlgebraicNumber{*this, BigRationalPoly::constant(BigRational::from_int(1))};
}

auto BigNumberField::from_bigrational(const BigRational& c) const -> BigAlgebraicNumber {
    return BigAlgebraicNumber{*this, BigRationalPoly::constant(c)};  // degree 0 < d
}

auto BigNumberField::from_poly(const BigRationalPoly& p) const -> Result<BigAlgebraicNumber> {
    auto dm = p.divide(*modulus_);  // modulus is monic, hence nonzero
    if (!dm) {
        return make_error<BigAlgebraicNumber>(dm.error());
    }
    return BigAlgebraicNumber{*this, std::move(dm->remainder)};
}

auto BigNumberField::generator() const -> Result<BigAlgebraicNumber> {
    return from_poly(BigRationalPoly::monomial(BigRational::from_int(1), 1));  // x mod m
}

// --- BigAlgebraicNumber -----------------------------------------------------

auto BigAlgebraicNumber::add(const BigAlgebraicNumber& o) const -> Result<BigAlgebraicNumber> {
    if (!field_.is_same(o.field_)) {
        return make_error<BigAlgebraicNumber>(MathError::domain_error);
    }
    // Both degree < d => the sum is already reduced (infallible over BigRational).
    return BigAlgebraicNumber{field_, value_.add(o.value_)};
}

auto BigAlgebraicNumber::subtract(const BigAlgebraicNumber& o) const
    -> Result<BigAlgebraicNumber> {
    if (!field_.is_same(o.field_)) {
        return make_error<BigAlgebraicNumber>(MathError::domain_error);
    }
    return BigAlgebraicNumber{field_, value_.subtract(o.value_)};  // degree < d, reduced
}

auto BigAlgebraicNumber::negate() const -> Result<BigAlgebraicNumber> {
    return BigAlgebraicNumber{field_, value_.negate()};  // infallible; Result for parity
}

auto BigAlgebraicNumber::scale(const BigRational& s) const -> Result<BigAlgebraicNumber> {
    return BigAlgebraicNumber{field_, value_.scale(s)};  // degree unchanged, still reduced
}

auto BigAlgebraicNumber::multiply(const BigAlgebraicNumber& o) const
    -> Result<BigAlgebraicNumber> {
    if (!field_.is_same(o.field_)) {
        return make_error<BigAlgebraicNumber>(MathError::domain_error);
    }
    BigRationalPoly prod = value_.multiply(o.value_);   // degree up to 2d-2 (infallible)
    auto dm = prod.divide(field_.modulus());            // reduce mod m
    if (!dm) {
        return make_error<BigAlgebraicNumber>(dm.error());
    }
    return BigAlgebraicNumber{field_, std::move(dm->remainder)};
}

auto BigAlgebraicNumber::inverse() const -> Result<BigAlgebraicNumber> {
    if (is_zero()) {
        return make_error<BigAlgebraicNumber>(MathError::division_by_zero);
    }
    // u*value_ + v*m = g. When m is irreducible g is a nonzero constant unit and
    // value_^{-1} = (u / g) mod m. When m is reducible and value_ shares a factor with it,
    // deg g >= 1 (not a unit): the element is a zero divisor, so it has no inverse.
    auto eg = ext_gcd(value_, field_.modulus());
    if (!eg) {
        return make_error<BigAlgebraicNumber>(eg.error());
    }
    if (eg->g.degree() != 0) {  // non-unit gcd => not invertible (reducible modulus)
        return make_error<BigAlgebraicNumber>(MathError::division_by_zero);
    }
    const BigRational g_const = eg->g.coefficient(0);  // g has degree 0 here
    if (g_const.is_zero()) {  // unreachable for a nonzero element; honest guard regardless
        return make_error<BigAlgebraicNumber>(MathError::division_by_zero);
    }
    auto inv_g = g_const.reciprocal();  // g_const != 0
    if (!inv_g) {
        return make_error<BigAlgebraicNumber>(inv_g.error());
    }
    return field_.from_poly(eg->u.scale(*inv_g));  // (u / g) reduced mod m
}

auto BigAlgebraicNumber::divide(const BigAlgebraicNumber& o) const
    -> Result<BigAlgebraicNumber> {
    if (!field_.is_same(o.field_)) {
        return make_error<BigAlgebraicNumber>(MathError::domain_error);
    }
    auto inv = o.inverse();  // division_by_zero when o has no inverse
    if (!inv) {
        return make_error<BigAlgebraicNumber>(inv.error());
    }
    return multiply(*inv);
}

auto BigAlgebraicNumber::pow(std::int64_t exponent) const -> Result<BigAlgebraicNumber> {
    if (exponent < 0) {
        return make_error<BigAlgebraicNumber>(MathError::domain_error);
    }
    BigAlgebraicNumber result = field_.one();
    BigAlgebraicNumber base = *this;
    std::int64_t e = exponent;
    while (e > 0) {
        if ((e & 1) != 0) {
            auto r = result.multiply(base);
            if (!r) {
                return make_error<BigAlgebraicNumber>(r.error());
            }
            result = std::move(*r);
        }
        e >>= 1;
        if (e > 0) {
            auto b = base.multiply(base);
            if (!b) {
                return make_error<BigAlgebraicNumber>(b.error());
            }
            base = std::move(*b);
        }
    }
    return result;
}

auto BigAlgebraicNumber::mult_matrix() const
    -> Result<std::vector<std::vector<BigRational>>> {
    const std::int64_t d = field_.degree();  // >= 1
    const std::size_t dd = static_cast<std::size_t>(d);
    const BigRationalPoly x = BigRationalPoly::monomial(BigRational::from_int(1), 1);
    std::vector<std::vector<BigRational>> rows(dd, std::vector<BigRational>(dd));
    BigRationalPoly current = value_;  // (*this) * alpha^0, already reduced
    for (std::size_t j = 0; j < dd; ++j) {
        for (std::size_t i = 0; i < dd; ++i) {
            rows[i][j] = current.coefficient(i);  // coordinate of alpha^i in current
        }
        if (j + 1 < dd) {
            BigRationalPoly prod = current.multiply(x);  // multiply by alpha (infallible)
            auto dm = prod.divide(field_.modulus());     // then reduce mod m
            if (!dm) {
                return make_error<std::vector<std::vector<BigRational>>>(dm.error());
            }
            current = std::move(dm->remainder);
        }
    }
    return rows;
}

auto BigAlgebraicNumber::norm() const -> Result<BigRational> {
    auto m = mult_matrix();
    if (!m) {
        return make_error<BigRational>(m.error());
    }
    return determinant(std::move(*m));
}

auto BigAlgebraicNumber::trace() const -> Result<BigRational> {
    auto m = mult_matrix();
    if (!m) {
        return make_error<BigRational>(m.error());
    }
    BigRational acc;  // canonical zero
    for (std::size_t i = 0; i < m->size(); ++i) {
        acc = acc.add((*m)[i][i]);  // diagonal sum
    }
    return acc;
}

}  // namespace nimblecas
