// NimbleCAS exact arithmetic in a simple algebraic number field Q(alpha) (§7 algebra
// substrate; groundwork for a Jordan-canonical-form module).
// @author Olumuyiwa Oluwasanmi
//
// A number field here is a simple algebraic extension Q(alpha) = Q[x]/(m(x)), where m is
// a MONIC IRREDUCIBLE polynomial over Q of degree d >= 1. It lets us represent an
// irrational or complex ALGEBRAIC number exactly, as a residue class of RationalPoly:
// sqrt(2) is the class of x in Q[x]/(x^2 - 2), i is the class of x in Q[x]/(x^2 + 1),
// cbrt(2) is the class of x in Q[x]/(x^3 - 2). Every element is the exact rational-
// coefficient polynomial it mathematically is — never a floating-point stand-in — so a
// downstream eigenvalue / canonical-form computation over these fields is exact.
//
// REPRESENTATION. A NumberField holds its monic irreducible minimal polynomial m. An
// AlgebraicNumber (an element of the field) holds that field together with its CANONICAL
// residue: a RationalPoly of degree < d, the unique remainder of any representative mod m.
// Because {1, alpha, ..., alpha^(d-1)} is a Q-basis of the field, this residue is a normal
// form and equality is a plain coefficient-vector compare.
//
// ARITHMETIC. Add / subtract / negate are componentwise on the residue (a sum of two
// degree-<d polynomials is already reduced). Multiply is a polynomial product reduced mod m.
// Inverse of a nonzero element uses the extended Euclidean algorithm in Q[x]: since m is
// irreducible and 0 != a has deg a < d, gcd(a, m) is a nonzero constant g with
// u*a + v*m = g, whence a^{-1} = (u / g) mod m. Division is a * b^{-1}; pow is
// non-negative-exponent repeated squaring. The field norm and trace of an element are the
// exact determinant and trace of its multiplication-by map on the basis {1, ..., alpha^(d-1)}.
//
// HONESTY (Rule 32). Every fallible operation returns Result<T>; nothing throws and nothing
// returns a plausible-but-wrong value. Field construction VERIFIES irreducibility with
// factor_over_Q: a reducible or constant/zero m is MathError::domain_error, so a NumberField
// that exists is always an honest field. Inverting or dividing by the zero element is
// MathError::division_by_zero; combining elements of different fields is domain_error; a
// negative power is domain_error. All exact int64 rational arithmetic is overflow-checked and
// surfaces MathError::overflow rather than wrapping (inherited from Rational / RationalPoly).

module;
#include <cassert>

export module nimblecas.algnum;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.factor;
import nimblecas.matrix;

export namespace nimblecas {

class AlgebraicNumber;  // element of a NumberField; defined below.

// ---------------------------------------------------------------------------
// NumberField — a simple algebraic extension Q(alpha) = Q[x]/(m(x)).
// ---------------------------------------------------------------------------
// Constructed from a monic irreducible minimal polynomial m of degree d >= 1 (a non-monic
// m is normalised to monic; a reducible, constant, or zero m is rejected). The field acts
// as the factory for its elements and carries the modulus that their arithmetic reduces by.
// The modulus is held behind a CowPtr so copying a field (and thus every element that
// embeds it) is an O(1) refcount bump (Rule 22).
class NumberField {
public:
    // Build Q[x]/(m). The minimal polynomial is normalised to monic (dividing by its
    // leading coefficient — same ideal, same field). Fails with:
    //   * domain_error — m is the zero polynomial, a nonzero constant (degree 0), or
    //     REDUCIBLE over Q (verified via factor_over_Q: m must be a single irreducible
    //     factor of multiplicity 1);
    //   * overflow / not_implemented — propagated from the exact arithmetic or the bounded
    //     irreducibility search inside factor_over_Q.
    [[nodiscard]] static auto create(const RationalPoly& minimal) -> Result<NumberField>;

    // The degree d = deg m of the extension ([Q(alpha):Q]).
    [[nodiscard]] auto degree() const -> std::int64_t { return modulus_->degree(); }
    // The monic irreducible minimal polynomial m.
    [[nodiscard]] auto modulus() const -> const RationalPoly& { return *modulus_; }
    // Same field iff the minimal polynomials are identical.
    [[nodiscard]] auto is_same(const NumberField& o) const -> bool {
        return modulus_->is_equal(*o.modulus_);
    }

    // --- element factories ---
    [[nodiscard]] auto zero() const -> AlgebraicNumber;             // 0
    [[nodiscard]] auto one() const -> AlgebraicNumber;              // 1
    [[nodiscard]] auto from_rational(const Rational& c) const -> AlgebraicNumber;  // constant c
    // The generator alpha = x mod m. (For a degree-1 field Q[x]/(x - c) this is the
    // constant c, since x == c there.) Fails only on overflow.
    [[nodiscard]] auto generator() const -> Result<AlgebraicNumber>;
    // The class of an arbitrary p in Q[x], i.e. p reduced mod m. Fails only on overflow.
    [[nodiscard]] auto from_poly(const RationalPoly& p) const -> Result<AlgebraicNumber>;

    [[nodiscard]] auto to_string(std::string_view var = "a") const -> std::string {
        return std::format("Q[{}]/({})", var, modulus_->to_string(var));
    }

private:
    explicit NumberField(CowPtr<RationalPoly> m) : modulus_(std::move(m)) {}
    CowPtr<RationalPoly> modulus_;
};

// ---------------------------------------------------------------------------
// AlgebraicNumber — an element of a NumberField, in canonical residue form.
// ---------------------------------------------------------------------------
// Holds its field and the canonical residue (a RationalPoly of degree < d). Binary
// operations require both operands to live in the SAME field (equal minimal polynomials),
// else MathError::domain_error.
class AlgebraicNumber {
public:
    [[nodiscard]] auto field() const -> const NumberField& { return field_; }
    // The canonical residue: coefficients of 1, alpha, ..., alpha^(d-1) (degree < d).
    [[nodiscard]] auto value() const -> const RationalPoly& { return value_; }

    [[nodiscard]] auto is_zero() const -> bool { return value_.is_zero(); }
    [[nodiscard]] auto is_one() const -> bool {
        return value_.is_equal(RationalPoly::constant(Rational::from_int(1)));
    }
    // Equality: same field AND same residue.
    [[nodiscard]] auto is_equal(const AlgebraicNumber& o) const -> bool {
        return field_.is_same(o.field_) && value_.is_equal(o.value_);
    }

    // --- field operations (all exact; overflow-checked) ---
    [[nodiscard]] auto add(const AlgebraicNumber& o) const -> Result<AlgebraicNumber>;
    [[nodiscard]] auto subtract(const AlgebraicNumber& o) const -> Result<AlgebraicNumber>;
    [[nodiscard]] auto negate() const -> Result<AlgebraicNumber>;
    [[nodiscard]] auto multiply(const AlgebraicNumber& o) const -> Result<AlgebraicNumber>;
    // Multiplicative inverse via the extended Euclidean algorithm in Q[x]. The zero
    // element has no inverse: MathError::division_by_zero.
    [[nodiscard]] auto inverse() const -> Result<AlgebraicNumber>;
    // a / b = a * b^{-1}; division_by_zero when b == 0.
    [[nodiscard]] auto divide(const AlgebraicNumber& o) const -> Result<AlgebraicNumber>;
    // Non-negative integer power by repeated squaring; pow(0) == 1. A negative exponent
    // is MathError::domain_error (use inverse() first for negative powers).
    [[nodiscard]] auto pow(std::int64_t exponent) const -> Result<AlgebraicNumber>;

    // The field norm N(a) = det, and trace Tr(a) = trace, of the Q-linear
    // multiplication-by-a map on the basis {1, alpha, ..., alpha^(d-1)}. Both are exact
    // rationals (for a in Q(alpha) of degree d over Q, N and Tr equal the constant term and
    // the negated subleading coefficient of the characteristic polynomial of that map).
    [[nodiscard]] auto norm() const -> Result<Rational>;
    [[nodiscard]] auto trace() const -> Result<Rational>;

    [[nodiscard]] auto to_string(std::string_view var = "a") const -> std::string {
        return value_.to_string(var);
    }

private:
    friend class NumberField;
    AlgebraicNumber(NumberField f, RationalPoly v)
        : field_(std::move(f)), value_(std::move(v)) {}

    // The d x d rational matrix of multiplication-by-*this on {1, alpha, ..., alpha^(d-1)}:
    // column j holds the residue coordinates of (*this) * alpha^j.
    [[nodiscard]] auto mult_matrix() const -> Result<Matrix>;

    NumberField field_;
    RationalPoly value_;  // canonical residue, degree < d
};

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// Extended Euclidean algorithm in Q[x]: returns (g, u, v) with u*a + v*b == g, where g is
// a greatest common divisor of a and b (not normalised to monic — the raw Bezout g is what
// the inverse routine needs, since it divides through by g's own leading value). Fails only
// on overflow, propagated from the underlying RationalPoly arithmetic.
struct ExtGcd {
    RationalPoly g;
    RationalPoly u;
    RationalPoly v;
};

[[nodiscard]] auto ext_gcd(const RationalPoly& a, const RationalPoly& b) -> Result<ExtGcd> {
    RationalPoly r0 = a;
    RationalPoly r1 = b;
    RationalPoly s0 = RationalPoly::constant(Rational::from_int(1));
    RationalPoly s1;  // zero
    RationalPoly t0;  // zero
    RationalPoly t1 = RationalPoly::constant(Rational::from_int(1));
    while (!r1.is_zero()) {
        auto dm = r0.divide(r1);  // r1 != 0
        if (!dm) {
            return make_error<ExtGcd>(dm.error());
        }
        const RationalPoly& q = dm->quotient;
        // (s0, s1) <- (s1, s0 - q*s1); (t0, t1) <- (t1, t0 - q*t1); (r0, r1) <- (r1, rem).
        auto qs = q.multiply(s1);
        if (!qs) {
            return make_error<ExtGcd>(qs.error());
        }
        auto s_next = s0.subtract(*qs);
        if (!s_next) {
            return make_error<ExtGcd>(s_next.error());
        }
        auto qt = q.multiply(t1);
        if (!qt) {
            return make_error<ExtGcd>(qt.error());
        }
        auto t_next = t0.subtract(*qt);
        if (!t_next) {
            return make_error<ExtGcd>(t_next.error());
        }
        s0 = std::move(s1);
        s1 = std::move(*s_next);
        t0 = std::move(t1);
        t1 = std::move(*t_next);
        r0 = std::move(r1);
        r1 = std::move(dm->remainder);
    }
    return ExtGcd{.g = std::move(r0), .u = std::move(s0), .v = std::move(t0)};
}

}  // namespace

// --- NumberField ------------------------------------------------------------

auto NumberField::create(const RationalPoly& minimal) -> Result<NumberField> {
    if (minimal.is_zero() || minimal.degree() < 1) {
        // The zero polynomial and nonzero constants define no proper extension.
        return make_error<NumberField>(MathError::domain_error);
    }
    // Normalise to monic (same ideal, hence same field).
    auto monic = minimal.monic();
    if (!monic) {
        return make_error<NumberField>(monic.error());
    }
    // Verify irreducibility over Q: the factorization of the primitive part must be a
    // single irreducible factor of multiplicity 1. A reducible m (e.g. x^2 - 1) yields
    // two factors and is rejected here, keeping the field honest.
    auto factors = factor_over_Q(*monic);
    if (!factors) {
        return make_error<NumberField>(factors.error());
    }
    if (factors->size() != 1 || factors->front().second != 1) {
        return make_error<NumberField>(MathError::domain_error);
    }
    return NumberField{CowPtr<RationalPoly>::make(std::move(*monic))};
}

auto NumberField::zero() const -> AlgebraicNumber {
    return AlgebraicNumber{*this, RationalPoly{}};
}

auto NumberField::one() const -> AlgebraicNumber {
    return AlgebraicNumber{*this, RationalPoly::constant(Rational::from_int(1))};
}

auto NumberField::from_rational(const Rational& c) const -> AlgebraicNumber {
    return AlgebraicNumber{*this, RationalPoly::constant(c)};  // degree 0 < d, already reduced
}

auto NumberField::from_poly(const RationalPoly& p) const -> Result<AlgebraicNumber> {
    auto dm = p.divide(*modulus_);  // modulus is monic, hence nonzero
    if (!dm) {
        return make_error<AlgebraicNumber>(dm.error());
    }
    return AlgebraicNumber{*this, std::move(dm->remainder)};
}

auto NumberField::generator() const -> Result<AlgebraicNumber> {
    return from_poly(RationalPoly::monomial(Rational::from_int(1), 1));  // x mod m
}

// --- AlgebraicNumber --------------------------------------------------------

auto AlgebraicNumber::add(const AlgebraicNumber& o) const -> Result<AlgebraicNumber> {
    if (!field_.is_same(o.field_)) {
        return make_error<AlgebraicNumber>(MathError::domain_error);
    }
    auto sum = value_.add(o.value_);  // both degree < d => sum degree < d, already reduced
    if (!sum) {
        return make_error<AlgebraicNumber>(sum.error());
    }
    return AlgebraicNumber{field_, std::move(*sum)};
}

auto AlgebraicNumber::subtract(const AlgebraicNumber& o) const -> Result<AlgebraicNumber> {
    if (!field_.is_same(o.field_)) {
        return make_error<AlgebraicNumber>(MathError::domain_error);
    }
    auto diff = value_.subtract(o.value_);  // degree < d, already reduced
    if (!diff) {
        return make_error<AlgebraicNumber>(diff.error());
    }
    return AlgebraicNumber{field_, std::move(*diff)};
}

auto AlgebraicNumber::negate() const -> Result<AlgebraicNumber> {
    auto neg = value_.scale(Rational::from_int(-1));
    if (!neg) {
        return make_error<AlgebraicNumber>(neg.error());
    }
    return AlgebraicNumber{field_, std::move(*neg)};
}

auto AlgebraicNumber::multiply(const AlgebraicNumber& o) const -> Result<AlgebraicNumber> {
    if (!field_.is_same(o.field_)) {
        return make_error<AlgebraicNumber>(MathError::domain_error);
    }
    auto prod = value_.multiply(o.value_);  // degree up to 2d-2
    if (!prod) {
        return make_error<AlgebraicNumber>(prod.error());
    }
    auto dm = prod->divide(field_.modulus());  // reduce mod m
    if (!dm) {
        return make_error<AlgebraicNumber>(dm.error());
    }
    return AlgebraicNumber{field_, std::move(dm->remainder)};
}

auto AlgebraicNumber::inverse() const -> Result<AlgebraicNumber> {
    if (is_zero()) {
        return make_error<AlgebraicNumber>(MathError::division_by_zero);
    }
    // gcd(value_, m) is a nonzero constant g (m irreducible, 0 != deg value_ < d), with
    // u*value_ + v*m = g. Then value_^{-1} = (u / g) reduced mod m.
    auto eg = ext_gcd(value_, field_.modulus());
    if (!eg) {
        return make_error<AlgebraicNumber>(eg.error());
    }
    const Rational g_const = eg->g.coefficient(0);  // g has degree 0 here
    assert(eg->g.degree() == 0 && !g_const.is_zero() &&
           "gcd(a, m) must be a nonzero constant for irreducible m and 0 != deg a < d");
    if (g_const.is_zero()) {  // unreachable for a valid field; honest guard regardless
        return make_error<AlgebraicNumber>(MathError::domain_error);
    }
    auto inv_g = Rational::from_int(1).divide(g_const);
    if (!inv_g) {
        return make_error<AlgebraicNumber>(inv_g.error());
    }
    auto u_scaled = eg->u.scale(*inv_g);  // u / g
    if (!u_scaled) {
        return make_error<AlgebraicNumber>(u_scaled.error());
    }
    return field_.from_poly(*u_scaled);  // reduce mod m
}

auto AlgebraicNumber::divide(const AlgebraicNumber& o) const -> Result<AlgebraicNumber> {
    if (!field_.is_same(o.field_)) {
        return make_error<AlgebraicNumber>(MathError::domain_error);
    }
    auto inv = o.inverse();  // division_by_zero when o == 0
    if (!inv) {
        return make_error<AlgebraicNumber>(inv.error());
    }
    return multiply(*inv);
}

auto AlgebraicNumber::pow(std::int64_t exponent) const -> Result<AlgebraicNumber> {
    if (exponent < 0) {
        return make_error<AlgebraicNumber>(MathError::domain_error);
    }
    AlgebraicNumber result = field_.one();
    AlgebraicNumber base = *this;
    std::int64_t e = exponent;
    while (e > 0) {
        if ((e & 1) != 0) {
            auto r = result.multiply(base);
            if (!r) {
                return make_error<AlgebraicNumber>(r.error());
            }
            result = std::move(*r);
        }
        e >>= 1;
        if (e > 0) {
            auto b = base.multiply(base);
            if (!b) {
                return make_error<AlgebraicNumber>(b.error());
            }
            base = std::move(*b);
        }
    }
    return result;
}

auto AlgebraicNumber::mult_matrix() const -> Result<Matrix> {
    const std::int64_t d = field_.degree();  // >= 1
    const std::size_t dd = static_cast<std::size_t>(d);
    const RationalPoly x = RationalPoly::monomial(Rational::from_int(1), 1);
    std::vector<std::vector<Rational>> rows(dd, std::vector<Rational>(dd));
    RationalPoly current = value_;  // (*this) * alpha^0, already reduced
    for (std::size_t j = 0; j < dd; ++j) {
        for (std::size_t i = 0; i < dd; ++i) {
            rows[i][j] = current.coefficient(i);  // coordinate of alpha^i in current
        }
        if (j + 1 < dd) {
            auto prod = current.multiply(x);  // multiply by alpha, then reduce mod m
            if (!prod) {
                return make_error<Matrix>(prod.error());
            }
            auto dm = prod->divide(field_.modulus());
            if (!dm) {
                return make_error<Matrix>(dm.error());
            }
            current = std::move(dm->remainder);
        }
    }
    return Matrix::from_rows(std::move(rows));
}

auto AlgebraicNumber::norm() const -> Result<Rational> {
    auto m = mult_matrix();
    if (!m) {
        return make_error<Rational>(m.error());
    }
    return m->determinant();
}

auto AlgebraicNumber::trace() const -> Result<Rational> {
    auto m = mult_matrix();
    if (!m) {
        return make_error<Rational>(m.error());
    }
    return m->trace();
}

}  // namespace nimblecas
