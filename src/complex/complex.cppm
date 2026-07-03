// NimbleCAS exact complex numbers over the rationals — Gaussian rationals (ROADMAP 7.1).
// @author Olumuyiwa Oluwasanmi
//
// A Complex here is a pair of exact Rationals (real, imaginary), so every field
// operation stays exact: no floating point ever enters. The subring Q + Qi is closed
// under +, -, *, and / (a field, the Gaussian rationals), which is exactly what a CAS
// needs when a computation temporarily crosses into the complex numbers but must return
// an exact answer.
//
// Following the rest of the engine, arithmetic is overflow-checked and railway-oriented
// (Rule 32): when an int64 numerator or denominator inside a Rational part would
// overflow, the operation returns MathError::overflow rather than wrapping. Division by
// the zero complex number returns MathError::division_by_zero.
//
// Intentionally omitted: the modulus |z| = sqrt(a^2 + b^2) and the argument arg(z) are
// irrational in general and have no exact Rational representation. What *is* exact is the
// squared modulus a^2 + b^2, provided here as norm_squared(). |z| and arg(z) belong to a
// later numeric/symbolic layer that can carry radicals or floating-point approximations.

export module nimblecas.complex;

import std;
import nimblecas.core;
import nimblecas.ratpoly;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// Complex — an exact complex number re + im*i with Rational parts.
// ---------------------------------------------------------------------------
// Both parts are canonical Rationals, so equality is a field-wise compare and the
// default-constructed value is the exact zero 0 + 0i.
class Complex {
public:
    Complex() = default;  // 0 + 0i

    // Assemble re + im*i. The parts are already canonical Rationals, so this cannot fail.
    [[nodiscard]] static auto make(Rational re, Rational im) -> Complex {
        Complex c;
        c.re_ = re;
        c.im_ = im;
        return c;
    }
    // A purely real number re + 0i.
    [[nodiscard]] static auto from_real(Rational re) -> Complex { return make(re, Rational{}); }
    // The integer v as v + 0i.
    [[nodiscard]] static auto from_int(std::int64_t v) -> Complex {
        return make(Rational::from_int(v), Rational{});
    }
    // The imaginary unit 0 + 1i.
    [[nodiscard]] static auto i() -> Complex { return make(Rational{}, Rational::from_int(1)); }

    [[nodiscard]] auto real() const noexcept -> Rational { return re_; }
    [[nodiscard]] auto imag() const noexcept -> Rational { return im_; }
    [[nodiscard]] auto is_real() const noexcept -> bool { return im_.is_zero(); }
    // A purely imaginary value (or zero): the real part vanishes.
    [[nodiscard]] auto is_imaginary() const noexcept -> bool { return re_.is_zero(); }
    [[nodiscard]] auto is_zero() const noexcept -> bool { return re_.is_zero() && im_.is_zero(); }

    [[nodiscard]] auto add(const Complex& o) const -> Result<Complex>;
    [[nodiscard]] auto subtract(const Complex& o) const -> Result<Complex>;
    // (a + bi)(c + di) = (ac - bd) + (ad + bc)i.
    [[nodiscard]] auto multiply(const Complex& o) const -> Result<Complex>;
    [[nodiscard]] auto negate() const -> Result<Complex>;
    // The complex conjugate a - bi. Overflow-free for canonical parts (negating the
    // imaginary part can only fail on an unreachable INT64_MIN), but returns Result to
    // stay uniform with the negate-based railway.
    [[nodiscard]] auto conjugate() const -> Result<Complex>;
    // The exact squared modulus |z|^2 = a^2 + b^2. |z| itself is irrational and omitted.
    [[nodiscard]] auto norm_squared() const -> Result<Rational>;
    // (a + bi)/(c + di) = ((ac + bd) + (bc - ad)i) / (c^2 + d^2).
    // Fails with division_by_zero when o == 0.
    [[nodiscard]] auto divide(const Complex& o) const -> Result<Complex>;
    // 1 / z; fails with division_by_zero when z == 0.
    [[nodiscard]] auto reciprocal() const -> Result<Complex>;

    [[nodiscard]] auto operator==(const Complex& o) const noexcept -> bool {
        return re_ == o.re_ && im_ == o.im_;
    }
    // e.g. "3 - 4i", "2", "5i", "0".
    [[nodiscard]] auto to_string() const -> std::string;

private:
    Rational re_{};  // 0/1
    Rational im_{};  // 0/1
};

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

auto Complex::add(const Complex& o) const -> Result<Complex> {
    auto re = re_.add(o.re_);
    if (!re) {
        return make_error<Complex>(re.error());
    }
    auto im = im_.add(o.im_);
    if (!im) {
        return make_error<Complex>(im.error());
    }
    return make(*re, *im);
}

auto Complex::subtract(const Complex& o) const -> Result<Complex> {
    auto re = re_.subtract(o.re_);
    if (!re) {
        return make_error<Complex>(re.error());
    }
    auto im = im_.subtract(o.im_);
    if (!im) {
        return make_error<Complex>(im.error());
    }
    return make(*re, *im);
}

auto Complex::multiply(const Complex& o) const -> Result<Complex> {
    // (a + bi)(c + di) = (ac - bd) + (ad + bc)i.
    auto ac = re_.multiply(o.re_);
    if (!ac) {
        return make_error<Complex>(ac.error());
    }
    auto bd = im_.multiply(o.im_);
    if (!bd) {
        return make_error<Complex>(bd.error());
    }
    auto ad = re_.multiply(o.im_);
    if (!ad) {
        return make_error<Complex>(ad.error());
    }
    auto bc = im_.multiply(o.re_);
    if (!bc) {
        return make_error<Complex>(bc.error());
    }
    auto re = ac->subtract(*bd);
    if (!re) {
        return make_error<Complex>(re.error());
    }
    auto im = ad->add(*bc);
    if (!im) {
        return make_error<Complex>(im.error());
    }
    return make(*re, *im);
}

auto Complex::negate() const -> Result<Complex> {
    auto re = re_.negate();
    if (!re) {
        return make_error<Complex>(re.error());
    }
    auto im = im_.negate();
    if (!im) {
        return make_error<Complex>(im.error());
    }
    return make(*re, *im);
}

auto Complex::conjugate() const -> Result<Complex> {
    auto im = im_.negate();
    if (!im) {
        return make_error<Complex>(im.error());
    }
    return make(re_, *im);
}

auto Complex::norm_squared() const -> Result<Rational> {
    auto a2 = re_.multiply(re_);
    if (!a2) {
        return make_error<Rational>(a2.error());
    }
    auto b2 = im_.multiply(im_);
    if (!b2) {
        return make_error<Rational>(b2.error());
    }
    return a2->add(*b2);
}

auto Complex::divide(const Complex& o) const -> Result<Complex> {
    if (o.is_zero()) {
        return make_error<Complex>(MathError::division_by_zero);
    }
    // Denominator c^2 + d^2 is a positive Rational (o != 0), so the two divides below
    // cannot themselves hit division_by_zero.
    auto denom = o.norm_squared();
    if (!denom) {
        return make_error<Complex>(denom.error());
    }
    auto ac = re_.multiply(o.re_);
    if (!ac) {
        return make_error<Complex>(ac.error());
    }
    auto bd = im_.multiply(o.im_);
    if (!bd) {
        return make_error<Complex>(bd.error());
    }
    auto bc = im_.multiply(o.re_);
    if (!bc) {
        return make_error<Complex>(bc.error());
    }
    auto ad = re_.multiply(o.im_);
    if (!ad) {
        return make_error<Complex>(ad.error());
    }
    auto re_num = ac->add(*bd);  // ac + bd
    if (!re_num) {
        return make_error<Complex>(re_num.error());
    }
    auto im_num = bc->subtract(*ad);  // bc - ad
    if (!im_num) {
        return make_error<Complex>(im_num.error());
    }
    auto re = re_num->divide(*denom);
    if (!re) {
        return make_error<Complex>(re.error());
    }
    auto im = im_num->divide(*denom);
    if (!im) {
        return make_error<Complex>(im.error());
    }
    return make(*re, *im);
}

auto Complex::reciprocal() const -> Result<Complex> {
    return from_int(1).divide(*this);  // propagates division_by_zero when *this == 0
}

auto Complex::to_string() const -> std::string {
    if (im_.is_zero()) {
        return re_.to_string();  // purely real (covers 0)
    }
    const std::int64_t num = im_.numerator();
    const std::int64_t den = im_.denominator();  // > 0 (canonical Rational)
    if (re_.is_zero()) {
        // Purely imaginary: keep the sign inline, collapsing +/-1 to a bare unit.
        if (den == 1 && num == 1) {
            return "i";
        }
        if (den == 1 && num == -1) {
            return "-i";
        }
        return im_.to_string() + "i";
    }
    // Both parts present: split the imaginary sign out as " + " / " - ".
    const bool negative = num < 0;
    const std::int64_t mag = negative ? -num : num;  // num != INT64_MIN (canonical)
    std::string term;
    if (den == 1 && mag == 1) {
        term = "i";
    } else if (den == 1) {
        term = std::format("{}i", mag);
    } else {
        term = std::format("{}/{}i", mag, den);
    }
    return std::format("{} {} {}", re_.to_string(), negative ? "-" : "+", term);
}

}  // namespace nimblecas
