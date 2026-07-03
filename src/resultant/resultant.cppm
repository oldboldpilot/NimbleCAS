// NimbleCAS resultant and discriminant over the rationals Q[x] (ROADMAP 7.17).
// @author Olumuyiwa Oluwasanmi
//
// The resultant res(A, B) of two polynomials is the product of lc(A)^deg B, lc(B)^deg A
// and all pairwise differences of their roots; it vanishes exactly when A and B share a
// root (a common factor). It is the algebraic substrate for §7.17's subresultant PRS and
// multivariate GCD, for the discriminant, and for the Rothstein-Trager resultant that
// drives the logarithmic part of rational-function integration (§7.19).
//
// This module computes the resultant over the coefficient field Q via the Euclidean
// remainder sequence, using the recurrence
//
//     res(A, B) = (-1)^{deg A * deg B} * lc(B)^{deg A - deg R} * res(B, R),   R = A mod B,
//
// with the base case res(A, c) = c^{deg A} for a constant c. Every operation is exact and
// overflow-checked: an int64 coefficient limit surfaces as MathError::overflow (Rule 32).

export module nimblecas.resultant;

import std;
import nimblecas.core;
import nimblecas.ratpoly;

export namespace nimblecas {

// Resultant res(a, b) in Q. Zero when a and b share a factor (or either is the zero
// polynomial). res(constant, constant) is 1 (the empty product).
[[nodiscard]] auto resultant(const RationalPoly& a, const RationalPoly& b) -> Result<Rational>;

// Discriminant disc(a) = (-1)^{n(n-1)/2} / lc(a) * res(a, a'), n = deg a. It vanishes
// exactly when a has a repeated root. A constant or linear polynomial has discriminant 1.
[[nodiscard]] auto discriminant(const RationalPoly& a) -> Result<Rational>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// base^exp for exp >= 0, exact and overflow-checked.
[[nodiscard]] auto rat_pow(const Rational& base, std::int64_t exp) -> Result<Rational> {
    Rational acc = Rational::from_int(1);
    for (std::int64_t k = 0; k < exp; ++k) {
        auto next = acc.multiply(base);
        if (!next) {
            return next;
        }
        acc = *next;
    }
    return acc;
}

// True when both degrees are odd, i.e. when (-1)^{deg a * deg b} == -1.
[[nodiscard]] auto sign_is_negative(std::int64_t deg_a, std::int64_t deg_b) -> bool {
    return (deg_a & 1) != 0 && (deg_b & 1) != 0;
}

}  // namespace

auto resultant(const RationalPoly& a_in, const RationalPoly& b_in) -> Result<Rational> {
    if (a_in.is_zero() || b_in.is_zero()) {
        return Rational{};  // 0
    }
    RationalPoly a = a_in;
    RationalPoly b = b_in;
    Rational result = Rational::from_int(1);

    // Arrange deg a >= deg b, paying the (-1)^{deg a * deg b} swap sign once.
    if (a.degree() < b.degree()) {
        if (sign_is_negative(a.degree(), b.degree())) {
            auto neg = result.negate();
            if (!neg) {
                return neg;
            }
            result = *neg;
        }
        std::swap(a, b);
    }

    // Euclidean descent: fold each step's sign and lc(b)^{deg a - deg r} into `result`.
    while (b.degree() > 0) {
        auto dm = a.divide(b);
        if (!dm) {
            return make_error<Rational>(dm.error());
        }
        const RationalPoly r = std::move(dm->remainder);
        if (r.is_zero()) {
            return Rational{};  // common factor => resultant 0
        }
        if (sign_is_negative(a.degree(), b.degree())) {
            auto neg = result.negate();
            if (!neg) {
                return neg;
            }
            result = *neg;
        }
        auto factor = rat_pow(b.leading_coefficient(), a.degree() - r.degree());
        if (!factor) {
            return factor;
        }
        auto prod = result.multiply(*factor);
        if (!prod) {
            return prod;
        }
        result = *prod;
        a = std::move(b);
        b = r;
    }

    // Base case: b is now a non-zero constant c, and res(a, c) = c^{deg a}.
    auto tail = rat_pow(b.coefficient(0), a.degree());
    if (!tail) {
        return tail;
    }
    return result.multiply(*tail);
}

auto discriminant(const RationalPoly& a) -> Result<Rational> {
    const std::int64_t n = a.degree();
    if (n <= 1) {
        return Rational::from_int(1);  // constant / linear: no repeated roots by convention
    }
    auto deriv = a.derivative();
    if (!deriv) {
        return make_error<Rational>(deriv.error());
    }
    auto res = resultant(a, *deriv);
    if (!res) {
        return res;
    }
    // disc = (-1)^{n(n-1)/2} / lc(a) * res(a, a').
    auto quotient = res->divide(a.leading_coefficient());  // lc != 0 (deg a >= 2)
    if (!quotient) {
        return quotient;
    }
    // n(n-1)/2 is odd exactly when n === 2 or 3 (mod 4), i.e. (n & 2) != 0. Computing the
    // parity directly avoids forming n(n-1), so no overflow guard is needed.
    if ((n & 2) != 0) {
        return quotient->negate();
    }
    return *quotient;
}

}  // namespace nimblecas
