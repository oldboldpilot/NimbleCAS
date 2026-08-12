// NimbleCAS resultant and discriminant over the unbounded rationals Q[x] (BigRational).
// @author Olumuyiwa Oluwasanmi
//
// This is the arbitrary-precision analogue of nimblecas.resultant. Where that module
// carries int64 Rational coefficients and surfaces a saturated numerator or denominator as
// MathError::overflow, this one carries BigRational coefficients and has NO such ceiling:
// it is the field-norm substrate for a bignum Trager splitting-field path, where the
// intermediate resultant values routinely exceed the int64 envelope.
//
// The resultant res(A, B) of two polynomials is the product of lc(A)^deg B, lc(B)^deg A
// and all pairwise differences of their roots; it vanishes exactly when A and B share a
// root (a common factor). This module computes it over the coefficient field Q via the
// Euclidean remainder sequence, using the recurrence
//
//     res(A, B) = (-1)^{deg A * deg B} * lc(B)^{deg A - deg R} * res(B, R),   R = A mod B,
//
// with the base case res(A, c) = c^{deg A} for a constant c. Because BigRational's
// magnitude-combining operations (add, subtract, multiply, negate) and the polynomial
// operations built on them are INFALLIBLE (arbitrary precision cannot overflow), the sign
// folding and the lc-power accumulation here never fail. The only fallible step is the
// Euclidean division, which can only report a zero divisor. MathError::overflow is NEVER
// produced — that is the entire point of the bignum tier.

export module nimblecas.bigresultant;

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.bigratpoly;

export namespace nimblecas {

// Resultant res(a, b) in Q. Zero when a and b share a factor (or either is the zero
// polynomial). res(constant, constant) is 1 (the empty product). Never overflows.
[[nodiscard]] auto resultant(const BigRationalPoly& a, const BigRationalPoly& b)
    -> Result<BigRational>;

// Discriminant disc(a) = (-1)^{n(n-1)/2} / lc(a) * res(a, a'), n = deg a. It vanishes
// exactly when a has a repeated root. A constant or linear polynomial has discriminant 1.
[[nodiscard]] auto discriminant(const BigRationalPoly& a) -> Result<BigRational>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// base^exp for exp >= 0, exact. Infallible: BigRational::multiply only combines magnitudes
// and cannot overflow, so no MathError channel is ever needed (mirrors rat_pow's loop, but
// over the unbounded tier there is nothing to check).
[[nodiscard]] auto big_pow(const BigRational& base, std::int64_t exp) -> BigRational {
    BigRational acc = BigRational::from_int(1);
    for (std::int64_t k = 0; k < exp; ++k) {
        acc = acc.multiply(base);
    }
    return acc;
}

// True when both degrees are odd, i.e. when (-1)^{deg a * deg b} == -1.
[[nodiscard]] auto sign_is_negative(std::int64_t deg_a, std::int64_t deg_b) -> bool {
    return (deg_a & 1) != 0 && (deg_b & 1) != 0;
}

}  // namespace

auto resultant(const BigRationalPoly& a_in, const BigRationalPoly& b_in) -> Result<BigRational> {
    if (a_in.is_zero() || b_in.is_zero()) {
        return BigRational{};  // 0
    }
    BigRationalPoly a = a_in;
    BigRationalPoly b = b_in;
    BigRational result = BigRational::from_int(1);

    // Arrange deg a >= deg b, paying the (-1)^{deg a * deg b} swap sign once.
    if (a.degree() < b.degree()) {
        if (sign_is_negative(a.degree(), b.degree())) {
            result = result.negate();  // infallible over BigRational
        }
        std::swap(a, b);
    }

    // Euclidean descent: fold each step's sign and lc(b)^{deg a - deg r} into `result`.
    while (b.degree() > 0) {
        auto dm = a.divide(b);  // only fails on a zero divisor, ruled out (deg b > 0)
        if (!dm) {
            return make_error<BigRational>(dm.error());
        }
        const BigRationalPoly r = std::move(dm->remainder);
        if (r.is_zero()) {
            return BigRational{};  // common factor => resultant 0
        }
        if (sign_is_negative(a.degree(), b.degree())) {
            result = result.negate();
        }
        const BigRational factor = big_pow(b.leading_coefficient(), a.degree() - r.degree());
        result = result.multiply(factor);
        a = std::move(b);
        b = r;
    }

    // Base case: b is now a non-zero constant c, and res(a, c) = c^{deg a}.
    const BigRational tail = big_pow(b.coefficient(0), a.degree());
    return result.multiply(tail);
}

auto discriminant(const BigRationalPoly& a) -> Result<BigRational> {
    const std::int64_t n = a.degree();
    if (n <= 1) {
        return BigRational::from_int(1);  // constant / linear: no repeated roots by convention
    }
    const BigRationalPoly deriv = a.derivative();  // infallible over BigRational
    auto res = resultant(a, deriv);
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
        return quotient->negate();  // infallible over BigRational
    }
    return *quotient;
}

}  // namespace nimblecas
