// NimbleCAS linear homogeneous constant-coefficient recurrences (ROADMAP 7.9).
// @author Olumuyiwa Oluwasanmi
//
// A first slice of difference equations / recurrence relations. Given a linear
// homogeneous recurrence with constant rational coefficients
//
//     a_n = c_0 a_{n-1} + c_1 a_{n-2} + ... + c_{k-1} a_{n-k}   (k = order),
//
// its behaviour is governed by the characteristic polynomial
//
//     x^k - c_0 x^{k-1} - c_1 x^{k-2} - ... - c_{k-1}
//
// (monic, degree k). Each distinct root r of this polynomial, with multiplicity m,
// contributes a family of basis solutions r^n, n r^n, ..., n^{m-1} r^n to the general
// closed form a_n = sum over roots of (polynomial of degree m-1 in n) * r^n; the
// coefficients of those polynomials are fixed by the initial conditions.
//
// SCOPE: only the RATIONAL-characteristic-root case is fully resolved here. The rational
// roots (with multiplicities) are found exactly via nimblecas.roots::rational_roots on
// the characteristic polynomial, and all_roots_rational() reports whether the polynomial
// splits completely over Q (so the closed form is expressible with rational bases alone).
// When it does not — the Fibonacci recurrence x^2 - x - 1 is the canonical example, whose
// roots are the irrational golden-ratio conjugates — the closed form requires irrational
// or complex roots. Radical / RootOf closed forms for that case are a planned extension
// (mirroring the same documented limitation in nimblecas.roots) and are not produced here;
// characteristic_roots() then returns only the rational roots (possibly none).
//
// Following the rest of the engine, arithmetic is exact and overflow-checked and every
// fallible step is threaded through Result (Rule 32).

module;
#include <cassert>

export module nimblecas.recurrence;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.roots;

export namespace nimblecas {

// The characteristic polynomial x^k - c_0 x^{k-1} - c_1 x^{k-2} - ... - c_{k-1} of the
// order-k recurrence a_n = c_0 a_{n-1} + ... + c_{k-1} a_{n-k}, returned as a monic
// RationalPoly of degree k. The coefficients are given low-order first, so coeffs[0] is
// c_0 (the a_{n-1} weight) and coeffs.back() is c_{k-1} (the a_{n-k} weight). Empty
// coeffs describe no recurrence and are rejected with MathError::domain_error. Fails only
// on overflow of the exact rational negation.
[[nodiscard]] auto characteristic_polynomial(std::span<const Rational> coeffs)
    -> Result<RationalPoly>;

// The distinct RATIONAL characteristic roots of the recurrence, each paired with its
// multiplicity (>= 1), obtained by running rational_roots() on the characteristic
// polynomial. A distinct rational root r of multiplicity m contributes the basis terms
// r^n, n r^n, ..., n^{m-1} r^n to the general solution. Roots are returned in no
// particular order. Irrational / complex roots are NOT returned (see the module header on
// scope): for a recurrence whose characteristic polynomial does not split over Q this
// yields only the rational part, possibly the empty vector. Empty coeffs ->
// MathError::domain_error.
[[nodiscard]] auto characteristic_roots(std::span<const Rational> coeffs)
    -> Result<std::vector<std::pair<Rational, std::int64_t>>>;

// Whether the characteristic polynomial splits completely over Q — i.e. whether the sum
// of the rational-root multiplicities equals the order k. When true, the closed form is
// fully expressible with rational bases and characteristic_roots() accounts for every
// root. When false, the remaining roots are irrational or complex and the closed form
// needs radical / RootOf machinery that is a planned extension. Empty coeffs ->
// MathError::domain_error.
[[nodiscard]] auto all_roots_rational(std::span<const Rational> coeffs) -> Result<bool>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

auto characteristic_polynomial(std::span<const Rational> coeffs) -> Result<RationalPoly> {
    if (coeffs.empty()) {
        return make_error<RationalPoly>(MathError::domain_error);  // no recurrence
    }
    const std::size_t k = coeffs.size();
    // Build x^k - c_0 x^{k-1} - ... - c_{k-1} low-order first: the x^k term is monic (1),
    // and the coefficient of x^(k-1-i) is -c_i, i.e. coeff[k-1-i] = -coeffs[i].
    std::vector<Rational> out(k + 1);
    out[k] = Rational::from_int(1);
    for (std::size_t i = 0; i < k; ++i) {
        auto neg = coeffs[i].negate();
        if (!neg) {
            return make_error<RationalPoly>(neg.error());
        }
        out[k - 1 - i] = *neg;
    }
    return RationalPoly::from_coeffs(std::move(out));
}

auto characteristic_roots(std::span<const Rational> coeffs)
    -> Result<std::vector<std::pair<Rational, std::int64_t>>> {
    using Roots = std::vector<std::pair<Rational, std::int64_t>>;
    auto poly = characteristic_polynomial(coeffs);
    if (!poly) {
        return make_error<Roots>(poly.error());
    }
    return rational_roots(*poly);  // rational roots with multiplicities; monic, non-zero
}

auto all_roots_rational(std::span<const Rational> coeffs) -> Result<bool> {
    auto roots = characteristic_roots(coeffs);
    if (!roots) {
        return make_error<bool>(roots.error());
    }
    // The characteristic polynomial has degree k == coeffs.size(); it splits over Q iff
    // the rational-root multiplicities account for all k of its roots.
    std::int64_t total = 0;
    for (const auto& [root, mult] : *roots) {
        total += mult;  // each mult >= 1, at most k roots, so the sum cannot overflow int64
    }
    return total == static_cast<std::int64_t>(coeffs.size());
}

}  // namespace nimblecas
