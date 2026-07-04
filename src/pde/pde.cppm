// NimbleCAS exact linear evolution PDEs via power series in time.
// @author Olumuyiwa Oluwasanmi
//
// A linear evolution PDE is
//
//     u_t = L[u],    u(x, 0) = phi(x),
//
// where L is a spatial differential operator with rational-polynomial coefficients
// acting on functions of x, and the initial datum phi is a polynomial in x. Its exact
// solution is the classical time power series (the Cauchy-Kovalevskaya construction)
//
//     u(x, t) = sum_{n>=0} (L^n[phi](x) / n!) t^n,
//
// so the coefficient of t^n is c_n(x) = L^n[phi] / n!, itself a polynomial in x. We build
// the c_n iteratively without ever forming a factorial: c_0 = phi and
//
//     c_n = L[c_{n-1}] * (1/n),
//
// which telescopes to L^n[phi] / n! exactly. The spatial polynomials are RationalPoly
// over Q[x]; L is supplied as a SpatialOperator (phi -> L[phi]) so the caller chooses the
// physics. Convenience builders assemble the common constant-coefficient operators from
// RationalPoly::derivative and scale: heat (L = a * d^2/dx^2), transport (L = c * d/dx),
// and advection-diffusion (L = c * d/dx + a * d^2/dx^2).
//
// HONESTY. This is EXACT over Q for polynomial initial data phi under a
// polynomial-coefficient L: every c_n is an exact RationalPoly and, because a
// constant-coefficient L strictly lowers the degree of a polynomial, the series
// TERMINATES (L^n[phi] = 0 once n exceeds the degree budget) — the truncated result is the
// closed-form solution. It is the whole-line / formal power-series construction: NO
// boundary conditions are imposed, and it is neither a boundary-value nor a Fourier
// solver. For non-polynomial phi (or an L that does not preserve polynomials) the same
// recurrence yields the exact Taylor coefficients up to the requested order, i.e. a
// truncated approximation whose error is bounded by the truncation order.
//
// Rule 32 railway: every RationalPoly / Rational Result error is propagated; order 0 and
// missing operators are MathError::domain_error.

export module nimblecas.pde;

import std;
import nimblecas.core;
import nimblecas.ratpoly;

export namespace nimblecas {

// The spatial operator L of the evolution PDE u_t = L[u]: it consumes a spatial
// polynomial p(x) and returns L[p](x) on the railway. L = phi -> phi'' models the heat
// equation, L = phi -> phi' the transport equation, and so on.
using SpatialOperator = std::function<Result<RationalPoly>(const RationalPoly&)>;

// Solve u_t = L[u], u(x,0) = phi, returning the time-series coefficients c_0..c_order,
// where c_n(x) is the coefficient of t^n. c_0 = phi and c_n = L[c_{n-1}] * (1/n), so the
// returned vector holds order+1 RationalPolys and equals L^n[phi]/n! term by term without
// forming any factorial. For polynomial phi under a polynomial-preserving L the trailing
// c_n are the zero polynomial (the series terminates). order == 0 or an empty operator is
// a domain_error; any error raised by L is propagated.
[[nodiscard]] auto solve_evolution_pde(SpatialOperator l, const RationalPoly& phi,
                                       std::size_t order) -> Result<std::vector<RationalPoly>>;

// Evaluate the truncated solution sum_n c_n(x) t^n exactly at the rational point (x, t),
// using Horner in t and exact polynomial evaluation of each c_n at x. The coefficient list
// must be non-empty (else domain_error); Rational overflow is propagated.
[[nodiscard]] auto evaluate(const std::vector<RationalPoly>& coeffs, const Rational& x,
                            const Rational& t) -> Result<Rational>;

// --- Convenience operator builders ------------------------------------------
// Heat / diffusion operator L[phi] = diffusivity * phi'' (u_t = a u_xx).
[[nodiscard]] auto heat_operator(Rational diffusivity) -> SpatialOperator;
// Transport / advection operator L[phi] = speed * phi' (u_t = c u_x).
[[nodiscard]] auto transport_operator(Rational speed) -> SpatialOperator;
// Combined advection-diffusion operator L[phi] = speed * phi' + diffusivity * phi''.
[[nodiscard]] auto advection_diffusion(Rational speed, Rational diffusivity) -> SpatialOperator;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

namespace {

// Exact evaluation of a RationalPoly at a rational point via Horner from the top degree
// down: acc <- acc * x + coeff_i. The zero polynomial (degree -1) yields 0.
[[nodiscard]] auto eval_poly(const RationalPoly& p, const Rational& x) -> Result<Rational> {
    Rational acc;  // 0
    for (std::int64_t i = p.degree(); i >= 0; --i) {
        auto scaled = acc.multiply(x);
        if (!scaled) {
            return make_error<Rational>(scaled.error());
        }
        auto sum = scaled->add(p.coefficient(static_cast<std::size_t>(i)));
        if (!sum) {
            return make_error<Rational>(sum.error());
        }
        acc = *sum;
    }
    return acc;
}

}  // namespace

auto solve_evolution_pde(SpatialOperator l, const RationalPoly& phi, std::size_t order)
    -> Result<std::vector<RationalPoly>> {
    using Coeffs = std::vector<RationalPoly>;
    if (order == 0 || !l) {
        return make_error<Coeffs>(MathError::domain_error);
    }
    // The step index n is cast to int64 for Rational::make below; guard the physically
    // unreachable case of an order beyond INT64_MAX so the cast can never wrap.
    if (order > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        return make_error<Coeffs>(MathError::overflow);
    }
    Coeffs coeffs;
    coeffs.reserve(order + 1);
    coeffs.push_back(phi);  // c_0 = phi
    for (std::size_t n = 1; n <= order; ++n) {
        auto applied = l(coeffs.back());  // L[c_{n-1}]
        if (!applied) {
            return make_error<Coeffs>(applied.error());
        }
        auto inv_n = Rational::make(1, static_cast<std::int64_t>(n));  // 1/n
        if (!inv_n) {
            return make_error<Coeffs>(inv_n.error());
        }
        auto c_n = applied->scale(*inv_n);  // c_n = L[c_{n-1}] / n
        if (!c_n) {
            return make_error<Coeffs>(c_n.error());
        }
        coeffs.push_back(std::move(*c_n));
    }
    return coeffs;
}

auto evaluate(const std::vector<RationalPoly>& coeffs, const Rational& x, const Rational& t)
    -> Result<Rational> {
    if (coeffs.empty()) {
        return make_error<Rational>(MathError::domain_error);
    }
    // Horner in t: acc <- acc * t + c_n(x), sweeping n from the top coefficient down.
    Rational acc;  // 0
    for (std::size_t idx = coeffs.size(); idx-- > 0;) {
        auto cn_x = eval_poly(coeffs[idx], x);
        if (!cn_x) {
            return make_error<Rational>(cn_x.error());
        }
        auto scaled = acc.multiply(t);
        if (!scaled) {
            return make_error<Rational>(scaled.error());
        }
        auto sum = scaled->add(*cn_x);
        if (!sum) {
            return make_error<Rational>(sum.error());
        }
        acc = *sum;
    }
    return acc;
}

auto heat_operator(Rational diffusivity) -> SpatialOperator {
    return [diffusivity](const RationalPoly& phi) -> Result<RationalPoly> {
        auto d1 = phi.derivative();
        if (!d1) {
            return make_error<RationalPoly>(d1.error());
        }
        auto d2 = d1->derivative();  // phi''
        if (!d2) {
            return make_error<RationalPoly>(d2.error());
        }
        return d2->scale(diffusivity);
    };
}

auto transport_operator(Rational speed) -> SpatialOperator {
    return [speed](const RationalPoly& phi) -> Result<RationalPoly> {
        auto d1 = phi.derivative();  // phi'
        if (!d1) {
            return make_error<RationalPoly>(d1.error());
        }
        return d1->scale(speed);
    };
}

auto advection_diffusion(Rational speed, Rational diffusivity) -> SpatialOperator {
    return [speed, diffusivity](const RationalPoly& phi) -> Result<RationalPoly> {
        auto d1 = phi.derivative();  // phi'
        if (!d1) {
            return make_error<RationalPoly>(d1.error());
        }
        auto advection = d1->scale(speed);  // speed * phi'
        if (!advection) {
            return make_error<RationalPoly>(advection.error());
        }
        auto d2 = d1->derivative();  // phi''
        if (!d2) {
            return make_error<RationalPoly>(d2.error());
        }
        auto diffusion = d2->scale(diffusivity);  // diffusivity * phi''
        if (!diffusion) {
            return make_error<RationalPoly>(diffusion.error());
        }
        return advection->add(*diffusion);
    };
}

}  // namespace nimblecas
