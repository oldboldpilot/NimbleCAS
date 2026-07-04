// NimbleCAS semi-analytical perturbation methods for ODEs — ADM, HPM, HAM (exact power series).
// @author Olumuyiwa Oluwasanmi
//
// This module implements three classical semi-analytical schemes for the first-order
// initial value problem
//
//     u'(x) = f(u),    u(0) = u0   (u0 a Rational constant),
//
//   * ADM — Adomian Decomposition Method,
//   * HPM — Homotopy Perturbation Method,
//   * HAM — Homotopy Analysis Method (with convergence-control parameter ħ),
//
// entirely as EXACT arithmetic over Q in the truncated ring Q[[x]]/(x^N). The engine is
// nimblecas.powerseries: x is the independent variable = PowerSeries::variable(), the
// linear operator L = d/dx is PowerSeries::derivative(), and its exact right inverse
// L^{-1} (definite integral from 0, zero constant of integration) is
// PowerSeries::integrate(). Rewriting the IVP as the fixed point
//
//     u = u0 + L^{-1}[ f(u) ]
//
// turns the whole solution into exact truncated power-series arithmetic.
//
// The nonlinear operator f is supplied as a SeriesOperator: given the current solution
// series u it returns f(u) as a series. So f = identity gives u' = u; f = u->u^2 gives
// u' = u^2; f = u->1 + u^2 gives the Riccati equation u' = 1 + u^2. f may use any of the
// powerseries operations (products, inverse/divide, exp/log/compose), so polynomial,
// rational and (truncated) analytic nonlinearities are all in scope.
//
// GRADING. Because u0 is a constant and each Picard update integrates once, the ADM/HPM
// components u_0, u_1, u_2, ... produced here are HOMOGENEOUS: u_i has its only nonzero
// term at x^i (u_0 = u0 x^0, u_1 = t_1 x^1, ...). The decomposition grade i therefore
// coincides exactly with the x-degree i. Under this identification the m-th Adomian
// polynomial A_m — the grade-m contribution to N(u_0 + u_1 + ...) — is precisely the
// x-degree-m graded projection Π_m of N applied to the partial sum:
//
//     A_m = Π_m [ N(u_0 + u_1 + ... + u_m) ]      (place [x^m] N(·) back at x^m).
//
// Note this is the projection of N of the TOTAL partial sum, NOT the naive difference
// N(Σ_{i<=m}) − N(Σ_{i<m}): that difference double-drops same-grade cross terms. For
// N(u) = u^2 with components u_0, u_1, u_2 the difference at level 2 would miss the u_1^2
// piece (it is "new at level 2" yet built only from u_1); the graded projection of the
// total correctly returns A_2 = 2 u_0 u_2 + u_1^2. Components of index > m are of degree
// > m and cannot affect [x^m], so projecting N of the full sum is exact and cheapest.
// This is exact for polynomial nonlinearities and, through the exact exp/log/compose
// coefficient recurrences of the powerseries engine, exact within the truncation for the
// Maclaurin expansion of analytic f.
//
// EQUIVALENCE. ADM (u_{n+1} = L^{-1}[A_n]) and HPM (embed v = Σ p^n v_n in the homotopy
// (1−p)(v' − u0') + p(v' − f(v)) = 0, which collapses to v' = p f(v); then v_0 = u0 and
// v_n' = A_{n-1}) are two organisations of the SAME graded Picard/Taylor iteration and
// return the identical series. HAM's m-th order deformation is
//
//     u_m = χ_m u_{m−1} + ħ · L^{-1}[R_m],   R_m = u_{m−1}' − A_{m−1},
//     χ_m = 0 (m = 1), 1 (m ≥ 2),
//
// where ħ is an extra degree of freedom (the convergence-control parameter) that HPM/ADM
// lack. At ħ = −1 the u_{m−1}' term cancels u_{m−1} (the components are homogeneous of
// degree ≥ 1, so L^{-1}[u_{m−1}'] = u_{m−1}) and u_m reduces to L^{-1}[A_{m−1}]: HAM(ħ=−1)
// recovers ADM/HPM exactly.
//
// HONESTY. Everything here is EXACT over Q as a power series TRUNCATED to the requested
// order. For polynomial/rational f these are the exact Taylor coefficients of the
// solution to that order. The methods make NO claim about the radius of convergence or
// about any closed-form / transcendental solution; ħ ≠ −1 selects a different (still
// exactly rational) member of the HAM deformation family rather than the Taylor series.
// Rule 32 railway: every powerseries Result error is propagated; bad order/arguments are
// MathError::domain_error.

export module nimblecas.perturbation;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.powerseries;

export namespace nimblecas {

// The nonlinear operator f of the IVP u' = f(u): consumes the current solution series
// and returns f(u) as a series (on the railway). f = identity models u' = u, etc.
using SeriesOperator = std::function<Result<PowerSeries>(const PowerSeries&)>;

// Adomian polynomials A_0..A_k of the operator N for the components u_0..u_k (k =
// components.size() − 1). A_m is the grade-m contribution to N(u_0 + ... + u_k), returned
// as the homogeneous series ([x^m] N(Σ u_i)) x^m — the x-degree-m graded projection of N
// of the partial sum (see the module header for why this, and not the raw difference of
// partial sums, is the exact operator). All components must share the same order; the
// returned polynomials share that order. Exact for polynomial nonlinearities.
//
// Failure modes (MathError::domain_error): empty component list, an empty operator, or
// components of differing order; f preserving order is required (else domain_error). Any
// error raised by N is propagated.
[[nodiscard]] auto adomian_polynomials(SeriesOperator N,
                                       const std::vector<PowerSeries>& components)
    -> Result<std::vector<PowerSeries>>;

// ADM solution of u' = f(u), u(0) = u0, as a power series with `order` coefficients
// (terms x^0..x^{order-1}). u_0 = u0; u_{n+1} = L^{-1}[A_n] with A_n the Adomian
// polynomials of the components so far; the result is Σ u_n. This is the exact Taylor
// polynomial of the solution to the requested order. order == 0 is a domain_error.
[[nodiscard]] auto adm_solve(SeriesOperator f, Rational u0, std::size_t order)
    -> Result<PowerSeries>;

// HPM solution of the same problem. The homotopy (1−p)(v'−u0') + p(v'−f(v)) = 0 with
// v = Σ p^n v_n gives v_0 = u0 and v_n' = A_{n−1}, integrated term by term. This is the
// same graded recursion as adm_solve and returns the IDENTICAL series. order == 0 is a
// domain_error.
[[nodiscard]] auto hpm_solve(SeriesOperator f, Rational u0, std::size_t order)
    -> Result<PowerSeries>;

// HAM solution with convergence-control parameter ħ. The m-th order deformation is
// u_m = χ_m u_{m−1} + ħ L^{-1}[R_m], R_m = u_{m−1}' − A_{m−1}, χ_m = 0 (m=1) else 1; the
// result is Σ u_m to `order` coefficients. ħ = −1 recovers HPM/ADM exactly. ħ is the
// extra degree of freedom HAM adds over HPM. order == 0 is a domain_error.
[[nodiscard]] auto ham_solve(SeriesOperator f, Rational u0, Rational hbar, std::size_t order)
    -> Result<PowerSeries>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

namespace {

// Coefficient-wise sum of the components, all required to share `order`.
auto sum_components(const std::vector<PowerSeries>& comps, std::size_t order)
    -> Result<PowerSeries> {
    auto acc = PowerSeries::zero(order);
    if (!acc) {
        return make_error<PowerSeries>(acc.error());
    }
    PowerSeries s = *acc;
    for (const auto& c : comps) {
        if (c.order() != order) {
            return make_error<PowerSeries>(MathError::domain_error);
        }
        auto t = s.add(c);
        if (!t) {
            return make_error<PowerSeries>(t.error());
        }
        s = *t;
    }
    return s;
}

// The homogeneous monomial c * x^degree in the order-`order` ring (0 when degree >= order).
auto homogeneous_monomial(const Rational& c, std::size_t degree, std::size_t order)
    -> Result<PowerSeries> {
    std::vector<Rational> v(order);
    if (degree < order) {
        v[degree] = c;
    }
    return PowerSeries::from_coeffs(std::move(v), order);
}

}  // namespace

auto adomian_polynomials(SeriesOperator N, const std::vector<PowerSeries>& components)
    -> Result<std::vector<PowerSeries>> {
    using Polys = std::vector<PowerSeries>;
    if (components.empty() || !N) {
        return make_error<Polys>(MathError::domain_error);
    }
    const std::size_t order = components.front().order();

    // Partial sum P = Σ u_i. Components of index > m are of degree > m and so do not
    // affect [x^m]; projecting N of the full sum is exact for every A_m with m <= k.
    auto sum = sum_components(components, order);
    if (!sum) {
        return make_error<Polys>(sum.error());
    }
    auto image = N(*sum);
    if (!image) {
        return make_error<Polys>(image.error());
    }
    if (image->order() != order) {
        return make_error<Polys>(MathError::domain_error);  // f must preserve the order
    }

    const std::size_t k = components.size() - 1;
    Polys polys;
    polys.reserve(components.size());
    for (std::size_t m = 0; m <= k; ++m) {
        // A_m = Π_m N(P) = ([x^m] N(P)) x^m.
        auto a_m = homogeneous_monomial(image->coefficient(m), m, order);
        if (!a_m) {
            return make_error<Polys>(a_m.error());
        }
        polys.push_back(std::move(*a_m));
    }
    return polys;
}

namespace {

// Shared graded Picard/Taylor recursion underlying both ADM and HPM: u_0 = u0 and
// u_{n+1} = L^{-1}[A_n]. Each component is homogeneous of degree n, so Σ u_n is the exact
// Taylor polynomial of the solution to `order` coefficients.
//
// EFFICIENCY. The partial sum P = u_0 + ... + u_k is maintained INCREMENTALLY, and each
// step extracts only the single coefficient A_k = ([x^k] f(P)) it needs. This is the
// difference between O(n) and O(n^2) bookkeeping: calling adomian_polynomials in the loop
// would re-sum all components AND build every A_0..A_k monomial only to discard all but the
// last, on every step. It is exact to do so because u_{k+1} is homogeneous of degree k+1,
// so folding it into P cannot change [x^m] f(P) for any m <= k already consumed. One
// evaluation of f per order (the irreducible cost) is retained.
auto graded_series_solve(const SeriesOperator& f, const Rational& u0, std::size_t order)
    -> Result<PowerSeries> {
    if (order == 0 || !f) {
        return make_error<PowerSeries>(MathError::domain_error);
    }
    const std::size_t n = order;
    auto p0 = PowerSeries::constant(u0, n);  // running partial sum P = u_0
    if (!p0) {
        return make_error<PowerSeries>(p0.error());
    }
    PowerSeries partial = std::move(*p0);

    // Build u_1..u_{n-1}, folding each into P; the step at k supplies the x^{k+1} coefficient.
    for (std::size_t k = 0; k + 1 < n; ++k) {
        auto image = f(partial);  // single f-evaluation on the sum so far
        if (!image) {
            return make_error<PowerSeries>(image.error());
        }
        if (image->order() != n) {
            return make_error<PowerSeries>(MathError::domain_error);  // f must preserve order
        }
        // A_k = ([x^k] f(P)) x^k, then u_{k+1} = L^{-1}[A_k] = integrate(A_k).
        auto a_k = homogeneous_monomial(image->coefficient(k), k, n);
        if (!a_k) {
            return make_error<PowerSeries>(a_k.error());
        }
        auto u_next = a_k->integrate();
        if (!u_next) {
            return make_error<PowerSeries>(u_next.error());
        }
        auto grown = partial.add(*u_next);  // fold u_{k+1} into the running sum P
        if (!grown) {
            return make_error<PowerSeries>(grown.error());
        }
        partial = std::move(*grown);
    }
    return partial;
}

}  // namespace

auto adm_solve(SeriesOperator f, Rational u0, std::size_t order) -> Result<PowerSeries> {
    return graded_series_solve(f, u0, order);
}

auto hpm_solve(SeriesOperator f, Rational u0, std::size_t order) -> Result<PowerSeries> {
    // The homotopy collapses to v' = p f(v); collecting powers of p gives v_0 = u0 and
    // v_n' = A_{n-1}, i.e. the same graded recursion as ADM — hence an identical series.
    return graded_series_solve(f, u0, order);
}

auto ham_solve(SeriesOperator f, Rational u0, Rational hbar, std::size_t order)
    -> Result<PowerSeries> {
    if (order == 0 || !f) {
        return make_error<PowerSeries>(MathError::domain_error);
    }
    const std::size_t n = order;
    auto u0_series = PowerSeries::constant(u0, n);
    if (!u0_series) {
        return make_error<PowerSeries>(u0_series.error());
    }
    // As in graded_series_solve, keep the running partial sum P incrementally (avoiding the
    // O(n^2) re-summation) and keep only the previous component u_{m-1} (needed for the
    // deformation's u_{m-1}' term and the χ_m u_{m-1} term). One f-evaluation per order.
    PowerSeries partial = *u0_series;  // P = Σ_{j<m} u_j, starts at u_0
    PowerSeries prev = *u0_series;     // u_{m-1}, starts at u_0

    for (std::size_t m = 1; m < n; ++m) {
        // A_{m-1} = Π_{m-1} f(Σ_{j<m} u_j): the grade-(m-1) projection of f on the sum so far.
        auto image = f(partial);
        if (!image) {
            return make_error<PowerSeries>(image.error());
        }
        if (image->order() != n) {
            return make_error<PowerSeries>(MathError::domain_error);
        }
        auto a_prev = homogeneous_monomial(image->coefficient(m - 1), m - 1, n);
        if (!a_prev) {
            return make_error<PowerSeries>(a_prev.error());
        }

        // R_m = u_{m-1}' - A_{m-1}, then L^{-1}[R_m] and its ħ-scaling.
        auto u_deriv = prev.derivative();
        if (!u_deriv) {
            return make_error<PowerSeries>(u_deriv.error());
        }
        auto r_m = u_deriv->subtract(*a_prev);
        if (!r_m) {
            return make_error<PowerSeries>(r_m.error());
        }
        auto linv = r_m->integrate();
        if (!linv) {
            return make_error<PowerSeries>(linv.error());
        }
        auto scaled = linv->scale(hbar);
        if (!scaled) {
            return make_error<PowerSeries>(scaled.error());
        }

        // u_m = χ_m u_{m-1} + ħ L^{-1}[R_m]; χ_m = 0 for m == 1, else 1.
        PowerSeries u_m = *scaled;
        if (m >= 2) {
            auto sum = prev.add(*scaled);
            if (!sum) {
                return make_error<PowerSeries>(sum.error());
            }
            u_m = std::move(*sum);
        }
        auto grown = partial.add(u_m);  // fold u_m into the running sum P
        if (!grown) {
            return make_error<PowerSeries>(grown.error());
        }
        partial = std::move(*grown);
        prev = std::move(u_m);
    }
    return partial;
}

}  // namespace nimblecas
