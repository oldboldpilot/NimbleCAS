// NimbleCAS Homotopy Analysis Method (HAM) for nonlinear evolution PDEs, with the
// convergence-control parameter ħ, over the exact symbolic ring (Q[x])[[t]]/(t^N).
// @author Olumuyiwa Oluwasanmi
//
// This module implements the Homotopy Analysis Method — Liao's ħ-deformation — for the
// semilinear evolution PDE
//
//     u_t = F[u],   u(x, 0) = u_0(x),   F[u] = ν u_xx + c · u u_x + Σ_p a_p u^p,
//
// i.e. a diffusion term (ν u_xx), an optional Burgers/advection convection term
// (c · u u_x), and a polynomial reaction term f(u) = Σ_p a_p u^p. This is EXACTLY the
// class handled; an operator outside it (a different linear part, a non-polynomial
// nonlinearity, mixed higher derivatives) is a `MathError::not_implemented`.
//
// WHY THIS IS DISTINCT FROM ADM/HPM. The forward solvers in `nimblecas.pde`
// (solve_nonlinear_evolution_pde / _hpm, burgers, reaction_diffusion_quadratic, kdv)
// build the single Cauchy-Kovalevskaya / Adomian time series — the Taylor polynomial of
// the solution in t. HAM instead embeds the problem in a one-parameter homotopy carrying
// the convergence-control parameter ħ. The zeroth-order deformation is
//
//     (1 − q) L[φ(x,t;q) − u_0] = q ħ H(x,t) 𝒩[φ],   𝒩[φ] = φ_t − F[φ],
//
// with L = ∂_t the auxiliary linear operator, q ∈ [0,1] the embedding parameter and
// H ≡ 1 the auxiliary function. Expanding φ = u_0 + Σ_{m≥1} u_m q^m and matching powers
// of q gives the m-th order deformation equation
//
//     L[u_m − χ_m u_{m−1}] = ħ R_m,   R_m = (u_{m−1})_t − A_{m−1},   χ_m = [m ≥ 2],
//
// where A_{m−1} = [q^{m−1}] F[φ] is the (m−1)-th Adomian/q-graded polynomial of F. With
// L = ∂_t its exact right inverse is L^{-1} = ∫_0^t (·) dτ (zero datum, since every
// u_j for j ≥ 1 vanishes at t = 0), so each order is solved by one exact integration in t
//
//     u_m = χ_m u_{m−1} + ħ ∫_0^t R_m dτ,     u ≈ Σ_{m=0}^{order} u_m,
//
// and ħ is an extra degree of freedom the ADM/HPM Taylor series do not have.
//
// THE TWO GRADINGS (q vs t). This is the generalisation `nimblecas.pde` left open in its
// header ("does NOT (yet) offer a HAM variant … generalising the ħ-parameterised
// deformation from a single Q[[x]] series to the two-index (Q[x])[[t]] ring needs care").
// Here each component u_m(x,t) is a full truncated time series (a vector of x-polynomials,
// one per power of t), so φ is a series in BOTH q and t. A_{m−1} = [q^{m−1}] F[φ] is
// obtained by an exact q-convolution of the components u_0..u_{m−1} (each a t-series),
// x-derivatives taken by `nimblecas.diff`, and t-products by the truncated Cauchy product
// in the (Q[x])[[t]] ring. Because F only differentiates in x (q-degree preserving) and
// forms products (q-degree raising), [q^{m−1}] F depends solely on u_0..u_{m−1}: the
// recurrence is causal and exact.
//
// ħ = −1 REDUCTION. At ħ = −1 the deformation collapses to u_m = ∫_0^t A_{m−1} dτ (the
// (u_{m−1})_t term integrates back to χ_m u_{m−1} and cancels it, since u_{m−1}(x,0) = 0
// for m ≥ 2), which is exactly the ADM recursion — and hence the Cauchy-Kovalevskaya
// Taylor series that `nimblecas.pde` computes. At ħ = −1 every component is homogeneous of
// t-degree m, so Σ_{m=0}^{order} u_m reproduces the forward series c_0..c_order term for
// term. The `pdeham_tests` reduction test asserts this equality on a shared example.
//
// HONESTY (Rule 32). The result is an EXACT symbolic truncated series in t: a polynomial
// Σ_{k=0}^{order} c_k(x) t^k with c_k exact over Q, NOT a closed-form solution. It is exact
// term-by-term to the stated order under the HAM deformation; the truncation error is
// O(t^{order+1}) and no claim is made about the radius of convergence. Two honesty points
// specific to HAM: (1) only ħ = −1 yields the Taylor polynomial whose PDE residual is
// O(t^{order}); a general ħ selects a different (still exactly rational) member of the
// deformation family whose lower-order t-coefficients keep changing with the HAM order —
// that is the intended convergence-control freedom, not the Taylor series. (2) Because
// higher HAM orders correct lower t-coefficients, truncating at HAM order m and at t-order
// m are different truncations for ħ ≠ −1. Every symbolic `Result` error is propagated;
// order 0 or an operator outside the handled class is a `MathError` (domain_error /
// not_implemented).

export module nimblecas.pdeham;

import std;
import nimblecas.core;
import nimblecas.ratpoly;   // Rational (exact ħ and PDE coefficients)
import nimblecas.symbolic;  // Expr
import nimblecas.simplify;  // simplify (canonical folding of sums/scalings)
import nimblecas.diff;      // differentiate (exact spatial x-derivatives)
import nimblecas.expand;    // expand (exact distribution of polynomial products in x)

export namespace nimblecas {

// The exact class of nonlinear evolution PDEs handled: u_t = F[u] with
//
//     F[u] = diffusivity · u_xx  +  convection · u u_x  +  Σ_p reaction_coeffs[p] · u^p .
//
// `reaction_coeffs[p]` is the coefficient of u^p in the polynomial reaction f(u); an empty
// vector means f ≡ 0. `convection = −1` is the Burgers convective term (u u_x moved to the
// RHS as −u u_x); `convection = 0` drops it. `diffusivity = 0` drops the diffusion term.
// Any PDE expressible in this form is handled exactly; anything else is not_implemented.
struct HamPde {
    Rational diffusivity{};                 // ν, coefficient of u_xx
    Rational convection{};                  // c, coefficient of the convective term u u_x
    std::vector<Rational> reaction_coeffs{}; // f(u) = Σ_p reaction_coeffs[p] u^p
};

// Solve u_t = F[u], u(x,0) = u0, by the Homotopy Analysis Method with convergence-control
// parameter ħ = `hbar`, returning the truncated HAM approximation u ≈ Σ_{m=0}^{order} u_m
// as an Expr polynomial in the symbols `xvar` (space) and `tvar` (time), truncated at
// t^{order}. `u0` is the initial datum, an Expr in `xvar` (a polynomial in x for an exact
// terminating spatial degree; any Expr is accepted and handled exactly symbolically).
// `hbar = −1` recovers the ADM/HPM/Cauchy-Kovalevskaya forward series of `nimblecas.pde`.
//
// Errors: order == 0, or `xvar == tvar`, is domain_error. A HamPde outside the handled
// class cannot be constructed (the struct IS the class), so the only not_implemented path
// is a degenerate/empty description with no diffusion, convection, or reaction (nothing to
// solve). Any symbolic Result error (overflow while folding a coefficient, an undefined
// form) is propagated verbatim.
[[nodiscard]] auto ham_pde_solve(const HamPde& pde, const Expr& u0, Rational hbar,
                                 std::size_t order, std::string_view tvar,
                                 std::string_view xvar) -> Result<Expr>;

// Convenience: reaction-diffusion u_t = diffusivity · u_xx + Σ_p reaction_coeffs[p] u^p.
[[nodiscard]] auto ham_reaction_diffusion(Rational diffusivity,
                                          std::vector<Rational> reaction_coeffs, const Expr& u0,
                                          Rational hbar, std::size_t order, std::string_view tvar,
                                          std::string_view xvar) -> Result<Expr>;

// Convenience: Burgers' equation u_t + u u_x = viscosity · u_xx, i.e. u_t = ν u_xx − u u_x.
[[nodiscard]] auto ham_burgers(Rational viscosity, const Expr& u0, Rational hbar,
                               std::size_t order, std::string_view tvar, std::string_view xvar)
    -> Result<Expr>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

namespace {

// A truncated time series u(x,t) = Σ_{k=0}^{N-1} coeff[k](x) t^k, held as N Exprs in x
// (coeff[k] = the coefficient of t^k). Kept in expanded, simplified form throughout.
using TimeSeries = std::vector<Expr>;
// A q-series φ = Σ_j comp[j] q^j whose coefficients comp[j] are themselves time series;
// used only transiently to extract the Adomian coefficient A_{m-1} = [q^{m-1}] F[φ].
using QSeries = std::vector<TimeSeries>;

[[nodiscard]] auto expr_zero() -> Expr { return Expr::integer(0); }

// Canonical normal form for an x-coefficient: expand distributes polynomial products and
// then simplifies (collecting like monomials), so structural equality and cancellation of
// like terms both work. Exact — expand/simplify only ever rewrite to an equal expression.
[[nodiscard]] auto norm(const Expr& e) -> Result<Expr> { return expand(e); }

// An Expr constant from an exact Rational (integer when the denominator is 1).
[[nodiscard]] auto rat_to_expr(const Rational& r) -> Result<Expr> {
    if (r.is_integer()) {
        return Expr::integer(r.numerator());
    }
    return Expr::rational(r.numerator(), r.denominator());
}

// --- time-series primitives (all fixed length N) ---------------------------

[[nodiscard]] auto ts_zero(std::size_t n) -> TimeSeries {
    return TimeSeries(n, expr_zero());
}

// Elementwise a + b, normalised.
[[nodiscard]] auto ts_add(const TimeSeries& a, const TimeSeries& b) -> Result<TimeSeries> {
    const std::size_t n = a.size();
    TimeSeries out(n, expr_zero());
    for (std::size_t k = 0; k < n; ++k) {
        auto s = norm(Expr::sum({a[k], b[k]}));
        if (!s) {
            return make_error<TimeSeries>(s.error());
        }
        out[k] = std::move(*s);
    }
    return out;
}

// Elementwise a − b, normalised (subtraction via + (−1)·b, expanded so terms cancel).
[[nodiscard]] auto ts_sub(const TimeSeries& a, const TimeSeries& b) -> Result<TimeSeries> {
    const std::size_t n = a.size();
    TimeSeries out(n, expr_zero());
    for (std::size_t k = 0; k < n; ++k) {
        Expr neg = Expr::product({Expr::integer(-1), b[k]});
        auto s = norm(Expr::sum({a[k], std::move(neg)}));
        if (!s) {
            return make_error<TimeSeries>(s.error());
        }
        out[k] = std::move(*s);
    }
    return out;
}

// Scale every coefficient by the rational s.
[[nodiscard]] auto ts_scale(const TimeSeries& a, const Rational& s) -> Result<TimeSeries> {
    auto se = rat_to_expr(s);
    if (!se) {
        return make_error<TimeSeries>(se.error());
    }
    const std::size_t n = a.size();
    TimeSeries out(n, expr_zero());
    for (std::size_t k = 0; k < n; ++k) {
        auto p = norm(Expr::product({*se, a[k]}));
        if (!p) {
            return make_error<TimeSeries>(p.error());
        }
        out[k] = std::move(*p);
    }
    return out;
}

// Spatial x-derivative applied coefficient-wise (d/dx of each c_k(x)) via nimblecas.diff.
[[nodiscard]] auto ts_dx(const TimeSeries& a, std::string_view xvar) -> Result<TimeSeries> {
    const std::size_t n = a.size();
    TimeSeries out(n, expr_zero());
    for (std::size_t k = 0; k < n; ++k) {
        auto d = differentiate(a[k], xvar);
        if (!d) {
            return make_error<TimeSeries>(d.error());
        }
        auto nd = norm(*d);
        if (!nd) {
            return make_error<TimeSeries>(nd.error());
        }
        out[k] = std::move(*nd);
    }
    return out;
}

// Second spatial derivative u_xx.
[[nodiscard]] auto ts_dxx(const TimeSeries& a, std::string_view xvar) -> Result<TimeSeries> {
    auto d1 = ts_dx(a, xvar);
    if (!d1) {
        return make_error<TimeSeries>(d1.error());
    }
    return ts_dx(*d1, xvar);
}

// Time derivative ∂_t: [t^{k-1}] ← k · [t^k]. The top coefficient (t^{N-1}) has no image and
// the result stays length N (its t^{N-1} coefficient is 0).
[[nodiscard]] auto ts_dt(const TimeSeries& a) -> Result<TimeSeries> {
    const std::size_t n = a.size();
    TimeSeries out(n, expr_zero());
    for (std::size_t k = 1; k < n; ++k) {
        auto p = norm(Expr::product({Expr::integer(static_cast<std::int64_t>(k)), a[k]}));
        if (!p) {
            return make_error<TimeSeries>(p.error());
        }
        out[k - 1] = std::move(*p);
    }
    return out;
}

// Exact right inverse of ∂_t: L^{-1}[·] = ∫_0^t (·) dτ, i.e. [t^{k+1}] ← [t^k] / (k+1),
// zero constant of integration. The t^{N-1} coefficient would map to t^N and is truncated.
[[nodiscard]] auto ts_integrate(const TimeSeries& a) -> Result<TimeSeries> {
    const std::size_t n = a.size();
    TimeSeries out(n, expr_zero());
    for (std::size_t k = 0; k + 1 < n; ++k) {
        auto inv = Rational::make(1, static_cast<std::int64_t>(k + 1));  // k+1 >= 1, never zero
        if (!inv) {
            return make_error<TimeSeries>(inv.error());
        }
        auto ie = rat_to_expr(*inv);
        if (!ie) {
            return make_error<TimeSeries>(ie.error());
        }
        auto p = norm(Expr::product({*ie, a[k]}));
        if (!p) {
            return make_error<TimeSeries>(p.error());
        }
        out[k + 1] = std::move(*p);
    }
    return out;
}

// Truncated Cauchy product in t: [t^k](a·b) = Σ_{i+j=k} a_i(x) b_j(x), 0 <= k < N.
[[nodiscard]] auto ts_mul(const TimeSeries& a, const TimeSeries& b) -> Result<TimeSeries> {
    const std::size_t n = a.size();
    TimeSeries out(n, expr_zero());
    for (std::size_t k = 0; k < n; ++k) {
        std::vector<Expr> terms;
        terms.reserve(k + 1);
        for (std::size_t i = 0; i <= k; ++i) {
            terms.push_back(Expr::product({a[i], b[k - i]}));  // spatial polynomial product
        }
        auto s = norm(Expr::sum(std::move(terms)));  // expand distributes and collects
        if (!s) {
            return make_error<TimeSeries>(s.error());
        }
        out[k] = std::move(*s);
    }
    return out;
}

// --- q-series primitives (for the Adomian coefficient A_{m-1}) --------------

// Elementwise q-series sum (both length qlen; each entry a length-N time series).
[[nodiscard]] auto q_add(const QSeries& a, const QSeries& b, std::size_t n) -> Result<QSeries> {
    const std::size_t qlen = a.size();
    QSeries out(qlen, ts_zero(n));
    for (std::size_t j = 0; j < qlen; ++j) {
        auto s = ts_add(a[j], b[j]);
        if (!s) {
            return make_error<QSeries>(s.error());
        }
        out[j] = std::move(*s);
    }
    return out;
}

// Scale every q-coefficient (a time series) by the rational s.
[[nodiscard]] auto q_scale(const QSeries& a, const Rational& s) -> Result<QSeries> {
    QSeries out(a.size(), TimeSeries{});
    for (std::size_t j = 0; j < a.size(); ++j) {
        auto t = ts_scale(a[j], s);
        if (!t) {
            return make_error<QSeries>(t.error());
        }
        out[j] = std::move(*t);
    }
    return out;
}

// Truncated Cauchy product in q of two q-series to length qlen: the q-coefficients multiply
// as time series (ts_mul), so this is exact in the (Q[x])[[t]] ring.
[[nodiscard]] auto q_conv(const QSeries& a, const QSeries& b, std::size_t qlen, std::size_t n)
    -> Result<QSeries> {
    QSeries out(qlen, ts_zero(n));
    for (std::size_t k = 0; k < qlen; ++k) {
        TimeSeries acc = ts_zero(n);
        for (std::size_t i = 0; i <= k; ++i) {
            const std::size_t j = k - i;
            if (i >= a.size() || j >= b.size()) {
                continue;
            }
            auto prod = ts_mul(a[i], b[j]);
            if (!prod) {
                return make_error<QSeries>(prod.error());
            }
            auto sum = ts_add(acc, *prod);
            if (!sum) {
                return make_error<QSeries>(sum.error());
            }
            acc = std::move(*sum);
        }
        out[k] = std::move(acc);
    }
    return out;
}

// φ raised to the integer power p (p >= 1) as a q-series truncated to length qlen.
[[nodiscard]] auto q_pow(const QSeries& phi, std::int64_t p, std::size_t qlen, std::size_t n)
    -> Result<QSeries> {
    // Truncate phi to qlen first (components beyond q^{qlen-1} cannot affect [q^{<qlen}]).
    QSeries base(qlen, ts_zero(n));
    for (std::size_t j = 0; j < qlen && j < phi.size(); ++j) {
        base[j] = phi[j];
    }
    QSeries acc = base;
    for (std::int64_t e = 1; e < p; ++e) {
        auto next = q_conv(acc, base, qlen, n);
        if (!next) {
            return make_error<QSeries>(next.error());
        }
        acc = std::move(*next);
    }
    return acc;
}

// The Adomian / q-graded coefficient A_{m-1} = [q^{m-1}] F[φ] for the handled operator
// class, where m = components.size() and φ = Σ_{j<m} components[j] q^j. Includes the linear
// diffusion contribution ν (u_{m-1})_xx (a pure [q^{m-1}] term), the convective term
// c [q^{m-1}](φ φ_x), and the polynomial reaction Σ_p a_p [q^{m-1}] φ^p.
[[nodiscard]] auto ham_adomian(const QSeries& components, const HamPde& pde, std::size_t n,
                               std::string_view xvar) -> Result<TimeSeries> {
    const std::size_t m = components.size();  // want index m-1
    const std::size_t qlen = m;
    const std::size_t last = m - 1;

    TimeSeries acc = ts_zero(n);

    // Reaction f(u) = Σ_p a_p u^p.
    for (std::size_t p = 0; p < pde.reaction_coeffs.size(); ++p) {
        const Rational& a_p = pde.reaction_coeffs[p];
        if (a_p.is_zero()) {
            continue;
        }
        if (p == 0) {
            // Constant term a_0: only the q^0 coefficient, so it reaches A_{m-1} iff m == 1.
            if (last == 0) {
                auto ce = rat_to_expr(a_p);
                if (!ce) {
                    return make_error<TimeSeries>(ce.error());
                }
                TimeSeries c0 = ts_zero(n);
                c0[0] = *ce;
                auto s = ts_add(acc, c0);
                if (!s) {
                    return make_error<TimeSeries>(s.error());
                }
                acc = std::move(*s);
            }
            continue;
        }
        auto pw = q_pow(components, static_cast<std::int64_t>(p), qlen, n);
        if (!pw) {
            return make_error<TimeSeries>(pw.error());
        }
        auto scaled = ts_scale((*pw)[last], a_p);  // a_p · [q^{m-1}] φ^p
        if (!scaled) {
            return make_error<TimeSeries>(scaled.error());
        }
        auto s = ts_add(acc, *scaled);
        if (!s) {
            return make_error<TimeSeries>(s.error());
        }
        acc = std::move(*s);
    }

    // Convection c · u u_x.
    if (!pde.convection.is_zero()) {
        QSeries phi_x(qlen, ts_zero(n));
        for (std::size_t j = 0; j < qlen; ++j) {
            auto dx = ts_dx(components[j], xvar);
            if (!dx) {
                return make_error<TimeSeries>(dx.error());
            }
            phi_x[j] = std::move(*dx);
        }
        auto conv = q_conv(components, phi_x, qlen, n);  // [q^{m-1}](φ φ_x)
        if (!conv) {
            return make_error<TimeSeries>(conv.error());
        }
        auto scaled = ts_scale((*conv)[last], pde.convection);
        if (!scaled) {
            return make_error<TimeSeries>(scaled.error());
        }
        auto s = ts_add(acc, *scaled);
        if (!s) {
            return make_error<TimeSeries>(s.error());
        }
        acc = std::move(*s);
    }

    // Diffusion ν u_xx: linear, so [q^{m-1}](ν φ_xx) = ν (u_{m-1})_xx.
    if (!pde.diffusivity.is_zero()) {
        auto uxx = ts_dxx(components[last], xvar);
        if (!uxx) {
            return make_error<TimeSeries>(uxx.error());
        }
        auto scaled = ts_scale(*uxx, pde.diffusivity);
        if (!scaled) {
            return make_error<TimeSeries>(scaled.error());
        }
        auto s = ts_add(acc, *scaled);
        if (!s) {
            return make_error<TimeSeries>(s.error());
        }
        acc = std::move(*s);
    }

    return acc;
}

// Assemble the truncated series Σ_k coeff[k] t^k into a single Expr (expanded).
[[nodiscard]] auto series_to_expr(const TimeSeries& coeffs, std::string_view tvar)
    -> Result<Expr> {
    const Expr t = Expr::symbol(std::string(tvar));
    std::vector<Expr> terms;
    terms.reserve(coeffs.size());
    for (std::size_t k = 0; k < coeffs.size(); ++k) {
        if (k == 0) {
            terms.push_back(coeffs[k]);
        } else {
            terms.push_back(Expr::product(
                {coeffs[k], Expr::power(t, Expr::integer(static_cast<std::int64_t>(k)))}));
        }
    }
    return norm(Expr::sum(std::move(terms)));
}

}  // namespace

auto ham_pde_solve(const HamPde& pde, const Expr& u0, Rational hbar, std::size_t order,
                   std::string_view tvar, std::string_view xvar) -> Result<Expr> {
    if (order == 0 || tvar == xvar) {
        return make_error<Expr>(MathError::domain_error);
    }
    // Every step index (k, k+1, m) is cast to int64 for exact Rational scalings; guard the
    // physically unreachable order beyond INT64_MAX so no cast can ever wrap.
    if (order > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        return make_error<Expr>(MathError::overflow);
    }
    // A description with no diffusion, convection, and no reaction is not a solvable PDE in
    // this class (F ≡ 0 has nothing to deform against) — reject honestly.
    const bool has_reaction =
        std::ranges::any_of(pde.reaction_coeffs, [](const Rational& r) { return !r.is_zero(); });
    if (pde.diffusivity.is_zero() && pde.convection.is_zero() && !has_reaction) {
        return make_error<Expr>(MathError::not_implemented);
    }

    const std::size_t n = order + 1;  // t^0 .. t^{order}

    // u_0 = u0(x) as a time series [u0, 0, ..., 0].
    auto u0n = norm(u0);
    if (!u0n) {
        return make_error<Expr>(u0n.error());
    }
    TimeSeries u_zero = ts_zero(n);
    u_zero[0] = std::move(*u0n);

    QSeries components;           // u_0, u_1, ... (each a time series)
    components.push_back(u_zero);
    TimeSeries partial = u_zero;  // running sum Σ_j u_j

    for (std::size_t m = 1; m <= order; ++m) {
        // A_{m-1} = [q^{m-1}] F[φ] from the components u_0..u_{m-1} available so far.
        auto a_prev = ham_adomian(components, pde, n, xvar);
        if (!a_prev) {
            return make_error<Expr>(a_prev.error());
        }
        const TimeSeries& u_prev = components.back();  // u_{m-1}

        // R_m = (u_{m-1})_t − A_{m-1}, then ħ · L^{-1}[R_m].
        auto u_prev_t = ts_dt(u_prev);
        if (!u_prev_t) {
            return make_error<Expr>(u_prev_t.error());
        }
        auto r_m = ts_sub(*u_prev_t, *a_prev);
        if (!r_m) {
            return make_error<Expr>(r_m.error());
        }
        auto integ = ts_integrate(*r_m);
        if (!integ) {
            return make_error<Expr>(integ.error());
        }
        auto scaled = ts_scale(*integ, hbar);
        if (!scaled) {
            return make_error<Expr>(scaled.error());
        }

        // u_m = χ_m u_{m-1} + ħ L^{-1}[R_m]; χ_m = 0 for m == 1, else 1.
        TimeSeries u_m = *scaled;
        if (m >= 2) {
            auto sum = ts_add(u_prev, *scaled);
            if (!sum) {
                return make_error<Expr>(sum.error());
            }
            u_m = std::move(*sum);
        }

        auto grown = ts_add(partial, u_m);  // fold u_m into the running sum
        if (!grown) {
            return make_error<Expr>(grown.error());
        }
        partial = std::move(*grown);
        components.push_back(std::move(u_m));
    }

    return series_to_expr(partial, tvar);
}

auto ham_reaction_diffusion(Rational diffusivity, std::vector<Rational> reaction_coeffs,
                            const Expr& u0, Rational hbar, std::size_t order,
                            std::string_view tvar, std::string_view xvar) -> Result<Expr> {
    HamPde pde{.diffusivity = std::move(diffusivity),
               .convection = Rational{},
               .reaction_coeffs = std::move(reaction_coeffs)};
    return ham_pde_solve(pde, u0, hbar, order, tvar, xvar);
}

auto ham_burgers(Rational viscosity, const Expr& u0, Rational hbar, std::size_t order,
                 std::string_view tvar, std::string_view xvar) -> Result<Expr> {
    // u_t + u u_x = ν u_xx  =>  u_t = ν u_xx − u u_x  =>  convection coefficient = −1.
    HamPde pde{.diffusivity = std::move(viscosity),
               .convection = Rational::from_int(-1),
               .reaction_coeffs = {}};
    return ham_pde_solve(pde, u0, hbar, order, tvar, xvar);
}

}  // namespace nimblecas
