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

// ===========================================================================
// Nonlinear evolution PDEs (Cauchy-Kovalevskaya / Adomian time series) and a
// 1-D boundary-value (Poisson) solver. ADDITIVE to the linear API above.
// @author Olumuyiwa Oluwasanmi
//
// NONLINEAR EVOLUTION. For u_t = L[u] + N[u], u(x,0) = phi, with L a linear
// polynomial-coefficient spatial operator and N a (possibly nonlinear) spatial term,
// expand u(x,t) = sum_n c_n(x) t^n. Matching [t^k] on both sides (u_t shifts the index
// by one) gives the exact Cauchy-Kovalevskaya recurrence
//
//     (k+1) c_{k+1} = L[c_k] + A_k,      A_k = [t^k] N( sum_{i<=k} c_i(x) t^i ),
//
// where A_k is the k-th ADOMIAN polynomial of N: the time-grade-k projection of N
// applied to the partial sum. This is exactly the graded-projection Adomian
// construction of nimblecas.perturbation (see that module's header for why the graded
// projection of N of the whole partial sum, not a naive difference of partial sums, is
// the correct A_k), transported from Q[[x]] to the ring (Q[x])[[t]]: each series
// coefficient c_n(x) is a full spatial RationalPoly rather than a Rational, so the
// projection is performed natively over RationalPoly and NO link dependency on
// nimblecas.perturbation is added. Because [t^k] N depends only on c_0..c_k (products
// cannot lower the t-degree and the spatial derivative preserves it), the recurrence is
// causal and well defined.
//
// HONESTY (nonlinear). The result is the EXACT truncated Taylor series in t: the
// returned c_0..c_order are exact rationals-in-x and the discrete law
// (k+1) c_{k+1} = L[c_k] + A_k holds exactly on every retained term. For a genuine
// nonlinearity the series does NOT in general terminate; it is a LOCAL-in-t solution,
// valid only inside the time radius of convergence, and is NOT a global closed form
// (e.g. inviscid Burgers u(x,0) = x yields x - x t + x t^2 - ... = x/(1+t), singular at
// t = -1; reaction u_t = u^2 with u(x,0) = 1 yields 1 + t + t^2 + ... = 1/(1-t)). No
// boundary conditions are imposed (whole-line / formal construction), matching the
// linear solver above. Rule 32 railway: every RationalPoly / Rational error propagates.

// A spatial term acting on a TRUNCATED TIME SERIES. Given the coefficients c_0..c_M of
// u(x,t) = sum_n c_n(x) t^n (a vector of M+1 RationalPolys), it returns the time series
// of N[u](x,t) as a vector of the SAME length. For Burgers N[u] = -(u * u_x); for the
// quadratic reaction N[u] = u^2. Built from the series_* primitives below.
using TimeSeriesOperator =
    std::function<Result<std::vector<RationalPoly>>(const std::vector<RationalPoly>&)>;

// --- Time-series primitives (Cauchy product in t / spatial derivative) -------
// Scale every time coefficient by the rational s.
[[nodiscard]] auto series_scale(const std::vector<RationalPoly>& a, const Rational& s)
    -> Result<std::vector<RationalPoly>>;
// Spatial x-derivative applied coefficient-wise (d/dx of each c_n(x)); the length is
// preserved, so the series of u_x is returned to the same time-truncation order.
[[nodiscard]] auto series_dx(const std::vector<RationalPoly>& a)
    -> Result<std::vector<RationalPoly>>;
// Truncated Cauchy product in t: result_k = sum_{i+j=k} a_i(x) b_j(x), 0 <= k < len.
// Both operands must have equal length (else domain_error).
[[nodiscard]] auto series_product(const std::vector<RationalPoly>& a,
                                  const std::vector<RationalPoly>& b)
    -> Result<std::vector<RationalPoly>>;

// Solve u_t = L[u] + N[u], u(x,0) = phi, returning the time coefficients c_0..c_order
// (order+1 RationalPolys, c_n = coefficient of t^n) via the Adomian / Cauchy-Kovalevskaya
// recurrence above. `linear` may be empty (treated as L = 0, i.e. a pure-nonlinear PDE);
// `nonlinear` MUST be non-empty. order == 0, a null nonlinear term, or a nonlinear term
// that does not return the truncation length is a domain_error; any error raised by L or
// N is propagated. EXACT truncated series (see HONESTY above): local in t, not global.
[[nodiscard]] auto solve_nonlinear_evolution_pde(SpatialOperator linear,
                                                 TimeSeriesOperator nonlinear,
                                                 const RationalPoly& phi, std::size_t order)
    -> Result<std::vector<RationalPoly>>;

// --- Concrete nonlinear builders --------------------------------------------
// Burgers' equation u_t + u u_x = viscosity * u_xx, i.e. u_t = viscosity u_xx - u u_x.
// L = heat_operator(viscosity), N[u] = -(u * u_x). viscosity == 0 gives the inviscid
// equation. Returns c_0..c_order (EXACT truncated time series; local in t).
[[nodiscard]] auto burgers(Rational viscosity, const RationalPoly& phi, std::size_t order)
    -> Result<std::vector<RationalPoly>>;
// Quadratic reaction-diffusion u_t = diffusivity * u_xx + u^2 (Fisher-type source).
// L = heat_operator(diffusivity), N[u] = u^2. Returns c_0..c_order (EXACT truncated; local).
[[nodiscard]] auto reaction_diffusion_quadratic(Rational diffusivity, const RationalPoly& phi,
                                                std::size_t order)
    -> Result<std::vector<RationalPoly>>;

// --- Boundary-value / steady-state: 1-D Poisson -----------------------------
// Solve the Dirichlet BVP u''(x) = f(x) on [a, b] with u(a) = alpha, u(b) = beta, for a
// polynomial source f. EXACT over Q: a particular p with p'' = f is obtained by
// integrating f twice, then u = p + C1 x + C0 with (C1, C0) fixed by the exact 2x2
// rational solve of the two boundary equations. Unlike the evolution series this is a
// genuine closed-form polynomial (no truncation). a == b is a degenerate interval and a
// domain_error; Rational overflow is propagated.
[[nodiscard]] auto solve_poisson_bvp_1d(const RationalPoly& f, const Rational& a,
                                        const Rational& alpha, const Rational& b,
                                        const Rational& beta) -> Result<RationalPoly>;

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

// ===========================================================================
// Nonlinear evolution PDEs and the 1-D Poisson BVP.
// ===========================================================================

namespace {

// Exact indefinite integral of a RationalPoly with zero constant of integration:
// integral of sum a_i x^i dx = sum a_i/(i+1) x^{i+1}. The zero polynomial integrates to
// zero. (RationalPoly exposes derivative() but not integrate(), so we supply it here.)
[[nodiscard]] auto integrate_poly(const RationalPoly& p) -> Result<RationalPoly> {
    if (p.is_zero()) {
        return RationalPoly{};
    }
    const auto deg = static_cast<std::size_t>(p.degree());
    std::vector<Rational> out(deg + 2);  // out[0] = 0 (constant of integration)
    for (std::size_t i = 0; i <= deg; ++i) {
        auto denom = Rational::from_int(static_cast<std::int64_t>(i + 1));
        auto q = p.coefficient(i).divide(denom);  // denom >= 1, never zero
        if (!q) {
            return make_error<RationalPoly>(q.error());
        }
        out[i + 1] = *q;
    }
    return RationalPoly::from_coeffs(std::move(out));
}

}  // namespace

auto series_scale(const std::vector<RationalPoly>& a, const Rational& s)
    -> Result<std::vector<RationalPoly>> {
    std::vector<RationalPoly> out;
    out.reserve(a.size());
    for (const auto& c : a) {
        auto scaled = c.scale(s);
        if (!scaled) {
            return make_error<std::vector<RationalPoly>>(scaled.error());
        }
        out.push_back(std::move(*scaled));
    }
    return out;
}

auto series_dx(const std::vector<RationalPoly>& a) -> Result<std::vector<RationalPoly>> {
    std::vector<RationalPoly> out;
    out.reserve(a.size());
    for (const auto& c : a) {
        auto d = c.derivative();
        if (!d) {
            return make_error<std::vector<RationalPoly>>(d.error());
        }
        out.push_back(std::move(*d));
    }
    return out;
}

auto series_product(const std::vector<RationalPoly>& a, const std::vector<RationalPoly>& b)
    -> Result<std::vector<RationalPoly>> {
    if (a.size() != b.size()) {
        return make_error<std::vector<RationalPoly>>(MathError::domain_error);
    }
    const std::size_t len = a.size();
    std::vector<RationalPoly> out(len);  // each entry the zero polynomial
    for (std::size_t k = 0; k < len; ++k) {
        RationalPoly acc;  // 0
        for (std::size_t i = 0; i <= k; ++i) {
            auto prod = a[i].multiply(b[k - i]);  // spatial polynomial product
            if (!prod) {
                return make_error<std::vector<RationalPoly>>(prod.error());
            }
            auto sum = acc.add(*prod);
            if (!sum) {
                return make_error<std::vector<RationalPoly>>(sum.error());
            }
            acc = std::move(*sum);
        }
        out[k] = std::move(acc);
    }
    return out;
}

auto solve_nonlinear_evolution_pde(SpatialOperator linear, TimeSeriesOperator nonlinear,
                                   const RationalPoly& phi, std::size_t order)
    -> Result<std::vector<RationalPoly>> {
    using Coeffs = std::vector<RationalPoly>;
    if (order == 0 || !nonlinear) {
        return make_error<Coeffs>(MathError::domain_error);
    }
    // The step index k+1 is cast to int64 for Rational::make; guard the physically
    // unreachable overflow of an order beyond INT64_MAX so the cast can never wrap.
    if (order > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        return make_error<Coeffs>(MathError::overflow);
    }
    const std::size_t len = order + 1;

    // The running partial sum P = sum_{i<=k} c_i(x) t^i, held as a length-`len` time
    // series with the not-yet-known coefficients left zero. Since [t^k] N(P) depends only
    // on c_0..c_k, those trailing zeros never affect the projection A_k we extract.
    Coeffs partial(len);  // all zero polynomials
    partial[0] = phi;     // c_0 = phi

    for (std::size_t k = 0; k + 1 <= order; ++k) {
        // A_k = [t^k] N(P): one nonlinear evaluation on the sum so far. (Re-evaluating N
        // per step is the clear O(order) x O(product) route; an incremental single-pass
        // scheme as in nimblecas.perturbation is possible but not needed for correctness.)
        auto image = nonlinear(partial);
        if (!image) {
            return make_error<Coeffs>(image.error());
        }
        if (image->size() != len) {
            return make_error<Coeffs>(MathError::domain_error);  // N must preserve the length
        }
        RationalPoly rhs = (*image)[k];  // A_k

        // Add the linear contribution L[c_k] when a linear operator is supplied.
        if (linear) {
            auto lin = linear(partial[k]);
            if (!lin) {
                return make_error<Coeffs>(lin.error());
            }
            auto sum = rhs.add(*lin);
            if (!sum) {
                return make_error<Coeffs>(sum.error());
            }
            rhs = std::move(*sum);
        }

        // c_{k+1} = (L[c_k] + A_k) / (k+1).
        auto inv = Rational::make(1, static_cast<std::int64_t>(k + 1));
        if (!inv) {
            return make_error<Coeffs>(inv.error());
        }
        auto c_next = rhs.scale(*inv);
        if (!c_next) {
            return make_error<Coeffs>(c_next.error());
        }
        partial[k + 1] = std::move(*c_next);
    }
    return partial;
}

auto burgers(Rational viscosity, const RationalPoly& phi, std::size_t order)
    -> Result<std::vector<RationalPoly>> {
    // N[u] = -(u * u_x): differentiate the series in x, Cauchy-multiply by u, negate.
    TimeSeriesOperator convective = [](const std::vector<RationalPoly>& u)
        -> Result<std::vector<RationalPoly>> {
        auto ux = series_dx(u);
        if (!ux) {
            return make_error<std::vector<RationalPoly>>(ux.error());
        }
        auto prod = series_product(u, *ux);
        if (!prod) {
            return make_error<std::vector<RationalPoly>>(prod.error());
        }
        return series_scale(*prod, Rational::from_int(-1));
    };
    return solve_nonlinear_evolution_pde(heat_operator(std::move(viscosity)),
                                         std::move(convective), phi, order);
}

auto reaction_diffusion_quadratic(Rational diffusivity, const RationalPoly& phi,
                                  std::size_t order) -> Result<std::vector<RationalPoly>> {
    // N[u] = u^2: the truncated Cauchy square of the series.
    TimeSeriesOperator square = [](const std::vector<RationalPoly>& u)
        -> Result<std::vector<RationalPoly>> { return series_product(u, u); };
    return solve_nonlinear_evolution_pde(heat_operator(std::move(diffusivity)),
                                         std::move(square), phi, order);
}

auto solve_poisson_bvp_1d(const RationalPoly& f, const Rational& a, const Rational& alpha,
                          const Rational& b, const Rational& beta) -> Result<RationalPoly> {
    // Degenerate interval: the two Dirichlet conditions collapse onto one point.
    if (a == b) {
        return make_error<RationalPoly>(MathError::domain_error);
    }
    // Particular solution p with p'' = f: integrate f twice (zero constants).
    auto first = integrate_poly(f);  // p' candidate
    if (!first) {
        return make_error<RationalPoly>(first.error());
    }
    auto particular = integrate_poly(*first);  // p, p'' = f exactly
    if (!particular) {
        return make_error<RationalPoly>(particular.error());
    }
    auto pa = eval_poly(*particular, a);
    if (!pa) {
        return make_error<RationalPoly>(pa.error());
    }
    auto pb = eval_poly(*particular, b);
    if (!pb) {
        return make_error<RationalPoly>(pb.error());
    }
    // u = p + C1 x + C0 with  p(a)+C1 a+C0 = alpha,  p(b)+C1 b+C0 = beta. Subtracting,
    // C1 = ( (alpha - beta) - (p(a) - p(b)) ) / (a - b), then C0 = alpha - p(a) - C1 a.
    auto a_minus_b = a.subtract(b);  // != 0 (checked above)
    if (!a_minus_b) {
        return make_error<RationalPoly>(a_minus_b.error());
    }
    auto d_alpha = alpha.subtract(beta);
    if (!d_alpha) {
        return make_error<RationalPoly>(d_alpha.error());
    }
    auto d_p = pa->subtract(*pb);
    if (!d_p) {
        return make_error<RationalPoly>(d_p.error());
    }
    auto c1_num = d_alpha->subtract(*d_p);
    if (!c1_num) {
        return make_error<RationalPoly>(c1_num.error());
    }
    auto c1 = c1_num->divide(*a_minus_b);
    if (!c1) {
        return make_error<RationalPoly>(c1.error());
    }
    auto c1a = c1->multiply(a);
    if (!c1a) {
        return make_error<RationalPoly>(c1a.error());
    }
    auto tmp = alpha.subtract(*pa);
    if (!tmp) {
        return make_error<RationalPoly>(tmp.error());
    }
    auto c0 = tmp->subtract(*c1a);
    if (!c0) {
        return make_error<RationalPoly>(c0.error());
    }
    // Assemble u = p + C1 x + C0.
    auto with_linear = particular->add(RationalPoly::monomial(*c1, 1));  // + C1 x
    if (!with_linear) {
        return make_error<RationalPoly>(with_linear.error());
    }
    return with_linear->add(RationalPoly::constant(*c0));  // + C0
}

}  // namespace nimblecas
