// NimbleCAS optimize module: unconstrained NUMERICAL optimization over R^n.
// @author Olumuyiwa Oluwasanmi
//
// Conforms to config/cpp_details.txt: C++23 modules, `import std`, trailing return
// types, no owning raw pointers, std::expected error handling (no exceptions),
// [[nodiscard]] on every observer / fallible call, std::span non-owning views.
//
// ---------------------------------------------------------------------------
// HONESTY BOUNDARY (this is a NUMERICAL module — read before trusting a result)
// ---------------------------------------------------------------------------
// Everything here is IEEE-754 double-precision, LOCAL, iterative optimization.
//   * It finds a LOCAL stationary point (grad ~ 0), NOT a global minimum. There is
//     NO claim of global optimality and NO claim of exactness.
//   * Convergence is NOT guaranteed for non-convex, ill-conditioned, discontinuous,
//     or non-smooth objectives. On a convex/well-conditioned problem the methods
//     converge; elsewhere they may stall, cycle, or crawl.
//   * Results DEPEND ON THE INITIAL GUESS x0 (different basins -> different minima).
//   * Finite-difference gradients/Hessians carry discretization error O(h^2) plus
//     round-off O(eps/h); the reported gradient norm is therefore approximate.
//   * Line searches, damping, and the simplex are heuristics with tuning constants;
//     they trade robustness for guarantees.
// A run that exhausts max_iterations is NOT an error: it returns a valid
// OptimizeResult with converged == false and the best iterate found. Only genuinely
// invalid input (empty x0, or a non-finite entry / non-finite objective at the start)
// is reported as MathError::domain_error.

export module nimblecas.optimize;

import std;
import nimblecas.core;
import nimblecas.parallel;

export namespace nimblecas::optimize {

// ---------------------------------------------------------------------------
// Callable contracts.
// ---------------------------------------------------------------------------
// Objective   : x -> f(x), a scalar. x is a non-owning view valid only for the call.
// Gradient    : x -> grad f(x), an n-vector (same length as x). Optional; when absent
//               a central finite-difference gradient is used.
// HessianFn   : x -> Hessian, ROW-MAJOR n*n (H[i*n+j]). Optional; when absent a
//               finite-difference Hessian is used (from the gradient if one is given,
//               else from a second difference of the objective).
using Objective = std::function<double(std::span<const double>)>;
using Gradient  = std::function<std::vector<double>(std::span<const double>)>;
using HessianFn = std::function<std::vector<double>(std::span<const double>)>;

// Nonlinear-CG update rule for the conjugate direction coefficient beta.
enum class CGVariant : std::uint8_t {
    fletcher_reeves,  // beta = (g'.g') / (g.g)
    polak_ribiere,    // beta = max(0, g'.(g'-g) / (g.g))  (PR+, auto-restart)
};

// ---------------------------------------------------------------------------
// Tuning knobs. All defaults are conservative, general-purpose values.
// ---------------------------------------------------------------------------
struct Options {
    // Convergence.
    double grad_tol = 1e-6;             // stop when ||grad f||_2 <= grad_tol.
    double step_tol = 1e-12;            // stop when ||x_{k+1}-x_k||_2 <= step_tol.
    std::size_t max_iterations = 1000;  // hard cap; hitting it => converged=false.

    // Finite differences.
    double fd_step = 1e-6;              // base step h; per-coord h_i = h*max(1,|x_i|).

    // Line search (Armijo sufficient-decrease + strong-Wolfe curvature).
    double armijo_c1 = 1e-4;            // sufficient-decrease constant (0 < c1 < c2 < 1).
    double wolfe_c2 = 0.9;             // curvature constant (0.9 quasi-Newton, 0.1 CG).
    double backtrack_rho = 0.5;        // Armijo step contraction factor.
    std::size_t max_line_search = 50;  // max line-search / zoom iterations.

    // L-BFGS history length (number of (s,y) pairs retained).
    std::size_t lbfgs_memory = 10;

    // Implicit filtering stencil-scale schedule.
    double imf_h0 = 0.5;        // initial finite-difference / stencil scale.
    double imf_h_min = 1e-8;    // finest scale; the schedule halves h down to this.

    // Nelder-Mead simplex coefficients.
    double nm_reflect = 1.0;
    double nm_expand = 2.0;
    double nm_contract = 0.5;
    double nm_shrink = 0.5;
};

// ---------------------------------------------------------------------------
// Outcome of an optimization run.
// ---------------------------------------------------------------------------
struct OptimizeResult {
    std::vector<double> x;        // best iterate found.
    double fx = 0.0;              // objective at x.
    std::size_t iterations = 0;   // iterations actually performed.
    bool converged = false;       // true iff a stopping tolerance was met (NOT max-iter).
    double grad_norm = 0.0;       // ||grad f(x)||_2 (approx; 0 reported for derivative-free).
};

// ===========================================================================
// Public numerical utilities (exported: tests exercise the FD gradient directly).
// ===========================================================================

// Central finite-difference gradient of f at x with base step h. Per-coordinate step
// h_i = h * max(1, |x_i|) balances truncation vs. round-off. O(2n) evaluations.
[[nodiscard]] auto finite_difference_gradient(const Objective& f, std::span<const double> x,
                                              double h) -> std::vector<double>;

// Finite-difference Hessian (ROW-MAJOR n*n, symmetrized). If `grad` is provided the
// Hessian is a central difference of the gradient (O(2n) grad calls); otherwise it is a
// second difference of f (O(2n^2) f calls). Both carry the usual FD error.
[[nodiscard]] auto finite_difference_hessian(const Objective& f, const Gradient& grad,
                                             std::span<const double> x, double h)
    -> std::vector<double>;

// ===========================================================================
// Optimizers. Each returns Result<OptimizeResult>:
//   * domain_error  — x0 empty, or a non-finite x0 / f(x0) / grad(x0).
//   * otherwise a value (which may have converged == false on max-iter).
// A default-constructed (empty) Gradient/HessianFn selects finite differences.
// ===========================================================================

// Steepest descent with Armijo backtracking line search. Robust but linearly
// convergent; slow on ill-conditioned (elongated) valleys.
[[nodiscard]] auto gradient_descent(Objective f, std::span<const double> x0,
                                    Gradient grad = {}, Options opts = {})
    -> Result<OptimizeResult>;

// Damped Newton. Uses the Hessian (given or finite-difference); solves H p = -g by
// dense LU with partial pivoting. When H is not positive definite (or singular) a
// Levenberg-style multiple of the identity, tau*I, is added and grown until p is a
// genuine descent direction; if that fails it falls back to steepest descent. Then an
// Armijo backtracking line search globalizes the step.
[[nodiscard]] auto newton_method(Objective f, std::span<const double> x0, Gradient grad = {},
                                 HessianFn hess = {}, Options opts = {})
    -> Result<OptimizeResult>;

// BFGS quasi-Newton with a maintained INVERSE-Hessian approximation and strong-Wolfe
// line search. Skips the rank-2 update when the curvature condition s.y > 0 fails
// (keeps the approximation positive definite). Superlinear near the solution.
[[nodiscard]] auto bfgs(Objective f, std::span<const double> x0, Gradient grad = {},
                        Options opts = {}) -> Result<OptimizeResult>;

// Limited-memory BFGS: the two-loop recursion over the last `lbfgs_memory` (s,y)
// pairs, with the Nocedal H0 = (s.y)/(y.y) I scaling. O(n * memory) per step, no n*n
// storage — the method of choice for large n.
[[nodiscard]] auto l_bfgs(Objective f, std::span<const double> x0, Gradient grad = {},
                          Options opts = {}) -> Result<OptimizeResult>;

// Nonlinear conjugate gradient (Fletcher-Reeves or Polak-Ribiere) with strong-Wolfe
// line search, automatic restart every n steps and whenever the new direction is not a
// descent direction.
[[nodiscard]] auto conjugate_gradient(Objective f, std::span<const double> x0,
                                      Gradient grad = {},
                                      CGVariant variant = CGVariant::polak_ribiere,
                                      Options opts = {}) -> Result<OptimizeResult>;

// Nelder-Mead downhill simplex: DERIVATIVE-FREE (no gradient used or needed). Uses
// reflection / expansion / contraction / shrink on an (n+1)-vertex simplex seeded
// around x0. Robust on non-smooth / noisy objectives but only linearly convergent and
// with no stationary-point guarantee. Convergence here means the simplex's relative
// spread of objective values fell below grad_tol.
[[nodiscard]] auto nelder_mead(Objective f, std::span<const double> x0, Options opts = {})
    -> Result<OptimizeResult>;

// C.T. Kelley's IMPLICIT FILTERING (derivative-free, for NOISY / nondifferentiable
// objectives that are a smooth trend plus low-amplitude noise). At each outer step it
// forms a central finite-difference "simplex gradient" at the current stencil scale h,
// takes a BFGS-style (or steepest-descent) step, and accepts it only via an Armijo test
// that demands decrease beyond what the scale-h model predicts — so the shrinking scale
// filters the noise. On a "stencil failure" (no productive step at h) it HALVES h and
// retries, down to opts.imf_h_min. Optional box constraints [lo, hi] are honoured by
// projecting every iterate and stencil point onto the box (bound-constrained implicit
// filtering); pass empty lo/hi for the unconstrained problem.
//
// HONESTY: numerical, derivative-free, LOCAL. It suits functions with low-amplitude
// noise on a smooth trend; it filters noise through the scale schedule but gives NO
// global-optimality guarantee, and convergence depends on the noise level relative to
// the scale schedule. Non-convergence returns converged==false (not an error). Empty x0,
// non-finite inputs, a bounds/x0 length mismatch, or lo > hi all yield domain_error.
[[nodiscard]] auto implicit_filtering(Objective f, std::span<const double> x0,
                                      std::span<const double> lo = {},
                                      std::span<const double> hi = {}, Options opts = {})
    -> Result<OptimizeResult>;

// ===========================================================================
// PARALLEL + DISTRIBUTED multistart (built on nimblecas.parallel).
// ===========================================================================
// HONESTY (read before trusting a multistart result)
// ---------------------------------------------------------------------------
// Multistart runs a LOCAL optimizer (bfgs / newton / nelder_mead / ...) from many
// initial points and keeps the best. This is the standard way to attack multimodal /
// non-convex problems with local methods, but it inherits every caveat of the
// underlying local method:
//   * Each local run finds a LOCAL stationary point, NOT a global minimum. Trying more
//     starts IMPROVES THE CHANCE of finding a good/global basin but gives NO
//     global-optimality guarantee and NO exactness claim.
//   * A start whose local run ends with converged == false is NOT an error; it is a
//     valid (if unpolished) candidate and still competes for "best". Only genuinely
//     invalid input (empty x0 / non-finite entry / non-finite f(x0)) makes a start's
//     Result an error, and such starts are simply skipped in the reduction.
//   * DETERMINISM: the argmin over starts is fixed as (lowest fx, then LOWEST start
//     index). Because every local run is itself deterministic (same start -> identical
//     result), the winner is identical regardless of thread count, of serial vs.
//     parallel execution, and of how the starts are partitioned across distributed
//     shards. parallel_multistart therefore returns exactly what a serial loop with the
//     same tie-break would return.
// PRECONDITION (parallel path): the objective, gradient, Hessian and the selected local
// optimizer must be safe to INVOKE CONCURRENTLY for distinct starts (stateless or
// internally synchronised). The starts are optimized on multiple threads at once.

// Local optimizers multistart can dispatch to (see make_local_optimizer).
enum class Method : std::uint8_t {
    gradient_descent,
    newton,
    bfgs,
    l_bfgs,
    conjugate_gradient,
    nelder_mead,
};

// A local optimizer bound to its objective / derivatives / options: it maps a single
// start x0 to a Result<OptimizeResult>. This is the general "method selector" the caller
// supplies to parallel_multistart / multistart_shard; use make_local_optimizer to build
// one from a Method, or provide any custom callable with this shape.
// PRECONDITION: must be safe to invoke concurrently for distinct start points.
using LocalOptimizer = std::function<Result<OptimizeResult>(std::span<const double>)>;

// Outcome of a multistart (or shard) run.
struct MultistartResult {
    OptimizeResult best;          // best local result: lowest fx, ties broken by start index.
    std::size_t best_start = 0;   // GLOBAL index (into `starts`) of the winning start.
    std::size_t succeeded = 0;    // number of starts that produced a value (not a domain_error).
};

// Bind `method` (with the given optional gradient / Hessian / CG variant / Options) into
// a LocalOptimizer. Methods that ignore a callback simply never read it (nelder_mead uses
// neither gradient nor Hessian; only newton reads the Hessian; only conjugate_gradient
// reads the CG variant). A default-constructed Gradient/HessianFn selects finite
// differences, exactly as in the single-method entry points.
[[nodiscard]] auto make_local_optimizer(Method method, Objective f, Gradient grad = {},
                                        HessianFn hess = {},
                                        CGVariant cg_variant = CGVariant::polak_ribiere,
                                        Options opts = {}) -> LocalOptimizer;

// Run `local` from every start in `starts` IN PARALLEL (nimblecas.parallel; serial below
// the runtime's grain, or when parallel == false) and return the deterministic best
// (lowest fx, ties -> lowest start index). Starts whose local run errors are skipped.
//   * domain_error — `starts` empty, or EVERY start errored (the first/lowest-index
//     start's error code is propagated).
//   * otherwise a MultistartResult whose best_start identifies the winning start.
[[nodiscard]] auto parallel_multistart(const LocalOptimizer& local,
                                       std::span<const std::vector<double>> starts,
                                       bool parallel = true) -> Result<MultistartResult>;

// Convenience overload: pick the local method by enum instead of pre-binding a
// LocalOptimizer. Equivalent to parallel_multistart(make_local_optimizer(...), starts).
[[nodiscard]] auto multistart(Method method, Objective f,
                              std::span<const std::vector<double>> starts, Gradient grad = {},
                              HessianFn hess = {},
                              CGVariant cg_variant = CGVariant::polak_ribiere, Options opts = {},
                              bool parallel = true) -> Result<MultistartResult>;

// DISTRIBUTED shard entry point (stateless, pure — safe to map across processes / nodes
// via SGE, Ray, etc.). Runs `local` over ONLY this shard's subset of starts, namely every
// starts[i] with i % num_shards == shard_index, and returns THIS shard's best local result
// with best_start set to the GLOBAL index. A driver then takes reduce_shards over the
// per-shard results to recover the global winner — identical to a single-process
// parallel_multistart over the same starts, for any num_shards.
//   * domain_error — num_shards == 0, shard_index >= num_shards, `starts` empty, this
//     shard was assigned no starts, or every assigned start errored.
[[nodiscard]] auto multistart_shard(const LocalOptimizer& local,
                                    std::span<const std::vector<double>> starts,
                                    std::size_t shard_index, std::size_t num_shards,
                                    bool parallel = true) -> Result<MultistartResult>;

// Driver reduction over per-shard MultistartResults: the global argmin (lowest fx, ties
// -> smallest GLOBAL best_start). Order-independent — the shards may be gathered in any
// order — because best_start values are disjoint across shards, so the (fx, best_start)
// order is total. `succeeded` in the result is the sum across shards. Empty span ->
// domain_error.
[[nodiscard]] auto reduce_shards(std::span<const MultistartResult> shard_results)
    -> Result<MultistartResult>;

}  // namespace nimblecas::optimize

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::optimize {
namespace {

// --- small dense linear-algebra helpers over double (module-local) ----------

[[nodiscard]] auto all_finite(std::span<const double> v) noexcept -> bool {
    return std::ranges::all_of(v, [](double x) noexcept { return std::isfinite(x); });
}

[[nodiscard]] auto dot(std::span<const double> a, std::span<const double> b) noexcept -> double {
    // a and b are the same length by construction at every call site.
    double acc = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        acc = std::fma(a[i], b[i], acc);
    }
    return acc;
}

[[nodiscard]] auto norm2(std::span<const double> a) noexcept -> double {
    return std::sqrt(dot(a, a));
}

// out = x + alpha * p  (out may alias neither x nor p; caller supplies a fresh buffer).
[[nodiscard]] auto axpy(std::span<const double> x, double alpha, std::span<const double> p)
    -> std::vector<double> {
    std::vector<double> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        out[i] = std::fma(alpha, p[i], x[i]);
    }
    return out;
}

// Solve A z = b for z, A row-major n*n (consumed by value), b consumed by value.
// Gaussian elimination with partial pivoting; std::nullopt if (near-)singular.
[[nodiscard]] auto lu_solve(std::vector<double> a, std::vector<double> b, std::size_t n)
    -> std::optional<std::vector<double>> {
    for (std::size_t col = 0; col < n; ++col) {
        // Partial pivot: largest magnitude at or below the diagonal in this column.
        std::size_t pivot = col;
        double best = std::abs(a[col * n + col]);
        for (std::size_t r = col + 1; r < n; ++r) {
            const double mag = std::abs(a[r * n + col]);
            if (mag > best) {
                best = mag;
                pivot = r;
            }
        }
        if (!(best > 0.0) || !std::isfinite(best)) {
            return std::nullopt;  // singular column (all zero / non-finite).
        }
        if (pivot != col) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(a[col * n + j], a[pivot * n + j]);
            }
            std::swap(b[col], b[pivot]);
        }
        const double diag = a[col * n + col];
        for (std::size_t r = col + 1; r < n; ++r) {
            const double factor = a[r * n + col] / diag;
            if (factor == 0.0) {
                continue;
            }
            for (std::size_t j = col; j < n; ++j) {
                a[r * n + j] = std::fma(-factor, a[col * n + j], a[r * n + j]);
            }
            b[r] = std::fma(-factor, b[col], b[r]);
        }
    }
    // Back substitution.
    std::vector<double> z(n);
    for (std::size_t i = n; i-- > 0;) {
        double s = b[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            s = std::fma(-a[i * n + j], z[j], s);
        }
        const double diag = a[i * n + i];
        if (!(std::abs(diag) > 0.0)) {
            return std::nullopt;
        }
        z[i] = s / diag;
    }
    if (!all_finite(z)) {
        return std::nullopt;
    }
    return z;
}

// Build the effective gradient: the user's callback, or a central FD closure over f.
[[nodiscard]] auto effective_gradient(const Objective& f, const Gradient& grad, double h)
    -> Gradient {
    if (grad) {
        return grad;
    }
    return [f, h](std::span<const double> x) -> std::vector<double> {
        return finite_difference_gradient(f, x, h);
    };
}

// Newton search direction p solving (H + tau I) p = -g, with tau grown from 0 until p
// is a strict descent direction (g.p < 0). Falls back to steepest descent -g if every
// damped solve fails. `hess` is row-major n*n; `g` is the gradient at the current x.
[[nodiscard]] auto damped_newton_direction(std::span<const double> hess,
                                           std::span<const double> g, std::size_t n)
    -> std::vector<double> {
    double diag_scale = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        diag_scale = std::max(diag_scale, std::abs(hess[i * n + i]));  // |H_ii|
    }
    const double base = diag_scale > 0.0 ? diag_scale : 1.0;
    const double gnorm = norm2(g);

    std::vector<double> neg_g(n);
    for (std::size_t i = 0; i < n; ++i) {
        neg_g[i] = -g[i];
    }

    double tau = 0.0;
    // If any diagonal entry is non-positive the matrix cannot be SPD; pre-shift.
    for (std::size_t i = 0; i < n; ++i) {
        if (hess[i * n + i] <= 0.0) {
            tau = 1e-3 * base;
            break;
        }
    }
    constexpr std::size_t max_attempts = 40;
    for (std::size_t attempt = 0; attempt < max_attempts; ++attempt) {
        std::vector<double> m(hess.begin(), hess.end());
        for (std::size_t i = 0; i < n; ++i) {
            m[i * n + i] += tau;
        }
        auto sol = lu_solve(std::move(m), neg_g, n);
        if (sol) {
            const double gp = dot(g, *sol);
            const double pn = norm2(*sol);
            // Require a strict descent direction with a non-degenerate angle to -g.
            if (gp < -1e-12 * std::max(1.0, gnorm * pn)) {
                return std::move(*sol);
            }
        }
        tau = (tau == 0.0) ? 1e-3 * base : tau * 4.0;
    }
    return neg_g;  // steepest-descent fallback.
}

// BFGS inverse-Hessian rank-2 update, shared by bfgs() and implicit_filtering():
//   H+ = (I - rho s y^T) H (I - rho y s^T) + rho s s^T,   rho = 1/(s.y).
// A no-op when the curvature condition s.y > 0 fails (keeps H positive definite).
auto bfgs_inverse_update(std::vector<double>& hinv, std::span<const double> s,
                         std::span<const double> y, std::size_t n) -> void {
    const double sy = dot(s, y);
    if (!(sy > 1e-12 * std::max(1.0, norm2(s) * norm2(y)))) {
        return;  // skip: non-positive curvature.
    }
    const double rho = 1.0 / sy;
    std::vector<double> hy(n, 0.0);  // hy = H y.
    for (std::size_t i = 0; i < n; ++i) {
        double acc = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            acc = std::fma(hinv[i * n + j], y[j], acc);
        }
        hy[i] = acc;
    }
    const double yhy = dot(y, hy);
    const double coeff = rho * rho * yhy + rho;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double v = hinv[i * n + j];
            v -= rho * (s[i] * hy[j] + hy[i] * s[j]);
            v += coeff * s[i] * s[j];
            hinv[i * n + j] = v;
        }
    }
}

// --- box constraints & implicit-filtering stencil ---------------------------

// Project a point onto the box [lo, hi]; identity when lo/hi are empty (unconstrained).
[[nodiscard]] auto project(std::span<const double> v, std::span<const double> lo,
                           std::span<const double> hi) -> std::vector<double> {
    std::vector<double> out(v.begin(), v.end());
    if (!lo.empty()) {
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] = std::clamp(out[i], lo[i], hi[i]);
        }
    }
    return out;
}

// Central finite-difference "simplex gradient" at absolute stencil scale h, with each
// stencil point projected onto the box. The i-th slope divides by the ACTUAL projected
// coordinate span (which may be < 2h near a bound); a degenerate span yields 0. A
// non-finite objective at a stencil point also yields 0 for that component.
[[nodiscard]] auto simplex_gradient(const Objective& f, std::span<const double> x, double h,
                                    std::span<const double> lo, std::span<const double> hi)
    -> std::vector<double> {
    const std::size_t n = x.size();
    std::vector<double> g(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> xp(x.begin(), x.end());
        std::vector<double> xm(x.begin(), x.end());
        xp[i] += h;
        xm[i] -= h;
        xp = project(xp, lo, hi);
        xm = project(xm, lo, hi);
        const double denom = xp[i] - xm[i];
        if (!(std::abs(denom) > 0.0)) {
            continue;  // stencil collapsed against a bound: no information here.
        }
        const double fp = f(xp);
        const double fm = f(xm);
        if (!std::isfinite(fp) || !std::isfinite(fm)) {
            continue;
        }
        g[i] = (fp - fm) / denom;
    }
    return g;
}

// Projected-gradient stationarity measure: ||x - proj(x - g)||_2. Equals ||g|| at an
// interior point and shrinks to 0 at a KKT point on the boundary.
[[nodiscard]] auto projected_grad_norm(std::span<const double> x, std::span<const double> g,
                                       std::span<const double> lo, std::span<const double> hi)
    -> double {
    std::vector<double> step = axpy(x, -1.0, g);       // x - g.
    std::vector<double> proj = project(step, lo, hi);   // proj(x - g).
    std::vector<double> diff = axpy(x, -1.0, proj);     // x - proj(x - g).
    return norm2(diff);
}

// --- line searches ----------------------------------------------------------

struct LineSearch {
    double alpha = 0.0;
    std::vector<double> x_new;
    double f_new = 0.0;
    std::vector<double> g_new;  // populated only by the strong-Wolfe search.
    bool ok = false;
};

// Armijo backtracking: find alpha in (0,1] with f(x+alpha p) <= f + c1 alpha (g.p).
// Requires p to be a descent direction (g.p < 0); returns ok=false if none is found or
// the objective is non-finite along p.
[[nodiscard]] auto armijo_backtracking(const Objective& f, std::span<const double> x, double fx,
                                       std::span<const double> g, std::span<const double> p,
                                       const Options& opts) -> LineSearch {
    LineSearch ls;
    const double gp = dot(g, p);
    if (!(gp < 0.0)) {
        return ls;  // not a descent direction.
    }
    double alpha = 1.0;
    for (std::size_t i = 0; i < opts.max_line_search; ++i) {
        std::vector<double> trial = axpy(x, alpha, p);
        const double ft = f(trial);
        if (std::isfinite(ft) && ft <= std::fma(opts.armijo_c1 * alpha, gp, fx)) {
            ls.alpha = alpha;
            ls.x_new = std::move(trial);
            ls.f_new = ft;
            ls.ok = true;
            return ls;
        }
        alpha *= opts.backtrack_rho;
    }
    return ls;  // exhausted; ok stays false.
}

// Strong-Wolfe line search (Nocedal & Wright Alg. 3.5 + zoom 3.6). Guarantees, on
// success, both sufficient decrease and |g_new.p| <= c2 |g.p|, so the BFGS/L-BFGS
// curvature condition holds. Always fills g_new on success. ok=false if p is not a
// descent direction or the search cannot satisfy the conditions.
[[nodiscard]] auto strong_wolfe(const Objective& f, const Gradient& grad,
                                std::span<const double> x, double fx, std::span<const double> g,
                                std::span<const double> p, const Options& opts) -> LineSearch {
    LineSearch ls;
    const double phi0 = fx;
    const double dphi0 = dot(g, p);
    if (!(dphi0 < 0.0)) {
        return ls;  // not a descent direction.
    }
    const double c1 = opts.armijo_c1;
    const double c2 = opts.wolfe_c2;
    const std::size_t max_it = opts.max_line_search;

    auto make_ok = [&](double alpha, std::vector<double> xn, double fn,
                       std::vector<double> gn) -> LineSearch {
        return LineSearch{alpha, std::move(xn), fn, std::move(gn), true};
    };

    // zoom() narrows a bracket [lo, hi] known to contain an acceptable point.
    auto zoom = [&](double a_lo, double phi_lo, double a_hi) -> LineSearch {
        for (std::size_t j = 0; j < max_it; ++j) {
            const double alpha = 0.5 * (a_lo + a_hi);  // bisection (robust, no NaNs).
            std::vector<double> xt = axpy(x, alpha, p);
            const double phi = f(xt);
            if (!std::isfinite(phi) || phi > std::fma(c1 * alpha, dphi0, phi0) || phi >= phi_lo) {
                a_hi = alpha;
                continue;
            }
            std::vector<double> gt = grad(xt);
            const double dphi = dot(gt, p);
            if (std::abs(dphi) <= -c2 * dphi0) {
                return make_ok(alpha, std::move(xt), phi, std::move(gt));
            }
            if (dphi * (a_hi - a_lo) >= 0.0) {
                a_hi = a_lo;
            }
            a_lo = alpha;
            phi_lo = phi;
        }
        return LineSearch{};  // could not satisfy curvature within budget.
    };

    double alpha_prev = 0.0;
    double phi_prev = phi0;
    double alpha = 1.0;
    constexpr double alpha_max = 1e10;
    for (std::size_t i = 0; i < max_it; ++i) {
        std::vector<double> xt = axpy(x, alpha, p);
        const double phi = f(xt);
        if (!std::isfinite(phi) || phi > std::fma(c1 * alpha, dphi0, phi0) ||
            (i > 0 && phi >= phi_prev)) {
            return zoom(alpha_prev, phi_prev, alpha);
        }
        std::vector<double> gt = grad(xt);
        const double dphi = dot(gt, p);
        if (std::abs(dphi) <= -c2 * dphi0) {
            return make_ok(alpha, std::move(xt), phi, std::move(gt));
        }
        if (dphi >= 0.0) {
            return zoom(alpha, phi, alpha_prev);
        }
        alpha_prev = alpha;
        phi_prev = phi;
        alpha = std::min(alpha * 2.0, alpha_max);
    }
    return ls;  // ok=false.
}

// Assemble a finished result (objective + gradient norm) at x.
[[nodiscard]] auto finish(std::vector<double> x, double fx, std::size_t iters, bool converged,
                          double gnorm) -> OptimizeResult {
    return OptimizeResult{std::move(x), fx, iters, converged, gnorm};
}

// Validate x0 and the objective/gradient at x0; return the copied start vector.
[[nodiscard]] auto validate_start(const Objective& f, const Gradient* grad,
                                  std::span<const double> x0)
    -> Result<std::vector<double>> {
    if (x0.empty() || !all_finite(x0)) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    std::vector<double> x(x0.begin(), x0.end());
    const double f0 = f(x);
    if (!std::isfinite(f0)) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    if (grad != nullptr && *grad) {
        auto g0 = (*grad)(x);
        if (g0.size() != x.size() || !all_finite(g0)) {
            return make_error<std::vector<double>>(MathError::domain_error);
        }
    }
    return x;
}

}  // namespace

// --- public FD utilities ----------------------------------------------------

auto finite_difference_gradient(const Objective& f, std::span<const double> x, double h)
    -> std::vector<double> {
    const std::size_t n = x.size();
    std::vector<double> g(n);
    std::vector<double> xm(x.begin(), x.end());  // mutable working copy.
    for (std::size_t i = 0; i < n; ++i) {
        const double hi = h * std::max(1.0, std::abs(x[i]));
        const double orig = xm[i];
        xm[i] = orig + hi;
        const double fp = f(xm);
        xm[i] = orig - hi;
        const double fm = f(xm);
        xm[i] = orig;  // restore before the next coordinate.
        g[i] = (fp - fm) / (2.0 * hi);
    }
    return g;
}

auto finite_difference_hessian(const Objective& f, const Gradient& grad,
                               std::span<const double> x, double h) -> std::vector<double> {
    const std::size_t n = x.size();
    std::vector<double> hmat(n * n, 0.0);
    std::vector<double> xm(x.begin(), x.end());

    if (grad) {
        // Central difference of the gradient: column j = (g(x+h e_j) - g(x-h e_j))/(2h).
        for (std::size_t j = 0; j < n; ++j) {
            const double hj = h * std::max(1.0, std::abs(x[j]));
            const double orig = xm[j];
            xm[j] = orig + hj;
            auto gp = grad(xm);
            xm[j] = orig - hj;
            auto gm = grad(xm);
            xm[j] = orig;
            for (std::size_t i = 0; i < n; ++i) {
                hmat[i * n + j] = (gp[i] - gm[i]) / (2.0 * hj);
            }
        }
    } else {
        // Second difference of f. Diagonal via the 3-point stencil, off-diagonal via
        // the 4-point mixed stencil. O(n^2) evaluations.
        const double f0 = f(xm);
        for (std::size_t i = 0; i < n; ++i) {
            const double hi = h * std::max(1.0, std::abs(x[i]));
            const double oi = xm[i];
            xm[i] = oi + hi;
            const double fp = f(xm);
            xm[i] = oi - hi;
            const double fm = f(xm);
            xm[i] = oi;
            hmat[i * n + i] = (fp - 2.0 * f0 + fm) / (hi * hi);
            for (std::size_t j = i + 1; j < n; ++j) {
                const double hj = h * std::max(1.0, std::abs(x[j]));
                const double oj = xm[j];
                xm[i] = oi + hi; xm[j] = oj + hj; const double fpp = f(xm);
                xm[i] = oi + hi; xm[j] = oj - hj; const double fpm = f(xm);
                xm[i] = oi - hi; xm[j] = oj + hj; const double fmp = f(xm);
                xm[i] = oi - hi; xm[j] = oj - hj; const double fmm = f(xm);
                xm[i] = oi; xm[j] = oj;
                const double mixed = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj);
                hmat[i * n + j] = mixed;
                hmat[j * n + i] = mixed;
            }
        }
    }
    // Symmetrize away FD asymmetry.
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double s = 0.5 * (hmat[i * n + j] + hmat[j * n + i]);
            hmat[i * n + j] = s;
            hmat[j * n + i] = s;
        }
    }
    return hmat;
}

// --- gradient descent -------------------------------------------------------

auto gradient_descent(Objective f, std::span<const double> x0, Gradient grad, Options opts)
    -> Result<OptimizeResult> {
    auto start = validate_start(f, &grad, x0);
    if (!start) {
        return make_error<OptimizeResult>(start.error());
    }
    std::vector<double> x = std::move(*start);
    const Gradient g_of = effective_gradient(f, grad, opts.fd_step);

    double fx = f(x);
    std::vector<double> g = g_of(x);
    std::size_t iter = 0;
    for (; iter < opts.max_iterations; ++iter) {
        const double gnorm = norm2(g);
        if (gnorm <= opts.grad_tol) {
            return finish(std::move(x), fx, iter, true, gnorm);
        }
        std::vector<double> p(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) {
            p[i] = -g[i];  // steepest descent.
        }
        auto ls = armijo_backtracking(f, x, fx, g, p, opts);
        if (!ls.ok) {
            return finish(std::move(x), fx, iter, false, gnorm);  // stalled line search.
        }
        std::vector<double> step = axpy(ls.x_new, -1.0, x);  // x_new - x.
        const double snorm = norm2(step);
        x = std::move(ls.x_new);
        fx = ls.f_new;
        g = g_of(x);
        if (snorm <= opts.step_tol) {
            return finish(std::move(x), fx, iter + 1, true, norm2(g));
        }
    }
    return finish(std::move(x), fx, iter, false, norm2(g));  // max-iter: NOT an error.
}

// --- Newton -----------------------------------------------------------------

auto newton_method(Objective f, std::span<const double> x0, Gradient grad, HessianFn hess,
                   Options opts) -> Result<OptimizeResult> {
    auto start = validate_start(f, &grad, x0);
    if (!start) {
        return make_error<OptimizeResult>(start.error());
    }
    std::vector<double> x = std::move(*start);
    const std::size_t n = x.size();
    const Gradient g_of = effective_gradient(f, grad, opts.fd_step);

    double fx = f(x);
    std::vector<double> g = g_of(x);
    std::size_t iter = 0;
    for (; iter < opts.max_iterations; ++iter) {
        const double gnorm = norm2(g);
        if (gnorm <= opts.grad_tol) {
            return finish(std::move(x), fx, iter, true, gnorm);
        }
        std::vector<double> h = hess ? hess(x)
                                     : finite_difference_hessian(f, grad, x, opts.fd_step);
        if (h.size() != n * n || !all_finite(h)) {
            // Unusable Hessian: fall back to a steepest-descent step this iteration.
            h.assign(n * n, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                h[i * n + i] = 1.0;
            }
        }
        std::vector<double> p = damped_newton_direction(h, g, n);
        auto ls = armijo_backtracking(f, x, fx, g, p, opts);
        if (!ls.ok) {
            return finish(std::move(x), fx, iter, false, gnorm);
        }
        std::vector<double> step = axpy(ls.x_new, -1.0, x);
        const double snorm = norm2(step);
        x = std::move(ls.x_new);
        fx = ls.f_new;
        g = g_of(x);
        if (snorm <= opts.step_tol) {
            return finish(std::move(x), fx, iter + 1, true, norm2(g));
        }
    }
    return finish(std::move(x), fx, iter, false, norm2(g));
}

// --- BFGS -------------------------------------------------------------------

auto bfgs(Objective f, std::span<const double> x0, Gradient grad, Options opts)
    -> Result<OptimizeResult> {
    auto start = validate_start(f, &grad, x0);
    if (!start) {
        return make_error<OptimizeResult>(start.error());
    }
    std::vector<double> x = std::move(*start);
    const std::size_t n = x.size();
    const Gradient g_of = effective_gradient(f, grad, opts.fd_step);

    // Inverse-Hessian approximation H, initialized to the identity (row-major n*n).
    std::vector<double> hinv(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        hinv[i * n + i] = 1.0;
    }

    double fx = f(x);
    std::vector<double> g = g_of(x);
    std::size_t iter = 0;
    for (; iter < opts.max_iterations; ++iter) {
        const double gnorm = norm2(g);
        if (gnorm <= opts.grad_tol) {
            return finish(std::move(x), fx, iter, true, gnorm);
        }
        // p = -H g.
        std::vector<double> p(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            double acc = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                acc = std::fma(hinv[i * n + j], g[j], acc);
            }
            p[i] = -acc;
        }
        if (!(dot(g, p) < 0.0)) {
            // Numerical loss of positive-definiteness: reset to steepest descent.
            for (std::size_t i = 0; i < n; ++i) {
                p[i] = -g[i];
            }
        }
        auto ls = strong_wolfe(f, g_of, x, fx, g, p, opts);
        if (!ls.ok) {
            ls = armijo_backtracking(f, x, fx, g, p, opts);  // fall back.
            if (!ls.ok) {
                return finish(std::move(x), fx, iter, false, gnorm);
            }
            ls.g_new = g_of(ls.x_new);
        }
        std::vector<double> s = axpy(ls.x_new, -1.0, x);        // s = x_new - x.
        std::vector<double> y = axpy(ls.g_new, -1.0, g);        // y = g_new - g.
        x = std::move(ls.x_new);
        fx = ls.f_new;
        std::vector<double> g_next = std::move(ls.g_new);
        const double snorm = norm2(s);
        bfgs_inverse_update(hinv, s, y, n);  // no-op when curvature s.y <= 0.
        g = std::move(g_next);
        if (snorm <= opts.step_tol) {
            return finish(std::move(x), fx, iter + 1, true, norm2(g));
        }
    }
    return finish(std::move(x), fx, iter, false, norm2(g));
}

// --- L-BFGS -----------------------------------------------------------------

auto l_bfgs(Objective f, std::span<const double> x0, Gradient grad, Options opts)
    -> Result<OptimizeResult> {
    auto start = validate_start(f, &grad, x0);
    if (!start) {
        return make_error<OptimizeResult>(start.error());
    }
    std::vector<double> x = std::move(*start);
    const std::size_t n = x.size();
    const Gradient g_of = effective_gradient(f, grad, opts.fd_step);
    const std::size_t mem = std::max<std::size_t>(1, opts.lbfgs_memory);

    std::deque<std::vector<double>> s_hist;
    std::deque<std::vector<double>> y_hist;
    std::deque<double> rho_hist;

    double fx = f(x);
    std::vector<double> g = g_of(x);
    std::size_t iter = 0;
    for (; iter < opts.max_iterations; ++iter) {
        const double gnorm = norm2(g);
        if (gnorm <= opts.grad_tol) {
            return finish(std::move(x), fx, iter, true, gnorm);
        }
        // Two-loop recursion to compute p = -H_k g.
        std::vector<double> q = g;
        const std::size_t k = s_hist.size();
        std::vector<double> alpha(k, 0.0);
        for (std::size_t idx = k; idx-- > 0;) {  // newest -> oldest.
            const double a = rho_hist[idx] * dot(s_hist[idx], q);
            alpha[idx] = a;
            for (std::size_t i = 0; i < n; ++i) {
                q[i] = std::fma(-a, y_hist[idx][i], q[i]);
            }
        }
        double gamma = 1.0;  // H0 scaling = (s.y)/(y.y) from the most recent pair.
        if (k > 0) {
            const double yy = dot(y_hist.back(), y_hist.back());
            const double sy = dot(s_hist.back(), y_hist.back());
            if (yy > 0.0) {
                gamma = sy / yy;
            }
        }
        std::vector<double> r(n);
        for (std::size_t i = 0; i < n; ++i) {
            r[i] = gamma * q[i];
        }
        for (std::size_t idx = 0; idx < k; ++idx) {  // oldest -> newest.
            const double beta = rho_hist[idx] * dot(y_hist[idx], r);
            const double diff = alpha[idx] - beta;
            for (std::size_t i = 0; i < n; ++i) {
                r[i] = std::fma(diff, s_hist[idx][i], r[i]);
            }
        }
        std::vector<double> p(n);
        for (std::size_t i = 0; i < n; ++i) {
            p[i] = -r[i];
        }
        if (!(dot(g, p) < 0.0)) {
            for (std::size_t i = 0; i < n; ++i) {
                p[i] = -g[i];  // safeguard.
            }
        }
        auto ls = strong_wolfe(f, g_of, x, fx, g, p, opts);
        if (!ls.ok) {
            ls = armijo_backtracking(f, x, fx, g, p, opts);
            if (!ls.ok) {
                return finish(std::move(x), fx, iter, false, gnorm);
            }
            ls.g_new = g_of(ls.x_new);
        }
        std::vector<double> s = axpy(ls.x_new, -1.0, x);
        std::vector<double> y = axpy(ls.g_new, -1.0, g);
        const double sy = dot(s, y);
        const double snorm = norm2(s);
        x = std::move(ls.x_new);
        fx = ls.f_new;
        std::vector<double> g_next = std::move(ls.g_new);

        if (sy > 1e-12 * std::max(1.0, norm2(s) * norm2(y))) {
            if (s_hist.size() == mem) {
                s_hist.pop_front();
                y_hist.pop_front();
                rho_hist.pop_front();
            }
            rho_hist.push_back(1.0 / sy);
            s_hist.push_back(std::move(s));
            y_hist.push_back(std::move(y));
        }
        g = std::move(g_next);
        if (snorm <= opts.step_tol) {
            return finish(std::move(x), fx, iter + 1, true, norm2(g));
        }
    }
    return finish(std::move(x), fx, iter, false, norm2(g));
}

// --- nonlinear conjugate gradient -------------------------------------------

auto conjugate_gradient(Objective f, std::span<const double> x0, Gradient grad,
                        CGVariant variant, Options opts) -> Result<OptimizeResult> {
    auto start = validate_start(f, &grad, x0);
    if (!start) {
        return make_error<OptimizeResult>(start.error());
    }
    std::vector<double> x = std::move(*start);
    const std::size_t n = x.size();
    const Gradient g_of = effective_gradient(f, grad, opts.fd_step);

    double fx = f(x);
    std::vector<double> g = g_of(x);
    std::vector<double> d(n);
    for (std::size_t i = 0; i < n; ++i) {
        d[i] = -g[i];  // first direction: steepest descent.
    }
    std::size_t iter = 0;
    for (; iter < opts.max_iterations; ++iter) {
        const double gnorm = norm2(g);
        if (gnorm <= opts.grad_tol) {
            return finish(std::move(x), fx, iter, true, gnorm);
        }
        auto ls = strong_wolfe(f, g_of, x, fx, g, d, opts);
        if (!ls.ok) {
            // Restart along steepest descent and retry once.
            for (std::size_t i = 0; i < n; ++i) {
                d[i] = -g[i];
            }
            ls = strong_wolfe(f, g_of, x, fx, g, d, opts);
            if (!ls.ok) {
                ls = armijo_backtracking(f, x, fx, g, d, opts);
                if (!ls.ok) {
                    return finish(std::move(x), fx, iter, false, gnorm);
                }
                ls.g_new = g_of(ls.x_new);
            }
        }
        std::vector<double> s = axpy(ls.x_new, -1.0, x);
        const double snorm = norm2(s);
        std::vector<double> g_new = std::move(ls.g_new);
        x = std::move(ls.x_new);
        fx = ls.f_new;

        const double gg_old = dot(g, g);
        double beta = 0.0;
        if (gg_old > 0.0) {
            if (variant == CGVariant::fletcher_reeves) {
                beta = dot(g_new, g_new) / gg_old;
            } else {
                std::vector<double> yk = axpy(g_new, -1.0, g);  // g_new - g.
                beta = std::max(0.0, dot(g_new, yk) / gg_old);  // PR+ (>= 0).
            }
        }
        for (std::size_t i = 0; i < n; ++i) {
            d[i] = std::fma(beta, d[i], -g_new[i]);  // d = -g_new + beta d.
        }
        g = std::move(g_new);
        // Restart every n steps, or whenever d is not a descent direction.
        if ((iter + 1) % n == 0 || !(dot(g, d) < 0.0)) {
            for (std::size_t i = 0; i < n; ++i) {
                d[i] = -g[i];
            }
        }
        if (snorm <= opts.step_tol) {
            return finish(std::move(x), fx, iter + 1, true, norm2(g));
        }
    }
    return finish(std::move(x), fx, iter, false, norm2(g));
}

// --- Nelder-Mead (derivative-free) ------------------------------------------

auto nelder_mead(Objective f, std::span<const double> x0, Options opts)
    -> Result<OptimizeResult> {
    auto start = validate_start(f, nullptr, x0);
    if (!start) {
        return make_error<OptimizeResult>(start.error());
    }
    const std::vector<double> x_start = std::move(*start);
    const std::size_t n = x_start.size();

    // Objective wrapper: non-finite values become +inf so they sort as "worst".
    auto eval = [&](std::span<const double> v) -> double {
        const double val = f(v);
        return std::isfinite(val) ? val : std::numeric_limits<double>::infinity();
    };

    // Seed an (n+1)-vertex simplex: x0 plus one per-axis perturbation.
    std::vector<std::vector<double>> verts;
    verts.reserve(n + 1);
    std::vector<double> fval;
    fval.reserve(n + 1);
    verts.push_back(x_start);
    fval.push_back(eval(x_start));
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> v = x_start;
        const double delta = (v[i] != 0.0) ? 0.05 * v[i] : 0.00025;  // standard NM seeding.
        v[i] += delta;
        fval.push_back(eval(v));
        verts.push_back(std::move(v));
    }

    auto order = [&]() -> std::vector<std::size_t> {
        std::vector<std::size_t> idx(n + 1);
        std::iota(idx.begin(), idx.end(), std::size_t{0});
        std::ranges::sort(idx, [&](std::size_t a, std::size_t b) { return fval[a] < fval[b]; });
        return idx;
    };

    std::size_t iter = 0;
    bool converged = false;
    for (; iter < opts.max_iterations; ++iter) {
        std::vector<std::size_t> idx = order();
        const std::size_t best = idx.front();
        const std::size_t worst = idx.back();
        const std::size_t second_worst = idx[idx.size() - 2];

        // Convergence: relative spread of objective values across the simplex.
        const double fb = fval[best];
        const double fw = fval[worst];
        const double spread = std::abs(fw - fb) /
                              (0.5 * (std::abs(fw) + std::abs(fb)) + 1e-300);
        if (spread <= opts.grad_tol) {
            converged = true;
            break;
        }

        // Centroid of all vertices except the worst.
        std::vector<double> centroid(n, 0.0);
        for (std::size_t v = 0; v < n + 1; ++v) {
            if (v == worst) {
                continue;
            }
            for (std::size_t i = 0; i < n; ++i) {
                centroid[i] += verts[v][i];
            }
        }
        for (std::size_t i = 0; i < n; ++i) {
            centroid[i] /= static_cast<double>(n);
        }

        // Reflection: xr = centroid + alpha (centroid - worst).
        std::vector<double> dir = axpy(centroid, -1.0, verts[worst]);  // centroid - worst.
        std::vector<double> xr = axpy(centroid, opts.nm_reflect, dir);
        const double fr = eval(xr);

        if (fr < fval[best]) {
            // Expansion.
            std::vector<double> xe = axpy(centroid, opts.nm_expand, dir);
            const double fe = eval(xe);
            if (fe < fr) {
                verts[worst] = std::move(xe);
                fval[worst] = fe;
            } else {
                verts[worst] = std::move(xr);
                fval[worst] = fr;
            }
        } else if (fr < fval[second_worst]) {
            verts[worst] = std::move(xr);
            fval[worst] = fr;
        } else {
            // Contraction.
            bool accept = false;
            std::vector<double> xc;
            double fc = 0.0;
            if (fr < fval[worst]) {
                // Outside contraction toward the reflected point.
                xc = axpy(centroid, opts.nm_contract, dir);
                fc = eval(xc);
                accept = fc <= fr;
            } else {
                // Inside contraction toward the worst vertex.
                std::vector<double> inward = axpy(verts[worst], -1.0, centroid);  // worst-cent.
                xc = axpy(centroid, opts.nm_contract, inward);
                fc = eval(xc);
                accept = fc < fval[worst];
            }
            if (accept) {
                verts[worst] = std::move(xc);
                fval[worst] = fc;
            } else {
                // Shrink every vertex toward the best.
                for (std::size_t v = 0; v < n + 1; ++v) {
                    if (v == best) {
                        continue;
                    }
                    for (std::size_t i = 0; i < n; ++i) {
                        verts[v][i] = verts[best][i] +
                                      opts.nm_shrink * (verts[v][i] - verts[best][i]);
                    }
                    fval[v] = eval(verts[v]);
                }
            }
        }
    }

    // Return the best vertex. grad_norm is reported 0 (derivative-free).
    std::vector<std::size_t> idx = order();
    const std::size_t best = idx.front();
    double best_f = fval[best];
    if (!std::isfinite(best_f)) {
        best_f = f(verts[best]);  // re-evaluate to report the true (finite) value if any.
    }
    return finish(std::move(verts[best]), best_f, iter, converged, 0.0);
}

// --- implicit filtering (Kelley; derivative-free, noise-aware) ---------------

auto implicit_filtering(Objective f, std::span<const double> x0, std::span<const double> lo,
                        std::span<const double> hi, Options opts) -> Result<OptimizeResult> {
    // Validate x0 and the objective at x0.
    auto start = validate_start(f, nullptr, x0);
    if (!start) {
        return make_error<OptimizeResult>(start.error());
    }
    const std::size_t n = x0.size();

    // Validate optional bounds: both present with matching length and lo <= hi, or both
    // absent. A partial or ragged or inverted box is a domain_error.
    const bool bounded = !lo.empty() || !hi.empty();
    if (bounded) {
        if (lo.size() != n || hi.size() != n || !all_finite(lo) || !all_finite(hi)) {
            return make_error<OptimizeResult>(MathError::domain_error);
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (lo[i] > hi[i]) {
                return make_error<OptimizeResult>(MathError::domain_error);
            }
        }
    }

    std::vector<double> x = project(*start, lo, hi);  // start feasible.
    double fx = f(x);
    if (!std::isfinite(fx)) {
        return make_error<OptimizeResult>(MathError::domain_error);
    }

    // Inverse-Hessian approximation, reset to identity at each scale change.
    std::vector<double> hinv(n * n, 0.0);
    auto reset_hinv = [&] {
        std::ranges::fill(hinv, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            hinv[i * n + i] = 1.0;
        }
    };
    reset_hinv();

    const double h_min = std::max(opts.imf_h_min, 0.0);
    double h = std::max(opts.imf_h0, h_min);
    bool converged = false;
    std::size_t iter = 0;

    while (iter < opts.max_iterations) {
        std::vector<double> g = simplex_gradient(f, x, h, lo, hi);
        const double crit = projected_grad_norm(x, g, lo, hi);
        if (crit <= opts.grad_tol) {
            if (h <= h_min) {
                converged = true;
                break;              // stationary at the finest scale.
            }
            h *= 0.5;               // refine: shrink the scale and reset the model.
            reset_hinv();
            ++iter;                 // count refinements against the hard max_iterations cap
            if (!std::isfinite(h)) {
                break;              // degenerate scale (e.g. non-finite imf_h0) -> stop safely
            }
            continue;
        }

        // Quasi-Newton direction p = -H g, with a steepest-descent safeguard.
        std::vector<double> p(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            double acc = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                acc = std::fma(hinv[i * n + j], g[j], acc);
            }
            p[i] = -acc;
        }
        if (!(dot(g, p) < 0.0)) {
            for (std::size_t i = 0; i < n; ++i) {
                p[i] = -g[i];
            }
        }

        // Projected Armijo line search. The step actually taken is proj(x+alpha p), so
        // sufficient decrease is measured against the model along that projected step:
        //   f(x_trial) < f(x) + c1 * (g . (x_trial - x)).
        // The strict inequality + the scale-h simplex gradient is the noise filter: a
        // step is accepted only when it beats the decrease the scale-h model predicts.
        bool productive = false;
        double alpha = 1.0;
        std::vector<double> x_trial;
        double f_trial = 0.0;
        std::vector<double> trial_step;
        for (std::size_t k = 0; k < opts.max_line_search; ++k) {
            std::vector<double> cand = project(axpy(x, alpha, p), lo, hi);
            const double fc = f(cand);
            std::vector<double> step = axpy(cand, -1.0, x);  // proj(x+ap) - x.
            if (std::isfinite(fc) && fc < std::fma(opts.armijo_c1, dot(g, step), fx)) {
                productive = true;
                x_trial = std::move(cand);
                f_trial = fc;
                trial_step = std::move(step);
                break;
            }
            alpha *= opts.backtrack_rho;
        }

        if (!productive) {
            // Stencil failure: no productive step at this scale.
            if (h <= h_min) {
                break;              // exhausted the schedule; converged stays false.
            }
            h *= 0.5;
            reset_hinv();
            ++iter;
            continue;
        }

        // Accept the step; update the BFGS model with a same-scale gradient at x_trial.
        std::vector<double> g_new = simplex_gradient(f, x_trial, h, lo, hi);
        std::vector<double> y = axpy(g_new, -1.0, g);  // y = g_new - g.
        const double snorm = norm2(trial_step);
        bfgs_inverse_update(hinv, trial_step, y, n);
        x = std::move(x_trial);
        fx = f_trial;
        ++iter;
        if (snorm <= opts.step_tol) {
            converged = true;
            break;
        }
    }

    const double final_crit =
        projected_grad_norm(x, simplex_gradient(f, x, std::max(h, h_min), lo, hi), lo, hi);
    return finish(std::move(x), fx, iter, converged, final_crit);
}

// --- parallel + distributed multistart --------------------------------------
namespace {

// Canonical multistart order. Returns true iff candidate (cf, ci) should REPLACE the
// incumbent (bf, bi): a strictly lower fx wins; on an exact fx tie the SMALLER start
// index wins. A non-finite candidate fx never displaces a (finite) incumbent — the
// comparisons below are all false when cf is NaN.
[[nodiscard]] auto better_candidate(double cf, std::size_t ci, double bf,
                                    std::size_t bi) noexcept -> bool {
    if (cf < bf) {
        return true;
    }
    if (cf > bf) {
        return false;
    }
    if (cf == bf) {  // exact tie (excludes NaN, for which == is false): break by index.
        return ci < bi;
    }
    return false;  // any NaN involved: keep the incumbent.
}

// Deterministic in-order argmin over per-start results. `global_index[j]` is the global
// start index of results[j] (strictly increasing at every call site, so equal-fx ties
// resolve to the earliest start). Skips error results; on none-valid, propagates the
// first (lowest-slot) error, or domain_error if there were no results at all.
[[nodiscard]] auto argmin_results(std::vector<Result<OptimizeResult>> results,
                                  std::span<const std::size_t> global_index)
    -> Result<MultistartResult> {
    std::optional<std::size_t> best_slot;
    std::size_t succeeded = 0;
    bool saw_error = false;
    MathError first_error = MathError::domain_error;
    for (std::size_t j = 0; j < results.size(); ++j) {
        if (results[j].has_value()) {
            ++succeeded;
            if (!best_slot.has_value() ||
                better_candidate(results[j]->fx, global_index[j], results[*best_slot]->fx,
                                 global_index[*best_slot])) {
                best_slot = j;
            }
        } else if (!saw_error) {
            saw_error = true;
            first_error = results[j].error();
        }
    }
    if (!best_slot.has_value()) {
        return make_error<MultistartResult>(saw_error ? first_error : MathError::domain_error);
    }
    return MultistartResult{std::move(*results[*best_slot]), global_index[*best_slot], succeeded};
}

}  // namespace

auto make_local_optimizer(Method method, Objective f, Gradient grad, HessianFn hess,
                          CGVariant cg_variant, Options opts) -> LocalOptimizer {
    return [method, f = std::move(f), grad = std::move(grad), hess = std::move(hess), cg_variant,
            opts](std::span<const double> x0) -> Result<OptimizeResult> {
        switch (method) {
            case Method::gradient_descent:
                return gradient_descent(f, x0, grad, opts);
            case Method::newton:
                return newton_method(f, x0, grad, hess, opts);
            case Method::bfgs:
                return bfgs(f, x0, grad, opts);
            case Method::l_bfgs:
                return l_bfgs(f, x0, grad, opts);
            case Method::conjugate_gradient:
                return conjugate_gradient(f, x0, grad, cg_variant, opts);
            case Method::nelder_mead:
                return nelder_mead(f, x0, opts);
        }
        return make_error<OptimizeResult>(MathError::not_implemented);  // unreachable.
    };
}

auto parallel_multistart(const LocalOptimizer& local,
                         std::span<const std::vector<double>> starts, bool parallel)
    -> Result<MultistartResult> {
    const std::size_t k = starts.size();
    if (k == 0 || !local) {
        return make_error<MultistartResult>(MathError::domain_error);
    }
    // Order-preserving, deterministic parallel map: one independent local run per start.
    // grain 1 (via transform_index_if) so even a handful of expensive starts fan out.
    std::vector<Result<OptimizeResult>> results = nimblecas::parallel::transform_index_if(
        parallel, k, [&](std::size_t i) -> Result<OptimizeResult> {
            return local(std::span<const double>(starts[i]));
        });
    std::vector<std::size_t> global_index(k);
    std::iota(global_index.begin(), global_index.end(), std::size_t{0});
    return argmin_results(std::move(results), global_index);
}

auto multistart(Method method, Objective f, std::span<const std::vector<double>> starts,
                Gradient grad, HessianFn hess, CGVariant cg_variant, Options opts, bool parallel)
    -> Result<MultistartResult> {
    LocalOptimizer local = make_local_optimizer(method, std::move(f), std::move(grad),
                                                std::move(hess), cg_variant, opts);
    return parallel_multistart(local, starts, parallel);
}

auto multistart_shard(const LocalOptimizer& local, std::span<const std::vector<double>> starts,
                      std::size_t shard_index, std::size_t num_shards, bool parallel)
    -> Result<MultistartResult> {
    if (num_shards == 0 || shard_index >= num_shards || starts.empty() || !local) {
        return make_error<MultistartResult>(MathError::domain_error);
    }
    // Global indices assigned to this shard: i = shard_index, shard_index+num_shards, ...
    // (exactly the i with i % num_shards == shard_index), kept in increasing order.
    std::vector<std::size_t> assigned;
    for (std::size_t i = shard_index; i < starts.size(); i += num_shards) {
        assigned.push_back(i);
    }
    if (assigned.empty()) {
        return make_error<MultistartResult>(MathError::domain_error);  // no work for this shard.
    }
    std::vector<Result<OptimizeResult>> results = nimblecas::parallel::transform_index_if(
        parallel, assigned.size(), [&](std::size_t j) -> Result<OptimizeResult> {
            return local(std::span<const double>(starts[assigned[j]]));
        });
    return argmin_results(std::move(results), assigned);
}

auto reduce_shards(std::span<const MultistartResult> shard_results) -> Result<MultistartResult> {
    if (shard_results.empty()) {
        return make_error<MultistartResult>(MathError::domain_error);
    }
    std::optional<std::size_t> best;
    std::size_t total_succeeded = 0;
    for (std::size_t k = 0; k < shard_results.size(); ++k) {
        total_succeeded += shard_results[k].succeeded;
        if (!best.has_value() ||
            better_candidate(shard_results[k].best.fx, shard_results[k].best_start,
                             shard_results[*best].best.fx, shard_results[*best].best_start)) {
            best = k;
        }
    }
    MultistartResult out = shard_results[*best];  // copy the winning shard's best.
    out.succeeded = total_succeeded;              // aggregate across shards.
    return out;
}

}  // namespace nimblecas::optimize
