// NimbleCAS nlsolve module: iterative methods for nonlinear systems F(x)=0 in R^n
// (C.T. Kelley, "Iterative Methods for Linear and Nonlinear Equations" / "Solving
// Nonlinear Equations with Newton's Method"). ROADMAP §7.14 — the numeric counterpart
// to nimblecas.optimize (minimization).
// @author Olumuyiwa Oluwasanmi
//
// Conforms to config/cpp_details.txt: C++23 modules, `import std`, trailing return
// types, no owning raw pointers, std::expected error handling (no exceptions),
// [[nodiscard]] throughout.
//
// ---------------------------------------------------------------------------
// HONESTY BOUNDARY — read before trusting a result.
// ---------------------------------------------------------------------------
// This is a NUMERIC module: every quantity is an IEEE-754 double and every method is
// LOCALLY convergent. Concretely:
//   * Newton's method is LOCALLY quadratically convergent near a SIMPLE root (nonsingular
//     Jacobian); it is NOT globally convergent. From a poor initial guess plain Newton
//     can diverge — the Armijo line search (opts.line_search, on by default) is what
//     turns most such runs into progress, and even it can stall.
//   * chord / Shamanskii / Broyden / Newton-Krylov (JFNK) are APPROXIMATE Newton
//     variants: they trade Jacobian accuracy for cost and are only locally convergent,
//     typically superlinearly (or linearly for the chord method).
//   * Finite-difference Jacobians and Jacobian-vector products add TRUNCATION error on
//     top of rounding; an analytic Jacobian is always more accurate when available.
//   * Anderson acceleration accelerates a fixed-point map g(x)=x; it is a heuristic with
//     no global guarantee and can diverge if the map is not (locally) contractive.
//   * Levenberg-Marquardt finds a STATIONARY point of ||F(x)||^2, which need not be a
//     global minimiser and need not have F=0.
// Results depend on the initial guess. NO exactness and NO global-convergence claims are
// made anywhere. Failure policy:
//   * Non-finite inputs (x0), or F/g/Jacobian returning a wrong-dimension or non-finite
//     value at the START point, or an empty system -> MathError::domain_error.
//   * Running out of iterations, stagnation (line-search collapse), or a singular
//     Jacobian encountered mid-iteration -> a VALID Result carrying converged=false
//     (NOT an error): the best iterate found is returned with its residual norm.
//
// Self-contained: this module implements its own dense Gaussian-elimination linear
// solver and a compact restarted GMRES(m); it depends only on nimblecas.core.

export module nimblecas.nlsolve;

import std;
import nimblecas.core;
import nimblecas.parallel;  // deterministic fork-join runtime (PPL/TBB/serial)

// ===========================================================================
// Public types.
// ===========================================================================
export namespace nimblecas::nlsolve {

// A residual map F : R^n -> R^m, evaluated at the point given as a span. For the
// square-system solvers (newton, chord, shamanskii, broyden, newton_krylov) m == n.
// For a fixed-point map (anderson) it returns g(x) with m == n. For least squares
// (levenberg_marquardt) m may exceed n. The callable MUST NOT retain the span.
using ResidualFn = std::function<std::vector<double>(std::span<const double>)>;

// An analytic Jacobian J : R^n -> R^{m x n}, returned DENSE and ROW-MAJOR: entry
// (i, j) = dF_i/dx_j lives at index i*n + j, so the returned vector has size m*n.
using JacobianFn = std::function<std::vector<double>(std::span<const double>)>;

// Outcome of a solve. `x` is the best iterate found (always populated on a non-error
// return, even when converged == false). `residual_norm` is ||F(x)||_2 at that iterate.
struct SolveResult {
    std::vector<double> x{};            // best iterate located
    double residual_norm{0.0};          // ||F(x)||_2 there
    std::size_t iterations{0};          // outer iterations performed
    bool converged{false};              // did the stopping test pass?
    std::size_t function_evals{0};      // number of F/g evaluations (diagnostics)
};

// Shared tuning knobs. Sensible defaults are provided; the globalisation (Armijo line
// search) is ON by default because plain Newton is not globally convergent.
struct Options {
    double tol{1e-10};                  // absolute stopping tolerance on ||F||_2
    std::size_t max_iter{100};          // maximum outer iterations
    double fd_step{1e-7};               // base finite-difference step (relative-scaled)
    bool central_diff{false};           // central (2 evals/col) vs forward FD Jacobian
    bool line_search{true};             // Armijo backtracking on ||F|| (globalisation)
    double armijo_alpha{1e-4};          // sufficient-decrease parameter
    double armijo_reduction{0.5};       // backtracking step-length factor in (0,1)
    std::size_t max_backtrack{30};      // maximum backtracking halvings per step
};

// Which Broyden rank-1 update to apply to the approximate inverse Jacobian.
enum class BroydenVariant : std::uint8_t { good, bad };

}  // namespace nimblecas::nlsolve

// ===========================================================================
// Internal helpers (module-local; NOT exported).
// ===========================================================================
namespace nimblecas::nlsolve::detail {

using Vec = std::vector<double>;
// A matrix-free linear operator y = A x that may fail (e.g. a non-finite evaluation);
// used by GMRES so the Jacobian-vector product need never be materialised.
using LinOp = std::function<Result<Vec>(std::span<const double>)>;

[[nodiscard]] auto all_finite(std::span<const double> v) noexcept -> bool {
    for (double e : v) {
        if (!std::isfinite(e)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] auto dot(std::span<const double> a, std::span<const double> b) noexcept -> double {
    double acc = 0.0;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        acc = std::fma(a[i], b[i], acc);
    }
    return acc;
}

[[nodiscard]] auto norm2(std::span<const double> v) noexcept -> double {
    return std::sqrt(dot(v, v));
}

[[nodiscard]] auto sub(std::span<const double> a, std::span<const double> b) -> Vec {
    const std::size_t n = a.size();
    Vec out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = a[i] - b[i];
    }
    return out;
}

[[nodiscard]] auto scale(std::span<const double> v, double s) -> Vec {
    Vec out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        out[i] = v[i] * s;
    }
    return out;
}

// y <- y + alpha*x  (in place; sizes must match).
auto axpy(Vec& y, double alpha, std::span<const double> x) -> void {
    for (std::size_t i = 0; i < y.size(); ++i) {
        y[i] = std::fma(alpha, x[i], y[i]);
    }
}

// x + lambda*s as a fresh vector.
[[nodiscard]] auto step_point(std::span<const double> x, double lambda,
                              std::span<const double> s) -> Vec {
    Vec out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        out[i] = std::fma(lambda, s[i], x[i]);
    }
    return out;
}

// Evaluate F(x), requiring the result to have exactly `expect` entries and be finite.
// A dimension mismatch or non-finite value is a hard domain_error (the caller controls
// whether that is fatal or merely a mid-iteration breakdown).
[[nodiscard]] auto eval_checked(const ResidualFn& F, std::span<const double> x,
                                std::size_t expect, std::size_t& fevals) -> Result<Vec> {
    Vec y = F(x);
    ++fevals;
    if (y.size() != expect) {
        return make_error<Vec>(MathError::domain_error);
    }
    if (!all_finite(y)) {
        return make_error<Vec>(MathError::domain_error);
    }
    return y;
}

// Dense forward/central finite-difference Jacobian, ROW-MAJOR m x n, where m = |F(x)|
// and n = |x|. Column j perturbs x_j by h_j = step*(1 + |x_j|) (a relative step that
// avoids a zero increment at x_j == 0). Fx is the (already computed) value F(x), reused
// for the forward difference. Truncation error is O(h) forward, O(h^2) central.
[[nodiscard]] auto fd_jacobian(const ResidualFn& F, std::span<const double> x,
                               std::span<const double> Fx, double step, bool central,
                               std::size_t& fevals) -> Result<Vec> {
    const std::size_t n = x.size();
    const std::size_t m = Fx.size();
    Vec J(m * n, 0.0);
    Vec xp(x.begin(), x.end());
    for (std::size_t j = 0; j < n; ++j) {
        const double h = step * (1.0 + std::abs(x[j]));
        const double xj = x[j];
        if (central) {
            xp[j] = xj + h;
            auto fp = eval_checked(F, xp, m, fevals);
            if (!fp) {
                return make_error<Vec>(fp.error());
            }
            xp[j] = xj - h;
            auto fm = eval_checked(F, xp, m, fevals);
            if (!fm) {
                return make_error<Vec>(fm.error());
            }
            const double inv = 1.0 / (2.0 * h);
            for (std::size_t i = 0; i < m; ++i) {
                J[i * n + j] = ((*fp)[i] - (*fm)[i]) * inv;
            }
        } else {
            xp[j] = xj + h;
            auto fp = eval_checked(F, xp, m, fevals);
            if (!fp) {
                return make_error<Vec>(fp.error());
            }
            const double inv = 1.0 / h;
            for (std::size_t i = 0; i < m; ++i) {
                J[i * n + j] = ((*fp)[i] - Fx[i]) * inv;
            }
        }
        xp[j] = xj;  // restore for the next column
    }
    return J;
}

// Matrix-free Jacobian-vector product Jv ~= (F(x + eps v) - Fx)/eps with Kelley's step
// eps = sqrt(u) (1 + ||x||) / ||v||, u = machine epsilon. Returns 0 for v == 0.
[[nodiscard]] auto jv_product(const ResidualFn& F, std::span<const double> x,
                              std::span<const double> Fx, std::span<const double> v,
                              std::size_t& fevals) -> Result<Vec> {
    const std::size_t m = Fx.size();
    const double vn = norm2(v);
    if (vn == 0.0) {
        return Vec(m, 0.0);
    }
    static const double sqrt_eps = std::sqrt(std::numeric_limits<double>::epsilon());
    const double eps = sqrt_eps * (1.0 + norm2(x)) / vn;
    const Vec xp = step_point(x, eps, v);
    auto fp = eval_checked(F, xp, m, fevals);
    if (!fp) {
        return make_error<Vec>(fp.error());
    }
    Vec jv = sub(*fp, Fx);
    const double inv = 1.0 / eps;
    for (double& e : jv) {
        e *= inv;
    }
    return jv;
}

// Solve the dense n x n system A x = b by Gaussian elimination with partial pivoting.
// A and b are taken by value (destroyed in place). A zero pivot column (structurally
// singular) or a non-finite pivot yields domain_error, which callers translate into a
// converged=false result rather than a hard failure.
[[nodiscard]] auto lu_solve(Vec A, Vec b, std::size_t n) -> Result<Vec> {
    for (std::size_t col = 0; col < n; ++col) {
        // Partial pivot: pick the row with the largest magnitude in this column.
        std::size_t piv = col;
        double best = std::abs(A[col * n + col]);
        for (std::size_t r = col + 1; r < n; ++r) {
            const double v = std::abs(A[r * n + col]);
            if (v > best) {
                best = v;
                piv = r;
            }
        }
        if (!(best > 0.0) || !std::isfinite(best)) {
            return make_error<Vec>(MathError::domain_error);  // singular
        }
        if (piv != col) {
            for (std::size_t j = col; j < n; ++j) {
                std::swap(A[col * n + j], A[piv * n + j]);
            }
            std::swap(b[col], b[piv]);
        }
        const double pivot = A[col * n + col];
        for (std::size_t r = col + 1; r < n; ++r) {
            const double factor = A[r * n + col] / pivot;
            if (factor == 0.0) {
                continue;
            }
            for (std::size_t j = col; j < n; ++j) {
                A[r * n + j] = std::fma(-factor, A[col * n + j], A[r * n + j]);
            }
            b[r] = std::fma(-factor, b[col], b[r]);
        }
    }
    // Back-substitution.
    Vec x(n, 0.0);
    for (std::size_t i = n; i-- > 0;) {
        double s = b[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            s = std::fma(-A[i * n + j], x[j], s);
        }
        const double diag = A[i * n + i];
        if (!std::isfinite(diag) || diag == 0.0) {
            return make_error<Vec>(MathError::domain_error);
        }
        x[i] = s / diag;
    }
    if (!all_finite(x)) {
        return make_error<Vec>(MathError::domain_error);
    }
    return x;
}

// Dense inverse of an n x n matrix (row-major) via n right-hand-side solves. Returns
// nullopt when the matrix is singular; callers fall back to a scaled identity.
[[nodiscard]] auto invert(const Vec& A, std::size_t n) -> std::optional<Vec> {
    Vec inv(n * n, 0.0);
    for (std::size_t k = 0; k < n; ++k) {
        Vec e(n, 0.0);
        e[k] = 1.0;
        auto col = lu_solve(A, std::move(e), n);
        if (!col) {
            return std::nullopt;
        }
        for (std::size_t i = 0; i < n; ++i) {
            inv[i * n + k] = (*col)[i];
        }
    }
    return inv;
}

// y = M v for a dense row-major n x n matrix M.
[[nodiscard]] auto matvec(const Vec& M, std::span<const double> v, std::size_t n) -> Vec {
    Vec y(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double acc = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            acc = std::fma(M[i * n + j], v[j], acc);
        }
        y[i] = acc;
    }
    return y;
}

// Result of an Armijo backtracking line search along the direction s from x.
struct LineSearch {
    Vec x_new{};        // accepted point x + lambda*s
    Vec F_new{};        // F(x_new)
    double fnorm{0.0};  // ||F(x_new)||_2
    double lambda{0.0}; // accepted step length
    bool ok{false};     // did we find an acceptable (finite / decreasing) step?
};

// Armijo backtracking on the residual norm: accept the first lambda in {1, r, r^2, ...}
// with ||F(x + lambda s)|| <= (1 - alpha*lambda) ||F(x)||. When opts.line_search is off
// this degenerates to the full undamped Newton step (lambda == 1), accepted as long as
// F(x + s) is finite — that is exactly the globalisation-free behaviour that lets plain
// Newton diverge from a poor start.
[[nodiscard]] auto line_search(const ResidualFn& F, std::span<const double> x,
                               std::span<const double> s, double fnorm, std::size_t m,
                               const Options& opts, std::size_t& fevals) -> LineSearch {
    LineSearch out;
    if (!opts.line_search) {
        Vec xn = step_point(x, 1.0, s);
        if (!all_finite(xn)) {
            return out;  // ok == false: diverged off to non-finite
        }
        Vec fn = F(xn);
        ++fevals;
        if (fn.size() != m || !all_finite(fn)) {
            return out;
        }
        out.x_new = std::move(xn);
        out.fnorm = norm2(fn);
        out.F_new = std::move(fn);
        out.lambda = 1.0;
        out.ok = true;
        return out;
    }
    double lambda = 1.0;
    for (std::size_t bt = 0; bt <= opts.max_backtrack; ++bt) {
        Vec xn = step_point(x, lambda, s);
        if (all_finite(xn)) {
            Vec fn = F(xn);
            ++fevals;
            if (fn.size() == m && all_finite(fn)) {
                const double fnn = norm2(fn);
                if (fnn <= (1.0 - opts.armijo_alpha * lambda) * fnorm) {
                    out.x_new = std::move(xn);
                    out.fnorm = fnn;
                    out.F_new = std::move(fn);
                    out.lambda = lambda;
                    out.ok = true;
                    return out;
                }
            }
        }
        lambda *= opts.armijo_reduction;
    }
    return out;  // ok == false: could not achieve sufficient decrease
}

// Compact restarted GMRES(m) for A x = b with a matrix-free operator, no preconditioner,
// x0 = 0. Modified Gram-Schmidt Arnoldi + Givens rotations. `rtol` is relative to ||b||.
// Returns the best iterate found even if the inner tolerance was not met (an inexact
// Newton step is acceptable). Propagates an operator failure as its error.
[[nodiscard]] auto gmres(const LinOp& A, const Vec& b, std::size_t m, double rtol,
                         std::size_t max_restarts) -> Result<Vec> {
    const std::size_t n = b.size();
    Vec x(n, 0.0);
    const double bnorm = norm2(b);
    if (bnorm == 0.0 || m == 0) {
        return x;
    }
    const double target = rtol * bnorm;
    for (std::size_t restart = 0; restart <= max_restarts; ++restart) {
        Vec r;
        if (restart == 0) {
            r = b;  // since x == 0
        } else {
            auto Ax = A(x);
            if (!Ax) {
                return make_error<Vec>(Ax.error());
            }
            r = sub(b, *Ax);
        }
        double beta = norm2(r);
        if (beta <= target) {
            return x;
        }
        std::vector<Vec> V;
        V.reserve(m + 1);
        V.push_back(scale(r, 1.0 / beta));
        std::vector<Vec> H(m, Vec(m + 1, 0.0));  // H[col][row]
        Vec cs(m, 0.0);
        Vec sn(m, 0.0);
        Vec g(m + 1, 0.0);
        g[0] = beta;
        std::size_t kdim = 0;
        for (std::size_t k = 0; k < m; ++k) {
            auto w_ = A(V[k]);
            if (!w_) {
                return make_error<Vec>(w_.error());
            }
            Vec w = std::move(*w_);
            for (std::size_t i = 0; i <= k; ++i) {
                const double hik = dot(w, V[i]);
                H[k][i] = hik;
                axpy(w, -hik, V[i]);
            }
            const double hkp = norm2(w);
            H[k][k + 1] = hkp;
            const bool lucky = !(hkp > 1e-14 * beta);
            if (!lucky) {
                V.push_back(scale(w, 1.0 / hkp));
            }
            // Apply the previously computed Givens rotations to column k.
            for (std::size_t i = 0; i < k; ++i) {
                const double t = H[k][i];
                H[k][i] = cs[i] * t + sn[i] * H[k][i + 1];
                H[k][i + 1] = -sn[i] * t + cs[i] * H[k][i + 1];
            }
            // New rotation to annihilate the sub-diagonal H[k][k+1].
            const double h0 = H[k][k];
            const double h1 = H[k][k + 1];
            const double denom = std::hypot(h0, h1);
            if (denom == 0.0) {
                cs[k] = 1.0;
                sn[k] = 0.0;
            } else {
                cs[k] = h0 / denom;
                sn[k] = h1 / denom;
            }
            H[k][k] = cs[k] * h0 + sn[k] * h1;
            H[k][k + 1] = 0.0;
            const double g0 = g[k];
            g[k] = cs[k] * g0;
            g[k + 1] = -sn[k] * g0;
            kdim = k + 1;
            if (std::abs(g[k + 1]) <= target || lucky) {
                break;
            }
        }
        // Solve the kdim x kdim upper-triangular least-squares system R y = g.
        Vec y(kdim, 0.0);
        for (std::size_t i = kdim; i-- > 0;) {
            double s = g[i];
            for (std::size_t j = i + 1; j < kdim; ++j) {
                s = std::fma(-H[j][i], y[j], s);
            }
            const double diag = H[i][i];
            if (diag == 0.0) {
                break;  // breakdown: keep the partial update accumulated so far
            }
            y[i] = s / diag;
        }
        for (std::size_t j = 0; j < kdim; ++j) {
            axpy(x, y[j], V[j]);
        }
        auto Ax = A(x);
        if (!Ax) {
            return make_error<Vec>(Ax.error());
        }
        if (norm2(sub(b, *Ax)) <= target) {
            return x;
        }
    }
    return x;  // best inexact step
}

// A provider of the dense n x n Jacobian at (x, Fx): analytic wrappers ignore Fx, the
// finite-difference wrapper reuses it for the forward difference.
using JacProvider = std::function<Result<Vec>(std::span<const double>, std::span<const double>)>;

// Shared driver for the dense Newton family. `refresh` controls how often the Jacobian
// is recomputed: 1 = every step (Newton), 0 = only once (chord), m = every m steps
// (Shamanskii). The linear step J s = -F is solved by dense Gaussian elimination; a
// singular factorisation or a failed line search ends the run with converged=false.
[[nodiscard]] auto dense_family(const ResidualFn& F, const JacProvider& jac,
                                std::span<const double> x0, const Options& opts,
                                std::size_t refresh) -> Result<SolveResult> {
    const std::size_t n = x0.size();
    if (n == 0 || !all_finite(x0)) {
        return make_error<SolveResult>(MathError::domain_error);
    }
    std::size_t fevals = 0;
    Vec x(x0.begin(), x0.end());
    auto Fx_ = eval_checked(F, x, n, fevals);
    if (!Fx_) {
        return make_error<SolveResult>(Fx_.error());
    }
    Vec Fx = std::move(*Fx_);
    double fnorm = norm2(Fx);
    Vec J;
    bool have_J = false;
    for (std::size_t it = 0; it < opts.max_iter; ++it) {
        if (fnorm <= opts.tol) {
            return SolveResult{std::move(x), fnorm, it, true, fevals};
        }
        const bool refresh_now = refresh >= 1 && (it % refresh == 0);
        if (!have_J || refresh_now) {
            auto Jr = jac(x, Fx);
            if (!Jr) {
                // A singular/failed Jacobian at the start point is a domain_error; once
                // iterating it is a breakdown -> stop with the best iterate so far.
                if (!have_J) {
                    return make_error<SolveResult>(Jr.error());
                }
                return SolveResult{std::move(x), fnorm, it, false, fevals};
            }
            J = std::move(*Jr);
            have_J = true;
        }
        Vec rhs = scale(Fx, -1.0);
        auto step = lu_solve(J, std::move(rhs), n);
        if (!step) {
            return SolveResult{std::move(x), fnorm, it, false, fevals};  // singular
        }
        auto ls = line_search(F, x, *step, fnorm, n, opts, fevals);
        if (!ls.ok) {
            return SolveResult{std::move(x), fnorm, it, false, fevals};  // stagnation
        }
        x = std::move(ls.x_new);
        Fx = std::move(ls.F_new);
        fnorm = ls.fnorm;
    }
    // The convergence test sits at the loop TOP, so an iterate that first meets tolerance on
    // the final permitted iteration would exit here — re-check the residual so it is reported
    // converged rather than falsely converged=false.
    return SolveResult{std::move(x), fnorm, opts.max_iter, fnorm <= opts.tol, fevals};
}

}  // namespace nimblecas::nlsolve::detail

// ===========================================================================
// Public API.
// ===========================================================================
export namespace nimblecas::nlsolve {

// --- finite-difference Jacobian (exposed for verification / callers) --------

// Dense forward (default) or central finite-difference Jacobian of F at x, ROW-MAJOR
// with m = |F(x)| rows and n = |x| columns. Non-finite x, or a non-finite / mismatched
// F evaluation, yields domain_error. Carries O(step) (forward) or O(step^2) (central)
// truncation error — see the module honesty note.
[[nodiscard]] auto finite_difference_jacobian(const ResidualFn& F, std::span<const double> x,
                                              double step = 1e-7, bool central = false)
    -> Result<std::vector<double>> {
    if (x.empty() || !detail::all_finite(x)) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    std::size_t fevals = 0;
    std::vector<double> Fx = F(x);
    ++fevals;
    if (Fx.empty() || !detail::all_finite(Fx)) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    return detail::fd_jacobian(F, x, Fx, step, central, fevals);
}

// --- Newton (full) ----------------------------------------------------------

// Newton's method with a finite-difference Jacobian. Locally quadratically convergent
// near a simple root; globalised by the Armijo line search (opts.line_search).
[[nodiscard]] auto newton(const ResidualFn& F, std::span<const double> x0,
                          const Options& opts = {}) -> Result<SolveResult> {
    detail::JacProvider jac = [&](std::span<const double> x,
                                  std::span<const double> Fx) -> Result<detail::Vec> {
        std::size_t local = 0;
        return detail::fd_jacobian(F, x, Fx, opts.fd_step, opts.central_diff, local);
    };
    // Finite-difference evaluations inside the provider are not threaded into the public
    // function_evals count (they are an implementation detail of the Jacobian estimate).
    return detail::dense_family(F, jac, x0, opts, /*refresh=*/1);
}

// Newton's method with a user-supplied analytic Jacobian (more accurate than FD).
[[nodiscard]] auto newton(const ResidualFn& F, const JacobianFn& J,
                          std::span<const double> x0, const Options& opts = {})
    -> Result<SolveResult> {
    const std::size_t n = x0.size();
    detail::JacProvider jac = [&, n](std::span<const double> x,
                                     std::span<const double>) -> Result<detail::Vec> {
        std::vector<double> j = J(x);
        if (j.size() != n * n || !detail::all_finite(j)) {
            return make_error<detail::Vec>(MathError::domain_error);
        }
        return j;
    };
    return detail::dense_family(F, jac, x0, opts, /*refresh=*/1);
}

// --- chord (frozen Jacobian) ------------------------------------------------

// The chord method: the Jacobian is formed ONCE at x0 and reused for every step. Cheap
// per iteration but only linearly convergent.
[[nodiscard]] auto chord(const ResidualFn& F, std::span<const double> x0,
                         const Options& opts = {}) -> Result<SolveResult> {
    detail::JacProvider jac = [&](std::span<const double> x,
                                  std::span<const double> Fx) -> Result<detail::Vec> {
        std::size_t local = 0;
        return detail::fd_jacobian(F, x, Fx, opts.fd_step, opts.central_diff, local);
    };
    return detail::dense_family(F, jac, x0, opts, /*refresh=*/0);
}

[[nodiscard]] auto chord(const ResidualFn& F, const JacobianFn& J,
                         std::span<const double> x0, const Options& opts = {})
    -> Result<SolveResult> {
    const std::size_t n = x0.size();
    detail::JacProvider jac = [&, n](std::span<const double> x,
                                     std::span<const double>) -> Result<detail::Vec> {
        std::vector<double> j = J(x);
        if (j.size() != n * n || !detail::all_finite(j)) {
            return make_error<detail::Vec>(MathError::domain_error);
        }
        return j;
    };
    return detail::dense_family(F, jac, x0, opts, /*refresh=*/0);
}

// --- Shamanskii (Jacobian reused for m steps) -------------------------------

// The Shamanskii method: the Jacobian is refreshed every `m` steps (m == 1 recovers
// Newton; m -> infinity recovers the chord method). Superlinear for small m > 1.
[[nodiscard]] auto shamanskii(const ResidualFn& F, std::span<const double> x0,
                              std::size_t m, const Options& opts = {})
    -> Result<SolveResult> {
    const std::size_t refresh = m == 0 ? 1 : m;
    detail::JacProvider jac = [&](std::span<const double> x,
                                  std::span<const double> Fx) -> Result<detail::Vec> {
        std::size_t local = 0;
        return detail::fd_jacobian(F, x, Fx, opts.fd_step, opts.central_diff, local);
    };
    return detail::dense_family(F, jac, x0, opts, refresh);
}

[[nodiscard]] auto shamanskii(const ResidualFn& F, const JacobianFn& J,
                              std::span<const double> x0, std::size_t m,
                              const Options& opts = {}) -> Result<SolveResult> {
    const std::size_t n = x0.size();
    const std::size_t refresh = m == 0 ? 1 : m;
    detail::JacProvider jac = [&, n](std::span<const double> x,
                                     std::span<const double>) -> Result<detail::Vec> {
        std::vector<double> j = J(x);
        if (j.size() != n * n || !detail::all_finite(j)) {
            return make_error<detail::Vec>(MathError::domain_error);
        }
        return j;
    };
    return detail::dense_family(F, jac, x0, opts, refresh);
}

// --- Broyden (quasi-Newton, rank-1 inverse update) --------------------------

// Broyden's method maintaining an approximate INVERSE Jacobian H (seeded from the FD
// Jacobian at x0, or the identity if that is singular) and updating it by a rank-1
// correction each step. `good` uses the Sherman-Morrison inverse update; `bad` uses the
// (often less robust) alternative. Locally superlinearly convergent.
[[nodiscard]] auto broyden(const ResidualFn& F, std::span<const double> x0,
                           const Options& opts = {},
                           BroydenVariant variant = BroydenVariant::good)
    -> Result<SolveResult> {
    using detail::Vec;
    const std::size_t n = x0.size();
    if (n == 0 || !detail::all_finite(x0)) {
        return make_error<SolveResult>(MathError::domain_error);
    }
    std::size_t fevals = 0;
    Vec x(x0.begin(), x0.end());
    auto Fx_ = detail::eval_checked(F, x, n, fevals);
    if (!Fx_) {
        return make_error<SolveResult>(Fx_.error());
    }
    Vec Fx = std::move(*Fx_);
    double fnorm = detail::norm2(Fx);

    // Seed H ~ J(x0)^{-1}; fall back to the identity when J0 is singular.
    Vec H(n * n, 0.0);
    {
        std::size_t local = 0;
        auto J0 = detail::fd_jacobian(F, x, Fx, opts.fd_step, opts.central_diff, local);
        std::optional<Vec> Hopt = J0 ? detail::invert(*J0, n) : std::nullopt;
        if (Hopt) {
            H = std::move(*Hopt);
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                H[i * n + i] = 1.0;
            }
        }
    }

    for (std::size_t it = 0; it < opts.max_iter; ++it) {
        if (fnorm <= opts.tol) {
            return SolveResult{std::move(x), fnorm, it, true, fevals};
        }
        // Quasi-Newton direction d = -H F.
        Vec d = detail::matvec(H, Fx, n);
        for (double& e : d) {
            e = -e;
        }
        auto ls = detail::line_search(F, x, d, fnorm, n, opts, fevals);
        if (!ls.ok) {
            return SolveResult{std::move(x), fnorm, it, false, fevals};
        }
        Vec s = detail::sub(ls.x_new, x);     // s = x_new - x = lambda*d
        Vec y = detail::sub(ls.F_new, Fx);    // y = F_new - F
        Vec Hy = detail::matvec(H, y, n);
        Vec u = detail::sub(s, Hy);           // s - H y
        if (variant == BroydenVariant::good) {
            const double denom = detail::dot(s, Hy);  // s^T H y
            if (std::abs(denom) > 1e-300) {
                // sTH_j = sum_i s_i H[i][j]  (row vector s^T H)
                Vec sTH(n, 0.0);
                for (std::size_t i = 0; i < n; ++i) {
                    const double si = s[i];
                    for (std::size_t j = 0; j < n; ++j) {
                        sTH[j] = std::fma(si, H[i * n + j], sTH[j]);
                    }
                }
                const double inv = 1.0 / denom;
                for (std::size_t i = 0; i < n; ++i) {
                    const double ui = u[i] * inv;
                    for (std::size_t j = 0; j < n; ++j) {
                        H[i * n + j] = std::fma(ui, sTH[j], H[i * n + j]);
                    }
                }
            }
        } else {
            const double denom = detail::dot(y, y);  // y^T y
            if (denom > 1e-300) {
                const double inv = 1.0 / denom;
                for (std::size_t i = 0; i < n; ++i) {
                    const double ui = u[i] * inv;
                    for (std::size_t j = 0; j < n; ++j) {
                        H[i * n + j] = std::fma(ui, y[j], H[i * n + j]);
                    }
                }
            }
        }
        x = std::move(ls.x_new);
        Fx = std::move(ls.F_new);
        fnorm = ls.fnorm;
    }
    return SolveResult{std::move(x), fnorm, opts.max_iter, fnorm <= opts.tol, fevals};
}

// --- Newton-Krylov (JFNK) ---------------------------------------------------

// Jacobian-FREE Newton-Krylov: each Newton step is solved inexactly by an inner GMRES(m)
// that never forms the Jacobian, using finite-difference Jacobian-vector products
// Jv ~= (F(x+eps v) - F(x))/eps. The inner tolerance eta_k is chosen adaptively by the
// Eisenstat-Walker "Choice 2" forcing term (with the standard safeguards) to avoid
// oversolving the linear system far from the root. Globalised by the Armijo line search.
// `krylov_dim` is the GMRES restart length m.
[[nodiscard]] auto newton_krylov(const ResidualFn& F, std::span<const double> x0,
                                 const Options& opts = {}, std::size_t krylov_dim = 30)
    -> Result<SolveResult> {
    using detail::Vec;
    const std::size_t n = x0.size();
    if (n == 0 || !detail::all_finite(x0)) {
        return make_error<SolveResult>(MathError::domain_error);
    }
    std::size_t fevals = 0;
    Vec x(x0.begin(), x0.end());
    auto Fx_ = detail::eval_checked(F, x, n, fevals);
    if (!Fx_) {
        return make_error<SolveResult>(Fx_.error());
    }
    Vec Fx = std::move(*Fx_);
    double fnorm = detail::norm2(Fx);

    const std::size_t m = krylov_dim == 0 ? std::min<std::size_t>(n, 30) : krylov_dim;
    constexpr double eta_max = 0.9;
    constexpr double gamma = 0.9;
    double eta = 0.5;  // initial forcing term

    for (std::size_t it = 0; it < opts.max_iter; ++it) {
        if (fnorm <= opts.tol) {
            return SolveResult{std::move(x), fnorm, it, true, fevals};
        }
        // Matrix-free operator v |-> J(x) v via a finite-difference directional derivative.
        detail::LinOp A = [&](std::span<const double> v) -> Result<Vec> {
            return detail::jv_product(F, x, Fx, v, fevals);
        };
        const Vec b = detail::scale(Fx, -1.0);  // solve J s = -F
        auto step = detail::gmres(A, b, m, eta, /*max_restarts=*/2);
        if (!step) {
            // Non-finite Jv breakdown inside the inner solve -> stop with best iterate.
            return SolveResult{std::move(x), fnorm, it, false, fevals};
        }
        auto ls = detail::line_search(F, x, *step, fnorm, n, opts, fevals);
        if (!ls.ok) {
            return SolveResult{std::move(x), fnorm, it, false, fevals};
        }
        const double fnorm_old = fnorm;
        x = std::move(ls.x_new);
        Fx = std::move(ls.F_new);
        fnorm = ls.fnorm;

        // Eisenstat-Walker Choice 2 update with the two standard safeguards.
        double eta_new = gamma * (fnorm / fnorm_old) * (fnorm / fnorm_old);
        const double eta_safe = gamma * eta * eta;
        if (eta_safe > 0.1) {
            eta_new = std::max(eta_new, eta_safe);
        }
        eta = std::min(eta_max, eta_new);
        // Avoid oversolving near convergence: never demand more than ~0.5*tol/||F||.
        if (fnorm > 0.0) {
            eta = std::max(eta, 0.5 * opts.tol / fnorm);
        }
        eta = std::min(eta, eta_max);
    }
    return SolveResult{std::move(x), fnorm, opts.max_iter, fnorm <= opts.tol, fevals};
}

// --- Anderson acceleration (fixed-point g(x) = x) ---------------------------

// Anderson acceleration for a fixed-point iteration x <- g(x) (Kelley / Walker-Ni),
// mixing over a window of the last `window` residuals f = g(x) - x via a small
// least-squares problem. `window == 0` degenerates to plain Picard iteration. Accelerates
// linearly-convergent fixed points; it is a heuristic with no global guarantee.
[[nodiscard]] auto anderson(const ResidualFn& g, std::span<const double> x0,
                            std::size_t window = 5, const Options& opts = {})
    -> Result<SolveResult> {
    using detail::Vec;
    const std::size_t n = x0.size();
    if (n == 0 || !detail::all_finite(x0)) {
        return make_error<SolveResult>(MathError::domain_error);
    }
    std::size_t fevals = 0;
    Vec x(x0.begin(), x0.end());

    // History of g-values and residuals f = g(x) - x, capped to window+1 columns.
    std::deque<Vec> gh;
    std::deque<Vec> fh;

    auto residual = [&](const Vec& xv) -> Result<std::pair<Vec, Vec>> {
        auto gv = detail::eval_checked(g, xv, n, fevals);
        if (!gv) {
            return make_error<std::pair<Vec, Vec>>(gv.error());
        }
        Vec f = detail::sub(*gv, xv);
        return std::pair<Vec, Vec>{std::move(*gv), std::move(f)};
    };

    auto first = residual(x);
    if (!first) {
        return make_error<SolveResult>(first.error());
    }
    gh.push_back(std::move(first->first));
    fh.push_back(std::move(first->second));
    double fnorm = detail::norm2(fh.back());

    for (std::size_t it = 0; it < opts.max_iter; ++it) {
        if (fnorm <= opts.tol) {
            return SolveResult{std::move(x), fnorm, it, true, fevals};
        }
        const Vec& g_k = gh.back();
        const Vec& f_k = fh.back();
        const std::size_t hist = fh.size() - 1;             // available difference columns
        const std::size_t mk = std::min(window, hist);       // mixing depth this step

        Vec x_next;
        if (mk == 0) {
            x_next = g_k;  // plain Picard step
        } else {
            // Build difference matrices over the last mk+1 history entries:
            //   dF columns: f_{j+1} - f_j     (n x mk)
            //   dG columns: g_{j+1} - g_j     (n x mk)
            const std::size_t base = fh.size() - mk - 1;
            std::vector<Vec> dF(mk);
            std::vector<Vec> dG(mk);
            for (std::size_t c = 0; c < mk; ++c) {
                dF[c] = detail::sub(fh[base + c + 1], fh[base + c]);
                dG[c] = detail::sub(gh[base + c + 1], gh[base + c]);
            }
            // Normal equations (dF^T dF + mu I) gamma = dF^T f_k, mu a tiny Tikhonov term.
            Vec ATA(mk * mk, 0.0);
            Vec ATb(mk, 0.0);
            for (std::size_t a = 0; a < mk; ++a) {
                for (std::size_t bcol = 0; bcol < mk; ++bcol) {
                    ATA[a * mk + bcol] = detail::dot(dF[a], dF[bcol]);
                }
                ATb[a] = detail::dot(dF[a], f_k);
            }
            double tr = 0.0;
            for (std::size_t a = 0; a < mk; ++a) {
                tr += ATA[a * mk + a];
            }
            const double mu = 1e-12 * (tr > 0.0 ? tr : 1.0);
            for (std::size_t a = 0; a < mk; ++a) {
                ATA[a * mk + a] += mu;
            }
            auto gam = detail::lu_solve(ATA, ATb, mk);
            if (!gam) {
                x_next = g_k;  // ill-conditioned mixing -> safe Picard fallback
            } else {
                // x_{k+1} = g_k - sum_c gamma_c dG_c.
                x_next = g_k;
                for (std::size_t c = 0; c < mk; ++c) {
                    detail::axpy(x_next, -(*gam)[c], dG[c]);
                }
            }
        }

        if (!detail::all_finite(x_next)) {
            return SolveResult{std::move(x), fnorm, it, false, fevals};  // breakdown
        }
        auto nxt = residual(x_next);
        if (!nxt) {
            return SolveResult{std::move(x), fnorm, it, false, fevals};
        }
        x = std::move(x_next);
        gh.push_back(std::move(nxt->first));
        fh.push_back(std::move(nxt->second));
        while (fh.size() > window + 1) {
            fh.pop_front();
            gh.pop_front();
        }
        fnorm = detail::norm2(fh.back());
    }
    return SolveResult{std::move(x), fnorm, opts.max_iter, fnorm <= opts.tol, fevals};
}

// --- Levenberg-Marquardt (nonlinear least squares) --------------------------

namespace detail_lm {
using detail::Vec;

// One LM driver parameterised by a Jacobian provider returning the DENSE m x n Jacobian.
[[nodiscard]] inline auto run(const ResidualFn& F,
                              const std::function<Result<Vec>(std::span<const double>, std::size_t)>& jac,
                              std::span<const double> x0, const Options& opts, double lambda0)
    -> Result<SolveResult> {
    const std::size_t n = x0.size();
    if (n == 0 || !detail::all_finite(x0)) {
        return make_error<SolveResult>(MathError::domain_error);
    }
    std::size_t fevals = 0;
    Vec x(x0.begin(), x0.end());
    Vec Fx = F(x);
    ++fevals;
    const std::size_t m = Fx.size();
    if (m == 0 || m < n || !detail::all_finite(Fx)) {
        // m < n is an under-determined least-squares problem (rank-deficient normal
        // equations); reject up front rather than silently returning a bogus point.
        return make_error<SolveResult>(MathError::domain_error);
    }
    double fnorm = detail::norm2(Fx);
    double lambda = lambda0 > 0.0 ? lambda0 : 1e-3;

    for (std::size_t it = 0; it < opts.max_iter; ++it) {
        auto Jr = jac(x, m);
        if (!Jr) {
            if (it == 0) {
                return make_error<SolveResult>(Jr.error());
            }
            return SolveResult{std::move(x), fnorm, it, false, fevals};
        }
        const Vec& J = *Jr;  // m x n row-major
        // Normal-equation pieces: A = J^T J (n x n), grad = J^T F (n).
        Vec A(n * n, 0.0);
        Vec grad(n, 0.0);
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                const double jij = J[i * n + j];
                grad[j] = std::fma(jij, Fx[i], grad[j]);
                for (std::size_t k = 0; k < n; ++k) {
                    A[j * n + k] = std::fma(jij, J[i * n + k], A[j * n + k]);
                }
            }
        }
        // Stationarity test on the gradient of ||F||^2 (== 2 J^T F).
        double gmax = 0.0;
        for (double gv : grad) {
            gmax = std::max(gmax, std::abs(gv));
        }
        if (fnorm <= opts.tol || gmax <= opts.tol) {
            return SolveResult{std::move(x), fnorm, it, true, fevals};
        }
        // Inner damping loop: grow lambda until the trial step reduces the cost.
        bool accepted = false;
        for (std::size_t inner = 0; inner < 40; ++inner) {
            Vec M = A;
            for (std::size_t j = 0; j < n; ++j) {
                const double djj = A[j * n + j];
                // Marquardt scaling: (J^T J + lambda diag(J^T J)) delta = -J^T F.
                M[j * n + j] += lambda * (djj > 0.0 ? djj : 1.0);
            }
            Vec rhs = detail::scale(grad, -1.0);
            auto delta = detail::lu_solve(std::move(M), std::move(rhs), n);
            if (!delta) {
                lambda *= 4.0;  // singular damped system -> more damping
                if (lambda > 1e18) {
                    return SolveResult{std::move(x), fnorm, it, false, fevals};
                }
                continue;
            }
            Vec x_try = detail::step_point(x, 1.0, *delta);
            if (!detail::all_finite(x_try)) {
                lambda *= 4.0;
                continue;
            }
            Vec F_try = F(x_try);
            ++fevals;
            if (F_try.size() != m || !detail::all_finite(F_try)) {
                lambda *= 4.0;
                if (lambda > 1e18) {
                    return SolveResult{std::move(x), fnorm, it, false, fevals};
                }
                continue;
            }
            const double fn_try = detail::norm2(F_try);
            if (fn_try < fnorm) {
                x = std::move(x_try);
                Fx = std::move(F_try);
                fnorm = fn_try;
                lambda = std::max(lambda / 3.0, 1e-12);  // trust step -> less damping
                accepted = true;
                break;
            }
            lambda *= 4.0;  // reject -> gradient-descent-like, smaller step
            if (lambda > 1e18) {
                return SolveResult{std::move(x), fnorm, it + 1, false, fevals};
            }
        }
        if (!accepted) {
            return SolveResult{std::move(x), fnorm, it + 1, false, fevals};
        }
    }
    return SolveResult{std::move(x), fnorm, opts.max_iter, fnorm <= opts.tol, fevals};
}

}  // namespace detail_lm

// Levenberg-Marquardt for the nonlinear least-squares problem min ||F(x)||^2, where F may
// be over-determined (|F(x)| = m >= n). The damping lambda interpolates between the
// Gauss-Newton step (small lambda) and scaled gradient descent (large lambda); it is
// grown on rejected steps and shrunk on accepted ones. Uses a finite-difference Jacobian.
// Finds a STATIONARY point of the cost, which need not be a global minimiser or a root.
[[nodiscard]] auto levenberg_marquardt(const ResidualFn& F, std::span<const double> x0,
                                       const Options& opts = {}, double lambda0 = 1e-3)
    -> Result<SolveResult> {
    auto jac = [&](std::span<const double> x, std::size_t m) -> Result<detail::Vec> {
        std::size_t local = 0;
        std::vector<double> Fx = F(x);
        ++local;
        if (Fx.size() != m || !detail::all_finite(Fx)) {
            return make_error<detail::Vec>(MathError::domain_error);
        }
        return detail::fd_jacobian(F, x, Fx, opts.fd_step, opts.central_diff, local);
    };
    return detail_lm::run(F, jac, x0, opts, lambda0);
}

// Levenberg-Marquardt with a user-supplied analytic Jacobian J : R^n -> R^{m x n}
// (row-major). More accurate than the finite-difference variant.
[[nodiscard]] auto levenberg_marquardt(const ResidualFn& F, const JacobianFn& J,
                                       std::span<const double> x0, const Options& opts = {},
                                       double lambda0 = 1e-3) -> Result<SolveResult> {
    auto jac = [&](std::span<const double> x, std::size_t m) -> Result<detail::Vec> {
        std::vector<double> j = J(x);
        if (j.size() != m * x.size() || !detail::all_finite(j)) {
            return make_error<detail::Vec>(MathError::domain_error);
        }
        return j;
    };
    return detail_lm::run(F, jac, x0, opts, lambda0);
}

}  // namespace nimblecas::nlsolve

// ===========================================================================
// PARALLEL + DISTRIBUTED multistart (ROADMAP §7 acceleration layer).
// ---------------------------------------------------------------------------
// ADDITIVE: none of the solvers above change. This layer runs an EXISTING solver
// (newton / broyden / chord / ... — the caller picks) from MANY starting points and
// keeps the single best result, using nimblecas.parallel for the fan-out.
//
// HONESTY BOUNDARY — read before trusting a multistart result.
//   * Nonlinear solving is NUMERIC with LOCAL convergence (see the module header). A
//     multistart sweep only improves the CHANCE of locating a root of a system with
//     several solutions by seeding the local solver from many basins; it is NOT a global
//     solver and gives NO guarantee that a root exists or was found. If nothing converges
//     the layer still returns a value: the smallest-residual iterate it saw.
//   * converged == false is a RESULT, not an error. An individual start that hits a hard
//     domain_error (e.g. a non-finite start, or F breaking down at the start point) is
//     simply not a candidate; only if NO start yields a value does the whole sweep report
//     domain_error.
//   * The reduction is DETERMINISTIC. Candidates are ordered by a fixed TOTAL order —
//     (converged desc, residual_norm asc, then lowest GLOBAL start index) — so the answer
//     is bit-identical regardless of thread count, backend (ppl/tbb/serial), or how the
//     starts are partitioned across distributed shards. It equals the serial sweep over
//     the same starts, and reducing per-shard results equals the single-process sweep
//     (partition independence), because the winner is the global optimum of that total
//     order and the globally-best start is, a fortiori, the best within whichever shard
//     owns it. Ties never depend on scheduling: the global start index is unique.
//
// CONCURRENCY PRECONDITION: F and the chosen solver are invoked concurrently for distinct
// starts, so both callables MUST be safe to call from multiple threads at once (stateless
// or internally synchronised). Each start is solved independently over its own point; the
// per-start numeric result is identical to solving it in isolation.
// ===========================================================================
export namespace nimblecas::nlsolve {

// A solver adaptor: (F, x0, opts) -> Result<SolveResult>. Bind any solver above whose
// square-system signature matches, e.g. newton or broyden:
//   Solver s = [](const ResidualFn& F, std::span<const double> x0, const Options& o) {
//                  return broyden(F, x0, o); };
using Solver =
    std::function<Result<SolveResult>(const ResidualFn&, std::span<const double>, const Options&)>;

// Default solver used when the caller does not pass one: finite-difference Newton.
inline const Solver default_solver =
    [](const ResidualFn& F, std::span<const double> x0, const Options& o) -> Result<SolveResult> {
    return newton(F, x0, o);
};

// A shard's best result over its subset of starts. `start_index` is the GLOBAL index of
// the winning start (carried so a distributed driver can reproduce the exact tie-break
// when reducing across shards). `valid == false` means the subset produced no candidate
// (empty subset, or every start errored), in which case `result` is unspecified.
struct MultistartResult {
    SolveResult result{};
    std::size_t start_index{0};
    bool valid{false};
};

}  // namespace nimblecas::nlsolve

namespace nimblecas::nlsolve::detail {

// A comparable summary of one start's outcome under the deterministic total order.
struct Candidate {
    std::size_t index{0};
    double residual{std::numeric_limits<double>::infinity()};
    bool converged{false};
    bool valid{false};
};

// Strict "a is better than b" under (valid > invalid, converged desc, residual asc,
// index asc). A total order over valid candidates (indices are unique), so the chosen
// minimum is independent of evaluation/fold order — the source of determinism.
[[nodiscard]] inline auto better_candidate(const Candidate& a, const Candidate& b) noexcept
    -> bool {
    if (a.valid != b.valid) {
        return a.valid;  // any real result beats "no result"
    }
    if (!a.valid) {
        return false;  // neither is a candidate
    }
    if (a.converged != b.converged) {
        return a.converged;  // a converged root beats a non-converged iterate
    }
    if (a.residual != b.residual) {
        return a.residual < b.residual;  // then smaller residual
    }
    return a.index < b.index;  // then lowest global start index
}

// Evaluate the solver from every start whose global index is in `which`, in parallel and
// order-preserving (results[k] <-> which[k]), then reduce to the single best under the
// deterministic total order. `which` MUST be ascending so the reduction's index tie-break
// is unambiguous; the fold is serial over a total order, hence thread-count independent.
[[nodiscard]] inline auto multistart_over(const ResidualFn& F,
                                          std::span<const std::vector<double>> starts,
                                          std::span<const std::size_t> which,
                                          const Solver& solver, const Options& opts,
                                          std::size_t grain) -> MultistartResult {
    MultistartResult best{};  // valid == false
    const std::size_t n = which.size();
    if (n == 0 || !solver) {
        return best;
    }
    // Fan-out: transform_index is blocking, order-preserving, and deterministic (the
    // result depends only on the solver, not on scheduling). Each task writes its own slot.
    std::vector<Result<SolveResult>> results = nimblecas::parallel::transform_index(
        n,
        [&](std::size_t k) -> Result<SolveResult> {
            return solver(F, std::span<const double>{starts[which[k]]}, opts);
        },
        grain);

    Candidate best_cand{};  // invalid sentinel (residual = +inf)
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t g = which[k];
        auto& r = results[k];
        const Candidate cand{
            g,
            r.has_value() ? r->residual_norm : std::numeric_limits<double>::infinity(),
            r.has_value() && r->converged,
            r.has_value()};
        if (better_candidate(cand, best_cand)) {
            best_cand = cand;
            best.result = std::move(*r);  // only reached when r has a value
            best.start_index = g;
            best.valid = true;
        }
    }
    return best;
}

}  // namespace nimblecas::nlsolve::detail

export namespace nimblecas::nlsolve {

// Run `solver` from EVERY start in `starts` in parallel and return the deterministic best:
// the converged root of smallest residual (ties broken by lowest start index), or — if
// nothing converged — the smallest-residual iterate seen. Bit-identical to the serial
// sweep over the same starts, for any thread count / backend. Returns domain_error only
// when no start produced a value at all (e.g. empty `starts`, or all starts errored).
// `grain` forwards to the parallel runtime (1 = one task per start; the backend chunks).
[[nodiscard]] auto parallel_multistart(const ResidualFn& F,
                                       std::span<const std::vector<double>> starts,
                                       const Options& opts = {},
                                       const Solver& solver = default_solver,
                                       std::size_t grain = 1) -> Result<SolveResult> {
    std::vector<std::size_t> which(starts.size());
    std::iota(which.begin(), which.end(), std::size_t{0});
    MultistartResult best =
        detail::multistart_over(F, starts, std::span<const std::size_t>{which}, solver, opts, grain);
    if (!best.valid) {
        return make_error<SolveResult>(MathError::domain_error);
    }
    return std::move(best.result);
}

// DISTRIBUTED shard entrypoint: a stateless PURE function returning THIS shard's best
// result over its subset of `starts` (the global indices i with i % num_shards ==
// shard_index). Intended to run one-per-worker under SGE / Ray / etc.; a driver collects
// the MultistartResults and calls reduce_multistart to obtain the final answer. Returns an
// invalid MultistartResult when the shard spec is degenerate (num_shards == 0 or
// shard_index >= num_shards) or its subset produced no candidate.
[[nodiscard]] auto multistart_shard(const ResidualFn& F,
                                    std::span<const std::vector<double>> starts,
                                    std::size_t shard_index, std::size_t num_shards,
                                    const Options& opts = {},
                                    const Solver& solver = default_solver) -> MultistartResult {
    if (num_shards == 0 || shard_index >= num_shards) {
        return MultistartResult{};  // invalid: no result
    }
    // i % num_shards == shard_index, enumerated in ascending global-index order.
    std::vector<std::size_t> which;
    which.reserve(starts.size() / num_shards + 1);
    for (std::size_t i = shard_index; i < starts.size(); i += num_shards) {
        which.push_back(i);
    }
    return detail::multistart_over(F, starts, std::span<const std::size_t>{which}, solver, opts,
                                   /*grain=*/1);
}

// Driver-side reduction over shard results, using the SAME deterministic total order as
// the single-process sweep. Because shards partition the starts (disjoint global indices)
// and each shard reports its own best, folding the shard bests reproduces the global best
// exactly — the sweep is partition-independent. Returns domain_error when no shard yielded
// a candidate.
[[nodiscard]] auto reduce_multistart(std::span<const MultistartResult> shard_results)
    -> Result<SolveResult> {
    detail::Candidate best_cand{};  // invalid sentinel
    const MultistartResult* best = nullptr;
    for (const MultistartResult& s : shard_results) {
        const detail::Candidate cand{
            s.start_index,
            s.valid ? s.result.residual_norm : std::numeric_limits<double>::infinity(),
            s.valid && s.result.converged,
            s.valid};
        if (detail::better_candidate(cand, best_cand)) {
            best_cand = cand;
            best = &s;
        }
    }
    if (best == nullptr || !best->valid) {
        return make_error<SolveResult>(MathError::domain_error);
    }
    return best->result;
}

}  // namespace nimblecas::nlsolve
