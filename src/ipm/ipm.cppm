// NimbleCAS numeric linear programming: primal-dual interior-point method (ROADMAP 7.22).
// @author Olumuyiwa Oluwasanmi
//
// Solves the standard-form linear program
//
//     minimize  c . x   subject to  A x = b,  x >= 0,
//
// by a **Mehrotra predictor-corrector primal-dual path-following** interior-point
// method. This is the NUMERICAL companion to the exact-rational Simplex in
// `nimblecas.lp`: where that solver pivots in closed rational arithmetic and reports a
// vertex as an exact `p/q`, an interior-point method is inherently iterative — it walks
// the central path driving the duality measure mu = (x . s)/n toward zero — and so it
// runs in floating point (`double`). The trade is deliberate: Simplex is exact but its
// worst case is exponential in the number of constraints; the interior-point method
// converges in a small, roughly problem-size-independent number of Newton iterations,
// which is the right tool for large models. The two are siblings, not rivals.
//
// -------------------------------------------------------------------------------------
// Honesty boundary (Rule 32)
// -------------------------------------------------------------------------------------
// Results here are accurate to a tolerance, NOT exact rationals. `solve_ipm` returns
// `IpmStatus::optimal` only after the primal residual ||A x - b||, the dual residual
// ||A^T y + s - c||, and the duality measure mu are ALL below `tolerance` — the small
// duality gap is what certifies optimality (weak duality: c^T x - b^T y = x . s = n*mu,
// so a tiny mu bounds the distance from the true optimum). The reported `objective`,
// `x`, and `y` are therefore correct to about `tolerance`, and a caller must not read
// them as exact. When the iteration cap is reached without meeting all three criteria
// and the iterates are not diverging, we do NOT invent a plausible "optimal" — we
// return `MathError::not_implemented` (the closest honest variant in `MathError` for a
// genuine failure to converge; the enum has no dedicated `no_convergence`). Divergence
// is classified best-effort as `infeasible` or `unbounded` (see below); when it cannot
// be told apart honestly the same `not_implemented` is returned rather than a guess.
//
// -------------------------------------------------------------------------------------
// The method
// -------------------------------------------------------------------------------------
// State: primal x > 0 (length n), dual multipliers y (length m, free), dual slacks
// s > 0 (length n). The perturbed KKT conditions of the central path are
//
//     A x = b,   A^T y + s = c,   x_i s_i = mu,   x, s >= 0.
//
// Each iteration applies one Newton step, reduced to the **normal equations**
//
//     (A D A^T) dy = r,     D = diag(x / s)  (m x m, symmetric positive definite),
//
// solved by a dense **Cholesky factorization written inline in this module** (no
// external double-Cholesky is assumed). From dy: ds = r_d - A^T dy and
// dx = -D ds + r_xs / s. Mehrotra's scheme takes an affine (predictor) step with
// target mu = 0, derives a centering parameter sigma = (mu_aff / mu)^3 from the
// affine step lengths, then a corrector step whose complementarity target is
// -x_i s_i + sigma*mu - dx_aff_i * ds_aff_i (the second-order term is Mehrotra's
// correction). A **fraction-to-the-boundary** rule scales the step so x, s stay
// strictly positive. The same D — and hence the same Cholesky factor — serves both the
// predictor and corrector solves, so we factor once per iteration and solve twice.
//
// A Mehrotra-style starting point (computed from A A^T, reusing the same solver with
// D = I) places the initial iterate well inside the positive orthant; if that AA^T
// system is not factorable the solver falls back to x = s = 1, y = 0.
//
// Every fallible step returns via `Result` (Rule 32): dimension/domain checks up front,
// a Cholesky breakdown or non-convergence as an honest `MathError`.

export module nimblecas.ipm;

import std;
import nimblecas.core;

export namespace nimblecas {

// Outcome classification. `optimal` is gap-certified to `tolerance`; `infeasible` and
// `unbounded` are best-effort classifications of a diverging primal-dual iterate.
enum class IpmStatus : std::uint8_t { optimal, infeasible, unbounded };

// The result of a solved LP. On `optimal`, `objective` is c . x (accurate to about the
// solve tolerance), `x` is the primal optimum (length n), and `y` is the dual optimum /
// shadow prices (length m). On `infeasible` / `unbounded`, `objective` is 0 and `x`,
// `y` are empty — they are not meaningful.
struct IpmSolution {
    IpmStatus status;
    double objective;
    std::vector<double> x;
    std::vector<double> y;
};

// Solve  min c . x  s.t.  A x = b, x >= 0  with A an m x n matrix (m constraints, n
// variables), b of length m, c of length n. Dimensions are derived as m = A.size() and
// n = c.size().
//
// Domain errors (MathError::domain_error): m == 0 or n == 0; a ragged A (some row's
// width != n); b.size() != m; or any non-finite entry in A, b, or c.
//
// On success `IpmStatus::optimal` is returned once the primal residual, dual residual,
// and duality measure mu are all below the internal tolerance (1e-9). A diverging
// iterate is classified `unbounded` (primal iterate blows up) or `infeasible` (dual
// iterate blows up). If the iteration cap is reached with neither convergence nor a
// clear divergence classification, `MathError::not_implemented` is returned to signal
// an honest failure to converge (see the header's honesty boundary). A Cholesky
// breakdown on the normal equations (numerically rank-deficient A) likewise surfaces as
// `MathError::not_implemented` rather than a fabricated answer.
[[nodiscard]] auto solve_ipm(const std::vector<std::vector<double>>& A,
                             const std::vector<double>& b,
                             const std::vector<double>& c) -> Result<IpmSolution>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// --- Solver constants -------------------------------------------------------------
inline constexpr double kTolerance = 1e-9;     // primal/dual residual and mu target
inline constexpr int    kMaxIter   = 200;      // Newton-iteration cap before giving up
inline constexpr double kGamma     = 0.99;     // fraction-to-the-boundary factor
inline constexpr double kDiverge   = 1e13;     // inf-norm above this => divergence
inline constexpr double kRegular   = 1e-12;    // tiny SPD regularization of A D A^T

// --- Small dense vector helpers ---------------------------------------------------

// y = A x, with A an m x n matrix and x length n; result length m.
[[nodiscard]] auto mat_vec(const std::vector<std::vector<double>>& A,
                           const std::vector<double>& x) -> std::vector<double> {
    std::vector<double> y(A.size(), 0.0);
    for (std::size_t i = 0; i < A.size(); ++i) {
        double acc = 0.0;
        for (std::size_t j = 0; j < x.size(); ++j) {
            acc += A[i][j] * x[j];
        }
        y[i] = acc;
    }
    return y;
}

// z = A^T y, with A an m x n matrix and y length m; result length n.
[[nodiscard]] auto matt_vec(const std::vector<std::vector<double>>& A, std::size_t n,
                            const std::vector<double>& y) -> std::vector<double> {
    std::vector<double> z(n, 0.0);
    for (std::size_t i = 0; i < A.size(); ++i) {
        const double yi = y[i];
        for (std::size_t j = 0; j < n; ++j) {
            z[j] += A[i][j] * yi;
        }
    }
    return z;
}

[[nodiscard]] auto dot(const std::vector<double>& a, const std::vector<double>& b) -> double {
    double acc = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

[[nodiscard]] auto inf_norm(const std::vector<double>& v) -> double {
    double m = 0.0;
    for (const double x : v) {
        m = std::max(m, std::fabs(x));
    }
    return m;
}

// Largest alpha in (0, 1] keeping v + alpha*dv >= 0 componentwise. Since v > 0 strictly
// and only dv_i < 0 can drive a component to zero, every candidate ratio is positive.
[[nodiscard]] auto step_to_boundary(const std::vector<double>& v,
                                    const std::vector<double>& dv) -> double {
    double alpha = 1.0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (dv[i] < 0.0) {
            alpha = std::min(alpha, -v[i] / dv[i]);
        }
    }
    return alpha;
}

// --- Inline dense Cholesky over double --------------------------------------------

// Cholesky factor of the symmetric matrix M (M = L L^T), returned as the lower triangle
// L. Returns nullopt if a non-positive pivot is met (M is not positive definite, i.e.
// the normal matrix is numerically rank-deficient). Written here rather than borrowed:
// the module must not assume an external double-precision Cholesky exists.
[[nodiscard]] auto cholesky(const std::vector<std::vector<double>>& M)
    -> std::optional<std::vector<std::vector<double>>> {
    const std::size_t n = M.size();
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double sum = M[i][j];
            for (std::size_t k = 0; k < j; ++k) {
                sum -= L[i][k] * L[j][k];
            }
            if (i == j) {
                if (!(sum > 0.0)) {  // catches <= 0 and NaN
                    return std::nullopt;
                }
                L[i][i] = std::sqrt(sum);
            } else {
                L[i][j] = sum / L[j][j];
            }
        }
    }
    return L;
}

// Solve L L^T z = rhs by forward then back substitution given the Cholesky factor L.
[[nodiscard]] auto cholesky_solve(const std::vector<std::vector<double>>& L,
                                  const std::vector<double>& rhs) -> std::vector<double> {
    const std::size_t n = L.size();
    std::vector<double> z(n, 0.0);
    // Forward: L w = rhs (reuse z as w).
    for (std::size_t i = 0; i < n; ++i) {
        double sum = rhs[i];
        for (std::size_t k = 0; k < i; ++k) {
            sum -= L[i][k] * z[k];
        }
        z[i] = sum / L[i][i];
    }
    // Back: L^T z = w.
    for (std::size_t ii = n; ii-- > 0;) {
        double sum = z[ii];
        for (std::size_t k = ii + 1; k < n; ++k) {
            sum -= L[k][ii] * z[k];
        }
        z[ii] = sum / L[ii][ii];
    }
    return z;
}

// Build and factor the SPD normal matrix M = A diag(d) A^T + reg*I (m x m). A tiny
// regularization keeps the factorization robust when d has near-zero or very large
// entries as the iterate approaches the boundary; it is negligible (1e-12) against the
// O(1) data of well-posed problems and does not compromise the 1e-9 solve tolerance.
[[nodiscard]] auto factor_normal(const std::vector<std::vector<double>>& A,
                                 const std::vector<double>& d, double reg)
    -> std::optional<std::vector<std::vector<double>>> {
    const std::size_t m = A.size();
    const std::size_t n = d.size();
    std::vector<std::vector<double>> M(m, std::vector<double>(m, 0.0));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = i; j < m; ++j) {
            double acc = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                acc += A[i][k] * d[k] * A[j][k];
            }
            M[i][j] = acc;
            M[j][i] = acc;
        }
        M[i][i] += reg;
    }
    return cholesky(M);
}

// --- Mehrotra starting point ------------------------------------------------------
// Places (x, y, s) well inside the positive orthant. Uses the least-norm primal
// x_bar = A^T (A A^T)^{-1} b and the least-squares dual y_bar = (A A^T)^{-1} A c,
// s_bar = c - A^T y_bar, then shifts x, s to be positive and centered (Mehrotra 1992).
// Falls back to the all-ones interior point if A A^T is not factorable.
struct StartPoint {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> s;
};

[[nodiscard]] auto initial_point(const std::vector<std::vector<double>>& A,
                                 const std::vector<double>& b,
                                 const std::vector<double>& c, std::size_t m,
                                 std::size_t n) -> StartPoint {
    const std::vector<double> ones_d(n, 1.0);
    auto L = factor_normal(A, ones_d, kRegular);
    if (!L) {
        // Defensive fallback only. With finite, validated inputs A A^T + kRegular*I is SPD, so
        // this rarely triggers; when it does (e.g. a non-finite slip past validation) we start
        // from the neutral interior point x=s=1, y=0 rather than fail the whole solve here.
        return StartPoint{std::vector<double>(n, 1.0), std::vector<double>(m, 0.0),
                          std::vector<double>(n, 1.0)};
    }
    // x_bar = A^T (A A^T)^{-1} b.
    const auto z = cholesky_solve(*L, b);
    auto x = matt_vec(A, n, z);
    // y_bar = (A A^T)^{-1} A c ;  s_bar = c - A^T y_bar.
    const auto Ac = mat_vec(A, c);
    auto y = cholesky_solve(*L, Ac);
    const auto Aty = matt_vec(A, n, y);
    std::vector<double> s(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        s[j] = c[j] - Aty[j];
    }
    // Shift x, s into the positive orthant.
    double min_x = 0.0;
    double min_s = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        min_x = std::min(min_x, x[j]);
        min_s = std::min(min_s, s[j]);
    }
    const double dx = std::max(-1.5 * min_x, 0.0) + 1.0;  // +1 guarantees strict interior
    const double ds = std::max(-1.5 * min_s, 0.0) + 1.0;
    for (std::size_t j = 0; j < n; ++j) {
        x[j] += dx;
        s[j] += ds;
    }
    // Center: add half the average complementarity to each coordinate.
    double sum_x = 0.0;
    double sum_s = 0.0;
    const double xs = dot(x, s);
    for (std::size_t j = 0; j < n; ++j) {
        sum_x += x[j];
        sum_s += s[j];
    }
    if (sum_x > 0.0 && sum_s > 0.0 && xs > 0.0) {
        const double px = 0.5 * xs / sum_s;
        const double ps = 0.5 * xs / sum_x;
        for (std::size_t j = 0; j < n; ++j) {
            x[j] += px;
            s[j] += ps;
        }
    }
    // Final guard: any residual non-positive coordinate is clamped to a strict interior.
    for (std::size_t j = 0; j < n; ++j) {
        if (!(x[j] > 0.0)) {
            x[j] = 1.0;
        }
        if (!(s[j] > 0.0)) {
            s[j] = 1.0;
        }
    }
    return StartPoint{std::move(x), std::move(y), std::move(s)};
}

[[nodiscard]] auto all_finite(const std::vector<double>& v) -> bool {
    for (const double x : v) {
        if (!std::isfinite(x)) {
            return false;
        }
    }
    return true;
}

}  // namespace

auto solve_ipm(const std::vector<std::vector<double>>& A, const std::vector<double>& b,
               const std::vector<double>& c) -> Result<IpmSolution> {
    const std::size_t m = A.size();
    const std::size_t n = c.size();

    // --- Dimension and domain validation ---------------------------------------------
    if (m == 0 || n == 0) {
        return make_error<IpmSolution>(MathError::domain_error);
    }
    if (b.size() != m) {
        return make_error<IpmSolution>(MathError::domain_error);
    }
    for (const auto& row : A) {
        if (row.size() != n || !all_finite(row)) {  // ragged / non-finite
            return make_error<IpmSolution>(MathError::domain_error);
        }
    }
    if (!all_finite(b) || !all_finite(c)) {
        return make_error<IpmSolution>(MathError::domain_error);
    }

    // --- Starting point (strictly interior) ------------------------------------------
    auto start = initial_point(A, b, c, m, n);
    std::vector<double> x = std::move(start.x);
    std::vector<double> y = std::move(start.y);
    std::vector<double> s = std::move(start.s);

    const double b_scale = 1.0 + inf_norm(b);
    const double c_scale = 1.0 + inf_norm(c);

    // Scratch reused across iterations.
    std::vector<double> d(n, 0.0);        // D = diag(x / s)
    std::vector<double> r_xs(n, 0.0);     // complementarity residual target
    std::vector<double> rhs_y(m, 0.0);    // right-hand side of the normal equations

    for (int iter = 0; iter < kMaxIter; ++iter) {
        // Residuals:  r_p = b - A x  (primal),  r_d = c - A^T y - s  (dual).
        const auto Ax = mat_vec(A, x);
        std::vector<double> r_p(m, 0.0);
        for (std::size_t i = 0; i < m; ++i) {
            r_p[i] = b[i] - Ax[i];
        }
        const auto Aty = matt_vec(A, n, y);
        std::vector<double> r_d(n, 0.0);
        for (std::size_t j = 0; j < n; ++j) {
            r_d[j] = c[j] - Aty[j] - s[j];
        }
        const double mu = dot(x, s) / static_cast<double>(n);

        // --- Convergence: primal residual, dual residual, and mu all below tolerance --
        const double primal_res = inf_norm(r_p) / b_scale;
        const double dual_res = inf_norm(r_d) / c_scale;
        if (primal_res < kTolerance && dual_res < kTolerance && mu < kTolerance) {
            return IpmSolution{.status = IpmStatus::optimal,
                               .objective = dot(c, x),
                               .x = std::move(x),
                               .y = std::move(y)};
        }

        // --- Divergence classification (best-effort, honest) --------------------------
        // A blown-up primal iterate signals an unbounded objective (the feasible region
        // recedes along a cost-decreasing ray); a blown-up dual iterate signals primal
        // infeasibility (the dual is unbounded). See the header's honesty boundary.
        if (inf_norm(x) > kDiverge) {
            return IpmSolution{.status = IpmStatus::unbounded, .objective = 0.0, .x = {}, .y = {}};
        }
        if (inf_norm(y) > kDiverge || inf_norm(s) > kDiverge) {
            return IpmSolution{.status = IpmStatus::infeasible, .objective = 0.0, .x = {}, .y = {}};
        }
        if (!all_finite(x) || !all_finite(y) || !all_finite(s)) {
            return make_error<IpmSolution>(MathError::not_implemented);  // numerical blow-up
        }

        // --- Factor the normal matrix A D A^T once; reuse for both solves -------------
        for (std::size_t j = 0; j < n; ++j) {
            d[j] = x[j] / s[j];
        }
        auto L = factor_normal(A, d, kRegular);
        if (!L) {
            // Cholesky breakdown: a non-positive/non-finite pivot. Because factor_normal adds
            // the kRegular jitter to each diagonal, an ordinarily rank-deficient (PSD) A D A^T
            // is nudged strictly positive-definite and still factors; this branch therefore
            // fires on genuine numerical breakdown (negative pivot from a non-finite/degenerate
            // scaling), NOT on mere rank loss. Either way the failure is surfaced honestly.
            return make_error<IpmSolution>(MathError::not_implemented);
        }

        // --- Predictor (affine) step: complementarity target 0 ------------------------
        // r_xs = -x .* s ;  rhs_y = r_p + A (D r_d - r_xs / s).
        for (std::size_t j = 0; j < n; ++j) {
            r_xs[j] = -x[j] * s[j];
        }
        {
            std::vector<double> w(n, 0.0);
            for (std::size_t j = 0; j < n; ++j) {
                w[j] = d[j] * r_d[j] - r_xs[j] / s[j];
            }
            const auto Aw = mat_vec(A, w);
            for (std::size_t i = 0; i < m; ++i) {
                rhs_y[i] = r_p[i] + Aw[i];
            }
        }
        const auto dy_aff = cholesky_solve(*L, rhs_y);
        const auto Atdy_aff = matt_vec(A, n, dy_aff);
        std::vector<double> ds_aff(n, 0.0);
        std::vector<double> dx_aff(n, 0.0);
        for (std::size_t j = 0; j < n; ++j) {
            ds_aff[j] = r_d[j] - Atdy_aff[j];          // ds = r_d - A^T dy
            dx_aff[j] = -d[j] * ds_aff[j] + r_xs[j] / s[j];  // dx = -D ds + r_xs / s
        }

        // Affine step lengths and the centering parameter sigma = (mu_aff / mu)^3.
        const double ap_aff = step_to_boundary(x, dx_aff);
        const double ad_aff = step_to_boundary(s, ds_aff);
        double mu_aff = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            mu_aff += (x[j] + ap_aff * dx_aff[j]) * (s[j] + ad_aff * ds_aff[j]);
        }
        mu_aff /= static_cast<double>(n);
        double sigma = 0.0;
        if (mu > 0.0) {
            const double ratio = mu_aff / mu;
            sigma = ratio * ratio * ratio;
        }
        sigma = std::min(std::max(sigma, 0.0), 1.0);

        // --- Corrector step -----------------------------------------------------------
        // r_xs = -x .* s + sigma*mu - dx_aff .* ds_aff  (Mehrotra second-order term).
        for (std::size_t j = 0; j < n; ++j) {
            r_xs[j] = -x[j] * s[j] + sigma * mu - dx_aff[j] * ds_aff[j];
        }
        {
            std::vector<double> w(n, 0.0);
            for (std::size_t j = 0; j < n; ++j) {
                w[j] = d[j] * r_d[j] - r_xs[j] / s[j];
            }
            const auto Aw = mat_vec(A, w);
            for (std::size_t i = 0; i < m; ++i) {
                rhs_y[i] = r_p[i] + Aw[i];
            }
        }
        const auto dy = cholesky_solve(*L, rhs_y);
        const auto Atdy = matt_vec(A, n, dy);
        std::vector<double> ds(n, 0.0);
        std::vector<double> dx(n, 0.0);
        for (std::size_t j = 0; j < n; ++j) {
            ds[j] = r_d[j] - Atdy[j];
            dx[j] = -d[j] * ds[j] + r_xs[j] / s[j];
        }

        // --- Fraction-to-the-boundary step; keeps x, s strictly positive --------------
        const double ap = std::min(1.0, kGamma * step_to_boundary(x, dx));
        const double ad = std::min(1.0, kGamma * step_to_boundary(s, ds));
        for (std::size_t j = 0; j < n; ++j) {
            x[j] += ap * dx[j];
            s[j] += ad * ds[j];
        }
        for (std::size_t i = 0; i < m; ++i) {
            y[i] += ad * dy[i];
        }
    }

    // Iteration cap reached without convergence and without a clear divergence verdict:
    // return an honest failure rather than a fabricated "optimal" (Rule 32).
    return make_error<IpmSolution>(MathError::not_implemented);
}

}  // namespace nimblecas
