// NimbleCAS extended risk / portfolio numerics + French depreciation + annuity variants.
// @author Olumuyiwa Oluwasanmi
//
// SCOPE. A NUMERICAL/STATISTICAL companion to `nimblecas.analytics` and
// `nimblecas.portfolio`, covering three families the closed-form mean-variance layer
// leaves out:
//
//   SECTION D — RISK / PORTFOLIO
//     * corr2cov / cov2corr — the correlation<->covariance change of variables.
//     * lower_partial_moment (LPM_n) and downside_deviation — below-threshold risk.
//     * ew_mean / ew_cov — exponentially-weighted (RiskMetrics-style) moments with decay
//       lambda in (0,1]; lambda == 1 degenerates to the equal-weight (population) moment.
//     * simulate_correlated_returns — reproducible correlated Gaussian draws through a
//       robust Cholesky factor (the counter-based `nimblecas.rng` gives seed-reproducible,
//       partition-independent streams).
//     * constrained mean-variance optimisation with a BOX (0 <= w <= ub, fully invested)
//       via a primal ACTIVE-SET quadratic-program solver (Nocedal & Wright Alg. 16.3),
//       with a bounded iteration budget and an honest `not_converged` on failure.
//     * cvar_optimal_weights — CVaR-minimising portfolio via the Rockafellar-Uryasev
//       LINEAR PROGRAM, solved by a self-contained two-phase dense simplex (`linprog`).
//
//   SECTION E — SCALAR DEPRECIATION
//     * amorlinc / amordegrc — the French linear and degressive methods (Excel AMORLINC /
//       AMORDEGRC), the latter with the standard life-keyed coefficient table (documented
//       at `amordegrc_coefficient`) and half-up rounding.
//
//   SECTION F — RATE / ANNUITY VARIANTS
//     * effective_continuous / nominal_continuous — the force-of-interest (continuous
//       compounding) analogues of EFFECT / NOMINAL.
//     * amortize — a full level-payment amortisation schedule (interest/principal/balance).
//     * pay_per / pay_odd / pay_uni — the level payment, the odd-first-period interest, and
//       the odd-period-capitalised uniform payment of an annuity with an irregular first
//       period.
//
// HONESTY (config/cpp_details.txt Rule 32, AGENTS.md). Every fallible routine returns
// Result<T>; nothing throws and nothing returns a NaN/inf dressed as a value. A singular
// KKT system, a non-positive-definite covariance, an infeasible constrained program, or an
// exhausted iteration budget rides the railway as domain_error / not_converged. Every dense
// matrix routine REJECTS a non-finite entry rather than letting a NaN defeat a pivot test
// and flow silently into the output. DoS-sized inputs are refused, not allocated (matrix
// dimension caps and series/period-count caps are documented at each function). This layer
// is a SAMPLE-ESTIMATE / NUMERICAL layer: outputs carry the usual estimation and
// floating-point error and are documented as such, never claimed exact.

module;
#include <cassert>

export module nimblecas.riskextra;

import std;
import nimblecas.core;
import nimblecas.rng;

export namespace nimblecas::riskextra {

// ---------------------------------------------------------------------------
// Shared result records.
// ---------------------------------------------------------------------------

// One point on a (constrained) efficient frontier: the weight vector, its portfolio
// standard deviation, and its portfolio expected return.
struct FrontierPoint {
    std::vector<double> weights;
    double risk{0.0};
    double ret{0.0};
};

// ===========================================================================
// SECTION D — risk / portfolio.
// ===========================================================================

// corr2cov: Sigma_ij = R_ij * s_i * s_j. `stddevs` must be non-negative and match the
// dimension of the (square) correlation matrix. Non-finite / mismatched / oversized
// (> 4096) inputs -> domain_error.
[[nodiscard]] auto corr2cov(std::span<const std::vector<double>> corr,
                            std::span<const double> stddevs)
    -> Result<std::vector<std::vector<double>>>;

// cov2corr: R_ij = Sigma_ij / (s_i s_j) with s_i = sqrt(Sigma_ii). A non-positive diagonal
// entry (a zero/negative variance) makes the correlation undefined -> domain_error.
[[nodiscard]] auto cov2corr(std::span<const std::vector<double>> cov)
    -> Result<std::vector<std::vector<double>>>;

// Lower partial moment of `order` (>= 0) below `threshold`:
//   LPM_order(tau) = (1/N) * sum_k max(tau - r_k, 0)^order.
// Only strictly-below-threshold observations contribute (a zero shortfall contributes 0,
// so LPM_0 is the shortfall probability). Empty series / negative order -> domain_error.
[[nodiscard]] auto lower_partial_moment(std::span<const double> returns, double order,
                                        double threshold) -> Result<double>;
// Downside deviation below `mar` = sqrt(LPM_2(mar)) (population, divisor N).
[[nodiscard]] auto downside_deviation(std::span<const double> returns, double mar)
    -> Result<double>;

// Exponentially-weighted mean of `x` with decay lambda in (0,1]: the most-recent element
// x[N-1] has weight 1 and element x[t] has weight lambda^{(N-1-t)}, normalised by their sum.
// lambda == 1 recovers the ordinary (equal-weight) mean. Empty x / lambda out of (0,1] ->
// domain_error.
[[nodiscard]] auto ew_mean(std::span<const double> x, double lambda) -> Result<double>;
// Exponentially-weighted covariance of x,y under the same weighting, with the UNBIASED
// reliability-weights normalisation (divisor W1 - W2/W1, W1 = sum w, W2 = sum w^2), so
// lambda == 1 recovers the ordinary (n-1) SAMPLE covariance. Mismatched/short -> domain_error.
[[nodiscard]] auto ew_cov(std::span<const double> x, std::span<const double> y, double lambda)
    -> Result<double>;
// Exponentially-weighted covariance matrix of a set of series (assets in rows, observations
// in columns). lambda == 1 -> the (n-1) sample covariance matrix. Ragged/oversized (> 4096)
// -> domain_error.
[[nodiscard]] auto ew_covariance_matrix(std::span<const std::vector<double>> series,
                                        double lambda)
    -> Result<std::vector<std::vector<double>>>;

// Draw `n` correlated Gaussian return vectors x = mean + L z (z ~ N(0, I), Sigma = L L' the
// lower Cholesky factor), reproducible for a fixed `seed` via the counter-based RNG. Returns
// an n x d matrix (rows are samples, columns are assets). A non-positive-definite covariance
// is REJECTED (domain_error) rather than factored into garbage; non-finite entries, a
// dimension mismatch, dimension > 4096, or n > 100000 -> domain_error.
[[nodiscard]] auto simulate_correlated_returns(std::span<const double> mean,
                                               std::span<const std::vector<double>> cov,
                                               std::size_t n, std::uint64_t seed)
    -> Result<std::vector<std::vector<double>>>;

// --- Linear program (self-contained two-phase dense simplex) ----------------
// Minimise c . x subject to  A_le x <= b_le,  A_eq x = b_eq,  x >= 0.
// Bland's rule guarantees termination; the tableau is dense so the dimensions are bounded
// (variables + constraints <= 8192) to keep a solve well-bounded. Also used internally to
// seed the active-set QP and to solve the CVaR program.
enum class LinProgStatus : std::uint8_t { optimal, infeasible, unbounded };
struct LinProgResult {
    LinProgStatus status{LinProgStatus::optimal};
    double objective{0.0};
    std::vector<double> x{};
};
[[nodiscard]] auto linprog(std::span<const double> c,
                           std::span<const std::vector<double>> A_le,
                           std::span<const double> b_le,
                           std::span<const std::vector<double>> A_eq,
                           std::span<const double> b_eq) -> Result<LinProgResult>;

// --- Box-constrained mean-variance optimisation (active-set QP) --------------
// Global minimum-variance portfolio subject to a box (lower <= w <= upper, default lower 0)
// and full investment (sum w = 1). `upper` sizes the problem (== cov dimension). Infeasible
// box (e.g. sum of uppers < 1) or a singular/indefinite KKT -> domain_error / not_converged.
// Dimension is capped at 512 (the QP builds an O(n)-constraint dense KKT system).
[[nodiscard]] auto constrained_min_variance(std::span<const std::vector<double>> cov,
                                            std::span<const double> upper,
                                            std::span<const double> lower = {})
    -> Result<std::vector<double>>;
// Minimum-variance box-constrained portfolio achieving `target_return` (adds the equality
// mu . w = target_return). Infeasible target under the box -> domain_error.
[[nodiscard]] auto constrained_efficient_portfolio(std::span<const std::vector<double>> cov,
                                                   std::span<const double> mean_returns,
                                                   double target_return,
                                                   std::span<const double> upper,
                                                   std::span<const double> lower = {})
    -> Result<FrontierPoint>;
// A box-constrained efficient frontier of `points` (>= 2) portfolios, sampled at target
// returns evenly spaced between the box-constrained global-min-variance return and the
// maximum return attainable under the box (both found honestly, the latter by an LP).
[[nodiscard]] auto constrained_frontier(std::span<const std::vector<double>> cov,
                                        std::span<const double> mean_returns, int points,
                                        std::span<const double> upper,
                                        std::span<const double> lower = {})
    -> Result<std::vector<FrontierPoint>>;

// --- CVaR-minimising portfolio (Rockafellar-Uryasev LP) ---------------------
// Given `scenarios` (S rows, each an N-vector of per-asset returns in that scenario) and a
// confidence `beta` in [0,1), minimise the beta-CVaR of the portfolio loss over fully
// invested long-only weights (optionally capped by `upper`). Returns the weights, the
// minimised CVaR (the RU objective value), and the optimal VaR (the RU alpha). The loss in
// scenario k is L_k = -(w . r_k); CVaR_beta is the mean of the worst (1-beta) tail. The dense
// simplex bounds the instance to S <= 1024 scenarios and N <= 128 assets.
struct CVaRResult {
    std::vector<double> weights;
    double cvar{0.0};
    double var{0.0};
};
[[nodiscard]] auto cvar_optimal_weights(std::span<const std::vector<double>> scenarios,
                                        double beta, std::span<const double> upper = {})
    -> Result<CVaRResult>;

// ===========================================================================
// SECTION E — French depreciation (AMORLINC / AMORDEGRC).
// ===========================================================================

// AMORDEGRC accelerates the straight-line rate 1/life by a coefficient keyed on the asset
// life (life == 1/rate). The standard French / Excel table (documented):
//     life < 3 years        -> 1.0   (no acceleration; safe fallback)
//     3 <= life <= 4 years  -> 1.5
//     5 <= life <= 6 years  -> 2.0
//     life > 6 years        -> 2.5
// (The 4 < life < 5 gap maps to 1.5, matching the LibreOffice/Excel `< 5` boundary.)
[[nodiscard]] auto amordegrc_coefficient(double asset_life) -> double;

// AMORLINC — French linear (prorated straight-line) depreciation for accounting `period`
// (0-based; period 0 is the first, partial period). `first_period_fraction` in (0,1] is the
// fraction of a full year the asset was held in that first period (what Excel derives from
// the purchase and first-period dates and the day-count basis). Each full period depreciates
// cost*rate; the first is prorated and the final period takes the remainder so the total is
// exactly cost - salvage. rate <= 0, cost <= 0, salvage < 0 or > cost, period < 0, or
// first_period_fraction outside (0,1] -> domain_error.
[[nodiscard]] auto amorlinc(double cost, double salvage, std::int64_t period, double rate,
                            double first_period_fraction) -> Result<double>;
// AMORDEGRC — French degressive depreciation for accounting `period` (0-based). The rate is
// the accelerated rate coefficient*rate (see amordegrc_coefficient); each period's charge is
// round-half-up(accelerated_rate * book), switching to straight-line over the remaining life
// once that deducts more, and never dropping the book value below salvage. Period 0 is
// prorated by first_period_fraction. Same domain guards as amorlinc.
[[nodiscard]] auto amordegrc(double cost, double salvage, std::int64_t period, double rate,
                             double first_period_fraction) -> Result<double>;

// ===========================================================================
// SECTION F — continuous-compounding rate conversion & annuity variants.
// ===========================================================================

// Effective annual rate under CONTINUOUS compounding (force of interest r): e^r - 1.
[[nodiscard]] auto effective_continuous(double nominal_rate) -> Result<double>;
// Inverse: the continuously-compounded nominal rate (force of interest) giving an effective
// annual rate: ln(1 + effective). effective <= -1 -> domain_error (log of a non-positive).
[[nodiscard]] auto nominal_continuous(double effective_rate) -> Result<double>;

// A level-payment amortisation schedule (ordinary annuity, end-of-period). `payment` is the
// constant payment; interest[t]/principal[t]/balance[t] are the split and the remaining
// balance AFTER payment t (balance.back() ~ 0). nper < 1, nper > 100000, principal <= 0, or
// rate <= -1 -> domain_error.
struct AmortSchedule {
    double payment{0.0};
    std::vector<double> interest{};
    std::vector<double> principal{};
    std::vector<double> balance{};
};
[[nodiscard]] auto amortize(double rate, std::int64_t nper, double principal)
    -> Result<AmortSchedule>;

// pay_per — the level periodic payment fully amortising `pv` over `nper` periods at per-period
// `rate` (ordinary annuity, fv 0): pv*r/(1-(1+r)^-n), or pv/n when r == 0.
[[nodiscard]] auto pay_per(double rate, std::int64_t nper, double pv) -> Result<double>;
// pay_odd — the interest accruing over an IRREGULAR first period of length
// `odd_period_fraction` (in periods) before regular amortisation begins: simple interest
// pv*r*odd (default) or compounded pv*((1+r)^odd - 1). This is the "odd-days interest" of a
// loan whose first instalment is not a whole period away.
[[nodiscard]] auto pay_odd(double rate, double odd_period_fraction, double pv,
                           bool simple = true) -> Result<double>;
// pay_uni — the uniform (level) payment when the odd-period interest is CAPITALISED into the
// principal: pay_per(rate, nper, pv + pay_odd(...)). odd_period_fraction == 0 recovers
// pay_per(rate, nper, pv).
[[nodiscard]] auto pay_uni(double rate, std::int64_t nper, double pv,
                           double odd_period_fraction, bool simple = true) -> Result<double>;

}  // namespace nimblecas::riskextra

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::riskextra {
namespace {

constexpr std::size_t kMaxMatrixDim = 4096;   // dense covariance / correlation cap (DoS guard)
constexpr std::size_t kMaxQpDim = 512;        // active-set QP asset cap (dense KKT is O(n)-sized)
constexpr std::size_t kMaxLpDim = 8192;       // simplex variables + constraints cap
constexpr std::size_t kMaxScenarios = 1024;   // CVaR scenario cap (dense simplex; each scenario
                                              // adds an LP variable and a constraint, kept well
                                              // inside kMaxLpDim so a within-cap instance always
                                              // fits the tableau)
constexpr std::size_t kMaxCvarAssets = 128;   // CVaR asset cap
constexpr std::int64_t kMaxPeriods = 100'000; // amortisation period cap

// --- small matrix predicates -----------------------------------------------

[[nodiscard]] auto is_square_finite(std::span<const std::vector<double>> m) -> bool {
    const std::size_t n = m.size();
    if (n == 0 || n > kMaxMatrixDim) { return false; }
    for (const auto& row : m) {
        if (row.size() != n) { return false; }
        for (double v : row) {
            if (!std::isfinite(v)) { return false; }
        }
    }
    return true;
}

// --- dense unsymmetric solve (Gaussian elimination, partial pivoting) -------
// Solves A x = b for a general (possibly indefinite — the QP's KKT block is) square A.
// Rejects any non-finite entry so a NaN cannot defeat the |pivot| test and flow into x.
[[nodiscard]] auto dense_solve(std::vector<std::vector<double>> a, std::vector<double> b)
    -> std::optional<std::vector<double>> {
    const std::size_t n = a.size();
    if (n == 0 || b.size() != n) { return std::nullopt; }
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i].size() != n) { return std::nullopt; }
        for (double v : a[i]) {
            if (!std::isfinite(v)) { return std::nullopt; }
        }
        if (!std::isfinite(b[i])) { return std::nullopt; }
    }
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t piv = k;
        double best = std::abs(a[k][k]);
        for (std::size_t i = k + 1; i < n; ++i) {
            if (std::abs(a[i][k]) > best) { best = std::abs(a[i][k]); piv = i; }
        }
        if (best < 1e-300) { return std::nullopt; }  // singular
        if (piv != k) { std::swap(a[k], a[piv]); std::swap(b[k], b[piv]); }
        for (std::size_t i = k + 1; i < n; ++i) {
            const double f = a[i][k] / a[k][k];
            if (f == 0.0) { continue; }
            for (std::size_t j = k; j < n; ++j) { a[i][j] -= f * a[k][j]; }
            b[i] -= f * b[k];
        }
    }
    std::vector<double> x(n, 0.0);
    for (std::size_t ii = n; ii-- > 0;) {
        double s = b[ii];
        for (std::size_t j = ii + 1; j < n; ++j) { s -= a[ii][j] * x[j]; }
        x[ii] = s / a[ii][ii];
        if (!std::isfinite(x[ii])) { return std::nullopt; }
    }
    return x;
}

// --- lower Cholesky (robust relative pivot floor; rejects non-PD/non-finite) -
// Matches the honesty contract of the analytics Cholesky: refuses a positive-SEMI-definite
// (collinear) matrix whose Schur pivot is a tiny positive float, and refuses a NaN pivot.
[[nodiscard]] auto cholesky_lower(std::span<const std::vector<double>> m)
    -> std::optional<std::vector<std::vector<double>>> {
    const std::size_t n = m.size();
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double s = m[i][j];
            for (std::size_t k = 0; k < j; ++k) { s -= L[i][k] * L[j][k]; }
            if (i == j) {
                const double pivot_floor = 1e-12 * std::abs(m[i][i]);
                if (!std::isfinite(s) || s <= pivot_floor) { return std::nullopt; }
                L[i][j] = std::sqrt(s);
            } else {
                L[i][j] = s / L[j][j];
            }
        }
    }
    return L;
}

// --- two-phase dense simplex core -------------------------------------------
// A single constraint after sign-normalisation (rhs made >= 0).
struct Con {
    std::vector<double> a;  // coefficients over the n structural variables
    int rel;                // 0: <=, 1: ==, 2: >=
    double rhs;             // >= 0
};

// Runs the simplex (minimisation) on a canonical tableau. Returns 0 optimal, 1 unbounded,
// 2 iteration budget exhausted. `forbidden[j]` blocks column j from entering the basis
// (used to freeze artificial columns out of phase 2). Entering column: smallest index with a
// strictly-negative reduced cost (Bland). Leaving row: min-ratio, ties broken by smallest
// basic index (Bland) — together these guarantee termination.
[[nodiscard]] auto simplex_run(std::vector<std::vector<double>>& tab, std::vector<double>& rhs,
                               std::vector<std::size_t>& basis, const std::vector<double>& cost,
                               const std::vector<bool>& forbidden, int max_iter) -> int {
    const std::size_t m = tab.size();
    if (m == 0) { return 0; }
    const std::size_t total = tab[0].size();
    for (int iter = 0; iter < max_iter; ++iter) {
        std::size_t entering = total;
        for (std::size_t j = 0; j < total; ++j) {
            if (forbidden[j]) { continue; }
            double red = cost[j];
            for (std::size_t i = 0; i < m; ++i) { red -= cost[basis[i]] * tab[i][j]; }
            if (red < -1e-9) { entering = j; break; }
        }
        if (entering == total) { return 0; }  // optimal
        std::size_t leaving = m;
        double best_ratio = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            const double col = tab[i][entering];
            if (col <= 1e-9) { continue; }
            const double ratio = rhs[i] / col;
            if (leaving == m || ratio < best_ratio - 1e-12 ||
                (std::abs(ratio - best_ratio) <= 1e-12 && basis[i] < basis[leaving])) {
                leaving = i;
                best_ratio = ratio;
            }
        }
        if (leaving == m) { return 1; }  // unbounded
        const double piv = tab[leaving][entering];
        for (std::size_t j = 0; j < total; ++j) { tab[leaving][j] /= piv; }
        rhs[leaving] /= piv;
        for (std::size_t i = 0; i < m; ++i) {
            if (i == leaving) { continue; }
            const double f = tab[i][entering];
            if (f == 0.0) { continue; }
            for (std::size_t j = 0; j < total; ++j) { tab[i][j] -= f * tab[leaving][j]; }
            rhs[i] -= f * rhs[leaving];
        }
        basis[leaving] = entering;
    }
    return 2;  // budget exhausted
}

// Solves min c.x s.t. the (already sign-normalised) constraint list, x >= 0, via two phases.
[[nodiscard]] auto solve_lp(std::size_t n, const std::vector<Con>& cons,
                            std::span<const double> c) -> Result<LinProgResult> {
    const std::size_t m = cons.size();
    // No constraints: the feasible region is just x >= 0. Any strictly-negative objective
    // coefficient lets that variable grow without bound; otherwise the optimum is x = 0.
    if (m == 0) {
        for (double v : c) {
            if (v < -1e-12) { return LinProgResult{LinProgStatus::unbounded, 0.0, {}}; }
        }
        return LinProgResult{LinProgStatus::optimal, 0.0, std::vector<double>(n, 0.0)};
    }
    // Count auxiliary columns.
    std::size_t extra = 0;
    for (const auto& con : cons) {
        if (con.rel == 0) { extra += 1; }        // slack
        else if (con.rel == 2) { extra += 2; }   // surplus + artificial
        else { extra += 1; }                     // artificial
    }
    const std::size_t total = n + extra;
    if (total > kMaxLpDim || m > kMaxLpDim) {
        return make_error<LinProgResult>(MathError::domain_error);
    }
    std::vector<std::vector<double>> tab(m, std::vector<double>(total, 0.0));
    std::vector<double> rhs(m, 0.0);
    std::vector<std::size_t> basis(m, 0);
    std::vector<bool> is_art(total, false);
    std::size_t col = n;
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) { tab[i][j] = cons[i].a[j]; }
        rhs[i] = cons[i].rhs;
        if (cons[i].rel == 0) {                 // <=
            tab[i][col] = 1.0; basis[i] = col; ++col;
        } else if (cons[i].rel == 2) {          // >=
            tab[i][col] = -1.0; ++col;          // surplus
            tab[i][col] = 1.0; is_art[col] = true; basis[i] = col; ++col;
        } else {                                // ==
            tab[i][col] = 1.0; is_art[col] = true; basis[i] = col; ++col;
        }
    }
    // Phase 1: minimise the sum of artificials.
    std::vector<double> cost1(total, 0.0);
    for (std::size_t j = 0; j < total; ++j) { if (is_art[j]) { cost1[j] = 1.0; } }
    std::vector<bool> none(total, false);
    const int budget = static_cast<int>(20 * (m + total) + 1000);
    if (simplex_run(tab, rhs, basis, cost1, none, budget) == 2) {
        return make_error<LinProgResult>(MathError::not_converged);
    }
    double phase1_obj = 0.0;
    for (std::size_t i = 0; i < m; ++i) { phase1_obj += cost1[basis[i]] * rhs[i]; }
    if (phase1_obj > 1e-7) {
        return LinProgResult{LinProgStatus::infeasible, 0.0, {}};
    }
    // Drive any basic artificials out of the basis where the row permits it (keeps a stray
    // artificial from re-inflating during phase 2).
    for (std::size_t i = 0; i < m; ++i) {
        if (!is_art[basis[i]]) { continue; }
        for (std::size_t j = 0; j < total; ++j) {
            if (is_art[j] || std::abs(tab[i][j]) <= 1e-9) { continue; }
            const double piv = tab[i][j];
            for (std::size_t k = 0; k < total; ++k) { tab[i][k] /= piv; }
            rhs[i] /= piv;
            for (std::size_t r = 0; r < m; ++r) {
                if (r == i) { continue; }
                const double f = tab[r][j];
                if (f == 0.0) { continue; }
                for (std::size_t k = 0; k < total; ++k) { tab[r][k] -= f * tab[i][k]; }
                rhs[r] -= f * rhs[i];
            }
            basis[i] = j;
            break;
        }
    }
    // Phase 2: minimise the real objective; artificials are frozen out.
    std::vector<double> cost2(total, 0.0);
    for (std::size_t j = 0; j < n; ++j) { cost2[j] = c[j]; }
    const int status = simplex_run(tab, rhs, basis, cost2, is_art, budget);
    if (status == 2) { return make_error<LinProgResult>(MathError::not_converged); }
    if (status == 1) { return LinProgResult{LinProgStatus::unbounded, 0.0, {}}; }
    std::vector<double> x(n, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        if (basis[i] < n) { x[basis[i]] = rhs[i]; }
    }
    double obj = 0.0;
    for (std::size_t j = 0; j < n; ++j) { obj += c[j] * x[j]; }
    return LinProgResult{LinProgStatus::optimal, obj, std::move(x)};
}

// --- active-set box QP ------------------------------------------------------
// A linear equality a.x = b or inequality a.x >= b.
struct LinCon {
    std::vector<double> a;
    double b;
};

// Minimise 1/2 x'G x subject to the equalities `eqs` and the box lo <= x <= hi, starting from
// a feasible x0. Primal active-set method (Nocedal & Wright, Alg. 16.3): box faces enter the
// working set as they block a step and leave when their multiplier turns strictly negative.
// Returns nullopt (mapped to not_converged by callers) if a KKT solve is singular or the
// iteration budget is exhausted.
[[nodiscard]] auto active_set_box_qp(const std::vector<std::vector<double>>& G,
                                     const std::vector<LinCon>& eqs,
                                     const std::vector<double>& lo, const std::vector<double>& hi,
                                     std::vector<double> x) -> std::optional<std::vector<double>> {
    const std::size_t n = G.size();
    const std::size_t n_eq = eqs.size();
    // Inequality faces: x_i >= lo_i  (a = +e_i, b = lo_i) and x_i <= hi_i  (a = -e_i, b = -hi_i).
    struct Face { std::size_t idx; int sign; double bound; };  // sign +1 lower, -1 upper
    std::vector<Face> faces;
    faces.reserve(2 * n);
    for (std::size_t i = 0; i < n; ++i) { faces.push_back({i, +1, lo[i]}); }
    for (std::size_t i = 0; i < n; ++i) { faces.push_back({i, -1, hi[i]}); }
    const auto face_dot = [&](const Face& f, const std::vector<double>& v) -> double {
        return static_cast<double>(f.sign) * v[f.idx];
    };
    const auto face_val = [&](const Face& f) -> double { return static_cast<double>(f.sign) * f.bound; };

    std::vector<bool> in_ws(faces.size(), false);  // active inequality faces

    const int max_iter = 500;
    for (int iter = 0; iter < max_iter; ++iter) {
        // Assemble the working set A_W: equalities first, then active faces.
        std::vector<std::vector<double>> aw;  // rows
        std::vector<std::size_t> ws_face;     // face index per active-face row
        aw.reserve(n_eq + faces.size());
        for (const auto& e : eqs) { aw.push_back(e.a); }
        for (std::size_t fi = 0; fi < faces.size(); ++fi) {
            if (!in_ws[fi]) { continue; }
            std::vector<double> row(n, 0.0);
            row[faces[fi].idx] = static_cast<double>(faces[fi].sign);
            aw.push_back(std::move(row));
            ws_face.push_back(fi);
        }
        const std::size_t m_a = aw.size();
        const std::size_t dim = n + m_a;
        // Gradient g = G x.
        std::vector<double> g(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < n; ++j) { s += G[i][j] * x[j]; }
            g[i] = s;
        }
        // KKT system  [ G   -A_W' ] [ p ]   [ -g ]
        //             [ A_W   0   ] [ lam ] = [ 0 ].
        std::vector<std::vector<double>> K(dim, std::vector<double>(dim, 0.0));
        std::vector<double> rhs(dim, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) { K[i][j] = G[i][j]; }
            for (std::size_t r = 0; r < m_a; ++r) { K[i][n + r] = -aw[r][i]; }
            rhs[i] = -g[i];
        }
        for (std::size_t r = 0; r < m_a; ++r) {
            for (std::size_t j = 0; j < n; ++j) { K[n + r][j] = aw[r][j]; }
        }
        auto sol = dense_solve(std::move(K), rhs);
        if (!sol) { return std::nullopt; }
        std::vector<double> p((*sol).begin(), (*sol).begin() + static_cast<std::ptrdiff_t>(n));
        double pnorm = 0.0;
        for (double v : p) { pnorm += v * v; }
        pnorm = std::sqrt(pnorm);

        if (pnorm < 1e-11) {
            // Stationary on the current working set: check inequality multipliers. lam for the
            // active faces sits in sol[n + n_eq ...]; a strictly-negative one means releasing
            // that face lowers the objective.
            double most_neg = -1e-12;
            std::size_t drop_row = m_a;
            for (std::size_t r = 0; r < ws_face.size(); ++r) {
                const double lam = (*sol)[n + n_eq + r];
                if (lam < most_neg) { most_neg = lam; drop_row = r; }
            }
            if (drop_row == m_a) { return x; }  // KKT satisfied -> optimal
            in_ws[ws_face[drop_row]] = false;   // release and continue
            continue;
        }
        // Ratio test: shrink the step so no inactive face is violated.
        double alpha = 1.0;
        std::size_t block = faces.size();
        for (std::size_t fi = 0; fi < faces.size(); ++fi) {
            if (in_ws[fi]) { continue; }
            const double ap = face_dot(faces[fi], p);
            if (ap >= -1e-14) { continue; }  // moving into the feasible side (or parallel)
            const double slack = face_dot(faces[fi], x) - face_val(faces[fi]);  // >= 0
            const double t = slack / (-ap);
            if (t < alpha - 1e-15) { alpha = std::max(t, 0.0); block = fi; }
        }
        for (std::size_t i = 0; i < n; ++i) { x[i] += alpha * p[i]; }
        if (block != faces.size()) { in_ws[block] = true; }
    }
    return std::nullopt;  // budget exhausted
}

// Build the box vectors, validate, seed a feasible point via linprog, and run the QP.
[[nodiscard]] auto solve_box_qp(std::span<const std::vector<double>> cov,
                                const std::vector<LinCon>& eqs, std::span<const double> upper,
                                std::span<const double> lower)
    -> Result<std::vector<double>> {
    if (!is_square_finite(cov)) { return make_error<std::vector<double>>(MathError::domain_error); }
    const std::size_t n = cov.size();
    if (n > kMaxQpDim || upper.size() != n) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    std::vector<double> lo(n, 0.0);
    std::vector<double> hi(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!lower.empty()) {
            if (i >= lower.size() || !std::isfinite(lower[i])) {
                return make_error<std::vector<double>>(MathError::domain_error);
            }
            lo[i] = lower[i];
        }
        if (!std::isfinite(upper[i]) || upper[i] < lo[i]) {
            return make_error<std::vector<double>>(MathError::domain_error);
        }
        hi[i] = upper[i];
    }
    // Feasibility LP: minimise 0 s.t. box (via <= and >= rows) and the equalities.
    std::vector<double> zero_c(n, 0.0);
    std::vector<std::vector<double>> A_le;
    std::vector<double> b_le;
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> row(n, 0.0); row[i] = 1.0;  // x_i <= hi_i
        A_le.push_back(row); b_le.push_back(hi[i]);
        if (lo[i] != 0.0) {                              // x_i >= lo_i
            std::vector<double> rl(n, 0.0); rl[i] = -1.0;
            A_le.push_back(rl); b_le.push_back(-lo[i]);
        }
    }
    std::vector<std::vector<double>> A_eq;
    std::vector<double> b_eq;
    for (const auto& e : eqs) { A_eq.push_back(e.a); b_eq.push_back(e.b); }
    auto feas = solve_lp(n, [&] {
        std::vector<Con> cons;
        for (std::size_t i = 0; i < A_le.size(); ++i) {
            double r = b_le[i]; std::vector<double> a = A_le[i]; int rel = 0;
            if (r < 0.0) { for (double& v : a) { v = -v; } r = -r; rel = 2; }
            cons.push_back({std::move(a), rel, r});
        }
        for (std::size_t i = 0; i < A_eq.size(); ++i) {
            double r = b_eq[i]; std::vector<double> a = A_eq[i];
            if (r < 0.0) { for (double& v : a) { v = -v; } r = -r; }
            cons.push_back({std::move(a), 1, r});
        }
        return cons;
    }(), zero_c);
    if (!feas) { return make_error<std::vector<double>>(feas.error()); }
    if (feas->status != LinProgStatus::optimal) {
        return make_error<std::vector<double>>(MathError::domain_error);  // infeasible box/target
    }
    const std::vector<std::vector<double>> G(cov.begin(), cov.end());
    std::vector<LinCon> eq_copy = eqs;
    auto w = active_set_box_qp(G, eq_copy, lo, hi, feas->x);
    if (!w) { return make_error<std::vector<double>>(MathError::not_converged); }
    // Clamp tiny box excursions from round-off back into [lo, hi].
    for (std::size_t i = 0; i < n; ++i) {
        (*w)[i] = std::min(std::max((*w)[i], lo[i]), hi[i]);
        if (!std::isfinite((*w)[i])) { return make_error<std::vector<double>>(MathError::not_converged); }
    }
    return *w;
}

[[nodiscard]] auto quad_form(std::span<const double> w, std::span<const std::vector<double>> cov)
    -> double {
    double s = 0.0;
    for (std::size_t i = 0; i < w.size(); ++i) {
        for (std::size_t j = 0; j < w.size(); ++j) { s += w[i] * cov[i][j] * w[j]; }
    }
    return s;
}

// Fill `out` with independent standard normals via Box-Muller from the RNG's uniform draws.
auto fill_normals(nimblecas::Rng& rng, std::span<double> out) -> void {
    const double two_pi = 2.0 * std::numbers::pi;
    const auto pos_unit = [&]() -> double {
        const double u = rng.next_unit();
        return u <= 0.0 ? std::numeric_limits<double>::min() : u;
    };
    std::size_t i = 0;
    for (; i + 1 < out.size(); i += 2) {
        const double r = std::sqrt(-2.0 * std::log(pos_unit()));
        const double a = two_pi * rng.next_unit();
        out[i] = r * std::cos(a);
        out[i + 1] = r * std::sin(a);
    }
    if (i < out.size()) {
        const double r = std::sqrt(-2.0 * std::log(pos_unit()));
        const double a = two_pi * rng.next_unit();
        out[i] = r * std::cos(a);
    }
}

[[nodiscard]] auto round_half_up(double x) -> double { return std::floor(x + 0.5); }

}  // namespace

// --- corr <-> cov -----------------------------------------------------------

auto corr2cov(std::span<const std::vector<double>> corr, std::span<const double> stddevs)
    -> Result<std::vector<std::vector<double>>> {
    if (!is_square_finite(corr)) {
        return make_error<std::vector<std::vector<double>>>(MathError::domain_error);
    }
    const std::size_t n = corr.size();
    if (stddevs.size() != n) {
        return make_error<std::vector<std::vector<double>>>(MathError::domain_error);
    }
    for (double s : stddevs) {
        if (!std::isfinite(s) || s < 0.0) {
            return make_error<std::vector<std::vector<double>>>(MathError::domain_error);
        }
    }
    std::vector<std::vector<double>> cov(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) { cov[i][j] = corr[i][j] * stddevs[i] * stddevs[j]; }
    }
    return cov;
}

auto cov2corr(std::span<const std::vector<double>> cov)
    -> Result<std::vector<std::vector<double>>> {
    if (!is_square_finite(cov)) {
        return make_error<std::vector<std::vector<double>>>(MathError::domain_error);
    }
    const std::size_t n = cov.size();
    std::vector<double> sd(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (cov[i][i] <= 0.0) {  // zero/negative variance -> correlation undefined
            return make_error<std::vector<std::vector<double>>>(MathError::domain_error);
        }
        sd[i] = std::sqrt(cov[i][i]);
    }
    std::vector<std::vector<double>> corr(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) { corr[i][j] = cov[i][j] / (sd[i] * sd[j]); }
    }
    return corr;
}

// --- lower partial moment / downside deviation ------------------------------

auto lower_partial_moment(std::span<const double> returns, double order, double threshold)
    -> Result<double> {
    if (returns.empty() || !(order >= 0.0) || !std::isfinite(threshold)) {
        return make_error<double>(MathError::domain_error);
    }
    double acc = 0.0;
    for (double r : returns) {
        if (!std::isfinite(r)) { return make_error<double>(MathError::domain_error); }
        const double shortfall = threshold - r;
        if (shortfall > 0.0) { acc += std::pow(shortfall, order); }
    }
    const double lpm = acc / static_cast<double>(returns.size());
    if (!std::isfinite(lpm)) { return make_error<double>(MathError::overflow); }
    return lpm;
}

auto downside_deviation(std::span<const double> returns, double mar) -> Result<double> {
    return lower_partial_moment(returns, 2.0, mar).transform(
        [](double v) { return std::sqrt(v); });
}

// --- exponentially-weighted moments -----------------------------------------

auto ew_mean(std::span<const double> x, double lambda) -> Result<double> {
    if (x.empty() || !(lambda > 0.0 && lambda <= 1.0)) {
        return make_error<double>(MathError::domain_error);
    }
    const std::size_t n = x.size();
    double wsum = 0.0;
    double acc = 0.0;
    double w = 1.0;  // weight of the most recent element x[n-1]; older elements get lambda^k
    for (std::size_t k = 0; k < n; ++k) {
        const double v = x[n - 1 - k];
        if (!std::isfinite(v)) { return make_error<double>(MathError::domain_error); }
        acc += w * v;
        wsum += w;
        w *= lambda;
    }
    if (wsum == 0.0) { return make_error<double>(MathError::division_by_zero); }
    const double m = acc / wsum;
    if (!std::isfinite(m)) { return make_error<double>(MathError::overflow); }
    return m;
}

auto ew_cov(std::span<const double> x, std::span<const double> y, double lambda) -> Result<double> {
    const std::size_t n = x.size();
    if (n < 2 || y.size() != n || !(lambda > 0.0 && lambda <= 1.0)) {
        return make_error<double>(MathError::domain_error);
    }
    auto mx = ew_mean(x, lambda);
    auto my = ew_mean(y, lambda);
    if (!mx || !my) { return make_error<double>(MathError::domain_error); }
    double w1 = 0.0;   // sum of weights
    double w2 = 0.0;   // sum of squared weights
    double acc = 0.0;
    double w = 1.0;
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t t = n - 1 - k;
        if (!std::isfinite(x[t]) || !std::isfinite(y[t])) {
            return make_error<double>(MathError::domain_error);
        }
        acc += w * (x[t] - *mx) * (y[t] - *my);
        w1 += w;
        w2 += w * w;
        w *= lambda;
    }
    // Unbiased reliability-weights denominator; == n-1 when every weight is 1 (lambda == 1),
    // so the estimator reduces to the ordinary sample covariance in that limit.
    const double denom = w1 - w2 / w1;
    if (!(denom > 0.0)) { return make_error<double>(MathError::division_by_zero); }
    const double c = acc / denom;
    if (!std::isfinite(c)) { return make_error<double>(MathError::overflow); }
    return c;
}

auto ew_covariance_matrix(std::span<const std::vector<double>> series, double lambda)
    -> Result<std::vector<std::vector<double>>> {
    if (series.empty() || series.size() > kMaxMatrixDim || !(lambda > 0.0 && lambda <= 1.0)) {
        return make_error<std::vector<std::vector<double>>>(MathError::domain_error);
    }
    const std::size_t k = series.size();
    const std::size_t obs = series[0].size();
    for (const auto& s : series) {
        if (s.size() != obs) {
            return make_error<std::vector<std::vector<double>>>(MathError::domain_error);
        }
    }
    std::vector<std::vector<double>> cov(k, std::vector<double>(k, 0.0));
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = i; j < k; ++j) {
            auto c = ew_cov(series[i], series[j], lambda);
            if (!c) { return make_error<std::vector<std::vector<double>>>(c.error()); }
            cov[i][j] = *c;
            cov[j][i] = *c;
        }
    }
    return cov;
}

// --- correlated Gaussian simulation -----------------------------------------

auto simulate_correlated_returns(std::span<const double> mean,
                                 std::span<const std::vector<double>> cov, std::size_t n,
                                 std::uint64_t seed)
    -> Result<std::vector<std::vector<double>>> {
    if (!is_square_finite(cov)) {
        return make_error<std::vector<std::vector<double>>>(MathError::domain_error);
    }
    const std::size_t d = cov.size();
    // Bound the OUTPUT PRODUCT n*d, not just each factor: 100000 rows * 4096 cols would
    // materialize ~3.3 GB. 1e7 cells (~80 MB of doubles) is a generous, safe ceiling.
    constexpr std::size_t kMaxSimCells = 10'000'000;
    if (mean.size() != d || n == 0 || n > 100'000 || (d != 0 && n > kMaxSimCells / d)) {
        return make_error<std::vector<std::vector<double>>>(MathError::domain_error);
    }
    for (double v : mean) {
        if (!std::isfinite(v)) {
            return make_error<std::vector<std::vector<double>>>(MathError::domain_error);
        }
    }
    auto L = cholesky_lower(cov);
    if (!L) { return make_error<std::vector<std::vector<double>>>(MathError::domain_error); }  // non-PD
    nimblecas::Rng rng = nimblecas::Rng::seeded(seed);
    std::vector<std::vector<double>> out(n, std::vector<double>(d, 0.0));
    std::vector<double> z(d, 0.0);
    for (std::size_t s = 0; s < n; ++s) {
        fill_normals(rng, z);
        for (std::size_t i = 0; i < d; ++i) {
            double v = mean[i];
            for (std::size_t k = 0; k <= i; ++k) { v += (*L)[i][k] * z[k]; }
            out[s][i] = v;
        }
    }
    return out;
}

// --- linprog ----------------------------------------------------------------

auto linprog(std::span<const double> c, std::span<const std::vector<double>> A_le,
             std::span<const double> b_le, std::span<const std::vector<double>> A_eq,
             std::span<const double> b_eq) -> Result<LinProgResult> {
    const std::size_t n = c.size();
    if (n == 0 || n > kMaxLpDim) { return make_error<LinProgResult>(MathError::domain_error); }
    for (double v : c) {
        if (!std::isfinite(v)) { return make_error<LinProgResult>(MathError::domain_error); }
    }
    if (A_le.size() != b_le.size() || A_eq.size() != b_eq.size()) {
        return make_error<LinProgResult>(MathError::domain_error);
    }
    // Bound the tableau PRODUCT m*n: two individually-in-range dimensions (e.g. 8192 vars x
    // 8192 constraints, 6.7e7 cells) would build a ~500 MB tableau and a pivot loop long enough
    // to be an effective hang. The ceiling must stay ABOVE the largest LP the in-house callers
    // build so the documented "within-cap always fits" invariant (see kMaxScenarios) holds: a
    // CVaR at its caps (S=1024, N=128) is ~1.33e6 cells, so 4e6 clears it with headroom while
    // still refusing the 8192^2 case. Larger is refused honestly.
    const std::size_t m = A_le.size() + A_eq.size();
    constexpr std::size_t kMaxLpCells = 4'000'000;
    if (m > kMaxLpDim || (m != 0 && n > kMaxLpCells / m)) {
        return make_error<LinProgResult>(MathError::domain_error);
    }
    std::vector<Con> cons;
    const auto add = [&](std::span<const std::vector<double>> A, std::span<const double> b,
                         bool eq) -> bool {
        for (std::size_t i = 0; i < A.size(); ++i) {
            if (A[i].size() != n || !std::isfinite(b[i])) { return false; }
            std::vector<double> a = A[i];
            double r = b[i];
            int rel = eq ? 1 : 0;
            for (double v : a) { if (!std::isfinite(v)) { return false; } }
            if (r < 0.0) {                         // normalise rhs >= 0
                for (double& v : a) { v = -v; }
                r = -r;
                if (!eq) { rel = 2; }              // flipped <= becomes >=
            }
            cons.push_back({std::move(a), rel, r});
        }
        return true;
    };
    if (!add(A_le, b_le, false) || !add(A_eq, b_eq, true)) {
        return make_error<LinProgResult>(MathError::domain_error);
    }
    return solve_lp(n, cons, c);
}

// --- constrained mean-variance ----------------------------------------------

auto constrained_min_variance(std::span<const std::vector<double>> cov,
                              std::span<const double> upper, std::span<const double> lower)
    -> Result<std::vector<double>> {
    if (!is_square_finite(cov)) { return make_error<std::vector<double>>(MathError::domain_error); }
    const std::size_t n = cov.size();
    std::vector<LinCon> eqs;
    eqs.push_back({std::vector<double>(n, 1.0), 1.0});  // fully invested
    return solve_box_qp(cov, eqs, upper, lower);
}

auto constrained_efficient_portfolio(std::span<const std::vector<double>> cov,
                                     std::span<const double> mean_returns, double target_return,
                                     std::span<const double> upper, std::span<const double> lower)
    -> Result<FrontierPoint> {
    if (!is_square_finite(cov)) { return make_error<FrontierPoint>(MathError::domain_error); }
    const std::size_t n = cov.size();
    if (mean_returns.size() != n || !std::isfinite(target_return)) {
        return make_error<FrontierPoint>(MathError::domain_error);
    }
    for (double v : mean_returns) {
        if (!std::isfinite(v)) { return make_error<FrontierPoint>(MathError::domain_error); }
    }
    std::vector<LinCon> eqs;
    eqs.push_back({std::vector<double>(n, 1.0), 1.0});
    eqs.push_back({std::vector<double>(mean_returns.begin(), mean_returns.end()), target_return});
    auto w = solve_box_qp(cov, eqs, upper, lower);
    if (!w) { return make_error<FrontierPoint>(w.error()); }
    double ret = 0.0;
    for (std::size_t i = 0; i < n; ++i) { ret += (*w)[i] * mean_returns[i]; }
    return FrontierPoint{std::move(*w), std::sqrt(std::max(quad_form(*w, cov), 0.0)), ret};
}

auto constrained_frontier(std::span<const std::vector<double>> cov,
                          std::span<const double> mean_returns, int points,
                          std::span<const double> upper, std::span<const double> lower)
    -> Result<std::vector<FrontierPoint>> {
    if (!is_square_finite(cov)) {
        return make_error<std::vector<FrontierPoint>>(MathError::domain_error);
    }
    const std::size_t n = cov.size();
    if (points < 2 || mean_returns.size() != n) {
        return make_error<std::vector<FrontierPoint>>(MathError::domain_error);
    }
    // Low end: return of the box-constrained global-min-variance portfolio.
    auto gmv = constrained_min_variance(cov, upper, lower);
    if (!gmv) { return make_error<std::vector<FrontierPoint>>(gmv.error()); }
    double ret_lo = 0.0;
    for (std::size_t i = 0; i < n; ++i) { ret_lo += (*gmv)[i] * mean_returns[i]; }
    // High end: maximum feasible return under the box (LP: maximise mu . w == minimise -mu . w).
    std::vector<double> neg_mu(n);
    for (std::size_t i = 0; i < n; ++i) { neg_mu[i] = -mean_returns[i]; }
    std::vector<std::vector<double>> A_le;
    std::vector<double> b_le;
    std::vector<double> lo(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        if (!lower.empty() && i < lower.size()) { lo[i] = lower[i]; }
        if (i >= upper.size()) { return make_error<std::vector<FrontierPoint>>(MathError::domain_error); }
        std::vector<double> ru(n, 0.0); ru[i] = 1.0;
        A_le.push_back(ru); b_le.push_back(upper[i]);
        if (lo[i] != 0.0) { std::vector<double> rl(n, 0.0); rl[i] = -1.0; A_le.push_back(rl); b_le.push_back(-lo[i]); }
    }
    std::vector<std::vector<double>> A_eq{std::vector<double>(n, 1.0)};
    std::vector<double> b_eq{1.0};
    auto hi_lp = linprog(neg_mu, A_le, b_le, A_eq, b_eq);
    if (!hi_lp) { return make_error<std::vector<FrontierPoint>>(hi_lp.error()); }
    if (hi_lp->status != LinProgStatus::optimal) {
        return make_error<std::vector<FrontierPoint>>(MathError::domain_error);
    }
    const double ret_hi = -hi_lp->objective;
    std::vector<FrontierPoint> frontier;
    frontier.reserve(static_cast<std::size_t>(points));
    for (int i = 0; i < points; ++i) {
        double t = ret_lo;
        if (ret_hi > ret_lo) {
            t = ret_lo + (ret_hi - ret_lo) * static_cast<double>(i) / static_cast<double>(points - 1);
        }
        auto pt = constrained_efficient_portfolio(cov, mean_returns, t, upper, lower);
        if (!pt) { return make_error<std::vector<FrontierPoint>>(pt.error()); }
        frontier.push_back(std::move(*pt));
    }
    return frontier;
}

// --- CVaR (Rockafellar-Uryasev LP) ------------------------------------------

auto cvar_optimal_weights(std::span<const std::vector<double>> scenarios, double beta,
                          std::span<const double> upper) -> Result<CVaRResult> {
    if (scenarios.empty() || scenarios.size() > kMaxScenarios || !(beta >= 0.0 && beta < 1.0)) {
        return make_error<CVaRResult>(MathError::domain_error);
    }
    const std::size_t S = scenarios.size();
    const std::size_t N = scenarios[0].size();
    if (N == 0 || N > kMaxCvarAssets) { return make_error<CVaRResult>(MathError::domain_error); }
    for (const auto& row : scenarios) {
        if (row.size() != N) { return make_error<CVaRResult>(MathError::domain_error); }
        for (double v : row) {
            if (!std::isfinite(v)) { return make_error<CVaRResult>(MathError::domain_error); }
        }
    }
    const bool has_ub = !upper.empty();
    if (has_ub && upper.size() != N) { return make_error<CVaRResult>(MathError::domain_error); }
    // Variable layout: w_0..w_{N-1}, ap, an (alpha = ap - an, free), z_0..z_{S-1}.
    const std::size_t n = N + 2 + S;
    const std::size_t I_ap = N, I_an = N + 1, I_z0 = N + 2;
    const double tail = (1.0 - beta) * static_cast<double>(S);
    if (tail <= 0.0) { return make_error<CVaRResult>(MathError::domain_error); }
    std::vector<double> c(n, 0.0);
    c[I_ap] = 1.0;               // + alpha
    c[I_an] = -1.0;              // - (an) => alpha = ap - an
    for (std::size_t k = 0; k < S; ++k) { c[I_z0 + k] = 1.0 / tail; }
    // Scenario constraints: z_k + sum_i w_i r_{k,i} + ap - an >= 0.
    std::vector<std::vector<double>> A_le;   // (we express >= via A_eq? no — use A_le with negation)
    std::vector<double> b_le;
    for (std::size_t k = 0; k < S; ++k) {
        std::vector<double> row(n, 0.0);
        for (std::size_t i = 0; i < N; ++i) { row[i] = scenarios[k][i]; }
        row[I_ap] = 1.0; row[I_an] = -1.0; row[I_z0 + k] = 1.0;
        // z_k + w.r + ap - an >= 0  <=>  -(...) <= 0.
        for (double& v : row) { v = -v; }
        A_le.push_back(std::move(row));
        b_le.push_back(0.0);
    }
    // Optional box: w_i <= ub_i.
    if (has_ub) {
        for (std::size_t i = 0; i < N; ++i) {
            if (!std::isfinite(upper[i]) || upper[i] < 0.0) {
                return make_error<CVaRResult>(MathError::domain_error);
            }
            std::vector<double> row(n, 0.0); row[i] = 1.0;
            A_le.push_back(std::move(row)); b_le.push_back(upper[i]);
        }
    }
    // Budget equality: sum_i w_i = 1.
    std::vector<std::vector<double>> A_eq;
    std::vector<double> b_eq;
    {
        std::vector<double> row(n, 0.0);
        for (std::size_t i = 0; i < N; ++i) { row[i] = 1.0; }
        A_eq.push_back(std::move(row)); b_eq.push_back(1.0);
    }
    auto lp = linprog(c, A_le, b_le, A_eq, b_eq);
    if (!lp) { return make_error<CVaRResult>(lp.error()); }
    if (lp->status == LinProgStatus::infeasible) {
        return make_error<CVaRResult>(MathError::domain_error);
    }
    if (lp->status == LinProgStatus::unbounded) {
        // The RU program is bounded below; unboundedness signals a malformed instance.
        return make_error<CVaRResult>(MathError::not_converged);
    }
    CVaRResult out;
    out.weights.assign(lp->x.begin(), lp->x.begin() + static_cast<std::ptrdiff_t>(N));
    out.var = lp->x[I_ap] - lp->x[I_an];
    out.cvar = lp->objective;
    if (!std::isfinite(out.cvar) || !std::isfinite(out.var)) {
        return make_error<CVaRResult>(MathError::not_converged);
    }
    return out;
}

// --- French depreciation ----------------------------------------------------

auto amordegrc_coefficient(double asset_life) -> double {
    if (asset_life < 3.0) { return 1.0; }
    if (asset_life < 5.0) { return 1.5; }   // covers 3 and 4 year lives
    if (asset_life <= 6.0) { return 2.0; }  // covers 5 and 6 year lives
    return 2.5;
}

namespace {
[[nodiscard]] auto amor_valid(double cost, double salvage, std::int64_t period, double rate,
                              double f) -> bool {
    return std::isfinite(cost) && std::isfinite(salvage) && std::isfinite(rate) &&
           std::isfinite(f) && cost > 0.0 && rate > 0.0 && salvage >= 0.0 &&
           salvage <= cost && period >= 0 && period <= kMaxPeriods && f > 0.0 && f <= 1.0;
}
}  // namespace

auto amorlinc(double cost, double salvage, std::int64_t period, double rate,
              double first_period_fraction) -> Result<double> {
    if (!amor_valid(cost, salvage, period, rate, first_period_fraction)) {
        return make_error<double>(MathError::domain_error);
    }
    const double one_rate = cost * rate;                 // full-period depreciation
    const double cost_delta = cost - salvage;            // total depreciable base
    const double f0 = first_period_fraction * rate * cost;  // prorated first period
    if (one_rate <= 0.0) { return make_error<double>(MathError::domain_error); }
    const double full = (cost - salvage - f0) / one_rate;
    if (!std::isfinite(full)) {
        // Individually-finite hostile inputs can overflow `one_rate`/`f0` to +inf, making
        // `full = (finite - inf)/inf = NaN`. NaN defeats the clamp below (min/max with NaN
        // returns NaN), so floor(NaN) then static_cast<int64_t>(NaN) would be UB. Refuse here.
        return make_error<double>(MathError::domain_error);
    }
    // Clamp before the float->int64 cast: a tiny rate makes `full` ~1e298, and casting a double
    // that exceeds INT64_MAX is undefined behaviour. 4e18 is far above the kMaxPeriods cap yet
    // safely below 2^63, so the clamp changes no realistic result — it only tames a hostile rate.
    constexpr double kSafeIntCeil = 4.0e18;
    const double full_capped = std::min(std::max(full, 0.0), kSafeIntCeil);
    const std::int64_t n_full = static_cast<std::int64_t>(std::floor(full_capped));
    if (period == 0) { return f0; }
    if (period <= n_full) { return one_rate; }
    if (period == n_full + 1) {
        const double remainder = cost_delta - one_rate * static_cast<double>(n_full) - f0;
        return std::max(remainder, 0.0);
    }
    return 0.0;
}

auto amordegrc(double cost, double salvage, std::int64_t period, double rate,
               double first_period_fraction) -> Result<double> {
    if (!amor_valid(cost, salvage, period, rate, first_period_fraction)) {
        return make_error<double>(MathError::domain_error);
    }
    const double life = 1.0 / rate;
    const double arate = rate * amordegrc_coefficient(life);
    // Whole number of periods over which the asset is written down (used to switch to
    // straight-line completion near the end, the defining French terminal behaviour). Clamp
    // before the float->int64 cast: a tiny rate makes `life` ~1e300, and casting a double past
    // INT64_MAX is UB. 4e18 is far above kMaxPeriods yet below 2^63 — no realistic result moves.
    constexpr double kSafeIntCeil = 4.0e18;
    const std::int64_t total_life =
        std::max<std::int64_t>(1, static_cast<std::int64_t>(std::min(round_half_up(life), kSafeIntCeil)));
    // Period 0: the prorated (rounded) first charge.
    double dep = round_half_up(first_period_fraction * arate * cost);
    dep = std::min(dep, cost - salvage);
    double book = cost - dep;
    if (period == 0) { return std::max(dep, 0.0); }
    for (std::int64_t p = 1; p <= period; ++p) {
        const double remaining_base = book - salvage;
        if (remaining_base <= 0.0) { dep = 0.0; book = salvage; continue; }
        const std::int64_t remaining_life = std::max<std::int64_t>(1, total_life - p);
        const double degressive = arate * book;
        const double straight = remaining_base / static_cast<double>(remaining_life);
        double charge = round_half_up(std::max(degressive, straight));
        charge = std::min(charge, remaining_base);  // never below salvage
        dep = charge;
        book -= charge;
    }
    return std::max(dep, 0.0);
}

// --- continuous compounding & annuity variants ------------------------------

auto effective_continuous(double nominal_rate) -> Result<double> {
    if (!std::isfinite(nominal_rate)) { return make_error<double>(MathError::domain_error); }
    const double e = std::expm1(nominal_rate);
    if (!std::isfinite(e)) { return make_error<double>(MathError::overflow); }
    return e;
}

auto nominal_continuous(double effective_rate) -> Result<double> {
    if (!std::isfinite(effective_rate) || effective_rate <= -1.0) {
        return make_error<double>(MathError::domain_error);
    }
    return std::log1p(effective_rate);
}

auto pay_per(double rate, std::int64_t nper, double pv) -> Result<double> {
    if (!std::isfinite(rate) || !std::isfinite(pv) || nper < 1 || nper > kMaxPeriods ||
        rate <= -1.0) {
        return make_error<double>(MathError::domain_error);
    }
    if (rate == 0.0) { return pv / static_cast<double>(nper); }
    const double disc = std::pow(1.0 + rate, -static_cast<double>(nper));
    const double denom = 1.0 - disc;
    if (denom == 0.0) { return make_error<double>(MathError::division_by_zero); }
    const double p = pv * rate / denom;
    if (!std::isfinite(p)) { return make_error<double>(MathError::overflow); }
    return p;
}

auto pay_odd(double rate, double odd_period_fraction, double pv, bool simple) -> Result<double> {
    if (!std::isfinite(rate) || !std::isfinite(pv) || !std::isfinite(odd_period_fraction) ||
        odd_period_fraction < 0.0 || rate <= -1.0) {
        return make_error<double>(MathError::domain_error);
    }
    const double factor = simple ? rate * odd_period_fraction
                                 : std::pow(1.0 + rate, odd_period_fraction) - 1.0;
    const double interest = pv * factor;
    if (!std::isfinite(interest)) { return make_error<double>(MathError::overflow); }
    return interest;
}

auto pay_uni(double rate, std::int64_t nper, double pv, double odd_period_fraction, bool simple)
    -> Result<double> {
    auto odd = pay_odd(rate, odd_period_fraction, pv, simple);
    if (!odd) { return odd; }
    return pay_per(rate, nper, pv + *odd);
}

auto amortize(double rate, std::int64_t nper, double principal) -> Result<AmortSchedule> {
    if (!std::isfinite(rate) || !std::isfinite(principal) || nper < 1 || nper > kMaxPeriods ||
        principal <= 0.0 || rate <= -1.0) {
        return make_error<AmortSchedule>(MathError::domain_error);
    }
    auto pmt = pay_per(rate, nper, principal);
    if (!pmt) { return make_error<AmortSchedule>(pmt.error()); }
    AmortSchedule sch;
    sch.payment = *pmt;
    sch.interest.reserve(static_cast<std::size_t>(nper));
    sch.principal.reserve(static_cast<std::size_t>(nper));
    sch.balance.reserve(static_cast<std::size_t>(nper));
    double bal = principal;
    for (std::int64_t p = 0; p < nper; ++p) {
        const double interest = bal * rate;
        double princ = *pmt - interest;
        if (p == nper - 1) { princ = bal; }  // absorb residual round-off into the last payment
        bal -= princ;
        sch.interest.push_back(interest);
        sch.principal.push_back(princ);
        sch.balance.push_back(bal);
    }
    return sch;
}

}  // namespace nimblecas::riskextra
