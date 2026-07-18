// NimbleCAS integrated portfolio analytics — one-call risk report + robust mean-variance.
// @author Olumuyiwa Oluwasanmi
//
// SCOPE. Two things the lower-level `nimblecas.analytics` module leaves to the caller:
//   1. An INTEGRATED risk report — Sharpe, Sortino, Treynor, Jensen's alpha, beta, max
//      drawdown, historical VaR/CVaR and parametric (Gaussian) VaR/CVaR — computed in a
//      single `analyze()` call over a return series and its benchmark, so a book is scored
//      in one shot rather than by stitching a dozen primitive calls.
//   2. A ROBUST mean-variance optimizer (Markowitz efficient frontier: global minimum
//      variance, tangency/max-Sharpe, and any target-return frontier point) that stays
//      well-posed on ill-conditioned or singular sample covariance via DIAGONAL RIDGE
//      REGULARIZATION (Sigma + lambda*I) and an LU-with-partial-pivoting solver — where the
//      Cholesky path in `nimblecas.analytics` refuses a non-positive-definite matrix, this
//      one regularizes it and solves, which is what production estimation from short return
//      histories actually needs.
//
// This layer REUSES `nimblecas.analytics` for the ratio/VaR primitives (no re-derivation);
// it adds the aggregation and the regularized solver. It is NUMERICAL/STATISTICAL: every
// figure is a sample estimate, failure rides the railway (Result<T>/MathError), and the
// regularizer is honest — `ridge_lambda` is an explicit argument, never a hidden fudge, and
// the report records the lambda actually used.

module;
#include <cassert>

export module nimblecas.portfolio;

import std;
import nimblecas.core;
import nimblecas.analytics;

export namespace nimblecas::portfolio {

// The integrated risk report. Ratios are per-period unless an Annualisation is supplied;
// VaR/CVaR are positive loss fractions at the requested confidence.
struct RiskReport {
    double sharpe{0.0};
    double sortino{0.0};
    double treynor{0.0};
    double jensen_alpha{0.0};
    double beta{0.0};
    double max_drawdown{0.0};
    double var_historical{0.0};
    double cvar_historical{0.0};
    double var_parametric{0.0};
    double cvar_parametric{0.0};
    double confidence{0.95};
};

// Compute the full report from a return series, its benchmark/market series (same length),
// a per-period risk-free rate, a tail confidence, and an annualisation for the ratios. The
// equity curve for the drawdown is reconstructed from the returns (compounded from 1.0).
// Mismatched/short series -> domain_error; a zero-volatility or zero-beta denominator on an
// individual ratio surfaces as that ratio's error (the whole report then fails honestly).
[[nodiscard]] auto analyze(std::span<const double> returns, std::span<const double> market,
                           double risk_free_per_period = 0.0, double confidence = 0.95,
                           analytics::Annualisation annualisation = {}) -> Result<RiskReport>;

// ---------------------------------------------------------------------------
// Robust mean-variance optimization (ridge-regularized, LU-solved).
// ---------------------------------------------------------------------------
// Solve (Sigma + lambda*I) x = b by LU decomposition with partial pivoting. Exposed because
// it is the reusable numerical core of every optimizer below. Returns nullopt only if the
// regularized matrix is still numerically singular (lambda == 0 on a singular Sigma).
[[nodiscard]] auto lu_solve_ridge(std::span<const std::vector<double>> matrix,
                                  std::span<const double> rhs, double ridge_lambda)
    -> std::optional<std::vector<double>>;

// Global minimum-variance weights of the ridge-regularized covariance:
//   w = (Sigma+lambda*I)^{-1} 1 / (1' (Sigma+lambda*I)^{-1} 1), fully invested (sum 1).
[[nodiscard]] auto min_variance_weights(std::span<const std::vector<double>> cov,
                                        double ridge_lambda = 0.0) -> Result<std::vector<double>>;
// Tangency (max-Sharpe) weights: (Sigma+lambda*I)^{-1}(mu - rf) normalised to sum 1.
[[nodiscard]] auto tangency_weights(std::span<const std::vector<double>> cov,
                                    std::span<const double> mean_returns, double risk_free,
                                    double ridge_lambda = 0.0) -> Result<std::vector<double>>;

struct FrontierPoint {
    std::vector<double> weights;
    double risk{0.0};
    double ret{0.0};
};
// Minimum-variance portfolio achieving `target_return` on the ridge-regularized covariance,
// via the two-fund closed form (A/B/C/D scalars from LU solves against 1 and mu).
[[nodiscard]] auto efficient_portfolio(std::span<const std::vector<double>> cov,
                                       std::span<const double> mean_returns, double target_return,
                                       double ridge_lambda = 0.0) -> Result<FrontierPoint>;
// A frontier sampled at `points` target returns between the global-min-variance return and
// the largest single-asset mean.
[[nodiscard]] auto efficient_frontier(std::span<const std::vector<double>> cov,
                                      std::span<const double> mean_returns, int points,
                                      double ridge_lambda = 0.0) -> Result<std::vector<FrontierPoint>>;

}  // namespace nimblecas::portfolio

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::portfolio {
namespace {

[[nodiscard]] auto is_square_nonempty(std::span<const std::vector<double>> m) -> bool {
    if (m.empty()) { return false; }
    for (const auto& row : m) {
        if (row.size() != m.size()) { return false; }
    }
    return true;
}

// Dot product / vector sum helpers.
[[nodiscard]] auto vsum(std::span<const double> v) -> double {
    double s = 0.0;
    for (double x : v) { s += x; }
    return s;
}
[[nodiscard]] auto vdot(std::span<const double> a, std::span<const double> b) -> double {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) { s += a[i] * b[i]; }
    return s;
}

// The frontier's A/B/C/D scalars from two ridge LU solves (Sigma^{-1}1 and Sigma^{-1}mu).
struct FrontierScalars {
    std::vector<double> si_one;  // (Sigma+lambda I)^{-1} 1
    std::vector<double> si_mu;   // (Sigma+lambda I)^{-1} mu
    double A{0.0}, B{0.0}, C{0.0}, D{0.0};
};

[[nodiscard]] auto frontier_scalars(std::span<const std::vector<double>> cov,
                                    std::span<const double> mu, double lambda)
    -> std::optional<FrontierScalars> {
    const std::size_t n = cov.size();
    const std::vector<double> ones(n, 1.0);
    auto si_one = lu_solve_ridge(cov, ones, lambda);
    auto si_mu = lu_solve_ridge(cov, mu, lambda);
    if (!si_one || !si_mu) { return std::nullopt; }
    FrontierScalars fs;
    fs.si_one = std::move(*si_one);
    fs.si_mu = std::move(*si_mu);
    fs.A = vdot(ones, fs.si_mu);   // 1' Si^-1 mu
    fs.B = vdot(mu, fs.si_mu);     // mu' Si^-1 mu
    fs.C = vdot(ones, fs.si_one);  // 1' Si^-1 1
    fs.D = fs.B * fs.C - fs.A * fs.A;
    return fs;
}

[[nodiscard]] auto portfolio_variance(std::span<const double> w,
                                      std::span<const std::vector<double>> cov) -> double {
    double s = 0.0;
    for (std::size_t i = 0; i < w.size(); ++i) {
        for (std::size_t j = 0; j < w.size(); ++j) { s += w[i] * cov[i][j] * w[j]; }
    }
    return s;
}

}  // namespace

auto lu_solve_ridge(std::span<const std::vector<double>> matrix, std::span<const double> rhs,
                    double ridge_lambda) -> std::optional<std::vector<double>> {
    const std::size_t n = matrix.size();
    if (n == 0 || rhs.size() != n) { return std::nullopt; }
    // Build the regularized working copy A = Sigma + lambda*I and the rhs.
    std::vector<std::vector<double>> a(n, std::vector<double>(n));
    for (std::size_t i = 0; i < n; ++i) {
        if (matrix[i].size() != n) { return std::nullopt; }
        for (std::size_t j = 0; j < n; ++j) {
            a[i][j] = matrix[i][j] + (i == j ? ridge_lambda : 0.0);
        }
    }
    std::vector<double> b(rhs.begin(), rhs.end());
    // Gaussian elimination with PARTIAL PIVOTING (row interchange on the largest pivot).
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t piv = k;
        double best = std::abs(a[k][k]);
        for (std::size_t i = k + 1; i < n; ++i) {
            if (std::abs(a[i][k]) > best) { best = std::abs(a[i][k]); piv = i; }
        }
        if (best < 1e-300) { return std::nullopt; }  // singular even after regularization
        if (piv != k) { std::swap(a[k], a[piv]); std::swap(b[k], b[piv]); }
        for (std::size_t i = k + 1; i < n; ++i) {
            const double f = a[i][k] / a[k][k];
            if (f == 0.0) { continue; }
            for (std::size_t j = k; j < n; ++j) { a[i][j] -= f * a[k][j]; }
            b[i] -= f * b[k];
        }
    }
    // Back substitution.
    std::vector<double> x(n, 0.0);
    for (std::size_t ii = n; ii-- > 0;) {
        double s = b[ii];
        for (std::size_t j = ii + 1; j < n; ++j) { s -= a[ii][j] * x[j]; }
        x[ii] = s / a[ii][ii];
    }
    return x;
}

auto analyze(std::span<const double> returns, std::span<const double> market,
             double risk_free_per_period, double confidence,
             analytics::Annualisation annualisation) -> Result<RiskReport> {
    if (returns.size() != market.size() || returns.size() < 2 ||
        !(confidence > 0.0 && confidence < 1.0)) {
        return make_error<RiskReport>(MathError::domain_error);
    }
    RiskReport r{};
    r.confidence = confidence;

    auto sharpe = analytics::sharpe_ratio(returns, risk_free_per_period, annualisation);
    auto sortino = analytics::sortino_ratio(returns, risk_free_per_period, annualisation);
    auto treynor = analytics::treynor_ratio(returns, market, risk_free_per_period);
    auto alpha = analytics::alpha(returns, market, risk_free_per_period);
    auto beta = analytics::beta(returns, market);
    auto var_h = analytics::value_at_risk_historical(returns, confidence);
    auto cvar_h = analytics::conditional_var_historical(returns, confidence);
    auto mu = analytics::mean(returns);
    auto sd = analytics::stddev(returns);
    if (!sharpe || !sortino || !treynor || !alpha || !beta || !var_h || !cvar_h || !mu || !sd) {
        // Propagate the first failure honestly rather than filling a partial report.
        for (const auto& e : {sharpe, sortino, treynor, alpha, beta, var_h, cvar_h, mu, sd}) {
            if (!e) { return make_error<RiskReport>(e.error()); }
        }
    }
    auto var_p = analytics::value_at_risk_gaussian(*mu, *sd, confidence);
    auto cvar_p = analytics::conditional_var_gaussian(*mu, *sd, confidence);
    // Reconstruct a compounded equity curve (starting at 1.0) from the returns for drawdown.
    std::vector<double> equity;
    equity.reserve(returns.size() + 1);
    double level = 1.0;
    equity.push_back(level);
    for (double rr : returns) { level *= (1.0 + rr); equity.push_back(level); }
    auto mdd = analytics::max_drawdown(equity);
    if (!var_p || !cvar_p || !mdd) {
        for (const auto& e : {var_p, cvar_p, mdd}) {
            if (!e) { return make_error<RiskReport>(e.error()); }
        }
    }
    r.sharpe = *sharpe; r.sortino = *sortino; r.treynor = *treynor;
    r.jensen_alpha = *alpha; r.beta = *beta;
    r.max_drawdown = *mdd;
    r.var_historical = *var_h; r.cvar_historical = *cvar_h;
    r.var_parametric = *var_p; r.cvar_parametric = *cvar_p;
    return r;
}

auto min_variance_weights(std::span<const std::vector<double>> cov, double ridge_lambda)
    -> Result<std::vector<double>> {
    if (!is_square_nonempty(cov)) { return make_error<std::vector<double>>(MathError::domain_error); }
    const std::size_t n = cov.size();
    const std::vector<double> ones(n, 1.0);
    auto z = lu_solve_ridge(cov, ones, ridge_lambda);
    if (!z) { return make_error<std::vector<double>>(MathError::domain_error); }
    const double denom = vsum(*z);
    if (denom == 0.0) { return make_error<std::vector<double>>(MathError::division_by_zero); }
    for (double& w : *z) { w /= denom; }
    return *z;
}

auto tangency_weights(std::span<const std::vector<double>> cov, std::span<const double> mean_returns,
                      double risk_free, double ridge_lambda) -> Result<std::vector<double>> {
    if (!is_square_nonempty(cov) || mean_returns.size() != cov.size()) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    const std::size_t n = cov.size();
    std::vector<double> excess(n);
    for (std::size_t i = 0; i < n; ++i) { excess[i] = mean_returns[i] - risk_free; }
    auto z = lu_solve_ridge(cov, excess, ridge_lambda);
    if (!z) { return make_error<std::vector<double>>(MathError::domain_error); }
    const double denom = vsum(*z);
    if (denom == 0.0) { return make_error<std::vector<double>>(MathError::division_by_zero); }
    for (double& w : *z) { w /= denom; }
    return *z;
}

auto efficient_portfolio(std::span<const std::vector<double>> cov, std::span<const double> mean_returns,
                         double target_return, double ridge_lambda) -> Result<FrontierPoint> {
    if (!is_square_nonempty(cov) || mean_returns.size() != cov.size()) {
        return make_error<FrontierPoint>(MathError::domain_error);
    }
    const std::vector<double> mu(mean_returns.begin(), mean_returns.end());
    auto fs = frontier_scalars(cov, mu, ridge_lambda);
    if (!fs) { return make_error<FrontierPoint>(MathError::domain_error); }
    if (std::abs(fs->D) < 1e-18) { return make_error<FrontierPoint>(MathError::domain_error); }
    const double lam = (fs->C * target_return - fs->A) / fs->D;
    const double gam = (fs->B - fs->A * target_return) / fs->D;
    const std::size_t n = cov.size();
    std::vector<double> w(n);
    for (std::size_t i = 0; i < n; ++i) { w[i] = lam * fs->si_mu[i] + gam * fs->si_one[i]; }
    const double var = std::max(portfolio_variance(w, cov), 0.0);
    return FrontierPoint{std::move(w), std::sqrt(var), target_return};
}

auto efficient_frontier(std::span<const std::vector<double>> cov, std::span<const double> mean_returns,
                        int points, double ridge_lambda) -> Result<std::vector<FrontierPoint>> {
    if (points < 2 || mean_returns.empty()) {
        return make_error<std::vector<FrontierPoint>>(MathError::domain_error);
    }
    auto gmv = min_variance_weights(cov, ridge_lambda);
    if (!gmv) { return make_error<std::vector<FrontierPoint>>(gmv.error()); }
    double lo = 0.0;
    for (std::size_t i = 0; i < gmv->size(); ++i) { lo += (*gmv)[i] * mean_returns[i]; }
    const double hi = *std::ranges::max_element(mean_returns);
    std::vector<FrontierPoint> frontier;
    frontier.reserve(static_cast<std::size_t>(points));
    for (int i = 0; i < points; ++i) {
        const double t = lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(points - 1);
        auto pt = efficient_portfolio(cov, mean_returns, t, ridge_lambda);
        if (!pt) { return make_error<std::vector<FrontierPoint>>(pt.error()); }
        frontier.push_back(std::move(*pt));
    }
    return frontier;
}

}  // namespace nimblecas::portfolio
