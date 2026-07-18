// NimbleCAS portfolio & risk analytics — performance ratios, risk measures, optimization.
// @author Olumuyiwa Oluwasanmi
//
// SCOPE. Return series construction (simple / log), summary statistics (mean, variance,
// std, covariance/correlation matrices), performance ratios (Sharpe, Sortino, Treynor,
// information ratio, beta/alpha vs a benchmark), drawdown, value-at-risk and conditional
// VaR (expected shortfall) both historical and parametric-Gaussian, and MEAN-VARIANCE
// PORTFOLIO OPTIMIZATION (global minimum-variance and tangency/max-Sharpe portfolios in
// closed form, plus an efficient frontier by the two-fund theorem), backed by an internal
// Cholesky solve of the covariance system. This is the MATLAB portopt/Portfolio +
// financial-ratio surface.
//
// HONESTY (config/cpp_details.txt Rule 32). This layer is NUMERICAL/STATISTICAL: every
// output is a sample estimate over the supplied series, not an exact quantity, and VaR/CVaR
// carry the usual estimation error of their sample. Failure rides the railway — an empty or
// too-short series, a mismatched pair of series, a non-positive-definite covariance (a
// singular or collinear market), or a degenerate optimizer input all return a MathError
// rather than a NaN. Ratios that would divide by a zero standard deviation return
// division_by_zero, never +/-inf dressed up as a number.

module;
#include <cassert>

export module nimblecas.analytics;

import std;
import nimblecas.core;

export namespace nimblecas::analytics {

// Annualisation policy for a ratio: multiply the numerator by `periods_per_year` and the
// denominator by its square root (the standard sqrt-of-time scaling). periods == 1 leaves
// the ratio per-period.
struct Annualisation {
    double periods_per_year{1.0};
};

// --- Return series ----------------------------------------------------------
// Simple period returns r_t = P_t/P_{t-1} - 1 (needs >= 2 prices). Log returns
// ln(P_t/P_{t-1}). Non-positive prices in the log form -> domain_error.
[[nodiscard]] auto simple_returns(std::span<const double> prices) -> Result<std::vector<double>>;
[[nodiscard]] auto log_returns(std::span<const double> prices) -> Result<std::vector<double>>;

// --- Summary statistics -----------------------------------------------------
[[nodiscard]] auto mean(std::span<const double> x) -> Result<double>;
// Sample (n-1) variance/std by default; population when sample == false.
[[nodiscard]] auto variance(std::span<const double> x, bool sample = true) -> Result<double>;
[[nodiscard]] auto stddev(std::span<const double> x, bool sample = true) -> Result<double>;
[[nodiscard]] auto covariance(std::span<const double> x, std::span<const double> y,
                              bool sample = true) -> Result<double>;
[[nodiscard]] auto correlation(std::span<const double> x, std::span<const double> y)
    -> Result<double>;
// Covariance matrix of a set of return series (assets in rows, observations in columns).
[[nodiscard]] auto covariance_matrix(std::span<const std::vector<double>> series, bool sample = true)
    -> Result<std::vector<std::vector<double>>>;

// --- Performance ratios -----------------------------------------------------
// (mean(returns) - risk_free_per_period) / std(returns), optionally annualised.
[[nodiscard]] auto sharpe_ratio(std::span<const double> returns, double risk_free_per_period,
                                Annualisation ann = {}) -> Result<double>;
// Like Sharpe but the denominator is the downside deviation below `mar` (minimum acceptable
// return). Zero downside -> division_by_zero.
[[nodiscard]] auto sortino_ratio(std::span<const double> returns, double mar,
                                 Annualisation ann = {}) -> Result<double>;
// (mean(returns) - risk_free) / beta(returns, market).
[[nodiscard]] auto treynor_ratio(std::span<const double> returns, std::span<const double> market,
                                 double risk_free_per_period) -> Result<double>;
// mean(active) / std(active) where active = returns - benchmark (the tracking-error ratio).
[[nodiscard]] auto information_ratio(std::span<const double> returns,
                                     std::span<const double> benchmark) -> Result<double>;
// CAPM beta = cov(asset, market)/var(market); alpha = mean(asset) - beta*mean(market) (both
// per period; add the risk-free adjustment at the call site if desired).
[[nodiscard]] auto beta(std::span<const double> asset, std::span<const double> market)
    -> Result<double>;
[[nodiscard]] auto alpha(std::span<const double> asset, std::span<const double> market,
                         double risk_free_per_period) -> Result<double>;

// --- Drawdown & tail risk ---------------------------------------------------
// Maximum peak-to-trough fractional decline of an equity curve (a non-negative number;
// 0.2 == a 20% drawdown).
[[nodiscard]] auto max_drawdown(std::span<const double> equity_curve) -> Result<double>;
// Historical VaR at confidence `conf` (e.g. 0.95): the negated `1-conf` quantile of the
// return distribution, reported as a POSITIVE loss fraction. CVaR is the mean loss beyond it.
[[nodiscard]] auto value_at_risk_historical(std::span<const double> returns, double conf)
    -> Result<double>;
[[nodiscard]] auto conditional_var_historical(std::span<const double> returns, double conf)
    -> Result<double>;
// Parametric Gaussian VaR/CVaR from a mean and std at confidence `conf` (positive loss).
[[nodiscard]] auto value_at_risk_gaussian(double mean_return, double std_return, double conf)
    -> Result<double>;
[[nodiscard]] auto conditional_var_gaussian(double mean_return, double std_return, double conf)
    -> Result<double>;

// --- Portfolio construction & mean-variance optimization --------------------
[[nodiscard]] auto portfolio_return(std::span<const double> weights, std::span<const double> mean_returns)
    -> Result<double>;
[[nodiscard]] auto portfolio_variance(std::span<const double> weights,
                                      std::span<const std::vector<double>> cov) -> Result<double>;
// Global minimum-variance weights: Sigma^{-1} 1 / (1' Sigma^{-1} 1). Fully invested
// (weights sum to 1), long/short allowed. Non-PD covariance -> domain_error.
[[nodiscard]] auto min_variance_weights(std::span<const std::vector<double>> cov)
    -> Result<std::vector<double>>;
// Tangency (max-Sharpe) weights: Sigma^{-1}(mu - rf) normalised to sum 1.
[[nodiscard]] auto tangency_weights(std::span<const std::vector<double>> cov,
                                    std::span<const double> mean_returns, double risk_free)
    -> Result<std::vector<double>>;
// One efficient-frontier point: the minimum-variance portfolio achieving `target_return`,
// via the two-fund closed form. Returns (weights, portfolio std, portfolio return).
struct FrontierPoint {
    std::vector<double> weights;
    double risk{0.0};
    double ret{0.0};
};
[[nodiscard]] auto efficient_portfolio(std::span<const std::vector<double>> cov,
                                       std::span<const double> mean_returns, double target_return)
    -> Result<FrontierPoint>;
// A frontier sampled at `points` returns evenly spaced between the min-variance return and
// the max single-asset mean.
[[nodiscard]] auto efficient_frontier(std::span<const std::vector<double>> cov,
                                      std::span<const double> mean_returns, int points)
    -> Result<std::vector<FrontierPoint>>;

}  // namespace nimblecas::analytics

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::analytics {
namespace {

constexpr double kSqrt2 = 1.4142135623730951;

[[nodiscard]] auto norm_pdf(double x) -> double {
    return 0.3989422804014327 * std::exp(-0.5 * x * x);
}
[[nodiscard]] auto norm_cdf(double x) -> double { return 0.5 * std::erfc(-x / kSqrt2); }

// Inverse standard-normal CDF (Acklam + one Halley step) — used by the parametric VaR.
[[nodiscard]] auto inv_norm(double p) -> std::optional<double> {
    if (!(p > 0.0 && p < 1.0)) { return std::nullopt; }
    static constexpr std::array<double, 6> a{-3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01,
        2.506628277459239e+00};
    static constexpr std::array<double, 5> b{-5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01};
    static constexpr std::array<double, 6> c{-7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00,
        2.938163982698783e+00};
    static constexpr std::array<double, 4> d{7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00};
    constexpr double plow = 0.02425;
    double x = 0.0;
    if (p < plow) {
        const double q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= 1.0 - plow) {
        const double q = p - 0.5;
        const double r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
             ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    const double e = norm_cdf(x) - p;
    const double u = e / norm_pdf(x);
    return x - u / (1.0 + 0.5 * x * u);
}

// Cholesky factorisation of a symmetric positive-definite matrix into lower L (Sigma = L L').
// Returns nullopt if a non-positive pivot appears (matrix not PD).
[[nodiscard]] auto cholesky(const std::vector<std::vector<double>>& m)
    -> std::optional<std::vector<std::vector<double>>> {
    const std::size_t n = m.size();
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double s = m[i][j];
            for (std::size_t k = 0; k < j; ++k) { s -= L[i][k] * L[j][k]; }
            if (i == j) {
                if (s <= 0.0) { return std::nullopt; }
                L[i][j] = std::sqrt(s);
            } else {
                L[i][j] = s / L[j][j];
            }
        }
    }
    return L;
}

// Solve Sigma x = b given Sigma's Cholesky factor L (forward then back substitution).
[[nodiscard]] auto chol_solve(const std::vector<std::vector<double>>& L, std::span<const double> b)
    -> std::vector<double> {
    const std::size_t n = L.size();
    std::vector<double> y(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double s = b[i];
        for (std::size_t k = 0; k < i; ++k) { s -= L[i][k] * y[k]; }
        y[i] = s / L[i][i];
    }
    std::vector<double> x(n, 0.0);
    for (std::size_t ii = n; ii-- > 0;) {
        double s = y[ii];
        for (std::size_t k = ii + 1; k < n; ++k) { s -= L[k][ii] * x[k]; }
        x[ii] = s / L[ii][ii];
    }
    return x;
}

[[nodiscard]] auto is_square_nonempty(std::span<const std::vector<double>> m) -> bool {
    if (m.empty()) { return false; }
    for (const auto& row : m) {
        if (row.size() != m.size()) { return false; }
    }
    return true;
}

}  // namespace

// --- Return series ----------------------------------------------------------

auto simple_returns(std::span<const double> prices) -> Result<std::vector<double>> {
    if (prices.size() < 2) { return make_error<std::vector<double>>(MathError::domain_error); }
    std::vector<double> r;
    r.reserve(prices.size() - 1);
    for (std::size_t i = 1; i < prices.size(); ++i) {
        if (prices[i - 1] == 0.0) { return make_error<std::vector<double>>(MathError::division_by_zero); }
        r.push_back(prices[i] / prices[i - 1] - 1.0);
    }
    return r;
}

auto log_returns(std::span<const double> prices) -> Result<std::vector<double>> {
    if (prices.size() < 2) { return make_error<std::vector<double>>(MathError::domain_error); }
    std::vector<double> r;
    r.reserve(prices.size() - 1);
    for (std::size_t i = 1; i < prices.size(); ++i) {
        if (prices[i] <= 0.0 || prices[i - 1] <= 0.0) {
            return make_error<std::vector<double>>(MathError::domain_error);
        }
        r.push_back(std::log(prices[i] / prices[i - 1]));
    }
    return r;
}

// --- Summary statistics -----------------------------------------------------

auto mean(std::span<const double> x) -> Result<double> {
    if (x.empty()) { return make_error<double>(MathError::domain_error); }
    double s = 0.0;
    for (double v : x) { s += v; }
    return s / static_cast<double>(x.size());
}

auto variance(std::span<const double> x, bool sample) -> Result<double> {
    const std::size_t n = x.size();
    if (n < (sample ? 2u : 1u)) { return make_error<double>(MathError::domain_error); }
    auto m = mean(x);
    if (!m) { return m; }
    double s = 0.0;
    for (double v : x) { const double d = v - *m; s += d * d; }
    return s / static_cast<double>(sample ? n - 1 : n);
}

auto stddev(std::span<const double> x, bool sample) -> Result<double> {
    return variance(x, sample).transform([](double v) { return std::sqrt(v); });
}

auto covariance(std::span<const double> x, std::span<const double> y, bool sample) -> Result<double> {
    const std::size_t n = x.size();
    if (n != y.size() || n < (sample ? 2u : 1u)) {
        return make_error<double>(MathError::domain_error);
    }
    auto mx = mean(x);
    auto my = mean(y);
    if (!mx || !my) { return make_error<double>(MathError::domain_error); }
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) { s += (x[i] - *mx) * (y[i] - *my); }
    return s / static_cast<double>(sample ? n - 1 : n);
}

auto correlation(std::span<const double> x, std::span<const double> y) -> Result<double> {
    auto c = covariance(x, y);
    auto sx = stddev(x);
    auto sy = stddev(y);
    if (!c || !sx || !sy) { return make_error<double>(MathError::domain_error); }
    if (*sx == 0.0 || *sy == 0.0) { return make_error<double>(MathError::division_by_zero); }
    return *c / (*sx * *sy);
}

auto covariance_matrix(std::span<const std::vector<double>> series, bool sample)
    -> Result<std::vector<std::vector<double>>> {
    if (series.empty()) { return make_error<std::vector<std::vector<double>>>(MathError::domain_error); }
    const std::size_t k = series.size();
    const std::size_t n = series[0].size();
    for (const auto& s : series) {
        if (s.size() != n) { return make_error<std::vector<std::vector<double>>>(MathError::domain_error); }
    }
    std::vector<std::vector<double>> cov(k, std::vector<double>(k, 0.0));
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = i; j < k; ++j) {
            auto c = covariance(series[i], series[j], sample);
            if (!c) { return make_error<std::vector<std::vector<double>>>(c.error()); }
            cov[i][j] = *c;
            cov[j][i] = *c;
        }
    }
    return cov;
}

// --- Performance ratios -----------------------------------------------------

auto sharpe_ratio(std::span<const double> returns, double risk_free_per_period, Annualisation ann)
    -> Result<double> {
    auto m = mean(returns);
    auto sd = stddev(returns);
    if (!m || !sd) { return make_error<double>(MathError::domain_error); }
    if (*sd == 0.0) { return make_error<double>(MathError::division_by_zero); }
    const double per_period = (*m - risk_free_per_period) / *sd;
    return per_period * std::sqrt(ann.periods_per_year);
}

auto sortino_ratio(std::span<const double> returns, double mar, Annualisation ann) -> Result<double> {
    auto m = mean(returns);
    if (!m || returns.empty()) { return make_error<double>(MathError::domain_error); }
    double downside = 0.0;
    for (double r : returns) { const double d = std::min(r - mar, 0.0); downside += d * d; }
    downside = std::sqrt(downside / static_cast<double>(returns.size()));
    if (downside == 0.0) { return make_error<double>(MathError::division_by_zero); }
    return (*m - mar) / downside * std::sqrt(ann.periods_per_year);
}

auto treynor_ratio(std::span<const double> returns, std::span<const double> market,
                   double risk_free_per_period) -> Result<double> {
    auto m = mean(returns);
    auto b = beta(returns, market);
    if (!m || !b) { return make_error<double>(MathError::domain_error); }
    if (*b == 0.0) { return make_error<double>(MathError::division_by_zero); }
    return (*m - risk_free_per_period) / *b;
}

auto information_ratio(std::span<const double> returns, std::span<const double> benchmark)
    -> Result<double> {
    if (returns.size() != benchmark.size() || returns.size() < 2) {
        return make_error<double>(MathError::domain_error);
    }
    std::vector<double> active(returns.size());
    for (std::size_t i = 0; i < returns.size(); ++i) { active[i] = returns[i] - benchmark[i]; }
    auto m = mean(active);
    auto sd = stddev(active);
    if (!m || !sd) { return make_error<double>(MathError::domain_error); }
    if (*sd == 0.0) { return make_error<double>(MathError::division_by_zero); }
    return *m / *sd;
}

auto beta(std::span<const double> asset, std::span<const double> market) -> Result<double> {
    auto cov = covariance(asset, market);
    auto vm = variance(market);
    if (!cov || !vm) { return make_error<double>(MathError::domain_error); }
    if (*vm == 0.0) { return make_error<double>(MathError::division_by_zero); }
    return *cov / *vm;
}

auto alpha(std::span<const double> asset, std::span<const double> market, double risk_free_per_period)
    -> Result<double> {
    auto b = beta(asset, market);
    auto ma = mean(asset);
    auto mm = mean(market);
    if (!b || !ma || !mm) { return make_error<double>(MathError::domain_error); }
    // Jensen's alpha: mean_asset - [rf + beta*(mean_market - rf)].
    return *ma - (risk_free_per_period + *b * (*mm - risk_free_per_period));
}

// --- Drawdown & tail risk ---------------------------------------------------

auto max_drawdown(std::span<const double> equity_curve) -> Result<double> {
    if (equity_curve.empty()) { return make_error<double>(MathError::domain_error); }
    double peak = equity_curve[0];
    double worst = 0.0;
    for (double v : equity_curve) {
        peak = std::max(peak, v);
        if (peak > 0.0) { worst = std::max(worst, (peak - v) / peak); }
    }
    return worst;
}

auto value_at_risk_historical(std::span<const double> returns, double conf) -> Result<double> {
    if (returns.empty() || !(conf > 0.0 && conf < 1.0)) {
        return make_error<double>(MathError::domain_error);
    }
    std::vector<double> sorted(returns.begin(), returns.end());
    std::ranges::sort(sorted);
    // The ceil((1-conf)*n)-th smallest return is the VaR order statistic (consistent with the
    // CVaR tail below); report the loss as a positive number.
    const std::size_t n = sorted.size();
    const std::size_t cutoff =
        std::min(std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil((1.0 - conf) * static_cast<double>(n)))), n);
    return -sorted[cutoff - 1];
}

auto conditional_var_historical(std::span<const double> returns, double conf) -> Result<double> {
    if (returns.empty() || !(conf > 0.0 && conf < 1.0)) {
        return make_error<double>(MathError::domain_error);
    }
    std::vector<double> sorted(returns.begin(), returns.end());
    std::ranges::sort(sorted);
    // Mean loss over the tail AT OR BEYOND VaR — the same ceil((1-conf)*n) cutoff as the VaR
    // order statistic, so CVaR >= VaR by construction (they share the boundary observation).
    const std::size_t n = sorted.size();
    const std::size_t cutoff =
        std::min(std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil((1.0 - conf) * static_cast<double>(n)))), n);
    double s = 0.0;
    for (std::size_t i = 0; i < cutoff; ++i) { s += sorted[i]; }
    return -s / static_cast<double>(cutoff);  // mean loss in the tail, positive
}

auto value_at_risk_gaussian(double mean_return, double std_return, double conf) -> Result<double> {
    if (std_return < 0.0 || !(conf > 0.0 && conf < 1.0)) {
        return make_error<double>(MathError::domain_error);
    }
    auto z = inv_norm(1.0 - conf);
    if (!z) { return make_error<double>(MathError::domain_error); }
    return -(mean_return + std_return * *z);  // positive loss at the (1-conf) quantile
}

auto conditional_var_gaussian(double mean_return, double std_return, double conf) -> Result<double> {
    if (std_return < 0.0 || !(conf > 0.0 && conf < 1.0)) {
        return make_error<double>(MathError::domain_error);
    }
    const double alpha_tail = 1.0 - conf;
    auto z = inv_norm(alpha_tail);
    if (!z) { return make_error<double>(MathError::domain_error); }
    // ES = -mean + std * phi(z)/alpha (closed form for the Gaussian expected shortfall).
    return -mean_return + std_return * norm_pdf(*z) / alpha_tail;
}

// --- Portfolio optimization -------------------------------------------------

auto portfolio_return(std::span<const double> weights, std::span<const double> mean_returns)
    -> Result<double> {
    if (weights.size() != mean_returns.size() || weights.empty()) {
        return make_error<double>(MathError::domain_error);
    }
    double s = 0.0;
    for (std::size_t i = 0; i < weights.size(); ++i) { s += weights[i] * mean_returns[i]; }
    return s;
}

auto portfolio_variance(std::span<const double> weights, std::span<const std::vector<double>> cov)
    -> Result<double> {
    if (!is_square_nonempty(cov) || weights.size() != cov.size()) {
        return make_error<double>(MathError::domain_error);
    }
    double s = 0.0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        for (std::size_t j = 0; j < weights.size(); ++j) { s += weights[i] * cov[i][j] * weights[j]; }
    }
    return s;
}

auto min_variance_weights(std::span<const std::vector<double>> cov) -> Result<std::vector<double>> {
    if (!is_square_nonempty(cov)) { return make_error<std::vector<double>>(MathError::domain_error); }
    const std::vector<std::vector<double>> m(cov.begin(), cov.end());
    auto L = cholesky(m);
    if (!L) { return make_error<std::vector<double>>(MathError::domain_error); }  // not PD
    const std::size_t n = cov.size();
    const std::vector<double> ones(n, 1.0);
    const std::vector<double> z = chol_solve(*L, ones);  // Sigma^{-1} 1
    double denom = 0.0;
    for (double v : z) { denom += v; }                   // 1' Sigma^{-1} 1
    if (denom == 0.0) { return make_error<std::vector<double>>(MathError::division_by_zero); }
    std::vector<double> w(n);
    for (std::size_t i = 0; i < n; ++i) { w[i] = z[i] / denom; }
    return w;
}

auto tangency_weights(std::span<const std::vector<double>> cov, std::span<const double> mean_returns,
                      double risk_free) -> Result<std::vector<double>> {
    if (!is_square_nonempty(cov) || mean_returns.size() != cov.size()) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    const std::vector<std::vector<double>> m(cov.begin(), cov.end());
    auto L = cholesky(m);
    if (!L) { return make_error<std::vector<double>>(MathError::domain_error); }
    const std::size_t n = cov.size();
    std::vector<double> excess(n);
    for (std::size_t i = 0; i < n; ++i) { excess[i] = mean_returns[i] - risk_free; }
    const std::vector<double> z = chol_solve(*L, excess);  // Sigma^{-1}(mu - rf)
    double denom = 0.0;
    for (double v : z) { denom += v; }
    if (denom == 0.0) { return make_error<std::vector<double>>(MathError::division_by_zero); }
    std::vector<double> w(n);
    for (std::size_t i = 0; i < n; ++i) { w[i] = z[i] / denom; }
    return w;
}

auto efficient_portfolio(std::span<const std::vector<double>> cov, std::span<const double> mean_returns,
                         double target_return) -> Result<FrontierPoint> {
    if (!is_square_nonempty(cov) || mean_returns.size() != cov.size()) {
        return make_error<FrontierPoint>(MathError::domain_error);
    }
    const std::vector<std::vector<double>> m(cov.begin(), cov.end());
    auto L = cholesky(m);
    if (!L) { return make_error<FrontierPoint>(MathError::domain_error); }
    const std::size_t n = cov.size();
    const std::vector<double> ones(n, 1.0);
    const std::vector<double> mu(mean_returns.begin(), mean_returns.end());
    const std::vector<double> si_one = chol_solve(*L, ones);  // Sigma^{-1} 1
    const std::vector<double> si_mu = chol_solve(*L, mu);      // Sigma^{-1} mu
    // Standard frontier scalars A=1'Si^-1 mu, B=mu'Si^-1 mu, C=1'Si^-1 1, D=BC-A^2.
    double A = 0.0;
    double B = 0.0;
    double C = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        A += ones[i] * si_mu[i];
        B += mu[i] * si_mu[i];
        C += ones[i] * si_one[i];
    }
    const double D = B * C - A * A;
    if (std::abs(D) < 1e-18) { return make_error<FrontierPoint>(MathError::domain_error); }
    // w = Sigma^{-1}(lambda*mu + gamma*1) with the multipliers solving the return/budget pair.
    const double lambda = (C * target_return - A) / D;
    const double gamma = (B - A * target_return) / D;
    std::vector<double> w(n);
    for (std::size_t i = 0; i < n; ++i) { w[i] = lambda * si_mu[i] + gamma * si_one[i]; }
    auto var = portfolio_variance(w, cov);
    if (!var) { return make_error<FrontierPoint>(var.error()); }
    return FrontierPoint{std::move(w), std::sqrt(std::max(*var, 0.0)), target_return};
}

auto efficient_frontier(std::span<const std::vector<double>> cov, std::span<const double> mean_returns,
                        int points) -> Result<std::vector<FrontierPoint>> {
    if (points < 2 || mean_returns.empty()) {
        return make_error<std::vector<FrontierPoint>>(MathError::domain_error);
    }
    auto mv = min_variance_weights(cov);
    if (!mv) { return make_error<std::vector<FrontierPoint>>(mv.error()); }
    auto lo = portfolio_return(*mv, mean_returns);
    if (!lo) { return make_error<std::vector<FrontierPoint>>(lo.error()); }
    const double hi = *std::ranges::max_element(mean_returns);
    std::vector<FrontierPoint> frontier;
    frontier.reserve(static_cast<std::size_t>(points));
    for (int i = 0; i < points; ++i) {
        const double t = *lo + (hi - *lo) * static_cast<double>(i) / static_cast<double>(points - 1);
        auto pt = efficient_portfolio(cov, mean_returns, t);
        if (!pt) { return make_error<std::vector<FrontierPoint>>(pt.error()); }
        frontier.push_back(std::move(*pt));
    }
    return frontier;
}

}  // namespace nimblecas::analytics
