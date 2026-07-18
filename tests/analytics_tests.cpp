// Tests for nimblecas.analytics: portfolio & risk analytics.
// @author Olumuyiwa Oluwasanmi
//
// Numerical/statistical layer checked against hand-computed statistics and the closed-form
// optima: returns, mean/variance/covariance/correlation on tiny exact series; Sharpe/beta/
// alpha/drawdown on constructed inputs; historical and Gaussian VaR/CVaR at known quantiles;
// and the mean-variance optima (min-variance weights proportional to inverse variances,
// tangency weights, fully-invested frontier points) via the internal Cholesky solve.

import std;
import nimblecas.core;
import nimblecas.analytics;
import nimblecas.testing;

using namespace nimblecas::analytics;
using nimblecas::MathError;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {
[[nodiscard]] auto close(double a, double b, double tol = 1e-9) -> bool { return std::abs(a - b) < tol; }
}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.analytics")
        .test("returns and summary statistics",
              [](TestContext& t) {
                  const std::array<double, 3> px{100.0, 110.0, 121.0};
                  auto r = simple_returns(px).value();
                  t.expect(r.size() == 2 && close(r[0], 0.10) && close(r[1], 0.10),
                           "simple returns of 100,110,121 == [0.1,0.1]");
                  const std::array<double, 3> x{1.0, 2.0, 3.0};
                  t.expect(close(mean(x).value(), 2.0), "mean == 2");
                  t.expect(close(variance(x).value(), 1.0), "sample variance == 1");
                  t.expect(close(stddev(x).value(), 1.0), "sample std == 1");
                  const std::array<double, 3> y{3.0, 2.0, 1.0};
                  t.expect(close(covariance(x, y).value(), -1.0), "cov(asc,desc) == -1");
                  t.expect(close(correlation(x, y).value(), -1.0), "corr(asc,desc) == -1");
                  t.expect(close(correlation(x, x).value(), 1.0), "corr(x,x) == 1");
              })
        .test("performance ratios",
              [](TestContext& t) {
                  const std::array<double, 3> ret{0.01, 0.02, 0.03};
                  // mean 0.02, sample std 0.01 -> Sharpe (rf 0) == 2.
                  t.expect(close(sharpe_ratio(ret, 0.0).value(), 2.0), "Sharpe == 2");
                  // Annualising by 4 scales by sqrt(4) == 2 -> 4.
                  t.expect(close(sharpe_ratio(ret, 0.0, Annualisation{4.0}).value(), 4.0),
                           "annualised Sharpe == 4");
                  const std::array<double, 3> mkt{0.01, 0.02, 0.03};
                  t.expect(close(beta(ret, mkt).value(), 1.0), "beta(x,x) == 1");
                  t.expect(close(alpha(ret, mkt, 0.005).value(), 0.0, 1e-12),
                           "alpha(x,x) == 0");
              })
        .test("drawdown and value-at-risk",
              [](TestContext& t) {
                  const std::array<double, 4> eq{100.0, 120.0, 90.0, 110.0};
                  t.expect(close(max_drawdown(eq).value(), 0.25), "max drawdown == 25%");
                  const std::array<double, 5> ret{-0.05, -0.02, 0.0, 0.01, 0.03};
                  t.expect(close(value_at_risk_historical(ret, 0.95).value(), 0.05),
                           "hist VaR95 == 0.05 (worst return)");
                  t.expect(close(conditional_var_historical(ret, 0.95).value(), 0.05),
                           "hist CVaR95 == 0.05");
                  // Gaussian VaR at 95% of a standard normal is the 1.6449 quantile.
                  t.expect(close(value_at_risk_gaussian(0.0, 1.0, 0.95).value(), 1.6448536, 1e-5),
                           "Gaussian VaR95 == 1.6449");
              })
        .test("mean-variance optimization closed forms",
              [](TestContext& t) {
                  // Diagonal covariance: min-variance weights are proportional to 1/variance.
                  const std::vector<std::vector<double>> cov{{0.04, 0.0}, {0.0, 0.09}};
                  auto w = min_variance_weights(cov).value();
                  const double sum = w[0] + w[1];
                  t.expect(close(sum, 1.0), "min-var weights sum to 1");
                  // 1/0.04 : 1/0.09 == 25 : 11.111 -> 0.6923, 0.3077.
                  t.expect(close(w[0], 25.0 / (25.0 + 100.0 / 9.0), 1e-9), "min-var w0 == 0.6923");
                  // Equal variances + equal means -> tangency splits 50/50.
                  const std::vector<std::vector<double>> cov2{{0.04, 0.0}, {0.0, 0.04}};
                  const std::array<double, 2> mu{0.10, 0.10};
                  auto tw = tangency_weights(cov2, mu, 0.02).value();
                  t.expect(close(tw[0], 0.5) && close(tw[1], 0.5), "tangency 50/50");
                  // portfolio variance of a 50/50 diagonal book == 0.02.
                  const std::array<double, 2> half{0.5, 0.5};
                  t.expect(close(portfolio_variance(half, cov2).value(), 0.02),
                           "portfolio variance == 0.02");
                  // A frontier point achieves its target return and sums to 1.
                  auto pt = efficient_portfolio(cov2, mu, 0.10).value();
                  t.expect(close(pt.ret, 0.10), "frontier point return == target");
                  t.expect(close(pt.weights[0] + pt.weights[1], 1.0), "frontier weights sum to 1");
                  // Non-PD covariance is refused, not silently solved.
                  const std::vector<std::vector<double>> bad{{0.0, 0.0}, {0.0, 0.0}};
                  t.expect(!min_variance_weights(bad).has_value(), "singular cov -> error");
              })
        .test("efficient frontier is monotone in return",
              [](TestContext& t) {
                  const std::vector<std::vector<double>> cov{{0.04, 0.01}, {0.01, 0.09}};
                  const std::array<double, 2> mu{0.08, 0.12};
                  auto ef = efficient_frontier(cov, mu, 5).value();
                  t.expect(ef.size() == 5, "frontier has 5 points");
                  bool monotone = true;
                  for (std::size_t i = 1; i < ef.size(); ++i) {
                      if (ef[i].ret < ef[i - 1].ret - 1e-12) { monotone = false; }
                  }
                  t.expect(monotone, "frontier returns non-decreasing");
              })
        .run();
}
