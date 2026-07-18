// Tests for nimblecas.portfolio: integrated risk report + ridge-regularized LU mean-variance.
// @author Olumuyiwa Oluwasanmi
//
// Checks the LU-with-partial-pivoting solver against a hand-solved 2x2 system, the ridge
// regularization (a SINGULAR covariance that the plain Cholesky path rejects is solved once
// lambda > 0), the min-variance / tangency / frontier closed forms, and the one-call
// integrated risk report (Sharpe/beta/alpha/drawdown/VaR on a constructed series).

import std;
import nimblecas.core;
import nimblecas.analytics;
import nimblecas.portfolio;
import nimblecas.testing;

using namespace nimblecas::portfolio;
using nimblecas::MathError;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {
[[nodiscard]] auto close(double a, double b, double tol = 1e-9) -> bool { return std::abs(a - b) < tol; }
}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.portfolio")
        .test("LU partial-pivoting solver",
              [](TestContext& t) {
                  // 4x + y = 1, x + 3y = 2  ->  x = 1/11, y = 7/11.
                  const std::vector<std::vector<double>> a{{4.0, 1.0}, {1.0, 3.0}};
                  const std::array<double, 2> b{1.0, 2.0};
                  auto x = lu_solve_ridge(a, b, 0.0);
                  t.expect(x.has_value(), "LU solve succeeds");
                  t.expect(close((*x)[0], 1.0 / 11.0) && close((*x)[1], 7.0 / 11.0),
                           "LU solution == [1/11, 7/11]");
                  // Pivoting: a zero leading pivot must be handled by row interchange.
                  const std::vector<std::vector<double>> p{{0.0, 2.0}, {1.0, 1.0}};
                  const std::array<double, 2> pb{4.0, 3.0};  // 2y=4 -> y=2; x+y=3 -> x=1
                  auto xp = lu_solve_ridge(p, pb, 0.0);
                  t.expect(xp && close((*xp)[0], 1.0) && close((*xp)[1], 2.0),
                           "partial pivoting handles zero leading pivot");
              })
        .test("ridge regularization rescues a singular covariance",
              [](TestContext& t) {
                  const std::vector<std::vector<double>> singular{{0.0, 0.0}, {0.0, 0.0}};
                  // Without regularization the system is singular -> honest error.
                  t.expect(!min_variance_weights(singular, 0.0).has_value(),
                           "singular cov, lambda=0 -> error");
                  // With ridge lambda>0 the matrix becomes lambda*I -> weights [0.5, 0.5].
                  auto w = min_variance_weights(singular, 0.1);
                  t.expect(w.has_value(), "singular cov + ridge solves");
                  t.expect(close((*w)[0], 0.5) && close((*w)[1], 0.5),
                           "ridge min-variance == [0.5, 0.5]");
              })
        .test("mean-variance optima match the closed forms",
              [](TestContext& t) {
                  const std::vector<std::vector<double>> cov{{0.04, 0.0}, {0.0, 0.09}};
                  auto w = min_variance_weights(cov, 0.0).value();
                  t.expect(close(w[0] + w[1], 1.0), "min-var weights sum to 1");
                  t.expect(close(w[0], 25.0 / (25.0 + 100.0 / 9.0), 1e-9), "min-var w0 == 0.6923");
                  const std::vector<std::vector<double>> cov2{{0.04, 0.0}, {0.0, 0.04}};
                  const std::array<double, 2> mu{0.10, 0.10};
                  auto tw = tangency_weights(cov2, mu, 0.02, 0.0).value();
                  t.expect(close(tw[0], 0.5) && close(tw[1], 0.5), "tangency 50/50");
                  const std::array<double, 2> mu2{0.08, 0.12};
                  auto pt = efficient_portfolio(cov2, mu2, 0.10, 0.0).value();
                  t.expect(close(pt.ret, 0.10) && close(pt.weights[0] + pt.weights[1], 1.0),
                           "frontier point hits target and sums to 1");
                  auto ef = efficient_frontier(cov2, mu2, 5, 0.0).value();
                  t.expect(ef.size() == 5, "frontier has 5 points");
              })
        .test("integrated risk report",
              [](TestContext& t) {
                  const std::array<double, 3> ret{-0.02, 0.01, 0.03};
                  const std::array<double, 3> mkt{-0.02, 0.01, 0.03};  // identical -> beta 1, alpha 0
                  auto rep = analyze(ret, mkt, 0.0, 0.95);
                  t.expect(rep.has_value(), "analyze succeeds");
                  t.expect(close(rep->beta, 1.0), "beta(x,x) == 1");
                  t.expect(close(rep->jensen_alpha, 0.0, 1e-12), "alpha(x,x) == 0");
                  // Max drawdown: 1 -> 0.98 is a 2% peak-to-trough decline.
                  t.expect(close(rep->max_drawdown, 0.02, 1e-9), "max drawdown == 2%");
                  // Historical VaR95 on 3 points = worst loss = 0.02.
                  t.expect(close(rep->var_historical, 0.02), "hist VaR == 0.02");
                  t.expect(std::isfinite(rep->sharpe) && std::isfinite(rep->var_parametric),
                           "Sharpe and parametric VaR are finite");
                  t.expect(rep->confidence == 0.95, "confidence recorded");
              })
        .run();
}
