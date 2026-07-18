// Tests for nimblecas.riskextra: extended risk/portfolio numerics, French depreciation,
// and continuous-compounding / odd-period annuity variants.
// @author Olumuyiwa Oluwasanmi
//
// Every numeric expectation is an ORACLE re-derived by hand or by an independent identity:
//   * corr2cov and cov2corr are mutual inverses (round-trip identity).
//   * LPM_2 below the mean equals the semivariance, == variance/2 on a symmetric sample.
//   * exponentially-weighted moments reduce to the population moment at lambda == 1 and to a
//     hand-summed weighted average at lambda == 0.5.
//   * simulated correlated draws recover the input correlation at large n (loose tolerance)
//     and are bit-reproducible under a fixed seed.
//   * the two-phase simplex reproduces small LP optima computed by hand; the active-set QP
//     returns the boxed weights on a tight 2-asset box and the analytic min-variance weights
//     when the box is slack.
//   * the CVaR (Rockafellar-Uryasev) program matches a hand solution on a 1-asset and a
//     symmetric 2-asset instance.
//   * AMORLINC's schedule sums to cost - salvage; AMORDEGRC's first charge is
//     coefficient * straight-line-rate * cost.
//   * effective/nominal continuous compounding are inverses; the amortisation schedule closes
//     to a zero balance with interest + principal == payment each period.

import std;
import nimblecas.core;
import nimblecas.riskextra;
import nimblecas.testing;

using namespace nimblecas::riskextra;
using nimblecas::MathError;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto close(double a, double b, double tol = 1e-9) -> bool {
    return std::abs(a - b) < tol;
}

// Independent Pearson correlation over two equal-length columns (population moments), used to
// check the recovered correlation of simulated draws without depending on nimblecas.analytics.
[[nodiscard]] auto pearson(const std::vector<double>& x, const std::vector<double>& y) -> double {
    const double n = static_cast<double>(x.size());
    double mx = 0.0;
    double my = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) { mx += x[i]; my += y[i]; }
    mx /= n; my /= n;
    double sxy = 0.0;
    double sxx = 0.0;
    double syy = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        sxy += (x[i] - mx) * (y[i] - my);
        sxx += (x[i] - mx) * (x[i] - mx);
        syy += (y[i] - my) * (y[i] - my);
    }
    return sxy / std::sqrt(sxx * syy);
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.riskextra")
        .test("corr2cov and cov2corr round-trip",
              [](TestContext& t) {
                  const std::vector<std::vector<double>> corr{{1.0, 0.5}, {0.5, 1.0}};
                  const std::array<double, 2> sd{2.0, 3.0};
                  auto cov = corr2cov(corr, sd).value();
                  // Sigma = [[4, 3], [3, 9]].
                  t.expect(close(cov[0][0], 4.0) && close(cov[1][1], 9.0) && close(cov[0][1], 3.0),
                           "corr2cov -> [[4,3],[3,9]]");
                  auto back = cov2corr(cov).value();
                  t.expect(close(back[0][0], 1.0) && close(back[1][1], 1.0) && close(back[0][1], 0.5),
                           "cov2corr recovers the correlation (round-trip identity)");
                  // cov2corr then corr2cov is also the identity.
                  auto cov2 = corr2cov(back, sd).value();
                  t.expect(close(cov2[0][1], 3.0) && close(cov2[0][0], 4.0),
                           "corr2cov o cov2corr == identity");
                  // A zero-variance diagonal makes the correlation undefined.
                  const std::vector<std::vector<double>> bad{{0.0, 0.0}, {0.0, 1.0}};
                  t.expect(!cov2corr(bad).has_value(), "cov2corr rejects zero variance");
              })
        .test("lower partial moment and downside deviation",
              [](TestContext& t) {
                  // Symmetric sample about mean 0: {-2,-1,1,2}. LPM_2(0) = (4+1)/4 = 1.25.
                  const std::array<double, 4> r{-2.0, -1.0, 1.0, 2.0};
                  t.expect(close(lower_partial_moment(r, 2.0, 0.0).value(), 1.25),
                           "LPM_2 below mean == 1.25");
                  // Population variance = (4+1+1+4)/4 = 2.5; semivariance is exactly half.
                  t.expect(close(lower_partial_moment(r, 2.0, 0.0).value(), 2.5 / 2.0),
                           "LPM_2 below mean == variance/2 (symmetric)");
                  // LPM_1(0) = (2+1)/4 = 0.75; downside deviation = sqrt(1.25).
                  t.expect(close(lower_partial_moment(r, 1.0, 0.0).value(), 0.75), "LPM_1 == 0.75");
                  t.expect(close(downside_deviation(r, 0.0).value(), std::sqrt(1.25)),
                           "downside deviation == sqrt(LPM_2)");
                  // LPM_0 counts the strictly-below fraction: 2 of 4 == 0.5.
                  t.expect(close(lower_partial_moment(r, 0.0, 0.0).value(), 0.5), "LPM_0 == 0.5");
                  t.expect(!lower_partial_moment({}, 2.0, 0.0).has_value(), "empty -> error");
              })
        .test("exponentially-weighted moments",
              [](TestContext& t) {
                  const std::array<double, 4> x{1.0, 2.0, 3.0, 4.0};
                  // lambda == 1: equal weights -> ordinary mean 2.5.
                  t.expect(close(ew_mean(x, 1.0).value(), 2.5), "ew_mean(lambda=1) == mean");
                  // lambda == 0.5: weights (from oldest) 0.125,0.25,0.5,1 -> 6.125 / 1.875.
                  t.expect(close(ew_mean(x, 0.5).value(), 6.125 / 1.875),
                           "ew_mean(lambda=0.5) == hand-summed weighted average");
                  // lambda == 1 covariance is the ordinary (n-1) SAMPLE covariance.
                  const std::array<double, 3> a{1.0, 2.0, 3.0};
                  t.expect(close(ew_cov(a, a, 1.0).value(), 1.0),
                           "ew_cov(lambda=1) == sample variance 1.0");
                  const std::vector<std::vector<double>> series{{1.0, 2.0, 3.0}, {3.0, 2.0, 1.0}};
                  auto m = ew_covariance_matrix(series, 1.0).value();
                  // Sample cov of ascending vs descending 1,2,3 is -1.0; sample var is 1.0.
                  t.expect(close(m[0][1], -1.0) && close(m[0][0], 1.0),
                           "ew covariance matrix (lambda=1) == sample cov");
                  t.expect(!ew_mean(x, 0.0).has_value() && !ew_mean(x, 1.5).has_value(),
                           "lambda outside (0,1] -> error");
              })
        .test("simulate correlated returns recovers correlation and is reproducible",
              [](TestContext& t) {
                  const std::array<double, 2> mu{0.0, 0.0};
                  const std::vector<std::vector<double>> cov{{1.0, 0.5}, {0.5, 1.0}};
                  auto draws = simulate_correlated_returns(mu, cov, 40000, 12345).value();
                  t.expect(draws.size() == 40000 && draws[0].size() == 2, "shape n x d");
                  std::vector<double> c0(draws.size());
                  std::vector<double> c1(draws.size());
                  for (std::size_t i = 0; i < draws.size(); ++i) { c0[i] = draws[i][0]; c1[i] = draws[i][1]; }
                  t.expect(close(pearson(c0, c1), 0.5, 0.03),
                           "recovered correlation ~= 0.5 at large n");
                  // Reproducibility: same seed -> identical first sample.
                  auto again = simulate_correlated_returns(mu, cov, 40000, 12345).value();
                  t.expect(draws[0][0] == again[0][0] && draws[7][1] == again[7][1],
                           "fixed seed is bit-reproducible");
                  // Non-positive-definite covariance is refused, not factored into garbage.
                  const std::vector<std::vector<double>> bad{{1.0, 2.0}, {2.0, 1.0}};
                  t.expect(!simulate_correlated_returns(mu, bad, 10, 1).has_value(),
                           "non-PD covariance -> error");
              })
        .test("linprog reproduces small LP optima",
              [](TestContext& t) {
                  // min -x0 - 2 x1  s.t. x0 + x1 <= 4, x0 <= 2, x >= 0. Optimum (0,4), obj -8.
                  const std::array<double, 2> c{-1.0, -2.0};
                  const std::vector<std::vector<double>> A_le{{1.0, 1.0}, {1.0, 0.0}};
                  const std::array<double, 2> b_le{4.0, 2.0};
                  auto r = linprog(c, A_le, b_le, {}, {}).value();
                  t.expect(r.status == LinProgStatus::optimal, "bounded LP is optimal");
                  t.expect(close(r.objective, -8.0) && close(r.x[0], 0.0) && close(r.x[1], 4.0),
                           "LP optimum (0,4), obj -8");
                  // Equality: min x0 + x1 s.t. x0 + x1 == 3 -> obj 3.
                  const std::array<double, 2> c2{1.0, 1.0};
                  const std::vector<std::vector<double>> A_eq{{1.0, 1.0}};
                  const std::array<double, 1> b_eq{3.0};
                  auto r2 = linprog(c2, {}, {}, A_eq, b_eq).value();
                  t.expect(r2.status == LinProgStatus::optimal && close(r2.objective, 3.0),
                           "equality-constrained optimum == 3");
                  // Infeasible: x0 + x1 == 3 and x0 + x1 <= 1.
                  const std::vector<std::vector<double>> A_le3{{1.0, 1.0}};
                  const std::array<double, 1> b_le3{1.0};
                  auto r3 = linprog(c2, A_le3, b_le3, A_eq, b_eq).value();
                  t.expect(r3.status == LinProgStatus::infeasible, "conflicting constraints -> infeasible");
                  // Unbounded: min -x0 with no constraints.
                  const std::array<double, 1> c4{-1.0};
                  auto r4 = linprog(c4, {}, {}, {}, {}).value();
                  t.expect(r4.status == LinProgStatus::unbounded, "descent with no bound -> unbounded");
              })
        .test("box-constrained mean-variance QP",
              [](TestContext& t) {
                  const std::vector<std::vector<double>> cov{{0.01, 0.0}, {0.0, 0.04}};
                  // Slack box [0,1]^2: analytic min-variance weights 1/var proportions (0.8,0.2).
                  const std::array<double, 2> ub_slack{1.0, 1.0};
                  auto w0 = constrained_min_variance(cov, ub_slack).value();
                  t.expect(close(w0[0], 0.8, 1e-6) && close(w0[1], 0.2, 1e-6),
                           "slack box -> analytic (0.8,0.2)");
                  // Tight box on asset 0 (<= 0.3) forces the boxed weights (0.3,0.7).
                  const std::array<double, 2> ub_tight{0.3, 1.0};
                  auto w1 = constrained_min_variance(cov, ub_tight).value();
                  t.expect(close(w1[0], 0.3, 1e-6) && close(w1[1], 0.7, 1e-6),
                           "tight box -> boxed weights (0.3,0.7)");
                  t.expect(close(w1[0] + w1[1], 1.0, 1e-9), "boxed weights fully invested");
                  // Efficient portfolio at a return target: 2 equalities pin (0.5,0.5).
                  const std::array<double, 2> mu{0.05, 0.15};
                  auto pt = constrained_efficient_portfolio(cov, mu, 0.10, ub_slack).value();
                  t.expect(close(pt.weights[0], 0.5, 1e-6) && close(pt.ret, 0.10, 1e-9),
                           "efficient portfolio hits target return with (0.5,0.5)");
                  // Frontier is non-decreasing in return.
                  auto ef = constrained_frontier(cov, mu, 5, ub_slack).value();
                  bool monotone = ef.size() == 5;
                  for (std::size_t i = 1; i < ef.size(); ++i) {
                      if (ef[i].ret < ef[i - 1].ret - 1e-9) { monotone = false; }
                  }
                  t.expect(monotone, "constrained frontier non-decreasing in return");
                  // Infeasible box (uppers cannot reach the fully-invested budget).
                  const std::array<double, 2> ub_bad{0.2, 0.2};
                  t.expect(!constrained_min_variance(cov, ub_bad).has_value(),
                           "sum(upper) < 1 -> infeasible");
              })
        .test("CVaR-optimal weights (Rockafellar-Uryasev)",
              [](TestContext& t) {
                  // Single asset, 2 scenarios, beta 0.5 (worst-1-of-2 tail). Losses {-0.10, 0.20};
                  // CVaR_0.5 == 0.20, weight forced to 1 by the budget.
                  const std::vector<std::vector<double>> s1{{0.10}, {-0.20}};
                  auto r1 = cvar_optimal_weights(s1, 0.5).value();
                  t.expect(close(r1.weights[0], 1.0, 1e-6), "single asset weight == 1");
                  t.expect(close(r1.cvar, 0.20, 1e-6), "single-asset CVaR_0.5 == 0.20");
                  // Symmetric 2-asset instance: the equal-weight book zeroes both losses, so the
                  // minimum CVaR is 0 at (0.5,0.5) (the unique minimiser).
                  const std::vector<std::vector<double>> s2{{0.10, -0.10}, {-0.10, 0.10}};
                  auto r2 = cvar_optimal_weights(s2, 0.5).value();
                  t.expect(close(r2.weights[0], 0.5, 1e-6) && close(r2.weights[1], 0.5, 1e-6),
                           "symmetric CVaR optimum == (0.5,0.5)");
                  t.expect(close(r2.cvar, 0.0, 1e-6), "symmetric min CVaR == 0");
                  t.expect(close(r2.weights[0] + r2.weights[1], 1.0, 1e-6), "CVaR weights sum to 1");
                  // Validation.
                  t.expect(!cvar_optimal_weights(s2, 1.0).has_value(), "beta >= 1 -> error");
                  t.expect(!cvar_optimal_weights({}, 0.5).has_value(), "empty scenarios -> error");
              })
        .test("French depreciation: AMORLINC and AMORDEGRC",
              [](TestContext& t) {
                  // AMORLINC: cost 1000, salvage 100, rate 0.25, full first period.
                  // one_rate 250, base 900, f0 250 -> n_full = floor(650/250) = 2.
                  t.expect(close(amorlinc(1000.0, 100.0, 0, 0.25, 1.0).value(), 250.0),
                           "AMORLINC period 0 == 250");
                  t.expect(close(amorlinc(1000.0, 100.0, 2, 0.25, 1.0).value(), 250.0),
                           "AMORLINC full period == 250");
                  t.expect(close(amorlinc(1000.0, 100.0, 3, 0.25, 1.0).value(), 150.0),
                           "AMORLINC final period == remainder 150");
                  t.expect(close(amorlinc(1000.0, 100.0, 4, 0.25, 1.0).value(), 0.0),
                           "AMORLINC beyond life == 0");
                  double sum = 0.0;
                  for (std::int64_t p = 0; p <= 4; ++p) {
                      sum += amorlinc(1000.0, 100.0, p, 0.25, 1.0).value();
                  }
                  t.expect(close(sum, 900.0), "AMORLINC schedule sums to cost - salvage");
                  // AMORDEGRC coefficient table.
                  t.expect(close(amordegrc_coefficient(4.0), 1.5) &&
                               close(amordegrc_coefficient(5.0), 2.0) &&
                               close(amordegrc_coefficient(10.0), 2.5),
                           "AMORDEGRC coefficient table (1.5 / 2.0 / 2.5)");
                  // Year-1 charge == coefficient * straight-line-rate * cost.
                  // life 4 -> coeff 1.5, accelerated rate 0.375: 0.375 * 1000 == 375.
                  t.expect(close(amordegrc(1000.0, 0.0, 0, 0.25, 1.0).value(), 375.0),
                           "AMORDEGRC period 0 == coeff*rate*cost == 375");
                  // life 5 -> coeff 2.0: 0.2*2.0*1000 == 400.
                  t.expect(close(amordegrc(1000.0, 0.0, 0, 0.2, 1.0).value(), 400.0),
                           "AMORDEGRC (life 5) period 0 == 400");
                  // Prorated first period (half a year): round(0.5 * 375) == 188 (half-up).
                  t.expect(close(amordegrc(1000.0, 0.0, 0, 0.25, 0.5).value(), 188.0),
                           "AMORDEGRC prorated first period == 188");
              })
        .test("continuous compounding and annuity variants",
              [](TestContext& t) {
                  // e^0.1 - 1 and its inverse.
                  t.expect(close(effective_continuous(0.1).value(), std::exp(0.1) - 1.0),
                           "effective_continuous(0.1) == e^0.1 - 1");
                  t.expect(close(nominal_continuous(effective_continuous(0.1).value()).value(), 0.1),
                           "nominal_continuous o effective_continuous == identity");
                  t.expect(!nominal_continuous(-1.0).has_value(), "effective <= -1 -> error");
                  // Level payment: 1000 at 10% over 3 periods -> 402.11480...
                  const double pmt = pay_per(0.1, 3, 1000.0).value();
                  t.expect(close(pmt, 1000.0 * 0.1 / (1.0 - std::pow(1.1, -3.0)), 1e-9),
                           "pay_per matches the annuity formula");
                  // Amortisation closes to a zero balance; interest+principal == payment.
                  auto sch = amortize(0.1, 3, 1000.0).value();
                  t.expect(close(sch.interest[0], 100.0) && close(sch.principal[0], pmt - 100.0, 1e-9),
                           "first period interest 100, principal payment-100");
                  t.expect(close(sch.balance.back(), 0.0, 1e-9), "amortisation closes to zero balance");
                  double principal_sum = 0.0;
                  for (double p : sch.principal) { principal_sum += p; }
                  t.expect(close(principal_sum, 1000.0, 1e-9), "principal repayments sum to the loan");
                  // Odd-period interest: full period, simple -> pv*rate.
                  t.expect(close(pay_odd(0.1, 1.0, 1000.0).value(), 100.0), "pay_odd full period == pv*rate");
                  // Uniform payment with no odd period reduces to pay_per.
                  t.expect(close(pay_uni(0.1, 3, 1000.0, 0.0).value(), pmt, 1e-9),
                           "pay_uni(odd=0) == pay_per");
                  // Capitalised odd-period interest raises the level payment.
                  t.expect(pay_uni(0.1, 3, 1000.0, 0.5).value() > pmt,
                           "capitalised odd interest raises the payment");
              })
        .test("depreciation with a hostile tiny rate does not invoke float->int64 UB",
              [](TestContext& t) {
                  // A near-zero rate makes the internal period count ~1e298; the pre-cast clamp
                  // must keep the float->int64 conversion well-defined (regression for the audit).
                  const auto al = amorlinc(1000.0, 100.0, 5, 1e-300, 1.0);
                  t.expect(!al.has_value() || std::isfinite(al.value()),
                           "amorlinc(tiny rate) -> finite or error, no UB");
                  const auto ad = amordegrc(1000.0, 100.0, 5, 1e-300, 1.0);
                  t.expect(!ad.has_value() || std::isfinite(ad.value()),
                           "amordegrc(tiny rate) -> finite or error, no UB");
              })
        .run();
}
