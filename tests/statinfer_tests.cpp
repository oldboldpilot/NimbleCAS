// Tests for nimblecas.statinfer: exact OLS / ridge / weighted regression, R^2, the exact
// rational coefficient covariance, and linear-fractional method of moments.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.statinfer;
import nimblecas.testing;

using nimblecas::MathError;
using nimblecas::Matrix;
using nimblecas::Rational;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

[[nodiscard]] auto ri(std::int64_t v) -> Rational {
    return Rational::from_int(v);
}

// Build a std::vector<Rational> from integer values (each over denominator 1).
[[nodiscard]] auto ints(std::vector<std::int64_t> vs) -> std::vector<Rational> {
    std::vector<Rational> out;
    out.reserve(vs.size());
    for (const std::int64_t v : vs) {
        out.push_back(Rational::from_int(v));
    }
    return out;
}

[[nodiscard]] auto mat(std::vector<std::vector<Rational>> rows) -> Matrix {
    return Matrix::from_rows(std::move(rows)).value();
}

[[nodiscard]] auto sp(const std::vector<Rational>& v) -> std::span<const Rational> {
    return std::span<const Rational>{v};
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.statinfer")
        .test("ols_line_through_collinear_points",
              [](TestContext& t) {
                  // y = 2x + 1 at x = 0,1,2  =>  y = 1,3,5. Design [1, x] with intercept.
                  const auto X = mat({{ri(1), ri(0)}, {ri(1), ri(1)}, {ri(1), ri(2)}});
                  const auto y = ints({1, 3, 5});
                  auto beta = nimblecas::ols(X, sp(y)).value();
                  t.expect(beta.size() == 2, "beta has 2 coefficients");
                  t.expect(beta[0] == ri(1), "intercept = 1");
                  t.expect(beta[1] == ri(2), "slope = 2");
                  // Perfect fit => R^2 = 1.
                  t.expect(nimblecas::r_squared(X, sp(y), sp(beta)).value() == ri(1),
                           "R^2 of exact line = 1");
              })
        .test("ols_nontrivial_rational_fit",
              [](TestContext& t) {
                  // Points (0,0),(1,0),(2,2): least-squares line through Q.
                  // xbar = 1, ybar = 2/3. slope = Sxy/Sxx, Sxy = sum (x-1)(y-2/3) = 2,
                  // Sxx = 2, slope = 1; intercept = 2/3 - 1*1 = -1/3.
                  const auto X = mat({{ri(1), ri(0)}, {ri(1), ri(1)}, {ri(1), ri(2)}});
                  const auto y = ints({0, 0, 2});
                  auto beta = nimblecas::ols(X, sp(y)).value();
                  t.expect(beta[0] == rat(-1, 3), "intercept = -1/3");
                  t.expect(beta[1] == ri(1), "slope = 1");
              })
        .test("ridge_shrinks_slope",
              [](TestContext& t) {
                  // Centered data so the intercept is not penalised meaningfully; use a
                  // single-feature design x = -1,0,1 with y = -1,0,1 (slope 1 for OLS).
                  const auto X = mat({{ri(-1)}, {ri(0)}, {ri(1)}});
                  const auto y = ints({-1, 0, 1});
                  // OLS: X^T X = 2, X^T y = 2 => slope 1.
                  auto ols_beta = nimblecas::ols(X, sp(y)).value();
                  t.expect(ols_beta[0] == ri(1), "unpenalised slope = 1");
                  // Ridge lambda = 2: (2 + 2) b = 2 => b = 1/2, exact.
                  auto ridge_beta = nimblecas::ridge(X, sp(y), ri(2)).value();
                  t.expect(ridge_beta[0] == rat(1, 2), "ridge(lambda=2) slope = 1/2");
                  // lambda = 0 reduces to OLS.
                  auto ridge0 = nimblecas::ridge(X, sp(y), ri(0)).value();
                  t.expect(ridge0[0] == ri(1), "ridge(lambda=0) == OLS");
                  // Negative lambda is a domain error.
                  t.expect(nimblecas::ridge(X, sp(y), ri(-1)).error() == MathError::domain_error,
                           "negative lambda is domain_error");
              })
        .test("weighted_ols_matches_hand_solution",
              [](TestContext& t) {
                  // Single feature x = 1,1,2 with y = 1,3,2 and weights 1,1,2.
                  // X^T W X = 1*1 + 1*1 + 2*(2*2) = 1 + 1 + 8 = 10.
                  // X^T W y = 1*1*1 + 1*1*3 + 2*2*2 = 1 + 3 + 8 = 12.
                  // slope b = 12/10 = 6/5.
                  const auto X = mat({{ri(1)}, {ri(1)}, {ri(2)}});
                  const auto y = ints({1, 3, 2});
                  const auto w = ints({1, 1, 2});
                  auto beta = nimblecas::weighted_ols(X, sp(y), sp(w)).value();
                  t.expect(beta[0] == rat(6, 5), "weighted slope = 6/5");
                  // A zero weight drops an observation: weights 1,1,0 => uses only x=1,1.
                  // X^T W X = 2, X^T W y = 1 + 3 = 4, slope = 2.
                  const auto w0 = ints({1, 1, 0});
                  auto beta0 = nimblecas::weighted_ols(X, sp(y), sp(w0)).value();
                  t.expect(beta0[0] == ri(2), "zero weight drops the third point, slope = 2");
                  // Negative weight is a domain error.
                  const auto wneg = ints({1, 1, -1});
                  t.expect(nimblecas::weighted_ols(X, sp(y), sp(wneg)).error() ==
                               MathError::domain_error,
                           "negative weight is domain_error");
              })
        .test("r_squared_mean_only_model_is_zero",
              [](TestContext& t) {
                  // Intercept-only design (column of ones): beta = ybar, predictions = ybar,
                  // SS_res = SS_tot => R^2 = 0.
                  const auto X = mat({{ri(1)}, {ri(1)}, {ri(1)}});
                  const auto y = ints({1, 2, 6});  // ybar = 3
                  auto beta = nimblecas::ols(X, sp(y)).value();
                  t.expect(beta[0] == ri(3), "mean-only fit = ybar = 3");
                  t.expect(nimblecas::r_squared(X, sp(y), sp(beta)).value() == ri(0),
                           "R^2 of mean-only model = 0");
              })
        .test("r_squared_partial_fit_exact_fraction",
              [](TestContext& t) {
                  // Reuse (0,0),(1,0),(2,2): beta = (-1/3, 1). yhat = -1/3, 2/3, 5/3.
                  // residuals: 1/3, -2/3, 1/3 => SS_res = 1/9+4/9+1/9 = 6/9 = 2/3.
                  // ybar = 2/3; SS_tot = (0-2/3)^2+(0-2/3)^2+(2-2/3)^2 = 4/9+4/9+16/9 = 24/9 = 8/3.
                  // R^2 = 1 - (2/3)/(8/3) = 1 - 1/4 = 3/4.
                  const auto X = mat({{ri(1), ri(0)}, {ri(1), ri(1)}, {ri(1), ri(2)}});
                  const auto y = ints({0, 0, 2});
                  auto beta = nimblecas::ols(X, sp(y)).value();
                  t.expect(nimblecas::r_squared(X, sp(y), sp(beta)).value() == rat(3, 4),
                           "R^2 = 3/4 exactly");
              })
        .test("r_squared_constant_response_is_domain_error",
              [](TestContext& t) {
                  const auto X = mat({{ri(1), ri(0)}, {ri(1), ri(1)}, {ri(1), ri(2)}});
                  const auto y = ints({4, 4, 4});  // SS_tot = 0
                  auto beta = nimblecas::ols(X, sp(y)).value();
                  t.expect(nimblecas::r_squared(X, sp(y), sp(beta)).error() ==
                               MathError::domain_error,
                           "constant response => R^2 undefined => domain_error");
              })
        .test("singular_design_is_domain_error",
              [](TestContext& t) {
                  // Two identical (collinear) feature columns => X^T X singular.
                  const auto X = mat({{ri(1), ri(1)}, {ri(1), ri(1)}, {ri(1), ri(1)}});
                  const auto y = ints({1, 2, 3});
                  t.expect(nimblecas::ols(X, sp(y)).error() == MathError::domain_error,
                           "rank-deficient design => domain_error (never a bogus beta)");
                  // More features than observations is likewise singular.
                  const auto wide = mat({{ri(1), ri(2), ri(3)}});
                  const auto y1 = ints({1});
                  t.expect(nimblecas::ols(wide, sp(y1)).error() == MathError::domain_error,
                           "m < n design => domain_error");
              })
        .test("coefficient_covariance_is_exact_rational",
              [](TestContext& t) {
                  // Points (0,0),(1,0),(2,2): beta = (-1/3, 1). SS_res = 2/3, m=3, n=2,
                  // sigma^2 = SS_res/(m-n) = (2/3)/1 = 2/3.
                  // X = [[1,0],[1,1],[1,2]]. X^T X = [[3,3],[3,5]], det = 6,
                  // (X^T X)^{-1} = [[5/6, -1/2], [-1/2, 1/2]].
                  // Cov = sigma^2 * inv = (2/3)*[[5/6,-1/2],[-1/2,1/2]]
                  //     = [[5/9, -1/3], [-1/3, 1/3]].
                  const auto X = mat({{ri(1), ri(0)}, {ri(1), ri(1)}, {ri(1), ri(2)}});
                  const auto y = ints({0, 0, 2});
                  auto cov = nimblecas::coefficient_covariance(X, sp(y)).value();
                  auto expected = mat({{rat(5, 9), rat(-1, 3)}, {rat(-1, 3), rat(1, 3)}});
                  t.expect(cov.is_equal(expected), "Cov(beta) = [[5/9,-1/3],[-1/3,1/3]] exactly");
                  t.expect(cov.at(0, 1) == cov.at(1, 0), "covariance matrix is symmetric");
                  // m <= n has no residual dof => domain_error.
                  const auto Xsq = mat({{ri(1), ri(0)}, {ri(1), ri(1)}});
                  const auto ysq = ints({1, 2});
                  t.expect(nimblecas::coefficient_covariance(Xsq, sp(ysq)).error() ==
                               MathError::domain_error,
                           "m == n (no residual dof) => domain_error");
              })
        .test("method_of_moments_supported_forms",
              [](TestContext& t) {
                  // Bernoulli / Poisson: E[X] = theta (a=1,b=0,c=0,d=1) => theta = mbar.
                  t.expect(nimblecas::method_of_moments(rat(3, 7), ri(1), ri(0), ri(0), ri(1))
                                   .value() == rat(3, 7),
                           "Bernoulli MoM: theta = mbar");
                  // Uniform(0,theta): E[X] = theta/2 (a=1/2) => theta = 2*mbar.
                  t.expect(nimblecas::method_of_moments(ri(3), rat(1, 2), ri(0), ri(0), ri(1))
                                   .value() == ri(6),
                           "Uniform(0,theta) MoM: theta = 2*mbar");
                  // Binomial(N=5, p): E[X] = 5p (a=5) => p = mbar/5.
                  t.expect(nimblecas::method_of_moments(ri(2), ri(5), ri(0), ri(0), ri(1))
                                   .value() == rat(2, 5),
                           "Binomial(5,p) MoM: p = mbar/5");
                  // Exponential rate: E[X] = 1/lambda (a=0,b=1,c=1,d=0) => lambda = 1/mbar.
                  t.expect(nimblecas::method_of_moments(rat(1, 4), ri(0), ri(1), ri(1), ri(0))
                                   .value() == ri(4),
                           "Exponential MoM: lambda = 1/mbar");
                  // Degenerate: mbar*c - a = 0 (e.g. a=1,c=0 gives -1 != 0, so pick a=0,c=0
                  // => denom 0) => domain_error.
                  t.expect(nimblecas::method_of_moments(ri(2), ri(0), ri(1), ri(0), ri(1))
                                   .error() == MathError::domain_error,
                           "no unique theta => domain_error");
              })
        .run();
}
