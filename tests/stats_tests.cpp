// Tests for nimblecas.stats: exact mean, variance, covariance, and covariance matrix.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.stats;
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

// Build a std::vector<Rational> from integer values (each over denominator 1).
[[nodiscard]] auto ints(std::vector<std::int64_t> vs) -> std::vector<Rational> {
    std::vector<Rational> out;
    out.reserve(vs.size());
    for (const std::int64_t v : vs) {
        out.push_back(Rational::from_int(v));
    }
    return out;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.stats")
        .test("mean_basic",
              [](TestContext& t) {
                  const auto data = ints({1, 2, 3, 4});
                  t.expect(nimblecas::mean(std::span<const Rational>{data}).value() == rat(5, 2),
                           "mean of {1,2,3,4} = 5/2");
              })
        .test("variance_population_and_sample",
              [](TestContext& t) {
                  const auto data = ints({1, 2, 3});  // mean 2
                  // population: ((1)+(0)+(1))/3 = 2/3
                  t.expect(nimblecas::variance(std::span<const Rational>{data}, false).value() ==
                               rat(2, 3),
                           "population variance of {1,2,3} = 2/3");
                  // sample: 2/2 = 1
                  t.expect(nimblecas::variance(std::span<const Rational>{data}, true).value() ==
                               Rational::from_int(1),
                           "sample variance of {1,2,3} = 1");
              })
        .test("covariance_correlated_and_anticorrelated",
              [](TestContext& t) {
                  const auto x = ints({1, 2, 3});
                  const auto y = ints({3, 2, 1});
                  // cov(x, x, sample) == sample variance of x == 1
                  t.expect(nimblecas::covariance(std::span<const Rational>{x},
                                                 std::span<const Rational>{x}, true)
                                   .value() == Rational::from_int(1),
                           "cov({1,2,3},{1,2,3}) = 1 (= sample variance)");
                  // perfectly anti-correlated -> -1
                  t.expect(nimblecas::covariance(std::span<const Rational>{x},
                                                 std::span<const Rational>{y}, true)
                                   .value() == rat(-1, 1),
                           "cov({1,2,3},{3,2,1}) = -1");
              })
        .test("covariance_matrix_correlated",
              [](TestContext& t) {
                  const auto x = ints({1, 2, 3});
                  const auto y = ints({1, 2, 3});
                  std::vector<std::span<const Rational>> vars{std::span<const Rational>{x},
                                                              std::span<const Rational>{y}};
                  auto sigma = nimblecas::covariance_matrix(vars, true).value();
                  const auto one = Rational::from_int(1);
                  auto expected = Matrix::from_rows({{one, one}, {one, one}}).value();
                  t.expect(sigma.is_equal(expected), "cov matrix of X=Y={1,2,3} = [[1,1],[1,1]]");
                  t.expect(sigma.rows() == 2 && sigma.cols() == 2, "sigma is 2x2");
                  // symmetric
                  t.expect(sigma.at(0, 1) == sigma.at(1, 0), "sigma is symmetric");
                  // diagonal equals the per-variable sample variances
                  t.expect(sigma.at(0, 0) ==
                               nimblecas::variance(std::span<const Rational>{x}, true).value(),
                           "sigma_00 = var(X)");
                  t.expect(sigma.at(1, 1) ==
                               nimblecas::variance(std::span<const Rational>{y}, true).value(),
                           "sigma_11 = var(Y)");
              })
        .test("covariance_matrix_anticorrelated",
              [](TestContext& t) {
                  const auto x = ints({1, 2, 3});
                  const auto y = ints({3, 2, 1});
                  std::vector<std::span<const Rational>> vars{std::span<const Rational>{x},
                                                              std::span<const Rational>{y}};
                  auto sigma = nimblecas::covariance_matrix(vars, true).value();
                  const auto one = Rational::from_int(1);
                  const auto neg = rat(-1, 1);
                  auto expected = Matrix::from_rows({{one, neg}, {neg, one}}).value();
                  t.expect(sigma.is_equal(expected),
                           "cov matrix of X={1,2,3}, Y={3,2,1} = [[1,-1],[-1,1]]");
              })
        .test("error_cases",
              [](TestContext& t) {
                  // empty data -> domain_error
                  const std::vector<Rational> empty;
                  t.expect(nimblecas::mean(std::span<const Rational>{empty}).error() ==
                               MathError::domain_error,
                           "mean of empty data is domain_error");

                  // sample variance of a single point -> domain_error (needs n >= 2)
                  const auto one_point = ints({5});
                  t.expect(nimblecas::variance(std::span<const Rational>{one_point}, true).error() ==
                               MathError::domain_error,
                           "sample variance of a single point is domain_error");
                  // population variance of a single point is well-defined (0)
                  t.expect(nimblecas::variance(std::span<const Rational>{one_point}, false).value() ==
                               Rational{},
                           "population variance of a single point is 0");

                  // ragged variables -> domain_error
                  const auto a = ints({1, 2, 3});
                  const auto b = ints({1, 2});
                  std::vector<std::span<const Rational>> ragged{std::span<const Rational>{a},
                                                                std::span<const Rational>{b}};
                  t.expect(nimblecas::covariance_matrix(ragged, true).error() ==
                               MathError::domain_error,
                           "ragged variables are domain_error");

                  // mismatched covariance lengths -> domain_error
                  t.expect(nimblecas::covariance(std::span<const Rational>{a},
                                                 std::span<const Rational>{b}, true)
                                   .error() == MathError::domain_error,
                           "covariance of unequal-length inputs is domain_error");

                  // empty variable list -> domain_error
                  std::vector<std::span<const Rational>> no_vars;
                  t.expect(nimblecas::covariance_matrix(no_vars, true).error() ==
                               MathError::domain_error,
                           "empty variable list is domain_error");
              })
        .run();
}
