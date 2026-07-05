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
        .test("weighted_mean_basic",
              [](TestContext& t) {
                  const auto data = ints({1, 2, 3});
                  const auto w = ints({1, 2, 3});
                  // (1*1 + 2*2 + 3*3) / (1+2+3) = 14/6 = 7/3
                  t.expect(nimblecas::weighted_mean(std::span<const Rational>{data},
                                                    std::span<const Rational>{w})
                                   .value() == rat(7, 3),
                           "weighted_mean {1,2,3} by {1,2,3} = 7/3");
                  // uniform weights reduce to the arithmetic mean
                  const auto uni = ints({1, 1, 1});
                  t.expect(nimblecas::weighted_mean(std::span<const Rational>{data},
                                                    std::span<const Rational>{uni})
                                   .value() ==
                               nimblecas::mean(std::span<const Rational>{data}).value(),
                           "uniform weighted_mean = mean");
                  // length mismatch -> domain_error
                  const auto short_w = ints({1, 2});
                  t.expect(nimblecas::weighted_mean(std::span<const Rational>{data},
                                                    std::span<const Rational>{short_w})
                                   .error() == MathError::domain_error,
                           "weighted_mean length mismatch is domain_error");
                  // zero total weight -> domain_error
                  const auto zsum = std::vector<Rational>{rat(1, 1), rat(-1, 1)};
                  const auto zdata = ints({5, 7});
                  t.expect(nimblecas::weighted_mean(std::span<const Rational>{zdata},
                                                    std::span<const Rational>{zsum})
                                   .error() == MathError::domain_error,
                           "weighted_mean zero total weight is domain_error");
              })
        .test("moments",
              [](TestContext& t) {
                  const auto data = ints({1, 2, 3, 4, 5});  // mean 3
                  // raw moment k=2: (1+4+9+16+25)/5 = 55/5 = 11
                  t.expect(nimblecas::raw_moment(std::span<const Rational>{data}, 2).value() ==
                               Rational::from_int(11),
                           "raw_moment k=2 of {1..5} = 11");
                  // k=0 -> 1
                  t.expect(nimblecas::raw_moment(std::span<const Rational>{data}, 0).value() ==
                               Rational::from_int(1),
                           "raw_moment k=0 = 1");
                  // central moment k=2 = population variance = 2
                  t.expect(nimblecas::central_moment(std::span<const Rational>{data}, 2).value() ==
                               Rational::from_int(2),
                           "central_moment k=2 = population variance = 2");
                  t.expect(nimblecas::central_moment(std::span<const Rational>{data}, 2).value() ==
                               nimblecas::variance(std::span<const Rational>{data}, false).value(),
                           "central_moment k=2 == population variance");
                  // central moment k=1 = 0
                  t.expect(nimblecas::central_moment(std::span<const Rational>{data}, 1).value() ==
                               Rational{},
                           "central_moment k=1 = 0");
                  // empty -> domain_error
                  const std::vector<Rational> empty;
                  t.expect(nimblecas::raw_moment(std::span<const Rational>{empty}, 2).error() ==
                               MathError::domain_error,
                           "raw_moment of empty is domain_error");
              })
        .test("excess_kurtosis_and_skewness_squared",
              [](TestContext& t) {
                  // {1,2,3,4,5}: m2 = 2, m4 = 34/5, excess kurtosis = (34/5)/4 - 3 = -13/10
                  const auto data = ints({1, 2, 3, 4, 5});
                  t.expect(nimblecas::excess_kurtosis(std::span<const Rational>{data}).value() ==
                               rat(-13, 10),
                           "excess_kurtosis {1..5} = -13/10");
                  // symmetric data -> skewness_squared = 0 (m3 = 0)
                  t.expect(nimblecas::skewness_squared(std::span<const Rational>{data}).value() ==
                               Rational{},
                           "skewness_squared of symmetric {1..5} = 0");
                  // {0,0,0,0,4}: skewness = 3/2, so skewness_squared = 9/4; m3 > 0
                  const auto skew = ints({0, 0, 0, 0, 4});
                  t.expect(nimblecas::skewness_squared(std::span<const Rational>{skew}).value() ==
                               rat(9, 4),
                           "skewness_squared {0,0,0,0,4} = 9/4");
                  // sign of skewness recoverable from central_moment(_,3) > 0
                  auto m3 = nimblecas::central_moment(std::span<const Rational>{skew}, 3).value();
                  t.expect(m3.numerator() > 0, "central_moment k=3 > 0 => positive skew");
                  // constant data -> zero variance -> domain_error
                  const auto constant = ints({4, 4, 4});
                  t.expect(nimblecas::excess_kurtosis(std::span<const Rational>{constant}).error() ==
                               MathError::domain_error,
                           "excess_kurtosis of constant data is domain_error");
                  t.expect(
                      nimblecas::skewness_squared(std::span<const Rational>{constant}).error() ==
                          MathError::domain_error,
                      "skewness_squared of constant data is domain_error");
              })
        .test("median",
              [](TestContext& t) {
                  // even n -> average of the two middle: {1,2,3,4} -> 5/2
                  const auto even = ints({1, 2, 3, 4});
                  t.expect(nimblecas::median(std::span<const Rational>{even}).value() == rat(5, 2),
                           "median {1,2,3,4} = 5/2");
                  // odd n -> middle: {1,2,3} -> 2
                  const auto odd = ints({1, 2, 3});
                  t.expect(nimblecas::median(std::span<const Rational>{odd}).value() ==
                               Rational::from_int(2),
                           "median {1,2,3} = 2");
                  // unsorted input is sorted internally: {3,1,2} -> 2
                  const auto unsorted = ints({3, 1, 2});
                  t.expect(nimblecas::median(std::span<const Rational>{unsorted}).value() ==
                               Rational::from_int(2),
                           "median {3,1,2} = 2 (sorted internally)");
                  const std::vector<Rational> empty;
                  t.expect(nimblecas::median(std::span<const Rational>{empty}).error() ==
                               MathError::domain_error,
                           "median of empty is domain_error");
              })
        .test("quantile_range_iqr",
              [](TestContext& t) {
                  const auto data = ints({1, 2, 3, 4});
                  // type-7 median (p=1/2): 5/2
                  t.expect(nimblecas::quantile(std::span<const Rational>{data}, rat(1, 2)).value() ==
                               rat(5, 2),
                           "quantile p=1/2 of {1,2,3,4} = 5/2");
                  // Q1 (p=1/4): h=3/4 -> 1 + 3/4*(2-1) = 7/4
                  t.expect(nimblecas::quantile(std::span<const Rational>{data}, rat(1, 4)).value() ==
                               rat(7, 4),
                           "quantile p=1/4 = 7/4");
                  // Q3 (p=3/4): h=9/4 -> 3 + 1/4*(4-3) = 13/4
                  t.expect(nimblecas::quantile(std::span<const Rational>{data}, rat(3, 4)).value() ==
                               rat(13, 4),
                           "quantile p=3/4 = 13/4");
                  // p=0 -> min, p=1 -> max
                  t.expect(nimblecas::quantile(std::span<const Rational>{data}, Rational{}).value() ==
                               Rational::from_int(1),
                           "quantile p=0 = min = 1");
                  t.expect(nimblecas::quantile(std::span<const Rational>{data},
                                               Rational::from_int(1))
                                   .value() == Rational::from_int(4),
                           "quantile p=1 = max = 4");
                  // range = max - min = 3
                  t.expect(nimblecas::range(std::span<const Rational>{data}).value() ==
                               Rational::from_int(3),
                           "range {1,2,3,4} = 3");
                  // iqr = Q3 - Q1 = 13/4 - 7/4 = 3/2
                  t.expect(nimblecas::iqr(std::span<const Rational>{data}).value() == rat(3, 2),
                           "iqr {1,2,3,4} = 3/2");
                  // p out of [0,1] -> domain_error
                  t.expect(nimblecas::quantile(std::span<const Rational>{data}, rat(3, 2)).error() ==
                               MathError::domain_error,
                           "quantile p>1 is domain_error");
                  t.expect(
                      nimblecas::quantile(std::span<const Rational>{data}, rat(-1, 2)).error() ==
                          MathError::domain_error,
                      "quantile p<0 is domain_error");
              })
        .test("mode",
              [](TestContext& t) {
                  // unique mode
                  const auto data = ints({1, 2, 2, 3, 3, 3});
                  t.expect(nimblecas::mode(std::span<const Rational>{data}).value() ==
                               Rational::from_int(3),
                           "mode {1,2,2,3,3,3} = 3");
                  auto ms = nimblecas::modes(std::span<const Rational>{data}).value();
                  t.expect(ms.size() == 1 && ms[0] == Rational::from_int(3),
                           "modes {1,2,2,3,3,3} = {3}");
                  // tie -> mode is the smallest value; modes returns both ascending
                  const auto tie = ints({2, 2, 1, 1});
                  t.expect(nimblecas::mode(std::span<const Rational>{tie}).value() ==
                               Rational::from_int(1),
                           "mode of tie {1,1,2,2} = 1 (smallest)");
                  auto tms = nimblecas::modes(std::span<const Rational>{tie}).value();
                  t.expect(tms.size() == 2 && tms[0] == Rational::from_int(1) &&
                               tms[1] == Rational::from_int(2),
                           "modes of tie = {1,2} ascending");
                  const std::vector<Rational> empty;
                  t.expect(nimblecas::mode(std::span<const Rational>{empty}).error() ==
                               MathError::domain_error,
                           "mode of empty is domain_error");
              })
        .test("pearson_correlation_squared",
              [](TestContext& t) {
                  const auto x = ints({1, 2, 3});
                  const auto yp = ints({2, 4, 6});  // perfectly correlated (y = 2x)
                  t.expect(nimblecas::pearson_correlation_squared(std::span<const Rational>{x},
                                                                  std::span<const Rational>{yp})
                                   .value() == Rational::from_int(1),
                           "r^2 of perfectly correlated data = 1");
                  const auto yn = ints({6, 4, 2});  // perfectly anti-correlated
                  t.expect(nimblecas::pearson_correlation_squared(std::span<const Rational>{x},
                                                                  std::span<const Rational>{yn})
                                   .value() == Rational::from_int(1),
                           "r^2 of perfectly anti-correlated data = 1");
                  // correlation_squared_matrix diagonal is exactly 1
                  std::vector<std::span<const Rational>> vars{std::span<const Rational>{x},
                                                              std::span<const Rational>{yp}};
                  auto r2 = nimblecas::correlation_squared_matrix(vars).value();
                  const auto one = Rational::from_int(1);
                  t.expect(r2.at(0, 0) == one && r2.at(1, 1) == one, "r^2 matrix diagonal = 1");
                  t.expect(r2.at(0, 1) == one && r2.at(0, 1) == r2.at(1, 0),
                           "r^2 matrix off-diagonal = 1 and symmetric");
                  // zero-variance variable -> domain_error
                  const auto flat = ints({5, 5, 5});
                  t.expect(nimblecas::pearson_correlation_squared(std::span<const Rational>{x},
                                                                  std::span<const Rational>{flat})
                                   .error() == MathError::domain_error,
                           "r^2 with a zero-variance variable is domain_error");
              })
        .test("coefficient_of_variation_squared",
              [](TestContext& t) {
                  // {1,2,3}: mean 2, sample variance 1 -> cv^2 = 1/4
                  const auto data = ints({1, 2, 3});
                  t.expect(nimblecas::coefficient_of_variation_squared(
                               std::span<const Rational>{data}, true)
                                   .value() == rat(1, 4),
                           "cv^2 (sample) of {1,2,3} = 1/4");
                  // population variance 2/3 -> cv^2 = (2/3)/4 = 1/6
                  t.expect(nimblecas::coefficient_of_variation_squared(
                               std::span<const Rational>{data}, false)
                                   .value() == rat(1, 6),
                           "cv^2 (population) of {1,2,3} = 1/6");
                  // zero mean -> domain_error
                  const auto zmean = std::vector<Rational>{rat(-1, 1), rat(1, 1)};
                  t.expect(nimblecas::coefficient_of_variation_squared(
                               std::span<const Rational>{zmean}, true)
                                   .error() == MathError::domain_error,
                           "cv^2 with zero mean is domain_error");
              })
        .run();
}
