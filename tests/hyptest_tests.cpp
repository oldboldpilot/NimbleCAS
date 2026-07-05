// Tests for nimblecas.hyptest: exact rational test statistics (t^2, chi^2, F, ANOVA,
// z^2), exact rational MLE point estimates, symbolic MLE models (score, Fisher), and
// the exact rational Wald / score statistics. Every expected value is hand-derived.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.hyptest;
import nimblecas.testing;

using nimblecas::Expr;
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

[[nodiscard]] auto ints(std::vector<std::int64_t> vs) -> std::vector<Rational> {
    std::vector<Rational> out;
    out.reserve(vs.size());
    for (const std::int64_t v : vs) {
        out.push_back(Rational::from_int(v));
    }
    return out;
}

[[nodiscard]] auto sp(const std::vector<Rational>& v) -> std::span<const Rational> {
    return std::span<const Rational>{v};
}

[[nodiscard]] auto mat(std::vector<std::vector<Rational>> rows) -> Matrix {
    return Matrix::from_rows(std::move(rows)).value();
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.hyptest")
        // ----- test statistics: exact rational -----
        .test("one_sample_t_squared_exact_fraction",
              [](TestContext& t) {
                  // data = 1,2,3,4 ; mu0 = 2. xbar = 5/2, sample var s^2 = 5/3.
                  // t^2 = n (xbar - mu0)^2 / s^2 = 4 * (1/2)^2 / (5/3) = 3/5.  df = n - 1 = 3.
                  const auto x = ints({1, 2, 3, 4});
                  auto ts = nimblecas::one_sample_t_squared(sp(x), ri(2)).value();
                  t.expect(ts.value == rat(3, 5), "t^2 = 3/5 exactly");
                  t.expect(ts.df1 == 3, "df = n - 1 = 3");
                  t.expect(!ts.df2.has_value(), "one-sample t^2 has a single df");
                  // Zero-variance (constant) data has no t-statistic.
                  const auto c = ints({7, 7, 7});
                  t.expect(nimblecas::one_sample_t_squared(sp(c), ri(0)).error() ==
                               MathError::domain_error,
                           "constant data => domain_error");
              })
        .test("two_sample_pooled_t_squared",
              [](TestContext& t) {
                  // x = 1,2,3 (mean 2, var 1); y = 3,4,5 (mean 4, var 1).
                  // pooled s_p^2 = (2*1 + 2*1)/(3+3-2) = 1. se^2 = 1*(1/3 + 1/3) = 2/3.
                  // t^2 = (2 - 4)^2 / (2/3) = 4 * 3/2 = 6. df = 4.
                  const auto x = ints({1, 2, 3});
                  const auto y = ints({3, 4, 5});
                  auto ts = nimblecas::two_sample_t_squared(sp(x), sp(y)).value();
                  t.expect(ts.value == ri(6), "pooled t^2 = 6 exactly");
                  t.expect(ts.df1 == 4, "df = n1 + n2 - 2 = 4");
              })
        .test("paired_t_squared",
              [](TestContext& t) {
                  // x = 2,4,6 ; y = 1,2,3 => d = 1,2,3. one-sample t^2 of d vs 0:
                  // dbar = 2, sample var = 1. t^2 = 3 * 2^2 / 1 = 12. df = 2.
                  const auto x = ints({2, 4, 6});
                  const auto y = ints({1, 2, 3});
                  auto ts = nimblecas::paired_t_squared(sp(x), sp(y)).value();
                  t.expect(ts.value == ri(12), "paired t^2 = 12 exactly");
                  t.expect(ts.df1 == 2, "df = n - 1 = 2");
              })
        .test("z_squared_known_variance",
              [](TestContext& t) {
                  // data = 1,2,3,4 ; mu0 = 2 ; sigma^2 = 1. z^2 = n (xbar-mu0)^2/sigma^2
                  // = 4 * (1/2)^2 / 1 = 1. z^2 ~ chi^2(1) so df = 1.
                  const auto x = ints({1, 2, 3, 4});
                  auto ts = nimblecas::z_squared(sp(x), ri(2), ri(1)).value();
                  t.expect(ts.value == ri(1), "z^2 = 1 exactly");
                  t.expect(ts.df1 == 1, "z^2 has 1 degree of freedom");
                  t.expect(nimblecas::z_squared(sp(x), ri(2), ri(0)).error() ==
                               MathError::domain_error,
                           "non-positive variance => domain_error");
              })
        .test("chi_squared_goodness_of_fit_exact",
              [](TestContext& t) {
                  // O = 10,20,30 ; E = 15,15,30. sum (O-E)^2/E = 25/15 + 25/15 + 0 = 10/3.
                  // df = k - 1 = 2.
                  const auto obs = ints({10, 20, 30});
                  const auto exp = ints({15, 15, 30});
                  auto ts = nimblecas::chi_squared_goodness_of_fit(sp(obs), sp(exp)).value();
                  t.expect(ts.value == rat(10, 3), "chi^2 = 10/3 exactly");
                  t.expect(ts.df1 == 2, "df = k - 1 = 2");
                  // A non-positive expected count is a domain error.
                  const auto bad = ints({15, 0, 30});
                  t.expect(nimblecas::chi_squared_goodness_of_fit(sp(obs), sp(bad)).error() ==
                               MathError::domain_error,
                           "zero expected count => domain_error");
              })
        .test("chi_squared_independence_exact",
              [](TestContext& t) {
                  // 2x2 table [[10,20],[30,40]]: R = 30,70 ; C = 40,60 ; N = 100.
                  // E = [[12,18],[28,42]]. sum (O-E)^2/E = 4/12+4/18+4/28+4/42 = 50/63.
                  // df = (2-1)(2-1) = 1.
                  const auto table = mat({{ri(10), ri(20)}, {ri(30), ri(40)}});
                  auto ts = nimblecas::chi_squared_independence(table).value();
                  t.expect(ts.value == rat(50, 63), "chi^2 independence = 50/63 exactly");
                  t.expect(ts.df1 == 1, "df = (r-1)(c-1) = 1");
              })
        .test("variance_ratio_f_exact",
              [](TestContext& t) {
                  // x = 1,2,3 (var 1); y = 1,3,5 (var 4). F = 1/4. df = (2, 2).
                  const auto x = ints({1, 2, 3});
                  const auto y = ints({1, 3, 5});
                  auto ts = nimblecas::variance_ratio_f(sp(x), sp(y)).value();
                  t.expect(ts.value == rat(1, 4), "F = 1/4 exactly");
                  t.expect(ts.df1 == 2, "numerator df = n_x - 1 = 2");
                  t.expect(ts.df2.has_value() && *ts.df2 == 2, "denominator df = n_y - 1 = 2");
              })
        .test("one_way_anova_f_exact",
              [](TestContext& t) {
                  // Groups (1,2,3),(4,5,6),(7,8,9): means 2,5,8; grand mean 5.
                  // SS_between = 3*9 + 0 + 3*9 = 54 ; SS_within = 2+2+2 = 6.
                  // F = (54/2)/(6/6) = 27. df = (k-1, N-k) = (2, 6).
                  const auto g1 = ints({1, 2, 3});
                  const auto g2 = ints({4, 5, 6});
                  const auto g3 = ints({7, 8, 9});
                  std::vector<std::span<const Rational>> groups{sp(g1), sp(g2), sp(g3)};
                  auto ts = nimblecas::one_way_anova_f(groups).value();
                  t.expect(ts.value == ri(27), "ANOVA F = 27 exactly");
                  t.expect(ts.df1 == 2, "between df = k - 1 = 2");
                  t.expect(ts.df2.has_value() && *ts.df2 == 6, "within df = N - k = 6");
              })
        .test("exceeds_is_exact_comparison",
              [](TestContext& t) {
                  // 10/3 vs critical 3 => exceeds ; vs critical 4 => not.
                  t.expect(nimblecas::exceeds(rat(10, 3), ri(3)).value(), "10/3 > 3");
                  t.expect(!nimblecas::exceeds(rat(10, 3), ri(4)).value(), "10/3 !> 4");
              })
        // ----- MLE point estimates: exact rational -----
        .test("bernoulli_poisson_mle_are_sample_mean",
              [](TestContext& t) {
                  const auto b = ints({1, 0, 1, 1, 0});  // xbar = 3/5
                  t.expect(nimblecas::bernoulli_mle(sp(b)).value() == rat(3, 5),
                           "Bernoulli p-hat = xbar = 3/5");
                  const auto p = ints({2, 3, 4});  // xbar = 3
                  t.expect(nimblecas::poisson_mle(sp(p)).value() == ri(3),
                           "Poisson lambda-hat = xbar = 3");
                  t.expect(nimblecas::normal_mean_mle(sp(p)).value() == ri(3),
                           "Normal mu-hat = xbar = 3");
              })
        .test("exponential_geometric_mle_are_reciprocal_mean",
              [](TestContext& t) {
                  const auto e = ints({1, 1, 2});  // xbar = 4/3 => 1/xbar = 3/4
                  t.expect(nimblecas::exponential_mle(sp(e)).value() == rat(3, 4),
                           "Exponential lambda-hat = 1/xbar = 3/4");
                  t.expect(nimblecas::geometric_mle(sp(e)).value() == rat(3, 4),
                           "Geometric p-hat = 1/xbar = 3/4");
                  // A zero sample mean makes 1/xbar undefined.
                  const auto z = ints({-1, 1});  // xbar = 0
                  t.expect(nimblecas::exponential_mle(sp(z)).error() == MathError::domain_error,
                           "zero mean => domain_error");
              })
        .test("normal_variance_mle_is_population_variance",
              [](TestContext& t) {
                  const auto d = ints({1, 2, 3});  // pop var = ((-1)^2+0+1^2)/3 = 2/3
                  t.expect(nimblecas::normal_variance_mle(sp(d)).value() == rat(2, 3),
                           "Normal sigma^2-hat = population variance = 2/3");
              })
        // ----- MLE symbolic models: score vanishes at the MLE -----
        .test("poisson_score_vanishes_at_mle",
              [](TestContext& t) {
                  auto model = nimblecas::poisson_mle_model().value();
                  // U(lambda) with lambda := lambda-hat = m must simplify to 0.
                  auto at_mle = nimblecas::substitute(
                      model.score, Expr::symbol(model.parameter), model.mle);
                  auto zero = nimblecas::simplify(at_mle).value();
                  t.expect(zero == Expr::integer(0), "Poisson score U(m) = 0");
              })
        .test("exponential_score_vanishes_at_mle",
              [](TestContext& t) {
                  auto model = nimblecas::exponential_mle_model().value();
                  // lambda-hat = m^(-1); U(m^(-1)) must simplify to 0.
                  auto at_mle = nimblecas::substitute(
                      model.score, Expr::symbol(model.parameter), model.mle);
                  auto zero = nimblecas::simplify(at_mle).value();
                  t.expect(zero == Expr::integer(0), "Exponential score U(1/m) = 0");
              })
        .test("normal_mean_score_vanishes_at_mle",
              [](TestContext& t) {
                  auto model = nimblecas::normal_mean_mle_model().value();
                  auto at_mle = nimblecas::substitute(
                      model.score, Expr::symbol(model.parameter), model.mle);
                  auto zero = nimblecas::simplify(at_mle).value();
                  t.expect(zero == Expr::integer(0), "Normal-mean score U(m) = 0");
              })
        .test("bernoulli_score_vanishes_at_mle",
              [](TestContext& t) {
                  auto model = nimblecas::bernoulli_mle_model().value();
                  // p-hat = m; U(m) folds via (1-m)^1 (1-m)^(-1) -> 1 and m^1 m^(-1) -> 1.
                  auto at_mle = nimblecas::substitute(
                      model.score, Expr::symbol(model.parameter), model.mle);
                  auto zero = nimblecas::simplify(at_mle).value();
                  t.expect(zero == Expr::integer(0), "Bernoulli score U(m) = 0");
              })
        // ----- Fisher information -----
        .test("bernoulli_fisher_information_identity",
              [](TestContext& t) {
                  // i(p) = 1/(p(1-p)) is verified by the identity i(p) * p * (1-p) = 1,
                  // which folds by like-base cancellation in a single simplify pass.
                  auto model = nimblecas::bernoulli_mle_model().value();
                  const Expr p = Expr::symbol("p");
                  const Expr one_minus_p = Expr::sum({Expr::integer(1),
                                                      Expr::product({Expr::integer(-1), p})});
                  auto identity = Expr::product({model.fisher_information, p, one_minus_p});
                  auto folded = nimblecas::simplify(identity).value();
                  t.expect(folded == Expr::integer(1), "i(p) * p * (1-p) = 1");
              })
        .test("poisson_fisher_information_identity",
              [](TestContext& t) {
                  // i(lambda) = 1/lambda, verified by i(lambda) * lambda = 1.
                  auto model = nimblecas::poisson_mle_model().value();
                  const Expr lambda = Expr::symbol("lambda");
                  auto identity = Expr::product({model.fisher_information, lambda});
                  auto folded = nimblecas::simplify(identity).value();
                  t.expect(folded == Expr::integer(1), "i(lambda) * lambda = 1");
              })
        // ----- Wald / score statistics: exact rational -----
        .test("bernoulli_wald_and_score_statistics",
              [](TestContext& t) {
                  const auto d = ints({1, 0, 1, 1, 0});  // p-hat = 3/5, n = 5
                  // Wald = n (p-hat - p0)^2 / (p-hat(1-p-hat)) = 5*(1/10)^2/(6/25) = 5/24.
                  t.expect(nimblecas::bernoulli_wald_statistic(sp(d), rat(1, 2)).value() ==
                               rat(5, 24),
                           "Bernoulli Wald = 5/24");
                  // Score = n (xbar - p0)^2 / (p0(1-p0)) = 5*(1/10)^2/(1/4) = 1/5.
                  t.expect(nimblecas::bernoulli_score_statistic(sp(d), rat(1, 2)).value() ==
                               rat(1, 5),
                           "Bernoulli score = 1/5");
                  // A degenerate null p0 in {0,1} is a domain error for the score statistic.
                  t.expect(nimblecas::bernoulli_score_statistic(sp(d), ri(1)).error() ==
                               MathError::domain_error,
                           "score at p0 = 1 => domain_error");
              })
        .test("poisson_wald_and_score_statistics",
              [](TestContext& t) {
                  const auto d = ints({2, 3, 4});  // lambda-hat = 3, n = 3
                  // Wald = n (lambda-hat - lambda0)^2 / lambda-hat = 3*1/3 = 1.
                  t.expect(nimblecas::poisson_wald_statistic(sp(d), ri(2)).value() == ri(1),
                           "Poisson Wald = 1");
                  // Score = n (xbar - lambda0)^2 / lambda0 = 3*1/2 = 3/2.
                  t.expect(nimblecas::poisson_score_statistic(sp(d), ri(2)).value() == rat(3, 2),
                           "Poisson score = 3/2");
              })
        // ----- likelihood-ratio statistic: exact symbolic (transcendental) -----
        .test("log_likelihood_ratio_is_symbolic",
              [](TestContext& t) {
                  // G^2 = 2(ell(theta-hat) - ell(theta0)) is transcendental (contains ln):
                  // the honest deliverable is an exact Expr, not a fabricated rational.
                  auto model = nimblecas::poisson_mle_model().value();
                  auto g = nimblecas::log_likelihood_ratio(model, model.mle, Expr::integer(1));
                  t.expect(g.has_value(), "log_likelihood_ratio returns an exact symbolic Expr");
              })
        .run();
}
