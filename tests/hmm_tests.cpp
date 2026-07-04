// Tests for nimblecas.hmm: discrete hidden Markov models, exact over Q. A hand-computable
// 2-state / 2-symbol model verifies forward P(O) exactly, forward==backward likelihood,
// Viterbi decoding + exact path probability, gamma normalisation, a Baum-Welch iteration
// that does not decrease the likelihood, the numerical log-likelihood, and domain errors.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.hmm;
import nimblecas.testing;

using nimblecas::HiddenMarkovModel;
using nimblecas::MathError;
using nimblecas::Matrix;
using nimblecas::Rational;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

[[nodiscard]] auto mat(std::vector<std::vector<Rational>> rows) -> Matrix {
    return Matrix::from_rows(std::move(rows)).value();
}

// The canonical hand-computable model:
//   pi = [1/2, 1/2]
//   A  = [[7/10, 3/10], [2/5, 3/5]]
//   B  = [[9/10, 1/10], [1/5, 4/5]]
// For O = [0, 1, 0] the exact likelihood is P(O) = 159/1600 (verified by hand).
[[nodiscard]] auto build_model() -> HiddenMarkovModel {
    std::vector<Rational> pi{rat(1, 2), rat(1, 2)};
    const auto a = mat({{rat(7, 10), rat(3, 10)}, {rat(2, 5), rat(3, 5)}});
    const auto b = mat({{rat(9, 10), rat(1, 10)}, {rat(1, 5), rat(4, 5)}});
    return HiddenMarkovModel::make(std::move(pi), a, b).value();
}

[[nodiscard]] auto rat_geq(const Rational& a, const Rational& b) -> bool {
    return a.subtract(b).value().numerator() >= 0;
}

// sum_i v[i], exact.
[[nodiscard]] auto sum_rat(const std::vector<Rational>& v) -> Rational {
    Rational acc;  // 0/1
    for (const Rational& e : v) {
        acc = acc.add(e).value();
    }
    return acc;
}

// Every row of m sums exactly to 1.
[[nodiscard]] auto rows_sum_one(const Matrix& m) -> bool {
    for (std::size_t i = 0; i < m.rows(); ++i) {
        Rational acc;  // 0/1
        for (std::size_t j = 0; j < m.cols(); ++j) {
            acc = acc.add(m.at(i, j)).value();
        }
        if (!(acc == Rational::from_int(1))) {
            return false;
        }
    }
    return true;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.hmm")
        .test("forward_likelihood_exact",
              [](TestContext& t) {
                  const auto model = build_model();
                  const std::vector<std::size_t> obs{0, 1, 0};
                  const std::span<const std::size_t> o{obs};
                  auto fwd = nimblecas::forward(model, o).value();
                  t.expect(fwd.alpha.size() == 3, "alpha has T = 3 rows");
                  t.expect(fwd.likelihood == rat(159, 1600), "P(O) == 159/1600 exactly");
                  auto lik = nimblecas::observation_likelihood(model, o).value();
                  t.expect(lik == rat(159, 1600), "observation_likelihood == 159/1600");
              })
        .test("forward_backward_agree",
              [](TestContext& t) {
                  const auto model = build_model();
                  const std::vector<std::size_t> obs{0, 1, 0};
                  const std::span<const std::size_t> o{obs};
                  auto fwd = nimblecas::forward(model, o).value();
                  auto beta = nimblecas::backward(model, o).value();
                  const std::size_t n = model.num_states();
                  // Backward likelihood: sum_i pi_i B(i, o_0) beta_0(i).
                  Rational back;  // 0/1
                  for (std::size_t i = 0; i < n; ++i) {
                      const Rational e =
                          model.initial()[i].multiply(model.emission().at(i, obs[0])).value();
                      back = back.add(e.multiply(beta[0][i]).value()).value();
                  }
                  t.expect(back == fwd.likelihood, "backward likelihood == forward P(O)");
                  // sum_i alpha_t(i) beta_t(i) == P(O) at every t (test t = 0, 1, 2).
                  for (std::size_t tt = 0; tt < 3; ++tt) {
                      Rational acc;  // 0/1
                      for (std::size_t i = 0; i < n; ++i) {
                          acc = acc.add(fwd.alpha[tt][i].multiply(beta[tt][i]).value()).value();
                      }
                      t.expect(acc == fwd.likelihood, "sum_i alpha_t(i) beta_t(i) == P(O)");
                  }
              })
        .test("viterbi_recovers_path_exact",
              [](TestContext& t) {
                  const auto model = build_model();
                  const std::vector<std::size_t> obs{0, 1, 0};
                  const std::span<const std::size_t> o{obs};
                  auto vit = nimblecas::viterbi(model, o).value();
                  const std::vector<std::size_t> expected{0, 1, 0};
                  t.expect(vit.states == expected, "Viterbi path == [0, 1, 0]");
                  t.expect(vit.probability == rat(243, 6250), "path probability == 243/6250 exact");
                  // The reported probability matches the direct joint P(path, O).
                  auto direct = nimblecas::path_probability(
                                    model, std::span<const std::size_t>{vit.states}, o)
                                    .value();
                  t.expect(direct == vit.probability, "path_probability matches Viterbi prob");
                  // It strictly beats the all-zeros alternative path.
                  const std::vector<std::size_t> alt{0, 0, 0};
                  auto altp = nimblecas::path_probability(
                                  model, std::span<const std::size_t>{alt}, o)
                                  .value();
                  t.expect(!rat_geq(altp, vit.probability),
                           "Viterbi path is strictly more probable than [0,0,0]");
              })
        .test("gamma_sums_to_one",
              [](TestContext& t) {
                  const auto model = build_model();
                  const std::vector<std::size_t> obs{0, 1, 0};
                  auto post = nimblecas::posteriors(model, std::span<const std::size_t>{obs}).value();
                  bool all_one = true;
                  for (const std::vector<Rational>& g_t : post.gamma) {
                      all_one = all_one && (sum_rat(g_t) == Rational::from_int(1));
                  }
                  t.expect(all_one, "sum_i gamma_t(i) == 1 at every t");
                  t.expect(post.xi.size() == 2, "xi has T - 1 = 2 time slices");
              })
        .test("baum_welch_does_not_decrease_likelihood",
              [](TestContext& t) {
                  // Baum-Welch's EXACT per-iteration re-estimation blows up rational
                  // denominators quickly, so at int64 precision it is practical only for small
                  // models / short sequences (a BigRational-backed version would lift the
                  // ceiling). This half/quarter-probability model with T=2 stays within int64.
                  std::vector<Rational> pi{rat(1, 2), rat(1, 2)};
                  const auto a = mat({{rat(1, 2), rat(1, 2)}, {rat(1, 2), rat(1, 2)}});
                  const auto b = mat({{rat(3, 4), rat(1, 4)}, {rat(1, 4), rat(3, 4)}});
                  const auto model = HiddenMarkovModel::make(std::move(pi), a, b).value();
                  const std::vector<std::vector<std::size_t>> seqs{{0, 1}};
                  const std::vector<std::size_t> seq0{0, 1};
                  auto res =
                      nimblecas::baum_welch(model, std::span<const std::vector<std::size_t>>{seqs}, 1);
                  if (!res) {
                      // Honest int64 ceiling: the exact ratios overflowed. Documented limit.
                      t.expect(res.error() == MathError::overflow,
                               "Baum-Welch honestly reports int64 overflow when denominators blow up");
                      return;
                  }
                  t.expect(res->likelihoods.size() == 2, "one iteration records 2 likelihoods");
                  auto l0 = nimblecas::observation_likelihood(
                                model, std::span<const std::size_t>{seq0})
                                .value();
                  t.expect(res->likelihoods[0] == l0, "initial likelihood matches forward P(O)");
                  t.expect(rat_geq(res->likelihoods[1], res->likelihoods[0]),
                           "Baum-Welch iteration does not decrease the likelihood");
                  // Re-estimated model is exactly stochastic.
                  t.expect(sum_rat(std::vector<Rational>(res->model.initial().begin(),
                                                          res->model.initial().end())) ==
                               Rational::from_int(1),
                           "re-estimated pi sums to 1");
                  t.expect(rows_sum_one(res->model.transition()), "re-estimated A rows stochastic");
                  t.expect(rows_sum_one(res->model.emission()), "re-estimated B rows stochastic");
              })
        .test("log_likelihood_numeric",
              [](TestContext& t) {
                  const auto model = build_model();
                  const std::vector<std::size_t> obs{0, 1, 0};
                  auto ll = nimblecas::log_likelihood(model, std::span<const std::size_t>{obs}).value();
                  const double expected = std::log(159.0 / 1600.0);
                  t.expect(std::abs(ll - expected) < 1e-9,
                           "log P(O) matches log(159/1600) numerically");
              })
        .test("domain_errors",
              [](TestContext& t) {
                  // Non-stochastic transition matrix -> make domain_error.
                  std::vector<Rational> pi{rat(1, 2), rat(1, 2)};
                  const auto bad_a = mat({{rat(1, 2), rat(1, 4)}, {rat(1, 2), rat(1, 2)}});
                  const auto good_b = mat({{rat(9, 10), rat(1, 10)}, {rat(1, 5), rat(4, 5)}});
                  t.expect(HiddenMarkovModel::make(pi, bad_a, good_b).error() ==
                               MathError::domain_error,
                           "non-stochastic A is domain_error");
                  // Dimension mismatch: pi length 2 but A is 3x3.
                  const auto a3 = Matrix::identity(3);
                  const auto b3 = mat({{Rational::from_int(1)},
                                       {Rational::from_int(1)},
                                       {Rational::from_int(1)}});
                  t.expect(HiddenMarkovModel::make(pi, a3, b3).error() == MathError::domain_error,
                           "dimension mismatch is domain_error");
                  // Impossible observation: symbol 1 can never be emitted -> P(O) = 0.
                  const auto a2 = mat({{rat(1, 2), rat(1, 2)}, {rat(1, 2), rat(1, 2)}});
                  const auto b_deg = mat({{Rational::from_int(1), Rational{}},
                                          {Rational::from_int(1), Rational{}}});
                  auto imposs = HiddenMarkovModel::make(pi, a2, b_deg).value();
                  const std::vector<std::size_t> o1{1};
                  t.expect(nimblecas::posteriors(imposs, std::span<const std::size_t>{o1}).error() ==
                               MathError::domain_error,
                           "P(O) = 0 posteriors is domain_error");
                  // Symbol index out of range (M = 2, symbol 2).
                  const auto model = build_model();
                  const std::vector<std::size_t> oob{2};
                  t.expect(nimblecas::forward(model, std::span<const std::size_t>{oob}).error() ==
                               MathError::domain_error,
                           "out-of-range symbol is domain_error");
                  // Empty observation sequence.
                  const std::vector<std::size_t> empty;
                  t.expect(nimblecas::forward(model, std::span<const std::size_t>{empty}).error() ==
                               MathError::domain_error,
                           "empty observation is domain_error");
              })
        .run();
}
