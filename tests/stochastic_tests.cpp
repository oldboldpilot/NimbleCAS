// Tests for nimblecas.stochastic: Markov chains (exact stationary/hitting), WSS sample
// analysis, AR/MA/ARMA correlation, power spectral density (numeric), ergodicity.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.stats;
import nimblecas.stochastic;
import nimblecas.testing;

using nimblecas::MathError;
using nimblecas::Matrix;
using nimblecas::Rational;
using nimblecas::Spectrum;
using nimblecas::StabilityCertificate;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

[[nodiscard]] auto ints(std::vector<std::int64_t> vs) -> std::vector<Rational> {
    std::vector<Rational> out;
    out.reserve(vs.size());
    for (const std::int64_t v : vs) {
        out.push_back(Rational::from_int(v));
    }
    return out;
}

// Build a Matrix from an initializer of rational rows (asserts on success for tests).
[[nodiscard]] auto mat(std::vector<std::vector<Rational>> rows) -> Matrix {
    return Matrix::from_rows(std::move(rows)).value();
}

[[nodiscard]] auto close(double got, double expected, double tol = 1e-9) -> bool {
    return std::abs(got - expected) < tol;
}

// Verify pi P == pi exactly (pi as a row vector against transition matrix P).
[[nodiscard]] auto pi_fixed_point(const std::vector<Rational>& pi, const Matrix& p) -> bool {
    const std::size_t n = pi.size();
    for (std::size_t j = 0; j < n; ++j) {
        Rational acc;  // 0/1
        for (std::size_t i = 0; i < n; ++i) {
            acc = acc.add(pi[i].multiply(p.at(i, j)).value()).value();
        }
        if (!(acc == pi[j])) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] auto sum_is_one(const std::vector<Rational>& pi) -> bool {
    Rational acc;  // 0/1
    for (const Rational& v : pi) {
        acc = acc.add(v).value();
    }
    return acc == Rational::from_int(1);
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.stochastic")
        .test("markov_stationary_2x2_exact",
              [](TestContext& t) {
                  // P = [[3/4, 1/4], [1/2, 1/2]] -> pi = [2/3, 1/3].
                  const auto p = mat({{rat(3, 4), rat(1, 4)}, {rat(1, 2), rat(1, 2)}});
                  auto pi = nimblecas::stationary_distribution(p).value();
                  t.expect(pi.size() == 2, "pi has two components");
                  t.expect(pi[0] == rat(2, 3) && pi[1] == rat(1, 3), "pi == [2/3, 1/3]");
                  t.expect(pi_fixed_point(pi, p), "pi P == pi exactly");
                  t.expect(sum_is_one(pi), "sum pi == 1");
              })
        .test("markov_stationary_3x3_exact",
              [](TestContext& t) {
                  // P = [[1/2,1/2,0],[1/4,1/2,1/4],[0,1/2,1/2]] -> pi = [1/4,1/2,1/4].
                  const auto p = mat({{rat(1, 2), rat(1, 2), Rational{}},
                                      {rat(1, 4), rat(1, 2), rat(1, 4)},
                                      {Rational{}, rat(1, 2), rat(1, 2)}});
                  auto pi = nimblecas::stationary_distribution(p).value();
                  t.expect(pi[0] == rat(1, 4) && pi[1] == rat(1, 2) && pi[2] == rat(1, 4),
                           "pi == [1/4, 1/2, 1/4]");
                  t.expect(pi_fixed_point(pi, p), "pi P == pi exactly (3x3)");
                  t.expect(sum_is_one(pi), "sum pi == 1 (3x3)");
              })
        .test("n_step_transition_row_stochastic",
              [](TestContext& t) {
                  const auto p = mat({{rat(3, 4), rat(1, 4)}, {rat(1, 2), rat(1, 2)}});
                  // P^0 == identity.
                  auto p0 = nimblecas::n_step_transition(p, 0).value();
                  t.expect(p0.is_equal(Matrix::identity(2)), "P^0 == I");
                  // P^1 == P.
                  auto p1 = nimblecas::n_step_transition(p, 1).value();
                  t.expect(p1.is_equal(p), "P^1 == P");
                  // P^3 stays row-stochastic exactly.
                  auto p3 = nimblecas::n_step_transition(p, 3).value();
                  t.expect(nimblecas::is_stochastic(p3).value(), "P^3 is row-stochastic");
              })
        .test("is_stochastic_true_and_false",
              [](TestContext& t) {
                  const auto good = mat({{rat(3, 4), rat(1, 4)}, {rat(1, 2), rat(1, 2)}});
                  t.expect(nimblecas::is_stochastic(good).value(), "valid row-stochastic P");
                  // Row 0 sums to 3/4 != 1.
                  const auto bad_sum = mat({{rat(1, 2), rat(1, 4)}, {rat(1, 2), rat(1, 2)}});
                  t.expect(!nimblecas::is_stochastic(bad_sum).value(), "row not summing to 1");
                  // A negative entry.
                  const auto bad_neg = mat({{rat(3, 2), rat(-1, 2)}, {rat(1, 2), rat(1, 2)}});
                  t.expect(!nimblecas::is_stochastic(bad_neg).value(), "negative entry rejected");
                  // Non-square is not stochastic.
                  const auto rect = mat({{rat(1, 2), rat(1, 4), rat(1, 4)}});
                  t.expect(!nimblecas::is_stochastic(rect).value(), "non-square is not stochastic");
              })
        .test("irreducible_and_aperiodic",
              [](TestContext& t) {
                  const auto p = mat({{rat(3, 4), rat(1, 4)}, {rat(1, 2), rat(1, 2)}});
                  t.expect(nimblecas::is_irreducible(p).value(), "chain is irreducible");
                  t.expect(nimblecas::is_aperiodic(p).value(), "chain is aperiodic");
                  // A pure cycle 0->1->2->0 is irreducible but has period 3.
                  const auto cyc = mat({{Rational{}, Rational::from_int(1), Rational{}},
                                        {Rational{}, Rational{}, Rational::from_int(1)},
                                        {Rational::from_int(1), Rational{}, Rational{}}});
                  t.expect(nimblecas::is_irreducible(cyc).value(), "cycle is irreducible");
                  t.expect(!nimblecas::is_aperiodic(cyc).value(), "cycle is periodic (period 3)");
              })
        .test("mean_first_passage_and_recurrence",
              [](TestContext& t) {
                  const auto p = mat({{rat(3, 4), rat(1, 4)}, {rat(1, 2), rat(1, 2)}});
                  auto m = nimblecas::mean_first_passage_times(p).value();
                  // Mean recurrence times are 1/pi_i: 1/(2/3)=3/2, 1/(1/3)=3.
                  t.expect(m.at(0, 0) == rat(3, 2), "mean recurrence of state 0 == 3/2");
                  t.expect(m.at(1, 1) == Rational::from_int(3), "mean recurrence of state 1 == 3");
                  // First passage 0->1 is 4 and 1->0 is 2 for this chain.
                  t.expect(m.at(0, 1) == Rational::from_int(4), "E[hit 1 | start 0] == 4");
                  t.expect(m.at(1, 0) == Rational::from_int(2), "E[hit 0 | start 1] == 2");
              })
        .test("ergodic_stationary_mean_exact",
              [](TestContext& t) {
                  const auto p = mat({{rat(3, 4), rat(1, 4)}, {rat(1, 2), rat(1, 2)}});
                  // f(state) = [3, 0]; stationary mean = 2/3*3 + 1/3*0 = 2.
                  const auto f = ints({3, 0});
                  auto em = nimblecas::ergodic_mean(p, std::span<const Rational>{f}).value();
                  t.expect(em == Rational::from_int(2), "ergodic (space) mean == 2");
              })
        .test("ctmc_stationary_exact",
              [](TestContext& t) {
                  // Generator Q = [[-1, 1], [2, -2]] -> pi Q = 0 -> pi = [2/3, 1/3].
                  const auto q = mat({{rat(-1, 1), rat(1, 1)}, {rat(2, 1), rat(-2, 1)}});
                  t.expect(nimblecas::is_generator(q).value(), "Q is a valid generator");
                  auto pi = nimblecas::ctmc_stationary_distribution(q).value();
                  t.expect(pi[0] == rat(2, 3) && pi[1] == rat(1, 3), "CTMC pi == [2/3, 1/3]");
                  t.expect(sum_is_one(pi), "CTMC sum pi == 1");
                  // A matrix whose row does not sum to 0 is not a generator.
                  const auto notgen = mat({{rat(-1, 1), rat(2, 1)}, {rat(2, 1), rat(-2, 1)}});
                  t.expect(!nimblecas::is_generator(notgen).value(), "row-sum != 0 rejected");
              })
        .test("autocovariance_exact_R0_is_variance",
              [](TestContext& t) {
                  const auto x = ints({1, 2, 3});
                  const std::span<const Rational> sx{x};
                  // Biased R(0) equals the population variance 2/3.
                  auto r0 = nimblecas::autocovariance_at(sx, 0, true).value();
                  t.expect(r0 == rat(2, 3), "biased R(0) == 2/3");
                  t.expect(r0 == nimblecas::variance(sx, false).value(),
                           "R(0) == population variance");
                  // Autocorrelation is 1 at lag 0.
                  auto rho = nimblecas::autocorrelation(sx, 1, true).value();
                  t.expect(rho[0] == Rational::from_int(1), "rho(0) == 1");
                  // A record with genuine lag-1 structure: {1,2,1,2}, rho(1) = -3/4.
                  const auto y = ints({1, 2, 1, 2});
                  auto rhoy = nimblecas::autocorrelation(std::span<const Rational>{y}, 1, true).value();
                  t.expect(rhoy[1] == rat(-3, 4), "rho(1) of {1,2,1,2} == -3/4");
              })
        .test("cross_covariance_exact",
              [](TestContext& t) {
                  const auto x = ints({1, 2, 3});
                  const auto y = ints({3, 2, 1});
                  // Biased lag-0 cross-covariance == population covariance == -2/3.
                  auto c0 = nimblecas::cross_covariance_at(std::span<const Rational>{x},
                                                           std::span<const Rational>{y}, 0, true)
                                .value();
                  t.expect(c0 == rat(-2, 3), "cross-cov at lag 0 == -2/3");
                  // Self cross-covariance at lag 0 equals the biased autocovariance.
                  auto self = nimblecas::cross_covariance_at(std::span<const Rational>{x},
                                                             std::span<const Rational>{x}, 0, true)
                                  .value();
                  t.expect(self == nimblecas::autocovariance_at(std::span<const Rational>{x}, 0, true)
                                       .value(),
                           "cross-cov(x,x,0) == autocov(x,0)");
              })
        .test("is_wss_diagnostic",
              [](TestContext& t) {
                  // A repeating record: the two halves match exactly -> WSS-consistent.
                  const auto stat = ints({1, 2, 1, 2, 1, 2, 1, 2});
                  t.expect(nimblecas::is_wss(std::span<const Rational>{stat}, 1).value(),
                           "repeating record diagnoses as WSS");
                  // A record with a mean shift between halves -> not WSS.
                  const auto shift = ints({1, 1, 1, 1, 5, 5, 5, 5});
                  t.expect(!nimblecas::is_wss(std::span<const Rational>{shift}, 1).value(),
                           "mean-shifted record diagnoses as non-WSS");
              })
        .test("ar1_yule_walker_exact",
              [](TestContext& t) {
                  // AR(1) with phi = 1/2: rho(k) = (1/2)^k exactly.
                  const auto ar = std::vector<Rational>{rat(1, 2)};
                  auto rho = nimblecas::yule_walker_autocorrelation(std::span<const Rational>{ar}, 4)
                                 .value();
                  t.expect(rho[0] == Rational::from_int(1), "rho(0) == 1");
                  t.expect(rho[1] == rat(1, 2), "rho(1) == 1/2");
                  t.expect(rho[2] == rat(1, 4), "rho(2) == 1/4");
                  t.expect(rho[3] == rat(1, 8), "rho(3) == 1/8");
                  t.expect(rho[4] == rat(1, 16), "rho(4) == 1/16");
                  // Process variance with sigma^2 = 1: gamma(0) = 1 / (1 - 1/4) = 4/3.
                  auto gamma = nimblecas::ar_autocovariance(std::span<const Rational>{ar},
                                                            Rational::from_int(1), 2)
                                   .value();
                  t.expect(gamma[0] == rat(4, 3), "AR(1) variance gamma(0) == 4/3");
                  t.expect(gamma[1] == rat(2, 3), "gamma(1) == gamma(0)*rho(1) == 2/3");
              })
        .test("ar_stationarity_certificate",
              [](TestContext& t) {
                  // |phi| < 1 -> root 1/phi = 2 outside unit circle -> stable.
                  const auto stable = std::vector<Rational>{rat(1, 2)};
                  t.expect(nimblecas::ar_is_stationary(std::span<const Rational>{stable}).value() ==
                               StabilityCertificate::stable,
                           "AR(1) phi=1/2 is stationary");
                  // phi = 2 -> root 1/2 inside unit circle -> unstable.
                  const auto unstable = std::vector<Rational>{Rational::from_int(2)};
                  t.expect(nimblecas::ar_is_stationary(std::span<const Rational>{unstable}).value() ==
                               StabilityCertificate::unstable,
                           "AR(1) phi=2 is non-stationary");
                  // Phi(z) = 1 - z^2/2 has roots +-sqrt(2): irrational, unenumerated over Q.
                  const auto irr = std::vector<Rational>{Rational{}, rat(1, 2)};
                  t.expect(nimblecas::ar_is_stationary(std::span<const Rational>{irr}).value() ==
                               StabilityCertificate::indeterminate,
                           "irrational roots -> indeterminate (honest)");
              })
        .test("ma1_autocovariance_exact",
              [](TestContext& t) {
                  // MA(1) theta = 1/3, sigma^2 = 1: gamma(0)=10/9, gamma(1)=1/3, gamma(2)=0.
                  const auto ma = std::vector<Rational>{rat(1, 3)};
                  auto gamma = nimblecas::ma_autocovariance(std::span<const Rational>{ma},
                                                            Rational::from_int(1), 3)
                                   .value();
                  t.expect(gamma[0] == rat(10, 9), "MA(1) gamma(0) == 10/9");
                  t.expect(gamma[1] == rat(1, 3), "MA(1) gamma(1) == 1/3");
                  t.expect(gamma[2] == Rational{}, "MA(1) gamma(2) == 0");
                  t.expect(gamma[3] == Rational{}, "MA(1) gamma(3) == 0");
                  // MA(1) with |theta| < 1 is invertible.
                  t.expect(nimblecas::ma_is_invertible(std::span<const Rational>{ma}).value() ==
                               StabilityCertificate::stable,
                           "MA(1) theta=1/3 is invertible");
              })
        .test("psd_white_noise_flat_numeric",
              [](TestContext& t) {
                  // White-noise autocovariance R = [1, 0, 0]: S(f) is flat at 1.
                  const auto acov = ints({1, 0, 0});
                  auto spec = nimblecas::power_spectral_density(std::span<const Rational>{acov}, 5)
                                  .value();
                  t.expect(spec.values.size() == 5, "spectrum has 5 samples");
                  bool flat = true;
                  for (double v : spec.values) {
                      flat = flat && close(v, 1.0);
                  }
                  t.expect(flat, "white-noise PSD is flat at 1.0");
                  // ARMA closed form with no AR/MA terms and sigma^2 = 2 is flat at 2.
                  const std::vector<Rational> none;
                  auto aspec = nimblecas::arma_spectral_density(std::span<const Rational>{none},
                                                                std::span<const Rational>{none}, 2.0, 4)
                                   .value();
                  bool flat2 = true;
                  for (double v : aspec.values) {
                      flat2 = flat2 && close(v, 2.0);
                  }
                  t.expect(flat2, "white-noise ARMA PSD is flat at sigma^2 = 2");
              })
        .test("domain_errors",
              [](TestContext& t) {
                  // Non-square P -> stationary_distribution domain_error.
                  const auto rect = mat({{rat(1, 2), rat(1, 4), rat(1, 4)}});
                  t.expect(nimblecas::stationary_distribution(rect).error() ==
                               MathError::domain_error,
                           "non-square P is domain_error");
                  // Non-stochastic square P -> domain_error.
                  const auto bad = mat({{rat(1, 2), rat(1, 4)}, {rat(1, 2), rat(1, 2)}});
                  t.expect(nimblecas::stationary_distribution(bad).error() ==
                               MathError::domain_error,
                           "non-stochastic P is domain_error");
                  // Too-short record: lag >= N.
                  const auto x = ints({1, 2});
                  t.expect(nimblecas::autocovariance_at(std::span<const Rational>{x}, 2, true).error() ==
                               MathError::domain_error,
                           "lag >= N is domain_error");
                  // is_wss on a too-short record.
                  const auto tiny = ints({1, 2});
                  t.expect(nimblecas::is_wss(std::span<const Rational>{tiny}, 1).error() ==
                               MathError::domain_error,
                           "is_wss with max_lag too large is domain_error");
              })
        .run();
}
