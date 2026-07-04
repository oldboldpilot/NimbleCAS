// Tests for nimblecas.smc: Sequential Monte Carlo (ROADMAP 7.8).
// @author Olumuyiwa Oluwasanmi
//
// SMC output is a STATISTICAL ESTIMATE, not an exact value, so the statistical checks use
// a fixed seed, large sample budgets and deliberately loose tolerances; the determinism,
// count-preservation and domain-error checks are exact. The particle filter is validated
// against an exact 1-D Kalman filter computed inline in the test.

import std;
import nimblecas.core;
import nimblecas.rng;
import nimblecas.smc;
import nimblecas.testing;

using nimblecas::MathError;
using nimblecas::Rng;
using nimblecas::smc::antithetic_estimate;
using nimblecas::smc::bootstrap_particle_filter;
using nimblecas::smc::control_variate_estimate;
using nimblecas::smc::effective_sample_size;
using nimblecas::smc::Estimate;
using nimblecas::smc::multinomial_resample;
using nimblecas::smc::plain_estimate;
using nimblecas::smc::ResampleScheme;
using nimblecas::smc::residual_resample;
using nimblecas::smc::stratified_estimate;
using nimblecas::smc::stratified_resample;
using nimblecas::smc::systematic_resample;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Standard-normal draw via Box-Muller on two uniforms from the counter-based Rng. The tiny
// floor on u1 avoids log(0) = -inf if next_unit() returns exactly 0.
[[nodiscard]] auto gaussian(Rng& rng) -> double {
    const double u1 = std::max(rng.next_unit(), 1e-300);
    const double u2 = rng.next_unit();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * std::numbers::pi * u2);
}

// True whenever every parent index is a valid particle index in [0, n).
[[nodiscard]] auto indices_valid(std::span<const std::size_t> parents, std::size_t n) -> bool {
    for (const std::size_t p : parents) {
        if (p >= n) {
            return false;
        }
    }
    return true;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.smc")
        .test("resampling_preserves_count_and_validity",
              [](TestContext& t) {
                  // A non-degenerate weight vector: every scheme must return exactly N valid
                  // parent indices.
                  const std::vector<double> w{0.1, 0.2, 0.05, 0.4, 0.25};
                  const std::size_t n = w.size();
                  auto mn = multinomial_resample(w, 11);
                  auto sy = systematic_resample(w, 11);
                  auto st = stratified_resample(w, 11);
                  auto re = residual_resample(w, 11);
                  t.expect(mn && mn->size() == n && indices_valid(*mn, n),
                           "multinomial returns N valid indices");
                  t.expect(sy && sy->size() == n && indices_valid(*sy, n),
                           "systematic returns N valid indices");
                  t.expect(st && st->size() == n && indices_valid(*st, n),
                           "stratified returns N valid indices");
                  t.expect(re && re->size() == n && indices_valid(*re, n),
                           "residual returns N valid indices");
              })
        .test("degenerate_weights_pick_the_survivor",
              [](TestContext& t) {
                  // One particle carries all the mass: every scheme must select it for every
                  // slot.
                  const std::vector<double> w{0.0, 0.0, 1.0, 0.0};
                  const std::size_t hot = 2;
                  auto check_all_hot = [&](const std::vector<std::size_t>& p) -> bool {
                      if (p.size() != w.size()) {
                          return false;
                      }
                      for (const std::size_t idx : p) {
                          if (idx != hot) {
                              return false;
                          }
                      }
                      return true;
                  };
                  auto mn = multinomial_resample(w, 5);
                  auto sy = systematic_resample(w, 5);
                  auto st = stratified_resample(w, 5);
                  auto re = residual_resample(w, 5);
                  t.expect(mn && check_all_hot(*mn), "multinomial picks only the survivor");
                  t.expect(sy && check_all_hot(*sy), "systematic picks only the survivor");
                  t.expect(st && check_all_hot(*st), "stratified picks only the survivor");
                  t.expect(re && check_all_hot(*re), "residual picks only the survivor");
              })
        .test("effective_sample_size_bounds",
              [](TestContext& t) {
                  // Uniform weights => ESS == N (no degeneracy).
                  const std::vector<double> uniform(8, 1.0 / 8.0);
                  auto eu = effective_sample_size(uniform);
                  t.expect(eu && std::abs(eu.value() - 8.0) < 1e-9,
                           "uniform weights => ESS == N");

                  // One-hot weights => ESS == 1 (total collapse).
                  const std::vector<double> one_hot{0.0, 1.0, 0.0, 0.0};
                  auto eo = effective_sample_size(one_hot);
                  t.expect(eo && std::abs(eo.value() - 1.0) < 1e-9,
                           "one-hot weights => ESS == 1");
              })
        .test("low_variance_schemes_are_deterministic",
              [](TestContext& t) {
                  // Systematic & stratified must be bit-identical across calls with the same
                  // seed (this determinism is exactly what makes them thread-count
                  // independent: every draw is a pure function of the seed).
                  const std::vector<double> w{0.15, 0.25, 0.1, 0.3, 0.2};
                  auto sy1 = systematic_resample(w, 2024);
                  auto sy2 = systematic_resample(w, 2024);
                  auto st1 = stratified_resample(w, 2024);
                  auto st2 = stratified_resample(w, 2024);
                  t.expect(sy1 && sy2 && *sy1 == *sy2,
                           "systematic is deterministic for a fixed seed");
                  t.expect(st1 && st2 && *st1 == *st2,
                           "stratified is deterministic for a fixed seed");
              })
        .test("resampling_domain_errors",
              [](TestContext& t) {
                  const std::vector<double> empty;
                  auto e1 = systematic_resample(empty, 1);
                  t.expect(!e1 && e1.error() == MathError::domain_error,
                           "empty weights => domain_error");

                  const std::vector<double> all_zero{0.0, 0.0, 0.0};
                  auto e2 = multinomial_resample(all_zero, 1);
                  t.expect(!e2 && e2.error() == MathError::domain_error,
                           "all-zero (non-normalizable) weights => domain_error");

                  const std::vector<double> has_nan{
                      0.5, std::numeric_limits<double>::quiet_NaN(), 0.5};
                  auto e3 = stratified_resample(has_nan, 1);
                  t.expect(!e3 && e3.error() == MathError::undefined_value,
                           "NaN weight => undefined_value");

                  auto e4 = effective_sample_size(empty);
                  t.expect(!e4 && e4.error() == MathError::domain_error,
                           "ESS on empty => domain_error");
              })
        .test("bootstrap_filter_tracks_kalman",
              [](TestContext& t) {
                  // Linear-Gaussian 1-D model: x_t = A x_{t-1} + w, w~N(0,Q);
                  // y_t = x_t + v, v~N(0,R); x_0 ~ N(m0, P0). The particle filter's filtered
                  // mean must track the EXACT Kalman posterior mean (computed inline) within
                  // a few Monte Carlo standard errors.
                  constexpr double A = 0.8;
                  constexpr double Q = 0.5;
                  constexpr double R = 0.5;
                  constexpr double m0 = 0.0;
                  constexpr double P0 = 1.0;
                  const std::vector<double> ys{0.5, 1.2, 0.9, 1.8, 2.1,
                                               1.5, 1.0, 0.3, -0.2, 0.4};

                  // Exact Kalman filter.
                  std::vector<double> kalman_means;
                  kalman_means.reserve(ys.size());
                  double m = m0;
                  double p = P0;
                  for (const double y : ys) {
                      const double m_pred = A * m;
                      const double p_pred = A * A * p + Q;
                      const double s = p_pred + R;
                      const double k = p_pred / s;
                      m = m_pred + k * (y - m_pred);
                      p = (1.0 - k) * p_pred;
                      kalman_means.push_back(m);
                  }

                  // Particle filter over the same observations.
                  auto init = [&](Rng& rng) -> double { return m0 + std::sqrt(P0) * gaussian(rng); };
                  auto propagate = [&](double x, Rng& rng) -> double {
                      return A * x + std::sqrt(Q) * gaussian(rng);
                  };
                  auto loglik = [&](double x, double y) -> double {
                      const double d = y - x;
                      return -0.5 * d * d / R;  // + const (drops out under normalisation)
                  };

                  auto r = bootstrap_particle_filter(init, propagate, loglik, ys, 40000, 1234,
                                                     0.5, ResampleScheme::systematic);
                  t.expect(r.has_value(), "filter runs on the linear-Gaussian model");
                  if (r) {
                      t.expect(r->filtered_means.size() == ys.size(),
                               "one filtered mean per observation");
                      // Compare the FINAL filtered mean to the exact Kalman mean (loose tol
                      // for Monte Carlo noise with N = 40000).
                      const double pf_last = r->filtered_means.back();
                      const double kf_last = kalman_means.back();
                      t.expect(std::abs(pf_last - kf_last) < 0.1,
                               "PF filtered mean tracks the exact Kalman mean (tol 0.1)");
                      // ESS diagnostics are within [1, N].
                      bool ess_ok = r->effective_sample_sizes.size() == ys.size();
                      for (const double e : r->effective_sample_sizes) {
                          if (!(e >= 1.0 - 1e-6 && e <= 40000.0 + 1e-6)) {
                              ess_ok = false;
                          }
                      }
                      t.expect(ess_ok, "every step's ESS lies in [1, N]");
                  }

                  // Reproducibility: the same seed reproduces the estimate bit-for-bit.
                  auto again = bootstrap_particle_filter(init, propagate, loglik, ys, 40000,
                                                         1234, 0.5, ResampleScheme::systematic);
                  t.expect(again.has_value(), "rerun succeeds");
                  if (r && again) {
                      bool same = r->filtered_means.size() == again->filtered_means.size();
                      for (std::size_t i = 0; same && i < r->filtered_means.size(); ++i) {
                          if (r->filtered_means[i] != again->filtered_means[i]) {
                              same = false;
                          }
                      }
                      t.expect(same && r->log_marginal_likelihood ==
                                           again->log_marginal_likelihood,
                               "same seed => bit-identical filter estimate");
                  }
              })
        .test("bootstrap_filter_domain_errors",
              [](TestContext& t) {
                  auto init = [](Rng&) -> double { return 0.0; };
                  auto prop = [](double x, Rng&) -> double { return x; };
                  auto ll = [](double, double) -> double { return 0.0; };
                  const std::vector<double> ys{1.0, 2.0};

                  auto no_particles = bootstrap_particle_filter(init, prop, ll, ys, 0, 1);
                  t.expect(!no_particles && no_particles.error() == MathError::domain_error,
                           "particles == 0 => domain_error");

                  const std::vector<double> no_obs;
                  auto empty_obs = bootstrap_particle_filter(init, prop, ll, no_obs, 100, 1);
                  t.expect(!empty_obs && empty_obs.error() == MathError::domain_error,
                           "empty observations => domain_error");

                  // A log-likelihood that is -inf everywhere makes a step non-normalizable.
                  auto ll_dead = [](double, double) -> double {
                      return -std::numeric_limits<double>::infinity();
                  };
                  auto dead = bootstrap_particle_filter(init, prop, ll_dead, ys, 100, 1);
                  t.expect(!dead && dead.error() == MathError::domain_error,
                           "all-zero likelihood step => domain_error");
              })
        .test("antithetic_unbiased_and_lower_variance",
              [](TestContext& t) {
                  // Integrate f(x) = x on [0, 1]; exact value 1/2. Monotone integrand, so the
                  // antithetic pair is negatively correlated and its std_error must beat plain
                  // Monte Carlo at the SAME sample count (2 * pairs).
                  auto f = [](double x) -> double { return x; };
                  const std::uint64_t pairs = 50000;
                  auto anti = antithetic_estimate(f, 0.0, 1.0, pairs, 7);
                  auto plain = plain_estimate(f, 0.0, 1.0, 2 * pairs, 7);
                  t.expect(anti.has_value() && plain.has_value(), "both estimators run");
                  if (anti && plain) {
                      t.expect(std::abs(anti->value - 0.5) < 1e-3,
                               "antithetic estimate ≈ 0.5 (unbiased, tol 1e-3)");
                      t.expect(anti->std_error < plain->std_error,
                               "antithetic std_error < plain MC at equal sample count");
                  }
              })
        .test("control_variate_unbiased_and_lower_variance",
              [](TestContext& t) {
                  // Integrate f(x) = x^2 on [0, 1]; exact value 1/3. Control h(x) = x has known
                  // mean 1/2 over Uniform[0,1] and is strongly correlated with x^2, so the
                  // std_error must fall below plain MC at the same sample count.
                  auto f = [](double x) -> double { return x * x; };
                  auto h = [](double x) -> double { return x; };
                  const std::uint64_t samples = 100000;
                  auto cv = control_variate_estimate(f, h, 0.5, 0.0, 1.0, samples, 99);
                  auto plain = plain_estimate(f, 0.0, 1.0, samples, 99);
                  t.expect(cv.has_value() && plain.has_value(), "both estimators run");
                  if (cv && plain) {
                      t.expect(std::abs(cv->value - (1.0 / 3.0)) < 2e-3,
                               "control-variate estimate ≈ 1/3 (consistent, tol 2e-3)");
                      t.expect(cv->std_error < plain->std_error,
                               "control-variate std_error < plain MC at equal sample count");
                  }
              })
        .test("stratified_unbiased_and_lower_variance",
              [](TestContext& t) {
                  // Integrate f(x) = x on [0, 1]; exact 1/2. Stratification removes the
                  // between-stratum variance, so the reported std_error is below plain MC at
                  // the same strata * per_stratum sample count.
                  auto f = [](double x) -> double { return x; };
                  const std::uint64_t strata = 100;
                  const std::uint64_t per = 500;
                  auto strat = stratified_estimate(f, 0.0, 1.0, strata, per, 3);
                  auto plain = plain_estimate(f, 0.0, 1.0, strata * per, 3);
                  t.expect(strat.has_value() && plain.has_value(), "both estimators run");
                  if (strat && plain) {
                      t.expect(std::abs(strat->value - 0.5) < 1e-3,
                               "stratified estimate ≈ 0.5 (unbiased, tol 1e-3)");
                      t.expect(strat->std_error < plain->std_error,
                               "stratified std_error < plain MC at equal sample count");
                  }
              })
        .test("estimator_domain_errors",
              [](TestContext& t) {
                  auto f = [](double x) -> double { return x; };
                  auto bad_a = plain_estimate(f, 1.0, 0.0, 100, 1);  // b < a
                  t.expect(!bad_a && bad_a.error() == MathError::domain_error,
                           "plain: b < a => domain_error");
                  auto no_pairs = antithetic_estimate(f, 0.0, 1.0, 0, 1);
                  t.expect(!no_pairs && no_pairs.error() == MathError::domain_error,
                           "antithetic: pairs == 0 => domain_error");
                  auto no_strata = stratified_estimate(f, 0.0, 1.0, 0, 10, 1);
                  t.expect(!no_strata && no_strata.error() == MathError::domain_error,
                           "stratified: strata == 0 => domain_error");
              })
        .run();
}
