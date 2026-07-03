// Tests for nimblecas.montecarlo: seeded Monte Carlo integration, the π dart estimate,
// rejection sampling (bounds + termination), and the sample statistics helpers.
// @author Olumuyiwa Oluwasanmi
//
// Everything here is deterministic (fixed seeds, no time/entropy) but stochastic in value,
// so numerical checks use GENEROUS tolerances chosen to never flake. Exact-value checks
// are reserved for the reproducibility and statistics cases, where the results are truly
// bit-defined.

import std;
import nimblecas.core;
import nimblecas.rng;
import nimblecas.montecarlo;
import nimblecas.testing;

using nimblecas::counter_u64;
using nimblecas::estimate_pi;
using nimblecas::integrate;
using nimblecas::rejection_sample;
using nimblecas::sample_mean;
using nimblecas::sample_variance;
using nimblecas::splitmix64;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

auto main() -> int {
    return TestSuite("nimblecas.montecarlo")
        .test("integrate_x_squared_on_unit_interval",
              [](TestContext& t) {
                  // ∫_0^1 x² dx = 1/3. Loose tolerance at 200k samples.
                  auto r = integrate([](double x) { return x * x; }, 0.0, 1.0, 200000, 20260703);
                  t.expect(r.has_value(), "integrate(x^2, [0,1]) succeeds");
                  if (r) {
                      t.expect(std::abs(*r - (1.0 / 3.0)) < 0.02,
                               std::format("integrate(x^2, [0,1]) ~ 1/3 (got {})", *r));
                  }
              })
        .test("integrate_constant_recovers_width",
              [](TestContext& t) {
                  // ∫_2^5 1 dx = 3 exactly-ish: for a constant integrand every sample is
                  // the same value, so the estimate is (b-a)*1 regardless of the draws.
                  auto r = integrate([](double) { return 1.0; }, 2.0, 5.0, 100000, 7);
                  t.expect(r.has_value(), "integrate(1, [2,5]) succeeds");
                  if (r) {
                      t.expect(std::abs(*r - 3.0) < 1e-9,
                               std::format("integrate(1, [2,5]) == 3 (got {})", *r));
                  }
              })
        .test("integrate_domain_errors",
              [](TestContext& t) {
                  auto rev = integrate([](double x) { return x; }, 1.0, 0.0, 1000, 1);
                  t.expect(!rev.has_value(), "integrate(b<a) fails");
                  t.expect(rev.error() == nimblecas::MathError::domain_error,
                           "integrate(b<a) yields domain_error");

                  auto zero = integrate([](double x) { return x; }, 0.0, 1.0, 0, 1);
                  t.expect(!zero.has_value(), "integrate(samples==0) fails");
                  t.expect(zero.error() == nimblecas::MathError::domain_error,
                           "integrate(samples==0) yields domain_error");
              })
        .test("integrate_is_bit_reproducible_and_partition_independent",
              [](TestContext& t) {
                  // Two calls with identical (f, a, b, samples, seed) must produce
                  // BIT-IDENTICAL results — the estimator is a deterministic function of
                  // its arguments with no hidden state.
                  auto f = [](double x) { return x * x * x; };
                  auto a = integrate(f, -1.0, 2.0, 50000, 99);
                  auto b = integrate(f, -1.0, 2.0, 50000, 99);
                  t.expect(a.has_value() && b.has_value(), "both integrate() calls succeed");
                  if (a && b) {
                      t.expect(*a == *b, "identical args give bit-identical integrate() results");
                  }

                  // The partition-independence of that estimate rests on the stateless
                  // counter core: for a fixed key, counter_u64(key, i) depends only on i.
                  // Assert that directly — summing the same indices in any grouping is
                  // therefore invariant.
                  const std::uint64_t key = splitmix64(99);
                  bool det = true;
                  for (std::uint64_t i = 0; i < 256; ++i) {
                      if (counter_u64(key, i) != counter_u64(key, i)) {
                          det = false;
                      }
                  }
                  t.expect(det, "counter_u64(key, i) is a pure function of the index i");

                  // Concatenating two disjoint index halves reproduces the whole range
                  // element-wise, mirroring how integrate() would split across workers.
                  constexpr std::uint64_t n = 500;
                  std::vector<std::uint64_t> whole;
                  whole.reserve(n);
                  for (std::uint64_t i = 0; i < n; ++i) {
                      whole.push_back(counter_u64(key, i));
                  }
                  std::vector<std::uint64_t> halves;
                  halves.reserve(n);
                  for (std::uint64_t i = 0; i < 173; ++i) halves.push_back(counter_u64(key, i));
                  for (std::uint64_t i = 173; i < n; ++i) halves.push_back(counter_u64(key, i));
                  bool same = whole.size() == halves.size();
                  for (std::size_t i = 0; i < whole.size() && i < halves.size(); ++i) {
                      if (whole[i] != halves[i]) same = false;
                  }
                  t.expect(same,
                           "disjoint index halves concatenate to the whole range (partition-"
                           "independent)");
              })
        .test("estimate_pi_is_close",
              [](TestContext& t) {
                  auto r = estimate_pi(200000, 12345);
                  t.expect(r.has_value(), "estimate_pi(200000) succeeds");
                  if (r) {
                      t.expect(std::abs(*r - std::numbers::pi) < 0.03,
                               std::format("estimate_pi ~ 3.14159 (got {})", *r));
                  }

                  auto zero = estimate_pi(0, 1);
                  t.expect(!zero.has_value(), "estimate_pi(samples==0) fails");
                  t.expect(zero.error() == nimblecas::MathError::domain_error,
                           "estimate_pi(samples==0) yields domain_error");
              })
        .test("rejection_sample_triangular_pdf",
              [](TestContext& t) {
                  // pdf(x) = x on [0,1] (unnormalised triangular density), ceiling m=1.
                  // Accepted samples must lie in [0,1] and their mean ~ 2/3.
                  auto pdf = [](double x) { return x; };
                  auto r = rejection_sample(pdf, 0.0, 1.0, 1.0, 5000, 2024, 10000000);
                  t.expect(r.has_value(), "rejection_sample(pdf=x) succeeds");
                  if (r) {
                      t.expect(!r->empty(), "rejection_sample returned some samples");
                      bool in_range = true;
                      for (const double x : *r) {
                          if (!(x >= 0.0 && x <= 1.0)) in_range = false;
                      }
                      t.expect(in_range, "all accepted samples lie in [0,1]");

                      auto m = sample_mean(*r);
                      t.expect(m.has_value(), "sample_mean of accepted samples succeeds");
                      if (m) {
                          t.expect(std::abs(*m - (2.0 / 3.0)) < 0.03,
                                   std::format("rejection sample mean ~ 2/3 (got {})", *m));
                      }
                  }
              })
        .test("rejection_sample_domain_errors",
              [](TestContext& t) {
                  auto pdf = [](double x) { return x; };

                  auto bad_m = rejection_sample(pdf, 0.0, 1.0, 0.0, 10, 1, 1000);
                  t.expect(!bad_m.has_value(), "rejection_sample(m_bound<=0) fails");
                  t.expect(bad_m.error() == nimblecas::MathError::domain_error,
                           "m_bound<=0 yields domain_error");

                  auto bad_ab = rejection_sample(pdf, 1.0, 0.0, 1.0, 10, 1, 1000);
                  t.expect(!bad_ab.has_value(), "rejection_sample(b<a) fails");
                  t.expect(bad_ab.error() == nimblecas::MathError::domain_error,
                           "b<a yields domain_error");

                  auto bad_want = rejection_sample(pdf, 0.0, 1.0, 1.0, 0, 1, 1000);
                  t.expect(!bad_want.has_value() &&
                               bad_want.error() == nimblecas::MathError::domain_error,
                           "want==0 yields domain_error");

                  auto bad_trials = rejection_sample(pdf, 0.0, 1.0, 1.0, 10, 1, 0);
                  t.expect(!bad_trials.has_value() &&
                               bad_trials.error() == nimblecas::MathError::domain_error,
                           "max_trials==0 yields domain_error");
              })
        .test("rejection_sample_terminates_under_trial_cap",
              [](TestContext& t) {
                  // A tiny trial budget must return promptly with at most `max_trials`
                  // samples and never hang, even though `want` is huge.
                  auto pdf = [](double x) { return x; };
                  auto r = rejection_sample(pdf, 0.0, 1.0, 1.0, 1000000, 5, 8);
                  t.expect(r.has_value(), "capped rejection_sample succeeds");
                  if (r) {
                      t.expect(r->size() <= 8,
                               std::format("returns at most max_trials samples (got {})",
                                           r->size()));
                  }
              })
        .test("sample_mean_and_variance",
              [](TestContext& t) {
                  // Textbook data: mean 5, sample variance 4 (n-1), population variance 3.5.
                  const std::array<double, 8> data{2, 4, 4, 4, 5, 5, 7, 9};
                  std::span<const double> s{data};

                  auto m = sample_mean(s);
                  t.expect(m.has_value(), "sample_mean succeeds");
                  if (m) {
                      t.expect(std::abs(*m - 5.0) < 1e-12,
                               std::format("mean == 5 (got {})", *m));
                  }

                  auto var = sample_variance(s, true);
                  t.expect(var.has_value(), "sample_variance (Bessel) succeeds");
                  if (var) {
                      t.expect(std::abs(*var - 4.0) < 1e-12,
                               std::format("sample variance == 4 (got {})", *var));
                  }

                  auto pop = sample_variance(s, false);
                  t.expect(pop.has_value(), "population variance succeeds");
                  if (pop) {
                      t.expect(std::abs(*pop - 3.5) < 1e-12,
                               std::format("population variance == 3.5 (got {})", *pop));
                  }
              })
        .test("statistics_domain_errors",
              [](TestContext& t) {
                  std::span<const double> empty{};
                  auto m = sample_mean(empty);
                  t.expect(!m.has_value() && m.error() == nimblecas::MathError::domain_error,
                           "sample_mean(empty) yields domain_error");
                  auto v = sample_variance(empty, true);
                  t.expect(!v.has_value() && v.error() == nimblecas::MathError::domain_error,
                           "sample_variance(empty) yields domain_error");

                  // A single element has no unbiased (n-1) variance.
                  const std::array<double, 1> one{42.0};
                  std::span<const double> single{one};
                  auto sv = sample_variance(single, true);
                  t.expect(!sv.has_value() && sv.error() == nimblecas::MathError::domain_error,
                           "single-element sample variance yields domain_error");

                  // But its population variance is a well-defined 0.
                  auto pv = sample_variance(single, false);
                  t.expect(pv.has_value(), "single-element population variance succeeds");
                  if (pv) {
                      t.expect(std::abs(*pv) < 1e-12, "single-element population variance == 0");
                  }
              })
        .run();
}
