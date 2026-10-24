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
using nimblecas::estimate_pi_parallel;
using nimblecas::integrate;
using nimblecas::integrate_parallel;
using nimblecas::montecarlo_block;
using nimblecas::rejection_sample;
using nimblecas::sample_mean;
using nimblecas::sample_variance;
using nimblecas::splitmix64;
using nimblecas::uniform_unit;
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
                  const auto f = [](double x) { return x * x * x; };
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
                  // Materialise the whole sequence, then generate it again and compare the two
                  // runs. Comparing counter_u64(key, i) against counter_u64(key, i) inline is a
                  // tautology: both sides are the same call on a pure function, which the
                  // optimiser may fold to `true` without ever running the generator.
                  std::vector<std::uint64_t> first;
                  first.reserve(256);
                  for (std::uint64_t i = 0; i < 256; ++i) {
                      first.push_back(counter_u64(key, i));
                  }
                  bool det = true;
                  for (std::uint64_t i = 0; i < 256; ++i) {
                      if (counter_u64(key, i) != first[i]) {
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
        .test("integrate_parallel_is_bit_identical_to_serial",
              [](TestContext& t) {
                  // Sample counts straddling every block edge: inside one block, exactly one
                  // block, one past it, and a ragged multi-block tail.
                  const auto f = [](double x) { return std::sin(x) * std::exp(-x * x); };
                  const std::array<std::uint64_t, 6> counts{
                      1, montecarlo_block - 1, montecarlo_block, montecarlo_block + 1,
                      5 * montecarlo_block + 12345, 1000003};
                  for (const std::uint64_t n : counts) {
                      auto s = integrate(f, -1.5, 2.5, n, 4242);
                      auto p = integrate_parallel(f, -1.5, 2.5, n, 4242);
                      t.expect(s.has_value() && p.has_value(),
                               std::format("integrate and integrate_parallel succeed (n={})", n));
                      if (s && p) {
                          t.expect(std::bit_cast<std::uint64_t>(*s) ==
                                       std::bit_cast<std::uint64_t>(*p),
                                   std::format("integrate_parallel == integrate bit-for-bit "
                                               "(n={}, serial {}, parallel {})",
                                               n, *s, *p));
                      }
                  }
                  auto bad = integrate_parallel([](double x) { return x; }, 1.0, 0.0, 10, 1);
                  t.expect(!bad.has_value() && bad.error() == nimblecas::MathError::domain_error,
                           "integrate_parallel(b < a) yields domain_error");
                  const double nan = std::numeric_limits<double>::quiet_NaN();
                  for (const auto& [lo, hi] : {std::pair{nan, 1.0}, std::pair{0.0, nan}}) {
                      auto rs = integrate([](double x) { return x; }, lo, hi, 10, 1);
                      auto rp = integrate_parallel([](double x) { return x; }, lo, hi, 10, 1);
                      t.expect(!rs.has_value() && rs.error() == nimblecas::MathError::domain_error &&
                                   !rp.has_value() &&
                                   rp.error() == nimblecas::MathError::domain_error,
                               "a NaN bound yields domain_error, not a NaN estimate");
                  }
                  // A single sample is one partial block. (The block-count wrap near UINT64_MAX
                  // that the quotient+remainder form fixes is not testable: a correct run at that
                  // size never finishes. It is covered by review, not by this suite.)
                  auto one = integrate_parallel([](double) { return 2.0; }, 0.0, 1.0, 1, 3);
                  t.expect(one.has_value() && *one == 2.0, "one sample is one partial block");
                  auto zero = integrate_parallel([](double x) { return x; }, 0.0, 1.0, 0, 1);
                  t.expect(!zero.has_value() && zero.error() == nimblecas::MathError::domain_error,
                           "integrate_parallel(samples == 0) yields domain_error");
                  auto conv = integrate_parallel([](double x) { return x * x; }, 0.0, 1.0,
                                                 400000, 20260703);
                  t.expect(conv.has_value() && std::abs(*conv - (1.0 / 3.0)) < 0.01,
                           "integrate_parallel(x^2, [0,1]) ~ 1/3");
              })
        .test("integrate_uses_the_documented_block_grouping",
              [](TestContext& t) {
                  // Recompute the estimate by hand with the documented grouping -- each block
                  // summed in index order, partials summed in block order -- and require the
                  // library to match it bit-for-bit. Without this pin, serial and parallel could
                  // drift to some other grouping together and the test above would still pass.
                  //
                  // 64 full blocks plus a 3-sample tail: many partials of very different size
                  // make the sum order-sensitive. A 4-block input was not (summing its partials
                  // in reverse gave the same bits, so a reversed-combine mutant survived).
                  const auto f = [](double x) { return 1.0 / (1.0 + x * x); };
                  constexpr std::uint64_t n = 64 * montecarlo_block + 3;
                  const double a = 0.25;
                  const double width = 3.0;
                  const std::uint64_t key = splitmix64(77);
                  std::vector<double> partials;
                  for (std::uint64_t begin = 0; begin < n; begin += montecarlo_block) {
                      const std::uint64_t end = std::min(n, begin + montecarlo_block);
                      double block = 0.0;
                      for (std::uint64_t i = begin; i < end; ++i) {
                          block += f(a + uniform_unit(counter_u64(key, i)) * width);
                      }
                      partials.push_back(block);
                  }
                  const auto estimate = [&](double total) {
                      return width * (total / static_cast<double>(n));
                  };
                  double forward = 0.0;
                  for (const double p : partials) forward += p;
                  double reversed = 0.0;
                  for (auto it = partials.rbegin(); it != partials.rend(); ++it) reversed += *it;
                  const double expected = estimate(forward);

                  // Guard the guard: if this input were order-insensitive, the pin below could
                  // not fail, and passing it would prove nothing.
                  t.expect(std::bit_cast<std::uint64_t>(estimate(reversed)) !=
                               std::bit_cast<std::uint64_t>(expected),
                           "this input is order-sensitive (block order changes the last bits)");

                  auto r = integrate(f, a, a + width, n, 77);
                  t.expect(r.has_value() && std::bit_cast<std::uint64_t>(*r) ==
                                                std::bit_cast<std::uint64_t>(expected),
                           "integrate matches the hand-blocked sum bit-for-bit");
                  auto rp = integrate_parallel(f, a, a + width, n, 77);
                  t.expect(rp.has_value() && std::bit_cast<std::uint64_t>(*rp) ==
                                                 std::bit_cast<std::uint64_t>(expected),
                           "integrate_parallel matches the hand-blocked sum bit-for-bit");
              })
        .test("parallel_rounds_beyond_the_first_are_summed",
              [](TestContext& t) {
                  // The parallel entry points walk blocks in rounds of 4096. With a constant
                  // integrand every block partial is its exact sample count, so the estimate is
                  // exactly the width -- unless a block is dropped, repeated or mis-indexed.
                  // 4096 full blocks + 5 samples puts a 5-sample block alone in round two; a
                  // round-two block indexed from 0 instead of 4096 would add 65536, not 5.
                  constexpr std::uint64_t n = 4096 * montecarlo_block + 5;
                  auto r = integrate_parallel([](double) { return 1.0; }, 0.0, 3.0, n, 8);
                  t.expect(r.has_value() && *r == 3.0,
                           std::format("multi-round integrate_parallel of 1 over [0,3] == 3 "
                                       "(got {})", r ? *r : -1.0));
              })
        .test("estimate_pi_reads_counters_2i_and_2i_plus_1",
              [](TestContext& t) {
                  // Pin estimate_pi's draws independently of pi_block, so serial and parallel
                  // cannot share a wrong pairing (e.g. i and i+1, which correlates x and y).
                  constexpr std::uint64_t n = 70000;  // two blocks
                  const std::uint64_t key = splitmix64(31);
                  std::uint64_t hits = 0;
                  for (std::uint64_t i = 0; i < n; ++i) {
                      const double x = uniform_unit(counter_u64(key, 2 * i));
                      const double y = uniform_unit(counter_u64(key, 2 * i + 1));
                      if (x * x + y * y < 1.0) ++hits;
                  }
                  const double expected = 4.0 * static_cast<double>(hits) / static_cast<double>(n);
                  auto r = estimate_pi(n, 31);
                  t.expect(r.has_value() && *r == expected,
                           "estimate_pi matches a hand count over counters 2i, 2i+1");
                  auto over = estimate_pi_parallel((std::uint64_t{1} << 63U) + 1, 1);
                  t.expect(!over.has_value() && over.error() == nimblecas::MathError::domain_error,
                           "estimate_pi beyond 2^63 samples is a domain_error, not a wrapped replay");
              })
        .test("guards_reject_empty_functions_and_non_finite_input",
              [](TestContext& t) {
                  using nimblecas::MathError;
                  const std::function<double(double)> empty;
                  const double inf = std::numeric_limits<double>::infinity();
                  const double nan = std::numeric_limits<double>::quiet_NaN();
                  auto e1 = integrate_parallel(empty, 0.0, 1.0, 10, 1);
                  t.expect(!e1 && e1.error() == MathError::domain_error,
                           "integrate_parallel(empty f) is a domain_error, not a throw");
                  auto e2 = rejection_sample(empty, 0.0, 1.0, 1.0, 1, 1, 1);
                  t.expect(!e2 && e2.error() == MathError::domain_error,
                           "rejection_sample(empty pdf) is a domain_error, not a throw");
                  auto i1 = integrate([](double) { return 1.0; }, inf, inf, 10, 1);
                  t.expect(!i1 && i1.error() == MathError::domain_error,
                           "integrate over [inf, inf] is a domain_error, not NaN");
                  auto i2 = integrate_parallel([](double) { return 1.0; }, -1e308, 1e308, 10, 1);
                  t.expect(!i2 && i2.error() == MathError::overflow,
                           "integrate over [-1e308, 1e308] reports overflow of b - a");
                  auto eq = integrate_parallel([](double) { return 5.0; }, 2.0, 2.0, 10, 1);
                  t.expect(eq.has_value() && *eq == 0.0, "b == a integrates to 0");
                  auto r1 = rejection_sample([](double) { return 1.0; }, nan, 1.0, 1.0, 4, 1, 10);
                  t.expect(!r1 && r1.error() == MathError::domain_error,
                           "rejection_sample with a NaN bound is a domain_error");
                  auto r2 = rejection_sample([](double x) { return x; }, 0.0, 1.0, nan, 4, 1, 10);
                  t.expect(!r2 && r2.error() == MathError::domain_error,
                           "rejection_sample with a NaN ceiling is a domain_error");
                  // Huge want, small trial budget, nothing accepted: an empty success, and no
                  // up-front reserve of `want`.
                  constexpr auto big = std::numeric_limits<std::uint64_t>::max();
                  auto r3 = rejection_sample([](double) { return 0.0; }, 0.0, 1.0, 1.0, big, 1,
                                             1000);
                  t.expect(r3.has_value() && r3->empty(),
                           "huge want with a small trial budget returns an empty success");
              })
        .test("estimate_pi_parallel_is_bit_identical_to_serial",
              [](TestContext& t) {
                  const std::array<std::uint64_t, 5> counts{
                      1, montecarlo_block, montecarlo_block + 1, 7 * montecarlo_block + 3, 900001};
                  for (const std::uint64_t n : counts) {
                      auto s = estimate_pi(n, 12345);
                      auto p = estimate_pi_parallel(n, 12345);
                      t.expect(s.has_value() && p.has_value() && *s == *p,
                               std::format("estimate_pi_parallel == estimate_pi (n={})", n));
                  }
                  auto zero = estimate_pi_parallel(0, 1);
                  t.expect(!zero.has_value() && zero.error() == nimblecas::MathError::domain_error,
                           "estimate_pi_parallel(0) yields domain_error");
                  auto big = estimate_pi_parallel(2000000, 99);
                  t.expect(big.has_value() && std::abs(*big - std::numbers::pi) < 0.01,
                           "estimate_pi_parallel(2e6) ~ pi");
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
                  const auto pdf = [](double x) { return x; };
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
                  const auto pdf = [](double x) { return x; };

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
                  const auto pdf = [](double x) { return x; };
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
                  // Textbook data with mean 5 and sum of squared deviations 32 (the classic
                  // "standard deviation 2" set): population variance 32/8 = 4, and the
                  // Bessel-corrected sample variance 32/7 = 4.5714285...
                  const std::array<double, 8> data{2, 4, 4, 4, 5, 5, 7, 9};
                  const std::span<const double> s{data};

                  auto m = sample_mean(s);
                  t.expect(m.has_value(), "sample_mean succeeds");
                  if (m) {
                      t.expect(std::abs(*m - 5.0) < 1e-12,
                               std::format("mean == 5 (got {})", *m));
                  }

                  auto var = sample_variance(s, true);
                  t.expect(var.has_value(), "sample_variance (Bessel) succeeds");
                  if (var) {
                      t.expect(std::abs(*var - 32.0 / 7.0) < 1e-12,
                               std::format("sample variance == 32/7 (got {})", *var));
                  }

                  auto pop = sample_variance(s, false);
                  t.expect(pop.has_value(), "population variance succeeds");
                  if (pop) {
                      t.expect(std::abs(*pop - 4.0) < 1e-12,
                               std::format("population variance == 4 (got {})", *pop));
                  }
              })
        .test("statistics_domain_errors",
              [](TestContext& t) {
                  const std::span<const double> empty{};
                  auto m = sample_mean(empty);
                  t.expect(!m.has_value() && m.error() == nimblecas::MathError::domain_error,
                           "sample_mean(empty) yields domain_error");
                  auto v = sample_variance(empty, true);
                  t.expect(!v.has_value() && v.error() == nimblecas::MathError::domain_error,
                           "sample_variance(empty) yields domain_error");

                  // A single element has no unbiased (n-1) variance.
                  const std::array<double, 1> one{42.0};
                  const std::span<const double> single{one};
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
