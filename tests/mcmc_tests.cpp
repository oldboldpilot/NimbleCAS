// Tests for nimblecas.mcmc: random-walk Metropolis-Hastings sampling (ROADMAP 7.8).
// @author Olumuyiwa Oluwasanmi
//
// MCMC output is both stochastic and autocorrelated, so every statistical check uses a
// deterministic seed, a large sample budget, a healthy burn-in, and deliberately generous
// tolerances. The determinism and bookkeeping tests are exact.

import std;
import nimblecas.core;
import nimblecas.mcmc;
import nimblecas.testing;

using nimblecas::chain_mean;
using nimblecas::chain_variance;
using nimblecas::MathError;
using nimblecas::McmcResult;
using nimblecas::metropolis_hastings;
using nimblecas::run_parallel_chains;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Unnormalised log-density of the standard normal: log N(x; 0, 1) = −x²/2 + const.
[[nodiscard]] auto standard_normal_log(double x) -> double {
    return -0.5 * x * x;
}

// Unnormalised log-density of Uniform[0, 1]: 0 on the support, −inf elsewhere so any
// proposal outside [0, 1] is rejected.
[[nodiscard]] auto uniform01_log(double x) -> double {
    if (x < 0.0 || x > 1.0) {
        return -std::numeric_limits<double>::infinity();
    }
    return 0.0;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.mcmc")
        .test("standard_normal_moments",
              [](TestContext& t) {
                  // Target N(0, 1): mean ≈ 0, variance ≈ 1. Loose tolerances on purpose.
                  auto r = metropolis_hastings(standard_normal_log, 0.0, 2.0, 200000, 5000,
                                               2024);
                  t.expect(r.has_value(), "chain runs for the normal target");
                  if (r) {
                      t.expect(r->chain.size() == 200000, "retains exactly `samples` draws");
                      auto m = chain_mean(r->chain);
                      auto v = chain_variance(r->chain);
                      t.expect(m.has_value() && std::abs(m.value()) < 0.15,
                               "chain_mean ≈ 0 (tol 0.15)");
                      t.expect(v.has_value() && std::abs(v.value() - 1.0) < 0.3,
                               "chain_variance ≈ 1 (tol 0.3)");
                  }
              })
        .test("uniform01_support_and_mean",
              [](TestContext& t) {
                  // Target Uniform[0, 1]: every retained sample must lie in [0, 1] and the
                  // mean sits near 0.5.
                  auto r = metropolis_hastings(uniform01_log, 0.5, 0.5, 200000, 5000, 7);
                  t.expect(r.has_value(), "chain runs for the uniform target");
                  if (r) {
                      bool in_support = true;
                      for (const double x : r->chain) {
                          if (x < 0.0 || x > 1.0) {
                              in_support = false;
                              break;
                          }
                      }
                      t.expect(in_support, "all samples lie in [0, 1]");
                      auto m = chain_mean(r->chain);
                      t.expect(m.has_value() && std::abs(m.value() - 0.5) < 0.1,
                               "chain_mean ≈ 0.5 (tol 0.1)");
                  }
              })
        .test("determinism_bit_identical",
              [](TestContext& t) {
                  // Two identical calls must produce a bit-identical chain and counts.
                  auto a = metropolis_hastings(standard_normal_log, 0.0, 2.0, 5000, 1000, 99);
                  auto b = metropolis_hastings(standard_normal_log, 0.0, 2.0, 5000, 1000, 99);
                  t.expect(a.has_value() && b.has_value(), "both calls succeed");
                  if (a && b) {
                      t.expect(a->accepted == b->accepted && a->proposed == b->proposed,
                               "identical accept/propose counts");
                      bool same = a->chain.size() == b->chain.size();
                      for (std::size_t i = 0; same && i < a->chain.size(); ++i) {
                          // Bit-identical, not merely close: same seed => same draws.
                          if (a->chain[i] != b->chain[i]) {
                              same = false;
                          }
                      }
                      t.expect(same, "bit-identical chains for equal seeds");
                  }
              })
        .test("acceptance_bookkeeping",
              [](TestContext& t) {
                  const std::uint64_t samples = 20000;
                  const std::uint64_t burn = 2000;
                  auto r = metropolis_hastings(standard_normal_log, 0.0, 2.0, samples, burn,
                                               123);
                  t.expect(r.has_value(), "chain runs");
                  if (r) {
                      t.expect(r->proposed == burn + samples,
                               "proposed == burn_in + samples");
                      t.expect(r->accepted <= r->proposed, "accepted <= proposed");
                      const double rate = static_cast<double>(r->accepted) /
                                          static_cast<double>(r->proposed);
                      t.expect(rate > 0.0 && rate < 1.0,
                               "acceptance rate in (0, 1) for the normal target");
                  }
              })
        .test("parallel_chains_count_and_independence",
              [](TestContext& t) {
                  auto rs = run_parallel_chains(standard_normal_log, 0.0, 2.0, 5000, 1000, 42,
                                                4);
                  t.expect(rs.has_value(), "parallel chains run");
                  if (rs) {
                      t.expect(rs->size() == 4, "chains == 4 returns 4 results");
                      // Different chains use different seeds => not bit-identical.
                      const auto& c0 = (*rs)[0].chain;
                      const auto& c1 = (*rs)[1].chain;
                      bool differ = c0.size() != c1.size();
                      for (std::size_t i = 0; !differ && i < c0.size(); ++i) {
                          if (c0[i] != c1[i]) {
                              differ = true;
                          }
                      }
                      t.expect(differ, "distinct chains are not bit-identical");
                  }

                  // Rerunning the whole ensemble is reproducible.
                  auto again = run_parallel_chains(standard_normal_log, 0.0, 2.0, 5000, 1000,
                                                   42, 4);
                  t.expect(again.has_value(), "rerun succeeds");
                  if (rs && again) {
                      bool same = rs->size() == again->size();
                      for (std::size_t c = 0; same && c < rs->size(); ++c) {
                          const auto& x = (*rs)[c].chain;
                          const auto& y = (*again)[c].chain;
                          same = x.size() == y.size();
                          for (std::size_t i = 0; same && i < x.size(); ++i) {
                              if (x[i] != y[i]) {
                                  same = false;
                              }
                          }
                      }
                      t.expect(same, "parallel ensemble is reproducible across reruns");
                  }
              })
        .test("zero_support_walk_in_then_stay",
              [](TestContext& t) {
                  // Start OUTSIDE the support of Uniform[0, 1] (log-density −inf). The chain
                  // must walk back in — a finite proposal from an −inf point gives Δ = +inf,
                  // which the explicit `Δ >= 0` branch always accepts — and thereafter never
                  // leave, since any proposal back outside gives Δ = −inf and is always
                  // rejected (this is the branch the v == 0 / log(V) = −inf corner used to
                  // break). We assert the chain enters the support and its tail is entirely
                  // inside it. Deterministic seed, step small enough to reach [0,1] from 1.3.
                  auto r = metropolis_hastings(uniform01_log, 1.3, 0.5, 20000, 2000, 555);
                  t.expect(r.has_value(), "chain runs from outside the support");
                  if (r) {
                      // After burn-in the chain has long since walked in; the whole retained
                      // portion must lie in [0, 1] — no −inf-support sample survives.
                      bool tail_in_support = true;
                      for (const double x : r->chain) {
                          if (x < 0.0 || x > 1.0) {
                              tail_in_support = false;
                              break;
                          }
                      }
                      t.expect(tail_in_support, "retained chain lies entirely in [0, 1]");
                      t.expect(r->accepted > 0, "at least the walk-in move was accepted");
                  }
              })
        .test("zero_support_hold_when_unreachable",
              [](TestContext& t) {
                  // Start far outside the support with a step too small to ever reach it: every
                  // proposal has −inf density too, so Δ = −inf − −inf = NaN, which the explicit
                  // isnan branch rejects — the chain holds its place and NOTHING is accepted.
                  // This pins that a NaN Δ never spuriously accepts.
                  auto r = metropolis_hastings(uniform01_log, 10.0, 0.5, 500, 0, 3);
                  t.expect(r.has_value(), "chain runs while stuck outside the support");
                  if (r) {
                      t.expect(r->accepted == 0, "no move accepted when every point is −inf");
                      bool all_start = true;
                      for (const double x : r->chain) {
                          if (x != 10.0) {
                              all_start = false;
                              break;
                          }
                      }
                      t.expect(all_start, "chain holds exactly at the start (NaN Δ rejects)");
                  }
              })
        .test("domain_errors",
              [](TestContext& t) {
                  auto bad_step = metropolis_hastings(standard_normal_log, 0.0, 0.0, 100, 10,
                                                      1);
                  t.expect(!bad_step.has_value() &&
                               bad_step.error() == MathError::domain_error,
                           "step <= 0 => domain_error");

                  auto no_samples = metropolis_hastings(standard_normal_log, 0.0, 1.0, 0, 10,
                                                        1);
                  t.expect(!no_samples.has_value() &&
                               no_samples.error() == MathError::domain_error,
                           "samples == 0 => domain_error");

                  auto no_chains =
                      run_parallel_chains(standard_normal_log, 0.0, 1.0, 100, 10, 1, 0);
                  t.expect(!no_chains.has_value() &&
                               no_chains.error() == MathError::domain_error,
                           "chains == 0 => domain_error");

                  std::vector<double> empty;
                  auto em = chain_mean(empty);
                  auto ev = chain_variance(empty);
                  t.expect(!em.has_value() && em.error() == MathError::domain_error,
                           "chain_mean on empty => domain_error");
                  t.expect(!ev.has_value() && ev.error() == MathError::domain_error,
                           "chain_variance on empty => domain_error");
              })
        .run();
}
