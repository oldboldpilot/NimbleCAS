// Tests for nimblecas.sde: seeded Euler-Maruyama & Milstein SDE integrators, terminal-moment
// estimation, determinism, and the domain-error railway.
// @author Olumuyiwa Oluwasanmi
//
// Everything here is deterministic (fixed seeds, no time/entropy) but stochastic in value, so
// ensemble-average checks use GENEROUS tolerances chosen never to flake. Exact-value checks are
// reserved for the deterministic-limit (zero diffusion), scheme-agreement, reproducibility and
// domain-error cases, where the results are truly bit-defined.

import std;
import nimblecas.core;
import nimblecas.rng;
import nimblecas.sde;
import nimblecas.testing;

using nimblecas::euler_maruyama;
using nimblecas::milstein;
using nimblecas::simulate_terminal;
using nimblecas::terminal_moments;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

auto main() -> int {
    return TestSuite("nimblecas.sde")
        .test("gbm_terminal_mean_matches_analytic_euler",
              [](TestContext& t) {
                  // Geometric Brownian motion dX = μX dt + σX dW has E[X_T] = x0·exp(μT).
                  // For μ=0.1, σ=0.2, x0=1, T=1 that is exp(0.1) ≈ 1.10517. Average X_T over
                  // many seeded Euler-Maruyama paths and check the sample mean is close.
                  const double mu = 0.1;
                  const double sigma = 0.2;
                  auto a = [mu](double x) { return mu * x; };
                  auto b = [sigma](double x) { return sigma * x; };

                  auto mom = terminal_moments(a, b, /*b_prime=*/{}, 1.0, 1.0, 200, 40000, 20260703,
                                              /*use_milstein=*/false);
                  t.expect(mom.has_value(), "terminal_moments (Euler, GBM) succeeds");
                  if (mom) {
                      const double expected = std::exp(mu * 1.0);
                      t.expect(std::abs(mom->first - expected) < 0.01,
                               std::format("E[X_T] ~ exp(0.1) = {} (got {})", expected, mom->first));
                      t.expect(mom->second > 0.0,
                               std::format("terminal variance is positive (got {})", mom->second));
                  }
              })
        .test("gbm_terminal_mean_matches_analytic_milstein",
              [](TestContext& t) {
                  // Same GBM target, Milstein scheme (b'(x) = σ). E[X_T] is unchanged at exp(μT);
                  // Milstein only reduces path-wise discretisation error, not the true mean.
                  const double mu = 0.1;
                  const double sigma = 0.2;
                  auto a = [mu](double x) { return mu * x; };
                  auto b = [sigma](double x) { return sigma * x; };
                  auto bp = [sigma](double) { return sigma; };

                  auto mom = terminal_moments(a, b, bp, 1.0, 1.0, 200, 40000, 777,
                                              /*use_milstein=*/true);
                  t.expect(mom.has_value(), "terminal_moments (Milstein, GBM) succeeds");
                  if (mom) {
                      const double expected = std::exp(mu * 1.0);
                      t.expect(std::abs(mom->first - expected) < 0.01,
                               std::format("Milstein E[X_T] ~ {} (got {})", expected, mom->first));
                  }
              })
        .test("zero_diffusion_reduces_to_deterministic_euler",
              [](TestContext& t) {
                  // With b ≡ 0 the SDE collapses to the ODE dX = a(X) dt. For a(x)=x, X(0)=1,
                  // T=1 the exact solution is X_T = e ≈ 2.71828; explicit Euler with a finite
                  // step undershoots, so use a loose tolerance reflecting the discretisation
                  // error rather than sampling noise (there is none once b = 0).
                  auto a = [](double x) { return x; };
                  auto zero = [](double) { return 0.0; };

                  auto r = euler_maruyama(a, zero, 1.0, 1.0, 2000, 123);
                  t.expect(r.has_value(), "euler_maruyama (b=0, a=x) succeeds");
                  if (r) {
                      t.expect(r->times.size() == 2001 && r->values.size() == 2001,
                               std::format("path has steps+1 nodes (got {} times)", r->times.size()));
                      t.expect(r->times.front() == 0.0, "path starts at t=0");
                      t.expect(r->times.back() == 1.0, "path ends exactly at t=T");
                      t.expect(r->values.front() == 1.0, "path starts at x0");
                      t.expect(std::abs(r->values.back() - std::numbers::e) < 0.01,
                               std::format("X_T ~ e = {} (got {})", std::numbers::e,
                                           r->values.back()));
                  }
              })
        .test("euler_equals_milstein_when_diffusion_is_zero",
              [](TestContext& t) {
                  // When b ≡ 0 the Milstein correction ½ b b' (dW²−dt) vanishes identically, so
                  // the two schemes must produce BIT-IDENTICAL paths from the same seed (the
                  // Brownian increments are drawn the same way and are multiplied by b = 0).
                  auto a = [](double x) { return 0.5 * x + 1.0; };
                  auto zero = [](double) { return 0.0; };
                  auto zero_prime = [](double) { return 0.0; };

                  auto e = euler_maruyama(a, zero, 2.0, 1.0, 500, 555);
                  auto m = milstein(a, zero, zero_prime, 2.0, 1.0, 500, 555);
                  t.expect(e.has_value() && m.has_value(), "both schemes succeed with b=0");
                  if (e && m) {
                      bool identical = e->values.size() == m->values.size();
                      for (std::size_t i = 0; i < e->values.size() && i < m->values.size(); ++i) {
                          if (e->values[i] != m->values[i]) identical = false;
                      }
                      t.expect(identical, "Euler == Milstein bit-for-bit when b = 0");
                  }
              })
        .test("same_seed_gives_bit_identical_path",
              [](TestContext& t) {
                  // Determinism: identical arguments (including seed) must reproduce a
                  // bit-identical path — the integrator has no hidden state.
                  auto a = [](double x) { return 0.1 * x; };
                  auto b = [](double x) { return 0.2 * x; };
                  auto bp = [](double) { return 0.2; };

                  auto p1 = milstein(a, b, bp, 1.0, 1.0, 300, 2024);
                  auto p2 = milstein(a, b, bp, 1.0, 1.0, 300, 2024);
                  t.expect(p1.has_value() && p2.has_value(), "both milstein() calls succeed");
                  if (p1 && p2) {
                      bool same = p1->values.size() == p2->values.size();
                      for (std::size_t i = 0; i < p1->values.size(); ++i) {
                          if (p1->values[i] != p2->values[i]) same = false;
                      }
                      t.expect(same, "same seed => bit-identical Milstein path");

                      // A different seed must actually change the path (not a constant).
                      auto p3 = milstein(a, b, bp, 1.0, 1.0, 300, 2025);
                      t.expect(p3.has_value(), "milstein() with a different seed succeeds");
                      if (p3) {
                          t.expect(p3->values.back() != p1->values.back(),
                                   "a different seed yields a different terminal value");
                      }
                  }
              })
        .test("simulate_terminal_is_reproducible_and_paths_differ",
              [](TestContext& t) {
                  // The whole ensemble is a pure function of (seed): rerunning gives a
                  // bit-identical vector, while distinct path indices give distinct terminals.
                  auto a = [](double x) { return 0.1 * x; };
                  auto b = [](double x) { return 0.2 * x; };

                  auto r1 = simulate_terminal(a, b, {}, 1.0, 1.0, 100, 64, 909, false);
                  auto r2 = simulate_terminal(a, b, {}, 1.0, 1.0, 100, 64, 909, false);
                  t.expect(r1.has_value() && r2.has_value(), "both simulate_terminal() runs succeed");
                  if (r1 && r2) {
                      bool same = r1->size() == r2->size() && r1->size() == 64;
                      for (std::size_t i = 0; i < r1->size() && i < r2->size(); ++i) {
                          if ((*r1)[i] != (*r2)[i]) same = false;
                      }
                      t.expect(same, "simulate_terminal reruns bit-identically (reproducible ensemble)");

                      // Independent per-path seeding => the paths are not all the same value.
                      bool any_different = false;
                      for (std::size_t i = 1; i < r1->size(); ++i) {
                          if ((*r1)[i] != (*r1)[0]) any_different = true;
                      }
                      t.expect(any_different, "distinct path indices produce distinct terminals");
                  }
              })
        .test("domain_errors",
              [](TestContext& t) {
                  auto a = [](double x) { return x; };
                  auto b = [](double x) { return x; };
                  auto bp = [](double) { return 1.0; };

                  auto s0 = euler_maruyama(a, b, 1.0, 1.0, 0, 1);
                  t.expect(!s0.has_value() && s0.error() == nimblecas::MathError::domain_error,
                           "euler_maruyama(steps==0) yields domain_error");

                  auto t0 = euler_maruyama(a, b, 1.0, 0.0, 10, 1);
                  t.expect(!t0.has_value() && t0.error() == nimblecas::MathError::domain_error,
                           "euler_maruyama(T==0) yields domain_error");

                  auto tneg = milstein(a, b, bp, 1.0, -1.0, 10, 1);
                  t.expect(!tneg.has_value() && tneg.error() == nimblecas::MathError::domain_error,
                           "milstein(T<0) yields domain_error");

                  auto ms0 = milstein(a, b, bp, 1.0, 1.0, 0, 1);
                  t.expect(!ms0.has_value() && ms0.error() == nimblecas::MathError::domain_error,
                           "milstein(steps==0) yields domain_error");

                  auto p0 = simulate_terminal(a, b, {}, 1.0, 1.0, 10, 0, 1, false);
                  t.expect(!p0.has_value() && p0.error() == nimblecas::MathError::domain_error,
                           "simulate_terminal(paths==0) yields domain_error");

                  auto tm0 = terminal_moments(a, b, {}, 1.0, 1.0, 10, 0, 1, false);
                  t.expect(!tm0.has_value() && tm0.error() == nimblecas::MathError::domain_error,
                           "terminal_moments(paths==0) yields domain_error");

                  // Milstein requires a non-empty b'(x); requesting the Milstein scheme without
                  // one is a domain error rather than a std::bad_function_call thrown off-railway.
                  auto no_bp = simulate_terminal(a, b, {}, 1.0, 1.0, 10, 5, 1, /*use_milstein=*/true);
                  t.expect(!no_bp.has_value() && no_bp.error() == nimblecas::MathError::domain_error,
                           "Milstein without b_prime yields domain_error");

                  // An empty drift/diffusion functor is likewise rejected on the railway.
                  auto empty_fn = euler_maruyama({}, b, 1.0, 1.0, 10, 1);
                  t.expect(!empty_fn.has_value() &&
                               empty_fn.error() == nimblecas::MathError::domain_error,
                           "euler_maruyama(empty drift) yields domain_error");
              })
        .run();
}
