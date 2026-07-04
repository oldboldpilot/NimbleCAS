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
// New schemes and generic drivers under test.
using nimblecas::Scheme;
using nimblecas::simulate_terminal_scheme;
using nimblecas::splitmix64;
using nimblecas::srk;
using nimblecas::stochastic_heun;
using nimblecas::tamed_euler;
using nimblecas::terminal_moments_scheme;
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

                  // A non-finite T or x0 must be rejected, not silently produce an all-NaN
                  // "success" path (T <= 0 alone is false for NaN).
                  const double nan = std::numeric_limits<double>::quiet_NaN();
                  auto nan_t = euler_maruyama(a, b, 1.0, nan, 10, 1);
                  t.expect(!nan_t.has_value() &&
                               nan_t.error() == nimblecas::MathError::domain_error,
                           "euler_maruyama(T==NaN) yields domain_error");
                  auto nan_x0 = milstein(a, b, bp, nan, 1.0, 10, 1);
                  t.expect(!nan_x0.has_value() &&
                               nan_x0.error() == nimblecas::MathError::domain_error,
                           "milstein(x0==NaN) yields domain_error");
              })
        // ---------------------------------------------------------------------------
        // New schemes: Stochastic Heun (Stratonovich), derivative-free SRK (Itô),
        // Tamed Euler (Itô, stiff-stable), and their generic ensemble drivers.
        // ---------------------------------------------------------------------------
        .test("heun_gbm_converges_to_stratonovich_mean",
              [](TestContext& t) {
                  // Stochastic Heun re-uses dW in predictor and corrector, so for GBM a=μx, b=σx
                  // it converges to the STRATONOVICH solution, whose mean is x0·exp((μ+½σ²)T),
                  // NOT the Itô mean exp(μT). μ=0.1, σ=0.2, T=1 ⇒ exp(0.12) ≈ 1.12750. This is a
                  // deliberate, documented convention difference, not a bug.
                  const double mu = 0.1;
                  const double sigma = 0.2;
                  auto a = [mu](double x) { return mu * x; };
                  auto b = [sigma](double x) { return sigma * x; };

                  auto mom = terminal_moments_scheme(a, b, /*b_prime=*/{}, 1.0, 1.0, 200, 40000,
                                                     20260703, Scheme::stochastic_heun);
                  t.expect(mom.has_value(), "terminal_moments_scheme (Heun, GBM) succeeds");
                  if (mom) {
                      const double strat = std::exp((mu + 0.5 * sigma * sigma) * 1.0);
                      t.expect(std::abs(mom->first - strat) < 0.02,
                               std::format("Heun E[X_T] ~ exp(0.12) = {} (got {})", strat,
                                           mom->first));
                      // Sanity: it must NOT sit at the Itô mean exp(0.1) — the schemes really do
                      // integrate different SDEs.
                      t.expect(std::abs(mom->first - std::exp(mu)) > 0.005,
                               "Heun mean is the Stratonovich, not the Itô, value");
                  }
              })
        .test("srk_gbm_converges_to_ito_mean",
              [](TestContext& t) {
                  // The derivative-free SRK is an Itô scheme (strong order 1), so on GBM its mean
                  // matches Euler/Milstein at x0·exp(μT) = exp(0.1) ≈ 1.10517.
                  const double mu = 0.1;
                  const double sigma = 0.2;
                  auto a = [mu](double x) { return mu * x; };
                  auto b = [sigma](double x) { return sigma * x; };

                  auto mom = terminal_moments_scheme(a, b, /*b_prime=*/{}, 1.0, 1.0, 200, 40000, 4242,
                                                     Scheme::srk);
                  t.expect(mom.has_value(), "terminal_moments_scheme (SRK, GBM) succeeds");
                  if (mom) {
                      const double expected = std::exp(mu * 1.0);
                      t.expect(std::abs(mom->first - expected) < 0.02,
                               std::format("SRK E[X_T] ~ exp(0.1) = {} (got {})", expected,
                                           mom->first));
                      t.expect(mom->second > 0.0,
                               std::format("SRK terminal variance is positive (got {})",
                                           mom->second));
                  }
              })
        .test("tamed_euler_gbm_converges_to_ito_mean",
              [](TestContext& t) {
                  // Tamed Euler is an Itô scheme; on (benign, globally Lipschitz) GBM the O(dt)
                  // taming perturbation is tiny for small dt, so the mean still matches exp(μT).
                  const double mu = 0.1;
                  const double sigma = 0.2;
                  auto a = [mu](double x) { return mu * x; };
                  auto b = [sigma](double x) { return sigma * x; };

                  auto mom = terminal_moments_scheme(a, b, /*b_prime=*/{}, 1.0, 1.0, 400, 40000, 909091,
                                                     Scheme::tamed_euler);
                  t.expect(mom.has_value(), "terminal_moments_scheme (Tamed, GBM) succeeds");
                  if (mom) {
                      const double expected = std::exp(mu * 1.0);
                      t.expect(std::abs(mom->first - expected) < 0.02,
                               std::format("Tamed E[X_T] ~ exp(0.1) = {} (got {})", expected,
                                           mom->first));
                  }
              })
        .test("srk_equals_euler_when_diffusion_is_zero",
              [](TestContext& t) {
                  // With b ≡ 0 the SRK finite-difference correction (b(Ŷ)−b) is identically 0 and
                  // the diffusion term vanishes, so SRK must equal Euler-Maruyama BIT-FOR-BIT from
                  // the same seed (both consume the same one-normal-per-step Brownian stream).
                  auto a = [](double x) { return 0.5 * x + 1.0; };
                  auto zero = [](double) { return 0.0; };

                  auto e = euler_maruyama(a, zero, 2.0, 1.0, 500, 555);
                  auto s = srk(a, zero, 2.0, 1.0, 500, 555);
                  t.expect(e.has_value() && s.has_value(), "both schemes succeed with b=0");
                  if (e && s) {
                      bool identical = e->values.size() == s->values.size();
                      for (std::size_t i = 0; i < e->values.size() && i < s->values.size(); ++i) {
                          if (e->values[i] != s->values[i]) identical = false;
                      }
                      t.expect(identical, "SRK == Euler-Maruyama bit-for-bit when b = 0");
                  }
              })
        .test("new_schemes_are_reproducible_and_seed_sensitive",
              [](TestContext& t) {
                  // Determinism: identical (args, seed) reproduce a bit-identical path for each new
                  // scheme; a different seed must change the terminal value (not a constant).
                  auto a = [](double x) { return 0.1 * x; };
                  auto b = [](double x) { return 0.2 * x; };

                  const auto check = [&](const char* name, auto&& solver) {
                      auto p1 = solver(a, b, 1.0, 1.0, 300, 2024);
                      auto p2 = solver(a, b, 1.0, 1.0, 300, 2024);
                      t.expect(p1.has_value() && p2.has_value(),
                               std::format("{}: both calls succeed", name));
                      if (p1 && p2) {
                          bool same = p1->values.size() == p2->values.size();
                          for (std::size_t i = 0; i < p1->values.size(); ++i) {
                              if (p1->values[i] != p2->values[i]) same = false;
                          }
                          t.expect(same, std::format("{}: same seed => bit-identical path", name));
                          auto p3 = solver(a, b, 1.0, 1.0, 300, 2025);
                          t.expect(p3.has_value() && p3->values.back() != p1->values.back(),
                                   std::format("{}: a different seed changes the terminal", name));
                      }
                  };
                  check("heun", stochastic_heun);
                  check("srk", srk);
                  check("tamed_euler", tamed_euler);
              })
        .test("tamed_euler_stays_finite_where_euler_blows_up",
              [](TestContext& t) {
                  // Stiff/superlinear drift a(x) = −x³ (one-sided Lipschitz, NOT globally
                  // Lipschitz). Zero diffusion makes the comparison crisp: the true solution decays
                  // monotonically to 0, yet explicit Euler with a coarse step overshoots, cubes the
                  // overshoot, and diverges to a non-finite value within a handful of steps. Tamed
                  // Euler caps each drift increment below 1 in magnitude and stays finite/bounded.
                  auto drift = [](double x) { return -x * x * x; };
                  auto zero = [](double) { return 0.0; };

                  auto e = euler_maruyama(drift, zero, 5.0, 1.0, 10, 31337);
                  auto tm = tamed_euler(drift, zero, 5.0, 1.0, 10, 31337);
                  t.expect(e.has_value() && tm.has_value(), "both integrators return a path");
                  if (e && tm) {
                      t.expect(!std::isfinite(e->values.back()),
                               std::format("plain Euler diverges (X_T = {})", e->values.back()));
                      const double xt = tm->values.back();
                      t.expect(std::isfinite(xt) && std::abs(xt) <= 5.0,
                               std::format("tamed Euler stays finite and bounded (X_T = {})", xt));
                  }
              })
        .test("scheme_driver_is_partition_independent",
              [](TestContext& t) {
                  // The ensemble driver's element p must be a PURE function of (seed, p): equal to
                  // the single-path solver run on the same per-path seed splitmix64(seed ^ p). If
                  // every element reconstructs independently, then any split of 0..paths-1 across
                  // any number of workers, reassembled in index order, reproduces the vector
                  // bit-for-bit — i.e. the result is thread-count / partition independent.
                  auto a = [](double x) { return 0.15 * x; };
                  auto b = [](double x) { return 0.25 * x; };
                  const std::uint64_t seed = 20260703;
                  const std::uint64_t paths = 48;

                  auto full = simulate_terminal_scheme(a, b, {}, 1.0, 1.0, 120, paths, seed,
                                                       Scheme::srk);
                  t.expect(full.has_value() && full->size() == paths,
                           "simulate_terminal_scheme (SRK) returns `paths` terminals");
                  if (full) {
                      bool matches = true;
                      for (std::uint64_t p = 0; p < paths; ++p) {
                          auto one = srk(a, b, 1.0, 1.0, 120, splitmix64(seed ^ p));
                          if (!one || one->values.back() != (*full)[p]) matches = false;
                      }
                      t.expect(matches,
                               "each ensemble element == independent single-path run on its seed");

                      // Rerun reproducibility (no hidden state).
                      auto again = simulate_terminal_scheme(a, b, {}, 1.0, 1.0, 120, paths, seed,
                                                            Scheme::srk);
                      bool same = again.has_value() && again->size() == full->size();
                      for (std::size_t i = 0; same && i < full->size(); ++i) {
                          if ((*again)[i] != (*full)[i]) same = false;
                      }
                      t.expect(same, "simulate_terminal_scheme reruns bit-identically");
                  }
              })
        .test("scheme_driver_domain_errors",
              [](TestContext& t) {
                  auto a = [](double x) { return x; };
                  auto b = [](double x) { return x; };

                  // Milstein via the generic driver still requires a non-empty b'(x).
                  auto no_bp = simulate_terminal_scheme(a, b, {}, 1.0, 1.0, 10, 5, 1, Scheme::milstein);
                  t.expect(!no_bp.has_value() &&
                               no_bp.error() == nimblecas::MathError::domain_error,
                           "Milstein driver without b_prime yields domain_error");

                  // Derivative-free schemes never need b'; an empty b_prime is fine for them.
                  auto ok = simulate_terminal_scheme(a, b, {}, 1.0, 1.0, 10, 5, 1, Scheme::srk);
                  t.expect(ok.has_value(), "SRK driver succeeds with empty b_prime");

                  auto p0 = simulate_terminal_scheme(a, b, {}, 1.0, 1.0, 10, 0, 1, Scheme::tamed_euler);
                  t.expect(!p0.has_value() && p0.error() == nimblecas::MathError::domain_error,
                           "scheme driver (paths==0) yields domain_error");

                  const double nan = std::numeric_limits<double>::quiet_NaN();
                  auto nan_t = stochastic_heun(a, b, 1.0, nan, 10, 1);
                  t.expect(!nan_t.has_value() &&
                               nan_t.error() == nimblecas::MathError::domain_error,
                           "stochastic_heun(T==NaN) yields domain_error");
                  auto s0 = srk(a, b, 1.0, 1.0, 0, 1);
                  t.expect(!s0.has_value() && s0.error() == nimblecas::MathError::domain_error,
                           "srk(steps==0) yields domain_error");
                  auto empty_fn = tamed_euler({}, b, 1.0, 1.0, 10, 1);
                  t.expect(!empty_fn.has_value() &&
                               empty_fn.error() == nimblecas::MathError::domain_error,
                           "tamed_euler(empty drift) yields domain_error");
              })
        .test("scheme_driver_matches_legacy_bit_for_bit",
              [](TestContext& t) {
                  // The Scheme-parameterised driver must reproduce the legacy bool-selected
                  // driver BIT-FOR-BIT for euler_maruyama and milstein, locking the
                  // byte-identical guarantee against future refactors. Nonlinear a/b so the
                  // trajectories are non-trivial (not a degenerate constant).
                  auto a = [](double x) { return 0.3 * x - 0.05 * x * x; };
                  auto b = [](double x) { return 0.2 * x + 0.1; };
                  auto bp = [](double) { return 0.2; };

                  auto legacy_e = simulate_terminal(a, b, {}, 1.0, 1.5, 64, 500, 424242, false);
                  auto scheme_e = simulate_terminal_scheme(a, b, {}, 1.0, 1.5, 64, 500, 424242,
                                                           Scheme::euler_maruyama);
                  t.expect(legacy_e.has_value() && scheme_e.has_value(),
                           "both euler drivers succeed");
                  t.expect(legacy_e && scheme_e && *legacy_e == *scheme_e,
                           "Scheme::euler_maruyama == legacy use_milstein=false, bit-for-bit");

                  auto legacy_m = simulate_terminal(a, b, bp, 1.0, 1.5, 64, 500, 424242, true);
                  auto scheme_m = simulate_terminal_scheme(a, b, bp, 1.0, 1.5, 64, 500, 424242,
                                                           Scheme::milstein);
                  t.expect(legacy_m.has_value() && scheme_m.has_value(),
                           "both milstein drivers succeed");
                  t.expect(legacy_m && scheme_m && *legacy_m == *scheme_m,
                           "Scheme::milstein == legacy use_milstein=true, bit-for-bit");
              })
        .run();
}
