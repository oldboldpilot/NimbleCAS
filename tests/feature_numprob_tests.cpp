// Feature/integration tests: numerics & probability (numeric, rng, montecarlo, mcmc, pade, powerseries, dynamics).
// @author Olumuyiwa Oluwasanmi
//
// These are cross-module workflow and mathematical-identity tests, NOT per-module unit
// tests: the parallelisability guarantee of the counter RNG (sequential == partitioned),
// Monte Carlo / MCMC estimators driven off that substrate, the agreement of the three
// numeric root finders, the exact algebra of truncated power series over Q, Pade
// resummation of those series, and stability classification cross-checked between the
// Routh-Hurwitz criterion and the eigenvalue signs. Everything stochastic is seeded and
// deterministic; value checks on Monte Carlo use generous tolerances chosen never to
// flake, while the power-series / Pade / dynamics checks are exact (over Q).

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.rng;
import nimblecas.montecarlo;
import nimblecas.mcmc;
import nimblecas.numeric;
import nimblecas.powerseries;
import nimblecas.pade;
import nimblecas.dynamics;
import nimblecas.testing;

using nimblecas::chain_mean;
using nimblecas::chain_variance;
using nimblecas::classify_equilibrium;
using nimblecas::counter_u64;
using nimblecas::estimate_pi;
using nimblecas::fixed_point_affine;
using nimblecas::integrate;
using nimblecas::is_asymptotically_stable;
using nimblecas::MathError;
using nimblecas::Matrix;
using nimblecas::metropolis_hastings;
using nimblecas::pade;
using nimblecas::PowerSeries;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::rejection_sample;
using nimblecas::Rng;
using nimblecas::run_parallel_chains;
using nimblecas::sample_mean;
using nimblecas::sample_variance;
using nimblecas::splitmix64;
using nimblecas::uniform_unit;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace num = nimblecas::numeric;

namespace {

// --- concise builders -------------------------------------------------------

// A reduced fraction n/d; used only where n/d is known valid (non-zero, non-INT64_MIN).
[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

// A truncated power series over Q from integer coefficients (coeffs[i] multiplies x^i),
// held to exactly `order` retained terms.
[[nodiscard]] auto psi(std::vector<std::int64_t> cs, std::size_t order) -> PowerSeries {
    std::vector<Rational> r;
    r.reserve(cs.size());
    for (const std::int64_t v : cs) {
        r.push_back(Rational::from_int(v));
    }
    return PowerSeries::from_coeffs(std::move(r), order).value();
}

// The exp(x) series 1 + x + x^2/2 + ... to the given order (built via the module's own
// exp() over the variable x, whose constant term is 0 as exp() requires).
[[nodiscard]] auto exp_series(std::size_t order) -> PowerSeries {
    return PowerSeries::variable(order).value().exp().value();
}

// Evaluate a RationalPoly at a double abscissa by Horner's scheme (for Pade value checks).
[[nodiscard]] auto poly_eval(const RationalPoly& p, double x) -> double {
    double acc = 0.0;
    for (std::int64_t i = p.degree(); i >= 0; --i) {
        const Rational c = p.coefficient(static_cast<std::size_t>(i));
        acc = acc * x + static_cast<double>(c.numerator()) / static_cast<double>(c.denominator());
    }
    return acc;
}

// Evaluate a truncated power series as a plain polynomial at x (i.e. its Taylor value).
[[nodiscard]] auto series_eval(const PowerSeries& s, double x) -> double {
    double acc = 0.0;
    for (std::size_t k = s.order(); k-- > 0;) {
        const Rational c = s.coefficient(k);
        acc = acc * x + static_cast<double>(c.numerator()) / static_cast<double>(c.denominator());
    }
    return acc;
}

// A square rational matrix from an integer-valued grid.
[[nodiscard]] auto mat(std::vector<std::vector<std::int64_t>> grid) -> Matrix {
    std::vector<std::vector<Rational>> rows;
    rows.reserve(grid.size());
    for (const auto& g : grid) {
        std::vector<Rational> row;
        row.reserve(g.size());
        for (const std::int64_t v : g) {
            row.push_back(Rational::from_int(v));
        }
        rows.push_back(std::move(row));
    }
    return Matrix::from_rows(std::move(rows)).value();
}

}  // namespace

auto main() -> int {
    return TestSuite("feature.numprob")
        // ===================================================================
        // RNG substrate: the parallelisability guarantee.
        // ===================================================================
        .test("rng_partition_independence_ragged_slices",
              [](TestContext& t) {
                  // The whole point of the counter core: the stream produced sequentially
                  // must be ELEMENT-WISE IDENTICAL to the same index range cut into ragged
                  // disjoint slices, generated separately, and concatenated in index order.
                  const std::uint64_t seed = 20260703;
                  const std::uint64_t key = splitmix64(seed);
                  constexpr std::uint64_t n = 500;

                  // Reference: the stateful Rng walking 0..n-1 sequentially.
                  auto rng = Rng::seeded(seed);
                  std::vector<std::uint64_t> seq;
                  seq.reserve(n);
                  for (std::uint64_t i = 0; i < n; ++i) {
                      seq.push_back(rng.next_u64());
                  }

                  // The stateful Rng is exactly counter_u64(key, i) at index i.
                  bool rng_is_core = true;
                  for (std::uint64_t i = 0; i < n; ++i) {
                      if (counter_u64(key, i) != seq[i]) {
                          rng_is_core = false;
                      }
                  }
                  t.expect(rng_is_core,
                           "Rng::seeded stream == counter_u64(key, i) index-for-index");

                  // Three ragged disjoint slices covering [0, n): [0,137), [137,138), [138,n).
                  std::vector<std::uint64_t> slices;
                  slices.reserve(n);
                  for (std::uint64_t i = 0; i < 137; ++i) slices.push_back(counter_u64(key, i));
                  for (std::uint64_t i = 137; i < 138; ++i) slices.push_back(counter_u64(key, i));
                  for (std::uint64_t i = 138; i < n; ++i) slices.push_back(counter_u64(key, i));

                  bool concat_matches = slices.size() == seq.size();
                  for (std::size_t i = 0; i < seq.size() && i < slices.size(); ++i) {
                      if (slices[i] != seq[i]) concat_matches = false;
                  }
                  t.expect(concat_matches,
                           "ragged disjoint slices concatenate to the sequential stream "
                           "(partition-independent / parallelisable)");
              })
        .test("rng_reproducibility_and_split_independence",
              [](TestContext& t) {
                  // Equal seeds reproduce bit-identical streams.
                  auto a = Rng::seeded(9999);
                  auto b = Rng::seeded(9999);
                  bool equal_streams = true;
                  for (int i = 0; i < 256; ++i) {
                      if (a.next_u64() != b.next_u64()) equal_streams = false;
                  }
                  t.expect(equal_streams, "equal seeds give bit-identical streams");

                  // split() yields independent children and does not advance the parent.
                  auto parent = Rng::seeded(42);
                  auto c0 = parent.split(0);
                  auto c1 = parent.split(1);
                  t.expect_ne(c0.key(), c1.key(), "distinct split indices give distinct keys");
                  t.expect_ne(c0.key(), parent.key(), "child key differs from parent key");
                  t.expect_eq(parent.counter(), std::uint64_t{0},
                              "split() does not advance the parent");

                  // Two independent split streams should almost never agree draw-for-draw.
                  int diffs = 0;
                  for (int i = 0; i < 64; ++i) {
                      if (c0.next_u64() != c1.next_u64()) ++diffs;
                  }
                  t.expect(diffs >= 60, "independent split streams differ overwhelmingly");
              })
        .test("rng_distribution_ranges_stay_in_support",
              [](TestContext& t) {
                  auto rng = Rng::seeded(777);
                  bool int_ok = true;
                  bool dbl_ok = true;
                  bool unit_ok = true;
                  for (int i = 0; i < 20000; ++i) {
                      auto vi = rng.next_int(-5, 9);
                      if (!vi || *vi < -5 || *vi > 9) int_ok = false;
                      auto vd = rng.next_double(2.0, 3.0);
                      if (!vd || !(*vd >= 2.0 && *vd < 3.0)) dbl_ok = false;
                      const double u = rng.next_unit();
                      if (!(u >= 0.0 && u < 1.0)) unit_ok = false;
                  }
                  t.expect(int_ok, "uniform_int draws stay in [-5, 9]");
                  t.expect(dbl_ok, "uniform_double draws stay in [2, 3)");
                  t.expect(unit_ok, "next_unit draws stay in [0, 1)");
              })
        // ===================================================================
        // Monte Carlo on the RNG substrate.
        // ===================================================================
        .test("montecarlo_pi_and_known_integrals",
              [](TestContext& t) {
                  // ∫_0^1 x^2 dx = 1/3.
                  auto sq = integrate([](double x) { return x * x; }, 0.0, 1.0, 200000, 55);
                  t.expect(sq.has_value(), "integrate(x^2, [0,1]) succeeds");
                  if (sq) {
                      t.expect(std::abs(*sq - 1.0 / 3.0) < 0.02,
                               std::format("integrate(x^2) ~ 1/3 (got {})", *sq));
                  }

                  // ∫_0^1 4/(1+x^2) dx = π.
                  auto quad = integrate([](double x) { return 4.0 / (1.0 + x * x); }, 0.0, 1.0,
                                        200000, 55);
                  t.expect(quad.has_value(), "integrate(4/(1+x^2), [0,1]) succeeds");
                  if (quad) {
                      t.expect(std::abs(*quad - std::numbers::pi) < 0.02,
                               std::format("integrate(4/(1+x^2)) ~ pi (got {})", *quad));
                  }

                  // Dart estimate of π.
                  auto pi = estimate_pi(200000, 12345);
                  t.expect(pi.has_value(), "estimate_pi succeeds");
                  if (pi) {
                      t.expect(std::abs(*pi - std::numbers::pi) < 0.03,
                               std::format("estimate_pi ~ pi (got {})", *pi));
                  }
              })
        .test("montecarlo_rejection_support_and_statistics",
              [](TestContext& t) {
                  // Rejection sampling of the triangular density pdf(x) = x on [0,1] (ceiling
                  // m = 1): every accepted point must lie in the support, and the mean ~ 2/3.
                  auto pdf = [](double x) { return x; };
                  auto r = rejection_sample(pdf, 0.0, 1.0, 1.0, 4000, 2024, 10000000);
                  t.expect(r.has_value(), "rejection_sample succeeds");
                  if (r) {
                      t.expect(!r->empty(), "rejection_sample returns samples");
                      bool in_support = true;
                      for (const double x : *r) {
                          if (!(x >= 0.0 && x <= 1.0)) in_support = false;
                      }
                      t.expect(in_support, "all accepted samples lie in [0,1] (in support)");

                      auto m = sample_mean(*r);
                      t.expect(m.has_value() && std::abs(*m - 2.0 / 3.0) < 0.03,
                               std::format("accepted-sample mean ~ 2/3 (got {})",
                                           m ? *m : 0.0));
                  }

                  // Exact statistics on a fixed dataset (mean 5, pop var 4, sample var 32/7).
                  const std::array<double, 8> data{2, 4, 4, 4, 5, 5, 7, 9};
                  std::span<const double> s{data};
                  auto mean = sample_mean(s);
                  auto pop = sample_variance(s, false);
                  auto var = sample_variance(s, true);
                  t.expect(mean.has_value() && std::abs(*mean - 5.0) < 1e-12,
                           "sample_mean == 5 exactly");
                  t.expect(pop.has_value() && std::abs(*pop - 4.0) < 1e-12,
                           "population variance == 4 exactly");
                  t.expect(var.has_value() && std::abs(*var - 32.0 / 7.0) < 1e-12,
                           "Bessel sample variance == 32/7 exactly");
              })
        .test("montecarlo_matches_manual_rng_reconstruction",
              [](TestContext& t) {
                  // CROSS-MODULE determinism: integrate() must equal a hand-rolled sum built
                  // directly on the rng primitives (splitmix64 -> counter_u64 -> uniform_unit),
                  // BIT-FOR-BIT. This ties montecarlo to the exact substrate contract.
                  const std::uint64_t seed = 424242;
                  constexpr std::uint64_t samples = 4096;
                  auto f = [](double x) { return x * x; };
                  auto mc = integrate(f, 0.0, 1.0, samples, seed);
                  t.expect(mc.has_value(), "integrate() succeeds");

                  const std::uint64_t key = splitmix64(seed);
                  double sum = 0.0;
                  for (std::uint64_t i = 0; i < samples; ++i) {
                      const double x = 0.0 + uniform_unit(counter_u64(key, i)) * (1.0 - 0.0);
                      sum += f(x);
                  }
                  const double manual = (1.0 - 0.0) * (sum / static_cast<double>(samples));
                  if (mc) {
                      t.expect(*mc == manual,
                               "integrate() reproduces the manual counter_u64 sum bit-for-bit");
                  }
              })
        // ===================================================================
        // MCMC: standard-normal target, determinism, parallel chains.
        // ===================================================================
        .test("mcmc_standard_normal_moments",
              [](TestContext& t) {
                  // Random-walk Metropolis against log N(0,1) = -x^2/2 (unnormalised). A long
                  // seeded chain must recover mean ~ 0 and variance ~ 1.
                  auto log_density = [](double x) { return -0.5 * x * x; };
                  auto run = metropolis_hastings(log_density, 0.0, 2.5, 200000, 20000, 24680);
                  t.expect(run.has_value(), "metropolis_hastings succeeds");
                  if (run) {
                      t.expect(run->proposed == 220000, "proposed == burn_in + samples");
                      t.expect(run->chain.size() == 200000, "chain retains exactly `samples`");
                      auto m = chain_mean(run->chain);
                      auto v = chain_variance(run->chain);
                      t.expect(m.has_value() && std::abs(*m) < 0.10,
                               std::format("chain mean ~ 0 (got {})", m ? *m : 0.0));
                      t.expect(v.has_value() && std::abs(*v - 1.0) < 0.20,
                               std::format("chain variance ~ 1 (got {})", v ? *v : 0.0));
                  }
              })
        .test("mcmc_chain_is_bit_reproducible",
              [](TestContext& t) {
                  auto log_density = [](double x) { return -0.5 * x * x; };
                  auto a = metropolis_hastings(log_density, 0.3, 1.7, 5000, 500, 13579);
                  auto b = metropolis_hastings(log_density, 0.3, 1.7, 5000, 500, 13579);
                  t.expect(a.has_value() && b.has_value(), "both chains run");
                  if (a && b) {
                      t.expect(a->chain == b->chain,
                               "equal seeds give a bit-identical chain");
                      t.expect(a->accepted == b->accepted && a->proposed == b->proposed,
                               "equal seeds give identical accept/propose counts");
                  }
              })
        .test("mcmc_parallel_chains_reproducible",
              [](TestContext& t) {
                  // run_parallel_chains seeds chain c with splitmix64(seed ^ c). Verify that
                  // contract directly: chain c must equal a standalone metropolis_hastings run
                  // with exactly that derived seed, regardless of chain count / ordering.
                  auto log_density = [](double x) { return -0.5 * x * x; };
                  const std::uint64_t seed = 8642;
                  const std::uint64_t chains = 3;
                  auto par = run_parallel_chains(log_density, 0.0, 2.0, 400, 100, seed, chains);
                  t.expect(par.has_value(), "run_parallel_chains succeeds");
                  if (par) {
                      t.expect(par->size() == chains, "one result per chain");
                      bool all_match = par->size() == chains;
                      for (std::uint64_t c = 0; c < chains && c < par->size(); ++c) {
                          const std::uint64_t chain_seed = splitmix64(seed ^ c);
                          auto solo = metropolis_hastings(log_density, 0.0, 2.0, 400, 100,
                                                          chain_seed);
                          if (!solo || solo->chain != (*par)[c].chain) all_match = false;
                      }
                      t.expect(all_match,
                               "each parallel chain == solo run with splitmix64(seed ^ c)");
                  }
              })
        // ===================================================================
        // Numeric root finders: agreement and error paths.
        // ===================================================================
        .test("numeric_root_finders_agree_on_sqrt2",
              [](TestContext& t) {
                  // p(x) = x^2 - 2, ascending coefficients {-2, 0, 1}; the positive root is √2.
                  const std::array<double, 3> c{-2.0, 0.0, 1.0};
                  std::span<const double> p{c};
                  const double tol = 1e-12;
                  const double root = std::numbers::sqrt2;

                  auto bi = num::bisection(p, 0.0, 2.0, tol);
                  auto nw = num::newton(p, 2.0, tol, 200);
                  auto sc = num::secant(p, 1.0, 2.0, tol, 200);
                  t.expect(bi.has_value() && nw.has_value() && sc.has_value(),
                           "all three solvers converge");
                  if (bi && nw && sc) {
                      t.expect(std::abs(*bi - root) < 1e-9,
                               std::format("bisection -> sqrt2 (got {})", *bi));
                      t.expect(std::abs(*nw - root) < 1e-9,
                               std::format("newton -> sqrt2 (got {})", *nw));
                      t.expect(std::abs(*sc - root) < 1e-9,
                               std::format("secant -> sqrt2 (got {})", *sc));
                      // Cross-method agreement.
                      t.expect(std::abs(*bi - *nw) < 1e-6 && std::abs(*nw - *sc) < 1e-6,
                               "the three root finders agree with one another");
                  }
              })
        .test("numeric_endpoint_root_and_error_paths",
              [](TestContext& t) {
                  // Root exactly on the lower bracket endpoint: p(x) = x^2 - 1 on [1, 3],
                  // f(1) == 0, so bisection must return the endpoint 1 directly.
                  const std::array<double, 3> pm1{-1.0, 0.0, 1.0};
                  std::span<const double> q{pm1};
                  auto endpoint = num::bisection(q, 1.0, 3.0, 1e-12);
                  t.expect(endpoint.has_value() && *endpoint == 1.0,
                           "bisection returns an exact endpoint root");

                  // No sign change on [2, 3] (both values positive) => domain_error.
                  const std::array<double, 3> c{-2.0, 0.0, 1.0};
                  std::span<const double> p{c};
                  auto no_bracket = num::bisection(p, 2.0, 3.0, 1e-12);
                  t.expect(!no_bracket.has_value() &&
                               no_bracket.error() == MathError::domain_error,
                           "bisection without a sign change yields domain_error");

                  // Empty coefficient span => domain_error for every solver.
                  std::span<const double> empty{};
                  auto be = num::bisection(empty, 0.0, 1.0, 1e-12);
                  auto ne = num::newton(empty, 1.0, 1e-12, 50);
                  auto se = num::secant(empty, 0.0, 1.0, 1e-12, 50);
                  t.expect(!be.has_value() && be.error() == MathError::domain_error,
                           "bisection(empty) yields domain_error");
                  t.expect(!ne.has_value() && ne.error() == MathError::domain_error,
                           "newton(empty) yields domain_error");
                  t.expect(!se.has_value() && se.error() == MathError::domain_error,
                           "secant(empty) yields domain_error");
              })
        // ===================================================================
        // Power series over Q: exact transcendental and calculus identities.
        // ===================================================================
        .test("powerseries_exp_log_are_mutual_inverses",
              [](TestContext& t) {
                  constexpr std::size_t order = 8;
                  // exp(log(1+x)) == 1 + x, truncated.
                  const PowerSeries one_plus_x = psi({1, 1}, order);
                  auto lg = one_plus_x.log();
                  t.expect(lg.has_value(), "log(1+x) exists (constant term 1)");
                  if (lg) {
                      auto back = lg->exp();
                      t.expect(back.has_value() && back->is_equal(one_plus_x),
                               "exp(log(1+x)) == 1+x exactly over Q");
                  }

                  // log(exp(x)) == x, truncated.
                  const PowerSeries x = PowerSeries::variable(order).value();
                  auto ex = x.exp();
                  t.expect(ex.has_value(), "exp(x) exists (constant term 0)");
                  if (ex) {
                      auto lx = ex->log();
                      t.expect(lx.has_value() && lx->is_equal(x),
                               "log(exp(x)) == x exactly over Q");
                  }
              })
        .test("powerseries_inverse_and_geometric_identity",
              [](TestContext& t) {
                  constexpr std::size_t order = 8;
                  const PowerSeries one = PowerSeries::one(order).value();

                  // s * s^{-1} == 1 for a unit series (here s = exp(x), c_0 = 1).
                  const PowerSeries s = exp_series(order);
                  auto inv = s.inverse();
                  t.expect(inv.has_value(), "exp series is invertible");
                  if (inv) {
                      auto prod = s.multiply(*inv);
                      t.expect(prod.has_value() && prod->is_equal(one),
                               "s * s^{-1} == 1 exactly");
                  }

                  // 1/(1-x) == 1 + x + x^2 + ... : the inverse of (1 - x) is the all-ones
                  // geometric series, and (1-x) times it recovers 1.
                  const PowerSeries one_minus_x = psi({1, -1}, order);
                  auto geom = one_minus_x.inverse();
                  t.expect(geom.has_value(), "(1-x) is invertible");
                  if (geom) {
                      bool all_ones = true;
                      for (std::size_t k = 0; k < order; ++k) {
                          if (!(geom->coefficient(k) == Rational::from_int(1))) all_ones = false;
                      }
                      t.expect(all_ones, "1/(1-x) has every coefficient == 1 (geometric series)");
                      auto recovered = one_minus_x.multiply(*geom);
                      t.expect(recovered.has_value() && recovered->is_equal(one),
                               "(1-x) * 1/(1-x) == 1 exactly");
                  }
              })
        .test("powerseries_calculus_and_composition",
              [](TestContext& t) {
                  constexpr std::size_t order = 6;
                  // derivative(integrate(s)) == s when s has a zero top coefficient (so nothing
                  // is lost to truncation): here s = 3 + x + 4x^2 + x^3 + 5x^4 + 0*x^5.
                  const PowerSeries s = psi({3, 1, 4, 1, 5, 0}, order);
                  auto integ = s.integrate();
                  t.expect(integ.has_value(), "integrate(s) succeeds");
                  if (integ) {
                      auto back = integ->derivative();
                      t.expect(back.has_value() && back->is_equal(s),
                               "derivative(integrate(s)) == s (constant of integration aside)");
                  }

                  // Composition identity: f(x) composed with the series x is f itself.
                  const PowerSeries f = psi({3, 1, 4, 1, 5, 9}, order);
                  const PowerSeries x = PowerSeries::variable(order).value();
                  auto id = f.compose(x);
                  t.expect(id.has_value() && id->is_equal(f),
                           "f o x == f (identity composition)");

                  // 1/(1-x) composed with x^2 gives 1/(1-x^2): even coeffs 1, odd coeffs 0.
                  auto geom = psi({1, -1}, order).inverse();
                  const PowerSeries x_sq = psi({0, 0, 1}, order);
                  t.expect(geom.has_value(), "geometric series exists");
                  if (geom) {
                      auto comp = geom->compose(x_sq);
                      t.expect(comp.has_value(), "1/(1-x) o x^2 succeeds");
                      if (comp) {
                          bool pattern = true;
                          for (std::size_t k = 0; k < order; ++k) {
                              const Rational want =
                                  (k % 2 == 0) ? Rational::from_int(1) : Rational::from_int(0);
                              if (!(comp->coefficient(k) == want)) pattern = false;
                          }
                          t.expect(pattern,
                                   "1/(1-x^2) has 1 at even powers and 0 at odd powers");
                      }
                  }
              })
        // ===================================================================
        // Pade: known forms, exact rational recovery, and resummation quality.
        // ===================================================================
        .test("pade_exp_one_over_one_is_two_plus_x_over_two_minus_x",
              [](TestContext& t) {
                  // The [1/1] Pade approximant of e^x is (1 + x/2)/(1 - x/2), i.e. P = 1 + x/2,
                  // Q = 1 - x/2 in the Q(0)=1 normalisation the module uses.
                  const PowerSeries s = exp_series(3);
                  auto approx = pade(s, 1, 1);
                  t.expect(approx.has_value(), "pade[1/1] of exp exists");
                  if (approx) {
                      const auto& [p, q] = *approx;
                      t.expect(p.coefficient(0) == Rational::from_int(1) &&
                                   p.coefficient(1) == rat(1, 2) && p.degree() == 1,
                               "numerator P == 1 + x/2");
                      t.expect(q.coefficient(0) == Rational::from_int(1) &&
                                   q.coefficient(1) == rat(-1, 2) && q.degree() == 1,
                               "denominator Q == 1 - x/2");
                  }
              })
        .test("pade_reproduces_a_rational_function_exactly",
              [](TestContext& t) {
                  // Series of R(x) = (1 + 2x)/(1 + 3x + x^2); its [1/2] Pade must recover the
                  // numerator and denominator EXACTLY (both already Q(0)=1 / P(0)=1 normalised).
                  constexpr std::size_t order = 6;
                  const PowerSeries numer = psi({1, 2}, order);
                  const PowerSeries denom = psi({1, 3, 1}, order);
                  auto s = numer.divide(denom);
                  t.expect(s.has_value(), "power series of R(x) exists");
                  if (s) {
                      auto approx = pade(*s, 1, 2);
                      t.expect(approx.has_value(), "pade[1/2] of R's series exists");
                      if (approx) {
                          const auto& [p, q] = *approx;
                          t.expect(p.degree() == 1 && p.coefficient(0) == Rational::from_int(1) &&
                                       p.coefficient(1) == Rational::from_int(2),
                                   "recovered numerator == 1 + 2x");
                          t.expect(q.degree() == 2 && q.coefficient(0) == Rational::from_int(1) &&
                                       q.coefficient(1) == Rational::from_int(3) &&
                                       q.coefficient(2) == Rational::from_int(1),
                                   "recovered denominator == 1 + 3x + x^2");
                      }
                  }
              })
        .test("pade_resums_exp_better_than_taylor",
              [](TestContext& t) {
                  // CROSS: build the truncated exp series (order 5), resum it as a [2/2] Pade,
                  // and evaluate at x = 1. The rational resummation must beat the same-order
                  // Taylor truncation against the true value e.
                  constexpr double x = 1.0;
                  const double e = std::numbers::e;
                  const PowerSeries s = exp_series(5);

                  const double taylor = series_eval(s, x);
                  auto approx = pade(s, 2, 2);
                  t.expect(approx.has_value(), "pade[2/2] of exp exists");
                  if (approx) {
                      const auto& [p, q] = *approx;
                      const double denom = poly_eval(q, x);
                      t.expect(denom != 0.0, "Pade denominator is non-zero at x=1");
                      if (denom != 0.0) {
                          const double resummed = poly_eval(p, x) / denom;
                          t.expect(std::abs(resummed - e) < std::abs(taylor - e),
                                   std::format("Pade error {} < Taylor error {}",
                                               std::abs(resummed - e), std::abs(taylor - e)));
                          t.expect(std::abs(resummed - e) < 0.01,
                                   std::format("Pade resummation ~ e (got {})", resummed));
                      }
                  }
              })
        // ===================================================================
        // Dynamics: Routh-Hurwitz vs eigenvalue-sign stability, and equilibria.
        // ===================================================================
        .test("dynamics_stable_unstable_marginal_classification",
              [](TestContext& t) {
                  // Stable node: A = diag(-1, -2), both eigenvalues negative.
                  const Matrix stable = mat({{-1, 0}, {0, -2}});
                  auto st = is_asymptotically_stable(stable);
                  auto sc = classify_equilibrium(stable);
                  t.expect(st.has_value() && *st == true, "diag(-1,-2) is asymptotically stable");
                  t.expect(sc.has_value() && *sc == "stable node", "diag(-1,-2) is a stable node");

                  // Saddle: A = diag(1, -1), one positive and one negative eigenvalue.
                  const Matrix saddle = mat({{1, 0}, {0, -1}});
                  auto us = is_asymptotically_stable(saddle);
                  auto uc = classify_equilibrium(saddle);
                  t.expect(us.has_value() && *us == false, "diag(1,-1) is NOT asymptotically stable");
                  t.expect(uc.has_value() && *uc == "saddle", "diag(1,-1) is a saddle");

                  // Marginal rotation: A = [[0,-1],[1,0]] has the pure imaginary pair ±i, which
                  // Routh-Hurwitz flags (imaginary-axis roots) and rational eigenvalues cannot see.
                  const Matrix rotation = mat({{0, -1}, {1, 0}});
                  auto rs = is_asymptotically_stable(rotation);
                  auto rc = classify_equilibrium(rotation);
                  t.expect(rs.has_value() && *rs == false,
                           "rotation is NOT asymptotically stable (imaginary-axis pair)");
                  t.expect(rc.has_value() && *rc == "unstable or marginal (non-rational spectrum)",
                           "rotation classified as marginal / non-rational spectrum");
              })
        .test("dynamics_routh_hurwitz_agrees_and_fixed_point",
              [](TestContext& t) {
                  // Stable spiral A = [[-1,-1],[1,-1]] has eigenvalues -1 ± i (negative real
                  // part) — irrational to the rational-eigenvalue view, so the verdict rests on
                  // Routh-Hurwitz. It must agree that the system is asymptotically stable.
                  const Matrix spiral = mat({{-1, -1}, {1, -1}});
                  auto rh = is_asymptotically_stable(spiral);
                  auto cls = classify_equilibrium(spiral);
                  t.expect(rh.has_value() && *rh == true,
                           "Routh-Hurwitz: stable spiral is asymptotically stable");
                  t.expect(cls.has_value() && *cls == "asymptotically stable (spiral/node)",
                           "spiral classified as asymptotically stable");

                  // Affine equilibrium of x -> A x + b with A = diag(1/2, 1/3), b = (1, 1):
                  // solves (I - A) x = b, i.e. x = (2, 3/2) exactly over Q.
                  const Matrix a = Matrix::from_rows({{rat(1, 2), Rational::from_int(0)},
                                                      {Rational::from_int(0), rat(1, 3)}})
                                       .value();
                  const Matrix b = Matrix::from_rows({{Rational::from_int(1)},
                                                      {Rational::from_int(1)}})
                                       .value();
                  auto fp = fixed_point_affine(a, b);
                  t.expect(fp.has_value(), "fixed_point_affine solves (I-A)x = b");
                  if (fp) {
                      t.expect(fp->rows() == 2 && fp->cols() == 1, "equilibrium is a 2x1 column");
                      t.expect(fp->at(0, 0) == Rational::from_int(2) && fp->at(1, 0) == rat(3, 2),
                               "equilibrium == (2, 3/2) exactly over Q");
                  }
              })
        .run();
}
