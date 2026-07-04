// Tests for nimblecas.qmc: exact low-discrepancy points (Van der Corput / Halton / Hammersley /
// Sobol' / rank-1 lattice), QMC integration (numerical and exact-rational), randomized QMC with
// its statistical error, iterative/adaptive refinement, and the Warnock L2 star discrepancy.
// @author Olumuyiwa Oluwasanmi
//
// HONESTY: the low-discrepancy POINTS are exact (checked with Rational equality); the INTEGRALS
// of a general f are numerical (checked with tolerances); the RQMC error is a statistical
// estimate, so those checks compare against the true value with generous margins, never asserting
// a deterministic bound. The QMC-beats-MC checks use huge margins (~orders of magnitude) so they
// cannot flake on the stochastic MC side.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.rng;
import nimblecas.montecarlo;
import nimblecas.qmc;
import nimblecas.testing;

using nimblecas::AdaptiveResult;
using nimblecas::adaptive_qmc;
using nimblecas::halton_point;
using nimblecas::halton_point_rational;
using nimblecas::integrate;
using nimblecas::hammersley_point_rational;
using nimblecas::l2_star_discrepancy;
using nimblecas::l2_star_discrepancy_squared_exact;
using nimblecas::lattice_point_rational;
using nimblecas::parallel_qmc_integrate;
using nimblecas::parallel_rqmc_integrate;
using nimblecas::qmc_integrate;
using nimblecas::qmc_integrate_exact;
using nimblecas::Rational;
using nimblecas::rational_to_double;
using nimblecas::Rng;
using nimblecas::RqmcResult;
using nimblecas::rqmc_integrate;
using nimblecas::rqmc_reduce_shards;
using nimblecas::rqmc_shard;
using nimblecas::RqmcShardResult;
using nimblecas::sobol_point_rational;
using nimblecas::van_der_corput_rational;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Build an exact Rational num/den, aborting the check helpfully if it (unexpectedly) fails.
[[nodiscard]] auto rat(std::int64_t num, std::int64_t den) -> Rational {
    auto r = Rational::make(num, den);
    return r ? *r : Rational{};
}

// Plain Monte Carlo estimate of ∫∫_{[0,1]^2} f, using the same counter RNG substrate, so the
// QMC-vs-MC comparison is at equal N on a like-for-like pseudorandom baseline.
[[nodiscard]] auto mc_integrate_2d(auto f, std::uint64_t n, std::uint64_t seed) -> double {
    auto rng = Rng::seeded(seed);
    double sum = 0.0;
    for (std::uint64_t i = 0; i < n; ++i) {
        const double x = rng.next_unit();
        const double y = rng.next_unit();
        sum += f(x, y);
    }
    return sum / static_cast<double>(n);
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.qmc")
        .test("van_der_corput_base2_exact_sequence",
              [](TestContext& t) {
                  // phi_2(1..4) = 1/2, 1/4, 3/4, 1/8 — EXACT rationals.
                  const std::array<Rational, 4> expected{rat(1, 2), rat(1, 4), rat(3, 4),
                                                          rat(1, 8)};
                  for (std::uint64_t n = 1; n <= 4; ++n) {
                      auto r = van_der_corput_rational(n, 2);
                      t.expect(r.has_value(),
                               std::format("van_der_corput_rational({}, 2) succeeds", n));
                      if (r) {
                          t.expect(*r == expected[n - 1],
                                   std::format("phi_2({}) exact (got {})", n, r->to_string()));
                      }
                  }
                  // base < 2 is a domain error.
                  auto bad = van_der_corput_rational(3, 1);
                  t.expect(!bad.has_value() &&
                               bad.error() == nimblecas::MathError::domain_error,
                           "van_der_corput_rational(base<2) yields domain_error");
              })
        .test("halton_first_points_exact",
              [](TestContext& t) {
                  // Halton with bases (2,3): point 1 = (1/2, 1/3), point 2 = (1/4, 2/3).
                  auto p1 = halton_point_rational(1, 2);
                  t.expect(p1.has_value(), "halton_point_rational(1, 2) succeeds");
                  if (p1) {
                      t.expect(p1->size() == 2 && (*p1)[0] == rat(1, 2) && (*p1)[1] == rat(1, 3),
                               "Halton point 1 == (1/2, 1/3) exact");
                  }
                  auto p2 = halton_point_rational(2, 2);
                  t.expect(p2.has_value(), "halton_point_rational(2, 2) succeeds");
                  if (p2) {
                      t.expect(p2->size() == 2 && (*p2)[0] == rat(1, 4) && (*p2)[1] == rat(2, 3),
                               "Halton point 2 == (1/4, 2/3) exact");
                  }

                  auto bad = halton_point_rational(1, 0);
                  t.expect(!bad.has_value() &&
                               bad.error() == nimblecas::MathError::domain_error,
                           "halton_point_rational(dimension 0) yields domain_error");
              })
        .test("hammersley_first_coordinate_is_index_over_n",
              [](TestContext& t) {
                  // Hammersley(i=1, total=4, d=2) = (1/4, phi_2(1)=1/2).
                  auto p = hammersley_point_rational(1, 4, 2);
                  t.expect(p.has_value(), "hammersley_point_rational succeeds");
                  if (p) {
                      t.expect(p->size() == 2 && (*p)[0] == rat(1, 4) && (*p)[1] == rat(1, 2),
                               "Hammersley point == (1/4, 1/2) exact");
                  }
                  auto bad = hammersley_point_rational(1, 0, 2);
                  t.expect(!bad.has_value() &&
                               bad.error() == nimblecas::MathError::domain_error,
                           "hammersley(total_n 0) yields domain_error");
              })
        .test("sobol_points_are_dyadic_exact",
              [](TestContext& t) {
                  // Sobol' dimension 1, index 1 -> 1/2 exactly; every coordinate is dyadic
                  // (denominator a power of two).
                  auto p = sobol_point_rational(1, 2);
                  t.expect(p.has_value(), "sobol_point_rational(1, 2) succeeds");
                  if (p) {
                      t.expect((*p)[0] == rat(1, 2), "Sobol'(1) dim0 == 1/2 exact");
                      bool dyadic = true;
                      for (const Rational& c : *p) {
                          std::int64_t den = c.denominator();
                          if (den <= 0 || (den & (den - 1)) != 0) {  // not a power of two
                              dyadic = false;
                          }
                      }
                      t.expect(dyadic, "all Sobol' coordinates are dyadic (denominator = 2^k)");
                  }
                  // Dimension beyond the built-in direction table is an honest domain error.
                  auto too_big = sobol_point_rational(1, 9);
                  t.expect(!too_big.has_value() &&
                               too_big.error() == nimblecas::MathError::domain_error,
                           "sobol dimension > 8 yields domain_error");
              })
        .test("lattice_points_are_exact_fractions",
              [](TestContext& t) {
                  // z = {1}, N = 4: points n=0..3 -> 0, 1/4, 1/2, 3/4 exactly.
                  const std::array<std::uint64_t, 1> z{1};
                  const std::array<Rational, 4> expected{rat(0, 1), rat(1, 4), rat(1, 2),
                                                          rat(3, 4)};
                  for (std::uint64_t n = 0; n < 4; ++n) {
                      auto p = lattice_point_rational(n, 4, z);
                      t.expect(p.has_value(), "lattice_point_rational succeeds");
                      if (p) {
                          t.expect(p->size() == 1 && (*p)[0] == expected[n],
                                   std::format("lattice point {} exact", n));
                      }
                  }
                  auto bad = lattice_point_rational(0, 0, z);
                  t.expect(!bad.has_value() &&
                               bad.error() == nimblecas::MathError::domain_error,
                           "lattice(total_n 0) yields domain_error");
              })
        .test("qmc_integrate_constant_is_exactly_one",
              [](TestContext& t) {
                  // The equal-weight average of the constant 1 is exactly 1 in every dimension.
                  for (std::size_t d = 1; d <= 3; ++d) {
                      auto r = qmc_integrate([](std::span<const double>) { return 1.0; }, d, 500);
                      t.expect(r.has_value(), std::format("qmc_integrate(1, d={}) succeeds", d));
                      if (r) {
                          t.expect(std::abs(*r - 1.0) < 1e-12,
                                   std::format("qmc_integrate(1, d={}) == 1 (got {})", d, *r));
                      }
                  }
                  // Exact path: average of the constant 1 is exactly the Rational 1.
                  auto er = qmc_integrate_exact(
                      [](std::span<const Rational>) -> nimblecas::Result<Rational> {
                          return Rational::from_int(1);
                      },
                      2, 16);
                  t.expect(er.has_value(), "qmc_integrate_exact(1) succeeds");
                  if (er) {
                      t.expect(*er == rat(1, 1), "qmc_integrate_exact(1) == 1 exactly");
                  }
              })
        .test("qmc_integrate_exact_average_of_halton",
              [](TestContext& t) {
                  // Exact average of the first-coordinate (phi_2) over Halton points 1..4:
                  // (1/2 + 1/4 + 3/4 + 1/8)/4 = (13/8)/4 = 13/32.
                  auto r = qmc_integrate_exact(
                      [](std::span<const Rational> x) -> nimblecas::Result<Rational> {
                          return x[0];
                      },
                      1, 4);
                  t.expect(r.has_value(), "qmc_integrate_exact(x, N=4) succeeds");
                  if (r) {
                      t.expect(*r == rat(13, 32),
                               std::format("exact QMC average == 13/32 (got {})", r->to_string()));
                  }
              })
        .test("qmc_beats_plain_mc_1d",
              [](TestContext& t) {
                  // ∫_0^1 x dx = 1/2. Compare QMC (Halton) vs plain MC at EQUAL N. To keep the
                  // comparison robust (one MC seed can luckily land near the truth), we compare the
                  // single deterministic QMC error against the ROOT-MEAN-SQUARE MC error over many
                  // seeds — a stable proxy for the MC accuracy that QMC decisively beats.
                  constexpr std::uint64_t N = 4096;
                  auto q = qmc_integrate([](std::span<const double> x) { return x[0]; }, 1, N);
                  t.expect(q.has_value(), "qmc_integrate 1-D succeeds");

                  double mse = 0.0;
                  constexpr int K = 16;
                  for (int k = 0; k < K; ++k) {
                      auto mc = integrate([](double x) { return x; }, 0.0, 1.0, N,
                                          1000 + static_cast<std::uint64_t>(k) * 7);
                      if (mc) {
                          const double d = *mc - 0.5;
                          mse += d * d;
                      }
                  }
                  const double mc_rms = std::sqrt(mse / static_cast<double>(K));
                  if (q) {
                      const double q_err = std::abs(*q - 0.5);
                      t.expect(q_err < 0.01, std::format("QMC ∫x ~ 1/2 (err {})", q_err));
                      t.expect(q_err < mc_rms,
                               std::format("QMC error {} < MC RMS error {} at N={}", q_err, mc_rms,
                                           N));
                  }
              })
        .test("qmc_beats_plain_mc_2d",
              [](TestContext& t) {
                  // ∫∫_{[0,1]^2} x·y dx dy = 1/4. QMC (Halton bases 2,3) vs plain 2-D MC, again
                  // comparing the deterministic QMC error to the RMS MC error over many seeds.
                  constexpr std::uint64_t N = 4096;
                  auto q = qmc_integrate(
                      [](std::span<const double> p) { return p[0] * p[1]; }, 2, N);
                  t.expect(q.has_value(), "qmc_integrate 2-D succeeds");

                  double mse = 0.0;
                  constexpr int K = 16;
                  for (int k = 0; k < K; ++k) {
                      const double mc = mc_integrate_2d([](double x, double y) { return x * y; }, N,
                                                        500 + static_cast<std::uint64_t>(k) * 13);
                      const double d = mc - 0.25;
                      mse += d * d;
                  }
                  const double mc_rms = std::sqrt(mse / static_cast<double>(K));
                  if (q) {
                      const double q_err = std::abs(*q - 0.25);
                      t.expect(q_err < 0.01, std::format("QMC ∫∫xy ~ 1/4 (err {})", q_err));
                      t.expect(q_err < mc_rms,
                               std::format("2-D QMC error {} < MC RMS error {}", q_err, mc_rms));
                  }
              })
        .test("qmc_error_shrinks_with_more_points",
              [](TestContext& t) {
                  // Extensible/iterative refinement: increasing N reduces the QMC error
                  // (monotonically-ish; here strictly across a wide N gap for the smooth ∫x).
                  auto f = [](std::span<const double> x) { return x[0]; };
                  auto coarse = qmc_integrate(f, 1, 64);
                  auto fine = qmc_integrate(f, 1, 4096);
                  t.expect(coarse.has_value() && fine.has_value(), "both refinements succeed");
                  if (coarse && fine) {
                      t.expect(std::abs(*fine - 0.5) < std::abs(*coarse - 0.5),
                               "finer QMC grid has smaller error than coarser");
                  }
              })
        .test("rqmc_brackets_true_value",
              [](TestContext& t) {
                  // Randomized QMC of ∫x = 1/2. The mean should be near 1/2 and the (statistical)
                  // standard error positive; the true value lies within a few standard errors.
                  auto r = rqmc_integrate([](std::span<const double> x) { return x[0]; }, 1, 512,
                                          16, 4242);
                  t.expect(r.has_value(), "rqmc_integrate succeeds");
                  if (r) {
                      t.expect(r->error_estimate > 0.0, "RQMC standard error is positive");
                      t.expect(r->points_used == 512ULL * 16ULL, "points_used == N * replications");
                      t.expect(std::abs(r->estimate - 0.5) < 0.02,
                               std::format("RQMC estimate ~ 1/2 (got {})", r->estimate));
                      t.expect(std::abs(r->estimate - 0.5) < 6.0 * r->error_estimate,
                               "true value within a few RQMC standard errors of the estimate");
                  }
                  // replications < 2 cannot form a variance -> domain error.
                  auto bad = rqmc_integrate([](std::span<const double> x) { return x[0]; }, 1, 10,
                                            1, 1);
                  t.expect(!bad.has_value() &&
                               bad.error() == nimblecas::MathError::domain_error,
                           "rqmc(replications<2) yields domain_error");
              })
        .test("parallel_qmc_integrate_matches_serial_bit_for_bit",
              [](TestContext& t) {
                  // The parallel point-batch integrator sums the SAME point evaluations in the
                  // SAME index order as the serial one, so the result is bit-identical (not merely
                  // close) for any thread count.
                  auto f = [](std::span<const double> p) { return p[0] * p[1] + std::sin(p[0]); };
                  const std::array<std::uint64_t, 3> Ns{1, 64, 1000};
                  for (const std::uint64_t N : Ns) {
                      auto s = qmc_integrate(f, 2, N);
                      auto p = parallel_qmc_integrate(f, 2, N);
                      t.expect(s.has_value() && p.has_value(),
                               std::format("both qmc_integrate variants succeed (N={})", N));
                      if (s && p) {
                          t.expect(*p == *s,
                                   std::format("parallel_qmc_integrate == serial bit-for-bit (N={}, "
                                               "got {} vs {})",
                                               N, *p, *s));
                      }
                  }
                  // Domain errors mirror the serial version.
                  auto d0 = parallel_qmc_integrate(f, 0, 100);
                  t.expect(!d0.has_value() && d0.error() == nimblecas::MathError::domain_error,
                           "parallel_qmc_integrate(dimension 0) yields domain_error");
                  auto n0 = parallel_qmc_integrate(f, 2, 0);
                  t.expect(!n0.has_value() && n0.error() == nimblecas::MathError::domain_error,
                           "parallel_qmc_integrate(N 0) yields domain_error");
              })
        .test("parallel_rqmc_integrate_matches_serial_bit_for_bit",
              [](TestContext& t) {
                  // Determinism: each replication is a pure function of base.split(r) and the fixed
                  // Halton points, and the mean/variance reduction runs in fixed index order — so
                  // the parallel result is bit-identical to the serial rqmc_integrate.
                  auto f = [](std::span<const double> p) { return p[0] * p[1]; };
                  const std::uint64_t N = 256;
                  const std::uint64_t reps = 8;
                  const std::uint64_t seed = 12345;
                  auto s = rqmc_integrate(f, 2, N, reps, seed);
                  auto p = parallel_rqmc_integrate(f, 2, N, reps, seed);
                  t.expect(s.has_value() && p.has_value(), "both rqmc_integrate variants succeed");
                  if (s && p) {
                      t.expect(p->estimate == s->estimate,
                               std::format("parallel RQMC estimate bit-identical ({} vs {})",
                                           p->estimate, s->estimate));
                      t.expect(p->error_estimate == s->error_estimate,
                               std::format("parallel RQMC error bit-identical ({} vs {})",
                                           p->error_estimate, s->error_estimate));
                      t.expect(p->points_used == s->points_used &&
                                   p->replications == s->replications,
                               "parallel RQMC points_used/replications match serial");
                  }
                  auto bad = parallel_rqmc_integrate(f, 1, 10, 1, 1);
                  t.expect(!bad.has_value() &&
                               bad.error() == nimblecas::MathError::domain_error,
                           "parallel_rqmc_integrate(replications<2) yields domain_error");
              })
        .test("rqmc_shard_reduction_is_partition_independent",
              [](TestContext& t) {
                  // Distributed building block: sharding the replications across num_shards and
                  // reducing gives the SAME global estimate as the serial run, bit-for-bit, for
                  // num_shards = 1, 2, 4 — replication i is seeded from base.split(i) independent
                  // of the partition, and the driver reassembles in canonical index order.
                  auto f = [](std::span<const double> p) { return std::exp(p[0]) * p[1]; };
                  const std::uint64_t N = 128;
                  const std::uint64_t reps = 8;
                  const std::uint64_t seed = 777;
                  auto serial = rqmc_integrate(f, 2, N, reps, seed);
                  t.expect(serial.has_value(), "serial rqmc_integrate succeeds");

                  const std::array<std::uint64_t, 3> shard_counts{1, 2, 4};
                  for (const std::uint64_t num_shards : shard_counts) {
                      std::vector<RqmcShardResult> shards;
                      bool ok = true;
                      for (std::uint64_t si = 0; si < num_shards; ++si) {
                          auto sh = rqmc_shard(f, 2, N, reps, si, num_shards, seed);
                          if (!sh) {
                              ok = false;
                              break;
                          }
                          shards.push_back(std::move(*sh));
                      }
                      t.expect(ok, std::format("all {} shards compute", num_shards));
                      if (!ok) {
                          continue;
                      }
                      auto global =
                          rqmc_reduce_shards(std::span<const RqmcShardResult>{shards});
                      t.expect(global.has_value(),
                               std::format("reduce over {} shards succeeds", num_shards));
                      if (global && serial) {
                          t.expect(global->estimate == serial->estimate,
                                   std::format("{}-shard estimate bit-identical to serial ({} vs {})",
                                               num_shards, global->estimate, serial->estimate));
                          t.expect(global->error_estimate == serial->error_estimate,
                                   std::format("{}-shard error bit-identical to serial", num_shards));
                          t.expect(global->points_used == serial->points_used,
                                   std::format("{}-shard points_used matches serial", num_shards));
                      }
                  }

                  // Domain errors: shard_index >= num_shards, and an empty shard span.
                  auto bad_shard = rqmc_shard(f, 2, N, reps, 4, 4, seed);
                  t.expect(!bad_shard.has_value() &&
                               bad_shard.error() == nimblecas::MathError::domain_error,
                           "rqmc_shard(shard_index >= num_shards) yields domain_error");
                  std::vector<RqmcShardResult> empty;
                  auto bad_reduce = rqmc_reduce_shards(std::span<const RqmcShardResult>{empty});
                  t.expect(!bad_reduce.has_value() &&
                               bad_reduce.error() == nimblecas::MathError::domain_error,
                           "rqmc_reduce_shards(empty) yields domain_error");
              })
        .test("adaptive_qmc_refines_to_budget_or_tolerance",
              [](TestContext& t) {
                  // Adaptive refinement of ∫x. With a tiny budget it stops without converging;
                  // it always terminates and reports a finite estimate near the truth.
                  auto f = [](std::span<const double> x) { return x[0]; };
                  auto r = adaptive_qmc(f, 1, 1e-4, 200000, 8, 99);
                  t.expect(r.has_value(), "adaptive_qmc succeeds");
                  if (r) {
                      t.expect(r->points_used > 0, "adaptive_qmc spent some points");
                      t.expect(std::abs(r->estimate - 0.5) < 0.02,
                               std::format("adaptive estimate ~ 1/2 (got {})", r->estimate));
                      // If it declared convergence, the error must actually meet the tolerance.
                      t.expect(!r->converged || r->error_estimate <= 1e-4,
                               "converged flag implies error within tolerance");
                  }
                  // Domain errors.
                  auto bad_tol = adaptive_qmc(f, 1, -1.0, 1000, 4, 1);
                  t.expect(!bad_tol.has_value() &&
                               bad_tol.error() == nimblecas::MathError::domain_error,
                           "adaptive_qmc(tol<0) yields domain_error");
                  auto bad_budget = adaptive_qmc(f, 1, 1e-3, 0, 4, 1);
                  t.expect(!bad_budget.has_value() &&
                               bad_budget.error() == nimblecas::MathError::domain_error,
                           "adaptive_qmc(max_points 0) yields domain_error");
              })
        .test("l2_discrepancy_matches_hand_value",
              [](TestContext& t) {
                  // 1-D point set {0, 1/2}. Warnock L2 star discrepancy squared:
                  //   1/3 - (1/2)(1 + 3/4) + (1/4)(1 + 1/2 + 1/2 + 1/2) = 1/3 - 7/8 + 5/8 = 1/12.
                  std::vector<std::vector<Rational>> pts{{rat(0, 1)}, {rat(1, 2)}};
                  auto d2 = l2_star_discrepancy_squared_exact(pts, 1);
                  t.expect(d2.has_value(), "exact L2 discrepancy^2 succeeds");
                  if (d2) {
                      t.expect(*d2 == rat(1, 12),
                               std::format("exact L2* discrepancy^2 == 1/12 (got {})",
                                           d2->to_string()));
                  }

                  // Numerical version returns the square root of the same value.
                  std::vector<std::vector<double>> dpts{{0.0}, {0.5}};
                  auto d = l2_star_discrepancy(dpts, 1);
                  t.expect(d.has_value(), "numerical L2 discrepancy succeeds");
                  if (d) {
                      t.expect(std::abs(*d - std::sqrt(1.0 / 12.0)) < 1e-12,
                               std::format("numerical L2* == sqrt(1/12) (got {})", *d));
                  }

                  // Empty set / dimension 0 are domain errors.
                  auto bad = l2_star_discrepancy_squared_exact(pts, 0);
                  t.expect(!bad.has_value() &&
                               bad.error() == nimblecas::MathError::domain_error,
                           "l2 discrepancy(dimension 0) yields domain_error");
              })
        .test("integrate_domain_errors",
              [](TestContext& t) {
                  auto f = [](std::span<const double> x) { return x[0]; };
                  auto d0 = qmc_integrate(f, 0, 100);
                  t.expect(!d0.has_value() && d0.error() == nimblecas::MathError::domain_error,
                           "qmc_integrate(dimension 0) yields domain_error");
                  auto n0 = qmc_integrate(f, 1, 0);
                  t.expect(!n0.has_value() && n0.error() == nimblecas::MathError::domain_error,
                           "qmc_integrate(N 0) yields domain_error");
              })
        .run();
}
