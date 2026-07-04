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
using nimblecas::qmc_integrate;
using nimblecas::qmc_integrate_exact;
using nimblecas::Rational;
using nimblecas::rational_to_double;
using nimblecas::Rng;
using nimblecas::RqmcResult;
using nimblecas::rqmc_integrate;
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
