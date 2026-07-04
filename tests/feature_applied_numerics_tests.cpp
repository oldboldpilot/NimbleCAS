// Feature/integration tests: applied numerics across modules (interpolation, spectral,
// pdenum, qmc, extrapolation, optimize, nlsolve, control, stochastic).
// @author Olumuyiwa Oluwasanmi
//
// These are CROSS-MODULE invariants, not per-module unit tests: two independent EXACT
// routes to the same object must agree over Q, and each numerical method is checked
// against the value another discretisation / the exact answer prescribes. Concretely:
//   * interpolation vs spectral: the exact polynomial interpolant of samples of x^2 IS
//     x^2, and its Legendre/Chebyshev spectral coefficients (nimblecas.spectral) are the
//     known exact rationals — two exact routes agree.
//   * pdenum FDM vs FEM: for -u''=2 on [0,1], u(0)=u(1)=0 the 3-point finite-difference
//     stencil and the P1 finite-element solve both reproduce the exact nodal values of
//     u = x - x^2 (exact over Q), and they agree with each other.
//   * spectral Galerkin: -u''=2 on (-1,1), u(±1)=0 -> u = 1 - x^2 exactly over Q.
//   * qmc: qmc_integrate_exact over the Halton points is an exact rational (matches a
//     hand-rolled exact average); qmc_integrate of a smooth f converges (numeric).
//   * extrapolation: romberg_exact of ∫_0^1 x^2 dx == 1/3 exactly over Q; the numerical
//     Romberg's Richardson corner beats the raw trapezoid on a smooth integrand.
//   * optimize vs nlsolve: minimising a convex quadratic f = 1/2 x^T A x - b^T x yields
//     x* = A^{-1} b, and rooting its gradient grad f = A x - b yields the SAME x*.
//   * control + stochastic: a Hurwitz characteristic polynomial agrees across the control
//     stability criteria (Hurwitz determinant, companion Routh, Lyapunov); the Markov
//     chain P=[[3/4,1/4],[1/2,1/2]] has stationary pi=[2/3,1/3] exactly, with pi P == pi.
//
// Exact equality is used wherever the mathematics is exact over Q; honest, never-flaky
// tolerances are used for the genuinely numerical (double-precision) checks.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.interpolation;
import nimblecas.spectral;
import nimblecas.pdenum;
import nimblecas.qmc;
import nimblecas.extrapolation;
import nimblecas.optimize;
import nimblecas.nlsolve;
import nimblecas.control;
import nimblecas.stochastic;
import nimblecas.dynamics;
import nimblecas.testing;

using nimblecas::Matrix;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::Result;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace opt = nimblecas::optimize;
namespace nls = nimblecas::nlsolve;

namespace {

// --- concise builders -------------------------------------------------------

// The integer v as a Rational v/1.
[[nodiscard]] auto ri(std::int64_t v) -> Rational { return Rational::from_int(v); }

// A reduced fraction n/d (used only where n/d is known valid).
[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

// A dense rational matrix from a grid of Rationals.
[[nodiscard]] auto ratmat(std::vector<std::vector<Rational>> g) -> Matrix {
    return Matrix::from_rows(std::move(g)).value();
}

}  // namespace

auto main() -> int {
    return TestSuite("feature.applied_numerics")
        // ===================================================================
        // interpolation vs spectral — two EXACT routes to x^2 agree over Q.
        // ===================================================================
        .test("interp_vs_spectral_x_squared",
              [](TestContext& t) {
                  // Samples of x^2 at three distinct rational nodes.
                  const std::vector<Rational> nodes{ri(-1), ri(0), ri(1)};
                  const std::vector<Rational> vals{ri(1), ri(0), ri(1)};
                  const RationalPoly x2 = RationalPoly::monomial(ri(1), 2);  // x^2

                  auto lag = nimblecas::lagrange_polynomial(nodes, vals);
                  auto nwt = nimblecas::newton_polynomial(nodes, vals);
                  t.expect(lag.has_value() && nwt.has_value(),
                           "Lagrange and Newton interpolants build");
                  if (lag) {
                      t.expect(lag->is_equal(x2),
                               "the exact interpolant of x^2 samples == x^2 exactly over Q");
                  }
                  if (lag && nwt) {
                      t.expect(lag->is_equal(*nwt),
                               "Lagrange == Newton (the unique interpolant, two constructions)");
                  }

                  // A third interpolation route (Neville, value-only): p(2) = 4 exactly.
                  auto nev = nimblecas::neville_evaluate(nodes, vals, ri(2));
                  t.expect(nev.has_value() && *nev == ri(4),
                           "Neville evaluation of the interpolant at x=2 is 4 exactly");

                  // Spectral route: the interpolant's exact Legendre / Chebyshev coefficients.
                  if (lag) {
                      auto leg = nimblecas::legendre_coefficients(*lag);
                      t.expect(leg.has_value() && leg->size() == 3,
                               "Legendre transform of x^2 has coefficients c_0..c_2");
                      if (leg && leg->size() == 3) {
                          t.expect((*leg)[0] == rat(1, 3) && (*leg)[1] == ri(0) &&
                                       (*leg)[2] == rat(2, 3),
                                   "x^2 = (1/3) P_0 + (2/3) P_2 — exact rational coefficients");
                          auto back = nimblecas::legendre_from_coefficients(*leg);
                          t.expect(back.has_value() && back->is_equal(x2),
                                   "inverse Legendre transform reconstructs x^2 exactly");
                      }
                      auto cheb = nimblecas::chebyshev_coefficients(*lag);
                      t.expect(cheb.has_value() && cheb->size() == 3,
                               "Chebyshev transform of x^2 has coefficients c_0..c_2");
                      if (cheb && cheb->size() == 3) {
                          t.expect((*cheb)[0] == rat(1, 2) && (*cheb)[1] == ri(0) &&
                                       (*cheb)[2] == rat(1, 2),
                                   "x^2 = (1/2) T_0 + (1/2) T_2 — exact rational coefficients");
                          auto back = nimblecas::chebyshev_from_coefficients(*cheb);
                          t.expect(back.has_value() && back->is_equal(x2),
                                   "inverse Chebyshev transform reconstructs x^2 exactly");
                      }
                  }
              })
        // ===================================================================
        // pdenum FDM vs FEM — both reproduce u = x - x^2 exactly over Q.
        // ===================================================================
        .test("pdenum_fdm_vs_fem_poisson",
              [](TestContext& t) {
                  // -u'' = 2 on [0,1], u(0)=u(1)=0 has the exact solution u = x - x^2.
                  constexpr std::size_t N = 4;  // nodes 0, 1/4, 1/2, 3/4, 1
                  const auto bc0 = nimblecas::BoundaryCondition::dirichlet(ri(0));
                  // Exact nodal values u_i = x_i - x_i^2.
                  const std::vector<Rational> exact{ri(0), rat(3, 16), rat(1, 4), rat(3, 16),
                                                     ri(0)};

                  // Finite differences: 3-point stencil, f sampled at every node (= 2).
                  auto grid = nimblecas::Grid1D::make(ri(0), ri(1), N);
                  t.expect(grid.has_value(), "Grid1D::make succeeds");
                  const std::vector<Rational> f_nodal(N + 1, ri(2));
                  auto fd = grid ? nimblecas::solve_poisson_1d(*grid, f_nodal, bc0, bc0)
                                 : Result<std::vector<Rational>>{};

                  // Finite elements: P1 Galerkin, -(a u')' + c u = f with a=1, c=0, f=2.
                  auto mesh = nimblecas::Mesh1D::uniform(ri(0), ri(1), N);
                  const RationalPoly f2 = RationalPoly::constant(ri(2));
                  auto fe = mesh ? nimblecas::fem_p1_solve(*mesh, ri(1), ri(0), f2, bc0, bc0)
                                 : Result<std::vector<Rational>>{};

                  t.expect(fd.has_value() && fe.has_value(),
                           "both the FD and the FEM solves succeed");

                  if (fd) {
                      bool ok = fd->size() == N + 1;
                      for (std::size_t i = 0; i < exact.size() && i < fd->size(); ++i) {
                          if (!((*fd)[i] == exact[i])) {
                              ok = false;
                          }
                      }
                      t.expect(ok,
                               "FD 3-point stencil reproduces u = x - x^2 at every node exactly");
                  }
                  if (fe) {
                      bool ok = fe->size() == N + 1;
                      for (std::size_t i = 0; i < exact.size() && i < fe->size(); ++i) {
                          if (!((*fe)[i] == exact[i])) {
                              ok = false;
                          }
                      }
                      t.expect(ok, "P1 FEM reproduces u = x - x^2 at every node exactly");
                  }
                  if (fd && fe && fd->size() == fe->size()) {
                      bool same = true;
                      for (std::size_t i = 1; i + 1 < fd->size(); ++i) {  // interior nodes
                          if (!((*fd)[i] == (*fe)[i])) {
                              same = false;
                          }
                      }
                      t.expect(same,
                               "FDM and FEM nodal solutions agree exactly at interior nodes");
                  }
              })
        // ===================================================================
        // spectral Galerkin — -u''=2, u(±1)=0 -> u = 1 - x^2 exactly over Q.
        // ===================================================================
        .test("spectral_galerkin_poisson_exact",
              [](TestContext& t) {
                  const RationalPoly f2 = RationalPoly::constant(ri(2));
                  auto u = nimblecas::galerkin_poisson(f2);
                  t.expect(u.has_value(), "galerkin_poisson solves -u''=2, u(±1)=0");
                  const RationalPoly expected =
                      RationalPoly::from_coeffs({ri(1), ri(0), ri(-1)});  // 1 - x^2
                  if (u) {
                      t.expect(u->is_equal(expected),
                               "Legendre spectral-Galerkin gives u = 1 - x^2 exactly over Q");
                  }
              })
        // ===================================================================
        // qmc — exact rational average, and numerical convergence of a smooth f.
        // ===================================================================
        .test("qmc_exact_and_numeric_convergence",
              [](TestContext& t) {
                  constexpr std::size_t dim = 1;
                  constexpr std::uint64_t Nq = 8;

                  // EXACT: the equal-weight average of the first coordinate over the exact
                  // Halton points is an exact Rational — verify it against a hand-rolled sum.
                  nimblecas::RationalField fid =
                      [](std::span<const Rational> p) -> Result<Rational> { return p[0]; };
                  auto exact = nimblecas::qmc_integrate_exact(fid, dim, Nq);
                  t.expect(exact.has_value(), "qmc_integrate_exact succeeds");

                  Rational sum{};  // 0/1
                  bool manual_ok = true;
                  for (std::uint64_t n = 1; n <= Nq; ++n) {
                      auto pt = nimblecas::halton_point_rational(n, dim);
                      if (!pt) {
                          manual_ok = false;
                          break;
                      }
                      auto s = sum.add((*pt)[0]);
                      if (!s) {
                          manual_ok = false;
                          break;
                      }
                      sum = *s;
                  }
                  auto avg = sum.divide(Rational::from_int(static_cast<std::int64_t>(Nq)));
                  if (exact) {
                      t.expect(manual_ok && avg.has_value() && *exact == *avg,
                               "qmc_integrate_exact == hand-rolled exact rational average");
                  }

                  // NUMERICAL: the QMC average of a smooth integrand converges to ∫ = 1/3.
                  nimblecas::ScalarField fsq = [](std::span<const double> p) {
                      return p[0] * p[0];
                  };
                  auto coarse = nimblecas::qmc_integrate(fsq, dim, 64);
                  auto fine = nimblecas::qmc_integrate(fsq, dim, 4096);
                  t.expect(coarse.has_value() && fine.has_value(), "qmc_integrate succeeds");
                  if (coarse && fine) {
                      const double truth = 1.0 / 3.0;
                      t.expect(std::abs(*fine - truth) < 5e-3,
                               std::format("QMC of x^2 ~ 1/3 at N=4096 (got {})", *fine));
                      t.expect(std::abs(*fine - truth) < std::abs(*coarse - truth),
                               std::format("QMC error shrinks with N: {} < {}",
                                           std::abs(*fine - truth), std::abs(*coarse - truth)));
                  }
              })
        // ===================================================================
        // extrapolation — exact Romberg over Q, and numerical Richardson gain.
        // ===================================================================
        .test("extrapolation_romberg_exact_and_richardson",
              [](TestContext& t) {
                  // EXACT: ∫_0^1 x^2 dx == 1/3 over Q.
                  nimblecas::ExactFunction fq =
                      [](const Rational& x) -> Result<Rational> { return x.multiply(x); };
                  auto rex = nimblecas::romberg_exact(fq, ri(0), ri(1), 3);
                  t.expect(rex.has_value(), "romberg_exact succeeds");
                  if (rex) {
                      t.expect(rex->best == rat(1, 3),
                               "romberg_exact of ∫_0^1 x^2 dx == 1/3 exactly over Q");
                  }

                  // NUMERICAL: Richardson accelerates the composite trapezoid on a smooth
                  // integrand — the extrapolated corner beats the finest raw trapezoid.
                  nimblecas::RealFunction fexp = [](double x) { return std::exp(x); };
                  auto rd = nimblecas::romberg(fexp, 0.0, 1.0, 5);
                  t.expect(rd.has_value(), "romberg (numeric) succeeds");
                  if (rd && !rd->table.empty()) {
                      const double truth = std::numbers::e - 1.0;
                      const double trap = rd->table.back().front();  // finest trapezoid T(h_5)
                      const double best = rd->best;                  // Richardson corner
                      t.expect(std::abs(best - truth) < std::abs(trap - truth),
                               std::format("Richardson accelerates: |best-truth|={} < "
                                           "|trap-truth|={}",
                                           std::abs(best - truth), std::abs(trap - truth)));
                      t.expect(std::abs(best - truth) < 1e-9,
                               std::format("Richardson-accelerated Romberg ~ e-1 (got {})",
                                           best));
                  }
              })
        // ===================================================================
        // optimize vs nlsolve — minimiser of a convex quadratic == root of its gradient.
        // ===================================================================
        .test("optimize_vs_nlsolve_convex_quadratic",
              [](TestContext& t) {
                  // f = 1/2 x^T A x - b^T x, A = [[3,1],[1,2]] SPD, b = (1,1).
                  // x* = A^{-1} b = (1/5, 2/5) = (0.2, 0.4).
                  constexpr double a00 = 3.0, a01 = 1.0, a10 = 1.0, a11 = 2.0;
                  constexpr double b0 = 1.0, b1 = 1.0;
                  constexpr double xs0 = 0.2, xs1 = 0.4;

                  opt::Objective f = [=](std::span<const double> x) -> double {
                      const double u = x[0];
                      const double v = x[1];
                      return 0.5 * (a00 * u * u + (a01 + a10) * u * v + a11 * v * v) - b0 * u -
                             b1 * v;
                  };
                  opt::Gradient g = [=](std::span<const double> x) -> std::vector<double> {
                      const double u = x[0];
                      const double v = x[1];
                      return {a00 * u + a01 * v - b0, a10 * u + a11 * v - b1};
                  };
                  nls::ResidualFn grad_root =
                      [=](std::span<const double> x) -> std::vector<double> {
                      const double u = x[0];
                      const double v = x[1];
                      return {a00 * u + a01 * v - b0, a10 * u + a11 * v - b1};
                  };

                  const std::array<double, 2> x0{0.0, 0.0};
                  opt::Options oopts;
                  oopts.grad_tol = 1e-12;

                  auto minimum = opt::bfgs(f, std::span<const double>{x0}, g, oopts);
                  auto root = nls::newton(grad_root, std::span<const double>{x0});
                  t.expect(minimum.has_value(), "optimize::bfgs succeeds");
                  t.expect(root.has_value(), "nlsolve::newton on grad f succeeds");

                  if (minimum && minimum->x.size() == 2) {
                      t.expect(std::abs(minimum->x[0] - xs0) < 1e-6 &&
                                   std::abs(minimum->x[1] - xs1) < 1e-6,
                               std::format("minimiser x* = A^-1 b = (0.2, 0.4) (got {}, {})",
                                           minimum->x[0], minimum->x[1]));
                  }
                  if (root && root->x.size() == 2) {
                      t.expect(std::abs(root->x[0] - xs0) < 1e-6 &&
                                   std::abs(root->x[1] - xs1) < 1e-6,
                               std::format("root of grad f = A x - b at (0.2, 0.4) (got {}, {})",
                                           root->x[0], root->x[1]));
                  }
                  if (minimum && root && minimum->x.size() == 2 && root->x.size() == 2) {
                      t.expect(std::abs(minimum->x[0] - root->x[0]) < 1e-6 &&
                                   std::abs(minimum->x[1] - root->x[1]) < 1e-6,
                               "optimize (min f) and nlsolve (root of grad f) agree on x*");
                  }
              })
        // ===================================================================
        // control + stochastic — stability criteria agree; Markov stationary law exact.
        // ===================================================================
        .test("control_and_stochastic_stability_and_markov",
              [](TestContext& t) {
                  // Stable characteristic polynomial p(s) = s^2 + 2s + 2 (roots -1 ± i).
                  const RationalPoly ps = RationalPoly::from_coeffs({ri(2), ri(2), ri(1)});
                  auto hz = nimblecas::is_hurwitz_stable(ps);
                  t.expect(hz.has_value() && *hz == true,
                           "s^2+2s+2 is Hurwitz-stable (determinant / minors test)");

                  auto tf = nimblecas::TransferFunction::make(RationalPoly::constant(ri(1)), ps);
                  t.expect(tf.has_value(), "transfer function 1/(s^2+2s+2) builds");
                  if (tf) {
                      auto cont = nimblecas::is_stable_continuous(*tf);
                      t.expect(cont.has_value() && *cont == true,
                               "is_stable_continuous (companion Routh) agrees: stable");
                      if (hz && cont) {
                          t.expect(*hz == *cont,
                                   "Hurwitz-determinant test == companion Routh-Hurwitz verdict");
                      }
                  }

                  // The companion matrix A of p(s): its Lyapunov and Routh verdicts must match.
                  const Matrix a = ratmat({{ri(0), ri(1)}, {ri(-2), ri(-2)}});
                  auto lyap = nimblecas::is_lyapunov_stable(a);
                  auto dyn = nimblecas::is_asymptotically_stable(a);
                  t.expect(lyap.has_value() && *lyap == true,
                           "Lyapunov: companion A is asymptotically stable");
                  t.expect(dyn.has_value() && *dyn == true,
                           "dynamics Routh-Hurwitz: companion A is asymptotically stable");
                  if (lyap && dyn) {
                      t.expect(*lyap == *dyn,
                               "Lyapunov equation and Routh-Hurwitz agree on A");
                  }

                  // Unstable cross-check: p_u(s) = s^2 - 2s + 2 (roots 1 ± i).
                  const RationalPoly pu = RationalPoly::from_coeffs({ri(2), ri(-2), ri(1)});
                  auto hzu = nimblecas::is_hurwitz_stable(pu);
                  auto tfu = nimblecas::TransferFunction::make(RationalPoly::constant(ri(1)), pu);
                  if (hzu && tfu) {
                      auto contu = nimblecas::is_stable_continuous(*tfu);
                      t.expect(*hzu == false && contu.has_value() && *contu == false,
                               "s^2-2s+2 is unstable across both control criteria");
                  }

                  // Markov chain P = [[3/4, 1/4], [1/2, 1/2]] -> stationary pi = (2/3, 1/3).
                  const Matrix p = ratmat({{rat(3, 4), rat(1, 4)}, {rat(1, 2), rat(1, 2)}});
                  auto stoch = nimblecas::is_stochastic(p);
                  t.expect(stoch.has_value() && *stoch == true, "P is row-stochastic over Q");
                  auto pi = nimblecas::stationary_distribution(p);
                  t.expect(pi.has_value(), "stationary distribution solves exactly");
                  if (pi && pi->size() == 2) {
                      t.expect((*pi)[0] == rat(2, 3) && (*pi)[1] == rat(1, 3),
                               "stationary pi = (2/3, 1/3) exactly over Q");
                      // pi P == pi, computed exactly component by component.
                      bool fixed = true;
                      for (std::size_t j = 0; j < 2; ++j) {
                          Rational acc{};  // 0/1
                          for (std::size_t i = 0; i < 2; ++i) {
                              auto term = (*pi)[i].multiply(p.at(i, j));
                              if (!term) {
                                  fixed = false;
                                  break;
                              }
                              auto s = acc.add(*term);
                              if (!s) {
                                  fixed = false;
                                  break;
                              }
                              acc = *s;
                          }
                          if (!(acc == (*pi)[j])) {
                              fixed = false;
                          }
                      }
                      t.expect(fixed, "pi P == pi exactly (left Perron eigenvector over Q)");
                  }
              })
        .run();
}
