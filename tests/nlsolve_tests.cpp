// Tests for nimblecas.nlsolve: Kelley-style iterative solvers for F(x)=0 in R^n.
// @author Olumuyiwa Oluwasanmi
//
// NUMERIC, LOCAL convergence: tolerances are absolute (1e-6 for value comparisons),
// results depend on the initial guess, and non-convergence is a converged=false result
// rather than an error. These tests exercise correctness on known roots, the quadratic
// local convergence of Newton, Armijo globalisation from a poor start, Anderson
// acceleration of a linear fixed point, Levenberg-Marquardt least squares, FD-vs-analytic
// Jacobian agreement, the capped-iteration contract, and the domain-error boundary.

import std;
import nimblecas.core;
import nimblecas.nlsolve;
import nimblecas.testing;

using nimblecas::MathError;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;
namespace nl = nimblecas::nlsolve;

namespace {

auto close(double got, double expected) -> bool {
    return std::abs(got - expected) < 1e-6;
}

// Residual map F(x) = [x^2 - 2] as a 1-D system; root = sqrt(2).
auto scalar_sq_minus_two(std::span<const double> x) -> std::vector<double> {
    return {x[0] * x[0] - 2.0};
}
auto scalar_sq_minus_two_jac(std::span<const double> x) -> std::vector<double> {
    return {2.0 * x[0]};  // 1x1 Jacobian
}

// 2x2 nonlinear system: F = [x0^2 + x1^2 - 2, x0 - x1]; root (1, 1).
auto system2(std::span<const double> x) -> std::vector<double> {
    return {x[0] * x[0] + x[1] * x[1] - 2.0, x[0] - x[1]};
}
auto system2_jac(std::span<const double> x) -> std::vector<double> {
    // Row-major 2x2: [ dF0/dx0, dF0/dx1, dF1/dx0, dF1/dx1 ].
    return {2.0 * x[0], 2.0 * x[1], 1.0, -1.0};
}

// arctan residual F(x) = [atan(x)]; root 0. Undamped Newton diverges for |x0| large.
auto atan_res(std::span<const double> x) -> std::vector<double> {
    return {std::atan(x[0])};
}
auto atan_jac(std::span<const double> x) -> std::vector<double> {
    return {1.0 / (1.0 + x[0] * x[0])};
}

// Fixed-point map g(x) = [cos(x)]; linearly convergent to the Dottie number ~0.7390851.
auto cos_map(std::span<const double> x) -> std::vector<double> {
    return {std::cos(x[0])};
}

// Residual F(x) = [x^2 - 1] as a 1-D system with TWO roots, +1 and -1; Newton converges to
// whichever basin the start falls in — used to exercise the multistart layer.
auto sq_minus_one(std::span<const double> x) -> std::vector<double> {
    return {x[0] * x[0] - 1.0};
}

// A Solver adaptor binding finite-difference Newton (the caller-selected method).
auto newton_solver(const nl::ResidualFn& F, std::span<const double> x0,
                   const nl::Options& o) -> nimblecas::Result<nl::SolveResult> {
    return nl::newton(F, x0, o);
}

}  // namespace

auto main() -> int {
    const double root2 = std::numbers::sqrt2;

    return TestSuite("nimblecas.nlsolve")
        .test("newton_scalar_sqrt2",
              [&](TestContext& t) {
                  const std::array<double, 1> x0{1.5};
                  auto r = nl::newton(scalar_sq_minus_two, scalar_sq_minus_two_jac, x0);
                  t.expect(r.has_value(), "newton returns a result");
                  t.expect(r.has_value() && r->converged, "converged on x^2 - 2");
                  t.expect(r.has_value() && close(r->x[0], root2), "root == sqrt(2)");
                  t.expect(r.has_value() && r->residual_norm < 1e-6, "||F|| ~ 0");
              })
        .test("newton_fd_scalar_sqrt2",
              [&](TestContext& t) {
                  // No analytic Jacobian: the finite-difference variant must still solve.
                  const std::array<double, 1> x0{1.5};
                  auto r = nl::newton(scalar_sq_minus_two, x0);
                  t.expect(r.has_value() && r->converged, "FD Newton converges");
                  t.expect(r.has_value() && close(r->x[0], root2), "FD root == sqrt(2)");
              })
        .test("newton_2d_system",
              [&](TestContext& t) {
                  const std::array<double, 2> x0{1.5, 1.5};
                  auto r = nl::newton(system2, system2_jac, x0);
                  t.expect(r.has_value() && r->converged, "converged on 2x2 system");
                  t.expect(r.has_value() && close(r->x[0], 1.0) && close(r->x[1], 1.0),
                           "root == (1, 1)");
              })
        .test("newton_quadratic_local_convergence",
              [&](TestContext& t) {
                  // Cap the iteration count and watch the residual: near a simple root
                  // Newton is quadratic, so r_{k+1} < r_k^2 once r_k < 1.
                  const std::array<double, 1> x0{1.5};
                  nl::Options o;
                  o.tol = 1e-15;  // force the full iteration budget, no early stop
                  o.max_iter = 1;
                  auto r1 = nl::newton(scalar_sq_minus_two, scalar_sq_minus_two_jac, x0, o);
                  o.max_iter = 2;
                  auto r2 = nl::newton(scalar_sq_minus_two, scalar_sq_minus_two_jac, x0, o);
                  o.max_iter = 3;
                  auto r3 = nl::newton(scalar_sq_minus_two, scalar_sq_minus_two_jac, x0, o);
                  t.expect(r1.has_value() && r2.has_value() && r3.has_value(),
                           "capped runs all return results");
                  const double a = r1->residual_norm;
                  const double b = r2->residual_norm;
                  const double c = r3->residual_norm;
                  t.expect(b < a * a, "residual drops quadratically (step 2)");
                  t.expect(c < b * b, "residual drops quadratically (step 3)");
                  t.expect(c < 1e-9, "three Newton steps reach ~1e-9 residual");
              })
        .test("broyden_2d_system",
              [&](TestContext& t) {
                  const std::array<double, 2> x0{1.5, 1.2};
                  auto rg = nl::broyden(system2, x0);  // good Broyden (default)
                  t.expect(rg.has_value() && rg->converged, "good Broyden converges");
                  t.expect(rg.has_value() && close(rg->x[0], 1.0) && close(rg->x[1], 1.0),
                           "good Broyden root == (1, 1)");
                  auto rb = nl::broyden(system2, x0, {}, nl::BroydenVariant::bad);
                  t.expect(rb.has_value() && rb->converged, "bad Broyden converges");
                  t.expect(rb.has_value() && close(rb->x[0], 1.0) && close(rb->x[1], 1.0),
                           "bad Broyden root == (1, 1)");
              })
        .test("chord_and_shamanskii_2d",
              [&](TestContext& t) {
                  const std::array<double, 2> x0{1.4, 1.1};
                  auto rc = nl::chord(system2, system2_jac, x0);
                  t.expect(rc.has_value() && rc->converged, "chord converges");
                  t.expect(rc.has_value() && close(rc->x[0], 1.0) && close(rc->x[1], 1.0),
                           "chord root == (1, 1)");
                  auto rs = nl::shamanskii(system2, system2_jac, x0, 2);
                  t.expect(rs.has_value() && rs->converged, "shamanskii(m=2) converges");
                  t.expect(rs.has_value() && close(rs->x[0], 1.0) && close(rs->x[1], 1.0),
                           "shamanskii root == (1, 1)");
              })
        .test("newton_krylov_jfnk_2d",
              [&](TestContext& t) {
                  const std::array<double, 2> x0{1.5, 1.2};
                  auto r = nl::newton_krylov(system2, x0);
                  t.expect(r.has_value() && r->converged, "JFNK converges on 2x2 system");
                  t.expect(r.has_value() && close(r->x[0], 1.0) && close(r->x[1], 1.0),
                           "JFNK root == (1, 1)");
              })
        .test("armijo_enables_convergence_from_poor_start",
              [&](TestContext& t) {
                  // atan(x) from x0 = 10: undamped Newton diverges; the line search saves it.
                  const std::array<double, 1> x0{10.0};
                  nl::Options plain;
                  plain.line_search = false;
                  plain.max_iter = 50;
                  auto rp = nl::newton(atan_res, atan_jac, x0, plain);
                  t.expect(rp.has_value() && !rp->converged,
                           "undamped Newton does NOT converge from x0=10");

                  nl::Options damped;  // line_search on by default
                  damped.max_iter = 100;
                  auto rd = nl::newton(atan_res, atan_jac, x0, damped);
                  t.expect(rd.has_value() && rd->converged,
                           "Armijo-globalised Newton converges");
                  t.expect(rd.has_value() && close(rd->x[0], 0.0), "root == 0");
              })
        .test("anderson_accelerates_linear_fixed_point",
              [&](TestContext& t) {
                  const double dottie = 0.7390851332151607;  // cos(x) = x
                  const std::array<double, 1> x0{0.0};
                  auto r = nl::anderson(cos_map, x0, 3);
                  t.expect(r.has_value() && r->converged, "Anderson converges on cos map");
                  t.expect(r.has_value() && close(r->x[0], dottie),
                           "fixed point == Dottie number");
                  // Plain Picard needs ~58 steps for 1e-10; Anderson should be far fewer.
                  t.expect(r.has_value() && r->iterations < 30,
                           "Anderson accelerates (well under Picard's ~58 iters)");
              })
        .test("levenberg_marquardt_exponential_fit",
              [&](TestContext& t) {
                  // Fit y = a exp(b t) to noiseless data from a=2, b=0.5 -> recover (2, 0.5).
                  const std::array<double, 5> ts{0.0, 0.5, 1.0, 1.5, 2.0};
                  std::array<double, 5> ys{};
                  for (std::size_t i = 0; i < ts.size(); ++i) {
                      ys[i] = 2.0 * std::exp(0.5 * ts[i]);
                  }
                  auto resid = [ts, ys](std::span<const double> p) -> std::vector<double> {
                      std::vector<double> r(ts.size());
                      for (std::size_t i = 0; i < ts.size(); ++i) {
                          r[i] = p[0] * std::exp(p[1] * ts[i]) - ys[i];
                      }
                      return r;
                  };
                  const std::array<double, 2> p0{1.0, 1.0};
                  nl::Options o;
                  o.tol = 1e-12;
                  o.max_iter = 200;
                  auto r = nl::levenberg_marquardt(resid, p0, o);
                  t.expect(r.has_value() && r->converged, "LM reaches a stationary point");
                  t.expect(r.has_value() && close(r->x[0], 2.0), "recovered a == 2");
                  t.expect(r.has_value() && close(r->x[1], 0.5), "recovered b == 0.5");
                  t.expect(r.has_value() && r->residual_norm < 1e-4, "residual ~ 0");
              })
        .test("fd_jacobian_matches_analytic",
              [&](TestContext& t) {
                  const std::array<double, 2> x{1.3, 0.7};
                  auto jfd = nl::finite_difference_jacobian(system2, x);
                  const auto ja = system2_jac(x);
                  t.expect(jfd.has_value() && jfd->size() == 4, "FD Jacobian is 2x2");
                  bool matched = jfd.has_value();
                  if (jfd.has_value()) {
                      for (std::size_t i = 0; i < 4; ++i) {
                          if (std::abs((*jfd)[i] - ja[i]) > 1e-4) {
                              matched = false;
                          }
                      }
                  }
                  t.expect(matched, "FD Jacobian within 1e-4 of analytic");
              })
        .test("capped_iterations_returns_not_converged",
              [&](TestContext& t) {
                  const std::array<double, 1> x0{1.5};
                  nl::Options o;
                  o.tol = 1e-15;
                  o.max_iter = 1;  // far too few for 1e-15
                  auto r = nl::newton(scalar_sq_minus_two, scalar_sq_minus_two_jac, x0, o);
                  t.expect(r.has_value(), "capped run is a valid result, not an error");
                  t.expect(r.has_value() && !r->converged, "converged == false at the cap");
                  t.expect(r.has_value() && r->iterations == 1, "used exactly the budget");
              })
        .test("domain_errors",
              [&](TestContext& t) {
                  // Non-finite initial guess.
                  const std::array<double, 1> bad{std::numeric_limits<double>::infinity()};
                  auto r1 = nl::newton(scalar_sq_minus_two, bad);
                  t.expect(!r1.has_value() && r1.error() == MathError::domain_error,
                           "non-finite x0 -> domain_error");

                  // Empty system.
                  std::span<const double> empty{};
                  auto r2 = nl::newton(scalar_sq_minus_two, empty);
                  t.expect(!r2.has_value() && r2.error() == MathError::domain_error,
                           "empty x0 -> domain_error");

                  // Residual whose output dimension disagrees with n.
                  auto wrong_dim = [](std::span<const double> x) -> std::vector<double> {
                      return {x[0], x[0]};  // returns 2 for a 1-D system
                  };
                  const std::array<double, 1> x0{1.0};
                  auto r3 = nl::newton(wrong_dim, x0);
                  t.expect(!r3.has_value() && r3.error() == MathError::domain_error,
                           "dimension mismatch -> domain_error");

                  // Non-finite point given to the public FD Jacobian.
                  auto r4 = nl::finite_difference_jacobian(scalar_sq_minus_two, bad);
                  t.expect(!r4.has_value() && r4.error() == MathError::domain_error,
                           "FD Jacobian at non-finite x -> domain_error");
              })
        .test("parallel_multistart_deterministic_best",
              [&](TestContext& t) {
                  // F(x) = x^2 - 1 has roots +1 and -1; different starts land in different
                  // basins. parallel_multistart must return the SAME best as a serial sweep
                  // that applies the (converged desc, residual asc, index asc) tie-break.
                  const std::vector<std::vector<double>> starts{
                      {3.0}, {-3.0}, {0.7}, {-0.4}, {5.0}};
                  const std::span<const std::vector<double>> ss{starts};
                  nl::Solver solver = newton_solver;
                  const nl::Options o;

                  auto par = nl::parallel_multistart(sq_minus_one, ss, o, solver);
                  t.expect(par.has_value(), "parallel_multistart returns a result");
                  t.expect(par.has_value() && par->converged, "best is a converged root");
                  t.expect(par.has_value() &&
                               (close(par->x[0], 1.0) || close(par->x[0], -1.0)),
                           "best sits on a genuine root (+/-1)");

                  // Serial reference with the identical deterministic tie-break.
                  bool have = false;
                  std::size_t bidx = 0;
                  double bres = 0.0;
                  bool bconv = false;
                  std::vector<double> bx;
                  for (std::size_t i = 0; i < starts.size(); ++i) {
                      auto r = nl::newton(sq_minus_one, std::span<const double>{starts[i]}, o);
                      if (!r.has_value()) {
                          continue;
                      }
                      bool better = false;
                      if (!have) {
                          better = true;
                      } else if (r->converged != bconv) {
                          better = r->converged;
                      } else if (r->residual_norm != bres) {
                          better = r->residual_norm < bres;
                      } else {
                          better = i < bidx;
                      }
                      if (better) {
                          have = true;
                          bidx = i;
                          bres = r->residual_norm;
                          bconv = r->converged;
                          bx = r->x;
                      }
                  }
                  t.expect(have, "serial reference sweep found a best");
                  bool matches = par.has_value() && have && par->converged == bconv &&
                                 par->residual_norm == bres && par->x.size() == bx.size();
                  if (matches) {
                      for (std::size_t i = 0; i < bx.size(); ++i) {
                          if (par->x[i] != bx[i]) {
                              matches = false;
                          }
                      }
                  }
                  t.expect(matches, "parallel best is bit-identical to the serial best");
              })
        .test("multistart_shard_partition_independence",
              [&](TestContext& t) {
                  // Reducing over shards must be independent of how the starts are split:
                  // num_shards = 1 and num_shards = 2 give the identical final result, and
                  // both equal the whole-set parallel_multistart.
                  const std::vector<std::vector<double>> starts{
                      {3.0}, {-3.0}, {0.7}, {-0.4}, {5.0}, {-2.5}};
                  const std::span<const std::vector<double>> ss{starts};
                  nl::Solver solver = newton_solver;
                  const nl::Options o;

                  // One shard covering everything.
                  const std::array<nl::MultistartResult, 1> one{
                      nl::multistart_shard(sq_minus_one, ss, 0, 1, o, solver)};
                  auto r1 = nl::reduce_multistart(std::span<const nl::MultistartResult>{one});

                  // Two shards: even indices vs odd indices.
                  const std::array<nl::MultistartResult, 2> two{
                      nl::multistart_shard(sq_minus_one, ss, 0, 2, o, solver),
                      nl::multistart_shard(sq_minus_one, ss, 1, 2, o, solver)};
                  auto r2 = nl::reduce_multistart(std::span<const nl::MultistartResult>{two});

                  t.expect(r1.has_value() && r2.has_value(),
                           "both shard reductions produce a result");
                  bool same = r1.has_value() && r2.has_value() &&
                              r1->converged == r2->converged &&
                              r1->residual_norm == r2->residual_norm &&
                              r1->x.size() == r2->x.size();
                  if (same) {
                      for (std::size_t i = 0; i < r1->x.size(); ++i) {
                          if (r1->x[i] != r2->x[i]) {
                              same = false;
                          }
                      }
                  }
                  t.expect(same, "num_shards=1 and num_shards=2 reduce to the same result");

                  // And both equal the single-process whole-set sweep.
                  auto par = nl::parallel_multistart(sq_minus_one, ss, o, solver);
                  bool eq = par.has_value() && r2.has_value() &&
                            par->residual_norm == r2->residual_norm &&
                            par->x.size() == r2->x.size();
                  if (eq) {
                      for (std::size_t i = 0; i < par->x.size(); ++i) {
                          if (par->x[i] != r2->x[i]) {
                              eq = false;
                          }
                      }
                  }
                  t.expect(eq, "shard reduction equals whole-set parallel_multistart");
              })
        .run();
}
