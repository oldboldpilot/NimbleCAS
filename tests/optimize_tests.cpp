// Tests for nimblecas.optimize: unconstrained numerical optimization over R^n.
// @author Olumuyiwa Oluwasanmi
//
// These are NUMERICAL tests: every comparison uses a generous absolute tolerance
// (local, IEEE-754, iteration-capped optimization makes no exactness claim). A
// convex quadratic (known minimizer A^{-1} b) is minimized by every method; the
// non-convex Rosenbrock is driven near (1,1) by BFGS / Newton / Nelder-Mead.

import std;
import nimblecas.core;
import nimblecas.optimize;
import nimblecas.testing;

using nimblecas::MathError;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;
namespace opt = nimblecas::optimize;

namespace {

auto close(double got, double expected, double tol) -> bool {
    return std::abs(got - expected) < tol;
}

// Convex quadratic q(x) = 1/2 x^T A x - b^T x with SPD A = [[3,1],[1,2]], b = [1,1].
// Gradient A x - b; unique minimizer x* = A^{-1} b = (0.2, 0.4).
constexpr std::array<double, 4> kA{3.0, 1.0, 1.0, 2.0};  // row-major 2x2.
constexpr std::array<double, 2> kB{1.0, 1.0};
constexpr std::array<double, 2> kXstar{0.2, 0.4};

auto quad_f(std::span<const double> x) -> double {
    // 1/2 x^T A x - b^T x.
    const double ax0 = kA[0] * x[0] + kA[1] * x[1];
    const double ax1 = kA[2] * x[0] + kA[3] * x[1];
    return 0.5 * (x[0] * ax0 + x[1] * ax1) - (kB[0] * x[0] + kB[1] * x[1]);
}

auto quad_grad(std::span<const double> x) -> std::vector<double> {
    return {kA[0] * x[0] + kA[1] * x[1] - kB[0], kA[2] * x[0] + kA[3] * x[1] - kB[1]};
}

// Rosenbrock f(x,y) = (1-x)^2 + 100 (y - x^2)^2, minimizer (1,1), f* = 0.
auto rosen_f(std::span<const double> x) -> double {
    const double a = 1.0 - x[0];
    const double b = x[1] - x[0] * x[0];
    return a * a + 100.0 * b * b;
}

auto rosen_grad(std::span<const double> x) -> std::vector<double> {
    const double b = x[1] - x[0] * x[0];
    return {-2.0 * (1.0 - x[0]) - 400.0 * x[0] * b, 200.0 * b};
}

// Euclidean distance from a 2-vector to the quadratic minimizer x* = (0.2, 0.4).
auto dist_to_xstar(std::span<const double> x) -> double {
    const double dx = x[0] - kXstar[0];
    const double dy = x[1] - kXstar[1];
    return std::sqrt(dx * dx + dy * dy);
}

// Convex quadratic + a low-amplitude, high-frequency "noise" term. A plain finite-
// difference gradient (tiny step) sees the noise slope ~ a*w and stalls; implicit
// filtering's large-scale stencil averages the oscillation away.
auto quad_noisy(std::span<const double> x) -> double {
    return quad_f(x) + 2e-3 * std::sin(40.0 * x[0]) * std::sin(40.0 * x[1]);
}

// Heavy, high-frequency noise: at every stencil scale here the noise dominates, forcing
// repeated stencil failures + scale shrinks (used to exercise that path).
auto quad_heavy_noise(std::span<const double> x) -> double {
    return quad_f(x) + 0.1 * std::sin(1000.0 * x[0]);
}

// Tilted 1-D quartic double well: f(t) = 0.1 t^4 - t^2 + 0.2 t. Two local minima (a
// DEEPER left well near t ~ -2.285, a shallower right well near t ~ +2.19) separated by a
// local maximum near t ~ 0.1 — a clean multimodal test bed for multistart. The global
// minimizer is the left well; a start's basin decides which well its local run reaches.
auto multiwell_f(std::span<const double> x) -> double {
    const double t = x[0];
    return 0.1 * t * t * t * t - t * t + 0.2 * t;
}
auto multiwell_grad(std::span<const double> x) -> std::vector<double> {
    const double t = x[0];
    return {0.4 * t * t * t - 2.0 * t + 0.2};
}

// Deterministic serial reference: argmin over per-start bfgs runs using the SAME order as
// multistart (strictly-lower fx wins; exact ties keep the earliest start index). Returns
// (best fx, winning start index) over the starts that produced a value.
auto serial_multistart_reference(std::span<const std::vector<double>> starts)
    -> std::pair<double, std::size_t> {
    double best_fx = std::numeric_limits<double>::infinity();
    std::size_t best_idx = 0;
    bool have = false;
    for (std::size_t i = 0; i < starts.size(); ++i) {
        auto r = opt::bfgs(multiwell_f, starts[i], multiwell_grad, opt::Options{});
        if (!r.has_value()) {
            continue;
        }
        if (!have || r->fx < best_fx) {  // strict '<' => earliest index wins exact ties.
            best_fx = r->fx;
            best_idx = i;
            have = true;
        }
    }
    return {best_fx, best_idx};
}

// Checks a quadratic minimizer result reached x* to a generous tolerance.
auto check_quad(TestContext& t, const nimblecas::Result<opt::OptimizeResult>& r,
                std::string_view name) -> void {
    t.expect(r.has_value(), std::string(name) + ": returns a value");
    if (!r.has_value()) {
        return;
    }
    t.expect(r->converged, std::string(name) + ": converged");
    t.expect(close(r->x[0], kXstar[0], 1e-4) && close(r->x[1], kXstar[1], 1e-4),
             std::string(name) + ": x ~ A^{-1} b = (0.2, 0.4)");
}

}  // namespace

auto main() -> int {
    const std::array<double, 2> quad_start{5.0, -3.0};
    const std::array<double, 2> rosen_start{-1.2, 1.0};

    return TestSuite("nimblecas.optimize")
        // --- convex quadratic: every method reaches A^{-1} b -------------------
        .test("gradient_descent_quadratic",
              [&](TestContext& t) {
                  opt::Options o;
                  o.grad_tol = 1e-9;
                  o.max_iterations = 5000;
                  check_quad(t, opt::gradient_descent(quad_f, quad_start, quad_grad, o),
                             "gradient_descent");
              })
        .test("newton_quadratic_fd_hessian",
              [&](TestContext& t) {
                  // Analytic gradient, finite-difference Hessian (hess left empty).
                  auto r = opt::newton_method(quad_f, quad_start, quad_grad, {}, opt::Options{});
                  check_quad(t, r, "newton");
                  // A quadratic Newton step from any start solves in ~1 iteration.
                  t.expect(r.has_value() && r->iterations <= 3,
                           "newton solves the quadratic in a couple of iterations");
              })
        .test("bfgs_quadratic",
              [&](TestContext& t) {
                  check_quad(t, opt::bfgs(quad_f, quad_start, quad_grad, opt::Options{}), "bfgs");
              })
        .test("lbfgs_quadratic",
              [&](TestContext& t) {
                  check_quad(t, opt::l_bfgs(quad_f, quad_start, quad_grad, opt::Options{}),
                             "l_bfgs");
              })
        .test("cg_quadratic_polak_ribiere",
              [&](TestContext& t) {
                  check_quad(t,
                             opt::conjugate_gradient(quad_f, quad_start, quad_grad,
                                                     opt::CGVariant::polak_ribiere, opt::Options{}),
                             "cg_pr");
              })
        .test("cg_quadratic_fletcher_reeves",
              [&](TestContext& t) {
                  check_quad(t,
                             opt::conjugate_gradient(quad_f, quad_start, quad_grad,
                                                     opt::CGVariant::fletcher_reeves,
                                                     opt::Options{}),
                             "cg_fr");
              })
        .test("nelder_mead_quadratic",
              [&](TestContext& t) {
                  opt::Options o;
                  o.grad_tol = 1e-12;
                  o.max_iterations = 5000;
                  auto r = opt::nelder_mead(quad_f, quad_start, o);
                  t.expect(r.has_value(), "nelder_mead: returns a value");
                  t.expect(r.has_value() && close(r->x[0], kXstar[0], 1e-3) &&
                               close(r->x[1], kXstar[1], 1e-3),
                           "nelder_mead: x ~ (0.2, 0.4)");
              })
        // --- gradient-free Nelder-Mead on a smooth bowl -----------------------
        .test("nelder_mead_bowl_no_gradient",
              [&](TestContext& t) {
                  // f = (x-3)^2 + (y+2)^2 + (z-1)^2, minimizer (3,-2,1). No gradient used.
                  auto bowl = [](std::span<const double> v) -> double {
                      return (v[0] - 3.0) * (v[0] - 3.0) + (v[1] + 2.0) * (v[1] + 2.0) +
                             (v[2] - 1.0) * (v[2] - 1.0);
                  };
                  const std::array<double, 3> s{0.0, 0.0, 0.0};
                  opt::Options o;
                  o.grad_tol = 1e-12;
                  o.max_iterations = 8000;
                  auto r = opt::nelder_mead(bowl, s, o);
                  t.expect(r.has_value(), "bowl: returns a value");
                  t.expect(r.has_value() && close(r->x[0], 3.0, 1e-3) &&
                               close(r->x[1], -2.0, 1e-3) && close(r->x[2], 1.0, 1e-3),
                           "bowl: x ~ (3, -2, 1)");
                  t.expect(r.has_value() && r->fx < 1e-6, "bowl: f ~ 0");
              })
        // --- Rosenbrock near (1,1) for BFGS / Newton / Nelder-Mead ------------
        .test("bfgs_rosenbrock",
              [&](TestContext& t) {
                  opt::Options o;
                  o.grad_tol = 1e-8;
                  o.max_iterations = 2000;
                  auto r = opt::bfgs(rosen_f, rosen_start, rosen_grad, o);
                  t.expect(r.has_value(), "bfgs rosenbrock: returns a value");
                  t.expect(r.has_value() && close(r->x[0], 1.0, 1e-3) && close(r->x[1], 1.0, 1e-3),
                           "bfgs rosenbrock: x ~ (1, 1)");
              })
        .test("newton_rosenbrock",
              [&](TestContext& t) {
                  opt::Options o;
                  o.grad_tol = 1e-8;
                  o.max_iterations = 2000;
                  // Analytic gradient + finite-difference Hessian, damped Newton.
                  auto r = opt::newton_method(rosen_f, rosen_start, rosen_grad, {}, o);
                  t.expect(r.has_value(), "newton rosenbrock: returns a value");
                  t.expect(r.has_value() && close(r->x[0], 1.0, 1e-3) && close(r->x[1], 1.0, 1e-3),
                           "newton rosenbrock: x ~ (1, 1)");
              })
        .test("nelder_mead_rosenbrock",
              [&](TestContext& t) {
                  opt::Options o;
                  o.grad_tol = 1e-10;
                  o.max_iterations = 8000;
                  auto r = opt::nelder_mead(rosen_f, rosen_start, o);
                  t.expect(r.has_value(), "nm rosenbrock: returns a value");
                  t.expect(r.has_value() && close(r->x[0], 1.0, 1e-2) && close(r->x[1], 1.0, 1e-2),
                           "nm rosenbrock: x ~ (1, 1)");
              })
        // --- finite-difference gradient matches the analytic gradient ----------
        .test("fd_gradient_matches_analytic",
              [&](TestContext& t) {
                  const std::array<double, 2> p{0.7, -0.4};
                  auto fd = opt::finite_difference_gradient(quad_f, p, 1e-6);
                  auto an = quad_grad(p);
                  t.expect(fd.size() == an.size(), "fd gradient has the right length");
                  t.expect(close(fd[0], an[0], 1e-5) && close(fd[1], an[1], 1e-5),
                           "central FD gradient ~ analytic A x - b");
              })
        // --- a capped run reports converged=false WITHOUT erroring -------------
        .test("capped_iterations_not_converged",
              [&](TestContext& t) {
                  opt::Options o;
                  o.max_iterations = 2;   // deliberately far too few for Rosenbrock.
                  o.grad_tol = 1e-12;
                  auto r = opt::bfgs(rosen_f, rosen_start, rosen_grad, o);
                  t.expect(r.has_value(), "capped run still returns a value (no error)");
                  t.expect(r.has_value() && !r->converged,
                           "capped run reports converged == false");
                  t.expect(r.has_value() && r->iterations <= 2, "iterations respected the cap");
              })
        // --- implicit filtering (derivative-free, noise-aware) -----------------
        .test("implicit_filtering_beats_fd_gradient_on_noise",
              [&](TestContext& t) {
                  const std::span<const double> nobound{};
                  opt::Options o;
                  o.grad_tol = 1e-6;
                  o.max_iterations = 3000;
                  auto imf = opt::implicit_filtering(quad_noisy, quad_start, nobound, nobound, o);
                  t.expect(imf.has_value(), "imf: returns a value on the noisy quadratic");
                  const double imf_dist = imf.has_value() ? dist_to_xstar(imf->x)
                                                          : std::numeric_limits<double>::max();
                  t.expect(imf_dist < 1.5e-2, "imf: locates the minimizer despite the noise");

                  // A plain FD-gradient descent (no analytic gradient) stalls: its FD
                  // gradient is corrupted by the noise slope, so it stops farther out.
                  opt::Options g;
                  g.grad_tol = 1e-6;
                  g.max_iterations = 5000;
                  auto gd = opt::gradient_descent(quad_noisy, quad_start, {}, g);
                  t.expect(gd.has_value(), "fd gradient descent still returns a value");
                  const double gd_dist = gd.has_value() ? dist_to_xstar(gd->x)
                                                        : std::numeric_limits<double>::max();
                  t.expect(gd_dist > imf_dist,
                           "fd gradient descent stalls farther from x* than implicit filtering");
              })
        .test("implicit_filtering_bound_constrained_boundary",
              [&](TestContext& t) {
                  // Box lo=(0.5,-1), hi=(2,1). The unconstrained min (0.2,0.4) lies OUTSIDE
                  // the box (x0 < 0.5); the constrained solution sits on the boundary
                  // x0 = 0.5 with x1 = 0.25 (KKT: grad_x1 = 0, grad_x0 >= 0 at the bound).
                  const std::array<double, 2> lo{0.5, -1.0};
                  const std::array<double, 2> hi{2.0, 1.0};
                  const std::array<double, 2> start{1.5, 0.0};
                  opt::Options o;
                  o.grad_tol = 1e-6;
                  o.max_iterations = 3000;
                  auto r = opt::implicit_filtering(quad_f, start, lo, hi, o);
                  t.expect(r.has_value(), "bounded imf: returns a value");
                  t.expect(r.has_value() && close(r->x[0], 0.5, 1e-2),
                           "bounded imf: solution on the x0 = 0.5 boundary");
                  t.expect(r.has_value() && close(r->x[1], 0.25, 1e-2),
                           "bounded imf: x1 = 0.25 at the boundary optimum");
                  // Feasibility of the returned point.
                  t.expect(r.has_value() && r->x[0] >= 0.5 - 1e-9 && r->x[0] <= 2.0 + 1e-9 &&
                               r->x[1] >= -1.0 - 1e-9 && r->x[1] <= 1.0 + 1e-9,
                           "bounded imf: returned point is feasible");
              })
        .test("implicit_filtering_stencil_failure_shrinks_scale",
              [&](TestContext& t) {
                  // Heavy noise starting at the true minimizer: no scale yields a
                  // productive step, so h must shrink through the schedule and the run
                  // ends WITHOUT error, reporting converged == false but staying near x*.
                  const std::span<const double> nobound{};
                  opt::Options o;
                  o.grad_tol = 1e-6;
                  o.imf_h0 = 0.5;
                  o.imf_h_min = 1e-8;
                  o.max_iterations = 500;
                  o.max_line_search = 5;  // bound alpha away from a step_tol-sized step.
                  const std::array<double, 2> at_min{0.2, 0.4};
                  auto r = opt::implicit_filtering(quad_heavy_noise, at_min, nobound, nobound, o);
                  t.expect(r.has_value(), "stencil-failure run returns a value (no error)");
                  t.expect(r.has_value() && !r->converged,
                           "heavy noise -> cannot meet grad_tol -> converged == false");
                  t.expect(r.has_value() && dist_to_xstar(r->x) < 1e-1,
                           "stencil-failure run stays in the neighborhood of x*");
              })
        .test("implicit_filtering_domain_errors",
              [&](TestContext& t) {
                  const std::span<const double> nobound{};
                  std::span<const double> empty{};
                  auto re = opt::implicit_filtering(quad_f, empty, nobound, nobound, opt::Options{});
                  t.expect(!re.has_value() && re.error() == MathError::domain_error,
                           "empty x0 -> domain_error");

                  const std::array<double, 2> nan_x{std::numeric_limits<double>::quiet_NaN(), 1.0};
                  auto rn = opt::implicit_filtering(quad_f, nan_x, nobound, nobound, opt::Options{});
                  t.expect(!rn.has_value() && rn.error() == MathError::domain_error,
                           "NaN in x0 -> domain_error");

                  // lo > hi is an invalid box.
                  const std::array<double, 2> start{1.0, 1.0};
                  const std::array<double, 2> bad_lo{2.0, 0.0};
                  const std::array<double, 2> bad_hi{1.0, 1.0};  // bad_lo[0] > bad_hi[0].
                  auto rb = opt::implicit_filtering(quad_f, start, bad_lo, bad_hi, opt::Options{});
                  t.expect(!rb.has_value() && rb.error() == MathError::domain_error,
                           "lo > hi -> domain_error");

                  // Ragged bounds (length mismatch with x0).
                  const std::array<double, 1> short_lo{0.0};
                  const std::array<double, 2> ok_hi{2.0, 2.0};
                  auto rr = opt::implicit_filtering(quad_f, start, short_lo, ok_hi, opt::Options{});
                  t.expect(!rr.has_value() && rr.error() == MathError::domain_error,
                           "bounds length mismatch -> domain_error");
              })
        // --- domain errors -----------------------------------------------------
        .test("empty_x0_domain_error",
              [&](TestContext& t) {
                  std::span<const double> empty{};
                  auto r = opt::bfgs(quad_f, empty, quad_grad, opt::Options{});
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "empty x0 -> domain_error");
                  auto rn = opt::nelder_mead(quad_f, empty, opt::Options{});
                  t.expect(!rn.has_value() && rn.error() == MathError::domain_error,
                           "empty x0 (nelder_mead) -> domain_error");
              })
        .test("non_finite_x0_domain_error",
              [&](TestContext& t) {
                  const std::array<double, 2> bad{std::numeric_limits<double>::quiet_NaN(), 1.0};
                  auto r = opt::gradient_descent(quad_f, bad, quad_grad, opt::Options{});
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "NaN in x0 -> domain_error");
                  const std::array<double, 2> inf{std::numeric_limits<double>::infinity(), 0.0};
                  auto ri = opt::newton_method(quad_f, inf, quad_grad, {}, opt::Options{});
                  t.expect(!ri.has_value() && ri.error() == MathError::domain_error,
                           "Inf in x0 -> domain_error");
              })
        // --- PARALLEL multistart: deterministic best == serial argmin ----------
        .test("parallel_multistart_matches_serial_best",
              [&](TestContext& t) {
                  // Starts straddling both wells of the tilted quartic: some fall into the
                  // deeper LEFT (global) well, some into the shallower right well.
                  const std::vector<std::vector<double>> starts{
                      {2.5}, {1.8}, {-0.5}, {-2.5}, {0.5}, {3.0}, {-1.0}};
                  auto local = opt::make_local_optimizer(opt::Method::bfgs, multiwell_f,
                                                         multiwell_grad);

                  auto par = opt::parallel_multistart(local, starts, /*parallel=*/true);
                  t.expect(par.has_value(), "parallel_multistart: returns a value");

                  const auto [ref_fx, ref_idx] = serial_multistart_reference(starts);
                  // Every start is deterministic, so the parallel winner is bit-identical
                  // to the serial in-order argmin: same fx, same start index.
                  t.expect(par.has_value() && par->best.fx == ref_fx,
                           "parallel best fx == serial single-start argmin fx (exact)");
                  t.expect(par.has_value() && par->best_start == ref_idx,
                           "parallel best_start == serial argmin start index");
                  t.expect(par.has_value() && par->succeeded == starts.size(),
                           "all starts produced a value");
                  // The global minimizer is the left well ~ -2.285.
                  t.expect(par.has_value() && close(par->best.x[0], -2.285, 5e-2),
                           "parallel multistart located the deeper (global) left well");

                  // parallel == false must reproduce the identical winner.
                  auto ser = opt::parallel_multistart(local, starts, /*parallel=*/false);
                  t.expect(ser.has_value() && par.has_value() &&
                               ser->best.fx == par->best.fx && ser->best_start == par->best_start,
                           "serial execution reproduces the parallel winner exactly");

                  // The multistart(Method,...) convenience overload agrees.
                  auto via_enum = opt::multistart(opt::Method::bfgs, multiwell_f, starts,
                                                  multiwell_grad);
                  t.expect(via_enum.has_value() && par.has_value() &&
                               via_enum->best_start == par->best_start &&
                               via_enum->best.fx == par->best.fx,
                           "multistart(Method,...) overload matches parallel_multistart");
              })
        // --- DISTRIBUTED shards: partition independence (num_shards = 1 vs 2) ---
        .test("multistart_shard_partition_independence",
              [&](TestContext& t) {
                  const std::vector<std::vector<double>> starts{
                      {2.5}, {1.8}, {-0.5}, {-2.5}, {0.5}, {3.0}, {-1.0}};
                  auto local = opt::make_local_optimizer(opt::Method::bfgs, multiwell_f,
                                                         multiwell_grad);

                  // num_shards = 1: a single shard covers every start.
                  auto s1 = opt::multistart_shard(local, starts, /*shard_index=*/0,
                                                  /*num_shards=*/1);
                  t.expect(s1.has_value(), "shard(0/1): returns a value");
                  std::array<opt::MultistartResult, 1> one{};
                  if (s1.has_value()) {
                      one = {*s1};
                  }
                  auto whole = opt::reduce_shards(one);
                  t.expect(whole.has_value(), "reduce over 1 shard: returns a value");

                  // num_shards = 2: shard 0 gets even indices, shard 1 the odd ones.
                  auto a = opt::multistart_shard(local, starts, 0, 2);
                  auto b = opt::multistart_shard(local, starts, 1, 2);
                  t.expect(a.has_value() && b.has_value(), "both 2-way shards return values");
                  std::array<opt::MultistartResult, 2> parts{};
                  if (a.has_value() && b.has_value()) {
                      parts = {*a, *b};
                  }
                  auto two = opt::reduce_shards(parts);
                  t.expect(two.has_value(), "reduce over 2 shards: returns a value");

                  // Partition independence: 1-shard and 2-shard reductions agree exactly,
                  // and both equal a plain single-process parallel_multistart.
                  auto full = opt::parallel_multistart(local, starts);
                  const bool all = whole.has_value() && two.has_value() && full.has_value();
                  t.expect(all && whole->best_start == two->best_start &&
                               whole->best.fx == two->best.fx,
                           "num_shards = 1 and num_shards = 2 select the identical best");
                  t.expect(all && two->best_start == full->best_start &&
                               two->best.fx == full->best.fx,
                           "sharded reduction == single-process parallel_multistart");
                  t.expect(all && whole->succeeded == starts.size() &&
                               two->succeeded == starts.size(),
                           "aggregated `succeeded` is partition-independent");
              })
        // --- deterministic tie-break: exact fx tie -> lowest start index -------
        .test("multistart_tie_break_lowest_start_index",
              [&](TestContext& t) {
                  // A synthetic local optimizer with EXACT fx ties: fx = x0^2, x = start.
                  // Starts {3, -1, 1, -3} -> fx {9, 1, 1, 9}: the minimum fx = 1 is hit by
                  // BOTH index 1 and index 2, so the canonical tie-break must pick index 1.
                  opt::LocalOptimizer square_opt =
                      [](std::span<const double> x0) -> nimblecas::Result<opt::OptimizeResult> {
                      opt::OptimizeResult r;
                      r.x.assign(x0.begin(), x0.end());
                      r.fx = x0[0] * x0[0];
                      r.iterations = 0;
                      r.converged = true;
                      r.grad_norm = 0.0;
                      return r;
                  };
                  const std::vector<std::vector<double>> starts{{3.0}, {-1.0}, {1.0}, {-3.0}};

                  auto par = opt::parallel_multistart(square_opt, starts, /*parallel=*/true);
                  t.expect(par.has_value() && par->best.fx == 1.0 && par->best_start == 1,
                           "exact fx tie resolves to the LOWEST start index (parallel)");
                  auto ser = opt::parallel_multistart(square_opt, starts, /*parallel=*/false);
                  t.expect(ser.has_value() && ser->best_start == 1,
                           "tie-break is thread-count independent (serial matches)");

                  // Same tie-break holds through the distributed path: with num_shards = 2,
                  // indices 1 and 2 land on different shards, yet the global reduction still
                  // selects index 1.
                  auto a = opt::multistart_shard(square_opt, starts, 0, 2);
                  auto b = opt::multistart_shard(square_opt, starts, 1, 2);
                  t.expect(a.has_value() && b.has_value(), "tie-break shards return values");
                  if (a.has_value() && b.has_value()) {
                      std::array<opt::MultistartResult, 2> parts{*a, *b};
                      auto red = opt::reduce_shards(parts);
                      t.expect(red.has_value() && red->best_start == 1,
                               "sharded reduction preserves the lowest-index tie-break");
                  }
              })
        // --- empty starts -> domain_error; all-invalid starts -> error ---------
        .test("multistart_domain_errors",
              [&](TestContext& t) {
                  auto local = opt::make_local_optimizer(opt::Method::bfgs, multiwell_f,
                                                         multiwell_grad);
                  const std::span<const std::vector<double>> no_starts{};
                  auto e = opt::parallel_multistart(local, no_starts);
                  t.expect(!e.has_value() && e.error() == MathError::domain_error,
                           "empty starts -> domain_error");

                  // Every start invalid (empty x0) -> the per-start domain_error propagates.
                  const std::vector<std::vector<double>> bad_starts{{}, {}};
                  auto eb = opt::parallel_multistart(local, bad_starts);
                  t.expect(!eb.has_value() && eb.error() == MathError::domain_error,
                           "all starts invalid -> domain_error propagated");

                  // Shard indexing guards.
                  auto s = opt::multistart_shard(local, bad_starts, /*shard_index=*/2,
                                                 /*num_shards=*/2);
                  t.expect(!s.has_value() && s.error() == MathError::domain_error,
                           "shard_index >= num_shards -> domain_error");
                  const std::span<const opt::MultistartResult> none{};
                  auto rr = opt::reduce_shards(none);
                  t.expect(!rr.has_value() && rr.error() == MathError::domain_error,
                           "reduce over zero shards -> domain_error");
              })
        .run();
}
