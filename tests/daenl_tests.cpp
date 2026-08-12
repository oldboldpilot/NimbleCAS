// Tests for nimblecas.daenl: nonlinear / higher-index differential-algebraic equations.
// @author Olumuyiwa Oluwasanmi
//
// Oracles are hand-verified in three tiers, matching the module's three honesty tiers:
//
//   TIER 1 (exact power series, solve_semiexplicit_nonlinear_series):
//     * x' = y, 0 = y - x, x(0)=y(0)=1  =>  x = y = e^t  (Taylor coeff k = 1/k!).
//     * x' = 1, 0 = y - x^2, x(0)=y(0)=0 =>  x = t, y = t^2 (exact finite series).
//     * x' = y, 0 = y - exp(t), x(0)=0, y(0)=1 => y = e^t, x = e^t - 1
//       (exercises the exp() branch of the Expr->PowerSeries evaluator).
//   Refusals: a non-constant dg/dy (x*y - 1), and an unsupported function (sin) in f,
//   are MathError::not_implemented, never a wrong series.
//
//   TIER 2 (structural index, analyze_structure):
//     * x' = y, y' = x is a pure ODE => structural_index == 1 (matches at sweep 0).
//     * a constraint system (x' = y, 0 = x - t) is the module's honest boundary: it
//       either reports an index with a verified matching, or MathError::not_implemented.
//
//   TIER 3 (numerical index-1 stepping, solve_nonlinear_dae / consistent_initial_values):
//     * x' = -x by implicit Euler with a single unit step gives EXACTLY x_1 = x_0/(1+h)
//       = 1/2 (h=1) -- an exact algebraic oracle for the scheme, not a loose tolerance.
//     * consistent_initial_values for x' = -x at x=2 gives xdot = -2 exactly.
//     * x' = -x^3 by implicit Euler (h=1) yields the state satisfying s + s^3 = 1
//       (the scheme's own defining equation -- an exact residual oracle).
//   Refusal: a non-index-1 system is MathError::not_implemented (the module refuses to
//   time-step a higher-index trajectory it cannot stabilise).

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.symbolic;
import nimblecas.powerseries;
import nimblecas.dae;
import nimblecas.daenl;
import nimblecas.testing;

using nimblecas::DaeSolution;
using nimblecas::Expr;
using nimblecas::MathError;
using nimblecas::Rational;
using nimblecas::daenl::analyze_structure;
using nimblecas::daenl::consistent_initial_values;
using nimblecas::daenl::DaeSystem;
using nimblecas::daenl::NlDaeOptions;
using nimblecas::daenl::solve_nonlinear_dae;
using nimblecas::daenl::solve_semiexplicit_nonlinear_series;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Compact Expr builders (the module treats every leaf literally; no simplification needed
// here beyond what the module itself performs).
auto sym(std::string name) -> Expr { return Expr::symbol(std::move(name)); }
auto lit(std::int64_t v) -> Expr { return Expr::integer(v); }
auto neg(Expr e) -> Expr { return Expr::product({Expr::integer(-1), std::move(e)}); }
auto sub(Expr a, Expr b) -> Expr { return Expr::sum({std::move(a), neg(std::move(b))}); }
auto mul(Expr a, Expr b) -> Expr { return Expr::product({std::move(a), std::move(b)}); }
auto pw(Expr b, std::int64_t e) -> Expr { return Expr::power(std::move(b), Expr::integer(e)); }
auto call(std::string name, Expr arg) -> Expr {
    return Expr::apply(std::move(name), {std::move(arg)});
}

auto factorial(std::int64_t k) -> std::int64_t {
    std::int64_t f = 1;
    for (std::int64_t i = 2; i <= k; ++i) {
        f *= i;
    }
    return f;
}

// A normalized rational coefficient equals num/den (the series engine keeps coefficients in
// lowest terms with a positive denominator, so the oracles below are all coprime).
auto rat_is(const Rational& r, std::int64_t num, std::int64_t den) -> bool {
    return r.numerator() == num && r.denominator() == den;
}

constexpr double kTightTol = 1e-12;
constexpr double kResidualTol = 1e-8;

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.daenl")
        // ---- TIER 1: exact power series -----------------------------------------------
        .test("semi-explicit x'=y, 0=y-x gives x = y = e^t exactly",
              [&](TestContext& t) {
                  const std::vector<Expr> f{sym("y")};
                  const std::vector<Expr> g{sub(sym("y"), sym("x"))};
                  const std::size_t order = 6;
                  auto sol = solve_semiexplicit_nonlinear_series(
                      f, g, {"x"}, {"y"}, "t", {Rational::from_int(1)}, {Rational::from_int(1)},
                      order);
                  t.expect(sol.has_value(), "solve succeeded");
                  if (!sol) return;
                  t.expect(sol->x.size() == 1 && sol->y.size() == 1, "one x and one y series");
                  if (sol->x.size() != 1 || sol->y.size() != 1) return;
                  const auto xc = sol->x[0].coefficients();
                  const auto yc = sol->y[0].coefficients();
                  t.expect(xc.size() == order && yc.size() == order, "series carry `order` terms");
                  if (xc.size() != order || yc.size() != order) return;
                  bool ok = true;
                  for (std::size_t k = 0; k < order; ++k) {
                      const std::int64_t fk = factorial(static_cast<std::int64_t>(k));
                      ok = ok && rat_is(xc[k], 1, fk) && rat_is(yc[k], 1, fk);
                  }
                  t.expect(ok, "every Taylor coefficient equals 1/k! for both x and y");
              })
        .test("semi-explicit x'=1, 0=y-x^2 gives x = t, y = t^2 exactly",
              [&](TestContext& t) {
                  const std::vector<Expr> f{lit(1)};
                  const std::vector<Expr> g{sub(sym("y"), pw(sym("x"), 2))};
                  const std::size_t order = 5;
                  auto sol = solve_semiexplicit_nonlinear_series(
                      f, g, {"x"}, {"y"}, "t", {Rational::from_int(0)}, {Rational::from_int(0)},
                      order);
                  t.expect(sol.has_value(), "solve succeeded");
                  if (!sol) return;
                  const auto xc = sol->x[0].coefficients();
                  const auto yc = sol->y[0].coefficients();
                  // x = t  => [0, 1, 0, 0, 0];  y = t^2 => [0, 0, 1, 0, 0].
                  const bool x_ok = rat_is(xc[0], 0, 1) && rat_is(xc[1], 1, 1) &&
                                    rat_is(xc[2], 0, 1) && rat_is(xc[3], 0, 1) &&
                                    rat_is(xc[4], 0, 1);
                  const bool y_ok = rat_is(yc[0], 0, 1) && rat_is(yc[1], 0, 1) &&
                                    rat_is(yc[2], 1, 1) && rat_is(yc[3], 0, 1) &&
                                    rat_is(yc[4], 0, 1);
                  t.expect(x_ok, "x is exactly t");
                  t.expect(y_ok, "y is exactly t^2");
              })
        .test("semi-explicit with exp(t) in the constraint gives y = e^t, x = e^t - 1",
              [&](TestContext& t) {
                  const std::vector<Expr> f{sym("y")};
                  const std::vector<Expr> g{sub(sym("y"), call("exp", sym("t")))};
                  const std::size_t order = 6;
                  auto sol = solve_semiexplicit_nonlinear_series(
                      f, g, {"x"}, {"y"}, "t", {Rational::from_int(0)}, {Rational::from_int(1)},
                      order);
                  t.expect(sol.has_value(), "solve succeeded (exp branch exercised)");
                  if (!sol) return;
                  const auto xc = sol->x[0].coefficients();
                  const auto yc = sol->y[0].coefficients();
                  bool ok = rat_is(xc[0], 0, 1);  // x(0) = 0
                  for (std::size_t k = 1; k < order; ++k) {
                      const std::int64_t fk = factorial(static_cast<std::int64_t>(k));
                      ok = ok && rat_is(xc[k], 1, fk);  // e^t - 1 shares e^t's k>=1 coeffs
                  }
                  for (std::size_t k = 0; k < order; ++k) {
                      const std::int64_t fk = factorial(static_cast<std::int64_t>(k));
                      ok = ok && rat_is(yc[k], 1, fk);  // y = e^t
                  }
                  t.expect(ok, "x = e^t - 1 and y = e^t coefficient-for-coefficient");
              })
        .test("non-constant dg/dy (x*y - 1) is honestly not_implemented",
              [&](TestContext& t) {
                  const std::vector<Expr> f{sym("y")};
                  const std::vector<Expr> g{sub(mul(sym("x"), sym("y")), lit(1))};
                  auto sol = solve_semiexplicit_nonlinear_series(
                      f, g, {"x"}, {"y"}, "t", {Rational::from_int(1)}, {Rational::from_int(1)}, 4);
                  t.expect(!sol && sol.error() == MathError::not_implemented,
                           "x*y - 1 has a non-constant dg/dy => not_implemented");
              })
        .test("unsupported function (sin) in f is honestly not_implemented",
              [&](TestContext& t) {
                  const std::vector<Expr> f{call("sin", sym("t"))};
                  const std::vector<Expr> g{sub(sym("y"), sym("x"))};
                  auto sol = solve_semiexplicit_nonlinear_series(
                      f, g, {"x"}, {"y"}, "t", {Rational::from_int(0)}, {Rational::from_int(0)}, 4);
                  t.expect(!sol && sol.error() == MathError::not_implemented,
                           "sin is outside the supported grammar => not_implemented");
              })
        .test("degenerate shapes are domain_error",
              [&](TestContext& t) {
                  const std::vector<Expr> f{sym("y")};
                  const std::vector<Expr> g{sub(sym("y"), sym("x"))};
                  auto zero_order = solve_semiexplicit_nonlinear_series(
                      f, g, {"x"}, {"y"}, "t", {Rational::from_int(1)}, {Rational::from_int(1)}, 0);
                  t.expect(!zero_order && zero_order.error() == MathError::domain_error,
                           "order == 0 => domain_error");
                  auto mismatch = solve_semiexplicit_nonlinear_series(
                      f, g, {"x", "z"}, {"y"}, "t", {Rational::from_int(1)},
                      {Rational::from_int(1)}, 4);
                  t.expect(!mismatch && mismatch.error() == MathError::domain_error,
                           "f.size() != xvars.size() => domain_error");
              })
        // ---- TIER 2: structural index --------------------------------------------------
        .test("pure ODE x'=y, y'=x has structural_index 1",
              [&](TestContext& t) {
                  const DaeSystem sys{.residuals = {sub(sym("xp"), sym("y")),
                                                    sub(sym("yp"), sym("x"))},
                                      .vars = {"x", "y"},
                                      .ders = {"xp", "yp"}};
                  auto info = analyze_structure(sys);
                  t.expect(info.has_value(), "analysis succeeded");
                  if (!info) return;
                  t.expect(info->structural_index == 1, "an ODE matches at sweep 0 => index 1");
                  t.expect(info->augmented_residuals.size() == 2,
                           "no differentiation needed => augmented set is the original");
              })
        .test("constraint system x'=y, 0=x-t is at the honest structural boundary",
              [&](TestContext& t) {
                  const DaeSystem sys{.residuals = {sub(sym("xp"), sym("y")),
                                                    sub(sym("x"), sym("t"))},
                                      .vars = {"x", "y"},
                                      .ders = {"xp", "yp"}};
                  auto info = analyze_structure(sys, 3);
                  // Honest boundary: EITHER a verified matching at some index >= 1, OR an
                  // honest not_implemented -- never a fabricated structure.
                  if (info) {
                      t.expect(info->structural_index >= 1 && !info->augmented_residuals.empty(),
                               "if matched, the index is >= 1 with a non-empty augmented set");
                  } else {
                      t.expect(info.error() == MathError::not_implemented,
                               "if unmatched within max_index, the answer is not_implemented");
                  }
              })
        // ---- TIER 3: numerical index-1 stepping ---------------------------------------
        .test("implicit Euler on x'=-x is exact per step: x_1 = 1/2",
              [&](TestContext& t) {
                  const DaeSystem sys{.residuals = {Expr::sum({sym("xp"), sym("x")})},
                                      .vars = {"x"},
                                      .ders = {"xp"}};
                  NlDaeOptions opts;
                  opts.bdf_order = 1;
                  const double x0 = 1.0;
                  const double xd0 = -1.0;
                  auto sol = solve_nonlinear_dae(sys, {&x0, 1}, {&xd0, 1}, 0.0, 1.0, 1, opts);
                  t.expect(sol.has_value(), "index-1 solve succeeded");
                  if (!sol) return;
                  t.expect(sol->structural_index == 1, "reported structural index 1");
                  t.expect(sol->states.size() == 2 && sol->times.size() == 2,
                           "initial state plus one step");
                  if (sol->states.size() != 2) return;
                  // Implicit Euler: x_1 (1 + h) = x_0, h = 1 => x_1 = 1/2 exactly.
                  t.expect(std::abs(sol->states.back()[0] - 0.5) < kTightTol,
                           "single unit implicit-Euler step yields exactly 1/2");
              })
        .test("consistent_initial_values recovers xdot = -2 for x'=-x at x=2",
              [&](TestContext& t) {
                  const DaeSystem sys{.residuals = {Expr::sum({sym("xp"), sym("x")})},
                                      .vars = {"x"},
                                      .ders = {"xp"}};
                  const double xg = 2.0;
                  const double xdg = 0.0;  // deliberately wrong guess; Newton must correct it
                  auto civ = consistent_initial_values(sys, {&xg, 1}, {&xdg, 1}, 0.0);
                  t.expect(civ.has_value(), "consistent-initialisation Newton converged");
                  if (!civ) return;
                  t.expect(std::abs(civ->first[0] - 2.0) < kTightTol, "x held fixed at 2");
                  t.expect(std::abs(civ->second[0] + 2.0) < 1e-9,
                           "consistent xdot for x'=-x at x=2 is -2");
              })
        .test("implicit Euler on the nonlinear x'=-x^3 satisfies s + s^3 = 1",
              [&](TestContext& t) {
                  const DaeSystem sys{.residuals = {Expr::sum({sym("xp"), pw(sym("x"), 3)})},
                                      .vars = {"x"},
                                      .ders = {"xp"}};
                  NlDaeOptions opts;
                  opts.bdf_order = 1;
                  const double x0 = 1.0;
                  const double xd0 = -1.0;
                  auto sol = solve_nonlinear_dae(sys, {&x0, 1}, {&xd0, 1}, 0.0, 1.0, 1, opts);
                  t.expect(sol.has_value(), "nonlinear index-1 solve succeeded");
                  if (!sol) return;
                  if (sol->states.size() != 2) return;
                  const double s = sol->states.back()[0];
                  // h = 1 implicit Euler: s + h*s^3 = x_0 = 1  (the scheme's exact relation).
                  t.expect(std::abs(s + s * s * s - 1.0) < kResidualTol,
                           "the returned state satisfies the implicit-Euler defining equation");
              })
        .test("a non-index-1 constraint system is honestly not_implemented",
              [&](TestContext& t) {
                  const DaeSystem sys{.residuals = {sub(sym("xp"), sym("y")),
                                                    sub(sym("x"), sym("t"))},
                                      .vars = {"x", "y"},
                                      .ders = {"xp", "yp"}};
                  const std::array<double, 2> x0{0.0, 0.0};
                  const std::array<double, 2> xd0{1.0, 1.0};
                  auto sol = solve_nonlinear_dae(sys, x0, xd0, 0.0, 1.0, 1);
                  t.expect(!sol && sol.error() == MathError::not_implemented,
                           "the stepper refuses a system it cannot reduce to index 1");
              })
        .test("degenerate stepping arguments are domain_error",
              [&](TestContext& t) {
                  const DaeSystem sys{.residuals = {Expr::sum({sym("xp"), sym("x")})},
                                      .vars = {"x"},
                                      .ders = {"xp"}};
                  const double x0 = 1.0;
                  const double xd0 = -1.0;
                  auto zero_steps = solve_nonlinear_dae(sys, {&x0, 1}, {&xd0, 1}, 0.0, 1.0, 0);
                  t.expect(!zero_steps && zero_steps.error() == MathError::domain_error,
                           "steps == 0 => domain_error");
                  auto bad_span = solve_nonlinear_dae(sys, {&x0, 1}, {&xd0, 1}, 1.0, 0.0, 1);
                  t.expect(!bad_span && bad_span.error() == MathError::domain_error,
                           "t1 <= t0 => domain_error");
              })
        .run();
}
