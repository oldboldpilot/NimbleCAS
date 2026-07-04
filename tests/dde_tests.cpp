// Tests for nimblecas.dde: exact delay differential equations by the method of steps.
// @author Olumuyiwa Oluwasanmi
//
// Every case uses exact rational coefficients, hand-verified in the comments, for the
// piecewise-polynomial solution of a DDE with polynomial history. The classic linear
// benchmarks u' = -u(t-1) and u' = u(t-1) (constant history 1) are checked interval by
// interval; a nonlinear delayed operator u' = u(t-1)^2 exercises the per-interval Picard
// recursion; and evaluate() is checked at interior points, both interval endpoints, and
// out-of-range times. No floating point anywhere.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.powerseries;
import nimblecas.dde;
import nimblecas.testing;

using nimblecas::DdeOperator;
using nimblecas::DdeSolution;
using nimblecas::evaluate;
using nimblecas::MathError;
using nimblecas::PowerSeries;
using nimblecas::Rational;
using nimblecas::solve_method_of_steps;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Small exact rational helpers; inputs are tiny literals so unwrapping is safe.
auto ri(std::int64_t v) -> Rational { return Rational::from_int(v); }
auto rat(std::int64_t num, std::int64_t den) -> Rational { return *Rational::make(num, den); }

// Constant history piece (history(t) = c on [-tau, 0]) at the working order.
auto const_history(TestContext& t, const Rational& c, std::size_t order,
                   std::string_view what) -> PowerSeries {
    auto h = PowerSeries::constant(c, order);
    if (!h) {
        t.expect(false, std::format("{}: unexpected history construction error", what));
        return *PowerSeries::zero(1);
    }
    return *h;
}

// Assert a series has exactly the expected coefficients (index i is the coefficient of
// x^i) and the expected order.
auto expect_series(TestContext& t, const PowerSeries& s, const std::vector<Rational>& expected,
                   std::string_view what) -> void {
    t.expect(s.order() == expected.size(),
             std::format("{}: order = {}", what, expected.size()));
    for (std::size_t i = 0; i < expected.size(); ++i) {
        t.expect(s.coefficient(i) == expected[i],
                 std::format("{}: coefficient[{}] = {}", what, i, expected[i].to_string()));
    }
}

// The DDE right-hand sides used below (f such that u'(t) = f(t, u, u(t - tau))).
auto f_neg_delayed() -> DdeOperator {  // u' = -u(t-1)
    return [](const PowerSeries&, const PowerSeries& ud, const PowerSeries&) {
        return ud.scale(Rational::from_int(-1));
    };
}
auto f_delayed() -> DdeOperator {  // u' = u(t-1)
    return [](const PowerSeries&, const PowerSeries& ud,
              const PowerSeries&) -> nimblecas::Result<PowerSeries> { return ud; };
}
auto f_delayed_squared() -> DdeOperator {  // u' = u(t-1)^2
    return [](const PowerSeries&, const PowerSeries& ud, const PowerSeries&) {
        return ud.multiply(ud);
    };
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.dde")
        .test("negated_delay_pieces",
              [](TestContext& t) {
                  // u'(t) = -u(t-1), history u = 1 on [-1,0], tau = 1.
                  //   I0 [0,1]: delayed = 1, u(0) = 1 -> u' = -1 -> u = 1 - s.
                  //   I1 [1,2]: delayed = 1 - s (I0 piece), u(0) = (1-s)|_{s=1} = 0
                  //             -> u' = -(1 - s) = -1 + s -> u = -s + s^2/2.
                  const std::size_t order = 6;
                  auto sol = solve_method_of_steps(f_neg_delayed(),
                                                   const_history(t, ri(1), order, "hist"), ri(1),
                                                   2, order);
                  t.expect(sol.has_value(), "solve(u'=-u(t-1)) succeeds");
                  if (!sol) {
                      return;
                  }
                  t.expect(sol->pieces.size() == 2, "two interval pieces");
                  if (sol->pieces.size() != 2) {
                      return;
                  }
                  expect_series(t, sol->pieces[0],
                                {ri(1), ri(-1), ri(0), ri(0), ri(0), ri(0)}, "I0 = 1 - s");
                  expect_series(t, sol->pieces[1],
                                {ri(0), ri(-1), rat(1, 2), ri(0), ri(0), ri(0)},
                                "I1 = -s + s^2/2");
                  // evaluate at t = 3/2: I1 at s = 1/2 -> -1/2 + 1/8 = -3/8.
                  auto v = evaluate(*sol, rat(3, 2));
                  t.expect(v.has_value() && *v == rat(-3, 8), "evaluate(3/2) = -3/8");
                  // Interior of I0 at t = 1/2: 1 - 1/2 = 1/2.
                  auto v0 = evaluate(*sol, rat(1, 2));
                  t.expect(v0.has_value() && *v0 == rat(1, 2), "evaluate(1/2) = 1/2");
              })
        .test("positive_delay_pieces",
              [](TestContext& t) {
                  // u'(t) = u(t-1), history u = 1, tau = 1.
                  //   I0: delayed = 1, u(0) = 1 -> u' = 1 -> u = 1 + s.
                  //   I1: delayed = 1 + s, u(0) = (1+s)|_{s=1} = 2
                  //       -> u' = 1 + s -> u = 2 + s + s^2/2.
                  const std::size_t order = 6;
                  auto sol = solve_method_of_steps(f_delayed(),
                                                   const_history(t, ri(1), order, "hist"), ri(1),
                                                   2, order);
                  t.expect(sol.has_value(), "solve(u'=u(t-1)) succeeds");
                  if (!sol) {
                      return;
                  }
                  t.expect(sol->pieces.size() == 2, "two interval pieces");
                  if (sol->pieces.size() != 2) {
                      return;
                  }
                  expect_series(t, sol->pieces[0],
                                {ri(1), ri(1), ri(0), ri(0), ri(0), ri(0)}, "I0 = 1 + s");
                  expect_series(t, sol->pieces[1],
                                {ri(2), ri(1), rat(1, 2), ri(0), ri(0), ri(0)},
                                "I1 = 2 + s + s^2/2");
                  // evaluate at t = 3/2: I1 at s = 1/2 -> 2 + 1/2 + 1/8 = 21/8.
                  auto v = evaluate(*sol, rat(3, 2));
                  t.expect(v.has_value() && *v == rat(21, 8), "evaluate(3/2) = 21/8");
              })
        .test("nonlinear_delayed_operator",
              [](TestContext& t) {
                  // u'(t) = u(t-1)^2, history u = 1, tau = 1 (exercises per-interval Picard).
                  //   I0: delayed = 1, u(0) = 1 -> u' = 1 -> u = 1 + s.
                  //   I1: delayed = 1 + s, u(0) = 2 -> u' = (1+s)^2 = 1 + 2s + s^2
                  //       -> u = 2 + s + s^2 + s^3/3.
                  const std::size_t order = 6;
                  auto sol = solve_method_of_steps(f_delayed_squared(),
                                                   const_history(t, ri(1), order, "hist"), ri(1),
                                                   2, order);
                  t.expect(sol.has_value(), "solve(u'=u(t-1)^2) succeeds");
                  if (!sol) {
                      return;
                  }
                  t.expect(sol->pieces.size() == 2, "two interval pieces");
                  if (sol->pieces.size() != 2) {
                      return;
                  }
                  expect_series(t, sol->pieces[0],
                                {ri(1), ri(1), ri(0), ri(0), ri(0), ri(0)}, "I0 = 1 + s");
                  expect_series(t, sol->pieces[1],
                                {ri(2), ri(1), ri(1), rat(1, 3), ri(0), ri(0)},
                                "I1 = 2 + s + s^2 + s^3/3");
                  // evaluate at t = 2 (right endpoint): I1 at s = 1 -> 2+1+1+1/3 = 13/3.
                  auto v = evaluate(*sol, ri(2));
                  t.expect(v.has_value() && *v == rat(13, 3), "evaluate(2) = 13/3");
              })
        .test("evaluate_endpoints_and_out_of_range",
              [](TestContext& t) {
                  // Reuse u'(t) = -u(t-1) (history 1, tau 1, two intervals).
                  const std::size_t order = 6;
                  auto sol = solve_method_of_steps(f_neg_delayed(),
                                                   const_history(t, ri(1), order, "hist"), ri(1),
                                                   2, order);
                  t.expect(sol.has_value(), "solve succeeds");
                  if (!sol) {
                      return;
                  }
                  // Left endpoint t = 0: I0 at s = 0 -> u(0) = history(0) = 1.
                  auto v_left = evaluate(*sol, ri(0));
                  t.expect(v_left.has_value() && *v_left == ri(1), "evaluate(0) = 1");
                  // Shared knot t = 1: I1 at s = 0 -> 0 (continuity with I0 at s = 1).
                  auto v_knot = evaluate(*sol, ri(1));
                  t.expect(v_knot.has_value() && *v_knot == ri(0), "evaluate(1) = 0");
                  // Right endpoint t = 2: I1 at s = 1 -> -1 + 1/2 = -1/2.
                  auto v_right = evaluate(*sol, ri(2));
                  t.expect(v_right.has_value() && *v_right == rat(-1, 2), "evaluate(2) = -1/2");
                  // Before the range (t < 0) and beyond it (t > 2) are domain errors.
                  auto v_before = evaluate(*sol, ri(-1));
                  t.expect(!v_before.has_value() &&
                               v_before.error() == MathError::domain_error,
                           "evaluate(-1) is domain_error");
                  auto v_after = evaluate(*sol, ri(3));
                  t.expect(!v_after.has_value() && v_after.error() == MathError::domain_error,
                           "evaluate(3) is domain_error");
              })
        .test("non_unit_delay_recentering",
              [](TestContext& t) {
                  // tau = 2 checks that re-centering uses the local s in [0, tau], not t.
                  // u'(t) = -u(t-2), history u = 1 on [-2,0].
                  //   I0 [0,2]: delayed = 1, u(0) = 1 -> u' = -1 -> u = 1 - s.
                  //   I1 [2,4]: delayed = 1 - s (local), u(0) = (1-s)|_{s=2} = -1
                  //             -> u' = -(1 - s) = -1 + s -> u = -1 - s + s^2/2.
                  const std::size_t order = 5;
                  auto sol = solve_method_of_steps(f_neg_delayed(),
                                                   const_history(t, ri(1), order, "hist"), ri(2),
                                                   2, order);
                  t.expect(sol.has_value(), "solve(tau=2) succeeds");
                  if (!sol) {
                      return;
                  }
                  expect_series(t, sol->pieces[0], {ri(1), ri(-1), ri(0), ri(0), ri(0)},
                                "I0 = 1 - s");
                  expect_series(t, sol->pieces[1],
                                {ri(-1), ri(-1), rat(1, 2), ri(0), ri(0)},
                                "I1 = -1 - s + s^2/2");
                  // t = 3 lies in I1 at local s = 3 - 2 = 1: -1 - 1 + 1/2 = -3/2.
                  auto v = evaluate(*sol, ri(3));
                  t.expect(v.has_value() && *v == rat(-3, 2), "evaluate(3) = -3/2");
                  t.expect(sol->tau == ri(2), "solution retains tau = 2");
              })
        .test("degenerate_arguments_are_domain_errors",
              [](TestContext& t) {
                  const std::size_t order = 4;
                  auto hist = const_history(t, ri(1), order, "hist");

                  auto bad_tau = solve_method_of_steps(f_delayed(), hist, ri(0), 2, order);
                  t.expect(!bad_tau.has_value() &&
                               bad_tau.error() == MathError::domain_error,
                           "tau = 0 yields domain_error");

                  auto neg_tau = solve_method_of_steps(f_delayed(), hist, ri(-1), 2, order);
                  t.expect(!neg_tau.has_value() &&
                               neg_tau.error() == MathError::domain_error,
                           "tau < 0 yields domain_error");

                  auto bad_order = solve_method_of_steps(f_delayed(), hist, ri(1), 2, 0);
                  t.expect(!bad_order.has_value() &&
                               bad_order.error() == MathError::domain_error,
                           "order 0 yields domain_error");

                  auto bad_intervals =
                      solve_method_of_steps(f_delayed(), hist, ri(1), 0, order);
                  t.expect(!bad_intervals.has_value() &&
                               bad_intervals.error() == MathError::domain_error,
                           "num_intervals 0 yields domain_error");

                  // evaluate on an empty solution is also a domain_error.
                  DdeSolution empty{.pieces = {}, .tau = ri(1)};
                  auto v = evaluate(empty, ri(0));
                  t.expect(!v.has_value() && v.error() == MathError::domain_error,
                           "evaluate on empty solution is domain_error");
              })
        .run();
}
