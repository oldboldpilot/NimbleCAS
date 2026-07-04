// Tests for nimblecas.ode: exact power-series first-order systems and higher-order ODEs.
// @author Olumuyiwa Oluwasanmi
//
// Every case uses exact rational coefficients hand-verified in the comments. Benchmarks:
// the harmonic system (u'=v, v'=-u -> sin/cos), a coupled linear system (u'=u+v, v'=u-v,
// whose Taylor coefficients are cross-checked against the closed form cosh(sqrt2 x) +
// sinh(sqrt2 x)/sqrt2), the second-order equations u''=-u -> sin and u''=u -> e^x, a
// scalar n=1 cross-check against nimblecas.perturbation's adm_solve, exact Horner
// evaluation, and the domain_error guards. No floating point anywhere.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.powerseries;
import nimblecas.perturbation;
import nimblecas.ode;
import nimblecas.testing;

using nimblecas::adm_solve;
using nimblecas::evaluate;
using nimblecas::HigherOrderOperator;
using nimblecas::PowerSeries;
using nimblecas::Rational;
using nimblecas::solve_first_order_system;
using nimblecas::solve_higher_order;
using nimblecas::SystemOperator;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Small exact rational helpers; inputs are tiny literals so unwrapping is safe.
auto ri(std::int64_t v) -> Rational { return Rational::from_int(v); }
auto rat(std::int64_t num, std::int64_t den) -> Rational { return *Rational::make(num, den); }

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

// --- Vector fields (f such that u_i' = f_i(u)) ------------------------------------------

// Harmonic oscillator system: u' = v, v' = -u.
auto f_harmonic() -> SystemOperator {
    return [](const std::vector<PowerSeries>& u) -> nimblecas::Result<std::vector<PowerSeries>> {
        auto neg_u = u[0].scale(Rational::from_int(-1));
        if (!neg_u) {
            return nimblecas::make_error<std::vector<PowerSeries>>(neg_u.error());
        }
        return std::vector<PowerSeries>{u[1], *neg_u};
    };
}

// Coupled linear system: u' = u + v, v' = u - v.
auto f_coupled() -> SystemOperator {
    return [](const std::vector<PowerSeries>& u) -> nimblecas::Result<std::vector<PowerSeries>> {
        auto up = u[0].add(u[1]);       // u' = u + v
        auto vp = u[0].subtract(u[1]);  // v' = u - v
        if (!up || !vp) {
            return nimblecas::make_error<std::vector<PowerSeries>>(
                up ? vp.error() : up.error());
        }
        return std::vector<PowerSeries>{*up, *vp};
    };
}

// Scalar (n = 1) exponential system: u' = u.
auto f_scalar_identity() -> SystemOperator {
    return [](const std::vector<PowerSeries>& u) -> nimblecas::Result<std::vector<PowerSeries>> {
        return std::vector<PowerSeries>{u[0]};
    };
}

// --- Higher-order right-hand sides (f such that u^{(k)} = f(u, u', ...)) -----------------

// u'' = -u.
auto f_second_neg() -> HigherOrderOperator {
    return [](const std::vector<PowerSeries>& y) { return y[0].scale(Rational::from_int(-1)); };
}

// u'' = u.
auto f_second_pos() -> HigherOrderOperator {
    return [](const std::vector<PowerSeries>& y) -> nimblecas::Result<PowerSeries> {
        return y[0];
    };
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.ode")
        .test("harmonic_system_sin_cos",
              [](TestContext& t) {
                  // u' = v, v' = -u, u(0) = 0, v(0) = 1 -> u = sin(x), v = cos(x).
                  auto r = solve_first_order_system(f_harmonic(), {ri(0), ri(1)}, 8);
                  t.expect(r.has_value(), "harmonic system solves");
                  if (!r) {
                      return;
                  }
                  t.expect(r->size() == 2, "two solution components");
                  if (r->size() != 2) {
                      return;
                  }
                  // sin(x) = 0 + x - x^3/6 + x^5/120 - x^7/5040.
                  expect_series(t, (*r)[0],
                                {ri(0), ri(1), ri(0), rat(-1, 6), ri(0), rat(1, 120), ri(0),
                                 rat(-1, 5040)},
                                "sin series");
                  // cos(x) = 1 - x^2/2 + x^4/24 - x^6/720.
                  expect_series(t, (*r)[1],
                                {ri(1), ri(0), rat(-1, 2), ri(0), rat(1, 24), ri(0),
                                 rat(-1, 720), ri(0)},
                                "cos series");
              })
        .test("coupled_linear_system",
              [](TestContext& t) {
                  // u' = u + v, v' = u - v, u(0) = 1, v(0) = 0. The matrix A = [[1,1],[1,-1]]
                  // satisfies A^2 = 2I, so [u,v] = cosh(sqrt2 x)[1,0] + sinh(sqrt2 x)/sqrt2
                  // [1,1]. Hand-verified Taylor coefficients to order 6:
                  //   u = [1, 1, 1, 1/3, 1/6, 1/30], v = [0, 1, 0, 1/3, 0, 1/30].
                  auto r = solve_first_order_system(f_coupled(), {ri(1), ri(0)}, 6);
                  t.expect(r.has_value(), "coupled system solves");
                  if (!r) {
                      return;
                  }
                  t.expect(r->size() == 2, "two solution components");
                  if (r->size() != 2) {
                      return;
                  }
                  expect_series(t, (*r)[0],
                                {ri(1), ri(1), ri(1), rat(1, 3), rat(1, 6), rat(1, 30)},
                                "u = cosh(sqrt2 x) + sinh(sqrt2 x)/sqrt2");
                  expect_series(t, (*r)[1],
                                {ri(0), ri(1), ri(0), rat(1, 3), ri(0), rat(1, 30)},
                                "v = sinh(sqrt2 x)/sqrt2");
              })
        .test("scalar_system_matches_perturbation",
              [](TestContext& t) {
                  // n = 1 case u' = u, u(0) = 1 must reproduce the exp series that
                  // nimblecas.perturbation's adm_solve returns for the same problem.
                  auto sys = solve_first_order_system(f_scalar_identity(), {ri(1)}, 8);
                  auto adm = adm_solve(
                      [](const PowerSeries& u) -> nimblecas::Result<PowerSeries> { return u; },
                      ri(1), 8);
                  t.expect(sys.has_value() && adm.has_value(), "both solvers succeed");
                  if (!sys || !adm) {
                      return;
                  }
                  t.expect(sys->size() == 1, "single component");
                  if (sys->size() != 1) {
                      return;
                  }
                  // Both are the exp series 1 + x + x^2/2 + ... + x^7/5040.
                  expect_series(t, (*sys)[0],
                                {ri(1), ri(1), rat(1, 2), rat(1, 6), rat(1, 24), rat(1, 120),
                                 rat(1, 720), rat(1, 5040)},
                                "exp series");
                  t.expect((*sys)[0].is_equal(*adm),
                           "system n=1 solution == perturbation adm_solve");
              })
        .test("higher_order_second_neg_is_sine",
              [](TestContext& t) {
                  // u'' = -u, u(0) = 0, u'(0) = 1 -> u = sin(x).
                  auto r = solve_higher_order(f_second_neg(), {ri(0), ri(1)}, 8);
                  t.expect(r.has_value(), "u'' = -u solves");
                  if (!r) {
                      return;
                  }
                  expect_series(t, *r,
                                {ri(0), ri(1), ri(0), rat(-1, 6), ri(0), rat(1, 120), ri(0),
                                 rat(-1, 5040)},
                                "sin series");
              })
        .test("higher_order_second_pos_is_exp",
              [](TestContext& t) {
                  // u'' = u, u(0) = 1, u'(0) = 1 -> u = e^x (all derivatives at 0 equal 1).
                  auto r = solve_higher_order(f_second_pos(), {ri(1), ri(1)}, 8);
                  t.expect(r.has_value(), "u'' = u solves");
                  if (!r) {
                      return;
                  }
                  expect_series(t, *r,
                                {ri(1), ri(1), rat(1, 2), rat(1, 6), rat(1, 24), rat(1, 120),
                                 rat(1, 720), rat(1, 5040)},
                                "exp series");
              })
        .test("higher_order_third_order_cosine_component",
              [](TestContext& t) {
                  // A genuinely higher (k = 3) reduction: u''' = -u', u(0) = 1, u'(0) = 0,
                  // u''(0) = -1. Here u' satisfies w'' = -w with w(0) = 0, w'(0) = -1, so
                  // u' = -sin(x) and u = cos(x). Coefficients to order 8:
                  //   cos(x) = 1 - x^2/2 + x^4/24 - x^6/720.
                  auto f_third = [](const std::vector<PowerSeries>& y)
                      -> nimblecas::Result<PowerSeries> {
                      return y[1].scale(Rational::from_int(-1));  // u''' = -u'
                  };
                  auto r = solve_higher_order(f_third, {ri(1), ri(0), ri(-1)}, 8);
                  t.expect(r.has_value(), "u''' = -u' solves");
                  if (!r) {
                      return;
                  }
                  expect_series(t, *r,
                                {ri(1), ri(0), rat(-1, 2), ri(0), rat(1, 24), ri(0),
                                 rat(-1, 720), ri(0)},
                                "cos series");
              })
        .test("evaluate_truncated_series_by_horner",
              [](TestContext& t) {
                  // 1 + x + x^2 evaluated at x = 2 -> 1 + 2 + 4 = 7; at x = 1/2 -> 7/4.
                  auto poly = PowerSeries::from_coeffs({ri(1), ri(1), ri(1)}, 3);
                  t.expect(poly.has_value(), "series builds");
                  if (!poly) {
                      return;
                  }
                  auto at_two = evaluate(*poly, ri(2));
                  auto at_half = evaluate(*poly, rat(1, 2));
                  t.expect(at_two.has_value() && at_half.has_value(), "evaluations succeed");
                  if (!at_two || !at_half) {
                      return;
                  }
                  t.expect(*at_two == ri(7), "evaluate(1+x+x^2, 2) = 7");
                  t.expect(*at_half == rat(7, 4), "evaluate(1+x+x^2, 1/2) = 7/4");
                  // Evaluating the exp series at 0 returns the constant term.
                  auto expser = solve_higher_order(f_second_pos(), {ri(1), ri(1)}, 6);
                  t.expect(expser.has_value(), "exp series builds");
                  if (!expser) {
                      return;
                  }
                  auto at_zero = evaluate(*expser, ri(0));
                  t.expect(at_zero.has_value() && *at_zero == ri(1),
                           "evaluate(exp, 0) = 1 (constant term)");
              })
        .test("degenerate_arguments_are_domain_errors",
              [](TestContext& t) {
                  // order 0 rejected for the system solver.
                  auto bad_order = solve_first_order_system(f_harmonic(), {ri(0), ri(1)}, 0);
                  t.expect(!bad_order.has_value(), "system order 0 is rejected");
                  t.expect(bad_order.error() == nimblecas::MathError::domain_error,
                           "system order 0 yields domain_error");

                  // Empty u0 rejected.
                  auto empty_u0 = solve_first_order_system(f_harmonic(), {}, 8);
                  t.expect(!empty_u0.has_value(), "empty u0 is rejected");
                  t.expect(empty_u0.error() == nimblecas::MathError::domain_error,
                           "empty u0 yields domain_error");

                  // A vector field whose output length differs from n = |u0| is a mismatch.
                  auto wrong_size =
                      [](const std::vector<PowerSeries>& u)
                      -> nimblecas::Result<std::vector<PowerSeries>> {
                      return std::vector<PowerSeries>{u[0]};  // returns 1, expected 2
                  };
                  auto mismatch =
                      solve_first_order_system(wrong_size, {ri(0), ri(1)}, 8);
                  t.expect(!mismatch.has_value(), "f output-size mismatch is rejected");
                  t.expect(mismatch.error() == nimblecas::MathError::domain_error,
                           "size mismatch yields domain_error");

                  // Higher-order solver: order 0 and empty initial (k must be >= 1).
                  auto bad_ho_order = solve_higher_order(f_second_pos(), {ri(1), ri(1)}, 0);
                  t.expect(!bad_ho_order.has_value(), "higher-order order 0 is rejected");
                  t.expect(bad_ho_order.error() == nimblecas::MathError::domain_error,
                           "higher-order order 0 yields domain_error");

                  auto empty_initial = solve_higher_order(f_second_pos(), {}, 8);
                  t.expect(!empty_initial.has_value(), "empty initial (k=0) is rejected");
                  t.expect(empty_initial.error() == nimblecas::MathError::domain_error,
                           "empty initial yields domain_error");
              })
        .run();
}
