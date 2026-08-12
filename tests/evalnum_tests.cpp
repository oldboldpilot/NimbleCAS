// Tests for nimblecas.evalnum: numeric (IEEE-754 double) evaluation of a symbolic
// Expr, hand-verified against exact/analytic values.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.evalnum;
import nimblecas.testing;

using nimblecas::eval_double;
using nimblecas::Expr;
using nimblecas::MathError;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Convenience: an exact rational Expr (tests only use valid denominators).
[[nodiscard]] auto rat(std::int64_t p, std::int64_t q) -> Expr {
    return Expr::rational(p, q).value();
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.evalnum")
        .test("polynomial_single_var",
              [](TestContext& t) {
                  // x^2 + 1 at x = 3 -> 10.0 exactly.
                  auto x = Expr::symbol("x");
                  auto expr = Expr::sum({Expr::power(x, Expr::integer(2)), Expr::integer(1)});
                  auto result = eval_double(expr, "x", 3.0);
                  t.expect(result.has_value(), "x^2+1 at x=3 evaluates");
                  t.expect_eq(*result, 10.0, "x^2+1 at x=3 == 10.0 exactly");
              })
        .test("rational_leaf",
              [](TestContext& t) {
                  // A rational leaf 3/4 evaluates to 0.75 exactly-in-double.
                  auto result = eval_double(rat(3, 4), "x", 0.0);
                  t.expect(result.has_value(), "3/4 evaluates");
                  t.expect_eq(*result, 0.75, "3/4 == 0.75 exactly in double");
              })
        .test("trig_at_special_points",
              [](TestContext& t) {
                  // sin(0) == 0 exactly.
                  auto sin0 = eval_double(Expr::apply("sin", {Expr::integer(0)}), "x", 0.0);
                  t.expect(sin0.has_value(), "sin(0) evaluates");
                  t.expect_eq(*sin0, 0.0, "sin(0) == 0 exactly");
                  // sin(pi) via the "pi" symbol, within a few ULP of std::sin(pi).
                  auto sin_pi =
                      eval_double(Expr::apply("sin", {Expr::symbol("pi")}), "x", 0.0);
                  t.expect(sin_pi.has_value(), "sin(pi) evaluates");
                  const double expected = std::sin(std::numbers::pi);
                  t.expect(std::abs(*sin_pi - expected) <= 4.0 * std::numeric_limits<double>::epsilon(),
                           "sin(pi) within a few ULP of std::sin(pi)");
              })
        .test("unknown_symbol_and_function_are_honest_errors",
              [](TestContext& t) {
                  // "y" is unbound when only "x" is provided -> domain_error, never 0.
                  auto unknown_symbol = eval_double(Expr::symbol("y"), "x", 1.0);
                  t.expect(!unknown_symbol.has_value(), "unbound symbol y fails");
                  t.expect_eq(unknown_symbol.error(), MathError::domain_error,
                              "unbound symbol -> domain_error");
                  // An unrecognised function head -> domain_error.
                  auto unknown_fn =
                      eval_double(Expr::apply("lambertW", {Expr::integer(1)}), "x", 0.0);
                  t.expect(!unknown_fn.has_value(), "unknown function head fails");
                  t.expect_eq(unknown_fn.error(), MathError::domain_error,
                              "unknown function -> domain_error");
              })
        .test("map_overload_two_variables",
              [](TestContext& t) {
                  // x * y at {x: 2, y: 5} -> 10.0.
                  auto expr = Expr::product({Expr::symbol("x"), Expr::symbol("y")});
                  const std::unordered_map<std::string, double> bindings{{"x", 2.0}, {"y", 5.0}};
                  auto result = eval_double(expr, bindings);
                  t.expect(result.has_value(), "x*y at {x:2,y:5} evaluates");
                  t.expect_eq(*result, 10.0, "x*y at {x:2,y:5} == 10.0");
              })
        .test("non_finite_intermediate_is_domain_error",
              [](TestContext& t) {
                  // log(-1) is non-finite (NaN) in real double arithmetic -> domain_error,
                  // never a silently-propagated NaN.
                  auto result =
                      eval_double(Expr::apply("log", {Expr::integer(-1)}), "x", 0.0);
                  t.expect(!result.has_value(), "log(-1) fails");
                  t.expect_eq(result.error(), MathError::domain_error,
                              "non-finite intermediate -> domain_error");
              })
        .run();
}
