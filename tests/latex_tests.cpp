// Tests for nimblecas.latex: LaTeX rendering of symbolic expressions.
// @author Olumuyiwa Oluwasanmi
//
// Each expectation constructs an Expr explicitly and asserts the exact LaTeX string
// the renderer produces, so the tests double as documentation of the chosen
// conventions (precedence-driven \left(...\right), \frac for rationals/reciprocals,
// \sqrt for the 1/2 power, and \cdot only between adjacent numerals).

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.latex;
import nimblecas.testing;

using nimblecas::Expr;
using nimblecas::to_latex;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Convenience: an exact rational Expr (tests only use valid denominators).
[[nodiscard]] auto rat(std::int64_t p, std::int64_t q) -> Expr {
    return Expr::rational(p, q).value();
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.latex")
        .test("symbols_and_greek",
              [](TestContext& t) {
                  t.expect_eq(to_latex(Expr::symbol("x")), std::string("x"), "x verbatim");
                  t.expect_eq(to_latex(Expr::symbol("pi")), std::string("\\pi"),
                              "pi maps to \\pi");
                  t.expect_eq(to_latex(Expr::symbol("alpha")), std::string("\\alpha"),
                              "alpha maps to \\alpha");
                  t.expect_eq(to_latex(Expr::symbol("Omega")), std::string("\\Omega"),
                              "Omega maps to \\Omega");
                  t.expect_eq(to_latex(Expr::symbol("foo")), std::string("foo"),
                              "unknown name verbatim");
              })
        .test("constants",
              [](TestContext& t) {
                  t.expect_eq(to_latex(Expr::integer(42)), std::string("42"), "integer");
                  t.expect_eq(to_latex(Expr::integer(-7)), std::string("-7"),
                              "negative integer");
                  t.expect_eq(to_latex(rat(3, 4)), std::string("\\frac{3}{4}"),
                              "rational -> \\frac");
                  t.expect_eq(to_latex(rat(-1, 2)), std::string("-\\frac{1}{2}"),
                              "negative rational keeps sign outside \\frac");
                  t.expect_eq(to_latex(rat(6, 3)), std::string("2"),
                              "rational reducing to integer");
              })
        .test("sum",
              [](TestContext& t) {
                  auto x = Expr::symbol("x");
                  auto y = Expr::symbol("y");
                  t.expect_eq(to_latex(x.add(y)), std::string("x + y"), "x + y");
                  // x + (-3) y  ->  x - 3 y  (negative term folds into a minus)
                  auto neg_term = Expr::product({Expr::integer(-3), y});
                  t.expect_eq(to_latex(Expr::sum({x, neg_term})), std::string("x - 3 y"),
                              "negative product term renders as subtraction");
              })
        .test("product",
              [](TestContext& t) {
                  auto x = Expr::symbol("x");
                  auto y = Expr::symbol("y");
                  // 2 x^2
                  auto two_x_sq =
                      Expr::product({Expr::integer(2), Expr::power(x, Expr::integer(2))});
                  t.expect_eq(to_latex(two_x_sq), std::string("2 x^{2}"), "2 x^{2}");
                  // (1/2) x  ->  \frac{x}{2}
                  t.expect_eq(to_latex(Expr::product({rat(1, 2), x})),
                              std::string("\\frac{x}{2}"),
                              "rational coefficient forms a fraction");
                  // x y^{-1}  ->  \frac{x}{y}
                  auto x_over_y = Expr::product({x, Expr::power(y, Expr::integer(-1))});
                  t.expect_eq(to_latex(x_over_y), std::string("\\frac{x}{y}"),
                              "negative-power factor moves to the denominator");
                  // -x  (product of -1 and x)
                  t.expect_eq(to_latex(Expr::product({Expr::integer(-1), x})),
                              std::string("-x"), "coefficient -1 renders as a bare minus");
              })
        .test("power",
              [](TestContext& t) {
                  auto x = Expr::symbol("x");
                  t.expect_eq(to_latex(Expr::power(x, Expr::integer(2))),
                              std::string("x^{2}"), "x^{2}");
                  // x^(1/2) -> \sqrt{x}
                  t.expect_eq(to_latex(Expr::power(x, rat(1, 2))), std::string("\\sqrt{x}"),
                              "half power renders as a square root");
                  // x^(-1) -> \frac{1}{x}
                  t.expect_eq(to_latex(Expr::power(x, Expr::integer(-1))),
                              std::string("\\frac{1}{x}"),
                              "reciprocal renders as \\frac{1}{x}");
                  // x^(-2) -> \frac{1}{x^{2}}
                  t.expect_eq(to_latex(Expr::power(x, Expr::integer(-2))),
                              std::string("\\frac{1}{x^{2}}"),
                              "negative power renders as a reciprocal power");
              })
        .test("parenthesisation",
              [](TestContext& t) {
                  auto x = Expr::symbol("x");
                  auto y = Expr::symbol("y");
                  // (x + 1)^2 : a sum base must be parenthesised under a power
                  auto x_plus_1 = Expr::sum({x, Expr::integer(1)});
                  t.expect_eq(to_latex(Expr::power(x_plus_1, Expr::integer(2))),
                              std::string("\\left(x + 1\\right)^{2}"),
                              "sum base is wrapped under exponentiation");
                  // (x y)^2 : a product base must be parenthesised under a power
                  auto xy = Expr::product({x, y});
                  t.expect_eq(to_latex(Expr::power(xy, Expr::integer(2))),
                              std::string("\\left(x y\\right)^{2}"),
                              "product base is wrapped under exponentiation");
              })
        .test("functions",
              [](TestContext& t) {
                  auto x = Expr::symbol("x");
                  t.expect_eq(to_latex(Expr::apply("sin", {x})),
                              std::string("\\sin\\left(x\\right)"), "sin -> \\sin");
                  t.expect_eq(to_latex(Expr::apply("exp", {x})),
                              std::string("\\exp\\left(x\\right)"), "exp -> \\exp");
                  t.expect_eq(to_latex(Expr::apply("sqrt", {x})), std::string("\\sqrt{x}"),
                              "sqrt function -> \\sqrt{...}");
                  t.expect_eq(to_latex(Expr::apply("lambertW", {x})),
                              std::string("\\operatorname{lambertW}\\left(x\\right)"),
                              "unknown function -> \\operatorname{...}");
              })
        .test("nested_expression",
              [](TestContext& t) {
                  auto x = Expr::symbol("x");
                  // sin(x^2) + 2 x  ->  \sin\left(x^{2}\right) + 2 x
                  auto term1 = Expr::apply("sin", {Expr::power(x, Expr::integer(2))});
                  auto term2 = Expr::product({Expr::integer(2), x});
                  t.expect_eq(to_latex(Expr::sum({term1, term2})),
                              std::string("\\sin\\left(x^{2}\\right) + 2 x"),
                              "nested function, power and product compose correctly");
              })
        .run();
}
