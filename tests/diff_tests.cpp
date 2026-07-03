// Tests for nimblecas.diff: symbolic differentiation.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.diff;
import nimblecas.testing;

using nimblecas::differentiate;
using nimblecas::Expr;
using nimblecas::simplify;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// d/dvar(u) should equal the simplified `expected`.
auto diffs_to(TestContext& t, const Expr& u, std::string_view var, const Expr& expected,
              std::string_view what) -> void {
    auto d = differentiate(u, var);
    auto e = simplify(expected);
    if (!d.has_value() || !e.has_value()) {
        t.expect(false, std::format("{}: unexpected error", what));
        return;
    }
    t.expect(d->is_equivalent_to(*e), std::format("{}: got {}", what, d->to_string()));
}

auto fn(std::string name, const Expr& arg) -> Expr {
    return Expr::apply(std::move(name), {arg});
}

}  // namespace

auto main() -> int {
    const auto x = Expr::symbol("x");
    const auto y = Expr::symbol("y");
    const auto two = Expr::integer(2);
    const auto three = Expr::integer(3);

    return TestSuite("nimblecas.diff")
        .test("leaf_rules",
              [&](TestContext& t) {
                  diffs_to(t, Expr::integer(5), "x", Expr::integer(0), "d/dx 5 = 0");
                  diffs_to(t, x, "x", Expr::integer(1), "d/dx x = 1");
                  diffs_to(t, y, "x", Expr::integer(0), "d/dx y = 0");
              })
        .test("power_rule",
              [&](TestContext& t) {
                  diffs_to(t, x.pow(two), "x", two.mul(x), "d/dx x^2 = 2x");
                  diffs_to(t, x.pow(three), "x", three.mul(x.pow(two)), "d/dx x^3 = 3x^2");
              })
        .test("sum_rule",
              [&](TestContext& t) {
                  // d/dx (x^2 + 3x) = 2x + 3
                  auto u = x.pow(two).add(three.mul(x));
                  diffs_to(t, u, "x", two.mul(x).add(three), "d/dx (x^2 + 3x) = 2x + 3");
              })
        .test("product_rule",
              [&](TestContext& t) {
                  // d/dx (x * sin(x)) = sin(x) + x*cos(x)
                  auto u = x.mul(fn("sin", x));
                  auto expected = fn("sin", x).add(x.mul(fn("cos", x)));
                  diffs_to(t, u, "x", expected, "d/dx (x sin x) = sin x + x cos x");
              })
        .test("elementary_functions",
              [&](TestContext& t) {
                  diffs_to(t, fn("sin", x), "x", fn("cos", x), "d/dx sin x = cos x");
                  diffs_to(t, fn("exp", x), "x", fn("exp", x), "d/dx exp x = exp x");
                  diffs_to(t, fn("ln", x), "x", x.pow(Expr::integer(-1)), "d/dx ln x = 1/x");
              })
        .test("chain_rule",
              [&](TestContext& t) {
                  // d/dx sin(x^2) = 2x*cos(x^2)
                  auto u = fn("sin", x.pow(two));
                  auto expected = Expr::product({two, x, fn("cos", x.pow(two))});
                  diffs_to(t, u, "x", expected, "d/dx sin(x^2) = 2x cos(x^2)");
              })
        .test("unknown_function_is_left_unevaluated",
              [&](TestContext& t) {
                  auto u = fn("f", x);  // unknown function
                  auto d = differentiate(u, "x");
                  auto expected = Expr::apply("Derivative", {u, Expr::symbol("x")});
                  t.expect(d.has_value() && d->is_equivalent_to(expected),
                           "d/dx f(x) = Derivative(f(x), x)");
              })
        .run();
}
