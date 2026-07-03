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

// small builders mirroring the derivative table, for expected values
auto sq(const Expr& a) -> Expr { return Expr::power(a, Expr::integer(2)); }
auto isqrt(const Expr& a) -> Expr { return Expr::power(a, Expr::rational(-1, 2).value()); }
auto rc(const Expr& a) -> Expr { return Expr::power(a, Expr::integer(-1)); }
auto ng(const Expr& a) -> Expr { return Expr::product({Expr::integer(-1), a}); }
auto omin(const Expr& a) -> Expr { return Expr::sum({Expr::integer(1), ng(a)}); }
auto opl(const Expr& a) -> Expr { return Expr::sum({Expr::integer(1), a}); }

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
        .test("inverse_trig_and_hyperbolic",
              [&](TestContext& t) {
                  diffs_to(t, fn("asin", x), "x", isqrt(omin(sq(x))),
                           "d/dx asin x = (1-x^2)^(-1/2)");
                  diffs_to(t, fn("atan", x), "x", rc(opl(sq(x))), "d/dx atan x = 1/(1+x^2)");
                  diffs_to(t, fn("sinh", x), "x", fn("cosh", x), "d/dx sinh x = cosh x");
                  diffs_to(t, fn("cosh", x), "x", fn("sinh", x), "d/dx cosh x = sinh x");
                  diffs_to(t, fn("tanh", x), "x", omin(sq(fn("tanh", x))),
                           "d/dx tanh x = 1 - tanh^2 x");
              })
        .test("special_functions",
              [&](TestContext& t) {
                  // d/dx erf(x) = (2/sqrt(pi)) exp(-x^2)
                  auto erf_d = Expr::product(
                      {Expr::integer(2), rc(fn("sqrt", Expr::symbol("pi"))), fn("exp", ng(sq(x)))});
                  diffs_to(t, fn("erf", x), "x", erf_d, "d/dx erf x = (2/sqrt(pi)) exp(-x^2)");
                  // d/dx gamma(x) = gamma(x) digamma(x)
                  diffs_to(t, fn("gamma", x), "x", Expr::product({fn("gamma", x), fn("digamma", x)}),
                           "d/dx gamma x = gamma(x) digamma(x)");
                  // d/dx W(x) = W(x) / (x (1 + W(x)))
                  auto w = fn("lambertW", x);
                  diffs_to(t, w, "x", Expr::product({w, rc(Expr::product({x, opl(w)}))}),
                           "d/dx lambertW x = W/(x(1+W))");
                  // chain rule through a special function: d/dx erf(x^2) = erf'(x^2) * 2x
                  auto chained = Expr::product(
                      {Expr::integer(2), rc(fn("sqrt", Expr::symbol("pi"))),
                       fn("exp", ng(sq(sq(x)))), Expr::integer(2), x});
                  diffs_to(t, fn("erf", sq(x)), "x", chained, "d/dx erf(x^2) chain rule");
              })
        .test("large_sum_differentiates_correctly_via_parallel_path",
              [&](TestContext& t) {
                  // A sum of 400 terms (> the parallel grain cutoff) exercises the
                  // parallel sum rule; the result must equal the serial-correct answer.
                  // d/dx sum_{k=1..N} x^k = sum_{k=1..N} k * x^(k-1)
                  constexpr std::int64_t nterms = 400;
                  std::vector<Expr> terms;
                  std::vector<Expr> expected_terms;
                  for (std::int64_t k = 1; k <= nterms; ++k) {
                      terms.push_back(x.pow(Expr::integer(k)));
                      expected_terms.push_back(
                          Expr::integer(k).mul(x.pow(Expr::integer(k - 1))));
                  }
                  auto d = differentiate(Expr::sum(std::move(terms)), "x");
                  auto expected = simplify(Expr::sum(std::move(expected_terms)));
                  t.expect(d.has_value() && expected.has_value() &&
                               d->is_equivalent_to(*expected),
                           "d/dx sum x^k == sum k x^(k-1) over 400 terms");
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
