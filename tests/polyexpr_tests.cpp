// Tests for nimblecas.polyexpr: Expr <-> Polynomial bridge.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.polynomial;
import nimblecas.polyexpr;
import nimblecas.testing;

using nimblecas::Expr;
using nimblecas::from_polynomial;
using nimblecas::MathError;
using nimblecas::Polynomial;
using nimblecas::to_polynomial;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

auto main() -> int {
    const auto x = Expr::symbol("x");
    const auto i = [](std::int64_t v) { return Expr::integer(v); };

    return TestSuite("nimblecas.polyexpr")
        .test("expr_to_polynomial",
              [&](TestContext& t) {
                  // x^2 + 2x + 1  ->  {1, 2, 1}
                  auto e = x.pow(i(2)).add(i(2).mul(x)).add(i(1));
                  auto p = to_polynomial(e, "x");
                  t.expect(p.has_value() && p->is_equal(Polynomial{{1, 2, 1}}),
                           "x^2 + 2x + 1 -> {1,2,1}");
                  // pure constant
                  auto c = to_polynomial(i(7), "x");
                  t.expect(c.has_value() && c->is_equal(Polynomial::constant(7)), "7 -> 7");
              })
        .test("non_polynomials_are_rejected",
              [&](TestContext& t) {
                  t.expect(!to_polynomial(Expr::apply("sin", {x}), "x").has_value(),
                           "sin(x) is not a polynomial");
                  t.expect(!to_polynomial(Expr::symbol("y"), "x").has_value(),
                           "y is not a polynomial in x");
                  // negative exponent
                  t.expect(!to_polynomial(x.pow(i(-1)), "x").has_value(),
                           "x^-1 is not a polynomial");
              })
        .test("from_polynomial_round_trips",
              [&](TestContext& t) {
                  Polynomial p{{-1, 0, 3, 0, 2}};  // 2x^4 + 3x^2 - 1
                  auto e = from_polynomial(p, "x");
                  auto back = to_polynomial(e, "x");
                  t.expect(back.has_value() && back->is_equal(p),
                           std::format("round-trip preserves the polynomial: {}", p.to_string()));
                  t.expect(from_polynomial(Polynomial{}, "x").is_equivalent_to(Expr::integer(0)),
                           "zero polynomial -> 0");
              })
        .test("symbolic_polynomial_gcd",
              [&](TestContext& t) {
                  auto x2m1 = x.pow(i(2)).add(i(-1));  // x^2 - 1
                  auto xm1 = x.add(i(-1));             // x - 1
                  auto g = nimblecas::polynomial_gcd(x2m1, xm1, "x");
                  t.expect(g.has_value(), "gcd computed");
                  // verify via round-trip to Polynomial: gcd == x - 1
                  auto gp = to_polynomial(g.value(), "x");
                  t.expect(gp.has_value() && gp->is_equal(Polynomial{{-1, 1}}),
                           "gcd(x^2-1, x-1) = x-1");
              })
        .test("symbolic_square_free_factor",
              [&](TestContext& t) {
                  auto sq = x.add(i(1)).pow(i(2));  // (x+1)^2
                  auto fs = nimblecas::square_free_factor(sq, "x");
                  t.expect(fs.has_value() && fs->size() == 1, "one square-free factor");
                  if (fs.has_value() && fs->size() == 1) {
                      auto& [factor, mult] = fs->front();
                      auto fp = to_polynomial(factor, "x");
                      t.expect(mult == 2 && fp.has_value() && fp->is_equal(Polynomial{{1, 1}}),
                               "(x+1)^2 -> factor (x+1), multiplicity 2");
                  }
              })
        .run();
}
