// Tests for nimblecas.symbolic: expression construction, structural equality,
// FreeOf, and Substitute.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.symbolic;
import nimblecas.testing;

using nimblecas::Expr;
using nimblecas::free_of;
using nimblecas::substitute;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

auto main() -> int {
    return TestSuite("nimblecas.symbolic")
        .test("structural_equality",
              [](TestContext& t) {
                  auto x = Expr::symbol("x");
                  auto x2 = Expr::symbol("x");
                  auto y = Expr::symbol("y");
                  t.expect(x.is_equivalent_to(x2), "x == x");
                  t.expect(!x.is_equivalent_to(y), "x != y");
                  // x + y and x + y are structurally equal; x + y != y + x (no simplify yet)
                  t.expect(x.add(y).is_equivalent_to(x.add(y)), "x+y == x+y");
                  t.expect(!x.add(y).is_equivalent_to(y.add(x)), "x+y != y+x pre-simplify");
              })
        .test("constant_kinds_are_distinct",
              [](TestContext& t) {
                  t.expect(Expr::integer(2).is_equivalent_to(Expr::integer(2)), "2 == 2");
                  t.expect(!Expr::integer(2).is_equivalent_to(Expr::real(2.0)),
                           "int 2 != real 2.0 (distinct constant kinds)");
                  t.expect(Expr::rational(1, 2).is_equivalent_to(Expr::rational(1, 2)),
                           "1/2 == 1/2");
              })
        .test("free_of",
              [](TestContext& t) {
                  auto x = Expr::symbol("x");
                  auto y = Expr::symbol("y");
                  // u = x^2 + y
                  auto u = Expr::power(x, Expr::integer(2)).add(y);
                  t.expect(!free_of(u, x), "x^2 + y contains x");
                  t.expect(free_of(u, Expr::symbol("z")), "x^2 + y is free of z");
                  t.expect(!free_of(u, y), "x^2 + y contains y");
              })
        .test("substitute_replaces_subexpression",
              [](TestContext& t) {
                  auto x = Expr::symbol("x");
                  auto y = Expr::symbol("y");
                  // u = x^2 + x ; substitute x -> y  =>  y^2 + y
                  auto u = Expr::power(x, Expr::integer(2)).add(x);
                  auto expected = Expr::power(y, Expr::integer(2)).add(y);
                  auto result = substitute(u, x, y);
                  t.expect(result.is_equivalent_to(expected),
                           std::format("subst x->y in x^2+x gives {}", result.to_string()));
                  // original is untouched (immutability)
                  t.expect(u.is_equivalent_to(Expr::power(x, Expr::integer(2)).add(x)),
                           "original expression unchanged after substitute");
              })
        .test("substitute_whole_expression",
              [](TestContext& t) {
                  auto x = Expr::symbol("x");
                  auto expr = Expr::power(x, Expr::integer(2));
                  auto r = substitute(expr, expr, Expr::integer(7));
                  t.expect(r.is_equivalent_to(Expr::integer(7)),
                           "substituting the whole expression replaces it");
              })
        .run();
}
