// Tests for nimblecas.symbolic: expression construction, structural equality,
// FreeOf, and Substitute.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
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
              })
        .test("rational_validates_and_canonicalises",
              [](TestContext& t) {
                  auto half = Expr::rational(1, 2);
                  t.expect(half.has_value(), "1/2 is a valid rational");
                  // 2/4 reduces to 1/2; sign moves to numerator
                  t.expect(Expr::rational(2, 4).value().is_equivalent_to(half.value()),
                           "2/4 canonicalises to 1/2");
                  t.expect(Expr::rational(1, -2).value().is_equivalent_to(
                               Expr::rational(-1, 2).value()),
                           "1/-2 canonicalises to -1/2");
                  auto bad = Expr::rational(1, 0);
                  t.expect(!bad.has_value(), "zero denominator is rejected");
                  t.expect(bad.error() == nimblecas::MathError::division_by_zero,
                           "zero denominator yields division_by_zero");
                  // INT64_MIN would overflow on negation / inside gcd
                  auto overflow = Expr::rational(std::numeric_limits<std::int64_t>::min(), -1);
                  t.expect(!overflow.has_value(), "INT64_MIN/-1 is rejected");
                  t.expect(overflow.error() == nimblecas::MathError::overflow,
                           "INT64_MIN yields overflow");
              })
        .test("hash_matches_equality",
              [](TestContext& t) {
                  auto a = Expr::symbol("x").add(Expr::integer(1));
                  auto b = Expr::symbol("x").add(Expr::integer(1));
                  t.expect(a.is_equivalent_to(b), "a == b structurally");
                  t.expect_eq(nimblecas::hash_value(a), nimblecas::hash_value(b),
                              "equal expressions hash equal");
              })
        .test("nan_leaf_is_equivalent_to_itself",
              [](TestContext& t) {
                  auto nan_expr = Expr::real(std::numeric_limits<double>::quiet_NaN());
                  // structural identity must be syntactic, not IEEE (NaN != NaN)
                  t.expect(nan_expr.is_equivalent_to(nan_expr), "NaN leaf equals itself");
                  t.expect(!nimblecas::free_of(nan_expr, nan_expr),
                           "NaN expression is not free of itself");
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
