// Tests for nimblecas.simplify: automatic simplification (Cohen ASAE).
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.testing;

using nimblecas::Expr;
using nimblecas::simplify;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Simplify u and assert it is structurally equal to `expected`.
auto simplifies_to(TestContext& t, const Expr& u, const Expr& expected, std::string_view what)
    -> void {
    auto s = simplify(u);
    if (!s.has_value()) {
        t.expect(false, std::format("{}: unexpected error", what));
        return;
    }
    t.expect(s->is_equivalent_to(expected),
             std::format("{}: got {}", what, s->to_string()));
}

}  // namespace

auto main() -> int {
    const auto x = Expr::symbol("x");
    const auto y = Expr::symbol("y");
    const auto zero = Expr::integer(0);
    const auto one = Expr::integer(1);

    return TestSuite("nimblecas.simplify")
        .test("additive_identity",
              [&](TestContext& t) {
                  simplifies_to(t, x.add(zero), x, "x + 0 -> x");
                  simplifies_to(t, zero.add(x), x, "0 + x -> x");
              })
        .test("multiplicative_identities",
              [&](TestContext& t) {
                  simplifies_to(t, x.mul(one), x, "x * 1 -> x");
                  simplifies_to(t, x.mul(zero), zero, "x * 0 -> 0");
              })
        .test("power_identities",
              [&](TestContext& t) {
                  simplifies_to(t, x.pow(one), x, "x^1 -> x");
                  simplifies_to(t, x.pow(zero), one, "x^0 -> 1");
                  simplifies_to(t, one.pow(x), one, "1^x -> 1");
              })
        .test("zero_to_the_zero_is_undefined",
              [&](TestContext& t) {
                  auto s = simplify(zero.pow(zero));
                  t.expect(!s.has_value(), "0^0 is undefined");
                  t.expect(s.error() == nimblecas::MathError::undefined_value,
                           "0^0 -> undefined_value");
              })
        .test("constant_folding",
              [&](TestContext& t) {
                  simplifies_to(t, Expr::integer(2).add(Expr::integer(3)), Expr::integer(5),
                                "2 + 3 -> 5");
                  simplifies_to(t, Expr::integer(2).mul(Expr::integer(3)), Expr::integer(6),
                                "2 * 3 -> 6");
                  simplifies_to(t, Expr::integer(2).pow(Expr::integer(10)), Expr::integer(1024),
                                "2^10 -> 1024");
                  // 1/2 + 1/2 -> 1 (exact rational folding)
                  simplifies_to(t,
                                Expr::rational(1, 2).value().add(Expr::rational(1, 2).value()),
                                one, "1/2 + 1/2 -> 1");
              })
        .test("canonical_ordering",
              [&](TestContext& t) {
                  auto lhs = simplify(x.add(y)).value();
                  auto rhs = simplify(y.add(x)).value();
                  t.expect(lhs.is_equivalent_to(rhs), "x + y and y + x canonicalise equal");
              })
        .test("combine_like_terms",
              [&](TestContext& t) {
                  // x + x -> 2*x
                  simplifies_to(t, x.add(x), Expr::integer(2).mul(x), "x + x -> 2*x");
                  // 2*x + 3*x -> 5*x
                  auto lhs = Expr::integer(2).mul(x).add(Expr::integer(3).mul(x));
                  simplifies_to(t, lhs, Expr::integer(5).mul(x), "2x + 3x -> 5x");
              })
        .test("combine_like_bases",
              [&](TestContext& t) {
                  // x * x -> x^2
                  simplifies_to(t, x.mul(x), x.pow(Expr::integer(2)), "x * x -> x^2");
                  // x^2 * x^3 -> x^5
                  auto lhs = x.pow(Expr::integer(2)).mul(x.pow(Expr::integer(3)));
                  simplifies_to(t, lhs, x.pow(Expr::integer(5)), "x^2 * x^3 -> x^5");
              })
        .test("reconstructed_terms_stay_flat",
              [&](TestContext& t) {
                  // 2*x*y + 3*x*y -> 5*x*y  (must be a flat product, not Mul{Mul{x,y},5})
                  auto lhs = Expr::product({Expr::integer(2), x, y})
                                 .add(Expr::product({Expr::integer(3), x, y}));
                  simplifies_to(t, lhs, Expr::product({Expr::integer(5), x, y}),
                                "2xy + 3xy -> 5xy (flat)");
              })
        .test("overflow_is_reported",
              [&](TestContext& t) {
                  // 2^63 overflows int64 exact folding
                  auto s = simplify(Expr::integer(2).pow(Expr::integer(63)));
                  t.expect(!s.has_value(), "2^63 overflows");
                  t.expect(s.error() == nimblecas::MathError::overflow, "2^63 -> overflow");
              })
        .run();
}
