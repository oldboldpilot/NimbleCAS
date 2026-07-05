// Tests for nimblecas.expand: algebraic expansion (distribution + multinomial).
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.expand;
import nimblecas.testing;

using nimblecas::Expr;
using nimblecas::expand;
using nimblecas::simplify;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// expand(u) and simplify(expected) must be structurally equal. Both sides pass through
// simplify (expand ends with it), so the comparison is against the canonical form and
// is insensitive to operand ordering.
auto expands_to(TestContext& t, const Expr& u, const Expr& expected, std::string_view what)
    -> void {
    auto got = expand(u);
    auto want = simplify(expected);
    if (!got.has_value()) {
        t.expect(false, std::format("{}: unexpected expand error", what));
        return;
    }
    if (!want.has_value()) {
        t.expect(false, std::format("{}: expected side failed to simplify", what));
        return;
    }
    t.expect(got->is_equivalent_to(*want),
             std::format("{}: got {}", what, got->to_string()));
}

// Convenience monomials.
auto ipow(const Expr& b, std::int64_t e) -> Expr { return b.pow(Expr::integer(e)); }
auto scaled(std::int64_t c, std::vector<Expr> factors) -> Expr {
    factors.insert(factors.begin(), Expr::integer(c));
    return Expr::product(std::move(factors));
}

}  // namespace

auto main() -> int {
    const auto a = Expr::symbol("a");
    const auto b = Expr::symbol("b");
    const auto c = Expr::symbol("c");
    const auto x = Expr::symbol("x");
    const auto one = Expr::integer(1);

    return TestSuite("nimblecas.expand")
        .test("binomial_square",
              [&](TestContext& t) {
                  // (a + b)^2 -> a^2 + 2ab + b^2
                  const Expr u = ipow(a.add(b), 2);
                  const Expr expected =
                      Expr::sum({ipow(a, 2), scaled(2, {a, b}), ipow(b, 2)});
                  expands_to(t, u, expected, "(a+b)^2");
              })
        .test("difference_of_squares",
              [&](TestContext& t) {
                  // (x + 1)(x - 1) -> x^2 - 1
                  const Expr u = Expr::product(
                      {x.add(one), Expr::sum({x, Expr::integer(-1)})});
                  const Expr expected = Expr::sum({ipow(x, 2), Expr::integer(-1)});
                  expands_to(t, u, expected, "(x+1)(x-1)");
              })
        .test("binomial_cube",
              [&](TestContext& t) {
                  // (x + 1)^3 -> x^3 + 3x^2 + 3x + 1
                  const Expr u = ipow(x.add(one), 3);
                  const Expr expected = Expr::sum(
                      {ipow(x, 3), scaled(3, {ipow(x, 2)}), scaled(3, {x}), one});
                  expands_to(t, u, expected, "(x+1)^3");
              })
        .test("distribute_over_sum",
              [&](TestContext& t) {
                  // a*(b + c) -> a*b + a*c
                  const Expr u = Expr::product({a, b.add(c)});
                  const Expr expected =
                      Expr::sum({Expr::product({a, b}), Expr::product({a, c})});
                  expands_to(t, u, expected, "a*(b+c)");
              })
        .test("trinomial_square_multinomial",
              [&](TestContext& t) {
                  // (a + b + c)^2 -> a^2 + b^2 + c^2 + 2ab + 2ac + 2bc  (m > 2 path)
                  const Expr u = ipow(Expr::sum({a, b, c}), 2);
                  const Expr expected = Expr::sum({ipow(a, 2), ipow(b, 2), ipow(c, 2),
                                                   scaled(2, {a, b}), scaled(2, {a, c}),
                                                   scaled(2, {b, c})});
                  expands_to(t, u, expected, "(a+b+c)^2");
              })
        .test("nested_power_times_sum",
              [&](TestContext& t) {
                  // ((x + 1)^2)*(x + 2) -> x^3 + 4x^2 + 5x + 2
                  const Expr u = Expr::product(
                      {ipow(x.add(one), 2), Expr::sum({x, Expr::integer(2)})});
                  const Expr expected =
                      Expr::sum({ipow(x, 3), scaled(4, {ipow(x, 2)}), scaled(5, {x}),
                                 Expr::integer(2)});
                  expands_to(t, u, expected, "((x+1)^2)*(x+2)");
              })
        .test("symbolic_exponent_left_unexpanded",
              [&](TestContext& t) {
                  // (x + 1)^n with symbolic n: no distribution; still a PowerNode.
                  const Expr u = Expr::power(x.add(one), Expr::symbol("n"));
                  auto got = expand(u);
                  t.expect(got.has_value(), "(x+1)^n expands without error");
                  if (!got.has_value()) {
                      return;
                  }
                  t.expect(std::holds_alternative<nimblecas::PowerNode>(got->node().value),
                           std::format("(x+1)^n stays a power, got {}", got->to_string()));
                  auto want = simplify(u);
                  t.expect(want.has_value() && got->is_equivalent_to(*want),
                           "(x+1)^n == simplify((x+1)^n)");
              })
        .test("negative_exponent_left_unexpanded",
              [&](TestContext& t) {
                  // (x + 1)^(-2): negative integer exponent is left intact.
                  const Expr u = Expr::power(x.add(one), Expr::integer(-2));
                  auto got = expand(u);
                  t.expect(got.has_value(), "(x+1)^-2 expands without error");
                  if (got.has_value()) {
                      t.expect(
                          std::holds_alternative<nimblecas::PowerNode>(got->node().value),
                          std::format("(x+1)^-2 stays a power, got {}", got->to_string()));
                  }
              })
        .test("over_cap_exponent_left_unexpanded",
              [&](TestContext& t) {
                  // Exponent beyond max_expand_exponent (64): left as a power, not
                  // distributed. Must not hang and must not error.
                  const Expr u = ipow(x.add(one), 1000);
                  auto got = expand(u);
                  t.expect(got.has_value(), "(x+1)^1000 expands without error");
                  if (got.has_value()) {
                      t.expect(
                          std::holds_alternative<nimblecas::PowerNode>(got->node().value),
                          std::format("(x+1)^1000 stays a power, got {}", got->to_string()));
                  }
              })
        .test("idempotent_on_flat_expression",
              [&](TestContext& t) {
                  // Expanding an already-flat sum of monomials is a no-op (up to
                  // canonicalisation), and expanding twice equals expanding once.
                  const Expr flat =
                      Expr::sum({Expr::product({a, b}), Expr::product({a, c})});
                  auto once = expand(flat);
                  t.expect(once.has_value(), "flat expression expands");
                  if (!once.has_value()) {
                      return;
                  }
                  auto twice = expand(*once);
                  t.expect(twice.has_value() && twice->is_equivalent_to(*once),
                           "expand is idempotent on a flat expression");
                  auto simplified = simplify(flat);
                  t.expect(simplified.has_value() && once->is_equivalent_to(*simplified),
                           "expand of a flat expression equals simplify");
              })
        .test("nested_sum_distribution_collapses_zero",
              [&](TestContext& t) {
                  // Direct regression guard for the as_terms fixpoint fix: a scalar (here
                  // -1) multiplying a product-of-sums that only becomes a sum AFTER its own
                  // distribution must be fully distributed, so a genuinely-zero difference
                  // collapses to 0. Before the fix, expand_product left -1*(a*c+...) intact
                  // (simplify never distributes a scalar over a sum) and this stayed nonzero.
                  const Expr d = Expr::symbol("d");
                  const Expr e = Expr::symbol("e");
                  // ((a+b)*(c+d) + e) - (a*c + a*d + b*c + b*d + e)  ==  0
                  const Expr lhs = Expr::sum(
                      {Expr::product({a.add(b), c.add(d)}), e});
                  const Expr rhs = Expr::sum({Expr::product({a, c}), Expr::product({a, d}),
                                             Expr::product({b, c}), Expr::product({b, d}), e});
                  const Expr diff =
                      Expr::sum({lhs, Expr::product({Expr::integer(-1), rhs})});
                  auto got = expand(diff);
                  t.expect(got.has_value(), "nested-sum difference expands without error");
                  if (got.has_value()) {
                      t.expect(got->is_equivalent_to(Expr::integer(0)),
                               std::format("nested-sum difference expands to 0, got {}",
                                           got->to_string()));
                  }
              })
        .run();
}
