// Tests for nimblecas.limits: symbolic limits / L'Hopital.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.limits;
import nimblecas.testing;

using nimblecas::Expr;
using nimblecas::limit;
using nimblecas::limit_at_infinity;
using nimblecas::MathError;
using nimblecas::Result;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

auto intg(std::int64_t n) -> Expr { return Expr::integer(n); }
auto powi(const Expr& base, std::int64_t exp) -> Expr {
    return Expr::power(base, Expr::integer(exp));
}
// Build the quotient num / den as num * den^(-1).
auto frac(const Expr& num, const Expr& den) -> Expr {
    return Expr::product({num, Expr::power(den, Expr::integer(-1))});
}

auto expect_val(TestContext& t, const Result<Expr>& got, const Expr& expected,
                std::string_view what) -> void {
    if (!got.has_value()) {
        t.expect(false, std::format("{}: unexpected error '{}'", what,
                                    nimblecas::to_string_view(got.error())));
        return;
    }
    t.expect(got->is_equivalent_to(expected),
             std::format("{}: got {}", what, got->to_string()));
}

auto expect_err(TestContext& t, const Result<Expr>& got, MathError expected,
                std::string_view what) -> void {
    if (got.has_value()) {
        t.expect(false, std::format("{}: expected error but got {}", what, got->to_string()));
        return;
    }
    t.expect(got.error() == expected,
             std::format("{}: got error '{}'", what, nimblecas::to_string_view(got.error())));
}

}  // namespace

auto main() -> int {
    const auto x = Expr::symbol("x");
    const auto pos_inf = Expr::apply("inf", {});
    const auto neg_inf = Expr::apply("neg_inf", {});

    return TestSuite("nimblecas.limits")
        .test("continuity_direct_substitution",
              [&](TestContext& t) {
                  // lim_{x->2} (x^2 + 1) = 5
                  auto f = Expr::sum({powi(x, 2), intg(1)});
                  expect_val(t, limit(f, "x", intg(2)), intg(5), "lim x->2 (x^2+1) = 5");
              })
        .test("removable_0_over_0_via_lhopital",
              [&](TestContext& t) {
                  // lim_{x->1} (x^2 - 1)/(x - 1) = 2
                  auto f1 = frac(Expr::sum({powi(x, 2), intg(-1)}), Expr::sum({x, intg(-1)}));
                  expect_val(t, limit(f1, "x", intg(1)), intg(2),
                             "lim x->1 (x^2-1)/(x-1) = 2");
                  // lim_{x->1} (x^3 - 1)/(x - 1) = 3
                  auto f2 = frac(Expr::sum({powi(x, 3), intg(-1)}), Expr::sum({x, intg(-1)}));
                  expect_val(t, limit(f2, "x", intg(1)), intg(3),
                             "lim x->1 (x^3-1)/(x-1) = 3");
              })
        .test("rational_at_infinity_equal_and_lower_degree",
              [&](TestContext& t) {
                  // lim_{x->inf} (2x^2 + 3)/(x^2 - 1) = 2  (equal degree -> lead ratio)
                  auto num = Expr::sum({Expr::product({intg(2), powi(x, 2)}), intg(3)});
                  auto den = Expr::sum({powi(x, 2), intg(-1)});
                  expect_val(t, limit_at_infinity(frac(num, den), "x", true), intg(2),
                             "lim x->inf (2x^2+3)/(x^2-1) = 2");
                  // lim_{x->inf} (x + 1)/x^2 = 0  (lower degree)
                  auto g = frac(Expr::sum({x, intg(1)}), powi(x, 2));
                  expect_val(t, limit_at_infinity(g, "x", true), intg(0),
                             "lim x->inf (x+1)/x^2 = 0");
              })
        .test("divergent_to_signed_infinity",
              [&](TestContext& t) {
                  // lim_{x->+inf} (x^2 + 1)/(x + 1) = +inf  (higher numerator degree)
                  auto f = frac(Expr::sum({powi(x, 2), intg(1)}), Expr::sum({x, intg(1)}));
                  expect_val(t, limit_at_infinity(f, "x", true), pos_inf,
                             "lim x->+inf (x^2+1)/(x+1) = +inf");
                  // lim_{x->-inf} (x^3 + 1)/x^2 = -inf  (odd degree gap flips sign)
                  auto g = frac(Expr::sum({powi(x, 3), intg(1)}), powi(x, 2));
                  expect_val(t, limit_at_infinity(g, "x", false), neg_inf,
                             "lim x->-inf (x^3+1)/x^2 = -inf");
              })
        .test("honest_failures",
              [&](TestContext& t) {
                  // Oscillatory / essential singularity: sin(1/x) as x->0 is undecidable here.
                  auto osc = Expr::apply("sin", {powi(x, -1)});
                  expect_err(t, limit(osc, "x", intg(0)), MathError::not_implemented,
                             "lim x->0 sin(1/x) -> not_implemented");
                  // Genuine finite pole: 1/(x - 1) as x->1 has no finite two-sided limit.
                  auto pole = powi(Expr::sum({x, intg(-1)}), -1);
                  expect_err(t, limit(pole, "x", intg(1)), MathError::domain_error,
                             "lim x->1 1/(x-1) -> domain_error");
                  // Non-rational behaviour at infinity is out of scope here.
                  auto transc = Expr::apply("exp", {x});
                  expect_err(t, limit_at_infinity(transc, "x", true), MathError::not_implemented,
                             "lim x->inf exp(x) -> not_implemented");
              })
        .test("symbolic_leading_coeff_at_infinity_is_not_guessed",
              [&](TestContext& t) {
                  // Regression: a SYMBOLIC denominator leading coefficient could vanish,
                  // so the degree comparison is unsound and must NOT emit a value.
                  const auto a = Expr::symbol("a");
                  // 1/(a*x + 1): 0 for a!=0 but 1 at a=0 -> not_implemented, not 0.
                  auto f1 = frac(intg(1), Expr::sum({Expr::product({a, x}), intg(1)}));
                  expect_err(t, limit_at_infinity(f1, "x", true), MathError::not_implemented,
                             "lim x->inf 1/(a*x+1) -> not_implemented (a may be 0)");
                  // x/(a*x^2 + x): 0 for a!=0 but 1 at a=0 -> not_implemented, not 0.
                  auto f2 = frac(x, Expr::sum({Expr::product({a, powi(x, 2)}), x}));
                  expect_err(t, limit_at_infinity(f2, "x", true), MathError::not_implemented,
                             "lim x->inf x/(a*x^2+x) -> not_implemented (a may be 0)");
              })
        .test("fractional_power_at_domain_boundary_is_not_guessed",
              [&](TestContext& t) {
                  // Regression: sqrt(x) at 0 folds to 0 by substitution, but 0 is the
                  // boundary of sqrt's domain (undefined for x<0), so the two-sided /
                  // left limit does not exist -> honest error, never the value 0.
                  auto sqrt_x = Expr::power(x, Expr::rational(1, 2).value());
                  auto res = limit(sqrt_x, "x", intg(0));
                  t.expect(!res.has_value(),
                           std::format("lim x->0 sqrt(x): must not return a value, got {}",
                                       res.has_value() ? res->to_string() : std::string("<err>")));
                  if (!res.has_value()) {
                      t.expect(res.error() == MathError::not_implemented,
                               "lim x->0 sqrt(x) -> not_implemented");
                  }
                  // Interior point is unaffected: an ordinary polynomial still substitutes.
                  expect_val(t, limit(Expr::sum({powi(x, 2), intg(1)}), "x", intg(3)), intg(10),
                             "lim x->3 (x^2+1) = 10 (continuity intact)");
              })
        .test("crafted_polynomial_power_is_bounded_not_hung",
              [&](TestContext& t) {
                  // Regression: (x^256 + 1)^256 would expand to a degree-65536 dense
                  // polynomial (~1e9 nodes). The total-degree cap must reject it FAST as
                  // not_implemented rather than hang / exhaust memory.
                  auto inner = Expr::sum({powi(x, 256), intg(1)});
                  auto big = frac(Expr::power(inner, intg(256)), x);
                  expect_err(t, limit_at_infinity(big, "x", true), MathError::not_implemented,
                             "lim x->inf (x^256+1)^256/x -> not_implemented (degree-capped)");
              })
        .run();
}
