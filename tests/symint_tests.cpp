// Tests for nimblecas.symint: Expr-level symbolic integration int f dx.
// @author Olumuyiwa Oluwasanmi
//
// The correctness discipline for this module is the differentiate-back guarantee: every
// indefinite antiderivative F returned by integrate() must satisfy dF/dx == f exactly.
// Each indefinite case below both matches the expected closed form and verifies that
// nimblecas.diff differentiates the answer back to the integrand (Rule 32 honesty).

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.diff;
import nimblecas.symint;
import nimblecas.testing;

using nimblecas::differentiate;
using nimblecas::Expr;
using nimblecas::integrate;
using nimblecas::integrate_definite;
using nimblecas::MathError;
using nimblecas::simplify;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// simplify(e), or the unsimplified expression if simplification somehow fails (so a
// comparison helper never silently swallows a value).
[[nodiscard]] auto simplified(const Expr& e) -> Expr {
    auto s = simplify(e);
    return s ? *s : e;
}

// Structural equality after simplifying both sides to canonical form.
[[nodiscard]] auto same(const Expr& a, const Expr& b) -> bool {
    return simplified(a).is_equivalent_to(simplified(b));
}

// The honesty check: d/dx(F) == integrand, compared in canonical form.
[[nodiscard]] auto diffs_back(const Expr& antideriv, const Expr& integrand,
                              std::string_view var) -> bool {
    auto d = differentiate(antideriv, var);
    return d && same(*d, integrand);
}

}  // namespace

auto main() -> int {
    const auto x = Expr::symbol("x");
    const auto i = [](std::int64_t v) { return Expr::integer(v); };
    const auto rat = [](std::int64_t n, std::int64_t d) { return Expr::rational(n, d).value(); };
    const auto fn = [](std::string name, const Expr& arg) {
        return Expr::apply(std::move(name), {arg});
    };

    return TestSuite("nimblecas.symint")
        .test("power_rule_x_squared",
              [&](TestContext& t) {
                  const Expr f = x.pow(i(2));  // x^2
                  auto F = integrate(f, "x");
                  t.expect(F.has_value(), "int x^2 dx computed");
                  if (F) {
                      // x^3 / 3
                      t.expect(same(*F, Expr::product({rat(1, 3), x.pow(i(3))})),
                               "int x^2 dx = x^3/3");
                      t.expect(diffs_back(*F, f, "x"), "d/dx(x^3/3) = x^2");
                  }
              })
        .test("polynomial_linearity",
              [&](TestContext& t) {
                  // 3x^2 + 2x + 1
                  const Expr f = Expr::sum({Expr::product({i(3), x.pow(i(2))}),
                                            Expr::product({i(2), x}), i(1)});
                  auto F = integrate(f, "x");
                  t.expect(F.has_value(), "int (3x^2+2x+1) dx computed");
                  if (F) {
                      // x^3 + x^2 + x
                      const Expr expected = Expr::sum({x.pow(i(3)), x.pow(i(2)), x});
                      t.expect(same(*F, expected), "int (3x^2+2x+1) dx = x^3+x^2+x");
                      t.expect(diffs_back(*F, f, "x"), "derivative returns 3x^2+2x+1");
                  }
              })
        .test("exp_direct",
              [&](TestContext& t) {
                  const Expr f = fn("exp", x);  // e^x
                  auto F = integrate(f, "x");
                  t.expect(F.has_value() && same(*F, fn("exp", x)), "int e^x dx = e^x");
                  if (F) t.expect(diffs_back(*F, f, "x"), "d/dx(e^x) = e^x");
              })
        .test("exp_linear_substitution",
              [&](TestContext& t) {
                  const Expr f = fn("exp", Expr::product({i(2), x}));  // e^(2x)
                  auto F = integrate(f, "x");
                  t.expect(F.has_value(), "int e^(2x) dx computed");
                  if (F) {
                      t.expect(same(*F, Expr::product({rat(1, 2), fn("exp", Expr::product({i(2), x}))})),
                               "int e^(2x) dx = e^(2x)/2");
                      t.expect(diffs_back(*F, f, "x"), "d/dx(e^(2x)/2) = e^(2x)");
                  }
              })
        .test("sine_direct",
              [&](TestContext& t) {
                  const Expr f = fn("sin", x);
                  auto F = integrate(f, "x");
                  t.expect(F.has_value() && same(*F, Expr::product({i(-1), fn("cos", x)})),
                           "int sin(x) dx = -cos(x)");
                  if (F) t.expect(diffs_back(*F, f, "x"), "d/dx(-cos(x)) = sin(x)");
              })
        .test("cosine_linear_substitution",
              [&](TestContext& t) {
                  const Expr f = fn("cos", Expr::product({i(3), x}));  // cos(3x)
                  auto F = integrate(f, "x");
                  t.expect(F.has_value(), "int cos(3x) dx computed");
                  if (F) {
                      t.expect(same(*F, Expr::product({rat(1, 3), fn("sin", Expr::product({i(3), x}))})),
                               "int cos(3x) dx = sin(3x)/3");
                      t.expect(diffs_back(*F, f, "x"), "d/dx(sin(3x)/3) = cos(3x)");
                  }
              })
        .test("reciprocal_is_log",
              [&](TestContext& t) {
                  const Expr f = x.pow(i(-1));  // 1/x
                  auto F = integrate(f, "x");
                  t.expect(F.has_value() && same(*F, fn("ln", x)), "int 1/x dx = ln(x)");
                  if (F) t.expect(diffs_back(*F, f, "x"), "d/dx(ln(x)) = 1/x");
              })
        .test("arctangent_form",
              [&](TestContext& t) {
                  // 1/(x^2 + 1)
                  const Expr f = Expr::sum({x.pow(i(2)), i(1)}).pow(i(-1));
                  auto F = integrate(f, "x");
                  t.expect(F.has_value() && same(*F, fn("atan", x)),
                           "int 1/(x^2+1) dx = atan(x)");
                  if (F) t.expect(diffs_back(*F, f, "x"), "d/dx(atan(x)) = 1/(x^2+1)");
              })
        .test("negative_power_rule",
              [&](TestContext& t) {
                  const Expr f = x.pow(i(-2));  // x^-2
                  auto F = integrate(f, "x");
                  t.expect(F.has_value() && same(*F, Expr::product({i(-1), x.pow(i(-1))})),
                           "int x^-2 dx = -x^-1");
                  if (F) t.expect(diffs_back(*F, f, "x"), "d/dx(-x^-1) = x^-2");
              })
        .test("arcsine_form",
              [&](TestContext& t) {
                  // 1/sqrt(1 - x^2) = (1 - x^2)^(-1/2)
                  const Expr base = Expr::sum({i(1), Expr::product({i(-1), x.pow(i(2))})});
                  const Expr f = base.pow(rat(-1, 2));
                  auto F = integrate(f, "x");
                  t.expect(F.has_value() && same(*F, fn("asin", x)),
                           "int 1/sqrt(1-x^2) dx = asin(x)");
                  if (F) t.expect(diffs_back(*F, f, "x"), "d/dx(asin(x)) = 1/sqrt(1-x^2)");
              })
        .test("rational_bridge_simple_pole",
              [&](TestContext& t) {
                  // x/(x^2 + 1) -> (1/2) ln(x^2 + 1); this form differentiates back cleanly
                  // (a single product, no common-denominator recombination needed).
                  const Expr f = Expr::product({x, Expr::sum({x.pow(i(2)), i(1)}).pow(i(-1))});
                  auto F = integrate(f, "x");
                  t.expect(F.has_value(), "int x/(x^2+1) dx computed via rational bridge");
                  if (F) t.expect(diffs_back(*F, f, "x"), "d/dx((1/2)ln(x^2+1)) = x/(x^2+1)");
              })
        .test("definite_integral",
              [&](TestContext& t) {
                  const Expr f = x.pow(i(2));  // int_0^1 x^2 dx = 1/3
                  auto value = integrate_definite(f, "x", i(0), i(1));
                  t.expect(value.has_value() && same(*value, rat(1, 3)),
                           "int_0^1 x^2 dx = 1/3");
              })
        .test("unsupported_is_honest",
              [&](TestContext& t) {
                  // e^(x^2) has no elementary antiderivative in the supported class.
                  const Expr f = fn("exp", x.pow(i(2)));
                  auto F = integrate(f, "x");
                  t.expect(!F.has_value() && F.error() == MathError::not_implemented,
                           "int e^(x^2) dx -> not_implemented (never a guessed answer)");
              })
        .run();
}
