// Tests for nimblecas.laplace: table-driven symbolic Laplace transform.
// @author Olumuyiwa Oluwasanmi
//
// Each expected image is constructed here from the same Expr factories the module
// uses and then simplified, so the assertions match the module's own canonical
// output deterministically (rather than a hand-guessed normal form).

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.laplace;
import nimblecas.testing;

using nimblecas::Expr;
using nimblecas::laplace_transform;
using nimblecas::MathError;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Simplify an expression into its canonical form (transforms in these tests never
// fail, so unwrapping the Result is safe).
[[nodiscard]] auto simp(const Expr& e) -> Expr { return nimblecas::simplify(e).value(); }

}  // namespace

auto main() -> int {
    const Expr t = Expr::symbol("t");
    const Expr s = Expr::symbol("s");

    return TestSuite("nimblecas.laplace")
        .test("constant_one",
              [&](TestContext& t_ctx) {
                  // L{1} = 1/s
                  auto result = laplace_transform(Expr::integer(1), "t", "s");
                  auto expected = simp(Expr::power(s, Expr::integer(-1)));
                  t_ctx.expect(result.has_value(), "L{1} is handled");
                  t_ctx.expect(result.value().is_equivalent_to(expected),
                               std::format("L{{1}} = {}", result.value().to_string()));
              })
        .test("power_of_t",
              [&](TestContext& t_ctx) {
                  // L{t} = 1/s^2
                  auto lt = laplace_transform(t, "t", "s");
                  auto lt_expected = simp(Expr::power(s, Expr::integer(-2)));
                  t_ctx.expect(lt.has_value() && lt.value().is_equivalent_to(lt_expected),
                               std::format("L{{t}} = {}", lt.value().to_string()));
                  // L{t^2} = 2/s^3
                  auto t2 = Expr::power(t, Expr::integer(2));
                  auto lt2 = laplace_transform(t2, "t", "s");
                  auto lt2_expected = simp(
                      Expr::product({Expr::integer(2), Expr::power(s, Expr::integer(-3))}));
                  t_ctx.expect(lt2.has_value() && lt2.value().is_equivalent_to(lt2_expected),
                               std::format("L{{t^2}} = {}", lt2.value().to_string()));
              })
        .test("exponential",
              [&](TestContext& t_ctx) {
                  // L{exp(2 t)} = 1/(s - 2)
                  auto f = Expr::apply("exp", {Expr::product({Expr::integer(2), t})});
                  auto result = laplace_transform(f, "t", "s");
                  auto expected = simp(
                      Expr::power(Expr::sum({s, Expr::integer(-2)}), Expr::integer(-1)));
                  t_ctx.expect(result.has_value(), "L{exp(2t)} is handled");
                  t_ctx.expect(result.value().is_equivalent_to(expected),
                               std::format("L{{exp(2t)}} = {}", result.value().to_string()));
              })
        .test("sine_and_cosine",
              [&](TestContext& t_ctx) {
                  // L{sin(3 t)} = 3/(s^2 + 9)
                  auto fsin = Expr::apply("sin", {Expr::product({Expr::integer(3), t})});
                  auto lsin = laplace_transform(fsin, "t", "s");
                  auto lsin_expected = simp(Expr::product(
                      {Expr::integer(3),
                       Expr::power(Expr::sum({Expr::power(s, Expr::integer(2)),
                                              Expr::integer(9)}),
                                   Expr::integer(-1))}));
                  t_ctx.expect(lsin.has_value() && lsin.value().is_equivalent_to(lsin_expected),
                               std::format("L{{sin(3t)}} = {}", lsin.value().to_string()));
                  // L{cos(t)} = s/(s^2 + 1)
                  auto fcos = Expr::apply("cos", {t});
                  auto lcos = laplace_transform(fcos, "t", "s");
                  auto lcos_expected = simp(Expr::product(
                      {s, Expr::power(Expr::sum({Expr::power(s, Expr::integer(2)),
                                                 Expr::integer(1)}),
                                      Expr::integer(-1))}));
                  t_ctx.expect(lcos.has_value() && lcos.value().is_equivalent_to(lcos_expected),
                               std::format("L{{cos(t)}} = {}", lcos.value().to_string()));
              })
        .test("linearity",
              [&](TestContext& t_ctx) {
                  // L{2 + 3 t} = 2/s + 3/s^2
                  auto f = Expr::sum({Expr::integer(2), Expr::product({Expr::integer(3), t})});
                  auto result = laplace_transform(f, "t", "s");
                  auto expected = simp(Expr::sum(
                      {Expr::product({Expr::integer(2), Expr::power(s, Expr::integer(-1))}),
                       Expr::product({Expr::integer(3), Expr::power(s, Expr::integer(-2))})}));
                  t_ctx.expect(result.has_value(), "L{2 + 3t} is handled");
                  t_ctx.expect(result.value().is_equivalent_to(expected),
                               std::format("L{{2 + 3t}} = {}", result.value().to_string()));
              })
        .test("unhandled_is_not_implemented",
              [&](TestContext& t_ctx) {
                  // log(t) is not in the table.
                  auto flog = Expr::apply("log", {t});
                  auto rlog = laplace_transform(flog, "t", "s");
                  t_ctx.expect(!rlog.has_value(), "L{log(t)} is rejected");
                  t_ctx.expect(rlog.error() == MathError::not_implemented,
                               "L{log(t)} yields not_implemented");
                  // A product of two t-dependent factors (t * sin(t)) is unhandled.
                  auto prod = Expr::product({t, Expr::apply("sin", {t})});
                  auto rprod = laplace_transform(prod, "t", "s");
                  t_ctx.expect(!rprod.has_value() &&
                                   rprod.error() == MathError::not_implemented,
                               "L{t*sin(t)} yields not_implemented");
              })
        .run();
}
