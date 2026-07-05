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
using nimblecas::inverse_laplace;
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
        // --- Dirac / Heaviside forward-table entries -------------------------
        .test("dirac_and_heaviside_forward",
              [&](TestContext& t_ctx) {
                  // L{delta(t)} = 1
                  auto d0 = laplace_transform(Expr::apply("dirac", {t}), "t", "s");
                  t_ctx.expect(d0.has_value() &&
                                   d0.value().is_equivalent_to(simp(Expr::integer(1))),
                               std::format("L{{delta(t)}} = {}", d0.value().to_string()));
                  // L{delta(t - 3)} = e^(-3 s)
                  auto d3 = laplace_transform(
                      Expr::apply("dirac", {Expr::sum({t, Expr::integer(-3)})}), "t", "s");
                  auto d3_expected =
                      simp(Expr::apply("exp", {Expr::product({Expr::integer(-3), s})}));
                  t_ctx.expect(d3.has_value() && d3.value().is_equivalent_to(d3_expected),
                               std::format("L{{delta(t-3)}} = {}", d3.value().to_string()));
                  // L{u(t)} = 1/s
                  auto u0 = laplace_transform(Expr::apply("heaviside", {t}), "t", "s");
                  auto u0_expected = simp(Expr::power(s, Expr::integer(-1)));
                  t_ctx.expect(u0.has_value() && u0.value().is_equivalent_to(u0_expected),
                               std::format("L{{u(t)}} = {}", u0.value().to_string()));
                  // L{u(t - 2)} = e^(-2 s)/s
                  auto u2 = laplace_transform(
                      Expr::apply("heaviside", {Expr::sum({t, Expr::integer(-2)})}), "t", "s");
                  auto u2_expected = simp(Expr::product(
                      {Expr::apply("exp", {Expr::product({Expr::integer(-2), s})}),
                       Expr::power(s, Expr::integer(-1))}));
                  t_ctx.expect(u2.has_value() && u2.value().is_equivalent_to(u2_expected),
                               std::format("L{{u(t-2)}} = {}", u2.value().to_string()));
              })
        // --- Inverse transform: elementary table ----------------------------
        .test("inverse_exponential",
              [&](TestContext& t_ctx) {
                  // L^{-1}{1/(s-2)} = e^(2 t)
                  auto F = Expr::power(Expr::sum({s, Expr::integer(-2)}), Expr::integer(-1));
                  auto f = inverse_laplace(F, "s", "t");
                  auto expected =
                      simp(Expr::apply("exp", {Expr::product({Expr::integer(2), t})}));
                  t_ctx.expect(f.has_value(), "L^-1{1/(s-2)} is handled");
                  t_ctx.expect(f.value().is_equivalent_to(expected),
                               std::format("L^-1{{1/(s-2)}} = {}", f.value().to_string()));
              })
        .test("inverse_sine_and_cosine",
              [&](TestContext& t_ctx) {
                  // L^{-1}{3/(s^2+9)} = sin(3 t)
                  auto Fsin = Expr::product(
                      {Expr::integer(3),
                       Expr::power(Expr::sum({Expr::power(s, Expr::integer(2)),
                                              Expr::integer(9)}),
                                   Expr::integer(-1))});
                  auto fsin = inverse_laplace(Fsin, "s", "t");
                  auto sin_expected =
                      simp(Expr::apply("sin", {Expr::product({Expr::integer(3), t})}));
                  t_ctx.expect(fsin.has_value() &&
                                   fsin.value().is_equivalent_to(sin_expected),
                               std::format("L^-1{{3/(s^2+9)}} = {}", fsin.value().to_string()));
                  // L^{-1}{s/(s^2+1)} = cos(t)
                  auto Fcos = Expr::product(
                      {s, Expr::power(Expr::sum({Expr::power(s, Expr::integer(2)),
                                                 Expr::integer(1)}),
                                      Expr::integer(-1))});
                  auto fcos = inverse_laplace(Fcos, "s", "t");
                  auto cos_expected = simp(Expr::apply("cos", {t}));
                  t_ctx.expect(fcos.has_value() &&
                                   fcos.value().is_equivalent_to(cos_expected),
                               std::format("L^-1{{s/(s^2+1)}} = {}", fcos.value().to_string()));
              })
        .test("inverse_power_and_repeated_pole",
              [&](TestContext& t_ctx) {
                  // L^{-1}{1/s^2} = t
                  auto Ft = Expr::power(s, Expr::integer(-2));
                  auto ft = inverse_laplace(Ft, "s", "t");
                  t_ctx.expect(ft.has_value() && ft.value().is_equivalent_to(simp(t)),
                               std::format("L^-1{{1/s^2}} = {}", ft.value().to_string()));
                  // L^{-1}{1/(s-1)^2} = t e^t
                  auto Fte = Expr::power(Expr::sum({s, Expr::integer(-1)}), Expr::integer(-2));
                  auto fte = inverse_laplace(Fte, "s", "t");
                  auto te_expected = simp(Expr::product({t, Expr::apply("exp", {t})}));
                  t_ctx.expect(fte.has_value() &&
                                   fte.value().is_equivalent_to(te_expected),
                               std::format("L^-1{{1/(s-1)^2}} = {}", fte.value().to_string()));
              })
        .test("inverse_distinct_real_poles",
              [&](TestContext& t_ctx) {
                  // L^{-1}{1/(s^2-1)} = (1/2) e^t - (1/2) e^(-t)  (a square-free quadratic
                  // that factors over Q into two distinct real poles).
                  auto F = Expr::power(
                      Expr::sum({Expr::power(s, Expr::integer(2)), Expr::integer(-1)}),
                      Expr::integer(-1));
                  auto f = inverse_laplace(F, "s", "t");
                  auto expected = simp(Expr::sum(
                      {Expr::product(
                           {Expr::rational(1, 2).value(), Expr::apply("exp", {t})}),
                       Expr::product({Expr::rational(-1, 2).value(),
                                      Expr::apply("exp", {Expr::product({Expr::integer(-1),
                                                                         t})})})}));
                  t_ctx.expect(f.has_value(), "L^-1{1/(s^2-1)} is handled");
                  t_ctx.expect(f.value().is_equivalent_to(expected),
                               std::format("L^-1{{1/(s^2-1)}} = {}", f.value().to_string()));
              })
        .test("inverse_constant_is_dirac",
              [&](TestContext& t_ctx) {
                  // F(s) = 1 is improper (a constant excess): L^{-1}{1} = delta(t).
                  auto f = inverse_laplace(Expr::integer(1), "s", "t");
                  auto expected = simp(Expr::apply("dirac", {t}));
                  t_ctx.expect(f.has_value() && f.value().is_equivalent_to(expected),
                               std::format("L^-1{{1}} = {}", f.value().to_string()));
              })
        // --- Round trips: inverse_laplace inverts laplace_transform ----------
        .test("round_trip",
              [&](TestContext& t_ctx) {
                  const std::vector<std::pair<std::string, Expr>> cases = {
                      {"1", Expr::integer(1)},
                      {"t", t},
                      {"exp(2t)", Expr::apply("exp", {Expr::product({Expr::integer(2), t})})},
                      {"sin(3t)", Expr::apply("sin", {Expr::product({Expr::integer(3), t})})},
                      {"cos(t)", Expr::apply("cos", {t})},
                  };
                  for (const auto& [name, f] : cases) {
                      auto F = laplace_transform(f, "t", "s");
                      t_ctx.expect(F.has_value(), std::format("forward L{{{}}} handled", name));
                      auto back = inverse_laplace(F.value(), "s", "t");
                      t_ctx.expect(back.has_value() &&
                                       back.value().is_equivalent_to(simp(f)),
                                   std::format("round trip {} -> {} -> {}", name,
                                               F.value().to_string(), back.value().to_string()));
                  }
              })
        // --- Honesty boundary: never a wrong inverse ------------------------
        .test("inverse_unsupported_is_not_implemented",
              [&](TestContext& t_ctx) {
                  // Irreducible cubic denominator: outside the table.
                  auto Fcubic = Expr::power(
                      Expr::sum({Expr::power(s, Expr::integer(3)), s, Expr::integer(1)}),
                      Expr::integer(-1));
                  auto rc = inverse_laplace(Fcubic, "s", "t");
                  t_ctx.expect(!rc.has_value() && rc.error() == MathError::not_implemented,
                               "L^-1{1/(s^3+s+1)} yields not_implemented");
                  // Irrational real poles (s^2 - 2): cannot stay over Q.
                  auto Firr = Expr::power(
                      Expr::sum({Expr::power(s, Expr::integer(2)), Expr::integer(-2)}),
                      Expr::integer(-1));
                  auto ri = inverse_laplace(Firr, "s", "t");
                  t_ctx.expect(!ri.has_value() && ri.error() == MathError::not_implemented,
                               "L^-1{1/(s^2-2)} yields not_implemented");
                  // Improper F with a non-constant polynomial part (delta derivative).
                  auto rs = inverse_laplace(s, "s", "t");
                  t_ctx.expect(!rs.has_value() && rs.error() == MathError::not_implemented,
                               "L^-1{s} (delta') yields not_implemented");
                  // A non-rational F (a foreign transcendental) is not invertible here.
                  auto rlog = inverse_laplace(Expr::apply("log", {s}), "s", "t");
                  t_ctx.expect(!rlog.has_value() &&
                                   rlog.error() == MathError::not_implemented,
                               "L^-1{log(s)} yields not_implemented");
              })
        .run();
}
