// Tests for nimblecas.symconst: symbolic mathematical-constant leaves (pi, e, gamma, phi)
// and the numeric bridge back to the arbitrary-precision constants provider.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.symconst;
import nimblecas.diff;
import nimblecas.testing;

using nimblecas::Expr;
using nimblecas::MathError;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Absolute-error comparison for the rounded numeric-bridge values. 1e-9 is far coarser
// than the ~29 significant digits evaluate_constant_double carries, so it only guards
// against a wrong constant or a broken operator, never against last-ULP rounding.
[[nodiscard]] auto near(double actual, double expected) -> bool {
    return std::abs(actual - expected) < 1e-9;
}

// Reference values (IEEE double) of the four constants.
inline constexpr double kPi = 3.141592653589793;
inline constexpr double kE = 2.718281828459045;
inline constexpr double kPhi = 1.618033988749895;
inline constexpr double kGamma = 0.5772156649015329;

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.symconst")
        .test("factories_and_names",
              [](TestContext& t) {
                  // Each factory yields the reserved-name symbol and reports that name.
                  t.expect(nimblecas::named_constant_name(nimblecas::pi()) == "pi",
                           "pi() names \"pi\"");
                  t.expect(nimblecas::named_constant_name(nimblecas::e()) == "e",
                           "e() names \"e\"");
                  t.expect(nimblecas::named_constant_name(nimblecas::euler_gamma()) == "gamma",
                           "euler_gamma() names \"gamma\"");
                  t.expect(nimblecas::named_constant_name(nimblecas::golden_ratio()) == "phi",
                           "golden_ratio() names \"phi\"");
                  // The four leaves are pairwise distinct symbols.
                  t.expect(!nimblecas::pi().is_equivalent_to(nimblecas::e()), "pi != e");
              })
        .test("is_named_constant",
              [](TestContext& t) {
                  t.expect(nimblecas::is_named_constant(nimblecas::pi()), "pi() is a named constant");
                  t.expect(nimblecas::is_named_constant(nimblecas::golden_ratio()),
                           "golden_ratio() is a named constant");
                  // Ordinary variables and numeric literals are not named constants.
                  t.expect(!nimblecas::is_named_constant(Expr::symbol("x")), "x is not a constant");
                  t.expect(!nimblecas::is_named_constant(Expr::integer(3)), "3 is not a constant");
                  t.expect(!nimblecas::named_constant_name(Expr::symbol("x")).has_value(),
                           "name of x is nullopt");
              })
        .test("numeric_bridge_single_leaves",
              [](TestContext& t) {
                  // pi ~ 3.14159..., e ~ 2.71828..., phi ~ 1.61803..., gamma ~ 0.57721...
                  t.expect(near(nimblecas::evaluate_constant_double(nimblecas::pi()).value(), kPi),
                           "pi ~ 3.14159265");
                  t.expect(near(nimblecas::evaluate_constant_double(nimblecas::e()).value(), kE),
                           "e ~ 2.71828183");
                  t.expect(near(nimblecas::evaluate_constant_double(nimblecas::golden_ratio()).value(),
                                kPhi),
                           "phi ~ 1.61803399");
                  t.expect(near(nimblecas::evaluate_constant_double(nimblecas::euler_gamma()).value(),
                                kGamma),
                           "gamma ~ 0.57721566");
                  // The BigFloat path (evaluate_constant at an explicit precision) agrees.
                  auto bf = nimblecas::evaluate_constant(nimblecas::pi(), 128);
                  t.expect(bf.has_value() && near(bf.value().to_double(), kPi),
                           "evaluate_constant(pi, 128) ~ 3.14159265");
              })
        .test("numeric_bridge_compound",
              [](TestContext& t) {
                  // 2 * pi
                  const Expr two_pi = Expr::product({Expr::integer(2), nimblecas::pi()});
                  t.expect(near(nimblecas::evaluate_constant_double(two_pi).value(), 2.0 * kPi),
                           "2*pi ~ 6.28318531");
                  // pi^2 (integer power)
                  const Expr pi_sq = Expr::power(nimblecas::pi(), Expr::integer(2));
                  t.expect(near(nimblecas::evaluate_constant_double(pi_sq).value(), kPi * kPi),
                           "pi^2 ~ 9.86960440");
                  // pi^(-1) (negative integer power -> reciprocal)
                  const Expr inv_pi = Expr::power(nimblecas::pi(), Expr::integer(-1));
                  t.expect(near(nimblecas::evaluate_constant_double(inv_pi).value(), 1.0 / kPi),
                           "pi^-1 ~ 0.31830989");
                  // (1/2) * pi with an exact rational literal
                  const Expr half_pi =
                      Expr::product({Expr::rational(1, 2).value(), nimblecas::pi()});
                  t.expect(near(nimblecas::evaluate_constant_double(half_pi).value(), kPi / 2.0),
                           "(1/2)*pi ~ 1.57079633");
                  // e + phi mixes two different constants under addition
                  const Expr e_plus_phi = Expr::sum({nimblecas::e(), nimblecas::golden_ratio()});
                  t.expect(near(nimblecas::evaluate_constant_double(e_plus_phi).value(), kE + kPhi),
                           "e + phi ~ 4.33631582");
              })
        .test("diff_of_constant_is_zero",
              [](TestContext& t) {
                  // A constant leaf is free of every variable, so d(pi)/dx = 0 falls straight
                  // out of the existing differentiation engine with no special-casing.
                  auto d_pi = nimblecas::differentiate(nimblecas::pi(), "x");
                  t.expect(d_pi.has_value() && d_pi.value().is_equivalent_to(Expr::integer(0)),
                           "d(pi)/dx = 0");
                  // d(x*e)/dx = e (the constant factor is carried through unchanged).
                  auto d_xe = nimblecas::differentiate(
                      Expr::product({Expr::symbol("x"), nimblecas::e()}), "x");
                  t.expect(d_xe.has_value() && d_xe.value().is_equivalent_to(nimblecas::e()),
                           "d(x*e)/dx = e");
              })
        .test("reserved_name_and_errors",
              [](TestContext& t) {
                  // Documented aliasing caveat: a user symbol spelled "pi" IS the constant pi.
                  t.expect(nimblecas::is_named_constant(Expr::symbol("pi")),
                           "Expr::symbol(\"pi\") aliases the constant");
                  t.expect(near(nimblecas::evaluate_constant_double(Expr::symbol("pi")).value(), kPi),
                           "Expr::symbol(\"pi\") evaluates to 3.14159265");
                  // A free variable has no numeric value.
                  t.expect(nimblecas::evaluate_constant(Expr::symbol("x"), 64).error() ==
                               MathError::not_implemented,
                           "free variable -> not_implemented");
                  // A function application is out of scope for the constant bridge.
                  t.expect(nimblecas::evaluate_constant(Expr::apply("sin", {nimblecas::pi()}), 64)
                                   .error() == MathError::not_implemented,
                           "sin(pi) -> not_implemented");
                  // A non-integer (rational) exponent is not numerically tractable here.
                  t.expect(nimblecas::evaluate_constant(
                                   Expr::power(nimblecas::pi(), Expr::rational(1, 2).value()), 64)
                                   .error() == MathError::not_implemented,
                           "pi^(1/2) -> not_implemented");
                  // Non-positive precision is a domain error.
                  t.expect(nimblecas::evaluate_constant(nimblecas::pi(), 0).error() ==
                               MathError::domain_error,
                           "prec <= 0 -> domain_error");
              })
        .run();
}
