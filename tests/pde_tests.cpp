// Tests for nimblecas.pde: exact linear evolution PDEs via power series in time.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.ratpoly;
import nimblecas.pde;
import nimblecas.testing;

using nimblecas::MathError;
using nimblecas::Polynomial;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Build a RationalPoly from integer coefficients (low degree first).
[[nodiscard]] auto ipoly(std::vector<std::int64_t> c) -> RationalPoly {
    return RationalPoly::from_polynomial(Polynomial{std::move(c)});
}

[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

// Verify the discrete Cauchy-Kovalevskaya relation c_n * n == L[c_{n-1}] holds on the
// retained coefficients: this is exactly the truncated statement u_t == L[u].
[[nodiscard]] auto residual_holds(const nimblecas::SpatialOperator& l,
                                  const std::vector<RationalPoly>& c) -> bool {
    for (std::size_t n = 1; n < c.size(); ++n) {
        auto lhs = c[n].scale(Rational::from_int(static_cast<std::int64_t>(n)));  // n * c_n
        auto rhs = l(c[n - 1]);                                                   // L[c_{n-1}]
        if (!lhs || !rhs || !lhs->is_equal(*rhs)) {
            return false;
        }
    }
    return true;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.pde")
        .test("heat_x2_diffusivity_one",
              [](TestContext& t) {
                  // u_t = u_xx, phi = x^2  =>  u = x^2 + 2t  (c_0=x^2, c_1=2, c_2..=0).
                  auto l = nimblecas::heat_operator(Rational::from_int(1));
                  auto c = nimblecas::solve_evolution_pde(l, ipoly({0, 0, 1}), 3).value();
                  t.expect_eq(c.size(), std::size_t{4}, "returns c_0..c_3");
                  t.expect(c[0].is_equal(ipoly({0, 0, 1})), "c_0 = x^2");
                  t.expect(c[1].is_equal(ipoly({2})), "c_1 = 2");
                  t.expect(c[2].is_zero(), "c_2 = 0");
                  t.expect(c[3].is_zero(), "c_3 = 0 (series terminated)");
                  t.expect(residual_holds(l, c), "n*c_n == L[c_{n-1}] (u_t == u_xx)");
                  // u(1, 1) = 1 + 2 = 3.
                  auto v = nimblecas::evaluate(c, Rational::from_int(1), Rational::from_int(1));
                  t.expect(v.value() == Rational::from_int(3), "u(1,1) = 3");
              })
        .test("heat_x4_diffusivity_one",
              [](TestContext& t) {
                  // phi = x^4  =>  u = x^4 + 12 x^2 t + 12 t^2 (c_0=x^4, c_1=12x^2, c_2=12).
                  auto l = nimblecas::heat_operator(Rational::from_int(1));
                  auto c = nimblecas::solve_evolution_pde(l, ipoly({0, 0, 0, 0, 1}), 3).value();
                  t.expect(c[0].is_equal(ipoly({0, 0, 0, 0, 1})), "c_0 = x^4");
                  t.expect(c[1].is_equal(ipoly({0, 0, 12})), "c_1 = 12 x^2");
                  t.expect(c[2].is_equal(ipoly({12})), "c_2 = 12");
                  t.expect(c[3].is_zero(), "c_3 = 0");
                  t.expect(residual_holds(l, c), "n*c_n == L[c_{n-1}]");
                  // u(1, 1) = 1 + 12 + 12 = 25.
                  auto v = nimblecas::evaluate(c, Rational::from_int(1), Rational::from_int(1));
                  t.expect(v.value() == Rational::from_int(25), "u(1,1) = 25");
              })
        .test("transport_x2_speed_one",
              [](TestContext& t) {
                  // u_t = u_x, phi = x^2  =>  u = (x+t)^2 = x^2 + 2 x t + t^2.
                  auto l = nimblecas::transport_operator(Rational::from_int(1));
                  auto c = nimblecas::solve_evolution_pde(l, ipoly({0, 0, 1}), 3).value();
                  t.expect(c[0].is_equal(ipoly({0, 0, 1})), "c_0 = x^2");
                  t.expect(c[1].is_equal(ipoly({0, 2})), "c_1 = 2x");
                  t.expect(c[2].is_equal(ipoly({1})), "c_2 = 1");
                  t.expect(c[3].is_zero(), "c_3 = 0");
                  t.expect(residual_holds(l, c), "n*c_n == L[c_{n-1}] (u_t == u_x)");
                  // Check against expanding (x+t)^2 at (x,t) = (3,2): (3+2)^2 = 25.
                  auto v = nimblecas::evaluate(c, Rational::from_int(3), Rational::from_int(2));
                  t.expect(v.value() == Rational::from_int(25), "u(3,2) = (3+2)^2 = 25");
              })
        .test("heat_x4_diffusivity_half",
              [](TestContext& t) {
                  // Diffusivity 1/2 exercises rational scaling.
                  // phi = x^4  =>  u = x^4 + 6 x^2 t + 3 t^2.
                  auto l = nimblecas::heat_operator(rat(1, 2));
                  auto c = nimblecas::solve_evolution_pde(l, ipoly({0, 0, 0, 0, 1}), 3).value();
                  t.expect(c[0].is_equal(ipoly({0, 0, 0, 0, 1})), "c_0 = x^4");
                  t.expect(c[1].is_equal(ipoly({0, 0, 6})), "c_1 = 6 x^2");
                  t.expect(c[2].is_equal(ipoly({3})), "c_2 = 3");
                  t.expect(c[3].is_zero(), "c_3 = 0");
                  t.expect(residual_holds(l, c), "n*c_n == (1/2) c_{n-1}''");
                  // u(1, 1) = 1 + 6 + 3 = 10.
                  auto v = nimblecas::evaluate(c, Rational::from_int(1), Rational::from_int(1));
                  t.expect(v.value() == Rational::from_int(10), "u(1,1) = 10");
              })
        .test("transport_x2_speed_half",
              [](TestContext& t) {
                  // Speed 1/2 exercises rational scaling.
                  // phi = x^2  =>  u = (x + t/2)^2 = x^2 + x t + (1/4) t^2.
                  auto l = nimblecas::transport_operator(rat(1, 2));
                  auto c = nimblecas::solve_evolution_pde(l, ipoly({0, 0, 1}), 3).value();
                  t.expect(c[0].is_equal(ipoly({0, 0, 1})), "c_0 = x^2");
                  t.expect(c[1].is_equal(ipoly({0, 1})), "c_1 = x");
                  t.expect(c[2].is_equal(RationalPoly::from_coeffs({rat(1, 4)})), "c_2 = 1/4");
                  t.expect(c[3].is_zero(), "c_3 = 0");
                  t.expect(residual_holds(l, c), "n*c_n == (1/2) c_{n-1}'");
                  // u(1, 2) = (1 + 1)^2 = 4.
                  auto v = nimblecas::evaluate(c, Rational::from_int(1), Rational::from_int(2));
                  t.expect(v.value() == Rational::from_int(4), "u(1,2) = (1 + 1)^2 = 4");
              })
        .test("transport_x3_speed_three",
              [](TestContext& t) {
                  // Speed 3 != 1: u_t = 3 u_x, phi = x^2  =>  u = (x + 3t)^2.
                  auto l = nimblecas::transport_operator(Rational::from_int(3));
                  auto c = nimblecas::solve_evolution_pde(l, ipoly({0, 0, 1}), 3).value();
                  t.expect(c[0].is_equal(ipoly({0, 0, 1})), "c_0 = x^2");
                  t.expect(c[1].is_equal(ipoly({0, 6})), "c_1 = 6x");
                  t.expect(c[2].is_equal(ipoly({9})), "c_2 = 9");
                  t.expect(c[3].is_zero(), "c_3 = 0");
                  t.expect(residual_holds(l, c), "n*c_n == 3 c_{n-1}'");
                  // (x + 3t)^2 at (1, 1) = 16.
                  auto v = nimblecas::evaluate(c, Rational::from_int(1), Rational::from_int(1));
                  t.expect(v.value() == Rational::from_int(16), "u(1,1) = (1+3)^2 = 16");
              })
        .test("advection_diffusion_combined",
              [](TestContext& t) {
                  // u_t = u_x + u_xx, phi = x^2.
                  // L[x^2] = 2x + 2 => c_1 = 2x + 2.
                  // L[2x + 2] = 2 => c_2 = 2 * 1/2 = 1.
                  // L[1] = 0 => c_3 = 0.
                  auto l = nimblecas::advection_diffusion(Rational::from_int(1),
                                                          Rational::from_int(1));
                  auto c = nimblecas::solve_evolution_pde(l, ipoly({0, 0, 1}), 3).value();
                  t.expect(c[0].is_equal(ipoly({0, 0, 1})), "c_0 = x^2");
                  t.expect(c[1].is_equal(ipoly({2, 2})), "c_1 = 2 + 2x");
                  t.expect(c[2].is_equal(ipoly({1})), "c_2 = 1");
                  t.expect(c[3].is_zero(), "c_3 = 0");
                  t.expect(residual_holds(l, c), "n*c_n == c_{n-1}' + c_{n-1}''");
                  // u(0, 1) = 0 + 2 + 1 = 3.
                  auto v = nimblecas::evaluate(c, Rational::from_int(0), Rational::from_int(1));
                  t.expect(v.value() == Rational::from_int(3), "u(0,1) = 3");
              })
        .test("evaluate_horner_rational_point",
              [](TestContext& t) {
                  // Independent check of evaluate at a fractional (x, t) for the heat x^4
                  // series u = x^4 + 12 x^2 t + 12 t^2 at x = 1/2, t = 1/3.
                  // = 1/16 + 12*(1/4)*(1/3) + 12*(1/9) = 1/16 + 1 + 4/3 = 115/48.
                  auto l = nimblecas::heat_operator(Rational::from_int(1));
                  auto c = nimblecas::solve_evolution_pde(l, ipoly({0, 0, 0, 0, 1}), 2).value();
                  auto v = nimblecas::evaluate(c, rat(1, 2), rat(1, 3));
                  t.expect(v.value() == rat(115, 48), "u(1/2, 1/3) = 115/48");
              })
        .test("error_paths",
              [](TestContext& t) {
                  auto l = nimblecas::heat_operator(Rational::from_int(1));
                  // order 0 is a domain_error.
                  auto bad = nimblecas::solve_evolution_pde(l, ipoly({0, 0, 1}), 0);
                  t.expect(!bad.has_value(), "order 0 fails");
                  t.expect(bad.error() == MathError::domain_error, "order 0 is domain_error");
                  // an empty operator is a domain_error.
                  auto bad2 = nimblecas::solve_evolution_pde(nimblecas::SpatialOperator{},
                                                             ipoly({1}), 2);
                  t.expect(bad2.error() == MathError::domain_error, "empty operator is domain_error");
                  // evaluate on an empty coefficient list is a domain_error.
                  auto bad3 = nimblecas::evaluate({}, Rational::from_int(1), Rational::from_int(1));
                  t.expect(bad3.error() == MathError::domain_error, "empty evaluate is domain_error");
                  // an error raised by L propagates through solve_evolution_pde.
                  nimblecas::SpatialOperator failing =
                      [](const RationalPoly&) -> nimblecas::Result<RationalPoly> {
                      return nimblecas::make_error<RationalPoly>(MathError::overflow);
                  };
                  auto bad4 = nimblecas::solve_evolution_pde(failing, ipoly({1}), 2);
                  t.expect(bad4.error() == MathError::overflow, "L error propagates");
              })
        .run();
}
