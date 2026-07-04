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

// Verify the second-order-in-time relation (n+2)(n+1) c_{n+2} == L[c_n] on the retained
// coefficients: the truncated statement of u_tt == L[u].
[[nodiscard]] auto wave_residual_holds(const nimblecas::SpatialOperator& l,
                                       const std::vector<RationalPoly>& c) -> bool {
    for (std::size_t n = 0; n + 2 < c.size(); ++n) {
        auto factor = Rational::from_int(static_cast<std::int64_t>((n + 2) * (n + 1)));
        auto lhs = c[n + 2].scale(factor);  // (n+2)(n+1) c_{n+2}
        auto rhs = l(c[n]);                 // L[c_n]
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
        // ---- Nonlinear evolution: Adomian / Cauchy-Kovalevskaya time series ----
        .test("reaction_u2_constant_geometric",
              [](TestContext& t) {
                  // u_t = u_xx + u^2, phi = 1 (constant => u_xx = 0) reduces to the ODE
                  // u_t = u^2, u(0)=1, whose exact solution is 1/(1-t) = 1 + t + t^2 + ...
                  // so every time coefficient c_n(x) is the constant 1.
                  auto c =
                      nimblecas::reaction_diffusion_quadratic(Rational::from_int(1), ipoly({1}), 3)
                          .value();
                  t.expect_eq(c.size(), std::size_t{4}, "returns c_0..c_3");
                  t.expect(c[0].is_equal(ipoly({1})), "c_0 = 1");
                  t.expect(c[1].is_equal(ipoly({1})), "c_1 = 1");
                  t.expect(c[2].is_equal(ipoly({1})), "c_2 = 1");
                  t.expect(c[3].is_equal(ipoly({1})), "c_3 = 1");
                  // Truncated evaluation is the partial geometric sum, NOT the closed form:
                  // at t=1/2 the order-3 truncation is 1+1/2+1/4+1/8 = 15/8 (not 1/(1-1/2)=2).
                  auto v = nimblecas::evaluate(c, Rational::from_int(0), rat(1, 2));
                  t.expect(v.value() == rat(15, 8), "truncated u(0,1/2) = 15/8 (local, not global)");
              })
        .test("reaction_u2_linear_datum",
              [](TestContext& t) {
                  // u_t = u_xx + u^2, phi = x. Hand-derived graded recurrence
                  // (k+1)c_{k+1} = c_k'' + [t^k]u^2:
                  //   c_0 = x, c_1 = x^2, c_2 = 1 + x^3, c_3 = (8/3) x + x^4.
                  auto c =
                      nimblecas::reaction_diffusion_quadratic(Rational::from_int(1), ipoly({0, 1}), 3)
                          .value();
                  t.expect(c[0].is_equal(ipoly({0, 1})), "c_0 = x");
                  t.expect(c[1].is_equal(ipoly({0, 0, 1})), "c_1 = x^2");
                  t.expect(c[2].is_equal(ipoly({1, 0, 0, 1})), "c_2 = 1 + x^3");
                  t.expect(c[3].is_equal(RationalPoly::from_coeffs(
                               {rat(0, 1), rat(8, 3), rat(0, 1), rat(0, 1), rat(1, 1)})),
                           "c_3 = (8/3) x + x^4");
              })
        .test("inviscid_burgers_x_datum",
              [](TestContext& t) {
                  // u_t + u u_x = 0 (viscosity 0), phi = x. Exact solution x/(1+t) =
                  // x - x t + x t^2 - x t^3 + ..., so c_n = (-1)^n x.
                  auto c = nimblecas::burgers(Rational::from_int(0), ipoly({0, 1}), 3).value();
                  t.expect(c[0].is_equal(ipoly({0, 1})), "c_0 = x");
                  t.expect(c[1].is_equal(ipoly({0, -1})), "c_1 = -x");
                  t.expect(c[2].is_equal(ipoly({0, 1})), "c_2 = x");
                  t.expect(c[3].is_equal(ipoly({0, -1})), "c_3 = -x");
                  // Truncated u(1, 1/2): 1 - 1/2 + 1/4 - 1/8 = 5/8 (partial sum of x/(1+t)).
                  auto v = nimblecas::evaluate(c, Rational::from_int(1), rat(1, 2));
                  t.expect(v.value() == rat(5, 8), "truncated u(1,1/2) = 5/8");
              })
        .test("viscous_burgers_matches_inviscid_on_linear_datum",
              [](TestContext& t) {
                  // For phi = x every c_n is linear in x, so u_xx = 0 and the viscous term
                  // vanishes: viscosity=1 must reproduce the inviscid c_n = (-1)^n x.
                  auto c = nimblecas::burgers(Rational::from_int(1), ipoly({0, 1}), 3).value();
                  t.expect(c[1].is_equal(ipoly({0, -1})), "c_1 = -x (diffusion of a linear term is 0)");
                  t.expect(c[2].is_equal(ipoly({0, 1})), "c_2 = x");
              })
        .test("nonlinear_error_paths",
              [](TestContext& t) {
                  // order 0 is a domain_error.
                  auto bad = nimblecas::reaction_diffusion_quadratic(Rational::from_int(1),
                                                                     ipoly({0, 1}), 0);
                  t.expect(bad.error() == MathError::domain_error, "order 0 is domain_error");
                  // a null nonlinear term is a domain_error.
                  auto bad2 = nimblecas::solve_nonlinear_evolution_pde(
                      nimblecas::heat_operator(Rational::from_int(1)),
                      nimblecas::TimeSeriesOperator{}, ipoly({0, 1}), 3);
                  t.expect(bad2.error() == MathError::domain_error, "null nonlinear is domain_error");
                  // an error raised by N propagates.
                  nimblecas::TimeSeriesOperator failing =
                      [](const std::vector<RationalPoly>&)
                      -> nimblecas::Result<std::vector<RationalPoly>> {
                      return nimblecas::make_error<std::vector<RationalPoly>>(MathError::overflow);
                  };
                  auto bad3 = nimblecas::solve_nonlinear_evolution_pde(
                      nimblecas::SpatialOperator{}, failing, ipoly({0, 1}), 3);
                  t.expect(bad3.error() == MathError::overflow, "N error propagates");
              })
        // ---- Boundary-value / steady-state: exact 1-D Poisson over Q ----
        .test("poisson_uxx_2_zero_dirichlet",
              [](TestContext& t) {
                  // u'' = 2 on [0,1], u(0)=0, u(1)=0  =>  u = x^2 - x (exact over Q).
                  auto u = nimblecas::solve_poisson_bvp_1d(ipoly({2}), Rational::from_int(0),
                                                           Rational::from_int(0),
                                                           Rational::from_int(1),
                                                           Rational::from_int(0))
                               .value();
                  t.expect(u.is_equal(ipoly({0, -1, 1})), "u = x^2 - x");
              })
        .test("poisson_uxx_x_zero_dirichlet",
              [](TestContext& t) {
                  // u'' = x on [0,1], u(0)=0, u(1)=0  =>  u = x^3/6 - x/6 (exact rational).
                  auto u = nimblecas::solve_poisson_bvp_1d(ipoly({0, 1}), Rational::from_int(0),
                                                           Rational::from_int(0),
                                                           Rational::from_int(1),
                                                           Rational::from_int(0))
                               .value();
                  t.expect(u.is_equal(RationalPoly::from_coeffs(
                               {rat(0, 1), rat(-1, 6), rat(0, 1), rat(1, 6)})),
                           "u = x^3/6 - x/6");
              })
        .test("laplace_linear_interpolation",
              [](TestContext& t) {
                  // u'' = 0 on [0,2], u(0)=1, u(2)=5  =>  u = 1 + 2x (harmonic = linear in 1-D).
                  auto u = nimblecas::solve_poisson_bvp_1d(RationalPoly{}, Rational::from_int(0),
                                                           Rational::from_int(1),
                                                           Rational::from_int(2),
                                                           Rational::from_int(5))
                               .value();
                  t.expect(u.is_equal(ipoly({1, 2})), "u = 1 + 2x");
              })
        .test("poisson_degenerate_interval",
              [](TestContext& t) {
                  // a == b collapses the two boundary conditions: ill-posed => domain_error.
                  auto bad = nimblecas::solve_poisson_bvp_1d(ipoly({2}), Rational::from_int(1),
                                                             Rational::from_int(0),
                                                             Rational::from_int(1),
                                                             Rational::from_int(0));
                  t.expect(!bad.has_value(), "a == b fails");
                  t.expect(bad.error() == MathError::domain_error, "degenerate interval is domain_error");
              })
        // ---- Wave equation: second-order-in-time linear evolution ----
        .test("wave_x2_speed_one_static",
              [](TestContext& t) {
                  // u_tt = u_xx, u(x,0)=x^2, u_t(x,0)=0. c_0=x^2, c_1=0,
                  // c_2 = (x^2)''/2 = 1, c_3=0, c_4=(1)''/12=0 => u = x^2 + t^2
                  // (d'Alembert: ((x+t)^2+(x-t)^2)/2 = x^2 + t^2).
                  auto c = nimblecas::wave_equation(Rational::from_int(1), ipoly({0, 0, 1}),
                                                    RationalPoly{}, 4)
                               .value();
                  t.expect_eq(c.size(), std::size_t{5}, "returns c_0..c_4");
                  t.expect(c[0].is_equal(ipoly({0, 0, 1})), "c_0 = x^2");
                  t.expect(c[1].is_zero(), "c_1 = 0 (psi = 0)");
                  t.expect(c[2].is_equal(ipoly({1})), "c_2 = 1");
                  t.expect(c[3].is_zero(), "c_3 = 0");
                  t.expect(c[4].is_zero(), "c_4 = 0 (series terminated)");
                  auto l = nimblecas::heat_operator(Rational::from_int(1));  // L = 1 * d^2/dx^2
                  t.expect(wave_residual_holds(l, c), "(n+2)(n+1) c_{n+2} == c_n''");
                  // u(2,1) = 4 + 1 = 5.
                  auto v = nimblecas::evaluate(c, Rational::from_int(2), Rational::from_int(1));
                  t.expect(v.value() == Rational::from_int(5), "u(2,1) = 5");
              })
        .test("wave_c2_equals_half_c_squared_phi_pp",
              [](TestContext& t) {
                  // Task check: c_2 = (speed^2 / 2) * phi''. speed=2, phi=x^4:
                  // speed^2=4, phi''=12x^2 => c_2 = 4*12x^2/2 = 24 x^2.
                  auto c = nimblecas::wave_equation(Rational::from_int(2), ipoly({0, 0, 0, 0, 1}),
                                                    RationalPoly{}, 2)
                               .value();
                  t.expect(c[0].is_equal(ipoly({0, 0, 0, 0, 1})), "c_0 = x^4");
                  t.expect(c[1].is_zero(), "c_1 = 0");
                  t.expect(c[2].is_equal(ipoly({0, 0, 24})), "c_2 = (c^2/2) phi'' = 24 x^2");
              })
        .test("wave_nonzero_psi_dalembert",
              [](TestContext& t) {
                  // u_tt = u_xx, u(x,0)=0, u_t(x,0)=x. d'Alembert gives u = x t exactly.
                  // c_0=0, c_1=x, c_2=(0)''/2=0, c_3=(x)''/6=0 => u = x t.
                  auto c = nimblecas::wave_equation(Rational::from_int(1), RationalPoly{},
                                                    ipoly({0, 1}), 3)
                               .value();
                  t.expect(c[0].is_zero(), "c_0 = 0");
                  t.expect(c[1].is_equal(ipoly({0, 1})), "c_1 = x (psi)");
                  t.expect(c[2].is_zero(), "c_2 = 0");
                  t.expect(c[3].is_zero(), "c_3 = 0");
                  // u(3,2) = 3*2 = 6.
                  auto v = nimblecas::evaluate(c, Rational::from_int(3), Rational::from_int(2));
                  t.expect(v.value() == Rational::from_int(6), "u(3,2) = x t = 6");
              })
        .test("wave_error_paths",
              [](TestContext& t) {
                  // order 0 cannot carry the second datum psi => domain_error.
                  auto bad = nimblecas::wave_equation(Rational::from_int(1), ipoly({0, 0, 1}),
                                                      RationalPoly{}, 0);
                  t.expect(bad.error() == MathError::domain_error, "order 0 is domain_error");
                  // empty operator => domain_error.
                  auto bad2 = nimblecas::solve_wave_pde(nimblecas::SpatialOperator{}, ipoly({1}),
                                                        RationalPoly{}, 2);
                  t.expect(bad2.error() == MathError::domain_error, "empty operator is domain_error");
              })
        // ---- KdV: nonlinear dispersive evolution (u u_x + u_xxx) ----
        .test("kdv_x3_datum_hand_series",
              [](TestContext& t) {
                  // u_t + u u_x + u_xxx = 0, phi = x^3 (phi''' = 6, exercises dispersion).
                  // Hand-derived (k+1)c_{k+1} = -c_k''' - [t^k](u u_x):
                  //   c_0 = x^3
                  //   c_1 = -c_0''' - x^3*3x^2 = -6 - 3 x^5
                  //   c_2 = ( -c_1''' - [t^1](u u_x) ) / 2 = (180 x^2 + 24 x^7 + 18 x^2)/2
                  //       = 12 x^7 + 99 x^2
                  auto c = nimblecas::kdv(ipoly({0, 0, 0, 1}), 2).value();
                  t.expect_eq(c.size(), std::size_t{3}, "returns c_0..c_2");
                  t.expect(c[0].is_equal(ipoly({0, 0, 0, 1})), "c_0 = x^3");
                  t.expect(c[1].is_equal(ipoly({-6, 0, 0, 0, 0, -3})), "c_1 = -6 - 3 x^5");
                  t.expect(c[2].is_equal(ipoly({0, 0, 99, 0, 0, 0, 0, 12})),
                           "c_2 = 99 x^2 + 12 x^7");
              })
        .test("kdv_linear_datum_reduces_to_burgers",
              [](TestContext& t) {
                  // phi = x has phi''' = 0, so dispersion vanishes and KdV reduces to
                  // inviscid Burgers u_t + u u_x = 0, phi = x => c_n = (-1)^n x.
                  auto c = nimblecas::kdv(ipoly({0, 1}), 3).value();
                  t.expect(c[0].is_equal(ipoly({0, 1})), "c_0 = x");
                  t.expect(c[1].is_equal(ipoly({0, -1})), "c_1 = -x");
                  t.expect(c[2].is_equal(ipoly({0, 1})), "c_2 = x");
                  t.expect(c[3].is_equal(ipoly({0, -1})), "c_3 = -x");
              })
        .test("kdv_error_path",
              [](TestContext& t) {
                  auto bad = nimblecas::kdv(ipoly({0, 0, 0, 1}), 0);
                  t.expect(bad.error() == MathError::domain_error, "order 0 is domain_error");
              })
        // ---- Schrodinger: exact Gaussian-rational time series ----
        .test("schrodinger_free_x2_terminates",
              [](TestContext& t) {
                  // i u_t = -u_xx (V=0), phi = x^2. c_1 = i*(x^2)'' = 2i, c_1''=0 => c_2=0.
                  // Exact terminating series u = x^2 + 2 i t.
                  auto c = nimblecas::schrodinger_free_particle(ipoly({0, 0, 1}), 3).value();
                  t.expect_eq(c.size(), std::size_t{4}, "returns c_0..c_3");
                  t.expect(c[0].is_equal(nimblecas::ComplexPoly::make(ipoly({0, 0, 1}),
                                                                      RationalPoly{})),
                           "c_0 = x^2");
                  t.expect(c[1].is_equal(nimblecas::ComplexPoly::make(RationalPoly{}, ipoly({2}))),
                           "c_1 = 2 i");
                  t.expect(c[2].is_zero(), "c_2 = 0 (series terminated)");
                  t.expect(c[3].is_zero(), "c_3 = 0");
              })
        .test("schrodinger_free_x4_terminates",
              [](TestContext& t) {
                  // phi = x^4. c_1 = i*(x^4)'' = 12 i x^2; c_2 = i*(12 i x^2)''/2 = -12;
                  // c_3 = i*(-12)''/3 = 0. Exact series u = x^4 + 12 i x^2 t - 12 t^2.
                  auto c = nimblecas::schrodinger_free_particle(ipoly({0, 0, 0, 0, 1}), 3).value();
                  t.expect(c[0].is_equal(nimblecas::ComplexPoly::make(ipoly({0, 0, 0, 0, 1}),
                                                                      RationalPoly{})),
                           "c_0 = x^4");
                  t.expect(c[1].is_equal(
                               nimblecas::ComplexPoly::make(RationalPoly{}, ipoly({0, 0, 12}))),
                           "c_1 = 12 i x^2");
                  t.expect(c[2].is_equal(nimblecas::ComplexPoly::make(ipoly({-12}), RationalPoly{})),
                           "c_2 = -12");
                  t.expect(c[3].is_zero(), "c_3 = 0 (series terminated)");
              })
        .test("schrodinger_constant_potential_is_phase",
              [](TestContext& t) {
                  // i u_t = -u_xx + V u, V=1, phi=1 => u_xx=0, so u_t = -i u, u = e^{-i t} =
                  // 1 - i t - t^2/2 + i t^3/6 - ...  Exact Gaussian-rational coefficients.
                  auto c = nimblecas::solve_schrodinger(ipoly({1}), ipoly({1}), 3).value();
                  t.expect(c[0].is_equal(nimblecas::ComplexPoly::make(ipoly({1}), RationalPoly{})),
                           "c_0 = 1");
                  t.expect(c[1].is_equal(nimblecas::ComplexPoly::make(RationalPoly{}, ipoly({-1}))),
                           "c_1 = -i");
                  t.expect(c[2].is_equal(nimblecas::ComplexPoly::make(
                               RationalPoly::from_coeffs({rat(-1, 2)}), RationalPoly{})),
                           "c_2 = -1/2");
                  t.expect(c[3].is_equal(nimblecas::ComplexPoly::make(
                               RationalPoly{}, RationalPoly::from_coeffs({rat(1, 6)}))),
                           "c_3 = i/6");
              })
        .test("schrodinger_error_path",
              [](TestContext& t) {
                  auto bad = nimblecas::schrodinger_free_particle(ipoly({0, 0, 1}), 0);
                  t.expect(bad.error() == MathError::domain_error, "order 0 is domain_error");
              })
        .run();
}
