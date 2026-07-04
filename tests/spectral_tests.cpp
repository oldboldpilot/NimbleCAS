// Tests for nimblecas.spectral: spectral-Galerkin (exact over Q), Chebyshev
// collocation & Fourier spectral (numerical), and the spectral-element / DG analogues.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.spectral;
import nimblecas.testing;

using nimblecas::MathError;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

// Build a RationalPoly from rational coefficients (low degree first).
[[nodiscard]] auto rpoly(std::vector<Rational> c) -> RationalPoly {
    return RationalPoly::from_coeffs(std::move(c));
}

[[nodiscard]] auto approx(double a, double b, double tol) -> bool {
    return std::abs(a - b) <= tol;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.spectral")
        // --- EXACT spectral-Galerkin over Q -------------------------------
        .test("legendre_coefficients_exact",
              [](TestContext& t) {
                  // x^2 = (1/3) P_0 + (2/3) P_2.
                  const auto c = nimblecas::legendre_coefficients(rpoly({rat(0, 1), rat(0, 1),
                                                                         rat(1, 1)}))
                                     .value();
                  const std::vector<Rational> expect = {rat(1, 3), rat(0, 1), rat(2, 3)};
                  t.expect(c == expect, "legendre coeffs of x^2 = [1/3, 0, 2/3]");
              })
        .test("legendre_forward_inverse_roundtrip",
              [](TestContext& t) {
                  // f = 3 - x + 2 x^2; inverse(forward(f)) == f exactly.
                  const RationalPoly f = rpoly({rat(3, 1), rat(-1, 1), rat(2, 1)});
                  const auto c = nimblecas::legendre_coefficients(f).value();
                  const auto g = nimblecas::legendre_from_coefficients(c).value();
                  t.expect(g.is_equal(f), "legendre inverse(forward(f)) == f");
              })
        .test("chebyshev_coefficients_exact",
              [](TestContext& t) {
                  // x^3 = (3/4) T_1 + (1/4) T_3.
                  const auto c = nimblecas::chebyshev_coefficients(rpoly({rat(0, 1), rat(0, 1),
                                                                          rat(0, 1), rat(1, 1)}))
                                     .value();
                  const std::vector<Rational> expect = {rat(0, 1), rat(3, 4), rat(0, 1),
                                                        rat(1, 4)};
                  t.expect(c == expect, "chebyshev coeffs of x^3 = [0, 3/4, 0, 1/4]");
              })
        .test("chebyshev_forward_inverse_roundtrip",
              [](TestContext& t) {
                  const RationalPoly f = rpoly({rat(1, 1), rat(0, 1), rat(-2, 1), rat(1, 1)});
                  const auto c = nimblecas::chebyshev_coefficients(f).value();
                  const auto g = nimblecas::chebyshev_from_coefficients(c).value();
                  t.expect(g.is_equal(f), "chebyshev inverse(forward(f)) == f");
              })
        .test("spectral_differentiation_coefficient_space",
              [](TestContext& t) {
                  // d/dx (x^3) = 3 x^2, via Legendre coefficient-space differentiation.
                  const RationalPoly f = rpoly({rat(0, 1), rat(0, 1), rat(0, 1), rat(1, 1)});
                  const RationalPoly dfx = rpoly({rat(0, 1), rat(0, 1), rat(3, 1)});
                  const auto lc = nimblecas::legendre_coefficients(f).value();
                  const auto ldc = nimblecas::legendre_differentiate_coefficients(lc).value();
                  const auto lg = nimblecas::legendre_from_coefficients(ldc).value();
                  t.expect(lg.is_equal(dfx), "legendre coeff-space d/dx (x^3) = 3x^2");

                  // Same via Chebyshev coefficient-space differentiation.
                  const auto cc = nimblecas::chebyshev_coefficients(f).value();
                  const auto cdc = nimblecas::chebyshev_differentiate_coefficients(cc).value();
                  const auto cg = nimblecas::chebyshev_from_coefficients(cdc).value();
                  t.expect(cg.is_equal(dfx), "chebyshev coeff-space d/dx (x^3) = 3x^2");
              })
        .test("galerkin_poisson_exact",
              [](TestContext& t) {
                  // -u'' = 2, u(±1)=0  ->  u = 1 - x^2.
                  const auto u = nimblecas::galerkin_poisson(
                                     RationalPoly::constant(Rational::from_int(2)))
                                     .value();
                  t.expect(u.is_equal(rpoly({rat(1, 1), rat(0, 1), rat(-1, 1)})),
                           "galerkin -u''=2 -> 1 - x^2");

                  // -u'' = x^2, u(±1)=0  ->  u = (1 - x^4)/12.
                  const auto u2 = nimblecas::galerkin_poisson(
                                      rpoly({rat(0, 1), rat(0, 1), rat(1, 1)}))
                                      .value();
                  t.expect(u2.is_equal(rpoly({rat(1, 12), rat(0, 1), rat(0, 1), rat(0, 1),
                                              rat(-1, 12)})),
                           "galerkin -u''=x^2 -> (1 - x^4)/12");
              })
        // --- NUMERICAL Chebyshev collocation ------------------------------
        .test("chebyshev_collocation_differentiation",
              [](TestContext& t) {
                  // D applied to samples of x^3 reproduces 3 x^2 at the nodes (to tolerance).
                  const std::size_t n = 8;
                  const auto nodes = nimblecas::chebyshev_gauss_lobatto_nodes(n).value();
                  const auto d = nimblecas::chebyshev_differentiation_matrix(n).value();
                  std::vector<double> v(n + 1);
                  for (std::size_t j = 0; j <= n; ++j) {
                      v[j] = nodes[j] * nodes[j] * nodes[j];
                  }
                  const auto dv = nimblecas::apply_dense_matrix(d, v).value();
                  bool ok = true;
                  for (std::size_t j = 0; j <= n; ++j) {
                      ok = ok && approx(dv[j], 3.0 * nodes[j] * nodes[j], 1e-8);
                  }
                  t.expect(ok, "cheb collocation D reproduces d/dx(x^3) = 3x^2");
              })
        .test("chebyshev_collocation_poisson",
              [](TestContext& t) {
                  // -u'' = 2, u(±1)=0 -> u = 1 - x^2, at the collocation nodes numerically.
                  const std::size_t n = 8;
                  const auto nodes = nimblecas::chebyshev_gauss_lobatto_nodes(n).value();
                  const auto u = nimblecas::chebyshev_collocation_poisson(
                                     [](double) { return 2.0; }, n)
                                     .value();
                  bool ok = true;
                  for (std::size_t j = 0; j <= n; ++j) {
                      ok = ok && approx(u[j], 1.0 - nodes[j] * nodes[j], 1e-8);
                  }
                  t.expect(ok, "cheb collocation -u''=2 -> 1 - x^2 at nodes");
              })
        // --- NUMERICAL Fourier spectral -----------------------------------
        .test("fourier_spectral_derivative_sin",
              [](TestContext& t) {
                  // d/dx sin(x) = cos(x) on the periodic grid.
                  const std::size_t n = 16;
                  const auto grid = nimblecas::fourier_grid(n).value();
                  std::vector<double> u(n);
                  for (std::size_t j = 0; j < n; ++j) {
                      u[j] = std::sin(grid[j]);
                  }
                  const auto du = nimblecas::fourier_spectral_derivative(u).value();
                  bool ok = true;
                  for (std::size_t j = 0; j < n; ++j) {
                      ok = ok && approx(du[j], std::cos(grid[j]), 1e-9);
                  }
                  t.expect(ok, "fourier d/dx sin(x) = cos(x)");
              })
        .test("fourier_differentiation_matrix_sin",
              [](TestContext& t) {
                  const std::size_t n = 16;
                  const auto grid = nimblecas::fourier_grid(n).value();
                  const auto d = nimblecas::fourier_differentiation_matrix(n).value();
                  std::vector<double> u(n);
                  for (std::size_t j = 0; j < n; ++j) {
                      u[j] = std::sin(grid[j]);
                  }
                  const auto du = nimblecas::apply_dense_matrix(d, u).value();
                  bool ok = true;
                  for (std::size_t j = 0; j < n; ++j) {
                      ok = ok && approx(du[j], std::cos(grid[j]), 1e-8);
                  }
                  t.expect(ok, "fourier diff matrix: sin(x) -> cos(x)");
              })
        .test("fourier_periodic_solve",
              [](TestContext& t) {
                  // u - u'' = 2 sin(x)  ->  u = sin(x) (since sin - (-sin) = 2 sin).
                  const std::size_t n = 16;
                  const auto grid = nimblecas::fourier_grid(n).value();
                  std::vector<double> f(n);
                  for (std::size_t j = 0; j < n; ++j) {
                      f[j] = 2.0 * std::sin(grid[j]);
                  }
                  const auto u = nimblecas::fourier_periodic_solve(f).value();
                  bool ok = true;
                  for (std::size_t j = 0; j < n; ++j) {
                      ok = ok && approx(u[j], std::sin(grid[j]), 1e-9);
                  }
                  t.expect(ok, "fourier periodic (I - d^2)u = 2 sin -> sin");
              })
        // --- ANALOGUES: spectral element (C^0) ----------------------------
        .test("spectral_element_consistency",
              [](TestContext& t) {
                  // A global polynomial projected onto a 2-element rational mesh and
                  // reconstructed reproduces itself exactly on each element (exact, C^0).
                  const RationalPoly f = rpoly({rat(0, 1), rat(-2, 1), rat(0, 1), rat(1, 1)});
                  const std::vector<Rational> mesh = {rat(-1, 1), rat(0, 1), rat(1, 1)};
                  const auto coeffs = nimblecas::spectral_element_legendre(f, mesh).value();
                  t.expect(coeffs.size() == 2, "two elements");
                  const auto recon = nimblecas::spectral_element_reconstruct(coeffs, mesh).value();
                  t.expect(recon.size() == 2, "two reconstructed element polynomials");
                  t.expect(recon[0].is_equal(f), "element 0 reconstructs f exactly");
                  t.expect(recon[1].is_equal(f), "element 1 reconstructs f exactly");
                  // C^0 at the shared interior node x = 0.
                  const auto v0 = nimblecas::evaluate_poly(recon[0], rat(0, 1)).value();
                  const auto v1 = nimblecas::evaluate_poly(recon[1], rat(0, 1)).value();
                  t.expect(v0 == v1, "C^0 continuity at interior node");
              })
        // --- ANALOGUES: discontinuous Galerkin ----------------------------
        .test("dg_advection_constant_field",
              [](TestContext& t) {
                  // A constant field is stationary under advection: du/dt = 0 exactly.
                  const std::vector<std::vector<Rational>> state = {{rat(5, 1)}, {rat(5, 1)}};
                  const std::vector<Rational> mesh = {rat(0, 1), rat(1, 1), rat(2, 1)};
                  const auto rhs = nimblecas::dg_advection_rhs(state, mesh, Rational::from_int(1),
                                                              Rational::from_int(5))
                                       .value();
                  t.expect(rhs[0][0] == rat(0, 1) && rhs[1][0] == rat(0, 1),
                           "constant field: du/dt = 0 exactly");
              })
        .test("dg_advection_linear_field_exact",
              [](TestContext& t) {
                  // u(x)=x, a=1, inflow=0: du/dt = -a u_x = -1 (constant) on every element.
                  // Reference Legendre coeffs of x on [0,1]: [1/2,1/2]; on [1,2]: [3/2,1/2].
                  const std::vector<std::vector<Rational>> state = {{rat(1, 2), rat(1, 2)},
                                                                    {rat(3, 2), rat(1, 2)}};
                  const std::vector<Rational> mesh = {rat(0, 1), rat(1, 1), rat(2, 1)};
                  const auto rhs = nimblecas::dg_advection_rhs(state, mesh, Rational::from_int(1),
                                                              Rational::from_int(0))
                                       .value();
                  const bool ok = rhs[0][0] == rat(-1, 1) && rhs[0][1] == rat(0, 1) &&
                                  rhs[1][0] == rat(-1, 1) && rhs[1][1] == rat(0, 1);
                  t.expect(ok, "linear field: DG operator = exact -a u_x = -1 per element");
              })
        .test("dg_advection_step_runs",
              [](TestContext& t) {
                  // One exact rational Euler step of a constant field leaves it unchanged.
                  const std::vector<std::vector<Rational>> state = {{rat(5, 1)}, {rat(5, 1)}};
                  const std::vector<Rational> mesh = {rat(0, 1), rat(1, 1), rat(2, 1)};
                  const auto next = nimblecas::dg_advection_step(state, mesh,
                                                                Rational::from_int(1),
                                                                Rational::from_int(5),
                                                                rat(1, 10))
                                        .value();
                  t.expect(next[0][0] == rat(5, 1) && next[1][0] == rat(5, 1),
                           "constant field unchanged after one exact Euler step");
              })
        // --- domain errors ------------------------------------------------
        .test("domain_errors",
              [](TestContext& t) {
                  t.expect(nimblecas::chebyshev_gauss_lobatto_nodes(0).error() ==
                               MathError::domain_error,
                           "cheb nodes n=0 is domain_error");
                  t.expect(nimblecas::chebyshev_collocation_poisson([](double) { return 0.0; }, 1)
                                   .error() == MathError::domain_error,
                           "cheb collocation n<2 is domain_error");
                  t.expect(nimblecas::fourier_differentiation_matrix(3).error() ==
                               MathError::domain_error,
                           "fourier diff matrix odd n is domain_error");
                  {
                      std::array<double, 1> one{{1.0}};
                      t.expect(nimblecas::fourier_spectral_derivative(one).error() ==
                                   MathError::domain_error,
                               "fourier derivative n<2 is domain_error");
                  }
                  {
                      const std::vector<Rational> single = {rat(0, 1)};
                      t.expect(nimblecas::spectral_element_legendre(
                                   RationalPoly::constant(Rational::from_int(1)), single)
                                       .error() == MathError::domain_error,
                               "spectral element needs >= 2 mesh nodes");
                  }
                  {
                      const std::vector<std::vector<Rational>> empty_state;
                      const std::vector<Rational> mesh = {rat(0, 1)};
                      t.expect(nimblecas::dg_advection_rhs(empty_state, mesh,
                                                           Rational::from_int(1),
                                                           Rational::from_int(0))
                                       .error() == MathError::domain_error,
                               "dg with no elements is domain_error");
                  }
              })
        .run();
}
