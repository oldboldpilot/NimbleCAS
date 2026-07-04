// Cross-method tests for a single nonlinear PDE, spanning nimblecas.pde and nimblecas.pdenum:
// take one concrete PDE and solve/verify it by Finite Elements, Finite Differences, the exact
// whole-line power-series (Cauchy-Kovalevskaya) construction, and the ADM/HPM decomposition
// family, then cross-check that independently-implemented methods agree.
// @author Olumuyiwa Oluwasanmi
//
// THE PDE. Reaction-diffusion u_t = D u_xx + u^2 with u(x,0) = phi(x) = x - x^2 (chosen
// because it is QUADRATIC: pdenum's discrete second-difference/FEM stiffness is exactly
// phi'' at every node for a quadratic profile, so the discrete methods and the exact
// symbolic operator can be compared without any discretisation-error slack). D = 1.
//
// FOUR INDEPENDENTLY-BUILT PIECES ARE CROSS-VALIDATED:
//   1. FEM (nimblecas.pdenum::fem_p1_solve) solves the associated steady problem -u'' = 2
//      with the SAME quadratic profile as its manufactured exact solution (P1 FEM is
//      nodally exact here — see pdenum_tests.cpp's own "fem_p1_solve_recovers_exact_galerkin").
//   2. FDM (nimblecas.pdenum::fdm_d2_matrix), the discrete Laplacian, applied to phi's nodal
//      samples on the SAME grid, is checked to give the SAME -2 as (1) at every interior node.
//   3. The exact whole-line time series (nimblecas.pde::solve_evolution_pde with
//      heat_operator) reproduces phi'' = -2 symbolically and, because a quadratic under a
//      constant-coefficient Laplacian differentiates to a constant then to zero, the series
//      TERMINATES: the closed form is u(x,t) = phi(x) - 2t, exactly (no truncation error).
//   4. ADM (nimblecas.pde::reaction_diffusion_quadratic) and HPM
//      (nimblecas.pde::solve_nonlinear_evolution_pde_hpm) solve the FULL NONLINEAR PDE and
//      are checked to return BIT-IDENTICAL truncated series (the ADM == HPM identity, proven
//      once for ODEs in nimblecas.perturbation and once for integral equations in
//      nimblecas.inteq, now empirically exercised for a genuine PDE).
//
// HONEST GAP. nimblecas.pde has no HAM (ħ-parameterised) variant for PDEs yet — see the
// gap note on solve_nonlinear_evolution_pde_hpm's declaration. This suite therefore does not
// claim a PDE-HAM cross-check; only ADM/HPM are exercised for the nonlinear leg.
//
// NOTE ON WHAT IS NOT COMPARED. The whole-line series (piece 3/4, no boundary conditions,
// pde.cppm's own documented scope) and the homogeneous-Dirichlet-truncated operators used by
// FDM/FEM/method-of-lines (pdenum.cppm) solve DIFFERENT boundary conditions in general (the
// whole-line closed form u(x,t) = phi(x) - 2t does not satisfy u(0,t) = u(1,t) = 0 for t > 0).
// This suite therefore only cross-validates the SPATIAL operator (piece 2 vs piece 1/3, all at
// t = 0, where the boundary conditions coincide: phi(0) = phi(1) = 0), never a full boundary-
// value time evolution against the whole-line series — comparing those would be comparing the
// solutions of two different problems.

import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.pde;
import nimblecas.pdenum;
import nimblecas.testing;

using nimblecas::BoundaryCondition;
using nimblecas::heat_operator;
using nimblecas::MathError;
using nimblecas::Matrix;
using nimblecas::Mesh1D;
using nimblecas::Polynomial;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::reaction_diffusion_quadratic;
using nimblecas::solve_evolution_pde;
using nimblecas::solve_nonlinear_evolution_pde_hpm;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto ipoly(std::vector<std::int64_t> c) -> RationalPoly {
    return RationalPoly::from_polynomial(Polynomial{std::move(c)});
}

[[nodiscard]] auto R(std::int64_t n, std::int64_t d = 1) -> Rational {
    return Rational::make(n, d).value();
}

[[nodiscard]] auto rp(std::vector<Rational> c) -> RationalPoly {
    return RationalPoly::from_coeffs(std::move(c));
}

// phi(x) = x - x^2.
[[nodiscard]] auto quadratic_profile() -> RationalPoly {
    return ipoly({0, 1, -1});
}

// N[u] = u^2, the reaction term used by both reaction_diffusion_quadratic and the HPM leg —
// built from the SAME exported series_product primitive reaction_diffusion_quadratic uses
// internally, so the two solvers are driven by an identical nonlinearity, not a reimplemented
// (and possibly subtly different) one.
[[nodiscard]] auto square_op() -> nimblecas::TimeSeriesOperator {
    return [](const std::vector<RationalPoly>& u) -> nimblecas::Result<std::vector<RationalPoly>> {
        return nimblecas::series_product(u, u);
    };
}

}  // namespace

auto main() -> int {
    return TestSuite("pde_crossmethod")
        .test("fdm_matches_fem_on_quadratic_profile",
              [](TestContext& t) {
                  // -u'' = 2, Dirichlet 0/0, uniform mesh N=4: P1 FEM is nodally exact
                  // (pdenum_tests.cpp's own "fem_p1_solve_recovers_exact_galerkin"), giving
                  // the same x - x^2 nodal values reused here as phi's initial datum.
                  auto mesh = Mesh1D::uniform(R(0), R(1), 4);
                  t.expect(mesh.has_value(), "mesh builds");
                  if (!mesh) {
                      return;
                  }
                  auto fem = nimblecas::fem_p1_solve(*mesh, R(1), R(0), RationalPoly::constant(R(2)),
                                                     BoundaryCondition::dirichlet(R(0)),
                                                     BoundaryCondition::dirichlet(R(0)));
                  t.expect(fem.has_value(), "FEM solve succeeds");
                  if (!fem) {
                      return;
                  }
                  t.expect((*fem)[1] == R(3, 16) && (*fem)[2] == R(1, 4) && (*fem)[3] == R(3, 16),
                           "FEM nodal values == x - x^2 at x = 1/4, 1/2, 3/4");

                  // The SAME three interior nodal values, run through the discrete Laplacian
                  // (FDM) on the same grid (h = 1/4): every interior row must give exactly -2,
                  // matching phi''(x) = -2 pointwise (no discretisation error for a quadratic).
                  auto d2 = nimblecas::fdm_d2_matrix(3, R(1, 4));
                  t.expect(d2.has_value(), "fdm_d2_matrix builds");
                  if (!d2) {
                      return;
                  }
                  auto col = Matrix::from_rows({{R(3, 16)}, {R(1, 4)}, {R(3, 16)}});
                  t.expect(col.has_value(), "nodal column vector builds");
                  if (!col) {
                      return;
                  }
                  auto laplacian = d2->multiply(*col);
                  t.expect(laplacian.has_value(), "D2 * phi_nodal succeeds");
                  if (!laplacian) {
                      return;
                  }
                  t.expect(laplacian->at(0, 0) == R(-2) && laplacian->at(1, 0) == R(-2) &&
                               laplacian->at(2, 0) == R(-2),
                           "FDM discrete Laplacian == -2 at every interior node, matching FEM's f");
              })
        .test("exact_whole_line_series_terminates",
              [](TestContext& t) {
                  // u_t = u_xx (D = 1), u(x,0) = x - x^2. c_0 = phi, c_1 = phi'' = -2
                  // (constant), c_2 = (-2)''/2 = 0: the series TERMINATES, so
                  // u(x,t) = (x - x^2) - 2t exactly, a genuine closed form (no truncation).
                  auto phi = quadratic_profile();
                  auto c = solve_evolution_pde(heat_operator(R(1)), phi, 3);
                  t.expect(c.has_value(), "solve_evolution_pde succeeds");
                  if (!c) {
                      return;
                  }
                  t.expect(c->size() == 4, "4 coefficients (order 3)");
                  t.expect((*c)[0].is_equal(phi), "c_0 == phi");
                  t.expect((*c)[1].is_equal(RationalPoly::constant(R(-2))), "c_1 == -2");
                  t.expect((*c)[2].is_zero() && (*c)[3].is_zero(), "series terminates: c_2 = c_3 = 0");

                  // u(1/2, 1/10) = (1/2 - 1/4) - 2*(1/10) = 1/4 - 1/5 = 1/20.
                  auto v = nimblecas::evaluate(*c, R(1, 2), R(1, 10));
                  t.expect(v.has_value() && *v == R(1, 20),
                           "closed-form evaluation u(1/2, 1/10) == 1/20");
              })
        .test("nonlinear_adm_hpm_agree",
              [](TestContext& t) {
                  // The FULL nonlinear PDE u_t = u_xx + u^2, u(x,0) = x - x^2. ADM (the
                  // library's reaction_diffusion_quadratic convenience) and HPM
                  // (solve_nonlinear_evolution_pde_hpm, driven by the identical L and N) must
                  // return BIT-IDENTICAL truncated series (ADM == HPM, proven by the homotopy
                  // argument in solve_nonlinear_evolution_pde_hpm's doc comment).
                  auto phi = quadratic_profile();
                  auto adm = reaction_diffusion_quadratic(R(1), phi, 3);
                  t.expect(adm.has_value(), "ADM (reaction_diffusion_quadratic) succeeds");
                  auto hpm = solve_nonlinear_evolution_pde_hpm(heat_operator(R(1)), square_op(), phi, 3);
                  t.expect(hpm.has_value(), "HPM succeeds");
                  if (!adm || !hpm) {
                      return;
                  }
                  t.expect(adm->size() == hpm->size(), "same number of coefficients");
                  bool all_equal = true;
                  for (std::size_t i = 0; i < adm->size() && i < hpm->size(); ++i) {
                      all_equal = all_equal && (*adm)[i].is_equal((*hpm)[i]);
                  }
                  t.expect(all_equal, "ADM and HPM return bit-identical series");

                  // Hand-verified leading terms: c_0 = x - x^2; A_0 = c_0^2 = x^2 - 2x^3 + x^4;
                  // c_1 = (L[c_0] + A_0)/1 = -2 + (x^2 - 2x^3 + x^4).
                  t.expect((*adm)[0].is_equal(phi), "c_0 == phi");
                  t.expect((*adm)[1].is_equal(ipoly({-2, 0, 1, -2, 1})),
                           "c_1 == -2 + x^2 - 2x^3 + x^4 (hand-verified)");
              })
        .test("error_paths",
              [](TestContext& t) {
                  auto phi = quadratic_profile();
                  auto zero_order = solve_nonlinear_evolution_pde_hpm(heat_operator(R(1)), square_op(),
                                                                      phi, 0);
                  t.expect(!zero_order.has_value() && zero_order.error() == MathError::domain_error,
                           "order == 0 is domain_error");
                  nimblecas::TimeSeriesOperator empty_n;
                  auto no_n = solve_nonlinear_evolution_pde_hpm(heat_operator(R(1)), empty_n, phi, 2);
                  t.expect(!no_n.has_value(), "empty nonlinear operator fails");
              })
        .run();
}
