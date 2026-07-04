// Tests for nimblecas.pdenum: numerical methods for PDEs (FDM / FEM / FVM / method of lines).
// @author Olumuyiwa Oluwasanmi
//
// The SPATIAL discretizations and the linear solves are EXACT over Q on rational grids with
// rational/polynomial data, so almost every assertion checks a concrete exact rational value or
// an exact matrix entry (nothing "approximate"). The single deliberately-numerical case is the
// Crank-Nicolson time step, which is checked only for the qualitative property (a diffusion mode
// decays). Domain-error paths are exercised too.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.pdenum;
import nimblecas.testing;

using nimblecas::BoundaryCondition;
using nimblecas::DiffScheme;
using nimblecas::Grid1D;
using nimblecas::Grid2D;
using nimblecas::MathError;
using nimblecas::Matrix;
using nimblecas::Mesh1D;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto ri(std::int64_t v) -> Rational {
    return Rational::from_int(v);
}

[[nodiscard]] auto rq(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

[[nodiscard]] auto vec(std::vector<std::int64_t> entries) -> std::vector<Rational> {
    std::vector<Rational> v;
    v.reserve(entries.size());
    for (const std::int64_t e : entries) {
        v.push_back(Rational::from_int(e));
    }
    return v;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.pdenum")
        // ---- Finite difference ----
        .test("fdm_d2_matrix_is_tridiag_over_h2",
              [](TestContext& t) {
                  // Central second-derivative operator on h = 1/4: tridiag(1,-2,1)/h^2, h^2 = 1/16
                  // => diag = -32, off = 16.
                  auto d2 = nimblecas::fdm_d2_matrix(3, rq(1, 4));
                  t.expect(d2.has_value(), "fdm_d2_matrix builds");
                  if (d2) {
                      t.expect(d2->rows() == 3 && d2->cols() == 3, "3x3 operator");
                      t.expect(d2->at(0, 0) == ri(-32) && d2->at(1, 1) == ri(-32) &&
                                   d2->at(2, 2) == ri(-32),
                               "diagonal = -2/h^2 = -32");
                      t.expect(d2->at(0, 1) == ri(16) && d2->at(1, 0) == ri(16) &&
                                   d2->at(1, 2) == ri(16) && d2->at(2, 1) == ri(16),
                               "off-diagonal = 1/h^2 = 16");
                      t.expect(d2->at(0, 2) == ri(0) && d2->at(2, 0) == ri(0), "no far coupling");
                  }
              })
        .test("fdm_poisson_quadratic_exact_nodes",
              [](TestContext& t) {
                  // -u'' = 2 on [0,1], u(0)=u(1)=0, exact u = x - x^2. The 3-point stencil
                  // reproduces the exact nodal values on the coarse grid N=4 (hand-verified:
                  // u1=u3=3/16, u2=1/4) because the second difference of a quadratic is exact.
                  auto grid = Grid1D::make(ri(0), ri(1), 4);
                  t.expect(grid.has_value(), "grid builds");
                  if (grid) {
                      const auto f = vec({2, 2, 2, 2, 2});  // f sampled at all 5 nodes
                      auto u = nimblecas::solve_poisson_1d(*grid, f,
                                                           BoundaryCondition::dirichlet(ri(0)),
                                                           BoundaryCondition::dirichlet(ri(0)));
                      t.expect(u.has_value(), "poisson solve succeeds");
                      if (u) {
                          t.expect(u->size() == 5, "full nodal solution has 5 entries");
                          t.expect((*u)[0] == ri(0) && (*u)[4] == ri(0), "boundary nodes are 0");
                          t.expect((*u)[1] == rq(3, 16) && (*u)[2] == rq(1, 4) &&
                                       (*u)[3] == rq(3, 16),
                                   "interior = [3/16, 1/4, 3/16] = exact x - x^2 at nodes");
                      }
                  }
              })
        .test("fdm_poisson_mixed_neumann_linear_exact",
              [](TestContext& t) {
                  // -u'' = 0, u(0)=0, u'(1)=1 => exact u = x. The 2nd-order ghost-node Neumann row
                  // is exact for a linear solution: on N=2 the nodal values are [0, 1/2, 1].
                  auto grid = Grid1D::make(ri(0), ri(1), 2);
                  t.expect(grid.has_value(), "grid builds");
                  if (grid) {
                      const auto f = vec({0, 0, 0});
                      auto u = nimblecas::solve_poisson_1d(*grid, f,
                                                           BoundaryCondition::dirichlet(ri(0)),
                                                           BoundaryCondition::neumann(ri(1)));
                      t.expect(u.has_value(), "mixed BVP solve succeeds");
                      if (u) {
                          t.expect(u->size() == 3 && (*u)[0] == ri(0) && (*u)[1] == rq(1, 2) &&
                                       (*u)[2] == ri(1),
                                   "u = [0, 1/2, 1] = exact x at nodes");
                      }
                  }
              })
        .test("fdm_poisson_2d_linear_exact",
              [](TestContext& t) {
                  // -Δu = 0 with boundary u = x + y (harmonic, linear). The 5-point Laplacian is
                  // exact for a linear function, so interior nodal values equal x_i + y_j.
                  auto grid = Grid2D::make(ri(0), ri(1), ri(0), ri(1), 3, 3);
                  t.expect(grid.has_value(), "2-D grid builds");
                  if (grid) {
                      auto f = [](const Rational&, const Rational&) -> nimblecas::Result<Rational> {
                          return ri(0);
                      };
                      auto bc = [](const Rational& x,
                                   const Rational& y) -> nimblecas::Result<Rational> {
                          return x.add(y);
                      };
                      auto u = nimblecas::solve_poisson_2d(*grid, f, bc);
                      t.expect(u.has_value(), "2-D Poisson solve succeeds");
                      if (u) {
                          t.expect(u->rows() == 2 && u->cols() == 2, "interior is 2x2");
                          // nodes x,y in {1/3, 2/3}; entry (j,i) = x_{i+1} + y_{j+1}.
                          t.expect(u->at(0, 0) == rq(2, 3), "u(1/3,1/3) = 2/3");
                          t.expect(u->at(0, 1) == ri(1), "u(2/3,1/3) = 1");
                          t.expect(u->at(1, 0) == ri(1), "u(1/3,2/3) = 1");
                          t.expect(u->at(1, 1) == rq(4, 3), "u(2/3,2/3) = 4/3");
                      }
                  }
              })
        // ---- Finite element ----
        .test("fem_p1_stiffness_is_tridiag_over_h",
              [](TestContext& t) {
                  // Uniform mesh [0,1], N=4, h=1/4 => 1/h = 4. Global stiffness: interior diagonal
                  // 2/h = 8, off-diagonal -1/h = -4, corner diagonal 1/h = 4.
                  auto mesh = Mesh1D::uniform(ri(0), ri(1), 4);
                  t.expect(mesh.has_value(), "uniform mesh builds");
                  if (mesh) {
                      auto k = nimblecas::fem_p1_stiffness(*mesh);
                      t.expect(k.has_value(), "stiffness assembles");
                      if (k) {
                          t.expect(k->rows() == 5 && k->cols() == 5, "5x5 stiffness");
                          t.expect(k->at(0, 0) == ri(4) && k->at(4, 4) == ri(4), "corners = 1/h = 4");
                          t.expect(k->at(1, 1) == ri(8) && k->at(2, 2) == ri(8) &&
                                       k->at(3, 3) == ri(8),
                                   "interior diagonal = 2/h = 8");
                          t.expect(k->at(0, 1) == ri(-4) && k->at(1, 0) == ri(-4) &&
                                       k->at(2, 3) == ri(-4),
                                   "off-diagonal = -1/h = -4");
                          t.expect(k->at(0, 2) == ri(0), "no far coupling");
                      }
                  }
              })
        .test("fem_p1_mass_matrix_values",
              [](TestContext& t) {
                  // Uniform mesh [0,1], N=2, h=1/2. Element mass (h/6)[[2,1],[1,2]] => diagonal
                  // contribution h/3 = 1/6, off h/6 = 1/12; the shared node sums to 1/3.
                  auto mesh = Mesh1D::uniform(ri(0), ri(1), 2);
                  t.expect(mesh.has_value(), "mesh builds");
                  if (mesh) {
                      auto m = nimblecas::fem_p1_mass(*mesh);
                      t.expect(m.has_value(), "mass assembles");
                      if (m) {
                          t.expect(m->at(0, 0) == rq(1, 6) && m->at(2, 2) == rq(1, 6),
                                   "end diagonal = h/3 = 1/6");
                          t.expect(m->at(1, 1) == rq(1, 3), "shared diagonal = 2*(h/3) = 1/3");
                          t.expect(m->at(0, 1) == rq(1, 12) && m->at(1, 0) == rq(1, 12),
                                   "off-diagonal = h/6 = 1/12");
                      }
                  }
              })
        .test("fem_p1_solve_recovers_exact_galerkin",
              [](TestContext& t) {
                  // -u'' = 2 (a=1, c=0), Dirichlet 0/0, uniform N=4. P1 FEM is nodally exact for
                  // this problem, so the nodal values equal x - x^2 (identical to the FDM result).
                  auto mesh = Mesh1D::uniform(ri(0), ri(1), 4);
                  t.expect(mesh.has_value(), "mesh builds");
                  if (mesh) {
                      const RationalPoly f = RationalPoly::constant(ri(2));  // f(x) = 2
                      auto u = nimblecas::fem_p1_solve(*mesh, ri(1), ri(0), f,
                                                       BoundaryCondition::dirichlet(ri(0)),
                                                       BoundaryCondition::dirichlet(ri(0)));
                      t.expect(u.has_value(), "FEM solve succeeds");
                      if (u) {
                          t.expect(u->size() == 5, "5 nodal values");
                          t.expect((*u)[0] == ri(0) && (*u)[4] == ri(0), "Dirichlet nodes are 0");
                          t.expect((*u)[1] == rq(3, 16) && (*u)[2] == rq(1, 4) &&
                                       (*u)[3] == rq(3, 16),
                                   "Galerkin nodal values = exact x - x^2");
                      }
                  }
              })
        .test("fem_p1_load_linear_source_exact",
              [](TestContext& t) {
                  // f(x) = x on [0,1], N=2: exact element integrals b_i = ∫ x φ_i. Interior node
                  // (x=1/2): ∫_0^1 x φ_1 = 1/4. End nodes: b_0 = 1/24, b_2 = 5/24 (hand-checked).
                  auto mesh = Mesh1D::uniform(ri(0), ri(1), 2);
                  t.expect(mesh.has_value(), "mesh builds");
                  if (mesh) {
                      const RationalPoly f = RationalPoly::monomial(ri(1), 1);  // f(x) = x
                      auto b = nimblecas::fem_p1_load(*mesh, f);
                      t.expect(b.has_value(), "load assembles exactly");
                      if (b) {
                          t.expect(b->size() == 3, "3 load entries");
                          t.expect((*b)[0] == rq(1, 24), "b_0 = 1/24");
                          t.expect((*b)[1] == rq(1, 4), "b_1 = 1/4");
                          t.expect((*b)[2] == rq(5, 24), "b_2 = 5/24");
                      }
                  }
              })
        .test("fem_p2_element_matrices",
              [](TestContext& t) {
                  // P2 element matrices for h = 1: stiffness (1/3)[[7,-8,1],..], mass
                  // (1/30)[[4,2,-1],..].
                  auto ks = nimblecas::fem_p2_element_stiffness(ri(1));
                  auto ms = nimblecas::fem_p2_element_mass(ri(1));
                  t.expect(ks.has_value() && ms.has_value(), "P2 element matrices build");
                  if (ks) {
                      t.expect(ks->at(0, 0) == rq(7, 3) && ks->at(1, 1) == rq(16, 3) &&
                                   ks->at(0, 1) == rq(-8, 3) && ks->at(0, 2) == rq(1, 3),
                               "P2 stiffness = (1/3)[[7,-8,1],[-8,16,-8],[1,-8,7]]");
                  }
                  if (ms) {
                      t.expect(ms->at(0, 0) == rq(2, 15) && ms->at(1, 1) == rq(8, 15) &&
                                   ms->at(0, 2) == rq(-1, 30),
                               "P2 mass = (1/30)[[4,2,-1],[2,16,2],[-1,2,4]]");
                  }
              })
        // ---- Finite volume ----
        .test("fvm_diffusion_linear_exact_cells",
              [](TestContext& t) {
                  // -(k u')' = 0 with k=1, u(0)=0, u(1)=1 => exact u = x. Cell-centered FVM with
                  // N=4 gives the exact cell-center values x_i = (i + 1/2)/4 = 1/8,3/8,5/8,7/8.
                  auto grid = Grid1D::make(ri(0), ri(1), 4);
                  t.expect(grid.has_value(), "grid builds");
                  if (grid) {
                      const auto f = vec({0, 0, 0, 0});  // zero source per cell
                      auto u = nimblecas::fvm_solve_diffusion_1d(*grid, ri(1), f, ri(0), ri(1));
                      t.expect(u.has_value(), "FVM solve succeeds");
                      if (u) {
                          t.expect(u->size() == 4, "4 cell values");
                          t.expect((*u)[0] == rq(1, 8) && (*u)[1] == rq(3, 8) &&
                                       (*u)[2] == rq(5, 8) && (*u)[3] == rq(7, 8),
                                   "cell centers = exact x = [1/8, 3/8, 5/8, 7/8]");
                      }
                  }
              })
        // ---- Method of lines ----
        .test("mol_heat_operator_is_tridiag_over_h2",
              [](TestContext& t) {
                  // L = tridiag(1,-2,1)/h^2 with alpha=1, h=1/2 => h^2=1/4 => diag=-8, off=4.
                  auto l = nimblecas::mol_heat_operator(3, rq(1, 2), ri(1));
                  t.expect(l.has_value(), "heat operator builds");
                  if (l) {
                      t.expect(l->rows() == 3 && l->cols() == 3, "3x3 operator");
                      t.expect(l->at(0, 0) == ri(-8) && l->at(1, 1) == ri(-8) &&
                                   l->at(2, 2) == ri(-8),
                               "diagonal = -2/h^2 = -8");
                      t.expect(l->at(0, 1) == ri(4) && l->at(1, 0) == ri(4) &&
                                   l->at(1, 2) == ri(4) && l->at(2, 1) == ri(4),
                               "off-diagonal = 1/h^2 = 4");
                  }
              })
        .test("crank_nicolson_decays_a_mode",
              [](TestContext& t) {
                  // NUMERICAL: on the exact rational heat operator (converted to double), a single
                  // Fourier mode of the diffusion equation must decay under Crank-Nicolson. Grid
                  // [0,1], N=8, interior n=7, h=1/8; initial u_i = sin(pi x_i).
                  const std::size_t n = 7;
                  auto l = nimblecas::mol_heat_operator(n, rq(1, 8), ri(1));
                  t.expect(l.has_value(), "heat operator builds");
                  if (l) {
                      std::vector<double> u(n);
                      for (std::size_t i = 0; i < n; ++i) {
                          const double x = static_cast<double>(i + 1) / 8.0;
                          u[i] = std::sin(std::numbers::pi * x);
                      }
                      const double initial_peak =
                          *std::ranges::max_element(u, {}, [](double v) { return std::abs(v); });
                      bool ok = true;
                      double prev_peak = std::abs(initial_peak);
                      for (int step = 0; step < 5; ++step) {
                          auto next = nimblecas::crank_nicolson_step(*l, u, 0.01);
                          if (!next) {
                              ok = false;
                              break;
                          }
                          u = *next;
                          double peak = 0.0;
                          for (double v : u) {
                              peak = std::max(peak, std::abs(v));
                          }
                          ok = ok && peak < prev_peak && peak > 0.0;  // strictly decaying, nonzero
                          prev_peak = peak;
                      }
                      t.expect(ok, "the mode decays monotonically and stays positive (stable)");
                      t.expect(prev_peak < std::abs(initial_peak),
                               "final amplitude is below the initial amplitude");
                  }
              })
        // ---- domain errors ----
        .test("domain_errors",
              [](TestContext& t) {
                  // Bad grid: zero intervals, and b <= a.
                  t.expect(!Grid1D::make(ri(0), ri(1), 0).has_value(), "0 intervals => error");
                  auto bad_grid = Grid1D::make(ri(1), ri(0), 4);
                  t.expect(!bad_grid.has_value() && bad_grid.error() == MathError::domain_error,
                           "b <= a => domain_error");

                  auto grid = Grid1D::make(ri(0), ri(1), 4);
                  t.expect(grid.has_value(), "reference grid builds");
                  if (grid) {
                      // Wrong f_nodal length (needs num_nodes = 5, given 3).
                      auto bad_f = nimblecas::solve_poisson_1d(
                          *grid, vec({1, 1, 1}), BoundaryCondition::dirichlet(ri(0)),
                          BoundaryCondition::dirichlet(ri(0)));
                      t.expect(!bad_f.has_value() && bad_f.error() == MathError::domain_error,
                               "mismatched f_nodal length => domain_error");

                      // Pure Neumann/Neumann is singular (solution defined only up to a constant).
                      auto singular = nimblecas::solve_poisson_1d(
                          *grid, vec({0, 0, 0, 0, 0}), BoundaryCondition::neumann(ri(0)),
                          BoundaryCondition::neumann(ri(0)));
                      t.expect(!singular.has_value() &&
                                   singular.error() == MathError::domain_error,
                               "pure Neumann/Neumann => singular domain_error");

                      // FVM: zero diffusivity and wrong cell-source length.
                      auto zero_k =
                          nimblecas::fvm_solve_diffusion_1d(*grid, ri(0), vec({0, 0, 0, 0}), ri(0),
                                                            ri(1));
                      t.expect(!zero_k.has_value() && zero_k.error() == MathError::domain_error,
                               "k == 0 => domain_error");
                      auto bad_cells = nimblecas::fvm_solve_diffusion_1d(
                          *grid, ri(1), vec({0, 0, 0}), ri(0), ri(1));
                      t.expect(!bad_cells.has_value() &&
                                   bad_cells.error() == MathError::domain_error,
                               "wrong f_cells length => domain_error");
                  }

                  // Crank-Nicolson dimension mismatch.
                  auto l = nimblecas::mol_heat_operator(3, rq(1, 2), ri(1));
                  t.expect(l.has_value(), "operator builds");
                  if (l) {
                      const std::vector<double> u = {1.0, 2.0};  // wrong length (need 3)
                      auto bad = nimblecas::crank_nicolson_step(*l, u, 0.01);
                      t.expect(!bad.has_value() && bad.error() == MathError::domain_error,
                               "CN with mismatched state => domain_error");
                  }
              })
        .run();
}
