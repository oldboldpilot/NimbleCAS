// Tests for nimblecas.krylov: Krylov subspace methods (ROADMAP §7.2).
// @author Olumuyiwa Oluwasanmi
//
// Covers the honesty boundary explicitly:
//   * EXACT rational CG solves an SPD system to the exact fraction and cross-checks
//     both A*x == b and Matrix::solve; it terminates within n steps; a non-SPD input
//     is rejected with domain_error.
//   * The EXACT rational Krylov power basis equals {b, Ab, A^2 b}.
//   * NUMERICAL GMRES / BiCGSTAB solve a non-symmetric double system to a residual
//     tolerance, and a capped-iteration run reports converged == false without erroring.
//   * NUMERICAL CG solves an SPD system built via csr_matvec, exactly recovering a known
//     solution; on a symmetric INDEFINITE matrix it reports converged == false (a
//     breakdown, never a spuriously "converged" wrong answer). csr_matvec itself is
//     cross-checked against dense_matvec on a small hand-verified matrix.

import std;
import nimblecas.core;
import nimblecas.matrix;
import nimblecas.ratpoly;
import nimblecas.krylov;
import nimblecas.testing;

using nimblecas::MathError;
using nimblecas::Matrix;
using nimblecas::Rational;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

[[nodiscard]] auto ri(std::int64_t v) -> Rational {
    return Rational::from_int(v);
}

// Build a Matrix from integer rows (low-index row first).
[[nodiscard]] auto mat(std::vector<std::vector<std::int64_t>> rows) -> Matrix {
    std::vector<std::vector<Rational>> r;
    r.reserve(rows.size());
    for (const auto& row : rows) {
        std::vector<Rational> rr;
        rr.reserve(row.size());
        for (const std::int64_t v : row) {
            rr.push_back(Rational::from_int(v));
        }
        r.push_back(std::move(rr));
    }
    return Matrix::from_rows(std::move(r)).value();
}

// A rational column vector (n x 1 Matrix) from integer entries.
[[nodiscard]] auto col(std::vector<std::int64_t> entries) -> Matrix {
    std::vector<std::vector<Rational>> r;
    r.reserve(entries.size());
    for (const std::int64_t v : entries) {
        r.push_back(std::vector<Rational>{Rational::from_int(v)});
    }
    return Matrix::from_rows(std::move(r)).value();
}

// A std::vector<Rational> from integer entries.
[[nodiscard]] auto rvec(std::vector<std::int64_t> entries) -> std::vector<Rational> {
    std::vector<Rational> out;
    out.reserve(entries.size());
    for (const std::int64_t v : entries) {
        out.push_back(Rational::from_int(v));
    }
    return out;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.krylov")
        .test("exact_cg_solves_spd_to_exact_rational",
              [](TestContext& t) {
                  // A = [[4,1],[1,3]] is SPD; b = [1,2]. Exact solution x = [1/11, 7/11].
                  auto a = mat({{4, 1}, {1, 3}});
                  auto b = col({1, 2});
                  auto x = nimblecas::conjugate_gradient(a, b);
                  t.expect(x.has_value(), "CG succeeds on an SPD system");
                  auto expected = Matrix::from_rows(
                                      {{rat(1, 11)}, {rat(7, 11)}})
                                      .value();
                  t.expect(x.has_value() && *x == expected, "CG gives the exact rational solution");
                  // Cross-check A*x == b exactly.
                  t.expect(x.has_value() && a.multiply(*x).value() == b, "A*x == b exactly");
                  // Cross-check against the direct exact solver.
                  t.expect(x.has_value() && *x == a.solve(b).value(),
                           "CG agrees with Matrix::solve");
              })
        .test("exact_cg_terminates_within_n_steps",
              [](TestContext& t) {
                  // A 3x3 SPD system: CG must reach the exact answer in <= 3 steps.
                  auto a = mat({{4, 1, 0}, {1, 3, 1}, {0, 1, 2}});
                  auto b = col({1, 2, 3});
                  auto r = nimblecas::conjugate_gradient_steps(a, b);
                  t.expect(r.has_value(), "CG_steps succeeds on the 3x3 SPD system");
                  t.expect(r.has_value() && r->steps <= 3, "terminates within n = 3 steps");
                  t.expect(r.has_value() && a.multiply(r->solution).value() == b,
                           "the returned solution is exact (A*x == b)");
              })
        .test("non_spd_is_rejected",
              [](TestContext& t) {
                  // Symmetric but INDEFINite (eigenvalues 3, -1): CG hits a direction with
                  // p^T A p < 0 and must reject as domain_error.
                  auto indef = mat({{1, 2}, {2, 1}});
                  auto b = col({1, 0});
                  t.expect(nimblecas::conjugate_gradient(indef, b).error() == MathError::domain_error,
                           "symmetric-indefinite input -> domain_error");
                  // Non-symmetric input is rejected up front.
                  auto nonsym = mat({{1, 2}, {3, 4}});
                  t.expect(nimblecas::conjugate_gradient(nonsym, col({1, 1})).error() ==
                               MathError::domain_error,
                           "non-symmetric input -> domain_error");
                  // Shape mismatch is rejected.
                  t.expect(nimblecas::conjugate_gradient(mat({{4, 1}, {1, 3}}), col({1, 2, 3}))
                                   .error() == MathError::domain_error,
                           "wrong-shape b -> domain_error");
              })
        .test("exact_krylov_basis_matches_power_sequence",
              [](TestContext& t) {
                  // A = diag(2,3), b = [1,1]. {b, Ab, A^2 b} = {[1,1],[2,3],[4,9]}.
                  auto a = mat({{2, 0}, {0, 3}});
                  auto b = rvec({1, 1});
                  auto basis = nimblecas::krylov_basis(a, b, 3);
                  t.expect(basis.has_value(), "krylov_basis succeeds");
                  t.expect(basis.has_value() && basis->size() == 3, "three basis vectors");
                  if (basis.has_value() && basis->size() == 3) {
                      t.expect((*basis)[0] == rvec({1, 1}), "v0 == b");
                      t.expect((*basis)[1] == rvec({2, 3}), "v1 == A b");
                      t.expect((*basis)[2] == rvec({4, 9}), "v2 == A^2 b");
                  }
              })
        .test("exact_rational_arnoldi_and_lanczos_projection",
              [](TestContext& t) {
                  // Symmetric A: the unnormalised rational Lanczos reproduces A on the
                  // full Krylov space exactly (breakdown => A*basis == basis*T over Q).
                  auto a = mat({{2, 1}, {1, 2}});
                  auto b = rvec({1, 0});
                  auto lan = nimblecas::lanczos_rational(a, b, 2);
                  t.expect(lan.has_value(), "lanczos_rational succeeds on symmetric A");
                  if (lan.has_value()) {
                      // Diagonal alpha_0 = <A q0, q0>/<q0,q0> = A[0][0] = 2.
                      t.expect(!lan->alpha.empty() && lan->alpha.front() == ri(2),
                               "alpha_0 == 2");
                      // The tridiagonal has the unit subdiagonal (unnormalised form).
                      t.expect(lan->tridiagonal.rows() == 2 &&
                                   lan->tridiagonal.at(1, 0) == ri(1),
                               "unit subdiagonal in the exact tridiagonal");
                  }
                  // A non-symmetric matrix is rejected by Lanczos but accepted by Arnoldi.
                  auto nonsym = mat({{1, 2}, {3, 4}});
                  t.expect(nimblecas::lanczos_rational(nonsym, rvec({1, 0}), 2).error() ==
                               MathError::domain_error,
                           "Lanczos rejects a non-symmetric matrix");
                  auto arn = nimblecas::arnoldi_rational(nonsym, rvec({1, 0}), 2);
                  t.expect(arn.has_value(), "Arnoldi accepts a non-symmetric matrix");
              })
        .test("numerical_gmres_and_bicgstab_solve_nonsymmetric",
              [](TestContext& t) {
                  // A non-symmetric 3x3 double system.
                  const std::array<double, 9> adata{3.0, 1.0, 0.0,
                                                    0.0, 4.0, 1.0,
                                                    1.0, 0.0, 5.0};
                  const std::array<double, 3> b{1.0, 2.0, 3.0};
                  auto A = nimblecas::dense_matvec(adata, 3);

                  auto g = nimblecas::gmres(A, b, 1e-12, 100, 3);
                  t.expect(g.has_value(), "GMRES returns a result");
                  t.expect(g.has_value() && g->converged, "GMRES converges");
                  t.expect(g.has_value() && g->residual < 1e-8, "GMRES residual within tolerance");

                  auto bc = nimblecas::bicgstab(A, b, 1e-12, 100);
                  t.expect(bc.has_value(), "BiCGSTAB returns a result");
                  t.expect(bc.has_value() && bc->converged, "BiCGSTAB converges");
                  t.expect(bc.has_value() && bc->residual < 1e-8,
                           "BiCGSTAB residual within tolerance");

                  // The two numerical solutions must agree with each other.
                  if (g.has_value() && bc.has_value()) {
                      double diff = 0.0;
                      for (std::size_t i = 0; i < 3; ++i) {
                          diff = std::max(diff, std::abs(g->x[i] - bc->x[i]));
                      }
                      t.expect(diff < 1e-6, "GMRES and BiCGSTAB agree");
                  }
              })
        .test("capped_iterations_report_not_converged_without_error",
              [](TestContext& t) {
                  const std::array<double, 9> adata{3.0, 1.0, 0.0,
                                                    0.0, 4.0, 1.0,
                                                    1.0, 0.0, 5.0};
                  const std::array<double, 3> b{1.0, 2.0, 3.0};
                  auto A = nimblecas::dense_matvec(adata, 3);

                  // A single BiCGSTAB step at a tight tolerance cannot converge.
                  auto bc = nimblecas::bicgstab(A, b, 1e-14, 1);
                  t.expect(bc.has_value(), "capped BiCGSTAB does NOT error");
                  t.expect(bc.has_value() && !bc->converged,
                           "capped BiCGSTAB reports converged == false");
                  t.expect(bc.has_value() && bc->iterations == 1, "exactly one iteration ran");

                  // Likewise a single GMRES iteration on the same tight tolerance.
                  auto g = nimblecas::gmres(A, b, 1e-14, 1, 3);
                  t.expect(g.has_value() && !g->converged,
                           "capped GMRES reports converged == false without error");

                  // An empty system is the only genuinely invalid input.
                  std::array<double, 0> empty{};
                  t.expect(nimblecas::bicgstab(A, empty, 1e-10, 10).error() ==
                               MathError::domain_error,
                           "empty system -> domain_error");
              })
        .test("csr_matvec_matches_dense_reference",
              [](TestContext& t) {
                  // A = [[1,2,0],[0,3,4],[5,0,6]] (deliberately non-symmetric).
                  const std::array<double, 9> dense{1, 2, 0,
                                                    0, 3, 4,
                                                    5, 0, 6};
                  const std::array<int, 4> row_offsets{0, 2, 4, 6};
                  const std::array<int, 6> col_indices{0, 1, 1, 2, 0, 2};
                  const std::array<double, 6> values{1, 2, 3, 4, 5, 6};

                  auto dense_op = nimblecas::dense_matvec(dense, 3);
                  auto csr_op = nimblecas::csr_matvec(row_offsets, col_indices, values, 3);

                  const std::array<double, 3> x{1.0, 2.0, 3.0};
                  std::array<double, 3> y_dense{};
                  std::array<double, 3> y_csr{};
                  dense_op(x, y_dense);
                  csr_op(x, y_csr);
                  for (std::size_t i = 0; i < 3; ++i) {
                      t.expect(y_dense[i] == y_csr[i], "csr_matvec matches the dense reference");
                  }
                  // Cross-check the hand-computed values too: A*[1,2,3] = [5,18,23].
                  const std::array<double, 3> expected{5.0, 18.0, 23.0};
                  for (std::size_t i = 0; i < 3; ++i) {
                      t.expect(y_csr[i] == expected[i], "csr_matvec gives the hand-verified result");
                  }
              })
        .test("numerical_cg_solves_spd_system_via_csr_and_recovers_exact_solution",
              [](TestContext& t) {
                  // Tridiagonal SPD: diag 4, off-diag 1 (diagonally dominant => SPD).
                  // x_true = [1,2,3,4]; b = A * x_true, hand-computed: [6,12,18,19].
                  const std::array<int, 5> row_offsets{0, 2, 5, 8, 10};
                  const std::array<int, 10> col_indices{0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
                  const std::array<double, 10> values{4, 1, 1, 4, 1, 1, 4, 1, 1, 4};
                  auto A = nimblecas::csr_matvec(row_offsets, col_indices, values, 4);

                  const std::array<double, 4> b{6.0, 12.0, 18.0, 19.0};
                  auto r = nimblecas::cg(A, b, 1e-12, 100);
                  t.expect(r.has_value(), "cg returns a result");
                  t.expect(r.has_value() && r->converged,
                           "cg converges on the well-conditioned SPD tridiagonal system");
                  t.expect(r.has_value() && r->iterations <= 4,
                           "well-conditioned SPD 4x4 system: iterations <= n");
                  if (r.has_value()) {
                      const std::array<double, 4> expected{1.0, 2.0, 3.0, 4.0};
                      bool all_close = true;
                      for (std::size_t i = 0; i < 4; ++i) {
                          const double rel = std::abs(r->x[i] - expected[i]) / expected[i];
                          all_close = all_close && rel < 1e-9;
                      }
                      t.expect(all_close, "cg recovers x_true to within 1e-9 relative");
                  }
              })
        .test("numerical_cg_solves_2d_laplacian_csr",
              [](TestContext& t) {
                  // Standard 5-point Poisson stencil on a 10x10 grid (n = 100), implicit
                  // zero Dirichlet boundary: diagonal 4, off-diagonal -1 per existing
                  // neighbor. Diagonally dominant (strictly on the boundary) symmetric =>
                  // SPD.
                  constexpr std::size_t nx = 10;
                  constexpr std::size_t ny = 10;
                  constexpr std::size_t n = nx * ny;
                  std::vector<int> row_offsets;
                  std::vector<int> col_indices;
                  std::vector<double> values;
                  row_offsets.reserve(n + 1);
                  row_offsets.push_back(0);
                  for (std::size_t i = 0; i < nx; ++i) {
                      for (std::size_t j = 0; j < ny; ++j) {
                          const std::size_t idx = i * ny + j;
                          if (i > 0) {
                              col_indices.push_back(static_cast<int>(idx - ny));
                              values.push_back(-1.0);
                          }
                          if (j > 0) {
                              col_indices.push_back(static_cast<int>(idx - 1));
                              values.push_back(-1.0);
                          }
                          col_indices.push_back(static_cast<int>(idx));
                          values.push_back(4.0);
                          if (j + 1 < ny) {
                              col_indices.push_back(static_cast<int>(idx + 1));
                              values.push_back(-1.0);
                          }
                          if (i + 1 < nx) {
                              col_indices.push_back(static_cast<int>(idx + ny));
                              values.push_back(-1.0);
                          }
                          row_offsets.push_back(static_cast<int>(col_indices.size()));
                      }
                  }
                  auto A = nimblecas::csr_matvec(row_offsets, col_indices, values, n);
                  const std::vector<double> b(n, 1.0);
                  auto r = nimblecas::cg(A, b, 1e-10, 500);
                  t.expect(r.has_value(), "cg returns a result on the 2D Laplacian");
                  t.expect(r.has_value() && r->converged, "cg converges on the 2D Laplacian");
                  t.expect(r.has_value() && r->residual <= 1e-10 * std::sqrt(static_cast<double>(n)),
                           "true residual within tol * ||b||");
              })
        .test("numerical_cg_reports_not_converged_on_indefinite_breakdown",
              [](TestContext& t) {
                  // A = [[0,1],[1,0]] is symmetric but INDEFINITE (eigenvalues +1, -1).
                  // With b = [0,1] the very first search direction breaks down exactly:
                  // p0 = r0 = b, and p0^T A p0 = 0*1 + 1*0 = 0 exactly (not merely small),
                  // so CG cannot proceed past the first step. x must stay at the (wrong)
                  // starting guess [0,0] rather than being reported as a spuriously
                  // "converged" answer.
                  const std::array<double, 4> dense{0, 1,
                                                    1, 0};
                  auto A = nimblecas::dense_matvec(dense, 2);
                  const std::array<double, 2> b{0.0, 1.0};
                  auto r = nimblecas::cg(A, b, 1e-10, 100);
                  t.expect(r.has_value(), "cg does NOT error on an indefinite matrix");
                  t.expect(r.has_value() && !r->converged,
                           "indefinite breakdown -> converged == false, never a false positive");
                  t.expect(r.has_value() && r->iterations == 1,
                           "breakdown stops after exactly one iteration");
                  if (r.has_value()) {
                      t.expect(std::abs(r->x[0]) < 1e-15 && std::abs(r->x[1]) < 1e-15,
                               "x remains the zero initial guess (no wrong 'solution')");
                      t.expect(std::abs(r->residual - 1.0) < 1e-12,
                               "true residual equals ||b|| since x never moved");
                  }
              })
        .test("numerical_cg_empty_system_is_domain_error",
              [](TestContext& t) {
                  const std::array<double, 4> dense{4, 1, 1, 4};
                  auto A = nimblecas::dense_matvec(dense, 2);
                  const std::array<double, 0> empty{};
                  t.expect(nimblecas::cg(A, empty, 1e-10, 10).error() == MathError::domain_error,
                           "empty system -> domain_error");
              })
        .run();
}
