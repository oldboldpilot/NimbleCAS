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
        .run();
}
