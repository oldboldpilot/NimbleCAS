// Feature/integration tests: exact linear algebra (matrix, eigen, matdecomp, bandsolve, cmatrix, matexp, lp, stats).
// @author Olumuyiwa Oluwasanmi
//
// These are cross-module *feature* tests, not per-module unit tests: each case wires
// two or more of the exact linear-algebra modules together and asserts a concrete value
// or an exact mathematical identity that only holds because every intermediate step is a
// Rational (never a float). Nothing here is approximate — the empty product of the
// eigenvalues equals the LU pivot product equals Matrix::determinant, on the nose.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.complex;
import nimblecas.matrix;
import nimblecas.eigen;
import nimblecas.matdecomp;
import nimblecas.bandsolve;
import nimblecas.cmatrix;
import nimblecas.matexp;
import nimblecas.lp;
import nimblecas.stats;
import nimblecas.testing;

using nimblecas::characteristic_polynomial;
using nimblecas::Complex;
using nimblecas::ComplexMatrix;
using nimblecas::covariance;
using nimblecas::covariance_matrix;
using nimblecas::eigenvectors_for;
using nimblecas::is_nilpotent;
using nimblecas::is_symmetric;
using nimblecas::is_tridiagonal;
using nimblecas::is_upper_triangular;
using nimblecas::LpSolution;
using nimblecas::LpStatus;
using nimblecas::lu_decompose;
using nimblecas::MathError;
using nimblecas::Matrix;
using nimblecas::matrix_exp;
using nimblecas::matrix_exp_pade;
using nimblecas::matrix_exp_taylor;
using nimblecas::maximize;
using nimblecas::mean;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::rational_eigenvalues;
using nimblecas::solve_banded;
using nimblecas::solve_tridiagonal;
using nimblecas::variance;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// --- Rational / Matrix builders (mirroring tests/matrix_tests.cpp) ----------

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

// A single-column Matrix from a list of Rationals.
[[nodiscard]] auto col(std::vector<Rational> entries) -> Matrix {
    std::vector<std::vector<Rational>> rows;
    rows.reserve(entries.size());
    for (auto& e : entries) {
        rows.push_back({std::move(e)});
    }
    return Matrix::from_rows(std::move(rows)).value();
}

// The permutation matrix P with (P*A) row i equal to A row permutation[i]; i.e. P has a
// single 1 in row i at column permutation[i]. Used to check P*A == L*U from an LU result.
[[nodiscard]] auto perm_matrix(const std::vector<std::size_t>& permutation) -> Matrix {
    const std::size_t n = permutation.size();
    std::vector<std::vector<Rational>> rows(n, std::vector<Rational>(n, ri(0)));
    for (std::size_t i = 0; i < n; ++i) {
        rows[i][permutation[i]] = ri(1);
    }
    return Matrix::from_rows(std::move(rows)).value();
}

// --- ComplexMatrix builders --------------------------------------------------

[[nodiscard]] auto ci(std::int64_t v) -> Complex {
    return Complex::from_int(v);
}

[[nodiscard]] auto cmat(std::vector<std::vector<Complex>> rows) -> ComplexMatrix {
    return ComplexMatrix::from_rows(std::move(rows)).value();
}

// --- polynomial / spectral helpers ------------------------------------------

// Horner evaluation of an exact rational polynomial at x (RationalPoly exposes no eval).
[[nodiscard]] auto poly_eval(const RationalPoly& p, const Rational& x) -> Rational {
    Rational acc;  // 0/1
    const auto coeffs = p.coefficients();  // ascending: coeffs[i] multiplies x^i
    for (std::size_t i = coeffs.size(); i-- > 0;) {
        acc = acc.multiply(x).value().add(coeffs[i]).value();
    }
    return acc;
}

// Product of the eigenvalues counted with algebraic multiplicity.
[[nodiscard]] auto eigenvalue_product(
    const std::vector<std::pair<Rational, std::int64_t>>& spectrum) -> Rational {
    Rational prod = ri(1);
    for (const auto& [value, mult] : spectrum) {
        for (std::int64_t k = 0; k < mult; ++k) {
            prod = prod.multiply(value).value();
        }
    }
    return prod;
}

// Sum of the eigenvalues counted with algebraic multiplicity.
[[nodiscard]] auto eigenvalue_sum(
    const std::vector<std::pair<Rational, std::int64_t>>& spectrum) -> Rational {
    Rational total;  // 0/1
    for (const auto& [value, mult] : spectrum) {
        auto contribution = value.multiply(ri(mult)).value();
        total = total.add(contribution).value();
    }
    return total;
}

// Product of the diagonal of U (the pivots) from an LU decomposition.
[[nodiscard]] auto diagonal_product(const Matrix& u) -> Rational {
    Rational prod = ri(1);
    for (std::size_t i = 0; i < u.rows(); ++i) {
        prod = prod.multiply(u.at(i, i)).value();
    }
    return prod;
}

}  // namespace

auto main() -> int {
    return TestSuite("feature.linalg")
        // === MATRIX core: inverse / solve / singularity / product laws ==============
        .test("inverse_is_two_sided_and_solves_its_columns",
              [](TestContext& t) {
                  // A concrete invertible 3x3 (from the unit suite's solvable system).
                  auto a = mat({{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}});
                  auto inv = a.inverse().value();
                  t.expect(a.multiply(inv).value() == Matrix::identity(3), "A * A^-1 == I");
                  t.expect(inv.multiply(a).value() == Matrix::identity(3), "A^-1 * A == I");

                  // Solving A x = b then re-multiplying must return b bit-for-bit; the
                  // solution here is the exact integer vector (2, 3, -1).
                  auto b = col({ri(8), ri(-11), ri(-3)});
                  auto x = a.solve(b).value();
                  t.expect(x == col({ri(2), ri(3), ri(-1)}), "solve yields (2,3,-1)");
                  t.expect(a.multiply(x).value() == b, "A * solve(A,b) == b exactly");

                  // The inverse's columns are exactly solve(A, e_j): compare against I.
                  t.expect(a.solve(Matrix::identity(3)).value() == inv,
                           "solve(A, I) == A^-1");
              })
        .test("solve_with_rational_solution_is_exact",
              [](TestContext& t) {
                  // 3x + y = 1, x - y = 1  =>  x = 1/2, y = -1/2 (needs exact fractions).
                  auto a = mat({{3, 1}, {1, -1}});
                  auto b = col({ri(1), ri(1)});
                  auto x = a.solve(b).value();
                  t.expect(x == col({rat(1, 2), rat(-1, 2)}), "solution is (1/2, -1/2)");
                  t.expect(a.multiply(x).value() == b, "A * x == b over Q");
              })
        .test("singular_matrix_det_zero_inverse_and_solve_domain_error",
              [](TestContext& t) {
                  auto s = mat({{1, 2}, {2, 4}});  // rank 1, det 0
                  t.expect(s.determinant().value() == ri(0), "singular det == 0");
                  t.expect(s.inverse().error() == MathError::domain_error,
                           "inverse of singular -> domain_error");
                  t.expect(s.solve(col({ri(1), ri(2)})).error() == MathError::domain_error,
                           "solve of singular -> domain_error");
              })
        .test("transpose_reverses_a_product",
              [](TestContext& t) {
                  auto a = mat({{1, 2, 3}, {4, 5, 6}});   // 2x3
                  auto b = mat({{1, 0}, {0, 1}, {1, 1}});  // 3x2
                  auto lhs = a.multiply(b).value().transpose().value();     // (A*B)^T
                  auto rhs = b.transpose().value().multiply(a.transpose().value()).value();  // B^T*A^T
                  t.expect(lhs == rhs, "(A*B)^T == B^T * A^T");
              })
        .test("determinant_is_multiplicative",
              [](TestContext& t) {
                  auto a = mat({{1, 2}, {3, 4}});   // det -2
                  auto b = mat({{2, 0}, {1, 3}});   // det  6
                  auto det_a = a.determinant().value();
                  auto det_b = b.determinant().value();
                  auto det_ab = a.multiply(b).value().determinant().value();
                  t.expect(det_a == ri(-2) && det_b == ri(6), "det A, det B by hand");
                  t.expect(det_ab == det_a.multiply(det_b).value(), "det(A*B) == det A * det B");
                  t.expect(det_ab == ri(-12), "det(A*B) == -12");
              })
        // === EIGEN <-> characteristic polynomial ====================================
        .test("char_poly_vanishes_at_each_rational_eigenvalue",
              [](TestContext& t) {
                  // [[1,2],[2,1]] has characteristic polynomial (lambda-3)(lambda+1).
                  auto a = mat({{1, 2}, {2, 1}});
                  auto poly = characteristic_polynomial(a).value();
                  auto spectrum = rational_eigenvalues(a).value();
                  t.expect(spectrum.size() == 2, "two distinct rational eigenvalues");
                  for (const auto& [value, mult] : spectrum) {
                      t.expect(poly_eval(poly, value) == ri(0),
                               "p(eigenvalue) == 0 exactly");
                      t.expect(mult == 1, "each eigenvalue is simple");
                  }
              })
        .test("eigenvalue_product_is_det_and_sum_is_trace",
              [](TestContext& t) {
                  // A non-triangular matrix whose whole spectrum is rational: {3, -1}.
                  auto a = mat({{1, 2}, {2, 1}});
                  auto spectrum = rational_eigenvalues(a).value();
                  t.expect(eigenvalue_product(spectrum) == a.determinant().value(),
                           "prod(eigenvalues) == det(A) == -3");
                  t.expect(eigenvalue_product(spectrum) == ri(-3), "product is -3");
                  t.expect(eigenvalue_sum(spectrum) == a.trace().value(),
                           "sum(eigenvalues) == trace(A) == 2");
                  t.expect(eigenvalue_sum(spectrum) == ri(2), "sum is 2");
              })
        .test("eigenvector_satisfies_A_v_equals_lambda_v",
              [](TestContext& t) {
                  auto a = mat({{1, 2}, {2, 1}});
                  const Rational lambda = ri(3);
                  auto basis = eigenvectors_for(a, lambda).value();
                  t.expect(basis.size() == 1, "eigenspace of 3 is one-dimensional");
                  auto v = col(basis.front());              // the (1,1) eigenvector
                  auto av = a.multiply(v).value();          // A v
                  auto lv = v.scale(lambda).value();        // lambda v
                  t.expect(av == lv, "A v == lambda v exactly");
                  t.expect(av == col({ri(3), ri(3)}), "A v == (3,3)");
              })
        // === MATDECOMP: LU reconstruction, pivot-product determinant, structure =====
        .test("lu_reconstructs_P_A_equals_L_U",
              [](TestContext& t) {
                  // A zero (0,0) entry forces a row swap, exercising the permutation.
                  auto a = mat({{0, 2, 1}, {1, 1, 1}, {2, 1, 0}});
                  auto lu = lu_decompose(a).value();
                  auto p = perm_matrix(lu.permutation);
                  auto pa = p.multiply(a).value();
                  auto lu_prod = lu.l.multiply(lu.u).value();
                  t.expect(pa == lu_prod, "P*A == L*U exactly");
                  t.expect(is_upper_triangular(lu.u), "U is upper-triangular");
                  t.expect(lu.sign == -1, "a single swap gives sign -1");
              })
        .test("determinant_equals_sign_times_pivot_product",
              [](TestContext& t) {
                  auto a = mat({{2, 1, 1}, {4, 3, 3}, {8, 7, 9}});
                  auto lu = lu_decompose(a).value();
                  auto from_lu = ri(lu.sign).multiply(diagonal_product(lu.u)).value();
                  t.expect(from_lu == a.determinant().value(),
                           "sign * prod(diag U) == Matrix::determinant");
                  t.expect(from_lu == ri(4), "determinant is 4");
              })
        .test("structure_predicates_on_constructed_matrices",
              [](TestContext& t) {
                  t.expect(is_symmetric(mat({{1, 2, 3}, {2, 5, 6}, {3, 6, 9}})),
                           "constructed symmetric matrix is symmetric");
                  t.expect(!is_symmetric(mat({{1, 2}, {3, 4}})), "generic matrix not symmetric");
                  t.expect(is_upper_triangular(mat({{1, 2, 3}, {0, 4, 5}, {0, 0, 6}})),
                           "constructed upper-triangular matrix");
                  t.expect(is_tridiagonal(mat({{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}})),
                           "constructed tridiagonal matrix");
                  t.expect(!is_tridiagonal(mat({{2, 0, 1}, {0, 2, 0}, {1, 0, 2}})),
                           "corner entry breaks tridiagonality");
              })
        // === BANDSOLVE <-> MATRIX ===================================================
        .test("thomas_matches_dense_solve_on_the_1d_laplacian",
              [](TestContext& t) {
                  // tridiag(-1, 2, -1), n = 4, rhs = ones  =>  x = (2,3,3,2).
                  const std::vector<Rational> sub{ri(-1), ri(-1), ri(-1)};
                  const std::vector<Rational> diag{ri(2), ri(2), ri(2), ri(2)};
                  const std::vector<Rational> super{ri(-1), ri(-1), ri(-1)};
                  const std::vector<Rational> rhs{ri(1), ri(1), ri(1), ri(1)};

                  auto thomas = solve_tridiagonal(sub, diag, super, rhs).value();
                  auto expected = std::vector<Rational>{ri(2), ri(3), ri(3), ri(2)};
                  t.expect(thomas == expected, "Thomas solution is (2,3,3,2)");

                  // Same system solved densely; the two exact solutions must coincide.
                  auto dense_a = mat({{2, -1, 0, 0}, {-1, 2, -1, 0}, {0, -1, 2, -1}, {0, 0, -1, 2}});
                  auto b = col({ri(1), ri(1), ri(1), ri(1)});
                  auto dense_x = dense_a.solve(b).value();
                  t.expect(col(thomas) == dense_x, "Thomas == Matrix::solve");
                  t.expect(dense_a.multiply(dense_x).value() == b, "A * x == b");
              })
        .test("band_solver_bandwidth_one_matches_thomas",
              [](TestContext& t) {
                  const std::vector<Rational> sub{ri(-1), ri(-1), ri(-1)};
                  const std::vector<Rational> diag{ri(2), ri(2), ri(2), ri(2)};
                  const std::vector<Rational> super{ri(-1), ri(-1), ri(-1)};
                  const std::vector<Rational> rhs{ri(1), ri(1), ri(1), ri(1)};

                  auto thomas = solve_tridiagonal(sub, diag, super, rhs).value();
                  auto dense_a = mat({{2, -1, 0, 0}, {-1, 2, -1, 0}, {0, -1, 2, -1}, {0, 0, -1, 2}});
                  auto b = col({ri(1), ri(1), ri(1), ri(1)});
                  auto banded = solve_banded(dense_a, 1, 1, b).value();  // bandwidth-1 LU
                  t.expect(banded == col(thomas), "solve_banded(bw 1) == Thomas");
                  t.expect(banded == col({ri(2), ri(3), ri(3), ri(2)}), "band solution (2,3,3,2)");
              })
        // === CMATRIX: Pauli algebra, adjoint involution, unitary/normal =============
        .test("pauli_matrices_are_hermitian_and_unitary",
              [](TestContext& t) {
                  const Complex i = Complex::i();
                  auto neg_i = i.negate().value();
                  auto x = cmat({{ci(0), ci(1)}, {ci(1), ci(0)}});
                  auto y = cmat({{ci(0), neg_i}, {i, ci(0)}});
                  auto z = cmat({{ci(1), ci(0)}, {ci(0), ci(-1)}});
                  for (const ComplexMatrix* g : {&x, &y, &z}) {
                      t.expect(g->is_hermitian().value(), "Pauli gate is Hermitian");
                      t.expect(g->is_unitary().value(), "Pauli gate is unitary");
                  }
              })
        .test("pauli_product_XY_equals_i_Z",
              [](TestContext& t) {
                  const Complex i = Complex::i();
                  auto neg_i = i.negate().value();
                  auto x = cmat({{ci(0), ci(1)}, {ci(1), ci(0)}});
                  auto y = cmat({{ci(0), neg_i}, {i, ci(0)}});
                  auto z = cmat({{ci(1), ci(0)}, {ci(0), ci(-1)}});
                  auto xy = x.multiply(y).value();
                  auto iz = z.scale(i).value();
                  t.expect(xy == iz, "X*Y == i*Z");
                  t.expect(xy == cmat({{i, ci(0)}, {ci(0), neg_i}}), "X*Y == diag(i, -i)");
              })
        .test("adjoint_involution_and_non_hermitian_unitary_is_normal",
              [](TestContext& t) {
                  const Complex i = Complex::i();
                  auto one_plus_i = Complex::make(ri(1), ri(1));
                  auto a = cmat({{one_plus_i, ci(2)}, {i, ci(3)}});
                  t.expect(a.adjoint().value().adjoint().value() == a, "adjoint(adjoint(A)) == A");

                  // The real 90-degree rotation: unitary but NOT Hermitian, hence normal.
                  auto u = cmat({{ci(0), ci(-1)}, {ci(1), ci(0)}});
                  auto uh_u = u.adjoint().value().multiply(u).value();
                  t.expect(uh_u == ComplexMatrix::identity(2), "U^H * U == I");
                  t.expect(u.is_unitary().value(), "U is unitary");
                  t.expect(!u.is_hermitian().value(), "U is not Hermitian");
                  t.expect(u.is_normal().value(), "non-Hermitian unitary is normal");
              })
        // === MATEXP: exact exponentials of nilpotents, Taylor/Pade boundary =========
        .test("exp_of_nilpotent_jordan_block_is_exact",
              [](TestContext& t) {
                  // 3x3 Jordan block N (nilpotency index 3): e^N = I + N + N^2/2 exactly.
                  auto n = mat({{0, 1, 0}, {0, 0, 1}, {0, 0, 0}});
                  t.expect(is_nilpotent(n).value(), "N is nilpotent");
                  auto exp_n = matrix_exp_taylor(n, 5).value();  // terms >= index 3 => exact
                  auto expected = Matrix::from_rows({{ri(1), ri(1), rat(1, 2)},
                                                     {ri(0), ri(1), ri(1)},
                                                     {ri(0), ri(0), ri(1)}})
                                      .value();
                  t.expect(exp_n == expected, "e^N == I + N + N^2/2");

                  // e^0 == I via the scaling-and-squaring route.
                  t.expect(matrix_exp(Matrix::zero(3, 3), 2, 0).value() == Matrix::identity(3),
                           "matrix_exp(0) == I");
              })
        .test("taylor_and_pade_agree_only_past_the_exactness_boundary",
              [](TestContext& t) {
                  // 4x4 Jordan block: nilpotency index m = 4, so [q/q] Pade is exact iff
                  // 2q+1 >= 4 (q >= 2); q = 1 (2q+1 = 3 < 4) is a genuine approximation.
                  auto j = mat({{0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}, {0, 0, 0, 0}});
                  t.expect(is_nilpotent(j).value(), "4x4 Jordan block is nilpotent");
                  auto taylor = matrix_exp_taylor(j, 4).value();  // exact e^J
                  auto pade_hi = matrix_exp_pade(j, 2).value();   // 2q+1 = 5 >= 4: exact
                  auto pade_lo = matrix_exp_pade(j, 1).value();   // 2q+1 = 3 <  4: not exact
                  t.expect(taylor == pade_hi, "Taylor == Pade(q=2) at exactness boundary");
                  t.expect_ne(taylor, pade_lo, "Pade(q=1) differs (J^3/4 vs J^3/6)");
              })
        // === LP: exact Simplex optimum / vertex, unbounded, domain errors ===========
        .test("simplex_solves_a_concrete_program_to_its_vertex",
              [](TestContext& t) {
                  // maximize 3x + 2y  s.t.  x + y <= 4,  x + 3y <= 6,  x,y >= 0.
                  const std::vector<std::vector<Rational>> a{{ri(1), ri(1)}, {ri(1), ri(3)}};
                  const std::vector<Rational> b{ri(4), ri(6)};
                  const std::vector<Rational> c{ri(3), ri(2)};
                  auto sol = maximize(a, b, c).value();
                  t.expect(sol.status == LpStatus::optimal, "program is bounded/optimal");
                  t.expect(sol.value == ri(12), "optimum value is 12");
                  t.expect(sol.solution == std::vector<Rational>{ri(4), ri(0)},
                           "optimal vertex is (4, 0)");
              })
        .test("simplex_reports_unbounded_and_domain_errors",
              [](TestContext& t) {
                  // maximize x s.t. -x <= 0 (i.e. x >= 0): objective grows without bound.
                  auto unb = maximize({{ri(-1)}}, {ri(0)}, {ri(1)}).value();
                  t.expect(unb.status == LpStatus::unbounded, "feasible cone is unbounded");
                  t.expect(unb.solution.empty(), "unbounded solution vector is empty");

                  // A negative right-hand side breaks the slack-basis feasibility precondition.
                  auto bad = maximize({{ri(1), ri(1)}}, {ri(-1)}, {ri(1), ri(1)});
                  t.expect(bad.error() == MathError::domain_error, "b < 0 -> domain_error");
              })
        // === STATS: exact moments and the covariance/linearity identities ===========
        .test("mean_and_variance_are_exact_fractions",
              [](TestContext& t) {
                  const std::vector<Rational> data{ri(1), ri(2), ri(3), ri(4), ri(5)};
                  t.expect(mean(data).value() == ri(3), "mean of 1..5 is 3");
                  t.expect(variance(data, true).value() == rat(5, 2), "sample variance is 5/2");
                  t.expect(variance(data, false).value() == ri(2), "population variance is 2");
              })
        .test("covariance_matrix_is_symmetric_with_hand_entries",
              [](TestContext& t) {
                  // x = (1,2,3), y = 2x = (2,4,6): a perfectly linear relation.
                  const std::vector<Rational> x{ri(1), ri(2), ri(3)};
                  const std::vector<Rational> y{ri(2), ri(4), ri(6)};
                  std::vector<std::span<const Rational>> vars{std::span<const Rational>(x),
                                                              std::span<const Rational>(y)};
                  auto sigma = covariance_matrix(vars, true).value();
                  t.expect(is_symmetric(sigma), "covariance matrix is symmetric");
                  // Sigma = [[var(x), cov], [cov, var(y)]] = [[1, 2], [2, 4]] by hand.
                  t.expect(sigma == mat({{1, 2}, {2, 4}}), "Sigma == [[1,2],[2,4]]");
                  t.expect(sigma.at(0, 0) == variance(x, true).value(), "diagonal is var(x)");
                  t.expect(sigma.at(1, 1) == variance(y, true).value(), "diagonal is var(y)");

                  // Perfect linearity <=> correlation == 1, i.e. cov^2 == var(x)*var(y)
                  // exactly (correlation itself needs an irrational sqrt, so we test the
                  // exact squared identity that a rational CAS can certify).
                  auto cov = covariance(x, y, true).value();
                  auto cov_sq = cov.multiply(cov).value();
                  auto var_prod = variance(x, true).value().multiply(variance(y, true).value()).value();
                  t.expect(cov_sq == var_prod, "cov^2 == var(x)*var(y): correlation is 1");
              })
        // === CROSS: three independent routes to the same determinant ================
        .test("determinant_agrees_across_eigen_lu_and_gauss",
              [](TestContext& t) {
                  auto a = mat({{1, 2}, {2, 1}});  // spectrum {3, -1}, det -3
                  auto from_gauss = a.determinant().value();
                  auto from_eigen = eigenvalue_product(rational_eigenvalues(a).value());
                  auto lu = lu_decompose(a).value();
                  auto from_lu = ri(lu.sign).multiply(diagonal_product(lu.u)).value();
                  t.expect(from_gauss == ri(-3), "Gauss determinant is -3");
                  t.expect(from_eigen == from_gauss, "eigenvalue product == Gauss determinant");
                  t.expect(from_lu == from_gauss, "LU pivot product == Gauss determinant");
              })
        .run();
}
