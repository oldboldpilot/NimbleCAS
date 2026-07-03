// Tests for nimblecas.matdecomp: exact LU decomposition & special-matrix predicates (ROADMAP 7.2).
// @author Olumuyiwa Oluwasanmi
//
// Exact integer matrices are used throughout, so every pivot, multiplier, and structural
// comparison is exact and deterministic.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.matdecomp;
import nimblecas.testing;

using nimblecas::is_banded;
using nimblecas::is_diagonal;
using nimblecas::is_identity;
using nimblecas::is_lower_hessenberg;
using nimblecas::is_lower_triangular;
using nimblecas::is_skew_symmetric;
using nimblecas::is_symmetric;
using nimblecas::is_toeplitz;
using nimblecas::is_tridiagonal;
using nimblecas::is_upper_hessenberg;
using nimblecas::is_upper_triangular;
using nimblecas::lu_decompose;
using nimblecas::MathError;
using nimblecas::Matrix;
using nimblecas::Rational;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

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

// Build P*A directly from the recorded permutation: (P*A) row i is A row permutation[i].
[[nodiscard]] auto permute_rows(const Matrix& a, const std::vector<std::size_t>& perm) -> Matrix {
    std::vector<std::vector<Rational>> rows;
    rows.reserve(perm.size());
    for (const std::size_t src : perm) {
        std::vector<Rational> row;
        row.reserve(a.cols());
        for (std::size_t j = 0; j < a.cols(); ++j) {
            row.push_back(a.at(src, j));
        }
        rows.push_back(std::move(row));
    }
    return Matrix::from_rows(std::move(rows)).value();
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.matdecomp")
        .test("lu_standard_example",
              [](TestContext& t) {
                  // A = [[2,1,1],[4,3,3],[8,7,9]]. First-nonzero pivoting needs no swaps, so
                  // (by hand) L = [[1,0,0],[2,1,0],[4,3,1]], U = [[2,1,1],[0,1,1],[0,0,2]],
                  // permutation = [0,1,2], sign = +1. Verify L*U == A row-for-row:
                  //   row0: 1*[2,1,1]                       = [2,1,1]
                  //   row1: 2*[2,1,1] + 1*[0,1,1]           = [4,3,3]
                  //   row2: 4*[2,1,1] + 3*[0,1,1] + 1*[0,0,2] = [8,7,9]
                  const Matrix a = mat({{2, 1, 1}, {4, 3, 3}, {8, 7, 9}});
                  auto lu = lu_decompose(a);
                  t.expect(lu.has_value(), "lu_decompose succeeds on the non-singular example");
                  if (!lu) {
                      return;
                  }
                  t.expect(lu->sign == 1, "no swaps needed => sign +1");
                  t.expect(lu->permutation == std::vector<std::size_t>{0, 1, 2},
                           "permutation is the identity order");

                  // L unit lower-triangular.
                  bool l_ok = true;
                  for (std::size_t i = 0; i < 3; ++i) {
                      l_ok = l_ok && (lu->l.at(i, i) == ri(1));
                      for (std::size_t j = i + 1; j < 3; ++j) {
                          l_ok = l_ok && lu->l.at(i, j).is_zero();
                      }
                  }
                  t.expect(l_ok, "L has unit diagonal and zeros above it");

                  // U upper-triangular.
                  bool u_ok = true;
                  for (std::size_t i = 0; i < 3; ++i) {
                      for (std::size_t j = 0; j < i; ++j) {
                          u_ok = u_ok && lu->u.at(i, j).is_zero();
                      }
                  }
                  t.expect(u_ok, "U is zero below the diagonal");

                  // Reconstruct: P*A == L*U.
                  auto lhs = permute_rows(a, lu->permutation);
                  auto rhs = lu->l.multiply(lu->u);
                  t.expect(rhs.has_value(), "L*U multiplies");
                  t.expect(rhs.has_value() && lhs.is_equal(*rhs), "P*A == L*U");
              })
        .test("lu_requires_pivot_swap",
              [](TestContext& t) {
                  // A = [[0,1],[1,0]] has a zero (0,0) pivot: a swap is forced, flipping sign.
                  const Matrix a = mat({{0, 1}, {1, 0}});
                  auto lu = lu_decompose(a);
                  t.expect(lu.has_value(), "lu_decompose succeeds with a pivot swap");
                  if (!lu) {
                      return;
                  }
                  t.expect(lu->sign == -1, "a single row swap => sign -1");
                  t.expect(lu->permutation == std::vector<std::size_t>{1, 0},
                           "rows are swapped in the permutation");
                  auto lhs = permute_rows(a, lu->permutation);
                  auto rhs = lu->l.multiply(lu->u);
                  t.expect(rhs.has_value() && lhs.is_equal(*rhs), "P*A == L*U after the swap");
              })
        .test("lu_singular_is_domain_error",
              [](TestContext& t) {
                  // A = [[1,2],[2,4]]: second row is 2x the first => singular.
                  auto lu = lu_decompose(mat({{1, 2}, {2, 4}}));
                  t.expect(!lu.has_value(), "singular matrix has no standard LU");
                  t.expect(!lu.has_value() && lu.error() == MathError::domain_error,
                           "singular => domain_error");
              })
        .test("lu_non_square_is_domain_error",
              [](TestContext& t) {
                  auto lu = lu_decompose(mat({{1, 2, 3}, {4, 5, 6}}));
                  t.expect(!lu.has_value() && lu.error() == MathError::domain_error,
                           "non-square A => domain_error");
              })
        .test("predicate_symmetric",
              [](TestContext& t) {
                  t.expect(is_symmetric(mat({{1, 2}, {2, 3}})), "[[1,2],[2,3]] is symmetric");
                  t.expect(!is_symmetric(mat({{1, 2}, {3, 4}})), "[[1,2],[3,4]] is not symmetric");
                  t.expect(!is_symmetric(mat({{1, 2, 3}, {2, 4, 5}})),
                           "non-square => not symmetric");
              })
        .test("predicate_skew_symmetric",
              [](TestContext& t) {
                  t.expect(is_skew_symmetric(mat({{0, 2}, {-2, 0}})),
                           "[[0,2],[-2,0]] is skew-symmetric");
                  t.expect(!is_skew_symmetric(mat({{0, 2}, {2, 0}})),
                           "[[0,2],[2,0]] is not skew-symmetric");
                  t.expect(!is_skew_symmetric(mat({{1, 2}, {-2, 0}})),
                           "nonzero diagonal => not skew-symmetric");
              })
        .test("predicate_diagonal",
              [](TestContext& t) {
                  t.expect(is_diagonal(mat({{1, 0, 0}, {0, 2, 0}, {0, 0, 3}})),
                           "diag(1,2,3) is diagonal");
                  t.expect(!is_diagonal(mat({{1, 2, 0}, {0, 2, 0}, {0, 0, 3}})),
                           "an off-diagonal entry breaks diagonality");
              })
        .test("predicate_triangular",
              [](TestContext& t) {
                  t.expect(is_upper_triangular(mat({{1, 2}, {0, 3}})),
                           "[[1,2],[0,3]] is upper-triangular");
                  t.expect(!is_lower_triangular(mat({{1, 2}, {0, 3}})),
                           "[[1,2],[0,3]] is not lower-triangular");
                  t.expect(is_lower_triangular(mat({{1, 0}, {2, 3}})),
                           "[[1,0],[2,3]] is lower-triangular");
              })
        .test("predicate_tridiagonal",
              [](TestContext& t) {
                  t.expect(is_tridiagonal(mat({{1, 2, 0}, {3, 4, 5}, {0, 6, 7}})),
                           "[[1,2,0],[3,4,5],[0,6,7]] is tridiagonal");
                  t.expect(!is_tridiagonal(mat({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}})),
                           "a full 3x3 is not tridiagonal");
              })
        .test("predicate_banded",
              [](TestContext& t) {
                  // The tridiagonal example is (1,1)-banded but not (0,0)-banded.
                  const Matrix tri = mat({{1, 2, 0}, {3, 4, 5}, {0, 6, 7}});
                  t.expect(is_banded(tri, 1, 1), "tridiagonal is (1,1)-banded");
                  t.expect(!is_banded(tri, 0, 0), "tridiagonal is not (0,0)-banded");
                  t.expect(is_banded(mat({{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}), 0, 0),
                           "a diagonal matrix is (0,0)-banded");
              })
        .test("predicate_upper_hessenberg",
              [](TestContext& t) {
                  t.expect(is_upper_hessenberg(mat({{1, 2, 3}, {4, 5, 6}, {0, 7, 8}})),
                           "[[1,2,3],[4,5,6],[0,7,8]] is upper-Hessenberg");
                  t.expect(!is_upper_hessenberg(mat({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}})),
                           "a nonzero (2,0) breaks upper-Hessenberg");
              })
        .test("predicate_toeplitz",
              [](TestContext& t) {
                  t.expect(is_toeplitz(mat({{1, 2, 3}, {4, 1, 2}, {5, 4, 1}})),
                           "[[1,2,3],[4,1,2],[5,4,1]] is Toeplitz");
                  t.expect(!is_toeplitz(mat({{1, 2}, {3, 5}})),
                           "[[1,2],[3,5]] is not Toeplitz");
              })
        .test("predicate_identity",
              [](TestContext& t) {
                  t.expect(is_identity(Matrix::identity(3)), "identity(3) is the identity");
                  t.expect(!is_identity(mat({{1, 0, 0}, {0, 2, 0}, {0, 0, 3}})),
                           "diag(1,2,3) is not the identity");
              })
        .run();
}
