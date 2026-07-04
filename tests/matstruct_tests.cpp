// Tests for nimblecas.matstruct: structured builders/predicates + exact-over-Q
// factorizations (LDL^T, exact Cholesky, rational Hessenberg similarity) (ROADMAP 7.2).
// @author Olumuyiwa Oluwasanmi
//
// Exact rational entries throughout, so every reconstruction is verified for EXACT equality
// (P*A == L*D*L^T, G*G^T == A) with no tolerance. The Hessenberg similarity is checked by
// comparing the exact characteristic polynomial (Faddeev-LeVerrier, nimblecas.eigen) of the
// input and the reduced form — a similarity invariant.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.matdecomp;  // is_symmetric, is_upper_hessenberg (reused, not duplicated)
import nimblecas.eigen;      // characteristic_polynomial for the similarity check
import nimblecas.matstruct;
import nimblecas.testing;

using nimblecas::block_diagonal;
using nimblecas::block_upper_triangular;
using nimblecas::characteristic_polynomial;
using nimblecas::cholesky_exact;
using nimblecas::diagonal;
using nimblecas::hessenberg_form;
using nimblecas::is_block_diagonal;
using nimblecas::is_block_upper_triangular;
using nimblecas::is_symmetric_positive_definite;
using nimblecas::is_upper_hessenberg;
using nimblecas::ldlt_decompose;
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

// L * D * L^T as an exact Matrix (used to check LDL^T reconstruction).
[[nodiscard]] auto reconstruct_ldlt(const Matrix& l, const Matrix& d) -> Matrix {
    auto lt = l.transpose().value();
    auto ld = l.multiply(d).value();
    return ld.multiply(lt).value();
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.matstruct")
        .test("ldlt_reconstructs_spd",
              [](TestContext& t) {
                  // A = [[4,2,0],[2,5,2],[0,2,3]] is SPD. LDL^T (by hand):
                  //   D0 = 4, L10 = 1/2, L20 = 0
                  //   D1 = 5 - (1/2)^2 * 4 = 4, L21 = (2 - 0)/4 = 1/2
                  //   D2 = 3 - (1/2)^2 * 4 = 2
                  const Matrix a = mat({{4, 2, 0}, {2, 5, 2}, {0, 2, 3}});
                  auto ldlt = ldlt_decompose(a);
                  t.expect(ldlt.has_value(), "ldlt_decompose succeeds on an SPD matrix");
                  if (!ldlt) {
                      return;
                  }
                  // L unit lower-triangular (1 on diagonal, 0 strictly above).
                  bool l_ok = true;
                  for (std::size_t i = 0; i < 3; ++i) {
                      l_ok = l_ok && (ldlt->l.at(i, i) == ri(1));
                      for (std::size_t j = i + 1; j < 3; ++j) {
                          l_ok = l_ok && ldlt->l.at(i, j).is_zero();
                      }
                  }
                  t.expect(l_ok, "L is unit lower-triangular");
                  // D diagonal.
                  bool d_ok = true;
                  for (std::size_t i = 0; i < 3; ++i) {
                      for (std::size_t j = 0; j < 3; ++j) {
                          if (i != j) {
                              d_ok = d_ok && ldlt->d.at(i, j).is_zero();
                          }
                      }
                  }
                  t.expect(d_ok, "D is diagonal");
                  t.expect(ldlt->d.at(0, 0) == ri(4) && ldlt->d.at(1, 1) == ri(4) &&
                               ldlt->d.at(2, 2) == ri(2),
                           "pivots are (4, 4, 2)");
                  // Exact reconstruction L*D*L^T == A.
                  t.expect(reconstruct_ldlt(ldlt->l, ldlt->d).is_equal(a), "L*D*L^T == A exactly");
              })
        .test("ldlt_rejects_non_symmetric_and_zero_pivot",
              [](TestContext& t) {
                  auto ns = ldlt_decompose(mat({{1, 2}, {3, 4}}));
                  t.expect(!ns.has_value() && ns.error() == MathError::domain_error,
                           "non-symmetric A => domain_error");
                  // Symmetric but a zero leading pivot forces pivoting: [[0,1],[1,0]].
                  auto zp = ldlt_decompose(mat({{0, 1}, {1, 0}}));
                  t.expect(!zp.has_value() && zp.error() == MathError::domain_error,
                           "zero pivot => domain_error");
                  auto rect = ldlt_decompose(mat({{1, 2, 3}, {2, 4, 5}}));
                  t.expect(!rect.has_value() && rect.error() == MathError::domain_error,
                           "non-square A => domain_error");
              })
        .test("spd_detection",
              [](TestContext& t) {
                  t.expect(is_symmetric_positive_definite(mat({{2, 1}, {1, 2}})),
                           "[[2,1],[1,2]] is SPD (pivots 2, 3/2)");
                  t.expect(is_symmetric_positive_definite(mat({{4, 2, 0}, {2, 5, 2}, {0, 2, 3}})),
                           "the 3x3 example is SPD");
                  t.expect(!is_symmetric_positive_definite(mat({{1, 2}, {2, 1}})),
                           "symmetric but indefinite (det -3) is not SPD");
                  t.expect(!is_symmetric_positive_definite(mat({{1, 2}, {3, 4}})),
                           "non-symmetric is not SPD");
                  t.expect(!is_symmetric_positive_definite(mat({{0, 1}, {1, 0}})),
                           "zero-pivot symmetric matrix is not SPD");
              })
        .test("block_diagonal_determinant_is_product",
              [](TestContext& t) {
                  // det([[1,2],[3,4]]) = -2, det([[2]]) = 2, det([[1,1],[0,3]]) = 3.
                  const Matrix b0 = mat({{1, 2}, {3, 4}});
                  const Matrix b1 = mat({{2}});
                  const Matrix b2 = mat({{1, 1}, {0, 3}});
                  auto blk = block_diagonal({b0, b1, b2});
                  t.expect(blk.has_value(), "block_diagonal assembles");
                  if (!blk) {
                      return;
                  }
                  t.expect(blk->rows() == 5 && blk->cols() == 5, "block-diagonal is 5x5");
                  t.expect(is_block_diagonal(*blk, {2, 1, 2}),
                           "assembled matrix is block-diagonal for sizes (2,1,2)");
                  auto det = blk->determinant();
                  // product of block determinants = (-2) * 2 * 3 = -12.
                  t.expect(det.has_value() && *det == ri(-12),
                           "det(block-diagonal) == product of block determinants (-12)");
              })
        .test("hessenberg_is_upper_hessenberg_and_similar",
              [](TestContext& t) {
                  // A dense 4x4 with nonzero entries below the subdiagonal.
                  const Matrix a =
                      mat({{4, 1, 2, 3}, {1, 3, 5, 1}, {2, 5, 6, 2}, {7, 1, 4, 5}});
                  auto h = hessenberg_form(a);
                  t.expect(h.has_value(), "hessenberg_form succeeds");
                  if (!h) {
                      return;
                  }
                  t.expect(is_upper_hessenberg(*h), "result is upper-Hessenberg");
                  auto pa = characteristic_polynomial(a);
                  auto ph = characteristic_polynomial(*h);
                  t.expect(pa.has_value() && ph.has_value() && pa->is_equal(*ph),
                           "H is similar to A: identical characteristic polynomials");
              })
        .test("hessenberg_with_pivot_swap",
              [](TestContext& t) {
                  // The (1,0) subdiagonal entry is 0 but (2,0) is nonzero, forcing a
                  // symmetric row/column interchange during reduction.
                  const Matrix a = mat({{1, 2, 3}, {0, 4, 5}, {6, 7, 8}});
                  auto h = hessenberg_form(a);
                  t.expect(h.has_value(), "hessenberg_form succeeds with a pivot swap");
                  if (!h) {
                      return;
                  }
                  t.expect(is_upper_hessenberg(*h), "result is upper-Hessenberg after the swap");
                  auto pa = characteristic_polynomial(a);
                  auto ph = characteristic_polynomial(*h);
                  t.expect(pa.has_value() && ph.has_value() && pa->is_equal(*ph),
                           "characteristic polynomial is preserved by the interchange");
              })
        .test("cholesky_exact_for_perfect_square_pivots",
              [](TestContext& t) {
                  // A = [[4,2],[2,5]]: pivots D0 = 4, D1 = 4 are both perfect squares, so
                  // G = L * diag(2, 2) = [[2,0],[1,2]] is exact and G*G^T == A.
                  const Matrix a = mat({{4, 2}, {2, 5}});
                  auto g = cholesky_exact(a);
                  t.expect(g.has_value(), "exact Cholesky exists (perfect-square pivots)");
                  if (!g) {
                      return;
                  }
                  t.expect(g->at(0, 0) == ri(2) && g->at(0, 1).is_zero() &&
                               g->at(1, 0) == ri(1) && g->at(1, 1) == ri(2),
                           "G == [[2,0],[1,2]]");
                  auto gt = g->transpose();
                  t.expect(gt.has_value(), "G transposes");
                  if (!gt) {
                      return;
                  }
                  auto ggt = g->multiply(*gt);
                  t.expect(ggt.has_value() && ggt->is_equal(a), "G*G^T == A exactly");
              })
        .test("cholesky_exact_rejects_irrational_pivot",
              [](TestContext& t) {
                  // A = [[2,1],[1,2]] is SPD but D0 = 2 is not a perfect square: sqrt(2) is
                  // irrational, so no EXACT rational Cholesky exists.
                  auto g = cholesky_exact(mat({{2, 1}, {1, 2}}));
                  t.expect(!g.has_value() && g.error() == MathError::domain_error,
                           "non-perfect-square pivot => domain_error (exact path refuses)");
                  // Symmetric but not positive definite: pivot <= 0.
                  auto g2 = cholesky_exact(mat({{1, 2}, {2, 1}}));
                  t.expect(!g2.has_value() && g2.error() == MathError::domain_error,
                           "indefinite matrix has no Cholesky => domain_error");
                  // Non-symmetric propagates ldlt's domain_error.
                  auto g3 = cholesky_exact(mat({{1, 2}, {3, 4}}));
                  t.expect(!g3.has_value() && g3.error() == MathError::domain_error,
                           "non-symmetric => domain_error");
              })
        .test("diagonal_builder",
              [](TestContext& t) {
                  auto d = diagonal({ri(1), ri(2), ri(3)});
                  t.expect(d.has_value(), "diagonal assembles");
                  if (!d) {
                      return;
                  }
                  t.expect(d->rows() == 3 && d->cols() == 3, "diagonal(3 entries) is 3x3");
                  t.expect(d->at(0, 0) == ri(1) && d->at(1, 1) == ri(2) && d->at(2, 2) == ri(3),
                           "diagonal carries the entries");
                  t.expect(d->at(0, 1).is_zero() && d->at(2, 0).is_zero(),
                           "off-diagonal entries are zero");
                  auto empty = diagonal({});
                  t.expect(empty.has_value() && empty->rows() == 0 && empty->cols() == 0,
                           "diagonal({}) is the 0x0 matrix");
              })
        .test("block_upper_triangular_builder",
              [](TestContext& t) {
                  // Two diagonal blocks D0 (2x2), D1 (1x1) with an off-diagonal (0,1) block U
                  // (2x1). Below the block-diagonal is zero.
                  const Matrix d0 = mat({{1, 2}, {3, 4}});
                  const Matrix u01 = mat({{5}, {6}});
                  const Matrix d1 = mat({{7}});
                  auto but = block_upper_triangular({{d0, u01}, {d1}});
                  t.expect(but.has_value(), "block_upper_triangular assembles");
                  if (!but) {
                      return;
                  }
                  t.expect(but->rows() == 3 && but->cols() == 3, "assembled matrix is 3x3");
                  // Expected: [[1,2,5],[3,4,6],[0,0,7]].
                  const Matrix expected = mat({{1, 2, 5}, {3, 4, 6}, {0, 0, 7}});
                  t.expect(but->is_equal(expected), "blocks are placed in the upper triangle");
                  t.expect(is_block_upper_triangular(*but, {2, 1}),
                           "assembled matrix is block-upper-triangular for sizes (2,1)");
                  // Determinant is the product of the diagonal blocks' determinants:
                  // det(d0) = -2, det(d1) = 7 => -14.
                  auto det = but->determinant();
                  t.expect(det.has_value() && *det == ri(-14),
                           "det == product of diagonal block determinants (-14)");
                  // A shape violation (wrong off-diagonal column count) is rejected.
                  auto bad = block_upper_triangular({{d0, mat({{5, 9}, {6, 9}})}, {d1}});
                  t.expect(!bad.has_value() && bad.error() == MathError::domain_error,
                           "mismatched off-diagonal block => domain_error");
              })
        .test("block_predicates",
              [](TestContext& t) {
                  const Matrix bd = mat({{1, 2, 0, 0}, {3, 4, 0, 0}, {0, 0, 5, 6}, {0, 0, 7, 8}});
                  t.expect(is_block_diagonal(bd, {2, 2}), "2+2 block-diagonal is recognised");
                  t.expect(!is_block_diagonal(bd, {1, 3}),
                           "wrong partition (1,3) breaks block-diagonality");
                  t.expect(is_block_diagonal(bd, {2, 2}) && is_block_upper_triangular(bd, {2, 2}),
                           "block-diagonal is also block-upper-triangular");

                  const Matrix but = mat({{1, 2, 9, 9}, {3, 4, 9, 9}, {0, 0, 5, 6}, {0, 0, 7, 8}});
                  t.expect(is_block_upper_triangular(but, {2, 2}),
                           "2+2 block-upper-triangular is recognised");
                  t.expect(!is_block_diagonal(but, {2, 2}),
                           "nonzero upper block breaks block-diagonality");

                  const Matrix full = mat({{1, 2, 9, 9}, {3, 4, 9, 9}, {1, 0, 5, 6}, {0, 0, 7, 8}});
                  t.expect(!is_block_upper_triangular(full, {2, 2}),
                           "a nonzero strictly-lower block breaks block-upper-triangularity");

                  t.expect(!is_block_diagonal(mat({{1, 2, 3}, {4, 5, 6}}), {1, 1}),
                           "non-square => not block-diagonal");
                  t.expect(!is_block_diagonal(bd, {2, 1}),
                           "partition not summing to n => false");
              })
        .run();
}
