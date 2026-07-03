// Tests for nimblecas.bandsolve: exact banded & tridiagonal linear-system solvers (ROADMAP 7.2).
// @author Olumuyiwa Oluwasanmi
//
// Every system is built from exact rationals with a hand-verified solution, so each assertion
// checks a concrete value or an exact A*x == b substitution — nothing is "approximately" right.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.bandsolve;
import nimblecas.testing;

using nimblecas::MathError;
using nimblecas::Matrix;
using nimblecas::Rational;
using nimblecas::solve_banded;
using nimblecas::solve_tridiagonal;
using nimblecas::solve_tridiagonal_batch;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto ri(std::int64_t v) -> Rational {
    return Rational::from_int(v);
}

[[nodiscard]] auto rq(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

// A vector<Rational> from integer entries.
[[nodiscard]] auto vec(std::vector<std::int64_t> entries) -> std::vector<Rational> {
    std::vector<Rational> v;
    v.reserve(entries.size());
    for (const std::int64_t e : entries) {
        v.push_back(Rational::from_int(e));
    }
    return v;
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

// An n x 1 column matrix from integer entries.
[[nodiscard]] auto col(std::vector<std::int64_t> entries) -> Matrix {
    std::vector<std::vector<std::int64_t>> rows;
    rows.reserve(entries.size());
    for (const std::int64_t v : entries) {
        rows.push_back({v});
    }
    return mat(std::move(rows));
}

// The 3x3 discrete-Laplacian operator tridiag(-1, 2, -1) as (sub, diag, super) and as a dense
// matrix, so tridiagonal results can be cross-checked against a full A*x product.
const std::vector<Rational> kLapSub = vec({-1, -1});
const std::vector<Rational> kLapDiag = vec({2, 2, 2});
const std::vector<Rational> kLapSuper = vec({-1, -1});

[[nodiscard]] auto laplacian_dense() -> Matrix {
    return mat({{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}});
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.bandsolve")
        .test("tridiagonal_known_solution",
              [](TestContext& t) {
                  // tridiag(-1, 2, -1) x = [1, 1, 1] has the exact solution [3/2, 2, 3/2]:
                  //   2*(3/2) - 2         = 1
                  //   -(3/2) + 2*2 - (3/2) = 1
                  //   -2 + 2*(3/2)        = 1
                  auto x = solve_tridiagonal(kLapSub, kLapDiag, kLapSuper, vec({1, 1, 1}));
                  t.expect(x.has_value(), "Thomas solve succeeds");
                  if (x) {
                      t.expect(x->size() == 3, "solution has 3 entries");
                      t.expect((*x)[0] == rq(3, 2) && (*x)[1] == ri(2) && (*x)[2] == rq(3, 2),
                               "x = [3/2, 2, 3/2]");
                  }
              })
        .test("tridiagonal_substitution_check",
              [](TestContext& t) {
                  // Independently confirm A*x == b via a dense matrix product.
                  auto x = solve_tridiagonal(kLapSub, kLapDiag, kLapSuper, vec({1, 1, 1}));
                  t.expect(x.has_value(), "solve succeeds");
                  if (x) {
                      std::vector<std::vector<Rational>> rows;
                      for (const Rational& xi : *x) {
                          rows.push_back({xi});
                      }
                      auto xm = Matrix::from_rows(std::move(rows));
                      t.expect(xm.has_value(), "solution packs into a column");
                      if (xm) {
                          auto ax = laplacian_dense().multiply(*xm);
                          t.expect(ax.has_value(), "A*x forms");
                          t.expect(ax && ax->is_equal(col({1, 1, 1})), "A*x == b exactly");
                      }
                  }
              })
        .test("tridiagonal_size_mismatches",
              [](TestContext& t) {
                  // sub must have n-1 entries (here n = 3, so 2).
                  auto bad_sub = solve_tridiagonal(vec({-1, -1, -1}), kLapDiag, kLapSuper,
                                                   vec({1, 1, 1}));
                  t.expect(!bad_sub.has_value() && bad_sub.error() == MathError::domain_error,
                           "wrong sub length => domain_error");
                  // super must have n-1 entries.
                  auto bad_super = solve_tridiagonal(kLapSub, kLapDiag, vec({-1}), vec({1, 1, 1}));
                  t.expect(!bad_super.has_value() && bad_super.error() == MathError::domain_error,
                           "wrong super length => domain_error");
                  // rhs must have n entries.
                  auto bad_rhs = solve_tridiagonal(kLapSub, kLapDiag, kLapSuper, vec({1, 1}));
                  t.expect(!bad_rhs.has_value() && bad_rhs.error() == MathError::domain_error,
                           "wrong rhs length => domain_error");
                  // Empty diagonal (n == 0) is rejected.
                  auto empty = solve_tridiagonal({}, {}, {}, {});
                  t.expect(!empty.has_value() && empty.error() == MathError::domain_error,
                           "empty system => domain_error");
              })
        .test("tridiagonal_zero_pivot_singular",
              [](TestContext& t) {
                  // A = [[1, 1], [1, 1]] is singular: elimination yields a zero second pivot.
                  auto x = solve_tridiagonal(vec({1}), vec({1, 1}), vec({1}), vec({1, 2}));
                  t.expect(!x.has_value(), "singular tridiagonal is rejected");
                  t.expect(x.error() == MathError::domain_error, "zero pivot => domain_error");
              })
        .test("tridiagonal_leading_zero_pivot",
              [](TestContext& t) {
                  // A(0,0) == 0 with no pivoting => domain_error immediately.
                  auto x = solve_tridiagonal(vec({1}), vec({0, 1}), vec({1}), vec({1, 1}));
                  t.expect(!x.has_value() && x.error() == MathError::domain_error,
                           "zero leading diagonal => domain_error");
              })
        .test("batch_agrees_with_per_column_thomas",
              [](TestContext& t) {
                  // Two right-hand sides against the same operator.
                  //   rhs [1,1,1] -> [3/2, 2, 3/2]   (verified above)
                  //   rhs [0,2,0] -> [1, 2, 1]:  2*1-2=0, -1+4-1=2, -2+2*1... = 0
                  auto rhs_cols = mat({{1, 0}, {1, 2}, {1, 0}});
                  auto batch = solve_tridiagonal_batch(kLapSub, kLapDiag, kLapSuper, rhs_cols);
                  t.expect(batch.has_value(), "batch solve succeeds");
                  if (batch) {
                      t.expect(batch->rows() == 3 && batch->cols() == 2, "solution is 3x2");
                      // Column 0 matches the known single-RHS solution.
                      t.expect(batch->at(0, 0) == rq(3, 2) && batch->at(1, 0) == ri(2) &&
                                   batch->at(2, 0) == rq(3, 2),
                               "batch column 0 = [3/2, 2, 3/2]");
                      // Column 1 matches its hand solution.
                      t.expect(batch->at(0, 1) == ri(1) && batch->at(1, 1) == ri(2) &&
                                   batch->at(2, 1) == ri(1),
                               "batch column 1 = [1, 2, 1]");

                      // Cross-check every column against an independent per-column Thomas solve.
                      auto c0 = solve_tridiagonal(kLapSub, kLapDiag, kLapSuper, vec({1, 1, 1}));
                      auto c1 = solve_tridiagonal(kLapSub, kLapDiag, kLapSuper, vec({0, 2, 0}));
                      t.expect(c0.has_value() && c1.has_value(), "per-column solves succeed");
                      if (c0 && c1) {
                          bool same = true;
                          for (std::size_t i = 0; i < 3; ++i) {
                              same = same && batch->at(i, 0) == (*c0)[i] &&
                                     batch->at(i, 1) == (*c1)[i];
                          }
                          t.expect(same, "batch equals per-column Thomas for every entry");
                      }
                  }
              })
        .test("batch_size_mismatch",
              [](TestContext& t) {
                  // rhs_columns must have n (== 3) rows.
                  auto bad = solve_tridiagonal_batch(kLapSub, kLapDiag, kLapSuper,
                                                     mat({{1, 0}, {1, 2}}));
                  t.expect(!bad.has_value() && bad.error() == MathError::domain_error,
                           "rhs with wrong row count => domain_error");
              })
        .test("banded_bandwidth1_matches_tridiagonal",
              [](TestContext& t) {
                  // Bandwidth-1 band-LU on the dense Laplacian must reproduce Thomas exactly.
                  auto banded = solve_banded(laplacian_dense(), 1, 1, col({1, 1, 1}));
                  auto thomas = solve_tridiagonal(kLapSub, kLapDiag, kLapSuper, vec({1, 1, 1}));
                  t.expect(banded.has_value(), "banded solve succeeds");
                  t.expect(thomas.has_value(), "Thomas solve succeeds");
                  if (banded && thomas) {
                      t.expect(banded->rows() == 3 && banded->cols() == 1, "solution is 3x1");
                      bool same = true;
                      for (std::size_t i = 0; i < 3; ++i) {
                          same = same && banded->at(i, 0) == (*thomas)[i];
                      }
                      t.expect(same, "bandwidth-1 band-LU == Thomas");
                      t.expect(banded->at(0, 0) == rq(3, 2) && banded->at(1, 0) == ri(2) &&
                                   banded->at(2, 0) == rq(3, 2),
                               "solution = [3/2, 2, 3/2]");
                  }
              })
        .test("banded_bandwidth2_substitution",
              [](TestContext& t) {
                  // A symmetric, diagonally dominant bandwidth-2 (pentadiagonal) system.
                  //   [4 1 1 0]
                  //   [1 4 1 1]
                  //   [1 1 4 1]
                  //   [0 1 1 4]
                  // Diagonal dominance guarantees nonzero pivots without permutation. The
                  // solution is verified by the exact substitution A*x == b.
                  auto a = mat({{4, 1, 1, 0}, {1, 4, 1, 1}, {1, 1, 4, 1}, {0, 1, 1, 4}});
                  auto b = col({1, 2, 3, 4});
                  auto x = solve_banded(a, 2, 2, b);
                  t.expect(x.has_value(), "bandwidth-2 band-LU succeeds");
                  if (x) {
                      t.expect(x->rows() == 4 && x->cols() == 1, "solution is 4x1");
                      auto ax = a.multiply(*x);
                      t.expect(ax.has_value(), "A*x forms");
                      t.expect(ax && ax->is_equal(b), "A*x == b exactly");
                  }
              })
        .test("banded_shape_errors",
              [](TestContext& t) {
                  // Non-square a.
                  auto ns = solve_banded(mat({{1, 2, 3}, {4, 5, 6}}), 1, 1, col({1, 1}));
                  t.expect(!ns.has_value() && ns.error() == MathError::domain_error,
                           "non-square a => domain_error");
                  // b with the wrong number of rows.
                  auto badrows = solve_banded(laplacian_dense(), 1, 1, col({1, 1}));
                  t.expect(!badrows.has_value() && badrows.error() == MathError::domain_error,
                           "b with wrong row count => domain_error");
                  // b that is not a single column.
                  auto badcols =
                      solve_banded(mat({{2, 0}, {0, 2}}), 1, 1, mat({{1, 2}, {3, 4}}));
                  t.expect(!badcols.has_value() && badcols.error() == MathError::domain_error,
                           "b that is not a column => domain_error");
              })
        .test("banded_zero_pivot_singular",
              [](TestContext& t) {
                  // A = [[0, 1], [1, 0]] has a zero leading pivot and, without pivoting, is
                  // rejected as domain_error (the band-LU does not permute rows).
                  auto x = solve_banded(mat({{0, 1}, {1, 0}}), 1, 1, col({1, 1}));
                  t.expect(!x.has_value() && x.error() == MathError::domain_error,
                           "zero pivot => domain_error");
              })
        .run();
}
