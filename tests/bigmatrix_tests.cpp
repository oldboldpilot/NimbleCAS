// Tests for nimblecas.bigmatrix: exact dense linear algebra over Q with an unbounded,
// overflow-free determinant (the big-backed counterpart to nimblecas.matrix).
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.bigmatrix;
import nimblecas.testing;

using nimblecas::BigInt;
using nimblecas::BigMatrix;
using nimblecas::BigRational;
using nimblecas::MathError;
using nimblecas::Matrix;
using nimblecas::Rational;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Exact BigRational fraction n/d (denominator known non-zero at each call site).
[[nodiscard]] auto brat(std::int64_t n, std::int64_t d) -> BigRational {
    return BigRational::make(BigInt::from_i64(n), BigInt::from_i64(d)).value();
}

// The integer n lifted into Q as n/1.
[[nodiscard]] auto bi(std::int64_t v) -> BigRational {
    return BigRational::from_int(v);
}

// Build a BigMatrix from integer rows (low-index row first).
[[nodiscard]] auto bmat(std::vector<std::vector<std::int64_t>> rows) -> BigMatrix {
    std::vector<std::vector<BigRational>> r;
    r.reserve(rows.size());
    for (const auto& row : rows) {
        std::vector<BigRational> rr;
        rr.reserve(row.size());
        for (const std::int64_t v : row) {
            rr.push_back(BigRational::from_int(v));
        }
        r.push_back(std::move(rr));
    }
    return BigMatrix::from_rows(std::move(r)).value();
}

// Build an int64-Rational Matrix from integer rows (for promotion / cross-check tests).
[[nodiscard]] auto imat(std::vector<std::vector<std::int64_t>> rows) -> Matrix {
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

// Promote an int64 Rational into a BigRational the same way BigMatrix::from_matrix does.
[[nodiscard]] auto promote(const Rational& r) -> BigRational {
    return BigRational::make(BigInt::from_i64(r.numerator()), BigInt::from_i64(r.denominator()))
        .value();
}

// 10^p as an exact BigRational integer.
[[nodiscard]] auto pow10(std::uint64_t p) -> BigRational {
    return BigRational::from_bigint(BigInt::from_u64(10).pow(p));
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.bigmatrix")
        .test("construction_and_accessors",
              [](TestContext& t) {
                  auto m = bmat({{1, 2, 3}, {4, 5, 6}});
                  t.expect(m.rows() == 2 && m.cols() == 3, "2x3 shape");
                  t.expect(!m.is_square(), "2x3 is not square");
                  t.expect(m.at(1, 2) == bi(6), "entry (1,2) is 6");
                  t.expect(BigMatrix::identity(3).is_square(), "identity is square");

                  // ragged rows are rejected
                  std::vector<std::vector<BigRational>> ragged{{bi(1), bi(2)}, {bi(3)}};
                  t.expect(BigMatrix::from_rows(std::move(ragged)).error() ==
                               MathError::domain_error,
                           "ragged rows -> domain_error");

                  // an empty row list is rejected (unlike the empty-tolerant int64 Matrix)
                  t.expect(BigMatrix::from_rows({}).error() == MathError::domain_error,
                           "empty row list -> domain_error");
              })
        .test("identity_zero_equality",
              [](TestContext& t) {
                  auto id = BigMatrix::identity(3);
                  t.expect(id.at(0, 0) == bi(1) && id.at(1, 1) == bi(1) && id.at(2, 2) == bi(1),
                           "diagonal is 1");
                  t.expect(id.at(0, 1) == bi(0) && id.at(2, 0) == bi(0), "off-diagonal is 0");
                  t.expect(id == bmat({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}), "identity(3) == I_3");
                  t.expect(id.is_equal(bmat({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}})), "is_equal agrees");

                  auto z = BigMatrix::zero(2, 3);
                  t.expect(z.rows() == 2 && z.cols() == 3, "zero(2,3) shape");
                  t.expect(z == bmat({{0, 0, 0}, {0, 0, 0}}), "zero matrix entries");

                  // A * I == A
                  auto a = bmat({{1, 2, 3}, {4, 5, 6}, {7, 8, 10}});
                  t.expect(a.multiply(BigMatrix::identity(3)).value() == a, "A * I == A");
              })
        .test("arithmetic_add_subtract_scale",
              [](TestContext& t) {
                  auto a = bmat({{1, 2}, {3, 4}});
                  auto b = bmat({{5, 6}, {7, 8}});
                  t.expect(a.add(b).value() == bmat({{6, 8}, {10, 12}}), "entrywise add");
                  t.expect(b.subtract(a).value() == bmat({{4, 4}, {4, 4}}), "entrywise subtract");
                  t.expect(a.scale(bi(2)).value() == bmat({{2, 4}, {6, 8}}), "scale by 2");
                  // scaling by 1/2 stays exact
                  auto half = a.scale(brat(1, 2)).value();
                  t.expect(half.at(0, 0) == brat(1, 2) && half.at(1, 1) == bi(2), "scale by 1/2");

                  t.expect(a.add(bmat({{1, 2, 3}})).error() == MathError::domain_error,
                           "add with mismatched shape -> domain_error");
                  t.expect(a.subtract(bmat({{1, 2, 3}})).error() == MathError::domain_error,
                           "subtract with mismatched shape -> domain_error");
              })
        .test("multiply_and_transpose",
              [](TestContext& t) {
                  // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
                  auto prod = bmat({{1, 2}, {3, 4}}).multiply(bmat({{5, 6}, {7, 8}})).value();
                  t.expect(prod == bmat({{19, 22}, {43, 50}}), "2x2 product matches hand result");

                  // transpose of a 2x3
                  auto tr = bmat({{1, 2, 3}, {4, 5, 6}}).transpose().value();
                  t.expect(tr == bmat({{1, 4}, {2, 5}, {3, 6}}), "transpose is 3x2");

                  // dimension-mismatch multiply -> domain_error (2x3 * 2x2)
                  auto bad = bmat({{1, 2, 3}, {4, 5, 6}}).multiply(bmat({{1, 2}, {3, 4}}));
                  t.expect(bad.error() == MathError::domain_error,
                           "inner-dim mismatch -> domain_error");
              })
        .test("determinant_small_and_edge_cases",
              [](TestContext& t) {
                  // det [[1,2],[3,4]] = 1*4 - 2*3 = -2
                  t.expect(bmat({{1, 2}, {3, 4}}).determinant().value() == bi(-2), "det 2x2 = -2");

                  // det [[6,1,1],[4,-2,5],[2,8,7]] = -306 (by hand)
                  t.expect(bmat({{6, 1, 1}, {4, -2, 5}, {2, 8, 7}}).determinant().value() ==
                               bi(-306),
                           "det 3x3 = -306");

                  // determinant of the identity is 1
                  t.expect(BigMatrix::identity(4).determinant().value() == bi(1), "det I_4 = 1");

                  // a singular matrix (two equal rows) has determinant 0
                  t.expect(bmat({{1, 2, 3}, {1, 2, 3}, {4, 5, 6}}).determinant().value() == bi(0),
                           "two equal rows -> det 0");
                  t.expect(bmat({{1, 2}, {2, 4}}).determinant().value() == bi(0),
                           "rank-1 2x2 -> det 0");

                  // the 0x0 determinant is the empty product 1
                  t.expect(BigMatrix::identity(0).determinant().value() == bi(1), "det of 0x0 = 1");

                  // a 1x1 determinant is the sole entry
                  t.expect(bmat({{7}}).determinant().value() == bi(7), "det 1x1 = entry");

                  // non-square determinant -> domain_error
                  t.expect(bmat({{1, 2, 3}}).determinant().error() == MathError::domain_error,
                           "non-square det -> domain_error");
              })
        .test("determinant_needs_row_swap",
              [](TestContext& t) {
                  // A zero leading pivot forces the Bareiss row swap (and a sign flip).
                  // det [[0,1],[1,0]] = 0*0 - 1*1 = -1
                  t.expect(bmat({{0, 1}, {1, 0}}).determinant().value() == bi(-1),
                           "det [[0,1],[1,0]] = -1 (row swap)");
                  // det [[0,2,1],[1,0,0],[0,1,3]] = -5 (hand-checked); needs a swap at col 0
                  t.expect(bmat({{0, 2, 1}, {1, 0, 0}, {0, 1, 3}}).determinant().value() == bi(-5),
                           "det 3x3 with leading-zero pivot = -5");
              })
        .test("determinant_cross_check_int64_matrix",
              [](TestContext& t) {
                  // For small entries both tiers agree: promote the int64 Matrix and compare
                  // the BigMatrix determinant against the promoted Rational determinant.
                  auto a = imat({{1, 2}, {3, 4}});
                  auto big = BigMatrix::from_matrix(a).value();
                  t.expect(big.determinant().value() == promote(a.determinant().value()),
                           "2x2 det agrees with int64 Matrix");

                  auto b = imat({{6, 1, 1}, {4, -2, 5}, {2, 8, 7}});
                  auto bigb = BigMatrix::from_matrix(b).value();
                  t.expect(bigb.determinant().value() == promote(b.determinant().value()),
                           "3x3 det agrees with int64 Matrix");
                  t.expect(bigb.determinant().value() == bi(-306), "cross-checked value is -306");
              })
        .test("determinant_overflows_int64_diagonal",
              [](TestContext& t) {
                  // diag(10^10, 10^10, 10^10): det = 10^30, far beyond the ~9.2e18 int64
                  // ceiling. The int64 Matrix::determinant would overflow here; BigMatrix is
                  // exact.
                  const std::int64_t k = 10000000000LL;  // 10^10 (fits int64)
                  auto d = bmat({{k, 0, 0}, {0, k, 0}, {0, 0, k}});
                  t.expect(d.determinant().value() == pow10(30), "det diag(10^10)^3 = 10^30");

                  // The int64 tier really does overflow on the same matrix: even a 2x2
                  // diag(10^10, 10^10) has det 10^20 > int64 max, surfacing as overflow.
                  auto small = imat({{k, 0}, {0, k}});
                  t.expect(small.determinant().error() == MathError::overflow,
                           "int64 Matrix det overflows on diag(10^10, 10^10)");
                  // BigMatrix computes the same 2x2 exactly: 10^20.
                  t.expect(bmat({{k, 0}, {0, k}}).determinant().value() == pow10(20),
                           "BigMatrix det diag(10^10)^2 = 10^20 (exact)");
              })
        .test("determinant_overflows_int64_dense_4x4",
              [](TestContext& t) {
                  // A dense 4x4 Vandermonde on nodes {10, 20, 30, 40}. Its determinant is the
                  // product of pairwise differences:
                  //   prod_{i<j}(x_j - x_i)
                  //     = (20-10)(30-10)(40-10)(30-20)(40-20)(40-30)
                  //     = 10 * 20 * 30 * 10 * 20 * 10 = 12,000,000.
                  // (Small enough to hand-check, but its intermediate Bareiss products still
                  // exceed what a careless int64 path would hold; here we assert the exact
                  // value.)
                  auto v = bmat({{1, 10, 100, 1000},
                                 {1, 20, 400, 8000},
                                 {1, 30, 900, 27000},
                                 {1, 40, 1600, 64000}});
                  t.expect(v.determinant().value() == bi(12000000), "4x4 Vandermonde det");

                  // A diagonal 4x4 with 10^6 entries: det = 10^24, well past int64.
                  const std::int64_t m = 1000000LL;  // 10^6
                  auto big = bmat({{m, 0, 0, 0}, {0, m, 0, 0}, {0, 0, m, 0}, {0, 0, 0, m}});
                  t.expect(big.determinant().value() == pow10(24), "det diag(10^6)^4 = 10^24");
              })
        .test("determinant_multiplicativity_and_transpose",
              [](TestContext& t) {
                  // det(A*B) == det(A)*det(B) on a concrete pair.
                  auto a = bmat({{1, 2}, {3, 4}});     // det -2
                  auto b = bmat({{5, 6}, {7, 8}});     // det -2
                  auto da = a.determinant().value();
                  auto db = b.determinant().value();
                  auto dab = a.multiply(b).value().determinant().value();
                  t.expect(dab == da.multiply(db), "det(A*B) == det(A)*det(B)");
                  t.expect(dab == bi(4), "det(A*B) == 4 (hand-checked)");

                  // det(transpose) == det on a 3x3.
                  auto c = bmat({{6, 1, 1}, {4, -2, 5}, {2, 8, 7}});
                  t.expect(c.transpose().value().determinant().value() == c.determinant().value(),
                           "det(A^T) == det(A)");
              })
        .test("determinant_rational_entries",
              [](TestContext& t) {
                  // [[1/2, 1/3],[1/4, 1/5]]: det = 1/2*1/5 - 1/3*1/4 = 1/10 - 1/12 = 1/60.
                  auto m = BigMatrix::from_rows({{brat(1, 2), brat(1, 3)},
                                                 {brat(1, 4), brat(1, 5)}})
                               .value();
                  t.expect(m.determinant().value() == brat(1, 60),
                           "det of a 1/2,1/3,1/4,1/5 matrix = 1/60");

                  // [[2/3, 1/6],[1/2, 3/4]]: det = 2/3*3/4 - 1/6*1/2 = 1/2 - 1/12 = 5/12.
                  auto m2 = BigMatrix::from_rows({{brat(2, 3), brat(1, 6)},
                                                  {brat(1, 2), brat(3, 4)}})
                                .value();
                  t.expect(m2.determinant().value() == brat(5, 12), "det = 5/12 (exact fraction)");
              })
        .test("from_matrix_promotion_round_trip",
              [](TestContext& t) {
                  // Integer + fractional entries survive promotion value-for-value.
                  auto a = Matrix::from_rows({{Rational::from_int(3), Rational::make(1, 2).value()},
                                              {Rational::make(-5, 4).value(), Rational::from_int(0)}})
                               .value();
                  auto big = BigMatrix::from_matrix(a).value();
                  t.expect(big.rows() == 2 && big.cols() == 2, "promotion preserves shape");
                  t.expect(big.at(0, 0) == bi(3), "entry 3 round-trips");
                  t.expect(big.at(0, 1) == brat(1, 2), "entry 1/2 round-trips");
                  t.expect(big.at(1, 0) == brat(-5, 4), "entry -5/4 round-trips");
                  t.expect(big.at(1, 1) == bi(0), "entry 0 round-trips");

                  // A non-square Matrix promotes fine (from_matrix imposes no square rule).
                  auto wide = BigMatrix::from_matrix(imat({{1, 2, 3}})).value();
                  t.expect(wide.rows() == 1 && wide.cols() == 3, "1x3 promotion shape");
              })
        .run();
}
