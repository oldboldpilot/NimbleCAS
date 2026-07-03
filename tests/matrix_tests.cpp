// Tests for nimblecas.matrix: exact dense linear algebra over Q (ROADMAP 7.2).
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.matrix;
import nimblecas.ratpoly;
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

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.matrix")
        .test("construction_and_accessors",
              [](TestContext& t) {
                  auto m = mat({{1, 2, 3}, {4, 5, 6}});
                  t.expect(m.rows() == 2 && m.cols() == 3, "2x3 shape");
                  t.expect(!m.is_square(), "2x3 is not square");
                  t.expect(m.at(1, 2) == ri(6), "entry (1,2) is 6");
                  t.expect(Matrix::identity(3).is_square(), "identity is square");

                  // ragged rows are rejected
                  std::vector<std::vector<Rational>> ragged{{ri(1), ri(2)}, {ri(3)}};
                  t.expect(Matrix::from_rows(std::move(ragged)).error() == MathError::domain_error,
                           "ragged rows -> domain_error");
              })
        .test("identity_and_equality",
              [](TestContext& t) {
                  auto id = Matrix::identity(3);
                  t.expect(id.at(0, 0) == ri(1) && id.at(1, 1) == ri(1) && id.at(2, 2) == ri(1),
                           "diagonal is 1");
                  t.expect(id.at(0, 1) == ri(0) && id.at(2, 0) == ri(0), "off-diagonal is 0");
                  t.expect(id == mat({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}), "identity(3) == I_3");

                  // A * I == A
                  auto a = mat({{1, 2, 3}, {4, 5, 6}, {7, 8, 10}});
                  t.expect(a.multiply(Matrix::identity(3)).value() == a, "A * I == A");
              })
        .test("multiply_and_transpose",
              [](TestContext& t) {
                  // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
                  auto prod = mat({{1, 2}, {3, 4}}).multiply(mat({{5, 6}, {7, 8}})).value();
                  t.expect(prod == mat({{19, 22}, {43, 50}}), "2x2 product matches hand result");

                  // transpose of a 2x3
                  auto tr = mat({{1, 2, 3}, {4, 5, 6}}).transpose().value();
                  t.expect(tr == mat({{1, 4}, {2, 5}, {3, 6}}), "transpose is 3x2");

                  // dimension-mismatch multiply -> domain_error (2x3 * 2x2)
                  auto bad = mat({{1, 2, 3}, {4, 5, 6}}).multiply(mat({{1, 2}, {3, 4}}));
                  t.expect(bad.error() == MathError::domain_error, "inner-dim mismatch -> domain_error");
              })
        .test("add_subtract_scale_trace",
              [](TestContext& t) {
                  auto a = mat({{1, 2}, {3, 4}});
                  auto b = mat({{5, 6}, {7, 8}});
                  t.expect(a.add(b).value() == mat({{6, 8}, {10, 12}}), "entrywise add");
                  t.expect(b.subtract(a).value() == mat({{4, 4}, {4, 4}}), "entrywise subtract");
                  t.expect(a.scale(ri(2)).value() == mat({{2, 4}, {6, 8}}), "scale by 2");
                  t.expect(a.trace().value() == ri(5), "trace [[1,2],[3,4]] = 5");
                  t.expect(a.add(mat({{1, 2, 3}})).error() == MathError::domain_error,
                           "add with mismatched shape -> domain_error");
                  t.expect(mat({{1, 2, 3}}).trace().error() == MathError::domain_error,
                           "trace of non-square -> domain_error");
              })
        .test("determinant",
              [](TestContext& t) {
                  // det [[1,2],[3,4]] = 1*4 - 2*3 = -2
                  t.expect(mat({{1, 2}, {3, 4}}).determinant().value() == ri(-2),
                           "det 2x2 = -2");

                  // det [[6,1,1],[4,-2,5],[2,8,7]] = -306 (by hand)
                  t.expect(mat({{6, 1, 1}, {4, -2, 5}, {2, 8, 7}}).determinant().value() == ri(-306),
                           "det 3x3 = -306");

                  // determinant of the identity is 1
                  t.expect(Matrix::identity(4).determinant().value() == ri(1), "det I_4 = 1");

                  // a singular matrix has determinant 0
                  t.expect(mat({{1, 2}, {2, 4}}).determinant().value() == ri(0),
                           "singular det = 0");

                  // non-square determinant -> domain_error
                  t.expect(mat({{1, 2, 3}}).determinant().error() == MathError::domain_error,
                           "non-square det -> domain_error");
              })
        .test("solve_linear_systems",
              [](TestContext& t) {
                  // 2x2: 2x + y = 3, x - y = 0  => x = y = 1
                  auto a2 = mat({{2, 1}, {1, -1}});
                  auto b2 = mat({{3}, {0}});
                  auto x2 = a2.solve(b2).value();
                  t.expect(x2 == mat({{1}, {1}}), "2x2 solution is (1,1)");
                  t.expect(a2.multiply(x2).value() == b2, "A*x == b (2x2)");

                  // 3x3 classic system with solution (2, 3, -1)
                  auto a3 = mat({{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}});
                  auto b3 = mat({{8}, {-11}, {-3}});
                  auto x3 = a3.solve(b3).value();
                  t.expect(x3 == mat({{2}, {3}, {-1}}), "3x3 solution is (2,3,-1)");
                  t.expect(a3.multiply(x3).value() == b3, "A*x == b (3x3)");

                  // singular system -> domain_error
                  auto sing = mat({{1, 2}, {2, 4}});
                  t.expect(sing.solve(mat({{1}, {2}})).error() == MathError::domain_error,
                           "singular solve -> domain_error");
                  // mismatched row counts -> domain_error
                  t.expect(a2.solve(mat({{1}, {2}, {3}})).error() == MathError::domain_error,
                           "row-count mismatch -> domain_error");
              })
        .test("inverse",
              [](TestContext& t) {
                  // inverse of [[1,2],[3,4]] is [[-2,1],[3/2,-1/2]]
                  auto a = mat({{1, 2}, {3, 4}});
                  auto inv = a.inverse().value();
                  auto expected = Matrix::from_rows(
                                      {{rat(-2, 1), rat(1, 1)}, {rat(3, 2), rat(-1, 2)}})
                                      .value();
                  t.expect(inv == expected, "explicit 2x2 inverse");
                  t.expect(a.multiply(inv).value() == Matrix::identity(2), "A * A^-1 == I");
                  t.expect(inv.multiply(a).value() == Matrix::identity(2), "A^-1 * A == I");

                  // 3x3 inverse verified against identity
                  auto b = mat({{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}});
                  t.expect(b.multiply(b.inverse().value()).value() == Matrix::identity(3),
                           "3x3 A * A^-1 == I");

                  // singular inverse -> domain_error
                  t.expect(mat({{1, 2}, {2, 4}}).inverse().error() == MathError::domain_error,
                           "singular inverse -> domain_error");
                  // non-square inverse -> domain_error
                  t.expect(mat({{1, 2, 3}}).inverse().error() == MathError::domain_error,
                           "non-square inverse -> domain_error");
              })
        .test("rank",
              [](TestContext& t) {
                  t.expect(Matrix::identity(3).rank() == 3, "rank of I_3 is 3");
                  t.expect(mat({{1, 2}, {2, 4}}).rank() == 1, "rank of a rank-1 matrix is 1");
                  t.expect(mat({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}).rank() == 2,
                           "rank of [[1..9]] is 2");
                  t.expect(Matrix::zero(2, 3).rank() == 0, "rank of the zero matrix is 0");
              })
        .test("overflow_is_reported",
              [](TestContext& t) {
                  const std::int64_t big = std::numeric_limits<std::int64_t>::max();
                  auto a = Matrix::from_rows({{ri(big), ri(big)}, {ri(big), ri(big)}}).value();
                  // big + big overflows int64 during the entrywise sum.
                  t.expect(a.add(a).error() == MathError::overflow,
                           "int64 overflow surfaces as MathError::overflow");
              })
        .run();
}
