// Tests for nimblecas.kronprod: exact Kronecker product / sum, direct sum, Hadamard
// product, and column-major vec / unvec, plus the mixed-product, transpose and
// vec(A X B) identities on small rational matrices.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.kronprod;
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

[[nodiscard]] auto mat(std::vector<std::vector<Rational>> rows) -> Matrix {
    return Matrix::from_rows(std::move(rows)).value();
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.kronprod")
        .test("kronecker_product_dimensions_and_entries",
              [](TestContext& t) {
                  // A is 2x2, B is 2x3 => A (x) B is 4x6, block (i,j) = a_ij * B.
                  const auto A = mat({{ri(1), ri(2)}, {ri(3), ri(4)}});
                  const auto B = mat({{ri(0), ri(5), ri(1)}, {ri(6), ri(7), ri(2)}});
                  auto K = nimblecas::kronecker_product(A, B).value();
                  t.expect(K.rows() == 4 && K.cols() == 6, "A(x)B is 4x6");
                  // Spot-check the four blocks a_ij * B via representative entries.
                  // Block (0,0) = 1*B: K(1,1) = 1*7 = 7.
                  t.expect(K.at(1, 1) == ri(7), "block(0,0)=1*B: K(1,1)=7");
                  // Block (0,1) = 2*B: K(0,3) = 2*0 = 0, K(1,4) = 2*7 = 14.
                  t.expect(K.at(1, 4) == ri(14), "block(0,1)=2*B: K(1,4)=14");
                  // Block (1,0) = 3*B: K(2,1) = 3*5 = 15.
                  t.expect(K.at(2, 1) == ri(15), "block(1,0)=3*B: K(2,1)=15");
                  // Block (1,1) = 4*B: K(3,5) = 4*2 = 8.
                  t.expect(K.at(3, 5) == ri(8), "block(1,1)=4*B: K(3,5)=8");
              })
        .test("kronecker_product_rational_full_matrix",
              [](TestContext& t) {
                  // Small fully-hand-verified case with fractions.
                  // A = [[1/2, 0],[0, 2]], B = [[1, 3],[ -1, 1]].
                  // A(x)B = [[1/2, 3/2, 0,  0 ],
                  //          [-1/2, 1/2, 0, 0 ],
                  //          [ 0,   0,   2, 6 ],
                  //          [ 0,   0,  -2, 2 ]].
                  const auto A = mat({{rat(1, 2), ri(0)}, {ri(0), ri(2)}});
                  const auto B = mat({{ri(1), ri(3)}, {ri(-1), ri(1)}});
                  auto K = nimblecas::kronecker_product(A, B).value();
                  const auto expected = mat({{rat(1, 2), rat(3, 2), ri(0), ri(0)},
                                             {rat(-1, 2), rat(1, 2), ri(0), ri(0)},
                                             {ri(0), ri(0), ri(2), ri(6)},
                                             {ri(0), ri(0), ri(-2), ri(2)}});
                  t.expect(K.is_equal(expected), "A(x)B matches hand-computed 4x4");
              })
        .test("mixed_product_identity",
              [](TestContext& t) {
                  // (A (x) B)(C (x) D) = (A C) (x) (B D) when the inner dims conform.
                  // A,C are 2x2; B,D are 2x2 => both sides are 4x4.
                  const auto A = mat({{ri(1), ri(2)}, {ri(0), ri(1)}});
                  const auto C = mat({{ri(3), ri(0)}, {ri(1), ri(-1)}});
                  const auto B = mat({{rat(1, 2), ri(1)}, {ri(0), ri(2)}});
                  const auto D = mat({{ri(2), ri(0)}, {ri(1), ri(1)}});
                  auto lhs = nimblecas::kronecker_product(A, B)
                                 .value()
                                 .multiply(nimblecas::kronecker_product(C, D).value())
                                 .value();
                  auto AC = A.multiply(C).value();
                  auto BD = B.multiply(D).value();
                  auto rhs = nimblecas::kronecker_product(AC, BD).value();
                  t.expect(lhs.is_equal(rhs), "(A(x)B)(C(x)D) == (AC)(x)(BD)");
              })
        .test("transpose_identity",
              [](TestContext& t) {
                  // (A (x) B)^T = A^T (x) B^T.
                  const auto A = mat({{ri(1), ri(2), ri(3)}, {ri(4), ri(5), ri(6)}});
                  const auto B = mat({{rat(1, 3), ri(2)}, {ri(-1), ri(0)}});
                  auto lhs = nimblecas::kronecker_product(A, B).value().transpose().value();
                  auto rhs = nimblecas::kronecker_product(A.transpose().value(),
                                                          B.transpose().value())
                                 .value();
                  t.expect(lhs.is_equal(rhs), "(A(x)B)^T == A^T(x)B^T");
              })
        .test("vec_axb_identity",
              [](TestContext& t) {
                  // vec(A X B) = (B^T (x) A) vec(X), column-major vec.
                  // A is 2x2, X is 2x3, B is 3x2 => A X B is 2x2, vec is 4x1.
                  const auto A = mat({{ri(1), ri(2)}, {ri(0), ri(3)}});
                  const auto X = mat({{ri(1), ri(0), ri(2)}, {ri(-1), ri(4), ri(1)}});
                  const auto B = mat({{ri(1), ri(0)}, {ri(2), ri(1)}, {ri(0), ri(3)}});
                  auto AXB = A.multiply(X).value().multiply(B).value();
                  auto lhs = nimblecas::vec(AXB).value();
                  auto Bt = B.transpose().value();
                  auto op = nimblecas::kronecker_product(Bt, A).value();  // (B^T (x) A)
                  auto rhs = op.multiply(nimblecas::vec(X).value()).value();
                  t.expect(lhs.rows() == 4 && lhs.cols() == 1, "vec(AXB) is 4x1");
                  t.expect(lhs.is_equal(rhs), "vec(AXB) == (B^T(x)A)vec(X)");
              })
        .test("vec_unvec_roundtrip_column_major",
              [](TestContext& t) {
                  // vec stacks columns: vec([[1,2,3],[4,5,6]]) = [1,4,2,5,3,6]^T.
                  const auto M = mat({{ri(1), ri(2), ri(3)}, {ri(4), ri(5), ri(6)}});
                  auto v = nimblecas::vec(M).value();
                  t.expect(v.rows() == 6 && v.cols() == 1, "vec is 6x1 column");
                  const auto expected =
                      mat({{ri(1)}, {ri(4)}, {ri(2)}, {ri(5)}, {ri(3)}, {ri(6)}});
                  t.expect(v.is_equal(expected), "column-major stacking [1,4,2,5,3,6]");
                  auto back = nimblecas::unvec(v, 2).value();
                  t.expect(back.is_equal(M), "unvec(vec(M), rows) round-trips");
                  // unvec on a non-column argument is a domain error.
                  t.expect(nimblecas::unvec(M, 2).error() == MathError::domain_error,
                           "unvec of a non-column is domain_error");
                  // A length not divisible by the requested rows is a domain error.
                  t.expect(nimblecas::unvec(v, 4).error() == MathError::domain_error,
                           "length not divisible by rows is domain_error");
              })
        .test("kronecker_sum_definition",
              [](TestContext& t) {
                  // A (+) B = A (x) I_n + I_m (x) B, verified against the direct definition.
                  const auto A = mat({{ri(1), ri(2)}, {ri(3), ri(4)}});   // 2x2
                  const auto B = mat({{ri(0), ri(5)}, {ri(-1), ri(2)}});  // 2x2
                  auto S = nimblecas::kronecker_sum(A, B).value();
                  t.expect(S.rows() == 4 && S.cols() == 4, "A(+)B is 4x4");
                  auto lhs = nimblecas::kronecker_product(A, Matrix::identity(2)).value();
                  auto rhs = nimblecas::kronecker_product(Matrix::identity(2), B).value();
                  auto expected = lhs.add(rhs).value();
                  t.expect(S.is_equal(expected), "A(+)B == A(x)I + I(x)B");
                  // Non-square operands are a domain error.
                  const auto wide = mat({{ri(1), ri(2), ri(3)}});
                  t.expect(nimblecas::kronecker_sum(wide, B).error() == MathError::domain_error,
                           "non-square operand => domain_error");
              })
        .test("direct_sum_block_diagonal",
              [](TestContext& t) {
                  // diag(A, B): A in the top-left, B in the bottom-right, zeros elsewhere.
                  const auto A = mat({{ri(1), ri(2)}, {ri(3), ri(4)}});  // 2x2
                  const auto B = mat({{ri(5)}});                          // 1x1
                  auto D = nimblecas::direct_sum(A, B).value();
                  const auto expected = mat({{ri(1), ri(2), ri(0)},
                                             {ri(3), ri(4), ri(0)},
                                             {ri(0), ri(0), ri(5)}});
                  t.expect(D.rows() == 3 && D.cols() == 3, "diag(A,B) is 3x3");
                  t.expect(D.is_equal(expected), "direct sum is block diagonal");
                  // Rectangular operands: 1x2 (+) 2x1 => 3x3 with off blocks zero.
                  const auto P = mat({{ri(7), ri(8)}});          // 1x2
                  const auto Q = mat({{ri(9)}, {ri(10)}});       // 2x1
                  auto DR = nimblecas::direct_sum(P, Q).value();
                  const auto expectedR = mat({{ri(7), ri(8), ri(0)},
                                              {ri(0), ri(0), ri(9)},
                                              {ri(0), ri(0), ri(10)}});
                  t.expect(DR.is_equal(expectedR), "rectangular direct sum places blocks");
              })
        .test("hadamard_product_and_dimension_mismatch",
              [](TestContext& t) {
                  const auto A = mat({{ri(1), ri(2)}, {ri(3), ri(4)}});
                  const auto B = mat({{rat(1, 2), ri(0)}, {ri(-1), ri(5)}});
                  auto H = nimblecas::hadamard_product(A, B).value();
                  const auto expected = mat({{rat(1, 2), ri(0)}, {ri(-3), ri(20)}});
                  t.expect(H.is_equal(expected), "elementwise product [[1/2,0],[-3,20]]");
                  // Mismatched shapes are a domain error.
                  const auto C = mat({{ri(1), ri(2), ri(3)}});
                  t.expect(nimblecas::hadamard_product(A, C).error() == MathError::domain_error,
                           "shape mismatch => domain_error");
              })
        .test("kronecker_is_not_commutative_but_permutation_similar",
              [](TestContext& t) {
                  // A basic honesty check: A(x)B != B(x)A in general (no silent symmetrisation).
                  const auto A = mat({{ri(1), ri(2)}, {ri(0), ri(1)}});
                  const auto B = mat({{ri(0), ri(1)}, {ri(1), ri(0)}});
                  auto AB = nimblecas::kronecker_product(A, B).value();
                  auto BA = nimblecas::kronecker_product(B, A).value();
                  t.expect(!AB.is_equal(BA), "A(x)B differs from B(x)A for these operands");
                  // But traces agree: tr(A(x)B) = tr(A)tr(B) = tr(B(x)A).
                  t.expect(AB.trace().value() == BA.trace().value(),
                           "tr(A(x)B) == tr(B(x)A) == tr(A)tr(B)");
              })
        .run();
}
