// Tests for nimblecas.cmatrix: complex (Gaussian-rational) matrices (ROADMAP 7.2).
// @author Olumuyiwa Oluwasanmi
//
// Every entry is an exact Gaussian integer (real and imaginary parts are integers), so
// adjoints, products, and the Hermitian/unitary/normal predicates are exact and
// deterministic. The Pauli matrices are the ideal fixtures: all their entries lie in
// {0, ±1, ±i}, so the classic quantum-gate identities (XᴴX = I, XY = iZ, ...) stay in
// the Gaussian integers with no 1/sqrt(2) ever appearing.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.complex;
import nimblecas.cmatrix;
import nimblecas.testing;

using nimblecas::Complex;
using nimblecas::ComplexMatrix;
using nimblecas::MathError;
using nimblecas::Rational;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// A Gaussian-integer entry re + im*i.
[[nodiscard]] auto c(std::int64_t re, std::int64_t im) -> Complex {
    return Complex::make(Rational::from_int(re), Rational::from_int(im));
}

// Assemble a ComplexMatrix from rows of Complex entries (unwrapping the Result).
[[nodiscard]] auto cm(std::vector<std::vector<Complex>> rows) -> ComplexMatrix {
    return ComplexMatrix::from_rows(std::move(rows)).value();
}

// --- fixtures: the Pauli matrices and friends ---
// X = [[0,1],[1,0]], Y = [[0,-i],[i,0]], Z = [[1,0],[0,-1]].
[[nodiscard]] auto pauli_x() -> ComplexMatrix { return cm({{c(0, 0), c(1, 0)}, {c(1, 0), c(0, 0)}}); }
[[nodiscard]] auto pauli_y() -> ComplexMatrix { return cm({{c(0, 0), c(0, -1)}, {c(0, 1), c(0, 0)}}); }
[[nodiscard]] auto pauli_z() -> ComplexMatrix { return cm({{c(1, 0), c(0, 0)}, {c(0, 0), c(-1, 0)}}); }

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.cmatrix")
        // --- adjoint / conjugate / transpose ---
        .test("adjoint_conjugate_transpose",
              [](TestContext& t) {
                  // A = [[1+i, 2], [-i, 3]].
                  // conjugate  -> [[1-i, 2], [i, 3]]  (entrywise, same shape)
                  // transpose  -> [[1+i, -i], [2, 3]] (positions swap, no conjugation)
                  // adjoint    -> conjugate then transpose = [[1-i, i], [2, 3]]
                  const auto a = cm({{c(1, 1), c(2, 0)}, {c(0, -1), c(3, 0)}});

                  auto conj = a.conjugate();
                  t.expect(conj.has_value(), "conjugate succeeds");
                  if (conj) {
                      t.expect(conj->at(0, 0) == c(1, -1) && conj->at(0, 1) == c(2, 0) &&
                                   conj->at(1, 0) == c(0, 1) && conj->at(1, 1) == c(3, 0),
                               "conjugate is entrywise, shape preserved");
                  }

                  auto tr = a.transpose();
                  t.expect(tr.has_value(), "transpose succeeds");
                  if (tr) {
                      t.expect(tr->at(0, 0) == c(1, 1) && tr->at(0, 1) == c(0, -1) &&
                                   tr->at(1, 0) == c(2, 0) && tr->at(1, 1) == c(3, 0),
                               "transpose swaps positions without conjugating");
                  }

                  auto adj = a.adjoint();
                  t.expect(adj.has_value(), "adjoint succeeds");
                  if (adj) {
                      t.expect(adj->at(0, 0) == c(1, -1) && adj->at(0, 1) == c(0, 1) &&
                                   adj->at(1, 0) == c(2, 0) && adj->at(1, 1) == c(3, 0),
                               "adjoint = conjugate transpose [[1-i,i],[2,3]]");
                  }
              })
        .test("adjoint_non_square_shape",
              [](TestContext& t) {
                  // Aᴴ of a 2x3 is 3x2; conjugation is entrywise so purely-real entries
                  // are unchanged, only positions transpose.
                  const auto a = cm({{c(1, 1), c(2, -1), c(0, 3)}, {c(4, 0), c(0, -5), c(6, 1)}});
                  auto adj = a.adjoint();
                  t.expect(adj.has_value(), "adjoint of 2x3 succeeds");
                  if (adj) {
                      t.expect(adj->rows() == 3 && adj->cols() == 2, "adjoint is 3x2");
                      t.expect(adj->at(0, 0) == c(1, -1) && adj->at(2, 1) == c(6, -1),
                               "corner entries conjugated and transposed");
                  }
              })
        // --- Pauli X (real) ---
        .test("pauli_x_hermitian_unitary_normal",
              [](TestContext& t) {
                  const auto x = pauli_x();
                  auto herm = x.is_hermitian();
                  auto uni = x.is_unitary();
                  auto norm = x.is_normal();
                  t.expect(herm.has_value() && herm.value_or(false), "X is Hermitian");
                  t.expect(uni.has_value() && uni.value_or(false), "X is unitary (XᴴX = I)");
                  t.expect(norm.has_value() && norm.value_or(false), "X is normal");
              })
        // --- Pauli Y (genuinely complex) ---
        .test("pauli_y_hermitian_unitary_normal",
              [](TestContext& t) {
                  const auto y = pauli_y();
                  // Yᴴ = [[0,-i],[i,0]] = Y, so Hermitian despite the imaginary entries.
                  auto adj = y.adjoint();
                  t.expect(adj.has_value() && adj.value_or(pauli_x()) == y, "Yᴴ == Y");
                  auto herm = y.is_hermitian();
                  auto uni = y.is_unitary();
                  auto norm = y.is_normal();
                  t.expect(herm.has_value() && herm.value_or(false), "Y is Hermitian");
                  t.expect(uni.has_value() && uni.value_or(false), "Y is unitary");
                  t.expect(norm.has_value() && norm.value_or(false), "Y is normal");
              })
        // --- Pauli Z ---
        .test("pauli_z_hermitian_unitary",
              [](TestContext& t) {
                  const auto z = pauli_z();
                  auto herm = z.is_hermitian();
                  auto uni = z.is_unitary();
                  t.expect(herm.has_value() && herm.value_or(false), "Z is Hermitian");
                  t.expect(uni.has_value() && uni.value_or(false), "Z is unitary");
              })
        // --- skew-Hermitian: iX = [[0,i],[i,0]] ---
        .test("skew_hermitian_iX",
              [](TestContext& t) {
                  // iX has adjoint [[0,-i],[-i,0]] = -(iX), so it is skew-Hermitian and
                  // not Hermitian.
                  const auto ix = cm({{c(0, 0), c(0, 1)}, {c(0, 1), c(0, 0)}});
                  auto skew = ix.is_skew_hermitian();
                  auto herm = ix.is_hermitian();
                  t.expect(skew.has_value() && skew.value_or(false), "iX is skew-Hermitian");
                  t.expect(herm.has_value() && !herm.value_or(true), "iX is not Hermitian");
              })
        // --- a real shear: neither unitary nor Hermitian ---
        .test("shear_not_unitary_not_hermitian",
              [](TestContext& t) {
                  const auto s = cm({{c(1, 0), c(1, 0)}, {c(0, 0), c(1, 0)}});
                  auto uni = s.is_unitary();
                  auto herm = s.is_hermitian();
                  t.expect(uni.has_value() && !uni.value_or(true), "[[1,1],[0,1]] is not unitary");
                  t.expect(herm.has_value() && !herm.value_or(true),
                           "[[1,1],[0,1]] is not Hermitian");
              })
        // --- the identity XY = iZ (a Pauli algebra cross-check of complex multiply) ---
        .test("pauli_xy_equals_iZ",
              [](TestContext& t) {
                  // X·Y = [[i,0],[0,-i]] = iZ, where iZ = i * [[1,0],[0,-1]].
                  auto prod = pauli_x().multiply(pauli_y());
                  t.expect(prod.has_value(), "X·Y multiplies");
                  const auto iz = cm({{c(0, 1), c(0, 0)}, {c(0, 0), c(0, -1)}});
                  if (prod) {
                      t.expect(*prod == iz, "X·Y == iZ == [[i,0],[0,-i]]");
                  }
              })
        // --- error surfaces ---
        .test("from_rows_errors",
              [](TestContext& t) {
                  auto ragged = ComplexMatrix::from_rows({{c(1, 0), c(2, 0)}, {c(3, 0)}});
                  t.expect(!ragged.has_value() && ragged.error() == MathError::domain_error,
                           "ragged rows => domain_error");
                  auto empty = ComplexMatrix::from_rows({});
                  t.expect(!empty.has_value() && empty.error() == MathError::domain_error,
                           "empty => domain_error");
              })
        .test("multiply_dimension_mismatch",
              [](TestContext& t) {
                  // 2x2 times 3x2: inner dimensions 2 != 3.
                  const auto a = cm({{c(1, 0), c(0, 0)}, {c(0, 0), c(1, 0)}});
                  const auto b = cm({{c(1, 0), c(0, 0)}, {c(0, 0), c(1, 0)}, {c(1, 0), c(1, 0)}});
                  auto prod = a.multiply(b);
                  t.expect(!prod.has_value() && prod.error() == MathError::domain_error,
                           "inner-dimension mismatch => domain_error");
              })
        .test("predicates_require_square",
              [](TestContext& t) {
                  const auto a = cm({{c(1, 0), c(2, 0), c(3, 0)}, {c(4, 0), c(5, 0), c(6, 0)}});
                  auto herm = a.is_hermitian();
                  auto skew = a.is_skew_hermitian();
                  auto uni = a.is_unitary();
                  auto norm = a.is_normal();
                  t.expect(!herm.has_value() && herm.error() == MathError::domain_error,
                           "is_hermitian on non-square => domain_error");
                  t.expect(!skew.has_value() && skew.error() == MathError::domain_error,
                           "is_skew_hermitian on non-square => domain_error");
                  t.expect(!uni.has_value() && uni.error() == MathError::domain_error,
                           "is_unitary on non-square => domain_error");
                  t.expect(!norm.has_value() && norm.error() == MathError::domain_error,
                           "is_normal on non-square => domain_error");
              })
        .run();
}
