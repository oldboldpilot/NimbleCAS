// Tests for nimblecas.jordan: the Jordan canonical form J WITH the transforming matrix P
// (A = P*J*P^{-1}), exact over Q (Tier 1) and over a quadratic extension Q(alpha) (Tier 2),
// plus the honest not_implemented / domain_error boundary (Tier 3).
//
// Every returned (J, P) is checked against THE correctness property A*P == P*J, re-derived
// independently here rather than trusting the module's own internal verification.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.algnum;
import nimblecas.jordan;
import nimblecas.testing;

using nimblecas::AlgebraicNumber;
using nimblecas::MathError;
using nimblecas::Matrix;
using nimblecas::NumberField;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto ri(std::int64_t v) -> Rational {
    return Rational::from_int(v);
}

[[nodiscard]] auto mat(std::vector<std::vector<Rational>> rows) -> Matrix {
    return Matrix::from_rows(std::move(rows)).value();
}

// A polynomial from integer coefficients in ascending order (constant term first).
[[nodiscard]] auto poly(std::vector<std::int64_t> coeffs) -> RationalPoly {
    std::vector<Rational> rc;
    rc.reserve(coeffs.size());
    for (const std::int64_t v : coeffs) {
        rc.push_back(ri(v));
    }
    return RationalPoly::from_coeffs(std::move(rc));
}

// --- helpers for exact linear algebra over Q(alpha), used to re-check A*P == P*J ---

using AlgMat = std::vector<std::vector<AlgebraicNumber>>;

// Embed a Rational matrix into the field as an AlgMat.
[[nodiscard]] auto embed(const Matrix& a, const NumberField& field) -> AlgMat {
    AlgMat out;
    out.reserve(a.rows());
    for (std::size_t i = 0; i < a.rows(); ++i) {
        std::vector<AlgebraicNumber> row;
        row.reserve(a.cols());
        for (std::size_t j = 0; j < a.cols(); ++j) {
            row.push_back(field.from_rational(a.at(i, j)));
        }
        out.push_back(std::move(row));
    }
    return out;
}

// Exact product of two square AlgMat over the same field.
[[nodiscard]] auto alg_mul(const AlgMat& x, const AlgMat& y) -> AlgMat {
    const std::size_t n = x.size();
    const AlgebraicNumber zero = x.front().front().field().zero();
    AlgMat out(n, std::vector<AlgebraicNumber>(n, zero));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            AlgebraicNumber acc = zero;
            for (std::size_t k = 0; k < n; ++k) {
                acc = acc.add(x[i][k].multiply(y[k][j]).value()).value();
            }
            out[i][j] = acc;
        }
    }
    return out;
}

// Entrywise exact equality of two AlgMat.
[[nodiscard]] auto alg_eq(const AlgMat& x, const AlgMat& y) -> bool {
    if (x.size() != y.size()) {
        return false;
    }
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (x[i].size() != y[i].size()) {
            return false;
        }
        for (std::size_t j = 0; j < x[i].size(); ++j) {
            if (!x[i][j].is_equal(y[i][j])) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.jordan")
        // ---------------------------------------------------------------- Tier 1 ----
        .test("already_jordan_2x2_block",
              [](TestContext& t) {
                  // A = [[2,1],[0,2]] is a single 2x2 Jordan block (eigenvalue 2, defective).
                  // Its Jordan form is itself and P is the identity.
                  const auto A = mat({{ri(2), ri(1)}, {ri(0), ri(2)}});
                  auto r = nimblecas::rational_jordan_form(A).value();
                  t.expect(r.jordan.is_equal(mat({{ri(2), ri(1)}, {ri(0), ri(2)}})),
                           "J = [[2,1],[0,2]]");
                  t.expect(r.transform.is_equal(Matrix::identity(2)), "P = I_2");
                  // The correctness property, re-derived: A*P == P*J.
                  t.expect(A.multiply(r.transform).value().is_equal(
                               r.transform.multiply(r.jordan).value()),
                           "A*P == P*J");
              })
        .test("defective_2x2_nontrivial_transform",
              [](TestContext& t) {
                  // A = [[5,1],[-1,3]] has char poly (x-4)^2 with a single 2-block (the
                  // eigenspace of 4 is 1-dimensional). Hand-derived chain: eigenvector
                  // [1,-1], generalized vector [0,1]. So P = [[1,0],[-1,1]], J = [[4,1],[0,4]].
                  const auto A = mat({{ri(5), ri(1)}, {ri(-1), ri(3)}});
                  auto r = nimblecas::rational_jordan_form(A).value();
                  t.expect(r.jordan.is_equal(mat({{ri(4), ri(1)}, {ri(0), ri(4)}})),
                           "J = [[4,1],[0,4]]");
                  t.expect(r.transform.is_equal(mat({{ri(1), ri(0)}, {ri(-1), ri(1)}})),
                           "P = [[1,0],[-1,1]]");
                  t.expect(A.multiply(r.transform).value().is_equal(
                               r.transform.multiply(r.jordan).value()),
                           "A*P == P*J");
                  // P^{-1} A P == J, an independent second check.
                  auto pinv = r.transform.inverse().value();
                  t.expect(pinv.multiply(A).value().multiply(r.transform).value().is_equal(
                               r.jordan),
                           "P^{-1} A P == J");
              })
        .test("defective_3x3_block2_plus_block1",
              [](TestContext& t) {
                  // A = diag-block [[2,1,0],[0,2,0],[0,0,3]]: eigenvalue 2 with a single
                  // 2-block, eigenvalue 3 with a 1-block. Already in Jordan form => J == A,
                  // P == I (eigenvalue 2 is discovered before 3, fixing the block order).
                  const auto A = mat({{ri(2), ri(1), ri(0)},
                                      {ri(0), ri(2), ri(0)},
                                      {ri(0), ri(0), ri(3)}});
                  auto r = nimblecas::rational_jordan_form(A).value();
                  t.expect(r.jordan.is_equal(A), "J == A (already Jordan)");
                  t.expect(r.transform.is_equal(Matrix::identity(3)), "P == I_3");
                  t.expect(A.multiply(r.transform).value().is_equal(
                               r.transform.multiply(r.jordan).value()),
                           "A*P == P*J");
              })
        .test("diagonalizable_two_eigenvalues",
              [](TestContext& t) {
                  // A = [[1,2],[2,1]] is symmetric with eigenvalues -1 and 3 (discovered in
                  // that order). Diagonalizable => J diagonal. Hand-derived eigenvectors:
                  // for -1, [-1,1]; for 3, [1,1]. So P = [[-1,1],[1,1]], J = diag(-1,3).
                  const auto A = mat({{ri(1), ri(2)}, {ri(2), ri(1)}});
                  auto r = nimblecas::rational_jordan_form(A).value();
                  t.expect(r.jordan.is_equal(mat({{ri(-1), ri(0)}, {ri(0), ri(3)}})),
                           "J = diag(-1, 3)");
                  t.expect(r.transform.is_equal(mat({{ri(-1), ri(1)}, {ri(1), ri(1)}})),
                           "P = [[-1,1],[1,1]]");
                  t.expect(A.multiply(r.transform).value().is_equal(
                               r.transform.multiply(r.jordan).value()),
                           "A*P == P*J");
                  t.expect(r.transform.determinant().value() == ri(-2), "det P = -2 (invertible)");
              })
        .test("identity_is_its_own_jordan_form",
              [](TestContext& t) {
                  // I_3: eigenvalue 1 with three 1-blocks; J == I, P == I.
                  const auto I = Matrix::identity(3);
                  auto r = nimblecas::rational_jordan_form(I).value();
                  t.expect(r.jordan.is_equal(I), "J == I_3");
                  t.expect(r.transform.is_equal(I), "P == I_3");
                  t.expect(I.multiply(r.transform).value().is_equal(
                               r.transform.multiply(r.jordan).value()),
                           "A*P == P*J");
              })
        // ------------------------------------------------------- Tier 1 refusals ----
        .test("rational_form_refuses_nonsplitting_and_nonsquare",
              [](TestContext& t) {
                  // [[0,-1],[1,0]] has char poly x^2+1: no rational eigenvalues, so no
                  // Jordan form over Q. rational_jordan_form must refuse (domain_error).
                  const auto rot = mat({{ri(0), ri(-1)}, {ri(1), ri(0)}});
                  t.expect(nimblecas::rational_jordan_form(rot).error() == MathError::domain_error,
                           "non-splitting char poly => domain_error");
                  // Non-square input is a domain_error too.
                  const auto rect = mat({{ri(1), ri(2), ri(3)}, {ri(4), ri(5), ri(6)}});
                  t.expect(nimblecas::rational_jordan_form(rect).error() == MathError::domain_error,
                           "non-square => domain_error");
              })
        // ---------------------------------------------------------------- Tier 2 ----
        .test("rotation_over_Qi_diag_i_minus_i",
              [](TestContext& t) {
                  // A = [[0,-1],[1,0]], char poly x^2+1. Over Q(i) = Q[x]/(x^2+1) the Jordan
                  // form is diag(i, -i). Hand-derived eigenvectors: for i, [i,1]; for -i,
                  // [-i,1]. So P = [[i,-i],[1,1]], J = diag(i,-i).
                  const auto A = mat({{ri(0), ri(-1)}, {ri(1), ri(0)}});
                  auto r = nimblecas::jordan_form(A).value();
                  t.expect(r.field.degree() == 2, "extension degree 2");
                  t.expect(r.field.modulus().is_equal(poly({1, 0, 1})), "field Q[x]/(x^2+1)");

                  const AlgebraicNumber alpha = r.field.generator().value();  // i
                  const AlgebraicNumber conj = alpha.negate().value();        // -i
                  const AlgebraicNumber one = r.field.one();
                  const AlgebraicNumber zero = r.field.zero();

                  // J = diag(i, -i).
                  t.expect(r.jordan[0][0].is_equal(alpha) && r.jordan[1][1].is_equal(conj),
                           "J diagonal = (i, -i)");
                  t.expect(r.jordan[0][1].is_equal(zero) && r.jordan[1][0].is_equal(zero),
                           "J off-diagonal zero");
                  // P = [[i,-i],[1,1]].
                  t.expect(r.transform[0][0].is_equal(alpha) && r.transform[0][1].is_equal(conj),
                           "P first row = (i, -i)");
                  t.expect(r.transform[1][0].is_equal(one) && r.transform[1][1].is_equal(one),
                           "P second row = (1, 1)");
                  // The correctness property over Q(i): A*P == P*J, re-derived here.
                  const AlgMat ap = alg_mul(embed(A, r.field), r.transform);
                  const AlgMat pj = alg_mul(r.transform, r.jordan);
                  t.expect(alg_eq(ap, pj), "A*P == P*J over Q(i)");
                  // alpha is genuinely i: alpha^2 == -1.
                  t.expect(alpha.multiply(alpha).value().is_equal(
                               r.field.from_rational(ri(-1))),
                           "alpha^2 == -1");
              })
        .test("real_2x2_complex_conjugate_pair",
              [](TestContext& t) {
                  // A = [[2,-1],[1,2]] has char poly x^2 - 4x + 5, eigenvalues 2 +/- i, living
                  // in Q(alpha) = Q[x]/(x^2-4x+5). J is diagonal with the conjugate pair; we
                  // verify structure and the exact A*P == P*J property over the field.
                  const auto A = mat({{ri(2), ri(-1)}, {ri(1), ri(2)}});
                  auto r = nimblecas::jordan_form(A).value();
                  t.expect(r.field.degree() == 2, "extension degree 2");
                  t.expect(r.field.modulus().is_equal(poly({5, -4, 1})),
                           "field Q[x]/(x^2 - 4x + 5)");

                  const AlgebraicNumber alpha = r.field.generator().value();  // 2 + i
                  // conjugate = 4 - alpha (the two roots of x^2-4x+5 sum to 4).
                  const AlgebraicNumber conj =
                      r.field.from_rational(ri(4)).subtract(alpha).value();
                  const AlgebraicNumber zero = r.field.zero();

                  t.expect(r.jordan[0][1].is_equal(zero) && r.jordan[1][0].is_equal(zero),
                           "J diagonal");
                  t.expect(r.jordan[0][0].is_equal(alpha) && r.jordan[1][1].is_equal(conj),
                           "J = diag(alpha, 4-alpha)");
                  const AlgMat ap = alg_mul(embed(A, r.field), r.transform);
                  const AlgMat pj = alg_mul(r.transform, r.jordan);
                  t.expect(alg_eq(ap, pj), "A*P == P*J over Q(alpha)");
                  // alpha satisfies its minimal polynomial: alpha^2 - 4 alpha + 5 == 0.
                  auto a2 = alpha.multiply(alpha).value();
                  auto minus4a = alpha.multiply(r.field.from_rational(ri(-4))).value();
                  auto sum = a2.add(minus4a).value().add(r.field.from_rational(ri(5))).value();
                  t.expect(sum.is_zero(), "alpha^2 - 4 alpha + 5 == 0");
              })
        .test("defective_repeated_complex_pair_4x4",
              [](TestContext& t) {
                  // A = [[C, I2], [0, C]] with C = [[0,-1],[1,0]] (companion of x^2+1). Its
                  // char poly is (x^2+1)^2. Over Q(i) each of i, -i has a SINGLE Jordan block
                  // of size 2 (defective). J is block-diagonal with the i-block first, then
                  // the -i-block; P is invertible with A*P == P*J over Q(i).
                  const auto A = mat({{ri(0), ri(-1), ri(1), ri(0)},
                                      {ri(1), ri(0), ri(0), ri(1)},
                                      {ri(0), ri(0), ri(0), ri(-1)},
                                      {ri(0), ri(0), ri(1), ri(0)}});
                  auto r = nimblecas::jordan_form(A).value();
                  t.expect(r.field.modulus().is_equal(poly({1, 0, 1})), "field Q[x]/(x^2+1)");
                  t.expect(r.jordan.size() == 4 && r.transform.size() == 4, "4x4 J and P");

                  const AlgebraicNumber alpha = r.field.generator().value();  // i
                  const AlgebraicNumber conj = alpha.negate().value();        // -i
                  const AlgebraicNumber one = r.field.one();

                  // i-block occupies rows/cols 0..1, -i-block rows/cols 2..3.
                  t.expect(r.jordan[0][0].is_equal(alpha) && r.jordan[1][1].is_equal(alpha),
                           "diagonal of first block = i");
                  t.expect(r.jordan[0][1].is_equal(one), "superdiagonal 1 in i-block (defective)");
                  t.expect(r.jordan[2][2].is_equal(conj) && r.jordan[3][3].is_equal(conj),
                           "diagonal of second block = -i");
                  t.expect(r.jordan[2][3].is_equal(one), "superdiagonal 1 in -i-block (defective)");

                  const AlgMat ap = alg_mul(embed(A, r.field), r.transform);
                  const AlgMat pj = alg_mul(r.transform, r.jordan);
                  t.expect(alg_eq(ap, pj), "A*P == P*J over Q(i)");
              })
        // ---------------------------------------------------------------- Tier 3 ----
        .test("degree_three_factor_not_implemented",
              [](TestContext& t) {
                  // Companion of x^3 - 2 = [[0,0,2],[1,0,0],[0,1,0]]: char poly x^3 - 2 is
                  // irreducible of degree 3 (a cubic splitting field, out of scope).
                  const auto A = mat({{ri(0), ri(0), ri(2)},
                                      {ri(1), ri(0), ri(0)},
                                      {ri(0), ri(1), ri(0)}});
                  t.expect(nimblecas::jordan_form(A).error() == MathError::not_implemented,
                           "degree-3 irreducible factor => not_implemented");
                  // And rational_jordan_form refuses it (no rational eigenvalue) as domain_error.
                  t.expect(nimblecas::rational_jordan_form(A).error() == MathError::domain_error,
                           "x^3 - 2 does not split over Q => domain_error");
              })
        .test("two_distinct_quadratics_not_implemented",
              [](TestContext& t) {
                  // block diag( companion(x^2+1), companion(x^2-2) ): char poly
                  // (x^2+1)(x^2-2) has TWO distinct irreducible quadratic factors (a possibly
                  // composite extension), which is out of scope.
                  const auto A = mat({{ri(0), ri(-1), ri(0), ri(0)},
                                      {ri(1), ri(0), ri(0), ri(0)},
                                      {ri(0), ri(0), ri(0), ri(2)},
                                      {ri(0), ri(0), ri(1), ri(0)}});
                  t.expect(nimblecas::jordan_form(A).error() == MathError::not_implemented,
                           "two distinct quadratic factors => not_implemented");
              })
        .test("jordan_form_boundaries",
              [](TestContext& t) {
                  // A matrix that splits over Q has no extension: jordan_form defers to
                  // rational_jordan_form with a domain_error.
                  const auto split = mat({{ri(2), ri(1)}, {ri(0), ri(2)}});
                  t.expect(nimblecas::jordan_form(split).error() == MathError::domain_error,
                           "splits over Q => domain_error (use rational_jordan_form)");
                  // Non-square input is a domain_error.
                  const auto rect = mat({{ri(1), ri(2), ri(3)}, {ri(4), ri(5), ri(6)}});
                  t.expect(nimblecas::jordan_form(rect).error() == MathError::domain_error,
                           "non-square => domain_error");
              })
        .run();
}
