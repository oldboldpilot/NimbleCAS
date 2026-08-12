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

// Every column of a square AlgMat has at least one nonzero entry. For the columns of P this
// is a P-invertibility certificate that does NOT pin the (non-canonical) scaling of the
// eigenvectors: A*P == P*J makes each column an eigenvector of its block's eigenvalue, and a
// set of nonzero eigenvectors for DISTINCT eigenvalues is automatically linearly independent.
[[nodiscard]] auto columns_all_nonzero(const AlgMat& p) -> bool {
    const std::size_t n = p.size();
    for (std::size_t j = 0; j < n; ++j) {
        bool nonzero = false;
        for (std::size_t i = 0; i < n; ++i) {
            if (!p[i][j].is_zero()) {
                nonzero = true;
                break;
            }
        }
        if (!nonzero) {
            return false;
        }
    }
    return true;
}

// Find the JordanFactorStructure for `target` in a jordan_structure() result's factor
// list, by exact polynomial equality. Used so tests that don't care about the
// (implementation-detail-adjacent) canonical order among equal-degree factors can still
// pin exact expectations per factor.
[[nodiscard]] auto find_factor(const nimblecas::JordanStructure& s, const RationalPoly& target)
    -> const nimblecas::JordanFactorStructure* {
    for (const auto& f : s.factors) {
        if (f.factor.is_equal(target)) {
            return &f;
        }
    }
    return nullptr;
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
                  // The correctness property, re-derived: A*P == P*J. P itself is only
                  // defined up to a valid change of Jordan basis, so we assert invertibility
                  // rather than a specific representative.
                  t.expect(A.multiply(r.transform).value().is_equal(
                               r.transform.multiply(r.jordan).value()),
                           "A*P == P*J");
                  t.expect(r.transform.determinant().value() != ri(0), "P invertible");
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
                  t.expect(A.multiply(r.transform).value().is_equal(
                               r.transform.multiply(r.jordan).value()),
                           "A*P == P*J");
                  // P^{-1} A P == J (basis-independent): requires P invertible and holds for
                  // any valid Jordan basis, so it certifies correctness without pinning the
                  // (non-canonical) generalized-eigenvector representative.
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
                  t.expect(A.multiply(r.transform).value().is_equal(
                               r.transform.multiply(r.jordan).value()),
                           "A*P == P*J");
                  t.expect(r.transform.determinant().value() != ri(0), "P invertible");
              })
        .test("diagonalizable_two_eigenvalues",
              [](TestContext& t) {
                  // A = [[1,2],[2,1]] is symmetric with eigenvalues -1 and 3 (discovered in
                  // that order, so J is canonical). Diagonalizable => J = diag(-1, 3). The
                  // eigenvectors — hence P — are only defined up to a nonzero scale, so we
                  // assert J and A*P == P*J and that P is invertible, not a specific P.
                  const auto A = mat({{ri(1), ri(2)}, {ri(2), ri(1)}});
                  auto r = nimblecas::rational_jordan_form(A).value();
                  t.expect(r.jordan.is_equal(mat({{ri(-1), ri(0)}, {ri(0), ri(3)}})),
                           "J = diag(-1, 3)");
                  t.expect(A.multiply(r.transform).value().is_equal(
                               r.transform.multiply(r.jordan).value()),
                           "A*P == P*J");
                  t.expect(r.transform.determinant().value() != ri(0), "P invertible");
              })
        .test("identity_is_its_own_jordan_form",
              [](TestContext& t) {
                  // I_3: eigenvalue 1 with three 1-blocks; J == I, P == I.
                  const auto I = Matrix::identity(3);
                  auto r = nimblecas::rational_jordan_form(I).value();
                  t.expect(r.jordan.is_equal(I), "J == I_3");
                  t.expect(I.multiply(r.transform).value().is_equal(
                               r.transform.multiply(r.jordan).value()),
                           "A*P == P*J");
                  t.expect(r.transform.determinant().value() != ri(0), "P invertible");
              })
        .test("nilpotent_two_size2_blocks",
              [](TestContext& t) {
                  // A = [[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]] is nilpotent (A^2 = 0) with
                  // nullity sequence d1 = 2, d2 = 4 => two Jordan blocks of size 2 at
                  // eigenvalue 0. The canonical J is block-diag( [[0,1],[0,0]], [[0,1],[0,0]] )
                  // = [[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]] (which is NOT A itself).
                  const auto A = mat({{ri(0), ri(0), ri(1), ri(0)},
                                      {ri(0), ri(0), ri(0), ri(1)},
                                      {ri(0), ri(0), ri(0), ri(0)},
                                      {ri(0), ri(0), ri(0), ri(0)}});
                  auto r = nimblecas::rational_jordan_form(A).value();
                  t.expect(r.jordan.is_equal(mat({{ri(0), ri(1), ri(0), ri(0)},
                                                  {ri(0), ri(0), ri(0), ri(0)},
                                                  {ri(0), ri(0), ri(0), ri(1)},
                                                  {ri(0), ri(0), ri(0), ri(0)}})),
                           "J = two size-2 blocks at 0");
                  t.expect(A.multiply(r.transform).value().is_equal(
                               r.transform.multiply(r.jordan).value()),
                           "A*P == P*J");
                  t.expect(r.transform.determinant().value() != ri(0), "P invertible");
              })
        .test("nilpotent_block_sizes_3_and_1",
              [](TestContext& t) {
                  // A = J_3(0) (+) J_1(0) = [[0,1,0,0],[0,0,1,0],[0,0,0,0],[0,0,0,0]]:
                  // nullity sequence d1 = 2, d2 = 3, d3 = 4 => one size-3 block and one size-1
                  // block at eigenvalue 0. Largest-block-first ordering makes J == A.
                  const auto A = mat({{ri(0), ri(1), ri(0), ri(0)},
                                      {ri(0), ri(0), ri(1), ri(0)},
                                      {ri(0), ri(0), ri(0), ri(0)},
                                      {ri(0), ri(0), ri(0), ri(0)}});
                  auto r = nimblecas::rational_jordan_form(A).value();
                  t.expect(r.jordan.is_equal(A), "J = J_3(0) (+) J_1(0) == A");
                  t.expect(A.multiply(r.transform).value().is_equal(
                               r.transform.multiply(r.jordan).value()),
                           "A*P == P*J");
                  t.expect(r.transform.determinant().value() != ri(0), "P invertible");
              })
        .test("nilpotent_2_1_non_axis_aligned",
              [](TestContext& t) {
                  // A = [[1,-1,0],[1,-1,0],[0,0,0]] has char poly x^3 and nullity sequence
                  // d1 = 2, d2 = 3 (A^2 = 0) => blocks (2, 1) at eigenvalue 0. The generalized
                  // eigenvectors are NOT standard basis vectors, so P is genuinely non-trivial;
                  // we pin only the canonical J and the invariants, never P.
                  const auto A = mat({{ri(1), ri(-1), ri(0)},
                                      {ri(1), ri(-1), ri(0)},
                                      {ri(0), ri(0), ri(0)}});
                  auto r = nimblecas::rational_jordan_form(A).value();
                  t.expect(r.jordan.is_equal(mat({{ri(0), ri(1), ri(0)},
                                                  {ri(0), ri(0), ri(0)},
                                                  {ri(0), ri(0), ri(0)}})),
                           "J = one size-2 block + one size-1 block at 0");
                  t.expect(A.multiply(r.transform).value().is_equal(
                               r.transform.multiply(r.jordan).value()),
                           "A*P == P*J");
                  t.expect(r.transform.determinant().value() != ri(0), "P invertible");
              })
        .test("single_size3_jordan_block",
              [](TestContext& t) {
                  // A = [[2,1,0],[0,2,1],[0,0,2]] is a single size-3 Jordan block at eigenvalue
                  // 2 (nullity of A-2I stays 1 until (A-2I)^3): J == A.
                  const auto A = mat({{ri(2), ri(1), ri(0)},
                                      {ri(0), ri(2), ri(1)},
                                      {ri(0), ri(0), ri(2)}});
                  auto r = nimblecas::rational_jordan_form(A).value();
                  t.expect(r.jordan.is_equal(A), "J = single size-3 block == A");
                  t.expect(A.multiply(r.transform).value().is_equal(
                               r.transform.multiply(r.jordan).value()),
                           "A*P == P*J");
                  t.expect(r.transform.determinant().value() != ri(0), "P invertible");
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
                  const AlgebraicNumber zero = r.field.zero();

                  // J = diag(i, -i) is canonical (the block order follows the eigenvalue
                  // order alpha, conj). The eigenvectors that form P are only defined up to
                  // a nonzero scale, so P itself is not pinned.
                  t.expect(r.jordan[0][0].is_equal(alpha) && r.jordan[1][1].is_equal(conj),
                           "J diagonal = (i, -i)");
                  t.expect(r.jordan[0][1].is_equal(zero) && r.jordan[1][0].is_equal(zero),
                           "J off-diagonal zero");
                  // The correctness property over Q(i): A*P == P*J, re-derived here.
                  const AlgMat ap = alg_mul(embed(A, r.field), r.transform);
                  const AlgMat pj = alg_mul(r.transform, r.jordan);
                  t.expect(alg_eq(ap, pj), "A*P == P*J over Q(i)");
                  // P invertible over Q(i): each column is a NONZERO eigenvector (A*P == P*J
                  // gives the eigenvector relation), and nonzero eigenvectors for the distinct
                  // eigenvalues i, -i are automatically independent.
                  t.expect(columns_all_nonzero(r.transform),
                           "P columns are nonzero eigenvectors (P invertible over Q(i))");
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
                  // P invertible: nonzero eigenvectors for the distinct eigenvalues
                  // alpha, 4-alpha are automatically independent.
                  t.expect(columns_all_nonzero(r.transform),
                           "P columns are nonzero eigenvectors (P invertible)");
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
        .test("mixed_rational_and_complex_pair",
              [](TestContext& t) {
                  // A = [1] (+) [[0,-1],[1,0]] = [[1,0,0],[0,0,-1],[0,1,0]] has char poly
                  // (x-1)(x^2+1): a RATIONAL eigenvalue 1 together with a conjugate pair i, -i,
                  // all living in Q(i). Each eigenvalue is simple, so J = diag(1, i, -i) with
                  // the rational eigenvalue embedded via from_rational. Hand-verified that
                  // A*[1,0,0] = 1*[1,0,0], A*[0,i,1] = i*[0,i,1], A*[0,-i,1] = -i*[0,-i,1].
                  const auto A = mat({{ri(1), ri(0), ri(0)},
                                      {ri(0), ri(0), ri(-1)},
                                      {ri(0), ri(1), ri(0)}});
                  auto r = nimblecas::jordan_form(A).value();
                  t.expect(r.field.degree() == 2, "extension degree 2");
                  t.expect(r.field.modulus().is_equal(poly({1, 0, 1})), "field Q[x]/(x^2+1)");

                  const AlgebraicNumber emb1 = r.field.from_rational(ri(1));  // 1
                  const AlgebraicNumber alpha = r.field.generator().value();  // i
                  const AlgebraicNumber conj = alpha.negate().value();        // -i

                  // J is diagonal, and its diagonal is a permutation of { 1, i, -i } (checked
                  // by membership, so the test does not depend on the block order).
                  bool off_zero = true;
                  for (std::size_t i = 0; i < 3; ++i) {
                      for (std::size_t j = 0; j < 3; ++j) {
                          if (i != j && !r.jordan[i][j].is_zero()) {
                              off_zero = false;
                          }
                      }
                  }
                  t.expect(off_zero, "J is diagonal (all eigenvalues simple)");
                  int c1 = 0;
                  int ci = 0;
                  int cc = 0;
                  int cother = 0;
                  for (std::size_t k = 0; k < 3; ++k) {
                      const AlgebraicNumber& d = r.jordan[k][k];
                      if (d.is_equal(emb1)) {
                          ++c1;
                      } else if (d.is_equal(alpha)) {
                          ++ci;
                      } else if (d.is_equal(conj)) {
                          ++cc;
                      } else {
                          ++cother;
                      }
                  }
                  t.expect(c1 == 1 && ci == 1 && cc == 1 && cother == 0,
                           "J diagonal multiset == { 1, i, -i }");

                  const AlgMat ap = alg_mul(embed(A, r.field), r.transform);
                  const AlgMat pj = alg_mul(r.transform, r.jordan);
                  t.expect(alg_eq(ap, pj), "A*P == P*J over Q(i)");
                  // P invertible: nonzero eigenvectors for the three DISTINCT eigenvalues
                  // 1, i, -i are automatically independent.
                  t.expect(columns_all_nonzero(r.transform),
                           "P columns are nonzero eigenvectors (P invertible over Q(i))");
              })
        // ---------------------------------------------------------------- Tier 3 ----
        .test("degree_three_factor_not_implemented",
              [](TestContext& t) {
                  // Companion of x^3 - 2 = [[0,0,2],[1,0,0],[0,1,0]]: char poly x^3 - 2 is
                  // irreducible of degree 3. Its splitting field Q(cbrt2, omega) has degree 6
                  // (Galois group S3); building it drives factor_over_field's Trager norm to a
                  // degree deg(L)*deg(f) = 3*3 = 9 polynomial over Q, beyond factor_over_Q's
                  // own practical budget -- so even though 6 <= the default cap of 12, the
                  // general splitting-field path honestly refuses (not_implemented), a real
                  // boundary of the int64-exact envelope, not a fabricated field or root.
                  const auto A = mat({{ri(0), ri(0), ri(2)},
                                      {ri(1), ri(0), ri(0)},
                                      {ri(0), ri(1), ri(0)}});
                  auto jf_cubic = nimblecas::jordan_form(A);
                  if (jf_cubic.has_value()) {
                      std::println("  UNEXPECTED: jordan_form(x^3-2) succeeded (field degree {})",
                                   jf_cubic->field.degree());
                  }
                  t.expect(!jf_cubic.has_value() && jf_cubic.error() == MathError::not_implemented,
                           "cubic splitting field beyond the exact envelope => not_implemented");
                  // And rational_jordan_form refuses it (no rational eigenvalue) as domain_error.
                  auto rjf_cubic = nimblecas::rational_jordan_form(A);
                  t.expect(!rjf_cubic.has_value() && rjf_cubic.error() == MathError::domain_error,
                           "x^3 - 2 does not split over Q => domain_error");
                  // jordan_structure needs no splitting field at all, so it still answers
                  // exactly over Q: the honest fallback when jordan_form must refuse.
                  auto s = nimblecas::jordan_structure(A);
                  t.expect(s.has_value(), "jordan_structure still succeeds where jordan_form refuses");
                  if (!s) {
                      return;
                  }
                  t.expect(s->factors.size() == 1, "one irreducible factor");
                  if (s->factors.size() != 1) {
                      return;
                  }
                  t.expect(s->factors.front().block_sizes == std::vector<std::int64_t>{1},
                           "block_sizes {1} (diagonalizable over the splitting field)");
              })
        .test("two_distinct_quadratics_splitting_field",
              [](TestContext& t) {
                  // block diag( companion(x^2-2), companion(x^2-3) ): char poly
                  // (x^2-2)(x^2-3) has TWO distinct irreducible quadratic factors. This used
                  // to be an honest not_implemented (Tier 3 refusal); it is now within the
                  // general splitting-field path's default envelope: Q(sqrt2, sqrt3) has
                  // degree [Q(sqrt2):Q]*[Q(sqrt3):Q] = 4 (sqrt2, sqrt3 are independent
                  // quadratic irrationalities), well under kDefaultMaxSplittingFieldDegree.
                  const auto A = mat({{ri(0), ri(2), ri(0), ri(0)},
                                      {ri(1), ri(0), ri(0), ri(0)},
                                      {ri(0), ri(0), ri(0), ri(3)},
                                      {ri(0), ri(0), ri(1), ri(0)}});
                  auto jf = nimblecas::jordan_form(A);
                  if (!jf) {
                      std::println("  jordan_form(2 quadratics) failed: not_impl={} overflow={} domain={}",
                                   jf.error() == MathError::not_implemented,
                                   jf.error() == MathError::overflow,
                                   jf.error() == MathError::domain_error);
                  }
                  t.expect(jf.has_value(),
                           "two distinct quadratics => splitting-field jordan_form succeeds");
                  if (!jf) return;
                  const auto& r = *jf;
                  t.expect(r.field.degree() == 4, "splitting field Q(sqrt2, sqrt3) has degree 4");
                  t.expect(r.jordan.size() == 4 && r.transform.size() == 4, "4x4 J and P");

                  // J must be diagonal (all four eigenvalues are simple): each diagonal entry
                  // squares to 2 or 3 (canonical order is unspecified, so check by content,
                  // not position), and off-diagonal entries are zero.
                  const AlgebraicNumber two = r.field.from_rational(ri(2));
                  const AlgebraicNumber three = r.field.from_rational(ri(3));
                  bool off_zero = true;
                  int c2 = 0;
                  int c3 = 0;
                  int cother = 0;
                  for (std::size_t i = 0; i < 4; ++i) {
                      for (std::size_t j = 0; j < 4; ++j) {
                          if (i != j && !r.jordan[i][j].is_zero()) {
                              off_zero = false;
                          }
                      }
                      const AlgebraicNumber sq = r.jordan[i][i].multiply(r.jordan[i][i]).value();
                      if (sq.is_equal(two)) {
                          ++c2;
                      } else if (sq.is_equal(three)) {
                          ++c3;
                      } else {
                          ++cother;
                      }
                  }
                  t.expect(off_zero, "J is diagonal (all eigenvalues simple)");
                  t.expect(c2 == 2 && c3 == 2 && cother == 0,
                           "J diagonal squares to the multiset {2, 2, 3, 3} (+/-sqrt2, +/-sqrt3)");

                  // The correctness property, re-derived here rather than trusting the
                  // module's own internal verification: A*P == P*J over the splitting field.
                  const AlgMat ap = alg_mul(embed(A, r.field), r.transform);
                  const AlgMat pj = alg_mul(r.transform, r.jordan);
                  t.expect(alg_eq(ap, pj), "A*P == P*J over Q(sqrt2, sqrt3)");
                  t.expect(columns_all_nonzero(r.transform),
                           "P columns are nonzero eigenvectors (P invertible)");
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
        // ------------------------------------------------------- jordan_structure ----
        // jordan_structure(A) recovers the Segre characteristic (Jordan block-size
        // partition per irreducible factor of the char poly) EXACTLY OVER Q, without
        // ever constructing a splitting field -- valid for irrational/complex
        // eigenvalues too, unlike jordan_form's Tier 1/2/3 boundary.
        .test("structure_companion_x3_minus_2_diagonalizable",
              [](TestContext& t) {
                  // companion(x^3 - 2): char poly x^3-2 is irreducible over Q (no
                  // rational root: +/-1, +/-2 all fail). A single irreducible factor of
                  // degree 3, multiplicity 1: each of its 3 conjugate roots (all
                  // distinct) is a simple eigenvalue => block_sizes {1}. No field is
                  // ever constructed to reach this answer.
                  const auto A = mat({{ri(0), ri(0), ri(2)},
                                      {ri(1), ri(0), ri(0)},
                                      {ri(0), ri(1), ri(0)}});
                  auto r = nimblecas::jordan_structure(A).value();
                  t.expect(r.factors.size() == 1, "one irreducible factor");
                  if (r.factors.size() != 1) {
                      return;
                  }
                  const auto& f = r.factors.front();
                  t.expect(f.factor.is_equal(poly({-2, 0, 0, 1})), "factor == x^3 - 2");
                  t.expect(f.degree == 3, "degree 3");
                  t.expect(f.multiplicity == 1, "multiplicity 1");
                  t.expect(f.block_sizes == std::vector<std::int64_t>{1}, "block_sizes {1}");
              })
        .test("structure_companion_x3_minus_3x_minus_1_diagonalizable",
              [](TestContext& t) {
                  // companion(x^3 - 3x - 1): char poly x^3-3x-1 is irreducible over Q
                  // (rational root candidates +/-1 both fail; a cubic with no rational
                  // root is irreducible). By Cayley-Hamilton m(A) == 0 here (m is the
                  // whole char poly), so ker(m(A)) is everything => the 3 conjugate
                  // roots are each simple: block_sizes {1}.
                  const auto A = mat({{ri(0), ri(0), ri(1)},
                                      {ri(1), ri(0), ri(3)},
                                      {ri(0), ri(1), ri(0)}});
                  auto r = nimblecas::jordan_structure(A).value();
                  t.expect(r.factors.size() == 1, "one irreducible factor");
                  if (r.factors.size() != 1) {
                      return;
                  }
                  const auto& f = r.factors.front();
                  t.expect(f.factor.is_equal(poly({-1, -3, 0, 1})), "factor == x^3 - 3x - 1");
                  t.expect(f.degree == 3, "degree 3");
                  t.expect(f.multiplicity == 1, "multiplicity 1");
                  t.expect(f.block_sizes == std::vector<std::int64_t>{1}, "block_sizes {1}");
              })
        .test("structure_defective_6x6_repeated_cubic_factor",
              [](TestContext& t) {
                  // A = [[C, I3], [0, C]], C = companion(x^3 - 3x - 1). Char poly is
                  // (x^3-3x-1)^2: one irreducible factor of degree 3, multiplicity 2.
                  // Hand-derived: with M = m(A), rank(M) = 3 and rank(M^2) = 0, so
                  // nu_1 = (6-3)/3 = 1, nu_2 = (6-0)/3 = 2 -- each of the 3 conjugate
                  // roots has a SINGLE defective 2x2 block (block_sizes {2}), not two
                  // separate 1x1 blocks. This is exactly the case an extension-field
                  // construction would be needed for jordan_form/Tier-3 refusal, yet
                  // jordan_structure answers it exactly over Q.
                  const auto A = mat({{ri(0), ri(0), ri(1), ri(1), ri(0), ri(0)},
                                      {ri(1), ri(0), ri(3), ri(0), ri(1), ri(0)},
                                      {ri(0), ri(1), ri(0), ri(0), ri(0), ri(1)},
                                      {ri(0), ri(0), ri(0), ri(0), ri(0), ri(1)},
                                      {ri(0), ri(0), ri(0), ri(1), ri(0), ri(3)},
                                      {ri(0), ri(0), ri(0), ri(0), ri(1), ri(0)}});
                  auto r = nimblecas::jordan_structure(A).value();
                  t.expect(r.factors.size() == 1, "one irreducible factor");
                  if (r.factors.size() != 1) {
                      return;
                  }
                  const auto& f = r.factors.front();
                  t.expect(f.factor.is_equal(poly({-1, -3, 0, 1})), "factor == x^3 - 3x - 1");
                  t.expect(f.degree == 3, "degree 3");
                  t.expect(f.multiplicity == 2, "multiplicity 2");
                  t.expect(f.block_sizes == std::vector<std::int64_t>{2}, "block_sizes {2}");
              })
        .test("structure_diag_2_2_3_rational_eigenvalues",
              [](TestContext& t) {
                  // diag(2,2,3): char poly (x-2)^2(x-3), two rational (degree-1)
                  // irreducible factors. Already diagonal => eigenvalue 2 has two
                  // independent 1x1 blocks (block_sizes {1,1}), eigenvalue 3 has one
                  // 1x1 block (block_sizes {1}).
                  const auto A = mat({{ri(2), ri(0), ri(0)},
                                      {ri(0), ri(2), ri(0)},
                                      {ri(0), ri(0), ri(3)}});
                  auto r = nimblecas::jordan_structure(A).value();
                  t.expect(r.factors.size() == 2, "two irreducible factors");
                  if (r.factors.size() != 2) {
                      return;
                  }
                  // Canonical order: both factors are degree 1, ordered coefficient-lex
                  // by (constant term) descending significance from the top coefficient
                  // down -- x^1 coefficients tie (both 1), so the constant term -2 vs
                  // -3 breaks the tie; verify each factor by content rather than
                  // assuming which index it lands at.
                  const auto* f2 = find_factor(r, poly({-2, 1}));  // x - 2
                  const auto* f3 = find_factor(r, poly({-3, 1}));  // x - 3
                  t.expect(f2 != nullptr, "factor x - 2 present");
                  t.expect(f3 != nullptr, "factor x - 3 present");
                  if (f2 == nullptr || f3 == nullptr) {
                      return;
                  }
                  t.expect(f2->degree == 1 && f2->multiplicity == 2 &&
                               f2->block_sizes == std::vector<std::int64_t>{1, 1},
                           "x - 2: degree 1, multiplicity 2, block_sizes {1,1}");
                  t.expect(f3->degree == 1 && f3->multiplicity == 1 &&
                               f3->block_sizes == std::vector<std::int64_t>{1},
                           "x - 3: degree 1, multiplicity 1, block_sizes {1}");
                  // Canonical order is deterministic: verify it directly too.
                  t.expect(r.factors[0].factor.is_equal(poly({-3, 1})) &&
                               r.factors[1].factor.is_equal(poly({-2, 1})),
                           "canonical order: coefficient-lex among equal-degree factors");
              })
        .test("structure_defective_2x2_single_rational_eigenvalue",
              [](TestContext& t) {
                  // A = [[5,1],[-1,3]] (reused from defective_2x2_nontrivial_transform):
                  // char poly (x-4)^2, a single rational eigenvalue 4 with a single 2x2
                  // defective block (the eigenspace of 4 is 1-dimensional).
                  const auto A = mat({{ri(5), ri(1)}, {ri(-1), ri(3)}});
                  auto r = nimblecas::jordan_structure(A).value();
                  t.expect(r.factors.size() == 1, "one irreducible factor");
                  if (r.factors.size() != 1) {
                      return;
                  }
                  const auto& f = r.factors.front();
                  t.expect(f.factor.is_equal(poly({-4, 1})), "factor == x - 4");
                  t.expect(f.degree == 1, "degree 1");
                  t.expect(f.multiplicity == 2, "multiplicity 2");
                  t.expect(f.block_sizes == std::vector<std::int64_t>{2}, "block_sizes {2}");
              })
        .test("structure_rotation_irrational_pair_no_field_built",
              [](TestContext& t) {
                  // [[0,-1],[1,0]]: char poly x^2+1, irreducible over Q -- the classic
                  // case jordan_form needs Q(i) for. jordan_structure answers it exactly
                  // over Q alone: one degree-2 factor, multiplicity 1, block_sizes {1}
                  // (the two conjugate eigenvalues i, -i are each simple).
                  const auto A = mat({{ri(0), ri(-1)}, {ri(1), ri(0)}});
                  auto r = nimblecas::jordan_structure(A).value();
                  t.expect(r.factors.size() == 1, "one irreducible factor");
                  if (r.factors.size() != 1) {
                      return;
                  }
                  const auto& f = r.factors.front();
                  t.expect(f.factor.is_equal(poly({1, 0, 1})), "factor == x^2 + 1");
                  t.expect(f.degree == 2, "degree 2");
                  t.expect(f.multiplicity == 1, "multiplicity 1");
                  t.expect(f.block_sizes == std::vector<std::int64_t>{1}, "block_sizes {1}");
              })
        .test("structure_non_square_is_domain_error",
              [](TestContext& t) {
                  const auto rect = mat({{ri(1), ri(2), ri(3)}, {ri(4), ri(5), ri(6)}});
                  t.expect(nimblecas::jordan_structure(rect).error() == MathError::domain_error,
                           "non-square => domain_error");
              })
        .run();
}
