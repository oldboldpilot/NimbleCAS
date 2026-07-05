// Tests for nimblecas.frobenius: exact invariant factors, minimal polynomial, and the
// rational canonical (Frobenius) form over Q via the Smith normal form of x*I - A.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.eigen;
import nimblecas.frobenius;
import nimblecas.testing;

using nimblecas::MathError;
using nimblecas::Matrix;
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

// The product of a list of polynomials (empty product == the constant 1).
[[nodiscard]] auto product(const std::vector<RationalPoly>& fs) -> RationalPoly {
    RationalPoly acc = RationalPoly::constant(ri(1));
    for (const RationalPoly& f : fs) {
        acc = acc.multiply(f).value();
    }
    return acc;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.frobenius")
        .test("companion_matrix_is_its_own_rcf",
              [](TestContext& t) {
                  // A = companion(x^2 - 5x + 6) = [[0,-6],[1,5]] (char poly x^2 - 5x + 6).
                  // A cyclic (companion) matrix has a single invariant factor = its char
                  // poly, and is already in rational canonical form, so RCF(A) == A.
                  const auto A = mat({{ri(0), ri(-6)}, {ri(1), ri(5)}});
                  auto factors = nimblecas::invariant_factors(A).value();
                  t.expect(factors.size() == 1, "one invariant factor for a companion matrix");
                  t.expect(factors[0].is_equal(poly({6, -5, 1})),
                           "invariant factor = x^2 - 5x + 6");
                  auto rcf = nimblecas::rational_canonical_form(A).value();
                  t.expect(rcf.is_equal(A), "RCF of a companion matrix is the matrix itself");
                  t.expect(nimblecas::minimal_polynomial(A).value().is_equal(poly({6, -5, 1})),
                           "minimal polynomial = x^2 - 5x + 6");
              })
        .test("companion_matrix_builder_matches_convention",
              [](TestContext& t) {
                  // companion_matrix(x^2 + 1) = [[0,-1],[1,0]] (right-column form).
                  auto c = nimblecas::companion_matrix(poly({1, 0, 1})).value();
                  t.expect(c.is_equal(mat({{ri(0), ri(-1)}, {ri(1), ri(0)}})),
                           "companion(x^2+1) = [[0,-1],[1,0]]");
                  // A non-monic multiple normalises to the same companion.
                  auto c2 = nimblecas::companion_matrix(poly({3, 0, 3})).value();
                  t.expect(c2.is_equal(c), "non-monic 3x^2+3 gives the same companion as x^2+1");
                  // A constant polynomial has no companion.
                  t.expect(nimblecas::companion_matrix(poly({7})).error() == MathError::domain_error,
                           "constant polynomial => domain_error");
                  t.expect(nimblecas::companion_matrix(RationalPoly{}).error() ==
                               MathError::domain_error,
                           "zero polynomial => domain_error");
              })
        .test("diagonal_repeated_eigenvalue_invariant_factors",
              [](TestContext& t) {
                  // A = diag(2, 2, 3). Char poly (x-2)^2 (x-3). The invariant factors are
                  // (x-2) and (x-2)(x-3) = x^2 - 5x + 6 (the (x-2) blocks cannot be a single
                  // companion of (x-2)^2 because A is diagonalisable there).
                  const auto A = mat({{ri(2), ri(0), ri(0)},
                                      {ri(0), ri(2), ri(0)},
                                      {ri(0), ri(0), ri(3)}});
                  auto factors = nimblecas::invariant_factors(A).value();
                  t.expect(factors.size() == 2, "two invariant factors for diag(2,2,3)");
                  t.expect(factors[0].is_equal(poly({-2, 1})), "f_1 = x - 2");
                  t.expect(factors[1].is_equal(poly({6, -5, 1})), "f_2 = (x-2)(x-3) = x^2-5x+6");
                  t.expect(nimblecas::minimal_polynomial(A).value().is_equal(poly({6, -5, 1})),
                           "minimal polynomial = (x-2)(x-3)");
                  // RCF = diag( C(x-2), C(x^2-5x+6) ) = [[2,0,0],[0,0,-6],[0,1,5]].
                  auto rcf = nimblecas::rational_canonical_form(A).value();
                  t.expect(rcf.is_equal(mat({{ri(2), ri(0), ri(0)},
                                             {ri(0), ri(0), ri(-6)},
                                             {ri(0), ri(1), ri(5)}})),
                           "RCF(diag(2,2,3)) = [[2,0,0],[0,0,-6],[0,1,5]]");
              })
        .test("irrational_eigenvalues_stay_exact",
              [](TestContext& t) {
                  // A = [[0,-1],[1,0]] has char poly x^2 + 1 (eigenvalues +/- i, irrational
                  // over Q). The RCF is EXACT anyway: a single invariant factor x^2+1, and
                  // the form is the companion [[0,-1],[1,0]] = A itself. This is the key
                  // advantage over the Jordan form, which would need the eigenvalues.
                  const auto A = mat({{ri(0), ri(-1)}, {ri(1), ri(0)}});
                  auto factors = nimblecas::invariant_factors(A).value();
                  t.expect(factors.size() == 1, "single invariant factor");
                  t.expect(factors[0].is_equal(poly({1, 0, 1})), "invariant factor = x^2 + 1");
                  t.expect(nimblecas::rational_canonical_form(A).value().is_equal(A),
                           "RCF([[0,-1],[1,0]]) = the matrix itself");
              })
        .test("identity_has_repeated_linear_invariant_factors",
              [](TestContext& t) {
                  // The 2x2 identity is diagonalisable with a double eigenvalue 1: invariant
                  // factors (x-1), (x-1); minimal polynomial x-1 (not (x-1)^2). RCF == I.
                  const auto I = Matrix::identity(2);
                  auto factors = nimblecas::invariant_factors(I).value();
                  t.expect(factors.size() == 2, "two invariant factors for I_2");
                  t.expect(factors[0].is_equal(poly({-1, 1})), "f_1 = x - 1");
                  t.expect(factors[1].is_equal(poly({-1, 1})), "f_2 = x - 1");
                  t.expect(nimblecas::minimal_polynomial(I).value().is_equal(poly({-1, 1})),
                           "minimal polynomial of I_2 = x - 1");
                  t.expect(nimblecas::rational_canonical_form(I).value().is_equal(I),
                           "RCF(I_2) = I_2");
              })
        .test("product_of_invariant_factors_equals_characteristic_polynomial",
              [](TestContext& t) {
                  // Independent cross-check (Rule 32 honesty path): the product of the
                  // invariant factors must equal the Faddeev-LeVerrier characteristic
                  // polynomial exactly, for several distinct matrices.
                  const std::vector<Matrix> mats = {
                      mat({{ri(2), ri(1)}, {ri(0), ri(2)}}),            // Jordan block, char (x-2)^2
                      mat({{ri(1), ri(2), ri(3)},
                           {ri(0), ri(4), ri(5)},
                           {ri(0), ri(0), ri(6)}}),                     // upper triangular
                      mat({{ri(0), ri(-1)}, {ri(1), ri(0)}}),           // char x^2 + 1
                      mat({{ri(2), ri(0), ri(0)},
                           {ri(0), ri(2), ri(0)},
                           {ri(0), ri(0), ri(3)}}),                     // diag(2,2,3)
                  };
                  for (std::size_t k = 0; k < mats.size(); ++k) {
                      auto factors = nimblecas::invariant_factors(mats[k]).value();
                      auto charpoly = nimblecas::characteristic_polynomial(mats[k]).value();
                      t.expect(product(factors).is_equal(charpoly),
                               "prod(invariant factors) == characteristic polynomial");
                      // The minimal polynomial (last factor) always divides the char poly.
                      auto mp = nimblecas::minimal_polynomial(mats[k]).value();
                      t.expect(charpoly.divide(mp).value().remainder.is_zero(),
                               "minimal polynomial divides the characteristic polynomial");
                  }
                  // The Jordan block [[2,1],[0,2]] is non-derogatory: a single invariant
                  // factor (x-2)^2, so its minimal polynomial is the full char poly.
                  auto jf = nimblecas::invariant_factors(mats[0]).value();
                  t.expect(jf.size() == 1, "Jordan block has a single invariant factor");
                  t.expect(jf[0].is_equal(poly({4, -4, 1})), "invariant factor = (x-2)^2");
              })
        .test("non_square_is_domain_error",
              [](TestContext& t) {
                  const auto rect = mat({{ri(1), ri(2), ri(3)}, {ri(4), ri(5), ri(6)}});
                  t.expect(nimblecas::invariant_factors(rect).error() == MathError::domain_error,
                           "invariant_factors of a 2x3 => domain_error");
                  t.expect(nimblecas::minimal_polynomial(rect).error() == MathError::domain_error,
                           "minimal_polynomial of a 2x3 => domain_error");
                  t.expect(nimblecas::rational_canonical_form(rect).error() ==
                               MathError::domain_error,
                           "rational_canonical_form of a 2x3 => domain_error");
              })
        .run();
}
