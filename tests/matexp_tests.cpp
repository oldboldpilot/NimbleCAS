// Tests for nimblecas.matexp: exact-rational matrix exponentials (ROADMAP 7.2/7.19).
// @author Olumuyiwa Oluwasanmi
//
// Every case is chosen so the exact answer is well-defined over Q: e^0 = I, and nilpotent
// inputs whose Taylor series terminate (so Taylor, Pade, and scaling-and-squaring must all
// reproduce the same exact matrix). The one non-terminating check (Taylor of diag(1,0))
// pins the truncated partial sum entrywise as exact Rationals.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.matexp;
import nimblecas.testing;

using nimblecas::is_nilpotent;
using nimblecas::MathError;
using nimblecas::Matrix;
using nimblecas::matrix_exp;
using nimblecas::matrix_exp_pade;
using nimblecas::matrix_exp_taylor;
using nimblecas::Rational;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto ri(std::int64_t v) -> Rational {
    return Rational::from_int(v);
}

[[nodiscard]] auto rr(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

// Build a Matrix from integer rows (low-index row first).
[[nodiscard]] auto mat(std::vector<std::vector<std::int64_t>> rows) -> Matrix {
    std::vector<std::vector<Rational>> r;
    r.reserve(rows.size());
    for (const auto& row : rows) {
        std::vector<Rational> rr_row;
        rr_row.reserve(row.size());
        for (const std::int64_t v : row) {
            rr_row.push_back(Rational::from_int(v));
        }
        r.push_back(std::move(rr_row));
    }
    return Matrix::from_rows(std::move(r)).value();
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.matexp")
        .test("exp_zero_is_identity_taylor",
              [](TestContext& t) {
                  // e^0 = I exactly (only the k=0 term survives; the rest are powers of 0).
                  auto e = matrix_exp_taylor(Matrix::zero(2, 2), 5);
                  t.expect(e.has_value(), "taylor of the zero matrix succeeds");
                  t.expect(e && e->is_equal(Matrix::identity(2)), "e^0 = I via taylor");
              })
        .test("exp_zero_is_identity_pade",
              [](TestContext& t) {
                  // 0 is nilpotent, so the Pade approximant reproduces I exactly.
                  auto e = matrix_exp_pade(Matrix::zero(2, 2), 2);
                  t.expect(e.has_value(), "pade of the zero matrix succeeds");
                  t.expect(e && e->is_equal(Matrix::identity(2)), "e^0 = I via pade");
              })
        .test("exp_zero_is_identity_scaling",
              [](TestContext& t) {
                  // Scaling-and-squaring of 0: B = 0, e^B = I, and I^(2^s) = I.
                  auto e = matrix_exp(Matrix::zero(2, 2), 6, 4);
                  t.expect(e.has_value(), "scaling-and-squaring of the zero matrix succeeds");
                  t.expect(e && e->is_equal(Matrix::identity(2)), "e^0 = I via matrix_exp");
              })
        .test("nilpotent_2x2_exact_all_methods",
              [](TestContext& t) {
                  // N = [[0,1],[0,0]] has N^2 = 0, so e^N = I + N = [[1,1],[0,1]] EXACTLY.
                  // The Taylor series terminates after two terms; the [q/q] Pade numerator and
                  // denominator both truncate to I +/- (1/2)N and, since N^2 = 0,
                  //   D^{-1} N_mat = (I + N/2)(I + N/2) = I + N   (the N^2 term vanishes),
                  // so Pade for any q >= 1 -- and scaling-and-squaring, which squares a
                  // nilpotent-exact factor -- reproduce I + N exactly. Verified by hand.
                  const Matrix n = mat({{0, 1}, {0, 0}});
                  const Matrix expected = mat({{1, 1}, {0, 1}});

                  auto taylor2 = matrix_exp_taylor(n, 2);
                  t.expect(taylor2 && taylor2->is_equal(expected), "taylor(N, 2) = I + N");
                  auto taylor5 = matrix_exp_taylor(n, 5);
                  t.expect(taylor5 && taylor5->is_equal(expected),
                           "taylor(N, 5) = I + N (extra terms vanish)");
                  auto pade1 = matrix_exp_pade(n, 1);
                  t.expect(pade1 && pade1->is_equal(expected), "pade(N, 1) = I + N");
                  auto pade3 = matrix_exp_pade(n, 3);
                  t.expect(pade3 && pade3->is_equal(expected), "pade(N, 3) = I + N");
                  auto scaled = matrix_exp(n, 2, 3);
                  t.expect(scaled && scaled->is_equal(expected), "matrix_exp(N, 2, 3) = I + N");
              })
        .test("nilpotent_3x3_taylor_exact",
              [](TestContext& t) {
                  // J = [[0,1,0],[0,0,1],[0,0,0]] has J^3 = 0, so
                  //   e^J = I + J + J^2/2 = [[1,1,1/2],[0,1,1],[0,0,1]] exactly.
                  // (J^2 = [[0,0,1],[0,0,0],[0,0,0]].) Any terms past k=2 vanish.
                  const Matrix j = mat({{0, 1, 0}, {0, 0, 1}, {0, 0, 0}});
                  std::vector<std::vector<Rational>> exp_rows = {
                      {ri(1), ri(1), rr(1, 2)},
                      {ri(0), ri(1), ri(1)},
                      {ri(0), ri(0), ri(1)},
                  };
                  const Matrix expected = Matrix::from_rows(std::move(exp_rows)).value();

                  auto taylor3 = matrix_exp_taylor(j, 3);
                  t.expect(taylor3 && taylor3->is_equal(expected),
                           "taylor(J, 3) = I + J + J^2/2");
                  auto taylor6 = matrix_exp_taylor(j, 6);
                  t.expect(taylor6 && taylor6->is_equal(expected),
                           "taylor(J, 6) = same (extra terms vanish)");
              })
        .test("is_nilpotent_true_cases",
              [](TestContext& t) {
                  auto a = is_nilpotent(mat({{0, 1}, {0, 0}}));
                  t.expect(a.has_value() && a.value_or(false), "[[0,1],[0,0]] is nilpotent");
                  auto b = is_nilpotent(mat({{0, 1, 0}, {0, 0, 1}, {0, 0, 0}}));
                  t.expect(b.has_value() && b.value_or(false),
                           "the 3x3 Jordan nilpotent is nilpotent");
              })
        .test("is_nilpotent_false_cases",
              [](TestContext& t) {
                  auto a = is_nilpotent(Matrix::identity(2));
                  t.expect(a.has_value() && !a.value_or(true), "identity is not nilpotent");
                  auto b = is_nilpotent(mat({{1, 1}, {0, 1}}));
                  t.expect(b.has_value() && !b.value_or(true),
                           "[[1,1],[0,1]] is not nilpotent");
              })
        .test("is_nilpotent_shape_error",
              [](TestContext& t) {
                  auto ns = is_nilpotent(mat({{1, 2, 3}, {4, 5, 6}}));
                  t.expect(!ns.has_value() && ns.error() == MathError::domain_error,
                           "non-square => domain_error");
              })
        .test("taylor_value_check_diag_1_0",
              [](TestContext& t) {
                  // diag(1, 0), terms = 4: entry (0,0) = Sum_{k=0}^{3} 1/k! = 1 + 1 + 1/2 + 1/6
                  // = 8/3; entry (1,1) = 1 (only the k=0 term is nonzero there). Exact check.
                  auto e = matrix_exp_taylor(mat({{1, 0}, {0, 0}}), 4);
                  t.expect(e.has_value(), "taylor(diag(1,0), 4) succeeds");
                  if (e) {
                      t.expect(e->at(0, 0) == rr(8, 3), "(0,0) = 1 + 1 + 1/2 + 1/6 = 8/3");
                      t.expect(e->at(1, 1) == ri(1), "(1,1) = 1 exactly");
                  }
              })
        .test("taylor_pade_agree_on_nilpotent",
              [](TestContext& t) {
                  // On the nilpotent N both routes are exact, hence must agree. (We do NOT
                  // compare them on a non-nilpotent matrix, where they are different rational
                  // approximations of the transcendental e^A.)
                  const Matrix n = mat({{0, 1}, {0, 0}});
                  auto taylor = matrix_exp_taylor(n, 4);
                  auto pade = matrix_exp_pade(n, 2);
                  t.expect(taylor.has_value() && pade.has_value(), "both routes succeed");
                  t.expect(taylor && pade && taylor->is_equal(*pade),
                           "taylor and pade agree exactly on the nilpotent N");
              })
        .test("pade_exactness_boundary_on_high_index_nilpotent",
              [](TestContext& t) {
                  // The [q/q] Pade approximant matches e^x only through order x^{2q}, so on a
                  // nilpotent A it is exact ONLY when the nilpotency index m <= 2q+1. The 4x4
                  // Jordan block J4 has J4^4 = 0 (index m = 4), and the true exponential is
                  //   e^{J4} = I + J4 + J4^2/2 + J4^3/6.
                  // With q = 1 (2q+1 = 3 < 4) the J4^3 coefficient comes out 1/4, NOT 1/6, so
                  // Pade must DISAGREE with the exact Taylor value. With q = 2 (2q+1 = 5 >= 4)
                  // the approximant is exact and the two must AGREE. This pins the corrected
                  // "exact for nilpotent iff m <= 2q+1" contract as a regression.
                  const Matrix j4 = mat({{0, 1, 0, 0},
                                         {0, 0, 1, 0},
                                         {0, 0, 0, 1},
                                         {0, 0, 0, 0}});
                  auto exact = matrix_exp_taylor(j4, 4);  // terms = index => exact e^{J4}
                  t.expect(exact.has_value(), "taylor(J4, 4) succeeds");
                  // The true (0,3) entry is J4^3/6 = 1/6.
                  t.expect(exact && exact->at(0, 3) == rr(1, 6), "exact (0,3) = 1/6");

                  auto pade_low = matrix_exp_pade(j4, 1);  // 2q+1 = 3 < 4: NOT exact
                  t.expect(pade_low.has_value(), "pade(J4, 1) succeeds");
                  t.expect(pade_low && pade_low->at(0, 3) == rr(1, 4),
                           "pade(J4, 1) (0,3) = 1/4 (approximation, not 1/6)");
                  t.expect(exact && pade_low && !exact->is_equal(*pade_low),
                           "pade(J4, 1) disagrees with exact e^{J4} (index 4 > 2q+1 = 3)");

                  auto pade_ok = matrix_exp_pade(j4, 2);  // 2q+1 = 5 >= 4: exact
                  t.expect(pade_ok.has_value(), "pade(J4, 2) succeeds");
                  t.expect(exact && pade_ok && exact->is_equal(*pade_ok),
                           "pade(J4, 2) is exact (index 4 <= 2q+1 = 5)");
              })
        .test("domain_errors",
              [](TestContext& t) {
                  // terms < 1 => domain_error.
                  auto bad_terms = matrix_exp_taylor(Matrix::identity(2), 0);
                  t.expect(!bad_terms.has_value() &&
                               bad_terms.error() == MathError::domain_error,
                           "terms < 1 => domain_error");
                  // q < 1 => domain_error (pade and matrix_exp).
                  auto bad_q_pade = matrix_exp_pade(Matrix::identity(2), 0);
                  t.expect(!bad_q_pade.has_value() &&
                               bad_q_pade.error() == MathError::domain_error,
                           "pade q < 1 => domain_error");
                  auto bad_q_exp = matrix_exp(Matrix::identity(2), 0, 2);
                  t.expect(!bad_q_exp.has_value() &&
                               bad_q_exp.error() == MathError::domain_error,
                           "matrix_exp q < 1 => domain_error");
                  // Non-square => domain_error on each entry point.
                  const Matrix ns = mat({{1, 2, 3}, {4, 5, 6}});
                  auto e_taylor = matrix_exp_taylor(ns, 3);
                  t.expect(!e_taylor.has_value() &&
                               e_taylor.error() == MathError::domain_error,
                           "non-square taylor => domain_error");
                  auto e_pade = matrix_exp_pade(ns, 2);
                  t.expect(!e_pade.has_value() && e_pade.error() == MathError::domain_error,
                           "non-square pade => domain_error");
                  auto e_exp = matrix_exp(ns, 2, 1);
                  t.expect(!e_exp.has_value() && e_exp.error() == MathError::domain_error,
                           "non-square matrix_exp => domain_error");
              })
        .run();
}
