// Tests for nimblecas.svd: numeric thin SVD via one-sided Jacobi (A = U*diag(sigma)*V^T),
// numeric polar decomposition (A = U_p*P), and the exact companions gram_matrix /
// exact_singular_value_squares over Q.
// @author Olumuyiwa Oluwasanmi
//
// Every case is a hand-verified oracle. The NUMERIC path is checked by residual /
// orthonormality-defect invariants (never by literal U/V entries, since equal or zero
// singular values leave those bases genuinely non-unique) plus the hand-derived sigma
// values, which ARE determined uniquely. The EXACT path (gram_matrix,
// exact_singular_value_squares) is checked against hand-computed exact Rational values.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.qrschur;
import nimblecas.svd;
import nimblecas.testing;

using nimblecas::exact_singular_value_squares;
using nimblecas::gram_matrix;
using nimblecas::MathError;
using nimblecas::Matrix;
using nimblecas::orthonormality_defect;
using nimblecas::polar;
using nimblecas::polar_residual;
using nimblecas::Rational;
using nimblecas::svd;
using nimblecas::svd_residual;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto ri(std::int64_t v) -> Rational {
    return Rational::from_int(v);
}

[[nodiscard]] auto mat(std::vector<std::vector<Rational>> rows) -> Matrix {
    return Matrix::from_rows(std::move(rows)).value();
}

// Row-major double buffer of an m×n rational Matrix (for the numeric entry points).
[[nodiscard]] auto to_doubles(const Matrix& a) -> std::vector<double> {
    std::vector<double> out;
    out.reserve(a.rows() * a.cols());
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.cols(); ++j) {
            out.push_back(static_cast<double>(a.at(i, j).numerator()) /
                          static_cast<double>(a.at(i, j).denominator()));
        }
    }
    return out;
}

[[nodiscard]] auto frob_norm(std::span<const double> a) -> double {
    double acc = 0.0;
    for (double v : a) {
        acc += v * v;
    }
    return std::sqrt(acc);
}

[[nodiscard]] auto sigma_descending_nonneg(const std::vector<double>& s) -> bool {
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] < -1e-9) {
            return false;
        }
        if (i + 1 < s.size() && s[i] < s[i + 1] - 1e-9) {
            return false;
        }
    }
    return true;
}

// The largest-magnitude entry of every column of the rows×k row-major `u` (lowest index
// on ties) must be non-negative -- the sign convention that eliminates the +/- ambiguity.
[[nodiscard]] auto sign_convention_holds(std::span<const double> u, std::size_t rows,
                                        std::size_t k) -> bool {
    for (std::size_t j = 0; j < k; ++j) {
        std::size_t best_row = 0;
        double best_mag = -1.0;
        for (std::size_t i = 0; i < rows; ++i) {
            const double mag = std::fabs(u[i * k + j]);
            if (mag > best_mag) {
                best_mag = mag;
                best_row = i;
            }
        }
        if (u[best_row * k + j] < -1e-9) {
            return false;
        }
    }
    return true;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.svd")
        .test("svd_diagonal_zero_singular_value_and_sign_flip",
              [](TestContext& t) {
                  // diag(3,-2,0): sigma = (3,2,0). Exercises a zero singular value (the
                  // 3rd U column is still unit & orthogonal despite sigma3=0) and a sign
                  // flip (the -2 diagonal entry forces a column negation to restore the
                  // sign convention).
                  const auto a = mat({{ri(3), ri(0), ri(0)},
                                      {ri(0), ri(-2), ri(0)},
                                      {ri(0), ri(0), ri(0)}});
                  const auto ad = to_doubles(a);
                  const auto d = svd(ad, 3, 3).value();
                  t.expect(d.singular_values.size() == 3, "k = min(3,3) = 3");
                  t.expect(std::abs(d.singular_values[0] - 3.0) < 1e-9 &&
                               std::abs(d.singular_values[1] - 2.0) < 1e-9 &&
                               std::abs(d.singular_values[2]) < 1e-9,
                           "sigma = (3, 2, 0)");
                  t.expect(sigma_descending_nonneg(d.singular_values), "sigma descending & >= 0");
                  const double nrm = frob_norm(ad);
                  t.expect(svd_residual(d, ad).value() <= 1e-10 * nrm, "residual <= 1e-10*||A||");
                  const std::size_t k = d.singular_values.size();
                  t.expect(orthonormality_defect(d.u, d.rows, k).value() <= 1e-10,
                           "U orthonormal, including the zero-sigma column");
                  t.expect(orthonormality_defect(d.v, d.cols, k).value() <= 1e-10,
                           "V orthonormal");
                  t.expect(sign_convention_holds(d.u, d.rows, k), "sign convention holds");
              })
        .test("svd_two_by_two_matches_hand_eigenvalues",
              [](TestContext& t) {
                  // [[3,0],[4,5]]: A^T*A = [[25,20],[20,25]], eigenvalues 45 and 5, so
                  // sigma = (3*sqrt5, sqrt5) and sigma1*sigma2 = 15 = |det A|.
                  const auto a = mat({{ri(3), ri(0)}, {ri(4), ri(5)}});
                  const auto ad = to_doubles(a);
                  const auto d = svd(ad, 2, 2).value();
                  const double s1 = 3.0 * std::sqrt(5.0);
                  const double s2 = std::sqrt(5.0);
                  t.expect(std::abs(d.singular_values[0] - s1) < 1e-6, "sigma1 = 3*sqrt(5)");
                  t.expect(std::abs(d.singular_values[1] - s2) < 1e-6, "sigma2 = sqrt(5)");
                  t.expect(std::abs(d.singular_values[0] * d.singular_values[1] - 15.0) < 1e-6,
                           "sigma1*sigma2 = |det A| = 15");
                  t.expect(sigma_descending_nonneg(d.singular_values), "sigma descending & >= 0");
                  const double nrm = frob_norm(ad);
                  t.expect(svd_residual(d, ad).value() <= 1e-10 * nrm, "residual small");
                  t.expect(orthonormality_defect(d.u, d.rows, 2).value() <= 1e-10,
                           "U orthonormal");
                  t.expect(orthonormality_defect(d.v, d.cols, 2).value() <= 1e-10,
                           "V orthonormal");
                  t.expect(sign_convention_holds(d.u, d.rows, 2), "sign convention holds");

                  // exact_singular_value_squares offers NO exact sigma, only sigma^2.
                  const auto sq = exact_singular_value_squares(a).value();
                  t.expect(sq.size() == 2, "two exact sigma^2 values");
                  t.expect(sq[0].first == ri(45) && sq[0].second == 1,
                           "sigma^2 = 45 (mult 1), first (descending)");
                  t.expect(sq[1].first == ri(5) && sq[1].second == 1,
                           "sigma^2 = 5 (mult 1), second");
              })
        .test("svd_rank_one_matrix_zero_singular_value",
              [](TestContext& t) {
                  // [[1,2],[2,4]] is rank 1: A^T*A = [[5,10],[10,20]], sigma = (5, 0).
                  const auto a = mat({{ri(1), ri(2)}, {ri(2), ri(4)}});
                  const auto ad = to_doubles(a);
                  const auto d = svd(ad, 2, 2).value();
                  t.expect(std::abs(d.singular_values[0] - 5.0) < 1e-6, "sigma1 = 5");
                  t.expect(std::abs(d.singular_values[1]) < 1e-6, "sigma2 = 0");
                  t.expect(sigma_descending_nonneg(d.singular_values), "sigma descending & >= 0");
                  const double nrm = frob_norm(ad);
                  t.expect(svd_residual(d, ad).value() <= 1e-10 * nrm,
                           "residual small despite rank deficiency");
                  t.expect(orthonormality_defect(d.u, d.rows, 2).value() <= 1e-10,
                           "U column 2 orthonormal despite sigma2 = 0");
                  t.expect(orthonormality_defect(d.v, d.cols, 2).value() <= 1e-10,
                           "V orthonormal");

                  const auto sq = exact_singular_value_squares(a).value();
                  t.expect(sq.size() == 2, "two exact sigma^2 values");
                  t.expect(sq[0].first == ri(25) && sq[0].second == 1, "sigma^2 = 25, first");
                  t.expect(sq[1].first == ri(0) && sq[1].second == 1, "sigma^2 = 0, second");
              })
        .test("svd_rectangular_duality_transpose_swaps_u_and_v",
              [](TestContext& t) {
                  // A (3x2) and A^T (2x3) must yield identical sigma, with U and V swapped.
                  const auto a = mat({{ri(1), ri(2)}, {ri(3), ri(4)}, {ri(5), ri(6)}});
                  const auto at = a.transpose().value();
                  const auto ad = to_doubles(a);
                  const auto atd = to_doubles(at);
                  const auto d = svd(ad, 3, 2).value();
                  const auto dt = svd(atd, 2, 3).value();

                  t.expect(d.singular_values.size() == dt.singular_values.size(), "same k");
                  for (std::size_t i = 0; i < d.singular_values.size(); ++i) {
                      t.expect(std::abs(d.singular_values[i] - dt.singular_values[i]) < 1e-9,
                               "identical sigma between A and A^T");
                  }
                  t.expect(dt.u.size() == d.v.size(), "svd(A^T).u has svd(A).v's shape");
                  for (std::size_t i = 0; i < dt.u.size(); ++i) {
                      t.expect(std::abs(dt.u[i] - d.v[i]) < 1e-9,
                               "svd(A^T).u == svd(A).v entrywise");
                  }
                  t.expect(dt.v.size() == d.u.size(), "svd(A^T).v has svd(A).u's shape");
                  for (std::size_t i = 0; i < dt.v.size(); ++i) {
                      t.expect(std::abs(dt.v[i] - d.u[i]) < 1e-9,
                               "svd(A^T).v == svd(A).u entrywise");
                  }
                  t.expect(svd_residual(d, ad).value() <= 1e-10 * frob_norm(ad), "A residual small");
                  t.expect(svd_residual(dt, atd).value() <= 1e-10 * frob_norm(atd),
                           "A^T residual small");
              })
        .test("polar_decomposition_spd_and_rotation_identity",
              [](TestContext& t) {
                  const auto a = mat({{ri(3), ri(0)}, {ri(4), ri(5)}});
                  const auto ad = to_doubles(a);
                  const auto d = polar(ad, 2).value();
                  t.expect(orthonormality_defect(d.u, d.n, d.n).value() <= 1e-10,
                           "U_p orthogonal");
                  t.expect(std::abs(d.p[0 * 2 + 1] - d.p[1 * 2 + 0]) < 1e-9, "P symmetric");
                  // A 2x2 symmetric matrix is SPD iff trace>0 and det>0.
                  const double tr_p = d.p[0] + d.p[3];
                  const double det_p = d.p[0] * d.p[3] - d.p[1] * d.p[2];
                  t.expect(tr_p > 0.0 && det_p > 0.0, "P is SPD (trace>0, det>0)");
                  t.expect(polar_residual(d, ad).value() <= 1e-10 * frob_norm(ad),
                           "||U_p*P - A||_F small");

                  // ||P*P - A^T*A||_F <= 1e-9 against gram_matrix converted to double.
                  const auto g = gram_matrix(a).value();
                  std::vector<double> gd(4);
                  for (std::size_t i = 0; i < 2; ++i) {
                      for (std::size_t j = 0; j < 2; ++j) {
                          const Rational& r = g.at(i, j);
                          gd[i * 2 + j] = static_cast<double>(r.numerator()) /
                                          static_cast<double>(r.denominator());
                      }
                  }
                  std::vector<double> pp(4, 0.0);
                  for (std::size_t i = 0; i < 2; ++i) {
                      for (std::size_t j = 0; j < 2; ++j) {
                          double sum = 0.0;
                          for (std::size_t kk = 0; kk < 2; ++kk) {
                              sum += d.p[i * 2 + kk] * d.p[kk * 2 + j];
                          }
                          pp[i * 2 + j] = sum;
                      }
                  }
                  double diff = 0.0;
                  for (std::size_t idx = 0; idx < 4; ++idx) {
                      const double e = pp[idx] - gd[idx];
                      diff += e * e;
                  }
                  t.expect(std::sqrt(diff) <= 1e-9, "P*P == A^T*A (double) within 1e-9");

                  // Polar of a pure rotation is (A, I): sigma = (1,1) is DEGENERATE (equal),
                  // but U_p = U*V^T and P = V*diag(sigma)*V^T are basis-independent, so the
                  // result is unique even though the raw U/V columns are not.
                  const auto rot = mat({{ri(0), ri(-1)}, {ri(1), ri(0)}});
                  const auto rotd = to_doubles(rot);
                  const auto dr = polar(rotd, 2).value();
                  t.expect(polar_residual(dr, rotd).value() <= 1e-10, "U_p*P == the rotation");
                  for (std::size_t i = 0; i < 2; ++i) {
                      for (std::size_t j = 0; j < 2; ++j) {
                          const double target = (i == j) ? 1.0 : 0.0;
                          t.expect(std::abs(dr.p[i * 2 + j] - target) < 1e-9,
                                   "P = I for a pure rotation");
                          t.expect(std::abs(dr.u[i * 2 + j] - rotd[i * 2 + j]) < 1e-9,
                                   "U_p = A for a pure rotation");
                      }
                  }
              })
        .test("svd_polar_error_cases_and_empty",
              [](TestContext& t) {
                  const std::vector<double> three = {1.0, 2.0, 3.0};
                  t.expect(svd(three, 2, 2).error() == MathError::domain_error,
                           "svd size mismatch => domain_error");
                  // A 2x3 buffer fed through polar's square-only (single-n) entry point can
                  // only ever be a size mismatch against n*n, which is the honest signal for
                  // "this was not a square matrix".
                  const std::vector<double> six = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
                  t.expect(polar(six, 2).error() == MathError::domain_error,
                           "polar non-square (size mismatch) => domain_error");

                  const auto e = svd(std::span<const double>{}, 0, 0).value();
                  t.expect(e.rows == 0 && e.cols == 0 && e.singular_values.empty() &&
                               e.u.empty() && e.v.empty(),
                           "0x0 svd => empty decomposition");
                  const auto ep = polar(std::span<const double>{}, 0).value();
                  t.expect(ep.n == 0 && ep.u.empty() && ep.p.empty(),
                           "0x0 polar => empty decomposition");
              })
        .run();
}
