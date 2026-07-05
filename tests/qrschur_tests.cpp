// Tests for nimblecas.qrschur: exact Gram-Schmidt orthogonal decomposition over Q (Qᵀ·Q
// diagonal, Q·R == A exactly), numeric Householder QR (orthonormal, small residual), and
// numeric real Schur form (A = Q·T·Qᵀ, quasi-triangular T whose eigenvalues match numeigen).
// @author Olumuyiwa Oluwasanmi
//
// The EXACT path (exact_orthogonal_qr) is checked with hand-verified rational values and an
// identical reconstruction Q·R == A over Q. The NUMERIC paths are checked by their residuals
// (‖Q·R − A‖_F, ‖Q·T·Qᵀ − A‖_F, ‖QᵀQ − I‖_F) staying below 1e-9, and by comparing the Schur
// spectrum against nimblecas.numeigen::eigenvalues_qr as an order-independent multiset.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.numeigen;
import nimblecas.qrschur;
import nimblecas.testing;

using nimblecas::exact_orthogonal_qr;
using nimblecas::MathError;
using nimblecas::Matrix;
using nimblecas::numeric_qr;
using nimblecas::orthonormality_defect;
using nimblecas::qr_residual;
using nimblecas::Rational;
using nimblecas::real_schur;
using nimblecas::schur_eigenvalues;
using nimblecas::schur_residual;
using nimblecas::eigenvalues_qr;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

using cd = std::complex<double>;

namespace {

[[nodiscard]] auto ri(std::int64_t v) -> Rational {
    return Rational::from_int(v);
}

[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
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

// Multiset match up to ordering and tolerance (greedy pairing), mirroring numeigen_tests.
[[nodiscard]] auto approx_set(std::span<const cd> got, std::span<const cd> expected, double tol)
    -> bool {
    if (got.size() != expected.size()) {
        return false;
    }
    std::vector<bool> used(got.size(), false);
    for (const cd& e : expected) {
        bool matched = false;
        for (std::size_t i = 0; i < got.size(); ++i) {
            if (!used[i] && std::abs(got[i] - e) <= tol) {
                used[i] = true;
                matched = true;
                break;
            }
        }
        if (!matched) {
            return false;
        }
    }
    return true;
}

// Is `q` (m×n rational Matrix) column-orthogonal, i.e. Qᵀ·Q diagonal? Exact over Q.
[[nodiscard]] auto gram_is_diagonal(const Matrix& q) -> bool {
    const auto qt = q.transpose().value();
    const auto g = qt.multiply(q).value();  // n×n, exact
    for (std::size_t i = 0; i < g.rows(); ++i) {
        for (std::size_t j = 0; j < g.cols(); ++j) {
            if (i != j && !g.at(i, j).is_zero()) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.qrschur")
        .test("exact_gram_schmidt_square_reconstructs_over_Q",
              [](TestContext& t) {
                  // A = [[2,1],[1,1]]. By hand: q1=(2,1), ⟨q1,q1⟩=5; r12=⟨a2,q1⟩/5=3/5;
                  // q2 = a2 - (3/5)q1 = (-1/5, 2/5), ⟨q2,q2⟩=1/5.
                  const auto a = mat({{ri(2), ri(1)}, {ri(1), ri(1)}});
                  const auto d = exact_orthogonal_qr(a).value();
                  // R is upper-triangular with unit diagonal, r12 = 3/5.
                  t.expect(d.r.at(0, 0) == ri(1) && d.r.at(1, 1) == ri(1), "R unit diagonal");
                  t.expect(d.r.at(1, 0) == ri(0), "R strictly lower entry is 0");
                  t.expect(d.r.at(0, 1) == rat(3, 5), "r12 = 3/5 exactly");
                  // Q columns as computed by hand.
                  t.expect(d.q.at(0, 0) == ri(2) && d.q.at(1, 0) == ri(1), "q1 = (2,1)");
                  t.expect(d.q.at(0, 1) == rat(-1, 5) && d.q.at(1, 1) == rat(2, 5),
                           "q2 = (-1/5, 2/5)");
                  // Qᵀ·Q is diagonal (columns orthogonal) with the exact pseudo-norms 5, 1/5.
                  const auto g = d.q.transpose().value().multiply(d.q).value();
                  t.expect(gram_is_diagonal(d.q), "QᵀQ is diagonal");
                  t.expect(g.at(0, 0) == ri(5) && g.at(1, 1) == rat(1, 5),
                           "diagonal pseudo-norms = 5, 1/5");
                  // Q·R reconstructs A identically over Q.
                  t.expect(d.q.multiply(d.r).value().is_equal(a), "Q·R == A exactly");
              })
        .test("exact_gram_schmidt_tall_matrix",
              [](TestContext& t) {
                  // A = [[1,0],[0,1],[1,1]] (3×2). q1=(1,0,1), ⟨q1,q1⟩=2; r12=1/2;
                  // q2 = a2 - (1/2)q1 = (-1/2, 1, 1/2), ⟨q2,q2⟩=3/2.
                  const auto a = mat({{ri(1), ri(0)}, {ri(0), ri(1)}, {ri(1), ri(1)}});
                  const auto d = exact_orthogonal_qr(a).value();
                  t.expect(d.r.at(0, 1) == rat(1, 2), "r12 = 1/2 exactly");
                  t.expect(gram_is_diagonal(d.q), "QᵀQ diagonal for the tall case");
                  const auto g = d.q.transpose().value().multiply(d.q).value();
                  t.expect(g.at(0, 0) == ri(2) && g.at(1, 1) == rat(3, 2),
                           "pseudo-norms = 2, 3/2");
                  t.expect(d.q.multiply(d.r).value().is_equal(a), "Q·R == A exactly (3×2)");
              })
        .test("exact_gram_schmidt_rank_deficient_is_domain_error",
              [](TestContext& t) {
                  // Columns of [[1,2],[2,4]] are dependent (a2 = 2·a1): q2 = 0, zero
                  // pseudo-norm => honest domain_error, never a bogus factor.
                  const auto a = mat({{ri(1), ri(2)}, {ri(2), ri(4)}});
                  t.expect(exact_orthogonal_qr(a).error() == MathError::domain_error,
                           "rank-deficient A => domain_error");
                  // A wide 2×3 matrix cannot have 3 independent columns in R^2 either.
                  const auto wide = mat({{ri(1), ri(0), ri(1)}, {ri(0), ri(1), ri(1)}});
                  t.expect(exact_orthogonal_qr(wide).error() == MathError::domain_error,
                           "wide (m<n) A => domain_error");
              })
        .test("numeric_qr_residual_and_orthonormality",
              [](TestContext& t) {
                  // The classic 3×3 example; Householder QR must reconstruct it and Q must
                  // be orthonormal, both to well under 1e-9.
                  const auto a = mat({{ri(12), ri(-51), ri(4)},
                                      {ri(6), ri(167), ri(-68)},
                                      {ri(-4), ri(24), ri(-41)}});
                  const auto ad = to_doubles(a);
                  const auto d = numeric_qr(ad, 3, 3).value();
                  t.expect(qr_residual(d, ad).value() < 1e-9, "‖Q·R − A‖_F < 1e-9");
                  t.expect(orthonormality_defect(d.q, 3, 3).value() < 1e-9,
                           "‖QᵀQ − I‖_F < 1e-9");
                  // R is upper-triangular (strictly-lower entries are exactly zero).
                  t.expect(d.r[1 * 3 + 0] == 0.0 && d.r[2 * 3 + 0] == 0.0 && d.r[2 * 3 + 1] == 0.0,
                           "R strictly-lower triangle is exactly 0");
                  // A tall 3×2 QR also reconstructs and stays orthonormal.
                  const auto b = mat({{ri(1), ri(2)}, {ri(3), ri(4)}, {ri(5), ri(6)}});
                  const auto bd = to_doubles(b);
                  const auto db = numeric_qr(bd, 3, 2).value();
                  t.expect(qr_residual(db, bd).value() < 1e-9, "tall ‖Q·R − A‖_F < 1e-9");
                  t.expect(orthonormality_defect(db.q, 3, 3).value() < 1e-9,
                           "tall Q orthonormal");
                  // Dimension mismatch is an honest domain_error (not a silent guess).
                  t.expect(numeric_qr(ad, 2, 2).error() == MathError::domain_error,
                           "wrong dimensions => domain_error");
              })
        .test("real_schur_residual_and_eigenvalues_match_numeigen",
              [](TestContext& t) {
                  // Nonsymmetric 3×3 with eigenvalues 3 and 2 ± i (a real + a complex pair).
                  const auto a = mat({{ri(2), ri(-1), ri(0)},
                                      {ri(1), ri(2), ri(0)},
                                      {ri(0), ri(0), ri(3)}});
                  const auto ad = to_doubles(a);
                  const auto s = real_schur(ad, 3).value();
                  t.expect(schur_residual(s, ad).value() < 1e-9, "‖Q·T·Qᵀ − A‖_F < 1e-9");
                  t.expect(orthonormality_defect(s.q, 3, 3).value() < 1e-9,
                           "Schur Q is orthogonal");
                  const auto got = schur_eigenvalues(s).value();
                  const auto ref = eigenvalues_qr(ad, 3).value();
                  t.expect(approx_set(got, ref, 1e-6),
                           "T's eigenvalues match numeigen::eigenvalues_qr");
                  const std::vector<cd> expected = {cd{3.0, 0.0}, cd{2.0, 1.0}, cd{2.0, -1.0}};
                  t.expect(approx_set(got, expected, 1e-6), "eigenvalues are {3, 2±i}");
              })
        .test("real_schur_rotation_and_symmetric",
              [](TestContext& t) {
                  // Pure rotation [[0,-1],[1,0]] has eigenvalues ± i; the 2×2 block stays.
                  const auto rot = mat({{ri(0), ri(-1)}, {ri(1), ri(0)}});
                  const auto rd = to_doubles(rot);
                  const auto sr = real_schur(rd, 2).value();
                  t.expect(schur_residual(sr, rd).value() < 1e-9, "rotation ‖Q·T·Qᵀ − A‖ small");
                  const std::vector<cd> pmi = {cd{0.0, 1.0}, cd{0.0, -1.0}};
                  t.expect(approx_set(schur_eigenvalues(sr).value(), pmi, 1e-9),
                           "rotation eigenvalues = ± i");
                  // Symmetric [[2,1],[1,2]] => real eigenvalues 1, 3; T is upper-triangular.
                  const auto sym = mat({{ri(2), ri(1)}, {ri(1), ri(2)}});
                  const auto sd = to_doubles(sym);
                  const auto ss = real_schur(sd, 2).value();
                  t.expect(schur_residual(ss, sd).value() < 1e-9, "symmetric residual small");
                  const std::vector<cd> real13 = {cd{1.0, 0.0}, cd{3.0, 0.0}};
                  const auto se = schur_eigenvalues(ss).value();
                  t.expect(approx_set(se, real13, 1e-9), "symmetric eigenvalues = {1,3}");
                  // Real Schur form of a symmetric matrix is (numerically) diagonal: the
                  // subdiagonal deflated, so schur_eigenvalues sees two 1×1 blocks.
                  t.expect(std::abs(se[0].imag()) < 1e-12 && std::abs(se[1].imag()) < 1e-12,
                           "symmetric spectrum is real");
              })
        .test("real_schur_larger_general_matrix",
              [](TestContext& t) {
                  // A 4×4 NONsymmetric block-upper-triangular matrix with two complex pairs:
                  // top block [[3,-2],[4,-1]] => 1 ± 2i, bottom [[2,-1],[1,2]] => 2 ± i. The
                  // off-diagonal coupling makes the Hessenberg reduction and bulge chase
                  // nontrivial; the whole spectrum is complex.
                  const auto a = mat({{ri(3), ri(-2), ri(1), ri(0)},
                                      {ri(4), ri(-1), ri(0), ri(1)},
                                      {ri(0), ri(0), ri(2), ri(-1)},
                                      {ri(0), ri(0), ri(1), ri(2)}});
                  const auto ad = to_doubles(a);
                  const auto s = real_schur(ad, 4).value();
                  t.expect(schur_residual(s, ad).value() < 1e-9, "4×4 ‖Q·T·Qᵀ − A‖_F < 1e-9");
                  t.expect(orthonormality_defect(s.q, 4, 4).value() < 1e-9, "4×4 Q orthogonal");
                  const auto got = schur_eigenvalues(s).value();
                  t.expect(approx_set(got, eigenvalues_qr(ad, 4).value(), 1e-6),
                           "4×4 Schur spectrum matches numeigen");
                  const std::vector<cd> expected = {cd{1.0, 2.0}, cd{1.0, -2.0}, cd{2.0, 1.0},
                                                    cd{2.0, -1.0}};
                  t.expect(approx_set(got, expected, 1e-6), "spectrum = {1±2i, 2±i}");
              })
        .test("real_schur_dimension_mismatch_is_domain_error",
              [](TestContext& t) {
                  const std::vector<double> three = {1.0, 2.0, 3.0};
                  t.expect(real_schur(three, 2).error() == MathError::domain_error,
                           "size != n·n => domain_error");
                  // Empty matrix is a valid empty decomposition, not an error.
                  t.expect(real_schur({}, 0).value().n == 0, "0×0 => empty decomposition");
              })
        .run();
}
