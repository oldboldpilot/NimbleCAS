// Tests for nimblecas.analysis: conditioning, convergence, and Lyapunov stability.
// @author Olumuyiwa Oluwasanmi
//
// Exact rational matrices are used so every condition number, Lyapunov solution, and
// Sylvester verdict is exact and deterministic. The numerical routines (kappa_2, the root
// test, Lyapunov exponents) are checked against known values within a tolerance.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.dynamics;
import nimblecas.analysis;
import nimblecas.testing;

using nimblecas::alternating_series_test;
using nimblecas::comparison_test;
using nimblecas::condition_1;
using nimblecas::condition_2_estimate;
using nimblecas::condition_inf;
using nimblecas::is_asymptotically_stable;
using nimblecas::is_positive_definite;
using nimblecas::is_stable_lyapunov;
using nimblecas::lyapunov_exponent;
using nimblecas::lyapunov_solve;
using nimblecas::MathError;
using nimblecas::Matrix;
using nimblecas::matrix_norm_1;
using nimblecas::matrix_norm_inf;
using nimblecas::Rational;
using nimblecas::ratio_test;
using nimblecas::root_test;
using nimblecas::stability_cross_check;
using nimblecas::stein_solve;
using nimblecas::Verdict;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto ri(std::int64_t v) -> Rational {
    return Rational::from_int(v);
}

[[nodiscard]] auto rr(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

// Build a Matrix from integer rows.
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

// Build a Matrix from Rational rows.
[[nodiscard]] auto matq(std::vector<std::vector<Rational>> rows) -> Matrix {
    return Matrix::from_rows(std::move(rows)).value();
}

// 2^n as an int64 (n small).
[[nodiscard]] auto pow2(std::int64_t n) -> std::int64_t {
    return static_cast<std::int64_t>(1) << n;
}

// Reconstruct A^T P + P A exactly.
[[nodiscard]] auto lyap_residual(const Matrix& a, const Matrix& p) -> Matrix {
    const Matrix at = a.transpose().value();
    const Matrix atp = at.multiply(p).value();
    const Matrix pa = p.multiply(a).value();
    return atp.add(pa).value();
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.analysis")
        // --- condition number ------------------------------------------------
        .test("matrix_norms_hand_values",
              [](TestContext& t) {
                  // A = [[1,-2],[3,4]]: col abs sums {4, 6} -> ||A||_1 = 6;
                  //                     row abs sums {3, 7} -> ||A||_inf = 7.
                  const Matrix a = mat({{1, -2}, {3, 4}});
                  auto n1 = matrix_norm_1(a);
                  auto ninf = matrix_norm_inf(a);
                  t.expect(n1.has_value() && n1.value() == ri(6), "||A||_1 = 6 (max col abs sum)");
                  t.expect(ninf.has_value() && ninf.value() == ri(7),
                           "||A||_inf = 7 (max row abs sum)");
              })
        .test("condition_1_hand_value",
              [](TestContext& t) {
                  // A = [[1,2],[0,1]], A^{-1} = [[1,-2],[0,1]].
                  // ||A||_1 = 3, ||A^{-1}||_1 = 3 -> kappa_1 = 9 (exact).
                  const Matrix a = mat({{1, 2}, {0, 1}});
                  auto k1 = condition_1(a);
                  t.expect(k1.has_value(), "condition_1 succeeds");
                  t.expect(k1.value_or(ri(0)) == ri(9), "kappa_1([[1,2],[0,1]]) = 9 exactly");
                  auto kinf = condition_inf(a);
                  t.expect(kinf.value_or(ri(0)) == ri(9), "kappa_inf = 9 exactly");
              })
        .test("condition_1_diagonal",
              [](TestContext& t) {
                  // A = diag(2,4): ||A||_1 = 4, ||A^{-1}||_1 = 1/2 -> kappa_1 = 2.
                  const Matrix a = mat({{2, 0}, {0, 4}});
                  auto k1 = condition_1(a);
                  t.expect(k1.value_or(ri(0)) == ri(2), "kappa_1(diag(2,4)) = 2 exactly");
              })
        .test("condition_singular_is_error",
              [](TestContext& t) {
                  auto k1 = condition_1(mat({{1, 1}, {1, 1}}));
                  t.expect(!k1.has_value() && k1.error() == MathError::domain_error,
                           "singular A => domain_error");
              })
        .test("condition_2_estimate_diagonal",
              [](TestContext& t) {
                  // A = diag(2,4): singular values 2 and 4 -> kappa_2 = 2 (NUMERICAL).
                  auto k2 = condition_2_estimate(mat({{2, 0}, {0, 4}}));
                  t.expect(k2.has_value(), "condition_2_estimate succeeds");
                  t.expect(k2.has_value() && std::fabs(*k2 - 2.0) < 1e-6,
                           "kappa_2(diag(2,4)) ~ 2 (numerical)");
              })
        // --- convergence tests ----------------------------------------------
        .test("ratio_test_geometric_half_converges",
              [](TestContext& t) {
                  // a_n = (1/2)^n: ratio = 1/2 exactly -> converges, exact limit 1/2.
                  auto rt = ratio_test([](std::int64_t n) { return rr(1, pow2(n)); });
                  t.expect(rt.exact, "geometric ratio limit is exact over Q");
                  t.expect(rt.exact_limit.has_value() && *rt.exact_limit == rr(1, 2),
                           "exact ratio limit L = 1/2");
                  t.expect(rt.verdict == Verdict::converges, "r = 1/2 => converges");
              })
        .test("ratio_test_geometric_two_diverges",
              [](TestContext& t) {
                  // a_n = 2^n: ratio = 2 exactly -> diverges, exact limit 2.
                  auto rt = ratio_test([](std::int64_t n) { return ri(pow2(n)); });
                  t.expect(rt.exact, "geometric ratio limit is exact over Q");
                  t.expect(rt.exact_limit.has_value() && *rt.exact_limit == ri(2),
                           "exact ratio limit L = 2");
                  t.expect(rt.verdict == Verdict::diverges, "r = 2 => diverges");
              })
        .test("root_test_geometric_third",
              [](TestContext& t) {
                  // a_n = (1/3)^n: |a_n|^{1/n} = 1/3 -> converges (NUMERICAL).
                  auto rr3 = root_test([](std::int64_t n) {
                      return std::pow(1.0 / 3.0, static_cast<double>(n));
                  });
                  t.expect(std::fabs(rr3.numeric_limit - 1.0 / 3.0) < 1e-9,
                           "root-test statistic ~ 1/3");
                  t.expect(rr3.verdict == Verdict::converges, "root test => converges");
              })
        .test("alternating_harmonic_leibniz",
              [](TestContext& t) {
                  // b_n = 1/n, monotone decreasing to 0 -> Leibniz convergence.
                  auto v = alternating_series_test(
                      [](std::int64_t n) { return 1.0 / static_cast<double>(n); });
                  t.expect(v == Verdict::converges, "alternating harmonic converges (Leibniz)");
              })
        .test("comparison_test_dominated_converges",
              [](TestContext& t) {
                  // 0 <= 1/(n^2+1) <= 1/n^2, and sum 1/n^2 converges -> converges.
                  auto v = comparison_test(
                      [](std::int64_t n) { return 1.0 / (static_cast<double>(n) * n + 1.0); },
                      [](std::int64_t n) { return 1.0 / (static_cast<double>(n) * n); },
                      Verdict::converges);
                  t.expect(v == Verdict::converges, "term-wise dominated by a convergent series");
              })
        // --- Lyapunov equation ----------------------------------------------
        .test("lyapunov_solve_reconstructs_stable",
              [](TestContext& t) {
                  // A = [[-1,1],[0,-1]] (Hurwitz), Q = I. Solve A^T P + P A = -Q and
                  // verify the residual equals -Q exactly.
                  const Matrix a = mat({{-1, 1}, {0, -1}});
                  const Matrix q = Matrix::identity(2);
                  const Matrix neg_q = q.scale(ri(-1)).value();
                  auto p = lyapunov_solve(a, q);
                  t.expect(p.has_value(), "Lyapunov solve succeeds for a Hurwitz A");
                  if (p) {
                      const Matrix residual = lyap_residual(a, *p);
                      t.expect(residual == neg_q, "A^T P + P A == -Q exactly");
                      auto pd = is_positive_definite(*p);
                      t.expect(pd.value_or(false), "the Lyapunov solution P is positive definite");
                  }
              })
        .test("lyapunov_solve_singular_rotation",
              [](TestContext& t) {
                  // A = rotation (eigenvalues +/- i): i + (-i) = 0 makes the Kronecker sum
                  // singular -> domain_error.
                  auto p = lyapunov_solve(mat({{0, -1}, {1, 0}}), Matrix::identity(2));
                  t.expect(!p.has_value() && p.error() == MathError::domain_error,
                           "eigenvalue pair summing to zero => singular => domain_error");
              })
        .test("stein_solve_reconstructs",
              [](TestContext& t) {
                  // Discrete Stein A^T P A - P = -Q for a Schur-stable A = diag(1/2, 1/3).
                  const Matrix a = matq({{rr(1, 2), ri(0)}, {ri(0), rr(1, 3)}});
                  const Matrix q = Matrix::identity(2);
                  const Matrix neg_q = q.scale(ri(-1)).value();
                  auto p = stein_solve(a, q);
                  t.expect(p.has_value(), "Stein solve succeeds");
                  if (p) {
                      const Matrix at = a.transpose().value();
                      const Matrix atpa = at.multiply(*p).value().multiply(a).value();
                      const Matrix residual = atpa.subtract(*p).value();
                      t.expect(residual == neg_q, "A^T P A - P == -Q exactly");
                  }
              })
        // --- Sylvester positive-definiteness --------------------------------
        .test("is_positive_definite_sylvester",
              [](TestContext& t) {
                  // [[2,1],[1,2]]: minors 2 > 0, det 3 > 0 -> PD.
                  auto pd = is_positive_definite(mat({{2, 1}, {1, 2}}));
                  t.expect(pd.value_or(false), "[[2,1],[1,2]] is positive definite");
                  // [[1,2],[2,1]]: det = -3 < 0 -> not PD.
                  auto npd = is_positive_definite(mat({{1, 2}, {2, 1}}));
                  t.expect(npd.has_value() && !*npd, "[[1,2],[2,1]] is not positive definite");
              })
        // --- Lyapunov stability + Routh-Hurwitz cross-check -----------------
        .test("stable_A_is_positive_definite_and_agrees",
              [](TestContext& t) {
                  // A = [[-1,1],[0,-1]] Hurwitz: Lyapunov P > 0 and Routh-Hurwitz agree.
                  const Matrix a = mat({{-1, 1}, {0, -1}});
                  auto stable = is_stable_lyapunov(a);
                  t.expect(stable.value_or(false), "Lyapunov/Sylvester: A is stable");
                  auto rh = is_asymptotically_stable(a);
                  t.expect(rh.value_or(false), "Routh-Hurwitz: A is stable");
                  auto x = stability_cross_check(a);
                  t.expect(x.has_value() && x->lyapunov_stable && x->routh_hurwitz_stable && x->agree,
                           "cross-check: both exact verdicts agree (stable)");
              })
        .test("unstable_A_is_not_positive_definite_and_agrees",
              [](TestContext& t) {
                  // A = diag(1,2): positive eigenvalues -> unstable; both verdicts agree false.
                  const Matrix a = mat({{1, 0}, {0, 2}});
                  auto stable = is_stable_lyapunov(a);
                  t.expect(stable.has_value() && !*stable, "Lyapunov/Sylvester: A is not stable");
                  auto x = stability_cross_check(a);
                  t.expect(x.has_value() && !x->lyapunov_stable && !x->routh_hurwitz_stable && x->agree,
                           "cross-check: both exact verdicts agree (unstable)");
              })
        .test("marginal_rotation_is_not_stable",
              [](TestContext& t) {
                  // Rotation (eigenvalues +/- i): singular Lyapunov system -> not stable,
                  // matching Routh-Hurwitz.
                  const Matrix a = mat({{0, -1}, {1, 0}});
                  auto stable = is_stable_lyapunov(a);
                  t.expect(stable.has_value() && !*stable, "rotation is not asymptotically stable");
                  auto x = stability_cross_check(a);
                  t.expect(x.has_value() && x->agree && !x->lyapunov_stable,
                           "cross-check agrees the rotation is not stable");
              })
        // --- Lyapunov exponent (numerical) ----------------------------------
        .test("lyapunov_exponent_expanding",
              [](TestContext& t) {
                  // Constant Jacobian 2 I repeated: leading exponent = log 2 ~ 0.6931.
                  const Matrix j = mat({{2, 0}, {0, 2}});
                  std::vector<Matrix> js(12, j);
                  auto le = lyapunov_exponent(std::span<const Matrix>(js));
                  t.expect(le.has_value(), "Lyapunov exponent computes");
                  t.expect(le.has_value() && std::fabs(*le - std::log(2.0)) < 1e-9,
                           "expanding map exponent ~ log 2");
              })
        .test("lyapunov_exponent_contracting",
              [](TestContext& t) {
                  // Constant Jacobian (1/2) I repeated: leading exponent = log(1/2) ~ -0.6931.
                  const Matrix j = matq({{rr(1, 2), ri(0)}, {ri(0), rr(1, 2)}});
                  std::vector<Matrix> js(12, j);
                  auto le = lyapunov_exponent(std::span<const Matrix>(js));
                  t.expect(le.has_value() && std::fabs(*le - std::log(0.5)) < 1e-9,
                           "contracting map exponent ~ log(1/2)");
              })
        .run();
}
