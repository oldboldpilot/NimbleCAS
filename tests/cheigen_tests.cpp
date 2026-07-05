// Tests for nimblecas.cheigen: numeric eigenvalues of a COMPLEX matrix via the
// real-embedding trick over nimblecas.numeigen, with exact-structure dispatch.
// @author Olumuyiwa Oluwasanmi
//
// The results are NUMERICAL approximations, so eigenvalue sets are compared as MULTISETS
// (order-independent, within a tolerance) via approx_set — except where the honesty
// contract promises an EXACT value: a Hermitian matrix's imaginary parts and a
// skew-Hermitian matrix's real parts must be exactly 0.0, and those are asserted as ==.
// General eigenvalues are additionally verified independently by the 2x2 characteristic
// residual det(M - lambda*I) ~ 0 (no reliance on precomputed irrational values).

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.complex;
import nimblecas.cmatrix;
import nimblecas.cheigen;
import nimblecas.testing;

using nimblecas::Complex;
using nimblecas::ComplexMatrix;
using nimblecas::MathError;
using nimblecas::Rational;
using nimblecas::eigenvalues;
using nimblecas::hermitian_eigenvalues;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

using cd = std::complex<double>;

namespace {

// The Gaussian-rational complex number re + im*i (integer parts).
auto cx(std::int64_t re, std::int64_t im) -> Complex {
    return Complex::make(Rational::from_int(re), Rational::from_int(im));
}

// Build a ComplexMatrix from a list of rows; asserts construction succeeded.
auto mat(std::vector<std::vector<Complex>> rows) -> ComplexMatrix {
    return ComplexMatrix::from_rows(std::move(rows)).value();
}

// Do the multisets `got` and `expected` coincide up to ordering and tolerance `tol`?
auto approx_set(std::span<const cd> got, std::span<const cd> expected, double tol) -> bool {
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

// Same, for a real (sorted) spectrum.
auto approx_set_real(std::span<const double> got, std::span<const double> expected, double tol)
    -> bool {
    if (got.size() != expected.size()) {
        return false;
    }
    std::vector<bool> used(got.size(), false);
    for (double e : expected) {
        bool matched = false;
        for (std::size_t i = 0; i < got.size(); ++i) {
            if (!used[i] && std::fabs(got[i] - e) <= tol) {
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

auto max_abs_imag(std::span<const cd> got) -> double {
    double m = 0.0;
    for (const cd& z : got) {
        m = std::max(m, std::abs(z.imag()));
    }
    return m;
}

auto max_abs_real(std::span<const cd> got) -> double {
    double m = 0.0;
    for (const cd& z : got) {
        m = std::max(m, std::abs(z.real()));
    }
    return m;
}

// det(M - lambda*I) for a row-major 2x2 complex matrix `m`, evaluated at complex `lambda`.
// Independent residual check on the general path: every returned eigenvalue must annihilate
// the characteristic polynomial without the test hard-coding the eigenvalues themselves.
auto det2_shift(const std::array<cd, 4>& m, const cd& lambda) -> cd {
    const cd a = m[0] - lambda, b = m[1], c = m[2], d = m[3] - lambda;
    return a * d - b * c;
}

constexpr double kTol = 1e-6;

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.cheigen")
        .test("hermitian_2x2_real_spectrum_one_three",
              [&](TestContext& t) {
                  // [[2, i], [-i, 2]] is Hermitian; eigenvalues are the reals {1, 3}.
                  const auto m = mat({{cx(2, 0), cx(0, 1)}, {cx(0, -1), cx(2, 0)}});
                  auto e = hermitian_eigenvalues(m);
                  t.expect(e.has_value(), "hermitian_eigenvalues([[2,i],[-i,2]]) succeeds");
                  if (!e) {
                      return;
                  }
                  const std::vector<double> expected = {1.0, 3.0};
                  t.expect(approx_set_real(*e, expected, kTol), "real spectrum {1, 3}");
                  // eigenvalues() must agree and carry an EXACTLY-zero imaginary part.
                  auto g = eigenvalues(m);
                  t.expect(g.has_value(), "eigenvalues([[2,i],[-i,2]]) succeeds");
                  if (g) {
                      const std::vector<cd> ce = {{1, 0}, {3, 0}};
                      t.expect(approx_set(*g, ce, kTol), "eigenvalues() agrees: {1, 3}");
                      t.expect(max_abs_imag(*g) == 0.0,
                               "Hermitian => imaginary part exactly zero");
                  }
              })
        .test("hermitian_2x2_irrational_real_to_1e9",
              [&](TestContext& t) {
                  // [[1, i], [-i, 2]] is Hermitian; eigenvalues (3 +/- sqrt5)/2 are irrational
                  // but REAL — the imaginary part must be exactly 0 (well inside 1e-9).
                  const auto m = mat({{cx(1, 0), cx(0, 1)}, {cx(0, -1), cx(2, 0)}});
                  auto e = eigenvalues(m);
                  t.expect(e.has_value(), "eigenvalues([[1,i],[-i,2]]) succeeds");
                  if (!e) {
                      return;
                  }
                  const double s5 = std::sqrt(5.0);
                  const std::vector<cd> expected = {{(3.0 - s5) / 2.0, 0}, {(3.0 + s5) / 2.0, 0}};
                  t.expect(approx_set(*e, expected, kTol), "eigenvalues (3 +/- sqrt5)/2");
                  t.expect(max_abs_imag(*e) <= 1e-9, "eigenvalues real to 1e-9");
                  t.expect(max_abs_imag(*e) == 0.0, "Hermitian path yields exact-0 imaginary");
              })
        .test("pauli_y_is_hermitian_real_pair",
              [&](TestContext& t) {
                  // Pauli-Y = [[0, -i], [i, 0]] is Hermitian with real eigenvalues {+1, -1}.
                  const auto y = mat({{cx(0, 0), cx(0, -1)}, {cx(0, 1), cx(0, 0)}});
                  auto herm = y.is_hermitian();
                  t.expect(herm.has_value() && *herm, "Pauli-Y is Hermitian (exact)");
                  auto e = hermitian_eigenvalues(y);
                  t.expect(e.has_value(), "hermitian_eigenvalues(Y) succeeds");
                  if (!e) {
                      return;
                  }
                  const std::vector<double> expected = {-1.0, 1.0};
                  t.expect(approx_set_real(*e, expected, kTol), "eigenvalues {-1, +1}");
              })
        .test("skew_hermitian_diagonal_purely_imaginary",
              [&](TestContext& t) {
                  // diag(2i, i) is skew-Hermitian; eigenvalues {2i, i} are purely imaginary
                  // (not a +/- conjugate pair), recovered exactly via the iM-Hermitian route,
                  // and the real parts must be exactly 0.
                  const auto m = mat({{cx(0, 2), cx(0, 0)}, {cx(0, 0), cx(0, 1)}});
                  auto skew = m.is_skew_hermitian();
                  t.expect(skew.has_value() && *skew, "diag(2i, i) is skew-Hermitian (exact)");
                  auto e = eigenvalues(m);
                  t.expect(e.has_value(), "eigenvalues(diag(2i,i)) succeeds");
                  if (!e) {
                      return;
                  }
                  const std::vector<cd> expected = {{0, 2}, {0, 1}};
                  t.expect(approx_set(*e, expected, kTol), "eigenvalues {2i, i}");
                  t.expect(max_abs_real(*e) == 0.0,
                           "skew-Hermitian => real part exactly zero");
              })
        .test("skew_hermitian_conjugate_pair_recovered_exactly",
              [&](TestContext& t) {
                  // diag(i, -i) is skew-Hermitian with a purely imaginary CONJUGATE pair
                  // {i, -i}. The general embedding path cannot recover this (R-spectrum is
                  // {i,i,-i,-i}, shared with diag(i,i)); the iM-Hermitian reduction can and must
                  // — iM = diag(-1, 1), so mu = {-1, 1} and lambda = -i*mu = {i, -i}.
                  const auto m = mat({{cx(0, 1), cx(0, 0)}, {cx(0, 0), cx(0, -1)}});
                  auto skew = m.is_skew_hermitian();
                  t.expect(skew.has_value() && *skew, "diag(i, -i) is skew-Hermitian (exact)");
                  auto e = eigenvalues(m);
                  t.expect(e.has_value(), "eigenvalues(diag(i,-i)) succeeds");
                  if (!e) {
                      return;
                  }
                  const std::vector<cd> expected = {{0, 1}, {0, -1}};
                  t.expect(approx_set(*e, expected, kTol), "eigenvalues EXACTLY {i, -i}");
                  t.expect(max_abs_real(*e) == 0.0,
                           "skew-Hermitian => real part exactly zero");
              })
        .test("unitary_2x2_on_unit_circle",
              [&](TestContext& t) {
                  // diag(i, (3+4i)/5) is unitary (both diagonal entries have modulus 1). Its
                  // spectrum {i, (3+4i)/5} has NO real eigenvalue and is not conjugate-closed, so
                  // the general path recovers it cleanly; both eigenvalues lie on |lambda| = 1.
                  const auto e35 =
                      Complex::make(Rational::make(3, 5).value(), Rational::make(4, 5).value());
                  const auto m = mat({{cx(0, 1), cx(0, 0)}, {cx(0, 0), e35}});
                  auto uni = m.is_unitary();
                  t.expect(uni.has_value() && *uni, "diag(i, (3+4i)/5) is unitary (exact)");
                  auto e = eigenvalues(m);
                  t.expect(e.has_value(), "eigenvalues(diag(i, (3+4i)/5)) succeeds");
                  if (!e) {
                      return;
                  }
                  const std::vector<cd> expected = {{0, 1}, {0.6, 0.8}};
                  t.expect(approx_set(*e, expected, kTol), "eigenvalues {i, (3+4i)/5}");
                  for (const cd& z : *e) {
                      t.expect(std::fabs(std::abs(z) - 1.0) <= kTol, "|lambda| = 1 on the circle");
                  }
              })
        .test("general_upper_triangular_hand_values",
              [&](TestContext& t) {
                  // [[1+i, 5], [0, 2-3i]] is upper-triangular; eigenvalues are the diagonal
                  // {1+i, 2-3i} — genuinely complex and NOT a conjugate pair, so the recovery
                  // must select the correct half of the embedding's spectrum.
                  const auto m = mat({{cx(1, 1), cx(5, 0)}, {cx(0, 0), cx(2, -3)}});
                  auto e = eigenvalues(m);
                  t.expect(e.has_value(), "eigenvalues([[1+i,5],[0,2-3i]]) succeeds");
                  if (!e) {
                      return;
                  }
                  const std::vector<cd> expected = {{1, 1}, {2, -3}};
                  t.expect(approx_set(*e, expected, kTol), "eigenvalues {1+i, 2-3i}");
                  const std::array<cd, 4> mc = {cd{1, 1}, cd{5, 0}, cd{0, 0}, cd{2, -3}};
                  for (const cd& z : *e) {
                      t.expect(std::abs(det2_shift(mc, z)) <= 1e-6,
                               "det(M - lambda*I) ~ 0 for each eigenvalue");
                  }
              })
        .test("general_nontriangular_residual_check",
              [&](TestContext& t) {
                  // [[i, 1], [-1, 2]]: a genuinely non-triangular, unstructured complex matrix
                  // whose two eigenvalues are complex and NOT conjugates. Rather than hard-code
                  // them, verify each returned lambda via det(M - lambda*I) ~ 0.
                  const auto m = mat({{cx(0, 1), cx(1, 0)}, {cx(-1, 0), cx(2, 0)}});
                  auto e = eigenvalues(m);
                  t.expect(e.has_value(), "eigenvalues([[i,1],[-1,2]]) succeeds");
                  if (!e) {
                      return;
                  }
                  t.expect_eq(e->size(), std::size_t{2}, "two eigenvalues");
                  const std::array<cd, 4> mc = {cd{0, 1}, cd{1, 0}, cd{-1, 0}, cd{2, 0}};
                  for (const cd& z : *e) {
                      t.expect(std::abs(det2_shift(mc, z)) <= 1e-6,
                               "det(M - lambda*I) ~ 0 for each returned eigenvalue");
                  }
              })
        .test("conjugate_closed_spectrum_is_not_implemented",
              [&](TestContext& t) {
                  // The real matrix [[1,-2],[2,1]] (as a ComplexMatrix) has the complex-conjugate
                  // spectrum {1+2i, 1-2i}. Its real embedding R shares its spectrum with matrices
                  // that are NOT M's conjugate, so M is unrecoverable from R alone. Rule 32: the
                  // general path must return an honest not_implemented, never a wrong multiset
                  // such as {1+2i, 1+2i}. (Real matrices belong to numeigen::eigenvalues_qr.)
                  const auto m = mat({{cx(1, 0), cx(-2, 0)}, {cx(2, 0), cx(1, 0)}});
                  // Not Hermitian and not skew-Hermitian => it takes the general path.
                  auto herm = m.is_hermitian();
                  auto skew = m.is_skew_hermitian();
                  t.expect(herm.has_value() && !*herm, "[[1,-2],[2,1]] is not Hermitian");
                  t.expect(skew.has_value() && !*skew, "[[1,-2],[2,1]] is not skew-Hermitian");
                  auto e = eigenvalues(m);
                  t.expect(!e.has_value(), "conjugate-closed spectrum is not fabricated");
                  t.expect(!e && e.error() == MathError::not_implemented,
                           "conjugate-closed spectrum surfaces as not_implemented");
              })
        .test("non_square_is_domain_error",
              [&](TestContext& t) {
                  // A 2x3 complex matrix has no eigenvalues; both entry points reject it.
                  const auto m = mat({{cx(1, 0), cx(0, 1), cx(2, 0)},
                                      {cx(0, -1), cx(3, 0), cx(1, 1)}});
                  auto e = eigenvalues(m);
                  t.expect(!e.has_value(), "eigenvalues(2x3) rejected");
                  t.expect(!e && e.error() == MathError::domain_error, "domain_error reported");
                  auto h = hermitian_eigenvalues(m);
                  t.expect(!h && h.error() == MathError::domain_error,
                           "hermitian_eigenvalues(2x3) also domain_error");
              })
        .test("hermitian_eigenvalues_rejects_non_hermitian",
              [&](TestContext& t) {
                  // Honesty: claiming a real Hermitian spectrum requires a Hermitian matrix.
                  // [[1+i, 5], [0, 2-3i]] is not Hermitian, so hermitian_eigenvalues refuses.
                  const auto m = mat({{cx(1, 1), cx(5, 0)}, {cx(0, 0), cx(2, -3)}});
                  auto h = hermitian_eigenvalues(m);
                  t.expect(!h.has_value(), "non-Hermitian input rejected");
                  t.expect(!h && h.error() == MathError::domain_error, "domain_error reported");
              })
        .test("empty_matrix_has_no_eigenvalues",
              [&](TestContext& t) {
                  const ComplexMatrix m{};  // 0x0
                  auto e = eigenvalues(m);
                  t.expect(e.has_value() && e->empty(), "0x0 has an empty spectrum");
                  auto h = hermitian_eigenvalues(m);
                  t.expect(h.has_value() && h->empty(), "0x0 Hermitian spectrum is empty");
              })
        .run();
}
