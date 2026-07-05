// Tests for nimblecas.numeigen: numeric eigenvalues of a real matrix (all eigenvalues,
// real and complex) via the structure-aware QR dispatcher, plus companion-matrix roots.
// @author Olumuyiwa Oluwasanmi
//
// The results are NUMERICAL approximations, so every assertion is made up to a tolerance
// (1e-6) and every eigenvalue set is compared as a MULTISET (order-independent) via the
// approx_set helper. Companion-matrix roots are additionally verified by evaluating the
// originating polynomial at each returned root and checking the value is ~0.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.numeigen;
import nimblecas.testing;

using nimblecas::MathError;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::companion_eigenvalues;
using nimblecas::eigenvalues_qr;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

using cd = std::complex<double>;

namespace {

// Do the multisets `got` and `expected` coincide up to ordering and tolerance `tol`?
// Greedy matching: every expected value must pair with a distinct, still-unused got value
// within `tol`. Sizes must match.
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

// Largest imaginary magnitude across a spectrum (0 for a purely real result).
auto max_abs_imag(std::span<const cd> got) -> double {
    double m = 0.0;
    for (const cd& z : got) {
        m = std::max(m, std::abs(z.imag()));
    }
    return m;
}

// A RationalPoly x^... from ascending integer coefficients [c0, c1, ...].
auto poly(std::vector<std::int64_t> asc) -> RationalPoly {
    std::vector<Rational> c;
    c.reserve(asc.size());
    for (std::int64_t v : asc) {
        c.push_back(Rational::from_int(v));
    }
    return RationalPoly::from_coeffs(std::move(c));
}

// Evaluate a RationalPoly at a complex point by Horner's rule (numeric, for root checks).
auto eval_poly(const RationalPoly& p, const cd& z) -> cd {
    cd acc{0.0, 0.0};
    const auto coeffs = p.coefficients();
    for (std::size_t k = coeffs.size(); k-- > 0;) {
        const double c = static_cast<double>(coeffs[k].numerator()) /
                         static_cast<double>(coeffs[k].denominator());
        acc = acc * z + cd{c, 0.0};
    }
    return acc;
}

constexpr double kTol = 1e-6;

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.numeigen")
        .test("diagonal_matrix_eigenvalues_are_diagonal",
              [&](TestContext& t) {
                  // diag(1,2,3) -> {1,2,3} (diagonal dispatch branch).
                  const std::vector<double> a = {1, 0, 0, 0, 2, 0, 0, 0, 3};
                  auto e = eigenvalues_qr(a, 3);
                  t.expect(e.has_value(), "eigenvalues_qr(diag) succeeds");
                  if (!e) {
                      return;
                  }
                  const std::vector<cd> expected = {{1, 0}, {2, 0}, {3, 0}};
                  t.expect(approx_set(*e, expected, kTol), "eigenvalues are {1,2,3}");
                  t.expect(max_abs_imag(*e) <= kTol, "diagonal result is purely real");
              })
        .test("upper_triangular_eigenvalues_are_diagonal",
              [&](TestContext& t) {
                  // Upper-triangular; eigenvalues are the diagonal {1,4,6}, NOT influenced
                  // by the off-diagonal entries (triangular dispatch branch).
                  const std::vector<double> a = {1, 2, 3, 0, 4, 5, 0, 0, 6};
                  auto e = eigenvalues_qr(a, 3);
                  t.expect(e.has_value(), "eigenvalues_qr(triangular) succeeds");
                  if (!e) {
                      return;
                  }
                  const std::vector<cd> expected = {{1, 0}, {4, 0}, {6, 0}};
                  t.expect(approx_set(*e, expected, kTol), "eigenvalues are the diagonal {1,4,6}");
                  t.expect(max_abs_imag(*e) <= kTol, "triangular result is purely real");
              })
        .test("symmetric_2x2_real_spectrum",
              [&](TestContext& t) {
                  // [[2,1],[1,2]] -> {1,3}; symmetric branch must return REAL eigenvalues
                  // with exactly-zero (well within tol) imaginary parts.
                  const std::vector<double> a = {2, 1, 1, 2};
                  auto e = eigenvalues_qr(a, 2);
                  t.expect(e.has_value(), "eigenvalues_qr(symmetric) succeeds");
                  if (!e) {
                      return;
                  }
                  const std::vector<cd> expected = {{1, 0}, {3, 0}};
                  t.expect(approx_set(*e, expected, kTol), "eigenvalues are {1,3}");
                  t.expect(max_abs_imag(*e) == 0.0,
                           "symmetric path yields no spurious imaginary part");
              })
        .test("skew_symmetric_2x2_imaginary_pair",
              [&](TestContext& t) {
                  // [[0,-1],[1,0]] -> {+i, -i} (skew-symmetric -> general Francis path).
                  const std::vector<double> a = {0, -1, 1, 0};
                  auto e = eigenvalues_qr(a, 2);
                  t.expect(e.has_value(), "eigenvalues_qr(skew) succeeds");
                  if (!e) {
                      return;
                  }
                  const std::vector<cd> expected = {{0, 1}, {0, -1}};
                  t.expect(approx_set(*e, expected, kTol), "eigenvalues are +/- i");
              })
        .test("general_2x2_real_pair",
              [&](TestContext& t) {
                  // [[1,2],[3,4]]: eigenvalues (5 +/- sqrt(33))/2 ~ {5.37228, -0.37228}
                  // (general Francis path).
                  const std::vector<double> a = {1, 2, 3, 4};
                  auto e = eigenvalues_qr(a, 2);
                  t.expect(e.has_value(), "eigenvalues_qr(general 2x2) succeeds");
                  if (!e) {
                      return;
                  }
                  const double half = 0.5 * std::sqrt(33.0);
                  const std::vector<cd> expected = {{2.5 + half, 0}, {2.5 - half, 0}};
                  t.expect(approx_set(*e, expected, kTol), "eigenvalues (5 +/- sqrt(33))/2");
              })
        .test("size_mismatch_is_domain_error",
              [&](TestContext& t) {
                  const std::vector<double> a = {1, 2, 3};  // length 3 != 2*2
                  auto e = eigenvalues_qr(a, 2);
                  t.expect(!e.has_value(), "size mismatch rejected");
                  t.expect(!e && e.error() == MathError::domain_error, "domain_error reported");
              })
        .test("empty_matrix_has_no_eigenvalues",
              [&](TestContext& t) {
                  auto e = eigenvalues_qr(std::span<const double>{}, 0);
                  t.expect(e.has_value(), "0x0 succeeds");
                  t.expect(e && e->empty(), "0x0 has an empty spectrum");
              })
        .test("companion_quadratic_real_roots",
              [&](TestContext& t) {
                  // x^2 - 3x + 2 = (x-1)(x-2) -> roots {1, 2}.
                  const auto p = poly({2, -3, 1});
                  auto e = companion_eigenvalues(p);
                  t.expect(e.has_value(), "companion_eigenvalues(x^2-3x+2) succeeds");
                  if (!e) {
                      return;
                  }
                  const std::vector<cd> expected = {{1, 0}, {2, 0}};
                  t.expect(approx_set(*e, expected, kTol), "roots are {1,2}");
                  for (const cd& z : *e) {
                      t.expect(std::abs(eval_poly(p, z)) <= kTol, "p(root) ~ 0");
                  }
              })
        .test("companion_cubic_one_real_two_complex",
              [&](TestContext& t) {
                  // x^3 - 2: one real root cbrt(2) ~ 1.259921 and a complex-conjugate pair.
                  const auto p = poly({-2, 0, 0, 1});
                  auto e = companion_eigenvalues(p);
                  t.expect(e.has_value(), "companion_eigenvalues(x^3-2) succeeds");
                  if (!e) {
                      return;
                  }
                  t.expect_eq(e->size(), std::size_t{3}, "three roots");
                  const double cbrt2 = std::cbrt(2.0);
                  const std::vector<cd> expected = {
                      {cbrt2, 0.0},
                      {cbrt2 * std::cos(2.0 * std::numbers::pi / 3.0),
                       cbrt2 * std::sin(2.0 * std::numbers::pi / 3.0)},
                      {cbrt2 * std::cos(2.0 * std::numbers::pi / 3.0),
                       -cbrt2 * std::sin(2.0 * std::numbers::pi / 3.0)},
                  };
                  t.expect(approx_set(*e, expected, kTol),
                           "roots are the three cube roots of 2");
                  // Exactly one real root.
                  std::size_t real_count = 0;
                  for (const cd& z : *e) {
                      if (std::abs(z.imag()) <= kTol) {
                          ++real_count;
                          t.expect(std::abs(z.real() - cbrt2) <= kTol, "real root ~ cbrt(2)");
                      }
                      t.expect(std::abs(eval_poly(p, z)) <= kTol, "p(root) ~ 0");
                  }
                  t.expect_eq(real_count, std::size_t{1}, "exactly one real root");
              })
        .test("companion_quintic_five_roots_verified",
              [&](TestContext& t) {
                  // x^5 - x - 1: one real root ~1.1673, two complex-conjugate pairs. Verify
                  // by evaluating the polynomial at every returned root.
                  const auto p = poly({-1, -1, 0, 0, 0, 1});
                  auto e = companion_eigenvalues(p);
                  t.expect(e.has_value(), "companion_eigenvalues(x^5-x-1) succeeds");
                  if (!e) {
                      return;
                  }
                  t.expect_eq(e->size(), std::size_t{5}, "five roots");
                  std::size_t real_count = 0;
                  for (const cd& z : *e) {
                      t.expect(std::abs(eval_poly(p, z)) <= kTol, "p(root) ~ 0");
                      if (std::abs(z.imag()) <= kTol) {
                          ++real_count;
                          t.expect(std::abs(z.real() - 1.1673039782614187) <= 1e-4,
                                   "real root ~ 1.1673");
                      }
                  }
                  t.expect_eq(real_count, std::size_t{1}, "exactly one real root");
              })
        .test("companion_zero_polynomial_is_domain_error",
              [&](TestContext& t) {
                  auto e = companion_eigenvalues(RationalPoly{});
                  t.expect(!e.has_value(), "zero polynomial rejected");
                  t.expect(!e && e.error() == MathError::domain_error, "domain_error reported");
              })
        .test("companion_constant_polynomial_has_no_roots",
              [&](TestContext& t) {
                  auto e = companion_eigenvalues(poly({5}));  // constant 5
                  t.expect(e.has_value(), "non-zero constant succeeds");
                  t.expect(e && e->empty(), "a non-zero constant has no roots");
              })
        .run();
}
