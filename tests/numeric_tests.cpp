// Tests for nimblecas.numeric: real-root finders (Newton, bisection, secant).
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.numeric;
import nimblecas.testing;

using nimblecas::MathError;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;
namespace num = nimblecas::numeric;

namespace {
// Absolute-tolerance comparison used throughout, per the task contract.
auto close(double got, double expected) -> bool {
    return std::abs(got - expected) < 1e-6;
}

// Complex Horner evaluation of an ascending-order coefficient span, used to
// verify Durand-Kerner roots by residual |p(root)|.
auto ceval(std::span<const double> c, std::complex<double> z) -> std::complex<double> {
    std::complex<double> acc{0.0, 0.0};
    for (double coeff : std::ranges::reverse_view(c)) {
        acc = acc * z + coeff;
    }
    return acc;
}

// True iff every returned root has residual |p(root)| < tol.
auto all_roots_vanish(std::span<const double> coeffs,
                      std::span<const std::complex<double>> roots, double tol) -> bool {
    return std::ranges::all_of(roots, [&](std::complex<double> z) {
        return std::abs(ceval(coeffs, z)) < tol;
    });
}

// True iff some returned root is within tol of the expected (complex) value.
auto has_root_near(std::span<const std::complex<double>> roots, std::complex<double> want,
                   double tol) -> bool {
    return std::ranges::any_of(roots,
                               [&](std::complex<double> z) { return std::abs(z - want) < tol; });
}
}  // namespace

auto main() -> int {
    // sqrt(2), the positive root of x^2 - 2.
    const double root2 = std::numbers::sqrt2;
    // coefficients are ascending (low degree first).
    const std::array<double, 3> sq_minus_two{-2.0, 0.0, 1.0};  // x^2 - 2
    const std::array<double, 4> cubic{0.0, -1.0, 0.0, 1.0};    // x^3 - x
    const std::array<double, 3> sq_plus_one{1.0, 0.0, 1.0};    // x^2 + 1 (no real root)

    return TestSuite("nimblecas.numeric")
        .test("newton_sqrt2",
              [&](TestContext& t) {
                  auto r = num::newton(sq_minus_two, 1.5, 1e-12, 100);
                  t.expect(r.has_value(), "newton converges on x^2 - 2");
                  t.expect(r.has_value() && close(*r, root2), "root == sqrt(2)");
                  t.expect(r.has_value() && std::abs(num::eval(sq_minus_two, *r)) < 1e-6,
                           "p(root) ~ 0");
              })
        .test("bisection_sqrt2",
              [&](TestContext& t) {
                  auto r = num::bisection(sq_minus_two, 1.0, 2.0, 1e-9);
                  t.expect(r.has_value(), "bisection converges on [1, 2]");
                  t.expect(r.has_value() && close(*r, root2), "root == sqrt(2)");
                  t.expect(r.has_value() && std::abs(num::eval(sq_minus_two, *r)) < 1e-6,
                           "p(root) ~ 0");
              })
        .test("bisection_root_on_lower_endpoint",
              [&](TestContext& t) {
                  // p(x) = x has its root exactly at the lower bracket endpoint. With
                  // f(lo)==0 the sign test can never fire, so a naive loop would drift to
                  // the far bound; the endpoint guard must return lo directly.
                  const std::array<double, 2> line{0.0, 1.0};  // 0 + 1*x
                  auto r = num::bisection(line, 0.0, 2.0, 1e-9);
                  t.expect(r.has_value(), "bisection brackets [0, 2] with root at 0");
                  t.expect(r.has_value() && close(*r, 0.0),
                           "root at lower endpoint == 0 (not the far bound)");
              })
        .test("bisection_root_on_upper_endpoint",
              [&](TestContext& t) {
                  const std::array<double, 2> line{-2.0, 1.0};  // x - 2, root at x=2
                  auto r = num::bisection(line, 0.0, 2.0, 1e-9);
                  t.expect(r.has_value() && close(*r, 2.0), "root at upper endpoint == 2");
              })
        .test("secant_sqrt2",
              [&](TestContext& t) {
                  auto r = num::secant(sq_minus_two, 1.0, 2.0, 1e-12, 100);
                  t.expect(r.has_value(), "secant converges from (1, 2)");
                  t.expect(r.has_value() && close(*r, root2), "root == sqrt(2)");
                  t.expect(r.has_value() && std::abs(num::eval(sq_minus_two, *r)) < 1e-6,
                           "p(root) ~ 0");
              })
        .test("newton_cubic_selects_nearest_root",
              [&](TestContext& t) {
                  // x^3 - x has roots {-1, 0, 1}; Newton follows the nearest basin.
                  auto rp = num::newton(cubic, 1.5, 1e-12, 100);
                  t.expect(rp.has_value() && close(*rp, 1.0), "from x0=1.5 -> +1");
                  t.expect(rp.has_value() && std::abs(num::eval(cubic, *rp)) < 1e-6,
                           "p(+1) ~ 0");
                  auto rn = num::newton(cubic, -1.5, 1e-12, 100);
                  t.expect(rn.has_value() && close(*rn, -1.0), "from x0=-1.5 -> -1");
                  t.expect(rn.has_value() && std::abs(num::eval(cubic, *rn)) < 1e-6,
                           "p(-1) ~ 0");
              })
        .test("bisection_requires_sign_change",
              [&](TestContext& t) {
                  auto r = num::bisection(sq_plus_one, 0.0, 1.0, 1e-9);
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "no sign change over [0,1] -> domain_error");
              })
        .test("empty_coeffs_domain_error",
              [&](TestContext& t) {
                  std::span<const double> empty{};
                  auto r = num::newton(empty, 1.0, 1e-9, 100);
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "newton on empty coeffs -> domain_error");
                  auto rb = num::bisection(empty, 0.0, 1.0, 1e-9);
                  t.expect(!rb.has_value() && rb.error() == MathError::domain_error,
                           "bisection on empty coeffs -> domain_error");
                  auto rs = num::secant(empty, 0.0, 1.0, 1e-9, 100);
                  t.expect(!rs.has_value() && rs.error() == MathError::domain_error,
                           "secant on empty coeffs -> domain_error");
              })
        // --- Durand-Kerner: all roots (real + complex) simultaneously ----------
        .test("durand_kerner_real_quadratic",
              [&](TestContext& t) {
                  // x^2 - 3x + 2 = (x-1)(x-2), roots {1, 2}.  coeffs {2, -3, 1}.
                  const std::array<double, 3> q{2.0, -3.0, 1.0};
                  auto r = num::durand_kerner(q, 1e-12, 200);
                  t.expect(r.has_value(), "durand_kerner converges on x^2 - 3x + 2");
                  t.expect(r.has_value() && r->size() == 2, "two roots returned");
                  t.expect(r.has_value() && all_roots_vanish(q, *r, 1e-6),
                           "each |p(root)| < tol");
                  t.expect(r.has_value() && has_root_near(*r, {1.0, 0.0}, 1e-6) &&
                               has_root_near(*r, {2.0, 0.0}, 1e-6),
                           "root set == {1, 2}");
              })
        .test("durand_kerner_cbrt_two_and_complex_pair",
              [&](TestContext& t) {
                  // x^3 - 2: one real root cbrt(2), plus a complex-conjugate pair.
                  const std::array<double, 4> c{-2.0, 0.0, 0.0, 1.0};
                  auto r = num::durand_kerner(c, 1e-12, 300);
                  t.expect(r.has_value() && r->size() == 3, "three roots returned");
                  t.expect(r.has_value() && all_roots_vanish(c, *r, 1e-6),
                           "each |p(root)| < tol");
                  const double cbrt2 = std::cbrt(2.0);
                  t.expect(r.has_value() && has_root_near(*r, {cbrt2, 0.0}, 1e-6),
                           "real root ~ cbrt(2)");
                  // Conjugate pair cbrt(2) * exp(+-2*pi*i/3).
                  const double re = cbrt2 * std::cos(2.0 * std::numbers::pi / 3.0);
                  const double im = cbrt2 * std::sin(2.0 * std::numbers::pi / 3.0);
                  t.expect(r.has_value() && has_root_near(*r, {re, im}, 1e-6) &&
                               has_root_near(*r, {re, -im}, 1e-6),
                           "complex-conjugate pair located");
              })
        .test("durand_kerner_quintic_all_residuals",
              [&](TestContext& t) {
                  // x^5 - x - 1: five roots (one real ~1.1673, two conjugate pairs).
                  const std::array<double, 6> p{-1.0, -1.0, 0.0, 0.0, 0.0, 1.0};
                  auto r = num::durand_kerner(p, 1e-12, 500);
                  t.expect(r.has_value(), "durand_kerner converges on x^5 - x - 1");
                  t.expect(r.has_value() && r->size() == 5, "five roots returned");
                  t.expect(r.has_value() && all_roots_vanish(p, *r, 1e-6),
                           "each of the five |p(root)| < tol");
                  t.expect(r.has_value() && has_root_near(*r, {1.1673039783, 0.0}, 1e-6),
                           "the real root ~ 1.1673");
              })
        .test("durand_kerner_nonconvergence_is_not_implemented",
              [&](TestContext& t) {
                  // A hard quintic with a tiny iteration budget cannot reach tol.
                  const std::array<double, 6> p{-1.0, -1.0, 0.0, 0.0, 0.0, 1.0};
                  auto r = num::durand_kerner(p, 1e-14, 2);
                  t.expect(!r.has_value() && r.error() == MathError::not_implemented,
                           "starved iterations -> not_implemented (never a wrong value)");
              })
        .test("durand_kerner_empty_coeffs_domain_error",
              [&](TestContext& t) {
                  std::span<const double> empty{};
                  auto r = num::durand_kerner(empty, 1e-12, 100);
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "durand_kerner on empty coeffs -> domain_error");
              })
        .run();
}
