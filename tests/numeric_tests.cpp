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
        .run();
}
