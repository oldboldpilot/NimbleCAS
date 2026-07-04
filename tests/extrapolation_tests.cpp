// Tests for nimblecas.extrapolation: Richardson / Romberg / Aitken / Wynn epsilon.
// @author Olumuyiwa Oluwasanmi
//
// Exercises the HONESTY BOUNDARY directly: the `*_exact` / Rational paths must return
// EXACT fractions (compared with ==), the double paths converge numerically, and stalled
// or too-short inputs return MathError::domain_error rather than a wrong number.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.extrapolation;
import nimblecas.testing;

using nimblecas::MathError;
using nimblecas::make_error;
using nimblecas::Rational;
using nimblecas::Result;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;
namespace nx = nimblecas;

namespace {

// Exact rational from a fraction, unwrapped for terse test data.
[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}
[[nodiscard]] auto ri(std::int64_t v) -> Rational { return Rational::from_int(v); }

// Absolute-tolerance comparison for the numerical paths.
[[nodiscard]] auto close(double got, double expected) -> bool {
    return std::abs(got - expected) < 1e-6;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.extrapolation")
        // ---- Richardson: one exact step recovers A from A(h) = A + c h^2 ----
        .test("richardson_step_exact_recovers_A",
              [&](TestContext& t) {
                  // A = 5, c = 3, p = 2: A(1) = 8, A(1/2) = 5 + 3/4 = 23/4.
                  auto r = nx::richardson_step(ri(8), rat(23, 4), 2, 2);
                  t.expect(r.has_value(), "richardson_step succeeds");
                  t.expect(r.has_value() && *r == ri(5), "one step recovers A = 5 exactly over Q");
              })
        .test("richardson_tableau_exact_recovers_A",
              [&](TestContext& t) {
                  const std::array<Rational, 2> a{ri(8), rat(23, 4)};  // A(1), A(1/2)
                  auto tab = nx::richardson_tableau(std::span<const Rational>{a}, 2, 2);
                  t.expect(tab.has_value(), "tableau builds");
                  t.expect(tab.has_value() && tab->best == ri(5), "best == 5 exactly");
              })
        .test("richardson_tableau_double_converges",
              [&](TestContext& t) {
                  // A(h) = 5 + 3 h^2 sampled at h = 1, 1/2, 1/4.
                  const std::array<double, 3> a{8.0, 5.75, 5.1875};
                  auto tab = nx::richardson_tableau(std::span<const double>{a}, 2.0, 2.0);
                  t.expect(tab.has_value(), "double tableau builds");
                  t.expect(tab.has_value() && close(tab->best, 5.0), "best ~ 5");
              })
        .test("richardson_step_domain_errors",
              [&](TestContext& t) {
                  auto bad_r = nx::richardson_step(ri(8), rat(23, 4), 1, 2);  // r < 2
                  t.expect(!bad_r && bad_r.error() == MathError::domain_error, "r < 2 -> domain_error");
                  auto bad_p = nx::richardson_step(ri(8), rat(23, 4), 2, 0);  // p < 1
                  t.expect(!bad_p && bad_p.error() == MathError::domain_error, "p < 1 -> domain_error");
                  std::span<const Rational> empty{};
                  auto e = nx::richardson_tableau(empty, 2, 2);
                  t.expect(!e && e.error() == MathError::domain_error, "empty seq -> domain_error");
              })

        // ---- Romberg: EXACT over Q on rational integrand ∫_0^1 x^2 dx = 1/3 ----
        .test("romberg_exact_x_squared",
              [&](TestContext& t) {
                  auto sq = [](const Rational& x) -> Result<Rational> { return x.multiply(x); };
                  auto tab = nx::romberg_exact(sq, ri(0), ri(1), 3);
                  t.expect(tab.has_value(), "romberg_exact builds");
                  t.expect(tab.has_value() && tab->best == rat(1, 3),
                           "best == 1/3 EXACTLY over Q (Simpson column is exact for x^2)");
              })
        .test("romberg_double_transcendental",
              [&](TestContext& t) {
                  // ∫_0^1 e^x dx = e - 1, a smooth integrand: Romberg converges fast.
                  auto f = [](double x) -> double { return std::exp(x); };
                  auto tab = nx::romberg(f, 0.0, 1.0, 6);
                  t.expect(tab.has_value(), "romberg builds");
                  t.expect(tab.has_value() && close(tab->best, std::numbers::e - 1.0),
                           "best ~ e - 1");
              })
        .test("romberg_reversed_bounds_domain_error",
              [&](TestContext& t) {
                  auto f = [](double x) -> double { return x; };
                  auto tab = nx::romberg(f, 1.0, 0.0, 3);  // b < a
                  t.expect(!tab && tab.error() == MathError::domain_error, "b < a -> domain_error");
              })

        // ---- Aitken Δ²: accelerate geometric partial sums to the exact limit ----
        .test("aitken_exact_geometric",
              [&](TestContext& t) {
                  // Partial sums of Σ (1/2)^k: 1, 3/2, 7/4 -> limit 2.
                  const std::array<Rational, 3> x{ri(1), rat(3, 2), rat(7, 4)};
                  auto acc = nx::aitken(std::span<const Rational>{x});
                  t.expect(acc.has_value() && acc->size() == 1, "one accelerated value");
                  t.expect(acc.has_value() && (*acc)[0] == ri(2), "recovers limit 2 exactly over Q");
              })
        .test("aitken_stalled_delta2_zero",
              [&](TestContext& t) {
                  const std::array<Rational, 3> x{ri(2), ri(2), ri(2)};  // Δ² = 0
                  auto acc = nx::aitken(std::span<const Rational>{x});
                  t.expect(!acc && acc.error() == MathError::domain_error, "Δ² = 0 -> domain_error");
              })
        .test("aitken_too_short",
              [&](TestContext& t) {
                  const std::array<Rational, 2> x{ri(1), ri(2)};
                  auto acc = nx::aitken(std::span<const Rational>{x});
                  t.expect(!acc && acc.error() == MathError::domain_error, "< 3 terms -> domain_error");
              })

        // ---- Wynn epsilon (Shanks): sum a rational partial-sum sequence exactly ----
        .test("wynn_epsilon_exact_geometric",
              [&](TestContext& t) {
                  const std::array<Rational, 3> s{ri(1), rat(3, 2), rat(7, 4)};
                  auto best = nx::wynn_epsilon(std::span<const Rational>{s});
                  t.expect(best.has_value(), "epsilon table builds");
                  t.expect(best.has_value() && *best == ri(2), "ε_2 == 2 exactly over Q");
              })
        .test("wynn_epsilon_double_geometric",
              [&](TestContext& t) {
                  const std::array<double, 3> s{1.0, 1.5, 1.75};
                  auto best = nx::wynn_epsilon(std::span<const double>{s});
                  t.expect(best.has_value() && close(*best, 2.0), "numerical ε ~ 2");
              })
        .test("wynn_epsilon_stalled_division_by_zero",
              [&](TestContext& t) {
                  const std::array<Rational, 3> s{ri(1), ri(1), ri(1)};  // constant -> ε diff 0
                  auto best = nx::wynn_epsilon(std::span<const Rational>{s});
                  t.expect(!best && best.error() == MathError::domain_error,
                           "ε difference 0 -> domain_error");
              })
        .test("wynn_epsilon_too_short",
              [&](TestContext& t) {
                  const std::array<Rational, 2> s{ri(1), rat(3, 2)};
                  auto best = nx::wynn_epsilon(std::span<const Rational>{s});
                  t.expect(!best && best.error() == MathError::domain_error, "< 3 terms -> domain_error");
              })

        // ---- Richardson-accelerated derivative ----
        .test("richardson_derivative_exact_polynomial",
              [&](TestContext& t) {
                  // p(x) = x^3, p'(1) = 3. Central diff has a pure h^2 error; one Richardson
                  // level (2 rows) recovers 3 EXACTLY over Q.
                  auto cube = [](const Rational& x) -> Result<Rational> {
                      auto x2 = x.multiply(x);
                      if (!x2) {
                          return make_error<Rational>(x2.error());
                      }
                      return x2->multiply(x);
                  };
                  auto tab = nx::richardson_derivative_exact(cube, ri(1), ri(1), 1);
                  t.expect(tab.has_value(), "exact derivative tableau builds");
                  t.expect(tab.has_value() && tab->best == ri(3), "p'(1) == 3 exactly over Q");
              })
        .test("richardson_derivative_numeric_sin",
              [&](TestContext& t) {
                  // d/dx sin at 0 is cos(0) = 1.
                  auto f = [](double x) -> double { return std::sin(x); };
                  auto tab = nx::richardson_derivative(f, 0.0, 0.5, 5);
                  t.expect(tab.has_value(), "numeric derivative tableau builds");
                  t.expect(tab.has_value() && close(tab->best, 1.0), "sin'(0) ~ 1");
              })
        .run();
}
