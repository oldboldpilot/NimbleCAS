// Tests for nimblecas.pade: [m/n] Pade approximants of truncated power series.
// @author Olumuyiwa Oluwasanmi
//
// Every case uses exact rational series coefficients and every expected P/Q
// coefficient is verified by hand in the comments, so the tests are fully
// deterministic (no floating point anywhere).

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.powerseries;
import nimblecas.pade;
import nimblecas.testing;

using nimblecas::pade;
using nimblecas::PowerSeries;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Small exact rational helpers. Inputs here are tiny literals, so make()/from_int
// never fail; unwrapping directly keeps the test bodies readable.
auto ri(std::int64_t v) -> Rational { return Rational::from_int(v); }
auto rat(std::int64_t num, std::int64_t den) -> Rational { return *Rational::make(num, den); }

// Build a PowerSeries from the given coefficients at the given order, asserting success.
auto series(TestContext& t, std::vector<Rational> coeffs, std::size_t order,
            std::string_view what) -> PowerSeries {
    auto s = PowerSeries::from_coeffs(std::move(coeffs), order);
    if (!s) {
        t.expect(false, std::format("{}: unexpected series construction error", what));
        return *PowerSeries::zero(1);
    }
    return *s;
}

// Assert that `poly` has exactly the expected ascending coefficients (comparing one
// past the top to confirm there is no unexpected higher-degree term).
auto expect_coeffs(TestContext& t, const RationalPoly& poly,
                   const std::vector<Rational>& expected, std::string_view what) -> void {
    for (std::size_t i = 0; i < expected.size(); ++i) {
        t.expect(poly.coefficient(i) == expected[i],
                 std::format("{}: coefficient[{}] = {}", what, i, expected[i].to_string()));
    }
    t.expect(poly.coefficient(expected.size()) == ri(0),
             std::format("{}: no term beyond degree {}", what, expected.size() - 1));
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.pade")
        .test("exp_1_1_approximant",
              [](TestContext& t) {
                  // exp: c = {1, 1, 1/2, 1/6}. [1/1] Pade.
                  //   n=1 system: A[0][0] = c_1 = 1, b[0] = -c_2 = -1/2 => q_1 = -1/2.
                  //   P: p_0 = q_0 c_0 = 1; p_1 = q_0 c_1 + q_1 c_0 = 1 - 1/2 = 1/2.
                  // So P = 1 + x/2, Q = 1 - x/2.
                  auto s = series(t, {ri(1), ri(1), rat(1, 2), rat(1, 6)}, 4, "exp order 4");
                  auto r = pade(s, 1, 1);
                  t.expect(r.has_value(), "pade(exp, 1, 1) succeeds");
                  if (!r) {
                      return;
                  }
                  expect_coeffs(t, r->first, {ri(1), rat(1, 2)}, "P = 1 + x/2");
                  expect_coeffs(t, r->second, {ri(1), rat(-1, 2)}, "Q = 1 - x/2");
              })
        .test("exp_2_2_approximant",
              [](TestContext& t) {
                  // exp: c = {1, 1, 1/2, 1/6, 1/24}. [2/2] Pade.
                  //   System (rows k=3,4):
                  //     (1/2) q_1 + 1 q_2 = -1/6
                  //     (1/6) q_1 + (1/2) q_2 = -1/24
                  //   Solution q_1 = -1/2, q_2 = 1/12 (checked: row0 -1/4+1/12=-1/6,
                  //   row1 -1/12+1/24=-1/24).
                  //   P: p_0 = 1; p_1 = c_1 + q_1 c_0 = 1 - 1/2 = 1/2;
                  //      p_2 = c_2 + q_1 c_1 + q_2 c_0 = 1/2 - 1/2 + 1/12 = 1/12.
                  // So P = 1 + x/2 + x^2/12, Q = 1 - x/2 + x^2/12.
                  auto s = series(t, {ri(1), ri(1), rat(1, 2), rat(1, 6), rat(1, 24)}, 5,
                                  "exp order 5");
                  auto r = pade(s, 2, 2);
                  t.expect(r.has_value(), "pade(exp, 2, 2) succeeds");
                  if (!r) {
                      return;
                  }
                  expect_coeffs(t, r->first, {ri(1), rat(1, 2), rat(1, 12)},
                                "P = 1 + x/2 + x^2/12");
                  expect_coeffs(t, r->second, {ri(1), rat(-1, 2), rat(1, 12)},
                                "Q = 1 - x/2 + x^2/12");
              })
        .test("geometric_0_1_exact_recovery",
              [](TestContext& t) {
                  // 1/(1-x): c = {1,1,1,1}. [0/1] Pade must recover P=1, Q=1-x.
                  //   n=1: A[0][0] = c_0 = 1, b[0] = -c_1 = -1 => q_1 = -1.
                  //   P: p_0 = q_0 c_0 = 1 (m=0, so only p_0).
                  auto s = series(t, {ri(1), ri(1), ri(1), ri(1)}, 4, "1/(1-x) order 4");
                  auto r = pade(s, 0, 1);
                  t.expect(r.has_value(), "pade(1/(1-x), 0, 1) succeeds");
                  if (!r) {
                      return;
                  }
                  expect_coeffs(t, r->first, {ri(1)}, "P = 1");
                  expect_coeffs(t, r->second, {ri(1), ri(-1)}, "Q = 1 - x");
              })
        .test("geometric_1_1_degenerate_but_valid",
              [](TestContext& t) {
                  // 1/(1-x): c = {1,1,1,1}. [1/1] Pade is degenerate.
                  //   n=1: A[0][0] = c_1 = 1, b[0] = -c_2 = -1 => q_1 = -1, Q = 1 - x.
                  //   P: p_0 = c_0 = 1; p_1 = c_1 + q_1 c_0 = 1 - 1 = 0 (P trims to 1).
                  auto s = series(t, {ri(1), ri(1), ri(1), ri(1)}, 4, "1/(1-x) order 4");
                  auto r = pade(s, 1, 1);
                  t.expect(r.has_value(), "pade(1/(1-x), 1, 1) succeeds");
                  if (!r) {
                      return;
                  }
                  t.expect(r->first.coefficient(0) == ri(1), "P: p_0 = 1");
                  t.expect(r->first.coefficient(1) == ri(0), "P: p_1 = 0 (degenerate)");
                  expect_coeffs(t, r->second, {ri(1), ri(-1)}, "Q = 1 - x");
              })
        .test("n_zero_is_series_truncation",
              [](TestContext& t) {
                  // n = 0: Q = 1, P = degree-m truncation of the series.
                  // c = {1, 1, 1/2, 1/6}; pade(., 2, 0) => P = 1 + x + x^2/2, Q = 1.
                  auto s = series(t, {ri(1), ri(1), rat(1, 2), rat(1, 6)}, 4, "exp order 4");
                  auto r = pade(s, 2, 0);
                  t.expect(r.has_value(), "pade(s, 2, 0) succeeds");
                  if (!r) {
                      return;
                  }
                  expect_coeffs(t, r->first, {ri(1), ri(1), rat(1, 2)},
                                "P = truncation 1 + x + x^2/2");
                  expect_coeffs(t, r->second, {ri(1)}, "Q = 1");
              })
        .test("consistency_roundtrip_from_rational",
              [](TestContext& t) {
                  // (1 + 2x)/(1 - x) expands to 1 + 3x + 3x^2 + 3x^3 + ...
                  //   ((1+2x) * (1 + x + x^2 + x^3) truncated) = 1 + 3x + 3x^2 + 3x^3.
                  // The [1/1] Pade of that series must return P = 1 + 2x, Q = 1 - x.
                  //   n=1: A[0][0] = c_1 = 3, b[0] = -c_2 = -3 => q_1 = -1, Q = 1 - x.
                  //   P: p_0 = c_0 = 1; p_1 = c_1 + q_1 c_0 = 3 - 1 = 2.
                  auto s = series(t, {ri(1), ri(3), ri(3), ri(3)}, 4, "(1+2x)/(1-x) order 4");
                  auto r = pade(s, 1, 1);
                  t.expect(r.has_value(), "pade of (1+2x)/(1-x) series succeeds");
                  if (!r) {
                      return;
                  }
                  expect_coeffs(t, r->first, {ri(1), ri(2)}, "P = 1 + 2x");
                  expect_coeffs(t, r->second, {ri(1), ri(-1)}, "Q = 1 - x");
              })
        .test("order_too_small_is_domain_error",
              [](TestContext& t) {
                  // [1/1] needs c_0..c_2 (order >= 3); an order-2 series is rejected.
                  auto s = series(t, {ri(1), ri(1)}, 2, "order 2");
                  auto r = pade(s, 1, 1);
                  t.expect(!r.has_value(), "insufficient order is rejected");
                  t.expect(r.error() == nimblecas::MathError::domain_error,
                           "insufficient order yields domain_error");
              })
        .test("singular_system_is_domain_error",
              [](TestContext& t) {
                  // c = {1, 0, 1, 1}: for [1/1] the 1x1 Toeplitz matrix is [c_1] = [0],
                  // which is singular, so solve() fails and the domain_error propagates.
                  auto s = series(t, {ri(1), ri(0), ri(1), ri(1)}, 4, "singular order 4");
                  auto r = pade(s, 1, 1);
                  t.expect(!r.has_value(), "singular Toeplitz system is rejected");
                  t.expect(r.error() == nimblecas::MathError::domain_error,
                           "singular system yields domain_error (from solve)");
              })
        .run();
}
