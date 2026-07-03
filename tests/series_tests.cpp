// Tests for nimblecas.series: Taylor coefficients and truncated Taylor polynomials.
// @author Olumuyiwa Oluwasanmi
//
// Polynomial inputs are used throughout so every coefficient is exact and
// deterministic (no sin(0)/cos(0) style numeric evaluation is required).

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.series;
import nimblecas.testing;

using nimblecas::Expr;
using nimblecas::simplify;
using nimblecas::taylor_coefficients;
using nimblecas::taylor_polynomial;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// simplify(e), asserting success; returns Expr::integer(0) on the (unexpected) error
// branch and records the failure so the caller's later comparison also fails loudly.
auto simplified(TestContext& t, const Expr& e, std::string_view what) -> Expr {
    auto s = simplify(e);
    if (!s) {
        t.expect(false, std::format("{}: unexpected simplify error", what));
        return Expr::integer(0);
    }
    return *s;
}

// Evaluate an expression at var = value (as an integer), returning the simplified
// numeric result. Used to check function-equality of two polynomials without relying
// on the simplifier expanding powers of a binomial like (x - 1)^2.
auto eval_at(TestContext& t, const Expr& e, std::string_view var, std::int64_t value,
             std::string_view what) -> Expr {
    const Expr substituted = nimblecas::substitute(e, Expr::symbol(std::string(var)),
                                                   Expr::integer(value));
    return simplified(t, substituted, what);
}

}  // namespace

auto main() -> int {
    const auto x = Expr::symbol("x");
    const auto zero = Expr::integer(0);
    const auto one = Expr::integer(1);
    const auto two = Expr::integer(2);
    const auto three = Expr::integer(3);

    return TestSuite("nimblecas.series")
        .test("cubic_about_zero_coefficients",
              [&](TestContext& t) {
                  // f = x^3 about 0, order 3: c_0=c_1=c_2=0, c_3=1.
                  auto f = Expr::power(x, three);
                  auto coeffs = taylor_coefficients(f, "x", zero, 3);
                  t.expect(coeffs.has_value(), "taylor_coefficients(x^3, 0, 3) succeeds");
                  if (!coeffs) {
                      return;
                  }
                  t.expect_eq(coeffs->size(), std::size_t{4}, "returns 4 coefficients (c_0..c_3)");
                  t.expect((*coeffs)[0].is_equivalent_to(zero), "c_0 = 0");
                  t.expect((*coeffs)[1].is_equivalent_to(zero), "c_1 = 0");
                  t.expect((*coeffs)[2].is_equivalent_to(zero), "c_2 = 0");
                  t.expect((*coeffs)[3].is_equivalent_to(one), "c_3 = 1");
              })
        .test("general_polynomial_about_zero_coefficients",
              [&](TestContext& t) {
                  // f = x^2 + 2x + 1 about 0: c_0=1, c_1=2, c_2=1.
                  auto f = Expr::sum({Expr::power(x, two), two.mul(x), one});
                  auto coeffs = taylor_coefficients(f, "x", zero, 2);
                  t.expect(coeffs.has_value(), "taylor_coefficients(x^2+2x+1, 0, 2) succeeds");
                  if (!coeffs) {
                      return;
                  }
                  t.expect((*coeffs)[0].is_equivalent_to(one), "c_0 = 1");
                  t.expect((*coeffs)[1].is_equivalent_to(two), "c_1 = 2");
                  t.expect((*coeffs)[2].is_equivalent_to(one), "c_2 = 1");

                  // taylor_polynomial(order 2) reconstructs the original (point 0 needs
                  // no binomial expansion): compare canonical simplified forms.
                  auto poly = taylor_polynomial(f, "x", zero, 2);
                  t.expect(poly.has_value(), "taylor_polynomial succeeds");
                  if (poly) {
                      t.expect(poly->is_equivalent_to(simplified(t, f, "original f")),
                               std::format("reconstruct(x^2+2x+1) = {}", poly->to_string()));
                  }
              })
        .test("cubic_about_zero_polynomial_roundtrips",
              [&](TestContext& t) {
                  // taylor_polynomial(x^3, 0, 3) is_equivalent_to x^3.
                  auto f = Expr::power(x, three);
                  auto poly = taylor_polynomial(f, "x", zero, 3);
                  t.expect(poly.has_value(), "taylor_polynomial(x^3, 0, 3) succeeds");
                  if (poly) {
                      t.expect(poly->is_equivalent_to(simplified(t, f, "original x^3")),
                               std::format("reconstruct(x^3) = {}", poly->to_string()));
                  }
              })
        .test("quadratic_about_nonzero_point",
              [&](TestContext& t) {
                  // f = x^2 about point 1, order 2: since x^2 = 1 + 2(x-1) + (x-1)^2,
                  // c_0=1, c_1=2, c_2=1.
                  auto f = Expr::power(x, two);
                  auto coeffs = taylor_coefficients(f, "x", one, 2);
                  t.expect(coeffs.has_value(), "taylor_coefficients(x^2, 1, 2) succeeds");
                  if (!coeffs) {
                      return;
                  }
                  t.expect((*coeffs)[0].is_equivalent_to(one), "c_0 = 1");
                  t.expect((*coeffs)[1].is_equivalent_to(two), "c_1 = 2");
                  t.expect((*coeffs)[2].is_equivalent_to(one), "c_2 = 1");

                  // The reconstructed polynomial 1 + 2(x-1) + (x-1)^2 equals x^2 as a
                  // function. The simplifier does not expand the binomial powers, so
                  // equivalence is verified numerically at several sample points rather
                  // than by structural equality against x^2.
                  auto poly = taylor_polynomial(f, "x", one, 2);
                  t.expect(poly.has_value(), "taylor_polynomial(x^2, 1, 2) succeeds");
                  if (poly) {
                      for (std::int64_t v : {-2, 0, 3, 5}) {
                          auto got = eval_at(t, *poly, "x", v,
                                             std::format("reconstruct(x^2) at x={}", v));
                          auto want = eval_at(t, f, "x", v, std::format("x^2 at x={}", v));
                          t.expect(got.is_equivalent_to(want),
                                   std::format("reconstruct(x^2)={} matches x^2 at x={}",
                                               poly->to_string(), v));
                      }
                  }
              })
        .test("negative_order_is_domain_error",
              [&](TestContext& t) {
                  auto coeffs = taylor_coefficients(Expr::power(x, two), "x", zero, -1);
                  t.expect(!coeffs.has_value(), "negative order is rejected");
                  t.expect(coeffs.error() == nimblecas::MathError::domain_error,
                           "negative order yields domain_error");

                  auto poly = taylor_polynomial(Expr::power(x, two), "x", zero, -1);
                  t.expect(!poly.has_value(), "taylor_polynomial rejects negative order");
                  t.expect(poly.error() == nimblecas::MathError::domain_error,
                           "taylor_polynomial propagates domain_error");
              })
        .run();
}
