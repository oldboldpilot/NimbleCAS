// Tests for nimblecas.powerseries: truncated formal power series over Q.
// @author Olumuyiwa Oluwasanmi
//
// Every case uses exact Rational coefficients so results are deterministic: no
// floating point ever enters the picture.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.powerseries;
import nimblecas.testing;

using nimblecas::MathError;
using nimblecas::PowerSeries;
using nimblecas::Rational;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Rational num/den, asserting the construction succeeds; returns 0 on the unexpected
// error branch and records a failure so the caller's comparison also fails loudly.
auto rat(TestContext& t, std::int64_t num, std::int64_t den, std::string_view what)
    -> Rational {
    auto r = Rational::make(num, den);
    if (!r) {
        t.expect(false, std::format("{}: unexpected Rational::make error", what));
        return Rational::from_int(0);
    }
    return *r;
}

auto ri(std::int64_t v) -> Rational { return Rational::from_int(v); }

// Build a series from integer coefficients at a given order, asserting success.
auto series(TestContext& t, std::vector<std::int64_t> ints, std::size_t order,
            std::string_view what) -> PowerSeries {
    std::vector<Rational> coeffs;
    coeffs.reserve(ints.size());
    for (auto v : ints) {
        coeffs.push_back(Rational::from_int(v));
    }
    auto s = PowerSeries::from_coeffs(std::move(coeffs), order);
    if (!s) {
        t.expect(false, std::format("{}: unexpected from_coeffs error", what));
        return *PowerSeries::zero(order);
    }
    return *s;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.powerseries")
        .test("order_zero_factories_are_domain_errors",
              [&](TestContext& t) {
                  auto a = PowerSeries::from_coeffs({ri(1)}, 0);
                  t.expect(!a.has_value() && a.error() == MathError::domain_error,
                           "from_coeffs order 0 -> domain_error");
                  auto b = PowerSeries::constant(ri(1), 0);
                  t.expect(!b.has_value() && b.error() == MathError::domain_error,
                           "constant order 0 -> domain_error");
                  auto c = PowerSeries::variable(0);
                  t.expect(!c.has_value() && c.error() == MathError::domain_error,
                           "variable order 0 -> domain_error");
                  auto d = PowerSeries::zero(0);
                  t.expect(!d.has_value() && d.error() == MathError::domain_error,
                           "zero order 0 -> domain_error");
                  auto e = PowerSeries::one(0);
                  t.expect(!e.has_value() && e.error() == MathError::domain_error,
                           "one order 0 -> domain_error");
              })
        .test("from_coeffs_pads_and_truncates",
              [&](TestContext& t) {
                  auto padded = PowerSeries::from_coeffs({ri(1), ri(2)}, 4);
                  t.expect(padded.has_value(), "from_coeffs pads to order 4");
                  if (padded) {
                      t.expect_eq(padded->order(), std::size_t{4}, "order is 4");
                      t.expect_eq(padded->coefficient(1), ri(2), "c_1 = 2");
                      t.expect(padded->coefficient(2).is_zero(), "c_2 padded to 0");
                      t.expect(padded->coefficient(9).is_zero(), "out-of-range coeff is 0");
                  }
                  auto truncated = PowerSeries::from_coeffs({ri(1), ri(2), ri(3), ri(4)}, 2);
                  t.expect(truncated.has_value(), "from_coeffs truncates to order 2");
                  if (truncated) {
                      t.expect_eq(truncated->order(), std::size_t{2}, "order is 2");
                      t.expect_eq(truncated->coefficient(1), ri(2), "kept c_1 = 2");
                      t.expect_eq(truncated->coefficient(2), ri(0), "dropped c_2");
                  }
              })
        .test("multiply_binomial_square",
              [&](TestContext& t) {
                  // (1 + x)(1 + x) = 1 + 2x + x^2.
                  auto a = series(t, {1, 1, 0}, 3, "1+x");
                  auto product = a.multiply(a);
                  t.expect(product.has_value(), "(1+x)^2 multiply succeeds");
                  if (product) {
                      t.expect_eq(product->coefficient(0), ri(1), "c_0 = 1");
                      t.expect_eq(product->coefficient(1), ri(2), "c_1 = 2");
                      t.expect_eq(product->coefficient(2), ri(1), "c_2 = 1");
                  }
              })
        .test("multiply_mismatched_order_is_domain_error",
              [&](TestContext& t) {
                  auto a = series(t, {1, 1}, 2, "order2");
                  auto b = series(t, {1, 1, 1}, 3, "order3");
                  auto product = a.multiply(b);
                  t.expect(!product.has_value() &&
                               product.error() == MathError::domain_error,
                           "mismatched orders -> domain_error");
              })
        .test("inverse_geometric_series",
              [&](TestContext& t) {
                  // 1 / (1 - x) = 1 + x + x^2 + x^3 + x^4.
                  auto denom = series(t, {1, -1, 0, 0, 0}, 5, "1-x");
                  auto inv = denom.inverse();
                  t.expect(inv.has_value(), "inverse of 1-x succeeds");
                  if (inv) {
                      for (std::size_t k = 0; k < 5; ++k) {
                          t.expect_eq(inv->coefficient(k), ri(1),
                                      std::format("1/(1-x) coeff {} = 1", k));
                      }
                  }
              })
        .test("inverse_zero_constant_term_is_domain_error",
              [&](TestContext& t) {
                  // c_0 == 0 -> no multiplicative inverse.
                  auto s = series(t, {0, 1, 0}, 3, "x");
                  auto inv = s.inverse();
                  t.expect(!inv.has_value() && inv.error() == MathError::domain_error,
                           "inverse with c_0 = 0 -> domain_error");
              })
        .test("divide_matches_multiply_by_inverse",
              [&](TestContext& t) {
                  // (1) / (1 - x) = 1 + x + x^2 + ...
                  auto one = series(t, {1, 0, 0, 0}, 4, "one");
                  auto denom = series(t, {1, -1, 0, 0}, 4, "1-x");
                  auto quotient = one.divide(denom);
                  t.expect(quotient.has_value(), "1/(1-x) divide succeeds");
                  if (quotient) {
                      for (std::size_t k = 0; k < 4; ++k) {
                          t.expect_eq(quotient->coefficient(k), ri(1),
                                      std::format("divide coeff {} = 1", k));
                      }
                  }
              })
        .test("derivative_keeps_order_with_trailing_zero",
              [&](TestContext& t) {
                  // f = 1 + 2x + 3x^2 + 4x^3 (order 4). f' = 2 + 6x + 12x^2 (+ 0 x^3).
                  auto f = series(t, {1, 2, 3, 4}, 4, "f");
                  auto d = f.derivative();
                  t.expect(d.has_value(), "derivative succeeds");
                  if (d) {
                      t.expect_eq(d->order(), std::size_t{4}, "order preserved at 4");
                      t.expect_eq(d->coefficient(0), ri(2), "d_0 = 2");
                      t.expect_eq(d->coefficient(1), ri(6), "d_1 = 6");
                      t.expect_eq(d->coefficient(2), ri(12), "d_2 = 12");
                      t.expect(d->coefficient(3).is_zero(), "d_3 = 0 (trailing zero)");
                  }
              })
        .test("integrate_then_derivative_roundtrips",
              [&](TestContext& t) {
                  // Top coefficient is 0 so nothing is lost to truncation on the way back.
                  auto f = series(t, {2, 3, 4, 5, 0}, 5, "f");
                  auto integrated = f.integrate();
                  t.expect(integrated.has_value(), "integrate succeeds");
                  if (!integrated) {
                      return;
                  }
                  t.expect(integrated->coefficient(0).is_zero(), "integral constant term 0");
                  auto back = integrated->derivative();
                  t.expect(back.has_value(), "derivative of integral succeeds");
                  if (back) {
                      for (std::size_t k = 0; k < 5; ++k) {
                          t.expect_eq(back->coefficient(k), f.coefficient(k),
                                      std::format("roundtrip coeff {}", k));
                      }
                  }
              })
        .test("derivative_then_integrate_recovers_up_to_constant",
              [&](TestContext& t) {
                  // integrate(f') = f - c_0.
                  auto f = series(t, {7, 3, 4, 5}, 4, "f");
                  auto d = f.derivative();
                  t.expect(d.has_value(), "derivative succeeds");
                  if (!d) {
                      return;
                  }
                  auto back = d->integrate();
                  t.expect(back.has_value(), "integrate of derivative succeeds");
                  if (back) {
                      t.expect(back->coefficient(0).is_zero(), "constant term dropped to 0");
                      for (std::size_t k = 1; k < 4; ++k) {
                          t.expect_eq(back->coefficient(k), f.coefficient(k),
                                      std::format("recovered coeff {}", k));
                      }
                  }
              })
        .test("exp_of_x_is_reciprocal_factorials",
              [&](TestContext& t) {
                  // exp(x): coeffs 1, 1, 1/2, 1/6, 1/24.
                  auto x = series(t, {0, 1, 0, 0, 0}, 5, "x");
                  auto e = x.exp();
                  t.expect(e.has_value(), "exp(x) succeeds");
                  if (e) {
                      t.expect_eq(e->coefficient(0), ri(1), "e_0 = 1");
                      t.expect_eq(e->coefficient(1), ri(1), "e_1 = 1");
                      t.expect_eq(e->coefficient(2), rat(t, 1, 2, "1/2"), "e_2 = 1/2");
                      t.expect_eq(e->coefficient(3), rat(t, 1, 6, "1/6"), "e_3 = 1/6");
                      t.expect_eq(e->coefficient(4), rat(t, 1, 24, "1/24"), "e_4 = 1/24");
                  }
              })
        .test("exp_nonzero_constant_term_is_domain_error",
              [&](TestContext& t) {
                  auto s = series(t, {1, 1, 0}, 3, "1+x");
                  auto e = s.exp();
                  t.expect(!e.has_value() && e.error() == MathError::domain_error,
                           "exp with c_0 != 0 -> domain_error");
              })
        .test("log_of_one_plus_x",
              [&](TestContext& t) {
                  // log(1 + x): coeffs 0, 1, -1/2, 1/3, -1/4.
                  auto s = series(t, {1, 1, 0, 0, 0}, 5, "1+x");
                  auto l = s.log();
                  t.expect(l.has_value(), "log(1+x) succeeds");
                  if (l) {
                      t.expect_eq(l->coefficient(0), ri(0), "l_0 = 0");
                      t.expect_eq(l->coefficient(1), ri(1), "l_1 = 1");
                      t.expect_eq(l->coefficient(2), rat(t, -1, 2, "-1/2"), "l_2 = -1/2");
                      t.expect_eq(l->coefficient(3), rat(t, 1, 3, "1/3"), "l_3 = 1/3");
                      t.expect_eq(l->coefficient(4), rat(t, -1, 4, "-1/4"), "l_4 = -1/4");
                  }
              })
        .test("log_wrong_constant_term_is_domain_error",
              [&](TestContext& t) {
                  auto s = series(t, {2, 1, 0}, 3, "2+x");
                  auto l = s.log();
                  t.expect(!l.has_value() && l.error() == MathError::domain_error,
                           "log with c_0 != 1 -> domain_error");
              })
        .test("exp_log_identity",
              [&](TestContext& t) {
                  // exp(log(1 + x)) == 1 + x (truncated).
                  auto s = series(t, {1, 1, 0, 0, 0}, 5, "1+x");
                  auto l = s.log();
                  t.expect(l.has_value(), "log(1+x) succeeds");
                  if (!l) {
                      return;
                  }
                  auto e = l->exp();  // log(1+x) has zero constant term
                  t.expect(e.has_value(), "exp(log(1+x)) succeeds");
                  if (e) {
                      t.expect(e->is_equal(s),
                               std::format("exp(log(1+x)) = {}", e->to_string()));
                  }
              })
        .test("compose_evaluates_this_of_g",
              [&](TestContext& t) {
                  // f(y) = 1 + y + y^2, g = x + x^2 (g_0 = 0).
                  // f(g) = 1 + (x + x^2) + (x + x^2)^2
                  //      = 1 + (x + x^2) + (x^2 + 2x^3 + x^4)
                  //      = 1 + x + 2x^2 + 2x^3 + ...  (truncated at order 4: 1, 1, 2, 2)
                  auto f = series(t, {1, 1, 1, 0}, 4, "f");
                  auto g = series(t, {0, 1, 1, 0}, 4, "g");
                  auto comp = f.compose(g);
                  t.expect(comp.has_value(), "compose succeeds");
                  if (comp) {
                      t.expect_eq(comp->coefficient(0), ri(1), "c_0 = 1");
                      t.expect_eq(comp->coefficient(1), ri(1), "c_1 = 1");
                      t.expect_eq(comp->coefficient(2), ri(2), "c_2 = 2");
                      t.expect_eq(comp->coefficient(3), ri(2), "c_3 = 2");
                  }
              })
        .test("compose_nonzero_inner_constant_is_domain_error",
              [&](TestContext& t) {
                  auto f = series(t, {1, 1, 1}, 3, "f");
                  auto g = series(t, {1, 1, 0}, 3, "g_with_const");
                  auto comp = f.compose(g);
                  t.expect(!comp.has_value() && comp.error() == MathError::domain_error,
                           "compose with g_0 != 0 -> domain_error");
              })
        .test("compose_mismatched_order_is_domain_error",
              [&](TestContext& t) {
                  auto f = series(t, {1, 1}, 2, "f");
                  auto g = series(t, {0, 1, 0}, 3, "g");
                  auto comp = f.compose(g);
                  t.expect(!comp.has_value() && comp.error() == MathError::domain_error,
                           "compose mismatched orders -> domain_error");
              })
        .run();
}
