// Tests for nimblecas.bigpowerseries: truncated formal power series over Q with exact
// unbounded BigRational coefficients.
// @author Olumuyiwa Oluwasanmi
//
// These cases exercise precisely the range that the int64-backed nimblecas.powerseries
// CANNOT represent: the exp(x) coefficients 1/n! for n >= 21 whose denominators n! exceed
// int64 (21! ~ 5.1e19 > 9.2e18 = INT64_MAX). Every coefficient is hand-verified and exact
// -- no floating point anywhere -- and a cross-check pins the low-order BigPowerSeries
// coefficients against the int64 module so the two tiers agree where both are valid.

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.bigpowerseries;
import nimblecas.ratpoly;       // int64 Rational, for the cross-check
import nimblecas.powerseries;   // int64 PowerSeries, for the cross-check
import nimblecas.testing;

using nimblecas::BigInt;
using nimblecas::BigPowerSeries;
using nimblecas::BigRational;
using nimblecas::MathError;
using nimblecas::PowerSeries;
using nimblecas::Rational;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// BigRational num/den, asserting construction succeeds; on the unexpected error branch
// records a failure so the caller's comparison also fails loudly.
auto brat(TestContext& t, std::int64_t num, std::int64_t den, std::string_view what)
    -> BigRational {
    auto r = BigRational::make(BigInt::from_i64(num), BigInt::from_i64(den));
    if (!r) {
        t.expect(false, std::format("{}: unexpected BigRational::make error", what));
        return BigRational::from_int(0);
    }
    return *r;
}

auto bri(std::int64_t v) -> BigRational { return BigRational::from_int(v); }

// Build a BigPowerSeries from integer coefficients at a given order, asserting success.
auto bseries(TestContext& t, std::vector<std::int64_t> ints, std::size_t order,
             std::string_view what) -> BigPowerSeries {
    std::vector<BigRational> coeffs;
    coeffs.reserve(ints.size());
    for (auto v : ints) {
        coeffs.push_back(BigRational::from_int(v));
    }
    auto s = BigPowerSeries::from_coeffs(std::move(coeffs), order);
    if (!s) {
        t.expect(false, std::format("{}: unexpected from_coeffs error", what));
        return *BigPowerSeries::zero(order);
    }
    return *s;
}

// n! as an exact BigInt (0! == 1). This is the value that overflows int64 for n >= 21.
auto factorial(std::uint64_t n) -> BigInt {
    BigInt result = BigInt::from_u64(1);
    for (std::uint64_t i = 2; i <= n; ++i) {
        result = result.multiply(BigInt::from_u64(i));
    }
    return result;
}

// The exact rational 1/n!, built with a BigInt denominator so it is representable even
// when n! far exceeds int64.
auto inv_factorial(TestContext& t, std::uint64_t n, std::string_view what) -> BigRational {
    auto r = BigRational::make(BigInt::from_i64(1), factorial(n));
    if (!r) {
        t.expect(false, std::format("{}: unexpected BigRational::make error", what));
        return BigRational::from_int(0);
    }
    return *r;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.bigpowerseries")
        .test("order_zero_factories_are_domain_errors",
              [&](TestContext& t) {
                  auto a = BigPowerSeries::from_coeffs({bri(1)}, 0);
                  t.expect(!a.has_value() && a.error() == MathError::domain_error,
                           "from_coeffs order 0 -> domain_error");
                  auto b = BigPowerSeries::constant(bri(1), 0);
                  t.expect(!b.has_value() && b.error() == MathError::domain_error,
                           "constant order 0 -> domain_error");
                  auto c = BigPowerSeries::variable(0);
                  t.expect(!c.has_value() && c.error() == MathError::domain_error,
                           "variable order 0 -> domain_error");
                  auto d = BigPowerSeries::zero(0);
                  t.expect(!d.has_value() && d.error() == MathError::domain_error,
                           "zero order 0 -> domain_error");
                  auto e = BigPowerSeries::one(0);
                  t.expect(!e.has_value() && e.error() == MathError::domain_error,
                           "one order 0 -> domain_error");
              })
        .test("from_coeffs_pads_and_truncates",
              [&](TestContext& t) {
                  auto padded = BigPowerSeries::from_coeffs({bri(1), bri(2)}, 4);
                  t.expect(padded.has_value(), "from_coeffs pads to order 4");
                  if (padded) {
                      t.expect_eq(padded->order(), std::size_t{4}, "order is 4");
                      t.expect_eq(padded->coefficient(1), bri(2), "c_1 = 2");
                      t.expect(padded->coefficient(2).is_zero(), "c_2 padded to 0");
                      t.expect(padded->coefficient(9).is_zero(), "out-of-range coeff is 0");
                  }
                  auto truncated =
                      BigPowerSeries::from_coeffs({bri(1), bri(2), bri(3), bri(4)}, 2);
                  t.expect(truncated.has_value(), "from_coeffs truncates to order 2");
                  if (truncated) {
                      t.expect_eq(truncated->order(), std::size_t{2}, "order is 2");
                      t.expect_eq(truncated->coefficient(1), bri(2), "kept c_1 = 2");
                      t.expect_eq(truncated->coefficient(2), bri(0), "dropped c_2");
                  }
              })
        .test("basic_factories",
              [&](TestContext& t) {
                  auto x = BigPowerSeries::variable(4);
                  t.expect(x.has_value(), "variable(4) succeeds");
                  if (x) {
                      t.expect(x->coefficient(0).is_zero(), "x has c_0 = 0");
                      t.expect_eq(x->coefficient(1), bri(1), "x has c_1 = 1");
                  }
                  // order 1 truncates x away to 0.
                  auto x1 = BigPowerSeries::variable(1);
                  t.expect(x1.has_value() && x1->coefficient(1).is_zero(),
                           "variable(1) truncates x to 0");
                  auto one = BigPowerSeries::one(3);
                  t.expect(one.has_value(), "one(3) succeeds");
                  if (one) {
                      t.expect_eq(one->coefficient(0), bri(1), "one has c_0 = 1");
                      t.expect(one->coefficient(1).is_zero(), "one has c_1 = 0");
                  }
                  auto z = BigPowerSeries::zero(3);
                  t.expect(z.has_value() && z->coefficient(0).is_zero(), "zero is 0");
              })
        .test("equality_and_operator",
              [&](TestContext& t) {
                  auto a = bseries(t, {1, 2, 3}, 3, "a");
                  auto b = bseries(t, {1, 2, 3}, 3, "b");
                  auto c = bseries(t, {1, 2, 4}, 3, "c");
                  t.expect(a == b, "identical series compare equal");
                  t.expect(!(a == c), "differing series compare unequal");
                  t.expect(a.is_equal(b), "is_equal agrees with operator==");
              })
        .test("multiply_binomial_square",
              [&](TestContext& t) {
                  // (1 + x)(1 + x) = 1 + 2x + x^2.
                  auto a = bseries(t, {1, 1, 0}, 3, "1+x");
                  auto product = a.multiply(a);
                  t.expect(product.has_value(), "(1+x)^2 multiply succeeds");
                  if (product) {
                      t.expect_eq(product->coefficient(0), bri(1), "c_0 = 1");
                      t.expect_eq(product->coefficient(1), bri(2), "c_1 = 2");
                      t.expect_eq(product->coefficient(2), bri(1), "c_2 = 1");
                  }
              })
        .test("multiply_mismatched_order_is_domain_error",
              [&](TestContext& t) {
                  auto a = bseries(t, {1, 1}, 2, "order2");
                  auto b = bseries(t, {1, 1, 1}, 3, "order3");
                  auto product = a.multiply(b);
                  t.expect(!product.has_value() &&
                               product.error() == MathError::domain_error,
                           "mismatched orders -> domain_error");
              })
        .test("add_and_subtract",
              [&](TestContext& t) {
                  auto a = bseries(t, {1, 2, 3}, 3, "a");
                  auto b = bseries(t, {4, 5, 6}, 3, "b");
                  auto sum = a.add(b);
                  t.expect(sum.has_value(), "add succeeds");
                  if (sum) {
                      t.expect_eq(sum->coefficient(0), bri(5), "sum c_0 = 5");
                      t.expect_eq(sum->coefficient(2), bri(9), "sum c_2 = 9");
                  }
                  auto diff = b.subtract(a);
                  t.expect(diff.has_value(), "subtract succeeds");
                  if (diff) {
                      t.expect_eq(diff->coefficient(0), bri(3), "diff c_0 = 3");
                      t.expect_eq(diff->coefficient(2), bri(3), "diff c_2 = 3");
                  }
                  auto bad = a.add(bseries(t, {1}, 2, "wrong"));
                  t.expect(!bad.has_value() && bad.error() == MathError::domain_error,
                           "add mismatched orders -> domain_error");
              })
        .test("scale_multiplies_every_coefficient",
              [&](TestContext& t) {
                  auto a = bseries(t, {1, 2, 3}, 3, "a");
                  auto scaled = a.scale(brat(t, 1, 2, "1/2"));
                  t.expect(scaled.has_value(), "scale succeeds");
                  if (scaled) {
                      t.expect_eq(scaled->coefficient(0), brat(t, 1, 2, "1/2"), "c_0 = 1/2");
                      t.expect_eq(scaled->coefficient(1), bri(1), "c_1 = 1");
                      t.expect_eq(scaled->coefficient(2), brat(t, 3, 2, "3/2"), "c_2 = 3/2");
                  }
              })
        .test("inverse_geometric_series_high_order",
              [&](TestContext& t) {
                  // 1 / (1 - x) = 1 + x + x^2 + ... to order 40 (all ones).
                  constexpr std::size_t n = 40;
                  auto denom = BigPowerSeries::from_coeffs({bri(1), bri(-1)}, n);
                  t.expect(denom.has_value(), "build 1-x at order 40");
                  if (!denom) {
                      return;
                  }
                  auto inv = denom->inverse();
                  t.expect(inv.has_value(), "inverse of 1-x succeeds");
                  if (inv) {
                      for (std::size_t k = 0; k < n; ++k) {
                          t.expect_eq(inv->coefficient(k), bri(1),
                                      std::format("1/(1-x) coeff {} = 1", k));
                      }
                  }
              })
        .test("inverse_zero_constant_term_is_domain_error",
              [&](TestContext& t) {
                  auto s = bseries(t, {0, 1, 0}, 3, "x");
                  auto inv = s.inverse();
                  t.expect(!inv.has_value() && inv.error() == MathError::domain_error,
                           "inverse with c_0 = 0 -> domain_error");
              })
        .test("series_times_its_inverse_is_one",
              [&](TestContext& t) {
                  // s = 2 + 3x + 5x^2 + 7x^3 + 11x^4; s * s^{-1} == 1.
                  constexpr std::size_t n = 5;
                  auto s = bseries(t, {2, 3, 5, 7, 11}, n, "s");
                  auto inv = s.inverse();
                  t.expect(inv.has_value(), "inverse succeeds");
                  if (!inv) {
                      return;
                  }
                  auto prod = s.multiply(*inv);
                  t.expect(prod.has_value(), "s * s^-1 succeeds");
                  if (prod) {
                      t.expect_eq(prod->coefficient(0), bri(1), "product c_0 = 1");
                      for (std::size_t k = 1; k < n; ++k) {
                          t.expect(prod->coefficient(k).is_zero(),
                                   std::format("product c_{} = 0", k));
                      }
                  }
              })
        .test("divide_matches_multiply_by_inverse",
              [&](TestContext& t) {
                  // (1) / (1 - x) = 1 + x + x^2 + ...
                  auto one = bseries(t, {1, 0, 0, 0}, 4, "one");
                  auto denom = bseries(t, {1, -1, 0, 0}, 4, "1-x");
                  auto quotient = one.divide(denom);
                  t.expect(quotient.has_value(), "1/(1-x) divide succeeds");
                  if (quotient) {
                      for (std::size_t k = 0; k < 4; ++k) {
                          t.expect_eq(quotient->coefficient(k), bri(1),
                                      std::format("divide coeff {} = 1", k));
                      }
                  }
              })
        .test("derivative_keeps_order_with_trailing_zero",
              [&](TestContext& t) {
                  // f = 1 + 2x + 3x^2 + 4x^3. f' = 2 + 6x + 12x^2 (+ 0 x^3).
                  auto f = bseries(t, {1, 2, 3, 4}, 4, "f");
                  auto d = f.derivative();
                  t.expect(d.has_value(), "derivative succeeds");
                  if (d) {
                      t.expect_eq(d->order(), std::size_t{4}, "order preserved at 4");
                      t.expect_eq(d->coefficient(0), bri(2), "d_0 = 2");
                      t.expect_eq(d->coefficient(1), bri(6), "d_1 = 6");
                      t.expect_eq(d->coefficient(2), bri(12), "d_2 = 12");
                      t.expect(d->coefficient(3).is_zero(), "d_3 = 0 (trailing zero)");
                  }
              })
        .test("derivative_of_integrate_roundtrips",
              [&](TestContext& t) {
                  // Top coefficient is 0 so nothing is lost to truncation on the way back:
                  // derivative(integrate(s)) == s.
                  auto s = bseries(t, {2, 3, 4, 5, 0}, 5, "s");
                  auto integrated = s.integrate();
                  t.expect(integrated.has_value(), "integrate succeeds");
                  if (!integrated) {
                      return;
                  }
                  t.expect(integrated->coefficient(0).is_zero(), "integral constant term 0");
                  auto back = integrated->derivative();
                  t.expect(back.has_value(), "derivative of integral succeeds");
                  if (back) {
                      for (std::size_t k = 0; k < 5; ++k) {
                          t.expect_eq(back->coefficient(k), s.coefficient(k),
                                      std::format("roundtrip coeff {}", k));
                      }
                  }
              })
        .test("exp_of_x_is_reciprocal_factorials_unbounded",
              [&](TestContext& t) {
                  // exp(x) to order 30: coefficient(n) == 1/n! EXACTLY, including n where n!
                  // overflows int64 (n >= 21) and the int64 powerseries cannot represent it.
                  constexpr std::size_t n = 30;
                  auto x = BigPowerSeries::variable(n);
                  t.expect(x.has_value(), "variable(30) succeeds");
                  if (!x) {
                      return;
                  }
                  auto e = x->exp();
                  t.expect(e.has_value(), "exp(x) succeeds");
                  if (!e) {
                      return;
                  }
                  // Low order, still int64-representable.
                  t.expect_eq(e->coefficient(0), bri(1), "e_0 = 1 = 1/0!");
                  t.expect_eq(e->coefficient(1), bri(1), "e_1 = 1 = 1/1!");
                  t.expect_eq(e->coefficient(2), brat(t, 1, 2, "1/2"), "e_2 = 1/2!");
                  t.expect_eq(e->coefficient(5), inv_factorial(t, 5, "1/5!"), "e_5 = 1/5!");
                  // The unbounded range: these denominators exceed int64.
                  t.expect_eq(e->coefficient(21), inv_factorial(t, 21, "1/21!"),
                              "e_21 = 1/21! (denominator exceeds int64)");
                  t.expect_eq(e->coefficient(25), inv_factorial(t, 25, "1/25!"),
                              "e_25 = 1/25! (denominator exceeds int64)");
                  t.expect_eq(e->coefficient(29), inv_factorial(t, 29, "1/29!"),
                              "e_29 = 1/29! (denominator exceeds int64)");
                  // Pin the exact 1/25! denominator string so the unbounded value is nailed
                  // down independent of the factorial helper: 25! = 15511210043330985984000000.
                  t.expect_eq(e->coefficient(25).denominator().to_string(),
                              std::string("15511210043330985984000000"),
                              "1/25! denominator is exactly 25!");
                  t.expect_eq(e->coefficient(25).numerator().to_string(), std::string("1"),
                              "1/25! numerator is 1");
              })
        .test("exp_nonzero_constant_term_is_domain_error",
              [&](TestContext& t) {
                  auto s = bseries(t, {1, 1, 0}, 3, "1+x");
                  auto e = s.exp();
                  t.expect(!e.has_value() && e.error() == MathError::domain_error,
                           "exp with c_0 != 0 -> domain_error");
              })
        .test("log_of_one_plus_x_alternating",
              [&](TestContext& t) {
                  // log(1 + x): coefficient(n) == (-1)^{n+1}/n for n >= 1, to order 15.
                  constexpr std::size_t n = 15;
                  auto s = BigPowerSeries::from_coeffs({bri(1), bri(1)}, n);
                  t.expect(s.has_value(), "build 1+x at order 15");
                  if (!s) {
                      return;
                  }
                  auto l = s->log();
                  t.expect(l.has_value(), "log(1+x) succeeds");
                  if (!l) {
                      return;
                  }
                  t.expect(l->coefficient(0).is_zero(), "l_0 = 0");
                  for (std::size_t k = 1; k < n; ++k) {
                      const std::int64_t sign = (k % 2 == 1) ? 1 : -1;
                      auto expected = brat(t, sign, static_cast<std::int64_t>(k),
                                           std::format("(-1)^(k+1)/{}", k));
                      t.expect_eq(l->coefficient(k), expected,
                                  std::format("log(1+x) coeff {} = (-1)^(k+1)/{}", k, k));
                  }
              })
        .test("log_wrong_constant_term_is_domain_error",
              [&](TestContext& t) {
                  auto s = bseries(t, {2, 1, 0}, 3, "2+x");
                  auto l = s.log();
                  t.expect(!l.has_value() && l.error() == MathError::domain_error,
                           "log with c_0 != 1 -> domain_error");
              })
        .test("exp_log_identity",
              [&](TestContext& t) {
                  // exp(log(1 + x)) == 1 + x (truncated) at a high order.
                  constexpr std::size_t n = 20;
                  auto s = BigPowerSeries::from_coeffs({bri(1), bri(1)}, n);
                  t.expect(s.has_value(), "build 1+x at order 20");
                  if (!s) {
                      return;
                  }
                  auto l = s->log();
                  t.expect(l.has_value(), "log(1+x) succeeds");
                  if (!l) {
                      return;
                  }
                  auto e = l->exp();  // log(1+x) has zero constant term
                  t.expect(e.has_value(), "exp(log(1+x)) succeeds");
                  if (e) {
                      t.expect(e->is_equal(*s),
                               std::format("exp(log(1+x)) = {}", e->to_string()));
                  }
              })
        .test("compose_geometric_of_x_squared",
              [&](TestContext& t) {
                  // (1/(1-x)) ∘ x^2 == 1/(1-x^2): even coeffs 1, odd coeffs 0.
                  constexpr std::size_t n = 12;
                  auto denom = BigPowerSeries::from_coeffs({bri(1), bri(-1)}, n);
                  t.expect(denom.has_value(), "build 1-x");
                  if (!denom) {
                      return;
                  }
                  auto geo = denom->inverse();  // 1/(1-x) = sum x^k
                  t.expect(geo.has_value(), "1/(1-x) succeeds");
                  if (!geo) {
                      return;
                  }
                  // g = x^2 (g_0 = 0).
                  std::vector<BigRational> gc(n);
                  gc[2] = bri(1);
                  auto g = BigPowerSeries::from_coeffs(std::move(gc), n);
                  t.expect(g.has_value(), "build x^2");
                  if (!g) {
                      return;
                  }
                  auto comp = geo->compose(*g);
                  t.expect(comp.has_value(), "compose succeeds");
                  if (comp) {
                      for (std::size_t k = 0; k < n; ++k) {
                          auto expected = (k % 2 == 0) ? bri(1) : bri(0);
                          t.expect_eq(comp->coefficient(k), expected,
                                      std::format("1/(1-x^2) coeff {}", k));
                      }
                  }
              })
        .test("compose_evaluates_this_of_g",
              [&](TestContext& t) {
                  // f(y) = 1 + y + y^2, g = x + x^2 (g_0 = 0).
                  // f(g) = 1 + x + 2x^2 + 2x^3 + ...  (truncated at order 4: 1, 1, 2, 2)
                  auto f = bseries(t, {1, 1, 1, 0}, 4, "f");
                  auto g = bseries(t, {0, 1, 1, 0}, 4, "g");
                  auto comp = f.compose(g);
                  t.expect(comp.has_value(), "compose succeeds");
                  if (comp) {
                      t.expect_eq(comp->coefficient(0), bri(1), "c_0 = 1");
                      t.expect_eq(comp->coefficient(1), bri(1), "c_1 = 1");
                      t.expect_eq(comp->coefficient(2), bri(2), "c_2 = 2");
                      t.expect_eq(comp->coefficient(3), bri(2), "c_3 = 2");
                  }
              })
        .test("compose_nonzero_inner_constant_is_domain_error",
              [&](TestContext& t) {
                  auto f = bseries(t, {1, 1, 1}, 3, "f");
                  auto g = bseries(t, {1, 1, 0}, 3, "g_with_const");
                  auto comp = f.compose(g);
                  t.expect(!comp.has_value() && comp.error() == MathError::domain_error,
                           "compose with g_0 != 0 -> domain_error");
              })
        .test("compose_mismatched_order_is_domain_error",
              [&](TestContext& t) {
                  auto f = bseries(t, {1, 1}, 2, "f");
                  auto g = bseries(t, {0, 1, 0}, 3, "g");
                  auto comp = f.compose(g);
                  t.expect(!comp.has_value() && comp.error() == MathError::domain_error,
                           "compose mismatched orders -> domain_error");
              })
        .test("cross_check_low_order_matches_int64_powerseries",
              [&](TestContext& t) {
                  // For orders where int64 does not overflow, the BigPowerSeries coefficients
                  // must equal the int64 PowerSeries ones exactly. Compare the BigRational
                  // numerator/denominator (as BigInt) against the int64 Rational lifted via
                  // BigInt::from_i64. exp(x) at order 8 keeps every denominator (7! = 5040)
                  // well within int64.
                  constexpr std::size_t n = 8;

                  auto small_x = PowerSeries::variable(n);
                  auto big_x = BigPowerSeries::variable(n);
                  t.expect(small_x.has_value() && big_x.has_value(), "both variable(8)");
                  if (!small_x || !big_x) {
                      return;
                  }
                  auto small_e = small_x->exp();
                  auto big_e = big_x->exp();
                  t.expect(small_e.has_value() && big_e.has_value(), "both exp(x) succeed");
                  if (!small_e || !big_e) {
                      return;
                  }
                  for (std::size_t k = 0; k < n; ++k) {
                      const Rational rc = small_e->coefficient(k);
                      const BigRational bc = big_e->coefficient(k);
                      const bool num_ok =
                          bc.numerator() == BigInt::from_i64(rc.numerator());
                      const bool den_ok =
                          bc.denominator() == BigInt::from_i64(rc.denominator());
                      t.expect(num_ok && den_ok,
                               std::format("exp coeff {}: big {} == int64 {}", k,
                                           bc.to_string(), rc.to_string()));
                  }

                  // Also cross-check a division result: 1/(1-x) all-ones in both tiers.
                  auto small_d = PowerSeries::from_coeffs(
                      {Rational::from_int(1), Rational::from_int(-1)}, n);
                  auto big_d = BigPowerSeries::from_coeffs({bri(1), bri(-1)}, n);
                  t.expect(small_d.has_value() && big_d.has_value(), "both build 1-x");
                  if (!small_d || !big_d) {
                      return;
                  }
                  auto small_inv = small_d->inverse();
                  auto big_inv = big_d->inverse();
                  t.expect(small_inv.has_value() && big_inv.has_value(),
                           "both inverse succeed");
                  if (!small_inv || !big_inv) {
                      return;
                  }
                  for (std::size_t k = 0; k < n; ++k) {
                      const Rational rc = small_inv->coefficient(k);
                      const BigRational bc = big_inv->coefficient(k);
                      t.expect(bc.numerator() == BigInt::from_i64(rc.numerator()) &&
                                   bc.denominator() == BigInt::from_i64(rc.denominator()),
                               std::format("1/(1-x) coeff {} agrees across tiers", k));
                  }
              })
        .run();
}
