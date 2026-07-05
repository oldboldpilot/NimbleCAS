// Tests for nimblecas.laurent: truncated Laurent series over Q -- valuation shifts,
// principal/regular split, series inversion (negative powers), residue, and exact
// Laurent expansion of a rational function about a pole.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.powerseries;
import nimblecas.laurent;
import nimblecas.testing;

using nimblecas::Laurent;
using nimblecas::MathError;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

[[nodiscard]] auto ri(std::int64_t v) -> Rational {
    return Rational::from_int(v);
}

// vector<Rational> from integer values (each over denominator 1).
[[nodiscard]] auto ints(std::vector<std::int64_t> vs) -> std::vector<Rational> {
    std::vector<Rational> out;
    out.reserve(vs.size());
    for (const std::int64_t v : vs) {
        out.push_back(Rational::from_int(v));
    }
    return out;
}

// Laurent from a low exponent and integer coefficients.
[[nodiscard]] auto lau(std::int64_t order_min, std::vector<std::int64_t> vs) -> Laurent {
    return Laurent::from_coeffs(order_min, ints(std::move(vs))).value();
}

// RationalPoly from integer coefficients (c[i] * x^i).
[[nodiscard]] auto poly(std::vector<std::int64_t> vs) -> RationalPoly {
    return RationalPoly::from_coeffs(ints(std::move(vs)));
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.laurent")
        .test("reciprocal_of_x_minus_x_squared_has_valuation_minus_one",
              [](TestContext& t) {
                  // s = x - x^2, tracked to O(x^6). 1/s = x^{-1} + 1 + x + x^2 + x^3 + ...
                  // (from 1/(x(1-x)) = x^{-1} * 1/(1-x)). Valuation -1, residue 1.
                  const auto s = lau(1, {1, -1, 0, 0, 0});
                  const auto inv = s.inverse().value();
                  t.expect(inv.valuation().value() == -1, "valuation(1/(x-x^2)) == -1");
                  t.expect(inv.order_min() == -1, "principal part starts at x^{-1}");
                  t.expect(inv.residue().value() == ri(1), "residue == 1 (coeff of x^{-1})");
                  t.expect(inv.coefficient(-1) == ri(1), "c_{-1} == 1");
                  t.expect(inv.coefficient(0) == ri(1), "c_0 == 1");
                  t.expect(inv.coefficient(1) == ri(1), "c_1 == 1");
                  t.expect(inv.coefficient(-2) == ri(0), "c_{-2} == 0 (genuine zero below valuation)");
              })
        .test("from_rational_function_matches_direct_inverse",
              [](TestContext& t) {
                  // Same series via from_rational_function: 1/(x - x^2) about 0, order 5.
                  const auto lr = Laurent::from_rational_function(poly({1}), poly({0, 1, -1}),
                                                                 ri(0), 5)
                                      .value();
                  t.expect(lr.order_min() == -1, "expansion valuation == -1");
                  t.expect(lr.truncation_order() == 4, "tracks c_{-1}..c_3 (O(x^4))");
                  t.expect(lr.residue().value() == ri(1), "residue == 1");
                  t.expect(lr.coefficient(3) == ri(1), "c_3 == 1");
              })
        .test("multiply_adds_valuations",
              [](TestContext& t) {
                  // a = 2 x^{-2} + 3, valuation -2; b = 5 x + 7 x^2, valuation 1.
                  const auto a = lau(-2, {2, 0, 3});
                  const auto b = lau(1, {5, 7});
                  t.expect(a.valuation().value() == -2, "valuation(a) == -2");
                  t.expect(b.valuation().value() == 1, "valuation(b) == 1");
                  const auto p = a.multiply(b).value();
                  // valuation adds: -2 + 1 = -1. len = min(3,2) = 2.
                  t.expect(p.valuation().value() == -1, "valuation(a*b) == -1 (valuations add)");
                  t.expect(p.order_min() == -1, "product order_min == -1");
                  t.expect(p.size() == 2, "product keeps min(3,2) = 2 coefficients");
                  t.expect(p.coefficient(-1) == ri(10), "c_{-1} == 2*5 == 10");
                  t.expect(p.coefficient(0) == ri(14), "c_0 == 2*7 == 14");
              })
        .test("principal_and_regular_part_split",
              [](TestContext& t) {
                  // L = x^{-2} + 2 x^{-1} + 3 + 4x + 5x^2, tracked to O(x^3).
                  const auto L = lau(-2, {1, 2, 3, 4, 5});
                  const auto pp = L.principal_part().value();
                  const auto rp = L.regular_part().value();
                  // Principal part: the finite negative-power terms.
                  t.expect(pp.order_min() == -2, "principal part starts at x^{-2}");
                  t.expect(pp.coefficient(-2) == ri(1), "principal c_{-2} == 1");
                  t.expect(pp.coefficient(-1) == ri(2), "principal c_{-1} == 2");
                  t.expect(pp.coefficient(0) == ri(0), "principal has no x^0 term");
                  // Regular part: the exponent >= 0 tail, keeping the O(x^3) truncation.
                  t.expect(rp.order_min() == 0, "regular part starts at x^0");
                  t.expect(rp.truncation_order() == 3, "regular part keeps O(x^3)");
                  t.expect(rp.coefficient(0) == ri(3), "regular c_0 == 3");
                  t.expect(rp.coefficient(1) == ri(4), "regular c_1 == 4");
                  t.expect(rp.coefficient(2) == ri(5), "regular c_2 == 5");
                  t.expect(rp.coefficient(-1) == ri(0), "regular has no negative term");
              })
        .test("expand_rational_function_about_a_pole",
              [](TestContext& t) {
                  // 1/(x^2 - 1) about x = 1 (a simple pole). In powers of (x-1):
                  //   x^2 - 1 = (x-1)(x+1) = (x-1)(2 + (x-1)) so
                  //   1/(x^2-1) = (1/2)(x-1)^{-1} - 1/4 + (1/8)(x-1) - (1/16)(x-1)^2 + ...
                  // Residue at the simple pole x=1 is 1/2.
                  const auto lr = Laurent::from_rational_function(poly({1}), poly({-1, 0, 1}),
                                                                 ri(1), 4)
                                      .value();
                  t.expect(lr.order_min() == -1, "simple pole => valuation -1");
                  t.expect(lr.residue().value() == rat(1, 2), "residue at x=1 is 1/2");
                  t.expect(lr.coefficient(-1) == rat(1, 2), "c_{-1} == 1/2");
                  t.expect(lr.coefficient(0) == rat(-1, 4), "c_0 == -1/4");
                  t.expect(lr.coefficient(1) == rat(1, 8), "c_1 == 1/8");
                  t.expect(lr.coefficient(2) == rat(-1, 16), "c_2 == -1/16");
              })
        .test("reciprocal_round_trip_is_one_to_tracked_order",
              [](TestContext& t) {
                  // s = 3x^2 + x^3 + 4x^4 + x^5 (valuation 2, 4 tracked coeffs).
                  // s * (1/s) must be 1 + O(x^4) exactly (product order_min 2 + (-2) = 0).
                  const auto s = lau(2, {3, 1, 4, 1});
                  const auto inv = s.inverse().value();
                  t.expect(inv.valuation().value() == -2, "1/s has valuation -2");
                  const auto prod = s.multiply(inv).value();
                  t.expect(prod.order_min() == 0, "s*(1/s) starts at x^0");
                  t.expect(prod.coefficient(0) == ri(1), "s*(1/s) c_0 == 1");
                  t.expect(prod.coefficient(1) == ri(0), "s*(1/s) c_1 == 0");
                  t.expect(prod.coefficient(2) == ri(0), "s*(1/s) c_2 == 0");
                  t.expect(prod.coefficient(3) == ri(0), "s*(1/s) c_3 == 0");
              })
        .test("add_uses_the_smaller_truncation_order",
              [](TestContext& t) {
                  // a = x^{-1} + 1 + x + x^2 (O(x^3)); b = 1 + x (O(x^2)).
                  // Sum tracked only to O(x^2) (the smaller), from x^{-1}.
                  const auto a = lau(-1, {1, 1, 1, 1});
                  const auto b = lau(0, {1, 1});
                  const auto s = a.add(b).value();
                  t.expect(s.order_min() == -1, "sum order_min == min(-1, 0) == -1");
                  t.expect(s.truncation_order() == 2, "sum truncation == min(3, 2) == 2");
                  t.expect(s.coefficient(-1) == ri(1), "c_{-1} == 1");
                  t.expect(s.coefficient(0) == ri(2), "c_0 == 1 + 1 == 2");
                  t.expect(s.coefficient(1) == ri(2), "c_1 == 1 + 1 == 2");
              })
        .test("honesty_errors",
              [](TestContext& t) {
                  // Inverse of a series with no nonzero tracked term is a domain_error.
                  const auto z = Laurent::zero(0, 3).value();
                  t.expect(z.inverse().error() == MathError::domain_error,
                           "inverse of all-zero tracked part => domain_error");
                  t.expect(z.valuation().error() == MathError::domain_error,
                           "valuation of all-zero tracked part => domain_error");
                  // residue is unknown when x^{-1} is at/beyond the truncation order.
                  // s = x^{-5} + x^{-4} tracked to O(x^{-3}); c_{-1} is not tracked.
                  const auto s = lau(-5, {1, 1});
                  t.expect(s.residue().error() == MathError::domain_error,
                           "residue beyond truncation order => domain_error");
                  // ... but a residue below a positive valuation is a genuine zero.
                  const auto reg = lau(0, {1, 2, 3});
                  t.expect(reg.residue().value() == ri(0),
                           "residue of a regular (Taylor) series is a genuine 0");
                  // Empty coefficient list and a zero denominator are rejected honestly.
                  t.expect(Laurent::from_coeffs(0, {}).error() == MathError::domain_error,
                           "empty coefficient list => domain_error");
                  t.expect(Laurent::from_rational_function(poly({1}), poly({}), ri(0), 3).error() ==
                               MathError::division_by_zero,
                           "zero denominator => division_by_zero");
                  t.expect(Laurent::from_rational_function(poly({1}), poly({1}), ri(0), 0).error() ==
                               MathError::domain_error,
                           "order 0 => domain_error");
              })
        .run();
}
