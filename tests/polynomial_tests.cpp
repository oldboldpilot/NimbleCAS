// Tests for nimblecas.polynomial: exact univariate polynomial arithmetic + SIMD eval.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.testing;

using nimblecas::MathError;
using nimblecas::Polynomial;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {
auto poly(std::vector<std::int64_t> c) -> Polynomial { return Polynomial{std::move(c)}; }
}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.polynomial")
        .test("construction_trims_and_reports_shape",
              [](TestContext& t) {
                  auto p = poly({1, 2, 3, 0, 0});  // 1 + 2x + 3x^2
                  t.expect_eq(p.degree(), std::int64_t{2}, "trailing zeros trimmed");
                  t.expect_eq(p.leading_coefficient(), std::int64_t{3}, "leading coeff");
                  t.expect_eq(p.coefficient(1), std::int64_t{2}, "coeff of x");
                  t.expect_eq(p.coefficient(9), std::int64_t{0}, "coeff beyond degree is 0");
                  t.expect(Polynomial{}.is_zero(), "default is the zero polynomial");
                  t.expect_eq(Polynomial{}.degree(), std::int64_t{-1}, "deg(0) = -1");
              })
        .test("add_subtract",
              [](TestContext& t) {
                  auto a = poly({1, 2, 3});  // 1 + 2x + 3x^2
                  auto b = poly({0, 1, -3, 4});  // x - 3x^2 + 4x^3
                  auto sum = a.add(b);
                  t.expect(sum.has_value() && sum->is_equal(poly({1, 3, 0, 4})),
                           "sum combines like powers, cancels x^2");
                  auto diff = a.subtract(a);
                  t.expect(diff.has_value() && diff->is_zero(), "p - p = 0");
              })
        .test("scale_and_multiply",
              [](TestContext& t) {
                  auto a = poly({1, 1});   // x + 1
                  auto b = poly({-1, 1});  // x - 1
                  auto prod = a.multiply(b);
                  t.expect(prod.has_value() && prod->is_equal(poly({-1, 0, 1})),
                           "(x+1)(x-1) = x^2 - 1");
                  auto scaled = a.scale(3);
                  t.expect(scaled.has_value() && scaled->is_equal(poly({3, 3})), "3*(x+1)");
                  t.expect(a.scale(0).value().is_zero(), "scale by 0 -> zero polynomial");
              })
        .test("evaluate_exact",
              [](TestContext& t) {
                  auto p = poly({1, 2, 3});  // 1 + 2x + 3x^2
                  auto at2 = p.evaluate(2);  // 1 + 4 + 12 = 17
                  t.expect(at2.has_value() && at2.value() == 17, "P(2) = 17");
                  t.expect(poly({}).evaluate(5).value() == 0, "zero poly evaluates to 0");
              })
        .test("multiply_overflow_reported",
              [](TestContext& t) {
                  auto big = Polynomial::constant(std::numeric_limits<std::int64_t>::max());
                  auto r = big.multiply(poly({0, 2}));  // max * 2x -> overflow
                  t.expect(!r.has_value() && r.error() == MathError::overflow,
                           "int64 overflow in multiply is reported");
              })
        .test("evaluate_batch_matches_scalar_reference",
              [](TestContext& t) {
                  auto p = poly({2, -3, 1});  // 2 - 3x + x^2
                  std::vector<float> xs(1000);
                  for (std::size_t i = 0; i < xs.size(); ++i) {
                      xs[i] = static_cast<float>(i) * 0.01f - 5.0f;
                  }
                  auto got = p.evaluate_batch(xs);
                  bool ok = got.size() == xs.size();
                  for (std::size_t i = 0; ok && i < xs.size(); ++i) {
                      // scalar Horner reference with the same (fused) rounding
                      float ref = 1.0f;                       // leading coeff
                      ref = std::fma(ref, xs[i], -3.0f);      // * x + (-3)
                      ref = std::fma(ref, xs[i], 2.0f);       // * x + 2
                      ok = std::bit_cast<std::uint32_t>(got[i]) == std::bit_cast<std::uint32_t>(ref);
                  }
                  t.expect(ok, "SIMD batch Horner is bit-identical to the scalar reference");
              })
        .run();
}
