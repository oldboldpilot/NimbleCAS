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
        .test("content_and_primitive_part",
              [](TestContext& t) {
                  t.expect(poly({2, 4, 6}).content().value() == 2, "content(2,4,6) = 2");
                  t.expect(poly({3, 0, 9}).content().value() == 3, "content(3,0,9) = 3");
                  t.expect(poly({}).content().value() == 0, "content(0) = 0");
                  // 2x + 4 -> x + 2 (primitive)
                  t.expect(poly({4, 2}).primitive_part().value().is_equal(poly({2, 1})),
                           "primitive_part(4 + 2x) = 2 + x");
                  // -2x - 2 -> x + 1 (sign-normalised to positive leading)
                  t.expect(poly({-2, -2}).primitive_part().value().is_equal(poly({1, 1})),
                           "primitive_part(-2 - 2x) = 1 + x");
              })
        .test("pseudo_remainder",
              [](TestContext& t) {
                  // x^2 - 1 = (x - 1)(x + 1): x - 1 divides it, prem = 0
                  auto r = poly({-1, 0, 1}).pseudo_remainder(poly({-1, 1}));
                  t.expect(r.has_value() && r->is_zero(), "prem(x^2-1, x-1) = 0");
                  auto bad = poly({1, 1}).pseudo_remainder(poly({}));
                  t.expect(!bad.has_value() && bad.error() == MathError::division_by_zero,
                           "zero divisor rejected");
              })
        .test("gcd_over_Z",
              [](TestContext& t) {
                  auto g1 = poly({-1, 0, 1}).gcd(poly({-1, 1}));  // gcd(x^2-1, x-1)
                  t.expect(g1.has_value() && g1->is_equal(poly({-1, 1})), "gcd(x^2-1, x-1) = x-1");
                  auto g2 = poly({1, 2, 1}).gcd(poly({1, 1}));  // gcd((x+1)^2, x+1)
                  t.expect(g2.has_value() && g2->is_equal(poly({1, 1})), "gcd((x+1)^2, x+1) = x+1");
                  auto g3 = poly({-1, 0, 1}).gcd(poly({1, -2, 1}));  // gcd(x^2-1, (x-1)^2)
                  t.expect(g3.has_value() && g3->is_equal(poly({-1, 1})), "gcd(x^2-1, (x-1)^2) = x-1");
                  auto g4 = poly({-2, 0, 2}).gcd(poly({-2, 2}));  // gcd(2x^2-2, 2x-2)
                  t.expect(g4.has_value() && g4->is_equal(poly({-2, 2})),
                           "gcd(2x^2-2, 2x-2) = 2x-2 (content 2)");
              })
        .test("derivative_and_exact_division",
              [](TestContext& t) {
                  // d/dx (1 + 2x + x^3) = 2 + 3x^2
                  t.expect(poly({1, 2, 0, 1}).derivative().value().is_equal(poly({2, 0, 3})),
                           "derivative of 1 + 2x + x^3");
                  // (x^2 - 1) / (x - 1) = x + 1
                  auto q = poly({-1, 0, 1}).divide_exact(poly({-1, 1}));
                  t.expect(q.has_value() && q->is_equal(poly({1, 1})), "(x^2-1)/(x-1) = x+1");
                  // not exact over Z
                  t.expect(poly({0, 0, 1}).divide_exact(poly({0, 2})).error() ==
                               MathError::domain_error,
                           "x^2 / 2x is not exact over Z");
                  // nonzero remainder
                  t.expect(poly({1, 0, 1}).divide_exact(poly({-1, 1})).error() ==
                               MathError::domain_error,
                           "(x^2+1)/(x-1) has a remainder");
              })
        .test("square_free_factorization",
              [](TestContext& t) {
                  auto has = [](const auto& fs, const Polynomial& p, std::int64_t m) {
                      return std::ranges::any_of(fs, [&](const auto& f) {
                          return f.second == m && f.first.is_equal(p);
                      });
                  };
                  // (x+1)^2 -> one factor (x+1) with multiplicity 2
                  auto f1 = poly({1, 2, 1}).square_free_factorization();
                  t.expect(f1.has_value() && f1->size() == 1 && has(*f1, poly({1, 1}), 2),
                           "(x+1)^2 -> (x+1)^2");
                  // x^2 - 1 is square-free -> itself with multiplicity 1
                  auto f2 = poly({-1, 0, 1}).square_free_factorization();
                  t.expect(f2.has_value() && f2->size() == 1 && has(*f2, poly({-1, 0, 1}), 1),
                           "x^2-1 is square-free");
                  // (x-1)^2 (x+1) -> (x+1)^1 and (x-1)^2
                  auto f3 = poly({1, -1, -1, 1}).square_free_factorization();
                  t.expect(f3.has_value() && f3->size() == 2 &&
                               has(*f3, poly({1, 1}), 1) && has(*f3, poly({-1, 1}), 2),
                           "(x-1)^2 (x+1) factors by multiplicity");
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
