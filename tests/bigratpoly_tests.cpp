// Tests for nimblecas.bigratpoly: dense univariate polynomials over BigRational (Q[x]).
// @author Olumuyiwa Oluwasanmi
//
// Oracles are hand-verified over Q. The load-bearing case is a multiply whose intermediate
// coefficients OVERFLOW int64 — two polynomials with coefficients 10^10 produce product
// coefficients of 10^20 and 2*10^20, both far beyond INT64_MAX (~9.22*10^18). The int64
// RationalPoly tier would report MathError::overflow there; BigRationalPoly returns the
// exact bignum result, which is the whole reason this module exists.

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.ratpoly;
import nimblecas.bigratpoly;
import nimblecas.testing;

using nimblecas::BigRational;
using nimblecas::BigRationalPoly;
using nimblecas::MathError;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// A BigRational from a decimal/fraction literal (all literals below are valid by
// construction, so the dereference never faults).
auto big(std::string_view s) -> BigRational {
    auto r = BigRational::from_string(s);
    return *r;
}

auto bi(std::int64_t v) -> BigRational {
    return BigRational::from_int(v);
}

// A BigRationalPoly from low-degree-first coefficients.
auto poly(std::vector<BigRational> c) -> BigRationalPoly {
    return BigRationalPoly::from_coeffs(std::move(c));
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.bigratpoly")
        .test("from_coeffs trims and reports degree",
              [](TestContext& t) {
                  const BigRationalPoly z = BigRationalPoly::zero();
                  t.expect(z.is_zero() && z.degree() == -1, "zero polynomial has degree -1");
                  // Trailing zero coefficient is dropped: [1, 2, 0] -> degree 1.
                  const BigRationalPoly p = poly({bi(1), bi(2), bi(0)});
                  t.expect(p.degree() == 1, "trailing zero trimmed to degree 1");
                  t.expect(p.leading_coefficient() == bi(2), "leading coefficient is 2");
                  t.expect(p.coefficient(5) == bi(0), "out-of-range coefficient is 0");
              })
        .test("(x - 1)(x + 1) == x^2 - 1",
              [](TestContext& t) {
                  const BigRationalPoly xm1 = poly({bi(-1), bi(1)});  // x - 1
                  const BigRationalPoly xp1 = poly({bi(1), bi(1)});   // x + 1
                  const BigRationalPoly prod = xm1.multiply(xp1);
                  const BigRationalPoly target = poly({bi(-1), bi(0), bi(1)});  // x^2 - 1
                  t.expect(prod.is_equal(target), "product equals x^2 - 1");
              })
        .test("multiply past the int64 ceiling stays exact (bignum advantage)",
              [](TestContext& t) {
                  // (1e10 + 1e10 x) * (1e10 + 1e10 x)
                  //   = 1e20 + 2e20 x + 1e20 x^2, coefficients well beyond INT64_MAX.
                  const BigRationalPoly p = poly({big("10000000000"), big("10000000000")});
                  const BigRationalPoly prod = p.multiply(p);
                  const BigRationalPoly target = poly({big("100000000000000000000"),
                                                       big("200000000000000000000"),
                                                       big("100000000000000000000")});
                  t.expect(prod.is_equal(target), "1e20 + 2e20 x + 1e20 x^2 computed exactly");
                  t.expect(prod.coefficient(1) == big("200000000000000000000"),
                           "middle coefficient is exactly 2e20");
                  // Proof it truly exceeds the int64 tier: narrowing back must overflow.
                  auto narrow = prod.to_ratpoly();
                  t.expect(!narrow.has_value() && narrow.error() == MathError::overflow,
                           "to_ratpoly on 2e20 coefficient -> overflow");
              })
        .test("add and subtract combine coefficients",
              [](TestContext& t) {
                  const BigRationalPoly a = poly({bi(1), bi(2), bi(3)});
                  const BigRationalPoly b = poly({bi(5), bi(5)});
                  t.expect(a.add(b).is_equal(poly({bi(6), bi(7), bi(3)})), "add is coefficientwise");
                  // a - a collapses to the zero polynomial (full cancellation, trimmed).
                  t.expect(a.subtract(a).is_zero(), "a - a is the zero polynomial");
              })
        .test("divide: x^2 / (x - 1) == (x + 1) remainder 1",
              [](TestContext& t) {
                  const BigRationalPoly x2 = poly({bi(0), bi(0), bi(1)});  // x^2
                  const BigRationalPoly xm1 = poly({bi(-1), bi(1)});       // x - 1
                  auto dm = x2.divide(xm1);
                  t.expect(dm.has_value(), "divide succeeded");
                  if (!dm) return;
                  t.expect(dm->quotient.is_equal(poly({bi(1), bi(1)})), "quotient is x + 1");
                  t.expect(dm->remainder.is_equal(poly({bi(1)})), "remainder is 1");
              })
        .test("divide is exact for (x^2 - 1) / (x - 1)",
              [](TestContext& t) {
                  const BigRationalPoly x2m1 = poly({bi(-1), bi(0), bi(1)});  // x^2 - 1
                  const BigRationalPoly xm1 = poly({bi(-1), bi(1)});          // x - 1
                  auto dm = x2m1.divide(xm1);
                  t.expect(dm.has_value(), "divide succeeded");
                  if (!dm) return;
                  t.expect(dm->remainder.is_zero(), "remainder is zero");
                  t.expect(dm->quotient.is_equal(poly({bi(1), bi(1)})), "quotient is x + 1");
              })
        .test("gcd(x^2 - 1, 2x - 2) == x - 1 (monic)",
              [](TestContext& t) {
                  const BigRationalPoly x2m1 = poly({bi(-1), bi(0), bi(1)});  // x^2 - 1
                  const BigRationalPoly tx2 = poly({bi(-2), bi(2)});          // 2x - 2
                  auto g = x2m1.gcd(tx2);
                  t.expect(g.has_value(), "gcd succeeded");
                  if (!g) return;
                  // The Euclidean remainder chain ends at 2x - 2; the monic normalisation
                  // (a rational 1/2 scale) brings it to the canonical x - 1.
                  t.expect(g->is_equal(poly({bi(-1), bi(1)})), "monic gcd is x - 1");
              })
        .test("monic normalises 3(x - 1) to x - 1 over Q",
              [](TestContext& t) {
                  const BigRationalPoly scaled = poly({bi(-3), bi(3)});  // 3x - 3
                  auto m = scaled.monic();
                  t.expect(m.has_value(), "monic succeeded");
                  if (!m) return;
                  t.expect(m->is_equal(poly({bi(-1), bi(1)})), "monic(3x - 3) == x - 1");
              })
        .test("derivative of x^3 is 3x^2",
              [](TestContext& t) {
                  const BigRationalPoly x3 = poly({bi(0), bi(0), bi(0), bi(1)});  // x^3
                  const BigRationalPoly d = x3.derivative();
                  t.expect(d.degree() == 2, "derivative has degree 2");
                  t.expect(d.is_equal(poly({bi(0), bi(0), bi(3)})), "derivative is 3x^2");
              })
        .test("evaluate uses exact rational Horner",
              [](TestContext& t) {
                  const BigRationalPoly x2m1 = poly({bi(-1), bi(0), bi(1)});  // x^2 - 1
                  auto at3 = x2m1.evaluate(bi(3));  // 9 - 1 = 8
                  t.expect(at3.has_value() && *at3 == bi(8), "(x^2 - 1)(3) == 8");
                  auto athalf = x2m1.evaluate(big("1/2"));  // 1/4 - 1 = -3/4
                  t.expect(athalf.has_value() && *athalf == big("-3/4"),
                           "(x^2 - 1)(1/2) == -3/4");
              })
        .test("divide by the zero polynomial is an honest error",
              [](TestContext& t) {
                  const BigRationalPoly x2m1 = poly({bi(-1), bi(0), bi(1)});
                  auto dz = x2m1.divide(BigRationalPoly::zero());
                  t.expect(!dz.has_value() && dz.error() == MathError::division_by_zero,
                           "divide by zero -> division_by_zero");
              })
        .test("from_ratpoly / to_ratpoly bridge round-trips the int64 tier",
              [](TestContext& t) {
                  // x^2 - 1 as an int64 RationalPoly, widened and narrowed back.
                  const RationalPoly rp = RationalPoly::from_coeffs(
                      {Rational::from_int(-1), Rational::from_int(0), Rational::from_int(1)});
                  const BigRationalPoly wide = BigRationalPoly::from_ratpoly(rp);
                  t.expect(wide.is_equal(poly({bi(-1), bi(0), bi(1)})), "from_ratpoly lifts x^2 - 1");
                  auto back = wide.to_ratpoly();
                  t.expect(back.has_value(), "to_ratpoly succeeded");
                  if (!back) return;
                  t.expect(back->is_equal(rp), "round-trip recovers the int64 polynomial");
              })
        .run();
}
