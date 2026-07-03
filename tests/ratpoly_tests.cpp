// Tests for nimblecas.ratpoly: exact rationals and Q[x] (division, gcd).
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.ratpoly;
import nimblecas.testing;

using nimblecas::MathError;
using nimblecas::Polynomial;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Build a RationalPoly from integer coefficients (low degree first).
[[nodiscard]] auto ipoly(std::vector<std::int64_t> c) -> RationalPoly {
    return RationalPoly::from_polynomial(Polynomial{std::move(c)});
}

[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.ratpoly")
        .test("rational_canonicalisation",
              [](TestContext& t) {
                  t.expect(rat(2, 4) == rat(1, 2), "2/4 reduces to 1/2");
                  t.expect(rat(-1, -2) == rat(1, 2), "-1/-2 reduces to 1/2");
                  t.expect(rat(3, -6) == rat(-1, 2), "sign moves to the numerator");
                  t.expect(rat(0, 5) == Rational{}, "0/5 is the zero rational");
                  t.expect(Rational::from_int(7).is_integer(), "7 is integral");
                  t.expect(!Rational::make(1, 0).has_value(), "1/0 is division_by_zero");
              })
        .test("rational_arithmetic",
              [](TestContext& t) {
                  // 1/2 + 1/3 = 5/6
                  t.expect(rat(1, 2).add(rat(1, 3)).value() == rat(5, 6), "1/2 + 1/3 = 5/6");
                  // 1/2 - 1/3 = 1/6
                  t.expect(rat(1, 2).subtract(rat(1, 3)).value() == rat(1, 6), "1/2 - 1/3 = 1/6");
                  // 2/3 * 3/4 = 1/2
                  t.expect(rat(2, 3).multiply(rat(3, 4)).value() == rat(1, 2), "2/3 * 3/4 = 1/2");
                  // (2/3) / (4/9) = 3/2
                  t.expect(rat(2, 3).divide(rat(4, 9)).value() == rat(3, 2), "(2/3)/(4/9) = 3/2");
                  t.expect(!rat(1, 2).divide(Rational{}).has_value(), "divide by zero fails");
                  // overflow is reported, not wrapped
                  const std::int64_t big = std::numeric_limits<std::int64_t>::max();
                  t.expect(Rational::make(big, 1).value().multiply(Rational::from_int(2)).error() ==
                               MathError::overflow,
                           "int64 overflow in multiply is reported");
              })
        .test("division_with_remainder",
              [](TestContext& t) {
                  // (x^2 - 1) / (x - 1) = x + 1, remainder 0
                  auto dm = ipoly({-1, 0, 1}).divide(ipoly({-1, 1})).value();
                  t.expect(dm.quotient.is_equal(ipoly({1, 1})), "quotient is x + 1");
                  t.expect(dm.remainder.is_zero(), "remainder is 0");

                  // (x^2 + 1) / (2x) : non-integer coefficients appear in the quotient.
                  // x^2 + 1 = (1/2 x)(2x) + 1  =>  quotient 1/2 x, remainder 1
                  auto dm2 = ipoly({1, 0, 1}).divide(ipoly({0, 2})).value();
                  t.expect(dm2.quotient.is_equal(
                               RationalPoly::from_coeffs({Rational{}, rat(1, 2)})),
                           "quotient is (1/2) x");
                  t.expect(dm2.remainder.is_equal(ipoly({1})), "remainder is 1");

                  // reconstruction a == q*b + r for a non-trivial pair
                  auto a = ipoly({-4, 0, -2, 1});  // x^3 - 2x^2 - 4
                  auto b = ipoly({-3, 1});         // x - 3
                  auto d = a.divide(b).value();
                  auto qb = d.quotient.multiply(b).value();
                  auto recon = qb.add(d.remainder).value();
                  t.expect(recon.is_equal(a), "a == q*b + r");
                  t.expect(d.remainder.degree() < b.degree(), "deg(r) < deg(b)");
              })
        .test("gcd_is_monic",
              [](TestContext& t) {
                  // gcd(x^2 - 1, x - 1) = x - 1 (already monic)
                  auto g = ipoly({-1, 0, 1}).gcd(ipoly({-1, 1})).value();
                  t.expect(g.is_equal(ipoly({-1, 1})), "gcd(x^2-1, x-1) = x - 1");

                  // gcd(2x^2 - 2, 2x - 2) = x - 1 (monic normalisation divides out the 2)
                  auto g2 = ipoly({-2, 0, 2}).gcd(ipoly({-2, 2})).value();
                  t.expect(g2.is_equal(ipoly({-1, 1})), "gcd is normalised to monic x - 1");

                  // coprime -> gcd is the constant 1
                  auto g3 = ipoly({1, 1}).gcd(ipoly({-1, 1})).value();  // gcd(x+1, x-1)
                  t.expect(g3.is_equal(ipoly({1})), "gcd(x+1, x-1) = 1");

                  // gcd with zero returns the monic other operand
                  auto g4 = RationalPoly{}.gcd(ipoly({-2, 0, 2})).value();  // gcd(0, 2x^2-2)
                  t.expect(g4.is_equal(ipoly({-1, 0, 1})), "gcd(0, 2x^2-2) = x^2 - 1 (monic)");
              })
        .test("derivative_and_conversions",
              [](TestContext& t) {
                  // d/dx (x^3 + 2x + 1) = 3x^2 + 2
                  auto d = ipoly({1, 2, 0, 1}).derivative().value();
                  t.expect(d.is_equal(ipoly({2, 0, 3})), "derivative is 3x^2 + 2");

                  // round-trip Z[x] -> Q[x] -> Z[x]
                  Polynomial p{{3, -5, 7}};
                  auto back = RationalPoly::from_polynomial(p).to_polynomial().value();
                  t.expect(back.is_equal(p), "integer polynomial round-trips");

                  // a genuinely fractional polynomial cannot convert back to Z[x]
                  auto frac = RationalPoly::from_coeffs({rat(1, 2)});
                  t.expect(frac.to_polynomial().error() == MathError::domain_error,
                           "1/2 is not an integer polynomial");
              })
        .run();
}
