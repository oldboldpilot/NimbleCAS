// Tests for nimblecas.ratint: Hermite reduction of rational functions over Q(x).
// @author Olumuyiwa Oluwasanmi
//
// The decisive check is differentiate-back: for int A/D = g + int h, verify
//   d/dx(g) + h == A/D  exactly, by cross-multiplying the two rational functions.

import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.ratpoly;
import nimblecas.ratint;
import nimblecas.testing;

using nimblecas::hermite_reduce;
using nimblecas::HermiteReduction;
using nimblecas::MathError;
using nimblecas::Polynomial;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto ipoly(std::vector<std::int64_t> c) -> RationalPoly {
    return RationalPoly::from_polynomial(Polynomial{std::move(c)});
}

// d/dx(g) + h == A/D, checked exactly via cross-multiplication (L_num*D == A*L_den).
[[nodiscard]] auto integrates_to(const RationalPoly& a, const RationalPoly& d,
                                 const HermiteReduction& hr) -> bool {
    // g' = (rn' * rd - rn * rd') / rd^2.
    const RationalPoly rnp = hr.rational_num.derivative().value();
    const RationalPoly rdp = hr.rational_den.derivative().value();
    const RationalPoly gnum = rnp.multiply(hr.rational_den).value().subtract(
        hr.rational_num.multiply(rdp).value()).value();
    const RationalPoly gden = hr.rational_den.multiply(hr.rational_den).value();
    // LHS = g' + h = (gnum*hd + hn*gden) / (gden*hd).
    const RationalPoly lnum = gnum.multiply(hr.integrand_den).value().add(
        hr.integrand_num.multiply(gden).value()).value();
    const RationalPoly lden = gden.multiply(hr.integrand_den).value();
    // LHS == A/D  <=>  lnum*D == A*lden.
    return lnum.multiply(d).value().is_equal(a.multiply(lden).value());
}

// gcd(p, p') is a constant -> p is square-free.
[[nodiscard]] auto is_square_free(const RationalPoly& p) -> bool {
    if (p.degree() <= 0) {
        return true;
    }
    const RationalPoly g = p.gcd(p.derivative().value()).value();
    return g.degree() == 0;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.ratint")
        .test("pure_rational_double_pole",
              [](TestContext& t) {
                  // int 1/(x-1)^2 dx = -1/(x-1); no logarithmic part.
                  auto a = ipoly({1});
                  auto d = ipoly({1, -2, 1});  // (x - 1)^2
                  auto hr = hermite_reduce(a, d).value();
                  t.expect(hr.integrand_num.is_zero(), "no leftover integrand (pure rational)");
                  t.expect(!hr.rational_num.is_zero(), "a rational part was extracted");
                  t.expect(integrates_to(a, d, hr), "d/dx(g) + h == A/D");
              })
        .test("pure_logarithmic_squarefree",
              [](TestContext& t) {
                  // int 1/(x^2 - 1) dx is purely logarithmic: g == 0, integrand unchanged.
                  auto a = ipoly({1});
                  auto d = ipoly({-1, 0, 1});  // x^2 - 1 (square-free)
                  auto hr = hermite_reduce(a, d).value();
                  t.expect(hr.rational_num.is_zero(), "no rational part");
                  t.expect(is_square_free(hr.integrand_den), "integrand denominator square-free");
                  t.expect(integrates_to(a, d, hr), "d/dx(g) + h == A/D");
              })
        .test("both_parts_present",
              [](TestContext& t) {
                  // int 1/(x^2+1)^2 = x/(2(x^2+1)) + (1/2) int 1/(x^2+1): BOTH parts nonzero.
                  auto a = ipoly({1});
                  auto d = ipoly({1, 0, 2, 0, 1});  // (x^2 + 1)^2
                  auto hr = hermite_reduce(a, d).value();
                  t.expect(!hr.rational_num.is_zero(), "rational part present");
                  t.expect(!hr.integrand_num.is_zero(), "logarithmic integrand present");
                  t.expect(hr.integrand_den.is_equal(ipoly({1, 0, 1})),
                           "square-free integrand denominator is x^2 + 1");
                  t.expect(integrates_to(a, d, hr), "d/dx(g) + h == A/D");
              })
        .test("double_pole_reduces_to_zero_integrand",
              [](TestContext& t) {
                  // int x/(x^2+1)^2 = -1/(2(x^2+1)): a double pole yet purely rational.
                  auto a = ipoly({0, 1});
                  auto d = ipoly({1, 0, 2, 0, 1});  // (x^2 + 1)^2
                  auto hr = hermite_reduce(a, d).value();
                  t.expect(hr.integrand_num.is_zero(), "no logarithmic part");
                  t.expect(!hr.rational_num.is_zero(), "rational part present");
                  t.expect(integrates_to(a, d, hr), "d/dx(g) + h == A/D");
              })
        .test("triple_pole",
              [](TestContext& t) {
                  // int (x^2 + 1)/(x - 1)^3 dx — a pole of order 3 exercises two reduction
                  // steps (k: 3 -> 2 -> 1).
                  auto a = ipoly({1, 0, 1});
                  auto d = ipoly({-1, 3, -3, 1});  // (x - 1)^3
                  auto hr = hermite_reduce(a, d).value();
                  t.expect(is_square_free(hr.integrand_den), "integrand denominator square-free");
                  t.expect(integrates_to(a, d, hr), "d/dx(g) + h == A/D");
              })
        .test("improper_with_polynomial_part",
              [](TestContext& t) {
                  // int x^3/(x-1)^2 dx: long division contributes a polynomial antiderivative.
                  auto a = ipoly({0, 0, 0, 1});
                  auto d = ipoly({1, -2, 1});  // (x - 1)^2
                  auto hr = hermite_reduce(a, d).value();
                  t.expect(integrates_to(a, d, hr), "d/dx(g) + h == A/D");
              })
        .test("purely_polynomial",
              [](TestContext& t) {
                  // int (6x)/3 dx = int 2x dx = x^2: constant denominator, no poles.
                  auto a = ipoly({0, 6});
                  auto d = ipoly({3});
                  auto hr = hermite_reduce(a, d).value();
                  t.expect(hr.integrand_num.is_zero(), "no integrand remains");
                  t.expect(hr.rational_den.degree() <= 0, "rational denominator is constant");
                  t.expect(hr.rational_num.is_equal(ipoly({0, 0, 1})), "antiderivative is x^2");
                  t.expect(integrates_to(a, d, hr), "d/dx(g) + h == A/D");
              })
        .test("mixed_multiplicity_and_errors",
              [](TestContext& t) {
                  // int x/((x-1)^2 (x+1)) dx: distinct multiplicities, rational + log parts.
                  auto a = ipoly({0, 1});
                  auto d = ipoly({1, -2, 1}).multiply(ipoly({1, 1})).value();  // (x-1)^2 (x+1)
                  auto hr = hermite_reduce(a, d).value();
                  t.expect(is_square_free(hr.integrand_den), "integrand denominator square-free");
                  t.expect(integrates_to(a, d, hr), "d/dx(g) + h == A/D");
                  // A zero denominator is a division-by-zero error.
                  t.expect(hermite_reduce(ipoly({1}), RationalPoly{}).error() ==
                               MathError::division_by_zero,
                           "zero denominator fails");
              })
        .run();
}
