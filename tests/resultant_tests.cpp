// Tests for nimblecas.resultant: resultant and discriminant over Q[x].
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.ratpoly;
import nimblecas.resultant;
import nimblecas.testing;

using nimblecas::discriminant;
using nimblecas::Polynomial;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::resultant;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto ipoly(std::vector<std::int64_t> c) -> RationalPoly {
    return RationalPoly::from_polynomial(Polynomial{std::move(c)});
}

[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.resultant")
        .test("resultant_linear_pairs",
              [](TestContext& t) {
                  // res(x - a, x - b) = a - b (Sylvester det [[1,-a],[1,-b]] = a - b... = b - ... check).
                  // res(x - 2, x - 5): Sylvester det [[1,-2],[1,-5]] = -5 + 2 = -3.
                  t.expect(resultant(ipoly({-2, 1}), ipoly({-5, 1})).value() ==
                               Rational::from_int(-3),
                           "res(x-2, x-5) = -3");
                  // Anti-symmetry: res(A,B) = (-1)^{deg A deg B} res(B,A); here degs 1,1 -> -.
                  t.expect(resultant(ipoly({-5, 1}), ipoly({-2, 1})).value() ==
                               Rational::from_int(3),
                           "res(x-5, x-2) = 3 (negated)");
              })
        .test("resultant_hand_checked",
              [](TestContext& t) {
                  // res(x^2 - 1, x - 2): Sylvester 3x3 det = 3.
                  t.expect(resultant(ipoly({-1, 0, 1}), ipoly({-2, 1})).value() ==
                               Rational::from_int(3),
                           "res(x^2-1, x-2) = 3");
                  // res(x^2 + 1, 2x) = 4.
                  t.expect(resultant(ipoly({1, 0, 1}), ipoly({0, 2})).value() ==
                               Rational::from_int(4),
                           "res(x^2+1, 2x) = 4");
              })
        .test("resultant_vanishes_on_common_factor",
              [](TestContext& t) {
                  // (x^2 - 1) and (x - 1) share the root 1 -> resultant 0.
                  t.expect(resultant(ipoly({-1, 0, 1}), ipoly({-1, 1})).value().is_zero(),
                           "res(x^2-1, x-1) = 0");
                  // res(A, A) = 0 for deg A >= 1.
                  t.expect(resultant(ipoly({2, 0, 3}), ipoly({2, 0, 3})).value().is_zero(),
                           "res(A, A) = 0");
                  // A zero operand gives 0.
                  t.expect(resultant(ipoly({1, 1}), RationalPoly{}).value().is_zero(),
                           "res(A, 0) = 0");
              })
        .test("resultant_multiplicative",
              [](TestContext& t) {
                  // res(A, B*C) = res(A, B) * res(A, C).
                  auto a = ipoly({1, 0, 1});  // x^2 + 1
                  auto b = ipoly({-1, 1});    // x - 1
                  auto c = ipoly({-2, 1});    // x - 2
                  auto bc = b.multiply(c).value();
                  auto lhs = resultant(a, bc).value();
                  auto rhs = resultant(a, b).value().multiply(resultant(a, c).value()).value();
                  t.expect(lhs == rhs, "res(A, B*C) = res(A,B) * res(A,C)");
                  t.expect(lhs == Rational::from_int(10), "the product is 10");
              })
        .test("resultant_of_constants",
              [](TestContext& t) {
                  // res of two non-zero constants is the empty product 1.
                  t.expect(resultant(ipoly({7}), ipoly({3})).value() == Rational::from_int(1),
                           "res(const, const) = 1");
                  // res(constant c, B) = c^{deg B}: res(3, x^2 - 1) = 9.
                  t.expect(resultant(ipoly({3}), ipoly({-1, 0, 1})).value() ==
                               Rational::from_int(9),
                           "res(3, x^2-1) = 3^2 = 9");
              })
        .test("discriminant_quadratics",
              [](TestContext& t) {
                  // disc(x^2 - 1) = 4 (roots +-1, spacing 2).
                  t.expect(discriminant(ipoly({-1, 0, 1})).value() == Rational::from_int(4),
                           "disc(x^2 - 1) = 4");
                  // disc(x^2 + 1) = -4 (complex roots).
                  t.expect(discriminant(ipoly({1, 0, 1})).value() == Rational::from_int(-4),
                           "disc(x^2 + 1) = -4");
                  // disc(x^2 - x) = b^2 - 4ac = 1.
                  t.expect(discriminant(ipoly({0, -1, 1})).value() == Rational::from_int(1),
                           "disc(x^2 - x) = 1");
                  // A repeated root gives discriminant 0.
                  t.expect(discriminant(ipoly({1, -2, 1})).value().is_zero(),
                           "disc((x-1)^2) = 0");
              })
        .test("discriminant_cubic_and_rational",
              [](TestContext& t) {
                  // disc(x^3 - x) = -4p^3 - 27q^2 with p=-1, q=0 -> 4.
                  t.expect(discriminant(ipoly({0, -1, 0, 1})).value() == Rational::from_int(4),
                           "disc(x^3 - x) = 4");
                  // Non-monic / rational coefficients: disc(2x^2 - 2) = -4*a*c = -4*2*(-2)=16.
                  t.expect(discriminant(ipoly({-2, 0, 2})).value() == Rational::from_int(16),
                           "disc(2x^2 - 2) = 16");
                  // Linear polynomials have discriminant 1 by convention.
                  t.expect(discriminant(ipoly({3, 1})).value() == Rational::from_int(1),
                           "disc(x + 3) = 1");
              })
        .run();
}
