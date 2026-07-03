// Tests for nimblecas.orthopoly: classical orthogonal polynomials via their
// three-term recurrences.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.orthopoly;
import nimblecas.testing;

using nimblecas::MathError;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

// Build a RationalPoly from rational coefficients (low degree first).
[[nodiscard]] auto rpoly(std::vector<Rational> c) -> RationalPoly {
    return RationalPoly::from_coeffs(std::move(c));
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.orthopoly")
        .test("chebyshev_first_kind",
              [](TestContext& t) {
                  // T_0 = 1
                  t.expect(nimblecas::chebyshev_t(0).value().is_equal(
                               RationalPoly::constant(Rational::from_int(1))),
                           "T_0 = 1");
                  // T_1 = x
                  t.expect(nimblecas::chebyshev_t(1).value().is_equal(
                               rpoly({rat(0, 1), rat(1, 1)})),
                           "T_1 = x");
                  // T_2 = 2x^2 - 1
                  t.expect(nimblecas::chebyshev_t(2).value().is_equal(
                               rpoly({rat(-1, 1), rat(0, 1), rat(2, 1)})),
                           "T_2 = 2x^2 - 1");
                  // T_3 = 4x^3 - 3x
                  t.expect(nimblecas::chebyshev_t(3).value().is_equal(
                               rpoly({rat(0, 1), rat(-3, 1), rat(0, 1), rat(4, 1)})),
                           "T_3 = 4x^3 - 3x");
              })
        .test("chebyshev_second_kind",
              [](TestContext& t) {
                  // U_1 = 2x
                  t.expect(nimblecas::chebyshev_u(1).value().is_equal(
                               rpoly({rat(0, 1), rat(2, 1)})),
                           "U_1 = 2x");
                  // U_2 = 4x^2 - 1
                  t.expect(nimblecas::chebyshev_u(2).value().is_equal(
                               rpoly({rat(-1, 1), rat(0, 1), rat(4, 1)})),
                           "U_2 = 4x^2 - 1");
              })
        .test("legendre",
              [](TestContext& t) {
                  // P_2 = (3x^2 - 1)/2
                  t.expect(nimblecas::legendre(2).value().is_equal(
                               rpoly({rat(-1, 2), rat(0, 1), rat(3, 2)})),
                           "P_2 = (3x^2 - 1)/2");
                  // P_3 = (5x^3 - 3x)/2
                  t.expect(nimblecas::legendre(3).value().is_equal(
                               rpoly({rat(0, 1), rat(-3, 2), rat(0, 1), rat(5, 2)})),
                           "P_3 = (5x^3 - 3x)/2");
              })
        .test("laguerre",
              [](TestContext& t) {
                  // L_1 = 1 - x
                  t.expect(nimblecas::laguerre(1).value().is_equal(
                               rpoly({rat(1, 1), rat(-1, 1)})),
                           "L_1 = 1 - x");
                  // L_2 = (x^2 - 4x + 2)/2 = 1 - 2x + (1/2)x^2
                  t.expect(nimblecas::laguerre(2).value().is_equal(
                               rpoly({rat(1, 1), rat(-2, 1), rat(1, 2)})),
                           "L_2 = 1 - 2x + (1/2)x^2");
              })
        .test("hermite_physicists",
              [](TestContext& t) {
                  // H_2 = 4x^2 - 2
                  t.expect(nimblecas::hermite_phys(2).value().is_equal(
                               rpoly({rat(-2, 1), rat(0, 1), rat(4, 1)})),
                           "H_2 = 4x^2 - 2");
                  // H_3 = 8x^3 - 12x
                  t.expect(nimblecas::hermite_phys(3).value().is_equal(
                               rpoly({rat(0, 1), rat(-12, 1), rat(0, 1), rat(8, 1)})),
                           "H_3 = 8x^3 - 12x");
              })
        .test("hermite_probabilists",
              [](TestContext& t) {
                  // He_2 = x^2 - 1
                  t.expect(nimblecas::hermite_prob(2).value().is_equal(
                               rpoly({rat(-1, 1), rat(0, 1), rat(1, 1)})),
                           "He_2 = x^2 - 1");
                  // He_3 = x^3 - 3x
                  t.expect(nimblecas::hermite_prob(3).value().is_equal(
                               rpoly({rat(0, 1), rat(-3, 1), rat(0, 1), rat(1, 1)})),
                           "He_3 = x^3 - 3x");
              })
        .test("base_cases_and_domain",
              [](TestContext& t) {
                  const RationalPoly unit = RationalPoly::constant(Rational::from_int(1));
                  // n = 0 returns the constant 1 for every family.
                  t.expect(nimblecas::chebyshev_t(0).value().is_equal(unit), "T_0 = 1");
                  t.expect(nimblecas::chebyshev_u(0).value().is_equal(unit), "U_0 = 1");
                  t.expect(nimblecas::legendre(0).value().is_equal(unit), "P_0 = 1");
                  t.expect(nimblecas::laguerre(0).value().is_equal(unit), "L_0 = 1");
                  t.expect(nimblecas::hermite_phys(0).value().is_equal(unit), "H_0 = 1");
                  t.expect(nimblecas::hermite_prob(0).value().is_equal(unit), "He_0 = 1");

                  // n < 0 is a domain error for every family.
                  t.expect(nimblecas::chebyshev_t(-1).error() == MathError::domain_error,
                           "T_{-1} is domain_error");
                  t.expect(nimblecas::chebyshev_u(-1).error() == MathError::domain_error,
                           "U_{-1} is domain_error");
                  t.expect(nimblecas::legendre(-1).error() == MathError::domain_error,
                           "P_{-1} is domain_error");
                  t.expect(nimblecas::laguerre(-1).error() == MathError::domain_error,
                           "L_{-1} is domain_error");
                  t.expect(nimblecas::hermite_phys(-1).error() == MathError::domain_error,
                           "H_{-1} is domain_error");
                  t.expect(nimblecas::hermite_prob(-1).error() == MathError::domain_error,
                           "He_{-1} is domain_error");
              })
        .run();
}
