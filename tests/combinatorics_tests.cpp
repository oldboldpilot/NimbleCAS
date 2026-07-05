// Tests for nimblecas.combinatorics: factorials, binomials, permutations, Catalan,
// Fibonacci, Stirling numbers, and Bernoulli numbers.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.combinatorics;
import nimblecas.testing;

using nimblecas::bernoulli;
using nimblecas::binomial;
using nimblecas::catalan;
using nimblecas::factorial;
using nimblecas::fibonacci;
using nimblecas::generalized_harmonic;
using nimblecas::harmonic;
using nimblecas::MathError;
using nimblecas::permutations;
using nimblecas::Rational;
using nimblecas::stirling_first;
using nimblecas::stirling_second;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.combinatorics")
        .test("factorial",
              [](TestContext& t) {
                  t.expect(factorial(0).value() == 1, "0! = 1");
                  t.expect(factorial(1).value() == 1, "1! = 1");
                  t.expect(factorial(5).value() == 120, "5! = 120");
                  t.expect(factorial(10).value() == 3628800, "10! = 3628800");
                  t.expect(factorial(20).value() == 2432902008176640000LL,
                           "20! is the largest factorial in int64");
                  t.expect(factorial(21).error() == MathError::overflow, "21! overflows");
                  t.expect(factorial(-1).error() == MathError::domain_error,
                           "negative factorial is a domain error");
              })
        .test("binomial",
              [](TestContext& t) {
                  t.expect(binomial(52, 5).value() == 2598960, "C(52, 5) = 2598960");
                  t.expect(binomial(10, 0).value() == 1, "C(10, 0) = 1");
                  t.expect(binomial(10, 10).value() == 1, "C(10, 10) = 1");
                  t.expect(binomial(5, 7).value() == 0, "C(5, 7) = 0 (k > n)");
                  t.expect(binomial(5, -1).value() == 0, "C(5, -1) = 0 (k < 0)");
                  t.expect(binomial(6, 3).value() == 20, "C(6, 3) = 20 (symmetry)");
                  t.expect(binomial(30, 15).value() == 155117520, "C(30, 15) = 155117520");
                  // Large but exact: C(62, 31) ~ 4.65e17 fits in int64, and the
                  // running value never exceeds the answer thanks to divide-as-you-go.
                  // Verify via Pascal's identity C(n,k) = C(n-1,k-1) + C(n-1,k).
                  t.expect(binomial(62, 31).value() ==
                               binomial(61, 30).value() + binomial(61, 31).value(),
                           "C(62, 31) is exact (Pascal) and does not spuriously overflow");
                  t.expect(binomial(-1, 0).error() == MathError::domain_error,
                           "negative n is a domain error");
              })
        .test("permutations",
              [](TestContext& t) {
                  t.expect(permutations(5, 3).value() == 60, "P(5, 3) = 60");
                  t.expect(permutations(5, 0).value() == 1, "P(5, 0) = 1");
                  t.expect(permutations(5, 5).value() == 120, "P(5, 5) = 5!");
                  t.expect(permutations(5, 7).value() == 0, "P(5, 7) = 0 (k > n)");
                  t.expect(permutations(-1, 2).error() == MathError::domain_error,
                           "negative n is a domain error");
              })
        .test("catalan",
              [](TestContext& t) {
                  const std::array<std::int64_t, 6> expected{1, 1, 2, 5, 14, 42};
                  for (std::int64_t n = 0; n < 6; ++n) {
                      t.expect(catalan(n).value() == expected[static_cast<std::size_t>(n)],
                               "Catalan(0..5) = 1, 1, 2, 5, 14, 42");
                  }
                  t.expect(catalan(15).value() == 9694845, "Catalan(15) = 9694845");
                  t.expect(catalan(-1).error() == MathError::domain_error,
                           "negative index is a domain error");
              })
        .test("fibonacci",
              [](TestContext& t) {
                  t.expect(fibonacci(0).value() == 0, "F(0) = 0");
                  t.expect(fibonacci(1).value() == 1, "F(1) = 1");
                  t.expect(fibonacci(10).value() == 55, "F(10) = 55");
                  t.expect(fibonacci(90).value() == 2880067194370816120LL,
                           "F(90) is large but fits int64");
                  t.expect(fibonacci(92).value() == 7540113804746346429LL,
                           "F(92) is the largest Fibonacci number in int64");
                  t.expect(fibonacci(93).error() == MathError::overflow, "F(93) overflows");
                  t.expect(fibonacci(-1).error() == MathError::domain_error,
                           "negative index is a domain error");
              })
        .test("stirling_second",
              [](TestContext& t) {
                  t.expect(stirling_second(4, 2).value() == 7, "S(4, 2) = 7");
                  t.expect(stirling_second(0, 0).value() == 1, "S(0, 0) = 1");
                  t.expect(stirling_second(3, 0).value() == 0, "S(3, 0) = 0");
                  t.expect(stirling_second(5, 3).value() == 25, "S(5, 3) = 25");
                  t.expect(stirling_second(5, 5).value() == 1, "S(5, 5) = 1");
                  t.expect(stirling_second(3, 5).value() == 0, "S(3, 5) = 0 (k > n)");
                  t.expect(stirling_second(-1, 0).error() == MathError::domain_error,
                           "negative n is a domain error");
              })
        .test("stirling_first",
              [](TestContext& t) {
                  t.expect(stirling_first(4, 2).value() == 11, "c(4, 2) = 11");
                  t.expect(stirling_first(0, 0).value() == 1, "c(0, 0) = 1");
                  t.expect(stirling_first(4, 1).value() == 6, "c(4, 1) = 3! = 6");
                  t.expect(stirling_first(4, 4).value() == 1, "c(4, 4) = 1");
                  t.expect(stirling_first(3, 5).value() == 0, "c(3, 5) = 0 (k > n)");
              })
        .test("bernoulli",
              [](TestContext& t) {
                  t.expect(bernoulli(0).value() == Rational::from_int(1), "B_0 = 1");
                  t.expect(bernoulli(1).value() == rat(-1, 2), "B_1 = -1/2 (first convention)");
                  t.expect(bernoulli(2).value() == rat(1, 6), "B_2 = 1/6");
                  t.expect(bernoulli(3).value() == Rational{}, "B_3 = 0");
                  t.expect(bernoulli(4).value() == rat(-1, 30), "B_4 = -1/30");
                  t.expect(bernoulli(5).value() == Rational{}, "B_5 = 0");
                  t.expect(bernoulli(6).value() == rat(1, 42), "B_6 = 1/42");
                  t.expect(bernoulli(8).value() == rat(-1, 30), "B_8 = -1/30");
                  t.expect(bernoulli(10).value() == rat(5, 66), "B_10 = 5/66");
                  t.expect(bernoulli(-1).error() == MathError::domain_error,
                           "negative index is a domain error");
              })
        .test("harmonic",
              [](TestContext& t) {
                  t.expect(harmonic(0).value() == Rational{}, "H_0 = 0");
                  t.expect(harmonic(1).value() == Rational::from_int(1), "H_1 = 1");
                  t.expect(harmonic(2).value() == rat(3, 2), "H_2 = 3/2");
                  t.expect(harmonic(3).value() == rat(11, 6), "H_3 = 11/6");
                  t.expect(harmonic(4).value() == rat(25, 12), "H_4 = 25/12");
                  t.expect(harmonic(-1).error() == MathError::domain_error,
                           "negative n is a domain error");
              })
        .test("generalized_harmonic",
              [](TestContext& t) {
                  // H_{n,1} coincides with the ordinary harmonic number H_n.
                  t.expect(generalized_harmonic(4, 1).value() == harmonic(4).value(),
                           "H_{4,1} == H_4 = 25/12");
                  t.expect(generalized_harmonic(2, 2).value() == rat(5, 4),
                           "H_{2,2} = 1 + 1/4 = 5/4");
                  t.expect(generalized_harmonic(3, 2).value() == rat(49, 36),
                           "H_{3,2} = 1 + 1/4 + 1/9 = 49/36");
                  t.expect(generalized_harmonic(2, 3).value() == rat(9, 8),
                           "H_{2,3} = 1 + 1/8 = 9/8");
                  t.expect(generalized_harmonic(0, 2).value() == Rational{},
                           "H_{0,2} = 0 (empty sum)");
                  t.expect(generalized_harmonic(-1, 2).error() == MathError::domain_error,
                           "negative n is a domain error");
                  t.expect(generalized_harmonic(2, 0).error() == MathError::domain_error,
                           "r < 1 is a domain error");
              })
        .run();
}
