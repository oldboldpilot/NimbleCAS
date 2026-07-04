// Tests for nimblecas.bigcombinatorics: the unbounded (BigInt-backed) exact integer
// combinatorics — factorial, binomial, multinomial, Catalan, falling/rising factorial,
// double factorial, Fibonacci (fast doubling), Lucas, Stirling numbers, and Bell numbers.
// The suite pins hand-verified big literals BEYOND the int64 range, checks combinatorial
// identities on large arguments, and cross-checks every overlapping function against the
// int64 nimblecas.combinatorics module for the small n where the latter does not overflow.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigcombinatorics;
import nimblecas.combinatorics;
import nimblecas.testing;

using nimblecas::BigInt;
using nimblecas::MathError;
using nimblecas::Result;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

// Non-clashing unbounded functions (names unique to nimblecas.bigcombinatorics).
using nimblecas::bell;
using nimblecas::double_factorial;
using nimblecas::falling_factorial;
using nimblecas::lucas;
using nimblecas::multinomial;
using nimblecas::rising_factorial;
using nimblecas::stirling_first_unsigned;

// Non-clashing int64 functions used for cross-checking (names unique to
// nimblecas.combinatorics).
using nimblecas::permutations;
using nimblecas::stirling_first;

namespace {

// factorial/binomial/catalan/fibonacci/stirling_second are exported by BOTH modules and
// differ only in return type, so a direct call with both imported is ambiguous. Select
// each overload by its target type via a typed function pointer (overload resolution on
// the address of an overloaded function is done against the target type, [over.over]).
using BigFn1 = Result<BigInt> (*)(std::int64_t);
using BigFn2 = Result<BigInt> (*)(std::int64_t, std::int64_t);
using I64Fn1 = Result<std::int64_t> (*)(std::int64_t);
using I64Fn2 = Result<std::int64_t> (*)(std::int64_t, std::int64_t);

const BigFn1 bg_factorial = static_cast<BigFn1>(&nimblecas::factorial);
const BigFn2 bg_binomial = static_cast<BigFn2>(&nimblecas::binomial);
const BigFn1 bg_catalan = static_cast<BigFn1>(&nimblecas::catalan);
const BigFn1 bg_fibonacci = static_cast<BigFn1>(&nimblecas::fibonacci);
const BigFn2 bg_stirling2 = static_cast<BigFn2>(&nimblecas::stirling_second);

const I64Fn1 i64_factorial = static_cast<I64Fn1>(&nimblecas::factorial);
const I64Fn2 i64_binomial = static_cast<I64Fn2>(&nimblecas::binomial);
const I64Fn1 i64_catalan = static_cast<I64Fn1>(&nimblecas::catalan);
const I64Fn1 i64_fibonacci = static_cast<I64Fn1>(&nimblecas::fibonacci);
const I64Fn2 i64_stirling2 = static_cast<I64Fn2>(&nimblecas::stirling_second);

// True when a successful Result<BigInt> renders to exactly the decimal string s.
[[nodiscard]] auto big_is(const Result<BigInt>& r, std::string_view s) -> bool {
    return r.has_value() && r->to_string() == s;
}

// True when both results succeeded and the BigInt equals the int64 value (compared as
// decimal text, which is how the cross-check is specified).
[[nodiscard]] auto matches_i64(const Result<BigInt>& big,
                               const Result<std::int64_t>& small) -> bool {
    return big.has_value() && small.has_value() &&
           big->to_string() == std::to_string(small.value());
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.bigcombinatorics")
        .test("factorial",
              [](TestContext& t) {
                  t.expect(big_is(bg_factorial(0), "1"), "0! = 1");
                  t.expect(big_is(bg_factorial(1), "1"), "1! = 1");
                  t.expect(big_is(bg_factorial(5), "120"), "5! = 120");
                  t.expect(big_is(bg_factorial(10), "3628800"), "10! = 3628800");
                  // 20! is the largest factorial that fits int64; still exact here.
                  t.expect(big_is(bg_factorial(20), "2432902008176640000"),
                           "20! = 2432902008176640000");
                  // 25! overflows int64 (21! already does) but is exact as a BigInt.
                  t.expect(big_is(bg_factorial(25), "15511210043330985984000000"),
                           "25! is exact beyond int64");
                  // 50! to its full 65-digit decimal expansion.
                  t.expect(big_is(bg_factorial(50),
                                  "3041409320171337804361260816606476884437764156896"
                                  "0512000000000000"),
                           "50! full digit string");
                  t.expect(bg_factorial(-1).error() == MathError::domain_error,
                           "negative factorial is a domain error");
              })
        .test("binomial",
              [](TestContext& t) {
                  t.expect(big_is(bg_binomial(52, 5), "2598960"), "C(52, 5) = 2598960");
                  t.expect(big_is(bg_binomial(10, 0), "1"), "C(10, 0) = 1");
                  t.expect(big_is(bg_binomial(10, 10), "1"), "C(10, 10) = 1");
                  t.expect(big_is(bg_binomial(5, 7), "0"), "C(5, 7) = 0 (k > n)");
                  t.expect(big_is(bg_binomial(5, -1), "0"), "C(5, -1) = 0 (k < 0)");
                  // The exact 30-digit central binomial coefficient C(100, 50).
                  t.expect(big_is(bg_binomial(100, 50),
                                  "100891344545564193334812497256"),
                           "C(100, 50) = 100891344545564193334812497256");
                  // Symmetry C(n, k) == C(n, n-k) on a large, beyond-int64 case.
                  t.expect(bg_binomial(100, 30).value() == bg_binomial(100, 70).value(),
                           "C(100, 30) == C(100, 70) (symmetry)");
                  // Pascal's rule C(n, k) == C(n-1, k-1) + C(n-1, k) on large n.
                  t.expect(bg_binomial(200, 100).value() ==
                               bg_binomial(199, 99).value().add(
                                   bg_binomial(199, 100).value()),
                           "C(200, 100) == C(199, 99) + C(199, 100) (Pascal)");
                  t.expect(bg_binomial(-1, 0).error() == MathError::domain_error,
                           "negative n is a domain error");
              })
        .test("multinomial",
              [](TestContext& t) {
                  const std::array<std::int64_t, 3> ones{1, 1, 1};
                  t.expect(big_is(multinomial(ones), "6"),
                           "multinomial(1,1,1) = 3!/(1!1!1!) = 6");
                  const std::array<std::int64_t, 3> mix{2, 3, 4};
                  t.expect(big_is(multinomial(mix), "1260"),
                           "multinomial(2,3,4) = 9!/(2!3!4!) = 1260");
                  // An empty multiset: the empty product is 1.
                  t.expect(big_is(multinomial(std::span<const std::int64_t>{}), "1"),
                           "multinomial() = 1 (empty product)");
                  // Two-part multinomial reduces to a binomial: (30+70)!/(30!70!)==C(100,30).
                  const std::array<std::int64_t, 2> two{30, 70};
                  t.expect(multinomial(two).value() == bg_binomial(100, 30).value(),
                           "multinomial(30,70) == C(100, 30)");
                  const std::array<std::int64_t, 2> bad{-1, 2};
                  t.expect(multinomial(bad).error() == MathError::domain_error,
                           "a negative multiplicity is a domain error");
              })
        .test("catalan",
              [](TestContext& t) {
                  const std::array<const char*, 6> expected{"1", "1", "2", "5", "14", "42"};
                  for (std::int64_t n = 0; n < 6; ++n) {
                      t.expect(big_is(bg_catalan(n), expected[static_cast<std::size_t>(n)]),
                               "Catalan(0..5) = 1, 1, 2, 5, 14, 42");
                  }
                  // Catalan(20) is exact here (the int64 module overflows from n = 36).
                  t.expect(big_is(bg_catalan(20), "6564120420"),
                           "Catalan(20) = 6564120420");
                  // Cross-identity: C_n == C(2n, n) - C(2n, n+1).
                  t.expect(bg_catalan(25).value() ==
                               bg_binomial(50, 25).value().subtract(
                                   bg_binomial(50, 26).value()),
                           "Catalan(25) == C(50,25) - C(50,26)");
                  t.expect(bg_catalan(-1).error() == MathError::domain_error,
                           "negative index is a domain error");
              })
        .test("falling_and_rising_factorial",
              [](TestContext& t) {
                  t.expect(big_is(falling_factorial(5, 0), "1"),
                           "falling(5, 0) = 1 (empty product)");
                  t.expect(big_is(falling_factorial(5, 3), "60"),
                           "falling(5, 3) = 5*4*3 = 60");
                  t.expect(big_is(falling_factorial(5, 5), "120"),
                           "falling(5, 5) = 5! = 120");
                  // n is unrestricted: a negative base yields a signed result.
                  t.expect(big_is(falling_factorial(-1, 3), "-6"),
                           "falling(-1, 3) = (-1)(-2)(-3) = -6");
                  t.expect(big_is(rising_factorial(2, 0), "1"),
                           "rising(2, 0) = 1 (empty product)");
                  t.expect(big_is(rising_factorial(2, 4), "120"),
                           "rising(2, 4) = 2*3*4*5 = 120");
                  t.expect(big_is(rising_factorial(1, 6), "720"),
                           "rising(1, 6) = 6! = 720");
                  // Identity rising(n, k) == falling(n+k-1, k).
                  t.expect(rising_factorial(7, 5).value() ==
                               falling_factorial(11, 5).value(),
                           "rising(7,5) == falling(7+5-1, 5)");
                  t.expect(falling_factorial(5, -1).error() == MathError::domain_error,
                           "falling with k < 0 is a domain error");
                  t.expect(rising_factorial(5, -1).error() == MathError::domain_error,
                           "rising with k < 0 is a domain error");
              })
        .test("double_factorial",
              [](TestContext& t) {
                  t.expect(big_is(double_factorial(-1), "1"), "(-1)!! = 1");
                  t.expect(big_is(double_factorial(0), "1"), "0!! = 1");
                  t.expect(big_is(double_factorial(1), "1"), "1!! = 1");
                  t.expect(big_is(double_factorial(9), "945"),
                           "9!! = 9*7*5*3*1 = 945");
                  t.expect(big_is(double_factorial(8), "384"),
                           "8!! = 8*6*4*2 = 384");
                  t.expect(big_is(double_factorial(7), "105"),
                           "7!! = 7*5*3*1 = 105");
                  t.expect(double_factorial(-2).error() == MathError::domain_error,
                           "n < -1 is a domain error");
              })
        .test("fibonacci",
              [](TestContext& t) {
                  t.expect(big_is(bg_fibonacci(0), "0"), "F(0) = 0");
                  t.expect(big_is(bg_fibonacci(1), "1"), "F(1) = 1");
                  t.expect(big_is(bg_fibonacci(2), "1"), "F(2) = 1");
                  t.expect(big_is(bg_fibonacci(10), "55"), "F(10) = 55");
                  t.expect(big_is(bg_fibonacci(100), "354224848179261915075"),
                           "F(100) = 354224848179261915075");
                  // F(200) is far beyond int64 (F(93) already overflows it).
                  t.expect(big_is(bg_fibonacci(200),
                                  "280571172992510140037611932413038677189525"),
                           "F(200) is exact beyond int64");
                  // Fast doubling must agree with the naive iterative recurrence.
                  std::vector<BigInt> naive;
                  naive.push_back(BigInt{});             // F(0)
                  naive.push_back(BigInt::from_u64(1));  // F(1)
                  for (std::size_t i = 2; i <= 50; ++i) {
                      naive.push_back(naive[i - 1].add(naive[i - 2]));
                  }
                  bool all_match = true;
                  for (std::int64_t i = 0; i <= 50; ++i) {
                      if (bg_fibonacci(i).value().to_string() !=
                          naive[static_cast<std::size_t>(i)].to_string()) {
                          all_match = false;
                      }
                  }
                  t.expect(all_match,
                           "fast-doubling F(n) matches naive iterative F(n) for n in 0..50");
                  t.expect(bg_fibonacci(-1).error() == MathError::domain_error,
                           "negative index is a domain error");
              })
        .test("lucas",
              [](TestContext& t) {
                  t.expect(big_is(lucas(0), "2"), "L(0) = 2");
                  t.expect(big_is(lucas(1), "1"), "L(1) = 1");
                  t.expect(big_is(lucas(2), "3"), "L(2) = 3");
                  t.expect(big_is(lucas(10), "123"), "L(10) = 123");
                  t.expect(big_is(lucas(50), "28143753123"), "L(50) = 28143753123");
                  // Identity L(n) == F(n-1) + F(n+1) == 2 F(n+1) - F(n).
                  t.expect(lucas(40).value() ==
                               bg_fibonacci(41).value().add(bg_fibonacci(41).value())
                                   .subtract(bg_fibonacci(40).value()),
                           "L(40) == 2 F(41) - F(40)");
                  t.expect(lucas(-1).error() == MathError::domain_error,
                           "negative index is a domain error");
              })
        .test("stirling_second",
              [](TestContext& t) {
                  t.expect(big_is(bg_stirling2(0, 0), "1"), "S(0, 0) = 1");
                  t.expect(big_is(bg_stirling2(4, 2), "7"), "S(4, 2) = 7");
                  t.expect(big_is(bg_stirling2(5, 3), "25"), "S(5, 3) = 25");
                  t.expect(big_is(bg_stirling2(10, 3), "9330"), "S(10, 3) = 9330");
                  t.expect(big_is(bg_stirling2(3, 0), "0"), "S(3, 0) = 0");
                  t.expect(big_is(bg_stirling2(3, 5), "0"), "S(3, 5) = 0 (k > n)");
                  // Row sum: sum_k S(n, k) == Bell(n).
                  BigInt row_sum{};
                  for (std::int64_t k = 0; k <= 10; ++k) {
                      row_sum = row_sum.add(bg_stirling2(10, k).value());
                  }
                  t.expect(row_sum == bell(10).value(),
                           "sum_k S(10, k) == Bell(10)");
                  t.expect(bg_stirling2(-1, 0).error() == MathError::domain_error,
                           "negative n is a domain error");
                  t.expect(bg_stirling2(2, -1).error() == MathError::domain_error,
                           "negative k is a domain error");
              })
        .test("stirling_first_unsigned",
              [](TestContext& t) {
                  t.expect(big_is(stirling_first_unsigned(0, 0), "1"), "c(0, 0) = 1");
                  t.expect(big_is(stirling_first_unsigned(4, 2), "11"), "c(4, 2) = 11");
                  t.expect(big_is(stirling_first_unsigned(4, 1), "6"), "c(4, 1) = 3! = 6");
                  t.expect(big_is(stirling_first_unsigned(4, 4), "1"), "c(4, 4) = 1");
                  t.expect(big_is(stirling_first_unsigned(3, 5), "0"), "c(3, 5) = 0 (k > n)");
                  // Row sum: sum_k c(n, k) == n!.
                  BigInt row_sum{};
                  for (std::int64_t k = 0; k <= 6; ++k) {
                      row_sum = row_sum.add(stirling_first_unsigned(6, k).value());
                  }
                  t.expect(row_sum == bg_factorial(6).value(),
                           "sum_k c(6, k) == 6! = 720");
                  t.expect(stirling_first_unsigned(-1, 0).error() == MathError::domain_error,
                           "negative n is a domain error");
                  t.expect(stirling_first_unsigned(2, -1).error() == MathError::domain_error,
                           "negative k is a domain error");
              })
        .test("bell",
              [](TestContext& t) {
                  t.expect(big_is(bell(0), "1"), "Bell(0) = 1");
                  t.expect(big_is(bell(1), "1"), "Bell(1) = 1");
                  t.expect(big_is(bell(2), "2"), "Bell(2) = 2");
                  t.expect(big_is(bell(5), "52"), "Bell(5) = 52");
                  t.expect(big_is(bell(10), "115975"), "Bell(10) = 115975");
                  t.expect(bell(-1).error() == MathError::domain_error,
                           "negative index is a domain error");
              })
        .test("cross_check_against_int64_combinatorics",
              [](TestContext& t) {
                  // Where the int64 module does not overflow, the BigInt result must equal
                  // it exactly (compared as decimal text).
                  bool fact_ok = true;
                  for (std::int64_t n = 0; n <= 20; ++n) {
                      if (!matches_i64(bg_factorial(n), i64_factorial(n))) {
                          fact_ok = false;
                      }
                  }
                  t.expect(fact_ok, "factorial(n) matches int64 module for n in 0..20");

                  bool fib_ok = true;
                  for (std::int64_t n = 0; n <= 90; ++n) {
                      if (!matches_i64(bg_fibonacci(n), i64_fibonacci(n))) {
                          fib_ok = false;
                      }
                  }
                  t.expect(fib_ok, "fibonacci(n) matches int64 module for n in 0..90");

                  bool cat_ok = true;
                  for (std::int64_t n = 0; n <= 15; ++n) {
                      if (!matches_i64(bg_catalan(n), i64_catalan(n))) {
                          cat_ok = false;
                      }
                  }
                  t.expect(cat_ok, "catalan(n) matches int64 module for n in 0..15");

                  t.expect(matches_i64(bg_binomial(52, 5), i64_binomial(52, 5)),
                           "binomial(52, 5) matches int64 module");
                  t.expect(matches_i64(bg_binomial(30, 15), i64_binomial(30, 15)),
                           "binomial(30, 15) matches int64 module");
                  t.expect(matches_i64(bg_binomial(62, 31), i64_binomial(62, 31)),
                           "binomial(62, 31) matches int64 module");

                  t.expect(matches_i64(bg_stirling2(9, 4), i64_stirling2(9, 4)),
                           "stirling_second(9, 4) matches int64 module");

                  // falling_factorial mirrors the int64 permutations (falling factorial).
                  t.expect(matches_i64(falling_factorial(6, 4), permutations(6, 4)),
                           "falling_factorial(6, 4) matches int64 permutations(6, 4)");
                  // stirling_first_unsigned mirrors the int64 unsigned stirling_first.
                  t.expect(matches_i64(stirling_first_unsigned(7, 3), stirling_first(7, 3)),
                           "stirling_first_unsigned(7, 3) matches int64 stirling_first(7, 3)");
              })
        .run();
}
