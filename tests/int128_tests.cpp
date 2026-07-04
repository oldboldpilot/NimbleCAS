// Tests for nimblecas.int128: native 128-bit integers and exact 128-bit rationals.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.int128;
import nimblecas.ratpoly;
import nimblecas.testing;

using nimblecas::Int128;
using nimblecas::MathError;
using nimblecas::Rational;
using nimblecas::Rational128;
using nimblecas::checked_add;
using nimblecas::checked_mul;
using nimblecas::checked_sub;
using nimblecas::int128_from_string;
using nimblecas::int128_to_string;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Parse a decimal literal that may exceed the int64 range (test fixtures only; the
// value is expected to be well-formed and in range).
[[nodiscard]] auto i128(std::string_view s) -> Int128 {
    return int128_from_string(s).value();
}

[[nodiscard]] auto r128(Int128 n, Int128 d) -> Rational128 {
    return Rational128::make(n, d).value();
}

// The extreme signed 128-bit values, as decimal strings.
constexpr std::string_view kMax = "170141183460469231731687303715884105727";   //  2^127 - 1
constexpr std::string_view kMin = "-170141183460469231731687303715884105728";  // -2^127

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.int128")
        .test("int128_string_round_trips",
              [](TestContext& t) {
                  // The two extremes survive a parse -> render round-trip.
                  t.expect_eq(int128_to_string(i128(kMax)), std::string(kMax), "INT128_MAX round-trips");
                  t.expect_eq(int128_to_string(i128(kMin)), std::string(kMin), "INT128_MIN round-trips");

                  // A value far beyond int64 (10^30) round-trips exactly.
                  const std::string_view p30 = "1000000000000000000000000000000";
                  t.expect_eq(int128_to_string(i128(p30)), std::string(p30), "10^30 round-trips");

                  // Sign handling: leading '+', bare negatives, and zero forms.
                  t.expect_eq(int128_to_string(i128("-42")), std::string("-42"), "negative renders with sign");
                  t.expect_eq(int128_to_string(i128("+7")), std::string("7"), "leading plus is accepted");
                  t.expect_eq(int128_to_string(i128("-0")), std::string("0"), "negative zero is zero");
                  t.expect_eq(int128_to_string(i128("007")), std::string("7"), "leading zeros parse");

                  // Malformed input is syntax_error, not a wrapped or partial value.
                  t.expect(int128_from_string("").error() == MathError::syntax_error, "empty is syntax_error");
                  t.expect(int128_from_string("-").error() == MathError::syntax_error, "lone sign is syntax_error");
                  t.expect(int128_from_string("12a").error() == MathError::syntax_error, "trailing garbage is syntax_error");
                  t.expect(int128_from_string("1.5").error() == MathError::syntax_error, "decimal point is syntax_error");
                  t.expect(int128_from_string(" 3").error() == MathError::syntax_error, "leading space is syntax_error");
              })
        .test("int128_parse_overflow",
              [](TestContext& t) {
                  // One past each boundary must report overflow rather than wrap.
                  t.expect(int128_from_string("170141183460469231731687303715884105728").error() ==
                               MathError::overflow,
                           "INT128_MAX + 1 overflows");
                  t.expect(int128_from_string("-170141183460469231731687303715884105729").error() ==
                               MathError::overflow,
                           "INT128_MIN - 1 overflows");
                  t.expect(int128_from_string("999999999999999999999999999999999999999999").error() ==
                               MathError::overflow,
                           "a 42-digit value overflows");
              })
        .test("checked_arithmetic_boundary",
              [](TestContext& t) {
                  // A product that fits: 10^18 * 10^18 = 10^36 (well under ~1.7e38).
                  const Int128 e18 = i128("1000000000000000000");
                  auto prod = checked_mul(e18, e18);
                  t.expect(prod.has_value(), "10^18 * 10^18 fits in 128 bits");
                  t.expect_eq(int128_to_string(prod.value()),
                              std::string("1000000000000000000000000000000000000"), "product is 10^36");

                  // A product that does not fit: MAX * 2 overflows the 128-bit ceiling.
                  const Int128 mx = i128(kMax);
                  t.expect(checked_mul(mx, 2).error() == MathError::overflow, "MAX * 2 overflows");
                  t.expect(checked_add(mx, 1).error() == MathError::overflow, "MAX + 1 overflows");
                  t.expect(checked_sub(i128(kMin), 1).error() == MathError::overflow, "MIN - 1 overflows");

                  // Non-overflowing checked ops return the exact value.
                  t.expect_eq(checked_add(e18, e18).value(), i128("2000000000000000000"), "10^18 + 10^18 = 2*10^18");
              })
        .test("rational128_canonicalisation",
              [](TestContext& t) {
                  t.expect(r128(2, 4) == r128(1, 2), "2/4 reduces to 1/2");
                  t.expect(r128(2, -4) == r128(-1, 2), "sign moves to the numerator");
                  t.expect(r128(-1, -2) == r128(1, 2), "-1/-2 reduces to 1/2");
                  t.expect(r128(0, 5) == Rational128{}, "0/5 is the zero rational");
                  t.expect(Rational128::from_int(7).is_integer(), "7 is integral");
                  t.expect(r128(2, -4).denominator() > 0, "denominator kept positive");
                  t.expect(!Rational128::make(1, 0).has_value(), "1/0 is division_by_zero");
                  t.expect(Rational128::make(1, 0).error() == MathError::division_by_zero, "1/0 error is division_by_zero");
              })
        .test("rational128_exact_beyond_int64",
              [](TestContext& t) {
                  // (10^18 / 1) + (1 / 10^18) = (10^36 + 1) / 10^18. The denominator
                  // 10^18 already exceeds sqrt(INT64_MAX)-scale cross products, and the
                  // numerator 10^36 + 1 is far past what an int64 Rational could hold.
                  const Int128 e18 = i128("1000000000000000000");
                  auto a = r128(e18, 1);
                  auto b = r128(1, e18);
                  auto sum = a.add(b).value();
                  t.expect_eq(int128_to_string(sum.numerator()),
                              std::string("1000000000000000000000000000000000001"),
                              "numerator is 10^36 + 1");
                  t.expect_eq(int128_to_string(sum.denominator()),
                              std::string("1000000000000000000"), "denominator is 10^18");

                  // Multiplying the two big fractions collapses to exactly 1.
                  t.expect(a.multiply(b).value() == Rational128::from_int(1), "(10^18)(1/10^18) = 1");
              })
        .test("rational128_arithmetic_laws",
              [](TestContext& t) {
                  // 1/2 + 1/3 = 5/6
                  t.expect(r128(1, 2).add(r128(1, 3)).value() == r128(5, 6), "1/2 + 1/3 = 5/6");
                  // 1/2 - 1/3 = 1/6
                  t.expect(r128(1, 2).subtract(r128(1, 3)).value() == r128(1, 6), "1/2 - 1/3 = 1/6");
                  // 2/3 * 3/4 = 1/2
                  t.expect(r128(2, 3).multiply(r128(3, 4)).value() == r128(1, 2), "2/3 * 3/4 = 1/2");
                  // (2/3) / (4/9) = 3/2
                  t.expect(r128(2, 3).divide(r128(4, 9)).value() == r128(3, 2), "(2/3)/(4/9) = 3/2");
                  // reciprocal(3/7) = 7/3, and x * (1/x) = 1
                  t.expect(r128(3, 7).reciprocal().value() == r128(7, 3), "reciprocal(3/7) = 7/3");
                  auto x = r128(-5, 8);
                  t.expect(x.multiply(x.reciprocal().value()).value() == Rational128::from_int(1),
                           "x * reciprocal(x) = 1");
                  // negate keeps canonical form
                  t.expect(r128(3, 4).negate().value() == r128(-3, 4), "negate(3/4) = -3/4");
              })
        .test("rational128_error_paths",
              [](TestContext& t) {
                  t.expect(r128(1, 2).divide(Rational128{}).error() == MathError::division_by_zero,
                           "divide by zero fails");
                  t.expect(Rational128{}.reciprocal().error() == MathError::division_by_zero,
                           "reciprocal of zero fails");

                  // Overflow past 128 bits is reported, not wrapped: MAX * 2.
                  auto max_rat = r128(i128(kMax), 1);
                  t.expect(max_rat.multiply(Rational128::from_int(2)).error() == MathError::overflow,
                           "128-bit overflow in multiply is reported");
                  // Adding fractions whose common denominator overflows also surfaces overflow.
                  auto big = r128(i128(kMax), i128("1000000000000000000"));
                  t.expect(big.add(big).error() == MathError::overflow,
                           "overflow in the cross-multiplied denominator is reported");
              })
        .test("cross_check_with_int64_rational",
              [](TestContext& t) {
                  // For values both tiers can represent, Rational128 must agree with the
                  // int64 Rational field operations coefficient-for-coefficient.
                  auto check = [&t](const char* what, Rational128 got, Rational want) {
                      const bool eq = got.numerator() == static_cast<Int128>(want.numerator()) &&
                                      got.denominator() == static_cast<Int128>(want.denominator());
                      t.expect(eq, what);
                  };
                  check("1/2 + 1/3 agrees",
                        r128(1, 2).add(r128(1, 3)).value(),
                        Rational::make(1, 2).value().add(Rational::make(1, 3).value()).value());
                  check("2/3 * 3/4 agrees",
                        r128(2, 3).multiply(r128(3, 4)).value(),
                        Rational::make(2, 3).value().multiply(Rational::make(3, 4).value()).value());
                  check("(2/3)/(4/9) agrees",
                        r128(2, 3).divide(r128(4, 9)).value(),
                        Rational::make(2, 3).value().divide(Rational::make(4, 9).value()).value());
                  check("(7/9) - (1/6) agrees",
                        r128(7, 9).subtract(r128(1, 6)).value(),
                        Rational::make(7, 9).value().subtract(Rational::make(1, 6).value()).value());
              })
        .run();
}
