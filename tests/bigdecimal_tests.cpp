// Tests for nimblecas.bigdecimal: exact base-10 scaled decimal (money type + quantizer).
// @author Olumuyiwa Oluwasanmi
//
// The suite proves the three things BigDecimal exists to guarantee and BigRational cannot:
// (1) scale is SEMANTIC and preserved (2.50 != 2.5 by representation, == by value);
// (2) decimal literals are EXACT — the base-2 traps (0.1 + 0.2 != 0.3, round(2.675, 2))
//     that defeat double/BigFloat are handled correctly here; (3) division without a
//     rounding mode REFUSES (divide_exact -> inexact) rather than guessing, and every
//     rounding mode rounds exactly as specified. Every expected value is hand-computed.

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.bigdecimal;
import nimblecas.testing;

using nimblecas::BigDecimal;
using nimblecas::BigInt;
using nimblecas::BigRational;
using nimblecas::MathError;
using nimblecas::Rounding;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto bd(std::string_view s) -> BigDecimal { return BigDecimal::from_string(s).value(); }

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.bigdecimal")
        .test("parse and render preserve scale",
              [](TestContext& t) {
                  t.expect_eq(bd("1.50").to_string(), std::string{"1.50"}, "1.50 round-trips");
                  t.expect_eq(bd("1.50").scale(), 2, "1.50 has scale 2");
                  t.expect_eq(bd("1.50").unscaled(), BigInt::from_i64(150), "1.50 unscaled 150");
                  t.expect_eq(bd("123").to_string(), std::string{"123"}, "integer, scale 0");
                  t.expect_eq(bd("-0.007").to_string(), std::string{"-0.007"}, "small negative");
                  t.expect_eq(bd("-0.007").scale(), 3, "-0.007 scale 3");
                  t.expect_eq(bd("0.00").to_string(), std::string{"0.00"}, "signed-zero-free zero");
                  // Scientific notation folds into the scale: 1.5e3 == 1500, scale -2.
                  t.expect_eq(bd("1.5e3").to_string(), std::string{"1500"}, "1.5e3 -> 1500");
                  t.expect_eq(bd("2.5E-2").to_string(), std::string{"0.025"}, "2.5E-2 -> 0.025");
                  // Malformed input is rejected, not guessed.
                  t.expect(!BigDecimal::from_string("1.2.3").has_value(), "double dot -> error");
                  t.expect(!BigDecimal::from_string("abc").has_value(), "letters -> error");
                  t.expect(!BigDecimal::from_string("").has_value(), "empty -> error");
              })
        .test("numeric vs representation equality",
              [](TestContext& t) {
                  t.expect(bd("2.50") == bd("2.5"), "2.50 == 2.5 numerically");
                  t.expect(!bd("2.50").same_representation(bd("2.5")),
                           "2.50 and 2.5 differ in representation");
                  t.expect(bd("2.50").same_representation(bd("2.50")), "identical reps match");
                  t.expect(bd("0.1") < bd("0.10001"), "0.1 < 0.10001");
                  t.expect(bd("-1.0") < bd("-0.5"), "-1.0 < -0.5");
              })
        .test("exact decimal arithmetic (the base-2 traps)",
              [](TestContext& t) {
                  // 0.1 + 0.2 == 0.3 EXACTLY (fails in binary floating point).
                  t.expect(bd("0.1").add(bd("0.2")) == bd("0.3"), "0.1 + 0.2 == 0.3 exactly");
                  t.expect_eq(bd("0.1").add(bd("0.2")).to_string(), std::string{"0.3"},
                              "0.1 + 0.2 renders 0.3");
                  // Alignment picks the larger scale.
                  t.expect_eq(bd("1.50").add(bd("2.5")).to_string(), std::string{"4.00"},
                              "1.50 + 2.5 -> 4.00 (scale 2)");
                  t.expect_eq(bd("0.30").subtract(bd("0.10")).to_string(), std::string{"0.20"},
                              "0.30 - 0.10 -> 0.20 exact");
                  // Multiply adds scales.
                  t.expect_eq(bd("1.5").multiply(bd("1.5")).value().to_string(),
                              std::string{"2.25"}, "1.5 * 1.5 -> 2.25 (scale 2)");
                  t.expect_eq(bd("2").pow(10).value().to_string(), std::string{"1024"},
                              "2^10 == 1024");
              })
        .test("rounding modes are exact and explicit",
              [](TestContext& t) {
                  // The classic 2.675 -> 2.68 that binary double rounds to 2.67.
                  t.expect_eq(bd("2.675").quantize(2, Rounding::half_even).to_string(),
                              std::string{"2.68"}, "2.675 half_even -> 2.68");
                  t.expect_eq(bd("2.675").quantize(2, Rounding::down).to_string(),
                              std::string{"2.67"}, "2.675 down -> 2.67");
                  // Banker's rounding ties to even.
                  t.expect_eq(bd("2.5").quantize(0, Rounding::half_even).to_string(),
                              std::string{"2"}, "2.5 half_even -> 2 (even)");
                  t.expect_eq(bd("3.5").quantize(0, Rounding::half_even).to_string(),
                              std::string{"4"}, "3.5 half_even -> 4 (even)");
                  t.expect_eq(bd("2.5").quantize(0, Rounding::half_up).to_string(),
                              std::string{"3"}, "2.5 half_up -> 3");
                  // Directed rounding respects sign.
                  t.expect_eq(bd("-2.5").quantize(0, Rounding::floor).to_string(),
                              std::string{"-3"}, "-2.5 floor -> -3");
                  t.expect_eq(bd("-2.5").quantize(0, Rounding::ceiling).to_string(),
                              std::string{"-2"}, "-2.5 ceiling -> -2");
                  // Widening is exact and appends zeros.
                  t.expect_eq(bd("1.5").quantize(4, Rounding::down).to_string(),
                              std::string{"1.5000"}, "widen 1.5 -> 1.5000");
              })
        .test("division: exact-or-refuse, and controlled-scale rounding",
              [](TestContext& t) {
                  // Terminating quotient: exact.
                  t.expect_eq(bd("1").divide_exact(bd("8")).value().to_string(),
                              std::string{"0.125"}, "1/8 == 0.125 exactly");
                  t.expect_eq(bd("1").divide_exact(bd("4")).value().to_string(),
                              std::string{"0.25"}, "1/4 == 0.25 exactly");
                  // Non-terminating quotient: REFUSE with inexact, never a silent round.
                  t.expect(bd("1").divide_exact(bd("3")).error() == MathError::inexact,
                           "1/3 divide_exact -> inexact");
                  t.expect(bd("1").divide_exact(bd("0")).error() == MathError::division_by_zero,
                           "1/0 -> division_by_zero");
                  // With an explicit scale + mode, rounding is exact.
                  t.expect_eq(bd("1").divide(bd("3"), 4, Rounding::half_even).value().to_string(),
                              std::string{"0.3333"}, "1/3 @4 half_even -> 0.3333");
                  t.expect_eq(bd("2").divide(bd("3"), 4, Rounding::half_even).value().to_string(),
                              std::string{"0.6667"}, "2/3 @4 half_even -> 0.6667");
              })
        .test("conversions to/from BigRational and double",
              [](TestContext& t) {
                  t.expect(bd("1.50").to_bigrational() == BigRational::from_string("3/2").value(),
                           "1.50 -> 3/2");
                  t.expect(BigDecimal::from_bigrational_exact(
                               BigRational::from_string("3/8").value())
                               .value() == bd("0.375"),
                           "3/8 -> 0.375 exact");
                  t.expect(BigDecimal::from_bigrational_exact(
                               BigRational::from_string("1/3").value())
                               .error() == MathError::inexact,
                           "1/3 -> inexact (non-terminating)");
                  // from_double quantizes on entry: 0.1 -> 0.10 (no binary artifact leaks).
                  t.expect_eq(BigDecimal::from_double(0.1, 2, Rounding::half_even).value().to_string(),
                              std::string{"0.10"}, "from_double(0.1)@2 -> 0.10");
                  t.expect_eq(BigDecimal::from_bigrational(
                                  BigRational::from_string("22/7").value(), 5, Rounding::half_even)
                                  .to_string(),
                              std::string{"3.14286"}, "22/7 @5 -> 3.14286");
              })
        .run();
}
