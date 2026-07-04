// Tests for nimblecas.bigfloat: software arbitrary-precision binary floating point.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigfloat;
import nimblecas.testing;

using nimblecas::BigFloat;
using nimblecas::BigInt;
using nimblecas::MathError;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Terse constructors that unwrap the (here always-succeeding) Result.
[[nodiscard]] auto bf(std::int64_t v, std::int64_t prec) -> BigFloat {
    return BigFloat::from_i64(v, prec).value();
}
[[nodiscard]] auto bfd(double d, std::int64_t prec) -> BigFloat {
    return BigFloat::from_double(d, prec).value();
}
[[nodiscard]] auto bfs(std::string_view s, std::int64_t prec) -> BigFloat {
    return BigFloat::from_string(s, prec).value();
}

// Two values agree to within a handful of ULP if dropping the low `drop` bits makes their
// canonical forms identical (the difference is far below the reduced ULP).
[[nodiscard]] auto close(const BigFloat& a, const BigFloat& b, std::int64_t prec,
                         std::int64_t drop) -> bool {
    return a.with_precision(prec - drop).value() == b.with_precision(prec - drop).value();
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.bigfloat")
        .test("dyadic_values_are_exact",
              [](TestContext& t) {
                  // 0.5, 0.75, 3.25 are dyadic, hence represented with no rounding at all.
                  t.expect(bfd(0.5, 53).to_double() == 0.5, "0.5 is exact");
                  t.expect(bfd(0.75, 53).to_double() == 0.75, "0.75 is exact");
                  t.expect(bfd(3.25, 53).to_double() == 3.25, "3.25 is exact");
                  t.expect(bfs("3.25", 64) == bfd(3.25, 53), "decimal 3.25 == double 3.25");
                  t.expect(bfs("0.5", 64) == bfs(".5", 64), "\"0.5\" and \".5\" agree");
                  t.expect(bf(0, 64).is_zero(), "integer 0 is the canonical zero");
                  t.expect(bf(0, 64).sign() == 0 && bf(-7, 64).sign() == -1, "sign() is correct");
              })
        .test("double_round_trips",
              [](TestContext& t) {
                  // Exact doubles survive from_double -> to_double unchanged.
                  for (double d : {0.0, 1.0, -2.5, 0.125, 12345.75, -0.03125}) {
                      t.expect(bfd(d, 53).to_double() == d, "double round-trips exactly");
                  }
                  // to_string reflects the stored dyadic value precisely.
                  t.expect(bfd(3.25, 53).to_string(2) == "3.25", "3.25 -> \"3.25\"");
                  t.expect(bf(-125, 64).to_string(0) == "-125", "-125 -> \"-125\"");
              })
        .test("one_third_precision_and_convergence",
              [](TestContext& t) {
                  const auto third128 = bf(1, 128).divide(bf(3, 128), 128).value();
                  t.expect(third128.to_string(20) == "0.33333333333333333333",
                           "1/3 at 128 bits shows twenty 3s");
                  // Lower precision still agrees on its (shorter) leading digits...
                  const auto third32 = bf(1, 32).divide(bf(3, 32), 32).value();
                  t.expect(third32.to_string(6) == "0.333333", "1/3 at 32 bits shows six 3s");
                  // ...and raising precision converges (30 correct digits at 200 bits).
                  const auto third200 = bf(1, 200).divide(bf(3, 200), 200).value();
                  t.expect(third200.to_string(30) == "0.333333333333333333333333333333",
                           "1/3 at 200 bits shows thirty 3s");
              })
        .test("sqrt_two_high_precision",
              [](TestContext& t) {
                  const auto root2 = bf(2, 200).sqrt(200).value();
                  const std::string s = root2.to_string(45);
                  // Compare the leading "1." + 30 fractional digits against the known value
                  // 1.41421356237309504880168872420969807...  (printing 45 keeps these exact).
                  t.expect(s.substr(0, 32) == "1.414213562373095048801688724209",
                           "sqrt(2) matches its first 30 digits");
                  t.expect(root2.multiply(root2, 200).value().with_precision(190).value() ==
                               bf(2, 200).with_precision(190).value(),
                           "sqrt(2)^2 rounds back to 2");
              })
        .test("sqrt_of_square_recovers_value",
              [](TestContext& t) {
                  // a with a small mantissa: a*a is exact, so sqrt(a*a) == a exactly.
                  const auto a = bfs("3.25", 64);
                  const auto sq = a.multiply(a, 64).value();
                  t.expect(sq.sqrt(64).value() == a, "sqrt(3.25^2) == 3.25");
                  const auto b = bf(9, 64);
                  t.expect(b.sqrt(64).value() == bf(3, 64), "sqrt(9) == 3");
              })
        .test("associativity_within_one_ulp",
              [](TestContext& t) {
                  const std::int64_t p = 200;
                  const auto a = bf(1, p).divide(bf(3, p), p).value();
                  const auto b = bf(1, p).divide(bf(7, p), p).value();
                  const auto c = bf(1, p).divide(bf(11, p), p).value();
                  const auto left = a.add(b, p).value().add(c, p).value();
                  const auto right = a.add(b.add(c, p).value(), p).value();
                  // The two groupings differ by at most ~2 ulp at 200 bits (~1e-60), so their
                  // leading 45 decimal digits are identical while the roundings may not be.
                  const std::string ls = left.to_string(55);
                  const std::string rs = right.to_string(55);
                  t.expect(ls.substr(0, 47) == rs.substr(0, 47),
                           "(a+b)+c and a+(b+c) agree to 45 digits");
              })
        .test("reciprocal_product_is_one",
              [](TestContext& t) {
                  const std::int64_t p = 160;
                  const auto a = bf(7, p);
                  const auto inv = bf(1, p).divide(a, p).value();
                  const auto prod = a.multiply(inv, p).value();
                  t.expect(close(prod, bf(1, p), p, 8), "a * (1/a) ~ 1");
              })
        .test("large_exponent_beyond_double",
              [](TestContext& t) {
                  // 10^400 is far past the double range (~1.8e308) yet representable here.
                  const auto huge = bfs("1e400", 64);
                  t.expect(!huge.is_zero() && huge.sign() == 1, "1e400 is a positive value");
                  t.expect(std::isinf(huge.to_double()), "1e400 overflows to_double() to inf");
                  t.expect(huge > bf(1, 64), "1e400 orders above 1");
                  // A tiny value on the other end is also representable and nonzero.
                  const auto tiny = bfs("1e-400", 64);
                  t.expect(!tiny.is_zero() && tiny.sign() == 1, "1e-400 is a tiny positive value");
              })
        .test("string_round_trip_and_scientific",
              [](TestContext& t) {
                  t.expect(bfs("1.25e2", 64) == bf(125, 64), "1.25e2 == 125");
                  t.expect(bfs("6E-3", 128) == bfs("0.006", 128), "6E-3 == 0.006");
                  t.expect(bfs("-3.5", 64).to_string(1) == "-3.5", "-3.5 round-trips");
                  t.expect(bfs("100", 64).to_string(0) == "100", "100 round-trips");
                  // from_bigint keeps a large integer exact when it fits in the precision.
                  const BigInt big =
                      BigInt::from_string("123456789012345678901234567890").value();
                  t.expect(BigFloat::from_bigint(big, 128).value().to_string(0) ==
                               "123456789012345678901234567890",
                           "from_bigint is exact within precision");
              })
        .test("error_paths",
              [](TestContext& t) {
                  t.expect(bf(1, 64).divide(bf(0, 64), 64).error() == MathError::division_by_zero,
                           "divide by zero -> division_by_zero");
                  t.expect(bf(-1, 64).sqrt(64).error() == MathError::domain_error,
                           "sqrt(-1) -> domain_error");
                  t.expect(BigFloat::from_string("1.2.3", 64).error() == MathError::syntax_error,
                           "malformed literal -> syntax_error");
                  t.expect(BigFloat::from_string("1e", 64).error() == MathError::syntax_error,
                           "dangling exponent -> syntax_error");
                  t.expect(BigFloat::from_i64(5, 0).error() == MathError::domain_error,
                           "non-positive precision -> domain_error");
              })
        .test("batched_reductions",
              [](TestContext& t) {
                  const std::vector<BigFloat> xs{bfd(0.5, 64), bfd(0.25, 64), bfd(0.125, 64)};
                  t.expect(nimblecas::bigfloat_sum(xs, 64).value() == bfd(0.875, 64),
                           "sum(0.5, 0.25, 0.125) == 0.875");
                  const std::vector<BigFloat> a{bf(2, 64), bf(3, 64)};
                  const std::vector<BigFloat> b{bf(4, 64), bf(5, 64)};
                  t.expect(nimblecas::bigfloat_dot(a, b, 64).value() == bf(23, 64),
                           "dot([2,3],[4,5]) == 23");
              })
        .run();
}
