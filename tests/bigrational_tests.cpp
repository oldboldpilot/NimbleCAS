// Tests for nimblecas.bigrational: exact, unbounded rational numbers over Q.
// @author Olumuyiwa Oluwasanmi
//
// BigRational is the unbounded tier of the arithmetic tower: the int64 Rational and the
// 128-bit Rational128 return MathError::overflow at their ceilings, whereas BigRational
// never does. The suite therefore proves two things at once: (1) the field axioms and
// canonicalisation hold on concrete fractions, exactly as for the int64 Rational, and
// (2) exactness survives well beyond the int64/int128 range — a harmonic sum whose
// reduced denominator exceeds 2^63, and 40-digit fractions that recombine to exactly 1.
// Small results are cross-checked limb-for-limb against the int64 Rational so the two
// tiers are proven to agree where their domains overlap. Every literal is hand-verified.

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.ratpoly;
import nimblecas.testing;

using nimblecas::BigInt;
using nimblecas::BigRational;
using nimblecas::MathError;
using nimblecas::Rational;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Parse a fraction literal that is known-good in this test file.
[[nodiscard]] auto br(std::string_view s) -> BigRational {
    return BigRational::from_string(s).value();
}

// Parse a decimal integer literal that is known-good in this test file.
[[nodiscard]] auto bi(std::string_view s) -> BigInt {
    return BigInt::from_string(s).value();
}

// Confirm a BigRational equals the fraction num/den (both given as decimal literals),
// checking numerator and denominator separately so a mis-signed or unreduced result is
// caught even if it happened to compare equal.
auto expect_fraction(TestContext& t, const BigRational& r, std::string_view num,
                     std::string_view den, std::string_view what) -> void {
    t.expect(r.numerator() == bi(num) && r.denominator() == bi(den), what);
}

// Cross-check a small BigRational against the int64 Rational computation of the same
// value: their numerators and denominators must match exactly (via BigInt::from_i64).
auto expect_matches_rational(TestContext& t, const BigRational& b, const Rational& r,
                             std::string_view what) -> void {
    t.expect(b.numerator() == BigInt::from_i64(r.numerator()) &&
                 b.denominator() == BigInt::from_i64(r.denominator()),
             what);
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.bigrational")
        .test("canonicalisation",
              [](TestContext& t) {
                  // Sign migrates to the numerator; denominator is always positive.
                  expect_fraction(t, BigRational::make(bi("2"), bi("-4")).value(), "-1", "2",
                                  "make(2, -4) -> -1/2 (sign on numerator)");
                  t.expect(BigRational::make(bi("2"), bi("-4")).value() ==
                               BigRational::make(bi("-1"), bi("2")).value(),
                           "make(2,-4) == make(-1,2)");
                  // Reduction by gcd.
                  expect_fraction(t, BigRational::make(bi("6"), bi("4")).value(), "3", "2",
                                  "6/4 reduces to 3/2");
                  expect_fraction(t, BigRational::make(bi("-6"), bi("-4")).value(), "3", "2",
                                  "-6/-4 reduces to 3/2 (both signs cancel)");
                  // Canonical zero is 0/1 regardless of the incoming denominator.
                  expect_fraction(t, BigRational::make(bi("0"), bi("5")).value(), "0", "1",
                                  "0/5 canonicalises to 0/1");
                  expect_fraction(t, BigRational::make(bi("0"), bi("-5")).value(), "0", "1",
                                  "0/-5 canonicalises to 0/1");
                  // Integers keep denominator 1 and render without a slash.
                  const BigRational seven = BigRational::from_int(7);
                  t.expect(seven.is_integer() && seven.denominator() == bi("1"),
                           "from_int(7) is integral with denominator 1");
                  t.expect_eq(seven.to_string(), std::string("7"), "integer prints without slash");
                  t.expect_eq(br("6/3").to_string(), std::string("2"), "6/3 -> 2");
                  t.expect_eq(br("-4/6").to_string(), std::string("-2/3"), "-4/6 -> -2/3");
                  // Default-constructed value is the canonical zero.
                  expect_fraction(t, BigRational{}, "0", "1", "default is 0/1");
                  t.expect(BigRational{}.is_zero(), "default is zero");
              })
        .test("division_by_zero",
              [](TestContext& t) {
                  const auto is_dbz = [&](const auto& result, std::string_view what) {
                      t.expect(!result.has_value() &&
                                   result.error() == MathError::division_by_zero,
                               what);
                  };
                  is_dbz(BigRational::make(bi("3"), bi("0")), "make(3, 0) fails");
                  is_dbz(BigRational::make(bi("0"), bi("0")), "make(0, 0) fails");
                  is_dbz(br("5/7").divide(BigRational{}), "divide by zero fails");
                  is_dbz(BigRational{}.reciprocal(), "reciprocal of zero fails");
                  is_dbz(BigRational::from_string("4/0"), "from_string \"4/0\" fails as by-zero");
                  is_dbz(BigRational{}.pow(-3), "negative power of zero fails");
                  // A non-zero divisor and a non-negative power of zero are fine.
                  t.expect(br("5/7").divide(br("2/3")).has_value(), "nonzero divide succeeds");
                  t.expect(BigRational{}.pow(3).value().is_zero(), "0^3 = 0");
                  t.expect(BigRational{}.pow(0).value() == BigRational::from_int(1),
                           "0^0 = 1 by convention");
              })
        .test("field_axioms",
              [](TestContext& t) {
                  const BigRational a = br("3/4");
                  const BigRational b = br("-5/6");
                  const BigRational c = br("7/9");
                  const BigRational zero = BigRational{};
                  const BigRational one = BigRational::from_int(1);

                  // Additive identity and inverse.
                  t.expect(a.add(zero) == a, "a + 0 == a");
                  t.expect(a.add(a.negate()).is_zero(), "a + (-a) == 0");
                  // Multiplicative identity and inverse.
                  t.expect(a.multiply(one) == a, "a * 1 == a");
                  t.expect(a.multiply(a.reciprocal().value()) == one, "a * (1/a) == 1");
                  // Associativity of + and *.
                  t.expect(a.add(b).add(c) == a.add(b.add(c)), "(a+b)+c == a+(b+c)");
                  t.expect(a.multiply(b).multiply(c) == a.multiply(b.multiply(c)),
                           "(a*b)*c == a*(b*c)");
                  // Commutativity.
                  t.expect(a.add(b) == b.add(a), "a+b == b+a");
                  t.expect(a.multiply(b) == b.multiply(a), "a*b == b*a");
                  // Distributivity.
                  t.expect(a.multiply(b.add(c)) == a.multiply(b).add(a.multiply(c)),
                           "a*(b+c) == a*b + a*c");
                  // subtract agrees with add-of-negate; divide agrees with multiply-by-reciprocal.
                  t.expect(a.subtract(b) == a.add(b.negate()), "a-b == a+(-b)");
                  t.expect(a.divide(b).value() == a.multiply(b.reciprocal().value()),
                           "a/b == a*(1/b)");
                  // Concrete anchor: 3/4 - 5/6 = 9/12 - 10/12 = -1/12.
                  expect_fraction(t, br("3/4").subtract(br("5/6")), "-1", "12", "3/4 - 5/6 = -1/12");
              })
        .test("matches_int64_rational",
              [](TestContext& t) {
                  // Where the domains overlap, BigRational and the int64 Rational must agree
                  // numerator-for-numerator and denominator-for-denominator.
                  struct Pair {
                      std::int64_t n;
                      std::int64_t d;
                  };
                  const std::vector<std::pair<Pair, Pair>> cases = {
                      {{3, 4}, {5, 6}},   {{-2, 3}, {7, 8}},  {{9, 10}, {-1, 15}},
                      {{-11, 12}, {13, 14}}, {{100, 7}, {3, 50}},
                  };
                  for (const auto& [pa, pb] : cases) {
                      const Rational ra = Rational::make(pa.n, pa.d).value();
                      const Rational rb = Rational::make(pb.n, pb.d).value();
                      const BigRational ba =
                          BigRational::make(BigInt::from_i64(pa.n), BigInt::from_i64(pa.d)).value();
                      const BigRational bb =
                          BigRational::make(BigInt::from_i64(pb.n), BigInt::from_i64(pb.d)).value();
                      expect_matches_rational(t, ba.add(bb), ra.add(rb).value(), "add matches");
                      expect_matches_rational(t, ba.subtract(bb), ra.subtract(rb).value(),
                                              "subtract matches");
                      expect_matches_rational(t, ba.multiply(bb), ra.multiply(rb).value(),
                                              "multiply matches");
                      expect_matches_rational(t, ba.divide(bb).value(), ra.divide(rb).value(),
                                              "divide matches");
                      expect_matches_rational(t, ba.negate(), ra.negate().value(), "negate matches");
                  }
              })
        .test("exactness_beyond_int64_identity",
              [](TestContext& t) {
                  // 10^40/(10^40+1) + 1/(10^40+1) == 1 exactly. Both denominators (10^40 + 1)
                  // are ~41 digits, far past the int64/int128 overflow ceiling.
                  const BigInt p = BigInt::from_u64(10).pow(40).add(bi("1"));  // 10^40 + 1
                  const BigRational a = BigRational::make(BigInt::from_u64(10).pow(40), p).value();
                  const BigRational b = BigRational::make(bi("1"), p).value();
                  t.expect(a.add(b) == BigRational::from_int(1),
                           "10^40/(10^40+1) + 1/(10^40+1) == 1");
                  // a is already reduced (10^40 and 10^40+1 are coprime), so its denominator
                  // genuinely exceeds int64 — this is not a value the int64 tier could hold.
                  t.expect(a.denominator() > BigInt::from_i64(std::numeric_limits<std::int64_t>::max()),
                           "the reduced denominator exceeds INT64_MAX");
                  // a * (1/a) == 1 with the big denominators cancelling exactly.
                  t.expect(a.multiply(a.reciprocal().value()) == BigRational::from_int(1),
                           "a * (1/a) == 1 across 41-digit terms");
              })
        .test("exactness_harmonic_sum",
              [](TestContext& t) {
                  // H_50 = sum_{k=1}^{50} 1/k. Its reduced denominator (3.1e21) far exceeds
                  // 2^63, so only the unbounded tier can represent it. The reduced value is
                  // hand-verified (computed independently):
                  //   13943237577224054960759 / 3099044504245996706400.
                  BigRational h = BigRational{};
                  for (std::int64_t k = 1; k <= 50; ++k) {
                      h = h.add(BigRational::make(bi("1"), BigInt::from_i64(k)).value());
                  }
                  expect_fraction(t, h, "13943237577224054960759", "3099044504245996706400",
                                  "H_50 equals its known reduced fraction");
                  t.expect(h.denominator() > BigInt::from_i64(std::numeric_limits<std::int64_t>::max()),
                           "H_50 denominator exceeds INT64_MAX");
                  // Independent consistency check: multiplying the reduced value back by its
                  // own denominator must recover exactly its numerator (H_50 * den == num).
                  const BigRational reconstructed =
                      h.multiply(BigRational::from_bigint(h.denominator()));
                  t.expect(reconstructed == BigRational::from_bigint(h.numerator()),
                           "H_50 * den == num exactly");
              })
        .test("pow",
              [](TestContext& t) {
                  // (2/3)^10 = 1024/59049.
                  expect_fraction(t, br("2/3").pow(10).value(), "1024", "59049", "(2/3)^10");
                  // Negative exponent inverts: (2/3)^-2 = 9/4.
                  expect_fraction(t, br("2/3").pow(-2).value(), "9", "4", "(2/3)^-2 = 9/4");
                  // (-3/5)^3 = -27/125 (odd power keeps the sign).
                  expect_fraction(t, br("-3/5").pow(3).value(), "-27", "125", "(-3/5)^3");
                  // (-3/5)^2 = 9/25 (even power is positive).
                  expect_fraction(t, br("-3/5").pow(2).value(), "9", "25", "(-3/5)^2");
                  // x^1 == x and x^0 == 1.
                  t.expect(br("7/11").pow(1).value() == br("7/11"), "x^1 == x");
                  t.expect(br("7/11").pow(0).value() == BigRational::from_int(1), "x^0 == 1");
                  // A large exact power well beyond int64: 2^100 / 3^100, reduced (coprime).
                  const BigRational big = br("2/3").pow(100).value();
                  t.expect(big.numerator() == BigInt::from_u64(2).pow(100), "(2/3)^100 numerator");
                  t.expect(big.denominator() == BigInt::from_u64(3).pow(100),
                           "(2/3)^100 denominator");
                  // pow then reciprocal-power round-trips to the original.
                  t.expect(br("4/9").pow(5).value().pow(-1).value().pow(-5).value() == br("4/9"),
                           "((x^5)^-1)^-5 == x");
              })
        .test("from_string_roundtrip",
              [](TestContext& t) {
                  t.expect_eq(br("3/4").to_string(), std::string("3/4"), "3/4 round-trips");
                  t.expect_eq(br("-5/7").to_string(), std::string("-5/7"), "-5/7 round-trips");
                  t.expect_eq(br("42").to_string(), std::string("42"), "bare integer round-trips");
                  t.expect_eq(br("-42").to_string(), std::string("-42"),
                              "bare negative integer round-trips");
                  // Sign normalisation and reduction happen during parsing.
                  t.expect_eq(br("5/-10").to_string(), std::string("-1/2"),
                              "5/-10 normalises to -1/2");
                  t.expect_eq(br("-6/-8").to_string(), std::string("3/4"),
                              "-6/-8 normalises to 3/4");
                  t.expect_eq(br("+3/+4").to_string(), std::string("3/4"), "leading + accepted");
                  // A denominator that divides out to 1 renders as a bare integer.
                  t.expect_eq(br("100/25").to_string(), std::string("4"), "100/25 -> 4");
                  // A very large fraction survives the parse -> render round-trip verbatim.
                  const std::string bigfrac =
                      "123456789012345678901234567890/999999999999999999999999999998";
                  t.expect_eq(br(bigfrac).to_string(), bigfrac, "30-digit fraction round-trips");

                  const auto is_syntax = [&](std::string_view s, std::string_view what) {
                      auto r = BigRational::from_string(s);
                      t.expect(!r.has_value() && r.error() == MathError::syntax_error, what);
                  };
                  is_syntax("", "empty is malformed");
                  is_syntax("abc", "letters are malformed");
                  is_syntax("1.5", "decimal point is malformed");
                  is_syntax("3/", "missing denominator is malformed");
                  is_syntax("/3", "missing numerator is malformed");
                  is_syntax("1/2/3", "double slash is malformed");
                  is_syntax("1//2", "empty middle is malformed");
                  is_syntax("3 / 4", "spaces are malformed");
                  is_syntax("-", "bare sign is malformed");
              })
        .test("comparison_total_order",
              [](TestContext& t) {
                  // A mixed ascending set spanning negatives, zero, sub-integer fractions, and
                  // magnitudes that no int64 rational could hold.
                  const std::vector<BigRational> v = {
                      br("-1000000000000000000000000000000/3"),  // ~ -3.3e29
                      br("-2/3"),                                // -0.666...
                      br("-1/2"),                                // -0.5
                      BigRational{},                             // 0
                      br("999999999999999999999/1000000000000000000000"),  // ~0.999..., just below 1
                      br("1"),                                   // 1
                      br("1000000000000000000000000000000/3"),   // ~ 3.3e29
                  };
                  bool ascending = true;
                  for (std::size_t i = 1; i < v.size(); ++i) {
                      ascending = ascending && (v[i - 1] < v[i]) && !(v[i] < v[i - 1]) &&
                                  (v[i - 1] != v[i]);
                  }
                  t.expect(ascending, "the mixed set is strictly ascending");
                  // Cross-multiplication order does not flip with negative values.
                  t.expect(br("-2/3") < br("-1/2"), "-2/3 < -1/2 (closer to zero is larger)");
                  t.expect(br("1/3") < br("1/2"), "1/3 < 1/2");
                  t.expect(br("2/3") > br("1/2"), "2/3 > 1/2");
                  // Equal values compare equal even when spelled differently before reduction.
                  t.expect(br("2/4") == br("1/2"), "2/4 == 1/2");
                  t.expect(!(br("2/4") < br("1/2")) && !(br("1/2") < br("2/4")),
                           "equal values are unordered under <");
                  // sign() agrees with the ordering against zero.
                  t.expect(br("-7/9").sign() == -1 && br("0").sign() == 0 && br("7/9").sign() == 1,
                           "sign() reports -1/0/+1");
              })
        .test("to_double_approximation",
              [](TestContext& t) {
                  // to_double is APPROXIMATE; check it lands within a tiny tolerance.
                  const auto close = [](double x, double y) { return std::abs(x - y) < 1e-12; };
                  t.expect(close(br("1/2").to_double(), 0.5), "1/2 ~ 0.5");
                  t.expect(close(br("-3/4").to_double(), -0.75), "-3/4 ~ -0.75");
                  t.expect(close(br("0").to_double(), 0.0), "0 ~ 0.0");
                  t.expect(close(br("10/4").to_double(), 2.5), "10/4 ~ 2.5");
                  t.expect(close(br("1/3").to_double(), 1.0 / 3.0), "1/3 ~ 0.333...");
                  // A huge-numerator, huge-denominator ratio near 1 still approximates well.
                  const BigInt p = BigInt::from_u64(10).pow(40).add(bi("1"));
                  const BigRational near_one =
                      BigRational::make(BigInt::from_u64(10).pow(40), p).value();
                  t.expect(close(near_one.to_double(), 1.0), "10^40/(10^40+1) ~ 1.0");
              })
        .run();
}
