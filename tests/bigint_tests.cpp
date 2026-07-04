// Tests for nimblecas.bigint: exact arbitrary-precision signed integers.
// @author Olumuyiwa Oluwasanmi
//
// The trust anchor for everything built on unbounded integers (crypto, unbounded CAS
// arithmetic). Every decimal literal here is hand-verified; division — the #1 bug risk —
// is checked by the ring law a == q*divisor + r with |r| < |divisor| across single- and
// multi-limb divisors, all four sign combinations, and boundary magnitudes.

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.testing;

using nimblecas::BigInt;
using nimblecas::MathError;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Parse a decimal literal that is known-good in this test file.
[[nodiscard]] auto bi(std::string_view s) -> BigInt {
    return BigInt::from_string(s).value();
}

// The exact value of 100! (158 digits, 24 trailing zeros), written out in full.
constexpr std::string_view kFactorial100 =
    "9332621544394415268169923885626670049071596826438162146859296389521759999322991"
    "5608941463976156518286253697920827223758251185210916864000000000000000000000000";

// Verify the truncated-division ring law for one (a, b) pair with b != 0.
auto check_divmod(TestContext& t, const BigInt& a, const BigInt& b, std::string_view label)
    -> void {
    auto dm = a.divmod(b);
    if (!dm) {
        t.expect(false, label);
        return;
    }
    const BigInt& q = dm->first;
    const BigInt& r = dm->second;
    const BigInt recon = q.multiply(b).add(r);
    t.expect(recon == a, label);                    // a == q*b + r exactly
    t.expect(r.abs() < b.abs(), label);             // |r| < |b|
    if (!r.is_zero()) {
        t.expect(r.is_negative() == a.is_negative(), label);  // remainder takes dividend sign
    }
}

// Build a BigInt from base-2^32 limbs (least-significant first) via exact arithmetic, so
// tests can express the canonical Knuth/Warren division vectors limb-for-limb.
[[nodiscard]] auto from_limbs(std::span<const std::uint32_t> limbs) -> BigInt {
    const BigInt base = BigInt::from_u64(4294967296ULL);  // 2^32
    BigInt acc{};
    for (std::size_t i = limbs.size(); i-- > 0;) {
        acc = acc.multiply(base).add(BigInt::from_u64(limbs[i]));
    }
    return acc;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.bigint")
        .test("string_roundtrip",
              [](TestContext& t) {
                  t.expect_eq(bi("0").to_string(), std::string("0"), "0 round-trips");
                  t.expect_eq(bi("-0").to_string(), std::string("0"), "-0 canonicalises to 0");
                  t.expect_eq(bi("+0").to_string(), std::string("0"), "+0 canonicalises to 0");
                  t.expect_eq(bi("000").to_string(), std::string("0"), "000 is 0");
                  t.expect_eq(bi("-000").to_string(), std::string("0"), "-000 is 0");
                  t.expect_eq(bi("+42").to_string(), std::string("42"), "leading + is accepted");
                  t.expect_eq(bi("007").to_string(), std::string("7"), "leading zeros drop");
                  t.expect_eq(bi("-123456789").to_string(), std::string("-123456789"),
                              "negative round-trips");

                  // 80-digit and 100-digit values survive the round-trip verbatim.
                  const std::string big80 =
                      "12345678901234567890123456789012345678901234567890"
                      "123456789012345678901234567890";
                  t.expect_eq(bi(big80).to_string(), big80, "80-digit round-trip");
                  const std::string neg100 =
                      "-9999999999999999999999999999999999999999999999999"
                      "99999999999999999999999999999999999999999999999998";
                  t.expect_eq(bi(neg100).to_string(), neg100, "100-digit negative round-trip");
              })
        .test("string_malformed",
              [](TestContext& t) {
                  const auto is_syntax = [&](std::string_view s, std::string_view what) {
                      auto r = BigInt::from_string(s);
                      t.expect(!r.has_value() && r.error() == MathError::syntax_error, what);
                  };
                  is_syntax("", "empty string is malformed");
                  is_syntax("+", "bare + is malformed");
                  is_syntax("-", "bare - is malformed");
                  is_syntax("12a3", "embedded letter is malformed");
                  is_syntax("1.5", "decimal point is malformed");
                  is_syntax(" 12", "leading space is malformed");
                  is_syntax("12 ", "trailing space is malformed");
                  is_syntax("--5", "double sign is malformed");
                  is_syntax("+-5", "mixed sign is malformed");
                  is_syntax("0x1F", "hex is malformed");
              })
        .test("boundary_exactness",
              [](TestContext& t) {
                  // Limb boundaries and beyond, via distinct construction routes.
                  t.expect_eq(BigInt::from_u64(4294967296ULL).to_string(),
                              std::string("4294967296"), "2^32 exact");
                  t.expect_eq(BigInt::from_u64(2).pow(64).to_string(),
                              std::string("18446744073709551616"), "2^64 exact (via pow)");
                  t.expect_eq(BigInt::from_u64(2).pow(100).to_string(),
                              std::string("1267650600228229401496703205376"), "2^100 exact");

                  // (2^32 - 1)^2 computed through the schoolbook multiply.
                  const BigInt m = BigInt::from_u64(4294967295ULL);
                  t.expect_eq(m.multiply(m).to_string(),
                              std::string("18446744065119617025"), "(2^32-1)^2 exact");

                  // from_i64 at the signed extremes (INT64_MIN must not be negated in-place).
                  t.expect_eq(BigInt::from_i64(std::numeric_limits<std::int64_t>::min()).to_string(),
                              std::string("-9223372036854775808"), "INT64_MIN exact");
                  t.expect_eq(BigInt::from_i64(std::numeric_limits<std::int64_t>::max()).to_string(),
                              std::string("9223372036854775807"), "INT64_MAX exact");
              })
        .test("factorial_100",
              [](TestContext& t) {
                  BigInt f = BigInt::from_u64(1);
                  for (std::uint64_t k = 1; k <= 100; ++k) {
                      f = f.multiply(BigInt::from_u64(k));
                  }
                  t.expect_eq(f.to_string().size(), std::size_t{158}, "100! has 158 digits");
                  t.expect_eq(f.to_string(), std::string(kFactorial100), "100! matches exact value");
              })
        .test("addition_subtraction",
              [](TestContext& t) {
                  // Carry across a limb boundary: (2^32 - 1) + 1 == 2^32.
                  t.expect(BigInt::from_u64(4294967295ULL).add(BigInt::from_u64(1)) ==
                               BigInt::from_u64(4294967296ULL),
                           "carry across the limb boundary");
                  // Borrow across a limb boundary: 2^32 - 1 == 2^32 - 1.
                  t.expect(BigInt::from_u64(4294967296ULL).subtract(BigInt::from_u64(1)) ==
                               BigInt::from_u64(4294967295ULL),
                           "borrow across the limb boundary");
                  // Sign cancellation collapses to the canonical zero.
                  const BigInt x = bi("123456789012345678901234567890");
                  t.expect(x.subtract(x).is_zero(), "x - x == 0");
                  t.expect(x.add(x.negate()) == BigInt{}, "x + (-x) == 0");
                  // Mixed-sign addition picks up the larger magnitude's sign.
                  t.expect_eq(bi("-1000000000000000000000").add(bi("7")).to_string(),
                              std::string("-999999999999999999993"), "mixed-sign add");
                  t.expect_eq(bi("5").subtract(bi("9")).to_string(), std::string("-4"),
                              "small - larger goes negative");
                  // Associativity/commutativity on large values.
                  const BigInt a = bi("999999999999999999999999999999");
                  const BigInt b = bi("1");
                  t.expect(a.add(b) == b.add(a), "add commutes");
                  t.expect_eq(a.add(b).to_string(), std::string("1000000000000000000000000000000"),
                              "9...9 + 1 rolls over");
              })
        .test("division_knuth_add_back_vectors",
              [](TestContext& t) {
                  // The canonical Knuth Algorithm D / Warren divmnu vectors that force qhat
                  // over-estimation and the rare D6 ADD-BACK correction (base 2^32). The ring
                  // law recombination in check_divmod is a complete correctness check, so any
                  // add-back defect surfaces here regardless of which internal path each hits.
                  struct Vec {
                      std::vector<std::uint32_t> u;
                      std::vector<std::uint32_t> v;
                      std::string_view label;
                  };
                  const std::vector<Vec> vectors = {
                      // Classic add-back: dividend just below a divisor multiple.
                      {{0x00000000u, 0xFFFFFFFEu, 0xFFFFFFFFu}, {0xFFFFFFFFu, 0xFFFFFFFFu},
                       "u=2^96-2^32 / v=2^64-1 (add-back)"},
                      {{0x00000000u, 0x00000000u, 0x00000001u}, {0x00000001u, 0x00000000u, 0x00000001u},
                       "u=2^64 / v=2^64+1"},
                      {{0x00000000u, 0xFFFFFFFEu, 0x00000000u, 0x00000001u},
                       {0xFFFFFFFFu, 0x00000000u, 0x00000001u}, "4-limb / 3-limb add-back"},
                      // Tight qhat: top divisor limb 0x80000000 (normalisation shift 0).
                      {{0xFFFFFFFFu, 0xFFFFFFFFu, 0x7FFFFFFFu}, {0x00000001u, 0x80000000u},
                       "top limb 0x80000000, shift 0"},
                      {{0x0000FFFFu, 0x0000FFFFu, 0x0000FFFFu}, {0xFFFFFFFFu, 0x00008000u},
                       "sparse-limb correction"},
                  };
                  for (const auto& vec : vectors) {
                      const BigInt u = from_limbs(vec.u);
                      const BigInt v = from_limbs(vec.v);
                      check_divmod(t, u, v, vec.label);
                      check_divmod(t, u.negate(), v, vec.label);       // negative dividend
                      check_divmod(t, u, v.negate(), vec.label);       // negative divisor
                      check_divmod(t, u.negate(), v.negate(), vec.label);
                  }
              })
        .test("multiplication",
              [](TestContext& t) {
                  t.expect_eq(bi("-6").multiply(bi("7")).to_string(), std::string("-42"),
                              "sign of the product");
                  t.expect_eq(bi("-6").multiply(bi("-7")).to_string(), std::string("42"),
                              "negative times negative");
                  t.expect(bi("123456789").multiply(BigInt{}).is_zero(), "times zero is zero");
                  // A large exact product checked against a hand-verified literal.
                  t.expect_eq(bi("12345678901234567890").multiply(bi("98765432109876543210"))
                                  .to_string(),
                              std::string("1219326311370217952237463801111263526900"),
                              "20-digit x 20-digit product");
              })
        .test("division_law",
              [](TestContext& t) {
                  // Single-limb divisors, all four sign combinations.
                  check_divmod(t, bi("17"), bi("5"), "17 / 5");
                  check_divmod(t, bi("-17"), bi("5"), "-17 / 5");
                  check_divmod(t, bi("17"), bi("-5"), "17 / -5");
                  check_divmod(t, bi("-17"), bi("-5"), "-17 / -5");

                  // |a| < |b| gives q == 0, r == a.
                  auto small = bi("3").divmod(bi("100")).value();
                  t.expect(small.first.is_zero() && small.second == bi("3"), "3 / 100 = (0, 3)");
                  auto small_neg = bi("-3").divmod(bi("100")).value();
                  t.expect(small_neg.first.is_zero() && small_neg.second == bi("-3"),
                           "-3 / 100 = (0, -3)");

                  // Multi-limb divisors, large dividends, both signs.
                  check_divmod(t, bi("123456789012345678901234567890"), bi("987654321"),
                               "large / single-limb");
                  check_divmod(t, bi("98765432109876543210987654321098765432109876543210"),
                               bi("1234567890123456789"), "huge / multi-limb");
                  check_divmod(t, bi("-98765432109876543210987654321098765432109876543210"),
                               bi("1234567890123456789"), "huge negative / multi-limb");
                  check_divmod(t, bi("98765432109876543210987654321098765432109876543210"),
                               bi("-1234567890123456789"), "huge / negative multi-limb");
                  check_divmod(t, bi("340282366920938463463374607431768211455"),
                               bi("18446744073709551616"), "(2^128-1) / 2^64");

                  // Explicit quotient/remainder anchors (hand-verified).
                  auto e1 = bi("123456789012345678901234567890").divmod(bi("987654321")).value();
                  t.expect_eq(e1.first.to_string(), std::string("124999998873437499901"),
                              "explicit quotient");
                  t.expect_eq(e1.second.to_string(), std::string("574845669"),
                              "explicit remainder");

                  // Exact division: remainder is exactly zero.
                  auto exact =
                      bi("98765432109876543210987654321098765432109876543210").divmod(bi("987654321"))
                          .value();
                  t.expect_eq(exact.first.to_string(),
                              std::string("100000000010000000001000000000100000000010"),
                              "exact quotient");
                  t.expect(exact.second.is_zero(), "exact division has zero remainder");

                  // Truncated sign anchors.
                  t.expect_eq(bi("-17").divmod(bi("5")).value().first.to_string(),
                              std::string("-3"), "-17 / 5 quotient truncates to -3");
                  t.expect_eq(bi("-17").divmod(bi("5")).value().second.to_string(),
                              std::string("-2"), "-17 / 5 remainder is -2");

                  // Division by zero is reported, not undefined.
                  auto dz = bi("42").divmod(BigInt{});
                  t.expect(!dz.has_value() && dz.error() == MathError::division_by_zero,
                           "divide by zero fails");
                  t.expect(bi("42").divide(BigInt{}).error() == MathError::division_by_zero,
                           "divide() reports zero divisor");
                  t.expect(bi("42").mod(BigInt{}).error() == MathError::division_by_zero,
                           "mod() reports zero divisor");
              })
        .test("comparison_total_order",
              [](TestContext& t) {
                  // A deliberately mixed set (negatives, zero, limb boundaries, large).
                  std::vector<BigInt> v = {
                      bi("-100000000000000000000000000000"),
                      bi("-4294967296"),
                      bi("-1"),
                      bi("0"),
                      bi("1"),
                      bi("4294967295"),
                      bi("4294967296"),
                      bi("100000000000000000000000000000"),
                  };
                  // Already ascending: verify strict monotonicity via the spaceship order.
                  bool ordered = true;
                  for (std::size_t i = 1; i < v.size(); ++i) {
                      ordered = ordered && (v[i - 1] < v[i]) && !(v[i] < v[i - 1]);
                  }
                  t.expect(ordered, "the mixed set is strictly ascending");
                  t.expect(bi("-5") < bi("0"), "negatives precede zero");
                  t.expect(bi("0") < bi("5"), "zero precedes positives");
                  t.expect(bi("-100") < bi("-99"), "larger-magnitude negative is smaller");
                  t.expect(bi("123456789012345678901234567890") ==
                               bi("123456789012345678901234567890"),
                           "equality on large values");
                  t.expect(bi("-0") == bi("0"), "-0 compares equal to 0");
              })
        .test("gcd",
              [](TestContext& t) {
                  t.expect_eq(BigInt::gcd(bi("462"), bi("1071")).to_string(), std::string("21"),
                              "gcd(462, 1071) = 21");
                  t.expect(BigInt::gcd(BigInt{}, bi("-17")) == bi("17"),
                           "gcd(0, -17) = 17 (non-negative)");
                  t.expect(BigInt::gcd(bi("17"), BigInt{}) == bi("17"), "gcd(17, 0) = 17");
                  t.expect(BigInt::gcd(BigInt{}, BigInt{}).is_zero(), "gcd(0, 0) = 0");
                  t.expect(BigInt::gcd(bi("-462"), bi("1071")) == bi("21"),
                           "gcd ignores operand signs");

                  // Large known value: gcd(g*p, g*q) == g for coprime primes p, q.
                  const BigInt g = bi("123456789012345678901234567890");
                  const BigInt a = g.multiply(BigInt::from_u64(1000000007ULL));  // 10^9 + 7
                  const BigInt b = g.multiply(BigInt::from_u64(1000000009ULL));  // 10^9 + 9
                  t.expect(BigInt::gcd(a, b) == g, "gcd(g*p, g*q) = g for coprime p, q");
              })
        .test("pow",
              [](TestContext& t) {
                  t.expect_eq(BigInt::from_u64(2).pow(100).to_string(),
                              std::string("1267650600228229401496703205376"), "2^100");
                  t.expect_eq(BigInt::from_u64(3).pow(40).to_string(),
                              std::string("12157665459056928801"), "3^40");
                  t.expect(BigInt::from_u64(7).pow(0) == BigInt::from_u64(1), "7^0 = 1");
                  t.expect(BigInt::from_u64(5).pow(1) == BigInt::from_u64(5), "5^1 = 5");
                  t.expect_eq(bi("-2").pow(3).to_string(), std::string("-8"), "(-2)^3 = -8");
                  t.expect_eq(bi("-2").pow(4).to_string(), std::string("16"), "(-2)^4 = 16");
              })
        .test("modpow",
              [](TestContext& t) {
                  // Classic Carmichael example: 3^644 mod 645 = 36.
                  t.expect_eq(BigInt::from_u64(3)
                                  .modpow(BigInt::from_u64(644), BigInt::from_u64(645))
                                  .value()
                                  .to_string(),
                              std::string("36"), "3^644 mod 645 = 36");

                  // Fermat's little theorem: a^(p-1) mod p == 1 for prime p = 97.
                  const BigInt p = BigInt::from_u64(97);
                  const BigInt exp = BigInt::from_u64(96);
                  for (std::uint64_t base = 2; base <= 5; ++base) {
                      t.expect(BigInt::from_u64(base).modpow(exp, p).value() ==
                                   BigInt::from_u64(1),
                               "a^(p-1) mod p = 1 (p = 97)");
                  }

                  // Boundary reductions.
                  t.expect(BigInt::from_u64(12345)
                               .modpow(BigInt::from_u64(678), BigInt::from_u64(1))
                               .value()
                               .is_zero(),
                           "anything mod 1 is 0");
                  t.expect(BigInt::from_u64(12345)
                               .modpow(BigInt{}, BigInt::from_u64(97))
                               .value() == BigInt::from_u64(1),
                           "exp 0 gives 1 mod m");

                  // A negative base is reduced into [0, modulus).
                  t.expect(bi("-2").modpow(BigInt::from_u64(3), BigInt::from_u64(7)).value() ==
                               BigInt::from_u64(6),
                           "(-2)^3 mod 7 = -8 mod 7 = 6");

                  // Domain errors: negative exponent and non-positive modulus.
                  t.expect(BigInt::from_u64(2)
                               .modpow(bi("-1"), BigInt::from_u64(5))
                               .error() == MathError::domain_error,
                           "negative exponent is a domain error");
                  t.expect(BigInt::from_u64(2)
                               .modpow(BigInt::from_u64(3), BigInt{})
                               .error() == MathError::domain_error,
                           "zero modulus is a domain error");
                  t.expect(BigInt::from_u64(2)
                               .modpow(BigInt::from_u64(3), bi("-5"))
                               .error() == MathError::domain_error,
                           "negative modulus is a domain error");

                  // A larger, multi-limb modular exponentiation stays exact.
                  t.expect_eq(BigInt::from_u64(7)
                                  .modpow(BigInt::from_u64(256), bi("1000000000000000000000"))
                                  .value()
                                  .to_string(),
                              std::string("246243189632348313601"),
                              "7^256 mod 10^21 exact");
              })
        .run();
}
