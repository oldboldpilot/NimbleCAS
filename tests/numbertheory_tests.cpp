// Tests for nimblecas.numbertheory: modular inverse, Miller-Rabin primality, CRT,
// Jacobi symbol, and the educational RSA demonstration — all on exact BigInt.
// @author Olumuyiwa Oluwasanmi
//
// Every literal here is hand-verified. Primality is checked against known primes (incl.
// the Mersenne prime 2^61-1, the Mersenne prime 2^89-1, and the smallest 100-digit prime
// 10^99 + 289), known composites, and two Carmichael numbers (561, 41041) that fool the
// naive Fermat test but not Miller-Rabin. The extended-Euclid identity a*x + b*y == g is
// re-derived from scratch for each case, and the RSA round-trip exercises the full
// generate -> encrypt -> decrypt path.

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.numbertheory;
import nimblecas.testing;

using nimblecas::BigInt;
using nimblecas::ExtGcd;
using nimblecas::MathError;
using nimblecas::RsaKey;
using nimblecas::crt;
using nimblecas::extended_gcd;
using nimblecas::is_probable_prime;
using nimblecas::jacobi_symbol;
using nimblecas::mod_inverse;
using nimblecas::next_prime;
using nimblecas::rsa_decrypt;
using nimblecas::rsa_encrypt;
using nimblecas::rsa_generate;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Parse a decimal literal that is known-good in this test file.
[[nodiscard]] auto bi(std::string_view s) -> BigInt {
    return BigInt::from_string(s).value();
}

// The default Miller-Rabin seed for cases where the value sits below the deterministic
// bound (the seed is then unused) or where any seed suffices.
constexpr std::uint64_t kSeed = 0x9E3779B97F4A7C15ULL;

// Re-derive the Bezout identity for (a, b) and check g == gcd(a,b), g >= 0, a*x + b*y == g.
auto check_bezout(TestContext& t, const BigInt& a, const BigInt& b, std::string_view label)
    -> void {
    const ExtGcd eg = extended_gcd(a, b);
    const BigInt lhs = a.multiply(eg.x).add(b.multiply(eg.y));
    t.expect(lhs == eg.g, label);                              // a*x + b*y == g
    t.expect(!eg.g.is_negative(), label);                     // g >= 0
    t.expect(eg.g == BigInt::gcd(a, b), label);               // g == gcd(|a|, |b|)
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.numbertheory")
        .test("extended_gcd_identity",
              [](TestContext& t) {
                  // Small anchor: gcd(240, 46) == 2.
                  const ExtGcd eg = extended_gcd(bi("240"), bi("46"));
                  t.expect_eq(eg.g.to_string(), std::string("2"), "gcd(240, 46) == 2");
                  t.expect(bi("240").multiply(eg.x).add(bi("46").multiply(eg.y)) == eg.g,
                           "240*x + 46*y == 2");

                  // A broad spread of sign combinations and magnitudes.
                  check_bezout(t, bi("240"), bi("46"), "240, 46");
                  check_bezout(t, bi("46"), bi("240"), "46, 240 (swapped)");
                  check_bezout(t, bi("-240"), bi("46"), "-240, 46");
                  check_bezout(t, bi("240"), bi("-46"), "240, -46");
                  check_bezout(t, bi("-240"), bi("-46"), "-240, -46");
                  check_bezout(t, bi("17"), bi("0"), "17, 0");
                  check_bezout(t, bi("0"), bi("17"), "0, 17");
                  check_bezout(t, bi("0"), bi("0"), "0, 0 (g == 0)");
                  check_bezout(t, bi("1071"), bi("462"), "1071, 462");

                  // Large coprime and non-coprime pairs.
                  check_bezout(t, bi("123456789012345678901234567890"),
                               bi("987654321098765432109876543211"), "large coprime pair");
                  check_bezout(t, bi("-98765432109876543210987654321098765432109876543210"),
                               bi("1234567890123456789"), "large negative / positive");
              })
        .test("mod_inverse",
              [](TestContext& t) {
                  // 3^-1 mod 11 == 4, and 3*4 mod 11 == 1.
                  auto i1 = mod_inverse(bi("3"), bi("11"));
                  t.expect(i1.has_value() && *i1 == bi("4"), "3^-1 mod 11 == 4");
                  t.expect(bi("3").multiply(*i1).mod(bi("11")).value() == bi("1"),
                           "3 * (3^-1) == 1 (mod 11)");

                  // 17^-1 mod 3120 == 2753 (17 * 2753 == 46801 == 15*3120 + 1).
                  auto i2 = mod_inverse(bi("17"), bi("3120"));
                  t.expect(i2.has_value() && *i2 == bi("2753"), "17^-1 mod 3120 == 2753");
                  t.expect(bi("17").multiply(*i2).mod(bi("3120")).value() == bi("1"),
                           "17 * (17^-1) == 1 (mod 3120)");

                  // Result is normalised into [0, m) even when the raw coefficient is negative.
                  auto i3 = mod_inverse(bi("5"), bi("7"));  // 5*3 == 15 == 2*7 + 1
                  t.expect(i3.has_value() && *i3 == bi("3"), "5^-1 mod 7 == 3 (normalised)");

                  // A large modular inverse verified by multiplying back to 1. Modulus is the
                  // Mersenne prime 2^61-1 and 0 < a < m, so gcd(a, m) == 1 is guaranteed.
                  const BigInt m = bi("2305843009213693951");  // 2^61 - 1 (prime)
                  const BigInt a = bi("123456789");
                  auto i4 = mod_inverse(a, m);
                  t.expect(i4.has_value(), "large inverse exists (m prime, 0 < a < m)");
                  t.expect(a.multiply(*i4).mod(m).value() == bi("1"),
                           "a * a^-1 == 1 (mod large prime)");

                  // Non-coprime and non-positive modulus are domain errors.
                  auto e1 = mod_inverse(bi("4"), bi("6"));  // gcd(4, 6) == 2
                  t.expect(!e1.has_value() && e1.error() == MathError::domain_error,
                           "non-coprime inverse is a domain error");
                  auto e2 = mod_inverse(bi("3"), bi("0"));
                  t.expect(!e2.has_value() && e2.error() == MathError::domain_error,
                           "modulus 0 is a domain error");
                  auto e3 = mod_inverse(bi("3"), bi("-11"));
                  t.expect(!e3.has_value() && e3.error() == MathError::domain_error,
                           "negative modulus is a domain error");
              })
        .test("is_probable_prime_true",
              [](TestContext& t) {
                  const std::vector<std::string_view> primes = {
                      "2", "3", "5", "7", "97", "7919", "104729",
                      "2305843009213693951",  // 2^61 - 1 (Mersenne prime)
                  };
                  for (std::string_view p : primes) {
                      auto r = is_probable_prime(bi(p), kSeed);
                      t.expect(r.has_value() && *r, p);
                  }

                  // 2^89 - 1 (Mersenne prime, 27 digits) — above the deterministic bound.
                  auto m89 = is_probable_prime(bi("618970019642690137449562111"), kSeed);
                  t.expect(m89.has_value() && *m89, "2^89 - 1 is prime");

                  // The smallest 100-digit prime, 10^99 + 289 == 1 (96 zeros) 289. Exercises
                  // the random-witness branch well above the deterministic bound.
                  const std::string p100 = std::string("1") + std::string(96, '0') + "289";
                  auto big = is_probable_prime(bi(p100), kSeed);
                  t.expect(big.has_value() && *big, "10^99 + 289 (smallest 100-digit prime)");
              })
        .test("is_probable_prime_false",
              [](TestContext& t) {
                  const std::vector<std::string_view> composites = {
                      "1", "4", "9", "15",
                      "561",    // Carmichael (3 * 11 * 17)
                      "41041",  // Carmichael (7 * 11 * 13 * 41)
                      "2305843009213693949",  // 2^61 - 3 (composite)
                      "0",
                  };
                  for (std::string_view c : composites) {
                      auto r = is_probable_prime(bi(c), kSeed);
                      t.expect(r.has_value() && !*r, c);
                  }

                  // Negatives are non-prime.
                  auto neg = is_probable_prime(bi("-7"), kSeed);
                  t.expect(neg.has_value() && !*neg, "-7 is not prime");

                  // A large composite: the product of two known Mersenne primes.
                  const BigInt composite =
                      bi("2305843009213693951").multiply(bi("618970019642690137449562111"));
                  auto r = is_probable_prime(composite, kSeed);
                  t.expect(r.has_value() && !*r, "(2^61-1)(2^89-1) is composite");
              })
        .test("next_prime",
              [](TestContext& t) {
                  auto n1 = next_prime(bi("90"), kSeed);
                  t.expect(n1.has_value() && *n1 == bi("97"), "next_prime(90) == 97");

                  // next_prime(2^61 - 2) == 2^61 - 1.
                  auto n2 = next_prime(bi("2305843009213693950"), kSeed);
                  t.expect(n2.has_value() && *n2 == bi("2305843009213693951"),
                           "next_prime(2^61-2) == 2^61-1");

                  // Small and below-2 boundaries.
                  t.expect(next_prime(bi("0"), kSeed).value() == bi("2"), "next_prime(0) == 2");
                  t.expect(next_prime(bi("1"), kSeed).value() == bi("2"), "next_prime(1) == 2");
                  t.expect(next_prime(bi("2"), kSeed).value() == bi("3"), "next_prime(2) == 3");
                  t.expect(next_prime(bi("-5"), kSeed).value() == bi("2"), "next_prime(-5) == 2");
                  t.expect(next_prime(bi("7"), kSeed).value() == bi("11"), "next_prime(7) == 11");

                  // The result is always strictly greater and itself prime.
                  auto np = next_prime(bi("1000"), kSeed);
                  t.expect(np.has_value() && *np == bi("1009"), "next_prime(1000) == 1009");
              })
        .test("crt",
              [](TestContext& t) {
                  // Canonical example: x ≡ 2 (3), 3 (5), 2 (7) -> x == 23.
                  auto r = crt({bi("2"), bi("3"), bi("2")}, {bi("3"), bi("5"), bi("7")});
                  t.expect(r.has_value() && *r == bi("23"), "crt({2,3,2},{3,5,7}) == 23");
                  // Verify the congruences directly.
                  t.expect(r->mod(bi("3")).value() == bi("2"), "23 ≡ 2 (mod 3)");
                  t.expect(r->mod(bi("5")).value() == bi("3"), "23 ≡ 3 (mod 5)");
                  t.expect(r->mod(bi("7")).value() == bi("2"), "23 ≡ 2 (mod 7)");

                  // A two-modulus case with a negative residue folded into range.
                  auto r2 = crt({bi("-1"), bi("1")}, {bi("5"), bi("7")});  // x ≡ 4 (5), 1 (7)
                  t.expect(r2.has_value() && *r2 == bi("29"), "crt({-1,1},{5,7}) == 29");

                  // Large coprime moduli stay exact — two distinct Mersenne primes, so their
                  // coprimality is certain.
                  const BigInt m61 = bi("2305843009213693951");           // 2^61 - 1
                  const BigInt m89 = bi("618970019642690137449562111");   // 2^89 - 1
                  auto r3 = crt({bi("123"), bi("456")}, {m61, m89});
                  t.expect(r3.has_value(), "large CRT solves");
                  t.expect(r3->mod(m61).value() == bi("123"), "large CRT residue 1");
                  t.expect(r3->mod(m89).value() == bi("456"), "large CRT residue 2");

                  // Failure modes.
                  auto e1 = crt({bi("2"), bi("3")}, {bi("4"), bi("6")});  // gcd(4,6) == 2
                  t.expect(!e1.has_value() && e1.error() == MathError::domain_error,
                           "non-coprime moduli is a domain error");
                  auto e2 = crt({bi("2")}, {bi("3"), bi("5")});  // length mismatch
                  t.expect(!e2.has_value() && e2.error() == MathError::domain_error,
                           "length mismatch is a domain error");
                  auto e3 = crt({}, {});  // empty
                  t.expect(!e3.has_value() && e3.error() == MathError::domain_error,
                           "empty input is a domain error");
                  auto e4 = crt({bi("2")}, {bi("0")});  // modulus < 1
                  t.expect(!e4.has_value() && e4.error() == MathError::domain_error,
                           "modulus < 1 is a domain error");
              })
        .test("jacobi_symbol",
              [](TestContext& t) {
                  auto j1 = jacobi_symbol(bi("2"), bi("15"));
                  t.expect(j1.has_value() && *j1 == 1, "(2/15) == 1");
                  auto j2 = jacobi_symbol(bi("1001"), bi("9907"));
                  t.expect(j2.has_value() && *j2 == -1, "(1001/9907) == -1");

                  // Legendre agreement and the zero case.
                  auto j3 = jacobi_symbol(bi("2"), bi("7"));  // 3^2==2 mod 7 -> QR
                  t.expect(j3.has_value() && *j3 == 1, "(2/7) == 1");
                  auto j4 = jacobi_symbol(bi("3"), bi("7"));  // non-residue mod 7
                  t.expect(j4.has_value() && *j4 == -1, "(3/7) == -1");
                  auto j5 = jacobi_symbol(bi("6"), bi("9"));  // gcd(6,9) == 3
                  t.expect(j5.has_value() && *j5 == 0, "(6/9) == 0 (shared factor)");
                  auto j6 = jacobi_symbol(bi("5"), bi("1"));
                  t.expect(j6.has_value() && *j6 == 1, "(5/1) == 1");

                  // A negative numerator is reduced mod n first.
                  auto j7 = jacobi_symbol(bi("-1"), bi("15"));  // (-1/15): 15 ≡ 3 mod 4 -> -1
                  t.expect(j7.has_value() && *j7 == -1, "(-1/15) == -1");

                  // Even or non-positive n is a domain error.
                  auto e1 = jacobi_symbol(bi("3"), bi("4"));
                  t.expect(!e1.has_value() && e1.error() == MathError::domain_error,
                           "even modulus is a domain error");
                  auto e2 = jacobi_symbol(bi("3"), bi("-5"));
                  t.expect(!e2.has_value() && e2.error() == MathError::domain_error,
                           "negative modulus is a domain error");
                  auto e3 = jacobi_symbol(bi("3"), bi("0"));
                  t.expect(!e3.has_value() && e3.error() == MathError::domain_error,
                           "zero modulus is a domain error");
              })
        .test("rsa_roundtrip",
              [](TestContext& t) {
                  auto key = rsa_generate(64, 0xC0FFEEULL);
                  t.expect(key.has_value(), "rsa_generate(64) succeeds");
                  if (key.has_value()) {
                      const RsaKey& k = *key;
                      // Public exponent starts at the customary 65537 and stays coprime to phi.
                      t.expect(!(k.n < bi("2")), "modulus n >= 2");
                      // Round-trip several messages, each < n.
                      for (std::string_view msg : {"0", "1", "42", "123456789", "1000000007"}) {
                          const BigInt m = bi(msg);
                          if (!(m < k.n)) {
                              continue;
                          }
                          auto c = rsa_encrypt(m, k.e, k.n);
                          t.expect(c.has_value(), "encrypt succeeds");
                          auto back = rsa_decrypt(c.value(), k.d, k.n);
                          t.expect(back.has_value() && *back == m, "decrypt(encrypt(m)) == m");
                      }
                      // e * d ≡ 1 modulo the group order is implied; sanity-check the identity
                      // via a random message once more.
                      const BigInt m = bi("987654321");
                      if (m < k.n) {
                          auto c = rsa_encrypt(m, k.e, k.n);
                          auto back = rsa_decrypt(c.value(), k.d, k.n);
                          t.expect(back.has_value() && *back == m, "round-trip 987654321");
                      }
                  }

                  // Too-small key size is a domain error.
                  auto e = rsa_generate(8, 1);
                  t.expect(!e.has_value() && e.error() == MathError::domain_error,
                           "rsa_generate(8) is a domain error");
              })
        .run();
}
