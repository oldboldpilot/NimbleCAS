// Feature/integration tests: the WIDE-ARITHMETIC TOWER end to end (int128, bigint,
// bigrational, bigfloat, doubledouble, bigcombinatorics, bigpowerseries, bigmatrix,
// bigeigen, constants, numbertheory).
// @author Olumuyiwa Oluwasanmi
//
// These are cross-module *feature* tests, not per-module unit tests: each case wires two or
// more tiers of the exact-arithmetic tower together and asserts a concrete value or an exact
// mathematical identity. Where a result is an exact element of Z or Q the assertions use
// bit-for-bit equality (BigInt / BigRational / Rational128 are canonical, so == is a value
// compare). Only the transcendental / floating tiers (BigFloat, constants, DoubleDouble) use
// a stated digit tolerance, because those are honestly rounded, not exact.
//
// The spine of the suite is TIER CONSISTENCY: a computation that overflows the int64
// Rational is exact in Rational128; one that overflows Rational128 is exact in BigRational;
// and the same mathematical value agrees across every tier that can represent it.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.int128;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.bigfloat;
import nimblecas.doubledouble;
import nimblecas.bigcombinatorics;
import nimblecas.bigpowerseries;
import nimblecas.bigmatrix;
import nimblecas.bigeigen;
import nimblecas.constants;
import nimblecas.numbertheory;
import nimblecas.testing;

using nimblecas::BigFloat;
using nimblecas::BigInt;
using nimblecas::BigMatrix;
using nimblecas::BigPowerSeries;
using nimblecas::BigRational;
using nimblecas::DoubleDouble;
using nimblecas::MathError;
using nimblecas::Matrix;
using nimblecas::Rational;
using nimblecas::Rational128;

// bigcombinatorics (BigInt-valued integer sequences). NOTE: `catalan` is deliberately NOT
// imported by name here — nimblecas.bigcombinatorics and nimblecas.constants both export a
// `catalan(std::int64_t)`, so an unqualified call would be ambiguous. The Catalan NUMBER is
// obtained below via binomial(2n,n)/(n+1) instead, keeping both modules importable.
using nimblecas::bell;
using nimblecas::binomial;
using nimblecas::double_factorial;
using nimblecas::factorial;
using nimblecas::falling_factorial;
using nimblecas::fibonacci;
using nimblecas::lucas;
using nimblecas::multinomial;
using nimblecas::rising_factorial;
using nimblecas::stirling_first_unsigned;
using nimblecas::stirling_second;

// bigeigen (over BigMatrix).
using nimblecas::characteristic_polynomial;
using nimblecas::determinant;
using nimblecas::inverse;
using nimblecas::rational_eigenvalues;

// bigfloat batched reductions.
using nimblecas::bigfloat_dot;
using nimblecas::bigfloat_sum;

// doubledouble EFTs / reductions.
using nimblecas::dd_dot;
using nimblecas::dd_dot_scalar;
using nimblecas::dd_sum;
using nimblecas::dd_sum_scalar;
using nimblecas::two_sum;

// int128 free functions.
using nimblecas::int128_from_i64;
using nimblecas::int128_from_string;
using nimblecas::int128_to_string;

// constants (transcendental / irrational, correctly-rounded BigFloat).
using nimblecas::e;
using nimblecas::euler_mascheroni;
using nimblecas::golden_ratio;
using nimblecas::ln10;
using nimblecas::ln2;
using nimblecas::pi;
using nimblecas::sqrt2;

// numbertheory.
using nimblecas::crt;
using nimblecas::extended_gcd;
using nimblecas::is_probable_prime;
using nimblecas::jacobi_symbol;
using nimblecas::mod_inverse;
using nimblecas::next_prime;

using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// --- BigInt / BigRational builders -----------------------------------------

[[nodiscard]] auto bi(std::string_view s) -> BigInt { return BigInt::from_string(s).value(); }
[[nodiscard]] auto biv(std::int64_t v) -> BigInt { return BigInt::from_i64(v); }
[[nodiscard]] auto bri(std::int64_t v) -> BigRational { return BigRational::from_int(v); }
[[nodiscard]] auto brmake(std::int64_t n, std::int64_t d) -> BigRational {
    return BigRational::make(biv(n), biv(d)).value();
}

// --- BigFloat builder ------------------------------------------------------

[[nodiscard]] auto bf(std::int64_t v, std::int64_t prec) -> BigFloat {
    return BigFloat::from_i64(v, prec).value();
}

// --- int64 Matrix builder (mirrors tests/feature_linalg_tests.cpp) ---------

[[nodiscard]] auto mat(std::vector<std::vector<std::int64_t>> rows) -> Matrix {
    std::vector<std::vector<Rational>> r;
    r.reserve(rows.size());
    for (const auto& row : rows) {
        std::vector<Rational> rr;
        rr.reserve(row.size());
        for (const std::int64_t v : row) {
            rr.push_back(Rational::from_int(v));
        }
        r.push_back(std::move(rr));
    }
    return Matrix::from_rows(std::move(r)).value();
}

// --- BigMatrix builder from integer rows -----------------------------------

[[nodiscard]] auto bmat(std::vector<std::vector<std::int64_t>> rows) -> BigMatrix {
    std::vector<std::vector<BigRational>> r;
    r.reserve(rows.size());
    for (const auto& row : rows) {
        std::vector<BigRational> rr;
        rr.reserve(row.size());
        for (const std::int64_t v : row) {
            rr.push_back(BigRational::from_int(v));
        }
        r.push_back(std::move(rr));
    }
    return BigMatrix::from_rows(std::move(r)).value();
}

// Horner evaluation of a low-degree-first BigRational polynomial at x (all infallible).
[[nodiscard]] auto bpoly_eval(const std::vector<BigRational>& p, const BigRational& x)
    -> BigRational {
    BigRational acc;  // 0/1
    for (std::size_t i = p.size(); i-- > 0;) {
        acc = acc.multiply(x).add(p[i]);
    }
    return acc;
}

[[nodiscard]] auto close(double a, double b, double tol) -> bool {
    return std::fabs(a - b) < tol;
}

}  // namespace

auto main() -> int {
    return TestSuite("feature.arithmetic")
        // ===================================================================
        // TIER CONSISTENCY: int64 Rational -> Rational128 -> BigRational.
        // ===================================================================
        .test("int64_overflows_where_rational128_and_bigrational_are_exact",
              [](TestContext& t) {
                  const std::int64_t ten10 = 10000000000LL;  // 10^10 (fits int64)
                  const std::string p20 = biv(10).pow(20).to_string();

                  // 10^10 * 10^10 == 10^20 overflows the ~9.2e18 int64 Rational ceiling.
                  auto r64 = Rational::from_int(ten10).multiply(Rational::from_int(ten10));
                  t.expect(!r64.has_value(), "int64 Rational 10^20 overflows");
                  t.expect(r64.error() == MathError::overflow, "and reports MathError::overflow");

                  // Rational128 represents 10^20 exactly (native __int128, ~1.7e38 ceiling).
                  auto r128 = Rational128::from_int(ten10).multiply(Rational128::from_int(ten10));
                  t.expect(r128.has_value(), "Rational128 holds 10^20");
                  t.expect(r128->to_string() == p20, "Rational128 value == 10^20");

                  // BigRational agrees on the same value (unbounded exact tier).
                  auto rb = BigRational::from_int(ten10).multiply(BigRational::from_int(ten10));
                  t.expect(rb.is_integer() && rb.numerator().to_string() == p20,
                           "BigRational value == 10^20 (agrees with Rational128)");
              })
        .test("rational128_overflows_where_bigrational_stays_exact",
              [](TestContext& t) {
                  const std::int64_t ten10 = 10000000000LL;
                  const std::string p30 = biv(10).pow(30).to_string();
                  const std::string p40 = biv(10).pow(40).to_string();

                  auto a = Rational128::from_int(ten10);
                  auto a2 = a.multiply(a);          // 10^20
                  t.expect(a2.has_value(), "Rational128 10^20 ok");
                  auto a3 = a2->multiply(a);        // 10^30 (still < 2^127)
                  t.expect(a3.has_value(), "Rational128 10^30 ok");
                  auto a4 = a3->multiply(a);        // 10^40 exceeds the 128-bit ceiling
                  t.expect(!a4.has_value() && a4.error() == MathError::overflow,
                           "Rational128 10^40 overflows -> MathError::overflow");

                  // BigRational carries 10^40 exactly.
                  auto big4 = BigRational::from_int(ten10).pow(4).value();
                  t.expect(big4.numerator().to_string() == p40, "BigRational 10^40 exact");

                  // Cross-tier agreement at the largest commonly representable value, 10^30.
                  auto big3 = BigRational::from_int(ten10).pow(3).value();
                  t.expect(a3->to_string() == p30 && big3.numerator().to_string() == p30,
                           "Rational128 and BigRational agree at 10^30");
              })
        .test("int128_string_roundtrip_and_exact_rational128_fractions",
              [](TestContext& t) {
                  // 2^127 - 1 is the Int128 maximum; 2^127 is one past it.
                  auto vmax = int128_from_string("170141183460469231731687303715884105727");
                  t.expect(vmax.has_value(), "2^127-1 parses");
                  t.expect(int128_to_string(*vmax) == "170141183460469231731687303715884105727",
                           "Int128 max round-trips through decimal");
                  t.expect(!int128_from_string("170141183460469231731687303715884105728").has_value(),
                           "2^127 overflows Int128");

                  // 1/3 + 1/6 == 1/2 exactly over Int128, reduced.
                  auto third = Rational128::make(int128_from_i64(1), int128_from_i64(3)).value();
                  auto sixth = Rational128::make(int128_from_i64(1), int128_from_i64(6)).value();
                  auto half = Rational128::make(int128_from_i64(1), int128_from_i64(2)).value();
                  t.expect(third.add(sixth).value() == half, "1/3 + 1/6 == 1/2 over Int128");
              })
        // ===================================================================
        // BigInt: 100!, Fibonacci, gcd identities, Knuth-division round-trips.
        // ===================================================================
        .test("factorial_100_is_exact_and_matches_bigcombinatorics",
              [](TestContext& t) {
                  // 100! via a raw BigInt multiply loop, and via bigcombinatorics::factorial.
                  BigInt loop = biv(1);
                  for (std::int64_t i = 2; i <= 100; ++i) {
                      loop = loop.multiply(biv(i));
                  }
                  t.expect(loop == factorial(100).value(),
                           "100! by BigInt loop == bigcombinatorics::factorial(100)");
                  const std::string s = loop.to_string();
                  // Known structure of 100!: 158 digits, leading 93326215443944152681, and
                  // exactly floor(100/5)+floor(100/25) = 24 trailing zeros.
                  t.expect(s.size() == 158, "100! has 158 digits");
                  t.expect(s.substr(0, 20) == "93326215443944152681", "100! leading 20 digits");
                  t.expect(s.substr(s.size() - 24) == std::string(24, '0'),
                           "100! ends in 24 zeros");
              })
        .test("fibonacci_fast_doubling_matches_naive_and_is_exact_beyond_int64",
              [](TestContext& t) {
                  // F(100) overflows int64 (F(93) already does); its exact value is known.
                  t.expect(fibonacci(100).value().to_string() == "354224848179261915075",
                           "F(100) == 354224848179261915075 (beyond int64)");

                  // Fast-doubling fibonacci(n) agrees with the naive additive recurrence.
                  bool all_match = true;
                  BigInt a = biv(0);
                  BigInt b = biv(1);
                  for (std::int64_t n = 0; n <= 100; ++n) {
                      if (!(fibonacci(n).value() == a)) {
                          all_match = false;
                      }
                      BigInt next = a.add(b);
                      a = b;
                      b = next;
                  }
                  t.expect(all_match, "fast-doubling F(n) == naive recurrence for n in [0,100]");

                  // gcd(F_m, F_n) == F_gcd(m,n): gcd(F12, F18) == F6 == 8.
                  t.expect(BigInt::gcd(fibonacci(12).value(), fibonacci(18).value()) ==
                               fibonacci(6).value(),
                           "gcd(F12,F18) == F(gcd(12,18)) == F6");
                  t.expect(fibonacci(6).value() == biv(8), "F6 == 8");
              })
        .test("knuth_division_roundtrips_and_modpow_identities",
              [](TestContext& t) {
                  const BigInt a = bi("314159265358979323846264338327950288419716939937510");
                  const BigInt b = bi("2718281828459045235360287471352");

                  auto dm = a.divmod(b).value();
                  t.expect(dm.first.multiply(b).add(dm.second) == a,
                           "a == (a/b)*b + a%b (positive dividend)");
                  t.expect(dm.second.sign() >= 0 && dm.second < b, "0 <= r < b for positive a");

                  // Negative dividend: remainder takes the DIVIDEND's sign, |r| < |b|.
                  const BigInt an = a.negate();
                  auto dmn = an.divmod(b).value();
                  t.expect(dmn.first.multiply(b).add(dmn.second) == an,
                           "roundtrip holds for a negative dividend");
                  t.expect((dmn.second.is_zero() || dmn.second.is_negative()) &&
                               dmn.second.abs() < b,
                           "remainder inherits the dividend's sign, |r| < |b|");

                  // modpow sanity: 2^10 mod 1000 == 24, and x mod 1 == 0.
                  t.expect(biv(2).modpow(biv(10), biv(1000)).value() == biv(24),
                           "2^10 mod 1000 == 24");
                  t.expect(biv(5).modpow(biv(3), biv(1)).value().is_zero(), "x mod 1 == 0");
              })
        .test("fermat_little_theorem_via_modpow",
              [](TestContext& t) {
                  // a^(p-1) == 1 (mod p) for a prime p and a not divisible by p.
                  const BigInt p = bi("97");
                  for (const std::int64_t av : {2, 5, 10, 50}) {
                      t.expect(biv(av).modpow(p.subtract(biv(1)), p).value() == biv(1),
                               "Fermat: a^(p-1) == 1 mod 97");
                  }
                  // Same identity for a large prime discovered by next_prime.
                  const BigInt pr = next_prime(biv(1000000), 42).value();
                  t.expect(is_probable_prime(pr, 42).value(), "next_prime(1000000) is prime");
                  t.expect(biv(7).modpow(pr.subtract(biv(1)), pr).value() == biv(1),
                           "Fermat holds for a >10^6 prime");
              })
        // ===================================================================
        // BigRational: exact harmonic / telescoping sums, canonical reduction.
        // ===================================================================
        .test("harmonic_and_telescoping_sums_are_exact_reduced_fractions",
              [](TestContext& t) {
                  // H_5 = 1 + 1/2 + 1/3 + 1/4 + 1/5 == 137/60.
                  BigRational h;  // 0/1
                  for (std::int64_t k = 1; k <= 5; ++k) {
                      h = h.add(brmake(1, k));
                  }
                  t.expect(h == brmake(137, 60), "H_5 == 137/60 exactly");
                  t.expect(h.denominator().sign() > 0, "denominator kept strictly positive");
                  t.expect(BigInt::gcd(h.numerator(), h.denominator()) == biv(1),
                           "fraction is gcd-reduced (lowest terms)");

                  // Telescoping: sum_{k=1}^{10} 1/(k(k+1)) == 1 - 1/11 == 10/11.
                  BigRational s;
                  for (std::int64_t k = 1; k <= 10; ++k) {
                      s = s.add(brmake(1, k * (k + 1)));
                  }
                  t.expect(s == brmake(10, 11), "telescoping sum == 10/11");
              })
        // ===================================================================
        // bigcombinatorics vs bigint: binomial identities and sequences.
        // ===================================================================
        .test("binomial_identities_against_factorials_and_powers",
              [](TestContext& t) {
                  // C(20,7) == 20!/(7! 13!) == 77520.
                  const BigInt c207 = binomial(20, 7).value();
                  const BigInt num = factorial(20).value();
                  const BigInt den = factorial(7).value().multiply(factorial(13).value());
                  t.expect(c207 == num.divide(den).value(), "C(20,7) == 20!/(7! 13!)");
                  t.expect(c207 == biv(77520), "C(20,7) == 77520");

                  // Pascal's rule: C(30,12) == C(29,11) + C(29,12).
                  t.expect(binomial(30, 12).value() ==
                               binomial(29, 11).value().add(binomial(29, 12).value()),
                           "Pascal's rule C(n,k)=C(n-1,k-1)+C(n-1,k)");

                  // Row sum: sum_{k=0}^{25} C(25,k) == 2^25.
                  BigInt rowsum = biv(0);
                  for (std::int64_t k = 0; k <= 25; ++k) {
                      rowsum = rowsum.add(binomial(25, k).value());
                  }
                  t.expect(rowsum == biv(2).pow(25), "sum_k C(25,k) == 2^25");

                  // Vandermonde: sum_j C(8,j) C(9,7-j) == C(17,7).
                  BigInt vsum = biv(0);
                  for (std::int64_t j = 0; j <= 7; ++j) {
                      vsum = vsum.add(binomial(8, j).value().multiply(binomial(9, 7 - j).value()));
                  }
                  t.expect(vsum == binomial(17, 7).value(),
                           "Vandermonde sum_j C(8,j)C(9,7-j) == C(17,7)");

                  // Catalan number C_10 == C(20,10)/11 == 16796 (via binomial; see import note).
                  t.expect(binomial(20, 10).value().divide(biv(11)).value() == biv(16796),
                           "Catalan C_10 == C(20,10)/11 == 16796");
              })
        .test("integer_sequences_have_their_known_values",
              [](TestContext& t) {
                  // multinomial (2,3,4) = 9!/(2!3!4!) == 1260.
                  std::array<std::int64_t, 3> ks{2, 3, 4};
                  t.expect(multinomial(std::span<const std::int64_t>(ks)).value() == biv(1260),
                           "multinomial(2,3,4) == 1260");
                  t.expect(lucas(10).value() == biv(123), "L_10 == 123");
                  t.expect(bell(5).value() == biv(52), "Bell_5 == 52");
                  t.expect(stirling_second(5, 2).value() == biv(15), "S(5,2) == 15");
                  t.expect(stirling_first_unsigned(5, 2).value() == biv(50), "c(5,2) == 50");
                  t.expect(double_factorial(9).value() == biv(945), "9!! == 945");
                  t.expect(falling_factorial(10, 3).value() == biv(720), "10 falling 3 == 720");
                  t.expect(rising_factorial(10, 3).value() == biv(1320), "10 rising 3 == 1320");
              })
        // ===================================================================
        // bigpowerseries: exact transcendental identities over Q, unbounded.
        // ===================================================================
        .test("power_series_transcendental_identities_over_Q",
              [](TestContext& t) {
                  const std::size_t N = 12;
                  const auto x = BigPowerSeries::variable(N).value();
                  const auto one = BigPowerSeries::one(N).value();

                  // exp(x)^2 == exp(2x), coefficientwise.
                  const auto exp_x = x.exp().value();
                  const auto exp_2x = x.scale(bri(2)).value().exp().value();
                  t.expect(exp_x.multiply(exp_x).value() == exp_2x,
                           "exp(x)^2 == exp(2x) coefficientwise");

                  // (1 - x) * geometric == 1, and the geometric series has all-ones coeffs.
                  const auto omx = one.subtract(x).value();
                  const auto geo = omx.inverse().value();  // 1/(1-x) = sum x^k
                  t.expect(omx.multiply(geo).value() == one, "(1-x) * 1/(1-x) == 1");
                  bool geo_ones = true;
                  for (std::size_t k = 0; k < N; ++k) {
                      if (!(geo.coefficient(k) == bri(1))) {
                          geo_ones = false;
                      }
                  }
                  t.expect(geo_ones, "geometric series coefficients are all 1");

                  // log(exp(x)) == x.
                  t.expect(exp_x.log().value() == x, "log(exp(x)) == x");

                  // derivative and integrate are inverse (top term truncated to 0, so pick f
                  // whose x^{N-1} coefficient is 0): derivative(integrate(f)) == f.
                  const auto f =
                      BigPowerSeries::from_coeffs({bri(3), bri(5), bri(7), bri(2), bri(9)}, 6)
                          .value();
                  t.expect(f.integrate().value().derivative().value() == f,
                           "derivative(integrate(f)) == f");
              })
        .test("exp_series_coefficient_25_is_one_over_25_factorial",
              [](TestContext& t) {
                  // The x^25 coefficient of exp is 1/25! — 25! overflows the int64 powerseries
                  // tier but is exact here over BigRational.
                  const auto exp_x = BigPowerSeries::variable(30).value().exp().value();
                  const BigInt f25 = factorial(25).value();
                  const auto expected = BigRational::make(biv(1), f25).value();
                  t.expect(exp_x.coefficient(25) == expected, "[x^25] exp == 1/25!");
                  t.expect(!exp_x.coefficient(25).is_integer(), "1/25! is a genuine fraction");
              })
        // ===================================================================
        // bigmatrix + bigeigen: characteristic polynomial, determinant routes.
        // ===================================================================
        .test("triangular_char_poly_factors_and_spectrum_matches_trace_and_det",
              [](TestContext& t) {
                  // Upper-triangular with diagonal (2,3,5): char poly == (x-2)(x-3)(x-5),
                  // trace 10, determinant 30, spectrum {2,3,5} — all rational.
                  const auto A = bmat({{2, 1, 4}, {0, 3, 5}, {0, 0, 5}});

                  const auto cp = characteristic_polynomial(A).value();
                  t.expect(cp.size() == 4, "degree-3 char poly has 4 coefficients");
                  t.expect(cp.back() == bri(1), "char poly is monic");
                  t.expect(cp.front() == bri(-30), "constant term == (-1)^3 det == -30");
                  for (const std::int64_t d : {2, 3, 5}) {
                      t.expect(bpoly_eval(cp, bri(d)) == bri(0),
                               "char poly vanishes at each diagonal eigenvalue");
                  }

                  // Two independent exact determinants agree: Bareiss (BigMatrix) and
                  // Faddeev-LeVerrier (bigeigen).
                  const auto det_bareiss = A.determinant().value();
                  const auto det_faddeev = determinant(A).value();
                  t.expect(det_bareiss == bri(30) && det_faddeev == bri(30),
                           "det == 30 by both Bareiss and Faddeev-LeVerrier");
                  t.expect(det_bareiss == det_faddeev, "Bareiss det == Faddeev-LeVerrier det");

                  // Fully rational spectrum: sum == trace, product == det.
                  const auto eig = rational_eigenvalues(A).value();
                  std::int64_t total_mult = 0;
                  BigRational spec_sum;         // 0/1
                  BigRational spec_prod = bri(1);
                  for (const auto& [val, m] : eig) {
                      total_mult += m;
                      for (std::int64_t i = 0; i < m; ++i) {
                          spec_sum = spec_sum.add(val);
                          spec_prod = spec_prod.multiply(val);
                      }
                  }
                  t.expect(total_mult == 3, "all 3 eigenvalues are rational");
                  t.expect(eig.size() == 3 && eig[0].first == bri(2) && eig[1].first == bri(3) &&
                               eig[2].first == bri(5),
                           "sorted rational spectrum == {2,3,5}");
                  t.expect(spec_sum == bri(10), "sum of eigenvalues == trace == 10");
                  t.expect(spec_prod == det_bareiss, "product of eigenvalues == det == 30");

                  // Faddeev-LeVerrier inverse is a genuine two-sided inverse.
                  t.expect(A.multiply(inverse(A).value()).value() == BigMatrix::identity(3),
                           "A * A^-1 == I (Faddeev-LeVerrier inverse)");
              })
        .test("bigmatrix_determinant_exact_where_int64_matrix_overflows",
              [](TestContext& t) {
                  // diag(10^10, 10^10, 10^10): det == 10^30 dwarfs the int64 ceiling.
                  const std::int64_t big10 = 10000000000LL;
                  const auto m64 = mat({{big10, 0, 0}, {0, big10, 0}, {0, 0, big10}});
                  t.expect(!m64.determinant().has_value(),
                           "int64 Matrix determinant 10^30 overflows");

                  const auto mbig = BigMatrix::from_matrix(m64).value();  // exact promotion
                  const auto expected = BigRational::from_bigint(biv(10).pow(30));
                  t.expect(mbig.determinant().value() == expected,
                           "BigMatrix determinant == 10^30 exactly");
              })
        // ===================================================================
        // constants + bigfloat: high-precision transcendentals vs references.
        // ===================================================================
        .test("constants_agree_with_independent_references",
              [](TestContext& t) {
                  const std::int64_t P = 200;  // significant bits
                  t.expect(close(pi(P).value().to_double(), std::numbers::pi, 1e-12), "pi");
                  t.expect(close(e(P).value().to_double(), std::numbers::e, 1e-12), "e");
                  t.expect(close(ln2(P).value().to_double(), std::numbers::ln2, 1e-12), "ln2");
                  t.expect(close(ln10(P).value().to_double(), std::numbers::ln10, 1e-12), "ln10");
                  t.expect(close(sqrt2(P).value().to_double(), std::numbers::sqrt2, 1e-12),
                           "sqrt2");
                  t.expect(close(euler_mascheroni(P).value().to_double(), std::numbers::egamma,
                                 1e-12),
                           "Euler-Mascheroni gamma");
                  t.expect(close(golden_ratio(P).value().to_double(), std::numbers::phi, 1e-12),
                           "golden ratio");

                  // phi^2 == phi + 1 (the defining identity of the golden ratio).
                  const auto phi = golden_ratio(P).value();
                  const auto phi_sq = phi.multiply(phi, P).value();
                  const auto phi_1 = phi.add(bf(1, P), P).value();
                  t.expect(close(phi_sq.to_double(), phi_1.to_double(), 1e-12),
                           "phi^2 == phi + 1");
              })
        .test("constants_first_thirty_decimal_digits_are_correct",
              [](TestContext& t) {
                  // Request 35 digits (240-bit compute, ~72 correct digits) and compare the
                  // first 30 decimals, which rounding at digit 35 cannot perturb.
                  const std::int64_t P = 240;
                  t.expect(pi(P).value().to_string(35).substr(0, 32) ==
                               "3.141592653589793238462643383279",
                           "pi to 30 decimals");
                  t.expect(e(P).value().to_string(35).substr(0, 32) ==
                               "2.718281828459045235360287471352",
                           "e to 30 decimals");
                  t.expect(euler_mascheroni(P).value().to_string(35).substr(0, 32) ==
                               "0.577215664901532860606512090082",
                           "gamma to 30 decimals");
              })
        .test("bigfloat_correctly_rounded_roundtrips",
              [](TestContext& t) {
                  // Perfect squares: sqrt is exact.
                  t.expect(bf(4, 64).sqrt(64).value() == bf(2, 64), "sqrt(4) == 2 exactly");
                  t.expect(bf(9, 64).sqrt(64).value() == bf(3, 64), "sqrt(9) == 3 exactly");

                  // Dyadic add/multiply are exact (structural equality of canonical BigFloats).
                  const auto half = BigFloat::from_double(0.5, 64).value();
                  const auto quarter = BigFloat::from_double(0.25, 64).value();
                  const auto three_q = BigFloat::from_double(0.75, 64).value();
                  t.expect(half.multiply(half, 64).value() == quarter, "0.5 * 0.5 == 0.25 exact");
                  t.expect(quarter.add(three_q, 64).value() == bf(1, 64), "0.25 + 0.75 == 1 exact");

                  // sqrt(2)^2 rounds back to 2 within ~200 bits.
                  const auto r2 = bf(2, 200).sqrt(200).value();
                  const auto diff = r2.multiply(r2, 200).value().subtract(bf(2, 200), 200).value();
                  t.expect(std::fabs(diff.to_double()) < 1e-40, "sqrt(2)^2 ~ 2 to ~200 bits");

                  // Division by zero is on the railway.
                  t.expect(!bf(1, 64).divide(bf(0, 64), 64).has_value(),
                           "bigfloat divide by zero -> error");

                  // Batched reductions of exact integers.
                  std::vector<BigFloat> xs{bf(1, 64), bf(2, 64), bf(3, 64), bf(4, 64)};
                  std::vector<BigFloat> ys{bf(5, 64), bf(6, 64), bf(7, 64), bf(8, 64)};
                  t.expect(bigfloat_sum(std::span<const BigFloat>(xs), 64).value() == bf(10, 64),
                           "bigfloat_sum(1..4) == 10");
                  t.expect(bigfloat_dot(std::span<const BigFloat>(xs), std::span<const BigFloat>(ys),
                                        64)
                                   .value() == bf(70, 64),
                           "bigfloat_dot == 5+12+21+32 == 70");
              })
        // ===================================================================
        // doubledouble: error-free transforms and compensated reductions.
        // ===================================================================
        .test("doubledouble_efts_and_compensated_reductions",
              [](TestContext& t) {
                  // TwoSum recovers the low bit that a naive double add drops.
                  const auto ts = two_sum(1e16, 1.0);  // 1e16 + 1 rounds to 1e16, error 1
                  t.expect(ts.hi == 1e16 && ts.lo == 1.0, "two_sum(1e16,1) == {1e16, 1}");

                  // Compensated sum: {1e16, 1, -1e16} sums to exactly 1, not 0.
                  std::vector<double> cancel{1e16, 1.0, -1e16};
                  const auto s = dd_sum(std::span<const double>(cancel));
                  t.expect(s.to_double() == 1.0, "dd_sum compensates cancellation to exactly 1");
                  t.expect(s == dd_sum_scalar(std::span<const double>(cancel)),
                           "dd_sum SIMD path == scalar reference (bit-identical)");

                  // Exact integer dot product: 9 + 16 + 144 == 169, with a zero low word.
                  std::vector<double> a{3.0, 4.0, 12.0};
                  std::vector<double> b{3.0, 4.0, 12.0};
                  const auto d = dd_dot(std::span<const double>(a), std::span<const double>(b));
                  t.expect(d.to_double() == 169.0 && d.lo == 0.0, "dd_dot == 169 exactly");
                  t.expect(d == dd_dot_scalar(std::span<const double>(a),
                                              std::span<const double>(b)),
                           "dd_dot SIMD path == scalar reference (bit-identical)");

                  // Compensated dot with catastrophic cancellation: 1e16 + 1 - 1e16 == 1.
                  std::vector<double> ca{1e16, 1.0, 1.0};
                  std::vector<double> cb{1.0, 1.0, -1e16};
                  const auto cd = dd_dot(std::span<const double>(ca), std::span<const double>(cb));
                  t.expect(cd.to_double() == 1.0, "dd_dot compensated to exactly 1");

                  // sqrt to ~106 bits: sqrt(2)^2 is far closer to 2 than a plain double.
                  const auto r2 = DoubleDouble::from_double(2.0).sqrt().value();
                  const auto err = r2.multiply(r2).subtract(DoubleDouble::from_double(2.0));
                  t.expect(std::fabs(err.to_double()) < 1e-28, "dd sqrt(2)^2 ~ 2 to ~106 bits");
                  t.expect(DoubleDouble::from_double(16.0).sqrt().value().to_double() == 4.0,
                           "dd sqrt(16) == 4");

                  // Exact dyadic divide and the zero-divisor railway.
                  t.expect(DoubleDouble::from_double(1.0)
                                   .divide(DoubleDouble::from_double(4.0))
                                   .value() == DoubleDouble::from_double(0.25),
                           "dd 1/4 == 0.25 exact");
                  t.expect(!DoubleDouble::from_double(1.0)
                                .divide(DoubleDouble::from_double(0.0))
                                .has_value(),
                           "dd divide by zero -> error");
              })
        // ===================================================================
        // numbertheory: primality, CRT, modular inverse, Bezout, Jacobi.
        // ===================================================================
        .test("miller_rabin_classifies_primes_composites_and_carmichaels",
              [](TestContext& t) {
                  const std::uint64_t seed = 20260703;
                  for (const char* p : {"2", "3", "97", "7919", "104729", "2147483647"}) {
                      t.expect(is_probable_prime(bi(p), seed).value(), "prime classified prime");
                  }
                  // 561,1105,1729,2465 are Carmichael numbers (they fool the Fermat test but
                  // not Miller-Rabin); 2147483649 == 3 * 715827883.
                  for (const char* c : {"1", "100", "561", "1105", "1729", "2465", "2147483649"}) {
                      t.expect(!is_probable_prime(bi(c), seed).value(),
                               "composite/Carmichael classified composite");
                  }
                  t.expect(!is_probable_prime(bi("7919").multiply(bi("104729")), seed).value(),
                           "product of two primes is composite");
              })
        .test("crt_modinverse_extended_gcd_and_jacobi",
              [](TestContext& t) {
                  // CRT: x == 2 (3), 3 (5), 2 (7) -> x == 23 in [0, 105).
                  const auto x =
                      crt({bi("2"), bi("3"), bi("2")}, {bi("3"), bi("5"), bi("7")}).value();
                  t.expect(x == bi("23"), "CRT reconstruction == 23");
                  t.expect(x.mod(bi("3")).value() == bi("2") &&
                               x.mod(bi("5")).value() == bi("3") &&
                               x.mod(bi("7")).value() == bi("2"),
                           "CRT residues all satisfied");

                  // Modular inverse: 3^-1 == 4 (mod 11), and 3*4 == 1 (mod 11).
                  const auto inv3 = mod_inverse(bi("3"), bi("11")).value();
                  t.expect(inv3 == bi("4"), "3^-1 mod 11 == 4");
                  t.expect(bi("3").multiply(inv3).mod(bi("11")).value() == bi("1"),
                           "a * a^-1 == 1 (mod m)");
                  t.expect(!mod_inverse(bi("2"), bi("4")).has_value(),
                           "non-invertible element -> domain_error");

                  // Extended Euclid Bezout identity: 240x + 46y == gcd == 2.
                  const auto eg = extended_gcd(bi("240"), bi("46"));
                  t.expect(eg.g == bi("2"), "gcd(240,46) == 2");
                  t.expect(bi("240").multiply(eg.x).add(bi("46").multiply(eg.y)) == eg.g,
                           "Bezout: 240*x + 46*y == g");

                  // Jacobi symbol: (2/15)==1, (3/15)==0 (gcd>1), (2/7)==1; even n is an error.
                  t.expect(jacobi_symbol(bi("2"), bi("15")).value() == 1, "(2/15) == 1");
                  t.expect(jacobi_symbol(bi("3"), bi("15")).value() == 0, "(3/15) == 0");
                  t.expect(jacobi_symbol(bi("2"), bi("7")).value() == 1, "(2/7) == 1");
                  t.expect(!jacobi_symbol(bi("3"), bi("4")).has_value(),
                           "even modulus -> domain_error");
              })
        .run();
}
