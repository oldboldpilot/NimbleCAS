// Tests for nimblecas.doubledouble: double-double (~106-bit) arithmetic and its
// SIMD-batched compensated reductions.
// @author Olumuyiwa Oluwasanmi
//
// Covers: the error-free transforms (two_sum / two_prod / quick_two_sum) recover the
// exact rounding error; a double-double distinguishes 1 + 1e-18 that a plain double
// cannot; compensated dd_sum / dd_dot beat naive summation on ill-conditioned input; the
// dispatched SIMD path is bit-for-bit identical to the scalar reference (Rule 55); sqrt(2)
// and 1/3 agree with reference to ~30 digits; division-by-zero and sqrt(-1) are railway
// domain errors.

import std;
import nimblecas.core;
import nimblecas.doubledouble;
import nimblecas.simd;
import nimblecas.testing;

using nimblecas::DoubleDouble;
using nimblecas::MathError;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Deterministic pseudo-random doubles in ~[-2, 2) (no <random> global state).
auto fill(std::vector<double>& v, std::uint64_t seed) -> void {
    std::uint64_t s = seed;
    for (double& x : v) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;  // LCG
        x = static_cast<double>(static_cast<std::int64_t>(s)) / static_cast<double>(1ull << 62);
    }
}

auto bits_equal(double a, double b) -> bool {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

auto dd_bits_equal(DoubleDouble a, DoubleDouble b) -> bool {
    return bits_equal(a.hi, b.hi) && bits_equal(a.lo, b.lo);
}

auto abs_double(double x) -> double { return x < 0.0 ? -x : x; }

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.doubledouble")
        .test("reports_backend",
              [](TestContext& t) {
                  std::println("  active SIMD ISA: {}",
                               nimblecas::simd::to_string_view(nimblecas::simd::active_isa()));
                  std::println("  dd batched backend: {}", nimblecas::batched_backend());
                  t.expect(true, "backend is queryable");
              })

        // --- error-free transforms recover the exact rounding error -------------
        .test("two_sum_captures_the_lost_low_bits",
              [](TestContext& t) {
                  // 1e-20 is far below 0.5*ulp(1) ~ 1.1e-16, so a plain double drops it...
                  t.expect(bits_equal(1.0 + 1e-20, 1.0), "plain double loses 1e-20");
                  // ...but two_sum keeps it exactly in lo.
                  const DoubleDouble s = nimblecas::two_sum(1.0, 1e-20);
                  t.expect(bits_equal(s.hi, 1.0), "two_sum hi == 1.0");
                  t.expect(bits_equal(s.lo, 1e-20), "two_sum lo == 1e-20 (captured error)");

                  // A round-to-even case: 1 + 2^-53 rounds to 1.0, error 2^-53.
                  const double half_ulp = std::ldexp(1.0, -53);
                  const DoubleDouble h = nimblecas::two_sum(1.0, half_ulp);
                  t.expect(bits_equal(h.hi, 1.0), "1 + 2^-53 rounds to 1.0");
                  t.expect(bits_equal(h.lo, half_ulp), "two_sum lo == 2^-53");

                  // Exactly representable sum: no error.
                  const DoubleDouble ex = nimblecas::two_sum(3.0, 7.0);
                  t.expect(dd_bits_equal(ex, {10.0, 0.0}), "two_sum(3,7) == {10,0}");

                  // quick_two_sum agrees when |a| >= |b|.
                  const DoubleDouble q = nimblecas::quick_two_sum(1.0, 1e-20);
                  t.expect(dd_bits_equal(q, {1.0, 1e-20}), "quick_two_sum(1,1e-20) == {1,1e-20}");
              })

        .test("two_prod_is_an_exact_product",
              [](TestContext& t) {
                  // a = 1 + 2^-27 is exactly representable; a*a = 1 + 2^-26 + 2^-54.
                  // The 2^-54 tail is 1/4 ulp(1+2^-26) and is exactly what lo must hold.
                  const double a = 1.0 + std::ldexp(1.0, -27);
                  const DoubleDouble p = nimblecas::two_prod(a, a);
                  t.expect(bits_equal(p.hi, 1.0 + std::ldexp(1.0, -26)), "two_prod hi == 1+2^-26");
                  t.expect(bits_equal(p.lo, std::ldexp(1.0, -54)), "two_prod lo == 2^-54");

                  const DoubleDouble ex = nimblecas::two_prod(3.0, 7.0);
                  t.expect(dd_bits_equal(ex, {21.0, 0.0}), "two_prod(3,7) == {21,0}");

                  // Definitional exactness over random inputs: lo == fma(a,b,-hi) exactly
                  // (the FMA and Dekker-split paths both yield this exact error).
                  std::vector<double> xs(64);
                  std::vector<double> ys(64);
                  fill(xs, 111u);
                  fill(ys, 222u);
                  bool all_exact = true;
                  for (std::size_t i = 0; i < xs.size(); ++i) {
                      const DoubleDouble pr = nimblecas::two_prod(xs[i], ys[i]);
                      if (!bits_equal(pr.hi, xs[i] * ys[i]) ||
                          !bits_equal(pr.lo, std::fma(xs[i], ys[i], -pr.hi))) {
                          all_exact = false;
                      }
                  }
                  t.expect(all_exact, "two_prod hi==a*b and lo==fma(a,b,-hi) for all samples");
              })

        // --- double-double holds precision a plain double cannot ----------------
        .test("distinguishes_one_plus_1e18",
              [](TestContext& t) {
                  t.expect(bits_equal(1.0 + 1e-18, 1.0), "double: 1 + 1e-18 == 1");
                  const DoubleDouble d = nimblecas::two_sum(1.0, 1e-18);
                  t.expect(bits_equal(d.hi, 1.0), "dd hi == 1.0");
                  t.expect(!bits_equal(d.lo, 0.0), "dd lo != 0 (extra precision retained)");
                  t.expect(d != DoubleDouble::from_double(1.0), "dd(1+1e-18) != dd(1)");
                  // subtracting 1 recovers the 1e-18 that the double had discarded
                  const DoubleDouble back = d.subtract(DoubleDouble::from_double(1.0));
                  t.expect(abs_double(back.to_double() - 1e-18) < 1e-33, "dd - 1 == 1e-18");
              })

        // --- compensated reductions beat naive summation ------------------------
        .test("dd_sum_beats_naive_on_ill_conditioned_input",
              [](TestContext& t) {
                  const std::array<double, 3> x{1e16, 1.0, -1e16};  // true sum is 1
                  double naive = 0.0;
                  for (double v : x) naive += v;
                  t.expect(bits_equal(naive, 0.0), "naive double sum collapses to 0");

                  const DoubleDouble s = nimblecas::dd_sum(x);
                  t.expect(bits_equal(s.to_double(), 1.0), "dd_sum recovers 1");
                  t.expect(dd_bits_equal(s, nimblecas::dd_sum_scalar(x)), "dd_sum == scalar ref");
              })

        .test("dd_dot_beats_naive_on_ill_conditioned_input",
              [](TestContext& t) {
                  const std::array<double, 3> a{1e16, 1.0, -1e16};
                  const std::array<double, 3> b{1.0, 1.0, 1.0};  // true dot is 1
                  double naive = 0.0;
                  for (std::size_t i = 0; i < a.size(); ++i) naive += a[i] * b[i];
                  t.expect(bits_equal(naive, 0.0), "naive double dot collapses to 0");

                  const DoubleDouble d = nimblecas::dd_dot(a, b);
                  t.expect(bits_equal(d.to_double(), 1.0), "dd_dot recovers 1");
                  t.expect(dd_bits_equal(d, nimblecas::dd_dot_scalar(a, b)), "dd_dot == scalar ref");
              })

        .test("dd_sum_ill_conditioned_through_the_simd_lanes",
              [](TestContext& t) {
                  // The n=3 cancellation cases above stay in the scalar tail (n < 4 lanes). Use
                  // n=16 (four full 4-lane blocks) so the VECTORIZED accumulation itself carries
                  // the cancellation, verifying compensated ACCURACY on the SIMD path — not just
                  // that SIMD==scalar. Pattern: i%4==0 -> +1e16, i%4==2 -> -1e16, else 1.0, so
                  // the 1e16 terms cancel exactly and the true sum is the count of 1.0s (8).
                  std::vector<double> x(16);
                  for (std::size_t i = 0; i < x.size(); ++i) {
                      x[i] = (i % 4 == 0) ? 1e16 : (i % 4 == 2) ? -1e16 : 1.0;
                  }
                  double naive = 0.0;
                  for (double v : x) naive += v;
                  t.expect(!bits_equal(naive, 8.0), "naive double sum loses the small terms");

                  const DoubleDouble s = nimblecas::dd_sum(x);
                  t.expect(bits_equal(s.to_double(), 8.0),
                           "dd_sum recovers 8 exactly through the SIMD lanes");
                  t.expect(dd_bits_equal(s, nimblecas::dd_sum_scalar(x)),
                           "SIMD dd_sum == scalar ref on the ill-conditioned n=16 input");
              })

        // --- SIMD path == scalar path, bit-for-bit, across ragged sizes ---------
        .test("simd_path_matches_scalar_bit_for_bit",
              [](TestContext& t) {
                  for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{3},
                                        std::size_t{4}, std::size_t{5}, std::size_t{7},
                                        std::size_t{8}, std::size_t{16}, std::size_t{17},
                                        std::size_t{1000}}) {
                      std::vector<double> a(n);
                      std::vector<double> b(n);
                      fill(a, 900u + static_cast<std::uint32_t>(n));
                      fill(b, 700u + static_cast<std::uint32_t>(n));

                      t.expect(dd_bits_equal(nimblecas::dd_sum(a), nimblecas::dd_sum_scalar(a)),
                               std::format("dd_sum simd==scalar n={}", n));
                      t.expect(dd_bits_equal(nimblecas::dd_dot(a, b),
                                             nimblecas::dd_dot_scalar(a, b)),
                               std::format("dd_dot simd==scalar n={}", n));
                      t.expect(dd_bits_equal(nimblecas::dd_poly_eval(a, 1.5),
                                             nimblecas::dd_poly_eval_scalar(a, 1.5)),
                               std::format("dd_poly simd==scalar n={}", n));
                  }
              })

        // --- polynomial evaluation is correct -----------------------------------
        .test("dd_poly_eval_matches_known_values",
              [](TestContext& t) {
                  // 1 - 2x + x^2 = (x-1)^2 ; coeffs indexed by power.
                  const std::array<double, 3> c{1.0, -2.0, 1.0};
                  t.expect(bits_equal(nimblecas::dd_poly_eval(c, 3.0).to_double(), 4.0),
                           "(3-1)^2 == 4");
                  t.expect(bits_equal(nimblecas::dd_poly_eval(c, 1.0).to_double(), 0.0),
                           "(1-1)^2 == 0 (no cancellation error)");
                  // Empty polynomial is the zero function.
                  t.expect(nimblecas::dd_poly_eval(std::span<const double>{}, 5.0).is_zero(),
                           "empty poly evaluates to 0");
              })

        // --- sqrt and divide agree with reference to ~30 digits -----------------
        .test("sqrt_and_reciprocal_reach_106_bits",
              [](TestContext& t) {
                  const auto r = DoubleDouble::from_double(2.0).sqrt();
                  t.expect(r.has_value(), "sqrt(2) ok");
                  if (r) {
                      t.expect(bits_equal(r->hi, std::sqrt(2.0)), "sqrt(2) hi == double sqrt");
                      t.expect(!bits_equal(r->lo, 0.0), "sqrt(2) carries extra precision");
                      // Round-trip: (sqrt 2)^2 == 2 to ~1e-31, i.e. ~30 correct digits.
                      const DoubleDouble sq = r->multiply(*r);
                      const DoubleDouble err = sq.subtract(DoubleDouble::from_double(2.0));
                      t.expect(abs_double(err.to_double()) < 1e-30, "(sqrt 2)^2 == 2 to ~30 digits");
                  }

                  const auto third = DoubleDouble::from_double(1.0).divide(
                      DoubleDouble::from_double(3.0));
                  t.expect(third.has_value(), "1/3 ok");
                  if (third) {
                      t.expect(bits_equal(third->hi, 1.0 / 3.0), "1/3 hi == double 1/3");
                      t.expect(!bits_equal(third->lo, 0.0), "1/3 carries extra precision");
                      const DoubleDouble back = third->multiply(DoubleDouble::from_double(3.0));
                      const DoubleDouble err = back.subtract(DoubleDouble::from_double(1.0));
                      t.expect(abs_double(err.to_double()) < 1e-30, "(1/3)*3 == 1 to ~30 digits");
                      // to_string must show the extra precision: many more 3s than a double.
                      const std::string s = third->to_string();
                      t.expect(s.find("3.3333333333333333333") == 0,
                               std::format("1/3 to_string shows >19 threes: {}", s));
                  }
              })

        // --- railway domain errors ----------------------------------------------
        .test("domain_errors_are_railway_typed",
              [](TestContext& t) {
                  const auto dz = DoubleDouble::from_double(1.0).divide(
                      DoubleDouble::from_double(0.0));
                  t.expect(!dz.has_value(), "1/0 fails");
                  t.expect(!dz.has_value() && dz.error() == MathError::division_by_zero,
                           "1/0 -> division_by_zero");

                  const auto sn = DoubleDouble::from_double(-1.0).sqrt();
                  t.expect(!sn.has_value(), "sqrt(-1) fails");
                  t.expect(!sn.has_value() && sn.error() == MathError::domain_error,
                           "sqrt(-1) -> domain_error");

                  t.expect(DoubleDouble::from_double(0.0).sqrt().value_or(
                               DoubleDouble{1.0, 0.0}) == (DoubleDouble{0.0, 0.0}),
                           "sqrt(0) == 0");
              })
        .run();
}
