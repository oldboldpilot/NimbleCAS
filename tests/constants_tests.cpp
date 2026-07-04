// Tests for nimblecas.constants: arbitrary-precision classical constants.
// @author Olumuyiwa Oluwasanmi
//
// The constants are transcendental / irrational: there is no exact rational answer, only a
// BigFloat rounded to the requested precision. These tests therefore assert the LEADING
// decimal digits against the well-known reference values (35 fractional digits, well below
// the 256-bit working precision so the compared digits are exact), check that raising the
// precision keeps those digits stable, verify a couple of defining identities, and confirm
// the prec <= 0 domain_error path.

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigfloat;
import nimblecas.constants;
import nimblecas.testing;

using nimblecas::BigFloat;
using nimblecas::MathError;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// The first 35 fractional digits printed via to_string(45), i.e. "X." + 35 digits. Printing
// 45 digits at 256 bits keeps the first 35 exact (far from the final printed-rounding place).
[[nodiscard]] auto lead(const BigFloat& x) -> std::string { return x.to_string(45).substr(0, 37); }

}  // namespace

auto main() -> int {
    constexpr std::int64_t p = 256;
    return TestSuite("nimblecas.constants")
        .test("pi_matches_reference_digits",
              [](TestContext& t) {
                  const auto v = nimblecas::pi(p).value();
                  t.expect(lead(v) == "3.14159265358979323846264338327950288",
                           "pi matches its first 35 fractional digits");
              })
        .test("e_matches_reference_digits",
              [](TestContext& t) {
                  const auto v = nimblecas::e(p).value();
                  t.expect(lead(v) == "2.71828182845904523536028747135266249",
                           "e matches its first 35 fractional digits");
              })
        .test("euler_mascheroni_matches_reference_digits",
              [](TestContext& t) {
                  // The correctness-critical one: verify Brent-McMillan against the known
                  // gamma = 0.5772156649015328606065120900824024310...
                  const auto v = nimblecas::euler_mascheroni(p).value();
                  t.expect(lead(v) == "0.57721566490153286060651209008240243",
                           "gamma matches its first 35 fractional digits");
              })
        .test("ln2_matches_reference_digits",
              [](TestContext& t) {
                  const auto v = nimblecas::ln2(p).value();
                  t.expect(lead(v) == "0.69314718055994530941723212145817656",
                           "ln2 matches its first 35 fractional digits");
              })
        .test("ln10_matches_reference_digits",
              [](TestContext& t) {
                  const auto v = nimblecas::ln10(p).value();
                  t.expect(lead(v) == "2.30258509299404568401799145468436420",
                           "ln10 matches its first 35 fractional digits");
              })
        .test("sqrt2_matches_reference_digits",
              [](TestContext& t) {
                  const auto v = nimblecas::sqrt2(p).value();
                  t.expect(lead(v) == "1.41421356237309504880168872420969807",
                           "sqrt2 matches its first 35 fractional digits");
              })
        .test("golden_ratio_matches_reference_digits",
              [](TestContext& t) {
                  const auto v = nimblecas::golden_ratio(p).value();
                  t.expect(lead(v) == "1.61803398874989484820458683436563811",
                           "golden ratio matches its first 35 fractional digits");
              })
        .test("catalan_matches_reference_digits",
              [](TestContext& t) {
                  const auto v = nimblecas::catalan(p).value();
                  t.expect(lead(v) == "0.91596559417721901505460351493238411",
                           "Catalan's constant matches its first 35 fractional digits");
              })
        .test("increasing_precision_keeps_leading_digits_stable",
              [](TestContext& t) {
                  // The first 30 fractional digits must not move as precision rises.
                  const auto pi128 = nimblecas::pi(128).value().to_string(40).substr(0, 32);
                  const auto pi256 = nimblecas::pi(256).value().to_string(40).substr(0, 32);
                  const auto pi512 = nimblecas::pi(512).value().to_string(40).substr(0, 32);
                  t.expect(pi128 == pi256 && pi256 == pi512, "pi is stable across 128/256/512 bits");

                  const auto g128 = nimblecas::euler_mascheroni(128).value().to_string(40).substr(0, 32);
                  const auto g256 = nimblecas::euler_mascheroni(256).value().to_string(40).substr(0, 32);
                  t.expect(g128 == g256, "gamma is stable across 128/256 bits");

                  const auto e128 = nimblecas::e(128).value().to_string(40).substr(0, 32);
                  const auto e256 = nimblecas::e(256).value().to_string(40).substr(0, 32);
                  t.expect(e128 == e256, "e is stable across 128/256 bits");
              })
        .test("sqrt2_squared_rounds_back_to_two",
              [](TestContext& t) {
                  const std::int64_t prec = 256;
                  const auto r = nimblecas::sqrt2(prec).value();
                  const auto sq = r.multiply(r, prec).value();
                  // Dropping a few low bits collapses the tiny sqrt round-off, so sqrt2^2 == 2.
                  const auto two = BigFloat::from_i64(2, prec).value();
                  t.expect(sq.with_precision(prec - 8).value() == two.with_precision(prec - 8).value(),
                           "sqrt2^2 rounds back to 2");
              })
        .test("golden_ratio_satisfies_phi_squared_equals_phi_plus_one",
              [](TestContext& t) {
                  const std::int64_t prec = 256;
                  const auto phi = nimblecas::golden_ratio(prec).value();
                  const auto phi_sq = phi.multiply(phi, prec).value();
                  const auto one = BigFloat::from_i64(1, prec).value();
                  const auto phi_plus_one = phi.add(one, prec).value();
                  // phi^2 = phi + 1 by definition of the golden ratio (drop low bits for the
                  // shared rounding slack).
                  t.expect(phi_sq.with_precision(prec - 8).value() ==
                               phi_plus_one.with_precision(prec - 8).value(),
                           "phi^2 == phi + 1");
              })
        .test("pi_over_four_equals_arctan_identity_bounds",
              [](TestContext& t) {
                  // Coarse sanity: 3.14159 < pi < 3.14160 at low precision.
                  const auto pi64 = nimblecas::pi(64).value();
                  t.expect(pi64 > BigFloat::from_string("3.14159", 64).value(), "pi > 3.14159");
                  t.expect(pi64 < BigFloat::from_string("3.14160", 64).value(), "pi < 3.14160");
                  // ln10 = ln2 + ln5 sanity via ln10 > 3*ln2 (since ln10 = 3ln2 + ln(5/4)).
                  const auto l2 = nimblecas::ln2(128).value();
                  const auto l10 = nimblecas::ln10(128).value();
                  const auto three_l2 = l2.multiply(BigFloat::from_i64(3, 128).value(), 128).value();
                  t.expect(l10 > three_l2, "ln10 > 3*ln2");
              })
        .test("non_positive_precision_is_domain_error",
              [](TestContext& t) {
                  t.expect(nimblecas::pi(0).error() == MathError::domain_error,
                           "pi(0) -> domain_error");
                  t.expect(nimblecas::e(-5).error() == MathError::domain_error,
                           "e(-5) -> domain_error");
                  t.expect(nimblecas::euler_mascheroni(0).error() == MathError::domain_error,
                           "gamma(0) -> domain_error");
                  t.expect(nimblecas::catalan(-1).error() == MathError::domain_error,
                           "catalan(-1) -> domain_error");
                  t.expect(nimblecas::ln2(0).error() == MathError::domain_error,
                           "ln2(0) -> domain_error");
                  t.expect(nimblecas::sqrt2(0).error() == MathError::domain_error,
                           "sqrt2(0) -> domain_error");
                  t.expect(nimblecas::golden_ratio(0).error() == MathError::domain_error,
                           "golden_ratio(0) -> domain_error");
                  t.expect(nimblecas::ln10(0).error() == MathError::domain_error,
                           "ln10(0) -> domain_error");
              })
        .run();
}
