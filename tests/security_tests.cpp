// Regression tests for the adversarial security audit (DoS / OOM / UB hardening).
// @author Olumuyiwa Oluwasanmi
//
// Each case pins a fix from the audit: an untrusted numeric/string input that USED to trigger a
// multi-GB allocation, an unbounded loop, or silent-NaN output must now be REFUSED honestly on
// the Result<T>/MathError (or std::optional) railway — fast, with no allocation and no hang. The
// suite is intentionally cheap to run: every guarded call returns an error in O(1), so a passing
// run also proves the guard fires before the expensive path is entered.
//
//   F1  bigdecimal  from_string scale bomb (10^|scale| materialization)
//   F2  finance     (1+r)^nper exponent bomb via growth/effect/growing_annuity_pv
//   F3  finance     Date year-walk DoS (Date::of range + ymd clamp)
//   F4  pricing     paths/steps OOM + trinomial 2*steps+1 int overflow
//   F5  portfolio   lu_solve_ridge dimension cap + NaN rejection
//   F7  finance     depreciation life-loop cap (db/ddb/vdb)

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.bigdecimal;
import nimblecas.finance;
import nimblecas.pricing;
import nimblecas.analytics;
import nimblecas.portfolio;
import nimblecas.testing;

using nimblecas::BigDecimal;
using nimblecas::BigRational;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace fin = nimblecas::finance;
namespace px = nimblecas::pricing;
using nimblecas::portfolio::lu_solve_ridge;

namespace {
auto q(std::string_view s) -> BigRational { return BigRational::from_string(s).value(); }
}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.security")
        // --- F1: BigDecimal scale bomb -------------------------------------------------------
        .test("F1 from_string rejects a scale inside int32 but past the DoS bound",
              [](TestContext& t) {
                  // These parse-succeeded before the fix (scale fits int32) then detonated on
                  // first quantize/to_string as a ~2e9-digit BigInt. Now refused at parse.
                  t.expect(!BigDecimal::from_string("1e-2000000000").has_value(),
                           "1e-2000000000 refused (scale +2e9)");
                  t.expect(!BigDecimal::from_string("1e2000000000").has_value(),
                           "1e2000000000 refused (scale -2e9)");
                  // Exponent == INT64_MIN: refused BEFORE the frac_digits-exponent subtraction,
                  // which would otherwise be signed-overflow UB (D2).
                  t.expect(!BigDecimal::from_string("1e-9223372036854775808").has_value(),
                           "INT64_MIN exponent refused without UB");
                  // A sane money scale still parses.
                  t.expect(BigDecimal::from_string("1.2345").has_value(), "normal scale still ok");
                  t.expect(BigDecimal::from_string("1e-6").has_value(), "scale at 1e-6 still ok");
              })
        // --- F2: (1+r)^nper exponent bomb ----------------------------------------------------
        .test("F2 TVM functions reject an attacker-scale period count",
              [](TestContext& t) {
                  // fv/pv/pmt route nper into BigRational::pow; 1e9 built a ~1GB rational.
                  t.expect(!fin::fv(q("1/10"), 1'000'000'000, q("0"), q("100")).has_value(),
                           "fv rejects nper=1e9");
                  t.expect(!fin::pv(q("1/10"), 1'000'000'000, q("0"), q("100")).has_value(),
                           "pv rejects nper=1e9");
                  t.expect(!fin::pmt(q("1/10"), 1'000'000'000, q("100"), q("0")).has_value(),
                           "pmt rejects nper=1e9");
                  t.expect(!fin::effect(q("1/10"), 2'000'000'000).has_value(),
                           "effect rejects npery=2e9");
                  t.expect(!fin::growing_annuity_pv(q("1/10"), q("1/20"), 1'000'000'000, q("100"))
                                .has_value(),
                           "growing_annuity_pv rejects nper=1e9");
                  // A real term still works (10% for 360 months).
                  t.expect(fin::pmt(q("1/200"), 360, q("200000"), q("0")).has_value(),
                           "a 360-period mortgage still prices");
              })
        // --- F3: Date year-walk DoS ----------------------------------------------------------
        .test("F3 Date::of bounds the year and ymd clamps a hostile serial",
              [](TestContext& t) {
                  t.expect(!fin::Date::of(2'000'000'000, 1, 1).has_value(),
                           "Date::of rejects year 2e9");
                  t.expect(!fin::Date::of(-2'000'000'000, 1, 1).has_value(),
                           "Date::of rejects year -2e9");
                  t.expect(fin::Date::of(2024, 1, 1).has_value(), "a normal date still constructs");
                  // A directly-assigned out-of-range serial must not drive a billion-iteration
                  // walk: ymd() clamps and returns promptly (we only assert it terminates).
                  const fin::Date hostile{.serial = 700'000'000'000LL};
                  auto [y, m, d] = hostile.ymd();
                  // The point is termination: the clamp caps the walk at a few thousand
                  // iterations (a hostile serial cannot drive a billion-iteration loop). The
                  // decomposition is still well-formed, landing just past the spreadsheet range.
                  t.expect(y > 0 && y < 20000 && m >= 1 && m <= 12 && d >= 1,
                           "ymd on a hostile serial terminates with a well-formed date");
              })
        // --- F4: pricing paths/steps OOM + trinomial overflow --------------------------------
        .test("F4 pricing caps paths/steps before allocating",
              [](TestContext& t) {
                  auto spec = px::OptionSpec{}
                                  .with_spot(100).with_strike(100).with_rate(0.05)
                                  .with_volatility(0.2).with_expiry(1.0);
                  // 2*steps+1 overflowed int; now refused.
                  t.expect(!px::trinomial_price(spec, 1'500'000'000, px::Exercise::american, {})
                                .has_value(),
                           "trinomial rejects steps=1.5e9");
                  // 200e6 paths * 101 = ~160GB grid; now refused.
                  t.expect(!px::longstaff_schwartz_american(spec, 200'000'000, 100, 7).has_value(),
                           "LSM rejects a 160GB grid");
                  t.expect(!px::monte_carlo_european(spec, 5'000'000'000ULL, 7).has_value(),
                           "MC rejects paths=5e9 (hang guard)");
                  // The hang guard must bound the PRODUCT paths*steps, not just each factor: two
                  // individually-in-range values (1e9 paths * 1e5 steps = 1e14 iters) must refuse.
                  t.expect(!px::monte_carlo_asian(spec, 1'000'000'000ULL, 100'000, 7).has_value(),
                           "Asian MC rejects paths*steps=1e14");
                  t.expect(!px::barrier_option_mc(spec, 90.0, false, 1'000'000'000ULL, 100'000, 7)
                                .has_value(),
                           "barrier MC rejects paths*steps=1e14");
                  // A modest product still prices.
                  t.expect(px::monte_carlo_asian(spec, 20'000, 20, 7).has_value(),
                           "a 20k-path/20-step Asian still prices");
                  // Sane sizes still price.
                  t.expect(px::trinomial_price(spec, 200, px::Exercise::american, {}).has_value(),
                           "trinomial with 200 steps still prices");
              })
        // --- F5: lu_solve_ridge dimension cap + NaN rejection --------------------------------
        .test("F5 lu_solve_ridge rejects a NaN matrix instead of returning NaN weights",
              [](TestContext& t) {
                  const double nan = std::numeric_limits<double>::quiet_NaN();
                  const std::vector<std::vector<double>> nan_cov{{nan, 0.0}, {0.0, 0.04}};
                  const std::array<double, 2> rhs{1.0, 1.0};
                  t.expect(!lu_solve_ridge(nan_cov, rhs, 0.0).has_value(),
                           "NaN covariance -> nullopt, never a NaN solution");
                  // A NaN in the RHS (e.g. a NaN expected-return into tangency_weights) must also
                  // be refused, not flow into silently-NaN weights (D3).
                  const std::vector<std::vector<double>> ok_cov{{0.04, 0.0}, {0.0, 0.09}};
                  const std::array<double, 2> nan_rhs{nan, 1.0};
                  t.expect(!lu_solve_ridge(ok_cov, nan_rhs, 0.0).has_value(),
                           "NaN rhs -> nullopt, never NaN weights");
                  // A finite well-posed system still solves.
                  const std::vector<std::vector<double>> ok{{4.0, 1.0}, {1.0, 3.0}};
                  t.expect(lu_solve_ridge(ok, rhs, 0.0).has_value(), "finite system still solves");
                  // A rank-deficient (positive-SEMI-definite) covariance leaves a tiny-POSITIVE
                  // Cholesky pivot; the relative pivot floor must REFUSE it, not emit unstable
                  // garbage weights that pass a bare s>0 test.
                  const std::vector<std::vector<double>> collinear{{0.04, 0.04}, {0.04, 0.04}};
                  t.expect(!nimblecas::analytics::min_variance_weights(collinear).has_value(),
                           "collinear (semidefinite) cov refused honestly, not silently solved");
                  // A well-conditioned PD covariance still solves.
                  const std::vector<std::vector<double>> pd{{0.04, 0.0}, {0.0, 0.09}};
                  t.expect(nimblecas::analytics::min_variance_weights(pd).has_value(),
                           "well-conditioned PD cov still solves");
              })
        // --- F7: depreciation life-loop cap --------------------------------------------------
        .test("F7 depreciation rejects an attacker-scale life",
              [](TestContext& t) {
                  const auto cost = q("10000"), salv = q("1000"), two = q("2");
                  t.expect(!fin::ddb(cost, salv, 1'000'000'000, 1, two).has_value(),
                           "ddb rejects life=1e9");
                  t.expect(!fin::vdb(cost, salv, 1'000'000'000, 0, 1, two, false).has_value(),
                           "vdb rejects life=1e9");
                  t.expect(!fin::db(10000.0, 1000.0, 1'000'000'000, 1, 12).has_value(),
                           "db rejects life=1e9");
                  t.expect(fin::ddb(cost, salv, 10, 1, two).has_value(), "a 10-year life still works");
              })
        .run();
}
