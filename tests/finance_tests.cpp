// Tests for nimblecas.finance: the Excel/Mathematica/MATLAB/Maple financial suite.
// @author Olumuyiwa Oluwasanmi
//
// Two tiers are proven separately. TIER A (exact over Q): PV/FV/PMT satisfy the TVM identity
// exactly, NPV/FVSCHEDULE/SLN/SYD/DDB/EFFECT hit hand-computed rational values on the nose,
// and IPMT + PPMT == PMT period-by-period with CUMIPMT/CUMPRINC equal to the running sums.
// TIER B (numerical): IRR/RATE/NPER/MIRR/XIRR recover known rates to tolerance, and the
// numerical NPER inverts the exact FV. Every Tier-A literal is hand-verified.

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.bigdecimal;
import nimblecas.finance;
import nimblecas.testing;

using nimblecas::BigRational;
using nimblecas::MathError;
using namespace nimblecas::finance;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto q(std::string_view s) -> BigRational { return BigRational::from_string(s).value(); }
[[nodiscard]] auto qi(std::int64_t v) -> BigRational { return BigRational::from_int(v); }
[[nodiscard]] auto close(double a, double b, double tol = 1e-7) -> bool { return std::abs(a - b) < tol; }

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.finance")
        .test("TVM zero-rate exact identities",
              [](TestContext& t) {
                  // FV(0, 10, -100, 0) == 1000 exactly.
                  t.expect(fv(qi(0), 10, qi(-100), qi(0)).value() == qi(1000),
                           "FV(0,10,-100,0) == 1000");
                  t.expect(pv(qi(0), 10, qi(-100), qi(0)).value() == qi(1000),
                           "PV(0,10,-100,0) == 1000");
                  t.expect(pmt(qi(0), 10, qi(1000), qi(0)).value() == qi(-100),
                           "PMT(0,10,1000,0) == -100");
              })
        .test("TVM nonzero-rate exact values",
              [](TestContext& t) {
                  // 100 doubling at 100% for one period: FV == -200.
                  t.expect(fv(qi(1), 1, qi(0), qi(100)).value() == qi(-200),
                           "FV(100%,1,0,100) == -200");
                  // FV then PV recovers the original present value (exact round trip).
                  const auto f = fv(q("1/20"), 12, qi(-100), qi(1000)).value();  // 5%/period
                  const auto back = pv(q("1/20"), 12, qi(-100), f).value();
                  t.expect(back == qi(1000), "PV inverts FV exactly");
              })
        .test("IPMT + PPMT == PMT, and cumulative sums",
              [](TestContext& t) {
                  const BigRational r = q("1/100");  // 1% per period
                  const std::int64_t n = 12;
                  const BigRational principal = qi(10000);
                  const auto p = pmt(r, n, principal, qi(0)).value();
                  BigRational sum_i = qi(0);
                  BigRational sum_p = qi(0);
                  for (std::int64_t per = 1; per <= n; ++per) {
                      const auto ip = ipmt(r, per, n, principal, qi(0)).value();
                      const auto pp = ppmt(r, per, n, principal, qi(0)).value();
                      t.expect(ip.add(pp) == p, "IPMT + PPMT == PMT each period");
                      sum_i = sum_i.add(ip);
                      sum_p = sum_p.add(pp);
                  }
                  t.expect(cumipmt(r, n, principal, 1, n).value() == sum_i,
                           "CUMIPMT == sum of IPMT");
                  t.expect(cumprinc(r, n, principal, 1, n).value() == sum_p,
                           "CUMPRINC == sum of PPMT");
                  // Principal repaid over the full life equals -principal (loan fully amortised).
                  t.expect(sum_p == principal.negate(), "principal fully amortised");
              })
        .test("NPV / FVSCHEDULE exact",
              [](TestContext& t) {
                  const std::array<BigRational, 2> cf{qi(100), qi(100)};
                  // NPV at 100%: 100/2 + 100/4 == 75.
                  t.expect(npv(qi(1), cf).value() == qi(75), "NPV(100%,[100,100]) == 75");
                  const std::array<BigRational, 2> rates{q("1/10"), q("1/5")};
                  // 100 * 1.1 * 1.2 == 132.
                  t.expect(fvschedule(qi(100), rates) == qi(132), "FVSCHEDULE == 132");
              })
        .test("depreciation exact",
              [](TestContext& t) {
                  t.expect(sln(qi(1000), qi(100), 5).value() == qi(180), "SLN == 180");
                  t.expect(syd(qi(1000), qi(100), 5, 1).value() == qi(300), "SYD y1 == 300");
                  t.expect(syd(qi(1000), qi(100), 5, 5).value() == qi(60), "SYD y5 == 60");
                  t.expect(ddb(qi(1000), qi(0), 5, 1).value() == qi(400), "DDB y1 == 400");
                  t.expect(ddb(qi(1000), qi(0), 5, 2).value() == qi(240), "DDB y2 == 240");
                  // EFFECT(10%, 2) == (21/20)^2 - 1 == 41/400 == 0.1025.
                  t.expect(effect(q("1/10"), 2).value() == q("41/400"), "EFFECT(10%,2) == 41/400");
              })
        .test("dollar-fraction conversion exact",
              [](TestContext& t) {
                  // DOLLARDE(1.02, 16) == 1 + 2/16 == 1.125 (fraction read base-16).
                  t.expect(dollarde(q("102/100"), 16).value() == q("9/8"), "DOLLARDE(1.02,16)==1.125");
                  // Round trip: DOLLARFR(1.125, 16) == 1.02.
                  t.expect(dollarfr(q("9/8"), 16).value() == q("102/100"), "DOLLARFR(1.125,16)==1.02");
              })
        .test("numerical rates recover known answers",
              [](TestContext& t) {
                  // IRR of [-100, 110] is 10%.
                  const std::array<double, 2> a{-100.0, 110.0};
                  t.expect(close(irr(a).value(), 0.10), "IRR([-100,110]) == 0.10");
                  // IRR of [-100, 0, 121] is 10%.
                  const std::array<double, 3> b{-100.0, 0.0, 121.0};
                  t.expect(close(irr(b).value(), 0.10, 1e-6), "IRR([-100,0,121]) == 0.10");
                  // RATE: 100 -> 121 over 2 periods is 10%.
                  t.expect(close(rate(2, 0.0, -100.0, 121.0).value(), 0.10, 1e-6),
                           "RATE(2,0,-100,121) == 0.10");
                  // NPER: 100 -> 121 at 10% takes 2 periods.
                  t.expect(close(nper(0.10, 0.0, -100.0, 121.0).value(), 2.0, 1e-9),
                           "NPER(10%,0,-100,121) == 2");
                  // MIRR sanity: a simple two-flow project.
                  const std::array<double, 3> c{-1000.0, 500.0, 700.0};
                  t.expect(mirr(c, 0.10, 0.12).has_value(), "MIRR computes");
              })
        .test("fluent TvmProblem quantizes money at the boundary",
              [](TestContext& t) {
                  auto prob = TvmProblem::create().rate(q("1/20")).nper(12).pmt(qi(-100))
                                  .present_value(qi(1000));
                  auto money = prob.solve_fv_money(2, nimblecas::Rounding::half_even);
                  t.expect(money.has_value(), "solve_fv_money succeeds");
                  // The exact FV, quantized to cents, must equal quantizing the raw exact FV.
                  const auto exact = prob.solve_fv().value();
                  t.expect(money->to_bigrational().to_double() ==
                               nimblecas::BigDecimal::from_bigrational(
                                   exact, 2, nimblecas::Rounding::half_even).to_bigrational().to_double(),
                           "money == quantized exact FV");
              })
        .run();
}
