// Tests for nimblecas.fixedincome: sensitivities, cashflow schedules, coupon-date functions,
// odd-period pricing, maturity-paying securities, z-spread, yield adapters, FRNs, amortization.
// @author Olumuyiwa Oluwasanmi
//
// ORACLES. Each group pins a hand-checked invariant: a par bond prices at 100 and its DV01 ≈
// modified_duration * price * 1e-4; coup_days for a semiannual 30/360 bond == 180 and the
// before/after day counts sum to it; ODDFPRICE/ODDLPRICE with a REGULAR stub reproduce
// finance::bond_price on a 30/360 basis; z-spread against the bond's own flat curve is ~0; the
// key-rate durations sum to the effective duration; the amortization principal sums to the loan
// and the closing balance is exactly zero; a par floater values to par.

import std;
import nimblecas.core;
import nimblecas.finance;
import nimblecas.fixedincome;
import nimblecas.testing;

using nimblecas::MathError;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;
namespace fin = nimblecas::finance;
namespace fi = nimblecas::fixedincome;

namespace {

[[nodiscard]] auto D(int y, int m, int d) -> fin::Date { return fin::Date::of(y, m, d).value(); }
[[nodiscard]] auto close(double a, double b, double tol = 1e-7) -> bool { return std::abs(a - b) < tol; }

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.fixedincome")
        .test("par bond, DV01/PV01 and effective duration match modified duration",
              [](TestContext& t) {
                  const auto s = D(2020, 1, 1);
                  const auto m = D(2030, 1, 1);
                  const double c = 0.05;
                  // Par bond: coupon == yield priced on a coupon date == 100.
                  const double price =
                      fin::bond_price(s, m, c, c, 100.0, 2, fin::DayCount::thirty_360).value();
                  t.expect(close(price, 100.0, 1e-9), "par bond prices at 100");
                  const double mdur =
                      fin::bond_mduration(s, m, c, c, 2, fin::DayCount::thirty_360).value();
                  const double dv = fi::dv01(s, m, c, c, 100.0, 2, fin::DayCount::thirty_360).value();
                  const double pv = fi::pv01(s, m, c, c, 100.0, 2, fin::DayCount::thirty_360).value();
                  // DV01/PV01 ≈ modified_duration * price * 1e-4.
                  t.expect(close(dv, mdur * price * 1e-4, 5e-4), "DV01 ~ mdur*price*1e-4");
                  t.expect(close(pv, mdur * price * 1e-4, 5e-4), "PV01 ~ mdur*price*1e-4");
                  t.expect(dv > 0.0, "DV01 positive for an ordinary bond");
                  const double eff =
                      fi::duration_from_price(s, m, c, c, 100.0, 2, fin::DayCount::thirty_360).value();
                  t.expect(close(eff, mdur, 1e-3), "effective duration ~ modified duration");
              })
        .test("cfdates / cfamounts schedule",
              [](TestContext& t) {
                  const auto s = D(2020, 1, 1);
                  const auto m = D(2022, 1, 1);
                  auto dates = fi::cfdates(s, m, 2).value();
                  t.expect(dates.size() == 4, "semiannual 2y schedule has 4 payment dates");
                  t.expect(fin::coupon_num(s, m, 2).value() == 4, "matches COUPNUM");
                  t.expect(dates.back() == m, "last payment is at maturity");
                  auto cf = fi::cfamounts(s, m, 0.06, 2, fin::DayCount::thirty_360, 100.0).value();
                  t.expect(cf.amounts.size() == 4 && cf.dates.size() == 4, "4 parallel cashflows");
                  t.expect(close(cf.amounts[0], 3.0) && close(cf.amounts[1], 3.0), "regular coupon == 3");
                  t.expect(close(cf.amounts.back(), 103.0), "final = coupon + redemption");
                  double sum = 0.0;
                  for (double a : cf.amounts) { sum += a; }
                  t.expect(close(sum, 112.0), "total cashflow == 4*3 + 100");
                  // Short first stub: an issue mid-period prorates the first coupon below the regular.
                  auto stub = fi::cfamounts(s, m, 0.06, 2, fin::DayCount::thirty_360, 100.0, 100.0,
                                            D(2020, 5, 1)).value();
                  t.expect(stub.amounts[0] < 3.0 && stub.amounts[0] > 0.0, "short stub coupon < regular");
              })
        .test("coupon-date functions (exact integer days)",
              [](TestContext& t) {
                  const auto s = D(2020, 3, 15);
                  const auto m = D(2025, 1, 1);
                  t.expect(fi::coup_pcd(s, m, 2).value() == D(2020, 1, 1), "COUPPCD == 2020-01-01");
                  t.expect(fi::coup_ncd(s, m, 2).value() == D(2020, 7, 1), "COUPNCD == 2020-07-01");
                  const auto bs = fi::coup_days_bs(s, m, 2, fin::DayCount::thirty_360).value();
                  const auto nc = fi::coup_days_nc(s, m, 2, fin::DayCount::thirty_360).value();
                  const auto full = fi::coup_days(s, m, 2, fin::DayCount::thirty_360).value();
                  t.expect(full == 180, "COUPDAYS 30/360 semiannual == 180");
                  t.expect(bs == 74, "COUPDAYBS == 74");
                  t.expect(nc == 106, "COUPDAYSNC == 106");
                  t.expect(bs + nc == full, "days_bs + days_nc == days");
                  // Actual/365 reports true calendar days for the period around settlement.
                  const auto full_act = fi::coup_days(s, m, 2, fin::DayCount::actual_365).value();
                  t.expect(full_act == D(2020, 7, 1).serial - D(2020, 1, 1).serial,
                           "COUPDAYS actual == calendar days");
              })
        .test("ODDFPRICE with a regular first period equals the standard bond price",
              [](TestContext& t) {
                  const auto s = D(2020, 4, 1);
                  const auto m = D(2025, 1, 1);
                  const auto issue = D(2020, 1, 1);         // == first_coupon - one period (regular)
                  const auto first = D(2020, 7, 1);
                  const double std_price =
                      fin::bond_price(s, m, 0.06, 0.05, 100.0, 2, fin::DayCount::thirty_360).value();
                  const double oddf =
                      fi::odd_first_price(s, m, issue, first, 0.06, 0.05, 100.0, 2,
                                          fin::DayCount::thirty_360).value();
                  t.expect(close(oddf, std_price, 1e-6), "ODDFPRICE(regular stub) == bond_price");
                  // Odd (short) first stub round-trips through ODDFYIELD.
                  const auto issue2 = D(2020, 3, 1);        // 4-month short stub to first coupon
                  const double p = fi::odd_first_price(s, m, issue2, first, 0.06, 0.05, 100.0, 2,
                                                       fin::DayCount::thirty_360).value();
                  const double y = fi::odd_first_yield(s, m, issue2, first, 0.06, p, 100.0, 2,
                                                       fin::DayCount::thirty_360).value();
                  t.expect(close(y, 0.05, 1e-7), "ODDFYIELD inverts ODDFPRICE");
              })
        .test("ODDLPRICE with a regular last period equals the standard bond price",
              [](TestContext& t) {
                  const auto s = D(2020, 4, 1);
                  const auto m = D(2025, 1, 1);
                  const auto last_int = D(2024, 7, 1);      // == maturity - one period (regular last)
                  const double std_price =
                      fin::bond_price(s, m, 0.06, 0.05, 100.0, 2, fin::DayCount::thirty_360).value();
                  const double oddl = fi::odd_last_price(s, m, last_int, 0.06, 0.05, 100.0, 2,
                                                         fin::DayCount::thirty_360).value();
                  t.expect(close(oddl, std_price, 1e-6), "ODDLPRICE(regular stub) == bond_price");
                  // Odd (short) last stub round-trips through ODDLYIELD.
                  const auto m2 = D(2025, 4, 1);
                  const auto last2 = D(2025, 1, 1);          // 3-month final stub
                  const double p = fi::odd_last_price(s, m2, last2, 0.06, 0.05, 100.0, 2,
                                                      fin::DayCount::thirty_360).value();
                  const double y = fi::odd_last_yield(s, m2, last2, 0.06, p, 100.0, 2,
                                                      fin::DayCount::thirty_360).value();
                  t.expect(close(y, 0.05, 1e-7), "ODDLYIELD inverts ODDLPRICE");
              })
        .test("maturity-paying securities: PRICEMAT / YIELDMAT / ACCRINTM",
              [](TestContext& t) {
                  const auto issue = D(2020, 1, 1);
                  const auto s = D(2020, 7, 1);
                  const auto m = D(2021, 1, 1);
                  const double price =
                      fi::price_mat(s, m, issue, 0.05, 0.06, fin::DayCount::thirty_360).value();
                  t.expect(close(price, 99.44174757, 1e-6), "PRICEMAT hand value");
                  const double y =
                      fi::yield_mat(s, m, issue, 0.05, price, fin::DayCount::thirty_360).value();
                  t.expect(close(y, 0.06, 1e-9), "YIELDMAT inverts PRICEMAT");
                  const double ai =
                      fi::accrint_mat(issue, s, 0.05, fin::DayCount::thirty_360).value();
                  t.expect(close(ai, 2.5), "ACCRINTM == 100*rate*yearfrac == 2.5");
              })
        .test("z-spread against the bond's own flat curve is ~0, and key-rate durations sum",
              [](TestContext& t) {
                  const std::array<double, 5> times{1, 2, 3, 4, 5};
                  const std::array<double, 5> amounts{5, 5, 5, 5, 105};
                  const std::function<double(double)> flat = [](double tt) {
                      return std::exp(-0.05 * tt);
                  };
                  double target = 0.0;
                  for (std::size_t i = 0; i < 5; ++i) { target += amounts[i] * flat(times[i]); }
                  const double s = fi::z_spread(times, amounts, flat, target).value();
                  t.expect(close(s, 0.0, 1e-7), "z-spread against own flat curve ~ 0");
                  // A richer (higher) target price implies a negative spread.
                  const double s2 = fi::z_spread(times, amounts, flat, target + 2.0).value();
                  t.expect(s2 < 0.0, "higher price -> negative z-spread");

                  const double eff = fi::effective_duration(times, amounts, flat).value();
                  const std::array<double, 5> keys{1, 2, 3, 4, 5};
                  auto krd = fi::key_rate_durations(times, amounts, flat, keys).value();
                  t.expect(krd.size() == 5, "one duration per key tenor");
                  double sum = 0.0;
                  for (double k : krd) { sum += k; }
                  t.expect(close(sum, eff, 1e-6), "sum of key-rate durations == effective duration");
                  // A single key tenor (flat tent) equals the parallel effective duration.
                  const std::array<double, 1> onekey{2.5};
                  auto one = fi::key_rate_durations(times, amounts, flat, onekey).value();
                  t.expect(close(one[0], eff, 1e-6), "single key tenor == effective duration");
              })
        .test("yield-convention adapters round-trip and hit known values",
              [](TestContext& t) {
                  // simple<->compound over t periods.
                  const double simp = fi::compound_to_simple(0.05, 2.0).value();
                  t.expect(close(simp, 0.05125), "compound 5% over 2 -> simple 0.05125");
                  t.expect(close(fi::simple_to_compound(simp, 2.0).value(), 0.05, 1e-12),
                           "simple->compound inverts");
                  // annual<->semiannual nominal.
                  const double semi = fi::annual_to_semiannual_nominal(0.10).value();
                  t.expect(close(semi, 0.09761769634, 1e-9), "annual 10% -> semi nominal");
                  t.expect(close(fi::semiannual_to_annual_nominal(semi).value(), 0.10, 1e-12),
                           "semi->annual inverts");
                  // nominal_convert with equal frequencies is the identity.
                  t.expect(close(fi::nominal_convert(0.08, 2, 2).value(), 0.08, 1e-12),
                           "nominal_convert(m,m) == identity");
                  t.expect(!fi::nominal_convert(0.05, 0, 2).has_value(), "invalid frequency -> error");
              })
        .test("FRN par floater and level-payment amortization",
              [](TestContext& t) {
                  const std::array<double, 4> acc{0.5, 0.5, 0.5, 0.5};
                  const std::array<double, 4> fwd{0.04, 0.045, 0.05, 0.055};
                  std::array<double, 4> df{};
                  double p = 1.0;
                  for (std::size_t i = 0; i < 4; ++i) {
                      p /= (1.0 + fwd[i] * acc[i]);
                      df[i] = p;
                  }
                  // A par floater (margin 0) discounted off its own forwards values to par.
                  const double v = fi::frn_value(acc, fwd, df, 0.0, 100.0).value();
                  t.expect(close(v, 100.0, 1e-9), "par floater values to notional");
                  const double vp = fi::frn_value(acc, fwd, df, 0.01, 100.0).value();
                  t.expect(vp > v, "positive margin raises FRN value");

                  auto sched = fi::amortizing_schedule(10000.0, 0.01, 12).value();
                  t.expect(sched.interest.size() == 12, "12 periods");
                  t.expect(close(sched.interest[0], 100.0), "first interest == 10000*1%");
                  double prin_sum = 0.0;
                  for (double pr : sched.principal) { prin_sum += pr; }
                  t.expect(close(prin_sum, 10000.0, 1e-6), "principal sums to the loan amount");
                  t.expect(sched.balance.back() == 0.0, "final balance is exactly zero");
                  t.expect(sched.balance.front() < 10000.0, "balance decreases");
                  // Zero-rate loan is straight-line principal.
                  auto zero = fi::amortizing_schedule(1200.0, 0.0, 12).value();
                  t.expect(close(zero.payment, 100.0) && close(zero.principal[0], 100.0),
                           "zero-rate amortization is straight line");
                  t.expect(!fi::amortizing_schedule(1000.0, 0.05, 200001).has_value(),
                           "DoS-sized period count -> error");
              })
        .test("price_mat rejects a non-positive discount denominator",
              [](TestContext& t) {
                  // A deeply negative yield drives 1 + yield*t_to_maturity <= 0; the old code
                  // returned a large negative "price" as a valid value. It must now be refused.
                  const auto s = D(2020, 6, 1);
                  const auto m = D(2021, 1, 1);
                  const auto iss = D(2020, 1, 1);
                  const auto bad = fi::price_mat(s, m, iss, 0.06, -3.0, fin::DayCount::thirty_360);
                  t.expect(!bad.has_value(), "price_mat(den <= 0) -> error, not a negative price");
                  // A normal yield still prices fine.
                  const auto ok = fi::price_mat(s, m, iss, 0.06, 0.05, fin::DayCount::thirty_360);
                  t.expect(ok.has_value() && ok.value() > 0.0, "price_mat(normal yield) prices > 0");
              })
        .run();
}
