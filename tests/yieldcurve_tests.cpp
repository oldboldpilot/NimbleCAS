// Tests for nimblecas.yieldcurve: the zero-curve / term-structure toolkit.
// @author Olumuyiwa Oluwasanmi
//
// Every oracle here is either hand-computed or self-consistent:
//   * a flat 5% continuous curve has DF(t)=exp(-0.05 t) and forward == 5% everywhere;
//   * zero2disc . disc2zero and zero2fwd . fwd2zero are the identity;
//   * par2zero . zero2par is the identity (plus a hand par-yield value);
//   * bootstrapping a set of bonds then re-pricing them reproduces the input prices;
//   * a Nelson-Siegel fit of points sampled FROM a known NS curve recovers those betas;
//   * a Hull-White lattice calibrated to a curve reprices that curve's discount factors.

import std;
import nimblecas.core;
import nimblecas.finance;
import nimblecas.yieldcurve;
import nimblecas.testing;

using nimblecas::MathError;
using namespace nimblecas::yieldcurve;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto close(double a, double b, double tol = 1e-9) -> bool {
    return std::abs(a - b) < tol;
}

// A flat continuously-compounded 5% curve on a wide grid.
[[nodiscard]] auto flat5() -> Curve {
    return Curve::create({0.25, 1.0, 2.0, 3.0, 4.0, 5.0}, {0.05, 0.05, 0.05, 0.05, 0.05, 0.05},
                         Interp::linear_zero, Compounding::continuous)
        .value();
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.yieldcurve")
        .test("flat continuous curve: DF=exp(-0.05 t), zero=5%, forward=5%",
              [](TestContext& t) {
                  const Curve c = flat5();
                  t.expect(close(c.discount_factor(1.0).value(), std::exp(-0.05)),
                           "DF(1) == exp(-0.05)");
                  t.expect(close(c.discount_factor(2.5).value(), std::exp(-0.05 * 2.5)),
                           "DF(2.5) == exp(-0.125)");
                  t.expect(close(c.zero_rate(1.5).value(), 0.05), "zero(1.5) == 5%");
                  t.expect(close(c.zero_rate(3.7).value(), 0.05), "zero(3.7) == 5%");
                  t.expect(close(c.forward_rate(1.0, 2.0).value(), 0.05), "fwd(1,2) == 5%");
                  t.expect(close(c.forward_rate(0.5, 4.0).value(), 0.05), "fwd(0.5,4) == 5%");
                  // Out of range and degenerate inputs are honest errors, not extrapolations.
                  t.expect(!c.discount_factor(0.1).has_value() &&
                               c.discount_factor(0.1).error() == MathError::domain_error,
                           "DF below first pillar -> domain_error");
                  t.expect(!c.discount_factor(6.0).has_value(), "DF beyond last pillar -> error");
                  t.expect(!c.forward_rate(2.0, 1.0).has_value(), "fwd with t2<=t1 -> error");
                  t.expect(!Curve::create({}, {}, Interp::linear_zero, Compounding::continuous)
                                .has_value(),
                           "empty curve -> error");
                  t.expect(!Curve::create({1.0, 1.0}, {0.03, 0.04}, Interp::linear_zero,
                                          Compounding::continuous)
                                .has_value(),
                           "non-increasing times -> error");
              })
        .test("point conversions and zero2disc . disc2zero identity",
              [](TestContext& t) {
                  // Hand values under each compounding at t == 1.
                  t.expect(close(discount_from_zero(0.05, 1.0, Compounding::annual).value(),
                                 1.0 / 1.05, 1e-12),
                           "annual DF(5%,1) == 1/1.05");
                  t.expect(close(discount_from_zero(0.06, 1.0, Compounding::semiannual).value(),
                                 1.0 / (1.03 * 1.03), 1e-12),
                           "semiannual DF(6%,1) == 1/1.03^2");
                  t.expect(close(discount_from_zero(0.08, 1.0, Compounding::k_per_year, 4).value(),
                                 1.0 / std::pow(1.02, 4.0), 1e-12),
                           "quarterly DF(8%,1) == 1/1.02^4");
                  t.expect(close(discount_from_zero(0.05, 0.0, Compounding::continuous).value(), 1.0),
                           "DF at t=0 == 1");
                  // Round trip zero -> disc -> zero on an arbitrary grid.
                  const std::vector<double> times{0.5, 1.0, 2.0, 5.0, 10.0};
                  const std::vector<double> zeros{0.02, 0.025, 0.03, 0.035, 0.04};
                  auto disc = zero2disc(times, zeros, Compounding::semiannual, 2);
                  t.expect(disc.has_value(), "zero2disc ok");
                  auto back = disc2zero(times, disc.value(), Compounding::semiannual, 2);
                  t.expect(back.has_value(), "disc2zero ok");
                  bool id = true;
                  for (std::size_t i = 0; i < zeros.size(); ++i) {
                      id = id && close(back.value()[i], zeros[i], 1e-10);
                  }
                  t.expect(id, "disc2zero . zero2disc == identity");
              })
        .test("zero2fwd . fwd2zero identity, and flat forward == zero",
              [](TestContext& t) {
                  const std::vector<double> times{1.0, 2.0, 3.0, 4.0};
                  const std::vector<double> zeros{0.03, 0.035, 0.04, 0.045};
                  auto fwd = zero2fwd(times, zeros, Compounding::continuous);
                  t.expect(fwd.has_value(), "zero2fwd ok");
                  auto back = fwd2zero(times, fwd.value(), Compounding::continuous);
                  t.expect(back.has_value(), "fwd2zero ok");
                  bool id = true;
                  for (std::size_t i = 0; i < zeros.size(); ++i) {
                      id = id && close(back.value()[i], zeros[i], 1e-10);
                  }
                  t.expect(id, "fwd2zero . zero2fwd == identity");
                  // Flat continuous curve: every interval forward equals the flat zero.
                  const std::vector<double> flat{0.05, 0.05, 0.05, 0.05};
                  auto ff = zero2fwd(times, flat, Compounding::continuous).value();
                  bool allflat = true;
                  for (double f : ff) { allflat = allflat && close(f, 0.05, 1e-10); }
                  t.expect(allflat, "flat curve forwards all == 5%");
              })
        .test("bootstrap zbtprice reproduces input bond prices",
              [](TestContext& t) {
                  // Instruments on a half-year grid so all coupon dates fall on pillar times.
                  const std::vector<CouponBond> bonds{
                      {0.5, 0.00, 2, 100.0},  // 6-month bill
                      {1.0, 0.04, 2, 100.0},  // 1y, 4% semiannual
                      {1.5, 0.04, 2, 100.0},  // 1.5y
                      {2.0, 0.05, 2, 100.0},  // 2y, 5% semiannual
                  };
                  const std::vector<double> prices{98.0, 99.37, 98.73, 99.4125};
                  auto curve = zbtprice(bonds, prices, Compounding::continuous);
                  t.expect(curve.has_value(), "zbtprice bootstraps");
                  // Re-price each bond on the bootstrapped curve; must match the inputs.
                  const Curve& cv = curve.value();
                  auto reprice = [&](const CouponBond& b) -> double {
                      const double f = static_cast<double>(b.frequency);
                      const int np = static_cast<int>(std::llround(b.maturity * f));
                      double price = 0.0;
                      if (b.coupon_rate == 0.0) {
                          return b.face * cv.discount_factor(b.maturity).value();
                      }
                      const double coupon = b.face * b.coupon_rate / f;
                      for (int k = 1; k <= np; ++k) {
                          const double tk = b.maturity - static_cast<double>(np - k) / f;
                          price += (coupon + (k == np ? b.face : 0.0)) *
                                   cv.discount_factor(tk).value();
                      }
                      return price;
                  };
                  bool ok = true;
                  for (std::size_t i = 0; i < bonds.size(); ++i) {
                      ok = ok && close(reprice(bonds[i]), prices[i], 1e-6);
                  }
                  t.expect(ok, "re-priced bonds == input prices");
                  // Discount factors strictly decrease across the bootstrapped pillars.
                  auto df = cv.pillar_times();
                  bool mono = true;
                  double prev = 1.0;
                  for (double tt : df) {
                      const double d = cv.discount_factor(tt).value();
                      mono = mono && (d < prev) && (d > 0.0);
                      prev = d;
                  }
                  t.expect(mono, "bootstrapped discount factors decrease in (0,1)");
              })
        .test("bootstrap zbtyield from zero-coupon yields",
              [](TestContext& t) {
                  // Pure discount bills: DF(m) = (1+y)^-m, so the recovered curve DF must match.
                  const std::vector<CouponBond> bonds{
                      {1.0, 0.0, 1, 100.0}, {2.0, 0.0, 1, 100.0}, {3.0, 0.0, 1, 100.0}};
                  const std::vector<double> yields{0.03, 0.035, 0.04};
                  auto curve = zbtyield(bonds, yields, Compounding::continuous);
                  t.expect(curve.has_value(), "zbtyield bootstraps");
                  const Curve& cv = curve.value();
                  bool ok = true;
                  for (std::size_t i = 0; i < bonds.size(); ++i) {
                      const double expected = std::pow(1.0 + yields[i], -bonds[i].maturity);
                      ok = ok && close(cv.discount_factor(bonds[i].maturity).value(), expected, 1e-9);
                  }
                  t.expect(ok, "zbtyield DF == (1+y)^-m for bills");
              })
        .test("par2zero . zero2par identity and hand par value",
              [](TestContext& t) {
                  // Single-pillar hand check: par yield == exp(z)-1 for annual coupons at t=1.
                  auto par1 = zero2par(std::vector<double>{1.0}, std::vector<double>{0.05},
                                       Compounding::continuous, 1, 1);
                  t.expect(par1.has_value() && close(par1.value()[0], std::exp(0.05) - 1.0, 1e-12),
                           "par(1y, z=5%) == exp(0.05)-1");
                  // Round trip zeros -> par -> zeros on an annual grid.
                  const std::vector<double> times{1.0, 2.0, 3.0, 4.0};
                  const std::vector<double> zeros{0.03, 0.035, 0.04, 0.042};
                  auto par = zero2par(times, zeros, Compounding::continuous, 1, 1);
                  t.expect(par.has_value(), "zero2par ok");
                  auto back = par2zero(times, par.value(), Compounding::continuous, 1, 1);
                  t.expect(back.has_value(), "par2zero ok");
                  bool id = true;
                  for (std::size_t i = 0; i < zeros.size(); ++i) {
                      id = id && close(back.value()[i], zeros[i], 1e-9);
                  }
                  t.expect(id, "par2zero . zero2par == identity");
              })
        .test("Nelson-Siegel fit recovers betas of a sampled NS curve",
              [](TestContext& t) {
                  const NelsonSiegel truth{0.05, -0.02, 0.03, 1.5};
                  // t -> 0 limit is beta0 + beta1.
                  t.expect(close(truth.zero_rate(0.0).value(), 0.05 - 0.02, 1e-12),
                           "NS(0) == beta0 + beta1");
                  const std::vector<double> times{0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0};
                  std::vector<double> zeros;
                  for (double tt : times) { zeros.push_back(truth.zero_rate(tt).value()); }
                  auto fit = fit_nelson_siegel(times, zeros);
                  t.expect(fit.has_value(), "NS fit converges");
                  const NelsonSiegel& f = fit.value();
                  t.expect(close(f.beta0, 0.05, 5e-3) && close(f.beta1, -0.02, 5e-3) &&
                               close(f.beta2, 0.03, 5e-3) && close(f.tau, 1.5, 5e-2),
                           "recovered betas ~ truth");
                  // Reproduction of the sampled zeros is tight.
                  bool repro = true;
                  for (std::size_t i = 0; i < times.size(); ++i) {
                      repro = repro && close(f.zero_rate(times[i]).value(), zeros[i], 1e-3);
                  }
                  t.expect(repro, "NS fit reproduces sampled zeros");
                  // Too few points is a domain error (4 params).
                  t.expect(!fit_nelson_siegel(std::vector<double>{1.0, 2.0},
                                              std::vector<double>{0.03, 0.04})
                                .has_value(),
                           "NS with <4 points -> error");
              })
        .test("Svensson fit reproduces a sampled Svensson curve",
              [](TestContext& t) {
                  const Svensson truth{0.04, -0.02, 0.02, 0.01, 1.0, 5.0};
                  const std::vector<double> times{0.25, 0.5, 1.0, 2.0, 3.0,
                                                  5.0,  7.0, 10.0, 15.0, 20.0};
                  std::vector<double> zeros;
                  for (double tt : times) { zeros.push_back(truth.zero_rate(tt).value()); }
                  auto fit = fit_svensson(times, zeros);
                  t.expect(fit.has_value(), "Svensson fit converges");
                  const Svensson& f = fit.value();
                  bool repro = true;
                  for (std::size_t i = 0; i < times.size(); ++i) {
                      repro = repro && close(f.zero_rate(times[i]).value(), zeros[i], 1e-2);
                  }
                  t.expect(repro, "Svensson fit reproduces sampled zeros (betas non-unique)");
              })
        .test("interpolation modes and Hull-White lattice reprices the curve",
              [](TestContext& t) {
                  // Linear-on-zero vs log-linear-on-DF at an interior point.
                  auto lz = Curve::create({1.0, 2.0}, {0.03, 0.05}, Interp::linear_zero,
                                          Compounding::continuous)
                                .value();
                  t.expect(close(lz.zero_rate(1.5).value(), 0.04, 1e-9), "linear zero(1.5)==4%");
                  t.expect(close(lz.discount_factor(1.5).value(), std::exp(-0.04 * 1.5), 1e-9),
                           "linear DF(1.5)==exp(-0.06)");
                  auto ld = Curve::create({1.0, 2.0}, {0.03, 0.05}, Interp::loglinear_df,
                                          Compounding::continuous)
                                .value();
                  // ln DF at pillars are -0.03 and -0.10; midpoint is -0.065.
                  t.expect(close(ld.discount_factor(1.5).value(), std::exp(-0.065), 1e-9),
                           "loglinear DF(1.5)==exp(-0.065)");
                  // Pillars are reproduced exactly under both modes.
                  t.expect(close(ld.discount_factor(2.0).value(), std::exp(-0.10), 1e-9),
                           "loglinear DF(2) exact at pillar");

                  // Hull-White calibrated to the flat 5% curve reprices its discount factors.
                  const Curve c = flat5();
                  auto lat = build_hull_white(c, 0.1, 0.01, 0.5, 8);
                  t.expect(lat.has_value(), "Hull-White builds");
                  const HullWhiteLattice& hw = lat.value();
                  bool ok = true;
                  for (int i : {1, 2, 4, 6, 8}) {
                      const double tT = static_cast<double>(i) * 0.5;
                      ok = ok && close(hw.discount(tT).value(), std::exp(-0.05 * tT), 1e-7);
                  }
                  t.expect(ok, "lattice discount == curve DF (reprice by construction)");
                  // A dated cashflow priced on the lattice matches manual discounting.
                  const std::vector<double> ct{1.0, 2.0};
                  const std::vector<double> ca{3.0, 103.0};
                  const double expected =
                      3.0 * std::exp(-0.05) + 103.0 * std::exp(-0.10);
                  t.expect(close(hw.bond_price(ct, ca).value(), expected, 1e-6),
                           "lattice bond_price matches manual discount");
                  // Degenerate parameters are honest errors.
                  t.expect(!build_hull_white(c, 0.0, 0.01, 0.5, 8).has_value(),
                           "a==0 -> error");
                  t.expect(!hw.discount(0.3).has_value(), "off-node maturity -> error");
              })
        .test("calendar bridge reuses finance Date/DayCount",
              [](TestContext& t) {
                  using nimblecas::finance::Date;
                  using nimblecas::finance::DayCount;
                  const Date settle = Date::of(2020, 1, 1).value();
                  const std::vector<Date> pillars{Date::of(2021, 1, 1).value(),
                                                  Date::of(2022, 1, 1).value(),
                                                  Date::of(2025, 1, 1).value()};
                  auto times = times_from_dates(settle, pillars, DayCount::actual_365);
                  t.expect(times.has_value(), "times_from_dates ok");
                  // 2020 is a leap year: 366 days / 365 for the first pillar under actual/365.
                  t.expect(close(times.value()[0], 366.0 / 365.0, 1e-12),
                           "first year fraction == 366/365");
                  // Non-increasing / pre-settlement dates are rejected.
                  const std::vector<Date> bad{Date::of(2019, 1, 1).value()};
                  t.expect(!times_from_dates(settle, bad, DayCount::actual_365).has_value(),
                           "pillar before settlement -> error");
                  // The resulting times build a valid curve.
                  auto curve = Curve::create(times.value(), {0.02, 0.025, 0.03},
                                             Interp::loglinear_df, Compounding::annual);
                  t.expect(curve.has_value(), "curve from dated pillars ok");
              })
        .run();
}
