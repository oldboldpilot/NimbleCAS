// Tests for nimblecas.optstrat: option trading strategies + expiry-P&L analytics.
// @author Olumuyiwa Oluwasanmi
//
// The expiry P&L of a vanilla option book is EXACTLY piecewise-linear in the terminal
// price, so every check here is a HAND-VERIFIED exact value of the analytics engine — net
// premium (debit/credit), max profit, max loss, and the breakeven set — across the
// standard structures (bull call spread, straddle, covered call, protective put, long
// butterfly, iron condor). Unbounded directions are asserted as unbounded, not as a
// fabricated finite number. The Greeks bridge is checked against pricing's Black-Scholes.

import std;
import nimblecas.core;
import nimblecas.pricing;
import nimblecas.optstrat;
import nimblecas.testing;

using namespace nimblecas::optstrat;
using nimblecas::pricing::OptionSpec;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {
[[nodiscard]] auto close(double a, double b, double tol) -> bool { return std::abs(a - b) < tol; }
[[nodiscard]] auto has_be(const StrategyAnalytics& a, double s) -> bool {
    return std::ranges::any_of(a.breakevens, [&](double e) { return std::abs(e - s) < 1e-6; });
}
}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.optstrat")
        .test("bull call spread: bounded debit structure",
              [](TestContext& t) {
                  // Long 100 call @5, short 110 call @2. Net debit 3.
                  const auto s = bull_call_spread(100.0, 5.0, 110.0, 2.0);
                  const auto a = s.analytics();
                  t.expect(close(a.net_premium, 3.0, 1e-9), "net debit == 3");
                  t.expect(!a.unbounded_profit && !a.unbounded_loss, "bounded both ways");
                  t.expect(close(a.max_profit, 7.0, 1e-9), "max profit == width − debit == 7");
                  t.expect(close(a.max_loss, -3.0, 1e-9), "max loss == −debit == −3");
                  t.expect(a.breakevens.size() == 1 && has_be(a, 103.0), "breakeven == 103");
                  // Spot checks of the payoff/pnl curve.
                  t.expect(close(s.pnl_at(95.0), -3.0, 1e-9), "below lower strike: −debit");
                  t.expect(close(s.pnl_at(115.0), 7.0, 1e-9), "above upper strike: max profit");
              })
        .test("bear put spread is a bearish DEBIT (long hi / short lo put)",
              [](TestContext& t) {
                  // Long 100 put @7, short 90 put @2 -> net debit 5, profits as price falls.
                  const auto s = bear_put_spread(90.0, 2.0, 100.0, 7.0);
                  const auto a = s.analytics();
                  t.expect(close(a.net_premium, 5.0, 1e-9), "net DEBIT == 5 (a bear put spread pays up)");
                  t.expect(close(s.pnl_at(80.0), 5.0, 1e-9), "profits when price falls (max profit below lo)");
                  t.expect(close(s.pnl_at(110.0), -5.0, 1e-9), "loses the debit when price rises");
                  t.expect(close(a.max_profit, 5.0, 1e-9) && close(a.max_loss, -5.0, 1e-9),
                           "max profit == width − debit; max loss == −debit");
                  t.expect(a.breakevens.size() == 1 && has_be(a, 95.0), "breakeven == hi − debit == 95");
                  // Its mirror, the bull put spread, is a bullish CREDIT — opposite sign.
                  t.expect(bull_put_spread(90.0, 2.0, 100.0, 7.0).net_premium() < 0.0,
                           "bull put spread is a credit (opposite structure)");
              })
        .test("long straddle: unbounded profit, two breakevens",
              [](TestContext& t) {
                  // Long 100 call @5 + long 100 put @4. Net debit 9.
                  const auto a = long_straddle(100.0, 5.0, 4.0).analytics();
                  t.expect(close(a.net_premium, 9.0, 1e-9), "net debit == 9");
                  t.expect(a.unbounded_profit, "upside unbounded");
                  t.expect(!a.unbounded_loss, "downside bounded (S floored at 0)");
                  t.expect(close(a.max_loss, -9.0, 1e-9), "max loss == −premium at strike");
                  t.expect(a.breakevens.size() == 2 && has_be(a, 91.0) && has_be(a, 109.0),
                           "breakevens at 91 and 109");
              })
        .test("covered call: capped upside, breakeven below entry",
              [](TestContext& t) {
                  // Long underlying @100 + short 105 call @3.
                  const auto s = covered_call(100.0, 105.0, 3.0);
                  const auto a = s.analytics();
                  t.expect(!a.unbounded_profit && !a.unbounded_loss, "bounded both ways");
                  t.expect(close(a.max_profit, 8.0, 1e-9), "max profit == (K−entry)+premium == 8");
                  t.expect(close(a.max_loss, -97.0, 1e-9), "max loss == −(entry−premium) at S=0");
                  t.expect(has_be(a, 97.0), "breakeven == entry − premium == 97");
              })
        .test("protective put: capped loss, unbounded upside",
              [](TestContext& t) {
                  // Long underlying @100 + long 95 put @2.
                  const auto a = protective_put(100.0, 95.0, 2.0).analytics();
                  t.expect(a.unbounded_profit, "upside unbounded");
                  t.expect(!a.unbounded_loss, "downside floored by the put");
                  t.expect(close(a.max_loss, -7.0, 1e-9), "max loss == −(entry−K+premium) == −7");
                  t.expect(has_be(a, 102.0), "breakeven == entry + premium == 102");
              })
        .test("long call butterfly: bounded, peak at the body",
              [](TestContext& t) {
                  // +1 c90 @12, −2 c100 @5, +1 c110 @2. Net debit 4.
                  const auto a = long_call_butterfly(90.0, 12.0, 100.0, 5.0, 110.0, 2.0).analytics();
                  t.expect(close(a.net_premium, 4.0, 1e-9), "net debit == 4");
                  t.expect(!a.unbounded_profit && !a.unbounded_loss, "bounded both ways");
                  t.expect(close(a.max_profit, 6.0, 1e-9), "peak == width − debit == 6 at body");
                  t.expect(close(a.max_loss, -4.0, 1e-9), "max loss == −debit == −4 on the wings");
                  t.expect(a.breakevens.size() == 2 && has_be(a, 94.0) && has_be(a, 106.0),
                           "breakevens at 94 and 106");
              })
        .test("iron condor: net credit, capped both ways",
              [](TestContext& t) {
                  // Long put 90 @1, short put 95 @3, short call 105 @3, long call 110 @1.
                  const auto a = iron_condor(90.0, 1.0, 95.0, 3.0, 105.0, 3.0, 110.0, 1.0).analytics();
                  t.expect(close(a.net_premium, -4.0, 1e-9), "net credit == −4");
                  t.expect(!a.unbounded_profit && !a.unbounded_loss, "capped both ways");
                  t.expect(close(a.max_profit, 4.0, 1e-9), "max profit == credit == 4");
                  t.expect(close(a.max_loss, -1.0, 1e-9), "max loss == −(width − credit) == −1");
                  t.expect(a.breakevens.size() == 2 && has_be(a, 91.0) && has_be(a, 109.0),
                           "breakevens at 91 and 109");
              })
        .test("aggregate Greeks bridge matches Black-Scholes",
              [](TestContext& t) {
                  // Canonical ATM market: S=K=100, r=5%, vol=20%, T=1.
                  const auto market = OptionSpec{}.with_spot(100).with_strike(100).with_rate(0.05)
                                          .with_volatility(0.2).with_expiry(1.0);
                  // Long straddle delta == call delta (0.6368) + put delta (0.6368 − 1) == 0.2736.
                  const auto g = long_straddle(100.0, 10.45, 5.57).aggregate_greeks(market).value();
                  t.expect(close(g.delta, 0.27366, 1e-4), "straddle delta == N(d1) + (N(d1)−1)");
                  t.expect(g.vega > 0.0, "long straddle is long vega");
                  // A short call's delta is negative.
                  const auto gc = short_call(100.0, 10.45).aggregate_greeks(market).value();
                  t.expect(gc.delta < 0.0, "short call delta < 0");
              })
        .test("box spread locks the strike width; net premium is a debit",
              [](TestContext& t) {
                  // Strikes 100/110; a box pays the 10-wide width at expiry regardless of S.
                  const auto s = box_spread(100.0, 6.0, 2.0, 110.0, 2.0, 8.0);
                  // Payoff is constant == width (10) for all S.
                  t.expect(close(s.payoff_at(80.0), 10.0, 1e-9) && close(s.payoff_at(130.0), 10.0, 1e-9),
                           "box payoff is a constant 10 (strike width)");
                  const auto a = s.analytics();
                  t.expect(!a.unbounded_profit && !a.unbounded_loss, "box is fully bounded");
                  // Net premium = (6−2) + (8−2) = 10 debit -> P&L == width − debit == 0 here.
                  t.expect(close(a.net_premium, 10.0, 1e-9), "net debit == 10");
                  t.expect(close(a.max_profit, 0.0, 1e-9) && close(a.max_loss, 0.0, 1e-9),
                           "priced-fair box: flat zero P&L");
              })
        .run();
}
