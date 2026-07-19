// Tests for nimblecas.futures: cost-of-carry valuation and futures strategies.
// @author Olumuyiwa Oluwasanmi
//
// Valuation is checked against the closed cost-of-carry identities (F = S·e^{bT} and its
// inverses) with hand-verified numbers; strategies against exact linear-P&L arithmetic
// and the structural facts a trader relies on: an outright's breakeven is its entry, a
// net-zero spread's P&L is the captured (convergence) spread, and cash-and-carry locks the
// basis. The honesty boundary is exercised too (bad spec -> domain_error; a pnl_at price
// vector of the wrong length -> domain_error, never a silent partial sum).

import std;
import nimblecas.core;
import nimblecas.futures;
import nimblecas.testing;

using namespace nimblecas::futures;
using nimblecas::MathError;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {
[[nodiscard]] auto close(double a, double b, double tol) -> bool { return std::abs(a - b) < tol; }
}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.futures")
        .test("cost-of-carry forward price and carry composition",
              [](TestContext& t) {
                  // S=100, r=5%, T=1 -> F = 100·e^0.05 = 105.1271096.
                  const auto eq = FuturesSpec{}.with_spot(100).with_rate(0.05).with_expiry(1.0);
                  t.expect(close(eq.carry_rate(), 0.05, 1e-12), "carry == r with no other terms");
                  t.expect(close(forward_price(eq).value(), 105.12710964, 1e-6), "F == S·e^{rT}");
                  t.expect(close(futures_price(eq).value(), forward_price(eq).value(), 1e-12),
                           "futures == forward under deterministic rates");
                  // Dividend yield q=2% lowers carry to 3%.
                  const auto div = eq.with_dividend(0.02);
                  t.expect(close(div.carry_rate(), 0.03, 1e-12), "carry == r − q");
                  t.expect(close(forward_price(div).value(), 103.04545340, 1e-6), "F with dividend");
                  // Commodity: r=5%, u=1%, y=3% -> carry 3%, same F as the q=2% case.
                  const auto comm = eq.with_storage(0.01).with_convenience(0.03);
                  t.expect(close(comm.carry_rate(), 0.03, 1e-12), "carry == r + u − y");
                  t.expect(close(forward_price(comm).value(), 103.04545340, 1e-6), "commodity F");
                  // Convergence: at T=0, F == S.
                  t.expect(close(forward_price(eq.with_expiry(0.0)).value(), 100.0, 1e-12),
                           "F -> S as T -> 0");
                  // Bad spec.
                  t.expect(!forward_price(eq.with_spot(-1)).has_value(), "S<=0 -> domain_error");
              })
        .test("discrete-income forward, basis, market state",
              [](TestContext& t) {
                  // (S − I)·e^{rT} = 95·e^0.05 = 99.87075410.
                  t.expect(close(forward_price_discrete_income(100.0, 5.0, 0.05, 1.0).value(),
                                 99.87075410, 1e-6), "discrete-income forward");
                  t.expect(!forward_price_discrete_income(5.0, 10.0, 0.05, 1.0).has_value(),
                           "S − I <= 0 -> domain_error");
                  t.expect(close(basis(100.0, 105.0), -5.0, 1e-12), "cash basis = spot − futures");
                  t.expect(market_state(100.0, 105.0) == MarketState::contango, "F>S contango");
                  t.expect(market_state(105.0, 100.0) == MarketState::backwardation, "F<S backwardation");
                  t.expect(market_state(100.0, 100.0) == MarketState::flat, "F==S flat");
              })
        .test("implied carry, implied convenience yield, forward value",
              [](TestContext& t) {
                  // ln(105.1271096/100)/1 == 0.05.
                  t.expect(close(implied_cost_of_carry(100.0, 105.12710964, 1.0).value(), 0.05, 1e-8),
                           "implied cost of carry recovers r");
                  // From F=103.0454534 with r=5%, q=u=0: y = 0.05 − ln(1.030454534) = 0.02.
                  const auto spec = FuturesSpec{}.with_spot(100).with_rate(0.05).with_expiry(1.0);
                  t.expect(close(implied_convenience_yield(spec, 103.04545340).value(), 0.02, 1e-8),
                           "implied convenience yield");
                  t.expect(!implied_cost_of_carry(100.0, -1.0, 1.0).has_value(), "F<=0 -> domain_error");
                  // Long forward struck at K=100: value = S − K·e^{−rT} = 100 − 95.1229424 = 4.8770576.
                  t.expect(close(forward_contract_value(spec, 100.0).value(), 4.87705760, 1e-6),
                           "long forward value = S − K·e^{−rT}");
                  // Struck exactly at the fair forward -> zero value.
                  const double fair = forward_price(spec).value();
                  t.expect(close(forward_contract_value(spec, fair).value(), 0.0, 1e-9),
                           "forward struck at fair value is worth 0");
              })
        .test("mark-to-market P&L, notional and tick value",
              [](TestContext& t) {
                  // 2 long ES contracts (multiplier 50) from 4500 to 4510 -> 2·50·10 = 1000.
                  t.expect(close(mark_to_market_pnl(4500.0, 4510.0, 50.0, 2.0), 1000.0, 1e-9), "long MtM");
                  t.expect(close(mark_to_market_pnl(4500.0, 4510.0, 50.0, -2.0), -1000.0, 1e-9), "short MtM");
                  t.expect(close(notional_value(4500.0, 50.0, 2.0), 450000.0, 1e-6), "notional = P·mult·|q|");
                  t.expect(close(tick_value(0.25, 50.0), 12.5, 1e-9), "tick value = tick·mult");
              })
        .test("outright strategy: linear P&L, unbounded, breakeven == entry",
              [](TestContext& t) {
                  const auto lng = long_futures("ES", 4500.0, 2.0, 50.0, 0.25);
                  t.expect(lng.size() == 1, "one leg");
                  t.expect(close(lng.pnl_at_uniform(4510.0), 1000.0, 1e-9), "long +1000 at +10");
                  t.expect(close(lng.pnl_at_uniform(4490.0), -1000.0, 1e-9), "long −1000 at −10");
                  const auto pl = lng.profile();
                  t.expect(pl.unbounded_profit && pl.unbounded_loss, "outright is unbounded both ways");
                  t.expect(pl.breakeven.has_value() && close(*pl.breakeven, 4500.0, 1e-9),
                           "breakeven == entry price");
                  t.expect(close(pl.net_quantity, 100.0, 1e-9), "net exposure = q·mult");
                  // short is the mirror image.
                  const auto sht = short_futures("ES", 4500.0, 2.0, 50.0);
                  t.expect(close(sht.pnl_at_uniform(4490.0), 1000.0, 1e-9), "short +1000 when price falls 10");
              })
        .test("calendar spread: net-zero exposure captures the convergence spread",
              [](TestContext& t) {
                  // Long near @4500, short far @4520, 1 lot, multiplier 50.
                  const auto cal = calendar_spread("ESM", 4500.0, "ESU", 4520.0, 1.0, 50.0);
                  t.expect(cal.size() == 2, "two legs");
                  // Different settlements: near 4510, far 4525 -> 50·10 − 50·5 = 250.
                  const std::array<double, 2> px{4510.0, 4525.0};
                  t.expect(close(cal.pnl_at(px).value(), 250.0, 1e-9), "spread P&L at distinct settles");
                  // Net exposure is zero; the flat P&L is the captured spread (far−near)·mult = 1000.
                  const auto pl = cal.profile();
                  t.expect(!pl.unbounded_profit && !pl.unbounded_loss, "spread P&L is bounded");
                  t.expect(close(pl.net_quantity, 0.0, 1e-9), "net exposure zero");
                  t.expect(close(pl.max_profit, 1000.0, 1e-9) && close(pl.max_loss, 1000.0, 1e-9),
                           "convergence P&L = (far−near)·mult");
                  // A wrong-length price vector is refused.
                  const std::array<double, 1> bad{4510.0};
                  t.expect(!cal.pnl_at(bad).has_value(), "length mismatch -> domain_error");
              })
        .test("inter-commodity spread and cash-and-carry lock the basis",
              [](TestContext& t) {
                  // Crack spread: long 3 crude @80, short 2 products @95, multiplier 1.
                  const auto crack = inter_commodity_spread("CL", 80.0, 3.0, "RB", 95.0, 2.0, 1.0);
                  const std::array<double, 2> px{85.0, 97.0};
                  // 3·(85−80) − 2·(97−95) = 15 − 4 = 11.
                  t.expect(close(crack.pnl_at(px).value(), 11.0, 1e-9), "crack spread P&L");
                  t.expect(close(crack.net_quantity(), 1.0, 1e-9), "net 3 long − 2 short = 1");
                  // Cash-and-carry: long spot @100, short futures @105 -> locks basis of 5.
                  const auto cc = cash_and_carry("SPOT", 100.0, "FUT", 105.0, 1.0, 1.0);
                  const auto pl = cc.profile();
                  t.expect(close(pl.max_profit, 5.0, 1e-9) && close(pl.net_quantity, 0.0, 1e-9),
                           "cash-and-carry locks the 5-point basis");
                  // At common delivery price 103, both legs settle there -> still 5.
                  const std::array<double, 2> conv{103.0, 103.0};
                  t.expect(close(cc.pnl_at(conv).value(), 5.0, 1e-9), "basis locked at convergence");
              })
        .run();
}
