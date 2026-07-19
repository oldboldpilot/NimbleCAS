// Tests for nimblecas.fxstrat: FX valuation and carry economics.
// @author Olumuyiwa Oluwasanmi
//
// Checks Garman-Kohlhagen against FX put-call parity, the covered-interest-parity forward
// and forward points against their closed forms, and the carry trade against the exact
// structural fact that its breakeven exit spot IS the CIP forward (uncovered interest
// parity) — the check that pins the domestic/foreign convention and catches a sign slip.

import std;
import nimblecas.core;
import nimblecas.fxstrat;
import nimblecas.testing;

using namespace nimblecas::fxstrat;
using nimblecas::MathError;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {
[[nodiscard]] auto close(double a, double b, double tol) -> bool { return std::abs(a - b) < tol; }
}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.fxstrat")
        .test("Garman-Kohlhagen satisfies FX put-call parity",
              [](TestContext& t) {
                  // EURUSD spot 1.10, K 1.10, r_dom 5%, r_for 2%, vol 10%, T 1.
                  const double c = garman_kohlhagen(true, 1.10, 1.10, 0.05, 0.02, 0.10, 1.0).value();
                  const double p = garman_kohlhagen(false, 1.10, 1.10, 0.05, 0.02, 0.10, 1.0).value();
                  t.expect(c > 0.0 && p > 0.0, "both premia positive");
                  // C − P == S·e^{−r_f T} − K·e^{−r_d T}.
                  const double parity = 1.10 * std::exp(-0.02) - 1.10 * std::exp(-0.05);
                  t.expect(close(c - p, parity, 1e-9), "GK put-call parity holds");
                  t.expect(!garman_kohlhagen(true, -1.0, 1.1, 0.05, 0.02, 0.1, 1.0).has_value(),
                           "spot<=0 -> domain_error");
              })
        .test("covered-interest-parity forward, points, and arbitrage",
              [](TestContext& t) {
                  // F = 1.10·e^{(0.05−0.02)·1} = 1.10·e^0.03 = 1.13350.
                  t.expect(close(cip_forward(1.10, 0.05, 0.02, 1.0).value(), 1.13349999, 1e-6),
                           "CIP forward");
                  t.expect(close(forward_points(1.10, 0.05, 0.02, 1.0).value(), 0.03349999, 1e-6),
                           "forward points = F − S");
                  // A market forward of 1.14 is rich by 1.14 − 1.13350 = 0.00650.
                  t.expect(close(covered_interest_arbitrage(1.10, 1.14, 0.05, 0.02, 1.0).value(),
                                 0.00650001, 1e-6), "CIA mispricing vs CIP");
                  t.expect(!cip_forward(-1.0, 0.05, 0.02, 1.0).has_value(), "spot<=0 -> domain_error");
              })
        .test("carry trade: breakeven exit spot is the CIP forward",
              [](TestContext& t) {
                  const auto ct = CarryTrade{}.with_notional(1.0).with_borrow(0.02).with_invest(0.05)
                                      .with_spot(1.10).with_time(1.0);
                  t.expect(close(ct.carry_rate(), 0.03, 1e-12), "carry rate = invest − borrow");
                  // Breakeven = 1.10·e^{(0.02−0.05)·1} = 1.10·e^{−0.03} = 1.067490.
                  const double be = ct.breakeven_spot();
                  t.expect(close(be, 1.06749019, 1e-6), "breakeven spot == uncovered-parity forward");
                  t.expect(close(ct.pnl_at(be), 0.0, 1e-9), "P&L is zero at breakeven");
                  // Flat FX (exit == entry): earn the interest differential e^{0.05} − e^{0.02}.
                  t.expect(close(ct.pnl_at(1.10), std::exp(0.05) - std::exp(0.02), 1e-9),
                           "flat FX earns the rate differential");
                  // A favourable move (foreign appreciates) adds to the carry.
                  t.expect(ct.pnl_at(1.15) > ct.pnl_at(1.10), "foreign appreciation adds P&L");
              })
        .run();
}
