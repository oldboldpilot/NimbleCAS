// Tests for nimblecas.pricing: options & instrument pricing.
// @author Olumuyiwa Oluwasanmi
//
// Because this layer is numerical, the suite checks it against ANALYTIC ORACLES and
// structural invariants rather than lone magic numbers: Black-Scholes matches its textbook
// value and satisfies put-call parity; the trinomial tree converges to Black-Scholes for
// European exercise; American >= European (and American call == European call with no
// dividend); the geometric-Asian closed form reduces EXACTLY to Black-Scholes at one
// averaging date; Monte Carlo lands within a few standard errors of the oracle AND is
// bit-reproducible under a fixed seed; the control variate shrinks the MC standard error.

import std;
import nimblecas.core;
import nimblecas.pricing;
import nimblecas.svgplot;
import nimblecas.testing;

using namespace nimblecas::pricing;
using nimblecas::PlotOptions;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {
[[nodiscard]] auto close(double a, double b, double tol) -> bool { return std::abs(a - b) < tol; }
// The canonical at-the-money one-year option used across several checks.
[[nodiscard]] auto atm() -> OptionSpec {
    return OptionSpec{}.with_spot(100).with_strike(100).with_rate(0.05)
        .with_volatility(0.2).with_expiry(1.0);
}
}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.pricing")
        .test("Black-Scholes value, parity and Greeks",
              [](TestContext& t) {
                  const auto call = atm();
                  const auto put = atm().with_type(OptionType::put);
                  const double c = black_scholes_price(call).value();
                  const double p = black_scholes_price(put).value();
                  t.expect(close(c, 10.450583572, 1e-6), "ATM 1y call == 10.4506");
                  // Put-call parity: C - P == S e^{-qT} - K e^{-rT}.
                  const double parity = 100.0 - 100.0 * std::exp(-0.05);
                  t.expect(close(c - p, parity, 1e-9), "put-call parity holds");
                  // Delta of the call == N(d1) == N(0.35) == 0.63683.
                  t.expect(close(black_scholes_greeks(call).value().delta, 0.636831, 1e-5),
                           "call delta == 0.6368");
                  // norm_cdf / inverse sanity.
                  t.expect(close(norm_cdf(0.0), 0.5, 1e-12), "Phi(0) == 0.5");
                  t.expect(close(inverse_norm_cdf(0.5).value(), 0.0, 1e-9), "Phi^-1(0.5) == 0");
                  t.expect(close(norm_cdf(1.959963985), 0.975, 1e-6), "Phi(1.96) == 0.975");
              })
        .test("trinomial tree converges to Black-Scholes; American >= European",
              [](TestContext& t) {
                  const auto call = atm();
                  const double bs = black_scholes_price(call).value();
                  const double tri = trinomial_price(call, 400, Exercise::european).value();
                  t.expect(close(tri, bs, 0.02), "trinomial European call ~ BS");
                  // No dividend: American call == European call.
                  const double amc = trinomial_price(call, 400, Exercise::american).value();
                  t.expect(close(amc, bs, 0.02), "American call (no div) == European call");
                  // American put has early-exercise premium: >= European put.
                  const auto put = atm().with_type(OptionType::put);
                  const double eup = trinomial_price(put, 400, Exercise::european).value();
                  const double amp = trinomial_price(put, 400, Exercise::american).value();
                  t.expect(amp >= eup - 1e-6, "American put >= European put");
              })
        .test("geometric Asian reduces to Black-Scholes at one averaging date",
              [](TestContext& t) {
                  const auto call = atm();
                  const double bs = black_scholes_price(call).value();
                  const double geo1 = geometric_asian_price(call, 1).value();
                  t.expect(close(geo1, bs, 1e-9), "geometric Asian (n=1) == BS exactly");
                  // Averaging lowers effective volatility, so a multi-date geometric Asian call
                  // is worth strictly less than the vanilla.
                  const double geo50 = geometric_asian_price(call, 50).value();
                  t.expect(geo50 < bs, "geometric Asian (n=50) < vanilla");
              })
        .test("Monte Carlo: accuracy, reproducibility, control-variate variance reduction",
              [](TestContext& t) {
                  const auto call = atm();
                  const double bs = black_scholes_price(call).value();
                  const auto mc = monte_carlo_european(call, 200000, 12345).value();
                  t.expect(std::abs(mc.price - bs) < 4.0 * mc.std_error + 1e-9,
                           "European MC within 4 standard errors of BS");
                  // Bit-reproducible: identical seed -> identical price.
                  const auto mc2 = monte_carlo_european(call, 200000, 12345).value();
                  t.expect(mc.price == mc2.price, "MC is bit-reproducible under a fixed seed");
                  // Control variate shrinks the standard error vs. plain MC on the Asian.
                  const auto cv = monte_carlo_asian(call, 50000, 20, 999, true).value();
                  const auto plain = monte_carlo_asian(call, 50000, 20, 999, false).value();
                  t.expect(cv.std_error < plain.std_error, "geometric CV reduces MC std error");
                  t.expect(cv.price > 0.0 && cv.price < bs, "arithmetic Asian call < vanilla");
              })
        .test("Longstaff-Schwartz American MC is a sensible lower bound",
              [](TestContext& t) {
                  const auto put = atm().with_type(OptionType::put);
                  const double eup = black_scholes_price(put).value();
                  const auto lsm = longstaff_schwartz_american(put, 40000, 40, 777).value();
                  t.expect(std::isfinite(lsm.price), "LSM price is finite");
                  // American put is worth at least the European put (allow MC noise).
                  t.expect(lsm.price > eup - 0.15, "LSM American put >= European put (within noise)");
                  t.expect(lsm.price >= put.payoff(put.spot), "LSM >= immediate exercise value");
              })
        .test("composable Portfolio aggregates value, Greeks and payoff",
              [](TestContext& t) {
                  const auto call = atm();
                  const auto put = atm().with_type(OptionType::put);
                  auto straddle = Portfolio::create().add(call, 1.0).add(put, 1.0);
                  const double c = black_scholes_price(call).value();
                  const double p = black_scholes_price(put).value();
                  t.expect(close(straddle.value().value(), c + p, 1e-9), "straddle value == C + P");
                  // A long straddle is delta-light near ATM (call delta + put delta ~ small).
                  t.expect(std::abs(straddle.greeks().value().delta) < 0.3, "straddle ~ delta-neutral");
                  // Payoff at expiry: |S - K|.
                  t.expect(close(straddle.payoff_at(120.0), 20.0, 1e-12), "straddle payoff at 120 == 20");
              })
        .test("extended Greeks match finite differences",
              [](TestContext& t) {
                  const auto s = atm();
                  const auto eg = black_scholes_extended_greeks(s).value();
                  const double h = 1e-4;
                  // vanna == d delta / d vol.
                  const double dd_dvol =
                      (black_scholes_greeks(s.with_volatility(0.2 + h)).value().delta -
                       black_scholes_greeks(s.with_volatility(0.2 - h)).value().delta) / (2 * h);
                  t.expect(close(eg.vanna, dd_dvol, 1e-3), "vanna == d delta/d vol");
                  // vomma == d vega / d vol.
                  const double dvega_dvol =
                      (black_scholes_greeks(s.with_volatility(0.2 + h)).value().vega -
                       black_scholes_greeks(s.with_volatility(0.2 - h)).value().vega) / (2 * h);
                  t.expect(close(eg.vomma, dvega_dvol, 1e-2), "vomma == d vega/d vol");
                  // speed == d gamma / d spot.
                  const double dg_ds =
                      (black_scholes_greeks(s.with_spot(100 + h)).value().gamma -
                       black_scholes_greeks(s.with_spot(100 - h)).value().gamma) / (2 * h);
                  t.expect(close(eg.speed, dg_ds, 1e-4), "speed == d gamma/d spot");
                  // dual_delta == d price / d strike.
                  const double dp_dk =
                      (black_scholes_price(s.with_strike(100 + h)).value() -
                       black_scholes_price(s.with_strike(100 - h)).value()) / (2 * h);
                  t.expect(close(eg.dual_delta, dp_dk, 1e-4), "dual_delta == d price/d strike");
              })
        .test("Black-76, digitals, option P&L, and barrier in/out parity",
              [](TestContext& t) {
                  // Black-76 on the forward F = S e^{(r-q)T} equals Black-Scholes (q=0).
                  const auto s = atm();
                  const double fwd = s.spot * std::exp(s.rate * s.time_to_expiry);
                  const double b76 = black76_price(true, fwd, s.strike, s.rate, s.volatility,
                                                   s.time_to_expiry).value();
                  t.expect(close(b76, black_scholes_price(s).value(), 1e-9),
                           "Black-76(F) == Black-Scholes");
                  // Vanilla call == asset-or-nothing - K * cash-or-nothing(1).
                  const double aon = digital_asset_or_nothing(s).value();
                  const double con = digital_cash_or_nothing(s, 1.0).value();
                  t.expect(close(black_scholes_price(s).value(), aon - s.strike * con, 1e-9),
                           "call == AON - K*CON");
                  // Option P&L at expiry: long ATM call, S_T = 120, premium 10 -> profit 10.
                  t.expect(close(option_pnl_at_expiry(s, 10.0, 120.0), 10.0, 1e-12), "call P&L == 10");
                  // Barrier in/out parity: knock-in + knock-out == vanilla (same path engine).
                  const auto in = barrier_option_mc(s, 90.0, true, 100000, 50, 2024).value();
                  const auto out = barrier_option_mc(s, 90.0, false, 100000, 50, 2024).value();
                  t.expect(std::abs((in.price + out.price) - black_scholes_price(s).value()) < 0.3,
                           "barrier in + out == vanilla (within MC error)");
                  t.expect(out.price <= black_scholes_price(s).value() + 1e-9,
                           "knock-out <= vanilla");
              })
        .test("plotting produces SVG",
              [](TestContext& t) {
                  PlotOptions opt{};
                  opt.title = "payoff";
                  auto svg = payoff_diagram_svg(atm(), 50.0, 150.0, 64, opt);
                  t.expect(svg.has_value() && svg->find("<svg") != std::string::npos,
                           "payoff diagram is SVG");
                  auto dens = terminal_density_svg(atm(), 40.0, 200.0, 128, opt);
                  t.expect(dens.has_value() && dens->find("<svg") != std::string::npos,
                           "terminal density is SVG");
                  auto curve = price_vs_spot_svg(atm(), 50.0, 150.0, 64, opt);
                  t.expect(curve.has_value(), "price-vs-spot curve renders");
              })
        .run();
}
