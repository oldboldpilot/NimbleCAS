// Tests for nimblecas.exotics: exotic-derivatives pricing.
// @author Olumuyiwa Oluwasanmi
//
// Like the pricing suite, this layer is numerical, so the checks are ANALYTIC ORACLES and
// structural invariants rather than lone magic numbers: the CRR binomial converges to
// Black-Scholes as steps grow; barrier knock-in + knock-out equals the vanilla (in/out
// parity) for all eight single-barrier types and the analytic barrier agrees with the
// existing barrier Monte Carlo; Crank-Nicolson (with Rannacher startup) matches
// Black-Scholes for a European and prices an American put above its European counterpart;
// Margrabe with one asset's volatility set to zero collapses to a vanilla; Kirk with a zero
// strike collapses to Margrabe; the bivariate-normal CDF factorises at rho = 0 and equals
// Phi(min(a,b)) at rho = 1; a floating-strike lookback dominates the vanilla; the chooser
// sits between max(call, put) and the straddle; the Geske compound collapses to the
// underlying call as its first strike vanishes; and a one-asset basket Monte Carlo lands
// within a few standard errors of Black-Scholes.

import std;
import nimblecas.core;
import nimblecas.pricing;
import nimblecas.exotics;
import nimblecas.testing;

using namespace nimblecas::exotics;
using nimblecas::pricing::OptionSpec;
using nimblecas::pricing::OptionType;
using nimblecas::pricing::Exercise;
using nimblecas::pricing::black_scholes_price;
using nimblecas::pricing::barrier_option_mc;
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
    return TestSuite("nimblecas.exotics")
        .test("CRR binomial converges to Black-Scholes; American vs European",
              [](TestContext& t) {
                  const auto call = atm();
                  const double bs = black_scholes_price(call).value();
                  // Even/odd oscillation of the ATM CRR tree is damped by averaging
                  // consecutive step counts -> ~1e-3 accuracy against Black-Scholes.
                  const double c1000 = crr_binomial(call, 1000, Exercise::european).value();
                  const double c1001 = crr_binomial(call, 1001, Exercise::european).value();
                  t.expect(close(0.5 * (c1000 + c1001), bs, 0.01),
                           "CRR European call (averaged) ~ BS");
                  t.expect(close(c1000, bs, 0.1), "single CRR European call within envelope");
                  // No dividend: American call == European call.
                  const double amc = crr_binomial(call, 1000, Exercise::american).value();
                  t.expect(close(amc, c1000, 1e-9), "American call (no div) == European call");
                  // American put carries an early-exercise premium over the European put.
                  const auto put = atm().with_type(OptionType::put);
                  const double eup = crr_binomial(put, 1000, Exercise::european).value();
                  const double amp = crr_binomial(put, 1000, Exercise::american).value();
                  t.expect(amp >= eup - 1e-9, "American put >= European put");
                  t.expect(amp > eup + 1e-4, "American put strictly above European (r>0)");
              })
        .test("CRR discrete dividends (escrowed spot)",
              [](TestContext& t) {
                  const auto call = atm();
                  // An empty dividend schedule reproduces the plain CRR price exactly.
                  const std::array<CashDividend, 0> none{};
                  const double plain = crr_binomial(call, 500, Exercise::european).value();
                  const double via_div =
                      crr_binomial_discrete_div(call, 500, Exercise::european, none).value();
                  t.expect(close(plain, via_div, 1e-12), "no dividends == plain CRR");
                  // A mid-life cash dividend lowers a call (escrowed spot drops).
                  const std::array<CashDividend, 1> divs{CashDividend{0.5, 5.0}};
                  const double with_div =
                      crr_binomial_discrete_div(call, 500, Exercise::european, divs).value();
                  t.expect(with_div < plain, "cash dividend lowers the call");
                  t.expect(with_div > 0.0, "dividend-adjusted call stays positive");
                  // Escrowing more than the whole spot is a domain error.
                  const std::array<CashDividend, 1> huge{CashDividend{0.5, 200.0}};
                  t.expect(!crr_binomial_discrete_div(call, 500, Exercise::european, huge)
                                .has_value(),
                           "over-escrowed spot -> error");
              })
        .test("barrier analytic in/out parity == vanilla (all 8 types)",
              [](TestContext& t) {
                  const auto call = atm();
                  const auto put = atm().with_type(OptionType::put);
                  const double bc = black_scholes_price(call).value();
                  const double bp = black_scholes_price(put).value();
                  // Down-barrier at 90 (< spot), up-barrier at 110 (> spot).
                  const double dn = 90.0;
                  const double up = 110.0;
                  auto parity = [&](const OptionSpec& s, double h, Barrier side, double vanilla,
                                    std::string_view label) {
                      const double in = barrier_analytic(s, h, side, true).value();
                      const double out = barrier_analytic(s, h, side, false).value();
                      t.expect(close(in + out, vanilla, 1e-8), label);
                      t.expect(out <= vanilla + 1e-9 && out >= -1e-12, "knock-out in [0, vanilla]");
                      t.expect(in >= -1e-12, "knock-in >= 0");
                  };
                  parity(call, dn, Barrier::down, bc, "down call in+out == vanilla");
                  parity(call, up, Barrier::up, bc, "up call in+out == vanilla");
                  parity(put, dn, Barrier::down, bp, "down put in+out == vanilla");
                  parity(put, up, Barrier::up, bp, "up put in+out == vanilla");
                  // Spot on the wrong side of the barrier is a domain error.
                  t.expect(!barrier_analytic(call, 120.0, Barrier::down, false).has_value(),
                           "down barrier above spot -> error");
              })
        .test("barrier analytic agrees with the barrier Monte Carlo",
              [](TestContext& t) {
                  // A far down-and-out call (barrier 80, spot 100): knock-out is rare, so the
                  // continuous closed form and the discretely-monitored MC nearly coincide.
                  const auto call = atm();
                  const double analytic =
                      barrier_analytic(call, 80.0, Barrier::down, false).value();
                  const auto mc = barrier_option_mc(call, 80.0, false, 200000, 1000, 4242).value();
                  t.expect(std::abs(analytic - mc.price) < 5.0 * mc.std_error + 0.15,
                           "analytic down-out call ~ MC (within error + monitoring bias)");
                  t.expect(analytic > 0.0, "analytic barrier price positive");
              })
        .test("Crank-Nicolson PDE matches Black-Scholes; American put premium",
              [](TestContext& t) {
                  const auto call = atm();
                  const auto put = atm().with_type(OptionType::put);
                  const double bc = black_scholes_price(call).value();
                  const double bp = black_scholes_price(put).value();
                  const double fdc = fd_pde_price(call, 400, 400, Exercise::european).value();
                  const double fdp = fd_pde_price(put, 400, 400, Exercise::european).value();
                  t.expect(close(fdc, bc, 0.05), "CN European call ~ BS");
                  t.expect(close(fdp, bp, 0.05), "CN European put ~ BS");
                  // American put priced by PSOR is at least the European put.
                  const double amp = fd_pde_price(put, 400, 400, Exercise::american).value();
                  t.expect(amp >= fdp - 0.02, "PSOR American put >= European put");
                  // Grid bounds are enforced.
                  t.expect(!fd_pde_price(call, 3, 10, Exercise::european).has_value(),
                           "too-small grid -> error");
              })
        .test("Margrabe reduces to a vanilla; Kirk reduces to Margrabe",
              [](TestContext& t) {
                  // Exchange with asset 2 deterministic (vol2 = 0) and its yield equal to the
                  // risk-free rate reproduces a Black-Scholes call on asset 1.
                  const double bs = black_scholes_price(atm()).value();
                  const double mg =
                      margrabe_exchange(100.0, 100.0, 0.0, 0.05, 0.2, 0.0, 0.0, 1.0).value();
                  t.expect(close(mg, bs, 1e-9), "Margrabe (vol2=0, q2=r) == BS call");
                  t.expect(mg > 0.0, "Margrabe value positive");
                  // Kirk with zero strike is exactly a Margrabe exchange.
                  const double kirk0 =
                      kirk_spread(100.0, 90.0, 0.0, 0.05, 0.0, 0.0, 0.2, 0.3, 0.5, 1.0).value();
                  const double marg =
                      margrabe_exchange(100.0, 90.0, 0.0, 0.0, 0.2, 0.3, 0.5, 1.0).value();
                  t.expect(close(kirk0, marg, 1e-9), "Kirk (K=0) == Margrabe");
                  // Kirk with a positive strike is worth less than the zero-strike spread.
                  const double kirkK =
                      kirk_spread(100.0, 90.0, 10.0, 0.05, 0.0, 0.0, 0.2, 0.3, 0.5, 1.0).value();
                  t.expect(kirkK < kirk0 && kirkK > 0.0, "Kirk decreases with strike");
                  // |rho| > 1 is a domain error.
                  t.expect(!margrabe_exchange(100.0, 100.0, 0.0, 0.0, 0.2, 0.2, 1.5, 1.0)
                                .has_value(),
                           "|rho|>1 -> error");
              })
        .test("bivariate-normal CDF: factorisation, comonotone limit, monotonicity",
              [](TestContext& t) {
                  const double a = 0.3;
                  const double b = -0.2;
                  // rho = 0 factorises to Phi(a) Phi(b).
                  const double na = nimblecas::pricing::norm_cdf(a);
                  const double nb = nimblecas::pricing::norm_cdf(b);
                  t.expect(close(bivariate_normal_cdf(a, b, 0.0), na * nb, 1e-12),
                           "rho=0 factorises");
                  // rho = 1 gives Phi(min(a,b)).
                  t.expect(close(bivariate_normal_cdf(0.5, 1.0, 1.0),
                                 nimblecas::pricing::norm_cdf(0.5), 1e-12),
                           "rho=1 == Phi(min)");
                  // rho = -1 gives max(Phi(a)+Phi(b)-1, 0).
                  t.expect(close(bivariate_normal_cdf(0.5, 0.5, -1.0),
                                 std::max(2.0 * nimblecas::pricing::norm_cdf(0.5) - 1.0, 0.0),
                                 1e-12),
                           "rho=-1 comonotone-opposite");
                  // Symmetric in its two arguments and monotone increasing in each.
                  t.expect(close(bivariate_normal_cdf(0.4, -0.1, 0.5),
                                 bivariate_normal_cdf(-0.1, 0.4, 0.5), 1e-12),
                           "symmetric in (a,b)");
                  t.expect(bivariate_normal_cdf(1.0, 0.0, 0.5) > bivariate_normal_cdf(0.0, 0.0, 0.5),
                           "monotone increasing in a");
                  // Deep in the joint tail approaches 1.
                  t.expect(bivariate_normal_cdf(8.0, 8.0, 0.3) > 0.999999, "joint upper tail ~ 1");
              })
        .test("Geske compound collapses to the underlying call",
              [](TestContext& t) {
                  const auto spec = atm();  // T2 = 1
                  const double inner = black_scholes_price(spec).value();
                  // A zero-premium compound option is exactly the underlying call.
                  const double zero = geske_compound(spec, 0.0, 100.0, 0.5).value();
                  t.expect(close(zero, inner, 1e-9), "strike1=0 compound == underlying call");
                  // A tiny first premium is still close to the underlying call (exercises the
                  // bivariate-normal path and the critical-price root).
                  const double tiny = geske_compound(spec, 0.01, 100.0, 0.5).value();
                  t.expect(close(tiny, inner, 0.3), "tiny strike1 ~ underlying call");
                  // A meaningful first premium is worth strictly less than the underlying call.
                  const double comp = geske_compound(spec, 5.0, 100.0, 0.5).value();
                  t.expect(comp > 0.0 && comp < inner, "compound in (0, underlying call)");
              })
        .test("floating-strike lookback dominates the vanilla",
              [](TestContext& t) {
                  const auto call = atm();
                  const auto put = atm().with_type(OptionType::put);
                  const double bc = black_scholes_price(call).value();
                  const double bp = black_scholes_price(put).value();
                  const double lc = lookback_price(call).value();  // fresh: min = spot
                  const double lp = lookback_price(put).value();   // fresh: max = spot
                  t.expect(lc >= bc - 1e-9, "lookback call >= vanilla call");
                  t.expect(lp >= bp - 1e-9, "lookback put >= vanilla put");
                  t.expect(std::isfinite(lc) && lc > 0.0, "lookback call finite & positive");
                  // b == 0 (r == q) is the removable-singularity case -> domain error.
                  const auto zerocarry = atm().with_dividend(0.05);
                  t.expect(!lookback_price(zerocarry).has_value(), "b=0 lookback -> error");
              })
        .test("simple chooser sits between max(call,put) and the straddle",
              [](TestContext& t) {
                  const auto call = atm();
                  const auto put = atm().with_type(OptionType::put);
                  const double c = black_scholes_price(call).value();
                  const double p = black_scholes_price(put).value();
                  const double ch = chooser_price(call, 0.5).value();
                  t.expect(ch >= std::max(c, p) - 1e-9, "chooser >= max(call, put)");
                  t.expect(ch <= c + p + 1e-9, "chooser <= straddle (call + put)");
                  // Choosing later (t1 -> T2) is worth more (approaches the straddle).
                  const double ch_late = chooser_price(call, 0.99).value();
                  t.expect(ch_late >= ch - 1e-9, "later choice is worth at least as much");
              })
        .test("basket Monte Carlo: single asset matches Black-Scholes; reproducible",
              [](TestContext& t) {
                  const double bs = black_scholes_price(atm()).value();
                  const std::array<BasketAsset, 1> one{
                      BasketAsset{100.0, 0.2, 0.0, 1.0}};
                  const std::array<double, 1> corr{1.0};
                  const auto mc = basket_mc(one, corr, 100.0, 0.05, 1.0, OptionType::call,
                                            200000, 20260718)
                                      .value();
                  t.expect(std::abs(mc.price - bs) < 4.0 * mc.std_error + 1e-6,
                           "one-asset basket call within 4 std errors of BS");
                  // Bit-reproducible under a fixed seed.
                  const auto mc2 = basket_mc(one, corr, 100.0, 0.05, 1.0, OptionType::call,
                                             200000, 20260718)
                                       .value();
                  t.expect(mc.price == mc2.price, "basket MC is bit-reproducible");
                  // A non-positive-definite correlation is rejected.
                  const std::array<BasketAsset, 2> two{
                      BasketAsset{100.0, 0.2, 0.0, 0.5}, BasketAsset{100.0, 0.3, 0.0, 0.5}};
                  const std::array<double, 4> bad{1.0, 1.5, 1.5, 1.0};  // |rho|>1: not PD
                  t.expect(!basket_mc(two, bad, 100.0, 0.05, 1.0, OptionType::call, 1000, 7)
                                .has_value(),
                           "non-PD correlation -> error");
                  // A well-formed two-asset basket prices positively.
                  const std::array<double, 4> good{1.0, 0.5, 0.5, 1.0};
                  const auto mcb = basket_mc(two, good, 100.0, 0.05, 1.0, OptionType::call,
                                             50000, 99)
                                       .value();
                  t.expect(mcb.price > 0.0 && std::isfinite(mcb.price), "two-asset basket positive");
              })
        .run();
}
