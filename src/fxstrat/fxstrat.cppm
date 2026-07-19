// NimbleCAS currency trading strategies — FX valuation and carry economics.
// @author Olumuyiwa Oluwasanmi
//
// SCOPE. The FX-specific valuation an option/futures strategy engine cannot supply on its
// own: Garman-Kohlhagen FX option pricing (Black-Scholes with the foreign rate as the
// dividend yield — a direct reuse of nimblecas.pricing), covered-interest-parity forwards
// and forward points, the covered-interest-arbitrage mispricing, and the CARRY TRADE — a
// fluent `CarryTrade` that borrows one currency to invest in another and reports its P&L,
// carry rate, and the exit spot at which the FX move exactly erases the interest pickup
// (which is precisely the CIP forward — uncovered interest parity). Option-based FX
// structures (straddles, risk reversals, butterflies on the smile) are built by feeding
// Garman-Kohlhagen premiums into nimblecas.optstrat, which owns the multi-leg P&L; this
// module deliberately does not duplicate that.
//
// HONESTY (config/cpp_details.txt Rule 32, AGENTS.md). Numerical: the prices are BS/GK
// `double`s (transcendental) and the parities are `e^{(r_d−r_f)T}` forms — nothing here
// claims exactness. Quoting convention is stated and fixed: `spot` is DOMESTIC per one
// unit of FOREIGN currency (e.g. USD per EUR), so `rate_domestic` funds and
// `rate_foreign` accrues; a sign slip there is the classic FX bug, so the convention is
// asserted in the tests (carry breakeven == CIP forward). All failure rides the railway;
// nothing throws.

export module nimblecas.fxstrat;

import std;
import nimblecas.core;
import nimblecas.pricing;

export namespace nimblecas::fxstrat {

// Garman-Kohlhagen FX option price: Black-Scholes on `spot` (domestic per foreign) with
// `rate_domestic` as the discount rate and `rate_foreign` as the (continuous) dividend
// yield. Reuses nimblecas.pricing exactly. spot/strike <= 0, time < 0, or vol < 0 ->
// domain_error (propagated from pricing).
[[nodiscard]] auto garman_kohlhagen(bool is_call, double spot, double strike,
                                    double rate_domestic, double rate_foreign, double vol,
                                    double time) -> Result<double>;

// Covered-interest-parity forward F = S·e^{(r_domestic − r_foreign)·T}. spot <= 0 or
// time < 0 -> domain_error.
[[nodiscard]] auto cip_forward(double spot, double rate_domestic, double rate_foreign, double time)
    -> Result<double>;

// Forward points = CIP forward − spot (positive: forward premium on the foreign currency).
[[nodiscard]] auto forward_points(double spot, double rate_domestic, double rate_foreign,
                                  double time) -> Result<double>;

// Covered-interest-arbitrage mispricing = market forward − CIP forward (per unit of
// foreign). Non-zero (beyond costs) is an arbitrage; positive means the forward is rich
// (sell forward / borrow domestic / lend foreign). Same guards as cip_forward.
[[nodiscard]] auto covered_interest_arbitrage(double spot, double forward_market,
                                              double rate_domestic, double rate_foreign,
                                              double time) -> Result<double>;

// A borrow-domestic / invest-foreign carry trade. `spot_entry` and any exit spot are
// DOMESTIC per FOREIGN. Built fluently; reused across exit scenarios.
struct CarryTrade {
    double notional{1.0};         // domestic units borrowed
    double rate_borrow{0.0};      // domestic funding rate (continuous)
    double rate_invest{0.0};      // foreign deposit rate (continuous)
    double spot_entry{1.0};       // domestic per foreign at inception
    double time{1.0};             // horizon in years

    // Nominal interest pickup rate (foreign − domestic); the carry earned if FX is flat.
    [[nodiscard]] auto carry_rate() const noexcept -> double { return rate_invest - rate_borrow; }

    // P&L in domestic currency if the pair is at `spot_exit` at the horizon:
    //   (notional/spot_entry)·e^{r_invest·T}·spot_exit − notional·e^{r_borrow·T}.
    [[nodiscard]] auto pnl_at(double spot_exit) const noexcept -> double {
        const double foreign_grown = (notional / spot_entry) * std::exp(rate_invest * time);
        return foreign_grown * spot_exit - notional * std::exp(rate_borrow * time);
    }
    // The exit spot at which P&L == 0 — exactly the CIP forward (uncovered interest parity):
    //   spot_entry·e^{(r_borrow − r_invest)·T}.
    [[nodiscard]] auto breakeven_spot() const noexcept -> double {
        return spot_entry * std::exp((rate_borrow - rate_invest) * time);
    }

    [[nodiscard]] auto with_notional(double n) const -> CarryTrade { auto c = *this; c.notional = n; return c; }
    [[nodiscard]] auto with_borrow(double r) const -> CarryTrade { auto c = *this; c.rate_borrow = r; return c; }
    [[nodiscard]] auto with_invest(double r) const -> CarryTrade { auto c = *this; c.rate_invest = r; return c; }
    [[nodiscard]] auto with_spot(double s) const -> CarryTrade { auto c = *this; c.spot_entry = s; return c; }
    [[nodiscard]] auto with_time(double t) const -> CarryTrade { auto c = *this; c.time = t; return c; }
};

}  // namespace nimblecas::fxstrat

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::fxstrat {

auto garman_kohlhagen(bool is_call, double spot, double strike, double rate_domestic,
                      double rate_foreign, double vol, double time) -> Result<double> {
    // GK == Black-Scholes with q := foreign rate. Reuse pricing wholesale.
    const auto spec = pricing::OptionSpec{}
                          .with_spot(spot)
                          .with_strike(strike)
                          .with_rate(rate_domestic)
                          .with_dividend(rate_foreign)
                          .with_volatility(vol)
                          .with_expiry(time)
                          .with_type(is_call ? pricing::OptionType::call : pricing::OptionType::put);
    return pricing::black_scholes_price(spec);
}

auto cip_forward(double spot, double rate_domestic, double rate_foreign, double time)
    -> Result<double> {
    if (spot <= 0.0 || time < 0.0) { return make_error<double>(MathError::domain_error); }
    return spot * std::exp((rate_domestic - rate_foreign) * time);
}

auto forward_points(double spot, double rate_domestic, double rate_foreign, double time)
    -> Result<double> {
    return cip_forward(spot, rate_domestic, rate_foreign, time)
        .transform([spot](double f) { return f - spot; });
}

auto covered_interest_arbitrage(double spot, double forward_market, double rate_domestic,
                                double rate_foreign, double time) -> Result<double> {
    return cip_forward(spot, rate_domestic, rate_foreign, time)
        .transform([forward_market](double f) { return forward_market - f; });
}

}  // namespace nimblecas::fxstrat
