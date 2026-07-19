// NimbleCAS option trading strategies — composable multi-leg structures + analytics.
// @author Olumuyiwa Oluwasanmi
//
// SCOPE. A COMPOSABLE, FLUENT option-strategy layer over nimblecas.pricing. A strategy is
// a signed bag of `StrategyLeg`s (long/short calls, puts, and the underlying); the module
// supplies (1) the standard NAMED structures as one-line builders — covered call,
// protective put, collar, bull/bear call & put spreads, straddle, strangle, strip, strap,
// butterfly (call & iron), condor (call & iron), box, risk reversal, ratio spread — and
// (2) a general EXPIRY-P&L ANALYTICS ENGINE that returns the net premium (debit/credit),
// max profit, max loss, and every breakeven for ANY such structure. Aggregate Black-Scholes
// Greeks are available by delegating each option leg to nimblecas.pricing against a market
// template (reuse, not re-derivation).
//
// HONESTY (config/cpp_details.txt Rule 32, AGENTS.md). The expiry P&L of a vanilla option
// book is EXACTLY a continuous PIECEWISE-LINEAR function of the terminal price with kinks
// only at the strikes — so the analytics here are computed EXACTLY on that structure, not
// sampled: max/min are attained at strikes or at ±infinity, and breakevens are the exact
// linear roots between kinks. When a payoff direction is genuinely unbounded (a naked long
// call's upside, a short straddle's downside) the analytics say so — `unbounded_profit` /
// `unbounded_loss` — rather than fabricating a finite number. The Greeks bridge is
// numerical (Black-Scholes) and inherits pricing's honesty boundary. All failure rides the
// railway (`Result<T>` / `MathError`); nothing throws.

export module nimblecas.optstrat;

import std;
import nimblecas.core;
import nimblecas.pricing;

export namespace nimblecas::optstrat {

using nimblecas::pricing::Greeks;
using nimblecas::pricing::OptionSpec;
using nimblecas::pricing::OptionType;

// What a leg is a position in. `underlying` is a linear (stock/futures) leg with no strike.
enum class LegKind : std::uint8_t { call, put, underlying };

// One signed leg of a strategy. `premium` is the per-unit price transacted: for an option
// it is the option premium; for the underlying it is the entry price. `quantity` is signed
// (long +, short −). `terminal_value` is the per-unit intrinsic at expiry.
struct StrategyLeg {
    LegKind kind{LegKind::call};
    double strike{0.0};        // ignored for `underlying`
    double quantity{1.0};      // signed: long (+) / short (−)
    double premium{0.0};       // per-unit price paid (option premium / underlying entry)

    // Per-unit terminal (intrinsic) value at underlying price s.
    [[nodiscard]] auto terminal_value(double s) const noexcept -> double {
        switch (kind) {
            case LegKind::call:       return std::max(s - strike, 0.0);
            case LegKind::put:        return std::max(strike - s, 0.0);
            case LegKind::underlying: return s;
        }
        return 0.0;
    }
    // Signed cash outlay of the position (debit > 0, credit < 0).
    [[nodiscard]] auto cost() const noexcept -> double { return quantity * premium; }
    // P&L of this leg alone at expiry price s.
    [[nodiscard]] auto pnl_at(double s) const noexcept -> double {
        return quantity * terminal_value(s) - cost();
    }
    // d(terminal_value)/ds far ABOVE every strike (calls & underlying slope +1, puts 0).
    [[nodiscard]] auto slope_above() const noexcept -> double {
        return (kind == LegKind::put) ? 0.0 : 1.0;
    }
    [[nodiscard]] auto with_quantity(double q) const -> StrategyLeg { auto c = *this; c.quantity = q; return c; }
    [[nodiscard]] auto with_premium(double p) const -> StrategyLeg { auto c = *this; c.premium = p; return c; }
};

// The exact expiry-P&L profile of a strategy.
struct StrategyAnalytics {
    double net_premium{0.0};        // Σ leg cost; > 0 net debit, < 0 net credit
    bool unbounded_profit{false};
    bool unbounded_loss{false};
    double max_profit{0.0};         // meaningful iff !unbounded_profit
    double max_loss{0.0};           // most-negative P&L; meaningful iff !unbounded_loss
    std::vector<double> breakevens{};  // ascending terminal prices where P&L == 0
};

// A composable, reusable option strategy: a bag of signed legs valued at expiry as a unit.
// Built fluently: OptionStrategy::create().add_call(...).add_underlying(...), or via a named
// builder below.
class OptionStrategy {
public:
    [[nodiscard]] static auto create() -> OptionStrategy { return OptionStrategy{}; }

    [[nodiscard]] auto with_leg(const StrategyLeg& leg) -> OptionStrategy& {
        legs_.push_back(leg);
        return *this;
    }
    [[nodiscard]] auto add_call(double strike, double quantity, double premium) -> OptionStrategy& {
        return with_leg(StrategyLeg{LegKind::call, strike, quantity, premium});
    }
    [[nodiscard]] auto add_put(double strike, double quantity, double premium) -> OptionStrategy& {
        return with_leg(StrategyLeg{LegKind::put, strike, quantity, premium});
    }
    [[nodiscard]] auto add_underlying(double quantity, double entry_price) -> OptionStrategy& {
        return with_leg(StrategyLeg{LegKind::underlying, 0.0, quantity, entry_price});
    }
    [[nodiscard]] auto legs() const noexcept -> std::span<const StrategyLeg> { return legs_; }
    [[nodiscard]] auto size() const noexcept -> std::size_t { return legs_.size(); }

    // Gross terminal value Σ quantity·terminal_value(s) (the payoff diagram, no premium).
    [[nodiscard]] auto payoff_at(double s) const noexcept -> double;
    // Net cash outlay Σ leg cost (> 0 debit, < 0 credit).
    [[nodiscard]] auto net_premium() const noexcept -> double;
    // P&L at expiry = payoff_at(s) − net_premium.
    [[nodiscard]] auto pnl_at(double s) const noexcept -> double { return payoff_at(s) - net_premium(); }

    // The exact expiry-P&L profile (piecewise-linear analysis on the strikes).
    [[nodiscard]] auto analytics() const -> StrategyAnalytics;

    // Aggregate Black-Scholes Greeks against a market template `market` (its rate / vol /
    // dividend / expiry / spot apply to every option leg; each leg overrides strike + type).
    // Underlying legs contribute delta == quantity. Reuses nimblecas.pricing. A leg's
    // domain error propagates.
    [[nodiscard]] auto aggregate_greeks(const OptionSpec& market) const -> Result<Greeks>;

private:
    OptionStrategy() = default;
    std::vector<StrategyLeg> legs_{};
};

// --- Named strategy builders (fluent, reusable) -----------------------------
// Every option quantity is per `lots` (default 1). Premiums are the per-unit prices
// transacted (typically from a marketdata OptionChain, or a pricing::black_scholes_price
// theoretical). Underlying legs use the entry price as their "premium".

[[nodiscard]] auto long_call(double strike, double premium, double lots = 1.0) -> OptionStrategy;
[[nodiscard]] auto long_put(double strike, double premium, double lots = 1.0) -> OptionStrategy;
[[nodiscard]] auto short_call(double strike, double premium, double lots = 1.0) -> OptionStrategy;
[[nodiscard]] auto short_put(double strike, double premium, double lots = 1.0) -> OptionStrategy;

// Long underlying + short call (income/partial-hedge).
[[nodiscard]] auto covered_call(double spot_entry, double call_strike, double call_premium,
                                double lots = 1.0) -> OptionStrategy;
// Long underlying + long put (insurance).
[[nodiscard]] auto protective_put(double spot_entry, double put_strike, double put_premium,
                                  double lots = 1.0) -> OptionStrategy;
// Long underlying + long put + short call (bounded band, often near-zero cost).
[[nodiscard]] auto collar(double spot_entry, double put_strike, double put_premium,
                          double call_strike, double call_premium, double lots = 1.0) -> OptionStrategy;

// Vertical spreads (lower/upper strike, their premiums).
[[nodiscard]] auto bull_call_spread(double lo_strike, double lo_premium, double hi_strike,
                                    double hi_premium, double lots = 1.0) -> OptionStrategy;
[[nodiscard]] auto bear_call_spread(double lo_strike, double lo_premium, double hi_strike,
                                    double hi_premium, double lots = 1.0) -> OptionStrategy;
[[nodiscard]] auto bull_put_spread(double lo_strike, double lo_premium, double hi_strike,
                                   double hi_premium, double lots = 1.0) -> OptionStrategy;
[[nodiscard]] auto bear_put_spread(double lo_strike, double lo_premium, double hi_strike,
                                   double hi_premium, double lots = 1.0) -> OptionStrategy;

// Volatility structures.
[[nodiscard]] auto long_straddle(double strike, double call_premium, double put_premium,
                                 double lots = 1.0) -> OptionStrategy;
[[nodiscard]] auto short_straddle(double strike, double call_premium, double put_premium,
                                  double lots = 1.0) -> OptionStrategy;
[[nodiscard]] auto long_strangle(double put_strike, double put_premium, double call_strike,
                                 double call_premium, double lots = 1.0) -> OptionStrategy;
[[nodiscard]] auto short_strangle(double put_strike, double put_premium, double call_strike,
                                  double call_premium, double lots = 1.0) -> OptionStrategy;
// Strip: 1 call + 2 puts long (bearish vol). Strap: 2 calls + 1 put long (bullish vol).
[[nodiscard]] auto strip(double strike, double call_premium, double put_premium, double lots = 1.0)
    -> OptionStrategy;
[[nodiscard]] auto strap(double strike, double call_premium, double put_premium, double lots = 1.0)
    -> OptionStrategy;

// Butterflies & condors (strikes ascending). Call butterfly: +1 lo, −2 mid, +1 hi.
[[nodiscard]] auto long_call_butterfly(double lo, double lo_prem, double mid, double mid_prem,
                                       double hi, double hi_prem, double lots = 1.0) -> OptionStrategy;
[[nodiscard]] auto short_call_butterfly(double lo, double lo_prem, double mid, double mid_prem,
                                        double hi, double hi_prem, double lots = 1.0) -> OptionStrategy;
// Iron butterfly: short ATM straddle + long OTM strangle wings (net credit).
[[nodiscard]] auto iron_butterfly(double put_wing, double put_wing_prem, double body_strike,
                                  double body_put_prem, double body_call_prem, double call_wing,
                                  double call_wing_prem, double lots = 1.0) -> OptionStrategy;
// Long call condor: +1 k1, −1 k2, −1 k3, +1 k4 (ascending).
[[nodiscard]] auto long_call_condor(double k1, double p1, double k2, double p2, double k3, double p3,
                                    double k4, double p4, double lots = 1.0) -> OptionStrategy;
// Iron condor: short put spread + short call spread (net credit). Strikes ascending:
// long put k1, short put k2, short call k3, long call k4.
[[nodiscard]] auto iron_condor(double k1, double p1, double k2, double p2, double k3, double p3,
                               double k4, double p4, double lots = 1.0) -> OptionStrategy;

// Box spread: bull call spread + bear put spread on the same strikes (a synthetic
// fixed payoff = strike width; arbitrage/financing structure).
[[nodiscard]] auto box_spread(double lo_strike, double lo_call_prem, double lo_put_prem,
                              double hi_strike, double hi_call_prem, double hi_put_prem,
                              double lots = 1.0) -> OptionStrategy;
// Risk reversal: long call + short put (synthetic long; often near-zero cost).
[[nodiscard]] auto risk_reversal(double put_strike, double put_premium, double call_strike,
                                 double call_premium, double lots = 1.0) -> OptionStrategy;
// Ratio call spread: long 1 lo call, short `ratio` hi calls (per lot).
[[nodiscard]] auto ratio_call_spread(double lo_strike, double lo_premium, double hi_strike,
                                     double hi_premium, double ratio = 2.0, double lots = 1.0)
    -> OptionStrategy;

}  // namespace nimblecas::optstrat

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::optstrat {

auto OptionStrategy::payoff_at(double s) const noexcept -> double {
    double total = 0.0;
    for (const auto& leg : legs_) { total += leg.quantity * leg.terminal_value(s); }
    return total;
}

auto OptionStrategy::net_premium() const noexcept -> double {
    double total = 0.0;
    for (const auto& leg : legs_) { total += leg.cost(); }
    return total;
}

auto OptionStrategy::analytics() const -> StrategyAnalytics {
    StrategyAnalytics a{};
    a.net_premium = net_premium();

    // Collect the distinct option strikes — the only kinks of the piecewise-linear P&L.
    std::vector<double> knots;
    knots.reserve(legs_.size() + 1);
    for (const auto& leg : legs_) {
        if (leg.kind != LegKind::underlying) { knots.push_back(leg.strike); }
    }
    std::ranges::sort(knots);
    knots.erase(std::ranges::unique(knots).begin(), knots.end());

    // Evaluation abscissae: 0 (the left boundary; s >= 0) plus every strike. The P&L is
    // linear between consecutive abscissae and on the ray beyond the largest strike.
    std::vector<double> xs;
    xs.reserve(knots.size() + 1);
    xs.push_back(0.0);
    for (double k : knots) {
        if (k > 0.0) { xs.push_back(k); }
    }
    std::ranges::sort(xs);
    xs.erase(std::ranges::unique(xs).begin(), xs.end());

    // Slope of total P&L for s ABOVE every strike (all calls & underlying legs count +q).
    double right_slope = 0.0;
    for (const auto& leg : legs_) { right_slope += leg.quantity * leg.slope_above(); }

    // P&L at each abscissa.
    std::vector<double> ys;
    ys.reserve(xs.size());
    for (double x : xs) { ys.push_back(pnl_at(x)); }

    constexpr double kEps = 1e-12;
    // Extrema: attained at abscissae, plus ±infinity along the right ray by its slope sign.
    a.max_profit = *std::ranges::max_element(ys);
    a.max_loss = *std::ranges::min_element(ys);
    if (right_slope > kEps) { a.unbounded_profit = true; }
    else if (right_slope < -kEps) { a.unbounded_loss = true; }
    // (No unbounded LEFT side: s is floored at 0, and ys[0] == pnl_at(0) captures it.)

    // Breakevens: exact roots of the piecewise-linear P&L.
    auto push_be = [&](double s) {
        if (std::ranges::none_of(a.breakevens, [&](double e) { return std::abs(e - s) < 1e-9; })) {
            a.breakevens.push_back(s);
        }
    };
    for (std::size_t i = 0; i + 1 < xs.size(); ++i) {
        const double y0 = ys[i];
        const double y1 = ys[i + 1];
        if (std::abs(y0) < kEps) { push_be(xs[i]); }
        if (y0 * y1 < 0.0) {  // strict sign change -> interior root
            push_be(xs[i] + (xs[i + 1] - xs[i]) * (-y0) / (y1 - y0));
        }
    }
    // Last abscissa itself, and the right ray beyond it.
    const double xr = xs.back();
    const double yr = ys.back();
    if (std::abs(yr) < kEps) { push_be(xr); }
    if (std::abs(right_slope) > kEps) {
        const double root = xr - yr / right_slope;      // pnl(xr) + slope*(s-xr) == 0
        if (root > xr + 1e-9) { push_be(root); }
    }
    std::ranges::sort(a.breakevens);
    return a;
}

auto OptionStrategy::aggregate_greeks(const OptionSpec& market) const -> Result<Greeks> {
    Greeks agg{};
    for (const auto& leg : legs_) {
        if (leg.kind == LegKind::underlying) {
            agg.price += leg.quantity * market.spot;   // mark the underlying at spot
            agg.delta += leg.quantity;                 // delta of the underlying is 1
            continue;
        }
        const auto spec = market.with_strike(leg.strike)
                              .with_type(leg.kind == LegKind::call ? OptionType::call : OptionType::put);
        auto g = pricing::black_scholes_greeks(spec);
        if (!g) { return make_error<Greeks>(g.error()); }
        agg.price += leg.quantity * g->price;
        agg.delta += leg.quantity * g->delta;
        agg.gamma += leg.quantity * g->gamma;
        agg.vega  += leg.quantity * g->vega;
        agg.theta += leg.quantity * g->theta;
        agg.rho   += leg.quantity * g->rho;
    }
    return agg;
}

// --- Named builders ---------------------------------------------------------

auto long_call(double strike, double premium, double lots) -> OptionStrategy {
    return OptionStrategy::create().add_call(strike, lots, premium);
}
auto long_put(double strike, double premium, double lots) -> OptionStrategy {
    return OptionStrategy::create().add_put(strike, lots, premium);
}
auto short_call(double strike, double premium, double lots) -> OptionStrategy {
    return OptionStrategy::create().add_call(strike, -lots, premium);
}
auto short_put(double strike, double premium, double lots) -> OptionStrategy {
    return OptionStrategy::create().add_put(strike, -lots, premium);
}

auto covered_call(double spot_entry, double call_strike, double call_premium, double lots)
    -> OptionStrategy {
    return OptionStrategy::create().add_underlying(lots, spot_entry).add_call(call_strike, -lots, call_premium);
}
auto protective_put(double spot_entry, double put_strike, double put_premium, double lots)
    -> OptionStrategy {
    return OptionStrategy::create().add_underlying(lots, spot_entry).add_put(put_strike, lots, put_premium);
}
auto collar(double spot_entry, double put_strike, double put_premium, double call_strike,
            double call_premium, double lots) -> OptionStrategy {
    return OptionStrategy::create()
        .add_underlying(lots, spot_entry)
        .add_put(put_strike, lots, put_premium)
        .add_call(call_strike, -lots, call_premium);
}

auto bull_call_spread(double lo_strike, double lo_premium, double hi_strike, double hi_premium,
                      double lots) -> OptionStrategy {
    return OptionStrategy::create().add_call(lo_strike, lots, lo_premium).add_call(hi_strike, -lots, hi_premium);
}
auto bear_call_spread(double lo_strike, double lo_premium, double hi_strike, double hi_premium,
                      double lots) -> OptionStrategy {
    return OptionStrategy::create().add_call(lo_strike, -lots, lo_premium).add_call(hi_strike, lots, hi_premium);
}
auto bull_put_spread(double lo_strike, double lo_premium, double hi_strike, double hi_premium,
                     double lots) -> OptionStrategy {
    return OptionStrategy::create().add_put(lo_strike, lots, lo_premium).add_put(hi_strike, -lots, hi_premium);
}
auto bear_put_spread(double lo_strike, double lo_premium, double hi_strike, double hi_premium,
                     double lots) -> OptionStrategy {
    return OptionStrategy::create().add_put(lo_strike, -lots, lo_premium).add_put(hi_strike, lots, hi_premium);
}

auto long_straddle(double strike, double call_premium, double put_premium, double lots)
    -> OptionStrategy {
    return OptionStrategy::create().add_call(strike, lots, call_premium).add_put(strike, lots, put_premium);
}
auto short_straddle(double strike, double call_premium, double put_premium, double lots)
    -> OptionStrategy {
    return OptionStrategy::create().add_call(strike, -lots, call_premium).add_put(strike, -lots, put_premium);
}
auto long_strangle(double put_strike, double put_premium, double call_strike, double call_premium,
                   double lots) -> OptionStrategy {
    return OptionStrategy::create().add_put(put_strike, lots, put_premium).add_call(call_strike, lots, call_premium);
}
auto short_strangle(double put_strike, double put_premium, double call_strike, double call_premium,
                    double lots) -> OptionStrategy {
    return OptionStrategy::create().add_put(put_strike, -lots, put_premium).add_call(call_strike, -lots, call_premium);
}
auto strip(double strike, double call_premium, double put_premium, double lots) -> OptionStrategy {
    return OptionStrategy::create().add_call(strike, lots, call_premium).add_put(strike, 2.0 * lots, put_premium);
}
auto strap(double strike, double call_premium, double put_premium, double lots) -> OptionStrategy {
    return OptionStrategy::create().add_call(strike, 2.0 * lots, call_premium).add_put(strike, lots, put_premium);
}

auto long_call_butterfly(double lo, double lo_prem, double mid, double mid_prem, double hi,
                         double hi_prem, double lots) -> OptionStrategy {
    return OptionStrategy::create()
        .add_call(lo, lots, lo_prem)
        .add_call(mid, -2.0 * lots, mid_prem)
        .add_call(hi, lots, hi_prem);
}
auto short_call_butterfly(double lo, double lo_prem, double mid, double mid_prem, double hi,
                          double hi_prem, double lots) -> OptionStrategy {
    return OptionStrategy::create()
        .add_call(lo, -lots, lo_prem)
        .add_call(mid, 2.0 * lots, mid_prem)
        .add_call(hi, -lots, hi_prem);
}
auto iron_butterfly(double put_wing, double put_wing_prem, double body_strike, double body_put_prem,
                    double body_call_prem, double call_wing, double call_wing_prem, double lots)
    -> OptionStrategy {
    return OptionStrategy::create()
        .add_put(put_wing, lots, put_wing_prem)
        .add_put(body_strike, -lots, body_put_prem)
        .add_call(body_strike, -lots, body_call_prem)
        .add_call(call_wing, lots, call_wing_prem);
}
auto long_call_condor(double k1, double p1, double k2, double p2, double k3, double p3, double k4,
                      double p4, double lots) -> OptionStrategy {
    return OptionStrategy::create()
        .add_call(k1, lots, p1)
        .add_call(k2, -lots, p2)
        .add_call(k3, -lots, p3)
        .add_call(k4, lots, p4);
}
auto iron_condor(double k1, double p1, double k2, double p2, double k3, double p3, double k4,
                 double p4, double lots) -> OptionStrategy {
    return OptionStrategy::create()
        .add_put(k1, lots, p1)
        .add_put(k2, -lots, p2)
        .add_call(k3, -lots, p3)
        .add_call(k4, lots, p4);
}
auto box_spread(double lo_strike, double lo_call_prem, double lo_put_prem, double hi_strike,
                double hi_call_prem, double hi_put_prem, double lots) -> OptionStrategy {
    // Bull call spread (long lo call, short hi call) + bear put spread (short lo put, long hi put).
    return OptionStrategy::create()
        .add_call(lo_strike, lots, lo_call_prem)
        .add_call(hi_strike, -lots, hi_call_prem)
        .add_put(hi_strike, lots, hi_put_prem)
        .add_put(lo_strike, -lots, lo_put_prem);
}
auto risk_reversal(double put_strike, double put_premium, double call_strike, double call_premium,
                   double lots) -> OptionStrategy {
    return OptionStrategy::create().add_call(call_strike, lots, call_premium).add_put(put_strike, -lots, put_premium);
}
auto ratio_call_spread(double lo_strike, double lo_premium, double hi_strike, double hi_premium,
                       double ratio, double lots) -> OptionStrategy {
    return OptionStrategy::create().add_call(lo_strike, lots, lo_premium).add_call(hi_strike, -ratio * lots, hi_premium);
}

}  // namespace nimblecas::optstrat
