// NimbleCAS futures & forwards — cost-of-carry valuation and trading strategies.
// @author Olumuyiwa Oluwasanmi
//
// SCOPE. The cost-of-carry model for forward/futures FAIR VALUE — F = S·e^{b·T} with a
// carry rate b = r − q + u − y (financing r, dividend/foreign yield q, storage u,
// convenience yield y) — plus its inverses (implied convenience yield, implied
// repo / net cost of carry), the basis and its contango/backwardation classification,
// the value of an off-market forward, futures mark-to-market P&L, and notional / tick
// arithmetic. On top of valuation sits a COMPOSABLE, FLUENT strategy layer: a signed
// `FuturesLeg` and a reusable `FuturesStrategy` that aggregates P&L across legs, with
// named builders for the standard structures — outright long/short, calendar
// (inter-delivery) spread, inter-commodity / crack-crush spread, long/short hedge, and
// the cash-and-carry arbitrage — each carrying its analytics (net entry cost, per-tick
// value, breakeven, and for a spread its breakeven spread level).
//
// HONESTY (config/cpp_details.txt Rule 32, AGENTS.md). This layer is NUMERICAL: the very
// first quantity, e^{b·T}, is transcendental, so NOTHING here claims exactness — prices
// are correctly-rounded `double`. Under DETERMINISTIC rates the futures price EQUALS the
// forward price (the standard result); when they would differ — stochastic-rate convexity
// — this module prices the forward and says so, rather than fabricating a convexity
// adjustment it cannot justify. Futures P&L is LINEAR and therefore UNBOUNDED for an
// outright position: `StrategyProfile` reports `unbounded_profit`/`unbounded_loss`
// truthfully instead of inventing a finite max. All failure rides the railway
// (`Result<T>` / `MathError`); nothing throws.

export module nimblecas.futures;

import std;
import nimblecas.core;

export namespace nimblecas::futures {

// How the futures price sits relative to spot. contango: F > S (positive net carry);
// backwardation: F < S (negative net carry, e.g. a high convenience yield); flat: F == S.
enum class MarketState : std::uint8_t { backwardation, flat, contango };

// Immutable forward/futures contract + market state, built fluently (Rules 15/47). The
// carry components are continuously-compounded ANNUAL rates; only the relevant ones need
// setting (equities: `dividend_yield`; commodities: `storage_cost` + `convenience_yield`;
// FX: put the foreign rate in `dividend_yield`). `contract_size` (the multiplier) and
// `tick_size` drive the P&L / notional helpers.
struct FuturesSpec {
    double spot{100.0};               // S — current underlying (cash) price
    double rate{0.0};                 // r — continuously-compounded financing rate
    double dividend_yield{0.0};       // q — dividend (equity) or foreign (FX) yield
    double storage_cost{0.0};         // u — storage/carry cost rate (commodities)
    double convenience_yield{0.0};    // y — convenience yield (commodities)
    double time_to_expiry{1.0};       // T — years to delivery
    double contract_size{1.0};        // multiplier (index points -> currency, bbl, etc.)
    double tick_size{0.0};            // minimum price increment

    // Net cost of carry b = r − q + u − y (the exponent's rate).
    [[nodiscard]] auto carry_rate() const noexcept -> double {
        return rate - dividend_yield + storage_cost - convenience_yield;
    }

    [[nodiscard]] auto with_spot(double s) const -> FuturesSpec { auto c = *this; c.spot = s; return c; }
    [[nodiscard]] auto with_rate(double r) const -> FuturesSpec { auto c = *this; c.rate = r; return c; }
    [[nodiscard]] auto with_dividend(double q) const -> FuturesSpec { auto c = *this; c.dividend_yield = q; return c; }
    [[nodiscard]] auto with_storage(double u) const -> FuturesSpec { auto c = *this; c.storage_cost = u; return c; }
    [[nodiscard]] auto with_convenience(double y) const -> FuturesSpec { auto c = *this; c.convenience_yield = y; return c; }
    [[nodiscard]] auto with_expiry(double t) const -> FuturesSpec { auto c = *this; c.time_to_expiry = t; return c; }
    [[nodiscard]] auto with_contract_size(double m) const -> FuturesSpec { auto c = *this; c.contract_size = m; return c; }
    [[nodiscard]] auto with_tick_size(double t) const -> FuturesSpec { auto c = *this; c.tick_size = t; return c; }
};

// --- Valuation --------------------------------------------------------------

// Fair forward/futures price F = S·e^{b·T}. Under deterministic rates the futures and
// forward prices coincide (the standard no-arbitrage result). S <= 0 or T < 0 -> domain_error.
[[nodiscard]] auto forward_price(const FuturesSpec& spec) -> Result<double>;
// Alias — identical to forward_price under deterministic rates; named for call-site clarity.
[[nodiscard]] auto futures_price(const FuturesSpec& spec) -> Result<double>;

// Fair forward when the underlying pays KNOWN DISCRETE income with present value
// `pv_income` (dividends/coupons): F = (S − pv_income)·e^{rT}. S − pv_income <= 0 or
// T < 0 -> domain_error.
[[nodiscard]] auto forward_price_discrete_income(double spot, double pv_income, double rate,
                                                 double time) -> Result<double>;

// Cash basis = spot − futures (the trader's convention: a POSITIVE basis is
// backwardation). Total on finite inputs.
[[nodiscard]] auto basis(double spot, double futures) noexcept -> double;

// Classify the market from spot and a (market or fair) futures price.
[[nodiscard]] auto market_state(double spot, double futures) noexcept -> MarketState;

// Implied net cost of carry b = ln(F/S)/T from an observed futures price (a.k.a. the
// implied repo rate when q=u=y=0). S <= 0, F <= 0, or T <= 0 -> domain_error.
[[nodiscard]] auto implied_cost_of_carry(double spot, double futures, double time) -> Result<double>;

// Implied convenience yield y backed out of a market futures price, holding r, q, u and
// the spec's other terms: y = r − q + u − ln(F/S)/T. Same guards as above.
[[nodiscard]] auto implied_convenience_yield(const FuturesSpec& spec, double market_futures)
    -> Result<double>;

// Value of a LONG forward contract with delivery (strike) price K:
// f = (F − K)·e^{−rT}, where F is the current fair forward. A short forward is the
// negation. T < 0 -> domain_error (propagated from forward_price).
[[nodiscard]] auto forward_contract_value(const FuturesSpec& spec, double delivery_price)
    -> Result<double>;

// Mark-to-market P&L of a futures position: quantity·contract_size·(current − entry).
// `quantity` is signed (long +, short −). Total on finite inputs.
[[nodiscard]] auto mark_to_market_pnl(double entry_price, double current_price,
                                      double contract_size, double quantity) noexcept -> double;

// Notional (contract) value = price·contract_size·|quantity|. Total.
[[nodiscard]] auto notional_value(double price, double contract_size, double quantity) noexcept
    -> double;

// Currency value of one tick for one contract = tick_size·contract_size. Total.
[[nodiscard]] auto tick_value(double tick_size, double contract_size) noexcept -> double;

// ===========================================================================
// Composable, fluent strategy layer.
// ===========================================================================

// One signed position in a specific futures contract, entered at `entry_price`. `label`
// names the underlying/expiry so a multi-leg strategy can report per-instrument P&L and
// so the pnl_at price vector lines up leg-for-leg.
struct FuturesLeg {
    std::string label{};
    double entry_price{0.0};
    double quantity{1.0};        // signed: long (+) / short (−), in contracts
    double contract_size{1.0};
    double tick_size{0.0};

    // P&L of this leg alone at settlement price `p`.
    [[nodiscard]] auto pnl_at(double p) const noexcept -> double {
        return quantity * contract_size * (p - entry_price);
    }
    [[nodiscard]] auto notional() const noexcept -> double {
        return entry_price * contract_size * std::abs(quantity);
    }
    [[nodiscard]] auto with_label(std::string l) const -> FuturesLeg { auto c = *this; c.label = std::move(l); return c; }
    [[nodiscard]] auto with_entry(double p) const -> FuturesLeg { auto c = *this; c.entry_price = p; return c; }
    [[nodiscard]] auto with_quantity(double q) const -> FuturesLeg { auto c = *this; c.quantity = q; return c; }
    [[nodiscard]] auto with_contract_size(double m) const -> FuturesLeg { auto c = *this; c.contract_size = m; return c; }
    [[nodiscard]] auto with_tick_size(double t) const -> FuturesLeg { auto c = *this; c.tick_size = t; return c; }
};

// The realised-risk profile of a strategy. Futures P&L is linear, so a direction with
// non-zero net exposure is UNBOUNDED — reported truthfully rather than as a fake finite
// number. `breakeven` is populated only when the whole strategy reduces to a single price
// variable (every leg shares one `contract_size`-weighted exposure); empty otherwise.
struct StrategyProfile {
    bool unbounded_profit{false};
    bool unbounded_loss{false};
    double max_profit{0.0};       // meaningful only when !unbounded_profit
    double max_loss{0.0};         // meaningful only when !unbounded_loss (a loss is < 0)
    std::optional<double> breakeven{};
    double net_quantity{0.0};     // Σ quantity·contract_size (signed net exposure)
};

// A reusable bag of futures legs valued as a unit — the composable strategy graph. Built
// fluently: FuturesStrategy::create().with_leg(a).with_leg(b). Immutable-friendly: the
// named builders below return a ready strategy.
class FuturesStrategy {
public:
    [[nodiscard]] static auto create() -> FuturesStrategy { return FuturesStrategy{}; }

    [[nodiscard]] auto with_leg(const FuturesLeg& leg) -> FuturesStrategy& {
        legs_.push_back(leg);
        return *this;
    }
    [[nodiscard]] auto legs() const noexcept -> std::span<const FuturesLeg> { return legs_; }
    [[nodiscard]] auto size() const noexcept -> std::size_t { return legs_.size(); }

    // Total P&L at the vector of settlement prices (parallel to legs()). A length mismatch
    // -> domain_error (never a silent partial sum).
    [[nodiscard]] auto pnl_at(std::span<const double> settlement_prices) const -> Result<double>;

    // Total P&L when EVERY leg settles at the same price `p` (the single-instrument case,
    // e.g. an outright multi-contract position or a stacked roll on one contract).
    [[nodiscard]] auto pnl_at_uniform(double p) const noexcept -> double;

    // Signed net exposure Σ quantity·contract_size.
    [[nodiscard]] auto net_quantity() const noexcept -> double;
    // Sum of leg notionals.
    [[nodiscard]] auto notional() const noexcept -> double;

    // Risk profile under the single-price-variable interpretation (uses pnl_at_uniform).
    // The breakeven is the uniform price where total P&L crosses zero (present iff net
    // exposure is non-zero).
    [[nodiscard]] auto profile() const -> StrategyProfile;

private:
    FuturesStrategy() = default;
    std::vector<FuturesLeg> legs_{};
};

// --- Named strategy builders (fluent, reusable) -----------------------------

// Outright directional positions.
[[nodiscard]] auto long_futures(std::string label, double entry, double quantity = 1.0,
                                double contract_size = 1.0, double tick_size = 0.0)
    -> FuturesStrategy;
[[nodiscard]] auto short_futures(std::string label, double entry, double quantity = 1.0,
                                 double contract_size = 1.0, double tick_size = 0.0)
    -> FuturesStrategy;

// Calendar (inter-delivery) spread: long the near contract, short the far one (a "bull"
// calendar; pass a negative `quantity` for the bear side). Legs order: [near, far].
[[nodiscard]] auto calendar_spread(std::string near_label, double near_entry,
                                   std::string far_label, double far_entry,
                                   double quantity = 1.0, double contract_size = 1.0)
    -> FuturesStrategy;

// Inter-commodity spread (crack, crush, or any 2-leg relative-value trade): long
// `long_qty` of leg A, short `short_qty` of leg B. Legs order: [A, B].
[[nodiscard]] auto inter_commodity_spread(std::string long_label, double long_entry, double long_qty,
                                          std::string short_label, double short_entry, double short_qty,
                                          double contract_size = 1.0) -> FuturesStrategy;

// Hedge of a physical/portfolio exposure with futures. `spot_quantity` is the underlying
// units held (long +), `hedge_ratio` the futures contracts shorted per unit; a long hedge
// (protecting a future purchase) passes a negative `spot_quantity`. Legs order: [futures].
// The spot leg is not a futures contract, so only the futures hedge leg is modelled here.
[[nodiscard]] auto short_hedge(std::string futures_label, double futures_entry,
                               double contracts, double contract_size = 1.0) -> FuturesStrategy;
[[nodiscard]] auto long_hedge(std::string futures_label, double futures_entry,
                              double contracts, double contract_size = 1.0) -> FuturesStrategy;

// Cash-and-carry arbitrage: long the cash/spot leg, short the futures leg (reverse
// cash-and-carry negates both). Modelled as the futures leg P&L plus the carry-adjusted
// spot leg; legs order: [spot, futures]. Both are priced at their own settlement in pnl_at.
[[nodiscard]] auto cash_and_carry(std::string spot_label, double spot_entry,
                                  std::string futures_label, double futures_entry,
                                  double quantity = 1.0, double contract_size = 1.0)
    -> FuturesStrategy;

}  // namespace nimblecas::futures

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::futures {

auto forward_price(const FuturesSpec& spec) -> Result<double> {
    if (spec.spot <= 0.0 || spec.time_to_expiry < 0.0) {
        return make_error<double>(MathError::domain_error);
    }
    return spec.spot * std::exp(spec.carry_rate() * spec.time_to_expiry);
}

auto futures_price(const FuturesSpec& spec) -> Result<double> { return forward_price(spec); }

auto forward_price_discrete_income(double spot, double pv_income, double rate, double time)
    -> Result<double> {
    if (time < 0.0 || (spot - pv_income) <= 0.0) { return make_error<double>(MathError::domain_error); }
    return (spot - pv_income) * std::exp(rate * time);
}

auto basis(double spot, double futures) noexcept -> double { return spot - futures; }

auto market_state(double spot, double futures) noexcept -> MarketState {
    if (futures > spot) { return MarketState::contango; }
    if (futures < spot) { return MarketState::backwardation; }
    return MarketState::flat;
}

auto implied_cost_of_carry(double spot, double futures, double time) -> Result<double> {
    if (spot <= 0.0 || futures <= 0.0 || time <= 0.0) { return make_error<double>(MathError::domain_error); }
    return std::log(futures / spot) / time;
}

auto implied_convenience_yield(const FuturesSpec& spec, double market_futures) -> Result<double> {
    // From F = S·e^{(r − q + u − y)T}: y = r − q + u − ln(F/S)/T.
    auto b = implied_cost_of_carry(spec.spot, market_futures, spec.time_to_expiry);
    if (!b) { return make_error<double>(b.error()); }
    return spec.rate - spec.dividend_yield + spec.storage_cost - *b;
}

auto forward_contract_value(const FuturesSpec& spec, double delivery_price) -> Result<double> {
    auto f = forward_price(spec);
    if (!f) { return make_error<double>(f.error()); }
    return (*f - delivery_price) * std::exp(-spec.rate * spec.time_to_expiry);
}

auto mark_to_market_pnl(double entry_price, double current_price, double contract_size,
                        double quantity) noexcept -> double {
    return quantity * contract_size * (current_price - entry_price);
}

auto notional_value(double price, double contract_size, double quantity) noexcept -> double {
    return price * contract_size * std::abs(quantity);
}

auto tick_value(double tick_size, double contract_size) noexcept -> double {
    return tick_size * contract_size;
}

// --- FuturesStrategy --------------------------------------------------------

auto FuturesStrategy::pnl_at(std::span<const double> settlement_prices) const -> Result<double> {
    if (settlement_prices.size() != legs_.size()) { return make_error<double>(MathError::domain_error); }
    double total = 0.0;
    for (std::size_t i = 0; i < legs_.size(); ++i) { total += legs_[i].pnl_at(settlement_prices[i]); }
    return total;
}

auto FuturesStrategy::pnl_at_uniform(double p) const noexcept -> double {
    double total = 0.0;
    for (const auto& leg : legs_) { total += leg.pnl_at(p); }
    return total;
}

auto FuturesStrategy::net_quantity() const noexcept -> double {
    double net = 0.0;
    for (const auto& leg : legs_) { net += leg.quantity * leg.contract_size; }
    return net;
}

auto FuturesStrategy::notional() const noexcept -> double {
    double total = 0.0;
    for (const auto& leg : legs_) { total += leg.notional(); }
    return total;
}

auto FuturesStrategy::profile() const -> StrategyProfile {
    StrategyProfile prof{};
    prof.net_quantity = net_quantity();
    // Under the single-price interpretation, P&L(p) = net_quantity·p − Σ qᵢ·cᵢ·entryᵢ, a
    // straight line. Non-zero slope -> unbounded both ways, with a breakeven at the root.
    // Zero slope -> a flat P&L (a pure spread with matched exposure): the constant is both
    // the max profit and max loss, and there is no single breakeven price.
    double intercept = 0.0;  // P&L at p == 0
    for (const auto& leg : legs_) { intercept += leg.quantity * leg.contract_size * (0.0 - leg.entry_price); }
    if (prof.net_quantity != 0.0) {
        prof.unbounded_profit = true;
        prof.unbounded_loss = true;
        prof.breakeven = -intercept / prof.net_quantity;  // P&L(p*) == 0
    } else {
        prof.unbounded_profit = false;
        prof.unbounded_loss = false;
        prof.max_profit = intercept;   // constant P&L
        prof.max_loss = intercept;
    }
    return prof;
}

// --- Named builders ---------------------------------------------------------

namespace {
[[nodiscard]] auto leg(std::string label, double entry, double qty, double size, double tick)
    -> FuturesLeg {
    return FuturesLeg{std::move(label), entry, qty, size, tick};
}
}  // namespace

auto long_futures(std::string label, double entry, double quantity, double contract_size,
                  double tick_size) -> FuturesStrategy {
    return FuturesStrategy::create().with_leg(leg(std::move(label), entry, quantity, contract_size, tick_size));
}

auto short_futures(std::string label, double entry, double quantity, double contract_size,
                   double tick_size) -> FuturesStrategy {
    return FuturesStrategy::create().with_leg(leg(std::move(label), entry, -quantity, contract_size, tick_size));
}

auto calendar_spread(std::string near_label, double near_entry, std::string far_label,
                     double far_entry, double quantity, double contract_size) -> FuturesStrategy {
    return FuturesStrategy::create()
        .with_leg(leg(std::move(near_label), near_entry, quantity, contract_size, 0.0))
        .with_leg(leg(std::move(far_label), far_entry, -quantity, contract_size, 0.0));
}

auto inter_commodity_spread(std::string long_label, double long_entry, double long_qty,
                            std::string short_label, double short_entry, double short_qty,
                            double contract_size) -> FuturesStrategy {
    return FuturesStrategy::create()
        .with_leg(leg(std::move(long_label), long_entry, long_qty, contract_size, 0.0))
        .with_leg(leg(std::move(short_label), short_entry, -short_qty, contract_size, 0.0));
}

auto short_hedge(std::string futures_label, double futures_entry, double contracts,
                 double contract_size) -> FuturesStrategy {
    return FuturesStrategy::create().with_leg(leg(std::move(futures_label), futures_entry, -contracts, contract_size, 0.0));
}

auto long_hedge(std::string futures_label, double futures_entry, double contracts,
                double contract_size) -> FuturesStrategy {
    return FuturesStrategy::create().with_leg(leg(std::move(futures_label), futures_entry, contracts, contract_size, 0.0));
}

auto cash_and_carry(std::string spot_label, double spot_entry, std::string futures_label,
                    double futures_entry, double quantity, double contract_size) -> FuturesStrategy {
    // Long the cash leg, short the futures leg: the classic arbitrage that locks in the
    // basis. Both legs are modelled as futures-style linear positions so pnl_at can take
    // each leg's own settlement (spot converges to futures at delivery -> the locked basis).
    return FuturesStrategy::create()
        .with_leg(leg(std::move(spot_label), spot_entry, quantity, contract_size, 0.0))
        .with_leg(leg(std::move(futures_label), futures_entry, -quantity, contract_size, 0.0));
}

}  // namespace nimblecas::futures
