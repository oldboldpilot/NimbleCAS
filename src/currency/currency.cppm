// NimbleCAS currency & foreign-exchange — exact money, rate graphs, cross rates, arbitrage.
// @author Olumuyiwa Oluwasanmi
//
// SCOPE. A tagged Money amount (a BigDecimal at a stated scale + an ISO-4217-style code),
// an exchange-rate GRAPH (the "computational graph" for FX problems: currencies are nodes,
// quoted pairs are edges), exact cross-rate derivation by shortest-path product through the
// graph, currency conversion that quantizes to the target's minor unit under an explicit
// rounding mode, triangular-arbitrage detection (is the product around a cycle != 1?), and
// covered-interest-parity forward rates. Bid/ask spreads are supported via directional
// edges (a->b and b->a need not be reciprocals).
//
// HONESTY (config/cpp_details.txt Rule 32). Rates and cross rates are EXACT rationals — no
// rounding until the final convert() quantizes to money under a caller-supplied scale and
// Rounding. A missing conversion path returns MathError::not_implemented (there is no route
// through the quoted graph), never a fabricated rate. Money arithmetic refuses to add across
// currencies (domain_error) rather than silently combining apples and oranges.

module;
#include <cassert>

export module nimblecas.currency;

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.bigdecimal;

export namespace nimblecas::currency {

// A signed amount of a specific currency. The code is an opaque tag (e.g. "USD", "EUR");
// arithmetic across differing codes is refused. Amounts are exact decimals (BigDecimal).
class Money {
public:
    [[nodiscard]] static auto of(BigDecimal amount, std::string code) -> Money {
        return Money{std::move(amount), std::move(code)};
    }
    // Parse "123.45" into the given currency at the literal's own scale.
    [[nodiscard]] static auto parse(std::string_view amount, std::string code) -> Result<Money> {
        auto a = BigDecimal::from_string(amount);
        if (!a) { return make_error<Money>(a.error()); }
        return Money{std::move(*a), std::string{code}};
    }

    [[nodiscard]] auto amount() const noexcept -> const BigDecimal& { return amount_; }
    [[nodiscard]] auto code() const noexcept -> const std::string& { return code_; }
    [[nodiscard]] auto is_zero() const noexcept -> bool { return amount_.is_zero(); }
    [[nodiscard]] auto negate() const -> Money { return Money{amount_.negate(), code_}; }

    // Same-currency addition/subtraction (exact). Differing codes -> domain_error.
    [[nodiscard]] auto add(const Money& o) const -> Result<Money> {
        if (code_ != o.code_) { return make_error<Money>(MathError::domain_error); }
        return Money{amount_.add(o.amount_), code_};
    }
    [[nodiscard]] auto subtract(const Money& o) const -> Result<Money> {
        if (code_ != o.code_) { return make_error<Money>(MathError::domain_error); }
        return Money{amount_.subtract(o.amount_), code_};
    }
    // Re-express at a stated minor-unit scale (e.g. 2 for cents) under an explicit mode.
    [[nodiscard]] auto rounded(std::int32_t scale, Rounding mode) const -> Money {
        return Money{amount_.quantize(scale, mode), code_};
    }
    [[nodiscard]] auto to_string() const -> std::string {
        return amount_.to_string() + " " + code_;
    }

private:
    Money(BigDecimal amount, std::string code)
        : amount_(std::move(amount)), code_(std::move(code)) {}
    BigDecimal amount_{};
    std::string code_{};
};

// A directed quoted rate base->quote: 1 unit of `base` buys `rate` units of `quote`. Exact.
struct Quote {
    std::string base;
    std::string quote;
    BigRational rate;  // must be > 0
};

// The FX rate graph. Add directed quotes (optionally auto-adding the reciprocal); query
// direct or cross rates, convert money, detect triangular arbitrage.
class RateTable {
public:
    [[nodiscard]] static auto create() -> RateTable { return RateTable{}; }

    // Add base->quote at `rate`. When add_reciprocal, also add quote->base at 1/rate (a
    // frictionless market); omit it to model a bid/ask spread with independent directions.
    // rate <= 0 -> domain_error.
    [[nodiscard]] auto add(std::string base, std::string quote, const BigRational& rate,
                           bool add_reciprocal = true) -> Result<std::reference_wrapper<RateTable>>;

    // Directly-quoted rate base->quote, if present.
    [[nodiscard]] auto direct_rate(std::string_view base, std::string_view quote) const
        -> std::optional<BigRational>;

    // Exact cross rate base->quote as the product along the SHORTEST path through the graph
    // (fewest hops; ties broken by lexicographic order for determinism). No route ->
    // not_implemented.
    [[nodiscard]] auto cross_rate(std::string_view base, std::string_view quote) const
        -> Result<BigRational>;

    // Convert an amount into `to_code`, quantized to `scale` minor units under `mode`.
    [[nodiscard]] auto convert(const Money& amount, std::string_view to_code, std::int32_t scale,
                               Rounding mode) const -> Result<Money>;

    // Product of rates around the cycle a->b->c->a. Exactly 1 means no triangular arbitrage;
    // > 1 means a profitable loop starting/ending in `a` (return value is the gross multiple).
    [[nodiscard]] auto triangular_product(std::string_view a, std::string_view b,
                                          std::string_view c) const -> Result<BigRational>;
    // True iff triangular_product deviates from 1 by more than `tolerance` (a BigRational, so
    // an exact zero-tolerance test is possible).
    [[nodiscard]] auto has_triangular_arbitrage(std::string_view a, std::string_view b,
                                                std::string_view c,
                                                const BigRational& tolerance) const -> Result<bool>;

private:
    RateTable() = default;
    // Adjacency: from-code -> list of (to-code, rate). Small graphs (dozens of currencies),
    // so a hashed adjacency with BFS pathfinding is ample.
    std::unordered_map<std::string, std::vector<std::pair<std::string, BigRational>>> adj_{};
};

// Covered-interest-parity forward rate: F = S * (1 + r_quote*t) / (1 + r_base*t), with S the
// spot (units of quote per unit of base), r_* the simple interest rates over the tenor, and t
// the year fraction. Exact over Q. A zero base-leg denominator -> division_by_zero.
[[nodiscard]] auto forward_rate(const BigRational& spot, const BigRational& rate_base,
                                const BigRational& rate_quote, const BigRational& year_fraction)
    -> Result<BigRational>;

}  // namespace nimblecas::currency

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::currency {

auto RateTable::add(std::string base, std::string quote, const BigRational& rate,
                    bool add_reciprocal) -> Result<std::reference_wrapper<RateTable>> {
    if (rate.sign() <= 0) {
        return make_error<std::reference_wrapper<RateTable>>(MathError::domain_error);
    }
    adj_[base].emplace_back(quote, rate);
    if (add_reciprocal) {
        auto inv = rate.reciprocal();
        assert(inv.has_value() && "positive rate has a reciprocal");
        adj_[quote].emplace_back(base, *inv);
    }
    return std::ref(*this);
}

auto RateTable::direct_rate(std::string_view base, std::string_view quote) const
    -> std::optional<BigRational> {
    const auto it = adj_.find(std::string{base});
    if (it == adj_.end()) { return std::nullopt; }
    for (const auto& [to, rate] : it->second) {
        if (to == quote) { return rate; }
    }
    return std::nullopt;
}

auto RateTable::cross_rate(std::string_view base, std::string_view quote) const
    -> Result<BigRational> {
    const std::string from{base};
    const std::string to{quote};
    if (from == to) { return BigRational::from_int(1); }
    // BFS over currencies, accumulating the exact product of edge rates to each node. The
    // first time `to` is dequeued we have a shortest-hop path; deterministic because
    // neighbours are visited in insertion order and nodes are marked on first discovery.
    std::unordered_map<std::string, BigRational> product;
    std::deque<std::string> queue;
    product.emplace(from, BigRational::from_int(1));
    queue.push_back(from);
    while (!queue.empty()) {
        const std::string cur = queue.front();
        queue.pop_front();
        const BigRational cur_rate = product.at(cur);
        const auto it = adj_.find(cur);
        if (it == adj_.end()) { continue; }
        for (const auto& [next, rate] : it->second) {
            if (product.contains(next)) { continue; }
            BigRational next_rate = cur_rate.multiply(rate);
            if (next == to) { return next_rate; }
            product.emplace(next, std::move(next_rate));
            queue.push_back(next);
        }
    }
    return make_error<BigRational>(MathError::not_implemented);  // no route through the graph
}

auto RateTable::convert(const Money& amount, std::string_view to_code, std::int32_t scale,
                        Rounding mode) const -> Result<Money> {
    auto rate = cross_rate(amount.code(), to_code);
    if (!rate) { return make_error<Money>(rate.error()); }
    const BigRational converted = amount.amount().to_bigrational().multiply(*rate);
    return Money::of(BigDecimal::from_bigrational(converted, scale, mode), std::string{to_code});
}

auto RateTable::triangular_product(std::string_view a, std::string_view b, std::string_view c) const
    -> Result<BigRational> {
    auto ab = cross_rate(a, b);
    if (!ab) { return ab; }
    auto bc = cross_rate(b, c);
    if (!bc) { return bc; }
    auto ca = cross_rate(c, a);
    if (!ca) { return ca; }
    return ab->multiply(*bc).multiply(*ca);
}

auto RateTable::has_triangular_arbitrage(std::string_view a, std::string_view b, std::string_view c,
                                         const BigRational& tolerance) const -> Result<bool> {
    auto prod = triangular_product(a, b, c);
    if (!prod) { return make_error<bool>(prod.error()); }
    const BigRational deviation = prod->subtract(BigRational::from_int(1));
    // |deviation| > tolerance.
    const BigRational mag = deviation.sign() < 0 ? deviation.negate() : deviation;
    return mag > tolerance;
}

auto forward_rate(const BigRational& spot, const BigRational& rate_base,
                  const BigRational& rate_quote, const BigRational& year_fraction)
    -> Result<BigRational> {
    const BigRational one = BigRational::from_int(1);
    const BigRational base_leg = one.add(rate_base.multiply(year_fraction));
    if (base_leg.is_zero()) { return make_error<BigRational>(MathError::division_by_zero); }
    const BigRational quote_leg = one.add(rate_quote.multiply(year_fraction));
    return spot.multiply(quote_leg).divide(base_leg);
}

}  // namespace nimblecas::currency
