// NimbleCAS bond trading strategies — portfolio construction and relative value.
// @author Olumuyiwa Oluwasanmi
//
// SCOPE. The strategy layer above nimblecas.finance / nimblecas.fixedincome bond
// analytics: weighted portfolio duration / convexity / yield; the classic curve
// structures built from per-bond durations — the two-bond BARBELL that hits a target
// duration, the duration-and-cash-neutral BUTTERFLY (long wings, short body), and the
// duration-neutral HEDGE RATIO between two bonds; plus carry-and-roll-down return. It
// takes per-bond analytics as inputs (price, modified duration, convexity, yield) rather
// than re-deriving them — those come from nimblecas.finance (`bond_price`,
// `bond_mduration`, `bond_convexity`) — so this module is a pure, exact combinatorial
// layer over them.
//
// HONESTY (config/cpp_details.txt Rule 32, AGENTS.md). The weighting algebra here is
// EXACT on its `double` inputs (linear solves of small systems); it introduces no new
// approximation beyond the durations it is handed. A degenerate system (equal durations,
// zero denominator) returns an honest `MathError` — `division_by_zero` when a hedge/weight
// denominator vanishes, `domain_error` on a length mismatch or non-normalisable weights —
// never a fabricated ratio. All failure rides the railway; nothing throws.

export module nimblecas.bondstrat;

import std;
import nimblecas.core;

export namespace nimblecas::bondstrat {

// Weighted portfolio statistic Σ wᵢ·xᵢ (duration, convexity, or yield). `weights` and
// `values` must be the same non-empty length -> else domain_error. Weights need not sum
// to 1 (the caller decides); this is the raw linear combination.
[[nodiscard]] auto weighted_average(std::span<const double> weights, std::span<const double> values)
    -> Result<double>;

// Two-bond barbell: the weights (w_short, w_long), summing to 1, whose blended modified
// duration equals `target_duration`. w_long = (target − D_short)/(D_long − D_short).
// D_long == D_short -> division_by_zero (no unique blend).
[[nodiscard]] auto barbell_weights(double target_duration, double dur_short, double dur_long)
    -> Result<std::array<double, 2>>;

// Duration-neutral hedge ratio: units of bond B to SHORT per unit of bond A held, so the
// combined DV01 is zero. ratio = (D_a·P_a)/(D_b·P_b). A zero denominator (D_b·P_b == 0) ->
// division_by_zero.
[[nodiscard]] auto duration_hedge_ratio(double dur_a, double price_a, double dur_b, double price_b)
    -> Result<double>;

// Duration- and cash-neutral butterfly: long the two wings, short one unit of the body.
// Returns the wing weights (w_lo, w_hi) solving
//   w_lo + w_hi = 1                       (cash-neutral against the unit body)
//   w_lo·D_lo + w_hi·D_hi = D_body        (duration-neutral)
// D_lo == D_hi -> division_by_zero.
[[nodiscard]] auto butterfly_weights(double dur_lo, double dur_body, double dur_hi)
    -> Result<std::array<double, 2>>;

// Total carry-and-roll-down return over a horizon, as a fraction of the entry price:
//   (price_rolled − price_now + coupon_income) / price_now.
// `price_rolled` is the bond's price after time passes and it rolls down a static curve.
// price_now <= 0 -> domain_error.
[[nodiscard]] auto carry_roll_return(double price_now, double price_rolled, double coupon_income)
    -> Result<double>;

// First-order P&L of a duration/convexity position under a parallel yield shift `dy`:
//   −D·dy + 0.5·C·dy²  (per unit of price). Total on finite inputs.
[[nodiscard]] auto duration_convexity_pnl(double modified_duration, double convexity, double dy)
    noexcept -> double;

}  // namespace nimblecas::bondstrat

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::bondstrat {

auto weighted_average(std::span<const double> weights, std::span<const double> values)
    -> Result<double> {
    if (weights.empty() || weights.size() != values.size()) {
        return make_error<double>(MathError::domain_error);
    }
    double acc = 0.0;
    for (std::size_t i = 0; i < weights.size(); ++i) { acc += weights[i] * values[i]; }
    return acc;
}

auto barbell_weights(double target_duration, double dur_short, double dur_long)
    -> Result<std::array<double, 2>> {
    const double denom = dur_long - dur_short;
    if (denom == 0.0) { return make_error<std::array<double, 2>>(MathError::division_by_zero); }
    const double w_long = (target_duration - dur_short) / denom;
    return std::array<double, 2>{1.0 - w_long, w_long};
}

auto duration_hedge_ratio(double dur_a, double price_a, double dur_b, double price_b)
    -> Result<double> {
    const double denom = dur_b * price_b;
    if (denom == 0.0) { return make_error<double>(MathError::division_by_zero); }
    return (dur_a * price_a) / denom;
}

auto butterfly_weights(double dur_lo, double dur_body, double dur_hi)
    -> Result<std::array<double, 2>> {
    // Solve { w_lo + w_hi = 1 ; w_lo·D_lo + w_hi·D_hi = D_body }.
    // -> w_hi = (D_body − D_lo)/(D_hi − D_lo), w_lo = 1 − w_hi.
    const double denom = dur_hi - dur_lo;
    if (denom == 0.0) { return make_error<std::array<double, 2>>(MathError::division_by_zero); }
    const double w_hi = (dur_body - dur_lo) / denom;
    return std::array<double, 2>{1.0 - w_hi, w_hi};
}

auto carry_roll_return(double price_now, double price_rolled, double coupon_income)
    -> Result<double> {
    if (price_now <= 0.0) { return make_error<double>(MathError::domain_error); }
    return (price_rolled - price_now + coupon_income) / price_now;
}

auto duration_convexity_pnl(double modified_duration, double convexity, double dy) noexcept
    -> double {
    return -modified_duration * dy + 0.5 * convexity * dy * dy;
}

}  // namespace nimblecas::bondstrat
