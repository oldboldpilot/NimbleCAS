// NimbleCAS exact delay differential equations via the method of steps.
// @author Olumuyiwa Oluwasanmi
//
// This module solves the scalar delay differential equation (DDE)
//
//     u'(t) = f(t, u(t), u(t - tau)),    u(t) = history(t) on [-tau, 0],
//
// EXACTLY over Q by the classical METHOD OF STEPS. The delay tau > 0 partitions the
// forward axis into intervals I_k = [k*tau, (k+1)*tau]. On I_k the delayed argument
// t - tau lands in the PREVIOUS interval I_{k-1} (or in the history for k = 0), where the
// solution is ALREADY KNOWN. The delayed term is therefore a known function of t and the
// DDE collapses, on each interval, to an ordinary initial value problem
//
//     u'(s) = f(u(s), u_delayed(s), s),   u(0) = <value carried over from the previous
//                                                 interval at its right endpoint>,
//
// written in the LOCAL variable s = t - k*tau in [0, tau]. Each such ODE is solved by the
// exact graded power-series (Picard) recursion of nimblecas.powerseries: the fixed point
// of u = u(0) + L^{-1}[ f(u, u_delayed, s) ] where L^{-1} = PowerSeries::integrate. Since
// each integration lifts the correctly-resolved degree by one, `order` sweeps of the
// Picard map pin down all `order` coefficients of the interval's series.
//
// RE-CENTERING (the crux). Represent the solution as one PowerSeries per interval,
// expanded in that interval's own local s. The delayed term on interval k, expressed in
// local s, is simply the PREVIOUS piece read at the SAME local coordinate s: for t in
// [k*tau, (k+1)*tau] we have t - tau in [(k-1)*tau, k*tau], whose local coordinate is
// (t - tau) - (k-1)*tau = t - k*tau = s. So no shifting of coefficients is needed —
// u_delayed_local IS pieces[k-1] verbatim. The history plays the role of "piece -1": the
// caller supplies it already expressed in that local coordinate (history_local(s) =
// history(s - tau), s in [0, tau]), so on interval 0 the delayed term is history_local and
// the initial value u(0) is history_local evaluated at s = tau (= history(0)). The same
// rule — initial value = previous piece at s = tau — then threads continuity through every
// interval uniformly.
//
// HONESTY. For polynomial/rational history and f this is EXACT over Q: a DDE with
// polynomial history is piecewise polynomial, and each piece is recovered exactly up to
// the truncation order (terms of degree >= order are discarded, an implicit O(s^order)
// tail per interval). Nothing is claimed beyond that truncation, and no floating point is
// used anywhere. Rule 32 railway: every powerseries/Rational Result error is propagated;
// bad arguments (tau <= 0, order 0, num_intervals 0) are MathError::domain_error.

export module nimblecas.dde;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.powerseries;

export namespace nimblecas {

// The right-hand side f of u'(t) = f(t, u(t), u(t - tau)), presented in the LOCAL frame of
// the current interval: it receives the current solution series u_local(s), the delayed
// solution series u_delayed_local(s) (the previous piece at the same local s), and the
// local independent variable s = PowerSeries::variable(order), and returns f(...) as the
// series u'(s). All three arguments share the working order and f must preserve it. E.g.
//   u' = -u(t-tau):   (u, ud, s) -> ud.scale(-1)
//   u' =  u(t-tau):   (u, ud, s) -> ud
//   u' =  s*u(t-tau): (u, ud, s) -> s.multiply(ud)
using DdeOperator = std::function<Result<PowerSeries>(
    const PowerSeries& u_local, const PowerSeries& u_delayed_local, const PowerSeries& s)>;

// The piecewise solution: pieces[k] is the solution on interval [k*tau, (k+1)*tau]
// expanded in the local variable s = t - k*tau (s in [0, tau]). tau is the (positive)
// delay shared by every interval.
struct DdeSolution {
    std::vector<PowerSeries> pieces;
    Rational tau;
};

// Exact truncated value of a power series at a rational point: sum_{i} c_i * s^i over the
// retained coefficients, by Horner over Q. Never fails on the series itself (only on a
// Rational overflow along the way).
[[nodiscard]] auto evaluate_series(const PowerSeries& series, const Rational& s)
    -> Result<Rational>;

// Solve u'(t) = f(t, u(t), u(t - tau)) on [0, num_intervals*tau] by the method of steps,
// returning one order-`order` PowerSeries per interval in local coordinates.
// `history_on_first_interval` is the history already expressed in interval 0's local frame
// (history_local(s) = history(s - tau) for s in [0, tau]); it is normalised to the working
// `order`. domain_error if tau <= 0, order == 0, or num_intervals == 0; any powerseries or
// Rational error raised while solving is propagated.
[[nodiscard]] auto solve_method_of_steps(DdeOperator f,
                                         PowerSeries history_on_first_interval, Rational tau,
                                         std::size_t num_intervals, std::size_t order)
    -> Result<DdeSolution>;

// Evaluate the piecewise solution at a rational time t in [0, num_intervals*tau]: locate
// the interval k = floor(t / tau) and return pieces[k] at the local s = t - k*tau (the
// right endpoint t = num_intervals*tau is served by the last piece at s = tau). t < 0 or
// t beyond the solved range is a domain_error, as is an empty solution.
[[nodiscard]] auto evaluate(const DdeSolution& sol, const Rational& t) -> Result<Rational>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

auto evaluate_series(const PowerSeries& series, const Rational& s) -> Result<Rational> {
    const auto coeffs = series.coefficients();  // index i is c_i; at least one entry
    // Horner from the top: acc = c_{N-1}; acc = acc*s + c_i down to c_0.
    Rational acc = coeffs.back();
    for (std::size_t i = coeffs.size() - 1; i-- > 0;) {
        auto scaled = acc.multiply(s);
        if (!scaled) {
            return make_error<Rational>(scaled.error());
        }
        auto sum = scaled->add(coeffs[i]);
        if (!sum) {
            return make_error<Rational>(sum.error());
        }
        acc = *sum;
    }
    return acc;
}

namespace {

// Solve one interval's IVP u'(s) = f(u, u_delayed, s), u(0) = ic, by the graded Picard
// fixed point u = ic + L^{-1}[f(u, u_delayed, s)]. Each iteration resolves one further
// coefficient (integration lifts the settled degree by one), so `order` sweeps suffice for
// all `order` coefficients. Exact for polynomial/rational f.
auto solve_interval(const DdeOperator& f, const PowerSeries& u_delayed, const Rational& ic,
                    const PowerSeries& s, std::size_t order) -> Result<PowerSeries> {
    auto ic_series = PowerSeries::constant(ic, order);
    if (!ic_series) {
        return make_error<PowerSeries>(ic_series.error());
    }
    PowerSeries u = *ic_series;  // zeroth guess: the constant initial value
    for (std::size_t iter = 0; iter < order; ++iter) {
        auto rhs = f(u, u_delayed, s);  // u'(s) = f(u, u_delayed, s)
        if (!rhs) {
            return make_error<PowerSeries>(rhs.error());
        }
        if (rhs->order() != order) {
            return make_error<PowerSeries>(MathError::domain_error);  // f must preserve order
        }
        auto integ = rhs->integrate();  // L^{-1}: zero constant of integration
        if (!integ) {
            return make_error<PowerSeries>(integ.error());
        }
        auto u_next = ic_series->add(*integ);  // reinstate u(0) = ic
        if (!u_next) {
            return make_error<PowerSeries>(u_next.error());
        }
        u = *u_next;
    }
    return u;
}

}  // namespace

auto solve_method_of_steps(DdeOperator f, PowerSeries history_on_first_interval, Rational tau,
                           std::size_t num_intervals, std::size_t order)
    -> Result<DdeSolution> {
    if (!f || order == 0 || num_intervals == 0 || tau.numerator() <= 0) {
        return make_error<DdeSolution>(MathError::domain_error);
    }

    // The local variable s = t - k*tau, shared by every interval at the working order.
    auto s = PowerSeries::variable(order);
    if (!s) {
        return make_error<DdeSolution>(s.error());
    }

    // Normalise the history to the working order; it is the "piece -1" that seeds both the
    // interval-0 delayed term and the interval-0 initial value.
    std::vector<Rational> hist_coeffs(history_on_first_interval.coefficients().begin(),
                                      history_on_first_interval.coefficients().end());
    auto prev = PowerSeries::from_coeffs(std::move(hist_coeffs), order);
    if (!prev) {
        return make_error<DdeSolution>(prev.error());
    }

    std::vector<PowerSeries> pieces;
    pieces.reserve(num_intervals);
    for (std::size_t k = 0; k < num_intervals; ++k) {
        // Continuity: u(0) on this interval = previous piece at its right endpoint s = tau.
        auto ic = evaluate_series(*prev, tau);
        if (!ic) {
            return make_error<DdeSolution>(ic.error());
        }
        // The delayed term in local s is exactly the previous piece read at the same s.
        auto piece = solve_interval(f, *prev, *ic, *s, order);
        if (!piece) {
            return make_error<DdeSolution>(piece.error());
        }
        pieces.push_back(*piece);
        prev = std::move(piece);
    }
    return DdeSolution{.pieces = std::move(pieces), .tau = tau};
}

auto evaluate(const DdeSolution& sol, const Rational& t) -> Result<Rational> {
    if (sol.pieces.empty() || sol.tau.numerator() <= 0) {
        return make_error<Rational>(MathError::domain_error);
    }
    // Interval index k = floor(t / tau), computed exactly over Q.
    auto q = t.divide(sol.tau);  // tau != 0, so this cannot divide by zero
    if (!q) {
        return make_error<Rational>(q.error());
    }
    const std::int64_t n = q->numerator();
    const std::int64_t d = q->denominator();  // canonical: d > 0
    std::int64_t k = n / d;
    const std::int64_t rem = n % d;
    if (rem != 0 && n < 0) {
        --k;  // floor division for negative t
    }
    const auto size = static_cast<std::int64_t>(sol.pieces.size());

    std::size_t idx = 0;
    Rational s;  // 0
    if (k < 0) {
        return make_error<Rational>(MathError::domain_error);  // before the solved range
    }
    if (k >= size) {
        // Only the exact right endpoint t = size*tau is admissible: last piece at s = tau.
        if (k == size && rem == 0) {
            idx = sol.pieces.size() - 1;
            s = sol.tau;
        } else {
            return make_error<Rational>(MathError::domain_error);  // beyond the solved range
        }
    } else {
        idx = static_cast<std::size_t>(k);
        // Local coordinate s = t - k*tau.
        auto shift = Rational::from_int(k).multiply(sol.tau);
        if (!shift) {
            return make_error<Rational>(shift.error());
        }
        auto local = t.subtract(*shift);
        if (!local) {
            return make_error<Rational>(local.error());
        }
        s = *local;
    }
    return evaluate_series(sol.pieces[idx], s);
}

}  // namespace nimblecas
