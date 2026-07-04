// NimbleCAS exact power-series ODE systems & higher-order ODE solver.
// @author Olumuyiwa Oluwasanmi
//
// This module generalises the scalar perturbation solver (nimblecas.perturbation) to
// autonomous FIRST-ORDER SYSTEMS and to scalar HIGHER-ORDER initial value problems,
// entirely as EXACT arithmetic over Q in the truncated ring Q[[x]]/(x^N). The engine is
// nimblecas.powerseries: x is the independent variable, the linear operator L = d/dx is
// PowerSeries::derivative(), and its exact right inverse L^{-1} (definite integral from 0
// with zero constant of integration) is PowerSeries::integrate().
//
// FIRST-ORDER SYSTEM. For the autonomous IVP
//
//     u_i'(x) = f_i(u_1, ..., u_n),   u_i(0) = u0[i]   (u0[i] a Rational constant),
//
// the vector field f is supplied as a SystemOperator: given the current vector of solution
// series u it returns [f_1(u), ..., f_n(u)] as series. Rewriting the IVP as the vector
// fixed point
//
//     u = u0 + L^{-1}[ f(u) ]
//
// turns the whole solution into exact truncated power-series arithmetic. We solve it by
// the graded Picard/Taylor recursion, building all n components simultaneously: start from
// the constant vector u^(0) = u0, then repeatedly set u <- u0 + integrate(f(u)). Because
// L^{-1} raises the x-degree by one, one pass fixes one further Taylor coefficient of every
// component: if u^(m) agrees with the true solution through degree m then f(u^(m)) agrees
// through degree m (every powerseries operation preserves the agreement degree of its
// inputs) and integrating lifts the agreement to degree m+1. Hence `order` passes fix all
// coefficients x^0 .. x^{order-1}; the recursion has reached its fixed point and further
// passes leave the truncation unchanged.
//
// HIGHER-ORDER REDUCTION. The scalar k-th order IVP
//
//     u^{(k)} = f(u, u', ..., u^{(k-1)}),   [u(0), u'(0), ..., u^{(k-1)}(0)] = initial,
//
// is reduced to the first-order system y_0 = u, y_1 = u', ..., y_{k-1} = u^{(k-1)} with
// y_j' = y_{j+1} for j < k-1 and y_{k-1}' = f(y_0, ..., y_{k-1}), then solved by
// solve_first_order_system with y_j(0) = initial[j]; the solution component y_0 is the
// requested u. k is the length of `initial`.
//
// HONESTY. Everything here is EXACT over Q as a power series TRUNCATED to the requested
// order. For right-hand sides built from the available powerseries operations
// (polynomial/rational via multiply and inverse/divide, and the analytic exp/log/compose
// coefficient recurrences) these are the exact Taylor coefficients of the solution to that
// order. The solver makes NO claim about the radius of convergence or about any closed-form
// solution; it mirrors the exactness discipline of nimblecas.perturbation. Rule 32 railway:
// every powerseries Result error is propagated; bad order / vector sizes / order mismatch
// are MathError::domain_error.

export module nimblecas.ode;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.powerseries;

export namespace nimblecas {

// The autonomous vector field f of the system u_i' = f_i(u): consumes the current vector
// of solution series [u_1, ..., u_n] and returns [f_1(u), ..., f_n(u)] as series (on the
// railway). All returned series must share the order of the inputs and their number must
// match the system size (else domain_error).
using SystemOperator =
    std::function<Result<std::vector<PowerSeries>>(const std::vector<PowerSeries>&)>;

// The right-hand side f of a scalar higher-order IVP u^{(k)} = f(u, u', ..., u^{(k-1)}):
// consumes the current derivative vector [y_0, ..., y_{k-1}] = [u, u', ..., u^{(k-1)}] and
// returns u^{(k)} as a series (on the railway).
using HigherOrderOperator = std::function<Result<PowerSeries>(const std::vector<PowerSeries>&)>;

// Solve the autonomous first-order system u_i'(x) = f_i(u_1, ..., u_n), u_i(0) = u0[i], as
// exact truncated power series with `order` coefficients (terms x^0 .. x^{order-1}). n is
// the length of u0. Returns the n solution series in the same order as u0.
//
// Method: the graded Picard/Taylor recursion u <- u0 + integrate(f(u)) started from the
// constant vector u0 and run `order` passes; each pass fixes one further Taylor coefficient
// of every component, so the result is the exact Taylor polynomial of the solution to the
// requested order.
//
// Failure modes (MathError::domain_error): order == 0, empty u0, an empty operator, an f
// whose output length differs from n, or an f whose output series do not have order `order`
// (f must preserve the order). Any error raised by f or by the powerseries engine is
// propagated.
[[nodiscard]] auto solve_first_order_system(SystemOperator f, std::vector<Rational> u0,
                                            std::size_t order)
    -> Result<std::vector<PowerSeries>>;

// Solve the scalar k-th order IVP u^{(k)} = f(u, u', ..., u^{(k-1)}) with initial data
// initial = [u(0), u'(0), ..., u^{(k-1)}(0)] (so k = initial.size()), as an exact truncated
// power series with `order` coefficients. Reduces to the first-order system y_j' = y_{j+1}
// (j < k-1), y_{k-1}' = f(y) via solve_first_order_system and returns the component y_0 = u.
//
// Failure modes (MathError::domain_error): order == 0, empty `initial` (k must be >= 1), or
// an empty operator. Errors from f or the engine are propagated.
[[nodiscard]] auto solve_higher_order(HigherOrderOperator f, std::vector<Rational> initial,
                                      std::size_t order) -> Result<PowerSeries>;

// Exact Horner evaluation of the TRUNCATED series s at the rational point x, i.e. the value
// of the polynomial c_0 + c_1 x + ... + c_{N-1} x^{N-1}. This is the value of the truncated
// series, NOT the exact ODE solution unless the series terminates; it is provided for
// checking and sampling. Any overflow in the rational arithmetic is propagated.
[[nodiscard]] auto evaluate(const PowerSeries& s, const Rational& x) -> Result<Rational>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

auto solve_first_order_system(SystemOperator f, std::vector<Rational> u0, std::size_t order)
    -> Result<std::vector<PowerSeries>> {
    using Vec = std::vector<PowerSeries>;
    if (order == 0 || !f || u0.empty()) {
        return make_error<Vec>(MathError::domain_error);
    }
    const std::size_t n = u0.size();

    // Constant seeds u0[i] + O(x^order), reused as the additive base of every pass.
    Vec base;
    base.reserve(n);
    for (const auto& c : u0) {
        auto seed = PowerSeries::constant(c, order);
        if (!seed) {
            return make_error<Vec>(seed.error());
        }
        base.push_back(std::move(*seed));
    }

    // Current iterate u; start at the constant vector u^(0) = u0.
    Vec u = base;

    // Graded Picard/Taylor: u <- u0 + L^{-1}[f(u)]. One pass fixes one further coefficient
    // of every component, so `order` passes reach the fixed point of the truncated system.
    for (std::size_t pass = 0; pass < order; ++pass) {
        auto rhs = f(u);
        if (!rhs) {
            return make_error<Vec>(rhs.error());
        }
        if (rhs->size() != n) {
            return make_error<Vec>(MathError::domain_error);  // f must return n components
        }
        Vec next;
        next.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            if ((*rhs)[i].order() != order) {
                return make_error<Vec>(MathError::domain_error);  // f must preserve the order
            }
            auto integ = (*rhs)[i].integrate();  // L^{-1}[f_i(u)]
            if (!integ) {
                return make_error<Vec>(integ.error());
            }
            auto u_i = base[i].add(*integ);  // u0[i] + L^{-1}[f_i(u)]
            if (!u_i) {
                return make_error<Vec>(u_i.error());
            }
            next.push_back(std::move(*u_i));
        }
        u = std::move(next);
    }
    return u;
}

auto solve_higher_order(HigherOrderOperator f, std::vector<Rational> initial, std::size_t order)
    -> Result<PowerSeries> {
    if (order == 0 || !f || initial.empty()) {
        return make_error<PowerSeries>(MathError::domain_error);
    }
    const std::size_t k = initial.size();

    // Reduce to the companion first-order system: y_j' = y_{j+1} for j < k-1 and
    // y_{k-1}' = f(y_0, ..., y_{k-1}).
    SystemOperator companion =
        [f, k](const std::vector<PowerSeries>& y) -> Result<std::vector<PowerSeries>> {
        using Vec = std::vector<PowerSeries>;
        if (y.size() != k) {
            return make_error<Vec>(MathError::domain_error);
        }
        Vec rhs;
        rhs.reserve(k);
        for (std::size_t j = 0; j + 1 < k; ++j) {
            rhs.push_back(y[j + 1]);  // y_j' = y_{j+1}
        }
        auto top = f(y);  // y_{k-1}' = f(y)
        if (!top) {
            return make_error<Vec>(top.error());
        }
        rhs.push_back(std::move(*top));
        return rhs;
    };

    auto sol = solve_first_order_system(std::move(companion), std::move(initial), order);
    if (!sol) {
        return make_error<PowerSeries>(sol.error());
    }
    return std::move(sol->front());  // y_0 = u
}

auto evaluate(const PowerSeries& s, const Rational& x) -> Result<Rational> {
    const std::size_t n = s.order();
    // Horner from the top coefficient down: acc <- acc * x + c_k.
    Rational acc = s.coefficient(n - 1);
    for (std::size_t idx = n - 1; idx-- > 0;) {
        auto scaled = acc.multiply(x);
        if (!scaled) {
            return make_error<Rational>(scaled.error());
        }
        auto sum = scaled->add(s.coefficient(idx));
        if (!sum) {
            return make_error<Rational>(sum.error());
        }
        acc = *sum;
    }
    return acc;
}

}  // namespace nimblecas
