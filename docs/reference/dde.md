# `nimblecas.dde` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/dde/dde.cppm`

Exact delay differential equations by the classical **method of steps**. The
module solves the scalar DDE

```
u'(t) = f(t, u(t), u(t - tau)),   u(t) = history(t) on [-tau, 0]
```

**exactly over Q**. A positive delay `tau` partitions the forward axis into
intervals `I_k = [k·tau, (k+1)·tau]`; on `I_k` the delayed argument `t - tau`
lands in the previous interval `I_{k-1}` (or in the history for `k = 0`), where
the solution is *already known*. The delayed term is therefore a known function
of `t`, and on each interval the DDE collapses to an ordinary initial-value
problem `u'(s) = f(u, u_delayed, s)` in the local variable `s = t - k·tau ∈
[0, tau]`. Each such IVP is solved by the exact graded power-series (Picard)
recursion of the `powerseries` layer: the fixed point of
`u = u(0) + L⁻¹[f(u, u_delayed, s)]` where `L⁻¹` is `PowerSeries::integrate`.
The solution is carried as **one `PowerSeries` per interval** in that interval's
own local `s`, so no coefficient shifting is ever needed — the delayed term on
interval `k` is *literally* the previous piece read at the same local `s`.

**Honesty boundary.** For polynomial/rational `history` and `f`, a DDE with
polynomial history is piecewise polynomial and each piece is recovered
**exactly over Q up to the truncation order** `order`: terms of degree `≥ order`
are discarded, an implicit `O(s^order)` tail per interval. Nothing is claimed
beyond that truncation, and **no floating point is used anywhere**. The
per-interval Picard map converges in exactly `order` sweeps because each
integration lifts the settled degree by one; a purely delayed operator (whose
right-hand side ignores the current state) settles in a single integration, but
an operator that reads `u(t)` genuinely needs all `order` sweeps. Rule 32
railway: every `powerseries`/`Rational` `Result` error is propagated, and bad
arguments (`tau ≤ 0`, `order == 0`, `num_intervals == 0`, an empty callable) are
`MathError::domain_error`.

```cpp
import nimblecas.dde;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), and `powerseries` (the
exact graded power-series arithmetic that carries each interval's polynomial).

## `DdeOperator` — the right-hand side in local coordinates

```cpp
using DdeOperator = std::function<Result<PowerSeries>(
    const PowerSeries& u_local, const PowerSeries& u_delayed_local, const PowerSeries& s)>;
```

The operator `f` of `u'(t) = f(t, u(t), u(t - tau))`, presented in the **local
frame** of the current interval. It receives the current solution series
`u_local(s)`, the delayed solution series `u_delayed_local(s)` (the previous
piece at the same local `s`), and the local independent variable `s =
PowerSeries::variable(order)`, and returns `f(...)` as the series `u'(s)`.

| Requirement | Contract |
| :--- | :--- |
| Order preservation | All three arguments share the working `order`, and the returned series **must** have `order() == order`; a mismatch is reported as `MathError::domain_error` from inside the per-interval solve. |
| Fallibility | Returns a `Result<PowerSeries>`; any error it raises (e.g. an overflow from `multiply`) is propagated. |

Examples: `u' = -u(t-tau)` is `(u, ud, s) -> ud.scale(-1)`; `u' = u(t-tau)` is
`(u, ud, s) -> ud`; `u' = s·u(t-tau)` is `(u, ud, s) -> s.multiply(ud)`.

## `DdeSolution` — the piecewise solution

```cpp
struct DdeSolution {
    std::vector<PowerSeries> pieces;
    Rational tau;
};
```

| Member | Type | Meaning |
| :--- | :--- | :--- |
| `pieces` | `std::vector<PowerSeries>` | `pieces[k]` is the solution on `[k·tau, (k+1)·tau]`, expanded in the local variable `s = t - k·tau ∈ [0, tau]`. |
| `tau` | `Rational` | The positive delay shared by every interval. |

## Free functions

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `evaluate_series` | `[[nodiscard]] auto evaluate_series(const PowerSeries& series, const Rational& s) -> Result<Rational>` | Exact truncated value `Σ_i c_i·s^i` over the retained coefficients, by Horner over Q. Never fails on the series itself — only a `Rational` overflow along the way can fail it. |
| `solve_method_of_steps` | `[[nodiscard]] auto solve_method_of_steps(DdeOperator f, PowerSeries history_on_first_interval, Rational tau, std::size_t num_intervals, std::size_t order) -> Result<DdeSolution>` | Solve on `[0, num_intervals·tau]`, returning one order-`order` `PowerSeries` per interval in local coordinates. See below. |
| `evaluate` | `[[nodiscard]] auto evaluate(const DdeSolution& sol, const Rational& t) -> Result<Rational>` | Evaluate the piecewise solution at a rational time `t`. See below. |

### `solve_method_of_steps`

`history_on_first_interval` is the history **already expressed in interval 0's
local frame**: `history_local(s) = history(s - tau)` for `s ∈ [0, tau]`. It is
normalised to the working `order` and plays the role of "piece −1": it seeds both
the interval-0 delayed term and the interval-0 initial value `u(0) =
history_local(tau) = history(0)`. The same rule — *initial value = previous piece
evaluated at `s = tau`* — then threads continuity through every interval
uniformly.

`MathError::domain_error` if `f` is empty, `order == 0`, `num_intervals == 0`, or
`tau ≤ 0` (guarded by `tau.numerator() <= 0`). Any `powerseries` or `Rational`
error raised while solving — including an operator that fails to preserve the
working order — is propagated.

### `evaluate`

Locate the interval `k = floor(t / tau)` (computed exactly over Q, with correct
floor division for the sign) and return `pieces[k]` at the local `s = t - k·tau`.
The right endpoint `t = num_intervals·tau` is admissible and is served by the
**last** piece at `s = tau`. `t < 0`, a `t` strictly beyond the solved range, and
an empty solution (or `tau ≤ 0`) are all `MathError::domain_error`.

## Error model

| Condition | Error |
| :--- | :--- |
| `solve_method_of_steps` with empty `f`, `order == 0`, `num_intervals == 0`, or `tau ≤ 0` | `MathError::domain_error` |
| An operator returns a series whose `order()` differs from the working `order` | `MathError::domain_error` |
| `evaluate` on an empty solution, or with `tau ≤ 0` | `MathError::domain_error` |
| `evaluate` with `t < 0` (before the range) or `t` strictly beyond `num_intervals·tau` | `MathError::domain_error` |
| Any `int64` numerator/denominator computation in `Rational` Horner / floor / shift wraps | `MathError::overflow` (propagated) |
| Any `powerseries` operation (`from_coeffs`, `constant`, `integrate`, or the operator body) fails | that operation's error, propagated verbatim |

`DdeSolution` is a plain aggregate — constructing it never fails. Everything
fallible flows through `Result`.

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.powerseries;
import nimblecas.dde;
using namespace nimblecas;

auto ri  = [](std::int64_t v) { return Rational::from_int(v); };
auto rat = [](std::int64_t n, std::int64_t d) { return *Rational::make(n, d); };

// Constant history u = 1 on [-tau, 0], expressed in interval 0's local frame.
const std::size_t order = 6;
auto hist = *PowerSeries::constant(ri(1), order);

// --- u'(t) = u(t - 1), history 1, tau = 1, two intervals -----------------
// I0: delayed = 1, u(0) = 1 -> u' = 1        -> u = 1 + s.
// I1: delayed = 1 + s, u(0) = (1+s)|_{s=1}=2 -> u' = 1 + s -> u = 2 + s + s^2/2.
auto f_delayed = [](const PowerSeries&, const PowerSeries& ud,
                    const PowerSeries&) -> Result<PowerSeries> { return ud; };
auto sol = solve_method_of_steps(f_delayed, hist, ri(1), 2, order).value();
sol.pieces[0].coefficient(0);            // 1     (I0 = 1 + s)
sol.pieces[1].coefficient(2);            // 1/2   (I1 = 2 + s + s^2/2)
evaluate(sol, rat(3, 2)).value();        // 21/8  (I1 at s = 1/2)

// --- u'(t) = -u(t - 1): endpoints and out-of-range -----------------------
auto f_neg = [](const PowerSeries&, const PowerSeries& ud, const PowerSeries&) {
    return ud.scale(Rational::from_int(-1));
};
auto neg = solve_method_of_steps(f_neg, hist, ri(1), 2, order).value();
evaluate(neg, ri(0)).value();            // 1     (left endpoint, history(0))
evaluate(neg, ri(1)).value();            // 0     (shared knot, continuity)
evaluate(neg, ri(2)).value();            // -1/2  (right endpoint, I1 at s = 1)
evaluate(neg, ri(-1)).error();           // MathError::domain_error (before range)
evaluate(neg, ri(3)).error();            // MathError::domain_error (beyond range)

// --- Nonlinear delayed operator u'(t) = u(t - 1)^2 -----------------------
// Exercises the per-interval Picard recursion; I1 = 2 + s + s^2 + s^3/3.
auto f_sq = [](const PowerSeries&, const PowerSeries& ud, const PowerSeries&) {
    return ud.multiply(ud);
};
auto sq = solve_method_of_steps(f_sq, hist, ri(1), 2, order).value();
evaluate(sq, ri(2)).value();             // 13/3  (I1 at s = 1)

// --- Operator that reads the current state needs full Picard iteration ---
// u'(t) = u(t) + u(t - 1), history 1: on I0, u = 2 e^s - 1 truncated to order 6.
auto f_cur = [](const PowerSeries& u, const PowerSeries& ud, const PowerSeries&) {
    return u.add(ud);
};
auto cur = solve_method_of_steps(f_cur, hist, ri(1), 1, order).value();
cur.pieces[0].coefficient(3);            // 1/3   ([1, 2, 1, 1/3, 1/12, 1/60])

// --- Non-unit delay: re-centering uses local s in [0, tau], not t --------
auto tau2 = solve_method_of_steps(f_neg, *PowerSeries::constant(ri(1), 5),
                                  ri(2), 2, 5).value();
evaluate(tau2, ri(3)).value();           // -3/2  (I1 = -1 - s + s^2/2 at s = 1)
tau2.tau;                                 // 2     (solution retains tau)

// --- Degenerate arguments are domain errors ------------------------------
solve_method_of_steps(f_delayed, hist, ri(0),  2, order).error();  // domain_error (tau = 0)
solve_method_of_steps(f_delayed, hist, ri(-1), 2, order).error();  // domain_error (tau < 0)
solve_method_of_steps(f_delayed, hist, ri(1),  2, 0).error();      // domain_error (order 0)
solve_method_of_steps(f_delayed, hist, ri(1),  0, order).error();  // domain_error (0 intervals)
evaluate(DdeSolution{.pieces = {}, .tau = ri(1)}, ri(0)).error();  // domain_error (empty)
```

## See also

- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field every
  coefficient, delay, and evaluation point lives in.
- [`nimblecas.series`](series.md) and [`nimblecas.recurrence`](recurrence.md) —
  sibling exact-series and difference-equation solvers.
- [`nimblecas.laplace`](laplace.md) — the transform-based route to linear
  constant-coefficient (delay-free) ODEs.
- [Documentation hub](../Index.md)
