# `nimblecas.ode` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/ode/ode.cppm`

Exact **power-series** initial-value problem solver for autonomous **first-order
systems** and, by companion-system reduction, scalar **higher-order** ODEs. It
generalises the scalar perturbation solver (`nimblecas.perturbation`) from a
single equation to a vector field `u_i' = f_i(u_1, …, u_n)`, working entirely as
exact arithmetic over **Q** in the truncated ring `Q[[x]]/(x^N)`. The engine is
`nimblecas.powerseries`: `x` is the independent variable, the linear operator
`L = d/dx` is `PowerSeries::derivative()`, and its exact right inverse `L⁻¹`
(definite integral from `0` with zero constant of integration) is
`PowerSeries::integrate()`. Rewriting the IVP as the vector fixed point
`u = u0 + L⁻¹[f(u)]` turns the whole solution into truncated power-series
arithmetic, solved by a graded **Picard/Taylor** recursion that builds all `n`
components simultaneously.

**Honesty boundary.** Everything here is **exact over Q as a power series
truncated** to the requested `order` (terms `x^0 … x^{order-1}`). For
right-hand sides built from the available `powerseries` operations
(polynomial/rational via `multiply` and `inverse`/`divide`, and the analytic
`exp`/`log`/`compose` coefficient recurrences) these are the **exact Taylor
coefficients** of the solution to that order — nothing beyond the truncation is
retained, an implicit `O(x^order)` tail. The solver makes **no claim** about the
radius of convergence or about any closed-form solution; it mirrors the exactness
discipline of `nimblecas.perturbation`, and **no floating point is used
anywhere**. Rule 32 railway: every `powerseries`/`Rational` `Result` error is
propagated; a bad `order`, bad vector sizes, or an operator that fails to
preserve the order are `MathError::domain_error`.

```cpp
import nimblecas.ode;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), and `powerseries` (the
exact graded power-series arithmetic that carries every solution component).

## `SystemOperator` — the autonomous vector field

```cpp
using SystemOperator =
    std::function<Result<std::vector<PowerSeries>>(const std::vector<PowerSeries>&)>;
```

The vector field `f` of the system `u_i' = f_i(u)`: it consumes the current
vector of solution series `[u_1, …, u_n]` and returns `[f_1(u), …, f_n(u)]` as
series, on the railway.

| Requirement | Contract |
| :--- | :--- |
| Component count | The returned vector **must** have length `n = u0.size()`; any other length is `MathError::domain_error`. |
| Order preservation | Every returned series **must** have `order() == order`; a mismatch is `MathError::domain_error`. |
| Fallibility | Returns a `Result<std::vector<PowerSeries>>`; any error it raises (e.g. an overflow from `multiply`) is propagated verbatim. |

Examples: the harmonic system `u' = v, v' = -u` is
`u -> {u[1], u[0].scale(-1)}`; the coupled linear system `u' = u+v, v' = u-v` is
`u -> {u[0].add(u[1]), u[0].subtract(u[1])}`; the scalar `u' = u` is
`u -> {u[0]}`.

## `HigherOrderOperator` — the scalar higher-order right-hand side

```cpp
using HigherOrderOperator =
    std::function<Result<PowerSeries>(const std::vector<PowerSeries>&)>;
```

The right-hand side `f` of a scalar `k`-th order IVP
`u^{(k)} = f(u, u', …, u^{(k-1)})`: it consumes the current derivative vector
`[y_0, …, y_{k-1}] = [u, u', …, u^{(k-1)}]` and returns `u^{(k)}` as a single
series, on the railway. It is invoked inside the companion first-order system, so
the same order-preservation discipline applies to its result.

Examples: `u'' = -u` is `y -> y[0].scale(-1)`; `u'' = u` is `y -> y[0]`;
`u''' = -u'` is `y -> y[1].scale(-1)`.

## Free functions

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `solve_first_order_system` | `[[nodiscard]] auto solve_first_order_system(SystemOperator f, std::vector<Rational> u0, std::size_t order) -> Result<std::vector<PowerSeries>>` | Solve the autonomous system `u_i'(x) = f_i(u)`, `u_i(0) = u0[i]`, as exact truncated series with `order` coefficients. Returns the `n = u0.size()` solution series in the same order as `u0`. See below. |
| `solve_higher_order` | `[[nodiscard]] auto solve_higher_order(HigherOrderOperator f, std::vector<Rational> initial, std::size_t order) -> Result<PowerSeries>` | Solve the scalar `k`-th order IVP `u^{(k)} = f(u, …, u^{(k-1)})` with `initial = [u(0), …, u^{(k-1)}(0)]` (so `k = initial.size()`), returning the component `u`. See below. |
| `evaluate` | `[[nodiscard]] auto evaluate(const PowerSeries& s, const Rational& x) -> Result<Rational>` | Exact Horner value of the **truncated** series `c_0 + c_1·x + … + c_{N-1}·x^{N-1}` at the rational point `x`. See below. |

### `solve_first_order_system`

The method is the graded Picard/Taylor recursion `u <- u0 + L⁻¹[f(u)]` started
from the constant vector `u^(0) = u0` and run `order` passes. Because `L⁻¹`
(`integrate`) raises the `x`-degree by one, **one pass fixes one further Taylor
coefficient of every component**: if `u^(m)` agrees with the true solution
through degree `m`, then `f(u^(m))` agrees through `m` (every `powerseries`
operation preserves the agreement degree of its inputs) and integrating lifts the
agreement to `m+1`. Hence `order` passes fix all coefficients `x^0 … x^{order-1}`;
the recursion has reached the fixed point of the truncated system and further
passes leave the truncation unchanged. The result is the **exact Taylor
polynomial** of the solution to the requested order.

`MathError::domain_error` if `order == 0`, `u0` is empty, `f` is an empty
callable, `f` returns a vector whose length differs from `n`, or `f` returns a
series whose `order()` differs from `order` (i.e. `f` fails to preserve the
order). Any error raised by `f` or by the `powerseries` engine (e.g. from
`constant` / `integrate` / `add`) is propagated.

### `solve_higher_order`

The scalar `k`-th order IVP is reduced to the companion first-order system
`y_0 = u, y_1 = u', …, y_{k-1} = u^{(k-1)}` with `y_j' = y_{j+1}` for `j < k-1`
and `y_{k-1}' = f(y_0, …, y_{k-1})`, then solved by `solve_first_order_system`
with `y_j(0) = initial[j]`; the returned component is `y_0 = u`. `k` is the length
of `initial`.

`MathError::domain_error` if `order == 0`, `initial` is empty (`k` must be `≥ 1`),
or `f` is an empty callable. Errors from `f` or the engine are propagated. Note
the companion operator also raises `domain_error` if it is ever handed a state
vector whose length is not `k`.

### `evaluate`

Exact Horner evaluation from the top coefficient down (`acc <- acc·x + c_k`),
computed entirely over `Q`. This returns the value of the **truncated polynomial**
`c_0 + c_1·x + … + c_{N-1}·x^{N-1}`, where `N = s.order()` — **not** the exact ODE
solution unless the series terminates. It is provided for checking and sampling.
Evaluating at `x = 0` returns the constant term `c_0`. Any overflow in the
rational arithmetic surfaces as `MathError::overflow`, propagated.

## Error model

| Condition | Error |
| :--- | :--- |
| `solve_first_order_system` with `order == 0`, empty `u0`, or an empty `f` | `MathError::domain_error` |
| `f` returns a vector whose length differs from `n = u0.size()` | `MathError::domain_error` |
| `f` returns a series whose `order()` differs from the working `order` | `MathError::domain_error` |
| `solve_higher_order` with `order == 0`, empty `initial` (`k == 0`), or an empty `f` | `MathError::domain_error` |
| Any `int64` numerator/denominator computation in `Rational` (e.g. in `evaluate`'s Horner loop) wraps | `MathError::overflow` (propagated) |
| Any `powerseries` operation (`constant`, `integrate`, `add`, or the operator body) fails | that operation's error, propagated verbatim |

The operators are plain `std::function`s and the solvers thread every fallible
step through `Result`; nothing throws.

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.powerseries;
import nimblecas.ode;
using namespace nimblecas;

auto ri  = [](std::int64_t v) { return Rational::from_int(v); };
auto rat = [](std::int64_t n, std::int64_t d) { return *Rational::make(n, d); };

// --- Harmonic system: u' = v, v' = -u, u(0)=0, v(0)=1 -> sin, cos ----------
auto f_harmonic = [](const std::vector<PowerSeries>& u)
    -> Result<std::vector<PowerSeries>> {
    auto neg_u = u[0].scale(Rational::from_int(-1));
    if (!neg_u) {
        return make_error<std::vector<PowerSeries>>(neg_u.error());
    }
    return std::vector<PowerSeries>{u[1], *neg_u};
};
auto hs = solve_first_order_system(f_harmonic, {ri(0), ri(1)}, 8).value();
hs[0].coefficient(1);                 // 1      (sin: 0 + x - x^3/6 + x^5/120 - x^7/5040)
hs[0].coefficient(3);                 // -1/6
hs[1].coefficient(0);                 // 1      (cos: 1 - x^2/2 + x^4/24 - x^6/720)
hs[1].coefficient(2);                 // -1/2

// --- Coupled linear system: u' = u+v, v' = u-v, u(0)=1, v(0)=0 -------------
// A = [[1,1],[1,-1]] with A^2 = 2I, so u = cosh(sqrt2 x) + sinh(sqrt2 x)/sqrt2.
auto f_coupled = [](const std::vector<PowerSeries>& u)
    -> Result<std::vector<PowerSeries>> {
    auto up = u[0].add(u[1]);         // u' = u + v
    auto vp = u[0].subtract(u[1]);    // v' = u - v
    if (!up || !vp) {
        return make_error<std::vector<PowerSeries>>(up ? vp.error() : up.error());
    }
    return std::vector<PowerSeries>{*up, *vp};
};
auto cs = solve_first_order_system(f_coupled, {ri(1), ri(0)}, 6).value();
cs[0].coefficient(3);                 // 1/3    (u = [1, 1, 1, 1/3, 1/6, 1/30])
cs[1].coefficient(5);                 // 1/30   (v = [0, 1, 0, 1/3, 0, 1/30])

// --- Second-order u'' = -u, u(0)=0, u'(0)=1 -> sin(x) ----------------------
auto f_second_neg = [](const std::vector<PowerSeries>& y) {
    return y[0].scale(Rational::from_int(-1));
};
auto s2 = solve_higher_order(f_second_neg, {ri(0), ri(1)}, 8).value();
s2.coefficient(5);                    // 1/120  (sin series)

// --- Second-order u'' = u, u(0)=1, u'(0)=1 -> e^x --------------------------
auto f_second_pos = [](const std::vector<PowerSeries>& y) -> Result<PowerSeries> {
    return y[0];
};
auto ex = solve_higher_order(f_second_pos, {ri(1), ri(1)}, 8).value();
ex.coefficient(4);                    // 1/24   (exp: 1, 1, 1/2, 1/6, 1/24, ...)

// --- Genuine k = 3 reduction: u''' = -u', u(0)=1, u'(0)=0, u''(0)=-1 -> cos -
auto f_third = [](const std::vector<PowerSeries>& y) -> Result<PowerSeries> {
    return y[1].scale(Rational::from_int(-1));   // u''' = -u'
};
auto c3 = solve_higher_order(f_third, {ri(1), ri(0), ri(-1)}, 8).value();
c3.coefficient(2);                    // -1/2   (cos series)

// --- Exact Horner evaluation of a truncated series ------------------------
auto poly = PowerSeries::from_coeffs({ri(1), ri(1), ri(1)}, 3).value();  // 1 + x + x^2
evaluate(poly, ri(2)).value();        // 7      (1 + 2 + 4)
evaluate(poly, rat(1, 2)).value();    // 7/4
evaluate(ex, ri(0)).value();          // 1      (constant term of the exp series)

// --- Degenerate arguments are domain errors -------------------------------
solve_first_order_system(f_harmonic, {ri(0), ri(1)}, 0).error();   // domain_error (order 0)
solve_first_order_system(f_harmonic, {}, 8).error();               // domain_error (empty u0)
auto wrong_size = [](const std::vector<PowerSeries>& u)
    -> Result<std::vector<PowerSeries>> {
    return std::vector<PowerSeries>{u[0]};      // returns 1, expected 2
};
solve_first_order_system(wrong_size, {ri(0), ri(1)}, 8).error();   // domain_error (size mismatch)
solve_higher_order(f_second_pos, {ri(1), ri(1)}, 0).error();       // domain_error (order 0)
solve_higher_order(f_second_pos, {}, 8).error();                   // domain_error (empty initial)
```

## See also

- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field every initial
  value, coefficient, and evaluation point lives in.
- [`nimblecas.dde`](dde.md) — the exact power-series solver for **delay**
  differential equations by the method of steps, built on the same graded Picard
  recursion.
- [`nimblecas.series`](series.md) and [`nimblecas.recurrence`](recurrence.md) —
  sibling exact-series and difference-equation solvers.
- [`nimblecas.laplace`](laplace.md) — the transform-based route to linear
  constant-coefficient ODEs.
- [Documentation hub](../Index.md)
