# `nimblecas.daenl` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/daenl/daenl.cppm`

`nimblecas.daenl` is the **nonlinear / higher-index** companion to
[`nimblecas.dae`](dae.md) (which is scoped to *linear, constant-coefficient*
DAEs). It handles nonlinear differential-algebraic systems `F(t, x, x') = 0` in
**three honesty tiers of decreasing exactness**, each stating its boundary
explicitly and returning an honest `MathError` rather than a plausible-wrong
value (Code Policy Rule 32) wherever it cannot deliver.

```cpp
import nimblecas.daenl;
```

Depends on [`core`](core.md), [`symbolic`](symbolic.md),
[`simplify`](simplify.md), [`diff`](diff.md), [`evalnum`](evalnum.md),
[`nlsolve`](nlsolve.md), [`ratpoly`](ratpoly.md), [`matrix`](matrix.md),
[`powerseries`](powerseries.md), [`ode`](ode.md) and [`dae`](dae.md) — the
existing exact-series, symbolic-differentiation, and Newton infrastructure is
reused directly; nothing is re-derived.

## Honesty boundary at a glance

| Tier | Entry point | Exactness | Refuses with |
| :--- | :--- | :--- | :--- |
| **1** | `solve_semiexplicit_nonlinear_series` | **exact** truncated power series over `Q` | `not_implemented` / `domain_error` |
| **2** | `analyze_structure` | **exact** symbolic structural analysis | `not_implemented` |
| **3** | `consistent_initial_values`, `solve_nonlinear_dae` | **numerical** (Newton / implicit Euler / BDF-2 to a stated tolerance) | `not_converged` / `not_implemented` / `domain_error` |

## Tier 1 — exact power series for the semi-explicit form

`solve_semiexplicit_nonlinear_series(f, g, xvars, yvars, time, x0, y0, order)`
solves the semi-explicit nonlinear DAE

```
x' = f(x, y, t),     0 = g(x, y, t),     x(0) = x0,  y(0) = y0
```

as an **exact truncated power series over `Q`** (terms `t^0 … t^{order-1}`),
returning the shared [`DaeSolution`](dae.md) `{ x, y }` of `PowerSeries`
components.

**Honestly stated restriction (not silently assumed).** The reduction it
performs is

```
y' = -A^{-1} ( (dg/dx) · f + dg/dt ),     A = dg/dy,
```

which is exact **only when `A = dg/dy` is a *constant* rational matrix** — i.e.
`g` is affine in `y` with `x`/`t`-independent coefficients (`g = A y + h(x, t)`).
The differential and algebraic right-hand sides may be built only from `+`, `*`,
integer powers (positive or negative), `exp`, and `ln`. Anything outside that
grammar — a non-constant `dg/dy`, a fractional or symbolic exponent, a
`sin`/`cos`/… call, or a floating-point literal — is rejected with
`MathError::not_implemented`, never approximated. The `x`/`t`-dependence of `g`
(via `dg/dx`) is unrestricted.

Under those conditions the reduced system `x' = f`, `y' = -A^{-1}(dg/dx·f + dg/dt)`
is an autonomous first-order system fed to [`ode`](ode.md)'s
`solve_first_order_system`, and the result is **verified**: `g` is re-evaluated
on the computed series and every coefficient must vanish to the truncation order
(else `domain_error`).

```cpp
import nimblecas.daenl;
import nimblecas.symbolic;
import nimblecas.ratpoly;
using namespace nimblecas;

// x' = y, 0 = y - x,  x(0) = y(0) = 1   =>   x = y = e^t.
const std::vector<Expr> f{Expr::symbol("y")};
const std::vector<Expr> g{Expr::sum({Expr::symbol("y"),
                                     Expr::product({Expr::integer(-1), Expr::symbol("x")})})};
auto sol = daenl::solve_semiexplicit_nonlinear_series(
    f, g, {"x"}, {"y"}, "t", {Rational::from_int(1)}, {Rational::from_int(1)}, 6);
// sol->x[0] and sol->y[0] are both  1 + t + t^2/2 + t^3/6 + t^4/24 + t^5/120.
```

## Tier 2 — structural (Pantelides-style) index analysis

`analyze_structure(sys, max_index = 3)` takes a `DaeSystem`

```cpp
struct DaeSystem {
    std::vector<Expr> residuals;   // F_i(t, x, x') = 0
    std::vector<std::string> vars; // the state symbols x_j
    std::vector<std::string> ders; // ders[j] is the symbol for d(vars[j])/dt
    std::string time{"t"};
};
```

and performs a structural index/consistency sweep: it forms the bipartite
incidence between equations and derivative symbols, tests it for a **perfect
matching** (Kuhn's augmenting-path algorithm), and, while none exists,
**symbolically differentiates** the whole most-recent equation batch by the
multivariable chain rule (via [`diff`](diff.md), treating each level-`L` symbol's
time-derivative as the level-`(L+1)` symbol) — growing equations and derivative
columns in lockstep — until a matching is found or `max_index` sweeps are
exhausted.

```cpp
struct StructuralInfo {
    std::size_t structural_index;               // differentiation sweeps + 1
    std::vector<std::size_t> diff_count;         // per-equation differentiation count
    std::vector<Expr> augmented_residuals;       // original + differentiated equations
};
```

**Scope, stated honestly.** This is a *simplified, non-minimal* variant of
Pantelides: it differentiates the entire batch each sweep rather than isolating
the minimal structurally-singular subset, so it can **over-differentiate**
relative to the textbook algorithm, and because each sweep introduces a
top-level derivative column that no equation yet references, systems that need
differentiation to expose a constraint frequently exhaust `max_index` and return
`MathError::not_implemented`. It **never reports an index it has not certified by
an explicit matching** — a pure ODE (`x' = y`, `y' = x`) matches at sweep 0 and
reports `structural_index == 1`; a constraint system it cannot match within
`max_index` is an honest `not_implemented`, not a fabricated structure. Every
differentiation it does report is exact symbolic differentiation.

## Tier 3 — numerical index-1 stepping

Both Tier-3 entry points are **numerical** (stated tolerance, not exact) and
built on [`nlsolve`](nlsolve.md)'s Newton solver with an analytic Jacobian
assembled once by [`diff`](diff.md) and evaluated through
[`evalnum`](evalnum.md).

`consistent_initial_values(sys, x_guess, xdot_guess, t0, opts)` holds `x` fixed
at the guess and solves the index-reduced augmented system (from
`analyze_structure`) for a consistent derivative via Newton, returning
`{ x_guess, corrected xdot }`. Newton non-convergence is `not_converged`.

`solve_nonlinear_dae(sys, x0, xdot0, t0, t1, steps, opts)` time-steps the
**original** residuals by **implicit Euler** (`opts.bdf_order == 1`) or **BDF-2**
(`opts.bdf_order == 2`, falling back to Euler on the first step), returning the
sampled trajectory:

```cpp
struct NlDaeSolution {
    std::vector<double> times;
    std::vector<std::vector<double>> states;   // states[k] = x at times[k]
    std::size_t structural_index;
    bool initial_guess_consistent;
};
```

**Honest limit.** Only **`structural_index == 1`** systems are integrated;
anything higher returns `MathError::not_implemented`, because reliably
stabilising a higher-index trajectory needs the projection / dummy-derivative
machinery this module does not implement. Every Newton non-convergence, and any
post-step constraint drift beyond `opts.constraint_tol`, is reported as
`not_converged` — never accepted as a state.

Because implicit Euler on `x' = -x` reduces exactly to `x_{k+1}(1 + h) = x_k`, a
single unit step (`h = 1`) from `x_0 = 1` gives **exactly** `x_1 = 1/2` — the
scheme has an exact algebraic oracle even though the method is numerical.

## API

| Function / type | Signature | Behavior |
| :--- | :--- | :--- |
| `DaeSystem` | `struct { vector<Expr> residuals; vector<string> vars; vector<string> ders; string time; }` | A nonlinear residual system `F(t, x, x') = 0`; `residuals`, `vars`, `ders` must have equal length; `ders[j]` names `d(vars[j])/dt`. |
| `solve_semiexplicit_nonlinear_series` | `(f, g, xvars, yvars, time, x0, y0, order) -> Result<DaeSolution>` | Tier 1. Exact power series when `dg/dy` is constant and `f`/`g` are in the `+ * ^ exp ln` grammar; verified `g == 0` to order. Out-of-grammar ⇒ `not_implemented`; shape/`order == 0` ⇒ `domain_error`. |
| `StructuralInfo` / `analyze_structure` | `(sys, max_index = 3) -> Result<StructuralInfo>` | Tier 2. Structural index + augmented equation set via chain-rule differentiation and bipartite matching. Unmatched within `max_index` ⇒ `not_implemented`; malformed `sys` ⇒ `domain_error`. |
| `NlDaeOptions` | `struct { size_t max_index; size_t bdf_order; double constraint_tol; nlsolve::Options newton; }` | Tier-3 knobs. |
| `consistent_initial_values` | `(sys, x_guess, xdot_guess, t0, opts = {}) -> Result<pair<vector<double>, vector<double>>>` | Tier 3. Newton-corrected consistent `{x, xdot}`. Non-convergence ⇒ `not_converged`; bad shapes / non-finite ⇒ `domain_error`. |
| `NlDaeSolution` / `solve_nonlinear_dae` | `(sys, x0, xdot0, t0, t1, steps, opts = {}) -> Result<NlDaeSolution>` | Tier 3. Index-1 implicit-Euler / BDF-2 trajectory. `structural_index != 1` ⇒ `not_implemented`; Newton failure or constraint drift ⇒ `not_converged`; `steps == 0`, `t1 <= t0`, bad `bdf_order`, or bad shapes ⇒ `domain_error`. |

## Error model

| Condition | Error |
| :--- | :--- |
| Tier 1: non-constant `dg/dy`, a symbol/double where an exact rational is required, or an out-of-grammar `f`/`g` (unsupported function, non-integer power) | `MathError::not_implemented` |
| Tier 1: `order == 0`, empty/size-mismatched `f`/`g`/`xvars`/`yvars`/`x0`/`y0` | `MathError::domain_error` |
| Tier 1: the computed series fails the `g == 0`-to-order verification | `MathError::domain_error` |
| Tier 2: no perfect matching within `max_index` sweeps | `MathError::not_implemented` |
| Tier 2: empty system, size mismatch, or `max_index == 0` | `MathError::domain_error` |
| Tier 3: `solve_nonlinear_dae` on a `structural_index != 1` system | `MathError::not_implemented` |
| Tier 3: Newton non-convergence, or post-step residual drift beyond `constraint_tol` | `MathError::not_converged` |
| Tier 3: `steps == 0`, `t1 <= t0`, `bdf_order ∉ {1,2}`, bad shapes, or non-finite input | `MathError::domain_error` |
| Any tier: `int64` numerator/denominator overflow in the exact arithmetic | `MathError::overflow` (propagated) |

## See also

- [`nimblecas.dae`](dae.md) — the **linear** constant-coefficient DAE engine
  (exact over `Q`) whose `DaeSolution` this module reuses.
- [`nimblecas.ode`](ode.md) — the exact first-order power-series system solver the
  Tier-1 reduction feeds.
- [`nimblecas.nlsolve`](nlsolve.md) — the Newton family driving Tier 3.
- [`nimblecas.diff`](diff.md) / [`nimblecas.evalnum`](evalnum.md) — symbolic
  differentiation for the structural sweep and Jacobians, and numeric `Expr`
  evaluation for the residuals.
- [Documentation hub](../Index.md)
