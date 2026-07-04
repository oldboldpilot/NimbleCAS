# `nimblecas.perturbation` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/perturbation/perturbation.cppm`

Semi-analytical perturbation solvers for the first-order autonomous initial
value problem `u'(x) = f(u)`, `u(0) = u0` (ROADMAP §7.5): the **Adomian
Decomposition Method** (ADM), the **Homotopy Perturbation Method** (HPM), and the
**Homotopy Analysis Method** (HAM, with convergence-control parameter `ħ`). This
layer sits directly on the `powerseries` substrate: `x` is
`PowerSeries::variable()`, the linear operator `L = d/dx` is
`PowerSeries::derivative()`, and its exact right inverse `L⁻¹` (definite integral
from `0`, zero constant of integration) is `PowerSeries::integrate()`. Rewriting
the IVP as the fixed point `u = u0 + L⁻¹[f(u)]` turns the whole solution into
**exact** truncated power-series arithmetic in `Q[[x]]/(x^N)`.

The honest boundary: everything here is exact over `Q` **as a power series
truncated to the requested order**. For polynomial/rational `f` those are the
exact Taylor coefficients of the solution to that order; through the exact
`exp`/`log`/`compose` coefficient recurrences of the `powerseries` engine they are
also exact within the truncation for the Maclaurin expansion of analytic `f`. The
methods make **no** claim about the radius of convergence, and no claim of a
closed-form or transcendental solution — only truncated rational coefficients.
`ħ ≠ −1` selects a different (still exactly rational) member of the HAM
deformation family rather than the Taylor series. Every `powerseries` `Result`
error is propagated verbatim (Rule 32); bad order or arguments surface as
`MathError::domain_error`.

```cpp
import nimblecas.perturbation;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), and `powerseries` (the
exact truncated-series substrate `x`, `L`, `L⁻¹` act on).

## The nonlinear operator `f`

The nonlinearity `f` of `u' = f(u)` is supplied as a `SeriesOperator`: given the
current solution series `u` it returns `f(u)` as a series, on the railway.

```cpp
using SeriesOperator = std::function<Result<PowerSeries>(const PowerSeries&)>;
```

`f = identity` models `u' = u`; `f = u ↦ u²` models `u' = u²`; `f = u ↦ 1 + u²`
models the Riccati equation `u' = 1 + u²`. Because `f` may use any `powerseries`
operation (products, `inverse`/`divide`, `exp`/`log`/`compose`), polynomial,
rational and truncated-analytic nonlinearities are all in scope. `f` **must
preserve the ring order** — a returned series whose `order()` differs from the
working order is rejected as `MathError::domain_error`. Any error `f` itself
raises is propagated.

## Grading and the Adomian polynomials

Because `u0` is a constant and each Picard update integrates once, the components
`u_0, u_1, u_2, …` are **homogeneous**: `u_i` has its only nonzero term at `x^i`.
The decomposition grade `i` therefore coincides exactly with the `x`-degree `i`,
and the `m`-th Adomian polynomial `A_m` — the grade-`m` contribution to
`N(u_0 + u_1 + …)` — is precisely the `x`-degree-`m` graded projection of `N`
applied to the partial sum:

```
A_m = Π_m [ N(u_0 + u_1 + … + u_m) ]      (place [x^m] N(·) back at x^m).
```

This is the projection of `N` of the **total** partial sum, not the naive
difference `N(Σ_{i≤m}) − N(Σ_{i<m})`, which double-drops same-grade cross terms.
For `N(u) = u²` with components `u_0, u_1, u_2` the graded projection correctly
returns `A_2 = 2 u_0 u_2 + u_1²`; the raw difference would miss the `u_1²` piece.
Components of index `> m` are of degree `> m` and cannot affect `[x^m]`, so
projecting `N` of the full sum is both exact and cheapest. This is exact for
polynomial nonlinearities.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `adomian_polynomials` | `[[nodiscard]] auto adomian_polynomials(SeriesOperator N, const std::vector<PowerSeries>& components) -> Result<std::vector<PowerSeries>>` | The Adomian polynomials `A_0 … A_k` of the operator `N` for the components `u_0 … u_k` (`k = components.size() − 1`). `A_m` is returned as the homogeneous series `([x^m] N(Σ u_i)) x^m` — the `x`-degree-`m` graded projection of `N` of the partial sum. All components must share the same order; the returned polynomials share that order. |

Failure modes (`MathError::domain_error`): empty component list, an empty
operator `N`, components of differing order, or `N` not preserving the order. Any
error raised by `N` is propagated.

## The solvers

All three consume the operator `f`, the exact rational initial value `u0`, and an
`order`; they return the solution as a `PowerSeries` with `order` coefficients
(terms `x^0 … x^{order−1}`). Internally each maintains the partial sum
`P = u_0 + … + u_k` **incrementally** and extracts only the single new
coefficient it needs per step — `O(n)` bookkeeping and exactly one evaluation of
`f` per order (the irreducible cost), rather than the `O(n²)` re-summation a naive
loop over `adomian_polynomials` would incur.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `adm_solve` | `[[nodiscard]] auto adm_solve(SeriesOperator f, Rational u0, std::size_t order) -> Result<PowerSeries>` | Adomian Decomposition. `u_0 = u0`, `u_{n+1} = L⁻¹[A_n]`; the result is `Σ u_n`, the exact Taylor polynomial of the solution to `order`. |
| `hpm_solve` | `[[nodiscard]] auto hpm_solve(SeriesOperator f, Rational u0, std::size_t order) -> Result<PowerSeries>` | Homotopy Perturbation. The homotopy `(1−p)(v'−u0') + p(v'−f(v)) = 0` with `v = Σ pⁿ v_n` collapses to `v_0 = u0`, `v_n' = A_{n−1}` — the **same** graded recursion as `adm_solve`, so it returns the identical series. |
| `ham_solve` | `[[nodiscard]] auto ham_solve(SeriesOperator f, Rational u0, Rational hbar, std::size_t order) -> Result<PowerSeries>` | Homotopy Analysis, with convergence-control parameter `ħ`. The `m`-th order deformation is `u_m = χ_m u_{m−1} + ħ · L⁻¹[R_m]`, `R_m = u_{m−1}' − A_{m−1}`, `χ_m = 0` for `m = 1` else `1`; the result is `Σ u_m`. `ħ` is the extra degree of freedom HAM adds over HPM/ADM. |

### Equivalences

ADM and HPM are two organisations of the same graded Picard/Taylor iteration and
return the **identical** series. HAM adds `ħ`: at `ħ = −1` the `u_{m−1}'` term
cancels `u_{m−1}` (the components are homogeneous of degree `≥ 1`, so
`L⁻¹[u_{m−1}'] = u_{m−1}`) and `u_m` reduces to `L⁻¹[A_{m−1}]`, so **`ham_solve`
with `ħ = −1` recovers ADM/HPM exactly**. Any other `ħ` yields a different but
still exactly rational series — a distinct member of the HAM deformation family,
not the Taylor series.

## Error model

| Condition | Error |
| :--- | :--- |
| `order == 0` (`adm_solve` / `hpm_solve` / `ham_solve`) | `MathError::domain_error` |
| Empty operator (`!f` / `!N`) | `MathError::domain_error` |
| `adomian_polynomials` with an empty component list | `MathError::domain_error` |
| `adomian_polynomials` components of differing order | `MathError::domain_error` |
| `f`/`N` returns a series whose `order()` differs from the working order | `MathError::domain_error` |
| Any error raised by `f`/`N` (e.g. an overflow from `multiply`) | propagated verbatim |
| Any `powerseries` operation (`zero`, `constant`, `from_coeffs`, `add`, `subtract`, `scale`, `derivative`, `integrate`) fails | that operation's error, propagated |

An `int64` coefficient computation that would wrap surfaces as
`MathError::overflow` through the underlying `Rational`/`powerseries` arithmetic.

## Worked examples

```cpp
import nimblecas.perturbation;
import nimblecas.powerseries;
import nimblecas.ratpoly;
using namespace nimblecas;

auto ri  = [](std::int64_t v) { return Rational::from_int(v); };
auto rat = [](std::int64_t n, std::int64_t d) { return *Rational::make(n, d); };

// The IVP operators f such that u' = f(u).
SeriesOperator f_identity = [](const PowerSeries& u) -> Result<PowerSeries> {
    return u;                                   // u' = u
};
SeriesOperator f_square = [](const PowerSeries& u) {
    return u.multiply(u);                       // u' = u^2
};
SeriesOperator f_riccati = [](const PowerSeries& u) -> Result<PowerSeries> {
    auto one = PowerSeries::one(u.order());     // u' = 1 + u^2
    if (!one) return make_error<PowerSeries>(one.error());
    auto sq = u.multiply(u);
    if (!sq)  return make_error<PowerSeries>(sq.error());
    return one->add(*sq);
};

// u' = u, u(0) = 1  ->  e^x = Σ x^n/n!.
auto exp_series = adm_solve(f_identity, ri(1), 8).value();
exp_series.coefficient(0);   // 1
exp_series.coefficient(3);   // 1/6

// u' = u^2, u(0) = 1  ->  1/(1-x) = Σ x^n (every coefficient 1).
auto geom = adm_solve(f_square, ri(1), 8).value();
geom.coefficient(5);         // 1

// u' = 1 + u^2, u(0) = 0  ->  tan(x): 0, 1, 0, 1/3, 0, 2/15, 0, 17/315.
auto tan_series = adm_solve(f_riccati, ri(0), 8).value();
tan_series.coefficient(7);   // 17/315

// ADM == HPM == HAM(hbar = -1), coefficient for coefficient.
auto a = adm_solve(f_square, ri(1), 8).value();
auto h = hpm_solve(f_square, ri(1), 8).value();
auto m = ham_solve(f_square, ri(1), ri(-1), 8).value();
a.is_equal(h);               // true
a.is_equal(m);               // true

// hbar is a genuine free parameter: -1 recovers ADM, -2 gives a different series.
auto ham_m1 = ham_solve(f_identity, ri(1), ri(-1), 5).value();
auto ham_m2 = ham_solve(f_identity, ri(1), ri(-2), 5).value();
adm_solve(f_identity, ri(1), 5).value().is_equal(ham_m1);  // true
adm_solve(f_identity, ri(1), 5).value().is_equal(ham_m2);  // false

// Adomian polynomials of N(u) = u^2 for homogeneous components
// u_0 = 3, u_1 = 5x, u_2 = 7x^2:
//   A_0 = u_0^2          = 9
//   A_1 = 2 u_0 u_1      = 30x
//   A_2 = 2 u_0 u_2 + u_1^2 = 67x^2
const std::size_t order = 6;
auto u0 = PowerSeries::from_coeffs({ri(3)},               order).value();
auto u1 = PowerSeries::from_coeffs({ri(0), ri(5)},        order).value();
auto u2 = PowerSeries::from_coeffs({ri(0), ri(0), ri(7)}, order).value();
auto polys = adomian_polynomials(f_square, {u0, u1, u2}).value();
polys[0].coefficient(0);     // 9
polys[1].coefficient(1);     // 30
polys[2].coefficient(2);     // 67

// Degenerate arguments are domain errors.
adm_solve(f_identity, ri(1), 0).error();          // MathError::domain_error
adomian_polynomials(f_square, {}).error();        // MathError::domain_error
ham_solve(f_identity, ri(1), ri(-1), 0).error();  // MathError::domain_error
```

## See also

- `nimblecas.powerseries` — the exact truncated-series substrate supplying `x`,
  `L = derivative`, `L⁻¹ = integrate`, and the coefficient recurrences these
  solvers are built on.
- [`nimblecas.dde`](dde.md) — the sibling `powerseries`-consuming solver, applying
  the same `u = u(0) + L⁻¹[f(…)]` fixed point to delay differential equations.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field the series
  coefficients live in.
- [Documentation hub](../Index.md)
