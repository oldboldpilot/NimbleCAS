# `nimblecas.ipm` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/ipm/ipm.cppm`

**Linear programming done numerically** by a primal–dual interior-point method
(ROADMAP §7.22). `solve_ipm` solves a standard-form linear program

```
minimize  c · x    subject to   A x = b,   x >= 0,
```

with a **Mehrotra predictor–corrector path-following** algorithm in
floating-point (`double`). It is the *numerical companion* to the exact-rational
Simplex of [`nimblecas.lp`](lp.md): the same §7.22 standard form, solved by a
different engine with a different honesty contract.

```cpp
import nimblecas.ipm;
```

Depends only on [`core`](core.md).

## Honesty boundary — numerical, not exact

This module is the deliberate opposite of [`lp.md`](lp.md) on precision:

| | `nimblecas.lp` (Simplex) | `nimblecas.ipm` (interior-point) |
| :--- | :--- | :--- |
| Arithmetic | exact `Rational` | `double` (floating point) |
| Nature | combinatorial pivoting to a vertex | iterative walk down the central path, `mu → 0` |
| Result | exact `p/q` | accurate to `tolerance` (`1e-9`) |
| Optimality proof | Bland's rule termination | small **duality gap** `mu` |
| Cost | exact, worst-case exponential | fixed small iteration count, ill-suited to exactness |

An interior-point method **cannot** return an exact rational: it approaches the
optimum asymptotically and stops when the primal residual `‖A x − b‖`, the dual
residual `‖Aᵀy + s − c‖`, and the duality measure `mu = (x · s)/n` are **all**
below `tolerance`. By weak duality `c·x − b·y = x·s = n·mu`, so a tiny `mu`
*certifies* that the reported objective is within `≈ tolerance` of optimal — that
gap certificate is the whole point of the method. The returned `objective`, `x`,
and `y` are therefore correct to about `tolerance` and **must not** be read as
exact. For an exact answer on a small problem, use [`nimblecas.lp`](lp.md)
instead.

Non-convergence is reported honestly (Rule 32): if the iteration cap is reached
without meeting all three criteria and the iterate is not diverging, `solve_ipm`
returns `MathError::not_implemented` — the closest honest variant, since
`MathError` has no dedicated `no_convergence` code — rather than a fabricated
"optimal". Infeasibility and unboundedness are **best-effort** classifications of
a diverging iterate (see below); when they cannot be distinguished honestly the
same `not_implemented` is returned.

## The method

- **State.** Primal `x > 0` (length `n`), free dual multipliers `y` (length `m`),
  dual slacks `s > 0` (length `n`). The central path enforces the perturbed KKT
  system `A x = b`, `Aᵀy + s = c`, `xᵢ sᵢ = mu`, `x, s ≥ 0`.
- **Newton via normal equations.** Each iteration reduces the Newton system to
  `(A D Aᵀ) Δy = r` with `D = diag(x / s)`, an `m × m` **symmetric positive
  definite** system, and recovers `Δs = r_d − Aᵀ Δy`, `Δx = −D Δs + r_xs / s`.
- **Inline Cholesky.** `A D Aᵀ` is factored by a dense `L Lᵀ` Cholesky **written
  inline in this module** — `nimblecas.matdecomp` offers only exact-`Rational` LU,
  so no external `double` Cholesky is assumed. A tiny `1e-12` diagonal
  regularization keeps the factorization robust as `D` stretches toward the
  boundary; a genuine non-positive pivot (numerically rank-deficient `A`) surfaces
  as `MathError::not_implemented`.
- **Mehrotra predictor–corrector.** An affine (predictor) step with
  complementarity target `0` yields step lengths `α_aff`; from them the centering
  parameter `σ = (mu_aff / mu)³`. The corrector step retargets complementarity to
  `−xᵢ sᵢ + σ·mu − Δx_affᵢ Δs_affᵢ` (Mehrotra's second-order term). Because `D` is
  unchanged between predictor and corrector, the Cholesky factor is computed
  **once per iteration and reused for both solves**.
- **Fraction-to-the-boundary.** Steps are scaled by `γ = 0.99` times the largest
  factor keeping `x, s` strictly positive, so the iterate never leaves the
  interior.
- **Starting point.** A Mehrotra-style start from `A Aᵀ` (the same solver with
  `D = I`) places the initial iterate well inside the positive orthant; if `A Aᵀ`
  is not factorable it falls back to `x = s = 1`, `y = 0`.

## Defaults

| Constant | Value | Meaning |
| :--- | :--- | :--- |
| `tolerance` | `1e-9` | threshold on scaled primal residual, dual residual, and `mu` |
| max iterations | `200` | Newton-iteration cap before an honest `not_implemented` |
| `γ` (fraction-to-boundary) | `0.99` | keeps `x, s > 0` strictly |
| divergence norm | `1e13` | inf-norm above which the iterate is deemed diverging |
| regularization | `1e-12` | diagonal nudge keeping `A D Aᵀ` factorable |

## API

```cpp
enum class IpmStatus : std::uint8_t { optimal, infeasible, unbounded };

struct IpmSolution {
    IpmStatus status;
    double objective;         // c · x at the optimum (accurate to ~tolerance)
    std::vector<double> x;    // primal optimum, length n
    std::vector<double> y;    // dual optimum / shadow prices, length m
};

[[nodiscard]] auto solve_ipm(const std::vector<std::vector<double>>& A,
                             const std::vector<double>& b,
                             const std::vector<double>& c) -> Result<IpmSolution>;
```

### `solve_ipm`

Solve `min c · x` over `{ x : A x = b, x ≥ 0 }`, with `A` an `m × n` matrix
(`m` constraints, `n` variables), `b` of length `m`, and `c` of length `n`. The
dimensions are derived as `m = A.size()` and `n = c.size()`. Inequality problems
are cast to this form by the caller through slack variables (see the worked
example).

On success:

- **`IpmStatus::optimal`** — the gap-certified optimum: `objective` is `c · x`,
  `x` the primal solution, and `y` the dual solution / shadow prices, each
  accurate to about `tolerance`.
- **`IpmStatus::unbounded`** — the primal iterate diverges (the objective
  decreases without bound along a feasible ray). `objective` is `0` and `x`, `y`
  are empty and not meaningful.
- **`IpmStatus::infeasible`** — the dual iterate diverges (the feasible set is
  empty). `objective` is `0` and `x`, `y` are empty and not meaningful.

Infeasible/unbounded detection is a divergence heuristic on the iterate's
inf-norm and is best-effort; it is exercised by the tests but is not a proof.

## Error model

| Condition | Error |
| :--- | :--- |
| `m == 0` or `n == 0` (`m = A.size()`, `n = c.size()`) | `MathError::domain_error` |
| `b.size()` disagrees with the number of constraints `m` | `MathError::domain_error` |
| A ragged `A` — some row's width `!= n` | `MathError::domain_error` |
| A non-finite (`NaN`/`Inf`) entry in `A`, `b`, or `c` | `MathError::domain_error` |
| Normal matrix `A D Aᵀ` not SPD (numerically rank-deficient `A`) | `MathError::not_implemented` |
| Iteration cap reached without convergence or a divergence verdict | `MathError::not_implemented` |
| Numerical blow-up to non-finite iterates | `MathError::not_implemented` |

`not_implemented` is used for the "did not converge" cases because `MathError`
exposes no dedicated `no_convergence` code; it is the honest stand-in for a
genuine solver failure and is documented as such rather than masked by a wrong
"optimal".

## Worked examples

From `tests/ipm_tests.cpp`. Inequality LPs are converted to standard form by
adding slack variables and negating the objective to turn a `max` into a `min`.

```cpp
import nimblecas.ipm;
using namespace nimblecas;

// max 2x + 3y  s.t.  x + y <= 4,  x + 2y <= 5,  x,y >= 0.
// Standard form  min -2x - 3y  s.t.
//    x +  y + s1      = 4
//    x + 2y      + s2 = 5,     all vars >= 0.
// Known optimum: x = 3, y = 1, s1 = s2 = 0,  max = 9  (min objective = -9).
solve_ipm({{1, 1, 1, 0}, {1, 2, 0, 1}}, {4, 5}, {-2, -3, 0, 0}).value();
//   status = optimal,  objective ≈ -9,  x ≈ [3, 1, 0, 0]   (to ~1e-9)

// A native standard-form LP:  min x1 + 2 x2  s.t.  x1 + x2 = 1,  x >= 0.
// Known optimum: x1 = 1, x2 = 0, objective = 1, shadow price y1 = 1.
solve_ipm({{1, 1}}, {1}, {1, 2}).value();
//   status = optimal,  objective ≈ 1,  x ≈ [1, 0],  y ≈ [1]

// Unbounded:  min -x1  s.t.  x1 - x2 = 0,  x >= 0  (x1 = x2 grows freely).
solve_ipm({{1, -1}}, {0}, {-1, 0}).value();
//   status = unbounded   (never reported optimal)

// Infeasible:  x1 + x2 = -1 with x >= 0 is impossible; x3 = 1 keeps A full-rank.
solve_ipm({{1, 1, 0}, {0, 0, 1}}, {-1, 1}, {1, 1, 1}).value();
//   status = infeasible  (never reported optimal)
```

Because the solve is numerical, the tests assert `|got − expected| < 1e-6`
against the known exact optima rather than exact equality — the honest reflection
of a tolerance-converged, gap-certified result.

## See also

- [`nimblecas.lp`](lp.md) — the **exact-rational** Simplex over the same §7.22
  standard form; use it when an exact `p/q` optimum is required on a small
  problem, and `ipm` when a numerical solve of a larger model is wanted.
- [`nimblecas.matdecomp`](matdecomp.md) — exact-`Rational` LU and structural
  predicates; the `double` Cholesky here is separate and inline because that
  module is exact-only.
- [`nimblecas.core`](core.md) — the `Result` / `MathError` error model this
  module reports through.
- [Documentation hub](../Index.md)
```
