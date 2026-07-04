# `nimblecas.splines` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/splines/splines.cppm`

Exact piecewise-polynomial **geometry** over the rationals `Q` (ROADMAP §7). For
rational knots, values, control points, weights, and a rational parameter, every
construction in this module is **exact** — there is no floating point and no
tolerance anywhere. Failure is reported only on the railway
(`Result<T>` / `MathError`); nothing throws.

Four families are provided:

- **Cubic splines** — natural / clamped / periodic, via the tridiagonal moment system.
- **Hermite / PCHIP** — piecewise-cubic `C¹` curves from slopes.
- **Bézier** — Bernstein-basis curves (de Casteljau), 1-D and 2-D.
- **B-splines & NURBS** — Cox–de Boor basis; rational-weight NURBS.

## The overflow contract

`Rational` is an `int64`-backed reduced fraction whose arithmetic
(`add`/`subtract`/`multiply`/`divide`) returns `Result<Rational>`; an `int64`
boundary in any intermediate propagates as `MathError::overflow` (Rule 32).
Every construction and evaluation below threads that railway — a `MathError`
out is always an honest "could not compute exactly", never a wrong value.

## Honesty boundary

Everything here is **exact over `Q`** for rational data: evaluation at a rational
parameter is the exact rational value, with **no numerical error to report**.
Two boundaries are documented and enforced rather than papered over:

- **PCHIP slope choice is a heuristic.** `HermiteSpline::pchip` derives its knot
  slopes by the Fritsch–Carlson shape-preserving (monotonicity-tending) rule.
  Every arithmetic step is exact and reproducible bit-for-bit over `Q`, but the
  *slope selection itself* is a shape decision, not the unique slope forced by an
  interpolation condition. `HermiteSpline::from_slopes` is the fully-exact path
  when you supply the slopes.
- **NURBS covers the rational-weight case only.** Some classical shapes need
  irrational data (an exact circle needs weight `cos θ` at a corner); that lies
  outside the rational core and is deliberately **not** represented — not faked.

Conditioning and shape (Runge-like oscillation, poor spline shape for badly
spaced knots) are properties of the **data/knots**, not errors introduced here.

## Structural preconditions → `domain_error`

Rejected with `MathError::domain_error` (never UB / out-of-bounds):

- unsorted or duplicate knots where strict monotonicity is required,
- mismatched array sizes (values vs knots, control vs weights, control vs knot span),
- a Bézier parameter `t ∉ [0,1]`, or a B-spline/NURBS parameter `u` outside the knot domain,
- too few points for the requested construction,
- a non-periodic (`y₀ ≠ yₙ`) request to the periodic cubic-spline constructor,
- a zero NURBS weight-denominator at the evaluation parameter.

## Types

`struct Point2 { Rational x, y; }` — a 2-D rational point.
`enum class SplineBoundary { natural, clamped, periodic }`.

## `CubicSpline` — `C²` piecewise-cubic interpolation

The `C²`-continuity conditions on the second derivatives (the *moments*
`M_i = S''(x_i)`) form a **tridiagonal** linear system, assembled and solved
exactly over `Q` by the Thomas algorithm in `nimblecas.bandsolve`
(`solve_tridiagonal`).

| Constructor | Boundary condition |
| :--- | :--- |
| `static natural(xs, ys)` | `S'' = 0` at both ends (`M₀ = Mₙ = 0`); interior moments solved. |
| `static clamped(xs, ys, deriv_left, deriv_right)` | given end first-derivatives; all moments solved. |
| `static periodic(xs, ys)` | `S, S', S''` wrap (requires `y₀ = yₙ`). The wrap makes the system **cyclic**-tridiagonal; reduced to two ordinary Thomas solves by the exact Sherman–Morrison correction — still exact over `Q`. |

Accessors: `knots()`, `values()`, `moments()`, `boundary()`, `piece_count()`,
`piece(i)` (the exact cubic `RationalPoly` on interval `i`, in absolute `x`),
`pieces()`. Evaluation: `evaluate(x)` locates the interval and evaluates the
exact cubic.

## `HermiteSpline` — `C¹` piecewise-cubic Hermite

- `static from_slopes(xs, ys, slopes)` — the **exact** interpolant matching value
  `y_i` and slope `d_i` at each knot.
- `static pchip(xs, ys)` — slopes chosen by the Fritsch–Carlson shape rule (see
  Honesty boundary).

Accessors: `slopes()`, `piece(i)`; evaluation `evaluate(x)`.

## `BezierCurve` — Bernstein-basis, 1-D

- `static make(control)` — from rational control points.
- `evaluate(t)` — **de Casteljau** (repeated exact convex combinations), `t ∈ [0,1]`.
- `evaluate_bernstein(t)` — Bernstein-sum evaluation (agrees exactly with de Casteljau).
- `elevate()` — exact degree elevation.
- `to_power_basis() -> RationalPoly` / `static from_power_basis(p, degree)` — exact conversions.
- free `bezier_subdivide(curve, t) -> BezierSplit{left, right}` — exact de Casteljau split.

## `BezierCurve2` — Bernstein-basis, 2-D

`make` / `evaluate` / `elevate` / `control_points` / `degree`, evaluated per
coordinate through a scalar curve — exact over `Q`.

## `BSpline` — Cox–de Boor basis

- `static make(knots, control, degree)`.
- `basis_function(i, u)` — the Cox–de Boor recursion `N_{i,p}(u)`, exact over `Q`.
- `evaluate(u)` — `C(u) = Σ_i N_{i,p}(u) P_i`.
- `domain_min()` / `domain_max()`.

Open/clamped knot vectors are supported: the endpoint special case makes a
clamped curve interpolate its end control points and gives **partition of unity**
`Σ_i N_{i,p}(u) = 1` exactly.

## `NurbsCurve` — rational B-splines

- `static make(knots, control, weights, degree)`.
- `evaluate(u)` — `C(u) = (Σ_i w_i N_{i,p}(u) P_i) / (Σ_i w_i N_{i,p}(u))`, exact
  over `Q` for rational weights. A zero weight-denominator at `u` is a `domain_error`.

## Error model

| `MathError` | Cause |
| :--- | :--- |
| `domain_error` | any structural precondition above is violated. |
| `overflow` | an `int64` boundary in a `Rational` step (Rule 32). |

## See also

- [`interpolation.md`](interpolation.md) — global exact polynomial interpolation over `Q`.
- [`ratpoly.md`](ratpoly.md) — the `RationalPoly` piece type.
- `nimblecas.bandsolve` — the exact tridiagonal Thomas solver used for cubic-spline moments.
