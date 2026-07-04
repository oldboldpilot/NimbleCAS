# `nimblecas.dae` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/dae/dae.cppm`

Exact solver for **linear constant-coefficient differential-algebraic equations
(DAEs)** — systems that couple differential dynamics to algebraic constraints —
returned as truncated **power series over Q**. It handles two problem forms: the
original **semi-explicit index-1** block system and, by index reduction, the
**general** matrix-pencil form `E x' = A x + f` of **arbitrary differentiation
index**. Every coefficient matrix is an exact rational [`Matrix`](matrix.md),
every forcing is a vector of truncated [`powerseries`](../Index.md) over **Q**,
the linear algebra (inverses, solves, null spaces, RREF) is exact over **Q**, and
once index reduction has exposed an explicit ODE the trajectory is produced by the
graded Picard/Taylor recursion of [`nimblecas.ode`](ode.md) — the sibling
exact-DE module this one reduces onto. **No floating point is used anywhere.**

**Honesty boundary — scope.** This layer is scoped to **LINEAR
CONSTANT-COEFFICIENT** DAEs with a **REGULAR pencil** (i.e. `det(sE − A)` is not
identically zero). **Nonlinear** or **variable-coefficient** higher-index DAEs are
**out of scope** — they are not even expressible through the constant `Matrix`
`E`, `A` (and block `A`, `B`, `C`, `D`) interface, so the boundary is structural,
not a runtime check; a caller reaching for them should treat this as
`not_implemented`. A genuinely **non-regular / singular pencil** (the shuffle
never terminates: `E` stays singular after `n` passes) is rejected with
`MathError::domain_error` rather than returning a wrong answer.

**Honesty boundary — the two forms.** The **semi-explicit index-1** case
(`solve_linear_index1_dae`) is the original entry point: it pins the algebraic
variables pointwise via `y = −D⁻¹(C x + q)` (valid exactly when `D` is
invertible), substitutes to get the reduced ODE `x' = (A − B D⁻¹ C) x + (p − B
D⁻¹ q)`, and solves that with `nimblecas.ode`. A **singular `D`** there is a
higher-index problem and is refused with `domain_error`. The **general** case
(`solve_linear_dae` on `E x' = A x + f`) instead performs **index reduction** by
the classical **shuffle algorithm** (Gantmacher): compute the **left null space**
`W` of `E` (rows `w` with `w E = 0`), form the **algebraic constraints**
`(W A) x + (W f) = 0`, **differentiate** them to `(W A) x' = −(W f)'`, rebuild a
square pencil that keeps the surviving differential rows of `E` and installs `W A`
in place of the zero rows, and **repeat until `E` becomes invertible**; the
resulting plain ODE `x' = E⁻¹ A x + E⁻¹ f` is then handed to `nimblecas.ode`. The
**differentiation index** is the number of passes required (index 0 = the pencil
is already an ODE).

**Honesty boundary — the smoothness caveat (read this).** Index reduction
**differentiates the forcing `f`**, and `f` is **truncated to `order` BEFORE
reduction begins**. Each differentiation lowers the degree of the working
truncation, so for a **non-terminating** forcing the **top ≈ index coefficients**
of the solution are lost under differentiation — and this loss is **NOT
recovered** by supplying `f` at a higher input order, because it is retruncated to
`order` first. The exactly-solvable case is therefore **polynomial / terminating**
forcing whose nonzero terms all lie within `order` (exactly what `nimblecas.ode`
supports); such a problem is **exact to `order`**. Everything else in this layer
is **exact over Q** with an implicit `O(x^order)` tail. Rule 32 railway: every
`Rational` / `powerseries` / `matrix` / `ode` `Result` error is propagated;
dimension mismatches, `order == 0`, a non-regular pencil, a non-invertible `D`,
and an inconsistent `x0` under `require` surface as `MathError::domain_error`.

```cpp
import nimblecas.dae;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), [`matrix`](matrix.md),
`powerseries` (the exact graded series arithmetic carrying every forcing and
solution component), and [`ode`](ode.md) (the power-series ODE engine every
reduced system is solved by).

## Consistency and the constraint manifold

For any higher-index DAE the initial vector is **not free**. Every constraint
`(W A) x + (W f) = 0` accumulated during reduction must hold along the true
trajectory, hence at `t = 0` it constrains the supplied initial vector. The union
of these rows defines the affine **consistency manifold** of admissible initial
vectors. A vector `x0` is **consistent** when it satisfies every one of these
hidden constraints at `t = 0`. The library exposes whether a given `x0` lies on
the manifold, can orthogonally **project** it onto the manifold (exactly over Q),
and lets the solvers choose a policy for an inconsistent guess:

`enum class ConsistencyPolicy : std::uint8_t`

| Value | Meaning |
| :--- | :--- |
| `require` | Reject an inconsistent initial vector with `MathError::domain_error`. |
| `project` | Orthogonally project it (over Q) onto the manifold, then solve. |
| `underlying` | Solve the underlying ODE from the vector **as given** (may leave the manifold). |

For the index-1 semi-explicit form the algebraic initial value is likewise **not
free** but **determined** by consistency, `y(0) = −D⁻¹(C x0 + q(0))`, and is
produced as the `x^0` coefficient of the returned `y`.

## Result structures

```cpp
struct DaeSolution {
    std::vector<PowerSeries> x;  // differential variables, nd components
    std::vector<PowerSeries> y;  // algebraic variables,     na components
};

struct LinearDaeSolution {
    std::vector<PowerSeries> x;  // n state components
    std::size_t index{0};        // differentiation index
    bool consistent{true};       // did the supplied x0 lie on the constraint manifold?
};

struct SemiExplicitDaeSolution {
    std::vector<PowerSeries> x;  // differential components, nd
    std::vector<PowerSeries> y;  // algebraic   components, na
    std::size_t index{0};        // differentiation index
    bool consistent{true};       // did the supplied [x0; y0] lie on the manifold?
};
```

`DaeSolution` is the index-1 result: `x` are the `nd` differential and `y` the
`na` algebraic components, each a `PowerSeries` of the requested order; `y` is the
consistent algebraic trajectory `−D⁻¹(C x + q)`. `LinearDaeSolution` adds the
computed differentiation `index` (0 = the pencil is already an ODE) and
`consistent`, which reports whether the **originally supplied** initial vector lay
on the manifold (recorded **before** any projection under
`ConsistencyPolicy::project`). `SemiExplicitDaeSolution` is the higher-index
semi-explicit result, splitting the state back into `x` and `y`.

## API

### `solve_linear_index1_dae` — the original semi-explicit index-1 solver

```cpp
[[nodiscard]] auto solve_linear_index1_dae(
    const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D,
    const std::vector<PowerSeries>& p, const std::vector<PowerSeries>& q,
    const std::vector<Rational>& x0, std::size_t order) -> Result<DaeSolution>;
```

Solve the semi-explicit linear **index-1** DAE

```
x' = A x + B y + p,     0 = C x + D y + q,     x(0) = x0
```

exactly as truncated power series with `order` coefficients (terms `x^0 …
x^{order-1}`). Shapes (else `domain_error`): `A` is `nd × nd`, `B` is `nd × na`,
`C` is `na × nd`, `D` is `na × na`; `p` has length `nd`, `q` has length `na`, `x0`
has length `nd`; `order ≥ 1`. **`D` must be INVERTIBLE** — a singular `D` is a
higher-index (not index-1) problem and is rejected with `domain_error` (use
`solve_semiexplicit_dae` for that case). The forcings `p`, `q` may carry any
order; each component is retruncated to `order` (zero-padded or truncated) before
use. Method: reduce to `M = A − B D⁻¹ C` and `r = p − B D⁻¹ q`, solve
`x' = M x + r` via `nimblecas.ode::solve_first_order_system`, then recover
`y = −D⁻¹(C x + q)`.

### `linear_dae_index` — the differentiation index of a pencil

```cpp
[[nodiscard]] auto linear_dae_index(const Matrix& E, const Matrix& A)
    -> Result<std::size_t>;
```

The differentiation index of the regular matrix pencil `(E, A)`: the number of
times the algebraic constraints must be differentiated to expose an explicit ODE
for `x'`. `E` and `A` must be square and equal-sized (else `domain_error`); index
0 means `E` is already invertible. A **non-regular** pencil (`det(sE − A)`
identically zero — the shuffle fails to terminate) yields `domain_error`. The
index is **forcing-independent** (computed with a zero forcing at order 1).

### `linear_dae_is_consistent` — is `x0` on the constraint manifold?

```cpp
[[nodiscard]] auto linear_dae_is_consistent(
    const Matrix& E, const Matrix& A, const std::vector<PowerSeries>& f,
    const std::vector<Rational>& x0, std::size_t order) -> Result<bool>;
```

Whether `x0` is a consistent initial value for `E x' = A x + f`, i.e. it satisfies
every hidden algebraic constraint accumulated during index reduction, evaluated at
`t = 0`. Shapes: `E`, `A` are `n × n`, `f` and `x0` have length `n`, `order ≥ 1`
(else `domain_error`); a non-regular pencil is `domain_error`.

### `project_to_consistent` — nearest consistent initial vector

```cpp
[[nodiscard]] auto project_to_consistent(
    const Matrix& E, const Matrix& A, const std::vector<PowerSeries>& f,
    const std::vector<Rational>& x0, std::size_t order)
    -> Result<std::vector<Rational>>;
```

Orthogonally project `x0` (exactly over Q) onto the affine constraint manifold of
`E x' = A x + f` at `t = 0`, returning the nearest consistent initial vector; an
already consistent `x0` is returned unchanged. Shapes / non-regular-pencil errors
as above.

### `solve_linear_dae` — the general arbitrary-index solver

```cpp
[[nodiscard]] auto solve_linear_dae(
    const Matrix& E, const Matrix& A, const std::vector<PowerSeries>& f,
    const std::vector<Rational>& x0, std::size_t order,
    ConsistencyPolicy policy = ConsistencyPolicy::require)
    -> Result<LinearDaeSolution>;
```

Solve the general linear constant-coefficient DAE `E x' = A x + f(t)`,
`x(0) = x0`, exactly as truncated power series with `order` coefficients, by index
reduction (the shuffle algorithm) onto `nimblecas.ode`. Shapes (else
`domain_error`): `E`, `A` are `n × n`; `f` and `x0` have length `n`; `order ≥ 1`.
**`E` may be singular.** `policy` selects how an inconsistent `x0` is handled
(`require` / `project` / `underlying`; default `require`). Returns the state
trajectory, the computed differentiation index, and whether the supplied `x0` was
consistent. A non-regular pencil is rejected with `domain_error`; every
`rational` / `series` / `matrix` / `ode` error is propagated. (Recall the
smoothness caveat above: the forcing is truncated to `order` before it is
differentiated by reduction.)

### `solve_semiexplicit_dae` — higher-index semi-explicit solver

```cpp
[[nodiscard]] auto solve_semiexplicit_dae(
    const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D,
    const std::vector<PowerSeries>& p, const std::vector<PowerSeries>& q,
    const std::vector<Rational>& x0, const std::vector<Rational>& y0,
    std::size_t order, ConsistencyPolicy policy = ConsistencyPolicy::project)
    -> Result<SemiExplicitDaeSolution>;
```

Solve a possibly **higher-index** semi-explicit linear DAE

```
x' = A x + B y + p,     0 = C x + D y + q,     [x; y](0) = [x0; y0]
```

— the block layout of `solve_linear_index1_dae`, but with **no** requirement that
`D` be invertible. It is embedded as the general DAE `E z' = M z + f` with
`z = [x; y]`, `E = diag(I_nd, 0_na)`, `M = [[A, B], [C, D]]`, `f = [p; q]`, solved
by `solve_linear_dae`, then split back into `x` and `y`. Because higher index
constrains the initial data, the **full** guess `[x0; y0]` is supplied and, under
the default `ConsistencyPolicy::project`, projected onto the constraint manifold
(so a merely approximate `y0` guess is corrected exactly). Shapes as in
`solve_linear_index1_dae` with `y0` of length `na`; `order ≥ 1`. Non-regular
pencil / shape errors are `domain_error`.

## Error model

| Condition | Error |
| :--- | :--- |
| Any solver with `order == 0` | `MathError::domain_error` |
| Shape mismatch (`A`/`D` not square, off-diagonal blocks or `E`/`A` not conforming, `p`/`q`/`f`/`x0`/`y0` of the wrong length, `E`/`A` sizes unequal, `n == 0`) | `MathError::domain_error` |
| `solve_linear_index1_dae` with a **singular `D`** (a higher-index problem) | `MathError::domain_error` (surfaced as the matrix `inverse` failure) |
| **Non-regular pencil** in the general path (`E` still singular after `n` shuffle passes; `det(sE − A) ≡ 0`) | `MathError::domain_error` |
| Inconsistent `x0` under `ConsistencyPolicy::require` | `MathError::domain_error` |
| A `Matrix` `solve` / `inverse` failure (singular reduced pencil `E*`, singular `D`, singular normal matrix) | that operation's error, propagated |
| Any `int64` numerator/denominator computation in `Rational` wraps (RREF, matrix products, series arithmetic) | `MathError::overflow`, propagated |
| Any `powerseries` / `matrix` / `ode` operation fails | that operation's error, propagated verbatim |

The solvers thread every fallible step through `Result`; nothing throws.

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.powerseries;
import nimblecas.dae;
using namespace nimblecas;

auto ri  = [](std::int64_t v) { return Rational::from_int(v); };
auto rat = [](std::int64_t n, std::int64_t d) { return *Rational::make(n, d); };
auto mat = [](std::vector<std::vector<Rational>> rows) { return *Matrix::from_rows(std::move(rows)); };
auto ser = [](std::vector<Rational> c, std::size_t order) { return *PowerSeries::from_coeffs(std::move(c), order); };

// --- Index-1: x' = x, 0 = x - y  (A=[1], B=[0], C=[1], D=[-1]) -> x = y = e^t --
const std::size_t order = 7;
auto s1 = solve_linear_index1_dae(
    mat({{ri(1)}}), mat({{ri(0)}}), mat({{ri(1)}}), mat({{ri(-1)}}),
    {ser({ri(0)}, order)}, {ser({ri(0)}, order)}, {ri(1)}, order).value();
s1.x[0].coefficient(2);            // 1/2    (e^t: 1, 1, 1/2, 1/6, 1/24, ...)
s1.y[0].is_equal(s1.x[0]);         // true   (y = x exactly)

// --- Index-2 nilpotent-mass DAE:  E x' = A x + f  with
//     E = [[0,1],[0,0]],  A = I,  f = [0; t]  ->  closed form x = [-1, -t] --------
auto E = mat({{ri(0), ri(1)}, {ri(0), ri(0)}});
auto A = mat({{ri(1), ri(0)}, {ri(0), ri(1)}});
auto f = std::vector<PowerSeries>{ser({ri(0)}, 6), ser({ri(0), ri(1)}, 6)};  // [0, t]

linear_dae_index(E, A).value();                                 // 2   (differentiation index)
linear_dae_is_consistent(E, A, f, {ri(-1), ri(0)}, 6).value();  // true  (on the manifold)
linear_dae_is_consistent(E, A, f, {ri( 0), ri(0)}, 6).value();  // false (off the manifold)

// require: solve from the consistent x0 = [-1, 0].
auto d2 = solve_linear_dae(E, A, f, {ri(-1), ri(0)}, 6, ConsistencyPolicy::require).value();
d2.index;                          // 2
d2.consistent;                     // true
d2.x[0].coefficient(0);            // -1     (x1 = -1)
d2.x[1].coefficient(1);            // -1     (x2 = -t)

// require rejects an inconsistent x0; project repairs it to the same trajectory.
solve_linear_dae(E, A, f, {ri(0), ri(0)}, 6, ConsistencyPolicy::require).error();  // domain_error
auto pr = solve_linear_dae(E, A, f, {ri(0), ri(0)}, 6, ConsistencyPolicy::project).value();
pr.consistent;                     // false  (records the ORIGINAL x0 was inconsistent)
pr.x[0].coefficient(0);            // -1     (projected trajectory: x1 = -1)
pr.x[1].coefficient(1);            // -1     (x2 = -t)

// explicit projection lands exactly on the manifold [-1, 0].
auto proj = project_to_consistent(E, A, f, {ri(0), ri(0)}, 6).value();
proj[0];                           // -1
proj[1];                           //  0

// --- Higher-index semi-explicit (singular D):  x' = y, 0 = x - t  -----------------
//     A=[0], B=[1], C=[1], D=[0], p=0, q=-t  ->  x = t, y = 1 (index 2).
auto se = solve_semiexplicit_dae(
    mat({{ri(0)}}), mat({{ri(1)}}), mat({{ri(1)}}), mat({{ri(0)}}),
    {ser({ri(0)}, 6)}, {ser({ri(0), ri(-1)}, 6)},   // p = 0, q = -t
    {ri(0)}, {ri(0)}, 6, ConsistencyPolicy::project).value();
se.index;                          // 2
se.x[0].coefficient(1);            // 1      (x = t)
se.y[0].coefficient(0);            // 1      (y = 1, projected from the guess y0 = 0)

// --- Degenerate arguments are domain errors --------------------------------------
solve_linear_index1_dae(mat({{ri(1)}}), mat({{ri(0)}}), mat({{ri(1)}}), mat({{ri(0)}}),
                        {ser({ri(0)}, 6)}, {ser({ri(0)}, 6)}, {ri(0)}, 6).error();  // singular D
solve_linear_dae(E, A, f, {ri(-1), ri(0)}, 0).error();                             // order 0
```

## See also

- [`nimblecas.ode`](ode.md) — the exact power-series ODE engine every reduced
  system is solved by; the sibling exact-DE module this one reduces onto.
- [`nimblecas.dde`](dde.md) — exact power-series solver for **delay** differential
  equations by the method of steps.
- [`nimblecas.pde`](pde.md) — exact series methods for **partial** differential
  equations.
- [`nimblecas.sde`](sde.md) — the exact route to **stochastic** differential
  equations.
- [`nimblecas.matrix`](matrix.md) — the exact rational matrix layer supplying
  every `E`, `A`, `B`, `C`, `D`, and the inverses / solves / ranks / null spaces
  that drive index reduction.
- [Documentation hub](../Index.md)
