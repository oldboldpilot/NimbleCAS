# `nimblecas.solve` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/solve/solve.cppm`

Exact closed-form **polynomial root solving** (ROADMAP §7.21), building on
[`roots`](roots.md). Given a polynomial `p` over the rationals `Q[x]`,
`solve_poly` returns **all** of its roots as symbolic [`Expr`](symbolic.md)
values. It first peels the **rational** roots off with
[`rational_roots`](roots.md) (deflating `(x - r)` with multiplicity over
[`RationalPoly`](ratpoly.md)), then dispatches on the degree of the remaining
factor to the classical radical formulas — and, for a non-radical degree `>= 5`
remainder, to **numerical companion-matrix eigenvalues**.

```cpp
import nimblecas.solve;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), [`roots`](roots.md),
[`symbolic`](symbolic.md), and [`numeigen`](numeigen.md).

## What is returned

Each root is a `Root`:

```cpp
struct Root {
    Expr value;                             // the root, as a symbolic expression
    bool exact;                             // true = exact algebraic; false = numeric
    std::optional<std::uint64_t> multiplicity;  // set for rational roots; nullopt otherwise
};
```

| Factor | Method | `exact` |
| :--- | :--- | :---: |
| rational roots (any degree) | rational root theorem + deflation | `true` |
| remaining **linear** `ax + b` | `-b/a` | `true` |
| remaining **quadratic** | `(-b ± √(b²−4ac)) / (2a)` | `true` |
| remaining **cubic** | Cardano (depress, cube-root radicals, cube roots of unity) | `true` |
| remaining **quartic** | Ferrari (depress, resolvent cubic, two quadratics) | `true` |
| remaining **degree ≥ 5** (after rational-root peeling) | companion-matrix eigenvalues, in `double` | **`false`** |

Rational roots are listed **once**, carrying their `multiplicity`. Radical and
numeric roots are **flattened** (one entry per root, `multiplicity` = `nullopt`),
so the total entry count equals `deg(p)` when the higher factor is itself a
product of `≤ 4`-degree pieces.

## Honesty boundary

This is the crux of the module (Rule 32): **an approximation is never presented
as exact.**

- **Exact radicals for degree ≤ 4** and for any rational / `≤ quartic` factor
  peeled off a higher-degree polynomial. Radicals are `Expr::power(base,
  Expr::rational(1, n))`; no numeric approximation is produced on this path.
- **Imaginary / complex radicals kept symbolically.** A negative quantity under
  a square root is **not** rejected — it is kept as the exact imaginary radical
  `power(negative, 1/2)`. So the roots of `x² + 1` are `power(-1, 1/2)` (the
  imaginary unit `i`) and its negation. The cubic **casus irreducibilis**
  (negative Cardano discriminant, three real irrational roots) likewise keeps its
  **nested imaginary radicals** symbolically rather than switching to a
  trigonometric form: the result stays exact and algebraic. `i·k` is written
  `re + im · power(-1, 1/2)` in the numeric path.
- **Numeric for the degree ≥ 5 remainder.** By Abel–Ruffini the *general*
  quintic and higher has no radical solution. Rather than refuse, the roots of
  the leftover factor are computed as the **eigenvalues of its companion matrix**
  in double precision, via [`companion_eigenvalues`](numeigen.md). These are
  **iterative numerical approximations** to the tolerance `tol` (default
  `1e-12`), each tagged `exact = false`. They are honest approximations, clearly
  distinguished from the exact algebraic roots on the same result.
- **Factorisation before the numeric fallback.** After rational-root peeling the
  remainder is **factored into irreducibles over ℚ** ([`factor`](factor.md): Yun
  square-free + Kronecker). Each irreducible factor of degree ≤ 4 is solved
  **exactly by radicals**, so a *reducible* high-degree polynomial that splits
  into small pieces (e.g. `(x²−2)(x³−2)`, which has no rational roots yet exact
  radicals `±√2`, `∛2·ωᵏ`) is returned **exactly**. Only a **genuinely
  irreducible** factor of degree ≥ 5 (no radical solution by Abel–Ruffini) is sent
  to the numeric companion-eigenvalue path. If factorisation exhausts its search
  budget (`not_implemented`), the solver falls back to a degree-directed dispatch
  on the whole remainder (radicals if ≤ quartic, else numeric), so a result is
  always produced.

## API

```cpp
[[nodiscard]] auto solve_poly(const RationalPoly& p, double tol = 1e-12,
                              std::size_t max_iter = 1000) -> Result<std::vector<Root>>;
```

### `solve_poly`

All roots of `p`.

1. Reject the zero polynomial (`MathError::domain_error` — every value is a
   root); a nonzero constant yields an **empty** vector.
2. Peel rational roots with [`rational_roots`](roots.md); each becomes an
   `exact` `Root` carrying its multiplicity, and `(x - r)` is deflated out.
3. On the remaining factor of degree `d`:
   - `d ≤ 0`: done (fully split into rational roots).
   - `1 ≤ d ≤ 4`: emit the exact radical roots (linear / quadratic / cubic /
     quartic), each `exact = true`.
   - `d ≥ 5`: call [`companion_eigenvalues`](numeigen.md)`(remainder, tol,
     max_iter)`; map each `std::complex<double>` to an `Expr` — `Expr::real(re)`
     when `|im| < tol`, else `re + im · power(-1, 1/2)` — tagged `exact = false`.

`tol` sets both the real-vs-complex classification threshold and the iteration
tolerance; `max_iter` caps the eigenvalue iteration.

### The Ferrari resolvent (quartic)

After depressing to `t⁴ + p t² + q t + r`, for a root `y` of the resolvent cubic
`y³ + 2p y² + (p² − 4r) y − q² = 0` the quartic factors exactly as

```
(t² − A t + (M − B))·(t² + A t + (M + B)),   A = √y,  B = −q/(2A),  M = (p + y)/2.
```

This is an identity for **every** root `y` (and `y ≠ 0` since the resolvent's
constant term `−q² ≠ 0` when `q ≠ 0`), so the two quadratics are solved
symbolically to give the four roots. When `q = 0` the quartic is biquadratic and
solved directly (`t² = (−p ± √(p²−4r))/2`); the pure `t⁴ = −r` case yields the
clean fourth roots `⁴√(−r) · {1, i, −1, −i}`.

## Error model

| Condition | Error |
| :--- | :--- |
| `p` is the zero polynomial (every value is a root) | `MathError::domain_error` |
| An `int64` rational computation (depression coefficients, deflation) overflows | `MathError::overflow` |
| An `Expr::rational` leaf hits an `int64` boundary (e.g. `INT64_MIN`) | `MathError::overflow` |
| The numeric eigenvalue routine fails to converge / build the companion | propagated from [`numeigen`](numeigen.md) |

A nonzero constant returns an **empty vector** — not an error. Degree `≥ 5` is
**not** an error: it returns numeric roots tagged `exact = false`.

## Worked examples

From `tests/solve_tests.cpp` (integer polynomials low-degree-first, the
`RationalPoly` convention `coeffs[i] = coefficient of x^i`):

```cpp
import nimblecas.solve;
using namespace nimblecas;

// x^2 - 2            -> power(2, 1/2) and -power(2, 1/2)          (exact radicals)
// x^2 + 1            -> power(-1, 1/2) and its negation           (exact imaginary radicals)
// x^2 - 5x + 6       -> 2 and 3, exact, multiplicity 1            (rational)
// (x-1)(x^2-2)       -> 1 (exact), ±power(2, 1/2) (exact)         (rational peel + quadratic)
// x^3 - 2            -> power(2, 1/3) and two complex cube roots  (Cardano, exact)
// x^3 - 3x - 1       -> three nested-radical roots                (casus irreducibilis, exact)
// x^4 - 2            -> power(2, 1/4) · {1, i, -1, -i}            (exact)
// x^4 - 5x^2 + 6     -> four exact radicals   = (x^2-2)(x^2-3)
// x^4 - x^3 - x - 1  -> four exact radicals via Ferrari
// (x-1)(x-2)(x^3-2)  -> 1, 2 (exact) + three cube-root radicals (exact); no numeric path
// x^5 - x - 1        -> five NUMERIC eigenvalue roots (exact=false): one real ~1.1673,
//                       two complex-conjugate pairs
// (x-2)(x^5 - x - 1) -> 2 (exact) + five numeric eigenvalue roots (exact=false)
```

Every returned root — rational, radical, or numeric — satisfies `p(root) ≈ 0`
(the tests confirm this by substituting each root back and checking the residual).

## See also

- [`nimblecas.roots`](roots.md) — the rational-root pre-factoring (rational root
  theorem, `evaluate`, deflation) that `solve_poly` builds on.
- [`nimblecas.numeigen`](numeigen.md) — the companion-matrix eigenvalue routine
  used for the numeric degree `≥ 5` path.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Q[x]` substrate (`Rational`,
  `RationalPoly`, division-with-remainder).
- [`nimblecas.symbolic`](symbolic.md) — the `Expr` tree the roots are expressed in.
- [Documentation hub](../Index.md)
```
