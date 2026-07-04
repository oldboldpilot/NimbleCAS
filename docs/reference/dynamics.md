# `nimblecas.dynamics` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/dynamics/dynamics.cppm`

Exact stability analysis of **linear/affine autonomous systems** `dx/dt = A x (+ b)`
(ROADMAP §7.10). The whole module sits on top of the rational eigen substrate —
[`matrix`](matrix.md) and [`eigen`](eigen.md) over the exact `Rational` field —
and never touches floating point. Its distinguishing move is that stability is
decided from the **characteristic polynomial's coefficients** via the
Routh–Hurwitz criterion, not by finding roots: because the Routh array is built
in exact `Rational` arithmetic and only inspects signs, it settles
**irrational and complex spectra** that rational-root testing over `Q` cannot
see — for instance the pure-imaginary pair `±i` of a rotation. That is precisely
why Routh–Hurwitz is used here.

The honesty boundary. Everything stays inside exact `Rational` arithmetic, so
there is no numerical error and no root-finding tolerance anywhere. Two exact
capabilities are cleanly separated:

- `is_asymptotically_stable` is a **coefficient-level** decision. It is total
  over the whole spectrum (rational, irrational, or complex) because it never
  factors the polynomial — it only reads the Routh array's first column.
- `classify_equilibrium` gives a finer, human-readable verdict **only when the
  spectrum is fully rational** (the multiplicities returned by
  `rational_eigenvalues` sum to `n`). When part of the spectrum is
  irrational/complex — invisible over `Q` — it deliberately falls back to the
  Routh–Hurwitz verdict and never claims a `"center"`/`"focus"`, because
  rational eigenvalues cannot detect complex conjugate pairs.

Every fallible step is threaded on the [`Result`](core.md) railway. Overflow in
the exact arithmetic surfaces as `MathError::overflow` (an `int64`
numerator/denominator wrap inside a Routh entry), and every shape or
dimension violation surfaces as `MathError::domain_error`.

```cpp
import nimblecas.dynamics;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), [`matrix`](matrix.md),
and [`eigen`](eigen.md).

## Free functions

All three entry points are free functions in namespace `nimblecas`; there is no
class to construct. Each takes exact `Rational` matrices and returns a
[`Result`](core.md).

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `fixed_point_affine` | `[[nodiscard]] auto fixed_point_affine(const Matrix& a, const Matrix& b) -> Result<Matrix>` | The equilibrium (fixed point) of the affine map `x ↦ A x + b`, i.e. the exact solution of `(I − A) x = b` over `Q`. `a` must be square (`n × n`) and `b` an `n × 1` column, else `domain_error`. A non-isolated equilibrium — where `(I − A)` is singular — propagates as `domain_error` from the underlying solve. On success returns the `n × 1` column `x`. |
| `is_asymptotically_stable` | `[[nodiscard]] auto is_asymptotically_stable(const Matrix& a) -> Result<bool>` | Whether `dx/dt = A x` is asymptotically stable: `true` iff every eigenvalue of `A` has strictly negative real part (Hurwitz). Decided **exactly** from the characteristic polynomial via Routh–Hurwitz, with no root finding, so it also decides irrational/complex spectra. Requires a square matrix with `n ≥ 1`, else `domain_error`. |
| `classify_equilibrium` | `[[nodiscard]] auto classify_equilibrium(const Matrix& a) -> Result<std::string>` | A human-readable classification of the equilibrium at the origin of `dx/dt = A x`. Uses the exact rational sign pattern when the spectrum is fully rational; otherwise falls back to the Routh–Hurwitz verdict (see below). Requires a square matrix, else `domain_error`. |

### `fixed_point_affine` — the affine equilibrium

Solves `(I − A) x = b` exactly over `Q`. The map's fixed point satisfies
`x = A x + b`, i.e. `(I − A) x = b`; the routine forms `I − A` and defers to the
matrix solve. Because everything is `Rational`, the returned column is the exact
equilibrium, not an approximation.

- Non-square `a` → `domain_error`.
- `b` whose row count differs from `a.rows()`, or `b` that is not a single
  column (`b.cols() != 1`) → `domain_error`.
- Singular `(I − A)` (a continuum of equilibria rather than an isolated one) →
  `domain_error`, propagated from the solve.

### `is_asymptotically_stable` — exact Routh–Hurwitz

The characteristic polynomial of `A` (monic, degree `n`) is taken from
[`eigen`](eigen.md), its coefficients laid out in descending order, and a Routh
array of `n + 1` rows is filled in exact `Rational` arithmetic. The system is
Hurwitz-stable **iff every entry of the array's first column is nonzero and
shares the leading term's (positive) sign**.

Two edge cases both mean at least one root lies **on or to the right of** the
imaginary axis, and so are reported as **not** asymptotically stable (`false`),
never as an error:

- a **zero pivot** in the first column (whether or not the rest of that row is
  nonzero); and
- a **fully-zero row** (imaginary-axis roots) — e.g. the second Routh row of a
  rotation `[[0,−1],[1,0]]`, whose spectrum `±i` rational-root testing over `Q`
  could not decide.

### `classify_equilibrium` — rational sign pattern, with an exact fallback

`rational_eigenvalues(A)` supplies the rational part of the spectrum with
multiplicities. If those multiplicities **sum to `n`** the spectrum is fully
rational and the verdict is exact from the sign pattern. A positive eigenvalue
forces instability regardless of the rest, so the positive cases are decided
before the zero (marginal) case (otherwise `diag(0, 1)`, which grows like
`e^t`, would be mislabelled marginal):

| Rational spectrum | Verdict string |
| :--- | :--- |
| a positive **and** a negative eigenvalue | `"saddle"` |
| all non-negative, some positive, no zero | `"unstable node"` |
| all non-negative, some positive, a zero | `"unstable (with marginal direction)"` |
| all non-positive, some negative, no zero | `"stable node"` |
| all non-positive, some negative, a zero | `"marginally stable"` |
| all eigenvalues zero (or `n == 0`) | `"degenerate/marginal"` |

Otherwise part of the spectrum is irrational/complex and cannot be seen over
`Q`, so the verdict defers to the exact `is_asymptotically_stable` result:

| Routh–Hurwitz verdict | Verdict string |
| :--- | :--- |
| asymptotically stable | `"asymptotically stable (spiral/node)"` |
| not asymptotically stable | `"unstable or marginal (non-rational spectrum)"` |

Because rational eigenvalues cannot detect complex conjugate pairs,
`"center"`/`"focus"` is never claimed from them.

## Error model

| Condition | Error |
| :--- | :--- |
| `fixed_point_affine`: `a` not square | `MathError::domain_error` |
| `fixed_point_affine`: `b.rows() != a.rows()` or `b.cols() != 1` | `MathError::domain_error` |
| `fixed_point_affine`: `(I − A)` singular (non-isolated equilibrium) | `MathError::domain_error` (from the solve) |
| `is_asymptotically_stable`: `a` not square, or `a.rows() < 1` | `MathError::domain_error` |
| `classify_equilibrium`: `a` not square | `MathError::domain_error` |
| An `int64` numerator/denominator computation wraps in a Routh entry, the characteristic polynomial, or the eigenvalue/solve substrate | `MathError::overflow` |

The two boundary cases of `is_asymptotically_stable` — a zero first-column entry
and a fully-zero row — are **results, not errors**: they return `false`. A
`domain_error` or `overflow` from the underlying `eigen`/`matrix` substrate is
propagated unchanged.

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.dynamics;
using namespace nimblecas;

auto ri = [](std::int64_t v) { return Rational::from_int(v); };

// Build an integer matrix from rows, and an n×1 integer column.
auto mat = [&](std::vector<std::vector<std::int64_t>> rows) {
    std::vector<std::vector<Rational>> r;
    for (const auto& row : rows) {
        std::vector<Rational> rr;
        for (std::int64_t v : row) rr.push_back(ri(v));
        r.push_back(std::move(rr));
    }
    return Matrix::from_rows(std::move(r)).value();
};
auto col = [&](std::vector<std::int64_t> es) {
    std::vector<std::vector<std::int64_t>> rows;
    for (std::int64_t v : es) rows.push_back({v});
    return mat(std::move(rows));
};

// --- fixed_point_affine: exact equilibrium of x ↦ A x + b -----------------
// A = 0 ⇒ (I − A) = I ⇒ x = b.
auto x0 = fixed_point_affine(mat({{0, 0}, {0, 0}}), col({3, 5})).value();
x0.at(0, 0); x0.at(1, 0);                       // 3, 5

// A = 2I ⇒ (I − A) = −I ⇒ x = −b = [−1, −1].
auto x1 = fixed_point_affine(mat({{2, 0}, {0, 2}}), col({1, 1})).value();
x1.at(0, 0); x1.at(1, 0);                       // −1, −1

// A = I ⇒ (I − A) = 0, a non-isolated equilibrium ⇒ domain_error.
fixed_point_affine(mat({{1, 0}, {0, 1}}), col({1, 1})).error();  // domain_error

// --- is_asymptotically_stable: exact Routh–Hurwitz ------------------------
// Eigenvalues −1, −2 (λ² + 3λ + 2): Hurwitz.
is_asymptotically_stable(mat({{-1, 0}, {0, -2}})).value();       // true

// Eigenvalues 1, 1: not Hurwitz.
is_asymptotically_stable(mat({{1, 0}, {0, 1}})).value();         // false

// Rotation, eigenvalues ±i (λ² + 1): a fully-zero Routh row — the
// imaginary-axis case rational-root testing over Q could not decide.
is_asymptotically_stable(mat({{0, -1}, {1, 0}})).value();        // false

// Defective Jordan block, eigenvalue −1 (λ² + 2λ + 1): Hurwitz.
is_asymptotically_stable(mat({{-1, 1}, {0, -1}})).value();       // true

// --- classify_equilibrium: rational sign pattern + fallback ---------------
classify_equilibrium(mat({{-1, 0}, {0, -2}})).value();  // "stable node"
classify_equilibrium(mat({{1, 0}, {0, 2}})).value();    // "unstable node"
classify_equilibrium(mat({{-1, 0}, {0, 2}})).value();   // "saddle"

// A positive eigenvalue dominates a zero: diag(0, 1) grows like e^t.
classify_equilibrium(mat({{0, 0}, {0, 1}})).value();
                                    // "unstable (with marginal direction)"
// diag(0, −1): bounded but not asymptotically stable.
classify_equilibrium(mat({{0, 0}, {0, -1}})).value();   // "marginally stable"

// Non-rational spectrum (±i) ⇒ Routh–Hurwitz fallback string.
classify_equilibrium(mat({{0, -1}, {1, 0}})).value();
                                    // "unstable or marginal (non-rational spectrum)"

// Shape errors.
is_asymptotically_stable(mat({{1, 2, 3}, {4, 5, 6}})).error();  // domain_error
classify_equilibrium(mat({{1, 2, 3}, {4, 5, 6}})).error();      // domain_error
```

## See also

- [`nimblecas.matrix`](matrix.md) — the exact `Rational` matrices, the solve
  behind `fixed_point_affine`, and the substrate for the spectrum.
- [`nimblecas.eigen`](eigen.md) — `characteristic_polynomial` and
  `rational_eigenvalues`, the exact spectral inputs this module consumes.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field every entry,
  coefficient, and Routh-array value lives in.
- [`nimblecas.complex`](complex.md) — the exact Gaussian rationals `Q + Qi`,
  the sibling exact-numeric module for spectra that leave the reals.
- [Documentation hub](../Index.md)
