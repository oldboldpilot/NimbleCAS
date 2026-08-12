# `nimblecas.svd` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/svd/svd.cppm`

Numeric thin **singular value decomposition** (one-sided Jacobi) and **polar
decomposition**, plus an exact-over-`Q` slice of the singular spectrum. Like
[`qrschur`](qrschur.md), this module is deliberately split along its honesty
boundary: a real `m x n` matrix's singular values are the square roots of the
eigenvalues of `AᵀA`, and those eigenvalues are in general irrational, so **no
exact SVD over `Q` exists for almost every input** — faking one would violate
Rule 32. The numeric path and the exact path are offered as two separate,
clearly labelled products.

```cpp
import nimblecas.svd;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md) (`Rational`),
[`matrix`](matrix.md) (the exact rational `Matrix`, and `gram_matrix`'s
`transpose`/`multiply`), and [`eigen`](eigen.md) (`rational_eigenvalues`, the
source of `exact_singular_value_squares`).

## Honesty boundary

| Product | Regime | What it returns |
| :--- | :--- | :--- |
| `svd` | **NUMERIC** (double) | Thin SVD `A = U·diag(sigma)·Vᵀ` by **one-sided Jacobi rotation directly on `A`** (never by forming `AᵀA` and eigendecomposing it, which would square `A`'s condition number and roughly halve attainable accuracy). Accurate to `~tol` relative. |
| `polar` | **NUMERIC** (double) | Polar decomposition `A = U·P` (`U` orthogonal, `P` symmetric PSD), built from the thin SVD. |
| `gram_matrix` | **EXACT over `Q`** | The Gram matrix `AᵀA`, overflow-checked `Rational`. |
| `exact_singular_value_squares` | **EXACT over `Q`, a SLICE** | The **rational** eigenvalues of `AᵀA` — i.e. the rational `sigma^2` values, with multiplicity, descending. An irrational `sigma^2` is honestly **absent** from the result, never approximated and dressed up as exact. **No function in this module ever returns an exact irrational singular value** — `sigma` itself is never claimed exact even when `sigma^2` is rational. |

Three honest consequences enforced in code:

- **Exhausting the sweep budget is never silently accepted.** `svd` /
  `polar` return `MathError::not_converged` (propagated) if `max_sweeps`
  cyclic Jacobi sweeps all still find an off-diagonal pair to rotate —
  nothing partial or silently inaccurate is ever returned.
- **A zero singular value still yields a genuine orthonormal `U` column.**
  Rather than leaving a degenerate column at `0` (or `NaN`), it is completed
  by deterministic Gram-Schmidt against the standard basis, so `U` is always
  a real orthonormal set — verifiable via `orthonormality_defect`
  ([`qrschur`](qrschur.md)).
- **Determinism.** A fixed ascending pair order each sweep, a stable
  descending sort of `sigma` (with `U`/`V` columns carried under the same
  permutation), and a fixed sign convention (each column's largest-magnitude
  entry, lowest index on ties, is forced non-negative — negating the paired
  `U` **and** `V` column together) eliminate the usual sign/order
  nondeterminism of Jacobi-family SVD.

`exact_singular_value_squares` is a genuine **slice**: for a typical matrix
most or all of `sigma^2` are irrational and simply do not appear in the
result. It never approximates the missing entries and never reports a wrong
count as complete — callers must not assume the returned list is exhaustive.

## API

All entry points are free functions in namespace `nimblecas`, `[[nodiscard]]`.

### Numeric SVD

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `svd` (span) | `auto svd(std::span<const double> a, std::size_t rows, std::size_t cols, double tol = 1e-12, std::size_t max_sweeps = 60) -> Result<NumericSvd>` | Thin SVD of the `rows x cols` row-major `a` by one-sided Jacobi. `k = min(rows, cols)`. |
| `svd` (Matrix) | `auto svd(const Matrix& a, double tol = 1e-12, std::size_t max_sweeps = 60) -> Result<NumericSvd>` | Converts every exact `Rational` entry to `double` first — **lossy**; leaves the exact regime entirely. Use `gram_matrix`/`exact_singular_value_squares` to stay exact. |
| `svd_residual` | `auto svd_residual(const NumericSvd& d, std::span<const double> a) -> Result<double>` | Frobenius residual `‖U·diag(sigma)·Vᵀ − A‖_F`. |

```cpp
struct NumericSvd {
    std::size_t rows{}, cols{};
    std::vector<double> u{};                 // rows x k, orthonormal columns
    std::vector<double> singular_values{};    // length k, descending, >= 0
    std::vector<double> v{};                  // cols x k, orthonormal columns
};
```

### Numeric polar decomposition (square only)

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `polar` | `auto polar(std::span<const double> a, std::size_t n, double tol = 1e-12, std::size_t max_sweeps = 60) -> Result<NumericPolar>` | `A = U·P` of the `n x n` row-major `a`: `U = U_svd·V_svdᵀ` (orthogonal), `P = V_svd·diag(sigma)·V_svdᵀ` symmetrised (SPSD, `P² = AᵀA`). Built from `svd`, so it propagates `overflow`/`not_converged`. |
| `polar_residual` | `auto polar_residual(const NumericPolar& d, std::span<const double> a) -> Result<double>` | Frobenius residual `‖U·P − A‖_F`. |

```cpp
struct NumericPolar {
    std::size_t n{};
    std::vector<double> u{};  // n x n, orthogonal
    std::vector<double> p{};  // n x n, symmetric positive semidefinite
};
```

### Exact companions over `Q`

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `gram_matrix` | `auto gram_matrix(const Matrix& a) -> Result<Matrix>` | The exact `AᵀA` (via `Matrix::transpose`/`multiply`). |
| `exact_singular_value_squares` | `auto exact_singular_value_squares(const Matrix& a) -> Result<std::vector<std::pair<Rational, std::int64_t>>>` | The rational eigenvalues of `AᵀA` (`{sigma^2, multiplicity}`, descending) — a **slice**, never a claim of completeness. |

## Error model

| Condition | Error |
| :--- | :--- |
| `svd(span, ...)` / `polar(span, n, ...)`: buffer size does not match `rows*cols` / `n*n` | `MathError::domain_error` |
| Index-arithmetic (`rows*cols`, `rows*k`, `cols*k`, `n*n`) would wrap `std::size_t` | `MathError::overflow` |
| `svd` / `polar`: `max_sweeps` sweeps all still find a rotation to apply | `MathError::not_converged` |
| `gram_matrix` / `exact_singular_value_squares`: propagated `Matrix`/`eigen` errors | `MathError::domain_error` / `MathError::overflow` |

A `0x0` (or any `rows*cols == 0` / `n == 0`) input is **not** an error on any
path: it yields an empty decomposition.

## Worked examples

```cpp
import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.qrschur;   // orthonormality_defect
import nimblecas.svd;
using namespace nimblecas;

auto ri = [](std::int64_t v) { return Rational::from_int(v); };
auto mat = [](std::vector<std::vector<Rational>> r) {
    return Matrix::from_rows(std::move(r)).value();
};

// [[3,0],[4,5]]: A^T*A = [[25,20],[20,25]], exact eigenvalues 45 and 5.
// => sigma = (3*sqrt(5), sqrt(5)), and sigma1*sigma2 = 15 = |det A|.
const auto A = mat({{ri(3), ri(0)}, {ri(4), ri(5)}});
const std::vector<double> ad{3.0, 0.0, 4.0, 5.0};

// NUMERIC thin SVD: sigma matches the hand-derived irrational values to ~1e-6.
const auto d = svd(ad, 2, 2).value();
d.singular_values[0];  // ~ 3*sqrt(5) ~ 6.708...
d.singular_values[1];  // ~ sqrt(5)   ~ 2.236...
svd_residual(d, ad).value();                        // ~ 1e-13 (‖U*diag(sigma)*V^T - A‖_F)
orthonormality_defect(d.u, d.rows, 2).value();       // ~ 1e-15 (U orthonormal)

// EXACT companion: sigma^2 (never sigma itself) is exactly representable over Q.
const auto sq = exact_singular_value_squares(A).value();
sq[0].first == ri(45) && sq[0].second == 1;   // sigma^2 = 45 (mult 1) — first, descending
sq[1].first == ri(5)  && sq[1].second == 1;   // sigma^2 = 5  (mult 1) — second
// exact_singular_value_squares offers NO exact sigma, only sigma^2.

// NUMERIC polar decomposition of the same A: U orthogonal, P symmetric PSD, P^2 == A^T*A.
const auto pd = polar(ad, 2).value();
polar_residual(pd, ad).value();   // ~ 1e-13 (‖U*P - A‖_F)

// A zero singular value still yields a genuine orthonormal U: diag(3,-2,0), sigma=(3,2,0).
const std::vector<double> z{3, 0, 0, 0, -2, 0, 0, 0, 0};
const auto dz = svd(z, 3, 3).value();
dz.singular_values;  // {3, 2, 0} — descending, non-negative
orthonormality_defect(dz.u, dz.rows, 3).value();  // ~ 0 (even the sigma=0 column)

// Error cases.
svd(std::vector<double>{1, 2, 3}, 2, 2).error();   // MathError::domain_error (size mismatch)
polar(std::vector<double>{1, 2, 3, 4, 5, 6}, 2).error();  // domain_error (non-square)
```

## See also

- [`nimblecas.qrschur`](qrschur.md) — the sibling numeric/exact-split module:
  exact Gram-Schmidt QR over `Q`, numeric Householder QR, and numeric real
  Schur form; supplies `orthonormality_defect` used to verify `U`/`V` here.
- [`nimblecas.eigen`](eigen.md) — the exact rational eigenvalues of `AᵀA` that
  `exact_singular_value_squares` reads off via `rational_eigenvalues`.
- [`nimblecas.matrix`](matrix.md) — the exact rational `Matrix`
  (`transpose`, `multiply`) `gram_matrix` is built from.
- [Documentation hub](../Index.md)
