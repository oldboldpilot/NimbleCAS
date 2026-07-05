# `nimblecas.numeigen` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/numeigen/numeigen.cppm`

The **numeric** companion to the exact [`eigen`](eigen.md) module. Where
[`eigen`](eigen.md) returns only the *rational slice* of a spectrum exactly, and
[`analysis`](analysis.md)`::dominant_eigenvalue` returns a single dominant
magnitude, this module computes **all** eigenvalues of a general **real** matrix
— real eigenvalues and complex-conjugate pairs alike — as
`std::complex<double>`, by the **QR algorithm** reduced to real Schur form. It
also exposes a numeric polynomial root finder (`companion_eigenvalues`) built on
top of it (ROADMAP §7.21 numeric path), which is what a numeric
[`solve`](solve.md) path consumes.

```cpp
import nimblecas.numeigen;
```

Depends on [`core`](core.md) (`Result` / `MathError`) and
[`ratpoly`](ratpoly.md) (`RationalPoly` / `Rational`). Eigenvalues are returned
as `std::complex<double>` from `import std;`.

## Honesty boundary

Unlike [`eigen`](eigen.md), the results here are **NOT exact**. They are
double-precision **approximations** accurate to roughly `tol`, produced by an
iterative floating-point algorithm. Two honest consequences:

- **Accuracy degrades on hard inputs.** A *defective* matrix (fewer eigenvectors
  than its dimension) or a *severely ill-conditioned* one can lose several digits
  in the returned eigenvalues. The returned numbers are the best the algorithm
  achieved, not a certified bound.
- **Non-convergence is reported, never faked.** If a deflation block fails to
  reach the tolerance within `max_iter` iterations, the function returns a
  `MathError` rather than a plausible-but-wrong partial spectrum (Rule 32).
  Because the core [`MathError`](core.md) enum has no dedicated *no-convergence*
  variant, the closest honest code — `MathError::not_implemented` — is used to
  mean "this input was not solved". It is never returned alongside a truncated or
  invented result.

Real eigenvalues come back with a **(near-)zero imaginary part**; the symmetric,
diagonal, and triangular branches set the imaginary part to **exactly** `0.0`.

## Structure-aware dispatch

`eigenvalues_qr` is a dispatcher, not a single algorithm. It first classifies the
matrix (O(n²), up to a relative tolerance `tol · max|aᵢⱼ|`) into a `MatrixKind`
and routes to the fastest algorithm that is **correct** for that shape. A
mis-classification can only cost speed, never correctness — the fallback is
always the general path.

| `MatrixKind` | Detection (within `tol`) | Algorithm | Spectrum | Complexity |
| :--- | :--- | :--- | :--- | :--- |
| `diagonal` | all off-diagonal ≈ 0 | read the diagonal | real | O(n) |
| `triangular` | all sub- **or** all super-diagonal ≈ 0 | read the diagonal | real | O(n) |
| `symmetric` | `A ≈ Aᵀ` | cyclic **Jacobi** rotations | **real** (exact-zero imag) | O(n³) per sweep |
| `skew_symmetric` | `A ≈ −Aᵀ`, zero diagonal | general real-Schur QR | purely imaginary ± pairs (+ a 0 for odd n) | O(n³) |
| `general` | none of the above | Hessenberg + **Francis** double-shift QR with 1×1/2×2 deflation | real + complex pairs | O(n³) |

Notes on the branches:

- **Symmetric ⇒ real.** A real symmetric matrix has an entirely real spectrum, so
  the Jacobi path returns eigenvalues with imaginary part **exactly** `0.0` — it
  can never manufacture a spurious imaginary component the way a general solver's
  rounding might. This is both faster and more accurate than the general QR.
- **Positive-definite** matrices are a special case of symmetric, so an SPD matrix
  takes the symmetric path automatically — no separate algorithm is needed. (Its
  eigenvalues additionally come out all `> 0`, which is one way to *test* definiteness.)
- **Skew-symmetric** matrices have purely imaginary eigenvalues in `± i·μ`
  conjugate pairs (plus one `0` when n is odd). They are routed through the
  general real-Schur QR, whose 2×2 blocks already yield those imaginary pairs
  correctly; no separate skew solver is used.
- **General / companion** — the classic EISPACK path: reduce to upper Hessenberg
  by Gaussian elimination with pivoting (`elmhes`), then run the Francis
  double-shift QR (`hqr`) with deflation. A trailing 1×1 block gives a real
  eigenvalue; a trailing 2×2 block is solved by the quadratic formula (a real
  pair when the discriminant is ≥ 0, else a complex-conjugate pair). Small
  subdiagonal entries are deflated when `|aₗ,ₗ₋₁| ≤ tol · (|aₗ₋₁,ₗ₋₁| + |aₗ,ₗ|)`,
  and Wilkinson-style exceptional shifts are injected periodically to break cycles.

**Scope.** The input is a **real** matrix (`std::span<const double>`), so for
real entries "symmetric" is the Hermitian case and "skew-symmetric" the
skew-Hermitian case. Genuinely **complex-Hermitian** input (a complex matrix) is
a future extension and is **not** handled here — it is not faked.

## API

Both entry points are free functions in `namespace nimblecas`, `[[nodiscard]]`,
returning `Result<std::vector<std::complex<double>>>`.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `eigenvalues_qr` | `auto eigenvalues_qr(std::span<const double> a, std::size_t n, double tol = 1e-12, std::size_t max_iter = 1000) -> Result<std::vector<std::complex<double>>>` | All eigenvalues of the `n×n` real matrix given **row-major** in `a` (length `n·n`), via the structure-aware dispatch above. `tol` sets the relative deflation/classification threshold; `max_iter` caps the QR sweeps per deflated block (and the Jacobi sweep count). A `0×0` matrix yields an empty vector. |
| `companion_eigenvalues` | `auto companion_eigenvalues(const RationalPoly& p, double tol = 1e-12, std::size_t max_iter = 1000) -> Result<std::vector<std::complex<double>>>` | All roots of `p`, as the eigenvalues of its companion matrix. Builds the real, `double` companion matrix of the **monic-normalized** `p` (each `Rational` coefficient converted via `double(num)/double(den)`) and calls `eigenvalues_qr`. This is the numeric polynomial root finder. |

The companion matrix used is the standard last-column form of the monic
`xᵈ + c_{d−1}xᵈ⁻¹ + … + c₀`: **ones on the subdiagonal** and the **negated
coefficients `−cᵢ` in the last column**. Its characteristic polynomial is exactly
the monic-normalized `p`, so its eigenvalues are the roots of `p`.

### Defaults

`tol = 1e-12`, `max_iter = 1000`. `max_iter = 1000` is far above the ~30 sweeps a
well-behaved block needs, leaving generous headroom before the honest
non-convergence error is returned.

## Error model

| Condition | Error |
| :--- | :--- |
| `eigenvalues_qr`: `a.size() != n·n` | `MathError::domain_error` |
| `eigenvalues_qr`: index arithmetic for `n` would overflow `std::ptrdiff_t` | `MathError::overflow` |
| `eigenvalues_qr`: a block does not converge within `max_iter` iterations | `MathError::not_implemented` (honest "not solved" signal) |
| `companion_eigenvalues`: `p` is the zero polynomial (root set undefined) | `MathError::domain_error` |
| `companion_eigenvalues`: `d·d` companion allocation would overflow | `MathError::overflow` |
| `companion_eigenvalues`: `p` is a non-zero **constant** | success, **empty** vector (no roots) |

A `0×0` matrix and a non-zero constant polynomial are **not** errors: both are an
empty spectrum.

## Worked example

```cpp
import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.numeigen;
using namespace nimblecas;

// A general real matrix, row-major. [[0,-1],[1,0]] is a rotation: eigenvalues ± i.
const std::vector<double> rot = {0, -1, 1, 0};
const auto ev = eigenvalues_qr(rot, 2).value();
// ev is {+i, -i} in some order (complex-conjugate pair from the 2x2 block).

// A symmetric matrix takes the Jacobi path and returns purely real eigenvalues.
const std::vector<double> sym = {2, 1, 1, 2};        // [[2,1],[1,2]]
const auto se = eigenvalues_qr(sym, 2).value();       // {1, 3}, imaginary part exactly 0

// Numeric polynomial roots via the companion matrix: x^3 - 2.
// coefficients ascending: c0=-2, c1=0, c2=0, c3=1.
auto r = [](std::int64_t v) { return Rational::from_int(v); };
const auto p = RationalPoly::from_coeffs({r(-2), r(0), r(0), r(1)});
const auto roots = companion_eigenvalues(p).value();  // cbrt(2) and a complex pair
// roots.size() == 3; one root ≈ 1.259921 with ~0 imaginary part, verified by |p(root)| ≈ 0.

// The zero polynomial has an undefined root set.
companion_eigenvalues(RationalPoly{}).error();        // MathError::domain_error
```

## See also

- [`nimblecas.eigen`](eigen.md) — the **exact** rational counterpart: rational
  eigenvalues, characteristic polynomial, and eigenspace bases over `Q`.
- [`nimblecas.analysis`](analysis.md) — `dominant_eigenvalue` (single dominant
  magnitude by power iteration) and other numeric/exact analysis facilities.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field and the
  `RationalPoly` whose roots `companion_eigenvalues` finds.
- [`nimblecas.solve`](solve.md) — the polynomial solver that consumes
  `companion_eigenvalues` for its numeric root-finding path.
- [Documentation hub](../Index.md)
```
