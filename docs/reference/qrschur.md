# `nimblecas.qrschur` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/qrschur/qrschur.cppm`

QR decomposition and real **Schur** (quasi-triangular) form. This module is deliberately
split along its honesty boundary, because a truly *orthonormal* QR of a rational matrix is,
in general, **not representable over ℚ at all** — the column 2-norms are irrational. Rather
than fake an "exact" orthonormal factorisation, `qrschur` offers three distinct products:

1. an **exact** Gram-Schmidt *orthogonal* (not orthonormal) decomposition over ℚ,
2. a **numeric** Householder QR with a genuinely orthonormal `Q`, and
3. a **numeric** real Schur form `A = Q·T·Qᵀ`,

each labelled precisely for what it is.

```cpp
import nimblecas.qrschur;
```

Depends on [`core`](core.md) (`Result` / `MathError`), [`matrix`](matrix.md) (the exact
rational `Matrix`), [`ratpoly`](ratpoly.md) (`Rational`), and
[`numeigen`](numeigen.md) (its `eigenvalues_qr`, used to cross-check the Schur spectrum;
the Schur iteration itself is the same Francis double-shift algorithm family). Numeric
results use `std::span<const double>` / `std::vector<double>` and `std::complex<double>`
from `import std;`.

## Honesty boundary

| Product | Regime | `Q` | `R` / `T` | Reconstruction |
| :--- | :--- | :--- | :--- | :--- |
| `exact_orthogonal_qr` | **EXACT over ℚ** | columns **orthogonal**, not unit (`QᵀQ` diagonal) | `R` upper-triangular, **unit diagonal** | `Q·R == A` **identically** |
| `numeric_qr` | **NUMERIC** (double) | **orthonormal** (`QᵀQ = I` to rounding) | `R` upper-triangular | `Q·R ≈ A` (≈ machine ε) |
| `real_schur` | **NUMERIC** (double) | **orthogonal** | `T` quasi-upper-triangular (1×1 / 2×2 blocks) | `Q·T·Qᵀ ≈ A` (≈ machine ε) |

Three honest consequences, each enforced in code (Rule 32):

- **The exact path is orthogonal, never orthonormal.** Classical Gram-Schmidt keeps every
  entry of `Q` and `R` an exact `Rational`. Its `Q` has *mutually orthogonal* columns, so
  `Qᵀ·Q` is **diagonal** — the diagonal entries are the exact rational squared pseudo-norms
  `⟨qₖ,qₖ⟩`, **not** 1. Unit-normalising would require the irrational `√⟨qₖ,qₖ⟩` and leave
  ℚ, so it is not attempted. The reconstruction `Q·R == A` holds *exactly*, verifiable with
  `Matrix::multiply` and `Matrix::operator==`.
- **Rank deficiency is an honest error on the exact path.** A column lying in the span of
  the earlier ones gives `⟨qₖ,qₖ⟩ = 0`; the algorithm would divide by that zero, so it
  returns `MathError::domain_error` instead of a fabricated factor. (Every wide `m<n` matrix
  is rank-deficient in this sense and so is rejected.) The **numeric** QR does *not* treat
  rank deficiency as an error — a zero pivot column simply leaves a zero on `R`'s diagonal
  while `Q` stays orthonormal and `Q·R ≈ A`.
- **Schur non-convergence is reported, never faked.** A deflation block that fails to reach
  the tolerance within `max_iter` sweeps returns `MathError::not_implemented` — the closest
  honest "not solved" signal, exactly as in [`numeigen`](numeigen.md) — never a partial or
  garbage `T`.

The numeric results are double approximations accurate to roughly machine epsilon; they are
**never** presented as exact. Every one is independently **residual-checkable**
(`qr_residual`, `schur_residual`, `orthonormality_defect`).

## Algorithms

- **`exact_orthogonal_qr`** — classical Gram-Schmidt over ℚ. For each column `aₖ`,
  `qₖ = aₖ − Σ_{j<k} rⱼₖ·qⱼ` with `rⱼₖ = ⟨aₖ,qⱼ⟩ / ⟨qⱼ,qⱼ⟩` and `rₖₖ = 1`; all arithmetic
  is exact checked `Rational`, so overflow of an `int64` numerator/denominator surfaces as
  `MathError::overflow`.
- **`numeric_qr`** — Householder QR. Each step reflects the sub-column below the pivot to a
  multiple of `e₁` (with the stability sign `−sign(x₀)·‖x‖`); reflections are applied to `R`
  from the left and accumulated into `Q` from the right, so `Q` is the full `m×m` orthonormal
  factor. The strict lower triangle of `R` is then forced to exact zero.
- **`real_schur`** — orthogonal Householder reduction to upper Hessenberg (**orthes**,
  accumulated into `Q` via **ortran**), then the Francis double-shift QR iteration with
  deflation (the **hqr2** iteration, stopped at the Schur form — the eigenvector
  back-substitution of hqr2 is **not** performed). This is the same double-shift family as
  [`numeigen`](numeigen.md)'s `hqr`, but with the orthogonal transformations accumulated;
  `numeigen`'s Gaussian `elmhes` is eigenvalue-only and **cannot** yield an orthogonal `Q`,
  which is why an orthogonal reduction is used here. Wilkinson-style ad-hoc exceptional
  shifts break cycles. Small subdiagonals are deflated when `|aₗ,ₗ₋₁| < tol·(|aₗ₋₁,ₗ₋₁| +
  |aₗ,ₗ|)`.

## API

All entry points are free functions in `namespace nimblecas`, `[[nodiscard]]`.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `exact_orthogonal_qr` | `auto exact_orthogonal_qr(const Matrix& a) -> Result<ExactOrthogonalQr>` | Exact Gram-Schmidt `A = Q·R` over ℚ. `q` is `m×n` with `QᵀQ` diagonal; `r` is `n×n` upper-triangular unit-diagonal; `Q·R == A` exactly. |
| `numeric_qr` | `auto numeric_qr(std::span<const double> a, std::size_t rows, std::size_t cols) -> Result<NumericQr>` | Householder QR of the `m×n` row-major real `a`. `q` is `m×m` orthonormal, `r` is `m×n` upper-triangular, `Q·R ≈ A`. |
| `real_schur` | `auto real_schur(std::span<const double> a, std::size_t n, double tol = 1e-12, std::size_t max_iter = 1000) -> Result<NumericSchur>` | Real Schur form `A = Q·T·Qᵀ` of the `n×n` row-major real `a`. `q` orthogonal, `t` quasi-upper-triangular. |
| `schur_eigenvalues` | `auto schur_eigenvalues(const NumericSchur& s, double tol = 1e-9) -> Result<std::vector<std::complex<double>>>` | Eigenvalues read off `T`'s 1×1 / 2×2 blocks. Agrees with `numeigen::eigenvalues_qr`. |
| `qr_residual` | `auto qr_residual(const NumericQr& d, std::span<const double> a) -> Result<double>` | Frobenius residual `‖Q·R − A‖_F`. |
| `schur_residual` | `auto schur_residual(const NumericSchur& s, std::span<const double> a) -> Result<double>` | Frobenius residual `‖Q·T·Qᵀ − A‖_F`. |
| `orthonormality_defect` | `auto orthonormality_defect(std::span<const double> q, std::size_t rows, std::size_t cols) -> Result<double>` | Frobenius defect `‖QᵀQ − I‖_F` (0 iff exactly orthonormal). |

### Result structs

```cpp
struct ExactOrthogonalQr { Matrix q;   // m×n, QᵀQ diagonal (exact over ℚ)
                           Matrix r; };  // n×n, upper-triangular, unit diagonal
struct NumericQr { std::size_t rows, cols;
                   std::vector<double> q;   // m×m orthonormal, row-major
                   std::vector<double> r; };  // m×n upper-triangular, row-major
struct NumericSchur { std::size_t n;
                      std::vector<double> q;   // n×n orthogonal Schur vectors
                      std::vector<double> t; };  // n×n quasi-upper-triangular real Schur form
```

## Error model

| Condition | Error |
| :--- | :--- |
| `exact_orthogonal_qr`: `A` rank-deficient (zero pseudo-norm), incl. every wide `m<n` | `MathError::domain_error` |
| `exact_orthogonal_qr`: `int64` entry arithmetic overflows | `MathError::overflow` |
| `numeric_qr` / `real_schur`: `a.size()` ≠ product of the dimensions | `MathError::domain_error` |
| `numeric_qr` / `real_schur`: the dimension product would wrap `std::size_t` | `MathError::overflow` |
| `real_schur`: a block does not converge within `max_iter` sweeps | `MathError::not_implemented` (honest "not solved") |
| `schur_eigenvalues` / residual helpers: struct/span size mismatch | `MathError::domain_error` |

A `0×0` matrix is **not** an error on any path: it is an empty decomposition.

## Worked examples

```cpp
import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.qrschur;
using namespace nimblecas;

auto ri = [](std::int64_t v) { return Rational::from_int(v); };

// (1) EXACT Gram-Schmidt over Q. A = [[2,1],[1,1]].
//     q1=(2,1), <q1,q1>=5; r12 = <a2,q1>/5 = 3/5; q2 = a2 - (3/5)q1 = (-1/5, 2/5).
const auto A = Matrix::from_rows({{ri(2), ri(1)}, {ri(1), ri(1)}}).value();
const auto qr = exact_orthogonal_qr(A).value();
// qr.r.at(0,1) == 3/5, qr.r diagonal == 1; qr.q.multiply(qr.r).value() == A  (exactly).
// (qr.q)^T (qr.q) is diagonal(5, 1/5) — orthogonal, NOT orthonormal.

// Rank-deficient columns => honest error (never a bogus factor).
const auto dep = Matrix::from_rows({{ri(1), ri(2)}, {ri(2), ri(4)}}).value();
exact_orthogonal_qr(dep).error();  // MathError::domain_error

// (2) NUMERIC Householder QR. Q orthonormal, R upper-triangular, Q·R ≈ A.
const std::vector<double> a = {12, -51, 4, 6, 167, -68, -4, 24, -41};
const auto nq = numeric_qr(a, 3, 3).value();
qr_residual(nq, a).value();               // ‖Q·R − A‖_F  ~ 1e-13
orthonormality_defect(nq.q, 3, 3).value();  // ‖QᵀQ − I‖_F ~ 1e-15

// (3) NUMERIC real Schur form. [[2,-1,0],[1,2,0],[0,0,3]] has eigenvalues 3 and 2 ± i.
const std::vector<double> g = {2, -1, 0, 1, 2, 0, 0, 0, 3};
const auto s = real_schur(g, 3).value();
schur_residual(s, g).value();       // ‖Q·T·Qᵀ − A‖_F ~ 1e-14
schur_eigenvalues(s).value();       // {3, 2+i, 2−i} — matches numeigen::eigenvalues_qr(g,3)
```

## See also

- [`nimblecas.numeigen`](numeigen.md) — all numeric eigenvalues by the QR algorithm; its
  `eigenvalues_qr` is the reference against which `schur_eigenvalues` is cross-checked.
- [`nimblecas.matdecomp`](matdecomp.md) — the **exact** LU decomposition `P·A = L·U` over ℚ
  and the structural `is_*` predicates.
- [`nimblecas.matrix`](matrix.md) — the exact rational `Matrix` (`multiply`, `transpose`,
  `operator==`) used to verify `Q·R == A` on the exact path.
- [`nimblecas.eigen`](eigen.md) — the exact rational spectrum (rational eigenvalues,
  characteristic polynomial, eigenspaces over ℚ).
- [Documentation hub](../Index.md)
```
