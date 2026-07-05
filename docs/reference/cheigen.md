# `nimblecas.cheigen` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/cheigen/cheigen.cppm`

The **complex** companion to [`numeigen`](numeigen.md). Where
[`numeigen`](numeigen.md) computes all eigenvalues of a **real** matrix, this
module computes all eigenvalues of an exact
[`ComplexMatrix`](cmatrix.md) — a matrix over the Gaussian rationals `Q + Qi`
— as `std::complex<double>`, exploiting the matrix's exact structure
(**Hermitian**, **skew-Hermitian**, **unitary**, **normal**) for both speed and
honesty. It introduces **no new eigensolver**: it reuses
[`numeigen`](numeigen.md)`::eigenvalues_qr` on a real embedding (ROADMAP §7.2
spectral).

```cpp
import nimblecas.cheigen;
```

Depends on [`core`](core.md) (`Result` / `MathError`), [`ratpoly`](ratpoly.md)
(`Rational`), [`complex`](complex.md) (`Complex`), [`cmatrix`](cmatrix.md)
(`ComplexMatrix` and its exact structural predicates), and
[`numeigen`](numeigen.md) (`eigenvalues_qr`, the reused real eigensolver).
Eigenvalues are returned as `std::complex<double>` from `import std;`.

## The real-embedding trick

An `n×n` complex matrix `M = A + iB` (with `A = Re(M)`, `B = Im(M)`, both real
`n×n`) maps to the `2n×2n` **real** matrix

```
R = [ A  -B ]
    [ B   A ]
```

whose spectrum is exactly the spectrum of `M` **together with its complex
conjugate**: every eigenvalue `λ` of `M` appears in `R` alongside `λ̄`. Feeding
`R` to [`numeigen`](numeigen.md)`::eigenvalues_qr` yields all `2n` numbers using
the existing, battle-tested real Jacobi / Francis QR — no complex QR is written.

`M`'s own `n` eigenvalues are then **recovered** by one of three disjoint paths,
selected on the **exact** matrix:

- **Hermitian `M`** — `A` is symmetric and `B` skew-symmetric, so `R` is
  **symmetric**. `numeigen` routes it through the **Jacobi** path, which returns
  real eigenvalues with imaginary part **exactly `0.0`**, each **doubled**. The
  doubled spectrum is sorted and collapsed pairwise back to `n` reals.
- **skew-Hermitian `M`** — recovered **exactly**, sidestepping the embedding
  entirely. `iM` is Hermitian when `M` is skew-Hermitian
  (`(iM)† = −i M† = −i(−M) = iM`), so the Hermitian path applied to `iM` yields
  its real eigenvalues `μₖ`, and `M`'s eigenvalues are the purely imaginary
  `λₖ = −i·μₖ` (real part **exactly `0.0`**). E.g. `diag(i, −i)` maps to the
  Hermitian `diag(1, −1)`, giving `μ = {1, −1}` and `λ = {−i, i}`.
- **General `M`** (neither of the above; includes unitary and normal) — `R` is
  real, so its spectrum is closed under conjugation
  (`spec(R) = spec(M) ⊎ conj(spec(M))`). `M`'s own half is the subset that
  annihilates `M`'s characteristic polynomial, so the candidates with normalized
  residual `|det(λI − M)| ≈ 0` are selected (see the honesty boundary for when
  this is refused). Every returned `λ` satisfies `det(λI − M) ≈ 0`.

## Honesty boundary

The eigenvalues of a general complex matrix are irrational / complex and have
**no exact Gaussian-rational representation**, so — exactly like
[`numeigen`](numeigen.md) — the results here are **NUMERIC**
(`std::complex<double>`) approximations, **never** presented as exact. The single
point where exactness is lost is the entrywise projection of each Gaussian-rational
`Complex` to a `double` when building the embedding `R`; everything downstream is
floating point.

**Structure is decided on the EXACT matrix, before any float appears**, and drives
both correctness and the honesty of the output:

| Exact structure (checked on `ComplexMatrix`) | Spectrum guarantee | What the output does |
| :--- | :--- | :--- |
| **Hermitian** (`is_hermitian`) | real | imaginary part is **exactly `0.0`** (not "≈ 0") |
| **skew-Hermitian** (`is_skew_hermitian`) | purely imaginary | real part is **exactly `0.0`** — recovered exactly via the `iM`-Hermitian reduction |
| **unitary** (`is_unitary`, general path) | on the unit circle `|λ| = 1` | magnitudes are the numeric ones produced; documented, **not** faked |
| **normal / general** | complex | each `λ` satisfies `det(λI − M) ≈ 0`, **or** `not_implemented` (below) |

A component is emitted as exactly zero **only** because the structure was *proven*
on the exact entries — never merely observed to be small in the float result. A
block that fails to converge surfaces as an honest `MathError::not_implemented`
(inherited from [`numeigen`](numeigen.md)), never a partial or invented spectrum.

**The general path refuses conjugate-closed spectra — it never guesses.** The real
embedding `R` genuinely *loses information* when `M`'s exact spectrum is closed
under conjugation: `diag(i, i)` and `diag(i, −i)` produce the **same** R-spectrum
`{i, i, −i, −i}`, so `M` cannot be recovered from `R`'s eigenvalues alone. This
case — which includes **every real matrix that has complex-conjugate eigenvalues**
(e.g. `[[1, −2], [2, 1]]` with spectrum `{1 ± 2i}`) — is detected (the residual
selection has size `≠ n`, or a selected non-real `λ` has its conjugate also
selected) and returns `MathError::not_implemented`. It **never** returns a wrong
multiset such as `{1+2i, 1+2i}`. For a **real** matrix with complex eigenvalues,
call [`numeigen`](numeigen.md)`::eigenvalues_qr` directly — it solves the real
problem without the embedding's conjugate ambiguity. (Purely imaginary conjugate
pairs from a *skew-Hermitian* matrix, such as `{i, −i}`, are **not** affected: they
are handled exactly by the dedicated skew-Hermitian path above.)

## API

Both entry points are free functions in `namespace nimblecas`, `[[nodiscard]]`.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `hermitian_eigenvalues` | `auto hermitian_eigenvalues(const ComplexMatrix& m, double tol = 1e-12, std::size_t max_iter = 1000) -> Result<std::vector<double>>` | Eigenvalues of a **Hermitian** matrix, returned **real and ascending**. The matrix is verified Hermitian on its exact entries first; a non-Hermitian matrix is a `domain_error` (its spectrum is not real — use `eigenvalues`). |
| `eigenvalues` | `auto eigenvalues(const ComplexMatrix& m, double tol = 1e-12, std::size_t max_iter = 1000) -> Result<std::vector<std::complex<double>>>` | All eigenvalues of a general complex matrix, with exact-structure dispatch: Hermitian → real (`0` imaginary); skew-Hermitian → imaginary (`0` real, via the `iM`-Hermitian reduction); else the embedding + residual-selection path with `det(λI − M) ≈ 0`, which returns `not_implemented` for a conjugate-closed spectrum. |

`tol` and `max_iter` are forwarded verbatim to
[`numeigen`](numeigen.md)`::eigenvalues_qr` (applied to the `2n×2n` embedding):
`tol` is the relative deflation / classification threshold and `max_iter` caps the
QR / Jacobi sweeps per deflated block. A `0×0` matrix yields an empty vector.

## Error model

| Condition | Error |
| :--- | :--- |
| `m` is non-square | `MathError::domain_error` |
| `hermitian_eigenvalues`: `m` is not Hermitian | `MathError::domain_error` |
| `eigenvalues`, general path: `m`'s spectrum is closed under conjugation (unrecoverable from the real embedding — includes every real matrix with complex eigenvalues) | `MathError::not_implemented` |
| a deflation block does not converge within `max_iter` | `MathError::not_implemented` (honest "not solved") |
| index arithmetic for the `2n×2n` embedding would overflow `std::size_t` | `MathError::overflow` |
| entry-operation overflow while probing structure (e.g. `is_hermitian`) | propagated from [`cmatrix`](cmatrix.md) (`overflow`) |

A `0×0` matrix is **not** an error: its spectrum is the empty vector.

## Worked example

```cpp
import std;
import nimblecas.ratpoly;
import nimblecas.complex;
import nimblecas.cmatrix;
import nimblecas.cheigen;
using namespace nimblecas;

auto cx = [](std::int64_t re, std::int64_t im) {
    return Complex::make(Rational::from_int(re), Rational::from_int(im));
};

// Hermitian [[2, i], [-i, 2]] has the REAL spectrum {1, 3}.
const auto H = ComplexMatrix::from_rows({{cx(2, 0), cx(0, 1)},
                                         {cx(0, -1), cx(2, 0)}}).value();
const auto hr = hermitian_eigenvalues(H).value();   // {1.0, 3.0}, ascending
const auto hc = eigenvalues(H).value();              // {1, 3} with imaginary part exactly 0

// skew-Hermitian diag(i, -i): purely imaginary CONJUGATE pair {i, -i}, real part exactly 0.
// Recovered exactly via the iM-Hermitian reduction (iM = diag(1, -1)); the general embedding
// alone could not distinguish this from diag(i, i).
const auto S = ComplexMatrix::from_rows({{cx(0, 1), cx(0, 0)},
                                         {cx(0, 0), cx(0, -1)}}).value();
const auto se = eigenvalues(S).value();              // {i, -i}

// General upper-triangular [[1+i, 5], [0, 2-3i]]: eigenvalues are the diagonal {1+i, 2-3i}.
const auto G = ComplexMatrix::from_rows({{cx(1, 1), cx(5, 0)},
                                         {cx(0, 0), cx(2, -3)}}).value();
const auto ge = eigenvalues(G).value();              // {1+i, 2-3i}, each with det(λI−G) ≈ 0

// A real matrix with a complex-conjugate spectrum is unrecoverable from the real embedding:
// eigenvalues() refuses rather than guess. Use numeigen::eigenvalues_qr for real matrices.
const auto R = ComplexMatrix::from_rows({{cx(1, 0), cx(-2, 0)},
                                         {cx(2, 0), cx(1, 0)}}).value();  // spectrum {1 ± 2i}
eigenvalues(R).error();                              // MathError::not_implemented

// Asking for the real Hermitian spectrum of a non-Hermitian matrix is a category error.
hermitian_eigenvalues(G).error();                    // MathError::domain_error
```

## See also

- [`nimblecas.numeigen`](numeigen.md) — the **real** eigensolver this module
  reuses on the `2n×2n` embedding (structure-aware QR / Jacobi).
- [`nimblecas.cmatrix`](cmatrix.md) — the exact `ComplexMatrix` and its
  `is_hermitian` / `is_skew_hermitian` / `is_unitary` / `is_normal` predicates
  that drive the honest structural dispatch.
- [`nimblecas.complex`](complex.md) — the exact Gaussian-rational `Complex` entry
  type.
- [`nimblecas.eigen`](eigen.md) — the **exact** rational eigen-facilities for real
  matrices (rational spectrum, characteristic polynomial, eigenspaces).
- [Documentation hub](../Index.md)
```
