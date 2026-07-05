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

`M`'s own `n` eigenvalues are then **recovered** from `R`'s `2n`:

- **Hermitian `M`** — `A` is symmetric and `B` skew-symmetric, so `R` is
  **symmetric**. `numeigen` routes it through the **Jacobi** path, which returns
  real eigenvalues with imaginary part **exactly `0.0`**, each **doubled**. The
  doubled spectrum is sorted and collapsed pairwise back to `n` reals.
- **General `M`** — `R` is real, so its spectrum is closed under conjugation
  (`M`'s spectrum ⊎ `conj(M`'s spectrum`)`). Each candidate is paired with its
  conjugate partner, and from each pair the representative that best annihilates
  `M`'s characteristic polynomial — the one minimizing `|det(λI − M)|`, evaluated
  in complex-double arithmetic on `M` itself — is kept. This is what guarantees
  every returned `λ` satisfies `det(λI − M) ≈ 0` (rather than `det(λI − M̄) ≈ 0`).

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
| **Hermitian** (`is_hermitian`) | real | imaginary part set to **exactly `0.0`** (not "≈ 0") |
| **skew-Hermitian** (`is_skew_hermitian`) | purely imaginary | real part snapped to **exactly `0.0`** |
| **unitary** (`is_unitary`) | on the unit circle `|λ| = 1` | magnitudes are the numeric ones produced; documented, **not** faked |
| **normal / general** | complex | each `λ` satisfies `det(λI − M) ≈ 0` |

Snapping a component to exact zero is legitimate **only** because the structure was
*proven* on the exact entries — never merely observed to be small in the float
result. A block that fails to converge surfaces as an honest
`MathError::not_implemented` (inherited from [`numeigen`](numeigen.md)), never a
partial or invented spectrum.

**The one genuinely ambiguous case.** If `M`'s *exact* spectrum is itself closed
under conjugation (a real spectrum, or one made of `±` conjugate pairs), then for
a given pair *both* representatives are true eigenvalues of `M`; the recovery may
return either, but the returned set still satisfies the characteristic equation.
This is the honest limit of reconstructing a complex spectrum from a real
embedding; it is documented rather than hidden.

## API

Both entry points are free functions in `namespace nimblecas`, `[[nodiscard]]`.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `hermitian_eigenvalues` | `auto hermitian_eigenvalues(const ComplexMatrix& m, double tol = 1e-12, std::size_t max_iter = 1000) -> Result<std::vector<double>>` | Eigenvalues of a **Hermitian** matrix, returned **real and ascending**. The matrix is verified Hermitian on its exact entries first; a non-Hermitian matrix is a `domain_error` (its spectrum is not real — use `eigenvalues`). |
| `eigenvalues` | `auto eigenvalues(const ComplexMatrix& m, double tol = 1e-12, std::size_t max_iter = 1000) -> Result<std::vector<std::complex<double>>>` | All eigenvalues of a general complex matrix, with exact-structure dispatch: Hermitian → real (`0` imaginary), skew-Hermitian → imaginary (`0` real), else the embedding + conjugate-recovery path with `det(λI − M) ≈ 0`. |

`tol` and `max_iter` are forwarded verbatim to
[`numeigen`](numeigen.md)`::eigenvalues_qr` (applied to the `2n×2n` embedding):
`tol` is the relative deflation / classification threshold and `max_iter` caps the
QR / Jacobi sweeps per deflated block. A `0×0` matrix yields an empty vector.

## Error model

| Condition | Error |
| :--- | :--- |
| `m` is non-square | `MathError::domain_error` |
| `hermitian_eigenvalues`: `m` is not Hermitian | `MathError::domain_error` |
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

// skew-Hermitian diag(2i, i): purely imaginary spectrum {2i, i}, real part exactly 0.
const auto S = ComplexMatrix::from_rows({{cx(0, 2), cx(0, 0)},
                                         {cx(0, 0), cx(0, 1)}}).value();
const auto se = eigenvalues(S).value();              // {2i, i}

// General upper-triangular [[1+i, 5], [0, 2-3i]]: eigenvalues are the diagonal {1+i, 2-3i}.
const auto G = ComplexMatrix::from_rows({{cx(1, 1), cx(5, 0)},
                                         {cx(0, 0), cx(2, -3)}}).value();
const auto ge = eigenvalues(G).value();              // {1+i, 2-3i}, each with det(λI−G) ≈ 0

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
