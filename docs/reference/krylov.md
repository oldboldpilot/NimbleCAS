# `nimblecas.krylov` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/krylov/krylov.cppm`

Krylov subspace methods for iterative linear algebra (ROADMAP §7.2). This module
is deliberately split into two halves, and **the split is mathematically
principled, not cosmetic**. One half is **EXACT over the rationals `Q`**: the
Conjugate Gradient solver, the power Krylov basis, and the unnormalised rational
Arnoldi / Lanczos factorisations run entirely on inner products, scalar
divisions, and axpy updates — **no square root ever enters**, so every
intermediate stays an exact reduced fraction and the results are bit-for-bit
correct. The other half is **NUMERICAL over `double`**: GMRES, MINRES, and
BiCGSTAB solve general/large/indefinite systems, and `lanczos_ritz` /
`arnoldi_hessenberg` estimate eigenvalues — these carry round-off, convergence
is **not** guaranteed, and the Ritz values are **approximations**, not exact
eigenvalues. If a routine lives in the exact half its result is exact; if it
lives in the numerical half its result is an approximation. This boundary is the
contract, and it is stated on every entry below.

```cpp
import nimblecas.krylov;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md) (`Rational`), and
[`matrix`](matrix.md) (exact dense linear algebra over `Q`).

## The exact / numerical boundary

The key mathematical fact is that **Conjugate Gradient contains no square root**.
It is built from `<.,.>` inner products, scalar divisions, and axpy, so over `Q`
every value stays a reduced fraction and — for an `n × n` SPD system — it reaches
the **exact** solution in at most `n` iterations (finite termination is a
theorem, not a tolerance). Likewise the rational Arnoldi / Lanczos here
orthogonalise by scaling with the inner-product values `<q_i, q_i>` (a rational),
**never** by the norm `sqrt(<q_i, q_i>)` (an irrational). The resulting basis is
exactly orthogonal over `Q` but its vectors are **not** unit vectors, and the
projected operator carries a **unit subdiagonal** rather than the symmetric
tridiagonal of the classical process. The classical **normalised** Arnoldi /
Lanczos — orthonormal basis, symmetric tridiagonal — needs `sqrt` and is
therefore only available **numerically** (`lanczos_ritz` / `arnoldi_hessenberg`).

## Exported types

| Type | Half | Fields |
| :--- | :--- | :--- |
| `ExactCGResult` | EXACT | `Matrix solution` (the exact `n × 1` `Rational` solution), `std::size_t steps` (CG iterations performed, `<= n`). |
| `RationalArnoldi` | EXACT | `std::vector<std::vector<Rational>> basis` (mutually orthogonal, **unnormalised** vectors), `std::vector<Rational> gram_diagonal` (the scalings `<q_i, q_i>`, all `> 0`), `Matrix hessenberg` (`k × k` exact upper-Hessenberg with **unit subdiagonal**), `bool breakdown` (an invariant subspace was reached, i.e. `A·basis == basis·hessenberg` holds exactly). |
| `RationalLanczos` | EXACT | `std::vector<std::vector<Rational>> basis`, `std::vector<Rational> alpha` (the `k` diagonal entries `H[j][j]`), `std::vector<Rational> superdiagonal` (the `k-1` entries `H[j-1][j]`), `Matrix tridiagonal` (full `k × k`, subdiagonal is the unit sequence — **not** symmetric), `bool breakdown`. |
| `MatVec` | NUMERICAL | `std::function<void(std::span<const double> x, std::span<double> y)>` — a matrix-free operator `y <- A x`. The iterative solvers never see the matrix, only its action. |
| `IterativeResult` | NUMERICAL | `std::vector<double> x`, `std::size_t iterations`, `double residual` (the **true** `‖b − A x‖₂` recomputed at return), `bool converged` (`residual <= tol·‖b‖`). |
| `DoubleHessenberg` | NUMERICAL | `std::vector<double> h` (`dim × dim` row-major upper-Hessenberg `H = VᵀAV`), `std::size_t dim`. Its eigenvalues are the Arnoldi Ritz values. |

## EXACT over `Q` — API

All signatures live in namespace `nimblecas` and are `[[nodiscard]]`.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `conjugate_gradient_steps` | `auto conjugate_gradient_steps(const Matrix& A, const Matrix& b) -> Result<ExactCGResult>` | Solve `A x = b` **exactly** for a symmetric positive-definite `Rational` matrix `A` and an `n × 1` `Rational` column `b`, returning the exact solution **and** the CG step count (`<= n`). Uses only inner products, divisions, and axpy — no `sqrt`, so arithmetic stays in `Q`. |
| `conjugate_gradient` | `auto conjugate_gradient(const Matrix& A, const Matrix& b) -> Result<Matrix>` | Convenience wrapper returning just the exact `Rational` solution vector. |
| `krylov_basis` | `auto krylov_basis(const Matrix& A, std::span<const Rational> b, std::size_t m) -> Result<std::vector<std::vector<Rational>>>` | The exact power Krylov basis `{b, Ab, A²b, …, A^{m-1}b}` — the first `m` Krylov vectors. `m == 0` yields an empty result. |
| `arnoldi_rational` | `auto arnoldi_rational(const Matrix& A, std::span<const Rational> b, std::size_t m) -> Result<RationalArnoldi>` | Build up to `min(m, n)` mutually orthogonal (over `Q`) but **unnormalised** Krylov vectors and the exact upper-Hessenberg projected operator. `m == 0` yields an empty result. |
| `lanczos_rational` | `auto lanczos_rational(const Matrix& A, std::span<const Rational> b, std::size_t m) -> Result<RationalLanczos>` | For a **symmetric** `Rational` `A`, the Hessenberg projection collapses to tridiagonal; returns its diagonal (`alpha`) and superdiagonal exactly (the subdiagonal being the unit sequence). Requires symmetry (`domain_error` otherwise). |

CG rejects with `domain_error` when `A` is not square, not symmetric, `b` has the
wrong shape, or a breakdown proves `A` is **not** positive definite — a direction
with `pᵀA p <= 0`, or failure to terminate within `n` steps. This is how CG here
doubles as an exact SPD test: a non-positive `pᵀA p` is definite proof that `A`
is not SPD.

## NUMERICAL over `double` — API

All signatures live in namespace `nimblecas` and are `[[nodiscard]]`. In every
solver `tol` is **relative to `‖b‖`**, running out of iterations is reported as
`converged == false` (a legitimate outcome, **not** an error), and only an empty
system is a `domain_error`.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `dense_matvec` | `auto dense_matvec(std::span<const double> mat, std::size_t n) -> MatVec` | Wrap an `n × n` row-major dense buffer as a `MatVec`. The buffer is **copied** into the returned closure, so it stays valid independently of the caller's storage. |
| `gmres` | `auto gmres(const MatVec& A, std::span<const double> b, double tol = 1e-10, std::size_t max_iter = 1000, std::size_t restart = 30) -> Result<IterativeResult>` | Restarted GMRES with Givens rotations, for general (possibly non-symmetric) systems. |
| `minres` | `auto minres(const MatVec& A, std::span<const double> b, double tol = 1e-10, std::size_t max_iter = 1000) -> Result<IterativeResult>` | Paige-Saunders MINRES, for symmetric (possibly indefinite) systems. |
| `bicgstab` | `auto bicgstab(const MatVec& A, std::span<const double> b, double tol = 1e-10, std::size_t max_iter = 1000) -> Result<IterativeResult>` | BiCGSTAB, for general non-symmetric systems. Breakdown (`rho` or `omega` collapsing to zero) stops the iteration and is reported as `converged == false`, not as an error. |
| `lanczos_ritz` | `auto lanczos_ritz(const MatVec& A, std::span<const double> v0, std::size_t m) -> Result<std::vector<double>>` | Run `min(m, n)` steps of the classical **normalised** Lanczos on a symmetric operator and return the eigenvalues of the resulting tridiagonal — the Ritz values, sorted ascending. These are **approximations** of a subset of `A`'s spectrum (extreme eigenvalues emerge first), **not** exact eigenvalues. |
| `arnoldi_hessenberg` | `auto arnoldi_hessenberg(const MatVec& A, std::span<const double> v0, std::size_t m) -> Result<DoubleHessenberg>` | Run `min(m, n)` steps of the classical **orthonormal** Arnoldi and return the `dim × dim` upper-Hessenberg `H = VᵀAV`. Its eigenvalues are the Arnoldi Ritz values — spectral **approximations** whose extraction needs a general (non-symmetric) eigensolver, out of this module's scope. |

## Error model

| Condition | Error |
| :--- | :--- |
| `A` not square, or `b` wrong shape/dimension (any exact routine) | `MathError::domain_error` |
| `conjugate_gradient` / `conjugate_gradient_steps` on a non-symmetric `A` | `MathError::domain_error` |
| CG breakdown — a direction with `pᵀA p <= 0`, or no exact solution within `n` steps (proves `A` is not SPD) | `MathError::domain_error` |
| `lanczos_rational` on a non-symmetric `A` | `MathError::domain_error` |
| `gmres` / `minres` / `bicgstab` / `lanczos_ritz` / `arnoldi_hessenberg` on an empty system (`b`/`v0` size `0`, or `m == 0` for the estimators) | `MathError::domain_error` |
| `lanczos_ritz` / `arnoldi_hessenberg` on a zero starting vector (`‖v0‖ == 0`) | `MathError::domain_error` |
| `lanczos_ritz` when the tridiagonal QL iteration stalls (does not converge in 50 sweeps) | `MathError::not_implemented` |
| An `int64` numerator or denominator computation wraps in an exact routine | `MathError::overflow` |

Running out of iterations is **not** an error: `gmres` / `minres` / `bicgstab`
return `converged == false` and the true residual. A zero right-hand side `b`
(or a zero starting vector for the rational factorisations) is a degenerate input
reported via the `breakdown` flag, not an error.

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.matrix;
import nimblecas.ratpoly;
import nimblecas.krylov;
using namespace nimblecas;

auto ri  = [](std::int64_t v) { return Rational::from_int(v); };
auto rat = [](std::int64_t n, std::int64_t d) { return Rational::make(n, d).value(); };
auto mat = [](std::vector<std::vector<Rational>> r) {
    return Matrix::from_rows(std::move(r)).value();
};
auto col = [&](std::vector<std::int64_t> e) {
    std::vector<std::vector<Rational>> r;
    for (auto v : e) r.push_back({ri(v)});
    return mat(std::move(r));
};

// ---- EXACT: Conjugate Gradient on an SPD Rational system ----
// A = [[4,1],[1,3]] is SPD; b = [1,2]. Exact solution x = [1/11, 7/11].
auto A = mat({{ri(4), ri(1)}, {ri(1), ri(3)}});
auto b = col({1, 2});
auto x = conjugate_gradient(A, b).value();     // [[1/11], [7/11]]  — exact
A.multiply(x).value() == b;                     // true  (A*x == b exactly)
x == A.solve(b).value();                         // true  (agrees with the direct exact solver)

// Step count is bounded by n: a 3x3 SPD system terminates in <= 3 steps.
auto A3 = mat({{ri(4), ri(1), ri(0)},
               {ri(1), ri(3), ri(1)},
               {ri(0), ri(1), ri(2)}});
auto r3 = conjugate_gradient_steps(A3, col({1, 2, 3})).value();
r3.steps <= 3;                                   // true

// Non-SPD input is rejected, and CG doubles as an exact SPD test.
auto indef = mat({{ri(1), ri(2)}, {ri(2), ri(1)}});    // symmetric, indefinite
conjugate_gradient(indef, col({1, 0})).error();        // MathError::domain_error
auto nonsym = mat({{ri(1), ri(2)}, {ri(3), ri(4)}});
conjugate_gradient(nonsym, col({1, 1})).error();       // MathError::domain_error

// ---- EXACT: the power Krylov basis ----
// A = diag(2,3), b = [1,1]. {b, Ab, A^2 b} = {[1,1], [2,3], [4,9]}.
auto Ad = mat({{ri(2), ri(0)}, {ri(0), ri(3)}});
std::vector<Rational> bv{ri(1), ri(1)};
auto basis = krylov_basis(Ad, bv, 3).value();
basis[0] == std::vector<Rational>{ri(1), ri(1)};       // v0 == b
basis[1] == std::vector<Rational>{ri(2), ri(3)};       // v1 == A b
basis[2] == std::vector<Rational>{ri(4), ri(9)};       // v2 == A^2 b

// ---- EXACT: unnormalised rational Lanczos / Arnoldi ----
// Symmetric A: alpha_0 = <A q0, q0>/<q0,q0> = A[0][0] = 2; unit subdiagonal.
auto As = mat({{ri(2), ri(1)}, {ri(1), ri(2)}});
auto lan = lanczos_rational(As, std::vector<Rational>{ri(1), ri(0)}, 2).value();
lan.alpha.front() == ri(2);                            // alpha_0 == 2
lan.tridiagonal.at(1, 0) == ri(1);                     // unit subdiagonal (unnormalised)
// Lanczos rejects a non-symmetric matrix; Arnoldi accepts it.
lanczos_rational(nonsym, std::vector<Rational>{ri(1), ri(0)}, 2).error();  // domain_error
auto arn = arnoldi_rational(nonsym, std::vector<Rational>{ri(1), ri(0)}, 2);
arn.has_value();                                        // true

// ---- NUMERICAL: GMRES and BiCGSTAB on a non-symmetric double system ----
const std::array<double, 9> adata{3.0, 1.0, 0.0,
                                  0.0, 4.0, 1.0,
                                  1.0, 0.0, 5.0};
const std::array<double, 3> rhs{1.0, 2.0, 3.0};
auto Op = dense_matvec(adata, 3);

auto g = gmres(Op, rhs, 1e-12, 100, 3).value();
g.converged;                                           // true
g.residual < 1e-8;                                     // true (true residual ||b - A x||)

auto bc = bicgstab(Op, rhs, 1e-12, 100).value();
bc.converged;                                          // true — agrees with GMRES to ~1e-6

// ---- NUMERICAL: capped iterations report, they do not error ----
auto capped = bicgstab(Op, rhs, 1e-14, 1).value();     // one step, tight tolerance
capped.converged;                                       // false  (NOT an error)
capped.iterations == 1;                                 // true
// Only an empty system is genuinely invalid.
std::array<double, 0> empty{};
bicgstab(Op, empty, 1e-10, 10).error();                // MathError::domain_error
```

## See also

- [`nimblecas.matrix`](matrix.md) — the exact dense linear algebra over `Q` that
  supplies the SPD `A`, the right-hand side columns, and the `Matrix::solve`
  cross-check.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field every exact
  routine computes in.
- [`nimblecas.complex`](complex.md) — a sibling exact-over-`Q` numeric module.
- [Documentation hub](../Index.md)
