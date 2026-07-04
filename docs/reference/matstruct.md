# `nimblecas.matstruct` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/matstruct/matstruct.cppm`

The **structured** side of dense linear algebra over the rationals (ROADMAP
§7.2): builders that assemble block/diagonal shapes, predicates that recognise
them, and the two symmetric factorizations that stay inside the field `Q`.
Entries are [`Rational`](ratpoly.md) — a reduced `int64` fraction — so every
pivot, multiplier, and comparison is **exact**, with **no floating point** and
**no tolerance** anywhere. This module is deliberately honest about where the
rationals run out:

- **LDL^T** — `ldlt_decompose` computes `A = L·D·Lᵀ` with `L` unit
  lower-triangular and `D` diagonal, taking **no square roots**. It is the
  exact-over-`Q` analogue of Cholesky: `L` and `D` are exact rational matrices
  and `L·D·Lᵀ` reconstructs `A` identically.
- **Exact Cholesky** — `cholesky_exact` returns a genuine `A = G·Gᵀ` with
  lower-triangular `G` **only** when every LDL^T pivot `D_ii` is a positive
  **perfect rational square** (then `G = L·diag(√D)`). When a pivot is positive
  but not a perfect square, `√(D_ii)` is irrational and no exact rational `G`
  exists — the module returns `domain_error` rather than silently rounding. A
  numerically-rounded Cholesky belongs to a bigfloat layer, not here.
- **Rational Hessenberg** — `hessenberg_form` reduces `A` to upper-Hessenberg
  by **rational elementary similarity** `H = N·A·N⁻¹` (Gaussian, non-orthogonal).
  `H` is **similar** to `A`, so it has the **same characteristic polynomial**
  (eigenvalues, determinant, trace). It is deliberately **not** the orthogonal
  Householder Hessenberg (`QᵀAQ` with `Q` orthogonal), which needs `√·` for the
  reflector norms and leaves `Q`. The rational similarity form is the honest
  exact counterpart a purely rational spectral pipeline needs.

True orthogonal Cholesky/Householder-Hessenberg require irrational square roots
and are out of scope for this exact-rational module.

```cpp
import nimblecas.matstruct;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md),
[`matrix`](matrix.md), and [`matdecomp`](matdecomp.md). The scalar-shape
predicates (`is_symmetric`, `is_diagonal`, `is_upper_hessenberg`, …) are **not**
duplicated here — this module reuses them from [`matdecomp`](matdecomp.md) and
adds only the block-structure predicates and the LDL^T-based positive-definiteness
test.

## The overflow contract

Following the rest of the engine, arithmetic is **exact** and
**overflow-checked** (Rule 32): every entry operation flows through `Rational`'s
checked add / subtract / multiply / divide, so an `int64` numerator or
denominator that would overflow surfaces as `MathError::overflow` rather than
silently wrapping. Shape violations (non-square, non-symmetric, mismatched
blocks, zero pivots) surface as `MathError::domain_error`. Block-assembly
dimension sums are checked and overflow to `MathError::overflow`. The
`bool`-returning predicates are total — they never error; where an intermediate
rational operation would overflow, they conservatively return `false` (see
`is_symmetric_positive_definite`).

## `LdltDecomposition` — the result of `ldlt_decompose`

The exact `A = L·D·Lᵀ` factorization, returned by value.

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `l` | `Matrix` | Unit lower-triangular: `l(i,i) == 1`, `l(i,j) == 0` for `j > i`. |
| `d` | `Matrix` | Diagonal: the pivots `D_ii` on the diagonal, `0` off it. |

Reconstruction is `L·D·Lᵀ`, exact over `Rational`.

## Structured builders

All return `Result<Matrix>` and are exact over `Rational`. All symbols live in
namespace `nimblecas`.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `diagonal` | `auto diagonal(const std::vector<Rational>& entries) -> Result<Matrix>` | The `n × n` diagonal matrix with `entries` on the main diagonal (`0` elsewhere), `n == entries.size()`. An empty list yields the 0×0 matrix. |
| `block_diagonal` | `auto block_diagonal(const std::vector<Matrix>& blocks) -> Result<Matrix>` | The block-diagonal matrix `diag(blocks[0], blocks[1], …)`: each block placed on the diagonal, zeros elsewhere. Result is (Σ block rows) × (Σ block cols); blocks need not be square (a square result — needed for a determinant — needs square blocks). Empty list yields the 0×0 matrix. Summed dimensions that would wrap `std::size_t` fail with `overflow`. |
| `block_upper_triangular` | `auto block_upper_triangular(const std::vector<std::vector<Matrix>>& block_rows) -> Result<Matrix>` | A block-upper-triangular matrix from a **jagged** upper-triangular grid: `block_rows[i]` lists block-row `i` starting at block-column `i`, so `block_rows[i][0]` is the `(i,i)` **diagonal** block and `block_rows[i][t]` is the `(i, i+t)` block. Block-row `i` must supply exactly `k − i` blocks (`k` = number of block-rows). Diagonal blocks must be **square**; off-diagonal `(i, i+t)` must have as many rows as diagonal block `i` and as many columns as diagonal block `i+t`. Blocks strictly below the block-diagonal are implicitly zero. Shape violations fail `domain_error`; summed-dimension overflow fails `overflow`. Empty grid yields the 0×0 matrix. |

## Block-structure predicates

Exact `bool` predicates (the scalar-shape predicates live in
[`matdecomp`](matdecomp.md)). All in namespace `nimblecas`.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `is_block_diagonal` | `auto is_block_diagonal(const Matrix& a, const std::vector<std::size_t>& block_sizes) -> bool` | Is `A` block-diagonal for the partition `block_sizes` (which must sum to `n` for a square `n × n` `A`)? Every entry whose row and column fall in **different** diagonal blocks must be exactly zero. A non-square matrix, or a partition that does not sum to `n`, yields `false`. |
| `is_block_upper_triangular` | `auto is_block_upper_triangular(const Matrix& a, const std::vector<std::size_t>& block_sizes) -> bool` | Is `A` block-upper-triangular for `block_sizes`? Every entry whose row-block index **exceeds** its column-block index (strictly below the block-diagonal) must be exactly zero; entries on or above are unconstrained. A non-square matrix, or a partition that does not sum to `n`, yields `false`. |
| `is_symmetric_positive_definite` | `auto is_symmetric_positive_definite(const Matrix& a) -> bool` | Is `A` symmetric positive definite? Decided **exactly** through the sign of the LDL^T pivots: `A` must be symmetric and the exact factorization must exist with every `D_ii > 0`. A non-symmetric matrix, a zero pivot (indefinite/semidefinite/singular), or a negative pivot yields `false`. If an intermediate rational operation overflows `int64`, the result is conservatively `false` (the matrix could not be certified positive definite within the `int64` rational tier — the bigrational tier lifts that ceiling). No eigenvalue estimation, no tolerance. |

## Exact symmetric factorizations

All in namespace `nimblecas`.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `ldlt_decompose` | `auto ldlt_decompose(const Matrix& a) -> Result<LdltDecomposition>` | Exact rational LDL^T: `A = L·D·Lᵀ` with `L` unit lower-triangular and `D` diagonal, taking **no square roots**. Requires `A` square and symmetric (`domain_error` otherwise). This is the no-pivoting factorization: a zero pivot `D_jj == 0` would require a symmetric interchange to continue, which this exact form does not perform, so it fails `domain_error`. Entry-arithmetic overflow propagates as `overflow`. |
| `cholesky_exact` | `auto cholesky_exact(const Matrix& a) -> Result<Matrix>` | Exact Cholesky `A = G·Gᵀ` with `G` lower-triangular, available **only** when `A` is symmetric positive definite **and** every LDL^T pivot `D_ii` is a perfect rational square (then `G = L·diag(√(D_ii))` is exact over `Q`). Non-symmetric or a zero pivot fails `domain_error` (as `ldlt_decompose` does). A pivot `≤ 0` (not positive definite) or positive but **not** a perfect rational square (`√(D_ii)` irrational, so no exact rational `G` exists) fails `domain_error`: the exact path deliberately refuses to round. |
| `hessenberg_form` | `auto hessenberg_form(const Matrix& a) -> Result<Matrix>` | Exact rational upper-Hessenberg reduction by elementary similarity: returns `H = N·A·N⁻¹` (`N` a product of exact rational Gauss/permutation transforms) that is upper-Hessenberg and **similar** to `A` — same characteristic polynomial, eigenvalues, determinant, and trace. This is the Gaussian (non-orthogonal) reduction, **not** the orthogonal Householder Hessenberg (which needs `√·` and leaves `Q`). Requires `A` square (`domain_error` otherwise). Entry overflow propagates as `overflow`. |

## Error model

| Condition | Error |
| :--- | :--- |
| `ldlt_decompose` / `cholesky_exact` / `hessenberg_form` on a non-square `A` | `MathError::domain_error` |
| `ldlt_decompose` / `cholesky_exact` on a non-symmetric `A` | `MathError::domain_error` |
| `ldlt_decompose` / `cholesky_exact` hits a zero pivot (symmetric interchange required) | `MathError::domain_error` |
| `cholesky_exact` on a non-positive pivot (not positive definite) | `MathError::domain_error` |
| `cholesky_exact` on a positive but non-perfect-square pivot (irrational `√`) | `MathError::domain_error` |
| `block_upper_triangular` with the wrong block count per row, a non-square diagonal block, or a mismatched off-diagonal block | `MathError::domain_error` |
| Summed block dimensions wrap `std::size_t` (`block_diagonal` / `block_upper_triangular`) | `MathError::overflow` |
| An `int64` numerator or denominator computation wraps | `MathError::overflow` |

The `bool`-returning predicates (`is_block_diagonal`,
`is_block_upper_triangular`, `is_symmetric_positive_definite`) never return an
error — a bad shape or partition, and (for SPD) an intermediate overflow, all
map conservatively to `false`.

## Worked examples

```cpp
import nimblecas.matstruct;
import nimblecas.matrix;
import nimblecas.ratpoly;
using namespace nimblecas;

auto ri  = [](std::int64_t v) { return Rational::from_int(v); };
auto mat = [](std::vector<std::vector<Rational>> r) {
    return Matrix::from_rows(std::move(r)).value();
};

// --- Structured builders ------------------------------------------------------

// diagonal(1, 2, 3) is the 3x3 diag matrix; off-diagonal entries are exact 0.
auto d = diagonal({ri(1), ri(2), ri(3)}).value();       // [[1,0,0],[0,2,0],[0,0,3]]
diagonal({}).value();                                    // the 0x0 matrix

// block_diagonal: determinant is the product of the block determinants.
//   det[[1,2],[3,4]] = -2, det[[2]] = 2, det[[1,1],[0,3]] = 3  =>  -12.
auto blk = block_diagonal({mat({{ri(1), ri(2)}, {ri(3), ri(4)}}),
                           mat({{ri(2)}}),
                           mat({{ri(1), ri(1)}, {ri(0), ri(3)}})}).value();
blk.determinant().value();                               // -12   (5x5 block-diagonal)
is_block_diagonal(blk, {2, 1, 2});                       // true

// block_upper_triangular: diagonal blocks D0 (2x2), D1 (1x1) with an
// off-diagonal (0,1) block U (2x1). Below the block-diagonal is zero.
auto but = block_upper_triangular({{mat({{ri(1), ri(2)}, {ri(3), ri(4)}}),  // D0
                                     mat({{ri(5)}, {ri(6)}})},               // U01
                                    {mat({{ri(7)}})}}).value();              // D1
but.is_equal(mat({{ri(1), ri(2), ri(5)},
                  {ri(3), ri(4), ri(6)},
                  {ri(0), ri(0), ri(7)}}));              // true
but.determinant().value();                               // -14  (= det(D0)*det(D1))
// A mismatched off-diagonal block (wrong column count) is rejected.
block_upper_triangular({{mat({{ri(1), ri(2)}, {ri(3), ri(4)}}),
                         mat({{ri(5), ri(9)}, {ri(6), ri(9)}})},
                        {mat({{ri(7)}})}}).error();       // MathError::domain_error

// --- Exact LDL^T --------------------------------------------------------------

// A = [[4,2,0],[2,5,2],[0,2,3]] is SPD. Pivots are (4, 4, 2), L is unit
// lower-triangular, and L*D*L^T reconstructs A exactly.
const auto a = mat({{ri(4), ri(2), ri(0)},
                    {ri(2), ri(5), ri(2)},
                    {ri(0), ri(2), ri(3)}});
auto ldlt = ldlt_decompose(a).value();
ldlt.d.at(0, 0);                                         // 4
ldlt.d.at(1, 1);                                         // 4
ldlt.d.at(2, 2);                                         // 2
auto lt  = ldlt.l.transpose().value();
auto ld  = ldlt.l.multiply(ldlt.d).value();
ld.multiply(lt).value().is_equal(a);                     // true  (L*D*L^T == A)

// LDL^T declines non-symmetric, zero-pivot, and non-square inputs.
ldlt_decompose(mat({{ri(1), ri(2)}, {ri(3), ri(4)}})).error();  // domain_error (non-symmetric)
ldlt_decompose(mat({{ri(0), ri(1)}, {ri(1), ri(0)}})).error();  // domain_error (zero pivot)

// --- Exact positive-definiteness ---------------------------------------------

is_symmetric_positive_definite(mat({{ri(2), ri(1)}, {ri(1), ri(2)}}));  // true  (pivots 2, 3/2)
is_symmetric_positive_definite(mat({{ri(1), ri(2)}, {ri(2), ri(1)}}));  // false (indefinite, det -3)
is_symmetric_positive_definite(mat({{ri(1), ri(2)}, {ri(3), ri(4)}}));  // false (non-symmetric)

// --- Exact Cholesky (perfect-square pivots only) -----------------------------

// A = [[4,2],[2,5]]: pivots D0 = 4, D1 = 4 are perfect squares, so
// G = L*diag(2,2) = [[2,0],[1,2]] is exact and G*G^T == A.
auto g  = cholesky_exact(mat({{ri(4), ri(2)}, {ri(2), ri(5)}})).value();
g.is_equal(mat({{ri(2), ri(0)}, {ri(1), ri(2)}}));       // true
g.multiply(g.transpose().value()).value()
 .is_equal(mat({{ri(4), ri(2)}, {ri(2), ri(5)}}));       // true  (G*G^T == A)

// A = [[2,1],[1,2]] is SPD but D0 = 2 is not a perfect square: sqrt(2) is
// irrational, so no EXACT rational Cholesky exists — the exact path refuses.
cholesky_exact(mat({{ri(2), ri(1)}, {ri(1), ri(2)}})).error();  // domain_error

// --- Rational Hessenberg similarity ------------------------------------------

// A dense 4x4 reduces to upper-Hessenberg by rational similarity: H = N*A*N^-1
// is similar to A, so it shares A's characteristic polynomial (checked with
// nimblecas.eigen::characteristic_polynomial).
const auto h = hessenberg_form(mat({{ri(4), ri(1), ri(2), ri(3)},
                                    {ri(1), ri(3), ri(5), ri(1)},
                                    {ri(2), ri(5), ri(6), ri(2)},
                                    {ri(7), ri(1), ri(4), ri(5)}})).value();
// is_upper_hessenberg(h) == true; characteristic_polynomial(h) == characteristic_polynomial(A).
```

## See also

- [`nimblecas.matrix`](matrix.md) — the dense exact-rational `Matrix` this
  module builds on (`determinant`, `multiply`, `transpose`, `inverse`).
- [`nimblecas.matdecomp`](matdecomp.md) — exact LU plus the scalar-shape
  predicates (`is_symmetric`, `is_upper_hessenberg`, …) reused here.
- [`nimblecas.eigen`](eigen.md) — `characteristic_polynomial`, the similarity
  invariant that certifies `hessenberg_form` preserves the spectrum.
- [`nimblecas.bandsolve`](bandsolve.md) — fast solvers that *exploit* the
  structures this module recognises.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field the entries
  live in.
- [Documentation hub](../Index.md)
```
