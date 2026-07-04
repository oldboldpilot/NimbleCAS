# `nimblecas.bandsolve` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/bandsolve/bandsolve.cppm`

Exact direct solvers for **tridiagonal** and general **banded** linear systems
over the field `Q` (ROADMAP §7.2). Every pivot, multiplier, elimination step,
and back-substitution is a [`Rational`](ratpoly.md) operation, so the returned
`x` satisfies `A·x = b` **identically** — not "up to rounding". There is **no
floating point and no tolerance** anywhere in this module; the only exact/numeric
boundary is that these are *direct* factorisations (Thomas, band-LU), so their
cost and exact-coefficient growth track the classical algorithms, and failure is
reported solely on the railway ([`Result<T>`](core.md) / `MathError`) — nothing
throws.

The module sits above [`matrix`](matrix.md) and [`ratpoly`](ratpoly.md) in the
tower and is a consumer of the [`parallel`](parallel.md) fan-out primitives. It
exposes three free functions and no types of its own.

**Pivoting boundary.** All three solvers are **partial-pivoting-free**. Thomas
and band-LU both assume a nonzero pivot at each step (the classical
diagonal-dominance precondition). A zero pivot — an original zero diagonal *or*
one produced by exact elimination — is reported as `MathError::domain_error`
("singular, or needs a permutation we do not perform") rather than worked around,
so a returned solution is always genuine.

**Parallelism boundary (honesty).** The Thomas recurrence is inherently `O(n)`
**sequential** — each eliminated pivot feeds the next — so a single right-hand
side cannot be sped up by threads here. The parallel win is the **multi-RHS
batch**: distinct columns share one forward factorisation and are then solved
completely independently (embarrassingly parallel). `solve_tridiagonal_batch`
fans the columns out with [`parallel::transform_index`](parallel.md); because
every column is a pure function of the shared immutable factorisation and its own
data, the result is **bit-identical regardless of thread count or partitioning**.
This many-RHS shape is exactly what a GPU backend would offload (one thread/warp
per column); that offload is noted but deliberately **not implemented** — there
is no CUDA in this module. An exact parallel single-RHS cyclic-reduction solver
is likewise intentionally omitted: it is expressible over `Q`, but the level-wise
recombination grows rational numerators/denominators in a way whose bit-identical
reproducibility across arbitrary thread partitions has not been validated, so the
single-RHS path stays the plain sequential Thomas recurrence.

```cpp
import nimblecas.bandsolve;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), [`matrix`](matrix.md), and
[`parallel`](parallel.md).

## Tridiagonal solve — the Thomas algorithm

For an `n × n` tridiagonal `A`, the operator is given by three spans plus the
right-hand side. With diagonal entries `b₁..bₙ`, subdiagonal entries `a₂..aₙ`
(where `aᵢ = A(i, i−1)`), and superdiagonal entries `c₁..c_{n−1}` (where
`cᵢ = A(i, i+1)`):

| Argument | Length | Meaning |
| :--- | :--- | :--- |
| `sub` | `n − 1` | subdiagonal `a₂..aₙ`, i.e. `A(i, i−1)` |
| `diag` | `n` | main diagonal `b₁..bₙ`; `n` is taken from `diag.size()` |
| `super` | `n − 1` | superdiagonal `c₁..c_{n−1}`, i.e. `A(i, i+1)` |
| `rhs` | `n` | right-hand-side vector |

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `solve_tridiagonal` | `[[nodiscard]] auto solve_tridiagonal(std::span<const Rational> sub, std::span<const Rational> diag, std::span<const Rational> super, std::span<const Rational> rhs) -> Result<std::vector<Rational>>` | Forward elimination then back-substitution in exact `Rational` arithmetic, returning the `n`-entry solution vector. Assumes diagonal dominance (no pivoting). `n` is `diag.size()`. |

The subdiagonal/superdiagonal orientation is load-bearing: swapping `sub` and
`super` solves `Aᵀ`, not `A`.

## Banded solve — band-restricted LU (Doolittle, no pivoting)

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `solve_banded` | `[[nodiscard]] auto solve_banded(const Matrix& a, std::size_t lower_bw, std::size_t upper_bw, const Matrix& b) -> Result<Matrix>` | Solves `A·x = b` by Doolittle elimination without pivoting, restricted to the band. Returns the `n × 1` solution column. |

`a` must be square (`n × n`) and `b` an `n × 1` column, else `domain_error`.
`lower_bw` and `upper_bw` are the sub- and super-diagonal bandwidths: entries of
`a` with `|i − j|` beyond the respective bandwidth are **assumed zero and never
touched**, so the elimination stays inside the band and no fill-in escapes it.
Like Thomas this is pivoting-free and assumes nonzero pivots; a zero pivot
(original or produced by elimination) yields `domain_error`. Bandwidth-1 band-LU
reproduces the Thomas result exactly.

## Batched multi-RHS tridiagonal solve — parallel over columns

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `solve_tridiagonal_batch` | `[[nodiscard]] auto solve_tridiagonal_batch(std::span<const Rational> sub, std::span<const Rational> diag, std::span<const Rational> super, const Matrix& rhs_columns) -> Result<Matrix>` | Solves the SAME tridiagonal operator `(sub, diag, super)` against every column of the `n × k` matrix `rhs_columns`, returning the `n × k` solution matrix whose column `j` solves `A·x = rhs_columns(:, j)`. |

The operator arguments are validated exactly as in `solve_tridiagonal`, with the
added requirement that `rhs_columns` has exactly `n` rows (else `domain_error`).
The forward factorisation of `A` is computed **once** and shared; the `k` columns
are then solved independently and in parallel via
[`parallel::transform_index`](parallel.md) (grain 1: one task per column, the
backend auto-chunks, index order preserved). Each column is a pure function of
the shared factorisation and its own data, so the result is identical regardless
of thread count. The first per-column failure (e.g. overflow) is surfaced on the
railway.

## Error model

| Condition | Error |
| :--- | :--- |
| `solve_tridiagonal` / `solve_tridiagonal_batch`: empty system (`diag.size() == 0`) | `MathError::domain_error` |
| `solve_tridiagonal` / `solve_tridiagonal_batch`: `sub` or `super` not of length `n − 1` | `MathError::domain_error` |
| `solve_tridiagonal`: `rhs` not of length `n` | `MathError::domain_error` |
| `solve_tridiagonal_batch`: `rhs_columns` does not have `n` rows | `MathError::domain_error` |
| `solve_banded`: `a` not square, or `b` not `n × 1` | `MathError::domain_error` |
| Zero leading diagonal `A(0,0) == 0` (no row to swap in) | `MathError::domain_error` |
| Zero pivot produced by elimination (singular, or needs a permutation we do not perform) | `MathError::domain_error` |
| Any `int64` `Rational`-op numerator/denominator computation wraps | `MathError::overflow` |

Overflow is checked at every `Rational` multiply/divide/subtract inside the
sweeps and back-substitutions, so an exact intermediate that would exceed `int64`
surfaces as `MathError::overflow` rather than silently wrapping. In the batch
path this check is per column; the first failing column's error is returned.

## Worked examples

```cpp
import nimblecas.bandsolve;
import nimblecas.ratpoly;
import nimblecas.matrix;
using namespace nimblecas;

auto ri = [](std::int64_t v) { return Rational::from_int(v); };
auto rq = [](std::int64_t n, std::int64_t d) { return Rational::make(n, d).value(); };
auto vec = [](std::vector<std::int64_t> es) {
    std::vector<Rational> v;
    for (std::int64_t e : es) v.push_back(Rational::from_int(e));
    return v;
};

// The 3x3 discrete Laplacian tridiag(-1, 2, -1) as (sub, diag, super).
const auto sub   = vec({-1, -1});      // A(1,0), A(2,1)
const auto diag  = vec({ 2,  2, 2});
const auto super = vec({-1, -1});      // A(0,1), A(1,2)

// Single-RHS Thomas: A·x = [1,1,1] has the exact solution [3/2, 2, 3/2].
auto x = solve_tridiagonal(sub, diag, super, vec({1, 1, 1})).value();
// x[0] == 3/2, x[1] == 2, x[2] == 3/2   (exact rationals)

// Orientation matters: sub != super. A = [[2,3,0],[-1,3,1],[0,-2,2]],
// b = [8,8,2] gives the exact x = [1,2,3]; a sub/super swap would solve Aᵀ.
auto y = solve_tridiagonal(vec({-1, -2}), vec({2, 3, 2}), vec({3, 1}),
                           vec({8, 8, 2})).value();  // [1, 2, 3]

// Singular / needs pivoting -> domain_error (no work-around).
solve_tridiagonal(vec({1}), vec({1, 1}), vec({1}), vec({1, 2})).error();
    // MathError::domain_error  (zero second pivot)
solve_tridiagonal(vec({1}), vec({0, 1}), vec({1}), vec({1, 1})).error();
    // MathError::domain_error  (zero leading diagonal)
solve_tridiagonal(sub, diag, super, vec({1, 1})).error();
    // MathError::domain_error  (rhs length != n)

// Band-LU on the dense Laplacian, bandwidth 1, reproduces Thomas exactly.
auto A = Matrix::from_rows({{ri(2), ri(-1), ri(0)},
                            {ri(-1), ri(2), ri(-1)},
                            {ri(0), ri(-1), ri(2)}}).value();
auto b = Matrix::from_rows({{ri(1)}, {ri(1)}, {ri(1)}}).value();
auto xb = solve_banded(A, 1, 1, b).value();     // 3x1: [3/2, 2, 3/2]

// Bandwidth-2 (pentadiagonal), diagonally dominant, checked by A·x == b.
auto P = Matrix::from_rows({{ri(4), ri(1), ri(1), ri(0)},
                            {ri(1), ri(4), ri(1), ri(1)},
                            {ri(1), ri(1), ri(4), ri(1)},
                            {ri(0), ri(1), ri(1), ri(4)}}).value();
auto pb = Matrix::from_rows({{ri(1)}, {ri(2)}, {ri(3)}, {ri(4)}}).value();
auto xp = solve_banded(P, 2, 2, pb).value();    // A·xp == pb exactly

// Non-square a -> domain_error.
solve_banded(Matrix::from_rows({{ri(1), ri(2), ri(3)},
                                {ri(4), ri(5), ri(6)}}).value(),
             1, 1, b).error();                  // MathError::domain_error

// Batched multi-RHS: two columns share one factorisation, solved in parallel.
// rhs_columns is 3x2: columns [1,1,1] and [0,2,0].
auto R = Matrix::from_rows({{ri(1), ri(0)},
                            {ri(1), ri(2)},
                            {ri(1), ri(0)}}).value();
auto X = solve_tridiagonal_batch(sub, diag, super, R).value();
// X is 3x2; column 0 == [3/2, 2, 3/2], column 1 == [1, 2, 1],
// bit-identical to per-column Thomas regardless of thread count.
```

## See also

- [`nimblecas.matrix`](matrix.md) — the `Matrix` type used for banded operators,
  RHS columns, and returned solutions.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field every entry
  lives in.
- [`nimblecas.parallel`](parallel.md) — the `transform_index` fan-out that drives
  the multi-RHS batch.
- [`nimblecas.core`](core.md) — the `Result<T>` / `MathError` railway.
- [Documentation hub](../Index.md)
