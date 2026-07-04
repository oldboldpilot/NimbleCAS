# `nimblecas.matdecomp` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/matdecomp/matdecomp.cppm`

Exact **LU decomposition** over the field `Q`, plus a family of **structural
predicates** that recognise the shape of a matrix — symmetric, triangular,
banded, Toeplitz, Hessenberg, and friends (ROADMAP §7.2, *Special Matrix
Structures* and *Matrix Decompositions*). These are precisely the structure
classes that the [`bandsolve`](bandsolve.md) layer provides fast *solvers* for:
this module *recognises* the shape, `bandsolve` *exploits* it. Because entries
are [`Rational`](ratpoly.md), every pivot, multiplier, and comparison is
**exact** — the LU factors reconstruct `P·A = L·U` identically (not "up to
rounding"), and a matrix is reported symmetric only when `A(i,j)` equals
`A(j,i)` as fractions, with **no tolerance** involved and **no floating point**
anywhere.

```cpp
import nimblecas.matdecomp;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), and [`matrix`](matrix.md).

## The exact/numerical boundary

The decomposition is **Doolittle elimination with partial pivoting**: `l` is
unit lower-triangular (`1`s on its diagonal), `u` is upper-triangular,
`permutation` records the row order applied to `A`, and `sign` is the
determinant of that permutation (`+1` / `-1`). Over exact arithmetic **any**
nonzero pivot serves, so the algorithm takes the *first* nonzero entry at or
below the diagonal — no floating-point magnitude comparison is needed and the
multipliers stay simple.

Two boundaries are worth stating honestly:

- **Singularity is a hard failure, not a small residual.** A column that is
  entirely zero at/below the pivot means `A` is singular; no unit-lower/upper
  factorisation with the standard normalisation exists, so the call returns
  `MathError::domain_error` rather than a rank-deficient approximation.
- **Exact-coefficient growth, and overflow.** Elimination over `Rational`
  keeps numerators and denominators exact, so intermediate entries can grow.
  Every pivot division, multiplier product, and subtraction flows through
  `Rational`'s checked arithmetic; an `int64` numerator or denominator that
  would wrap surfaces as `MathError::overflow` (Rule 32) rather than silently
  wrapping.

The predicates are **total, infallible** `bool` functions: they never allocate a
`Result` and never fail, because comparing two canonical `Rational`s for
(anti)equality is exact and cannot overflow. (`is_skew_symmetric` internally
negates an entry to compare, but a canonical `Rational` can only fail to negate
on an unreachable `INT64_MIN`, so the predicate still reports a plain `bool`.)

## `LuDecomposition` — the result of `lu_decompose`

```cpp
struct LuDecomposition {
    Matrix l;
    Matrix u;
    std::vector<std::size_t> permutation;
    int sign;
};
```

The factorisation `P·A = L·U`.

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `l` | `Matrix` | Unit lower-triangular: `l(i,i) == 1`, `l(i,j) == 0` for `j > i`. |
| `u` | `Matrix` | Upper-triangular: `u(i,j) == 0` for `i > j`. |
| `permutation` | `std::vector<std::size_t>` | The row order applied to `A`: `permutation[i]` is the index of the original row of `A` that now sits in position `i`, so `(P·A)` row `i` equals `A` row `permutation[i]`. |
| `sign` | `int` | `det(P)`: `+1` for an even number of row swaps, `-1` for an odd number. |

### Decomposition

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `lu_decompose` | `[[nodiscard]] auto lu_decompose(const Matrix& a) -> Result<LuDecomposition>` | Exact LU with partial pivoting: computes `P`, `L`, `U` with `P·A = L·U`. Requires `A` square (`domain_error` otherwise). Selects at each column the first nonzero pivot at or below the diagonal, recording the swap in `permutation` and flipping `sign`. An entirely-zero pivot column means `A` is singular → `domain_error`. Entry-arithmetic overflow propagates as `overflow`. |

## Structure predicates — exact, over `Rational`

All predicates are `[[nodiscard]]` and return a plain `bool`. Those that imply
squareness — `is_symmetric`, `is_skew_symmetric`, `is_diagonal`,
`is_upper_triangular`, `is_lower_triangular`, `is_tridiagonal`,
`is_upper_hessenberg`, `is_lower_hessenberg`, `is_identity` — return `false` for a
non-square input. `is_toeplitz` and `is_banded` also accept **rectangular**
matrices.

| Predicate | Signature | True when |
| :--- | :--- | :--- |
| `is_symmetric` | `auto is_symmetric(const Matrix& a) -> bool` | `A(i,j) == A(j,i)` for all `i, j`. |
| `is_skew_symmetric` | `auto is_skew_symmetric(const Matrix& a) -> bool` | `A(i,j) == -A(j,i)` for all `i, j` (forcing a zero diagonal). |
| `is_diagonal` | `auto is_diagonal(const Matrix& a) -> bool` | Every off-diagonal entry is zero. |
| `is_upper_triangular` | `auto is_upper_triangular(const Matrix& a) -> bool` | Every entry below the main diagonal is zero (`A(i,j) == 0` for `i > j`). |
| `is_lower_triangular` | `auto is_lower_triangular(const Matrix& a) -> bool` | Every entry above the main diagonal is zero (`A(i,j) == 0` for `i < j`). |
| `is_tridiagonal` | `auto is_tridiagonal(const Matrix& a) -> bool` | Zero outside the main diagonal and the first sub/super-diagonals (`\|i - j\| <= 1`). |
| `is_banded` | `auto is_banded(const Matrix& a, std::size_t lower_bandwidth, std::size_t upper_bandwidth) -> bool` | `A(i,j) == 0` whenever `j < i - lower_bandwidth` or `j > i + upper_bandwidth`. Rectangular matrices accepted; the unsigned `i - lower_bandwidth` is guarded. |
| `is_upper_hessenberg` | `auto is_upper_hessenberg(const Matrix& a) -> bool` | Zero below the first subdiagonal (`A(i,j) == 0` for `i > j + 1`). |
| `is_lower_hessenberg` | `auto is_lower_hessenberg(const Matrix& a) -> bool` | Zero above the first superdiagonal (`A(i,j) == 0` for `j > i + 1`). |
| `is_toeplitz` | `auto is_toeplitz(const Matrix& a) -> bool` | Constant along every diagonal: `A(i,j) == A(i-1,j-1)` for all `i, j >= 1`. Rectangular matrices accepted. |
| `is_identity` | `auto is_identity(const Matrix& a) -> bool` | Square with `1`s on the diagonal and `0`s elsewhere. |

## Error model

Only `lu_decompose` is fallible; the predicates never error.

| Condition | Error |
| :--- | :--- |
| `lu_decompose` on a non-square `A` | `MathError::domain_error` |
| `lu_decompose` on a singular `A` (an entirely-zero pivot column at/below the diagonal) | `MathError::domain_error` |
| An `int64` numerator or denominator computation wraps during elimination | `MathError::overflow` |

The structure predicates are total: they consume any `Matrix` (square or
rectangular) and always return a `bool`, never a `Result`.

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.matdecomp;
using namespace nimblecas;

auto ri = [](std::int64_t v) { return Rational::from_int(v); };
// Build a Matrix from integer rows (low-index row first).
auto mat = [](std::vector<std::vector<std::int64_t>> rows) {
    std::vector<std::vector<Rational>> r;
    for (const auto& row : rows) {
        std::vector<Rational> rr;
        for (const std::int64_t v : row) rr.push_back(Rational::from_int(v));
        r.push_back(std::move(rr));
    }
    return Matrix::from_rows(std::move(r)).value();
};

// --- LU decomposition: P·A = L·U ---------------------------------------------
// A = [[2,1,1],[4,3,3],[8,7,9]] needs no swaps under first-nonzero pivoting:
//   L = [[1,0,0],[2,1,0],[4,3,1]], U = [[2,1,1],[0,1,1],[0,0,2]].
const Matrix a  = mat({{2, 1, 1}, {4, 3, 3}, {8, 7, 9}});
const auto    lu = lu_decompose(a).value();
lu.sign;                                   // +1  (no swaps)
lu.permutation;                            // {0, 1, 2}  (identity order)
lu.l.at(0, 0) == ri(1);                    // true  (unit diagonal)
// Reconstruct exactly: L·U equals P·A row-for-row.
const auto reconstructed = lu.l.multiply(lu.u).value();  // == A here (P is identity)

// A forced pivot swap flips the sign.
const auto lu2 = lu_decompose(mat({{0, 1}, {1, 0}})).value();
lu2.sign;                                  // -1  (one row swap)
lu2.permutation;                           // {1, 0}

// Singular and non-square inputs fail on the railway.
lu_decompose(mat({{1, 2}, {2, 4}})).error();      // MathError::domain_error (singular)
lu_decompose(mat({{1, 2, 3}, {4, 5, 6}})).error();// MathError::domain_error (non-square)

// --- Structure predicates (exact, infallible) --------------------------------
is_symmetric(mat({{1, 2}, {2, 3}}));       // true
is_symmetric(mat({{1, 2}, {3, 4}}));       // false
is_symmetric(mat({{1, 2, 3}, {2, 4, 5}})); // false  (non-square => false)

is_skew_symmetric(mat({{0, 2}, {-2, 0}})); // true
is_skew_symmetric(mat({{1, 2}, {-2, 0}})); // false  (nonzero diagonal)

is_diagonal(mat({{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}));  // true
is_upper_triangular(mat({{1, 2}, {0, 3}}));           // true
is_lower_triangular(mat({{1, 0}, {2, 3}}));           // true

// Tridiagonal is (1,1)-banded but not (0,0)-banded; a diagonal is (0,0)-banded.
const Matrix tri = mat({{1, 2, 0}, {3, 4, 5}, {0, 6, 7}});
is_tridiagonal(tri);                       // true
is_banded(tri, 1, 1);                      // true
is_banded(tri, 0, 0);                      // false

is_upper_hessenberg(mat({{1, 2, 3}, {4, 5, 6}, {0, 7, 8}}));  // true
is_toeplitz(mat({{1, 2, 3}, {4, 1, 2}, {5, 4, 1}}));          // true
is_identity(Matrix::identity(3));                             // true
```

## See also

- [`nimblecas.matrix`](matrix.md) — the exact-`Rational` dense `Matrix` this
  module decomposes and inspects.
- [`nimblecas.bandsolve`](bandsolve.md) — exact direct solvers for the
  tridiagonal / banded structures these predicates recognise.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field every entry,
  pivot, and multiplier lives in.
- [`nimblecas.core`](core.md) — the `Result<T>` / `MathError` railway.
- [Documentation hub](../Index.md)
