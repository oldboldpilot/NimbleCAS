# `nimblecas.matrix` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/matrix/matrix.cppm`

Dense linear algebra done **exactly** over the rationals, `Q^{m x n}` (ROADMAP
§7.2). Entries are [`Rational`](ratpoly.md) — a reduced `int64` fraction — so
determinants, solutions of linear systems, and inverses carry **no rounding
error at all**: a 3×3 determinant is the integer/fraction it mathematically is,
not a `double` that happens to be close. Elimination runs over the field `Q`,
which is exactly why plain Gaussian / Gauss-Jordan elimination suffices — no
fraction-free Bareiss step is needed to stay in a ring, because `Rational`
already reduces every intermediate. Storage is dense row-major: entry `(i, j)`
lives at `data_[i * cols_ + j]`.

```cpp
import nimblecas.matrix;
```

Depends on [`core`](core.md) and [`ratpoly`](ratpoly.md).

## The overflow contract

Following the rest of the engine, arithmetic is **exact** and
**overflow-checked** (Rule 32): every entry operation flows through `Rational`'s
checked add / subtract / multiply / divide, so an `int64` numerator or
denominator that would overflow surfaces as `MathError::overflow` rather than
silently wrapping. Dimension violations surface as `MathError::domain_error`,
and a **singular** system (no inverse, no unique solution) surfaces as
`domain_error` too. The default-constructed matrix is the empty 0×0 matrix — a
valid, if degenerate, value whose determinant is the empty product `1`.

## `Matrix` — a dense `rows × cols` grid of exact `Rational` entries

### Construction

| Constructor / factory | Signature | Notes |
| :--- | :--- | :--- |
| default | `Matrix()` | The empty 0×0 matrix (determinant `1`). |
| `from_rows` | `static auto from_rows(std::vector<std::vector<Rational>> rows) -> Result<Matrix>` | Build from a list of rows. Every row must have the same length, else `domain_error` (ragged). An empty list yields the 0×0 matrix; a list of empty rows yields an `r × 0` matrix. |
| `identity` | `static auto identity(std::size_t n) -> Matrix` | The `n × n` identity (`1` on the diagonal, `0` elsewhere). `identity(0)` is the 0×0 matrix. |
| `zero` | `static auto zero(std::size_t rows, std::size_t cols) -> Matrix` | The `rows × cols` all-zero matrix. |

### Accessors

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `rows` | `auto rows() const noexcept -> std::size_t` | Row count. |
| `cols` | `auto cols() const noexcept -> std::size_t` | Column count. |
| `is_square` | `auto is_square() const noexcept -> bool` | `true` when `rows() == cols()`. |
| `at` | `auto at(std::size_t i, std::size_t j) const -> const Rational&` | Entry `(i, j)`. Asserted in-range — callers hold indices below `rows()` / `cols()`. |
| `operator==` | `auto operator==(const Matrix&) const noexcept -> bool` | Same shape and exact entry-vector equality. |
| `is_equal` | `auto is_equal(const Matrix&) const noexcept -> bool` | Alias for `operator==`. |
| `to_string` | `auto to_string() const -> std::string` | Human-readable rendering, e.g. `[[1, 2], [3, 4]]`. |

### Arithmetic (overflow-checked, exact)

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `add` | `auto add(const Matrix& o) const -> Result<Matrix>` | Entrywise sum; dimensions must match, else `domain_error`. |
| `subtract` | `auto subtract(const Matrix& o) const -> Result<Matrix>` | Entrywise difference; dimensions must match, else `domain_error`. |
| `scale` | `auto scale(const Rational& s) const -> Result<Matrix>` | Multiply every entry by scalar `s`. |
| `multiply` | `auto multiply(const Matrix& o) const -> Result<Matrix>` | Matrix product (`this` is `m × k`, `o` is `k × n`). Fails `domain_error` when the inner dimensions disagree (`cols() != o.rows()`). |
| `transpose` | `auto transpose() const -> Result<Matrix>` | The `cols × rows` transpose. Never fails, but returns `Result` for a uniform surface. |
| `trace` | `auto trace() const -> Result<Rational>` | Sum of the diagonal entries; requires a square matrix (`domain_error` otherwise). |

Each returns `MathError::overflow` if any entry computation wraps `int64`.

### Decompositions / solving

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `determinant` | `auto determinant() const -> Result<Rational>` | Exact determinant via Gaussian elimination with pivoting on nonzero pivots. Requires a square matrix (`domain_error` otherwise); the 0×0 determinant is `1`, and a singular matrix returns `0`. |
| `solve` | `auto solve(const Matrix& b) const -> Result<Matrix>` | Solve `A x = b` for `x`, where `A` is this (square, nonsingular) and `b` has the same number of rows as `A` (`b` may carry several right-hand-side columns). Uses Gauss-Jordan elimination with partial pivoting. Fails `domain_error` when `A` is not square, the row counts disagree, or `A` is singular. |
| `inverse` | `auto inverse() const -> Result<Matrix>` | The multiplicative inverse (`A A^{-1} = I`), computed by Gauss-Jordan on `[A | I]`. A singular or non-square matrix yields `domain_error`. |
| `rank` | `auto rank() const -> std::int64_t` | The rank (number of linearly independent rows) via row reduction over `Q`. Infallible by signature: should an intermediate entry overflow `int64`, reduction stops early and the rank counted so far is returned (a conservative lower bound). |

## Error model

| Condition | Error |
| :--- | :--- |
| Ragged rows in `from_rows` | `MathError::domain_error` |
| Dimension mismatch (`add`/`subtract`/`multiply`) | `MathError::domain_error` |
| Non-square where a square is required (`trace`/`determinant`/`solve`/`inverse`) | `MathError::domain_error` |
| Row-count mismatch in `solve` | `MathError::domain_error` |
| Singular matrix in `solve` / `inverse` | `MathError::domain_error` |
| An `int64` numerator or denominator computation wraps | `MathError::overflow` |

`rank` never returns an error — an intermediate overflow truncates it to a
conservative lower bound; `determinant` returns `0` (not an error) for a
singular matrix.

## Examples

```cpp
import nimblecas.matrix;
import nimblecas.ratpoly;
using namespace nimblecas;

auto ri  = [](std::int64_t v) { return Rational::from_int(v); };
auto mat = [](std::vector<std::vector<Rational>> r) {
    return Matrix::from_rows(std::move(r)).value();
};

// Determinant (exact): det [[6,1,1],[4,-2,5],[2,8,7]] = -306.
auto d = mat({{ri(6), ri(1), ri(1)},
              {ri(4), ri(-2), ri(5)},
              {ri(2), ri(8), ri(7)}}).determinant();       // -306

// Multiply: [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]].
auto p = mat({{ri(1), ri(2)}, {ri(3), ri(4)}})
             .multiply(mat({{ri(5), ri(6)}, {ri(7), ri(8)}}));

// Solve A x = b: 2x + y = 3, x - y = 0  =>  x = y = 1.
auto A = mat({{ri(2), ri(1)}, {ri(1), ri(-1)}});
auto b = mat({{ri(3)}, {ri(0)}});
auto x = A.solve(b).value();                                // [[1], [1]]

// Inverse: inv([[1,2],[3,4]]) = [[-2, 1], [3/2, -1/2]], and A * A^-1 = I.
auto M   = mat({{ri(1), ri(2)}, {ri(3), ri(4)}});
auto inv = M.inverse().value();
auto id  = M.multiply(inv).value();                         // == identity(2)
```

## See also

- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field the entries
  live in.
- [`nimblecas.combinatorics`](combinatorics.md) and
  [`nimblecas.orthopoly`](orthopoly.md) — the sibling `ratpoly`-consuming
  numeric modules.
- [Documentation hub](../Index.md)
