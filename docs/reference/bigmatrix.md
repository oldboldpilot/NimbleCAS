# `nimblecas.bigmatrix` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/bigmatrix/bigmatrix.cppm`

The **unbounded, overflow-free** counterpart to [`nimblecas.matrix`](matrix.md).
Where `Matrix` stores each entry as an `int64` [`Rational`](ratpoly.md) — so a
determinant whose intermediate products or final value exceed `int64` surfaces
as `MathError::overflow` (Rule 32) — `BigMatrix` holds every entry as a
`BigRational` (an arbitrary-precision `BigInt` fraction). This removes the
ceiling entirely: a determinant is the **exact element of `Q`** it
mathematically is, with no rounding, no overflow, and no bound on dimension or
magnitude. The trade-off is speed — heap-allocating `BigInt` limb arithmetic on
every entry is far slower than a register-width `int64` multiply, which makes
this the **slow-but-exact tier**. Values that comfortably fit `int64` belong in
`nimblecas.matrix`; reach for `BigMatrix` only when magnitude, not performance,
is the constraint. This is the big-backed sibling of the exact dense linear
algebra layer (ROADMAP §7.2).

The headline `determinant()` uses the **fraction-free Bareiss** algorithm:
integer-preserving Gaussian elimination that divides each updated entry by the
*previous* pivot with exact division. On integer inputs every intermediate stays
an integer (no fraction ever appears); over `Q` it is still exact and keeps
intermediate magnitudes far smaller than the fraction explosion of naive
Gaussian elimination. Because `BigRational` is a field the division is always
exact, and the divisor is always a prior nonzero pivot (or the seed `1`), so the
only theoretically-fallible step never actually fails.

```cpp
import nimblecas.bigmatrix;
```

Depends on [`core`](core.md), `bigint`, `bigrational`, [`ratpoly`](ratpoly.md),
and [`matrix`](matrix.md).

## The exactness contract

Every arithmetic step flows through `BigRational`'s add / subtract / multiply /
negate / divide. Because `BigRational` is an arbitrary-precision fraction,
**none of these can overflow** — the `int64` wrap that `Matrix` guards against
simply cannot occur here. As a result the arithmetic operations
(`add`, `subtract`, `scale`, `multiply`, `transpose`) never fail on the numbers
themselves; the *only* error any of them can raise is `MathError::domain_error`
from a dimension violation. Each still returns `Result` so the surface stays
uniform with the rest of the engine and with `Matrix`.

Storage is dense row-major: entry `(i, j)` lives at `data_[i * cols_ + j]`.

## `BigMatrix` — a dense `rows_ x cols_` grid of exact `BigRational` entries

The default-constructed matrix is the empty `0x0` matrix — a valid, if
degenerate, value whose determinant is the empty product `1`. Every fallible
operation returns `Result`; the only infallible members are the observers below.

### Construction

| Constructor / factory | Signature | Behavior |
| :--- | :--- | :--- |
| default | `BigMatrix()` | The empty `0x0` matrix. |
| `from_rows` | `static auto from_rows(std::vector<std::vector<BigRational>> rows) -> Result<BigMatrix>` | Build from a list of equal-length rows. Fails `domain_error` if the rows are ragged, **or** if the list is empty (unlike the empty-tolerant `int64` `Matrix`; a `0x0` `BigMatrix` is still reachable via the default constructor, `identity(0)`, or `zero(0, 0)`). |
| `identity` | `static auto identity(std::size_t n) -> BigMatrix` | The `n x n` identity (`1` on the diagonal, `0` elsewhere). `identity(0)` is the `0x0` matrix. Infallible. |
| `zero` | `static auto zero(std::size_t rows, std::size_t cols) -> BigMatrix` | The `rows x cols` all-zero matrix. Infallible. |
| `from_matrix` | `static auto from_matrix(const Matrix& a) -> Result<BigMatrix>` | **Promote** an `int64`-`Rational` `Matrix` into the unbounded tier, mapping each `a.at(i, j)` to `BigRational::make(from_i64(num), from_i64(den))`. A `Rational` is always canonical (`den >= 1`), so the promotion never actually fails; `Result` is threaded through only for a uniform, defensive surface. Imposes no square rule — non-square matrices promote fine. |

### Accessors (infallible)

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `rows` | `auto rows() const noexcept -> std::size_t` | Row count. |
| `cols` | `auto cols() const noexcept -> std::size_t` | Column count. |
| `is_square` | `auto is_square() const noexcept -> bool` | `true` when `rows() == cols()`. |
| `at` | `auto at(std::size_t i, std::size_t j) const -> const BigRational&` | Entry `(i, j)`. **Asserted in-range** — callers must hold `i < rows()` and `j < cols()`; there is no bounds-checked overload. |
| `operator==` | `auto operator==(const BigMatrix& o) const noexcept -> bool` | Shape-and-entrywise equality. Canonical `BigRational` parts make it exact. |
| `is_equal` | `auto is_equal(const BigMatrix& o) const noexcept -> bool` | Named alias for `operator==`. |
| `to_string` | `auto to_string() const -> std::string` | Human-readable rendering, e.g. `"[[1, 2], [3, 4]]"`. |

### Arithmetic (all `Result`; only `domain_error` can fire)

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `add` | `auto add(const BigMatrix& o) const -> Result<BigMatrix>` | Entrywise sum. Fails `domain_error` unless both dimensions match. |
| `subtract` | `auto subtract(const BigMatrix& o) const -> Result<BigMatrix>` | Entrywise difference. Fails `domain_error` unless both dimensions match. |
| `scale` | `auto scale(const BigRational& s) const -> Result<BigMatrix>` | Multiply every entry by the scalar `s`. Never fails, but returns `Result` for a uniform surface. |
| `multiply` | `auto multiply(const BigMatrix& o) const -> Result<BigMatrix>` | Matrix product (`this` is `m x k`, `o` is `k x n`). Fails `domain_error` when the inner dimensions disagree (`cols() != o.rows()`). |
| `transpose` | `auto transpose() const -> Result<BigMatrix>` | The `cols_ x rows_` transpose. Never fails, but returns `Result` for a uniform surface. |

### Determinant (the headline)

```cpp
[[nodiscard]] auto determinant() const -> Result<BigRational>;
```

The **exact** determinant via the fraction-free Bareiss algorithm. Requires a
square matrix (`domain_error` otherwise). Special cases and behavior:

- The `0x0` determinant is `1` (the empty product); a `1x1` determinant is its
  sole entry.
- A zero pivot at `(k, k)` is resolved by swapping up a lower row with a nonzero
  entry in that column, tracking the sign flip. If the whole column below the
  pivot is zero the matrix is **singular** and the determinant is `0`.
- **Exact and unbounded**: it *never* overflows, in direct contrast to the
  `int64` `Matrix::determinant`, which surfaces `overflow` once intermediates or
  the result exceed the `~9.2e18` `int64` ceiling. `diag(10^10, 10^10, 10^10)`,
  which overflows `Matrix`, here returns the exact `10^30`.

Although the internal exact division threads a `Result` (so a hypothetical
divide-by-zero would propagate), the divisor is always a prior nonzero pivot or
the seed `1`, so in practice the only error `determinant()` can return is
`domain_error` from a non-square matrix.

## Error model

| Condition | Error |
| :--- | :--- |
| `from_rows` with a ragged row list | `MathError::domain_error` |
| `from_rows` with an empty row list | `MathError::domain_error` |
| `add` / `subtract` on mismatched shapes | `MathError::domain_error` |
| `multiply` with inner-dimension mismatch (`cols() != o.rows()`) | `MathError::domain_error` |
| `determinant` of a non-square matrix | `MathError::domain_error` |

There is **no `overflow` row**: `BigRational` arithmetic cannot wrap, so the
`overflow` that `Matrix` can raise is unreachable here. `scale`, `transpose`,
`identity`, `zero`, and `from_matrix` never fail on their inputs (a singular
matrix is not an error — its determinant is simply `0`).

## Worked examples

```cpp
import nimblecas.bigmatrix;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.ratpoly;
import nimblecas.matrix;
using namespace nimblecas;

// Exact BigRational helpers: n/d, and the integer n as n/1.
auto brat = [](std::int64_t n, std::int64_t d) {
    return BigRational::make(BigInt::from_i64(n), BigInt::from_i64(d)).value();
};
auto bi = [](std::int64_t v) { return BigRational::from_int(v); };

// Construction, shape, and the asserted accessor.
auto m = BigMatrix::from_rows({{bi(1), bi(2), bi(3)}, {bi(4), bi(5), bi(6)}}).value();
m.rows();                       // 2
m.cols();                       // 3
m.is_square();                  // false
m.at(1, 2) == bi(6);            // true
BigMatrix::from_rows({}).error();  // MathError::domain_error (empty list rejected)

// Identity / zero and A * I == A.
auto a = BigMatrix::from_rows({{bi(1), bi(2)}, {bi(3), bi(4)}}).value();
a.multiply(BigMatrix::identity(2)).value() == a;   // true

// Arithmetic stays exact (scaling by 1/2 keeps the fraction).
auto b = BigMatrix::from_rows({{bi(5), bi(6)}, {bi(7), bi(8)}}).value();
a.add(b).value();                                  // [[6, 8], [10, 12]]
a.scale(brat(1, 2)).value().at(0, 0) == brat(1, 2);// true
a.add(BigMatrix::from_rows({{bi(1), bi(2), bi(3)}}).value()).error();
                                                    // MathError::domain_error

// Product and transpose.
a.multiply(b).value();                             // [[19, 22], [43, 50]]
m.transpose().value();                             // 3x2 transpose

// Determinants: small, singular, and edge cases.
a.determinant().value() == bi(-2);                 // det [[1,2],[3,4]] = -2
BigMatrix::identity(4).determinant().value() == bi(1);            // det I = 1
BigMatrix::from_rows({{bi(1), bi(2)}, {bi(2), bi(4)}}).value()
    .determinant().value() == bi(0);               // rank-1 -> 0
BigMatrix::identity(0).determinant().value() == bi(1);           // empty product
BigMatrix::from_rows({{bi(1), bi(2), bi(3)}}).value()
    .determinant().error();                        // domain_error (non-square)

// A zero leading pivot forces a Bareiss row swap and a sign flip.
BigMatrix::from_rows({{bi(0), bi(1)}, {bi(1), bi(0)}}).value()
    .determinant().value() == bi(-1);              // det [[0,1],[1,0]] = -1

// Rational entries: det [[1/2,1/3],[1/4,1/5]] = 1/10 - 1/12 = 1/60 (exact).
BigMatrix::from_rows({{brat(1, 2), brat(1, 3)}, {brat(1, 4), brat(1, 5)}}).value()
    .determinant().value() == brat(1, 60);         // true

// Unbounded: diag(10^10)^3 = 10^30, far past the int64 ceiling that
// Matrix::determinant overflows on.
const std::int64_t k = 10000000000LL;              // 10^10
auto pow10 = [](std::uint64_t p) {
    return BigRational::from_bigint(BigInt::from_u64(10).pow(p));
};
auto d = BigMatrix::from_rows({{bi(k), bi(0), bi(0)},
                               {bi(0), bi(k), bi(0)},
                               {bi(0), bi(0), bi(k)}}).value();
d.determinant().value() == pow10(30);              // exact 10^30

// The int64 tier really does overflow on the same shape (2x2 already 10^20):
Matrix::from_rows({{Rational::from_int(k), Rational::from_int(0)},
                   {Rational::from_int(0), Rational::from_int(k)}}).value()
    .determinant().error();                        // MathError::overflow

// Promotion round-trip: an int64 Matrix lifts value-for-value into the big tier.
auto src = Matrix::from_rows({{Rational::from_int(3), Rational::make(1, 2).value()},
                              {Rational::make(-5, 4).value(), Rational::from_int(0)}}).value();
auto big = BigMatrix::from_matrix(src).value();
big.at(0, 1) == brat(1, 2);                        // 1/2 round-trips
big.at(1, 0) == brat(-5, 4);                       // -5/4 round-trips

// Cross-check: for small entries both tiers agree exactly.
auto small = Matrix::from_rows({{Rational::from_int(6), Rational::from_int(1), Rational::from_int(1)},
                                {Rational::from_int(4), Rational::from_int(-2), Rational::from_int(5)},
                                {Rational::from_int(2), Rational::from_int(8), Rational::from_int(7)}}).value();
BigMatrix::from_matrix(small).value().determinant().value() == bi(-306);  // true
```

## See also

- [`nimblecas.matrix`](matrix.md) — the `int64`-`Rational`, overflow-checked
  sibling this module promotes from; use it when values fit `int64`.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field that `Matrix`
  entries (and the `BigRational` numerators/denominators) descend from.
- [`nimblecas.core`](core.md) — the `Result` / `MathError` error model these
  operations return through.
- [Documentation hub](../Index.md)
