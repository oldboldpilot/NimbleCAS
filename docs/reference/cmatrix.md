# `nimblecas.cmatrix` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/cmatrix/cmatrix.cppm`

Dense matrices whose entries are exact complex numbers over the rationals — the
**Gaussian rationals** `Q + Qi` (ROADMAP §7.2), the complex analogue of
[`Matrix`](matrix.md). Each entry is a [`Complex`](complex.md), itself a pair of
canonical [`Rational`](ratpoly.md)s, and every arithmetic step flows through
`Complex`'s overflow-checked, railway-oriented operations. The whole layer is
therefore **exact**: a conjugate transpose, a matrix product, or a unitarity
check is the number it mathematically is, **never a floating-point
approximation**. The honesty boundary is the same one `Complex` draws — anything
requiring an irrational quantity (a genuine `|z|`, an eigenvalue over `R`, an
SVD) lives in a later numeric layer, not here; and every `int64` numerator or
denominator that would wrap surfaces as `MathError::overflow` rather than
silently overflowing.

The API deliberately mirrors the real `Matrix` so downstream code reads the same
way, and adds the complex-specific structure ROADMAP §7.2 needs: `conjugate`,
`adjoint` (`Aᴴ`, the conjugate transpose), and the Hermitian / skew-Hermitian /
unitary / normal predicates that classify a matrix's symmetry. Those predicates
are exactly the vocabulary the quantum-gate layer (ROADMAP §7.15) is built on: a
gate is a unitary matrix, an observable is Hermitian, a rotation generator is
skew-Hermitian. Because the entries are exact, each predicate returns a
**definite** `bool`, not a tolerance-based guess.

```cpp
import nimblecas.cmatrix;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), and
[`complex`](complex.md).

## The overflow contract

Following the rest of the engine, arithmetic is **exact** and
**overflow-checked** (Rule 32): every entry operation flows through `Complex`'s
checked add / subtract / multiply / conjugate, so an `int64` numerator or
denominator that would overflow surfaces as `MathError::overflow` rather than
silently wrapping. Dimension violations surface as `MathError::domain_error`,
and the predicates that require a square matrix reject a non-square one with
`domain_error`. The default-constructed matrix is the empty 0×0 matrix — a
valid, if degenerate, value. Storage is dense row-major: entry `(i, j)` lives at
`data_[i * cols_ + j]`.

## `ComplexMatrix` — a dense `rows × cols` grid of exact `Complex` entries

### Construction

| Constructor / factory | Signature | Notes |
| :--- | :--- | :--- |
| default | `ComplexMatrix()` | The empty 0×0 matrix. |
| `from_rows` | `static auto from_rows(std::vector<std::vector<Complex>> rows) -> Result<ComplexMatrix>` | Build from a list of rows. Every row must have the same length, else `domain_error` (ragged). An empty list is rejected with `domain_error`, and so is a list whose rows have no columns — a `ComplexMatrix` always has at least one entry. |
| `identity` | `static auto identity(std::size_t n) -> ComplexMatrix` | The `n × n` identity (`1 + 0i` on the diagonal, `0 + 0i` elsewhere). `identity(0)` is the 0×0 matrix. |
| `zero` | `static auto zero(std::size_t rows, std::size_t cols) -> ComplexMatrix` | The `rows × cols` all-zero matrix (every entry `0 + 0i`). |

### Accessors

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `rows` | `auto rows() const noexcept -> std::size_t` | Row count. |
| `cols` | `auto cols() const noexcept -> std::size_t` | Column count. |
| `is_square` | `auto is_square() const noexcept -> bool` | `true` when `rows() == cols()`. |
| `at` | `auto at(std::size_t i, std::size_t j) const -> const Complex&` | Entry `(i, j)`. Asserted in-range — callers hold indices below `rows()` / `cols()`. |
| `operator==` | `auto operator==(const ComplexMatrix& o) const noexcept -> bool` | Same shape and exact entry-vector equality (canonical `Complex` parts make this exact). |
| `is_equal` | `auto is_equal(const ComplexMatrix& o) const noexcept -> bool` | Alias for `operator==`. |
| `to_string` | `auto to_string() const -> std::string` | Human-readable rendering, e.g. `[[1 + i, 2], [-i, 3]]`. |

### Arithmetic (overflow-checked, exact)

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `add` | `auto add(const ComplexMatrix& o) const -> Result<ComplexMatrix>` | Entrywise sum; dimensions must match, else `domain_error`. |
| `subtract` | `auto subtract(const ComplexMatrix& o) const -> Result<ComplexMatrix>` | Entrywise difference; dimensions must match, else `domain_error`. |
| `scale` | `auto scale(const Complex& s) const -> Result<ComplexMatrix>` | Multiply every entry by the complex scalar `s`. |
| `multiply` | `auto multiply(const ComplexMatrix& o) const -> Result<ComplexMatrix>` | Matrix product (`this` is `m × k`, `o` is `k × n`). Fails `domain_error` when the inner dimensions disagree (`cols() != o.rows()`); propagates any `Complex`-op error. |

Each returns `MathError::overflow` if any entry computation wraps `int64`.

### Transposition / conjugation

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `transpose` | `auto transpose() const -> Result<ComplexMatrix>` | The `cols × rows` plain transpose `Aᵀ` (positions swap, **no** conjugation). Never fails, but returns `Result` for a uniform surface. |
| `conjugate` | `auto conjugate() const -> Result<ComplexMatrix>` | The entrywise complex conjugate `Ā` (same shape, no transpose). |
| `adjoint` | `auto adjoint() const -> Result<ComplexMatrix>` | The adjoint / conjugate transpose `Aᴴ = (Ā)ᵀ` — the key complex operation, and the map under which `Hermitian`, `unitary`, and `normal` are all defined. |

### Structural predicates (require a square matrix, else `domain_error`)

Each compares exact Gaussian-rational entries, so the answer is a **definite**
`bool` — never a tolerance decision.

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `is_hermitian` | `auto is_hermitian() const -> Result<bool>` | `A == Aᴴ` (self-adjoint). Real symmetric matrices are the real special case. |
| `is_skew_hermitian` | `auto is_skew_hermitian() const -> Result<bool>` | `Aᴴ == −A`. (`i` times a Hermitian matrix is skew-Hermitian.) |
| `is_unitary` | `auto is_unitary() const -> Result<bool>` | `Aᴴ·A == I` (columns orthonormal under the Hermitian inner product). These are exactly the quantum gates of ROADMAP §7.15. |
| `is_normal` | `auto is_normal() const -> Result<bool>` | `A·Aᴴ == Aᴴ·A` (commutes with its own adjoint). Hermitian, skew-Hermitian, and unitary matrices are all normal. |

## Error model

| Condition | Error |
| :--- | :--- |
| Empty list, rows with no columns, or ragged rows in `from_rows` | `MathError::domain_error` |
| Dimension mismatch (`add` / `subtract`) | `MathError::domain_error` |
| Inner-dimension mismatch (`multiply`, `cols() != o.rows()`) | `MathError::domain_error` |
| Non-square where a square is required (`is_hermitian` / `is_skew_hermitian` / `is_unitary` / `is_normal`) | `MathError::domain_error` |
| An `int64` numerator or denominator computation wraps | `MathError::overflow` |

`transpose`, `conjugate`, and `adjoint` cannot fail on shape (they are total on
any matrix) but return `Result` to stay uniform with the railway; they still
propagate a `Complex`-level `overflow` from conjugation. `at` is asserted, not
`Result`: an out-of-range index is a caller bug, not a runtime error.

## Worked examples

```cpp
import nimblecas.cmatrix;
import nimblecas.complex;
import nimblecas.ratpoly;
using namespace nimblecas;

auto c  = [](std::int64_t re, std::int64_t im) {  // re + im·i, Gaussian integer
    return Complex::make(Rational::from_int(re), Rational::from_int(im));
};
auto cm = [](std::vector<std::vector<Complex>> rows) {
    return ComplexMatrix::from_rows(std::move(rows)).value();
};

// Adjoint = conjugate transpose. A = [[1+i, 2], [-i, 3]].
const auto a = cm({{c(1, 1), c(2, 0)}, {c(0, -1), c(3, 0)}});
a.conjugate().value();   // [[1-i, 2], [i, 3]]   (entrywise, shape preserved)
a.transpose().value();   // [[1+i, -i], [2, 3]]  (positions swap, no conjugation)
a.adjoint().value();     // [[1-i, i], [2, 3]]   (conjugate then transpose)

// Adjoint of a 2x3 is 3x2; purely-real entries are unchanged, positions transpose.
const auto r = cm({{c(1, 1), c(2, -1), c(0, 3)}, {c(4, 0), c(0, -5), c(6, 1)}});
auto radj = r.adjoint().value();
radj.rows();             // 3
radj.at(2, 1);           // 6 - i   (from entry (1,2) = 6 + i, conjugated)

// The Pauli matrices: entries in {0, ±1, ±i}, so the gate identities stay exact.
const auto X = cm({{c(0, 0), c(1, 0)}, {c(1, 0), c(0, 0)}});   // [[0,1],[1,0]]
const auto Y = cm({{c(0, 0), c(0, -1)}, {c(0, 1), c(0, 0)}});  // [[0,-i],[i,0]]
const auto Z = cm({{c(1, 0), c(0, 0)}, {c(0, 0), c(-1, 0)}});  // [[1,0],[0,-1]]

X.is_hermitian().value();   // true
X.is_unitary().value();     // true  (XᴴX = I)
X.is_normal().value();      // true
Y.is_hermitian().value();   // true  (Yᴴ == Y despite the imaginary entries)
Y.is_unitary().value();     // true

// Pauli algebra cross-check of complex multiply: X·Y = iZ = [[i,0],[0,-i]].
auto xy = X.multiply(Y).value();
xy == cm({{c(0, 1), c(0, 0)}, {c(0, 0), c(0, -1)}});           // true

// Skew-Hermitian: iX = [[0,i],[i,0]] has adjoint −(iX).
const auto iX = cm({{c(0, 0), c(0, 1)}, {c(0, 1), c(0, 0)}});
iX.is_skew_hermitian().value();  // true
iX.is_hermitian().value();       // false

// A real shear is neither unitary nor Hermitian.
const auto s = cm({{c(1, 0), c(1, 0)}, {c(0, 0), c(1, 0)}});   // [[1,1],[0,1]]
s.is_unitary().value();     // false
s.is_hermitian().value();   // false

// Error surfaces.
ComplexMatrix::from_rows({{c(1, 0), c(2, 0)}, {c(3, 0)}}).error();  // domain_error (ragged)
ComplexMatrix::from_rows({}).error();                              // domain_error (empty)

// 2x2 times 3x2: inner dimensions 2 != 3.
const auto b = cm({{c(1, 0), c(0, 0)}, {c(0, 0), c(1, 0)}, {c(1, 0), c(1, 0)}});
ComplexMatrix::identity(2).multiply(b).error();  // domain_error

// Predicates reject a non-square matrix.
cm({{c(1, 0), c(2, 0), c(3, 0)}, {c(4, 0), c(5, 0), c(6, 0)}})
    .is_hermitian().error();                     // domain_error
```

## See also

- [`nimblecas.complex`](complex.md) — the exact Gaussian-rational field the
  entries live in.
- [`nimblecas.matrix`](matrix.md) — the real analogue over `Q`, whose API this
  module tracks.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` the real and
  imaginary parts are built from.
- [Documentation hub](../Index.md)
