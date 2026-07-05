# `nimblecas.kronprod` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/kronprod/kronprod.cppm`

Kronecker product and related **structured matrix products**, done **exactly**
over the rationals `Q` (ROADMAP §7.2). Every operand is a
[`Matrix`](matrix.md) of [`Rational`](ratpoly.md) entries — reduced `int64`
fractions — so the Kronecker product `A ⊗ B`, the Kronecker sum `A ⊕ B`, the
direct sum `diag(A, B)`, the Hadamard product `A ∘ B`, and the `vec` / `unvec`
reshapings all carry **no rounding error at all**: `A ⊗ B` is the block matrix
it mathematically is, not a `double` that happens to be close.

This module exposes as a clean public API the same Kronecker semantics that
[`nimblecas.analysis`](analysis.md) uses internally for the Lyapunov / Stein
vec-trick, generalised from square operands to arbitrary shapes.

```cpp
import nimblecas.kronprod;
```

Depends on [`core`](core.md), [`matrix`](matrix.md), and [`ratpoly`](ratpoly.md).

## Honesty boundary

**Exact and complete.** There is no approximation anywhere in this module. Like
the rest of the engine, arithmetic is **overflow-checked** (Rule 32): every
entry product flows through `Rational`'s checked multiply, so an `int64`
numerator or denominator that would overflow surfaces as `MathError::overflow`
rather than silently wrapping. A result whose total row or column count would
overflow `std::size_t` (so the flat buffer could not be addressed) likewise
surfaces as `MathError::overflow` rather than allocating a wrapped-around
buffer. Shape violations — a non-square operand where a square one is required,
mismatched dimensions, or a non-column argument to `unvec` — surface as
`MathError::domain_error`.

## The `vec` convention

`vec` is **column-major**, matching the Lyapunov / Stein vec-trick inside
`nimblecas.analysis`: `vec(A)` stacks the columns of the `m × n` matrix `A` top
to bottom, so the entry `A(i, j)` lands at row `j * m + i` of the resulting
`(m·n) × 1` column vector. With this convention the mixed-product identity

```
vec(A X B) = (Bᵀ ⊗ A) vec(X)
```

holds, and `unvec(vec(A), rows(A))` reproduces `A`.

## API

### Products

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `kronecker_product` | `auto kronecker_product(const Matrix& a, const Matrix& b) -> Result<Matrix>` | For `A` of shape `m × n` and `B` of shape `p × q`, the `(m·p) × (n·q)` block matrix whose `(i, j)` block is `a_ij · B`; equivalently the entry at `(i·p + k, j·q + l)` is `A(i, j) · B(k, l)`. |
| `kronecker_sum` | `auto kronecker_sum(const Matrix& a, const Matrix& b) -> Result<Matrix>` | `A ⊕ B = A ⊗ Iₙ + Iₘ ⊗ B` for **square** `A` (`m × m`) and `B` (`n × n`); the result is `(m·n) × (m·n)`. A non-square operand yields `domain_error`. |
| `direct_sum` | `auto direct_sum(const Matrix& a, const Matrix& b) -> Result<Matrix>` | The block-diagonal `diag(A, B)`: `A` top-left, `B` bottom-right, zeros elsewhere, of shape `(rows(A)+rows(B)) × (cols(A)+cols(B))`. Never a shape error. |
| `hadamard_product` | `auto hadamard_product(const Matrix& a, const Matrix& b) -> Result<Matrix>` | The entrywise product `result(i, j) = A(i, j) · B(i, j)`. `A` and `B` must share the same shape, else `domain_error`. |

### Vectorisation

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `vec` | `auto vec(const Matrix& a) -> Result<Matrix>` | Column-major vectorisation: stacks the columns of `m × n` `A` into a `(m·n) × 1` column, with `A(i, j)` at row `j · m + i`. |
| `unvec` | `auto unvec(const Matrix& v, std::size_t rows) -> Result<Matrix>` | The inverse of `vec`: `v` must be a single column (`cols() == 1`) whose length is a multiple of `rows`, giving the `rows × (len/rows)` matrix `M` with `M(i, j) = v(j · rows + i, 0)`. |

## Identities (exact)

These hold entrywise over `Q` whenever the shapes conform, and are covered by
the module's tests:

| Identity | Condition |
| :--- | :--- |
| `(A ⊗ B)(C ⊗ D) = (A C) ⊗ (B D)` | inner dimensions of `A C` and `B D` conform |
| `(A ⊗ B)ᵀ = Aᵀ ⊗ Bᵀ` | always |
| `vec(A X B) = (Bᵀ ⊗ A) vec(X)` | `A X B` conforms |
| `tr(A ⊗ B) = tr(A) · tr(B)` | `A`, `B` square |

`A ⊗ B` is **not** commutative — the module never silently symmetrises; `A ⊗ B`
and `B ⊗ A` are only permutation-similar, and equal entrywise only by
coincidence.

## Error model

| Condition | Error |
| :--- | :--- |
| Non-square operand to `kronecker_sum` | `MathError::domain_error` |
| Dimension mismatch in `hadamard_product` | `MathError::domain_error` |
| Non-column `v`, or length not divisible by `rows`, in `unvec` | `MathError::domain_error` |
| An `int64` numerator or denominator computation wraps | `MathError::overflow` |
| A result's row/column count would overflow `std::size_t` | `MathError::overflow` |

## Examples

```cpp
import nimblecas.kronprod;
import nimblecas.matrix;
import nimblecas.ratpoly;
using namespace nimblecas;

auto ri  = [](std::int64_t v) { return Rational::from_int(v); };
auto mat = [](std::vector<std::vector<Rational>> r) {
    return Matrix::from_rows(std::move(r)).value();
};

// Kronecker product: [[1,2],[3,4]] (x) [[0,5],[6,7]] is 4x4 with block (i,j) = a_ij*B.
auto A = mat({{ri(1), ri(2)}, {ri(3), ri(4)}});
auto B = mat({{ri(0), ri(5)}, {ri(6), ri(7)}});
auto K = kronecker_product(A, B).value();          // 4x4, K(2,1) = 3*5 = 15

// Mixed-product identity: (A(x)B)(C(x)D) = (AC)(x)(BD).
auto C  = mat({{ri(3), ri(0)}, {ri(1), ri(-1)}});
auto lhs = kronecker_product(A, B).value()
               .multiply(kronecker_product(C, B).value()).value();
auto rhs = kronecker_product(A.multiply(C).value(),
                             B.multiply(B).value()).value();
// lhs == rhs

// Column-major vec / unvec round-trip.
auto M = mat({{ri(1), ri(2), ri(3)}, {ri(4), ri(5), ri(6)}});
auto v = vec(M).value();                            // [1,4,2,5,3,6]^T  (6x1)
auto back = unvec(v, 2).value();                    // == M

// Kronecker sum and direct sum.
auto S = kronecker_sum(A, C).value();               // A(x)I_2 + I_2(x)C, 4x4
auto D = direct_sum(A, mat({{ri(9)}})).value();     // diag(A, [9]), 3x3
```

## See also

- [`nimblecas.matrix`](matrix.md) — the exact `Matrix` these products build on.
- [`nimblecas.analysis`](analysis.md) — the Lyapunov / Stein solves whose
  internal vec-trick this module's `vec` convention matches.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field the entries
  live in.
- [Documentation hub](../Index.md)
```
