# `nimblecas.bigfrobenius` — Invariant factors and minimal polynomial, unbounded (Reference)

**Author:** Olumuyiwa Oluwasanmi

Source: `src/bigfrobenius/bigfrobenius.cppm`

The **unbounded, overflow-free** counterpart to [`nimblecas.frobenius`](frobenius.md).
Where that module carries `int64` `Rational` / `RationalPoly` coefficients and
reports `MathError::overflow` the moment a numerator or denominator
saturates, this module carries `BigRational` / [`BigRationalPoly`](bigratpoly.md)
coefficients over a [`BigMatrix`](bigmatrix.md) and has **no such ceiling**.
It exists because the `int64` `nimblecas.frobenius::minimal_polynomial`
overflows on the rational multiplication matrices that arise in
splitting-field primitive-element searches: the intermediate Smith-normal-form
arithmetic (and the final coefficients themselves) can dwarf the `~9.22e18`
`int64` envelope. On the bignum tier those values are the exact elements of
**Q** they mathematically are, so the search never has to abandon a candidate
because the arithmetic ran out of room.

The engine is identical to the `int64` tier: the **Smith normal form (SNF)**
of the characteristic matrix `x*I - A` over the principal ideal domain
`Q[x]`. Reducing `x*I - A` by `Q[x]` row/column operations (the
polynomial-degree Euclidean valuation drives the pivoting) to a diagonal
`diag(d_1, ..., d_n)` with `d_1 | d_2 | ... | d_n` (each monic) exposes the
invariant factors as the non-constant `d_i`. Their product is the
characteristic polynomial, the divisibility chain is what makes the result
canonical, and the last (largest) factor is the minimal polynomial.

```cpp
import nimblecas.bigfrobenius;
```

Depends on [`core`](core.md), `bigrational`, [`bigratpoly`](bigratpoly.md),
and [`bigmatrix`](bigmatrix.md).

## The honesty boundary

- **A non-square matrix is a `domain_error`.**
- **Exact over an unbounded rational tier — `MathError::overflow` can never
  be produced.** `BigRational`'s `add` / `subtract` / `multiply` are
  infallible (arbitrary precision cannot overflow), so the row/column
  operations built on them (`row_sub_mul`, `col_sub_mul`, `row_add`) are
  likewise infallible.
- **The only fallible primitives are `BigRationalPoly::divide` (zero
  divisor) and `monic` (zero leading coefficient).** Here the divisor is
  always the current nonzero pivot, so neither can actually fire, but
  `Result` is threaded through for an honest, uniform surface.
- This module returns the invariant factors / minimal polynomial only,
  **not** any change-of-basis matrix.

## API summary

```cpp
// The invariant factors f_1 | f_2 | ... | f_k of A: monic polynomials over Q[x], each
// dividing the next, in ascending order (f_1 smallest degree, f_k the minimal
// polynomial). Their product is the characteristic polynomial. NEVER overflows.
// The 0x0 matrix has no invariant factors (empty list). Requires a square matrix.
[[nodiscard]] auto invariant_factors(const BigMatrix& a) -> Result<std::vector<BigRationalPoly>>;

// The minimal polynomial of A: the monic generator of the ideal of polynomials that
// annihilate A, equal to the last (largest) invariant factor. For the 0x0 matrix the
// empty-product convention gives the constant polynomial 1. Requires a square matrix.
[[nodiscard]] auto minimal_polynomial(const BigMatrix& a) -> Result<BigRationalPoly>;
```

## Error model

| Condition | Error |
| :--- | :--- |
| `invariant_factors` / `minimal_polynomial` of a non-square matrix | `MathError::domain_error` |

There is **no `overflow` row**: `BigRational` / `BigRationalPoly` arithmetic
cannot wrap, so the `overflow` the `int64` [`frobenius`](frobenius.md) can
raise on saturated multiplication-matrix entries is unreachable here. The
`0x0` matrix is a valid input (empty invariant-factor list; minimal
polynomial `1`), not an error.

## Worked example

```cpp
import nimblecas.bigfrobenius;
import nimblecas.bigmatrix;
import nimblecas.bigratpoly;
import nimblecas.bigrational;
using namespace nimblecas;

auto bi  = [](std::int64_t v) { return BigRational::from_int(v); };
auto big = [](std::string_view s) { return BigRational::from_string(s).value(); };
auto mat = [](std::vector<std::vector<BigRational>> rows) {
    return BigMatrix::from_rows(std::move(rows)).value();
};

// A single 2x2 Jordan block is non-diagonalizable: its minimal polynomial is the
// full (x - 5)^2 = x^2 - 10x + 25, not the simple (x - 5).
const BigMatrix jordan = mat({{bi(5), bi(1)}, {bi(0), bi(5)}});
auto mp = minimal_polynomial(jordan).value();
mp.coefficient(0) == bi(25) && mp.coefficient(1) == bi(-10);  // x^2 - 10x + 25

// Past the int64 ceiling: diag(B, B) with B = 10^20 is the scalar matrix B*I; its
// minimal polynomial is (x - B). The coefficient -10^20 is far beyond int64's
// ~9.22e18 ceiling, exact only in this tier.
const BigRational B = big("100000000000000000000");  // 10^20
const BigMatrix scalar = mat({{B, bi(0)}, {bi(0), B}});
auto r = minimal_polynomial(scalar).value();
r.coefficient(0) == big("-100000000000000000000");   // -1e20, exact

// A non-square matrix is a domain_error for both entry points.
const BigMatrix wide = mat({{bi(1), bi(2), bi(3)}, {bi(4), bi(5), bi(6)}});
invariant_factors(wide).error() == MathError::domain_error;
minimal_polynomial(wide).error() == MathError::domain_error;
```

## See also

- [`nimblecas.frobenius`](frobenius.md) — the `int64`-`Rational`,
  overflow-checked sibling this module big-backs; use it when the
  multiplication-matrix entries fit `int64`.
- [`nimblecas.bigmatrix`](bigmatrix.md) — the unbounded `BigRational` matrix
  this module analyses (the input type).
- [`nimblecas.bigratpoly`](bigratpoly.md) — the unbounded `Q[x]` type the
  invariant factors / minimal polynomial are returned in.
- [`nimblecas.bigrational`](bigrational.md) — the exact unbounded fraction
  field underlying every layer.
- [Documentation hub](../Index.md)
