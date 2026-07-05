# `nimblecas.frobenius` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/frobenius/frobenius.cppm`

The **Frobenius / rational canonical form (RCF)** of a square matrix `A` over the
rationals `Q` (ROADMAP §7.10). The RCF is the block-diagonal matrix

```
RCF(A) = diag( C(f_1), C(f_2), ..., C(f_k) )
```

where `f_1 | f_2 | ... | f_k` are the **invariant factors** of `A` (monic
polynomials over `Q[x]`, each dividing the next, whose product is the
characteristic polynomial and whose last member `f_k` is the minimal polynomial)
and `C(f)` is the companion matrix of `f`.

```cpp
import nimblecas.frobenius;
```

Depends on [`core`](core.md), [`matrix`](matrix.md) and [`ratpoly`](ratpoly.md).

## Honesty boundary — exact over `Q`, and why that matters

Unlike the **Jordan** form, the rational canonical form is **exact over `Q`**:
it never needs the eigenvalues, so no field extension, no algebraic numbers, and
no floating point are involved — the entire computation stays inside `Q[x]`. A
matrix such as `[[0,-1],[1,0]]` has characteristic polynomial `x^2 + 1` with
eigenvalues `±i` that are not even real, yet its RCF is computed **exactly** as
the companion matrix `[[0,-1],[1,0]]` with the single invariant factor
`x^2 + 1`. This is the central advantage of the RCF over the Jordan form and the
reason this module exists.

Every result is either **exact over `Q`** or an honest `MathError` (Rule 32).
A non-square matrix is a `domain_error`; any `int64` numerator/denominator
overflow surfaced by the underlying `Rational` / `RationalPoly` arithmetic is
propagated as `overflow`.

**The transforming matrix `P` is NOT returned.** The RCF satisfies
`RCF(A) = P^{-1} A P` for some change-of-basis `P`, but computing `P` exactly is
a separate, harder task; returning a plausible-but-wrong `P` would violate the
honesty invariant, so it is deliberately omitted rather than faked. This module
returns the canonical form and the invariant factors — both exact — and nothing
that is merely approximate.

## How it works — Smith normal form over `Q[x]`

The invariant factors are read off the **Smith normal form (SNF)** of the
characteristic matrix `x*I - A`, a matrix of polynomials over the principal
ideal domain `Q[x]`. `x*I - A` is reduced by `Q[x]` row and column operations —
the polynomial-degree Euclidean valuation drives pivot selection (lowest-degree
nonzero entry), division-with-remainder reduces the rest of the pivot row and
column, and the standard gcd-combination step forces the divisibility chain
`d_1 | d_2 | ... | d_n` — to a diagonal `diag(d_1, ..., d_n)` with each `d_i`
monic. Because SNF preserves the determinant up to a unit and each `d_i` is
normalised monic, the product of the `d_i` equals the monic characteristic
polynomial exactly. The non-constant `d_i` are the invariant factors; the unit
entries `d_i = 1` are the trivial factors and are dropped.

## Companion-matrix convention

`companion_matrix(p)` uses the **right-column** companion form of the monic
polynomial `x^n + c_{n-1} x^{n-1} + ... + c_0`: ones on the sub-diagonal and the
negated coefficients `-c_0, ..., -c_{n-1}` down the last column. Its
characteristic polynomial is exactly that monic polynomial, so a companion
(cyclic) matrix is already in rational canonical form and is returned unchanged
by `rational_canonical_form`.

```
             [ 0  0  ...  0  -c_0     ]
             [ 1  0  ...  0  -c_1     ]
   C(p)  =   [ 0  1  ...  0  -c_2     ]
             [ ...           ...      ]
             [ 0  0  ...  1  -c_{n-1} ]
```

For example `C(x^2 + 1) = [[0,-1],[1,0]]` and `C(x^2 - 5x + 6) = [[0,-6],[1,5]]`.

## API

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `invariant_factors` | `auto invariant_factors(const Matrix& a) -> Result<std::vector<RationalPoly>>` | The invariant factors `f_1 \| f_2 \| ... \| f_k` in ascending order (`f_1` smallest degree, `f_k` the minimal polynomial), each monic. Their product is the characteristic polynomial, so their degrees sum to `n`. The 0×0 matrix yields an empty list. Non-square ⇒ `domain_error`. |
| `minimal_polynomial` | `auto minimal_polynomial(const Matrix& a) -> Result<RationalPoly>` | The minimal polynomial of `A` — the monic annihilating generator, equal to the last (largest) invariant factor. The 0×0 matrix yields the constant `1`. Non-square ⇒ `domain_error`. |
| `companion_matrix` | `auto companion_matrix(const RationalPoly& p) -> Result<Matrix>` | The right-column companion matrix of the monic form of `p` (degree `≥ 1`). A non-monic `p` is normalised to monic first; the zero or a constant (degree 0) polynomial ⇒ `domain_error`. |
| `rational_canonical_form` | `auto rational_canonical_form(const Matrix& a) -> Result<Matrix>` | The block-diagonal RCF `diag(C(f_1), ..., C(f_k))` over the invariant factors. Exact over `Q`. The 0×0 matrix maps to itself. Non-square ⇒ `domain_error`. Does **not** return the transforming matrix `P`. |

## Error model

| Condition | Error |
| :--- | :--- |
| Non-square input to `invariant_factors` / `minimal_polynomial` / `rational_canonical_form` | `MathError::domain_error` |
| Zero or constant polynomial passed to `companion_matrix` | `MathError::domain_error` |
| An `int64` numerator or denominator computation wraps in the underlying `Rational` / `RationalPoly` arithmetic | `MathError::overflow` |

## Examples

```cpp
import nimblecas.frobenius;
import nimblecas.matrix;
import nimblecas.ratpoly;
using namespace nimblecas;

auto ri  = [](std::int64_t v) { return Rational::from_int(v); };
auto mat = [](std::vector<std::vector<Rational>> r) {
    return Matrix::from_rows(std::move(r)).value();
};

// diag(2,2,3): invariant factors (x-2) and (x-2)(x-3) = x^2 - 5x + 6.
auto A = mat({{ri(2), ri(0), ri(0)},
              {ri(0), ri(2), ri(0)},
              {ri(0), ri(0), ri(3)}});
auto fs = invariant_factors(A).value();          // [ x-2, x^2-5x+6 ]
auto mp = minimal_polynomial(A).value();         // x^2 - 5x + 6
auto rc = rational_canonical_form(A).value();    // [[2,0,0],[0,0,-6],[0,1,5]]

// x^2 + 1 has no rational (indeed no real) eigenvalues, yet the RCF is exact:
auto R = mat({{ri(0), ri(-1)}, {ri(1), ri(0)}});
auto rcf = rational_canonical_form(R).value();   // == R itself (companion of x^2+1)
```

## See also

- [`nimblecas.eigen`](eigen.md) — the Faddeev-LeVerrier characteristic
  polynomial and rational eigenvalues (the product of the invariant factors
  equals that characteristic polynomial).
- [`nimblecas.matrix`](matrix.md) — the exact `Rational` matrix type the form is
  returned in.
- [`nimblecas.ratpoly`](ratpoly.md) — the `Q[x]` polynomials the invariant
  factors live in.
- [Documentation hub](../Index.md)
```
