# `nimblecas.roots` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/roots/roots.cppm`

A first slice of **analytical equation solving** (ROADMAP §7.21). Given a
polynomial `p` over the rationals `Q[x]`, `rational_roots` returns every
**rational** root of `p` together with its multiplicity, using the **rational
root theorem**. The polynomial is scaled to an integer polynomial
`a_n x^n + … + a_0` by clearing denominators; any rational root written `p/q` in
lowest terms must then have `p` dividing the lowest nonzero coefficient and `q`
dividing the leading coefficient `a_n`. Each candidate `±(divisor of a_0)/(divisor
of a_n)` is tested by exact evaluation ([`evaluate`](#evaluate), Horner), and a
confirmed root `r` is **deflated** out by dividing `(x - r)` repeatedly over
[`RationalPoly`](ratpoly.md) to recover its multiplicity.

```cpp
import nimblecas.roots;
```

Depends on [`core`](core.md) and [`ratpoly`](ratpoly.md).

## Scope

Only **rational** roots are returned. Irrational and complex roots — radical /
Cardano (cubic) / Ferrari (quartic) closed forms, and a symbolic `RootOf` for the
unsolvable-in-radicals case — are a **planned extension** and are not produced
here. A polynomial such as `x^2 + 1` (complex roots) or `x^2 - 2` (irrational
roots) therefore reports **no roots** — an empty vector, not an error.

## The overflow contract

Following the rest of the engine, arithmetic is **exact** and
**overflow-checked** (Rule 32): clearing denominators and forming candidate
magnitudes use checked `int64` multiplication and reject an `int64` boundary —
such as `INT64_MIN`, whose magnitude is unrepresentable — with
`MathError::overflow` rather than wrapping. This mirrors the checked-`gcd` guard
in [`ratpoly`](ratpoly.md).

## API

```cpp
[[nodiscard]] auto evaluate(const RationalPoly& p, const Rational& x)
    -> Result<Rational>;

[[nodiscard]] auto rational_roots(const RationalPoly& p)
    -> Result<std::vector<std::pair<Rational, std::int64_t>>>;
```

### `evaluate`

Evaluate `p` at `x` by **Horner's method**, folding from the highest-degree
coefficient down (`acc = acc·x + c_i`). The zero polynomial evaluates to `0`.
Fails only with `MathError::overflow` when the exact `int64` rational arithmetic
wraps.

### `rational_roots`

All **distinct** rational roots of `p`, each paired with its multiplicity
(`>= 1`). Roots are returned in **no particular order**. Only rational roots are
found (see [Scope](#scope)).

The algorithm:

1. Reject the zero polynomial (`MathError::domain_error`); a nonzero constant has
   no roots and yields an empty vector.
2. **Clear denominators** to integer coefficients `a_0 … a_n` (the LCM of the
   denominators times each coefficient), overflow-checked.
3. Find the **lowest nonzero coefficient** `a_low`. If `a_low` is not `a_0`
   (i.e. `a_0 = 0`), then `0` is a root of `x^k · (…)`, added as an explicit
   candidate; the numerator divisors are taken from `|a_low|`.
4. Enumerate candidates `±p/q` for every positive divisor `p` of `|a_low|` and
   `q` of `|a_n|` (the rational root theorem set).
5. For each candidate, test with `evaluate`; on a zero, **deflate** `(x - r)`
   out of the working polynomial repeatedly, counting the divisions as the
   multiplicity. Testing against the deflated polynomial makes non-reduced
   duplicate candidates (e.g. `2/2` and `1/1`) evaluate nonzero the second time,
   so no root is double-counted.

## Error model

| Condition | Error |
| :--- | :--- |
| `p` is the zero polynomial (every value is a root) | `MathError::domain_error` |
| Clearing denominators or a candidate magnitude exceeds `int64` (e.g. an `INT64_MIN` boundary) | `MathError::overflow` |
| An `int64` rational computation in `evaluate` / deflation wraps | `MathError::overflow` |

A nonzero constant, and any polynomial with no rational roots, return an **empty
vector** — not an error.

## Worked examples

From `tests/roots_tests.cpp` (integer polynomials given low-degree-first, the
`RationalPoly` convention: `coeffs[i]` is the coefficient of `x^i`):

```cpp
import nimblecas.roots;
import nimblecas.ratpoly;
using namespace nimblecas;

// (x-1)(x-2)(x+3) = x^3 - 7x + 6      coeffs {6, -7, 0, 1}
rational_roots(/* x^3 - 7x + 6 */).value();
//   -> { (1, 1), (2, 1), (-3, 1) }   three simple roots

// (x-1)^2 (x+2) = x^3 - 3x + 2        coeffs {2, -3, 0, 1}
rational_roots(/* x^3 - 3x + 2 */).value();
//   -> { (1, 2), (-2, 1) }           root 1 has multiplicity 2

// x^2 + 1                             coeffs {1, 0, 1}
rational_roots(/* x^2 + 1 */).value();
//   -> { }                           complex roots: none rational

// 2x^2 - 2 = 2(x-1)(x+1)             coeffs {-2, 0, 2}
rational_roots(/* 2x^2 - 2 */).value();
//   -> { (1, 1), (-1, 1) }           common factor 2 cleared first

// 2x - 1                              coeffs {-1, 2}
rational_roots(/* 2x - 1 */).value();
//   -> { (1/2, 1) }                  a fractional root p/q

// x^2 (x - 1)                         coeffs {0, 0, -1, 1}
rational_roots(/* x^3 - x^2 */).value();
//   -> { (0, 2), (1, 1) }            zero root with multiplicity

// degenerate inputs
rational_roots(RationalPoly{}).error();   // MathError::domain_error (zero poly)
rational_roots(/* 5 */).value();          // { }  nonzero constant: no roots
```

## See also

- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Q[x]` substrate (`Rational`,
  `RationalPoly`, division-with-remainder) this module tests and deflates over.
- [`nimblecas.polynomial`](polynomial.md) — the integer ring `Z[x]` and its
  square-free factorization.
- [`nimblecas.resultant`](resultant.md) — discriminant-based repeated-root
  detection over `Q[x]`.
- [Documentation hub](../Index.md)
