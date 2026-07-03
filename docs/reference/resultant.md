# `nimblecas.resultant` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/resultant/resultant.cppm`

The **resultant** and **discriminant** of polynomials over the rationals,
`Q[x]`. The resultant `res(A, B)` of two polynomials is the product of
`lc(A)^{deg B}`, `lc(B)^{deg A}` and all pairwise differences of their roots; it
**vanishes exactly when `A` and `B` share a factor** (a common root). The
discriminant `disc(A)` specialises this to a single polynomial and its
derivative, and so **vanishes exactly when `A` has a repeated root**. It is the
algebraic substrate of ROADMAP §7.17 — the subresultant PRS, multivariate GCD,
and specifically the Rothstein–Trager resultant that drives the logarithmic part
of rational-function integration (§7.19) — built entirely on the exact `Q[x]`
substrate of [`ratpoly`](ratpoly.md).

```cpp
import nimblecas.resultant;
```

Depends on [`core`](core.md) and [`ratpoly`](ratpoly.md).

## The overflow contract

Following the rest of the engine, every stage is **exact** and
**overflow-checked** (Rule 32): all arithmetic runs through the underlying
`Q[x]` operations, so an `int64` numerator or denominator that would overflow
surfaces as `MathError::overflow` rather than silently wrapping. No
`division_by_zero` arises from the resultant itself — the Euclidean descent only
divides by non-zero polynomials, and the zero-operand and constant cases return
directly — so `overflow` is the only failure mode, the same contract as
[`ratpoly`](ratpoly.md).

## The Euclidean remainder-sequence algorithm

Because the coefficient field is `Q`, the resultant is computed by folding the
sign and leading-coefficient bookkeeping of the classical recurrence into the
**Euclidean remainder sequence**:

```
res(A, B) = (-1)^{deg A * deg B} * lc(B)^{deg A - deg R} * res(B, R),   R = A mod B,
```

with the base case `res(A, c) = c^{deg A}` for a non-zero constant `c`. Each
descent step arranges `deg A >= deg B`, pays the `(-1)^{deg A * deg B}` swap
sign, multiplies in `lc(B)^{deg A - deg R}`, and recurses on `(B, R)` until the
second operand is a constant. Two cases short-circuit the descent:

- **A zero operand** — `res(A, B) = 0` when either `A` or `B` is the zero
  polynomial.
- **A common factor** — if any remainder `R` in the sequence vanishes then `A`
  and `B` share a factor, and `res(A, B) = 0`.

The resultant of two non-zero constants is the empty product `res(c, d) = 1`.

## The discriminant

The discriminant reduces to a resultant of `a` against its own derivative:

```
disc(a) = (-1)^{n(n-1)/2} / lc(a) * res(a, a'),   n = deg a.
```

It vanishes exactly when `a` has a repeated root (equivalently
`gcd(a, a') != 1`). A **constant or linear** polynomial has discriminant `1` by
convention (it has no repeated roots). For a quadratic `a x^2 + b x + c` the
formula reproduces the familiar `b^2 - 4ac`. The sign `(-1)^{n(n-1)/2}` is `-1`
exactly when `n ≡ 2` or `3 (mod 4)`, and the exact division by `lc(a)` is always
well-defined because `deg a >= 2` there.

## Public API

```cpp
[[nodiscard]] auto resultant(const RationalPoly& a, const RationalPoly& b)
    -> Result<Rational>;

[[nodiscard]] auto discriminant(const RationalPoly& a) -> Result<Rational>;
```

### `resultant(a, b)`

The resultant `res(a, b)` in `Q`. Returns `0` when `a` and `b` share a factor,
or when either operand is the zero polynomial; `res(constant, constant)` is the
empty product `1`. Fails `MathError::overflow` on an `int64` coefficient limit.

### `discriminant(a)`

The discriminant `disc(a) = (-1)^{n(n-1)/2} / lc(a) * res(a, a')`, `n = deg a`.
Returns `0` exactly when `a` has a repeated root; a constant or linear `a` has
discriminant `1` by convention. Fails `MathError::overflow` on an `int64`
coefficient limit.

## Examples

Worked from the tests (`tests/resultant_tests.cpp`). Inputs are built
low-degree-first from integer coefficients.

```cpp
import nimblecas.resultant;
import nimblecas.ratpoly;
using namespace nimblecas;

// Build x^k-style inputs from integer coefficients (low degree first).
auto ipoly = [](std::vector<std::int64_t> c) {
    return RationalPoly::from_polynomial(Polynomial{std::move(c)});
};

// Linear pairs: res(x - 2, x - 5) = -3 (the Sylvester determinant).
auto r1 = resultant(ipoly({-2, 1}), ipoly({-5, 1})).value();   // -3

// Hand-checked small cases.
auto r2 = resultant(ipoly({-1, 0, 1}), ipoly({-2, 1})).value(); // res(x^2 - 1, x - 2) = 3
auto r3 = resultant(ipoly({1, 0, 1}), ipoly({0, 2})).value();   // res(x^2 + 1, 2x) = 4

// A shared root makes the resultant vanish:
// x^2 - 1 and x - 1 share the root 1.
auto r4 = resultant(ipoly({-1, 0, 1}), ipoly({-1, 1})).value(); // 0 (common factor)

// Multiplicativity in the second argument: res(A, B*C) = res(A, B) * res(A, C).
auto a  = ipoly({1, 0, 1});                 // x^2 + 1
auto b  = ipoly({-1, 1});                   // x - 1
auto c  = ipoly({-2, 1});                   // x - 2
auto bc = b.multiply(c).value();
auto lhs = resultant(a, bc).value();
auto rhs = resultant(a, b).value().multiply(resultant(a, c).value()).value();
// lhs == rhs == 10.

// Discriminants: repeated roots vanish, complex roots go negative.
auto d1 = discriminant(ipoly({-1, 0, 1})).value();     // disc(x^2 - 1) = 4
auto d2 = discriminant(ipoly({1, 0, 1})).value();      // disc(x^2 + 1) = -4
auto d3 = discriminant(ipoly({1, -2, 1})).value();     // disc((x - 1)^2) = 0
auto d4 = discriminant(ipoly({0, -1, 0, 1})).value();  // disc(x^3 - x) = 4
```

## Relationship to integration

The resultant is the algebraic substrate for the **subresultant PRS** and
**multivariate GCD** (ROADMAP §7.17), and specifically for the **Rothstein–Trager
resultant**

```
R(t) = res_x(D, A - t*D'),
```

whose roots supply the constant multipliers of the logarithms in the logarithmic
part of `int A(x)/D(x) dx` with `D` square-free (ROADMAP §7.19). That step
consumes exactly the square-free-denominator integrand that
[`ratint`](ratint.md) leaves behind, making `resultant` the planned next piece
after Hermite reduction.

## See also

- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Q[x]` substrate (`Rational`,
  `RationalPoly`, division-with-remainder, monic Euclidean gcd, derivative) this
  module is built on.
- [`nimblecas.ratint`](ratint.md) — Hermite reduction, whose square-free
  logarithmic integrand the Rothstein–Trager resultant consumes.
- [`nimblecas.pfd`](pfd.md) — square-free partial-fraction decomposition, the
  other `ratpoly` consumer.
- [Documentation hub](../Index.md)
