# `nimblecas.bigresultant` — Resultant and discriminant over the unbounded rationals (Reference)

**Author:** Olumuyiwa Oluwasanmi

Source: `src/bigresultant/bigresultant.cppm`

The **unbounded, overflow-free** counterpart to [`nimblecas.resultant`](resultant.md).
Where that module carries `int64` `Rational` coefficients and surfaces a
saturated numerator or denominator as `MathError::overflow`, this one carries
`BigRational`-coefficient [`BigRationalPoly`](bigratpoly.md) operands and has
**no such ceiling**: it is the field-norm substrate for a bignum Trager
splitting-field path, where the intermediate resultant values routinely
exceed the `int64` envelope.

The resultant `res(A, B)` of two polynomials is the product of
`lc(A)^deg B`, `lc(B)^deg A`, and all pairwise differences of their roots; it
vanishes exactly when `A` and `B` share a root (a common factor). This module
computes it over the coefficient field **Q** via the Euclidean remainder
sequence, using the recurrence

```
res(A, B) = (-1)^{deg A * deg B} * lc(B)^{deg A - deg R} * res(B, R),   R = A mod B,
```

with the base case `res(A, c) = c^{deg A}` for a constant `c`. Because
`BigRational`'s magnitude-combining operations and the `BigRationalPoly`
operations built on them are **infallible** (arbitrary precision cannot
overflow), the sign folding and the `lc`-power accumulation never fail. The
only fallible step is the Euclidean division, which can only report a zero
divisor — never overflow. **`MathError::overflow` is never produced**; that
is the entire point of the bignum tier.

```cpp
import nimblecas.bigresultant;
```

Depends on [`core`](core.md), `bigint`, `bigrational`, and
[`bigratpoly`](bigratpoly.md).

## The honesty boundary

- **Exact over an unbounded rational tier — `MathError::overflow` can never
  be produced.** Every step (sign folding, `lc`-power accumulation, the
  Euclidean division) works over `BigRational` / `BigRationalPoly`, both of
  which cannot overflow.
- The only failure channel either function has is inherited from
  `BigRationalPoly::divide`'s zero-divisor check, and in both `resultant` and
  `discriminant` the divisor is always provably nonzero at the point it is
  invoked (`deg b > 0` inside the descent; `lc(a) != 0` for a degree-`>= 2`
  polynomial in `discriminant`). `Result` is threaded through for a uniform,
  defensive surface.

## API summary

```cpp
// Resultant res(a, b) in Q. Zero when a and b share a factor (or either is the
// zero polynomial). res(constant, constant) is 1 (the empty product). Never overflows.
[[nodiscard]] auto resultant(const BigRationalPoly& a, const BigRationalPoly& b)
    -> Result<BigRational>;

// Discriminant disc(a) = (-1)^{n(n-1)/2} / lc(a) * res(a, a'), n = deg a. Vanishes
// exactly when a has a repeated root. A constant or linear polynomial has discriminant 1.
[[nodiscard]] auto discriminant(const BigRationalPoly& a) -> Result<BigRational>;
```

## Error model

`resultant` and `discriminant` both return `Result<BigRational>`, but under
normal, well-formed inputs neither actually reports an error: the zero
polynomial and constants are handled as explicit conventions (`0` and `1`
respectively), not errors, and the only theoretically-fallible internal
step (a `BigRationalPoly::divide`) is always invoked with a provably nonzero
divisor. There is **no `overflow` row**: `BigRational` arithmetic cannot wrap,
so the `overflow` the `int64` [`resultant`](resultant.md) can raise is
unreachable here.

## Worked example

```cpp
import nimblecas.bigresultant;
import nimblecas.bigratpoly;
import nimblecas.bigrational;
using namespace nimblecas;

auto bi = [](std::int64_t v) { return BigRational::from_int(v); };
auto bigpoly = [](std::initializer_list<std::int64_t> cs) {
    std::vector<BigRational> v;
    for (std::int64_t c : cs) v.push_back(bi(c));
    return BigRationalPoly::from_coeffs(std::move(v));
};

// Res(x^2 - a, x^2 - b) == (a - b)^2, with a = 1e10, b = 2e10 (both fit int64), so
// a - b = -1e10 and (a-b)^2 = 1e20 — beyond the int64 ceiling (~9.22e18).
constexpr std::int64_t a = 10000000000LL;
constexpr std::int64_t b = 20000000000LL;
auto r = resultant(bigpoly({-a, 0, 1}), bigpoly({-b, 0, 1}));
auto oracle = BigRational::from_string("100000000000000000000");  // 1e20
r.has_value() && *r == *oracle;   // true — exact, past the int64 ceiling

// Res(x - 1, x - 2) == -1, and a shared factor gives 0.
resultant(bigpoly({-1, 1}), bigpoly({-2, 1})).value() == bi(-1);
resultant(bigpoly({-1, 0, 1}), bigpoly({-1, 1})).value().is_zero();  // x^2-1 and x-1

// disc(x^2 - 1) == b^2 - 4c form: 0 - 4*(-1) == 4.
discriminant(bigpoly({-1, 0, 1})).value() == bi(4);
```

## See also

- [`nimblecas.resultant`](resultant.md) — the `int64`-`Rational`,
  overflow-checked sibling this module big-backs; use it when values fit
  `int64`.
- [`nimblecas.bigratpoly`](bigratpoly.md) — the unbounded `Q[x]` operand type.
- [`nimblecas.bigrational`](bigrational.md) — the exact unbounded fraction
  field the result lives in.
- [Documentation hub](../Index.md)
