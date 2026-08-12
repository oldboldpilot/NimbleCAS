# `nimblecas.bigfactor` — Polynomial factorization over Q, unbounded (Reference)

**Author:** Olumuyiwa Oluwasanmi

Source: `src/bigfactor/bigfactor.cppm`

The **unbounded, overflow-free** counterpart to [`nimblecas.factor`](factor.md).
Where `factor_over_Q` there runs the Yun -> Kronecker pipeline on the `int64`
`RationalPoly` and reports overflow the moment a numerator or denominator
saturates the `int64` ring, this module runs the **same** pipeline on
[`BigRationalPoly`](bigratpoly.md) and has no such ceiling: the exact
arithmetic **never** produces `MathError::overflow`. It is the factorization
step a bignum Trager splitting-field path needs, so it can factor
large-coefficient field norms that overrun the `int64` tier.

**Pipeline** (mirrors `nimblecas.factor`), given `p` in `Q[x]`:

1. clear denominators to an integer (`BigInt`-coefficient) polynomial;
2. take its primitive part (divide by the integer content, positive leading
   coefficient);
3. run Yun's square-free factorization -> (square-free primitive factor,
   multiplicity);
4. factor EACH square-free factor into irreducibles via Kronecker's
   algorithm;
5. combine multiplicities.

**Kronecker.** On a square-free primitive integer polynomial `f` of degree
`n >= 1`, a proper factor of degree `<= floor(n/2)` exists iff `f` is
reducible. For each target degree `s`, pick `s+1` distinct integer nodes; an
integer root peels a linear factor; otherwise every candidate factor `g`
satisfies `g(node) | f(node)`, so enumerating one divisor of `f(node)` per
node and Lagrange-interpolating a degree-`<= s` polynomial through those
values yields every candidate `g`. A candidate is accepted when integral and
its primitive part divides `f` exactly; the search then recurses on that
factor and its cofactor.

The arithmetic itself is exact and unbounded, so `MathError::overflow` is
**never** produced here. Two steps remain genuinely **bounded**, and this is
the honest boundary of the module: (a) enumerating the divisors of an integer
value `f(node)` needs that value's prime factorization, done by trial
division up to `kTrialDivisionBound` with a cap of `kMaxDivisorsPerValue` on
the divisor count; (b) the Cartesian product of the per-node divisor choices
is bounded by `kDivisorTupleBudget`. Exceeding either bound returns
`MathError::not_implemented` — an honest "could not factor within budget" —
rather than a wrong or partial factorization. The zero polynomial is
`MathError::domain_error`; a nonzero constant returns an empty list.
`factor_over_Q` either returns a fully irreducible factorization or an error,
**never** a partial result presented as complete.

```cpp
import nimblecas.bigfactor;
```

Depends on [`core`](core.md), `bigint`, `bigrational`, and
[`bigratpoly`](bigratpoly.md).

## The honesty boundary

- **`MathError::overflow` is never produced.** Every arithmetic step is exact
  `BigInt` / `BigRational` / `BigRationalPoly` work, unbounded by
  construction.
- **The bounded step is honestly named `not_implemented`, not a wrong
  answer.** Trial-division prime factorization of an integer `f(node)` is
  capped at `kTrialDivisionBound = 2'000'000` (with `kMaxDivisorsPerValue =
  1'000'000` divisors per value), and the shared Kronecker search budget is
  `kDivisorTupleBudget = 1'000'000` Lagrange divisor-tuples. Exceeding either
  cap returns `MathError::not_implemented` rather than a silently
  incomplete or heuristic factorization.
- **The zero polynomial is `domain_error`** (every value is a root); a
  nonzero constant is **not** an error — it returns an empty factor list.

## API summary

```cpp
// Factor p into irreducible factors over Q. Returns (irreducible_factor, multiplicity)
// pairs whose product, each factor raised to its multiplicity, equals the PRIMITIVE
// PART of the integer polynomial obtained from p by clearing denominators -- i.e. p up
// to a nonzero rational constant. Each returned factor is a primitive integer polynomial
// (content 1) with a positive leading coefficient, lifted into Q[x], and irreducible
// over Q. Factors are returned in no particular order.
//
// The zero polynomial is MathError::domain_error; a nonzero constant yields an empty
// list. An input whose factorization would exceed the internal trial-division or
// divisor-tuple budget returns MathError::not_implemented. The exact arithmetic never
// overflows, so MathError::overflow is never produced.
[[nodiscard]] auto factor_over_Q(const BigRationalPoly& p)
    -> Result<std::vector<std::pair<BigRationalPoly, std::int64_t>>>;
```

## Error model

| Condition | Error |
| :--- | :--- |
| `factor_over_Q` of the zero polynomial | `MathError::domain_error` |
| Trial-division factoring of a sample value `f(node)` exceeds `kTrialDivisionBound` / `kMaxDivisorsPerValue` | `MathError::not_implemented` |
| The Kronecker search's Cartesian-product divisor-tuple budget (`kDivisorTupleBudget`) is exhausted | `MathError::not_implemented` |

There is **no `overflow` row**: `BigInt` / `BigRational` / `BigRationalPoly`
arithmetic cannot wrap, so the `overflow` the `int64`
[`factor`](factor.md) can raise is unreachable here. A nonzero constant is a
valid input (empty factor list), not an error.

## Worked example

```cpp
import nimblecas.bigfactor;
import nimblecas.bigratpoly;
import nimblecas.bigrational;
import nimblecas.bigint;
using namespace nimblecas;

auto bi_str = [](std::string_view s) { return BigInt::from_string(s).value(); };
auto bpoly = [](std::initializer_list<std::string_view> cs) {
    std::vector<BigRational> v;
    for (std::string_view s : cs) v.push_back(BigRational::from_bigint(bi_str(s)));
    return BigRationalPoly::from_coeffs(std::move(v));
};

// (x - 2^32)(x + 2^32) = x^2 - 2^64. The constant term 2^64 = 18446744073709551616
// exceeds INT64_MAX (9223372036854775807), yet factors cleanly over BigRational.
// Kronecker's degree-1 search factors f(0) = -2^64 and f(1) = 1 - 2^64; the divisors
// it needs, 2^32 | 2^64 and (2^32 - 1) | (2^64 - 1), both exist within budget, so the
// interpolant x - 2^32 is recovered exactly.
auto r = factor_over_Q(bpoly({"-18446744073709551616", "0", "1"})).value();
r.size() == 2;                                            // two linear factors
// (x - 4294967296) and (x + 4294967296), each with multiplicity 1.

// The honest budget refusal, and the domain_error / empty-list conventions:
factor_over_Q(BigRationalPoly::zero()).error() == MathError::domain_error;
factor_over_Q(bpoly({"5"})).value().empty();  // nonzero constant: no non-unit factors
```

## See also

- [`nimblecas.factor`](factor.md) — the `int64`-`RationalPoly`,
  overflow-checked sibling this module big-backs; use it when coefficients
  fit `int64`.
- [`nimblecas.bigratpoly`](bigratpoly.md) — the unbounded `Q[x]` operand and
  result type.
- [`nimblecas.bigrational`](bigrational.md) — the exact unbounded fraction
  field underlying every layer.
- [Documentation hub](../Index.md)
