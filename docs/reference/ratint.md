# `nimblecas.ratint` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/ratint/ratint.cppm`

**Hermite reduction** of a rational function over the rationals, `Q(x)`. Given
`A(x)/B(x)` over `Q`, `hermite_reduce` splits the integral into an already-computed
exact **rational part** `g(x)` plus a leftover integrand whose denominator is
**square-free**, all **without fully factoring** `B`:

```
int A(x)/B(x) dx = g(x)  +  int h(x) dx,   h = num/den, den square-free.
```

The rational part `g` is elementary and produced exactly here. The remaining
`int h dx` is the logarithmic/transcendental part, later handled by
Rothstein–Trager (which needs precisely a square-free denominator — what this pass
leaves behind). It is the rational-part half of rational-function integration
(ROADMAP §7.19), built on the square-free towers of [`pfd`](pfd.md) over the exact
`Q[x]` substrate of [`ratpoly`](ratpoly.md).

```cpp
import nimblecas.ratint;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), and [`pfd`](pfd.md).

## The core reduction

Using the square-free partial-fraction towers from [`pfd`](pfd.md), the proper
part of `A/B` becomes a sum of `M(x)/V(x)^k` with `V` monic square-free and
`deg M < k * deg V`. For each tower with `k >= 2`, because `V` is square-free
`gcd(V, V') = 1`, so the Bézout identity

```
V*S + V'*T = M,   deg T < deg V,
```

is solvable; integrating by parts then gives

```
int M/V^k dx = -T/((k-1) V^{k-1})  +  int (S + T'/(k-1)) / V^{k-1} dx.
```

The new numerator `S + T'/(k-1)` again has degree `< (k-1)*deg V`, so **each step
lowers the pole order by one while staying proper**. Repeating until `k = 1`, the
finished `-T/((k-1) V^{k-1})` pieces accumulate into the rational part `g`, and the
remaining `k = 1` term `M/V` joins the square-free integrand `h`. The polynomial
part `P` of `A/B` integrates directly into `g` (`int (sum a_i x^i) dx =
sum a_i/(i+1) x^{i+1}`, constant of integration `0`). Every operation is exact and
overflow-checked (Rule 32).

## The overflow contract

Following the rest of the engine, every stage is **exact** and
**overflow-checked** (Rule 32): all arithmetic runs through the underlying `Q[x]`
operations, so an `int64` numerator or denominator that would overflow surfaces as
`MathError::overflow` rather than silently wrapping. A zero denominator is
`MathError::division_by_zero`. Because `Q` is a field, every polynomial division
taken along the way (the Bézout cofactors, the exact `S = (M - T*V')/V` quotient,
the fraction reductions) is exact and the failure modes are exactly those two — the
same contract as [`ratpoly`](ratpoly.md) and [`pfd`](pfd.md).

## Consuming the square-free towers

`hermite_reduce` calls `square_free_partial_fractions` from [`pfd`](pfd.md) and
works directly on the **tower form** `SquareFreePartialFraction` — the proper part
grouped by square-free factor, one summand `M_i / V_i^{k_i}` per distinct
multiplicity, *before* the base-`b` power expansion that `partial_fractions`
performs. That pre-expansion form is exactly what the by-parts recurrence needs:
each `V_i` is monic and square-free, so `gcd(V_i, V_i') = 1` is guaranteed and the
Bézout cofactor `T` exists for every tower. This is why `ratint` depends on `pfd`
(for the towers), `ratpoly` (for the `Q[x]` arithmetic), and `core`.

## Public API

```cpp
[[nodiscard]] auto hermite_reduce(const RationalPoly& numerator,
                                  const RationalPoly& denominator)
    -> Result<HermiteReduction>;
```

### Result type

```cpp
// int numerator/denominator dx
//     == rational_num/rational_den  +  int integrand_num/integrand_den dx,
// with integrand_den square-free and deg integrand_num < deg integrand_den.
struct HermiteReduction {
    RationalPoly rational_num;   // g = rational_num / rational_den (already integrated)
    RationalPoly rational_den;   // never zero (the constant 1 when g == 0)
    RationalPoly integrand_num;  // remaining integrand numerator (deg < deg integrand_den)
    RationalPoly integrand_den;  // square-free; the input to the logarithmic part
};
```

### `hermite_reduce(numerator, denominator)`

Hermite-reduces `numerator/denominator` over `Q(x)` and returns a
`HermiteReduction`. The output satisfies the identity and invariants:

- `int numerator/denominator dx == rational_num/rational_den + int integrand_num/integrand_den dx`;
- `integrand_den` is **square-free** and `deg integrand_num < deg integrand_den`
  (a proper, square-free-denominator integrand ready for Rothstein–Trager);
- when the whole integral is elementary-rational, `integrand_num` is **zero**
  (and `integrand_den` is the constant `1`);
- when the integral is purely logarithmic, `rational_num` is **zero** (and
  `rational_den` is the constant `1`).

Error model:

| Condition | Error |
| :--- | :--- |
| `denominator` is the zero polynomial | `MathError::division_by_zero` |
| an `int64` coefficient computation wraps | `MathError::overflow` |

## Examples

Worked from the tests (`tests/ratint_tests.cpp`). Inputs are built low-degree-first
from integer coefficients.

```cpp
import nimblecas.ratint;
import nimblecas.ratpoly;
using namespace nimblecas;

// Build x^k-style inputs from integer coefficients (low degree first).
auto ipoly = [](std::vector<std::int64_t> c) {
    return RationalPoly::from_polynomial(Polynomial{std::move(c)});
};

// Pure rational — a double pole with no logarithmic part:
// int 1/(x - 1)^2 dx = -1/(x - 1).
auto hr1 = hermite_reduce(ipoly({1}), ipoly({1, -2, 1})).value();
// hr1.integrand_num == 0 (no leftover integrand); rational part -1/(x - 1).

// Pure logarithmic — the denominator is already square-free, g == 0:
// int 1/(x^2 - 1) dx, integrand unchanged over the square-free x^2 - 1.
auto hr2 = hermite_reduce(ipoly({1}), ipoly({-1, 0, 1})).value();
// hr2.rational_num == 0; integrand_den square-free (x^2 - 1).

// Both parts present:
// int 1/(x^2 + 1)^2 = x/(2(x^2 + 1)) + (1/2) int 1/(x^2 + 1).
auto hr3 = hermite_reduce(ipoly({1}), ipoly({1, 0, 2, 0, 1})).value();
// rational_num and integrand_num both nonzero; integrand_den == x^2 + 1.

// Double pole yet purely rational:
// int x/(x^2 + 1)^2 = -1/(2(x^2 + 1)).
auto hr4 = hermite_reduce(ipoly({0, 1}), ipoly({1, 0, 2, 0, 1})).value();
// hr4.integrand_num == 0; rational part -1/(2(x^2 + 1)).

// Zero denominator fails.
auto bad = hermite_reduce(ipoly({1}), RationalPoly{});  // division_by_zero
```

The tests verify correctness by **differentiating back**: for `int A/D = g + int h`
they check `d/dx(g) + h == A/D` **exactly**, by cross-multiplying the two rational
functions (`L_num * D == A * L_den`) — no factoring or floating point involved.

## Relationship to integration

Hermite reduction is the **rational-part half** of rational-function integration
(ROADMAP §7.19). It reuses the square-free factorization and Bézout split of
[`pfd`](pfd.md) to peel off the elementary rational term `g` without a full
irreducible factorization of `B`, and hands the leftover square-free integrand
`h = integrand_num/integrand_den` to [**Rothstein–Trager**](rothstein.md), which
computes the logarithmic part via the resultant `res_x(D, A - t*D')`. `ratint`
produces exactly the square-free-denominator input it requires; the two passes are
run in sequence by a combined `integrate_rational` capstone.

## See also

- [`nimblecas.pfd`](pfd.md) — the square-free partial-fraction towers
  (`square_free_partial_fractions` → `SquareFreeTerm` / `SquareFreePartialFraction`)
  and the Bézout split this module consumes.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Q[x]` substrate (`Rational`,
  `RationalPoly`, division-with-remainder, monic Euclidean gcd, derivative) the
  reduction is built on.
- [Documentation hub](../Index.md)
