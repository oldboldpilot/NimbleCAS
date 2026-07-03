# `nimblecas.rothstein` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/rothstein/rothstein.cppm`

The **Rothstein–Trager** logarithmic integration of a rational function over the
rationals, `Q(x)`. Given `A(x)/D(x)` over `Q` with `D` **square-free** and
`deg A < deg D`, `log_part` computes the **logarithmic part** of the integral

```
int A/D dx = sum_{c : R(c) = 0}  c * log( gcd_x(A - c*D', D) ),
```

where `R(t) = res_x(D, A - t*D')` is the **Rothstein–Trager resultant**. Its
distinct roots `c` are the **residues** (the constant multipliers of the
logarithms), and each residue's logarithm argument is the gcd of `D` with
`A - c*D'`. It is the logarithmic-part half of rational-function integration
(ROADMAP §7.19): [`ratint`](ratint.md)'s Hermite reduction peels off the
elementary rational part and leaves exactly a square-free-denominator integrand;
this module handles that leftover. It is built on the Rothstein–Trager resultant
`R(t)` from [`nimblecas.resultant`](resultant.md) (§7.17) over the exact `Q[x]`
substrate of [`ratpoly`](ratpoly.md).

```cpp
import nimblecas.rothstein;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), and
[`resultant`](resultant.md).

## The rational/algebraic boundary

This is the single most important thing to understand about the output. Only
**rational** residues are handled. The Rothstein–Trager resultant `R(t)` may have
roots that are irrational or complex, and each such residue names a logarithm
whose argument lives in an **algebraic extension** of `Q` — machinery this pass
does not yet build. When every residue of `R` is rational the logarithmic part is
returned in full; when any residue is not, `log_part` returns
`MathError::not_implemented`.

- `int 1/(x^2 + 1) dx = arctan(x)` — the residues are `±i/2`, **complex**. `R(t)`
  has no rational root, so this is `not_implemented`.
- `int 1/(x^2 - 2) dx` — the residues are `±1/(2 sqrt 2)`, **irrational**. Again
  `not_implemented`.

A non-rational residue is detected by **completeness**: after stripping every
rational linear factor `(t - c)` from `R(t)`, a non-constant remainder means `R`
still has a root that is not rational, so the answer would be incomplete. That
extension-field case is deferred; expressing algebraic residues is the job of a
later pass, not this module.

## The overflow contract

Following the rest of the engine, every rational-case stage is **exact** and
**overflow-checked** (Rule 32): all arithmetic runs through the underlying `Q[x]`
operations, so an `int64` numerator or denominator that would overflow surfaces
as `MathError::overflow` rather than silently wrapping. A zero denominator is
`MathError::division_by_zero`. An improper input (`deg A >= deg D` after
reduction) or a non-rational residue is `MathError::not_implemented`. Because `Q`
is a field, every polynomial division taken along the way (the lowest-terms
reduction, the monic normalisation, the gcd arguments) is exact.

## The algorithm

`log_part` reduces the input, builds `R(t)` exactly, reads off its rational roots,
and emits one logarithm per distinct residue.

### 1. Reduce and normalise

`A/D` is first reduced to lowest terms so that `gcd(A, D) == 1` (a cancelled pole
drops out; `D` stays square-free as a divisor of the original). `D` is then made
**monic**, folding its leading constant into `A` so the value `A/D` is unchanged
(`A/D == (A/lc)/(D/lc)`). If the reduced input is not proper (`deg A >= deg D`)
the result is `not_implemented` — the caller is expected to hand over the proper,
square-free integrand that Hermite reduction produces.

### 2. Build `R(t)` by evaluation and interpolation

`R(t) = res_x(D, A - t*D')` has degree `<= deg D` in `t`, so `deg D + 1` samples
determine it. The **scalar** resultant `res_x(D, A - t*D')` from
[`resultant`](resultant.md) is evaluated at `t = 0, 1, …, deg D` and the results
are **Lagrange-interpolated** into `R(t)` over `Q`.

Making `D` monic in step 1 is **load-bearing** here, not cosmetic. At the special
`t` where `A - t*D'` happens to drop its formal `x`-degree (its leading
coefficient vanishes), the scalar resultant of the literal sampled polynomials
still equals the value of the smooth `R(t)` precisely because `D` is monic — so
every sample lands on the same degree-`<= deg D` polynomial and the interpolation
recovers `R` exactly. Without the monic normalisation the samples would not agree
with a single polynomial and the interpolation would be wrong.

### 3. Rational roots of `R`

The distinct rational roots of `R(t)` are found by the **rational-root theorem**:
clear denominators to integer coefficients, then test each candidate `p/q` with
`p` dividing the lowest nonzero coefficient and `q` the leading one. A zero
constant term contributes the root `0` (which adds nothing to the integral but is
recorded so the completeness check accounts for it).

### 4. Completeness check and assembly

After collecting the rational roots, `R(t)` is divided by each `(t - c)` as many
times as it goes; a **non-constant remainder** means a non-rational residue and
yields `not_implemented` (see the rational/algebraic boundary above). Otherwise,
for each **distinct** nonzero residue `c`, the term `c * log(gcd(A - c*D', D))` is
emitted, the gcd being monic and of degree `>= 1`.

## Residue merging — the minimal set of logarithms

Because the logarithms are indexed by the **distinct** roots of `R(t)` rather than
by the poles of `A/D`, two poles that happen to share a residue produce a
**single** logarithm of their combined factor. This is the minimal set of
logarithms, and an advantage over a naive per-pole partial-fraction expansion.

- `int 1/((x-1)(x-2)(x-3)) dx` — the residues are `1/D'(1) = 1/2`,
  `1/D'(2) = -1`, `1/D'(3) = 1/2`. The shared residue `1/2` at both `x = 1` and
  `x = 3` **merges**, so the result is the **two-term**
  `(1/2) log((x-1)(x-3)) - log(x - 2)` — one log for the combined factor
  `x^2 - 4x + 3`, not two separate logs.
- `int 1/(x(x-1)(x-4)) dx` — here `D'(0) = 4`, `D'(1) = -3`, `D'(4) = 12` give
  **three distinct** residues `1/4`, `-1/3`, `1/12`, so the result is the full
  **three-term** `(1/4) log(x) - (1/3) log(x - 1) + (1/12) log(x - 4)`.

## Public API

```cpp
[[nodiscard]] auto log_part(const RationalPoly& numerator,
                            const RationalPoly& denominator)
    -> Result<LogarithmicPart>;
```

### Result types

```cpp
// One logarithmic summand coefficient * log(argument), with argument monic of
// degree >= 1.
struct LogTerm {
    Rational coefficient;    // c — a rational residue
    RationalPoly argument;   // gcd(A - c*D', D), monic, deg >= 1
};

// int A/D dx = sum over terms of coefficient * log(argument).
struct LogarithmicPart {
    std::vector<LogTerm> terms;
};
```

### `log_part(numerator, denominator)`

Rothstein–Trager logarithmic integration of `numerator/denominator` over `Q(x)`,
with the denominator square-free and `deg numerator < deg denominator`. Returns a
`LogarithmicPart` whose terms satisfy the invariants:

- each `argument` is **monic** and of degree `>= 1`;
- the terms are indexed by the **distinct** rational residues (poles sharing a
  residue merge into one term);
- the integral of `0` has an **empty** logarithmic part.

Error model:

| Condition | Error |
| :--- | :--- |
| `denominator` is the zero polynomial | `MathError::division_by_zero` |
| improper input (`deg A >= deg D` after reduction) | `MathError::not_implemented` |
| a residue of `R(t)` is irrational or complex | `MathError::not_implemented` |
| an `int64` coefficient computation wraps | `MathError::overflow` |

## Examples

Worked from the tests (`tests/rothstein_tests.cpp`). Inputs are built
low-degree-first from integer coefficients.

```cpp
import nimblecas.rothstein;
import nimblecas.ratpoly;
using namespace nimblecas;

// Build x^k-style inputs from integer coefficients (low degree first).
auto ipoly = [](std::vector<std::int64_t> c) {
    return RationalPoly::from_polynomial(Polynomial{std::move(c)});
};

// Two rational residues:
// int 1/(x^2 - 1) dx = (1/2) log(x - 1) - (1/2) log(x + 1).
auto lp1 = log_part(ipoly({1}), ipoly({-1, 0, 1})).value();
// two terms: (1/2) log(x - 1) and (-1/2) log(x + 1).

// Composite square-free denominator:
// int 1/(x^2 - x) dx = -log(x) + log(x - 1).
auto lp2 = log_part(ipoly({1}), ipoly({0, -1, 1})).value();
// two terms: -log(x) and log(x - 1).

// Derivative over itself — a single log (a repeated root of R, residue 1):
// int 2x/(x^2 - 1) dx = log(x^2 - 1).
auto lp3 = log_part(ipoly({0, 2}), ipoly({-1, 0, 1})).value();
// one term: 1 * log(x^2 - 1).

// Residue merging — two poles share the residue 1/2:
// int 1/((x-1)(x-2)(x-3)) dx = (1/2) log((x-1)(x-3)) - log(x - 2).
auto d4 = ipoly({-1, 1}).multiply(ipoly({-2, 1})).value()
              .multiply(ipoly({-3, 1})).value();   // x^3 - 6x^2 + 11x - 6
auto lp4 = log_part(ipoly({1}), d4).value();
// two terms: (1/2) log(x^2 - 4x + 3) and -log(x - 2).

// Three distinct residues:
// int 1/(x(x-1)(x-4)) dx = (1/4) log(x) - (1/3) log(x-1) + (1/12) log(x-4).
auto d5 = ipoly({0, 1}).multiply(ipoly({-1, 1})).value()
              .multiply(ipoly({-4, 1})).value();   // x^3 - 5x^2 + 4x
auto lp5 = log_part(ipoly({1}), d5).value();
// three terms: (1/4) log(x), (-1/3) log(x - 1), (1/12) log(x - 4).

// A cancelled pole drops out: int x/(x(x-1)) = int 1/(x-1) = log(x - 1).
auto lp6 = log_part(ipoly({0, 1}), ipoly({0, -1, 1})).value();
// one term: log(x - 1).

// Complex residues ±i/2 — int 1/(x^2 + 1) = arctan(x): not_implemented.
auto e1 = log_part(ipoly({1}), ipoly({1, 0, 1}));  // not_implemented

// Irrational residues ±1/(2 sqrt 2) — int 1/(x^2 - 2): not_implemented.
auto e2 = log_part(ipoly({1}), ipoly({-2, 0, 1}));  // not_implemented

// Zero denominator fails.
auto e3 = log_part(ipoly({1}), RationalPoly{});  // division_by_zero

// Zero numerator integrates to no logarithmic part.
auto e4 = log_part(RationalPoly{}, ipoly({-1, 0, 1})).value();  // terms empty
```

The tests verify correctness by **differentiating back**: for
`int A/D = sum c_i log V_i` they check `sum c_i V_i'/V_i == A/D` **exactly**, by
cross-multiplying the two rational functions (`num * D == A * den`) — no factoring
or floating point involved.

## Relationship to integration

Rothstein–Trager is the **logarithmic-part half** of rational-function
integration (ROADMAP §7.19). Together with [`ratint`](ratint.md)'s Hermite
reduction — which computes the exact rational part and hands over precisely the
proper, square-free-denominator integrand this module requires — it completes
rational-function integration for the **rational-residue** class. The
Rothstein–Trager resultant `R(t) = res_x(D, A - t*D')` is built on
[`nimblecas.resultant`](resultant.md) (§7.17). A combined Hermite +
Rothstein–Trager `integrate_rational` capstone is the natural next step,
returning the rational part, the rational-residue logarithms, and
`not_implemented` for the algebraic-residue remainder until an extension field
lands.

## See also

- [`nimblecas.resultant`](resultant.md) — the resultant `res_x(D, A - t*D')`
  whose distinct roots are the residues; this module samples and interpolates it.
- [`nimblecas.ratint`](ratint.md) — Hermite reduction, whose square-free
  logarithmic integrand this module consumes.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Q[x]` substrate (`Rational`,
  `RationalPoly`, division-with-remainder, monic Euclidean gcd, derivative) this
  module is built on.
- [Documentation hub](../Index.md)
