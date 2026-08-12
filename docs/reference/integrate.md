# `nimblecas.integrate` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/integrate/integrate.cppm`

The **capstone** of rational-function integration over the rationals, `Q(x)`.
Given `A(x)/B(x)` over `Q`, `integrate_rational` assembles the **complete
indefinite integral** by running the two halves of the standard decomposition in
sequence:

```
int A(x)/B(x) dx = rational_num/rational_den  +  sum_i c_i * log(argument_i).
```

The first term is the elementary **rational part** `g`; the sum is the
**logarithmic part**, a set of residue-weighted logarithms. It is the join of the
two integration halves (ROADMAP §7.19): [`ratint`](ratint.md)'s Hermite reduction
peels off `g` and leaves a proper, square-free-denominator integrand, and
[`rothstein`](rothstein.md)'s Rothstein–Trager pass turns that leftover into
logarithms. Both run over the exact `Q[x]` substrate of [`ratpoly`](ratpoly.md).

```cpp
import nimblecas.integrate;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), [`ratint`](ratint.md), and
[`rothstein`](rothstein.md) (whose `log_part_extended` — and transitively
[`factor`](factor.md), [`algnum`](algnum.md), [`algpoly`](algpoly.md) — backs
`integrate_rational_extended` below).

## The pipeline

`integrate_rational` composes the two passes end to end; it holds no integration
mathematics of its own beyond wiring their outputs together:

### 1. Hermite reduction (the rational part)

The input `A/B` is handed to `hermite_reduce` from [`ratint`](ratint.md), which
splits off the exact rational part `g = rational_num/rational_den` and returns a
leftover integrand `integrand_num/integrand_den` that is **proper** and has a
**square-free denominator** — precisely the form Rothstein–Trager requires. The
polynomial part of an improper `A/B` integrates directly into `g`, so a
constant-denominator input comes back purely as `g` with no leftover.

### 2. Rothstein–Trager (the logarithmic part)

If the leftover integrand is **non-zero**, it is passed to `log_part` from
[`rothstein`](rothstein.md), which computes the residue resultant
`R(t) = res_x(D, A − t·D')`, reads off its rational roots, and emits one
`c_i * log(argument_i)` per distinct residue. When the leftover integrand is
**zero** — the whole integral is elementary-rational — the log step is **skipped**
entirely and the logarithmic part is empty.

The two results are packed into a `RationalIntegral`: `g` from step 1, the log
terms (if any) from step 2.

## The overflow contract

`integrate_rational` inherits the failure modes of the two passes it composes.
Following the rest of the engine, every stage is **exact** and
**overflow-checked** (Rule 32): all arithmetic runs through the underlying `Q[x]`
operations, so an `int64` numerator or denominator that would overflow surfaces as
`MathError::overflow` rather than silently wrapping. A zero denominator is
`MathError::division_by_zero`. A non-rational residue in the logarithmic part is
`MathError::not_implemented` (see below). Because `Q` is a field, every polynomial
division taken along the way is exact — the same contract as [`ratint`](ratint.md)
and [`rothstein`](rothstein.md).

## The rational/algebraic boundary

The rational part `g` is **always** computable — Hermite reduction is complete
over `Q`. The logarithmic part is not: the Rothstein–Trager resultant `R(t)` may
have residues that are irrational or complex, and each such residue names a
logarithm whose argument lives in an **algebraic extension** of `Q` that this pass
does not build. When any residue is non-rational, `log_part` returns
`MathError::not_implemented` and it **propagates** through `integrate_rational` —
so even though the rational part exists, the *whole* integral is not expressible
in this rational-plus-rational-logarithm form without an extension field. That
case is deferred; expressing algebraic residues is the job of a later pass.

## Public API

```cpp
[[nodiscard]] auto integrate_rational(const RationalPoly& numerator,
                                      const RationalPoly& denominator)
    -> Result<RationalIntegral>;
```

### Result type

```cpp
// int numerator/denominator dx
//     == rational_num/rational_den
//        + sum over log_terms of coefficient * log(argument).
struct RationalIntegral {
    RationalPoly rational_num;         // g numerator (the elementary rational part)
    RationalPoly rational_den;         // g denominator (never zero; constant 1 when g == 0)
    std::vector<LogTerm> log_terms;    // the logarithmic part; empty when purely rational
};
```

`LogTerm` is re-used from [`nimblecas.rothstein`](rothstein.md): a rational
`coefficient` (the residue `c`) and a monic `argument` of degree `>= 1`.

### `integrate_rational(numerator, denominator)`

Integrates `numerator/denominator` over `Q(x)` and returns a `RationalIntegral`.
The output satisfies the identity and invariants:

- `int numerator/denominator dx == rational_num/rational_den + sum_i coefficient_i * log(argument_i)`;
- `rational_den` is **never zero** (the constant `1` when `g == 0`);
- `log_terms` is **empty** when the integral is purely rational (the leftover
  Hermite integrand was zero);
- `rational_num` is **zero** when the integral is purely logarithmic.

Error model:

| Condition | Error |
| :--- | :--- |
| `denominator` is the zero polynomial | `MathError::division_by_zero` |
| a residue of the logarithmic part is irrational or complex | `MathError::not_implemented` |
| an `int64` coefficient computation wraps | `MathError::overflow` |

## `integrate_rational_extended` — every residue, rational or algebraic

`integrate_rational_extended` **mirrors `integrate_rational`** exactly, wiring
the same [`ratint`](ratint.md) Hermite reduction into
[`rothstein`](rothstein.md)'s `log_part_extended` instead of `log_part`, so
irrational/complex residues are expressed exactly via algebraic extension
fields (see [`rothstein`](rothstein.md#the-rationalalgebraic-boundary))
instead of being punted to `not_implemented`:

```cpp
[[nodiscard]] auto integrate_rational_extended(const RationalPoly& numerator,
                                               const RationalPoly& denominator)
    -> Result<RationalIntegralExtended>;
```

```cpp
// int numerator/denominator dx
//     == rational_num/rational_den
//        + sum over log_terms of coefficient * log(argument)
//        + sum over algebraic_log_terms of the field's conjugate-sum log block.
struct RationalIntegralExtended {
    RationalPoly rational_num;
    RationalPoly rational_den;
    std::vector<LogTerm> log_terms;                    // rational residues
    std::vector<AlgebraicLogTerm> algebraic_log_terms;  // irrational/complex residues, exact
};
```

`rational_num`/`rational_den` come from the **same** `hermite_reduce` call
`integrate_rational` uses — the rational part is always fully computable, the
same as before. `log_terms`/`algebraic_log_terms` are `log_part_extended`'s
`rational_terms`/`algebraic_terms`, re-packed unchanged. `AlgebraicLogTerm` is
re-used from [`rothstein`](rothstein.md): a `NumberField`, its `residue`
(always the field's `generator()`), and a monic `AlgebraicPoly` `argument`.

Error model:

| Condition | Error |
| :--- | :--- |
| `denominator` is the zero polynomial | `MathError::division_by_zero` |
| `log_part_extended`'s internal factorization (`factor_over_Q`) exhausts its search budget | `MathError::not_implemented` |
| `log_part_extended`'s completeness identity fails | `MathError::domain_error` (never alongside a partial result) |
| an `int64` coefficient computation wraps | `MathError::overflow` |

### Example

```cpp
// int 1/(x^2+1) dx: the denominator is already square-free, so hermite_reduce
// leaves rational_num == 0 and the FULL leftover integrand for log_part_extended.
// The complex residues +-i/2 are expressed exactly as one conjugate-sum block
// over Q(alpha) = Q[t]/(t^2 + 1/4) instead of not_implemented.
auto rie = integrate_rational_extended(ipoly({1}), ipoly({1, 0, 1})).value();
rie.rational_num.is_zero();          // true
rie.log_terms.empty();               // true — no rational residues
rie.algebraic_log_terms.size() == 1; // one conjugate-sum block, alpha^2 = -1/4

// int 1/(x^2-2) dx: same shape, over Q(alpha) = Q[t]/(t^2 - 1/8).
auto rie2 = integrate_rational_extended(ipoly({1}), ipoly({-2, 0, 1})).value();
rie2.algebraic_log_terms.size() == 1;  // alpha^2 = 1/8
```

## Examples

Worked from the tests (`tests/integrate_tests.cpp`). Inputs are built
low-degree-first from integer coefficients.

```cpp
import nimblecas.integrate;
import nimblecas.ratpoly;
using namespace nimblecas;

// Build x^k-style inputs from integer coefficients (low degree first).
auto ipoly = [](std::vector<std::int64_t> c) {
    return RationalPoly::from_polynomial(Polynomial{std::move(c)});
};

// Pure rational — a double pole, no logs:
// int 1/(x - 1)^2 dx = -1/(x - 1).
auto ri1 = integrate_rational(ipoly({1}), ipoly({1, -2, 1})).value();
// ri1.log_terms empty; rational part -1/(x - 1).

// Pure logarithmic — the denominator is already square-free, g == 0:
// int 1/(x^2 - 1) dx = (1/2) log(x - 1) - (1/2) log(x + 1).
auto ri2 = integrate_rational(ipoly({1}), ipoly({-1, 0, 1})).value();
// ri2.rational_num == 0; two log terms.

// Improper — polynomial-driven rational part plus one merged log:
// int x^3/(x^2 - 1) dx = x^2/2 + (1/2) log(x^2 - 1).
auto ri3 = integrate_rational(ipoly({0, 0, 0, 1}), ipoly({-1, 0, 1})).value();
// rational part x^2/2; one merged log (1/2) log(x^2 - 1).

// A double pole plus two simple poles:
// int 1/(x^2 (x - 1)) dx = 1/x - log(x) + log(x - 1).
auto ri4 = integrate_rational(ipoly({1}), ipoly({0, 0, -1, 1})).value();
// rational part 1/x; two log terms.

// Purely polynomial — constant denominator, no poles:
// int (2x)/1 dx = x^2.
auto ri5 = integrate_rational(ipoly({0, 2}), ipoly({1})).value();
// ri5.log_terms empty; ri5.rational_num == x^2.

// Complex residues in the log part — the rational part x/(2(x^2 + 1)) exists,
// but int 1/(x^2 + 1)^2 dx as a whole is not_implemented.
auto e1 = integrate_rational(ipoly({1}), ipoly({1, 0, 2, 0, 1}));  // not_implemented

// Irrational residues — int 1/(x^2 - 2) dx is not_implemented.
auto e2 = integrate_rational(ipoly({1}), ipoly({-2, 0, 1}));  // not_implemented

// Zero denominator fails.
auto e3 = integrate_rational(ipoly({1}), RationalPoly{});  // division_by_zero
```

The tests verify the whole result end to end by **differentiating it back**: for
`int A/D = g + sum c_i log(V_i)` they check `d/dx(g) + sum c_i V_i'/V_i == A/D`
**exactly**, by cross-multiplying the two rational functions (`num * D == A * den`)
— no factoring or floating point involved.

## Relationship to integration

`integrate_rational` **completes** rational-function integration for the
**rational-residue** class (ROADMAP §7.19), and `integrate_rational_extended`
extends that completeness to **every** residue class — rational, irrational,
or complex — that its factorization budget can reach, by expressing the
algebraic residues exactly over the extension field they generate rather than
deferring them. Both are the assembly point above the two halves that each
solve part of the problem:

- [`ratint`](ratint.md) — Hermite reduction, the **rational-part** half, which
  computes `g` exactly and leaves the proper, square-free integrand (shared by
  both capstones, unchanged);
- [`rothstein`](rothstein.md) — Rothstein–Trager, the **logarithmic-part**
  half, built on the resultant `R(t) = res_x(D, A − t·D')` from
  [`resultant`](resultant.md) (§7.17): `log_part` (rational residues only,
  behind `integrate_rational`) or `log_part_extended` (every residue, behind
  `integrate_rational_extended`) turn that integrand into the minimal set of
  residue-weighted logarithms.

**Transcendental** integrands (exp, log, trig arguments) remain a separate
future layer outside either capstone.

## See also

- [`nimblecas.ratint`](ratint.md) — Hermite reduction, the rational-part half
  (`hermite_reduce` → `HermiteReduction`) both capstones run first.
- [`nimblecas.rothstein`](rothstein.md) — Rothstein–Trager logarithmic
  integration, the logarithmic-part half (`log_part`/`log_part_extended` →
  `LogTerm`/`AlgebraicLogTerm`) both capstones run second.
- [`nimblecas.resultant`](resultant.md) — the resultant `res_x(D, A − t·D')`
  underneath the logarithmic part.
- [`nimblecas.factor`](factor.md) / [`nimblecas.algnum`](algnum.md) /
  [`nimblecas.algpoly`](algpoly.md) — the complete factorization and
  extension-field machinery `integrate_rational_extended` inherits from
  `log_part_extended`.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Q[x]` substrate (`Rational`,
  `RationalPoly`, division-with-remainder, monic Euclidean gcd, derivative) both
  halves are built on.
- [Documentation hub](../Index.md)
