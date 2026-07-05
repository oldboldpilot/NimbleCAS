# `nimblecas.symint` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/symint/symint.cppm`

Expr-level symbolic integration. `integrate(f, var)` returns an **indefinite**
antiderivative `int f dx` (no constant of integration); `integrate_definite`
evaluates `int_a^b f dx` as `F(b) - F(a)`. This module is the bridge that finally
makes integration reachable from an ordinary `Expr`: the exact rational integrator
[`integrate`](integrate.md) works on `RationalPoly`, and `symint` wires an `Expr`
integrand into it (and into a small table of elementary forms) while keeping the
engine's central promise intact.

```cpp
import nimblecas.symint;
```

Depends on [`core`](core.md), [`symbolic`](symbolic.md), [`simplify`](simplify.md),
[`diff`](diff.md), [`polynomial`](polynomial.md), [`polyexpr`](polyexpr.md),
[`ratpoly`](ratpoly.md), [`rothstein`](rothstein.md), and
[`integrate`](integrate.md).

## The honesty boundary

Integration is only partly algorithmic, so the supported class is deliberately
small and **explicit**. The invariant (Rule 32) is: `integrate` **never guesses**.
Either it returns an antiderivative that is provably correct, or it returns
`MathError::not_implemented`. It never returns a plausible-but-wrong closed form.

That promise is made checkable by the **differentiate-back guarantee**: for every
`F` this module returns, `d/dx(F)` under [`diff`](diff.md) equals the integrand `f`
exactly (after [`simplify`](simplify.md)). The elementary antiderivatives are
written with the **same function-name spellings** the derivative table inverts —
`exp`, `ln`, `sin`, `cos`, `sinh`, `cosh`, `atan`, `asin` — so `differentiate` is a
genuine left inverse of `integrate` on the supported class. (Note the spellings:
the inverse-trig results use `atan`/`asin`, matching `diff`, **not** `arctan`/
`arcsin`. `ln` is used for the logarithm, not `log`.) The test suite asserts this
back-differentiation for every indefinite case.

## Supported integrand class

| Rule | Form | Result |
| :--- | :--- | :--- |
| Constant | `f` free of `var` | `f * x` |
| Linearity | `sum(t_i)` | `sum(int t_i dx)` |
| Constant factor | `c * g`, `c` free of `var` | `c * int g dx` |
| Power rule | `(a*x+b)^n`, constant `n != -1` | `(a*x+b)^(n+1) / (a*(n+1))` |
| Logarithm | `(a*x+b)^-1` | `(1/a) ln(a*x+b)` |
| Exponential | `exp(a*x+b)` | `exp(a*x+b) / a` |
| Sine | `sin(a*x+b)` | `-cos(a*x+b) / a` |
| Cosine | `cos(a*x+b)` | `sin(a*x+b) / a` |
| Hyperbolic | `sinh(a*x+b)`, `cosh(a*x+b)` | `cosh/(a)`, `sinh/(a)` |
| Arctangent | `1/(x^2 + k^2)`, integer `k` | `(1/k) atan(x/k)` |
| Arcsine | `1/sqrt(k^2 - x^2)`, integer `k` | `asin(x/k)` |
| Rational | ratio of polynomials in `var` | rational part + `sum c_i ln(V_i)` |

Here `a`, `b` are any sub-expressions **free of** `var` with `a != 0`; the affine
(linear) argument `a*x+b` is detected exactly by checking that `d(inner)/dx` is free
of `var` (a zero second derivative). Only **linear** u-substitution is performed —
general substitution and integration by parts are out of scope for this first
version and fall through to `not_implemented`.

### Rational functions

When the elementary strategies decline, an integrand that is a ratio of
polynomials `N(x)/D(x)` (a product whose only `var`-dependent denominators are
`g^(-k)` factors) is converted via [`polyexpr`](polyexpr.md)'s `to_polynomial`
into two `RationalPoly` operands and handed to `integrate_rational` from
[`integrate`](integrate.md). Its `RationalIntegral` result — a rational part plus
residue-weighted logarithms — is rebuilt into an `Expr`:

```
int N/D dx = rational_num/rational_den + sum_i coefficient_i * ln(argument_i).
```

The rational integrator's boundary is inherited: a rational function with an
**irrational or complex residue** in its logarithmic part (for example
`1/(x^2 - 2)` or `1/(x^2+1)^2`) surfaces as `MathError::not_implemented`, because
naming those logarithms needs an algebraic extension field
(see [`integrate`](integrate.md)). The purely inverse-trig forms `1/(x^2+k^2)` and
`1/sqrt(k^2-x^2)` are handled by the table **above** the rational bridge precisely
because their honest antiderivatives are `atan`/`asin`, which the rational
integrator (a `Q(x)`-logarithm engine) cannot produce.

### Deliberately excluded

`sec^2(x) -> tan(x)` is **not** provided, even though it is elementary: the
derivative table in [`diff`](diff.md) represents `d/dx tan(x)` as `1 + tan^2(x)`
rather than `sec^2(x)`, so a `tan(x)` answer could not be verified by the
differentiate-back guarantee. Rather than return an unverifiable result, `symint`
declines it. This is the honesty boundary working as intended.

## Public API

```cpp
[[nodiscard]] auto integrate(const Expr& f, std::string_view var) -> Result<Expr>;

[[nodiscard]] auto integrate_definite(const Expr& f, std::string_view var,
                                      const Expr& a, const Expr& b) -> Result<Expr>;
```

### `integrate(f, var)`

Returns an antiderivative of `f` with respect to the symbol named `var`, with **no
constant of integration**, automatically simplified. Returns
`MathError::not_implemented` for any integrand outside the supported class.

### `integrate_definite(f, var, a, b)`

Computes `F(b) - F(a)` where `F = integrate(f, var)`, by substituting the bounds
into the antiderivative and simplifying. It propagates `not_implemented` from the
indefinite step. It assumes `f` is continuous on `[a, b]` — it does **not** detect
a pole strictly between the bounds, so the caller is responsible for the integrand
being integrable across the interval.

## Error model

| Condition | Error |
| :--- | :--- |
| Integrand outside the supported class (e.g. `exp(x^2)`, a non-linear substitution, a non-elementary function) | `MathError::not_implemented` |
| Rational integrand with an irrational/complex residue in its log part | `MathError::not_implemented` (from [`integrate`](integrate.md)) |
| An `int64` coefficient computation wraps during simplification/rational arithmetic | `MathError::overflow` |

The module **never** throws and **never** returns a wrong antiderivative (Rule 32).

## Worked examples

```cpp
import nimblecas.symint;
import nimblecas.symbolic;
import nimblecas.diff;
using namespace nimblecas;

const Expr x = Expr::symbol("x");
const auto i = [](std::int64_t v) { return Expr::integer(v); };

// Power rule:              int x^2 dx = x^3/3.
auto F1 = integrate(x.pow(i(2)), "x").value();

// Linearity + power rule:  int (3x^2 + 2x + 1) dx = x^3 + x^2 + x.
auto poly = Expr::sum({Expr::product({i(3), x.pow(i(2))}),
                       Expr::product({i(2), x}), i(1)});
auto F2 = integrate(poly, "x").value();

// Linear substitution:     int cos(3x) dx = sin(3x)/3.
auto F3 = integrate(Expr::apply("cos", {Expr::product({i(3), x})}), "x").value();

// Logarithm:               int 1/x dx = ln(x).
auto F4 = integrate(x.pow(i(-1)), "x").value();

// Inverse trig:            int 1/(x^2 + 1) dx = atan(x).
auto F5 = integrate(Expr::sum({x.pow(i(2)), i(1)}).pow(i(-1)), "x").value();

// Rational bridge:         int x/(x^2 + 1) dx = (1/2) ln(x^2 + 1).
auto F6 = integrate(
    Expr::product({x, Expr::sum({x.pow(i(2)), i(1)}).pow(i(-1))}), "x").value();

// Definite integral:       int_0^1 x^2 dx = 1/3.
auto A  = integrate_definite(x.pow(i(2)), "x", i(0), i(1)).value();

// Honest failure:          int e^(x^2) dx -> not_implemented (no guessed answer).
auto E  = integrate(Expr::apply("exp", {x.pow(i(2))}), "x");   // has_value() == false

// Every indefinite result verifies: d/dx(F1) == x^2, etc.
auto check = differentiate(F1, "x");   // simplifies to x^2
```

The tests in `tests/symint_tests.cpp` verify each indefinite result **both** by its
expected closed form **and** by differentiating it back to the integrand.

## Relationship to the rest of the engine

`symint` is the `Expr`-facing front end of integration; it holds the elementary
table and linearity/substitution logic and delegates the hard rational case:

- [`diff`](diff.md) — the inverse operation, and the oracle behind the
  differentiate-back guarantee; its derivative-table spellings dictate the
  antiderivative spellings used here.
- [`integrate`](integrate.md) — the exact rational-function integrator
  (`integrate_rational` → `RationalIntegral`) this module bridges an `Expr` into.
- [`ratint`](ratint.md) / [`rothstein`](rothstein.md) — the Hermite and
  Rothstein–Trager halves underneath the rational bridge (source of `LogTerm`).
- [`polyexpr`](polyexpr.md) — the `Expr` ↔ `Polynomial` bridge used to recognise a
  rational integrand and rebuild the result.
- Polynomial expansion is done here directly via linearity (there is no separate
  `expand` module in the tree); a future `expand.md` layer would slot in ahead of
  the rational bridge.

## See also

- [`nimblecas.diff`](diff.md) — differentiation, the verifying inverse.
- [`nimblecas.integrate`](integrate.md) — exact rational-function integration.
- [`nimblecas.ratint`](ratint.md) — Hermite reduction (rational part).
- [`nimblecas.rothstein`](rothstein.md) — Rothstein–Trager (logarithmic part).
- [`nimblecas.polyexpr`](polyexpr.md) — `Expr` ↔ `Polynomial` bridge.
- [`nimblecas.limits`](limits.md) — the neighbouring analysis operation.
- [Documentation hub](../Index.md)
