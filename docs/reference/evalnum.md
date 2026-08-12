# `nimblecas.evalnum` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/evalnum/evalnum.cppm`

The **numerical sampling counterpart** to the exact symbolic engine: evaluates
a `nimblecas.symbolic` `Expr` to an IEEE-754 `double` at a point, or under a
map of variable bindings.

```cpp
import nimblecas.evalnum;
```

Depends on [`core`](core.md) (`Result` / `MathError`) and
[`symbolic`](symbolic.md) (the `Expr` tree it walks — `ConstantNode`,
`SymbolNode`, `AddNode`, `MulNode`, `PowerNode`, `FunctionNode`).

## Honesty boundary

This module is **numerical, not exact**. It is the numeric-sampling
counterpart to the exact symbolic engine ([`symbolic`](symbolic.md)): every
leaf and intermediate result is an IEEE-754 `double`, so results carry
ordinary floating-point rounding error, unlike the exact rational /
exact-per-term arithmetic elsewhere in this codebase. The symbols `"pi"` and
`"e"` are given their numeric `std::numbers` mappings **here purely for
evaluation purposes** — that is a rendering-time convenience, not a claim that
they are exact constants in this module (contrast
[`symconst`](symconst.md), whose `Expr` leaves stay exact until a numeric
bridge is invoked).

Two failure modes are never silently swallowed (Rule 32):

- **An unbound symbol or an unrecognised function head is never guessed at**
  (never silently `0`) — it is an honest `MathError::domain_error`.
- **A non-finite intermediate** (`NaN`/`Inf`, e.g. from `log` of a negative
  number or a division producing `Inf`) is caught **immediately after the
  node that produced it** and converted to `MathError::domain_error`, so a
  caller never receives a `NaN`/`Inf` dressed up as a valid `double`.

## The function whitelist

`FunctionNode`s are evaluated only through an explicit whitelist; anything
else is `domain_error`, never a best-effort fallback:

`sin`, `cos`, `tan`, `cot` (`= cos/sin`), `sec` (`= 1/cos`), `csc` (`= 1/sin`),
`sinh`, `cosh`, `tanh`, `exp`, `log`, `sqrt`, `abs`.

Every whitelisted function takes exactly one argument; a `FunctionNode` with
any other arity is `domain_error`.

## API

All entry points are free functions in namespace `nimblecas`, `[[nodiscard]]`.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `eval_double` (single var) | `auto eval_double(const Expr& e, std::string_view var, double x) -> Result<double>` | Evaluate `e` to a `double`, substituting `x` for every occurrence of the symbol named `var`. Delegates to the map overload with a single binding. |
| `eval_double` (bindings) | `auto eval_double(const Expr& e, const std::unordered_map<std::string, double>& bindings) -> Result<double>` | Evaluate `e` using `bindings` to resolve symbols. `"pi"`/`"e"` resolve to `std::numbers::pi`/`std::numbers::e` when not shadowed by an explicit binding of the same name. Any other unbound symbol, or a function outside the whitelist, is `domain_error`; any non-finite intermediate is `domain_error`. |

`ConstantNode` values are evaluated directly: an `int64_t` constant converts
to `double` as-is, a `double` constant passes through, and a rational
`(num, den)` constant evaluates as `double(num) / double(den)` (so an exact
rational leaf loses exactness the instant it enters this module — that is the
point of the honesty boundary above).

## Error model

| Condition | Error |
| :--- | :--- |
| A symbol not in `bindings` and not `"pi"`/`"e"` | `MathError::domain_error` |
| A `FunctionNode` whose name is outside the whitelist | `MathError::domain_error` |
| A `FunctionNode` with an arity other than 1 | `MathError::domain_error` |
| Any node's evaluated value is non-finite (`NaN`/`Inf`) | `MathError::domain_error` |

## Worked example

```cpp
import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.evalnum;
using namespace nimblecas;

// x^2 + 1 at x = 3 -> 10.0 exactly (in double).
auto x = Expr::symbol("x");
auto expr = Expr::sum({Expr::power(x, Expr::integer(2)), Expr::integer(1)});
auto result = eval_double(expr, "x", 3.0).value();
// result == 10.0

// A rational leaf 3/4 evaluates to 0.75 exactly-in-double.
eval_double(Expr::rational(3, 4).value(), "x", 0.0).value();  // 0.75

// sin(0) == 0 exactly; sin(pi) via the "pi" symbol agrees with std::sin(pi) to a few ULP.
eval_double(Expr::apply("sin", {Expr::integer(0)}), "x", 0.0).value();       // 0.0
eval_double(Expr::apply("sin", {Expr::symbol("pi")}), "x", 0.0).value();     // ~1.2e-16

// Two-variable map overload: x*y at {x:2, y:5} -> 10.0.
auto prod = Expr::product({Expr::symbol("x"), Expr::symbol("y")});
const std::unordered_map<std::string, double> bindings{{"x", 2.0}, {"y", 5.0}};
eval_double(prod, bindings).value();  // 10.0

// Honest errors: never a silent 0 or a propagated NaN.
eval_double(Expr::symbol("y"), "x", 1.0).error();                          // domain_error (unbound)
eval_double(Expr::apply("lambertW", {Expr::integer(1)}), "x", 0.0).error(); // domain_error (unknown fn)
eval_double(Expr::apply("log", {Expr::integer(-1)}), "x", 0.0).error();     // domain_error (non-finite)
```

## See also

- [`nimblecas.symbolic`](symbolic.md) — the exact `Expr` tree this module
  samples; the honesty boundary this module deliberately crosses.
- [`nimblecas.symconst`](symconst.md) — named constant `Expr` leaves (`pi`,
  `e`, `gamma`, `phi`) that stay exact symbolically until bridged to a numeric
  value, in contrast with the always-numeric `"pi"`/`"e"` handling here.
- [`nimblecas.numeric`](numeric.md) — floating-point polynomial root-finders,
  a sibling numerical layer over a narrower (polynomial-only) domain.
- [Documentation hub](../Index.md)
