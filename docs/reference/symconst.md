# `nimblecas.symconst` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/symconst/symconst.cppm`

Named **mathematical-constant leaves** for the symbolic engine — π, Euler's
number `e`, the Euler–Mascheroni constant γ, and the golden ratio φ — as
expression-tree atoms, plus the numeric bridge back to the arbitrary-precision
[`constants`](constants.md) provider. This module closes the long-open "π / `e`
as `Expr` leaves" item: the numeric layer already computes each of these to
arbitrary precision, but until now there was no way to carry one *inside* a
symbolic [`Expr`](symbolic.md) so an expression such as `2·π + e` stays exact
and structural until the caller explicitly asks for a number.

```cpp
import nimblecas.symconst;
```

Depends on [`nimblecas.symbolic`](symbolic.md), [`nimblecas.constants`](constants.md),
[`nimblecas.bigfloat`](bigfloat.md), and [`nimblecas.core`](core.md).

## Design — reserved-name symbols, not a new core node kind

A mathematical constant here is simply an ordinary `SymbolNode` carrying a
canonical **reserved name** (`"pi"`, `"e"`, `"gamma"`, `"phi"`). The module
deliberately does **not** add a new alternative to the `ExprNode` variant. That
is the load-bearing, low-risk choice: the symbolic variant is depended on by
every visitor in the engine (structural equality, `free_of`, `substitute`,
[`differentiate`](diff.md), [`simplify`](simplify.md), `to_latex`, …), each of
which already handles `SymbolNode` exhaustively. A constant leaf therefore
inherits all of that machinery for free, and the "dependent-false"
exhaustiveness guards in those visitors stay satisfied — nothing in the heavily
depended-on symbolic core has to change.

The reserved spellings were chosen to render well through `nimblecas.latex`,
whose Greek-letter table maps `"pi" → \pi`, `"gamma" → \gamma`, and
`"phi" → \phi`; `"e"` renders verbatim as the upright letter `e`.

## What "constant" means here — the honest boundary

- **Symbolically**, these leaves are constants only in the sense that they are
  **free of every variable**. That is the entire reason differentiation and
  simplification already do the right thing with no special-casing: `d/dx` of a
  `SymbolNode` whose name is not the differentiation variable is `0` (see
  [`diff`](diff.md)), and [`simplify`](simplify.md) treats an unknown symbol as
  an **opaque atom**. So π, `e`, γ, and φ need no support code in either engine
  — `diff` yields derivative `0` and `simplify` leaves them unchanged.
- **Numerically**, their value is a transcendental (π, `e`, γ) or
  irrational-algebraic (φ) real with **no exact rational representation**.
  `evaluate_constant` therefore returns a **rounded `BigFloat`, never an exact
  `Rational`**, mirroring the precision discipline documented in
  [`constants`](constants.md): a high-precision, correctly-rounded float, not an
  exact number.

## Reserved-name caveat — the deliberate trade-off

Because a constant is just a named symbol, a user variable literally named
`"pi"` (`Expr::symbol("pi")`) **aliases** the constant: `is_named_constant`
reports it as `pi` and `evaluate_constant` hands back `3.14159…`. The same holds
for the spellings `"e"`, `"gamma"`, and `"phi"`. This collision is the accepted
cost of not growing the core variant; the four reserved spellings are documented
here so callers can avoid them for ordinary free variables. (The
elementary-function derivative table in [`diff`](diff.md) already uses
`Expr::symbol("pi")` for exactly this constant, so the reservation is consistent
with the rest of the engine.)

## Numeric evaluation — single leaves versus compounds

- A **single named-constant leaf** dispatches straight to
  [`constants`](constants.md) and evaluates to its **correctly-rounded value at
  the requested precision** — the constants provider is already rounded to
  `prec`, so the leaf path skips the guard-bit elevation and avoids a second
  (double) rounding.
- A **compound expression** built only from constant leaves, integer / rational
  / real literals, the operators `+ − × ÷`, and **integer** powers `^` is
  evaluated recursively at an **elevated guard precision** (`prec + 64` bits) and
  **rounded once** to `prec`. Being a tree of rounded floats, the compound
  result is an approximation, not an exact number — again consistent with the
  constants module's precision discipline.

## API

All symbols are exported free functions in namespace `nimblecas`.

### Constant-leaf factories

Each returns the constant as an immutable `SymbolNode` with its canonical
reserved name. They are free of every variable, so they differentiate to `0` and
simplify as atoms via the existing engines with no further support code.

```cpp
[[nodiscard]] auto pi() -> Expr;            // the circle constant,          symbol "pi"
[[nodiscard]] auto e() -> Expr;             // Euler's number,               symbol "e"
[[nodiscard]] auto euler_gamma() -> Expr;   // Euler–Mascheroni constant,    symbol "gamma"
[[nodiscard]] auto golden_ratio() -> Expr;  // (1 + √5)/2,                   symbol "phi"
```

### Recognition

```cpp
[[nodiscard]] auto is_named_constant(const Expr& u) -> bool;
[[nodiscard]] auto named_constant_name(const Expr& u) -> std::optional<std::string_view>;
```

| Function | Behavior |
| :--- | :--- |
| `is_named_constant` | `true` iff `u` is one of the four reserved constant leaves. Subject to the aliasing caveat: any symbol spelled `"pi"` / `"e"` / `"gamma"` / `"phi"` satisfies it. |
| `named_constant_name` | The canonical reserved name of `u` if it is a named constant, else `std::nullopt`. The returned view points at **static storage** (not into `u`'s node), so it never dangles. |

### Numeric bridge

```cpp
[[nodiscard]] auto evaluate_constant(const Expr& u, std::int64_t prec) -> Result<BigFloat>;
[[nodiscard]] auto evaluate_constant_double(const Expr& u) -> Result<double>;
```

| Function | Behavior |
| :--- | :--- |
| `evaluate_constant` | Evaluate `u` to a correctly-rounded `BigFloat` of ~`prec` significant bits. A single reserved leaf dispatches straight to [`constants`](constants.md); a compound over constant leaves, int/rational/real literals, `+ − × ÷`, and integer `^` is evaluated at elevated precision (`prec + 64`) and rounded once to `prec`. |
| `evaluate_constant_double` | Convenience: evaluate `u` to a native `double`. Internally computes at `96` bits (~29 decimal digits), comfortably beyond a double's 53-bit mantissa, so the returned value is faithful to double rounding. |

## Error model

Following the rest of the engine, the bridge is **railway-oriented** (Rule 32):
it never throws, and every underlying `BigFloat` `Result` error is propagated
unchanged.

| Condition | Error |
| :--- | :--- |
| `prec <= 0` (non-positive requested precision), checked before any evaluation | `MathError::domain_error` |
| A node the bridge cannot evaluate numerically: a non-reserved (free) symbol, a function application such as `sin(pi)`, or a non-integer exponent such as `pi^(1/2)` | `MathError::not_implemented` |
| `prec` near `INT64_MAX`, so the `prec + 64` guard-precision addition would overflow (compound path); an `INT64_MIN` integer exponent whose magnitude is not representable | `MathError::overflow` |
| A `0^(−n)` reciprocal (a negative integer power of a zero base), propagated from the underlying `BigFloat::divide` | `MathError::division_by_zero` |

The recognition predicates `is_named_constant` / `named_constant_name` are total
and never error.

## Worked examples

```cpp
import nimblecas.symconst;
import nimblecas.symbolic;
import nimblecas.diff;
import nimblecas.core;
using namespace nimblecas;

// --- single named-constant leaves resolve to their constants value ---
evaluate_constant_double(pi()).value();            // ~ 3.14159265
evaluate_constant_double(e()).value();             // ~ 2.71828183
evaluate_constant_double(golden_ratio()).value();  // ~ 1.61803399
evaluate_constant_double(euler_gamma()).value();   // ~ 0.57721566

// The explicit-precision BigFloat path agrees.
evaluate_constant(pi(), 128).value().to_double();  // ~ 3.14159265

// --- recognition ---
is_named_constant(pi());                    // true
named_constant_name(golden_ratio());        // "phi"
is_named_constant(Expr::symbol("x"));       // false
named_constant_name(Expr::symbol("x"));     // std::nullopt

// --- symbolically a constant is free of every variable, so diff gives 0 ---
differentiate(pi(), "x").value()
    .is_equivalent_to(Expr::integer(0));    // true  (d(pi)/dx = 0)
// A constant factor is carried through: d(x·e)/dx = e.
differentiate(Expr::product({Expr::symbol("x"), e()}), "x").value()
    .is_equivalent_to(e());                 // true

// --- compounds over constant leaves + literals + (+ - * /) and integer ^ ---
const Expr two_pi = Expr::product({Expr::integer(2), pi()});
evaluate_constant_double(two_pi).value();               // ~ 6.28318531  (2*pi)

const Expr pi_sq = Expr::power(pi(), Expr::integer(2));
evaluate_constant_double(pi_sq).value();                // ~ 9.86960440  (pi^2)

const Expr inv_pi = Expr::power(pi(), Expr::integer(-1));
evaluate_constant_double(inv_pi).value();               // ~ 0.31830989  (pi^-1)

const Expr half_pi = Expr::product({Expr::rational(1, 2).value(), pi()});
evaluate_constant_double(half_pi).value();              // ~ 1.57079633  ((1/2)*pi)

const Expr e_plus_phi = Expr::sum({e(), golden_ratio()});
evaluate_constant_double(e_plus_phi).value();           // ~ 4.33631582  (e + phi)

// --- the reserved-name aliasing caveat, in action ---
is_named_constant(Expr::symbol("pi"));                  // true  (aliases the constant)
evaluate_constant_double(Expr::symbol("pi")).value();   // ~ 3.14159265

// --- error paths ---
evaluate_constant(Expr::symbol("x"), 64).error();       // not_implemented (free variable)
evaluate_constant(Expr::apply("sin", {pi()}), 64).error();
                                                        // not_implemented (function application)
evaluate_constant(Expr::power(pi(), Expr::rational(1, 2).value()), 64).error();
                                                        // not_implemented (non-integer exponent)
evaluate_constant(pi(), 0).error();                     // domain_error   (prec <= 0)
```

## See also

- [`nimblecas.symbolic`](symbolic.md) — the `Expr` trees and `SymbolNode` these
  constant leaves are built on.
- [`nimblecas.constants`](constants.md) — the arbitrary-precision numeric
  provider each leaf resolves to.
- [`nimblecas.bigfloat`](bigfloat.md) — the correctly-rounded arbitrary-precision
  float type every numeric result is carried in.
- [`nimblecas.diff`](diff.md) — symbolic differentiation, which yields `0` for a
  constant leaf with no special-casing.
- [Documentation hub](../Index.md)
