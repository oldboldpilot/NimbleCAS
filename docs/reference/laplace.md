# `nimblecas.laplace` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/laplace/laplace.cppm`

A **table-driven** symbolic Laplace transform (ROADMAP §7.6). Given `f(t)`,
`laplace_transform` computes the image `F(s) = L{f(t)}` by recognising a small
set of elementary forms in the variable `t` and combining them by **linearity**.
The transform is assembled unevaluated from `Expr` factories and then reduced by
automatic [simplification](simplify.md) to a canonical form. Anything outside the
table yields `MathError::not_implemented`, so — together with the overflow-checked
factorial — the operation is **total** (Rule 32, no exceptions).

```cpp
import nimblecas.laplace;
```

Depends on [`core`](core.md), [`symbolic`](symbolic.md), and
[`simplify`](simplify.md).

## The transform table

| Time function `f(t)` | Image `F(s)` | Condition |
| :--- | :--- | :--- |
| `c` | `c / s` | `c` free of `t` (constants, other symbols, subexpressions) |
| `t^n` | `n! / s^(n+1)` | `n` a positive integer |
| `e^(a t)` | `1 / (s − a)` | `a` free of `t` |
| `sin(a t)` | `a / (s² + a²)` | argument linear in `t` (`a·t`) |
| `cos(a t)` | `s / (s² + a²)` | argument linear in `t` (`a·t`) |
| `δ(t − a)` | `e^(−a s)` | Dirac delta; `δ(t)` gives `1` (argument `t − a`) |
| `u(t − a)` | `e^(−a s) / s` | Heaviside step; `u(t)` gives `1/s` (argument `t − a`) |
| `f + g` | `L{f} + L{g}` | linearity over sums |
| `c · g` | `c · L{g}` | `c` a constant factor (free of `t`) |

The `exp` / `sin` / `cos` entries require an argument of the linear form `a·t`,
where the coefficient `a` is free of `t` (a bare `t` gives `a = 1`). Linearity is
applied structurally: a sum is transformed term by term, and a product is split
into its factors free of `t` (pulled out as constants) times the single remaining
`t`-dependent factor. A product with **more than one** `t`-dependent factor
(e.g. `t · sin(t)`) is outside the table.

The Dirac and Heaviside entries require a **shifted** argument `t − a` (a lone `t`
gives `a = 0`), where `a` is free of `t`. `δ` is written `apply("dirac", {t − a})`
and `u` is written `apply("heaviside", {t − a})`. The general **second-shift theorem**
`L{u(t − a) · f(t − a)} = e^(−a s) · F(s)` is the identity behind the step entry; only
the `f ≡ 1` step (and the bare/shifted delta) are recognised structurally. A product
`u(t − a) · g(t)` is **not** guessed at — it returns `not_implemented` rather than a
wrong transform.

## Inverse transform

```cpp
[[nodiscard]] auto inverse_laplace(const Expr& F, std::string_view s,
                                   std::string_view t) -> Result<Expr>;
```

`inverse_laplace` computes `f(t) = L⁻¹{F(s)}` for the **rational-function class** — the
exactly-invertible case. `s` is the image variable (the symbol appearing in `F`) and
`t` is the time variable of the result. `F` is read as an exact ratio of polynomials in
`s` over ℚ (integer/rational constants, `s`, sums, products, and integer powers compose;
a foreign symbol, an inexact real leaf, or a transcendental subexpression makes `F`
non-rational → `not_implemented`). The ratio is decomposed with
[`partial_fractions`](pfd.md) and each term is inverted through the standard table:

| Partial-fraction term | Inverse `f(t)` |
| :--- | :--- |
| `C / (s − a)^k` (real pole `a`, multiplicity `k`) | `C · t^(k−1) · e^(a t) / (k−1)!` |
| `C / s^k` (the `a = 0` case) | `C · t^(k−1) / (k−1)!` |
| `(c₁ s + c₀) / ((s + α)² + ω²)` (complex-conjugate poles, `ω² > 0`) | `e^(−α t) · [ c₁ cos(ω t) + ((c₀ − c₁ α)/ω) sin(ω t) ]` |
| `(c₁ s + c₀) / ((s − r₁)(s − r₂))` (distinct real rational poles) | `A e^(r₁ t) + B e^(r₂ t)`, `A = C(r₁)/(r₁ − r₂)`, `B = C(r₂)/(r₂ − r₁)` |

Because `partial_fractions` groups the denominator by **square-free** factor, each
term's denominator base is a monic square-free polynomial. Linear bases (any
multiplicity) and quadratic bases (multiplicity 1) are inverted: a quadratic is
completed to `(s + α)² + ω²` with `α = p/2`, `ω² = q − α²`. When `ω² > 0` the factor is
irreducible (complex poles) and gives the `cos`/`sin` form — `ω` is exact when `ω²` is a
perfect rational square, otherwise carried symbolically as `ω = (ω²)^(1/2)`. When
`ω² < 0` the base factors into two distinct real poles, inverted when those poles are
**rational** (a perfect-square discriminant). A **constant** polynomial part of an
improper `F` inverts to a Dirac delta `c · δ(t)`.

The transform result is built from the same function-name spellings the forward table
uses (`"exp"`, `"sin"`, `"cos"`) plus `"dirac"`, so `inverse_laplace ∘ laplace_transform`
round-trips on every covered case, and is reduced by [simplification](simplify.md) before
it is returned.

### Honesty boundary

`inverse_laplace` never fabricates an inverse for an `F(s)` outside the supported class —
it returns `MathError::not_implemented` instead (Rule 32). Outside the class:

- a denominator factor of **degree > 2** (e.g. an irreducible cubic),
- **repeated complex poles** (a quadratic base with multiplicity > 1),
- **irrational real poles** (a reducible quadratic whose roots are not rational),
- a **non-constant polynomial part** (delta derivatives `δ′`, `δ″`, …),
- a **non-rational** `F` (a foreign symbol, an inexact real, or a transcendental term).

## API

```cpp
[[nodiscard]] auto laplace_transform(const Expr& f, std::string_view t,
                                     std::string_view s) -> Result<Expr>;
```

`t` is the transform (time) variable — the argument name expected in `f` — and `s`
is the image variable. Elementary forms are recognised by structure; the assembled
transform is simplified before it is returned.

## Error model

| Condition | Error |
| :--- | :--- |
| `f` (or a `t`-dependent subterm) matches no table entry | `MathError::not_implemented` |
| A product of two or more `t`-dependent factors (e.g. `t · sin(t)`) | `MathError::not_implemented` |
| An `exp`/`sin`/`cos` argument that is not linear in `t` | `MathError::not_implemented` |
| A `dirac`/`heaviside` argument that is not a shift `t − a` | `MathError::not_implemented` |
| The factorial `n!` in `L{t^n}` (or `(k−1)!` in the inverse) overflows `int64` | `MathError::overflow` |
| Inverse: `F(s)` is not a rational function of `s` | `MathError::not_implemented` |
| Inverse: a denominator factor the table does not cover (degree > 2, repeated complex poles, irrational real poles) | `MathError::not_implemented` |
| Inverse: a non-constant polynomial part (delta derivatives) | `MathError::not_implemented` |
| Inverse: an `int64` coefficient overflow in the exact ℚ arithmetic | `MathError::overflow` |

## Worked examples

From `tests/laplace_tests.cpp` (each expected image is built from the same `Expr`
factories the module uses, then simplified, so the assertions match the module's
own canonical output):

```cpp
import nimblecas.laplace;
import nimblecas.symbolic;
using namespace nimblecas;

const Expr t = Expr::symbol("t");

// L{1} = 1/s
laplace_transform(Expr::integer(1), "t", "s").value();              // s^(-1)

// L{t} = 1/s^2 ,  L{t^2} = 2/s^3
laplace_transform(t, "t", "s").value();                            // s^(-2)
laplace_transform(Expr::power(t, Expr::integer(2)), "t", "s").value();  // 2·s^(-3)

// L{exp(2t)} = 1/(s - 2)
laplace_transform(Expr::apply("exp", {Expr::product({Expr::integer(2), t})}),
                  "t", "s").value();                               // (s - 2)^(-1)

// L{sin(3t)} = 3/(s^2 + 9) ,  L{cos(t)} = s/(s^2 + 1)
laplace_transform(Expr::apply("sin", {Expr::product({Expr::integer(3), t})}),
                  "t", "s").value();                               // 3·(s^2 + 9)^(-1)
laplace_transform(Expr::apply("cos", {t}), "t", "s").value();      // s·(s^2 + 1)^(-1)

// linearity: L{2 + 3t} = 2/s + 3/s^2
laplace_transform(Expr::sum({Expr::integer(2),
                             Expr::product({Expr::integer(3), t})}),
                  "t", "s").value();                               // 2·s^(-1) + 3·s^(-2)

// Dirac / Heaviside
laplace_transform(Expr::apply("dirac", {t}), "t", "s").value();  // 1
laplace_transform(Expr::apply("dirac", {Expr::sum({t, Expr::integer(-3)})}),
                  "t", "s").value();                             // e^(-3 s)
laplace_transform(Expr::apply("heaviside", {t}), "t", "s").value();  // s^(-1)

// unhandled forms
laplace_transform(Expr::apply("log", {t}), "t", "s").error();  // not_implemented
laplace_transform(Expr::product({t, Expr::apply("sin", {t})}),
                  "t", "s").error();                           // not_implemented
```

### Inverse transform

```cpp
const Expr s = Expr::symbol("s");

// L^-1{1/(s - 2)} = e^(2 t)
inverse_laplace(Expr::power(Expr::sum({s, Expr::integer(-2)}), Expr::integer(-1)),
                "s", "t").value();                              // exp(2 t)

// L^-1{3/(s^2 + 9)} = sin(3 t) ;  L^-1{s/(s^2 + 1)} = cos(t)
// L^-1{1/s^2} = t ;  L^-1{1/(s - 1)^2} = t · e^t
// L^-1{1/(s^2 - 1)} = (1/2) e^t - (1/2) e^(-t)   (distinct real poles)
// L^-1{1} = δ(t)                                  (constant part → Dirac)

// honesty boundary — never a wrong inverse
inverse_laplace(Expr::power(Expr::sum({Expr::power(s, Expr::integer(3)), s,
                                       Expr::integer(1)}), Expr::integer(-1)),
                "s", "t").error();                              // not_implemented (cubic)
inverse_laplace(Expr::power(Expr::sum({Expr::power(s, Expr::integer(2)),
                                       Expr::integer(-2)}), Expr::integer(-1)),
                "s", "t").error();                              // not_implemented (irrational poles)
```

Every covered case **round-trips**: `inverse_laplace(laplace_transform(f))` recovers `f`
for `1`, `t`, `e^(2t)`, `sin(3t)`, and `cos(t)`.

## See also

- [`nimblecas.simplify`](simplify.md) — reduces the unevaluated transform to its
  canonical form.
- [`nimblecas.symbolic`](symbolic.md) — the `Expr` model, `free_of`, and the node
  kinds the table dispatches over.
- [`nimblecas.diff`](diff.md) — the sibling calculus operation over the same
  `Expr` engine.
- [`nimblecas.pfd`](pfd.md) — the partial-fraction decomposition the inverse transform
  drives, over [`nimblecas.ratpoly`](ratpoly.md) (`Rational`, `RationalPoly`).
- [`nimblecas.symint`](symint.md) — symbolic integration, a neighbouring exact operation.
- [`nimblecas.inteq`](inteq.md) — integral equations, a downstream consumer of the
  transform pair.
- [Documentation hub](../Index.md)
