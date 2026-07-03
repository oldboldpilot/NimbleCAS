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
| `f + g` | `L{f} + L{g}` | linearity over sums |
| `c · g` | `c · L{g}` | `c` a constant factor (free of `t`) |

The `exp` / `sin` / `cos` entries require an argument of the linear form `a·t`,
where the coefficient `a` is free of `t` (a bare `t` gives `a = 1`). Linearity is
applied structurally: a sum is transformed term by term, and a product is split
into its factors free of `t` (pulled out as constants) times the single remaining
`t`-dependent factor. A product with **more than one** `t`-dependent factor
(e.g. `t · sin(t)`) is outside the table.

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
| The factorial `n!` in `L{t^n}` overflows `int64` | `MathError::overflow` |

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

// unhandled forms
laplace_transform(Expr::apply("log", {t}), "t", "s").error();  // not_implemented
laplace_transform(Expr::product({t, Expr::apply("sin", {t})}),
                  "t", "s").error();                           // not_implemented
```

## See also

- [`nimblecas.simplify`](simplify.md) — reduces the unevaluated transform to its
  canonical form.
- [`nimblecas.symbolic`](symbolic.md) — the `Expr` model, `free_of`, and the node
  kinds the table dispatches over.
- [`nimblecas.diff`](diff.md) — the sibling calculus operation over the same
  `Expr` engine.
- [Documentation hub](../Index.md)
