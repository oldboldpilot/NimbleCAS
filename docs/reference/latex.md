# `nimblecas.latex` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/latex/latex.cppm`

LaTeX math export. `to_latex(u)` walks an immutable symbolic `Expr` and renders
it as a LaTeX math string (no surrounding `$...$` delimiters). Parenthesisation
is driven by a four-level precedence lattice (sum < product < power < atom): a
sub-expression is wrapped in `\left( ... \right)` exactly when its outermost
operator binds more loosely than the position it sits in. The walk is **total**
(Rule 32 — no exceptions): every `ExprNode` alternative is handled and a
non-exhaustive visitor fails to compile.

```cpp
import nimblecas.latex;
```

Depends on [`core`](core.md) and [`symbolic`](symbolic.md).

## API

```cpp
[[nodiscard]] auto to_latex(const Expr& u) -> std::string;
```

Renders `u` as a LaTeX math string without `$...$` delimiters. It is a pure,
allocation-only traversal — it never fails and returns no `Result`.

## Rendering conventions

The renderer maps each node kind onto conventional mathematical notation:

| Construct | Rendering |
| :--- | :--- |
| symbol | verbatim, except a small [Greek map](#greek-letters) (`pi → \pi`, `Omega → \Omega`, …) |
| integer | the decimal literal; a negative sign is peelable so sums fold it into a subtraction |
| rational `p/q` (`q ≠ 1`) | `\frac{\|p\|}{q}` with the sign kept **outside** the fraction |
| rational `p/1` | the bare integer `p` |
| sum | terms joined by ` + `; a negative term folds into ` - ` with its magnitude |
| product | numeric factors fold into a sign plus a rational numerator/denominator; `base^{-n}` factors move to the denominator; the rest are juxtaposed; a non-empty denominator yields `\frac{...}{...}` |
| product spacing | factors juxtaposed with a space; `\cdot` is inserted **only** between two adjacent numerals so they do not fuse into one number |
| power (general) | `base^{exp}`, the base parenthesised unless it is an atom |
| power, exponent `1/2` | `\sqrt{base}` |
| power, exponent `-1` | `\frac{1}{base}` |
| power, exponent `-n` (`n > 1`) | `\frac{1}{base^{n}}` |
| function `f(args)` | a [control word](#function-names) `\sin`, `\exp`, … or `\operatorname{name}`, with the argument list in `\left( ... \right)` |
| `sqrt(x)` (unary function) | `\sqrt{x}` |

### Precedence and parenthesisation

Four precedence levels drive wrapping:

```
prec_sum (1)  <  prec_product (2)  <  prec_power (3)  <  prec_atom (4)
```

A fragment is wrapped in `\left( ... \right)` exactly when its precedence is
**strictly below** the minimum required by its position:

- Sum factors inside a product are wrapped (`(x + y) z`).
- Sum or product bases under a power are wrapped (`(x + 1)^{2}`, `(x y)^{2}`).
- Atoms — symbols, numbers, function applications, `\frac{}{}` and `\sqrt{}`
  — are self-delimiting and never wrap.

A leading unary minus is advertised as a peelable sign: a negative constant or
product reports sum-level precedence, so it is parenthesised in a tighter
context and folded into a subtraction inside a sum.

### Greek letters

Lowercase symbol names spelled like a LaTeX Greek command map to that command
(`alpha → \alpha`, `pi → \pi`, `omega → \omega`, …), as do the capitals that
have their own command (`Gamma`, `Delta`, `Theta`, `Lambda`, `Xi`, `Pi`,
`Sigma`, `Phi`, `Psi`, `Omega`). Every other name renders verbatim.

### Function names

Known elementary functions map to their control word: `sin`, `cos`, `tan`,
`cot`, `sec`, `csc`, `sinh`, `cosh`, `tanh`, `exp`, `log`, `ln`, `arcsin`,
`arccos`, `arctan`, `det`, `gcd`, `max`, `min`. Any other name uses
`\operatorname{name}` so it still typesets upright and correctly spaced.

## Examples

Each row is asserted verbatim in `tests/latex_tests.cpp`:

| Expression | `to_latex` output |
| :--- | :--- |
| `symbol("x")` | `x` |
| `symbol("pi")` | `\pi` |
| `symbol("Omega")` | `\Omega` |
| `symbol("foo")` | `foo` |
| `integer(42)` | `42` |
| `integer(-7)` | `-7` |
| `rational(3, 4)` | `\frac{3}{4}` |
| `rational(-1, 2)` | `-\frac{1}{2}` |
| `rational(6, 3)` | `2` |
| `x + y` | `x + y` |
| `x + (-3)·y` | `x - 3 y` |
| `2·x^2` | `2 x^{2}` |
| `(1/2)·x` | `\frac{x}{2}` |
| `x·y^{-1}` | `\frac{x}{y}` |
| `(-1)·x` | `-x` |
| `x^2` | `x^{2}` |
| `x^{1/2}` | `\sqrt{x}` |
| `x^{-1}` | `\frac{1}{x}` |
| `x^{-2}` | `\frac{1}{x^{2}}` |
| `(x + 1)^2` | `\left(x + 1\right)^{2}` |
| `(x y)^2` | `\left(x y\right)^{2}` |
| `sin(x)` | `\sin\left(x\right)` |
| `exp(x)` | `\exp\left(x\right)` |
| `sqrt(x)` | `\sqrt{x}` |
| `lambertW(x)` | `\operatorname{lambertW}\left(x\right)` |
| `sin(x^2) + 2·x` | `\sin\left(x^{2}\right) + 2 x` |

```cpp
import nimblecas.latex;
using namespace nimblecas;

const Expr x = Expr::symbol("x");

to_latex(Expr::power(x, Expr::integer(2)));           // "x^{2}"
to_latex(Expr::power(x, Expr::integer(-2)));          // "\frac{1}{x^{2}}"
to_latex(Expr::apply("sin", {x}));                    // "\sin\left(x\right)"

// sin(x^2) + 2 x
const Expr e = Expr::sum({Expr::apply("sin", {Expr::power(x, Expr::integer(2))}),
                          Expr::product({Expr::integer(2), x})});
to_latex(e);                                          // "\sin\left(x^{2}\right) + 2 x"
```

## Lineage

ROADMAP §7.12 — the LaTeX math exporter. It renders any [`Expr`](symbolic.md),
so it composes with the whole symbolic chain: the LaTeX of a
[differentiated](diff.md) or [simplified](simplify.md) expression is just
`to_latex` applied to the result.

## See also

- [`nimblecas.symbolic`](symbolic.md) — the `Expr` model it walks.
- [`nimblecas.diff`](diff.md) — produces expressions to render.
- [`nimblecas.simplify`](simplify.md) — canonicalises expressions before export.
- [Documentation hub](../Index.md)
