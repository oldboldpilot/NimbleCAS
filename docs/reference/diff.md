# `nimblecas.diff` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/diff/diff.cppm`

Symbolic differentiation. `differentiate(u, var)` applies `d/dx` structurally
and returns the automatically-simplified result. The operation is **total**: an
unknown or multi-argument function yields an unevaluated `Derivative(u, x)`
placeholder rather than failing (Rule 32 — no exceptions).

```cpp
import nimblecas.diff;
```

Depends on [`core`](core.md), [`symbolic`](symbolic.md),
[`simplify`](simplify.md), [`parallel`](parallel.md), and [`cache`](cache.md).

## API

```cpp
[[nodiscard]] auto differentiate(const Expr& u, std::string_view var) -> Result<Expr>;
```

Differentiates `u` with respect to the symbol named `var`. Internally it runs a
raw (memoized) derivative pass and then feeds the result through
[`simplify`](simplify.md); the returned `Result` therefore carries any
`MathError` that simplification of the derivative surfaces (e.g. overflow).

## Differentiation rules

| Node | Rule |
| :--- | :--- |
| constant | `0` |
| symbol | `1` if it is `var`, else `0` |
| sum | `d(Σ fᵢ) = Σ fᵢ'` (sum rule) |
| product | `d(Π fᵢ) = Σᵢ ( fᵢ' · Πⱼ≠ᵢ fⱼ )` (Leibniz / general product rule) |
| power | `d(fᵍ) = fᵍ · ( g'·ln(f) + g·f'/f )` (general power rule) |
| function `f(arg)` | chain rule `f'(arg) · arg'` when `f` is a known unary function |

Sum terms and Leibniz summands are independent, so for large subtrees
(`size() >= parallel::parallel_cost_threshold`, i.e. 512) they are computed
concurrently via `parallel::transform_index_if` (deterministic, order-
preserving).

## Known-function derivative table

Derivatives of known **unary** functions come from a built-in table (the outer
factor `f'(arg)`, combined with `arg'` by the chain rule):

| Group | Functions and derivatives |
| :--- | :--- |
| exp / log / roots | `exp' = exp`, `ln' = 1/u`, `sqrt' = (1/2)·u^(-1/2)` |
| trigonometric | `sin' = cos`, `cos' = -sin`, `tan' = 1 + tan²`, `cot' = -(1 + cot²)`, `sec' = sec·tan`, `csc' = -csc·cot` |
| inverse trig | `asin' = (1 - u²)^(-1/2)`, `acos' = -(1 - u²)^(-1/2)`, `atan' = 1/(1 + u²)` |
| hyperbolic | `sinh' = cosh`, `cosh' = sinh`, `tanh' = 1 - tanh²`, `asinh' = (1 + u²)^(-1/2)`, `acosh' = (u² - 1)^(-1/2)`, `atanh' = 1/(1 - u²)` |
| special | `erf' = (2/√π)·exp(-u²)`, `erfc' = -(2/√π)·exp(-u²)`, `gamma' = gamma·digamma`, `lambertW'(u) = W(u) / (u·(1 + W(u)))` |

## Unknown functions

A function whose name is not in the table, or any function with more than one
argument, differentiates to the unevaluated placeholder
`Derivative(u, var)` — built as `Expr::apply("Derivative", {u, Expr::symbol(var)})`.
This keeps `differentiate` total.

## Memoization

`differentiate` uses a per-call [`ExprMemo`](cache.md) over the raw-derivative
pass. Since `var` is fixed for the call, keying on the operand alone is
sufficient, so repeated operands (e.g. Jacobian/Hessian assembly that
differentiates the same subexpressions many times) are differentiated once.
Subtrees below `memo_threshold` (32 nodes) skip the memo.

## Example

```cpp
import nimblecas.diff;
using namespace nimblecas;

const Expr x = Expr::symbol("x");

auto d1 = differentiate(x.pow(Expr::integer(2)), "x");   // 2*x
auto d2 = differentiate(Expr::apply("sin", {x}), "x");   // cos(x)

// unknown function -> unevaluated placeholder
auto d3 = differentiate(Expr::apply("g", {x}), "x");     // Derivative(g(x), x)
```

## See also

- [`nimblecas.simplify`](simplify.md) — post-processes every derivative.
- [`nimblecas.symbolic`](symbolic.md) — the `Expr` model.
- [Documentation hub](../Index.md)
