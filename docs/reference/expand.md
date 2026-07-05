# `nimblecas.expand` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/expand/expand.cppm`

Algebraic **expansion**: the counterpart to [`simplify`](simplify.md).
`simplify` deliberately *never* distributes — it keeps `x*(y+z)` factored so
that like terms can be grouped by structural identity. `expand(u)` does the
opposite: it multiplies everything out into a flat sum of monomials and then
runs `simplify` to collect like terms.

```cpp
import nimblecas.expand;
```

Depends on [`core`](core.md), [`symbolic`](symbolic.md),
[`simplify`](simplify.md), and [`combinatorics`](combinatorics.md) (for the
binomial coefficients).

## API

```cpp
[[nodiscard]] auto expand(const Expr& u) -> Result<Expr>;
```

Returns the expanded, simplified form of `u`, or a `MathError` if — and only if
— the final `simplify` pass hits a condition it already owns (integer overflow
while folding a genuinely huge coefficient/constant, or an undefined form such
as `0^0`). Expansion itself never fails: an expression it cannot or will not
expand is returned intact, exact and unchanged.

## What it does

| Transformation | Example |
| :--- | :--- |
| Distribute a product over sums | `(a + b)*(c + d) -> a*c + a*d + b*c + b*d` |
| Distribute a factor over a sum | `a*(b + c) -> a*b + a*c` |
| Binomial power of a sum | `(x + 1)^3 -> x^3 + 3x^2 + 3x + 1` |
| Multinomial power of a sum | `(a + b + c)^2 -> a^2 + b^2 + c^2 + 2ab + 2ac + 2bc` |
| Recurse into subexpressions, then simplify | `((x + 1)^2)*(x + 2) -> x^3 + 4x^2 + 5x + 2` |

Powers of sums use the binomial theorem
`(f + r)^n = Σ_{k=0}^{n} C(n,k) f^k r^(n-k)`, applied iteratively so an
`m`-term base unfolds into its full multinomial expansion. The coefficients
`C(n,k)` come from [`combinatorics::binomial`](combinatorics.md). After the
distributed tree is built, `expand` calls [`simplify`](simplify.md) once to fold
constants, canonically order operands, and combine like terms.

## Honesty boundary

Expansion is **exact**: every rewrite replaces an expression with a
mathematically equal one. A power is expanded **only** when its exponent is a
**literal non-negative integer** (an integer `ConstantNode`, or a reduced
rational with denominator 1) **no greater than the exponent cap**. In every
other case the power is returned intact — this is not an error, it is the honest
unexpanded-but-exact answer:

| Exponent of a sum base | Result |
| :--- | :--- |
| literal non-negative integer `≤ 64` | expanded via binomial / multinomial |
| symbolic (e.g. `(x+1)^n`) | left intact |
| negative (e.g. `(x+1)^-2`) | left intact |
| rational / real (e.g. `(x+1)^(1/2)`) | left intact |
| literal integer `> 64` (over cap) | left intact |

A power whose base is **not a sum** (e.g. `(x*y)^3`) is also left for `simplify`
— `expand` distributes powers of *sums*, not powers of products.

## Blow-up guard

Expansion of a sum can grow super-linearly, so it is bounded two ways; exceeding
either leaves the relevant subtree **undistributed but exact** (never an error,
never an unbounded loop):

- **Exponent cap — `max_expand_exponent = 64`.** A literal integer exponent
  above this is left unexpanded. The value is chosen so that a two-term base's
  binomial coefficients always stay within `int64`: the largest, `C(64,32) ≈
  1.8e18`, is below `INT64_MAX ≈ 9.2e18`.
- **Term cap — `max_expand_terms = 4096`.** Before expanding a power the
  estimated monomial count `C(n + m − 1, m − 1)` (for an `m`-term base raised to
  `n`) is checked; and every distribution of two sums checks the size of its
  cross product. If either would exceed the cap (or the estimate overflows
  `int64`), the node is left as a raw power / product.

Because the exponent cap keeps every step bounded, `expand` always terminates.
Coefficients for a *multinomial* base may still be large; if the final
`simplify` cannot represent one within `int64` it returns
`MathError::overflow` (the same honest railway error `simplify` uses elsewhere).

## Example

```cpp
import nimblecas.expand;
using namespace nimblecas;

const Expr x = Expr::symbol("x");
const Expr one = Expr::integer(1);

auto a = expand(x.add(one).pow(Expr::integer(2)));   // x^2 + 2x + 1
auto b = expand(Expr::product({x, x.add(one)}));     // x^2 + x

// symbolic exponent: returned intact, still a power
auto c = expand(Expr::power(x.add(one), Expr::symbol("n")));  // (x + 1)^n
```

## See also

- [`nimblecas.simplify`](simplify.md) — canonicalises the expanded tree; does
  **not** distribute (which is why `expand` exists).
- [`nimblecas.symbolic`](symbolic.md) — the `Expr` model and node kinds.
- [`nimblecas.combinatorics`](combinatorics.md) — the `binomial` coefficients.
- [Documentation hub](../Index.md)
```

