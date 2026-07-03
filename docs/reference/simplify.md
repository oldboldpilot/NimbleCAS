# `nimblecas.simplify` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/simplify/simplify.cppm`

Automatic simplification following Joel S. Cohen's **Automatically Simplified
Algebraic Expression (ASAE)** rules. `simplify(u)` transforms an arbitrary
`Expr` into its canonical form so that structurally-distinct-but-mathematically-
equal inputs converge to the same tree.

```cpp
import nimblecas.simplify;
```

Depends on [`core`](core.md), [`symbolic`](symbolic.md),
[`parallel`](parallel.md), and [`cache`](cache.md).

## API

```cpp
[[nodiscard]] auto simplify(const Expr& u) -> Result<Expr>;
```

Returns the automatically-simplified form of `u`, or a `MathError` if an
undefined form or an integer overflow is encountered.

## What it does

| Transformation | Example |
| :--- | :--- |
| Exact rational constant folding (overflow-checked) | `2/4 + 1/4 -> 3/4` |
| Algebraic identities | `u+0 -> u`, `u*1 -> u`, `u*0 -> 0`, `u^0 -> 1`, `u^1 -> u`, `1^u -> 1` |
| Flatten nested sums / products | `x + (y + z) -> x + y + z` |
| Canonical ordering of operands | `y + x -> x + y` (sorted, so `x+y` and `y+x` agree) |
| Combine like terms | `n*x + m*x -> (n+m)*x` |
| Combine like bases | `x^a * x^b -> x^(a+b)` |
| Fold constant powers | `2^3 -> 8`, `(v^w)^n -> v^(w*n)` for integer `n` |

Sums and products are canonicalised: constants are folded into a single
coefficient/constant, non-constant terms are grouped by structural equality
(**not** by a `to_string` key, which is not injective), and the surviving
operands are sorted into a canonical order.

## Numeric domain

Integers and exact rationals are folded **precisely** with overflow detection.
Any `double` constant in a group degrades that group to `double` arithmetic
(`ConstVal::as_double`). Rational arithmetic uses overflow-checked `int64`
helpers throughout:

- addition `a/b + c/d = (a*d + c*b)/(b*d)`, multiplication, and integer power
  (via exponentiation-by-squaring, O(log exp), so a crafted large exponent
  cannot hang the simplifier) all return `MathError::overflow` on wrap;
- results are re-canonicalised through `make_number` (reduce by gcd; collapse
  `den == 1` to an integer; guard `INT64_MIN` and zero denominators).

## Undefined forms and errors

| Input | Result |
| :--- | :--- |
| `0^0` | `MathError::undefined_value` |
| `0^n`, `n < 0` (or non-positive) | `MathError::division_by_zero` |
| `0^n`, `n > 0` | `0` |
| integer/rational overflow during folding | `MathError::overflow` |

Errors propagate through the railway model (Rule 32) — no exceptions.

## Performance: parallelism + memoization

- **Parallel recursion.** For a subtree whose `size()` reaches
  `parallel::parallel_cost_threshold` (512), the independent operands of a sum,
  product, or function application are simplified concurrently via
  `parallel::transform_index_if`. Because `Expr` is immutable this is race-free,
  and the order-preserving map keeps the result deterministic.
- **Hash-consing (memoization).** Each `simplify` call creates a per-call
  [`ExprMemo`](cache.md). Subtrees at or above `memo_threshold` (32 nodes) have
  their simplified result cached and keyed by structural identity, so an
  expression that reuses the same subexpression (a DAG) simplifies each unique
  subtree exactly once — even under the parallel recursion. Small nodes skip the
  memo, since the lookup/lock would cost more than recomputing.

## Example

```cpp
import nimblecas.simplify;
using namespace nimblecas;

const Expr x = Expr::symbol("x");

auto a = simplify(x.add(x));                     // 2*x
auto b = simplify(x.add(Expr::integer(0)));      // x
auto c = simplify(Expr::integer(2).pow(Expr::integer(3)));  // 8

auto bad = simplify(Expr::integer(0).pow(Expr::integer(0))); // MathError::undefined_value
assert(!bad);
```

## See also

- [`nimblecas.symbolic`](symbolic.md) — the `Expr` model.
- [`nimblecas.cache`](cache.md) — the `ExprMemo` hash-cons table.
- [`nimblecas.diff`](diff.md) — differentiation simplifies its result through this module.
- [Documentation hub](../Index.md)
