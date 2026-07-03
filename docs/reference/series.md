# `nimblecas.series` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/series/series.cppm`

Taylor series expansion (ROADMAP §7.3) as thin, **exact** compositions over the
symbolic engines. `taylor_coefficients` returns the coefficients
`c_0 … c_order` of `f` expanded about a point, where
`c_k = f^{(k)}(point) / k!`; `taylor_polynomial` assembles those into the
truncated series `Σ_{k=0}^{order} c_k (var − point)^k`. Each `c_k` is obtained by
repeatedly [differentiating](diff.md) `f` (reusing the diff engine),
[substituting](symbolic.md) the expansion point for the variable, and dividing by
the running factorial `k!` — and every coefficient (and the assembled polynomial)
is passed through automatic [simplification](simplify.md), so for concrete inputs
the coefficients collapse **exactly**.

```cpp
import nimblecas.series;
```

Depends on [`core`](core.md), [`symbolic`](symbolic.md), [`diff`](diff.md),
and [`simplify`](simplify.md).

## The overflow contract

The factorial `k!` is accumulated in an `int64` with **overflow detection**
(Rule 32): an `order` large enough that `k!` would wrap `int64` fails cleanly with
`MathError::overflow` rather than producing a wrong reciprocal. (`20!` fits; `21!`
does not.) Any error raised by the underlying [`differentiate`](diff.md),
[`simplify`](simplify.md), or rational construction (`Expr::rational`) propagates
unchanged along the `Result` railway.

## API

```cpp
[[nodiscard]] auto taylor_coefficients(const Expr& f, std::string_view var,
                                       const Expr& point, std::int64_t order)
    -> Result<std::vector<Expr>>;

[[nodiscard]] auto taylor_polynomial(const Expr& f, std::string_view var,
                                     const Expr& point, std::int64_t order)
    -> Result<Expr>;
```

### `taylor_coefficients`

The coefficients `c_0 … c_order` of `f` expanded in `var` about `point`, where
`c_k = f^{(k)}(point) / k!`, each automatically simplified. The result vector has
exactly `order + 1` entries (`c_0` through `c_order`). Internally the derivative
`g_k` starts at `g_0 = f` and is advanced by one `differentiate(·, var)` per step,
so each successive coefficient reuses the previous derivative rather than
re-differentiating from scratch; the running factorial tracks `k!` alongside.

### `taylor_polynomial`

The truncated Taylor polynomial `Σ_{k=0}^{order} c_k (var − point)^k`, simplified.
The shift `(var − point)` is represented as `var + (−1)·point` so the simplifier
can fold it — at `point = 0` it collapses back to `var`, so the reconstruction
returns to the original monomial basis. Note the simplifier does **not** expand
binomial powers such as `(x − 1)^2`, so about a non-zero point the reconstruction
is left in the shifted `(var − point)^k` basis (equal as a function, not
structurally identical to the expanded form).

## Error model

| Condition | Error |
| :--- | :--- |
| `order < 0` | `MathError::domain_error` |
| `order` large enough that `k!` overflows `int64` | `MathError::overflow` |
| An error from `differentiate` / `simplify` / `Expr::rational` | propagated unchanged |

`taylor_polynomial` calls `taylor_coefficients` first, so it propagates every one
of these (a negative order surfaces as the same `domain_error`).

## Worked examples

From `tests/series_tests.cpp` (polynomial inputs are used so every coefficient is
exact and deterministic):

```cpp
import nimblecas.series;
import nimblecas.symbolic;
using namespace nimblecas;

const Expr x = Expr::symbol("x");

// f = x^3 about 0, order 3:  c_0 = c_1 = c_2 = 0,  c_3 = 1
auto c1 = taylor_coefficients(Expr::power(x, Expr::integer(3)), "x",
                              Expr::integer(0), 3).value();
//   -> { 0, 0, 0, 1 }   (four coefficients c_0..c_3)

// taylor_polynomial(x^3, 0, 3) reconstructs x^3 exactly (point 0 needs no
// binomial expansion)
auto p1 = taylor_polynomial(Expr::power(x, Expr::integer(3)), "x",
                            Expr::integer(0), 3).value();      // x^3

// f = x^2 + 2x + 1 about 0, order 2:  c_0 = 1, c_1 = 2, c_2 = 1
auto c2 = taylor_coefficients(
    Expr::sum({Expr::power(x, Expr::integer(2)),
               Expr::product({Expr::integer(2), x}), Expr::integer(1)}),
    "x", Expr::integer(0), 2).value();                        // { 1, 2, 1 }

// f = x^2 about the point 1, order 2:  x^2 = 1 + 2(x-1) + (x-1)^2
//   c_0 = 1, c_1 = 2, c_2 = 1
auto c3 = taylor_coefficients(Expr::power(x, Expr::integer(2)), "x",
                              Expr::integer(1), 2).value();    // { 1, 2, 1 }
// taylor_polynomial(x^2, 1, 2) = 1 + 2(x-1) + (x-1)^2 — equal to x^2 as a
// function, left in the shifted basis (binomials are not expanded).

// a negative order is rejected
taylor_coefficients(Expr::power(x, Expr::integer(2)), "x",
                    Expr::integer(0), -1).error();   // MathError::domain_error
```

## See also

- [`nimblecas.diff`](diff.md) — the differentiation engine each coefficient
  reuses, one derivative per order.
- [`nimblecas.simplify`](simplify.md) — folds every coefficient and the assembled
  polynomial to canonical form.
- [`nimblecas.symbolic`](symbolic.md) — the `Expr` model and `substitute`.
- [Documentation hub](../Index.md)
