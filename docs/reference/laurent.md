# `nimblecas.laurent` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/laurent/laurent.cppm`

Truncated **Laurent series** over the rationals — elements of `Q((x))` carrying a
*finite* principal part of negative-power terms plus a truncated Taylor tail. Where
[`powerseries`](powerseries.md) models `Q[[x]] / (x^N)` (coefficients `c_0..c_{N-1}`
with an implicit `O(x^N)` tail), `Laurent` lifts the lower bound to an arbitrary
integer valuation: it stores coefficients `c_k` for exponents `k` in
`[order_min, truncation_order)`, i.e. a leading (lowest) exponent `order_min` — which
**may be negative** — and a dense vector of exact [`Rational`](ratpoly.md)
coefficients, with everything of exponent `>= truncation_order` discarded behind an
implicit `O(x^{truncation_order})` tail. No floating point ever enters; every
coefficient is an exact fraction.

```cpp
import nimblecas.laurent;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), and
[`powerseries`](powerseries.md) (the unit inversion and rational-function division
reuse the tested `PowerSeries` recurrences).

## Honesty boundary

The two ends of a `Laurent` are **not** symmetric, and this is the whole point of
the type:

- **The upper end is truncated.** Coefficients at exponents `>= truncation_order()`
  are **unknown**. A result is only valid up to its tracked truncation order — exactly
  the discipline [`powerseries`](powerseries.md) enforces with its `O(x^N)` tail.
- **The lower end is exact.** The principal part of a Laurent series is *finite* by
  definition, so every coefficient at an exponent `< order_min()` is a **genuine
  zero**, not a truncation. This is why `valuation()` and `residue()` can return exact
  answers even though the series is truncated above.

Binary operations combine the two operands' tracked ranges honestly rather than
assuming a shared order:

- `add` / `subtract` take `order_min = min` of the two lower bounds and
  `truncation_order = min` of the two upper bounds — the result is honest only up to
  where **both** inputs are known.
- `multiply` shifts the valuation (`order_min` values **add**) and keeps the **smaller
  relative precision** `min(size(a), size(b))` — the honest number of coefficients the
  two truncated inputs jointly determine (a truncated Cauchy product cannot manufacture
  precision neither input had).
- `inverse` factors out the valuation `v` (writing the series as `x^v · unit`, `unit`
  a power series with nonzero constant term), inverts the `unit` with the
  `PowerSeries` recurrence, and returns a series of valuation `-v` and the same relative
  precision.

As with the rest of the engine the honest failure mode is **overflow**: because
`Rational` is backed by `int64`, an accumulating numerator or denominator that would
exceed range surfaces as `MathError::overflow` rather than silently wrapping (Rule 32).
Exponent bookkeeping (`order_min + size`, valuation shifts in `multiply`) is likewise
overflow-checked.

## `Laurent` — an element of `Q((x))`

### Factories

All return `Result<Laurent>`. A series must track at least one coefficient, so an empty
coefficient list / `size == 0` is a `domain_error`; `order_min + size` overflowing the
`int64` exponent range is an `overflow`.

| Factory | Signature | Behavior |
| :--- | :--- | :--- |
| `from_coeffs` | `static auto from_coeffs(std::int64_t order_min, std::vector<Rational> coeffs) -> Result<Laurent>` | Primary constructor: `coeffs[i]` is the coefficient of `x^{order_min + i}`. Leading/trailing zeros are **retained** as written (they carry precision information). |
| `zero` | `static auto zero(std::int64_t order_min, std::size_t size) -> Result<Laurent>` | The zero series tracked over `[order_min, order_min + size)`. |
| `constant` | `static auto constant(const Rational& c, std::size_t size) -> Result<Laurent>` | `c + O(x^size)` (`order_min == 0`). |
| `one` | `static auto one(std::size_t size) -> Result<Laurent>` | `1 + O(x^size)` (`order_min == 0`). |
| `monomial` | `static auto monomial(const Rational& c, std::int64_t exponent, std::int64_t order_min, std::size_t size) -> Result<Laurent>` | The single term `c·x^exponent` over `[order_min, order_min + size)`. Requires `order_min <= exponent < order_min + size` (else `domain_error` — the term would fall outside the window and be lost). |
| `from_power_series` | `static auto from_power_series(const PowerSeries& p) -> Result<Laurent>` | Lift `c_0..c_{N-1} + O(x^N)` into a Laurent series with `order_min == 0`, `truncation_order == N`. |
| `from_rational_function` | `static auto from_rational_function(const RationalPoly& num, const RationalPoly& den, const Rational& point, std::size_t order) -> Result<Laurent>` | Laurent-expand `num/den` about `point`, in powers of `(x - point)`, keeping `order` coefficients of relative precision. See below. |

#### `from_rational_function` — expansion about a point (including a pole)

The rational function `num/den` is expanded about `x = point` in powers of
`(x - point)`, **exactly over `Q`**. The valuation is discovered exactly: the
numerator and denominator are Taylor-shifted to `t = x - point`, their trailing-zero
multiplicities `v_num`, `v_den` are read off, and the result carries valuation
`v = v_num - v_den`. When `point` is a **pole** (`v_den > v_num`) this is negative and a
genuine principal part appears. The returned series has `order_min == v` and
`truncation_order == v + order`, i.e. exactly `order` tracked coefficients
`c_v .. c_{v+order-1}`. The coefficients themselves are exact (the rational function is
not itself an approximation); `order` simply chooses how many terms to compute.

`den` must not be the zero polynomial (`division_by_zero`); `order == 0` is a
`domain_error`. If `num` vanishes identically the quotient is the zero series
`0 + O(x^order)` at `order_min 0`.

### Accessors

Total (infallible) — plain values, not `Result`.

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `order_min` | `auto order_min() const noexcept -> std::int64_t` | Lowest tracked exponent (may be negative). Exponents strictly below are genuine zeros. |
| `truncation_order` | `auto truncation_order() const noexcept -> std::int64_t` | Exponent of the `O()` tail: coefficients at exponents `>= this` are **unknown**. Equals `order_min() + size()`. |
| `size` | `auto size() const noexcept -> std::size_t` | Number of tracked coefficients (the relative precision). |
| `coefficient` | `auto coefficient(std::int64_t k) const -> Rational` | Coefficient of `x^k`. Returns `0` for `k < order_min()` (a **genuine** zero) and also `0` for `k >= truncation_order()` (there the `0` is the **truncation** — query `truncation_order()` to tell them apart). |
| `coefficients` | `auto coefficients() const noexcept -> std::span<const Rational>` | Read-only view; index `i` is the coefficient of `x^{order_min() + i}`. |
| `to_string` | `auto to_string(std::string_view var = "x") const -> std::string` | Human-readable form, e.g. `"x^-1 + 1 + x + O(x^3)"`. |
| `is_equal` | `auto is_equal(const Laurent& o) const -> bool` | Structural equality: same `order_min` and identical coefficient vector. |

### Structure

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `valuation` | `auto valuation() const -> Result<std::int64_t>` | Exponent of the first nonzero tracked coefficient. `domain_error` when the tracked part is entirely zero (the valuation is `>= truncation_order` and cannot be determined). |
| `principal_part` | `auto principal_part() const -> Result<Laurent>` | The finite negative-power part (exponents `< 0`), as a Laurent whose truncation order is `min(0, truncation_order())`. The zero series `0 + O(x)` when there are no negative-power terms. |
| `regular_part` | `auto regular_part() const -> Result<Laurent>` | The regular (Taylor) part (exponents `>= 0`), keeping the original `O(x^{truncation_order()})` tail. The zero series `0 + O(x)` when nothing of exponent `>= 0` is tracked. |
| `residue` | `auto residue() const -> Result<Rational>` | The coefficient `c_{-1}`. A **genuine** `0` when `order_min() > -1`; `domain_error` when `x^{-1}` lies at or beyond the truncation order (its value is unknown). |

### Arithmetic (overflow-checked, exact)

Each returns `Result<Laurent>` and flows through the overflow-checked `Rational`
operations.

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `add` | `auto add(const Laurent& o) const -> Result<Laurent>` | Sum over the union range: `order_min = min`, `truncation_order = min`. |
| `subtract` | `auto subtract(const Laurent& o) const -> Result<Laurent>` | Difference over the union range (same bookkeeping as `add`). |
| `scale` | `auto scale(const Rational& s) const -> Result<Laurent>` | Multiply every coefficient by `s` (`order_min` and truncation unchanged). |
| `multiply` | `auto multiply(const Laurent& o) const -> Result<Laurent>` | Cauchy product with valuation shift: `order_min` adds, relative precision is `min(size(a), size(b))`. |
| `inverse` | `auto inverse() const -> Result<Laurent>` | Multiplicative inverse via `x^v · unit` factoring; the result has valuation `-v`. `domain_error` when the tracked part is entirely zero (no invertible leading term). |
| `divide` | `auto divide(const Laurent& o) const -> Result<Laurent>` | `this * o.inverse()` (`o`'s tracked part must have a nonzero leading term). |

## Error model

| Condition | Error |
| :--- | :--- |
| `from_coeffs` with an empty list; `zero`/`constant`/`one`/`monomial` with `size == 0`; `from_rational_function` with `order == 0` | `MathError::domain_error` |
| `monomial` with `exponent` outside `[order_min, order_min + size)` | `MathError::domain_error` |
| `from_rational_function` with a zero-polynomial `den` | `MathError::division_by_zero` |
| `valuation` / `inverse` / `divide` when the tracked part has no nonzero coefficient | `MathError::domain_error` |
| `residue` when `x^{-1}` is at or beyond the truncation order | `MathError::domain_error` |
| `order_min + size` (or a `multiply` valuation shift) exceeding the `int64` exponent range | `MathError::overflow` |
| An `int64` numerator or denominator wrapping during any coefficient arithmetic | `MathError::overflow` |

The accessors (`order_min`, `truncation_order`, `size`, `coefficient`, `coefficients`,
`to_string`, `is_equal`) are total and never error.

## Worked examples

```cpp
import nimblecas.laurent;
import nimblecas.ratpoly;
using namespace nimblecas;

auto ri   = [](std::int64_t v) { return Rational::from_int(v); };
auto lau  = [](std::int64_t lo, std::vector<std::int64_t> vs) {
    std::vector<Rational> c;
    for (auto v : vs) c.push_back(Rational::from_int(v));
    return Laurent::from_coeffs(lo, std::move(c)).value();
};
auto poly = [](std::vector<std::int64_t> vs) {
    std::vector<Rational> c;
    for (auto v : vs) c.push_back(Rational::from_int(v));
    return RationalPoly::from_coeffs(std::move(c));
};

// 1/(x - x^2) has valuation -1 (a simple pole at 0) and residue 1.
auto s   = lau(1, {1, -1, 0, 0, 0});         // x - x^2 + O(x^6)
auto inv = s.inverse().value();              // x^{-1} + 1 + x + x^2 + x^3 + O(x^4)
inv.valuation().value();                     // -1
inv.residue().value();                       // 1
inv.coefficient(-1);                         // 1
inv.coefficient(-2);                         // 0  (genuine zero, below the valuation)

// Valuations add under multiplication.
auto a = lau(-2, {2, 0, 3});                 // 2 x^{-2} + 3
auto b = lau(1, {5, 7});                     // 5 x + 7 x^2
auto p = a.multiply(b).value();
p.valuation().value();                       // -1  ==  (-2) + 1
p.coefficient(-1);                           // 10  ==  2*5

// Principal / regular split of x^{-2} + 2x^{-1} + 3 + 4x + 5x^2 + O(x^3).
auto L  = lau(-2, {1, 2, 3, 4, 5});
auto pp = L.principal_part().value();        // x^{-2} + 2 x^{-1}
auto rp = L.regular_part().value();          // 3 + 4x + 5x^2 + O(x^3)
pp.coefficient(-1);                          // 2
rp.coefficient(0);                           // 3

// Laurent expansion of a rational function about a pole:
// 1/(x^2 - 1) about x = 1  ->  (1/2)(x-1)^{-1} - 1/4 + (1/8)(x-1) - ...
auto lr = Laurent::from_rational_function(poly({1}), poly({-1, 0, 1}), ri(1), 4).value();
lr.order_min();                              // -1
lr.residue().value();                        // 1/2  (residue at the simple pole x=1)
lr.coefficient(0);                           // -1/4

// Reciprocal round-trip is 1 to the tracked order.
auto t     = lau(2, {3, 1, 4, 1});           // 3x^2 + x^3 + 4x^4 + x^5
auto ident = t.multiply(t.inverse().value()).value();
ident.coefficient(0);                        // 1
ident.coefficient(3);                        // 0

// The honesty boundary: a residue past the truncation order is not silently 0.
lau(-5, {1, 1}).residue().error();           // domain_error (x^{-1} is beyond O(x^{-3}))
Laurent::zero(0, 3).value().inverse().error();  // domain_error (no invertible term)
```

## See also

- [`nimblecas.powerseries`](powerseries.md) — the `Q[[x]]/(x^N)` truncated power series
  this module extends and reuses for unit inversion.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field and the `RationalPoly`
  numerator/denominator inputs to `from_rational_function`.
- [`nimblecas.bigpowerseries`](bigpowerseries.md) — arbitrary-precision power series,
  lifting the `int64` overflow ceiling.
- [Documentation hub](../Index.md)
```
