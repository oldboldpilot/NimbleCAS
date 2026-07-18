# `nimblecas.currency` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/currency/currency.cppm`

Exact currency and foreign-exchange arithmetic: a tagged `Money` amount, an
exchange-rate **graph** (currencies are nodes, quoted pairs are directed edges),
exact cross-rate derivation by shortest-path product through the graph, currency
conversion that quantizes to the target's minor unit only at the boundary,
triangular-arbitrage detection, and covered-interest-parity forward rates.
Bid/ask spreads are modelled by directional edges — `a→b` and `b→a` need not be
reciprocals.

## Honesty boundary

Rates, cross rates, and forwards are **EXACT rationals** — there is no rounding
until the final `convert()` quantizes to money under a caller-supplied scale and
`Rounding`. The two honesty commitments the tests pin down:

- **A missing route returns `MathError::not_implemented`, never a fabricated
  rate.** If no path through the quoted graph connects two currencies, the module
  says so rather than inventing one.
- **`Money` refuses to add across currencies** (`MathError::domain_error`) rather
  than silently combining apples and oranges. Same-currency arithmetic is exact.

Everything upstream of `convert()` is exact over Q: cross-rate products,
triangular-arbitrage deviation, and CIP forwards are computed on
[`BigRational`](bigrational.md), so a consistent triangle's product is **exactly**
1 and an arbitrage is detectable at **zero tolerance**.

```cpp
import nimblecas.currency;
```

Depends on [`core`](core.md), [`bigint`](bigint.md),
[`bigrational`](bigrational.md), and [`bigdecimal`](bigdecimal.md). Everything
lives in namespace `nimblecas::currency`.

## `Money` — a tagged exact amount

A signed [`BigDecimal`](bigdecimal.md) amount plus an opaque currency `code`
(e.g. `"USD"`, `"EUR"`). Arithmetic across differing codes is refused.

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `of` | `static auto of(BigDecimal amount, std::string code) -> Money` | Wrap an existing decimal amount. |
| `parse` | `static auto parse(std::string_view amount, std::string code) -> Result<Money>` | Parse `"123.45"` into `code` at the literal's own scale. Malformed → the underlying `BigDecimal` error (`syntax_error` / `overflow`). |
| `amount` | `auto amount() const noexcept -> const BigDecimal&` | The exact decimal amount. |
| `code` | `auto code() const noexcept -> const std::string&` | The currency tag. |
| `is_zero` | `auto is_zero() const noexcept -> bool` | Amount is zero. |
| `negate` | `auto negate() const -> Money` | Sign-flipped, same currency. |
| `add` | `auto add(const Money& o) const -> Result<Money>` | Same-currency exact addition. Differing codes → `domain_error`. |
| `subtract` | `auto subtract(const Money& o) const -> Result<Money>` | Same-currency exact subtraction. Differing codes → `domain_error`. |
| `rounded` | `auto rounded(std::int32_t scale, Rounding mode) const -> Money` | Re-express at a stated minor-unit scale (e.g. 2 for cents) under `mode`. |
| `to_string` | `auto to_string() const -> std::string` | `"90.00 EUR"` — amount then code. |

## `Quote` — a directed quoted rate

`struct Quote { std::string base; std::string quote; BigRational rate; }` —
`1` unit of `base` buys `rate` units of `quote`. `rate` must be `> 0`.

## `RateTable` — the FX rate graph

Add directed quotes (optionally auto-adding the reciprocal), then query direct or
cross rates, convert money, and detect triangular arbitrage. Small graphs (dozens
of currencies), so a hashed adjacency with BFS pathfinding is ample. Static
factory `create()`.

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `add` | `auto add(std::string base, std::string quote, const BigRational& rate, bool add_reciprocal = true) -> Result<std::reference_wrapper<RateTable>>` | Add `base→quote` at `rate`. When `add_reciprocal`, also add `quote→base` at `1/rate` (a frictionless market); omit it to model a bid/ask spread with independent directions. `rate <= 0` → `domain_error`. |
| `direct_rate` | `auto direct_rate(std::string_view base, std::string_view quote) const -> std::optional<BigRational>` | The directly-quoted rate if present, else `nullopt`. |
| `cross_rate` | `auto cross_rate(std::string_view base, std::string_view quote) const -> Result<BigRational>` | Exact cross rate as the product along the **shortest path** (fewest hops; deterministic because neighbours are visited in insertion order and nodes marked on first discovery). Identity `base == quote` returns `1`. No route → `not_implemented`. |
| `convert` | `auto convert(const Money& amount, std::string_view to_code, std::int32_t scale, Rounding mode) const -> Result<Money>` | Convert `amount` into `to_code`, quantized to `scale` minor units under `mode`. The **only** rounding step. No route → `not_implemented`. |
| `triangular_product` | `auto triangular_product(std::string_view a, std::string_view b, std::string_view c) const -> Result<BigRational>` | Exact product of the cross rates around the cycle `a→b→c→a`. Exactly `1` means no triangular arbitrage; `> 1` a profitable loop (the gross multiple). A broken leg propagates its error. |
| `has_triangular_arbitrage` | `auto has_triangular_arbitrage(std::string_view a, std::string_view b, std::string_view c, const BigRational& tolerance) const -> Result<bool>` | `true` iff the product deviates from `1` by more than `tolerance`. Because `tolerance` is a `BigRational`, an **exact zero-tolerance** test is possible. |

## Covered-interest-parity forward

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `forward_rate` | `auto forward_rate(const BigRational& spot, const BigRational& rate_base, const BigRational& rate_quote, const BigRational& year_fraction) -> Result<BigRational>` | `F = S · (1 + r_quote·t) / (1 + r_base·t)`, exact over Q. `S` is quote units per base unit, `r_*` simple interest over the tenor, `t` the year fraction. A zero base-leg denominator → `division_by_zero`. |

## Error model

| Condition | Error |
| :--- | :--- |
| `Money::add` / `Money::subtract` across differing currency codes | `MathError::domain_error` |
| `Money::parse` of a malformed amount | `MathError::syntax_error` (or `overflow`), propagated from `bigdecimal` |
| `RateTable::add` with `rate <= 0` | `MathError::domain_error` |
| `cross_rate` / `convert` / `triangular_product` / `has_triangular_arbitrage` when no path connects the currencies | `MathError::not_implemented` |
| `forward_rate` with a zero base-leg denominator (`1 + rate_base·t == 0`) | `MathError::division_by_zero` |

`direct_rate` returns `std::optional` (absence is not an error); `to_string`,
`negate`, `rounded`, and the accessors are total.

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.bigdecimal;
import nimblecas.currency;
using namespace nimblecas;
using namespace nimblecas::currency;

auto q = [](std::string_view s) { return BigRational::from_string(s).value(); };

// Direct rate, auto-reciprocal, and boundary-quantized conversion.
auto table = RateTable::create();
(void)table.add("USD", "EUR", q("9/10"));                 // 1 USD -> 0.9 EUR
table.direct_rate("USD", "EUR").value() == q("9/10");     // true
table.direct_rate("EUR", "USD").value() == q("10/9");     // reciprocal auto-added
auto m = table.convert(Money::parse("100", "USD").value(), "EUR", 2, Rounding::half_even);
m->amount().to_string();                                  // "90.00"

// Cross rate by graph pathfinding: no direct USD->GBP quote.
auto graph = RateTable::create();
(void)graph.add("USD", "EUR", q("9/10"));
(void)graph.add("EUR", "GBP", q("8/10"));
graph.cross_rate("USD", "GBP").value() == q("18/25");     // 0.9 * 0.8 == 0.72
graph.cross_rate("USD", "USD").value() == BigRational::from_int(1);   // identity
graph.cross_rate("USD", "JPY").error() == MathError::not_implemented; // no route (honest)

// Triangular arbitrage, detected exactly. A reciprocal-consistent triangle -> product 1.
auto fair = RateTable::create();
(void)fair.add("USD", "EUR", q("9/10"));
(void)fair.add("EUR", "GBP", q("8/10"));
(void)fair.add("GBP", "USD", q("25/18"));                 // exact inverse of 0.72
fair.triangular_product("USD", "EUR", "GBP").value() == BigRational::from_int(1);
fair.has_triangular_arbitrage("USD", "EUR", "GBP", BigRational::from_int(0)).value();  // false

// Covered-interest-parity forward: F = 100 * (1 + 0.02) / (1 + 0.05) == 680/7.
forward_rate(BigRational::from_int(100), q("1/20"), q("1/50"),
             BigRational::from_int(1)).value() == q("680/7");   // true

// Money refuses cross-currency addition.
auto usd = Money::parse("10.00", "USD").value();
auto eur = Money::parse("10.00", "EUR").value();
usd.add(eur).error() == MathError::domain_error;          // true
usd.add(Money::parse("5.00", "USD").value()).value().amount()
    == BigDecimal::from_string("15.00").value();          // USD + USD == 15.00
```

## See also

- [`nimblecas.bigdecimal`](bigdecimal.md) — `Money` is a tagged `BigDecimal`; the
  scale is the currency's minor unit.
- [`nimblecas.bigrational`](bigrational.md) — the exact rate arithmetic before the
  boundary quantize.
- [`nimblecas.finance`](finance.md) — the TVM/fixed-income sibling; its
  `currency_swap_value` complements the CIP forward here.
- [Documentation hub](../Index.md)
