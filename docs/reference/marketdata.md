# `nimblecas.marketdata` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/marketdata/marketdata.cppm`

A **provider-agnostic market-data feed**. It defines one normalised model —
`Quote`, `Bar`/`BarSeries`, `OptionQuote`/`OptionChain`, `RateQuote` — and a thin
**adapter per external provider** (Yahoo Finance, Alpha Vantage, Alpaca) that maps
that provider's JSON response into the normalised shape. Everything downstream
(pricing, the option/futures strategy engines, portfolio analytics) consumes the
normalised model and never sees a provider's wire format. **Adding a provider is one
`DataProvider` enumerator plus one adapter function** — no downstream code changes.
JSON is decoded by the vendored [`fastjson`](https://github.com/oldboldpilot/fastestjsoninthewest)
(fastestjsoninthewest, Code Policy Rule 37); this module never hand-rolls a parser.

## Honesty boundary

This is a pure **data-ingestion boundary** — it performs **no pricing math and
invents no values**. It faithfully copies provider fields into the typed model; a
**missing or wrong-typed field is an honest `MathError`, never a silent zero dressed
up as a real quote**. The failure map:

- Malformed JSON that `fastjson` rejects → `MathError::syntax_error`.
- A required field absent, present but the wrong JSON type, or a numeric string that
  does not parse → `MathError::domain_error`.
- A provider/response shape no adapter handles → `MathError::not_implemented`.

Numeric quotes are IEEE `double` (provider values are already decimal). Timestamps
are **epoch seconds**; providers that quote calendar / RFC-3339 strings are converted
via the exact proleptic-Gregorian day count (`days_from_civil`), and a field that
cannot be interpreted as a time is left `0` (== unknown), never guessed. All failure
rides the railway (`Result<T>` / `MathError`); **nothing throws** — provider JSON is
untrusted input and is treated as such.

```cpp
import nimblecas.marketdata;
```

Depends on [`core`](core.md) (`Result` / `MathError`) and `fastjson` (the vendored
JSON parser). Everything lives in namespace `nimblecas::marketdata`. Only the core
`import fastjson` module is linked — the OpenMP-accelerated `fastjson_parallel` variant
is intentionally excluded (the canonical flags are TBB-only, Rule 50).

## Types

| Type | Description |
| :--- | :--- |
| `AssetClass` | `enum class { equity, index, fx, future, option, bond, money_market, crypto, commodity, unknown }` — drives which downstream engine consumes a quote. |
| `DataProvider` | `enum class { yahoo, alpha_vantage, alpaca }` — the source of a payload. Extensible: a new provider is a new enumerator + adapter. |
| `OptionRight` | `enum class { call, put }`. |
| `Quote` | Point-in-time quote: `symbol`, `asset_class`, `last` (the pricing spot), `bid`, `ask`, `open`, `high`, `low`, `previous_close`, `volume`, `currency`, `timestamp` (epoch s). `mid()` = two-sided mid or `last`; `spread()` = `ask − bid` or `0`. Fluent `with_symbol/with_last/with_asset_class`. |
| `Bar` | One OHLCV bar: `timestamp` (epoch s), `open`, `high`, `low`, `close`, `volume`. |
| `BarSeries` | `symbol`, `asset_class`, `bars`. `size()`/`empty()`; `close_returns()` = consecutive close-to-close simple returns (`size − 1`; empty if `< 2` bars). |
| `OptionQuote` | One listed contract: `contract_symbol`, `right`, `strike`, `expiration` (epoch s), `last`, `bid`, `ask`, `implied_volatility`, `volume`, `open_interest`. |
| `OptionChain` | `underlying`, `underlying_price`, `expiration`, `calls`, `puts`. |
| `RateQuote` | Rate instrument: `symbol`, `asset_class`, `yield` (decimal), `price` (per 100 face), `maturity` (epoch s), `timestamp`. The bridge into the fixed-income / money-market engines. |
| `Feed` | Fluent, composable, reusable façade binding a provider (+ optional default asset class). |

## Calendar helpers (exact)

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `days_from_civil` | `auto days_from_civil(std::int64_t y, unsigned m, unsigned d) noexcept -> std::int64_t` | Days from the Unix epoch (1970-01-01) to a proleptic-Gregorian date, by Hinnant's exact integer algorithm. `days_from_civil(1970,1,1) == 0`. |
| `epoch_seconds_from_iso` | `auto epoch_seconds_from_iso(std::string_view s) -> Result<std::int64_t>` | Parse `"YYYY-MM-DD"` (midnight UTC) or `"YYYY-MM-DDThh:mm:ss[.frac][Z]"` (trailing zone ignored) → epoch seconds. A string matching neither shape → `domain_error`, never a fabricated time. |

## Provider adapters

Each takes the **raw JSON body of a documented endpoint** and returns the normalised
model. Selecting the endpoint (URL) is the caller's concern; this layer only decodes a
body once you have it — keeping the module free of any network dependency and fully
deterministic under test.

### `namespace yahoo`

| Function | Endpoint | Behavior |
| :--- | :--- | :--- |
| `parse_quote` | `/v8/finance/chart/{sym}` | Reads `chart.result[0].meta` → `Quote` (`regularMarketPrice` → `last`, `regularMarketTime` → `timestamp`). Missing symbol/price → `domain_error`. |
| `parse_chart` | `/v8/finance/chart/{sym}` | `timestamp[]` + `indicators.quote[0]` OHLCV arrays → `BarSeries`. Null-padded gap dates are skipped honestly; a non-parallel array → `domain_error`. |
| `parse_option_chain` | `/v7/finance/options/{sym}` | First expiry's `calls`/`puts` → `OptionChain`. A side may be absent; a malformed leg → `domain_error`. |

### `namespace alpha_vantage`

| Function | Endpoint | Behavior |
| :--- | :--- | :--- |
| `parse_global_quote` | `function=GLOBAL_QUOTE` | The `"Global Quote"` object (every field a **numeric string**) → `Quote`. `"07. latest trading day"` → `timestamp`. |
| `parse_time_series_daily` | `function=TIME_SERIES_DAILY` | The date-keyed `"Time Series (Daily)"` map → `BarSeries`, **sorted ascending** (Alpha Vantage emits newest-first). |

### `namespace alpaca`

| Function | Endpoint | Behavior |
| :--- | :--- | :--- |
| `parse_latest_quote` | `/v2/stocks/{sym}/quotes/latest` | The two-sided `quote` (`bp`/`ap`) → `Quote`; `last := mid` (latest-quote has no trade). RFC-3339 `t` → `timestamp`. |
| `parse_bars` | `/v2/stocks/{sym}/bars` | The `bars[]` array (compact `o`/`h`/`l`/`c`/`v` + RFC-3339 `t`) → `BarSeries`. |

### Provider dispatch (the extensible front door)

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `parse_quote` | `auto parse_quote(DataProvider p, std::string_view json) -> Result<Quote>` | Dispatch a single quote to the provider's adapter. An unwired provider → `not_implemented`. |
| `parse_bars` | `auto parse_bars(DataProvider p, std::string_view json) -> Result<BarSeries>` | Dispatch a bar series (Yahoo chart / AV daily / Alpaca bars). |

## `Feed` — fluent, composable, reusable façade

`Feed` binds a provider (and an optional default asset class) once, then decodes any
number of payloads through a uniform, chainable surface — the same instance is reused
across responses. It holds no mutable parse state, so a single `const Feed` is safe to
share across threads.

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `from` | `static auto from(DataProvider p) -> Feed` | Construct a feed for provider `p`. |
| `as_asset_class` | `auto as_asset_class(AssetClass a) -> Feed&` | Stamp decoded quotes/series with `a` (fluent). |
| `provider` / `asset_class` | accessors | The bound provider / default asset class. |
| `quote` | `auto quote(std::string_view json) const -> Result<Quote>` | Provider-dispatched quote, tagged with the asset class. |
| `bars` | `auto bars(std::string_view json) const -> Result<BarSeries>` | Provider-dispatched bar series, tagged. |
| `option_chain` | `auto option_chain(std::string_view json) const -> Result<OptionChain>` | Yahoo option chain; any other provider → `not_implemented`. |

## Error model

| Condition | Error |
| :--- | :--- |
| JSON body `fastjson` cannot parse | `MathError::syntax_error` |
| A required field absent, present but wrong-typed, or a numeric string that does not parse; a non-parallel OHLCV array; `epoch_seconds_from_iso` on an unrecognised shape | `MathError::domain_error` |
| A provider path not wired (e.g. `Feed::option_chain` on a non-Yahoo provider, or `parse_quote`/`parse_bars` for a provider with no adapter) | `MathError::not_implemented` |

`Quote::mid`/`spread`, `BarSeries::size`/`close_returns`, and `days_from_civil` are
total; every adapter returns `Result<T>`. **No field is ever fabricated** — absence is
an error or a documented default, never a plausible-looking wrong number.

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.marketdata;
using namespace nimblecas;
using namespace nimblecas::marketdata;

// A Yahoo chart body (as returned by /v8/finance/chart/AAPL).
constexpr std::string_view body = R"({"chart":{"result":[{"meta":{
    "symbol":"AAPL","currency":"USD","regularMarketPrice":227.52,
    "regularMarketTime":1721260800}}]}})";

// Direct adapter, or the fluent Feed — same result.
const auto q1 = yahoo::parse_quote(body).value();            // q1.last == 227.52
const auto feed = Feed::from(DataProvider::yahoo).as_asset_class(AssetClass::equity);
const auto q2 = feed.quote(body).value();                    // q2.asset_class == equity

// The normalised quote flows straight into pricing as the spot:
//   auto spec = pricing::OptionSpec{}.with_spot(q2.last).with_strike(230)...

// Exact calendar: 2024-07-18 midnight UTC.
days_from_civil(2024, 7, 18);                                // 19922
epoch_seconds_from_iso("2024-07-18T20:00:00Z").value();      // 1721332800

// Honesty: a missing price is an error, not a zero quote.
yahoo::parse_quote(R"({"chart":{"result":[{"meta":{"symbol":"X"}}]}})")
    .error();                                                // MathError::domain_error
```

## See also

- [`nimblecas.pricing`](pricing.md) — the derivatives-pricing engine that consumes a
  normalised `Quote` as its spot.
- [`nimblecas.finance`](finance.md) / [`nimblecas.fixedincome`](fixedincome.md) —
  bond & money-market analytics fed by `RateQuote`.
- [`nimblecas.currency`](currency.md) — exact FX, downstream of an `fx` `Quote`.
- [Documentation hub](../Index.md)
