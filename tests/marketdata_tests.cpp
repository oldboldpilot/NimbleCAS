// Tests for nimblecas.marketdata: provider-agnostic quote ingestion.
// @author Olumuyiwa Oluwasanmi
//
// This layer is a DATA BOUNDARY, so the suite drives it entirely from EMBEDDED JSON
// fixtures shaped exactly like each provider's real response (Yahoo Finance, Alpha
// Vantage, Alpaca) — no network, fully deterministic. It asserts: (1) each adapter maps
// the documented fields into the normalised model with hand-verified values; (2) the
// exact calendar helper (days_from_civil / epoch_seconds_from_iso) against known epochs;
// (3) the honesty contract — malformed JSON -> syntax_error, a missing/wrong-typed field
// -> domain_error, an unwired provider path -> not_implemented, never a fabricated value;
// (4) the fluent Feed façade dispatches by provider and stamps the asset class.

import std;
import nimblecas.core;
import nimblecas.marketdata;
import nimblecas.testing;

using namespace nimblecas::marketdata;
using nimblecas::MathError;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {
[[nodiscard]] auto close(double a, double b, double tol) -> bool { return std::abs(a - b) < tol; }

// --- Fixtures: real-shaped provider response bodies -------------------------

// Yahoo /v8/finance/chart/AAPL — meta block used by parse_quote.
constexpr std::string_view kYahooQuote = R"({
  "chart": {"result": [{"meta": {
    "symbol": "AAPL", "currency": "USD",
    "regularMarketPrice": 227.52, "previousClose": 225.01,
    "regularMarketDayHigh": 229.4, "regularMarketDayLow": 224.8,
    "regularMarketVolume": 41000000, "regularMarketTime": 1721260800
  }}], "error": null}
})";

// Yahoo chart with parallel timestamp[] / indicators.quote[0] OHLCV arrays, incl. a null gap.
constexpr std::string_view kYahooChart = R"({
  "chart": {"result": [{
    "meta": {"symbol": "AAPL"},
    "timestamp": [1721001600, 1721088000, 1721174400],
    "indicators": {"quote": [{
      "open":   [100.0, null, 102.0],
      "high":   [101.0, null, 103.0],
      "low":    [ 99.0, null, 101.5],
      "close":  [100.5, null, 102.5],
      "volume": [1000,  null, 1200]
    }]}
  }], "error": null}
})";

// Yahoo /v7/finance/options/AAPL — first expiry, one call + one put.
constexpr std::string_view kYahooOptions = R"({
  "optionChain": {"result": [{
    "underlyingSymbol": "AAPL",
    "quote": {"regularMarketPrice": 227.52},
    "options": [{
      "expirationDate": 1726704000,
      "calls": [{"contractSymbol":"AAPL240918C00230000","strike":230.0,"lastPrice":5.10,
                 "bid":5.00,"ask":5.20,"impliedVolatility":0.2841,"volume":1234,
                 "openInterest":5678,"expiration":1726704000}],
      "puts":  [{"contractSymbol":"AAPL240918P00220000","strike":220.0,"lastPrice":4.30,
                 "bid":4.20,"ask":4.40,"impliedVolatility":0.3012,"volume":890,
                 "openInterest":4321,"expiration":1726704000}]
    }]
  }]}
})";

// Alpha Vantage GLOBAL_QUOTE — every field a STRING.
constexpr std::string_view kAvQuote = R"({
  "Global Quote": {
    "01. symbol": "IBM", "02. open": "185.00", "03. high": "187.50",
    "04. low": "184.20", "05. price": "186.75", "06. volume": "3500000",
    "07. latest trading day": "2024-07-18", "08. previous close": "184.90",
    "09. change": "1.85", "10. change percent": "1.0006%"
  }
})";

// Alpha Vantage TIME_SERIES_DAILY — NEWEST FIRST (adapter must sort ascending).
// Custom delimiter: the JSON key "Time Series (Daily)" contains the sequence )" which
// would prematurely close a plain R"(...)" raw string.
constexpr std::string_view kAvDaily = R"JSON({
  "Meta Data": {"1. Information": "Daily", "2. Symbol": "IBM"},
  "Time Series (Daily)": {
    "2024-07-18": {"1. open":"185.00","2. high":"187.50","3. low":"184.20","4. close":"186.75","5. volume":"3500000"},
    "2024-07-17": {"1. open":"183.00","2. high":"185.10","3. low":"182.40","4. close":"184.90","5. volume":"3100000"}
  }
})JSON";

// Alpaca /v2/stocks/AAPL/quotes/latest — two-sided bp/ap + RFC-3339 t.
constexpr std::string_view kAlpacaQuote = R"({
  "symbol": "AAPL",
  "quote": {"t":"2024-07-18T20:00:00Z","bp":227.50,"ap":227.54,"bs":3,"as":5}
})";

// Alpaca /v2/stocks/AAPL/bars.
constexpr std::string_view kAlpacaBars = R"({
  "symbol": "AAPL",
  "bars": [
    {"t":"2024-07-17T00:00:00Z","o":183.0,"h":185.1,"l":182.4,"c":184.9,"v":3100000},
    {"t":"2024-07-18T00:00:00Z","o":185.0,"h":187.5,"l":184.2,"c":186.75,"v":3500000}
  ],
  "next_page_token": null
})";
}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.marketdata")
        .test("exact calendar: days_from_civil & epoch_seconds_from_iso",
              [](TestContext& t) {
                  t.expect(days_from_civil(1970, 1, 1) == 0, "epoch day == 0");
                  t.expect(days_from_civil(2000, 1, 1) == 10957, "2000-01-01 == 10957 days");
                  t.expect(days_from_civil(2024, 7, 18) == 19922, "2024-07-18 == 19922 days");
                  // 2024-07-18T00:00:00Z == 19922 * 86400.
                  t.expect(epoch_seconds_from_iso("2024-07-18").value() == 1721260800,
                           "date-only ISO -> midnight epoch");
                  t.expect(epoch_seconds_from_iso("2024-07-18T20:00:00Z").value() == 1721332800,
                           "RFC-3339 with time -> epoch + seconds");
                  t.expect(epoch_seconds_from_iso("1970-01-01T00:00:01Z").value() == 1,
                           "one second past epoch");
                  t.expect(!epoch_seconds_from_iso("not-a-date").has_value(),
                           "garbage -> error, not a fabricated time");
              })
        .test("Yahoo parse_quote maps meta block",
              [](TestContext& t) {
                  const auto q = yahoo::parse_quote(kYahooQuote).value();
                  t.expect(q.symbol == "AAPL", "symbol");
                  t.expect(close(q.last, 227.52, 1e-9), "regularMarketPrice -> last");
                  t.expect(close(q.previous_close, 225.01, 1e-9), "previousClose");
                  t.expect(close(q.high, 229.4, 1e-9) && close(q.low, 224.8, 1e-9), "day high/low");
                  t.expect(q.currency == "USD", "currency");
                  t.expect(q.timestamp == 1721260800, "regularMarketTime -> epoch");
                  // mid() falls back to last when no two-sided market present.
                  t.expect(close(q.mid(), 227.52, 1e-9), "mid falls back to last");
              })
        .test("Yahoo parse_chart builds OHLCV series and skips null gaps",
              [](TestContext& t) {
                  const auto s = yahoo::parse_chart(kYahooChart).value();
                  t.expect(s.symbol == "AAPL", "series symbol");
                  t.expect(s.size() == 2, "3 timestamps, 1 null gap skipped -> 2 bars");
                  t.expect(s.bars[0].timestamp == 1721001600, "first bar time");
                  t.expect(close(s.bars[0].close, 100.5, 1e-9), "first close");
                  t.expect(s.bars[1].timestamp == 1721174400, "second bar time (gap skipped)");
                  t.expect(close(s.bars[1].open, 102.0, 1e-9), "second open");
                  // close-to-close return of the two surviving bars.
                  const auto r = s.close_returns();
                  t.expect(r.size() == 1 && close(r[0], 102.5 / 100.5 - 1.0, 1e-12), "close return");
              })
        .test("Yahoo parse_option_chain splits calls and puts",
              [](TestContext& t) {
                  const auto ch = yahoo::parse_option_chain(kYahooOptions).value();
                  t.expect(ch.underlying == "AAPL", "underlying");
                  t.expect(close(ch.underlying_price, 227.52, 1e-9), "underlying price");
                  t.expect(ch.expiration == 1726704000, "expiry epoch");
                  t.expect(ch.calls.size() == 1 && ch.puts.size() == 1, "one call, one put");
                  t.expect(ch.calls[0].right == OptionRight::call && close(ch.calls[0].strike, 230.0, 1e-9),
                           "call strike 230");
                  t.expect(close(ch.calls[0].implied_volatility, 0.2841, 1e-9), "call IV");
                  t.expect(ch.puts[0].right == OptionRight::put && close(ch.puts[0].strike, 220.0, 1e-9),
                           "put strike 220");
                  t.expect(close(ch.puts[0].ask, 4.40, 1e-9), "put ask");
              })
        .test("Alpha Vantage global quote parses string-valued fields",
              [](TestContext& t) {
                  const auto q = alpha_vantage::parse_global_quote(kAvQuote).value();
                  t.expect(q.symbol == "IBM", "symbol");
                  t.expect(close(q.last, 186.75, 1e-9), "05. price string -> double");
                  t.expect(close(q.open, 185.0, 1e-9) && close(q.high, 187.5, 1e-9), "open/high");
                  t.expect(close(q.previous_close, 184.9, 1e-9), "previous close");
                  t.expect(close(q.volume, 3500000.0, 1e-3), "volume");
                  t.expect(q.timestamp == 1721260800, "latest trading day -> epoch");
              })
        .test("Alpha Vantage daily series sorts newest-first input ascending",
              [](TestContext& t) {
                  const auto s = alpha_vantage::parse_time_series_daily(kAvDaily).value();
                  t.expect(s.symbol == "IBM", "symbol from meta");
                  t.expect(s.size() == 2, "two bars");
                  t.expect(s.bars[0].timestamp < s.bars[1].timestamp, "sorted ascending");
                  t.expect(close(s.bars[0].close, 184.9, 1e-9), "earliest close (2024-07-17)");
                  t.expect(close(s.bars[1].close, 186.75, 1e-9), "latest close (2024-07-18)");
              })
        .test("Alpaca latest quote takes two-sided mid as last",
              [](TestContext& t) {
                  const auto q = alpaca::parse_latest_quote(kAlpacaQuote).value();
                  t.expect(q.symbol == "AAPL", "symbol");
                  t.expect(close(q.bid, 227.50, 1e-9) && close(q.ask, 227.54, 1e-9), "bid/ask");
                  t.expect(close(q.last, 227.52, 1e-9), "last == mid of bp/ap");
                  t.expect(close(q.spread(), 0.04, 1e-9), "spread == ask - bid");
                  t.expect(q.timestamp == 1721332800, "RFC-3339 t -> epoch");
              })
        .test("Alpaca bars parse compact o/h/l/c/v keys",
              [](TestContext& t) {
                  const auto s = alpaca::parse_bars(kAlpacaBars).value();
                  t.expect(s.symbol == "AAPL" && s.size() == 2, "symbol + two bars");
                  t.expect(close(s.bars[0].open, 183.0, 1e-9), "first open");
                  t.expect(close(s.bars[1].close, 186.75, 1e-9), "second close");
                  t.expect(s.bars[0].timestamp == 1721174400, "first bar epoch (2024-07-17)");
              })
        .test("honesty: malformed / missing / wrong-typed / unwired all ride the railway",
              [](TestContext& t) {
                  // Malformed JSON -> syntax_error.
                  const auto bad = yahoo::parse_quote("{not valid json");
                  t.expect(!bad && bad.error() == MathError::syntax_error, "bad JSON -> syntax_error");
                  // Well-formed but the required price field absent -> domain_error.
                  const auto miss = yahoo::parse_quote(R"({"chart":{"result":[{"meta":{"symbol":"X"}}]}})");
                  t.expect(!miss && miss.error() == MathError::domain_error, "missing price -> domain_error");
                  // Wrong type (price as a string where Yahoo emits a number) -> domain_error.
                  const auto wrong = yahoo::parse_quote(
                      R"({"chart":{"result":[{"meta":{"symbol":"X","regularMarketPrice":"oops"}}]}})");
                  t.expect(!wrong && wrong.error() == MathError::domain_error, "string price -> domain_error");
              })
        .test("fluent Feed dispatches by provider and stamps asset class",
              [](TestContext& t) {
                  const auto feed = Feed::from(DataProvider::yahoo).as_asset_class(AssetClass::equity);
                  const auto q = feed.quote(kYahooQuote).value();
                  t.expect(q.symbol == "AAPL" && q.asset_class == AssetClass::equity,
                           "Yahoo feed quote stamped equity");
                  // The SAME feed instance is reused for a chain (composability).
                  const auto ch = feed.option_chain(kYahooOptions).value();
                  t.expect(ch.calls.size() == 1, "reused feed decodes option chain");
                  // A non-Yahoo provider's option_chain is honestly not_implemented.
                  const auto av = Feed::from(DataProvider::alpha_vantage);
                  const auto none = av.option_chain(kYahooOptions);
                  t.expect(!none && none.error() == MathError::not_implemented,
                           "AV option_chain -> not_implemented");
                  // Provider-dispatched bars via the free function.
                  const auto bars = parse_bars(DataProvider::alpaca, kAlpacaBars).value();
                  t.expect(bars.size() == 2, "dispatched Alpaca bars");
              })
        .run();
}
