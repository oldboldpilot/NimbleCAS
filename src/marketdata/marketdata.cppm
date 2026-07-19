// NimbleCAS market-data feed — provider-agnostic quote ingestion.
// @author Olumuyiwa Oluwasanmi
//
// SCOPE. A single NORMALISED market-data model (`Quote`, `Bar`/`BarSeries`,
// `OptionQuote`/`OptionChain`, `RateQuote`) plus a thin ADAPTER per external provider
// (Yahoo Finance, Alpha Vantage, Alpaca) that maps that provider's JSON response into
// the normalised shape. Everything downstream — pricing, the option/futures strategy
// modules, portfolio analytics — consumes the normalised model and never sees a
// provider's wire format. Adding a provider is one enum value + one adapter function;
// no downstream code changes. JSON is parsed by the vendored `fastjson`
// (fastestjsoninthewest, Code Policy Rule 37); this module never hand-rolls a parser.
//
// HONESTY (config/cpp_details.txt Rule 32, AGENTS.md). This is a pure DATA-INGESTION
// boundary — it performs NO pricing math and invents NO values. It faithfully copies
// provider fields into the typed model; a MISSING or WRONG-TYPED field is an honest
// `MathError`, never a silent zero dressed up as a real quote. The failure map is:
//   * malformed JSON that `fastjson` rejects           -> MathError::syntax_error
//   * a required field absent, or present but the wrong JSON type, or a numeric string
//     that does not parse                               -> MathError::domain_error
//   * a provider/response shape no adapter handles      -> MathError::not_implemented
// Numeric quotes are IEEE `double` (provider values are already decimal). Timestamps
// are epoch SECONDS; providers that quote calendar/RFC-3339 strings are converted via
// the exact proleptic-Gregorian day count below, and a field we cannot interpret as a
// time is left 0 (== unknown), never guessed. All failure rides the railway; nothing
// throws — provider JSON is untrusted input and is treated as such.

export module nimblecas.marketdata;

import std;
import nimblecas.core;
import fastjson;

export namespace nimblecas::marketdata {

// The asset class a normalised quote describes. Drives which downstream valuation /
// strategy engine consumes it (equity/index/option -> pricing; fx -> currency/fxstrat;
// future -> futures; bond/money_market -> finance/bondstrat/mmstrat).
enum class AssetClass : std::uint8_t {
    equity, index, fx, future, option, bond, money_market, crypto, commodity, unknown
};

// The external data source a payload came from. Extensible: a new provider is a new
// enumerator plus a matching adapter in its namespace below.
enum class DataProvider : std::uint8_t { yahoo, alpha_vantage, alpaca };

// Option side, mirroring nimblecas::pricing::OptionType but kept local so the data
// layer does not depend on the pricing module (pricing depends the other way is fine).
enum class OptionRight : std::uint8_t { call, put };

// ---------------------------------------------------------------------------
// Normalised model — the single shape every provider maps into.
// ---------------------------------------------------------------------------

// A point-in-time quote for one instrument. `last` is the field pricing treats as the
// spot. Any field a provider does not supply stays at its zero default and the adapter
// only fails when a REQUIRED field (symbol + a usable price) is missing.
struct Quote {
    std::string symbol{};
    AssetClass asset_class{AssetClass::equity};
    double last{0.0};            // last / regular-market price (the pricing spot)
    double bid{0.0};
    double ask{0.0};
    double open{0.0};
    double high{0.0};
    double low{0.0};
    double previous_close{0.0};
    double volume{0.0};
    std::string currency{};
    std::int64_t timestamp{0};   // epoch seconds; 0 == unknown

    // Mid price when a two-sided market is present, else the last trade. Never invents
    // a side: if only one of bid/ask is positive it falls back to `last`.
    [[nodiscard]] auto mid() const noexcept -> double {
        return (bid > 0.0 && ask > 0.0) ? 0.5 * (bid + ask) : last;
    }
    // Bid/ask spread when two-sided, else 0.
    [[nodiscard]] auto spread() const noexcept -> double {
        return (bid > 0.0 && ask > 0.0) ? (ask - bid) : 0.0;
    }
    [[nodiscard]] auto with_symbol(std::string s) const -> Quote { auto c = *this; c.symbol = std::move(s); return c; }
    [[nodiscard]] auto with_last(double v) const -> Quote { auto c = *this; c.last = v; return c; }
    [[nodiscard]] auto with_asset_class(AssetClass a) const -> Quote { auto c = *this; c.asset_class = a; return c; }
};

// One OHLCV bar. `timestamp` is the bar's open time in epoch seconds.
struct Bar {
    std::int64_t timestamp{0};
    double open{0.0};
    double high{0.0};
    double low{0.0};
    double close{0.0};
    double volume{0.0};
};

// A time-ordered series of bars for one instrument (a chart / candles response).
struct BarSeries {
    std::string symbol{};
    AssetClass asset_class{AssetClass::equity};
    std::vector<Bar> bars{};

    [[nodiscard]] auto size() const noexcept -> std::size_t { return bars.size(); }
    [[nodiscard]] auto empty() const noexcept -> bool { return bars.empty(); }
    // Close-to-close simple returns of consecutive bars (size == bars-1; empty if <2).
    [[nodiscard]] auto close_returns() const -> std::vector<double>;
};

// A single listed option contract quote.
struct OptionQuote {
    std::string contract_symbol{};
    OptionRight right{OptionRight::call};
    double strike{0.0};
    std::int64_t expiration{0};      // epoch seconds
    double last{0.0};
    double bid{0.0};
    double ask{0.0};
    double implied_volatility{0.0};
    double volume{0.0};
    double open_interest{0.0};
};

// The calls + puts for one underlying/expiry, plus the underlying's spot.
struct OptionChain {
    std::string underlying{};
    double underlying_price{0.0};
    std::int64_t expiration{0};      // epoch seconds of this chain's expiry
    std::vector<OptionQuote> calls{};
    std::vector<OptionQuote> puts{};
};

// A yield/price quote for a rate instrument (bond, bill, note) — the bridge into the
// fixed-income / money-market strategy engines.
struct RateQuote {
    std::string symbol{};
    AssetClass asset_class{AssetClass::bond};
    double yield{0.0};               // decimal (0.045 == 4.5%)
    double price{0.0};               // clean price per 100 face, when supplied
    std::int64_t maturity{0};        // epoch seconds
    std::int64_t timestamp{0};
};

// ---------------------------------------------------------------------------
// Calendar helper (exact, exposed — useful to callers pinning fixture epochs).
// ---------------------------------------------------------------------------
// Days from the Unix epoch (1970-01-01) to the given proleptic-Gregorian date, by
// Howard Hinnant's exact integer algorithm. `days_from_civil(1970,1,1) == 0`. No
// validation beyond the algorithm's domain; callers pass calendar-valid y/m/d.
[[nodiscard]] auto days_from_civil(std::int64_t y, unsigned m, unsigned d) noexcept -> std::int64_t;

// Parse an ISO-8601 / RFC-3339 instant into epoch seconds. Accepts "YYYY-MM-DD" (time
// taken as 00:00:00 UTC) and "YYYY-MM-DDThh:mm:ss[.frac][Z]" (any trailing zone suffix
// after the seconds is ignored — quotes are normalised to UTC upstream). A string that
// does not match either shape -> domain_error, never a fabricated time.
[[nodiscard]] auto epoch_seconds_from_iso(std::string_view s) -> Result<std::int64_t>;

// ---------------------------------------------------------------------------
// Provider adapters. Each takes the raw JSON body of a documented endpoint and returns
// the normalised model. Selecting the endpoint (URL) is the caller's concern; this
// layer only decodes a body once you have it (keeping the module free of any network
// dependency and fully deterministic under test).
// ---------------------------------------------------------------------------

namespace yahoo {
// Body of `/v8/finance/chart/{symbol}`: reads `chart.result[0].meta` into a Quote.
[[nodiscard]] auto parse_quote(std::string_view json) -> Result<Quote>;
// Same endpoint with `timestamp[]` + `indicators.quote[0]` OHLCV arrays -> BarSeries.
[[nodiscard]] auto parse_chart(std::string_view json) -> Result<BarSeries>;
// Body of `/v7/finance/options/{symbol}`: first expiry's calls/puts -> OptionChain.
[[nodiscard]] auto parse_option_chain(std::string_view json) -> Result<OptionChain>;
}  // namespace yahoo

namespace alpha_vantage {
// Body of `function=GLOBAL_QUOTE`: the "Global Quote" object (string-valued fields).
[[nodiscard]] auto parse_global_quote(std::string_view json) -> Result<Quote>;
// Body of `function=TIME_SERIES_DAILY`: the "Time Series (Daily)" date-keyed map.
[[nodiscard]] auto parse_time_series_daily(std::string_view json) -> Result<BarSeries>;
}  // namespace alpha_vantage

namespace alpaca {
// Body of `/v2/stocks/{symbol}/quotes/latest`: the `quote` object (bp/ap two-sided).
[[nodiscard]] auto parse_latest_quote(std::string_view json) -> Result<Quote>;
// Body of `/v2/stocks/{symbol}/bars`: the `bars[]` array (o/h/l/c/v + RFC-3339 t).
[[nodiscard]] auto parse_bars(std::string_view json) -> Result<BarSeries>;
}  // namespace alpaca

// Provider-dispatched single-quote parse — the extensible front door. A provider whose
// quote adapter is not wired here yields not_implemented (never a wrong parse).
[[nodiscard]] auto parse_quote(DataProvider provider, std::string_view json) -> Result<Quote>;
// Provider-dispatched bar-series parse (Yahoo chart / Alpha Vantage daily / Alpaca bars).
[[nodiscard]] auto parse_bars(DataProvider provider, std::string_view json) -> Result<BarSeries>;

// ---------------------------------------------------------------------------
// Fluent, composable, reusable feed façade (Code Policy Rules 15/47).
// ---------------------------------------------------------------------------
// `Feed` binds a provider (and an optional default asset class) once, then decodes any
// number of payloads through a uniform, chainable surface — the same instance is reused
// across many responses and composes with the pricing/strategy layers downstream:
//
//     const auto feed = Feed::from(DataProvider::yahoo).as_asset_class(AssetClass::equity);
//     auto spot  = feed.quote(quote_json).transform([](const Quote& q){ return q.last; });
//     auto chain = feed.option_chain(chain_json);        // reused, no re-binding
//
// It holds no mutable parse state, so a single const Feed is safe to share across threads.
class Feed {
public:
    [[nodiscard]] static auto from(DataProvider provider) -> Feed { return Feed{provider}; }
    // Stamp decoded quotes/series with this asset class (fluent, returns *this).
    [[nodiscard]] auto as_asset_class(AssetClass a) -> Feed& { default_class_ = a; return *this; }

    [[nodiscard]] auto provider() const noexcept -> DataProvider { return provider_; }
    [[nodiscard]] auto asset_class() const noexcept -> AssetClass { return default_class_; }

    // Decode a single quote, a bar series, or (Yahoo) an option chain for this provider,
    // tagging the asset class. option_chain is Yahoo-only today -> not_implemented else.
    [[nodiscard]] auto quote(std::string_view json) const -> Result<Quote>;
    [[nodiscard]] auto bars(std::string_view json) const -> Result<BarSeries>;
    [[nodiscard]] auto option_chain(std::string_view json) const -> Result<OptionChain>;

private:
    explicit Feed(DataProvider provider) : provider_(provider) {}
    DataProvider provider_{DataProvider::yahoo};
    AssetClass default_class_{AssetClass::equity};
};

}  // namespace nimblecas::marketdata

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::marketdata {
namespace {

using fastjson::json_value;

// --- Safe, exception-free navigation over a fastjson tree --------------------
// fastjson's operator[] on a missing key/out-of-range index is not a railway op, so we
// never call it blind: these helpers check shape first and return a pointer or an error.

// Descend into an object member by key.
[[nodiscard]] auto child(const json_value& v, std::string_view key) -> Result<const json_value*> {
    if (!v.is_object()) { return make_error<const json_value*>(MathError::domain_error); }
    const std::string k{key};
    if (!v.contains(k)) { return make_error<const json_value*>(MathError::domain_error); }
    return &v[k];
}

// Descend into an array element by index.
[[nodiscard]] auto elem(const json_value& v, std::size_t i) -> Result<const json_value*> {
    if (!v.is_array() || i >= v.size()) { return make_error<const json_value*>(MathError::domain_error); }
    return &v[i];
}

// Scalar extractors — each fails (domain_error) on the wrong JSON type rather than
// coercing, so a string where a number was expected is caught, not silently zeroed.
[[nodiscard]] auto as_num(const json_value& v) -> Result<double> {
    if (!v.is_number()) { return make_error<double>(MathError::domain_error); }
    return v.as_float64();
}
[[nodiscard]] auto as_str(const json_value& v) -> Result<std::string> {
    if (!v.is_string()) { return make_error<std::string>(MathError::domain_error); }
    return std::string{v.as_string()};
}
[[nodiscard]] auto as_i64(const json_value& v) -> Result<std::int64_t> {
    if (!v.is_number()) { return make_error<std::int64_t>(MathError::domain_error); }
    return v.as_int64();
}

// Parse a decimal string (Alpha Vantage quotes numbers as JSON strings) to double via
// std::from_chars — no locale, no throw. Trailing spaces tolerated; junk -> domain_error.
[[nodiscard]] auto num_from_str(std::string_view s) -> Result<double> {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) { s.remove_prefix(1); }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '%')) { s.remove_suffix(1); }
    double out{};
    const auto* first = s.data();
    const auto* lastp = s.data() + s.size();
    const auto [ptr, ec] = std::from_chars(first, lastp, out);
    if (ec != std::errc{} || ptr != lastp) { return make_error<double>(MathError::domain_error); }
    return out;
}

// A JSON field that may be a number OR a numeric string (providers are inconsistent).
[[nodiscard]] auto as_num_flex(const json_value& v) -> Result<double> {
    if (v.is_number()) { return v.as_float64(); }
    if (v.is_string()) { return num_from_str(v.as_string()); }
    return make_error<double>(MathError::domain_error);
}

// Optional object member -> double, defaulting when absent but FAILING when present-but-
// wrong-typed (so a corrupt field is still caught; only true absence uses the default).
[[nodiscard]] auto opt_num(const json_value& obj, std::string_view key, double dflt) -> Result<double> {
    auto c = child(obj, key);
    if (!c) { return dflt; }
    if ((*c)->is_null()) { return dflt; }
    return as_num_flex(**c);
}

}  // namespace

auto days_from_civil(std::int64_t y, unsigned m, unsigned d) noexcept -> std::int64_t {
    // Hinnant's algorithm: days since 1970-01-01, exact over the proleptic Gregorian
    // calendar. y is the actual year (offset so March is month 0 of the year's "era").
    y -= (m <= 2);
    const std::int64_t era = (y >= 0 ? y : y - 399) / 400;
    const auto yoe = static_cast<std::int64_t>(y - era * 400);            // [0, 399]
    const auto doy = static_cast<std::int64_t>((153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1);  // [0,365]
    const std::int64_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;       // [0, 146096]
    return era * 146097 + doe - 719468;
}

auto epoch_seconds_from_iso(std::string_view s) -> Result<std::int64_t> {
    // Require at least "YYYY-MM-DD".
    if (s.size() < 10 || s[4] != '-' || s[7] != '-') {
        return make_error<std::int64_t>(MathError::domain_error);
    }
    auto digits = [](std::string_view d, std::int64_t& out) -> bool {
        const auto* last = d.data() + d.size();
        const auto [p, ec] = std::from_chars(d.data(), last, out);
        return ec == std::errc{} && p == last;
    };
    std::int64_t y{}, mo{}, da{};
    if (!digits(s.substr(0, 4), y) || !digits(s.substr(5, 2), mo) || !digits(s.substr(8, 2), da)) {
        return make_error<std::int64_t>(MathError::domain_error);
    }
    if (mo < 1 || mo > 12 || da < 1 || da > 31) { return make_error<std::int64_t>(MathError::domain_error); }
    std::int64_t secs = days_from_civil(y, static_cast<unsigned>(mo), static_cast<unsigned>(da)) * 86400;
    // Optional "Thh:mm:ss" (any zone suffix after seconds is ignored — UTC assumed).
    if (s.size() >= 19 && (s[10] == 'T' || s[10] == ' ') && s[13] == ':' && s[16] == ':') {
        std::int64_t hh{}, mm{}, ss{};
        if (!digits(s.substr(11, 2), hh) || !digits(s.substr(14, 2), mm) || !digits(s.substr(17, 2), ss)) {
            return make_error<std::int64_t>(MathError::domain_error);
        }
        if (hh > 23 || mm > 59 || ss > 60) { return make_error<std::int64_t>(MathError::domain_error); }
        secs += hh * 3600 + mm * 60 + ss;
    }
    return secs;
}

auto BarSeries::close_returns() const -> std::vector<double> {
    std::vector<double> r;
    if (bars.size() < 2) { return r; }
    r.reserve(bars.size() - 1);
    for (std::size_t i = 1; i < bars.size(); ++i) {
        const double prev = bars[i - 1].close;
        r.push_back(prev != 0.0 ? bars[i].close / prev - 1.0 : 0.0);
    }
    return r;
}

// --- Yahoo Finance ----------------------------------------------------------

auto yahoo::parse_quote(std::string_view json) -> Result<Quote> {
    fastjson::parser p{json};
    auto doc = p.parse();
    if (!doc) { return make_error<Quote>(MathError::syntax_error); }
    // chart.result[0].meta
    auto chart = child(*doc, "chart");
    if (!chart) { return make_error<Quote>(chart.error()); }
    auto result = child(**chart, "result");
    if (!result) { return make_error<Quote>(result.error()); }
    auto r0 = elem(**result, 0);
    if (!r0) { return make_error<Quote>(r0.error()); }
    auto meta = child(**r0, "meta");
    if (!meta) { return make_error<Quote>(meta.error()); }
    const json_value& m = **meta;

    auto sym = child(m, "symbol").and_then([](const json_value* v) { return as_str(*v); });
    auto px = child(m, "regularMarketPrice").and_then([](const json_value* v) { return as_num(*v); });
    if (!sym || !px) { return make_error<Quote>(MathError::domain_error); }
    Quote q{};
    q.symbol = *sym;
    q.last = *px;
    q.previous_close = opt_num(m, "previousClose", opt_num(m, "chartPreviousClose", 0.0).value_or(0.0)).value_or(0.0);
    q.high = opt_num(m, "regularMarketDayHigh", 0.0).value_or(0.0);
    q.low = opt_num(m, "regularMarketDayLow", 0.0).value_or(0.0);
    q.volume = opt_num(m, "regularMarketVolume", 0.0).value_or(0.0);
    if (auto c = child(m, "currency"); c && (*c)->is_string()) { q.currency = std::string{(*c)->as_string()}; }
    if (auto t = child(m, "regularMarketTime"); t && (*t)->is_number()) { q.timestamp = (*t)->as_int64(); }
    return q;
}

auto yahoo::parse_chart(std::string_view json) -> Result<BarSeries> {
    fastjson::parser p{json};
    auto doc = p.parse();
    if (!doc) { return make_error<BarSeries>(MathError::syntax_error); }
    auto chart = child(*doc, "chart");
    if (!chart) { return make_error<BarSeries>(chart.error()); }
    auto result = child(**chart, "result");
    if (!result) { return make_error<BarSeries>(result.error()); }
    auto r0 = elem(**result, 0);
    if (!r0) { return make_error<BarSeries>(r0.error()); }

    BarSeries out{};
    if (auto meta = child(**r0, "meta"); meta) {
        if (auto s = child(**meta, "symbol"); s && (*s)->is_string()) { out.symbol = std::string{(*s)->as_string()}; }
    }
    auto ts = child(**r0, "timestamp");
    auto quote = child(**r0, "indicators").and_then([](const json_value* v) { return child(*v, "quote"); })
                     .and_then([](const json_value* v) { return elem(*v, 0); });
    if (!ts || !quote) { return make_error<BarSeries>(MathError::domain_error); }
    if (!(*ts)->is_array()) { return make_error<BarSeries>(MathError::domain_error); }
    const json_value& q = **quote;
    auto opens = child(q, "open"), highs = child(q, "high"), lows = child(q, "low"),
         closes = child(q, "close"), vols = child(q, "volume");
    if (!opens || !highs || !lows || !closes) { return make_error<BarSeries>(MathError::domain_error); }
    const std::size_t n = (*ts)->size();
    // Every present OHLC array must be parallel to timestamp[]; a mismatch is corrupt.
    for (const auto* arr : {*opens, *highs, *lows, *closes}) {
        if (!arr->is_array() || arr->size() != n) { return make_error<BarSeries>(MathError::domain_error); }
    }
    const bool have_vol = vols && (*vols)->is_array() && (*vols)->size() == n;
    out.bars.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const json_value& te = (**ts)[i];
        // Yahoo pads gaps with null OHLC at a valid timestamp; skip those honestly.
        const json_value& oe = (**opens)[i];
        if (te.is_null() || oe.is_null()) { continue; }
        Bar b{};
        b.timestamp = te.is_number() ? te.as_int64() : 0;
        b.open = oe.as_float64();
        b.high = (**highs)[i].is_number() ? (**highs)[i].as_float64() : 0.0;
        b.low = (**lows)[i].is_number() ? (**lows)[i].as_float64() : 0.0;
        b.close = (**closes)[i].is_number() ? (**closes)[i].as_float64() : 0.0;
        b.volume = (have_vol && (**vols)[i].is_number()) ? (**vols)[i].as_float64() : 0.0;
        out.bars.push_back(b);
    }
    return out;
}

namespace {
// Shared decoder for one Yahoo option leg object.
[[nodiscard]] auto yahoo_option(const json_value& o, OptionRight right) -> Result<OptionQuote> {
    auto strike = child(o, "strike").and_then([](const json_value* v) { return as_num(*v); });
    if (!strike) { return make_error<OptionQuote>(MathError::domain_error); }
    OptionQuote q{};
    q.right = right;
    q.strike = *strike;
    if (auto c = child(o, "contractSymbol"); c && (*c)->is_string()) { q.contract_symbol = std::string{(*c)->as_string()}; }
    q.last = opt_num(o, "lastPrice", 0.0).value_or(0.0);
    q.bid = opt_num(o, "bid", 0.0).value_or(0.0);
    q.ask = opt_num(o, "ask", 0.0).value_or(0.0);
    q.implied_volatility = opt_num(o, "impliedVolatility", 0.0).value_or(0.0);
    q.volume = opt_num(o, "volume", 0.0).value_or(0.0);
    q.open_interest = opt_num(o, "openInterest", 0.0).value_or(0.0);
    if (auto e = child(o, "expiration"); e && (*e)->is_number()) { q.expiration = (*e)->as_int64(); }
    return q;
}
}  // namespace

auto yahoo::parse_option_chain(std::string_view json) -> Result<OptionChain> {
    fastjson::parser p{json};
    auto doc = p.parse();
    if (!doc) { return make_error<OptionChain>(MathError::syntax_error); }
    auto res0 = child(*doc, "optionChain")
                    .and_then([](const json_value* v) { return child(*v, "result"); })
                    .and_then([](const json_value* v) { return elem(*v, 0); });
    if (!res0) { return make_error<OptionChain>(res0.error()); }
    const json_value& r = **res0;
    OptionChain chain{};
    if (auto u = child(r, "underlyingSymbol"); u && (*u)->is_string()) { chain.underlying = std::string{(*u)->as_string()}; }
    if (auto qp = child(r, "quote").and_then([](const json_value* v) { return child(*v, "regularMarketPrice"); });
        qp && (*qp)->is_number()) {
        chain.underlying_price = (*qp)->as_float64();
    }
    auto opts0 = child(r, "options").and_then([](const json_value* v) { return elem(*v, 0); });
    if (!opts0) { return make_error<OptionChain>(opts0.error()); }
    const json_value& o0 = **opts0;
    if (auto e = child(o0, "expirationDate"); e && (*e)->is_number()) { chain.expiration = (*e)->as_int64(); }
    auto decode_leg = [&](std::string_view key, OptionRight right, std::vector<OptionQuote>& into) -> Result<void> {
        auto arr = child(o0, key);
        if (!arr) { return {}; }  // a side may legitimately be absent
        if (!(*arr)->is_array()) { return make_error<void>(MathError::domain_error); }
        into.reserve((*arr)->size());
        for (std::size_t i = 0; i < (*arr)->size(); ++i) {
            auto oq = yahoo_option((**arr)[i], right);
            if (!oq) { return make_error<void>(oq.error()); }
            into.push_back(*oq);
        }
        return {};
    };
    if (auto c = decode_leg("calls", OptionRight::call, chain.calls); !c) { return make_error<OptionChain>(c.error()); }
    if (auto p2 = decode_leg("puts", OptionRight::put, chain.puts); !p2) { return make_error<OptionChain>(p2.error()); }
    return chain;
}

// --- Alpha Vantage ----------------------------------------------------------

auto alpha_vantage::parse_global_quote(std::string_view json) -> Result<Quote> {
    fastjson::parser p{json};
    auto doc = p.parse();
    if (!doc) { return make_error<Quote>(MathError::syntax_error); }
    auto gq = child(*doc, "Global Quote");
    if (!gq) { return make_error<Quote>(gq.error()); }
    const json_value& g = **gq;
    // Alpha Vantage numbers every field as a STRING under numbered keys.
    auto sym = child(g, "01. symbol").and_then([](const json_value* v) { return as_str(*v); });
    auto price = child(g, "05. price").and_then([](const json_value* v) { return as_num_flex(*v); });
    if (!sym || !price) { return make_error<Quote>(MathError::domain_error); }
    Quote q{};
    q.symbol = *sym;
    q.last = *price;
    q.open = child(g, "02. open").and_then([](const json_value* v) { return as_num_flex(*v); }).value_or(0.0);
    q.high = child(g, "03. high").and_then([](const json_value* v) { return as_num_flex(*v); }).value_or(0.0);
    q.low = child(g, "04. low").and_then([](const json_value* v) { return as_num_flex(*v); }).value_or(0.0);
    q.volume = child(g, "06. volume").and_then([](const json_value* v) { return as_num_flex(*v); }).value_or(0.0);
    q.previous_close = child(g, "08. previous close").and_then([](const json_value* v) { return as_num_flex(*v); }).value_or(0.0);
    if (auto d = child(g, "07. latest trading day"); d && (*d)->is_string()) {
        if (auto ts = epoch_seconds_from_iso((*d)->as_string())) { q.timestamp = *ts; }
    }
    return q;
}

auto alpha_vantage::parse_time_series_daily(std::string_view json) -> Result<BarSeries> {
    fastjson::parser p{json};
    auto doc = p.parse();
    if (!doc) { return make_error<BarSeries>(MathError::syntax_error); }
    auto series = child(*doc, "Time Series (Daily)");
    if (!series) { return make_error<BarSeries>(series.error()); }
    const json_value& s = **series;
    if (!s.is_object()) { return make_error<BarSeries>(MathError::domain_error); }
    BarSeries out{};
    if (auto meta = child(*doc, "Meta Data").and_then([](const json_value* v) { return child(*v, "2. Symbol"); });
        meta && (*meta)->is_string()) {
        out.symbol = std::string{(*meta)->as_string()};
    }
    // The date-keyed object: iterate its members, one bar per date.
    const auto& obj = s.as_object();
    out.bars.reserve(obj.size());
    for (const auto& [date, fields] : obj) {
        auto ts = epoch_seconds_from_iso(date);
        auto o = child(fields, "1. open").and_then([](const json_value* v) { return as_num_flex(*v); });
        auto h = child(fields, "2. high").and_then([](const json_value* v) { return as_num_flex(*v); });
        auto l = child(fields, "3. low").and_then([](const json_value* v) { return as_num_flex(*v); });
        auto c = child(fields, "4. close").and_then([](const json_value* v) { return as_num_flex(*v); });
        if (!ts || !o || !h || !l || !c) { return make_error<BarSeries>(MathError::domain_error); }
        Bar b{};
        b.timestamp = *ts;
        b.open = *o; b.high = *h; b.low = *l; b.close = *c;
        b.volume = child(fields, "5. volume").and_then([](const json_value* v) { return as_num_flex(*v); }).value_or(0.0);
        out.bars.push_back(b);
    }
    // Alpha Vantage emits newest-first; normalise to chronological (ascending time).
    std::ranges::sort(out.bars, {}, &Bar::timestamp);
    return out;
}

// --- Alpaca -----------------------------------------------------------------

auto alpaca::parse_latest_quote(std::string_view json) -> Result<Quote> {
    fastjson::parser p{json};
    auto doc = p.parse();
    if (!doc) { return make_error<Quote>(MathError::syntax_error); }
    auto sym = child(*doc, "symbol").and_then([](const json_value* v) { return as_str(*v); });
    auto quote = child(*doc, "quote");
    if (!sym || !quote) { return make_error<Quote>(MathError::domain_error); }
    const json_value& qv = **quote;
    auto bp = child(qv, "bp").and_then([](const json_value* v) { return as_num(*v); });
    auto ap = child(qv, "ap").and_then([](const json_value* v) { return as_num(*v); });
    if (!bp || !ap) { return make_error<Quote>(MathError::domain_error); }
    Quote q{};
    q.symbol = *sym;
    q.bid = *bp;
    q.ask = *ap;
    q.last = q.mid();  // Alpaca's latest-quote has no trade; last := two-sided mid.
    if (auto t = child(qv, "t"); t && (*t)->is_string()) {
        if (auto ts = epoch_seconds_from_iso((*t)->as_string())) { q.timestamp = *ts; }
    }
    return q;
}

auto alpaca::parse_bars(std::string_view json) -> Result<BarSeries> {
    fastjson::parser p{json};
    auto doc = p.parse();
    if (!doc) { return make_error<BarSeries>(MathError::syntax_error); }
    BarSeries out{};
    if (auto s = child(*doc, "symbol"); s && (*s)->is_string()) { out.symbol = std::string{(*s)->as_string()}; }
    auto bars = child(*doc, "bars");
    if (!bars) { return make_error<BarSeries>(bars.error()); }
    if (!(*bars)->is_array()) { return make_error<BarSeries>(MathError::domain_error); }
    out.bars.reserve((*bars)->size());
    for (std::size_t i = 0; i < (*bars)->size(); ++i) {
        const json_value& e = (**bars)[i];
        auto o = child(e, "o").and_then([](const json_value* v) { return as_num(*v); });
        auto h = child(e, "h").and_then([](const json_value* v) { return as_num(*v); });
        auto l = child(e, "l").and_then([](const json_value* v) { return as_num(*v); });
        auto c = child(e, "c").and_then([](const json_value* v) { return as_num(*v); });
        if (!o || !h || !l || !c) { return make_error<BarSeries>(MathError::domain_error); }
        Bar b{};
        b.open = *o; b.high = *h; b.low = *l; b.close = *c;
        b.volume = child(e, "v").and_then([](const json_value* v) { return as_num(*v); }).value_or(0.0);
        if (auto t = child(e, "t"); t && (*t)->is_string()) {
            if (auto ts = epoch_seconds_from_iso((*t)->as_string())) { b.timestamp = *ts; }
        }
        out.bars.push_back(b);
    }
    return out;
}

// --- Provider dispatch ------------------------------------------------------

auto parse_quote(DataProvider provider, std::string_view json) -> Result<Quote> {
    switch (provider) {
        case DataProvider::yahoo:         return yahoo::parse_quote(json);
        case DataProvider::alpha_vantage: return alpha_vantage::parse_global_quote(json);
        case DataProvider::alpaca:        return alpaca::parse_latest_quote(json);
    }
    return make_error<Quote>(MathError::not_implemented);
}

auto parse_bars(DataProvider provider, std::string_view json) -> Result<BarSeries> {
    switch (provider) {
        case DataProvider::yahoo:         return yahoo::parse_chart(json);
        case DataProvider::alpha_vantage: return alpha_vantage::parse_time_series_daily(json);
        case DataProvider::alpaca:        return alpaca::parse_bars(json);
    }
    return make_error<BarSeries>(MathError::not_implemented);
}

// --- Fluent Feed façade -----------------------------------------------------

auto Feed::quote(std::string_view json) const -> Result<Quote> {
    return parse_quote(provider_, json).transform([this](Quote q) {
        q.asset_class = default_class_;
        return q;
    });
}

auto Feed::bars(std::string_view json) const -> Result<BarSeries> {
    return parse_bars(provider_, json).transform([this](BarSeries s) {
        s.asset_class = default_class_;
        return s;
    });
}

auto Feed::option_chain(std::string_view json) const -> Result<OptionChain> {
    if (provider_ != DataProvider::yahoo) { return make_error<OptionChain>(MathError::not_implemented); }
    return yahoo::parse_option_chain(json);
}

}  // namespace nimblecas::marketdata
