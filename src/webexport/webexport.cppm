// NimbleCAS JSON export bridge — serialize CAS plot data and documents to a fixed
// JSON contract a browser front-end will render.
// @author Olumuyiwa Oluwasanmi
//
// This is the DATA BRIDGE for the WebGPU/WASM front-end. There is NO rendering,
// interactivity, GPU, or WASM here: this module hand-rolls a deterministic JSON
// string (no external JSON library) that a front-end later renders. It is the JSON
// analogue of nimblecas.svgplot.
//
// Honesty (Rule 32 & project policy): this is a lossless data bridge. Plot points are
// the exact doubles the CAS produced — each number is emitted as its SHORTEST
// round-trippable decimal (std::format's default float form == std::to_chars), so it
// parses back to the identical IEEE double (no precision is discarded, and values are
// NOT pixel-mapped — unlike svgplot, which maps to pixels for drawing). Math blocks
// carry the LaTeX the nimblecas.latex module emits. No rendering claims are made here;
// rendering is the front-end's job.
//
// Emitted JSON contract:
//   PlotSpec:
//     {"type":"plot","title":<str>,"xLabel":<str>,"yLabel":<str>,
//      "xRange":[<num>,<num>]|null,"yRange":[<num>,<num>]|null,
//      "series":[{"kind":"line"|"scatter","label":<str>,"color":"#rrggbb",
//                 "points":[[x,y],...]}]}
//   Document:
//     {"type":"document","title":<str>,"blocks":[
//        {"kind":"prose","text":<str>} | {"kind":"math","latex":<str>}
//        | {"kind":"plot","plot":<PlotSpec>} ]}
//
// Numbers are emitted as their shortest round-trippable decimal via std::format, so
// output is byte-reproducible, locale-independent (no group separators), and lossless.
// Strings are JSON-escaped ("\"", "\\", control chars, and '<'/'>' as </> so
// the payload is safe to inline inside an HTML <script> block). Non-finite x/y (NaN/
// ±inf) are rejected with MathError::domain_error exactly as svgplot does.
//
// Conforms to config/cpp_details.txt: C++23 modules, `import std`, trailing return
// types, no owning raw pointers, std::expected error handling (no exceptions),
// [[nodiscard]].

export module nimblecas.webexport;

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.latex;

export namespace nimblecas {

// The two series shapes the front-end can render.
enum class SeriesKind : std::uint8_t { line, scatter };

// Options common to a whole PlotSpec (analogous to svgplot's PlotOptions). `color`
// and `label` seed the single series produced by plot_spec_line/scatter/function; the
// multi-series builder takes per-series colour/label from each Series instead. The
// optional ranges emit `[lo,hi]` when set and `null` when absent; when present they
// must be finite (domain_error otherwise).
struct PlotSpecOptions {
    std::string title = "";
    std::string xLabel = "";
    std::string yLabel = "";
    std::string color = "#1f77b4";  // default series colour (a valid #rrggbb)
    std::string label = "";
    std::optional<std::pair<double, double>> xRange = std::nullopt;
    std::optional<std::pair<double, double>> yRange = std::nullopt;
};

// One data series: its kind, front-end label/colour, and the raw (xs, ys) the CAS
// produced. xs.size() must equal ys.size() and all values must be finite.
struct Series {
    SeriesKind kind = SeriesKind::line;
    std::string label = "";
    std::string color = "#1f77b4";
    std::vector<double> xs{};
    std::vector<double> ys{};
};

// Single-series line PlotSpec JSON. Errors (MathError::domain_error): xs/ys length
// mismatch, empty data, any non-finite x/y, or a non-finite option range.
[[nodiscard]] auto plot_spec_line(std::span<const double> xs, std::span<const double> ys,
                                  const PlotSpecOptions& opt) -> Result<std::string>;

// Single-series scatter PlotSpec JSON. Same validation as plot_spec_line.
[[nodiscard]] auto plot_spec_scatter(std::span<const double> xs, std::span<const double> ys,
                                     const PlotSpecOptions& opt) -> Result<std::string>;

// Sample f at `samples` evenly spaced x in [xmin,xmax] (both ends inclusive), dropping
// any x whose f(x) is non-finite (mirrors svgplot::plot_function), then delegate to
// plot_spec_line. Errors (domain_error): samples < 2, xmin >= xmax, plus any error
// plot_spec_line raises on the surviving samples (e.g. every f(x) dropped -> empty).
[[nodiscard]] auto plot_spec_function(std::function<double(double)> f, double xmin,
                                      double xmax, std::size_t samples,
                                      const PlotSpecOptions& opt) -> Result<std::string>;

// Combine several series into one multi-series PlotSpec JSON (svgplot has no
// multi-series entry point). Each series is validated as in plot_spec_line; the shared
// title/labels/ranges come from `opt` (its color/label are ignored — series carry
// their own). Errors (domain_error): no series, or any series failing validation.
[[nodiscard]] auto plot_spec_multi(std::span<const Series> series,
                                   const PlotSpecOptions& opt) -> Result<std::string>;

// ---------------------------------------------------------------------------
// Document blocks (a type-safe union of the three block kinds).
// ---------------------------------------------------------------------------
struct ProseBlock {
    std::string text;
};
struct MathBlock {
    std::string latex;  // a LaTeX math string (as nimblecas.latex emits, no $ delims)
};
struct PlotBlock {
    std::string plot;  // an already-serialized PlotSpec JSON (from a plot_spec_* call)
};
using Block = std::variant<ProseBlock, MathBlock, PlotBlock>;

// Block constructors. math_block has two spellings: a raw LaTeX string, or an Expr
// wired through nimblecas.latex::to_latex.
[[nodiscard]] auto prose_block(std::string_view text) -> Block;
[[nodiscard]] auto math_block(std::string_view latex) -> Block;
[[nodiscard]] auto math_block(const Expr& e) -> Block;
[[nodiscard]] auto plot_block(std::string plot_json) -> Block;

// Assemble a Document JSON from a title and an ordered list of blocks. Errors
// (domain_error): a PlotBlock whose payload is not a JSON object (empty or not
// starting with '{').
[[nodiscard]] auto document_to_json(std::string_view title,
                                    const std::vector<Block>& blocks) -> Result<std::string>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

template <typename...>
inline constexpr bool always_false = false;

// Format a number as its SHORTEST round-trippable decimal (std::format's default
// float format == std::to_chars shortest form): the emitted token parses back to
// the exact same IEEE double, so the bridge is lossless, while staying
// byte-reproducible and locale-independent. Callers finite-check every value
// (validate_xy / range guards) before this runs, so no inf/nan token can appear.
[[nodiscard]] auto jnum(double v) -> std::string { return std::format("{}", v); }

// Escape a string for inclusion inside JSON double quotes: the two structural
// characters ('"' and '\'), the named control escapes, any remaining control
// character (< 0x20) as a \u00xx sequence, and '<' / '>' as < / > so the
// payload is also safe to inline verbatim inside an HTML <script> block (a '</script>'
// substring in a title/label/LaTeX can never terminate the script element).
[[nodiscard]] auto escape_json(std::string_view in) -> std::string {
    std::string out;
    out.reserve(in.size() + 2);
    for (const unsigned char c : in) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '<': out += "\\u003c"; break;
            case '>': out += "\\u003e"; break;
            default:
                if (c < 0x20) {
                    out += std::format("\\u{:04x}", static_cast<unsigned>(c));
                } else {
                    out += static_cast<char>(c);
                }
                break;
        }
    }
    return out;
}

// Emit a JSON string key/value pair fragment: "key":"escaped-value".
[[nodiscard]] auto jstr_field(std::string_view key, std::string_view value) -> std::string {
    return std::format("\"{}\":\"{}\"", key, escape_json(value));
}

// Validate an (xs, ys) pair: equal non-empty lengths and every coordinate finite.
// Reject non-finite outright (a NaN/inf would otherwise serialize as a literal
// "nan"/"inf" token, which is not valid JSON) — exactly as svgplot does.
[[nodiscard]] auto validate_xy(std::span<const double> xs, std::span<const double> ys)
    -> Result<std::monostate> {
    if (xs.size() != ys.size()) {
        return make_error<std::monostate>(MathError::domain_error);  // mismatched lengths
    }
    if (xs.empty()) {
        return make_error<std::monostate>(MathError::domain_error);  // nothing to plot
    }
    for (const double v : xs) {
        if (!std::isfinite(v)) {
            return make_error<std::monostate>(MathError::domain_error);  // non-finite x
        }
    }
    for (const double v : ys) {
        if (!std::isfinite(v)) {
            return make_error<std::monostate>(MathError::domain_error);  // non-finite y
        }
    }
    return std::monostate{};
}

// Render an optional [lo,hi] range: finite pair -> "[lo,hi]"; absent -> "null".
[[nodiscard]] auto render_range(const std::optional<std::pair<double, double>>& r)
    -> std::string {
    if (!r) {
        return "null";
    }
    return "[" + jnum(r->first) + "," + jnum(r->second) + "]";
}

// Serialize one series object: {"kind":...,"label":...,"color":...,"points":[[x,y],...]}.
[[nodiscard]] auto render_series(const Series& s) -> std::string {
    std::string out = "{";
    out += jstr_field("kind", s.kind == SeriesKind::scatter ? "scatter" : "line");
    out += "," + jstr_field("label", s.label);
    out += "," + jstr_field("color", s.color);
    out += ",\"points\":[";
    for (std::size_t i = 0; i < s.xs.size(); ++i) {
        if (i != 0) {
            out += ',';
        }
        out += "[" + jnum(s.xs[i]) + "," + jnum(s.ys[i]) + "]";
    }
    out += "]}";
    return out;
}

// Validate + serialize a whole PlotSpec from its series and shared options.
[[nodiscard]] auto render_plot_spec(std::span<const Series> series,
                                    const PlotSpecOptions& opt) -> Result<std::string> {
    if (series.empty()) {
        return make_error<std::string>(MathError::domain_error);  // a plot needs a series
    }
    for (const Series& s : series) {
        auto v = validate_xy(s.xs, s.ys);
        if (!v) {
            return make_error<std::string>(v.error());
        }
    }
    // A supplied range must be finite, else it would emit a "nan"/"inf" token.
    for (const auto* r : {&opt.xRange, &opt.yRange}) {
        if (*r && (!std::isfinite((*r)->first) || !std::isfinite((*r)->second))) {
            return make_error<std::string>(MathError::domain_error);  // non-finite range
        }
    }

    std::string out = "{";
    out += jstr_field("type", "plot");
    out += "," + jstr_field("title", opt.title);
    out += "," + jstr_field("xLabel", opt.xLabel);
    out += "," + jstr_field("yLabel", opt.yLabel);
    out += ",\"xRange\":" + render_range(opt.xRange);
    out += ",\"yRange\":" + render_range(opt.yRange);
    out += ",\"series\":[";
    for (std::size_t i = 0; i < series.size(); ++i) {
        if (i != 0) {
            out += ',';
        }
        out += render_series(series[i]);
    }
    out += "]}";
    return out;
}

// Build a single-series PlotSpec of the given kind from a raw (xs, ys) pair.
[[nodiscard]] auto single_series_spec(std::span<const double> xs, std::span<const double> ys,
                                      SeriesKind kind, const PlotSpecOptions& opt)
    -> Result<std::string> {
    Series s;
    s.kind = kind;
    s.label = opt.label;
    s.color = opt.color;
    s.xs.assign(xs.begin(), xs.end());
    s.ys.assign(ys.begin(), ys.end());
    const std::array<Series, 1> one{std::move(s)};
    return render_plot_spec(one, opt);
}

// Serialize one document block, or an error for an ill-formed PlotBlock payload.
[[nodiscard]] auto render_block(const Block& b) -> Result<std::string> {
    return std::visit(
        []<typename T>(const T& blk) -> Result<std::string> {
            if constexpr (std::is_same_v<T, ProseBlock>) {
                return "{" + jstr_field("kind", "prose") + "," +
                       jstr_field("text", blk.text) + "}";
            } else if constexpr (std::is_same_v<T, MathBlock>) {
                return "{" + jstr_field("kind", "math") + "," +
                       jstr_field("latex", blk.latex) + "}";
            } else if constexpr (std::is_same_v<T, PlotBlock>) {
                // The payload is embedded verbatim; guard that it is a JSON object so a
                // malformed document never escapes (a plot_spec_* result always is).
                if (blk.plot.empty() || blk.plot.front() != '{') {
                    return make_error<std::string>(MathError::domain_error);
                }
                return "{" + jstr_field("kind", "plot") + ",\"plot\":" + blk.plot + "}";
            } else {
                static_assert(always_false<T>, "render_block: unhandled Block kind");
            }
        },
        b);
}

}  // namespace

auto plot_spec_line(std::span<const double> xs, std::span<const double> ys,
                    const PlotSpecOptions& opt) -> Result<std::string> {
    return single_series_spec(xs, ys, SeriesKind::line, opt);
}

auto plot_spec_scatter(std::span<const double> xs, std::span<const double> ys,
                       const PlotSpecOptions& opt) -> Result<std::string> {
    return single_series_spec(xs, ys, SeriesKind::scatter, opt);
}

auto plot_spec_function(std::function<double(double)> f, double xmin, double xmax,
                        std::size_t samples, const PlotSpecOptions& opt)
    -> Result<std::string> {
    if (samples < 2) {
        return make_error<std::string>(MathError::domain_error);  // need an interval
    }
    if (xmin >= xmax) {
        return make_error<std::string>(MathError::domain_error);  // empty/inverted domain
    }

    std::vector<double> xs;
    std::vector<double> ys;
    xs.reserve(samples);
    ys.reserve(samples);

    const double step = (xmax - xmin) / static_cast<double>(samples - 1);
    for (std::size_t i = 0; i < samples; ++i) {
        // Anchor the final sample exactly at xmax to avoid step-accumulation drift.
        const double x = (i == samples - 1) ? xmax : xmin + static_cast<double>(i) * step;
        const double y = f(x);
        if (std::isfinite(y)) {  // drop NaN / ±inf (e.g. 1/x at x == 0)
            xs.push_back(x);
            ys.push_back(y);
        }
    }

    return plot_spec_line(xs, ys, opt);
}

auto plot_spec_multi(std::span<const Series> series, const PlotSpecOptions& opt)
    -> Result<std::string> {
    return render_plot_spec(series, opt);
}

auto prose_block(std::string_view text) -> Block { return ProseBlock{std::string(text)}; }

auto math_block(std::string_view latex) -> Block { return MathBlock{std::string(latex)}; }

auto math_block(const Expr& e) -> Block { return MathBlock{to_latex(e)}; }

auto plot_block(std::string plot_json) -> Block { return PlotBlock{std::move(plot_json)}; }

auto document_to_json(std::string_view title, const std::vector<Block>& blocks)
    -> Result<std::string> {
    std::string out = "{";
    out += jstr_field("type", "document");
    out += "," + jstr_field("title", title);
    out += ",\"blocks\":[";
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        if (i != 0) {
            out += ',';
        }
        auto r = render_block(blocks[i]);
        if (!r) {
            return make_error<std::string>(r.error());
        }
        out += *r;
    }
    out += "]}";
    return out;
}

}  // namespace nimblecas
