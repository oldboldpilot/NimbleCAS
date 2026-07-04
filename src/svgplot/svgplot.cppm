// NimbleCAS in-core SVG plotter — standalone SVG-string generation (no GUI).
// @author Olumuyiwa Oluwasanmi
//
// Emits a self-contained <svg> string for plotting data series and sampled
// functions. There is NO interactivity, animation, GPU, browser, or WASM here: this
// is deterministic static SVG-string generation only. It is the drawing substrate a
// future WebGPU/WASM interactive front-end or a notebook exporter would render or
// extend. All arithmetic is IEEE double — the pixel coordinates are for RENDERING,
// not exact symbolic math.
//
// Data-space -> pixel mapping (the crux; the tests pin it exactly). The data box
// [xmin,xmax] x [ymin,ymax] maps onto the inner plot rectangle
// [margin, width-margin] x [height-margin, margin]. SVG's y-axis grows DOWNWARD, so
// the data y-axis is FLIPPED (data ymin sits at the bottom, ymax at the top):
//
//   px = margin        + (x - xmin) / (xmax - xmin) * (width  - 2*margin)
//   py = (height-margin) - (y - ymin) / (ymax - ymin) * (height - 2*margin)
//
// Coordinates are formatted with a FIXED 3-decimal precision and a stable attribute
// order so the output string is reproducible and directly assertable. std::format is
// locale-independent, so numbers never acquire locale group separators (no commas).
//
// Conforms to config/cpp_details.txt: C++23 modules, `import std`, trailing return
// types, no owning raw pointers, std::expected error handling (no exceptions).

export module nimblecas.svgplot;

import std;
import nimblecas.core;

export namespace nimblecas {

// Rendering options for a single plot. Defaults give a 640x480 canvas with a 40px
// margin, a black stroke, axes drawn, and no title. `title` (if non-empty) is
// XML-escaped before emission.
struct PlotOptions {
    int width = 640;
    int height = 480;
    int margin = 40;
    std::string stroke = "black";
    std::string title = "";
    bool axes = true;
};

// Render a polyline through (xs[i], ys[i]). The data range is auto-scaled from the
// data; a degenerate axis (all xs equal, or all ys equal) is padded symmetrically so
// a flat series renders at mid-height instead of dividing by zero.
//
// Errors (MathError::domain_error): xs/ys length mismatch, empty data, non-positive
// width/height, or a margin that leaves no room for the inner plot rect
// (2*margin >= width or 2*margin >= height).
[[nodiscard]] auto plot_line(std::span<const double> xs, std::span<const double> ys,
                             const PlotOptions& opt) -> Result<std::string>;

// Render one <circle> per (xs[i], ys[i]) at its mapped pixel centre. Same auto-scale,
// padding, and error conditions as plot_line.
[[nodiscard]] auto plot_scatter(std::span<const double> xs, std::span<const double> ys,
                                const PlotOptions& opt) -> Result<std::string>;

// Sample f at `samples` evenly spaced x in [xmin,xmax] (inclusive of both ends),
// dropping any x whose f(x) is non-finite (NaN/±inf), then delegate to plot_line.
//
// Errors (MathError::domain_error): samples < 2, xmin >= xmax, plus any error
// plot_line raises on the surviving samples (e.g. every f(x) dropped -> empty data).
[[nodiscard]] auto plot_function(std::function<double(double)> f, double xmin, double xmax,
                                 int samples, const PlotOptions& opt) -> Result<std::string>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// A resolved, non-degenerate data range together with the canvas geometry needed to
// map a data point to a pixel. Constructed only after validation, so every span here
// is strictly positive and division is safe.
struct Mapper {
    double xmin, xmax, ymin, ymax;
    double width, height, margin;

    [[nodiscard]] auto px(double x) const -> double {
        return margin + (x - xmin) / (xmax - xmin) * (width - 2.0 * margin);
    }
    [[nodiscard]] auto py(double y) const -> double {
        return (height - margin) - (y - ymin) / (ymax - ymin) * (height - 2.0 * margin);
    }
};

// Format a pixel coordinate with fixed precision so the output is byte-reproducible.
[[nodiscard]] auto fmt(double v) -> std::string { return std::format("{:.3f}", v); }

// Escape the five significant XML characters in user-supplied text (titles). Ampersand
// is escaped first so the entity ampersands introduced afterwards are not re-escaped.
[[nodiscard]] auto escape_xml(std::string_view in) -> std::string {
    std::string out;
    out.reserve(in.size());
    for (char c : in) {
        switch (c) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            case '\'': out += "&apos;"; break;
            default: out += c; break;
        }
    }
    return out;
}

// Widen a degenerate [lo,hi] (lo == hi) to a symmetric interval about the common
// value, so the data point lands at the centre of its axis (a flat line renders at
// mid-height). The pad scales with magnitude so large flat values still separate.
auto pad_if_degenerate(double& lo, double& hi) -> void {
    if (lo == hi) {
        const double pad = (lo == 0.0) ? 1.0 : std::abs(lo) * 0.5;
        lo -= pad;
        hi += pad;
    }
}

// Validation shared by every entry point that consumes an (xs, ys) pair. On success
// returns a ready-to-use Mapper with the auto-scaled, padded data range.
[[nodiscard]] auto make_mapper(std::span<const double> xs, std::span<const double> ys,
                               const PlotOptions& opt) -> Result<Mapper> {
    if (xs.size() != ys.size()) {
        return make_error<Mapper>(MathError::domain_error);  // mismatched series lengths
    }
    if (xs.empty()) {
        return make_error<Mapper>(MathError::domain_error);  // nothing to plot
    }
    if (opt.width <= 0 || opt.height <= 0) {
        return make_error<Mapper>(MathError::domain_error);  // non-positive canvas
    }
    if (opt.margin < 0 || 2 * opt.margin >= opt.width || 2 * opt.margin >= opt.height) {
        return make_error<Mapper>(MathError::domain_error);  // negative margin or no plot area
    }
    // Reject non-finite input outright: a NaN/inf coordinate would defeat the range scan
    // (NaN comparisons are all false) and emit a literal "nan"/"inf" into the SVG. (plot_function
    // pre-drops non-finite samples, so its surviving data is already finite here.)
    for (const double v : xs) {
        if (!std::isfinite(v)) {
            return make_error<Mapper>(MathError::domain_error);  // non-finite x coordinate
        }
    }
    for (const double v : ys) {
        if (!std::isfinite(v)) {
            return make_error<Mapper>(MathError::domain_error);  // non-finite y coordinate
        }
    }

    auto [xlo, xhi] = std::ranges::minmax(xs);
    auto [ylo, yhi] = std::ranges::minmax(ys);
    pad_if_degenerate(xlo, xhi);
    pad_if_degenerate(ylo, yhi);

    return Mapper{xlo, xhi, ylo, yhi, static_cast<double>(opt.width),
                  static_cast<double>(opt.height), static_cast<double>(opt.margin)};
}

// Emit the opening <svg> tag plus, when enabled, the framing rect and the two axis
// lines (bottom x-axis, left y-axis), plus the optional <title>/<text>. Attribute
// order is fixed for reproducibility.
[[nodiscard]] auto svg_header(const PlotOptions& opt) -> std::string {
    std::string s = std::format(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" "
        "viewBox=\"0 0 {} {}\">",
        opt.width, opt.height, opt.width, opt.height);

    if (!opt.title.empty()) {
        const std::string t = escape_xml(opt.title);
        // <title> is the accessible name; <text> is the visible caption, centred near
        // the top of the canvas.
        s += std::format("<title>{}</title>", t);
        s += std::format(
            "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\">{}</text>",
            fmt(opt.width / 2.0), fmt(opt.margin / 2.0 + 5.0), t);
    }

    if (opt.axes) {
        const double left = opt.margin;
        const double right = opt.width - opt.margin;
        const double top = opt.margin;
        const double bottom = opt.height - opt.margin;
        // Framing rectangle of the inner plot area.
        s += std::format(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"none\" "
            "stroke=\"#888888\"/>",
            fmt(left), fmt(top), fmt(right - left), fmt(bottom - top));
        // Bottom (x) axis and left (y) axis lines.
        s += std::format(
            "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#888888\"/>",
            fmt(left), fmt(bottom), fmt(right), fmt(bottom));
        s += std::format(
            "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#888888\"/>",
            fmt(left), fmt(top), fmt(left), fmt(bottom));
    }
    return s;
}

}  // namespace

auto plot_line(std::span<const double> xs, std::span<const double> ys,
               const PlotOptions& opt) -> Result<std::string> {
    auto m = make_mapper(xs, ys, opt);
    if (!m) {
        return make_error<std::string>(m.error());
    }

    std::string points;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        if (i != 0) {
            points += ' ';
        }
        points += fmt(m->px(xs[i])) + "," + fmt(m->py(ys[i]));
    }

    std::string svg = svg_header(opt);
    svg += std::format("<polyline fill=\"none\" stroke=\"{}\" points=\"{}\"/>",
                       escape_xml(opt.stroke), points);
    svg += "</svg>";
    return svg;
}

auto plot_scatter(std::span<const double> xs, std::span<const double> ys,
                  const PlotOptions& opt) -> Result<std::string> {
    auto m = make_mapper(xs, ys, opt);
    if (!m) {
        return make_error<std::string>(m.error());
    }

    std::string svg = svg_header(opt);
    for (std::size_t i = 0; i < xs.size(); ++i) {
        svg += std::format("<circle cx=\"{}\" cy=\"{}\" r=\"3.000\" fill=\"{}\"/>",
                           fmt(m->px(xs[i])), fmt(m->py(ys[i])), escape_xml(opt.stroke));
    }
    svg += "</svg>";
    return svg;
}

auto plot_function(std::function<double(double)> f, double xmin, double xmax, int samples,
                   const PlotOptions& opt) -> Result<std::string> {
    if (samples < 2) {
        return make_error<std::string>(MathError::domain_error);  // need an interval
    }
    if (xmin >= xmax) {
        return make_error<std::string>(MathError::domain_error);  // empty/inverted domain
    }

    std::vector<double> xs;
    std::vector<double> ys;
    xs.reserve(static_cast<std::size_t>(samples));
    ys.reserve(static_cast<std::size_t>(samples));

    const double step = (xmax - xmin) / static_cast<double>(samples - 1);
    for (int i = 0; i < samples; ++i) {
        // Anchor the final sample exactly at xmax to avoid step-accumulation drift.
        const double x = (i == samples - 1) ? xmax : xmin + static_cast<double>(i) * step;
        const double y = f(x);
        if (std::isfinite(y)) {  // drop NaN / ±inf (e.g. 1/x at x == 0)
            xs.push_back(x);
            ys.push_back(y);
        }
    }

    return plot_line(xs, ys, opt);
}

}  // namespace nimblecas
