# `nimblecas.svgplot` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/svgplot/svgplot.cppm`

In-core, standalone SVG-string plotter (ROADMAP §7.11). Given a data series or a
sampled function, it emits a **self-contained `<svg>` string** — line plots,
scatter plots, and function plots — and nothing else. There is **no
interactivity, animation, GPU path, browser, or WASM** here: this is
deterministic static SVG-string generation. It is the drawing substrate a future
WebGPU/WASM interactive front-end or a notebook exporter (the rest of §7.11)
would render or extend. The exact-vs-numerical boundary is sharp and deliberate:
**all arithmetic is IEEE `double`**, because pixel coordinates are for
*rendering*, not exact symbolic math. What *is* pinned exactly is the
data-space → pixel **mapping** and the output *text*: coordinates are formatted
with a fixed 3-decimal precision (`{:.3f}`) in a stable attribute order via
locale-independent `std::format`, so the emitted string is byte-reproducible and
directly assertable — no locale group separators ever appear. This module does
not build on the `Expr` tower; it consumes plain `double` spans and callbacks.

```cpp
import nimblecas.svgplot;
```

Depends on [`core`](core.md) (for `Result<T>` and `MathError`).

## The data-space → pixel mapping

The data box `[xmin,xmax] × [ymin,ymax]` maps onto the inner plot rectangle
`[margin, width−margin] × [height−margin, margin]`. SVG's y-axis grows
**downward**, so the data y-axis is **flipped** — data `ymin` sits at the bottom,
`ymax` at the top:

```
px = margin        + (x − xmin) / (xmax − xmin) * (width  − 2·margin)
py = (height−margin) − (y − ymin) / (ymax − ymin) * (height − 2·margin)
```

The data range is **auto-scaled** from the input (`minmax` over the series). A
**degenerate axis** — all `xs` equal, or all `ys` equal — is padded symmetrically
about the common value (pad = `1.0` when the value is `0`, else `|value|·0.5`), so
a flat series renders at the axis centre (e.g. mid-height) instead of dividing by
zero. This padding is internal; there is no public knob for it.

## Rendering options

### `struct PlotOptions`

Plain aggregate of rendering knobs, passed by `const&` to every entry point.
Defaults give a 640×480 canvas with a 40 px margin, a black stroke, axes drawn,
and no title.

| Field | Type | Default | Meaning |
| :--- | :--- | :--- | :--- |
| `width` | `int` | `640` | Canvas width in pixels. Must be `> 0` and satisfy `2·margin < width`. |
| `height` | `int` | `480` | Canvas height in pixels. Must be `> 0` and satisfy `2·margin < height`. |
| `margin` | `int` | `40` | Border between canvas edge and inner plot rect. Must be `≥ 0` and leave a non-empty inner rectangle. |
| `stroke` | `std::string` | `"black"` | Stroke colour for the polyline / scatter dots. XML-escaped on emission. |
| `title` | `std::string` | `""` | If non-empty, emitted as an accessible `<title>` plus a centred `<text>` caption; XML-escaped first. |
| `axes` | `bool` | `true` | When `true`, emit a `#888888` framing `<rect>` plus a bottom (x) and a left (y) axis `<line>`. |

## Plotting functions

All three are free functions in namespace `nimblecas`, `[[nodiscard]]`, and
return `Result<std::string>` (the complete `<svg>…</svg>` document on success).
`plot_scatter` and `plot_line` share identical validation (via an internal
`Mapper`); `plot_function` samples first, then delegates to `plot_line`.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `plot_line` | `auto plot_line(std::span<const double> xs, std::span<const double> ys, const PlotOptions& opt) -> Result<std::string>` | Render a single `<polyline>` through the mapped `(xs[i], ys[i])`. Range auto-scaled and degenerate-padded. |
| `plot_scatter` | `auto plot_scatter(std::span<const double> xs, std::span<const double> ys, const PlotOptions& opt) -> Result<std::string>` | Render one `<circle r="3.000">` per point at its mapped pixel centre. Same auto-scale, padding, and error conditions as `plot_line`. |
| `plot_function` | `auto plot_function(std::function<double(double)> f, double xmin, double xmax, int samples, const PlotOptions& opt) -> Result<std::string>` | Sample `f` at `samples` evenly spaced `x` in `[xmin, xmax]` (both ends inclusive; the final sample is anchored exactly at `xmax` to avoid step drift), **drop** any `x` whose `f(x)` is non-finite (NaN/±inf, e.g. `1/x` at `x = 0`), then delegate to `plot_line` on the survivors. |

### Emitted document shape

Every successful call yields a document that starts with
`<svg xmlns="http://www.w3.org/2000/svg" width="…" height="…" viewBox="0 0 W H">`
and ends with `</svg>`. Between them, in fixed order: the optional
`<title>`/`<text>`; then (when `axes` is `true`) the framing `<rect>` and two axis
`<line>`s; then the data mark — a single `<polyline fill="none" stroke="…">` for
`plot_line`/`plot_function`, or a run of `<circle>` elements for `plot_scatter`.
`stroke` and `title` text are XML-escaped (`&`, `<`, `>`, `"`, `'`), with `&`
escaped first so introduced entities are not re-escaped.

## Error model

Every failure path returns `MathError::domain_error`; this module raises no other
error. Non-finite input is **rejected**, not emitted — a NaN/inf would defeat the
range scan (NaN comparisons are all false) and leak a literal `"nan"`/`"inf"` into
the SVG.

| Condition | Error |
| :--- | :--- |
| `xs.size() != ys.size()` (length mismatch) | `MathError::domain_error` |
| `xs.empty()` (nothing to plot) | `MathError::domain_error` |
| `opt.width <= 0` or `opt.height <= 0` | `MathError::domain_error` |
| `opt.margin < 0` (negative margin) | `MathError::domain_error` |
| `2·margin >= width` or `2·margin >= height` (no inner plot area) | `MathError::domain_error` |
| Any non-finite value in `xs` or `ys` (direct `plot_line`/`plot_scatter` input) | `MathError::domain_error` |
| `plot_function`: `samples < 2` | `MathError::domain_error` |
| `plot_function`: `xmin >= xmax` (empty/inverted domain) | `MathError::domain_error` |
| `plot_function`: every `f(x)` non-finite → empty surviving data → `plot_line` | `MathError::domain_error` |

There is no floating-point overflow contract here: this is `double` rendering
arithmetic, not the overflow-checked `int64`/`Rational` exact core.

## Worked examples

```cpp
import nimblecas.svgplot;
import nimblecas.core;
using namespace nimblecas;

// A tiny 100x100 canvas with a 10px margin: x in [0,2], y in [0,1] land on
// integer pixel gridlines, so the mapping is exact and assertable.
PlotOptions o;
o.width = 100;
o.height = 100;
o.margin = 10;

// Line plot: xs={0,1,2}, ys={0,1,0}. x=0->px=10, x=1->px=50, x=2->px=90;
// y=0->py=90 (bottom, flipped), y=1->py=10 (top).
std::vector<double> xs{0.0, 1.0, 2.0};
std::vector<double> ys{0.0, 1.0, 0.0};
const std::string svg = plot_line(xs, ys, o).value();
// svg contains: points="10.000,90.000 50.000,10.000 90.000,90.000"
// svg starts_with "<svg", ends_with "</svg>", carries viewBox="0 0 100 100".

// Flat series: all ys equal -> degenerate guard pads symmetrically, so every
// point lands at mid-height (py=50), never dividing by zero.
std::vector<double> flat{5.0, 5.0, 5.0};
plot_line(xs, flat, o).value();  // points="10.000,50.000 50.000,50.000 90.000,50.000"

// Scatter: one <circle> per point, same mapping as the line.
plot_scatter(xs, ys, o).value();  // 3x <circle>, first cx="10.000" cy="90.000"

// Function plot: f(x)=x on [0,4] with 5 samples -> 5 vertices.
plot_function([](double x) { return x; }, 0.0, 4.0, 5, o).value();

// Non-finite samples are dropped: 1/x at x=0 yields inf and is skipped,
// leaving 4 finite vertices; no "inf"/"nan" leaks into the SVG.
plot_function([](double x) { return 1.0 / x; }, -2.0, 2.0, 5, o).value();

// Titles are XML-escaped and never leak raw markup.
PlotOptions titled = o;
titled.title = "a & b < c > \"d\"";
plot_line(xs, ys, titled).value();  // contains &amp; &lt; &gt; &quot; and <title>

// Error paths (all domain_error).
std::vector<double> shortY{0.0, 1.0};
plot_line(xs, shortY, o).error();               // length mismatch
plot_line({}, {}, o).error();                    // empty data
PlotOptions bad = o; bad.margin = 60;
plot_line(xs, ys, bad).error();                  // 2*margin >= width -> no plot area
PlotOptions neg = o; neg.margin = -5;
plot_line(xs, ys, neg).error();                  // negative margin
plot_function([](double x) { return x; }, 0.0, 1.0, 1, o).error();  // samples < 2
plot_function([](double x) { return x; }, 1.0, 1.0, 5, o).error();  // xmin >= xmax
```

## See also

- [`nimblecas.core`](core.md) — the `Result<T>` / `MathError` error model this
  module reports through.
- [`nimblecas.latex`](latex.md) — the sibling static exporter (ROADMAP §7.12):
  AST → LaTeX, as this is data → SVG.
- [`nimblecas.gpu`](gpu.md) — the hardware-accelerated rendering path the future
  interactive front-end of §7.11 would layer over this static substrate.
- [Documentation hub](../Index.md)
