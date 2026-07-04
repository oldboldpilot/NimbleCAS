# `nimblecas.webexport` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/webexport/webexport.cppm`

The **JSON analogue of [`svgplot`](svgplot.md)**: it serializes CAS plot data and
documents into a **fixed JSON contract** that the browser front-end (`web/`)
renders. Where `svgplot` emits a self-contained `<svg>` string, this module
hand-rolls a deterministic JSON string (no external JSON library) that a WebGPU/WASM
front-end later renders. The boundary is honest and deliberate: this is a **pure
data bridge** — there is **no rendering, interactivity, GPU, or WASM here**.

Two properties make the bridge trustworthy:

- **Lossless.** Plot points are the exact doubles the CAS produced. Each number is
  emitted as its **shortest round-trippable decimal** (`std::format`'s default float
  form, which equals `std::to_chars`'s shortest form), so it parses back to the
  identical IEEE `double` — no precision is discarded. Unlike `svgplot`, values are
  **not** pixel-mapped: they are the raw data, not drawing coordinates.
- **HTML-inline safe.** Strings are JSON-escaped (`"`, `\`, the named control
  escapes, remaining control chars as `\u00xx`) **and** `<`/`>` are escaped as
  `<`/`>`, so the payload is safe to inline verbatim inside an HTML
  `<script>` block — a `</script>` substring in a title, label, or LaTeX string can
  never terminate the script element.

Non-finite `x`/`y` (NaN/±inf) are **rejected** with `MathError::domain_error`
(exactly as `svgplot` does), because a NaN/inf would otherwise serialize as a literal
`"nan"`/`"inf"` token, which is not valid JSON. Numbers are therefore
byte-reproducible and locale-independent (no group separators). Math blocks carry the
LaTeX the [`nimblecas.latex`](latex.md) module emits; no rendering claims are made
here — rendering is the front-end's job.

```cpp
import nimblecas.webexport;
```

Depends on [`core`](core.md) (for `Result<T>` and `MathError`),
[`symbolic`](symbolic.md) (for `Expr`, used by the `Expr` overload of `math_block`),
and [`latex`](latex.md) (for `to_latex`, which that overload routes through).

## The JSON contract

Every successful call emits one of two top-level object shapes. This is the exact
contract the `web/` front-end consumes:

```
PlotSpec:
  {"type":"plot","title":<str>,"xLabel":<str>,"yLabel":<str>,
   "xRange":[<num>,<num>]|null,"yRange":[<num>,<num>]|null,
   "series":[{"kind":"line"|"scatter","label":<str>,"color":"#rrggbb",
              "points":[[x,y],...]}]}

Document:
  {"type":"document","title":<str>,"blocks":[
     {"kind":"prose","text":<str>} | {"kind":"math","latex":<str>}
     | {"kind":"plot","plot":<PlotSpec>} ]}
```

Key order is fixed and stable. An unset `xRange`/`yRange` serializes as `null`; a set
range emits `[lo,hi]`. A `plot` block embeds a complete `PlotSpec` object verbatim
under its `"plot"` key, so the embedded object keeps its own `"type":"plot"`.

## `SeriesKind`

```cpp
enum class SeriesKind : std::uint8_t { line, scatter };
```

The two series shapes the front-end can render. Serialized as the string
`"line"` or `"scatter"` in a series object's `"kind"` field.

## `struct PlotSpecOptions`

Options common to a whole `PlotSpec` (analogous to `svgplot`'s `PlotOptions`), passed
by `const&` to every plot builder.

| Field | Type | Default | Meaning |
| :--- | :--- | :--- | :--- |
| `title` | `std::string` | `""` | Plot title. JSON-escaped on emission. |
| `xLabel` | `std::string` | `""` | X-axis label. JSON-escaped. |
| `yLabel` | `std::string` | `""` | Y-axis label. JSON-escaped. |
| `color` | `std::string` | `"#1f77b4"` | Default series colour (a valid `#rrggbb`). Seeds the single series produced by `plot_spec_line`/`scatter`/`function`; **ignored** by `plot_spec_multi`. |
| `label` | `std::string` | `""` | Default series label. Seeds the single-series builders; **ignored** by `plot_spec_multi`. |
| `xRange` | `std::optional<std::pair<double, double>>` | `std::nullopt` | Optional x-range. Emits `[lo,hi]` when set, `null` when absent. When set, both bounds must be finite (`domain_error` otherwise). |
| `yRange` | `std::optional<std::pair<double, double>>` | `std::nullopt` | Optional y-range, same rules as `xRange`. |

The multi-series builder takes per-series colour/label from each `Series` instead, so
its `opt.color`/`opt.label` are unused.

## `struct Series`

```cpp
struct Series {
    SeriesKind kind = SeriesKind::line;
    std::string label = "";
    std::string color = "#1f77b4";
    std::vector<double> xs{};
    std::vector<double> ys{};
};
```

One data series: its kind, front-end label/colour, and the raw `(xs, ys)` the CAS
produced. `xs.size()` must equal `ys.size()`, the pair must be non-empty, and all
values must be finite (`domain_error` otherwise). Used by `plot_spec_multi`.

## Plot builders

All four are free functions in namespace `nimblecas`, `[[nodiscard]]`, and return
`Result<std::string>` (the complete `PlotSpec` JSON on success).

```cpp
[[nodiscard]] auto plot_spec_line(std::span<const double> xs, std::span<const double> ys,
                                  const PlotSpecOptions& opt) -> Result<std::string>;

[[nodiscard]] auto plot_spec_scatter(std::span<const double> xs, std::span<const double> ys,
                                     const PlotSpecOptions& opt) -> Result<std::string>;

[[nodiscard]] auto plot_spec_function(std::function<double(double)> f, double xmin,
                                      double xmax, std::size_t samples,
                                      const PlotSpecOptions& opt) -> Result<std::string>;

[[nodiscard]] auto plot_spec_multi(std::span<const Series> series,
                                   const PlotSpecOptions& opt) -> Result<std::string>;
```

| Function | Behavior |
| :--- | :--- |
| `plot_spec_line` | Single-series **line** `PlotSpec` from a raw `(xs, ys)` pair. The series takes its `label`/`color` from `opt`. |
| `plot_spec_scatter` | Single-series **scatter** `PlotSpec`. Same validation as `plot_spec_line`. |
| `plot_spec_function` | Sample `f` at `samples` evenly spaced `x` in `[xmin, xmax]` (both ends inclusive; the final sample is anchored exactly at `xmax` to avoid step drift), **drop** any `x` whose `f(x)` is non-finite (mirrors `svgplot::plot_function`, e.g. `1/x` at `x = 0`), then delegate to `plot_spec_line` on the survivors. |
| `plot_spec_multi` | Combine several series into one **multi-series** `PlotSpec` (`svgplot` has no multi-series entry point). Each series is validated as in `plot_spec_line`; the shared `title`/labels/ranges come from `opt`, but its `color`/`label` are ignored — each series carries its own. |

## Document blocks

A `Document` is a title plus an ordered list of `Block`s, where `Block` is a
type-safe union of the three block kinds:

```cpp
struct ProseBlock { std::string text; };
struct MathBlock  { std::string latex; };  // LaTeX math (as nimblecas.latex emits, no $ delims)
struct PlotBlock  { std::string plot; };   // an already-serialized PlotSpec JSON
using Block = std::variant<ProseBlock, MathBlock, PlotBlock>;
```

### Block constructors

All are free functions in namespace `nimblecas` and `[[nodiscard]]`.

```cpp
[[nodiscard]] auto prose_block(std::string_view text) -> Block;
[[nodiscard]] auto math_block(std::string_view latex) -> Block;
[[nodiscard]] auto math_block(const Expr& e) -> Block;
[[nodiscard]] auto plot_block(std::string plot_json) -> Block;
```

| Constructor | Behavior |
| :--- | :--- |
| `prose_block` | A prose block carrying raw `text`. |
| `math_block(std::string_view)` | A math block from a **raw LaTeX string** (no `$` delimiters). |
| `math_block(const Expr&)` | A math block from an `Expr`, routed through [`nimblecas.latex`](latex.md)`::to_latex(e)`. |
| `plot_block` | A plot block wrapping an **already-serialized `PlotSpec` JSON** (from any `plot_spec_*` call). The payload is embedded verbatim; `document_to_json` guards that it is a JSON object. |

### `document_to_json`

```cpp
[[nodiscard]] auto document_to_json(std::string_view title,
                                    const std::vector<Block>& blocks) -> Result<std::string>;
```

Assemble a `Document` JSON from a title and an ordered list of blocks. A
`PlotBlock` payload must be a JSON **object** — non-empty and starting with `{` —
which every `plot_spec_*` result is; otherwise `domain_error`.

## Error model

Every failure path returns `MathError::domain_error`; this module raises no other
error. Non-finite input is **rejected**, not emitted — a NaN/inf would leak a literal
`"nan"`/`"inf"` token into the JSON.

| Condition | Error |
| :--- | :--- |
| `xs.size() != ys.size()` (length mismatch) | `MathError::domain_error` |
| `xs.empty()` (nothing to plot), or every `f(x)` dropped in `plot_spec_function` → empty surviving data | `MathError::domain_error` |
| Any non-finite value in `xs` or `ys` | `MathError::domain_error` |
| A set `opt.xRange`/`opt.yRange` bound is non-finite | `MathError::domain_error` |
| `plot_spec_function`: `samples < 2` | `MathError::domain_error` |
| `plot_spec_function`: `xmin >= xmax` (empty/inverted domain) | `MathError::domain_error` |
| `plot_spec_multi`: no series (empty span) | `MathError::domain_error` |
| `document_to_json`: a `PlotBlock` payload that is not a JSON object (empty or not starting with `{`) | `MathError::domain_error` |

There is no floating-point overflow contract here: this is `double` bridging
arithmetic, not the overflow-checked `int64`/`Rational` exact core.

## Worked examples

```cpp
import nimblecas.webexport;
import nimblecas.symbolic;
import nimblecas.core;
using namespace nimblecas;

// Line spec: xs={0,1,2}, ys={0,1,0}. Points are the raw doubles the CAS produced —
// NOT pixel-mapped — in shortest round-trip form (1.0 -> "1").
std::vector<double> xs{0.0, 1.0, 2.0};
std::vector<double> ys{0.0, 1.0, 0.0};
const std::string spec = plot_spec_line(xs, ys, PlotSpecOptions{}).value();
// spec contains: "points":[[0,0],[1,1],[2,0]]
// spec starts_with '{', carries "type":"plot", "kind":"line", "color":"#1f77b4",
//   and "xRange":null / "yRange":null (both unset).

// Options: title/labels are JSON-escaped; set ranges emit [lo,hi].
PlotSpecOptions o;
o.title  = "T";
o.xLabel = "X";
o.yLabel = "Y";
o.xRange = std::pair{0.0, 10.0};   // -> "xRange":[0,10]
o.yRange = std::pair{-1.0, 1.0};   // -> "yRange":[-1,1]
plot_spec_line(xs, ys, o).value(); // "title":"T", "xRange":[0,10], "yRange":[-1,1]

// Function spec: f(x)=x on [0,4] with 5 samples -> 5 exact points (2.5 -> "2.5").
plot_spec_function([](double x) { return x; }, 0.0, 4.0, 5, PlotSpecOptions{}).value();
// contains: "points":[[0,0],[1,1],[2,2],[3,3],[4,4]]

// Non-finite samples are dropped: 1/x at x=0 yields inf and is skipped, leaving 4
// finite points; no "inf"/"nan" leaks into the JSON.
plot_spec_function([](double x) { return 1.0 / x; }, -2.0, 2.0, 5, PlotSpecOptions{}).value();

// Multi-series: each Series keeps its own kind/label/colour (opt.color/label ignored).
Series a; a.kind = SeriesKind::line;    a.label = "sin"; a.color = "#ff0000";
a.xs = {0.0, 1.0}; a.ys = {0.0, 1.0};
Series b; b.kind = SeriesKind::scatter; b.label = "pts"; b.color = "#00ff00";
b.xs = {0.0, 1.0}; b.ys = {1.0, 0.0};
std::array<Series, 2> both{a, b};
plot_spec_multi(both, PlotSpecOptions{}).value();  // two "points" arrays, two colours

// A Document assembling all three block kinds: prose, math (from an Expr via
// to_latex), and a plot embedding the PlotSpec above.
auto x = Expr::symbol("x");
auto y = Expr::symbol("y");
std::vector<Block> blocks;
blocks.push_back(prose_block("Hello, world."));
blocks.push_back(math_block(x.add(y)));          // -> "latex":"x + y"
blocks.push_back(plot_block(spec));              // embeds the PlotSpec verbatim
const std::string doc = document_to_json("My Doc", blocks).value();
// doc contains: "type":"document", "title":"My Doc",
//   "kind":"prose" / "text":"Hello, world.",
//   "kind":"math"  / "latex":"x + y",
//   "kind":"plot"  / "plot":{"type":"plot" ...

// A math block can also carry raw LaTeX; its backslashes are JSON-escaped.
std::vector<Block> raw;
raw.push_back(math_block(std::string_view{"\\frac{1}{2}"}));
document_to_json("D", raw).value();  // contains "latex":"\\frac{1}{2}"

// Titles/labels/LaTeX are HTML-inline safe: '<'/'>' become </>, so a
// "</script>" substring can never terminate an HTML <script> embed.
PlotSpecOptions html;
html.title = "a\nb\tc</script>";
plot_spec_line(xs, ys, html).value();  // contains \n, \t, <, >; no "</script>"

// Error paths (all domain_error).
std::vector<double> shortY{0.0, 1.0};
plot_spec_line(xs, shortY, PlotSpecOptions{}).error();  // length mismatch
plot_spec_line({}, {}, PlotSpecOptions{}).error();      // empty data
plot_spec_function([](double v) { return v; }, 0.0, 1.0, 1, PlotSpecOptions{}).error();  // samples < 2
plot_spec_function([](double v) { return v; }, 1.0, 1.0, 5, PlotSpecOptions{}).error();  // xmin >= xmax
std::array<Series, 0> none{};
plot_spec_multi(none, PlotSpecOptions{}).error();       // no series
std::vector<Block> bad; bad.push_back(plot_block("not-json"));
document_to_json("D", bad).error();                     // malformed plot-block payload
```

## See also

- [`nimblecas.svgplot`](svgplot.md) — the SVG analogue: the same CAS plot data
  rendered to a static `<svg>` string instead of a JSON contract.
- [`nimblecas.latex`](latex.md) — the AST → LaTeX exporter that the `Expr` overload
  of `math_block` routes through (`to_latex`).
- [`nimblecas.symbolic`](symbolic.md) — the `Expr` tower a `MathBlock` can be built
  from.
- The [`web/` front-end](../../web/README.md) — the WebGPU/WASM browser renderer that
  consumes this JSON contract.
- [Documentation hub](../Index.md)
