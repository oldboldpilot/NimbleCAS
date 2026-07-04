# NimbleCAS Viewer

A self-contained WebGPU document + plotting viewer for [NimbleCAS](../README.md).
It renders documents produced by the C++ `webexport` module: prose (Markdown
subset), math (LaTeX → native MathML), and plots (real WebGPU pipeline with a
Canvas2D fallback). **No build toolchain, no network, no external files.**

## How to open

- **Simplest:** double-click `web/index.html` (or open it in Chrome via
  `File → Open`). It is fully standalone — inline CSS/JS/WGSL and an embedded
  sample document, so it renders immediately.
- **As a claude.ai Artifact:** paste the contents of `index.html`. Everything is
  inline, so it satisfies the strict Artifact CSP (no external hosts).
- **Module version with the live CAS:** open **`app.html`** — the ready-made shell
  that loads the split ES modules (`app.js` + `renderer.js`) **and** the real
  in-browser CAS engine (see below). It renders the WebGPU/Canvas2D plots *and*
  evaluates executable `nimblecas` cells and the live CAS box. Because the ES
  modules and the `.wasm` are fetched, serve it over `http(s)://` (e.g.
  `python -m http.server`); `index.html`, by contrast, needs no server.

`index.html` is the authoritative zero-dependency viewer and always works on its
own (no CAS engine, to keep it `file://`-openable). `app.html` is the fuller
served build that adds the live WASM CAS.

## In-browser CAS engine (WebAssembly)

Beyond the viewer, the **real exact symbolic engine** runs in the browser via
WebAssembly — not just the freestanding numeric kernel. `nimblecas.js` +
`nimblecas.wasm` (~390 KB) are the `core → parallel → symbolic → cache → simplify →
diff → latex → reader` slice compiled with Emscripten (see
[`docs/architecture/wasm-build.md`](../docs/architecture/wasm-build.md); rebuild with
[`scripts/build-wasm-slice.sh`](../scripts/build-wasm-slice.sh)).

- **`cas-repl.html`** is a minimal in-browser REPL over the engine: type an
  expression → it is parsed, simplified, and rendered as LaTeX (MathJax). Because it
  fetches `nimblecas.wasm`, serve it over `http(s)://` (e.g. `python -m http.server`),
  not `file://`.
- **API:** one C export, `nimblecas_eval_latex(const char*) -> const char*`
  (text → parse → simplify → LaTeX, exact over ℚ). Call it as
  `Module.ccall('nimblecas_eval_latex', 'string', ['string'], [expr])`. Everything
  stays exact — `2/4 + 1/4` yields `\frac{3}{4}`, not a float — and a parse/eval
  failure returns a LaTeX `\text{…}` marker rather than throwing.

This is the substrate a future WebGPU document front-end will call to evaluate live
`nimblecas` cells (the [`execdoc`](../docs/reference/execdoc.md) engine's browser
counterpart); wiring it into `app.js` alongside the plot renderer is the remaining
front-end step.

## Browser requirements

- **WebGPU path:** Chrome/Edge 113+ (desktop) with WebGPU enabled (default on
  recent versions). The viewer feature-detects `navigator.gpu` and requests an
  adapter/device.
- **Canvas2D fallback:** if `navigator.gpu` is absent, no adapter/device can be
  acquired, or a per-plot GPU render throws, the viewer automatically renders
  the identical plot with Canvas2D. Any modern browser (including Firefox and
  Safari) therefore works. A badge in the header shows **"WebGPU"** or
  **"Canvas2D fallback"** so you always know which backend is live.
- **MathML:** rendered natively (Chrome/Edge 109+, Firefox, Safari all support
  MathML Core). Unsupported LaTeX degrades to the raw source in a styled `code`
  span, so nothing breaks on older engines.

## JSON contract (consumed)

```jsonc
// PlotSpec
{
  "type": "plot",
  "title": "string",
  "xLabel": "string",
  "yLabel": "string",
  "xRange": [min, max],        // or null -> auto-fit from points
  "yRange": [min, max],        // or null -> auto-fit from points
  "series": [
    {
      "kind": "line" | "scatter",
      "label": "string",
      "color": "#rrggbb",       // optional; falls back to a categorical palette
      "points": [[x, y], ...]
    }
  ]
}

// Document
{
  "type": "document",
  "title": "string",
  "blocks": [
    { "kind": "prose", "text": "markdown-subset string" },
    { "kind": "math",  "latex": "latex-subset string" },
    { "kind": "plot",  "plot": PlotSpec }
  ]
}
```

Use **Load document…** in the header to load any JSON matching this contract; it
re-renders in place. The embedded sample remains the default. A ready-made
`sample-document.json` is included to try the loader.

### Prose (Markdown subset)

Headings `#`, `##`, `###`; `**bold**`; `` `code` ``; blank-line-separated
paragraphs; and `-`/`*` unordered lists. All text is HTML-escaped first.

### Math (LaTeX → MathML subset)

**Supported:** `\frac{}{}`, superscript `^`, subscript `_` (and combined
`^`+`_` → `msubsup`), `\sqrt{}`, `{}` grouping, Greek letters (`\pi \alpha
\beta \gamma \delta \theta \lambda \mu \sigma \phi \omega`, upper- and
lowercase, plus more), common operators (`\cdot \times \div \pm \le \ge \neq
\approx \int \sum \prod \infty \partial \to \in \cup \cap …`), function names
(`\sin \cos \tan \log \ln \exp \det \lim …` as upright identifiers), numbers,
identifiers, and basic operators `+ - = ( )`. `\left`/`\right` delimiters are
accepted (rendered as plain operators).

**Falls back to source:** any **unknown command** (e.g. `\begin{matrix}`,
`\overline`, `\hat`, environments/matrices, `\text`, alignment) causes the whole
block to display the original LaTeX in a styled `code` span. The converter is
intentionally small and correct for the common inline cases, not a full LaTeX
engine.

## Rendering backends

| Backend  | When it runs | What it draws |
|----------|--------------|---------------|
| WebGPU   | `navigator.gpu` present **and** adapter+device acquired | One triangle-list pipeline (WGSL in `shaders.wgsl`). Panel, gridlines, axis border, line series as triangulated quad strips (width honored), scatter as point quads. Axis/tick/title/legend **text** is drawn by a Canvas2D overlay on top (WebGPU cannot draw text). |
| Canvas2D | WebGPU absent/unavailable, or a GPU render throws | Identical geometry + text via the 2D context. Fully independent code path. |

Data→clip-space mapping honors `xRange`/`yRange`, auto-fitting (with 5% padding)
when either is `null`. Colors, ticks, and layout are shared so the two backends
match pixel-for-pixel in layout.

## Optional live-compute WASM hook

The **"Sample a polynomial live"** control evaluates a cubic across `x ∈ [−4, 4]`
and plots it through the active renderer.

**Assumed contract:** a same-directory `kernel.wasm` — a freestanding module
exporting:

- `memory` — a `WebAssembly.Memory`
- `poly_eval(ptr: i32, n: i32, x: f64) -> f64` — Horner evaluation of `n`
  little-endian `f64` coefficients `c0..c_{n-1}` (so `p(x) = Σ c_k·x^k`) located
  at byte offset `ptr` in `memory`.

The viewer writes the coefficients into `memory` at byte offset `1024`, then
calls `poly_eval` per sample. If `kernel.wasm` is absent, lacks these exports,
or fails to instantiate (e.g. wrong MIME under `file://`), the **identical**
polynomial is evaluated in plain JavaScript instead. A tag shows whether the
WASM kernel or the JS kernel produced the samples.

Example freestanding C compiled to WASM that satisfies the contract:

```c
// clang --target=wasm32 -nostdlib -Wl,--no-entry -Wl,--export-all \
//   -o kernel.wasm kernel.c
double poly_eval(const double* c, int n, double x) {
    double acc = 0.0;
    for (int k = n - 1; k >= 0; --k) acc = acc * x + c[k];
    return acc;
}
```

## Browser-compatibility caveats

- **WebGPU** is desktop-first. On browsers/OSes without it (Safari <18, older
  Firefox, most mobile) the badge reads "Canvas2D fallback" and everything still
  works.
- **`fetch()` under `file://`** is blocked by browsers, so the WASM hook and the
  module-version WGSL fetch will silently fall back when `index.html` is opened
  directly from disk. This is by design — `index.html` embeds its WGSL and works
  regardless; only the optional WASM path needs `http(s)://` to load a real
  kernel. Serve the folder (`python -m http.server`) to exercise WASM.
- **Line joins** in the WebGPU path use simple per-segment quads (no miter
  joins); at very sharp angles with thick lines you may see tiny gaps. The
  Canvas2D path uses rounded joins. Both are fine for typical data curves.
- **devicePixelRatio** is capped at 2 for the canvas backing store to bound
  memory on high-DPI displays.
