# `nimblecas.execdoc` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/execdoc/execdoc.cppm` · namespace `nimblecas::execdoc`

The **Executable Document Engine** (ROADMAP §7.13 / PRD §2.17 "Live Notebooks") —
turns a Markdown document with embedded CAS code cells into an executed, rendered
document. It was long blocked for want of a text→`Expr` eval surface; the
[`reader`](reader.md) parser unblocks it.

## What it is (and is not)

The engine **emits an HTML/data string** — it is a data bridge, like
[`webexport`](../reference/webexport.md) and [`svgplot`](../reference/svgplot.md).
It does **no** rendering, interactivity, GPU, or WASM; MathJax / KaTeX in the host
page renders the `\( … \)` LaTeX it emits. It executes **exact** CAS operations —
no floating point is introduced by the engine.

## Markdown & cell subset

- An **executable cell** is a fence whose opening line is exactly ` ```nimblecas `,
  closed by a line that is exactly ` ``` ` (an unterminated fence consumes to EOF).
- Any **other** ` ``` ` fence passes through verbatim inside an escaped
  `<pre><code>`.
- Everything else is prose, emitted **HTML-escaped** (no inline Markdown is
  interpreted — this is not a CommonMark parser).

Within a cell, statements are one per line (blanks ignored), each either an
**assignment** `name = <expr>` (single `=`, identifier LHS) or a bare
**expression**. The RHS is parsed, all current bindings are substituted (in order,
so dependencies are transitive), evaluated, and stored/recorded. A statement error
stops that cell but keeps prior bindings.

## Builtins

Only the **outermost** function-call head is treated as a builtin (everything else,
e.g. `sin(x)`, stays a symbolic `apply`):

- `diff(<expr>, <symbol>)` → `differentiate` (the 2nd argument must be a bare
  symbol, taken literally, not substituted; wrong arity / non-symbol → `domain_error`).
- `simplify(<expr>)` → `simplify` (wrong arity → `domain_error`).

## Public API (namespace `nimblecas::execdoc`)

- `struct CellResult { std::string source; std::optional<Expr> value; std::string latex; std::optional<MathError> error; bool from_cache; }`.
- `class Session` — `execute_cell(src) -> Result<CellResult>`, plus binding
  accessors; bindings are an ordered `vector<pair<string, Expr>>`.
- `class DocumentCache` — incremental-execution cache.
- `run_cells(md)` / `run_cells(md, cache) -> Result<vector<CellResult>>` — per-cell
  introspection (exposes `from_cache` / `value` / `error`).
- `run_document(md)` / `run_document(md, cache) -> Result<std::string>` — the HTML.

## Incremental caching — soundness first

A cell's key is the hash of its exact source text folded with, for every referenced
bound name, that binding's `structural_hash`. Two defences ensure a cache hit never
surfaces a **stale** result:

1. folding the binding **values** means any upstream change perturbs every
   dependent key; and
2. a key hit is accepted only after a `matches()` re-check of the stored source
   text and read-set bindings (`is_equivalent_to`) — a bare hash collision is a
   miss and re-executes.

The read-set (identifier tokens ∩ bound names) is a **sound over-approximation**;
when in doubt the engine re-executes rather than risk a stale value.

## Honesty & safety

- Exact CAS results; no float introduced by the engine.
- A cell error (parse error, unbound symbol as a value, overflow, bad builtin
  arity) is captured into `CellResult.error` and rendered as an inline error box —
  it never aborts the document and is never silently dropped.
- Prose and cell **source** text are HTML-escaped; LaTeX output is intentionally
  not escaped (it is math for MathJax).

## Verified in the tests

`x = 2` then `x^2 + 1` → `5`; cross-cell bindings persist; `diff(x^3, x) → 3x²`;
wrong-arity builtin captured; incremental re-run marks unchanged cells `from_cache`
while a changed cell **and its downstream dependents** re-execute and independent
cells stay cached; a malformed middle cell is captured without aborting neighbours;
prose `<`/`&`/`<b>` is escaped.

## See also

- [`reader.md`](reader.md) — the parser that is the cell eval surface.
- [`latex.md`](latex.md) — the `Expr → LaTeX` renderer used for results.
- [`webexport.md`](webexport.md) · [`svgplot.md`](svgplot.md) — sibling data bridges to the web front-end.
