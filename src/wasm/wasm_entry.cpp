// NimbleCAS WebAssembly entry surface — the real symbolic engine in the browser.
// @author Olumuyiwa Oluwasanmi
//
// This is a plain (non-module) translation unit that IMPORTS the CAS modules and exposes a
// tiny C ABI the `web/` front-end calls: text -> parse -> simplify -> LaTeX. It is the
// "full-CAS to WASM" slice described in docs/architecture/wasm-build.md — the symbolic core
// (core, symbolic, cache, simplify, diff, latex, reader) compiled with em++, NOT the
// freestanding numeric kernel. GPU/CUDA, the nanobind Python bindings, SIMD, and TBB are all
// excluded from this configuration (nimblecas.parallel auto-selects its serial backend on
// wasm via __has_include, so no code change is needed there).
//
// Honesty: this runs the SAME exact-over-Q engine as the native build. `nimblecas_eval_latex`
// returns the LaTeX of simplify(parse(input)); a parse or evaluation failure returns a LaTeX
// \text{...} marker string rather than crashing. The returned pointer is owned by a static
// std::string (valid until the next call) — the single-threaded call-then-read pattern the
// front-end uses.

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.reader;
import nimblecas.simplify;
import nimblecas.latex;

using nimblecas::parse;
using nimblecas::simplify;
using nimblecas::to_latex;

// Exported to JS as `_nimblecas_eval_latex` (call via Module.ccall(..., 'string', ['string'])).
extern "C" [[gnu::used]] auto nimblecas_eval_latex(const char* input) -> const char* {
    static std::string out;  // kept alive for the returned pointer (single-threaded use)
    const std::string_view text = input != nullptr ? std::string_view{input} : std::string_view{};

    auto parsed = parse(text);
    if (!parsed) {
        out = "\\text{parse error}";
        return out.c_str();
    }
    auto reduced = simplify(*parsed);
    if (!reduced) {
        out = "\\text{cannot evaluate}";
        return out.c_str();
    }
    out = to_latex(*reduced);
    return out.c_str();
}
