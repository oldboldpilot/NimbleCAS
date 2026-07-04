// NimbleCAS WebAssembly entry surface — the real symbolic + linear-algebra engine in the
// browser.
// @author Olumuyiwa Oluwasanmi
//
// This is a plain (non-module) translation unit that IMPORTS the CAS modules and exposes a
// tiny C ABI the `web/` front-end calls. It is the "full-CAS to WASM" slice described in
// docs/architecture/wasm-build.md — the symbolic core (core, symbolic, cache, simplify, diff,
// latex, reader) PLUS the numeric / linear-algebra chain (simd, polynomial, ratpoly, matrix,
// roots, numeric, matdecomp, bandsolve, eigen) compiled with em++, NOT the freestanding
// numeric kernel. GPU/CUDA and the nanobind Python bindings are excluded from this
// configuration. nimblecas.parallel auto-selects its serial backend on wasm (no TBB headers)
// and nimblecas.simd auto-selects its scalar-only backend on wasm (no x86 intrinsics — see
// the portability note at the top of src/simd/simd.cppm); no code change was needed in either
// for this port beyond simd's own header guard.
//
// Honesty: both endpoints run the SAME exact engine as the native build.
//   nimblecas_eval_latex        — text -> parse -> simplify -> LaTeX (symbolic core).
//   nimblecas_matrix_det_latex  — text -> exact rational matrix -> determinant -> LaTeX
//                                  (linear-algebra chain), proving the widened slice is
//                                  actually reachable, not just link-time dead weight.
// A parse/evaluation failure returns a LaTeX \text{...} marker string rather than crashing
// or guessing. Each returned pointer is owned by its own static std::string (valid until the
// next call to that same function) — the single-threaded call-then-read pattern the
// front-end uses.

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.reader;
import nimblecas.simplify;
import nimblecas.latex;
import nimblecas.ratpoly;
import nimblecas.matrix;

using nimblecas::ConstantNode;
using nimblecas::Expr;
using nimblecas::make_error;
using nimblecas::MathError;
using nimblecas::Matrix;
using nimblecas::parse;
using nimblecas::Rational;
using nimblecas::Result;
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

namespace {

// A parsed-and-simplified Expr constant as an exact Rational (an int64 leaf, or an int64/int64
// pair). A symbol, an unreduced expression, or a double literal is not a valid matrix cell.
[[nodiscard]] auto expr_to_rational(const Expr& e) -> Result<Rational> {
    const auto* c = std::get_if<ConstantNode>(&e.node().value);
    if (c == nullptr) {
        return make_error<Rational>(MathError::domain_error);
    }
    return std::visit(
        [](const auto& v) -> Result<Rational> {
            using V = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<V, std::int64_t>) {
                return Rational::from_int(v);
            } else if constexpr (std::is_same_v<V, std::pair<std::int64_t, std::int64_t>>) {
                return Rational::from_int(v.first).divide(Rational::from_int(v.second));
            } else {
                return make_error<Rational>(MathError::not_implemented);  // a double literal
            }
        },
        c->value);
}

// Parse "a,b;c,d" — semicolon-separated rows, comma-separated cells, each cell any expression
// `parse` + `simplify` reduces to an exact rational constant (e.g. "3/4", "-2", "1/2 + 1/3") —
// into a Matrix. Cells are trimmed of surrounding spaces before parsing. Malformed input
// (a non-numeric cell, ragged rows, empty input) surfaces the documented MathError from
// `parse` / `simplify` / `Matrix::from_rows` rather than silently producing a wrong matrix.
[[nodiscard]] auto parse_matrix(std::string_view text) -> Result<Matrix> {
    std::vector<std::vector<Rational>> rows;
    std::size_t row_start = 0;
    while (row_start <= text.size()) {
        const auto row_end = text.find(';', row_start);
        const std::string_view row_text = text.substr(
            row_start, row_end == std::string_view::npos ? std::string_view::npos
                                                          : row_end - row_start);
        std::vector<Rational> row;
        std::size_t cell_start = 0;
        while (cell_start <= row_text.size()) {
            const auto cell_end = row_text.find(',', cell_start);
            std::string_view cell = row_text.substr(
                cell_start, cell_end == std::string_view::npos ? std::string_view::npos
                                                                : cell_end - cell_start);
            while (!cell.empty() && cell.front() == ' ') {
                cell.remove_prefix(1);
            }
            while (!cell.empty() && cell.back() == ' ') {
                cell.remove_suffix(1);
            }
            auto parsed = parse(cell);
            if (!parsed) {
                return make_error<Matrix>(parsed.error());
            }
            auto reduced = simplify(*parsed);
            if (!reduced) {
                return make_error<Matrix>(reduced.error());
            }
            auto r = expr_to_rational(*reduced);
            if (!r) {
                return make_error<Matrix>(r.error());
            }
            row.push_back(*r);
            if (cell_end == std::string_view::npos) {
                break;
            }
            cell_start = cell_end + 1;
        }
        rows.push_back(std::move(row));
        if (row_end == std::string_view::npos) {
            break;
        }
        row_start = row_end + 1;
    }
    return Matrix::from_rows(std::move(rows));
}

}  // namespace

// Exported to JS as `_nimblecas_matrix_det_latex`. Input format: "a,b;c,d;..." (semicolon
// rows, comma cells). Returns the EXACT rational determinant as LaTeX, or a \text{...} error
// marker on malformed input / a non-square matrix — never an approximate or wrong value.
extern "C" [[gnu::used]] auto nimblecas_matrix_det_latex(const char* input) -> const char* {
    static std::string out;
    const std::string_view text = input != nullptr ? std::string_view{input} : std::string_view{};

    auto m = parse_matrix(text);
    if (!m) {
        out = "\\text{matrix error}";
        return out.c_str();
    }
    auto det = m->determinant();
    if (!det) {
        out = "\\text{determinant error}";
        return out.c_str();
    }
    auto e = Expr::rational(det->numerator(), det->denominator());
    if (!e) {
        out = "\\text{overflow}";
        return out.c_str();
    }
    out = to_latex(*e);
    return out.c_str();
}
