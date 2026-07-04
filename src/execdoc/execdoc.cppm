// NimbleCAS executable document engine — "Live Notebooks" (ROADMAP 7.13 / PRD 2.17).
// @author Olumuyiwa Oluwasanmi
//
// Turns a Markdown document with embedded CAS code cells into an EXECUTED, rendered
// HTML document. Previously blocked, this feature is now unblocked because a text ->
// Expr reader (nimblecas.reader) exists: cells are parsed with `parse`, evaluated
// against a running session of symbolic bindings, and rendered to LaTeX for a
// MathJax/KaTeX-equipped host page.
//
// ---------------------------------------------------------------------------
// DATA-BRIDGE HONESTY BOUNDARY (project core discipline; cf. nimblecas.webexport
// and nimblecas.svgplot).
// ---------------------------------------------------------------------------
// This module EMITS an HTML/data string. It performs NO rendering, layout,
// interactivity, GPU, or WASM work, and it does NOT bundle a math typesetter: each
// cell result is emitted as a LaTeX fragment wrapped in the MathJax inline
// delimiters `\( ... \)`. Loading MathJax/KaTeX and actually typesetting is the HOST
// PAGE's responsibility. The engine executes EXACT CAS operations: a cell's value is
// the exact Expr the symbolic core produced, rendered by nimblecas.latex. No floating
// point is introduced by the engine itself (a cell may still contain a double leaf if
// the user wrote one — that is the user's number, not ours).
//
// A cell error (reader syntax error, overflow surfaced by simplify, a builtin used
// with the wrong arity/argument shape) is CAPTURED into CellResult.error and rendered
// as an inline error box. It never aborts the whole document and is never silently
// dropped: the document still renders, and cells before/after an errored cell are
// unaffected (Rule 32 — railway errors, no exceptions).
//
// ---------------------------------------------------------------------------
// SUPPORTED MARKDOWN SUBSET (deliberately small — this is NOT a CommonMark parser).
// ---------------------------------------------------------------------------
// The document is split, line by line, into an ordered list of blocks:
//   * An EXECUTABLE CODE CELL is opened by a line whose trimmed text is exactly
//     "```nimblecas" and closed by a line whose trimmed text is exactly "```". Only
//     this fence language is executed. The lines in between are the cell source.
//   * Any OTHER fenced block (a line starting with "```" that is not "```nimblecas")
//     passes through VERBATIM: its fence lines and body are emitted, HTML-escaped,
//     inside a <pre><code> block and are NOT executed.
//   * Everything else is PROSE: consecutive non-fence lines are gathered into a prose
//     block and emitted HTML-escaped (no Markdown inline formatting is interpreted —
//     prose text is passed through, only made HTML-safe).
// An unterminated "```nimblecas" fence consumes the remainder of the document as one
// cell (no exception; the trailing cell is still executed).
//
// ---------------------------------------------------------------------------
// CELL / SESSION SEMANTICS.
// ---------------------------------------------------------------------------
// A Session holds an ORDERED list of bindings (name -> Expr). A cell is a sequence of
// statements, one per line; blank lines are ignored. Each statement is either:
//   * an ASSIGNMENT `name = <expr>` where `name` is an identifier
//     ([A-Za-z_][A-Za-z0-9_]*) and `=` is a single '=' (not '=='/'<='/'>='/'!='):
//     the RHS is parsed, all currently-bound names are substituted into it, the
//     result is evaluated (see below), the binding name -> value is stored (updated in
//     place if it already exists), and the value becomes the statement result; or
//   * a bare EXPRESSION: parsed, bound names substituted, evaluated, value recorded.
// The CellResult carries the LAST statement's value/LaTeX. If a statement errors, the
// cell stops at that statement, records the error, and keeps any bindings earlier
// statements already made.
//
// Substitution applies every current binding, in binding order, via
// nimblecas::substitute; because later bindings are applied after earlier ones, a
// binding that refers to an earlier symbol resolves transitively.
//
// BUILTINS (mapped from the ROOT FunctionNode head to an engine call BEFORE the
// generic simplify; only the outermost call is a builtin — nested calls stay
// symbolic):
//   * diff(<expr>, <symbol>)  -> nimblecas::differentiate(subst(<expr>), <symbol>).
//                                The 2nd argument must be a bare symbol and is taken
//                                literally (NOT substituted); wrong arity or a
//                                non-symbol 2nd argument -> MathError::domain_error.
//   * simplify(<expr>)        -> nimblecas::simplify(subst(<expr>)); wrong arity ->
//                                MathError::domain_error.
// Any other function head (e.g. f(x), sin(x)) is NOT a builtin: it is evaluated
// generically (substitute + simplify) and therefore remains a symbolic apply(...).
//
// ---------------------------------------------------------------------------
// INCREMENTAL EXECUTION & CACHE-SOUNDNESS RULE.
// ---------------------------------------------------------------------------
// A DocumentCache memoises executed cells. A cell's cache key folds a hash of its
// exact source text with, for each currently-bound name the cell REFERENCES (the
// read-set), that binding's structural_hash. On a cache hit the stored result is
// reused (marked from_cache) and the bindings the cell produced are replayed into the
// session so downstream cells see them. A changed cell gets a different source hash ->
// re-executes; a cell whose read-set binding values changed (because an upstream cell
// changed) gets a different key -> re-executes; cells that read nothing that changed
// keep their key -> stay cached.
//
// SOUNDNESS: the cache MUST NEVER surface a stale value. Two defences:
//   (1) the read-set folds the actual binding *values* (structural_hash), so any
//       upstream change perturbs every dependent key; and
//   (2) a key hit is only accepted after VERIFYING the stored source text and the
//       stored read-set bindings are equal/equivalent to the current ones — a mere
//       structural_hash collision is treated as a miss and re-executed.
// The read-set is a sound over-approximation (every currently-bound name that appears
// as an identifier token in the source is included); when in doubt we re-execute, we
// never show an old result.
//
// Conforms to config/cpp_details.txt: C++23 named modules, `import std`, trailing
// return types, no owning raw pointers, std::expected error handling (no exceptions),
// [[nodiscard]].

export module nimblecas.execdoc;

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.diff;
import nimblecas.latex;
import nimblecas.reader;

export namespace nimblecas::execdoc {

// ---------------------------------------------------------------------------
// Result of executing one code cell.
// ---------------------------------------------------------------------------
// `value`/`latex` are the last statement's exact value and its LaTeX rendering when
// the cell succeeded; `error` holds the captured MathError when it did not (in which
// case `value` is empty and `latex` is empty). `from_cache` is set by the document
// driver when the result was reused from a DocumentCache.
struct CellResult {
    std::string source;
    std::optional<Expr> value;
    std::string latex;
    std::optional<MathError> error;
    bool from_cache{false};
};

// ---------------------------------------------------------------------------
// A cached, previously-executed cell (public because it appears in DocumentCache's
// interface). `read_bindings` is the verified read-set snapshot (sorted by name);
// `delta` is the set of bindings the cell produced, replayed on a cache hit.
// ---------------------------------------------------------------------------
struct CachedCell {
    std::string source;
    std::vector<std::pair<std::string, Expr>> read_bindings;
    std::vector<std::pair<std::string, Expr>> delta;
    CellResult result;

    // Exact-match guard against structural_hash collisions: same source text and the
    // same read-set (names equal, values structurally equivalent). Both vectors are
    // sorted by name so a positional comparison is valid.
    [[nodiscard]] auto matches(std::string_view src,
                               const std::vector<std::pair<std::string, Expr>>& snap) const
        -> bool {
        if (source != src || read_bindings.size() != snap.size()) {
            return false;
        }
        for (std::size_t i = 0; i < snap.size(); ++i) {
            if (read_bindings[i].first != snap[i].first ||
                !read_bindings[i].second.is_equivalent_to(snap[i].second)) {
                return false;
            }
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// A running notebook session: an ordered list of name -> Expr bindings and the cell
// evaluator that reads and extends them.
// ---------------------------------------------------------------------------
class Session {
public:
    // Execute one cell's source, mutating the session bindings. The outer Result is
    // always a value: a cell error is captured inside CellResult.error (honesty rule),
    // never returned as the error branch.
    [[nodiscard]] auto execute_cell(std::string_view src) -> Result<CellResult>;

    // Execute one cell and also report the bindings it created/updated (for the
    // document cache to replay on a later hit).
    [[nodiscard]] auto execute_cell_tracked(std::string_view src)
        -> std::pair<CellResult, std::vector<std::pair<std::string, Expr>>>;

    // The currently-bound names the given cell source references (sorted by name) —
    // the read-set used to key the incremental cache.
    [[nodiscard]] auto referenced_bindings(std::string_view src) const
        -> std::vector<std::pair<std::string, Expr>>;

    // Replay a set of bindings (used on a cache hit instead of re-executing).
    auto apply_bindings(const std::vector<std::pair<std::string, Expr>>& delta) -> void;

    [[nodiscard]] auto binding(std::string_view name) const -> std::optional<Expr>;
    [[nodiscard]] auto bindings() const -> std::span<const std::pair<std::string, Expr>> {
        return bindings_;
    }
    auto clear() -> void { bindings_.clear(); }

private:
    [[nodiscard]] auto run_cell(std::string_view src,
                                std::vector<std::pair<std::string, Expr>>& delta) -> CellResult;
    [[nodiscard]] auto eval_expression(std::string_view text) -> Result<Expr>;
    [[nodiscard]] auto eval_expr(const Expr& e) -> Result<Expr>;
    [[nodiscard]] auto substitute_bindings(const Expr& e) const -> Expr;
    auto set_binding(std::string name, Expr value,
                     std::vector<std::pair<std::string, Expr>>& delta) -> void;

    std::vector<std::pair<std::string, Expr>> bindings_{};
};

// ---------------------------------------------------------------------------
// Incremental cache across whole-document runs. Keyed by the cell key (source hash +
// read-set value hashes); a hit is only accepted after CachedCell::matches verifies.
// ---------------------------------------------------------------------------
class DocumentCache {
public:
    // Return the cached cell for `key` iff it exactly matches `src` + `snap`; a mere
    // hash collision (matches() == false) reports a miss so the driver re-executes.
    [[nodiscard]] auto lookup(std::size_t key, std::string_view src,
                              const std::vector<std::pair<std::string, Expr>>& snap) const
        -> std::optional<CachedCell> {
        const auto it = entries_.find(key);
        if (it == entries_.end() || !it->second.matches(src, snap)) {
            return std::nullopt;
        }
        return it->second;
    }

    auto store(std::size_t key, CachedCell entry) -> void {
        entries_.insert_or_assign(key, std::move(entry));
    }

    [[nodiscard]] auto size() const noexcept -> std::size_t { return entries_.size(); }
    auto clear() -> void { entries_.clear(); }

private:
    std::unordered_map<std::size_t, CachedCell> entries_{};
};

// Execute the cells of `markdown` in order, returning each cell's result. The
// no-cache overload always re-executes; the cached overload consults/updates `cache`.
[[nodiscard]] auto run_cells(std::string_view markdown) -> Result<std::vector<CellResult>>;
[[nodiscard]] auto run_cells(std::string_view markdown, DocumentCache& cache)
    -> Result<std::vector<CellResult>>;

// Execute `markdown` and render it to a single HTML string (prose passed through
// escaped, each cell as its input <pre> plus its LaTeX output in `\( ... \)` or an
// inline error box). MathJax/KaTeX is the host page's responsibility.
[[nodiscard]] auto run_document(std::string_view markdown) -> Result<std::string>;
[[nodiscard]] auto run_document(std::string_view markdown, DocumentCache& cache)
    -> Result<std::string>;

}  // namespace nimblecas::execdoc

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::execdoc {
namespace {

// --- small text utilities --------------------------------------------------

[[nodiscard]] auto hash_combine(std::size_t seed, std::size_t value) noexcept -> std::size_t {
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

[[nodiscard]] auto is_ident_start(char c) noexcept -> bool {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

[[nodiscard]] auto is_ident_char(char c) noexcept -> bool {
    return is_ident_start(c) || (c >= '0' && c <= '9');
}

[[nodiscard]] auto is_identifier(std::string_view s) noexcept -> bool {
    if (s.empty() || !is_ident_start(s.front())) {
        return false;
    }
    return std::ranges::all_of(s, [](char c) { return is_ident_char(c); });
}

// Trim ASCII whitespace from both ends, returning a view into `s`.
[[nodiscard]] auto trim(std::string_view s) noexcept -> std::string_view {
    constexpr std::string_view ws = " \t\r\n";
    const auto a = s.find_first_not_of(ws);
    if (a == std::string_view::npos) {
        return {};
    }
    const auto b = s.find_last_not_of(ws);
    return s.substr(a, b - a + 1);
}

// Split into lines on '\n', dropping a trailing '\r' (CRLF tolerant). Views alias `s`.
[[nodiscard]] auto split_lines(std::string_view s) -> std::vector<std::string_view> {
    std::vector<std::string_view> out;
    std::size_t start = 0;
    while (start <= s.size()) {
        const auto nl = s.find('\n', start);
        const auto end = (nl == std::string_view::npos) ? s.size() : nl;
        std::string_view line = s.substr(start, end - start);
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }
        out.push_back(line);
        if (nl == std::string_view::npos) {
            break;
        }
        start = nl + 1;
    }
    return out;
}

// The set of identifier tokens appearing in `text`. A sound over-approximation of the
// symbols a statement reads (function heads and assignment targets are harmlessly
// included; only names that are actually bound become read-set dependencies).
[[nodiscard]] auto identifiers_in(std::string_view text) -> std::set<std::string> {
    std::set<std::string> out;
    std::size_t i = 0;
    while (i < text.size()) {
        if (is_ident_start(text[i])) {
            const std::size_t begin = i;
            while (i < text.size() && is_ident_char(text[i])) {
                ++i;
            }
            out.emplace(text.substr(begin, i - begin));
        } else {
            ++i;
        }
    }
    return out;
}

// Collect every free symbol name appearing in an expression tree. Used to close the cache
// read-set over TRANSITIVE dependencies: a binding stored as an unevaluated tree (e.g. f = x+1
// with x unbound) carries free symbols that never appear in a later cell's source but that the
// cell's value still depends on after substitution.
auto collect_symbols(const Expr& e, std::set<std::string>& out) -> void {
    const ExprNode& n = e.node();
    if (const auto* s = std::get_if<SymbolNode>(&n.value)) {
        out.insert(s->name);
    } else if (const auto* a = std::get_if<AddNode>(&n.value)) {
        for (const Expr& t : a->terms) {
            collect_symbols(t, out);
        }
    } else if (const auto* m = std::get_if<MulNode>(&n.value)) {
        for (const Expr& f : m->factors) {
            collect_symbols(f, out);
        }
    } else if (const auto* p = std::get_if<PowerNode>(&n.value)) {
        collect_symbols(p->base, out);
        collect_symbols(p->exponent, out);
    } else if (const auto* fn = std::get_if<FunctionNode>(&n.value)) {
        for (const Expr& arg : fn->args) {
            collect_symbols(arg, out);
        }
    }
    // ConstantNode: no symbols.
}

// The non-blank, trimmed statement lines of a cell source.
[[nodiscard]] auto split_statements(std::string_view src) -> std::vector<std::string> {
    std::vector<std::string> out;
    for (std::string_view line : split_lines(src)) {
        const std::string_view t = trim(line);
        if (!t.empty()) {
            out.emplace_back(t);
        }
    }
    return out;
}

// If `stmt` is an assignment `name = rhs`, return (name, rhs); else nullopt. Only the
// first '=' is considered, and it must be a plain assignment (not '=='/'<='/'>='/'!=')
// with an identifier LHS.
[[nodiscard]] auto as_assignment(std::string_view stmt)
    -> std::optional<std::pair<std::string_view, std::string_view>> {
    const auto pos = stmt.find('=');
    if (pos == std::string_view::npos) {
        return std::nullopt;
    }
    if (pos + 1 < stmt.size() && stmt[pos + 1] == '=') {
        return std::nullopt;  // '=='
    }
    if (pos > 0) {
        const char prev = stmt[pos - 1];
        if (prev == '!' || prev == '<' || prev == '>' || prev == '=') {
            return std::nullopt;  // '!=', '<=', '>='
        }
    }
    const std::string_view lhs = trim(stmt.substr(0, pos));
    const std::string_view rhs = trim(stmt.substr(pos + 1));
    if (!is_identifier(lhs)) {
        return std::nullopt;
    }
    return std::pair{lhs, rhs};
}

// --- document block splitting ----------------------------------------------

enum class RawKind : std::uint8_t { prose, verbatim, cell };

struct RawBlock {
    RawKind kind{RawKind::prose};
    std::string text;  // prose text, verbatim block text, or cell source
};

[[nodiscard]] auto split_blocks(std::string_view md) -> std::vector<RawBlock> {
    std::vector<RawBlock> blocks;
    const std::vector<std::string_view> lines = split_lines(md);
    std::string prose;
    const auto flush_prose = [&] {
        if (!prose.empty()) {
            blocks.push_back(RawBlock{.kind = RawKind::prose, .text = prose});
            prose.clear();
        }
    };

    std::size_t i = 0;
    while (i < lines.size()) {
        const std::string_view t = trim(lines[i]);
        if (t == "```nimblecas") {
            flush_prose();
            ++i;
            std::string body;
            bool first = true;
            while (i < lines.size() && trim(lines[i]) != "```") {
                if (!first) {
                    body += '\n';
                }
                body.append(lines[i]);
                first = false;
                ++i;
            }
            if (i < lines.size()) {
                ++i;  // consume the closing fence
            }
            blocks.push_back(RawBlock{.kind = RawKind::cell, .text = std::move(body)});
        } else if (t.starts_with("```")) {
            flush_prose();
            std::string body(lines[i]);  // include the opening fence verbatim
            ++i;
            while (i < lines.size() && trim(lines[i]) != "```") {
                body += '\n';
                body.append(lines[i]);
                ++i;
            }
            if (i < lines.size()) {
                body += '\n';
                body.append(lines[i]);  // include the closing fence verbatim
                ++i;
            }
            blocks.push_back(RawBlock{.kind = RawKind::verbatim, .text = std::move(body)});
        } else {
            if (!prose.empty()) {
                prose += '\n';
            }
            prose.append(lines[i]);
            ++i;
        }
    }
    flush_prose();
    return blocks;
}

// --- HTML emission ---------------------------------------------------------

[[nodiscard]] auto html_escape(std::string_view s) -> std::string {
    std::string out;
    out.reserve(s.size());
    for (const char c : s) {
        switch (c) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            case '\'': out += "&#39;"; break;
            default: out += c; break;
        }
    }
    return out;
}

[[nodiscard]] auto render_cell(const CellResult& cell) -> std::string {
    std::string s = "<section class=\"ncas-cell\" data-from-cache=\"";
    s += cell.from_cache ? "true" : "false";
    s += "\"><pre class=\"ncas-input\"><code>";
    s += html_escape(cell.source);
    s += "</code></pre>";
    if (cell.error) {
        // The MathError message is engine text (no HTML metacharacters), escaped for
        // safety anyway.
        s += "<div class=\"ncas-error\">Error: ";
        s += html_escape(to_string_view(*cell.error));
        s += "</div>";
    } else if (cell.value) {
        // LaTeX is math for the host typesetter; it must NOT be HTML-escaped (its
        // backslashes/braces are meaningful and it never contains raw HTML specials).
        s += "<div class=\"ncas-output\">\\(";
        s += cell.latex;
        s += "\\)</div>";
    } else {
        s += "<div class=\"ncas-output\"></div>";  // empty cell (no statements)
    }
    s += "</section>";
    return s;
}

// --- the shared document driver --------------------------------------------

struct DocBlock {
    RawKind kind{RawKind::prose};
    std::string text;    // prose / verbatim text
    CellResult cell{};   // when kind == cell
};

// Execute every block of `md`, consulting `cache` when non-null. Always succeeds (cell
// errors are captured inside each CellResult); the Result wrapper is kept for API
// symmetry and any future hard-failure surface.
[[nodiscard]] auto execute_document(std::string_view md, DocumentCache* cache)
    -> Result<std::vector<DocBlock>> {
    Session session;
    std::vector<DocBlock> out;
    for (const RawBlock& b : split_blocks(md)) {
        if (b.kind != RawKind::cell) {
            out.push_back(DocBlock{.kind = b.kind, .text = b.text, .cell = {}});
            continue;
        }
        if (cache != nullptr) {
            const auto snap = session.referenced_bindings(b.text);
            std::size_t key = std::hash<std::string_view>{}(b.text);
            for (const auto& [name, value] : snap) {
                key = hash_combine(key, std::hash<std::string>{}(name));
                key = hash_combine(key, value.structural_hash());
            }
            if (auto hit = cache->lookup(key, b.text, snap)) {
                session.apply_bindings(hit->delta);
                CellResult r = hit->result;
                r.from_cache = true;
                out.push_back(DocBlock{.kind = RawKind::cell, .text = {}, .cell = std::move(r)});
                continue;
            }
            auto [cr, delta] = session.execute_cell_tracked(b.text);
            cr.from_cache = false;
            cache->store(key, CachedCell{.source = std::string(b.text),
                                         .read_bindings = snap,
                                         .delta = delta,
                                         .result = cr});
            out.push_back(DocBlock{.kind = RawKind::cell, .text = {}, .cell = std::move(cr)});
        } else {
            auto [cr, delta] = session.execute_cell_tracked(b.text);
            cr.from_cache = false;
            out.push_back(DocBlock{.kind = RawKind::cell, .text = {}, .cell = std::move(cr)});
        }
    }
    return out;
}

[[nodiscard]] auto render_html(const std::vector<DocBlock>& blocks) -> std::string {
    std::string html = "<div class=\"nimblecas-document\">";
    for (const DocBlock& b : blocks) {
        switch (b.kind) {
            case RawKind::prose:
                html += "<div class=\"ncas-prose\">";
                html += html_escape(b.text);
                html += "</div>";
                break;
            case RawKind::verbatim:
                html += "<pre class=\"ncas-verbatim\"><code>";
                html += html_escape(b.text);
                html += "</code></pre>";
                break;
            case RawKind::cell:
                html += render_cell(b.cell);
                break;
        }
    }
    html += "</div>";
    return html;
}

[[nodiscard]] auto cells_of(const std::vector<DocBlock>& blocks) -> std::vector<CellResult> {
    std::vector<CellResult> cells;
    for (const DocBlock& b : blocks) {
        if (b.kind == RawKind::cell) {
            cells.push_back(b.cell);
        }
    }
    return cells;
}

}  // namespace

// --- Session ---------------------------------------------------------------

auto Session::binding(std::string_view name) const -> std::optional<Expr> {
    for (const auto& [n, v] : bindings_) {
        if (n == name) {
            return v;
        }
    }
    return std::nullopt;
}

auto Session::apply_bindings(const std::vector<std::pair<std::string, Expr>>& delta) -> void {
    for (const auto& [name, value] : delta) {
        bool found = false;
        for (auto& [n, v] : bindings_) {
            if (n == name) {
                v = value;
                found = true;
                break;
            }
        }
        if (!found) {
            bindings_.emplace_back(name, value);
        }
    }
}

auto Session::set_binding(std::string name, Expr value,
                          std::vector<std::pair<std::string, Expr>>& delta) -> void {
    // Update-or-append in the live bindings...
    bool found = false;
    for (auto& [n, v] : bindings_) {
        if (n == name) {
            v = value;
            found = true;
            break;
        }
    }
    if (!found) {
        bindings_.emplace_back(name, value);
    }
    // ...and coalesce into the per-cell delta (last write wins).
    for (auto& [n, v] : delta) {
        if (n == name) {
            v = std::move(value);
            return;
        }
    }
    delta.emplace_back(std::move(name), std::move(value));
}

auto Session::substitute_bindings(const Expr& e) const -> Expr {
    Expr current = e;
    for (const auto& [name, value] : bindings_) {
        current = substitute(current, Expr::symbol(name), value);
    }
    return current;
}

auto Session::referenced_bindings(std::string_view src) const
    -> std::vector<std::pair<std::string, Expr>> {
    // Start from the names textually present in the source, then close TRANSITIVELY over the
    // free symbols of each referenced binding's value. substitute_bindings() applies every
    // binding, so a cell referencing `f` (stored as the tree x+1) actually depends on `x` too;
    // without folding x into the read-set an upstream change to x would leave the cache key
    // unchanged and serve a STALE result. Iterating to a fixed point captures the full chain.
    std::set<std::string> needed = identifiers_in(src);
    for (bool changed = true; changed;) {
        changed = false;
        for (const auto& [n, v] : bindings_) {
            if (!needed.contains(n)) {
                continue;
            }
            std::set<std::string> syms;
            collect_symbols(v, syms);
            for (const auto& s : syms) {
                if (needed.insert(s).second) {
                    changed = true;  // a newly-pulled-in symbol may itself be a bound name
                }
            }
        }
    }
    std::vector<std::pair<std::string, Expr>> out;
    for (const auto& [n, v] : bindings_) {
        if (needed.contains(n)) {
            out.emplace_back(n, v);
        }
    }
    std::ranges::sort(out, {}, [](const auto& p) { return p.first; });
    return out;
}

auto Session::eval_expr(const Expr& e) -> Result<Expr> {
    // Root-level builtins dispatch to an engine call BEFORE the generic simplify.
    if (auto fn = as<FunctionNode>(e.node().value)) {
        const FunctionNode& f = **fn;
        if (f.name == "diff") {
            if (f.args.size() != 2) {
                return make_error<Expr>(MathError::domain_error);  // wrong arity
            }
            auto sym = as<SymbolNode>(f.args[1].node().value);
            if (!sym) {
                return make_error<Expr>(MathError::domain_error);  // 2nd arg not a symbol
            }
            const Expr arg = substitute_bindings(f.args[0]);
            return differentiate(arg, (*sym)->name);
        }
        if (f.name == "simplify") {
            if (f.args.size() != 1) {
                return make_error<Expr>(MathError::domain_error);  // wrong arity
            }
            return simplify(substitute_bindings(f.args[0]));
        }
        // Any other function head is not a builtin -> generic evaluation below.
    }
    return simplify(substitute_bindings(e));
}

auto Session::eval_expression(std::string_view text) -> Result<Expr> {
    auto parsed = parse(text);  // text -> Expr (reader); malformed input -> MathError
    if (!parsed) {
        return make_error<Expr>(parsed.error());
    }
    return eval_expr(*parsed);
}

auto Session::run_cell(std::string_view src, std::vector<std::pair<std::string, Expr>>& delta)
    -> CellResult {
    CellResult cr;
    cr.source = std::string(src);
    for (const std::string& stmt : split_statements(src)) {
        Result<Expr> value = [&]() -> Result<Expr> {
            if (auto assign = as_assignment(stmt)) {
                return eval_expression(assign->second);
            }
            return eval_expression(stmt);
        }();
        if (!value) {
            cr.error = value.error();
            cr.value.reset();
            cr.latex.clear();
            return cr;  // stop the cell; bindings from earlier statements persist
        }
        if (auto assign = as_assignment(stmt)) {
            set_binding(std::string(assign->first), *value, delta);
        }
        cr.value = *value;
        cr.latex = to_latex(*value);
        cr.error.reset();
    }
    return cr;
}

auto Session::execute_cell(std::string_view src) -> Result<CellResult> {
    std::vector<std::pair<std::string, Expr>> delta;
    return run_cell(src, delta);
}

auto Session::execute_cell_tracked(std::string_view src)
    -> std::pair<CellResult, std::vector<std::pair<std::string, Expr>>> {
    std::vector<std::pair<std::string, Expr>> delta;
    CellResult cr = run_cell(src, delta);
    return {std::move(cr), std::move(delta)};
}

// --- free functions --------------------------------------------------------

auto run_cells(std::string_view markdown) -> Result<std::vector<CellResult>> {
    auto blocks = execute_document(markdown, nullptr);
    if (!blocks) {
        return make_error<std::vector<CellResult>>(blocks.error());
    }
    return cells_of(*blocks);
}

auto run_cells(std::string_view markdown, DocumentCache& cache)
    -> Result<std::vector<CellResult>> {
    auto blocks = execute_document(markdown, &cache);
    if (!blocks) {
        return make_error<std::vector<CellResult>>(blocks.error());
    }
    return cells_of(*blocks);
}

auto run_document(std::string_view markdown) -> Result<std::string> {
    auto blocks = execute_document(markdown, nullptr);
    if (!blocks) {
        return make_error<std::string>(blocks.error());
    }
    return render_html(*blocks);
}

auto run_document(std::string_view markdown, DocumentCache& cache) -> Result<std::string> {
    auto blocks = execute_document(markdown, &cache);
    if (!blocks) {
        return make_error<std::string>(blocks.error());
    }
    return render_html(*blocks);
}

}  // namespace nimblecas::execdoc
