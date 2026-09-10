// NimbleCAS Prolog reader and writer — source text to terms, and terms back to source text.
// @author Olumuyiwa Oluwasanmi
//
// A complete Prolog READER for the term language of `nimblecas.logic`: a tokenizer, a
// user-extensible operator table, an operator-precedence parser, and a writer that emits source
// text which reads back as the same term. Together they turn the engine from something you
// build terms for with C++ factory calls into something you hand a program to.
//
// WHAT "COMPLETE" MEANS HERE, precisely, because the word is easy to overclaim:
//
//  * Tokens: names, quoted atoms with the full escape set, symbolic atoms by maximal munch,
//    variables, `_` as the anonymous variable, integers in decimal / `0x` / `0o` / `0b` /
//    `0'c` character-code form, all bracket and solo tokens, `%` and `/* */` comments, and the
//    end token (a `.` followed by layout, which is what separates `'.'(H,T)` and `=..` from a
//    clause terminator).
//  * Operators: the ISO table, and `op/3` directives in the source can add to it or remove
//    from it mid-file, so a program that defines its own notation reads correctly.
//  * Parsing: full precedence and associativity (xfx, xfy, yfx, fy, fx, xf, yf), lists with
//    `|` tails, curly terms, and the prefix-operator disambiguation rules that decide between
//    `- 1` the negative integer, `-(1)` the compound, and `-` the bare atom.
//  * Writing: `to_source` re-reads as the same term. That is a stronger property than "looks
//    right", it is the one worth testing, and it is what the round-trip tests check.
//
// WHAT IS DELIBERATELY ABSENT. There are no FLOATS: the engine's only numeric term is a
// 64-bit integer, so `3.14` is a `not_implemented` error rather than a silently truncated `3`.
// There are no strings, no `0.0e10` syntax, and no character-escape output beyond what
// round-trips. `{a}` is read as `'{}'(a)` exactly as ISO requires.
//
// HONESTY (Code Policy Rule 32). Every entry point returns `Result<T>`; a malformed input is a
// `syntax_error`, an integer that does not fit 64 bits is an `overflow`, and a float literal is
// `not_implemented`. Nothing is guessed, nothing is skipped silently, and no partially parsed
// term is ever returned as though it were the whole one.

export module nimblecas.logic_parser;

import std;
import nimblecas.core;
import nimblecas.logic;

export namespace nimblecas::logic_parser {

// ---------------------------------------------------------------------------
// The operator table.
// ---------------------------------------------------------------------------

// Operator classes, in ISO notation: `f` is the operator, `x` an argument whose priority must
// be strictly lower, `y` one that may equal the operator's own. So `yfx` is left-associative
// and `xfy` right-associative.
enum class OpType : std::uint8_t { xfx, xfy, yfx, fy, fx, xf, yf };

struct OpDef {
    std::string name;
    std::uint32_t priority;  // 1..1200
    OpType type;
};

// An immutable operator table with a fluent builder. `with_op` returns a modified COPY, so a
// table can be shared and specialised without any caller observing another's changes.
class OperatorTable {
public:
    // The ISO standard operator set.
    [[nodiscard]] static auto standard() -> OperatorTable;
    // No operators at all: every term must then be written in canonical `f(a,b)` form.
    [[nodiscard]] static auto empty() -> OperatorTable;

    // A copy with this operator added, replacing any existing definition of the same name in
    // the same CLASS (prefix, infix or postfix) — which is how `-` can be both a yfx 500 infix
    // and an fy 200 prefix at the same time.
    [[nodiscard]] auto with_op(std::uint32_t priority, OpType type, std::string name) const
        -> OperatorTable;
    // A copy with the definition of `name` in the class of `type` removed. Priority 0 in an
    // `op/3` directive means the same thing, and that is how a program un-defines an operator.
    [[nodiscard]] auto without_op(std::string_view name, OpType type) const -> OperatorTable;

    [[nodiscard]] auto definitions() const -> const std::vector<OpDef>&;

private:
    std::vector<OpDef> defs_{};
};

// ---------------------------------------------------------------------------
// Reading.
// ---------------------------------------------------------------------------

// One term, which must be followed by an end token (`.`) or end of input. Trailing text after
// the term is a syntax_error rather than being ignored.
[[nodiscard]] auto parse_term(std::string_view source, const OperatorTable& ops) -> Result<Term>;
[[nodiscard]] auto parse_term(std::string_view source) -> Result<Term>;

// One clause: `Head.` or `Head :- Body.`, with the body split on `,/2` into the goal list that
// `Clause` holds. A body goal that is a control construct (`;`, `->`) stays one goal.
[[nodiscard]] auto parse_clause(std::string_view source, const OperatorTable& ops)
    -> Result<Clause>;
[[nodiscard]] auto parse_clause(std::string_view source) -> Result<Clause>;

// A whole program: every clause in the text, in order.
//
// A `:- op(Priority, Type, Name).` directive is HONOURED: it updates the table used for the
// REST of the file, which is what makes a program that defines its own notation readable. Any
// other `:- Directive.` is a syntax_error rather than being skipped — silently ignoring a
// directive would mean reading a program that is not the one that was written.
[[nodiscard]] auto parse_program(std::string_view source, const OperatorTable& ops)
    -> Result<Program>;
[[nodiscard]] auto parse_program(std::string_view source) -> Result<Program>;

// A query: `Goal1, Goal2, ... .`, with an optional leading `?-`, split into the goal list that
// `solve` takes.
[[nodiscard]] auto parse_query(std::string_view source, const OperatorTable& ops)
    -> Result<std::vector<Term>>;
[[nodiscard]] auto parse_query(std::string_view source) -> Result<std::vector<Term>>;

// ---------------------------------------------------------------------------
// Writing.
// ---------------------------------------------------------------------------

// `t` as Prolog source text, using `ops` for operator notation and quoting atoms as needed.
// The guarantee is a ROUND TRIP: parse_term(to_source(t, ops), ops) == t.
[[nodiscard]] auto to_source(const Term& t, const OperatorTable& ops) -> std::string;
[[nodiscard]] auto to_source(const Term& t) -> std::string;

// A clause as source text, terminated by `.` — `head :- g1, g2.` or `head.` for a fact.
[[nodiscard]] auto to_source(const Clause& c, const OperatorTable& ops) -> std::string;
[[nodiscard]] auto to_source(const Clause& c) -> std::string;

// Whether `name` can be written without quotes and read back as the same atom.
[[nodiscard]] auto atom_needs_quotes(std::string_view name) -> bool;
// `name` as an atom literal: bare when it can be, single-quoted and escaped otherwise.
[[nodiscard]] auto quote_atom(std::string_view name) -> std::string;

}  // namespace nimblecas::logic_parser

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::logic_parser {

using nimblecas::atom_of;
using nimblecas::compound_of;
using nimblecas::int_of;
using nimblecas::is_atom;
using nimblecas::is_compound;
using nimblecas::is_cons;
using nimblecas::is_int;
using nimblecas::is_nil;
using nimblecas::is_var;
using nimblecas::var_of;

namespace {

// ---------------------------------------------------------------------------
// Tokens.
// ---------------------------------------------------------------------------

enum class TokKind : std::uint8_t {
    atom,         // unquoted name, quoted 'name', symbolic (+, -->, \+), or solo (! ; [] {})
    var,          // Name or _Name or bare _
    integer,      // decimal, 0x.., 0o.., 0b.., 0'c
    open,         // (  -- with NO layout before it: a functor application f(
    open_space,   // (  -- preceded by layout, i.e. a grouping parenthesis
    close,        // )
    open_list,    // [
    close_list,   // ]
    open_curly,   // {
    close_curly,  // }
    comma,        // ,
    bar,          // |
    end,          // the clause terminator: '.' followed by layout or end-of-input
    eof,
};

struct PToken {
    TokKind kind{TokKind::eof};
    std::string text;         // atom/var name, or the raw digits for an integer
    std::int64_t value{0};    // integer value when kind == integer
    bool quoted{false};       // came from '...', so it is NEVER an operator
    std::size_t position{0};  // byte offset in the source, for diagnostics
    // Set when the literal's MAGNITUDE is exactly 2^63: representable only once negated.
    // Carrying it as a flag rather than as a value is what lets `-9223372036854775808` be read
    // without the tokenizer ever holding a value it cannot represent.
    bool needs_negation{false};
};

// ---------------------------------------------------------------------------
// Operator lookups (used by both the parser and the writer).
// ---------------------------------------------------------------------------

[[nodiscard]] auto is_infix_type(OpType t) -> bool {
    return t == OpType::xfx || t == OpType::xfy || t == OpType::yfx;
}
[[nodiscard]] auto is_prefix_type(OpType t) -> bool {
    return t == OpType::fy || t == OpType::fx;
}
[[nodiscard]] auto is_postfix_type(OpType t) -> bool {
    return t == OpType::xf || t == OpType::yf;
}

[[nodiscard]] auto lookup_infix(const OperatorTable& t, std::string_view name)
    -> std::optional<OpDef> {
    for (const OpDef& d : t.definitions()) {
        if (d.name == name && is_infix_type(d.type)) {
            return d;
        }
    }
    return std::nullopt;
}
[[nodiscard]] auto lookup_prefix(const OperatorTable& t, std::string_view name)
    -> std::optional<OpDef> {
    for (const OpDef& d : t.definitions()) {
        if (d.name == name && is_prefix_type(d.type)) {
            return d;
        }
    }
    return std::nullopt;
}
[[nodiscard]] auto lookup_postfix(const OperatorTable& t, std::string_view name)
    -> std::optional<OpDef> {
    for (const OpDef& d : t.definitions()) {
        if (d.name == name && is_postfix_type(d.type)) {
            return d;
        }
    }
    return std::nullopt;
}
[[nodiscard]] auto is_any_operator(const OperatorTable& t, std::string_view name) -> bool {
    return std::ranges::any_of(t.definitions(),
                               [name](const OpDef& d) -> bool { return d.name == name; });
}

// ---------------------------------------------------------------------------
// Tokenizer.
// ---------------------------------------------------------------------------

[[nodiscard]] auto lex_is_layout(char c) -> bool {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v';
}

[[nodiscard]] auto lex_is_digit(char c) -> bool {
    return c >= '0' && c <= '9';
}

[[nodiscard]] auto lex_is_hex_digit(char c) -> bool {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

[[nodiscard]] auto lex_is_octal_digit(char c) -> bool {
    return c >= '0' && c <= '7';
}

[[nodiscard]] auto lex_is_binary_digit(char c) -> bool {
    return c == '0' || c == '1';
}

[[nodiscard]] auto lex_is_lower(char c) -> bool {
    return c >= 'a' && c <= 'z';
}

[[nodiscard]] auto lex_is_upper(char c) -> bool {
    return c >= 'A' && c <= 'Z';
}

[[nodiscard]] auto lex_is_ident_char(char c) -> bool {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || (c == '_');
}

[[nodiscard]] auto lex_is_symbol_char(char c) -> bool {
    switch (c) {
        case '+': case '-': case '*': case '/':
        case '\\': case '^': case '<': case '>':
        case '=': case '~': case ':': case '.':
        case '?': case '@': case '#': case '&':
            return true;
        default:
            return false;
    }
}

// Encodes a Unicode code point into UTF-8 bytes to ensure correct representation in atom text.
[[nodiscard]] auto lex_append_codepoint(std::string& out, std::uint64_t cp) -> bool {
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
        return true;
    }
    if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        return true;
    }
    // Unicode surrogate code points U+D800..U+DFFF are invalid scalar values.
    if (cp >= 0xD800 && cp <= 0xDFFF) {
        return false;
    }
    if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        return true;
    }
    if (cp <= 0x10FFFF) {
        out.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        return true;
    }
    return false;
}

[[nodiscard]] auto tokenise(std::string_view src) -> Result<std::vector<PToken>> {
    std::vector<PToken> tokens;
    std::size_t pos = 0;
    std::size_t prev_token_end = 0;

    while (true) {
        // Layout and comments are skipped, tracking the gap between tokens.
        while (pos < src.size()) {
            char c = src[pos];
            if (lex_is_layout(c)) {
                ++pos;
                continue;
            }
            if (c == '%') {
                ++pos;
                while (pos < src.size() && src[pos] != '\n' && src[pos] != '\r') {
                    ++pos;
                }
                continue;
            }
            if (c == '/' && pos + 1 < src.size() && src[pos + 1] == '*') {
                pos += 2;
                std::size_t end_comment = src.find("*/", pos);
                if (end_comment == std::string_view::npos) {
                    return make_error<std::vector<PToken>>(MathError::syntax_error);
                }
                pos = end_comment + 2;
                continue;
            }
            break;
        }

        // When source text is exhausted, emit exactly one EOF token to signal completion.
        if (pos >= src.size()) {
            PToken eof_tok;
            eof_tok.kind = TokKind::eof;
            eof_tok.position = src.size();
            tokens.push_back(std::move(eof_tok));
            return tokens;
        }

        const std::size_t tok_start = pos;
        const char ch = src[pos];

        // Parentheses: an opening parenthesis preceded immediately by a token is functor application.
        if (ch == '(') {
            PToken tok;
            const bool preceded_by_token = (tok_start > 0 && tok_start == prev_token_end);
            tok.kind = preceded_by_token ? TokKind::open : TokKind::open_space;
            tok.text = "(";
            tok.position = tok_start;
            tokens.push_back(std::move(tok));
            pos += 1;
            prev_token_end = pos;
            continue;
        }
        if (ch == ')') {
            PToken tok;
            tok.kind = TokKind::close;
            tok.text = ")";
            tok.position = tok_start;
            tokens.push_back(std::move(tok));
            pos += 1;
            prev_token_end = pos;
            continue;
        }

        // Brackets and braces: empty pairs [] and {} collapse to distinct atom tokens.
        if (ch == '[') {
            if (pos + 1 < src.size() && src[pos + 1] == ']') {
                PToken tok;
                tok.kind = TokKind::atom;
                tok.text = "[]";
                tok.position = tok_start;
                tokens.push_back(std::move(tok));
                pos += 2;
                prev_token_end = pos;
                continue;
            }
            PToken tok;
            tok.kind = TokKind::open_list;
            tok.text = "[";
            tok.position = tok_start;
            tokens.push_back(std::move(tok));
            pos += 1;
            prev_token_end = pos;
            continue;
        }
        if (ch == ']') {
            PToken tok;
            tok.kind = TokKind::close_list;
            tok.text = "]";
            tok.position = tok_start;
            tokens.push_back(std::move(tok));
            pos += 1;
            prev_token_end = pos;
            continue;
        }
        if (ch == '{') {
            if (pos + 1 < src.size() && src[pos + 1] == '}') {
                PToken tok;
                tok.kind = TokKind::atom;
                tok.text = "{}";
                tok.position = tok_start;
                tokens.push_back(std::move(tok));
                pos += 2;
                prev_token_end = pos;
                continue;
            }
            PToken tok;
            tok.kind = TokKind::open_curly;
            tok.text = "{";
            tok.position = tok_start;
            tokens.push_back(std::move(tok));
            pos += 1;
            prev_token_end = pos;
            continue;
        }
        if (ch == '}') {
            PToken tok;
            tok.kind = TokKind::close_curly;
            tok.text = "}";
            tok.position = tok_start;
            tokens.push_back(std::move(tok));
            pos += 1;
            prev_token_end = pos;
            continue;
        }

        // Solo tokens: punctuation characters that stand alone as atomic elements.
        if (ch == ',') {
            PToken tok;
            tok.kind = TokKind::comma;
            tok.text = ",";
            tok.position = tok_start;
            tokens.push_back(std::move(tok));
            pos += 1;
            prev_token_end = pos;
            continue;
        }
        if (ch == '|') {
            PToken tok;
            tok.kind = TokKind::bar;
            tok.text = "|";
            tok.position = tok_start;
            tokens.push_back(std::move(tok));
            pos += 1;
            prev_token_end = pos;
            continue;
        }
        if (ch == '!') {
            PToken tok;
            tok.kind = TokKind::atom;
            tok.text = "!";
            tok.position = tok_start;
            tokens.push_back(std::move(tok));
            pos += 1;
            prev_token_end = pos;
            continue;
        }
        if (ch == ';') {
            PToken tok;
            tok.kind = TokKind::atom;
            tok.text = ";";
            tok.position = tok_start;
            tokens.push_back(std::move(tok));
            pos += 1;
            prev_token_end = pos;
            continue;
        }

        // Quoted atoms: delimited by single quotes with character and escape processing.
        if (ch == '\'') {
            ++pos;
            std::string decoded;
            bool terminated = false;
            while (pos < src.size()) {
                char qc = src[pos];
                if (qc == '\n' || qc == '\r') {
                    return make_error<std::vector<PToken>>(MathError::syntax_error);
                }
                if (qc == '\'') {
                    if (pos + 1 < src.size() && src[pos + 1] == '\'') {
                        decoded.push_back('\'');
                        pos += 2;
                        continue;
                    }
                    ++pos;
                    terminated = true;
                    break;
                }
                if (qc == '\\') {
                    ++pos;
                    if (pos >= src.size()) {
                        return make_error<std::vector<PToken>>(MathError::syntax_error);
                    }
                    char esc = src[pos];
                    if (esc == '\r' && pos + 1 < src.size() && src[pos + 1] == '\n') {
                        pos += 2;
                        continue;
                    }
                    if (esc == '\n' || esc == '\r') {
                        pos += 1;
                        continue;
                    }
                    if (esc == '\\' || esc == '\'' || esc == '\"' || esc == '`') {
                        decoded.push_back(esc);
                        pos += 1;
                        continue;
                    }
                    if (esc == 'n') { decoded.push_back('\n'); pos += 1; continue; }
                    if (esc == 't') { decoded.push_back('\t'); pos += 1; continue; }
                    if (esc == 'r') { decoded.push_back('\r'); pos += 1; continue; }
                    if (esc == 'a') { decoded.push_back('\a'); pos += 1; continue; }
                    if (esc == 'b') { decoded.push_back('\b'); pos += 1; continue; }
                    if (esc == 'f') { decoded.push_back('\f'); pos += 1; continue; }
                    if (esc == 'v') { decoded.push_back('\v'); pos += 1; continue; }
                    if (esc == 'x' || esc == 'X') {
                        pos += 1;
                        std::size_t hex_start = pos;
                        while (pos < src.size() && lex_is_hex_digit(src[pos])) {
                            ++pos;
                        }
                        if (pos == hex_start || pos >= src.size() || src[pos] != '\\') {
                            return make_error<std::vector<PToken>>(MathError::syntax_error);
                        }
                        std::uint64_t cp = 0;
                        auto [ptr, ec] = std::from_chars(src.data() + hex_start, src.data() + pos, cp, 16);
                        if (ec != std::errc{} || !lex_append_codepoint(decoded, cp)) {
                            return make_error<std::vector<PToken>>(MathError::syntax_error);
                        }
                        pos += 1;
                        continue;
                    }
                    if (lex_is_octal_digit(esc)) {
                        std::size_t oct_start = pos;
                        while (pos < src.size() && lex_is_octal_digit(src[pos])) {
                            ++pos;
                        }
                        if (pos < src.size() && src[pos] == '\\') {
                            std::uint64_t cp = 0;
                            auto [ptr, ec] = std::from_chars(src.data() + oct_start, src.data() + pos, cp, 8);
                            if (ec != std::errc{} || !lex_append_codepoint(decoded, cp)) {
                                return make_error<std::vector<PToken>>(MathError::syntax_error);
                            }
                            pos += 1;
                            continue;
                        }
                        if (esc == '0' && pos == oct_start + 1) {
                            decoded.push_back('\0');
                            continue;
                        }
                        return make_error<std::vector<PToken>>(MathError::syntax_error);
                    }
                    return make_error<std::vector<PToken>>(MathError::syntax_error);
                }
                decoded.push_back(qc);
                ++pos;
            }
            if (!terminated) {
                return make_error<std::vector<PToken>>(MathError::syntax_error);
            }
            PToken tok;
            tok.kind = TokKind::atom;
            tok.text = std::move(decoded);
            tok.quoted = true;
            tok.position = tok_start;
            tokens.push_back(std::move(tok));
            prev_token_end = pos;
            continue;
        }

        // Numbers: decimal, base-prefixed integers, and character code literals.
        if (lex_is_digit(ch)) {
            // Character code literal: 0'c
            if (ch == '0' && pos + 1 < src.size() && src[pos + 1] == '\'') {
                pos += 2;
                if (pos >= src.size()) {
                    return make_error<std::vector<PToken>>(MathError::syntax_error);
                }
                std::int64_t code_val = 0;
                char cc = src[pos];
                if (cc == '\'') {
                    if (pos + 1 < src.size() && src[pos + 1] == '\'') {
                        code_val = 39;
                        pos += 2;
                    } else {
                        code_val = 39;
                        pos += 1;
                    }
                } else if (cc == '\\') {
                    pos += 1;
                    if (pos >= src.size()) {
                        return make_error<std::vector<PToken>>(MathError::syntax_error);
                    }
                    char esc = src[pos];
                    if (esc == '\\' || esc == '\'' || esc == '\"' || esc == '`') {
                        code_val = static_cast<unsigned char>(esc);
                        pos += 1;
                    } else if (esc == 'n') { code_val = 10; pos += 1; }
                    else if (esc == 't') { code_val = 9; pos += 1; }
                    else if (esc == 'r') { code_val = 13; pos += 1; }
                    else if (esc == 'a') { code_val = 7; pos += 1; }
                    else if (esc == 'b') { code_val = 8; pos += 1; }
                    else if (esc == 'f') { code_val = 12; pos += 1; }
                    else if (esc == 'v') { code_val = 11; pos += 1; }
                    else if (esc == 'x' || esc == 'X') {
                        pos += 1;
                        std::size_t hex_start = pos;
                        while (pos < src.size() && lex_is_hex_digit(src[pos])) {
                            ++pos;
                        }
                        if (pos == hex_start || pos >= src.size() || src[pos] != '\\') {
                            return make_error<std::vector<PToken>>(MathError::syntax_error);
                        }
                        std::int64_t cp = 0;
                        auto [ptr, ec] = std::from_chars(src.data() + hex_start, src.data() + pos, cp, 16);
                        if (ec == std::errc::result_out_of_range) {
                            return make_error<std::vector<PToken>>(MathError::overflow);
                        }
                        if (ec != std::errc{}) {
                            return make_error<std::vector<PToken>>(MathError::syntax_error);
                        }
                        code_val = cp;
                        pos += 1;
                    } else if (lex_is_octal_digit(esc)) {
                        std::size_t oct_start = pos;
                        while (pos < src.size() && lex_is_octal_digit(src[pos])) {
                            ++pos;
                        }
                        if (pos < src.size() && src[pos] == '\\') {
                            std::int64_t cp = 0;
                            auto [ptr, ec] = std::from_chars(src.data() + oct_start, src.data() + pos, cp, 8);
                            if (ec == std::errc::result_out_of_range) {
                                return make_error<std::vector<PToken>>(MathError::overflow);
                            }
                            if (ec != std::errc{}) {
                                return make_error<std::vector<PToken>>(MathError::syntax_error);
                            }
                            code_val = cp;
                            pos += 1;
                        } else if (esc == '0' && pos == oct_start + 1) {
                            code_val = 0;
                        } else {
                            return make_error<std::vector<PToken>>(MathError::syntax_error);
                        }
                    } else {
                        return make_error<std::vector<PToken>>(MathError::syntax_error);
                    }
                } else if (cc == '\n' || cc == '\r') {
                    return make_error<std::vector<PToken>>(MathError::syntax_error);
                } else {
                    code_val = static_cast<unsigned char>(cc);
                    pos += 1;
                }
                PToken tok;
                tok.kind = TokKind::integer;
                tok.text = std::string(src.substr(tok_start, pos - tok_start));
                tok.value = code_val;
                tok.position = tok_start;
                tokens.push_back(std::move(tok));
                prev_token_end = pos;
                continue;
            }

            // Hex integer: 0x... / 0X...
            if (ch == '0' && pos + 1 < src.size() && (src[pos + 1] == 'x' || src[pos + 1] == 'X')) {
                pos += 2;
                std::size_t hex_start = pos;
                while (pos < src.size() && lex_is_hex_digit(src[pos])) {
                    ++pos;
                }
                if (pos == hex_start) {
                    return make_error<std::vector<PToken>>(MathError::syntax_error);
                }
                if (pos < src.size() && lex_is_ident_char(src[pos])) {
                    return make_error<std::vector<PToken>>(MathError::syntax_error);
                }
                std::int64_t val = 0;
                auto [ptr, ec] = std::from_chars(src.data() + hex_start, src.data() + pos, val, 16);
                if (ec == std::errc::result_out_of_range) {
                    return make_error<std::vector<PToken>>(MathError::overflow);
                }
                if (ec != std::errc{}) {
                    return make_error<std::vector<PToken>>(MathError::syntax_error);
                }
                PToken tok;
                tok.kind = TokKind::integer;
                tok.text = std::string(src.substr(tok_start, pos - tok_start));
                tok.value = val;
                tok.position = tok_start;
                tokens.push_back(std::move(tok));
                prev_token_end = pos;
                continue;
            }

            // Octal integer: 0o... / 0O...
            if (ch == '0' && pos + 1 < src.size() && (src[pos + 1] == 'o' || src[pos + 1] == 'O')) {
                pos += 2;
                std::size_t oct_start = pos;
                while (pos < src.size() && lex_is_octal_digit(src[pos])) {
                    ++pos;
                }
                if (pos == oct_start) {
                    return make_error<std::vector<PToken>>(MathError::syntax_error);
                }
                if (pos < src.size() && lex_is_ident_char(src[pos])) {
                    return make_error<std::vector<PToken>>(MathError::syntax_error);
                }
                std::int64_t val = 0;
                auto [ptr, ec] = std::from_chars(src.data() + oct_start, src.data() + pos, val, 8);
                if (ec == std::errc::result_out_of_range) {
                    return make_error<std::vector<PToken>>(MathError::overflow);
                }
                if (ec != std::errc{}) {
                    return make_error<std::vector<PToken>>(MathError::syntax_error);
                }
                PToken tok;
                tok.kind = TokKind::integer;
                tok.text = std::string(src.substr(tok_start, pos - tok_start));
                tok.value = val;
                tok.position = tok_start;
                tokens.push_back(std::move(tok));
                prev_token_end = pos;
                continue;
            }

            // Binary integer: 0b... / 0B...
            if (ch == '0' && pos + 1 < src.size() && (src[pos + 1] == 'b' || src[pos + 1] == 'B')) {
                pos += 2;
                std::size_t bin_start = pos;
                while (pos < src.size() && lex_is_binary_digit(src[pos])) {
                    ++pos;
                }
                if (pos == bin_start) {
                    return make_error<std::vector<PToken>>(MathError::syntax_error);
                }
                if (pos < src.size() && lex_is_ident_char(src[pos])) {
                    return make_error<std::vector<PToken>>(MathError::syntax_error);
                }
                std::int64_t val = 0;
                auto [ptr, ec] = std::from_chars(src.data() + bin_start, src.data() + pos, val, 2);
                if (ec == std::errc::result_out_of_range) {
                    return make_error<std::vector<PToken>>(MathError::overflow);
                }
                if (ec != std::errc{}) {
                    return make_error<std::vector<PToken>>(MathError::syntax_error);
                }
                PToken tok;
                tok.kind = TokKind::integer;
                tok.text = std::string(src.substr(tok_start, pos - tok_start));
                tok.value = val;
                tok.position = tok_start;
                tokens.push_back(std::move(tok));
                prev_token_end = pos;
                continue;
            }

            // Decimal integer: run of decimal digits. Underscores are not permitted.
            while (pos < src.size() && lex_is_digit(src[pos])) {
                ++pos;
            }
            // Reject floating-point literals with not_implemented rather than misparsing.
            if (pos + 1 < src.size() && src[pos] == '.' && lex_is_digit(src[pos + 1])) {
                return make_error<std::vector<PToken>>(MathError::not_implemented);
            }
            if (pos < src.size() && lex_is_ident_char(src[pos])) {
                return make_error<std::vector<PToken>>(MathError::syntax_error);
            }
            std::int64_t val = 0;
            auto [ptr, ec] = std::from_chars(src.data() + tok_start, src.data() + pos, val, 10);
            if (ec == std::errc::result_out_of_range) {
                // 2^63 has no positive representation but IS the magnitude of the most
                // negative integer. Emitting it with a flag lets the parser build INT64_MIN
                // for `-9223372036854775808` while still rejecting the bare literal, which is
                // the only way to read that value without ever holding it unnegated.
                const std::string_view digits = src.substr(tok_start, pos - tok_start);
                std::string_view trimmed = digits;
                while (trimmed.size() > 1 && trimmed.front() == '0') {
                    trimmed.remove_prefix(1);
                }
                if (trimmed != "9223372036854775808") {
                    return make_error<std::vector<PToken>>(MathError::overflow);
                }
                PToken big;
                big.kind = TokKind::integer;
                big.text = std::string(digits);
                big.value = 0;
                big.needs_negation = true;
                big.position = tok_start;
                tokens.push_back(std::move(big));
                prev_token_end = pos;
                continue;
            }
            if (ec != std::errc{}) {
                return make_error<std::vector<PToken>>(MathError::syntax_error);
            }
            PToken tok;
            tok.kind = TokKind::integer;
            tok.text = std::string(src.substr(tok_start, pos - tok_start));
            tok.value = val;
            tok.position = tok_start;
            tokens.push_back(std::move(tok));
            prev_token_end = pos;
            continue;
        }

        // Variables: uppercase letter or underscore followed by alphanumeric/underscore run.
        if (lex_is_upper(ch) || ch == '_') {
            while (pos < src.size() && lex_is_ident_char(src[pos])) {
                ++pos;
            }
            PToken tok;
            tok.kind = TokKind::var;
            tok.text = std::string(src.substr(tok_start, pos - tok_start));
            tok.position = tok_start;
            tokens.push_back(std::move(tok));
            prev_token_end = pos;
            continue;
        }

        // Unquoted atoms: lowercase letter followed by alphanumeric/underscore run.
        if (lex_is_lower(ch)) {
            while (pos < src.size() && lex_is_ident_char(src[pos])) {
                ++pos;
            }
            PToken tok;
            tok.kind = TokKind::atom;
            tok.text = std::string(src.substr(tok_start, pos - tok_start));
            tok.position = tok_start;
            tokens.push_back(std::move(tok));
            prev_token_end = pos;
            continue;
        }

        // The clause terminator: a dot followed by layout, a comment, or end of input.
        if (ch == '.') {
            const bool followed_by_eof = (pos + 1 == src.size());
            const bool followed_by_layout = (pos + 1 < src.size() && lex_is_layout(src[pos + 1]));
            const bool followed_by_percent = (pos + 1 < src.size() && src[pos + 1] == '%');
            const bool followed_by_block_comment = (pos + 2 < src.size() && src[pos + 1] == '/' && src[pos + 2] == '*');
            if (followed_by_eof || followed_by_layout || followed_by_percent || followed_by_block_comment) {
                PToken tok;
                tok.kind = TokKind::end;
                tok.text = ".";
                tok.position = tok_start;
                tokens.push_back(std::move(tok));
                pos += 1;
                prev_token_end = pos;
                continue;
            }
        }

        // Symbolic atoms: maximal run of symbol characters, stopping before comment delimiters.
        if (lex_is_symbol_char(ch)) {
            while (pos < src.size() && lex_is_symbol_char(src[pos])) {
                if (src[pos] == '/' && pos + 1 < src.size() && src[pos + 1] == '*') {
                    break;
                }
                ++pos;
            }
            PToken tok;
            tok.kind = TokKind::atom;
            tok.text = std::string(src.substr(tok_start, pos - tok_start));
            tok.position = tok_start;
            tokens.push_back(std::move(tok));
            prev_token_end = pos;
            continue;
        }

        // Any character not matching known lexical rules causes an immediate syntax error.
        return make_error<std::vector<PToken>>(MathError::syntax_error);
    }
}

// ---------------------------------------------------------------------------
// Operator table data, atom quoting, and the writer.
// ---------------------------------------------------------------------------

// ISO standard operator set definitions ordered from lowest precedence to highest.
[[nodiscard]] auto standard_operator_defs() -> std::vector<OpDef> {
    return {
        OpDef{":-", 1200, OpType::xfx},
        OpDef{"-->", 1200, OpType::xfx},
        OpDef{":-", 1200, OpType::fx},
        OpDef{"?-", 1200, OpType::fx},
        OpDef{";", 1100, OpType::xfy},
        OpDef{"|", 1100, OpType::xfy},
        OpDef{"->", 1050, OpType::xfy},
        OpDef{"*->", 1050, OpType::xfy},
        OpDef{",", 1000, OpType::xfy},
        OpDef{":=", 990, OpType::xfx},
        OpDef{"\\+", 900, OpType::fy},
        OpDef{"=", 700, OpType::xfx},
        OpDef{"\\=", 700, OpType::xfx},
        OpDef{"==", 700, OpType::xfx},
        OpDef{"\\==", 700, OpType::xfx},
        OpDef{"@<", 700, OpType::xfx},
        OpDef{"@>", 700, OpType::xfx},
        OpDef{"@=<", 700, OpType::xfx},
        OpDef{"@>=", 700, OpType::xfx},
        OpDef{"=..", 700, OpType::xfx},
        OpDef{"is", 700, OpType::xfx},
        OpDef{"=:=", 700, OpType::xfx},
        OpDef{"=\\=", 700, OpType::xfx},
        OpDef{"<", 700, OpType::xfx},
        OpDef{">", 700, OpType::xfx},
        OpDef{"=<", 700, OpType::xfx},
        OpDef{">=", 700, OpType::xfx},
        OpDef{"as", 700, OpType::xfx},
        OpDef{">:<", 700, OpType::xfx},
        OpDef{":<", 700, OpType::xfx},
        OpDef{":", 600, OpType::xfy},
        OpDef{"+", 500, OpType::yfx},
        OpDef{"-", 500, OpType::yfx},
        OpDef{"/\\", 500, OpType::yfx},
        OpDef{"\\/", 500, OpType::yfx},
        OpDef{"xor", 500, OpType::yfx},
        OpDef{"?", 500, OpType::fx},
        OpDef{"*", 400, OpType::yfx},
        OpDef{"/", 400, OpType::yfx},
        OpDef{"//", 400, OpType::yfx},
        OpDef{"rem", 400, OpType::yfx},
        OpDef{"mod", 400, OpType::yfx},
        OpDef{"div", 400, OpType::yfx},
        OpDef{"<<", 400, OpType::yfx},
        OpDef{">>", 400, OpType::yfx},
        OpDef{"divmod", 400, OpType::yfx},
        OpDef{"**", 200, OpType::xfx},
        OpDef{"^", 200, OpType::xfy},
        OpDef{"-", 200, OpType::fy},
        OpDef{"+", 200, OpType::fy},
        OpDef{"\\", 200, OpType::fy},
        OpDef{".", 100, OpType::yfx},
        OpDef{"$", 1, OpType::fx},
    };
}

// We implement integer formatting directly with string building to prevent raw pointer exposure.
[[nodiscard]] auto wr_format_uint(std::uint64_t val) -> std::string {
    if (val == 0) {
        return "0";
    }
    std::string s;
    while (val > 0) {
        s.push_back(static_cast<char>('0' + (val % 10)));
        val /= 10;
    }
    std::ranges::reverse(s);
    return s;
}

// The most negative signed 64-bit integer cannot be negated in two's complement without signed overflow,
// so we compute its magnitude in unsigned arithmetic to avoid undefined behavior.
[[nodiscard]] auto wr_format_int(std::int64_t val) -> std::string {
    if (val == 0) {
        return "0";
    }
    std::string s;
    bool negative = false;
    std::uint64_t uval = 0;
    if (val < 0) {
        negative = true;
        uval = static_cast<std::uint64_t>(-(val + 1)) + 1;
    } else {
        uval = static_cast<std::uint64_t>(val);
    }
    while (uval > 0) {
        s.push_back(static_cast<char>('0' + (uval % 10)));
        uval /= 10;
    }
    if (negative) {
        s.push_back('-');
    }
    std::ranges::reverse(s);
    return s;
}

[[nodiscard]] auto wr_hex_char(unsigned int nibble) -> char {
    constexpr std::string_view digits = "0123456789abcdef";
    return digits[nibble & 0x0f];
}

// The solo atoms !, ;, [], and {} are self-delimiting lexical forms defined by the ISO standard.
[[nodiscard]] auto wr_is_solo_atom(std::string_view name) -> bool {
    return name == "!" || name == ";" || name == "[]" || name == "{}";
}

// Symbolic atoms are formed from maximal runs of these 16 characters matching the lexer specification.
[[nodiscard]] auto wr_is_symbol_char(char c) -> bool {
    switch (c) {
        case '+': case '-': case '*': case '/':
        case '\\': case '^': case '<': case '>':
        case '=': case '~': case ':': case '.':
        case '?': case '@': case '#': case '&':
            return true;
        default:
            return false;
    }
}

[[nodiscard]] auto wr_is_lower_letter(char c) -> bool {
    return c >= 'a' && c <= 'z';
}

[[nodiscard]] auto wr_is_ident_char(char c) -> bool {
    return (c >= 'a' && c <= 'z') ||
           (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') ||
           c == '_';
}

[[nodiscard]] auto wr_atom_needs_quotes(std::string_view name) -> bool {
    // Empty atoms cannot be bare in Prolog and must always be quoted as ''.
    if (name.empty()) {
        return true;
    }
    if (wr_is_solo_atom(name)) {
        return false;
    }
    if (wr_is_lower_letter(name.front())) {
        bool all_ident = true;
        for (std::size_t i = 1; i < name.size(); ++i) {
            if (!wr_is_ident_char(name[i])) {
                all_ident = false;
                break;
            }
        }
        if (all_ident) {
            return false;
        }
    }
    bool all_symbol = true;
    for (char c : name) {
        if (!wr_is_symbol_char(c)) {
            all_symbol = false;
            break;
        }
    }
    if (all_symbol) {
        // Two symbolic atoms cannot be written bare even though every character is a symbol
        // character: a lone `.` becomes the clause terminator once layout follows it, and
        // anything starting `/*` opens a block comment. Both re-read as something else.
        if (name == "." || name.starts_with("/*")) {
            return true;
        }
        return false;
    }
    return true;
}

[[nodiscard]] auto wr_quote_atom(std::string_view name) -> std::string {
    if (!wr_atom_needs_quotes(name)) {
        return std::string(name);
    }
    std::string out;
    out.push_back('\'');
    for (char c : name) {
        const auto uc = static_cast<unsigned char>(c);
        if (c == '\\') {
            out.append("\\\\");
        } else if (c == '\'') {
            out.append("\\'");
        } else if (c == '\n') {
            out.append("\\n");
        } else if (c == '\t') {
            out.append("\\t");
        } else if (c == '\r') {
            out.append("\\r");
        } else if (uc < 0x20) {
            // Control bytes below 0x20 use the ISO Prolog hex escape syntax \x<hex>\.
            out.append("\\x");
            out.push_back(wr_hex_char(static_cast<unsigned int>(uc >> 4)));
            out.push_back(wr_hex_char(static_cast<unsigned int>(uc & 0x0f)));
            out.push_back('\\');
        } else {
            // Bytes >= 0x80 (UTF-8) and printable ASCII characters pass through unchanged.
            out.push_back(c);
        }
    }
    out.push_back('\'');
    return out;
}

[[nodiscard]] auto wr_is_alpha_operator(std::string_view name) -> bool {
    if (name.empty()) {
        return false;
    }
    return wr_is_lower_letter(name.front());
}

[[nodiscard]] auto wr_needs_space(char a, char b) -> bool {
    // Adjacent symbol characters from different tokens would merge under maximal munch if unseparated.
    if (wr_is_symbol_char(a) && wr_is_symbol_char(b)) {
        return true;
    }
    // Adjacent alphanumeric characters from different tokens would merge into a single identifier.
    if (wr_is_ident_char(a) && wr_is_ident_char(b)) {
        return true;
    }
    return false;
}

[[nodiscard]] auto wr_term_priority(const Term& t, const OperatorTable& ops) -> std::uint32_t {
    if (is_compound(t)) {
        const auto& comp = compound_of(t);
        // List brackets and curly bracket terms delimit themselves and have priority 0 in Prolog.
        if (is_cons(t) || (comp.functor == "{}" && comp.args.size() == 1)) {
            return 0;
        }
        if (comp.args.size() == 2) {
            auto op = lookup_infix(ops, comp.functor);
            if (op.has_value()) {
                return op->priority;
            }
        } else if (comp.args.size() == 1) {
            auto op = lookup_prefix(ops, comp.functor);
            if (op.has_value()) {
                return op->priority;
            }
            auto post = lookup_postfix(ops, comp.functor);
            if (post.has_value()) {
                return post->priority;
            }
        }
        // Canonical compound expressions f(args...) have priority 0.
        return 0;
    }
    return 0;
}

[[nodiscard]] auto wr_write_var(const VarNode& var) -> std::string {
    if (var.generation == 0) {
        return var.name;
    }
    return "_" + var.name + "_" + wr_format_uint(var.generation);
}

[[nodiscard]] auto wr_write_int(const IntNode& num, bool is_operator_operand) -> std::string {
    std::string s = wr_format_int(num.value);
    // A negative integer appearing as an operator operand must be parenthesised
    // so it does not re-read as a prefix negation applied to an integer.
    if (num.value < 0 && is_operator_operand) {
        return "(" + s + ")";
    }
    return s;
}

[[nodiscard]] auto wr_write_atom(const AtomNode& atom, const OperatorTable& ops, bool quoted,
                                 bool is_operator_operand) -> std::string {
    std::string s = quoted ? wr_quote_atom(atom.name) : atom.name;
    // An atom that is also an operator name must be parenthesised when used as an operand
    // so the Pratt parser treats it as an atomic primary rather than an operator application.
    if (is_operator_operand && is_any_operator(ops, atom.name)) {
        return "(" + s + ")";
    }
    return s;
}

// Forward declaration of recursive worker for mutually recursive helpers.
[[nodiscard]] auto wr_to_source_internal(const Term& t, const OperatorTable& ops, bool quoted,
                                         std::size_t depth, std::size_t max_depth,
                                         std::uint32_t allowed_max_priority,
                                         bool is_operator_operand) -> std::string;

[[nodiscard]] auto wr_write_list(const Term& t, const OperatorTable& ops, bool quoted,
                                 std::size_t depth, std::size_t max_depth) -> std::string {
    std::string out = "[";
    Term curr = t;
    bool first = true;
    std::size_t spine_depth = depth;

    while (is_cons(curr)) {
        if (!first) {
            out.push_back(',');
        }
        first = false;

        const auto& comp = compound_of(curr);
        // Arguments of list cells are written at priority 999 so that ','/2 terms are parenthesised.
        out.append(wr_to_source_internal(comp.args[0], ops, quoted, spine_depth + 1, max_depth, 999, true));

        const auto& tail = comp.args[1];
        if (is_nil(tail)) {
            out.push_back(']');
            return out;
        }

        spine_depth += 1;
        if (spine_depth > max_depth) {
            // Guard against cyclic lists by truncating when the recursion budget is exhausted.
            out.append("|...]");
            return out;
        }

        if (is_cons(tail)) {
            curr = tail;
        } else {
            // Improper lists use bar notation to display the final tail term.
            out.push_back('|');
            out.append(wr_to_source_internal(tail, ops, quoted, spine_depth + 1, max_depth, 999, true));
            out.push_back(']');
            return out;
        }
    }

    out.push_back(']');
    return out;
}

[[nodiscard]] auto wr_write_curly(const CompoundNode& comp, const OperatorTable& ops, bool quoted,
                                  std::size_t depth, std::size_t max_depth) -> std::string {
    std::string out = "{";
    // The operand inside curly brackets is parsed at priority 1200 according to ISO Prolog.
    out.append(wr_to_source_internal(comp.args[0], ops, quoted, depth + 1, max_depth, 1200, false));
    out.push_back('}');
    return out;
}

[[nodiscard]] auto wr_write_canonical(const CompoundNode& comp, const OperatorTable& ops, bool quoted,
                                      std::size_t depth, std::size_t max_depth) -> std::string {
    std::string out;
    if (quoted) {
        out.append(wr_quote_atom(comp.functor));
    } else {
        out.append(comp.functor);
    }
    // Functor application has no whitespace before the opening parenthesis to distinguish f(x) from f (x).
    out.push_back('(');
    bool first = true;
    for (const auto& arg : comp.args) {
        if (!first) {
            out.push_back(',');
        }
        first = false;
        // Each argument is written at priority 999 so that ','/2 terms are parenthesised.
        out.append(wr_to_source_internal(arg, ops, quoted, depth + 1, max_depth, 999, true));
    }
    out.push_back(')');
    return out;
}

[[nodiscard]] auto wr_to_source_internal(const Term& t, const OperatorTable& ops, bool quoted,
                                         std::size_t depth, std::size_t max_depth,
                                         std::uint32_t allowed_max_priority,
                                         bool is_operator_operand) -> std::string {
    if (depth > max_depth) {
        return "...";
    }

    // A term whose priority exceeds the parent's allowed allowance must be wrapped in parentheses,
    // which resets the priority allowance to 1200 inside.
    const std::uint32_t my_priority = wr_term_priority(t, ops);
    if (my_priority > allowed_max_priority) {
        return "(" + wr_to_source_internal(t, ops, quoted, depth, max_depth, 1200, false) + ")";
    }

    if (is_var(t)) {
        return wr_write_var(var_of(t));
    }

    if (is_int(t)) {
        return wr_write_int(int_of(t), is_operator_operand);
    }

    if (is_atom(t)) {
        return wr_write_atom(atom_of(t), ops, quoted, is_operator_operand);
    }

    if (is_compound(t)) {
        const auto& comp = compound_of(t);

        // Rule 4: '.'/2 spines write in list notation.
        if (is_cons(t)) {
            return wr_write_list(t, ops, quoted, depth, max_depth);
        }

        // Rule 5: {}/1 writes as {X}.
        if (comp.functor == "{}" && comp.args.size() == 1) {
            return wr_write_curly(comp, ops, quoted, depth, max_depth);
        }

        // Rule 6: Infix operator of arity 2.
        if (comp.args.size() == 2) {
            auto op_opt = lookup_infix(ops, comp.functor);
            if (op_opt.has_value()) {
                const auto& op_def = *op_opt;
                const std::uint32_t P = op_def.priority;
                std::uint32_t left_max = 0;
                std::uint32_t right_max = 0;
                switch (op_def.type) {
                    case OpType::xfx:
                        left_max = (P > 0) ? P - 1 : 0;
                        right_max = (P > 0) ? P - 1 : 0;
                        break;
                    case OpType::xfy:
                        left_max = (P > 0) ? P - 1 : 0;
                        right_max = P;
                        break;
                    case OpType::yfx:
                        left_max = P;
                        right_max = (P > 0) ? P - 1 : 0;
                        break;
                    default:
                        left_max = (P > 0) ? P - 1 : 0;
                        right_max = (P > 0) ? P - 1 : 0;
                        break;
                }

                std::string left_str = wr_to_source_internal(comp.args[0], ops, quoted, depth + 1, max_depth, left_max, true);
                std::string right_str = wr_to_source_internal(comp.args[1], ops, quoted, depth + 1, max_depth, right_max, true);

                if (comp.functor == ",") {
                    // Rule 6: Write ',' without a leading space: a,b.
                    return left_str + "," + right_str;
                }
                if (comp.functor == ":-" || comp.functor == "-->") {
                    return left_str + " " + comp.functor + " " + right_str;
                }
                if (wr_is_alpha_operator(comp.functor)) {
                    return left_str + " " + comp.functor + " " + right_str;
                }
                if (comp.functor == "-" && is_int(comp.args[1]) && int_of(comp.args[1]).value < 0) {
                    // Line 110: 1 - (-2). Spaces around '-' are emitted when the right operand is a parenthesised negative integer.
                    return left_str + " - " + right_str;
                }

                std::string res = left_str;
                if (!res.empty() && !comp.functor.empty() && wr_needs_space(res.back(), comp.functor.front())) {
                    res.push_back(' ');
                }
                res.append(comp.functor);
                if (!comp.functor.empty() && !right_str.empty() && wr_needs_space(comp.functor.back(), right_str.front())) {
                    res.push_back(' ');
                }
                res.append(right_str);
                return res;
            }
        }

        // Rule 7: Prefix operator of arity 1.
        if (comp.args.size() == 1 && !wr_atom_needs_quotes(comp.functor)) {
            auto op_opt = lookup_prefix(ops, comp.functor);
            if (op_opt.has_value()) {
                const auto& op_def = *op_opt;
                const std::uint32_t P = op_def.priority;
                std::uint32_t arg_max = (op_def.type == OpType::fx && P > 0) ? P - 1 : P;

                std::string arg_str = wr_to_source_internal(comp.args[0], ops, quoted, depth + 1, max_depth, arg_max, true);


                // `-(2)` must NOT be written `-2`: the reader folds a sign directly into a following


                // integer literal, so `-2` would read back as the INTEGER minus two rather than as a


                // compound. Parenthesising the argument keeps the two distinguishable.


                const bool sign_folds =


                    (comp.functor == "-" || comp.functor == "+") && is_int(comp.args[0]);


                if (sign_folds) {


                    arg_str = "(" + arg_str + ")";


                }

                std::string res = comp.functor;
                // A space is mandatory after a prefix operator when followed by an opening parenthesis,
                // because layout determines whether the following parenthesis opens a grouped term or a canonical argument list.
                bool needs_space = wr_is_alpha_operator(comp.functor) ||
                                   comp.functor == "\\+" ||
                                   (!arg_str.empty() && arg_str.front() == '(') ||
                                   (!arg_str.empty() && wr_needs_space(comp.functor.back(), arg_str.front()));
                if (needs_space) {
                    res.push_back(' ');
                }
                res.append(arg_str);
                return res;
            }

            // Postfix operator of arity 1.
            auto post_opt = lookup_postfix(ops, comp.functor);
            if (post_opt.has_value()) {
                const auto& op_def = *post_opt;
                const std::uint32_t P = op_def.priority;
                std::uint32_t arg_max = (op_def.type == OpType::xf && P > 0) ? P - 1 : P;

                std::string arg_str = wr_to_source_internal(comp.args[0], ops, quoted, depth + 1, max_depth, arg_max, true);

                std::string res = arg_str;
                bool needs_space = wr_is_alpha_operator(comp.functor) ||
                                   (!res.empty() && wr_needs_space(res.back(), comp.functor.front()));
                if (needs_space) {
                    res.push_back(' ');
                }
                res.append(comp.functor);
                return res;
            }
        }

        // Rule 8: Canonical compound.
        return wr_write_canonical(comp, ops, quoted, depth, max_depth);
    }

    return "";
}

[[nodiscard]] auto wr_to_source_entry(const Term& t, const OperatorTable& ops, bool quoted,
                            std::size_t max_depth) -> std::string {
    // Guard against stack exhaustion on cyclic structures by bounding effective recursion depth.
    const std::size_t effective_max_depth = std::min(max_depth, std::size_t{10000});
    return wr_to_source_internal(t, ops, quoted, 0, effective_max_depth, 1200, false);
}

// ---------------------------------------------------------------------------
// Operator-precedence parser.
// ---------------------------------------------------------------------------

// The reader is recursive, and each nesting level costs two native frames. 10000 levels
// would overflow the stack before the cap ever fired, so the cap has to be low enough to be
// the thing that stops it: 1000 nested parentheses is far beyond any real source and well
// inside the 8 MB stack the build reserves.
constexpr std::uint32_t par_max_nesting = 1000;

struct VarScope {
    std::vector<std::pair<std::string, Term>> named;
    std::uint64_t anon_counter = 0;
};

struct ParsedTerm {
    Term term;
    std::uint32_t priority;
};

[[nodiscard]] auto par_parse_uint64(std::string_view s) -> Result<std::uint64_t> {
    if (s.empty()) {
        return make_error<std::uint64_t>(MathError::syntax_error);
    }
    int base = 10;
    if (s.size() > 2 && s[0] == '0') {
        if (s[1] == 'x' || s[1] == 'X') {
            base = 16;
            s.remove_prefix(2);
        } else if (s[1] == 'o' || s[1] == 'O') {
            base = 8;
            s.remove_prefix(2);
        } else if (s[1] == 'b' || s[1] == 'B') {
            base = 2;
            s.remove_prefix(2);
        }
    }
    std::uint64_t val = 0;
    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), val, base);
    if (ec == std::errc::result_out_of_range) {
        return make_error<std::uint64_t>(MathError::overflow);
    }
    if (ec != std::errc{} || ptr != s.data() + s.size()) {
        return make_error<std::uint64_t>(MathError::syntax_error);
    }
    return val;
}

[[nodiscard]] auto par_fold_sign_integer(bool is_minus, const PToken& int_tok) -> Result<std::int64_t> {
    if (int_tok.text.starts_with("0'")) {
        // Character code literals always fit within signed 64-bit integer limits.
        return is_minus ? -int_tok.value : int_tok.value;
    }
    auto u_res = par_parse_uint64(int_tok.text);
    if (!u_res) {
        if (u_res.error() == MathError::overflow) {
            return make_error<std::int64_t>(MathError::overflow);
        }
        if (int_tok.value >= 0) {
            return is_minus ? -int_tok.value : int_tok.value;
        }
        return make_error<std::int64_t>(MathError::overflow);
    }
    const std::uint64_t uval = *u_res;
    constexpr std::uint64_t kMaxAbsMinInt64 = 9223372036854775808ULL;

    if (uval > kMaxAbsMinInt64) {
        return make_error<std::int64_t>(MathError::overflow);
    }
    if (uval == kMaxAbsMinInt64) {
        if (is_minus) {
            // ISO Prolog requires -9223372036854775808 to evaluate to std::numeric_limits<std::int64_t>::min().
            return std::numeric_limits<std::int64_t>::min();
        }
        return make_error<std::int64_t>(MathError::overflow);
    }
    const auto sval = static_cast<std::int64_t>(uval);
    return is_minus ? -sval : sval;
}

[[nodiscard]] auto par_cannot_start_term(const PToken& tok, const OperatorTable& ops) -> bool {
    switch (tok.kind) {
        case TokKind::close:
        case TokKind::close_list:
        case TokKind::close_curly:
        case TokKind::comma:
        case TokKind::bar:
        case TokKind::end:
        case TokKind::eof:
            return true;
        case TokKind::atom: {
            if (!tok.quoted) {
                // An unquoted atom that has infix or postfix definitions but no prefix definition cannot begin a term.
                const bool has_prefix = lookup_prefix(ops, tok.text).has_value();
                const bool has_infix = lookup_infix(ops, tok.text).has_value();
                const bool has_postfix = lookup_postfix(ops, tok.text).has_value();
                if ((has_infix || has_postfix) && !has_prefix) {
                    return true;
                }
            }
            return false;
        }
        default:
            return false;
    }
}

[[nodiscard]] auto par_operator_priority(const OperatorTable& ops, std::string_view name) -> std::uint32_t {
    std::uint32_t max_p = 0;
    if (const auto d = lookup_infix(ops, name)) {
        max_p = std::max(max_p, d->priority);
    }
    if (const auto d = lookup_prefix(ops, name)) {
        max_p = std::max(max_p, d->priority);
    }
    if (const auto d = lookup_postfix(ops, name)) {
        max_p = std::max(max_p, d->priority);
    }
    return max_p;
}

[[nodiscard]] auto par_parse_term_impl(const std::vector<PToken>& toks, std::size_t& pos,
                                       std::uint32_t max_priority, const OperatorTable& ops,
                                       VarScope& scope, std::size_t depth) -> Result<ParsedTerm>;

[[nodiscard]] auto par_parse_canonical_compound(const std::vector<PToken>& toks, std::size_t& pos,
                                                 const OperatorTable& ops, VarScope& scope,
                                                 std::size_t depth) -> Result<ParsedTerm> {
    std::string functor = toks[pos].text;
    pos += 2; // consume atom and open '('
    if (pos < toks.size() && toks[pos].kind == TokKind::close) {
        // Zero arguments in f() is an explicit syntax error in ISO Prolog.
        return make_error<ParsedTerm>(MathError::syntax_error);
    }
    std::vector<Term> args;
    while (true) {
        auto arg = par_parse_term_impl(toks, pos, 999, ops, scope, depth + 1);
        if (!arg) {
            return arg;
        }
        args.push_back(std::move(arg->term));
        if (pos < toks.size() && toks[pos].kind == TokKind::comma) {
            pos++; // consume comma
            continue;
        }
        if (pos < toks.size() && toks[pos].kind == TokKind::close) {
            pos++; // consume close ')'
            break;
        }
        return make_error<ParsedTerm>(MathError::syntax_error);
    }
    return ParsedTerm{make_compound(std::move(functor), std::move(args)), 0};
}

[[nodiscard]] auto par_parse_list(const std::vector<PToken>& toks, std::size_t& pos,
                                  const OperatorTable& ops, VarScope& scope,
                                  std::size_t depth) -> Result<ParsedTerm> {
    pos++; // consume open_list '['
    if (pos < toks.size() && toks[pos].kind == TokKind::close_list) {
        pos++; // consume close_list ']'
        return ParsedTerm{make_nil(), 0};
    }
    std::vector<Term> elements;
    std::optional<Term> tail;
    while (true) {
        auto elem = par_parse_term_impl(toks, pos, 999, ops, scope, depth + 1);
        if (!elem) {
            return elem;
        }
        elements.push_back(std::move(elem->term));
        if (pos < toks.size() && toks[pos].kind == TokKind::comma) {
            pos++; // consume comma
            continue;
        }
        if (pos < toks.size() && toks[pos].kind == TokKind::bar) {
            pos++; // consume bar '|'
            auto tail_res = par_parse_term_impl(toks, pos, 999, ops, scope, depth + 1);
            if (!tail_res) {
                return tail_res;
            }
            tail = std::move(tail_res->term);
            if (pos >= toks.size() || toks[pos].kind != TokKind::close_list) {
                return make_error<ParsedTerm>(MathError::syntax_error);
            }
            pos++; // consume close_list ']'
            break;
        }
        if (pos < toks.size() && toks[pos].kind == TokKind::close_list) {
            pos++; // consume close_list ']'
            break;
        }
        return make_error<ParsedTerm>(MathError::syntax_error);
    }
    Term result = tail.has_value() ? std::move(*tail) : make_nil();
    for (std::size_t i = elements.size(); i > 0; --i) {
        result = make_compound(".", {std::move(elements[i - 1]), std::move(result)});
    }
    return ParsedTerm{std::move(result), 0};
}

[[nodiscard]] auto par_parse_curly(const std::vector<PToken>& toks, std::size_t& pos,
                                   const OperatorTable& ops, VarScope& scope,
                                   std::size_t depth) -> Result<ParsedTerm> {
    pos++; // consume open_curly '{'
    if (pos < toks.size() && toks[pos].kind == TokKind::close_curly) {
        pos++; // consume close_curly '}'
        return ParsedTerm{make_atom("{}"), 0};
    }
    auto inner = par_parse_term_impl(toks, pos, 1200, ops, scope, depth + 1);
    if (!inner) {
        return inner;
    }
    if (pos >= toks.size() || toks[pos].kind != TokKind::close_curly) {
        return make_error<ParsedTerm>(MathError::syntax_error);
    }
    pos++; // consume close_curly '}'
    return ParsedTerm{make_compound("{}", {std::move(inner->term)}), 0};
}

[[nodiscard]] auto par_parse_primary(const std::vector<PToken>& toks, std::size_t& pos,
                                     std::uint32_t max_priority, const OperatorTable& ops,
                                     VarScope& scope, std::size_t depth) -> Result<ParsedTerm> {
    if (pos >= toks.size() || toks[pos].kind == TokKind::eof) {
        return make_error<ParsedTerm>(MathError::syntax_error);
    }

    const auto& tok = toks[pos];

    if (tok.kind == TokKind::integer) {
        if (tok.needs_negation) {
            // The magnitude 2^63 reached here WITHOUT a preceding sign, so there is no
            // representable value to build.
            return make_error<ParsedTerm>(MathError::overflow);
        }
        Term t = make_int(tok.value);
        pos++;
        return ParsedTerm{std::move(t), 0};
    }

    if (tok.kind == TokKind::var) {
        const std::string& vname = tok.text;
        pos++;
        if (vname == "_") {
            // Anonymous variables never unify with earlier anonymous occurrences in the clause.
            std::string fresh_name = "_$" + std::to_string(scope.anon_counter++);
            return ParsedTerm{make_var(std::move(fresh_name), 0), 0};
        }
        auto it = std::ranges::find_if(scope.named, [&](const auto& pair) {
            return pair.first == vname;
        });
        if (it != scope.named.end()) {
            return ParsedTerm{it->second, 0};
        }
        Term fresh_var = make_var(vname, 0);
        scope.named.push_back({vname, fresh_var});
        return ParsedTerm{std::move(fresh_var), 0};
    }

    if (tok.kind == TokKind::open || tok.kind == TokKind::open_space) {
        pos++; // consume '('
        auto inner = par_parse_term_impl(toks, pos, 1200, ops, scope, depth + 1);
        if (!inner) {
            return inner;
        }
        if (pos >= toks.size() || toks[pos].kind != TokKind::close) {
            return make_error<ParsedTerm>(MathError::syntax_error);
        }
        pos++; // consume ')'
        return ParsedTerm{std::move(inner->term), 0};
    }

    if (tok.kind == TokKind::open_list) {
        return par_parse_list(toks, pos, ops, scope, depth);
    }

    if (tok.kind == TokKind::open_curly) {
        return par_parse_curly(toks, pos, ops, scope, depth);
    }

    if (tok.kind == TokKind::atom) {
        // Immediate open parenthesis without intervening layout denotes a canonical compound.
        if (pos + 1 < toks.size() && toks[pos + 1].kind == TokKind::open) {
            return par_parse_canonical_compound(toks, pos, ops, scope, depth);
        }

        // Quoted atoms never represent operators according to ISO Prolog rules.
        if (tok.quoted) {
            Term t = make_atom(tok.text);
            pos++;
            return ParsedTerm{std::move(t), 0};
        }

        const auto prefix_op = lookup_prefix(ops, tok.text);
        if (prefix_op.has_value()) {
            // Unary sign folding directly folds + and - with immediate integer literals.
            if ((tok.text == "-" || tok.text == "+") && pos + 1 < toks.size() &&
                toks[pos + 1].kind == TokKind::integer) {
                const bool is_minus = (tok.text == "-");
                auto folded = par_fold_sign_integer(is_minus, toks[pos + 1]);
                if (!folded) {
                    return make_error<ParsedTerm>(folded.error());
                }
                pos += 2;
                return ParsedTerm{make_int(*folded), 0};
            }

            // An atom followed by a token that cannot start a term must be treated as a plain atom.
            const bool next_cannot_start =
                (pos + 1 >= toks.size()) || par_cannot_start_term(toks[pos + 1], ops);
            if (next_cannot_start) {
                std::string name = tok.text;
                pos++;
                const std::uint32_t pri = par_operator_priority(ops, name);
                if (pri > max_priority) {
                    return make_error<ParsedTerm>(MathError::syntax_error);
                }
                return ParsedTerm{make_atom(std::move(name)), pri};
            }

            // Fall back to a plain atom if the prefix operator's priority exceeds the current allowance.
            if (prefix_op->priority > max_priority) {
                std::string name = tok.text;
                pos++;
                const std::uint32_t pri = par_operator_priority(ops, name);
                if (pri > max_priority) {
                    return make_error<ParsedTerm>(MathError::syntax_error);
                }
                return ParsedTerm{make_atom(std::move(name)), pri};
            }

            std::string name = tok.text;
            const std::uint32_t op_pri = prefix_op->priority;
            const std::uint32_t arg_max =
                (prefix_op->type == OpType::fx) ? (op_pri > 0 ? op_pri - 1 : 0) : op_pri;
            pos++;
            auto arg_res = par_parse_term_impl(toks, pos, arg_max, ops, scope, depth + 1);
            if (!arg_res) {
                return arg_res;
            }
            return ParsedTerm{make_compound(std::move(name), {std::move(arg_res->term)}), op_pri};
        }

        std::string name = tok.text;
        pos++;
        std::uint32_t pri = 0;
        if (is_any_operator(ops, name)) {
            pri = par_operator_priority(ops, name);
        }
        if (pri > max_priority) {
            return make_error<ParsedTerm>(MathError::syntax_error);
        }
        return ParsedTerm{make_atom(std::move(name)), pri};
    }

    return make_error<ParsedTerm>(MathError::syntax_error);
}

[[nodiscard]] auto par_parse_term_impl(const std::vector<PToken>& toks, std::size_t& pos,
                                       std::uint32_t max_priority, const OperatorTable& ops,
                                       VarScope& scope, std::size_t depth) -> Result<ParsedTerm> {
    if (depth > par_max_nesting) {
        // Defensive recursion backstop guards against stack exhaustion on pathological nestings.
        return make_error<ParsedTerm>(MathError::syntax_error);
    }
    auto left_res = par_parse_primary(toks, pos, max_priority, ops, scope, depth);
    if (!left_res) {
        return left_res;
    }
    auto left = std::move(*left_res);

    while (pos < toks.size()) {
        const auto& peek = toks[pos];

        if (peek.kind == TokKind::comma) {
            if (max_priority < 1000) {
                break;
            }
            constexpr std::uint32_t kPri = 1000;
            constexpr std::uint32_t kLeftMax = 999;
            constexpr std::uint32_t kRightMax = 1000;
            if (left.priority > kLeftMax) {
                return make_error<ParsedTerm>(MathError::syntax_error);
            }
            pos++;
            auto right_res = par_parse_term_impl(toks, pos, kRightMax, ops, scope, depth + 1);
            if (!right_res) {
                return right_res;
            }
            left.term = make_compound(",", {std::move(left.term), std::move(right_res->term)});
            left.priority = kPri;
            continue;
        }

        if (peek.kind == TokKind::bar) {
            if (max_priority < 1100) {
                break;
            }
            constexpr std::uint32_t kPri = 1100;
            constexpr std::uint32_t kLeftMax = 1099;
            constexpr std::uint32_t kRightMax = 1100;
            if (left.priority > kLeftMax) {
                return make_error<ParsedTerm>(MathError::syntax_error);
            }
            pos++;
            auto right_res = par_parse_term_impl(toks, pos, kRightMax, ops, scope, depth + 1);
            if (!right_res) {
                return right_res;
            }
            left.term = make_compound(";", {std::move(left.term), std::move(right_res->term)});
            left.priority = kPri;
            continue;
        }

        if (peek.kind == TokKind::atom && !peek.quoted) {
            const auto infix_def = lookup_infix(ops, peek.text);
            const auto postfix_def = lookup_postfix(ops, peek.text);

            bool is_infix = false;
            bool is_postfix = false;
            if (infix_def.has_value() && postfix_def.has_value()) {
                if (pos + 1 < toks.size() && !par_cannot_start_term(toks[pos + 1], ops)) {
                    is_infix = true;
                } else {
                    is_postfix = true;
                }
            } else if (infix_def.has_value()) {
                is_infix = true;
            } else if (postfix_def.has_value()) {
                is_postfix = true;
            }

            if (is_infix) {
                const auto& op = *infix_def;
                if (op.priority > max_priority) {
                    break;
                }
                std::uint32_t left_max = 0;
                std::uint32_t right_max = 0;
                switch (op.type) {
                    case OpType::xfx:
                        left_max = op.priority > 0 ? op.priority - 1 : 0;
                        right_max = op.priority > 0 ? op.priority - 1 : 0;
                        break;
                    case OpType::xfy:
                        left_max = op.priority > 0 ? op.priority - 1 : 0;
                        right_max = op.priority;
                        break;
                    case OpType::yfx:
                        left_max = op.priority;
                        right_max = op.priority > 0 ? op.priority - 1 : 0;
                        break;
                    default:
                        return make_error<ParsedTerm>(MathError::syntax_error);
                }
                if (left.priority > left_max) {
                    return make_error<ParsedTerm>(MathError::syntax_error);
                }
                std::string op_name = peek.text;
                pos++;
                auto right_res = par_parse_term_impl(toks, pos, right_max, ops, scope, depth + 1);
                if (!right_res) {
                    return right_res;
                }
                left.term = make_compound(std::move(op_name), {std::move(left.term), std::move(right_res->term)});
                left.priority = op.priority;
                continue;
            }

            if (is_postfix) {
                const auto& op = *postfix_def;
                if (op.priority > max_priority) {
                    break;
                }
                const std::uint32_t left_max =
                    (op.type == OpType::xf) ? (op.priority > 0 ? op.priority - 1 : 0) : op.priority;
                if (left.priority > left_max) {
                    return make_error<ParsedTerm>(MathError::syntax_error);
                }
                std::string op_name = peek.text;
                pos++;
                left.term = make_compound(std::move(op_name), {std::move(left.term)});
                left.priority = op.priority;
                continue;
            }
        }

        break;
    }

    return left;
}

[[nodiscard]] auto parse_term_at(const std::vector<PToken>& toks, std::size_t& pos,
                                 std::uint32_t max_priority, const OperatorTable& ops,
                                 VarScope& scope) -> Result<ParsedTerm> {
    return par_parse_term_impl(toks, pos, max_priority, ops, scope, 0);
}

// ---------------------------------------------------------------------------
// Reading: tokens to terms, terms to clauses.
// ---------------------------------------------------------------------------

// Splits a clause body on `,/2`. Only the TOP level is split: `(a ; b)` and `(a -> b)` are
// single goals with their own control semantics and must not be flattened into a conjunction.
auto par_flatten_conjunction(const Term& t, std::vector<Term>& out) -> void {
    if (is_compound(t)) {
        const CompoundNode& c = compound_of(t);
        if (c.functor == "," && c.args.size() == 2) {
            par_flatten_conjunction(c.args[0], out);
            par_flatten_conjunction(c.args[1], out);
            return;
        }
    }
    out.push_back(t);
}

// Reads one term followed by an end token, starting at `pos`. `pos` is left just past the end
// token, so a caller can read a whole file by looping. Returns nullopt at end of input.
[[nodiscard]] auto par_read_one(const std::vector<PToken>& toks, std::size_t& pos,
                                const OperatorTable& ops) -> Result<std::optional<Term>> {
    if (pos >= toks.size() || toks[pos].kind == TokKind::eof) {
        return std::optional<Term>{};
    }
    VarScope scope;
    auto parsed = parse_term_at(toks, pos, 1200, ops, scope);
    if (!parsed) {
        return make_error<std::optional<Term>>(parsed.error());
    }
    if (pos >= toks.size() || toks[pos].kind != TokKind::end) {
        // Stopping short of the terminator means the rest of the text was not understood.
        // Returning the prefix that WAS understood would be reading a different program.
        return make_error<std::optional<Term>>(MathError::syntax_error);
    }
    ++pos;
    return std::optional<Term>(parsed->term);
}

// Reads a `:- op(Priority, Type, Name)` directive, returning the updated table. A malformed
// directive is an error rather than a no-op.
[[nodiscard]] auto par_apply_op_directive(const Term& call, const OperatorTable& ops)
    -> Result<OperatorTable> {
    const CompoundNode& c = compound_of(call);
    if (c.args.size() != 3 || !is_int(c.args[0]) || !is_atom(c.args[1])) {
        return make_error<OperatorTable>(MathError::syntax_error);
    }
    const std::int64_t priority = int_of(c.args[0]).value;
    if (priority < 0 || priority > 1200) {
        return make_error<OperatorTable>(MathError::syntax_error);
    }
    const std::string& type_name = atom_of(c.args[1]).name;
    OpType type{};
    if (type_name == "xfx") {
        type = OpType::xfx;
    } else if (type_name == "xfy") {
        type = OpType::xfy;
    } else if (type_name == "yfx") {
        type = OpType::yfx;
    } else if (type_name == "fy") {
        type = OpType::fy;
    } else if (type_name == "fx") {
        type = OpType::fx;
    } else if (type_name == "xf") {
        type = OpType::xf;
    } else if (type_name == "yf") {
        type = OpType::yf;
    } else {
        return make_error<OperatorTable>(MathError::syntax_error);
    }
    // `op/3` accepts either one name or a list of them.
    std::vector<Term> names;
    if (is_atom(c.args[2])) {
        names.push_back(c.args[2]);
    } else {
        Term cur = c.args[2];
        while (is_cons(cur)) {
            names.push_back(compound_of(cur).args[0]);
            cur = compound_of(cur).args[1];
        }
        if (!is_nil(cur)) {
            return make_error<OperatorTable>(MathError::syntax_error);
        }
    }
    OperatorTable out = ops;
    for (const Term& n : names) {
        if (!is_atom(n)) {
            return make_error<OperatorTable>(MathError::syntax_error);
        }
        // Priority 0 REMOVES the definition; that is how a program un-defines an operator.
        out = priority == 0 ? out.without_op(atom_of(n).name, type)
                            : out.with_op(static_cast<std::uint32_t>(priority), type,
                                          atom_of(n).name);
    }
    return out;
}

[[nodiscard]] auto par_term_to_clause(const Term& t) -> Result<Clause> {
    Term head = t;
    std::vector<Term> body;
    if (is_compound(t)) {
        const CompoundNode& c = compound_of(t);
        if (c.functor == ":-" && c.args.size() == 2) {
            head = c.args[0];
            par_flatten_conjunction(c.args[1], body);
        }
    }
    if (!is_atom(head) && !is_compound(head)) {
        return make_error<Clause>(MathError::syntax_error);
    }
    std::erase_if(body, [](const Term& g) -> bool {
        return is_atom(g) && atom_of(g).name == "true";
    });
    return Clause{.head = std::move(head), .body = std::move(body)};
}

}  // namespace

// ---------------------------------------------------------------------------
// OperatorTable.
// ---------------------------------------------------------------------------

auto OperatorTable::standard() -> OperatorTable {
    OperatorTable t;
    t.defs_ = standard_operator_defs();
    return t;
}

auto OperatorTable::empty() -> OperatorTable { return OperatorTable{}; }

auto OperatorTable::with_op(std::uint32_t priority, OpType type, std::string name) const
    -> OperatorTable {
    OperatorTable out = *this;
    const auto same_class = [type](OpType other) -> bool {
        return (is_infix_type(type) && is_infix_type(other)) ||
               (is_prefix_type(type) && is_prefix_type(other)) ||
               (is_postfix_type(type) && is_postfix_type(other));
    };
    // Replacing within the CLASS is what lets `-` stay both an infix and a prefix operator
    // while a redefinition of one of them leaves the other alone.
    std::erase_if(out.defs_, [&](const OpDef& d) -> bool {
        return d.name == name && same_class(d.type);
    });
    out.defs_.push_back(OpDef{.name = std::move(name), .priority = priority, .type = type});
    return out;
}

auto OperatorTable::without_op(std::string_view name, OpType type) const -> OperatorTable {
    OperatorTable out = *this;
    const auto same_class = [type](OpType other) -> bool {
        return (is_infix_type(type) && is_infix_type(other)) ||
               (is_prefix_type(type) && is_prefix_type(other)) ||
               (is_postfix_type(type) && is_postfix_type(other));
    };
    std::erase_if(out.defs_, [&](const OpDef& d) -> bool {
        return d.name == name && same_class(d.type);
    });
    return out;
}

auto OperatorTable::definitions() const -> const std::vector<OpDef>& { return defs_; }

// ---------------------------------------------------------------------------
// Public reading entry points.
// ---------------------------------------------------------------------------

auto parse_term(std::string_view source, const OperatorTable& ops) -> Result<Term> {
    auto toks = tokenise(source);
    if (!toks) {
        return make_error<Term>(toks.error());
    }
    if (toks->empty() || toks->front().kind == TokKind::eof) {
        return make_error<Term>(MathError::syntax_error);  // empty input is not a term
    }
    std::size_t pos = 0;
    VarScope scope;
    auto parsed = parse_term_at(*toks, pos, 1200, ops, scope);
    if (!parsed) {
        return make_error<Term>(parsed.error());
    }
    // The clause terminator is OPTIONAL when reading a single term: `parse_term("a+b")` is the
    // common case, and requiring a `.` there would make every caller append punctuation the
    // term does not have. `parse_program` still requires one per clause, because there the `.`
    // is what separates the clauses.
    if (pos < toks->size() && (*toks)[pos].kind == TokKind::end) {
        ++pos;
    }
    if (pos < toks->size() && (*toks)[pos].kind != TokKind::eof) {
        return make_error<Term>(MathError::syntax_error);  // trailing text is not ignored
    }
    return parsed->term;
}

auto parse_term(std::string_view source) -> Result<Term> {
    return parse_term(source, OperatorTable::standard());
}

auto parse_clause(std::string_view source, const OperatorTable& ops) -> Result<Clause> {
    auto t = parse_term(source, ops);
    if (!t) {
        return make_error<Clause>(t.error());
    }
    return par_term_to_clause(*t);
}

auto parse_clause(std::string_view source) -> Result<Clause> {
    return parse_clause(source, OperatorTable::standard());
}

auto parse_program(std::string_view source, const OperatorTable& ops) -> Result<Program> {
    auto toks = tokenise(source);
    if (!toks) {
        return make_error<Program>(toks.error());
    }
    OperatorTable table = ops;
    Program out;
    std::size_t pos = 0;
    while (true) {
        auto one = par_read_one(*toks, pos, table);
        if (!one) {
            return make_error<Program>(one.error());
        }
        if (!*one) {
            return out;
        }
        const Term& t = **one;
        // A `:- Directive.` is not a clause. `op/3` is honoured because it changes how the REST
        // of the file reads; anything else is refused rather than skipped, since silently
        // ignoring a directive means reading a program that is not the one written.
        if (is_compound(t) && compound_of(t).functor == ":-" && compound_of(t).args.size() == 1) {
            const Term& call = compound_of(t).args[0];
            if (is_compound(call) && compound_of(call).functor == "op") {
                auto updated = par_apply_op_directive(call, table);
                if (!updated) {
                    return make_error<Program>(updated.error());
                }
                table = *updated;
                continue;
            }
            return make_error<Program>(MathError::not_implemented);
        }
        auto c = par_term_to_clause(t);
        if (!c) {
            return make_error<Program>(c.error());
        }
        out.push_back(std::move(*c));
    }
}

auto parse_program(std::string_view source) -> Result<Program> {
    return parse_program(source, OperatorTable::standard());
}

auto parse_query(std::string_view source, const OperatorTable& ops) -> Result<std::vector<Term>> {
    auto t = parse_term(source, ops);
    if (!t) {
        return make_error<std::vector<Term>>(t.error());
    }
    Term body = *t;
    // A leading `?-` is optional; both `?- a, b.` and `a, b.` are queries.
    if (is_compound(body) && compound_of(body).functor == "?-" &&
        compound_of(body).args.size() == 1) {
        body = compound_of(body).args[0];
    }
    std::vector<Term> goals;
    par_flatten_conjunction(body, goals);
    std::erase_if(goals, [](const Term& g) -> bool {
        return is_atom(g) && atom_of(g).name == "true";
    });
    return goals;
}

auto parse_query(std::string_view source) -> Result<std::vector<Term>> {
    return parse_query(source, OperatorTable::standard());
}

// ---------------------------------------------------------------------------
// Public writing entry points.
// ---------------------------------------------------------------------------

auto to_source(const Term& t, const OperatorTable& ops) -> std::string {
    // 5000 is far past any term a reader could have produced, so the depth guard is a backstop
    // against a hand-built structure rather than a limit callers meet in practice.
    return wr_to_source_entry(t, ops, true, 5000);
}

auto to_source(const Term& t) -> std::string { return to_source(t, OperatorTable::standard()); }

auto to_source(const Clause& c, const OperatorTable& ops) -> std::string {
    std::string out = to_source(c.head, ops);
    if (!c.body.empty()) {
        out += " :-";
        for (std::size_t i = 0; i < c.body.size(); ++i) {
            out += i == 0 ? " " : ", ";
            out += to_source(c.body[i], ops);
        }
    }
    out += '.';
    return out;
}

auto to_source(const Clause& c) -> std::string { return to_source(c, OperatorTable::standard()); }

auto atom_needs_quotes(std::string_view name) -> bool { return wr_atom_needs_quotes(name); }

auto quote_atom(std::string_view name) -> std::string { return wr_quote_atom(name); }

}  // namespace nimblecas::logic_parser
