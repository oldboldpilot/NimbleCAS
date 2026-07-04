// NimbleCAS expression reader: the text -> Expr eval surface.
// @author Olumuyiwa Oluwasanmi
//
// Conforms to config/cpp_details.txt: C++23 named module, `import std`, railway
// error handling (Result<T> = std::expected<T, MathError>, no exceptions), trailing
// return types, [[nodiscard]], no owning raw pointers.
//
// This module turns a single expression string into the *literal* structural Expr it
// denotes. It performs NO simplification, evaluation, or constant folding beyond what
// is required to name a leaf (a decimal literal becomes an exact rational leaf; a
// unary minus on a numeric literal becomes the negated numeric leaf). Simplification
// is deliberately the caller's choice (keep concerns separate), so `parse` is a pure
// front-end: every accepted string maps to exactly one Expr tree, and anything it
// cannot denote soundly is returned as a MathError rather than a guessed tree.
//
// ---------------------------------------------------------------------------
// GRAMMAR (Pratt / precedence-climbing recursive descent)
// ---------------------------------------------------------------------------
//   expr    := <precedence-climbing over the operator table below>
//   primary := NUMBER
//            | IDENT                      -> Expr::symbol(ident)
//            | IDENT '(' args? ')'        -> Expr::apply(ident, args)
//            | '(' expr ')'               -> grouping
//            | '-' expr                   -> unary minus (see below)
//   args    := expr (',' expr)*
//   NUMBER  := DIGIT+                     -> Expr::integer
//            | DIGIT* '.' DIGIT+          -> exact Expr::rational (see DECIMALS)
//   IDENT   := [A-Za-z_][A-Za-z0-9_]*
//
// PRECEDENCE / ASSOCIATIVITY (lowest binding first, highest last):
//   level  operators     arity   assoc     lowering
//   -----  ------------  ------  --------   ----------------------------------
//     1     +  -         binary  left       a-b  ->  a + (-1)*b
//     2     *  /         binary  left       a/b  ->  a * b^(-1)  (or rational)
//     3     -            prefix  right      -x   ->  (-1)*x      (looser than ^)
//     4     ^            binary  right      a^b  ->  Expr::power(a, b)   (highest)
//   Atoms (literals, symbols, calls, parenthesised groups) bind tighter than any
//   operator. `-x^2` parses as `-(x^2)` because unary minus (level 3) is looser than
//   `^` (level 4). `2^3^2` parses as `2^(3^2)` (right associative). `a-b-c` parses as
//   `(a-b)-c` (left associative).
//
// DIVISION: `a / b` becomes `a * b^(-1)`, EXCEPT when both operands are integer
//   literals, in which case it becomes an exact `Expr::rational(a, b)` (NOT a division
//   node). A zero denominator in that exact case is reported as
//   MathError::division_by_zero (honest: `3/0` denotes division by zero).
//
// DECIMALS: `1.5` becomes the exact rational 3/2 (15/10 reduced). The reader is
//   deliberately EXACT and never produces an Expr::real leaf: decimals are converted
//   to exact rationals, and an integer/decimal literal too large for std::int64_t is
//   an honest MathError::overflow rather than a lossy double. (Consequence: an Expr
//   built from Expr::real(...) does not round-trip through parse(to_string(...)); see
//   the round-trip note in reader_tests.cpp.)
//
// UNARY MINUS: applied to a numeric literal it folds into the negated literal
//   (`-5` -> Expr::integer(-5), `-3/4` -> Expr::rational(-3,4)); applied to anything
//   else it lowers to `(-1) * operand`. The fold is what lets the printer's `-5` /
//   `(-1 * x)` output re-parse to the identical tree.
//
// REJECTED (each returns MathError::syntax_error, never a partial tree):
//   empty input, an unknown character (e.g. '@'), unbalanced parentheses (`(1+2`),
//   a trailing token after a complete expression (`1 2`), an operator with a missing
//   operand (`1 +`, `*x`), a call with an unterminated argument list (`sin(`).

export module nimblecas.reader;

import std;
import nimblecas.core;
import nimblecas.symbolic;

export namespace nimblecas {

// Parse a single expression string into its literal structural Expr, or a MathError
// describing why the text could not be denoted soundly. No simplification is applied.
[[nodiscard]] auto parse(std::string_view text) -> Result<Expr>;

}  // namespace nimblecas

// ===========================================================================
// Implementation (not exported).
// ===========================================================================
namespace nimblecas {

namespace {

// --- Tokens ---------------------------------------------------------------

enum class TokKind : std::uint8_t {
    Number,
    Ident,
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    Comma,
    End,
};

struct Token {
    TokKind kind{TokKind::End};
    std::string_view text{};  // view into the caller's source string (valid for parse)
    std::size_t pos{0};
};

[[nodiscard]] constexpr auto is_digit(char c) noexcept -> bool {
    return c >= '0' && c <= '9';
}

[[nodiscard]] constexpr auto is_ident_start(char c) noexcept -> bool {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

[[nodiscard]] constexpr auto is_ident_cont(char c) noexcept -> bool {
    return is_ident_start(c) || is_digit(c);
}

[[nodiscard]] constexpr auto is_space(char c) noexcept -> bool {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

// Split the source into a token stream terminated by a single End token. An
// unrecognised character is an honest syntax error (Rule 32) rather than a skip.
[[nodiscard]] auto tokenize(std::string_view s) -> Result<std::vector<Token>> {
    std::vector<Token> out;
    const std::size_t n = s.size();
    std::size_t i = 0;
    while (i < n) {
        const char c = s[i];
        if (is_space(c)) {
            ++i;
            continue;
        }
        // A number is DIGIT+ with an optional '.' DIGIT+ fraction. The fraction is
        // consumed only when a digit follows the dot, so a trailing '.' is left to be
        // rejected as an unknown character rather than silently absorbed.
        if (is_digit(c) || (c == '.' && i + 1 < n && is_digit(s[i + 1]))) {
            const std::size_t start = i;
            while (i < n && is_digit(s[i])) {
                ++i;
            }
            if (i < n && s[i] == '.' && i + 1 < n && is_digit(s[i + 1])) {
                ++i;  // consume '.'
                while (i < n && is_digit(s[i])) {
                    ++i;
                }
            }
            out.push_back({.kind = TokKind::Number, .text = s.substr(start, i - start), .pos = start});
            continue;
        }
        if (is_ident_start(c)) {
            const std::size_t start = i;
            ++i;
            while (i < n && is_ident_cont(s[i])) {
                ++i;
            }
            out.push_back({.kind = TokKind::Ident, .text = s.substr(start, i - start), .pos = start});
            continue;
        }
        TokKind kind{};
        switch (c) {
            case '+': kind = TokKind::Plus; break;
            case '-': kind = TokKind::Minus; break;
            case '*': kind = TokKind::Star; break;
            case '/': kind = TokKind::Slash; break;
            case '^': kind = TokKind::Caret; break;
            case '(': kind = TokKind::LParen; break;
            case ')': kind = TokKind::RParen; break;
            case ',': kind = TokKind::Comma; break;
            default: return make_error<std::vector<Token>>(MathError::syntax_error);
        }
        out.push_back({.kind = kind, .text = s.substr(i, 1), .pos = i});
        ++i;
    }
    out.push_back({.kind = TokKind::End, .text = {}, .pos = n});
    return out;
}

// --- Numeric leaves -------------------------------------------------------

// Multiply-accumulate a decimal digit into a non-negative accumulator, guarding the
// signed-overflow boundary (Rule 32/35). Returns nullopt on overflow.
[[nodiscard]] auto push_digit(std::int64_t acc, char c) noexcept -> std::optional<std::int64_t> {
    constexpr std::int64_t max = std::numeric_limits<std::int64_t>::max();
    const int d = c - '0';
    if (acc > (max - d) / 10) {
        return std::nullopt;
    }
    return acc * 10 + d;
}

// Turn a NUMBER token's text into an exact leaf. Integers become Expr::integer;
// decimals become an exact reduced Expr::rational. Overflow of std::int64_t is an
// honest MathError::overflow (the reader never falls back to a lossy double).
[[nodiscard]] auto parse_number(std::string_view tok) -> Result<Expr> {
    const std::size_t dot = tok.find('.');
    if (dot == std::string_view::npos) {
        std::int64_t v = 0;
        const char* const first = tok.data();
        const char* const last = tok.data() + tok.size();
        const auto [ptr, ec] = std::from_chars(first, last, v);
        if (ec == std::errc{} && ptr == last) {
            return Expr::integer(v);
        }
        return make_error<Expr>(MathError::overflow);  // out of int64 range
    }
    // Decimal: numerator is all digits (integer part then fraction part) read as one
    // integer, denominator is 10^(fraction length). The lexer guarantees a non-empty
    // fraction, so denom > 1 and the rational is well-formed.
    const std::string_view int_part = tok.substr(0, dot);
    std::string_view frac_part = tok.substr(dot + 1);
    // Trailing fractional zeros do not change the value but inflate 10^(length); strip them so
    // e.g. "0.5000000000000000000" reduces to 1/2 instead of a spurious overflow at 10^19.
    while (!frac_part.empty() && frac_part.back() == '0') {
        frac_part.remove_suffix(1);
    }
    std::int64_t num = 0;
    if (frac_part.empty()) {
        // All fractional digits were zero (e.g. "3.000"): an integer value.
        for (const char c : int_part) {
            const auto next = push_digit(num, c);
            if (!next.has_value()) {
                return make_error<Expr>(MathError::overflow);
            }
            num = *next;
        }
        return Expr::integer(num);
    }
    for (const char c : int_part) {
        const auto next = push_digit(num, c);
        if (!next.has_value()) {
            return make_error<Expr>(MathError::overflow);
        }
        num = *next;
    }
    for (const char c : frac_part) {
        const auto next = push_digit(num, c);
        if (!next.has_value()) {
            return make_error<Expr>(MathError::overflow);
        }
        num = *next;
    }
    std::int64_t denom = 1;
    constexpr std::int64_t max = std::numeric_limits<std::int64_t>::max();
    for (std::size_t k = 0; k < frac_part.size(); ++k) {
        if (denom > max / 10) {
            return make_error<Expr>(MathError::overflow);
        }
        denom *= 10;
    }
    return Expr::rational(num, denom);  // canonicalises (reduces by gcd)
}

// --- Structural helpers ---------------------------------------------------

// If e is an integer literal leaf, return its value; otherwise nullopt. Used to decide
// whether `a / b` collapses to an exact rational.
[[nodiscard]] auto as_integer_literal(const Expr& e) -> std::optional<std::int64_t> {
    if (auto held = as<ConstantNode>(e.node().value); held.has_value()) {
        const ConstantNode& c = **held;
        if (std::holds_alternative<std::int64_t>(c.value)) {
            return std::get<std::int64_t>(c.value);
        }
    }
    return std::nullopt;
}

// Negate an expression. A numeric literal folds into its negated literal (so the
// printer's `-5`, `-3/4`, `(-1 * x)` output re-parses to an identical tree); anything
// else lowers to `(-1) * operand`.
[[nodiscard]] auto negate(const Expr& e) -> Result<Expr> {
    if (auto held = as<ConstantNode>(e.node().value); held.has_value()) {
        const ConstantNode& c = **held;
        return std::visit(
            [](const auto& v) -> Result<Expr> {
                using V = std::decay_t<decltype(v)>;
                constexpr std::int64_t vmin = std::numeric_limits<std::int64_t>::min();
                if constexpr (std::is_same_v<V, std::int64_t>) {
                    if (v == vmin) {
                        return make_error<Expr>(MathError::overflow);  // -INT64_MIN is UB
                    }
                    return Expr::integer(-v);
                } else if constexpr (std::is_same_v<V, double>) {
                    return Expr::real(-v);
                } else {  // std::pair<int64,int64> exact rational
                    if (v.first == vmin) {
                        return make_error<Expr>(MathError::overflow);
                    }
                    return Expr::rational(-v.first, v.second);
                }
            },
            c.value);
    }
    return Expr::product({Expr::integer(-1), e});
}

// Lower `a / b`. Two integer literals form an exact rational (a zero denominator is
// an honest division_by_zero); otherwise division is `a * b^(-1)`.
[[nodiscard]] auto make_division(const Expr& lhs, const Expr& rhs) -> Result<Expr> {
    // A literal-zero denominator in ANY numeric form (integer, decimal, or exact rational) is
    // an honest division_by_zero, matching parse("3/0"). Without this check, "3/0.0" (rhs stored
    // as the pair 0/1, not an int literal) would silently lower to the undefined `3 * 0^(-1)`.
    if (auto held = as<ConstantNode>(rhs.node().value); held.has_value()) {
        const bool zero_denom = std::visit(
            [](const auto& v) -> bool {
                using V = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<V, std::int64_t>) {
                    return v == 0;
                } else if constexpr (std::is_same_v<V, double>) {
                    return v == 0.0;
                } else {  // std::pair<int64,int64>
                    return v.first == 0;
                }
            },
            (**held).value);
        if (zero_denom) {
            return make_error<Expr>(MathError::division_by_zero);
        }
    }
    const auto ln = as_integer_literal(lhs);
    const auto rn = as_integer_literal(rhs);
    if (ln.has_value() && rn.has_value()) {
        return Expr::rational(*ln, *rn);  // reports division_by_zero on rn == 0
    }
    return Expr::product({lhs, Expr::power(rhs, Expr::integer(-1))});
}

// Combine two sub-expressions under a binary operator, applying the documented
// lowering (`-` -> `+ (-1)*`, `/` -> rational or `* ^(-1)`).
[[nodiscard]] auto apply_binop(TokKind op, const Expr& lhs, const Expr& rhs) -> Result<Expr> {
    switch (op) {
        case TokKind::Plus:
            return Expr::sum({lhs, rhs});
        case TokKind::Minus: {
            auto neg = negate(rhs);
            if (!neg.has_value()) {
                return neg;
            }
            return Expr::sum({lhs, *neg});
        }
        case TokKind::Star:
            return Expr::product({lhs, rhs});
        case TokKind::Slash:
            return make_division(lhs, rhs);
        case TokKind::Caret:
            return Expr::power(lhs, rhs);
        default:
            return make_error<Expr>(MathError::syntax_error);
    }
}

// Left binding power of an infix operator (0 = not an infix operator). Higher binds
// tighter. `^` is handled right-associatively by the parser.
[[nodiscard]] constexpr auto left_bp(TokKind kind) noexcept -> int {
    switch (kind) {
        case TokKind::Plus:
        case TokKind::Minus: return 10;
        case TokKind::Star:
        case TokKind::Slash: return 20;
        case TokKind::Caret: return 40;
        default: return 0;
    }
}

// Binding power used by the unary minus prefix when parsing its operand: looser than
// `^` (40) so `-x^2 == -(x^2)`, tighter than `*`/`/` (20) so `-x*y == (-x)*y`.
inline constexpr int unary_minus_bp = 30;

// --- Pratt parser ---------------------------------------------------------

class Parser {
public:
    explicit Parser(std::span<const Token> tokens) noexcept : tokens_(tokens) {}

    [[nodiscard]] auto peek() const noexcept -> const Token& { return tokens_[index_]; }

    auto advance() noexcept -> void {
        if (index_ + 1 < tokens_.size()) {
            ++index_;
        }
    }

    // precedence-climbing: parse a sub-expression whose operators bind at least min_bp.
    [[nodiscard]] auto parse_expr(int min_bp) -> Result<Expr> {
        auto first = parse_primary();
        if (!first.has_value()) {
            return first;
        }
        Expr lhs = std::move(*first);
        for (;;) {
            const TokKind kind = peek().kind;
            const int lbp = left_bp(kind);
            if (lbp == 0 || lbp < min_bp) {
                break;  // not an infix operator, or too weak to bind here
            }
            advance();
            // Right-associative `^` recurses at its own bp; left-associative ops recurse
            // one level higher so an equal-precedence operator stops the recursion.
            const int next_min = (kind == TokKind::Caret) ? lbp : lbp + 1;
            auto rhs = parse_expr(next_min);
            if (!rhs.has_value()) {
                return rhs;
            }
            auto combined = apply_binop(kind, lhs, *rhs);
            if (!combined.has_value()) {
                return combined;
            }
            lhs = std::move(*combined);
        }
        return lhs;
    }

private:
    // RAII bound on recursion depth: parse_primary <-> parse_expr recurse through nested
    // parentheses, unary minus, and nested calls, so pathological nesting ("((((..." or
    // "----...") would overflow the native stack. Guarding the single common node (parse_primary)
    // bounds the whole mutual recursion; beyond the cap we return an honest syntax_error rather
    // than crash. 512 is far past any real expression yet safe on a small (~1 MB) stack.
    static constexpr int max_depth = 512;
    struct DepthGuard {
        int& d;
        explicit DepthGuard(int& depth) noexcept : d(depth) { ++d; }
        ~DepthGuard() { --d; }
        DepthGuard(const DepthGuard&) = delete;
        auto operator=(const DepthGuard&) -> DepthGuard& = delete;
    };

    // primary := NUMBER | IDENT call? | '(' expr ')' | '-' expr
    [[nodiscard]] auto parse_primary() -> Result<Expr> {
        const DepthGuard guard(depth_);
        if (depth_ > max_depth) {
            return make_error<Expr>(MathError::syntax_error);  // nesting too deep
        }
        const Token tok = peek();
        switch (tok.kind) {
            case TokKind::Number: {
                advance();
                return parse_number(tok.text);
            }
            case TokKind::Ident: {
                advance();
                if (peek().kind == TokKind::LParen) {
                    return parse_call(tok.text);
                }
                return Expr::symbol(std::string(tok.text));
            }
            case TokKind::Minus: {
                advance();
                auto operand = parse_expr(unary_minus_bp);
                if (!operand.has_value()) {
                    return operand;
                }
                return negate(*operand);
            }
            case TokKind::LParen: {
                advance();
                auto inner = parse_expr(0);
                if (!inner.has_value()) {
                    return inner;
                }
                if (peek().kind != TokKind::RParen) {
                    return make_error<Expr>(MathError::syntax_error);  // unbalanced '('
                }
                advance();
                return inner;
            }
            default:
                // End, ')', ',', or a binary operator in operand position: missing operand.
                return make_error<Expr>(MathError::syntax_error);
        }
    }

    // call := IDENT '(' (expr (',' expr)*)? ')'  (current token is the '(')
    [[nodiscard]] auto parse_call(std::string_view name) -> Result<Expr> {
        advance();  // consume '('
        std::vector<Expr> args;
        if (peek().kind == TokKind::RParen) {
            advance();
            return Expr::apply(std::string(name), std::move(args));
        }
        for (;;) {
            auto arg = parse_expr(0);
            if (!arg.has_value()) {
                return arg;
            }
            args.push_back(std::move(*arg));
            const TokKind kind = peek().kind;
            if (kind == TokKind::Comma) {
                advance();
                continue;
            }
            if (kind == TokKind::RParen) {
                advance();
                break;
            }
            return make_error<Expr>(MathError::syntax_error);  // missing ',' or ')'
        }
        return Expr::apply(std::string(name), std::move(args));
    }

    std::span<const Token> tokens_;
    std::size_t index_{0};
    int depth_{0};
};

}  // namespace

auto parse(std::string_view text) -> Result<Expr> {
    auto tokens = tokenize(text);
    if (!tokens.has_value()) {
        return make_error<Expr>(tokens.error());
    }
    Parser parser(*tokens);
    auto expr = parser.parse_expr(0);
    if (!expr.has_value()) {
        return expr;
    }
    if (parser.peek().kind != TokKind::End) {
        return make_error<Expr>(MathError::syntax_error);  // trailing garbage, e.g. "1 2"
    }
    return expr;
}

}  // namespace nimblecas
