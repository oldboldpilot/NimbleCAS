# `nimblecas.logic_parser` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/logic_parser/logic_parser.cppm`

A complete Prolog reader and writer for the first-order term language of
[`nimblecas.logic`](logic.md): a tokenizer, an extensible ISO operator table, an
operator-precedence (Pratt) parser, and a writer that emits canonical Prolog source text.
Together they elevate the logic engine from a substrate manipulated exclusively via
C++ term factories into one that ingests programs and queries directly from text.

The parser and writer observe a strict **honesty boundary** (Code Policy Rule 32):
every fallible operation returns `Result<T> = std::expected<T, MathError>`, and
nothing is guessed or silently coerced.

- **Tokens:** identifiers, single-quoted atoms with full ISO character escapes, symbolic
  atoms by maximal munch, variables, `_` as the anonymous variable, 64-bit integer
  literals in decimal, hexadecimal (`0x`), octal (`0o`), binary (`0b`), and character-code
  (`0'c`) notation, all brackets and solo tokens, `%` line comments, `/* ... */` block
  comments, and the end token (a full stop `.` followed by layout whitespace or EOF).
- **Operators:** the complete ISO standard operator table is provided by default.
  Embedded `:- op(Priority, Type, Name).` directives in source text dynamically modify
  the operator table used for the remainder of the file, allowing domain-specific notations
  to parse seamlessly. Any other directive produces a `MathError::syntax_error` rather than
  being silently skipped.
- **Parsing:** full precedence (priorities 1 to 1200) and associativity (`xfx`, `xfy`,
  `yfx`, `fy`, `fx`, `xf`, `yf`), bracketed lists with optional `|` tails, curly bracket
  notation (`{a}` is parsed as `'{}'(a)` conforming to ISO Prolog), and prefix-operator
  disambiguation rules that differentiate between `- 1` (a negative integer literal),
  `-(1)` (a compound term), and `-` (a bare symbolic atom).
- **Writing and round-trip guarantee:** `to_source` emits Prolog text designed to
  re-read as the identical term structure. The guarantee is an exact round trip:
  `parse_term(to_source(t, ops), ops) == t`. Spacing and parentheses may differ, but the
  resulting abstract syntax tree is syntactically equal.
- **Deliberate limits:** there are **no floating-point numbers**; the engine's only
  numeric term is a 64-bit integer (`std::int64_t`). A float literal such as `3.14`
  fails honestly with `MathError::not_implemented` rather than silently truncating to `3`.
  There are no raw strings or exponential `0.0e10` syntax. Integers exceeding 64 bits
  surface as `MathError::overflow`.

```cpp
import nimblecas.logic_parser;
```

Depends on [`core`](core.md) (`Result`, `MathError`, `CowPtr`) and [`logic`](logic.md)
(`Term`, `Clause`, `Program`).

## The operator table

Prolog operators are categorised into classes defining their position (prefix, infix,
or postfix) and associativity. Argument priorities must either be strictly lower than
the operator's priority (`x`) or less than or equal to it (`y`).

| Class | Type | Associativity | Position |
| :--- | :--- | :--- | :--- |
| `OpType::xfx` | Infix | Non-associative | Binary infix |
| `OpType::xfy` | Infix | Right-associative | Binary infix |
| `OpType::yfx` | Infix | Left-associative | Binary infix |
| `OpType::fy` | Prefix | Associative prefix | Unary prefix |
| `OpType::fx` | Prefix | Non-associative prefix | Unary prefix |
| `OpType::xf` | Postfix | Non-associative postfix | Unary postfix |
| `OpType::yf` | Postfix | Associative postfix | Unary postfix |

The `OperatorTable` class represents an immutable lookup table of operator definitions.
Mutation methods return a modified copy, ensuring thread safety and isolation.

```cpp
struct OpDef {
    std::string name;
    std::uint32_t priority;  // 1..1200
    OpType type;
};

class OperatorTable {
public:
    [[nodiscard]] static auto standard() -> OperatorTable;
    [[nodiscard]] static auto empty() -> OperatorTable;

    [[nodiscard]] auto with_op(std::uint32_t priority, OpType type, std::string name) const
        -> OperatorTable;
    [[nodiscard]] auto without_op(std::string_view name, OpType type) const -> OperatorTable;
    [[nodiscard]] auto definitions() const -> const std::vector<OpDef>&;
};
```

`OperatorTable::standard()` initialises the full ISO operator table (e.g. `:-` at 1200 `xfx`,
`,` at 1000 `xfy`, arithmetic comparisons at 700 `xfx`, `+` and `-` at 500 `yfx` and 200 `fy`,
`*` and `/` at 400 `yfx`). An operator may have multiple entries across different classes
simultaneously (such as `-` acting as both infix `yfx` and prefix `fy`).

## Reading Prolog text

All reader functions consume Prolog source text and return a `Result<T>`. Overloads
omitting an explicit `OperatorTable` default to `OperatorTable::standard()`.

```cpp
[[nodiscard]] auto parse_term(std::string_view source, const OperatorTable& ops) -> Result<Term>;
[[nodiscard]] auto parse_term(std::string_view source) -> Result<Term>;

[[nodiscard]] auto parse_clause(std::string_view source, const OperatorTable& ops)
    -> Result<Clause>;
[[nodiscard]] auto parse_clause(std::string_view source) -> Result<Clause>;

[[nodiscard]] auto parse_program(std::string_view source, const OperatorTable& ops)
    -> Result<Program>;
[[nodiscard]] auto parse_program(std::string_view source) -> Result<Program>;

[[nodiscard]] auto parse_query(std::string_view source, const OperatorTable& ops)
    -> Result<std::vector<Term>>;
[[nodiscard]] auto parse_query(std::string_view source) -> Result<std::vector<Term>>;
```

| Function | Behaviour |
| :--- | :--- |
| `parse_term` | Parses a single first-order term. It must be followed by a full stop `.` or end of input. Trailing characters after the term result in `syntax_error`. |
| `parse_clause` | Parses a single Horn clause: `Head.` (a fact) or `Head :- Body.` (a rule). The rule body is split on `,/2` conjunctions into the goal list stored by `Clause`. Control constructs such as `;` or `->` remain intact as compound subgoals. |
| `parse_program` | Parses an entire source file into a `Program` (`std::vector<Clause>`). Each clause is read in source order. Embedded `:- op(P, T, N).` directives dynamically update the operator table for subsequent clauses. Any other directive returns `syntax_error`. |
| `parse_query` | Parses a query of the form `Goal1, Goal2, ... .` with an optional leading `?-`. Returns the constituent goals as a `std::vector<Term>` suitable for passing directly to `nimblecas::logic::solve`. |

## Writing Prolog text

The writer converts internal term and clause data structures back into valid Prolog
source text.

```cpp
[[nodiscard]] auto to_source(const Term& t, const OperatorTable& ops) -> std::string;
[[nodiscard]] auto to_source(const Term& t) -> std::string;

[[nodiscard]] auto to_source(const Clause& c, const OperatorTable& ops) -> std::string;
[[nodiscard]] auto to_source(const Clause& c) -> std::string;

[[nodiscard]] auto atom_needs_quotes(std::string_view name) -> bool;
[[nodiscard]] auto quote_atom(std::string_view name) -> std::string;
```

| Function | Behaviour |
| :--- | :--- |
| `to_source(Term)` | Emits Prolog source text for term `t`, formatting operators according to `ops` and applying minimal parentheses to preserve precedence. Guarantees `parse_term(to_source(t, ops), ops) == t`. |
| `to_source(Clause)` | Emits a clause as Prolog source text terminated with a full stop: `head.` for facts and `head :- g1, g2.` for rules. |
| `atom_needs_quotes` | Determines whether an atom name requires single quotes to re-parse as an atom rather than a variable, number, or syntax error. |
| `quote_atom` | Returns the atom name bare if unquoted parsing is unambiguous, or wrapped in single quotes with escaped internal quotes and control codes otherwise. |

## Error model

All parsing routines report failure through `Result<T>`:

| Condition | Error |
| :--- | :--- |
| Unterminated quotes, mismatched brackets, invalid tokens, trailing unconsumed text | `MathError::syntax_error` |
| Directive other than `:- op(...)` encountered in `parse_program` | `MathError::syntax_error` |
| Floating-point number literal (e.g. `3.14`) in input | `MathError::not_implemented` |
| Integer literal outside the signed 64-bit range `[INT64_MIN, INT64_MAX]` | `MathError::overflow` |
| Non-callable clause head (e.g. variable or number head) | `MathError::domain_error` |

## Worked examples

```cpp
import nimblecas.logic_parser;
import nimblecas.logic;
import nimblecas.core;

using namespace nimblecas;
using namespace nimblecas::logic_parser;

// 1. Parsing and executing a complete Prolog program
const std::string program_text = R"(
    parent(tom, bob).
    parent(bob, ann).
    parent(bob, pat).

    ancestor(X, Y) :- parent(X, Y).
    ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
)";

auto prog = parse_program(program_text).value();

// Parse a query with variable standardisation
auto query = parse_query("?- ancestor(tom, Who).").value();

// Solve via nimblecas.logic
auto answers = solve(prog, query).value();
// Found solutions binding Who -> bob, ann, pat

// 2. Round-trip term guarantee: parse -> to_source -> parse
const std::string original = "f(g(1, -42), [a, b, c], 1 + 2 * 3)";
auto term1 = parse_term(original).value();
std::string rendered = to_source(term1);
auto term2 = parse_term(rendered).value();

// Structural equality holds exactly
bool equal = (term1 == term2);  // true

// 3. Dynamic operator definition in source
const std::string custom_program = R"(
    :- op(500, xfx, has_skill).
    alice has_skill coding.
    bob has_skill testing.
)";

auto custom_prog = parse_program(custom_program).value();
auto custom_q = parse_query("alice has_skill What.").value();
auto custom_ans = solve(custom_prog, custom_q).value();

// 4. Honesty boundary: float literals fail rather than truncating
auto float_res = parse_term("3.14159.");
// float_res.error() == MathError::not_implemented
```

## See also

- [`nimblecas.logic`](logic.md) — the underlying term structures, Robinson unification,
  and SLD resolution engine.
- [`nimblecas.logic_index`](logic_index.md) — batched first-argument clause indexing.
- [`nimblecas.logic_compile`](logic_compile.md) — Prolog-to-source compilation for
  deterministic numeric predicates.
- [`nimblecas.logic_dist`](logic_dist.md) — distributed SLD resolution over `taskdag`.
- [Documentation hub](../Index.md)
