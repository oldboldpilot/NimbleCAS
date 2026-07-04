# `nimblecas.reader` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/reader/reader.cppm`

The **eval surface**: a text → `Expr` parser. NimbleCAS otherwise builds
expression trees only programmatically; this module is the inverse of
`Expr::to_string` and the prerequisite for a REPL and for executable-document
`{nimblecas}` cells ([`execdoc.md`](execdoc.md)).

## Public API (namespace `nimblecas`)

```cpp
[[nodiscard]] auto parse(std::string_view text) -> Result<Expr>;
```

`parse` returns the **literal structural** `Expr` the text denotes, or a
`MathError`. It does **not** simplify or evaluate — that is the caller's choice.

## Grammar

A Pratt / precedence-climbing recursive-descent parser. Precedence, loosest to
tightest:

| Level | Operators | Arity | Assoc | Lowering |
| :--- | :--- | :--- | :--- | :--- |
| 1 (loosest) | `+` `-` | binary | left | `a-b` → `a + (-1)*b` |
| 2 | `*` `/` | binary | left | `a/b` → exact rational, or `a * b^(-1)` |
| 3 | `-` | prefix | — | `-x` → `(-1)*x` (looser than `^`, tighter than `*`/`/`) |
| 4 (tightest) | `^` | binary | right | `Expr::power(a, b)` |

Atoms (literals, symbols, `name(args…)` calls, `(…)` groups) bind tightest.
Verified: `-x^2 → -(x^2)`, `2^3^2 → 2^(3^2)`, `a-b-c → (a-b)-c`, `1+2*3 → 1+(2*3)`.

## Exactness — never emits a real

- Integer literals → `Expr::integer`; a literal exceeding `int64` is an honest
  `MathError::overflow`, **not** a lossy `double`.
- `a/b` of two integer literals → an exact `Expr::rational`; otherwise `a*b^(-1)`.
- Decimals fold to exact rationals: `1.5 → 3/2`, `0.25 → 1/4`. Trailing fractional
  zeros are stripped (`0.5000000000000000000 → 1/2`, `3.000 → 3`) so a value whose
  reduced form fits `int64` is not falsely rejected.
- A **zero denominator in any numeric form** — `3/0`, `3/0.0`, `x/0.0` — is an
  honest `MathError::division_by_zero`.
- Unary minus on a numeric literal folds (`-5 → integer(-5)`, `-3/4 → rational(-3,4)`),
  which is what makes the printer's output re-parse to an identical tree.

## Soundness & robustness

- The parser is **sound**: it returns exactly one tree per accepted string or a
  `MathError` — never a partial or guessed tree.
- Malformed input → `MathError::syntax_error`: empty/blank, unknown character,
  unbalanced parens, trailing garbage, a missing operand, an unterminated call.
- Recursion is depth-bounded (512): pathological nesting (`((((…` or `----…`)
  returns `syntax_error` rather than overflowing the native stack.

## Round-trip

The key correctness property, tested against a set of hand-built expressions:
`parse(e.to_string())` `is_equivalent_to` `e`.

**Known printer limitation** (tracked separately): `Expr::to_string()`
under-parenthesizes a `PowerNode`, so a *rational or negative-constant* base under
`^` (`1/2^x`, `-2^x`) and a *left-nested* power (`a^b^c`) print in a form that
re-parses to a different tree. The parser is correct per standard precedence; the
printer is the lossy side. The round-trip test set avoids these until the printer
is fixed.

## See also

- [`symbolic.md`](symbolic.md) — the `Expr` tree this produces.
- [`execdoc.md`](execdoc.md) — the executable-document engine built on this parser.
- [`latex.md`](latex.md) — the complementary `Expr → LaTeX` renderer.
