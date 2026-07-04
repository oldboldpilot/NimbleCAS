# `nimblecas.limits` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/limits/limits.cppm`

Exact **symbolic limits** for a deliberately narrow, *decidable* class of
expressions (ROADMAP §7), built on `nimblecas.diff` and `nimblecas.simplify`.
This is **not** a general limit oracle: anything it cannot decide soundly
returns an honest `MathError` rather than a guessed value (Rule 32).

## Public API

```cpp
enum class LimitDirection : std::uint8_t { both, left, right };

// Limit of f as var -> point (a finite value, or the ±infinity markers below).
[[nodiscard]] auto limit(const Expr& f, std::string_view var, const Expr& point,
                         LimitDirection dir = LimitDirection::both) -> Result<Expr>;

// Limit of f as var -> +infinity (positive=true) or -infinity (positive=false).
[[nodiscard]] auto limit_at_infinity(const Expr& f, std::string_view var,
                                     bool positive = true) -> Result<Expr>;
```

## What is handled

- **Continuity / direct substitution.** If substituting `var := point` and
  simplifying yields a finite constant with no hidden division-by-zero, that
  constant **is** the limit (elementary functions are continuous on their domain).
- **Removable `0/0` at a finite point via iterated L'Hôpital.** The expression is
  split into a quotient `num/den` (a product whose negatively-powered factors form
  the denominator); when `num(point) = den(point) = 0`, `num/den` is replaced by
  `num'/den'` and retried, up to a **hard cap** (`kLHopitalCap = 16`) after which
  the result is `not_implemented` rather than an infinite loop.
- **Rational functions at `±∞`** by comparing leading polynomial degrees:

  | Degrees | Limit |
  | :--- | :--- |
  | `deg(num) < deg(den)` | `0` |
  | `deg(num) = deg(den)` | exact ratio of leading coefficients over `Q` |
  | `deg(num) > deg(den)` | a signed infinity (see convention) |

  Dense polynomial-in-`var` extraction is bounded by `kPolyPowCap = 256`.

## Infinity convention

Signed infinities are represented as nullary function markers — the **only**
places this module emits an infinity; every finite limit is a `ConstantNode`:

```
+infinity  ==  Expr::apply("inf", {})
-infinity  ==  Expr::apply("neg_inf", {})
```

`limit()` also accepts either marker as its `point` argument and forwards to
`limit_at_infinity()`.

## Honesty / out of scope

Everything numeric stays exact over the integers / `Q`; a `double` constant only
ever arises if the input already carried one. The following all return a
`MathError` — **never** a wrong value:

| Case | Error |
| :--- | :--- |
| Oscillatory / essential singularity (e.g. `sin(1/x)` as `x→0`) | `not_implemented` |
| Genuine pole at a finite point (`num(point) ≠ 0`, `den(point) = 0`) | `domain_error` |
| Non-rational behaviour at `∞` not L'Hôpital-reducible here | `not_implemented` |
| Multivariate / symbolic-coefficient forms whose value or sign can't be pinned to a constant | `not_implemented` |
| `int64` overflow in exact rational arithmetic | `overflow` |

**Direction.** `LimitDirection` is accepted for API completeness. For the
decidable class above the two-sided value equals both one-sided values, so it is
returned for any direction; genuinely direction-sensitive cases fall into the
pole / `not_implemented` branches.

## Worked examples

| Query | Result |
| :--- | :--- |
| `lim_{x→2} (x² + 1)` | `5` (continuity) |
| `lim_{x→1} (x² − 1)/(x − 1)` | `2` (L'Hôpital) |
| `lim_{x→1} (x³ − 1)/(x − 1)` | `3` |
| `lim_{x→∞} (2x² + 3)/(x² − 1)` | `2` (equal degree → leading-coeff ratio) |
| `lim_{x→∞} (x + 1)/x²` | `0` (lower degree) |
| `lim_{x→∞} (x² + 1)/(x + 1)` | `+∞` |
| `lim_{x→−∞} (x³ + 1)/x²` | `−∞` (odd degree-gap sign flip) |
| `lim_{x→0} sin(1/x)` | `MathError::not_implemented` |
| `lim_{x→1} 1/(x − 1)` | `MathError::domain_error` |
| `lim_{x→∞} exp(x)` | `MathError::not_implemented` |

## See also

- [`diff.md`](diff.md) — the differentiation used by L'Hôpital.
- [`simplify.md`](simplify.md) — the ASAE simplifier used at every step.
- [`series.md`](series.md) — Taylor expansion (an alternative route to some limits).
