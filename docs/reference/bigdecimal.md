# `nimblecas.bigdecimal` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/bigdecimal/bigdecimal.cppm`

Exact base-10 scaled decimal: a real number carried as `unscaled · 10^(-scale)`,
where `unscaled` is an arbitrary-precision [`BigInt`](bigint.md) and `scale` a
signed `int32`. This is the **money type** and the **boundary quantizer** — the
thing that turns an exact or numerical result into a *stated* decimal amount to a
chosen number of places under a rounding policy the caller names explicitly.

It exists because the two neighbouring exact types cannot carry a decimal
*quantity*:

- **vs [`BigRational`](bigrational.md).** `BigRational` already holds every
  decimal *value* exactly, but `normalise()` reduces to lowest terms on every
  operation, so `2.50` becomes `5/2` — indistinguishable from `2.5`. For money and
  spreadsheet semantics the **scale** ("this amount is stated to the cent") and
  the **rounding mode** are the *contract*, not a rendering detail: `2.50` and
  `2.5` are the **same value** but **different representations** here, and
  `same_representation()` can tell them apart when a test needs to.
- **vs [`BigFloat`](bigfloat.md).** A base-2 float cannot represent `0.1` at any
  precision, so a decimal literal would enter *already wrong*, violating the
  honesty invariant at the door. Base-10 storage means every finite decimal is
  exact from parse to render.

## Honesty boundary

Every finite decimal is **exact** here, and the fallible operations are precise
about where exactness ends:

- **Add / subtract / negate / abs / widening** are **exact and total** — they
  never round and never fail. Alignment to a common scale is a power-of-ten limb
  shift (no gcd), so `0.1 + 0.2 == 0.3` holds *exactly*, unlike IEEE floating
  point.
- **Multiply / pow** are **mathematically exact**; they are fallible *only*
  because the summed `int32` scale can overflow (`MathError::overflow`), never
  because of rounding.
- **Division is the honesty-critical operation.** `divide(scale, Rounding)` rounds
  under an **explicit** policy the caller supplies; `divide_exact()` **refuses**
  with `MathError::inexact` when the quotient does not terminate in base 10 (e.g.
  `1/3`) — it never silently rounds.
- **There is NO ambient / thread-local rounding context** (unlike Java's
  `MathContext` or Python's `decimal` context). Every fallible rounding takes its
  `(scale, Rounding)` by argument. That keeps the type immutable, thread-safe
  under TBB, and **bit-identical across hosts** by construction (it is
  integer-backed — stronger than the `-ffast-math` parity Rule 55 asks for).
- **Equality and ordering are NUMERIC** (`2.50 == 2.5`), matching the `BigFloat`
  precedent and side-stepping Java's `equals`-vs-`compareTo` trap.

```cpp
import nimblecas.bigdecimal;
```

Depends on [`core`](core.md) (the `Result` / `MathError` railway),
[`bigint`](bigint.md) (the unscaled integer), and [`bigrational`](bigrational.md)
(the exact-rational bridge and the terminating-decimal test).

## `Rounding` — the seven directed and half-way rules

An `enum class Rounding : std::uint8_t` — the IEEE 754 / Java `BigDecimal`
rounding modes. Each fallible-rounding operation takes one **explicitly**; there
is no default and no stored mode, so a rounding decision is never made behind the
caller's back.

| Value | Rule |
| :--- | :--- |
| `half_even` | To nearest, ties to **even** — banker's rounding, the money default. |
| `half_up` | To nearest, ties **away from zero** (Excel's `DB` rounds its rate this way at 3 decimals). |
| `half_down` | To nearest, ties **toward zero**. |
| `down` | Toward zero (truncate). |
| `up` | Away from zero. |
| `ceiling` | Toward +∞. |
| `floor` | Toward −∞. |

## `BigDecimal` — a scaled base-10 number, `unscaled · 10^(-scale)`

A **positive** scale means fractional digits (`150` at scale `2` is `1.50`); a
**negative** scale means trailing implied zeros (`15` at scale `-2` is `1500`).
Trailing zeros in `unscaled` are **preserved** — scale is semantic. The
default-constructed value is the canonical zero, `0` at scale `0`.

### Factories (total)

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| default ctor | `BigDecimal()` | The canonical zero (`0` at scale `0`). |
| `from_bigint` | `static auto from_bigint(BigInt unscaled, std::int32_t scale = 0) -> BigDecimal` | `unscaled · 10^(-scale)`; no normalisation of trailing zeros — scale is kept as given. |
| `from_i64` | `static auto from_i64(std::int64_t v, std::int32_t scale = 0) -> BigDecimal` | Same, from a machine integer. |
| `zero` | `static auto zero() -> BigDecimal` | `0` at scale `0`. |
| `one` | `static auto one() -> BigDecimal` | `1` at scale `0`. |

### Factories (fallible)

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `from_string` | `static auto from_string(std::string_view text) -> Result<BigDecimal>` | Parse an exact decimal literal: optional sign, digits, optional `.`, optional `e`/`E` signed exponent. Scale is `(fraction digits) − (exponent)`, so `"1.50"` → `(150, 2)` and `"1.5e3"` → `(15, −2)`. Malformed input → `syntax_error`; a scale that will not fit `int32` → `overflow`. |
| `from_double` | `static auto from_double(double d, std::int32_t scale, Rounding mode) -> Result<BigDecimal>` | Take the double's **exact** dyadic value, then round it to `scale` under `mode`. Deliberately has **no scale-free overload**, so Java's `new BigDecimal(0.1)` binary-artifact trap is unrepresentable. `NaN`/`inf` → `domain_error`. |
| `from_bigrational` | `static auto from_bigrational(const BigRational& r, std::int32_t scale, Rounding mode) -> BigDecimal` | Quantize an exact rational onto `scale` under `mode` (total but for scale overflow) — the reverse of `to_bigrational()`. |
| `from_bigrational_exact` | `static auto from_bigrational_exact(const BigRational& r) -> Result<BigDecimal>` | Exact rational → decimal **without rounding**: succeeds iff the reduced denominator is `2^a · 5^b` (the quotient terminates in base 10); otherwise `inexact`. |

### Accessors and rendering

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `unscaled` | `auto unscaled() const noexcept -> const BigInt&` | The unscaled integer (carries the sign). |
| `scale` | `auto scale() const noexcept -> std::int32_t` | The scale exponent. |
| `is_zero` | `auto is_zero() const noexcept -> bool` | `true` when the value is zero (at any scale). |
| `sign` | `auto sign() const noexcept -> int` | `−1` / `0` / `+1`. |
| `negate` | `auto negate() const -> BigDecimal` | Additive inverse, same scale. |
| `abs` | `auto abs() const -> BigDecimal` | Absolute value, same scale. |
| `to_string` | `auto to_string() const -> std::string` | Plain decimal notation, trailing zeros preserved, **never scientific** (a deliberate divergence from Java): `"1.50"`, `"1500"`, `"0.007"`, `"0.00"`. |
| `to_bigrational` | `auto to_bigrational() const -> BigRational` | Exact, total: `unscaled / 10^scale` as a reduced rational. |

### Arithmetic (exact)

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `add` | `auto add(const BigDecimal& o) const -> BigDecimal` | Align to `max(scale, o.scale)` by a power-of-ten shift, then add. **Infallible.** |
| `subtract` | `auto subtract(const BigDecimal& o) const -> BigDecimal` | `this − o` (via `add(o.negate())`). **Infallible.** |
| `multiply` | `auto multiply(const BigDecimal& o) const -> Result<BigDecimal>` | `unscaled · o.unscaled` at `scale + o.scale`. Exact; fails `overflow` only if the summed scale exceeds `int32`. |
| `pow` | `auto pow(std::int64_t exp) const -> Result<BigDecimal>` | Non-negative integer power, exact. `exp < 0` → `domain_error` (an exact reciprocal power belongs in `BigRational`). Scale overflow → `overflow`. |

### Division and rounding

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `divide` | `auto divide(const BigDecimal& o, std::int32_t result_scale, Rounding mode) const -> Result<BigDecimal>` | `this / o` rounded to `result_scale` under `mode`. Zero divisor → `division_by_zero`. |
| `divide_exact` | `auto divide_exact(const BigDecimal& o) const -> Result<BigDecimal>` | `this / o` **without rounding**: succeeds iff the quotient terminates in base 10, else `inexact`. Zero divisor → `division_by_zero`. |
| `quantize` | `auto quantize(std::int32_t new_scale, Rounding mode) const -> BigDecimal` | Re-express at `new_scale`. Widening (`new_scale >= scale`) is exact and appends zeros; narrowing rounds under `mode`. **Total.** |
| `round` | `auto round(std::int32_t new_scale, Rounding mode) const -> BigDecimal` | Alias for `quantize`, reading better at "round to N places" call sites. |

### Comparison

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `operator<=>` | `auto operator<=>(const BigDecimal& o) const -> std::strong_ordering` | **Numeric** three-way compare: `2.50 <=> 2.5` is `equal`. Aligns to the common scale by a non-negative power-of-ten shift; a total order. |
| `operator==` | `auto operator==(const BigDecimal& o) const -> bool` | Numeric equality (`2.50 == 2.5`). |
| `same_representation` | `auto same_representation(const BigDecimal& o) const -> bool` | Representation identity (scale-sensitive): `2.50` and `2.5` are **not** `same_representation`. |

## Error model

| Condition | Error |
| :--- | :--- |
| `from_string` on malformed input (letters, double dot, empty, lone sign, bad exponent) | `MathError::syntax_error` |
| `from_string` with a scale outside `int32` (e.g. `"1e-9999999999"`) | `MathError::overflow` |
| `from_double` of `NaN` / `±inf` | `MathError::domain_error` |
| `multiply` / `pow` whose summed scale overflows `int32` | `MathError::overflow` |
| `pow` with `exp < 0` | `MathError::domain_error` |
| `from_bigrational_exact` / `divide_exact` of a non-terminating quotient (a prime other than 2 or 5 in the reduced denominator) | `MathError::inexact` |
| `divide` / `divide_exact` by a zero divisor | `MathError::division_by_zero` |

`add` / `subtract` / `negate` / `abs` / `quantize` / `round` / `to_string` /
`to_bigrational` / the accessors and comparisons are **total** and never error.

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.bigdecimal;
using namespace nimblecas;

auto bd = [](std::string_view s) { return BigDecimal::from_string(s).value(); };

// Scale is semantic and preserved on parse/render.
bd("1.50").to_string();              // "1.50"
bd("1.50").scale();                  // 2
bd("1.5e3").to_string();             // "1500"  (scientific folds into the scale)

// Numeric vs representation equality.
bd("2.50") == bd("2.5");                    // true  (same value)
bd("2.50").same_representation(bd("2.5"));  // false (different representation)

// The base-2 traps that defeat double/BigFloat are exact here.
bd("0.1").add(bd("0.2")) == bd("0.3");        // true — 0.1 + 0.2 == 0.3 exactly
bd("2.675").round(2, Rounding::half_even).to_string();  // "2.68" (double gives 2.67)
bd("2.675").round(2, Rounding::down).to_string();       // "2.67"

// Banker's rounding ties to even; directed rounding respects sign.
bd("2.5").round(0, Rounding::half_even).to_string();    // "2"
bd("3.5").round(0, Rounding::half_even).to_string();    // "4"
bd("-2.5").round(0, Rounding::floor).to_string();       // "-3"
bd("-2.5").round(0, Rounding::ceiling).to_string();     // "-2"

// Exact arithmetic: multiply adds scales, widening appends zeros.
bd("1.5").multiply(bd("1.5")).value().to_string();      // "2.25"
bd("2").pow(10).value().to_string();                    // "1024"
bd("1.5").quantize(4, Rounding::down).to_string();      // "1.5000"

// Division: exact-or-refuse.
bd("1").divide_exact(bd("8")).value().to_string();      // "0.125" (terminates)
bd("1").divide_exact(bd("3")).error();                  // MathError::inexact (1/3 refuses)
bd("1").divide_exact(bd("0")).error();                  // MathError::division_by_zero
bd("1").divide(bd("3"), 4, Rounding::half_even).value().to_string();  // "0.3333"

// Conversions. from_double quantizes on entry — no binary artifact leaks.
BigDecimal::from_double(0.1, 2, Rounding::half_even).value().to_string();  // "0.10"
BigDecimal::from_bigrational_exact(BigRational::from_string("3/8").value())
    .value() == bd("0.375");                            // true
BigDecimal::from_bigrational_exact(BigRational::from_string("1/3").value())
    .error() == MathError::inexact;                     // true (non-terminating)
```

## See also

- [`nimblecas.bigrational`](bigrational.md) — the exact rational the decimal
  quantizes from and to; reach for it when the *value*, not a stated scale, is
  what matters.
- [`nimblecas.bigfloat`](bigfloat.md) — the base-2 arbitrary-precision float;
  correctly-rounded, but cannot hold `0.1` exactly.
- [`nimblecas.finance`](finance.md) — the primary consumer: its exact Tier-A
  results quantize to `BigDecimal` money at the boundary.
- [`nimblecas.currency`](currency.md) — `Money` is a tagged `BigDecimal`.
- [Documentation hub](../Index.md)
