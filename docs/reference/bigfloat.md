# `nimblecas.bigfloat` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/bigfloat/bigfloat.cppm`

Software **arbitrary-precision binary floating point** — the floating-point
companion to the exact integer `nimblecas.bigint`, and the analogue of MPFR's
`mpf` / GMP's `mpf_t`. A `BigFloat` is a sign-magnitude `BigInt` mantissa `m`
scaled by a binary exponent `e`, so its value is `m · 2^e`, carried to a
caller-chosen precision `prec` (significant bits). It sits *above*
[`bigint`](core.md) (its mantissa engine) and *beside* the exact numeric layers:
it is what a computation reaches for when it needs hundreds of correct bits of a
value that has no closed exact form, but it is **not** where exact answers live.

**Honesty boundary — read before trusting a result.** This is *software* precision:
no hardware 128/256-bit float exists, so every bit past a machine `double` is
produced by `BigInt` limb arithmetic. It is still **floating point, hence
inexact**: results are rounded to `prec` bits. For *exact* arithmetic use
[`bigint`](core.md) (integers) or `Rational` ([`ratpoly`](ratpoly.md) /
[`complex`](complex.md)). Rounding is **round-to-nearest, ties-to-even**
throughout; `add` / `subtract` / `multiply` / `divide` / `sqrt` are all
**correctly rounded** (the exact result — or a quotient/integer-root carried past
`prec` bits with an exact remainder feeding the sticky bit — is computed, then
rounded exactly once, with no double rounding). Because the canonical form is
unique per value, **dyadic** numbers (`0.5`, `0.75`, `3.25`, …) are represented
**exactly** and equality is a structural compare. There are **no transcendental
functions here** (`exp`, `sin`, `log`, …): this is the raw correctly-rounded
arithmetic substrate; transcendentals belong to a higher numeric layer and would
be numerical, not exact. AVX / AVX2 / AVX-512 change **no** single-operation
result — they only raise throughput of the limb-level mantissa multiply and the
batched reductions inside `BigInt`'s kernels; the `BigFloat` results are
bit-identical either way.

Fallible entry points are railway-oriented (Rule 32): they return
`Result<BigFloat>`. Every out-of-range binary exponent that would overflow
`std::int64_t` surfaces as `MathError::overflow` rather than wrapping.

```cpp
import nimblecas.bigfloat;
```

Depends on [`core`](core.md) and `nimblecas.bigint`.

## Canonical form

After every operation the mantissa is renormalised: trailing zero bits are
absorbed into the exponent (so the stored mantissa is the unique minimal **odd**
integer, or exactly `0` for zero), and a raw result carrying more than `prec`
significant bits is rounded to `prec` bits round-to-nearest, ties-to-even, using
an exact guard/round/sticky discipline. The canonical form is unique per value,
so equality and ordering compare **values, not representations** — a value at
precision 32 can compare equal to the same value at precision 200. The
default/zero value is the unique canonical zero (`mantissa == 0`, `exponent 0`).

## `BigFloat` — value `= mantissa · 2^exponent`, rounded to `precision` bits

### Construction

All constructors are fallible on a non-positive precision (`prec <= 0` →
`domain_error`).

| Factory | Signature | Behavior |
| :--- | :--- | :--- |
| `from_i64` | `static auto from_i64(std::int64_t v, std::int64_t prec) -> Result<BigFloat>` | The integer `v` at `prec` bits (exact whenever `v` fits in `prec` significant bits). |
| `from_bigint` | `static auto from_bigint(const BigInt& b, std::int64_t prec) -> Result<BigFloat>` | The arbitrary-precision integer `b` at `prec` bits. |
| `from_double` | `static auto from_double(double d, std::int64_t prec) -> Result<BigFloat>` | The IEEE `double d` converted **exactly** (a `double` is itself a dyadic rational) then rounded to `prec` bits. NaN / infinity have no representation → `domain_error`. |
| `from_string` | `static auto from_string(std::string_view text, std::int64_t prec) -> Result<BigFloat>` | Parse a decimal literal (optional sign, integer and/or fraction digits, optional `e`/`E` exponent, e.g. `"-3.14"`, `"1.25e2"`, `".5"`, `"6E-3"`). The decimal value is formed exactly then rounded to `prec` bits. A malformed literal → `syntax_error`; an exponent that overflows `int64` → `overflow`. |
| `with_precision` | `auto with_precision(std::int64_t prec) const -> Result<BigFloat>` | Re-round this value to a new precision (widen: exact; narrow: round-to-nearest-even). |

### Accessors

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `is_zero` | `auto is_zero() const noexcept -> bool` | `true` for the canonical zero. |
| `precision` | `auto precision() const noexcept -> std::int64_t` | The significant-bit budget of this value (`> 0` for every constructed value). |
| `sign` | `auto sign() const -> int` | `-1`, `0`, or `+1`. |
| `to_double` | `auto to_double() const -> double` | Nearest `double`, rounded to-nearest-even. May overflow to `±inf` or underflow to `0` for out-of-range exponents. |
| `to_string` | `auto to_string(std::size_t decimal_digits) const -> std::string` | Fixed-point decimal with exactly `decimal_digits` fractional digits, rounded to nearest (ties to even). Shows the full carried precision, e.g. `1/3` → `"0.3333…"`. |

`to_double` performs a genuine nearest-even rounding of the low bits (not a
truncation), so a value exactly halfway between two doubles rounds to the even
one; a magnitude past the `double` range returns `±inf`, and one far below it
returns `0`.

### Arithmetic (round-to-nearest-even at an explicit result precision)

```cpp
[[nodiscard]] auto add(const BigFloat& o, std::int64_t prec) const -> Result<BigFloat>;
[[nodiscard]] auto subtract(const BigFloat& o, std::int64_t prec) const -> Result<BigFloat>;
[[nodiscard]] auto multiply(const BigFloat& o, std::int64_t prec) const -> Result<BigFloat>;
[[nodiscard]] auto divide(const BigFloat& o, std::int64_t prec) const -> Result<BigFloat>;
[[nodiscard]] auto sqrt(std::int64_t prec) const -> Result<BigFloat>;
[[nodiscard]] auto negate() const -> BigFloat;
```

| Method | Behavior |
| :--- | :--- |
| `add` / `subtract` | Align to the smaller exponent (an exact shift), add exactly, then round once. `subtract` is `add(o.negate())`. |
| `multiply` | Exact mantissa product with summed exponents, rounded once. `0` if either operand is zero. |
| `divide` | Long division of the mantissas to `prec + guard` bits with an exact remainder feeding the sticky bit, then a single correct rounding. Divisor `0` → `division_by_zero`. |
| `sqrt` | Correctly-rounded square root via an integer-sqrt (Newton) on the scaled mantissa; the isqrt remainder is the sticky bit. A negative value → `domain_error` (zero → zero). |
| `negate` | The additive inverse — a pure sign flip; the precision is unchanged. **Infallible**: returns a plain `BigFloat`, not a `Result`. |

Every fallible arithmetic method takes the **result precision** `prec` explicitly
(`prec <= 0` → `domain_error`) and returns `MathError::overflow` if the resulting
binary exponent would overflow `std::int64_t`.

### Comparison (by value; precision is irrelevant)

| Operator | Signature | Behavior |
| :--- | :--- | :--- |
| `operator<=>` | `auto operator<=>(const BigFloat& o) const -> std::strong_ordering` | Aligns both mantissas to a common exponent (exact shifts) and compares as signed integers. |
| `operator==` | `auto operator==(const BigFloat& o) const -> bool` | `true` iff the three-way compare is `equal`. Two representations of the same value at different precisions compare equal. |

## Batched reductions (free functions)

A left-fold carried at `prec + 8` bits then rounded **once** to `prec`, so the
intermediate roundings do not accrete error. The AVX/AVX2/AVX-512 throughput win
lives inside `BigInt`'s limb kernels and does not alter these results.

```cpp
[[nodiscard]] auto bigfloat_sum(std::span<const BigFloat> xs, std::int64_t prec)
    -> Result<BigFloat>;
[[nodiscard]] auto bigfloat_dot(std::span<const BigFloat> a, std::span<const BigFloat> b,
                                std::int64_t prec) -> Result<BigFloat>;
```

| Function | Behavior |
| :--- | :--- |
| `bigfloat_sum` | `Σ xs` at `prec` bits (the empty sum is `0`). |
| `bigfloat_dot` | `Σ aᵢ·bᵢ` at `prec` bits. Requires `a.size() == b.size()` → otherwise `domain_error`. |

## Error model

| Condition | Error |
| :--- | :--- |
| Any fallible entry point with `prec <= 0` | `MathError::domain_error` |
| `from_double` of NaN or ±infinity | `MathError::domain_error` |
| `sqrt` of a negative value | `MathError::domain_error` |
| `bigfloat_dot` with mismatched span lengths | `MathError::domain_error` |
| `divide` by a zero `BigFloat` | `MathError::division_by_zero` |
| `from_string` on a malformed literal (bad syntax, trailing garbage, dangling exponent) | `MathError::syntax_error` |
| A binary or decimal exponent that would overflow `std::int64_t` | `MathError::overflow` |

`is_zero` / `precision` / `sign` / `to_double` / `to_string` are total and never
error, and `negate` is infallible (it returns a `BigFloat`, not a `Result`).

## Worked examples

```cpp
import nimblecas.bigfloat;
import nimblecas.bigint;
using namespace nimblecas;

auto bf  = [](std::int64_t v, std::int64_t p) { return BigFloat::from_i64(v, p).value(); };
auto bfd = [](double d, std::int64_t p)       { return BigFloat::from_double(d, p).value(); };
auto bfs = [](std::string_view s, std::int64_t p) { return BigFloat::from_string(s, p).value(); };

// Dyadic values are represented with no rounding at all.
bfd(0.5, 53).to_double();                 // 0.5   (exact)
bfd(3.25, 53).to_double();                // 3.25  (exact)
bfs("3.25", 64) == bfd(3.25, 53);         // true  (decimal 3.25 == double 3.25)
bfs("0.5", 64) == bfs(".5", 64);          // true
bf(0, 64).is_zero();                      // true  (canonical zero)
bf(-7, 64).sign();                        // -1

// Round-to-nearest, ties-to-EVEN at 4 significant bits.
bf(19, 4) == bf(20, 4);                   // true  (19 ties UP to even 20)
bf(21, 4) == bf(20, 4);                   // true  (21 ties DOWN to even 20)
bf(19, 5) == bf(20, 5);                   // false (19 is exact at 5 bits)

// to_double() rounds to the nearest double (ties to even), not truncation.
bfs("9007199254740995", 60).to_double();  // 9007199254740996.0   (2^53+3 -> even 2^53+4)
bfs("9007199254740993", 60).to_double();  // 9007199254740992.0   (2^53+1 -> 2^53)

// Correctly-rounded division converges as precision rises.
bf(1, 128).divide(bf(3, 128), 128).value().to_string(20);
                                          // "0.33333333333333333333"
bf(1, 200).divide(bf(3, 200), 200).value().to_string(30);
                                          // "0.333333333333333333333333333333"

// Correctly-rounded sqrt to hundreds of bits, and exact recovery on a perfect square.
bf(2, 200).sqrt(200).value().to_string(45).substr(0, 32);
                                          // "1.414213562373095048801688724209"
bf(9, 64).sqrt(64).value() == bf(3, 64);  // true
bfs("3.25", 64).multiply(bfs("3.25", 64), 64).value().sqrt(64).value()
    == bfs("3.25", 64);                   // true  (sqrt(3.25^2) == 3.25)

// Exponents far beyond the double range are representable.
const auto huge = bfs("1e400", 64);
huge.sign() == 1;                         // true  (1e400 is a positive value)
std::isinf(huge.to_double());             // true  (overflows to_double() to inf)
huge > bf(1, 64);                         // true
bfs("1e-400", 64).is_zero();              // false (a tiny positive value)

// Scientific notation and large exact integers within precision.
bfs("1.25e2", 64) == bf(125, 64);         // true
bfs("6E-3", 128) == bfs("0.006", 128);    // true
const BigInt big = BigInt::from_string("123456789012345678901234567890").value();
BigFloat::from_bigint(big, 128).value().to_string(0);
                                          // "123456789012345678901234567890"

// Batched reductions (accumulate at extra precision, round once).
const std::vector<BigFloat> xs{bfd(0.5, 64), bfd(0.25, 64), bfd(0.125, 64)};
bigfloat_sum(xs, 64).value() == bfd(0.875, 64);            // true
const std::vector<BigFloat> a{bf(2, 64), bf(3, 64)};
const std::vector<BigFloat> b{bf(4, 64), bf(5, 64)};
bigfloat_dot(a, b, 64).value() == bf(23, 64);              // true  (2·4 + 3·5)

// Error paths.
bf(1, 64).divide(bf(0, 64), 64).error();  // MathError::division_by_zero
bf(-1, 64).sqrt(64).error();              // MathError::domain_error
BigFloat::from_string("1.2.3", 64).error();  // MathError::syntax_error
BigFloat::from_string("1e", 64).error();     // MathError::syntax_error (dangling exponent)
BigFloat::from_i64(5, 0).error();            // MathError::domain_error (non-positive precision)
```

## See also

- [`nimblecas.core`](core.md) — the `Result` / `MathError` railway these entry
  points thread through.
- [`nimblecas.ratpoly`](ratpoly.md) and [`nimblecas.complex`](complex.md) — the
  **exact** `Rational` / Gaussian-rational alternatives to reach for when a value
  has a closed exact form and floating point must not enter.
- [`nimblecas.numeric`](numeric.md) — the double-precision numeric root-finding
  sibling on the floating-point side of the tower.
- [Documentation hub](../Index.md)
