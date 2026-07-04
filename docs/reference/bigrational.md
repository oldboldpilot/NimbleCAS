# `nimblecas.bigrational` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/bigrational/bigrational.cppm`

The **exact, unbounded** rational tier of the arithmetic tower. The engine's
rationals climb from the `int64` [`Rational`](ratpoly.md) through a 128-bit
`Rational128` to this top rung: a `BigRational` carries its numerator and
denominator as arbitrary-precision `nimblecas.bigint` `BigInt`s, so it **removes
the overflow ceiling entirely**. Where the `int64`/`int128` tiers report a
saturated numerator or denominator as `MathError::overflow`, `BigRational` never
does — a `BigInt` add / subtract / multiply can always allocate enough limbs to
hold the exact result.

That has a precise, honest consequence for the API surface (Rule 32): the field
operations that only *combine* magnitudes — `add`, `subtract`, `multiply`,
`negate` — **cannot overflow and are therefore INFALLIBLE**; they return a
`BigRational` by value, not a `Result`. Only the operations that can *divide by
zero* (`make`, `divide`, `reciprocal`, `pow`) or *fail to parse* (`from_string`)
return `Result`.

**Honesty boundary.** `BigRational` is exact and unbounded over **Q** — no
rounding, no overflow, ever. The cost is heap-allocating `BigInt` arithmetic on
every operation, so this is the **slow-but-exact** tier: values that comfortably
fit `int64` should use the `int64` [`Rational`](ratpoly.md). The **only lossy
operation on the type is `to_double()`**, which is an explicit approximation —
never use it where exactness matters.

```cpp
import nimblecas.bigrational;
```

Depends on [`core`](core.md) and `nimblecas.bigint`.

## Representation and invariants

A `BigRational` is a pair of `BigInt` members `num_` / `den_`, restored to
canonical form by an internal `normalise()` after every operation. The
invariants make equality a field-wise compare and comparison a total order:

- `den_ > 0` — never zero, never negative; the value's **entire sign lives in
  the numerator**.
- `gcd(|num_|, den_) == 1` — always fully reduced (lowest terms).
- the canonical zero is `0/1`; a default-constructed `BigRational` is exactly it.

## `BigRational` — an exact fraction `num/den` in lowest terms with `den > 0`

### Construction / factories

| Constructor / factory | Signature | Behavior |
| :--- | :--- | :--- |
| default | `BigRational()` | The canonical zero `0/1`. |
| `from_bigint` | `static auto from_bigint(const BigInt& n) -> BigRational` | Lift an integer `n` into **Q** as `n/1` (already reduced, denominator 1). Infallible. |
| `from_int` | `static auto from_int(std::int64_t v) -> BigRational` | The integer `v` as `v/1` (via `from_bigint`). Infallible. |
| `make` | `static auto make(BigInt num, BigInt den) -> Result<BigRational>` | Construct `num/den` in canonical form: reduced and with the sign moved onto the numerator. A zero denominator yields `MathError::division_by_zero`. |
| `from_string` | `static auto from_string(std::string_view s) -> Result<BigRational>` | Parse `"num/den"` or a bare integer `"num"`. Each side follows `BigInt::from_string` (optional leading sign, then decimal digits). A malformed side yields `MathError::syntax_error`; a denominator of zero yields `MathError::division_by_zero`. |

### Accessors

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `numerator` | `auto numerator() const noexcept -> const BigInt&` | The reduced numerator; carries the value's sign. |
| `denominator` | `auto denominator() const noexcept -> const BigInt&` | The reduced denominator; always `> 0`. |
| `is_zero` | `auto is_zero() const noexcept -> bool` | `true` when the numerator is zero. |
| `is_integer` | `auto is_integer() const -> bool` | `true` when `den_ == 1`. |
| `sign` | `auto sign() const noexcept -> int` | Sign of the value: `-1`, `0`, or `+1`. Since `den_ > 0`, exactly the sign of the numerator. |
| `to_string` | `auto to_string() const -> std::string` | `"num/den"`, or just `"num"` when the value is integral (`den_ == 1`). |
| `to_double` | `auto to_double() const -> double` | **APPROXIMATE** conversion (the only lossy operation on this type): the nearest `double` to `num_/den_` for in-range magnitudes, `±inf` when the integer part exceeds the `double` range. Never use where exactness matters. |

### Arithmetic (INFALLIBLE — arbitrary precision cannot overflow)

```cpp
[[nodiscard]] auto add(const BigRational& o) const -> BigRational;
[[nodiscard]] auto subtract(const BigRational& o) const -> BigRational;
[[nodiscard]] auto multiply(const BigRational& o) const -> BigRational;
[[nodiscard]] auto negate() const -> BigRational;
```

| Method | Behavior |
| :--- | :--- |
| `add` | `a/b + c/d = (a·d + c·b)/(b·d)`, renormalised. |
| `subtract` | `a/b − c/d = (a·d − c·b)/(b·d)`, renormalised. |
| `multiply` | `(a/b)·(c/d) = (a·c)/(b·d)`, renormalised. |
| `negate` | `−(a/b) = (−a)/b`; leaves `den_ > 0` and the gcd unchanged, so already canonical. |

These return a plain `BigRational` (not a `Result`): a `BigInt` result can always
allocate the limbs it needs, so there is no overflow branch to report.

### Division / powers (fallible: only on division by zero)

```cpp
[[nodiscard]] auto divide(const BigRational& o) const -> Result<BigRational>;
[[nodiscard]] auto reciprocal() const -> Result<BigRational>;
[[nodiscard]] auto pow(std::int64_t exp) const -> Result<BigRational>;
```

| Method | Behavior |
| :--- | :--- |
| `divide` | `(a/b)/(c/d) = (a·d)/(b·c)`, renormalised. Fails `MathError::division_by_zero` when `o.is_zero()`. |
| `reciprocal` | `1/(a/b) = b/a`, renormalised (the sign moves back onto the numerator when `a < 0`). Fails `MathError::division_by_zero` when the value is zero. |
| `pow` | `(num/den)^exp`. `exp == 0` gives `1` for every base (**`0^0 == 1`**, matching `BigInt::pow`). A negative exponent inverts, so a negative power of zero yields `MathError::division_by_zero`; otherwise a negative exponent is exact. `|exp|` is taken in unsigned space so `INT64_MIN` is never negated. |

### Comparison (total order)

```cpp
[[nodiscard]] auto operator<=>(const BigRational& o) const -> std::strong_ordering;
[[nodiscard]] auto operator==(const BigRational& o) const -> bool;
```

| Operator | Behavior |
| :--- | :--- |
| `operator<=>` | Compares `a/b` against `c/d` by cross-multiplying with the **positive** denominators: with `b, d > 0`, the sign of `a·d − c·b` is the sign of the true difference, so no direction flip is ever needed and the ordering is **total** (`std::strong_ordering`). |
| `operator==` | Field-wise compare of numerator and denominator. Canonical form (reduced, `den_ > 0`) makes this exact equality of values. |

## Error model

Every fallible entry point returns `Result<BigRational>`; the infallible
arithmetic (`add` / `subtract` / `multiply` / `negate`) and the predicates /
accessors never error.

| Condition | Error |
| :--- | :--- |
| `make(num, den)` with `den == 0` | `MathError::division_by_zero` |
| `from_string` with a zero denominator (e.g. `"4/0"`) | `MathError::division_by_zero` |
| `from_string` with a malformed side (empty, non-digit, extra `/`, spaces, decimal point, bare sign) | `MathError::syntax_error` |
| `divide(o)` with `o` zero | `MathError::division_by_zero` |
| `reciprocal()` of zero | `MathError::division_by_zero` |
| `pow(exp)` with `exp < 0` on a zero base | `MathError::division_by_zero` |

There is **no `overflow`** in this table: unbounded `BigInt` storage means the
magnitude-combining operations cannot overflow, which is the whole point of the
tier.

## Worked examples

```cpp
import nimblecas.bigrational;
import nimblecas.bigint;
using namespace nimblecas;

auto br = [](std::string_view s) { return BigRational::from_string(s).value(); };

// Canonicalisation: sign migrates to the numerator, fractions reduce by gcd.
BigRational::make(BigInt::from_i64(2), BigInt::from_i64(-4)).value();  // -1/2
BigRational::make(BigInt::from_i64(6), BigInt::from_i64(4)).value();   // 3/2
BigRational::from_int(7).is_integer();                                 // true (7/1)
br("6/3").to_string();                                                 // "2" (no slash)
br("-4/6").to_string();                                                // "-2/3"

// Field axioms hold exactly.
const auto a = br("3/4");
a.add(a.negate()).is_zero();                          // a + (-a) == 0
a.multiply(a.reciprocal().value()) == br("1");        // a * (1/a) == 1
br("3/4").subtract(br("5/6"));                         // -1/12  (INFALLIBLE — no .value())

// Division / reciprocal guard against zero.
br("5/7").divide(BigRational{}).error();              // MathError::division_by_zero
BigRational{}.reciprocal().error();                   // MathError::division_by_zero
BigRational::from_string("4/0").error();              // MathError::division_by_zero

// Powers, including negative exponents and the 0^0 convention.
br("2/3").pow(10).value();                            // 1024/59049
br("2/3").pow(-2).value();                            // 9/4  (inverts)
br("-3/5").pow(3).value();                            // -27/125 (odd power keeps sign)
BigRational{}.pow(0).value() == BigRational::from_int(1);  // 0^0 == 1
BigRational{}.pow(-3).error();                        // MathError::division_by_zero

// Exactness beyond the int64/int128 ceiling: 10^40/(10^40+1) + 1/(10^40+1) == 1.
const BigInt p = BigInt::from_u64(10).pow(40).add(BigInt::from_i64(1));  // 10^40 + 1
const auto big_a = BigRational::make(BigInt::from_u64(10).pow(40), p).value();
const auto big_b = BigRational::make(BigInt::from_i64(1), p).value();
big_a.add(big_b) == BigRational::from_int(1);         // true — den ~41 digits, exact
big_a.denominator() >                                 // reduced denominator > INT64_MAX
    BigInt::from_i64(std::numeric_limits<std::int64_t>::max());

// Harmonic sum H_50 = sum_{k=1}^{50} 1/k: reduced denominator ~3.1e21, only this tier
// can hold it.
BigRational h{};
for (std::int64_t k = 1; k <= 50; ++k) {
    h = h.add(BigRational::make(BigInt::from_i64(1), BigInt::from_i64(k)).value());
}
h.to_string();  // "13943237577224054960759/3099044504245996706400"

// Total order across magnitudes no int64 rational could hold.
br("-2/3") < br("-1/2");                              // closer to zero is larger
br("2/4") == br("1/2");                               // equal after reduction
br("-7/9").sign() == -1 && br("0").sign() == 0;       // sign() reports -1/0/+1

// APPROXIMATE conversion (the only lossy op) — use only when exactness is not required.
br("1/3").to_double();                                // ~0.333... (nearest double)
```

## See also

- [`nimblecas.ratpoly`](ratpoly.md) — the `int64` `Rational` field one rung
  below; use it when values comfortably fit `int64`.
- `nimblecas.bigint` — the arbitrary-precision integer the numerator and
  denominator are carried in.
- [`nimblecas.core`](core.md) — the `MathError` / `Result` error model these
  factories thread through.
- [Documentation hub](../Index.md)
