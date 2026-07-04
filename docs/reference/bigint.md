# `nimblecas.bigint` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/bigint/bigint.cppm`

An **exact signed integer of unbounded magnitude** (ROADMAP §7.18). Everywhere
else the engine works in the `int64` ring and reports a wrap as
`MathError::overflow`; `BigInt` removes that ceiling entirely. It is the
arbitrary-precision integer that unblocks cryptography (modular exponentiation,
primality, RSA) and unbounded CAS arithmetic — `50!`, `100!`, `2^1000` are all
represented exactly, digit for digit. The representation is **sign-magnitude**:
a `negative_` sign flag plus a magnitude stored as little-endian base-2³² limbs
(`mag_[0]` least significant), maintained in a canonical form (no trailing zero
limbs, and never a "−0") so that equality is a field-wise compare and the
ordering is a total order.

**The honesty boundary is the infallible/fallible split.** Because an
arbitrary-precision add, subtract, or multiply can always allocate enough limbs
to hold the exact result, those operations **cannot overflow** and are therefore
**infallible** — they return a `BigInt` by value, not a `Result`. The only ways
a `BigInt` operation can fail are a **division/modulo by zero**, a **malformed
`from_string`**, or a **domain violation in `modpow`** (negative exponent or
non-positive modulus); those alone return `Result` (Rule 32). Division is
**truncated toward zero** (the remainder takes the dividend's sign) and is
implemented by Knuth TAOCP Vol. 2 Algorithm D for multi-limb divisors.

```cpp
import nimblecas.bigint;
```

Depends on [`core`](core.md).

## `BigInt` — an exact signed integer of unbounded magnitude

### Construction and rendering

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| default | `BigInt()` | The canonical zero (empty magnitude, non-negative). |
| `from_i64` | `static auto from_i64(std::int64_t v) -> BigInt` | Exact value of a signed 64-bit integer. `INT64_MIN` is handled without ever negating in signed space (that would be UB). |
| `from_u64` | `static auto from_u64(std::uint64_t v) -> BigInt` | Exact value of an unsigned 64-bit integer. |
| `from_string` | `static auto from_string(std::string_view s) -> Result<BigInt>` | Parse an optionally signed decimal integer. Accepts a leading `+`/`-` then one or more decimal digits; `"0"` and `"-0"` both denote zero, leading zeros are dropped. Empty input, a bare sign, or any non-digit character yields `MathError::syntax_error`. |
| `to_string` | `auto to_string() const -> std::string` | Canonical decimal rendering: `"-"` only for negative values, `"0"` for zero, no superfluous leading zeros. |

### Sign and predicates

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `is_zero` | `auto is_zero() const noexcept -> bool` | `true` when the magnitude is empty. |
| `is_negative` | `auto is_negative() const noexcept -> bool` | `true` for strictly negative values (never for zero). |
| `sign` | `auto sign() const noexcept -> int` | `-1`, `0`, or `+1`. |
| `abs` | `auto abs() const -> BigInt` | Absolute value (magnitude with a cleared sign). |
| `negate` | `auto negate() const -> BigInt` | Additive inverse; negating zero stays the canonical zero (no "−0"). |

### Comparison (total order)

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `operator<=>` | `auto operator<=>(const BigInt& o) const noexcept -> std::strong_ordering` | Total order: by sign first, then by magnitude (length, then most-significant limb down). Yields all of `<`, `<=`, `>`, `>=`. |
| `operator==` | `auto operator==(const BigInt& o) const noexcept -> bool` | Field-wise equality; canonicalisation makes this exact (`-0 == 0`). |

### Arithmetic (infallible — arbitrary precision cannot overflow)

```cpp
[[nodiscard]] auto add(const BigInt& o) const -> BigInt;
[[nodiscard]] auto subtract(const BigInt& o) const -> BigInt;
[[nodiscard]] auto multiply(const BigInt& o) const -> BigInt;

[[nodiscard]] auto operator+(const BigInt& o) const -> BigInt;  // add
[[nodiscard]] auto operator-(const BigInt& o) const -> BigInt;  // subtract
[[nodiscard]] auto operator*(const BigInt& o) const -> BigInt;  // multiply
```

| Method | Behavior |
| :--- | :--- |
| `add` / `subtract` | Sign-magnitude addition: like signs add magnitudes; unlike signs subtract the smaller magnitude from the larger and inherit the larger's sign (equal magnitudes cancel to canonical zero). `subtract` is `add(o.negate())`. |
| `multiply` | Schoolbook (O(n·m)) magnitude product with `uint64` intermediates; the sign is the XOR of the operand signs, and any zero operand gives the exact zero. |

These return a `BigInt` **by value**, not a `Result`: the result is always exact,
so there is nothing to fail.

### Division (truncated toward zero)

```cpp
[[nodiscard]] auto divmod(const BigInt& divisor) const -> Result<std::pair<BigInt, BigInt>>;
[[nodiscard]] auto divide(const BigInt& divisor) const -> Result<BigInt>;
[[nodiscard]] auto mod(const BigInt& divisor) const -> Result<BigInt>;
```

| Method | Behavior |
| :--- | :--- |
| `divmod` | Returns `(quotient, remainder)` with `a == q*divisor + r` and `\|r\| < \|divisor\|` **exactly**. Truncated toward zero: the quotient's sign is the XOR of the operand signs, and the remainder inherits the **dividend's** sign. Dispatches to a single-limb fast path, a trivial "dividend smaller than divisor" case, or Knuth Algorithm D (normalised long division) for a multi-limb divisor. A zero divisor yields `MathError::division_by_zero`. |
| `divide` | The quotient alone (`divmod(...).first`). |
| `mod` | The remainder alone (`divmod(...).second`), carrying the dividend's sign. |

### Number theory (cryptography foundation)

```cpp
[[nodiscard]] auto pow(std::uint64_t exp) const -> BigInt;
[[nodiscard]] static auto gcd(const BigInt& a, const BigInt& b) -> BigInt;
[[nodiscard]] auto modpow(const BigInt& exp, const BigInt& modulus) const -> Result<BigInt>;
```

| Method | Behavior |
| :--- | :--- |
| `pow` | Exponentiation by squaring, returning `(*this)^exp` exactly. `pow(0) == 1` (so `0^0 == 1` by convention here). **Infallible** — the exact result may be enormous but never overflows. |
| `gcd` | Non-negative greatest common divisor via the Euclidean algorithm on magnitudes; operand signs are ignored. `gcd(0, n) == \|n\|` and `gcd(0, 0) == 0`. **Infallible.** |
| `modpow` | Modular exponentiation by squaring: `(*this)^exp mod modulus`, result in `[0, modulus)` (a negative base is folded into that range). Requires `exp >= 0` and `modulus > 0`, else `MathError::domain_error`. `exp == 0` gives `1 mod modulus`. |

## Error model

| Condition | Error |
| :--- | :--- |
| `from_string` on empty input, a bare `+`/`-`, or any non-digit character | `MathError::syntax_error` |
| `divmod` / `divide` / `mod` with a zero divisor | `MathError::division_by_zero` |
| `modpow` with a negative exponent | `MathError::domain_error` |
| `modpow` with a non-positive modulus (`modulus <= 0`) | `MathError::domain_error` |

`add`, `subtract`, `multiply`, `pow`, and `gcd` are **infallible** and never
error — arbitrary precision has no overflow condition to report. The sign and
comparison predicates (`is_zero`, `is_negative`, `sign`, `abs`, `negate`,
`operator<=>`, `operator==`) are total and never error.

## Worked examples

```cpp
import nimblecas.bigint;
using namespace nimblecas;

auto bi = [](std::string_view s) { return BigInt::from_string(s).value(); };

// Exactness past every int64 ceiling.
BigInt::from_u64(4294967296ULL).to_string();          // "4294967296"   (2^32)
BigInt::from_u64(2).pow(100).to_string();             // "1267650600228229401496703205376"  (2^100)
BigInt::from_i64(std::numeric_limits<std::int64_t>::min()).to_string();  // "-9223372036854775808"

// 100! computed exactly (158 digits) — infallible multiply, never overflows.
BigInt f = BigInt::from_u64(1);
for (std::uint64_t k = 1; k <= 100; ++k) {
    f = f.multiply(BigInt::from_u64(k));
}
f.to_string().size();                                 // 158

// Addition / subtraction (infallible, sign-magnitude).
bi("999999999999999999999999999999").add(bi("1"));    // 1000000000000000000000000000000
bi("5").subtract(bi("9")).to_string();                // "-4"
bi("-6").multiply(bi("7")).to_string();               // "-42"

// Division: truncated toward zero, remainder takes the dividend's sign.
auto dm = bi("-17").divmod(bi("5")).value();
dm.first.to_string();                                 // "-3"   (quotient truncates)
dm.second.to_string();                                // "-2"   (remainder, dividend's sign)
// Ring law holds exactly: -17 == (-3)*5 + (-2).

// Division by zero is reported, not undefined.
bi("42").divmod(BigInt{}).error();                    // MathError::division_by_zero
bi("42").divide(BigInt{}).error();                    // MathError::division_by_zero
bi("42").mod(BigInt{}).error();                        // MathError::division_by_zero

// Malformed input.
BigInt::from_string("12a3").error();                  // MathError::syntax_error
BigInt::from_string("-").error();                     // MathError::syntax_error (bare sign)

// GCD (non-negative, sign-agnostic).
BigInt::gcd(bi("462"), bi("1071")).to_string();       // "21"
BigInt::gcd(BigInt{}, bi("-17")).to_string();         // "17"   (gcd(0, n) = |n|)

// Modular exponentiation for cryptography.
BigInt::from_u64(3).modpow(BigInt::from_u64(644),
                           BigInt::from_u64(645)).value().to_string();  // "36"
// Fermat: a^(p-1) mod p == 1 for prime p = 97.
BigInt::from_u64(2).modpow(BigInt::from_u64(96),
                           BigInt::from_u64(97)).value();               // 1
// A negative base is reduced into [0, modulus): (-2)^3 mod 7 = 6.
bi("-2").modpow(BigInt::from_u64(3), BigInt::from_u64(7)).value();      // 6
// Domain guards.
BigInt::from_u64(2).modpow(bi("-1"), BigInt::from_u64(5)).error();     // MathError::domain_error
BigInt::from_u64(2).modpow(BigInt::from_u64(3), BigInt{}).error();     // MathError::domain_error
```

## See also

- [`nimblecas.core`](core.md) — the `Result<T>` / `MathError` railway every
  fallible `BigInt` operation threads through.
- [`nimblecas.combinatorics`](combinatorics.md) — the primary consumer, returning
  exact `BigInt` values for factorials and integer sequences.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field whose content
  reduction leans on `BigInt` GCD.
- [Documentation hub](../Index.md)
```
