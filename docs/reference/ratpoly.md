# `nimblecas.ratpoly` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/ratpoly/ratpoly.cppm`

Dense univariate polynomial arithmetic over the **rationals**, `Q[x]`, built on
an exact reduced `int64` fraction type. The integer ring
[`Polynomial`](polynomial.md) (`Z[x]`) is *not* a field — a general division of
one polynomial by another does not stay in `Z[x]` — so partial-fraction
decomposition, true division-with-remainder, and a Euclidean GCD with a *monic*
result all need the coefficient field `Q`. This module supplies that field
(`Rational`) and the dense polynomial over it (`RationalPoly`). Coefficients are
stored **trimmed** (no trailing zeros) so the degree is unambiguous and equality
is a plain vector compare.

```cpp
import nimblecas.ratpoly;
```

Depends on [`core`](core.md) and [`polynomial`](polynomial.md).

## The overflow contract

Following the rest of the engine, arithmetic is **exact** and
**overflow-checked**: when an `int64` numerator or denominator would overflow,
the operation returns `MathError::overflow` rather than silently wrapping
(Rule 32) — consistent with the `int64` [`Polynomial`](polynomial.md). Because
`Q` is a field, `RationalPoly::divide` always yields an exact quotient and a
remainder of strictly smaller degree; the only failure modes are a zero divisor
or an intermediate coefficient overflow.

## `Rational` — exact fraction `num/den`

An exact fraction in **lowest terms with `den > 0`**. The canonical form
(`den > 0`, `gcd(|num|, den) == 1`, zero represented as `0/1`) makes equality a
field-wise compare and keeps intermediate magnitudes as small as `int64` allows.

### Construction

| Constructor / factory | Signature | Notes |
| :--- | :--- | :--- |
| default | `Rational()` | Zero, `0/1`. |
| `make` | `static auto make(std::int64_t num, std::int64_t den) -> Result<Rational>` | Canonical `num/den`. Fails `division_by_zero` on `den == 0`; `overflow` on an `INT64_MIN` operand that cannot be sign-normalised. |
| `from_int` | `static auto from_int(std::int64_t v) -> Rational` | The integer `v` as `v/1`. |

### Accessors

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `numerator` | `auto numerator() const noexcept -> std::int64_t` | Numerator of the reduced fraction. |
| `denominator` | `auto denominator() const noexcept -> std::int64_t` | Denominator (`> 0`). |
| `is_zero` | `auto is_zero() const noexcept -> bool` | `true` when `num == 0`. |
| `is_integer` | `auto is_integer() const noexcept -> bool` | `true` when `den == 1`. |
| `operator==` | `auto operator==(const Rational&) const noexcept -> bool` | Field-wise equality (canonical form makes this exact). |
| `to_string` | `auto to_string() const -> std::string` | `"n"` when integral, else `"n/d"`. |

### Field operations (overflow-checked, exact)

```cpp
[[nodiscard]] auto add(const Rational& o) const -> Result<Rational>;
[[nodiscard]] auto subtract(const Rational& o) const -> Result<Rational>;
[[nodiscard]] auto multiply(const Rational& o) const -> Result<Rational>;
[[nodiscard]] auto divide(const Rational& o) const -> Result<Rational>;
[[nodiscard]] auto negate() const -> Result<Rational>;
```

- Each returns `MathError::overflow` if any `int64` numerator or denominator
  computation wraps; the result is always re-normalised to canonical form.
- **`divide`** returns `MathError::division_by_zero` when `o` is zero.
- **`negate`** fails (`overflow`) only on `INT64_MIN`, whose magnitude is
  unrepresentable.

## `RationalPoly` — dense polynomial over `Rational`

A dense univariate polynomial over `Rational`. `coeffs[i]` is the coefficient of
`x^i`, stored **trimmed** (`back()` is non-zero, empty for the zero polynomial).

### Construction

| Constructor / factory | Signature | Notes |
| :--- | :--- | :--- |
| default | `RationalPoly()` | The zero polynomial. |
| `from_coeffs` | `static auto from_coeffs(std::vector<Rational> coeffs) -> RationalPoly` | `coeffs[i]` = coeff of `x^i`; trailing zero `Rational`s trimmed. |
| `from_polynomial` | `static auto from_polynomial(const Polynomial& p) -> RationalPoly` | Lift an integer polynomial into `Q[x]` (each coefficient over denominator 1). |
| `constant` | `static auto constant(const Rational& c) -> RationalPoly` | Degree-0 polynomial `c` (zero polynomial if `c == 0`). |
| `monomial` | `static auto monomial(const Rational& coeff, std::size_t degree) -> RationalPoly` | `coeff · x^degree` (zero polynomial if `coeff == 0`). |

### Accessors

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `is_zero` | `auto is_zero() const noexcept -> bool` | `true` for the zero polynomial. |
| `degree` | `auto degree() const noexcept -> std::int64_t` | Degree; **-1** for the zero polynomial (conventional). |
| `coefficient` | `auto coefficient(std::size_t i) const -> Rational` | Coefficient of `x^i` (`0/1` beyond the stored degree). |
| `leading_coefficient` | `auto leading_coefficient() const -> Rational` | Highest-degree coefficient (`0/1` for the zero polynomial). |
| `coefficients` | `auto coefficients() const noexcept -> std::span<const Rational>` | View of the trimmed coefficient vector. |
| `is_equal` | `auto is_equal(const RationalPoly& o) const noexcept -> bool` | Exact coefficient-vector equality. |
| `to_string` | `auto to_string(std::string_view var = "x") const -> std::string` | Human-readable rendering. |

### Ring operations (overflow-checked, exact)

```cpp
[[nodiscard]] auto add(const RationalPoly& o) const -> Result<RationalPoly>;
[[nodiscard]] auto subtract(const RationalPoly& o) const -> Result<RationalPoly>;
[[nodiscard]] auto scale(const Rational& s) const -> Result<RationalPoly>;
[[nodiscard]] auto multiply(const RationalPoly& o) const -> Result<RationalPoly>;
```

Each returns `MathError::overflow` if any coefficient computation wraps `int64`.

### Field / division operations

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `monic` | `auto monic() const -> Result<RationalPoly>` | Divides the leading coefficient out, producing a monic polynomial (leading coefficient `1`). The zero polynomial maps to itself. |
| `divide` | `auto divide(const RationalPoly& divisor) const -> Result<RationalDivMod>` | Euclidean division: `(quotient, remainder)` with `*this == quotient · divisor + remainder` and `deg(remainder) < deg(divisor)`. Over the field this is always exact; fails `division_by_zero` on a zero divisor, or `overflow`. |
| `gcd` | `auto gcd(const RationalPoly& o) const -> Result<RationalPoly>` | Monic greatest common divisor via the Euclidean algorithm (`gcd(0, 0) == 0`). |
| `derivative` | `auto derivative() const -> Result<RationalPoly>` | Formal derivative `d/dx`. |
| `to_polynomial` | `auto to_polynomial() const -> Result<Polynomial>` | Convert back to an integer polynomial when every coefficient is integral; fails `domain_error` otherwise. |

The result of `divide` is the free struct:

```cpp
struct RationalDivMod {
    RationalPoly quotient;
    RationalPoly remainder;
};
```

## Example

```cpp
import nimblecas.ratpoly;
using namespace nimblecas;

// p = x^2 - 1  (lifted from Z[x])
const RationalPoly p = RationalPoly::from_polynomial(Polynomial{{-1, 0, 1}});
// q = 2x - 2
const RationalPoly q = RationalPoly::from_polynomial(Polynomial{{-2, 2}});

auto dm = p.divide(q);          // quotient = (1/2)*x + (1/2), remainder = 0
auto g  = p.gcd(q);             // monic: x - 1

auto d  = p.derivative();       // 2x
auto z  = p.to_polynomial();    // back to Z[x]: x^2 - 1
```

## See also

- [`nimblecas.polynomial`](polynomial.md) — the integer ring `Z[x]` this module
  lifts into `Q[x]`.
- [`nimblecas.polyexpr`](polyexpr.md) — bridges `Expr` and `Polynomial`.
- [Documentation hub](../Index.md)
