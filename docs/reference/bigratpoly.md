# `nimblecas.bigratpoly` — Dense univariate polynomials over the unbounded rationals (Reference)

**Author:** Olumuyiwa Oluwasanmi

Source: `src/bigratpoly/bigratpoly.cppm`

The **unbounded, overflow-free** counterpart to [`nimblecas.ratpoly`](ratpoly.md).
Where `RationalPoly` carries `int64` `Rational` coefficients and surfaces
`MathError::overflow` (Rule 32) the moment a numerator or denominator
saturates, `BigRationalPoly` carries `BigRational` coefficients and has **no
such ceiling** — it is the foundation for a bignum-backed splitting-field /
Jordan path that must exceed the `int64` overflow envelope. Because
`BigRational`'s magnitude-combining operations (`add`, `subtract`, `multiply`,
`negate`) are themselves **infallible** (arbitrary precision cannot overflow),
the polynomial operations built purely on them — `add`, `subtract`,
`multiply`, `scale`, `negate`, `derivative` — are likewise **infallible** and
return a `BigRationalPoly` by value, not a `Result`. Only the operations that
can genuinely fail return `Result`: `divide` / `monic` / `gcd` fail on a zero
divisor (`MathError::division_by_zero`).

The one fallible boundary that remains is the bridge back **down** to the
`int64` tier: **`to_ratpoly()`** narrows a `BigRationalPoly` to a
`RationalPoly` and fails with `MathError::overflow` when any coefficient's
numerator or denominator no longer fits an `int64` — an honest admission that
the value genuinely exceeds what the `int64` tier can hold, not a silent
truncation. `from_ratpoly()`, the widening direction, is always exact and
infallible: every `int64` `Rational` is already canonical, so lifting it into
`BigRational` cannot fail. Coefficients are stored low-degree-first and
trimmed (`back() != 0`) so the degree is unambiguous.

```cpp
import nimblecas.bigratpoly;
```

Depends on [`core`](core.md), `bigint`, `bigrational`, and
[`ratpoly`](ratpoly.md) (for the `from_ratpoly` / `to_ratpoly` bridge).

## The honesty boundary

- **`MathError::overflow` is never produced by the arithmetic itself.** Add,
  subtract, multiply, scale, negate, and derivative only combine `BigRational`
  magnitudes, which cannot overflow.
- **`to_ratpoly()` is the one fallible narrowing.** It is the sole place this
  module can report `overflow`: when a coefficient's `BigInt` numerator or
  denominator does not fit `int64`, narrowing honestly fails rather than
  silently wrapping or truncating.
- **`divide` / `monic` / `gcd` are fallible only on a zero divisor**
  (`MathError::division_by_zero`), never on magnitude.

## API summary

```cpp
[[nodiscard]] static auto from_coeffs(std::vector<BigRational> coeffs) -> BigRationalPoly;
[[nodiscard]] static auto constant(const BigRational& c) -> BigRationalPoly;
[[nodiscard]] static auto monomial(const BigRational& coeff, std::size_t degree) -> BigRationalPoly;
[[nodiscard]] static auto zero() -> BigRationalPoly;
[[nodiscard]] static auto from_ratpoly(const RationalPoly& p) -> BigRationalPoly;

[[nodiscard]] auto is_zero() const noexcept -> bool;
[[nodiscard]] auto degree() const noexcept -> std::int64_t;
[[nodiscard]] auto coefficient(std::size_t i) const -> BigRational;
[[nodiscard]] auto leading_coefficient() const -> BigRational;
[[nodiscard]] auto coefficients() const noexcept -> std::span<const BigRational>;

// Arithmetic (INFALLIBLE — arbitrary precision cannot overflow)
[[nodiscard]] auto add(const BigRationalPoly& o) const -> BigRationalPoly;
[[nodiscard]] auto subtract(const BigRationalPoly& o) const -> BigRationalPoly;
[[nodiscard]] auto multiply(const BigRationalPoly& o) const -> BigRationalPoly;
[[nodiscard]] auto scale(const BigRational& s) const -> BigRationalPoly;
[[nodiscard]] auto negate() const -> BigRationalPoly;
[[nodiscard]] auto derivative() const -> BigRationalPoly;  // formal d/dx

[[nodiscard]] auto evaluate(const BigRational& x) const -> Result<BigRational>;

// Field operations (fallible only on a zero divisor)
[[nodiscard]] auto monic() const -> Result<BigRationalPoly>;
[[nodiscard]] auto divide(const BigRationalPoly& divisor) const -> Result<BigRatPolyDivMod>;
[[nodiscard]] auto gcd(const BigRationalPoly& o) const -> Result<BigRationalPoly>;

// Narrow back to the int64 tier: fails with MathError::overflow if a coefficient no longer fits.
[[nodiscard]] auto to_ratpoly() const -> Result<RationalPoly>;

[[nodiscard]] auto is_equal(const BigRationalPoly& o) const -> bool;
[[nodiscard]] auto to_string(std::string_view var = "x") const -> std::string;
```

```cpp
struct BigRatPolyDivMod {
    BigRationalPoly quotient;
    BigRationalPoly remainder;
};
```

## Error model

| Condition | Error |
| :--- | :--- |
| `divide` by the zero polynomial | `MathError::division_by_zero` |
| `gcd` (internally divides) by an ultimately-zero step | `MathError::division_by_zero` |
| `to_ratpoly()` on a coefficient whose numerator/denominator exceeds `int64` | `MathError::overflow` |

There is **no `overflow` row for the arithmetic itself**: `add`, `subtract`,
`multiply`, `scale`, `negate`, and `derivative` cannot fail, so they return
plain `BigRationalPoly` values, not `Result`. `overflow` appears **only** at
the `to_ratpoly()` narrowing boundary.

## Worked example

```cpp
import nimblecas.bigratpoly;
import nimblecas.bigrational;
import nimblecas.ratpoly;
using namespace nimblecas;

auto big = [](std::string_view s) { return BigRational::from_string(s).value(); };
auto bi  = [](std::int64_t v) { return BigRational::from_int(v); };
auto poly = [](std::vector<BigRational> c) { return BigRationalPoly::from_coeffs(std::move(c)); };

// (1e10 + 1e10 x) * (1e10 + 1e10 x) = 1e20 + 2e20 x + 1e20 x^2 — every coefficient
// far beyond INT64_MAX (~9.22e18).
const BigRationalPoly p = poly({big("10000000000"), big("10000000000")});
const BigRationalPoly prod = p.multiply(p);
prod.coefficient(1) == big("200000000000000000000");  // exactly 2e20

// Proof it truly exceeds the int64 tier: narrowing back must overflow.
auto narrow = prod.to_ratpoly();
narrow.has_value();                        // false
narrow.error() == MathError::overflow;     // to_ratpoly on a 2e20 coefficient

// The from_ratpoly / to_ratpoly bridge round-trips values that DO fit int64.
const RationalPoly rp = RationalPoly::from_coeffs(
    {Rational::from_int(-1), Rational::from_int(0), Rational::from_int(1)});  // x^2 - 1
const BigRationalPoly wide = BigRationalPoly::from_ratpoly(rp);
auto back = wide.to_ratpoly();
back.has_value() && back->is_equal(rp);    // true — exact round-trip
```

## See also

- [`nimblecas.ratpoly`](ratpoly.md) — the `int64`-`Rational`, overflow-checked
  sibling this module big-backs and bridges to/from; use it when values fit
  `int64`.
- [`nimblecas.bigrational`](bigrational.md) — the exact unbounded fraction
  field the coefficients live in.
- [Documentation hub](../Index.md)
