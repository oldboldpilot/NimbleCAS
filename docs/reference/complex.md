# `nimblecas.complex` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/complex/complex.cppm`

Exact complex numbers over the rationals — the **Gaussian rationals** `Q + Qi`
(ROADMAP §7.1). A `Complex` is a pair of exact [`Rational`](ratpoly.md)s
(`real`, `imaginary`), so every field operation stays exact: **no floating point
ever enters**. The subring `Q + Qi` is closed under `+`, `-`, `*`, and `/` — a
field — which is exactly what a CAS needs when a computation temporarily crosses
into the complex numbers but must return an exact answer. Both parts are stored
as canonical `Rational`s, so equality is a field-wise compare and the
default-constructed value is the exact zero `0 + 0i`.

```cpp
import nimblecas.complex;
```

Depends on [`core`](core.md) and [`ratpoly`](ratpoly.md).

## The overflow contract

Following the rest of the engine, arithmetic is **exact** and
**overflow-checked** (Rule 32): every part flows through `Rational`'s checked
add / subtract / multiply / divide / negate, so an `int64` numerator or
denominator that would overflow surfaces as `MathError::overflow` rather than
silently wrapping. Division by the zero complex number returns
`MathError::division_by_zero`. Because the parts are already canonical
`Rational`s, `make` / `from_real` / `from_int` / `i` cannot fail and return a
plain `Complex`; only the arithmetic returns `Result`.

## Modulus and argument are intentionally omitted

The modulus `|z| = sqrt(a^2 + b^2)` and the argument `arg(z)` are **irrational in
general** and have no exact `Rational` representation, so they are deliberately
absent from this layer. What *is* exact is the **squared modulus** `a^2 + b^2`,
exposed as `norm_squared()`. `|z|` and `arg(z)` belong to a later
numeric/symbolic layer that can carry radicals or floating-point
approximations.

## `Complex` — an exact complex number `re + im·i` with `Rational` parts

### Construction

| Constructor / factory | Signature | Notes |
| :--- | :--- | :--- |
| default | `Complex()` | The exact zero `0 + 0i`. |
| `make` | `static auto make(Rational re, Rational im) -> Complex` | Assemble `re + im·i`. The parts are already canonical `Rational`s, so this cannot fail. |
| `from_real` | `static auto from_real(Rational re) -> Complex` | A purely real number `re + 0i`. |
| `from_int` | `static auto from_int(std::int64_t v) -> Complex` | The integer `v` as `v + 0i`. |
| `i` | `static auto i() -> Complex` | The imaginary unit `0 + 1i`. |

### Accessors

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `real` | `auto real() const noexcept -> Rational` | The real part. |
| `imag` | `auto imag() const noexcept -> Rational` | The imaginary part. |
| `is_real` | `auto is_real() const noexcept -> bool` | `true` when the imaginary part vanishes (`0` qualifies). |
| `is_imaginary` | `auto is_imaginary() const noexcept -> bool` | `true` when the real part vanishes — a purely imaginary value or zero. |
| `is_zero` | `auto is_zero() const noexcept -> bool` | `true` when both parts vanish. |
| `operator==` | `auto operator==(const Complex&) const noexcept -> bool` | Field-wise equality (canonical parts make this exact). |
| `to_string` | `auto to_string() const -> std::string` | Human-readable rendering, e.g. `"3 - 4i"`, `"2"`, `"5i"`, `"0"`. |

### Field operations (overflow-checked, exact)

```cpp
[[nodiscard]] auto add(const Complex& o) const -> Result<Complex>;
[[nodiscard]] auto subtract(const Complex& o) const -> Result<Complex>;
[[nodiscard]] auto multiply(const Complex& o) const -> Result<Complex>;
[[nodiscard]] auto divide(const Complex& o) const -> Result<Complex>;
[[nodiscard]] auto negate() const -> Result<Complex>;
[[nodiscard]] auto conjugate() const -> Result<Complex>;
[[nodiscard]] auto reciprocal() const -> Result<Complex>;
[[nodiscard]] auto norm_squared() const -> Result<Rational>;
```

| Method | Behavior |
| :--- | :--- |
| `add` / `subtract` | Field-wise `(a ± c) + (b ± d)i`. |
| `multiply` | `(a + bi)(c + di) = (ac − bd) + (ad + bc)i`. |
| `divide` | `(a + bi)/(c + di) = ((ac + bd) + (bc − ad)i) / (c² + d²)`. Fails `division_by_zero` when `o == 0`. |
| `negate` | `−z = −a − bi`. |
| `conjugate` | The complex conjugate `a − bi`. Overflow-free for canonical parts (negating the imaginary part can only fail on an unreachable `INT64_MIN`), but returns `Result` to stay uniform with the negate-based railway. |
| `reciprocal` | `1 / z`; fails `division_by_zero` when `z == 0` (evaluated as `from_int(1).divide(z)`). |
| `norm_squared` | The exact squared modulus `|z|² = a² + b²` as a `Rational`. `|z|` itself is irrational and omitted. |

Each returns `MathError::overflow` if any `int64` numerator or denominator
computation wraps.

## Error model

| Condition | Error |
| :--- | :--- |
| `divide` by the zero complex number (`o == 0`) | `MathError::division_by_zero` |
| `reciprocal` of the zero complex number | `MathError::division_by_zero` |
| An `int64` numerator or denominator computation wraps | `MathError::overflow` |

There is no `modulus` or `argument` to fail: those irrational quantities are not
provided (see above). `is_real` / `is_imaginary` / `is_zero` are total predicates
and never error.

## Worked examples

```cpp
import nimblecas.complex;
import nimblecas.ratpoly;
using namespace nimblecas;

auto ri = [](std::int64_t v) { return Rational::from_int(v); };
auto cx = [](std::int64_t re, std::int64_t im) {  // re + im·i
    return Complex::make(ri(re), ri(im));
};

// Addition / subtraction (exact, field-wise).
cx(1, 2).add(cx(3, -1)).value();          // 4 + i
cx(4, 1).subtract(cx(3, -1)).value();     // 1 + 2i

// Multiplication: (1 + i)(1 - i) = 2 (a real result), and i·i = -1.
cx(1, 1).multiply(cx(1, -1)).value();     // 2   (is_real() == true)
Complex::i().multiply(Complex::i()).value();  // -1

// Conjugate and the exact squared modulus.
cx(3, 4).conjugate().value();             // 3 - 4i
cx(3, 4).norm_squared().value();          // 25   (= 9 + 16, a Rational)

// Division and its zero guard.
cx(1, 1).divide(cx(1, -1)).value();       // i
cx(1, 1).divide(Complex{}).error();       // MathError::division_by_zero

// Reciprocal: 1/i = -i (and reciprocal of zero fails).
Complex::i().reciprocal().value();        // -i
Complex{}.reciprocal().error();           // MathError::division_by_zero

// Fractional (non-integer) parts stay exact: z = 1/2 + 1/2 i.
const auto half = Rational::make(1, 2).value();
const auto z    = Complex::make(half, half);
z.add(z.conjugate().value()).value();     // 1     (z + conj(z) = 2·Re(z))
z.multiply(z.conjugate().value()).value();// 1/2   (z · conj(z) = |z|²)
z.norm_squared().value();                 // 1/2
z.divide(z).value();                      // 1

// Rendering.
cx(3, -4).to_string();                    // "3 - 4i"
cx(3, 4).to_string();                     // "3 + 4i"
cx(2, 0).to_string();                     // "2"    (purely real, no i)
Complex::make(ri(0), ri(5)).to_string();  // "5i"
Complex::i().to_string();                 // "i"
cx(0, -1).to_string();                    // "-i"
Complex{}.to_string();                    // "0"
```

## See also

- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field the real and
  imaginary parts live in.
- [`nimblecas.matrix`](matrix.md), [`nimblecas.combinatorics`](combinatorics.md),
  and [`nimblecas.orthopoly`](orthopoly.md) — the sibling `ratpoly`-consuming
  numeric modules.
- [Documentation hub](../Index.md)
