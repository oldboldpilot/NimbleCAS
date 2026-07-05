# `nimblecas.recurrence` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/recurrence/recurrence.cppm`

A first slice of **difference equations / recurrence relations** (ROADMAP §7.9).
Given a linear homogeneous recurrence with constant rational coefficients

```
a_n = c_0 a_{n-1} + c_1 a_{n-2} + … + c_{k-1} a_{n-k}      (k = order),
```

its behaviour is governed by the **characteristic polynomial**

```
x^k − c_0 x^{k-1} − c_1 x^{k-2} − … − c_{k-1}             (monic, degree k).
```

Each distinct root `r` of this polynomial, with multiplicity `m`, contributes a
family of basis solutions `r^n, n r^n, …, n^{m-1} r^n` to the general closed form.
This module builds that characteristic polynomial over `Q[x]` and extracts its
**rational** roots (with multiplicity) via [`rational_roots`](roots.md), reusing
the exact `Q[x]` machinery of [`ratpoly`](ratpoly.md) and [`roots`](roots.md).
Two consumers sit on top: `closed_form` assembles the explicit solution `a(n)` as
a symbolic [`Expr`](symbolic.md), and `generating_function` returns the exact
rational generating function `G(x) = P(x)/Q(x)`.

```cpp
import nimblecas.recurrence;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), [`roots`](roots.md),
[`symbolic`](symbolic.md) (the `Expr` closed form), and
[`matrix`](matrix.md) (the exact linear solve for the constants).

## Scope and the honesty boundary

Only the **rational-characteristic-root** case yields a `closed_form`. The rational
roots are found exactly, and `all_roots_rational` reports whether the
characteristic polynomial splits completely over `Q` (so the closed form is
expressible with rational bases alone). When it does **not** — the Fibonacci
recurrence `x² − x − 1`, whose roots are the irrational golden-ratio conjugates,
is the canonical example — the closed form requires irrational or complex roots.
Radical / `RootOf` closed forms for that case are a **planned extension**
(mirroring the same documented limitation in [`roots`](roots.md)) and are not
produced here: `closed_form` returns **`MathError::not_implemented`** rather than
emitting a wrong answer, and `characteristic_roots` returns only the rational
roots (possibly none).

The **generating function** carries no such restriction. `generating_function`
is exact and needs no roots, so it is available for **every** linear homogeneous
constant-coefficient recurrence — including Fibonacci, whose generating function
`x / (1 − x − x²)` is exact even though its closed form is (for now) out of reach.
This is the practical advantage of the rational generating form over the explicit
solution.

| Given | `closed_form` | `generating_function` |
| :--- | :--- | :--- |
| char. poly splits over `Q` (e.g. `x² − 5x + 6`) | exact `Expr` `a(n)` | exact `P/Q` |
| char. poly does **not** split (e.g. Fibonacci `x² − x − 1`) | `not_implemented` | **exact `P/Q`** |

## The overflow contract

Following the rest of the engine, arithmetic is **exact** and
**overflow-checked** (Rule 32): the only arithmetic step,
negating each coefficient into the characteristic polynomial, is an exact
`Rational` negation that fails with `MathError::overflow` on an `int64` boundary
(such as `INT64_MIN`) rather than wrapping.

## Coefficient convention

Coefficients are given **low-order first**: `coeffs[0]` is `c_0` (the `a_{n-1}`
weight) and `coeffs.back()` is `c_{k-1}` (the `a_{n-k}` weight). An **empty**
`coeffs` describes no recurrence and is rejected with `MathError::domain_error` by
every function.

## API

```cpp
[[nodiscard]] auto characteristic_polynomial(std::span<const Rational> coeffs)
    -> Result<RationalPoly>;

[[nodiscard]] auto characteristic_roots(std::span<const Rational> coeffs)
    -> Result<std::vector<std::pair<Rational, std::int64_t>>>;

[[nodiscard]] auto all_roots_rational(std::span<const Rational> coeffs)
    -> Result<bool>;

[[nodiscard]] auto closed_form(std::span<const Rational> coeffs,
                               std::span<const Rational> initial)
    -> Result<Expr>;

struct GeneratingFunction {
    RationalPoly numerator;    // P(x)
    RationalPoly denominator;  // Q(x) = 1 − c₀x − c₁x² − … − c_{k-1}x^k
};

[[nodiscard]] auto generating_function(std::span<const Rational> coeffs,
                                       std::span<const Rational> initial)
    -> Result<GeneratingFunction>;
```

Both `closed_form` and `generating_function` take the same coefficient list as the
spectral functions plus an `initial` list holding **exactly** `k = coeffs.size()`
initial conditions `a₀, …, a_{k-1}` (`initial[i]` = `aᵢ`). A missing or extra
initial value is a `MathError::domain_error`.

### `characteristic_polynomial`

The monic characteristic polynomial `x^k − c_0 x^{k-1} − … − c_{k-1}` of the
order-`k` recurrence, returned as a [`RationalPoly`](ratpoly.md) of degree `k`
(the `x^k` term is `1`; the coefficient of `x^{k-1-i}` is `−c_i`). Fails
`domain_error` on empty `coeffs`, or `overflow` on the rational negation.

### `characteristic_roots`

The distinct **rational** characteristic roots, each paired with its multiplicity
(`>= 1`), obtained by running [`rational_roots`](roots.md) on the characteristic
polynomial. Roots are returned in **no particular order**. Irrational / complex
roots are not returned (see [Scope](#scope)): for a polynomial that does not split
over `Q` this yields only the rational part, possibly the empty vector.

### `all_roots_rational`

Whether the characteristic polynomial splits completely over `Q` — i.e. whether
the sum of the rational-root multiplicities equals the order `k`. When `true`,
`characteristic_roots` accounts for **every** root and the closed form is fully
expressible with rational bases. When `false`, the remaining roots are irrational
or complex (the planned-extension case).

## Error model

| Condition | Error |
| :--- | :--- |
| `coeffs` is empty (no recurrence) | `MathError::domain_error` |
| The exact rational negation of a coefficient overflows `int64` | `MathError::overflow` |

`characteristic_roots` and `all_roots_rational` both build the characteristic
polynomial first, so they propagate these conditions; any `overflow` surfaced by
the underlying [`rational_roots`](roots.md) also propagates unchanged.

## Worked examples

From `tests/recurrence_tests.cpp` (coefficient lists low-order first; the
characteristic polynomial printed low-degree-first in the `RationalPoly`
convention `coeffs[i]` = coefficient of `x^i`):

```cpp
import nimblecas.recurrence;
import nimblecas.ratpoly;
using namespace nimblecas;

// a_n = 5 a_{n-1} - 6 a_{n-2}   coeffs {5, -6}
//   char poly x^2 - 5x + 6 = (x-2)(x-3)
characteristic_roots(/* {5, -6} */).value();
//   -> { (2, 1), (3, 1) }        two simple rational roots; splits over Q

// a_n = 2 a_{n-1} - a_{n-2}     coeffs {2, -1}
//   char poly x^2 - 2x + 1 = (x-1)^2
characteristic_roots(/* {2, -1} */).value();
//   -> { (1, 2) }                double root; still splits over Q

// Fibonacci a_n = a_{n-1} + a_{n-2}   coeffs {1, 1}
//   char poly x^2 - x - 1 (golden-ratio roots, irrational)
characteristic_roots(/* {1, 1} */).value();   // -> { }  no rational roots
all_roots_rational(/* {1, 1} */).value();     // -> false (does not split over Q)

// a_n = 6 a_{n-1} - 11 a_{n-2} + 6 a_{n-3}   coeffs {6, -11, 6}
//   char poly x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
characteristic_roots(/* {6, -11, 6} */).value();
//   -> { (1, 1), (2, 1), (3, 1) }   cubic splits completely over Q

// empty coeffs describe no recurrence
characteristic_polynomial(std::span<const Rational>{}).error();  // domain_error
```

## See also

- [`nimblecas.roots`](roots.md) — the rational-root solver (rational root theorem
  plus deflation) this module runs on the characteristic polynomial; it carries
  the same irrational / `RootOf` planned-extension limitation.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Q[x]` substrate (`Rational`,
  `RationalPoly`) the characteristic polynomial is built over.
- [Documentation hub](../Index.md)
