# `nimblecas.powerseries` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/powerseries/powerseries.cppm`

Truncated formal power series over the rationals — elements of the ring
`Q[[x]] / (x^N)` (ROADMAP §7.4/7.5 dependency). A `PowerSeries` retains a fixed
number of coefficients `order = N`, representing `c_0 + c_1 x + ... + c_{N-1}
x^{N-1}` with everything of degree `>= N` discarded behind an implicit `O(x^N)`
tail. The coefficient vector always holds **exactly `N`** exact
[`Rational`](ratpoly.md)s, so every operation is **exact over `Q`** — no
floating point ever enters, and transcendental functions (`exp`, `log`) are
computed by exact coefficient recurrences, not numeric evaluation. Because the
underlying `Rational` is backed by `int64` numerators and denominators, the
honest boundary here is **overflow**: as coefficients accumulate (factorials in
`exp`, large binomials in high-order products) an `int64` numerator or
denominator can exceed its range, which surfaces as `MathError::overflow`
rather than silently wrapping (Rule 32). When a computation needs to run past
that `int64` ceiling, [`bigpowerseries`](bigpowerseries.md) lifts it to
arbitrary-precision `BigRational` coefficients with the same API.

Every binary operation requires the two operands to **share the same order**;
otherwise the result is `MathError::domain_error`. This keeps the truncation
ring `Q[[x]]/(x^N)` unambiguous.

```cpp
import nimblecas.powerseries;
```

Depends on [`core`](core.md) and [`ratpoly`](ratpoly.md).

## `PowerSeries` — an element of `Q[[x]] / (x^N)`

### Factories

Every factory rejects `order == 0` with `MathError::domain_error` (a series must
retain at least the constant term), and all return `Result<PowerSeries>`.

| Factory | Signature | Behavior |
| :--- | :--- | :--- |
| `from_coeffs` | `static auto from_coeffs(std::vector<Rational> coeffs, std::size_t order) -> Result<PowerSeries>` | Build from an explicit coefficient list, **padded with zeros or truncated** so the result holds exactly `order` coefficients (`c_0..c_{order-1}`). |
| `constant` | `static auto constant(const Rational& c, std::size_t order) -> Result<PowerSeries>` | The constant series `c + O(x^N)`. |
| `variable` | `static auto variable(std::size_t order) -> Result<PowerSeries>` | The series `x` (`c_1 = 1`). For `order == 1` this truncates to `0`. |
| `zero` | `static auto zero(std::size_t order) -> Result<PowerSeries>` | The additive identity `0 + O(x^N)`. |
| `one` | `static auto one(std::size_t order) -> Result<PowerSeries>` | The multiplicative identity `1 + O(x^N)`. |

### Accessors

These are total (infallible) — they return plain values, not `Result`.

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `order` | `auto order() const noexcept -> std::size_t` | Number of retained coefficients `N` (terms `x^0..x^{N-1}`). |
| `coefficient` | `auto coefficient(std::size_t k) const -> Rational` | Coefficient of `x^k`; returns `0` for `k >= order()`. |
| `coefficients` | `auto coefficients() const noexcept -> std::span<const Rational>` | Read-only view of all `N` coefficients (index `i` is the coefficient of `x^i`). |
| `to_string` | `auto to_string(std::string_view var = "x") const -> std::string` | Human-readable form, e.g. `"1 + 2*x + x^2 + O(x^3)"`. |
| `is_equal` | `auto is_equal(const PowerSeries& o) const -> bool` | `true` when the two series have the same order and identical coefficients. |

### Arithmetic (overflow-checked, exact)

All arithmetic flows through the overflow-checked `Rational` operations and
propagates their errors along the railway — no raw `int64` products that could
silently wrap. Each returns `Result<PowerSeries>`.

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `add` | `auto add(const PowerSeries& o) const -> Result<PowerSeries>` | Coefficient-wise sum. Requires equal order (else `domain_error`). |
| `subtract` | `auto subtract(const PowerSeries& o) const -> Result<PowerSeries>` | Coefficient-wise difference. Requires equal order (else `domain_error`). |
| `scale` | `auto scale(const Rational& s) const -> Result<PowerSeries>` | Multiply every coefficient by the scalar `s`. |
| `multiply` | `auto multiply(const PowerSeries& o) const -> Result<PowerSeries>` | Truncated Cauchy product: `(a*b)_k = sum_{i+j=k} a_i b_j` for `k < N`. Requires equal order (else `domain_error`). |
| `inverse` | `auto inverse() const -> Result<PowerSeries>` | Multiplicative inverse. Requires `c_0 != 0` (else `domain_error`). Uses the recurrence `b_0 = 1/a_0`, `b_k = -(1/a_0) sum_{i=1..k} a_i b_{k-i}`. |
| `divide` | `auto divide(const PowerSeries& o) const -> Result<PowerSeries>` | `this * o.inverse()`. Requires equal order and `o.coefficient(0) != 0` (else `domain_error`). |

### Calculus (exact formal operators)

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `derivative` | `auto derivative() const -> Result<PowerSeries>` | Formal derivative, **keeping the same order `N`** with the top coefficient set to `0` (the `x^{N-1}` term would come from the truncated `x^N` coefficient): `result_k = (k+1) c_{k+1}` for `k < N-1`, `result_{N-1} = 0`. |
| `integrate` | `auto integrate() const -> Result<PowerSeries>` | Formal integral with zero constant of integration: `result_0 = 0`, `result_k = c_{k-1}/k` for `1 <= k < N`. The old top term `c_{N-1}` is truncated away. |

Note the truncation asymmetry: `integrate` then `derivative` round-trips only
when the top coefficient is `0` (nothing lost to truncation); `derivative` then
`integrate` recovers `f` only **up to its constant term** (`c_0` is dropped to
`0`).

### Composition

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `compose` | `auto compose(const PowerSeries& g) const -> Result<PowerSeries>` | `(this ∘ g)(x) = this(g(x))`, evaluated by Horner over the truncated ring. Requires `g.coefficient(0) == 0` (else `domain_error`) so the result is a well-defined power series, and equal order (else `domain_error`). |

### Transcendental (exact coefficient recurrences over `Q`)

Both are computed by exact rational recurrences — **not** numeric evaluation —
so the coefficients are the exact Taylor coefficients as fractions.

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `exp` | `auto exp() const -> Result<PowerSeries>` | `exp(f)`. Requires `c_0 == 0` (else `domain_error`) so `e^{c_0} = 1` stays rational. Recurrence from `(exp f)' = f' exp f`: `e_0 = 1`, `e_k = (1/k) sum_{i=1..k} i f_i e_{k-i}`. |
| `log` | `auto log() const -> Result<PowerSeries>` | `log(f)`. Requires `c_0 == 1` (else `domain_error`). Computed as `integrate(f'/f)`, which gives `l_0 = 0` automatically. |

## Error model

| Condition | Error |
| :--- | :--- |
| Any factory called with `order == 0` | `MathError::domain_error` |
| Binary op (`add`, `subtract`, `multiply`, `divide`, `compose`) on mismatched orders | `MathError::domain_error` |
| `inverse` / `divide` when the constant term is `0` | `MathError::domain_error` |
| `compose` when the inner series has `g.coefficient(0) != 0` | `MathError::domain_error` |
| `exp` when `c_0 != 0` | `MathError::domain_error` |
| `log` when `c_0 != 1` | `MathError::domain_error` |
| An `int64` numerator or denominator computation wraps during any arithmetic | `MathError::overflow` |

The accessors (`order`, `coefficient`, `coefficients`, `to_string`, `is_equal`)
are total and never error. To compute past the `int64` overflow ceiling, use
[`bigpowerseries`](bigpowerseries.md).

## Worked examples

```cpp
import nimblecas.powerseries;
import nimblecas.ratpoly;
using namespace nimblecas;

auto ri = [](std::int64_t v) { return Rational::from_int(v); };
auto series = [](std::vector<std::int64_t> ints, std::size_t order) {
    std::vector<Rational> coeffs;
    for (auto v : ints) coeffs.push_back(Rational::from_int(v));
    return PowerSeries::from_coeffs(std::move(coeffs), order).value();
};

// from_coeffs pads with zeros or truncates to exactly `order`.
auto padded = PowerSeries::from_coeffs({ri(1), ri(2)}, 4).value();
padded.order();            // 4
padded.coefficient(1);     // 2
padded.coefficient(2);     // 0   (padded)
padded.coefficient(9);     // 0   (out of range)

// Truncated Cauchy product: (1 + x)^2 = 1 + 2x + x^2.
auto a       = series({1, 1, 0}, 3);
auto squared = a.multiply(a).value();
squared.coefficient(0);    // 1
squared.coefficient(1);    // 2
squared.coefficient(2);    // 1

// Every binary op requires equal order.
series({1, 1}, 2).multiply(series({1, 1, 1}, 3)).error();  // domain_error

// Inverse as a geometric series: 1/(1 - x) = 1 + x + x^2 + x^3 + x^4.
auto denom = series({1, -1, 0, 0, 0}, 5);
auto inv   = denom.inverse().value();
inv.coefficient(3);        // 1
series({0, 1, 0}, 3).inverse().error();  // domain_error (c_0 == 0)

// divide == multiply by the inverse.
series({1, 0, 0, 0}, 4).divide(series({1, -1, 0, 0}, 4)).value();  // 1 + x + x^2 + x^3

// Derivative keeps the order with a trailing zero;
// f = 1 + 2x + 3x^2 + 4x^3  ->  f' = 2 + 6x + 12x^2 (+ 0 x^3).
auto d = series({1, 2, 3, 4}, 4).derivative().value();
d.order();                 // 4
d.coefficient(0);          // 2
d.coefficient(3);          // 0   (trailing zero)

// exp(x) has the reciprocal-factorial coefficients 1, 1, 1/2, 1/6, 1/24.
auto e = series({0, 1, 0, 0, 0}, 5).exp().value();
e.coefficient(2);          // 1/2
e.coefficient(4);          // 1/24
series({1, 1, 0}, 3).exp().error();  // domain_error (c_0 != 0)

// log(1 + x): coeffs 0, 1, -1/2, 1/3, -1/4.
auto l = series({1, 1, 0, 0, 0}, 5).log().value();
l.coefficient(0);          // 0
l.coefficient(2);          // -1/2
series({2, 1, 0}, 3).log().error();  // domain_error (c_0 != 1)

// exp and log are inverse (truncated): exp(log(1 + x)) == 1 + x.
auto s        = series({1, 1, 0, 0, 0}, 5);
auto identity = s.log().value().exp().value();
identity.is_equal(s);      // true

// Composition f(g), requires g.coefficient(0) == 0.
// f(y) = 1 + y + y^2, g = x + x^2  ->  1 + x + 2x^2 + 2x^3 + ...
auto comp = series({1, 1, 1, 0}, 4).compose(series({0, 1, 1, 0}, 4)).value();
comp.coefficient(2);       // 2
comp.coefficient(3);       // 2
series({1, 1, 1}, 3).compose(series({1, 1, 0}, 3)).error();  // domain_error (g_0 != 0)
```

## See also

- [`nimblecas.bigpowerseries`](bigpowerseries.md) — the same truncated-series
  API over arbitrary-precision `BigRational` coefficients, lifting the `int64`
  overflow ceiling.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field the
  coefficients live in.
- [`nimblecas.series`](series.md) — Taylor/Laurent series expansion of symbolic
  expressions.
- [Documentation hub](../Index.md)
