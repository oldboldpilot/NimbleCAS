# `nimblecas.bigpowerseries` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/bigpowerseries/bigpowerseries.cppm`

Truncated formal power series over the rationals — an element of `Q[[x]] / (x^N)`
(ROADMAP §7.3). A `BigPowerSeries` retains a fixed number of coefficients
`order = N`, representing `c_0 + c_1 x + … + c_{N-1} x^{N-1}` with everything of
degree `≥ N` discarded (an implicit `O(x^N)` tail). The coefficient vector always
holds exactly `N` `BigRational`s, and every binary operation requires the two
operands to share the same order — otherwise `MathError::domain_error` — which
keeps the truncation ring unambiguous.

This is the **exact, unbounded** counterpart of `nimblecas.powerseries`. That
module carries its coefficients as the `int64`-backed `Rational` and therefore
reports `MathError::overflow` once a numerator or denominator no longer fits
`int64` — which happens quickly for the transcendental series (the `1/n!`
coefficients of `exp` overflow the `int64` denominator at `n ≥ 21`, since
`21! ≈ 5.1e19 > INT64_MAX`). `BigPowerSeries` removes that ceiling: coefficients
are `BigRational` over the arbitrary-precision `BigInt`, so **nothing overflows**
and the whole truncated series is exact over `Q` — `exp(x)` `coefficient(25)`
returns `1/25!` exactly. Because `BigRational` add / subtract / multiply / negate
are **infallible** (arbitrary precision cannot overflow), coefficient combining
never needs an overflow railway; only the divide / inverse / `exp` / `log` paths
(which can divide by zero) and the domain guards propagate a `MathError`
(Rule 32).

**Honesty — the slow-but-exact tier.** Each coefficient op heap-allocates `BigInt`
limbs, so this is materially slower than the `int64` `nimblecas.powerseries`.
Small, low-order work that comfortably fits `int64` should prefer that module
(faster, but bounded — it returns `MathError::overflow` at high order); reach for
`BigPowerSeries` precisely when the exact unbounded range is required.

```cpp
import nimblecas.bigpowerseries;
```

Depends on [`core`](core.md), `nimblecas.bigint` (arbitrary-precision integers),
and `nimblecas.bigrational` (the exact `Q` field the coefficients live in).

## The overflow contract (there isn't one)

Unlike the `int64` tiers, `BigPowerSeries` arithmetic **cannot overflow**: the
coefficients are `BigRational`, whose add / subtract / multiply / negate grow
`BigInt` limbs on demand. The infallible-vs-fallible split is therefore sharp:

- **Total (never error):** `order`, `coefficient`, `coefficients`, `to_string`,
  `is_equal`, `operator==` — these return plain values, not `Result`.
- **Fallible only on order/domain:** every operation that returns
  `Result<BigPowerSeries>` can fail **only** for a domain reason — a zero order
  at construction, an order mismatch between operands, or a violated precondition
  (`c_0 ≠ 0` for `inverse`, `g_0 = 0` for `compose`, `c_0 = 0` for `exp`,
  `c_0 = 1` for `log`). No arithmetic path wraps or divides by zero once those
  guards pass. `scale`, `derivative`, and `integrate` return `Result` to stay
  uniform with the railway but in practice never take the error branch.

## `BigPowerSeries` — a truncated power series `c_0 + … + c_{N-1} x^{N-1} + O(x^N)`

### Factories

Every factory rejects `order == 0` with `MathError::domain_error` (a truncation
ring modulo `x^0` is empty).

| Factory | Signature | Behavior |
| :--- | :--- | :--- |
| `from_coeffs` | `[[nodiscard]] static auto from_coeffs(std::vector<BigRational> coeffs, std::size_t order) -> Result<BigPowerSeries>` | Build from an explicit coefficient list, **padded with zeros or truncated** so the result holds exactly `order` coefficients `c_0…c_{order-1}`. |
| `constant` | `[[nodiscard]] static auto constant(const BigRational& c, std::size_t order) -> Result<BigPowerSeries>` | The constant series `c + O(x^N)`. |
| `variable` | `[[nodiscard]] static auto variable(std::size_t order) -> Result<BigPowerSeries>` | The series `x` (`c_1 = 1`). For `order == 1` the `x` term truncates away to `0`. |
| `zero` | `[[nodiscard]] static auto zero(std::size_t order) -> Result<BigPowerSeries>` | The additive identity `0 + O(x^N)`. |
| `one` | `[[nodiscard]] static auto one(std::size_t order) -> Result<BigPowerSeries>` | The multiplicative identity `1 + O(x^N)`. |

### Accessors

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `order` | `[[nodiscard]] auto order() const noexcept -> std::size_t` | Number of retained coefficients `N` (terms `x^0…x^{N-1}`). |
| `coefficient` | `[[nodiscard]] auto coefficient(std::size_t k) const -> BigRational` | Coefficient of `x^k`; returns `0` for `k ≥ order()`. |
| `coefficients` | `[[nodiscard]] auto coefficients() const noexcept -> std::span<const BigRational>` | Read-only view of all `N` coefficients (index `i` is the coefficient of `x^i`). |
| `to_string` | `[[nodiscard]] auto to_string(std::string_view var = "x") const -> std::string` | Human-readable form, e.g. `"1 + 2*x + x^2 + O(x^3)"`. |
| `is_equal` | `[[nodiscard]] auto is_equal(const BigPowerSeries& o) const -> bool` | Same order **and** identical coefficients. |
| `operator==` | `[[nodiscard]] auto operator==(const BigPowerSeries& o) const -> bool` | Alias for `is_equal`. |

### Arithmetic

All binary operators except `scale` require **equal order** (else
`MathError::domain_error`).

```cpp
[[nodiscard]] auto add(const BigPowerSeries& o) const -> Result<BigPowerSeries>;
[[nodiscard]] auto subtract(const BigPowerSeries& o) const -> Result<BigPowerSeries>;
[[nodiscard]] auto scale(const BigRational& s) const -> Result<BigPowerSeries>;
[[nodiscard]] auto multiply(const BigPowerSeries& o) const -> Result<BigPowerSeries>;
[[nodiscard]] auto inverse() const -> Result<BigPowerSeries>;
[[nodiscard]] auto divide(const BigPowerSeries& o) const -> Result<BigPowerSeries>;
```

| Method | Behavior |
| :--- | :--- |
| `add` / `subtract` | Coefficient-wise sum / difference. |
| `scale` | Multiply every coefficient by the scalar `s`. Takes no other operand, so it is order-agnostic and never errors. |
| `multiply` | Truncated **Cauchy product**: `(a·b)_k = Σ_{i+j=k} a_i b_j` for `k < N`. |
| `inverse` | Multiplicative inverse via `b_0 = 1/a_0`, `b_k = −(1/a_0) Σ_{i=1..k} a_i b_{k-i}`. Requires `c_0 ≠ 0` (else `domain_error`). |
| `divide` | `this · o.inverse()`. Requires equal order and `o.coefficient(0) ≠ 0` (the latter surfaces as `domain_error`, propagated from `inverse`). |

### Calculus

```cpp
[[nodiscard]] auto derivative() const -> Result<BigPowerSeries>;
[[nodiscard]] auto integrate() const -> Result<BigPowerSeries>;
```

| Method | Behavior |
| :--- | :--- |
| `derivative` | Formal derivative that **keeps the same order** `N`: `result_k = (k+1) c_{k+1}` for `k < N-1`, and `result_{N-1} = 0` (the `x^{N-1}` term would come from the truncated `x^N` coefficient, so it is set to zero by convention). |
| `integrate` | Formal integral with zero constant of integration: `result_0 = 0`, `result_k = c_{k-1}/k` for `1 ≤ k < N` (the old top term `c_{N-1}` is truncated away). |

`derivative` then `integrate` (or vice versa) round-trips exactly only when the
truncated-away term is zero — see the worked examples.

### Composition

```cpp
[[nodiscard]] auto compose(const BigPowerSeries& g) const -> Result<BigPowerSeries>;
```

`(this ∘ g)(x) = this(g(x))`, evaluated by Horner over the truncated ring.
Requires **equal order** and `g.coefficient(0) == 0` (else `domain_error`) so the
composition is a well-defined power series — a non-zero inner constant term would
mix infinitely many degrees into each output coefficient.

### Transcendental (exact coefficient recurrences over `Q`)

```cpp
[[nodiscard]] auto exp() const -> Result<BigPowerSeries>;
[[nodiscard]] auto log() const -> Result<BigPowerSeries>;
```

| Method | Behavior |
| :--- | :--- |
| `exp` | `exp(f)`. Requires `c_0 == 0` (else `domain_error`) so the constant term `e^{c_0} = 1` stays rational. Recurrence from `(exp f)' = f' exp f`: `e_0 = 1`, `e_k = (1/k) Σ_{i=1..k} i·f_i·e_{k-i}`. |
| `log` | `log(f)`. Requires `c_0 == 1` (else `domain_error`). Computed as `integrate(f'/f)`, which forces `l_0 = 0`. |

Because coefficients are `BigRational`, both recurrences stay exact to arbitrary
order: `exp(x)` yields `coefficient(n) = 1/n!` for every `n < N`, including the
`n ≥ 21` range where `n!` no longer fits `int64` and the `int64`
`nimblecas.powerseries` overflows.

## Error model

The only `MathError` these operations produce is `domain_error`; nothing here can
overflow, and the internal divide guards ensure no `division_by_zero` reaches the
caller once the documented precondition holds.

| Condition | Error |
| :--- | :--- |
| `from_coeffs` / `constant` / `variable` / `zero` / `one` with `order == 0` | `MathError::domain_error` |
| `add` / `subtract` / `multiply` / `divide` / `compose` with mismatched `order()` | `MathError::domain_error` |
| `inverse` when `coefficient(0) == 0` | `MathError::domain_error` |
| `divide` when `o.coefficient(0) == 0` (propagated from `inverse`) | `MathError::domain_error` |
| `compose` when `g.coefficient(0) != 0` | `MathError::domain_error` |
| `exp` when `coefficient(0) != 0` | `MathError::domain_error` |
| `log` when `coefficient(0) != 1` | `MathError::domain_error` |

`order`, `coefficient`, `coefficients`, `to_string`, `is_equal`, and `operator==`
are total and never error. There is no `overflow` branch: arbitrary-precision
coefficients cannot wrap.

## Worked examples

```cpp
import nimblecas.bigpowerseries;
import nimblecas.bigrational;
import nimblecas.bigint;
using namespace nimblecas;

auto bri = [](std::int64_t v) { return BigRational::from_int(v); };

// Construction pads or truncates to exactly `order` coefficients.
auto padded = BigPowerSeries::from_coeffs({bri(1), bri(2)}, 4).value();
padded.order();                       // 4
padded.coefficient(2).is_zero();      // true  (padded with 0)
padded.coefficient(9).is_zero();      // true  (out of range -> 0)
BigPowerSeries::from_coeffs({bri(1)}, 0).error();  // MathError::domain_error

// Cauchy product: (1 + x)^2 = 1 + 2x + x^2.
auto ompx = BigPowerSeries::from_coeffs({bri(1), bri(1), bri(0)}, 3).value();
auto sq   = ompx.multiply(ompx).value();
sq.coefficient(1);                    // 2

// Mismatched orders are a domain error.
auto o2 = BigPowerSeries::from_coeffs({bri(1), bri(1)}, 2).value();
auto o3 = BigPowerSeries::from_coeffs({bri(1), bri(1), bri(1)}, 3).value();
o2.multiply(o3).error();              // MathError::domain_error

// Geometric series to high order: 1/(1 - x) = 1 + x + x^2 + … (all ones).
auto denom = BigPowerSeries::from_coeffs({bri(1), bri(-1)}, 40).value();
auto geo   = denom.inverse().value();
geo.coefficient(39);                  // 1
BigPowerSeries::from_coeffs({bri(0), bri(1), bri(0)}, 3).value()
    .inverse().error();               // MathError::domain_error (c_0 == 0)

// divide == multiply by inverse.
auto one = BigPowerSeries::from_coeffs({bri(1), bri(0), bri(0), bri(0)}, 4).value();
auto omx = BigPowerSeries::from_coeffs({bri(1), bri(-1), bri(0), bri(0)}, 4).value();
one.divide(omx).value().coefficient(3);   // 1

// Derivative keeps order N with a trailing zero; integrate round-trips when the
// top coefficient is already zero.
auto f = BigPowerSeries::from_coeffs({bri(1), bri(2), bri(3), bri(4)}, 4).value();
auto d = f.derivative().value();
d.coefficient(2);                     // 12  (= 3 * 4)
d.coefficient(3).is_zero();           // true (trailing-zero convention)

// exp(x): coefficient(n) = 1/n! EXACTLY, past the int64 overflow point n >= 21.
auto x  = BigPowerSeries::variable(30).value();
auto ex = x.exp().value();
ex.coefficient(2);                    // 1/2!
ex.coefficient(25).numerator().to_string();    // "1"
ex.coefficient(25).denominator().to_string();  // "15511210043330985984000000"  (= 25!)
BigPowerSeries::from_coeffs({bri(1), bri(1), bri(0)}, 3).value()
    .exp().error();                   // MathError::domain_error (c_0 != 0)

// log(1 + x): coefficient(n) = (-1)^{n+1}/n; and exp(log(1+x)) == 1 + x.
auto opx = BigPowerSeries::from_coeffs({bri(1), bri(1)}, 20).value();
auto lg  = opx.log().value();
lg.coefficient(1);                    // 1
lg.exp().value().is_equal(opx);       // true
BigPowerSeries::from_coeffs({bri(2), bri(1), bri(0)}, 3).value()
    .log().error();                   // MathError::domain_error (c_0 != 1)

// Composition: (1/(1-x)) ∘ x^2 == 1/(1-x^2): even coeffs 1, odd coeffs 0.
auto geo12 = BigPowerSeries::from_coeffs({bri(1), bri(-1)}, 12).value()
                 .inverse().value();  // 1/(1-x) at order 12
std::vector<BigRational> gc(12);
gc[2] = bri(1);                       // g = x^2 (g_0 = 0)
auto g    = BigPowerSeries::from_coeffs(std::move(gc), 12).value();
auto comp = geo12.compose(g).value();  // even coeffs 1, odd coeffs 0
BigPowerSeries::from_coeffs({bri(1), bri(1), bri(0)}, 3).value()
    .compose(BigPowerSeries::from_coeffs({bri(1), bri(1), bri(0)}, 3).value())
    .error();                         // MathError::domain_error (inner g_0 != 0)
```

## See also

- [`nimblecas.core`](core.md) — the `Result` / `MathError` railway these
  operations thread through.
- `nimblecas.powerseries` — the faster `int64`-backed sibling this module mirrors
  exactly, bounded by `MathError::overflow` at high order.
- [`nimblecas.series`](series.md) — the symbolic Taylor-expansion layer built over
  the differentiation and simplification engines.
- [Documentation hub](../Index.md)
