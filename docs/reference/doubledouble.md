# `nimblecas.doubledouble` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/doubledouble/doubledouble.cppm`

Extended-precision floating point: a real number carried as the unevaluated sum
`hi + lo` of two IEEE-754 `binary64` doubles held under the invariant
`|lo| <= 0.5·ulp(hi)`. Because `hi` already carries 53 significand bits and `lo`
the next ~53, the pair models a **~106-bit significand** (≈ 31–32 decimal digits)
at a small multiple of `double` cost, built on error-free transforms (EFTs). It
sits between plain `double` and the exact rational tower: reach for it when you
want **more precision, still fast**, not when you need proof-grade exactness.

The honesty boundary is sharp and worth stating up front:

- **~106-bit, NOT IEEE binary128 / "quad".** There is no native 128-bit hardware
  float on x86, so double-double + EFTs is the fast, honest route. Only the
  **significand** is extended; the exponent range stays `binary64`'s (~1e±308).
- **Still floating point, hence still inexact.** Rounding happens; results are
  faithful approximations, not exact. When you need arithmetic with **zero
  rounding**, use the exact rational path instead — [`Rational`](ratpoly.md)
  (module `nimblecas.ratpoly`) is closed under `+`, `-`, `*`, `/` with no
  rounding, and the exact [`Complex`](complex.md) / rational layers build on it.
  Double-double is for precision, not for "provably exact".
- **SIMD accelerates BATCHES, not single ops.** AVX2/AVX-512 runs the EFTs of
  many array elements in parallel; it does **not** make one scalar add "128-bit"
  — one add is one add. The payoff lives in the batched kernels (`dd_sum` /
  `dd_dot` / `dd_poly_eval`).
- **Bit-identical scalar == SIMD.** The batched kernels use a fixed 4-lane
  (256-bit) reduction layout so the result is reproducible across machines and
  bit-for-bit identical whether the scalar reference or the SIMD path runs.
  AVX-512 hosts still run the 256-bit kernel to preserve that 4-lane layout,
  trading a little width for exact reproducibility.

```cpp
import nimblecas.doubledouble;
```

Depends on [`core`](core.md) (the `Result` / `MathError` railway) and
[`simd`](simd.md) (runtime CPU-feature detection for the batched dispatch).

## `DoubleDouble` — a real number as the unevaluated sum `hi + lo`

An **aggregate** (no user-declared constructors, public `double hi`, `double lo`)
so it can be brace-initialised as `{hi, lo}` and lives in `std::array` / SIMD
spill buffers. The invariant `|lo| <= 0.5·ulp(hi)`, maintained by every operation,
makes `(hi, lo)` a **canonical, order-preserving** pair: numeric ordering equals
lexicographic ordering on `(hi, lo)`, so the comparisons below are the correct
numeric ones.

### Construction and conversion

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `from_double` | `static constexpr auto from_double(double d) noexcept -> DoubleDouble` | The double `d` as the exact double-double `d + 0`. |
| `to_double` | `constexpr auto to_double() const noexcept -> double` | Nearest double (the leading component `hi`). Loses the extra ~53 bits carried in `lo`. |
| aggregate init | `DoubleDouble{hi, lo}` | Brace-initialise the two members directly (public data). |

### Predicates and comparison

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `is_zero` | `constexpr auto is_zero() const noexcept -> bool` | `true` when both `hi` and `lo` are `0.0`. |
| `operator==` | `auto operator==(const DoubleDouble& o) const noexcept -> bool` | Structural (`hi == o.hi && lo == o.lo`); coincides with numeric equality because the representation is canonical. |
| `operator<=>` | `auto operator<=>(const DoubleDouble& o) const noexcept -> std::partial_ordering` | Lexicographic on `(hi, lo)` = numeric ordering. NaN parts yield `std::partial_ordering::unordered`, matching IEEE semantics. |

### Field operations

Add / subtract / multiply / `negate` **cannot fail** (no domain, no zero divisor)
and return plain values; divide / sqrt are railway-typed (Rule 32) and return
`Result<DoubleDouble>`.

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `negate` | `constexpr auto negate() const noexcept -> DoubleDouble` | The additive inverse `-(hi + lo)`; never rounds, so it cannot fail. |
| `add` | `auto add(const DoubleDouble& o) const noexcept -> DoubleDouble` | Accurate `dd + dd` (Hida/Li/Bailey `ieee_add`). |
| `subtract` | `auto subtract(const DoubleDouble& o) const noexcept -> DoubleDouble` | `dd - dd`. |
| `multiply` | `auto multiply(const DoubleDouble& o) const noexcept -> DoubleDouble` | `dd * dd`; the sub-resolution `a.lo·b.lo` cross term is dropped. |
| `divide` | `auto divide(const DoubleDouble& o) const -> Result<DoubleDouble>` | Faithful ~106-bit quotient (three long-division correction steps). Fails `division_by_zero` when `o` is exactly zero. |
| `sqrt` | `auto sqrt() const -> Result<DoubleDouble>` | ~106-bit square root via one Newton step on `1/sqrt`. Fails `domain_error` for a negative operand; `sqrt(0) == 0`. The real square root of a negative belongs to the complex layer, not here. |

### Rendering

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `to_string` | `auto to_string(int precision = 31) const -> std::string` | Decimal string with enough significant digits to expose the extra precision — e.g. the double-double `1/3` shows ~31 threes, not a double's ~16. Emits scientific form (`d.dddde±NN`); `"nan"` / `"inf"` / `"-inf"` for non-finite `hi`, `"0"` for zero. `precision` is clamped to at least 1. |

## Error-free transforms (EFTs)

The primitives everything else is built from. Each returns a `DoubleDouble`
`{s, e}` where `s` is the rounded result and `e` is the **exact** rounding error,
so the pair reconstructs the true value with no loss (`a op b == s + e` exactly,
for representable, non-overflowing inputs). The returned pair already satisfies
the invariant `|e| <= 0.5·ulp(s)`. These are free functions in namespace
`nimblecas`; none can fail.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `two_sum` | `auto two_sum(double a, double b) noexcept -> DoubleDouble` | Knuth's TwoSum: `a + b == s + e` exactly, for **any** finite `a, b` (6 flops). |
| `quick_two_sum` | `auto quick_two_sum(double a, double b) noexcept -> DoubleDouble` | Dekker's quick-TwoSum: `a + b == s + e` exactly, but **only valid when `|a| >= |b|`** (or `a == 0`). Cheaper (3 flops); used after a renormalising add where the magnitude order is known. |
| `two_prod` | `auto two_prod(double a, double b) noexcept -> DoubleDouble` | `a * b == p + e` exactly. Uses one fused `std::fma` to capture the product error when hardware FMA is present, and falls back to Dekker's splitting TwoProduct when it is not — **both yield the identical exact `e`**, so results are bit-identical either way. |

## SIMD-batched reductions — the vectorisation payoff

Each reduction has a **scalar reference** (the `*_scalar` overload) **and** a
runtime-dispatched entry point that vectorises the EFTs across the fixed 4-lane
layout. The two paths are bit-for-bit identical (asserted in the tests, Rule 55);
the dispatched path is chosen by reusing `nimblecas.simd`'s CPU-feature detection
and **falls back to the scalar reference** when the AVX2+FMA baseline is
unavailable — so every result is correct and reproducible everywhere. All are
`noexcept` and infallible (empty input is well-defined).

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `dd_sum` | `auto dd_sum(std::span<const double> x) noexcept -> DoubleDouble` | Compensated summation to a double-double; recovers bits a naive `double` sum drops (e.g. `{1e16, 1, -1e16}` sums to `1`, not `0`). Dispatched. |
| `dd_sum_scalar` | `auto dd_sum_scalar(std::span<const double> x) noexcept -> DoubleDouble` | The scalar reference for `dd_sum`. |
| `dd_dot` | `auto dd_dot(std::span<const double> a, std::span<const double> b) noexcept -> DoubleDouble` | Compensated dot product `Σ a[i]·b[i]` (each product exact via `two_prod`). Uses `min(a.size(), b.size())` elements. Dispatched. |
| `dd_dot_scalar` | `auto dd_dot_scalar(std::span<const double> a, std::span<const double> b) noexcept -> DoubleDouble` | The scalar reference for `dd_dot`. |
| `dd_poly_eval` | `auto dd_poly_eval(std::span<const double> coeffs, double x) noexcept -> DoubleDouble` | Compensated polynomial `P(x) = Σ coeffs[i]·xⁱ` (`coeffs[i]` is the coefficient of `xⁱ`) via a 4-way interleaved Horner scheme. Empty `coeffs` is the zero function. Dispatched. |
| `dd_poly_eval_scalar` | `auto dd_poly_eval_scalar(std::span<const double> coeffs, double x) noexcept -> DoubleDouble` | The scalar reference for `dd_poly_eval`. |
| `batched_backend` | `auto batched_backend() noexcept -> std::string_view` | Which path the dispatched kernels use on this host — `"scalar"` or `"simd(256-bit AVX2+FMA, 4-lane)"` — for diagnostics/tests. |

> To evaluate **one** polynomial at **many** points at once (rather than one
> polynomial at one point in extended precision), see
> [`nimblecas.simd`](simd.md)'s `horner_step` instead.

## Error model

Only `divide` and `sqrt` are fallible; everything else is total.

| Condition | Error |
| :--- | :--- |
| `divide` by an exactly-zero double-double (`o.is_zero()`) | `MathError::division_by_zero` |
| `sqrt` of a negative operand (`hi < 0`, or `hi == 0 && lo < 0`) | `MathError::domain_error` |

`sqrt(0)` returns `0` (not an error). There is **no overflow error**: the
exponent range is `binary64`'s, so an out-of-range magnitude follows IEEE
semantics (`±inf`, `nan`) rather than the railway — `to_string` renders those as
`"inf"` / `"-inf"` / `"nan"`. `add` / `subtract` / `multiply` / `negate` /
`to_double` / `is_zero` / the comparisons and the EFTs / batched reductions are
total and never error.

## Worked examples

```cpp
import nimblecas.doubledouble;
using namespace nimblecas;

// Error-free transforms recover the exact rounding error a double drops.
// 1e-20 is far below 0.5·ulp(1) ≈ 1.1e-16, so a plain double loses it...
(1.0 + 1e-20 == 1.0);                       // true (double drops it)
const DoubleDouble s = two_sum(1.0, 1e-20); // ...but two_sum keeps it: {1.0, 1e-20}
two_prod(3.0, 7.0);                          // {21.0, 0.0}  (exact product)
// two_prod is exact by definition: lo == std::fma(a, b, -hi).

// Extra precision a plain double cannot hold: 1 + 1e-18.
const DoubleDouble d = two_sum(1.0, 1e-18);
(d != DoubleDouble::from_double(1.0));       // true — the 1e-18 is retained in lo
d.subtract(DoubleDouble::from_double(1.0));  // recovers ≈ 1e-18

// Compensated reductions beat naive summation on ill-conditioned input.
const std::array<double, 3> x{1e16, 1.0, -1e16};   // true sum is 1
dd_sum(x).to_double();                       // 1.0 (a naive double loop gives 0)
const std::array<double, 3> a{1e16, 1.0, -1e16};
const std::array<double, 3> b{1.0, 1.0, 1.0};      // true dot is 1
dd_dot(a, b).to_double();                    // 1.0

// The dispatched SIMD path is bit-for-bit identical to the scalar reference.
dd_sum(x) == dd_sum_scalar(x);               // true (Rule 55)

// Polynomial evaluation: 1 - 2x + x^2 = (x - 1)^2, coeffs indexed by power.
const std::array<double, 3> c{1.0, -2.0, 1.0};
dd_poly_eval(c, 3.0).to_double();            // 4.0  = (3-1)^2
dd_poly_eval(std::span<const double>{}, 5.0).is_zero();  // true (empty = zero fn)

// sqrt and divide reach ~106 bits (≈ 30 correct digits) and are railway-typed.
const auto root = DoubleDouble::from_double(2.0).sqrt();
root.value().multiply(root.value());         // ≈ 2 to ~1e-31
const auto third = DoubleDouble::from_double(1.0)
                       .divide(DoubleDouble::from_double(3.0));
third.value().to_string();                   // "3.333333333333333333...e-01" (>19 threes)

// Domain errors surface as MathError, never as exceptions.
DoubleDouble::from_double(1.0)
    .divide(DoubleDouble::from_double(0.0)).error();  // MathError::division_by_zero
DoubleDouble::from_double(-1.0).sqrt().error();       // MathError::domain_error
DoubleDouble::from_double(0.0).sqrt().value();        // {0.0, 0.0}  (sqrt(0) == 0)

// Which batched path is live on this host.
batched_backend();                           // "scalar" or "simd(256-bit AVX2+FMA, 4-lane)"
```

## See also

- [`nimblecas.core`](core.md) — the `Result` / `MathError` railway that
  `divide` and `sqrt` thread through.
- [`nimblecas.simd`](simd.md) — the CPU-feature detection reused for batched
  dispatch, and `horner_step` for one-polynomial-many-points evaluation.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field to reach for
  when you need zero rounding instead of extended precision.
- [`nimblecas.complex`](complex.md) — the exact complex layer that owns the
  square root of a negative number.
- [Documentation hub](../Index.md)
