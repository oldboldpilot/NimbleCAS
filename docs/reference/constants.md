# `nimblecas.constants` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/constants/constants.cppm`

Arbitrary-precision **classical constants of analysis** — π, Euler's number
`e`, the Euler–Mascheroni constant γ, `ln2`, `ln10`, `sqrt2`, the golden ratio
φ, and Catalan's constant `G`. This is the **numeric high-precision provider**
in the tower: it sits above `nimblecas.bigfloat` (which supplies the
correctly-rounded arbitrary-precision float arithmetic) and hands each constant
back as a `BigFloat`. The honest boundary is the whole point of the module:
every one of these numbers is **transcendental or irrational**, so **none has an
exact `BigInt`/`Rational` representation** — there is no exact answer to return.
The deliverable is a `BigFloat` **correctly rounded to the caller's requested
precision**, i.e. a rounded floating-point value, **not** an exact number. A
*symbolic* π or `e` — a leaf that stays exact inside an expression tree —
belongs to the symbolic layer ([`nimblecas.symbolic`](symbolic.md)), **not**
here.

```cpp
import nimblecas.constants;
```

Depends on [`core`](core.md), `nimblecas.bigint`, and `nimblecas.bigfloat`.

## Precision discipline — what "`prec` bits" means

Every entry point takes a single argument `prec`, the number of requested
**significant bits**, and returns a `BigFloat` accurate to roughly that many
bits. Internally the computation is carried at an **elevated working precision**
`work = prec + 32 + 4·bit_width(prec) + 16` (where `bit_width(prec) = ⌊log₂ prec⌋ + 1`):
the fixed 32 guard bits absorb the
single final rounding, and the log-sized slack absorbs the round-off
accumulated across the `O(prec)` summed series terms. The result is rounded
**once** to `prec` (via `BigFloat::with_precision`) at the public boundary, so
leading digits stay stable as `prec` rises — but the value is still a rounded
float, never exact.

Each series shares an explicit **stopping rule**: a threshold `ε = 2^-work` is
formed, and summation stops as soon as the magnitude of the newly added term
falls below `ε` (at that point the untruncated tail is below the working ULP and
cannot change the rounded result). The Brent–McMillan sum for γ is the one
exception: its terms **rise** to a peak near `k = N` before decaying, so its
stop test additionally requires `k > N`. A generous per-series iteration cap
(`16·work + 100`) guards only against a pathological non-terminating loop, never
against correctness — every series converges long before it.

## Algorithms

| Constant | Algorithm |
| :--- | :--- |
| `e` | `Σ_{n≥0} 1/n!`, the reciprocal factorial maintained by dividing the previous term by `n`. |
| `pi` | Machin's formula `π = 16·arctan(1/5) − 4·arctan(1/239)`, each `arctan(1/x) = Σ (−1)ⁿ/((2n+1)·x^(2n+1))`. |
| `ln2` | `2·atanh(1/3) = 2·Σ 1/((2n+1)·3^(2n+1))`. |
| `ln10` | `3·ln2 + 2·atanh(1/9)` = `ln8 + ln(5/4)` = `ln10` exactly. |
| `sqrt2` | `BigFloat::sqrt(2)` (correctly-rounded integer-sqrt of the scaled mantissa). |
| `golden_ratio` | `(1 + sqrt(5)) / 2` — algebraic, via `BigFloat::sqrt`. |
| `catalan` | `G = (π/8)·ln(2+√3) + (3/8)·Σ 1/((2n+1)²·C(2n,n))`; the binomial series converges ~2 bits/term, and `ln(2+√3)` is evaluated as `(2/√3)·Σ 1/((2n+1)·3ⁿ)`. |
| `euler_mascheroni` | Brent–McMillan **B1**: with `N ~ ⌈work·ln2/4⌉`, `γ = S/I − ln(N)` where `t_k = (Nᵏ/k!)²`, `I = Σ t_k`, `S = Σ t_k·H_k`, `H_k` the harmonic numbers. |

## API — the eight constant functions

Every entry point is an exported free function in namespace `nimblecas` with the
same shape: it takes the requested precision and returns a `Result<BigFloat>`.
There are no overloads and no other exported symbols (the series kernels,
`working_precision`, `epsilon_at`, `atanh_recip`, `arctan_recip`, `ln_int`,
`gamma_val`, and the rest live in an anonymous namespace and are **not** public).

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `e` | `[[nodiscard]] auto e(std::int64_t prec) -> Result<BigFloat>` | Euler's number `e` to ~`prec` significant bits. |
| `pi` | `[[nodiscard]] auto pi(std::int64_t prec) -> Result<BigFloat>` | π to ~`prec` significant bits (Machin). |
| `ln2` | `[[nodiscard]] auto ln2(std::int64_t prec) -> Result<BigFloat>` | Natural log of 2. |
| `ln10` | `[[nodiscard]] auto ln10(std::int64_t prec) -> Result<BigFloat>` | Natural log of 10. |
| `sqrt2` | `[[nodiscard]] auto sqrt2(std::int64_t prec) -> Result<BigFloat>` | Square root of 2. |
| `golden_ratio` | `[[nodiscard]] auto golden_ratio(std::int64_t prec) -> Result<BigFloat>` | The golden ratio φ = `(1+√5)/2`. |
| `catalan` | `[[nodiscard]] auto catalan(std::int64_t prec) -> Result<BigFloat>` | Catalan's constant `G`. |
| `euler_mascheroni` | `[[nodiscard]] auto euler_mascheroni(std::int64_t prec) -> Result<BigFloat>` | The Euler–Mascheroni constant γ (Brent–McMillan B1). |

Each computes at `working_precision(prec)`, then rounds once to `prec`. `prec`
must be `> 0` (see below).

## Error model

Following the rest of the engine, these functions are **railway-oriented**
(Rule 32): they never throw, and every underlying `BigFloat` `Result` error is
propagated unchanged rather than swallowed.

| Condition | Error |
| :--- | :--- |
| `prec <= 0` (non-positive requested precision) | `MathError::domain_error` |
| An underlying `BigFloat` arithmetic step fails (e.g. an `int64` exponent/limb computation wraps at very large `prec`) | that same `MathError` (e.g. `MathError::overflow`), propagated |

There is no `division_by_zero` path from the public surface — every series
denominator is a positive integer constant. The `prec <= 0` guard is checked
**before** any computation, so an invalid precision fails immediately with
`domain_error`.

## Worked examples

```cpp
import nimblecas.constants;
import nimblecas.bigfloat;
using namespace nimblecas;

// Each constant to ~256 bits, correctly rounded. Leading fractional digits match
// the well-known reference values.
pi(256).value().to_string(45);
// "3.14159265358979323846264338327950288..."
e(256).value().to_string(45);
// "2.71828182845904523536028747135266249..."
euler_mascheroni(256).value().to_string(45);
// "0.57721566490153286060651209008240243..."  (Brent-McMillan B1)
ln2(256).value().to_string(45);   // "0.69314718055994530941723212145817656..."
ln10(256).value().to_string(45);  // "2.30258509299404568401799145468436420..."
sqrt2(256).value().to_string(45); // "1.41421356237309504880168872420969807..."
golden_ratio(256).value().to_string(45); // "1.61803398874989484820458683436563811..."
catalan(256).value().to_string(45);      // "0.91596559417721901505460351493238411..."

// Raising the precision does not move the leading digits (stability).
const auto pi128 = pi(128).value().to_string(40).substr(0, 32);
const auto pi512 = pi(512).value().to_string(40).substr(0, 32);
// pi128 == pi512   (first 30 fractional digits identical)

// Defining identities hold once the shared rounding slack is dropped.
const std::int64_t prec = 256;
const auto r  = sqrt2(prec).value();
const auto sq = r.multiply(r, prec).value();
sq.with_precision(prec - 8).value() == BigFloat::from_i64(2, prec).value()
                                          .with_precision(prec - 8).value();  // true

const auto phi   = golden_ratio(prec).value();
const auto phiSq = phi.multiply(phi, prec).value();
const auto one   = BigFloat::from_i64(1, prec).value();
const auto phiP1 = phi.add(one, prec).value();
phiSq.with_precision(prec - 8).value() == phiP1.with_precision(prec - 8).value(); // phi^2 == phi + 1

// Non-positive precision is a domain error on every entry point.
pi(0).error();               // MathError::domain_error
e(-5).error();               // MathError::domain_error
euler_mascheroni(0).error(); // MathError::domain_error
catalan(-1).error();         // MathError::domain_error
```

## See also

- [`nimblecas.core`](core.md) — the `Result<T>` / `MathError` railway these
  functions return through.
- `nimblecas.bigfloat` — the correctly-rounded arbitrary-precision float type
  that carries every result, and whose `sqrt` / `with_precision` this module
  builds on.
- [`nimblecas.symbolic`](symbolic.md) — where a *symbolic* π or `e` (an exact
  leaf inside an expression tree) lives, as opposed to these numeric values.
- [`nimblecas.numeric`](numeric.md) — the sibling floating-point (root-finding)
  numeric layer.
- [Documentation hub](../Index.md)
