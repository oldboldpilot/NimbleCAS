# `nimblecas.simd` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/simd/simd.cppm`

A multi-register SIMD engine for elementwise `float32` array kernels, with
**runtime dynamic dispatch** to the best instruction set the CPU supports. It is
the numeric fast path behind, for example,
[`Polynomial::evaluate_batch`](polynomial.md).

```cpp
import nimblecas.simd;
```

Namespace: `nimblecas::simd`. Depends only on `std`.

## Dispatch model

At process start-up a single CPU capability query (`__builtin_cpu_supports`)
picks the ISA, waterfalling **AVX-512 → AVX2(+FMA) → AVX → scalar**. A function-
pointer dispatch table, resolved once at static-init time, routes every kernel
call to the chosen implementation. AVX-512 paths are compiled per-function with
`[[gnu::target("avx512f")]]`, so the binary stays runnable on machines without
AVX-512 (Rule 50).

The canonical build baseline is **x86-64-v3 (AVX2+FMA)**, so AVX2 is the
practical floor and the scalar path is the portable correctness fallback.

### Determinism (Rule 55)

All kernels are strictly **elementwise** (`out[i] = f(a[i], b[i], …)`), so every
ISA path produces **bit-identical** results regardless of vector width. The
fused kernels use `std::fma` on the scalar/AVX tiers (single rounding) so they
match the vector `_mm*_fmadd` paths exactly. Reductions whose result depends on
summation order are deliberately **excluded** from this module.

## `enum class Isa : std::uint8_t`

| Enumerator | `to_string_view` |
| :--- | :--- |
| `scalar` | `"scalar"` |
| `avx` | `"avx"` |
| `avx2` | `"avx2"` |
| `avx512` | `"avx512"` |

```cpp
[[nodiscard]] constexpr auto to_string_view(Isa isa) noexcept -> std::string_view;
[[nodiscard]] auto active_isa() noexcept -> Isa;   // the ISA selected for this process
```

## Kernels

Each kernel takes `std::span` arguments and is `noexcept`. To prevent any
out-of-bounds access on a caller size mismatch, each kernel operates over the
**smallest common length** of its spans.

| Kernel | Signature | Operation |
| :--- | :--- | :--- |
| `add` | `auto add(span<const float> a, span<const float> b, span<float> out) noexcept -> void` | `out = a + b` (`out` may alias `a` or `b`) |
| `mul` | `auto mul(span<const float> a, span<const float> b, span<float> out) noexcept -> void` | `out = a * b` |
| `axpy` | `auto axpy(float scale, span<const float> a, span<const float> b, span<float> out) noexcept -> void` | `out = a * scale + b` (fused multiply-add) |
| `horner_step` | `auto horner_step(span<float> acc, span<const float> x, float c) noexcept -> void` | `acc = acc * x + c` (`c` broadcast) — one Horner step |
| `exp_into` | `auto exp_into(span<const double> x, span<double> out) noexcept -> void` | `out[i] = exp(x[i])` in **double** precision (`out` may alias `x`) |

`horner_step` is the building block for evaluating a polynomial at many points
at once (see [`Polynomial::evaluate_batch`](polynomial.md)).

`exp_into` is a **deterministic vector exponential** (the one non-float, non-elementwise-trivial
kernel): Cody-Waite range reduction to `|r| ≤ ln2/2`, a degree-13 Horner polynomial, then a
`2^k` exponent construction. Like every kernel here it is **bit-identical across ISAs** — the
AVX-512 path (which additionally needs AVX512DQ; an AVX512F-only part falls back) and the scalar
reference use the same `std::fma`/`_mm512_fmadd` order, so results reproduce across hosts. It is
**accurate to 1 ulp** vs libm and validated in `simd_tests`. **Domain (honesty boundary):** defined
for `|x| ≲ 700` — the whole representable-`exp` range, which contains every finance exponential;
inputs that would overflow (`x ≳ 709`) or underflow (`x ≲ -745`) double are out of contract
(unspecified value, not a saturated `inf`/`0`). It backs the vectorised GBM exponentials in the
Monte-Carlo pricers ([`monte_carlo_european`](pricing.md)) — a `perf`-measured ~1.38× on the
serial European MC.

## Example

```cpp
import nimblecas.simd;
using namespace nimblecas;

std::array<float, 4> a{1, 2, 3, 4};
std::array<float, 4> b{10, 20, 30, 40};
std::array<float, 4> out{};

simd::axpy(2.0f, a, b, out);   // out = a*2 + b = {12, 24, 36, 44}

std::println("dispatched ISA: {}", simd::to_string_view(simd::active_isa()));
```

## See also

- [`nimblecas.polynomial`](polynomial.md) — batch evaluation via `horner_step`.
- [Parallel tree computation §6](../architecture/parallel-tree-computation.md) — linearization, the bridge to SIMD/GPU.
- [Documentation hub](../Index.md)
