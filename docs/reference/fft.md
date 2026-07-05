# `nimblecas.fft` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/fft/fft.cppm`

The **fast Fourier transform** — the `O(n log n)` discrete Fourier transform over
`std::complex<double>`. This is the **numeric transform companion** to the exact
[`spectral`](spectral.md) tooling. Where `spectral` is *exact over `Q`* for the
Legendre / Chebyshev-T polynomial transforms, the DFT is built from the
transcendental twiddle factors `exp(-2*pi*i*k*j/n)` — complex roots of unity that
are **not** rational — so this module is unavoidably **numerical**: every value
carries floating-point rounding.

`spectral` already contains a compact, naive `O(n^2)` DFT used internally by its
Fourier-spectral derivatives, and it says so honestly ("a fast `O(N log N)` FFT
would compute the same result asymptotically faster — we implement the honest
`O(N^2)` form for clarity"). **This module is that fast form**, exposed as a
first-class numeric transform.

```cpp
import nimblecas.fft;
```

Depends only on [`core`](core.md) (for `Result` / `MathError`) and the standard
library.

## Honesty boundary

**Numerical (double precision, complex roots of unity).** Nothing here is exact.
The results agree with the mathematical DFT to about **`1e-10`** for reasonable
sizes; the error grows slowly with `n`, as is inherent to floating-point
transforms. In keeping with Rule 32, the engine **never returns a wrong value
silently**: the only failure mode is a size guard —
`MathError::overflow` — returned when the internal padded FFT length would exceed
`std::size_t` (a length no real machine could allocate). There is no
`domain_error` on ordinary inputs, because **every length is supported** (see the
algorithm note). Empty input is not an error: it yields an empty vector.

## Algorithms

| Case | Method |
| :--- | :--- |
| Power-of-two `n` | Iterative **radix-2 Cooley-Tukey**: an in-place bit-reversal permutation followed by `log2(n)` butterfly stages. Twiddle factors are formed with `std::polar` (unit-modulus roots of unity). |
| Arbitrary `n` (including primes) | **Bluestein's chirp-z algorithm**, which rewrites `k*j = (k^2 + j^2 - (k-j)^2)/2` to turn an arbitrary-length DFT into a *linear convolution* of the chirp-modulated input against the chirp kernel, evaluated by a zero-padded radix-2 FFT of length `2^ceil(log2(2n-1))`. So `fft` / `ifft` accept **any** length. |
| Inverse | `ifft = conj . fft . conj`, normalised by `1/n`. This reuses whichever forward path (radix-2 or Bluestein) `fft` selects. |
| Linear convolution | Zero-pad both real inputs to the next power of two `>= la + lb - 1`, multiply their transforms pointwise, invert, and take the real part, then truncate to `la + lb - 1`. Padding past `la + lb - 1` makes the cyclic FFT convolution coincide with the desired **linear** convolution. |

## API

```cpp
[[nodiscard]] auto fft(std::span<const std::complex<double>> x)
    -> Result<std::vector<std::complex<double>>>;
[[nodiscard]] auto ifft(std::span<const std::complex<double>> spectrum)
    -> Result<std::vector<std::complex<double>>>;
[[nodiscard]] auto fft_real(std::span<const double> x)
    -> Result<std::vector<std::complex<double>>>;
[[nodiscard]] auto convolve(std::span<const double> a, std::span<const double> b)
    -> Result<std::vector<double>>;
```

| Function | Behavior |
| :--- | :--- |
| `fft` | Forward DFT, `X_k = sum_j x_j exp(-2*pi*i*k*j/n)`. Radix-2 for power-of-two `n`, Bluestein otherwise. Empty input → empty. |
| `ifft` | Inverse DFT, `x_j = (1/n) sum_k X_k exp(+2*pi*i*k*j/n)`, via `conj . fft . conj / n`. Round-trips `fft` to `~1e-10`. Empty input → empty. |
| `fft_real` | Real-input convenience: embeds real samples as complex numbers with zero imaginary part and forwards to `fft`. Any length. Empty input → empty. |
| `convolve` | **Linear** convolution of two real sequences, `(a * b)_k = sum_j a_j b_{k-j}`, returning a length-`la + lb - 1` real vector. Either input empty → empty. |

## Error model

`Result<T>` (`std::expected<T, MathError>`) throughout — nothing throws. The sole
error is `MathError::overflow`, returned only when the internal padded length
required by Bluestein or `convolve` would overflow `std::size_t`. All ordinary
inputs succeed. The numerical (double) paths do not overflow-check arithmetic —
they carry the usual floating-point rounding — because a rounding error is not a
`MathError`; it is the documented numerical nature of the transform.

## Worked example — convolution

Multiplying the polynomials `1 + 2x + 3x^2` and `1 + x` corresponds to convolving
their coefficient sequences `{1, 2, 3}` and `{1, 1}`:

```cpp
import nimblecas.fft;

const std::vector<double> a = {1.0, 2.0, 3.0};  // 1 + 2x + 3x^2
const std::vector<double> b = {1.0, 1.0};       // 1 + x
const auto c = nimblecas::convolve(a, b).value();
// c == {1, 3, 5, 3}   (i.e. 1 + 3x + 5x^2 + 3x^3), accurate to ~1e-10
```

Internally the inputs are zero-padded to length `4` (the next power of two `>= 3 +
2 - 1 = 4`), transformed with the radix-2 FFT, multiplied in the frequency domain,
inverse-transformed, and the real parts of the first four entries are returned.

## Round-trip

```cpp
const auto spectrum = nimblecas::fft(samples).value();
const auto recovered = nimblecas::ifft(spectrum).value();  // ≈ samples to ~1e-10
```

## See also

- [`spectral.md`](spectral.md) — the exact-over-`Q` spectral transforms this
  module numerically complements (and whose internal `O(n^2)` DFT it accelerates).
- [`wavelets.md`](wavelets.md) — the companion time-frequency transform family
  (Haar exact over `Q`; Daubechies / symlet / coiflet / biorthogonal numerical).
- [`complex.md`](complex.md) — the engine's *exact* Gaussian-rational complex
  numbers, the algebraic counterpart to the `std::complex<double>` values used
  here.
