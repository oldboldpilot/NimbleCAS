# `nimblecas.wavelets` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/wavelets/wavelets.cppm`

Wavelet transforms across the modern family (ROADMAP §7 signal layer). The
module carries a hard **honesty boundary** between what is exact and what is
merely numerical, and every signature reflects it: the **Haar** transform (both
the sum/difference filter bank and its lifting realisation) is **EXACT over the
rationals `Q`** — it works on [`Rational`](ratpoly.md) coefficients and
`haar_idwt(haar_dwt(x)) == x` bit-for-bit. **Everything else is NUMERICAL**
(`double` / `std::complex<double>`): the orthogonal families (Daubechies,
Symlets, Coiflets) have irrational filter taps; the biorthogonal transform is the
CDF 5/3 lifting; and the continuous wavelets (Morlet, Ricker, Meyer, Shannon,
complex Gaussian) with the CWT are numerical quadratures. Perfect reconstruction
for the numerical paths holds **to floating-point round-off, not exactly** (the
CDF 5/3, whose lifting weights are dyadic, is bit-exact as a happy accident of
its weights). The directional multiscale systems — **curvelets** and
**shearlets** — are **honest `not_implemented` stubs**, not faked.

```cpp
import nimblecas.wavelets;   // namespace nimblecas::wavelets
```

Depends on [`core`](core.md) (for `Result<T>` / `MathError`) and
[`ratpoly`](ratpoly.md) (for the exact `Rational` the Haar path lives in).

## The exact/numerical boundary at a glance

| Path | Domain | Exactness |
| :--- | :--- | :--- |
| `haar_dwt` / `haar_idwt` (+ multi-level) | `Rational` (`Q`) | **Exact.** Round-trip is bit-for-bit. |
| `haar_lifting_forward` / `haar_lifting_inverse` | `Rational` (`Q`) | **Exact**, and identical to the filter bank by construction. |
| `daubechies` / `symlet` / `coiflet`, `dwt` / `idwt`, `swt`, `wavelet_packet` | `double` | Numerical. PR to round-off. |
| `cdf53_forward` / `cdf53_inverse` (biorthogonal) | `double` | Numerical, but PR **by construction**; dyadic weights make the round-trip bit-exact. |
| `morlet` / `ricker` / `meyer` / `shannon` / `complex_gaussian`, `cwt` | `double` / `complex<double>` | Numerical. Continuous wavelets and a quadrature CWT. |
| `curvelet_transform_2d` / `shearlet_transform_2d` | — | **`not_implemented`** (honest stub). |

## Haar — exact over `Q`

The unnormalised Haar filters keep the transform inside `Q`: keeping the
`1/√2` factor **out** of the filters (a sum/difference pair) is exactly what
makes every coefficient a `Rational`.

### Single level

```cpp
struct HaarLevel {
    std::vector<Rational> approx;   // approx[i] = x[2i] + x[2i+1]  (low band)
    std::vector<Rational> detail;   // detail[i] = x[2i] - x[2i+1]  (high band)
};

[[nodiscard]] auto haar_dwt(std::span<const Rational> signal) -> Result<HaarLevel>;
[[nodiscard]] auto haar_idwt(std::span<const Rational> approx,
                             std::span<const Rational> detail)
    -> Result<std::vector<Rational>>;
```

| Function | Behavior |
| :--- | :--- |
| `haar_dwt` | One decimated level. `approx[i] = x[2i] + x[2i+1]`, `detail[i] = x[2i] − x[2i+1]`. Requires a non-empty **even** length, else `domain_error`. |
| `haar_idwt` | Exact inverse: `x[2i] = (approx[i] + detail[i]) / 2`, `x[2i+1] = (approx[i] − detail[i]) / 2`. The `/2` is exact in `Q`. `approx` and `detail` must be the same length, else `domain_error`. |

### Multi-level (Mallat pyramid)

```cpp
struct HaarDecomposition {
    std::vector<std::vector<Rational>> details;  // details[0] = finest … last = coarsest
    std::vector<Rational> approx;                // final approximation band
};

[[nodiscard]] auto haar_dwt_multi(std::span<const Rational> signal, std::size_t levels)
    -> Result<HaarDecomposition>;
[[nodiscard]] auto haar_idwt_multi(const HaarDecomposition& decomp)
    -> Result<std::vector<Rational>>;
```

`haar_dwt_multi` recursively transforms the approximation band `levels` times;
`details[0]` is the finest (level-1) detail and `approx` is the final coarse
band. The signal length must be divisible by `2^levels` (each level halves it),
and `levels` must be in `1 … 62`; otherwise `domain_error`. `haar_idwt_multi`
folds the pyramid back up, exactly.

### Haar via the lifting scheme (second generation)

```cpp
[[nodiscard]] auto haar_lifting_forward(std::span<const Rational> signal) -> Result<HaarLevel>;
[[nodiscard]] auto haar_lifting_inverse(std::span<const Rational> approx,
                                        std::span<const Rational> detail)
    -> Result<std::vector<Rational>>;
```

Split / predict / update with a final integer rescale, chosen so the output is
**bit-for-bit identical** to `haar_dwt`:

```
split:   e = x[2i], o = x[2i+1]
predict: o <- o - e            (=> b - a)
update:  e <- e + o/2          (=> (a + b) / 2)
rescale: approx = 2e = a + b,  detail = -o = a - b
```

Every step is exactly invertible in `Q`, so this is perfect reconstruction and,
by design, `haar_lifting_forward == haar_dwt` exactly. Same `domain_error`
guards as the filter-bank path (non-empty even length; equal-length bands on the
inverse).

## Numerical orthogonal filter banks (Daubechies / Symlet / Coiflet)

Coefficients are **irrational** — this is numerical. For an orthogonal wavelet
the synthesis filters equal the analysis filters (synthesis is the adjoint of
analysis).

```cpp
struct FilterBank {
    std::vector<double> analysis_lo;
    std::vector<double> analysis_hi;
    std::vector<double> synthesis_lo;
    std::vector<double> synthesis_hi;
};

enum class Family : std::uint8_t { daubechies, symlet, coiflet };

[[nodiscard]] auto make_orthogonal(std::vector<double> lo) -> FilterBank;
[[nodiscard]] auto daubechies(int order) -> Result<FilterBank>;
[[nodiscard]] auto symlet(int order) -> Result<FilterBank>;
[[nodiscard]] auto coiflet(int order) -> Result<FilterBank>;
[[nodiscard]] auto orthogonal_filters(Family fam, int order) -> Result<FilterBank>;
```

| Function | Behavior |
| :--- | :--- |
| `make_orthogonal` | Build a full bank from a low-pass analysis filter `lo`, deriving the high-pass by the quadrature-mirror relation `g[k] = (−1)^k h[L−1−k]`. Total (no `Result`). |
| `daubechies` | `dbN` for `order` in `1 … 10` (db1 = Haar; db1/db2 closed-form, db3–db10 tabulated). Unknown order → `not_implemented`. |
| `symlet` | `symN` for `order` in `2 … 8` (sym2 = db2, sym3 = db3). Unknown order → `not_implemented`. |
| `coiflet` | `coifN` for `order` in `1 … 3`. Unknown order → `not_implemented`. |
| `orthogonal_filters` | Dispatch on `Family`. Forwards to the above (and its `not_implemented` contract). |

### Sum-rule / orthogonality diagnostics (all numerical, total)

```cpp
[[nodiscard]] auto lowpass_sum(std::span<const double> h) -> double;
[[nodiscard]] auto lowpass_alternating_moment(std::span<const double> h, int m) -> double;
[[nodiscard]] auto highpass_moment(std::span<const double> g, int m) -> double;
[[nodiscard]] auto orthogonality_defect(std::span<const double> h, int m) -> double;
```

| Function | Value |
| :--- | :--- |
| `lowpass_sum` | `Σ h_k`; should equal `√2` for a normalised scaling filter. |
| `lowpass_alternating_moment` | `Σ_k (−1)^k k^m h_k`; vanishes for `m = 0 … p−1` with `p` vanishing moments. |
| `highpass_moment` | `Σ_k k^m g_k`; the equivalent high-pass moment, vanishing for `m = 0 … p−1`. |
| `orthogonality_defect` | `(Σ_k h_k h_{k+2m}) − δ_{m,0}`; the double-shift autocorrelation minus Kronecker delta, `≈ 0` for an orthonormal filter. |

These return plain `double`s (no `Result`) — they are pure diagnostics.

## Numerical decimated DWT / IDWT (periodic Mallat filter bank)

```cpp
struct DwtLevel {
    std::vector<double> approx;
    std::vector<double> detail;
};

[[nodiscard]] auto dwt(std::span<const double> signal, const FilterBank& fb) -> Result<DwtLevel>;
[[nodiscard]] auto idwt(std::span<const double> approx, std::span<const double> detail_band,
                        const FilterBank& fb) -> Result<std::vector<double>>;

struct Decomposition {
    std::vector<std::vector<double>> details;  // finest … coarsest
    std::vector<double> approx;
};

[[nodiscard]] auto dwt_multi(std::span<const double> signal, const FilterBank& fb,
                             std::size_t levels) -> Result<Decomposition>;
[[nodiscard]] auto idwt_multi(const Decomposition& decomp, const FilterBank& fb)
    -> Result<std::vector<double>>;
```

| Function | Behavior |
| :--- | :--- |
| `dwt` | One periodic analysis level (convolve + decimate by 2). Requires a non-empty **even** length, else `domain_error`. |
| `idwt` | Transpose-convolution synthesis; the adjoint is the inverse to round-off for an orthonormal bank. `approx` / `detail_band` must be equal length, else `domain_error`. |
| `dwt_multi` | Mallat pyramid to `levels`. Length must be divisible by `2^levels`, and `levels` in `1 … 62`, else `domain_error`. |
| `idwt_multi` | Reconstructs from a `Decomposition` and its bank, numerically to round-off. |

## Stationary / undecimated wavelet transform (SWT, à-trous)

```cpp
struct SwtLevel {
    std::vector<double> approx;
    std::vector<double> detail;
};

[[nodiscard]] auto swt(std::span<const double> signal, const FilterBank& fb,
                       std::size_t level = 1) -> Result<SwtLevel>;
```

No decimation: at `level` the filters are dilated by inserting `2^(level−1) − 1`
zeros between taps (the à-trous holes). Both output bands keep the **full** input
length `n` — the defining length-invariance of the SWT. Requires a non-empty
signal and `level` in `1 … 30`, else `domain_error`.

## Wavelet packet decomposition

```cpp
[[nodiscard]] auto wavelet_packet(std::span<const double> signal, const FilterBank& fb,
                                  std::size_t level)
    -> Result<std::vector<std::vector<double>>>;
```

Full binary tree: at each level **both** the approximation and the detail band
are split again, returning the `2^level` leaf subbands left-to-right in natural
order, each of length `n / 2^level`. Length must be divisible by `2^level`, and
`level` in `1 … 62`, else `domain_error`.

## Biorthogonal transform via the lifting factorisation

The CDF 5/3 (bior2.2 / LeGall, the JPEG2000 reversible filter). Periodic
lifting; every step is exactly invertible and the weights are dyadic, so the
`double` round-trip is bit-exact. This is a genuine **biorthogonal** (not
orthogonal) wavelet.

```cpp
[[nodiscard]] auto cdf53_forward(std::span<const double> signal) -> Result<DwtLevel>;
[[nodiscard]] auto cdf53_inverse(std::span<const double> approx,
                                 std::span<const double> detail_band)
    -> Result<std::vector<double>>;

[[nodiscard]] auto biorthogonal_filters(int recon_order, int decon_order) -> Result<FilterBank>;
```

| Function | Behavior |
| :--- | :--- |
| `cdf53_forward` | `predict: d[i] = o[i] − (e[i] + e[i+1])/2`, `update: s[i] = e[i] + (d[i−1] + d[i])/4`. Requires a non-empty **even** length, else `domain_error`. |
| `cdf53_inverse` | Undoes update then predict. `approx` / `detail_band` must be equal length, else `domain_error`. |
| `biorthogonal_filters` | The **tabulated** analysis+synthesis filter *pair* for inspection / sum-rule checks. Only `bior2.2` (`recon_order == 2 && decon_order == 2`, the CDF 5/3 pair) is provided; all other orders are honestly `not_implemented`. Note this is *data* — the working transform is the `cdf53_*` lifting, not these filters. |

## Continuous wavelets `ψ(t)` and the CWT

All numerical. The generating wavelets are ordinary functions of `t`; the CWT
correlates a signal against dilated/translated conjugates of one.

```cpp
[[nodiscard]] auto morlet(double t, double omega0 = 5.0) -> std::complex<double>;
[[nodiscard]] auto morlet_real(double t, double omega0 = 5.0) -> double;
[[nodiscard]] auto ricker(double t, double sigma = 1.0) -> double;
[[nodiscard]] auto shannon(double t) -> double;
[[nodiscard]] auto meyer_hat(double omega) -> double;
[[nodiscard]] auto meyer(double t) -> double;
[[nodiscard]] auto complex_gaussian(double t) -> std::complex<double>;

[[nodiscard]] auto cwt(std::span<const double> signal, std::span<const double> scales,
                       const std::function<std::complex<double>(double)>& psi)
    -> Result<std::vector<std::vector<std::complex<double>>>>;
```

| Function | Definition |
| :--- | :--- |
| `morlet` | Complex Morlet `ψ(t) = π^{−1/4} (e^{iω₀t} − e^{−ω₀²/2}) e^{−t²/2}`; the second term enforces the (near-)zero-mean admissibility condition. |
| `morlet_real` | The real part of the complex Morlet. |
| `ricker` | Ricker / Mexican-hat, the negative second derivative of a Gaussian: `ψ(t) = 2/(√(3σ) π^{1/4}) (1 − (t/σ)²) e^{−t²/(2σ²)}`. |
| `shannon` | Shannon (sinc / band-pass): `ψ(t) = (sin 2πt − sin πt)/(πt)`, with the removable singularity `ψ(0) = 1`. |
| `meyer_hat` | The Meyer frequency-domain magnitude window `|ψ̂(ω)|` (ν-smoothed band on `[2π/3, 8π/3]`; the `e^{iω/2}` phase is dropped). Exposed for spectral inspection. |
| `meyer` | Meyer `ψ(t)` by a bounded numerical quadrature of the inverse Fourier transform of the window: `ψ(t) = (1/π) ∫₀^{8π/3} M(ω) cos(ω(t + ½)) dω`. A quadrature approximation, not a closed form. |
| `complex_gaussian` | First-order complex Gaussian `cgau1`, the normalised first derivative of `g(t) = e^{−t²} e^{−it}`, i.e. `√2 π^{−1/4} (−i − 2t) e^{−t²} e^{−it}`. |

`cwt` computes `W(a, b) = (1/√a) Σ_t x[t] · conj(ψ((t − b)/a))` at integer
translations `b = 0 … n−1`. Row `i` corresponds to `scales[i]`; each row has `n`
columns. Requires a non-empty signal and a non-empty `scales`, and **every scale
must be positive** — a non-positive (or non-empty-but-`≤ 0`) scale →
`domain_error`.

## Directional multiscale systems — honest `not_implemented`

```cpp
[[nodiscard]] auto curvelet_transform_2d(std::span<const double> data,
                                         std::size_t rows, std::size_t cols)
    -> Result<std::vector<std::vector<double>>>;
[[nodiscard]] auto shearlet_transform_2d(std::span<const double> data,
                                         std::size_t rows, std::size_t cols)
    -> Result<std::vector<std::vector<double>>>;
```

Curvelets and shearlets need a genuine 2D framing (parabolic scaling,
shear/rotation, and a wedge tiling of the frequency plane) beyond this first
1D-centric module. They are declared so the API is discoverable, `data` being a
row-major `rows × cols` image, but **always** return `MathError::not_implemented`
rather than a fabricated result. They are not faked.

## Error model

| Condition | Error |
| :--- | :--- |
| Odd or zero length where an even length is required (`haar_dwt`, `haar_lifting_forward`, `dwt`, `cdf53_forward`) | `MathError::domain_error` |
| Mismatched `approx` / `detail` lengths on any inverse (`haar_idwt`, `haar_lifting_inverse`, `idwt`, `cdf53_inverse`) | `MathError::domain_error` |
| `levels` / `level` out of range, or length not divisible by `2^levels` (`haar_dwt_multi`, `dwt_multi`, `wavelet_packet`) | `MathError::domain_error` |
| Empty signal, or `level` out of `1 … 30` (`swt`) | `MathError::domain_error` |
| `cwt` with an empty signal, empty `scales`, or a non-positive scale | `MathError::domain_error` |
| Unknown wavelet order (`daubechies` / `symlet` / `coiflet` / `orthogonal_filters`), or an unsupported `biorthogonal_filters` pair | `MathError::not_implemented` |
| Any call to `curvelet_transform_2d` / `shearlet_transform_2d` | `MathError::not_implemented` |
| A `Rational` add / subtract / multiply / divide overflowing `int64` inside a Haar path | `MathError::overflow` (propagated from `Rational`) |

The diagnostics (`lowpass_sum`, `lowpass_alternating_moment`, `highpass_moment`,
`orthogonality_defect`) and the continuous wavelet evaluators (`morlet`,
`morlet_real`, `ricker`, `shannon`, `meyer_hat`, `meyer`, `complex_gaussian`)
return plain values and never fail. `make_orthogonal` is likewise total.

## Worked examples

Mirroring `tests/wavelets_tests.cpp`. Honesty is enforced by the *comparison*:
Haar with exact rational `==`, everything else with a numerical tolerance.

```cpp
import nimblecas.wavelets;
import nimblecas.ratpoly;
import nimblecas.core;
using nimblecas::Rational;
using nimblecas::MathError;
namespace wl = nimblecas::wavelets;

// --- Haar: EXACT round-trip over Q, even on non-dyadic rationals ---
std::vector<Rational> x{*Rational::make(1, 3), *Rational::make(2, 5),
                        Rational::from_int(7),  *Rational::make(-4, 9),
                        *Rational::make(11, 2), Rational::from_int(0),
                        *Rational::make(-1, 7), *Rational::make(8, 3)};
auto fwd = wl::haar_dwt(x).value();
auto rec = wl::haar_idwt(fwd.approx, fwd.detail).value();   // == x EXACTLY (bit-for-bit in Q)

// Odd length is rejected.
auto bad = wl::haar_dwt(std::vector<Rational>{Rational::from_int(1),
                                              Rational::from_int(2),
                                              Rational::from_int(3)});
bad.error();                                                // MathError::domain_error

// Constant signal -> all details exactly 0, approx == 2*c.
std::vector<Rational> c(8, Rational::from_int(5));
auto cd = wl::haar_dwt(c).value();                          // every detail == 0, every approx == 10

// Multi-level: a length-8 constant collapses to a single approx coeff.
auto md = wl::haar_dwt_multi(c, 3).value();                 // md.approx.size() == 1, all details 0
auto back = wl::haar_idwt_multi(md).value();                // == c exactly

// Lifting equals the filter bank, exactly.
std::vector<Rational> y{Rational::from_int(3), Rational::from_int(-1),
                        Rational::from_int(4), Rational::from_int(1),
                        Rational::from_int(5), Rational::from_int(9),
                        Rational::from_int(2), Rational::from_int(6)};
auto lift = wl::haar_lifting_forward(y).value();            // lift.approx == haar_dwt(y).approx, exactly
wl::haar_lifting_inverse(lift.approx, lift.detail).value(); // == y exactly

// --- Daubechies db2: sum rule and orthonormality (NUMERICAL) ---
auto fb2 = wl::daubechies(2).value();
wl::lowpass_sum(fb2.analysis_lo);                           // ~ sqrt(2)
wl::lowpass_alternating_moment(fb2.analysis_lo, 0);         // ~ 0  (2 vanishing moments: m = 0, 1)
wl::orthogonality_defect(fb2.analysis_lo, 0);               // ~ 0  (Sum h^2 == 1)
wl::orthogonality_defect(fb2.analysis_lo, 1);               // ~ 0  (double-shift autocorr)

// db4 DWT / IDWT reconstructs to round-off (not exactly).
auto fb4 = wl::daubechies(4).value();
std::vector<double> s{1.0, 2.0, -3.0, 0.5, 4.0, -1.5, 2.5, 3.0};
auto lvl = wl::dwt(s, fb4).value();
auto srec = wl::idwt(lvl.approx, lvl.detail, fb4).value();  // ~ s to 1e-9 (NUMERICAL)

// Unknown order is honest.
wl::daubechies(99).error();                                 // MathError::not_implemented

// --- Biorthogonal CDF 5/3: perfect reconstruction (bit-exact via dyadic lifting) ---
std::vector<double> b{1.0, 2.0, -3.0, 0.5, 4.0, -1.5, 2.5, 3.0,
                      0.25, -0.75, 6.0, -2.0, 1.0, 1.0, -4.0, 2.0};
auto blvl = wl::cdf53_forward(b).value();
wl::cdf53_inverse(blvl.approx, blvl.detail).value();        // ~ b to 1e-10
auto pair = wl::biorthogonal_filters(2, 2).value();         // bior2.2 tabulated pair
wl::lowpass_sum(pair.analysis_lo);                          // ~ sqrt(2)

// --- SWT is length-invariant (undecimated) ---
auto s1 = wl::swt(s, fb2, 1).value();                       // s1.approx.size() == s.size()
auto s2 = wl::swt(s, fb2, 2).value();                       // à-trous dilated, still full length

// --- Wavelet packets: 2^level leaves, each n / 2^level long ---
auto wp = wl::wavelet_packet(s, fb2, 2).value();            // wp.size() == 4, each band size 2

// --- CWT: the Ricker magnitude peaks at the feature location ---
std::vector<double> spike(32, 0.0);
spike[12] = 1.0;
std::vector<double> scales{1.0, 2.0, 4.0};
auto coeffs = wl::cwt(spike, scales,
                      [](double tt) { return std::complex<double>{wl::ricker(tt), 0.0}; }).value();
// argmax over |coeffs| lands at b == 12.

// A non-positive scale is rejected.
wl::cwt(spike, std::vector<double>{1.0, 0.0},
        [](double tt) { return std::complex<double>{wl::ricker(tt), 0.0}; }).error();  // domain_error

// --- Directional systems are honestly not_implemented ---
std::vector<double> img(16, 0.0);
wl::curvelet_transform_2d(img, 4, 4).error();               // MathError::not_implemented
wl::shearlet_transform_2d(img, 4, 4).error();               // MathError::not_implemented
```

## See also

- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field the Haar
  transform stays inside.
- [`nimblecas.complex`](complex.md) — exact Gaussian rationals; contrast with the
  numerical `std::complex<double>` the CWT and Morlet/complex-Gaussian wavelets
  use here.
- [`nimblecas.numeric`](numeric.md) — the sibling floating-point signal/solver
  layer with the same honest exact-vs-numerical stance.
- [`nimblecas.svgplot`](svgplot.md) — for rendering CWT scalograms and
  decomposition bands.
- [Documentation hub](../Index.md)
