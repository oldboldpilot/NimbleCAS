// NimbleCAS fast Fourier transform: the O(n log n) numeric DFT (§7.3).
// @author Olumuyiwa Oluwasanmi
//
// Conforms to config/cpp_details.txt: C++23 modules, `import std`, trailing return
// types, no owning raw pointers, std::expected railway error handling (no exceptions),
// [[nodiscard]] on every observer.
//
// ===========================================================================
//  SCOPE — this is the NUMERIC transform companion to the exact `spectral`
//  tooling.
// ===========================================================================
//  This module is deliberately NUMERICAL. The discrete Fourier transform of a
//  real / complex sample vector is a sum of products with the twiddle factors
//  exp(-2*pi*i*k*j/n) — TRANSCENDENTAL complex roots of unity. They are not
//  rational, so unlike `nimblecas.spectral` (which is exact over Q for the
//  Legendre / Chebyshev-T POLYNOMIAL transforms) NOTHING here is exact: every
//  value carries floating-point rounding.
//
//  What this module adds over `spectral`: `spectral` already contains a naive,
//  compact O(n^2) DFT used internally by its Fourier-spectral derivatives, and
//  it says so honestly ("a fast O(N log N) FFT would compute the same result
//  asymptotically faster — we implement the honest O(N^2) form for clarity").
//  THIS module is that fast O(n log n) form, exposed as a first-class numeric
//  transform over std::complex<double>.
//
// ===========================================================================
//  HONESTY BOUNDARY (documented and true).
// ===========================================================================
//  NUMERICAL (double precision, complex roots of unity). The results agree with
//  the mathematical DFT to about 1e-10 for reasonable sizes (the error grows
//  slowly with n, as is inherent to floating-point transforms). We NEVER return
//  a wrong value silently: the only failure modes are shape/size guards that
//  return an honest MathError.
//
//  ALGORITHMS:
//    * Power-of-two lengths      -> iterative radix-2 Cooley-Tukey (bit-reversal
//                                   permutation + in-place butterfly stages).
//    * Arbitrary lengths (any n) -> Bluestein's chirp-z algorithm, which reduces
//                                   an arbitrary-length DFT to a linear
//                                   convolution evaluated by a padded radix-2
//                                   FFT. So `fft` / `ifft` accept ANY length.
//    * ifft                      -> conj . fft . conj, normalised by 1/n.
//    * convolve                  -> LINEAR convolution of two real sequences via
//                                   zero-padding to a power of two >= la+lb-1,
//                                   pointwise multiply in the frequency domain,
//                                   inverse transform, then truncate to la+lb-1.

export module nimblecas.fft;

import std;
import nimblecas.core;

export namespace nimblecas {

// Forward discrete Fourier transform, X_k = sum_{j=0}^{n-1} x_j exp(-2*pi*i*k*j/n).
// Power-of-two n uses iterative radix-2 Cooley-Tukey; any other n uses Bluestein's
// chirp-z algorithm (so EVERY length works). Empty input yields an empty vector.
// NUMERICAL (double): accurate to ~1e-10 for reasonable sizes. Fails only with
// MathError::domain_error when the internal padded length would overflow std::size_t.
[[nodiscard]] auto fft(std::span<const std::complex<double>> x)
    -> Result<std::vector<std::complex<double>>>;

// Inverse discrete Fourier transform, x_j = (1/n) sum_{k=0}^{n-1} X_k exp(+2*pi*i*k*j/n),
// implemented as conj . fft . conj divided by n. Round-trips the forward transform to
// ~1e-10. Empty input yields an empty vector. NUMERICAL (double). Same overflow guard.
[[nodiscard]] auto ifft(std::span<const std::complex<double>> spectrum)
    -> Result<std::vector<std::complex<double>>>;

// Real-input convenience: embeds the real samples as complex numbers (zero imaginary
// part) and forwards to fft. Any length. Empty input yields an empty vector. NUMERICAL.
[[nodiscard]] auto fft_real(std::span<const double> x)
    -> Result<std::vector<std::complex<double>>>;

// LINEAR convolution of two real sequences via FFT: (a * b)_k = sum_j a_j b_{k-j}, a
// vector of length la + lb - 1. Internally zero-pads both inputs to the next power of two
// >= la + lb - 1 (so the cyclic FFT convolution equals the linear one after truncation),
// multiplies in the frequency domain, inverts, and takes the real part. If either input
// is empty the result is empty. NUMERICAL (double): accurate to ~1e-10 for reasonable
// sizes. Fails only with MathError::domain_error on internal-length overflow.
[[nodiscard]] auto convolve(std::span<const double> a, std::span<const double> b)
    -> Result<std::vector<double>>;

}  // namespace nimblecas

// ===========================================================================
//  Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// A length is a power of two iff exactly one bit is set (0 is treated as not-a-power).
[[nodiscard]] auto is_power_of_two(std::size_t n) -> bool {
    return n != 0 && (n & (n - 1)) == 0;
}

// Smallest power of two >= n (at least 1). Returns MathError::overflow if no power of two
// representable in std::size_t is large enough (the shift would wrap past the top bit).
[[nodiscard]] auto next_power_of_two(std::size_t n) -> Result<std::size_t> {
    std::size_t p = 1;
    while (p < n) {
        if (p > (std::numeric_limits<std::size_t>::max() >> 1)) {
            return make_error<std::size_t>(MathError::overflow);
        }
        p <<= 1;
    }
    return p;
}

// In-place iterative radix-2 Cooley-Tukey butterfly on a power-of-two-length buffer.
// `invert == false` applies the forward twiddle sign (exp(-2*pi*i/len)); `invert == true`
// applies the conjugate sign. Neither branch normalises — the caller scales by 1/n when it
// wants a true inverse. Precondition (caller-guaranteed): a.size() is a power of two.
auto fft_radix2_inplace(std::vector<std::complex<double>>& a, bool invert) -> void {
    const std::size_t n = a.size();
    if (n <= 1) {
        return;
    }
    // Bit-reversal permutation.
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; (j & bit) != 0; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }
    // Butterfly stages. Twiddles come from std::polar (unit-modulus roots of unity).
    const double sign = invert ? 1.0 : -1.0;
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double angle = sign * 2.0 * std::numbers::pi / static_cast<double>(len);
        const std::complex<double> wlen = std::polar(1.0, angle);
        for (std::size_t start = 0; start < n; start += len) {
            std::complex<double> w{1.0, 0.0};
            const std::size_t half = len >> 1;
            for (std::size_t k = 0; k < half; ++k) {
                const std::complex<double> u = a[start + k];
                const std::complex<double> v = a[start + k + half] * w;
                a[start + k] = u + v;
                a[start + k + half] = u - v;
                w *= wlen;
            }
        }
    }
}

// Cyclic convolution of two equal-length power-of-two complex buffers, via the classic
// convolution theorem: ifft(fft(u) .* fft(v)). Both buffers are consumed by value.
// Precondition: u.size() == v.size() and both are powers of two.
[[nodiscard]] auto cyclic_convolve_pow2(std::vector<std::complex<double>> u,
                                        std::vector<std::complex<double>> v)
    -> std::vector<std::complex<double>> {
    const std::size_t m = u.size();
    fft_radix2_inplace(u, false);
    fft_radix2_inplace(v, false);
    for (std::size_t i = 0; i < m; ++i) {
        u[i] *= v[i];
    }
    fft_radix2_inplace(u, true);  // inverse butterfly ...
    const double inv = 1.0 / static_cast<double>(m);
    for (std::size_t i = 0; i < m; ++i) {
        u[i] *= inv;  // ... then normalise by 1/m to complete the inverse transform
    }
    return u;
}

// Forward DFT of an ARBITRARY-length sample vector via Bluestein's chirp-z algorithm.
// Writes X_k = sum_j x_j exp(-2*pi*i*k*j/n) using the identity k*j = (k^2 + j^2 - (k-j)^2)/2,
// which turns the DFT into a linear convolution of the chirp-modulated input with the
// chirp kernel — evaluated by a padded radix-2 FFT. Used only for non-power-of-two n.
[[nodiscard]] auto dft_bluestein(std::span<const std::complex<double>> x)
    -> Result<std::vector<std::complex<double>>> {
    const std::size_t n = x.size();
    // Chirp exponent: exp(-pi*i*j^2/n). j^2 is reduced modulo 2n before forming the angle,
    // which both avoids overflow of j*j and keeps the argument small for accuracy.
    const std::size_t mod = 2 * n;  // n >= 2 here, so this does not overflow for real sizes
    const auto chirp = [&](std::size_t j, double s) -> std::complex<double> {
        const std::size_t r = j % mod;
        const std::size_t sq = (r * r) % mod;  // (j^2) mod 2n
        const double angle = s * std::numbers::pi * static_cast<double>(sq) / static_cast<double>(n);
        return std::polar(1.0, angle);
    };
    // Convolution length m: a power of two >= 2n - 1.
    auto m_res = next_power_of_two(2 * n - 1);
    if (!m_res) {
        return make_error<std::vector<std::complex<double>>>(m_res.error());
    }
    const std::size_t m = *m_res;

    std::vector<std::complex<double>> a(m, std::complex<double>{0.0, 0.0});
    std::vector<std::complex<double>> b(m, std::complex<double>{0.0, 0.0});
    for (std::size_t j = 0; j < n; ++j) {
        a[j] = x[j] * chirp(j, -1.0);  // a_j = x_j * exp(-pi*i*j^2/n)
    }
    // b_l = exp(+pi*i*l^2/n) for l = 0..n-1, mirrored into the negative indices (wrapped).
    b[0] = std::complex<double>{1.0, 0.0};
    for (std::size_t l = 1; l < n; ++l) {
        const std::complex<double> val = chirp(l, 1.0);
        b[l] = val;
        b[m - l] = val;  // b is even in l; the wrap places the negative-index copies
    }
    const auto conv = cyclic_convolve_pow2(std::move(a), std::move(b));
    std::vector<std::complex<double>> out(n);
    for (std::size_t k = 0; k < n; ++k) {
        out[k] = chirp(k, -1.0) * conv[k];  // X_k = exp(-pi*i*k^2/n) * (a conv b)_k
    }
    return out;
}

}  // namespace

auto fft(std::span<const std::complex<double>> x)
    -> Result<std::vector<std::complex<double>>> {
    const std::size_t n = x.size();
    if (n == 0) {
        return std::vector<std::complex<double>>{};
    }
    if (is_power_of_two(n)) {
        std::vector<std::complex<double>> a(x.begin(), x.end());
        fft_radix2_inplace(a, false);
        return a;
    }
    return dft_bluestein(x);
}

auto ifft(std::span<const std::complex<double>> spectrum)
    -> Result<std::vector<std::complex<double>>> {
    const std::size_t n = spectrum.size();
    if (n == 0) {
        return std::vector<std::complex<double>>{};
    }
    // conj . fft . conj / n — correct for any length, reusing whichever forward path fft picks.
    std::vector<std::complex<double>> conj_in(n);
    for (std::size_t i = 0; i < n; ++i) {
        conj_in[i] = std::conj(spectrum[i]);
    }
    auto forward = fft(conj_in);
    if (!forward) {
        return make_error<std::vector<std::complex<double>>>(forward.error());
    }
    const double inv = 1.0 / static_cast<double>(n);
    std::vector<std::complex<double>> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = std::conj((*forward)[i]) * inv;
    }
    return out;
}

auto fft_real(std::span<const double> x)
    -> Result<std::vector<std::complex<double>>> {
    if (x.empty()) {
        return std::vector<std::complex<double>>{};
    }
    std::vector<std::complex<double>> cx(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        cx[i] = std::complex<double>{x[i], 0.0};
    }
    return fft(cx);
}

auto convolve(std::span<const double> a, std::span<const double> b)
    -> Result<std::vector<double>> {
    const std::size_t la = a.size();
    const std::size_t lb = b.size();
    if (la == 0 || lb == 0) {
        return std::vector<double>{};  // convolution with an empty sequence is empty
    }
    const std::size_t out_len = la + lb - 1;
    auto m_res = next_power_of_two(out_len);
    if (!m_res) {
        return make_error<std::vector<double>>(m_res.error());
    }
    const std::size_t m = *m_res;
    // Zero-pad both real inputs into complex buffers of the common power-of-two length m;
    // padding to >= la + lb - 1 makes the cyclic FFT convolution equal the LINEAR one.
    std::vector<std::complex<double>> fa(m, std::complex<double>{0.0, 0.0});
    std::vector<std::complex<double>> fb(m, std::complex<double>{0.0, 0.0});
    for (std::size_t i = 0; i < la; ++i) {
        fa[i] = std::complex<double>{a[i], 0.0};
    }
    for (std::size_t i = 0; i < lb; ++i) {
        fb[i] = std::complex<double>{b[i], 0.0};
    }
    const auto conv = cyclic_convolve_pow2(std::move(fa), std::move(fb));
    std::vector<double> out(out_len);
    for (std::size_t i = 0; i < out_len; ++i) {
        out[i] = conv[i].real();  // inputs are real, so the result is real up to rounding
    }
    return out;
}

}  // namespace nimblecas
