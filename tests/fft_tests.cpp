// Tests for nimblecas.fft: the O(n log n) numeric FFT (radix-2 Cooley-Tukey +
// Bluestein chirp-z), its inverse, the real-input convenience, and FFT-based linear
// convolution. All checks are tolerance-based — this is a NUMERICAL module.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.fft;
import nimblecas.testing;

using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

using cd = std::complex<double>;

// Two complex numbers agree within tol (default 1e-9) in both parts.
[[nodiscard]] auto approx(const cd& a, const cd& b, double tol = 1e-9) -> bool {
    return std::abs(a.real() - b.real()) <= tol && std::abs(a.imag() - b.imag()) <= tol;
}

// Two complex vectors are equal length and agree elementwise within tol.
[[nodiscard]] auto approx_vec(std::span<const cd> a, std::span<const cd> b,
                              double tol = 1e-9) -> bool {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!approx(a[i], b[i], tol)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] auto approx_real(std::span<const double> a, std::span<const double> b,
                               double tol = 1e-9) -> bool {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tol) {
            return false;
        }
    }
    return true;
}

// Reference naive DFT for cross-checking: X_k = sum_j x_j exp(-2*pi*i*k*j/n).
[[nodiscard]] auto naive_dft(std::span<const cd> x) -> std::vector<cd> {
    const std::size_t n = x.size();
    std::vector<cd> out(n);
    for (std::size_t k = 0; k < n; ++k) {
        cd acc{0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) {
            const double angle = -2.0 * std::numbers::pi *
                                 (static_cast<double>(k) * static_cast<double>(j)) /
                                 static_cast<double>(n);
            acc += x[j] * std::polar(1.0, angle);
        }
        out[k] = acc;
    }
    return out;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.fft")
        // fft of a constant vector: only the DC (k=0) bin is non-zero.
        .test("fft_constant_vector",
              [](TestContext& t) {
                  const std::vector<cd> x = {cd{1, 0}, cd{1, 0}, cd{1, 0}, cd{1, 0}};
                  const auto X = nimblecas::fft(x).value();
                  const std::vector<cd> expect = {cd{4, 0}, cd{0, 0}, cd{0, 0}, cd{0, 0}};
                  t.expect(approx_vec(X, expect), "fft({1,1,1,1}) == {4,0,0,0}");
              })
        // fft of a unit impulse at index 0: a flat all-ones spectrum.
        .test("fft_impulse",
              [](TestContext& t) {
                  const std::vector<cd> x = {cd{1, 0}, cd{0, 0}, cd{0, 0}, cd{0, 0}};
                  const auto X = nimblecas::fft(x).value();
                  const std::vector<cd> expect = {cd{1, 0}, cd{1, 0}, cd{1, 0}, cd{1, 0}};
                  t.expect(approx_vec(X, expect), "fft({1,0,0,0}) == {1,1,1,1}");
              })
        // fft agrees with a hand-computed / naive DFT on a length-4 example.
        .test("fft_matches_naive_dft_len4",
              [](TestContext& t) {
                  const std::vector<cd> x = {cd{1, 0}, cd{2, -1}, cd{0, -1}, cd{-1, 2}};
                  const auto X = nimblecas::fft(x).value();
                  const auto ref = naive_dft(x);
                  t.expect(approx_vec(X, ref), "fft == naive DFT on a length-4 complex vector");
                  // Hand-computed X_0 is the plain sum of the samples = 2 + 0i.
                  t.expect(approx(X[0], cd{2, 0}), "X[0] == sum of samples == 2+0i");
              })
        // ifft(fft(x)) round-trips a fixed length-8 vector.
        .test("ifft_fft_roundtrip_len8",
              [](TestContext& t) {
                  const std::vector<cd> x = {cd{0.5, -1.0}, cd{2.0, 0.3},  cd{-1.5, 0.7},
                                             cd{3.1, -2.2}, cd{0.0, 1.0},  cd{-0.4, -0.9},
                                             cd{1.7, 0.2},  cd{-2.3, 1.1}};
                  const auto X = nimblecas::fft(x).value();
                  const auto y = nimblecas::ifft(X).value();
                  t.expect(approx_vec(x, y), "ifft(fft(x)) == x for a length-8 vector");
              })
        // Bluestein path: non-power-of-two lengths must still round-trip and match naive DFT.
        .test("fft_non_power_of_two_len6",
              [](TestContext& t) {
                  const std::vector<cd> x = {cd{1, 0},  cd{2, 1}, cd{-1, 0},
                                             cd{0, -2}, cd{3, 1}, cd{-2, 0}};
                  const auto X = nimblecas::fft(x).value();
                  const auto ref = naive_dft(x);
                  t.expect(approx_vec(X, ref), "fft (Bluestein, n=6) matches naive DFT");
                  const auto y = nimblecas::ifft(X).value();
                  t.expect(approx_vec(x, y), "ifft(fft(x)) == x for n=6 (Bluestein)");
              })
        // Bluestein on a prime length (n=5) — exercises the general-length path again.
        .test("fft_prime_length_len5",
              [](TestContext& t) {
                  const std::vector<cd> x = {cd{1, 0}, cd{0, 1}, cd{-1, 0}, cd{0, -1}, cd{2, 2}};
                  const auto X = nimblecas::fft(x).value();
                  const auto ref = naive_dft(x);
                  t.expect(approx_vec(X, ref), "fft (Bluestein, n=5) matches naive DFT");
              })
        // fft_real embeds real samples and agrees with the naive DFT.
        .test("fft_real_matches_naive",
              [](TestContext& t) {
                  const std::vector<double> r = {1.0, -2.0, 3.0, 0.5};
                  std::vector<cd> embed(r.size());
                  for (std::size_t i = 0; i < r.size(); ++i) {
                      embed[i] = cd{r[i], 0.0};
                  }
                  const auto X = nimblecas::fft_real(r).value();
                  const auto ref = naive_dft(embed);
                  t.expect(approx_vec(X, ref), "fft_real matches naive DFT of the embedded samples");
              })
        // Linear convolution via FFT: {1,2,3} * {1,1} == {1,3,5,3}.
        .test("convolve_linear_example",
              [](TestContext& t) {
                  const std::vector<double> a = {1.0, 2.0, 3.0};
                  const std::vector<double> b = {1.0, 1.0};
                  const auto c = nimblecas::convolve(a, b).value();
                  const std::vector<double> expect = {1.0, 3.0, 5.0, 3.0};
                  t.expect(approx_real(c, expect), "convolve({1,2,3},{1,1}) == {1,3,5,3}");
                  t.expect(c.size() == 4, "linear convolution length == la + lb - 1");
              })
        // Cross-check convolve against a direct O(n*m) computation on a larger example.
        .test("convolve_matches_direct",
              [](TestContext& t) {
                  const std::vector<double> a = {2.0, -1.0, 0.5, 3.0, 1.0};
                  const std::vector<double> b = {1.0, 4.0, -2.0};
                  const auto c = nimblecas::convolve(a, b).value();
                  std::vector<double> direct(a.size() + b.size() - 1, 0.0);
                  for (std::size_t i = 0; i < a.size(); ++i) {
                      for (std::size_t j = 0; j < b.size(); ++j) {
                          direct[i + j] += a[i] * b[j];
                      }
                  }
                  t.expect(approx_real(c, direct), "FFT convolution matches direct convolution");
              })
        // Empty-input contract: fft / ifft / fft_real / convolve all return empty, not error.
        .test("empty_inputs_yield_empty",
              [](TestContext& t) {
                  const std::vector<cd> ec{};
                  const std::vector<double> er{};
                  t.expect(nimblecas::fft(ec).value().empty(), "fft of empty is empty");
                  t.expect(nimblecas::ifft(ec).value().empty(), "ifft of empty is empty");
                  t.expect(nimblecas::fft_real(er).value().empty(), "fft_real of empty is empty");
                  t.expect(nimblecas::convolve(er, er).value().empty(),
                           "convolve of empty is empty");
              })
        // A single element transforms to itself (both radix-2 and the length-1 trivial case).
        .test("fft_length_one_identity",
              [](TestContext& t) {
                  const std::vector<cd> x = {cd{3.5, -1.25}};
                  const auto X = nimblecas::fft(x).value();
                  t.expect(X.size() == 1 && approx(X[0], cd{3.5, -1.25}),
                           "fft of a single sample is that sample");
              })
        .run();
}
