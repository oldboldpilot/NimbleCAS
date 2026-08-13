// Tests for nimblecas.gpu: batch polynomial evaluation on the GPU vs a CPU reference.
// @author Olumuyiwa Oluwasanmi
//
// Built and run only with -DNIMBLECAS_CUDA=ON on a machine with a CUDA device.

import std;
import nimblecas.core;
import nimblecas.gpu;
import nimblecas.pricing;
import nimblecas.optstrat;
import nimblecas.futures;
import nimblecas.testing;
import nimblecas.control;
import nimblecas.ratpoly;
import nimblecas.wavelets;
import nimblecas.qmc;
import nimblecas.krylov;
import nimblecas.nlsolve;

namespace gpu = nimblecas::gpu;
namespace nlsolve = nimblecas::nlsolve;
using nimblecas::MathError;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// CPU reference Horner evaluation, matching the kernel's order.
[[nodiscard]] auto cpu_poly_eval(std::span<const double> coeffs, double xi) -> double {
    double acc = 0.0;
    for (std::size_t k = coeffs.size(); k-- > 0;) {
        acc = acc * xi + coeffs[k];
    }
    return acc;
}

// Relative tolerance (GPU may contract to FMA where the CPU does not).
[[nodiscard]] auto approx(double a, double b) -> bool {
    const double diff = std::abs(a - b);
    return diff <= 1e-9 * (1.0 + std::abs(b));
}

// CPU Black-Scholes reference (matching the device kernel's formula) for validating the
// batch GPU pricer against the authoritative CPU closed form.
[[nodiscard]] auto cpu_bs(const gpu::BsOption& o) -> double {
    const double S = o.spot, K = o.strike, r = o.rate, q = o.dividend, v = o.volatility, T = o.time;
    const bool call = o.is_call;
    auto ncdf = [](double x) { return 0.5 * std::erfc(-x * 0.7071067811865475244); };
    if (T == 0.0 || v == 0.0) {
        const double fwd = S * std::exp((r - q) * T);
        const double intr = call ? std::max(fwd - K, 0.0) : std::max(K - fwd, 0.0);
        return std::exp(-r * T) * intr;
    }
    const double sq = std::sqrt(T);
    const double d1 = (std::log(S / K) + (r - q + 0.5 * v * v) * T) / (v * sq);
    const double d2 = d1 - v * sq;
    const double dr = std::exp(-r * T), dq = std::exp(-q * T);
    return call ? (S * dq * ncdf(d1) - K * dr * ncdf(d2)) : (K * dr * ncdf(-d2) - S * dq * ncdf(-d1));
}

// Tiny CPU Levenshtein reference over integer code points, matching the kernel's recurrence.
[[nodiscard]] auto cpu_levenshtein(std::span<const int> a, std::span<const int> b) -> int {
    std::vector<int> prev(b.size() + 1);
    std::vector<int> curr(b.size() + 1);
    for (std::size_t j = 0; j <= b.size(); ++j) {
        prev[j] = static_cast<int>(j);
    }
    for (std::size_t i = 1; i <= a.size(); ++i) {
        curr[0] = static_cast<int>(i);
        for (std::size_t j = 1; j <= b.size(); ++j) {
            const int cost = a[i - 1] != b[j - 1] ? 1 : 0;
            const int del = prev[j] + 1;
            const int ins = curr[j - 1] + 1;
            const int sub = prev[j - 1] + cost;
            curr[j] = std::min(std::min(del, ins), sub);
        }
        std::swap(prev, curr);
    }
    return prev[b.size()];
}

// CPU reference for one Haar DWT level over a single block, matching the kernel's normalization
// and its approximation-then-detail packing.
[[nodiscard]] auto cpu_haar_level(std::span<const double> block) -> std::vector<double> {
    const std::size_t half = block.size() / 2;
    const double inv_sqrt2 = 1.0 / std::numbers::sqrt2;
    std::vector<double> out(block.size());
    for (std::size_t k = 0; k < half; ++k) {
        const double e = block[2 * k];
        const double o = block[2 * k + 1];
        out[k] = (e + o) * inv_sqrt2;
        out[half + k] = (e - o) * inv_sqrt2;
    }
    return out;
}

// CPU reference batched matmul over row-major blocks, matching the kernel's accumulation order:
// C_b[i,j] = sum_l A_b[i,l] * B_b[l,j], with the `batch` problems packed contiguously.
[[nodiscard]] auto cpu_batched_matmul(std::span<const double> a, std::span<const double> b,
                                      int batch, int m, int k, int n) -> std::vector<double> {
    std::vector<double> c(static_cast<std::size_t>(batch) * static_cast<std::size_t>(m) *
                              static_cast<std::size_t>(n),
                          0.0);
    for (int bi = 0; bi < batch; ++bi) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                double acc = 0.0;
                for (int l = 0; l < k; ++l) {
                    acc += a[static_cast<std::size_t>(bi * m * k + i * k + l)] *
                           b[static_cast<std::size_t>(bi * k * n + l * n + j)];
                }
                c[static_cast<std::size_t>(bi * m * n + i * n + j)] = acc;
            }
        }
    }
    return c;
}

// CPU reference forward DFT, direct O(n^2): X_k = sum_j x_j e^{-2*pi*i*k*j/n}. Self-contained
// (no dependency on the fft module) so it independently pins the GPU kernel's twiddle sign and
// bit-reversal. `sig` is one signal of n complex samples as 2*n interleaved doubles (re, im, ...);
// returns the transform in the same interleaved layout.
[[nodiscard]] auto cpu_dft(std::span<const double> sig, int n) -> std::vector<double> {
    std::vector<double> out(static_cast<std::size_t>(2 * n), 0.0);
    for (int k = 0; k < n; ++k) {
        double re = 0.0;
        double im = 0.0;
        for (int j = 0; j < n; ++j) {
            const double ang = -2.0 * std::numbers::pi * static_cast<double>(k) *
                               static_cast<double>(j) / static_cast<double>(n);
            const double c = std::cos(ang);
            const double s = std::sin(ang);
            const double xr = sig[static_cast<std::size_t>(2 * j)];
            const double xi = sig[static_cast<std::size_t>(2 * j + 1)];
            // (xr + i*xi) * (c + i*s) = (xr*c - xi*s) + i*(xr*s + xi*c)
            re += xr * c - xi * s;
            im += xr * s + xi * c;
        }
        out[static_cast<std::size_t>(2 * k)] = re;
        out[static_cast<std::size_t>(2 * k + 1)] = im;
    }
    return out;
}

// Convert a string to a vector of int code points, matching how the batch test flattens input.
[[nodiscard]] auto code_points(std::string_view s) -> std::vector<int> {
    std::vector<int> out;
    out.reserve(s.size());
    for (char c : s) {
        out.push_back(static_cast<int>(static_cast<unsigned char>(c)));
    }
    return out;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.gpu")
        .test("device_available",
              [](TestContext& t) {
                  t.expect(gpu::device_count() >= 1, "at least one CUDA device is present");
                  t.expect(gpu::available(), "GPU reported available");
              })
        .test("batch_poly_eval_matches_cpu",
              [](TestContext& t) {
                  // p(x) = 1 + 2x + 3x^2.
                  const std::vector<double> coeffs = {1.0, 2.0, 3.0};
                  const std::vector<double> x = {0.0, 1.0, 2.0, -1.0, 3.5, 10.0, -4.25, 0.5};
                  auto got = gpu::poly_eval(coeffs, x).value();
                  t.expect(got.size() == x.size(), "one result per input point");
                  bool all = true;
                  for (std::size_t i = 0; i < x.size(); ++i) {
                      if (!approx(got[i], cpu_poly_eval(coeffs, x[i]))) {
                          all = false;
                      }
                  }
                  t.expect(all, "GPU batch evaluation matches the CPU Horner reference");
              })
        .test("large_batch",
              [](TestContext& t) {
                  // A larger batch exercises multiple blocks (> 256 threads).
                  const std::vector<double> coeffs = {-3.0, 0.0, 2.0, 1.0};  // x^3 + 2x^2 - 3
                  std::vector<double> x(10000);
                  for (std::size_t i = 0; i < x.size(); ++i) {
                      x[i] = static_cast<double>(i) * 0.001 - 5.0;
                  }
                  auto got = gpu::poly_eval(coeffs, x).value();
                  t.expect(got.size() == x.size(), "result size matches");
                  t.expect(approx(got.front(), cpu_poly_eval(coeffs, x.front())) &&
                               approx(got.back(), cpu_poly_eval(coeffs, x.back())),
                           "endpoints match the CPU reference");
              })
        .test("edge_cases",
              [](TestContext& t) {
                  // Empty input -> empty output.
                  const std::vector<double> coeffs = {5.0};
                  const std::vector<double> none;
                  t.expect(gpu::poly_eval(coeffs, none).value().empty(),
                           "empty point set yields empty output");
                  // Constant polynomial p(x) = 7.
                  const std::vector<double> c = {7.0};
                  const std::vector<double> x = {1.0, 2.0, 3.0};
                  auto got = gpu::poly_eval(c, x).value();
                  t.expect(got.size() == 3 && approx(got[0], 7.0) && approx(got[2], 7.0),
                           "constant polynomial evaluates to 7 everywhere");
              })
        .test("edit_distance_batch_matches_cpu",
              [](TestContext& t) {
                  struct Pair {
                      std::string_view a;
                      std::string_view b;
                  };
                  const std::vector<Pair> pairs = {{"kitten", "sitting"}, {"", "abc"},
                                                   {"abc", "abc"},        {"flaw", "lawn"},
                                                   {"gumbo", "gambol"},   {"sitting", ""}};
                  // Flatten the pairs into code-point arrays with prefix-offset boundaries.
                  std::vector<int> a_flat;
                  std::vector<int> b_flat;
                  std::vector<int> a_off = {0};
                  std::vector<int> b_off = {0};
                  for (const auto& p : pairs) {
                      for (char c : p.a) {
                          a_flat.push_back(static_cast<int>(static_cast<unsigned char>(c)));
                      }
                      for (char c : p.b) {
                          b_flat.push_back(static_cast<int>(static_cast<unsigned char>(c)));
                      }
                      a_off.push_back(static_cast<int>(a_flat.size()));
                      b_off.push_back(static_cast<int>(b_flat.size()));
                  }
                  auto got = gpu::edit_distance_batch(a_flat, a_off, b_flat, b_off).value();
                  t.expect(got.size() == pairs.size(), "one distance per pair");
                  bool all = got.size() == pairs.size();
                  for (std::size_t i = 0; i < pairs.size() && all; ++i) {
                      const auto ai = code_points(pairs[i].a);
                      const auto bi = code_points(pairs[i].b);
                      if (got[i] != cpu_levenshtein(ai, bi)) {
                          all = false;
                      }
                  }
                  t.expect(all, "GPU batch edit distance matches the CPU reference");
                  t.expect(!got.empty() && got[0] == 3, "kitten -> sitting is 3");
              })
        .test("edit_distance_long_sequence_symmetric_and_bounded",
              [](TestContext& t) {
                  // Regression: a SHORT a against a LONG b (400 code points). The kernel rolls
                  // over the shorter side, so this computes correctly (min side <= 256) rather
                  // than silently truncating the long side. Empty a vs a length-400 b => 400.
                  std::vector<int> a_flat;                 // a is empty
                  std::vector<int> b_flat(400, 7);         // 400-long b
                  std::vector<int> a_off = {0, 0};
                  std::vector<int> b_off = {0, 400};
                  auto got = gpu::edit_distance_batch(a_flat, a_off, b_flat, b_off);
                  t.expect(got.has_value(), "short-vs-long pair is accepted (rolled over short)");
                  t.expect(got && got->size() == 1 && (*got)[0] == 400,
                           "distance from empty to a 400-long string is 400, not truncated");

                  // Both sides longer than the 256 short-side bound => overflow, never a wrong
                  // silently-truncated number.
                  std::vector<int> big_a(300, 1);
                  std::vector<int> big_b(300, 2);
                  std::vector<int> off = {0, 300};
                  auto ov = gpu::edit_distance_batch(big_a, off, big_b, off);
                  t.expect(!ov.has_value() && ov.error() == MathError::overflow,
                           "both sides > 256 short-side limit => overflow");
              })
        .test("bfs_csr_distances",
              [](TestContext& t) {
                  // Undirected graph: 0-1, 0-2, 1-3, 2-3, 3-4, and an isolated vertex 5.
                  const std::vector<int> row_offsets = {0, 2, 4, 6, 9, 10, 10};
                  const std::vector<int> col_indices = {1, 2, 0, 3, 0, 3, 1, 2, 4, 3};
                  auto dist = gpu::bfs(row_offsets, col_indices, 0).value();
                  const std::vector<int> expected = {0, 1, 1, 2, 3, -1};
                  t.expect(dist.size() == expected.size(), "one distance per vertex");
                  bool ok = dist.size() == expected.size();
                  for (std::size_t i = 0; i < expected.size() && ok; ++i) {
                      if (dist[i] != expected[i]) {
                          ok = false;
                      }
                  }
                  t.expect(ok, "BFS distances match the hand-computed graph");
              })
        .test("nqueens_count_known",
              [](TestContext& t) {
                  t.expect(gpu::nqueens_count(4).value() == 2ull, "4-queens has 2 solutions");
                  t.expect(gpu::nqueens_count(6).value() == 4ull, "6-queens has 4 solutions");
                  t.expect(gpu::nqueens_count(8).value() == 92ull, "8-queens has 92 solutions");
                  t.expect(gpu::nqueens_count(10).value() == 724ull,
                           "10-queens has 724 solutions");
              })
        .test("qmc_poly_integrate_matches_cpu_mean",
              [](TestContext& t) {
                  // Integrand p(x) = 1 + 2x + 3x^2, whose exact integral over [0,1] is 3. Sample
                  // it at midpoints of a fine grid (a stand-in for a low-discrepancy point set).
                  const std::vector<double> coeffs = {1.0, 2.0, 3.0};
                  std::vector<double> points(4096);
                  for (std::size_t i = 0; i < points.size(); ++i) {
                      points[i] = (static_cast<double>(i) + 0.5) / static_cast<double>(points.size());
                  }
                  if (gpu::available()) {
                      auto got = gpu::qmc_poly_integrate(coeffs, points);
                      t.expect(got.has_value(), "estimate computed on the device");
                      // CPU reference: equal-weight average summed in index order.
                      double sum = 0.0;
                      for (const double xi : points) {
                          sum += cpu_poly_eval(coeffs, xi);
                      }
                      const double cpu_mean = sum / static_cast<double>(points.size());
                      t.expect(got && approx(*got, cpu_mean),
                               "GPU QMC mean matches the CPU reference (up to reduction-order bits)");
                      t.expect(got && std::abs(*got - 3.0) < 1e-3,
                               "estimate approximates the true integral 3");
                      // Empty point set has no defined average -> documented gpu_error.
                      const std::vector<double> none;
                      auto empty = gpu::qmc_poly_integrate(coeffs, none);
                      t.expect(!empty.has_value() && empty.error() == MathError::gpu_error,
                               "empty point set yields gpu_error");
                  } else {
                      // CUDA-disabled path: the wrapper returns the documented error so the
                      // default CI build passes without a device.
                      auto got = gpu::qmc_poly_integrate(coeffs, points);
                      t.expect(!got.has_value() && got.error() == MathError::gpu_error,
                               "CUDA-disabled path returns the documented gpu_error");
                  }
              })
        .test("haar_dwt_batch_matches_cpu",
              [](TestContext& t) {
                  // Two length-8 signal blocks laid out row-major.
                  const int batch = 2;
                  const int len = 8;
                  const std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                                                    8.0, 8.0, 0.0, 0.0, 4.0, 4.0, 2.0, 6.0};
                  if (gpu::available()) {
                      auto got = gpu::haar_dwt_batch(data, batch, len);
                      t.expect(got.has_value(), "batch transform computed on the device");
                      bool all = got.has_value() && got->size() == data.size();
                      for (int b = 0; b < batch && all; ++b) {
                          const std::span<const double> block{data.data() + b * len,
                                                              static_cast<std::size_t>(len)};
                          const auto ref = cpu_haar_level(block);
                          for (int i = 0; i < len; ++i) {
                              if (!approx((*got)[b * len + i], ref[i])) {
                                  all = false;
                              }
                          }
                      }
                      t.expect(all, "GPU Haar DWT matches the CPU reference block by block");
                      // Odd len is rejected up front, independent of the device.
                      const std::vector<double> odd(6, 1.0);
                      auto bad = gpu::haar_dwt_batch(odd, 2, 3);
                      t.expect(!bad.has_value() && bad.error() == MathError::domain_error,
                               "odd block length yields domain_error");
                      // Mismatched size (data.size() != batch*len) also fails on the railway.
                      auto mism = gpu::haar_dwt_batch(data, batch, 4);
                      t.expect(!mism.has_value() && mism.error() == MathError::domain_error,
                               "size mismatch yields domain_error");
                  } else {
                      // CUDA-disabled path returns the documented error (checked first, before the
                      // device-independent argument validation), so the default build passes.
                      auto got = gpu::haar_dwt_batch(data, batch, len);
                      t.expect(!got.has_value() && got.error() == MathError::gpu_error,
                               "CUDA-disabled path returns the documented gpu_error");
                  }
              })
        .test("batched_matmul_matches_cpu",
              [](TestContext& t) {
                  // Three independent problems of uniform shape (m x k) * (k x n), packed as one
                  // batch: A blocks are 2x3, B blocks are 3x2, so each product C_b is 2x2.
                  const int batch = 3;
                  const int m = 2;
                  const int k = 3;
                  const int n = 2;
                  const std::vector<double> a = {
                      1.0, 2.0, 3.0,  4.0, 5.0, 6.0,   // A0
                      1.0, 0.0, 0.0,  0.0, 1.0, 0.0,   // A1
                      2.0, -1.0, 0.0, 0.0, 3.0, 1.0};  // A2
                  const std::vector<double> b = {
                      7.0, 8.0, 9.0, 10.0, 11.0, 12.0,  // B0
                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0,     // B1
                      1.0, 0.0, 2.0, 1.0, 0.0, 4.0};    // B2
                  if (gpu::available()) {
                      auto got = gpu::batched_matmul(a, b, batch, m, k, n);
                      t.expect(got.has_value(), "batched matmul computed on the device");
                      const auto ref = cpu_batched_matmul(a, b, batch, m, k, n);
                      bool all = got.has_value() && got->size() == ref.size();
                      for (std::size_t i = 0; i < ref.size() && all; ++i) {
                          if (!approx((*got)[i], ref[i])) {
                              all = false;
                          }
                      }
                      t.expect(all, "GPU batched matmul matches the CPU reference block by block");
                      // Hand-checked C0 = A0 * B0 first row:
                      //   C0[0,0] = 1*7 + 2*9 + 3*11 = 58; C0[0,1] = 1*8 + 2*10 + 3*12 = 64.
                      t.expect(got && approx((*got)[0], 58.0) && approx((*got)[1], 64.0),
                               "hand-checked C0 first row is [58, 64]");
                      // Non-positive dimension is rejected up front.
                      auto baddim = gpu::batched_matmul(a, b, batch, m, 0, n);
                      t.expect(!baddim.has_value() && baddim.error() == MathError::domain_error,
                               "k = 0 yields domain_error");
                      // Span size disagreeing with the dimensions fails on the railway.
                      auto badsize = gpu::batched_matmul(a, b, batch, m, k, n + 1);
                      t.expect(!badsize.has_value() && badsize.error() == MathError::domain_error,
                               "size mismatch yields domain_error");
                  } else {
                      // CUDA-disabled path returns the documented error so the default build passes.
                      auto got = gpu::batched_matmul(a, b, batch, m, k, n);
                      t.expect(!got.has_value() && got.error() == MathError::gpu_error,
                               "CUDA-disabled path returns the documented gpu_error");
                  }
              })
        .test("fft_batch_matches_cpu_dft",
              [](TestContext& t) {
                  // Two length-4 signals, interleaved (re, im) per sample:
                  //   signal 0 = unit impulse [1,0,0,0]  -> DFT is all ones  [1,1,1,1]
                  //   signal 1 = constant     [1,1,1,1]  -> DFT is [4,0,0,0]
                  const int batch = 2;
                  const int n = 4;
                  const std::vector<double> in = {
                      1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   // signal 0: impulse
                      1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0};  // signal 1: constant 1
                  if (gpu::available()) {
                      auto got = gpu::fft_batch(in, batch, n);
                      t.expect(got.has_value(), "batch FFT computed on the device");
                      // Cross-check every element against the direct CPU DFT, signal by signal.
                      bool all = got.has_value() && got->size() == in.size();
                      for (int b = 0; b < batch && all; ++b) {
                          const std::span<const double> sig{
                              in.data() + static_cast<std::size_t>(b * 2 * n),
                              static_cast<std::size_t>(2 * n)};
                          const auto ref = cpu_dft(sig, n);
                          for (int i = 0; i < 2 * n; ++i) {
                              if (!approx((*got)[static_cast<std::size_t>(b * 2 * n + i)],
                                          ref[static_cast<std::size_t>(i)])) {
                                  all = false;
                              }
                          }
                      }
                      t.expect(all, "GPU FFT matches the CPU O(n^2) DFT block by block");
                      // Hand-checkable: FFT of the impulse [1,0,0,0] is [1,1,1,1] (all ones).
                      bool impulse_ok = got.has_value();
                      for (int k = 0; k < n && impulse_ok; ++k) {
                          impulse_ok = approx((*got)[static_cast<std::size_t>(2 * k)], 1.0) &&
                                       approx((*got)[static_cast<std::size_t>(2 * k + 1)], 0.0);
                      }
                      t.expect(impulse_ok, "FFT of [1,0,0,0] is [1,1,1,1]");
                      // Hand-checkable: FFT of the constant [1,1,1,1] is [4,0,0,0].
                      bool const_ok = got.has_value();
                      if (const_ok) {
                          const auto* s1 = got->data() + static_cast<std::size_t>(2 * n);
                          const_ok = approx(s1[0], 4.0) && approx(s1[1], 0.0);
                          for (int k = 1; k < n && const_ok; ++k) {
                              const_ok = approx(s1[static_cast<std::size_t>(2 * k)], 0.0) &&
                                         approx(s1[static_cast<std::size_t>(2 * k + 1)], 0.0);
                          }
                      }
                      t.expect(const_ok, "FFT of constant [1,1,1,1] is [4,0,0,0]");

                      // General n=8, batch=2 with arbitrary complex samples, cross-checked
                      // against the direct DFT to ~1e-9.
                      const int n8 = 8;
                      std::vector<double> in8(static_cast<std::size_t>(2 * 2 * n8));
                      for (int j = 0; j < n8; ++j) {
                          in8[static_cast<std::size_t>(2 * j)] = std::cos(0.7 * j) - 0.3 * j;
                          in8[static_cast<std::size_t>(2 * j + 1)] = std::sin(1.1 * j) + 0.2;
                          in8[static_cast<std::size_t>(2 * n8 + 2 * j)] = (j % 3 == 0) ? 2.0 : -1.0;
                          in8[static_cast<std::size_t>(2 * n8 + 2 * j + 1)] = 0.5 * j - 1.0;
                      }
                      auto got8 = gpu::fft_batch(in8, 2, n8);
                      t.expect(got8.has_value(), "n=8 batch FFT computed on the device");
                      bool all8 = got8.has_value() && got8->size() == in8.size();
                      for (int b = 0; b < 2 && all8; ++b) {
                          const std::span<const double> sig{
                              in8.data() + static_cast<std::size_t>(b * 2 * n8),
                              static_cast<std::size_t>(2 * n8)};
                          const auto ref = cpu_dft(sig, n8);
                          for (int i = 0; i < 2 * n8; ++i) {
                              if (!approx((*got8)[static_cast<std::size_t>(b * 2 * n8 + i)],
                                          ref[static_cast<std::size_t>(i)])) {
                                  all8 = false;
                              }
                          }
                      }
                      t.expect(all8, "GPU FFT (n=8) matches the CPU O(n^2) DFT block by block");

                      // Non-power-of-two length is rejected with domain_error (n=3, size 2*3).
                      const std::vector<double> in3(6, 0.0);
                      auto bad = gpu::fft_batch(in3, 1, 3);
                      t.expect(!bad.has_value() && bad.error() == MathError::domain_error,
                               "non-power-of-two n yields domain_error");
                      // Size mismatch (in.size() != batch*2*n) also fails on the railway.
                      auto mism = gpu::fft_batch(in, batch, 8);
                      t.expect(!mism.has_value() && mism.error() == MathError::domain_error,
                               "size mismatch yields domain_error");
                  } else {
                      // CUDA-disabled path returns the documented error so the default build passes
                      // without a device.
                      auto got = gpu::fft_batch(in, batch, n);
                      t.expect(!got.has_value() && got.error() == MathError::gpu_error,
                               "CUDA-disabled path returns the documented gpu_error");
                  }
              })
        .test("black_scholes_batch mirrors the CPU closed form; CUDA-graph replay agrees",
              [](TestContext& t) {
                  // A small option grid: calls and puts over a spot sweep at K=100, r=5%,
                  // vol=20%, T=1 (so the S=100 call is the textbook 10.4506).
                  std::vector<gpu::BsOption> opts;
                  for (double s : {80.0, 90.0, 100.0, 110.0, 120.0}) {
                      opts.push_back(gpu::BsOption{s, 100.0, 0.05, 0.0, 0.2, 1.0, true});
                      opts.push_back(gpu::BsOption{s, 100.0, 0.05, 0.0, 0.2, 1.0, false});
                  }
                  if (gpu::available()) {
                      auto got = gpu::black_scholes_batch(opts);
                      t.expect(got.has_value() && got->size() == opts.size(), "batch priced");
                      bool all_match = true;
                      for (std::size_t i = 0; i < opts.size(); ++i) {
                          all_match = all_match && approx((*got)[i], cpu_bs(opts[i]));
                      }
                      t.expect(all_match, "every GPU price matches the CPU closed form");
                      t.expect(std::abs((*got)[4] - 10.4505835) < 1e-4, "ATM 1y call == 10.4506");
                      // The CUDA-graph replay (4 iterations) yields identical prices.
                      auto graphed = gpu::black_scholes_batch_graphed(opts, 4);
                      t.expect(graphed.has_value(), "graphed batch ok");
                      bool identical = graphed->size() == got->size();
                      for (std::size_t i = 0; i < got->size() && identical; ++i) {
                          identical = (*graphed)[i] == (*got)[i];
                      }
                      t.expect(identical, "CUDA-graph replay is bit-identical to the direct launch");
                      // A non-physical option rides the railway.
                      std::vector<gpu::BsOption> bad{gpu::BsOption{-1.0, 100.0, 0.05, 0.0, 0.2, 1.0, true}};
                      auto br = gpu::black_scholes_batch(bad);
                      t.expect(!br.has_value() && br.error() == MathError::domain_error,
                               "non-physical spot -> domain_error");
                  } else {
                      auto got = gpu::black_scholes_batch(opts);
                      t.expect(!got.has_value() && got.error() == MathError::gpu_error,
                               "CUDA-disabled path returns gpu_error");
                  }
              })
        .test("cg_csr_solves_small_spd_system",
              [](TestContext& t) {
                  // SPD A = [[4,1],[1,3]], b = [1,2] in CSR. The exact solution is
                  //   x = A^{-1} b = [1/11, 7/11] ~= [0.0909090909, 0.6363636364]
                  // (check: 4*(1/11) + 1*(7/11) = 11/11 = 1; 1*(1/11) + 3*(7/11) = 22/11 = 2).
                  const std::vector<int> row_offsets = {0, 2, 4};
                  const std::vector<int> col_indices = {0, 1, 0, 1};
                  const std::vector<double> values = {4.0, 1.0, 1.0, 3.0};
                  const std::vector<double> b = {1.0, 2.0};
                  if (gpu::available()) {
                      auto got = gpu::cg_csr(row_offsets, col_indices, values, b, 100, 1e-12);
                      t.expect(got.has_value(), "CG solve computed on the device");
                      t.expect(got && got->x.size() == 2, "one solution entry per unknown");
                      const double x0 = 1.0 / 11.0;  // 0.09090909...
                      const double x1 = 7.0 / 11.0;  // 0.63636363...
                      t.expect(got && std::abs(got->x[0] - x0) < 1e-9 &&
                                   std::abs(got->x[1] - x1) < 1e-9,
                               "GPU CG matches the hand-verified exact solution [1/11, 7/11]");
                      t.expect(got && got->converged, "CG reports convergence on the SPD system");
                      // Malformed CSR (row_offsets length != n+1) rides the railway.
                      const std::vector<int> bad_row = {0, 2};
                      auto bad = gpu::cg_csr(bad_row, col_indices, values, b, 100, 1e-12);
                      t.expect(!bad.has_value() && bad.error() == MathError::domain_error,
                               "row_offsets length mismatch yields domain_error");
                  } else {
                      // CUDA-disabled path returns the documented error so the default build passes.
                      auto got = gpu::cg_csr(row_offsets, col_indices, values, b, 100, 1e-12);
                      t.expect(!got.has_value() && got.error() == MathError::gpu_error,
                               "CUDA-disabled path returns the documented gpu_error");
                  }
              })
        .test("monte_carlo_european_batch mirrors the CPU counter-based MC",
              [](TestContext& t) {
                  namespace pr = nimblecas::pricing;
                  // Three contracts sharing r=5%, vol=20%, T=1: ATM call, OTM call, ITM put.
                  std::vector<gpu::BsOption> opts = {
                      gpu::BsOption{100.0, 100.0, 0.05, 0.0, 0.2, 1.0, true},
                      gpu::BsOption{100.0, 110.0, 0.05, 0.0, 0.2, 1.0, true},
                      gpu::BsOption{100.0, 110.0, 0.05, 0.0, 0.2, 1.0, false}};
                  const std::uint64_t paths = 200000;
                  const std::uint64_t seed = 42;
                  auto got = gpu::monte_carlo_european_batch(opts, paths, seed);
                  t.expect(got.has_value() && got->size() == opts.size(), "batch MC priced");
                  bool mirror = got.has_value();
                  bool stat = got.has_value();
                  for (std::size_t i = 0; i < opts.size() && mirror; ++i) {
                      const auto spec = pr::OptionSpec{}
                                            .with_spot(opts[i].spot).with_strike(opts[i].strike)
                                            .with_rate(opts[i].rate).with_dividend(opts[i].dividend)
                                            .with_volatility(opts[i].volatility)
                                            .with_expiry(opts[i].time)
                                            .with_type(opts[i].is_call ? pr::OptionType::call
                                                                       : pr::OptionType::put);
                      const auto cpu = pr::monte_carlo_european(spec, paths, seed).value();
                      // FP-reassociation + device-libm last bits only: far below the MC std error.
                      mirror = std::abs((*got)[i].price - cpu.price) < 1e-6 &&
                               std::abs((*got)[i].std_error - cpu.std_error) < 1e-6;
                      const double bs = pr::black_scholes_price(spec).value();
                      stat = stat && std::abs((*got)[i].price - bs) <
                                         4.0 * (*got)[i].std_error + 1e-9;
                  }
                  t.expect(mirror, "every GPU MC estimate equals the CPU counter-based MC to 1e-6");
                  t.expect(stat, "every estimate within 4 standard errors of Black-Scholes");
                  // Reproducibility: a second identical call is bit-identical (pure function of
                  // (opts, paths, seed) — never of launch geometry, threads, or time).
                  auto again = gpu::monte_carlo_european_batch(opts, paths, seed);
                  bool identical = again.has_value() && again->size() == got->size();
                  for (std::size_t i = 0; i < got->size() && identical; ++i) {
                      identical = (*again)[i].price == (*got)[i].price &&
                                  (*again)[i].std_error == (*got)[i].std_error;
                  }
                  t.expect(identical, "repeated batch MC is bit-identical (equal seeds, equal bits)");
                  // Domain guards ride the railway.
                  t.expect(!gpu::monte_carlo_european_batch(opts, 0, seed).has_value(),
                           "zero paths -> error");
                  std::vector<gpu::BsOption> bad{gpu::BsOption{-1.0, 100.0, 0.05, 0.0, 0.2, 1.0, true}};
                  auto br = gpu::monte_carlo_european_batch(bad, paths, seed);
                  t.expect(!br.has_value() && br.error() == MathError::domain_error,
                           "non-physical spot -> domain_error");
              })
        .test("black_scholes_greeks_batch matches the CPU closed form field by field",
              [](TestContext& t) {
                  namespace pr = nimblecas::pricing;
                  std::vector<gpu::BsOption> opts;
                  for (double s : {80.0, 100.0, 120.0}) {
                      opts.push_back(gpu::BsOption{s, 100.0, 0.05, 0.01, 0.2, 1.0, true});
                      opts.push_back(gpu::BsOption{s, 100.0, 0.05, 0.01, 0.2, 1.0, false});
                  }
                  opts.push_back(gpu::BsOption{100.0, 90.0, 0.05, 0.0, 0.2, 0.0, true});  // T==0 branch
                  auto got = gpu::black_scholes_greeks_batch(opts);
                  t.expect(got.has_value() && got->size() == opts.size(), "batch Greeks computed");
                  auto near = [](double a, double b) {
                      return std::abs(a - b) <= 1e-9 * (1.0 + std::abs(b));
                  };
                  bool all = got.has_value();
                  for (std::size_t i = 0; i < opts.size() && all; ++i) {
                      const auto spec = pr::OptionSpec{}
                                            .with_spot(opts[i].spot).with_strike(opts[i].strike)
                                            .with_rate(opts[i].rate).with_dividend(opts[i].dividend)
                                            .with_volatility(opts[i].volatility)
                                            .with_expiry(opts[i].time)
                                            .with_type(opts[i].is_call ? pr::OptionType::call
                                                                       : pr::OptionType::put);
                      const auto cpu = pr::black_scholes_greeks(spec).value();
                      const auto& g = (*got)[i];
                      all = near(g.price, cpu.price) && near(g.delta, cpu.delta) &&
                            near(g.gamma, cpu.gamma) && near(g.vega, cpu.vega) &&
                            near(g.theta, cpu.theta) && near(g.rho, cpu.rho);
                  }
                  t.expect(all, "every field of every option matches the CPU closed form to 1e-9");
                  // Hand oracle: the ATM 1y call (r=5%, q=1%, vol=20%): d1 = (0.04 + 0.02)/0.2 = 0.3,
                  // delta = e^{-0.01} * N(0.3) ~= 0.99005 * 0.61791 ~= 0.61177.
                  t.expect(std::abs((*got)[2].delta - 0.61177) < 1e-4,
                           "hand-checked ATM call delta ~= 0.61177");
                  // T==0 degenerate: price is the intrinsic 10, delta is the limit e^{-q*0} = 1.
                  t.expect(near((*got)[6].price, 10.0) && near((*got)[6].delta, 1.0),
                           "T==0 ITM call collapses to intrinsic 10 with limit delta 1");
              })
        .test("black_scholes_extended_greeks_batch matches the CPU extended set",
              [](TestContext& t) {
                  namespace pr = nimblecas::pricing;
                  std::vector<gpu::BsOption> opts = {
                      gpu::BsOption{100.0, 100.0, 0.05, 0.0, 0.2, 1.0, true},
                      gpu::BsOption{100.0, 110.0, 0.03, 0.02, 0.35, 0.5, false},
                      gpu::BsOption{90.0, 100.0, 0.01, 0.0, 0.15, 2.0, true}};
                  auto got = gpu::black_scholes_extended_greeks_batch(opts);
                  t.expect(got.has_value() && got->size() == opts.size(), "extended batch computed");
                  auto near7 = [](double a, double b) {
                      return std::abs(a - b) <= 1e-7 * std::max(1.0, std::abs(b));
                  };
                  bool all = got.has_value();
                  for (std::size_t i = 0; i < opts.size() && all; ++i) {
                      const auto spec = pr::OptionSpec{}
                                            .with_spot(opts[i].spot).with_strike(opts[i].strike)
                                            .with_rate(opts[i].rate).with_dividend(opts[i].dividend)
                                            .with_volatility(opts[i].volatility)
                                            .with_expiry(opts[i].time)
                                            .with_type(opts[i].is_call ? pr::OptionType::call
                                                                       : pr::OptionType::put);
                      const auto cpu = pr::black_scholes_extended_greeks(spec).value();
                      const auto& g = (*got)[i];
                      all = near7(g.vanna, cpu.vanna) && near7(g.charm, cpu.charm) &&
                            near7(g.vomma, cpu.vomma) && near7(g.veta, cpu.veta) &&
                            near7(g.speed, cpu.speed) && near7(g.zomma, cpu.zomma) &&
                            near7(g.color, cpu.color) && near7(g.lambda, cpu.lambda) &&
                            near7(g.dual_delta, cpu.dual_delta) && near7(g.dual_gamma, cpu.dual_gamma) &&
                            near7(g.epsilon, cpu.epsilon) && near7(g.vera, cpu.vera) &&
                            near7(g.ultima, cpu.ultima);
                  }
                  t.expect(all, "all 13 extended fields match the CPU set to 1e-7");
                  // T == 0 violates the strict extended-set domain -> domain_error, never a number.
                  std::vector<gpu::BsOption> bad{gpu::BsOption{100.0, 100.0, 0.05, 0.0, 0.2, 0.0, true}};
                  auto br = gpu::black_scholes_extended_greeks_batch(bad);
                  t.expect(!br.has_value() && br.error() == MathError::domain_error,
                           "T == 0 -> domain_error for the extended set");
              })
        .test("strategy_pnl_grid equals the exact piecewise-linear CPU oracle",
              [](TestContext& t) {
                  namespace os = nimblecas::optstrat;
                  // Bull call spread 95/105, premiums 6.5/2.5 -> net debit 4. Exact P&L:
                  // s <= 95: -4; s = 99: 0 (the breakeven 95 + 4); s >= 105: +6 (width 10 - 4).
                  const auto spread = os::bull_call_spread(95.0, 6.5, 105.0, 2.5);
                  const std::vector<double> grid = {80.0, 90.0, 95.0, 99.0, 100.0, 105.0, 110.0, 120.0};
                  auto got = gpu::strategy_pnl_grid(spread.legs(), grid);
                  t.expect(got.has_value() && got->size() == grid.size(), "one P&L per grid point");
                  bool all = got.has_value();
                  for (std::size_t j = 0; j < grid.size() && all; ++j) {
                      const double cpu = spread.pnl_at(grid[j]);
                      all = std::abs((*got)[j] - cpu) <= 1e-12 * std::max(1.0, std::abs(cpu));
                  }
                  t.expect(all, "GPU sweep equals OptionStrategy::pnl_at at every grid point");
                  // Hand-computed exact values (all representable doubles): breakeven, floor, cap.
                  t.expect(got && (*got)[3] == 0.0, "P&L at the exact breakeven 99 is exactly 0");
                  t.expect(got && (*got)[0] == -4.0 && (*got)[7] == 6.0,
                           "floor -4 below 95 and cap +6 above 105, exactly");
                  // Gross payoff sweep: max(s-95,0) - max(s-105,0); at 110 that is 15 - 5 = 10.
                  auto pay = gpu::strategy_payoff_grid(spread.legs(), grid);
                  t.expect(pay.has_value() && (*pay)[6] == 10.0 && (*pay)[0] == 0.0,
                           "gross payoff sweep: 0 below, exactly 10 at/above the cap");
                  // Empty grid -> empty result on the same railway.
                  t.expect(gpu::strategy_pnl_grid(spread.legs(), std::vector<double>{})
                               .value().empty(),
                           "empty grid yields an empty sweep");
              })
        .test("strategy grids handle underlying legs and multi-leg books",
              [](TestContext& t) {
                  namespace os = nimblecas::optstrat;
                  // Covered call: long underlying at 100, short 105 call at 3. Exact P&L:
                  // s = 90: -7; breakeven 97: 0; s >= 105: capped at 8 (105 - 100 + 3).
                  const auto cc = os::covered_call(100.0, 105.0, 3.0);
                  const std::vector<double> grid = {90.0, 97.0, 100.0, 105.0, 130.0};
                  auto got = gpu::strategy_pnl_grid(cc.legs(), grid);
                  t.expect(got.has_value(), "covered-call sweep computed");
                  t.expect(got && (*got)[0] == -7.0 && (*got)[1] == 0.0 && (*got)[3] == 8.0 &&
                               (*got)[4] == 8.0,
                           "hand-checked covered-call P&L: -7 / 0 at breakeven 97 / capped 8");
                  // Iron condor 90/95/105/110 (premiums 1, 2.5, 2.6, 1.1): cross-check every point
                  // against the exact CPU analytics oracle, including between and beyond strikes.
                  const auto ic = os::iron_condor(90.0, 1.0, 95.0, 2.5, 105.0, 2.6, 110.0, 1.1);
                  std::vector<double> wide;
                  for (double s = 80.0; s <= 120.0; s += 1.0) { wide.push_back(s); }
                  auto sweep = gpu::strategy_pnl_grid(ic.legs(), wide);
                  t.expect(sweep.has_value() && sweep->size() == wide.size(), "condor sweep sized");
                  bool all = sweep.has_value();
                  for (std::size_t j = 0; j < wide.size() && all; ++j) {
                      const double cpu = ic.pnl_at(wide[j]);
                      all = std::abs((*sweep)[j] - cpu) <= 1e-12 * std::max(1.0, std::abs(cpu));
                  }
                  t.expect(all, "iron-condor sweep equals the exact piecewise-linear oracle");
              })
        .test("strategy sweep is bit-exact vs the CPU oracle on non-representable products",
              [](TestContext& t) {
                  namespace os = nimblecas::optstrat;
                  // Deliberately "ugly" book: quantities/strikes/premiums whose products are
                  // NOT exactly representable, so a fused-vs-unfused accumulation would diverge
                  // in the last bit. Both the GPU kernel (non-contracted __d*_rn) and the CPU
                  // optstrat oracle (pinned FP_CONTRACT off) round identically, so every grid
                  // point must match BIT-FOR-BIT. This is the test that would catch a
                  // regression in that parity — exact ==, not a tolerance.
                  auto book = os::OptionStrategy::create();
                  std::ignore = book.with_leg(os::StrategyLeg{os::LegKind::call, 100.3, 0.3, 1.1});
                  std::ignore = book.with_leg(os::StrategyLeg{os::LegKind::put, 95.7, -0.7, 2.9});
                  std::ignore =
                      book.with_leg(os::StrategyLeg{os::LegKind::underlying, 0.0, 0.1, 100.9});
                  std::vector<double> grid;
                  for (double s = 80.3; s <= 120.0; s += 1.7) { grid.push_back(s); }
                  auto pnl = gpu::strategy_pnl_grid(book.legs(), grid);
                  auto pay = gpu::strategy_payoff_grid(book.legs(), grid);
                  t.expect(pnl.has_value() && pay.has_value() &&
                               pnl->size() == grid.size() && pay->size() == grid.size(),
                           "ugly-book sweeps computed");
                  bool exact = pnl.has_value() && pay.has_value();
                  for (std::size_t j = 0; j < grid.size() && exact; ++j) {
                      exact = (*pnl)[j] == book.pnl_at(grid[j]) &&
                              (*pay)[j] == book.payoff_at(grid[j]);
                  }
                  t.expect(exact, "GPU sweep is bit-for-bit identical to the CPU oracle (exact ==)");
              })
        .test("futures_pnl_grid mirrors FuturesStrategy::pnl_at_uniform",
              [](TestContext& t) {
                  namespace fu = nimblecas::futures;
                  // Outright: long 2 contracts, size 50, entry 100 -> P&L = 100*(s - 100) exactly.
                  const auto outright = fu::long_futures("CLZ6", 100.0, 2.0, 50.0);
                  const std::vector<double> grid = {95.0, 100.0, 101.0, 110.0};
                  auto got = gpu::futures_pnl_grid(outright.legs(), grid);
                  t.expect(got.has_value() && got->size() == grid.size(), "one P&L per grid point");
                  t.expect(got && (*got)[0] == -500.0 && (*got)[1] == 0.0 && (*got)[2] == 100.0 &&
                               (*got)[3] == 1000.0,
                           "hand-checked outright futures P&L: 100*(s-100) exactly");
                  // Matched calendar spread (long near at 102, short far at 100, qty 1, size 1):
                  // net exposure is zero, so the uniform-settlement P&L is a locked CONSTANT
                  // (1*(s-102) - 1*(s-100) = -2) at EVERY price. Validate against the CPU oracle
                  // rather than the hand constant to stay robust to the builder's leg convention.
                  const auto cal = fu::calendar_spread("CLZ6", 102.0, "CLM7", 100.0);
                  auto sweep = gpu::futures_pnl_grid(cal.legs(), grid);
                  t.expect(sweep.has_value(), "calendar-spread sweep computed");
                  bool all = sweep.has_value();
                  for (std::size_t j = 0; j < grid.size() && all; ++j) {
                      const double cpu = cal.pnl_at_uniform(grid[j]);
                      all = std::abs((*sweep)[j] - cpu) <= 1e-12 * std::max(1.0, std::abs(cpu));
                  }
                  t.expect(all, "matched spread sweep equals pnl_at_uniform at every point");
                  // Empty legs -> a well-defined all-zero sweep (the empty book's P&L).
                  const auto empty = fu::FuturesStrategy::create();
                  auto zs = gpu::futures_pnl_grid(empty.legs(), grid);
                  t.expect(zs.has_value() && (*zs)[0] == 0.0 && (*zs)[3] == 0.0,
                           "empty book sweeps to exactly zero");
              })
        .test("monte_carlo_asian_batch mirrors the CPU Asian MC",
              [](TestContext& t) {
                  namespace pr = nimblecas::pricing;
                  // Batch of >1 options sharing stream: ATM call, OTM call, ITM put.
                  std::vector<gpu::BsOption> opts = {
                      gpu::BsOption{100.0, 100.0, 0.05, 0.0, 0.2, 1.0, true},
                      gpu::BsOption{100.0, 110.0, 0.05, 0.0, 0.2, 1.0, true},
                      gpu::BsOption{100.0, 110.0, 0.05, 0.0, 0.2, 1.0, false}};
                  const std::uint64_t paths = 200000;
                  const int steps = 64;
                  const std::uint64_t seed = 42;
                  auto got = gpu::monte_carlo_asian_batch(opts, paths, steps, seed);
                  t.expect(got.has_value() && got->size() == opts.size(), "asian batch MC priced");
                  bool mirror = got.has_value();
                  for (std::size_t i = 0; i < opts.size() && mirror; ++i) {
                      const auto spec = pr::OptionSpec{}
                                            .with_spot(opts[i].spot).with_strike(opts[i].strike)
                                            .with_rate(opts[i].rate).with_dividend(opts[i].dividend)
                                            .with_volatility(opts[i].volatility)
                                            .with_expiry(opts[i].time)
                                            .with_type(opts[i].is_call ? pr::OptionType::call
                                                                       : pr::OptionType::put);
                      const auto cpu = pr::monte_carlo_asian(spec, paths, steps, seed, false).value();
                      const double price_diff = std::abs((*got)[i].price - cpu.price);
                      const double tol = 1e-6 * std::max(1.0, std::abs(cpu.price));
                      const double se_diff = std::abs((*got)[i].std_error - cpu.std_error);
                      const double se_tol = 1e-6 * std::max(1.0, std::abs(cpu.std_error));
                      mirror = (price_diff <= tol) && (se_diff <= se_tol);
                  }
                  t.expect(mirror, "every GPU Asian MC estimate equals CPU monte_carlo_asian(..., false) to stated 1e-6 tolerance");

                  // Determinism: same inputs twice -> identical bits (==).
                  auto again = gpu::monte_carlo_asian_batch(opts, paths, steps, seed);
                  bool identical = again.has_value() && again->size() == got->size();
                  for (std::size_t i = 0; i < got->size() && identical; ++i) {
                      identical = ((*again)[i].price == (*got)[i].price) &&
                                  ((*again)[i].std_error == (*got)[i].std_error);
                  }
                  t.expect(identical, "repeated Asian MC is bit-identical (equal seeds, equal bits)");

                  // Domain guards ride the railway.
                  t.expect(!gpu::monte_carlo_asian_batch(opts, 0, steps, seed).has_value(),
                           "zero paths -> domain_error");
                  t.expect(!gpu::monte_carlo_asian_batch(opts, paths, 0, seed).has_value(),
                           "steps < 1 -> domain_error");
                  std::vector<gpu::BsOption> bad{gpu::BsOption{-1.0, 100.0, 0.05, 0.0, 0.2, 1.0, true}};
                  auto br = gpu::monte_carlo_asian_batch(bad, paths, steps, seed);
                  t.expect(!br.has_value() && br.error() == MathError::domain_error,
                           "non-physical spot -> domain_error");
              })
        .test("barrier_option_mc_batch mirrors CPU barrier_option_mc",
              [](TestContext& t) {
                  namespace pr = nimblecas::pricing;
                  // Non-grazing barriers: spot 100 with barrier 80 (down) and 120 (up), 0.8x/1.2x.
                  const gpu::BsOption opt{100.0, 100.0, 0.05, 0.0, 0.2, 1.0, true};
                  const std::vector<gpu::BsOption> opts = {opt};
                  const auto spec_call = pr::OptionSpec{}
                                             .with_spot(opt.spot).with_strike(opt.strike)
                                             .with_rate(opt.rate).with_dividend(opt.dividend)
                                             .with_volatility(opt.volatility).with_expiry(opt.time)
                                             .with_type(pr::OptionType::call);
                  const std::uint64_t paths = 100000;
                  const int steps = 50;
                  const std::uint64_t seed = 12345;

                  struct Case { double barrier; bool knock_in; std::string name; };
                  const std::vector<Case> cases = {
                      {80.0, false, "down-and-out"},  {80.0, true, "down-and-in"},
                      {120.0, false, "up-and-out"},   {120.0, true, "up-and-in"}};

                  for (const auto& c : cases) {
                      auto got = gpu::barrier_option_mc_batch(opts, c.barrier, c.knock_in, paths, steps, seed);
                      auto cpu = pr::barrier_option_mc(spec_call, c.barrier, c.knock_in, paths, steps, seed);
                      t.expect(got.has_value() && cpu.has_value(), "barrier MC executed (" + c.name + ")");
                      if (got && cpu) {
                          const double tol = 1e-5 * (1.0 + std::abs(cpu->price));
                          t.expect(std::abs((*got)[0].price - cpu->price) <= tol,
                                   "GPU barrier price matches CPU within documented tolerance (" + c.name + ")");
                          const double se_tol = 1e-5 * (1.0 + std::abs(cpu->std_error));
                          t.expect(std::abs((*got)[0].std_error - cpu->std_error) <= se_tol,
                                   "GPU barrier std_error matches CPU within tolerance (" + c.name + ")");
                      }
                  }

                  // Grazing-immune identity: knock_in + knock_out == vanilla per path, so the SUM of
                  // the two legs is insensitive to any ULP-level knock flip (a flip merely moves a
                  // path's terminal payoff between the legs; the sum counts it exactly once). The GPU
                  // sum therefore tracks the CPU sum to the tight ~1e-6 exp bound even at a barrier
                  // where an individual leg could diverge more.
                  for (double b : {80.0, 120.0}) {
                      auto gin = gpu::barrier_option_mc_batch(opts, b, true, paths, steps, seed);
                      auto gout = gpu::barrier_option_mc_batch(opts, b, false, paths, steps, seed);
                      auto cin = pr::barrier_option_mc(spec_call, b, true, paths, steps, seed);
                      auto cout = pr::barrier_option_mc(spec_call, b, false, paths, steps, seed);
                      if (gin && gout && cin && cout) {
                          const double gsum = (*gin)[0].price + (*gout)[0].price;
                          const double csum = cin->price + cout->price;
                          t.expect(std::abs(gsum - csum) <= 1e-6 * (1.0 + std::abs(csum)),
                                   "knock_in+knock_out sum is grazing-immune and matches CPU to 1e-6");
                      }
                  }
              })
        .test("barrier_option_mc_batch determinism and batching",
              [](TestContext& t) {
                  // Batch of > 1 options sharing the stream.
                  const std::vector<gpu::BsOption> opts = {
                      gpu::BsOption{100.0, 100.0, 0.05, 0.0, 0.2, 1.0, true},
                      gpu::BsOption{100.0, 110.0, 0.05, 0.0, 0.2, 1.0, true}};
                  const double barrier = 85.0;
                  const bool knock_in = false;
                  const std::uint64_t paths = 50000;
                  const int steps = 20;
                  const std::uint64_t seed = 999;

                  auto run1 = gpu::barrier_option_mc_batch(opts, barrier, knock_in, paths, steps, seed);
                  auto run2 = gpu::barrier_option_mc_batch(opts, barrier, knock_in, paths, steps, seed);

                  t.expect(run1.has_value() && run1->size() == 2, "batch of 2 options evaluated");
                  t.expect(run2.has_value() && run2->size() == 2, "second run evaluated");
                  if (run1 && run2 && run1->size() == 2 && run2->size() == 2) {
                      t.expect((*run1)[0].price == (*run2)[0].price && (*run1)[0].std_error == (*run2)[0].std_error,
                               "option 0 deterministic across runs");
                      t.expect((*run1)[1].price == (*run2)[1].price && (*run1)[1].std_error == (*run2)[1].std_error,
                               "option 1 deterministic across runs");
                  }
              })
        .test("barrier_option_mc_batch domain guards",
              [](TestContext& t) {
                  const std::vector<gpu::BsOption> opts = {
                      gpu::BsOption{100.0, 100.0, 0.05, 0.0, 0.2, 1.0, true}};

                  auto r_paths = gpu::barrier_option_mc_batch(opts, 90.0, false, 0, 10, 42);
                  t.expect(!r_paths.has_value() && r_paths.error() == MathError::domain_error, "paths == 0 yields domain_error");

                  auto r_steps = gpu::barrier_option_mc_batch(opts, 90.0, false, 1000, 0, 42);
                  t.expect(!r_steps.has_value() && r_steps.error() == MathError::domain_error, "steps < 1 yields domain_error");

                  auto r_barrier = gpu::barrier_option_mc_batch(opts, 0.0, false, 1000, 10, 42);
                  t.expect(!r_barrier.has_value() && r_barrier.error() == MathError::domain_error, "barrier <= 0 yields domain_error");

                  // Physical domain matches the CPU oracle on both device and fallback paths.
                  const std::vector<gpu::BsOption> bad_time = {
                      gpu::BsOption{100.0, 100.0, 0.05, 0.0, 0.2, 0.0, true}};
                  auto r_time = gpu::barrier_option_mc_batch(bad_time, 90.0, false, 1000, 10, 42);
                  t.expect(!r_time.has_value() && r_time.error() == MathError::domain_error, "time <= 0 yields domain_error");
              })
        .test("longstaff_schwartz_american_batch pricing and invariants",
              [](TestContext& t) {
                  namespace pr = nimblecas::pricing;

                  // 1. Bitwise repeatability (same inputs twice -> exact ==).
                  const std::vector<gpu::BsOption> opts = {
                      gpu::BsOption{100.0, 100.0, 0.05, 0.0, 0.2, 1.0, false},  // ATM Put
                      gpu::BsOption{100.0, 90.0, 0.05, 0.0, 0.2, 1.0, false},   // OTM Put
                      gpu::BsOption{100.0, 110.0, 0.05, 0.0, 0.2, 1.0, false},  // ITM Put
                      gpu::BsOption{100.0, 90.0, 0.05, 0.0, 0.2, 1.0, true}    // ITM Call
                  };
                  const std::uint64_t paths = 20000;
                  const int steps = 25;
                  const std::uint64_t seed = 42;

                  auto run1 = gpu::longstaff_schwartz_american_batch(opts, paths, steps, seed);
                  auto run2 = gpu::longstaff_schwartz_american_batch(opts, paths, steps, seed);
                  t.expect(run1.has_value() && run1->size() == opts.size(), "LSM American batch evaluated");
                  t.expect(run2.has_value() && run2->size() == opts.size(), "second LSM run evaluated");

                  if (run1 && run2 && run1->size() == opts.size() && run2->size() == opts.size()) {
                      bool repeat_exact = true;
                      for (std::size_t i = 0; i < opts.size(); ++i) {
                          if ((*run1)[i].price != (*run2)[i].price || (*run1)[i].std_error != (*run2)[i].std_error) {
                              repeat_exact = false;
                          }
                      }
                      t.expect(repeat_exact, "repeated LSM batch runs are bit-identical");
                  }

                  // 2. Intrinsic lower bound: price >= max(S-K, 0) for call, max(K-S, 0) for put (EXACT >=)
                  if (run1) {
                      bool bound_ok = true;
                      for (std::size_t i = 0; i < opts.size(); ++i) {
                          const double intrinsic = opts[i].is_call ? std::max(opts[i].spot - opts[i].strike, 0.0)
                                                                   : std::max(opts[i].strike - opts[i].spot, 0.0);
                          if ((*run1)[i].price < intrinsic) {
                              bound_ok = false;
                          }
                      }
                      t.expect(bound_ok, "American price >= intrinsic payoff EXACTLY for every option");
                  }

                  // 3. American put >= European put price - 3*std_error
                  if (run1) {
                      bool put_ge_euro = true;
                      for (std::size_t i = 0; i < opts.size(); ++i) {
                          if (!opts[i].is_call) {
                              const auto spec = pr::OptionSpec{}
                                                    .with_spot(opts[i].spot).with_strike(opts[i].strike)
                                                    .with_rate(opts[i].rate).with_dividend(opts[i].dividend)
                                                    .with_volatility(opts[i].volatility).with_expiry(opts[i].time)
                                                    .with_type(pr::OptionType::put);
                              const double euro_put = pr::black_scholes_price(spec).value();
                              if ((*run1)[i].price < euro_put - 3.0 * (*run1)[i].std_error) {
                                  put_ge_euro = false;
                              }
                          }
                      }
                      t.expect(put_ge_euro, "American put price >= European put price - 3*std_error");
                  }

                  // 4. q=0 American CALL ~= European call within 4*std_error (never optimal to early-exercise call when q=0)
                  if (run1) {
                      const auto call_spec = pr::OptionSpec{}
                                                 .with_spot(opts[3].spot).with_strike(opts[3].strike)
                                                 .with_rate(opts[3].rate).with_dividend(opts[3].dividend)
                                                 .with_volatility(opts[3].volatility).with_expiry(opts[3].time)
                                                 .with_type(pr::OptionType::call);
                      const double euro_call = pr::black_scholes_price(call_spec).value();
                      const double call_diff = std::abs((*run1)[3].price - euro_call);
                      t.expect(call_diff <= 4.0 * (*run1)[3].std_error + 1e-4,
                               "q=0 American call price matches European call within 4*std_error");
                  }

                  // 5. Strike monotonicity within one shared-draw batch: put price increasing in K
                  // Opt 1: K=90 put, Opt 0: K=100 put, Opt 2: K=110 put
                  if (run1) {
                      const double p90 = (*run1)[1].price;
                      const double p100 = (*run1)[0].price;
                      const double p110 = (*run1)[2].price;
                      t.expect(p90 <= p100 + 3.0 * (*run1)[1].std_error && p100 <= p110 + 3.0 * (*run1)[0].std_error,
                               "American put prices are monotonic in strike (K=90 <= K=100 <= K=110)");
                  }

                  // 6. Loose match to CPU longstaff_schwartz_american on ONE well-conditioned fixed-seed ATM put
                  const std::vector<gpu::BsOption> atm_put = {
                      gpu::BsOption{100.0, 100.0, 0.05, 0.0, 0.2, 1.0, false}
                  };
                  const std::uint64_t cpu_paths = 50000;
                  const int cpu_steps = 50;
                  const std::uint64_t cpu_seed = 42;

                  auto got_atm = gpu::longstaff_schwartz_american_batch(atm_put, cpu_paths, cpu_steps, cpu_seed);
                  const auto spec_atm = pr::OptionSpec{}
                                             .with_spot(100.0).with_strike(100.0)
                                             .with_rate(0.05).with_dividend(0.0)
                                             .with_volatility(0.2).with_expiry(1.0)
                                             .with_type(pr::OptionType::put);
                  auto cpu_atm = pr::longstaff_schwartz_american(spec_atm, cpu_paths, cpu_steps, cpu_seed);

                  t.expect(got_atm.has_value() && cpu_atm.has_value(), "ATM put evaluated on GPU and CPU");
                  if (got_atm && cpu_atm) {
                      const double price_diff = std::abs((*got_atm)[0].price - cpu_atm->price);
                      const double tol = 1e-3 * (1.0 + std::abs(cpu_atm->price));
                      t.expect(price_diff <= tol, "GPU LSM price matches CPU LSM price within 1e-3 relative tolerance");
                      // std_error must also track the CPU (a wrong variance divisor would slip past a
                      // price-only check); a loose relative bound stays robust to the few exercise flips.
                      const double se_diff = std::abs((*got_atm)[0].std_error - cpu_atm->std_error);
                      t.expect(se_diff <= 1e-2 * (1.0 + std::abs(cpu_atm->std_error)),
                               "GPU LSM std_error matches CPU LSM std_error within tolerance");
                  }

                  // 7. Domain guards — exact parity with pricing::longstaff_schwartz_american, each
                  //     asserting the error TYPE (domain_error), applied before the device/fallback split.
                  auto r_paths = gpu::longstaff_schwartz_american_batch(opts, 3, steps, seed);
                  t.expect(!r_paths.has_value() && r_paths.error() == MathError::domain_error,
                           "paths < 4 -> domain_error");
                  auto r_steps0 = gpu::longstaff_schwartz_american_batch(opts, paths, 0, seed);
                  t.expect(!r_steps0.has_value() && r_steps0.error() == MathError::domain_error,
                           "steps < 1 -> domain_error");
                  auto r_bigsteps = gpu::longstaff_schwartz_american_batch(opts, paths, 100001, seed);
                  t.expect(!r_bigsteps.has_value() && r_bigsteps.error() == MathError::domain_error,
                           "steps > 100000 -> domain_error");
                  // paths*(steps+1) > kMaxCells (5e8): steps=100, paths=6e6 -> ~6.06e8 cells.
                  auto r_cells = gpu::longstaff_schwartz_american_batch(opts, 6'000'000, 100, seed);
                  t.expect(!r_cells.has_value() && r_cells.error() == MathError::domain_error,
                           "paths*(steps+1) over kMaxCells -> domain_error");

                  std::vector<gpu::BsOption> bad_vol = {gpu::BsOption{100.0, 100.0, 0.05, 0.0, 0.0, 1.0, false}};
                  auto r_vol = gpu::longstaff_schwartz_american_batch(bad_vol, paths, steps, seed);
                  t.expect(!r_vol.has_value() && r_vol.error() == MathError::domain_error,
                           "volatility <= 0 -> domain_error");
                  std::vector<gpu::BsOption> bad_time = {gpu::BsOption{100.0, 100.0, 0.05, 0.0, 0.2, 0.0, false}};
                  auto r_time = gpu::longstaff_schwartz_american_batch(bad_time, paths, steps, seed);
                  t.expect(!r_time.has_value() && r_time.error() == MathError::domain_error,
                           "time <= 0 -> domain_error");
                  std::vector<gpu::BsOption> bad_spot = {gpu::BsOption{0.0, 100.0, 0.05, 0.0, 0.2, 1.0, false}};
                  auto r_spot = gpu::longstaff_schwartz_american_batch(bad_spot, paths, steps, seed);
                  t.expect(!r_spot.has_value() && r_spot.error() == MathError::domain_error,
                           "spot <= 0 -> domain_error");
                  std::vector<gpu::BsOption> bad_strike = {gpu::BsOption{100.0, 0.0, 0.05, 0.0, 0.2, 1.0, false}};
                  auto r_strike = gpu::longstaff_schwartz_american_batch(bad_strike, paths, steps, seed);
                  t.expect(!r_strike.has_value() && r_strike.error() == MathError::domain_error,
                           "strike <= 0 -> domain_error (shared batch-POD precondition, both device and fallback)");

                  // 8. Intrinsic-clamp-binding case: deep-ITM low-vol short-T American put, where immediate
                  //     exercise dominates so price = max(mean, payoff(spot)) is pinned at/above intrinsic and
                  //     std_error is carried through (NOT zeroed) when the lower bound wins.
                  std::vector<gpu::BsOption> deep_itm = {gpu::BsOption{100.0, 150.0, 0.05, 0.0, 0.05, 0.1, false}};
                  auto r_clamp = gpu::longstaff_schwartz_american_batch(deep_itm, paths, steps, seed);
                  t.expect(r_clamp.has_value(), "deep-ITM put priced");
                  if (r_clamp) {
                      const double intrinsic = 150.0 - 100.0;  // 50 — immediate-exercise value
                      t.expect((*r_clamp)[0].price >= intrinsic,
                               "deep-ITM American put price >= intrinsic (clamp binds)");
                      t.expect(std::isfinite((*r_clamp)[0].std_error) && (*r_clamp)[0].std_error >= 0.0,
                               "std_error finite and >= 0 even when the intrinsic lower bound wins");
                  }
              })
        .test("bode_sweep / nyquist_sweep mirror control::bode/nyquist",
              [](TestContext& t) {
                  // Build a 2nd-order TransferFunction G(s) = 10 / (s^2 + 2s + 10)
                  const auto num_p = nimblecas::RationalPoly::from_coeffs(
                      {nimblecas::Rational::from_int(10)});
                  const auto den_p = nimblecas::RationalPoly::from_coeffs(
                      {nimblecas::Rational::from_int(10), nimblecas::Rational::from_int(2),
                       nimblecas::Rational::from_int(1)});
                  const auto tf_res = nimblecas::TransferFunction::make(num_p, den_p);
                  t.expect(tf_res.has_value(), "TransferFunction constructed");
                  if (!tf_res.has_value()) {
                      return;
                  }
                  const auto tf = *tf_res;

                  // Log-spaced grid of ~50 frequencies in [0.1, 100.0]
                  const auto omegas = nimblecas::logspace(0.1, 100.0, 50);
                  t.expect(omegas.size() == 50, "logspace produced 50 omegas");

                  // 1. Bode sweep vs CPU control::bode
                  const auto cpu_bode = nimblecas::bode(tf, omegas);
                  const auto gpu_bode_res = gpu::bode_sweep(tf, omegas);
                  t.expect(gpu_bode_res.has_value(), "bode_sweep returned value");
                  if (gpu_bode_res) {
                      const auto& gpu_bode = *gpu_bode_res;
                      t.expect(gpu_bode.size() == cpu_bode.size(), "bode_sweep size matches cpu");
                      bool bode_match = true;
                      for (std::size_t i = 0; i < omegas.size() && i < gpu_bode.size(); ++i) {
                          const double mag_diff = std::abs(gpu_bode[i].magnitude_db - cpu_bode[i].magnitude_db);
                          const double phase_diff = std::abs(gpu_bode[i].phase_deg - cpu_bode[i].phase_deg);
                          const double mag_tol = 1e-9 * (1.0 + std::abs(cpu_bode[i].magnitude_db));
                          const double phase_tol = 1e-9 * (1.0 + std::abs(cpu_bode[i].phase_deg));
                          if (mag_diff > mag_tol || phase_diff > phase_tol) {
                              bode_match = false;
                              break;
                          }
                      }
                      t.expect(bode_match,
                               "GPU bode_sweep magnitude_db and phase_deg match CPU control::bode within ~1e-9 relative/absolute tolerance");
                  }

                  // 2. Nyquist sweep vs CPU control::nyquist
                  const auto cpu_nyquist = nimblecas::nyquist(tf, omegas);
                  const auto gpu_nyquist_res = gpu::nyquist_sweep(tf, omegas);
                  t.expect(gpu_nyquist_res.has_value(), "nyquist_sweep returned value");
                  if (gpu_nyquist_res) {
                      const auto& gpu_nyquist = *gpu_nyquist_res;
                      t.expect(gpu_nyquist.size() == cpu_nyquist.size(), "nyquist_sweep size matches cpu");
                      bool nyquist_match = true;
                      for (std::size_t i = 0; i < omegas.size() && i < gpu_nyquist.size(); ++i) {
                          const double re_diff = std::abs(gpu_nyquist[i].re - cpu_nyquist[i].re);
                          const double im_diff = std::abs(gpu_nyquist[i].im - cpu_nyquist[i].im);
                          const double re_tol = 1e-9 * (1.0 + std::abs(cpu_nyquist[i].re));
                          const double im_tol = 1e-9 * (1.0 + std::abs(cpu_nyquist[i].im));
                          if (re_diff > re_tol || im_diff > im_tol) {
                              nyquist_match = false;
                              break;
                          }
                      }
                      t.expect(nyquist_match,
                               "GPU nyquist_sweep re and im match CPU control::nyquist within ~1e-9 relative/absolute tolerance");
                  }

                  // 2b. Large-magnitude denominator: at very high omega, |den(iomega)| exceeds the
                  //     ~1.3e154 point where a naive den_re^2+den_im^2 overflows to +inf (yielding a
                  //     bogus (0,0)/-inf-dB result). Scaled (Smith) complex division must stay finite
                  //     and track the CPU oracle. A naive kernel fails this; the happy-path grid cannot.
                  const std::vector<double> big_omegas = {1e70, 1e78, 1e90};
                  const auto cpu_big = nimblecas::bode(tf, big_omegas);
                  const auto gpu_big = gpu::bode_sweep(tf, big_omegas);
                  t.expect(gpu_big.has_value(), "bode_sweep evaluated at large omega");
                  if (gpu_big) {
                      bool big_ok = true;
                      for (std::size_t i = 0; i < big_omegas.size() && i < gpu_big->size(); ++i) {
                          const double g = (*gpu_big)[i].magnitude_db;
                          const double c = cpu_big[i].magnitude_db;
                          if (!std::isfinite(g) || std::abs(g - c) > 1e-9 * (1.0 + std::abs(c))) {
                              big_ok = false;
                              break;
                          }
                      }
                      t.expect(big_ok,
                               "GPU bode magnitude stays finite and matches CPU at |den| > 1e154 (scaled complex division)");
                  }

                  // 3. Determinism / bitwise repeat check
                  const auto bode_run2 = gpu::bode_sweep(tf, omegas);
                  const auto nyquist_run2 = gpu::nyquist_sweep(tf, omegas);
                  if (gpu_bode_res && bode_run2) {
                      bool bode_bitwise = true;
                      for (std::size_t i = 0; i < gpu_bode_res->size(); ++i) {
                          if ((*gpu_bode_res)[i].magnitude_db != (*bode_run2)[i].magnitude_db ||
                              (*gpu_bode_res)[i].phase_deg != (*bode_run2)[i].phase_deg) {
                              bode_bitwise = false;
                              break;
                          }
                      }
                      t.expect(bode_bitwise, "bode_sweep is bitwise deterministic across repeated runs");
                  }
                  if (gpu_nyquist_res && nyquist_run2) {
                      bool nyquist_bitwise = true;
                      for (std::size_t i = 0; i < gpu_nyquist_res->size(); ++i) {
                          if ((*gpu_nyquist_res)[i].re != (*nyquist_run2)[i].re ||
                              (*gpu_nyquist_res)[i].im != (*nyquist_run2)[i].im) {
                              nyquist_bitwise = false;
                              break;
                          }
                      }
                      t.expect(nyquist_bitwise, "nyquist_sweep is bitwise deterministic across repeated runs");
                  }

                  // 4. Edge cases: empty omegas grid
                  const std::vector<double> empty_omegas;
                  const auto empty_bode = gpu::bode_sweep(tf, empty_omegas);
                  const auto empty_nyquist = gpu::nyquist_sweep(tf, empty_omegas);
                  t.expect(empty_bode.has_value() && empty_bode->empty(),
                           "empty omegas -> empty Bode result");
                  t.expect(empty_nyquist.has_value() && empty_nyquist->empty(),
                           "empty omegas -> empty Nyquist result");

                  // 5. Device-requiring asserts (when device is present)
                  if (gpu::available()) {
                      t.expect(gpu::device_count() > 0, "device count > 0 when available()");
                  }
              })
        .test("dwt_batch and swt_batch mirror wavelets::dwt and wavelets::swt",
              [](TestContext& t) {
                  namespace wv = nimblecas::wavelets;
                  // Daubechies-4 / db2 filter bank from built-in factory
                  const auto fb = wv::daubechies(2).value();
                  const int batch = 3;
                  const int len = 8;
                  const std::vector<double> data = {
                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                      8.0, 6.0, 4.0, 2.0, 0.0, -2.0, -4.0, -6.0,
                      0.5, -1.0, 1.5, -2.0, 2.5, -3.0, 3.5, -4.0};

                  // 1. DWT batch cross-check against CPU wavelets::dwt
                  auto got_dwt = gpu::dwt_batch(data, batch, len, fb);
                  t.expect(got_dwt.has_value() && got_dwt->size() == data.size(),
                           "dwt_batch produced expected output size");

                  if (got_dwt && got_dwt->size() == data.size()) {
                      bool dwt_ok = true;
                      const std::size_t half = static_cast<std::size_t>(len / 2);
                      for (int b = 0; b < batch && dwt_ok; ++b) {
                          const std::span<const double> sig{
                              data.data() + static_cast<std::size_t>(b * len),
                              static_cast<std::size_t>(len)};
                          const auto cpu = wv::dwt(sig, fb).value();
                          for (std::size_t i = 0; i < half; ++i) {
                              const double g_a = (*got_dwt)[static_cast<std::size_t>(b * len) + i];
                              const double g_d = (*got_dwt)[static_cast<std::size_t>(b * len) + half + i];
                              if (std::abs(g_a - cpu.approx[i]) > 1e-12 * (1.0 + std::abs(cpu.approx[i])) ||
                                  std::abs(g_d - cpu.detail[i]) > 1e-12 * (1.0 + std::abs(cpu.detail[i]))) {
                                  dwt_ok = false;
                              }
                          }
                      }
                      t.expect(dwt_ok, "dwt_batch matches wavelets::dwt per signal block to ~1e-12");
                  }

                  // 2. SWT batch cross-check against CPU wavelets::swt (level 1)
                  auto got_swt = gpu::swt_batch(data, batch, len, fb);
                  const std::size_t expected_swt_len = static_cast<std::size_t>(batch * 2 * len);
                  t.expect(got_swt.has_value() && got_swt->size() == expected_swt_len,
                           "swt_batch produced expected output size (2*len per block)");

                  if (got_swt && got_swt->size() == expected_swt_len) {
                      bool swt_ok = true;
                      const std::size_t slen = static_cast<std::size_t>(len);
                      for (int b = 0; b < batch && swt_ok; ++b) {
                          const std::span<const double> sig{
                              data.data() + static_cast<std::size_t>(b * len),
                              slen};
                          const auto cpu = wv::swt(sig, fb, 1).value();
                          for (std::size_t i = 0; i < slen; ++i) {
                              const double g_a = (*got_swt)[static_cast<std::size_t>(b * 2 * len) + i];
                              const double g_d = (*got_swt)[static_cast<std::size_t>(b * 2 * len) + slen + i];
                              if (std::abs(g_a - cpu.approx[i]) > 1e-12 * (1.0 + std::abs(cpu.approx[i])) ||
                                  std::abs(g_d - cpu.detail[i]) > 1e-12 * (1.0 + std::abs(cpu.detail[i]))) {
                                  swt_ok = false;
                              }
                          }
                      }
                      t.expect(swt_ok, "swt_batch matches wavelets::swt level 1 per signal block to ~1e-12");
                  }

                  // 3. Bitwise determinism run-to-run (identical outputs on repeated calls)
                  auto dwt_again = gpu::dwt_batch(data, batch, len, fb);
                  t.expect(dwt_again.has_value() && *dwt_again == *got_dwt,
                           "dwt_batch is bitwise-deterministic run-to-run");

                  auto swt_again = gpu::swt_batch(data, batch, len, fb);
                  t.expect(swt_again.has_value() && *swt_again == *got_swt,
                           "swt_batch is bitwise-deterministic run-to-run");

                  // 4. Energy preservation check for orthogonal DWT (Parseval's identity)
                  // ||signal||^2 == ||approx||^2 + ||detail||^2 to ~1e-12
                  if (got_dwt) {
                      bool energy_ok = true;
                      for (int b = 0; b < batch; ++b) {
                          double e_sig = 0.0;
                          for (int i = 0; i < len; ++i) {
                              const double v = data[static_cast<std::size_t>(b * len + i)];
                              e_sig += v * v;
                          }
                          double e_coeff = 0.0;
                          for (int i = 0; i < len; ++i) {
                              const double c = (*got_dwt)[static_cast<std::size_t>(b * len + i)];
                              e_coeff += c * c;
                          }
                          if (std::abs(e_sig - e_coeff) > 1e-12 * (1.0 + e_sig)) {
                              energy_ok = false;
                          }
                      }
                      t.expect(energy_ok, "orthonormal DWT preserves signal energy (Parseval identity)");
                  }

                  // 5. Domain guards
                  const std::vector<double> odd_data(6, 1.0);
                  auto bad_len = gpu::dwt_batch(odd_data, 2, 3, fb);
                  t.expect(!bad_len.has_value() && bad_len.error() == MathError::domain_error,
                           "odd block length yields domain_error for DWT");

                  auto mism_size = gpu::dwt_batch(data, batch, 4, fb);
                  t.expect(!mism_size.has_value() && mism_size.error() == MathError::domain_error,
                           "size mismatch yields domain_error for DWT");

                  const wv::FilterBank empty_fb;
                  auto bad_fb = gpu::dwt_batch(data, batch, len, empty_fb);
                  t.expect(!bad_fb.has_value() && bad_fb.error() == MathError::domain_error,
                           "empty FilterBank yields domain_error");
              })
        .test("qmc GPU: discrepancy + sobol/halton batch",
              [](TestContext& t) {
                  namespace qmc = nimblecas;
                  const std::size_t count = 100;
                  const std::size_t dim = 5;
                  const std::uint64_t n0 = 1;

                  // 1. Sobol batch vs CPU sobol_point
                  auto sob_gpu = gpu::sobol_batch(n0, count, dim);
                  t.expect(sob_gpu.has_value() && sob_gpu->size() == count * dim,
                           "sobol_batch evaluated");
                  if (sob_gpu) {
                      bool sob_ok = true;
                      for (std::size_t i = 0; i < count && sob_ok; ++i) {
                          auto pt_cpu = qmc::sobol_point(n0 + i, dim).value();
                          for (std::size_t d = 0; d < dim; ++d) {
                              if ((*sob_gpu)[i * dim + d] != pt_cpu[d]) {
                                  sob_ok = false;
                              }
                          }
                      }
                      t.expect(sob_ok, "sobol_batch is bit-exact to CPU sobol_point (==)");
                  }

                  // 2. Halton batch vs CPU halton_point
                  auto hal_gpu = gpu::halton_batch(n0, count, dim);
                  t.expect(hal_gpu.has_value() && hal_gpu->size() == count * dim,
                           "halton_batch evaluated");
                  if (hal_gpu) {
                      bool hal_ok = true;
                      for (std::size_t i = 0; i < count && hal_ok; ++i) {
                          auto pt_cpu = qmc::halton_point(n0 + i, dim).value();
                          for (std::size_t d = 0; d < dim; ++d) {
                              const double diff = std::abs((*hal_gpu)[i * dim + d] - pt_cpu[d]);
                              if (diff > 1e-12) {
                                  hal_ok = false;
                              }
                          }
                      }
                      t.expect(hal_ok, "halton_batch matches CPU halton_point to 1e-12");
                  }

                  // 3. L2 star discrepancy vs CPU l2_star_discrepancy
                  std::vector<std::vector<double>> pts;
                  pts.reserve(count);
                  for (std::size_t i = 0; i < count; ++i) {
                      pts.push_back(qmc::halton_point(n0 + i, dim).value());
                  }
                  auto disc_cpu = qmc::l2_star_discrepancy(pts, dim).value();
                  auto disc_gpu = gpu::l2_star_discrepancy(pts, dim);
                  t.expect(disc_gpu.has_value(), "l2_star_discrepancy computed");
                  if (disc_gpu) {
                      const double rel_diff = std::abs(*disc_gpu - disc_cpu) / disc_cpu;
                      t.expect(rel_diff <= 1e-10,
                               "GPU L2 star discrepancy matches CPU to 1e-10 relative");
                  }

                  // 4. Bitwise repeat / determinism
                  auto sob_again = gpu::sobol_batch(n0, count, dim);
                  auto hal_again = gpu::halton_batch(n0, count, dim);
                  auto disc_again = gpu::l2_star_discrepancy(pts, dim);
                  t.expect(sob_again.has_value() && *sob_again == *sob_gpu,
                           "sobol_batch is bit-for-bit reproducible");
                  t.expect(hal_again.has_value() && *hal_again == *hal_gpu,
                           "halton_batch is bit-for-bit reproducible");
                  t.expect(disc_again.has_value() && *disc_again == *disc_gpu,
                           "l2_star_discrepancy is bit-for-bit reproducible");

                  // 5. Domain guards
                  auto bad_dim_s = gpu::sobol_batch(n0, count, 0);
                  t.expect(!bad_dim_s.has_value() && bad_dim_s.error() == MathError::domain_error,
                           "sobol_batch dim == 0 -> domain_error");
                  auto bad_dim_s9 = gpu::sobol_batch(n0, count, 9);
                  t.expect(!bad_dim_s9.has_value() && bad_dim_s9.error() == MathError::domain_error,
                           "sobol_batch dim > 8 -> domain_error");
                  auto bad_dim_h = gpu::halton_batch(n0, count, 0);
                  t.expect(!bad_dim_h.has_value() && bad_dim_h.error() == MathError::domain_error,
                           "halton_batch dim == 0 -> domain_error");
                  auto empty_pts = gpu::l2_star_discrepancy(std::vector<std::vector<double>>{}, dim);
                  t.expect(!empty_pts.has_value() && empty_pts.error() == MathError::domain_error,
                           "empty point set -> domain_error");
              })
        .test("bicgstab_csr solves a non-symmetric CSR system",
              [](TestContext& t) {
                  // Non-symmetric upwinded convection-diffusion sparse system on a 1D grid of size n = 32.
                  // Tridiagonal: A[i,i] = 4.0, A[i, i-1] = -1.5 (sub), A[i, i+1] = -0.5 (super).
                  // Strictly diagonally dominant: |4.0| > |-1.5| + |-0.5| = 2.0.
                  const int n = 32;
                  std::vector<int> row_offsets;
                  std::vector<int> col_indices;
                  std::vector<double> values;
                  std::vector<double> b(n);
                  row_offsets.push_back(0);
                  for (int i = 0; i < n; ++i) {
                      b[static_cast<std::size_t>(i)] = std::sin(static_cast<double>(i + 1));
                      if (i > 0) {
                          col_indices.push_back(i - 1);
                          values.push_back(-1.5);
                      }
                      col_indices.push_back(i);
                      values.push_back(4.0);
                      if (i + 1 < n) {
                          col_indices.push_back(i + 1);
                          values.push_back(-0.5);
                      }
                      row_offsets.push_back(static_cast<int>(col_indices.size()));
                  }

                  const double tol = 1e-10;
                  const int max_iters = 1000;

                  // 1. Solve on GPU / fallback
                  auto got = gpu::bicgstab_csr(row_offsets, col_indices, values, b, max_iters, tol);
                  t.expect(got.has_value(), "bicgstab_csr computed");
                  if (got) {
                      t.expect(got->x.size() == static_cast<std::size_t>(n), "solution size matches n");
                      t.expect(got->converged, "bicgstab_csr converged on well-conditioned non-symmetric system");
                      t.expect(got->residual <= tol * 5.0, "returned true residual norm <= tol * ||b|| bound");

                      // Compute host true residual norm ||b - A x||
                      double b_norm2 = 0.0;
                      double res_norm2 = 0.0;
                      for (int i = 0; i < n; ++i) {
                          double bi = b[static_cast<std::size_t>(i)];
                          b_norm2 += bi * bi;
                          double Ax_i = 0.0;
                          const int start = row_offsets[static_cast<std::size_t>(i)];
                          const int end = row_offsets[static_cast<std::size_t>(i + 1)];
                          for (int e = start; e < end; ++e) {
                              Ax_i += values[static_cast<std::size_t>(e)] *
                                      got->x[static_cast<std::size_t>(col_indices[static_cast<std::size_t>(e)])];
                          }
                          const double r_i = bi - Ax_i;
                          res_norm2 += r_i * r_i;
                      }
                      const double b_norm = std::sqrt(b_norm2);
                      const double true_res = std::sqrt(res_norm2);
                      t.expect(true_res <= tol * b_norm + 1e-12, "host-computed true residual <= tol * ||b||");
                      t.expect(std::abs(got->residual - true_res) <= 1e-10 * (1.0 + true_res),
                               "returned residual matches host true residual to 1e-10 relative");

                      // Compare with CPU oracle krylov::bicgstab
                      auto A_cpu = nimblecas::csr_matvec(row_offsets, col_indices, values, n);
                      auto cpu_res = nimblecas::bicgstab(A_cpu, b, tol, max_iters);
                      t.expect(cpu_res.has_value() && cpu_res->converged, "CPU krylov::bicgstab also converged");
                      if (cpu_res && cpu_res->converged) {
                          double max_diff = 0.0;
                          for (int i = 0; i < n; ++i) {
                              max_diff = std::max(max_diff, std::abs(got->x[static_cast<std::size_t>(i)] -
                                                                     cpu_res->x[static_cast<std::size_t>(i)]));
                          }
                          t.expect(max_diff <= 1e-6, "GPU solution agrees with CPU bicgstab to 1e-6 inf-norm");
                      }
                  }

                  // 2. max_iters = 1 non-convergence honesty case
                  auto got_1 = gpu::bicgstab_csr(row_offsets, col_indices, values, b, 1, tol);
                  t.expect(got_1.has_value(), "max_iters=1 case computed without error");
                  if (got_1) {
                      t.expect(!got_1->converged, "max_iters=1 does not falsely claim convergence");
                      t.expect(got_1->iterations == 1, "iterations count is 1");
                  }

                  // 3. Determinism: bitwise repeatability on identical inputs
                  auto got_repeat = gpu::bicgstab_csr(row_offsets, col_indices, values, b, max_iters, tol);
                  t.expect(got_repeat.has_value(), "repeat solve computed");
                  if (got && got_repeat) {
                      t.expect(got->x == got_repeat->x, "bitwise repeatable solution vector");
                      t.expect(got->residual == got_repeat->residual, "bitwise repeatable residual");
                  }

                  // 4. Domain guards
                  std::vector<double> empty_b;
                  auto bad_b = gpu::bicgstab_csr(row_offsets, col_indices, values, empty_b, max_iters, tol);
                  t.expect(!bad_b.has_value() && bad_b.error() == MathError::domain_error,
                           "empty b yields domain_error");

                  std::vector<int> bad_row_offsets = {0, 2};
                  auto bad_row = gpu::bicgstab_csr(bad_row_offsets, col_indices, values, b, max_iters, tol);
                  t.expect(!bad_row.has_value() && bad_row.error() == MathError::domain_error,
                           "bad row_offsets length yields domain_error");

                  std::vector<int> bad_cols = col_indices;
                  if (!bad_cols.empty()) bad_cols.pop_back();
                  auto bad_col = gpu::bicgstab_csr(row_offsets, bad_cols, values, b, max_iters, tol);
                  t.expect(!bad_col.has_value() && bad_col.error() == MathError::domain_error,
                           "mismatched col_indices/values length yields domain_error");

                  // 5. Breakdown honesty on a singular (zero) matrix with b != 0: v = A*p = 0 forces an
                  //    rhatv breakdown in iteration 1. That iteration must be COUNTED (mirroring the CPU
                  //    oracle's ++iter at loop top) and convergence honestly false.
                  {
                      std::vector<int> zero_rows(b.size() + 1, 0);  // all-zero matrix, nnz = 0
                      std::vector<int> no_cols;
                      std::vector<double> no_vals;
                      auto sing = gpu::bicgstab_csr(zero_rows, no_cols, no_vals, b, max_iters, tol);
                      t.expect(sing.has_value(), "singular (zero) matrix solve returns a value");
                      if (sing) {
                          t.expect(!sing->converged, "singular system honestly reports not-converged");
                          t.expect(sing->iterations == 1, "iteration-1 breakdown is counted as 1, not 0");
                      }
                  }

                  // 6. b == 0: converged immediately with x == 0 and zero iterations.
                  {
                      std::vector<double> b_zero(b.size(), 0.0);
                      auto z = gpu::bicgstab_csr(row_offsets, col_indices, values, b_zero, max_iters, tol);
                      t.expect(z.has_value(), "b == 0 solve returns a value");
                      if (z) {
                          t.expect(z->converged && z->iterations == 0, "b == 0 converges at iteration 0");
                          bool all_zero = true;
                          for (double xv : z->x) { if (xv != 0.0) { all_zero = false; break; } }
                          t.expect(all_zero, "b == 0 gives x == 0");
                      }
                  }

                  // 7. Device/fallback parity guards: negative max_iters (hangs the CPU path, no-ops the
                  //    device path if unguarded) and an out-of-range column index must both reject.
                  auto neg_it = gpu::bicgstab_csr(row_offsets, col_indices, values, b, -1, tol);
                  t.expect(!neg_it.has_value() && neg_it.error() == MathError::domain_error,
                           "negative max_iters yields domain_error");
                  {
                      std::vector<int> oob_cols = col_indices;
                      if (!oob_cols.empty()) { oob_cols[0] = static_cast<int>(b.size()); }  // == n, out of range
                      auto oob = gpu::bicgstab_csr(row_offsets, oob_cols, values, b, max_iters, tol);
                      t.expect(!oob.has_value() && oob.error() == MathError::domain_error,
                               "out-of-range column index yields domain_error");
                  }
              })
        .test("batched_cg_csr solves K independent SPD systems",
              [](nimblecas::testing::TestContext& t) {
                  using namespace nimblecas;

                  // 1. Define K = 3 independent SPD CSR systems of different sizes:
                  // System 0: 2x2 SPD matrix [[4, 1], [1, 3]], b = [1, 2]
                  std::vector<int> row0 = {0, 2, 4};
                  std::vector<int> col0 = {0, 1, 0, 1};
                  std::vector<double> val0 = {4.0, 1.0, 1.0, 3.0};
                  std::vector<double> b0 = {1.0, 2.0};

                  // System 1: 3x3 Poisson-1D tridiagonal [[2, -1, 0], [-1, 2, -1], [0, -1, 2]], b = [1, 0, 1]
                  std::vector<int> row1 = {0, 2, 5, 7};
                  std::vector<int> col1 = {0, 1, 0, 1, 2, 1, 2};
                  std::vector<double> val1 = {2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0};
                  std::vector<double> b1 = {1.0, 0.0, 1.0};

                  // System 2: 1x1 diagonal [[5.0]], b = [10.0]
                  std::vector<int> row2 = {0, 1};
                  std::vector<int> col2 = {0};
                  std::vector<double> val2 = {5.0};
                  std::vector<double> b2 = {10.0};

                  std::vector<gpu::CsrSystem> sys_list = {
                      {row0, col0, val0, b0},
                      {row1, col1, val1, b1},
                      {row2, col2, val2, b2}
                  };

                  auto res = gpu::batched_cg_csr(sys_list, 100, 1e-10);
                  t.expect(res.has_value(), "batched_cg_csr succeeds on valid batch");
                  if (res) {
                      t.expect(res->size() == 3, "returns 3 results for 3 systems");

                      // Check convergence and residuals
                      for (std::size_t i = 0; i < res->size(); ++i) {
                          t.expect((*res)[i].converged, "system " + std::to_string(i) + " converged");
                          t.expect((*res)[i].residual <= 1e-8, "system " + std::to_string(i) + " residual small");
                      }

                      // Verify System 2 exact solution: 5 * x = 10 => x = 2
                      t.expect((*res)[2].x.size() == 1 && std::abs((*res)[2].x[0] - 2.0) < 1e-8,
                               "1x1 system solves to x = 2");

                      // Compare each system against solo cg_csr / CPU krylov::cg
                      auto solo0 = gpu::cg_csr(row0, col0, val0, b0, 100, 1e-10);
                      if (solo0) {
                          t.expect(solo0->converged, "solo cg_csr converged for system 0");
                          for (std::size_t k = 0; k < b0.size(); ++k) {
                              t.expect(std::abs((*res)[0].x[k] - solo0->x[k]) < 1e-6,
                                       "batch system 0 agrees with solo cg_csr to tolerance");
                          }
                      }
                  }

                  // 2. Bitwise repeatability run-to-run:
                  auto res_repeat = gpu::batched_cg_csr(sys_list, 100, 1e-10);
                  if (res && res_repeat) {
                      bool bitwise_match = true;
                      for (std::size_t i = 0; i < sys_list.size(); ++i) {
                          if ((*res)[i].x != (*res_repeat)[i].x ||
                              (*res)[i].residual != (*res_repeat)[i].residual ||
                              (*res)[i].iterations != (*res_repeat)[i].iterations ||
                              (*res)[i].converged != (*res_repeat)[i].converged) {
                              bitwise_match = false;
                              break;
                          }
                      }
                      t.expect(bitwise_match, "batched_cg_csr is bitwise repeatable run-to-run");
                  }

                  // 3. Non-SPD isolation: batch containing one non-SPD system mid-batch
                  std::vector<int> row_neg = {0, 1};
                  std::vector<int> col_neg = {0};
                  std::vector<double> val_neg = {-2.0};
                  std::vector<double> b_neg = {1.0};

                  std::vector<gpu::CsrSystem> mixed_list = {
                      {row0, col0, val0, b0},
                      {row_neg, col_neg, val_neg, b_neg},  // non-SPD breakdown
                      {row2, col2, val2, b2}
                  };

                  auto mixed_res = gpu::batched_cg_csr(mixed_list, 100, 1e-10);
                  t.expect(mixed_res.has_value(), "mixed non-SPD batch completes without error");
                  if (mixed_res) {
                      t.expect((*mixed_res)[0].converged, "sys 0 in mixed batch converged");
                      t.expect(!(*mixed_res)[1].converged, "sys 1 (non-SPD) in mixed batch honestly reports not converged");
                      t.expect((*mixed_res)[2].converged, "sys 2 in mixed batch converged");
                  }

                  // 4. max_iters honesty: max_iters = 1 on multi-iter system
                  auto max_it_res = gpu::batched_cg_csr(sys_list, 1, 1e-10);
                  if (max_it_res) {
                      t.expect((*max_it_res)[0].iterations == 1, "max_iters=1 performs 1 iteration");
                  }

                  // 5. Domain error guards:
                  auto empty_res = gpu::batched_cg_csr({}, 100, 1e-10);
                  t.expect(empty_res.has_value() && empty_res->empty(), "empty systems span returns empty success");

                  auto neg_iters_res = gpu::batched_cg_csr(sys_list, -1, 1e-10);
                  t.expect(!neg_iters_res.has_value() && neg_iters_res.error() == MathError::domain_error,
                           "negative max_iters yields domain_error");

                  std::vector<double> empty_b;
                  std::vector<gpu::CsrSystem> bad_b_list = {
                      {row0, col0, val0, empty_b}
                  };
                  auto bad_b_res = gpu::batched_cg_csr(bad_b_list, 100, 1e-10);
                  t.expect(!bad_b_res.has_value() && bad_b_res.error() == MathError::domain_error,
                           "system with empty b yields domain_error");

                  auto neg_tol_res = gpu::batched_cg_csr(sys_list, 100, -1.0);
                  t.expect(!neg_tol_res.has_value() && neg_tol_res.error() == MathError::domain_error,
                           "negative tol yields domain_error");

                  // 6. Larger batch: exercises the one-block-per-system launch across many blocks
                  // (the K=3 batch above would fit a single block; this forces a multi-block grid).
                  std::vector<std::vector<int>> big_row(32), big_col(32);
                  std::vector<std::vector<double>> big_val(32), big_b(32);
                  std::vector<gpu::CsrSystem> big_list;
                  big_list.reserve(32);
                  for (int s = 0; s < 32; ++s) {
                      // 2x2 SPD [[4+s, 1], [1, 3]], b = [1, 2]; distinct per system.
                      big_row[s] = {0, 2, 4};
                      big_col[s] = {0, 1, 0, 1};
                      big_val[s] = {4.0 + s, 1.0, 1.0, 3.0};
                      big_b[s] = {1.0, 2.0};
                      big_list.push_back({big_row[s], big_col[s], big_val[s], big_b[s]});
                  }
                  auto big_res = gpu::batched_cg_csr(big_list, 100, 1e-10);
                  t.expect(big_res.has_value() && big_res->size() == 32,
                           "32-system batch returns 32 results");
                  if (big_res) {
                      bool all_conv = true;
                      for (std::size_t i = 0; i < big_res->size(); ++i) {
                          if (!(*big_res)[i].converged || (*big_res)[i].residual > 1e-8) {
                              all_conv = false;
                          }
                          auto solo = gpu::cg_csr(big_row[i], big_col[i], big_val[i], big_b[i], 100, 1e-10);
                          if (solo) {
                              for (std::size_t k = 0; k < big_b[i].size(); ++k) {
                                  t.expect(std::abs((*big_res)[i].x[k] - solo->x[k]) < 1e-6,
                                           "large-batch system agrees with solo cg_csr");
                              }
                          }
                      }
                      t.expect(all_conv, "all 32 systems converged with small residual");
                  }
              })
        .test("gmres_csr solves a non-symmetric CSR system",
              [](nimblecas::testing::TestContext& t) {
                  using namespace nimblecas;

                  // 1. Non-symmetric 8x8 convection-diffusion CSR system
                  const int n = 8;
                  std::vector<int> row_offsets;
                  std::vector<int> col_indices;
                  std::vector<double> values;
                  row_offsets.push_back(0);
                  for (int i = 0; i < n; ++i) {
                      if (i > 0) {
                          col_indices.push_back(i - 1);
                          values.push_back(-0.8);
                      }
                      col_indices.push_back(i);
                      values.push_back(2.0 + i * 0.1);
                      if (i + 1 < n) {
                          col_indices.push_back(i + 1);
                          values.push_back(-1.2);
                      }
                      row_offsets.push_back(static_cast<int>(values.size()));
                  }
                  std::vector<double> b(n, 1.0);
                  const double tol = 1e-10;
                  const int max_iters = 200;
                  const int restart = 30;

                  auto host_spmv = [&](const std::vector<double>& vx) {
                      std::vector<double> ax(n, 0.0);
                      for (int r = 0; r < n; ++r) {
                          double sum = 0.0;
                          for (int e = row_offsets[r]; e < row_offsets[r + 1]; ++e) {
                              sum += values[e] * vx[col_indices[e]];
                          }
                          ax[r] = sum;
                      }
                      return ax;
                  };
                  auto host_norm = [](const std::vector<double>& v) {
                      double s = 0.0;
                      for (double d : v) s += d * d;
                      return std::sqrt(s);
                  };
                  auto host_true_resid = [&](const std::vector<double>& vx) {
                      auto ax = host_spmv(vx);
                      std::vector<double> r(n, 0.0);
                      for (int i = 0; i < n; ++i) r[i] = b[i] - ax[i];
                      return host_norm(r);
                  };

                  const double bnorm = host_norm(b);

                  // Primary solve
                  auto got = gpu::gmres_csr(row_offsets, col_indices, values, b, max_iters, tol, restart);
                  t.expect(got.has_value(), "gmres_csr computed without error");
                  if (got) {
                      t.expect(got->converged, "system converged");
                      t.expect(got->residual <= tol * bnorm, "reported residual <= tol * ||b||");

                      const double tr = host_true_resid(got->x);
                      t.expect(tr <= tol * bnorm, "true residual <= tol * ||b||");

                      // Compare against CPU oracle
                      auto A = nimblecas::csr_matvec(row_offsets, col_indices, values, n);
                      auto cpu_res = nimblecas::gmres(A, b, tol, static_cast<std::size_t>(max_iters),
                                                      static_cast<std::size_t>(restart));
                      t.expect(cpu_res.has_value(), "CPU oracle computed");
                      if (cpu_res) {
                          t.expect(cpu_res->converged, "CPU oracle converged");
                          bool near = true;
                          for (int i = 0; i < n; ++i) {
                              if (std::abs(got->x[i] - cpu_res->x[i]) > 1e-8) {
                                  near = false;
                                  break;
                              }
                          }
                          t.expect(near, "GPU solution agrees with CPU krylov::gmres within 1e-8");
                      }
                  }

                  // 2. Honesty at exhaustion: max_iters = 1, restart = 1
                  auto got_1 = gpu::gmres_csr(row_offsets, col_indices, values, b, 1, tol, 1);
                  t.expect(got_1.has_value(), "max_iters=1 computed without error");
                  if (got_1) {
                      t.expect(!got_1->converged, "max_iters=1 does not falsely claim convergence");
                      t.expect(got_1->iterations == 1, "iteration count is 1");
                      const double tr_1 = host_true_resid(got_1->x);
                      t.expect(std::abs(got_1->residual - tr_1) <= 1e-12 * std::max(1.0, tr_1),
                               "reported residual matches host true residual");
                  }

                  // 3. Determinism: bitwise repeatability on identical inputs
                  auto got_repeat = gpu::gmres_csr(row_offsets, col_indices, values, b, max_iters, tol, restart);
                  t.expect(got_repeat.has_value(), "repeat solve computed");
                  if (got && got_repeat) {
                      t.expect(got->x == got_repeat->x, "bitwise repeatable solution vector");
                      t.expect(got->residual == got_repeat->residual, "bitwise repeatable residual");
                      t.expect(got->iterations == got_repeat->iterations, "bitwise repeatable iteration count");
                  }

                  // 4. Restart edge: restart = 2 forcing >1 outer restart cycle
                  auto got_r2 = gpu::gmres_csr(row_offsets, col_indices, values, b, max_iters, tol, 2);
                  t.expect(got_r2.has_value(), "restart=2 solve computed");
                  if (got_r2) {
                      t.expect(got_r2->converged, "restart=2 system converged");
                      const double tr_r2 = host_true_resid(got_r2->x);
                      t.expect(tr_r2 <= tol * bnorm, "restart=2 true residual <= tol * ||b||");
                  }

                  // 5. Edge systems: Identity matrix
                  {
                      std::vector<int> id_row = {0, 1, 2, 3, 4};
                      std::vector<int> id_col = {0, 1, 2, 3};
                      std::vector<double> id_val = {1.0, 1.0, 1.0, 1.0};
                      std::vector<double> id_b = {2.0, -3.0, 0.5, 1.5};
                      auto id_res = gpu::gmres_csr(id_row, id_col, id_val, id_b, 100, tol, 10);
                      t.expect(id_res.has_value(), "Identity system solved");
                      if (id_res) {
                          t.expect(id_res->converged && id_res->iterations == 1, "Identity converges in 1 iteration");
                          bool match = true;
                          for (std::size_t i = 0; i < id_b.size(); ++i) {
                              if (std::abs(id_res->x[i] - id_b[i]) > 1e-12) match = false;
                          }
                          t.expect(match, "Identity x == b to 1e-12");
                      }
                  }

                  // b == all zeros
                  {
                      std::vector<double> b_zero(n, 0.0);
                      auto z_res = gpu::gmres_csr(row_offsets, col_indices, values, b_zero, max_iters, tol, restart);
                      t.expect(z_res.has_value(), "b == 0 solve returns value");
                      if (z_res) {
                          t.expect(z_res->converged && z_res->iterations == 0, "b == 0 converges at iteration 0");
                          bool all_zero = true;
                          for (double xv : z_res->x) {
                              if (xv != 0.0) all_zero = false;
                          }
                          t.expect(all_zero, "b == 0 gives x == 0");
                          t.expect(z_res->residual == 0.0, "b == 0 residual is 0");
                      }
                  }

                  // nnz == 0 (all-zero matrix A, b != 0)
                  {
                      std::vector<int> zero_rows(n + 1, 0);
                      std::vector<int> no_cols;
                      std::vector<double> no_vals;
                      auto sing = gpu::gmres_csr(zero_rows, no_cols, no_vals, b, max_iters, tol, restart);
                      t.expect(sing.has_value(), "nnz == 0 matrix solve returns value");
                      if (sing) {
                          t.expect(!sing->converged, "nnz == 0 matrix honestly reports not converged");
                          t.expect(std::abs(sing->residual - bnorm) <= 1e-12 * bnorm, "nnz == 0 residual equals ||b||");
                      }
                  }

                  // 6. Domain guards
                  std::vector<double> empty_b;
                  auto bad_b = gpu::gmres_csr(row_offsets, col_indices, values, empty_b, max_iters, tol, restart);
                  t.expect(!bad_b.has_value() && bad_b.error() == MathError::domain_error,
                           "empty b yields domain_error");

                  std::vector<int> bad_row_offsets = {0, 2};
                  auto bad_row = gpu::gmres_csr(bad_row_offsets, col_indices, values, b, max_iters, tol, restart);
                  t.expect(!bad_row.has_value() && bad_row.error() == MathError::domain_error,
                           "bad row_offsets length yields domain_error");

                  std::vector<int> bad_cols = col_indices;
                  if (!bad_cols.empty()) bad_cols.pop_back();
                  auto bad_col = gpu::gmres_csr(row_offsets, bad_cols, values, b, max_iters, tol, restart);
                  t.expect(!bad_col.has_value() && bad_col.error() == MathError::domain_error,
                           "mismatched col_indices/values length yields domain_error");

                  auto bad_restart = gpu::gmres_csr(row_offsets, col_indices, values, b, max_iters, tol, 0);
                  t.expect(!bad_restart.has_value() && bad_restart.error() == MathError::domain_error,
                           "restart < 1 yields domain_error");

                  auto neg_it = gpu::gmres_csr(row_offsets, col_indices, values, b, -1, tol, restart);
                  t.expect(!neg_it.has_value() && neg_it.error() == MathError::domain_error,
                           "negative max_iters yields domain_error");

                  auto neg_tol = gpu::gmres_csr(row_offsets, col_indices, values, b, max_iters, -1.0, restart);
                  t.expect(!neg_tol.has_value() && neg_tol.error() == MathError::domain_error,
                           "negative tol yields domain_error");

                  std::vector<int> oob_cols = col_indices;
                  if (!oob_cols.empty()) oob_cols[0] = n;
                  auto oob = gpu::gmres_csr(row_offsets, oob_cols, values, b, max_iters, tol, restart);
                  t.expect(!oob.has_value() && oob.error() == MathError::domain_error,
                           "out-of-range col index yields domain_error");
              })
        .test("batched_lm_curvefit (Family I)", [](TestContext& t) {
            auto eval_model_data = [](gpu::FitModel model, double ti, std::span<const double> theta) -> double {
                const std::size_t m = theta.size();
                switch (model) {
                    case gpu::FitModel::polynomial: {
                        double f = 0.0, p = 1.0;
                        for (std::size_t j = 0; j < m; ++j) { f += theta[j] * p; p *= ti; }
                        return f;
                    }
                    case gpu::FitModel::exponential:
                        return theta[0] * std::exp(theta[1] * ti) + theta[2];
                    case gpu::FitModel::gaussian: {
                        double u = (ti - theta[1]) / theta[2];
                        return theta[0] * std::exp(-0.5 * u * u);
                    }
                    case gpu::FitModel::logistic:
                        return theta[0] / (1.0 + std::exp(-theta[1] * (ti - theta[2])));
                    case gpu::FitModel::sinusoid:
                        return theta[0] * std::sin(theta[1] * ti + theta[2]) + theta[3];
                    case gpu::FitModel::power_law:
                        return theta[0] * std::pow(ti, theta[1]);
                }
                return 0.0;
            };

            // 1. Exact-data convergence for all 6 models
            {
                struct ModelTestCase {
                    gpu::FitModel model;
                    std::vector<double> theta_true;
                    std::vector<double> theta0;
                    double t_min;
                    double t_max;
                };

                std::vector<ModelTestCase> cases = {
                    {gpu::FitModel::polynomial, {1.5, -2.0, 0.5, 0.1}, {1.2, -1.6, 0.4, 0.08}, -1.0, 1.0},
                    {gpu::FitModel::exponential, {2.0, -0.5, 1.0}, {1.8, -0.4, 0.8}, 0.0, 2.0},
                    {gpu::FitModel::gaussian, {5.0, 1.0, 0.8}, {4.2, 0.85, 0.95}, -1.0, 3.0},
                    {gpu::FitModel::logistic, {10.0, 1.5, 2.0}, {8.5, 1.2, 1.7}, 0.0, 4.0},
                    {gpu::FitModel::sinusoid, {3.0, 2.0, 0.5, 1.0}, {2.6, 1.7, 0.4, 0.8}, 0.0, 3.0},
                    {gpu::FitModel::power_law, {2.5, 1.8}, {2.1, 1.5}, 0.5, 3.5}
                };

                std::vector<gpu::CurveFitProblem> problems;
                std::vector<std::vector<double>> t_buffers;
                std::vector<std::vector<double>> y_buffers;
                const std::size_t n_pts = 64;

                for (const auto& c : cases) {
                    std::vector<double> t_vec(n_pts);
                    std::vector<double> y_vec(n_pts);
                    double step = (c.t_max - c.t_min) / static_cast<double>(n_pts - 1);
                    for (std::size_t i = 0; i < n_pts; ++i) {
                        t_vec[i] = c.t_min + static_cast<double>(i) * step;
                        y_vec[i] = eval_model_data(c.model, t_vec[i], c.theta_true);
                    }
                    t_buffers.push_back(std::move(t_vec));
                    y_buffers.push_back(std::move(y_vec));
                }

                for (std::size_t k = 0; k < cases.size(); ++k) {
                    problems.push_back(gpu::CurveFitProblem{
                        .model = cases[k].model,
                        .t = t_buffers[k],
                        .y = y_buffers[k],
                        .theta0 = cases[k].theta0
                    });
                }

                auto res = gpu::batched_curve_fit_lm(problems);
                t.expect(res.has_value(), "exact-data batched_curve_fit_lm returns value");
                if (res) {
                    t.expect(res->size() == cases.size(), "exact-data result size matches batch");
                    for (std::size_t k = 0; k < cases.size(); ++k) {
                        const auto& r = (*res)[k];
                        const auto& c = cases[k];
                        t.expect(r.converged, "exact-data problem converged");
                        t.expect(r.residual_norm <= 1e-8, "exact-data residual_norm <= 1e-8");
                        t.expect(r.theta.size() == c.theta_true.size(), "exact-data theta size matches");
                        for (std::size_t j = 0; j < c.theta_true.size(); ++j) {
                            double rel_err = std::abs(r.theta[j] - c.theta_true[j]) / std::max(1.0, std::abs(c.theta_true[j]));
                            t.expect(rel_err <= 1e-6, "exact-data theta_j agrees with true to 1e-6 rel");
                        }
                    }
                }
            }

            // 2. Noisy-data agreement with the CPU oracle
            {
                const std::size_t K = 16;
                const std::size_t n_pts = 128;
                std::vector<std::vector<double>> t_bufs(K);
                std::vector<std::vector<double>> y_bufs(K);
                std::vector<std::vector<double>> th0_bufs(K);
                std::vector<gpu::CurveFitProblem> problems(K);

                std::vector<double> theta_true = {4.0, 1.2, 0.7};
                for (std::size_t k = 0; k < K; ++k) {
                    t_bufs[k].resize(n_pts);
                    y_bufs[k].resize(n_pts);
                    double t_min = -1.0 + 0.1 * static_cast<double>(k);
                    double t_max = 3.0 + 0.1 * static_cast<double>(k);
                    double step = (t_max - t_min) / static_cast<double>(n_pts - 1);
                    for (std::size_t i = 0; i < n_pts; ++i) {
                        double tv = t_min + static_cast<double>(i) * step;
                        t_bufs[k][i] = tv;
                        double noise = 1e-3 * std::sin(static_cast<double>(k * 100 + i));
                        y_bufs[k][i] = eval_model_data(gpu::FitModel::gaussian, tv, theta_true) + noise;
                    }
                    th0_bufs[k] = {3.5 + 0.05 * static_cast<double>(k), 1.0 + 0.02 * static_cast<double>(k), 0.8};
                    problems[k] = gpu::CurveFitProblem{
                        .model = gpu::FitModel::gaussian,
                        .t = t_bufs[k],
                        .y = y_bufs[k],
                        .theta0 = th0_bufs[k]
                    };
                }

                auto gpu_res = gpu::batched_curve_fit_lm(problems, gpu::LmFitOptions{}.with_tol(1e-8));
                t.expect(gpu_res.has_value(), "noisy-data batch fit returns value");
                if (gpu_res) {
                    for (std::size_t k = 0; k < K; ++k) {
                        const auto& r_gpu = (*gpu_res)[k];
                        t.expect(r_gpu.converged, "noisy-data problem converged");

                        const auto& p = problems[k];
                        nlsolve::ResidualFn F = [&p](std::span<const double> th) -> std::vector<double> {
                            std::vector<double> r(p.t.size());
                            for (std::size_t i = 0; i < p.t.size(); ++i) {
                                double u = (p.t[i] - th[1]) / th[2];
                                r[i] = th[0] * std::exp(-0.5 * u * u) - p.y[i];
                            }
                            return r;
                        };
                        nlsolve::JacobianFn J = [&p](std::span<const double> th) -> std::vector<double> {
                            std::size_t n = p.t.size();
                            std::vector<double> j_flat(n * 3);
                            for (std::size_t i = 0; i < n; ++i) {
                                double u = (p.t[i] - th[1]) / th[2];
                                double e = std::exp(-0.5 * u * u);
                                j_flat[i * 3 + 0] = e;
                                j_flat[i * 3 + 1] = th[0] * e * u / th[2];
                                j_flat[i * 3 + 2] = th[0] * e * u * u / th[2];
                            }
                            return j_flat;
                        };
                        nlsolve::Options o{};
                        o.tol = 1e-8;
                        o.max_iter = 100;
                        auto cpu_sol = nlsolve::levenberg_marquardt(F, J, p.theta0, o, 1e-3);
                        t.expect(cpu_sol.has_value(), "noisy-data CPU oracle solve succeeded");
                        if (cpu_sol) {
                            t.expect(cpu_sol->converged, "noisy-data CPU oracle converged");
                            double cost_diff = std::abs(r_gpu.residual_norm - cpu_sol->residual_norm);
                            t.expect(cost_diff <= 1e-8 * std::max(1.0, cpu_sol->residual_norm),
                                     "cost agrees with CPU oracle to 1e-8 rel");
                            for (std::size_t j = 0; j < 3; ++j) {
                                double th_diff = std::abs(r_gpu.theta[j] - cpu_sol->x[j]);
                                t.expect(th_diff <= 1e-6 * std::max(1.0, std::abs(cpu_sol->x[j])),
                                         "theta agrees with CPU oracle to 1e-6 rel");
                            }
                        }
                    }
                }
            }

            // 3. Linear-model exact oracle (polynomial m=4, n=32)
            {
                std::vector<double> th_true = {2.0, -1.5, 0.8, -0.2};
                std::vector<double> th0 = {0.0, 0.0, 0.0, 0.0};
                const std::size_t n_pts = 32;
                std::vector<double> t_vec(n_pts);
                std::vector<double> y_vec(n_pts);
                for (std::size_t i = 0; i < n_pts; ++i) {
                    t_vec[i] = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(n_pts - 1);
                    y_vec[i] = eval_model_data(gpu::FitModel::polynomial, t_vec[i], th_true);
                }

                gpu::CurveFitProblem poly_prob{
                    .model = gpu::FitModel::polynomial,
                    .t = t_vec,
                    .y = y_vec,
                    .theta0 = th0
                };
                auto poly_res = gpu::batched_curve_fit_lm(std::span<const gpu::CurveFitProblem>{&poly_prob, 1});
                t.expect(poly_res.has_value(), "linear-model polynomial fit returns value");
                if (poly_res) {
                    t.expect((*poly_res)[0].converged, "polynomial fit converged");
                    t.expect((*poly_res)[0].iterations <= 4, "linear model converges in <= 4 iterations");
                    for (std::size_t j = 0; j < 4; ++j) {
                        t.expect(std::abs((*poly_res)[0].theta[j] - th_true[j]) <= 1e-9,
                                 "polynomial coefficient matches exact solution to 1e-9 abs");
                    }
                }
            }

            // 4. Determinism (bitwise, run-to-run) & 5. Batch-composition independence + isolation
            {
                std::vector<double> t_pts = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5};
                std::vector<double> y_pts = {2.0, 2.5, 3.2, 4.1, 5.3, 6.8, 8.7, 11.0};
                std::vector<double> th0 = {1.5, 0.3, 0.5};

                std::vector<double> t_flat(8, 1.0);
                std::vector<double> y_flat(8, 2.0);

                std::vector<gpu::CurveFitProblem> batch_probs;
                for (std::size_t k = 0; k < 8; ++k) {
                    if (k == 3) {
                        batch_probs.push_back(gpu::CurveFitProblem{
                            .model = gpu::FitModel::exponential,
                            .t = t_flat,
                            .y = y_flat,
                            .theta0 = th0
                        });
                    } else {
                        batch_probs.push_back(gpu::CurveFitProblem{
                            .model = gpu::FitModel::exponential,
                            .t = t_pts,
                            .y = y_pts,
                            .theta0 = th0
                        });
                    }
                }

                auto run1 = gpu::batched_curve_fit_lm(batch_probs);
                auto run2 = gpu::batched_curve_fit_lm(batch_probs);
                t.expect(run1.has_value() && run2.has_value(), "batch runs return value");
                if (run1 && run2) {
                    if (gpu::available()) {
                        t.expect((*run1)[0].theta == (*run2)[0].theta, "run-to-run bitwise theta equality");
                        t.expect((*run1)[0].residual_norm == (*run2)[0].residual_norm, "run-to-run bitwise resid equality");
                        t.expect((*run1)[0].iterations == (*run2)[0].iterations, "run-to-run bitwise iterations equality");
                    }
                    // Problem 3's data is rank-deficient (every t == 1.0, every y == 2.0): the
                    // exponential a*exp(b*1)+c == 2 has a continuum of exact solutions, so LM
                    // legitimately drives the residual to ~0 and CONVERGES via Marquardt damping.
                    // converged==true is honest numerics here; the batch-isolation intent is carried
                    // by the finite-residual check, the healthy-neighbour checks, and the batch-of-one
                    // bitwise-identity checks below (a degenerate problem must not poison the batch).
                    t.expect(std::isfinite((*run1)[3].residual_norm), "rank-deficient problem 3 has finite residual_norm");
                    for (std::size_t k = 0; k < 8; ++k) {
                        if (k != 3) {
                            t.expect((*run1)[k].converged, "healthy problem in batch converged");
                        }
                        auto single_run = gpu::batched_curve_fit_lm(std::span<const gpu::CurveFitProblem>{&batch_probs[k], 1});
                        t.expect(single_run.has_value(), "batch-of-one run returns value");
                        if (single_run && gpu::available()) {
                            t.expect((*single_run)[0].theta == (*run1)[k].theta, "batch-of-one theta bitwise identical");
                            t.expect((*single_run)[0].residual_norm == (*run1)[k].residual_norm, "batch-of-one resid bitwise identical");
                            t.expect((*single_run)[0].iterations == (*run1)[k].iterations, "batch-of-one iterations bitwise identical");
                            t.expect((*single_run)[0].converged == (*run1)[k].converged, "batch-of-one converged bitwise identical");
                        }
                    }
                }
            }

            // 5b. High-occupancy determinism: a large batch of tiny problems forces
            // many concurrent CUDA blocks (high warp contention) -- the regime that
            // exposes divergent-__syncthreads races on reused shared control slots that
            // small-K batches cannot surface. Kept modest and fast for CI; the
            // compute-sanitizer racecheck verification temporarily raises K to 4096.
            {
                const std::size_t K = 512;
                std::vector<double> t_pts = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5};
                std::vector<double> y_pts = {2.0, 2.5, 3.2, 4.1, 5.3, 6.8, 8.7, 11.0};
                std::vector<double> th0 = {1.5, 0.3, 0.5};
                std::vector<gpu::CurveFitProblem> probs;
                probs.reserve(K);
                for (std::size_t k = 0; k < K; ++k) {
                    probs.push_back(gpu::CurveFitProblem{
                        .model = gpu::FitModel::exponential,
                        .t = t_pts,
                        .y = y_pts,
                        .theta0 = th0
                    });
                }
                auto r1 = gpu::batched_curve_fit_lm(probs, gpu::LmFitOptions{}.with_tol(1e-8));
                auto r2 = gpu::batched_curve_fit_lm(probs, gpu::LmFitOptions{}.with_tol(1e-8));
                t.expect(r1.has_value() && r2.has_value(), "high-occupancy batch returns value");
                if (r1 && r2 && gpu::available()) {
                    bool run_to_run = true;
                    for (std::size_t k = 0; k < K; ++k) {
                        if ((*r1)[k].theta != (*r2)[k].theta ||
                            (*r1)[k].residual_norm != (*r2)[k].residual_norm ||
                            (*r1)[k].iterations != (*r2)[k].iterations ||
                            (*r1)[k].converged != (*r2)[k].converged) {
                            run_to_run = false;
                            break;
                        }
                    }
                    t.expect(run_to_run, "high-occupancy K-batch is bitwise deterministic run-to-run");
                    // Every problem is identical, so every result must be identical: a
                    // shared-slot race would let one block's decision bleed into another.
                    bool uniform = true;
                    for (std::size_t k = 1; k < K; ++k) {
                        if ((*r1)[k].theta != (*r1)[0].theta ||
                            (*r1)[k].residual_norm != (*r1)[0].residual_norm ||
                            (*r1)[k].iterations != (*r1)[0].iterations ||
                            (*r1)[k].converged != (*r1)[0].converged) {
                            uniform = false;
                            break;
                        }
                    }
                    t.expect(uniform, "identical problems in a high-occupancy batch agree bitwise");
                }
            }

            // 6. Non-convergence honesty (max_iter = 2)
            {
                std::vector<double> t_pts = {0.0, 1.0, 2.0, 3.0, 4.0};
                std::vector<double> y_pts = {1.0, 3.0, 2.0, 5.0, 4.0};
                std::vector<double> th0 = {10.0, -5.0, 10.0};
                gpu::CurveFitProblem hard_prob{
                    .model = gpu::FitModel::exponential,
                    .t = t_pts,
                    .y = y_pts,
                    .theta0 = th0
                };
                auto opts = gpu::LmFitOptions{}.with_max_iter(2);
                auto res = gpu::batched_curve_fit_lm(std::span<const gpu::CurveFitProblem>{&hard_prob, 1}, opts);
                t.expect(res.has_value(), "max_iter=2 fit returns value");
                if (res) {
                    const auto& r = (*res)[0];
                    t.expect(!r.converged, "max_iter=2 reports converged=false");
                    t.expect(r.iterations == 2, "max_iter=2 reports iterations == 2");
                    double sum_sq = 0.0;
                    for (std::size_t i = 0; i < t_pts.size(); ++i) {
                        double fi = r.theta[0] * std::exp(r.theta[1] * t_pts[i]) + r.theta[2];
                        double ri = fi - y_pts[i];
                        sum_sq += ri * ri;
                    }
                    double host_norm = std::sqrt(sum_sq);
                    double rel_diff = std::abs(r.residual_norm - host_norm) / std::max(1.0, host_norm);
                    t.expect(rel_diff <= 1e-12, "residual_norm matches host recompute within 1e-12 rel");
                }
            }

            // 7. Domain guards
            {
                std::vector<double> t_good = {1.0, 2.0, 3.0, 4.0};
                std::vector<double> y_good = {1.0, 2.0, 3.0, 4.0};
                std::vector<double> th3 = {1.0, 1.0, 1.0};

                std::vector<double> t_short = {1.0, 2.0};
                gpu::CurveFitProblem p_mismatch{.model = gpu::FitModel::exponential, .t = t_short, .y = y_good, .theta0 = th3};
                auto res_mismatch = gpu::batched_curve_fit_lm(std::span<const gpu::CurveFitProblem>{&p_mismatch, 1});
                t.expect(!res_mismatch.has_value() && res_mismatch.error() == MathError::domain_error,
                         "mismatched t/y lengths yield domain_error");

                std::vector<double> empty_vec;
                gpu::CurveFitProblem p_empty{.model = gpu::FitModel::exponential, .t = empty_vec, .y = empty_vec, .theta0 = th3};
                auto res_empty = gpu::batched_curve_fit_lm(std::span<const gpu::CurveFitProblem>{&p_empty, 1});
                t.expect(!res_empty.has_value() && res_empty.error() == MathError::domain_error,
                         "empty t yields domain_error");

                std::vector<double> t_2 = {1.0, 2.0};
                std::vector<double> y_2 = {1.0, 2.0};
                gpu::CurveFitProblem p_under{.model = gpu::FitModel::exponential, .t = t_2, .y = y_2, .theta0 = th3};
                auto res_under = gpu::batched_curve_fit_lm(std::span<const gpu::CurveFitProblem>{&p_under, 1});
                t.expect(!res_under.has_value() && res_under.error() == MathError::domain_error,
                         "under-determined n < m yields domain_error");

                std::vector<double> th4 = {1.0, 1.0, 1.0, 1.0};
                gpu::CurveFitProblem p_arity{.model = gpu::FitModel::gaussian, .t = t_good, .y = y_good, .theta0 = th4};
                auto res_arity = gpu::batched_curve_fit_lm(std::span<const gpu::CurveFitProblem>{&p_arity, 1});
                t.expect(!res_arity.has_value() && res_arity.error() == MathError::domain_error,
                         "wrong model arity yields domain_error");

                std::vector<double> th9(9, 1.0);
                std::vector<double> t9(10, 1.0);
                std::vector<double> y9(10, 1.0);
                gpu::CurveFitProblem p_m9{.model = gpu::FitModel::polynomial, .t = t9, .y = y9, .theta0 = th9};
                auto res_m9 = gpu::batched_curve_fit_lm(std::span<const gpu::CurveFitProblem>{&p_m9, 1});
                t.expect(!res_m9.has_value() && res_m9.error() == MathError::domain_error,
                         "polynomial m=9 yields domain_error");

                std::vector<double> y_nan = {1.0, std::numeric_limits<double>::quiet_NaN(), 3.0, 4.0};
                gpu::CurveFitProblem p_nan{.model = gpu::FitModel::exponential, .t = t_good, .y = y_nan, .theta0 = th3};
                auto res_nan = gpu::batched_curve_fit_lm(std::span<const gpu::CurveFitProblem>{&p_nan, 1});
                t.expect(!res_nan.has_value() && res_nan.error() == MathError::domain_error,
                         "NaN in y yields domain_error");

                std::vector<double> t_zero = {0.0, 1.0, 2.0, 3.0};
                std::vector<double> th2 = {1.0, 1.0};
                gpu::CurveFitProblem p_pow0{.model = gpu::FitModel::power_law, .t = t_zero, .y = y_good, .theta0 = th2};
                auto res_pow0 = gpu::batched_curve_fit_lm(std::span<const gpu::CurveFitProblem>{&p_pow0, 1});
                t.expect(!res_pow0.has_value() && res_pow0.error() == MathError::domain_error,
                         "power_law with t containing 0 yields domain_error");

                gpu::CurveFitProblem p_ok{.model = gpu::FitModel::exponential, .t = t_good, .y = y_good, .theta0 = th3};
                auto res_neg_iter = gpu::batched_curve_fit_lm(std::span<const gpu::CurveFitProblem>{&p_ok, 1}, gpu::LmFitOptions{}.with_max_iter(-1));
                t.expect(!res_neg_iter.has_value() && res_neg_iter.error() == MathError::domain_error,
                         "negative max_iter yields domain_error");

                std::vector<double> t_huge = {1e3, 1e3 + 1, 1e3 + 2, 1e3 + 3};
                std::vector<double> th_overflow = {1.0, 1e6, 0.0};
                gpu::CurveFitProblem p_oflow{.model = gpu::FitModel::exponential, .t = t_huge, .y = y_good, .theta0 = th_overflow};
                auto res_oflow = gpu::batched_curve_fit_lm(std::span<const gpu::CurveFitProblem>{&p_oflow, 1});
                t.expect(!res_oflow.has_value() && res_oflow.error() == MathError::domain_error,
                         "non-finite start residual yields domain_error");
            }

            // 8. Overflow guards
            {
                std::vector<double> t_buf(100, 1.0);
                std::vector<double> y_buf(100, 1.0);
                std::vector<double> th0 = {1.0, 1.0, 1.0};
                constexpr std::size_t int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
                struct FakeSpan {
                    const double* data_ptr;
                    std::size_t sz;
                };
                FakeSpan fake_t{t_buf.data(), int_max + 5};
                std::span<const double> huge_span(fake_t.data_ptr, fake_t.sz);
                gpu::CurveFitProblem p_huge{.model = gpu::FitModel::exponential, .t = huge_span, .y = huge_span, .theta0 = th0};
                auto res_huge = gpu::batched_curve_fit_lm(std::span<const gpu::CurveFitProblem>{&p_huge, 1});
                t.expect(!res_huge.has_value() && res_huge.error() == MathError::overflow,
                         "span size exceeding INT_MAX yields overflow error");
            }

            // 9. FD-vs-analytic cross-check
            {
                const std::size_t K = 8;
                const std::size_t n_pts = 64;
                std::vector<std::vector<double>> t_bufs(K);
                std::vector<std::vector<double>> y_bufs(K);
                std::vector<std::vector<double>> th0_bufs(K);
                std::vector<gpu::CurveFitProblem> problems(K);

                for (std::size_t k = 0; k < K; ++k) {
                    t_bufs[k].resize(n_pts);
                    y_bufs[k].resize(n_pts);
                    std::vector<double> th_true = {2.5 + 0.1 * static_cast<double>(k), 1.5, 0.4, 0.8};
                    for (std::size_t i = 0; i < n_pts; ++i) {
                        double tv = 3.0 * static_cast<double>(i) / static_cast<double>(n_pts - 1);
                        t_bufs[k][i] = tv;
                        y_bufs[k][i] = eval_model_data(gpu::FitModel::sinusoid, tv, th_true);
                    }
                    th0_bufs[k] = {2.2 + 0.1 * static_cast<double>(k), 1.3, 0.3, 0.7};
                    problems[k] = gpu::CurveFitProblem{
                        .model = gpu::FitModel::sinusoid,
                        .t = t_bufs[k],
                        .y = y_bufs[k],
                        .theta0 = th0_bufs[k]
                    };
                }

                auto res_ana = gpu::batched_curve_fit_lm(problems, gpu::LmFitOptions{}.with_analytic_jacobian(true));
                auto res_fd = gpu::batched_curve_fit_lm(problems, gpu::LmFitOptions{}.with_analytic_jacobian(false).with_fd_step(1e-7));
                t.expect(res_ana.has_value() && res_fd.has_value(), "analytic and FD fits return value");
                if (res_ana && res_fd) {
                    for (std::size_t k = 0; k < K; ++k) {
                        t.expect((*res_ana)[k].converged, "analytic fit converged");
                        t.expect((*res_fd)[k].converged, "FD fit converged");
                        double cost_diff = std::abs((*res_ana)[k].residual_norm - (*res_fd)[k].residual_norm);
                        double ref_cost = std::max(1.0, (*res_ana)[k].residual_norm);
                        t.expect(cost_diff <= 1e-6 * ref_cost, "FD and analytic final costs agree to 1e-6 relative");
                    }
                }
            }

            // 10. Fallback-oracle equality (CPU-only path test when !available())
            if (!gpu::available()) {
                std::vector<double> t_pts = {0.0, 0.5, 1.0, 1.5, 2.0};
                std::vector<double> th_true = {3.0, 1.5, 0.5};
                std::vector<double> y_pts(5);
                for (std::size_t i = 0; i < 5; ++i) {
                    y_pts[i] = th_true[0] / (1.0 + std::exp(-th_true[1] * (t_pts[i] - th_true[2])));
                }
                std::vector<double> th0 = {2.5, 1.2, 0.4};
                gpu::CurveFitProblem p{.model = gpu::FitModel::logistic, .t = t_pts, .y = y_pts, .theta0 = th0};

                auto fallback_res = gpu::batched_curve_fit_lm(std::span<const gpu::CurveFitProblem>{&p, 1});
                t.expect(fallback_res.has_value(), "CPU fallback returns value");
                if (fallback_res) {
                    nlsolve::ResidualFn F = [&p](std::span<const double> th) -> std::vector<double> {
                        std::vector<double> r(p.t.size());
                        for (std::size_t i = 0; i < p.t.size(); ++i) {
                            r[i] = th[0] / (1.0 + std::exp(-th[1] * (p.t[i] - th[2]))) - p.y[i];
                        }
                        return r;
                    };
                    nlsolve::JacobianFn J = [&p](std::span<const double> th) -> std::vector<double> {
                        std::size_t n = p.t.size();
                        std::vector<double> j_flat(n * 3);
                        for (std::size_t i = 0; i < n; ++i) {
                            double s = 1.0 / (1.0 + std::exp(-th[1] * (p.t[i] - th[2])));
                            j_flat[i * 3 + 0] = s;
                            j_flat[i * 3 + 1] = th[0] * s * (1.0 - s) * (p.t[i] - th[2]);
                            j_flat[i * 3 + 2] = -th[0] * s * (1.0 - s) * th[1];
                        }
                        return j_flat;
                    };
                    nlsolve::Options o{};
                    auto oracle_res = nlsolve::levenberg_marquardt(F, J, p.theta0, o, 1e-3);
                    t.expect(oracle_res.has_value(), "CPU oracle returns value");
                    if (oracle_res) {
                        t.expect((*fallback_res)[0].theta == oracle_res->x, "CPU fallback theta equals oracle bit-for-bit");
                        t.expect((*fallback_res)[0].residual_norm == oracle_res->residual_norm, "CPU fallback residual equals oracle bit-for-bit");
                    }
                }
            }
        })
        .run();
}

