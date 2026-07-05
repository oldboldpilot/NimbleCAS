// Tests for nimblecas.gpu: batch polynomial evaluation on the GPU vs a CPU reference.
// @author Olumuyiwa Oluwasanmi
//
// Built and run only with -DNIMBLECAS_CUDA=ON on a machine with a CUDA device.

import std;
import nimblecas.core;
import nimblecas.gpu;
import nimblecas.testing;

namespace gpu = nimblecas::gpu;
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
        .run();
}
