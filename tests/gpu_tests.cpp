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
        .run();
}
