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
        .run();
}
