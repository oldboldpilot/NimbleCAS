// Tests for nimblecas.simd: dynamic-dispatch elementwise kernels.
// @author Olumuyiwa Oluwasanmi
//
// The dispatched path (AVX-512 / AVX2 / scalar, chosen at runtime) must produce
// bit-identical results to an independent scalar reference — verifying both
// correctness and cross-ISA reproducibility (Rule 55), including the ragged tail.

import std;
import nimblecas.simd;
import nimblecas.testing;

namespace simd = nimblecas::simd;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// deterministic pseudo-random floats (no <random> global state needed)
auto fill(std::vector<float>& v, std::uint32_t seed) -> void {
    std::uint32_t s = seed;
    for (float& x : v) {
        s = s * 1664525u + 1013904223u;  // LCG
        x = static_cast<float>(static_cast<std::int32_t>(s)) / 65536.0f;
    }
}

auto all_bit_equal(std::span<const float> a, std::span<const float> b) -> bool {
    return a.size() == b.size() &&
           std::ranges::equal(a, b, [](float x, float y) {
               return std::bit_cast<std::uint32_t>(x) == std::bit_cast<std::uint32_t>(y);
           });
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.simd")
        .test("reports_an_isa",
              [](TestContext& t) {
                  std::println("  active SIMD ISA: {}", simd::to_string_view(simd::active_isa()));
                  t.expect(true, "active_isa is queryable");
              })
        .test("add_mul_axpy_match_scalar_reference_over_ragged_sizes",
              [](TestContext& t) {
                  // sizes crossing the 8- and 16-lane widths, incl. tails and empties
                  for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{7},
                                        std::size_t{8}, std::size_t{16}, std::size_t{17},
                                        std::size_t{31}, std::size_t{1000}}) {
                      std::vector<float> a(n);
                      std::vector<float> b(n);
                      fill(a, 12345u + static_cast<std::uint32_t>(n));
                      fill(b, 67890u + static_cast<std::uint32_t>(n));

                      std::vector<float> got(n);
                      std::vector<float> ref(n);

                      simd::add(a, b, got);
                      for (std::size_t i = 0; i < n; ++i) ref[i] = a[i] + b[i];
                      t.expect(all_bit_equal(got, ref), std::format("add n={}", n));

                      simd::mul(a, b, got);
                      for (std::size_t i = 0; i < n; ++i) ref[i] = a[i] * b[i];
                      t.expect(all_bit_equal(got, ref), std::format("mul n={}", n));

                      const float s = 2.5f;
                      simd::axpy(s, a, b, got);
                      for (std::size_t i = 0; i < n; ++i) ref[i] = std::fma(a[i], s, b[i]);
                      t.expect(all_bit_equal(got, ref), std::format("axpy n={}", n));
                  }
              })
        .run();
}
