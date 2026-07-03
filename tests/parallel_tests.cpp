// Tests for nimblecas.parallel: deterministic fork-join combinators.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.parallel;
import nimblecas.testing;

namespace np = nimblecas::parallel;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

auto main() -> int {
    return TestSuite("nimblecas.parallel")
        .test("reports_backend_and_concurrency",
              [](TestContext& t) {
                  std::println("  parallel backend: {}, worker threads: {}", np::backend(),
                               np::max_concurrency());
                  t.expect(np::max_concurrency() >= 1, "at least one worker");
                  t.expect(np::backend() == "tbb" || np::backend() == "ppl" ||
                               np::backend() == "serial",
                           "backend is one of tbb/ppl/serial");
              })
        .test("transform_index_is_order_preserving_and_correct",
              [](TestContext& t) {
                  // n well above the grain cutoff, so the parallel path is exercised
                  const std::size_t n = 100'000;
                  auto got = np::transform_index(n, [](std::size_t i) { return i * i; });
                  bool ok = got.size() == n;
                  for (std::size_t i = 0; ok && i < n; ++i) {
                      ok = got[i] == i * i;  // result[i] must correspond to index i
                  }
                  t.expect(ok, "transform_index[i] == i*i for all i");
              })
        .test("parallel_matches_serial_reference",
              [](TestContext& t) {
                  const std::size_t n = 50'000;
                  auto f = [](std::size_t i) { return (i * 2654435761u) ^ (i << 3); };
                  auto par = np::transform_index(n, f);  // parallel (n >= grain)
                  std::vector<std::size_t> ser(n);
                  for (std::size_t i = 0; i < n; ++i) ser[i] = f(i);
                  t.expect(par.size() == ser.size() && std::ranges::equal(par, ser),
                           "parallel result identical to serial reference");
              })
        .test("small_input_stays_serial_same_result",
              [](TestContext& t) {
                  // below grain -> serial path; must give the same answer
                  auto got = np::transform_index(std::size_t{5}, [](std::size_t i) { return i + 1; });
                  t.expect(got == std::vector<std::size_t>{1, 2, 3, 4, 5}, "serial path correct");
              })
        .test("transform_over_span",
              [](TestContext& t) {
                  std::vector<int> in(2000);
                  std::ranges::iota(in, 0);
                  auto out = np::transform(std::span<const int>(in),
                                           [](int x) { return x + 100; });
                  bool ok = out.size() == in.size();
                  for (std::size_t i = 0; ok && i < in.size(); ++i) ok = out[i] == in[i] + 100;
                  t.expect(ok, "transform maps each element in order");
              })
        .test("invoke2_runs_both_tasks",
              [](TestContext& t) {
                  std::atomic<int> a{0};
                  std::atomic<int> b{0};
                  np::invoke2([&] { a.store(1); }, [&] { b.store(1); });
                  t.expect(a.load() == 1 && b.load() == 1, "both tasks executed");
              })
        .run();
}
