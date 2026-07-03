// Tests for nimblecas.cache: concurrent hash-consing memo.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.cache;
import nimblecas.testing;

using nimblecas::Expr;
using nimblecas::ExprMemo;
using nimblecas::Result;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {
auto constant_fn(Expr value) -> std::function<Result<Expr>()> {
    return [value]() -> Result<Expr> { return value; };
}
}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.cache")
        .test("computes_once_for_structurally_equal_keys",
              [](TestContext& t) {
                  ExprMemo memo;
                  std::atomic<int> calls{0};
                  auto compute = [&]() -> Result<Expr> {
                      calls.fetch_add(1, std::memory_order_relaxed);
                      return Expr::symbol("result");
                  };
                  // two distinct objects, structurally equal
                  auto k1 = Expr::symbol("x").add(Expr::integer(1));
                  auto k2 = Expr::symbol("x").add(Expr::integer(1));
                  auto r1 = memo.get_or_compute(k1, compute);
                  auto r2 = memo.get_or_compute(k2, compute);
                  t.expect(calls.load() == 1, "compute ran once for equivalent keys");
                  t.expect(r1.has_value() && r2.has_value() && r1->is_equivalent_to(*r2),
                           "both calls return the same result");
                  t.expect_eq(memo.size(), std::size_t{1}, "one cached entry");
              })
        .test("distinct_keys_cache_separately",
              [](TestContext& t) {
                  ExprMemo memo;
                  memo.get_or_compute(Expr::symbol("a"), constant_fn(Expr::integer(1)));
                  memo.get_or_compute(Expr::symbol("b"), constant_fn(Expr::integer(2)));
                  memo.get_or_compute(Expr::symbol("a"), constant_fn(Expr::integer(9)));  // hit
                  t.expect_eq(memo.size(), std::size_t{2}, "two distinct entries (a reused)");
              })
        .test("error_results_are_cached",
              [](TestContext& t) {
                  ExprMemo memo;
                  auto key = Expr::symbol("z");
                  auto r = memo.get_or_compute(key, []() -> Result<Expr> {
                      return nimblecas::make_error<Expr>(nimblecas::MathError::overflow);
                  });
                  t.expect(!r.has_value() && r.error() == nimblecas::MathError::overflow,
                           "error propagated and cached");
                  // second lookup returns the cached error without recomputing
                  auto r2 = memo.get_or_compute(key, constant_fn(Expr::integer(0)));
                  t.expect(!r2.has_value(), "cached error returned on hit");
              })
        .test("concurrent_same_key_is_race_free_and_consistent",
              [](TestContext& t) {
                  ExprMemo memo;
                  const auto key = Expr::symbol("x").pow(Expr::integer(2));
                  std::atomic<int> calls{0};
                  auto compute = [&]() -> Result<Expr> {
                      calls.fetch_add(1, std::memory_order_relaxed);
                      std::this_thread::sleep_for(std::chrono::milliseconds(2));  // widen race
                      return Expr::integer(42);
                  };
                  constexpr int n = 16;
                  std::vector<std::optional<Result<Expr>>> out(n);
                  std::vector<std::thread> threads;
                  threads.reserve(n);
                  for (int i = 0; i < n; ++i) {
                      threads.emplace_back(
                          [&, i] { out[i].emplace(memo.get_or_compute(key, compute)); });
                  }
                  for (auto& th : threads) {
                      th.join();
                  }
                  bool all_ok = true;
                  for (auto& o : out) {
                      all_ok = all_ok && o->has_value() && (*o)->is_equivalent_to(Expr::integer(42));
                  }
                  t.expect(all_ok, "all threads observed the same result (42)");
                  t.expect_eq(memo.size(), std::size_t{1}, "exactly one entry after the race");
                  t.expect(calls.load() >= 1 && calls.load() <= n, "compute ran a bounded number of times");
              })
        .run();
}
