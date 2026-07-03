// Tests for nimblecas.core: CowPtr semantics and railway-oriented Result handling.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.testing;

using nimblecas::CowPtr;
using nimblecas::MathError;
using nimblecas::Result;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

auto divide(double numerator, double denominator) -> Result<double> {
    if (denominator == 0.0) {
        return nimblecas::make_error<double>(MathError::division_by_zero);
    }
    return numerator / denominator;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.core")
        .test("cow_shares_payload_until_write",
              [](TestContext& t) {
                  auto original = CowPtr<std::vector<int>>::make(std::vector<int>{1, 2, 3});
                  auto shared = original;  // O(1) refcount bump, shares payload
                  t.expect_eq(original.use_count(), 2L, "two handles share one payload");

                  shared.write().push_back(4);  // copy-on-write detach

                  t.expect_eq(original.read().size(), std::size_t{3}, "original unchanged");
                  t.expect_eq(shared.read().size(), std::size_t{4}, "mutated copy grew");
                  t.expect_eq(original.use_count(), 1L, "handles detached after write");
              })
        .test("cow_write_when_unique_does_not_copy",
              [](TestContext& t) {
                  auto p = CowPtr<std::vector<int>>::make(std::vector<int>{10});
                  const auto* before = p.operator->();
                  p.write().push_back(20);  // sole owner: no detach expected
                  t.expect_eq(p.operator->(), before, "unique payload mutated in place");
                  t.expect_eq(p.read().size(), std::size_t{2}, "value updated");
              })
        .test("result_carries_value",
              [](TestContext& t) {
                  auto r = divide(6.0, 2.0);
                  t.expect(r.has_value(), "6/2 yields a value");
                  t.expect_eq(r.value(), 3.0, "6/2 == 3");
              })
        .test("result_error_propagates_through_railway",
              [](TestContext& t) {
                  auto r = divide(1.0, 0.0).transform([](double v) { return v * 2.0; });
                  t.expect(!r.has_value(), "division by zero short-circuits the chain");
                  t.expect(r.error() == MathError::division_by_zero,
                           "error tag preserved through transform");
                  t.expect_eq(nimblecas::to_string_view(r.error()),
                              std::string_view{"division by zero"}, "error renders to text");
              })
        .run();
}
