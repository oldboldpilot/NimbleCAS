// Tests for nimblecas.mmstrat: money-market strategies.
// @author Olumuyiwa Oluwasanmi
//
// Exact day-count identities, each hand-verified: repo interest/repayment on actual/360,
// the discount-price <-> bank-discount-rate round trip, the bank-discount -> bond-equivalent
// yield conversion, holding-period return, and the compounded effective rate of a deposit
// strip. Guards (bad face/price/basis/days, empty or mismatched strip) ride the railway.

import std;
import nimblecas.core;
import nimblecas.mmstrat;
import nimblecas.testing;

using namespace nimblecas::mmstrat;
using nimblecas::MathError;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {
[[nodiscard]] auto close(double a, double b, double tol) -> bool { return std::abs(a - b) < tol; }
}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.mmstrat")
        .test("repo interest and repayment (actual/360)",
              [](TestContext& t) {
                  // 1,000,000 at 5% for 30 days: 1e6·0.05·30/360 = 4166.6667.
                  t.expect(close(repo_interest(1'000'000.0, 0.05, 30.0).value(), 4166.66666667, 1e-6),
                           "repo interest");
                  t.expect(close(repo_repayment(1'000'000.0, 0.05, 30.0).value(), 1'004'166.66666667, 1e-5),
                           "repo repayment = principal + interest");
                  t.expect(!repo_interest(1e6, 0.05, 30.0, 0.0).has_value(), "basis<=0 -> domain_error");
              })
        .test("discount price <-> bank-discount rate round trip",
              [](TestContext& t) {
                  // 90-day bill, 5% discount: price = 100·(1 − 0.05·90/360) = 98.75.
                  t.expect(close(discount_price(100.0, 0.05, 90.0).value(), 98.75, 1e-9), "discount price");
                  t.expect(close(discount_rate_from_price(100.0, 98.75, 90.0).value(), 0.05, 1e-12),
                           "recover the 5% discount rate");
                  t.expect(!discount_rate_from_price(100.0, 98.75, 0.0).has_value(),
                           "days<=0 -> domain_error");
              })
        .test("bond-equivalent yield and holding-period return",
              [](TestContext& t) {
                  // BEY = 365·0.05/(360 − 0.05·90) = 18.25/355.5.
                  t.expect(close(bond_equivalent_yield(0.05, 90.0).value(), 18.25 / 355.5, 1e-12),
                           "bank-discount -> bond-equivalent yield");
                  // (100 − 98.75)/98.75.
                  t.expect(close(holding_period_return(98.75, 100.0, 0.0).value(), 1.25 / 98.75, 1e-12),
                           "holding-period return");
                  t.expect(!holding_period_return(0.0, 100.0, 0.0).has_value(), "begin<=0 -> domain_error");
              })
        .test("deposit strip effective rate compounds forward deposits",
              [](TestContext& t) {
                  // (1+0.04·0.5)(1+0.05·0.5) = 1.02·1.025 = 1.04550 over 1y -> 4.55%.
                  const std::array<double, 2> rates{0.04, 0.05};
                  const std::array<double, 2> accr{0.5, 0.5};
                  t.expect(close(deposit_strip_effective_rate(rates, accr).value(), 0.045500, 1e-9),
                           "compounded effective rate over the strip");
                  const std::array<double, 1> bad{0.5};
                  t.expect(!deposit_strip_effective_rate(rates, bad).has_value(),
                           "length mismatch -> domain_error");
                  t.expect(!deposit_strip_effective_rate({}, {}).has_value(), "empty strip -> domain_error");
              })
        .run();
}
