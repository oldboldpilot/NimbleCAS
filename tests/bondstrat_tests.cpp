// Tests for nimblecas.bondstrat: bond portfolio construction and relative value.
// @author Olumuyiwa Oluwasanmi
//
// The weighting algebra is exact, so every check is a hand-verified value: the barbell that
// hits a target duration, the duration-and-cash-neutral butterfly, the duration-neutral
// hedge ratio, and the carry/roll and duration-convexity P&L identities. Degenerate systems
// (equal durations, zero denominators, length mismatch) are asserted to ride the railway.

import std;
import nimblecas.core;
import nimblecas.bondstrat;
import nimblecas.testing;

using namespace nimblecas::bondstrat;
using nimblecas::MathError;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {
[[nodiscard]] auto close(double a, double b, double tol) -> bool { return std::abs(a - b) < tol; }
}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.bondstrat")
        .test("weighted average duration and its guards",
              [](TestContext& t) {
                  const std::array<double, 2> w{0.5, 0.5};
                  const std::array<double, 2> d{4.0, 8.0};
                  t.expect(close(weighted_average(w, d).value(), 6.0, 1e-12), "0.5·4 + 0.5·8 == 6");
                  const std::array<double, 3> d3{1.0, 2.0, 3.0};
                  t.expect(!weighted_average(w, d3).has_value(), "length mismatch -> domain_error");
              })
        .test("barbell hits the target duration",
              [](TestContext& t) {
                  // Target 6y from a 2y/10y barbell -> 50/50.
                  const auto w = barbell_weights(6.0, 2.0, 10.0).value();
                  t.expect(close(w[0], 0.5, 1e-12) && close(w[1], 0.5, 1e-12), "weights 0.5/0.5");
                  t.expect(close(w[0] * 2.0 + w[1] * 10.0, 6.0, 1e-12), "blended duration == target");
                  t.expect(!barbell_weights(6.0, 5.0, 5.0).has_value(), "equal durations -> div_by_zero");
              })
        .test("duration-neutral hedge ratio",
              [](TestContext& t) {
                  // (5·100)/(8·95) = 500/760.
                  t.expect(close(duration_hedge_ratio(5.0, 100.0, 8.0, 95.0).value(), 500.0 / 760.0, 1e-12),
                           "hedge ratio = (Da·Pa)/(Db·Pb)");
                  t.expect(!duration_hedge_ratio(5.0, 100.0, 0.0, 95.0).has_value(),
                           "zero denominator -> div_by_zero");
              })
        .test("duration- and cash-neutral butterfly",
              [](TestContext& t) {
                  // Wings 2y/10y, body 5y -> w_hi = (5−2)/(10−2) = 0.375, w_lo = 0.625.
                  const auto w = butterfly_weights(2.0, 5.0, 10.0).value();
                  t.expect(close(w[0], 0.625, 1e-12) && close(w[1], 0.375, 1e-12), "wing weights");
                  t.expect(close(w[0] + w[1], 1.0, 1e-12), "cash-neutral (sum 1)");
                  t.expect(close(w[0] * 2.0 + w[1] * 10.0, 5.0, 1e-12), "duration-neutral (== body)");
              })
        .test("carry/roll and duration-convexity P&L",
              [](TestContext& t) {
                  // (99 − 98 + 3)/98.
                  t.expect(close(carry_roll_return(98.0, 99.0, 3.0).value(), 4.0 / 98.0, 1e-12),
                           "carry + roll total return");
                  t.expect(!carry_roll_return(0.0, 99.0, 3.0).has_value(), "price<=0 -> domain_error");
                  // −D·dy + 0.5·C·dy²  = −7·0.01 + 0.5·50·0.0001 = −0.0675.
                  t.expect(close(duration_convexity_pnl(7.0, 50.0, 0.01), -0.0675, 1e-12),
                           "second-order price change");
              })
        .run();
}
