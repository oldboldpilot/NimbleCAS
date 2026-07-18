// Tests for nimblecas.currency: exact FX — money, rate graph, cross rates, arbitrage.
// @author Olumuyiwa Oluwasanmi
//
// Proves the FX graph is EXACT and honest: direct and cross rates are exact rationals,
// conversion quantizes to minor units only at the boundary, cross-rate pathfinding derives
// an unquoted pair by chaining through the graph, a missing route is refused (not_implemented)
// rather than fabricated, triangular arbitrage is detected exactly, and covered-interest-parity
// forwards hit a hand-computed rational. Money refuses cross-currency addition.

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.bigdecimal;
import nimblecas.currency;
import nimblecas.testing;

using nimblecas::BigDecimal;
using nimblecas::BigRational;
using nimblecas::MathError;
using nimblecas::Rounding;
using namespace nimblecas::currency;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {
[[nodiscard]] auto q(std::string_view s) -> BigRational { return BigRational::from_string(s).value(); }
}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.currency")
        .test("direct rate, reciprocal, and conversion",
              [](TestContext& t) {
                  auto table = RateTable::create();
                  t.expect(table.add("USD", "EUR", q("9/10")).has_value(), "add USD->EUR 0.9");
                  t.expect(table.direct_rate("USD", "EUR").value() == q("9/10"),
                           "direct USD->EUR == 9/10");
                  // Reciprocal auto-added.
                  t.expect(table.direct_rate("EUR", "USD").value() == q("10/9"),
                           "reciprocal EUR->USD == 10/9");
                  // Convert 100 USD -> EUR at cents: 100 * 0.9 == 90.00.
                  auto m = table.convert(Money::parse("100", "USD").value(), "EUR", 2,
                                         Rounding::half_even);
                  t.expect(m.has_value() && m->code() == "EUR", "converted to EUR");
                  t.expect_eq(m->amount().to_string(), std::string{"90.00"}, "100 USD -> 90.00 EUR");
              })
        .test("cross rate via graph pathfinding",
              [](TestContext& t) {
                  auto table = RateTable::create();
                  (void)table.add("USD", "EUR", q("9/10"));
                  (void)table.add("EUR", "GBP", q("8/10"));
                  // No direct USD->GBP quote; derived by chaining: 0.9 * 0.8 == 0.72 == 18/25.
                  t.expect(!table.direct_rate("USD", "GBP").has_value(), "USD->GBP not direct");
                  t.expect(table.cross_rate("USD", "GBP").value() == q("18/25"),
                           "USD->GBP cross == 18/25");
                  // Identity and missing-route honesty.
                  t.expect(table.cross_rate("USD", "USD").value() == BigRational::from_int(1),
                           "USD->USD == 1");
                  t.expect(table.cross_rate("USD", "JPY").error() == MathError::not_implemented,
                           "no route -> not_implemented");
              })
        .test("triangular arbitrage detection (exact)",
              [](TestContext& t) {
                  // Reciprocal-consistent quotes -> product exactly 1 -> no arbitrage.
                  auto fair = RateTable::create();
                  (void)fair.add("USD", "EUR", q("9/10"));
                  (void)fair.add("EUR", "GBP", q("8/10"));
                  (void)fair.add("GBP", "USD", q("25/18"));  // exact inverse of 0.72
                  t.expect(fair.triangular_product("USD", "EUR", "GBP").value() ==
                               BigRational::from_int(1),
                           "consistent triangle product == 1");
                  t.expect(fair.has_triangular_arbitrage("USD", "EUR", "GBP",
                                                         BigRational::from_int(0)).value() == false,
                           "no arbitrage at zero tolerance");
                  // Mispriced GBP->USD leg -> product != 1 -> arbitrage.
                  auto arb = RateTable::create();
                  (void)arb.add("USD", "EUR", q("9/10"), false);
                  (void)arb.add("EUR", "GBP", q("8/10"), false);
                  (void)arb.add("GBP", "USD", q("15/10"), false);  // too generous
                  t.expect(arb.has_triangular_arbitrage("USD", "EUR", "GBP",
                                                        BigRational::from_int(0)).value() == true,
                           "mispriced triangle flags arbitrage");
              })
        .test("covered-interest-parity forward (exact)",
              [](TestContext& t) {
                  // F = 100 * (1 + 0.02) / (1 + 0.05) == 10200/105 == 680/7.
                  auto f = forward_rate(BigRational::from_int(100), q("1/20"), q("1/50"),
                                        BigRational::from_int(1));
                  t.expect(f.value() == q("680/7"), "forward == 680/7");
              })
        .test("money refuses cross-currency addition",
              [](TestContext& t) {
                  auto usd = Money::parse("10.00", "USD").value();
                  auto eur = Money::parse("10.00", "EUR").value();
                  t.expect(usd.add(eur).error() == MathError::domain_error,
                           "USD + EUR -> domain_error");
                  t.expect(usd.add(Money::parse("5.00", "USD").value()).value().amount() ==
                               BigDecimal::from_string("15.00").value(),
                           "USD + USD == 15.00");
              })
        .run();
}
