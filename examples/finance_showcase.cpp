// @author Olumuyiwa Oluwasanmi
//
// finance_showcase.cpp — a comprehensive, self-contained demonstration of the NimbleCAS
// financial-mathematics stack: exact decimals (BigDecimal), the Excel-suite finance module
// (TVM, cashflows, depreciation, fixed income, swaps), currency/FX graphs, derivatives
// pricing (Black-Scholes + full Greeks, trees, Monte Carlo, LSM), portfolio & risk
// analytics, mean-variance optimization — plus a few symbolic-CAS examples to show
// end-to-end integration with the core.
//
// Every example prints its computed result; the expected value (hand-computed from the
// underlying formula) is stated in an inline comment so the output can be verified.
// Exact (Tier A) results are EXACT over Q; numerical (Tier B) results match to the stated
// tolerance; Monte Carlo results are reported with their standard error.

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.bigdecimal;
import nimblecas.finance;
import nimblecas.currency;
import nimblecas.pricing;
import nimblecas.analytics;
import nimblecas.portfolio;
import nimblecas.symbolic;
import nimblecas.diff;
import nimblecas.simplify;

using nimblecas::BigDecimal;
using nimblecas::BigRational;
using nimblecas::Expr;
using nimblecas::MathError;
using nimblecas::Rounding;
using nimblecas::to_string_view;

namespace fin = nimblecas::finance;
namespace cur = nimblecas::currency;
namespace pr  = nimblecas::pricing;
namespace an  = nimblecas::analytics;
namespace pf  = nimblecas::portfolio;

namespace {

// Exact rational from "num/den" or integer text (BigRational::from_string syntax).
[[nodiscard]] auto q(std::string_view s) -> BigRational {
    return BigRational::from_string(s).value();
}
[[nodiscard]] auto qi(std::int64_t v) -> BigRational { return BigRational::from_int(v); }
// Exact decimal literal.
[[nodiscard]] auto bd(std::string_view s) -> BigDecimal {
    return BigDecimal::from_string(s).value();
}
// Quantize an exact rational to a money string at `scale` decimals (banker's rounding).
[[nodiscard]] auto money(const BigRational& r, std::int32_t scale = 2) -> std::string {
    return BigDecimal::from_bigrational(r, scale, Rounding::half_even).to_string();
}

auto header(std::string_view title) -> void { std::println("\n=== {} ===", title); }

// Render the error of a Result that is EXPECTED to fail (an intentional honest-failure demo).
// Guards the .error()-on-a-success UB: if the call unexpectedly succeeded, say so instead.
template <class R>
[[nodiscard]] auto err_of(const R& r) -> std::string {
    if (r.has_value()) { return "SOLVED (unexpected)"; }
    return std::string(to_string_view(r.error()));
}

}  // namespace

auto main() -> int {
    std::println("NimbleCAS financial-mathematics showcase");

    // =======================================================================
    header("1. BigDecimal — exact base-10 arithmetic");
    // =======================================================================
    {
        // The classic binary-float failure: in IEEE double, 0.1 + 0.2 != 0.3.
        const BigDecimal sum = bd("0.1").add(bd("0.2"));
        std::println("0.1 + 0.2                = {}  (== 0.3 ? {})", sum.to_string(),
                     sum == bd("0.3"));                       // expect 0.3, true (exact)

        // 2.675 rounded to 2 places. In double, 2.675 is stored as 2.67499...,
        // so double-based rounding gives 2.67; exact decimal gives 2.68.
        std::println("round(2.675, 2) half_up  = {}",
                     bd("2.675").round(2, Rounding::half_up).to_string());   // expect 2.68
        std::println("round(2.675, 2) half_even= {}",
                     bd("2.675").round(2, Rounding::half_even).to_string()); // expect 2.68 (268 even)
        std::println("round(2.675, 2) half_down= {}",
                     bd("2.675").round(2, Rounding::half_down).to_string()); // expect 2.67

        // divide_exact: succeeds iff the quotient terminates in base 10 (honesty invariant).
        std::println("1 / 8 exact              = {}",
                     bd("1").divide_exact(bd("8")).value().to_string());     // expect 0.125
        const auto inexact = bd("1").divide_exact(bd("3"));
        std::println("1 / 3 exact              -> error: {}",
                     err_of(inexact));  // expect "inexact (rounding mode required)"
        std::println("1 / 3 at scale 6         = {}",
                     bd("1").divide(bd("3"), 6, Rounding::half_even).value().to_string());
                                                                             // expect 0.333333
        // Currency-scale quantize: ties resolved under an explicit policy.
        std::println("10.005 -> cents half_even= {}",
                     bd("10.005").quantize(2, Rounding::half_even).to_string()); // expect 10.00
        std::println("10.005 -> cents half_up  = {}",
                     bd("10.005").quantize(2, Rounding::half_up).to_string());   // expect 10.01

        // Exact integer-power compounding: 1.05^10 with every digit exact.
        std::println("1.05^10 exact            = {}",
                     bd("1.05").pow(10).value().to_string()); // expect 1.62889462677744140625

        // What a double literal really is: the exact dyadic value of 0.1.
        std::println("double 0.1 at 20 places  = {}",
                     BigDecimal::from_double(0.1, 20, Rounding::half_even).value().to_string());
                                                     // expect 0.10000000000000000555
    }

    // =======================================================================
    header("2. Finance TVM — a 30-year mortgage (exact over Q)");
    // =======================================================================
    {
        // $300,000 at 6% nominal annual, monthly payments: r = 0.005 = 1/200, n = 360.
        const BigRational r = q("1/200");
        const std::int64_t n = 360;
        const BigRational principal = qi(300000);

        const BigRational payment = fin::pmt(r, n, principal, qi(0)).value();
        std::println("PMT                      = {}", money(payment));   // expect ~ -1798.65

        // Amortization: interest + principal split, first three months and the last.
        for (std::int64_t per : {std::int64_t{1}, std::int64_t{2}, std::int64_t{3},
                                 std::int64_t{360}}) {
            const BigRational i = fin::ipmt(r, per, n, principal, qi(0)).value();
            const BigRational p = fin::ppmt(r, per, n, principal, qi(0)).value();
            std::println("  month {:>3}: IPMT {}  PPMT {}", per, money(i), money(p));
        }
        // month 1: IPMT -1500.00, PPMT -298.65; month 360: IPMT ~ -8.95, PPMT ~ -1789.70

        // IPMT + PPMT == PMT, exactly, every period.
        const BigRational i1 = fin::ipmt(r, 1, n, principal, qi(0)).value();
        const BigRational p1 = fin::ppmt(r, 1, n, principal, qi(0)).value();
        std::println("IPMT+PPMT == PMT exactly ? {}", i1.add(p1) == payment);  // expect true

        // Cumulative principal over the full term repays the loan exactly.
        const BigRational cp = fin::cumprinc(r, n, principal, 1, 360).value();
        std::println("CUMPRINC(1..360)         = {}", money(cp));        // expect -300000.00 exactly
        const BigRational ci = fin::cumipmt(r, n, principal, 1, 360).value();
        std::println("CUMIPMT(1..360)          = {}", money(ci));        // expect ~ -347514.57

        // The same problem through the fluent TvmProblem builder, quantized to cents.
        const BigDecimal fluent_pmt = fin::TvmProblem::create()
                                          .rate(q("1/200"))
                                          .nper(360)
                                          .present_value(qi(300000))
                                          .solve_pmt_money(2, Rounding::half_even)
                                          .value();
        std::println("fluent PMT (money)       = {}", fluent_pmt.to_string()); // expect -1798.65

        // FV of a savings plan: $500/month for 10 years at 6%/12.
        const BigRational fv_save = fin::fv(q("1/200"), 120, qi(-500), qi(0)).value();
        std::println("FV(0.5%,120,-500,0)      = {}", money(fv_save));   // expect ~ 81939.67

        // PV of a $10,000 lump sum in 5 years at 8%.
        const BigRational pv_lump = fin::pv(q("2/25"), 5, qi(0), qi(10000)).value();
        std::println("PV(8%,5,0,10000)         = {}", money(pv_lump));   // expect ~ -6805.83

        // Annuity due (payments at the START of the period, Excel type 1).
        const BigRational fv_due =
            fin::fv(q("7/100"), 10, qi(-2000), qi(0), fin::PaymentTiming::begin).value();
        std::println("FV due(7%,10,-2000)      = {}", money(fv_due));    // expect ~ 29567.20

        // NPER (Tier B, numerical): how long to double money at 5%?
        std::println("NPER(5%, doubling)       = {:.4f}", fin::nper(0.05, 0.0, -1000.0, 2000.0).value());
                                          // expect ~ 14.2067 (= ln 2 / ln 1.05)
    }

    // =======================================================================
    header("3. Cashflow analysis — NPV, IRR, XIRR, MIRR");
    // =======================================================================
    {
        // Excel's NPV convention (first value discounted one period), exact over Q.
        const std::vector<BigRational> flows{qi(-10000), qi(3000), qi(4200), qi(6800)};
        const BigRational npv_x = fin::npv(q("1/10"), flows).value();
        std::println("NPV(10%; -10k,3k,4.2k,6.8k) = {}", money(npv_x));  // expect ~ 1188.44

        // IRR (numerical) — the Excel documentation example.
        const std::vector<double> irr_flows{-70000, 12000, 15000, 18000, 21000, 26000};
        std::println("IRR                      = {:.6f}", fin::irr(irr_flows).value());
                                                                         // expect ~ 0.086630
        // MIRR with 10% finance and 12% reinvestment rates.
        const std::vector<double> mirr_flows{-1000, 300, 400, 400, 300};
        std::println("MIRR(10%,12%)            = {:.6f}",
                     fin::mirr(mirr_flows, 0.10, 0.12).value());         // expect ~ 0.1370

        // Dated flows through the fluent CashflowSchedule -> XIRR.
        const fin::Date d0 = fin::Date::of(2025, 1, 1).value();
        const fin::Date d1 = fin::Date::of(2025, 7, 1).value();
        const fin::Date d2 = fin::Date::of(2026, 1, 1).value();
        auto sched = fin::CashflowSchedule::create();
        const double xirr_v =
            sched.at(d0, qi(-10000)).at(d1, qi(2500)).at(d2, qi(8500)).extended_irr().value();
        std::println("XIRR (dated schedule)    = {:.4f}", xirr_v);       // expect ~ 0.114

        // Undated fluent schedule: NPV discounts flow 0 at period 0 (IRR convention).
        auto sched2 = fin::CashflowSchedule::create();
        const BigRational npv0 = sched2.flow(qi(-10000)).flow(qi(3000)).flow(qi(4200))
                                     .flow(qi(6800)).net_present_value(q("1/10")).value();
        std::println("schedule NPV at t=0      = {}", money(npv0));      // expect ~ 1307.29
                                            // (= 1188.44 * 1.1 — the two conventions differ by one period)
    }

    // =======================================================================
    header("4. Rate conversion & annuity variants");
    // =======================================================================
    {
        // EFFECT: 5.25% nominal compounded quarterly — exact rational, then quantized.
        const BigRational eff = fin::effect(q("21/400"), 4).value();
        std::println("EFFECT(5.25%, 4) exact   = {}", eff.to_string());
        std::println("EFFECT(5.25%, 4)         = {}", money(eff, 7));    // expect ~ 0.0535427
        std::println("NOMINAL(back)            = {:.6f}",
                     fin::nominal(0.0535427, 4).value());                // expect ~ 0.052500

        std::println("RRI(10, 1000->2000)      = {:.6f}", fin::rri(10, 1000, 2000).value());
                                                          // expect ~ 0.071773 (= 2^(1/10)-1)

        // Growing annuity: first payment 1000, r=8%, g=3%, 20 years.
        const BigRational gpv =
            fin::growing_annuity_pv(q("2/25"), q("3/100"), 20, qi(1000)).value();
        std::println("growing annuity PV       = {}", money(gpv));       // expect ~ 12250.05

        // Growing perpetuity: 100/(8% - 3%) = 2000 exactly.
        const BigRational perp =
            fin::growing_perpetuity_pv(q("2/25"), q("3/100"), qi(100)).value();
        std::println("growing perpetuity PV    = {}", money(perp));      // expect 2000.00 exactly

        // Honest failure: growth >= rate diverges -> domain_error, never a number.
        const auto diverge = fin::growing_perpetuity_pv(q("3/100"), q("1/20"), qi(100));
        std::println("g>r perpetuity           -> error: {}", err_of(diverge));
                                                             // expect "domain error"
        // DOLLARDE / DOLLARFR: 1.02 read in 32nds -> 1 + 2/32 = 1.0625 exactly.
        std::println("DOLLARDE(1.02, 32)       = {}", money(fin::dollarde(q("51/50"), 32).value(), 4));
                                                                         // expect 1.0625 exactly
        std::println("DOLLARFR(1.0625, 32)     = {}", money(fin::dollarfr(q("17/16"), 32).value(), 2));
                                                                         // expect 1.02 exactly
    }

    // =======================================================================
    header("5. Depreciation schedule — cost 10000, salvage 1000, life 5");
    // =======================================================================
    {
        const BigRational cost = qi(10000);
        const BigRational salv = qi(1000);
        std::println("SLN per year             = {}", money(fin::sln(cost, salv, 5).value()));
                                                                         // expect 1800.00 exactly
        // Full SYD and DDB schedules, exact over Q.
        for (std::int64_t per = 1; per <= 5; ++per) {
            const BigRational s = fin::syd(cost, salv, 5, per).value();
            const BigRational d = fin::ddb(cost, salv, 5, per).value();
            std::println("  year {}: SYD {}   DDB {}", per, money(s), money(d));
        }
        // SYD: 3000, 2400, 1800, 1200, 600 (sum 9000 exactly)
        // DDB: 4000, 2400, 1440, 864, 296  (year 5 capped at salvage floor)

        // DB (Excel hybrid: rate rounded to 3 decimals -> 0.369).
        std::println("DB year 1                = {:.2f}", fin::db(10000, 1000, 5, 1).value());
                                                                         // expect 3690.00
        std::println("DB year 2                = {:.2f}", fin::db(10000, 1000, 5, 2).value());
                                                                         // expect ~ 2328.39
        // VDB: DDB-with-SL-switch totals over sub-ranges.
        std::println("VDB periods 0..3         = {}", money(fin::vdb(cost, salv, 5, 0, 3).value()));
                                                                         // expect 7840.00 exactly
        std::println("VDB periods 0..5         = {}", money(fin::vdb(cost, salv, 5, 0, 5).value()));
                                                                         // expect 9000.00 exactly
    }

    // =======================================================================
    header("6. Fixed income — bond analytics & T-bills");
    // =======================================================================
    {
        // 5-year 5% semiannual bond at 6% yield, settled on a coupon date, 30/360.
        const fin::Date settle = fin::Date::of(2026, 1, 15).value();
        const fin::Date mature = fin::Date::of(2031, 1, 15).value();
        const double px = fin::bond_price(settle, mature, 0.05, 0.06, 100.0, 2,
                                          fin::DayCount::thirty_360).value();
        std::println("bond price               = {:.4f}", px);           // expect ~ 95.7349
        std::println("bond yield (from price)  = {:.6f}",
                     fin::bond_yield(settle, mature, 0.05, px, 100.0, 2,
                                     fin::DayCount::thirty_360).value());  // expect ~ 0.060000
        std::println("Macaulay duration        = {:.4f}",
                     fin::bond_duration(settle, mature, 0.05, 0.06, 2,
                                        fin::DayCount::thirty_360).value()); // expect ~ 4.4717 yrs
        std::println("modified duration        = {:.4f}",
                     fin::bond_mduration(settle, mature, 0.05, 0.06, 2,
                                         fin::DayCount::thirty_360).value()); // expect ~ 4.3415 yrs
        std::println("convexity                = {:.3f}",
                     fin::bond_convexity(settle, mature, 0.05, 0.06, 2,
                                         fin::DayCount::thirty_360).value()); // expect ~ 22.3 yrs^2
        std::println("coupons remaining        = {}",
                     fin::coupon_num(settle, mature, 2).value());        // expect 10 exactly

        // Accrued interest 60/360 of a year after the last coupon at 5%.
        const fin::Date last_cpn = fin::Date::of(2026, 1, 15).value();
        const fin::Date mid = fin::Date::of(2026, 3, 15).value();
        std::println("accrued interest         = {:.6f}",
                     fin::accrued_interest(last_cpn, mid, 0.05, 2,
                                           fin::DayCount::thirty_360).value()); // expect ~ 0.833333

        // T-bill: 180 days at a 4.5% discount rate (actual/360).
        const fin::Date ts = fin::Date::of(2026, 2, 1).value();
        const fin::Date tm = fin::Date::of(2026, 7, 31).value();
        const double tp = fin::tbill_price(ts, tm, 0.045).value();
        std::println("T-bill price             = {:.4f}", tp);           // expect 97.7500 exactly
        std::println("T-bill yield             = {:.6f}", fin::tbill_yield(ts, tm, tp).value());
                                                                         // expect ~ 0.046036
        std::println("T-bill bond-equiv yield  = {:.6f}", fin::tbill_eq(ts, tm, 0.045).value());
                                                                         // expect ~ 0.046675
    }

    // =======================================================================
    header("7. Interest-rate & currency swaps");
    // =======================================================================
    {
        // 3-year annual payer swap: accruals 1.0; curve dfs; forwards.
        const std::vector<double> accr{1.0, 1.0, 1.0};
        const std::vector<double> dfs{0.97, 0.94, 0.91};
        const std::vector<double> fwd{0.030, 0.032, 0.034};
        const double par = fin::swap_par_rate(accr, dfs, fwd).value();
        std::println("swap par rate            = {:.6f}", par);          // expect ~ 0.032312
        const double sv = fin::interest_rate_swap_value(1'000'000.0, 0.03, accr, dfs, fwd).value();
        std::println("payer swap value @3%     = {:.2f}", sv);           // expect 5520.00 exactly
        // At the par rate the swap values to zero (by construction).
        std::println("payer swap value @par    = {:.6f}",
                     fin::interest_rate_swap_value(1'000'000.0, par, accr, dfs, fwd).value());
                                                                         // expect ~ 0.000000
        std::println("ccy swap value           = {:.2f}",
                     fin::currency_swap_value(1'000'000.0, 850'000.0, 1.10).value());
                                                     // expect 65000.00 (= 1e6 - 1.10*850e3)
    }

    // =======================================================================
    header("8. Currency & FX — exact rate graph");
    // =======================================================================
    {
        auto fx = cur::RateTable::create();
        (void)fx.add("EUR", "USD", q("11/10"));   // 1 EUR = 1.10 USD (+ reciprocal)
        (void)fx.add("USD", "JPY", qi(150));      // 1 USD = 150 JPY
        (void)fx.add("GBP", "USD", q("5/4"));     // 1 GBP = 1.25 USD

        // Direct conversion, quantized to the target's minor unit.
        const cur::Money m100 = cur::Money::parse("100.00", "EUR").value();
        std::println("100.00 EUR -> USD        = {}",
                     fx.convert(m100, "USD", 2, Rounding::half_even).value().to_string());
                                                                         // expect 110.00 USD
        // Cross rate via the graph: EUR -> USD -> JPY = 1.10 * 150 = 165 exactly.
        std::println("cross EUR->JPY           = {}",
                     fx.cross_rate("EUR", "JPY").value().to_string());   // expect 165
        std::println("100.00 EUR -> JPY        = {}",
                     fx.convert(m100, "JPY", 0, Rounding::half_even).value().to_string());
                                                                         // expect 16500 JPY
        // Cross through the reciprocal edge: GBP -> JPY = 1.25 * 150 = 187.5.
        std::println("cross GBP->JPY           = {}",
                     fx.cross_rate("GBP", "JPY").value().to_string());   // expect 375/2

        // Honest failure: no quoted route -> not_implemented, never a fabricated rate.
        const auto no_route = fx.convert(m100, "CHF", 2, Rounding::half_even);
        std::println("EUR -> CHF               -> error: {}", err_of(no_route));
                                                             // expect "not implemented"
        // Money refuses cross-currency arithmetic.
        const auto bad_add = cur::Money::parse("1.00", "USD").value()
                                 .add(cur::Money::parse("1.00", "EUR").value());
        std::println("1 USD + 1 EUR            -> error: {}", err_of(bad_add));
                                                             // expect "domain error"

        // Triangular arbitrage on DIRECT quotes only (no auto-reciprocals):
        // EUR->USD 1.10, USD->GBP 0.80, GBP->EUR 1.15: product 1.012 -> 1.2% gross profit.
        auto tri = cur::RateTable::create();
        (void)tri.add("EUR", "USD", q("11/10"), false);
        (void)tri.add("USD", "GBP", q("4/5"), false);
        (void)tri.add("GBP", "EUR", q("23/20"), false);
        const BigRational loop = tri.triangular_product("EUR", "USD", "GBP").value();
        std::println("triangular product       = {} (= {})", loop.to_string(), money(loop, 4));
                                                                         // expect 253/250 = 1.012
        std::println("arbitrage (tol 1/1000)?  = {}",
                     tri.has_triangular_arbitrage("EUR", "USD", "GBP", q("1/1000")).value());
                                                                         // expect true

        // Covered-interest-parity forward: S=1.10 USD/EUR, r_EUR=2%, r_USD=5%, t=1/2.
        const BigRational f = cur::forward_rate(q("11/10"), q("1/50"), q("1/20"), q("1/2")).value();
        std::println("CIP 6m forward           = {} (= {})", f.to_string(), money(f, 6));
                                                     // expect 451/404 = ~1.116337 (USD at a
                                                     // forward discount to the higher-rate leg)
    }

    // =======================================================================
    header("9. Options — Black-Scholes, full Greeks, implied vol");
    // =======================================================================
    // Reference contract: S=100, K=100, r=5%, q=0, sigma=20%, T=1 (d1=0.35, d2=0.15).
    const pr::OptionSpec call = pr::OptionSpec{}
                                    .with_spot(100.0).with_strike(100.0).with_rate(0.05)
                                    .with_volatility(0.20).with_expiry(1.0)
                                    .with_type(pr::OptionType::call);
    const pr::OptionSpec put = call.with_type(pr::OptionType::put);
    {
        const double c = pr::black_scholes_price(call).value();
        const double p = pr::black_scholes_price(put).value();
        std::println("BS call                  = {:.4f}", c);            // expect ~ 10.4506
        std::println("BS put                   = {:.4f}", p);            // expect ~ 5.5735
        // Put-call parity: C - P == S - K e^{-rT} = 100 - 95.1229 = 4.8771.
        std::println("C - P                    = {:.4f} (parity S-Ke^-rT = {:.4f})",
                     c - p, 100.0 - 100.0 * std::exp(-0.05));            // expect both ~ 4.8771

        const pr::Greeks g = pr::black_scholes_greeks(call).value();
        std::println("delta                    = {:.6f}", g.delta);      // expect ~ 0.636831
        std::println("gamma                    = {:.6f}", g.gamma);      // expect ~ 0.018762
        std::println("vega                     = {:.4f}", g.vega);       // expect ~ 37.5240
        std::println("theta                    = {:.4f}", g.theta);      // expect ~ -6.4140
        std::println("rho                      = {:.4f}", g.rho);        // expect ~ 53.2325

        const pr::ExtendedGreeks x = pr::black_scholes_extended_greeks(call).value();
        std::println("vanna                    = {:.6f}", x.vanna);      // expect ~ -0.281430
        std::println("charm                    = {:.6f}", x.charm);      // expect ~ +0.065667
                                                    // (this codebase: -dDelta/d calendar time)
        std::println("vomma                    = {:.4f}", x.vomma);      // expect ~ 9.8500
        std::println("veta                     = {:.4f}", x.veta);       // expect ~ -16.464
        std::println("speed                    = {:.6f}", x.speed);      // expect ~ -0.000516
        std::println("zomma                    = {:.6f}", x.zomma);      // expect ~ -0.088885
        std::println("color                    = {:.6f}", x.color);      // expect ~ +0.010530
        std::println("lambda (elasticity)      = {:.4f}", x.lambda);     // expect ~ 6.0937
        std::println("dual delta               = {:.6f}", x.dual_delta); // expect ~ -0.532323
        std::println("dual gamma               = {:.6f}", x.dual_gamma); // expect ~ 0.018762
        std::println("epsilon (psi)            = {:.4f}", x.epsilon);    // expect ~ -63.6831
        std::println("vera                     = {:.4f}", x.vera);       // expect ~ -65.667
        std::println("ultima                   = {:.3f}", x.ultima);     // expect ~ -182.69

        // Implied volatility inverts the price back to sigma.
        std::println("implied vol at BS price  = {:.6f}",
                     pr::implied_volatility(call, c).value());           // expect ~ 0.200000
    }

    // =======================================================================
    header("10. Lattices & Monte Carlo");
    // =======================================================================
    {
        // Kamrad-Ritchken trinomial: European converges to Black-Scholes.
        std::println("trinomial Euro call 500  = {:.4f}",
                     pr::trinomial_price(call, 500, pr::Exercise::european).value());
                                                                         // expect ~ 10.45
        // American put carries an early-exercise premium over the European 5.5735.
        std::println("trinomial Amer put 500   = {:.4f}",
                     pr::trinomial_price(put, 500, pr::Exercise::american).value());
                                                                         // expect ~ 6.09
        // Bermudan (exercise only at t=0.5) sits between European and American.
        const std::array<double, 1> berm{0.5};
        std::println("trinomial Berm put (0.5) = {:.4f}",
                     pr::trinomial_price(put, 500, pr::Exercise::bermudan, berm).value());
                                                     // expect between 5.5735 and ~6.09
        // Monte Carlo European (antithetic, reproducible).
        const pr::McResult mc = pr::monte_carlo_european(call, 200'000, 12345).value();
        std::println("MC Euro call             = {:.4f} +/- {:.4f} ({} paths)",
                     mc.price, mc.std_error, mc.paths);   // expect ~ 10.45 within ~2 SE

        // Asian: geometric closed form (oracle) vs arithmetic MC with control variate.
        std::println("geometric Asian (50)     = {:.4f}",
                     pr::geometric_asian_price(call, 50).value());       // expect ~ 5.5 (≈ vol/sqrt3)
        const pr::McResult asian = pr::monte_carlo_asian(call, 100'000, 50, 42, true).value();
        std::println("arithmetic Asian MC+CV   = {:.4f} +/- {:.4f}", asian.price, asian.std_error);
                                                     // expect ~ 5.75 (slightly above geometric)
        // Longstaff-Schwartz American put (least-squares MC).
        const pr::McResult lsm = pr::longstaff_schwartz_american(put, 100'000, 50, 7).value();
        std::println("LSM American put         = {:.4f} +/- {:.4f}", lsm.price, lsm.std_error);
                                                     // expect ~ 6.09 (matches the lattice)
    }

    // =======================================================================
    header("11. Black-76, digitals, barriers, a straddle book");
    // =======================================================================
    {
        // Black-76 on a forward F=100 (options on futures).
        std::println("Black-76 call F=K=100    = {:.4f}",
                     pr::black76_price(true, 100.0, 100.0, 0.05, 0.20, 1.0).value());
                                                                         // expect ~ 7.5770
        // Digitals: cash-or-nothing pays e^{-rT} N(d2); asset-or-nothing pays S N(d1).
        std::println("digital cash (pays 1)    = {:.6f}",
                     pr::digital_cash_or_nothing(call, 1.0).value());    // expect ~ 0.532323
        std::println("digital asset            = {:.4f}",
                     pr::digital_asset_or_nothing(call).value());        // expect ~ 63.6831
        // In/out barrier parity: down-and-in + down-and-out == vanilla (up to MC error).
        const pr::McResult b_in = pr::barrier_option_mc(call, 90.0, true, 100'000, 50, 2024).value();
        const pr::McResult b_out = pr::barrier_option_mc(call, 90.0, false, 100'000, 50, 2024).value();
        std::println("down-in call MC          = {:.4f} +/- {:.4f}", b_in.price, b_in.std_error);
        std::println("down-out call MC         = {:.4f} +/- {:.4f}", b_out.price, b_out.std_error);
        std::println("in + out (parity check)  = {:.4f}", b_in.price + b_out.price);
                                                     // expect ~ 10.45 (the vanilla BS call)

        // A long straddle as a composable Portfolio: value/Greeks/payoff aggregate legs.
        auto book = pr::Portfolio::create();
        auto& straddle = book.add(call, 1.0).add(put, 1.0);
        std::println("straddle value           = {:.4f}", straddle.value().value());
                                                     // expect ~ 16.0242 (10.4506 + 5.5735)
        const pr::Greeks sg = straddle.greeks().value();
        std::println("straddle delta           = {:.6f}", sg.delta);     // expect ~ 0.273662
        std::println("straddle gamma           = {:.6f}", sg.gamma);     // expect ~ 0.037524
        std::println("straddle vega            = {:.4f}", sg.vega);      // expect ~ 75.0479
        std::println("straddle payoff at 80    = {:.2f}", straddle.payoff_at(80.0));  // expect 20.00
        std::println("straddle payoff at 100   = {:.2f}", straddle.payoff_at(100.0)); // expect 0.00
        std::println("straddle payoff at 125   = {:.2f}", straddle.payoff_at(125.0)); // expect 25.00
    }

    // =======================================================================
    header("12. Analytics — returns, ratios, drawdown, VaR");
    // =======================================================================
    // A 12-period return series and its market benchmark (used again in section 13).
    const std::vector<double> rets{0.02, -0.01, 0.03, 0.015, -0.005, 0.025,
                                   0.01, -0.02, 0.04, 0.005, 0.02, -0.015};
    const std::vector<double> mkt{0.015, -0.005, 0.02, 0.01, 0.0, 0.02,
                                  0.005, -0.015, 0.03, 0.0, 0.015, -0.01};
    {
        // Returns from a price path.
        const std::vector<double> prices{100, 102, 101, 105, 107, 110, 108, 112};
        const auto sr = an::simple_returns(prices).value();
        std::println("simple returns[0..2]     = {:.6f}, {:.6f}, {:.6f}", sr[0], sr[1], sr[2]);
                                            // expect 0.020000, -0.009804, 0.039604
        std::println("mean(returns)            = {:.6f}", an::mean(sr).value());   // expect ~ 0.016534
        std::println("stddev(returns)          = {:.6f}", an::stddev(sr).value()); // expect ~ 0.022369
        std::println("max drawdown             = {:.6f}", an::max_drawdown(prices).value());
                                            // expect ~ 0.018182 (110 -> 108)

        // Statistics on the 12-period series vs its benchmark.
        std::println("mean / std (12-series)   = {:.6f} / {:.6f}",
                     an::mean(rets).value(), an::stddev(rets).value());
                                            // expect ~ 0.009583 / 0.018885
        std::println("covariance(a, m)         = {:.8f}", an::covariance(rets, mkt).value());
                                            // expect ~ 0.00025322
        std::println("correlation(a, m)        = {:.6f}", an::correlation(rets, mkt).value());
                                            // expect ~ 0.99 (highly correlated by construction)
        std::println("Sharpe (rf=0)            = {:.4f}", an::sharpe_ratio(rets, 0.0).value());
                                            // expect ~ 0.5074
        std::println("Sharpe annualised (12/y) = {:.4f}",
                     an::sharpe_ratio(rets, 0.0, an::Annualisation{12.0}).value());
                                            // expect ~ 1.7577 (= 0.5074 * sqrt 12)
        std::println("Sortino (mar=0)          = {:.4f}", an::sortino_ratio(rets, 0.0).value());
                                            // expect ~ 1.2122
        std::println("beta                     = {:.4f}", an::beta(rets, mkt).value());
                                            // expect ~ 1.3770
        std::println("Jensen alpha (rf=0)      = {:.6f}", an::alpha(rets, mkt, 0.0).value());
                                            // expect ~ -0.000171
        std::println("Treynor (rf=0)           = {:.6f}", an::treynor_ratio(rets, mkt, 0.0).value());
                                            // expect ~ 0.006960
        std::println("information ratio        = {:.4f}", an::information_ratio(rets, mkt).value());
                                            // expect ~ 0.4282

        // Tail risk: historical (order statistic) and parametric (Gaussian).
        std::println("hist VaR 95%             = {:.6f}",
                     an::value_at_risk_historical(rets, 0.95).value());  // expect 0.020000
        std::println("hist CVaR 95%            = {:.6f}",
                     an::conditional_var_historical(rets, 0.95).value()); // expect 0.020000
                                            // (n=12 -> ceil(0.05*12)=1 tail observation: -0.02)
        std::println("Gaussian VaR 95%         = {:.6f}",
                     an::value_at_risk_gaussian(0.001, 0.02, 0.95).value());
                                            // expect ~ 0.031897 (= 1.64485*0.02 - 0.001)
        std::println("Gaussian CVaR 95%        = {:.6f}",
                     an::conditional_var_gaussian(0.001, 0.02, 0.95).value());
                                            // expect ~ 0.040254 (= 0.02*phi(z)/0.05 - 0.001)
    }

    // =======================================================================
    header("13. Portfolio optimization & integrated risk report");
    // =======================================================================
    {
        // Two assets: sigma1=20%, sigma2=30%, cov=0.01 (rho=1/6); mu=[8%,12%]; rf=2%.
        const std::vector<std::vector<double>> cov{{0.04, 0.01}, {0.01, 0.09}};
        const std::vector<double> mu{0.08, 0.12};

        const auto mv = an::min_variance_weights(cov).value();
        std::println("min-variance weights     = [{:.6f}, {:.6f}]", mv[0], mv[1]);
                                            // expect [0.727273, 0.272727] (= 8/11, 3/11)
        std::println("min-variance risk        = {:.6f}",
                     std::sqrt(an::portfolio_variance(mv, cov).value())); // expect ~ 0.178377
        std::println("min-variance return      = {:.6f}",
                     an::portfolio_return(mv, mu).value());              // expect ~ 0.090909 (= 1/11)

        const auto tw = an::tangency_weights(cov, mu, 0.02).value();
        std::println("tangency weights         = [{:.6f}, {:.6f}]", tw[0], tw[1]);
                                            // expect [0.564103, 0.435897] (= 22/39, 17/39)

        // Efficient frontier: 3 points from the min-variance return up to max(mu).
        const auto frontier = an::efficient_frontier(cov, mu, 3).value();
        for (const auto& pt : frontier) {
            std::println("  frontier: ret {:.4f}  risk {:.4f}  w=[{:.4f}, {:.4f}]",
                         pt.ret, pt.risk, pt.weights[0], pt.weights[1]);
        }
        // expect: (0.0909, 0.1784, [0.7273,0.2727]), (~0.1055, ~0.21, ...),
        //         (0.1200, 0.3000, [0.0000,1.0000]) — the top point is 100% asset 2.

        // A singular covariance (two identical assets): the Cholesky path refuses honestly...
        const std::vector<std::vector<double>> singular{{0.04, 0.04}, {0.04, 0.04}};
        const auto refused = an::min_variance_weights(singular);
        std::println("singular cov (Cholesky)  -> error: {}", err_of(refused));
                                                             // expect "error: domain error"
        // ...while the ridge-regularized LU path solves it (Sigma + lambda*I).
        const auto ridge_w = pf::min_variance_weights(singular, 1e-8).value();
        std::println("ridge min-var weights    = [{:.6f}, {:.6f}]", ridge_w[0], ridge_w[1]);
                                            // expect [0.500000, 0.500000] by symmetry
        const auto x = pf::lu_solve_ridge(singular, std::vector<double>{1.0, 1.0}, 1e-8).value();
        std::println("lu_solve_ridge x         = [{:.4f}, {:.4f}]", x[0], x[1]);
                                            // expect ~ [12.5, 12.5] (= 1/(0.08+lambda) each)
        const auto ridge_tan = pf::tangency_weights(cov, mu, 0.02, 0.0).value();
        std::println("ridge tangency (l=0)     = [{:.6f}, {:.6f}]", ridge_tan[0], ridge_tan[1]);
                                            // expect [0.564103, 0.435897] (matches Cholesky path)

        // The one-call integrated risk report over the section-12 series.
        const pf::RiskReport rr = pf::analyze(rets, mkt, 0.0, 0.95).value();
        std::println("RiskReport:");
        std::println("  sharpe        = {:.4f}", rr.sharpe);          // expect ~ 0.5074
        std::println("  sortino       = {:.4f}", rr.sortino);         // expect ~ 1.2122
        std::println("  treynor       = {:.6f}", rr.treynor);         // expect ~ 0.006960
        std::println("  jensen alpha  = {:.6f}", rr.jensen_alpha);    // expect ~ -0.000171
        std::println("  beta          = {:.4f}", rr.beta);            // expect ~ 1.3770
        std::println("  max drawdown  = {:.4f}", rr.max_drawdown);    // expect 0.0200
        std::println("  hist VaR/CVaR = {:.4f} / {:.4f}", rr.var_historical, rr.cvar_historical);
                                                                      // expect 0.0200 / 0.0200
        std::println("  parm VaR/CVaR = {:.4f} / {:.4f}", rr.var_parametric, rr.cvar_parametric);
                                                                      // expect ~ 0.0215 / ~ 0.0294
    }

    // =======================================================================
    header("14. Symbolic CAS core — differentiate & simplify");
    // =======================================================================
    {
        const Expr x = Expr::symbol("x");
        const Expr y = Expr::symbol("y");

        // d/dx (x^3 + 2x) = 3x^2 + 2.
        const Expr poly = Expr::sum({x.pow(Expr::integer(3)), Expr::integer(2).mul(x)});
        const Expr dpoly = nimblecas::simplify(nimblecas::differentiate(poly, "x").value()).value();
        std::println("d/dx (x^3 + 2x)          = {}", dpoly.to_string());
                                            // expect 2 + 3*x^2 (term order may vary)

        // d/dx sin(x) = cos(x).
        const Expr dsin = nimblecas::simplify(
            nimblecas::differentiate(Expr::apply("sin", {x}), "x").value()).value();
        std::println("d/dx sin(x)              = {}", dsin.to_string());  // expect cos(x)

        // d/dx (x * exp(x)) = exp(x) + x*exp(x) (product rule).
        const Expr dxex = nimblecas::simplify(
            nimblecas::differentiate(x.mul(Expr::apply("exp", {x})), "x").value()).value();
        std::println("d/dx (x*exp(x))          = {}", dxex.to_string());
                                            // expect exp(x) + x*exp(x) (form may vary)

        // simplify(x + x + 0*y) = 2x.
        const Expr messy = Expr::sum({x, x, Expr::integer(0).mul(y)});
        std::println("simplify(x + x + 0*y)    = {}",
                     nimblecas::simplify(messy).value().to_string());     // expect 2*x
    }

    std::println("\nAll 14 sections completed — NimbleCAS finance stack showcase done.");
    return 0;
}
