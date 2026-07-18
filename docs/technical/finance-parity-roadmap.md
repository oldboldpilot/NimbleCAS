# Financial Mathematics — Parity Roadmap & Convention Notes

**Author:** Olumuyiwa Oluwasanmi

This document is the honest record of where the NimbleCAS financial-mathematics stack
(`bigdecimal`, `finance`, `currency`, `pricing`, `analytics`) stands against the
**superset-of-Excel/Mathematica/MATLAB/Maple** goal: what is implemented, what is not yet,
and where the four reference systems *disagree on convention* so we document divergence
rather than claim false parity. It follows the project honesty invariant (AGENTS.md): a
feature is listed as present only when it ships with a passing test.

## Implemented (shipping, tested)

- **Money & exactness:** `BigDecimal` (base-10 scaled, explicit rounding modes, exact-or-refuse
  division), exact `BigRational` money math, `Money` + FX rate graph, cross rates, triangular
  arbitrage, covered-interest-parity forwards.
- **TVM (exact over ℚ):** PV, FV, PMT, IPMT, PPMT, CUMIPMT, CUMPRINC, NPV, FVSCHEDULE, ISPMT,
  growing annuity / perpetuity / growing perpetuity.
- **TVM (numerical):** NPER, RATE, IRR, XNPV, XIRR, MIRR, NOMINAL, RRI, PDURATION, EFFECT (exact).
- **Depreciation (exact):** SLN, SYD, DDB, VDB; DB (hybrid, Excel 3-decimal rate rounding).
- **Bonds/fixed income:** PRICE, YIELD, DURATION, MDURATION, convexity, accrued interest,
  COUPNUM, PRICEDISC, YIELDDISC, DISC, INTRATE, RECEIVED, TBILLPRICE/YIELD/EQ; day-count layer
  (30/360, 30E/360, ACT/360, ACT/365, ACT/ACT).
- **Swaps:** interest-rate swap value + par rate, currency-swap value (discount-factor based).
- **Options/derivatives:** Black-Scholes + first- and higher-order Greeks (delta/gamma/vega/
  theta/rho + vanna/charm/vomma/veta/speed/zomma/color/lambda/dual-delta/dual-gamma), implied
  vol, Kamrad-Ritchken trinomial (European/American/Bermudan), reproducible Monte Carlo
  (European + arithmetic Asian with geometric control variate), Longstaff-Schwartz American MC,
  Black-76, European digitals (cash-/asset-or-nothing), single-barrier MC, composable Portfolio,
  option P&L, payoff/density/price plotting; Triton GPU MC kernel.
- **Risk/portfolio:** returns, covariance/correlation, Sharpe/Sortino/Treynor/information ratio,
  beta/alpha, max drawdown, historical & Gaussian VaR/CVaR, mean-variance optimization
  (min-variance, tangency, efficient frontier). *Historical VaR/CVaR and the equality-constrained
  frontier are a **better-than-parity** capability — none of the four rivals offer exact VaR.*

## Backlog (not yet implemented — do NOT claim parity for these)

Priorities from the feature-gap analysis. **P0** = required for a genuine superset; P1 strong;
P2 nice-to-have.

### (A) Bonds & fixed income
- **P0** DV01/PV01; duration-from-price; `cfamounts`/`cfdates` (exported dated cashflow schedule
  incl. odd first/last periods, arbitrary frequency); COUPPCD/COUPNCD/COUPDAYBS/COUPDAYS/COUPDAYSNC.
- **P0** ODDFPRICE/ODDFYIELD/ODDLPRICE/ODDLYIELD; PRICEMAT/YIELDMAT; ACCRINTM.
- **P0** Z-spread / static spread; price-off-curve (`prbyzero`); key-rate durations (`bndkrdur`).
- **P1** Yield-convention adapters (street/true, simple, annual↔semiannual); callable/puttable
  bonds + OAS (needs a short-rate lattice); FRN, amortizing schedules.

### (B) Exotic options & pricing methods
- **P0** Finite-difference PDE pricer (Crank-Nicolson + Rannacher, American via penalty/PSOR);
  CRR binomial with discrete dividends (`binprice`); barrier analytic (Reiner-Rubinstein 8-case);
  lookback (Goldman-Sosin-Gatto).
- **P1** Margrabe exchange, Kirk spread, Geske compound (needs bivariate-normal CDF), chooser,
  basket; named SDE processes (GBM/OU-Vasicek/CIR/Heston/Merton jump) with correlated drivers.

### (C) Yield-curve / term-structure toolkit (greenfield — nothing exists yet)
- **P0** Curve object (discount factors + interpolation + compounding conventions);
  `zero2disc`/`disc2zero`/`zero2fwd`/`fwd2zero`; bootstrapping (`zbtprice`/`zbtyield`).
- **P1** Par-yield↔zero; Nelson-Siegel/Svensson/spline fits; short-rate lattice (Hull-White/BDT).

### (D) Risk & portfolio analytics
- **P1** Box/inequality-constrained frontier (active-set QP); lower partial moments; EW moments;
  `corr2cov`/`cov2corr`; correlated return simulation; CVaR-optimal portfolios.

### (E) Remaining Excel scalar functions
- **P1** AMORLINC, AMORDEGRC (French depreciation — coefficient table + rounding quirks).

### (F) Rate/annuity variants
- **P1** Continuous compounding / force-of-interest in EFFECT/NOMINAL (`EffectiveInterest`);
  `amortize` full schedule; `payodd`/`payuni`/`payper`.

### Performance
- **P0** AVX-512 vectorization of the `counter_u64` RNG core — profiling (perf, 2026-07-18) shows
  it at ~73% self-time in the Monte Carlo path engine; the current 23 Mpaths/s (post
  Halley-drop optimization) is RNG-bound.

## Convention divergences to document, not fake

The four systems genuinely disagree here; our contract states which we reproduce.

1. **Day-count basis numbering** differs (Excel basis 0 = 30/360 US; MATLAB basis 0 = ACT/ACT).
   Our `DayCount` enum is **named**, never a raw integer.
2. **ACT/ACT is three conventions** (Excel YEARFRAC basis-1, ISDA, ICMA) and Excel is internally
   inconsistent. We use a rational ACT/ACT and document it.
3. **30/360 variants** disagree on Feb 28/29 endpoints across Excel/OOo/Gnumeric.
4. **IRR root selection:** multiple sign changes ⇒ multiple roots; Excel returns whichever
   Newton-from-guess finds. We bracket-and-Brent and return `not_converged` rather than an
   arbitrary root (an all-roots variant is future work).
5. **XIRR/XNPV** hard-code ACT/365 exponents (Excel); MATLAB allows a basis.
6. **DB** rounds its rate to 3 decimals by spec — we reproduce this deliberately (hybrid tier).
7. **TBILLEQ** is piecewise (>182 days uses a quadratic); we ship the ≤182-day simple form and
   document the divergence for longer bills.
8. **Date epoch:** our serial dates use the astronomical proleptic Gregorian count **without**
   Excel's 1900-leap-year bug — a documented divergence (dates differ by one before 1900-03-01).
9. **Exact vs Excel:** Tier-A exact-over-ℚ results differ from Excel in trailing digits because
   Excel computes in IEEE double. We are **more accurate**, never "Excel-bit-identical".
10. **VaR/quantile:** historical VaR depends on the quantile type (Excel PERCENTILE.INC = type 7
    vs MATLAB ≈ type 5); loss reported positive (MATLAB `portvrisk` convention).
11. **NPV timing:** Excel NPV discounts the first flow one full period; a "NPV at t=0" differs.
    Both `npv` (Excel convention) and the fluent schedule's `net_present_value` (t=0) exist.

## Security & resource bounds (a documented input contract)

Every function that takes untrusted numeric or string input is **DoS-bounded**: a
hostile value returns an honest `Result<T>`/`MathError` (or `std::nullopt`) instead of
provoking an OOM, an unbounded loop, or UB. These bounds are a deliberate contract, not
silent truncation — a rejected input is refused loudly on the railway. Covered by
`tests/security_tests.cpp` (an adversarial-audit regression suite).

| Lever | Bound | Rejected with |
| :--- | :--- | :--- |
| `BigDecimal::from_string` scale/exponent | `\|scale\|`, `\|exponent\|` ≤ 10⁶ | `overflow` (a ~15-byte `1e-2000000000` would otherwise materialize a ~2×10⁹-digit `BigInt`) |
| TVM period count (`fv`/`pv`/`pmt`/`effect`/growing annuities) | `\|nper\|` ≤ 10⁵ | `domain_error`/`overflow` (bounds the `(1+r)^n` exponent; negative `n` is still supported for discounting) |
| `Date::of` year, and `Date::ymd` on a raw serial | year ∈ [1, 9999]; serial walk clamped | `domain_error` (an unbounded year drove a ~2×10⁹-iteration calendar walk) |
| Pricing `steps`/`paths` (trinomial, MC, LSM, barrier, Asian) | `steps` ≤ 10⁵; `paths·steps` product bounded (~10⁹, LSM cells ~5×10⁸) | `domain_error` (prevents `2·steps+1` int overflow and multi-GB grids / multi-hour loops) |
| Depreciation `life` (`db`/`ddb`/`vdb`) | `life` ≤ 10⁵ | `domain_error` |
| Optimizer dimension (`lu_solve_ridge`, `covariance_matrix`) | `n` ≤ 4096; non-finite matrix/rhs rejected | `nullopt`/`domain_error` (an O(n³) solve on unbounded `n`; NaN never flows into silent weights) |

The Cholesky path additionally uses a **relative pivot floor** (`s ≤ 1e-12·|diagᵢ|`) so a
rank-deficient (positive-*semi*-definite) covariance — e.g. two perfectly collinear assets —
is refused honestly rather than yielding unstable garbage weights, while genuinely small-variance
positive-definite matrices still solve.
