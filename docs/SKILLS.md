# NimbleCAS — Skills & Capabilities

**Author:** Olumuyiwa Oluwasanmi

The canonical, version-controlled **skills document for the CAS**: what NimbleCAS can
do, the invariants any new capability must uphold, and how the quantitative-finance
stack fits together. It is the human-and-agent onramp that sits above the per-module
[`docs/Index.md`](Index.md) catalog and the `docs/reference/<module>.md` API pages.
(A mirror for Claude Code tooling lives at `.claude/skills/nimblecas/SKILL.md`, which is
git-ignored and local-only — **this file is the synced source of truth**.)

## What NimbleCAS is

An **exact** Computer Algebra System in C++23: one module per concern (`import` graph,
no headers for internal code), `import std`, `std::expected`-based error handling, no
exceptions. Symbolic algebra & calculus, exact rational/bignum arithmetic, numerics
(linear algebra, ODE/PDE, roots, series), and a large **quantitative-finance** suite.

## The one invariant: honesty (Code Policy Rule 32)

Every fallible operation returns `Result<T> = std::expected<T, MathError>`; **nothing
throws, and nothing returns a plausible-looking wrong value.** If an exact answer isn't
available, return an honest `MathError` (`not_implemented`, `domain_error`, `overflow`,
`not_converged`, `syntax_error`, …) — never a silently-approximate value dressed up as
exact. Each module's reference doc states whether it is *exact over ℚ*, *exact per
term*, or *numerical with a stated tolerance*. Preserve and extend that boundary.

## Conventions any new capability must match

- C++23 modules + `import std`; **trailing return types everywhere**; `[[nodiscard]]`.
- No raw pointers; no exceptions; railway-oriented `Result` / `make_error<T>(...)`.
- **Fluent, composable, reusable** APIs (Rules 15/47): immutable `with_*` builders
  returning a modified copy; `create()` / `from()` factories; chainable façades.
- TBB (Linux/macOS) / PPL (Windows); dynamic-dispatch SIMD waterfall (AVX-512→AVX2→scalar).
- **Internal `nimblecas.testing` framework only** — never GTest/Catch2. Tests assert
  hand-verified exact values or analytic oracles, not vague tolerances.
- Canonical build flags (Rule 50) shared by every module; JSON via the vendored
  `fastjson` (fastestjsoninthewest, Rule 37).

## Quantitative-finance stack

| Module | Concern | Honesty |
| :--- | :--- | :--- |
| `nimblecas.finance` | TVM, cashflow (NPV/IRR/XIRR), depreciation, bonds, swaps, T-bills; fluent `CashflowSchedule`/`TvmProblem` | Tier A exact over ℚ / Tier B numerical |
| `nimblecas.fixedincome` | DV01, durations, odd-period pricing, Z-spread, FRN, amortization | numerical |
| `nimblecas.yieldcurve` | pillar `Curve`, bootstrapping, Nelson-Siegel/Svensson, Hull-White lattice, callable-bond OAS | numerical |
| `nimblecas.currency` | exact FX `Money`, cross-rate graph, triangular arbitrage, covered-interest forwards | exact over ℚ |
| `nimblecas.pricing` | Black-Scholes + full/extended Greeks, trinomial trees, reproducible Monte Carlo, Longstaff-Schwartz, Black-76, digitals, barriers, composable `Portfolio` | numerical/statistical |
| `nimblecas.exotics` | CRR binomial, analytic barriers, lookback, Crank-Nicolson PDE, Margrabe/Kirk/basket | numerical |
| `nimblecas.analytics` / `nimblecas.portfolio` / `nimblecas.riskextra` | Sharpe/Sortino/VaR/CVaR, Markowitz optimization, CVaR-optimal weights | numerical |
| `nimblecas.marketdata` | **provider-agnostic quote ingestion** — normalised `Quote`/`Bar`/`OptionChain`/`RateQuote` + Yahoo/Alpha Vantage/Alpaca adapters via `fastjson`; fluent `Feed` | data boundary (no fabricated values) |
| `nimblecas.futures` | cost-of-carry forward/futures valuation (basis, convenience yield, implied carry, MtM) + composable `FuturesStrategy` (outright, calendar/inter-commodity spreads, hedges, cash-and-carry) | numerical |
| `nimblecas.optstrat` | composable option strategies (covered call, spreads, straddle/strangle, strip/strap, butterfly, condor, collar, box, risk reversal, ratio) + exact piecewise-linear expiry-P&L analytics (net premium, max profit/loss, breakevens) + BS Greeks bridge | exact expiry P&L / numerical Greeks |
| `nimblecas.fxstrat` | Garman-Kohlhagen FX options, covered-interest-parity forwards/points, covered-interest-arbitrage, fluent `CarryTrade` (breakeven == CIP forward) | numerical |
| `nimblecas.bondstrat` | portfolio duration/convexity/yield, barbell/butterfly weights, duration-neutral hedge ratio, carry/roll + duration-convexity P&L | exact weighting algebra |
| `nimblecas.mmstrat` | repo interest, discount price↔rate, bond-equivalent yield, holding-period return, deposit-strip effective rate | exact day-count identities |

**Market-data usage** (feeds every valuation engine above):

```cpp
import nimblecas.marketdata;
using namespace nimblecas::marketdata;
const auto feed = Feed::from(DataProvider::yahoo).as_asset_class(AssetClass::equity);
const auto q = feed.quote(yahoo_chart_json).value();   // q.last is the pricing spot
// -> pricing::OptionSpec{}.with_spot(q.last).with_strike(...)...
```

Adding a provider = one `DataProvider` enumerator + one adapter function; no downstream
change. Tests drive adapters from **embedded JSON fixtures** — deterministic, no network.

## Roadmap (this workstream)

1. ~~**Futures/forward valuation** — cost-of-carry, basis, convenience yield, roll, MtM.~~ ✅ `nimblecas.futures` shipped.
2. ~~**Trading strategies across five asset classes** — options, futures, FX, bonds, money
   markets — each with strategy analytics.~~ ✅ `optstrat`, `futures` strategies, `fxstrat`,
   `bondstrat`, `mmstrat` shipped.
3. **GPU acceleration** of compute-heavy paths (Monte Carlo pricing, payoff-grid /
   strategy sweeps, batch valuation) in the `nimblecas.gpu` pattern: **Triton kernel,
   CUDA, CUDA Graphs, and CuTile** variants (opt-in `-DNIMBLECAS_CUDA=ON`). The CPU path
   stays authoritative and bit-reproducible; GPU is a performance mirror validated
   against it.
4. **Profiling-gated optimization** (Rules 43/58/59): no speedup claim without
   evidence — **perf / VTune** for CPU hot paths, **nsys / ncu** for GPU kernels,
   profiled on the servers.

Keep this table and the [`Index.md`](Index.md) catalog in sync as each module lands.

## Build, test & review (this repo does NOT build on Windows)

- Build/test on the Linux server (`clang++-22` + libc++): edit on Windows → commit →
  `git push mgpu <branch>` → `ssh mgpu 'cd /scratch/NimbleCAS && ninja -C build <targets> && ctest --test-dir build'`.
- After every major unit: an **adversarial code review** (hunt for bugs/UB/races/policy
  violations), then commit + push to all remotes (github + gitea + mgpu) and the NAS
  backup. **No stragglers, no stale branches/worktrees.**
- **Authorship (mandatory):** every commit attributed solely to Olumuyiwa Oluwasanmi;
  never add AI co-author trailers or `@author` AI attributions.

## See also

- [`docs/Index.md`](Index.md) — module catalog + dependency graph.
- [`AGENTS.md`](../AGENTS.md), `config/cpp_details.txt`, `config/update_policy.txt` — the policy.
