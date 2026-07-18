# Session log — Financial mathematics suite: security hardening, GPU/Python verification, docs

**Date:** 2026-07-18
**Author:** Olumuyiwa Oluwasanmi
**Build/verify host:** `oluwasanmi-tradingbot-server` (clang++-22 + libc++; RTX 5090, CUDA 13.2)

## Scope

Continuation of the financial-mathematics work (six modules: `bigdecimal`, `finance`,
`currency`, `pricing`, `analytics`, `portfolio`). This session covered: an adversarial
security audit with fixes, a comprehensive runnable showcase, an adversarial review of the
fixes, verification of the CUDA + Triton GPU paths and the nanobind Python bindings, and a
documentation pass.

## What was done

### 1. Adversarial security audit + fixes (`tests/security_tests.cpp`)
A Fable adversarial audit found 7 findings (untrusted numeric/string input → OOM / hang / UB /
silent-wrong). All fixed with regression tests, then a second Fable review of the fix diff found
7 more (D1–D7), also fixed:

- **F1** `BigDecimal::from_string` caps `|scale|` and `|exponent|` at 10⁶ (D2: exponent checked
  before the `frac_digits - exponent` subtraction to avoid INT64_MIN signed-overflow UB).
- **F2** `growth`/`effect`/`growing_annuity_pv` bound `|nper|` at 10⁵ (D4: magnitude bound, not
  sign — negative `nper` discounting is preserved; D6: consistent `domain_error`).
- **F3** `Date::of` bounds the year to 1..9999; `Date::ymd` clamps a hostile raw serial.
- **F4** trinomial caps `steps` (int-overflow of `2·steps+1`); LSM bounds `paths·(steps+1)`;
  MC loops bound `paths` (D1: barrier/Asian bound the `paths·steps` **product**, not each factor).
- **F5** `lu_solve_ridge` caps dimension (4096) and rejects a non-finite matrix **and** rhs (D3);
  `analytics::cholesky` uses a relative pivot floor so a rank-deficient (semidefinite) covariance
  is refused honestly instead of yielding unstable garbage weights.
- **F7** `db`/`ddb`/`vdb` cap `life`.

### 2. Comprehensive showcase (`examples/finance_showcase.cpp`)
A 14-section runnable tour of the whole stack with hand-computed expecteds. Builds clean, runs to
exit 0. Textbook values confirmed (BS ATM call 10.4506, delta 0.6368, put-call parity, exact
CUMPRINC −300000, mortgage PMT −1798.65, singular-cov Cholesky refusal). D7: intentional-failure
demos routed through an `err_of()` helper that guards `.error()`-on-success UB.

### 3. GPU + Python verification (RTX 5090)
- **CUDA kernels** (`nvcc` 13.2, `-arch=native` → sm_120): `gpu_tests` — 12 groups / 41 checks pass
  (poly-eval, edit-distance, BFS, N-queens, QMC, Haar DWT, batched matmul, batched FFT).
- **nsys** traced all 9 kernels; pipeline is transfer-bound. **ncu** blocked by `ERR_NVGPUCTRPERM`
  (GPU perf counters need root / a driver flag) — documented, `nsys` needs no such permission.
- **Python bindings** (nanobind 2.13): `nimblecas_ext` builds, `test_bindings.py` passes; the
  finance/pricing/analytics submodules **reuse the C++ engine** — verified numerically through
  Python (BS 10.4506, IRR 0.10, min-var [0.6923, 0.3077]).
- **Triton** (`python/triton/mc_option.py`, torch 2.13 / triton 3.7.1): 8M-path GPU MC prices the
  European call at 10.45238 ± 0.00260 — 0.69 std errors from Black-Scholes. Reuses the same MC
  model as `pricing::monte_carlo_european`.

### 4. Documentation
- `ROADMAP.md`: added the Financial mathematics module family (was the only domain missing).
- `finance-parity-roadmap.md`: documented the DoS/resource-bounds contract.
- `python-bindings.md`: documented the finance submodules (reuse emphasis) with verified values.
- `gpu.md`: full API (8 batch kernels), real 12-group test suite, Triton finance MC, nsys/ncu notes.
- `financial-mathematics.md`: GPU acceleration & Python bindings section.

## Verification
All finance-stack suites + `security_tests` + `gpu_tests` + `python_bindings` pass on clang++-22 /
RTX 5090. `examples/finance_showcase` runs to completion.

## Review process
Per [[nimblecas-code-review-policy]] ("2 of 3 acceptable"): Fable served as the reliable
adversarial reviewer for both the audit and the fix diff. agy needs interactive login; the claude
review CLI returned empty twice earlier this session (known-flaky).

## Deferred
- SIMD/AVX-512 vectorization of the counter-RNG MC hotspot (roadmap perf backlog).
- ncu deep-counter GPU profiling (needs elevated GPU-counter permission on the host).

---

## Addendum — capability backlog completion + AVX-512 RNG (same session)

### New modules (all Result-based, honesty-preserving, DoS-bounded; 140/140 on clang++-22)
- **`exotics`** — CRR binomial (+escrowed discrete divs), Reiner-Rubinstein analytic barriers
  (8 types + rebate), Goldman-Sosin-Gatto lookback, Crank-Nicolson/Rannacher PDE (Thomas/PSOR),
  Margrabe/Kirk/Geske/chooser/basket via a Drezner bivariate-normal CDF.
- **`yieldcurve`** — pillar Curve (interp/compounding), zero2disc/disc2zero/zero2fwd/fwd2zero,
  par<->zero, zbtprice/zbtyield bootstrap, Nelson-Siegel/Svensson, Hull-White trinomial lattice.
- **`fixedincome`** — DV01/PV01, key-rate durations, cfdates/cfamounts, COUP* dates, ODDF/ODDL,
  PRICEMAT/YIELDMAT/ACCRINTM, z-spread, rate adapters, FRN + amortization.
- **`riskextra`** — corr2cov/cov2corr, LPM, EW moments, correlated sim, active-set QP frontier,
  Rockafellar-Uryasev CVaR (self-contained two-phase simplex), AMORLINC/AMORDEGRC, continuous
  rates/annuities.

### Performance backlog item (done)
- AVX-512F batched Threefry-2x64-20 core `counter_u64_batch` (runtime-dispatched, bit-identical
  to scalar), `Rng::fill_u64`, and the four pricing MC engines rewired to draw through it.
  ~8.2x on the RNG core (72 -> 592 M draws/s, Zen 5); pricing values unchanged (reproducibility
  test still passes). Benchmark: `examples/rng_bench.cpp`.

### Documentation
- Four reference pages (`docs/reference/{exotics,yieldcurve,fixedincome,riskextra}.md`) + Index rows.
- `finance-parity-roadmap.md`: backlog A-F + RNG moved to Implemented; honest P1 remainder
  (callable/OAS, named-SDE factories, spline curve fit) — explicitly not claimed (no test yet).

### Review process (two adversarial rounds, Fable)
- Round 1 (new code): RNG SIMD core + MC batching verified bit-identical/clean; 10 real defects
  + 1 nit found in exotics/pricing/riskextra/yieldcurve/fixedincome — 5 NaN-through-Result
  honesty leaks, 1 float->int64 UB, 1 batching regression (O(steps) buffer), 3 DoS bounds. All
  fixed with regression tests (commit 4250baf).
- Round 2 (fix diff): found 2 of my own fixes wrong — the amorlinc clamp reordered a NaN check
  out of existence (reintroduced the UB), and kMaxLpCells=1e6 over-rejected in-cap CVaR. Both
  fixed + regression-tested (commit c08c8e0).

## Deferred (updated)
- Callable/puttable bond OAS, named-SDE process factories, spline zero-curve fit (P1; building
  blocks now exist — see finance-parity-roadmap "Remaining").
- ncu deep-counter GPU profiling (needs elevated GPU-counter permission on the host).
