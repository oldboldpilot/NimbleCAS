# `nimblecas.smc` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/smc/smc.cppm` · namespace `nimblecas::smc`

Sequential Monte Carlo — bootstrap **particle filters**, resampling schemes, and
variance-reduction estimators (ROADMAP §7.8, "deep stochastic-process simulation
beyond the sampling / MCMC primitives").

## This module is numerical, and honest about it

Outputs are **statistical estimates on `double`, never exact**. The honesty
contract here is *not* exactness but:

1. **Determinism / reproducibility.** Every draw is a pure function of the
   counter-based Threefry RNG (`Rng::seeded` / `Rng::split`, `splitmix64`,
   `counter_u64`) — no shared mutable state. The same seed yields bit-identical
   results regardless of thread count or partitioning, exactly like the other
   parallel modules.
2. **Documented statistical properties.** Plain / antithetic / stratified integral
   estimators are **unbiased**; control-variate is **consistent** (`O(1/n)` bias
   from estimating `c* = Cov/Var`); the filter's marginal-likelihood estimate is
   unbiased and the filtered means are consistent with finite-`N` Monte Carlo
   variance plus resampling bias. `std_error` advertises remaining noise, not
   exactness.

## Public API (namespace `nimblecas::smc`)

**Resampling** — each returns `Result<std::vector<std::size_t>>` of exactly `N`
parent indices, driven by the counter-based RNG (deterministic given a seed):
- `multinomial_resample`, `systematic_resample`, `stratified_resample`,
  `residual_resample(weights, seed)`.
- `effective_sample_size(weights) -> Result<double>` = `(Σw)²/Σw²` (`= 1/Σwᵢ²`
  when normalized), in `[1, N]`.

**Particle filter:**
- `enum class ResampleScheme { multinomial, systematic, stratified, residual }`.
- callbacks `StateSampler = fn(Rng&)`, `TransitionSampler = fn(double, Rng&)`,
  `LogWeight = fn(double, double)`.
- `struct BootstrapFilterResult { filtered_means; effective_sample_sizes; log_marginal_likelihood; }`.
- `bootstrap_particle_filter(initial_sampler, propagate, log_weight, observations,
  particles, seed, ess_threshold = 0.5, scheme = systematic)` — SIR with adaptive
  resampling (resample only when `ESS < ess_threshold·N`; weights carried between
  steps in log-space via log-sum-exp so the marginal-likelihood increment is
  numerically stable).

**Variance reduction** — each returns `Result<Estimate{value, std_error}>`:
- `plain_estimate` (baseline), `antithetic_estimate`,
  `control_variate_estimate(f, control, control_mean, …)`, `stratified_estimate`.

## Determinism / seeding scheme

Per-`(step, particle)` streams are derived by two-level `Rng::split` so each draw
depends only on `(seed, step, particle)` and cannot collide across pairs: `root =
Rng::seeded(seed)`; init uses `root.split(0).split(j)`; step `t` uses
`root.split(t+1).split(0)` for propagation (then `.split(j)` per particle) and
`root.split(t+1).split(1)` for resampling. Integration estimators index the
counter core by sample position, so they are partition-independent like
`montecarlo::integrate`.

## Error model

Guards return the documented `MathError` — never a garbage number:
- empty / `N = 0` / all-zero / negative weights, `b < a` domain → `domain_error`;
- NaN / Inf weights, NaN / `+Inf` log-weights → `undefined_value`.

## Verified in the tests

- Resampling preserves count `N` and returns only valid indices (all four schemes);
  one-hot weights make every scheme pick the survivor.
- `effective_sample_size`: uniform → `N`; one-hot → `1`.
- Systematic / stratified resampling are bit-identical across calls with the same
  seed (deterministic, thread-count independent).
- Bootstrap filter on a **linear-Gaussian** model tracks the exact **Kalman** mean
  (`N = 40000`, fixed seed) with `ESS ∈ [1, N]`, and is reproducible.
- Antithetic / control-variate / stratified estimators are unbiased on known
  integrals with a `std_error` strictly below plain Monte Carlo at equal sample count.

## See also

- [`montecarlo.md`](montecarlo.md) · [`mcmc.md`](mcmc.md) — the sampling primitives this builds on.
- [`qmc.md`](qmc.md) — low-discrepancy alternatives for integration.
