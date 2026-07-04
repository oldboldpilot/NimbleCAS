# `nimblecas.mcmc` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/mcmc/mcmc.cppm`

Deterministic random-walk **Metropolis-Hastings** sampling (ROADMAP §7.8). This
is a **numerical / Monte Carlo** layer, not an exact one: it draws
floating-point samples from a target distribution supplied as an *unnormalised*
`log_density`, so every result is a stochastic estimate, never an exact rational
answer. It sits above the parallelisable counter-based RNG substrate
(`nimblecas.rng`) and consumes only its `Rng::seeded` / `next_unit` /
`splitmix64` surface. Working in **log space** avoids underflow for sharply
peaked targets, and because the unknown normalising constant cancels in the
acceptance ratio only *differences* of log-densities are ever needed. The honest
boundary is this: there is **no time or entropy seeding anywhere** — a chain is a
pure function of its seed and is bit-reproducible — and all failure travels the
`Result<T>` / `MathError` railway, never an exception. Statistical accuracy is
the caller's responsibility (adequate `burn_in`, `samples`, and `step`); the
module guarantees determinism and bookkeeping, not mixing.

```cpp
import nimblecas.mcmc;
```

Depends on [`core`](core.md) and `nimblecas.rng` (the seeded counter-based RNG
substrate — `Rng::seeded`, `next_unit`, `splitmix64`).

## The determinism and edge-case contract

Each chain is driven by a single `Rng::seeded(seed)` that supplies exactly **two
draws per step** — the symmetric proposal `U`, then the acceptance test `V`, both
`~ Uniform[0, 1)`. Equal seeds therefore reproduce a **bit-identical** chain and
identical accept/propose counts. The proposal is symmetric — `x' = x + step ·
(2U − 1)` — so its density cancels and the plain **Metropolis** (not full
Hastings) ratio applies:

```
accept  <=>  log(V) <= log_density(x') − log_density(x)
```

The IEEE edge cases around `−inf` support are decided **explicitly** rather than
left to the raw `log(V) <= Δ` compare, because `next_unit()` can return exactly
`0.0` (giving `log(V) = −inf`), which would otherwise let `−inf <= −inf` wrongly
accept a proposal into a zero-support region. Writing `Δ = log_density(x') −
log_density(x)`, the branch table is:

| Δ (`proposal_log − current_log`) | Meaning | Decision |
| :--- | :--- | :--- |
| `NaN` (only from `−inf − −inf`, both densities `−inf`) | stuck outside the support, proposal no better | **reject**, hold place |
| `Δ >= 0` (uphill or equal; includes current `−inf` and finite proposal → `Δ = +inf`) | proposal at least as likely | **always accept** — lets a chain started outside the support walk back in |
| `Δ == −inf` (finite point, proposal into zero support) | proposal has zero density | **always reject** |
| finite negative | downhill | accept **iff** `log(V) <= Δ` |

Because the loop runs a fixed `burn_in + samples` iterations the routine **always
terminates**.

## `McmcResult` — the output of a single chain

The retained samples plus the accept/propose bookkeeping. The empirical
acceptance rate is `accepted / proposed`, counted over **all** iterations
(burn-in included), while `chain` holds only the post-burn-in draws.

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `chain` | `std::vector<double>` | The `samples` retained draws (post burn-in), in order. |
| `accepted` | `std::uint64_t` | Proposals accepted over all `burn_in + samples` steps. |
| `proposed` | `std::uint64_t` | Total proposals attempted, `== burn_in + samples`. |

## Sampling

```cpp
[[nodiscard]] auto metropolis_hastings(std::function<double(double)> log_density, double x0,
                                       double step, std::uint64_t samples,
                                       std::uint64_t burn_in, std::uint64_t seed)
    -> Result<McmcResult>;

[[nodiscard]] auto run_parallel_chains(std::function<double(double)> log_density, double x0,
                                       double step, std::uint64_t samples_per_chain,
                                       std::uint64_t burn_in, std::uint64_t seed,
                                       std::uint64_t chains) -> Result<std::vector<McmcResult>>;
```

| Function | Behavior |
| :--- | :--- |
| `metropolis_hastings` | Run one random-walk Metropolis chain against the unnormalised `log_density`, starting at `x0` with symmetric uniform proposals of half-width `step`. Runs exactly `burn_in + samples` iterations, discards the first `burn_in`, and returns the final `samples` states in `chain`; `accepted` / `proposed` count every iteration. Fails `domain_error` if `step <= 0` or `samples == 0`, and `overflow` if `burn_in + samples` would wrap `uint64`. |
| `run_parallel_chains` | Launch `chains` **independent** chains against the same target with the same `x0` / `step` / `samples_per_chain` / `burn_in`, each with its own reproducible seed. Chain `c` is seeded with `splitmix64(seed ^ c)`. Runs the chains serially and returns one `McmcResult` per chain. Fails `domain_error` if `chains == 0`, and propagates the per-chain error if the shared arguments are invalid (`domain_error` for `step <= 0` / `samples_per_chain == 0`, `overflow` if `burn_in + samples_per_chain` wraps). |

**Parallelisation contract.** In `run_parallel_chains`, chain `c`'s seed is
`splitmix64(seed ^ c)`. SplitMix64 is a bijective bit-mixer, so distinct `c`
yield distinct, well-separated seeds and hence mutually independent streams.
Because a chain's seed depends only on `(seed, c)` — **not** on the total number
of chains nor on the order they run — the ensemble is fully reproducible and each
chain is identical no matter how many others are launched. The chains are summed
serially here, but this per-chain-index seeding is exactly what lets a parallel
executor hand each chain to its own worker with no coordination.

## Chain statistics

```cpp
[[nodiscard]] auto chain_mean(std::span<const double> chain) -> Result<double>;
[[nodiscard]] auto chain_variance(std::span<const double> chain) -> Result<double>;
```

| Function | Behavior |
| :--- | :--- |
| `chain_mean` | Arithmetic mean of a chain. Self-contained (no `montecarlo` dependency). Fails `domain_error` on an empty span. |
| `chain_variance` | **Population** variance — sum of squared deviations divided by `n`, **not** `n − 1`. Two-pass (mean first, then squared deviations) for stability. A single point is allowed (variance `0`). Fails `domain_error` on an empty span. |

## Error model

| Condition | Error |
| :--- | :--- |
| `metropolis_hastings` with `step <= 0` or `samples == 0` | `MathError::domain_error` |
| `metropolis_hastings` where `burn_in + samples` overflows `uint64` | `MathError::overflow` |
| `run_parallel_chains` with `chains == 0` | `MathError::domain_error` |
| `run_parallel_chains` with an invalid shared `step` / `samples_per_chain` | propagated `MathError::domain_error` from the per-chain run |
| `run_parallel_chains` where the shared `burn_in + samples_per_chain` overflows `uint64` | propagated `MathError::overflow` from the per-chain run |
| `chain_mean` / `chain_variance` on an empty span | `MathError::domain_error` |

There is no `division_by_zero` or `undefined_value` path: the acceptance test
never divides, and the `−inf` / `NaN` log-density cases are resolved by the
explicit branch table above rather than surfacing as errors. A degenerate target
(e.g. every point outside the support) produces a valid `McmcResult` with
`accepted == 0`, not an error.

## Worked examples

```cpp
import std;
import nimblecas.core;
import nimblecas.mcmc;
using namespace nimblecas;

// Unnormalised log-density of the standard normal: log N(x; 0, 1) = −x²/2 + const.
auto standard_normal_log = [](double x) { return -0.5 * x * x; };

// Unnormalised log-density of Uniform[0, 1]: 0 on the support, −inf outside.
auto uniform01_log = [](double x) {
    if (x < 0.0 || x > 1.0) return -std::numeric_limits<double>::infinity();
    return 0.0;
};

// Sample the standard normal: mean ≈ 0, variance ≈ 1 (Monte Carlo estimates, not exact).
auto r = metropolis_hastings(standard_normal_log, 0.0, 2.0, 200000, 5000, 2024).value();
r.chain.size();                       // 200000  (retains exactly `samples` draws)
chain_mean(r.chain).value();          // ≈ 0
chain_variance(r.chain).value();      // ≈ 1  (population variance)

// Bookkeeping is exact: proposed == burn_in + samples, accepted <= proposed.
r.proposed;                           // 205000 == 5000 + 200000

// Equal seeds reproduce a bit-identical chain (no time/entropy seeding).
auto a = metropolis_hastings(standard_normal_log, 0.0, 2.0, 5000, 1000, 99).value();
auto b = metropolis_hastings(standard_normal_log, 0.0, 2.0, 5000, 1000, 99).value();
// a.chain == b.chain, a.accepted == b.accepted, a.proposed == b.proposed

// A bounded target: every retained sample lies inside the support.
auto u = metropolis_hastings(uniform01_log, 0.5, 0.5, 200000, 5000, 7).value();
chain_mean(u.chain).value();          // ≈ 0.5, and every u.chain[i] in [0, 1]

// Start OUTSIDE the support: a finite proposal from an −inf point has Δ = +inf and is
// always accepted, so the chain walks in and then stays (Δ = −inf back out is rejected).
auto w = metropolis_hastings(uniform01_log, 1.3, 0.5, 20000, 2000, 555).value();
w.accepted;                           // > 0  (at least the walk-in move)

// Stuck far outside with too small a step: every point is −inf, Δ = NaN, nothing accepts.
auto stuck = metropolis_hastings(uniform01_log, 10.0, 0.5, 500, 0, 3).value();
stuck.accepted;                       // 0  (chain holds exactly at 10.0)

// Independent, reproducible ensemble: chain c seeded splitmix64(seed ^ c).
auto ens = run_parallel_chains(standard_normal_log, 0.0, 2.0, 5000, 1000, 42, 4).value();
ens.size();                           // 4, distinct chains, reproducible across reruns

// Error paths (railway, never exceptions).
metropolis_hastings(standard_normal_log, 0.0, 0.0, 100, 10, 1).error();   // domain_error (step <= 0)
metropolis_hastings(standard_normal_log, 0.0, 1.0, 0, 10, 1).error();     // domain_error (samples == 0)
run_parallel_chains(standard_normal_log, 0.0, 1.0, 100, 10, 1, 0).error();// domain_error (chains == 0)
std::vector<double> empty;
chain_mean(empty).error();            // domain_error
chain_variance(empty).error();        // domain_error
```

## See also

- [`nimblecas.stats`](stats.md) — the sibling statistical / numerical layer.
- [`nimblecas.core`](core.md) — the `Result<T>` / `MathError` railway every
  return type rides on.
- [Documentation hub](../Index.md)
