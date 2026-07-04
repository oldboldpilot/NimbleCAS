# `nimblecas.montecarlo` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/montecarlo/montecarlo.cppm`

Deterministic (seeded) Monte Carlo integration, sampling, and the sample
statistics helpers (ROADMAP §7.8), built on the parallelisable RNG substrate
[`nimblecas.rng`](rng.md). This layer sits **above** the exact rational core:
unlike the Gaussian-rational [`complex`](complex.md) or the `Q`-exact
[`stats`](stats.md) module, **every result here is NUMERICAL** — an `f64`
estimate carrying Monte Carlo sampling error, not an exact rational. The honesty
boundary is therefore statistical rather than algebraic: `integrate` and
`estimate_pi` converge to their true values only as `samples → ∞`, and the tests
assert them only within generous tolerances.

What *is* exact and bit-defined is **reproducibility**. There is no time or
entropy seeding anywhere: equal seeds reproduce equal results, bit-for-bit. The
two integration estimators are expressed over the **stateless** counter core
`counter_u64(key, i)` — because sample `i` is a pure function of its index, any
partition of the range `0..samples-1` across workers, concatenated in index
order, reproduces the single-threaded estimate exactly (**partition-independent
by construction**; summed serially here, but the design admits parallel
summation). `rejection_sample` instead consumes a **stateful** `Rng::seeded`
trial stream and is hard-capped at `max_trials` proposals so it can never loop
forever. All failure is reported on the railway (`Result<T>` / `MathError`),
never by throwing.

```cpp
import nimblecas.montecarlo;
```

Depends on [`core`](core.md) and [`rng`](rng.md) (`splitmix64`, `counter_u64`,
`uniform_unit`, `Rng::seeded` / `next_unit`).

## Integration & sampling estimators (free functions)

All estimators are free functions in namespace `nimblecas`; there is no class to
construct. Each takes an explicit `seed`, so the caller — not a hidden global —
owns the stream.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `integrate` | `[[nodiscard]] auto integrate(std::function<double(double)> f, double a, double b, std::uint64_t samples, std::uint64_t seed) -> Result<double>` | Monte Carlo estimate of `∫_a^b f(x) dx = (b − a)·mean(f(U))`, `U ~ Uniform[a, b]`. A key is derived from `seed` via `splitmix64`; sample `i` is `x_i = a + uniform_unit(counter_u64(key, i))·(b − a)`, a pure function of `i` — hence **partition-independent**. `domain_error` if `b < a` or `samples == 0` (`b == a` is allowed and integrates to `0`). |
| `estimate_pi` | `[[nodiscard]] auto estimate_pi(std::uint64_t samples, std::uint64_t seed) -> Result<double>` | Classic dart estimate of π: per sample draw `(x, y)` in `[0, 1)²` from two **decorrelated** counter draws — `counter_u64(key, 2i)` and `counter_u64(key, 2i+1)` — count darts with `x² + y² < 1`, return `4·hits / samples`. Each sample is a pure function of its index, so the estimate is partition-independent. `domain_error` if `samples == 0`. |
| `rejection_sample` | `[[nodiscard]] auto rejection_sample(std::function<double(double)> pdf, double a, double b, double m_bound, std::uint64_t want, std::uint64_t seed, std::uint64_t max_trials) -> Result<std::vector<double>>` | Draw up to `want` samples on `[a, b]` with density ∝ `pdf` by rejection against a uniform proposal with ceiling `m_bound`: a candidate `x ~ Uniform[a, b]` is accepted when `U·m_bound ≤ pdf(x)`, `U ~ Uniform[0, 1)`. `pdf` need **not** be normalised — only its shape and the ceiling matter. A stateful `Rng::seeded(seed)` supplies the trial stream. `domain_error` if `b < a`, `m_bound ≤ 0`, `want == 0`, or `max_trials == 0`. |

### `rejection_sample` termination guarantee

The routine issues **at most `max_trials` proposals** and stops as soon as either
`want` samples are accepted or the trial budget is exhausted, returning however
many were accepted — possibly fewer than `want`, possibly empty. It therefore
**always terminates** and can never spin forever, even for a pathological
`pdf`/`m_bound`. The returned vector holds only accepted `x` values, each in
`[a, b]`. The result is reserved to `min(want, max_trials)`, never the raw
`want` — an un-clamped huge `want` cannot make `reserve()` throw and escape the
no-throw railway contract.

## Sample statistics helpers (free functions)

These operate on caller-supplied data (`std::span<const double>`) and carry no
RNG; they are the numerical-`f64` counterparts of the exact-`Q` estimators in
[`stats`](stats.md).

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `sample_mean` | `[[nodiscard]] auto sample_mean(std::span<const double> data) -> Result<double>` | Arithmetic mean of the sample. `domain_error` on an empty span. |
| `sample_variance` | `[[nodiscard]] auto sample_variance(std::span<const double> data, bool sample = true) -> Result<double>` | Variance via a numerically stable two-pass computation (mean, then summed squared deviations). With `sample == true` the **unbiased** estimator with Bessel's `(n − 1)` correction is used and at least **2** points are required; with `sample == false` the **population** variance (divide by `n`) is returned and a single point is allowed. An empty span is always a `domain_error`. |

## Error model

| Condition | Error |
| :--- | :--- |
| `integrate` with `b < a`, or `samples == 0` | `MathError::domain_error` |
| `estimate_pi` with `samples == 0` | `MathError::domain_error` |
| `rejection_sample` with `b < a`, `m_bound ≤ 0`, `want == 0`, or `max_trials == 0` | `MathError::domain_error` |
| `sample_mean` on an empty span | `MathError::domain_error` |
| `sample_variance` on an empty span, or `sample == true` with fewer than 2 points | `MathError::domain_error` |

Every fallible path returns `Result<T>`; nothing throws. `integrate` with
`b == a` succeeds and returns `0`; `sample_variance(single, false)` succeeds and
returns `0`. There is no `overflow` path — the arithmetic is `f64`, and a
budget-exhausted `rejection_sample` is **success** (a shorter or empty vector),
not an error.

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.rng;
import nimblecas.montecarlo;
using namespace nimblecas;

// Monte Carlo integration: ∫_0^1 x² dx = 1/3 (numerical, within sampling error).
integrate([](double x) { return x * x; }, 0.0, 1.0, 200000, 20260703).value();  // ≈ 0.333

// A constant integrand recovers the width exactly regardless of the draws:
// ∫_2^5 1 dx = 3.
integrate([](double) { return 1.0; }, 2.0, 5.0, 100000, 7).value();            // == 3.0

// Bit-reproducible: identical (f, a, b, samples, seed) give identical results.
auto f = [](double x) { return x * x * x; };
integrate(f, -1.0, 2.0, 50000, 99).value()
    == integrate(f, -1.0, 2.0, 50000, 99).value();                             // true

// Domain guards.
integrate([](double x) { return x; }, 1.0, 0.0, 1000, 1).error();  // domain_error (b < a)
integrate([](double x) { return x; }, 0.0, 1.0, 0, 1).error();     // domain_error (samples == 0)

// The dart estimate of π.
estimate_pi(200000, 12345).value();   // ≈ 3.14159
estimate_pi(0, 1).error();            // MathError::domain_error

// Rejection sampling from the unnormalised triangular density pdf(x) = x on
// [0, 1] with ceiling m = 1; accepted samples lie in [0, 1] and mean ≈ 2/3.
auto pdf = [](double x) { return x; };
auto draws = rejection_sample(pdf, 0.0, 1.0, 1.0, 5000, 2024, 10000000).value();
sample_mean(draws).value();           // ≈ 0.667

// Termination under a tiny trial budget: never hangs, returns ≤ max_trials samples
// even when `want` is huge.
rejection_sample(pdf, 0.0, 1.0, 1.0, 1000000, 5, 8).value().size();  // ≤ 8

// Domain guards for rejection sampling.
rejection_sample(pdf, 0.0, 1.0, 0.0, 10, 1, 1000).error();  // domain_error (m_bound <= 0)
rejection_sample(pdf, 1.0, 0.0, 1.0, 10, 1, 1000).error();  // domain_error (b < a)
rejection_sample(pdf, 0.0, 1.0, 1.0, 0, 1, 1000).error();   // domain_error (want == 0)
rejection_sample(pdf, 0.0, 1.0, 1.0, 10, 1, 0).error();     // domain_error (max_trials == 0)

// Sample statistics: the classic {2,4,4,4,5,5,7,9} set, mean 5, SSQ 32.
const std::array<double, 8> data{2, 4, 4, 4, 5, 5, 7, 9};
std::span<const double> s{data};
sample_mean(s).value();               // 5
sample_variance(s, true).value();     // 32/7  (Bessel-corrected sample variance)
sample_variance(s, false).value();    // 4     (population variance = 32/8)

// A single point has no unbiased variance, but a well-defined population variance of 0.
const std::array<double, 1> one{42.0};
std::span<const double> single{one};
sample_variance(single, true).error();  // domain_error (n < 2)
sample_variance(single, false).value(); // 0

// Empty-span guards.
std::span<const double> empty{};
sample_mean(empty).error();             // MathError::domain_error
sample_variance(empty, true).error();   // MathError::domain_error
```

## See also

- [`nimblecas.rng`](rng.md) — the counter-based RNG substrate (`counter_u64`,
  `uniform_unit`, `Rng`) these estimators are seeded from.
- [`nimblecas.mcmc`](mcmc.md) — the Markov-chain sibling built on the same RNG.
- [`nimblecas.stats`](stats.md) — the exact-over-`Q` descriptive statistics this
  module mirrors numerically over `f64`.
- [`nimblecas.numeric`](numeric.md) — the numerical/`f64` neighbours in the
  tower.
- [`nimblecas.integrate`](integrate.md) — exact symbolic integration over `Q(x)`,
  the algebraic counterpart to Monte Carlo `integrate`.
- [Documentation hub](../Index.md)
