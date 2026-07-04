# `nimblecas.sde` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/sde/sde.cppm`

Deterministic (seeded), parallelisable integrators for scalar Itô stochastic
differential equations `dX = a(X) dt + b(X) dW_t`, `X(0) = x0`, on `[0, T]` with
`steps` uniform steps `dt = T / steps` and standard-normal Wiener increments
`dW = sqrt(dt) · Z`, `Z ~ N(0, 1)`. Five single-path schemes are provided
(Euler-Maruyama, Milstein, stochastic Heun, derivative-free SRK, tamed Euler),
plus ensemble drivers that average their terminal values.

The honest boundary is this: **these are NUMERICAL IEEE-754 double-precision
approximations, NOT exact symbolic results.** Unlike the exact power-series
`nimblecas.ode` / `nimblecas.perturbation` tools — or the exact rational
`dde` / `dae` / `pde` layers — the paths of an SDE are almost surely
non-differentiable and have no representation over `Q`, so these routines return
floating-point values carrying both **discretisation error** (`O(√dt)` or
`O(dt)` strong, per the scheme's order) and **Monte Carlo sampling error** in any
ensemble average. None of these schemes is exact; none claims exactness over `Q`.

What they *do* guarantee is **determinism**: every draw is a pure function of a
seed, so equal seeds reproduce **bit-identical** paths. Each path is generated
from the stateless counter core `counter_u64(key, i)` of `nimblecas.rng` (keyed
`splitmix64(seed)`), one `N(0,1)` per step via Box-Muller; each path index `p` in
an ensemble is seeded independently via `splitmix64(seed ^ p)`, mirroring mcmc's
`run_parallel_chains` contract. A path is thus a pure function of `(seed, p)`:
any split of the range `0..paths-1` across workers, reassembled in index order,
reproduces the ensemble bit-for-bit, **independent of worker count or
scheduling**. There is no time/entropy seeding and no global mutable state, and
all failure travels the `Result<T>` / `MathError` railway, never an exception.

**CONVENTION HAZARD (Itô vs Stratonovich).** Euler-Maruyama, Milstein, SRK and
tamed Euler all approximate the **Itô** solution of `dX = a dt + b dW`.
Stochastic Heun, being a predictor-corrector that re-uses the **same** increment
`dW` in both stages, instead converges to the **Stratonovich** solution
`dX = a dt + b ∘ dW` — equivalently the Itô SDE with drift `a + ½ b b'`. The two
agree only when `b b' ≡ 0` (e.g. additive noise). For geometric Brownian motion
`a = μx, b = σx` this means `E[X_T] = x0·e^{μT}` for the Itô schemes but
`x0·e^{(μ+½σ²)T}` for Heun. Do not mix the conventions unknowingly.

```cpp
import nimblecas.sde;
```

Depends on [`core`](core.md) (the `Result<T>` / `MathError` railway) and
`nimblecas.rng` (the stateless counter-based RNG substrate — `counter_u64`,
`uniform_unit`, `splitmix64`).

## `SdePath` — a single simulated sample path

`times[n]` is the grid time `n·dt` (with the **final** entry set exactly to `T`
to avoid the rounding drift of `steps·(T/steps)`); `values[n]` is the scheme's
approximation of `X` at that time. Both vectors have length `steps + 1` (the
initial point `X(0) = x0` is included), so `times.size() == values.size()`.

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `times` | `std::vector<double>` | Grid times `0, dt, 2dt, …, T`, length `steps + 1`, last entry exactly `T`. |
| `values` | `std::vector<double>` | `X` at each grid time; `values.front() == x0`, `values.back()` is the terminal `X_T`. |

## `Scheme` — integration-scheme selector

Selects the integrator for the generic ensemble drivers
(`simulate_terminal_scheme` / `terminal_moments_scheme`). Each value names one of
the single-path integrators below. `milstein` is the **only** value that consumes
`b_prime`; for every other scheme `b_prime` is ignored (pass `{}`).

```cpp
enum class Scheme : std::uint8_t {
    euler_maruyama,   // strong 1/2, weak 1   (Itô)
    milstein,         // strong 1             (Itô; requires b')
    stochastic_heun,  // strong 1             (STRATONOVICH)
    srk,              // strong 1             (Itô; derivative-free)
    tamed_euler,      // strong 1/2           (Itô; stiff-stable)
};
```

## Single-path integrators

Every integrator draws from the **same** seeded Brownian stream — one `N(0,1)`
per step, increment `n` consuming counter draws `2n` and `2n+1` keyed
`splitmix64(seed)` — so with `b ≡ 0` the Itô schemes reproduce Euler-Maruyama
bit-for-bit, and `(seed, path)` is reproducible across schemes. All reject the
same [domain-error conditions](#error-model) on the railway.

### Euler-Maruyama — strong 1/2, weak 1 (Itô)

```cpp
[[nodiscard]] auto euler_maruyama(std::function<double(double)> a, std::function<double(double)> b,
                                  double x0, double T, std::uint64_t steps, std::uint64_t seed)
    -> Result<SdePath>;
```

`X_{n+1} = X_n + a(X_n) dt + b(X_n) dW_n`, `dW_n = sqrt(dt) · Z_n`. The
plain forward-Euler Itô scheme; **strong order 1/2, weak order 1**. Needs no `b'`.

### Milstein — strong 1 (Itô; requires `b'`)

```cpp
[[nodiscard]] auto milstein(std::function<double(double)> a, std::function<double(double)> b,
                            std::function<double(double)> b_prime, double x0, double T,
                            std::uint64_t steps, std::uint64_t seed) -> Result<SdePath>;
```

Adds the first-order Itô correction that raises the **strong order to 1**:
`X_{n+1} = X_n + a(X_n) dt + b(X_n) dW_n + ½ b(X_n) b'(X_n) (dW_n² − dt)`.
`b_prime` is the derivative `b'(x)` of the diffusion coefficient, supplied by the
caller (the scheme needs it explicitly; it is the only scheme that consults it).
The correction vanishes in mean but lifts the strong order. With `b ≡ 0` it
coincides with Euler-Maruyama bit-for-bit.

### Stochastic Heun — strong 1 (STRATONOVICH)

```cpp
[[nodiscard]] auto stochastic_heun(std::function<double(double)> a, std::function<double(double)> b,
                                   double x0, double T, std::uint64_t steps, std::uint64_t seed)
    -> Result<SdePath>;
```

Predictor-corrector trapezoid:

```
X̃      = X_n + a(X_n) dt + b(X_n) dW_n                       (Euler predictor)
X_{n+1} = X_n + ½(a(X_n)+a(X̃)) dt + ½(b(X_n)+b(X̃)) dW_n     (trapezoidal corrector)
```

Because the **same** increment `dW_n` is used in predictor and corrector, this
scheme converges to the **Stratonovich** solution `dX = a dt + b ∘ dW`
(equivalently the Itô SDE with drift `a + ½ b b'`), **not** the Itô solution the
other schemes target. **Strong order 1.0, weak order 1.0.** Derivative-free (no
`b'`). With `b ≡ 0` it reduces to the deterministic Heun/RK2 ODE step, so it does
**not** coincide with Euler-Maruyama there.

### SRK (Platen) — strong 1 (Itô; derivative-free)

```cpp
[[nodiscard]] auto srk(std::function<double(double)> a, std::function<double(double)> b, double x0,
                       double T, std::uint64_t steps, std::uint64_t seed) -> Result<SdePath>;
```

Platen's order-1.0 stochastic Runge-Kutta (the derivative-free Milstein scheme,
Kloeden-Platen §11.1):

```
Ŷ      = X_n + a(X_n) dt + b(X_n) √dt                        (supporting value)
X_{n+1} = X_n + a(X_n) dt + b(X_n) dW_n
          + (b(Ŷ) − b(X_n)) (dW_n² − dt) / (2 √dt)
```

Reproduces Milstein's **strong order 1.0 in the Itô sense** while replacing the
analytic derivative `b'(x)` with a finite difference of `b` — so it composes with
the plain `a`/`b` callback and needs no `b'`. **Weak order 1.0.** With `b ≡ 0`
the correction term vanishes and it coincides with Euler-Maruyama bit-for-bit.

### Tamed Euler — strong 1/2 (Itô; stiff-stable)

```cpp
[[nodiscard]] auto tamed_euler(std::function<double(double)> a, std::function<double(double)> b,
                               double x0, double T, std::uint64_t steps, std::uint64_t seed)
    -> Result<SdePath>;
```

Hutzenthaler-Jentzen-Kloeden (2012) tamed drift:
`X_{n+1} = X_n + a(X_n) dt / (1 + |a(X_n)| dt) + b(X_n) dW_n`. For
**superlinearly growing drift** (e.g. `a(x) = −x³`, one-sided Lipschitz but not
globally Lipschitz) explicit Euler-Maruyama diverges — its absolute moments blow
up as the step count grows. Taming caps the per-step drift increment at `1/dt` in
magnitude (`|a dt/(1+|a|dt)| < 1`), so the step stays finite where plain Euler
overflows, while leaving the **strong order at 1/2** and the limit unchanged (the
taming perturbation is `O(dt)` per step). **Itô** convention. Derivative-free.

## Ensemble drivers

Both families simulate `paths` independent seeded paths (path `p` seeded
`splitmix64(seed ^ p)`), reduce in index order, and return only the terminal
values `X_T` — the natural input for a terminal expectation such as an option
price. Element `p` is a pure function of `(seed, p)`, independent of `paths` and
of how `0..paths-1` is partitioned across workers.

### Legacy (bool-selected) drivers

```cpp
[[nodiscard]] auto simulate_terminal(std::function<double(double)> a,
                                     std::function<double(double)> b,
                                     std::function<double(double)> b_prime, double x0, double T,
                                     std::uint64_t steps, std::uint64_t paths, std::uint64_t seed,
                                     bool use_milstein) -> Result<std::vector<double>>;

[[nodiscard]] auto terminal_moments(std::function<double(double)> a, std::function<double(double)> b,
                                    std::function<double(double)> b_prime, double x0, double T,
                                    std::uint64_t steps, std::uint64_t paths, std::uint64_t seed,
                                    bool use_milstein) -> Result<std::pair<double, double>>;
```

| Function | Behavior |
| :--- | :--- |
| `simulate_terminal` | Return the terminal `X_T` of each of `paths` seeded paths. With `use_milstein == false` the Euler-Maruyama scheme is used and `b_prime` is ignored (pass `{}`); with `use_milstein == true` the Milstein scheme is used and `b_prime` must be non-empty. |
| `terminal_moments` | Return `{ sample mean, sample variance }` of the terminal values. The variance uses the unbiased (Bessel, `n − 1`) estimator when `paths >= 2` and is `0` for a single path. Scheme selection and the seeding/partition contract are exactly as for `simulate_terminal`. |

### Generic (Scheme-parameterised) drivers

```cpp
[[nodiscard]] auto simulate_terminal_scheme(std::function<double(double)> a,
                                            std::function<double(double)> b,
                                            std::function<double(double)> b_prime, double x0,
                                            double T, std::uint64_t steps, std::uint64_t paths,
                                            std::uint64_t seed, Scheme scheme)
    -> Result<std::vector<double>>;

[[nodiscard]] auto terminal_moments_scheme(std::function<double(double)> a,
                                           std::function<double(double)> b,
                                           std::function<double(double)> b_prime, double x0,
                                           double T, std::uint64_t steps, std::uint64_t paths,
                                           std::uint64_t seed, Scheme scheme)
    -> Result<std::pair<double, double>>;
```

These mirror `simulate_terminal` / `terminal_moments` exactly — same per-path
seeding `splitmix64(seed ^ p)`, same in-index-order reduction, same
partition/thread-count independence — but select the integrator through the
[`Scheme`](#scheme--integration-scheme-selector) enum instead of the
`use_milstein` bool, so Heun, SRK and tamed Euler get the same reproducible
multi-path driver. `b_prime` is consulted **only** when `scheme == Scheme::milstein`;
for every other scheme it is ignored (pass `{}`). `Scheme::euler_maruyama` and
`Scheme::milstein` reproduce the legacy drivers bit-for-bit.

| Function | Behavior |
| :--- | :--- |
| `simulate_terminal_scheme` | Terminal `X_T` of each of `paths` seeded paths under the chosen `scheme`. |
| `terminal_moments_scheme` | `{ sample mean, unbiased (n−1) sample variance }` of `X_T` under the chosen `scheme`. |

## Error model

Every entry point rejects invalid input on the railway with
`MathError::domain_error` — a non-finite `T` or `x0` is rejected rather than
silently producing an all-NaN "success" path (`T <= 0` alone is `false` for
`NaN`).

| Condition | Error |
| :--- | :--- |
| `steps == 0` | `MathError::domain_error` |
| `T` non-finite (`NaN` / `inf`) or `T <= 0` | `MathError::domain_error` |
| `x0` non-finite (`NaN` / `inf`) | `MathError::domain_error` |
| `a` or `b` an empty `std::function` | `MathError::domain_error` |
| Milstein requested with an empty `b_prime` (`milstein`, `simulate_terminal` with `use_milstein == true`, or `simulate_terminal_scheme` with `Scheme::milstein`) | `MathError::domain_error` |
| `paths == 0` (any ensemble driver) | `MathError::domain_error` |

Requesting the Milstein scheme without a `b_prime` is a **domain error on the
railway**, never a `std::bad_function_call` thrown off-railway. Derivative-free
schemes never consult `b_prime`, so an empty `b_prime` is fine for them. The
ensemble drivers propagate the per-path domain error unchanged. There is no
`overflow` or `division_by_zero` path.

## Worked examples

```cpp
import std;
import nimblecas.core;
import nimblecas.rng;
import nimblecas.sde;
using namespace nimblecas;

// Geometric Brownian motion dX = μX dt + σX dW, with μ=0.1, σ=0.2, x0=1, T=1.
const double mu = 0.1, sigma = 0.2;
auto a  = [mu](double x)  { return mu * x; };
auto b  = [sigma](double x) { return sigma * x; };
auto bp = [sigma](double)   { return sigma; };   // b'(x) = σ, for Milstein

// Itô schemes: E[X_T] = x0·exp(μT) = exp(0.1) ≈ 1.10517 (Monte Carlo estimate).
auto ito = terminal_moments(a, b, /*b_prime=*/{}, 1.0, 1.0, 200, 40000, 20260703,
                            /*use_milstein=*/false);
ito->first;   // ≈ exp(0.1) ≈ 1.10517   (sample mean)
ito->second;  // > 0                     (unbiased sample variance)

// Milstein reduces path-wise error, not the true mean — still exp(μT).
terminal_moments(a, b, bp, 1.0, 1.0, 200, 40000, 777, /*use_milstein=*/true)->first;  // ≈ 1.10517

// SRK and tamed Euler are Itô too: same exp(μT) mean via the generic driver.
terminal_moments_scheme(a, b, {}, 1.0, 1.0, 200, 40000, 4242,   Scheme::srk)->first;          // ≈ 1.10517
terminal_moments_scheme(a, b, {}, 1.0, 1.0, 400, 40000, 909091, Scheme::tamed_euler)->first;  // ≈ 1.10517

// Stochastic Heun integrates the STRATONOVICH SDE: E[X_T] = x0·exp((μ+½σ²)T) = exp(0.12) ≈ 1.12750,
// NOT the Itô mean — a deliberate, documented convention difference.
terminal_moments_scheme(a, b, {}, 1.0, 1.0, 200, 40000, 20260703, Scheme::stochastic_heun)->first;  // ≈ 1.12750

// Zero diffusion collapses to the ODE dX = a dt; a(x)=x, x0=1, T=1 ⇒ X_T ≈ e (discretisation error only).
auto ode_like = [](double x) { return x; };
auto zero     = [](double)   { return 0.0; };
euler_maruyama(ode_like, zero, 1.0, 1.0, 2000, 123)->values.back();  // ≈ e ≈ 2.71828
// With b ≡ 0 the Itô schemes agree bit-for-bit: Euler == Milstein == SRK.

// Reproducibility: identical (args, seed) reproduce a bit-identical path (no hidden state).
auto p1 = milstein(a, b, bp, 1.0, 1.0, 300, 2024);
auto p2 = milstein(a, b, bp, 1.0, 1.0, 300, 2024);
// p1->values == p2->values, element-wise; a different seed changes the terminal value.

// Ensemble is partition-independent: element p == the single-path run on splitmix64(seed ^ p).
auto full = simulate_terminal_scheme(a, b, {}, 1.0, 1.0, 120, 48, 20260703, Scheme::srk).value();
auto one  = srk(a, b, 1.0, 1.0, 120, splitmix64(20260703 ^ 3)).value();
// full[3] == one.values.back()

// Tamed Euler stays finite on superlinear drift where plain Euler diverges.
auto stiff = [](double x) { return -x * x * x; };   // one-sided Lipschitz, NOT globally Lipschitz
euler_maruyama(stiff, zero, 5.0, 1.0, 10, 31337)->values.back();  // non-finite: plain Euler blows up
tamed_euler   (stiff, zero, 5.0, 1.0, 10, 31337)->values.back();  // finite and bounded

// Error paths (railway, never exceptions).
euler_maruyama(a, b, 1.0, 1.0, 0, 1).error();                       // domain_error (steps == 0)
euler_maruyama(a, b, 1.0, 0.0, 10, 1).error();                      // domain_error (T <= 0)
milstein(a, b, bp, std::numeric_limits<double>::quiet_NaN(), 1.0, 10, 1).error();  // domain_error (x0 NaN)
simulate_terminal(a, b, {}, 1.0, 1.0, 10, 0, 1, false).error();     // domain_error (paths == 0)
simulate_terminal(a, b, {}, 1.0, 1.0, 10, 5, 1, /*use_milstein=*/true).error();  // domain_error (no b')
simulate_terminal_scheme(a, b, {}, 1.0, 1.0, 10, 5, 1, Scheme::milstein).error();  // domain_error (no b')
```

## See also

- [`nimblecas.mcmc`](mcmc.md) — the sibling numerical / Monte Carlo layer whose
  per-chain-index seeding contract this module mirrors.
- `nimblecas.rng` ([rng.md](rng.md)) — the stateless counter-based RNG substrate
  (`counter_u64`, `uniform_unit`, `splitmix64`) that drives every path.
- [`nimblecas.ode`](ode.md), [`nimblecas.dde`](dde.md), [`nimblecas.dae`](dae.md),
  [`nimblecas.pde`](pde.md) — the **exact** (rational / power-series) differential-
  equation layers, in contrast to this numerical one.
- [Documentation hub](../Index.md)
