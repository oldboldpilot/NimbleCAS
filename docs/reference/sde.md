# `nimblecas.sde` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/sde/sde.cppm`

Deterministic (seeded), parallelisable integrators for scalar Itô stochastic
differential equations `dX = a(X) dt + b(X) dW_t`, `X(0) = x0`, on `[0, T]` with
`steps` uniform steps `dt = T / steps` and standard-normal Wiener increments
`dW = sqrt(dt) · Z`, `Z ~ N(0, 1)`. Five single-path schemes are provided
(Euler-Maruyama, Milstein, stochastic Heun, derivative-free SRK, tamed Euler),
plus ensemble drivers that average their terminal values. Layered on top:
**jump-diffusion** (`jump_euler_maruyama`, a Merton log-normal jump
specification via `merton_jumps`), a **drift-implicit theta scheme**
(`theta_euler`/`jump_theta_euler`, for stiff drift) driven through
[`nlsolve`](nlsolve.md)'s Newton solver, and an `EnsembleEstimate` reduction
whose multi-path ensemble now fans out over [`nimblecas.parallel`](parallel.md)
(TBB/PPL) — bit-identically to a serial reduction.

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

Depends on [`core`](core.md) (the `Result<T>` / `MathError` railway),
`nimblecas.rng` (the stateless counter-based RNG substrate — `counter_u64`,
`uniform_unit`, `splitmix64`), [`nlsolve`](nlsolve.md) (`newton`, the implicit
solver `theta_euler`/`jump_theta_euler` drive per step), and
[`parallel`](parallel.md) (the TBB/PPL fork-join layer the jump-ensemble
drivers fan paths out over).

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

## Jump-diffusion and drift-implicit (theta) additions

These are **additive only**: none of the five schemes above change, and every
routine here consumes the Brownian stream **exactly** as `euler_maruyama`
does — `standard_normal(key, n)` at counter indices `2n`/`2n+1` of
`key = splitmix64(seed)` — so a variable jump count or an implicit drift solve
can never shift a Brownian draw. All jump randomness (Poisson counts and jump
marks) lives on a **separate, domain-separated** sub-key derived from the same
path key (`counter_u64(key, kJumpDomain)` for the Poisson count,
further separated by step and mark index for each mark), so it is likewise a
pure function of `(seed, path, step, mark index)` and never touches the
Brownian indices. Two consequences are both **bit-for-bit** and tested:
`lambda == 0` makes `jump_euler_maruyama` **identical** to `euler_maruyama`,
and `theta == 0` makes `theta_euler` **identical** to `euler_maruyama` — both
special-cased to the exact same arithmetic expression, not merely "close".

### `JumpSpec` and `merton_jumps` — a compound-Poisson jump specification

For a jump-diffusion SDE `dX = a(X) dt + b(X) dW + dJ`, `dJ` a compound
Poisson process with intensity `lambda` (jumps per unit time):

```cpp
struct JumpSpec {
    double lambda{0.0};                                // jump intensity (jumps / unit time)
    std::function<double(double)> size_quantile{};      // u in (0,1) -> jump mark J
    std::function<double(double, double)> impulse{};    // (x, J) -> state increment c(x, J)
};
```

`size_quantile(u)` maps a uniform `u in (0,1)` to a jump mark `J` via the
inverse CDF of the mark distribution (marks are drawn by inversion, one
uniform per mark); `impulse(x, J)` maps the **pre-jump** state `x` and a mark
`J` to the state increment applied at a jump. `lambda == 0.0` (the default)
disables jumps entirely; `size_quantile`/`impulse` may then be empty.

```cpp
[[nodiscard]] auto merton_jumps(double lambda, double mu_j, double sigma_j) -> Result<JumpSpec>;
```

Builds a **Merton (1976)** log-normal jump specification: marks
`J ~ N(mu_j, sigma_j^2)` drawn by the inverse-normal quantile transform of a
single uniform, applied to a price-like process via the multiplicative
impulse `c(x, J) = x·(e^J − 1) = x·expm1(J)` (a jump multiplies the pre-jump
state by `e^J`). `domain_error` if `lambda` is negative/non-finite, `mu_j` is
non-finite, or `sigma_j` is negative/non-finite.

### `jump_euler_maruyama` — Euler-Maruyama plus an explicit jump sum

```cpp
[[nodiscard]] auto jump_euler_maruyama(std::function<double(double)> a,
                                       std::function<double(double)> b, const JumpSpec& jumps,
                                       double x0, double T, std::uint64_t steps,
                                       std::uint64_t seed) -> Result<SdePath>;
```

```
X_{n+1} = X_n + a(X_n) dt + b(X_n) dW_n + sum_{i=1}^{N_n} impulse(X_n, J_{n,i}),
N_n ~ Poisson(lambda dt).
```

`dW_n` is exactly `euler_maruyama`'s draw; `N_n` and every mark `J_{n,i}` come
from the domain-separated jump sub-key, so they never perturb the Brownian
stream. With `jumps.lambda == 0.0` this is **bit-for-bit identical** to
`euler_maruyama`. `domain_error` if `steps == 0`, `T` non-finite/`<= 0`, `x0`
non-finite, `a`/`b` empty, `jumps.lambda` negative/non-finite,
`jumps.lambda * dt > 700` (the point past which `e^{-lambda dt}` underflows to
`0` in double precision), or `jumps.lambda > 0` with an empty
`size_quantile`/`impulse`.

### `theta_euler` — drift-implicit (theta) Euler-Maruyama for stiff SDEs

```cpp
[[nodiscard]] auto theta_euler(std::function<double(double)> a,
                               std::function<double(double)> a_prime,
                               std::function<double(double)> b, double x0, double T,
                               std::uint64_t steps, std::uint64_t seed, double theta)
    -> Result<SdePath>;
```

Diffusion **explicit** (Itô), drift **implicit** by a `theta`-blend, solved
per step for `xi` via `nlsolve::newton` on the 1-D residual:

```
xi = X_n + (theta a(xi) + (1-theta) a(X_n)) dt + b(X_n) dW_n,   X_{n+1} = xi.
```

The analytic Jacobian `1 - theta a'(xi) dt` is used when `a_prime` is
supplied; an empty `a_prime` falls back to `nlsolve`'s finite-difference
Jacobian. `theta == 1` is fully implicit backward Euler on the drift
(unconditionally A-stable in the deterministic/linear sense); `theta == 0.5`
is the (drift) trapezoidal/Crank-Nicolson rule; **`theta == 0.0` is exactly
`euler_maruyama`**, special-cased to the identical arithmetic expression
rather than routed through Newton on a trivial linear residual, so the two
are bit-for-bit identical.

**Stability motivation.** For stiff linear mean reversion `a(x) = -kappa(x -
m)` with `kappa·dt >> 1` (e.g. a fast-reverting short-rate or variance process
on a coarse grid), explicit Euler-Maruyama's amplification factor
`|1 - kappa dt|` exceeds `1` and the scheme oscillates/blows up; `theta >=
1/2` keeps the implicit drift step stable regardless of `kappa dt`.

**Newton's `converged == false`** (stagnation, singular Jacobian, iteration
budget exhausted) is a **hard `not_converged` for the whole path** — an
unconverged iterate is never accepted as a step. `domain_error` if
`steps == 0`, `T` non-finite/`<= 0`, `x0` non-finite, `a`/`b` empty, or
`theta` outside `[0, 1]` or non-finite.

### `jump_theta_euler` — theta-implicit drift plus jumps

```cpp
[[nodiscard]] auto jump_theta_euler(std::function<double(double)> a,
                                    std::function<double(double)> a_prime,
                                    std::function<double(double)> b, const JumpSpec& jumps,
                                    double x0, double T, std::uint64_t steps,
                                    std::uint64_t seed, double theta) -> Result<SdePath>;
```

`theta_euler`'s drift-implicit, diffusion-explicit step **plus**
`jump_euler_maruyama`'s explicit compound-Poisson jump sum (marks and impulse
applied to the pre-step state `X_n`, exactly as in `jump_euler_maruyama`).
`theta == 0` and `jumps.lambda == 0` together reduce this to
`euler_maruyama` bit-for-bit. Domain-error conditions are the union of
`theta_euler`'s and `jump_euler_maruyama`'s.

### `EnsembleEstimate` and the jump/theta ensemble drivers

```cpp
struct EnsembleEstimate {
    double mean{0.0};
    double variance{0.0};
    double std_error{0.0};   // sqrt(variance / paths) -- the standard error of the MEAN
    std::uint64_t paths{0};
};

[[nodiscard]] auto simulate_terminal_jump(std::function<double(double)> a,
                                          std::function<double(double)> a_prime,
                                          std::function<double(double)> b, const JumpSpec& jumps,
                                          double x0, double T, std::uint64_t steps,
                                          std::uint64_t paths, std::uint64_t seed, double theta)
    -> Result<std::vector<double>>;

[[nodiscard]] auto terminal_estimate_jump(std::function<double(double)> a,
                                          std::function<double(double)> a_prime,
                                          std::function<double(double)> b, const JumpSpec& jumps,
                                          double x0, double T, std::uint64_t steps,
                                          std::uint64_t paths, std::uint64_t seed, double theta)
    -> Result<EnsembleEstimate>;
```

`EnsembleEstimate` carries its **own** Monte Carlo statistical error, in the
same spirit as `nimblecas::pricing::McResult`: `mean`/`variance` are the
sample mean and unbiased (`n−1`) sample variance of `X_T` over `paths` seeded
`jump_theta_euler` paths, and `std_error = sqrt(variance / paths)` is the
standard error **of the mean** (not of a single draw). `theta == 0` and
`jumps.lambda == 0` together recover plain `euler_maruyama`.
`simulate_terminal_jump` returns the raw terminal values; `terminal_estimate_jump`
reduces them to an `EnsembleEstimate`. **Partition-independence**: path `p` is
seeded `splitmix64(seed ^ p)` — exactly `simulate_terminal_scheme`'s contract
— so `X_T` for path `p` is a pure function of `(seed, p)`, independent of
`paths` and of how `0..paths-1` is partitioned across workers. **If any
path's implicit solve fails to converge the whole call returns
`not_converged`** — a partially-valid ensemble is never returned.

### Ensemble parallelism over `nimblecas.parallel`

`simulate_terminal_jump` fans its `paths` iterations out over
`nimblecas.parallel::transform_index` (TBB on Linux/macOS, PPL on Windows)
instead of a serial loop. Because `transform_index` is **order-preserving**
and each path index `p` writes only its own output slot, and because every
path is already a pure function of `(seed, p)` (the partition-independence
contract above), the parallel result is **bit-identical to a serial
reduction** regardless of worker count or scheduling — the same
determinism guarantee [`taskdag`](taskdag.md) and [`montecarlo`](montecarlo.md)
give for their own parallel fan-outs. The callback closures (`a`, `b`,
`a_prime`, and the `JumpSpec`'s `size_quantile`/`impulse`) must be **pure**,
since they are invoked concurrently for distinct path indices — the same
purity requirement `nimblecas.taskdag`'s `TaskFn` places on its callers. A
single failing path's `not_converged` is detected deterministically: the
**lowest-index** failing path is reported, independent of which worker
happened to finish it first.

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
| `paths == 0` (any ensemble driver, including `simulate_terminal_jump`/`terminal_estimate_jump`) | `MathError::domain_error` |
| `merton_jumps`: `lambda` negative/non-finite, `mu_j` non-finite, or `sigma_j` negative/non-finite | `MathError::domain_error` |
| `jump_euler_maruyama`/`jump_theta_euler`/`simulate_terminal_jump`/`terminal_estimate_jump`: `jumps.lambda` negative/non-finite, or `jumps.lambda * dt > 700` | `MathError::domain_error` |
| `jumps.lambda > 0` with an empty `size_quantile` or `impulse` | `MathError::domain_error` |
| `theta_euler`/`jump_theta_euler`/`simulate_terminal_jump`/`terminal_estimate_jump`: `theta` outside `[0, 1]` or non-finite | `MathError::domain_error` |
| The implicit Newton solve fails to converge (stagnation, singular Jacobian, iteration budget exhausted), anywhere in the path or ensemble | `MathError::not_converged` |

Requesting the Milstein scheme without a `b_prime` is a **domain error on the
railway**, never a `std::bad_function_call` thrown off-railway. Derivative-free
schemes never consult `b_prime`, so an empty `b_prime` is fine for them. The
ensemble drivers propagate the per-path domain error unchanged. There is no
`overflow` or `division_by_zero` path. `not_converged` from an implicit
(`theta`) step is a **hard failure for the whole call** — never a partial
path or a partially-valid ensemble.

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

// --- Jump-diffusion (Merton) ---
// dX = mu X dt + sigma X dW + dJ, dJ compound Poisson with marks J ~ N(mu_j, sigma_j^2)
// and multiplicative impulse c(x,J) = x(e^J - 1). Analytic terminal mean (Merton 1976):
//   E[X_T] = x0 exp(mu T + lambda T (exp(mu_j + sigma_j^2/2) - 1)).
const double mu2 = 0.05, sigma2 = 0.2, lambda = 1.0, mu_j = -0.1, sigma_j = 0.15;
auto a2 = [mu2](double x) { return mu2 * x; };
auto b2 = [sigma2](double x) { return sigma2 * x; };
auto jumps = merton_jumps(lambda, mu_j, sigma_j).value();
auto est = terminal_estimate_jump(a2, /*a_prime=*/{}, b2, jumps, 1.0, 1.0, 500, 40000,
                                  20260812, /*theta=*/0.0).value();
const double expected_mean =
    std::exp(mu2 * 1.0 + lambda * 1.0 * (std::exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0));
std::abs(est.mean - expected_mean) <= 4.0 * est.std_error;  // true (Monte Carlo agreement)
est.paths == 40000;                                          // true — echoes the input paths

// --- Drift-implicit (theta) Euler-Maruyama stabilises a stiff mean-reverting OU ---
// dX = -kappa(X-m) dt + sigma dW, kappa=50, dt=0.05 => kappa*dt=2.5 puts explicit
// Euler-Maruyama's amplification factor |1-kappa*dt|=1.5 outside the stability disc:
// plain Euler diverges, theta_euler(theta=1) (fully implicit drift) stays finite and
// tracks the analytic OU moments E[X_T]=m+(x0-m)e^{-kappa T}, Var=sigma^2/(2kappa)(1-e^{-2kappa T}).
const double kappa = 50.0, m = 0.05, sig = 0.1;
auto a3 = [kappa, m](double x) { return -kappa * (x - m); };
auto a3p = [kappa](double) { return -kappa; };
auto b3 = [sig](double) { return sig; };

auto euler_mom = terminal_moments(a3, b3, {}, 1.0, 2.0, 40, 5000, 990099, false).value();
std::abs(euler_mom.first) > 1000.0;   // true — plain Euler diverges on the stiff OU

auto est2 = terminal_estimate_jump(a3, a3p, b3, JumpSpec{}, 1.0, 2.0, 40, 5000, 990099,
                                   /*theta=*/1.0).value();
std::isfinite(est2.mean);             // true — theta_euler(theta=1) stays finite
// est2.mean ~ m + (x0-m)*exp(-kappa*T), est2.variance ~ sigma^2/(2kappa)(1-exp(-2kappa*T)),
// both within 4*std_error.

// theta==0 and lambda==0 both reduce bit-for-bit to euler_maruyama.
auto baseline = euler_maruyama(a, b, 1.0, 1.0, 300, 4242424).value();
auto th0 = theta_euler(a, /*a_prime=*/[](double){return mu;}, b, 1.0, 1.0, 300, 4242424, 0.0).value();
auto jp0 = jump_euler_maruyama(a, b, JumpSpec{}, 1.0, 1.0, 300, 4242424).value();
// baseline.values == th0.values == jp0.values, element-wise, EXACTLY.

// Ensemble parallelism: simulate_terminal_jump fans out over nimblecas.parallel and is
// bit-identical to the serial per-path reconstruction (partition-independence).
auto ens = simulate_terminal_jump(a3, a3p, b3, jumps, 1.0, 1.0, 100, 200, 8675309, 0.5).value();
auto ens_again = simulate_terminal_jump(a3, a3p, b3, jumps, 1.0, 1.0, 100, 200, 8675309, 0.5).value();
// ens == ens_again element-wise (reruns bit-identically), and ens[p] equals an
// independent jump_theta_euler(a3, a3p, b3, jumps, 1.0, 1.0, 100, splitmix64(8675309 ^ p), 0.5)
// for every p.
```

## See also

- [`nimblecas.mcmc`](mcmc.md) — the sibling numerical / Monte Carlo layer whose
  per-chain-index seeding contract this module mirrors.
- `nimblecas.rng` ([rng.md](rng.md)) — the stateless counter-based RNG substrate
  (`counter_u64`, `uniform_unit`, `splitmix64`) that drives every path.
- [`nimblecas.nlsolve`](nlsolve.md) — the Newton solver (`newton`)
  `theta_euler`/`jump_theta_euler` drive per step for the implicit drift.
- [`nimblecas.parallel`](parallel.md) — the deterministic TBB/PPL fork-join
  layer `simulate_terminal_jump`/`terminal_estimate_jump` fan paths out over,
  bit-identically to a serial reduction.
- [`nimblecas.taskdag`](taskdag.md) — a sibling single-node deterministic
  scheduler over the same `parallel` layer, with an analogous
  serial/parallel bit-identity contract.
- [`nimblecas.ode`](ode.md), [`nimblecas.dde`](dde.md), [`nimblecas.dae`](dae.md),
  [`nimblecas.pde`](pde.md) — the **exact** (rational / power-series) differential-
  equation layers, in contrast to this numerical one.
- [Documentation hub](../Index.md)
