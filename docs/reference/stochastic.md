# `nimblecas.stochastic` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/stochastic/stochastic.cppm`

**Stochastic-process analysis** (ROADMAP §7.8, stationary-process slice): the
*structure* of a stationary process — its stationary law, its correlation in
time, and its spectrum. It is deliberately distinct from the sampling modules
([`sde`](sde.md) simulates paths, [`montecarlo`](montecarlo.md) /
[`mcmc`](mcmc.md) draw samples). **Nothing here draws a random number**; every
result is an exact algebraic statement about the process, or — where the
mathematics leaves the field `Q` — an honestly-labelled numerical evaluation.

The **honesty boundary** is documented and enforced by the API split:

- **Exact over Q:** `is_stochastic`, the Markov `stationary_distribution`, `P^n`,
  irreducibility / aperiodicity, the mean-first-passage / fundamental-matrix
  quantities, `ergodic_mean`, the CTMC `ctmc_stationary_distribution`, the
  autocovariance / autocorrelation / cross-covariance of rational records, the
  Yule–Walker autocorrelation and AR/MA autocovariance, and the unit-circle test
  of **rational** characteristic roots. These are pure reduced-fraction results
  from overflow-checked `Rational` arithmetic and exact rational linear solves —
  **not** power iteration.
- **Numerical (double):** every power-spectral-density / frequency-domain value.
  The Wiener–Khinchin transform is a trigonometric sum, so
  `power_spectral_density` and `arma_spectral_density` leave `Q` and return
  `(frequency, value)` arrays over the one-sided grid `[0, 1/2]`.
- **Diagnostic only:** `is_wss` on a finite record. A finite sample can only
  **diagnose** wide-sense stationarity of the record in hand; it can **never
  certify** the stationarity of the underlying process law, which is a statement
  about an infinite ensemble.
- **`indeterminate`:** stationarity / invertibility tests decide from the
  characteristic roots. Rational roots are found and tested exactly; when
  irrational/complex roots remain **unenumerated** over `Q` the verdict is
  honestly reported as `indeterminate` (unless a rational root already proves it
  unstable). A normalised cross-*correlation* coefficient needs
  `sqrt(var_x var_y)` and so is **not** exact over Q — the exact deliverable is
  the cross-*covariance*.

Every failure travels the railway (`Result<T>` / `MathError` via `make_error`);
nothing throws. All exact arithmetic is overflow-checked `Rational` (Rule 32).

```cpp
import nimblecas.stochastic;   // namespace nimblecas
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), [`matrix`](matrix.md),
[`stats`](stats.md), and [`roots`](roots.md).

## Discrete-time finite-state Markov chains (exact over Q)

A discrete-time chain is a **row-stochastic** rational transition matrix `P`
(rows sum to `1`, entries `>= 0`, checked exactly over Q).

```cpp
[[nodiscard]] auto is_stochastic(const Matrix& p) -> Result<bool>;
[[nodiscard]] auto stationary_distribution(const Matrix& p) -> Result<std::vector<Rational>>;
[[nodiscard]] auto n_step_transition(const Matrix& p, std::size_t n) -> Result<Matrix>;
[[nodiscard]] auto is_irreducible(const Matrix& p) -> Result<bool>;
[[nodiscard]] auto is_aperiodic(const Matrix& p) -> Result<bool>;
[[nodiscard]] auto fundamental_matrix(const Matrix& p) -> Result<Matrix>;
[[nodiscard]] auto mean_first_passage_times(const Matrix& p) -> Result<Matrix>;
[[nodiscard]] auto ergodic_mean(const Matrix& p, std::span<const Rational> values) -> Result<Rational>;
```

| Function | Behavior |
| :--- | :--- |
| `is_stochastic` | `true` iff `P` is square, every entry `>= 0`, and every row sums **exactly** to `1`. A non-square matrix is simply not row-stochastic and yields `false` (asking is not an error). Propagates overflow from the exact row sums. |
| `stationary_distribution` | The unique probability row vector `π` with `π P = π` and `Σ π_i = 1`, solved **exactly** as the left null space of `(P^T − I)` with the last row replaced by the normalisation `1^T π = 1` (`Matrix::solve` over Q — not power iteration). `domain_error` when `P` is not row-stochastic or the system is singular (a reducible chain with several stationary laws). |
| `n_step_transition` | `P^n` by exact binary exponentiation; `P^0` is the identity. Requires a square `P` (`domain_error` otherwise). `P` need not be stochastic for the raw matrix power, but the probabilistic reading assumes it is. Propagates overflow. |
| `is_irreducible` | `true` iff every state reaches every other — the directed graph of positive transitions is strongly connected, by transitive closure over Q (an entry "exists" iff **nonzero**, exact, no tolerance). Requires a non-empty square `P` (`domain_error` otherwise). |
| `is_aperiodic` | `true` iff an **irreducible** chain has period 1, decided by primitivity (some boolean power of the adjacency matrix is strictly positive, reached within Wielandt's bound `(n−1)^2 + 1` if at all). A **reducible** chain has per-class periods and yields `domain_error` (aperiodicity only where tractable). |
| `fundamental_matrix` | `Z = (I − P + W)^{-1}` of an ergodic chain, where `W` has every row equal to `π`. Exact over Q. `domain_error` when `π` does not exist or `I − P + W` is singular. |
| `mean_first_passage_times` | The matrix `M` where `M(i, j)` (`i != j`) is the expected steps to first reach `j` from `i`, and the diagonal `M(j, j) = 1/π_j` is the mean recurrence time. From the fundamental matrix, `M(i, j) = (Z(j, j) − Z(i, j)) / π_j`. Exact over Q; requires an ergodic chain with a stationary distribution (`domain_error` otherwise). |
| `ergodic_mean` | The stationary (space) average `Σ_i π_i f(i)` of a state function `f` given as one rational value per state. By the ergodic theorem this is the almost-sure limit of the time average `(1/T) Σ_t f(X_t)` for an irreducible aperiodic chain. Exact over Q. `values` must have one entry per state (`domain_error` otherwise) and `π` must exist. |

## Continuous-time Markov chains (exact over Q)

A continuous-time chain is a rational **generator** `Q` (rows sum to `0`,
off-diagonals `>= 0`).

```cpp
[[nodiscard]] auto is_generator(const Matrix& q) -> Result<bool>;
[[nodiscard]] auto ctmc_stationary_distribution(const Matrix& q) -> Result<std::vector<Rational>>;
```

| Function | Behavior |
| :--- | :--- |
| `is_generator` | `true` iff `Q` is square, every off-diagonal entry `>= 0`, and every row sums **exactly** to `0` (each diagonal is minus its off-diagonal row sum). A non-square matrix yields `false`. Propagates overflow from the exact row sums. |
| `ctmc_stationary_distribution` | The probability row vector `π` with `π Q = 0` and `Σ π_i = 1`, solved **exactly** as the left null space of `Q` with the last equation replaced by the normalisation. `domain_error` when `Q` is not a generator or the system is singular. |

## Absorbing chains, random walks, resolvent, birth–death (exact over Q)

These are exact-`Rational` closed forms and matrix inverses for the classical
absorbing / hitting analyses.

```cpp
struct AbsorbingChain {
    std::vector<std::size_t> transient;    // original indices of the transient states
    std::vector<std::size_t> absorbing;    // original indices of the absorbing states
    Matrix fundamental;                    // N = (I − Q)^{-1}
    std::vector<Rational> expected_steps;  // t = N 1
    Matrix absorption_probabilities;       // B = N R
};

[[nodiscard]] auto absorbing_analysis(const Matrix& p) -> Result<AbsorbingChain>;
[[nodiscard]] auto resolvent(const Matrix& q, const Rational& s) -> Result<Matrix>;
[[nodiscard]] auto gamblers_ruin_probability(std::size_t n, std::size_t k, const Rational& p)
    -> Result<Rational>;
[[nodiscard]] auto gamblers_ruin_duration(std::size_t n, std::size_t k, const Rational& p)
    -> Result<Rational>;
[[nodiscard]] auto birth_death_stationary(std::span<const Rational> birth,
                                          std::span<const Rational> death)
    -> Result<std::vector<Rational>>;
```

| Function | Behavior |
| :--- | :--- |
| `absorbing_analysis` | The canonical-form analysis of a row-stochastic `P`. A state `i` is **absorbing** iff `P(i, i) = 1` (row-stochasticity then zeroes the rest of the row); the others are **transient**. Relabelling transient-first exposes `P = [[Q, R], [0, I]]`: `fundamental` is `N = (I − Q)^{-1}` (`t × t`), `expected_steps[a] = (N 1)_a` is the expected steps to absorption from `transient[a]`, and `absorption_probabilities = N R` (`t × r`) gives `P(absorbed in absorbing[c] | start transient[a])`. All **exact over Q**. `domain_error` when `P` is not row-stochastic, has no absorbing state, or `I − Q` is singular (a transient class that never absorbs). |
| `resolvent` | `(s I − Q)^{-1}` evaluated **exactly** at a given rational `s` (`Q` any square rational matrix — a CTMC generator, or an absorbing chain's transient block). `domain_error` when `Q` is not square, or `s I − Q` is singular (`s` an eigenvalue of `Q` — a pole of the resolvent). This is the resolvent **at a point**; the fully symbolic matrix of rational functions of `s` is not formed. |
| `gamblers_ruin_probability` | The **exact** probability that a biased ±1 walk on `{0, …, n}` with up-probability `p` reaches `0` before `n`, started from `k`. For `p ≠ 1/2` with `r = (1 − p)/p` it is `(r^k − r^n)/(1 − r^n)`; for `p = 1/2` it is `(n − k)/n`. Boundaries `k = 0 → 1`, `k = n → 0`. `domain_error` when `n = 0`, `k > n`, or `p ∉ (0, 1)`. |
| `gamblers_ruin_duration` | The **exact** expected number of steps to absorption of the same walk from `k`. For `p = 1/2` it is `k(n − k)`; for `p ≠ 1/2` (`q = 1 − p`, `r = q/p`) it is `(1/(q − p)) [k − n (1 − r^k)/(1 − r^n)]`. Boundaries `k = 0, n → 0`. `domain_error` as `gamblers_ruin_probability`. |
| `birth_death_stationary` | The **exact** stationary law of a birth–death chain on `{0, …, n−1}` from birth rates `birth[i]` (rate `i → i+1`) and death rates `death[i]` (rate `(i+1) → i`), via detailed balance `π_i ∝ Π_{j=1}^{i} λ_{j−1}/μ_j`, normalised. `birth` and `death` must have equal length `L = n − 1` (empty ⇒ single state `[1]`). `domain_error` on unequal lengths or a negative rate; `division_by_zero` if an interior death rate is `0`. |

## Wide-sense-stationary (WSS) sample analysis

For a finite record of rational samples the mean, the autocovariance, the
autocorrelation and the cross-covariance are **exact over Q**. `is_wss` is a
finite-record **diagnostic**.

```cpp
[[nodiscard]] auto autocovariance_at(std::span<const Rational> x, std::size_t lag, bool biased)
    -> Result<Rational>;
[[nodiscard]] auto autocovariance(std::span<const Rational> x, std::size_t max_lag, bool biased)
    -> Result<std::vector<Rational>>;
[[nodiscard]] auto autocorrelation(std::span<const Rational> x, std::size_t max_lag, bool biased)
    -> Result<std::vector<Rational>>;
[[nodiscard]] auto cross_covariance_at(std::span<const Rational> x, std::span<const Rational> y,
                                       std::int64_t lag, bool biased) -> Result<Rational>;
[[nodiscard]] auto cross_covariance(std::span<const Rational> x, std::span<const Rational> y,
                                    std::size_t max_lag, bool biased) -> Result<std::vector<Rational>>;
[[nodiscard]] auto is_wss(std::span<const Rational> x, std::size_t max_lag) -> Result<bool>;
```

| Function | Behavior |
| :--- | :--- |
| `autocovariance_at` | `R(τ) = (1/D) Σ_t (x_t − x̄)(x_{t+τ} − x̄)` over the `N − τ` overlapping pairs, with `D = N` (biased) or `D = N − τ` (unbiased). `R(0)` with the biased normalisation is exactly the population variance. `domain_error` when the record is empty or `lag >= N`. |
| `autocovariance` | The sequence `R(0), R(1), ..., R(max_lag)` (exact). `domain_error` if `max_lag >= N`. |
| `autocorrelation` | `ρ(τ) = R(τ) / R(0)`, `τ = 0..max_lag`; always `ρ(0) = 1`. `division_by_zero` for a constant record (`R(0) = 0`); `domain_error` if `max_lag >= N`. |
| `cross_covariance_at` | `R_xy(lag) = (1/D) Σ_t (x_t − x̄)(y_{t+lag} − ȳ)` over the `|N − lag|` overlapping pairs, `D = N` (biased) or `D = N − |lag|` (unbiased). Positive lag leads `y`; negative lag leads `x`. `domain_error` on unequal lengths, an empty record, or `|lag| >= N`. (A normalised cross-correlation coefficient needs `sqrt(var_x var_y)` and is **not** exact over Q; this returns the cross-**covariance**.) |
| `cross_covariance` | The sequence `R_xy(0..max_lag)` (exact, non-negative lags). Fails as `cross_covariance_at`. |
| `is_wss` | **Diagnostic.** Splits the record into two equal halves and returns `true` iff their means agree exactly **and** their biased autocovariances agree exactly at every lag `0..max_lag` — the two defining features of wide-sense stationarity on the record in hand. It is a diagnostic of the **sample**, not a proof of the process law. `domain_error` when a half is empty or `max_lag` is too large for a half. |

## AR / MA / ARMA processes (autocorrelation & autocovariance exact over Q)

```cpp
enum class StabilityCertificate : std::uint8_t {
    stable,        // every root proven strictly outside the unit circle
    unstable,      // a rational root found on or inside the unit circle
    indeterminate  // no inside/on-circle rational root found, but roots remain unenumerated
};

[[nodiscard]] auto yule_walker_autocorrelation(std::span<const Rational> ar, std::size_t max_lag)
    -> Result<std::vector<Rational>>;
[[nodiscard]] auto ar_autocovariance(std::span<const Rational> ar, const Rational& sigma2,
                                     std::size_t max_lag) -> Result<std::vector<Rational>>;
[[nodiscard]] auto ma_autocovariance(std::span<const Rational> ma, const Rational& sigma2,
                                     std::size_t max_lag) -> Result<std::vector<Rational>>;
[[nodiscard]] auto ar_is_stationary(std::span<const Rational> ar) -> Result<StabilityCertificate>;
[[nodiscard]] auto ma_is_invertible(std::span<const Rational> ma) -> Result<StabilityCertificate>;
```

`StabilityCertificate` is the verdict of a root-location test against the unit
circle, with an honest third state (`indeterminate`) for irrational/complex roots
that cannot be enumerated exactly over Q.

| Function | Behavior |
| :--- | :--- |
| `yule_walker_autocorrelation` | The autocorrelation `ρ(0..max_lag)` of an AR(p) process `x_t = Σ_k φ_k x_{t−k} + w_t` from the Yule–Walker equations. `ρ(1..p)` solve the exact `p × p` rational system `ρ(k) = Σ_j φ_j ρ(|k−j|)` (with `ρ(0) = 1`), and `ρ(k)` for `k > p` is continued by the same recursion. Exact over Q; `ρ(0) = 1`. `ar` holds `φ_1..φ_p` (empty span ⇒ white noise: `ρ = [1, 0, 0, ...]`). `domain_error` if the Yule–Walker system is singular (a non-stationary parameterisation). |
| `ar_autocovariance` | The theoretical autocovariance `γ(0..max_lag)` with white-noise variance `σ^2`: `γ(0) = σ^2 / (1 − Σ_j φ_j ρ(j))` is the process variance, and `γ(k) = γ(0) ρ(k)`. Exact over Q. Fails as `yule_walker_autocorrelation`, or `division_by_zero` if the variance denominator vanishes. |
| `ma_autocovariance` | The theoretical autocovariance `γ(0..max_lag)` of an MA(q) process `x_t = w_t + Σ_{k=1}^q θ_k w_{t−k}`: `γ(τ) = σ^2 Σ_{k=0}^{q−τ} θ_k θ_{k+τ}` (`θ_0 = 1`) for `0 <= τ <= q`, and `γ(τ) = 0` for `τ > q`. Exact over Q. `ma` holds `θ_1..θ_q`. |
| `ar_is_stationary` | Stationarity from the roots of `Φ(z) = 1 − Σ_k φ_k z^k`: stationary iff every root lies strictly **outside** the unit circle. Rational roots are found and tested (`|num|` vs `den`); if every root of `Φ` is rational the verdict is definitive, otherwise unenumerated irrational/complex roots make it `indeterminate` (unless a rational root already proves it `unstable`). `ar` holds `φ_1..φ_p` (empty ⇒ white noise ⇒ `stable`). |
| `ma_is_invertible` | Invertibility from the roots of `Θ(z) = 1 + Σ_k θ_k z^k` against the unit circle, with the same exact-rational / `indeterminate` contract as `ar_is_stationary`. `ma` holds `θ_1..θ_q`. |

## Power spectral density (numerical — the Fourier boundary)

`S(f)` is the Fourier transform of the autocovariance (Wiener–Khinchin). This is
the one place the mathematics leaves `Q`, so these routines are **numerical**
(`double`) and return a sampled `Spectrum` over the one-sided grid `[0, 1/2]`.

```cpp
struct Spectrum {
    std::vector<double> frequencies;
    std::vector<double> values;   // value[k] is the density at frequencies[k]
};

[[nodiscard]] auto to_doubles(std::span<const Rational> xs) -> std::vector<double>;
[[nodiscard]] auto power_spectral_density(std::span<const double> autocov_onesided,
                                          std::size_t num_points) -> Result<Spectrum>;
[[nodiscard]] auto power_spectral_density(std::span<const Rational> autocov_onesided,
                                          std::size_t num_points) -> Result<Spectrum>;
[[nodiscard]] auto arma_spectral_density(std::span<const Rational> ar, std::span<const Rational> ma,
                                         double sigma2, std::size_t num_points) -> Result<Spectrum>;
```

| Function | Behavior |
| :--- | :--- |
| `to_doubles` | Convert exact rational values to `double` (`num/den`) — the point at which exact analysis hands off to the numerical spectral routines. |
| `power_spectral_density` (`double`) | `S(f) = R(0) + 2 Σ_{τ=1}^m R(τ) cos(2π f τ)`, sampled at `num_points` frequencies on the one-sided grid `f_k = (1/2) k / (num_points − 1)`. **Numerical.** For white noise (`R = [σ^2, 0, ...]`) `S` is flat at `σ^2`. `domain_error` if the autocovariance is empty or `num_points < 2`. |
| `power_spectral_density` (`Rational`) | Convenience overload taking an exact rational autocovariance and evaluating `S(f)` numerically (via `to_doubles`). |
| `arma_spectral_density` | The closed form `S(f) = σ^2 |Θ(e^{-iω})|^2 / |Φ(e^{-iω})|^2`, `ω = 2π f`, with `Φ(x) = 1 − Σ_k φ_k x^k` (AR) and `Θ(x) = 1 + Σ_k θ_k x^k` (MA), evaluated **numerically** on the one-sided grid. `ar`/`ma` may be empty (pure MA / pure AR / white noise). `domain_error` if `num_points < 2`; `not_implemented` if `Φ` vanishes on the grid (a root on the unit circle — a non-stationary AR part). |

## Error model

| Condition | Error |
| :--- | :--- |
| `stationary_distribution` on a non-square / non-stochastic `P`, or a singular system | `MathError::domain_error` |
| `n_step_transition` / `is_irreducible` on a non-square (or empty) `P` | `MathError::domain_error` |
| `is_aperiodic` on a reducible chain (period is per-class) | `MathError::domain_error` |
| `fundamental_matrix` / `mean_first_passage_times` when `π` does not exist or the core matrix is singular | `MathError::domain_error` |
| `ergodic_mean` with `values.size()` ≠ number of states | `MathError::domain_error` |
| `ctmc_stationary_distribution` on a non-generator or singular system | `MathError::domain_error` |
| `autocovariance(_at)` / `cross_covariance(_at)` with an empty record, unequal lengths, or `lag >= N` | `MathError::domain_error` |
| `autocorrelation` on a constant record (`R(0) = 0`) | `MathError::division_by_zero` |
| `is_wss` with an empty half or `max_lag` too large for a half | `MathError::domain_error` |
| `yule_walker_autocorrelation` / `ar_autocovariance` with a singular Yule–Walker system | `MathError::domain_error` |
| `ar_autocovariance` when the variance denominator `1 − Σ_j φ_j ρ(j)` vanishes | `MathError::division_by_zero` |
| `power_spectral_density` with an empty autocovariance or `num_points < 2` | `MathError::domain_error` |
| `arma_spectral_density` with `num_points < 2` | `MathError::domain_error` |
| `arma_spectral_density` when `Φ` vanishes on the grid (unit-circle AR root) | `MathError::not_implemented` |
| Any `int64` numerator/denominator computation wraps on an exact path | `MathError::overflow` |

`is_stochastic` / `is_generator` are total predicates over well-formed matrices —
a non-square input is `false`, not an error. `ar_is_stationary` /
`ma_is_invertible` never error on the honesty boundary itself: an unenumerated
irrational root is the value `StabilityCertificate::indeterminate`, not a failure.

## Worked examples

From `tests/stochastic_tests.cpp` (exact paths compared with `==`; numerical PSD
checked to a tolerance):

```cpp
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.stats;
import nimblecas.stochastic;
using nimblecas::Matrix;
using nimblecas::Rational;
using nimblecas::StabilityCertificate;

auto rat = [](std::int64_t n, std::int64_t d) { return Rational::make(n, d).value(); };
auto mat = [](std::vector<std::vector<Rational>> r) { return Matrix::from_rows(std::move(r)).value(); };
auto ints = [](std::vector<std::int64_t> vs) {
    std::vector<Rational> out;
    for (auto v : vs) { out.push_back(Rational::from_int(v)); }
    return out;
};

// --- Markov chain: exact stationary distribution ---
// P = [[3/4, 1/4], [1/2, 1/2]] -> pi = [2/3, 1/3].
const auto p = mat({{rat(3, 4), rat(1, 4)}, {rat(1, 2), rat(1, 2)}});
auto pi = nimblecas::stationary_distribution(p).value();     // [2/3, 1/3]  (pi P == pi)

nimblecas::n_step_transition(p, 0).value();                  // identity  (P^0)
nimblecas::is_stochastic(nimblecas::n_step_transition(p, 3).value()).value();  // true
nimblecas::is_irreducible(p).value();                        // true
nimblecas::is_aperiodic(p).value();                          // true

// A pure 3-cycle is irreducible but periodic (period 3).
const auto cyc = mat({{Rational{}, Rational::from_int(1), Rational{}},
                      {Rational{}, Rational{}, Rational::from_int(1)},
                      {Rational::from_int(1), Rational{}, Rational{}}});
nimblecas::is_aperiodic(cyc).value();                        // false

// Mean recurrence 1/pi_i and first-passage times (exact).
auto m = nimblecas::mean_first_passage_times(p).value();
m.at(0, 0);   // 3/2  (= 1/(2/3))       m.at(1, 1);   // 3   (= 1/(1/3))
m.at(0, 1);   // 4    (E[hit 1 | 0])    m.at(1, 0);   // 2   (E[hit 0 | 1])

// Ergodic (space) average of f = [3, 0]:  2/3*3 + 1/3*0 = 2.
nimblecas::ergodic_mean(p, std::span<const Rational>{ints({3, 0})}).value();  // 2

// --- CTMC: generator Q = [[-1, 1], [2, -2]] -> pi = [2/3, 1/3] ---
const auto q = mat({{rat(-1, 1), rat(1, 1)}, {rat(2, 1), rat(-2, 1)}});
nimblecas::is_generator(q).value();                          // true
nimblecas::ctmc_stationary_distribution(q).value();          // [2/3, 1/3]

// --- WSS sample analysis (exact over Q) ---
const auto x = ints({1, 2, 3});
nimblecas::autocovariance_at(std::span<const Rational>{x}, 0, true).value();  // 2/3 (= population var)
// Record {1,2,1,2} has genuine lag-1 structure: rho(1) = -3/4.
const auto y = ints({1, 2, 1, 2});
nimblecas::autocorrelation(std::span<const Rational>{y}, 1, true).value()[1]; // -3/4

// Cross-covariance (biased lag-0 == population covariance).
nimblecas::cross_covariance_at(std::span<const Rational>{ints({1,2,3})},
                               std::span<const Rational>{ints({3,2,1})}, 0, true).value();  // -2/3

// is_wss is a DIAGNOSTIC of the record, not the process law.
nimblecas::is_wss(std::span<const Rational>{ints({1,2,1,2,1,2,1,2})}, 1).value();  // true
nimblecas::is_wss(std::span<const Rational>{ints({1,1,1,1,5,5,5,5})}, 1).value();  // false (mean shift)

// --- AR(1) with phi = 1/2: rho(k) = (1/2)^k exactly ---
const auto ar = std::vector<Rational>{rat(1, 2)};
auto rho = nimblecas::yule_walker_autocorrelation(std::span<const Rational>{ar}, 4).value();
// rho == [1, 1/2, 1/4, 1/8, 1/16]
auto gamma = nimblecas::ar_autocovariance(std::span<const Rational>{ar},
                                          Rational::from_int(1), 2).value();
// gamma(0) == 4/3  (= 1 / (1 - 1/4)),  gamma(1) == 2/3

// Stationarity certificate: stable / unstable / honestly indeterminate.
nimblecas::ar_is_stationary(std::span<const Rational>{ar}).value();          // stable  (root 2 outside)
nimblecas::ar_is_stationary(std::span<const Rational>{{Rational::from_int(2)}}).value();  // unstable
// Phi(z) = 1 - z^2/2 has roots +-sqrt(2): irrational, unenumerated over Q.
nimblecas::ar_is_stationary(std::span<const Rational>{{Rational{}, rat(1, 2)}}).value();  // indeterminate

// --- MA(1) theta = 1/3, sigma^2 = 1: gamma(0)=10/9, gamma(1)=1/3, gamma(2)=0 ---
const auto ma = std::vector<Rational>{rat(1, 3)};
nimblecas::ma_autocovariance(std::span<const Rational>{ma}, Rational::from_int(1), 3).value();
nimblecas::ma_is_invertible(std::span<const Rational>{ma}).value();          // stable (|theta| < 1)

// --- Power spectral density (NUMERICAL) ---
// White-noise autocovariance R = [1, 0, 0]: S(f) flat at 1.
auto spec = nimblecas::power_spectral_density(std::span<const Rational>{ints({1, 0, 0})}, 5).value();
// spec.values are all ~ 1.0
// ARMA closed form with no AR/MA terms and sigma^2 = 2 is flat at 2.
const std::vector<Rational> none;
nimblecas::arma_spectral_density(std::span<const Rational>{none},
                                 std::span<const Rational>{none}, 2.0, 4).value();  // ~ 2.0 flat
```

## See also

- [`nimblecas.matrix`](matrix.md) — the exact-`Rational` linear-algebra backbone
  (solve / inverse) behind every Markov-chain result.
- [`nimblecas.roots`](roots.md) — the rational-root finder that decides the
  AR/MA stability certificates (and honestly leaves irrational roots
  unenumerated).
- [`nimblecas.stats`](stats.md) — the exact mean / variance / covariance the WSS
  sample analysis builds on.
- [`nimblecas.sde`](sde.md), [`nimblecas.montecarlo`](montecarlo.md), and
  [`nimblecas.mcmc`](mcmc.md) — the sampling / simulation modules this analytic
  module is deliberately distinct from.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field every exact
  result computes in.
- [Documentation hub](../Index.md)
