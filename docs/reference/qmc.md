# `nimblecas.qmc` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/qmc/qmc.cppm`

The low-discrepancy counterpart of [`nimblecas.montecarlo`](montecarlo.md)
(ROADMAP §7.8). Where plain Monte Carlo scatters pseudorandom points and
converges at the dimension-independent `O(1/√N)` rate, **quasi**-Monte Carlo lays
down a *deterministic* low-discrepancy point set whose extra uniformity gives,
for smooth low-dimensional integrands, the faster `O((log N)^d / N)` rate. This
module has a sharp, API-enforced honesty boundary between what is exact and what
is numerical:

- **The low-discrepancy POINTS are EXACT.** A radical-inverse (Van der Corput)
  of an integer in base `b` is an exact reduced fraction; Halton / Hammersley
  stack those per dimension; Sobol' direction numbers are dyadic, so its
  coordinates are exact multiples of `2⁻³²`; and a rank-1 lattice point
  `n·z/N mod 1` is an exact fraction with denominator `N`. Each is offered on an
  exact [`Rational`](ratpoly.md) path (`*_rational`) **plus** a `double` view.
- **QMC INTEGRATION of a general `f` is NUMERICAL** — the equal-weight average of
  `f` over the points is an `f64` estimate. Only when `f` is itself
  rational-valued on rational points is the average an exact `Rational`; that
  path is offered separately (`qmc_integrate_exact`), and "exact" there refers to
  the *arithmetic*, not to the integral being resolved (a finite average still
  only approximates `∫`).
- **The RQMC error is a STATISTICAL (variance) estimate**, not a deterministic
  bound. The deterministic Koksma–Hlawka bound `|error| ≤ V(f)·D*(P)` needs the
  Hardy–Krause total variation `V(f)`, which this module does **not** compute —
  so it reports the empirical RQMC standard error and says so.
- **No universal-superiority claim is made.** QMC beats plain MC only for
  integrands that are smooth and of low effective dimension; for rough or
  high-dimensional `f` the `(log N)^d` factor dominates and MC can be competitive
  or better.

Every failure travels the railway (`Result<T>` / `MathError`) — nothing throws.
Point routines are stateless pure functions of their index, so any partition of
the index range reproduces the whole set; the RQMC randomizations are seeded from
[`nimblecas.rng`](rng.md), so equal seeds reproduce equal results, bit-for-bit.

```cpp
import nimblecas.qmc;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md) (the exact `Rational`
field), and [`rng`](rng.md) (`Rng::seeded` / `split` / `next_unit` for the RQMC
randomizations). All exported symbols live in namespace `nimblecas`.

## Low-discrepancy points — exact by construction

Each generator comes as a pair: an exact `*_rational` path returning
`Rational`-valued coordinates, and a `double` view of the same construction.
Indices are `std::uint64_t`; dimensions are `std::size_t`.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `rational_to_double` | `[[nodiscard]] auto rational_to_double(const Rational& r) noexcept -> double` | The `num/den` view of an exact `Rational` in IEEE-754. A pure, non-failing utility for taking a `double` of any exact coordinate or discrepancy. |
| `van_der_corput_rational` | `[[nodiscard]] auto van_der_corput_rational(std::uint64_t n, std::uint64_t base) -> Result<Rational>` | Radical inverse `φ_b(n)`: reflect the base-`b` digits of `n` about the radix point, as an **exact** reduced fraction in `[0, 1)`. Base 2: `φ(1)=1/2`, `φ(2)=1/4`, `φ(3)=3/4`, `φ(4)=1/8`, … `domain_error` if `base < 2`; `overflow` if the denominator `b^(#digits)` exceeds `int64` (use the `double` view for those indices). |
| `van_der_corput` | `[[nodiscard]] auto van_der_corput(std::uint64_t n, std::uint64_t base) -> Result<double>` | Numerical `double` view of the same radical inverse; accumulates the fraction directly in floating point, so it **never overflows** and works for every index. `domain_error` only if `base < 2`. |
| `halton_point_rational` | `[[nodiscard]] auto halton_point_rational(std::uint64_t n, std::size_t dimension) -> Result<std::vector<Rational>>` | Halton point of index `n` using the first `dimension` primes as the pairwise-coprime per-dimension bases: coordinate `j` is `φ_{p_j}(n)`. **Exact.** `domain_error` if `dimension == 0`; `overflow` if any coordinate's denominator exceeds `int64`. |
| `halton_point` | `[[nodiscard]] auto halton_point(std::uint64_t n, std::size_t dimension) -> Result<std::vector<double>>` | The `double` view of the Halton point; per-coordinate `van_der_corput`. |
| `hammersley_point_rational` | `[[nodiscard]] auto hammersley_point_rational(std::uint64_t i, std::uint64_t total_n, std::size_t dimension) -> Result<std::vector<Rational>>` | Hammersley point `i` of a set of size `total_n`: coordinate 0 is the **exact** fraction `i/total_n`, coordinates `1..d-1` are Halton coordinates `φ_{p_k}(i)` with the first `d-1` primes. Because coordinate 0 needs the set size, Hammersley is a **finite** point set (unlike the extensible Halton). `domain_error` if `dimension == 0` or `total_n == 0`; `overflow` on an `int64` denominator boundary. |
| `hammersley_point` | `[[nodiscard]] auto hammersley_point(std::uint64_t i, std::uint64_t total_n, std::size_t dimension) -> Result<std::vector<double>>` | The `double` view of the Hammersley point. |
| `sobol_point_rational` | `[[nodiscard]] auto sobol_point_rational(std::uint64_t n, std::size_t dimension) -> Result<std::vector<Rational>>` | Sobol' point `n` (a base-2 digital net from integer direction numbers). Coordinates are **dyadic-exact** — each an exact multiple of `2⁻³²`. The built-in Joe–Kuo / Bratley–Fox direction table covers **dimensions 1..8**; a larger dimension yields `domain_error` (honest: no fabricated direction numbers). `n` must be `< 2³²`; a larger `n` yields `domain_error`. `n == 0` maps to the origin. |
| `sobol_point` | `[[nodiscard]] auto sobol_point(std::uint64_t n, std::size_t dimension) -> Result<std::vector<double>>` | The `double` view of the Sobol' point (each integer coordinate times `2⁻³²`). |
| `lattice_point_rational` | `[[nodiscard]] auto lattice_point_rational(std::uint64_t n, std::uint64_t total_n, std::span<const std::uint64_t> z) -> Result<std::vector<Rational>>` | Rank-1 lattice point `n` of a set of size `total_n` with generating vector `z` (one entry per dimension — `z.size()` **is** the dimension): coordinate `j` is the **exact** fraction `((n·z_j) mod total_n)/total_n`. `domain_error` if `total_n == 0` or `z` is empty; `overflow` if `total_n > 2³²` (beyond which the modular product would exceed 64 bits). |
| `lattice_point` | `[[nodiscard]] auto lattice_point(std::uint64_t n, std::uint64_t total_n, std::span<const std::uint64_t> z) -> Result<std::vector<double>>` | The `double` view of the lattice point (computed from the exact path, then viewed). |

## QMC integration

Two integrand aliases fix the numerical / exact split:

```cpp
using ScalarField   = std::function<double(std::span<const double>)>;
using RationalField = std::function<Result<Rational>(std::span<const Rational>)>;
```

A `ScalarField` maps a point in `[0, 1]^d` to a real value; a `RationalField`
maps an exact rational point to a `Result<Rational>`, so an integrand undefined
at a point can report it on the railway.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `qmc_integrate` | `[[nodiscard]] auto qmc_integrate(const ScalarField& f, std::size_t dimension, std::uint64_t N) -> Result<double>` | Estimate `∫_{[0,1]^d} f` as the equal-weight average of `f` over the first `N` Halton points (indices `1..N`; index 0 = origin is skipped). **NUMERICAL.** `domain_error` if `dimension == 0` or `N == 0`; propagates any error from generating a Halton coordinate. |
| `qmc_integrate_exact` | `[[nodiscard]] auto qmc_integrate_exact(const RationalField& f, std::size_t dimension, std::uint64_t N) -> Result<Rational>` | Exact variant: when `f` is rational-valued on the exact Halton points, the equal-weight average is an exact `Rational` ("exact" is the arithmetic, not the integral being resolved). `domain_error` if `dimension == 0` or `N == 0`; `overflow` if `N` exceeds `int64` or from the exact rational sum; propagates any error `f` returns. |

## Randomized QMC (RQMC)

`rqmc_integrate` reintroduces controlled randomness — a Cranley–Patterson
rotation — so that a *statistical* error can be estimated from independent
randomizations. This is **not** the deterministic Koksma–Hlawka bound.

```cpp
struct RqmcResult {
    double        estimate;        // mean of the per-replication QMC averages (UNBIASED for ∫ f)
    double        error_estimate;  // standard error of that mean (sqrt(var / replications))
    std::uint64_t points_used;     // total integrand evaluations = N * replications
    std::uint64_t replications;    // number of independent randomizations
};
```

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `rqmc_integrate` | `[[nodiscard]] auto rqmc_integrate(const ScalarField& f, std::size_t dimension, std::uint64_t N, std::uint64_t replications, std::uint64_t seed) -> Result<RqmcResult>` | For each replication a uniform random shift `s ∈ [0,1)^d` (drawn from an `Rng` child stream `base.split(r)`) is added mod 1 to every Halton point, then `f` is averaged. Each replication is an **unbiased** estimator of `∫ f`, so the mean over replications is unbiased and their spread yields the standard error (Bessel-corrected variance, divided by `replications`). `domain_error` if `dimension == 0`, `N == 0`, or `replications < 2` (two are needed for a variance). |

## Iterative / adaptive QMC

`adaptive_qmc` progressively refines an RQMC estimate until a tolerance is met or
a hard evaluation budget is exhausted. It **always terminates**.

```cpp
struct AdaptiveResult {
    double        estimate;
    double        error_estimate;
    std::uint64_t points_used;   // cumulative integrand evaluations across all refinement levels
    bool          converged;     // true iff error_estimate <= tol before the budget ran out
};
```

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `adaptive_qmc` | `[[nodiscard]] auto adaptive_qmc(const ScalarField& f, std::size_t dimension, double tol, std::uint64_t max_points, std::uint64_t replications, std::uint64_t seed) -> Result<AdaptiveResult>` | Refine by **doubling** the Halton point count `N` (starting at 64). The Halton sequence is extensible, so each level's points extend the previous level's — earlier points are conceptually reused. At each level `replications` randomizations give a running estimate and an RQMC standard error; refinement stops as soon as the error drops to `tol` or the cumulative budget `max_points` is reached. **Always terminates** (`N` doubles and the budget is a hard cap; growth also stops before `N` would overflow). `domain_error` if `dimension == 0`, `tol < 0`, `max_points == 0`, or `replications < 2`. |

## Discrepancy diagnostic (Warnock L2 star discrepancy)

The `L2` star discrepancy of a point set measures its non-uniformity — a smaller
value means a more uniform set. Warnock's closed form is a finite sum of products
of coordinates, so it has both a numerical and an exact path.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `l2_star_discrepancy` | `[[nodiscard]] auto l2_star_discrepancy(std::span<const std::vector<double>> points, std::size_t dimension) -> Result<double>` | `L2` star discrepancy via Warnock's closed form. **NUMERICAL** (`double`). Each point must have exactly `dimension` coordinates in `[0, 1]`. `domain_error` on an empty set, `dimension == 0`, or a point of the wrong size. |
| `l2_star_discrepancy_squared_exact` | `[[nodiscard]] auto l2_star_discrepancy_squared_exact(std::span<const std::vector<Rational>> points, std::size_t dimension) -> Result<Rational>` | The **exact SQUARED** `L2` star discrepancy of exact rational points (Warnock's formula is a finite sum of products of rationals, hence exact). The square root is generally irrational, so the squared value is returned — take `rational_to_double(...)` then `std::sqrt` for the norm. `domain_error` on an empty set / dimension mismatch; `overflow` (the products grow as `denom^(2·dimension)` and quickly exceed `int64` — the numerical version has no such limit). |

## Error model

| Condition | Error |
| :--- | :--- |
| `van_der_corput_rational` / `van_der_corput` with `base < 2` | `MathError::domain_error` |
| `van_der_corput_rational` denominator `b^(#digits)` exceeds `int64` | `MathError::overflow` |
| `halton_point*` / `hammersley_point*` / `qmc_integrate*` with `dimension == 0` | `MathError::domain_error` |
| `hammersley_point*` with `total_n == 0` | `MathError::domain_error` |
| `sobol_point*` with `dimension` outside `1..8`, or `n ≥ 2³²` | `MathError::domain_error` |
| `lattice_point*` with `total_n == 0` or empty `z` | `MathError::domain_error` |
| `lattice_point*` with `total_n > 2³²` | `MathError::overflow` |
| `qmc_integrate*` with `N == 0` | `MathError::domain_error` |
| `qmc_integrate_exact` with `N` exceeding `int64`, or an exact-sum overflow | `MathError::overflow` |
| `rqmc_integrate` with `dimension == 0`, `N == 0`, or `replications < 2` | `MathError::domain_error` |
| `adaptive_qmc` with `dimension == 0`, `tol < 0`, `max_points == 0`, or `replications < 2` | `MathError::domain_error` |
| `l2_star_discrepancy*` on an empty set, `dimension == 0`, or a point of the wrong size | `MathError::domain_error` |
| `l2_star_discrepancy_squared_exact` when an exact product exceeds `int64` | `MathError::overflow` |
| Any exact `Rational` denominator / numerator computation wraps `int64` | `MathError::overflow` |

Every fallible path returns `Result<T>`; nothing throws. `rational_to_double` is
total and never fails. Any error from a user-supplied `RationalField` propagates
verbatim through `qmc_integrate_exact`. A budget-exhausted `adaptive_qmc` is
**success** with `converged == false`, not an error.

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.rng;
import nimblecas.qmc;
using namespace nimblecas;

auto rat = [](std::int64_t num, std::int64_t den) {  // exact num/den
    return Rational::make(num, den).value();
};

// ── Exact low-discrepancy points ───────────────────────────────────────────
// Van der Corput base 2: φ_2(1..4) = 1/2, 1/4, 3/4, 1/8 — EXACT rationals.
van_der_corput_rational(1, 2).value();   // 1/2
van_der_corput_rational(3, 2).value();   // 3/4
van_der_corput_rational(3, 1).error();   // domain_error (base < 2)

// Halton (bases 2, 3): point 1 = (1/2, 1/3), point 2 = (1/4, 2/3).
halton_point_rational(1, 2).value();     // {1/2, 1/3}
halton_point_rational(2, 2).value();     // {1/4, 2/3}
halton_point_rational(1, 0).error();     // domain_error (dimension 0)

// Hammersley(i=1, total=4, d=2) = (i/total = 1/4, φ_2(1) = 1/2).
hammersley_point_rational(1, 4, 2).value();  // {1/4, 1/2}

// Sobol' dimension 1, index 1 -> 1/2 exactly; every coordinate is dyadic.
sobol_point_rational(1, 2).value();      // dim 0 == 1/2, all denominators = 2^k
sobol_point_rational(1, 9).error();      // domain_error (dimension > 8)

// Rank-1 lattice, z = {1}, N = 4: n = 0..3 -> 0, 1/4, 1/2, 3/4 exactly.
const std::array<std::uint64_t, 1> z{1};
lattice_point_rational(2, 4, z).value(); // {1/2}
lattice_point_rational(0, 0, z).error(); // domain_error (total_n 0)

// ── QMC integration ────────────────────────────────────────────────────────
// The equal-weight average of the constant 1 is exactly 1 in every dimension.
qmc_integrate([](std::span<const double>) { return 1.0; }, 2, 500).value();  // ≈ 1

// Exact-rational average of φ_2 over Halton points 1..4:
// (1/2 + 1/4 + 3/4 + 1/8)/4 = (13/8)/4 = 13/32.
qmc_integrate_exact(
    [](std::span<const Rational> x) -> Result<Rational> { return x[0]; },
    1, 4).value();                       // 13/32 exactly

// ∫_0^1 x dx = 1/2 (numerical, within QMC error).
qmc_integrate([](std::span<const double> x) { return x[0]; }, 1, 4096).value();  // ≈ 0.5

// ∫∫_{[0,1]^2} x·y dx dy = 1/4.
qmc_integrate([](std::span<const double> p) { return p[0] * p[1]; }, 2, 4096).value();  // ≈ 0.25

// Refinement: a finer grid has smaller error than a coarser one for smooth ∫x.
qmc_integrate([](std::span<const double> x) { return x[0]; }, 1, 64);    // coarse
qmc_integrate([](std::span<const double> x) { return x[0]; }, 1, 4096);  // finer, closer to 1/2

// Domain guards.
qmc_integrate([](std::span<const double> x) { return x[0]; }, 0, 100).error();  // dimension 0
qmc_integrate([](std::span<const double> x) { return x[0]; }, 1, 0).error();    // N == 0

// ── Randomized QMC: a STATISTICAL error, not a Koksma–Hlawka bound ─────────
auto r = rqmc_integrate([](std::span<const double> x) { return x[0]; },
                        /*dimension=*/1, /*N=*/512, /*replications=*/16, /*seed=*/4242).value();
r.estimate;         // ≈ 0.5 (unbiased mean of ∫x)
r.error_estimate;   // > 0   (empirical standard error — the true value lies within a few of these)
r.points_used;      // 512 * 16
// replications < 2 cannot form a variance.
rqmc_integrate([](std::span<const double> x) { return x[0]; }, 1, 10, 1, 1).error();  // domain_error

// ── Adaptive refinement: always terminates ─────────────────────────────────
auto a = adaptive_qmc([](std::span<const double> x) { return x[0]; },
                      /*dimension=*/1, /*tol=*/1e-4, /*max_points=*/200000,
                      /*replications=*/8, /*seed=*/99).value();
a.estimate;    // ≈ 0.5
a.points_used; // > 0; a.converged implies a.error_estimate <= tol
adaptive_qmc([](std::span<const double> x) { return x[0]; }, 1, -1.0, 1000, 4, 1).error();  // tol < 0
adaptive_qmc([](std::span<const double> x) { return x[0]; }, 1, 1e-3, 0, 4, 1).error();     // budget 0

// ── L2 star discrepancy ────────────────────────────────────────────────────
// 1-D set {0, 1/2}: exact Warnock squared discrepancy = 1/12.
std::vector<std::vector<Rational>> pts{{rat(0, 1)}, {rat(1, 2)}};
l2_star_discrepancy_squared_exact(pts, 1).value();   // 1/12 exactly

// The numerical version returns the square root of the same value.
std::vector<std::vector<double>> dpts{{0.0}, {0.5}};
l2_star_discrepancy(dpts, 1).value();                // ≈ sqrt(1/12)
l2_star_discrepancy_squared_exact(pts, 0).error();   // domain_error (dimension 0)
```

## See also

- [`nimblecas.montecarlo`](montecarlo.md) — the pseudorandom Monte Carlo sibling
  this module accelerates for smooth, low-dimensional integrands.
- [`nimblecas.rng`](rng.md) — the counter-based RNG substrate (`Rng::seeded`,
  `split`, `next_unit`) the RQMC randomizations draw their Cranley–Patterson
  shifts from.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field the
  low-discrepancy coordinates and exact discrepancy live in.
- [`nimblecas.stats`](stats.md) — the exact-over-`Q` descriptive statistics; the
  RQMC standard error is the numerical analogue of its variance.
- [`nimblecas.integrate`](integrate.md) — exact symbolic integration over `Q(x)`,
  the algebraic counterpart to numerical QMC integration.
- [Documentation hub](../Index.md)
