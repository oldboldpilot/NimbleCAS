# `nimblecas.control` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/control/control.cppm`

Classical **linear control systems** built directly on the exact rational
substrate (ROADMAP §7). A SISO plant is either a
[`TransferFunction`](#transferfunction--a-siso-rational-transfer-function-numsdens-over-q)
`num(s)/den(s)` — a ratio of exact [`RationalPoly`](ratpoly.md)s — or a
[`StateSpace`](#statespace--an-exact-rational-linear-system) model
`x' = A x + B u, y = C x + D u` over the [`Matrix`](matrix.md) field. The
algebra of interconnections (series/parallel/feedback), the two conversions
`tf_to_ss`/`ss_to_tf`, the Kalman controllability/observability tests, the whole
family of stability criteria (Routh–Hurwitz, Hurwitz determinant, Kharitonov,
Lyapunov, Schur/Jury), and the bilinear (Tustin) transform are **all exact over
`Q`** — no floating point ever enters them.

**The honesty boundary** (documented and true). Only the parts that are
irreducibly transcendental are numerical, and they are separated cleanly:

- **Exact over `Q`:** `TransferFunction`/`StateSpace` algebra, `tf_to_ss` /
  `ss_to_tf`, `controllability_matrix` / `observability_matrix` and their
  Kalman-rank predicates, `is_stable_continuous` (Routh–Hurwitz, reusing
  [`dynamics`](dynamics.md) via a companion matrix), `is_stable_discrete`
  (Schur/Jury, decided exactly by an exact Möbius map of the unit disc to the
  open left half-plane), the Hurwitz-determinant / Kharitonov / Lyapunov tests,
  `bilinear_c2d` / `bilinear_d2c`, and `evaluate_exact` (`H` at a
  Gaussian-rational point, staying in `Q + Qi`).
- **Exact only for the rational part:** `poles()` and `zeros()` extract only the
  **rational** roots (via [`roots`](roots.md)); any irrational or complex root is
  reported as a **count** of not-fully-extracted poles/zeros
  ([`RootReport::unextracted`](#rootreport--exact-rational-roots-with-an-honest-remainder-count)),
  never silently dropped or approximated.
- **Numerical (double precision):** `bode()`, `nyquist()`, `nyquist_criterion`'s
  encirclement count `N`, and `stability_margins`. These evaluate `H(iω)` in
  `std::complex<double>` and take `20·log10|H|` / `atan2`.

```cpp
import nimblecas.control;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), [`matrix`](matrix.md),
[`dynamics`](dynamics.md), [`roots`](roots.md), and [`complex`](complex.md).

## The overflow contract

Following the rest of the engine, every exact step is threaded on the
[`Result`](core.md) railway (Rule 32): an `int64` numerator/denominator that
would overflow surfaces as `MathError::overflow` (never a silent wrap), a zero
denominator as `MathError::division_by_zero`, and a dimension or well-formedness
violation as `MathError::domain_error` — never an exception. The three
double-precision routines (`bode`, `nyquist`, `stability_margins`) return their
plain result types directly and have no railway: an imaginary-axis pole simply
produces an `inf`/`nan` sample.

## Reporting structs

### `RootReport` — exact rational roots with an honest remainder count

The exact-rational roots of a polynomial together with a count of the roots that
could **not** be extracted over `Q` (irrational or complex).

| Member | Type | Meaning |
| :--- | :--- | :--- |
| `rational` | `std::vector<std::pair<Rational, std::int64_t>>` | Distinct rational roots, each paired with its multiplicity (`>= 1`). |
| `unextracted` | `std::int64_t` | Number of roots (with multiplicity) that are irrational/complex and therefore **not** represented here: `degree(p) − Σ multiplicities`. Zero means the spectrum is fully rational. |
| `fully_extracted()` | `auto fully_extracted() const noexcept -> bool` | `true` iff `unextracted == 0` (the report is complete). |

### `BodePoint` / `NyquistPoint` — NUMERICAL frequency-response samples

| Struct | Members | Meaning |
| :--- | :--- | :--- |
| `BodePoint` | `double omega`, `double magnitude_db`, `double phase_deg` | One Bode sample: angular frequency `ω` (rad/s), magnitude `20·log10|H(iω)|` in dB, and phase `arg H(iω)` in degrees. Double precision. |
| `NyquistPoint` | `double omega`, `double re`, `double im` | One Nyquist sample: `ω` and the point `H(iω) = re + im·i`. Double precision. |

## `TransferFunction` — a SISO rational transfer function `num(s)/den(s)` over `Q`

Both numerator and denominator are exact [`RationalPoly`](ratpoly.md), so every
interconnection is exact rational polynomial arithmetic. The variable is `s`
(continuous) or `z` (discrete) depending on how the object is used; the algebra
is identical, so the class carries no domain tag.

### Construction and accessors

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `make` | `static auto make(RationalPoly num, RationalPoly den) -> Result<TransferFunction>` | Build `num/den`. Fails `domain_error` if the denominator is the zero polynomial. |
| `numerator` | `auto numerator() const noexcept -> const RationalPoly&` | The numerator. |
| `denominator` | `auto denominator() const noexcept -> const RationalPoly&` | The denominator. |
| `is_proper` | `auto is_proper() const noexcept -> bool` | `deg(num) <= deg(den)`: realisable without pure differentiators. |
| `is_strictly_proper` | `auto is_strictly_proper() const noexcept -> bool` | `deg(num) < deg(den)`: no direct feedthrough term. |
| `relative_degree` | `auto relative_degree() const noexcept -> std::int64_t` | `deg(den) − deg(num)`. Positive for a strictly proper plant. |
| `to_string` | `auto to_string(std::string_view var = "s") const -> std::string` | Renders `"(num) / (den)"` in the named variable. |

### Interconnections (exact)

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `series` | `auto series(const TransferFunction& o) const -> Result<TransferFunction>` | Cascade `G1·G2 = (n1 n2)/(d1 d2)`. |
| `parallel` | `auto parallel(const TransferFunction& o) const -> Result<TransferFunction>` | `G1 + G2 = (n1 d2 + n2 d1)/(d1 d2)`. |
| `feedback` | `auto feedback(const TransferFunction& h, bool negative_feedback = true) const -> Result<TransferFunction>` | Closed loop of forward plant `*this` (`G`) with feedback element `h` (`H`): `G/(1 ± G H) = (n1 d2)/(d1 d2 ± n1 n2)`. Negative feedback (default) uses `+`; positive uses `−`. Fails `domain_error` if the resulting denominator is the zero polynomial. |
| `unity_feedback` | `auto unity_feedback(bool negative_feedback = true) const -> Result<TransferFunction>` | The unity-feedback special case `H = 1`: `G/(1 ± G)`. |

### Analysis (exact, except the extraction boundary in `poles`/`zeros`)

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `poles` | `auto poles() const -> Result<RootReport>` | Roots of the denominator. **Exact for rational roots**; irrational/complex poles are counted in `RootReport::unextracted`. A constant denominator has no poles (empty report). |
| `zeros` | `auto zeros() const -> Result<RootReport>` | Roots of the numerator, same exact/rational semantics as `poles`. The zero transfer function (`num == 0`) has no well-defined finite zeros → `domain_error`. |
| `dc_gain` | `auto dc_gain() const -> Result<Rational>` | Exact DC gain `H(0) = num(0)/den(0)`. Fails `division_by_zero` when the plant has a pole at the origin (`den(0) == 0`). |
| `equivalent` | `auto equivalent(const TransferFunction& o) const -> Result<bool>` | Are `*this` and `o` the **same** rational function? Compared exactly by cross-multiplication (`n1 d2 == n2 d1`), so scaled representations compare equal. |

## `StateSpace` — an exact-rational linear system

`x' = A x + B u, y = C x + D u`, with `A` (`n×n`), `B` (`n×m`), `C` (`p×n`),
`D` (`p×m`), all over `Q`. Every observer below is exact; the
controllability/observability ranks use `Matrix::rank` (exact over `Q`).

### Construction and accessors

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `make` | `static auto make(Matrix a, Matrix b, Matrix c, Matrix d) -> Result<StateSpace>` | Assemble `(A, B, C, D)` with consistent dimensions (`A` square; `B` rows `== n`; `C` cols `== n`; `D` rows `== C` rows and `D` cols `== B` cols), else `domain_error`. |
| `a` / `b` / `c` / `d` | `auto a() const noexcept -> const Matrix&` (and `b`, `c`, `d`) | The four matrices. |
| `n_states` | `auto n_states() const noexcept -> std::size_t` | `n` (`A.rows()`). |
| `n_inputs` | `auto n_inputs() const noexcept -> std::size_t` | `m` (`B.cols()`). |
| `n_outputs` | `auto n_outputs() const noexcept -> std::size_t` | `p` (`C.rows()`). |

### Structural analysis (exact over `Q`)

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `controllability_matrix` | `auto controllability_matrix() const -> Result<Matrix>` | The Kalman controllability matrix `[B  AB  A²B  …  Aⁿ⁻¹B]` (`n × n·m`). |
| `observability_matrix` | `auto observability_matrix() const -> Result<Matrix>` | The Kalman observability matrix `[C; CA; CA²; …; CAⁿ⁻¹]` (`n·p × n`). |
| `is_controllable` | `auto is_controllable() const -> Result<bool>` | `true` iff the controllability matrix has full rank `n` (exact rational rank). |
| `is_observable` | `auto is_observable() const -> Result<bool>` | `true` iff the observability matrix has full rank `n` (exact rational rank). |
| `is_asymptotically_stable` | `auto is_asymptotically_stable() const -> Result<bool>` | Continuous-time asymptotic stability of `x' = A x`: every eigenvalue of `A` in the open left half-plane, decided **exactly** via Routh–Hurwitz ([`dynamics`](dynamics.md)). A zero-state system is vacuously stable. |
| `to_transfer_function` | `auto to_transfer_function() const -> Result<TransferFunction>` | Convert to the SISO transfer function `C(sI − A)⁻¹B + D` (exact; defers to `ss_to_tf`). Requires a single input and single output, else `domain_error`. |

## TF ↔ SS conversions (both exact over `Q`)

Free functions in namespace `nimblecas`.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `tf_to_ss` | `[[nodiscard]] auto tf_to_ss(const TransferFunction& tf) -> Result<StateSpace>` | Realise a **proper** SISO transfer function in controllable canonical form. The denominator is made monic and the numerator scaled to match; the feedthrough `D` captures any direct part (`deg num == deg den`). Fails `domain_error` when the plant is improper (`deg num > deg den`) or the denominator is zero. A degree-0 denominator becomes a pure static gain (zero states, single `D` entry). |
| `ss_to_tf` | `[[nodiscard]] auto ss_to_tf(const StateSpace& ss) -> Result<TransferFunction>` | Recover the SISO transfer function via the **Faddeev–LeVerrier** resolvent: `C·adj(sI − A)·B / det(sI − A) + D`, all exact over `Q`. Requires a single input and single output, else `domain_error`. |

## Stability criteria (exact over `Q`)

### Continuous and discrete decision (free functions)

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `is_stable_continuous` | `[[nodiscard]] auto is_stable_continuous(const TransferFunction& tf) -> Result<bool>` | Every pole in the open left half-plane. Decided **exactly** by Routh–Hurwitz on the denominator — built as a companion matrix and handed to [`dynamics`](dynamics.md)`::is_asymptotically_stable`, so irrational/complex poles are settled without ever finding a root. A constant denominator (no poles) is stable. |
| `is_stable_discrete` | `[[nodiscard]] auto is_stable_discrete(const TransferFunction& tf) -> Result<bool>` | Every pole strictly inside the unit circle (Schur/Jury). Decided **exactly** by mapping the open unit disc onto the open left half-plane with the Möbius transform `z = (1 + w)/(1 − w)` and testing the transformed denominator for the Hurwitz property — the same exact Routh machinery. A boundary pole at `z = −1` (which the transform sends to infinity, dropping the degree) is correctly reported as **not** stable. |

### Hurwitz determinant test

The Hurwitz-matrix / leading-principal-minor formulation of Routh–Hurwitz, an
exact cross-check of `is_stable_continuous` on the same denominator.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `hurwitz_matrix` | `[[nodiscard]] auto hurwitz_matrix(const RationalPoly& char_poly) -> Result<Matrix>` | The `n × n` Hurwitz matrix of a polynomial of degree `n >= 1`, first made monic. Entry `(i, j)` is `a_{2j−i+1}` with `a_k` = coefficient of `sⁿ⁻ᵏ` (and `a_k = 0` outside `0..n`). Fails `domain_error` on the zero polynomial; a degree-0 polynomial yields the empty `0×0` matrix. |
| `hurwitz_minors` | `[[nodiscard]] auto hurwitz_minors(const RationalPoly& char_poly) -> Result<std::vector<Rational>>` | The sequence of leading principal minors `Δ₁, …, Δₙ`, computed exactly over `Q`. The polynomial is Hurwitz-stable iff every `Δ_k > 0`. A degree-0 polynomial yields an empty sequence. |
| `is_hurwitz_stable` | `[[nodiscard]] auto is_hurwitz_stable(const RationalPoly& char_poly) -> Result<bool>` | `true` iff all leading principal minors are strictly positive. Exact over `Q`; by the Routh–Hurwitz theorem it agrees with `is_stable_continuous` on the same denominator. A constant (degree-0, root-free) polynomial is vacuously stable. |

### Kharitonov robust (interval) stability

Robust Hurwitz stability of an **interval polynomial** whose coefficient of `sⁱ`
lies in `[lower[i], upper[i]]` (ascending order, both vectors the same length).

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `kharitonov_polynomials` | `[[nodiscard]] auto kharitonov_polynomials(std::span<const Rational> lower, std::span<const Rational> upper) -> Result<std::array<RationalPoly, 4>>` | The four Kharitonov polynomials by the sign pattern `(−,−,+,+),(+,+,−,−),(+,−,−,+),(−,+,+,−)` repeating on `i mod 4` (`−` = lower bound, `+` = upper bound). Fails `domain_error` if the two vectors differ in length or are empty, or if some `lower[i] > upper[i]`. |
| `is_robustly_stable` | `[[nodiscard]] auto is_robustly_stable(std::span<const Rational> lower, std::span<const Rational> upper) -> Result<bool>` | Kharitonov's theorem: the whole interval family is robustly Hurwitz-stable iff all four Kharitonov polynomials are Hurwitz-stable (`is_hurwitz_stable` on each). Exact over `Q`. Same well-formedness requirements as above. |

### Lyapunov stability (state-space side)

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `is_spd` | `[[nodiscard]] auto is_spd(const Matrix& p) -> Result<bool>` | Exact positive-definiteness via **Sylvester's criterion**: a symmetric `P` is positive definite iff every leading principal minor is strictly positive (exact rational determinants). Requires a square matrix, else `domain_error`; the `0×0` matrix is vacuously positive definite. |
| `lyapunov_solve` | `[[nodiscard]] auto lyapunov_solve(const Matrix& a) -> Result<Matrix>` | The unique solution `P` of `Aᵀ P + P A = −I`, obtained exactly by assembling the **Kronecker-sum** system `(I ⊗ Aᵀ + Aᵀ ⊗ I) vec(P) = −vec(I)` and solving with exact rational elimination. Requires a square `A`; fails `domain_error` when the Kronecker-sum operator is singular (some `λ_i + λ_j = 0`, e.g. a purely imaginary spectrum — which already means `A` is not asymptotically stable). |
| `is_lyapunov_stable` | `[[nodiscard]] auto is_lyapunov_stable(const Matrix& a) -> Result<bool>` | Lyapunov's theorem: `A` is asymptotically stable iff `Aᵀ P + P A = −I` has a positive-definite solution `P`. Exact over `Q`; agrees with `is_asymptotically_stable` / `is_stable_continuous`. A singular Kronecker-sum operator (no unique `P`) means `A` is not asymptotically stable, reported as `false`. |

## Discrete transforms — bilinear (Tustin) (exact over `Q`)

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `bilinear_c2d` | `[[nodiscard]] auto bilinear_c2d(const TransferFunction& tf, const Rational& sample_time) -> Result<TransferFunction>` | Continuous → discrete via `s = (2/T)·(z − 1)/(z + 1)`, with sample time `T > 0` (a `Rational`). Exact rational substitution. An s-domain pole `p` maps to the z-domain pole `(1 + pT/2)/(1 − pT/2)`, so the open left half-plane maps to the open unit disc — Tustin preserves stability. Fails `division_by_zero` when `T == 0`. |
| `bilinear_d2c` | `[[nodiscard]] auto bilinear_d2c(const TransferFunction& tf, const Rational& sample_time) -> Result<TransferFunction>` | Discrete → continuous, the inverse map `z = (1 + (T/2) s)/(1 − (T/2) s)`. Exact. Fails `division_by_zero` when `T == 0`. |

## Frequency response

`evaluate_exact` is exact (`Q + Qi`); `bode`, `nyquist`, and `logspace` are the
honestly **numerical** double-precision path.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `evaluate_exact` | `[[nodiscard]] auto evaluate_exact(const TransferFunction& tf, const Complex& s) -> Result<Complex>` | **Exact** evaluation of `H` at a Gaussian-rational point `s` (e.g. `s = iω` with `ω` rational), staying entirely inside `Q + Qi`. Fails `division_by_zero` when `den(s) == 0`. |
| `bode` | `[[nodiscard]] auto bode(const TransferFunction& tf, std::span<const double> omegas) -> std::vector<BodePoint>` | **Numerical** Bode data: one `BodePoint` per `ω`, with `magnitude_db = 20·log10|H(iω)|` and `phase_deg = arg H(iω)`. Double-precision `std::complex`. |
| `nyquist` | `[[nodiscard]] auto nyquist(const TransferFunction& tf, std::span<const double> omegas) -> std::vector<NyquistPoint>` | **Numerical** Nyquist trace: `H(iω) = re + im·i` sampled at each `ω`. |
| `logspace` | `[[nodiscard]] auto logspace(double w_start, double w_end, std::size_t count) -> std::vector<double>` | `count` logarithmically spaced frequencies in `[w_start, w_end]` (both must be `> 0`). Returns a single point when `count <= 1` and an empty vector when `count == 0`. |

## Nyquist stability criterion (`P` exact; `N` and the trace NUMERICAL)

### `NyquistResult`

| Member | Type | Meaning |
| :--- | :--- | :--- |
| `encirclements` | `std::int64_t` | `N`: clockwise encirclements of the `−1` point. **Numerical** — a winding number from the double-precision Nyquist trace. |
| `open_loop_rhp_poles` | `std::int64_t` | `P`: open-loop poles of `L` in the open right half-plane, counted from the **rational** poles (see `p_exact`). |
| `closed_loop_rhp_poles` | `std::int64_t` | `Z = N + P`: implied closed-loop poles in the right half-plane. |
| `closed_loop_stable` | `bool` | `true` iff `Z == 0`. |
| `p_exact` | `bool` | `true` when `L` has no irrational/complex poles that `poles()` could not extract; when `false`, treat `P`/`Z` as a rational-only lower read. |

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `nyquist_criterion` | `[[nodiscard]] auto nyquist_criterion(const TransferFunction& open_loop, std::span<const double> omegas) -> Result<NyquistResult>` | Apply the Nyquist criterion to the open-loop `L` over a **positive**, ascending frequency grid. `N` is a numerical winding number of the full mirror-completed contour about `−1`; `P` is counted from `L.poles()` (exact for rational poles). Fails `domain_error` on an empty grid or when `L`'s poles cannot be obtained. An open-loop integrator (pole at the origin) is not represented by a finite grid and is outside this routine's scope. |

## Gain & phase margins (NUMERICAL)

### `StabilityMargins`

| Member | Type | Meaning |
| :--- | :--- | :--- |
| `gain_margin` | `double` | Linear gain margin `1/|L(iω_pc)|`. |
| `gain_margin_db` | `double` | Gain margin in dB. |
| `phase_crossover` | `double` | Phase-crossover frequency `ω_pc` (rad/s), where `arg L = −180°`. |
| `phase_margin` | `double` | Phase margin (degrees) `= 180° + arg L(iω_gc)`. |
| `gain_crossover` | `double` | Gain-crossover frequency `ω_gc` (rad/s), where `|L| = 1`. |
| `has_gain_margin` / `has_phase_margin` | `bool` | Whether the respective crossover was actually found on the supplied grid. |

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `stability_margins` | `[[nodiscard]] auto stability_margins(const TransferFunction& open_loop, std::span<const double> omegas) -> StabilityMargins` | Gain and phase margins of the open-loop `L`, found **numerically** by locating the crossover frequencies on the double-precision Bode data over the supplied positive grid and linearly interpolating. The phase is unwrapped before locating the `−180°` crossing. Returns an all-default `StabilityMargins` (both `has_*` flags `false`) when fewer than two grid points are supplied. |

## Error model

| Condition | Error |
| :--- | :--- |
| `TransferFunction::make` / any interconnection with a zero denominator | `MathError::domain_error` |
| `feedback` whose combined denominator is the zero polynomial | `MathError::domain_error` |
| `zeros()` of the zero transfer function (`num == 0`) | `MathError::domain_error` |
| `dc_gain()` with a pole at the origin (`den(0) == 0`) | `MathError::division_by_zero` |
| `StateSpace::make` with inconsistent dimensions | `MathError::domain_error` |
| `ss_to_tf` / `to_transfer_function` on a non-SISO model | `MathError::domain_error` |
| `tf_to_ss` on an improper plant (`deg num > deg den`) or zero denominator | `MathError::domain_error` |
| `is_stable_discrete` / `hurwitz_matrix` / `hurwitz_minors` on the zero polynomial | `MathError::domain_error` |
| `kharitonov_polynomials` / `is_robustly_stable`: mismatched/empty vectors or `lower[i] > upper[i]` | `MathError::domain_error` |
| `is_spd` / `lyapunov_solve` / `is_lyapunov_stable` on a non-square matrix | `MathError::domain_error` |
| `lyapunov_solve` with a singular Kronecker-sum operator | `MathError::domain_error` (surfaced as `false` by `is_lyapunov_stable`) |
| `bilinear_c2d` / `bilinear_d2c` with `T == 0` | `MathError::division_by_zero` |
| `evaluate_exact` with `den(s) == 0` | `MathError::division_by_zero` |
| `nyquist_criterion` on an empty grid or unobtainable poles | `MathError::domain_error` |
| An `int64` numerator/denominator computation wraps in any exact step | `MathError::overflow` |

An irrational/complex spectrum is a **result, not an error**: `poles()`/`zeros()`
still succeed and report the remainder in `RootReport::unextracted`. The three
double-precision routines (`bode`, `nyquist`, `stability_margins`) do not use the
railway; an imaginary-axis pole produces an `inf`/`nan` sample rather than an
error.

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.complex;
import nimblecas.control;
using namespace nimblecas;

auto ri = [](std::int64_t v) { return Rational::from_int(v); };
auto rq = [](std::int64_t n, std::int64_t d) { return Rational::make(n, d).value(); };

// A RationalPoly from integer coefficients, ascending (index i is the coeff of x^i).
auto rp = [&](std::vector<std::int64_t> cs) {
    std::vector<Rational> r;
    for (std::int64_t v : cs) r.push_back(ri(v));
    return RationalPoly::from_coeffs(std::move(r));
};
auto TF = [](RationalPoly num, RationalPoly den) {
    return TransferFunction::make(std::move(num), std::move(den)).value();
};
auto mat = [&](std::vector<std::vector<std::int64_t>> rows) {
    std::vector<std::vector<Rational>> r;
    for (const auto& row : rows) {
        std::vector<Rational> rr;
        for (std::int64_t v : row) rr.push_back(ri(v));
        r.push_back(std::move(rr));
    }
    return Matrix::from_rows(std::move(r)).value();
};

// --- poles: exact for rational roots, honest about the rest ----------------
// H = 1/(s^2 + 3s + 2): poles at s = -1 and s = -2, fully rational.
auto p = TF(rp({1}), rp({2, 3, 1})).poles().value();
p.fully_extracted();                         // true  (unextracted == 0)
p.rational.size();                           // 2     ({-1, -2}, each mult 1)

// H = 1/(s^2 - 2): poles at ±sqrt(2), irrational -> NOT extracted over Q.
auto pr = TF(rp({1}), rp({-2, 0, 1})).poles().value();
pr.rational.empty();                         // true  (no rational poles)
pr.fully_extracted();                        // false
pr.unextracted;                              // 2     (both poles counted, not dropped)

// --- DC gain and properness ------------------------------------------------
// H = (2s + 4)/(s^2 + 3s + 2): H(0) = 4/2 = 2, strictly proper.
const auto h = TF(rp({4, 2}), rp({2, 3, 1}));
h.dc_gain().value();                         // 2
h.is_strictly_proper();                      // true
h.relative_degree();                         // 1

// --- interconnection algebra (exact) ---------------------------------------
const auto g1 = TF(rp({1}), rp({0, 1}));     // 1/s
const auto g2 = TF(rp({1}), rp({1, 1}));     // 1/(s+1)
g1.series(g2).value().equivalent(TF(rp({1}), rp({0, 1, 1}))).value();   // true: 1/(s^2+s)
g1.unity_feedback().value().equivalent(TF(rp({1}), rp({1, 1}))).value();// true: 1/(s+1)
g1.parallel(g2).value().equivalent(TF(rp({1, 2}), rp({0, 1, 1}))).value();// true: (2s+1)/(s^2+s)

// --- TF <-> SS round-trip is exactly equivalent ----------------------------
// H = (s + 1)/(s^2 + 3s + 2).
const auto h2 = TF(rp({1, 1}), rp({2, 3, 1}));
auto ss = tf_to_ss(h2).value();
ss.n_states();                               // 2
h2.equivalent(ss_to_tf(ss).value()).value(); // true
tf_to_ss(TF(rp({0, 0, 1}), rp({1, 1}))).error();  // domain_error (improper: deg num > deg den)

// --- controllability / observability (exact Kalman rank) -------------------
auto sys = StateSpace::make(mat({{0, 1}, {0, 0}}), mat({{0}, {1}}),
                            mat({{1, 0}}), mat({{0}})).value();
sys.is_controllable().value();               // true
sys.is_observable().value();                 // true
// Uncontrollable pair (A = I, B = [1,0]^T): [B AB] has rank 1.
StateSpace::make(mat({{1, 0}, {0, 1}}), mat({{1}, {0}}),
                 mat({{1, 0}}), mat({{0}})).value().is_controllable().value();  // false

// --- continuous stability: exact Routh-Hurwitz -----------------------------
is_stable_continuous(TF(rp({1}), rp({2, 3, 1}))).value();  // true  (roots -1, -2)
is_stable_continuous(TF(rp({1}), rp({-1, 0, 1}))).value(); // false (roots ±1)
is_stable_continuous(TF(rp({1}), rp({1, 0, 1}))).value();  // false (roots ±i, imag axis)

// --- discrete stability: exact Schur/Jury ----------------------------------
is_stable_discrete(TF(rp({1}), RationalPoly::from_coeffs({rq(-1, 2), ri(1)}))).value();
                                             // true  (pole at z = 1/2, inside)
is_stable_discrete(TF(rp({1}), rp({-2, 1}))).value();   // false (pole at z = 2, outside)
is_stable_discrete(TF(rp({1}), rp({1, 1}))).value();    // false (boundary pole z = -1)

// --- Hurwitz determinant test agrees with Routh ----------------------------
is_hurwitz_stable(rp({2, 3, 1})).value();    // true  (s^2 + 3s + 2)
is_hurwitz_stable(rp({8, 2, 1, 1})).value(); // false (s^3+s^2+2s+8: a2*a1 < a0)
hurwitz_minors(rp({2, 3, 1})).value().size();// 2     (both minors > 0 for a stable poly)

// --- Kharitonov robust interval stability ----------------------------------
// a0 in [1,2], a1 in [2,4], a2 in [2,4], a3 = 1: robustly stable.
const std::vector<Rational> lo{ri(1), ri(2), ri(2), ri(1)};
const std::vector<Rational> hi{ri(2), ri(4), ri(4), ri(1)};
is_robustly_stable(lo, hi).value();          // true
kharitonov_polynomials(lo, hi).value().size();  // 4
// a1 in [-1, 1] -> a Kharitonov vertex has a negative coefficient: not robust.
is_robustly_stable({ri(1), ri(-1), ri(2), ri(1)}, {ri(1), ri(1), ri(2), ri(1)}).value();  // false

// --- Lyapunov: exact P of A^T P + P A = -I ----------------------------------
const auto a = mat({{-1, 0}, {0, -2}});
auto P = lyapunov_solve(a).value();
P.at(0, 0);                                  // 1/2
P.at(1, 1);                                  // 1/4   (P = diag(1/2, 1/4) exactly)
is_spd(P).value();             // true
is_lyapunov_stable(a).value();               // true  (agrees with Routh-Hurwitz)
// Rotation [[0,-1],[1,0]] (eigenvalues ±i): singular Kronecker sum.
lyapunov_solve(mat({{0, -1}, {1, 0}})).error();      // domain_error
is_lyapunov_stable(mat({{0, -1}, {1, 0}})).value();  // false

// --- bilinear (Tustin) transform (exact) -----------------------------------
// H(s) = 1/(s+1), T = 2 => H(z) = (z+1)/(2z).
bilinear_c2d(TF(rp({1}), rp({1, 1})), ri(2)).value()
    .equivalent(TF(rp({1, 1}), rp({0, 2}))).value();  // true
bilinear_c2d(TF(rp({1}), rp({1, 1})), ri(0)).error(); // division_by_zero (T == 0)

// --- exact frequency-response evaluation (stays in Q + Qi) ------------------
// H = 1/(s+1) at s = i: exact value 1/2 - (1/2) i.
auto v = evaluate_exact(TF(rp({1}), rp({1, 1})), Complex::i()).value();
v.real();                                    // 1/2
v.imag();                                    // -1/2

// --- NUMERICAL Bode data ----------------------------------------------------
// H = 1/(s+1) at omega = 1: |H(i)| = 1/sqrt(2) -> -3.0103 dB, phase -45 deg.
const std::array<double, 1> w{1.0};
auto bd = bode(TF(rp({1}), rp({1, 1})), w);
bd[0].magnitude_db;                          // ~ -3.0103
bd[0].phase_deg;                             // ~ -45.0

// --- NUMERICAL Nyquist criterion & margins ----------------------------------
const auto grid = logspace(1e-2, 1e2, 2000);
auto nr = nyquist_criterion(TF(rp({1}), rp({1, 1})), grid).value();  // L = 1/(s+1)
nr.open_loop_rhp_poles;                      // 0    (exact: p_exact == true)
nr.closed_loop_stable;                       // true (no encirclement of -1)

// L = 2/(s+1)^3: GM ~ 12.04 dB at omega_pc ~ sqrt(3), PM ~ 67.64 deg.
const auto m = stability_margins(TF(rp({2}), rp({1, 3, 3, 1})), logspace(1e-2, 1e2, 4000));
m.has_gain_margin && m.has_phase_margin;     // true
m.gain_margin_db;                            // ~ 12.0412
m.phase_crossover;                           // ~ 1.73205 (sqrt 3)
```

## See also

- [`nimblecas.dynamics`](dynamics.md) — the exact Routh–Hurwitz decision
  (`is_asymptotically_stable`) this module reuses for every continuous/discrete
  stability verdict.
- [`nimblecas.matrix`](matrix.md) — the exact `Rational` matrices, ranks, and
  solves behind `StateSpace`, the conversions, and the Lyapunov equation.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field and
  `RationalPoly` every numerator, denominator, and coefficient lives in.
- [`nimblecas.roots`](roots.md) — `rational_roots`, the exact rational-root
  extraction behind `poles()`/`zeros()`.
- [`nimblecas.complex`](complex.md) — the exact Gaussian rationals `Q + Qi` that
  `evaluate_exact` stays inside.
- [Documentation hub](../Index.md)
</content>
</invoke>
