# `nimblecas.analysis` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/analysis/analysis.cppm`

Numerical / exact analysis on top of the exact rational [`matrix`](matrix.md)
substrate and the [`dynamics`](dynamics.md) Routh–Hurwitz engine, grouping three
classical facilities — **condition numbers**, **series convergence tests**, and
**Lyapunov stability**. The module is scrupulously honest about which results are
**EXACT over the rationals `Q`** and which are **NUMERICAL** `double` estimates,
and it keeps the two kinds cleanly separated:

- **Exact over `Q`.** The `1`- and `∞`-norms and their condition numbers
  (`condition_1` / `condition_inf`); the ratio test's limit *when the ratio is a
  constant rational*; the continuous Lyapunov and discrete Stein solves (by
  Kronecker-sum vectorization); Sylvester positive-definiteness; the Lyapunov
  stability verdict (`is_stable_lyapunov`) and its Routh–Hurwitz cross-check.
  These never touch floating point.
- **Numerical (double), an estimate.** The spectral condition number
  `condition_2_estimate` (power iteration on `AᵀA`); the root / comparison /
  integral / alternating-series tests; the ratio test's *finite-`n` fallback*;
  and `lyapunov_exponent` (probe-vector growth). These carry sampling and
  floating-point error and are named or documented as estimates.

```cpp
import nimblecas.analysis;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), [`matrix`](matrix.md),
and [`dynamics`](dynamics.md).

Every fallible step is threaded on the [`Result`](core.md) railway. Dimension
violations surface as `MathError::domain_error`; an `int64`
numerator/denominator that would wrap in the exact arithmetic surfaces as
`MathError::overflow`. All symbols below live in namespace `nimblecas`.

## Condition number

Both operator norms are the usual induced norms — `‖A‖₁` the maximum absolute
column sum, `‖A‖∞` the maximum absolute row sum — computed with exact `Rational`
arithmetic, so the norms and the two condition numbers they build are **exact**.
The spectral condition number is the honest exception: it is a **numerical
estimate** and named `condition_2_estimate` to say so.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `matrix_norm_1` | `[[nodiscard]] auto matrix_norm_1(const Matrix& a) -> Result<Rational>` | `‖A‖₁ = maxⱼ Σᵢ \|aᵢⱼ\|` (max absolute column sum), **exact over `Q`**. A `0×0` or zero-column matrix yields `domain_error`. |
| `matrix_norm_inf` | `[[nodiscard]] auto matrix_norm_inf(const Matrix& a) -> Result<Rational>` | `‖A‖∞ = maxᵢ Σⱼ \|aᵢⱼ\|` (max absolute row sum), **exact over `Q`**. A `0`-row matrix yields `domain_error`. |
| `condition_1` | `[[nodiscard]] auto condition_1(const Matrix& a) -> Result<Rational>` | `κ₁(A) = ‖A‖₁ · ‖A⁻¹‖₁`, **exact over `Q`**. Requires a square, nonsingular `A`; a singular or non-square `A` yields `domain_error` (from the inverse). |
| `condition_inf` | `[[nodiscard]] auto condition_inf(const Matrix& a) -> Result<Rational>` | `κ∞(A) = ‖A‖∞ · ‖A⁻¹‖∞`, **exact over `Q`**. Same requirements as `condition_1`. |
| `condition_2_estimate` | `[[nodiscard]] auto condition_2_estimate(const Matrix& a) -> Result<double>` | The spectral condition number `κ₂(A) = σ_max / σ_min`, the singular values being the square roots of the eigenvalues of `AᵀA`. **NUMERICAL:** `AᵀA` is built exactly then bridged to `double` for power iteration on `AᵀA` and on `(AᵀA)⁻¹` — an **estimate**, not an exact rational value. Requires a square, nonsingular `A` (a singular `A` means `κ₂ = ∞` and surfaces as `domain_error`). |

## Convergence tests for a series `Σ aₙ`

Two sequence callables feed the tests — a `double`-valued term for the numerical
tests and an exact `Rational`-valued term for the ratio test — and every verdict
is one of three values.

```cpp
using RealSequence     = std::function<double(std::int64_t)>;    // aₙ, n ≥ 0
using RationalSequence = std::function<Rational(std::int64_t)>;  // exact aₙ, n ≥ 0

enum class Verdict : std::uint8_t { converges, diverges, inconclusive };

[[nodiscard]] auto to_string_view(Verdict v) noexcept -> std::string_view;
```

Both sequence aliases are **total by signature** (`n ≥ 0`) and never fail.
`to_string_view` renders a `Verdict` as `"converges"`, `"diverges"`, or
`"inconclusive"`.

### Ratio (d'Alembert) test — exact when the ratio is a constant rational

```cpp
struct RatioTest {
    Verdict verdict{Verdict::inconclusive};
    bool exact{false};                      // true iff L was determined exactly over Q
    std::optional<Rational> exact_limit{};  // L = lim |a_{n+1}/a_n| when `exact`
    double numeric_limit{0.0};              // a double view of the limit / finite-n estimate
};

[[nodiscard]] auto ratio_test(const RationalSequence& a) -> RatioTest;
```

`ratio_test` samples `|a_{n+1}/aₙ|` at a few small indices (small keeps a
geometric `rⁿ` inside `int64` before it overflows). **If every computable ratio
is the same exact rational** — as for a geometric series `aₙ = c·rⁿ`, whose ratio
is exactly `|r|` — that constant is returned as the **EXACT limit `L`**
(`exact = true`, `exact_limit = L`) and the verdict follows exactly: `L < 1`
converges, `L > 1` diverges, `L == 1` inconclusive. **Otherwise** the limit is
not pinned down exactly and a **NUMERICAL** finite-`n` estimate at the largest
sampled index is reported (`exact = false`) with a heuristic verdict — a
documented approximation, since the ratio test is genuinely inconclusive as
`L → 1`. Zero terms are skipped, and an overflow in a single exact ratio drops
that one sample rather than failing.

### Numerical tests

```cpp
struct NumericTest {
    Verdict verdict{Verdict::inconclusive};
    double numeric_limit{0.0};  // the computed test statistic (NUMERICAL estimate)
};
```

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `root_test` | `[[nodiscard]] auto root_test(const RealSequence& a, std::int64_t n = 64) -> NumericTest` | Root (Cauchy) test: estimate `L = \|aₙ\|^{1/n}` at the sample index `n` (**NUMERICAL**). `L < 1` converges, `L > 1` diverges, `L ≈ 1` inconclusive. `n < 1` returns `inconclusive` with statistic `0`. |
| `comparison_test` | `[[nodiscard]] auto comparison_test(const RealSequence& a, const RealSequence& b, Verdict b_behaviour, std::int64_t samples = 200) -> Verdict` | Direct comparison test (**NUMERICAL**). Given the known `b_behaviour` of a non-negative reference `bₙ`, decide `aₙ` by term-wise domination over `n = 1..samples`: `b` converges and `0 ≤ aₙ ≤ bₙ` throughout ⇒ **converges**; `b` diverges and `aₙ ≥ bₙ ≥ 0` throughout ⇒ **diverges**; otherwise **inconclusive**. `samples < 1` ⇒ inconclusive. |
| `integral_test` | `[[nodiscard]] auto integral_test(const RealSequence& f, std::int64_t upper = 4096) -> NumericTest` | Integral-test hook (**NUMERICAL**). For a positive, decreasing `f`, `Σ aₙ` converges iff the improper integral of `f` converges; the integral is estimated by the trapezoidal rule on the integer grid and convergence inferred from the vanishing tail increment `I(2N) − I(N)`. `numeric_limit` carries the partial integral `I(upper)`. `upper < 2` ⇒ inconclusive with statistic `0`. |
| `alternating_series_test` | `[[nodiscard]] auto alternating_series_test(const RealSequence& magnitude, std::int64_t samples = 1000) -> Verdict` | Alternating-series (Leibniz) test (**NUMERICAL**). `magnitude` supplies `bₙ = \|aₙ\| ≥ 0` of `Σ(−1)ⁿ bₙ`. Converges when `bₙ` is monotonically non-increasing and `bₙ → 0`; diverges when `bₙ` does not tend to `0` (nth-term test); otherwise inconclusive. Sampled over `n = 1..samples` (and `2·samples` for the limit check). `samples < 2` ⇒ inconclusive. |

The alternating (Leibniz) test is exact in its logic but evaluated on sampled
doubles, so it is reported under the numerical family.

## Lyapunov equations and stability

The two matrix equations are solved **exactly over `Q`** by vectorization
against a Kronecker-sum system, and the definiteness and stability verdicts that
consume them are likewise exact. The Lyapunov *exponent* is the honest numerical
outlier at the end.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `lyapunov_solve` | `[[nodiscard]] auto lyapunov_solve(const Matrix& a, const Matrix& q) -> Result<Matrix>` | Solve the continuous Lyapunov equation `AᵀP + PA = −Q` for `P`, **EXACTLY over `Q`**, by column-major vectorization `(I ⊗ Aᵀ + Aᵀ ⊗ I) vec(P) = −vec(Q)`. `A` and `Q` must both be square and of the same size `n ≥ 1`, else `domain_error`. The Kronecker-sum system is singular exactly when two eigenvalues of `A` sum to zero (e.g. a zero or purely-imaginary eigenvalue) — that surfaces as `domain_error` from the underlying solve. |
| `stein_solve` | `[[nodiscard]] auto stein_solve(const Matrix& a, const Matrix& q) -> Result<Matrix>` | Solve the discrete Stein equation `AᵀPA − P = −Q` for `P`, **EXACTLY over `Q`**, via `(Aᵀ ⊗ Aᵀ − I) vec(P) = −vec(Q)`. Same shape requirements; the system is singular exactly when a product of two eigenvalues of `A` equals `1`. |
| `is_positive_definite` | `[[nodiscard]] auto is_positive_definite(const Matrix& p) -> Result<bool>` | Test `P ≻ 0` via **Sylvester's criterion**: all `n` leading principal minors strictly positive, each an **exact rational determinant**. The criterion presumes a **symmetric** `P` (as produced by `lyapunov_solve` for a symmetric `Q`); it is not a definiteness test for general non-symmetric matrices. Requires a square matrix, else `domain_error`. |
| `is_stable_lyapunov` | `[[nodiscard]] auto is_stable_lyapunov(const Matrix& a) -> Result<bool>` | Continuous-LTI asymptotic stability of `dx/dt = A x`, decided **EXACTLY over `Q`**: pick `Q = I`, solve `AᵀP + PA = −I`, and return whether `P ≻ 0` by Sylvester's criterion. `A` is asymptotically stable iff such a `P` is positive definite. A singular Kronecker-sum system (an eigenvalue pair summing to zero — necessarily non-Hurwitz) is reported as **not stable** (`false`), not an error. Requires a square matrix with `n ≥ 1`, else `domain_error`. |
| `stability_cross_check` | `[[nodiscard]] auto stability_cross_check(const Matrix& a) -> Result<StabilityCrossCheck>` | Cross-check the exact Lyapunov/Sylvester verdict against the exact Routh–Hurwitz verdict (`is_asymptotically_stable` from [`dynamics`](dynamics.md)) on the characteristic polynomial. Both are exact over `Q` and must agree. Requires a square matrix `n ≥ 1`, else `domain_error`. |
| `lyapunov_exponent` | `[[nodiscard]] auto lyapunov_exponent(std::span<const Matrix> jacobians) -> Result<double>` | The leading Lyapunov exponent of a Jacobian product `J_count … J_1`: `(1/count)·log‖J_count … J_1‖`, estimated by probe-vector growth with per-step renormalization in `double` — **NUMERICAL, an estimate**. `jacobians` holds `J_1, J_2, …` in application order (`J_1` first). Requires a non-empty span of square matrices of one common size `n ≥ 1`, else `domain_error`. A product that collapses a direction to zero yields `−∞` (total contraction). |

```cpp
struct StabilityCrossCheck {
    bool lyapunov_stable{false};      // exact: Sylvester on the Lyapunov solution (Q = I)
    bool routh_hurwitz_stable{false}; // exact: nimblecas.dynamics Routh–Hurwitz criterion
    bool agree{false};                // lyapunov_stable == routh_hurwitz_stable
};
```

## Error model

| Condition | Error |
| :--- | :--- |
| `matrix_norm_1` / `matrix_norm_inf`: empty (`0`-row or `0`-column) matrix | `MathError::domain_error` |
| `condition_1` / `condition_inf` / `condition_2_estimate`: `a` not square | `MathError::domain_error` |
| `condition_1` / `condition_inf` / `condition_2_estimate`: singular `A` (no inverse) | `MathError::domain_error` (from the inverse) |
| `condition_2_estimate`: the estimated `κ₂²` is not positive | `MathError::domain_error` |
| `lyapunov_solve` / `stein_solve`: `a` not square, or `a.rows() < 1` | `MathError::domain_error` |
| `lyapunov_solve` / `stein_solve`: `q` not the same size as `a` | `MathError::domain_error` |
| `lyapunov_solve` / `stein_solve`: singular Kronecker-sum system | `MathError::domain_error` (from the solve) |
| `is_positive_definite`: `p` not square | `MathError::domain_error` |
| `is_stable_lyapunov` / `stability_cross_check`: `a` not square, or `a.rows() < 1` | `MathError::domain_error` |
| `lyapunov_exponent`: empty span, or a Jacobian not square / of a differing size | `MathError::domain_error` |
| An `int64` numerator/denominator computation wraps anywhere in the exact arithmetic | `MathError::overflow` |

Two outcomes are **results, not errors**: a singular Kronecker-sum system inside
`is_stable_lyapunov` (its `Q = I` Lyapunov solve failing with `domain_error`)
returns `false` — a non-Hurwitz `A` — and a Jacobian product that annihilates the
probe direction returns `−∞` from `lyapunov_exponent`. The sequence callables
and `to_string_view` are total and never error. Note that `ratio_test`,
`root_test`, `comparison_test`, `integral_test`, and `alternating_series_test`
return their verdict structs/enums directly (no `Result`): they cannot fail.

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.dynamics;
import nimblecas.analysis;
using namespace nimblecas;

auto ri = [](std::int64_t v) { return Rational::from_int(v); };
auto rr = [](std::int64_t n, std::int64_t d) { return Rational::make(n, d).value(); };
auto mat = [&](std::vector<std::vector<std::int64_t>> rows) {   // integer matrix
    std::vector<std::vector<Rational>> r;
    for (const auto& row : rows) {
        std::vector<Rational> q;
        for (std::int64_t v : row) q.push_back(ri(v));
        r.push_back(std::move(q));
    }
    return Matrix::from_rows(std::move(r)).value();
};
auto pow2 = [](std::int64_t n) { return static_cast<std::int64_t>(1) << n; };

// --- condition number -----------------------------------------------------
// A = [[1,-2],[3,4]]: col abs sums {4,6} => ‖A‖₁ = 6; row abs sums {3,7} => ‖A‖∞ = 7.
matrix_norm_1(mat({{1, -2}, {3, 4}})).value();     // 6
matrix_norm_inf(mat({{1, -2}, {3, 4}})).value();   // 7

// A = [[1,2],[0,1]], A⁻¹ = [[1,-2],[0,1]]: ‖A‖₁ = ‖A⁻¹‖₁ = 3 => κ₁ = 9 (exact).
condition_1(mat({{1, 2}, {0, 1}})).value();        // 9
condition_inf(mat({{1, 2}, {0, 1}})).value();      // 9

// diag(2,4): ‖A‖₁ = 4, ‖A⁻¹‖₁ = 1/2 => κ₁ = 2 exactly; κ₂ ≈ 2 (numerical).
condition_1(mat({{2, 0}, {0, 4}})).value();          // 2
condition_2_estimate(mat({{2, 0}, {0, 4}})).value(); // ≈ 2.0

// Singular A => domain_error.
condition_1(mat({{1, 1}, {1, 1}})).error();        // MathError::domain_error

// --- convergence tests ----------------------------------------------------
// aₙ = (1/2)ⁿ: ratio 1/2 exactly => converges, exact limit L = 1/2.
auto g = ratio_test([&](std::int64_t n) { return rr(1, pow2(n)); });
g.exact;                                            // true
*g.exact_limit;                                     // 1/2
g.verdict;                                          // Verdict::converges

// aₙ = 2ⁿ: ratio 2 exactly => diverges, exact limit L = 2.
auto d = ratio_test([&](std::int64_t n) { return ri(pow2(n)); });
d.exact_limit.value();                              // 2
d.verdict;                                          // Verdict::diverges

// Root test on (1/3)ⁿ: statistic ≈ 1/3 => converges (NUMERICAL).
root_test([](std::int64_t n) { return std::pow(1.0 / 3.0, double(n)); }).verdict;
                                                    // Verdict::converges

// Alternating harmonic bₙ = 1/n: monotone down to 0 => Leibniz convergence.
alternating_series_test([](std::int64_t n) { return 1.0 / double(n); });
                                                    // Verdict::converges

// 0 <= 1/(n²+1) <= 1/n², and Σ 1/n² converges => converges.
comparison_test([](std::int64_t n) { return 1.0 / (double(n) * n + 1.0); },
                [](std::int64_t n) { return 1.0 / (double(n) * n); },
                Verdict::converges);                // Verdict::converges

to_string_view(Verdict::diverges);                  // "diverges"

// --- Lyapunov equation & stability ----------------------------------------
// A = [[-1,1],[0,-1]] (Hurwitz), Q = I. Solve AᵀP + PA = -Q; the residual is -Q.
const Matrix a = mat({{-1, 1}, {0, -1}});
auto p = lyapunov_solve(a, Matrix::identity(2)).value();
// aᵀp + p·a reconstructs -I exactly, and P is positive definite.
is_positive_definite(p).value();                    // true

// Rotation (eigenvalues ±i): i + (-i) = 0 => singular Kronecker sum => domain_error.
lyapunov_solve(mat({{0, -1}, {1, 0}}), Matrix::identity(2)).error();
                                                    // MathError::domain_error

// Discrete Stein AᵀPA - P = -Q for Schur-stable A = diag(1/2, 1/3).
const Matrix as = Matrix::from_rows(
    {{rr(1, 2), ri(0)}, {ri(0), rr(1, 3)}}).value();
stein_solve(as, Matrix::identity(2)).has_value();   // true (residual == -I)

// Sylvester positive-definiteness (exact rational minors).
is_positive_definite(mat({{2, 1}, {1, 2}})).value();   // true  (minors 2, 3 > 0)
is_positive_definite(mat({{1, 2}, {2, 1}})).value();   // false (det = -3 < 0)

// Exact Lyapunov stability, cross-checked against exact Routh–Hurwitz.
is_stable_lyapunov(a).value();                      // true
auto x = stability_cross_check(a).value();
x.lyapunov_stable; x.routh_hurwitz_stable; x.agree; // true, true, true

// diag(1,2): positive eigenvalues => unstable; both exact verdicts agree false.
is_stable_lyapunov(mat({{1, 0}, {0, 2}})).value();  // false

// Rotation: singular Lyapunov system => not stable, matching Routh–Hurwitz.
is_stable_lyapunov(mat({{0, -1}, {1, 0}})).value(); // false

// --- Lyapunov exponent (NUMERICAL) ----------------------------------------
// Constant Jacobian 2·I repeated 12x: leading exponent = log 2 ≈ 0.6931.
std::vector<Matrix> expanding(12, mat({{2, 0}, {0, 2}}));
lyapunov_exponent(std::span<const Matrix>(expanding)).value();  // ≈ log 2

// Constant Jacobian (1/2)·I repeated 12x: leading exponent = log(1/2) ≈ -0.6931.
const Matrix half_i = Matrix::from_rows(
    {{rr(1, 2), ri(0)}, {ri(0), rr(1, 2)}}).value();
std::vector<Matrix> contracting(12, half_i);
lyapunov_exponent(std::span<const Matrix>(contracting)).value(); // ≈ log(1/2)
```

## See also

- [`nimblecas.matrix`](matrix.md) — the exact `Rational` matrices, the inverse
  and solve behind the condition numbers and the Kronecker-sum Lyapunov/Stein
  systems.
- [`nimblecas.dynamics`](dynamics.md) — the exact Routh–Hurwitz
  `is_asymptotically_stable` that `stability_cross_check` corroborates.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field every norm,
  minor, and Lyapunov entry lives in.
- [`nimblecas.eigen`](eigen.md) — the rational spectrum underlying both the
  stability verdicts and the singular-value interpretation of `κ₂`.
- [Documentation hub](../Index.md)
