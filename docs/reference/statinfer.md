# `nimblecas.statinfer` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/statinfer/statinfer.cppm`

Exact **inferential statistics and linear regression** over the rationals
(ROADMAP §7.7.5). Given a design matrix `X` (`m` observations in rows, `n`
features in columns) and a response `y` of length `m`, the ordinary-least-squares
coefficients `β` are the solution of the **normal equations**

```
(Xᵀ X) β = Xᵀ y.
```

Every entry of `X` and `y` is a [`Rational`](ratpoly.md) — a reduced `int64`
fraction — and the normal system is solved over the field `Q` by exact
Gauss-Jordan elimination in [`nimblecas.matrix`](matrix.md), so `β` is the
fraction it mathematically **is**, never a `double` that happens to be close.
Ridge regression adds a rational penalty; weighted least squares a diagonal
weight matrix; the coefficient of determination `R²` is likewise an exact
rational.

```cpp
import nimblecas.statinfer;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), [`matrix`](matrix.md),
and [`stats`](stats.md).

## The exact normal-equations method

For OLS the estimator minimises the residual sum of squares
`‖y − Xβ‖²`. Setting the gradient to zero gives the normal equations
`Xᵀ X β = Xᵀ y`. Because the coefficient field is `Q`, `Xᵀ X` (an `n × n` Gram
matrix) and `Xᵀ y` (an `n × 1` vector) are formed by exact rational matrix
products, and the system is solved by `Matrix::solve` — plain Gauss-Jordan over
`Q`, no floating point. The estimator exists and is unique **iff** `Xᵀ X` is
nonsingular, i.e. `X` has full column rank `n` (which requires `m ≥ n`). The two
generalisations reuse the same machinery:

| Model | System solved |
| :--- | :--- |
| OLS | `(Xᵀ X) β = Xᵀ y` |
| Ridge (penalty `λ ≥ 0`) | `(Xᵀ X + λ I) β = Xᵀ y` |
| Weighted (diagonal `W = diag(w)`) | `(Xᵀ W X) β = Xᵀ W y` |

The intercept is **not** special-cased: to fit one, the caller supplies a
leading all-ones column in `X` (its coefficient is the intercept).

## Honesty boundary

Everything that is rational is returned **exactly**; nothing that would require
an irrational is faked (the convention shared with [`stats`](stats.md) and
[`probdist`](probdist.md)).

- **`β` and `R²` are exact over `Q`.** `R² = 1 − SS_res / SS_tot` with
  `SS_res = Σ(yᵢ − ŷᵢ)²` and `SS_tot = Σ(yᵢ − ȳ)²`, both exact rationals.
- **Standard errors, t-statistics and p-values are omitted.** The standard error
  of `βⱼ` is `√(Cov(β)ⱼⱼ)`, and the square root of a rational is in general
  irrational — outside the exact-`Q` contract. In its place,
  `coefficient_covariance` returns the **exact rational** covariance matrix
  `Cov(β) = σ̂² (Xᵀ X)⁻¹`, with `σ̂² = SS_res / (m − n)`. A caller may take square
  roots of its diagonal at whatever precision it chooses.
- **A singular `Xᵀ X` is an honest failure.** A rank-deficient design yields
  `MathError::domain_error`, never a bogus `β`.
- **Method of moments** is supported only in the modest **linear-fractional
  (Möbius)** form `E[X](θ) = (aθ + b)/(cθ + d)` — see below. A general
  method-of-moments engine is out of scope.

## API

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `ols` | `auto ols(const Matrix& X, std::span<const Rational> y) -> Result<std::vector<Rational>>` | The length-`n` coefficient vector `β` solving `Xᵀ X β = Xᵀ y` exactly. `y.size() != X.rows()`, or a rank-deficient design (singular `Xᵀ X`, which includes every `m < n`), fails `domain_error`. |
| `ridge` | `auto ridge(const Matrix& X, std::span<const Rational> y, const Rational& lambda) -> Result<std::vector<Rational>>` | Solve `(Xᵀ X + λ I) β = Xᵀ y` for `λ ≥ 0`. `λ = 0` reduces to `ols`. A negative `λ`, a length mismatch, or a singular penalised system fails `domain_error`. |
| `weighted_ols` | `auto weighted_ols(const Matrix& X, std::span<const Rational> y, std::span<const Rational> weights) -> Result<std::vector<Rational>>` | Solve `(Xᵀ W X) β = Xᵀ W y` for `W = diag(weights)`. `weights` has length `m` and each weight must be `≥ 0` (a zero weight drops that observation). A length mismatch, a negative weight, or a singular `Xᵀ W X` fails `domain_error`. |
| `predict` | `auto predict(const Matrix& X, std::span<const Rational> beta) -> Result<std::vector<Rational>>` | Fitted values `ŷ = X β` (length `m`). `beta.size() != X.cols()` fails `domain_error`. |
| `r_squared` | `auto r_squared(const Matrix& X, std::span<const Rational> y, std::span<const Rational> beta) -> Result<Rational>` | `R² = 1 − SS_res / SS_tot`, exact rational. Perfect fit → `1`; mean-only model → `0`. A constant response (`SS_tot = 0`) leaves `R²` undefined → `domain_error`; a length mismatch (`y` vs `X.rows()`, `beta` vs `X.cols()`) also fails `domain_error`. |
| `coefficient_covariance` | `auto coefficient_covariance(const Matrix& X, std::span<const Rational> y) -> Result<Matrix>` | The exact rational `Cov(β) = σ̂² (Xᵀ X)⁻¹`, `σ̂² = SS_res / (m − n)`. Requires `m > n` (positive residual dof) and nonsingular `Xᵀ X`; else `domain_error`. Standard errors `√diag` are omitted (irrational). |
| `method_of_moments` | `auto method_of_moments(const Rational& sample_mean, const Rational& a, const Rational& b, const Rational& c, const Rational& d) -> Result<Rational>` | Solve `mbar = (aθ + b)/(cθ + d)` for `θ = (b − mbar·d)/(mbar·c − a)`. `mbar·c − a = 0` (no unique `θ`) or a `θ` making `cθ + d = 0` fails `domain_error`. |

### Method-of-moments supported form

The moment equation must be a **linear-fractional function of the single
parameter** `θ`: `E[X](θ) = (aθ + b)/(cθ + d)`. Setting it equal to the sample
mean `m̄` gives a *linear* equation, solved exactly as
`θ = (b − m̄·d)/(m̄·c − a)`. This covers the common one-parameter families:

| Family | `E[X](θ)` | `(a, b, c, d)` | Recovered `θ` |
| :--- | :--- | :--- | :--- |
| Bernoulli(θ) / Poisson(θ) | `θ` | `(1, 0, 0, 1)` | `m̄` |
| Binomial(`N`, `p`), `N` known | `N·p` | `(N, 0, 0, 1)` | `m̄ / N` |
| Uniform(0, θ) | `θ / 2` | `(1/2, 0, 0, 1)` | `2·m̄` |
| Exponential rate λ | `1 / λ` | `(0, 1, 1, 0)` | `1 / m̄` |

For the affine sub-case `E[X] = aθ + b` set `c = 0, d = 1`. A general
method-of-moments engine (higher moments, multi-parameter, nonlinear systems) is
out of scope.

## Error model

| Condition | Error |
| :--- | :--- |
| `y.size() != X.rows()` (`ols`/`ridge`/`weighted_ols`/`r_squared`/`coefficient_covariance`) | `MathError::domain_error` |
| `beta.size() != X.cols()` (`predict`/`r_squared`) | `MathError::domain_error` |
| `weights.size() != X.rows()` (`weighted_ols`) | `MathError::domain_error` |
| Singular `Xᵀ X` / `Xᵀ X + λ I` / `Xᵀ W X` (rank-deficient design, incl. `m < n`) | `MathError::domain_error` |
| Negative ridge penalty `λ` | `MathError::domain_error` |
| Negative weight | `MathError::domain_error` |
| Constant response, `SS_tot = 0` (`r_squared`) | `MathError::domain_error` |
| `m ≤ n`, no residual dof (`coefficient_covariance`) | `MathError::domain_error` |
| No unique `θ` (`m̄·c − a = 0`) or model denominator vanishes (`method_of_moments`) | `MathError::domain_error` |
| An `int64` numerator or denominator computation wraps | `MathError::overflow` |

## Worked example

```cpp
import nimblecas.statinfer;
import nimblecas.ratpoly;
import nimblecas.matrix;
using namespace nimblecas;

auto ri  = [](std::int64_t v) { return Rational::from_int(v); };
auto mat = [](std::vector<std::vector<Rational>> r) {
    return Matrix::from_rows(std::move(r)).value();
};
auto sp  = [](const std::vector<Rational>& v) { return std::span<const Rational>{v}; };

// Fit y = 2x + 1 through the exactly-collinear points (0,1), (1,3), (2,5).
// Design [1, x] (leading ones column is the intercept).
auto X = mat({{ri(1), ri(0)}, {ri(1), ri(1)}, {ri(1), ri(2)}});
std::vector<Rational> y{ri(1), ri(3), ri(5)};

auto beta = ols(X, sp(y)).value();     // β = [1, 2]  (intercept 1, slope 2)
auto r2   = r_squared(X, sp(y), sp(beta)).value();  // 1  (perfect fit)

// Least-squares line through (0,0), (1,0), (2,2): β = [-1/3, 1], R² = 3/4.
std::vector<Rational> y2{ri(0), ri(0), ri(2)};
auto b2  = ols(X, sp(y2)).value();     // [-1/3, 1]
auto r22 = r_squared(X, sp(y2), sp(b2)).value();     // 3/4

// Exact rational coefficient covariance (σ̂² = 2/3):
auto cov = coefficient_covariance(X, sp(y2)).value();  // [[5/9, -1/3], [-1/3, 1/3]]

// Ridge with λ = 2 and weighted least squares reuse the same call shape:
auto br = ridge(X, sp(y2), ri(2)).value();
std::vector<Rational> w{ri(1), ri(1), ri(2)};
auto bw = weighted_ols(X, sp(y2), sp(w)).value();

// Method of moments for Exponential rate: E[X] = 1/λ, sample mean 1/4 ⇒ λ = 4.
auto lambda = method_of_moments(Rational::make(1, 4).value(),
                                ri(0), ri(1), ri(1), ri(0)).value();  // 4
```

## See also

- [`nimblecas.matrix`](matrix.md) — the exact linear algebra (`transpose`,
  `multiply`, `solve`, `inverse`) the normal equations are solved with.
- [`nimblecas.stats`](stats.md) — exact descriptive statistics; supplies the
  mean used by `R²` and the covariance matrix that composes with this module.
- [`nimblecas.probdist`](probdist.md) — exact probability distributions; the
  same honesty boundary (rational moments exact, `√` omitted).
- [`nimblecas.ipm`](ipm.md) — the exact interior-point / optimisation sibling in
  the numeric layer.
- [Documentation hub](../Index.md)
