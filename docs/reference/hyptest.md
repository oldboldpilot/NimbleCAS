# `nimblecas.hyptest` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/hyptest/hyptest.cppm`

Exact **hypothesis-test statistics** and **maximum-likelihood estimation** over the
rationals and the symbolic engine (ROADMAP §7.7.7). Two deliverables:

1. **Test statistics.** A statistic formed from `Rational` data by the four
   arithmetic operations is itself an exact `Rational`, returned as the fraction it
   mathematically **is** — never a `double`. Covered: one-/two-sample and paired
   `t²`, one-sample `z²`, chi-squared goodness-of-fit, chi-squared test of
   independence, variance-ratio `F`, and one-way ANOVA `F`, each with its exact value
   and integer degrees of freedom.
2. **Maximum-likelihood estimation.** For the standard one-parameter families the
   module builds the log-likelihood `ℓ(θ)` as an [`Expr`](symbolic.md), differentiates
   it with [`nimblecas.diff`](diff.md) to obtain the score `U(θ) = ∂ℓ/∂θ`, and records
   the closed-form MLE `θ̂` and per-observation Fisher information `i(θ)`. The point
   estimate from actual rational data (`p̂ = x̄`, `λ̂ = 1/x̄`, …) is an exact `Rational`
   via [`nimblecas.stats`](stats.md).

```cpp
import nimblecas.hyptest;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), [`matrix`](matrix.md),
[`stats`](stats.md), [`statinfer`](statinfer.md), [`probdist`](probdist.md),
[`symbolic`](symbolic.md), and [`diff`](diff.md).

## Honesty boundary

Everything rational is returned **exactly**; nothing that would require an irrational
or a transcendental is faked (the convention shared with [`stats`](stats.md),
[`statinfer`](statinfer.md), and [`probdist`](probdist.md)).

- **`t` and `z` are irrational; their squares are exact.** A `t`-statistic carries a
  `1/√(s²/n)` factor and a `z`-statistic a `1/√σ²` factor, so `t` and `z` are in
  general irrational. Following the `stats.cppm` `*_squared` convention, this module
  returns the exact rational **square** `t²` / `z²`. This is exactly the quantity a
  table lookup compares anyway, since `t² ∼ F(1, df)` and `z² ∼ χ²(1)`. The **sign**
  of `t` (or `z`) is recoverable exactly as `sign(x̄ − μ₀)`.
- **`χ²`, `F`, and ANOVA statistics are rational and returned directly.** `Σ(O−E)²/E`,
  `s_x²/s_y²`, and `(SS_b/df_b)/(SS_w/df_w)` are all exact over `Q`.
- **No p-values.** A p-value is the tail integral of a `t` / `χ²` / `F` / normal
  density — a **transcendental** number **not** representable over `Q`. This module
  returns **no** p-values. It returns the exact statistic and its degrees of freedom,
  plus `exceeds(statistic, critical)`, an exact rational comparison against a
  caller-supplied **rational** critical value (for a `t²` / `z²` statistic pass the
  **square** of the critical value). That is the honest decision rule; the transcendental
  CDF is deliberately absent rather than approximated and presented as exact.
- **The likelihood-ratio statistic is symbolic.** `G² = 2(ℓ(θ̂) − ℓ(θ₀))` is a
  difference of logarithms and is transcendental in general, so `log_likelihood_ratio`
  returns it as an exact **symbolic** `Expr` (it may contain `ln`), never a fabricated
  rational. The **Wald** and **score** statistics, whose closed forms **are** rational
  for the Bernoulli and Poisson families, are returned as exact `Rational`s.

## Test statistics

| Function | Statistic | df |
| :--- | :--- | :--- |
| `one_sample_t_squared(data, mu0)` | `t² = n(x̄−μ₀)²/s²` (sample variance `s²`) | `n−1` |
| `two_sample_t_squared(x, y)` | pooled `t² = (x̄−ȳ)²/(s_p²(1/n₁+1/n₂))`, `s_p² = ((n₁−1)s₁²+(n₂−1)s₂²)/(n₁+n₂−2)` | `n₁+n₂−2` |
| `paired_t_squared(x, y)` | one-sample `t²` of `dᵢ = xᵢ−yᵢ` against `0` | `n−1` |
| `z_squared(data, mu0, pop_variance)` | `z² = n(x̄−μ₀)²/σ²` (known `σ²`) | `1` (`z² ∼ χ²₁`) |
| `chi_squared_goodness_of_fit(observed, expected)` | `Σ(Oᵢ−Eᵢ)²/Eᵢ` | `k−1` |
| `chi_squared_independence(table)` | `Σ(O_ij−E_ij)²/E_ij`, `E_ij = RᵢCⱼ/N` | `(r−1)(c−1)` |
| `variance_ratio_f(x, y)` | `F = s_x²/s_y²` | `(n_x−1, n_y−1)` |
| `one_way_anova_f(groups)` | `F = (SS_b/(k−1))/(SS_w/(N−k))` | `(k−1, N−k)` |

Each returns `Result<TestStatistic>`:

```cpp
struct TestStatistic {
    Rational value;                  // the exact rational statistic (t²/z² for the t/z family)
    std::int64_t df1;                // (numerator) degrees of freedom
    std::optional<std::int64_t> df2; // denominator df for the F family; nullopt otherwise
};
```

The exact decision helper:

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `exceeds` | `auto exceeds(const Rational& statistic, const Rational& critical) -> Result<bool>` | `true` iff `statistic > critical`, by exact (cross-multiplied, overflow-checked) comparison. For a `t²`/`z²` statistic pass the **square** of the critical value. |

## Maximum-likelihood estimation

### Symbolic models

`*_mle_model()` returns `Result<MleModel>` for a family. Symbols: `parameter` (θ),
`"n"` (sample size), `"m"` (sample mean `x̄`), and for the Normal `"v"` (mean of
squares `(1/n)Σxᵢ²`) and `"sigma2"` (known variance).

```cpp
struct MleModel {
    std::string parameter;      // name of the θ symbol
    Expr log_likelihood;        // ℓ(θ), θ-dependent part (additive constants dropped)
    Expr score;                 // U(θ) = ∂ℓ/∂θ, computed by differentiate()
    Expr mle;                   // θ̂ as an Expr in the summary symbols
    Expr fisher_information;    // per-observation i(θ); total sample info is n·i(θ)
};
```

Because `score` is the actual `differentiate(log_likelihood, parameter)`, substituting
`θ = mle` and running `simplify` yields `0` — the defining property of the MLE, which
the test suite verifies symbolically for Bernoulli, Poisson, Exponential, and Normal-mean.

| Family | `ℓ(θ)` (θ-part) | Score `U(θ)` | `θ̂` | `i(θ)` |
| :--- | :--- | :--- | :--- | :--- |
| `bernoulli_mle_model` | `nm·ln p + n(1−m)·ln(1−p)` | `nm/p − n(1−m)/(1−p)` | `m` | `1/(p(1−p))` |
| `poisson_mle_model` | `nm·ln λ − nλ` | `nm/λ − n` | `m` | `1/λ` |
| `exponential_mle_model` | `n·ln λ − nmλ` | `n/λ − nm` | `m⁻¹` | `1/λ²` |
| `normal_mean_mle_model` | `−(1/2σ²)(nv − 2nmμ + nμ²)` | `(nm − nμ)/σ²` | `m` | `1/σ²` |
| `geometric_mle_model` | `n·ln p + n(m−1)·ln(1−p)` | `n/p − n(m−1)/(1−p)` | `m⁻¹` | `1/(p²(1−p))` |

The Fisher information is a **per-observation** closed form (the `probdist`-style
hand-built catalog value); the total sample information is `n·i(θ)`. `i(p)` for the
Bernoulli/Geometric is written as a product of reciprocals `p⁻¹(1−p)⁻¹` so the identity
`i(p)·p·(1−p) = 1` folds by like-base cancellation in a single Cohen pass.

### Exact rational point estimates

| Function | Estimate | Fails |
| :--- | :--- | :--- |
| `bernoulli_mle(data)` | `p̂ = x̄` | empty data |
| `poisson_mle(data)` | `λ̂ = x̄` | empty data |
| `normal_mean_mle(data)` | `μ̂ = x̄` | empty data |
| `exponential_mle(data)` | `λ̂ = 1/x̄` | empty data, `x̄ = 0` |
| `geometric_mle(data)` | `p̂ = 1/x̄` | empty data, `x̄ = 0` |
| `normal_variance_mle(data)` | `σ̂² = (1/n)Σ(xᵢ−x̄)²` (population variance) | empty data |

### Wald / score / likelihood-ratio

| Function | Statistic | Fails |
| :--- | :--- | :--- |
| `bernoulli_wald_statistic(data, p0)` | `n(p̂−p₀)²/(p̂(1−p̂))` | empty data, `p̂ ∈ {0,1}` |
| `bernoulli_score_statistic(data, p0)` | `n(x̄−p₀)²/(p₀(1−p₀))` | empty data, `p₀ ∈ {0,1}` |
| `poisson_wald_statistic(data, lambda0)` | `n(λ̂−λ₀)²/λ̂` | empty data, `λ̂ = 0` |
| `poisson_score_statistic(data, lambda0)` | `n(x̄−λ₀)²/λ₀` | empty data, `λ₀ = 0` |
| `log_likelihood_ratio(model, θ̂, θ₀)` | `simplify(2(ℓ(θ̂) − ℓ(θ₀)))` — exact **symbolic** `Expr` (contains `ln`) | simplify overflow |

## Error model

| Condition | Error |
| :--- | :--- |
| Empty / too-small sample (`n<2` for `t²`/`F`/variance; empty for MLE) | `MathError::domain_error` |
| Length mismatch (`paired_t_squared`, `chi_squared_goodness_of_fit`) | `MathError::domain_error` |
| Zero variance / zero `SS_within` / zero pooled variance (statistic undefined) | `MathError::domain_error` |
| Non-positive expected count, row/column/grand total, or known variance | `MathError::domain_error` |
| Fewer than 2 categories / rows / columns / groups | `MathError::domain_error` |
| `x̄ = 0` in a reciprocal MLE; `p̂/p₀ ∈ {0,1}` or `λ̂/λ₀ = 0` in Wald/score | `MathError::domain_error` |
| An `int64` numerator/denominator computation wraps | `MathError::overflow` |

## Worked example

```cpp
import nimblecas.hyptest;
import nimblecas.ratpoly;
import nimblecas.symbolic;
import nimblecas.simplify;
using namespace nimblecas;

auto ri = [](std::int64_t v) { return Rational::from_int(v); };
auto sp = [](const std::vector<Rational>& v) { return std::span<const Rational>{v}; };

// Exact one-sample t² on 1,2,3,4 against μ₀ = 2: t² = 3/5, df = 3.
std::vector<Rational> x{ri(1), ri(2), ri(3), ri(4)};
auto ts = one_sample_t_squared(sp(x), ri(2)).value();   // ts.value == 3/5, ts.df1 == 3

// Exact chi-squared goodness-of-fit: Σ(O−E)²/E = 10/3, df = 2.
std::vector<Rational> O{ri(10), ri(20), ri(30)}, E{ri(15), ri(15), ri(30)};
auto gof = chi_squared_goodness_of_fit(sp(O), sp(E)).value();  // 10/3, df 2

// Exact Bernoulli MLE p̂ = x̄ = 3/5 on 1,0,1,1,0.
std::vector<Rational> b{ri(1), ri(0), ri(1), ri(1), ri(0)};
auto phat = bernoulli_mle(sp(b)).value();               // 3/5

// The Poisson score vanishes at the MLE (symbolically):
auto model = poisson_mle_model().value();
auto at_mle = substitute(model.score, Expr::symbol(model.parameter), model.mle);
auto zero = simplify(at_mle).value();                   // Expr::integer(0)
```

## See also

- [`nimblecas.stats`](stats.md) — exact descriptive statistics (mean, variance) that
  supply the MLE point estimates and the `t`/`F`/ANOVA building blocks; same
  `*_squared` honesty convention.
- [`nimblecas.statinfer`](statinfer.md) — exact regression / method of moments; the
  inferential sibling with the same rational honesty boundary.
- [`nimblecas.probdist`](probdist.md) — exact symbolic distribution catalog (the
  reference `t`/`χ²`/`F`/normal families whose CDFs the omitted p-values would need).
- [`nimblecas.diff`](diff.md) / [`nimblecas.symbolic`](symbolic.md) — the symbolic
  differentiation and expression trees that produce the score and Fisher information.
- [Documentation hub](../Index.md)
```
