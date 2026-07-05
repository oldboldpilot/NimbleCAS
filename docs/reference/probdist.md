# `nimblecas.probdist` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/probdist/probdist.cppm`

An **exact symbolic** catalog of standard probability distributions (ROADMAP
§7.7.3 / §7.7.4 / §7.7.6). For each distribution it returns the **moment
generating function** `M_X(t) = E[e^{tX}]` — and, for the integer-supported
families, the **probability generating function** `G_X(z) = E[z^X]` — together
with the **mean** and **variance**, all as [`Expr`](symbolic.md) trees in the
parameter symbols and a transform variable (`t` for the MGF, `z` for the PGF).
Nothing is numeric: a parameter such as `p`, `lambda`, `mu`, or `sigma^2` is a
symbol (or any caller-supplied `Expr`), so every result is the expression it
mathematically *is*. On top of the catalog it provides the **moment / cumulant**
machinery driven by [`differentiate`](diff.md), and a set of classical **tail
inequalities** returned as symbolic bound expressions.

```cpp
import nimblecas.probdist;
```

Depends on [`core`](core.md), [`symbolic`](symbolic.md), [`diff`](diff.md), and
[`simplify`](simplify.md).

## The distribution catalog

Each constructor returns a `DistInfo`:

```cpp
struct DistInfo {
    std::optional<Expr> mgf;   // moment generating function  M_X(t)  (nullopt if none)
    std::optional<Expr> pgf;   // probability generating fn    G_X(z)  (discrete only)
    Expr mean;                 // E[X]
    Expr variance;             // Var(X)
};
```

Both `mgf` and `pgf` are `std::optional<Expr>`. `pgf` is present only for the
integer-supported (discrete) families. `mgf` is **`std::nullopt` for every family
whose moment generating function has no elementary closed form or does not exist
at all** — this is an honest not-available signal (Rule 32); a plausible-but-wrong
closed form is never fabricated in its place (see the honesty boundary below).

| Distribution | Constructor | MGF `M_X(t)` | PGF `G_X(z)` | Mean | Variance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Bernoulli(`p`) | `bernoulli(p)` | `1 - p + p e^t` | `1 - p + p z` | `p` | `p(1-p)` |
| Binomial(`n`,`p`) | `binomial(n, p)` | `(1 - p + p e^t)^n` | `(1 - p + p z)^n` | `n p` | `n p (1-p)` |
| Poisson(`lambda`) | `poisson(lambda)` | `exp(lambda(e^t - 1))` | `exp(lambda(z - 1))` | `lambda` | `lambda` |
| Geometric(`p`), support `k = 1,2,…` | `geometric(p)` | `p e^t / (1 - (1-p) e^t)` | `p z / (1 - (1-p) z)` | `1/p` | `(1-p)/p^2` |
| DiscreteUniform(`a`,`b`), `n = b-a+1` | `discrete_uniform(a, b)` | `(e^{at} - e^{(b+1)t}) / (n(1 - e^t))` | `(z^a - z^{b+1}) / (n(1 - z))` | `(a+b)/2` | `(n^2 - 1)/12` |
| NegativeBinomial(`r`,`p`) | `negative_binomial(r, p)` | `(p/(1 - (1-p) e^t))^r` | `(p/(1 - (1-p) z))^r` | `r(1-p)/p` | `r(1-p)/p^2` |
| Hypergeometric(`N`,`K`,`n`) | `hypergeometric(N, K, n)` | — (no elementary form) | — (no elementary form) | `nK/N` | `n(K/N)((N-K)/N)((N-n)/(N-1))` |
| Exponential(`lambda`) | `exponential(lambda)` | `lambda/(lambda - t)` | — | `1/lambda` | `1/lambda^2` |
| Normal(`mu`, `sigma^2`) | `normal(mu, sigma2)` | `exp(mu t + sigma^2 t^2 / 2)` | — | `mu` | `sigma^2` |
| Gamma(`alpha`, `theta`) | `gamma(alpha, theta)` | `(1 - theta t)^{-alpha}` | — | `alpha theta` | `alpha theta^2` |
| ContinuousUniform(`a`,`b`) | `continuous_uniform(a, b)` | `(e^{tb} - e^{ta}) / (t(b-a))` | — | `(a+b)/2` | `(b-a)^2/12` |
| ChiSquared(`k`) | `chi_squared(k)` | `(1 - 2t)^{-k/2}` | — | `k` | `2k` |
| Student-t(`nu`) | `student_t(nu)` | — (MGF does not exist) | — | `0` (`nu>1`) | `nu/(nu-2)` (`nu>2`) |
| Beta(`alpha`,`beta`) | `beta(alpha, beta)` | — (confluent hypergeometric) | — | `alpha/(alpha+beta)` | `alpha beta/((alpha+beta)^2 (alpha+beta+1))` |
| Weibull(`k`,`lambda`) | `weibull(k, lambda)` | — (no elementary form) | — | `lambda Γ(1+1/k)` | `lambda^2 (Γ(1+2/k) - Γ(1+1/k)^2)` |
| Pareto(`xm`,`alpha`) | `pareto(xm, alpha)` | — (MGF does not exist) | — | `alpha xm/(alpha-1)` (`alpha>1`) | `xm^2 alpha/((alpha-1)^2 (alpha-2))` (`alpha>2`) |
| Log-normal(`mu`,`sigma^2`) | `lognormal(mu, sigma2)` | — (MGF diverges) | — | `exp(mu + sigma^2/2)` | `(exp(sigma^2)-1) exp(2mu+sigma^2)` |

The continuous families have `pgf == std::nullopt`. `Geometric` is the "number of
trials up to and including the first success" convention (support `{1, 2, …}`),
hence mean `1/p`; `NegativeBinomial(r, p)` is the "number of **failures** before
the `r`-th success" (Pascal) convention, support `{0, 1, …}`. `DiscreteUniform`
is over the integers `{a, a+1, …, b}` with `n = b - a + 1` equally likely values.
`Normal` and `Log-normal` take the **variance** `sigma^2` as their second
argument (written `sigma2` in code), not the standard deviation. `Gamma` uses the
shape–scale parametrisation `(alpha, theta)`. `Weibull` uses shape `k` and scale
`lambda`, and its moments use the Gamma function `Γ = apply("gamma", ·)`.

### Distributions with no moment generating function

A `mgf == std::nullopt` means one of two mathematically distinct situations,
documented per family so the caller is never misled:

- **No elementary closed form** (the MGF exists analytically but only as a special
  function): **Hypergeometric** and **Beta** (Gauss / confluent hypergeometric
  forms), and **Weibull** (an infinite series of Gamma values). Their PGFs are
  likewise not elementary.
- **The MGF does not exist** on any neighbourhood of `t = 0`: **Student-t** (the
  defining integral diverges for every `t ≠ 0`), **Pareto** (diverges for every
  `t > 0`), and **Log-normal** (diverges for every `t > 0`). Their moments still
  have closed forms and are returned.

In all six cases the mean and variance are exact closed-form `Expr`s; only the
`mgf` (and, for the discrete Hypergeometric, the `pgf`) is withheld.

## How division and `exp` are represented

To stay inside the engine's canonical algebra, this module builds every
expression from `Expr::symbol/integer/sum/product/power` and `Expr::apply`:

- **Division** is multiplication by a reciprocal **power**: `x^{-1} =
  Expr::power(x, Expr::integer(-1))`, and `a / b = Expr::product({a, power(b,
  -1)})`. No `Expr::rational` is needed anywhere in this module; the Normal's
  `1/2` is written `2^{-1} = Expr::power(Expr::integer(2), Expr::integer(-1))`.
- **`exp`** is `Expr::apply("exp", {arg})`, and the natural log used for the
  cumulant generating function is `Expr::apply("ln", {M})` — the name `ln` (not
  `log`) is the one [`differentiate`](diff.md) recognises, so `d/dt log M_X(t)`
  actually reduces through the table entry `ln' = 1/u`.

## Moments and cumulants from the MGF

```cpp
[[nodiscard]] auto raw_moment(const Expr& mgf, std::size_t k) -> Result<Expr>;
[[nodiscard]] auto cumulant (const Expr& mgf, std::size_t k) -> Result<Expr>;
```

- **`raw_moment(M, k)`** returns the `k`-th raw moment
  `E[X^k] = [ d^k/dt^k M_X(t) ]_{t=0}`. It differentiates `M` `k` times with
  respect to `t` via [`differentiate`](diff.md), substitutes `t = 0` with
  [`substitute`](symbolic.md), and runs [`simplify`](simplify.md). `k = 0` yields
  `M_X(0) = 1`.
- **`cumulant(M, k)`** returns the `k`-th cumulant
  `kappa_k = [ d^k/dt^k log M_X(t) ]_{t=0}`, taking the cumulant generating
  function as `ln(M_X(t))`. Then `kappa_1` is the mean and `kappa_2` is the
  variance.

Both propagate any `MathError` that differentiation or simplification surfaces
(e.g. `overflow`); otherwise they return the exact symbolic result.

### What "exact but possibly unsimplified" means here

The results are **always exact**. Whether they collapse to the textbook closed
form depends on what the engine's automatic simplifier does — and, by design, it
does **not** evaluate a transcendental function at a point: `exp(0)` is left as
`exp(0)`, not folded to `1` (see [`simplify`](simplify.md)). Consequently:

- **Rational / polynomial MGFs** (Exponential, Gamma) reduce fully. For example
  `raw_moment(*exponential(lambda).mgf, 1)` simplifies to `lambda^{-1}`, and
  `raw_moment(*gamma(alpha, theta).mgf, 1)` to `alpha·theta` — because at `t = 0`
  a factor like `(1 - theta·t)^{…}` becomes `1^{…} = 1` purely by rational
  simplification.
- **Cumulants of `exp`-based MGFs** reduce when the leading exponential cancels:
  `d/dt ln(M) = M^{-1}·M'` becomes `exp(A)^{-1}·exp(A)·A' → A'` because
  `simplify` combines like bases (`exp(A)^{-1}·exp(A)^{1} = exp(A)^0 = 1`). This
  is why `cumulant(*normal(mu, sigma2).mgf, 1) = mu` and
  `cumulant(…, 2) = sigma^2` come back as the bare parameters.
- **Raw moments of `exp`-based MGFs** (e.g. `raw_moment(*poisson.mgf, 1)`) are
  returned in an exact **but not fully reduced** form: a residual `exp(0)` factor
  remains because the simplifier will not evaluate it. The value is still
  correct; it is simply `lambda·exp(0)` rather than `lambda`. No fabricated
  "cleaned up" answer is ever returned in place of the honest expression
  (Rule 32).

If you need the numeric collapse of such residuals, substitute the transform
variable and evaluate downstream; this module refuses to *assert* a simplified
form it did not actually derive.

## Integral transforms of the generating functions

```cpp
[[nodiscard]] auto characteristic_function(const Expr& mgf) -> Expr;
[[nodiscard]] auto factorial_moment(const Expr& pgf, std::size_t k) -> Result<Expr>;
[[nodiscard]] auto laplace_stieltjes(const Expr& mgf) -> Expr;
```

- **`characteristic_function(M)`** returns `phi_X(t) = E[e^{i t X}] = M_X(i t)`,
  formed by the formal substitution `t → i·t`. The imaginary unit is represented
  as `i = (-1)^{1/2} = Expr::power(Expr::integer(-1), Expr::power(Expr::integer(2),
  Expr::integer(-1)))`. **No** simplification of the resulting `i^2 = -1` is
  performed — the result is the exact substituted `Expr`. Unlike the MGF, `phi_X`
  always exists; this helper only performs the formal substitution on a supplied
  MGF expression, so pass one only when the MGF is available.
- **`factorial_moment(G, k)`** returns the `k`-th factorial moment
  `E[X(X-1)…(X-k+1)] = [ d^k/dz^k G_X(z) ]_{z=1}`. It differentiates the PGF `k`
  times with respect to `z` via [`differentiate`](diff.md), substitutes `z = 1`,
  and runs [`simplify`](simplify.md); `k = 0` yields `G_X(1) = 1`. Exact but may be
  **unsimplified**: for a polynomial PGF (e.g. Binomial) it reduces fully — the
  first factorial moment of `binomial(n, p)` collapses to `n·p` — whereas for an
  `exp`-based PGF (e.g. Poisson) the residual `exp(0)` remains, so the `k`-th
  factorial moment of `poisson(lambda)` comes back as `lambda^k · exp(0)` (still
  exact, never a fabricated `lambda^k`). Propagates any `MathError`.
- **`laplace_stieltjes(M)`** returns `LST_X(s) = E[e^{-s X}] = M_X(-s)`, formed by
  the formal substitution `t → -s` (transform variable `s`). Exact substituted
  `Expr`, no simplification — e.g. the Exponential returns `lambda/(lambda-(-s))`,
  i.e. `lambda/(lambda+s)`.

## Tail inequalities

Each helper returns the **right-hand side** of a classical bound as an `Expr`.
It is an exact inequality **only under the stated hypotheses** — it is not a
claim of tightness.

| Bound | Function | Inequality | Hypotheses |
| :--- | :--- | :--- | :--- |
| Markov | `markov_bound(mean, alpha)` | `P(X ≥ alpha) ≤ E[X] / alpha` | `X ≥ 0` almost surely, `alpha > 0`. |
| Chebyshev | `chebyshev_bound(variance, k)` | `P(|X − mu| ≥ k) ≤ sigma^2 / k^2` | finite variance `sigma^2`, `k > 0`. |
| Cantelli (one-sided) | `cantelli_bound(variance, k)` | `P(X − mu ≥ k) ≤ sigma^2 / (sigma^2 + k^2)` | finite variance `sigma^2`, `k > 0`. |
| Chernoff (generic) | `chernoff_bound(mgf, alpha)` | `P(X ≥ alpha) ≤ e^{−t·alpha} M_X(t)` for every `t > 0` | `M_X(t)` finite for the chosen `t > 0`. |

### Concentration inequalities

For sums and functions of independent random variables. Each helper builds the
sum of squares of the supplied per-variable constants internally; the constant
vectors must be **non-empty**.

| Bound | Function | Inequality | Hypotheses |
| :--- | :--- | :--- | :--- |
| Hoeffding | `hoeffding_bound(t, widths)` | `P(S − E[S] ≥ t) ≤ exp(−2 t^2 / Σ_i (b_i − a_i)^2)` | `S = Σ_i X_i` with independent `X_i ∈ [a_i, b_i]`; `t > 0`. `widths` holds the range widths `(b_i − a_i)`. |
| Bernstein | `bernstein_bound(t, variance, bound)` | `P(S ≥ t) ≤ exp(−t^2 / (2(v + M t/3)))` | independent mean-zero `X_i` with `|X_i| ≤ M` (`bound`) and total variance `v = Σ_i Var(X_i)` (`variance`); `t > 0`. |
| McDiarmid | `mcdiarmid_bound(t, diffs)` | `P(f − E[f] ≥ t) ≤ exp(−2 t^2 / Σ_i c_i^2)` | `f` has bounded differences `c_i` (changing coordinate `i` moves `f` by at most `c_i`); independent inputs; `t > 0`. |
| Azuma–Hoeffding | `azuma_bound(t, diffs)` | `P(X_n − X_0 ≥ t) ≤ exp(−t^2 / (2 Σ_i c_i^2))` | martingale (or super-martingale) with bounded differences `|X_k − X_{k-1}| ≤ c_k`; `t > 0`. |

The Chernoff helper returns the **pre-optimization** bound `e^{−t·alpha}·M_X(t)`;
the tightest bound is the infimum over `t > 0`, which is the caller's to take
(differentiate the bound in `t`, solve the stationarity condition, and
substitute — the ingredients are all `Expr`s this module produces).

## Honesty boundary

- Generating functions, mean, and variance are exact closed forms and **cannot
  fail**. Where a family has no elementary (or no existing) MGF/PGF, the field is
  `std::nullopt` — a truthful not-available signal, never a fabricated form.
- `raw_moment` / `cumulant` / `factorial_moment` are exact symbolic derivatives
  (at `t = 0` or `z = 1`). They may be returned unsimplified (e.g. a residual
  `exp(0)`; see above) and propagate a `MathError` from the differentiation /
  simplification pipeline; they never fabricate a simplified value.
- `characteristic_function` / `laplace_stieltjes` are exact formal substitutions
  (`t → i·t`, `t → -s`) of a supplied generating function; they perform no
  simplification and cannot fail.
- Tail-bound and concentration helpers return the exact bounding expression under
  the hypotheses tabulated above; they encode inequalities, not equalities, and
  make no tightness claim.

## API summary

| Name | Signature | Returns |
| :--- | :--- | :--- |
| `bernoulli` | `auto bernoulli(const Expr& p) -> DistInfo` | Bernoulli catalog entry. |
| `binomial` | `auto binomial(const Expr& n, const Expr& p) -> DistInfo` | Binomial catalog entry. |
| `poisson` | `auto poisson(const Expr& lambda) -> DistInfo` | Poisson catalog entry. |
| `geometric` | `auto geometric(const Expr& p) -> DistInfo` | Geometric catalog entry (support `1,2,…`). |
| `exponential` | `auto exponential(const Expr& lambda) -> DistInfo` | Exponential catalog entry (no PGF). |
| `normal` | `auto normal(const Expr& mu, const Expr& sigma2) -> DistInfo` | Normal catalog entry (`sigma2` = variance; no PGF). |
| `gamma` | `auto gamma(const Expr& alpha, const Expr& theta) -> DistInfo` | Gamma catalog entry (shape/scale; no PGF). |
| `discrete_uniform` | `auto discrete_uniform(const Expr& a, const Expr& b) -> DistInfo` | Discrete Uniform on `{a,…,b}`. |
| `negative_binomial` | `auto negative_binomial(const Expr& r, const Expr& p) -> DistInfo` | Negative Binomial (failures before `r`-th success). |
| `hypergeometric` | `auto hypergeometric(const Expr& N, const Expr& K, const Expr& n) -> DistInfo` | Hypergeometric (`mgf`/`pgf` = `std::nullopt`). |
| `continuous_uniform` | `auto continuous_uniform(const Expr& a, const Expr& b) -> DistInfo` | Continuous Uniform on `[a,b]` (no PGF). |
| `chi_squared` | `auto chi_squared(const Expr& k) -> DistInfo` | Chi-squared with `k` d.o.f. (no PGF). |
| `student_t` | `auto student_t(const Expr& nu) -> DistInfo` | Student-t (`mgf` = `std::nullopt`; no PGF). |
| `beta` | `auto beta(const Expr& alpha, const Expr& beta) -> DistInfo` | Beta (`mgf` = `std::nullopt`; no PGF). |
| `weibull` | `auto weibull(const Expr& k, const Expr& lambda) -> DistInfo` | Weibull shape/scale (`mgf` = `std::nullopt`; no PGF). |
| `pareto` | `auto pareto(const Expr& xm, const Expr& alpha) -> DistInfo` | Pareto scale/shape (`mgf` = `std::nullopt`; no PGF). |
| `lognormal` | `auto lognormal(const Expr& mu, const Expr& sigma2) -> DistInfo` | Log-normal (`mgf` = `std::nullopt`; no PGF). |
| `raw_moment` | `auto raw_moment(const Expr& mgf, std::size_t k) -> Result<Expr>` | `E[X^k]` from the MGF. |
| `cumulant` | `auto cumulant(const Expr& mgf, std::size_t k) -> Result<Expr>` | `kappa_k` from the MGF. |
| `characteristic_function` | `auto characteristic_function(const Expr& mgf) -> Expr` | `phi_X(t) = M_X(i t)` (`t → i·t`). |
| `factorial_moment` | `auto factorial_moment(const Expr& pgf, std::size_t k) -> Result<Expr>` | `E[X(X-1)…(X-k+1)]` from the PGF. |
| `laplace_stieltjes` | `auto laplace_stieltjes(const Expr& mgf) -> Expr` | `LST_X(s) = M_X(-s)` (`t → -s`). |
| `markov_bound` | `auto markov_bound(const Expr& mean, const Expr& alpha) -> Expr` | `E[X]/alpha`. |
| `chebyshev_bound` | `auto chebyshev_bound(const Expr& variance, const Expr& k) -> Expr` | `sigma^2/k^2`. |
| `cantelli_bound` | `auto cantelli_bound(const Expr& variance, const Expr& k) -> Expr` | `sigma^2/(sigma^2+k^2)`. |
| `chernoff_bound` | `auto chernoff_bound(const Expr& mgf, const Expr& alpha) -> Expr` | `e^{-t·alpha}·M_X(t)`. |
| `hoeffding_bound` | `auto hoeffding_bound(const Expr& t, const std::vector<Expr>& widths) -> Expr` | `exp(-2 t^2 / Σ(b_i-a_i)^2)`. |
| `bernstein_bound` | `auto bernstein_bound(const Expr& t, const Expr& variance, const Expr& bound) -> Expr` | `exp(-t^2/(2(v + M t/3)))`. |
| `mcdiarmid_bound` | `auto mcdiarmid_bound(const Expr& t, const std::vector<Expr>& diffs) -> Expr` | `exp(-2 t^2 / Σ c_i^2)`. |
| `azuma_bound` | `auto azuma_bound(const Expr& t, const std::vector<Expr>& diffs) -> Expr` | `exp(-t^2/(2 Σ c_i^2))`. |

Constants `mgf_variable == "t"`, `pgf_variable == "z"`, and `laplace_variable ==
"s"` name the transform variables (all `std::string_view`).

## Error model

| Condition | Result |
| :--- | :--- |
| Any catalog constructor (`bernoulli` … `lognormal`) | Never fails — returns a `DistInfo` by value (with `mgf`/`pgf` possibly `std::nullopt`). |
| Any tail-bound / concentration helper, `characteristic_function`, `laplace_stieltjes` | Never fails — returns an `Expr` by value. |
| `raw_moment` / `cumulant` / `factorial_moment`, differentiation or simplification overflows an `int64` | `MathError::overflow` (propagated from [`simplify`](simplify.md)). |
| `raw_moment` / `cumulant` / `factorial_moment`, undefined algebraic form encountered while simplifying | the corresponding `MathError` (e.g. `undefined_value`, `division_by_zero`) is propagated. |

There is no `domain_error` surface of its own: parameters are opaque symbolic
`Expr`s, so parameter-range hypotheses (such as `0 ≤ p ≤ 1` or `lambda > 0`) are
the caller's to honour and are documented, not enforced.

## Example

```cpp
import nimblecas.probdist;
import nimblecas.symbolic;
using namespace nimblecas;

const Expr lambda = Expr::symbol("lambda");
const Expr mu     = Expr::symbol("mu");
const Expr sigma2 = Expr::symbol("sigma2");

// Catalog entries. `mgf` / `pgf` are std::optional<Expr>; the continuous families
// below always have an MGF, so *expo.mgf / *norm.mgf are engaged.
const auto expo = exponential(lambda);   // MGF lambda/(lambda - t)
const auto norm = normal(mu, sigma2);     // MGF exp(mu t + sigma^2 t^2 / 2)

// Exact raw moments from the rational MGF reduce fully:
auto m1 = raw_moment(*expo.mgf, 1).value();  // lambda^(-1)  == expo.mean
auto m2 = raw_moment(*expo.mgf, 2).value();  // 2 * lambda^(-2)

// Cumulants of the Normal MGF: kappa_1 = mean, kappa_2 = variance.
auto k1 = cumulant(*norm.mgf, 1).value();    // mu     == norm.mean
auto k2 = cumulant(*norm.mgf, 2).value();    // sigma2 == norm.variance

// Characteristic function and Laplace–Stieltjes transform (formal substitutions).
auto phi = characteristic_function(*norm.mgf);  // t -> i*t
auto lst = laplace_stieltjes(*expo.mgf);        // t -> -s : lambda/(lambda + s)

// A family with no MGF signals it honestly (Rule 32):
const auto st = student_t(Expr::symbol("nu"));
bool has_mgf = st.mgf.has_value();           // false — Student-t MGF does not exist
auto var_t   = st.variance;                   // nu/(nu - 2)

// Factorial moments from a PGF (Binomial reduces fully to n p at z = 1).
const auto bin = binomial(Expr::symbol("n"), Expr::symbol("p"));
auto fm1 = factorial_moment(*bin.pgf, 1).value();  // n p

// Tail / concentration bounds (symbolic RHS of the inequality).
const Expr alpha = Expr::symbol("alpha");
const Expr k     = Expr::symbol("k");
const Expr tt    = Expr::symbol("t");
auto markov   = markov_bound(expo.mean, alpha);      // (1/lambda)/alpha
auto cheb     = chebyshev_bound(norm.variance, k);   // sigma^2 / k^2
auto chernoff = chernoff_bound(*expo.mgf, alpha);    // e^{-t*alpha} * lambda/(lambda - t)
auto hoeff    = hoeffding_bound(tt, {Expr::symbol("w1"), Expr::symbol("w2")});
                                                     // exp(-2 t^2 / (w1^2 + w2^2))
```

## See also

- [`nimblecas.stats`](stats.md) — exact *empirical* mean / variance / covariance
  over `Rational` data (the numeric-sample counterpart to this symbolic catalog).
- [`nimblecas.symbolic`](symbolic.md) — the `Expr` model, `substitute`, and
  structural equality used throughout.
- [`nimblecas.diff`](diff.md) — the `differentiate` engine that powers
  `raw_moment` and `cumulant`.
- [`nimblecas.simplify`](simplify.md) — the automatic simplifier that reduces the
  derivative forms (and deliberately leaves `exp(0)` unevaluated).
- [Documentation hub](../Index.md)
```
