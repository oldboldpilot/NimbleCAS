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
    Expr mgf;                  // moment generating function  M_X(t)
    std::optional<Expr> pgf;   // probability generating fn    G_X(z)  (discrete only)
    Expr mean;                 // E[X]
    Expr variance;             // Var(X)
};
```

| Distribution | Constructor | MGF `M_X(t)` | PGF `G_X(z)` | Mean | Variance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Bernoulli(`p`) | `bernoulli(p)` | `1 - p + p e^t` | `1 - p + p z` | `p` | `p(1-p)` |
| Binomial(`n`,`p`) | `binomial(n, p)` | `(1 - p + p e^t)^n` | `(1 - p + p z)^n` | `n p` | `n p (1-p)` |
| Poisson(`lambda`) | `poisson(lambda)` | `exp(lambda(e^t - 1))` | `exp(lambda(z - 1))` | `lambda` | `lambda` |
| Geometric(`p`), support `k = 1,2,…` | `geometric(p)` | `p e^t / (1 - (1-p) e^t)` | `p z / (1 - (1-p) z)` | `1/p` | `(1-p)/p^2` |
| Exponential(`lambda`) | `exponential(lambda)` | `lambda/(lambda - t)` | — | `1/lambda` | `1/lambda^2` |
| Normal(`mu`, `sigma^2`) | `normal(mu, sigma2)` | `exp(mu t + sigma^2 t^2 / 2)` | — | `mu` | `sigma^2` |
| Gamma(`alpha`, `theta`) | `gamma(alpha, theta)` | `(1 - theta t)^{-alpha}` | — | `alpha theta` | `alpha theta^2` |

The three continuous families (Exponential, Normal, Gamma) have `pgf ==
std::nullopt`. `Geometric` is the "number of trials up to and including the first
success" convention (support `{1, 2, …}`), hence mean `1/p`. `Normal` takes the
**variance** `sigma^2` as its second argument (written `sigma2` in code), not the
standard deviation. `Gamma` uses the shape–scale parametrisation `(alpha,
theta)`.

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
  `raw_moment(exponential(lambda).mgf, 1)` simplifies to `lambda^{-1}`, and
  `raw_moment(gamma(alpha, theta).mgf, 1)` to `alpha·theta` — because at `t = 0`
  a factor like `(1 - theta·t)^{…}` becomes `1^{…} = 1` purely by rational
  simplification.
- **Cumulants of `exp`-based MGFs** reduce when the leading exponential cancels:
  `d/dt ln(M) = M^{-1}·M'` becomes `exp(A)^{-1}·exp(A)·A' → A'` because
  `simplify` combines like bases (`exp(A)^{-1}·exp(A)^{1} = exp(A)^0 = 1`). This
  is why `cumulant(normal(mu, sigma2).mgf, 1) = mu` and
  `cumulant(…, 2) = sigma^2` come back as the bare parameters.
- **Raw moments of `exp`-based MGFs** (e.g. `raw_moment(poisson.mgf, 1)`) are
  returned in an exact **but not fully reduced** form: a residual `exp(0)` factor
  remains because the simplifier will not evaluate it. The value is still
  correct; it is simply `lambda·exp(0)` rather than `lambda`. No fabricated
  "cleaned up" answer is ever returned in place of the honest expression
  (Rule 32).

If you need the numeric collapse of such residuals, substitute the transform
variable and evaluate downstream; this module refuses to *assert* a simplified
form it did not actually derive.

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

The Chernoff helper returns the **pre-optimization** bound `e^{−t·alpha}·M_X(t)`;
the tightest bound is the infimum over `t > 0`, which is the caller's to take
(differentiate the bound in `t`, solve the stationarity condition, and
substitute — the ingredients are all `Expr`s this module produces).

## Honesty boundary

- Generating functions, mean, and variance are exact closed forms and **cannot
  fail**.
- `raw_moment` / `cumulant` are exact symbolic derivatives at `t = 0`. They may
  be returned unsimplified (see above) and propagate a `MathError` from the
  differentiation / simplification pipeline; they never fabricate a simplified
  value.
- Tail-bound helpers return the exact bounding expression under the hypotheses
  tabulated above; they encode inequalities, not equalities, and make no
  tightness claim.

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
| `raw_moment` | `auto raw_moment(const Expr& mgf, std::size_t k) -> Result<Expr>` | `E[X^k]` from the MGF. |
| `cumulant` | `auto cumulant(const Expr& mgf, std::size_t k) -> Result<Expr>` | `kappa_k` from the MGF. |
| `markov_bound` | `auto markov_bound(const Expr& mean, const Expr& alpha) -> Expr` | `E[X]/alpha`. |
| `chebyshev_bound` | `auto chebyshev_bound(const Expr& variance, const Expr& k) -> Expr` | `sigma^2/k^2`. |
| `cantelli_bound` | `auto cantelli_bound(const Expr& variance, const Expr& k) -> Expr` | `sigma^2/(sigma^2+k^2)`. |
| `chernoff_bound` | `auto chernoff_bound(const Expr& mgf, const Expr& alpha) -> Expr` | `e^{-t·alpha}·M_X(t)`. |

Constants `mgf_variable == "t"` and `pgf_variable == "z"` name the transform
variables (both `std::string_view`).

## Error model

| Condition | Result |
| :--- | :--- |
| Any catalog constructor (`bernoulli` … `gamma`) | Never fails — returns a `DistInfo` by value. |
| Any tail-bound helper | Never fails — returns an `Expr` by value. |
| `raw_moment` / `cumulant`, differentiation or simplification overflows an `int64` | `MathError::overflow` (propagated from [`simplify`](simplify.md)). |
| `raw_moment` / `cumulant`, undefined algebraic form encountered while simplifying | the corresponding `MathError` (e.g. `undefined_value`, `division_by_zero`) is propagated. |

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

// Catalog entries.
const auto expo = exponential(lambda);   // MGF lambda/(lambda - t)
const auto norm = normal(mu, sigma2);     // MGF exp(mu t + sigma^2 t^2 / 2)

// Exact raw moments from the rational MGF reduce fully:
auto m1 = raw_moment(expo.mgf, 1).value();  // lambda^(-1)  == expo.mean
auto m2 = raw_moment(expo.mgf, 2).value();  // 2 * lambda^(-2)

// Cumulants of the Normal MGF: kappa_1 = mean, kappa_2 = variance.
auto k1 = cumulant(norm.mgf, 1).value();    // mu     == norm.mean
auto k2 = cumulant(norm.mgf, 2).value();    // sigma2 == norm.variance

// Tail bounds (symbolic RHS of the inequality).
const Expr alpha = Expr::symbol("alpha");
const Expr k     = Expr::symbol("k");
auto markov   = markov_bound(expo.mean, alpha);      // (1/lambda)/alpha
auto cheb     = chebyshev_bound(norm.variance, k);   // sigma^2 / k^2
auto chernoff = chernoff_bound(expo.mgf, alpha);     // e^{-t*alpha} * lambda/(lambda - t)
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
