# `nimblecas.tensor` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/tensor/tensor.cppm`

Tensor calculus and differential geometry over the rationals (ROADMAP §7,
extending the vector calculus of §7.19). Everything is **exact symbolic** via
`diff` + `simplify` — no floating point, no tolerance; the only failure channel
is the railway (`Result<T>` / `MathError`).

## Representation

Tensor/metric components are symbolic `Expr` (e.g. the 2-sphere metric is
`diag(a², a²·sin(θ)²)`), so they are held as `Expr` in nested vectors, **not** the
`Rational`-only `nimblecas.matrix`:

- `ExprVector`, `ExprMatrix`, `ExprRank3` (`[i][j][k]`), `ExprRank4`.
- `struct Metric { std::vector<std::string> coords; ExprMatrix g; dim(); }` — a
  rank-2 covariant metric, assumed symmetric.

## Public API (namespace `nimblecas`)

- `make_metric(coords, g) -> Result<Metric>` — validates non-empty squareness.
- `metric_determinant(m) -> Result<Expr>`, `inverse_metric(m) -> Result<ExprMatrix>` —
  symbolic cofactor / Laplace expansion over `Expr` (bounded to dimension ≤ 5).
- `christoffel(m) -> Result<ExprRank3>` — `Γ^k_{ij}` at `[k][i][j]`,
  `Γ^k_{ij} = ½ g^{kl}(∂_i g_{jl} + ∂_j g_{il} − ∂_l g_{ij})`.
- `covariant_derivative_vector(m, v) -> Result<ExprMatrix>` — `∇_i v^k` at `[i][k]`.
- `riemann_tensor(m) -> Result<ExprRank4>` — `R^ρ_{σμν}` at `[ρ][σ][μ][ν]`.
- `ricci_tensor(m) -> Result<ExprMatrix>` — `R_{μν} = R^ρ_{μρν}`.
- `scalar_curvature(m) -> Result<Expr>` — `R = g^{μν} R_{μν}`.
- `einstein_tensor(m) -> Result<ExprMatrix>` — `G_{μν} = R_{μν} − ½ R g_{μν}`.
- `geodesic_coefficients(m) -> Result<ExprRank3>` — the Christoffel data for
  `x''^k + Γ^k_{ij} x'^i x'^j = 0`.
- `metric_gradient(m, f)`, `covariant_divergence(m, field)`,
  `laplace_beltrami(m, f) -> Result<Expr>` = `(1/√g) ∂_i(√g g^{ij} ∂_j f)`.

## Exactness note — the fixed-point simplifier

`simplify` is a single-pass Cohen ASAE: it fuses same-base powers only once their
summed exponent is integral, **never distributes an integer power over a product**
(`(a·b)⁻¹ ↛ a⁻¹·b⁻¹`), and **never distributes a product over a sum**. On their own
those gaps leave curvature results like `(a⁴sin²θ)⁻¹·a⁴·sin³θ·cos θ` or
`((−cot²θ − 1)·(−1)) + (−cot²θ)` un-reduced even though both equal a simple form.
This module therefore drives results to a bounded **fixed point** (`≤ 8` passes)
that first `distribute_all` (expand integer powers over products, and products
over sums — both exact identities, with a soft cap against blow-up) and then
`simplify`. This is what lets the 2-sphere reduce to the hallmark **`R = 2/a²`**.
Powers of a *sum* are deliberately never expanded, so `(a+b)⁻¹` is never wrongly
turned into `a⁻¹+b⁻¹`.

## Honesty boundary

- Exact symbolic throughout; `√g` and raised indices are exact `Expr`
  (`power(g, 1/2)`, `power(x, -1)`). A result that cannot be reduced to a pretty
  form is still returned as the exact (if verbose) `Expr` — never fudged.
- `not_implemented` beyond dimension 5 (the cofactor cap).
- `domain_error` for a singular metric (determinant simplifies to exact `0`, no
  inverse) and for dimension / length mismatches.
- `overflow` for any `int64` boundary in a `Rational` step.

## Results verified in the tests

- **Flat (Euclidean) space** — all Christoffel, Riemann, Ricci entries and the
  scalar curvature are `0`.
- **2-sphere** `diag(a², a²sin²θ)` — `Γ^θ_{φφ} = -sinθcosθ`, `Γ^φ_{θφ} = cotθ`;
  `det g = a⁴sin²θ`, `g^{φφ} = a⁻²sin⁻²θ`; `R_{θθ}=1`, `R_{φφ}=sin²θ`; **scalar
  curvature `R = 2/a²`**; Einstein tensor `≡ 0` (2-D).
- **Polar flat plane** `diag(1, r²)` — Laplace–Beltrami of `r²cosθ` reproduces the
  polar Laplacian `f_rr + (1/r)f_r + (1/r²)f_θθ`.

## See also

- [`forms.md`](forms.md) — differential forms / exterior calculus over the same `Expr` substrate.
- [`vectorcalc.md`](vectorcalc.md) — the flat-space operators this generalizes.
- [`diff.md`](diff.md) · [`simplify.md`](simplify.md) — the exact symbolic engine underneath.
