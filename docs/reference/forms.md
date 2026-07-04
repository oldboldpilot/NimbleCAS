# `nimblecas.forms` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/forms/forms.cppm`

Exterior calculus — **differential forms** over symbolic `Expr` coefficients on a
fixed ordered list of coordinates (ROADMAP §7). Everything is **exact symbolic**
via `diff` + `simplify`; the only failure channel is the railway
(`Result<T>` / `MathError`). No floating point, no tolerances.

## Representation

A `DifferentialForm` carries the ordered coordinate names, its degree `p`, and a
`std::map` from **strictly-increasing** index tuples `(i₀<…<i_{p-1})` to an `Expr`
coefficient — only canonical basis components are stored. Antisymmetry is derived:
any query canonicalizes its index tuple (each inversion flips the sign, a repeated
index annihilates the term). Coefficients pass through `simplify` and zero
components are dropped, so `is_zero()` is exactly "empty map".

## Public API (namespace `nimblecas`)

Constructors — all `Result<DifferentialForm>`:
- `DifferentialForm::scalar(coords, value)` — a 0-form.
- `DifferentialForm::zero(coords, degree)`.
- `DifferentialForm::basis(coords, indices, coeff)` — `coeff · dx_{i₀}∧…∧dx_{iₚ}`.
- `DifferentialForm::from_components(coords, degree, terms)`.

Accessors: `dimension()`, `degree()`, `coordinates()`, `components()`,
`is_zero()`, `component(indices)` (sign-canonicalized), `to_string()`.

Free functions:
- `operator==` — structural form equality.
- `wedge(a, b) -> Result<DifferentialForm>` — graded-antisymmetric exterior
  product, degree `p+q` (the zero form when `p+q > n`); sign from the permutation
  parity of the merged index tuple.
- `exterior_derivative(w) -> Result<DifferentialForm>` — `d`, raising a p-form to
  `p+1` via `differentiate`.
- `hodge_star_euclidean(w) -> Result<DifferentialForm>` and
  `hodge_star(w, metric) -> Result<DifferentialForm>` (see scope below).
- `interior_product(w, field) -> Result<DifferentialForm>` — contraction `ι_V`,
  `p → p-1`.
- `is_closed(w) -> Result<bool>` (sound and total) and
  `is_exact(w) -> Result<bool>` (sound-but-partial, see below).

## Hodge star scope

`hodge_star(w, metric)` supports the **Euclidean / orthonormal (identity) metric
only**. It accepts the symbolic `Expr` metric, validates that every entry
simplifies exactly to the identity, then delegates to `hodge_star_euclidean`. Any
other metric returns `MathError::not_implemented` rather than a wrong dual — a
general metric needs `√det g`, which is not soundly representable in the current
exact simplifier domain. This is stated in the module header and enforced, not
faked.

## Honesty boundary

- Everything exact symbolic via `diff`+`simplify`; the only failure channel is the
  railway. No floats, no tolerances.
- `hodge_star` never fabricates a non-Euclidean result — `not_implemented`.
- `is_exact` is **sound-but-partial**: `Ok(true)` only for the identically-zero
  form (trivially exact); `not_implemented` for any nonzero form (general
  exactness is de Rham cohomology, not decided here). `is_closed` (the necessary
  condition, `d w = 0`) is sound and total.
- Bounds: mismatched coordinate lists, index `≥ n`, or a wrong-length tuple →
  `domain_error`; degree `> n` is the zero form (not an error); a `Rational`
  overflow propagates from the `diff`/`simplify` railway.

## Identities verified in the tests

- **`d² = 0`** — `exterior_derivative(exterior_derivative(w)) = 0` for a generic 1-form.
- **Graded anticommutativity** — `a ∧ b = (-1)^{pq} b ∧ a`; `dx ∧ dx = 0`, `dx ∧ dy = -dy ∧ dx`.
- **`d` of a 0-form** `f` is the gradient 1-form `Σ (∂f/∂x_i) dx^i`.
- **Euclidean Hodge** — in `ℝ³`, `⋆dx = dy∧dz`, `⋆dy = -dx∧dz`, `⋆(dx∧dy) = dz`; double star `⋆⋆ = (-1)^{p(n-p)}`.
- **Interior product** — `ι_V(dx∧dy) = x·dy - y·dx` for `V = (x,y,z)`.

## See also

- [`tensor.md`](tensor.md) — metrics, curvature, and the covariant derivative that pair with forms.
- [`diff.md`](diff.md) · [`simplify.md`](simplify.md) — the exact symbolic engine underneath.
