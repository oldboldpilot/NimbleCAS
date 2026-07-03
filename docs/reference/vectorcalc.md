# `nimblecas.vectorcalc` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/vectorcalc/vectorcalc.cppm`

Vector calculus operators — gradient, divergence, curl, Laplacian, Jacobian,
Hessian, and directional / total derivatives — as thin, **exact** compositions
over the symbolic differentiation engine. A partial derivative is
`differentiate(f, x_i)` with the other symbols held fixed; the vector operators
assemble those partials into gradients, matrices, and the classical field
operators. Every result is passed through automatic
[simplification](simplify.md) so that, for concrete fields, the calculus
identities collapse **exactly** — the mixed partials of a gradient cancel by
Clairaut's theorem, giving `curl(grad f) = 0` and `div(curl F) = 0`.

```cpp
import nimblecas.vectorcalc;
```

Depends on [`core`](core.md), [`symbolic`](symbolic.md), [`diff`](diff.md),
and [`simplify`](simplify.md).

## Type aliases

```cpp
using ExprVector = std::vector<Expr>;                 // a vector field / gradient
using ExprMatrix = std::vector<std::vector<Expr>>;    // a Jacobian / Hessian (row-major)
```

## API

```cpp
[[nodiscard]] auto gradient(const Expr& f, const std::vector<std::string>& vars)
    -> Result<ExprVector>;
[[nodiscard]] auto divergence(const ExprVector& field, const std::vector<std::string>& vars)
    -> Result<Expr>;
[[nodiscard]] auto curl(const ExprVector& field, const std::vector<std::string>& vars)
    -> Result<ExprVector>;
[[nodiscard]] auto laplacian(const Expr& f, const std::vector<std::string>& vars)
    -> Result<Expr>;
[[nodiscard]] auto jacobian(const ExprVector& field, const std::vector<std::string>& vars)
    -> Result<ExprMatrix>;
[[nodiscard]] auto hessian(const Expr& f, const std::vector<std::string>& vars)
    -> Result<ExprMatrix>;
[[nodiscard]] auto directional_derivative(const Expr& f, const std::vector<std::string>& vars,
                                          const ExprVector& direction) -> Result<Expr>;
[[nodiscard]] auto total_derivative(const Expr& f, std::string_view indep,
                                    const std::vector<std::string>& dep_vars,
                                    const ExprVector& dep_derivs) -> Result<Expr>;
```

| Operator | Notation | Returns | Definition |
| :--- | :--- | :--- | :--- |
| `gradient(f, vars)` | ∇f | `ExprVector` | one entry per variable, `(∂f/∂x₁, …, ∂f/∂xₙ)` |
| `divergence(field, vars)` | ∇·F | `Expr` | `Σᵢ ∂Fᵢ/∂xᵢ`; requires `field.size() == vars.size()` |
| `curl(field, vars)` | ∇×F | `ExprVector` | three-dimensional only; requires `field.size() == vars.size() == 3` |
| `laplacian(f, vars)` | ∇²f | `Expr` | `Σᵢ ∂²f/∂xᵢ²` |
| `jacobian(field, vars)` | J | `ExprMatrix` | `Jᵢⱼ = ∂Fᵢ/∂xⱼ` (`field.size()` rows, `vars.size()` columns); row *i* is `grad Fᵢ` |
| `hessian(f, vars)` | H | `ExprMatrix` | `Hᵢⱼ = ∂²f/∂xᵢ∂xⱼ`, symmetric by Clairaut's theorem |
| `directional_derivative(f, vars, direction)` | ∇ᵤf | `Expr` | `∇f·u`; requires `direction.size() == vars.size()`; `u` used **as given** (not normalised) |
| `total_derivative(f, indep, dep_vars, dep_derivs)` | df/dt | `Expr` | multivariable chain rule (below); requires `dep_vars.size() == dep_derivs.size()` |

The `curl` component *k* is `∂F_{k+2}/∂x_{k+1} − ∂F_{k+1}/∂x_{k+2}` with indices
taken mod 3, i.e. the classical `(∂F_z/∂y − ∂F_y/∂z, ∂F_x/∂z − ∂F_z/∂x,
∂F_y/∂x − ∂F_x/∂y)`.

The total derivative applies the multivariable chain rule for
`f(t, x₁(t), …, xₙ(t))`:

```
df/dt = ∂f/∂t + Σᵢ (∂f/∂xᵢ)(dxᵢ/dt)
```

`indep` is the independent variable `t`, `dep_vars` are the dependent symbols
`xᵢ(t)`, and `dep_derivs[i]` is the supplied derivative `dxᵢ/dt`. The `∂f/∂t`
term captures any **explicit** dependence on `t`.

## Error model

Dimension mismatches are reported as `MathError::domain_error` (Rule 32 — no
exceptions):

| Operator | `domain_error` when |
| :--- | :--- |
| `curl` | `field.size() != 3` or `vars.size() != 3` (curl is 3-dimensional) |
| `divergence` | `field.size() != vars.size()` |
| `directional_derivative` | `direction.size() != vars.size()` |
| `total_derivative` | `dep_vars.size() != dep_derivs.size()` |

`gradient`, `laplacian`, `jacobian`, and `hessian` never cross-index the field
against the variable list, so an **empty** `vars` simply yields empty output
(an empty vector, a folded `0`, or an empty matrix) rather than an error.

Any error raised by the underlying [`differentiate`](diff.md) or
[`simplify`](simplify.md) — for instance an integer overflow surfaced during
simplification — propagates unchanged through the `Result`.

## Exact identities

Because every partial is differentiated symbolically and the assembled result
is simplified, the classical vector-calculus identities hold **exactly** for
concrete fields (they are verified in the tests):

- **`curl(grad f) = (0, 0, 0)`** — the mixed second partials cancel pairwise by
  Clairaut's theorem (`∂²f/∂x∂y = ∂²f/∂y∂x`).
- **`div(curl F) = 0`** — the six second-partial terms cancel for the same
  reason.
- **Hessian symmetry** — `Hᵢⱼ = Hⱼᵢ`, so the computed matrix is symmetric.

## Examples

```cpp
import nimblecas.vectorcalc;
using namespace nimblecas;

const Expr x = Expr::symbol("x");
const Expr y = Expr::symbol("y");
const Expr z = Expr::symbol("z");

// grad(x^2 y + z) = (2xy, x^2, 1)
const Expr f = Expr::sum({Expr::product({Expr::power(x, Expr::integer(2)), y}), z});
auto g = gradient(f, {"x", "y", "z"}).value();          // {2*x*y, x^2, 1}

// div(x^2, y^2, z^2) = 2x + 2y + 2z
std::vector<Expr> field = {Expr::power(x, Expr::integer(2)),
                           Expr::power(y, Expr::integer(2)),
                           Expr::power(z, Expr::integer(2))};
auto d = divergence(field, {"x", "y", "z"}).value();    // 2*x + 2*y + 2*z

// laplacian(x^2 + y^2 + z^2) = 6
auto l = laplacian(Expr::sum({Expr::power(x, Expr::integer(2)),
                              Expr::power(y, Expr::integer(2)),
                              Expr::power(z, Expr::integer(2))}),
                   {"x", "y", "z"}).value();             // 6

// curl(-y, x, 0) = (0, 0, 2)
auto c = curl({Expr::product({Expr::integer(-1), y}), x, Expr::integer(0)},
              {"x", "y", "z"}).value();                  // {0, 0, 2}

// Hessian of x^2 y = [[2y, 2x], [2x, 0]]  (symmetric)
auto h = hessian(Expr::product({Expr::power(x, Expr::integer(2)), y}),
                 {"x", "y"}).value();                    // {{2*y, 2*x}, {2*x, 0}}

// directional derivative of x^2 + y^2 along (1, 0) is 2x
auto dd = directional_derivative(
    Expr::sum({Expr::power(x, Expr::integer(2)), Expr::power(y, Expr::integer(2))}),
    {"x", "y"}, {Expr::integer(1), Expr::integer(0)}).value();   // 2*x

// total derivative of x*y with x = x(t), y = y(t) is y*x' + x*y'
const Expr xp = Expr::symbol("u");   // x'(t)
const Expr yp = Expr::symbol("v");   // y'(t)
auto td = total_derivative(Expr::product({x, y}), "t", {"x", "y"}, {xp, yp}).value();
                                                          // y*x' + x*y'
```

## Lineage

ROADMAP §7.19 (Symbolic Differentiation and Integration) — these are the
multivariable-differentiation / vector-calculus operators. They build on the
recursive symbolic [differentiation engine](diff.md) and its automatic
[simplification](simplify.md), which is what makes the calculus identities
collapse to exact zeros rather than structurally-distinct-but-equal trees.

## See also

- [`nimblecas.diff`](diff.md) — the per-symbol partial derivative every operator composes.
- [`nimblecas.simplify`](simplify.md) — post-processes every assembled result.
- [`nimblecas.symbolic`](symbolic.md) — the `Expr` model.
- [Documentation hub](../Index.md)
