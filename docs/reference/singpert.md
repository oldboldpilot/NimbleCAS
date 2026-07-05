# `nimblecas.singpert` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/singpert/singpert.cppm`

Singular-perturbation solver (ROADMAP §7.4): the **leading-order matched
asymptotic expansion** of the constant-coefficient linear two-point
boundary-layer problem

```
ε y''(x) + a y'(x) + b y(x) = 0,   0 < x < 1,   y(0) = α,  y(1) = β,   a > 0.
```

With `a > 0` the boundary layer sits at the **left** endpoint `x = 0`. The
coefficients `a, b, α, β` are exact `Rational`s; the three classical pieces —
outer solution, inner (boundary-layer) solution, and uniform composite — are
returned as symbolic `Expr` in the physical variable `x = symbol("x")` and the
small parameter `ε = symbol("epsilon")`. Exponentials are `exp(·)` function
nodes; division by `ε` is the power `ε^(−1)` (the symbolic engine has no divide
node).

```cpp
import nimblecas.singpert;
```

Depends on [`core`](core.md) (the `Result`/`MathError` railway), the `Rational`
field of [`ratpoly`](ratpoly.md), and the expression trees of
[`symbolic`](symbolic.md).

## Honesty boundary

What this module returns is a **leading-order asymptotic approximation valid for
small `ε`** — it is **not** an exact closed-form solution of the ODE. The exact
solution is a combination of two real exponentials with rates
`(−a ± √(a² − 4εb)) / (2ε)`, generally irrational and not representable in this
exact-rational engine; the matched expansion is the small-`ε` limit of that,
correct to leading order. Only this specific class is handled:

- **constant** coefficients `a, b`,
- **`a > 0`** (a single boundary layer at the **left** endpoint `x = 0`),
- **leading order** only (no higher-order corrections).

Anything outside this class is out of scope and reported honestly rather than
approximated wrongly (Rule 32): `a ≤ 0` — which has no decaying inner mode
`exp(−aξ)` under this matching — returns `MathError::domain_error`. Any overflow
in the exact `Rational`/`Expr` coefficient construction is propagated verbatim.
No plausible-but-wrong expansion is ever returned.

## Derivation

**Outer solution.** Setting `ε = 0` degenerates the equation to `a y' + b y = 0`,
so `y_out = C·exp(−(b/a)x)`. The outer solution is valid away from the layer and
carries the far endpoint `y_out(1) = β`, fixing `C` and giving

```
y_out(x) = β · exp((b/a)(1 − x)).
```

**Inner solution.** Rescaling with the stretched coordinate `ξ = x/ε` turns the
equation, to leading order, into `Y'' + a Y' = 0`, whose decaying solution
(`a > 0`) is `Y(ξ) = A + B·exp(−aξ)`. Matching the outer limit of the inner to
the inner limit of the outer, `Y(∞) = y_out(0)`, fixes `A = β·exp(b/a)`; the wall
condition `Y(0) = α` fixes `B = α − β·exp(b/a)`. Written back in `x` via
`ξ = x/ε`:

```
y_in(x) = β·exp(b/a) + (α − β·exp(b/a))·exp(−a x/ε).
```

The `inner` field is returned in the physical variable (with `ξ = x/ε`
substituted) precisely so that the composite below is a genuine expression
identity `y_unif = y_out + y_in − y_common`.

**Common part (overlap / matching limit).** The shared limit of the two regions,

```
y_common = lim_{ξ→∞} y_in = lim_{x→0} y_out = A = β·exp(b/a).
```

**Uniform composite.** Additive matching subtracts the double-counted common part:
`y_unif = y_out + y_in − y_common`. The constant level `A` inside `y_in` cancels
`y_common` exactly, leaving the classical leading-order composite

```
y_unif(x) = β·exp((b/a)(1 − x)) + (α − β·exp(b/a))·exp(−a x/ε).
```

At `x = 0` the layer factor is `exp(0) = 1` and `y_unif(0) = α` **exactly**; at
`x = 1` it equals `β` up to the exponentially small `exp(−a/ε)`, as leading-order
matching guarantees (the residual is beyond all orders in `ε`).

## API

```cpp
struct MatchedExpansion {
    Expr outer;      // y_out(x)  = β·exp((b/a)(1 − x))
    Expr inner;      // y_in(x)   = β·exp(b/a) + (α − β·exp(b/a))·exp(−a x/ε)
    Expr composite;  // y_unif(x) = β·exp((b/a)(1 − x)) + (α − β·exp(b/a))·exp(−a x/ε)
};
```

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `matched_asymptotic` | `[[nodiscard]] auto matched_asymptotic(Rational a, Rational b, Rational alpha, Rational beta) -> Result<MatchedExpansion>` | Leading-order matched asymptotic expansion of `ε y'' + a y' + b y = 0`, `y(0)=α`, `y(1)=β`, for `a > 0` (left boundary layer). Returns the outer, inner, and uniform-composite `Expr` in `x` and `ε`. |

## Error model

| Condition | Error |
| :--- | :--- |
| `a ≤ 0` (no decaying left-boundary-layer mode; out of scope) | `MathError::domain_error` |
| Overflow forming `b/a`, `−a`, or `−β`, or building a rational `Expr` constant (e.g. an `INT64_MIN` operand) | propagated verbatim (`MathError::overflow`) |

Non-constant coefficients, right/interior layers, and higher-order corrections
are not modelled at all — there is no code path that silently approximates them.

## Worked example

Matching the test `canonical_case_a1_b1_alpha0_beta1`: the problem
`ε y'' + y' + y = 0`, `y(0) = 0`, `y(1) = 1` has `a = b = 1 > 0`, so `b/a = 1`,
`α = 0`, `β = 1`.

```cpp
import nimblecas.singpert;
import nimblecas.symbolic;
import nimblecas.ratpoly;
using namespace nimblecas;

auto ri = [](std::int64_t v) { return Rational::from_int(v); };

auto exp = matched_asymptotic(ri(1), ri(1), ri(0), ri(1)).value();

// exp.outer     == 1·exp(1·(1 − x))
// exp.inner     == 1·exp(1) + (0 + (−1)·exp(1))·exp(−1·x·ε^(−1))
// exp.composite == 1·exp(1·(1 − x)) + (0 + (−1)·exp(1))·exp(−1·x·ε^(−1))
//               =  β·exp((b/a)(1 − x)) + (α − β·exp(b/a))·exp(−a x/ε)

// Rebuilding the composite from literal Expr pieces reproduces it exactly:
Expr x       = Expr::symbol("x");
Expr eps     = Expr::symbol("epsilon");
Expr exp_ba  = Expr::apply("exp", {*Expr::rational(1, 1)});
Expr amp     = Expr::sum({*Expr::rational(0, 1),
                          Expr::product({*Expr::rational(-1, 1), exp_ba})});
Expr layer   = Expr::apply("exp", {Expr::product({*Expr::rational(-1, 1), x,
                                                  Expr::power(eps, Expr::integer(-1))})});
Expr outer   = Expr::product(
    {*Expr::rational(1, 1),
     Expr::apply("exp", {Expr::product({*Expr::rational(1, 1),
        Expr::sum({Expr::integer(1), Expr::product({Expr::integer(-1), x})})})})});
Expr composite = Expr::sum({outer, Expr::product({amp, layer})});

composite.is_equivalent_to(exp.composite);  // true

// a ≤ 0 is out of scope, not silently approximated:
matched_asymptotic(ri(0), ri(1), ri(0), ri(1)).error();  // MathError::domain_error
```

The wall boundary condition holds exactly over `Q`: in `y_in(0) = A + B` the
`exp(b/a)` coefficients `β` and `−β` cancel, leaving the constant `α` (verified at
the `Rational` level in `boundary_condition_matching_is_exact_in_Q`, since the
structural `exp(0)` layer factor is not auto-simplified).

## See also

- [`nimblecas.perturbation`](perturbation.md) — the sibling perturbation-family
  module: exact truncated-power-series ADM/HPM/HAM solvers for `u' = f(u)`.
- [`nimblecas.symbolic`](symbolic.md) — the expression-tree engine supplying
  `Expr::symbol`, `Expr::rational`, `Expr::sum/product/power/apply`, and the
  structural `is_equivalent_to` these expansions are built and compared with.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field the coefficients
  `a, b, α, β` live in.
- [Documentation hub](../Index.md)
