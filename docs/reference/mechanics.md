# `nimblecas.mechanics` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/mechanics/mechanics.cppm`

Exact **symbolic Hamiltonian mechanics** over the expression trees of
[`nimblecas.symbolic`](symbolic.md). Given a Lagrangian or Hamiltonian written as
an `Expr`, this layer produces the **governing equations** of a mechanical
system: the Legendre transform `L(q,q̇,t) → H(q,p,t)`, Hamilton's canonical
first-order system, Poisson brackets, conserved-quantity tests, phase-space level
sets and vector fields, and the one-degree-of-freedom action integrand. Every
operator is a partial-derivative composition on the differentiation engine
([`nimblecas.diff`](diff.md)) followed by Cohen automatic simplification
([`nimblecas.simplify`](simplify.md)); a canonical coordinate system carries the
symbol names of the generalized coordinates `q_i`, their velocities `q̇_i`, and
their conjugate momenta `p_i`, and `q` and `p` are treated as **independent**
phase-space coordinates exactly as Hamilton's formalism requires. **Nothing here
is numeric — no floating point ever enters.**

**Honesty boundary.** This module produces governing equations; it **does not
solve ODEs** (feed the resulting `HamiltonSystem` to [`nimblecas.ode`](ode.md) for
a series solution). Two exact-vs-symbolic boundaries are reported honestly rather
than fabricated:

- The Legendre momentum inversion `p_i = ∂L/∂q̇_i → q̇_i(q,p,t)` is solved by
  Cramer's rule and so is exact **only when `L` is affine/quadratic in the
  velocities** — the Hessian (mass matrix) `M_ij = ∂²L/∂q̇_i∂q̇_j` must be free of
  the velocities and non-singular. A velocity-dependent `M` (cubic-or-higher `L`)
  or a singular `M` (`det M` simplifying to `0`) is `MathError::not_implemented`.
- The 1-DOF action `J = (1/2π)∮ p dq` is returned as its **integrand** `p(q,E)`,
  obtained by solving `H(q,p)=E` for `p` (elementary only when `H` is at most
  quadratic in `p`). The closed-form loop integral is generally **non-elementary**
  (`p(q,E)` is irrational, e.g. `sqrt(2E − ω²q²)`, outside the rational domain of
  [`nimblecas.integrate`](integrate.md)), so the closed form is left **absent**
  and the angle variable `θ = ∂W/∂J` is reported as `not_implemented` rather than
  invented.

Dimension mismatches are `MathError::domain_error`; any differentiation or
simplification failure propagates on the `Result` railway (Rule 32).

```cpp
import nimblecas.mechanics;
```

Depends on [`core`](core.md), [`symbolic`](symbolic.md), [`diff`](diff.md), and
[`simplify`](simplify.md).

## `Coordinates` — a canonical coordinate system

```cpp
struct Coordinates {
    std::vector<std::string> q;      // generalized coordinates q_i
    std::vector<std::string> qdot;   // velocities q̇_i (same size as q)
    std::vector<std::string> p;      // conjugate momenta p_i (same size as q)
    std::string time{"t"};           // evolution-parameter symbol
};
```

The three symbol lists name the `i`-th generalized coordinate `q_i`, its velocity
`q̇_i`, and its conjugate momentum `p_i`. They **must all have the same, non-zero
length** — the number of degrees of freedom — or every operator that touches the
system returns `MathError::domain_error`. `time` names the evolution parameter and
defaults to `"t"`.

## `Hamiltonian` — the result of a Legendre transform

```cpp
struct Hamiltonian {
    Expr H;                        // H(q,p,t) = Σ p_i q̇_i − L, in phase-space variables
    std::vector<Expr> momenta;     // p_i = ∂L/∂q̇_i, in (q,q̇,t) — the momentum definition
    std::vector<Expr> velocities;  // q̇_i(q,p,t), the inverted momentum map
};
```

| Field | Meaning |
| :--- | :--- |
| `H` | The Hamiltonian `H(q,p,t) = Σ p_i q̇_i − L`, expressed in the phase-space variables `(q,p,t)`. |
| `momenta` | The momentum definitions `p_i = ∂L/∂q̇_i`, still in the `(q,q̇,t)` variables. |
| `velocities` | The inverted momentum map `q̇_i(q,p,t)` recovered from `M q̇ = p − b` by Cramer's rule. |

## `HamiltonSystem` — the canonical first-order system

```cpp
struct HamiltonSystem {
    std::vector<Expr> qdot;  // q̇_i = ∂H/∂p_i
    std::vector<Expr> pdot;  // ṗ_i = −∂H/∂q_i
};
```

Hamilton's canonical equations as a `2n` first-order system — equivalently, the
phase-space velocity field `(q̇, ṗ)` used for sampling and plotting.

## `ActionIntegral` — the 1-DOF action integrand

```cpp
struct ActionIntegral {
    Expr integrand;                   // p(q,E): the ∮ p dq integrand at energy E
    std::string coordinate;           // the q variable of integration
    std::optional<Expr> closed_form;  // the elementary ∮ p dq, if one was found (else absent)
};
```

| Field | Meaning |
| :--- | :--- |
| `integrand` | `p(q,E)`, the `∮ p dq` integrand obtained by solving `H(q,p)=E` for `p`. |
| `coordinate` | The `q` symbol integrated over (`coords.q[0]`). |
| `closed_form` | The elementary loop integral **if one was found** — deliberately `std::nullopt` here, since the integrand is generally irrational (see the honesty boundary). |

## Free functions

All are exported at namespace `nimblecas` (not a nested namespace).

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `legendre_transform` | `[[nodiscard]] auto legendre_transform(const Expr& lagrangian, const Coordinates& coords) -> Result<Hamiltonian>` | Legendre transform `L(q,q̇,t) → H(q,p,t)`. Defines `p_i = ∂L/∂q̇_i`, inverts the (linear) momentum map to `q̇_i(q,p,t)`, and forms `H = Σ p_i q̇_i − L` in phase-space variables. See below. |
| `hamilton_equations` | `[[nodiscard]] auto hamilton_equations(const Expr& H, const Coordinates& coords) -> Result<HamiltonSystem>` | Hamilton's canonical equations `q̇_i = ∂H/∂p_i`, `ṗ_i = −∂H/∂q_i` as a `2n` first-order system. |
| `poisson_bracket` | `[[nodiscard]] auto poisson_bracket(const Expr& f, const Expr& g, const Coordinates& coords) -> Result<Expr>` | The Poisson bracket `{f,g} = Σ_i (∂f/∂q_i ∂g/∂p_i − ∂f/∂p_i ∂g/∂q_i)`, simplified. |
| `total_time_derivative` | `[[nodiscard]] auto total_time_derivative(const Expr& f, const Expr& H, const Coordinates& coords) -> Result<Expr>` | The total derivative along the Hamiltonian flow, `df/dt = {f,H} + ∂f/∂t`. |
| `is_constant_of_motion` | `[[nodiscard]] auto is_constant_of_motion(const Expr& f, const Expr& H, const Coordinates& coords) -> Result<bool>` | Whether `f` is a constant of motion, i.e. `df/dt = {f,H} + ∂f/∂t` simplifies to `0`. |
| `cyclic_coordinates` | `[[nodiscard]] auto cyclic_coordinates(const Expr& H, const Coordinates& coords) -> Result<std::vector<std::size_t>>` | Indices `i` of cyclic (ignorable) coordinates: `H` is free of `q_i`, so `p_i` is conserved. |
| `phase_curve` | `[[nodiscard]] auto phase_curve(const Expr& H, const Expr& energy) -> Result<Expr>` | The implicit phase curve of the level set `H(q,p)=E`: returns the relation `H − E` (which is `0` on the curve). |
| `phase_portrait_field` | `[[nodiscard]] auto phase_portrait_field(const Expr& H, const Coordinates& coords) -> Result<HamiltonSystem>` | The phase-space vector field `(q̇,ṗ)` for sampling/plotting — Hamilton's equations by another name. |
| `action_integral` | `[[nodiscard]] auto action_integral(const Expr& H, const Coordinates& coords, const Expr& energy) -> Result<ActionIntegral>` | 1-DOF action integrand: solve `H(q,p)=E` for `p(q,E)`, the `∮ p dq` integrand. See below. |
| `angle_variable` | `[[nodiscard]] auto angle_variable(const Expr& H, const Coordinates& coords, const Expr& energy) -> Result<Expr>` | The angle variable `θ = ∂W/∂J`, `W = ∫ p dq`. See below. |

### `legendre_transform`

Computes the conjugate momenta `p_i = ∂L/∂q̇_i`, then the mass (Hessian) matrix
`M_ij = ∂p_i/∂q̇_j = ∂²L/∂q̇_i∂q̇_j`. **Every entry of `M` must be free of the
velocities** — i.e. `L` is affine/quadratic in `q̇` — otherwise the momentum map is
nonlinear and the inversion is `MathError::not_implemented`. It splits the affine
offsets `b_i = p_i|_{q̇=0}` so that `M q̇ = (p − b)` (with `p_i` now the momentum
*symbols*), inverts by Cramer's rule, and assembles the exact quadratic-form value
`H = ½ Σ_i (p_i − b_i) q̇_i − c` with `c = L|_{q̇=0}` the velocity-free remainder.

Fails with `MathError::domain_error` on an inconsistent `Coordinates`, and
`MathError::not_implemented` when `L` is not quadratic in the velocities or the
mass matrix is singular (`det M` simplifies to `0`). Any `diff` / `simplify`
failure propagates.

### `action_integral`

Requires **exactly one degree of freedom** (`coords.q.size() == 1`), else
`MathError::domain_error`. It solves `H(q,p) = E` for `p`, treating `H` as at most
a quadratic polynomial in the momentum symbol `p` (`H = a p² + b p + c` with
`a,b,c` free of `p`): the quadratic root is
`p = (−b + sqrt(b² − 4a(c − E))) / (2a)` (the `p ≥ 0` branch), or the linear root
`p = (E − c)/b` when `a == 0`. `H` higher than quadratic in `p`, or a momentum that
cannot be isolated (`a == 0` and `b == 0`), is `MathError::not_implemented`. The
`closed_form` is always `std::nullopt` here — the loop integral of the irrational
integrand is outside the rational integration domain, and this module will not
fabricate one.

### `angle_variable`

`θ = ∂W/∂J` with `W = ∫ p dq`. It is elementary only when the action's closed form
exists; since `action_integral` leaves `closed_form` absent for the non-elementary
(irrational) integrand, the angle variable is honestly `MathError::not_implemented`
here (a `domain_error` still propagates first for a non-1-DOF system).

## Error model

| Condition | Error |
| :--- | :--- |
| `Coordinates` lists differ in length, or are empty (any operator) | `MathError::domain_error` |
| `action_integral` / `angle_variable` on a system that is not exactly 1-DOF | `MathError::domain_error` |
| `legendre_transform`: `L` not affine/quadratic in `q̇` (mass matrix depends on velocities) | `MathError::not_implemented` |
| `legendre_transform`: singular mass matrix (`det M` simplifies to `0`) | `MathError::not_implemented` |
| `action_integral`: `H` higher than quadratic in `p`, or momentum not isolable (`a == 0` and `b == 0`) | `MathError::not_implemented` |
| `angle_variable`: no closed-form action (always, given the irrational integrand) | `MathError::not_implemented` |
| Any `differentiate` / `simplify` step fails (e.g. an overflow) | that operation's error, propagated verbatim |

`phase_curve` takes no `Coordinates` and only simplifies `H − E`, so it fails only
if that simplification does. `cyclic_coordinates` and `is_constant_of_motion`
return their result on the railway; nothing throws.

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.mechanics;
using namespace nimblecas;

auto sym   = [](std::string n) { return Expr::symbol(std::move(n)); };
auto I     = [](std::int64_t n) { return Expr::integer(n); };
auto S     = [](const Expr& e) { return simplify(e).value(); };  // canonical form
auto half  = [] { return Expr::rational(1, 2).value(); };
auto neg   = [&](const Expr& a) { return Expr::product({I(-1), a}); };
auto sqr   = [&](const Expr& a) { return Expr::power(a, I(2)); };
auto recip = [&](const Expr& a) { return Expr::power(a, I(-1)); };

const Expr q = sym("q"), qd = sym("qd"), p = sym("p");
const Expr w = sym("w"), m = sym("m"), E = sym("E");

// One-DOF canonical coordinate system (q, q̇ = qd, p).
const Coordinates one{.q = {"q"}, .qdot = {"qd"}, .p = {"p"}, .time = "t"};

// --- Legendre transform: harmonic oscillator ------------------------------
// L = 1/2 q̇² − 1/2 ω² q²   ->   p = q̇,  H = 1/2 p² + 1/2 ω² q².
const Expr L_sho = Expr::sum({Expr::product({half(), sqr(qd)}),
                              neg(Expr::product({half(), sqr(w), sqr(q)}))});
auto ham = legendre_transform(L_sho, one).value();
ham.momenta[0];        // qd          (p = ∂L/∂q̇ = q̇)
ham.velocities[0];     // p           (inverted map q̇ = p)
ham.H == S(Expr::sum({Expr::product({half(), sqr(p)}),
                      Expr::product({half(), sqr(w), sqr(q)})}));   // true

// --- Legendre transform: free particle with mass --------------------------
// L = 1/2 m q̇²  ->  p = m q̇,  H = p²/(2m).
const Expr L_free = Expr::product({half(), m, sqr(qd)});
auto hfree = legendre_transform(L_free, one).value();
hfree.momenta[0]    == S(Expr::product({m, qd}));            // p = m q̇
hfree.velocities[0] == S(Expr::product({p, recip(m)}));      // q̇ = p/m
hfree.H             == S(Expr::product({half(), sqr(p), recip(m)}));  // H = p²/(2m)

// --- Honesty boundary: a Lagrangian cubic in the velocity -----------------
legendre_transform(Expr::power(qd, I(3)), one).error();     // not_implemented

// --- Hamilton's canonical equations ---------------------------------------
// H = 1/2 p² + 1/2 ω² q²  ->  q̇ = p,  ṗ = −ω² q.
const Expr H = Expr::sum({Expr::product({half(), sqr(p)}),
                          Expr::product({half(), sqr(w), sqr(q)})});
auto sys = hamilton_equations(H, one).value();
sys.qdot[0] == p;                                   // q̇ = ∂H/∂p = p
sys.pdot[0] == S(neg(Expr::product({sqr(w), q})));  // ṗ = −∂H/∂q = −ω² q

// --- Poisson brackets and energy conservation -----------------------------
poisson_bracket(q, p, one).value() == I(1);   // canonical {q,p} = 1
poisson_bracket(q, q, one).value() == I(0);   // {q,q} = 0
poisson_bracket(p, p, one).value() == I(0);   // {p,p} = 0
poisson_bracket(H, H, one).value() == I(0);   // {H,H} = 0
is_constant_of_motion(H, H, one).value();     // true — energy is conserved

// --- Cyclic coordinate => conserved momentum (free rotor) -----------------
const Coordinates rot{.q = {"theta"}, .qdot = {"thetadot"},
                      .p = {"ptheta"}, .time = "t"};
const Expr H_rot = Expr::product({half(), sqr(sym("ptheta"))});
auto cyc = cyclic_coordinates(H_rot, rot).value();
cyc.size() == 1 && cyc[0] == 0;                              // theta is cyclic
is_constant_of_motion(sym("ptheta"), H_rot, rot).value();   // p_θ conserved
is_constant_of_motion(sym("theta"),  H_rot, rot).value();   // false ({θ,H} = p_θ)

// --- Phase curve and phase-portrait field ---------------------------------
phase_curve(H, E).value() == S(Expr::sum({H, neg(E)}));      // level set H − E
auto field = phase_portrait_field(H, one).value();
field.qdot[0] == p;                                          // (q̇, ṗ) = (p, −ω² q)
field.pdot[0] == S(neg(Expr::product({sqr(w), q})));

// --- 1-DOF action integrand -----------------------------------------------
// Solving H = E gives p = sqrt(2E − ω² q²), the ∮ p dq integrand. That integrand
// is irrational — outside the rational domain of nimblecas.integrate — so the loop
// integral is not attempted here and closed_form stays absent.
auto act = action_integral(H, one, E).value();
act.integrand == S(Expr::apply("sqrt",
    {Expr::sum({Expr::product({I(2), E}), neg(Expr::product({sqr(w), sqr(q)}))})}));
act.coordinate == "q";
act.closed_form.has_value();                     // false (honest not_implemented)
angle_variable(H, one, E).error();               // not_implemented (no closed form)

// --- Degenerate arguments are domain errors -------------------------------
const Coordinates bad{.q = {"q"}, .qdot = {"qd", "qd2"}, .p = {"p"}};
hamilton_equations(sym("q"), bad).error();       // domain_error (size mismatch)
const Coordinates two{.q = {"x", "y"}, .qdot = {"xd", "yd"}, .p = {"px", "py"}};
action_integral(sym("px"), two, E).error();      // domain_error (action-angle is 1-DOF)
```

## See also

- [`nimblecas.diff`](diff.md) — the differentiation engine every partial
  derivative (momenta, Hessian, canonical equations, Poisson brackets) is built
  on.
- [`nimblecas.simplify`](simplify.md) — the Cohen ASAE canonicalizer that reduces
  each result to its clean form.
- [`nimblecas.symbolic`](symbolic.md) — the immutable `Expr` trees, `free_of`, and
  `substitute` this layer manipulates.
- [`nimblecas.ode`](ode.md) — solve the `HamiltonSystem` produced here: this module
  yields the governing equations, `ode` integrates them as exact power series.
- [`nimblecas.dynamics`](dynamics.md) — equilibria and stability analysis of the
  resulting first-order vector field.
- [`nimblecas.integrate`](integrate.md) — the rational-function integrator whose
  domain the irrational action integrand `p(q,E)` lies outside of.
- [Documentation hub](../Index.md)
