# `nimblecas.calcvar` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/calcvar/calcvar.cppm`

Calculus of variations — Euler–Lagrange equations (functional and mechanics
forms), the Beltrami first integral, holonomic / non-holonomic (Pfaffian)
constraint handling via Lagrange multipliers, and Frobenius classification of
Pfaffian forms (ROADMAP §7.19). Everything here is a thin, **exact** symbolic
composition over the differentiation engine ([`diff`](diff.md)) and automatic
[simplification](simplify.md): a functional `J[y] = ∫ F(x, y, y') dx` has
stationary paths characterised by the Euler–Lagrange equation `∂F/∂y −
d/dx(∂F/∂y') = 0`, where `∂F/∂y` and `∂F/∂y'` are ordinary partials (`y`, `y'`
treated as independent symbols) and `d/dx` is the **total** derivative that
chains through the jet `x → (explicit)`, `y → y'`, `y' → y''`.

**Honesty boundary.** This module produces the **governing equations exactly**;
it does **not** solve the resulting ODEs/PDEs — that is the job of
`nimblecas.ode` / `nimblecas.pde`. Integrability classification of a Pfaffian
form is symbolic and best-effort: exactness (closedness) and the Frobenius
condition are checked by **structural zero-testing after simplification**, a
sound-but-incomplete decision procedure. When a Frobenius obstruction neither
reduces to zero nor is a provably nonzero constant, the form is reported as
`not_determinable` rather than guessed. **No floating-point heuristics are used.**

```cpp
import nimblecas.calcvar;
```

Depends on [`core`](core.md), [`symbolic`](symbolic.md), [`diff`](diff.md),
and [`simplify`](simplify.md). Everything is exported in namespace `nimblecas`.

## Type alias

```cpp
using ExprVector = std::vector<Expr>;    // a system of expressions / equations
```

## Data structures

A generalized coordinate is carried as a **jet** of three independent `Expr`
symbols — the value `q`, its velocity `q' = dq/dt`, and its acceleration
`q'' = d²q/dt²`. The three are distinct symbols in the `Expr` layer; the
total-derivative machinery links them by the chain rule `q → q' → q''`.

```cpp
struct Coordinate {
    std::string value;         // q  (or y)
    std::string velocity;      // q' (or y')
    std::string acceleration;  // q'' (or y''), introduced by the total d/dt
};

struct PfaffianConstraint {
    ExprVector coeffs;   // Σ_i coeffs[i] dq_i
    Expr time_coeff;     // + time_coeff dt   (use Expr::integer(0) when absent)
};

struct ConstrainedSystem {
    ExprVector equations;     // one Euler–Lagrange equation (== 0) per coordinate
    ExprVector constraints;   // the constraint equations (each == 0)
};

struct MultiplierSystem {
    ExprVector stationarity;  // df/dx_i − Σ_k λ_k dg_k/dx_i  (== 0) per variable
    ExprVector constraints;   // the constraints g_k = 0
};

enum class ConstraintClass : std::uint8_t {
    holonomic,         // integrable: equivalent to g(q, t) = const (exact or Frobenius-integrable)
    non_holonomic,     // a provable Frobenius obstruction (a nonzero-constant ω ∧ dω term)
    not_determinable,  // an obstruction that could not be proven zero or nonzero here
};
```

| Field | Meaning |
| :--- | :--- |
| `PfaffianConstraint::coeffs` | The `dq_i` coefficients; `coeffs.size()` must equal the number of coordinates. |
| `PfaffianConstraint::time_coeff` | The `dt` term (use `Expr::integer(0)` when absent). |
| `ConstraintClass::holonomic` | The form is integrable — exact/closed, in fewer than three variables, or Frobenius-integrable. |
| `ConstraintClass::non_holonomic` | A **provable** Frobenius obstruction (a nonzero-constant `ω ∧ dω` triple). |
| `ConstraintClass::not_determinable` | An obstruction that structural zero-testing could prove neither zero nor a nonzero constant. |

## Building coordinates

```cpp
[[nodiscard]] auto coordinate(std::string_view base) -> Coordinate;
```

| Helper | Behavior |
| :--- | :--- |
| `coordinate(base)` | The primed jet of a base name: `{base, base + "'", base + "''"}`, e.g. `coordinate("q") == {"q", "q'", "q''"}`. |

## Euler–Lagrange equations

```cpp
[[nodiscard]] auto euler_lagrange(const Expr& F, std::string_view indep,
                                  const std::vector<Coordinate>& coords) -> Result<ExprVector>;
[[nodiscard]] auto euler_lagrange(const Expr& F, std::string_view y, std::string_view yp,
                                  std::string_view ypp, std::string_view x) -> Result<Expr>;
[[nodiscard]] auto lagrange_equations(const Expr& L, std::string_view indep,
                                      const std::vector<Coordinate>& coords) -> Result<ExprVector>;
[[nodiscard]] auto variational_derivative(const Expr& F, std::string_view indep,
                                          const std::vector<Coordinate>& coords)
    -> Result<ExprVector>;
```

| Function | Notation | Returns | Definition |
| :--- | :--- | :--- | :--- |
| `euler_lagrange(F, indep, coords)` | `δJ/δy_i` | `ExprVector` | variational form `∂F/∂y_i − d/d(indep)(∂F/∂y_i')`, one equation (== 0) per coordinate |
| `euler_lagrange(F, y, yp, ypp, x)` | `δJ/δy` | `Expr` | single-variable convenience: the scalar `∂F/∂y − d/dx(∂F/∂y')`; `y`, `yp`, `ypp` name the value / first- / second-derivative symbols and `x` the independent variable |
| `lagrange_equations(L, indep, coords)` | `d/dt(∂L/∂q_i') − ∂L/∂q_i` | `ExprVector` | **mechanics** form (the variational EL negated — Newtonian sign convention, e.g. `q'' + ω²q = 0`), one equation (== 0) per generalized coordinate |
| `variational_derivative(F, indep, coords)` | `δJ/δy_i` | `ExprVector` | the variational derivatives `∂F/∂y_i − d/d(indep)(∂F/∂y_i')` (the EL integrand factors); the extremal condition is that every entry equals zero — identical to `euler_lagrange` |

Several dependent variables / generalized coordinates are supported: the total
`d/d(indep)` chains through **every** coordinate's jet, so coupled systems are
handled exactly. The first variation of `J[y] = ∫ F dx` is
`δJ = ∫ Σ_i (∂F/∂y_i − d/dx ∂F/∂y_i') η_i dx`, and the extremal condition
`δJ = 0` (for arbitrary variations `η_i` vanishing at the endpoints) is exactly
`EL_i = 0`.

## Beltrami first integral

```cpp
[[nodiscard]] auto beltrami_identity(const Expr& F, std::string_view indep,
                                     const std::vector<Coordinate>& coords) -> Result<Expr>;
[[nodiscard]] auto beltrami_identity(const Expr& F, std::string_view y, std::string_view yp,
                                     std::string_view x) -> Result<Expr>;
```

| Function | Returns | Definition |
| :--- | :--- | :--- |
| `beltrami_identity(F, indep, coords)` | `Expr` | `F − Σ_i y_i' ∂F/∂y_i'`, constant along extremals — a first-order first integral. **Requires** `F` free of explicit `indep` dependence; otherwise fails `MathError::domain_error`. |
| `beltrami_identity(F, y, yp, x)` | `Expr` | single-variable convenience: `y`, `yp` name the value / first-derivative symbols and `x` the independent variable. |

When `F` has no **explicit** dependence on the independent variable,
`F − Σ_i y_i' ∂F/∂y_i'` is constant along extremals. Beltrami does **not**
apply when `F` depends explicitly on `indep`; that case is rejected with
`MathError::domain_error` rather than returning a meaningless expression.

## Constrained Euler–Lagrange via Lagrange multipliers

```cpp
[[nodiscard]] auto constrained_euler_lagrange(const Expr& L, std::string_view indep,
                                              const std::vector<Coordinate>& coords,
                                              const ExprVector& holonomic,
                                              const std::vector<std::string>& multipliers)
    -> Result<ConstrainedSystem>;
[[nodiscard]] auto constrained_euler_lagrange(const Expr& L, std::string_view indep,
                                              const std::vector<Coordinate>& coords,
                                              const std::vector<PfaffianConstraint>& constraints,
                                              const std::vector<std::string>& multipliers)
    -> Result<ConstrainedSystem>;
[[nodiscard]] auto lagrange_multipliers(const Expr& f, const std::vector<std::string>& vars,
                                        const ExprVector& constraints,
                                        const std::vector<std::string>& multipliers)
    -> Result<MultiplierSystem>;
```

| Function | Constraint kind | Definition |
| :--- | :--- | :--- |
| `constrained_euler_lagrange(L, indep, coords, holonomic, multipliers)` | holonomic `g_k(q, t) = 0` | forms `L* = L + Σ_k λ_k g_k` and returns the augmented mechanics EL equations `d/dt(∂L*/∂q_i') − ∂L*/∂q_i = 0` together with the echoed constraints `g_k = 0`. `multipliers` names the `λ_k` (size == `holonomic.size()`). |
| `constrained_euler_lagrange(L, indep, coords, constraints, multipliers)` | non-holonomic (Pfaffian) | each coordinate equation becomes `d/dt(∂L/∂q_i') − ∂L/∂q_i − Σ_k λ_k a_{k,i} = 0`; the returned constraints are the velocity forms `Σ_i a_{k,i} q_i' + a_{k,t} = 0`. |
| `lagrange_multipliers(f, vars, constraints, multipliers)` | finite-dimensional | stationary points of `f` subject to `g_k = 0` via `∇f = Σ_k λ_k ∇g_k`; returns the stationarity equations `∂f/∂x_i − Σ_k λ_k ∂g_k/∂x_i = 0` (one per variable) plus the echoed constraints. `multipliers` names `λ_k` (size == `constraints.size()`). |

## Pfaffian classification

```cpp
[[nodiscard]] auto classify_pfaffian(const ExprVector& coeffs,
                                     const std::vector<std::string>& vars)
    -> Result<ConstraintClass>;
```

Classify a Pfaffian 1-form `Σ_i coeffs[i] d(vars[i]) = 0`. The decision
procedure is layered, cheapest-first:

1. **Exactness (closedness).** If `∂a_i/∂x_j == ∂a_j/∂x_i` for all `i < j`
   (structural zero after simplification), the form is `dP` for a potential
   `P` — **holonomic**.
2. **Low dimension.** In fewer than three variables a 1-form is always
   integrable (`ω ∧ dω` is a 3-form and vanishes identically), so it admits an
   integrating factor — **holonomic**.
3. **Frobenius.** Otherwise test `ω ∧ dω = 0` over every variable triple. A
   triple that simplifies to a **provably nonzero constant** proves a
   Frobenius obstruction — **non_holonomic**. A triple that is nonzero here but
   not provably so in general leaves the form **not_determinable**. If every
   triple vanishes, the form is **holonomic**.

`coeffs.size()` must equal `vars.size()` (and be nonzero).

## Error model

Every entry point returns `Result<T>` (Rule 32 — no exceptions). Dimension
mismatches and inapplicable inputs surface as `MathError::domain_error`:

| Function | `domain_error` when |
| :--- | :--- |
| `euler_lagrange` / `lagrange_equations` / `variational_derivative` | `coords` is empty |
| `beltrami_identity` | `coords` is empty, **or** `F` depends explicitly on `indep` (Beltrami does not apply) |
| `constrained_euler_lagrange` (holonomic) | `holonomic.size() != multipliers.size()` |
| `constrained_euler_lagrange` (Pfaffian) | `constraints.size() != multipliers.size()`, **or** any `PfaffianConstraint::coeffs.size() != coords.size()` |
| `lagrange_multipliers` | `constraints.size() != multipliers.size()` |
| `classify_pfaffian` | `coeffs.size() != vars.size()`, or `coeffs` is empty |

Any error raised by the underlying [`differentiate`](diff.md) or
[`simplify`](simplify.md) — for instance an integer overflow surfaced during
simplification — propagates unchanged through the `Result`.

Note that an **undecidable** Frobenius obstruction is **not** an error: it is
reported in-band as `ConstraintClass::not_determinable`, preserving the honesty
boundary that the module never guesses integrability.

## Worked examples

```cpp
import nimblecas.calcvar;
import nimblecas.symbolic;
import nimblecas.simplify;
using namespace nimblecas;

const Expr x   = Expr::symbol("x");
const Expr y   = Expr::symbol("y");
const Expr z   = Expr::symbol("z");
const Expr yp  = Expr::symbol("yp");    // y'
const Expr ypp = Expr::symbol("ypp");   // y''
auto I = [](std::int64_t n) { return Expr::integer(n); };
auto R = [](std::int64_t n, std::int64_t d) { return Expr::rational(n, d).value(); };

// Shortest path: F = sqrt(1 + y'^2). F is free of y, so dF/dy = 0 and the whole
// EL equation is proportional to y''; a straight line (y'' = 0) is an exact extremal.
const Expr F = Expr::power(Expr::sum({I(1), Expr::power(yp, I(2))}), R(1, 2));
auto el = euler_lagrange(F, "y", "yp", "ypp", "x").value();
free_of(el, y);                                    // true (independent of y)
simplify(substitute(el, ypp, I(0))).value() == I(0);  // true — y'' = 0 satisfies EL exactly

// Brachistochrone: F = sqrt((1 + y'^2)/y) has no explicit x, so Beltrami applies.
const Expr Fb = Expr::power(
    Expr::product({Expr::sum({I(1), Expr::power(yp, I(2))}), Expr::power(y, I(-1))}),
    R(1, 2));
auto b = beltrami_identity(Fb, "y", "yp", "x").value();
free_of(b, x);                                     // true — first integral has no explicit x
free_of(b, ypp);                                   // true — first order (no y'')

// Beltrami rejects an integrand with explicit x.
const Expr Fx = Expr::product({x, Expr::power(Expr::sum({I(1), Expr::power(yp, I(2))}), R(1, 2))});
beltrami_identity(Fx, "y", "yp", "x").error();     // MathError::domain_error

// Harmonic oscillator: L = 1/2 q'^2 - 1/2 w^2 q^2  ->  q'' + w^2 q = 0.
const Expr q = Expr::symbol("q"), qd = Expr::symbol("qd"), qdd = Expr::symbol("qdd");
const Expr w = Expr::symbol("omega");
const Expr L = Expr::sum(
    {Expr::product({R(1, 2), Expr::power(qd, I(2))}),
     Expr::product({R(-1, 2), Expr::power(w, I(2)), Expr::power(q, I(2))})});
auto eqs = lagrange_equations(L, "t",
    {Coordinate{.value = "q", .velocity = "qd", .acceleration = "qdd"}}).value();
// eqs.size() == 1;  eqs[0] == qdd + w^2 * q

// Constrained pendulum: L = 1/2 m (vx^2 + vy^2) - m g y, holonomic g = x^2 + y^2 - l^2 = 0.
const Expr m = Expr::symbol("m"), grav = Expr::symbol("grav"), ell = Expr::symbol("ell");
const std::vector<Coordinate> coords = {
    Coordinate{.value = "x", .velocity = "vx", .acceleration = "ax"},
    Coordinate{.value = "y", .velocity = "vy", .acceleration = "ay"}};
const Expr Lp = Expr::sum(
    {Expr::product({R(1, 2), m, Expr::power(Expr::symbol("vx"), I(2))}),
     Expr::product({R(1, 2), m, Expr::power(Expr::symbol("vy"), I(2))}),
     Expr::product({I(-1), m, grav, y})});
const Expr g = Expr::sum(
    {Expr::power(x, I(2)), Expr::power(y, I(2)), Expr::product({I(-1), Expr::power(ell, I(2))})});
auto sys = constrained_euler_lagrange(Lp, "t", coords, {g}, {"lam"}).value();
// sys.equations[0] == m*ax - 2*lam*x        (EL_x)
// sys.equations[1] == m*ay + m*grav - 2*lam*y  (EL_y)
// sys.constraints[0] == x^2 + y^2 - l^2      (echoed)

// Non-holonomic (as posed) free particle: L = 1/2 m (vx^2 + vy^2), Pfaffian dx - dy = 0.
const PfaffianConstraint pf{.coeffs = {I(1), I(-1)}, .time_coeff = I(0)};
auto nh = constrained_euler_lagrange(
    Expr::sum({Expr::product({R(1, 2), m, Expr::power(Expr::symbol("vx"), I(2))}),
               Expr::product({R(1, 2), m, Expr::power(Expr::symbol("vy"), I(2))})}),
    "t", coords, {pf}, {"lam"}).value();
// nh.equations[0] == m*ax - lam;  nh.equations[1] == m*ay + lam
// nh.constraints[0] == vx - vy               (velocity form)

// Finite-dimensional multipliers: extremize f = x + y subject to g = x^2 + y^2 - 1 = 0.
auto ms = lagrange_multipliers(
    Expr::sum({x, y}), {"x", "y"},
    {Expr::sum({Expr::power(x, I(2)), Expr::power(y, I(2)), I(-1)})}, {"lam"}).value();
// ms.stationarity[0] == 1 - 2*lam*x;  ms.stationarity[1] == 1 - 2*lam*y
// ms.constraints[0]  == x^2 + y^2 - 1

// Pfaffian classification.
classify_pfaffian({Expr::product({I(-1), y}), I(0), I(1)}, {"x", "y", "z"}).value();
                                                   // ConstraintClass::non_holonomic  (dz - y dx)
classify_pfaffian({y, x}, {"x", "y"}).value();     // ConstraintClass::holonomic  (y dx + x dy = d(x y))
classify_pfaffian({Expr::product({y, z}), Expr::product({x, z}), Expr::product({x, y})},
                  {"x", "y", "z"}).value();         // ConstraintClass::holonomic  (d(x y z))
classify_pfaffian({y, x}, {"x", "y", "z"}).error();// MathError::domain_error  (count mismatch)
```

## See also

- [`nimblecas.diff`](diff.md) — the per-symbol partial derivative every
  Euler–Lagrange assembly and Frobenius test composes.
- [`nimblecas.simplify`](simplify.md) — post-processes every assembled equation,
  and whose structural zero-testing drives the Pfaffian classification.
- [`nimblecas.symbolic`](symbolic.md) — the `Expr` model the coordinates and
  constraints are built from.
- [`nimblecas.vectorcalc`](vectorcalc.md) — the sibling exact multivariable
  differentiation operators from the same ROADMAP §7.19 layer.
- [Documentation hub](../Index.md)
