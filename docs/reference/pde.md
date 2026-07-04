# `nimblecas.pde` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/pde/pde.cppm`

Exact **partial differential equation** solvers over the rationals. Everything
here is exact over **Q** — polynomial initial/boundary data with
rational-polynomial coefficients — and **no floating point ever enters**. Several
families live in this module, each with an honest boundary on what "exact" buys
you:

**(a) Linear evolution** `u_t = L[u]`, `u(x,0) = φ(x)`, where `L` is a spatial
differential operator with rational-polynomial coefficients acting on functions
of `x`. Its exact solution is the classical time power series (the
**Cauchy–Kovalevskaya construction**)

```
u(x, t) = Σ_{n≥0} (L^n[φ](x) / n!) t^n,
```

so the coefficient of `t^n` is `c_n(x) = L^n[φ] / n!`, itself a polynomial in
`x`. The `c_n` are built iteratively without ever forming a factorial:
`c_0 = φ` and `c_n = L[c_{n-1}] · (1/n)`, which telescopes to `L^n[φ]/n!`
exactly. For polynomial `φ` under a **constant-coefficient** `L` (which strictly
lowers a polynomial's degree) the series **terminates** — `L^n[φ] = 0` once `n`
exceeds the degree budget — so the truncated result *is* the closed-form
solution. This is the whole-line / formal power-series construction: **no
boundary conditions are imposed**, and it is neither a boundary-value nor a
Fourier solver. For non-polynomial `φ` (or an `L` that does not preserve
polynomials) the same recurrence yields the exact Taylor coefficients up to the
requested order — a truncated approximation whose error is bounded by the
truncation order.

**(b) Nonlinear evolution** `u_t = L[u] + N[u]`, `u(x,0) = φ`, with `L` a linear
polynomial-coefficient operator and `N` a (possibly nonlinear) spatial term.
Expanding `u(x,t) = Σ_n c_n(x) t^n` and matching `[t^k]` on both sides (`u_t`
shifts the index by one) gives the exact recurrence

```
(k+1) c_{k+1} = L[c_k] + A_k,     A_k = [t^k] N( Σ_{i≤k} c_i(x) t^i ),
```

where `A_k` is the `k`-th **Adomian polynomial** of `N`: the time-grade-`k`
projection of `N` applied to the partial sum. This is the same
graded-projection Adomian construction as [`nimblecas.perturbation`](perturbation.md)
(see that module for why the graded projection of `N` of the *whole* partial
sum — not a naive difference of partial sums — is the correct `A_k`),
transported from `Q[[x]]` to the ring `(Q[x])[[t]]`: each series coefficient
`c_n(x)` is a full spatial `RationalPoly` rather than a `Rational`, so the
projection is performed natively over `RationalPoly` and **no link dependency on
`nimblecas.perturbation` is added**. Because `[t^k] N` depends only on
`c_0 … c_k` (products cannot lower the `t`-degree and the spatial derivative
preserves it), the recurrence is causal and well defined.

**(c) 1-D Poisson / Dirichlet BVP** `u''(x) = f(x)` on `[a, b]` with
`u(a) = α`, `u(b) = β`, for a polynomial source `f`. A particular `p` with
`p'' = f` is obtained by integrating `f` twice; then `u = p + C₁x + C₀` with
`(C₁, C₀)` fixed by the exact 2×2 rational solve of the two boundary equations.

**(d) Wave equation** `u_tt = L[u]`, `u(x,0) = φ`, `u_t(x,0) = ψ` — a
**second-order-in-time** linear evolution. Being second order it needs **two**
Cauchy data (φ and ψ); matching `[t^n]` (with `u_tt` shifting the index *down* by
two) gives the exact two-step recurrence `(n+2)(n+1) c_{n+2} = L[c_n]`,
`c_0 = φ`, `c_1 = ψ`, so the even coefficients descend from `φ` and the odd ones
from `ψ`. For polynomial data under a constant-coefficient `L` the series
**terminates** and the truncation is the closed form (e.g. `u_tt = c² u_xx`,
`φ = x²`, `ψ = 0` ⟹ `u = x² + c² t²`); otherwise it is the exact truncated Taylor
series in `t`.

**(e) KdV** `u_t + u u_x + u_xxx = 0` — the dispersive third-derivative term is
**linear**, so it is carried as `L[u] = −u_xxx`, while the convective `u u_x` is
the same nonlinearity shape as inviscid Burgers and reuses the Adomian machinery
of **(b)**. Exact truncated series, **local in `t`**: for a genuine (non-linear)
datum it does **not** terminate and is **not** a closed-form soliton.

**(f) Schrödinger** `i u_t = −u_xx + V u`, `u(x,0) = φ` — rearranged to the
first-order-in-time form `u_t = i(u_xx − V u)`, whose coefficients are
**intrinsically complex** (the recurrence carries the exact factor `i`) and so
cannot live in the real `RationalPoly` ring. They live instead in a small
Gaussian-rational polynomial type `ComplexPoly` (`re + i·im`, both `RationalPoly`
over `Q[x]`), keeping the whole construction exact in `(Q + iQ)[x]`. Free-particle
polynomial data terminates (`φ = x²` ⟹ `x² + 2 i t`); a constant potential
recovers the `e^{−it}` phase series.

**Honesty boundary.** The **linear evolution** series is exact and (for
polynomial data under a constant-coefficient `L`) closed-form because it
terminates. The **nonlinear evolution** results are the **exact truncated Taylor
series in `t`**: the returned `c_0 … c_order` are exact rationals-in-`x` and the
discrete law `(k+1) c_{k+1} = L[c_k] + A_k` holds *exactly* on every retained
term, but for a genuine nonlinearity the series does **not** in general
terminate. It is a **LOCAL-in-`t`** solution, valid only inside the time radius
of convergence, and is **NOT a global closed form** — exact coefficient by
coefficient up to `order`, nothing more. (Inviscid Burgers `u(x,0) = x` yields
`x − x t + x t² − … = x/(1+t)`, singular at `t = −1`; reaction `u_t = u²` with
`u(x,0) = 1` yields `1 + t + t² + … = 1/(1−t)`.) The **BVP**, unlike the
evolution series, is a genuine **closed-form polynomial over Q — no truncation**.
No boundary conditions are imposed by either evolution solver (whole-line /
formal construction). Rule 32 railway: every `RationalPoly` / `Rational`
`Result` error is propagated; `order == 0`, missing operators, and a degenerate
BVP interval surface as `MathError::domain_error`.

```cpp
import nimblecas.pde;
```

Depends on [`core`](core.md) and [`ratpoly`](ratpoly.md).

## Linear evolution: `u_t = L[u]`

### `SpatialOperator` — the spatial operator `L`

```cpp
using SpatialOperator = std::function<Result<RationalPoly>(const RationalPoly&)>;
```

The spatial operator `L` of the evolution PDE `u_t = L[u]`: it consumes a
spatial polynomial `p(x)` and returns `L[p](x)` on the railway. `L = φ ↦ φ''`
models the heat equation, `L = φ ↦ φ'` the transport equation, and so on.

### Free functions

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `solve_evolution_pde` | `[[nodiscard]] auto solve_evolution_pde(SpatialOperator l, const RationalPoly& phi, std::size_t order) -> Result<std::vector<RationalPoly>>` | Solve `u_t = L[u]`, `u(x,0) = φ`, returning the time-series coefficients `c_0 … c_order`, where `c_n(x)` is the coefficient of `t^n`. `c_0 = φ` and `c_n = L[c_{n-1}] · (1/n)`, so the returned vector holds `order+1` `RationalPoly`s and equals `L^n[φ]/n!` term by term without forming any factorial. For polynomial `φ` under a polynomial-preserving `L` the trailing `c_n` are the zero polynomial (the series terminates). See below. |
| `evaluate` | `[[nodiscard]] auto evaluate(const std::vector<RationalPoly>& coeffs, const Rational& x, const Rational& t) -> Result<Rational>` | Evaluate the **truncated** solution `Σ_n c_n(x) t^n` exactly at the rational point `(x, t)`, using Horner in `t` and exact polynomial evaluation of each `c_n` at `x`. The coefficient list must be non-empty. See below. |

#### `solve_evolution_pde`

`MathError::domain_error` if `order == 0` or `l` is an empty callable. An
`order` beyond `INT64_MAX` (physically unreachable) is `MathError::overflow`, so
the internal cast for `1/n` can never wrap. Any error raised by `L` is
propagated verbatim.

#### `evaluate`

Horner in `t` (`acc ← acc·t + c_n(x)`) sweeping `n` from the top coefficient
down, with each `c_n(x)` itself evaluated by exact Horner in `x`. This is the
value of the **truncated** series only — the exact solution unless the series
terminates. An empty coefficient list is `MathError::domain_error`; any
`Rational` overflow is propagated.

### Convenience operator builders

| Function | Signature | Operator |
| :--- | :--- | :--- |
| `heat_operator` | `[[nodiscard]] auto heat_operator(Rational diffusivity) -> SpatialOperator` | Heat / diffusion `L[φ] = diffusivity · φ''` (models `u_t = a u_xx`). |
| `transport_operator` | `[[nodiscard]] auto transport_operator(Rational speed) -> SpatialOperator` | Transport / advection `L[φ] = speed · φ'` (models `u_t = c u_x`). |
| `advection_diffusion` | `[[nodiscard]] auto advection_diffusion(Rational speed, Rational diffusivity) -> SpatialOperator` | Combined `L[φ] = speed · φ' + diffusivity · φ''`. |

Each builds a constant-coefficient `L` from `RationalPoly::derivative` and
`scale`; any `RationalPoly` error is propagated through the returned callable.

## Nonlinear evolution: `u_t = L[u] + N[u]`

### `TimeSeriesOperator` — a spatial term on a truncated time series

```cpp
using TimeSeriesOperator =
    std::function<Result<std::vector<RationalPoly>>(const std::vector<RationalPoly>&)>;
```

A spatial term acting on a **truncated time series**. Given the coefficients
`c_0 … c_M` of `u(x,t) = Σ_n c_n(x) t^n` (a vector of `M+1` `RationalPoly`s), it
returns the time series of `N[u](x,t)` as a vector of the **same length**. For
Burgers `N[u] = −(u · u_x)`; for the quadratic reaction `N[u] = u²`. Built from
the `series_*` primitives below.

### Time-series primitives (Cauchy product in `t` / spatial derivative)

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `series_scale` | `[[nodiscard]] auto series_scale(const std::vector<RationalPoly>& a, const Rational& s) -> Result<std::vector<RationalPoly>>` | Scale every time coefficient by the rational `s`. |
| `series_dx` | `[[nodiscard]] auto series_dx(const std::vector<RationalPoly>& a) -> Result<std::vector<RationalPoly>>` | Spatial `x`-derivative applied coefficient-wise (`d/dx` of each `c_n(x)`); the length is preserved, so the series of `u_x` is returned to the same time-truncation order. |
| `series_product` | `[[nodiscard]] auto series_product(const std::vector<RationalPoly>& a, const std::vector<RationalPoly>& b) -> Result<std::vector<RationalPoly>>` | Truncated Cauchy product in `t`: `result_k = Σ_{i+j=k} a_i(x) b_j(x)`, `0 ≤ k < len`. Both operands must have **equal length** (else `domain_error`). |

### `solve_nonlinear_evolution_pde`

```cpp
[[nodiscard]] auto solve_nonlinear_evolution_pde(SpatialOperator linear,
                                                 TimeSeriesOperator nonlinear,
                                                 const RationalPoly& phi,
                                                 std::size_t order)
    -> Result<std::vector<RationalPoly>>;
```

Solve `u_t = L[u] + N[u]`, `u(x,0) = φ`, returning the time coefficients
`c_0 … c_order` (`order+1` `RationalPoly`s, `c_n` = coefficient of `t^n`) via the
Adomian / Cauchy–Kovalevskaya recurrence `(k+1) c_{k+1} = L[c_k] + A_k` above.
`linear` **may be empty** (treated as `L = 0`, i.e. a pure-nonlinear PDE);
`nonlinear` **must be non-empty**. The result is the **EXACT truncated series**:
local in `t`, not global (see the honesty boundary above).

`MathError::domain_error` if `order == 0`, if `nonlinear` is a null callable, or
if the nonlinear term returns a vector whose length differs from the truncation
length `order+1` (`N` must preserve the length). An `order` beyond `INT64_MAX`
is `MathError::overflow`. Any error raised by `L` or `N` is propagated verbatim.

### Concrete nonlinear builders

| Function | Signature | PDE |
| :--- | :--- | :--- |
| `burgers` | `[[nodiscard]] auto burgers(Rational viscosity, const RationalPoly& phi, std::size_t order) -> Result<std::vector<RationalPoly>>` | Burgers' equation `u_t + u u_x = viscosity · u_xx`, i.e. `u_t = viscosity · u_xx − u u_x`. Takes `L = heat_operator(viscosity)`, `N[u] = −(u · u_x)`. `viscosity == 0` gives the **inviscid** equation. |
| `reaction_diffusion_quadratic` | `[[nodiscard]] auto reaction_diffusion_quadratic(Rational diffusivity, const RationalPoly& phi, std::size_t order) -> Result<std::vector<RationalPoly>>` | Quadratic reaction–diffusion `u_t = diffusivity · u_xx + u²` (Fisher-type source). Takes `L = heat_operator(diffusivity)`, `N[u] = u²`. |

Both return `c_0 … c_order` — the **exact truncated time series, local in `t`**
— and forward to `solve_nonlinear_evolution_pde`, so they share its error model.

## Boundary-value / steady-state: 1-D Poisson

```cpp
[[nodiscard]] auto solve_poisson_bvp_1d(const RationalPoly& f, const Rational& a,
                                        const Rational& alpha, const Rational& b,
                                        const Rational& beta) -> Result<RationalPoly>;
```

Solve the Dirichlet BVP `u''(x) = f(x)` on `[a, b]` with `u(a) = α`, `u(b) = β`,
for a polynomial source `f`. **EXACT over Q**: a particular `p` with `p'' = f` is
obtained by integrating `f` twice (zero constants), then `u = p + C₁x + C₀` with
`(C₁, C₀)` fixed by the exact 2×2 rational solve

```
C₁ = ( (α − β) − (p(a) − p(b)) ) / (a − b),      C₀ = α − p(a) − C₁ a.
```

Unlike the evolution series this is a **genuine closed-form polynomial — no
truncation**. `a == b` is a degenerate interval (the two Dirichlet conditions
collapse onto one point) and is `MathError::domain_error`; any `Rational`
overflow is propagated. Note that with `f = 0` this returns the exact linear
interpolant `u(a) = α`, `u(b) = β` — the 1-D harmonic (Laplace) solution.

## Wave equation: `u_tt = L[u]` (second order in time)

```cpp
[[nodiscard]] auto solve_wave_pde(SpatialOperator l, const RationalPoly& phi,
                                  const RationalPoly& psi, std::size_t order)
    -> Result<std::vector<RationalPoly>>;
[[nodiscard]] auto wave_equation(Rational speed, const RationalPoly& phi,
                                 const RationalPoly& psi, std::size_t order)
    -> Result<std::vector<RationalPoly>>;
```

`solve_wave_pde` solves the **second-order-in-time** linear PDE `u_tt = L[u]`
with **two** initial data `u(x,0) = φ` and `u_t(x,0) = ψ` — for a second-order
equation one datum no longer determines the solution. Expanding
`u(x,t) = Σ_n c_n(x) t^n` and matching `[t^n]` (where `u_tt` shifts the time
index *down* by two) gives the exact **two-step Cauchy–Kovalevskaya recurrence**

```
(n+2)(n+1) c_{n+2} = L[c_n],      c_0 = φ,  c_1 = ψ,
```

so the even coefficients descend from `φ` and the odd ones from `ψ`. The two
divisions by `(n+1)` and `(n+2)` are performed **separately** (never as the
product `(n+1)(n+2)`) so the step scalars cannot overflow `int64` for large
`order`. The returned vector holds `order+1` `RationalPoly`s. For polynomial
`φ`, `ψ` under a constant-coefficient `L` the series **terminates** and the
truncation is the closed form (e.g. `u_tt = c² u_xx`, `φ = x²`, `ψ = 0` ⟹
`u = x² + c² t²`); otherwise it is the **exact truncated Taylor series in `t`**
(local in `t`, no boundary conditions imposed).

`wave_equation` wires the classical wave equation `u_tt = speed² · u_xx` into
`solve_wave_pde` by taking `L = heat_operator(speed²)` (i.e. `speed² · d²/dx²`),
hence `c_2 = (speed² / 2) · φ''`. `speed` may be any rational, including `0`.

`MathError::domain_error` if `order == 0` (which cannot carry the second datum
`ψ`) or `l` is an empty callable; an `order` beyond `INT64_MAX` is
`MathError::overflow`; any error raised by `L` (or by the `speed²` product in
`wave_equation`) is propagated verbatim.

## KdV: `u_t + u u_x + u_xxx = 0`

```cpp
[[nodiscard]] auto kdv(const RationalPoly& phi, std::size_t order)
    -> Result<std::vector<RationalPoly>>;
```

Solve the Korteweg–de Vries equation `u_t + u u_x + u_xxx = 0`, i.e.
`u_t = −u_xxx − u u_x`, `u(x,0) = φ`. The dispersive third-derivative term is
**linear** and is carried as the `SpatialOperator` `L[u] = −u_xxx`; the
convective term `N[u] = −(u · u_x)` is the same nonlinearity shape as inviscid
Burgers and reuses the Adomian / Cauchy–Kovalevskaya machinery of
`solve_nonlinear_evolution_pde`. Returns the time coefficients `c_0 … c_order`
(`order+1` `RationalPoly`s).

This is the **EXACT truncated Taylor series in `t`**: for a genuine
(non-linear-datum) `φ` the series does **not** terminate — it is a **LOCAL-in-`t`**
exact solution, **NOT** a global closed form and **NOT** a travelling-wave
soliton in closed form. (For a linear datum `φ = x`, `φ''' = 0`, dispersion
vanishes and KdV reduces to inviscid Burgers with `c_n = (−1)^n x`.)
`MathError::domain_error` if `order == 0`; any `RationalPoly` error propagates.

## Schrödinger equation: `i u_t = −u_xx + V u`

### `ComplexPoly` — Gaussian-rational polynomial

```cpp
struct ComplexPoly {
    RationalPoly re;  // real part (polynomial in x over Q)
    RationalPoly im;  // imaginary part (polynomial in x over Q)

    [[nodiscard]] static auto make(RationalPoly real_part, RationalPoly imag_part) -> ComplexPoly;
    [[nodiscard]] auto is_zero() const noexcept -> bool;
    [[nodiscard]] auto is_equal(const ComplexPoly& o) const noexcept -> bool;
};
```

A **Gaussian-rational polynomial** `re + i·im`, where `re` and `im` are exact
`RationalPoly`s over `Q[x]`. A Schrödinger time coefficient is intrinsically
**complex** (the recurrence carries the exact factor `i`), so it cannot live in
the real `RationalPoly` ring; `ComplexPoly` keeps the whole construction exact in
`(Q + iQ)[x]` using only the existing `RationalPoly` substrate — **no new module
dependency** is pulled in. `make` builds a value from its real and imaginary
parts; `is_zero` holds when both parts are the zero polynomial; `is_equal`
compares both parts.

### Free functions

```cpp
[[nodiscard]] auto solve_schrodinger(const RationalPoly& potential, const RationalPoly& phi,
                                     std::size_t order) -> Result<std::vector<ComplexPoly>>;
[[nodiscard]] auto schrodinger_free_particle(const RationalPoly& phi, std::size_t order)
    -> Result<std::vector<ComplexPoly>>;
```

`solve_schrodinger` solves the time-dependent Schrödinger equation
`i u_t = −u_xx + V(x) u`, `u(x,0) = φ`, for a real polynomial potential `V` and
real polynomial initial datum `φ`. Rearranged to the first-order-in-time form
`u_t = M[u]` with the **exact complex operator**

```
M[u] = i · ( u_xx − V·u ),
```

the Cauchy–Kovalevskaya recurrence is `c_0 = φ` (as `re = φ`, `im = 0`) and
`c_n = M[c_{n-1}] / n`, evaluated natively over `ComplexPoly`. Returns the exact
truncated series `c_0 … c_order` (`order+1` `ComplexPoly`s).

Coefficients are **exact Gaussian-rational polynomials** and the discrete law
`n · c_n = M[c_{n-1}]` holds exactly on every retained term. For the **free
particle** (`V = 0`) and polynomial `φ` the operator strictly lowers spatial
degree, so the series **terminates** and the truncation is the closed form (e.g.
`φ = x²` ⟹ `x² + 2 i t`). For a non-zero polynomial potential `V`,
multiplication by `V` raises degree and the series need **not** terminate: it is
then the **EXACT truncated Taylor series in `t`**, **LOCAL in `t`**, with no
boundary conditions imposed (a constant potential recovers the `e^{−it}` phase
series). `MathError::domain_error` if `order == 0`; an `order` beyond `INT64_MAX`
is `MathError::overflow`; any `RationalPoly` error propagates.

`schrodinger_free_particle` is the free-particle special case `i u_t = −u_xx`
(`V = 0`) — a thin wrapper over `solve_schrodinger` with a zero potential; for
polynomial `φ` the exact series terminates (closed form).

## Error model

| Condition | Error |
| :--- | :--- |
| `solve_evolution_pde` with `order == 0` or an empty `SpatialOperator` | `MathError::domain_error` |
| `solve_nonlinear_evolution_pde` (or `burgers` / `reaction_diffusion_quadratic`) with `order == 0` | `MathError::domain_error` |
| `solve_nonlinear_evolution_pde` with a null `TimeSeriesOperator` (`nonlinear`) | `MathError::domain_error` |
| The nonlinear term `N` returns a vector whose length differs from `order+1` | `MathError::domain_error` |
| `series_product` operands of differing length | `MathError::domain_error` |
| `evaluate` on an empty coefficient list | `MathError::domain_error` |
| `solve_poisson_bvp_1d` with a degenerate interval `a == b` | `MathError::domain_error` |
| `solve_wave_pde` (or `wave_equation`) with `order == 0` (cannot carry the second datum `ψ`) or an empty `SpatialOperator` | `MathError::domain_error` |
| `kdv` with `order == 0` | `MathError::domain_error` |
| `solve_schrodinger` (or `schrodinger_free_particle`) with `order == 0` | `MathError::domain_error` |
| `order` beyond `INT64_MAX` (physically unreachable step-index cast guard; applies to every series solver) | `MathError::overflow` |
| Any `int64` numerator/denominator computation in `Rational` wraps (Horner evaluation, scaling, the 2×2 BVP solve) | `MathError::overflow` (propagated) |
| Any error raised by `L` or `N`, or by an underlying `RationalPoly` operation (`derivative`, `scale`, `multiply`, `add`, `divide`) | that operation's error, propagated verbatim |

The operators are plain `std::function`s and the solvers thread every fallible
step through `Result`; nothing throws.

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.ratpoly;
import nimblecas.pde;
using namespace nimblecas;

auto ipoly = [](std::vector<std::int64_t> c) {          // RationalPoly, low degree first
    return RationalPoly::from_polynomial(Polynomial{std::move(c)});
};
auto ri  = [](std::int64_t v) { return Rational::from_int(v); };
auto rat = [](std::int64_t n, std::int64_t d) { return *Rational::make(n, d); };

// --- Linear heat: u_t = u_xx, phi = x^2 -> u = x^2 + 2t (series TERMINATES) -
auto heat = heat_operator(ri(1));
auto c = solve_evolution_pde(heat, ipoly({0, 0, 1}), 3).value();
c[0].is_equal(ipoly({0, 0, 1}));   // c_0 = x^2
c[1].is_equal(ipoly({2}));         // c_1 = 2
c[2].is_zero();                    // c_2 = 0  (terminated: closed form)
c[3].is_zero();                    // c_3 = 0
evaluate(c, ri(1), ri(1)).value(); // 3   (u(1,1) = 1 + 2)

// --- Linear transport: u_t = u_x, phi = x^2 -> (x+t)^2 --------------------
auto tr = solve_evolution_pde(transport_operator(ri(1)), ipoly({0, 0, 1}), 3).value();
tr[1].is_equal(ipoly({0, 2}));     // c_1 = 2x
tr[2].is_equal(ipoly({1}));        // c_2 = 1
evaluate(tr, ri(3), ri(2)).value();// 25  = (3 + 2)^2

// --- Heat x^4 evaluated at a fractional point (truncation is exact here) --
auto h4 = solve_evolution_pde(heat, ipoly({0, 0, 0, 0, 1}), 2).value();  // x^4 + 12x^2 t + 12 t^2
evaluate(h4, rat(1, 2), rat(1, 3)).value();   // 115/48

// --- Nonlinear reaction u_t = u_xx + u^2, phi = 1: the ODE u_t = u^2 ------
// Exact solution 1/(1-t) = 1 + t + t^2 + ..., so every c_n(x) = 1.
auto rq = reaction_diffusion_quadratic(ri(1), ipoly({1}), 3).value();
rq[0].is_equal(ipoly({1}));        // c_0 = 1
rq[1].is_equal(ipoly({1}));        // c_1 = 1
rq[2].is_equal(ipoly({1}));        // c_2 = 1
rq[3].is_equal(ipoly({1}));        // c_3 = 1
// TRUNCATION SEMANTICS: at t = 1/2 the order-3 partial sum is 1 + 1/2 + 1/4 + 1/8,
// which is 15/8 -- NOT the closed form 1/(1-1/2) = 2. Local in t, not global.
evaluate(rq, ri(0), rat(1, 2)).value();   // 15/8   (assert the PARTIAL sum)

// --- Nonlinear reaction, phi = x: hand-derived graded recurrence ----------
// c_0 = x, c_1 = x^2, c_2 = 1 + x^3, c_3 = (8/3) x + x^4.
auto rx = reaction_diffusion_quadratic(ri(1), ipoly({0, 1}), 3).value();
rx[2].is_equal(ipoly({1, 0, 0, 1}));      // c_2 = 1 + x^3
rx[3].is_equal(RationalPoly::from_coeffs({rat(0,1), rat(8,3), rat(0,1), rat(0,1), rat(1,1)}));

// --- Inviscid Burgers: u_t + u u_x = 0, phi = x -> x/(1+t) ----------------
// Exact solution x - x t + x t^2 - ..., so c_n = (-1)^n x.
auto bg = burgers(ri(0), ipoly({0, 1}), 3).value();
bg[1].is_equal(ipoly({0, -1}));    // c_1 = -x
bg[2].is_equal(ipoly({0,  1}));    // c_2 =  x
// TRUNCATION: u(1, 1/2) partial sum 1 - 1/2 + 1/4 - 1/8 = 5/8 (NOT x/(1+t) = 2/3).
evaluate(bg, ri(1), rat(1, 2)).value();   // 5/8   (assert the PARTIAL sum)

// Viscous Burgers on a linear datum: u_xx = 0, so viscosity is inert here.
burgers(ri(1), ipoly({0, 1}), 3).value()[1].is_equal(ipoly({0, -1}));  // c_1 = -x

// --- 1-D Poisson BVP: u'' = 2 on [0,1], u(0)=u(1)=0 -> x^2 - x (EXACT) ----
auto u = solve_poisson_bvp_1d(ipoly({2}), ri(0), ri(0), ri(1), ri(0)).value();
u.is_equal(ipoly({0, -1, 1}));     // u = x^2 - x   (closed form, no truncation)

// u'' = x on [0,1], u(0)=u(1)=0 -> x^3/6 - x/6 (exact rational).
solve_poisson_bvp_1d(ipoly({0, 1}), ri(0), ri(0), ri(1), ri(0)).value()
    .is_equal(RationalPoly::from_coeffs({rat(0,1), rat(-1,6), rat(0,1), rat(1,6)}));

// Laplace (f = 0) on [0,2], u(0)=1, u(2)=5 -> linear interpolant 1 + 2x.
solve_poisson_bvp_1d(RationalPoly{}, ri(0), ri(1), ri(2), ri(5)).value()
    .is_equal(ipoly({1, 2}));      // u = 1 + 2x

// --- Wave: u_tt = u_xx, phi = x^2, psi = 0 -> u = x^2 + t^2 (TERMINATES) ---
// d'Alembert ((x+t)^2 + (x-t)^2)/2 = x^2 + t^2.
auto w = wave_equation(ri(1), ipoly({0, 0, 1}), RationalPoly{}, 4).value();
w[0].is_equal(ipoly({0, 0, 1}));   // c_0 = x^2
w[1].is_zero();                    // c_1 = 0  (psi = 0)
w[2].is_equal(ipoly({1}));         // c_2 = (x^2)''/2 = 1
w[3].is_zero();                    // c_3 = 0
w[4].is_zero();                    // c_4 = 0  (terminated: closed form)
evaluate(w, ri(2), ri(1)).value(); // 5   (u(2,1) = 4 + 1)

// Nonzero psi: u_tt = u_xx, phi = 0, psi = x -> u = x t exactly.
auto wp = wave_equation(ri(1), RationalPoly{}, ipoly({0, 1}), 3).value();
wp[1].is_equal(ipoly({0, 1}));     // c_1 = x (psi)
evaluate(wp, ri(3), ri(2)).value();// 6   (u(3,2) = x t)

// --- KdV: u_t + u u_x + u_xxx = 0, phi = x -> reduces to inviscid Burgers --
// phi''' = 0, so dispersion vanishes and c_n = (-1)^n x.
auto kv = kdv(ipoly({0, 1}), 3).value();
kv[1].is_equal(ipoly({0, -1}));    // c_1 = -x   (LOCAL in t, not a soliton)

// --- Schrodinger free particle: i u_t = -u_xx, phi = x^2 -> x^2 + 2 i t ----
auto sf = schrodinger_free_particle(ipoly({0, 0, 1}), 3).value();  // vector<ComplexPoly>
sf[0].is_equal(ComplexPoly::make(ipoly({0, 0, 1}), RationalPoly{}));  // c_0 = x^2
sf[1].is_equal(ComplexPoly::make(RationalPoly{}, ipoly({2})));        // c_1 = 2 i
sf[2].is_zero();                   // c_2 = 0  (terminated: closed form)

// Constant potential V = 1, phi = 1: recovers e^{-i t} = 1 - i t - t^2/2 + ...
auto sc = solve_schrodinger(ipoly({1}), ipoly({1}), 3).value();
sc[1].is_equal(ComplexPoly::make(RationalPoly{}, ipoly({-1})));       // c_1 = -i
sc[2].is_equal(ComplexPoly::make(RationalPoly::from_coeffs({rat(-1,2)}), RationalPoly{})); // c_2 = -1/2

// --- Degenerate arguments are domain errors -------------------------------
solve_evolution_pde(heat, ipoly({0, 0, 1}), 0).error();          // domain_error (order 0)
solve_evolution_pde(SpatialOperator{}, ipoly({1}), 2).error();   // domain_error (empty L)
evaluate({}, ri(1), ri(1)).error();                              // domain_error (empty coeffs)
reaction_diffusion_quadratic(ri(1), ipoly({0, 1}), 0).error();   // domain_error (order 0)
solve_nonlinear_evolution_pde(heat_operator(ri(1)), TimeSeriesOperator{},
                              ipoly({0, 1}), 3).error();          // domain_error (null N)
solve_poisson_bvp_1d(ipoly({2}), ri(1), ri(0), ri(1), ri(0)).error();  // domain_error (a == b)
wave_equation(ri(1), ipoly({0, 0, 1}), RationalPoly{}, 0).error(); // domain_error (order 0, no psi)
kdv(ipoly({0, 0, 0, 1}), 0).error();                               // domain_error (order 0)
schrodinger_free_particle(ipoly({0, 0, 1}), 0).error();            // domain_error (order 0)
```

## See also

- [`nimblecas.ode`](ode.md) — the exact power-series IVP solver for autonomous
  first-order systems and scalar higher-order ODEs, built on the same
  `u = u0 + L⁻¹[f(u)]` graded fixed point.
- [`nimblecas.perturbation`](perturbation.md) — the Adomian / HPM / HAM sibling;
  the graded-projection Adomian construction reused (over `RationalPoly`) by the
  nonlinear evolution solver here.
- [`nimblecas.dde`](dde.md), [`nimblecas.dae`](dae.md), and
  [`nimblecas.sde`](sde.md) — the delay, differential-algebraic, and stochastic
  differential-equation siblings in the exact-solver family.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field and
  `RationalPoly` ring every coefficient, datum, and evaluation point lives in.
- [Documentation hub](../Index.md)
