# `nimblecas.pdeham` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/pdeham/pdeham.cppm`

The **Homotopy Analysis Method** (HAM, Liao) with the **convergence-control
parameter `ħ`**, for the semilinear nonlinear evolution PDE

```
u_t = F[u],   u(x, 0) = u_0(x),
F[u] = ν u_xx + c · u u_x + Σ_p a_p u^p     (ν = diffusivity, c = convection, a_p = reaction).
```

This is the `ħ`-deformation counterpart of the ADM/HPM forward solvers in
[`pde`](pde.md): those build the single Cauchy-Kovalevskaya / Adomian **Taylor
series in `t`**; HAM embeds the problem in a one-parameter homotopy carrying `ħ`,
an extra degree of freedom the Taylor construction does not have. It fills the
gap [`pde`](pde.md) explicitly leaves open in its header ("does **not** (yet) offer
a HAM variant … generalising the `ħ`-parameterised deformation from a single
`Q[[x]]` series to the two-index `(Q[x])[[t]]` ring needs care").

```cpp
import nimblecas.pdeham;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md) (the exact `Rational` `ħ`
and PDE coefficients), [`symbolic`](symbolic.md) (`Expr`), [`simplify`](simplify.md)
and [`expand`](expand.md) (exact canonical folding and distribution of polynomial
products in `x`), and [`diff`](diff.md) (exact spatial `x`-derivatives).

## The exact PDE class handled

```cpp
struct HamPde {
    Rational diffusivity{};                  // ν, coefficient of u_xx
    Rational convection{};                   // c, coefficient of the convective term u u_x
    std::vector<Rational> reaction_coeffs{}; // f(u) = Σ_p reaction_coeffs[p] u^p
};
```

The struct **is** the operator class: `F[u] = ν u_xx + c · u u_x + Σ_p a_p u^p`. It
covers reaction–diffusion (`ν ≠ 0`, `f` polynomial — Fisher/KPP-type sources),
pure reaction (`ν = 0`), Burgers/advection (`c ≠ 0`, e.g. `c = −1` for
`u_t + u u_x = ν u_xx`), and any combination. An operator **outside** this class
(a different linear part, a non-polynomial nonlinearity, mixed or higher spatial
derivatives) is not expressible as a `HamPde` and is therefore **not** silently
mishandled; a degenerate empty description (no diffusion, convection, or reaction)
returns `MathError::not_implemented`.

## The `ħ`-deformation

With the auxiliary linear operator `L = ∂_t`, embedding parameter `q ∈ [0,1]`, and
auxiliary function `H ≡ 1`, the zeroth-order deformation is

```
(1 − q) L[φ(x,t;q) − u_0] = q ħ 𝒩[φ],   𝒩[φ] = φ_t − F[φ].
```

Expanding `φ = u_0 + Σ_{m≥1} u_m q^m` and matching powers of `q` gives the m-th
order deformation equation

```
L[u_m − χ_m u_{m−1}] = ħ R_m,   R_m = (u_{m−1})_t − A_{m−1},   χ_m = [m ≥ 2],
```

where `A_{m−1} = [q^{m−1}] F[φ]` is the `(m−1)`-th Adomian / `q`-graded polynomial
of `F`. Because `L = ∂_t`, its exact right inverse is `L⁻¹ = ∫_0^t (·) dτ` (zero
datum, since every `u_j` with `j ≥ 1` vanishes at `t = 0`), so each order is one
exact integration in `t`:

```
u_m = χ_m u_{m−1} + ħ ∫_0^t R_m dτ,     u ≈ Σ_{m=0}^{order} u_m.
```

### The two gradings (`q` versus `t`)

Each component `u_m(x,t)` is a full truncated **time series** — a vector of
`x`-polynomials, one per power of `t` — so `φ` is a series in *both* `q` and `t`.
`A_{m−1} = [q^{m−1}] F[φ]` is obtained by an exact `q`-convolution of the
components `u_0 … u_{m−1}` (each a `t`-series), with spatial `x`-derivatives taken
by [`diff`](diff.md) and `t`-products by the truncated Cauchy product in the
`(Q[x])[[t]]` ring (via [`expand`](expand.md) for exact polynomial multiplication
in `x`). Because `F` only differentiates in `x` (`q`-degree preserving) and forms
products (`q`-degree raising), `[q^{m−1}] F` depends solely on `u_0 … u_{m−1}`: the
recurrence is **causal** and exact.

### `ħ = −1` recovers the forward (ADM/HPM) series

At `ħ = −1` the `(u_{m−1})_t` term integrates back to `χ_m u_{m−1}` and cancels it,
collapsing the deformation to `u_m = ∫_0^t A_{m−1} dτ` — the ADM recursion, hence
the Cauchy-Kovalevskaya Taylor series that [`pde`](pde.md) computes. At `ħ = −1`
every component is homogeneous of `t`-degree `m`, so `Σ_{m=0}^{order} u_m`
reproduces the forward coefficients `c_0 … c_order` **term for term** (the
`pdeham_tests` reduction test asserts this against
`reaction_diffusion_quadratic` and `burgers`).

## API

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `ham_pde_solve` | `auto ham_pde_solve(const HamPde& pde, const Expr& u0, Rational hbar, std::size_t order, std::string_view tvar, std::string_view xvar) -> Result<Expr>` | Truncated HAM approximation `u ≈ Σ_{m=0}^{order} u_m` as an `Expr` polynomial in `xvar`, `tvar`, truncated at `t^{order}`. `hbar = −1` recovers the forward series. |
| `ham_reaction_diffusion` | `auto ham_reaction_diffusion(Rational diffusivity, std::vector<Rational> reaction_coeffs, const Expr& u0, Rational hbar, std::size_t order, std::string_view tvar, std::string_view xvar) -> Result<Expr>` | Convenience wrapper for `u_t = ν u_xx + Σ_p a_p u^p` (`convection = 0`). |
| `ham_burgers` | `auto ham_burgers(Rational viscosity, const Expr& u0, Rational hbar, std::size_t order, std::string_view tvar, std::string_view xvar) -> Result<Expr>` | Convenience wrapper for `u_t + u u_x = ν u_xx` (i.e. `convection = −1`). |

`u0` is the initial datum, an `Expr` in `xvar` (a polynomial in `x` gives an exact
terminating spatial degree; any `Expr` is accepted and handled exactly
symbolically).

### Error model

| Condition | Error |
| :--- | :--- |
| `order == 0` | `MathError::domain_error` |
| `tvar == xvar` | `MathError::domain_error` |
| `order` beyond `INT64_MAX` (unreachable) | `MathError::overflow` |
| empty operator (no `ν`, `c`, or reaction) | `MathError::not_implemented` |
| any `simplify`/`expand`/`diff`/`Rational` failure (overflow, undefined form) | propagated verbatim |

## Honesty boundary

The result is an **exact symbolic truncated series in `t`**: a polynomial
`Σ_{k=0}^{order} c_k(x) t^k` with each `c_k` exact over `Q`. It is exact
term-by-term to the stated order under the HAM deformation, with truncation error
`O(t^{order+1})` — it is **not** a closed-form solution, and no claim is made about
the radius of convergence. Two HAM-specific caveats:

1. **Only `ħ = −1` yields the Taylor polynomial.** At `ħ = −1` the truncated series
   is the Cauchy-Kovalevskaya Taylor polynomial, whose PDE residual
   `u_t − F[u]` is `O(t^{order})` (its `t^0 … t^{order−1}` coefficients vanish
   exactly — the `pdeham_tests` residual test verifies this by differentiating the
   series back with [`diff`](diff.md) and substituting). A **general `ħ`** selects a
   different, still exactly rational, member of the deformation family; its
   lower-order `t`-coefficients are *not* the Taylor coefficients — that is the
   intended convergence-control freedom (choosing an optimal `ħ` accelerates
   convergence of the full series), not an approximation of the Taylor series.

2. **HAM-order and `t`-order truncations differ for `ħ ≠ −1`.** Higher HAM orders
   correct lower `t`-coefficients, so truncating at HAM order `m` and at `t`-order
   `m` are different truncations unless `ħ = −1`. The `order-1` and `order-2`
   results at `ħ = −1/2` differ in their `t^1` coefficient (`1/2 x^2` vs `3/4 x^2`)
   for exactly this reason.

## Worked example

```cpp
// u_t = u^2, u(x,0) = x, HAM order 2 at ħ = -1/2 (pure reaction, ν = 0).
auto u = nimblecas::ham_reaction_diffusion(
    Rational::from_int(0),                                   // ν = 0
    {Rational::from_int(0), Rational::from_int(0), Rational::from_int(1)},  // f(u) = u^2
    Expr::symbol("x"),                                       // u0 = x
    Rational::make(-1, 2).value(),                           // ħ = -1/2
    2, "t", "x").value();
// u  ==  x + (3/4) x^2 t + (1/4) x^3 t^2   (exact rational HAM member).

// The same PDE at ħ = -1 reproduces the forward Taylor series x + x^2 t + x^3 t^2
// (== nimblecas.pde reaction_diffusion_quadratic with ν = 0), i.e. x/(1 - x t) to O(t^3).
```
