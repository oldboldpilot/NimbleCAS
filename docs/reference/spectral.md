# `nimblecas.spectral` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/spectral/spectral.cppm`

**Spectral discretisations** — global (and per-element) expansions in orthogonal
bases with the associated differentiation and boundary-value solvers. This module
is deliberately distinct from `nimblecas.pde` (exact power-series time evolution,
no boundary conditions) and `nimblecas.pdenum` (low-order FDM/FEM/FVM stencils):
everything here is a Galerkin, collocation, Fourier, spectral-element, or DG
construction.

The honesty boundary is documented per function and split cleanly in two.
**Exact over `Q`** (rational arithmetic, no rounding) for *polynomial data on
rational meshes*, because every inner product is an exact rational integral over
`[-1, 1]` and every orthogonal polynomial has exact rational coefficients: the
Legendre / Chebyshev-T forward and inverse transforms, the coefficient-space
differentiations, the Legendre-Galerkin Poisson solve, the spectral-element
projection / reconstruction, and the discontinuous-Galerkin *spatial operator*
(and a single explicit step). **Numerical** (double precision — irrational nodes
and/or complex exponentials) for the Chebyshev-Gauss-Lobatto collocation family
and the Fourier family: any spectral-accuracy or convergence claim about these is
a *numerical* property, not exactness. A separate **time-marching caveat**
applies to DG: each individual forward-Euler step is an exact rational update, but
marching many steps to a final time is first-order numerical time integration of
the continuous evolution — we do not claim the time-marched result is the exact
PDE solution.

```cpp
import nimblecas.spectral;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md),
[`orthopoly`](orthopoly.md), and [`matrix`](matrix.md).

## The overflow contract

Following the rest of the engine, every exact-over-`Q` path is
**overflow-checked** (Rule 32): an `int64` boundary in any `Rational` or
`RationalPoly` operation propagates as `MathError::overflow` rather than wrapping.
The numerical (double) paths do not overflow-check — they carry the usual
floating-point rounding, and their `domain_error` returns guard the *shape* of
the input (grid size, matrix conformance, mesh validity), not conditioning.

## Exact utility

```cpp
[[nodiscard]] auto evaluate(const RationalPoly& p, const Rational& x) -> Result<Rational>;
```

| Function | Behavior |
| :--- | :--- |
| `evaluate` | Evaluate a `RationalPoly` at a rational point by Horner's scheme; exact over `Q`. The zero polynomial evaluates to `0`. Fails only on `Rational` overflow. |

## Exact spectral-Galerkin over `Q`

All coefficient vectors are low-degree/low-mode first (`c[k]` multiplies `P_k` or
`T_k`). Forward transforms return `c_0 .. c_{deg f}` — higher coefficients are
exactly zero for polynomial data — and the zero polynomial yields an empty
vector.

```cpp
[[nodiscard]] auto legendre_coefficients(const RationalPoly& f) -> Result<std::vector<Rational>>;
[[nodiscard]] auto legendre_from_coefficients(std::span<const Rational> coeffs)
    -> Result<RationalPoly>;
[[nodiscard]] auto chebyshev_coefficients(const RationalPoly& f) -> Result<std::vector<Rational>>;
[[nodiscard]] auto chebyshev_from_coefficients(std::span<const Rational> coeffs)
    -> Result<RationalPoly>;
[[nodiscard]] auto legendre_differentiate_coefficients(std::span<const Rational> coeffs)
    -> Result<std::vector<Rational>>;
[[nodiscard]] auto chebyshev_differentiate_coefficients(std::span<const Rational> coeffs)
    -> Result<std::vector<Rational>>;
[[nodiscard]] auto galerkin_poisson(const RationalPoly& f) -> Result<RationalPoly>;
```

| Function | Behavior |
| :--- | :--- |
| `legendre_coefficients` | Forward Legendre transform: `c_k = <f, P_k> / <P_k, P_k>` with `<g, h> = ∫_{-1}^{1} g h dx`, so `f = Σ_k c_k P_k`. **Exact.** |
| `legendre_from_coefficients` | Inverse Legendre transform: reconstruct `Σ_k coeffs[k] P_k(x)` exactly. **Exact.** |
| `chebyshev_coefficients` | Forward Chebyshev-T transform under the weight `w(x) = 1/√(1 − x²)`: `c_k = <f, T_k>_w / <T_k, T_k>_w`. The shared `π` cancels between numerator and denominator, so every `c_k` is an exact rational. **Exact.** |
| `chebyshev_from_coefficients` | Inverse Chebyshev-T transform: reconstruct `Σ_k coeffs[k] T_k(x)` exactly. **Exact.** |
| `legendre_differentiate_coefficients` | Spectral differentiation in Legendre coefficient space: given `a = {a_k}` with `f = Σ a_k P_k`, returns `b = {b_k}` with `f' = Σ b_k P_k` via `b_k = (2k+1) Σ_{p>k, p+k odd} a_p`. Trailing zeros trimmed. **Exact.** |
| `chebyshev_differentiate_coefficients` | Spectral differentiation in Chebyshev-T coefficient space: `b_k = (2 / c_k) Σ_{p>k, p+k odd} p·a_p`, with `c_0 = 2` and `c_k = 1` for `k ≥ 1`. Trailing zeros trimmed. **Exact.** |
| `galerkin_poisson` | Legendre spectral-Galerkin solution of `−u''(x) = f(x)` on `(−1, 1)`, `u(±1) = 0`, for polynomial `f`. Uses the homogeneous-Dirichlet modal basis `φ_k = P_k − P_{k+2}` (each vanishes at `x = ±1`), assembles the exact rational stiffness `S_{jk} = ∫ φ_j' φ_k' dx` and load `F_j = ∫ f φ_j dx`, and solves `S u = F` exactly over `Q`. For polynomial `f` the exact solution lies in the chosen span, so the returned `RationalPoly` is the exact solution; `f == 0` returns the zero polynomial. **Exact.** |

## Numerical Chebyshev collocation / pseudospectral

Double precision — the Gauss-Lobatto nodes `x_j = cos(π j / n)` are **irrational**,
so spectral accuracy here is a numerical property.

```cpp
[[nodiscard]] auto chebyshev_gauss_lobatto_nodes(std::size_t n) -> Result<std::vector<double>>;
[[nodiscard]] auto chebyshev_differentiation_matrix(std::size_t n)
    -> Result<std::vector<std::vector<double>>>;
[[nodiscard]] auto apply_dense_matrix(const std::vector<std::vector<double>>& a,
                                      std::span<const double> v) -> Result<std::vector<double>>;
[[nodiscard]] auto chebyshev_collocation_poisson(const std::function<double(double)>& f,
                                                 std::size_t n) -> Result<std::vector<double>>;
```

| Function | Behavior |
| :--- | :--- |
| `chebyshev_gauss_lobatto_nodes` | The `n+1` nodes `x_j = cos(π j / n)`, `j = 0..n` (`x_0 = 1` down to `x_n = −1`). Requires `n ≥ 1`. **Numerical.** |
| `chebyshev_differentiation_matrix` | The `(n+1) × (n+1)` collocation differentiation matrix `D` at the Gauss-Lobatto nodes: `(D v)_i ≈ u'(x_i)` when `v_j = u(x_j)`. Off-diagonals use the classical formula; the diagonal is set by the negative-sum trick `D_ii = −Σ_{k≠i} D_ik`. Requires `n ≥ 1`. **Numerical.** |
| `apply_dense_matrix` | Apply a dense row-major matrix `a` (each inner vector a row) to a vector `v`. Fails `domain_error` on a ragged matrix or a column/length mismatch. **Numerical.** |
| `chebyshev_collocation_poisson` | Pseudospectral solve of `−u'' = f` on `(−1, 1)`, `u(±1) = 0`, by imposing the ODE at the interior collocation nodes and homogeneous Dirichlet conditions at the two boundary nodes, then solving the dense double system. Returns `u` at all `n+1` nodes (boundary entries exactly `0`). Requires `n ≥ 2`. **Numerical** — its spectral accuracy is a numerical property. |

## Numerical Fourier spectral

Double / complex exponentials on the equispaced periodic grid. The transforms use
a compact `O(N²)` DFT — a fast `O(N log N)` FFT would compute the same result
asymptotically faster; the honest `O(N²)` form is implemented for clarity.

```cpp
[[nodiscard]] auto fourier_grid(std::size_t n) -> Result<std::vector<double>>;
[[nodiscard]] auto fourier_differentiation_matrix(std::size_t n)
    -> Result<std::vector<std::vector<double>>>;
[[nodiscard]] auto fourier_spectral_derivative(std::span<const double> samples)
    -> Result<std::vector<double>>;
[[nodiscard]] auto fourier_periodic_solve(std::span<const double> f_samples)
    -> Result<std::vector<double>>;
```

| Function | Behavior |
| :--- | :--- |
| `fourier_grid` | The `n` equispaced periodic grid points `x_j = 2π j / n` on `[0, 2π)`, `j = 0..n−1`. Requires `n ≥ 1`. **Numerical.** |
| `fourier_differentiation_matrix` | The `n × n` differentiation matrix for **even** `n`: `D_jk = ½ (−1)^{j−k} cot(π (j−k) / n)` for `j ≠ k`, `D_jj = 0`. Requires `n` even and `n ≥ 2` (odd `n` has a different closed form and is rejected with `domain_error`). **Numerical.** |
| `fourier_spectral_derivative` | Spectral derivative of a periodic sampled function via a direct DFT: transform, multiply mode `k` by `i·k'` (`k'` the signed wavenumber, Nyquist zeroed for this odd-order derivative), invert, take the real part. Requires `n ≥ 2`. **Numerical** (compact `O(N²)` DFT). |
| `fourier_periodic_solve` | Periodic solve of the Helmholtz-type problem `u − u'' = f` on `[0, 2π)` with periodic BCs, done in Fourier space: `û_k = f̂_k / (1 + k'²)`. Always invertible (no compatibility condition). Returns `u` sampled on the same grid as `f`. Requires `n ≥ 2`. **Numerical** (`O(N²)` DFT). |

## Analogues — spectral element (`C⁰`) and discontinuous Galerkin

Per-element modal Legendre expansions on rational meshes. These are **exact over
`Q`** for rational data on a rational mesh; the only numerical property is the
DG *time marching* (see below).

```cpp
[[nodiscard]] auto spectral_element_legendre(const RationalPoly& f, std::span<const Rational> mesh)
    -> Result<std::vector<std::vector<Rational>>>;
[[nodiscard]] auto spectral_element_reconstruct(
    const std::vector<std::vector<Rational>>& elem_coeffs, std::span<const Rational> mesh)
    -> Result<std::vector<RationalPoly>>;
[[nodiscard]] auto dg_advection_rhs(const std::vector<std::vector<Rational>>& state,
                                    std::span<const Rational> mesh, const Rational& a,
                                    const Rational& inflow)
    -> Result<std::vector<std::vector<Rational>>>;
[[nodiscard]] auto dg_advection_step(const std::vector<std::vector<Rational>>& state,
                                     std::span<const Rational> mesh, const Rational& a,
                                     const Rational& inflow, const Rational& dt)
    -> Result<std::vector<std::vector<Rational>>>;
```

| Function | Behavior |
| :--- | :--- |
| `spectral_element_legendre` | 1-D spectral-element representation of a polynomial `f` on a strictly increasing rational mesh `z_0 < z_1 < ... < z_M` (`M+1` nodes, `M` elements). Each element `[z_e, z_{e+1}]` is affinely mapped to `[-1, 1]` and `f`'s restriction is expanded in the Legendre basis (exact forward transform), yielding that element's reference coefficients. Because every element carries the same underlying `f`, the piecewise representation is automatically `C⁰` (adjacent reconstructions agree at shared nodes). Requires `≥ 2` mesh nodes, strictly increasing. **Exact over `Q`.** |
| `spectral_element_reconstruct` | Inverse of `spectral_element_legendre`: reconstruct each element's physical-space `RationalPoly` from its reference Legendre coefficients (inverting the affine map). Requires `elem_coeffs.size() + 1 == mesh.size()` and a valid increasing mesh. **Exact over `Q`.** |
| `dg_advection_rhs` | 1-D discontinuous-Galerkin semi-discrete spatial operator for `u_t + a u_x = 0` (`a > 0`), per-element modal Legendre basis on `[-1, 1]` with the **upwind** flux. `state[e]` holds element `e`'s reference coefficients (all elements share the same degree `p-1`); `mesh` the `M+1` break points, `a` the rational speed, `inflow` the prescribed value at the left inflow boundary. Returns `d(state)/dt` element by element: `dU^e_n/dt = [ a Σ_m U^e_m S_{nm} − (F̂_R P_n(1) − F̂_L P_n(−1)) ] · (2n+1) / h_e`, with `S_{nm} = ∫_{-1}^{1} P_m P_n' dξ`, `F̂` the upwind interface fluxes, `h_e` the element length. Requires `≥ 1` element, a matching mesh, and a uniform non-empty coefficient count; assumes `a > 0`. **Exact over `Q`.** |
| `dg_advection_step` | One explicit (forward) Euler step: `state + dt · dg_advection_rhs(state, ...)`. Each step is an **exact** rational update. **Exact per step** — but repeated marching is a first-order *numerical* time integration of the continuous evolution (see the time-marching caveat above); the time-marched result is not claimed to be the exact PDE solution. |

## Error model

| Condition | Error |
| :--- | :--- |
| `chebyshev_gauss_lobatto_nodes` / `chebyshev_differentiation_matrix` with `n < 1` | `MathError::domain_error` |
| `chebyshev_collocation_poisson` with `n < 2` (no interior nodes) | `MathError::domain_error` |
| `apply_dense_matrix` on a ragged matrix or column/length mismatch | `MathError::domain_error` |
| `fourier_grid` with `n < 1` | `MathError::domain_error` |
| `fourier_differentiation_matrix` with `n` odd or `n < 2` | `MathError::domain_error` |
| `fourier_spectral_derivative` / `fourier_periodic_solve` with `n < 2` | `MathError::domain_error` |
| Mesh with `< 2` nodes, or not strictly increasing (spectral-element / DG) | `MathError::domain_error` |
| `spectral_element_reconstruct` with `elem_coeffs.size() + 1 != mesh.size()` | `MathError::domain_error` |
| `dg_advection_rhs` with no elements, a mismatched mesh, or a non-uniform / empty coefficient count | `MathError::domain_error` |
| The interior pseudospectral system is (near-)singular | `MathError::domain_error` |
| An `int64` numerator or denominator computation wraps (exact paths) | `MathError::overflow` |

The zero polynomial is a total input on every exact transform: `legendre_coefficients`
/ `chebyshev_coefficients` return an empty vector, `galerkin_poisson` returns the
zero polynomial, and the coefficient-space derivatives of a constant or the zero
polynomial return an empty vector.

## Worked examples

```cpp
import nimblecas.spectral;
import nimblecas.ratpoly;
using namespace nimblecas;

auto rat   = [](std::int64_t n, std::int64_t d) { return Rational::make(n, d).value(); };
auto rpoly = [](std::vector<Rational> c) { return RationalPoly::from_coeffs(std::move(c)); };

// --- EXACT spectral-Galerkin over Q -------------------------------------
// x^2 = (1/3) P_0 + (2/3) P_2.
legendre_coefficients(rpoly({rat(0,1), rat(0,1), rat(1,1)})).value();
    // {1/3, 0, 2/3}

// Forward then inverse round-trips exactly: f = 3 - x + 2x^2.
const RationalPoly f = rpoly({rat(3,1), rat(-1,1), rat(2,1)});
legendre_from_coefficients(legendre_coefficients(f).value()).value().is_equal(f);  // true

// x^3 = (3/4) T_1 + (1/4) T_3 (Chebyshev-T, exact rationals).
chebyshev_coefficients(rpoly({rat(0,1), rat(0,1), rat(0,1), rat(1,1)})).value();
    // {0, 3/4, 0, 1/4}

// Coefficient-space differentiation: d/dx (x^3) = 3x^2, via Legendre modes.
const RationalPoly cube = rpoly({rat(0,1), rat(0,1), rat(0,1), rat(1,1)});
const auto lc  = legendre_coefficients(cube).value();
const auto ldc = legendre_differentiate_coefficients(lc).value();
legendre_from_coefficients(ldc).value().is_equal(rpoly({rat(0,1), rat(0,1), rat(3,1)}));  // true

// Legendre-Galerkin Poisson: -u'' = 2, u(±1)=0  ->  u = 1 - x^2 (exact).
galerkin_poisson(RationalPoly::constant(Rational::from_int(2))).value()
    .is_equal(rpoly({rat(1,1), rat(0,1), rat(-1,1)}));  // true
// -u'' = x^2, u(±1)=0  ->  u = (1 - x^4)/12.
galerkin_poisson(rpoly({rat(0,1), rat(0,1), rat(1,1)})).value()
    .is_equal(rpoly({rat(1,12), rat(0,1), rat(0,1), rat(0,1), rat(-1,12)}));  // true

// --- NUMERICAL Chebyshev collocation (double) ---------------------------
const std::size_t n = 8;
const auto nodes = chebyshev_gauss_lobatto_nodes(n).value();
const auto D     = chebyshev_differentiation_matrix(n).value();
std::vector<double> v(n + 1);
for (std::size_t j = 0; j <= n; ++j) v[j] = nodes[j] * nodes[j] * nodes[j];
const auto dv = apply_dense_matrix(D, v).value();   // dv[j] ≈ 3 * nodes[j]^2
// -u'' = 2, u(±1)=0 -> u ≈ 1 - x^2 at the nodes.
chebyshev_collocation_poisson([](double) { return 2.0; }, n).value();

// --- NUMERICAL Fourier spectral (double / complex) ----------------------
const std::size_t m = 16;
const auto grid = fourier_grid(m).value();
std::vector<double> u(m);
for (std::size_t j = 0; j < m; ++j) u[j] = std::sin(grid[j]);
fourier_spectral_derivative(u).value();             // ≈ cos(grid[j])
// u - u'' = 2 sin(x)  ->  u = sin(x).
std::vector<double> rhs(m);
for (std::size_t j = 0; j < m; ++j) rhs[j] = 2.0 * std::sin(grid[j]);
fourier_periodic_solve(rhs).value();                // ≈ sin(grid[j])

// --- ANALOGUES: spectral element (exact, C^0) ---------------------------
// f = -2x + x^3 on a 2-element mesh reproduces itself on each element.
const RationalPoly fe = rpoly({rat(0,1), rat(-2,1), rat(0,1), rat(1,1)});
const std::vector<Rational> mesh = {rat(-1,1), rat(0,1), rat(1,1)};
const auto elem  = spectral_element_legendre(fe, mesh).value();      // 2 elements
const auto recon = spectral_element_reconstruct(elem, mesh).value();
recon[0].is_equal(fe);                              // true
evaluate(recon[0], rat(0,1)).value() == evaluate(recon[1], rat(0,1)).value();  // C^0

// --- ANALOGUES: discontinuous Galerkin (exact operator over Q) ----------
// Linear field u(x)=x, a=1, inflow=0: du/dt = -a u_x = -1 per element.
const std::vector<std::vector<Rational>> state = {{rat(1,2), rat(1,2)},
                                                  {rat(3,2), rat(1,2)}};
const std::vector<Rational> dgmesh = {rat(0,1), rat(1,1), rat(2,1)};
const auto drhs = dg_advection_rhs(state, dgmesh, Rational::from_int(1),
                                   Rational::from_int(0)).value();
drhs[0][0] == rat(-1,1) && drhs[0][1] == rat(0,1);  // true (exact)
// One exact rational Euler step of a constant field leaves it unchanged.
const std::vector<std::vector<Rational>> cst = {{rat(5,1)}, {rat(5,1)}};
dg_advection_step(cst, dgmesh, Rational::from_int(1), Rational::from_int(5),
                  rat(1,10)).value();               // unchanged: {{5}, {5}}

// --- domain errors ------------------------------------------------------
chebyshev_gauss_lobatto_nodes(0).error();           // MathError::domain_error
fourier_differentiation_matrix(3).error();          // MathError::domain_error (odd n)
```

## See also

- [`nimblecas.orthopoly`](orthopoly.md) — the exact Legendre / Chebyshev families
  whose recurrences supply the modal bases and inner products used here.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Q[x]` substrate (`Rational`,
  `RationalPoly`) every exact transform is expressed over.
- [`nimblecas.matrix`](matrix.md) — the exact rational linear solver behind the
  Galerkin stiffness system.
- [`nimblecas.complex`](complex.md) — the exact Gaussian-rational field; the
  Fourier paths here instead use numerical `std::complex<double>`.
- [Documentation hub](../Index.md)
