# `nimblecas.inteq` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/inteq/inteq.cppm`

Exact solvers for linear and nonlinear **integral equations** — Fredholm and
Volterra of the first and second kind — over the rationals `Q`. The exact-over-`Q`
engine works on polynomials in `Q[x]` ([`ratpoly`](ratpoly.md)): the unknown `phi`,
the free term `f`, and the **separable** (degenerate) kernel
`K(x,t) = Σ_i g_i(x) h_i(t)` are all `RationalPoly`, and every kernel action reduces
to elementary polynomial integrals evaluated through
[`integrate`](integrate.md)'s `integrate_rational` at rational limits. Under those
hypotheses the results are the **exact** rational solution (the linear-algebra
reduction) or the **exact** truncated Neumann / Picard / Adomian series — no floating
point, no fabricated closed form.

The **honesty boundary** is sharp. A separable Fredholm-2 equation reduces to a
finite linear system solved exactly over `Q` — that solution is *complete and exact*.
Everything else that does not terminate is a **truncated series**: a local,
finite-order approximation of the true solution unless it happens to terminate (e.g. a
nilpotent operator). An integrand whose antiderivative is not a plain polynomial — a
logarithmic part or a genuine rational-function remainder — returns
`MathError::not_implemented` rather than a fabricated closed form; likewise a Laplace
pair off the table. Fredholm-1 is **ill-posed**: only the separable moments are
recoverable, never `phi` itself. Every operation is total and railway-oriented
(Rule 32): no exceptions, `Result<T>` throughout.

```cpp
import nimblecas.inteq;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), [`matrix`](matrix.md),
[`integrate`](integrate.md), [`symbolic`](symbolic.md), [`laplace`](laplace.md), and
[`simplify`](simplify.md).

## The exact/approximate boundary

Two very different guarantees live in this module, and the API keeps them apart:

- **Exact and complete.** `fredholm2_separable` builds the `r×r` moment system
  `(I − lambda M) c = d` over `Q` and solves it with [`matrix`](matrix.md); the
  returned `phi = f + lambda Σ c_i g_i` is the *exact* solution, not a truncation.
  `fredholm1_separable` recovers the separable moments exactly over `Q` (but not
  `phi`). These are closed, finite computations.

- **Exact per term, truncated overall.** The Neumann (`fredholm2_neumann`), Picard
  (`volterra2_picard`, `volterra_nonlinear_picard`, `hammerstein_picard`), and
  decomposition/homotopy series (`adm_solve`, `hpm_solve`, `ham_solve`) compute each
  term exactly over `Q`, but the returned polynomial is a **finite-order partial sum**.
  It equals the true solution only where the series terminates. Fredholm–Neumann
  converges to the true `phi` only when `|lambda|·‖𝒦‖ < 1`; Volterra–Picard always
  converges; a definite (Fredholm) Picard iterate need not converge at all and is
  returned honestly as a finite iterate.

A third category stays in the transform domain: `volterra_convolution_laplace` returns
the **`s`-domain** solution `Φ(s)` — no general inverse transform is attempted.

## Cross-checks the series family satisfies

The decomposition/homotopy solvers are deliberately redundant, and the redundancy is a
correctness contract exercised in the tests:

- For a **linear** nonlinearity (`Nonlinearity::identity()`) the Adomian polynomials
  reduce to `A_n == phi_n`, so **ADM == the Neumann / Picard series**.
- **ADM == HPM** — the homotopy-perturbation homotopy collapses to the *same* graded
  recursion, so `hpm_solve` returns the identical series to `adm_solve` by construction.
- **HAM(ħ = −1) == ADM** — the convergence-control parameter `ħ` recovers ADM/HPM
  exactly at `ħ = −1`; other `ħ` give a different, still-exactly-rational family member.

## The overflow contract

Following the rest of the engine, arithmetic is **exact** and **overflow-checked**
(Rule 32): every kernel action, moment integral, and series term flows through the
checked `Rational` / `RationalPoly` operations of [`ratpoly`](ratpoly.md), so an
`int64` numerator or denominator that would overflow surfaces as
`MathError::overflow` rather than silently wrapping. A non-polynomial antiderivative
(a logarithmic part or a non-zero rational remainder) surfaces as
`MathError::not_implemented`; a singular moment system or a ragged kernel is
`MathError::domain_error`.

## `Nonlinearity` — an autonomous polynomial nonlinearity `N(phi) = Σ_j c_j phi^j`

The nonlinear reaction term for Hammerstein / nonlinear-Volterra equations, stored as
rational coefficients ascending in the power of `phi`. `N =` identity models the
**linear** equation; `N = phi^p` is the classic Hammerstein case. A `t`-dependent
Hammerstein weight `a(t)` folds into the kernel factor `h_i` (since
`∫ K a(t) M(phi) = ∫ [K a] M(phi)`), so this autonomous form covers the separable
nonlinear equations exactly over `Q`.

### Construction

| Factory | Signature | Notes |
| :--- | :--- | :--- |
| `polynomial` | `static auto polynomial(std::vector<Rational> coeffs) -> Nonlinearity` | `N(phi) = Σ_j coeffs[j] phi^j` (`coeffs` ascending in the power of `phi`). |
| `identity` | `static auto identity() -> Nonlinearity` | `N(phi) = phi` (coeffs `{0, 1}`) — the linear case. |
| `power` | `static auto power(std::size_t p) -> Nonlinearity` | `N(phi) = phi^p` (`p >= 1`; `p == 0` gives the constant `1`). |

### Accessors and application

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `coefficients` | `auto coefficients() const noexcept -> std::span<const Rational>` | The coefficients, ascending in the power of `phi`. |
| `is_linear` | `auto is_linear() const noexcept -> bool` | `true` iff `N` is exactly the identity `{0, 1}` — the case where the decomposition series reduce to Neumann/Picard (`A_n == phi_n`). |
| `apply` | `auto apply(const RationalPoly& phi) const -> Result<RationalPoly>` | Pointwise `N(phi)` as a polynomial (Horner over `Q[x]`). Exact for any polynomial `phi`; propagates `overflow`. |

## `SeparableKernel` — the degenerate kernel `K(x,t) = Σ_i g_i(x) h_i(t)`

```cpp
struct SeparableKernel {
    std::vector<RationalPoly> g;  // g_i(x)
    std::vector<RationalPoly> h;  // h_i(t)

    [[nodiscard]] auto rank() const noexcept -> std::size_t;   // g.size()
    [[nodiscard]] auto is_valid() const noexcept -> bool;      // g.size() == h.size()
};
```

The `g` and `h` lists must be the same length (`is_valid()`); a ragged kernel is a
`domain_error` at every entry point that consumes it. `rank()` is the number of
separable terms `r`.

## `IntegralOperator` — the linear action `u -> lambda ∫ K(x,t) u(t) dt`

The reusable kernel operator behind every series recursion. **Fredholm** integrates
over the fixed interval `[a,b]` (`∫_a^b h_i u` is a constant, so the action is
`lambda Σ_i g_i(x) · const`); **Volterra** integrates to a variable upper limit
(`∫_a^x h_i(t) u(t) dt` is a polynomial in `x`). `apply` **folds `lambda` in**, so the
object it computes is the full `lambda·𝒦`.

### Construction (factories)

| Factory | Signature |
| :--- | :--- |
| `fredholm` | `static auto fredholm(SeparableKernel kernel, Rational lambda, Rational a, Rational b) -> Result<IntegralOperator>` |
| `volterra` | `static auto volterra(SeparableKernel kernel, Rational lambda, Rational a) -> Result<IntegralOperator>` |

Both return `MathError::domain_error` when the kernel is ragged.

### Accessors and application

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `apply` | `auto apply(const RationalPoly& u) const -> Result<RationalPoly>` | `(lambda 𝒦 u)(x)`. `not_implemented` when an integrand has no polynomial antiderivative; propagates `overflow`. |
| `lambda` | `auto lambda() const noexcept -> Rational` | The folded-in `lambda`. |
| `is_volterra` | `auto is_volterra() const noexcept -> bool` | `true` for a Volterra operator (variable upper limit). |
| `kernel` | `auto kernel() const noexcept -> const SeparableKernel&` | The underlying separable kernel. |

## Solution payloads

```cpp
// phi(x) = f(x) + lambda Σ_i coefficients[i] g_i(x), with
// coefficients[i] = c_i = ∫_a^b h_i(t) phi(t) dt the solved moment constants.
struct FredholmSolution {
    RationalPoly phi;                    // the exact rational solution
    std::vector<Rational> coefficients;  // c_i (the separable moments)
};

// The recoverable information from an (ill-posed) Fredholm-1 separable equation:
// the moments c_i = ∫_a^b h_i(t) phi(t) dt determined by f ∈ span{g_i}.
// phi ITSELF IS NON-UNIQUE.
struct FirstKindSolution {
    std::vector<Rational> moments;  // c_i ; phi is NOT determined (regularization required)
};
```

## Fredholm 2nd kind — `phi(x) = f(x) + lambda ∫_a^b K(x,t) phi(t) dt`

```cpp
[[nodiscard]] auto fredholm2_separable(const RationalPoly& f, const SeparableKernel& kernel,
                                       const Rational& lambda, const Rational& a,
                                       const Rational& b) -> Result<FredholmSolution>;

[[nodiscard]] auto fredholm2_neumann(const RationalPoly& f, const SeparableKernel& kernel,
                                     const Rational& lambda, const Rational& a,
                                     const Rational& b, std::size_t order)
    -> Result<RationalPoly>;
```

| Function | Behavior |
| :--- | :--- |
| `fredholm2_separable` | **Exact.** Builds `M_ij = ∫_a^b h_i g_j` and `d_i = ∫_a^b h_i f`, solves the `r×r` system `(I − lambda M) c = d` over `Q` with [`matrix`](matrix.md), and returns `phi = f + lambda Σ c_i g_i`. An empty kernel returns `phi == f`. |
| `fredholm2_neumann` | **Truncated.** The Neumann (resolvent) series `Σ_{n=0}^{order} (lambda 𝒦)^n f` — exact term by term, a local finite-order approximation. Terminates only for a nilpotent operator; converges to the true `phi` when `|lambda|·‖𝒦‖ < 1`. |

`fredholm2_separable` fails `domain_error` when the kernel is ragged or when
`(I − lambda M)` is singular (`lambda` an eigenvalue), and `not_implemented` when a
moment integral is non-polynomial.

## Fredholm 1st kind (ill-posed) — `f(x) = ∫_a^b K(x,t) phi(t) dt`

```cpp
[[nodiscard]] auto fredholm1_separable(const RationalPoly& f, const SeparableKernel& kernel,
                                       const Rational& a, const Rational& b)
    -> Result<FirstKindSolution>;
```

Separable-kernel case: recover the moments `c_i` from `f = Σ_i c_i g_i` **exactly over
`Q`** (by equating polynomial coefficients through the exact normal equations, then
verifying consistency). `phi` is **not reconstructed** — the equation is ill-posed and
admits infinitely many solutions; only the moments are determined. The interval limits
`a, b` do not enter this recovery. Fails `domain_error` when `f ∉ span{g_i}` (no
solution), and `not_implemented` when the `g_i` are linearly dependent (the recoverable
moments are themselves non-unique).

## Volterra 2nd kind — `phi(x) = f(x) + lambda ∫_a^x K(x,t) phi(t) dt`

```cpp
[[nodiscard]] auto volterra2_picard(const RationalPoly& f, const SeparableKernel& kernel,
                                    const Rational& lambda, const Rational& a, std::size_t order)
    -> Result<RationalPoly>;

[[nodiscard]] auto volterra_convolution_laplace(const Expr& f, const Expr& kernel,
                                                const Rational& lambda, std::string_view var,
                                                std::string_view s) -> Result<Expr>;
```

| Function | Behavior |
| :--- | :--- |
| `volterra2_picard` | Linear Picard successive approximation to `order`; for the linear equation the Picard iterate equals the truncated Neumann sum `Σ_{n=0}^{order} (lambda 𝒦)^n f` (here `𝒦` is the Volterra operator `∫_a^x`), computed exactly over `Q`. Always convergent for Volterra. |
| `volterra_convolution_laplace` | **Convolution kernel** `K(x,t) = k(x−t)` via the Laplace transform: `Φ(s) = F(s)/(1 − lambda·k̂(s))`, with `F = L{f}` and `k̂ = L{kernel}` (kernel `k` supplied in the variable `var`). Returns the **transform-domain** solution `Φ(s)` simplified — no inverse transform. `not_implemented` if `f` or the kernel is off the Laplace table. |

## Nonlinear (Hammerstein / nonlinear Volterra) — Picard

```cpp
[[nodiscard]] auto volterra_nonlinear_picard(const RationalPoly& f, const SeparableKernel& kernel,
                                             const Rational& lambda, const Rational& a,
                                             const Nonlinearity& n, std::size_t iterations)
    -> Result<RationalPoly>;

[[nodiscard]] auto hammerstein_picard(const RationalPoly& f, const SeparableKernel& kernel,
                                      const Rational& lambda, const Rational& a, const Rational& b,
                                      const Nonlinearity& n, std::size_t iterations)
    -> Result<RationalPoly>;
```

| Function | Behavior |
| :--- | :--- |
| `volterra_nonlinear_picard` | Nonlinear Volterra Picard iterate after `iterations` steps: `psi_0 = f`, `psi_{k+1} = f + lambda ∫_a^x K N(psi_k)`. Exact over `Q` while `N` keeps the iterate polynomial (any polynomial `N` does). Returns the truncated iterate — a local approximation. |
| `hammerstein_picard` | Hammerstein (nonlinear Fredholm) Picard iterate: `psi_{k+1} = f + lambda ∫_a^b K N(psi_k)`. Exact over `Q` for polynomial data; returns the truncated iterate. For a definite (Fredholm) integral Picard need **not** converge — this is honestly a finite-order iterate. |

## Decomposition / homotopy series (ADM / HPM / HAM)

```cpp
[[nodiscard]] auto adomian_polynomials(const Nonlinearity& n,
                                       const std::vector<RationalPoly>& components)
    -> Result<std::vector<RationalPoly>>;

[[nodiscard]] auto adm_solve(const IntegralOperator& op, const RationalPoly& f,
                             const Nonlinearity& n, std::size_t order) -> Result<RationalPoly>;

[[nodiscard]] auto hpm_solve(const IntegralOperator& op, const RationalPoly& f,
                             const Nonlinearity& n, std::size_t order) -> Result<RationalPoly>;

[[nodiscard]] auto ham_solve(const IntegralOperator& op, const RationalPoly& f,
                             const Nonlinearity& n, const Rational& hbar, std::size_t order)
    -> Result<RationalPoly>;
```

| Function | Behavior |
| :--- | :--- |
| `adomian_polynomials` | The Adomian polynomials `A_0..A_k` of `N` for components `phi_0..phi_k` (`k = components.size() − 1`). `A_m` is the grade-`m` projection `[p^m] N(Σ_i p^i phi_i)`, computed exactly over `Q` by truncated arithmetic in the formal parameter `p`. For `N =` identity, `A_m` reduces to `phi_m`. `domain_error` on an empty component list; `overflow` propagated. |
| `adm_solve` | Adomian Decomposition to `order`: `phi_0 = f`, `phi_{n+1} = lambda 𝒦 A_n`, result `Σ_{n=0}^{order} phi_n`. `op` is the linear integral operator (Fredholm or Volterra). Exact over `Q`; truncated series. For a linear `N` this is identically the Neumann/Picard series. |
| `hpm_solve` | Homotopy Perturbation. The homotopy collapses to the **same** graded recursion as ADM, so this returns the identical series (**ADM == HPM**). |
| `ham_solve` | Homotopy Analysis with convergence-control parameter `ħ`: `phi_m = χ_m phi_{m−1} + ħ(phi_{m−1} − δ_{m,1} f − lambda 𝒦 A_{m−1})`, `χ_m = 0` at `m=1` else `1`; result `Σ_{m=0}^{order} phi_m`. `ħ = −1` recovers ADM/HPM exactly; other `ħ` give a different, still-exactly-rational family member. |

## Error model

| Condition | Error |
| :--- | :--- |
| Ragged kernel (`g.size() != h.size()`) at any entry point | `MathError::domain_error` |
| `fredholm2_separable`: `(I − lambda M)` singular (`lambda` an eigenvalue) | `MathError::domain_error` |
| `fredholm1_separable`: `f ∉ span{g_i}` (no solution) | `MathError::domain_error` |
| `fredholm1_separable`: `g_i` linearly dependent (moments non-unique) | `MathError::not_implemented` |
| A kernel integrand has no polynomial antiderivative (log part or rational remainder) | `MathError::not_implemented` |
| `volterra_convolution_laplace`: `f` or the kernel off the Laplace table | `MathError::not_implemented` |
| `adomian_polynomials`: empty component list | `MathError::domain_error` |
| An `int64` numerator or denominator computation wraps | `MathError::overflow` |

## Worked examples

Worked from the tests (`tests/inteq_tests.cpp`). Inputs are built low-degree-first
from integer coefficients.

```cpp
import nimblecas.inteq;
import nimblecas.ratpoly;
import nimblecas.polynomial;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.laplace;
using namespace nimblecas;

// Integer-coefficient polynomial (low degree first), a rational, and an
// explicit-coefficient polynomial.
auto ipoly = [](std::vector<std::int64_t> c) {
    return RationalPoly::from_polynomial(Polynomial{std::move(c)});
};
auto R  = [](std::int64_t n, std::int64_t d = 1) { return Rational::make(n, d).value(); };
auto rp = [](std::vector<Rational> c) { return RationalPoly::from_coeffs(std::move(c)); };
// Single-term separable kernel K(x,t) = g(x) h(t).
auto kernel1 = [](RationalPoly g, RationalPoly h) {
    return SeparableKernel{.g = {std::move(g)}, .h = {std::move(h)}};
};

// --- Fredholm 2nd kind, separable (EXACT) --------------------------------------------
// K(x,t) = x t on [0,1], f = x, lambda = 1. c = ∫_0^1 t phi = 1/2, so phi = (3/2) x.
const auto k = kernel1(ipoly({0, 1}), ipoly({0, 1}));
auto sol = fredholm2_separable(ipoly({0, 1}), k, R(1), R(0), R(1)).value();
sol.phi;               // (3/2) x   (exact, complete)
sol.coefficients[0];   // 1/2       (the separable moment c)

// --- Fredholm 2nd kind, Neumann (TRUNCATED) ------------------------------------------
// Terms are x/3^n: order-2 sum = (13/9) x, order-3 = (40/27) x, -> (3/2) x.
const auto f = ipoly({0, 1});
auto n2 = fredholm2_neumann(f, k, R(1), R(0), R(1), 2).value();  // (13/9) x
auto n3 = fredholm2_neumann(f, k, R(1), R(0), R(1), 3).value();  // (40/27) x
// ADM with the identity nonlinearity reproduces the Neumann series exactly.
auto opF = IntegralOperator::fredholm(k, R(1), R(0), R(1)).value();
auto adm = adm_solve(opF, f, Nonlinearity::identity(), 2).value();  // == n2

// --- Fredholm 1st kind (ILL-POSED): only the moment is recoverable -------------------
// f = c x from ∫ x t phi. For f = 2x the moment is c = 2; f = x^2 ∉ span{x}.
auto fk = fredholm1_separable(ipoly({0, 2}), k, R(0), R(1)).value();
fk.moments[0];  // 2   (phi itself is NON-UNIQUE)
fredholm1_separable(ipoly({0, 0, 1}), k, R(0), R(1)).error();  // domain_error

// --- Volterra 2nd kind, Picard: the e^x partial sum ----------------------------------
// Kernel 1, f = 1, lambda = 1, a = 0: phi_n = Σ_{j<=n} x^j/j!.
const auto k1 = kernel1(ipoly({1}), ipoly({1}));
auto phi = volterra2_picard(ipoly({1}), k1, R(1), R(0), 4).value();
phi.coefficient(4);  // 1/24   (x^4/4!)

// --- Volterra convolution via Laplace (TRANSFORM DOMAIN) -----------------------------
// K = 1, f = 1, lambda = 2: Φ(s) = (1/s)/(1 - 2/s) = 1/(s - 2)  (inverts to e^{2x}).
const Expr one = Expr::integer(1);
auto phiS = volterra_convolution_laplace(one, one, R(2), "x", "s").value();  // 1/(s - 2)
// An off-table free term is honest about the boundary.
const Expr tanx = Expr::apply("tan", {Expr::symbol("x")});
volterra_convolution_laplace(tanx, one, R(1), "x", "s").error();  // not_implemented

// --- Nonlinear Volterra Picard: N(phi) = phi^2 ---------------------------------------
// f = x, kernel 1, a = 0. psi_1 = x + ∫_0^x t^2 dt = x + x^3/3.
const auto sq = Nonlinearity::power(2);
auto psi1 = volterra_nonlinear_picard(ipoly({0, 1}), k1, R(1), R(0), sq, 1).value();
psi1;  // x + x^3/3

// --- Adomian polynomials and ADM order 2 ---------------------------------------------
// Components phi_0 = x, phi_1 = x^3/3 with N = phi^2: A_0 = x^2, A_1 = (2/3) x^4.
auto A = adomian_polynomials(sq, {ipoly({0, 1}), rp({R(0), R(0), R(0), R(1, 3)})}).value();
A[0];  // x^2
A[1];  // (2/3) x^4
// ADM order 2 for the nonlinear Volterra equation: x + x^3/3 + (2/15) x^5.
auto opV = IntegralOperator::volterra(k1, R(1), R(0)).value();
auto adm2 = adm_solve(opV, ipoly({0, 1}), sq, 2).value();  // x + x^3/3 + (2/15) x^5

// --- ADM == HPM, HAM(ħ = -1) == ADM --------------------------------------------------
auto admN = adm_solve(opV, ipoly({0, 1}), sq, 3).value();
auto hpmN = hpm_solve(opV, ipoly({0, 1}), sq, 3).value();   // == admN
auto hamN = ham_solve(opV, ipoly({0, 1}), sq, R(-1), 3).value();  // == admN
auto ham2 = ham_solve(opV, ipoly({0, 1}), sq, R(-1, 2), 3).value();  // differs (still exact)

// --- Hammerstein (nonlinear Fredholm) Picard: K = 1 on [0,1], f = 1, N = phi^2 -------
// psi_1 = 1 + ∫_0^1 1 dt = 2;  psi_2 = 1 + ∫_0^1 2^2 dt = 5.
auto hp1 = hammerstein_picard(ipoly({1}), k1, R(1), R(0), R(1), sq, 1).value();  // 2
auto hp2 = hammerstein_picard(ipoly({1}), k1, R(1), R(0), R(1), sq, 2).value();  // 5

// --- Error paths ---------------------------------------------------------------------
const SeparableKernel ragged{.g = {ipoly({0, 1})}, .h = {}};
fredholm2_separable(ipoly({0, 1}), ragged, R(1), R(0), R(1)).error();  // domain_error
// K = x t on [0,1] has M = 1/3, so lambda = 3 makes (I - lambda M) singular.
fredholm2_separable(ipoly({0, 1}), k, R(3), R(0), R(1)).error();       // domain_error
```

The tests hand-verify every exact case, and cross-check the decomposition solvers
against each other (`ADM == HPM`, `HAM(ħ=-1) == ADM`) and against the Neumann/Picard
series for the linear case (`ADM(identity) == Neumann`).

## See also

- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Q[x]` substrate (`Rational`,
  `RationalPoly`) the unknown, free term, and kernel are built on.
- [`nimblecas.integrate`](integrate.md) — `integrate_rational`, the elementary
  polynomial integration every kernel action reduces to, and the source of the
  `not_implemented` boundary on non-polynomial antiderivatives.
- [`nimblecas.matrix`](matrix.md) — the exact `Q` linear solver behind the separable
  Fredholm moment system.
- [`nimblecas.laplace`](laplace.md) — `laplace_transform`, the transform behind the
  convolution-kernel Volterra solver.
- [`nimblecas.symbolic`](symbolic.md) and [`nimblecas.simplify`](simplify.md) — the
  `Expr` layer and canonical simplifier used by the transform-domain solution.
- `tests/inteq_crossmethod_tests.cpp` and `docs/examples/inteq-multimethod.md` —
  Fredholm-exact / Neumann / Picard / ADM / HPM / HAM cross-validated on a linear and a
  nonlinear Volterra equation, the latter run through [`execdoc`](execdoc.md).
- [Documentation hub](../Index.md)
