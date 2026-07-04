# `nimblecas.semigroup` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/semigroup/semigroup.cppm`

Functional analysis and one-parameter operator semigroups, realized in the
tractable **finite-dimensional** (matrix-operator) setting over the exact
rationals `Q`. A square rational [`Matrix`](matrix.md) `A` is read as a bounded
linear operator on `Qⁿ`; the associated **C₀-semigroup** is `T(t) = e^{tA}`, the
flow of the abstract Cauchy problem `du/dt = A u, u(0) = u0`. Everything rides
the `Result` railway and stays inside exact `Rational` arithmetic.

**Honesty about exactness.** This module has a sharp exact-vs-numerical boundary:

- **Exact over `Q` (no rounding whatsoever):** the resolvent
  `R(λ, A) = (λI − A)⁻¹`, the *rational slice* of the spectrum (rational
  eigenvalues + multiplicities), the induced operator 1-norm and inf-norm, the
  adjoint (transpose of a real rational operator), the Sylvester solve
  `A X + X B = C` and its Lyapunov special case `B = Aᵀ`, and the dissipativity
  and Hurwitz verdicts — the last two decided from characteristic-polynomial
  coefficient signs, so they settle irrational/complex spectra without root
  finding.
- **Inherited from [`nimblecas.matexp`](matexp.md)** (exact **iff** the generator
  is nilpotent with enough Taylor terms, otherwise an honest exact-*rational*
  truncation of the transcendental `e^{tA}`): the semigroup `T(t)` and everything
  built on it — the Cauchy solution `T(t)u0`, the semigroup-property check
  `T(s+t) = T(s)T(t)`, variation of constants, and the abstract-PDE helper.
- **Spectrum honesty:** irrational/complex eigenvalues are **not** extracted
  (that needs algebraic-number support, out of scope). Spectrum results report
  whether the rational eigenvalues account for the whole spectrum
  (`fully_extracted`) or not.

**Scope.** This is the finite-dimensional / matrix realization of semigroup
theory. Genuine **infinite-dimensional** functional analysis — unbounded
operators, Banach-space C₀-semigroups, domains and cores, spectral theory of
differential operators — is **out of scope** and not claimed. The abstract-PDE
helper only consumes an already-discretized (finite-dimensional) generator
matrix, e.g. a spatial discretization from [`nimblecas.pde`](pde.md).

```cpp
import nimblecas.semigroup;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), [`matrix`](matrix.md),
[`matexp`](matexp.md), [`eigen`](eigen.md), and [`dynamics`](dynamics.md).

## Two identities documented here

Stated once, they structure the API:

- **Generator / Cauchy problem.** `u(t) = T(t) u0` solves `u' = A u, u(0) = u0`,
  and `d/dt T(t)|_{t=0} = A` (`A` is the infinitesimal generator).
- **Resolvent as Laplace transform.** For `Re(λ)` greater than the spectral
  abscissa, `R(λ, A) = (λI − A)⁻¹ = ∫₀^∞ e^{−λt} T(t) dt`, connecting the
  resolvent (exact over `Q`) to the semigroup (transcendental in general).

## Result structs

Two small report structs carry the honesty flags.

```cpp
struct Spectrum {
    std::vector<std::pair<Rational, std::int64_t>> rational_values;
    std::int64_t rational_count{0};
    std::int64_t dimension{0};
    bool fully_extracted{false};
};

struct SpectralRadius {
    Rational value;
    bool exact{false};
};
```

| Field | Meaning |
| :--- | :--- |
| `Spectrum::rational_values` | Each rational eigenvalue paired with its algebraic multiplicity. |
| `Spectrum::rational_count` | Sum of those multiplicities. |
| `Spectrum::dimension` | The operator dimension `n`. |
| `Spectrum::fully_extracted` | `true` when `rational_count == dimension` — the rational eigenvalues are the **whole** spectrum. `false` means part of the spectrum is irrational/complex and unrepresented. |
| `SpectralRadius::value` | The spectral radius `ρ(A) = max\|eigenvalue\|` when `exact`, else a guaranteed rational **upper bound** on `ρ(A)`. |
| `SpectralRadius::exact` | `true` iff the spectrum is fully rational (then `value` is the exact radius); `false` means `value` is only a bound `ρ(A) ≤ ‖A‖`. |

## Operator / functional-analysis tools (all exact over `Q`)

All are free functions in `namespace nimblecas`, `[[nodiscard]]`, returning
`Result<T>`.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `resolvent` | `auto resolvent(const Matrix& a, const Rational& lambda) -> Result<Matrix>` | The resolvent `R(λ, A) = (λI − A)⁻¹`, computed exactly over `Q` via `Matrix::inverse`. Requires `A` square (else `domain_error`). At a spectral value — `λ` an eigenvalue, so `λI − A` is singular — the inverse does not exist and this surfaces as `domain_error` (propagated from `inverse()`). Overflow propagated. |
| `spectrum` | `auto spectrum(const Matrix& a) -> Result<Spectrum>` | The rational slice of the spectrum (see `Spectrum`). Requires `A` square (else `domain_error`). Exact over `Q`; irrational/complex eigenvalues are reported as **not** fully extracted rather than approximated. |
| `spectral_radius` | `auto spectral_radius(const Matrix& a) -> Result<SpectralRadius>` | Exact `max\|rational eigenvalue\|` when the spectrum is fully rational, otherwise a rational upper bound — the tighter of the induced 1- and inf-norms (`ρ(A) ≤ ‖A‖` for any submultiplicative norm). Requires `A` square (else `domain_error`). |
| `adjoint` | `auto adjoint(const Matrix& a) -> Result<Matrix>` | The adjoint operator. For a real rational operator the adjoint is the **transpose**, returned exactly. (The conjugate-transpose of a complex operator lives in [`nimblecas.cmatrix`](cmatrix.md), out of scope here.) |
| `operator_norm_1` | `auto operator_norm_1(const Matrix& a) -> Result<Rational>` | The induced 1-norm `‖A‖₁ = maxⱼ Σᵢ \|A(i,j)\|` (max absolute **column** sum), exact over `Q`. Defined for any shape; the empty matrix has norm `0`. Overflow propagated. |
| `operator_norm_inf` | `auto operator_norm_inf(const Matrix& a) -> Result<Rational>` | The induced inf-norm `‖A‖_∞ = maxᵢ Σⱼ \|A(i,j)\|` (max absolute **row** sum), exact over `Q`. Defined for any shape; the empty matrix has norm `0`. Overflow propagated. |

## C₀-semigroup generated by `A`: `T(t) = e^{tA}`

Exactness inherits [`matexp`](matexp.md)'s Taylor contract: exact **iff** `tA` is
nilpotent and `terms` is at least the nilpotency index (`t ≠ 0` preserves
nilpotency; `t = 0` gives `T(0) = I` exactly for any `terms ≥ 1`); otherwise an
honest exact-rational truncation, never a claim of the transcendental truth.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `semigroup` | `auto semigroup(const Matrix& a, const Rational& t, std::int64_t terms) -> Result<Matrix>` | The semigroup operator `T(t) = e^{tA}`, computed as the truncated Taylor series of `tA` with `terms` terms (via `matrix_exp_taylor`). Requires `A` square and `terms ≥ 1` (else `domain_error`). Exactness as above. Overflow propagated. |
| `cauchy_solution` | `auto cauchy_solution(const Matrix& a, const Matrix& u0, const Rational& t, std::int64_t terms) -> Result<Matrix>` | The abstract Cauchy problem `du/dt = A u, u(0) = u0`: returns `u(t) = T(t) u0`. Requires `A` square (`n×n`), `u0` an `n×1` column, and `terms ≥ 1` (else `domain_error`). Inherits `semigroup()`'s exactness. Overflow propagated. |
| `verify_semigroup_property` | `auto verify_semigroup_property(const Matrix& a, const Rational& s, const Rational& t, std::int64_t terms) -> Result<bool>` | Verifies the semigroup property `T(s+t) = T(s) T(t)` at the given `s, t` (`T(0) = I` is the `s=t=0` case). Returns `true` **iff** the truncated operators satisfy it exactly: for a nilpotent `A` with `terms ≥` its index this holds; for a truncated non-nilpotent `A` the identity is only approximate and this honestly returns `false`. Requires `A` square and `terms ≥ 1` (else `domain_error`). |
| `pde_semigroup_solution` | `auto pde_semigroup_solution(const Matrix& generator, const Matrix& u0, const Rational& t, std::int64_t terms) -> Result<Matrix>` | Abstract-PDE connection. A linear evolution PDE `u_t = L[u]` becomes `u' = A u` once `L` is replaced by its finite-dimensional spatial discretization `A` (the generator), e.g. from [`nimblecas.pde`](pde.md). Given that discretized generator and initial data `u0`, returns `T(t) u0` — identical to `cauchy_solution`, named for the PDE context. Requires `A` square (`n×n`), `u0` an `n×1` column, `terms ≥ 1` (else `domain_error`). |

## Hille–Yosida / Lumer–Phillips (finite-dimensional realization)

Every matrix `A` generates a C₀-semigroup (the finite-dimensional Hille–Yosida
statement), so the interesting content is the **contraction / decay** conditions,
each decided **exactly**.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `is_dissipative` | `auto is_dissipative(const Matrix& a) -> Result<bool>` | Whether `A` is dissipative: its symmetric part is negative semidefinite, `A + Aᵀ ≤ 0` (equivalently `xᵀ(A + Aᵀ)x ≤ 0` for all `x`). Decided exactly by testing that `−(A + Aᵀ)` is positive semidefinite via a characteristic-polynomial coefficient-sign test (a real symmetric `S` with monic char. poly `Σ c_k λ^k` of degree `n` is PSD iff `(−1)^{n+k} c_k ≥ 0` for all `k`). Like Routh–Hurwitz this settles irrational eigenvalues without root finding. Requires `A` square (else `domain_error`). |
| `is_contraction_generator` | `auto is_contraction_generator(const Matrix& a) -> Result<bool>` | Whether `A` generates a **contraction** semigroup (`‖T(t)‖ ≤ 1` for all `t ≥ 0`). By Lumer–Phillips (finite-dimensional realization) this is exactly dissipativity, so it delegates to `is_dissipative` and returns an exact verdict. Requires `A` square (else `domain_error`). |
| `is_hurwitz` | `auto is_hurwitz(const Matrix& a) -> Result<bool>` | Whether `A` is **Hurwitz**: every eigenvalue has strictly negative real part, so `‖T(t)‖ → 0` as `t → ∞` (uniform exponential stability). Decided exactly by the Routh–Hurwitz criterion (via [`nimblecas.dynamics`](dynamics.md)), so it settles irrational/complex spectra. Requires `A` square with `n ≥ 1` (else `domain_error`). |

## Operator equations

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `sylvester_solve` | `auto sylvester_solve(const Matrix& a, const Matrix& b, const Matrix& c) -> Result<Matrix>` | Solves the Sylvester equation `A X + X B = C` exactly over `Q` by vectorization: with column-stacking `vec`, `(Iₙ ⊗ A + Bᵀ ⊗ Iₘ) vec(X) = vec(C)`, an `(mn)×(mn)` exact rational linear system. Requires `A` square (`m×m`), `B` square (`n×n`), and `C` of shape `m×n` (else `domain_error`). The system is singular — propagated as `domain_error` from the solve — **iff** `A` and `−B` share an eigenvalue (no unique solution). Overflow propagated. |
| `lyapunov_solve` | `auto lyapunov_solve(const Matrix& a, const Matrix& c) -> Result<Matrix>` | The Lyapunov special case `B = Aᵀ`: solves `A X + X Aᵀ = C` exactly over `Q`. Requires `A` square (`n×n`) and `C` of shape `n×n` (else `domain_error`). Singular iff `A` and `−Aᵀ` share an eigenvalue (i.e. `λᵢ + λⱼ = 0` for some eigenvalues), propagated as `domain_error`. |
| `variation_of_constants` | `auto variation_of_constants(const Matrix& a, const Matrix& u0, const std::vector<Matrix>& forcing, const Rational& t, std::int64_t terms) -> Result<Matrix>` | Variation of constants for `u' = A u + f(s), u(0) = u0` with polynomial forcing `f(s) = Σⱼ forcing[j] · sʲ` (each `forcing[j]` an `n×1` vector; an empty list means the homogeneous problem `f = 0`). Returns the closed form `u(t) = T(t) u0 + ∫₀^t T(t−s) f(s) ds`, every operator series truncated at `terms` terms. Exact **iff** `A` is nilpotent and `terms ≥` its index (then every series is finite and the integral is closed-form over `Q`); otherwise the honest exact-rational truncation. Requires `A` square (`n×n`), `u0` and each `forcing[j]` of shape `n×1`, and `terms ≥ 1` (else `domain_error`). Overflow propagated. (Exponential and other non-polynomial forcing are out of scope here — pass a polynomial or use the unevaluated form.) |

## Error model

| Condition | Error |
| :--- | :--- |
| Any function given a non-square `A` (where squareness is required) | `MathError::domain_error` |
| `resolvent` at a spectral value `λ` (`λI − A` singular) | `MathError::domain_error` (from `inverse()`) |
| `semigroup` / `cauchy_solution` / `verify_semigroup_property` / `pde_semigroup_solution` / `variation_of_constants` with `terms < 1` | `MathError::domain_error` |
| `cauchy_solution` / `pde_semigroup_solution` / `variation_of_constants` with `u0` (or a forcing term) not an `n×1` column | `MathError::domain_error` |
| `sylvester_solve` with `A` or `B` non-square, or `C` not `m×n` | `MathError::domain_error` |
| `sylvester_solve` / `lyapunov_solve` singular system (`A` and `−B` share an eigenvalue) | `MathError::domain_error` (from the solve) |
| An `int64` numerator or denominator computation wraps | `MathError::overflow` |

The exactness contracts are **not** error conditions: an inexact result on a
non-nilpotent generator (or under-`terms` truncation) is a well-defined
*rational approximation*, returned as a success value, not a failure. The
`is_dissipative` / `is_contraction_generator` / `is_hurwitz` verdicts are exact
`bool`s and only fail on the shape guard.

## Worked examples

```cpp
import nimblecas.semigroup;
import nimblecas.matrix;
import nimblecas.ratpoly;
using namespace nimblecas;

auto ri = [](std::int64_t v) { return Rational::from_int(v); };
auto mat = [](std::vector<std::vector<std::int64_t>> rows) {
    std::vector<std::vector<Rational>> r;
    for (const auto& row : rows) {
        std::vector<Rational> rr;
        for (std::int64_t v : row) rr.push_back(Rational::from_int(v));
        r.push_back(std::move(rr));
    }
    return Matrix::from_rows(std::move(r)).value();
};
auto col = [&](std::vector<std::int64_t> e) {
    std::vector<std::vector<std::int64_t>> rows;
    for (std::int64_t v : e) rows.push_back({v});
    return mat(std::move(rows));
};

// --- Resolvent, exact off the spectrum ---------------------------------------
// A = [[1,2],[0,3]] has eigenvalues 1, 3. At lambda = 2, (2I - A) = [[1,-2],[0,-1]]
// is its own inverse, so R(2, A) = [[1,-2],[0,-1]] exactly.
const Matrix a = mat({{1, 2}, {0, 3}});
auto r = resolvent(a, ri(2)).value();
r.at(0, 1) == ri(-2);                     // true
// At an eigenvalue the resolvent does not exist.
resolvent(a, ri(1)).error();              // MathError::domain_error

// --- Spectrum and spectral radius --------------------------------------------
auto s = spectrum(a).value();
s.fully_extracted;                        // true  (triangular: spectrum {1, 3})
s.rational_count == 2 && s.dimension == 2;// true
auto rho = spectral_radius(a).value();
rho.exact && rho.value == ri(3);          // true  (exact rho = max(|1|,|3|) = 3)

// Rotation [[0,-1],[1,0]] has eigenvalues +/- i: no rational part, so the radius
// falls back to a rational upper bound = min(||A||_1, ||A||_inf) = 1.
const Matrix rot = mat({{0, -1}, {1, 0}});
auto rho2 = spectral_radius(rot).value();
!rho2.exact && rho2.value == ri(1);       // true  (a bound, not the exact radius)

// --- Operator norms and adjoint ----------------------------------------------
const Matrix b = mat({{1, -2}, {3, 4}});
operator_norm_1(b).value()   == ri(6);    // max abs column sum = max(4, 6)
operator_norm_inf(b).value() == ri(7);    // max abs row sum    = max(3, 7)
adjoint(mat({{1, 2}, {3, 4}})).value().is_equal(mat({{1, 3}, {2, 4}}));  // transpose

// --- C0-semigroup on a nilpotent generator (genuinely exact) -----------------
// N = [[0,1],[0,0]] is nilpotent (N^2 = 0), so T(t) = I + tN exactly.
const Matrix n = mat({{0, 1}, {0, 0}});
semigroup(n, ri(0), 3).value().is_equal(Matrix::identity(2));   // T(0) = I
semigroup(n, ri(3), 4).value().is_equal(mat({{1, 3}, {0, 1}})); // T(3) = I + 3N
verify_semigroup_property(n, ri(1), ri(2), 4).value();          // true: T(3) = T(1)T(2)

// Cauchy problem u' = N u, u0 = [0,1]^T  =>  u(t) = [t, 1]^T.
const Matrix u0 = col({0, 1});
cauchy_solution(n, u0, ri(1), 3).value().is_equal(col({1, 1}));      // u(1) = [1,1]
pde_semigroup_solution(n, u0, ri(1), 3).value().is_equal(col({1, 1}));// same, PDE-named

// --- Dissipativity / contraction / Hurwitz (exact verdicts) ------------------
const Matrix stable = mat({{-1, 0}, {0, -2}});
is_dissipative(stable).value();            // true  (symmetric part diag(-2,-4) <= 0)
is_contraction_generator(stable).value();  // true  (Lumer-Phillips)
is_hurwitz(stable).value();                // true  (eigenvalues -1, -2)

// Skew J = [[0,1],[-1,0]]: A + A^T = 0, so dissipative (contraction), but the
// spectrum {+/- i} sits on the imaginary axis -> NOT Hurwitz. Distinguishes them.
const Matrix skew = mat({{0, 1}, {-1, 0}});
is_dissipative(skew).value();              // true
is_hurwitz(skew).value();                  // false

// Identity: symmetric part diag(2,2) is positive definite -> NOT dissipative.
is_dissipative(Matrix::identity(2)).value();  // false

// --- Sylvester and Lyapunov solves (exact reconstruction) --------------------
// A X + X B = C with A = diag(1,2), B = diag(3,4), X = [[1,2],[3,4]] gives
// C = [[4,10],[15,24]]; the solver recovers X exactly.
auto xsol = sylvester_solve(mat({{1, 0}, {0, 2}}), mat({{3, 0}, {0, 4}}),
                            mat({{4, 10}, {15, 24}})).value();
xsol.is_equal(mat({{1, 2}, {3, 4}}));      // true

// A X + X A^T = C, A = diag(-1,-2), X = [[2,1],[1,3]] -> C = [[-4,-3],[-3,-12]].
lyapunov_solve(mat({{-1, 0}, {0, -2}}), mat({{-4, -3}, {-3, -12}}))
    .value().is_equal(mat({{2, 1}, {1, 3}}));  // true

// --- Variation of constants (nilpotent => exact closed form) -----------------
// u' = N u + f, u0 = 0, constant f = [1,0]^T  =>  u(t) = [t, 0]^T.
const std::vector<Matrix> fconst = {col({1, 0})};
variation_of_constants(n, col({0, 0}), fconst, ri(2), 3)
    .value().is_equal(col({2, 0}));        // u(2) = [2, 0]

// Linear forcing f(s) = [s,0]^T (c0 = 0, c1 = [1,0]): u(t) = [t^2/2, 0]^T.
const std::vector<Matrix> flin = {col({0, 0}), col({1, 0})};
variation_of_constants(n, col({0, 0}), flin, ri(2), 4)
    .value().is_equal(col({2, 0}));        // u(2) = [t^2/2, 0]|_{t=2} = [2, 0]

// Empty forcing reduces to the homogeneous Cauchy solution.
variation_of_constants(n, col({0, 1}), {}, ri(1), 3).value()
    .is_equal(cauchy_solution(n, col({0, 1}), ri(1), 3).value());  // true

// --- Guards ------------------------------------------------------------------
const Matrix ns = mat({{1, 2, 3}, {4, 5, 6}});   // non-square
resolvent(ns, ri(1)).error();                    // domain_error
semigroup(Matrix::identity(2), ri(1), 0).error();// domain_error (terms < 1)
cauchy_solution(Matrix::identity(2), col({1, 2, 3}), ri(1), 3).error();  // domain_error
```

## See also

- [`nimblecas.matexp`](matexp.md) — the truncated matrix exponential
  `T(t) = e^{tA}` and its exactness contract, which this module inherits.
- [`nimblecas.matrix`](matrix.md) — the exact `Rational` matrix type these
  operators build on (`multiply`, `scale`, `inverse`, `transpose`, `solve`).
- [`nimblecas.eigen`](eigen.md) — rational eigenvalues and the characteristic
  polynomial behind `spectrum`, `spectral_radius`, and the PSD test.
- [`nimblecas.dynamics`](dynamics.md) — the Routh–Hurwitz stability criterion
  behind `is_hurwitz`.
- [`nimblecas.pde`](pde.md) — the source of discretized generators consumed by
  `pde_semigroup_solution`.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field every entry
  lives in.
- [Documentation hub](../Index.md)
