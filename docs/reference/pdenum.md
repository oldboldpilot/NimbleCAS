# `nimblecas.pdenum` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/pdenum/pdenum.cppm`

Numerical methods for partial differential equations — finite differences
(FDM), finite elements (FEM), finite volumes (FVM), and the method of lines
(MoL). This is the **numerical/discretization** counterpart to
[`nimblecas.pde`](pde.md) (the exact power-series-in-time evolution solver):
here a differential operator is discretized on a grid or mesh and the resulting
linear system is solved.

The module keeps a hard, documented **honesty boundary**:

- The **spatial discretizations** — finite-difference stencils, finite-element
  stiffness/mass/load, finite-volume face fluxes — and the linear **solves**
  built from them are **exact over the field `Q`** whenever the grid nodes and
  the data (source `f`, coefficients, boundary values) are rational, and for FEM
  whenever the source is a **polynomial** (the element integrals `∫ f φ` are then
  exact rationals). Every entry flows through [`Rational`](ratpoly.md)'s
  overflow-checked arithmetic, and the systems are solved by the exact rational
  cores ([`bandsolve`](bandsolve.md) Thomas for tridiagonal systems,
  [`matrix`](matrix.md) Gauss–Jordan for the 2-D system). **No floating point
  and no tolerance** enters any spatial-discretization or solve path.
- What is exact is the solution of the **discrete system**, **not** the true PDE
  solution. The **discretization error** (discrete solution vs. exact PDE
  solution) is a separate, documented matter. For some special data — a
  quadratic exact solution under the 3-point second-difference stencil, a linear
  exact solution under the 5-point Laplacian — the discrete nodal values
  *coincide* with the exact solution at the nodes, and the tests exploit that;
  but that is a property of the data, not a claim that discretization equals the
  PDE in general.
- **Time stepping is numerical.** [`crank_nicolson_step`](#method-of-lines-mol)
  — the module's **only** time integrator — operates in double precision and is
  explicitly marked so. Any statement about convergence order, stability, or a
  CFL condition is a numerical-analysis statement, not an exact algebraic one,
  and is not asserted here. The method-of-lines builders return the semi-discrete
  operator `L` **exactly over `Q`** (so it can be fed to exact ODE machinery);
  only the wall-clock time integration is floating point.
- Non-rational data and nonlinear PDEs are **out of scope**: they surface as
  `not_implemented` / `domain_error` on the railway. Nothing throws (Rule 32).

```cpp
import nimblecas.pdenum;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), [`matrix`](matrix.md),
and [`bandsolve`](bandsolve.md).

## Discretization options and data

| Type | Definition | Meaning |
| :--- | :--- | :--- |
| `DiffScheme` | `enum class DiffScheme : std::uint8_t { central, forward, backward }` | First-derivative finite-difference stencil orientation. `central` is 2nd-order interior; `forward` / `backward` are 1st-order. |
| `BCKind` | `enum class BCKind : std::uint8_t { dirichlet, neumann }` | The kind of boundary condition imposed at a domain endpoint. |
| `BoundaryCondition` | `struct { BCKind kind; Rational value; }` | A Dirichlet value `u = value` or a Neumann value `u' = value` (the flux convention is documented per solver). The stored `Rational` keeps the boundary datum exact. |
| `RationalField1D` | `std::function<Result<Rational>(const Rational& x)>` | A rational scalar field of one variable, sampled exactly on the railway. |
| `RationalField2D` | `std::function<Result<Rational>(const Rational& x, const Rational& y)>` | A rational scalar field of two variables, sampled exactly on the railway. |

`BoundaryCondition` has two factory helpers (both `[[nodiscard]] static`):

```cpp
static auto dirichlet(Rational v) -> BoundaryCondition;  // u = v
static auto neumann(Rational v)   -> BoundaryCondition;  // u' = v
```

## Grids and meshes

### `Grid1D` — a uniform rational grid on `[a, b]`

`a = x₀ < x₁ < … < x_N = b`. `n_intervals` is `N` (the number of cells); there
are `N + 1` nodes. The spacing `h = (b − a)/N` is exact over `Q`.

| Member | Signature | Behavior |
| :--- | :--- | :--- |
| fields | `Rational a; Rational b; std::size_t n_intervals;` | Endpoints and interval count. |
| `make` | `static auto make(Rational a, Rational b, std::size_t n_intervals) -> Result<Grid1D>` | Requires `N ≥ 1` and `b > a`, else `domain_error`. |
| `num_nodes` | `auto num_nodes() const noexcept -> std::size_t` | `n_intervals + 1`. |
| `spacing` | `auto spacing() const -> Result<Rational>` | The exact spacing `h = (b − a)/N`. |
| `node` | `auto node(std::size_t i) const -> Result<Rational>` | The exact node `a + i·h`; `i > N` is `domain_error`. |

### `Grid2D` — a uniform rational grid on `[ax, bx] × [ay, by]`

`nx`, `ny` are the interval counts in `x` and `y`; there are `(nx+1)(ny+1)`
nodes.

| Member | Signature | Behavior |
| :--- | :--- | :--- |
| fields | `Rational ax, bx, ay, by; std::size_t nx, ny;` | Rectangle corners and interval counts. |
| `make` | `static auto make(Rational ax, Rational bx, Rational ay, Rational by, std::size_t nx, std::size_t ny) -> Result<Grid2D>` | Requires `nx, ny ≥ 1` and `bx > ax`, `by > ay`, else `domain_error`. |
| `hx` / `hy` | `auto hx() const -> Result<Rational>` / `auto hy() const -> Result<Rational>` | The exact spacings `(bx − ax)/nx`, `(by − ay)/ny`. |
| `node_x` / `node_y` | `auto node_x(std::size_t i) const -> Result<Rational>` / `auto node_y(std::size_t j) const -> Result<Rational>` | The exact nodes `ax + i·hx`, `ay + j·hy`; index past the boundary is `domain_error`. |

### `Mesh1D` — a (possibly non-uniform) 1-D mesh

Strictly increasing rational nodes, used by the finite-element assembly where
element lengths may vary. `nodes.size() ≥ 2`.

| Member | Signature | Behavior |
| :--- | :--- | :--- |
| field | `std::vector<Rational> nodes;` | Strictly increasing, size `≥ 2`. |
| `from_nodes` | `static auto from_nodes(std::vector<Rational> nodes) -> Result<Mesh1D>` | Fewer than 2 nodes, or a non-strictly-increasing sequence, is `domain_error`. |
| `uniform` | `static auto uniform(const Rational& a, const Rational& b, std::size_t n) -> Result<Mesh1D>` | The `n`-interval uniform mesh on `[a, b]` (built via `Grid1D::make`). |
| `num_nodes` | `auto num_nodes() const noexcept -> std::size_t` | `nodes.size()`. |
| `num_elements` | `auto num_elements() const noexcept -> std::size_t` | `nodes.size() − 1`. |
| `element_length` | `auto element_length(std::size_t e) const -> Result<Rational>` | The exact length `nodes[e+1] − nodes[e]`; out-of-range `e` is `domain_error`. |

## Finite difference (FDM)

All FDM builders are free functions in namespace `nimblecas`; the spatial
matrices and solves are exact over `Q`.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `fdm_d1_matrix` | `[[nodiscard]] auto fdm_d1_matrix(std::size_t n, const Rational& h, DiffScheme scheme) -> Result<Matrix>` | The `n × n` first-derivative differentiation matrix on a uniform grid of spacing `h`. Interior rows use `scheme`; a boundary row that would need a ghost point falls back to the one-sided stencil (forward at the first row, backward at the last) so the matrix stays square without ghosts. Requires `n ≥ 2` and `h ≠ 0`, else `domain_error`. |
| `fdm_d2_matrix` | `[[nodiscard]] auto fdm_d2_matrix(std::size_t n, const Rational& h) -> Result<Matrix>` | The `n × n` central second-derivative matrix `tridiag(1, −2, 1)/h²` (interior operator; homogeneous-Dirichlet boundaries implied by truncation at the first/last row). Requires `n ≥ 1` and `h ≠ 0`, else `domain_error`. |
| `solve_poisson_1d` | `[[nodiscard]] auto solve_poisson_1d(const Grid1D& grid, std::span<const Rational> f_nodal, const BoundaryCondition& left, const BoundaryCondition& right) -> Result<std::vector<Rational>>` | Solves `−u'' = f` on `[a, b]` by the 3-point stencil, **exactly over `Q`**. `f_nodal` holds `f` at *every* grid node (`num_nodes()` entries; values at Dirichlet nodes are ignored). Dirichlet and Neumann ends are supported (the Neumann value is `u'(a)` at the left / `u'(b)` at the right, imposed by the 2nd-order ghost-node relation that keeps the system tridiagonal). Solved by the exact Thomas algorithm ([`bandsolve`](bandsolve.md)). Returns the full nodal solution (boundary nodes included). Wrong `f_nodal` length is `domain_error`; a singular system (e.g. pure Neumann/Neumann, defined only up to a constant) surfaces as `domain_error` from the solver. |
| `poisson_2d_laplacian` | `[[nodiscard]] auto poisson_2d_laplacian(std::size_t nx, std::size_t ny, const Rational& hx, const Rational& hy) -> Result<Matrix>` | The 5-point Laplacian matrix for `−Δu` on the interior of an `(nx × ny)`-interval rectangle (homogeneous-Dirichlet coupling only; size `(nx−1)(ny−1)` square, lexicographic `x`-fastest ordering). Requires `nx, ny ≥ 2` and `hx, hy ≠ 0`. |
| `solve_poisson_2d` | `[[nodiscard]] auto solve_poisson_2d(const Grid2D& grid, const RationalField2D& f, const RationalField2D& boundary) -> Result<Matrix>` | Solves `−Δu = f` on the rectangle with Dirichlet data `boundary`, **exactly over `Q`**, via the 5-point stencil and the exact rational dense solver ([`matrix`](matrix.md)). `f` and `boundary` are sampled exactly at rational points. Returns the interior nodal values as a `(ny−1) × (nx−1)` matrix (row `j`, column `i` ⇒ node `(i+1, j+1)`). Requires `nx, ny ≥ 2` (a non-empty interior), else `domain_error`; any sampler error propagates. |

## Finite element (FEM)

P1 (linear Lagrange) assembly is exact over `Q`; the load vector is exact when
the source `f` is a [`RationalPoly`](ratpoly.md) (`f φ_i` is then a polynomial on
each element, integrated by exact antiderivative + evaluation). The P2
(quadratic Lagrange) element matrices are exact closed forms.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `fem_p1_stiffness` | `[[nodiscard]] auto fem_p1_stiffness(const Mesh1D& mesh) -> Result<Matrix>` | The global P1 stiffness `K_ij = ∫ φ_i' φ_j'`, assembled element by element as `(1/h_e)[[1,−1],[−1,1]]`. On a uniform mesh the interior is `tridiag(−1, 2, −1)/h` exactly. Size `num_nodes() × num_nodes()`; no boundary conditions applied. |
| `fem_p1_mass` | `[[nodiscard]] auto fem_p1_mass(const Mesh1D& mesh) -> Result<Matrix>` | The global P1 mass `M_ij = ∫ φ_i φ_j`, assembled as `(h_e/6)[[2,1],[1,2]]`. Size `num_nodes() × num_nodes()`; no boundary conditions applied. |
| `fem_p1_load` | `[[nodiscard]] auto fem_p1_load(const Mesh1D& mesh, const RationalPoly& f) -> Result<std::vector<Rational>>` | The global P1 load `b_i = ∫ f φ_i` for a **polynomial** source `f`. Because `f φ_i` is a polynomial on each element, the element integrals are exact rationals. Size `num_nodes()`. |
| `fem_p1_solve` | `[[nodiscard]] auto fem_p1_solve(const Mesh1D& mesh, const Rational& a_coeff, const Rational& c_coeff, const RationalPoly& f, const BoundaryCondition& left, const BoundaryCondition& right) -> Result<std::vector<Rational>>` | Solves the Galerkin P1 problem `−(a u')' + c u = f` with constant rational `a`, `c` and polynomial `f`, **exactly over `Q`**. Forms `(a K + c M) u = b`; Neumann data enters weakly as the natural boundary term (left: `b_0 −= a·value`; right: `b_N += a·value`; `value = u'` at that end); Dirichlet data is imposed by symmetric row/column elimination; the reduced system is solved exactly ([`matrix`](matrix.md)). Returns the full nodal solution. A singular reduced system (e.g. pure homogeneous Neumann with `c = 0`) is `domain_error`. |
| `fem_p2_element_stiffness` | `[[nodiscard]] auto fem_p2_element_stiffness(const Rational& h) -> Result<Matrix>` | The `3 × 3` P2 element stiffness `(1/(3h))[[7,−8,1],[−8,16,−8],[1,−8,7]]` for an element of length `h` (node order: left, mid, right). Exact over `Q`. `h ≠ 0`. |
| `fem_p2_element_mass` | `[[nodiscard]] auto fem_p2_element_mass(const Rational& h) -> Result<Matrix>` | The `3 × 3` P2 element mass `(h/30)[[4,2,−1],[2,16,2],[−1,2,4]]` for an element of length `h`. Exact over `Q`. `h ≠ 0`. |

## Finite volume (FVM)

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `fvm_solve_diffusion_1d` | `[[nodiscard]] auto fvm_solve_diffusion_1d(const Grid1D& grid, const Rational& k, std::span<const Rational> f_cells, const Rational& u_left, const Rational& u_right) -> Result<std::vector<Rational>>` | Solves the 1-D steady conservation law `−(k u')' = f` by a **cell-centered** finite volume method on the uniform grid, **exactly over `Q`**. There are `N = n_intervals` cells; `f_cells` holds the cell-averaged source per cell (`N` entries). Constant diffusivity `k ≠ 0`. Dirichlet values `u_left`, `u_right` are imposed at `x = a`, `x = b` via half-cell face-flux reconstruction. The rational tridiagonal system is solved by the exact Thomas algorithm. Returns the `N` cell-center values `u_i` (at `x = a + (i + ½)·h`). Wrong `f_cells` length or `k == 0` is `domain_error`. |

## Method of lines (MoL)

The operator builders return the semi-discrete operator `L` **exactly over `Q`**
(`du/dt = L u`, ready for exact ODE machinery). Only `crank_nicolson_step` is
numerical.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `mol_heat_operator` | `[[nodiscard]] auto mol_heat_operator(std::size_t n, const Rational& h, const Rational& alpha) -> Result<Matrix>` | The semi-discrete heat operator for `u_t = α u_xx`: `L = (α/h²) tridiag(1, −2, 1)`, an `n × n` matrix over the interior (homogeneous-Dirichlet) nodes. **Exact over `Q`.** Requires `n ≥ 1`, `h ≠ 0`. |
| `mol_advection_operator` | `[[nodiscard]] auto mol_advection_operator(std::size_t n, const Rational& h, const Rational& c, DiffScheme scheme) -> Result<Matrix>` | The semi-discrete advection operator for `u_t = −c u_x`: `L = −c·D1` with the chosen scheme (upwind = `backward` for `c > 0`). **Exact over `Q`.** Requires `n ≥ 2`, `h ≠ 0`. |
| `crank_nicolson_step` | `[[nodiscard]] auto crank_nicolson_step(const Matrix& l, std::span<const double> u, double dt) -> Result<std::vector<double>>` | Advances one Crank–Nicolson step of `du/dt = L u`: `(I − ½dt·L) u^{n+1} = (I + ½dt·L) u^n`. **Numerical** — the one deliberately floating-point routine in the module: `L` is converted to `double` and the step solves a double-precision dense linear system. Crank–Nicolson is unconditionally stable for the diffusion operator (a numerical-analysis fact, not asserted here). Requires `L` square with `l.rows() == u.size()`, else `domain_error`; a singular `(I − ½dt·L)` is `domain_error`. |

## Error model

Everything reports failure on the railway ([`Result<T>`](core.md) / `MathError`);
nothing throws (Rule 32).

| Condition | Error |
| :--- | :--- |
| `Grid1D::make` with `n_intervals == 0`, or `b ≤ a` | `MathError::domain_error` |
| `Grid2D::make` with `nx == 0` or `ny == 0`, or `bx ≤ ax` or `by ≤ ay` | `MathError::domain_error` |
| `Mesh1D::from_nodes` with fewer than 2 nodes, or a non-strictly-increasing sequence | `MathError::domain_error` |
| `Grid1D::node` / `Grid2D::node_x` / `node_y` / `Mesh1D::element_length` with an out-of-range index | `MathError::domain_error` |
| `fdm_d1_matrix` with `n < 2` or `h == 0` | `MathError::domain_error` |
| `fdm_d2_matrix` / `mol_heat_operator` with `n == 0` or `h == 0` | `MathError::domain_error` |
| `poisson_2d_laplacian` / `solve_poisson_2d` with `nx < 2` or `ny < 2` (empty interior), or `hx == 0` / `hy == 0` | `MathError::domain_error` |
| `solve_poisson_2d` with a null `f` or `boundary` sampler | `MathError::domain_error` |
| `solve_poisson_1d` with `f_nodal.size() != num_nodes()` | `MathError::domain_error` |
| `solve_poisson_1d` / `fem_p1_solve` on a singular system (e.g. pure Neumann/Neumann) | `MathError::domain_error` (from the exact solver) |
| `fvm_solve_diffusion_1d` with `k == 0`, or `f_cells.size() != n_intervals` | `MathError::domain_error` |
| `fem_p2_element_stiffness` / `fem_p2_element_mass` with `h == 0` | `MathError::domain_error` |
| `crank_nicolson_step` with non-square `L`, or `l.rows() != u.size()`, or a singular `(I − ½dt·L)` | `MathError::domain_error` |
| Any `int64` `Rational`-op numerator/denominator computation wraps | `MathError::overflow` |

Overflow is checked at every `Rational` add/subtract/multiply/divide in every
assembly and solve, so an exact intermediate that would exceed `int64` surfaces
as `MathError::overflow` rather than silently wrapping.

## Worked examples

```cpp
import nimblecas.pdenum;
import nimblecas.ratpoly;
import nimblecas.matrix;
using namespace nimblecas;

auto ri = [](std::int64_t v) { return Rational::from_int(v); };
auto rq = [](std::int64_t n, std::int64_t d) { return Rational::make(n, d).value(); };
auto vec = [](std::vector<std::int64_t> es) {
    std::vector<Rational> v;
    for (std::int64_t e : es) v.push_back(Rational::from_int(e));
    return v;
};

// --- FDM: the central 2nd-derivative operator on h = 1/4 -> tridiag(1,-2,1)/h^2.
// h^2 = 1/16, so diag = -2/h^2 = -32 and off = 1/h^2 = 16.
auto d2 = fdm_d2_matrix(3, rq(1, 4)).value();   // 3x3
d2.at(0, 0);                                     // -32
d2.at(0, 1);                                     // 16

// --- FDM: -u'' = 2 on [0,1], u(0)=u(1)=0, exact u = x - x^2. On N=4 the 3-point
// stencil is nodally exact for a quadratic: interior = [3/16, 1/4, 3/16].
auto g1 = Grid1D::make(ri(0), ri(1), 4).value();
auto u1 = solve_poisson_1d(g1, vec({2, 2, 2, 2, 2}),   // f at all 5 nodes
                           BoundaryCondition::dirichlet(ri(0)),
                           BoundaryCondition::dirichlet(ri(0))).value();
// u1 == [0, 3/16, 1/4, 3/16, 0]   (exact rationals)

// Mixed BC: -u'' = 0, u(0)=0, u'(1)=1 -> exact u = x; on N=2, u == [0, 1/2, 1].
auto g2 = Grid1D::make(ri(0), ri(1), 2).value();
auto u2 = solve_poisson_1d(g2, vec({0, 0, 0}),
                           BoundaryCondition::dirichlet(ri(0)),
                           BoundaryCondition::neumann(ri(1))).value();  // [0, 1/2, 1]

// --- FDM 2-D: -Δu = 0 with boundary u = x + y (harmonic, linear). The 5-point
// Laplacian is exact for a linear function; interior (2x2) equals x_i + y_j.
auto g2d = Grid2D::make(ri(0), ri(1), ri(0), ri(1), 3, 3).value();
auto f0  = [&](const Rational&, const Rational&) -> Result<Rational> { return ri(0); };
auto bxy = [&](const Rational& x, const Rational& y) -> Result<Rational> { return x.add(y); };
auto u2d = solve_poisson_2d(g2d, f0, bxy).value();  // 2x2
u2d.at(0, 0);   // 2/3  (= 1/3 + 1/3)
u2d.at(1, 1);   // 4/3  (= 2/3 + 2/3)

// --- FEM P1: stiffness on uniform [0,1], N=4, h=1/4 -> tridiag(-1,2,-1)/h,
// interior diagonal 2/h = 8, off -1/h = -4, corner 1/h = 4.
auto mesh = Mesh1D::uniform(ri(0), ri(1), 4).value();
auto K = fem_p1_stiffness(mesh).value();   // 5x5
K.at(1, 1);   // 8
K.at(0, 0);   // 4
K.at(0, 1);   // -4

// FEM P1 Galerkin solve of -u'' = 2 (a=1, c=0), Dirichlet 0/0. Nodally exact:
// same [3/16, 1/4, 3/16] interior as the FDM result.
auto uf = fem_p1_solve(mesh, ri(1), ri(0), RationalPoly::constant(ri(2)),
                       BoundaryCondition::dirichlet(ri(0)),
                       BoundaryCondition::dirichlet(ri(0))).value();
// uf == [0, 3/16, 1/4, 3/16, 0]

// FEM P1 load for the polynomial source f(x) = x on [0,1], N=2 (exact integrals).
auto m2 = Mesh1D::uniform(ri(0), ri(1), 2).value();
auto b  = fem_p1_load(m2, RationalPoly::monomial(ri(1), 1)).value();
// b == [1/24, 1/4, 5/24]

// FEM P2 element matrices for h = 1 (exact closed forms).
auto Kp2 = fem_p2_element_stiffness(ri(1)).value();   // (1/3)[[7,-8,1],...]
Kp2.at(1, 1);   // 16/3
auto Mp2 = fem_p2_element_mass(ri(1)).value();        // (1/30)[[4,2,-1],...]
Mp2.at(0, 0);   // 2/15

// --- FVM: -(k u')' = 0 with k=1, u(0)=0, u(1)=1 -> exact u = x. Cell-centered
// N=4 gives the exact cell centers x_i = (i + 1/2)/4.
auto uv = fvm_solve_diffusion_1d(g1, ri(1), vec({0, 0, 0, 0}), ri(0), ri(1)).value();
// uv == [1/8, 3/8, 5/8, 7/8]

// --- MoL: the semi-discrete heat operator L = tridiag(1,-2,1)/h^2 (alpha=1,
// h=1/2 -> diag = -8, off = 4), exact over Q.
auto L = mol_heat_operator(3, rq(1, 2), ri(1)).value();
L.at(0, 0);   // -8
L.at(0, 1);   // 4

// One Crank-Nicolson step (NUMERICAL, double precision) advances a state under L.
std::vector<double> u0 = {0.3, 0.5, 0.3};
auto un = crank_nicolson_step(L, u0, 0.01).value();  // next double-precision state

// --- Error paths.
Grid1D::make(ri(1), ri(0), 4).error();                        // domain_error (b <= a)
solve_poisson_1d(g1, vec({1, 1, 1}),                          // wrong f_nodal length
                 BoundaryCondition::dirichlet(ri(0)),
                 BoundaryCondition::dirichlet(ri(0))).error();  // domain_error
fvm_solve_diffusion_1d(g1, ri(0), vec({0, 0, 0, 0}),          // k == 0
                       ri(0), ri(1)).error();                   // domain_error
crank_nicolson_step(L, std::vector<double>{1.0, 2.0}, 0.01).error();  // domain_error (size)
```

## See also

- [`nimblecas.pde`](pde.md) — the exact power-series-in-time evolution PDE
  solver this module is the numerical counterpart to.
- [`nimblecas.bandsolve`](bandsolve.md) — the exact Thomas tridiagonal solver
  behind the 1-D FDM/FVM solves.
- [`nimblecas.matrix`](matrix.md) — the `Matrix` type used for every operator
  and the exact dense solver behind the 2-D and FEM systems.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field every entry
  lives in, and the `RationalPoly` source for the FEM load.
- [`nimblecas.core`](core.md) — the `Result<T>` / `MathError` railway.
- [Documentation hub](../Index.md)
