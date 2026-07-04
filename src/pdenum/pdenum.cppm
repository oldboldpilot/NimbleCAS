// NimbleCAS numerical methods for PDEs: finite differences, finite elements,
// finite volumes, and the method of lines.
// @author Olumuyiwa Oluwasanmi
//
// This module is the NUMERICAL/DISCRETIZATION counterpart to nimblecas.pde (which is the
// EXACT power-series-in-time evolution PDE solver). Here we DISCRETIZE a differential
// operator on a grid/mesh and solve the resulting linear system. The design keeps a hard,
// documented honesty boundary:
//
//   * The SPATIAL discretizations — finite-difference stencils, finite-element stiffness/mass
//     matrices, finite-volume face fluxes — and the linear SOLVES built from them are EXACT
//     over the field Q whenever the grid nodes and the data (source f, coefficients, boundary
//     values) are rational, and for FEM whenever the source is a POLYNOMIAL (the element
//     integrals ∫ f φ are then exact rationals). Every entry flows through Rational's
//     overflow-checked arithmetic, and the linear systems are solved by the exact rational
//     cores (nimblecas.bandsolve Thomas for tridiagonal systems, nimblecas.matrix Gauss-Jordan
//     for the dense/sparse-ish 2-D system). There is NO floating point and NO tolerance in any
//     of the spatial-discretization or solve paths.
//
//   * What is EXACT is the solution of the DISCRETE system, NOT the true PDE solution. The
//     discretization error (the difference between the discrete solution and the exact PDE
//     solution) is a separate, documented matter. For some special data — e.g. a quadratic
//     exact solution under the 3-point second-difference stencil, or a linear exact solution
//     under the 5-point Laplacian — the discrete nodal values COINCIDE with the exact solution
//     at the nodes, and the tests exploit that; but that is a property of the data, not a
//     claim that discretization == PDE in general.
//
//   * TIME stepping is NUMERICAL. The Crank-Nicolson stepper operates in double precision and
//     is explicitly marked so. Likewise, any statement about convergence order, stability, or a
//     CFL condition is a NUMERICAL-ANALYSIS statement, not an exact algebraic one, and is not
//     asserted by this module. The method-of-lines builders return the semi-discrete operator L
//     EXACTLY over Q (so it can be fed to the exact ODE machinery); only the wall-clock time
//     integration is floating point.
//
//   * Non-rational data or nonlinear PDEs are out of scope: they would be numerical or are
//     reported as not_implemented / domain_error on the railway. Nothing throws (Rule 32).

module;
#include <cassert>  // assert macro (unavailable via `import std`); active when !NDEBUG

export module nimblecas.pdenum;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.bandsolve;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// Discretization options.
// ---------------------------------------------------------------------------

// The finite-difference stencil orientation for a first derivative.
enum class DiffScheme : std::uint8_t { central, forward, backward };

// The kind of boundary condition imposed at a domain endpoint.
enum class BCKind : std::uint8_t { dirichlet, neumann };

// A boundary condition: a Dirichlet value u = value, or a Neumann value u' = value (the
// outward-flux convention is documented per solver). The stored Rational keeps the whole
// boundary datum exact.
struct BoundaryCondition {
    BCKind kind{BCKind::dirichlet};
    Rational value{};

    [[nodiscard]] static auto dirichlet(Rational v) -> BoundaryCondition {
        return BoundaryCondition{BCKind::dirichlet, std::move(v)};
    }
    [[nodiscard]] static auto neumann(Rational v) -> BoundaryCondition {
        return BoundaryCondition{BCKind::neumann, std::move(v)};
    }
};

// A rational scalar field of one variable, sampled exactly on the railway.
using RationalField1D = std::function<Result<Rational>(const Rational& x)>;
// A rational scalar field of two variables, sampled exactly on the railway.
using RationalField2D = std::function<Result<Rational>(const Rational& x, const Rational& y)>;

// ---------------------------------------------------------------------------
// Grid1D — a uniform rational grid a = x_0 < x_1 < ... < x_N = b on [a, b].
// ---------------------------------------------------------------------------
// n_intervals is N (the number of cells); there are N+1 nodes. The spacing h = (b - a)/N is
// exact over Q. make() requires N >= 1 and b > a (a strictly ordered, non-degenerate grid).
struct Grid1D {
    Rational a{};
    Rational b{};
    std::size_t n_intervals{};

    [[nodiscard]] static auto make(Rational a, Rational b, std::size_t n_intervals)
        -> Result<Grid1D> {
        if (n_intervals == 0) {
            return make_error<Grid1D>(MathError::domain_error);
        }
        auto diff = b.subtract(a);
        if (!diff) {
            return make_error<Grid1D>(diff.error());
        }
        if (diff->numerator() <= 0) {  // den > 0 always, so the sign is the numerator's
            return make_error<Grid1D>(MathError::domain_error);  // require b > a
        }
        return Grid1D{std::move(a), std::move(b), n_intervals};
    }

    [[nodiscard]] auto num_nodes() const noexcept -> std::size_t { return n_intervals + 1; }

    [[nodiscard]] auto spacing() const -> Result<Rational> {
        auto diff = b.subtract(a);
        if (!diff) {
            return make_error<Rational>(diff.error());
        }
        assert(n_intervals <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()));
        return diff->divide(Rational::from_int(static_cast<std::int64_t>(n_intervals)));
    }

    [[nodiscard]] auto node(std::size_t i) const -> Result<Rational> {
        if (i > n_intervals) {
            return make_error<Rational>(MathError::domain_error);
        }
        auto h = spacing();
        if (!h) {
            return make_error<Rational>(h.error());
        }
        assert(i <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()));
        auto off = Rational::from_int(static_cast<std::int64_t>(i)).multiply(*h);
        if (!off) {
            return make_error<Rational>(off.error());
        }
        return a.add(*off);
    }
};

// ---------------------------------------------------------------------------
// Grid2D — a uniform rational grid on the rectangle [ax, bx] x [ay, by].
// ---------------------------------------------------------------------------
// nx, ny are the interval counts in x and y; there are (nx+1)(ny+1) nodes.
struct Grid2D {
    Rational ax{};
    Rational bx{};
    Rational ay{};
    Rational by{};
    std::size_t nx{};
    std::size_t ny{};

    [[nodiscard]] static auto make(Rational ax, Rational bx, Rational ay, Rational by,
                                   std::size_t nx, std::size_t ny) -> Result<Grid2D> {
        if (nx == 0 || ny == 0) {
            return make_error<Grid2D>(MathError::domain_error);
        }
        auto dx = bx.subtract(ax);
        auto dy = by.subtract(ay);
        if (!dx) {
            return make_error<Grid2D>(dx.error());
        }
        if (!dy) {
            return make_error<Grid2D>(dy.error());
        }
        if (dx->numerator() <= 0 || dy->numerator() <= 0) {
            return make_error<Grid2D>(MathError::domain_error);
        }
        return Grid2D{std::move(ax), std::move(bx), std::move(ay), std::move(by), nx, ny};
    }

    [[nodiscard]] auto hx() const -> Result<Rational> {
        auto d = bx.subtract(ax);
        if (!d) {
            return make_error<Rational>(d.error());
        }
        assert(nx <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()));
        return d->divide(Rational::from_int(static_cast<std::int64_t>(nx)));
    }
    [[nodiscard]] auto hy() const -> Result<Rational> {
        auto d = by.subtract(ay);
        if (!d) {
            return make_error<Rational>(d.error());
        }
        assert(ny <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()));
        return d->divide(Rational::from_int(static_cast<std::int64_t>(ny)));
    }
    [[nodiscard]] auto node_x(std::size_t i) const -> Result<Rational> {
        if (i > nx) {
            return make_error<Rational>(MathError::domain_error);
        }
        auto h = hx();
        if (!h) {
            return make_error<Rational>(h.error());
        }
        assert(i <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()));
        auto off = Rational::from_int(static_cast<std::int64_t>(i)).multiply(*h);
        if (!off) {
            return make_error<Rational>(off.error());
        }
        return ax.add(*off);
    }
    [[nodiscard]] auto node_y(std::size_t j) const -> Result<Rational> {
        if (j > ny) {
            return make_error<Rational>(MathError::domain_error);
        }
        auto h = hy();
        if (!h) {
            return make_error<Rational>(h.error());
        }
        assert(j <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()));
        auto off = Rational::from_int(static_cast<std::int64_t>(j)).multiply(*h);
        if (!off) {
            return make_error<Rational>(off.error());
        }
        return ay.add(*off);
    }
};

// ---------------------------------------------------------------------------
// Mesh1D — a (possibly non-uniform) 1-D mesh of strictly increasing rational nodes.
// ---------------------------------------------------------------------------
// Used by the finite-element assembly, where element lengths may vary. nodes.size() >= 2.
struct Mesh1D {
    std::vector<Rational> nodes;  // strictly increasing, size >= 2

    [[nodiscard]] static auto from_nodes(std::vector<Rational> nodes) -> Result<Mesh1D> {
        if (nodes.size() < 2) {
            return make_error<Mesh1D>(MathError::domain_error);
        }
        for (std::size_t i = 1; i < nodes.size(); ++i) {
            auto d = nodes[i].subtract(nodes[i - 1]);
            if (!d) {
                return make_error<Mesh1D>(d.error());
            }
            if (d->numerator() <= 0) {
                return make_error<Mesh1D>(MathError::domain_error);  // not strictly increasing
            }
        }
        return Mesh1D{std::move(nodes)};
    }

    [[nodiscard]] static auto uniform(const Rational& a, const Rational& b, std::size_t n)
        -> Result<Mesh1D> {
        auto g = Grid1D::make(a, b, n);
        if (!g) {
            return make_error<Mesh1D>(g.error());
        }
        std::vector<Rational> pts;
        pts.reserve(g->num_nodes());
        for (std::size_t i = 0; i < g->num_nodes(); ++i) {
            auto x = g->node(i);
            if (!x) {
                return make_error<Mesh1D>(x.error());
            }
            pts.push_back(*x);
        }
        return Mesh1D{std::move(pts)};
    }

    [[nodiscard]] auto num_nodes() const noexcept -> std::size_t { return nodes.size(); }
    [[nodiscard]] auto num_elements() const noexcept -> std::size_t { return nodes.size() - 1; }
    [[nodiscard]] auto element_length(std::size_t e) const -> Result<Rational> {
        if (e + 1 >= nodes.size()) {
            return make_error<Rational>(MathError::domain_error);
        }
        return nodes[e + 1].subtract(nodes[e]);
    }
};

// ===========================================================================
// Public API — free functions (declarations; defined out-of-line below).
// ===========================================================================

// --- Finite difference (FDM) ------------------------------------------------

// The n x n first-derivative differentiation matrix on a uniform grid of spacing h. The
// interior rows use the requested scheme; a boundary row that would need a ghost point falls
// back to the one-sided (forward at the first row / backward at the last row) stencil so the
// matrix stays square without ghosts. central is 2nd-order interior, forward/backward 1st.
// Requires n >= 2 and h != 0, else domain_error.
[[nodiscard]] auto fdm_d1_matrix(std::size_t n, const Rational& h, DiffScheme scheme)
    -> Result<Matrix>;

// The n x n central second-derivative differentiation matrix tridiag(1, -2, 1)/h^2 on a
// uniform grid of spacing h (interior operator; homogeneous-Dirichlet boundaries are implied
// by truncation at the first/last row). Requires n >= 1 and h != 0, else domain_error.
[[nodiscard]] auto fdm_d2_matrix(std::size_t n, const Rational& h) -> Result<Matrix>;

// Solve the 1-D boundary-value problem -u''(x) = f(x) on [a, b] by the standard 3-point
// stencil on the uniform grid, EXACTLY over Q. f_nodal holds f sampled at EVERY grid node
// (num_nodes() entries; values at Dirichlet nodes are ignored). Dirichlet and Neumann
// conditions are supported at each end (the Neumann value is u'(a) at the left / u'(b) at the
// right, imposed by the 2nd-order ghost-node relation, which keeps the system tridiagonal).
// The tridiagonal system is solved by the exact Thomas algorithm (nimblecas.bandsolve). The
// full nodal solution (num_nodes() entries, boundary nodes included) is returned. A wrong
// f_nodal length is domain_error; a singular system (e.g. pure Neumann/Neumann, whose solution
// is only defined up to a constant) surfaces as domain_error from the solver.
[[nodiscard]] auto solve_poisson_1d(const Grid1D& grid, std::span<const Rational> f_nodal,
                                    const BoundaryCondition& left, const BoundaryCondition& right)
    -> Result<std::vector<Rational>>;

// The 5-point Laplacian operator matrix for -Δu on the interior of an (nx x ny)-interval
// rectangular grid with spacings hx, hy (homogeneous-Dirichlet coupling only; size
// (nx-1)(ny-1) square, lexicographic x-fastest ordering). Requires nx, ny >= 2 and hx, hy != 0.
[[nodiscard]] auto poisson_2d_laplacian(std::size_t nx, std::size_t ny, const Rational& hx,
                                        const Rational& hy) -> Result<Matrix>;

// Solve -Δu = f on the rectangle with Dirichlet data `boundary`, EXACTLY over Q, via the
// 5-point stencil and the exact rational dense solver (nimblecas.matrix). f and boundary are
// sampled exactly at rational points. Returns the interior nodal values as a (ny-1) x (nx-1)
// matrix (row j, column i => node (i+1, j+1)). Requires nx, ny >= 2 (a non-empty interior),
// else domain_error; any sampler error propagates.
[[nodiscard]] auto solve_poisson_2d(const Grid2D& grid, const RationalField2D& f,
                                    const RationalField2D& boundary) -> Result<Matrix>;

// --- Finite element (FEM) ---------------------------------------------------

// The global P1 (linear Lagrange) stiffness matrix K, K_ij = ∫ φ_i' φ_j', assembled element by
// element as (1/h_e)[[1,-1],[-1,1]]. On a uniform mesh the interior is tridiag(-1, 2, -1)/h
// exactly. Size num_nodes() x num_nodes(); no boundary conditions applied.
[[nodiscard]] auto fem_p1_stiffness(const Mesh1D& mesh) -> Result<Matrix>;

// The global P1 mass matrix M, M_ij = ∫ φ_i φ_j, assembled as (h_e/6)[[2,1],[1,2]]. Size
// num_nodes() x num_nodes(); no boundary conditions applied.
[[nodiscard]] auto fem_p1_mass(const Mesh1D& mesh) -> Result<Matrix>;

// The global P1 load vector b_i = ∫ f φ_i for a POLYNOMIAL source f (RationalPoly). Because
// f φ_i is a polynomial on each element, the element integrals are EXACT rationals (computed by
// exact antiderivative + evaluation). Size num_nodes().
[[nodiscard]] auto fem_p1_load(const Mesh1D& mesh, const RationalPoly& f)
    -> Result<std::vector<Rational>>;

// Solve the Galerkin P1 finite-element problem for -(a u')' + c u = f with constant rational a
// and c and polynomial source f, EXACTLY over Q. The global system (a K + c M) u = b is formed,
// Neumann data enters weakly as the natural boundary term (left: b_0 -= a*value; right:
// b_{N} += a*value; value = u' at that end), Dirichlet data is imposed by symmetric row/column
// elimination, and the reduced system is solved exactly (nimblecas.matrix). Returns the full
// nodal solution. A singular reduced system (e.g. pure homogeneous Neumann with c = 0) is
// domain_error.
[[nodiscard]] auto fem_p1_solve(const Mesh1D& mesh, const Rational& a_coeff,
                                const Rational& c_coeff, const RationalPoly& f,
                                const BoundaryCondition& left, const BoundaryCondition& right)
    -> Result<std::vector<Rational>>;

// The 3 x 3 P2 (quadratic Lagrange) element stiffness matrix (1/(3h))[[7,-8,1],[-8,16,-8],
// [1,-8,7]] for an element of length h (node order: left, mid, right). Exact over Q. h != 0.
[[nodiscard]] auto fem_p2_element_stiffness(const Rational& h) -> Result<Matrix>;

// The 3 x 3 P2 element mass matrix (h/30)[[4,2,-1],[2,16,2],[-1,2,4]] for an element of length
// h. Exact over Q.
[[nodiscard]] auto fem_p2_element_mass(const Rational& h) -> Result<Matrix>;

// --- Finite volume (FVM) ----------------------------------------------------

// Solve the 1-D steady diffusion / conservation law -(k u')' = f by a cell-centered finite
// volume method on the uniform grid, EXACTLY over Q. There are N = n_intervals cells; f_cells
// holds the cell-averaged source per cell (N entries). Constant diffusivity k != 0. Dirichlet
// values u_left, u_right are imposed at the domain endpoints x = a, x = b via half-cell
// face-flux reconstruction. The resulting rational tridiagonal system is solved by the exact
// Thomas algorithm. Returns the N cell-center values u_i (at x = a + (i + 1/2) h). A wrong
// f_cells length or k == 0 is domain_error.
[[nodiscard]] auto fvm_solve_diffusion_1d(const Grid1D& grid, const Rational& k,
                                          std::span<const Rational> f_cells,
                                          const Rational& u_left, const Rational& u_right)
    -> Result<std::vector<Rational>>;

// --- Method of lines (MoL) --------------------------------------------------

// The semi-discrete heat operator L for u_t = alpha u_xx: L = alpha/h^2 tridiag(1, -2, 1),
// an n x n matrix over the interior (homogeneous-Dirichlet) nodes. EXACT over Q — this is the
// operator du/dt = L u that the exact ODE machinery can integrate. Requires n >= 1, h != 0.
[[nodiscard]] auto mol_heat_operator(std::size_t n, const Rational& h, const Rational& alpha)
    -> Result<Matrix>;

// The semi-discrete advection operator L for u_t = -c u_x: L = -c D1 with the chosen FDM scheme
// (upwind = backward for c > 0). EXACT over Q. Requires n >= 2, h != 0.
[[nodiscard]] auto mol_advection_operator(std::size_t n, const Rational& h, const Rational& c,
                                          DiffScheme scheme) -> Result<Matrix>;

// Advance one Crank-Nicolson time step of du/dt = L u: (I - dt/2 L) u^{n+1} = (I + dt/2 L) u^n.
//
// NUMERICAL (this is the one deliberately floating-point routine in the module): L is converted
// to double, and the step solves a double-precision dense linear system. Crank-Nicolson is
// unconditionally stable for the diffusion operator (a NUMERICAL-ANALYSIS fact, not asserted
// here). Requires L square with L.rows() == u.size(); else domain_error. A singular (I - dt/2 L)
// is domain_error.
[[nodiscard]] auto crank_nicolson_step(const Matrix& l, std::span<const double> u, double dt)
    -> Result<std::vector<double>>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// acc + a*b, exactly, propagating any Rational-op error. The recurring shape in RHS assembly.
[[nodiscard]] auto fma_r(const Rational& acc, const Rational& a, const Rational& b)
    -> Result<Rational> {
    auto p = a.multiply(b);
    if (!p) {
        return make_error<Rational>(p.error());
    }
    return acc.add(*p);
}

// The product of a small fixed list of rationals, exactly.
[[nodiscard]] auto rprod(std::initializer_list<Rational> xs) -> Result<Rational> {
    Rational acc = Rational::from_int(1);
    for (const auto& x : xs) {
        auto p = acc.multiply(x);
        if (!p) {
            return make_error<Rational>(p.error());
        }
        acc = *p;
    }
    return acc;
}

// A rows x cols grid of zero rationals (Rational{} default is 0/1).
[[nodiscard]] auto zero_grid(std::size_t rows, std::size_t cols)
    -> std::vector<std::vector<Rational>> {
    return std::vector<std::vector<Rational>>(rows, std::vector<Rational>(cols));
}

// g[r][c] += v, exactly.
[[nodiscard]] auto add_into(std::vector<std::vector<Rational>>& g, std::size_t r, std::size_t c,
                            const Rational& v) -> Result<void> {
    auto s = g[r][c].add(v);
    if (!s) {
        return make_error<void>(s.error());
    }
    g[r][c] = *s;
    return {};
}

// A dense Matrix copied into a mutable row grid (for in-place BC elimination).
[[nodiscard]] auto to_grid(const Matrix& m) -> std::vector<std::vector<Rational>> {
    auto g = zero_grid(m.rows(), m.cols());
    for (std::size_t i = 0; i < m.rows(); ++i) {
        for (std::size_t j = 0; j < m.cols(); ++j) {
            g[i][j] = m.at(i, j);
        }
    }
    return g;
}

// A column matrix (k x 1) from a rational vector.
[[nodiscard]] auto column(std::span<const Rational> v) -> Result<Matrix> {
    std::vector<std::vector<Rational>> rows;
    rows.reserve(v.size());
    for (const auto& e : v) {
        rows.push_back({e});
    }
    return Matrix::from_rows(std::move(rows));
}

// p(x) by Horner over Q, exactly.
[[nodiscard]] auto poly_eval(const RationalPoly& p, const Rational& x) -> Result<Rational> {
    Rational acc;  // 0/1
    const auto co = p.coefficients();
    for (std::size_t k = co.size(); k-- > 0;) {
        auto m = acc.multiply(x);
        if (!m) {
            return make_error<Rational>(m.error());
        }
        auto a = m->add(co[k]);
        if (!a) {
            return make_error<Rational>(a.error());
        }
        acc = *a;
    }
    return acc;
}

// The antiderivative of p with zero constant term (a[i+1] = c[i]/(i+1)).
[[nodiscard]] auto poly_integral(const RationalPoly& p) -> Result<RationalPoly> {
    const auto co = p.coefficients();
    if (co.empty()) {
        return RationalPoly{};
    }
    std::vector<Rational> c(co.size() + 1);  // c[0] = 0
    for (std::size_t i = 0; i < co.size(); ++i) {
        assert(i < static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()));
        auto t = co[i].divide(Rational::from_int(static_cast<std::int64_t>(i) + 1));
        if (!t) {
            return make_error<RationalPoly>(t.error());
        }
        c[i + 1] = *t;
    }
    return RationalPoly::from_coeffs(std::move(c));
}

// The exact definite integral ∫_lo^hi p(x) dx.
[[nodiscard]] auto poly_definite(const RationalPoly& p, const Rational& lo, const Rational& hi)
    -> Result<Rational> {
    auto anti = poly_integral(p);
    if (!anti) {
        return make_error<Rational>(anti.error());
    }
    auto phi = poly_eval(*anti, hi);
    if (!phi) {
        return make_error<Rational>(phi.error());
    }
    auto plo = poly_eval(*anti, lo);
    if (!plo) {
        return make_error<Rational>(plo.error());
    }
    return phi->subtract(*plo);
}

// Rational -> double (numerator/denominator). Used only in the NUMERICAL Crank-Nicolson path.
[[nodiscard]] auto to_double(const Rational& r) noexcept -> double {
    return static_cast<double>(r.numerator()) / static_cast<double>(r.denominator());
}

// Solve a dense double system A x = b (row-major a, size n) by Gaussian elimination with partial
// pivoting. NUMERICAL. A (near-)zero pivot yields domain_error (treated as singular).
[[nodiscard]] auto solve_dense_double(std::vector<double> a, std::size_t n, std::vector<double> b)
    -> Result<std::vector<double>> {
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t piv = col;
        double best = std::abs(a[col * n + col]);
        for (std::size_t r = col + 1; r < n; ++r) {
            const double v = std::abs(a[r * n + col]);
            if (v > best) {
                best = v;
                piv = r;
            }
        }
        if (best == 0.0) {
            return make_error<std::vector<double>>(MathError::domain_error);  // singular
        }
        if (piv != col) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(a[col * n + j], a[piv * n + j]);
            }
            std::swap(b[col], b[piv]);
        }
        const double d = a[col * n + col];
        for (std::size_t r = col + 1; r < n; ++r) {
            const double f = a[r * n + col] / d;
            if (f != 0.0) {
                for (std::size_t j = col; j < n; ++j) {
                    a[r * n + j] -= f * a[col * n + j];
                }
                b[r] -= f * b[col];
            }
        }
    }
    std::vector<double> x(n);
    for (std::size_t i = n; i-- > 0;) {
        double s = b[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            s -= a[i * n + j] * x[j];
        }
        x[i] = s / a[i * n + i];
    }
    return x;
}

}  // namespace

// --- Finite difference ------------------------------------------------------

auto fdm_d1_matrix(std::size_t n, const Rational& h, DiffScheme scheme) -> Result<Matrix> {
    if (n < 2 || h.is_zero()) {
        return make_error<Matrix>(MathError::domain_error);
    }
    auto invh = Rational::from_int(1).divide(h);  // 1/h
    if (!invh) {
        return make_error<Matrix>(invh.error());
    }
    auto ninvh = invh->negate();  // -1/h
    if (!ninvh) {
        return make_error<Matrix>(ninvh.error());
    }
    auto two_h = h.multiply(Rational::from_int(2));  // 2h
    if (!two_h) {
        return make_error<Matrix>(two_h.error());
    }
    auto invh2 = Rational::from_int(1).divide(*two_h);  // 1/(2h)
    if (!invh2) {
        return make_error<Matrix>(invh2.error());
    }
    auto ninvh2 = invh2->negate();  // -1/(2h)
    if (!ninvh2) {
        return make_error<Matrix>(ninvh2.error());
    }
    auto g = zero_grid(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        const bool has_left = i > 0;
        const bool has_right = i + 1 < n;
        // Decide the stencil for this row: the requested scheme in the interior, one-sided at a
        // boundary row that lacks the needed neighbour.
        bool use_central = false;
        bool use_forward = false;
        bool use_backward = false;
        switch (scheme) {
            case DiffScheme::central:
                if (has_left && has_right) {
                    use_central = true;
                } else if (!has_left) {
                    use_forward = true;
                } else {
                    use_backward = true;
                }
                break;
            case DiffScheme::forward:
                if (has_right) {
                    use_forward = true;
                } else {
                    use_backward = true;
                }
                break;
            case DiffScheme::backward:
                if (has_left) {
                    use_backward = true;
                } else {
                    use_forward = true;
                }
                break;
        }
        if (use_central) {
            g[i][i - 1] = *ninvh2;
            g[i][i + 1] = *invh2;
        } else if (use_forward) {
            g[i][i] = *ninvh;
            g[i][i + 1] = *invh;
        } else {  // backward
            g[i][i - 1] = *ninvh;
            g[i][i] = *invh;
        }
    }
    return Matrix::from_rows(std::move(g));
}

auto fdm_d2_matrix(std::size_t n, const Rational& h) -> Result<Matrix> {
    if (n == 0 || h.is_zero()) {
        return make_error<Matrix>(MathError::domain_error);
    }
    auto h2 = h.multiply(h);
    if (!h2) {
        return make_error<Matrix>(h2.error());
    }
    auto inv = Rational::from_int(1).divide(*h2);  // 1/h^2
    if (!inv) {
        return make_error<Matrix>(inv.error());
    }
    auto diag = Rational::from_int(-2).multiply(*inv);  // -2/h^2
    if (!diag) {
        return make_error<Matrix>(diag.error());
    }
    const Rational off = *inv;  // 1/h^2
    auto g = zero_grid(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        g[i][i] = *diag;
        if (i > 0) {
            g[i][i - 1] = off;
        }
        if (i + 1 < n) {
            g[i][i + 1] = off;
        }
    }
    return Matrix::from_rows(std::move(g));
}

auto solve_poisson_1d(const Grid1D& grid, std::span<const Rational> f_nodal,
                      const BoundaryCondition& left, const BoundaryCondition& right)
    -> Result<std::vector<Rational>> {
    const std::size_t N = grid.n_intervals;
    const std::size_t nn = grid.num_nodes();
    if (f_nodal.size() != nn) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    auto h = grid.spacing();
    if (!h) {
        return make_error<std::vector<Rational>>(h.error());
    }
    auto h2 = h->multiply(*h);
    if (!h2) {
        return make_error<std::vector<Rational>>(h2.error());
    }

    // The full nodal solution; Dirichlet nodes are known immediately.
    std::vector<Rational> u(nn);
    if (left.kind == BCKind::dirichlet) {
        u[0] = left.value;
    }
    if (right.kind == BCKind::dirichlet) {
        u[N] = right.value;
    }

    // The unknown node range [lo, hi]: a Dirichlet end removes its node, a Neumann end keeps it.
    const std::size_t lo = (left.kind == BCKind::dirichlet) ? 1u : 0u;
    const std::size_t hi = (right.kind == BCKind::dirichlet) ? (N - 1) : N;
    if (lo > hi) {  // no interior unknowns (e.g. N == 1 with Dirichlet at both ends)
        return u;
    }
    const std::size_t m = hi - lo + 1;

    // Assemble the tridiagonal system over the unknown nodes; every equation couples only to its
    // neighbours, so the operator stays tridiagonal even with the ghost-node Neumann rows.
    std::vector<Rational> sub(m == 0 ? 0 : m - 1);
    std::vector<Rational> diag(m);
    std::vector<Rational> super(m == 0 ? 0 : m - 1);
    std::vector<Rational> rhs(m);

    for (std::size_t j = 0; j < m; ++j) {
        const std::size_t k = lo + j;  // global node index
        const bool is_left_neumann = (k == 0 && left.kind == BCKind::neumann);
        const bool is_right_neumann = (k == N && right.kind == BCKind::neumann);

        Rational D;
        Rational lc;  // coefficient of u_{k-1}
        Rational rc;  // coefficient of u_{k+1}
        Rational s;   // right-hand side

        if (is_left_neumann) {
            // Ghost u_{-1} = u_1 - 2 h g gives 2 u_0 - 2 u_1 = h^2 f_0 - 2 h g.
            D = Rational::from_int(2);
            lc = Rational{};  // no left neighbour
            rc = Rational::from_int(-2);
            auto t1 = h2->multiply(f_nodal[0]);
            if (!t1) {
                return make_error<std::vector<Rational>>(t1.error());
            }
            auto t2 = rprod({Rational::from_int(2), *h, left.value});
            if (!t2) {
                return make_error<std::vector<Rational>>(t2.error());
            }
            auto sv = t1->subtract(*t2);
            if (!sv) {
                return make_error<std::vector<Rational>>(sv.error());
            }
            s = *sv;
        } else if (is_right_neumann) {
            // Ghost u_{N+1} = u_{N-1} + 2 h g gives -2 u_{N-1} + 2 u_N = h^2 f_N + 2 h g.
            D = Rational::from_int(2);
            lc = Rational::from_int(-2);
            rc = Rational{};  // no right neighbour
            auto t1 = h2->multiply(f_nodal[N]);
            if (!t1) {
                return make_error<std::vector<Rational>>(t1.error());
            }
            auto t2 = rprod({Rational::from_int(2), *h, right.value});
            if (!t2) {
                return make_error<std::vector<Rational>>(t2.error());
            }
            auto sv = t1->add(*t2);
            if (!sv) {
                return make_error<std::vector<Rational>>(sv.error());
            }
            s = *sv;
        } else {  // interior node: -u_{k-1} + 2 u_k - u_{k+1} = h^2 f_k
            D = Rational::from_int(2);
            lc = Rational::from_int(-1);
            rc = Rational::from_int(-1);
            auto t1 = h2->multiply(f_nodal[k]);
            if (!t1) {
                return make_error<std::vector<Rational>>(t1.error());
            }
            s = *t1;
        }

        // Left neighbour: an unknown couples into the subdiagonal; a known (Dirichlet) neighbour
        // moves to the RHS as s -= lc * u_known.
        if (k > 0) {
            const std::size_t kl = k - 1;
            if (kl >= lo) {
                sub[j - 1] = lc;
            } else {
                auto nlc = lc.negate();
                if (!nlc) {
                    return make_error<std::vector<Rational>>(nlc.error());
                }
                auto sv = fma_r(s, *nlc, u[kl]);  // s + (-lc)*u_known = s - lc*u_known
                if (!sv) {
                    return make_error<std::vector<Rational>>(sv.error());
                }
                s = *sv;
            }
        }
        // Right neighbour, symmetrically.
        if (k < N) {
            const std::size_t kr = k + 1;
            if (kr <= hi) {
                super[j] = rc;
            } else {
                auto nrc = rc.negate();
                if (!nrc) {
                    return make_error<std::vector<Rational>>(nrc.error());
                }
                auto sv = fma_r(s, *nrc, u[kr]);
                if (!sv) {
                    return make_error<std::vector<Rational>>(sv.error());
                }
                s = *sv;
            }
        }

        diag[j] = D;
        rhs[j] = s;
    }

    auto sol = solve_tridiagonal(sub, diag, super, rhs);
    if (!sol) {
        return make_error<std::vector<Rational>>(sol.error());
    }
    for (std::size_t j = 0; j < m; ++j) {
        u[lo + j] = (*sol)[j];
    }
    return u;
}

auto poisson_2d_laplacian(std::size_t nx, std::size_t ny, const Rational& hx, const Rational& hy)
    -> Result<Matrix> {
    if (nx < 2 || ny < 2 || hx.is_zero() || hy.is_zero()) {
        return make_error<Matrix>(MathError::domain_error);
    }
    auto hx2 = hx.multiply(hx);
    auto hy2 = hy.multiply(hy);
    if (!hx2) {
        return make_error<Matrix>(hx2.error());
    }
    if (!hy2) {
        return make_error<Matrix>(hy2.error());
    }
    auto ix = Rational::from_int(1).divide(*hx2);  // 1/hx^2
    auto iy = Rational::from_int(1).divide(*hy2);  // 1/hy^2
    if (!ix) {
        return make_error<Matrix>(ix.error());
    }
    if (!iy) {
        return make_error<Matrix>(iy.error());
    }
    auto nix = ix->negate();
    auto niy = iy->negate();
    if (!nix) {
        return make_error<Matrix>(nix.error());
    }
    if (!niy) {
        return make_error<Matrix>(niy.error());
    }
    // center = 2/hx^2 + 2/hy^2.
    auto center = rprod({Rational::from_int(2), *ix});
    if (!center) {
        return make_error<Matrix>(center.error());
    }
    {
        auto tmp = rprod({Rational::from_int(2), *iy});
        if (!tmp) {
            return make_error<Matrix>(tmp.error());
        }
        auto c = center->add(*tmp);
        if (!c) {
            return make_error<Matrix>(c.error());
        }
        center = *c;
    }
    const std::size_t mx = nx - 1;
    const std::size_t my = ny - 1;
    const std::size_t total = mx * my;
    auto g = zero_grid(total, total);
    for (std::size_t jj = 0; jj < my; ++jj) {
        for (std::size_t ii = 0; ii < mx; ++ii) {
            const std::size_t p = jj * mx + ii;
            g[p][p] = *center;
            if (ii > 0) {
                g[p][p - 1] = *nix;
            }
            if (ii + 1 < mx) {
                g[p][p + 1] = *nix;
            }
            if (jj > 0) {
                g[p][p - mx] = *niy;
            }
            if (jj + 1 < my) {
                g[p][p + mx] = *niy;
            }
        }
    }
    return Matrix::from_rows(std::move(g));
}

auto solve_poisson_2d(const Grid2D& grid, const RationalField2D& f,
                      const RationalField2D& boundary) -> Result<Matrix> {
    if (grid.nx < 2 || grid.ny < 2) {
        return make_error<Matrix>(MathError::domain_error);  // empty interior
    }
    if (!f || !boundary) {
        return make_error<Matrix>(MathError::domain_error);
    }
    auto hx = grid.hx();
    auto hy = grid.hy();
    if (!hx) {
        return make_error<Matrix>(hx.error());
    }
    if (!hy) {
        return make_error<Matrix>(hy.error());
    }
    auto hx2 = hx->multiply(*hx);
    auto hy2 = hy->multiply(*hy);
    if (!hx2) {
        return make_error<Matrix>(hx2.error());
    }
    if (!hy2) {
        return make_error<Matrix>(hy2.error());
    }
    auto ix = Rational::from_int(1).divide(*hx2);
    auto iy = Rational::from_int(1).divide(*hy2);
    if (!ix) {
        return make_error<Matrix>(ix.error());
    }
    if (!iy) {
        return make_error<Matrix>(iy.error());
    }
    auto nix = ix->negate();
    auto niy = iy->negate();
    if (!nix) {
        return make_error<Matrix>(nix.error());
    }
    if (!niy) {
        return make_error<Matrix>(niy.error());
    }
    Rational center;
    {
        auto a = rprod({Rational::from_int(2), *ix});
        auto b = rprod({Rational::from_int(2), *iy});
        if (!a) {
            return make_error<Matrix>(a.error());
        }
        if (!b) {
            return make_error<Matrix>(b.error());
        }
        auto c = a->add(*b);
        if (!c) {
            return make_error<Matrix>(c.error());
        }
        center = *c;
    }
    const std::size_t mx = grid.nx - 1;
    const std::size_t my = grid.ny - 1;
    const std::size_t total = mx * my;
    auto g = zero_grid(total, total);
    std::vector<Rational> rhs(total);

    for (std::size_t jj = 0; jj < my; ++jj) {
        auto yj = grid.node_y(jj + 1);
        if (!yj) {
            return make_error<Matrix>(yj.error());
        }
        for (std::size_t ii = 0; ii < mx; ++ii) {
            auto xi = grid.node_x(ii + 1);
            if (!xi) {
                return make_error<Matrix>(xi.error());
            }
            const std::size_t p = jj * mx + ii;
            g[p][p] = center;

            auto fv = f(*xi, *yj);
            if (!fv) {
                return make_error<Matrix>(fv.error());
            }
            Rational s = *fv;

            // For each neighbour: an interior neighbour couples into the matrix (coeff -1/h^2);
            // a boundary neighbour moves its Dirichlet value to the RHS as s += (1/h^2)*g_bdry.
            // x-left
            if (ii > 0) {
                g[p][p - 1] = *nix;
            } else {
                auto x0 = grid.node_x(0);
                if (!x0) {
                    return make_error<Matrix>(x0.error());
                }
                auto bv = boundary(*x0, *yj);
                if (!bv) {
                    return make_error<Matrix>(bv.error());
                }
                auto sv = fma_r(s, *ix, *bv);
                if (!sv) {
                    return make_error<Matrix>(sv.error());
                }
                s = *sv;
            }
            // x-right
            if (ii + 1 < mx) {
                g[p][p + 1] = *nix;
            } else {
                auto xn = grid.node_x(grid.nx);
                if (!xn) {
                    return make_error<Matrix>(xn.error());
                }
                auto bv = boundary(*xn, *yj);
                if (!bv) {
                    return make_error<Matrix>(bv.error());
                }
                auto sv = fma_r(s, *ix, *bv);
                if (!sv) {
                    return make_error<Matrix>(sv.error());
                }
                s = *sv;
            }
            // y-down
            if (jj > 0) {
                g[p][p - mx] = *niy;
            } else {
                auto y0 = grid.node_y(0);
                if (!y0) {
                    return make_error<Matrix>(y0.error());
                }
                auto bv = boundary(*xi, *y0);
                if (!bv) {
                    return make_error<Matrix>(bv.error());
                }
                auto sv = fma_r(s, *iy, *bv);
                if (!sv) {
                    return make_error<Matrix>(sv.error());
                }
                s = *sv;
            }
            // y-up
            if (jj + 1 < my) {
                g[p][p + mx] = *niy;
            } else {
                auto yn = grid.node_y(grid.ny);
                if (!yn) {
                    return make_error<Matrix>(yn.error());
                }
                auto bv = boundary(*xi, *yn);
                if (!bv) {
                    return make_error<Matrix>(bv.error());
                }
                auto sv = fma_r(s, *iy, *bv);
                if (!sv) {
                    return make_error<Matrix>(sv.error());
                }
                s = *sv;
            }
            rhs[p] = s;
        }
    }

    auto a = Matrix::from_rows(std::move(g));
    if (!a) {
        return make_error<Matrix>(a.error());
    }
    auto bcol = column(rhs);
    if (!bcol) {
        return make_error<Matrix>(bcol.error());
    }
    auto sol = a->solve(*bcol);  // exact rational dense solve; singular => domain_error
    if (!sol) {
        return make_error<Matrix>(sol.error());
    }
    // Reshape the (total x 1) solution into (my x mx).
    std::vector<std::vector<Rational>> out(my, std::vector<Rational>(mx));
    for (std::size_t jj = 0; jj < my; ++jj) {
        for (std::size_t ii = 0; ii < mx; ++ii) {
            out[jj][ii] = sol->at(jj * mx + ii, 0);
        }
    }
    return Matrix::from_rows(std::move(out));
}

// --- Finite element ---------------------------------------------------------

auto fem_p1_stiffness(const Mesh1D& mesh) -> Result<Matrix> {
    const std::size_t nn = mesh.num_nodes();
    auto g = zero_grid(nn, nn);
    for (std::size_t e = 0; e < mesh.num_elements(); ++e) {
        auto he = mesh.element_length(e);
        if (!he) {
            return make_error<Matrix>(he.error());
        }
        auto ke = Rational::from_int(1).divide(*he);  // 1/h_e
        if (!ke) {
            return make_error<Matrix>(ke.error());
        }
        auto nke = ke->negate();
        if (!nke) {
            return make_error<Matrix>(nke.error());
        }
        // Element stiffness (1/h_e)[[1,-1],[-1,1]] scattered into the two element nodes.
        if (auto rr = add_into(g, e, e, *ke); !rr) {
            return make_error<Matrix>(rr.error());
        }
        if (auto rr = add_into(g, e, e + 1, *nke); !rr) {
            return make_error<Matrix>(rr.error());
        }
        if (auto rr = add_into(g, e + 1, e, *nke); !rr) {
            return make_error<Matrix>(rr.error());
        }
        if (auto rr = add_into(g, e + 1, e + 1, *ke); !rr) {
            return make_error<Matrix>(rr.error());
        }
    }
    return Matrix::from_rows(std::move(g));
}

auto fem_p1_mass(const Mesh1D& mesh) -> Result<Matrix> {
    const std::size_t nn = mesh.num_nodes();
    auto g = zero_grid(nn, nn);
    for (std::size_t e = 0; e < mesh.num_elements(); ++e) {
        auto he = mesh.element_length(e);
        if (!he) {
            return make_error<Matrix>(he.error());
        }
        auto md = he->divide(Rational::from_int(3));  // h_e/3 (diagonal)
        if (!md) {
            return make_error<Matrix>(md.error());
        }
        auto mo = he->divide(Rational::from_int(6));  // h_e/6 (off-diagonal)
        if (!mo) {
            return make_error<Matrix>(mo.error());
        }
        // Element mass (h_e/6)[[2,1],[1,2]] scattered into the two element nodes.
        if (auto rr = add_into(g, e, e, *md); !rr) {
            return make_error<Matrix>(rr.error());
        }
        if (auto rr = add_into(g, e + 1, e + 1, *md); !rr) {
            return make_error<Matrix>(rr.error());
        }
        if (auto rr = add_into(g, e, e + 1, *mo); !rr) {
            return make_error<Matrix>(rr.error());
        }
        if (auto rr = add_into(g, e + 1, e, *mo); !rr) {
            return make_error<Matrix>(rr.error());
        }
    }
    return Matrix::from_rows(std::move(g));
}

auto fem_p1_load(const Mesh1D& mesh, const RationalPoly& f) -> Result<std::vector<Rational>> {
    const std::size_t nn = mesh.num_nodes();
    std::vector<Rational> b(nn);
    for (std::size_t e = 0; e < mesh.num_elements(); ++e) {
        const Rational& x0 = mesh.nodes[e];
        const Rational& x1 = mesh.nodes[e + 1];
        auto he = x1.subtract(x0);
        if (!he) {
            return make_error<std::vector<Rational>>(he.error());
        }
        auto invh = Rational::from_int(1).divide(*he);  // 1/h_e
        if (!invh) {
            return make_error<std::vector<Rational>>(invh.error());
        }
        auto ninvh = invh->negate();
        if (!ninvh) {
            return make_error<std::vector<Rational>>(ninvh.error());
        }
        // phi0 = (x1 - x)/h_e = x1/h_e - (1/h_e) x.
        auto x1h = x1.divide(*he);
        if (!x1h) {
            return make_error<std::vector<Rational>>(x1h.error());
        }
        auto phi0 = RationalPoly::from_coeffs({*x1h, *ninvh});
        // phi1 = (x - x0)/h_e = -x0/h_e + (1/h_e) x.
        auto x0h = x0.divide(*he);
        if (!x0h) {
            return make_error<std::vector<Rational>>(x0h.error());
        }
        auto nx0h = x0h->negate();
        if (!nx0h) {
            return make_error<std::vector<Rational>>(nx0h.error());
        }
        auto phi1 = RationalPoly::from_coeffs({*nx0h, *invh});

        auto f0 = f.multiply(phi0);
        if (!f0) {
            return make_error<std::vector<Rational>>(f0.error());
        }
        auto f1 = f.multiply(phi1);
        if (!f1) {
            return make_error<std::vector<Rational>>(f1.error());
        }
        auto i0 = poly_definite(*f0, x0, x1);
        if (!i0) {
            return make_error<std::vector<Rational>>(i0.error());
        }
        auto i1 = poly_definite(*f1, x0, x1);
        if (!i1) {
            return make_error<std::vector<Rational>>(i1.error());
        }
        auto s0 = b[e].add(*i0);
        if (!s0) {
            return make_error<std::vector<Rational>>(s0.error());
        }
        b[e] = *s0;
        auto s1 = b[e + 1].add(*i1);
        if (!s1) {
            return make_error<std::vector<Rational>>(s1.error());
        }
        b[e + 1] = *s1;
    }
    return b;
}

auto fem_p1_solve(const Mesh1D& mesh, const Rational& a_coeff, const Rational& c_coeff,
                  const RationalPoly& f, const BoundaryCondition& left,
                  const BoundaryCondition& right) -> Result<std::vector<Rational>> {
    const std::size_t nn = mesh.num_nodes();
    auto k = fem_p1_stiffness(mesh);
    if (!k) {
        return make_error<std::vector<Rational>>(k.error());
    }
    auto mm = fem_p1_mass(mesh);
    if (!mm) {
        return make_error<std::vector<Rational>>(mm.error());
    }
    auto ak = k->scale(a_coeff);
    if (!ak) {
        return make_error<std::vector<Rational>>(ak.error());
    }
    auto cm = mm->scale(c_coeff);
    if (!cm) {
        return make_error<std::vector<Rational>>(cm.error());
    }
    auto amat = ak->add(*cm);  // A = a K + c M
    if (!amat) {
        return make_error<std::vector<Rational>>(amat.error());
    }
    auto load = fem_p1_load(mesh, f);
    if (!load) {
        return make_error<std::vector<Rational>>(load.error());
    }

    auto ag = to_grid(*amat);
    std::vector<Rational> b = *load;

    // Neumann natural boundary terms: left b_0 -= a*g_left; right b_{N} += a*g_right.
    if (left.kind == BCKind::neumann) {
        auto t = a_coeff.multiply(left.value);
        if (!t) {
            return make_error<std::vector<Rational>>(t.error());
        }
        auto s = b[0].subtract(*t);
        if (!s) {
            return make_error<std::vector<Rational>>(s.error());
        }
        b[0] = *s;
    }
    if (right.kind == BCKind::neumann) {
        auto t = a_coeff.multiply(right.value);
        if (!t) {
            return make_error<std::vector<Rational>>(t.error());
        }
        auto s = b[nn - 1].add(*t);
        if (!s) {
            return make_error<std::vector<Rational>>(s.error());
        }
        b[nn - 1] = *s;
    }

    // Collect the Dirichlet constraints.
    struct Dbc {
        std::size_t idx;
        Rational val;
    };
    std::vector<Dbc> dir;
    if (left.kind == BCKind::dirichlet) {
        dir.push_back({0, left.value});
    }
    if (right.kind == BCKind::dirichlet) {
        dir.push_back({nn - 1, right.value});
    }

    // Symmetric elimination: first move the Dirichlet columns to the RHS (using the ORIGINAL A),
    // then zero their rows/columns and pin the diagonal to 1 with b = value.
    for (const auto& d : dir) {
        for (std::size_t i = 0; i < nn; ++i) {
            auto t = ag[i][d.idx].multiply(d.val);
            if (!t) {
                return make_error<std::vector<Rational>>(t.error());
            }
            auto s = b[i].subtract(*t);
            if (!s) {
                return make_error<std::vector<Rational>>(s.error());
            }
            b[i] = *s;
        }
    }
    for (const auto& d : dir) {
        for (std::size_t i = 0; i < nn; ++i) {
            ag[d.idx][i] = Rational{};
            ag[i][d.idx] = Rational{};
        }
        ag[d.idx][d.idx] = Rational::from_int(1);
        b[d.idx] = d.val;
    }

    auto am = Matrix::from_rows(std::move(ag));
    if (!am) {
        return make_error<std::vector<Rational>>(am.error());
    }
    auto bcol = column(b);
    if (!bcol) {
        return make_error<std::vector<Rational>>(bcol.error());
    }
    auto sol = am->solve(*bcol);  // exact; singular => domain_error
    if (!sol) {
        return make_error<std::vector<Rational>>(sol.error());
    }
    std::vector<Rational> u(nn);
    for (std::size_t i = 0; i < nn; ++i) {
        u[i] = sol->at(i, 0);
    }
    return u;
}

namespace {

// Build a 3 x 3 element matrix (scale) * [[integer pattern]].
[[nodiscard]] auto p2_element(const Rational& scale,
                              const std::array<std::array<std::int64_t, 3>, 3>& pat)
    -> Result<Matrix> {
    std::vector<std::vector<Rational>> g(3, std::vector<Rational>(3));
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            auto v = scale.multiply(Rational::from_int(pat[i][j]));
            if (!v) {
                return make_error<Matrix>(v.error());
            }
            g[i][j] = *v;
        }
    }
    return Matrix::from_rows(std::move(g));
}

}  // namespace

auto fem_p2_element_stiffness(const Rational& h) -> Result<Matrix> {
    if (h.is_zero()) {
        return make_error<Matrix>(MathError::domain_error);
    }
    auto three_h = h.multiply(Rational::from_int(3));  // 3h
    if (!three_h) {
        return make_error<Matrix>(three_h.error());
    }
    auto scale = Rational::from_int(1).divide(*three_h);  // 1/(3h)
    if (!scale) {
        return make_error<Matrix>(scale.error());
    }
    return p2_element(*scale, {{{7, -8, 1}, {-8, 16, -8}, {1, -8, 7}}});
}

auto fem_p2_element_mass(const Rational& h) -> Result<Matrix> {
    if (h.is_zero()) {
        return make_error<Matrix>(MathError::domain_error);
    }
    auto scale = h.divide(Rational::from_int(30));  // h/30
    if (!scale) {
        return make_error<Matrix>(scale.error());
    }
    return p2_element(*scale, {{{4, 2, -1}, {2, 16, 2}, {-1, 2, 4}}});
}

// --- Finite volume ----------------------------------------------------------

auto fvm_solve_diffusion_1d(const Grid1D& grid, const Rational& k, std::span<const Rational> f_cells,
                            const Rational& u_left, const Rational& u_right)
    -> Result<std::vector<Rational>> {
    const std::size_t N = grid.n_intervals;
    if (k.is_zero()) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    if (f_cells.size() != N) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    auto dx = grid.spacing();
    if (!dx) {
        return make_error<std::vector<Rational>>(dx.error());
    }
    auto dx2 = dx->multiply(*dx);
    if (!dx2) {
        return make_error<std::vector<Rational>>(dx2.error());
    }
    auto dx2_over_k = dx2->divide(k);  // dx^2 / k (scales the cell source term)
    if (!dx2_over_k) {
        return make_error<std::vector<Rational>>(dx2_over_k.error());
    }

    std::vector<Rational> sub(N == 0 ? 0 : N - 1);
    std::vector<Rational> diag(N);
    std::vector<Rational> super(N == 0 ? 0 : N - 1);
    std::vector<Rational> rhs(N);

    for (std::size_t i = 0; i < N; ++i) {
        // Base interior balance (k/dx) tridiag(-1, 2, -1) u = f dx, scaled by dx/k to a pure
        // tridiag with rhs (dx^2/k) f. A boundary cell adds +1 to its diagonal (half-cell face)
        // and 2*u_boundary to its RHS.
        Rational D = Rational::from_int(2);
        auto base = dx2_over_k->multiply(f_cells[i]);
        if (!base) {
            return make_error<std::vector<Rational>>(base.error());
        }
        Rational s = *base;
        if (i == 0) {
            auto d = D.add(Rational::from_int(1));
            if (!d) {
                return make_error<std::vector<Rational>>(d.error());
            }
            D = *d;
            auto sv = fma_r(s, Rational::from_int(2), u_left);
            if (!sv) {
                return make_error<std::vector<Rational>>(sv.error());
            }
            s = *sv;
        }
        if (i == N - 1) {
            auto d = D.add(Rational::from_int(1));
            if (!d) {
                return make_error<std::vector<Rational>>(d.error());
            }
            D = *d;
            auto sv = fma_r(s, Rational::from_int(2), u_right);
            if (!sv) {
                return make_error<std::vector<Rational>>(sv.error());
            }
            s = *sv;
        }
        diag[i] = D;
        rhs[i] = s;
        if (i > 0) {
            sub[i - 1] = Rational::from_int(-1);
        }
        if (i + 1 < N) {
            super[i] = Rational::from_int(-1);
        }
    }

    auto sol = solve_tridiagonal(sub, diag, super, rhs);
    if (!sol) {
        return make_error<std::vector<Rational>>(sol.error());
    }
    return *sol;
}

// --- Method of lines --------------------------------------------------------

auto mol_heat_operator(std::size_t n, const Rational& h, const Rational& alpha) -> Result<Matrix> {
    auto d2 = fdm_d2_matrix(n, h);
    if (!d2) {
        return make_error<Matrix>(d2.error());
    }
    return d2->scale(alpha);  // L = alpha * tridiag(1,-2,1)/h^2
}

auto mol_advection_operator(std::size_t n, const Rational& h, const Rational& c, DiffScheme scheme)
    -> Result<Matrix> {
    auto d1 = fdm_d1_matrix(n, h, scheme);
    if (!d1) {
        return make_error<Matrix>(d1.error());
    }
    auto nc = c.negate();  // L = -c D1
    if (!nc) {
        return make_error<Matrix>(nc.error());
    }
    return d1->scale(*nc);
}

auto crank_nicolson_step(const Matrix& l, std::span<const double> u, double dt)
    -> Result<std::vector<double>> {
    if (!l.is_square() || l.rows() != u.size()) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    const std::size_t n = l.rows();
    const double half = dt * 0.5;
    // A = I - (dt/2) L ; B = I + (dt/2) L (both in double — this is the NUMERICAL step).
    std::vector<double> amat(n * n);
    std::vector<double> bmat(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const double lij = to_double(l.at(i, j));
            const double id = (i == j) ? 1.0 : 0.0;
            amat[i * n + j] = id - half * lij;
            bmat[i * n + j] = id + half * lij;
        }
    }
    // rhs = B u.
    std::vector<double> rhs(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double acc = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            acc += bmat[i * n + j] * u[j];
        }
        rhs[i] = acc;
    }
    return solve_dense_double(std::move(amat), n, std::move(rhs));
}

}  // namespace nimblecas
