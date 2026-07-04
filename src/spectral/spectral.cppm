// NimbleCAS spectral methods: spectral-Galerkin, pseudospectral collocation,
// Fourier spectral, and their spectral-element / discontinuous-Galerkin analogues.
// @author Olumuyiwa Oluwasanmi
//
// Conforms to config/cpp_details.txt: C++23 modules, `import std`, trailing return
// types, no owning raw pointers, std::expected railway error handling (no exceptions),
// [[nodiscard]] on every observer.
//
// SCOPE. This module is about SPECTRAL discretisations — global (or per-element)
// expansions in orthogonal bases with the associated differentiation and boundary-value
// solvers. It is deliberately distinct from:
//   * nimblecas.pde     — exact linear evolution PDEs by a power series in time
//                         (Cauchy-Kovalevskaya); no boundary conditions, not spectral.
//   * nimblecas.pdenum  — low-order FDM/FEM/FVM stencils.
// Nothing here duplicates those: the constructs below are Galerkin / collocation /
// Fourier / spectral-element / DG discretisations.
//
// ===========================================================================
//  HONESTY BOUNDARY (documented and true — see each function).
// ===========================================================================
//  EXACT over Q (rational arithmetic, no rounding) for POLYNOMIAL data on rational
//  meshes, because every inner product used is an exact rational integral over [-1, 1]
//  and every orthogonal polynomial has exact rational coefficients:
//    * legendre_coefficients / legendre_from_coefficients        (forward / inverse)
//    * chebyshev_coefficients / chebyshev_from_coefficients       (forward / inverse)
//    * legendre_differentiate_coefficients / chebyshev_differentiate_coefficients
//    * galerkin_poisson         (-u'' = f, u(±1)=0, Legendre-Galerkin, exact system)
//    * spectral_element_legendre / spectral_element_reconstruct   (piecewise, C^0)
//    * dg_advection_rhs / dg_advection_step (the SEMI-DISCRETE spatial operator and a
//      single explicit Euler update are exact rationals; see NUMERICAL note below)
//
//  NUMERICAL (double precision; irrational nodes and/or complex exponentials) — any
//  spectral-accuracy / convergence claim about these is a numerical property, NOT
//  exactness:
//    * chebyshev_gauss_lobatto_nodes / chebyshev_differentiation_matrix /
//      chebyshev_collocation_poisson   (nodes x_j = cos(pi j / N) are irrational)
//    * fourier_grid / fourier_differentiation_matrix / fourier_spectral_derivative /
//      fourier_periodic_solve          (complex exponentials, a compact O(N^2) DFT)
//
//  DG caveat: the spatial operator dg_advection_rhs and a single dg_advection_step are
//  EXACT over Q. Marching many steps to a final time is time-STEPPING — forward Euler is
//  a first-order NUMERICAL approximation of the continuous evolution even though each
//  individual rational step is computed exactly. We do not claim the time-marched result
//  is the exact PDE solution.

export module nimblecas.spectral;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.orthopoly;
import nimblecas.matrix;

export namespace nimblecas {

// ===========================================================================
//  Small exact utility.
// ===========================================================================

// Evaluate a RationalPoly p at a rational point x by Horner's scheme; exact over Q.
// The zero polynomial evaluates to 0. Fails only on Rational overflow.
[[nodiscard]] auto evaluate_poly(const RationalPoly& p, const Rational& x) -> Result<Rational>;

// ===========================================================================
//  EXACT spectral-Galerkin over Q.
// ===========================================================================

// Forward Legendre transform of a polynomial f: the exact coefficients c_k with
// f = sum_{k=0}^{deg f} c_k P_k(x), c_k = <f, P_k> / <P_k, P_k>, and the inner product
// <g, h> = integral_{-1}^{1} g h dx. Returns c_0..c_{deg f} (higher coefficients are
// exactly zero for polynomial data). The zero polynomial yields an empty vector. EXACT.
[[nodiscard]] auto legendre_coefficients(const RationalPoly& f) -> Result<std::vector<Rational>>;

// Inverse Legendre transform: reconstruct sum_k coeffs[k] P_k(x) exactly. EXACT.
[[nodiscard]] auto legendre_from_coefficients(std::span<const Rational> coeffs)
    -> Result<RationalPoly>;

// Forward Chebyshev-T transform of a polynomial f: the exact coefficients c_k with
// f = sum_k c_k T_k(x), c_k = <f, T_k>_w / <T_k, T_k>_w under the Chebyshev weight
// w(x) = 1/sqrt(1 - x^2). The shared factor pi cancels between numerator and denominator,
// so every c_k is an exact rational. Empty for the zero polynomial. EXACT.
[[nodiscard]] auto chebyshev_coefficients(const RationalPoly& f) -> Result<std::vector<Rational>>;

// Inverse Chebyshev-T transform: reconstruct sum_k coeffs[k] T_k(x) exactly. EXACT.
[[nodiscard]] auto chebyshev_from_coefficients(std::span<const Rational> coeffs)
    -> Result<RationalPoly>;

// Spectral differentiation in Legendre coefficient space. Given a = {a_k} with
// f = sum a_k P_k, returns b = {b_k} with f' = sum b_k P_k, via the exact recurrence
// b_k = (2k+1) sum_{p>k, p+k odd} a_p. Result is trimmed of trailing zeros. EXACT.
[[nodiscard]] auto legendre_differentiate_coefficients(std::span<const Rational> coeffs)
    -> Result<std::vector<Rational>>;

// Spectral differentiation in Chebyshev-T coefficient space. Given a = {a_k} with
// f = sum a_k T_k, returns b = {b_k} with f' = sum b_k T_k, via the exact recurrence
// b_k = (2 / c_k) sum_{p>k, p+k odd} p a_p, with c_0 = 2 and c_k = 1 for k >= 1.
// Result is trimmed of trailing zeros. EXACT.
[[nodiscard]] auto chebyshev_differentiate_coefficients(std::span<const Rational> coeffs)
    -> Result<std::vector<Rational>>;

// Legendre spectral-Galerkin solution of the boundary value problem
//     -u''(x) = f(x) on (-1, 1),   u(-1) = u(1) = 0,
// with f a polynomial. Uses the homogeneous-Dirichlet modal basis phi_k = P_k - P_{k+2}
// (each vanishes at x = ±1), assembles the exact rational stiffness matrix
// S_{jk} = integral phi_j' phi_k' dx and load F_j = integral f phi_j dx, and solves
// S u = F exactly over Q. For polynomial f the exact solution lies in the chosen span, so
// the returned RationalPoly is the exact solution. f == 0 returns the zero polynomial.
// EXACT.
[[nodiscard]] auto galerkin_poisson(const RationalPoly& f) -> Result<RationalPoly>;

// ===========================================================================
//  NUMERICAL Chebyshev collocation / pseudospectral (double precision).
// ===========================================================================

// The n+1 Chebyshev-Gauss-Lobatto nodes x_j = cos(pi j / n), j = 0..n (x_0 = 1 down to
// x_n = -1). These are IRRATIONAL, hence double. Requires n >= 1 (domain_error otherwise).
[[nodiscard]] auto chebyshev_gauss_lobatto_nodes(std::size_t n) -> Result<std::vector<double>>;

// The (n+1) x (n+1) Chebyshev collocation differentiation matrix D at the Gauss-Lobatto
// nodes: (D v)_i approximates u'(x_i) when v_j = u(x_j). Off-diagonal entries use the
// classical formula; the diagonal is set by the negative-sum trick D_ii = -sum_{k!=i} D_ik.
// NUMERICAL (double). Requires n >= 1.
[[nodiscard]] auto chebyshev_differentiation_matrix(std::size_t n)
    -> Result<std::vector<std::vector<double>>>;

// Apply a dense row-major matrix a (each inner vector is a row) to a vector v.
// Fails with domain_error on a ragged matrix or a column/length mismatch. NUMERICAL.
[[nodiscard]] auto apply_dense_matrix(const std::vector<std::vector<double>>& a,
                                      std::span<const double> v) -> Result<std::vector<double>>;

// Pseudospectral solve of -u'' = f on (-1, 1), u(±1) = 0, by imposing the ODE at the
// interior Chebyshev-Gauss-Lobatto collocation nodes and the homogeneous Dirichlet
// conditions at the two boundary nodes, then solving the dense double system. Returns u at
// all n+1 nodes (boundary entries are exactly 0). Requires n >= 2. NUMERICAL (double);
// its spectral accuracy is a numerical property.
[[nodiscard]] auto chebyshev_collocation_poisson(const std::function<double(double)>& f,
                                                 std::size_t n) -> Result<std::vector<double>>;

// ===========================================================================
//  NUMERICAL Fourier spectral (double / complex exponentials).
// ===========================================================================

// The n equispaced periodic grid points x_j = 2*pi*j/n on [0, 2*pi), j = 0..n-1.
// Requires n >= 1. NUMERICAL (double).
[[nodiscard]] auto fourier_grid(std::size_t n) -> Result<std::vector<double>>;

// The n x n Fourier differentiation matrix on the equispaced periodic grid (EVEN n):
// D_jk = 0.5 * (-1)^(j-k) * cot(pi (j-k) / n) for j != k, and D_jj = 0. Requires n even
// and n >= 2 (odd n has a different closed form and is rejected with domain_error).
// NUMERICAL (double).
[[nodiscard]] auto fourier_differentiation_matrix(std::size_t n)
    -> Result<std::vector<std::vector<double>>>;

// Spectral derivative of a periodic sampled function via a direct discrete Fourier
// transform: transform, multiply mode k by i*k' (k' the signed wavenumber, Nyquist zeroed
// for this odd-order derivative), invert, take the real part. This is a COMPACT O(N^2) DFT;
// a fast O(N log N) FFT would compute the same result asymptotically faster — we implement
// the honest O(N^2) form for clarity. Requires n >= 2. NUMERICAL (double / complex).
[[nodiscard]] auto fourier_spectral_derivative(std::span<const double> samples)
    -> Result<std::vector<double>>;

// Periodic boundary-value solve of the Helmholtz-type problem u - u'' = f on [0, 2*pi)
// with periodic boundary conditions, done in Fourier space: hat_u_k = hat_f_k / (1 + k'^2).
// This operator is always invertible (no compatibility condition). Returns u sampled on the
// same grid as f. Requires n >= 2. NUMERICAL (double / complex, O(N^2) DFT).
[[nodiscard]] auto fourier_periodic_solve(std::span<const double> f_samples)
    -> Result<std::vector<double>>;

// ===========================================================================
//  ANALOGUES — spectral element (C^0) and discontinuous Galerkin.
// ===========================================================================

// 1-D SPECTRAL ELEMENT representation of a polynomial f on a rational mesh. The mesh is a
// strictly increasing sequence of rational break points z_0 < z_1 < ... < z_M (M+1 nodes,
// M elements). For each element [z_e, z_{e+1}] the restriction of f is affinely mapped to
// the reference interval [-1, 1] and expanded in the Legendre basis (exact forward
// transform), yielding that element's reference coefficients. Because all elements carry
// the SAME underlying polynomial f, the piecewise representation is automatically C^0
// (adjacent reconstructions agree at shared nodes). EXACT over Q. Requires >= 2 mesh nodes,
// strictly increasing (domain_error otherwise).
[[nodiscard]] auto spectral_element_legendre(const RationalPoly& f, std::span<const Rational> mesh)
    -> Result<std::vector<std::vector<Rational>>>;

// Reconstruct, for each element, the physical-space RationalPoly from its reference
// Legendre coefficients produced by spectral_element_legendre (inverse of the affine map).
// EXACT over Q. Requires elem_coeffs.size() + 1 == mesh.size() and a valid increasing mesh.
[[nodiscard]] auto spectral_element_reconstruct(
    const std::vector<std::vector<Rational>>& elem_coeffs, std::span<const Rational> mesh)
    -> Result<std::vector<RationalPoly>>;

// 1-D DISCONTINUOUS GALERKIN semi-discrete spatial operator for the linear advection /
// conservation law u_t + a u_x = 0 (a > 0), using a per-element modal Legendre basis on the
// reference interval and the UPWIND numerical flux. `state[e]` holds element e's reference
// Legendre coefficients (all elements share the same polynomial degree p-1). `mesh` is the
// M+1 rational break points, `a` the (rational) advection speed, `inflow` the prescribed
// solution value at the left inflow boundary. Returns d(state)/dt element by element:
//     dU^e_n/dt = [ a sum_m U^e_m S_{nm} - (Fhat_R P_n(1) - Fhat_L P_n(-1)) ] * (2n+1) / h_e,
// S_{nm} = integral_{-1}^{1} P_m P_n' dξ, Fhat the upwind interface fluxes, h_e the element
// length. EXACT over Q for rational data on a rational mesh. Requires >= 1 element, a
// matching mesh, and a uniform, non-empty coefficient count; assumes a > 0 (upwinding from
// the left). NUMERICAL note applies only to time marching, not to this operator.
[[nodiscard]] auto dg_advection_rhs(const std::vector<std::vector<Rational>>& state,
                                    std::span<const Rational> mesh, const Rational& a,
                                    const Rational& inflow)
    -> Result<std::vector<std::vector<Rational>>>;

// One explicit (forward) Euler DG step: state + dt * dg_advection_rhs(state, ...). Each step
// is an EXACT rational update, but repeated marching is a first-order NUMERICAL time
// integration of the continuous evolution (see the module honesty note). EXACT per step.
[[nodiscard]] auto dg_advection_step(const std::vector<std::vector<Rational>>& state,
                                     std::span<const Rational> mesh, const Rational& a,
                                     const Rational& inflow, const Rational& dt)
    -> Result<std::vector<std::vector<Rational>>>;

}  // namespace nimblecas

// ===========================================================================
//  Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// --- exact rational helpers -------------------------------------------------

// integral_{-1}^{1} p(x) dx, exact. integral x^i = 2/(i+1) for even i, 0 for odd i.
[[nodiscard]] auto integrate_unit(const RationalPoly& p) -> Result<Rational> {
    Rational acc;  // 0/1
    const auto coeffs = p.coefficients();
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        if (i % 2 != 0) {
            continue;  // odd powers integrate to 0 over the symmetric interval
        }
        auto w = Rational::make(2, static_cast<std::int64_t>(i) + 1);
        if (!w) {
            return make_error<Rational>(w.error());
        }
        auto term = coeffs[i].multiply(*w);
        if (!term) {
            return make_error<Rational>(term.error());
        }
        auto sum = acc.add(*term);
        if (!sum) {
            return make_error<Rational>(sum.error());
        }
        acc = *sum;
    }
    return acc;
}

// The reduced Chebyshev-weight moment W(x^n) = integral_{-1}^{1} x^n / sqrt(1-x^2) dx / pi.
// W = 0 for odd n; for n = 2m, W = prod_{i=1}^{m} (2i-1)/(2i) (= C(2m,m)/4^m). Exact.
[[nodiscard]] auto weighted_moment(std::size_t n) -> Result<Rational> {
    if (n % 2 != 0) {
        return Rational{};  // 0
    }
    Rational acc = Rational::from_int(1);
    const std::size_t m = n / 2;
    for (std::size_t i = 1; i <= m; ++i) {
        auto t = Rational::make(static_cast<std::int64_t>(2 * i - 1),
                                static_cast<std::int64_t>(2 * i));
        if (!t) {
            return make_error<Rational>(t.error());
        }
        auto prod = acc.multiply(*t);
        if (!prod) {
            return make_error<Rational>(prod.error());
        }
        acc = *prod;
    }
    return acc;
}

// integral_{-1}^{1} p(x) / sqrt(1-x^2) dx, with the shared pi factor divided out. Exact.
[[nodiscard]] auto weighted_integrate(const RationalPoly& p) -> Result<Rational> {
    Rational acc;  // 0/1
    const auto coeffs = p.coefficients();
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        auto w = weighted_moment(i);
        if (!w) {
            return make_error<Rational>(w.error());
        }
        if (w->is_zero()) {
            continue;
        }
        auto term = coeffs[i].multiply(*w);
        if (!term) {
            return make_error<Rational>(term.error());
        }
        auto sum = acc.add(*term);
        if (!sum) {
            return make_error<Rational>(sum.error());
        }
        acc = *sum;
    }
    return acc;
}

// f(alpha + beta*x) as an exact RationalPoly, via Horner over polynomial arithmetic.
[[nodiscard]] auto compose_affine(const RationalPoly& f, const Rational& alpha,
                                  const Rational& beta) -> Result<RationalPoly> {
    const std::int64_t d = f.degree();
    if (d < 0) {
        return RationalPoly{};  // f == 0
    }
    // y = alpha + beta*x (trimmed automatically by from_coeffs).
    const RationalPoly y = RationalPoly::from_coeffs(std::vector<Rational>{alpha, beta});
    RationalPoly acc = RationalPoly::constant(f.coefficient(static_cast<std::size_t>(d)));
    for (std::int64_t i = d - 1; i >= 0; --i) {
        auto times = acc.multiply(y);
        if (!times) {
            return make_error<RationalPoly>(times.error());
        }
        auto added = times->add(RationalPoly::constant(f.coefficient(static_cast<std::size_t>(i))));
        if (!added) {
            return make_error<RationalPoly>(added.error());
        }
        acc = std::move(*added);
    }
    return acc;
}

// Trim trailing zero Rationals so a coefficient vector is canonical.
[[nodiscard]] auto trim_rationals(std::vector<Rational> c) -> std::vector<Rational> {
    while (!c.empty() && c.back().is_zero()) {
        c.pop_back();
    }
    return c;
}

// --- numeric (double) helpers ----------------------------------------------

// Dense Gaussian elimination with partial pivoting: solve A x = b (A is n x n row-major as
// a vector of rows). Fails with domain_error on a (near-)singular pivot. NUMERICAL.
[[nodiscard]] auto solve_dense(std::vector<std::vector<double>> a, std::vector<double> b)
    -> Result<std::vector<double>> {
    const std::size_t n = a.size();
    for (const auto& row : a) {
        if (row.size() != n) {
            return make_error<std::vector<double>>(MathError::domain_error);
        }
    }
    if (b.size() != n) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    for (std::size_t col = 0; col < n; ++col) {
        // Partial pivot: largest magnitude at or below the diagonal.
        std::size_t pivot = col;
        double best = std::abs(a[col][col]);
        for (std::size_t r = col + 1; r < n; ++r) {
            const double v = std::abs(a[r][col]);
            if (v > best) {
                best = v;
                pivot = r;
            }
        }
        if (best < 1e-14) {
            return make_error<std::vector<double>>(MathError::domain_error);  // singular
        }
        if (pivot != col) {
            std::swap(a[col], a[pivot]);
            std::swap(b[col], b[pivot]);
        }
        const double diag = a[col][col];
        for (std::size_t r = col + 1; r < n; ++r) {
            const double factor = a[r][col] / diag;
            if (factor == 0.0) {
                continue;
            }
            for (std::size_t j = col; j < n; ++j) {
                a[r][j] = std::fma(-factor, a[col][j], a[r][j]);
            }
            b[r] = std::fma(-factor, b[col], b[r]);
        }
    }
    // Back substitution.
    std::vector<double> x(n, 0.0);
    for (std::size_t i = n; i-- > 0;) {
        double acc = b[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            acc = std::fma(-a[i][j], x[j], acc);
        }
        x[i] = acc / a[i][i];
    }
    return x;
}

// C = A * B for two dense row-major matrices. Assumes conforming dimensions (caller-checked).
[[nodiscard]] auto matmul(const std::vector<std::vector<double>>& lhs,
                          const std::vector<std::vector<double>>& rhs)
    -> std::vector<std::vector<double>> {
    const std::size_t m = lhs.size();
    const std::size_t k = rhs.size();
    const std::size_t n = (k == 0) ? 0 : rhs[0].size();
    std::vector<std::vector<double>> out(m, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t p = 0; p < k; ++p) {
            const double lip = lhs[i][p];
            for (std::size_t j = 0; j < n; ++j) {
                out[i][j] = std::fma(lip, rhs[p][j], out[i][j]);
            }
        }
    }
    return out;
}

// Forward discrete Fourier transform (unnormalised): X_k = sum_j x_j exp(-2*pi*i*k*j/N).
[[nodiscard]] auto dft_forward(std::span<const std::complex<double>> x)
    -> std::vector<std::complex<double>> {
    const std::size_t n = x.size();
    const double twopi = 2.0 * std::numbers::pi;
    std::vector<std::complex<double>> out(n);
    for (std::size_t k = 0; k < n; ++k) {
        std::complex<double> acc{0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) {
            const double angle = -twopi * (static_cast<double>(k) * static_cast<double>(j)) /
                                 static_cast<double>(n);
            acc += x[j] * std::complex<double>(std::cos(angle), std::sin(angle));
        }
        out[k] = acc;
    }
    return out;
}

// Inverse discrete Fourier transform: x_j = (1/N) sum_k X_k exp(2*pi*i*k*j/N).
[[nodiscard]] auto dft_inverse(std::span<const std::complex<double>> x)
    -> std::vector<std::complex<double>> {
    const std::size_t n = x.size();
    const double twopi = 2.0 * std::numbers::pi;
    std::vector<std::complex<double>> out(n);
    for (std::size_t j = 0; j < n; ++j) {
        std::complex<double> acc{0.0, 0.0};
        for (std::size_t k = 0; k < n; ++k) {
            const double angle = twopi * (static_cast<double>(k) * static_cast<double>(j)) /
                                 static_cast<double>(n);
            acc += x[k] * std::complex<double>(std::cos(angle), std::sin(angle));
        }
        out[j] = acc / static_cast<double>(n);
    }
    return out;
}

// Signed wavenumber for DFT index k on N points: k for k <= N/2, else k - N.
[[nodiscard]] auto signed_wavenumber(std::size_t k, std::size_t n) -> long long {
    const long long kk = static_cast<long long>(k);
    const long long nn = static_cast<long long>(n);
    return (kk <= nn / 2) ? kk : kk - nn;
}

}  // namespace

// --- exact utility ----------------------------------------------------------

auto evaluate_poly(const RationalPoly& p, const Rational& x) -> Result<Rational> {
    const std::int64_t d = p.degree();
    if (d < 0) {
        return Rational{};  // zero polynomial
    }
    Rational acc = p.coefficient(static_cast<std::size_t>(d));
    for (std::int64_t i = d - 1; i >= 0; --i) {
        auto scaled = acc.multiply(x);
        if (!scaled) {
            return make_error<Rational>(scaled.error());
        }
        auto sum = scaled->add(p.coefficient(static_cast<std::size_t>(i)));
        if (!sum) {
            return make_error<Rational>(sum.error());
        }
        acc = *sum;
    }
    return acc;
}

// --- exact spectral-Galerkin ------------------------------------------------

auto legendre_coefficients(const RationalPoly& f) -> Result<std::vector<Rational>> {
    const std::int64_t d = f.degree();
    if (d < 0) {
        return std::vector<Rational>{};  // f == 0
    }
    std::vector<Rational> out(static_cast<std::size_t>(d) + 1);
    for (std::int64_t k = 0; k <= d; ++k) {
        auto pk = legendre(k);
        if (!pk) {
            return make_error<std::vector<Rational>>(pk.error());
        }
        auto fpk = f.multiply(*pk);
        if (!fpk) {
            return make_error<std::vector<Rational>>(fpk.error());
        }
        auto num = integrate_unit(*fpk);
        if (!num) {
            return make_error<std::vector<Rational>>(num.error());
        }
        auto pkpk = pk->multiply(*pk);
        if (!pkpk) {
            return make_error<std::vector<Rational>>(pkpk.error());
        }
        auto den = integrate_unit(*pkpk);  // = 2/(2k+1), never zero
        if (!den) {
            return make_error<std::vector<Rational>>(den.error());
        }
        auto ck = num->divide(*den);
        if (!ck) {
            return make_error<std::vector<Rational>>(ck.error());
        }
        out[static_cast<std::size_t>(k)] = *ck;
    }
    return out;
}

auto legendre_from_coefficients(std::span<const Rational> coeffs) -> Result<RationalPoly> {
    RationalPoly acc;  // zero
    for (std::size_t k = 0; k < coeffs.size(); ++k) {
        if (coeffs[k].is_zero()) {
            continue;
        }
        auto pk = legendre(static_cast<std::int64_t>(k));
        if (!pk) {
            return make_error<RationalPoly>(pk.error());
        }
        auto scaled = pk->scale(coeffs[k]);
        if (!scaled) {
            return make_error<RationalPoly>(scaled.error());
        }
        auto sum = acc.add(*scaled);
        if (!sum) {
            return make_error<RationalPoly>(sum.error());
        }
        acc = std::move(*sum);
    }
    return acc;
}

auto chebyshev_coefficients(const RationalPoly& f) -> Result<std::vector<Rational>> {
    const std::int64_t d = f.degree();
    if (d < 0) {
        return std::vector<Rational>{};  // f == 0
    }
    std::vector<Rational> out(static_cast<std::size_t>(d) + 1);
    for (std::int64_t k = 0; k <= d; ++k) {
        auto tk = chebyshev_t(k);
        if (!tk) {
            return make_error<std::vector<Rational>>(tk.error());
        }
        auto ftk = f.multiply(*tk);
        if (!ftk) {
            return make_error<std::vector<Rational>>(ftk.error());
        }
        auto num = weighted_integrate(*ftk);
        if (!num) {
            return make_error<std::vector<Rational>>(num.error());
        }
        auto tktk = tk->multiply(*tk);
        if (!tktk) {
            return make_error<std::vector<Rational>>(tktk.error());
        }
        auto den = weighted_integrate(*tktk);  // = 1 (k=0) or 1/2 (k>=1), never zero
        if (!den) {
            return make_error<std::vector<Rational>>(den.error());
        }
        auto ck = num->divide(*den);
        if (!ck) {
            return make_error<std::vector<Rational>>(ck.error());
        }
        out[static_cast<std::size_t>(k)] = *ck;
    }
    return out;
}

auto chebyshev_from_coefficients(std::span<const Rational> coeffs) -> Result<RationalPoly> {
    RationalPoly acc;  // zero
    for (std::size_t k = 0; k < coeffs.size(); ++k) {
        if (coeffs[k].is_zero()) {
            continue;
        }
        auto tk = chebyshev_t(static_cast<std::int64_t>(k));
        if (!tk) {
            return make_error<RationalPoly>(tk.error());
        }
        auto scaled = tk->scale(coeffs[k]);
        if (!scaled) {
            return make_error<RationalPoly>(scaled.error());
        }
        auto sum = acc.add(*scaled);
        if (!sum) {
            return make_error<RationalPoly>(sum.error());
        }
        acc = std::move(*sum);
    }
    return acc;
}

auto legendre_differentiate_coefficients(std::span<const Rational> coeffs)
    -> Result<std::vector<Rational>> {
    const std::size_t n = coeffs.size();
    if (n <= 1) {
        return std::vector<Rational>{};  // derivative of a constant / zero is 0
    }
    std::vector<Rational> out(n - 1);
    for (std::size_t k = 0; k + 1 < n; ++k) {
        Rational sum;  // 0
        for (std::size_t p = k + 1; p < n; p += 2) {  // p + k odd
            auto s = sum.add(coeffs[p]);
            if (!s) {
                return make_error<std::vector<Rational>>(s.error());
            }
            sum = *s;
        }
        auto bk = sum.multiply(Rational::from_int(2 * static_cast<std::int64_t>(k) + 1));
        if (!bk) {
            return make_error<std::vector<Rational>>(bk.error());
        }
        out[k] = *bk;
    }
    return trim_rationals(std::move(out));
}

auto chebyshev_differentiate_coefficients(std::span<const Rational> coeffs)
    -> Result<std::vector<Rational>> {
    const std::size_t n = coeffs.size();
    if (n <= 1) {
        return std::vector<Rational>{};
    }
    std::vector<Rational> out(n - 1);
    for (std::size_t k = 0; k + 1 < n; ++k) {
        Rational sum;  // 0
        for (std::size_t p = k + 1; p < n; p += 2) {  // p + k odd
            auto term = coeffs[p].multiply(Rational::from_int(static_cast<std::int64_t>(p)));
            if (!term) {
                return make_error<std::vector<Rational>>(term.error());
            }
            auto s = sum.add(*term);
            if (!s) {
                return make_error<std::vector<Rational>>(s.error());
            }
            sum = *s;
        }
        // b_k = (2 / c_k) * sum, with c_0 = 2 (factor 1) and c_k = 1 for k >= 1 (factor 2).
        const Rational factor = (k == 0) ? Rational::from_int(1) : Rational::from_int(2);
        auto bk = sum.multiply(factor);
        if (!bk) {
            return make_error<std::vector<Rational>>(bk.error());
        }
        out[k] = *bk;
    }
    return trim_rationals(std::move(out));
}

auto galerkin_poisson(const RationalPoly& f) -> Result<RationalPoly> {
    const std::int64_t d = f.degree();
    if (d < 0) {
        return RationalPoly{};  // f == 0 => u == 0
    }
    const std::size_t nbasis = static_cast<std::size_t>(d) + 1;  // phi_0..phi_d
    // Build the homogeneous-Dirichlet basis phi_k = P_k - P_{k+2} and its derivative.
    std::vector<RationalPoly> phi(nbasis);
    std::vector<RationalPoly> dphi(nbasis);
    for (std::size_t k = 0; k < nbasis; ++k) {
        auto pk = legendre(static_cast<std::int64_t>(k));
        if (!pk) {
            return make_error<RationalPoly>(pk.error());
        }
        auto pk2 = legendre(static_cast<std::int64_t>(k) + 2);
        if (!pk2) {
            return make_error<RationalPoly>(pk2.error());
        }
        auto pk_diff = pk->subtract(*pk2);
        if (!pk_diff) {
            return make_error<RationalPoly>(pk_diff.error());
        }
        auto der = pk_diff->derivative();
        if (!der) {
            return make_error<RationalPoly>(der.error());
        }
        phi[k] = std::move(*pk_diff);
        dphi[k] = std::move(*der);
    }
    // Assemble the exact stiffness matrix S and load vector F.
    std::vector<std::vector<Rational>> srows(nbasis, std::vector<Rational>(nbasis));
    std::vector<std::vector<Rational>> frows(nbasis, std::vector<Rational>(1));
    for (std::size_t j = 0; j < nbasis; ++j) {
        for (std::size_t k = 0; k < nbasis; ++k) {
            auto prod = dphi[j].multiply(dphi[k]);
            if (!prod) {
                return make_error<RationalPoly>(prod.error());
            }
            auto s = integrate_unit(*prod);
            if (!s) {
                return make_error<RationalPoly>(s.error());
            }
            srows[j][k] = *s;
        }
        auto fprod = f.multiply(phi[j]);
        if (!fprod) {
            return make_error<RationalPoly>(fprod.error());
        }
        auto fj = integrate_unit(*fprod);
        if (!fj) {
            return make_error<RationalPoly>(fj.error());
        }
        frows[j][0] = *fj;
    }
    auto smat = Matrix::from_rows(std::move(srows));
    if (!smat) {
        return make_error<RationalPoly>(smat.error());
    }
    auto fmat = Matrix::from_rows(std::move(frows));
    if (!fmat) {
        return make_error<RationalPoly>(fmat.error());
    }
    auto sol = smat->solve(*fmat);  // exact rational solve; SPD, never singular
    if (!sol) {
        return make_error<RationalPoly>(sol.error());
    }
    // Reconstruct u = sum_k u_k phi_k.
    RationalPoly u;  // zero
    for (std::size_t k = 0; k < nbasis; ++k) {
        auto scaled = phi[k].scale(sol->at(k, 0));
        if (!scaled) {
            return make_error<RationalPoly>(scaled.error());
        }
        auto sum = u.add(*scaled);
        if (!sum) {
            return make_error<RationalPoly>(sum.error());
        }
        u = std::move(*sum);
    }
    return u;
}

// --- numerical Chebyshev collocation ---------------------------------------

auto chebyshev_gauss_lobatto_nodes(std::size_t n) -> Result<std::vector<double>> {
    if (n < 1) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    std::vector<double> nodes(n + 1);
    for (std::size_t j = 0; j <= n; ++j) {
        nodes[j] = std::cos(std::numbers::pi * static_cast<double>(j) / static_cast<double>(n));
    }
    return nodes;
}

auto chebyshev_differentiation_matrix(std::size_t n) -> Result<std::vector<std::vector<double>>> {
    if (n < 1) {
        return make_error<std::vector<std::vector<double>>>(MathError::domain_error);
    }
    auto nodes = chebyshev_gauss_lobatto_nodes(n);
    if (!nodes) {
        return make_error<std::vector<std::vector<double>>>(nodes.error());
    }
    const std::size_t m = n + 1;
    const auto& x = *nodes;
    // c_i = (2 if i is 0 or n, else 1) * (-1)^i.
    std::vector<double> c(m);
    for (std::size_t i = 0; i < m; ++i) {
        const double edge = (i == 0 || i == n) ? 2.0 : 1.0;
        const double sign = (i % 2 == 0) ? 1.0 : -1.0;
        c[i] = edge * sign;
    }
    std::vector<std::vector<double>> d(m, std::vector<double>(m, 0.0));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            if (i != j) {
                d[i][j] = (c[i] / c[j]) / (x[i] - x[j]);
            }
        }
    }
    // Diagonal by the negative-sum trick (row sums of a derivative operator vanish).
    for (std::size_t i = 0; i < m; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < m; ++j) {
            if (j != i) {
                s += d[i][j];
            }
        }
        d[i][i] = -s;
    }
    return d;
}

auto apply_dense_matrix(const std::vector<std::vector<double>>& a, std::span<const double> v)
    -> Result<std::vector<double>> {
    std::vector<double> out(a.size(), 0.0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != v.size()) {
            return make_error<std::vector<double>>(MathError::domain_error);
        }
        double acc = 0.0;
        for (std::size_t j = 0; j < v.size(); ++j) {
            acc = std::fma(a[i][j], v[j], acc);
        }
        out[i] = acc;
    }
    return out;
}

auto chebyshev_collocation_poisson(const std::function<double(double)>& f, std::size_t n)
    -> Result<std::vector<double>> {
    if (n < 2) {
        return make_error<std::vector<double>>(MathError::domain_error);  // need interior nodes
    }
    auto nodes = chebyshev_gauss_lobatto_nodes(n);
    if (!nodes) {
        return make_error<std::vector<double>>(nodes.error());
    }
    auto dmat = chebyshev_differentiation_matrix(n);
    if (!dmat) {
        return make_error<std::vector<double>>(dmat.error());
    }
    const auto d2 = matmul(*dmat, *dmat);  // second-derivative operator
    // Reduce to the interior nodes 1..n-1 and impose -u'' = f there (u = 0 at boundaries).
    const std::size_t inner = n - 1;
    std::vector<std::vector<double>> a(inner, std::vector<double>(inner, 0.0));
    std::vector<double> rhs(inner, 0.0);
    for (std::size_t i = 0; i < inner; ++i) {
        for (std::size_t j = 0; j < inner; ++j) {
            a[i][j] = -d2[i + 1][j + 1];
        }
        rhs[i] = f((*nodes)[i + 1]);
    }
    auto xin = solve_dense(std::move(a), std::move(rhs));
    if (!xin) {
        return make_error<std::vector<double>>(xin.error());
    }
    std::vector<double> u(n + 1, 0.0);  // boundary entries stay 0
    for (std::size_t i = 0; i < inner; ++i) {
        u[i + 1] = (*xin)[i];
    }
    return u;
}

// --- numerical Fourier ------------------------------------------------------

auto fourier_grid(std::size_t n) -> Result<std::vector<double>> {
    if (n < 1) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    std::vector<double> x(n);
    const double twopi = 2.0 * std::numbers::pi;
    for (std::size_t j = 0; j < n; ++j) {
        x[j] = twopi * static_cast<double>(j) / static_cast<double>(n);
    }
    return x;
}

auto fourier_differentiation_matrix(std::size_t n) -> Result<std::vector<std::vector<double>>> {
    if (n < 2 || n % 2 != 0) {
        return make_error<std::vector<std::vector<double>>>(MathError::domain_error);
    }
    std::vector<std::vector<double>> d(n, std::vector<double>(n, 0.0));
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t k = 0; k < n; ++k) {
            if (j == k) {
                continue;  // D_jj = 0
            }
            const long long diff = static_cast<long long>(j) - static_cast<long long>(k);
            const double sign = (diff % 2 == 0) ? 1.0 : -1.0;
            const double theta = std::numbers::pi * static_cast<double>(diff) /
                                 static_cast<double>(n);
            d[j][k] = 0.5 * sign * (std::cos(theta) / std::sin(theta));  // 0.5 (-1)^(j-k) cot
        }
    }
    return d;
}

auto fourier_spectral_derivative(std::span<const double> samples) -> Result<std::vector<double>> {
    const std::size_t n = samples.size();
    if (n < 2) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    std::vector<std::complex<double>> u(n);
    for (std::size_t j = 0; j < n; ++j) {
        u[j] = std::complex<double>(samples[j], 0.0);
    }
    auto uhat = dft_forward(u);
    for (std::size_t k = 0; k < n; ++k) {
        long long m = signed_wavenumber(k, n);
        if (n % 2 == 0 && k == n / 2) {
            m = 0;  // Nyquist mode carries no odd-derivative information; zero it
        }
        uhat[k] *= std::complex<double>(0.0, static_cast<double>(m));
    }
    auto v = dft_inverse(uhat);
    std::vector<double> out(n);
    for (std::size_t j = 0; j < n; ++j) {
        out[j] = v[j].real();
    }
    return out;
}

auto fourier_periodic_solve(std::span<const double> f_samples) -> Result<std::vector<double>> {
    const std::size_t n = f_samples.size();
    if (n < 2) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    std::vector<std::complex<double>> f(n);
    for (std::size_t j = 0; j < n; ++j) {
        f[j] = std::complex<double>(f_samples[j], 0.0);
    }
    auto fhat = dft_forward(f);
    for (std::size_t k = 0; k < n; ++k) {
        const long long m = signed_wavenumber(k, n);
        const double denom = 1.0 + static_cast<double>(m) * static_cast<double>(m);  // 1 + k'^2
        fhat[k] /= denom;
    }
    auto u = dft_inverse(fhat);
    std::vector<double> out(n);
    for (std::size_t j = 0; j < n; ++j) {
        out[j] = u[j].real();
    }
    return out;
}

// --- analogues: spectral element (C^0) -------------------------------------

namespace {

// Validate a rational mesh: at least two nodes, strictly increasing. Returns the element
// count M (= size - 1) on success.
[[nodiscard]] auto validate_mesh(std::span<const Rational> mesh) -> Result<std::size_t> {
    if (mesh.size() < 2) {
        return make_error<std::size_t>(MathError::domain_error);
    }
    for (std::size_t e = 0; e + 1 < mesh.size(); ++e) {
        auto h = mesh[e + 1].subtract(mesh[e]);
        if (!h) {
            return make_error<std::size_t>(h.error());
        }
        if (h->numerator() <= 0) {  // denominator is always positive in canonical form
            return make_error<std::size_t>(MathError::domain_error);  // not strictly increasing
        }
    }
    return mesh.size() - 1;
}

// Affine map data for element [z_e, z_{e+1}]: x = alpha + beta*xi, xi in [-1, 1].
struct AffineMap {
    Rational alpha;  // midpoint (z_e + z_{e+1}) / 2
    Rational beta;   // half-length (z_{e+1} - z_e) / 2
};

[[nodiscard]] auto element_map(const Rational& lo, const Rational& hi) -> Result<AffineMap> {
    auto sum = lo.add(hi);
    if (!sum) {
        return make_error<AffineMap>(sum.error());
    }
    auto alpha = sum->divide(Rational::from_int(2));
    if (!alpha) {
        return make_error<AffineMap>(alpha.error());
    }
    auto diff = hi.subtract(lo);
    if (!diff) {
        return make_error<AffineMap>(diff.error());
    }
    auto beta = diff->divide(Rational::from_int(2));
    if (!beta) {
        return make_error<AffineMap>(beta.error());
    }
    return AffineMap{.alpha = *alpha, .beta = *beta};
}

}  // namespace

auto spectral_element_legendre(const RationalPoly& f, std::span<const Rational> mesh)
    -> Result<std::vector<std::vector<Rational>>> {
    auto elements = validate_mesh(mesh);
    if (!elements) {
        return make_error<std::vector<std::vector<Rational>>>(elements.error());
    }
    std::vector<std::vector<Rational>> out(*elements);
    for (std::size_t e = 0; e < *elements; ++e) {
        auto map = element_map(mesh[e], mesh[e + 1]);
        if (!map) {
            return make_error<std::vector<std::vector<Rational>>>(map.error());
        }
        // Restrict f to this element in the reference coordinate: g(xi) = f(alpha + beta*xi).
        auto g = compose_affine(f, map->alpha, map->beta);
        if (!g) {
            return make_error<std::vector<std::vector<Rational>>>(g.error());
        }
        auto coeffs = legendre_coefficients(*g);
        if (!coeffs) {
            return make_error<std::vector<std::vector<Rational>>>(coeffs.error());
        }
        out[e] = std::move(*coeffs);
    }
    return out;
}

auto spectral_element_reconstruct(const std::vector<std::vector<Rational>>& elem_coeffs,
                                  std::span<const Rational> mesh)
    -> Result<std::vector<RationalPoly>> {
    auto elements = validate_mesh(mesh);
    if (!elements) {
        return make_error<std::vector<RationalPoly>>(elements.error());
    }
    if (elem_coeffs.size() != *elements) {
        return make_error<std::vector<RationalPoly>>(MathError::domain_error);
    }
    std::vector<RationalPoly> out(*elements);
    for (std::size_t e = 0; e < *elements; ++e) {
        auto map = element_map(mesh[e], mesh[e + 1]);
        if (!map) {
            return make_error<std::vector<RationalPoly>>(map.error());
        }
        // Reference polynomial g(xi) = sum_k coeffs[k] P_k(xi).
        auto gref = legendre_from_coefficients(elem_coeffs[e]);
        if (!gref) {
            return make_error<std::vector<RationalPoly>>(gref.error());
        }
        // Invert xi = (x - alpha) / beta = (-alpha/beta) + (1/beta) x, then g(xi(x)).
        auto inv_beta = Rational::from_int(1).divide(map->beta);
        if (!inv_beta) {
            return make_error<std::vector<RationalPoly>>(inv_beta.error());
        }
        auto neg_alpha = map->alpha.negate();
        if (!neg_alpha) {
            return make_error<std::vector<RationalPoly>>(neg_alpha.error());
        }
        auto a2 = neg_alpha->divide(map->beta);
        if (!a2) {
            return make_error<std::vector<RationalPoly>>(a2.error());
        }
        auto phys = compose_affine(*gref, *a2, *inv_beta);
        if (!phys) {
            return make_error<std::vector<RationalPoly>>(phys.error());
        }
        out[e] = std::move(*phys);
    }
    return out;
}

// --- analogues: discontinuous Galerkin -------------------------------------

auto dg_advection_rhs(const std::vector<std::vector<Rational>>& state,
                      std::span<const Rational> mesh, const Rational& a, const Rational& inflow)
    -> Result<std::vector<std::vector<Rational>>> {
    const std::size_t elements = state.size();
    if (elements == 0 || mesh.size() != elements + 1) {
        return make_error<std::vector<std::vector<Rational>>>(MathError::domain_error);
    }
    const std::size_t p = state[0].size();
    if (p == 0) {
        return make_error<std::vector<std::vector<Rational>>>(MathError::domain_error);
    }
    for (const auto& coeffs : state) {
        if (coeffs.size() != p) {
            return make_error<std::vector<std::vector<Rational>>>(MathError::domain_error);
        }
    }
    // Precompute S_{nm} = integral_{-1}^{1} P_m(xi) P_n'(xi) dxi (exact rationals).
    std::vector<RationalPoly> leg(p);
    std::vector<RationalPoly> legd(p);
    for (std::size_t k = 0; k < p; ++k) {
        auto pk = legendre(static_cast<std::int64_t>(k));
        if (!pk) {
            return make_error<std::vector<std::vector<Rational>>>(pk.error());
        }
        auto der = pk->derivative();
        if (!der) {
            return make_error<std::vector<std::vector<Rational>>>(der.error());
        }
        leg[k] = std::move(*pk);
        legd[k] = std::move(*der);
    }
    std::vector<std::vector<Rational>> smat(p, std::vector<Rational>(p));
    for (std::size_t nn = 0; nn < p; ++nn) {
        for (std::size_t mm = 0; mm < p; ++mm) {
            auto prod = legd[nn].multiply(leg[mm]);
            if (!prod) {
                return make_error<std::vector<std::vector<Rational>>>(prod.error());
            }
            auto s = integrate_unit(*prod);
            if (!s) {
                return make_error<std::vector<std::vector<Rational>>>(s.error());
            }
            smat[nn][mm] = *s;
        }
    }
    // Helper: sum of an element's coefficients = u_e(xi = +1) since P_m(1) = 1.
    const auto sum_coeffs = [](const std::vector<Rational>& c) -> Result<Rational> {
        Rational acc;  // 0
        for (const auto& v : c) {
            auto s = acc.add(v);
            if (!s) {
                return make_error<Rational>(s.error());
            }
            acc = *s;
        }
        return acc;
    };
    std::vector<std::vector<Rational>> out(elements, std::vector<Rational>(p));
    for (std::size_t e = 0; e < elements; ++e) {
        auto h = mesh[e + 1].subtract(mesh[e]);
        if (!h) {
            return make_error<std::vector<std::vector<Rational>>>(h.error());
        }
        if (h->numerator() <= 0) {
            return make_error<std::vector<std::vector<Rational>>>(MathError::domain_error);
        }
        // Upwind fluxes (a > 0): F_R = a * u_e(+1); F_L = a * u_{e-1}(+1), inflow at e == 0.
        auto ur = sum_coeffs(state[e]);
        if (!ur) {
            return make_error<std::vector<std::vector<Rational>>>(ur.error());
        }
        auto fr = a.multiply(*ur);
        if (!fr) {
            return make_error<std::vector<std::vector<Rational>>>(fr.error());
        }
        Rational fl;
        if (e == 0) {
            auto v = a.multiply(inflow);
            if (!v) {
                return make_error<std::vector<std::vector<Rational>>>(v.error());
            }
            fl = *v;
        } else {
            auto ul = sum_coeffs(state[e - 1]);
            if (!ul) {
                return make_error<std::vector<std::vector<Rational>>>(ul.error());
            }
            auto v = a.multiply(*ul);
            if (!v) {
                return make_error<std::vector<std::vector<Rational>>>(v.error());
            }
            fl = *v;
        }
        for (std::size_t nn = 0; nn < p; ++nn) {
            // Volume term: a * sum_m U_m S_{nm}.
            Rational vsum;  // 0
            for (std::size_t mm = 0; mm < p; ++mm) {
                auto term = state[e][mm].multiply(smat[nn][mm]);
                if (!term) {
                    return make_error<std::vector<std::vector<Rational>>>(term.error());
                }
                auto s = vsum.add(*term);
                if (!s) {
                    return make_error<std::vector<std::vector<Rational>>>(s.error());
                }
                vsum = *s;
            }
            auto vol = a.multiply(vsum);
            if (!vol) {
                return make_error<std::vector<std::vector<Rational>>>(vol.error());
            }
            // Flux term: F_R P_n(1) - F_L P_n(-1) = F_R - F_L * (-1)^n.
            const Rational sign = (nn % 2 == 0) ? Rational::from_int(1) : Rational::from_int(-1);
            auto fl_signed = fl.multiply(sign);
            if (!fl_signed) {
                return make_error<std::vector<std::vector<Rational>>>(fl_signed.error());
            }
            auto flux = fr->subtract(*fl_signed);
            if (!flux) {
                return make_error<std::vector<std::vector<Rational>>>(flux.error());
            }
            auto rn = vol->subtract(*flux);
            if (!rn) {
                return make_error<std::vector<std::vector<Rational>>>(rn.error());
            }
            // dU_n/dt = R_n * (2n+1) / h.
            auto scaled = rn->multiply(Rational::from_int(2 * static_cast<std::int64_t>(nn) + 1));
            if (!scaled) {
                return make_error<std::vector<std::vector<Rational>>>(scaled.error());
            }
            auto dudt = scaled->divide(*h);
            if (!dudt) {
                return make_error<std::vector<std::vector<Rational>>>(dudt.error());
            }
            out[e][nn] = *dudt;
        }
    }
    return out;
}

auto dg_advection_step(const std::vector<std::vector<Rational>>& state,
                       std::span<const Rational> mesh, const Rational& a, const Rational& inflow,
                       const Rational& dt) -> Result<std::vector<std::vector<Rational>>> {
    auto rhs = dg_advection_rhs(state, mesh, a, inflow);
    if (!rhs) {
        return make_error<std::vector<std::vector<Rational>>>(rhs.error());
    }
    std::vector<std::vector<Rational>> out(state.size());
    for (std::size_t e = 0; e < state.size(); ++e) {
        out[e].resize(state[e].size());
        for (std::size_t n = 0; n < state[e].size(); ++n) {
            auto incr = dt.multiply((*rhs)[e][n]);
            if (!incr) {
                return make_error<std::vector<std::vector<Rational>>>(incr.error());
            }
            auto updated = state[e][n].add(*incr);
            if (!updated) {
                return make_error<std::vector<std::vector<Rational>>>(updated.error());
            }
            out[e][n] = *updated;
        }
    }
    return out;
}

}  // namespace nimblecas
