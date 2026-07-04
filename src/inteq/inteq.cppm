// NimbleCAS integral equations: Fredholm / Volterra solvers, exact over Q where possible.
// @author Olumuyiwa Oluwasanmi
//
// This module solves linear and nonlinear INTEGRAL EQUATIONS. The exact-over-Q engine
// works on polynomials in Q[x] (nimblecas.ratpoly): the unknown phi, the free term f, and
// the SEPARABLE (degenerate) kernel K(x,t) = Σ_i g_i(x) h_i(t) are all RationalPoly, and
// every kernel action reduces to elementary polynomial integrals evaluated through
// nimblecas.integrate (integrate_rational) at rational limits. Under those hypotheses the
// results below are the EXACT rational solution (the linear-algebra reduction) or the EXACT
// truncated Neumann/Picard/Adomian series — no floating point, no fabricated closed form.
//
//   * FREDHOLM 2nd kind  phi(x) = f(x) + lambda ∫_a^b K(x,t) phi(t) dt
//       - SEPARABLE kernel  -> a finite linear system (I − lambda M) c = d over Q, solved
//         exactly with nimblecas.matrix; phi = f + lambda Σ_i c_i g_i is EXACT.
//       - NEUMANN series    -> Σ_{n} (lambda 𝒦)^n f, truncated (exact per term; converges
//         when |lambda|·‖𝒦‖ < 1).
//   * FREDHOLM 1st kind  f(x) = ∫_a^b K(x,t) phi(t) dt  — ILL-POSED. For a separable kernel
//     only the moments c_i = ∫ h_i phi are recoverable (phi is NON-UNIQUE, needs
//     regularization). We return those moments when f ∈ span{g_i}; otherwise not_implemented
//     / domain_error. No phi is ever fabricated.
//   * VOLTERRA 2nd kind  phi(x) = f(x) + lambda ∫_a^x K(x,t) phi(t) dt
//       - PICARD successive approximation (always converges): the linear iterate equals the
//         truncated Neumann sum and is EXACT for polynomial data.
//       - CONVOLUTION kernel K(x,t) = k(x−t): the LAPLACE method Φ(s) = F(s)/(1 − lambda·k̂(s))
//         via nimblecas.laplace; returns the TRANSFORM-DOMAIN solution (no general inverse).
//   * NONLINEAR (HAMMERSTEIN / nonlinear VOLTERRA)  N(phi) a polynomial nonlinearity in phi:
//     solved by PICARD, and by the decomposition/homotopy family below.
//   * DECOMPOSITION / HOMOTOPY series (the analogues of Neumann/Picard for the nonlinear case):
//       - ADM  Adomian Decomposition: phi = Σ phi_n, phi_0 = f, phi_{n+1} = lambda 𝒦 A_n with
//         A_n the Adomian polynomials of N (graded projection). For N = identity A_n = phi_n,
//         so ADM reduces exactly to the Neumann/Picard series (a built-in cross-check).
//       - HPM  Homotopy Perturbation: the same graded recursion as ADM (ADM == HPM).
//       - HAM  Homotopy Analysis: convergence-control parameter ħ; HAM(ħ = −1) recovers ADM.
//
// HONESTY BOUNDARY. EXACT over Q for separable/polynomial kernels with elementary integrals
// and rational limits (the linear-algebra reduction and the truncated Neumann / Picard /
// Adomian / homotopy series). Truncated series are LOCAL, finite-order approximations of the
// true solution unless they terminate. A non-polynomial antiderivative (a logarithmic part or
// a genuine rational-function remainder) or a Laplace pair off the table returns
// MathError::not_implemented — never a fabricated closed form. Fredholm 1st kind is ill-posed.
// Every operation is total and railway-oriented (Rule 32): no exceptions, Result<T> throughout.

export module nimblecas.inteq;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.integrate;
import nimblecas.symbolic;
import nimblecas.laplace;
import nimblecas.simplify;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// Nonlinearity — an autonomous polynomial nonlinearity N(phi) = Σ_j coeffs[j] phi^j.
// ---------------------------------------------------------------------------
// N = identity (coeffs {0,1}) models the LINEAR equation; N = phi^p models the classic
// Hammerstein / nonlinear-Volterra case. A t-dependent Hammerstein weight a(t) folds into
// the kernel factor h_i (since ∫ K a(t) M(phi) = ∫ [K a] M(phi)), so this autonomous form
// covers the separable nonlinear equations exactly over Q.
class Nonlinearity {
public:
    // N(phi) = Σ_j coeffs[j] phi^j (coeffs ascending in the power of phi).
    [[nodiscard]] static auto polynomial(std::vector<Rational> coeffs) -> Nonlinearity {
        return Nonlinearity{std::move(coeffs)};
    }
    // N(phi) = phi.
    [[nodiscard]] static auto identity() -> Nonlinearity {
        return Nonlinearity{std::vector<Rational>{Rational::from_int(0), Rational::from_int(1)}};
    }
    // N(phi) = phi^p (p >= 1; p == 0 gives the constant 1).
    [[nodiscard]] static auto power(std::size_t p) -> Nonlinearity {
        std::vector<Rational> c(p + 1);
        c[p] = Rational::from_int(1);
        return Nonlinearity{std::move(c)};
    }

    [[nodiscard]] auto coefficients() const noexcept -> std::span<const Rational> {
        return coeffs_;
    }
    // True iff N is exactly the identity {0,1}: the case in which the decomposition series
    // reduce to the Neumann/Picard series (A_n == phi_n).
    [[nodiscard]] auto is_linear() const noexcept -> bool {
        return coeffs_.size() == 2 && coeffs_[0].is_zero() &&
               coeffs_[1] == Rational::from_int(1);
    }

    // Pointwise N(phi) as a polynomial (Horner over Q[x]). Exact for any polynomial phi.
    [[nodiscard]] auto apply(const RationalPoly& phi) const -> Result<RationalPoly>;

private:
    explicit Nonlinearity(std::vector<Rational> coeffs) : coeffs_(std::move(coeffs)) {}
    std::vector<Rational> coeffs_;  // ascending in the power of phi
};

// ---------------------------------------------------------------------------
// SeparableKernel — the degenerate kernel K(x,t) = Σ_i g[i](x) h[i](t).
// ---------------------------------------------------------------------------
struct SeparableKernel {
    std::vector<RationalPoly> g;  // g_i(x)
    std::vector<RationalPoly> h;  // h_i(t)

    [[nodiscard]] auto rank() const noexcept -> std::size_t { return g.size(); }
    [[nodiscard]] auto is_valid() const noexcept -> bool { return g.size() == h.size(); }
};

// ---------------------------------------------------------------------------
// IntegralOperator — the linear action u -> lambda ∫ K(x,t) u(t) dt over a separable K.
// ---------------------------------------------------------------------------
// Fredholm: ∫_a^b (definite) -> lambda Σ_i g_i(x) [∫_a^b h_i u]. Volterra: ∫_a^x (variable
// upper limit) -> lambda Σ_i g_i(x) [∫_a^x h_i(t) u(t) dt], a polynomial in x. apply() FOLDS
// lambda in, so the whole action is lambda·𝒦 (the object appearing in every recursion here).
class IntegralOperator {
public:
    [[nodiscard]] static auto fredholm(SeparableKernel kernel, Rational lambda, Rational a,
                                       Rational b) -> Result<IntegralOperator>;
    [[nodiscard]] static auto volterra(SeparableKernel kernel, Rational lambda, Rational a)
        -> Result<IntegralOperator>;

    // (lambda 𝒦 u)(x). Fails with not_implemented when an integrand has no polynomial
    // antiderivative (non-elementary over Q for this representation), else propagates overflow.
    [[nodiscard]] auto apply(const RationalPoly& u) const -> Result<RationalPoly>;

    [[nodiscard]] auto lambda() const noexcept -> Rational { return lambda_; }
    [[nodiscard]] auto is_volterra() const noexcept -> bool { return volterra_; }
    [[nodiscard]] auto kernel() const noexcept -> const SeparableKernel& { return kernel_; }

private:
    IntegralOperator(SeparableKernel kernel, Rational lambda, Rational a, Rational b,
                     bool volterra)
        : kernel_(std::move(kernel)), lambda_(lambda), a_(a), b_(b), volterra_(volterra) {}

    SeparableKernel kernel_;
    Rational lambda_;
    Rational a_;
    Rational b_;      // ignored for a Volterra operator
    bool volterra_;
};

// ---------------------------------------------------------------------------
// Solution payloads.
// ---------------------------------------------------------------------------

// The exact separable Fredholm-2 solution phi(x) = f(x) + lambda Σ_i coefficients[i] g_i(x),
// with coefficients[i] = c_i = ∫_a^b h_i(t) phi(t) dt the solved moment constants.
struct FredholmSolution {
    RationalPoly phi;                    // the exact rational solution
    std::vector<Rational> coefficients;  // c_i (the separable moments)
};

// The recoverable information from an (ill-posed) Fredholm-1 separable equation: the moments
// c_i = ∫_a^b h_i(t) phi(t) dt determined by f ∈ span{g_i}. phi ITSELF IS NON-UNIQUE.
struct FirstKindSolution {
    std::vector<Rational> moments;  // c_i ; phi is NOT determined (regularization required)
};

// ===========================================================================
// FREDHOLM 2nd kind.
// ===========================================================================

// Exact separable solution. Builds M_ij = ∫_a^b h_i g_j and d_i = ∫_a^b h_i f, solves the
// r×r system (I − lambda M) c = d over Q with nimblecas.matrix, and returns
// phi = f + lambda Σ c_i g_i. domain_error if the kernel is ragged, if a moment integral is
// non-polynomial (not_implemented), or if (I − lambda M) is singular (lambda an eigenvalue).
[[nodiscard]] auto fredholm2_separable(const RationalPoly& f, const SeparableKernel& kernel,
                                       const Rational& lambda, const Rational& a,
                                       const Rational& b) -> Result<FredholmSolution>;

// Truncated Neumann (resolvent) series Σ_{n=0}^{order} (lambda 𝒦)^n f. Exact term by term;
// a local finite-order approximation of the true solution (it terminates only for a nilpotent
// operator). Convergence to the true phi requires |lambda|·‖𝒦‖ < 1.
[[nodiscard]] auto fredholm2_neumann(const RationalPoly& f, const SeparableKernel& kernel,
                                     const Rational& lambda, const Rational& a,
                                     const Rational& b, std::size_t order)
    -> Result<RationalPoly>;

// ===========================================================================
// FREDHOLM 1st kind (ill-posed).
// ===========================================================================

// Separable-kernel case: recover the moments c_i from f = Σ_i c_i g_i (exact over Q). phi is
// NOT reconstructed — the equation is ill-posed and admits infinitely many solutions; only the
// moments are determined. domain_error when f ∉ span{g_i} (no solution); not_implemented when
// the g_i are linearly dependent (the recoverable moments are themselves non-unique).
[[nodiscard]] auto fredholm1_separable(const RationalPoly& f, const SeparableKernel& kernel,
                                       const Rational& a, const Rational& b)
    -> Result<FirstKindSolution>;

// ===========================================================================
// VOLTERRA 2nd kind.
// ===========================================================================

// Linear Picard successive approximation to `order`. For the linear equation the Picard
// iterate equals the truncated Neumann sum Σ_{n=0}^{order} (lambda 𝒦)^n f, computed exactly
// over Q (𝒦 here is the Volterra operator ∫_a^x). Always convergent for Volterra.
[[nodiscard]] auto volterra2_picard(const RationalPoly& f, const SeparableKernel& kernel,
                                    const Rational& lambda, const Rational& a, std::size_t order)
    -> Result<RationalPoly>;

// Convolution-kernel Volterra via the Laplace transform: Φ(s) = F(s)/(1 − lambda·k̂(s)), where
// F = L{f} and k̂ = L{kernel} (the kernel k(x−t) supplied as k in the variable `var`). Returns
// the TRANSFORM-DOMAIN solution Φ(s) simplified; a general inverse transform is not provided,
// so this is honest about staying in the s-domain. not_implemented if f or the kernel is off
// the Laplace table.
[[nodiscard]] auto volterra_convolution_laplace(const Expr& f, const Expr& kernel,
                                                const Rational& lambda, std::string_view var,
                                                std::string_view s) -> Result<Expr>;

// ===========================================================================
// NONLINEAR (Hammerstein / nonlinear Volterra) — Picard.
// ===========================================================================

// Nonlinear Volterra Picard iterate after `iterations` steps: psi_0 = f,
// psi_{k+1} = f + lambda ∫_a^x K(x,t) N(psi_k(t)) dt. Exact over Q while N keeps the iterate
// polynomial (any polynomial N does). Returns the truncated iterate (a local approximation).
[[nodiscard]] auto volterra_nonlinear_picard(const RationalPoly& f, const SeparableKernel& kernel,
                                             const Rational& lambda, const Rational& a,
                                             const Nonlinearity& n, std::size_t iterations)
    -> Result<RationalPoly>;

// Hammerstein (nonlinear Fredholm) Picard iterate: psi_{k+1} = f + lambda ∫_a^b K N(psi_k).
// Exact over Q for polynomial data; returns the truncated iterate. Note that for a definite
// (Fredholm) integral Picard need not converge — this is a finite-order iterate, honestly.
[[nodiscard]] auto hammerstein_picard(const RationalPoly& f, const SeparableKernel& kernel,
                                      const Rational& lambda, const Rational& a, const Rational& b,
                                      const Nonlinearity& n, std::size_t iterations)
    -> Result<RationalPoly>;

// ===========================================================================
// DECOMPOSITION / HOMOTOPY series (ADM / HPM / HAM).
// ===========================================================================

// Adomian polynomials A_0..A_k of the nonlinearity N for the components phi_0..phi_k
// (k = components.size() − 1). A_m is the grade-m projection [p^m] N(Σ_i p^i phi_i), computed
// exactly over Q by truncated arithmetic in the formal parameter p. For N = identity, A_m
// reduces to phi_m. domain_error on an empty component list; overflow propagated.
[[nodiscard]] auto adomian_polynomials(const Nonlinearity& n,
                                       const std::vector<RationalPoly>& components)
    -> Result<std::vector<RationalPoly>>;

// ADM solution to `order`: phi_0 = f, phi_{n+1} = lambda 𝒦 A_n, result Σ_{n=0}^{order} phi_n.
// `op` is the linear integral operator (Fredholm or Volterra). Exact over Q; truncated series.
// For a linear N this is identically the Neumann/Picard series (cross-check).
[[nodiscard]] auto adm_solve(const IntegralOperator& op, const RationalPoly& f,
                             const Nonlinearity& n, std::size_t order) -> Result<RationalPoly>;

// HPM solution. The homotopy (1−p)(phi − f) + p(phi − f − lambda 𝒦 N(phi)) = 0 collapses to the
// SAME graded recursion as ADM, so this returns the identical series (ADM == HPM).
[[nodiscard]] auto hpm_solve(const IntegralOperator& op, const RationalPoly& f,
                             const Nonlinearity& n, std::size_t order) -> Result<RationalPoly>;

// HAM solution with convergence-control parameter ħ. The m-th order deformation is
// phi_m = χ_m phi_{m−1} + ħ (phi_{m−1} − δ_{m,1} f − lambda 𝒦 A_{m−1}), χ_m = 0 (m=1) else 1;
// result Σ_{m=0}^{order} phi_m. ħ = −1 recovers ADM/HPM exactly (asserted in tests). ħ is the
// extra freedom HAM adds; other ħ give a different, still-exactly-rational family member.
[[nodiscard]] auto ham_solve(const IntegralOperator& op, const RationalPoly& f,
                             const Nonlinearity& n, const Rational& hbar, std::size_t order)
    -> Result<RationalPoly>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// --- exact elementary integration of polynomials over Q, via nimblecas.integrate ----------

// Evaluate a polynomial at a rational point by Horner (overflow-checked).
[[nodiscard]] auto eval_poly(const RationalPoly& p, const Rational& x) -> Result<Rational> {
    Rational acc;  // 0/1
    const std::span<const Rational> c = p.coefficients();
    for (std::size_t i = c.size(); i-- > 0;) {
        auto m = acc.multiply(x);
        if (!m) {
            return make_error<Rational>(m.error());
        }
        auto s = m->add(c[i]);
        if (!s) {
            return make_error<Rational>(s.error());
        }
        acc = *s;
    }
    return acc;
}

// The polynomial antiderivative G of p with G(0) = 0, obtained by reusing nimblecas.integrate
// (integrate_rational over the constant denominator 1). Returns not_implemented when the
// antiderivative is not a plain polynomial — a logarithmic part or a genuine rational-function
// remainder — i.e. the integral is not elementary-over-Q in this representation.
[[nodiscard]] auto elementary_antiderivative(const RationalPoly& p) -> Result<RationalPoly> {
    if (p.is_zero()) {
        return RationalPoly{};
    }
    const RationalPoly one = RationalPoly::constant(Rational::from_int(1));
    auto ri = integrate_rational(p, one);
    if (!ri) {
        return make_error<RationalPoly>(ri.error());
    }
    if (!ri->log_terms.empty()) {
        return make_error<RationalPoly>(MathError::not_implemented);  // non-elementary here
    }
    // G = rational_num / rational_den must be a polynomial (den is a nonzero constant for a
    // polynomial integrand); a non-zero remainder means it is not, so decline honestly.
    auto dm = ri->rational_num.divide(ri->rational_den);
    if (!dm) {
        return make_error<RationalPoly>(dm.error());
    }
    if (!dm->remainder.is_zero()) {
        return make_error<RationalPoly>(MathError::not_implemented);
    }
    return dm->quotient;
}

// ∫_a^b p dt (definite), exact over Q.
[[nodiscard]] auto definite_integral(const RationalPoly& p, const Rational& a, const Rational& b)
    -> Result<Rational> {
    auto g = elementary_antiderivative(p);
    if (!g) {
        return make_error<Rational>(g.error());
    }
    auto gb = eval_poly(*g, b);
    if (!gb) {
        return make_error<Rational>(gb.error());
    }
    auto ga = eval_poly(*g, a);
    if (!ga) {
        return make_error<Rational>(ga.error());
    }
    return gb->subtract(*ga);
}

// ∫_a^x p(t) dt (indefinite, variable upper limit x) as a polynomial in x: G(x) − G(a).
[[nodiscard]] auto integral_from(const RationalPoly& p, const Rational& a)
    -> Result<RationalPoly> {
    auto g = elementary_antiderivative(p);
    if (!g) {
        return make_error<RationalPoly>(g.error());
    }
    auto ga = eval_poly(*g, a);
    if (!ga) {
        return make_error<RationalPoly>(ga.error());
    }
    return g->subtract(RationalPoly::constant(*ga));
}

// Coefficient-wise sum of a list of polynomials.
[[nodiscard]] auto sum_polys(const std::vector<RationalPoly>& v) -> Result<RationalPoly> {
    RationalPoly acc;  // zero
    for (const RationalPoly& p : v) {
        auto s = acc.add(p);
        if (!s) {
            return make_error<RationalPoly>(s.error());
        }
        acc = *s;
    }
    return acc;
}

// --- formal p-series over Q[x] (for the Adomian graded projection) ------------------------
// A p-series of length L is a vector of L RationalPoly, index i holding the coefficient of p^i.

using PSeries = std::vector<RationalPoly>;

// Convolution product truncated to length L (drop p-powers >= L).
[[nodiscard]] auto pseries_mul(const PSeries& a, const PSeries& b, std::size_t len)
    -> Result<PSeries> {
    PSeries out(len);  // L zero polynomials
    for (std::size_t i = 0; i < a.size() && i < len; ++i) {
        if (a[i].is_zero()) {
            continue;
        }
        for (std::size_t j = 0; j + i < len && j < b.size(); ++j) {
            auto prod = a[i].multiply(b[j]);
            if (!prod) {
                return make_error<PSeries>(prod.error());
            }
            auto s = out[i + j].add(*prod);
            if (!s) {
                return make_error<PSeries>(s.error());
            }
            out[i + j] = *s;
        }
    }
    return out;
}

// N(S) as a p-series (Horner over p-series arithmetic), for N(phi) = Σ_j c_j phi^j.
[[nodiscard]] auto apply_nonlinearity_pseries(const Nonlinearity& n, const PSeries& s,
                                              std::size_t len) -> Result<PSeries> {
    const std::span<const Rational> c = n.coefficients();
    // acc starts at the highest coefficient as a constant p-series (all at p^0).
    PSeries acc(len);
    if (c.empty()) {
        return acc;  // N == 0
    }
    acc[0] = RationalPoly::constant(c.back());
    for (std::size_t idx = c.size() - 1; idx-- > 0;) {
        auto prod = pseries_mul(acc, s, len);
        if (!prod) {
            return prod;
        }
        acc = std::move(*prod);
        auto s0 = acc[0].add(RationalPoly::constant(c[idx]));
        if (!s0) {
            return make_error<PSeries>(s0.error());
        }
        acc[0] = *s0;
    }
    return acc;
}

// --- shared linear / Picard recursions ----------------------------------------------------

// Truncated Neumann sum Σ_{n=0}^{order} (lambda 𝒦)^n f (op.apply already folds lambda in).
// The linear Picard iterate coincides with this for both Fredholm and Volterra operators.
[[nodiscard]] auto neumann_sum(const IntegralOperator& op, const RationalPoly& f,
                               std::size_t order) -> Result<RationalPoly> {
    RationalPoly term = f;
    RationalPoly sum = f;
    for (std::size_t n = 1; n <= order; ++n) {
        auto next = op.apply(term);
        if (!next) {
            return next;
        }
        term = std::move(*next);
        auto s = sum.add(term);
        if (!s) {
            return s;
        }
        sum = std::move(*s);
    }
    return sum;
}

// Nonlinear Picard fixed-point iterate: psi_{k+1} = f + lambda 𝒦 N(psi_k).
[[nodiscard]] auto picard_iterate(const IntegralOperator& op, const RationalPoly& f,
                                  const Nonlinearity& n, std::size_t iterations)
    -> Result<RationalPoly> {
    RationalPoly psi = f;
    for (std::size_t k = 0; k < iterations; ++k) {
        auto nphi = n.apply(psi);
        if (!nphi) {
            return nphi;
        }
        auto kphi = op.apply(*nphi);
        if (!kphi) {
            return kphi;
        }
        auto s = f.add(*kphi);
        if (!s) {
            return s;
        }
        psi = std::move(*s);
    }
    return psi;
}

// Convert an exact Rational to its canonical Expr constant (integer when integral).
[[nodiscard]] auto rational_to_expr(const Rational& r) -> Result<Expr> {
    if (r.is_integer()) {
        return Expr::integer(r.numerator());
    }
    return Expr::rational(r.numerator(), r.denominator());
}

}  // namespace

// --- Nonlinearity -------------------------------------------------------------------------

auto Nonlinearity::apply(const RationalPoly& phi) const -> Result<RationalPoly> {
    if (coeffs_.empty()) {
        return RationalPoly{};  // N == 0
    }
    // Horner: acc = c_J; acc = acc*phi + c_j.
    RationalPoly acc = RationalPoly::constant(coeffs_.back());
    for (std::size_t idx = coeffs_.size() - 1; idx-- > 0;) {
        auto prod = acc.multiply(phi);
        if (!prod) {
            return prod;
        }
        auto s = prod->add(RationalPoly::constant(coeffs_[idx]));
        if (!s) {
            return s;
        }
        acc = std::move(*s);
    }
    return acc;
}

// --- IntegralOperator ---------------------------------------------------------------------

auto IntegralOperator::fredholm(SeparableKernel kernel, Rational lambda, Rational a, Rational b)
    -> Result<IntegralOperator> {
    if (!kernel.is_valid()) {
        return make_error<IntegralOperator>(MathError::domain_error);
    }
    return IntegralOperator{std::move(kernel), lambda, a, b, /*volterra=*/false};
}

auto IntegralOperator::volterra(SeparableKernel kernel, Rational lambda, Rational a)
    -> Result<IntegralOperator> {
    if (!kernel.is_valid()) {
        return make_error<IntegralOperator>(MathError::domain_error);
    }
    return IntegralOperator{std::move(kernel), lambda, a, a, /*volterra=*/true};
}

auto IntegralOperator::apply(const RationalPoly& u) const -> Result<RationalPoly> {
    RationalPoly acc;  // zero
    for (std::size_t i = 0; i < kernel_.g.size(); ++i) {
        auto hu = kernel_.h[i].multiply(u);
        if (!hu) {
            return hu;
        }
        RationalPoly term;
        if (volterra_) {
            auto integ = integral_from(*hu, a_);  // polynomial in x
            if (!integ) {
                return integ;
            }
            auto t = kernel_.g[i].multiply(*integ);
            if (!t) {
                return t;
            }
            term = std::move(*t);
        } else {
            auto c = definite_integral(*hu, a_, b_);  // constant
            if (!c) {
                return make_error<RationalPoly>(c.error());
            }
            auto t = kernel_.g[i].scale(*c);
            if (!t) {
                return t;
            }
            term = std::move(*t);
        }
        auto s = acc.add(term);
        if (!s) {
            return s;
        }
        acc = std::move(*s);
    }
    return acc.scale(lambda_);  // fold lambda in: result is lambda·𝒦·u
}

// --- Fredholm 2nd kind --------------------------------------------------------------------

auto fredholm2_separable(const RationalPoly& f, const SeparableKernel& kernel,
                         const Rational& lambda, const Rational& a, const Rational& b)
    -> Result<FredholmSolution> {
    if (!kernel.is_valid()) {
        return make_error<FredholmSolution>(MathError::domain_error);
    }
    const std::size_t r = kernel.rank();
    if (r == 0) {
        return FredholmSolution{.phi = f, .coefficients = {}};  // no kernel: phi == f
    }

    // System (I − lambda M) c = d, with M_ij = ∫ h_i g_j, d_i = ∫ h_i f.
    std::vector<std::vector<Rational>> rows(r, std::vector<Rational>(r));
    std::vector<std::vector<Rational>> dcol(r, std::vector<Rational>(1));
    for (std::size_t i = 0; i < r; ++i) {
        for (std::size_t j = 0; j < r; ++j) {
            auto hg = kernel.h[i].multiply(kernel.g[j]);
            if (!hg) {
                return make_error<FredholmSolution>(hg.error());
            }
            auto mij = definite_integral(*hg, a, b);
            if (!mij) {
                return make_error<FredholmSolution>(mij.error());
            }
            auto lm = lambda.multiply(*mij);  // lambda·M_ij
            if (!lm) {
                return make_error<FredholmSolution>(lm.error());
            }
            const Rational kron = (i == j) ? Rational::from_int(1) : Rational::from_int(0);
            auto entry = kron.subtract(*lm);  // (I − lambda M)_ij
            if (!entry) {
                return make_error<FredholmSolution>(entry.error());
            }
            rows[i][j] = *entry;
        }
        auto hf = kernel.h[i].multiply(f);
        if (!hf) {
            return make_error<FredholmSolution>(hf.error());
        }
        auto di = definite_integral(*hf, a, b);
        if (!di) {
            return make_error<FredholmSolution>(di.error());
        }
        dcol[i][0] = *di;
    }

    auto amat = Matrix::from_rows(std::move(rows));
    if (!amat) {
        return make_error<FredholmSolution>(amat.error());
    }
    auto bvec = Matrix::from_rows(std::move(dcol));
    if (!bvec) {
        return make_error<FredholmSolution>(bvec.error());
    }
    auto cvec = amat->solve(*bvec);  // singular (lambda an eigenvalue) -> domain_error
    if (!cvec) {
        return make_error<FredholmSolution>(cvec.error());
    }

    // phi = f + lambda Σ_i c_i g_i.
    std::vector<Rational> coeffs(r);
    RationalPoly phi = f;
    for (std::size_t i = 0; i < r; ++i) {
        const Rational ci = cvec->at(i, 0);
        coeffs[i] = ci;
        auto lc = lambda.multiply(ci);
        if (!lc) {
            return make_error<FredholmSolution>(lc.error());
        }
        auto term = kernel.g[i].scale(*lc);
        if (!term) {
            return make_error<FredholmSolution>(term.error());
        }
        auto s = phi.add(*term);
        if (!s) {
            return make_error<FredholmSolution>(s.error());
        }
        phi = std::move(*s);
    }
    return FredholmSolution{.phi = std::move(phi), .coefficients = std::move(coeffs)};
}

auto fredholm2_neumann(const RationalPoly& f, const SeparableKernel& kernel,
                       const Rational& lambda, const Rational& a, const Rational& b,
                       std::size_t order) -> Result<RationalPoly> {
    auto op = IntegralOperator::fredholm(kernel, lambda, a, b);
    if (!op) {
        return make_error<RationalPoly>(op.error());
    }
    return neumann_sum(*op, f, order);
}

// --- Fredholm 1st kind (ill-posed) --------------------------------------------------------

auto fredholm1_separable(const RationalPoly& f, const SeparableKernel& kernel, const Rational& a,
                         const Rational& b) -> Result<FirstKindSolution> {
    (void)a;
    (void)b;  // the moments come from f = Σ c_i g_i; the limits do not enter this recovery.
    if (!kernel.is_valid()) {
        return make_error<FirstKindSolution>(MathError::domain_error);
    }
    const std::size_t r = kernel.rank();
    if (r == 0) {
        // No kernel: f must be identically zero, and then every c is vacuous.
        if (f.is_zero()) {
            return FirstKindSolution{.moments = {}};
        }
        return make_error<FirstKindSolution>(MathError::domain_error);  // no solution
    }

    // Solve f = Σ_i c_i g_i exactly by equating polynomial coefficients: G c = fvec, where G's
    // column i is g_i's coefficient vector. Use the (exact over Q) normal equations
    // (G^T G) c = G^T f, then verify G c == fvec for consistency.
    std::int64_t maxdeg = f.degree();
    for (const RationalPoly& gi : kernel.g) {
        maxdeg = std::max(maxdeg, gi.degree());
    }
    if (maxdeg < 0) {
        // f and every g_i are the zero polynomial: 0 = Σ c_i·0 holds for ANY c, so the moments
        // are genuinely non-unique — honestly not recoverable (same verdict as the dependent-g_i
        // singular-Gram case below), never a fabricated all-zero "solution".
        return make_error<FirstKindSolution>(MathError::not_implemented);
    }
    const std::size_t rowsn = static_cast<std::size_t>(maxdeg) + 1;

    std::vector<std::vector<Rational>> gmat(rowsn, std::vector<Rational>(r));
    std::vector<std::vector<Rational>> fcol(rowsn, std::vector<Rational>(1));
    for (std::size_t k = 0; k < rowsn; ++k) {
        for (std::size_t i = 0; i < r; ++i) {
            gmat[k][i] = kernel.g[i].coefficient(k);
        }
        fcol[k][0] = f.coefficient(k);
    }
    auto G = Matrix::from_rows(std::move(gmat));
    if (!G) {
        return make_error<FirstKindSolution>(G.error());
    }
    auto fv = Matrix::from_rows(std::move(fcol));
    if (!fv) {
        return make_error<FirstKindSolution>(fv.error());
    }
    auto Gt = G->transpose();
    if (!Gt) {
        return make_error<FirstKindSolution>(Gt.error());
    }
    auto GtG = Gt->multiply(*G);
    if (!GtG) {
        return make_error<FirstKindSolution>(GtG.error());
    }
    auto Gtf = Gt->multiply(*fv);
    if (!Gtf) {
        return make_error<FirstKindSolution>(Gtf.error());
    }
    auto c = GtG->solve(*Gtf);
    if (!c) {
        // Singular Gram matrix: the g_i are linearly dependent, so the moments are themselves
        // non-unique — honestly not recoverable without more structure.
        return make_error<FirstKindSolution>(MathError::not_implemented);
    }
    // Consistency: f must actually lie in span{g_i}.
    auto recon = G->multiply(*c);
    if (!recon) {
        return make_error<FirstKindSolution>(recon.error());
    }
    if (!(*recon == *fv)) {
        return make_error<FirstKindSolution>(MathError::domain_error);  // f ∉ span{g_i}
    }
    std::vector<Rational> moments(r);
    for (std::size_t i = 0; i < r; ++i) {
        moments[i] = c->at(i, 0);
    }
    return FirstKindSolution{.moments = std::move(moments)};
}

// --- Volterra 2nd kind --------------------------------------------------------------------

auto volterra2_picard(const RationalPoly& f, const SeparableKernel& kernel, const Rational& lambda,
                      const Rational& a, std::size_t order) -> Result<RationalPoly> {
    auto op = IntegralOperator::volterra(kernel, lambda, a);
    if (!op) {
        return make_error<RationalPoly>(op.error());
    }
    return neumann_sum(*op, f, order);  // linear Picard iterate == truncated Neumann sum
}

auto volterra_convolution_laplace(const Expr& f, const Expr& kernel, const Rational& lambda,
                                  std::string_view var, std::string_view s) -> Result<Expr> {
    auto F = laplace_transform(f, var, s);
    if (!F) {
        return F;  // off-table free term
    }
    auto K = laplace_transform(kernel, var, s);
    if (!K) {
        return K;  // off-table kernel
    }
    auto lam = rational_to_expr(lambda);
    if (!lam) {
        return lam;
    }
    // denom = 1 − lambda·k̂(s); simplify it first so its canonical form is the base that the
    // inverse power reuses (a consumer can then cancel Φ·denom == F exactly).
    Expr raw_denom =
        Expr::sum({Expr::integer(1), Expr::product({Expr::integer(-1), *lam, *K})});
    auto denom = simplify(raw_denom);
    if (!denom) {
        return denom;
    }
    // Φ(s) = F(s) · denom^{-1}.
    Expr phi = Expr::product({*F, Expr::power(*denom, Expr::integer(-1))});
    return simplify(phi);
}

// --- Nonlinear Picard ---------------------------------------------------------------------

auto volterra_nonlinear_picard(const RationalPoly& f, const SeparableKernel& kernel,
                               const Rational& lambda, const Rational& a, const Nonlinearity& n,
                               std::size_t iterations) -> Result<RationalPoly> {
    auto op = IntegralOperator::volterra(kernel, lambda, a);
    if (!op) {
        return make_error<RationalPoly>(op.error());
    }
    return picard_iterate(*op, f, n, iterations);
}

auto hammerstein_picard(const RationalPoly& f, const SeparableKernel& kernel,
                        const Rational& lambda, const Rational& a, const Rational& b,
                        const Nonlinearity& n, std::size_t iterations) -> Result<RationalPoly> {
    auto op = IntegralOperator::fredholm(kernel, lambda, a, b);
    if (!op) {
        return make_error<RationalPoly>(op.error());
    }
    return picard_iterate(*op, f, n, iterations);
}

// --- ADM / HPM / HAM ----------------------------------------------------------------------

auto adomian_polynomials(const Nonlinearity& n, const std::vector<RationalPoly>& components)
    -> Result<std::vector<RationalPoly>> {
    if (components.empty()) {
        return make_error<std::vector<RationalPoly>>(MathError::domain_error);
    }
    const std::size_t len = components.size();
    // The p-series S = Σ_i p^i phi_i; A_m = [p^m] N(S). Components of index > m cannot affect
    // [p^m], so N of the whole partial sum yields every A_m for m < len exactly.
    auto ns = apply_nonlinearity_pseries(n, components, len);
    if (!ns) {
        return make_error<std::vector<RationalPoly>>(ns.error());
    }
    return *ns;  // ns[m] == A_m
}

namespace {

// Shared ADM/HPM engine: phi_0 = f, phi_{n+1} = lambda 𝒦 A_n, return Σ_{n=0}^{order} phi_n.
[[nodiscard]] auto adm_engine(const IntegralOperator& op, const RationalPoly& f,
                              const Nonlinearity& n, std::size_t order) -> Result<RationalPoly> {
    std::vector<RationalPoly> components{f};  // phi_0 = f
    for (std::size_t step = 0; step < order; ++step) {
        auto polys = adomian_polynomials(n, components);  // A_0..A_step
        if (!polys) {
            return make_error<RationalPoly>(polys.error());
        }
        auto next = op.apply(polys->back());  // phi_{step+1} = lambda 𝒦 A_step
        if (!next) {
            return next;
        }
        components.push_back(std::move(*next));
    }
    return sum_polys(components);
}

}  // namespace

auto adm_solve(const IntegralOperator& op, const RationalPoly& f, const Nonlinearity& n,
               std::size_t order) -> Result<RationalPoly> {
    return adm_engine(op, f, n, order);
}

auto hpm_solve(const IntegralOperator& op, const RationalPoly& f, const Nonlinearity& n,
               std::size_t order) -> Result<RationalPoly> {
    // Same graded recursion as ADM — identical series by construction (ADM == HPM).
    return adm_engine(op, f, n, order);
}

auto ham_solve(const IntegralOperator& op, const RationalPoly& f, const Nonlinearity& n,
               const Rational& hbar, std::size_t order) -> Result<RationalPoly> {
    std::vector<RationalPoly> components{f};  // phi_0 = f
    for (std::size_t m = 1; m <= order; ++m) {
        // A_{m-1} from phi_0..phi_{m-1} (components currently holds exactly those m entries).
        auto polys = adomian_polynomials(n, components);
        if (!polys) {
            return make_error<RationalPoly>(polys.error());
        }
        auto ka = op.apply(polys->back());  // lambda 𝒦 A_{m-1}
        if (!ka) {
            return ka;
        }
        // R_m = phi_{m-1} − δ_{m,1} f − lambda 𝒦 A_{m-1}.
        RationalPoly rm = components.back();  // phi_{m-1}
        if (m == 1) {
            auto d = rm.subtract(f);
            if (!d) {
                return d;
            }
            rm = std::move(*d);
        }
        auto d2 = rm.subtract(*ka);
        if (!d2) {
            return d2;
        }
        rm = std::move(*d2);
        auto scaled = rm.scale(hbar);  // ħ·R_m
        if (!scaled) {
            return scaled;
        }
        // phi_m = χ_m phi_{m-1} + ħ R_m ; χ_m = 0 (m == 1) else 1.
        RationalPoly phi_m = *scaled;
        if (m >= 2) {
            auto s = components.back().add(*scaled);
            if (!s) {
                return s;
            }
            phi_m = std::move(*s);
        }
        components.push_back(std::move(phi_m));
    }
    return sum_polys(components);
}

}  // namespace nimblecas
