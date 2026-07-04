// NimbleCAS Krylov subspace methods (ROADMAP §7.2 — iterative linear algebra).
// @author Olumuyiwa Oluwasanmi
//
// Conforms to config/cpp_details.txt: C++23 modules, `import std`, trailing return
// types, no owning raw pointers, std::expected (Result<T>) railway error handling via
// make_error, no exceptions, [[nodiscard]] everywhere.
//
// ===========================================================================
//  THE HONESTY BOUNDARY — exact-over-Q vs numerical-over-doubles
// ===========================================================================
// This module deliberately splits into two halves, and the split is mathematically
// principled, not cosmetic:
//
//   EXACT over the rationals Q (no rounding, no square roots, terminates):
//     * conjugate_gradient / conjugate_gradient_steps — Conjugate Gradient for a
//       SYMMETRIC POSITIVE-DEFINITE (SPD) Rational matrix. CG is built entirely from
//       inner products, scalar divisions, and axpy updates. It contains NO square
//       root, so over Q every intermediate stays an exact reduced fraction. For an
//       n x n SPD system it reaches the EXACT solution in at most n iterations
//       (finite termination — a theorem, not a tolerance).
//     * krylov_basis — the exact power basis {b, Ab, A^2 b, ...} over Q.
//     * arnoldi_rational / lanczos_rational — Krylov projections built with a
//       rational, UNNORMALISED Gram-Schmidt. We orthogonalise by scaling with the
//       inner-product values <q_i, q_i> (a rational), NEVER by the norm sqrt(<q_i,q_i>)
//       (an irrational). The resulting basis is exactly orthogonal over Q but its
//       vectors are NOT unit vectors. The projected operator is exact: an upper
//       Hessenberg matrix (Arnoldi) or a tridiagonal matrix (symmetric Lanczos) with
//       a UNIT subdiagonal. The classical NORMALISED Arnoldi/Lanczos that yields an
//       orthoNORMAL basis and a SYMMETRIC tridiagonal needs sqrt and is therefore
//       only available numerically (see lanczos_ritz / arnoldi_hessenberg below).
//
//   NUMERICAL over doubles (iterative, round-off present, convergence NOT guaranteed):
//     * gmres (restarted, Givens rotations), minres, bicgstab — solvers for general,
//       large, or non-symmetric / indefinite systems where an exact rational method is
//       inapplicable or impractical. They return {x, iterations, residual, converged}.
//       Running out of iterations is reported as converged == false; it is NOT an
//       error — only genuinely invalid input (an empty system) is a domain_error.
//     * lanczos_ritz / arnoldi_hessenberg — the NORMALISED Lanczos/Arnoldi used to
//       estimate eigenvalues. The returned Ritz values are APPROXIMATIONS of a subset
//       of the spectrum, not exact eigenvalues.
//
// If a routine lives in the exact half its result is bit-for-bit correct; if it lives
// in the numerical half its result is an approximation. This comment is the contract.

export module nimblecas.krylov;

import std;
import nimblecas.core;
import nimblecas.ratpoly;   // Rational
import nimblecas.matrix;    // Matrix (exact dense linear algebra over Q)

// ===========================================================================
//  Public (exported) types.
// ===========================================================================
export namespace nimblecas {

// --- EXACT: Conjugate Gradient result carrying the step count -----------------
// solution is the exact n x 1 Rational solution vector; steps is the number of CG
// iterations actually performed (guaranteed <= n for an SPD system).
struct ExactCGResult {
    Matrix solution{};
    std::size_t steps{0};
};

// --- EXACT: unnormalised rational Arnoldi factorisation -----------------------
// basis            : the mutually orthogonal (over Q) but UNNORMALISED vectors q_0..q_{k-1}.
// gram_diagonal    : the scaling values <q_i, q_i> used in place of norms (all > 0).
// hessenberg       : the k x k exact upper-Hessenberg projected operator H, with unit
//                    subdiagonal entries. Satisfies A q_j = sum_{i<=j} H[i][j] q_i + q_{j+1}.
// breakdown        : true iff an invariant Krylov subspace was reached (q_k == 0) before
//                    m steps — then A basis == basis * hessenberg holds EXACTLY.
struct RationalArnoldi {
    std::vector<std::vector<Rational>> basis{};
    std::vector<Rational> gram_diagonal{};
    Matrix hessenberg{};
    bool breakdown{false};
};

// --- EXACT: unnormalised rational symmetric Lanczos factorisation -------------
// For a SYMMETRIC Rational matrix the Hessenberg collapses to tridiagonal.
// alpha        : the k diagonal entries H[j][j].
// superdiagonal: the k-1 superdiagonal entries H[j-1][j].
// The subdiagonal is the unit (all-ones) sequence, so tridiagonal is NOT symmetric —
// it is the exact companion-style projection over Q. tridiagonal is the full k x k Matrix.
struct RationalLanczos {
    std::vector<std::vector<Rational>> basis{};
    std::vector<Rational> alpha{};
    std::vector<Rational> superdiagonal{};
    Matrix tridiagonal{};
    bool breakdown{false};
};

// --- NUMERICAL: a matrix-free operator y <- A x over doubles ------------------
// The iterative solvers never see the matrix, only its action. Build one from a dense
// buffer with dense_matvec, or supply any std::function (e.g. a sparse/stencil apply).
using MatVec = std::function<void(std::span<const double> x, std::span<double> y)>;

// --- NUMERICAL: outcome of an iterative double solver -------------------------
// residual is the TRUE ||b - A x||_2 recomputed at return (not merely the recursive
// estimate). converged == (residual <= tol * ||b||); false simply means "ran out of
// iterations / stalled", which is a legitimate outcome, not an error.
struct IterativeResult {
    std::vector<double> x{};
    std::size_t iterations{0};
    double residual{0.0};
    bool converged{false};
};

// --- NUMERICAL: dense normalised-Arnoldi Hessenberg (eigenvalue estimation) ---
// h is the dim x dim row-major upper-Hessenberg matrix H = V^T A V from the classical
// (orthoNORMAL) Arnoldi process. Its eigenvalues are the Arnoldi Ritz values — spectral
// APPROXIMATIONS. Extracting them needs a general (non-symmetric) eigensolver, which is
// out of this module's scope; the symmetric case is served directly by lanczos_ritz.
struct DoubleHessenberg {
    std::vector<double> h{};
    std::size_t dim{0};
};

}  // export namespace nimblecas

// ===========================================================================
//  Internal helpers (NOT exported).
// ===========================================================================
namespace nimblecas {
namespace {

using RVec = std::vector<Rational>;

// Exact inner product <a, b> over Q. Fails only on int64 overflow.
[[nodiscard]] auto rdot(std::span<const Rational> a, std::span<const Rational> b) -> Result<Rational> {
    Rational acc{};  // 0/1
    for (std::size_t i = 0; i < a.size(); ++i) {
        auto term = a[i].multiply(b[i]);
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

// Exact matrix-vector product A v over Q (A is rows x cols, v has cols entries).
[[nodiscard]] auto rmatvec(const Matrix& A, std::span<const Rational> v) -> Result<RVec> {
    const std::size_t rows = A.rows();
    const std::size_t cols = A.cols();
    RVec out(rows);
    for (std::size_t i = 0; i < rows; ++i) {
        Rational acc{};
        for (std::size_t j = 0; j < cols; ++j) {
            auto term = A.at(i, j).multiply(v[j]);
            if (!term) {
                return make_error<RVec>(term.error());
            }
            auto sum = acc.add(*term);
            if (!sum) {
                return make_error<RVec>(sum.error());
            }
            acc = *sum;
        }
        out[i] = acc;
    }
    return out;
}

// Exact axpy: returns a + s * b over Q.
[[nodiscard]] auto raxpy(std::span<const Rational> a, const Rational& s,
                         std::span<const Rational> b) -> Result<RVec> {
    RVec out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        auto term = s.multiply(b[i]);
        if (!term) {
            return make_error<RVec>(term.error());
        }
        auto sum = a[i].add(*term);
        if (!sum) {
            return make_error<RVec>(sum.error());
        }
        out[i] = *sum;
    }
    return out;
}

// Core UNNORMALISED rational Arnoldi via modified Gram-Schmidt. Caller has already
// validated that A is square n x n and b has n entries. Builds up to m orthogonal
// (not orthonormal) vectors, orthogonalising by dividing with <q_i, q_i> — never a
// sqrt — so the whole factorisation is exact over Q.
[[nodiscard]] auto arnoldi_core(const Matrix& A, std::span<const Rational> b, std::size_t m)
    -> Result<RationalArnoldi> {
    std::vector<RVec> Q;
    RVec gram;                          // gram[i] = <q_i, q_i>  (always > 0)
    std::vector<RVec> hcolumns;         // hcolumns[j][i] = H[i][j], i = 0..j

    RVec q0(b.begin(), b.end());
    auto g0 = rdot(q0, q0);
    if (!g0) {
        return make_error<RationalArnoldi>(g0.error());
    }
    if (g0->is_zero()) {
        // b == 0: the Krylov space is trivial. Report an empty factorisation with the
        // breakdown flag set (this is a degenerate input, not an error).
        return RationalArnoldi{{}, {}, Matrix{}, true};
    }
    Q.push_back(std::move(q0));
    gram.push_back(*g0);

    bool breakdown = false;
    for (std::size_t j = 0; j < m; ++j) {
        auto w_res = rmatvec(A, Q[j]);
        if (!w_res) {
            return make_error<RationalArnoldi>(w_res.error());
        }
        RVec w = std::move(*w_res);
        RVec hcol(j + 1, Rational{});
        for (std::size_t i = 0; i <= j; ++i) {
            auto num = rdot(w, Q[i]);
            if (!num) {
                return make_error<RationalArnoldi>(num.error());
            }
            auto h = num->divide(gram[i]);  // gram[i] > 0, division is safe
            if (!h) {
                return make_error<RationalArnoldi>(h.error());
            }
            hcol[i] = *h;
            auto neg = h->negate();
            if (!neg) {
                return make_error<RationalArnoldi>(neg.error());
            }
            auto w2 = raxpy(w, *neg, Q[i]);  // w <- w - H[i][j] q_i
            if (!w2) {
                return make_error<RationalArnoldi>(w2.error());
            }
            w = std::move(*w2);
        }
        hcolumns.push_back(std::move(hcol));

        if (j + 1 >= m) {
            break;  // requested dimension reached; last subdiagonal is truncated
        }
        auto gnext = rdot(w, w);
        if (!gnext) {
            return make_error<RationalArnoldi>(gnext.error());
        }
        if (gnext->is_zero()) {
            breakdown = true;  // invariant subspace: A basis == basis * H exactly
            break;
        }
        Q.push_back(std::move(w));
        gram.push_back(*gnext);
    }

    const std::size_t k = Q.size();
    std::vector<std::vector<Rational>> H(k, std::vector<Rational>(k, Rational{}));
    const Rational one = Rational::from_int(1);
    for (std::size_t j = 0; j < k; ++j) {
        const RVec& hcol = hcolumns[j];
        for (std::size_t i = 0; i < hcol.size() && i < k; ++i) {
            H[i][j] = hcol[i];
        }
        if (j + 1 < k) {
            H[j + 1][j] = one;  // unit subdiagonal of the unnormalised projection
        }
    }
    auto Hmat = Matrix::from_rows(std::move(H));
    if (!Hmat) {
        return make_error<RationalArnoldi>(Hmat.error());
    }
    return RationalArnoldi{std::move(Q), std::move(gram), std::move(*Hmat), breakdown};
}

// --- double helpers for the numerical solvers ---------------------------------

[[nodiscard]] auto ddot(std::span<const double> a, std::span<const double> b) noexcept -> double {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        s = std::fma(a[i], b[i], s);
    }
    return s;
}

[[nodiscard]] auto dnorm(std::span<const double> a) noexcept -> double {
    return std::sqrt(ddot(a, a));
}

// The TRUE residual ||b - A x||_2, recomputed directly from the operator.
[[nodiscard]] auto residual_norm(const MatVec& A, std::span<const double> b,
                                 std::span<const double> x) -> double {
    std::vector<double> ax(b.size(), 0.0);
    A(x, ax);
    double s = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        const double d = b[i] - ax[i];
        s = std::fma(d, d, s);
    }
    return std::sqrt(s);
}

// Eigenvalues of a symmetric tridiagonal matrix in place (QL with implicit shifts,
// the classic EISPACK/`tqli` recurrence, eigenvalues only). d holds the diagonal;
// e[i] is the off-diagonal linking d[i] and d[i+1] with e[n-1] == 0. NUMERICAL.
[[nodiscard]] auto tqli_eigenvalues(std::vector<double>& d, std::vector<double>& e)
    -> Result<void> {
    const std::size_t n = d.size();
    const double eps = std::numeric_limits<double>::epsilon();
    for (std::size_t l = 0; l < n; ++l) {
        std::size_t iter = 0;
        std::size_t m = l;
        do {
            for (m = l; m + 1 < n; ++m) {
                const double dd = std::fabs(d[m]) + std::fabs(d[m + 1]);
                if (std::fabs(e[m]) <= eps * dd) {
                    break;
                }
            }
            if (m != l) {
                if (iter++ == 50) {
                    return make_error<void>(MathError::not_implemented);  // did not converge
                }
                double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                double r = std::hypot(g, 1.0);
                const double sign_r = (g >= 0.0) ? std::fabs(r) : -std::fabs(r);
                g = d[m] - d[l] + e[l] / (g + sign_r);
                double s = 1.0;
                double c = 1.0;
                double p = 0.0;
                bool zeroed = false;
                std::size_t i = m;
                for (i = m; i-- > l;) {  // i = m-1, m-2, ..., l
                    double f = s * e[i];
                    const double bb = c * e[i];
                    r = std::hypot(f, g);
                    e[i + 1] = r;
                    if (r == 0.0) {
                        d[i + 1] -= p;
                        e[m] = 0.0;
                        zeroed = true;
                        break;
                    }
                    s = f / r;
                    c = g / r;
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * bb;
                    p = s * r;
                    d[i + 1] = g + p;
                    g = c * r - bb;
                }
                if (zeroed) {
                    continue;
                }
                d[l] -= p;
                e[l] = g;
                e[m] = 0.0;
            }
        } while (m != l);
    }
    return {};
}

}  // namespace
}  // namespace nimblecas

// ===========================================================================
//  Public (exported) API.
// ===========================================================================
export namespace nimblecas {

// ---------------------------------------------------------------------------
//  EXACT over Q — Conjugate Gradient for an SPD Rational system.
// ---------------------------------------------------------------------------
// Solves A x = b EXACTLY for a symmetric positive-definite Rational matrix A and an
// n x 1 Rational column b. CG uses only <.,.> inner products, scalar divisions, and
// axpy, so there is no sqrt and the arithmetic stays in Q; the solution is reached in
// at most n steps. Fails with domain_error when A is not square, not symmetric, b has
// the wrong shape, or a breakdown reveals A is not positive definite (a direction with
// p^T A p <= 0, or non-termination within n steps).
[[nodiscard]] auto conjugate_gradient_steps(const Matrix& A, const Matrix& b)
    -> Result<ExactCGResult> {
    if (!A.is_square()) {
        return make_error<ExactCGResult>(MathError::domain_error);
    }
    const std::size_t n = A.rows();
    if (b.rows() != n || b.cols() != 1) {
        return make_error<ExactCGResult>(MathError::domain_error);
    }
    auto At = A.transpose();
    if (!At) {
        return make_error<ExactCGResult>(At.error());
    }
    if (!(*At == A)) {
        return make_error<ExactCGResult>(MathError::domain_error);  // not symmetric
    }

    RVec bx(n);
    for (std::size_t i = 0; i < n; ++i) {
        bx[i] = b.at(i, 0);
    }

    RVec x(n);        // x_0 = 0
    RVec r = bx;      // r_0 = b - A x_0 = b
    RVec p = r;       // p_0 = r_0
    auto rtr_res = rdot(r, r);
    if (!rtr_res) {
        return make_error<ExactCGResult>(rtr_res.error());
    }
    Rational rtr = *rtr_res;

    std::size_t steps = 0;
    for (std::size_t k = 0; k < n; ++k) {
        if (rtr.is_zero()) {
            break;  // r == 0: exact solution found
        }
        auto Ap_res = rmatvec(A, p);
        if (!Ap_res) {
            return make_error<ExactCGResult>(Ap_res.error());
        }
        const RVec Ap = std::move(*Ap_res);
        auto denom_res = rdot(p, Ap);
        if (!denom_res) {
            return make_error<ExactCGResult>(denom_res.error());
        }
        const Rational denom = *denom_res;
        // For an SPD A and p != 0 we have p^T A p > 0. A non-positive value is a
        // definite proof that A is NOT positive definite -> reject as domain_error.
        if (denom.numerator() <= 0) {
            return make_error<ExactCGResult>(MathError::domain_error);
        }
        auto alpha_res = rtr.divide(denom);
        if (!alpha_res) {
            return make_error<ExactCGResult>(alpha_res.error());
        }
        const Rational alpha = *alpha_res;

        auto x_new = raxpy(x, alpha, p);  // x <- x + alpha p
        if (!x_new) {
            return make_error<ExactCGResult>(x_new.error());
        }
        x = std::move(*x_new);

        auto neg_alpha = alpha.negate();
        if (!neg_alpha) {
            return make_error<ExactCGResult>(neg_alpha.error());
        }
        auto r_new = raxpy(r, *neg_alpha, Ap);  // r <- r - alpha A p
        if (!r_new) {
            return make_error<ExactCGResult>(r_new.error());
        }
        r = std::move(*r_new);
        ++steps;

        auto rtr_new_res = rdot(r, r);
        if (!rtr_new_res) {
            return make_error<ExactCGResult>(rtr_new_res.error());
        }
        const Rational rtr_new = *rtr_new_res;
        if (rtr_new.is_zero()) {
            rtr = rtr_new;
            break;  // converged
        }
        auto beta_res = rtr_new.divide(rtr);
        if (!beta_res) {
            return make_error<ExactCGResult>(beta_res.error());
        }
        const Rational beta = *beta_res;
        auto p_new = raxpy(r, beta, p);  // p <- r + beta p
        if (!p_new) {
            return make_error<ExactCGResult>(p_new.error());
        }
        p = std::move(*p_new);
        rtr = rtr_new;
    }

    if (!rtr.is_zero()) {
        // No exact solution within n steps => A is not SPD (breakdown).
        return make_error<ExactCGResult>(MathError::domain_error);
    }
    std::vector<std::vector<Rational>> rows;
    rows.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        rows.push_back(std::vector<Rational>{x[i]});
    }
    auto sol = Matrix::from_rows(std::move(rows));
    if (!sol) {
        return make_error<ExactCGResult>(sol.error());
    }
    return ExactCGResult{std::move(*sol), steps};
}

// Convenience wrapper returning just the exact Rational solution vector.
[[nodiscard]] auto conjugate_gradient(const Matrix& A, const Matrix& b) -> Result<Matrix> {
    auto res = conjugate_gradient_steps(A, b);
    if (!res) {
        return make_error<Matrix>(res.error());
    }
    return std::move(res->solution);
}

// ---------------------------------------------------------------------------
//  EXACT over Q — the power Krylov basis {b, Ab, A^2 b, ..., A^{m-1} b}.
// ---------------------------------------------------------------------------
// Returns the first m Krylov vectors exactly. Fails with domain_error when A is not
// square or b does not match its dimension.
[[nodiscard]] auto krylov_basis(const Matrix& A, std::span<const Rational> b, std::size_t m)
    -> Result<std::vector<std::vector<Rational>>> {
    if (!A.is_square()) {
        return make_error<std::vector<std::vector<Rational>>>(MathError::domain_error);
    }
    if (b.size() != A.rows()) {
        return make_error<std::vector<std::vector<Rational>>>(MathError::domain_error);
    }
    std::vector<std::vector<Rational>> basis;
    basis.reserve(m);
    if (m == 0) {
        return basis;
    }
    RVec cur(b.begin(), b.end());
    basis.push_back(cur);
    for (std::size_t k = 1; k < m; ++k) {
        auto nxt = rmatvec(A, cur);
        if (!nxt) {
            return make_error<std::vector<std::vector<Rational>>>(nxt.error());
        }
        cur = std::move(*nxt);
        basis.push_back(cur);
    }
    return basis;
}

// ---------------------------------------------------------------------------
//  EXACT over Q — unnormalised rational Arnoldi (upper Hessenberg projection).
// ---------------------------------------------------------------------------
// Builds up to min(m, n) mutually orthogonal (over Q) but UNNORMALISED Krylov vectors
// and the exact upper-Hessenberg projected operator. m == 0 yields an empty result.
[[nodiscard]] auto arnoldi_rational(const Matrix& A, std::span<const Rational> b, std::size_t m)
    -> Result<RationalArnoldi> {
    if (!A.is_square()) {
        return make_error<RationalArnoldi>(MathError::domain_error);
    }
    const std::size_t n = A.rows();
    if (b.size() != n) {
        return make_error<RationalArnoldi>(MathError::domain_error);
    }
    if (m == 0) {
        return RationalArnoldi{{}, {}, Matrix{}, false};
    }
    return arnoldi_core(A, b, std::min(m, n));
}

// ---------------------------------------------------------------------------
//  EXACT over Q — unnormalised rational symmetric Lanczos (tridiagonal projection).
// ---------------------------------------------------------------------------
// Requires a SYMMETRIC Rational matrix (domain_error otherwise). The Hessenberg
// projection collapses to tridiagonal; this returns its diagonal and superdiagonal
// exactly (the subdiagonal being the unit sequence). NOTE: the classical Lanczos with
// an orthoNORMAL basis and a SYMMETRIC tridiagonal needs sqrt and is unavailable over Q
// — see lanczos_ritz for the numerical, normalised variant.
[[nodiscard]] auto lanczos_rational(const Matrix& A, std::span<const Rational> b, std::size_t m)
    -> Result<RationalLanczos> {
    if (!A.is_square()) {
        return make_error<RationalLanczos>(MathError::domain_error);
    }
    const std::size_t n = A.rows();
    if (b.size() != n) {
        return make_error<RationalLanczos>(MathError::domain_error);
    }
    auto At = A.transpose();
    if (!At) {
        return make_error<RationalLanczos>(At.error());
    }
    if (!(*At == A)) {
        return make_error<RationalLanczos>(MathError::domain_error);  // Lanczos needs symmetry
    }
    if (m == 0) {
        return RationalLanczos{{}, {}, {}, Matrix{}, false};
    }
    auto arn = arnoldi_core(A, b, std::min(m, n));
    if (!arn) {
        return make_error<RationalLanczos>(arn.error());
    }
    const std::size_t k = arn->basis.size();
    std::vector<Rational> alpha;
    std::vector<Rational> superdiag;
    alpha.reserve(k);
    if (k > 0) {
        superdiag.reserve(k - 1);
    }
    for (std::size_t j = 0; j < k; ++j) {
        alpha.push_back(arn->hessenberg.at(j, j));
        if (j > 0) {
            superdiag.push_back(arn->hessenberg.at(j - 1, j));
        }
    }
    return RationalLanczos{std::move(arn->basis), std::move(alpha), std::move(superdiag),
                           std::move(arn->hessenberg), arn->breakdown};
}

// ---------------------------------------------------------------------------
//  NUMERICAL over doubles — dense operator factory.
// ---------------------------------------------------------------------------
// Wraps an n x n row-major dense buffer as a MatVec. The buffer is COPIED into the
// returned closure so it stays valid independently of the caller's storage.
[[nodiscard]] auto dense_matvec(std::span<const double> mat, std::size_t n) -> MatVec {
    std::vector<double> a(mat.begin(), mat.end());
    return [a = std::move(a), n](std::span<const double> x, std::span<double> y) {
        for (std::size_t i = 0; i < n; ++i) {
            double acc = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                acc = std::fma(a[i * n + j], x[j], acc);
            }
            y[i] = acc;
        }
    };
}

// ---------------------------------------------------------------------------
//  NUMERICAL over doubles — restarted GMRES with Givens rotations.
// ---------------------------------------------------------------------------
// General (possibly non-symmetric) systems. tol is RELATIVE to ||b||. Returns
// converged == false (not an error) when the iteration budget is exhausted; a
// domain_error is reserved for an empty system.
[[nodiscard]] auto gmres(const MatVec& A, std::span<const double> b, double tol = 1e-10,
                         std::size_t max_iter = 1000, std::size_t restart = 30)
    -> Result<IterativeResult> {
    const std::size_t n = b.size();
    if (n == 0) {
        return make_error<IterativeResult>(MathError::domain_error);
    }
    if (restart == 0) {
        restart = std::min<std::size_t>(n, max_iter == 0 ? n : max_iter);
    }
    if (restart == 0) {
        restart = 1;
    }
    const double bnorm = dnorm(b);
    const double stop = tol * (bnorm > 0.0 ? bnorm : 1.0);

    std::vector<double> x(n, 0.0);
    std::vector<double> cs(restart, 0.0);
    std::vector<double> sn(restart, 0.0);
    std::vector<double> g(restart + 1, 0.0);
    std::vector<std::vector<double>> V;
    std::size_t iter = 0;
    bool converged = false;

    while (iter < max_iter && !converged) {
        std::vector<double> r(n, 0.0);
        A(x, r);
        for (std::size_t i = 0; i < n; ++i) {
            r[i] = b[i] - r[i];
        }
        const double beta = dnorm(r);
        if (beta <= stop) {
            converged = true;
            break;
        }
        V.clear();
        V.emplace_back(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            V[0][i] = r[i] / beta;
        }
        std::ranges::fill(g, 0.0);
        g[0] = beta;

        std::vector<std::vector<double>> Hc;  // Hc[j] is column j, length j+2
        for (std::size_t j = 0; j < restart && iter < max_iter; ++j) {
            std::vector<double> w(n, 0.0);
            A(V[j], w);
            std::vector<double> hcol(j + 2, 0.0);
            for (std::size_t i = 0; i <= j; ++i) {
                const double hij = ddot(w, V[i]);
                hcol[i] = hij;
                for (std::size_t t = 0; t < n; ++t) {
                    w[t] -= hij * V[i][t];
                }
            }
            const double hnext = dnorm(w);
            hcol[j + 1] = hnext;
            if (hnext > 0.0) {
                std::vector<double> vnext(n, 0.0);
                for (std::size_t t = 0; t < n; ++t) {
                    vnext[t] = w[t] / hnext;
                }
                V.push_back(std::move(vnext));
            } else {
                V.emplace_back(n, 0.0);  // lucky breakdown; this vector is unused
            }
            // Apply the previously accumulated Givens rotations to the new column.
            for (std::size_t i = 0; i < j; ++i) {
                const double tmp = cs[i] * hcol[i] + sn[i] * hcol[i + 1];
                hcol[i + 1] = -sn[i] * hcol[i] + cs[i] * hcol[i + 1];
                hcol[i] = tmp;
            }
            // Compute and apply the rotation that annihilates the subdiagonal entry.
            const double denom = std::hypot(hcol[j], hcol[j + 1]);
            if (denom == 0.0) {
                cs[j] = 1.0;
                sn[j] = 0.0;
            } else {
                cs[j] = hcol[j] / denom;
                sn[j] = hcol[j + 1] / denom;
            }
            const double gtmp = cs[j] * g[j];
            g[j + 1] = -sn[j] * g[j];
            g[j] = gtmp;
            hcol[j] = cs[j] * hcol[j] + sn[j] * hcol[j + 1];
            hcol[j + 1] = 0.0;
            Hc.push_back(std::move(hcol));
            ++iter;
            if (std::fabs(g[j + 1]) <= stop) {
                converged = true;
                break;
            }
        }

        // Back-substitute the upper-triangular least-squares system H y = g.
        const std::size_t kk = Hc.size();
        std::vector<double> y(kk, 0.0);
        for (std::size_t ii = kk; ii-- > 0;) {
            double sum = g[ii];
            for (std::size_t c = ii + 1; c < kk; ++c) {
                sum -= Hc[c][ii] * y[c];
            }
            const double diag = Hc[ii][ii];
            y[ii] = (diag != 0.0) ? sum / diag : 0.0;
        }
        for (std::size_t c = 0; c < kk; ++c) {
            for (std::size_t t = 0; t < n; ++t) {
                x[t] += y[c] * V[c][t];
            }
        }
        if (kk == 0) {
            break;  // safety: made no progress this cycle
        }
    }

    const double tr = residual_norm(A, b, x);
    return IterativeResult{std::move(x), iter, tr, tr <= stop};
}

// ---------------------------------------------------------------------------
//  NUMERICAL over doubles — MINRES (symmetric, possibly indefinite).
// ---------------------------------------------------------------------------
// Paige-Saunders MINRES. tol is RELATIVE to ||b||. Out of iterations => converged==false.
[[nodiscard]] auto minres(const MatVec& A, std::span<const double> b, double tol = 1e-10,
                          std::size_t max_iter = 1000) -> Result<IterativeResult> {
    const std::size_t n = b.size();
    if (n == 0) {
        return make_error<IterativeResult>(MathError::domain_error);
    }
    std::vector<double> x(n, 0.0);
    std::vector<double> r1(b.begin(), b.end());  // r1 = b - A x_0 = b
    std::vector<double> y = r1;                   // unpreconditioned: y = r1
    const double beta1 = std::sqrt(ddot(r1, y));
    const double bnorm = beta1;
    const double stop = tol * (bnorm > 0.0 ? bnorm : 1.0);
    if (beta1 == 0.0) {
        return IterativeResult{std::move(x), 0, 0.0, true};
    }

    double oldb = 0.0;
    double beta = beta1;
    double dbar = 0.0;
    double epsln = 0.0;
    double phibar = beta1;
    double cs = -1.0;
    double sn = 0.0;
    std::vector<double> w(n, 0.0);
    std::vector<double> w1(n, 0.0);
    std::vector<double> w2(n, 0.0);
    std::vector<double> r2 = r1;
    std::vector<double> v(n, 0.0);

    std::size_t iter = 0;
    while (iter < max_iter) {
        ++iter;
        const double s = 1.0 / beta;
        for (std::size_t i = 0; i < n; ++i) {
            v[i] = s * y[i];
        }
        A(v, y);  // y = A v
        if (iter >= 2) {
            const double coef = beta / oldb;
            for (std::size_t i = 0; i < n; ++i) {
                y[i] -= coef * r1[i];
            }
        }
        const double alfa = ddot(v, y);
        const double coef2 = alfa / beta;
        for (std::size_t i = 0; i < n; ++i) {
            y[i] -= coef2 * r2[i];
        }
        r1 = r2;
        r2 = y;
        oldb = beta;
        beta = std::sqrt(ddot(r2, r2));

        const double oldeps = epsln;
        const double delta = cs * dbar + sn * alfa;
        const double gbar = sn * dbar - cs * alfa;
        epsln = sn * beta;
        dbar = -cs * beta;

        double gamma = std::sqrt(gbar * gbar + beta * beta);
        gamma = std::max(gamma, 1e-300);
        cs = gbar / gamma;
        sn = beta / gamma;
        const double phi = cs * phibar;
        phibar = sn * phibar;

        const double invg = 1.0 / gamma;
        w1 = w2;
        w2 = w;
        for (std::size_t i = 0; i < n; ++i) {
            w[i] = (v[i] - oldeps * w1[i] - delta * w2[i]) * invg;
        }
        for (std::size_t i = 0; i < n; ++i) {
            x[i] += phi * w[i];
        }
        if (phibar <= stop) {
            break;
        }
        if (beta == 0.0) {
            break;  // exact invariant subspace
        }
    }

    const double tr = residual_norm(A, b, x);
    return IterativeResult{std::move(x), iter, tr, tr <= stop};
}

// ---------------------------------------------------------------------------
//  NUMERICAL over doubles — BiCGSTAB (general non-symmetric).
// ---------------------------------------------------------------------------
// tol is RELATIVE to ||b||. Breakdown (rho or omega collapsing to zero) stops the
// iteration and is reported as converged == false, not as an error.
[[nodiscard]] auto bicgstab(const MatVec& A, std::span<const double> b, double tol = 1e-10,
                            std::size_t max_iter = 1000) -> Result<IterativeResult> {
    const std::size_t n = b.size();
    if (n == 0) {
        return make_error<IterativeResult>(MathError::domain_error);
    }
    std::vector<double> x(n, 0.0);
    std::vector<double> r(b.begin(), b.end());  // r = b - A x_0 = b
    const std::vector<double> rhat = r;         // fixed shadow residual r0_hat
    const double bnorm = dnorm(b);
    const double stop = tol * (bnorm > 0.0 ? bnorm : 1.0);
    constexpr double tiny = 1e-300;

    if (dnorm(r) <= stop) {
        const double tr0 = residual_norm(A, b, x);
        return IterativeResult{std::move(x), 0, tr0, true};
    }

    double rho_old = 1.0;
    double alpha = 1.0;
    double omega = 1.0;
    std::vector<double> v(n, 0.0);
    std::vector<double> p(n, 0.0);
    std::vector<double> s(n, 0.0);
    std::vector<double> t(n, 0.0);

    std::size_t iter = 0;
    while (iter < max_iter) {
        ++iter;
        const double rho = ddot(rhat, r);
        if (std::fabs(rho) <= tiny) {
            break;  // breakdown
        }
        const double beta = (rho / rho_old) * (alpha / omega);
        for (std::size_t i = 0; i < n; ++i) {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }
        A(p, v);
        const double rhatv = ddot(rhat, v);
        if (std::fabs(rhatv) <= tiny) {
            break;  // breakdown
        }
        alpha = rho / rhatv;
        for (std::size_t i = 0; i < n; ++i) {
            s[i] = r[i] - alpha * v[i];
        }
        if (dnorm(s) <= stop) {
            for (std::size_t i = 0; i < n; ++i) {
                x[i] += alpha * p[i];
            }
            break;
        }
        A(s, t);
        const double tt = ddot(t, t);
        if (tt <= tiny) {
            break;  // breakdown
        }
        omega = ddot(t, s) / tt;
        for (std::size_t i = 0; i < n; ++i) {
            x[i] += alpha * p[i] + omega * s[i];
        }
        for (std::size_t i = 0; i < n; ++i) {
            r[i] = s[i] - omega * t[i];
        }
        rho_old = rho;
        if (dnorm(r) <= stop) {
            break;
        }
        if (std::fabs(omega) <= tiny) {
            break;  // breakdown
        }
    }

    const double tr = residual_norm(A, b, x);
    return IterativeResult{std::move(x), iter, tr, tr <= stop};
}

// ---------------------------------------------------------------------------
//  NUMERICAL over doubles — Lanczos Ritz-value estimation (symmetric A).
// ---------------------------------------------------------------------------
// Runs min(m, n) steps of the classical NORMALISED Lanczos on a symmetric operator and
// returns the eigenvalues of the resulting tridiagonal matrix — the Ritz values, sorted
// ascending. These are APPROXIMATIONS of a subset of A's spectrum (extreme eigenvalues
// emerge first), NOT exact eigenvalues. Fails with domain_error on an empty system or a
// zero starting vector, and not_implemented if the tridiagonal QL iteration stalls.
[[nodiscard]] auto lanczos_ritz(const MatVec& A, std::span<const double> v0, std::size_t m)
    -> Result<std::vector<double>> {
    const std::size_t n = v0.size();
    if (n == 0 || m == 0) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    const std::size_t mm = std::min(m, n);
    const double nv = dnorm(v0);
    if (nv == 0.0) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    std::vector<double> vprev(n, 0.0);
    std::vector<double> v(n, 0.0);
    std::vector<double> w(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = v0[i] / nv;
    }
    std::vector<double> alpha;
    std::vector<double> off;  // off[j] links alpha[j] and alpha[j+1]
    alpha.reserve(mm);
    double betaj = 0.0;
    for (std::size_t j = 0; j < mm; ++j) {
        A(v, w);  // w = A v
        const double a = ddot(v, w);
        alpha.push_back(a);
        for (std::size_t i = 0; i < n; ++i) {
            w[i] -= a * v[i] + betaj * vprev[i];
        }
        const double bnext = dnorm(w);
        if (bnext <= 1e-300) {
            break;  // invariant subspace reached
        }
        if (j + 1 < mm) {
            vprev = v;
            for (std::size_t i = 0; i < n; ++i) {
                v[i] = w[i] / bnext;
            }
            off.push_back(bnext);
            betaj = bnext;
        }
    }

    const std::size_t k = alpha.size();
    std::vector<double> d = alpha;
    std::vector<double> e(k, 0.0);
    for (std::size_t i = 0; i + 1 < k; ++i) {
        e[i] = off[i];
    }
    auto st = tqli_eigenvalues(d, e);
    if (!st) {
        return make_error<std::vector<double>>(st.error());
    }
    std::ranges::sort(d);
    return d;
}

// ---------------------------------------------------------------------------
//  NUMERICAL over doubles — normalised Arnoldi Hessenberg (non-symmetric A).
// ---------------------------------------------------------------------------
// Runs min(m, n) steps of the classical orthoNORMAL Arnoldi and returns the dim x dim
// upper-Hessenberg H = V^T A V. Its eigenvalues are the Arnoldi Ritz values — spectral
// APPROXIMATIONS whose extraction needs a general eigensolver (out of scope here).
[[nodiscard]] auto arnoldi_hessenberg(const MatVec& A, std::span<const double> v0, std::size_t m)
    -> Result<DoubleHessenberg> {
    const std::size_t n = v0.size();
    if (n == 0 || m == 0) {
        return make_error<DoubleHessenberg>(MathError::domain_error);
    }
    const std::size_t mm = std::min(m, n);
    const double nv = dnorm(v0);
    if (nv == 0.0) {
        return make_error<DoubleHessenberg>(MathError::domain_error);
    }
    std::vector<std::vector<double>> V;
    V.reserve(mm + 1);
    V.emplace_back(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        V[0][i] = v0[i] / nv;
    }
    std::vector<double> H(mm * mm, 0.0);
    std::size_t dim = 0;
    for (std::size_t j = 0; j < mm; ++j) {
        dim = j + 1;
        std::vector<double> w(n, 0.0);
        A(V[j], w);
        for (std::size_t i = 0; i <= j; ++i) {
            const double hij = ddot(w, V[i]);
            H[i * mm + j] = hij;
            for (std::size_t t = 0; t < n; ++t) {
                w[t] -= hij * V[i][t];
            }
        }
        const double hnext = dnorm(w);
        if (j + 1 < mm) {
            H[(j + 1) * mm + j] = hnext;
            if (hnext <= 1e-300) {
                break;  // invariant subspace
            }
            std::vector<double> vn(n, 0.0);
            for (std::size_t t = 0; t < n; ++t) {
                vn[t] = w[t] / hnext;
            }
            V.push_back(std::move(vn));
        }
    }
    // On an invariant-subspace breakdown dim < mm, but H was laid out with row stride mm.
    // Repack the valid leading dim x dim block into a contiguous buffer so the returned
    // {h, dim} honors its documented "dim x dim row-major" contract (a consumer indexing
    // h[i*dim + j] must not read foreign-row entries).
    if (dim < mm) {
        std::vector<double> Hc(dim * dim, 0.0);
        for (std::size_t i = 0; i < dim; ++i) {
            for (std::size_t j = 0; j < dim; ++j) {
                Hc[i * dim + j] = H[i * mm + j];
            }
        }
        return DoubleHessenberg{std::move(Hc), dim};
    }
    return DoubleHessenberg{std::move(H), dim};
}

}  // export namespace nimblecas
