// NimbleCAS QR decomposition and real Schur (quasi-triangular) form (ROADMAP 7.2x).
// @author Olumuyiwa Oluwasanmi
//
// This module answers two related factorisation questions and is scrupulous about which
// answers are EXACT (over the field Q) and which are NUMERIC (double-precision), because a
// truly orthonormal QR of a rational matrix is, in general, NOT representable over Q at all
// (the column 2-norms are irrational). Rather than fake it, the two regimes are separated.
//
// HONESTY BOUNDARY (Rule 32).
//
//   (1) exact_orthogonal_qr  — EXACT over Q, but ORTHOGONAL (not orthonormal). Classical
//       Gram-Schmidt produces A = Q * R where Q's columns are mutually orthogonal but NOT
//       unit length, so Qᵀ·Q is DIAGONAL (its diagonal entries are the exact rational
//       squared pseudo-norms ⟨qₖ,qₖ⟩) and R is upper-triangular with a UNIT diagonal.
//       Every entry of Q and R is an exact Rational; the reconstruction Q·R == A holds
//       identically, not "up to rounding". A rank-deficient input (a column linearly
//       dependent on the earlier ones) makes some pseudo-norm ⟨qₖ,qₖ⟩ = 0; the algorithm
//       would then divide by that zero, so it returns an honest domain_error instead of a
//       bogus factor. This is the ONLY QR that can be exact, and it is deliberately labelled
//       "orthogonal", never "orthonormal".
//
//   (2) numeric_qr — NUMERIC (double). Householder QR of a real matrix: Q is genuinely
//       ORTHONORMAL (Qᵀ·Q = I to rounding) and R is upper-triangular, with Q·R ≈ A. The
//       result is a double approximation accurate to ~machine epsilon, never presented as
//       exact. Rank deficiency is NOT an error here — a zero pivot column simply leaves a
//       zero on R's diagonal; Q stays orthonormal and Q·R ≈ A regardless.
//
//   (3) real_schur — NUMERIC (double). Real Schur form A = Q·T·Qᵀ with Q orthogonal and T
//       quasi-upper-triangular (1×1 blocks for real eigenvalues, 2×2 blocks for
//       complex-conjugate pairs), by orthogonal (Householder) Hessenberg reduction followed
//       by the Francis double-shift QR iteration with deflation — the same algorithm family
//       as nimblecas.numeigen's hqr, but with the orthogonal transformations ACCUMULATED
//       into Q (numeigen's Gaussian elmhes is eigenvalue-only and cannot yield an orthogonal
//       Q, so an orthogonal orthes reduction is used here instead). A block that fails to
//       deflate within max_iter sweeps returns MathError::not_implemented — the honest
//       "not solved" signal, never a partial or garbage T. T's eigenvalues (read off its
//       1×1/2×2 blocks by schur_eigenvalues) agree with nimblecas.numeigen::eigenvalues_qr.
//
// Every factorisation is residual-checkable: qr_residual / schur_residual recompute
// ‖Q·R − A‖_F and ‖Q·T·Qᵀ − A‖_F, and orthonormality_defect recomputes ‖QᵀQ − I‖_F, so a
// caller can independently verify the numeric results (and the exact Q·R == A via Matrix).

export module nimblecas.qrschur;

import std;
import nimblecas.core;
import nimblecas.matrix;
import nimblecas.ratpoly;
import nimblecas.numeigen;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// (1) Exact orthogonal decomposition over Q — A = Q * R (Gram-Schmidt).
// ---------------------------------------------------------------------------
// `q` is m×n with MUTUALLY ORTHOGONAL columns: Qᵀ·Q is diagonal, its k-th diagonal entry
// being the exact rational squared pseudo-norm ⟨qₖ,qₖ⟩ (NOT 1 — the columns are not
// normalised, because unit-normalising would leave Q). `r` is n×n upper-triangular with a
// unit diagonal (r(k,k) == 1). The reconstruction Q·R equals A exactly over Q.
struct ExactOrthogonalQr {
    Matrix q;  // m×n, columns mutually orthogonal (Qᵀ·Q diagonal, exact over Q)
    Matrix r;  // n×n, upper-triangular, unit diagonal
};

// Exact Gram-Schmidt orthogonal decomposition A = Q·R over Q. Q has mutually orthogonal
// columns (Qᵀ·Q diagonal) and R is upper-triangular with unit diagonal; Q·R == A exactly.
//
// Fails with domain_error when A is rank-deficient (some column lies in the span of the
// earlier ones, giving a zero pseudo-norm the algorithm would divide by) — including every
// wide m<n matrix and every matrix with a zero row-space such as a 0×n one. Entry-arithmetic
// overflow over int64 propagates as overflow. A 0×0 (or m×0) input yields empty Q, R.
[[nodiscard]] auto exact_orthogonal_qr(const Matrix& a) -> Result<ExactOrthogonalQr>;

// ---------------------------------------------------------------------------
// (2) Numeric Householder QR (double) — A = Q * R, Q orthonormal.
// ---------------------------------------------------------------------------
// `q` is the m×m orthonormal factor (Qᵀ·Q = I to rounding) and `r` is the m×n
// upper-triangular factor, both row-major, with Q·R ≈ A. NUMERIC: accurate to ~machine
// epsilon, not exact.
struct NumericQr {
    std::size_t rows{};       // m
    std::size_t cols{};       // n
    std::vector<double> q{};  // m×m orthonormal, row-major
    std::vector<double> r{};  // m×n upper-triangular, row-major
};

// Numeric Householder QR of the m×n real matrix given row-major in `a` (length m·n).
// Returns an orthonormal Q (m×m) and upper-triangular R (m×n) with Q·R ≈ A. Handles
// rank-deficient and wide (m<n) inputs without error (a zero pivot column just yields a
// zero on R's diagonal). Fails with domain_error when a.size() != m·n, overflow when the
// m·m / m·n index arithmetic would wrap. A 0×0 input yields empty factors.
[[nodiscard]] auto numeric_qr(std::span<const double> a, std::size_t rows, std::size_t cols)
    -> Result<NumericQr>;

// ---------------------------------------------------------------------------
// (3) Numeric real Schur form (double) — A = Q * T * Qᵀ.
// ---------------------------------------------------------------------------
// `q` is orthogonal (n×n) and `t` is the quasi-upper-triangular real Schur form: 1×1
// diagonal blocks carry real eigenvalues, 2×2 blocks (with a non-negligible subdiagonal
// entry) carry complex-conjugate eigenvalue pairs. Row-major; NUMERIC.
struct NumericSchur {
    std::size_t n{};          // matrix dimension
    std::vector<double> q{};  // n×n orthogonal Schur vectors, row-major
    std::vector<double> t{};  // n×n quasi-upper-triangular real Schur form, row-major
};

// Numeric real Schur decomposition A = Q·T·Qᵀ of the n×n real matrix given row-major in `a`
// (length n·n): orthogonal Householder Hessenberg reduction, then Francis double-shift QR
// with deflation, accumulating the orthogonal transformations into Q. `tol` sets the
// relative subdiagonal deflation threshold, `max_iter` caps the QR sweeps per deflated
// block. Fails with domain_error when a.size() != n·n, overflow when n·n would wrap, and
// not_implemented when a block does not converge within max_iter (an honest "not solved"
// signal — never a partial/garbage T). A 0×0 input yields an empty decomposition.
[[nodiscard]] auto real_schur(std::span<const double> a, std::size_t n, double tol = 1e-12,
                              std::size_t max_iter = 1000) -> Result<NumericSchur>;

// Eigenvalues read off the 1×1 and 2×2 diagonal blocks of a real Schur form. A 2×2 block is
// recognised when its subdiagonal entry exceeds `tol` relative to the block's diagonal
// magnitude; it yields a real pair or a complex-conjugate pair by the quadratic formula.
// These agree (to numerical accuracy) with nimblecas.numeigen::eigenvalues_qr on the same
// matrix. Fails with domain_error when s.t.size() != s.n·s.n.
[[nodiscard]] auto schur_eigenvalues(const NumericSchur& s, double tol = 1e-9)
    -> Result<std::vector<std::complex<double>>>;

// ---------------------------------------------------------------------------
// Residual / verification helpers (numeric).
// ---------------------------------------------------------------------------

// Frobenius residual ‖Q·R − A‖_F of a NumericQr against the original row-major A (length
// rows·cols). domain_error on any size mismatch.
[[nodiscard]] auto qr_residual(const NumericQr& d, std::span<const double> a) -> Result<double>;

// Frobenius residual ‖Q·T·Qᵀ − A‖_F of a NumericSchur against the original row-major A
// (length n·n). domain_error on any size mismatch.
[[nodiscard]] auto schur_residual(const NumericSchur& s, std::span<const double> a)
    -> Result<double>;

// Orthonormality defect ‖Qᵀ·Q − I‖_F of a row-major rows×cols matrix Q (0 iff the columns
// are exactly orthonormal). domain_error when q.size() != rows·cols.
[[nodiscard]] auto orthonormality_defect(std::span<const double> q, std::size_t rows,
                                         std::size_t cols) -> Result<double>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================

// Unwrap a Result or propagate its error as make_error<RetType>. Each enclosing function
// defines `using RetType = ...;` before its first use. Variadic so the wrapped expression
// may itself contain commas.
#define TRY(var, ...)                                                \
    auto var##__r = (__VA_ARGS__);                                   \
    if (!(var##__r)) return make_error<RetType>((var##__r).error()); \
    auto var = std::move(*var##__r)

namespace nimblecas {
namespace {

// Guard a rows*cols allocation size against std::size_t wrap.
[[nodiscard]] auto area_fits(std::size_t rows, std::size_t cols) -> bool {
    return rows == 0 || cols <= std::numeric_limits<std::size_t>::max() / rows;
}

// --- (3) Schur internals: orthogonal Hessenberg + Francis double-shift QR ---
//
// Both routines operate on 0-indexed row-major buffers H (N×N) and V (N×N) with N == n.
// They are a faithful transcription of the classic EISPACK orthes/ortran + hqr2 algorithm
// (as popularised by JAMA), specialised to stop at the real Schur form: H becomes the
// quasi-triangular T and V accumulates the orthogonal Schur vectors, with A = V·H·Vᵀ.

// Householder reduction of H to upper Hessenberg (orthes), accumulating the orthogonal
// similarity into V (ortran). On entry V is ignored; on exit V is orthogonal and
// A0 = V·H·Vᵀ where A0 was H's entry value. A no-op transform for N < 3.
void hessenberg_orthes(std::vector<double>& H, std::vector<double>& V, std::ptrdiff_t N) {
    const std::ptrdiff_t low = 0;
    const std::ptrdiff_t high = N - 1;
    std::vector<double> ort(static_cast<std::size_t>(N), 0.0);
    auto Hx = [&](std::ptrdiff_t i, std::ptrdiff_t j) -> double& {
        return H[static_cast<std::size_t>(i * N + j)];
    };
    auto Vx = [&](std::ptrdiff_t i, std::ptrdiff_t j) -> double& {
        return V[static_cast<std::size_t>(i * N + j)];
    };
    auto O = [&](std::ptrdiff_t i) -> double& { return ort[static_cast<std::size_t>(i)]; };

    for (std::ptrdiff_t m = low + 1; m <= high - 1; ++m) {
        // Scale the column below the diagonal to avoid under/overflow.
        double scale = 0.0;
        for (std::ptrdiff_t i = m; i <= high; ++i) {
            scale += std::fabs(Hx(i, m - 1));
        }
        if (scale != 0.0) {
            // Compute the Householder vector into ort[m..high].
            double h = 0.0;
            for (std::ptrdiff_t i = high; i >= m; --i) {
                O(i) = Hx(i, m - 1) / scale;
                h += O(i) * O(i);
            }
            double g = std::sqrt(h);
            if (O(m) > 0.0) {
                g = -g;
            }
            h = h - O(m) * g;
            O(m) = O(m) - g;
            // Apply H := (I - u·uᵀ/h)·H (rows m..high).
            for (std::ptrdiff_t j = m; j < N; ++j) {
                double f = 0.0;
                for (std::ptrdiff_t i = high; i >= m; --i) {
                    f += O(i) * Hx(i, j);
                }
                f = f / h;
                for (std::ptrdiff_t i = m; i <= high; ++i) {
                    Hx(i, j) -= f * O(i);
                }
            }
            // Apply H := H·(I - u·uᵀ/h) (columns m..high).
            for (std::ptrdiff_t i = 0; i <= high; ++i) {
                double f = 0.0;
                for (std::ptrdiff_t j = high; j >= m; --j) {
                    f += O(j) * Hx(i, j);
                }
                f = f / h;
                for (std::ptrdiff_t j = m; j <= high; ++j) {
                    Hx(i, j) -= f * O(j);
                }
            }
            O(m) = scale * O(m);
            Hx(m, m - 1) = scale * g;
        }
    }

    // V := I, then accumulate the Householder reflections back into V (ortran). ort[m]
    // persists from the reduction (each column-step m writes only ort[m..high], and steps
    // run in increasing m, so ort[m] is last written by step m itself); the tail
    // ort[m+1..high] is re-read from H's untouched subdiagonal column.
    for (std::ptrdiff_t i = 0; i < N; ++i) {
        for (std::ptrdiff_t j = 0; j < N; ++j) {
            Vx(i, j) = (i == j) ? 1.0 : 0.0;
        }
    }
    for (std::ptrdiff_t m = high - 1; m >= low + 1; --m) {
        if (Hx(m, m - 1) != 0.0) {
            for (std::ptrdiff_t i = m + 1; i <= high; ++i) {
                O(i) = Hx(i, m - 1);
            }
            for (std::ptrdiff_t j = m; j <= high; ++j) {
                double gg = 0.0;
                for (std::ptrdiff_t i = m; i <= high; ++i) {
                    gg += O(i) * Vx(i, j);
                }
                gg = (gg / O(m)) / Hx(m, m - 1);  // double division guards underflow
                for (std::ptrdiff_t i = m; i <= high; ++i) {
                    Vx(i, j) += gg * O(i);
                }
            }
        }
    }
}

// Francis double-shift QR on the upper-Hessenberg H, accumulating the orthogonal
// transformations into V, until H is the quasi-triangular real Schur form. Returns
// std::nullopt on success, or MathError::not_implemented if a deflated block fails to
// converge within `max_iter` sweeps (honest non-convergence). `tol` is the relative
// deflation threshold. Transcribed from EISPACK hqr2, stopped at the Schur form (the
// eigenvector back-substitution of hqr2 is deliberately NOT performed).
[[nodiscard]] auto schur_qr(std::vector<double>& H, std::vector<double>& V, std::ptrdiff_t N,
                            double tol, std::size_t max_iter) -> std::optional<MathError> {
    auto Hx = [&](std::ptrdiff_t i, std::ptrdiff_t j) -> double& {
        return H[static_cast<std::size_t>(i * N + j)];
    };
    auto Vx = [&](std::ptrdiff_t i, std::ptrdiff_t j) -> double& {
        return V[static_cast<std::size_t>(i * N + j)];
    };
    const std::ptrdiff_t low = 0;
    const std::ptrdiff_t high = N - 1;

    // Matrix (block) norm used for the deflation cutoff on a wholly-zero shoulder.
    double norm = 0.0;
    for (std::ptrdiff_t i = 0; i < N; ++i) {
        for (std::ptrdiff_t j = std::max<std::ptrdiff_t>(i - 1, 0); j < N; ++j) {
            norm += std::fabs(Hx(i, j));
        }
    }

    std::ptrdiff_t en = high;  // trailing eigenvalue index currently being isolated
    double exshift = 0.0;      // accumulated shift folded back into the diagonal
    double p = 0.0, q = 0.0, r = 0.0, s = 0.0, z = 0.0, w = 0.0, x = 0.0, y = 0.0;
    std::size_t iter = 0;

    while (en >= low) {
        // Search for a negligible subdiagonal entry to deflate at (row l).
        std::ptrdiff_t l = en;
        while (l > low) {
            s = std::fabs(Hx(l - 1, l - 1)) + std::fabs(Hx(l, l));
            if (s == 0.0) {
                s = norm;
            }
            if (std::fabs(Hx(l, l - 1)) < tol * s) {
                break;
            }
            --l;
        }

        if (l == en) {
            // One real eigenvalue (1×1 block) split off.
            Hx(en, en) += exshift;
            --en;
            iter = 0;
        } else if (l == en - 1) {
            // Trailing 2×2 block.
            w = Hx(en, en - 1) * Hx(en - 1, en);
            p = 0.5 * (Hx(en - 1, en - 1) - Hx(en, en));
            q = p * p + w;
            z = std::sqrt(std::fabs(q));
            Hx(en, en) += exshift;
            Hx(en - 1, en - 1) += exshift;
            x = Hx(en, en);
            if (q >= 0.0) {
                // Real pair: rotate the block to upper-triangular form (two 1×1 blocks).
                z = (p >= 0.0) ? p + z : p - z;
                x = Hx(en, en - 1);
                s = std::fabs(x) + std::fabs(z);
                p = x / s;
                q = z / s;
                r = std::sqrt(p * p + q * q);
                p = p / r;
                q = q / r;
                for (std::ptrdiff_t j = en - 1; j < N; ++j) {  // row modification
                    z = Hx(en - 1, j);
                    Hx(en - 1, j) = q * z + p * Hx(en, j);
                    Hx(en, j) = q * Hx(en, j) - p * z;
                }
                for (std::ptrdiff_t i = 0; i <= en; ++i) {  // column modification
                    z = Hx(i, en - 1);
                    Hx(i, en - 1) = q * z + p * Hx(i, en);
                    Hx(i, en) = q * Hx(i, en) - p * z;
                }
                for (std::ptrdiff_t i = low; i <= high; ++i) {  // accumulate into V
                    z = Vx(i, en - 1);
                    Vx(i, en - 1) = q * z + p * Vx(i, en);
                    Vx(i, en) = q * Vx(i, en) - p * z;
                }
            }
            // Complex pair (q < 0): the 2×2 block stays; its subdiagonal is preserved.
            en -= 2;
            iter = 0;
        } else {
            // No deflation yet: perform a Francis double-shift QR sweep.
            if (iter >= max_iter) {
                return MathError::not_implemented;  // honest non-convergence signal
            }
            x = Hx(en, en);
            y = 0.0;
            w = 0.0;
            if (l < en) {
                y = Hx(en - 1, en - 1);
                w = Hx(en, en - 1) * Hx(en - 1, en);
            }
            if (iter == 10) {  // Wilkinson exceptional shift
                exshift += x;
                for (std::ptrdiff_t i = low; i <= en; ++i) {
                    Hx(i, i) -= x;
                }
                s = std::fabs(Hx(en, en - 1)) + std::fabs(Hx(en - 1, en - 2));
                x = y = 0.75 * s;
                w = -0.4375 * s * s;
            }
            if (iter == 30) {  // MATLAB exceptional shift
                s = 0.5 * (y - x);
                s = s * s + w;
                if (s > 0.0) {
                    s = std::sqrt(s);
                    if (y < x) {
                        s = -s;
                    }
                    s = x - w / (0.5 * (y - x) + s);
                    for (std::ptrdiff_t i = low; i <= en; ++i) {
                        Hx(i, i) -= s;
                    }
                    exshift += s;
                    x = y = w = 0.964;
                }
            }
            ++iter;

            // Locate the start row m of the bulge chase (two consecutive small subdiagonals).
            std::ptrdiff_t m = en - 2;
            while (m >= l) {
                z = Hx(m, m);
                r = x - z;
                s = y - z;
                p = (r * s - w) / Hx(m + 1, m) + Hx(m, m + 1);
                q = Hx(m + 1, m + 1) - z - r - s;
                r = Hx(m + 2, m + 1);
                s = std::fabs(p) + std::fabs(q) + std::fabs(r);
                p /= s;
                q /= s;
                r /= s;
                if (m == l) {
                    break;
                }
                if (std::fabs(Hx(m, m - 1)) * (std::fabs(q) + std::fabs(r)) <
                    tol * (std::fabs(p) * (std::fabs(Hx(m - 1, m - 1)) + std::fabs(z) +
                                           std::fabs(Hx(m + 1, m + 1))))) {
                    break;
                }
                --m;
            }
            for (std::ptrdiff_t i = m + 2; i <= en; ++i) {
                Hx(i, i - 2) = 0.0;
                if (i > m + 2) {
                    Hx(i, i - 3) = 0.0;
                }
            }

            // Chase the bulge down the subdiagonal with Householder reflections.
            for (std::ptrdiff_t k = m; k <= en - 1; ++k) {
                const bool notlast = (k != en - 1);
                if (k != m) {
                    p = Hx(k, k - 1);
                    q = Hx(k + 1, k - 1);
                    r = notlast ? Hx(k + 2, k - 1) : 0.0;
                    x = std::fabs(p) + std::fabs(q) + std::fabs(r);
                    if (x != 0.0) {
                        p /= x;
                        q /= x;
                        r /= x;
                    }
                }
                if (x == 0.0) {
                    break;
                }
                s = std::sqrt(p * p + q * q + r * r);
                if (p < 0.0) {
                    s = -s;
                }
                if (s != 0.0) {
                    if (k != m) {
                        Hx(k, k - 1) = -s * x;
                    } else if (l != m) {
                        Hx(k, k - 1) = -Hx(k, k - 1);
                    }
                    p += s;
                    x = p / s;
                    y = q / s;
                    z = r / s;
                    q /= p;
                    r /= p;
                    for (std::ptrdiff_t j = k; j < N; ++j) {  // row modification
                        p = Hx(k, j) + q * Hx(k + 1, j);
                        if (notlast) {
                            p += r * Hx(k + 2, j);
                            Hx(k + 2, j) -= p * z;
                        }
                        Hx(k, j) -= p * x;
                        Hx(k + 1, j) -= p * y;
                    }
                    const std::ptrdiff_t ihi = std::min(en, k + 3);
                    for (std::ptrdiff_t i = 0; i <= ihi; ++i) {  // column modification
                        p = x * Hx(i, k) + y * Hx(i, k + 1);
                        if (notlast) {
                            p += z * Hx(i, k + 2);
                            Hx(i, k + 2) -= p * r;
                        }
                        Hx(i, k) -= p;
                        Hx(i, k + 1) -= p * q;
                    }
                    for (std::ptrdiff_t i = low; i <= high; ++i) {  // accumulate into V
                        p = x * Vx(i, k) + y * Vx(i, k + 1);
                        if (notlast) {
                            p += z * Vx(i, k + 2);
                            Vx(i, k + 2) -= p * r;
                        }
                        Vx(i, k) -= p;
                        Vx(i, k + 1) -= p * q;
                    }
                }
            }
        }
    }
    return std::nullopt;
}

}  // namespace

// --- (1) exact orthogonal decomposition over Q -----------------------------

auto exact_orthogonal_qr(const Matrix& a) -> Result<ExactOrthogonalQr> {
    using RetType = ExactOrthogonalQr;
    const std::size_t m = a.rows();
    const std::size_t n = a.cols();

    // Q columns (each length m) and their exact squared pseudo-norms ⟨qₖ,qₖ⟩.
    std::vector<std::vector<Rational>> qcol(n, std::vector<Rational>(m));
    std::vector<Rational> qn(n);
    // R (n×n) upper-triangular with unit diagonal; default Rational is 0/1.
    std::vector<std::vector<Rational>> rrows(n, std::vector<Rational>(n));
    const Rational one = Rational::from_int(1);

    for (std::size_t k = 0; k < n; ++k) {
        // Start qₖ as column k of A.
        for (std::size_t i = 0; i < m; ++i) {
            qcol[k][i] = a.at(i, k);
        }
        // Subtract the projections onto the earlier orthogonal columns.
        for (std::size_t j = 0; j < k; ++j) {
            Rational num;  // ⟨aₖ, qⱼ⟩, starts 0
            for (std::size_t i = 0; i < m; ++i) {
                TRY(prod, a.at(i, k).multiply(qcol[j][i]));
                TRY(sum, num.add(prod));
                num = sum;
            }
            TRY(rjk, num.divide(qn[j]));  // qn[j] != 0 (verified when column j was formed)
            rrows[j][k] = rjk;
            for (std::size_t i = 0; i < m; ++i) {
                TRY(proj, rjk.multiply(qcol[j][i]));
                TRY(diff, qcol[k][i].subtract(proj));
                qcol[k][i] = diff;
            }
        }
        rrows[k][k] = one;
        // ⟨qₖ, qₖ⟩ — a zero here means column k is dependent on the earlier ones.
        Rational nn;  // 0
        for (std::size_t i = 0; i < m; ++i) {
            TRY(sq, qcol[k][i].multiply(qcol[k][i]));
            TRY(sum, nn.add(sq));
            nn = sum;
        }
        if (nn.is_zero()) {
            return make_error<RetType>(MathError::domain_error);  // rank-deficient
        }
        qn[k] = nn;
    }

    // Assemble Q (m×n, column k is qcol[k]) and R (n×n).
    std::vector<std::vector<Rational>> qmat(m, std::vector<Rational>(n));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            qmat[i][k] = qcol[k][i];
        }
    }
    TRY(qout, Matrix::from_rows(std::move(qmat)));
    TRY(rout, Matrix::from_rows(std::move(rrows)));
    return ExactOrthogonalQr{std::move(qout), std::move(rout)};
}

// --- (2) numeric Householder QR --------------------------------------------

auto numeric_qr(std::span<const double> a, std::size_t m, std::size_t n) -> Result<NumericQr> {
    if (!area_fits(m, n)) {
        return make_error<NumericQr>(MathError::overflow);
    }
    if (a.size() != m * n) {
        return make_error<NumericQr>(MathError::domain_error);
    }
    if (!area_fits(m, m)) {
        return make_error<NumericQr>(MathError::overflow);
    }

    std::vector<double> R(a.begin(), a.end());  // m×n, becomes upper-triangular
    std::vector<double> Q(m * m, 0.0);          // m×m, becomes orthonormal
    for (std::size_t i = 0; i < m; ++i) {
        Q[i * m + i] = 1.0;
    }
    std::vector<double> v(m, 0.0);  // Householder vector, active in rows k..m-1

    const std::size_t steps = std::min(m, n);
    for (std::size_t k = 0; k < steps; ++k) {
        if (m - k < 2) {
            break;  // only the final 1×1 pivot remains; nothing below it to eliminate
        }
        double normx = 0.0;
        for (std::size_t i = k; i < m; ++i) {
            normx += R[i * n + k] * R[i * n + k];
        }
        normx = std::sqrt(normx);
        if (normx == 0.0) {
            continue;  // column already zero at/below the pivot; R diagonal stays 0
        }
        const double x0 = R[k * n + k];
        const double alpha = (x0 > 0.0) ? -normx : normx;  // -sign(x0)·‖x‖ for stability
        v[k] = x0 - alpha;
        for (std::size_t i = k + 1; i < m; ++i) {
            v[i] = R[i * n + k];
        }
        double vn2 = 0.0;
        for (std::size_t i = k; i < m; ++i) {
            vn2 += v[i] * v[i];
        }
        if (vn2 == 0.0) {
            continue;
        }
        const double beta = 2.0 / vn2;
        // R := (I - beta·v·vᵀ)·R (rows k..m-1).
        for (std::size_t j = 0; j < n; ++j) {
            double d = 0.0;
            for (std::size_t i = k; i < m; ++i) {
                d += v[i] * R[i * n + j];
            }
            d *= beta;
            for (std::size_t i = k; i < m; ++i) {
                R[i * n + j] -= v[i] * d;
            }
        }
        // Q := Q·(I - beta·v·vᵀ) (columns k..m-1).
        for (std::size_t i = 0; i < m; ++i) {
            double d = 0.0;
            for (std::size_t c = k; c < m; ++c) {
                d += Q[i * m + c] * v[c];
            }
            d *= beta;
            for (std::size_t c = k; c < m; ++c) {
                Q[i * m + c] -= d * v[c];
            }
        }
    }

    // Force the strictly-lower triangle of R to exact zero (the reflections leave ~1e-16
    // there); this makes R exactly upper-triangular at a residual cost far below tol.
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n && j < i; ++j) {
            R[i * n + j] = 0.0;
        }
    }
    return NumericQr{m, n, std::move(Q), std::move(R)};
}

// --- (3) numeric real Schur form -------------------------------------------

auto real_schur(std::span<const double> a, std::size_t n, double tol, std::size_t max_iter)
    -> Result<NumericSchur> {
    if (!area_fits(n, n)) {
        return make_error<NumericSchur>(MathError::overflow);
    }
    if (a.size() != n * n) {
        return make_error<NumericSchur>(MathError::domain_error);
    }
    if (n == 0) {
        return NumericSchur{0, {}, {}};
    }
    std::vector<double> H(a.begin(), a.end());
    std::vector<double> V(n * n, 0.0);
    const auto N = static_cast<std::ptrdiff_t>(n);

    hessenberg_orthes(H, V, N);
    if (auto err = schur_qr(H, V, N, tol, max_iter); err) {
        return make_error<NumericSchur>(*err);
    }
    return NumericSchur{n, std::move(V), std::move(H)};
}

auto schur_eigenvalues(const NumericSchur& s, double tol)
    -> Result<std::vector<std::complex<double>>> {
    using cd = std::complex<double>;
    const std::size_t n = s.n;
    if (s.t.size() != n * n) {
        return make_error<std::vector<cd>>(MathError::domain_error);
    }
    const auto& T = s.t;
    std::vector<cd> out;
    out.reserve(n);
    std::size_t i = 0;
    while (i < n) {
        bool two = false;
        if (i + 1 < n) {
            const double sub = std::fabs(T[(i + 1) * n + i]);
            const double diag = std::fabs(T[i * n + i]) + std::fabs(T[(i + 1) * n + (i + 1)]);
            const double ref = (diag == 0.0) ? 1.0 : diag;
            two = sub > tol * ref;  // a live subdiagonal => genuine 2×2 block
        }
        if (two) {
            const double aii = T[i * n + i];
            const double ajj = T[(i + 1) * n + (i + 1)];
            const double mean = 0.5 * (aii + ajj);
            const double half = 0.5 * (aii - ajj);
            const double bc = T[i * n + (i + 1)] * T[(i + 1) * n + i];
            const double disc = half * half + bc;
            if (disc >= 0.0) {
                const double d = std::sqrt(disc);
                out.emplace_back(mean + d, 0.0);
                out.emplace_back(mean - d, 0.0);
            } else {
                const double d = std::sqrt(-disc);
                out.emplace_back(mean, d);
                out.emplace_back(mean, -d);
            }
            i += 2;
        } else {
            out.emplace_back(T[i * n + i], 0.0);
            i += 1;
        }
    }
    return out;
}

// --- residual / verification helpers ---------------------------------------

auto qr_residual(const NumericQr& d, std::span<const double> a) -> Result<double> {
    const std::size_t m = d.rows;
    const std::size_t n = d.cols;
    if (!area_fits(m, n) || !area_fits(m, m) || a.size() != m * n || d.q.size() != m * m ||
        d.r.size() != m * n) {
        return make_error<double>(MathError::domain_error);
    }
    double acc = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < m; ++k) {
                sum += d.q[i * m + k] * d.r[k * n + j];
            }
            const double e = sum - a[i * n + j];
            acc += e * e;
        }
    }
    return std::sqrt(acc);
}

auto schur_residual(const NumericSchur& s, std::span<const double> a) -> Result<double> {
    const std::size_t n = s.n;
    if (!area_fits(n, n) || a.size() != n * n || s.q.size() != n * n || s.t.size() != n * n) {
        return make_error<double>(MathError::domain_error);
    }
    // M = Q·T, then residual of M·Qᵀ against A.
    std::vector<double> M(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                sum += s.q[i * n + k] * s.t[k * n + j];
            }
            M[i * n + j] = sum;
        }
    }
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                sum += M[i * n + k] * s.q[j * n + k];  // (Qᵀ)_{k,j} = Q_{j,k}
            }
            const double e = sum - a[i * n + j];
            acc += e * e;
        }
    }
    return std::sqrt(acc);
}

auto orthonormality_defect(std::span<const double> q, std::size_t rows, std::size_t cols)
    -> Result<double> {
    if (!area_fits(rows, cols) || q.size() != rows * cols) {
        return make_error<double>(MathError::domain_error);
    }
    double acc = 0.0;
    for (std::size_t p = 0; p < cols; ++p) {
        for (std::size_t r = 0; r < cols; ++r) {
            double sum = 0.0;
            for (std::size_t i = 0; i < rows; ++i) {
                sum += q[i * cols + p] * q[i * cols + r];
            }
            const double target = (p == r) ? 1.0 : 0.0;
            const double e = sum - target;
            acc += e * e;
        }
    }
    return std::sqrt(acc);
}

}  // namespace nimblecas
