// NimbleCAS numeric eigenvalues of a real matrix: all eigenvalues (real and complex)
// via the shifted QR algorithm reduced to real Schur form (ROADMAP 7.21 numeric path).
// @author Olumuyiwa Oluwasanmi
//
// The engine already answers two exact spectral questions: nimblecas.eigen returns the
// RATIONAL slice of the spectrum of an exact Matrix, and analysis::dominant_eigenvalue
// returns the single dominant eigenvalue magnitude of a symmetric double matrix. Neither
// returns ALL eigenvalues of a general real matrix, including the complex-conjugate pairs
// of a non-symmetric one. This module fills that gap NUMERICALLY.
//
// HONESTY BOUNDARY (Rule 32). Unlike the rest of nimblecas.eigen, the results here are
// NOT exact: they are double-precision approximations accurate to roughly `tol`, obtained
// by an iterative floating-point algorithm. A defective or severely ill-conditioned
// matrix can lose accuracy in the returned eigenvalues; a genuinely non-convergent block
// is reported as an honest error rather than returned as a plausible-but-wrong value.
// Because the core MathError enum has no dedicated "no convergence" variant, a block that
// fails to reach the deflation tolerance within `max_iter` QR sweeps returns
// MathError::not_implemented (the closest honest variant available) — it is the signal
// "this input was not solved", never a silently truncated or fabricated spectrum.
//
// ALGORITHM. The matrix is first reduced to upper Hessenberg form by Gaussian elimination
// with pivoting (EISPACK elmhes), then driven to real Schur form by the Francis
// double-shift QR iteration with deflation (EISPACK hqr): small subdiagonal entries are
// zeroed when |a_{l,l-1}| <= tol*(|a_{l-1,l-1}| + |a_{l,l}|), a trailing 1x1 block yields
// a real eigenvalue, and a trailing 2x2 block yields its two eigenvalues by the quadratic
// formula — a real pair when the discriminant is non-negative, else a complex-conjugate
// pair. Wilkinson-style exceptional shifts are injected periodically to break cycles.

export module nimblecas.numeigen;

import std;
import nimblecas.core;
import nimblecas.ratpoly;

export namespace nimblecas {

// All eigenvalues of the n x n real matrix given row-major in `a` (length n*n), computed
// numerically by the QR algorithm. Returned as std::complex<double>; real eigenvalues have
// (near-)zero imaginary part. `tol` sets the relative subdiagonal deflation threshold and
// `max_iter` caps the QR sweeps allowed per deflated block. A 0x0 matrix yields an empty
// vector. Fails with domain_error when a.size() != n*n, overflow when the index arithmetic
// for n would wrap, and not_implemented when a block does not converge within max_iter.
[[nodiscard]] auto eigenvalues_qr(std::span<const double> a, std::size_t n,
                                  double tol = 1e-12, std::size_t max_iter = 1000)
    -> Result<std::vector<std::complex<double>>>;

// Convenience: all roots of a polynomial as the eigenvalues of its companion matrix.
// Builds the (real, double) companion matrix of the monic-normalized `p` and calls
// eigenvalues_qr. This is the numeric polynomial root finder (ROADMAP 7.21 numeric path).
// The zero polynomial is a domain_error (undefined root set); a non-zero constant has no
// roots and returns an empty vector.
[[nodiscard]] auto companion_eigenvalues(const RationalPoly& p,
                                         double tol = 1e-12, std::size_t max_iter = 1000)
    -> Result<std::vector<std::complex<double>>>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// Structural class of a real square matrix, used to dispatch to the fastest correct
// eigen-algorithm. For a REAL matrix, "symmetric" is the Hermitian case and
// "skew_symmetric" the skew-Hermitian case; genuinely complex-Hermitian input (a complex
// matrix) is a future extension, not handled here.
enum class MatrixKind { diagonal, triangular, symmetric, skew_symmetric, general };

// Classify `a` (n x n, row-major) up to a relative tolerance. Detection is order-sensitive:
// diagonal is a special triangular, and both are checked before symmetry, so a diagonal
// matrix never reports "symmetric". An entry is treated as zero when its magnitude is
// <= tol * (largest magnitude entry). When no structure is detected the honest fallback is
// `general`, which is always correct (merely slower) — a mis-detection can only cost speed,
// never correctness, and the symmetric/skew tests below never fire on a non-symmetric input.
[[nodiscard]] auto classify(std::span<const double> a, std::size_t n, double tol) -> MatrixKind {
    double scale = 0.0;
    for (double v : a) {
        scale = std::max(scale, std::fabs(v));
    }
    if (scale == 0.0) {
        return MatrixKind::diagonal;  // the zero matrix
    }
    const double eps = tol * scale;
    bool lower_zero = true;  // all entries strictly below the diagonal are ~0 (upper tri)
    bool upper_zero = true;  // all entries strictly above the diagonal are ~0 (lower tri)
    bool sym = true;         // a_ij ~ a_ji
    bool skew = true;        // a_ij ~ -a_ji and diagonal ~0
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const double aij = a[i * n + j];
            if (i < j && std::fabs(aij) > eps) {
                upper_zero = false;
            }
            if (i > j && std::fabs(aij) > eps) {
                lower_zero = false;
            }
            if (i != j) {
                const double aji = a[j * n + i];
                if (std::fabs(aij - aji) > eps) {
                    sym = false;
                }
                if (std::fabs(aij + aji) > eps) {
                    skew = false;
                }
            } else if (std::fabs(aij) > eps) {
                skew = false;  // a skew-symmetric matrix has a zero diagonal
            }
        }
    }
    if (upper_zero && lower_zero) {
        return MatrixKind::diagonal;
    }
    if (upper_zero || lower_zero) {
        return MatrixKind::triangular;
    }
    if (sym) {
        return MatrixKind::symmetric;
    }
    if (skew) {
        return MatrixKind::skew_symmetric;
    }
    return MatrixKind::general;
}

// Reduce the 1-indexed N x N matrix stored in H (row stride S = N+1, entries H[i*S+j] for
// i,j in 1..N) to upper Hessenberg form in place by Gaussian elimination with partial
// pivoting (EISPACK elmhes). A no-op for N < 3.
void elmhes(std::vector<double>& H, std::ptrdiff_t N, std::ptrdiff_t S) {
    auto A = [&](std::ptrdiff_t i, std::ptrdiff_t j) -> double& { return H[i * S + j]; };
    for (std::ptrdiff_t m = 2; m < N; ++m) {
        double x = 0.0;
        std::ptrdiff_t piv = m;
        for (std::ptrdiff_t j = m; j <= N; ++j) {
            if (std::fabs(A(j, m - 1)) > std::fabs(x)) {
                x = A(j, m - 1);
                piv = j;
            }
        }
        if (piv != m) {  // interchange rows and columns to bring the pivot into place
            for (std::ptrdiff_t j = m - 1; j <= N; ++j) {
                std::swap(A(piv, j), A(m, j));
            }
            for (std::ptrdiff_t j = 1; j <= N; ++j) {
                std::swap(A(j, piv), A(j, m));
            }
        }
        if (x != 0.0) {
            for (std::ptrdiff_t i = m + 1; i <= N; ++i) {
                double y = A(i, m - 1);
                if (y != 0.0) {
                    y /= x;
                    A(i, m - 1) = y;
                    for (std::ptrdiff_t j = m; j <= N; ++j) {
                        A(i, j) -= y * A(m, j);
                    }
                    for (std::ptrdiff_t j = 1; j <= N; ++j) {
                        A(j, m) += y * A(j, i);
                    }
                }
            }
        }
    }
}

// Francis double-shift QR on the upper-Hessenberg H (1-indexed, stride S), writing the N
// eigenvalues into wr/wi (real and imaginary parts, 1-indexed 1..N). Returns std::nullopt
// on success, or MathError::not_implemented if a block fails to reach the deflation
// tolerance within `max_iter` sweeps (an honest non-convergence signal). Ported from the
// EISPACK hqr algorithm; the deflation and shift-cutoff tests use the relative `tol`.
[[nodiscard]] auto hqr(std::vector<double>& H, std::ptrdiff_t N, std::ptrdiff_t S,
                       std::vector<double>& wr, std::vector<double>& wi, double tol,
                       std::size_t max_iter) -> std::optional<MathError> {
    auto A = [&](std::ptrdiff_t i, std::ptrdiff_t j) -> double& { return H[i * S + j]; };
    auto sgn = [](double a, double b) { return b >= 0.0 ? std::fabs(a) : -std::fabs(a); };

    double anorm = 0.0;
    for (std::ptrdiff_t i = 1; i <= N; ++i) {
        for (std::ptrdiff_t j = std::max<std::ptrdiff_t>(i - 1, 1); j <= N; ++j) {
            anorm += std::fabs(A(i, j));
        }
    }

    std::ptrdiff_t nn = N;  // active trailing submatrix is rows/cols 1..nn
    double t = 0.0;         // accumulated shift restored into the eigenvalues
    while (nn >= 1) {
        std::size_t its = 0;
        std::ptrdiff_t l = 0;
        do {
            // Search for a negligible subdiagonal entry to deflate at (row l).
            for (l = nn; l >= 2; --l) {
                double s = std::fabs(A(l - 1, l - 1)) + std::fabs(A(l, l));
                if (s == 0.0) {
                    s = anorm;
                }
                if (std::fabs(A(l, l - 1)) <= tol * s) {
                    A(l, l - 1) = 0.0;
                    break;
                }
            }
            double x = A(nn, nn);
            if (l == nn) {
                // One real eigenvalue split off (1x1 block).
                wr[nn] = x + t;
                wi[nn] = 0.0;
                --nn;
            } else {
                double y = A(nn - 1, nn - 1);
                double w = A(nn, nn - 1) * A(nn - 1, nn);
                if (l == nn - 1) {
                    // Two eigenvalues of the trailing 2x2 block by the quadratic formula.
                    double p = 0.5 * (y - x);
                    double q = p * p + w;
                    double z = std::sqrt(std::fabs(q));
                    x += t;
                    if (q >= 0.0) {  // real pair
                        z = p + sgn(z, p);
                        wr[nn - 1] = wr[nn] = x + z;
                        if (z != 0.0) {
                            wr[nn] = x - w / z;
                        }
                        wi[nn - 1] = wi[nn] = 0.0;
                    } else {  // complex-conjugate pair
                        wr[nn - 1] = wr[nn] = x + p;
                        wi[nn] = z;
                        wi[nn - 1] = -z;
                    }
                    nn -= 2;
                } else {
                    // No deflation yet: perform a Francis double-shift QR sweep.
                    if (its >= max_iter) {
                        return MathError::not_implemented;  // honest non-convergence signal
                    }
                    if (its != 0 && its % 10 == 0) {
                        // Exceptional (ad-hoc Wilkinson) shift to break a cycle.
                        t += x;
                        for (std::ptrdiff_t i = 1; i <= nn; ++i) {
                            A(i, i) -= x;
                        }
                        double s = std::fabs(A(nn, nn - 1)) + std::fabs(A(nn - 1, nn - 2));
                        y = x = 0.75 * s;
                        w = -0.4375 * s * s;
                    }
                    ++its;

                    // Locate the start row m of the bulge chase.
                    std::ptrdiff_t m = nn - 2;
                    double p = 0.0;
                    double q = 0.0;
                    double r = 0.0;
                    double z = 0.0;
                    for (m = nn - 2; m >= l; --m) {
                        z = A(m, m);
                        double rr = x - z;
                        double ss = y - z;
                        p = (rr * ss - w) / A(m + 1, m) + A(m, m + 1);
                        q = A(m + 1, m + 1) - z - rr - ss;
                        r = A(m + 2, m + 1);
                        double s = std::fabs(p) + std::fabs(q) + std::fabs(r);
                        p /= s;
                        q /= s;
                        r /= s;
                        if (m == l) {
                            break;
                        }
                        double u = std::fabs(A(m, m - 1)) * (std::fabs(q) + std::fabs(r));
                        double v = std::fabs(p) * (std::fabs(A(m - 1, m - 1)) + std::fabs(z) +
                                                   std::fabs(A(m + 1, m + 1)));
                        if (u <= tol * v) {
                            break;
                        }
                    }
                    for (std::ptrdiff_t i = m + 2; i <= nn; ++i) {
                        A(i, i - 2) = 0.0;
                        if (i != m + 2) {
                            A(i, i - 3) = 0.0;
                        }
                    }
                    // Chase the bulge down the subdiagonal with Householder reflections.
                    for (std::ptrdiff_t k = m; k <= nn - 1; ++k) {
                        if (k != m) {
                            p = A(k, k - 1);
                            q = A(k + 1, k - 1);
                            r = 0.0;
                            if (k != nn - 1) {
                                r = A(k + 2, k - 1);
                            }
                            x = std::fabs(p) + std::fabs(q) + std::fabs(r);
                            if (x != 0.0) {
                                p /= x;
                                q /= x;
                                r /= x;
                            }
                        }
                        double s = sgn(std::sqrt(p * p + q * q + r * r), p);
                        if (s != 0.0) {
                            if (k == m) {
                                if (l != m) {
                                    A(k, k - 1) = -A(k, k - 1);
                                }
                            } else {
                                A(k, k - 1) = -s * x;
                            }
                            p += s;
                            x = p / s;
                            y = q / s;
                            z = r / s;
                            q /= p;
                            r /= p;
                            for (std::ptrdiff_t j = k; j <= nn; ++j) {  // row transform
                                p = A(k, j) + q * A(k + 1, j);
                                if (k != nn - 1) {
                                    p += r * A(k + 2, j);
                                    A(k + 2, j) -= p * z;
                                }
                                A(k + 1, j) -= p * y;
                                A(k, j) -= p * x;
                            }
                            std::ptrdiff_t mmin = (nn < k + 3) ? nn : k + 3;
                            for (std::ptrdiff_t i = l; i <= mmin; ++i) {  // column transform
                                p = x * A(i, k) + y * A(i, k + 1);
                                if (k != nn - 1) {
                                    p += z * A(i, k + 2);
                                    A(i, k + 2) -= p * r;
                                }
                                A(i, k + 1) -= p * q;
                                A(i, k) -= p;
                            }
                        }
                    }
                }
            }
        } while (l < nn - 1);
    }
    return std::nullopt;
}

// Eigenvalues of a real SYMMETRIC n x n matrix (0-indexed, row-major copy `M`, taken by
// value) by the cyclic Jacobi rotation method, written into `eig`. The eigenvalues of a
// real symmetric matrix are all REAL, so no complex output is possible here. Each rotation
// annihilates one off-diagonal pair by a similarity transform that preserves the spectrum;
// off-diagonal mass decays quadratically. Returns std::nullopt on success, or
// MathError::not_implemented if it does not reach the tolerance within `max_iter` sweeps.
[[nodiscard]] auto jacobi_symmetric(std::vector<double> M, std::ptrdiff_t n,
                                    std::vector<double>& eig, double tol, std::size_t max_iter)
    -> std::optional<MathError> {
    auto A = [&](std::ptrdiff_t i, std::ptrdiff_t j) -> double& { return M[i * n + j]; };
    double scale = 0.0;
    for (double v : M) {
        scale = std::max(scale, std::fabs(v));
    }
    if (scale == 0.0) {
        scale = 1.0;
    }
    const double thresh = tol * scale;
    for (std::size_t sweep = 0; sweep < max_iter; ++sweep) {
        double offmax = 0.0;
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            for (std::ptrdiff_t j = i + 1; j < n; ++j) {
                offmax = std::max(offmax, std::fabs(A(i, j)));
            }
        }
        if (offmax <= thresh) {
            for (std::ptrdiff_t i = 0; i < n; ++i) {
                eig[static_cast<std::size_t>(i)] = A(i, i);
            }
            return std::nullopt;
        }
        for (std::ptrdiff_t p = 0; p < n - 1; ++p) {
            for (std::ptrdiff_t q = p + 1; q < n; ++q) {
                const double apq = A(p, q);
                if (apq == 0.0) {
                    continue;
                }
                const double app = A(p, p);
                const double aqq = A(q, q);
                // Rotation angle that zeros the (p,q) entry: t = tan(phi) is the smaller
                // root of t^2 + 2*theta*t - 1 = 0 with theta = (aqq - app) / (2*apq).
                const double theta = (aqq - app) / (2.0 * apq);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                                 (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                // Similarity M <- J^T M J, with new column p = c*col_p - s*col_q.
                for (std::ptrdiff_t i = 0; i < n; ++i) {
                    if (i == p || i == q) {
                        continue;
                    }
                    const double mip = A(i, p);
                    const double miq = A(i, q);
                    A(i, p) = c * mip - s * miq;
                    A(p, i) = A(i, p);
                    A(i, q) = s * mip + c * miq;
                    A(q, i) = A(i, q);
                }
                A(p, p) = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                A(q, q) = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                A(p, q) = 0.0;
                A(q, p) = 0.0;
            }
        }
    }
    return MathError::not_implemented;  // did not converge within max_iter sweeps
}

}  // namespace

auto eigenvalues_qr(std::span<const double> a, std::size_t n, double tol, std::size_t max_iter)
    -> Result<std::vector<std::complex<double>>> {
    using cd = std::complex<double>;
    if (n == 0) {
        return std::vector<cd>{};
    }
    // Guard the 1-indexed padded-buffer arithmetic: n+1 must not wrap, and (n+1)^2 must
    // stay within std::ptrdiff_t so every index i*S+j is representable.
    const std::size_t smax = std::numeric_limits<std::size_t>::max();
    const auto pmax = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
    if (n >= smax) {
        return make_error<std::vector<cd>>(MathError::overflow);
    }
    const std::size_t stride = n + 1;
    if (stride > pmax / stride) {
        return make_error<std::vector<cd>>(MathError::overflow);
    }
    if (a.size() != n * n) {  // n*n <= (n+1)^2 <= PTRDIFF_MAX, so this cannot wrap
        return make_error<std::vector<cd>>(MathError::domain_error);
    }

    const auto N = static_cast<std::ptrdiff_t>(n);
    const auto S = static_cast<std::ptrdiff_t>(stride);

    // Structure-aware dispatch: pick the fastest algorithm that is correct for this shape.
    switch (classify(a, n, tol)) {
        case MatrixKind::diagonal:
        case MatrixKind::triangular: {
            // Eigenvalues of a (block-free) triangular matrix are its diagonal entries,
            // all real. O(n).
            std::vector<cd> out;
            out.reserve(n);
            for (std::size_t i = 0; i < n; ++i) {
                out.emplace_back(a[i * n + i], 0.0);
            }
            return out;
        }
        case MatrixKind::symmetric: {
            // Symmetric => all eigenvalues real; solve by cyclic Jacobi (accurate, real).
            std::vector<double> eig(n, 0.0);
            std::vector<double> copy(a.begin(), a.end());
            if (auto err = jacobi_symmetric(std::move(copy), N, eig, tol, max_iter); err) {
                return make_error<std::vector<cd>>(*err);
            }
            std::vector<cd> out;
            out.reserve(n);
            for (double e : eig) {
                out.emplace_back(e, 0.0);
            }
            return out;
        }
        case MatrixKind::skew_symmetric:  // purely imaginary +/- pairs; the general path's
        case MatrixKind::general:         // 2x2 blocks already yield them correctly.
            break;
    }

    // General (and skew-symmetric) real matrix: reduce to Hessenberg, then real-Schur QR.
    // Copy the row-major input into a 1-indexed padded working buffer (row/col 0 unused).
    std::vector<double> H(stride * stride, 0.0);
    for (std::ptrdiff_t i = 1; i <= N; ++i) {
        for (std::ptrdiff_t j = 1; j <= N; ++j) {
            H[i * S + j] = a[static_cast<std::size_t>((i - 1) * N + (j - 1))];
        }
    }

    elmhes(H, N, S);

    std::vector<double> wr(stride, 0.0);
    std::vector<double> wi(stride, 0.0);
    if (auto err = hqr(H, N, S, wr, wi, tol, max_iter); err) {
        return make_error<std::vector<cd>>(*err);
    }

    std::vector<cd> out;
    out.reserve(n);
    for (std::ptrdiff_t i = 1; i <= N; ++i) {
        out.emplace_back(wr[static_cast<std::size_t>(i)], wi[static_cast<std::size_t>(i)]);
    }
    return out;
}

auto companion_eigenvalues(const RationalPoly& p, double tol, std::size_t max_iter)
    -> Result<std::vector<std::complex<double>>> {
    using cd = std::complex<double>;
    if (p.is_zero()) {
        return make_error<std::vector<cd>>(MathError::domain_error);
    }
    const std::int64_t deg = p.degree();
    if (deg == 0) {
        return std::vector<cd>{};  // a non-zero constant polynomial has no roots
    }
    const auto d = static_cast<std::size_t>(deg);
    if (d > std::numeric_limits<std::size_t>::max() / d) {  // guard d*d allocation
        return make_error<std::vector<cd>>(MathError::overflow);
    }

    // Monic normalization: divide through by the (non-zero) leading coefficient.
    const Rational lc = p.leading_coefficient();
    const double lead =
        static_cast<double>(lc.numerator()) / static_cast<double>(lc.denominator());
    if (lead == 0.0) {  // unreachable for a trimmed non-zero polynomial; stay honest
        return make_error<std::vector<cd>>(MathError::domain_error);
    }

    // Companion matrix of x^d + c_{d-1} x^{d-1} + ... + c_0, with c_i = coeff(i)/lead:
    // ones on the subdiagonal and the negated coefficients in the last column. Its
    // characteristic polynomial is exactly the monic-normalized p.
    std::vector<double> comp(d * d, 0.0);
    for (std::size_t i = 0; i < d; ++i) {
        const Rational ci = p.coefficient(i);
        const double cid =
            (static_cast<double>(ci.numerator()) / static_cast<double>(ci.denominator())) / lead;
        comp[i * d + (d - 1)] = -cid;
    }
    for (std::size_t i = 1; i < d; ++i) {
        comp[i * d + (i - 1)] = 1.0;
    }

    return eigenvalues_qr(comp, d, tol, max_iter);
}

}  // namespace nimblecas
