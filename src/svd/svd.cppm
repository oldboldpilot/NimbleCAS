// NimbleCAS numeric singular value decomposition (one-sided Jacobi) and polar
// decomposition, plus the exact A^T*A slice of the singular spectrum (ROADMAP 7.2x).
// @author Olumuyiwa Oluwasanmi
//
// HONESTY BOUNDARY (Rule 32).
//
//   The thin SVD (svd) and the polar decomposition (polar) below are NUMERIC (double
//   precision). A real m x n matrix's singular values are the square roots of the
//   eigenvalues of A^T*A, and those eigenvalues are in general irrational, so no EXACT
//   SVD over Q exists for almost every input -- faking one would violate the honesty
//   invariant. The numeric path is computed by ONE-SIDED JACOBI rotation directly on A
//   (never by forming A^T*A and eigendecomposing it, which would square A's condition
//   number and roughly halve the attainable accuracy). Results are accurate to ~tol
//   relative and independently verifiable: svd_residual / polar_residual recompute the
//   Frobenius reconstruction error, and nimblecas.qrschur::orthonormality_defect (reused
//   by the tests, not by this module) recomputes ||U^T*U - I||_F. Exhausting max_sweeps
//   without a clean sweep returns MathError::not_converged and NOTHING else -- never a
//   partial or silently-inaccurate factor.
//
//   The exact companion API stays honestly inside Q: gram_matrix computes A^T*A exactly
//   (overflow-checked Rational, via nimblecas.matrix), and exact_singular_value_squares
//   returns the RATIONAL eigenvalues of A^T*A -- i.e. the rational sigma^2 values, with
//   multiplicity, descending -- via nimblecas.eigen::rational_eigenvalues. This is a
//   SLICE of the spectrum: irrational sigma^2 are honestly ABSENT from the result, never
//   approximated and dressed up as exact. No function in this module ever returns an
//   exact irrational singular value.

export module nimblecas.svd;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.eigen;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// Numeric thin SVD: A = U * diag(sigma) * V^T, k = min(rows, cols).
// ---------------------------------------------------------------------------
// `u` is rows x k row-major with orthonormal columns, `v` is cols x k row-major with
// orthonormal columns, and `singular_values` (length k) is sorted DESCENDING and
// non-negative. NUMERIC: accurate to ~tol relative, never exact.
struct NumericSvd {
    std::size_t rows{};
    std::size_t cols{};
    std::vector<double> u{};
    std::vector<double> singular_values{};
    std::vector<double> v{};
};

// Thin numeric SVD of the rows x cols matrix given row-major in `a` (length rows*cols),
// by one-sided Jacobi rotation. `tol` is the relative off-diagonal convergence threshold
// for the Gram-matrix inner products; `max_sweeps` caps the number of cyclic sweeps.
//
// Deterministic: a fixed ascending pair order (p < q) each sweep, a stable descending
// sort of the singular values (with U/V columns carried along under the same
// permutation), and a fixed sign convention (see below) eliminate the usual sign/order
// nondeterminism of Jacobi-family SVD. A singular value at or below a relative cutoff of
// the largest one is treated as (numerically) zero; its U column is still completed to a
// unit vector orthogonal to every other accepted U column by deterministic Gram-Schmidt
// against the standard basis e0, e1, ... in index order (never left as 0, never NaN).
// Every column's sign is then fixed by requiring its largest-magnitude entry (lowest
// index on ties) to be non-negative, negating that U column AND the paired V column
// together when it is not.
//
// Fails with domain_error when a.size() != rows*cols, overflow when the rows*cols /
// rows*k / cols*k index arithmetic would wrap, and not_converged when max_sweeps sweeps
// all still find a rotation to apply (an honest "not solved" signal -- nothing partial is
// returned). A 0x0 (or any rows*cols == 0) input yields an empty (k = 0) decomposition.
[[nodiscard]] auto svd(std::span<const double> a, std::size_t rows, std::size_t cols,
                       double tol = 1e-12, std::size_t max_sweeps = 60) -> Result<NumericSvd>;

// Numeric SVD of a Matrix, converting every entry to double first. LOSSY: this leaves
// the exact rational regime entirely -- use gram_matrix / exact_singular_value_squares
// below to stay exact. domain_error / overflow propagate the same as the span overload.
[[nodiscard]] auto svd(const Matrix& a, double tol = 1e-12, std::size_t max_sweeps = 60)
    -> Result<NumericSvd>;

// ---------------------------------------------------------------------------
// Numeric polar decomposition (square only): A = U * P.
// ---------------------------------------------------------------------------
// `u` (n x n) is orthogonal and `p` (n x n) is symmetric positive semidefinite (P^2 =
// A^T*A), both row-major. NUMERIC.
struct NumericPolar {
    std::size_t n{};
    std::vector<double> u{};
    std::vector<double> p{};
};

// Polar decomposition A = U*P of the SQUARE n x n matrix given row-major in `a` (length
// n*n), built from the thin SVD (U*V^T, V*diag(sigma)*V^T symmetrised). Fails with
// domain_error when a.size() != n*n (including any attempt to feed a non-square shape
// through this square-only entry point), propagates svd's overflow / not_converged.
// A 0x0 input yields an empty decomposition.
[[nodiscard]] auto polar(std::span<const double> a, std::size_t n, double tol = 1e-12,
                         std::size_t max_sweeps = 60) -> Result<NumericPolar>;

// ---------------------------------------------------------------------------
// Residual / verification helpers (numeric).
// ---------------------------------------------------------------------------

// Frobenius residual ||U*diag(sigma)*V^T - A||_F of a NumericSvd against the original
// row-major A (length d.rows*d.cols). domain_error on any size mismatch, overflow when
// the index arithmetic would wrap.
[[nodiscard]] auto svd_residual(const NumericSvd& d, std::span<const double> a)
    -> Result<double>;

// Frobenius residual ||U*P - A||_F of a NumericPolar against the original row-major A
// (length d.n*d.n). domain_error on any size mismatch.
[[nodiscard]] auto polar_residual(const NumericPolar& d, std::span<const double> a)
    -> Result<double>;

// ---------------------------------------------------------------------------
// Exact companions over Q.
// ---------------------------------------------------------------------------

// The exact Gram matrix A^T*A (overflow-checked Rational, via Matrix::transpose /
// Matrix::multiply). domain_error / overflow propagate from those.
[[nodiscard]] auto gram_matrix(const Matrix& a) -> Result<Matrix>;

// The RATIONAL eigenvalues of A^T*A -- i.e. the exact rational squared singular values
// sigma^2 -- each paired with its algebraic multiplicity, sorted DESCENDING. This is a
// SLICE: an irrational sigma^2 (the common case) is honestly ABSENT, never approximated.
// Propagates gram_matrix's errors and nimblecas.eigen::rational_eigenvalues' errors.
[[nodiscard]] auto exact_singular_value_squares(const Matrix& a)
    -> Result<std::vector<std::pair<Rational, std::int64_t>>>;

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

// a < b for exact Rationals, via a 128-bit cross-multiply (both denominators are
// canonically positive, so no sign flip is needed and the products cannot overflow
// int64*int64 widened to __int128). Used only to order exact_singular_value_squares'
// output -- never to compute a returned value, so this stays outside the checked-Rational
// arithmetic surface deliberately.
[[nodiscard]] auto rational_less(const Rational& a, const Rational& b) -> bool {
    const __int128 lhs = static_cast<__int128>(a.numerator()) * static_cast<__int128>(b.denominator());
    const __int128 rhs = static_cast<__int128>(b.numerator()) * static_cast<__int128>(a.denominator());
    return lhs < rhs;
}

// One-sided Jacobi SVD core: rotates the m x n (m >= n) row-major working copy `B` and
// accumulates the rotations into the n x n row-major `V` (which must enter as the
// identity) until every column pair (p, q), p < q, has |<b_p,b_q>| <= tol*||b_p||*||b_q||
// in one full cyclic sweep. On return B's columns are mutually orthogonal (their norms
// are the singular values) and A0 = B0 == B_final * V^T where B0 was B's entry value (so
// A0's SVD-relevant content lives entirely in B_final and V). Fixed ascending pair order
// each sweep for determinism. Returns std::nullopt on success, or
// MathError::not_converged if max_sweeps sweeps all still find a rotation to apply.
[[nodiscard]] auto one_sided_jacobi(std::vector<double>& B, std::vector<double>& V,
                                    std::size_t m, std::size_t n, double tol,
                                    std::size_t max_sweeps) -> std::optional<MathError> {
    auto Bx = [&](std::size_t i, std::size_t j) -> double& { return B[i * n + j]; };
    auto Vx = [&](std::size_t i, std::size_t j) -> double& { return V[i * n + j]; };

    for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
        bool rotated = false;
        for (std::size_t p = 0; p + 1 < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                double alpha = 0.0, beta = 0.0, gamma = 0.0;  // ||b_p||^2, ||b_q||^2, <b_p,b_q>
                for (std::size_t i = 0; i < m; ++i) {
                    const double bip = Bx(i, p);
                    const double biq = Bx(i, q);
                    alpha += bip * bip;
                    beta += biq * biq;
                    gamma += bip * biq;
                }
                if (alpha == 0.0 || beta == 0.0) {
                    continue;  // a zero column carries no rotation information; skip 0/0
                }
                if (std::fabs(gamma) <= tol * std::sqrt(alpha * beta)) {
                    continue;  // already orthogonal (to tolerance)
                }
                rotated = true;
                // Rotation angle that zeros <b_p,b_q>: t = tan(phi) is the smaller root of
                // t^2 + 2*theta*t - 1 = 0 with theta = (beta - alpha) / (2*gamma).
                const double theta = (beta - alpha) / (2.0 * gamma);
                const double sign_theta = (theta >= 0.0) ? 1.0 : -1.0;
                const double t = sign_theta / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                for (std::size_t i = 0; i < m; ++i) {
                    const double bip = Bx(i, p);
                    const double biq = Bx(i, q);
                    Bx(i, p) = c * bip - s * biq;
                    Bx(i, q) = s * bip + c * biq;
                }
                for (std::size_t i = 0; i < n; ++i) {
                    const double vip = Vx(i, p);
                    const double viq = Vx(i, q);
                    Vx(i, p) = c * vip - s * viq;
                    Vx(i, q) = s * vip + c * viq;
                }
            }
        }
        if (!rotated) {
            return std::nullopt;  // a full clean sweep: converged
        }
    }
    return MathError::not_converged;  // budget exhausted; nothing partial is returned
}

// Sigma_j = ||b_j|| (Euclidean norm of column j of the converged m x n `B`) for every
// column, plus U (m x n, column j) filled by normalising when sigma_j is above the
// relative cutoff, or by deterministic Gram-Schmidt basis completion (standard basis e0,
// e1, ... in index order, against every U column ACCEPTED so far, taking the first
// surviving candidate whose residual norm exceeds 1/2; if none clears 1/2 -- which cannot
// happen once accepted.size() < m, since the standard basis vectors' squared residual
// norms against an orthonormal accepted set sum to m - accepted.size() >= 1 -- the
// largest residual found is used as a deterministic fallback, still guarded against 0/0
// so a zero residual never yields NaN) so U's columns stay a genuine orthonormal set.
auto fill_sigma_and_u(const std::vector<double>& B, std::size_t m, std::size_t n,
                      std::vector<double>& sigma, std::vector<double>& U) -> void {
    sigma.assign(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        double acc = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            const double v = B[i * n + j];
            acc += v * v;
        }
        sigma[j] = std::sqrt(acc);
    }
    double sigma_max = 0.0;
    for (double s : sigma) {
        sigma_max = std::max(sigma_max, s);
    }
    const double cutoff = sigma_max * std::numeric_limits<double>::epsilon() *
                          static_cast<double>(std::max<std::size_t>(m, 1)) * 10.0;

    U.assign(m * n, 0.0);
    std::vector<std::vector<double>> accepted;  // orthonormal U columns found so far
    accepted.reserve(n);

    // Pass 1: accept every genuine (above-cutoff) column u_j = b_j / sigma_j FIRST. The
    // basis completion below must see the full set of real singular directions, regardless
    // of where the zero columns sit in index order -- otherwise a zero column preceding a
    // nonzero one (e.g. rank-1 [[1,2],[2,4]], whose column 0 is annihilated) would complete
    // against an incomplete set and land non-orthogonal to a later real column.
    for (std::size_t j = 0; j < n; ++j) {
        if (sigma[j] <= cutoff) {
            continue;
        }
        std::vector<double> col(m);
        for (std::size_t i = 0; i < m; ++i) {
            col[i] = B[i * n + j] / sigma[j];
        }
        for (std::size_t i = 0; i < m; ++i) {
            U[i * n + j] = col[i];
        }
        accepted.push_back(std::move(col));
    }

    // Pass 2: complete each zero/degenerate column to a unit vector orthogonal to EVERY
    // accepted column (all the real ones, plus any earlier completion column), by
    // deterministic Gram-Schmidt of the standard basis e0, e1, ... in index order, taking
    // the first candidate whose residual norm exceeds 1/2 (the largest residual is used as a
    // deterministic fallback, guarded against 0/0 so it never yields NaN).
    for (std::size_t j = 0; j < n; ++j) {
        if (sigma[j] > cutoff) {
            continue;
        }
        double best_norm = -1.0;
        std::vector<double> best;
        for (std::size_t k = 0; k < m && best_norm <= 0.5; ++k) {
            std::vector<double> e(m, 0.0);
            e[k] = 1.0;
            for (const auto& a : accepted) {
                double dot = 0.0;
                for (std::size_t i = 0; i < m; ++i) {
                    dot += a[i] * e[i];
                }
                for (std::size_t i = 0; i < m; ++i) {
                    e[i] -= dot * a[i];
                }
            }
            double norm2 = 0.0;
            for (double v : e) {
                norm2 += v * v;
            }
            const double norm = std::sqrt(norm2);
            if (norm > best_norm) {
                best_norm = norm;
                best = std::move(e);
            }
        }
        // best_norm is provably > 0 while accepted.size() < m; guard 0/0 anyway.
        std::vector<double> unit(m, 0.0);
        if (best_norm > 0.0) {
            for (std::size_t i = 0; i < m; ++i) {
                unit[i] = best[i] / best_norm;
            }
        }
        for (std::size_t i = 0; i < m; ++i) {
            U[i * n + j] = unit[i];
        }
        accepted.push_back(std::move(unit));
    }
}

// Stable descending sort of `sigma` (length n), permuting the columns of the m x n `U`
// and the n x n `V` (row-major, k = n columns each) identically, then fixing each
// column's sign so its largest-magnitude entry (lowest index on ties) is non-negative,
// negating the paired U and V columns together.
auto sort_and_sign(std::vector<double>& sigma, std::vector<double>& U, std::vector<double>& V,
                   std::size_t m, std::size_t n) -> void {
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    std::stable_sort(idx.begin(), idx.end(),
                     [&](std::size_t a, std::size_t b) { return sigma[a] > sigma[b]; });

    std::vector<double> sigma2(n);
    std::vector<double> U2(m * n);
    std::vector<double> V2(n * n);
    for (std::size_t j = 0; j < n; ++j) {
        const std::size_t src = idx[j];
        sigma2[j] = sigma[src];
        for (std::size_t i = 0; i < m; ++i) {
            U2[i * n + j] = U[i * n + src];
        }
        for (std::size_t i = 0; i < n; ++i) {
            V2[i * n + j] = V[i * n + src];
        }
    }
    sigma = std::move(sigma2);
    U = std::move(U2);
    V = std::move(V2);

    for (std::size_t j = 0; j < n; ++j) {
        std::size_t best_row = 0;
        double best_mag = -1.0;
        for (std::size_t i = 0; i < m; ++i) {
            const double mag = std::fabs(U[i * n + j]);
            if (mag > best_mag) {
                best_mag = mag;
                best_row = i;
            }
        }
        if (U[best_row * n + j] < 0.0) {
            for (std::size_t i = 0; i < m; ++i) {
                U[i * n + j] = -U[i * n + j];
            }
            for (std::size_t i = 0; i < n; ++i) {
                V[i * n + j] = -V[i * n + j];
            }
        }
    }
}

}  // namespace

auto svd(std::span<const double> a, std::size_t rows, std::size_t cols, double tol,
         std::size_t max_sweeps) -> Result<NumericSvd> {
    using RetType = NumericSvd;
    if (!area_fits(rows, cols)) {
        return make_error<RetType>(MathError::overflow);
    }
    if (a.size() != rows * cols) {
        return make_error<RetType>(MathError::domain_error);
    }
    if (rows == 0 || cols == 0) {
        return NumericSvd{rows, cols, {}, {}, {}};
    }

    const bool transposed = rows < cols;
    const std::size_t m = transposed ? cols : rows;  // internally always m >= n
    const std::size_t n = transposed ? rows : cols;
    if (!area_fits(m, n) || !area_fits(n, n)) {
        return make_error<RetType>(MathError::overflow);
    }

    std::vector<double> B(m * n);
    if (transposed) {
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                B[j * n + i] = a[i * cols + j];  // B = A^T
            }
        }
    } else {
        B.assign(a.begin(), a.end());
    }
    std::vector<double> V(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        V[i * n + i] = 1.0;
    }

    if (auto err = one_sided_jacobi(B, V, m, n, tol, max_sweeps); err) {
        return make_error<RetType>(*err);
    }

    std::vector<double> sigma;
    std::vector<double> U;
    fill_sigma_and_u(B, m, n, sigma, U);
    sort_and_sign(sigma, U, V, m, n);

    if (!transposed) {
        return NumericSvd{rows, cols, std::move(U), std::move(sigma), std::move(V)};
    }
    // A = B0^T, so SVD(A) = V_B * diag(sigma) * U_B^T: swap U and V on return.
    return NumericSvd{rows, cols, std::move(V), std::move(sigma), std::move(U)};
}

auto svd(const Matrix& a, double tol, std::size_t max_sweeps) -> Result<NumericSvd> {
    using RetType = NumericSvd;
    const std::size_t rows = a.rows();
    const std::size_t cols = a.cols();
    if (!area_fits(rows, cols)) {
        return make_error<RetType>(MathError::overflow);
    }
    std::vector<double> ad;
    ad.reserve(rows * cols);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            const Rational& r = a.at(i, j);
            ad.push_back(static_cast<double>(r.numerator()) / static_cast<double>(r.denominator()));
        }
    }
    return svd(std::span<const double>(ad), rows, cols, tol, max_sweeps);
}

auto polar(std::span<const double> a, std::size_t n, double tol, std::size_t max_sweeps)
    -> Result<NumericPolar> {
    using RetType = NumericPolar;
    if (!area_fits(n, n)) {
        return make_error<RetType>(MathError::overflow);
    }
    if (a.size() != n * n) {
        return make_error<RetType>(MathError::domain_error);
    }
    if (n == 0) {
        return NumericPolar{0, {}, {}};
    }
    auto d = svd(a, n, n, tol, max_sweeps);
    if (!d) {
        return make_error<RetType>(d.error());
    }
    const auto& s = *d;  // s.u, s.v are n x n; s.singular_values has length n

    // U_p = U * V^T.
    std::vector<double> up(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t t = 0; t < n; ++t) {
                sum += s.u[i * n + t] * s.v[j * n + t];
            }
            up[i * n + j] = sum;
        }
    }
    // P = V * diag(sigma) * V^T, then symmetrised: P := (P + P^T) / 2.
    std::vector<double> pp(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t t = 0; t < n; ++t) {
                sum += s.v[i * n + t] * s.singular_values[t] * s.v[j * n + t];
            }
            pp[i * n + j] = sum;
        }
    }
    std::vector<double> psym(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            psym[i * n + j] = 0.5 * (pp[i * n + j] + pp[j * n + i]);
        }
    }
    return NumericPolar{n, std::move(up), std::move(psym)};
}

auto svd_residual(const NumericSvd& d, std::span<const double> a) -> Result<double> {
    using RetType = double;
    const std::size_t rows = d.rows;
    const std::size_t cols = d.cols;
    const std::size_t k = d.singular_values.size();
    if (!area_fits(rows, cols) || a.size() != rows * cols) {
        return make_error<RetType>(MathError::domain_error);
    }
    if (!area_fits(rows, k) || !area_fits(cols, k)) {
        return make_error<RetType>(MathError::overflow);
    }
    if (d.u.size() != rows * k || d.v.size() != cols * k) {
        return make_error<RetType>(MathError::domain_error);
    }
    double acc = 0.0;
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            double sum = 0.0;
            for (std::size_t t = 0; t < k; ++t) {
                sum += d.u[i * k + t] * d.singular_values[t] * d.v[j * k + t];
            }
            const double e = sum - a[i * cols + j];
            acc += e * e;
        }
    }
    return std::sqrt(acc);
}

auto polar_residual(const NumericPolar& d, std::span<const double> a) -> Result<double> {
    using RetType = double;
    const std::size_t n = d.n;
    if (!area_fits(n, n) || a.size() != n * n || d.u.size() != n * n || d.p.size() != n * n) {
        return make_error<RetType>(MathError::domain_error);
    }
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t t = 0; t < n; ++t) {
                sum += d.u[i * n + t] * d.p[t * n + j];
            }
            const double e = sum - a[i * n + j];
            acc += e * e;
        }
    }
    return std::sqrt(acc);
}

auto gram_matrix(const Matrix& a) -> Result<Matrix> {
    using RetType = Matrix;
    TRY(at, a.transpose());
    TRY(g, at.multiply(a));
    return g;
}

auto exact_singular_value_squares(const Matrix& a)
    -> Result<std::vector<std::pair<Rational, std::int64_t>>> {
    using RetType = std::vector<std::pair<Rational, std::int64_t>>;
    TRY(g, gram_matrix(a));
    TRY(eigs, rational_eigenvalues(g));
    std::sort(eigs.begin(), eigs.end(), [](const auto& lhs, const auto& rhs) {
        return rational_less(rhs.first, lhs.first);  // descending
    });
    return eigs;
}

}  // namespace nimblecas
