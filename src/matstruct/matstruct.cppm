// NimbleCAS structured matrices & exact-over-Q factorizations (ROADMAP 7.2).
// @author Olumuyiwa Oluwasanmi
//
// This module adds the *structured* side of dense linear algebra to the exact-rational
// stack: builders that assemble block/diagonal shapes, predicates that recognise those
// shapes, and the two symmetric factorizations that stay inside the field Q.
//
// -------------------------------------------------------------------------------------
// HONESTY BOUNDARY (this is load-bearing, not a disclaimer).
// -------------------------------------------------------------------------------------
// Everything in this module is EXACT over the rationals — every pivot, multiplier, and
// comparison is a reduced int64 fraction with no rounding, exactly as in matrix.cppm and
// matdecomp.cppm. Concretely:
//
//   * ldlt_decompose computes A = L * D * L^T with L unit-lower-triangular and D diagonal.
//     This is the exact-over-Q analogue of Cholesky: it takes NO square roots, so L and D
//     are exact rational matrices and L*D*L^T reconstructs A identically. (Classical
//     Cholesky A = G*G^T pushes sqrt(D_ii) into G, which is irrational in general.)
//
//   * cholesky_exact returns a genuine A = G*G^T with lower-triangular G ONLY when every
//     LDL^T pivot D_ii is a positive PERFECT RATIONAL SQUARE (then G = L * diag(sqrt D)).
//     When some D_ii is positive but not a perfect square, sqrt(D_ii) is irrational and no
//     exact rational G exists: we return domain_error rather than silently rounding. A
//     numerically-rounded Cholesky (accepting sqrt of any positive pivot) is the job of a
//     bigfloat / arbitrary-precision-float layer, NOT this exact-rational module.
//
//   * hessenberg_form reduces A to upper-Hessenberg by RATIONAL elementary similarity
//     (Gaussian-style, non-orthogonal): H = N * A * N^{-1} where N is a product of exact
//     rational Gauss/permutation transforms. H is upper-Hessenberg and SIMILAR to A, so it
//     has the SAME characteristic polynomial (same eigenvalues, determinant, trace). This
//     is deliberately NOT the orthogonal Householder Hessenberg (Q^T A Q with Q orthogonal)
//     — orthogonal reduction needs sqrt for the reflector norms and lands outside Q. The
//     rational similarity form is the honest exact counterpart and is all that a purely
//     rational spectral pipeline (e.g. characteristic_polynomial) needs.
//
//   * is_symmetric_positive_definite decides positive-definiteness EXACTLY through the sign
//     of the LDL^T pivots (A symmetric and every D_ii > 0). No eigenvalue estimation, no
//     tolerance.
//
//   * A block-diagonal (or block-upper-triangular) matrix has determinant equal to the
//     product of its diagonal blocks' determinants — exact, and directly verifiable via
//     Matrix::determinant() on the assembled matrix.
//
// The structural predicates that already live in matdecomp.cppm (is_diagonal, is_symmetric,
// is_upper_hessenberg, ...) are NOT duplicated here; this module reuses them via
// `import nimblecas.matdecomp` and only adds the block-structure predicates and the
// LDL^T-based positive-definiteness test.

export module nimblecas.matstruct;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.matdecomp;  // reuse is_symmetric et al. (no duplicated predicates)

export namespace nimblecas {

// ---------------------------------------------------------------------------
// LdltDecomposition — the result of ldlt_decompose: A = L * D * L^T.
// ---------------------------------------------------------------------------
// `l` is unit lower-triangular (l(i,i) == 1, l(i,j) == 0 for j > i). `d` is a diagonal
// matrix holding the pivots D_ii on its diagonal (0 off-diagonal). Reconstruction is
// L * D * L^T, exact over Rational.
struct LdltDecomposition {
    Matrix l;
    Matrix d;
};

// --- structured builders (all exact over Rational) -------------------------

// The n x n diagonal matrix with `entries` on the main diagonal (0 elsewhere), where
// n == entries.size(). An empty list yields the 0x0 matrix. Fails only if the internal
// row assembly fails (never for a well-formed square shape).
[[nodiscard]] auto diagonal(const std::vector<Rational>& entries) -> Result<Matrix>;

// The block-diagonal matrix diag(blocks[0], blocks[1], ...): each block is placed on the
// diagonal with zeros elsewhere. The result is (sum of block rows) x (sum of block cols);
// blocks need not be square (though a square result — required for a determinant — needs
// square blocks). An empty list yields the 0x0 matrix. Fails with overflow if the summed
// dimensions would wrap std::size_t.
[[nodiscard]] auto block_diagonal(const std::vector<Matrix>& blocks) -> Result<Matrix>;

// A block-upper-triangular matrix assembled from an upper-triangular grid of blocks given
// in jagged form: block_rows[i] lists the blocks of block-row i starting at block-column i,
// so block_rows[i][0] is the (i,i) DIAGONAL block and block_rows[i][t] is the (i, i+t)
// block. Block-row i must therefore supply exactly (k - i) blocks, where k is the number of
// block-rows. Diagonal blocks must be SQUARE (so the assembled matrix is square); an
// off-diagonal (i, i+t) block must have as many rows as the (i,i) diagonal block and as
// many columns as the (i+t, i+t) diagonal block. Blocks strictly below the block-diagonal
// are implicitly zero. Shape violations fail with domain_error; summed-dimension overflow
// fails with overflow. An empty grid yields the 0x0 matrix.
[[nodiscard]] auto block_upper_triangular(const std::vector<std::vector<Matrix>>& block_rows)
    -> Result<Matrix>;

// --- block-structure predicates (exact; the scalar-shape predicates live in matdecomp) ---

// Is A block-diagonal for the given partition `block_sizes` (which must sum to n for a
// square n x n A)? Every entry whose row and column fall in different diagonal blocks must
// be exactly zero. A non-square matrix, or a partition that does not sum to n, yields false.
[[nodiscard]] auto is_block_diagonal(const Matrix& a, const std::vector<std::size_t>& block_sizes)
    -> bool;

// Is A block-upper-triangular for the given partition `block_sizes`? Every entry whose row
// block index exceeds its column block index (strictly below the block-diagonal) must be
// exactly zero; entries on or above the block-diagonal are unconstrained. A non-square
// matrix, or a partition that does not sum to n, yields false.
[[nodiscard]] auto is_block_upper_triangular(const Matrix& a,
                                             const std::vector<std::size_t>& block_sizes) -> bool;

// Is A symmetric positive definite? Decided EXACTLY: A must be symmetric and the exact
// LDL^T factorization must exist with every pivot D_ii > 0. A non-symmetric matrix, a zero
// pivot (indefinite/semidefinite/singular), or a negative pivot yields false. If an
// intermediate rational operation overflows int64 the result is conservatively false (the
// matrix could not be certified positive definite within the int64 rational tier — the
// bigrational tier lifts that ceiling).
[[nodiscard]] auto is_symmetric_positive_definite(const Matrix& a) -> bool;

// --- exact symmetric factorizations ----------------------------------------

// Exact rational LDL^T: A = L * D * L^T with L unit-lower-triangular and D diagonal, taking
// no square roots. Requires A square and symmetric (domain_error otherwise). This is the
// no-pivoting factorization; if a zero pivot D_jj == 0 is encountered a symmetric pivot
// (row/column interchange) would be required to continue, which this exact form does not
// perform, so it fails with domain_error. Entry-arithmetic overflow propagates as overflow.
[[nodiscard]] auto ldlt_decompose(const Matrix& a) -> Result<LdltDecomposition>;

// Exact Cholesky A = G * G^T with G lower-triangular, available ONLY when A is symmetric
// positive definite AND every LDL^T pivot D_ii is a perfect rational square. In that case
// G = L * diag(sqrt(D_ii)) is exact over Q. If A is not symmetric or the factorization hits
// a zero pivot, fails with domain_error (as ldlt_decompose does). If a pivot is <= 0
// (not positive definite) or positive but NOT a perfect rational square (sqrt(D_ii) is
// irrational, so no exact rational G exists), fails with domain_error: the exact-rational
// path deliberately refuses to round. A rounded Cholesky belongs to a bigfloat layer.
[[nodiscard]] auto cholesky_exact(const Matrix& a) -> Result<Matrix>;

// Exact rational upper-Hessenberg reduction by elementary similarity: returns H = N*A*N^{-1}
// (N a product of exact rational Gauss/permutation transforms) that is upper-Hessenberg and
// SIMILAR to A — same characteristic polynomial, eigenvalues, determinant and trace. This is
// the Gaussian (non-orthogonal) reduction; it is NOT the orthogonal Householder Hessenberg
// (which needs sqrt and leaves Q). Requires A square (domain_error otherwise). Entry
// overflow propagates as overflow.
[[nodiscard]] auto hessenberg_form(const Matrix& a) -> Result<Matrix>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// Overflow-checked std::size_t addition (dimension bookkeeping for block assembly).
[[nodiscard]] auto checked_add(std::size_t a, std::size_t b) -> std::optional<std::size_t> {
    std::size_t s = 0;
    if (__builtin_add_overflow(a, b, &s)) {
        return std::nullopt;
    }
    return s;
}

// If v is a non-negative perfect square, return its integer square root; else nullopt.
// Overflow-safe: candidate squares are formed with a checked multiply.
[[nodiscard]] auto integer_sqrt(std::int64_t v) -> std::optional<std::int64_t> {
    if (v < 0) {
        return std::nullopt;
    }
    if (v == 0) {
        return std::int64_t{0};
    }
    const auto guess = static_cast<std::int64_t>(std::sqrt(static_cast<double>(v)));
    // The double sqrt is exact-or-adjacent for int64 perfect squares; scan a small window.
    for (std::int64_t c = guess - 1; c <= guess + 1; ++c) {
        if (c < 0) {
            continue;
        }
        std::int64_t sq = 0;
        if (!__builtin_mul_overflow(c, c, &sq) && sq == v) {
            return c;
        }
    }
    return std::nullopt;
}

// The exact rational square root of r, when it exists: r must be a non-negative perfect
// rational square. A reduced fraction num/den (gcd == 1, den > 0) is a perfect square iff
// both num and den are perfect squares, and then sqrt = sqrt(num)/sqrt(den). Returns nullopt
// for negative r or when the root is irrational.
[[nodiscard]] auto exact_rational_sqrt(const Rational& r) -> std::optional<Rational> {
    if (r.numerator() < 0) {
        return std::nullopt;
    }
    const auto sn = integer_sqrt(r.numerator());
    const auto sd = integer_sqrt(r.denominator());
    if (!sn || !sd) {
        return std::nullopt;
    }
    auto root = Rational::make(*sn, *sd);  // sd > 0, gcd(sn, sd) == 1: never fails here
    if (!root) {
        return std::nullopt;
    }
    return *root;
}

// Assemble a Matrix from a flat row-major buffer of exactly rows*cols Rationals.
[[nodiscard]] auto matrix_from_flat(const std::vector<Rational>& buf, std::size_t rows,
                                    std::size_t cols) -> Result<Matrix> {
    std::vector<std::vector<Rational>> out(rows, std::vector<Rational>(cols));
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            out[i][j] = buf[i * cols + j];
        }
    }
    return Matrix::from_rows(std::move(out));
}

// Map each of n indices to the block it belongs to under the partition `sizes`. Returns
// nullopt if the partition does not sum to exactly n.
[[nodiscard]] auto block_of(const std::vector<std::size_t>& sizes, std::size_t n)
    -> std::optional<std::vector<std::size_t>> {
    std::vector<std::size_t> owner(n);
    std::size_t idx = 0;
    for (std::size_t b = 0; b < sizes.size(); ++b) {
        for (std::size_t t = 0; t < sizes[b]; ++t) {
            if (idx >= n) {
                return std::nullopt;  // partition overshoots n
            }
            owner[idx++] = b;
        }
    }
    if (idx != n) {
        return std::nullopt;  // partition undershoots n
    }
    return owner;
}

}  // namespace

// --- structured builders ----------------------------------------------------

auto diagonal(const std::vector<Rational>& entries) -> Result<Matrix> {
    const std::size_t n = entries.size();
    std::vector<Rational> buf(n * n);  // Rational{} default is 0/1
    for (std::size_t i = 0; i < n; ++i) {
        buf[i * n + i] = entries[i];
    }
    return matrix_from_flat(buf, n, n);
}

auto block_diagonal(const std::vector<Matrix>& blocks) -> Result<Matrix> {
    std::size_t total_rows = 0;
    std::size_t total_cols = 0;
    for (const auto& b : blocks) {
        auto tr = checked_add(total_rows, b.rows());
        auto tc = checked_add(total_cols, b.cols());
        if (!tr || !tc) {
            return make_error<Matrix>(MathError::overflow);
        }
        total_rows = *tr;
        total_cols = *tc;
    }
    if (total_rows == 0 && total_cols == 0) {
        return Matrix{};  // 0x0
    }
    std::vector<std::vector<Rational>> out(total_rows, std::vector<Rational>(total_cols));
    std::size_t row_off = 0;
    std::size_t col_off = 0;
    for (const auto& b : blocks) {
        for (std::size_t i = 0; i < b.rows(); ++i) {
            for (std::size_t j = 0; j < b.cols(); ++j) {
                out[row_off + i][col_off + j] = b.at(i, j);
            }
        }
        row_off += b.rows();
        col_off += b.cols();
    }
    return Matrix::from_rows(std::move(out));
}

auto block_upper_triangular(const std::vector<std::vector<Matrix>>& block_rows) -> Result<Matrix> {
    const std::size_t k = block_rows.size();
    if (k == 0) {
        return Matrix{};  // 0x0
    }
    // Each block-row i must supply exactly (k - i) blocks: diagonal (i,i) then (i, i+1..k-1).
    for (std::size_t i = 0; i < k; ++i) {
        if (block_rows[i].size() != k - i) {
            return make_error<Matrix>(MathError::domain_error);
        }
    }
    // Diagonal blocks must be square; collect their sizes and the prefix offsets.
    std::vector<std::size_t> dim(k);
    std::size_t total = 0;
    for (std::size_t i = 0; i < k; ++i) {
        const Matrix& diag_block = block_rows[i][0];
        if (!diag_block.is_square()) {
            return make_error<Matrix>(MathError::domain_error);
        }
        dim[i] = diag_block.rows();
        auto t = checked_add(total, dim[i]);
        if (!t) {
            return make_error<Matrix>(MathError::overflow);
        }
        total = *t;
    }
    std::vector<std::size_t> offset(k);
    std::size_t acc = 0;
    for (std::size_t i = 0; i < k; ++i) {
        offset[i] = acc;
        acc += dim[i];
    }
    std::vector<std::vector<Rational>> out(total, std::vector<Rational>(total));
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t t = 0; t < block_rows[i].size(); ++t) {
            const std::size_t j = i + t;  // column-block index
            const Matrix& blk = block_rows[i][t];
            // Off-diagonal (i,j) must be dim[i] x dim[j]; the diagonal block already is
            // dim[i] x dim[i] by the squareness check above.
            if (blk.rows() != dim[i] || blk.cols() != dim[j]) {
                return make_error<Matrix>(MathError::domain_error);
            }
            for (std::size_t r = 0; r < blk.rows(); ++r) {
                for (std::size_t c = 0; c < blk.cols(); ++c) {
                    out[offset[i] + r][offset[j] + c] = blk.at(r, c);
                }
            }
        }
    }
    return Matrix::from_rows(std::move(out));
}

// --- block-structure predicates ---------------------------------------------

auto is_block_diagonal(const Matrix& a, const std::vector<std::size_t>& block_sizes) -> bool {
    if (!a.is_square()) {
        return false;
    }
    const std::size_t n = a.rows();
    const auto owner = block_of(block_sizes, n);
    if (!owner) {
        return false;
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if ((*owner)[i] != (*owner)[j] && !a.at(i, j).is_zero()) {
                return false;  // off-block entry must be zero
            }
        }
    }
    return true;
}

auto is_block_upper_triangular(const Matrix& a, const std::vector<std::size_t>& block_sizes)
    -> bool {
    if (!a.is_square()) {
        return false;
    }
    const std::size_t n = a.rows();
    const auto owner = block_of(block_sizes, n);
    if (!owner) {
        return false;
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            // Strictly below the block-diagonal: row block index > column block index.
            if ((*owner)[i] > (*owner)[j] && !a.at(i, j).is_zero()) {
                return false;
            }
        }
    }
    return true;
}

auto is_symmetric_positive_definite(const Matrix& a) -> bool {
    auto ldlt = ldlt_decompose(a);  // fails on non-symmetric, zero pivot, or overflow
    if (!ldlt) {
        return false;
    }
    const std::size_t n = a.rows();
    for (std::size_t i = 0; i < n; ++i) {
        // A Rational is positive iff its numerator is (denominator is always > 0).
        if (ldlt->d.at(i, i).numerator() <= 0) {
            return false;
        }
    }
    return true;
}

// --- exact LDL^T -------------------------------------------------------------

auto ldlt_decompose(const Matrix& a) -> Result<LdltDecomposition> {
    if (!a.is_square()) {
        return make_error<LdltDecomposition>(MathError::domain_error);
    }
    if (!is_symmetric(a)) {  // from nimblecas.matdecomp
        return make_error<LdltDecomposition>(MathError::domain_error);
    }
    const std::size_t n = a.rows();
    const Rational one = Rational::from_int(1);

    // L stored full (row-major); D as a vector of pivots. L is built with a unit diagonal.
    std::vector<Rational> l(n * n);  // 0/1 default
    std::vector<Rational> d(n);

    for (std::size_t j = 0; j < n; ++j) {
        // D_jj = A_jj - sum_{k<j} L_jk^2 * D_k.
        Rational acc;  // 0/1
        for (std::size_t k = 0; k < j; ++k) {
            auto sq = l[j * n + k].multiply(l[j * n + k]);
            if (!sq) {
                return make_error<LdltDecomposition>(sq.error());
            }
            auto term = sq->multiply(d[k]);
            if (!term) {
                return make_error<LdltDecomposition>(term.error());
            }
            auto sum = acc.add(*term);
            if (!sum) {
                return make_error<LdltDecomposition>(sum.error());
            }
            acc = *sum;
        }
        auto djj = a.at(j, j).subtract(acc);
        if (!djj) {
            return make_error<LdltDecomposition>(djj.error());
        }
        if (djj->is_zero()) {
            // A zero pivot would force a symmetric interchange; the exact form declines.
            return make_error<LdltDecomposition>(MathError::domain_error);
        }
        d[j] = *djj;
        l[j * n + j] = one;

        // L_ij = (A_ij - sum_{k<j} L_ik * L_jk * D_k) / D_jj, for i > j.
        for (std::size_t i = j + 1; i < n; ++i) {
            Rational acc2;  // 0/1
            for (std::size_t k = 0; k < j; ++k) {
                auto p1 = l[i * n + k].multiply(l[j * n + k]);
                if (!p1) {
                    return make_error<LdltDecomposition>(p1.error());
                }
                auto p2 = p1->multiply(d[k]);
                if (!p2) {
                    return make_error<LdltDecomposition>(p2.error());
                }
                auto sum = acc2.add(*p2);
                if (!sum) {
                    return make_error<LdltDecomposition>(sum.error());
                }
                acc2 = *sum;
            }
            auto num = a.at(i, j).subtract(acc2);
            if (!num) {
                return make_error<LdltDecomposition>(num.error());
            }
            auto lij = num->divide(*djj);  // djj != 0
            if (!lij) {
                return make_error<LdltDecomposition>(lij.error());
            }
            l[i * n + j] = *lij;
        }
    }

    auto lmat = matrix_from_flat(l, n, n);
    if (!lmat) {
        return make_error<LdltDecomposition>(lmat.error());
    }
    std::vector<Rational> dbuf(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        dbuf[i * n + i] = d[i];
    }
    auto dmat = matrix_from_flat(dbuf, n, n);
    if (!dmat) {
        return make_error<LdltDecomposition>(dmat.error());
    }
    return LdltDecomposition{.l = std::move(*lmat), .d = std::move(*dmat)};
}

// --- exact Cholesky (perfect-square pivots only) ----------------------------

auto cholesky_exact(const Matrix& a) -> Result<Matrix> {
    auto ldlt = ldlt_decompose(a);  // domain_error if non-symmetric or zero pivot
    if (!ldlt) {
        return make_error<Matrix>(ldlt.error());
    }
    const std::size_t n = a.rows();

    // Each column j of G is column j of L scaled by s_j = sqrt(D_jj). This is exact only
    // when D_jj is a positive perfect rational square; otherwise the exact path refuses.
    std::vector<Rational> s(n);
    for (std::size_t j = 0; j < n; ++j) {
        const Rational djj = ldlt->d.at(j, j);
        if (djj.numerator() <= 0) {
            return make_error<Matrix>(MathError::domain_error);  // not positive definite
        }
        auto root = exact_rational_sqrt(djj);
        if (!root) {
            return make_error<Matrix>(MathError::domain_error);  // irrational sqrt: not exact
        }
        s[j] = *root;
    }

    std::vector<Rational> g(n * n);  // lower-triangular, 0/1 default fills the upper part
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            auto gij = ldlt->l.at(i, j).multiply(s[j]);
            if (!gij) {
                return make_error<Matrix>(gij.error());
            }
            g[i * n + j] = *gij;
        }
    }
    return matrix_from_flat(g, n, n);
}

// --- exact rational Hessenberg reduction (elementary similarity) ------------

auto hessenberg_form(const Matrix& a) -> Result<Matrix> {
    if (!a.is_square()) {
        return make_error<Matrix>(MathError::domain_error);
    }
    const std::size_t n = a.rows();

    // Working copy, row-major.
    std::vector<Rational> m(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            m[i * n + j] = a.at(i, j);
        }
    }

    // Reduce column k (below the subdiagonal, i.e. rows k+2..n-1) for each k with k+2 < n.
    for (std::size_t k = 0; k + 2 < n; ++k) {
        const std::size_t p = k + 1;  // subdiagonal pivot row for column k

        // Ensure a nonzero pivot at (p, k); if zero, a symmetric row/column interchange
        // (a similarity by a transposition) brings a nonzero entry up. If the whole column
        // below the subdiagonal is already zero, nothing to eliminate here.
        if (m[p * n + k].is_zero()) {
            std::size_t r = p + 1;
            while (r < n && m[r * n + k].is_zero()) {
                ++r;
            }
            if (r == n) {
                continue;  // column already reduced
            }
            for (std::size_t j = 0; j < n; ++j) {  // swap rows p and r
                std::swap(m[p * n + j], m[r * n + j]);
            }
            for (std::size_t i = 0; i < n; ++i) {  // swap columns p and r (completes P A P)
                std::swap(m[i * n + p], m[i * n + r]);
            }
        }

        const Rational pivot = m[p * n + k];

        // Multipliers factor[i] = A_ik / A_pk for i = k+2..n-1.
        std::vector<Rational> factor(n);  // 0/1 default; only k+2..n-1 are meaningful
        for (std::size_t i = p + 1; i < n; ++i) {
            if (m[i * n + k].is_zero()) {
                continue;
            }
            auto f = m[i * n + k].divide(pivot);
            if (!f) {
                return make_error<Matrix>(f.error());
            }
            factor[i] = *f;
        }

        // Left transform N: row_i -= factor[i] * row_p (zeros column k below the subdiagonal).
        // Applied as a batch so every row reads the same, unmodified pivot row p.
        for (std::size_t i = p + 1; i < n; ++i) {
            if (factor[i].is_zero()) {
                continue;
            }
            for (std::size_t j = 0; j < n; ++j) {
                auto prod = factor[i].multiply(m[p * n + j]);
                if (!prod) {
                    return make_error<Matrix>(prod.error());
                }
                auto diff = m[i * n + j].subtract(*prod);
                if (!diff) {
                    return make_error<Matrix>(diff.error());
                }
                m[i * n + j] = *diff;
            }
        }

        // Right transform N^{-1}: col_p += factor[i] * col_i (restores similarity). Applied
        // after all row ops, so each col_i is read in its post-row-op state (correct for
        // (N A) N^{-1}); column ops touch only column p, leaving the col_i sources intact.
        for (std::size_t i = p + 1; i < n; ++i) {
            if (factor[i].is_zero()) {
                continue;
            }
            for (std::size_t r = 0; r < n; ++r) {
                auto prod = factor[i].multiply(m[r * n + i]);
                if (!prod) {
                    return make_error<Matrix>(prod.error());
                }
                auto sum = m[r * n + p].add(*prod);
                if (!sum) {
                    return make_error<Matrix>(sum.error());
                }
                m[r * n + p] = *sum;
            }
        }
    }

    return matrix_from_flat(m, n, n);
}

}  // namespace nimblecas
