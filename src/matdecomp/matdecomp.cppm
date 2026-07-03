// NimbleCAS special matrices & LU decomposition (ROADMAP 7.2).
// @author Olumuyiwa Oluwasanmi
//
// Exact LU decomposition over the field Q, plus a family of structural predicates that
// recognise the shape of a matrix (symmetric, triangular, banded, Toeplitz, ...). Because
// entries are Rational, every pivot, multiplier, and comparison is exact: the LU factors
// reconstruct P*A = L*U identically (not "up to rounding"), and a matrix is reported
// symmetric only when A(i,j) equals A(j,i) as fractions with no tolerance involved.
//
// The decomposition is Doolittle elimination with partial pivoting: L is unit
// lower-triangular (1s on its diagonal), U is upper-triangular, `permutation` records the
// row order applied to A, and `sign` is the determinant of that permutation (+1/-1). Over
// exact arithmetic any nonzero pivot serves, so we take the first nonzero entry at or below
// the diagonal. A column that is entirely zero at/below the pivot means the matrix is
// singular and no unit-lower/upper factorisation with the standard normalisation exists,
// which surfaces as MathError::domain_error.

export module nimblecas.matdecomp;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// LuDecomposition — the result of lu_decompose: P*A = L*U.
// ---------------------------------------------------------------------------
// `l` is unit lower-triangular (l(i,i) == 1, l(i,j) == 0 for j > i), `u` is
// upper-triangular (u(i,j) == 0 for i > j). `permutation` is the row order applied to A:
// permutation[i] is the index of the original row of A that now sits in position i, so
// (P*A) row i equals A row permutation[i]. `sign` is det(P), i.e. +1 for an even number of
// row swaps and -1 for an odd number.
struct LuDecomposition {
    Matrix l;
    Matrix u;
    std::vector<std::size_t> permutation;
    int sign;
};

// Exact LU decomposition with partial pivoting: computes P, L, U with P*A = L*U.
//
// Requires A square (domain_error otherwise). Uses Doolittle elimination over Rational,
// selecting at each column the first nonzero pivot at or below the diagonal and recording
// the row swap in `permutation` (flipping `sign` each time). If an entire column at/below
// the pivot is zero, A is singular and no standard unit-lower/upper factorisation exists,
// so the call fails with domain_error. Entry-arithmetic overflow propagates as overflow.
[[nodiscard]] auto lu_decompose(const Matrix& a) -> Result<LuDecomposition>;

// --- structure predicates (exact, over Rational) ---------------------------
//
// Predicates that imply squareness (symmetric, skew-symmetric, diagonal, triangular,
// tridiagonal, Hessenberg, identity) return false for a non-square input. is_toeplitz and
// is_banded also accept rectangular matrices.

// A(i,j) == A(j,i) for all i, j.
[[nodiscard]] auto is_symmetric(const Matrix& a) -> bool;

// A(i,j) == -A(j,i) for all i, j (forcing a zero diagonal).
[[nodiscard]] auto is_skew_symmetric(const Matrix& a) -> bool;

// Every off-diagonal entry is zero.
[[nodiscard]] auto is_diagonal(const Matrix& a) -> bool;

// Every entry below the main diagonal is zero (A(i,j) == 0 for i > j).
[[nodiscard]] auto is_upper_triangular(const Matrix& a) -> bool;

// Every entry above the main diagonal is zero (A(i,j) == 0 for i < j).
[[nodiscard]] auto is_lower_triangular(const Matrix& a) -> bool;

// Zero outside the main diagonal and the first sub/super-diagonals (|i - j| <= 1).
[[nodiscard]] auto is_tridiagonal(const Matrix& a) -> bool;

// A(i,j) == 0 whenever j < i - lower_bandwidth or j > i + upper_bandwidth. Rectangular
// matrices are accepted; the unsigned subtraction i - lower_bandwidth is guarded.
[[nodiscard]] auto is_banded(const Matrix& a, std::size_t lower_bandwidth,
                             std::size_t upper_bandwidth) -> bool;

// Zero below the first subdiagonal (A(i,j) == 0 for i > j + 1).
[[nodiscard]] auto is_upper_hessenberg(const Matrix& a) -> bool;

// Zero above the first superdiagonal (A(i,j) == 0 for j > i + 1).
[[nodiscard]] auto is_lower_hessenberg(const Matrix& a) -> bool;

// Constant along every diagonal: A(i,j) == A(i-1,j-1) for all i, j >= 1. Rectangular
// matrices are accepted.
[[nodiscard]] auto is_toeplitz(const Matrix& a) -> bool;

// Square with 1s on the diagonal and 0s elsewhere.
[[nodiscard]] auto is_identity(const Matrix& a) -> bool;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// Is entry (i, j) of A equal to the given integer constant?
[[nodiscard]] auto entry_is(const Matrix& a, std::size_t i, std::size_t j, std::int64_t v)
    -> bool {
    return a.at(i, j) == Rational::from_int(v);
}

}  // namespace

// --- LU decomposition -------------------------------------------------------

auto lu_decompose(const Matrix& a) -> Result<LuDecomposition> {
    if (!a.is_square()) {
        return make_error<LuDecomposition>(MathError::domain_error);
    }
    const std::size_t n = a.rows();

    // Working copy `m` (row-major, n x n). After elimination it holds U in its upper
    // triangle (diagonal included) and the L multipliers strictly below the diagonal.
    std::vector<Rational> m(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            m[i * n + j] = a.at(i, j);
        }
    }

    std::vector<std::size_t> perm(n);
    for (std::size_t i = 0; i < n; ++i) {
        perm[i] = i;
    }
    int sign = 1;

    for (std::size_t k = 0; k < n; ++k) {
        // First nonzero pivot at or below the diagonal in column k. Any nonzero pivot is
        // valid for exact arithmetic, so the first one keeps the multipliers simple.
        std::size_t pivot_row = k;
        bool found = false;
        for (std::size_t r = k; r < n; ++r) {
            if (!m[r * n + k].is_zero()) {
                pivot_row = r;
                found = true;
                break;
            }
        }
        if (!found) {
            return make_error<LuDecomposition>(MathError::domain_error);  // singular
        }
        if (pivot_row != k) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(m[k * n + j], m[pivot_row * n + j]);
            }
            std::swap(perm[k], perm[pivot_row]);
            sign = -sign;  // each row swap flips det(P)
        }

        const Rational pivot = m[k * n + k];
        for (std::size_t i = k + 1; i < n; ++i) {
            auto factor = m[i * n + k].divide(pivot);
            if (!factor) {
                return make_error<LuDecomposition>(factor.error());
            }
            m[i * n + k] = *factor;  // store the L multiplier in place
            for (std::size_t j = k + 1; j < n; ++j) {
                auto prod = factor->multiply(m[k * n + j]);
                if (!prod) {
                    return make_error<LuDecomposition>(prod.error());
                }
                auto diff = m[i * n + j].subtract(*prod);
                if (!diff) {
                    return make_error<LuDecomposition>(diff.error());
                }
                m[i * n + j] = *diff;
            }
        }
    }

    // Split `m` into unit-lower L and upper U.
    const Rational one = Rational::from_int(1);
    std::vector<std::vector<Rational>> lrows(n, std::vector<Rational>(n));
    std::vector<std::vector<Rational>> urows(n, std::vector<Rational>(n));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (i > j) {
                lrows[i][j] = m[i * n + j];  // multiplier
            } else if (i == j) {
                lrows[i][j] = one;
                urows[i][j] = m[i * n + j];
            } else {
                urows[i][j] = m[i * n + j];
            }
        }
    }

    auto l = Matrix::from_rows(std::move(lrows));
    if (!l) {
        return make_error<LuDecomposition>(l.error());
    }
    auto u = Matrix::from_rows(std::move(urows));
    if (!u) {
        return make_error<LuDecomposition>(u.error());
    }
    return LuDecomposition{std::move(*l), std::move(*u), std::move(perm), sign};
}

// --- structure predicates ---------------------------------------------------

auto is_symmetric(const Matrix& a) -> bool {
    if (!a.is_square()) {
        return false;
    }
    const std::size_t n = a.rows();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (!(a.at(i, j) == a.at(j, i))) {
                return false;
            }
        }
    }
    return true;
}

auto is_skew_symmetric(const Matrix& a) -> bool {
    if (!a.is_square()) {
        return false;
    }
    const std::size_t n = a.rows();
    for (std::size_t i = 0; i < n; ++i) {
        // Diagonal must be zero: A(i,i) == -A(i,i) forces A(i,i) == 0.
        if (!a.at(i, i).is_zero()) {
            return false;
        }
        for (std::size_t j = i + 1; j < n; ++j) {
            auto neg = a.at(j, i).negate();
            if (!neg || !(a.at(i, j) == *neg)) {
                return false;
            }
        }
    }
    return true;
}

auto is_diagonal(const Matrix& a) -> bool {
    if (!a.is_square()) {
        return false;
    }
    const std::size_t n = a.rows();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (i != j && !a.at(i, j).is_zero()) {
                return false;
            }
        }
    }
    return true;
}

auto is_upper_triangular(const Matrix& a) -> bool {
    if (!a.is_square()) {
        return false;
    }
    const std::size_t n = a.rows();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j) {  // strictly below the diagonal
            if (!a.at(i, j).is_zero()) {
                return false;
            }
        }
    }
    return true;
}

auto is_lower_triangular(const Matrix& a) -> bool {
    if (!a.is_square()) {
        return false;
    }
    const std::size_t n = a.rows();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {  // strictly above the diagonal
            if (!a.at(i, j).is_zero()) {
                return false;
            }
        }
    }
    return true;
}

auto is_tridiagonal(const Matrix& a) -> bool {
    if (!a.is_square()) {
        return false;
    }
    const std::size_t n = a.rows();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            // Zero outside the main diagonal and the first sub/super-diagonals.
            const std::size_t diff = (i > j) ? (i - j) : (j - i);
            if (diff > 1 && !a.at(i, j).is_zero()) {
                return false;
            }
        }
    }
    return true;
}

auto is_banded(const Matrix& a, std::size_t lower_bandwidth, std::size_t upper_bandwidth)
    -> bool {
    const std::size_t rows = a.rows();
    const std::size_t cols = a.cols();
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            // Below the lower band: j < i - lower_bandwidth (guard the unsigned subtraction).
            const bool below = (i > lower_bandwidth) && (j < i - lower_bandwidth);
            // Above the upper band: j > i + upper_bandwidth.
            const bool above = j > i + upper_bandwidth;
            if ((below || above) && !a.at(i, j).is_zero()) {
                return false;
            }
        }
    }
    return true;
}

auto is_upper_hessenberg(const Matrix& a) -> bool {
    if (!a.is_square()) {
        return false;
    }
    const std::size_t n = a.rows();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (i > j + 1 && !a.at(i, j).is_zero()) {  // below the first subdiagonal
                return false;
            }
        }
    }
    return true;
}

auto is_lower_hessenberg(const Matrix& a) -> bool {
    if (!a.is_square()) {
        return false;
    }
    const std::size_t n = a.rows();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (j > i + 1 && !a.at(i, j).is_zero()) {  // above the first superdiagonal
                return false;
            }
        }
    }
    return true;
}

auto is_toeplitz(const Matrix& a) -> bool {
    const std::size_t rows = a.rows();
    const std::size_t cols = a.cols();
    for (std::size_t i = 1; i < rows; ++i) {
        for (std::size_t j = 1; j < cols; ++j) {
            if (!(a.at(i, j) == a.at(i - 1, j - 1))) {
                return false;
            }
        }
    }
    return true;
}

auto is_identity(const Matrix& a) -> bool {
    if (!a.is_square()) {
        return false;
    }
    const std::size_t n = a.rows();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const std::int64_t expected = (i == j) ? 1 : 0;
            if (!entry_is(a, i, j, expected)) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace nimblecas
