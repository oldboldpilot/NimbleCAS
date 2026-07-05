// NimbleCAS Kronecker product and related structured matrix products, over Q (ROADMAP 7.2).
// @author Olumuyiwa Oluwasanmi
//
// Structured products of exact rational matrices — every entry is a Rational (a reduced
// int64 fraction), so the Kronecker product, Kronecker sum, direct sum, Hadamard product
// and the vec / unvec reshapings all carry NO rounding error whatsoever. The result of
// A (x) B is the block matrix it mathematically is, not a double that happens to be close.
//
// Following the rest of the engine, arithmetic is overflow-checked (Rule 32): every entry
// product flows through Rational's checked multiply, so an int64 numerator or denominator
// that would overflow surfaces as MathError::overflow rather than silently wrapping. A
// result whose total row/column count would overflow std::size_t likewise surfaces as
// MathError::overflow rather than allocating a wrapped-around buffer. Shape violations
// (non-square where a square is required, mismatched dimensions, a non-column argument to
// unvec) surface as MathError::domain_error. There is no approximation anywhere here: the
// module is exact and complete over Q.
//
// The vec convention is COLUMN-MAJOR, matching the Lyapunov / Stein vec-trick used inside
// nimblecas.analysis: vec(A) stacks the columns of A top to bottom, so for an m x n matrix
// the entry A(i, j) lands at row (j * m + i) of the resulting (m*n) x 1 column vector. The
// mixed-product identity vec(A X B) = (B^T (x) A) vec(X) holds with this convention.

module;
#include <cassert>  // assert macro (unavailable via `import std`); active when !NDEBUG

export module nimblecas.kronprod;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// Kronecker product A (x) B.
// ---------------------------------------------------------------------------
// For A of shape m x n and B of shape p x q, the result is the (m*p) x (n*q) block matrix
// whose (i, j) block is a_ij * B; equivalently, the entry at (i*p + k, j*q + l) equals
// A(i, j) * B(k, l). This mirrors the internal kron() helper in nimblecas.analysis exactly
// (there specialised to square operands). Fails with overflow if an entry product wraps
// int64, or if the (m*p) x (n*q) dimensions would overflow std::size_t.
[[nodiscard]] auto kronecker_product(const Matrix& a, const Matrix& b) -> Result<Matrix>;

// ---------------------------------------------------------------------------
// Kronecker sum A (+) B = A (x) I_n + I_m (x) B.
// ---------------------------------------------------------------------------
// Defined for square operands: A is m x m and B is n x n. The result is the (m*n) x (m*n)
// matrix A (x) I_n + I_m (x) B. A non-square A or B yields domain_error; entry overflow or
// a dimension overflow yields overflow.
[[nodiscard]] auto kronecker_sum(const Matrix& a, const Matrix& b) -> Result<Matrix>;

// ---------------------------------------------------------------------------
// Direct sum diag(A, B).
// ---------------------------------------------------------------------------
// The block-diagonal matrix with A in the top-left and B in the bottom-right, zeros
// elsewhere: shape (rows(A)+rows(B)) x (cols(A)+cols(B)). Fails with overflow only if the
// summed dimensions would overflow std::size_t. Never a shape error — any two matrices
// have a direct sum.
[[nodiscard]] auto direct_sum(const Matrix& a, const Matrix& b) -> Result<Matrix>;

// ---------------------------------------------------------------------------
// Hadamard (elementwise) product A (o) B.
// ---------------------------------------------------------------------------
// The entrywise product: result(i, j) = A(i, j) * B(i, j). A and B must share the same
// shape, else domain_error; an entry product that wraps int64 yields overflow.
[[nodiscard]] auto hadamard_product(const Matrix& a, const Matrix& b) -> Result<Matrix>;

// ---------------------------------------------------------------------------
// vec / unvec — column-major vectorisation.
// ---------------------------------------------------------------------------
// vec(A) stacks the columns of the m x n matrix A into a single (m*n) x 1 column: the entry
// A(i, j) lands at row (j * m + i). Fails with overflow only if m*n overflows std::size_t.
[[nodiscard]] auto vec(const Matrix& a) -> Result<Matrix>;

// unvec(v, rows) is the inverse of vec: v must be a single column (cols() == 1) whose
// length is a multiple of `rows` (with rows > 0 unless v is empty), and the result is the
// rows x (len/rows) matrix M with M(i, j) = v(j * rows + i, 0). A non-column v, a length
// not divisible by `rows`, or rows == 0 with a non-empty v yields domain_error.
[[nodiscard]] auto unvec(const Matrix& v, std::size_t rows) -> Result<Matrix>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================

// Unwrap a Result or propagate its error as make_error<RetType>. Each enclosing function
// defines `using RetType = ...;` before its first use.
#define TRY(var, ...)                                                    \
    auto var##__r = (__VA_ARGS__);                                       \
    if (!(var##__r)) return make_error<RetType>((var##__r).error());     \
    auto var = std::move(*var##__r)

namespace nimblecas {
namespace {

// Multiply two std::size_t dimensions, reporting overflow rather than wrapping. A product
// that would exceed SIZE_MAX (so the flat buffer could not be addressed) is an honest
// overflow, mirroring the int64 overflow discipline used for entry arithmetic.
[[nodiscard]] auto checked_mul(std::size_t a, std::size_t b) -> Result<std::size_t> {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        return make_error<std::size_t>(MathError::overflow);
    }
    return a * b;
}

// Add two std::size_t dimensions, reporting overflow rather than wrapping.
[[nodiscard]] auto checked_add(std::size_t a, std::size_t b) -> Result<std::size_t> {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        return make_error<std::size_t>(MathError::overflow);
    }
    return a + b;
}

}  // namespace

// --- Kronecker product ------------------------------------------------------

auto kronecker_product(const Matrix& a, const Matrix& b) -> Result<Matrix> {
    using RetType = Matrix;
    const std::size_t am = a.rows();
    const std::size_t an = a.cols();
    const std::size_t bp = b.rows();
    const std::size_t bq = b.cols();
    TRY(big_rows, checked_mul(am, bp));
    TRY(big_cols, checked_mul(an, bq));
    // A zero-dimension result has no entries to fill; build it directly so its shape is
    // preserved (from_rows on an empty row list would collapse to 0 x 0).
    if (big_rows == 0 || big_cols == 0) {
        return Matrix::zero(big_rows, big_cols);
    }
    std::vector<std::vector<Rational>> rows(big_rows, std::vector<Rational>(big_cols, Rational{}));
    for (std::size_t i = 0; i < am; ++i) {
        for (std::size_t j = 0; j < an; ++j) {
            const Rational& aij = a.at(i, j);
            for (std::size_t k = 0; k < bp; ++k) {
                for (std::size_t l = 0; l < bq; ++l) {
                    TRY(prod, aij.multiply(b.at(k, l)));
                    rows[i * bp + k][j * bq + l] = std::move(prod);
                }
            }
        }
    }
    return Matrix::from_rows(std::move(rows));
}

// --- Kronecker sum ----------------------------------------------------------

auto kronecker_sum(const Matrix& a, const Matrix& b) -> Result<Matrix> {
    using RetType = Matrix;
    if (!a.is_square() || !b.is_square()) {
        return make_error<Matrix>(MathError::domain_error);
    }
    const std::size_t m = a.rows();
    const std::size_t n = b.rows();
    TRY(left, kronecker_product(a, Matrix::identity(n)));   // A (x) I_n
    TRY(right, kronecker_product(Matrix::identity(m), b));  // I_m (x) B
    return left.add(right);  // matching (m*n) x (m*n) shapes; add propagates overflow
}

// --- direct sum -------------------------------------------------------------

auto direct_sum(const Matrix& a, const Matrix& b) -> Result<Matrix> {
    using RetType = Matrix;
    TRY(total_rows, checked_add(a.rows(), b.rows()));
    TRY(total_cols, checked_add(a.cols(), b.cols()));
    if (total_rows == 0 || total_cols == 0) {
        return Matrix::zero(total_rows, total_cols);
    }
    std::vector<std::vector<Rational>> rows(total_rows,
                                            std::vector<Rational>(total_cols, Rational{}));
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.cols(); ++j) {
            rows[i][j] = a.at(i, j);
        }
    }
    for (std::size_t i = 0; i < b.rows(); ++i) {
        for (std::size_t j = 0; j < b.cols(); ++j) {
            rows[a.rows() + i][a.cols() + j] = b.at(i, j);
        }
    }
    return Matrix::from_rows(std::move(rows));
}

// --- Hadamard product -------------------------------------------------------

auto hadamard_product(const Matrix& a, const Matrix& b) -> Result<Matrix> {
    using RetType = Matrix;
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        return make_error<Matrix>(MathError::domain_error);
    }
    if (a.rows() == 0 || a.cols() == 0) {
        return Matrix::zero(a.rows(), a.cols());
    }
    std::vector<std::vector<Rational>> rows(a.rows(), std::vector<Rational>(a.cols(), Rational{}));
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.cols(); ++j) {
            TRY(prod, a.at(i, j).multiply(b.at(i, j)));
            rows[i][j] = std::move(prod);
        }
    }
    return Matrix::from_rows(std::move(rows));
}

// --- vec / unvec ------------------------------------------------------------

auto vec(const Matrix& a) -> Result<Matrix> {
    using RetType = Matrix;
    const std::size_t m = a.rows();
    const std::size_t n = a.cols();
    TRY(len, checked_mul(m, n));
    if (len == 0) {
        return Matrix::zero(len, 1);  // an empty operand vectorises to the 0 x 1 column
    }
    std::vector<std::vector<Rational>> rows(len, std::vector<Rational>(1, Rational{}));
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < m; ++i) {
            rows[j * m + i][0] = a.at(i, j);  // column-major stacking
        }
    }
    return Matrix::from_rows(std::move(rows));
}

auto unvec(const Matrix& v, std::size_t rows) -> Result<Matrix> {
    using RetType = Matrix;
    if (v.cols() != 1) {
        return make_error<Matrix>(MathError::domain_error);  // vec() output is a single column
    }
    const std::size_t len = v.rows();
    if (len == 0) {
        return Matrix::zero(rows, 0);  // an empty column reshapes to rows x 0
    }
    if (rows == 0 || len % rows != 0) {
        return make_error<Matrix>(MathError::domain_error);
    }
    const std::size_t cols = len / rows;
    std::vector<std::vector<Rational>> out(rows, std::vector<Rational>(cols, Rational{}));
    for (std::size_t j = 0; j < cols; ++j) {
        for (std::size_t i = 0; i < rows; ++i) {
            out[i][j] = v.at(j * rows + i, 0);  // inverse of the column-major stacking
        }
    }
    return Matrix::from_rows(std::move(out));
}

}  // namespace nimblecas

#undef TRY
