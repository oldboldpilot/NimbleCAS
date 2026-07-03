// NimbleCAS dense matrices over the rationals, Q^{m x n} (ROADMAP 7.2).
// @author Olumuyiwa Oluwasanmi
//
// Linear algebra done *exactly*. Entries are Rational (a reduced int64 fraction), so
// determinants, solutions of linear systems, and inverses are computed with no rounding
// error at all — a 3x3 determinant is the integer/fraction it mathematically is, not a
// double that happens to be close. Elimination is performed over the field Q, which is
// why plain Gaussian / Gauss-Jordan elimination suffices (no fraction-free Bareiss step
// is needed to stay in a ring — Rational already reduces every intermediate).
//
// Following the rest of the engine, arithmetic is overflow-checked (Rule 32): every
// entry operation flows through Rational's checked add/subtract/multiply/divide, so an
// int64 numerator or denominator that would overflow surfaces as MathError::overflow
// rather than silently wrapping. Dimension violations surface as MathError::domain_error,
// and a singular system (no inverse / no unique solution) surfaces as domain_error too.
//
// Storage is dense row-major: entry (i, j) lives at data_[i * cols_ + j].

module;
#include <cassert>  // assert macro (unavailable via `import std`); active when !NDEBUG

export module nimblecas.matrix;

import std;
import nimblecas.core;
import nimblecas.ratpoly;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// Matrix — a dense rows_ x cols_ grid of exact Rational entries.
// ---------------------------------------------------------------------------
// The default-constructed matrix is the empty 0x0 matrix (a valid, if degenerate,
// value: its determinant is the empty product 1). Every fallible operation returns
// Result; the only infallible observers are the accessors below.
class Matrix {
public:
    Matrix() = default;  // the empty 0x0 matrix

    // Build from a list of rows. Every row must have the same length, otherwise the
    // matrix is ragged and construction fails with domain_error. An empty list yields
    // the 0x0 matrix; a list of empty rows yields an r x 0 matrix.
    [[nodiscard]] static auto from_rows(std::vector<std::vector<Rational>> rows) -> Result<Matrix>;

    // The n x n identity (1 on the diagonal, 0 elsewhere). identity(0) is the 0x0 matrix.
    [[nodiscard]] static auto identity(std::size_t n) -> Matrix;

    // The rows x cols all-zero matrix.
    [[nodiscard]] static auto zero(std::size_t rows, std::size_t cols) -> Matrix;

    [[nodiscard]] auto rows() const noexcept -> std::size_t { return rows_; }
    [[nodiscard]] auto cols() const noexcept -> std::size_t { return cols_; }
    [[nodiscard]] auto is_square() const noexcept -> bool { return rows_ == cols_; }

    // Entry (i, j). Asserted in-range: callers hold indices below rows()/cols().
    [[nodiscard]] auto at(std::size_t i, std::size_t j) const -> const Rational& {
        assert(i < rows_ && j < cols_ && "Matrix::at out of range");
        return data_[i * cols_ + j];
    }

    [[nodiscard]] auto operator==(const Matrix& o) const noexcept -> bool {
        return rows_ == o.rows_ && cols_ == o.cols_ && data_ == o.data_;
    }
    [[nodiscard]] auto is_equal(const Matrix& o) const noexcept -> bool { return *this == o; }

    // --- arithmetic (all Result; dimension / overflow errors propagate) ---

    // Entrywise sum / difference; dimensions must match, else domain_error.
    [[nodiscard]] auto add(const Matrix& o) const -> Result<Matrix>;
    [[nodiscard]] auto subtract(const Matrix& o) const -> Result<Matrix>;

    // Multiply every entry by a scalar.
    [[nodiscard]] auto scale(const Rational& s) const -> Result<Matrix>;

    // Matrix product (this is m x k, o is k x n). Fails with domain_error when the inner
    // dimensions disagree (cols() != o.rows()).
    [[nodiscard]] auto multiply(const Matrix& o) const -> Result<Matrix>;

    // The cols_ x rows_ transpose. Never fails, but returns Result for a uniform surface.
    [[nodiscard]] auto transpose() const -> Result<Matrix>;

    // Sum of the diagonal entries; requires a square matrix (domain_error otherwise).
    [[nodiscard]] auto trace() const -> Result<Rational>;

    // --- decompositions / solving ---

    // The exact determinant via Gaussian elimination with pivoting on nonzero pivots.
    // Requires a square matrix (domain_error otherwise); the 0x0 determinant is 1.
    [[nodiscard]] auto determinant() const -> Result<Rational>;

    // Solve A x = b for x, where A is this (square, nonsingular) and b has the same
    // number of rows as A (b may carry several right-hand-side columns). Uses Gauss-
    // Jordan elimination with partial pivoting on nonzero pivots. Fails with domain_error
    // when A is not square, the row counts disagree, or A is singular.
    [[nodiscard]] auto solve(const Matrix& b) const -> Result<Matrix>;

    // The multiplicative inverse (A A^{-1} = I). Requires a square, nonsingular matrix;
    // a singular or non-square matrix yields domain_error.
    [[nodiscard]] auto inverse() const -> Result<Matrix>;

    // The rank (number of linearly independent rows) via row reduction over Q. This is
    // infallible by signature; should an intermediate entry overflow int64, reduction
    // stops early and the rank counted so far is returned (a conservative lower bound).
    [[nodiscard]] auto rank() const -> std::int64_t;

    [[nodiscard]] auto to_string() const -> std::string;

private:
    Matrix(std::size_t rows, std::size_t cols, std::vector<Rational> data)
        : rows_(rows), cols_(cols), data_(std::move(data)) {}

    std::size_t rows_{0};
    std::size_t cols_{0};
    std::vector<Rational> data_{};  // row-major, size rows_ * cols_
};

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// Guard a rows*cols allocation size against std::size_t wrap before reserving.
[[nodiscard]] auto fits_size(std::size_t rows, std::size_t cols) -> bool {
    return rows == 0 || cols <= std::numeric_limits<std::size_t>::max() / rows;
}

// Swap two rows of a row-major buffer that has `total` columns.
auto swap_rows(std::vector<Rational>& m, std::size_t total, std::size_t a, std::size_t b) -> void {
    for (std::size_t j = 0; j < total; ++j) {
        std::swap(m[a * total + j], m[b * total + j]);
    }
}

// Reduce the leading n columns of a row-major augmented matrix (n rows, `total`
// columns, total >= n) to the identity via Gauss-Jordan elimination with partial
// pivoting on nonzero pivots. Any trailing columns are carried along, so on success
// they hold A^{-1} times the original trailing block. Fails with domain_error when the
// leading block is singular, or with overflow when an entry operation overflows.
[[nodiscard]] auto gauss_jordan(std::vector<Rational>& m, std::size_t n, std::size_t total)
    -> Result<void> {
    for (std::size_t col = 0; col < n; ++col) {
        // Locate a nonzero pivot at or below the diagonal in this column.
        std::size_t pivot_row = col;
        bool found = false;
        for (std::size_t r = col; r < n; ++r) {
            if (!m[r * total + col].is_zero()) {
                pivot_row = r;
                found = true;
                break;
            }
        }
        if (!found) {
            return make_error<void>(MathError::domain_error);  // singular
        }
        if (pivot_row != col) {
            swap_rows(m, total, col, pivot_row);
        }
        // Normalise the pivot row so the pivot becomes 1.
        const Rational pivot = m[col * total + col];
        for (std::size_t j = col; j < total; ++j) {
            auto q = m[col * total + j].divide(pivot);
            if (!q) {
                return make_error<void>(q.error());
            }
            m[col * total + j] = *q;
        }
        // Clear the pivot column in every other row.
        for (std::size_t r = 0; r < n; ++r) {
            if (r == col) {
                continue;
            }
            const Rational factor = m[r * total + col];
            if (factor.is_zero()) {
                continue;
            }
            for (std::size_t j = col; j < total; ++j) {
                auto prod = m[col * total + j].multiply(factor);
                if (!prod) {
                    return make_error<void>(prod.error());
                }
                auto diff = m[r * total + j].subtract(*prod);
                if (!diff) {
                    return make_error<void>(diff.error());
                }
                m[r * total + j] = *diff;
            }
        }
    }
    return {};
}

}  // namespace

// --- construction -----------------------------------------------------------

auto Matrix::from_rows(std::vector<std::vector<Rational>> rows) -> Result<Matrix> {
    if (rows.empty()) {
        return Matrix{};
    }
    const std::size_t nrows = rows.size();
    const std::size_t ncols = rows.front().size();
    for (const auto& row : rows) {
        if (row.size() != ncols) {
            return make_error<Matrix>(MathError::domain_error);  // ragged
        }
    }
    std::vector<Rational> data;
    data.reserve(nrows * ncols);  // nrows*ncols already realised by the caller's rows
    for (auto& row : rows) {
        for (auto& e : row) {
            data.push_back(e);
        }
    }
    return Matrix{nrows, ncols, std::move(data)};
}

auto Matrix::identity(std::size_t n) -> Matrix {
    assert(fits_size(n, n) && "Matrix::identity dimension too large");
    std::vector<Rational> data(n * n);  // Rational{} default is 0/1
    const Rational one = Rational::from_int(1);
    for (std::size_t i = 0; i < n; ++i) {
        data[i * n + i] = one;
    }
    return Matrix{n, n, std::move(data)};
}

auto Matrix::zero(std::size_t rows, std::size_t cols) -> Matrix {
    assert(fits_size(rows, cols) && "Matrix::zero dimension too large");
    return Matrix{rows, cols, std::vector<Rational>(rows * cols)};
}

// --- arithmetic -------------------------------------------------------------

auto Matrix::add(const Matrix& o) const -> Result<Matrix> {
    if (rows_ != o.rows_ || cols_ != o.cols_) {
        return make_error<Matrix>(MathError::domain_error);
    }
    std::vector<Rational> data(data_.size());
    for (std::size_t i = 0; i < data_.size(); ++i) {
        auto sum = data_[i].add(o.data_[i]);
        if (!sum) {
            return make_error<Matrix>(sum.error());
        }
        data[i] = *sum;
    }
    return Matrix{rows_, cols_, std::move(data)};
}

auto Matrix::subtract(const Matrix& o) const -> Result<Matrix> {
    if (rows_ != o.rows_ || cols_ != o.cols_) {
        return make_error<Matrix>(MathError::domain_error);
    }
    std::vector<Rational> data(data_.size());
    for (std::size_t i = 0; i < data_.size(); ++i) {
        auto diff = data_[i].subtract(o.data_[i]);
        if (!diff) {
            return make_error<Matrix>(diff.error());
        }
        data[i] = *diff;
    }
    return Matrix{rows_, cols_, std::move(data)};
}

auto Matrix::scale(const Rational& s) const -> Result<Matrix> {
    std::vector<Rational> data(data_.size());
    for (std::size_t i = 0; i < data_.size(); ++i) {
        auto p = data_[i].multiply(s);
        if (!p) {
            return make_error<Matrix>(p.error());
        }
        data[i] = *p;
    }
    return Matrix{rows_, cols_, std::move(data)};
}

auto Matrix::multiply(const Matrix& o) const -> Result<Matrix> {
    if (cols_ != o.rows_) {
        return make_error<Matrix>(MathError::domain_error);
    }
    const std::size_t n = o.cols_;
    const std::size_t k = cols_;  // == o.rows_
    std::vector<Rational> data(rows_ * n);
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            Rational acc;  // 0/1
            for (std::size_t p = 0; p < k; ++p) {
                auto term = data_[i * cols_ + p].multiply(o.data_[p * n + j]);
                if (!term) {
                    return make_error<Matrix>(term.error());
                }
                auto sum = acc.add(*term);
                if (!sum) {
                    return make_error<Matrix>(sum.error());
                }
                acc = *sum;
            }
            data[i * n + j] = acc;
        }
    }
    return Matrix{rows_, n, std::move(data)};
}

auto Matrix::transpose() const -> Result<Matrix> {
    std::vector<Rational> data(data_.size());
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            data[j * rows_ + i] = data_[i * cols_ + j];
        }
    }
    return Matrix{cols_, rows_, std::move(data)};
}

auto Matrix::trace() const -> Result<Rational> {
    if (!is_square()) {
        return make_error<Rational>(MathError::domain_error);
    }
    Rational acc;  // 0/1
    for (std::size_t i = 0; i < rows_; ++i) {
        auto sum = acc.add(data_[i * cols_ + i]);
        if (!sum) {
            return make_error<Rational>(sum.error());
        }
        acc = *sum;
    }
    return acc;
}

// --- determinant ------------------------------------------------------------

auto Matrix::determinant() const -> Result<Rational> {
    if (!is_square()) {
        return make_error<Rational>(MathError::domain_error);
    }
    const std::size_t n = rows_;
    if (n == 0) {
        return Rational::from_int(1);  // empty product
    }
    std::vector<Rational> m = data_;  // working copy
    Rational det = Rational::from_int(1);
    for (std::size_t col = 0; col < n; ++col) {
        // Find a nonzero pivot at or below the diagonal.
        std::size_t pivot_row = col;
        bool found = false;
        for (std::size_t r = col; r < n; ++r) {
            if (!m[r * n + col].is_zero()) {
                pivot_row = r;
                found = true;
                break;
            }
        }
        if (!found) {
            return Rational{};  // an all-zero column below => singular => det 0
        }
        if (pivot_row != col) {
            swap_rows(m, n, col, pivot_row);
            auto neg = det.negate();  // a row swap flips the determinant's sign
            if (!neg) {
                return make_error<Rational>(neg.error());
            }
            det = *neg;
        }
        const Rational pivot = m[col * n + col];
        auto scaled = det.multiply(pivot);  // det == product of pivots (times sign)
        if (!scaled) {
            return make_error<Rational>(scaled.error());
        }
        det = *scaled;
        // Eliminate entries below the pivot.
        for (std::size_t r = col + 1; r < n; ++r) {
            const Rational below = m[r * n + col];
            if (below.is_zero()) {
                continue;
            }
            auto factor = below.divide(pivot);
            if (!factor) {
                return make_error<Rational>(factor.error());
            }
            for (std::size_t j = col; j < n; ++j) {
                auto prod = m[col * n + j].multiply(*factor);
                if (!prod) {
                    return make_error<Rational>(prod.error());
                }
                auto diff = m[r * n + j].subtract(*prod);
                if (!diff) {
                    return make_error<Rational>(diff.error());
                }
                m[r * n + j] = *diff;
            }
        }
    }
    return det;
}

// --- solve / inverse --------------------------------------------------------

auto Matrix::solve(const Matrix& b) const -> Result<Matrix> {
    if (!is_square() || b.rows_ != rows_) {
        return make_error<Matrix>(MathError::domain_error);
    }
    const std::size_t n = rows_;
    const std::size_t rhs = b.cols_;
    const std::size_t total = n + rhs;
    // Assemble the augmented matrix [A | b].
    std::vector<Rational> m(n * total);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            m[i * total + j] = data_[i * cols_ + j];
        }
        for (std::size_t j = 0; j < rhs; ++j) {
            m[i * total + n + j] = b.data_[i * b.cols_ + j];
        }
    }
    auto reduced = gauss_jordan(m, n, total);  // singular => domain_error
    if (!reduced) {
        return make_error<Matrix>(reduced.error());
    }
    // The trailing rhs columns now hold the solution x.
    std::vector<Rational> data(n * rhs);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < rhs; ++j) {
            data[i * rhs + j] = m[i * total + n + j];
        }
    }
    return Matrix{n, rhs, std::move(data)};
}

auto Matrix::inverse() const -> Result<Matrix> {
    if (!is_square()) {
        return make_error<Matrix>(MathError::domain_error);
    }
    const std::size_t n = rows_;
    const std::size_t total = 2 * n;
    // Assemble [A | I]; after Gauss-Jordan the right half is A^{-1}.
    std::vector<Rational> m(n * total);
    const Rational one = Rational::from_int(1);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            m[i * total + j] = data_[i * cols_ + j];
        }
        m[i * total + n + i] = one;
    }
    auto reduced = gauss_jordan(m, n, total);  // singular => domain_error
    if (!reduced) {
        return make_error<Matrix>(reduced.error());
    }
    std::vector<Rational> data(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            data[i * n + j] = m[i * total + n + j];
        }
    }
    return Matrix{n, n, std::move(data)};
}

auto Matrix::rank() const -> std::int64_t {
    if (rows_ == 0 || cols_ == 0) {
        return 0;
    }
    std::vector<Rational> m = data_;  // working copy
    std::size_t rank = 0;
    for (std::size_t col = 0; col < cols_ && rank < rows_; ++col) {
        // Find a nonzero pivot at or below the current rank row in this column.
        std::size_t pivot_row = rank;
        bool found = false;
        for (std::size_t r = rank; r < rows_; ++r) {
            if (!m[r * cols_ + col].is_zero()) {
                pivot_row = r;
                found = true;
                break;
            }
        }
        if (!found) {
            continue;  // free column, move on without consuming a rank
        }
        if (pivot_row != rank) {
            swap_rows(m, cols_, rank, pivot_row);
        }
        const Rational pivot = m[rank * cols_ + col];
        for (std::size_t r = rank + 1; r < rows_; ++r) {
            const Rational below = m[r * cols_ + col];
            if (below.is_zero()) {
                continue;
            }
            auto factor = below.divide(pivot);
            if (!factor) {
                return static_cast<std::int64_t>(rank);  // overflow: conservative result
            }
            for (std::size_t j = col; j < cols_; ++j) {
                auto prod = m[rank * cols_ + j].multiply(*factor);
                if (!prod) {
                    return static_cast<std::int64_t>(rank);
                }
                auto diff = m[r * cols_ + j].subtract(*prod);
                if (!diff) {
                    return static_cast<std::int64_t>(rank);
                }
                m[r * cols_ + j] = *diff;
            }
        }
        ++rank;
    }
    return static_cast<std::int64_t>(rank);
}

auto Matrix::to_string() const -> std::string {
    std::string out = "[";
    for (std::size_t i = 0; i < rows_; ++i) {
        out += (i == 0) ? "[" : ", [";
        for (std::size_t j = 0; j < cols_; ++j) {
            if (j != 0) {
                out += ", ";
            }
            out += data_[i * cols_ + j].to_string();
        }
        out += "]";
    }
    out += "]";
    return out;
}

}  // namespace nimblecas
