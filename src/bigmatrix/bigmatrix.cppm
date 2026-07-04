// NimbleCAS big-backed dense matrix over BigRational (exact determinant, no overflow).
// @author Olumuyiwa Oluwasanmi
//
// This is the unbounded, overflow-free counterpart to nimblecas.matrix. That module holds
// its entries as the int64 Rational, so a determinant whose intermediate products or final
// value exceed int64 surfaces as MathError::overflow (Rule 32). BigMatrix removes that
// ceiling entirely: entries are BigRational (an arbitrary-precision BigInt fraction), so a
// determinant is the exact element of Q it mathematically is — no rounding, no overflow, no
// ceiling on dimension or magnitude. The cost is heap-allocating BigInt arithmetic on every
// entry operation, which makes this the slow-but-exact tier; values that comfortably fit
// int64 belong in nimblecas.matrix.
//
// HONESTY: exact and UNBOUNDED over Q. Where Matrix::determinant of diag(10^10, 10^10,
// 10^10) overflows int64 (the answer 10^30 dwarfs the ~9.2e18 int64 ceiling), this returns
// the exact 10^30. The trade-off is speed: BigInt limbs on the heap are far slower than a
// register-width int64 multiply.
//
// The headline determinant uses the FRACTION-FREE BAREISS algorithm: integer-preserving
// Gaussian elimination that divides each updated entry by the *previous* pivot with EXACT
// division. On integer inputs every intermediate stays an integer (no fraction ever
// appears); over Q it is still exact and keeps intermediate magnitudes far smaller than the
// fraction explosion of naive Gaussian elimination. Because BigRational is a field, that
// division is always exact and only the by-zero case could fail — and the divisor is always
// a prior nonzero pivot (or the seed 1), so it never does.
//
// Storage is dense row-major: entry (i, j) lives at data_[i * cols_ + j].

module;
#include <cassert>  // assert macro (unavailable via `import std`); active when !NDEBUG

export module nimblecas.bigmatrix;

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.ratpoly;
import nimblecas.matrix;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// BigMatrix — a dense rows_ x cols_ grid of exact BigRational entries.
// ---------------------------------------------------------------------------
// The default-constructed matrix is the empty 0x0 matrix (a valid, if degenerate, value:
// its determinant is the empty product 1). Every fallible operation returns Result; the
// only infallible observers are the accessors below.
class BigMatrix {
public:
    BigMatrix() = default;  // the empty 0x0 matrix

    // Build from a list of rows. Every row must have the same length, otherwise the matrix
    // is ragged and construction fails with domain_error. Unlike the empty-tolerant int64
    // Matrix, an empty row list is also rejected with domain_error (a 0x0 BigMatrix is still
    // reachable via the default constructor or identity(0)/zero(0, 0)).
    [[nodiscard]] static auto from_rows(std::vector<std::vector<BigRational>> rows)
        -> Result<BigMatrix>;

    // The n x n identity (1 on the diagonal, 0 elsewhere). identity(0) is the 0x0 matrix.
    [[nodiscard]] static auto identity(std::size_t n) -> BigMatrix;

    // The rows x cols all-zero matrix.
    [[nodiscard]] static auto zero(std::size_t rows, std::size_t cols) -> BigMatrix;

    // PROMOTE an int64-Rational Matrix into the unbounded tier, mapping each entry
    // a.at(i, j) to BigRational::make(from_i64(num), from_i64(den)). A Rational is always
    // canonical (den >= 1), so the promotion never fails; the Result surface is uniform.
    [[nodiscard]] static auto from_matrix(const Matrix& a) -> Result<BigMatrix>;

    [[nodiscard]] auto rows() const noexcept -> std::size_t { return rows_; }
    [[nodiscard]] auto cols() const noexcept -> std::size_t { return cols_; }
    [[nodiscard]] auto is_square() const noexcept -> bool { return rows_ == cols_; }

    // Entry (i, j). Asserted in-range: callers hold indices below rows()/cols().
    [[nodiscard]] auto at(std::size_t i, std::size_t j) const -> const BigRational& {
        assert(i < rows_ && j < cols_ && "BigMatrix::at out of range");
        return data_[i * cols_ + j];
    }

    [[nodiscard]] auto operator==(const BigMatrix& o) const noexcept -> bool {
        return rows_ == o.rows_ && cols_ == o.cols_ && data_ == o.data_;
    }
    [[nodiscard]] auto is_equal(const BigMatrix& o) const noexcept -> bool { return *this == o; }

    // --- arithmetic (all Result; dimension errors propagate as domain_error) ---
    // BigRational add/subtract/multiply/negate cannot overflow, so these never fail on the
    // arithmetic itself — only a dimension violation surfaces (as domain_error).

    // Entrywise sum / difference; dimensions must match, else domain_error.
    [[nodiscard]] auto add(const BigMatrix& o) const -> Result<BigMatrix>;
    [[nodiscard]] auto subtract(const BigMatrix& o) const -> Result<BigMatrix>;

    // Multiply every entry by a scalar. Never fails, but returns Result for a uniform surface.
    [[nodiscard]] auto scale(const BigRational& s) const -> Result<BigMatrix>;

    // Matrix product (this is m x k, o is k x n). Fails with domain_error when the inner
    // dimensions disagree (cols() != o.rows()).
    [[nodiscard]] auto multiply(const BigMatrix& o) const -> Result<BigMatrix>;

    // The cols_ x rows_ transpose. Never fails, but returns Result for a uniform surface.
    [[nodiscard]] auto transpose() const -> Result<BigMatrix>;

    // --- determinant (the headline) ---

    // The exact determinant via the fraction-free Bareiss algorithm (see the module note).
    // Requires a square matrix (domain_error otherwise); the 0x0 determinant is 1. A zero
    // pivot is resolved by a row swap (tracking the sign); a fully-zero column below the
    // pivot means the matrix is singular and the determinant is 0. Exact and unbounded: it
    // NEVER overflows, in contrast to the int64 Matrix::determinant.
    [[nodiscard]] auto determinant() const -> Result<BigRational>;

    [[nodiscard]] auto to_string() const -> std::string;

private:
    BigMatrix(std::size_t rows, std::size_t cols, std::vector<BigRational> data)
        : rows_(rows), cols_(cols), data_(std::move(data)) {}

    std::size_t rows_{0};
    std::size_t cols_{0};
    std::vector<BigRational> data_{};  // row-major, size rows_ * cols_
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
auto swap_rows(std::vector<BigRational>& m, std::size_t total, std::size_t a, std::size_t b)
    -> void {
    for (std::size_t j = 0; j < total; ++j) {
        std::swap(m[a * total + j], m[b * total + j]);
    }
}

}  // namespace

// --- construction -----------------------------------------------------------

auto BigMatrix::from_rows(std::vector<std::vector<BigRational>> rows) -> Result<BigMatrix> {
    if (rows.empty()) {
        return make_error<BigMatrix>(MathError::domain_error);  // empty list rejected
    }
    const std::size_t nrows = rows.size();
    const std::size_t ncols = rows.front().size();
    for (const auto& row : rows) {
        if (row.size() != ncols) {
            return make_error<BigMatrix>(MathError::domain_error);  // ragged
        }
    }
    std::vector<BigRational> data;
    data.reserve(nrows * ncols);  // nrows*ncols already realised by the caller's rows
    for (auto& row : rows) {
        for (auto& e : row) {
            data.push_back(std::move(e));
        }
    }
    return BigMatrix{nrows, ncols, std::move(data)};
}

auto BigMatrix::identity(std::size_t n) -> BigMatrix {
    assert(fits_size(n, n) && "BigMatrix::identity dimension too large");
    std::vector<BigRational> data(n * n);  // BigRational{} default is 0/1
    const BigRational one = BigRational::from_int(1);
    for (std::size_t i = 0; i < n; ++i) {
        data[i * n + i] = one;
    }
    return BigMatrix{n, n, std::move(data)};
}

auto BigMatrix::zero(std::size_t rows, std::size_t cols) -> BigMatrix {
    assert(fits_size(rows, cols) && "BigMatrix::zero dimension too large");
    return BigMatrix{rows, cols, std::vector<BigRational>(rows * cols)};
}

auto BigMatrix::from_matrix(const Matrix& a) -> Result<BigMatrix> {
    const std::size_t nrows = a.rows();
    const std::size_t ncols = a.cols();
    std::vector<BigRational> data;
    data.reserve(nrows * ncols);
    for (std::size_t i = 0; i < nrows; ++i) {
        for (std::size_t j = 0; j < ncols; ++j) {
            const Rational& e = a.at(i, j);
            // Rational is canonical (den >= 1), so make() cannot divide by zero here; the
            // Result is threaded through only for a uniform, defensive surface.
            auto promoted =
                BigRational::make(BigInt::from_i64(e.numerator()), BigInt::from_i64(e.denominator()));
            if (!promoted) {
                return make_error<BigMatrix>(promoted.error());
            }
            data.push_back(std::move(*promoted));
        }
    }
    return BigMatrix{nrows, ncols, std::move(data)};
}

// --- arithmetic -------------------------------------------------------------

auto BigMatrix::add(const BigMatrix& o) const -> Result<BigMatrix> {
    if (rows_ != o.rows_ || cols_ != o.cols_) {
        return make_error<BigMatrix>(MathError::domain_error);
    }
    std::vector<BigRational> data(data_.size());
    for (std::size_t i = 0; i < data_.size(); ++i) {
        data[i] = data_[i].add(o.data_[i]);  // infallible
    }
    return BigMatrix{rows_, cols_, std::move(data)};
}

auto BigMatrix::subtract(const BigMatrix& o) const -> Result<BigMatrix> {
    if (rows_ != o.rows_ || cols_ != o.cols_) {
        return make_error<BigMatrix>(MathError::domain_error);
    }
    std::vector<BigRational> data(data_.size());
    for (std::size_t i = 0; i < data_.size(); ++i) {
        data[i] = data_[i].subtract(o.data_[i]);  // infallible
    }
    return BigMatrix{rows_, cols_, std::move(data)};
}

auto BigMatrix::scale(const BigRational& s) const -> Result<BigMatrix> {
    std::vector<BigRational> data(data_.size());
    for (std::size_t i = 0; i < data_.size(); ++i) {
        data[i] = data_[i].multiply(s);  // infallible
    }
    return BigMatrix{rows_, cols_, std::move(data)};
}

auto BigMatrix::multiply(const BigMatrix& o) const -> Result<BigMatrix> {
    if (cols_ != o.rows_) {
        return make_error<BigMatrix>(MathError::domain_error);
    }
    const std::size_t n = o.cols_;
    const std::size_t k = cols_;  // == o.rows_
    std::vector<BigRational> data(rows_ * n);
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            BigRational acc;  // 0/1
            for (std::size_t p = 0; p < k; ++p) {
                acc = acc.add(data_[i * cols_ + p].multiply(o.data_[p * n + j]));  // infallible
            }
            data[i * n + j] = std::move(acc);
        }
    }
    return BigMatrix{rows_, n, std::move(data)};
}

auto BigMatrix::transpose() const -> Result<BigMatrix> {
    std::vector<BigRational> data(data_.size());
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            data[j * rows_ + i] = data_[i * cols_ + j];
        }
    }
    return BigMatrix{cols_, rows_, std::move(data)};
}

// --- determinant ------------------------------------------------------------

auto BigMatrix::determinant() const -> Result<BigRational> {
    if (!is_square()) {
        return make_error<BigRational>(MathError::domain_error);
    }
    const std::size_t n = rows_;
    if (n == 0) {
        return BigRational::from_int(1);  // empty product
    }
    // Fraction-free Bareiss elimination. `prev` is the pivot of the PREVIOUS step, seeded
    // with 1 to stand in for the notional M[-1][-1]; each interior entry is updated to
    //   M[i][j] <- (M[k][k] * M[i][j] - M[i][k] * M[k][j]) / prev
    // with exact division. `prev` is always a prior nonzero pivot (or the seed 1), so the
    // division never divides by zero; over Q it is exact regardless.
    std::vector<BigRational> m = data_;  // working copy
    BigRational prev = BigRational::from_int(1);
    int sign = 1;  // flipped by every row swap
    for (std::size_t k = 0; k < n; ++k) {
        // Ensure a nonzero pivot at (k, k), swapping up a lower row when needed.
        if (m[k * n + k].is_zero()) {
            std::size_t swap_row = k;
            bool found = false;
            for (std::size_t i = k + 1; i < n; ++i) {
                if (!m[i * n + k].is_zero()) {
                    swap_row = i;
                    found = true;
                    break;
                }
            }
            if (!found) {
                return BigRational{};  // an all-zero column below => singular => det 0
            }
            swap_rows(m, n, k, swap_row);
            sign = -sign;  // a row swap flips the determinant's sign
        }
        const BigRational pivot = m[k * n + k];
        for (std::size_t i = k + 1; i < n; ++i) {
            for (std::size_t j = k + 1; j < n; ++j) {
                // numerator = pivot*M[i][j] - M[i][k]*M[k][j]; both terms are infallible.
                const BigRational numerator =
                    pivot.multiply(m[i * n + j]).subtract(m[i * n + k].multiply(m[k * n + j]));
                auto quotient = numerator.divide(prev);  // exact; prev is a nonzero pivot
                if (!quotient) {
                    return make_error<BigRational>(quotient.error());
                }
                m[i * n + j] = std::move(*quotient);
            }
            m[i * n + k] = BigRational{};  // the column below the pivot is now cleared
        }
        prev = pivot;  // this pivot becomes the divisor for the next elimination step
    }
    // After the sweep the determinant's magnitude is the final (bottom-right) pivot.
    const BigRational& det = m[(n - 1) * n + (n - 1)];
    return sign < 0 ? det.negate() : det;
}

auto BigMatrix::to_string() const -> std::string {
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
