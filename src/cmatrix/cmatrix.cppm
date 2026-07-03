// NimbleCAS complex (Gaussian-rational) matrices (ROADMAP 7.2).
// @author Olumuyiwa Oluwasanmi
//
// Dense matrices whose entries are exact complex numbers over the rationals — the
// Gaussian rationals Q + Qi (see nimblecas.complex). Every arithmetic step flows through
// Complex's overflow-checked, railway-oriented operations, so the whole matrix layer is
// exact: a conjugate transpose, a unitarity check, or a product of two matrices is the
// number it mathematically is, never a floating-point approximation.
//
// The API mirrors the real Matrix (nimblecas.matrix) so downstream code reads the same
// way, and adds the complex-specific structure that ROADMAP 7.2 needs: conjugate,
// adjoint (Aᴴ, conjugate transpose), and the Hermitian / skew-Hermitian / unitary /
// normal predicates that classify a matrix's symmetry. These predicates are exactly the
// vocabulary the quantum-gate layer (ROADMAP 7.15) is built on: a gate is a unitary
// matrix, an observable is Hermitian, a generator of a rotation is skew-Hermitian.
//
// Following the rest of the engine (Rule 32): dimension violations surface as
// MathError::domain_error, entry-operation overflow surfaces as MathError::overflow, and
// the predicates that require a square matrix reject a non-square one with domain_error.
//
// Storage is dense row-major: entry (i, j) lives at data_[i * cols_ + j].

module;
#include <cassert>  // assert macro (unavailable via `import std`); active when !NDEBUG

export module nimblecas.cmatrix;

import std;
import nimblecas.core;
import nimblecas.ratpoly;   // Rational (real/imag parts of a Complex)
import nimblecas.complex;   // Complex — the entry type

export namespace nimblecas {

// ---------------------------------------------------------------------------
// ComplexMatrix — a dense rows_ x cols_ grid of exact Complex entries.
// ---------------------------------------------------------------------------
// The default-constructed matrix is the empty 0x0 matrix (a valid, if degenerate,
// value). Every fallible operation returns Result; the only infallible observers are the
// accessors below. The API deliberately tracks the real Matrix so the two feel uniform.
class ComplexMatrix {
public:
    ComplexMatrix() = default;  // the empty 0x0 matrix

    // Build from a list of rows. Every row must have the same length, otherwise the
    // matrix is ragged and construction fails with domain_error. An empty list is also
    // rejected with domain_error (a complex matrix always has at least one entry here).
    [[nodiscard]] static auto from_rows(std::vector<std::vector<Complex>> rows)
        -> Result<ComplexMatrix>;

    // The n x n identity (1 on the diagonal, 0 elsewhere). identity(0) is the 0x0 matrix.
    [[nodiscard]] static auto identity(std::size_t n) -> ComplexMatrix;

    // The rows x cols all-zero matrix (every entry 0 + 0i).
    [[nodiscard]] static auto zero(std::size_t rows, std::size_t cols) -> ComplexMatrix;

    [[nodiscard]] auto rows() const noexcept -> std::size_t { return rows_; }
    [[nodiscard]] auto cols() const noexcept -> std::size_t { return cols_; }
    [[nodiscard]] auto is_square() const noexcept -> bool { return rows_ == cols_; }

    // Entry (i, j). Asserted in-range: callers hold indices below rows()/cols().
    [[nodiscard]] auto at(std::size_t i, std::size_t j) const -> const Complex& {
        assert(i < rows_ && j < cols_ && "ComplexMatrix::at out of range");
        return data_[i * cols_ + j];
    }

    [[nodiscard]] auto operator==(const ComplexMatrix& o) const noexcept -> bool {
        return rows_ == o.rows_ && cols_ == o.cols_ && data_ == o.data_;
    }
    [[nodiscard]] auto is_equal(const ComplexMatrix& o) const noexcept -> bool {
        return *this == o;
    }

    // --- arithmetic (all Result; dimension / overflow errors propagate) ---

    // Entrywise sum / difference; dimensions must match, else domain_error.
    [[nodiscard]] auto add(const ComplexMatrix& o) const -> Result<ComplexMatrix>;
    [[nodiscard]] auto subtract(const ComplexMatrix& o) const -> Result<ComplexMatrix>;

    // Multiply every entry by a complex scalar.
    [[nodiscard]] auto scale(const Complex& s) const -> Result<ComplexMatrix>;

    // Matrix product (this is m x k, o is k x n). Fails with domain_error when the inner
    // dimensions disagree (cols() != o.rows()); propagates any Complex-op error.
    [[nodiscard]] auto multiply(const ComplexMatrix& o) const -> Result<ComplexMatrix>;

    // --- transposition / conjugation ---

    // The cols_ x rows_ plain transpose Aᵀ (no conjugation). Uniform Result surface.
    [[nodiscard]] auto transpose() const -> Result<ComplexMatrix>;

    // The entrywise complex conjugate Ā (same shape, no transpose).
    [[nodiscard]] auto conjugate() const -> Result<ComplexMatrix>;

    // The adjoint / conjugate transpose Aᴴ = (Ā)ᵀ — the key complex operation. This is
    // the map under which "Hermitian", "unitary", and "normal" are all defined.
    [[nodiscard]] auto adjoint() const -> Result<ComplexMatrix>;

    // --- structural predicates (require a square matrix, else domain_error) ---
    // Each compares exact Gaussian-rational entries, so the answer is a definite bool.

    // Hermitian: A == Aᴴ (self-adjoint). Real symmetric matrices are the real special case.
    [[nodiscard]] auto is_hermitian() const -> Result<bool>;

    // Skew-Hermitian: Aᴴ == −A. (i times a Hermitian matrix is skew-Hermitian.)
    [[nodiscard]] auto is_skew_hermitian() const -> Result<bool>;

    // Unitary: Aᴴ·A == I (columns orthonormal under the Hermitian inner product). These
    // are exactly the quantum gates of ROADMAP 7.15.
    [[nodiscard]] auto is_unitary() const -> Result<bool>;

    // Normal: A·Aᴴ == Aᴴ·A (commutes with its own adjoint). Hermitian, skew-Hermitian,
    // and unitary matrices are all normal.
    [[nodiscard]] auto is_normal() const -> Result<bool>;

    [[nodiscard]] auto to_string() const -> std::string;

private:
    ComplexMatrix(std::size_t rows, std::size_t cols, std::vector<Complex> data)
        : rows_(rows), cols_(cols), data_(std::move(data)) {}

    std::size_t rows_{0};
    std::size_t cols_{0};
    std::vector<Complex> data_{};  // row-major, size rows_ * cols_
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

}  // namespace

// --- construction -----------------------------------------------------------

auto ComplexMatrix::from_rows(std::vector<std::vector<Complex>> rows) -> Result<ComplexMatrix> {
    if (rows.empty()) {
        return make_error<ComplexMatrix>(MathError::domain_error);  // empty
    }
    const std::size_t nrows = rows.size();
    const std::size_t ncols = rows.front().size();
    if (ncols == 0) {
        return make_error<ComplexMatrix>(MathError::domain_error);  // rows with no columns
    }
    for (const auto& row : rows) {
        if (row.size() != ncols) {
            return make_error<ComplexMatrix>(MathError::domain_error);  // ragged
        }
    }
    std::vector<Complex> data;
    data.reserve(nrows * ncols);  // nrows*ncols already realised by the caller's rows
    for (auto& row : rows) {
        for (auto& e : row) {
            data.push_back(e);
        }
    }
    return ComplexMatrix{nrows, ncols, std::move(data)};
}

auto ComplexMatrix::identity(std::size_t n) -> ComplexMatrix {
    assert(fits_size(n, n) && "ComplexMatrix::identity dimension too large");
    std::vector<Complex> data(n * n);  // Complex{} default is 0 + 0i
    const Complex one = Complex::from_int(1);
    for (std::size_t i = 0; i < n; ++i) {
        data[i * n + i] = one;
    }
    return ComplexMatrix{n, n, std::move(data)};
}

auto ComplexMatrix::zero(std::size_t rows, std::size_t cols) -> ComplexMatrix {
    assert(fits_size(rows, cols) && "ComplexMatrix::zero dimension too large");
    return ComplexMatrix{rows, cols, std::vector<Complex>(rows * cols)};
}

// --- arithmetic -------------------------------------------------------------

auto ComplexMatrix::add(const ComplexMatrix& o) const -> Result<ComplexMatrix> {
    if (rows_ != o.rows_ || cols_ != o.cols_) {
        return make_error<ComplexMatrix>(MathError::domain_error);
    }
    std::vector<Complex> data(data_.size());
    for (std::size_t i = 0; i < data_.size(); ++i) {
        auto sum = data_[i].add(o.data_[i]);
        if (!sum) {
            return make_error<ComplexMatrix>(sum.error());
        }
        data[i] = *sum;
    }
    return ComplexMatrix{rows_, cols_, std::move(data)};
}

auto ComplexMatrix::subtract(const ComplexMatrix& o) const -> Result<ComplexMatrix> {
    if (rows_ != o.rows_ || cols_ != o.cols_) {
        return make_error<ComplexMatrix>(MathError::domain_error);
    }
    std::vector<Complex> data(data_.size());
    for (std::size_t i = 0; i < data_.size(); ++i) {
        auto diff = data_[i].subtract(o.data_[i]);
        if (!diff) {
            return make_error<ComplexMatrix>(diff.error());
        }
        data[i] = *diff;
    }
    return ComplexMatrix{rows_, cols_, std::move(data)};
}

auto ComplexMatrix::scale(const Complex& s) const -> Result<ComplexMatrix> {
    std::vector<Complex> data(data_.size());
    for (std::size_t i = 0; i < data_.size(); ++i) {
        auto p = data_[i].multiply(s);
        if (!p) {
            return make_error<ComplexMatrix>(p.error());
        }
        data[i] = *p;
    }
    return ComplexMatrix{rows_, cols_, std::move(data)};
}

auto ComplexMatrix::multiply(const ComplexMatrix& o) const -> Result<ComplexMatrix> {
    if (cols_ != o.rows_) {
        return make_error<ComplexMatrix>(MathError::domain_error);
    }
    const std::size_t n = o.cols_;
    const std::size_t k = cols_;  // == o.rows_
    std::vector<Complex> data(rows_ * n);
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            Complex acc;  // 0 + 0i
            for (std::size_t p = 0; p < k; ++p) {
                auto term = data_[i * cols_ + p].multiply(o.data_[p * n + j]);
                if (!term) {
                    return make_error<ComplexMatrix>(term.error());
                }
                auto sum = acc.add(*term);
                if (!sum) {
                    return make_error<ComplexMatrix>(sum.error());
                }
                acc = *sum;
            }
            data[i * n + j] = acc;
        }
    }
    return ComplexMatrix{rows_, n, std::move(data)};
}

// --- transposition / conjugation --------------------------------------------

auto ComplexMatrix::transpose() const -> Result<ComplexMatrix> {
    std::vector<Complex> data(data_.size());
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            data[j * rows_ + i] = data_[i * cols_ + j];
        }
    }
    return ComplexMatrix{cols_, rows_, std::move(data)};
}

auto ComplexMatrix::conjugate() const -> Result<ComplexMatrix> {
    std::vector<Complex> data(data_.size());
    for (std::size_t i = 0; i < data_.size(); ++i) {
        auto c = data_[i].conjugate();
        if (!c) {
            return make_error<ComplexMatrix>(c.error());
        }
        data[i] = *c;
    }
    return ComplexMatrix{rows_, cols_, std::move(data)};
}

auto ComplexMatrix::adjoint() const -> Result<ComplexMatrix> {
    // Aᴴ_(j,i) = conj(A_(i,j)): conjugate every entry while transposing its position.
    std::vector<Complex> data(data_.size());
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            auto c = data_[i * cols_ + j].conjugate();
            if (!c) {
                return make_error<ComplexMatrix>(c.error());
            }
            data[j * rows_ + i] = *c;
        }
    }
    return ComplexMatrix{cols_, rows_, std::move(data)};
}

// --- structural predicates --------------------------------------------------

auto ComplexMatrix::is_hermitian() const -> Result<bool> {
    if (!is_square()) {
        return make_error<bool>(MathError::domain_error);
    }
    auto adj = adjoint();  // square, so Aᴴ has this's shape
    if (!adj) {
        return make_error<bool>(adj.error());
    }
    return *this == *adj;  // A == Aᴴ
}

auto ComplexMatrix::is_skew_hermitian() const -> Result<bool> {
    if (!is_square()) {
        return make_error<bool>(MathError::domain_error);
    }
    auto adj = adjoint();
    if (!adj) {
        return make_error<bool>(adj.error());
    }
    // Aᴴ == −A. Form −A by scaling by (−1 + 0i), then compare.
    auto neg = scale(Complex::from_int(-1));
    if (!neg) {
        return make_error<bool>(neg.error());
    }
    return *adj == *neg;
}

auto ComplexMatrix::is_unitary() const -> Result<bool> {
    if (!is_square()) {
        return make_error<bool>(MathError::domain_error);
    }
    auto adj = adjoint();
    if (!adj) {
        return make_error<bool>(adj.error());
    }
    auto prod = adj->multiply(*this);  // Aᴴ·A
    if (!prod) {
        return make_error<bool>(prod.error());
    }
    return *prod == identity(rows_);  // Aᴴ·A == I
}

auto ComplexMatrix::is_normal() const -> Result<bool> {
    if (!is_square()) {
        return make_error<bool>(MathError::domain_error);
    }
    auto adj = adjoint();
    if (!adj) {
        return make_error<bool>(adj.error());
    }
    auto aah = multiply(*adj);  // A·Aᴴ
    if (!aah) {
        return make_error<bool>(aah.error());
    }
    auto aha = adj->multiply(*this);  // Aᴴ·A
    if (!aha) {
        return make_error<bool>(aha.error());
    }
    return *aah == *aha;
}

auto ComplexMatrix::to_string() const -> std::string {
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
