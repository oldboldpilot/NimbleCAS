// NimbleCAS exact banded & tridiagonal solvers (ROADMAP 7.2).
// @author Olumuyiwa Oluwasanmi
//
// Exact linear-system solvers for tridiagonal and general banded operators over the field
// Q. Every pivot, multiplier, elimination step, and back-substitution is a Rational
// operation, so the returned x satisfies A*x = b IDENTICALLY (not "up to rounding") — there
// is no floating point and no tolerance anywhere in this module. Failure is reported only on
// the railway (Result<T> / MathError); nothing throws.
//
// Three entry points:
//   * solve_tridiagonal   — the Thomas algorithm (forward elimination + back-substitution)
//                           for a single right-hand side.
//   * solve_banded        — band-restricted LU (Doolittle, no pivoting) that only touches
//                           entries inside the (lower, upper) band.
//   * solve_tridiagonal_batch — the SAME tridiagonal operator against many right-hand-side
//                           columns at once, parallelised over columns.
//
// PIVOTING: these solvers are partial-pivoting-free. Thomas and band-LU both assume a
// nonzero pivot at each step (the classic diagonal-dominance precondition). A zero pivot —
// either an original zero diagonal or one produced by exact elimination — is reported as
// MathError::domain_error ("singular, or needs a permutation we do not perform") rather than
// worked around, so a returned solution is always genuine.
//
// PARALLELISM & HONESTY: the Thomas recurrence is inherently O(n) SEQUENTIAL — each
// eliminated pivot feeds the next, so a single right-hand side cannot be sped up by threads
// here. The parallel win is the multi-RHS batch: distinct columns share one forward
// factorisation of the operator and are then solved completely independently, which is
// embarrassingly parallel. solve_tridiagonal_batch fans the columns out with
// nimblecas::parallel::transform_index; because every column is a pure function of the shared
// (immutable) factorisation and its own data, the result is bit-identical regardless of
// thread count or partitioning. This many-RHS shape is also exactly what a GPU backend would
// offload (one thread/warp per column); that offload is noted here but deliberately NOT
// implemented — there is no CUDA in this module.
//
// An exact parallel single-RHS cyclic-reduction solver is intentionally NOT provided. It is
// expressible over Q, but the reduction combines rows into new rational coefficients whose
// numerators/denominators grow with each level, and reproducing a bit-identical exact result
// across arbitrary thread partitions would require care we have not validated. Since the
// batch path already delivers the parallel speedup for the common many-RHS workload, the
// single-RHS path stays the plain sequential Thomas recurrence.

export module nimblecas.bandsolve;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.parallel;

export namespace nimblecas {

// Solve a tridiagonal system A*x = rhs by the Thomas algorithm, exactly over Q.
//
// For an n x n tridiagonal A: `diag` holds the n diagonal entries b_1..b_n, `sub` holds the
// n-1 subdiagonal entries a_2..a_n (a_i = A(i, i-1)), `super` holds the n-1 superdiagonal
// entries c_1..c_{n-1} (c_i = A(i, i+1)), and `rhs` holds the n right-hand-side entries.
// n is taken from diag.size(); if it is 0, or sub/super do not have exactly n-1 entries, or
// rhs does not have n entries, the call fails with domain_error.
//
// Forward elimination then back-substitution are carried out in exact Rational arithmetic.
// The algorithm assumes diagonal dominance (no pivoting): if the original leading diagonal
// entry or any pivot produced by elimination is zero the system is singular or needs
// pivoting we do not do, and the call fails with domain_error. Any Rational-op overflow is
// propagated as MathError::overflow.
[[nodiscard]] auto solve_tridiagonal(std::span<const Rational> sub, std::span<const Rational> diag,
                                     std::span<const Rational> super, std::span<const Rational> rhs)
    -> Result<std::vector<Rational>>;

// Solve a banded system A*x = b with band-restricted LU (Doolittle elimination without
// pivoting), exactly over Q.
//
// `a` must be square (n x n) and `b` an n x 1 column, else domain_error. `lower_bw` and
// `upper_bw` are the sub- and super-diagonal bandwidths: entries of `a` outside the band
// (|i - j| beyond the respective bandwidth) are assumed zero and never touched, so the
// elimination stays inside the band and no fill-in escapes it. Like Thomas this is
// pivoting-free and assumes nonzero pivots; a zero pivot (original or produced by
// elimination) yields domain_error. Returns the n x 1 solution column. Rational-op overflow
// propagates as overflow.
[[nodiscard]] auto solve_banded(const Matrix& a, std::size_t lower_bw, std::size_t upper_bw,
                                const Matrix& b) -> Result<Matrix>;

// Solve the SAME tridiagonal operator against many right-hand-side columns at once.
//
// The operator is given by (sub, diag, super) exactly as in solve_tridiagonal; `rhs_columns`
// is an n x k matrix whose k columns are the independent right-hand sides. Returns the n x k
// solution matrix whose column j solves A*x = rhs_columns(:, j). Size validation matches
// solve_tridiagonal, with the added requirement that rhs_columns has exactly n rows (else
// domain_error).
//
// The forward factorisation of A is computed ONCE and shared; the k columns are then solved
// independently and in parallel over columns via nimblecas::parallel::transform_index. Each
// column is a pure function of the shared factorisation and its own data, so the result is
// identical regardless of thread count. A singular operator (zero pivot) fails with
// domain_error; per-column Rational-op overflow propagates as overflow.
[[nodiscard]] auto solve_tridiagonal_batch(std::span<const Rational> sub,
                                           std::span<const Rational> diag,
                                           std::span<const Rational> super,
                                           const Matrix& rhs_columns) -> Result<Matrix>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// c - a*b, exactly, propagating any Rational-op error. The recurring shape in both the
// forward sweeps (pivot update, transformed RHS) and the back-substitutions.
[[nodiscard]] auto sub_prod(const Rational& c, const Rational& a, const Rational& b)
    -> Result<Rational> {
    auto prod = a.multiply(b);
    if (!prod) {
        return make_error<Rational>(prod.error());
    }
    return c.subtract(*prod);
}

// Forward factorisation of a tridiagonal operator, independent of any right-hand side.
// `pivot[i]` is the modified diagonal m_i produced by elimination (pivot[0] == diag[0]);
// `cprime[i]` is the eliminated superdiagonal c'_i = super[i] / pivot[i] for i in [0, n-1).
// `sub` is a copy of the subdiagonal, needed by the per-column RHS recurrence. Every pivot
// is guaranteed nonzero once this struct is built.
struct TridiagFactor {
    std::size_t n;
    std::vector<Rational> pivot;   // n entries (m_0..m_{n-1})
    std::vector<Rational> cprime;  // n-1 entries (c'_0..c'_{n-2})
    std::vector<Rational> sub;     // n-1 entries (a_2..a_n)
};

// Eliminate the operator once. Fails with domain_error on a zero pivot (singular / needs
// pivoting), or propagates a Rational-op error. Sizes are assumed already validated.
[[nodiscard]] auto factor_tridiagonal(std::span<const Rational> sub, std::span<const Rational> diag,
                                      std::span<const Rational> super) -> Result<TridiagFactor> {
    const std::size_t n = diag.size();
    TridiagFactor f;
    f.n = n;
    f.pivot.resize(n);
    f.cprime.resize(n == 0 ? 0 : n - 1);
    f.sub.assign(sub.begin(), sub.end());

    // Leading pivot is the raw diagonal; a zero here means A(0,0) == 0 with no row to swap in.
    if (diag[0].is_zero()) {
        return make_error<TridiagFactor>(MathError::domain_error);
    }
    f.pivot[0] = diag[0];
    if (n > 1) {
        auto c0 = super[0].divide(f.pivot[0]);
        if (!c0) {
            return make_error<TridiagFactor>(c0.error());
        }
        f.cprime[0] = *c0;
    }

    for (std::size_t i = 1; i < n; ++i) {
        // m_i = diag[i] - a_i * c'_{i-1}  (a_i == sub[i-1]).
        auto m = sub_prod(diag[i], f.sub[i - 1], f.cprime[i - 1]);
        if (!m) {
            return make_error<TridiagFactor>(m.error());
        }
        if (m->is_zero()) {
            return make_error<TridiagFactor>(MathError::domain_error);  // singular / needs pivoting
        }
        f.pivot[i] = *m;
        if (i + 1 < n) {
            auto c = super[i].divide(f.pivot[i]);
            if (!c) {
                return make_error<TridiagFactor>(c.error());
            }
            f.cprime[i] = *c;
        }
    }
    return f;
}

// Solve the factored operator against one right-hand side. Pure in (f, rhs): reads only the
// shared factorisation and its own rhs and returns a fresh vector, so it is safe to invoke
// concurrently for distinct columns. `rhs` must have f.n entries (caller guarantees).
[[nodiscard]] auto solve_with_factor(const TridiagFactor& f, std::span<const Rational> rhs)
    -> Result<std::vector<Rational>> {
    const std::size_t n = f.n;

    // Forward: transformed RHS d'_i. d'_0 = rhs[0]/m_0; d'_i = (rhs[i] - a_i*d'_{i-1})/m_i.
    std::vector<Rational> dprime(n);
    auto d0 = rhs[0].divide(f.pivot[0]);
    if (!d0) {
        return make_error<std::vector<Rational>>(d0.error());
    }
    dprime[0] = *d0;
    for (std::size_t i = 1; i < n; ++i) {
        auto num = sub_prod(rhs[i], f.sub[i - 1], dprime[i - 1]);
        if (!num) {
            return make_error<std::vector<Rational>>(num.error());
        }
        auto d = num->divide(f.pivot[i]);
        if (!d) {
            return make_error<std::vector<Rational>>(d.error());
        }
        dprime[i] = *d;
    }

    // Back-substitution: x_{n-1} = d'_{n-1}; x_i = d'_i - c'_i * x_{i+1}.
    std::vector<Rational> x(n);
    x[n - 1] = dprime[n - 1];
    for (std::size_t i = n - 1; i-- > 0;) {
        auto xi = sub_prod(dprime[i], f.cprime[i], x[i + 1]);
        if (!xi) {
            return make_error<std::vector<Rational>>(xi.error());
        }
        x[i] = *xi;
    }
    return x;
}

// Do the tridiagonal argument spans describe a consistent n x n system?
[[nodiscard]] auto tridiag_sizes_ok(std::span<const Rational> sub, std::span<const Rational> diag,
                                    std::span<const Rational> super, std::size_t rhs_rows) -> bool {
    const std::size_t n = diag.size();
    if (n == 0) {
        return false;
    }
    return sub.size() == n - 1 && super.size() == n - 1 && rhs_rows == n;
}

}  // namespace

// --- single-RHS tridiagonal (Thomas) ---------------------------------------

auto solve_tridiagonal(std::span<const Rational> sub, std::span<const Rational> diag,
                       std::span<const Rational> super, std::span<const Rational> rhs)
    -> Result<std::vector<Rational>> {
    if (!tridiag_sizes_ok(sub, diag, super, rhs.size())) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    auto factor = factor_tridiagonal(sub, diag, super);
    if (!factor) {
        return make_error<std::vector<Rational>>(factor.error());
    }
    return solve_with_factor(*factor, rhs);
}

// --- banded LU (Doolittle, no pivoting, band-restricted) --------------------

auto solve_banded(const Matrix& a, std::size_t lower_bw, std::size_t upper_bw, const Matrix& b)
    -> Result<Matrix> {
    if (!a.is_square() || b.rows() != a.rows() || b.cols() != 1) {
        return make_error<Matrix>(MathError::domain_error);
    }
    const std::size_t n = a.rows();

    // Working copy of A (row-major) and of the right-hand side; both mutate during
    // elimination. Only band entries are ever read/written, so out-of-band entries of `a`
    // are treated as the zeros the band declares them to be.
    std::vector<Rational> m(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            m[i * n + j] = a.at(i, j);
        }
    }
    std::vector<Rational> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = b.at(i, 0);
    }

    // Forward elimination. Row k eliminates rows k+1..k+lower_bw; within each such row only
    // columns k+1..k+upper_bw are touched (row k is nonzero only there), so no fill-in
    // leaves the band.
    for (std::size_t k = 0; k < n; ++k) {
        const Rational pivot = m[k * n + k];
        if (pivot.is_zero()) {
            return make_error<Matrix>(MathError::domain_error);  // singular / needs pivoting
        }
        const std::size_t last_row = std::min(k + lower_bw, n - 1);
        const std::size_t last_col = std::min(k + upper_bw, n - 1);
        for (std::size_t i = k + 1; i <= last_row; ++i) {
            auto factor = m[i * n + k].divide(pivot);
            if (!factor) {
                return make_error<Matrix>(factor.error());
            }
            for (std::size_t j = k + 1; j <= last_col; ++j) {
                auto updated = sub_prod(m[i * n + j], *factor, m[k * n + j]);
                if (!updated) {
                    return make_error<Matrix>(updated.error());
                }
                m[i * n + j] = *updated;
            }
            m[i * n + k] = Rational::from_int(0);  // eliminated below the pivot
            auto rhs_i = sub_prod(x[i], *factor, x[k]);
            if (!rhs_i) {
                return make_error<Matrix>(rhs_i.error());
            }
            x[i] = *rhs_i;
        }
    }

    // Back-substitution over the upper band: x_i = (x_i - sum_{j>i} U(i,j) x_j) / U(i,i),
    // with j running only up to i + upper_bw (U keeps A's upper bandwidth).
    for (std::size_t i = n; i-- > 0;) {
        Rational sum = x[i];
        const std::size_t last_col = std::min(i + upper_bw, n - 1);
        for (std::size_t j = i + 1; j <= last_col; ++j) {
            auto next = sub_prod(sum, m[i * n + j], x[j]);
            if (!next) {
                return make_error<Matrix>(next.error());
            }
            sum = *next;
        }
        auto xi = sum.divide(m[i * n + i]);
        if (!xi) {
            return make_error<Matrix>(xi.error());
        }
        x[i] = *xi;
    }

    // Package the solution as an n x 1 column.
    std::vector<std::vector<Rational>> rows(n, std::vector<Rational>(1));
    for (std::size_t i = 0; i < n; ++i) {
        rows[i][0] = x[i];
    }
    return Matrix::from_rows(std::move(rows));
}

// --- batched multi-RHS tridiagonal (parallel over columns) ------------------

auto solve_tridiagonal_batch(std::span<const Rational> sub, std::span<const Rational> diag,
                             std::span<const Rational> super, const Matrix& rhs_columns)
    -> Result<Matrix> {
    if (!tridiag_sizes_ok(sub, diag, super, rhs_columns.rows())) {
        return make_error<Matrix>(MathError::domain_error);
    }
    const std::size_t n = diag.size();
    const std::size_t k = rhs_columns.cols();

    // Factor the shared operator once; every column reuses it read-only.
    auto factor = factor_tridiagonal(sub, diag, super);
    if (!factor) {
        return make_error<Matrix>(factor.error());
    }

    // Solve the k columns independently and in parallel. Each task extracts its own column
    // and calls the pure solve_with_factor, so the vector of results depends only on the data
    // — not on how the column range was partitioned across threads (grain 1: one task per
    // column, the backend auto-chunks). transform_index preserves index order.
    auto column_results = parallel::transform_index(
        k,
        [&](std::size_t j) -> Result<std::vector<Rational>> {
            std::vector<Rational> rhs(n);
            for (std::size_t i = 0; i < n; ++i) {
                rhs[i] = rhs_columns.at(i, j);
            }
            return solve_with_factor(*factor, rhs);
        },
        std::size_t{1});

    // Surface the first per-column failure (e.g. overflow) on the railway.
    for (const auto& col : column_results) {
        if (!col) {
            return make_error<Matrix>(col.error());
        }
    }

    // Assemble the n x k solution: row i gathers entry i of every column solution.
    std::vector<std::vector<Rational>> rows(n, std::vector<Rational>(k));
    for (std::size_t j = 0; j < k; ++j) {
        const std::vector<Rational>& sol = *column_results[j];
        for (std::size_t i = 0; i < n; ++i) {
            rows[i][j] = sol[i];
        }
    }
    return Matrix::from_rows(std::move(rows));
}

}  // namespace nimblecas
