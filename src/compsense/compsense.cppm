// NimbleCAS compressed sensing / sparse recovery (ROADMAP §7 algorithmics layer).
// @author Olumuyiwa Oluwasanmi
//
// Conforms to config/cpp_details.txt: C++23 modules, `import std`, trailing return
// types, no owning raw pointers, std::expected error handling (no exceptions),
// [[nodiscard]] everywhere, and — above all — an HONEST exact-vs-numerical boundary.
//
// ---------------------------------------------------------------------------
// The honesty boundary (documented and true)
// ---------------------------------------------------------------------------
// * BASIS PURSUIT is EXACT over Q. It solves  min ||x||_1  s.t.  A x = b  as a
//   linear program and returns the exact rational minimiser. There is no rounding:
//   an answer reported as p/q is exactly p/q. This is a genuine exact-rational L1
//   sparse recovery — exact whenever the underlying LP has a rational optimum
//   (which it always does when A, b are rational and the program is feasible and
//   bounded, i.e. b in range(A)).
//
//   Mechanics: with x split into x = x+ - x- (x+, x- >= 0) the program becomes
//       min 1.(x+ + x-)   s.t.  [A | -A] (x+; x-) = b,   (x+; x-) >= 0,
//   a standard-form equality LP. nimblecas.lp's Simplex (maximize) solves only
//   INEQUALITY programs A z <= b with b >= 0 (a single-phase, origin-feasible
//   Simplex — no Phase I), so it cannot take an equality/RHS-signed program
//   directly. We therefore feed nimblecas.lp the exact DUAL of the split program,
//       max b.y   s.t.  |a_j . y| <= 1  for every atom a_j,   y free,
//   which is exactly an inequality LP with an all-ones (hence >= 0) right-hand
//   side — a perfect fit for maximize. Strong duality gives the optimal L1 value,
//   and the optimal y* pins the support via complementary slackness: x_j can be
//   nonzero only where |a_j . y*| = 1 (an EXACT rational equality test). The
//   values on that support are then recovered exactly by an exact-rational normal-
//   equation solve through nimblecas.matrix. Everything is over Q; nothing rounds.
//
//   Honest caveat: the support read-off is exact under the standard BASIS-PURSUIT
//   non-degeneracy / uniqueness regime (the regime in which L1 recovery is meant to
//   be used). Under LP degeneracy the tight set can be a superset of the true
//   support; the normal-equation solve still returns the exact minimiser as long as
//   the tight columns are linearly independent, and otherwise reports domain_error
//   rather than guessing.
//
// * ORTHOGONAL MATCHING PURSUIT, CoSaMP and ITERATIVE HARD THRESHOLDING are
//   NUMERICAL (double precision), greedy / thresholding heuristics. They carry NO
//   global-optimality guarantee: they recover the sparse signal only under
//   sparsity / RIP / incoherence conditions, and can fail silently outside them.
//   They are fast and practical, not exact.
//
// * MUTUAL COHERENCE is provided EXACTLY in its SQUARED form over Q — we compare
//   <a_i,a_j>^2 / (||a_i||^2 ||a_j||^2) to sidestep the sqrt, so the diagnostic is
//   an exact rational — and NUMERICALLY as the plain coherence (a double sqrt). The
//   companion recovery bound  k < (1 + 1/mu)/2  is only a SUFFICIENT condition:
//   satisfying it guarantees OMP / basis-pursuit recovery of every k-sparse signal;
//   violating it does NOT imply failure. Recovery is never universal.
//
// ---------------------------------------------------------------------------
// PARALLEL + DISTRIBUTED batch recovery (acceleration only; see below)
// ---------------------------------------------------------------------------
// A single measurement matrix (dictionary) A is frequently reused to recover MANY
// signals from MANY measurement vectors b_1..b_K. Each recovery x_k = f(A, b_k) is
// INDEPENDENT of the others (it reads only the shared, immutable A and its own b_k
// and writes only its own output slot), so the batch is an EMBARRASSINGLY PARALLEL,
// order-preserving map k -> f(A, b_k). We express it over nimblecas.parallel's
// deterministic fork-join runtime (TBB on Linux/macOS, PPL on Windows, serial
// fallback otherwise).
//
// HONESTY (documented and true):
//   * The parallel/distributed batch is DETERMINISTIC and BIT-IDENTICAL to recovering
//     each signal serially, REGARDLESS of thread count, backend, or shard split. This
//     is guaranteed structurally: recoveries share no mutable state, results are
//     written to per-index slots (never reduced with a non-associative combine), and
//     the map is order-preserving (result[k] depends only on f and b_k, never on
//     scheduling). Running on 1 thread, 64 threads, or 8 machines yields the same
//     vectors in the same order.
//   * Consequently the acceleration is WALL-CLOCK ONLY. It changes NOTHING about the
//     numerics or the guarantees: basis_pursuit stays EXACT over Q for every column;
//     OMP stays NUMERICAL (double, greedy) for every column. Recovery of any given
//     b_k still holds only under that method's sparsity / coherence / RIP regime —
//     parallelism neither strengthens nor weakens per-signal recovery.
//   * The DISTRIBUTED entrypoints (recover_batch_shard / basis_pursuit_batch_shard)
//     are STATELESS PURE functions: shard s independently recovers exactly the columns
//     i with i % num_shards == s, in ascending index order, touching no other shard's
//     data. concat_shards deterministically reassembles the shards back into original
//     input order, so the sharded result equals the whole-batch result equals the
//     serial result. A driver (SGE / Ray / MPI) may run shards on separate machines.

export module nimblecas.compsense;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.lp;
import nimblecas.parallel;

export namespace nimblecas {

// ===========================================================================
// EXACT-RATIONAL sparse recovery.
// ===========================================================================

// Basis pursuit: the exact minimiser of ||x||_1 subject to A x = b, over Q.
//
// A is an m x n rational matrix given as m rows of width n (the n atoms are its
// columns); b has length m. Returns the length-n exact rational minimiser x.
//
// Errors (MathError):
//   * domain_error   — a ragged A, an empty A / zero-width rows, a b whose length
//                      disagrees with A, an INFEASIBLE system (b not in range(A)),
//                      or a degenerate optimum whose tight columns are dependent so
//                      the exact support cannot be uniquely recovered.
//   * overflow       — an int64 numerator/denominator overflow in the exact tableau
//                      or the recovery solve (propagated from Rational / lp / matrix).
[[nodiscard]] auto basis_pursuit(const std::vector<std::vector<Rational>>& A,
                                 const std::vector<Rational>& b)
    -> Result<std::vector<Rational>>;

// The EXACT squared mutual coherence of a dictionary whose atoms are the columns of
// A (m rows x n columns):  max_{i != j} <a_i,a_j>^2 / (||a_i||^2 ||a_j||^2).
// Squaring avoids the sqrt so the result is an exact rational in [0, 1].
//
// Errors: domain_error for a ragged A, fewer than two atoms, or a zero-norm atom
// (its coherence is undefined); overflow from the exact inner products.
[[nodiscard]] auto mutual_coherence_squared(const std::vector<std::vector<Rational>>& A)
    -> Result<Rational>;

// The EXACT sufficient recovery test  k < (1 + 1/mu)/2, evaluated without a sqrt.
// For k >= 1 and mu > 0 the bound is equivalent to  mu^2 (2k - 1)^2 < 1, an exact
// rational comparison taking the squared coherence directly. Returns true when the
// (sufficient, NOT necessary) guarantee holds. domain_error for k == 0.
[[nodiscard]] auto coherence_guarantees_recovery(const Rational& mu_squared, std::size_t k)
    -> Result<bool>;

// ===========================================================================
// NUMERICAL (double) sparse recovery. Greedy / thresholding heuristics; no global
// optimality — correct only under sparsity / RIP / incoherence conditions.
// ===========================================================================
// Every measurement matrix is passed as a row-major std::span<const double> of
// rows*cols entries: entry (i, j) lives at data[i * cols + j]; its n atoms are the
// columns. Domain errors (shape mismatch) surface as MathError::domain_error.

// Orthogonal Matching Pursuit. Greedily selects, at each step, the atom most
// correlated with the current residual, re-solves the least squares over the active
// set, and updates the residual. Stops at `sparsity` atoms or when the residual norm
// falls to <= tol. Returns the length-cols coefficient vector.
[[nodiscard]] auto orthogonal_matching_pursuit(std::span<const double> A, std::size_t rows,
                                               std::size_t cols, std::span<const double> b,
                                               std::size_t sparsity, double tol = 1e-9)
    -> Result<std::vector<double>>;

// CoSaMP (Compressive Sampling Matching Pursuit). Each iteration forms the signal
// proxy A^T r, merges its 2k largest-magnitude coordinates with the current support,
// least-squares-fits on that enlarged set, then prunes back to the k largest. Stops
// on a small residual or after max_iter iterations.
[[nodiscard]] auto cosamp(std::span<const double> A, std::size_t rows, std::size_t cols,
                          std::span<const double> b, std::size_t sparsity, int max_iter = 50,
                          double tol = 1e-9) -> Result<std::vector<double>>;

// Iterative Hard Thresholding:  x <- H_k( x + step * A^T (b - A x) ), where H_k keeps
// the k largest-magnitude coordinates and zeroes the rest. Stops when the update is
// smaller than tol or after max_iter iterations.
[[nodiscard]] auto iterative_hard_thresholding(std::span<const double> A, std::size_t rows,
                                               std::size_t cols, std::span<const double> b,
                                               std::size_t sparsity, double step = 1.0,
                                               int max_iter = 500, double tol = 1e-9)
    -> Result<std::vector<double>>;

// The NUMERICAL mutual coherence of the column-atom dictionary A (m x n): the plain
// max_{i != j} |<a_i,a_j>| / (||a_i|| ||a_j||) as a double. domain_error for a ragged
// shape, fewer than two atoms, or a zero-norm atom.
[[nodiscard]] auto mutual_coherence(std::span<const double> A, std::size_t rows,
                                    std::size_t cols) -> Result<double>;

// ===========================================================================
// PARALLEL batch recovery (acceleration only; DETERMINISTIC == serial).
// ===========================================================================
// A batch is a range of measurement vectors b_0..b_{K-1} sharing ONE dictionary A.
// Each returns its own independent recovery, so the batch return type is a per-signal
// vector of Results (one signal's failure never poisons the others) in INPUT ORDER.
// `grain` is the nimblecas.parallel cutoff: below `grain` measurements the map runs
// serially. It defaults to 1 because each recovery is coarse-grained work, so one task
// per signal (with the backend auto-chunking) is the right granularity — but ANY grain
// yields identical results, differing only in wall-clock time.

// NUMERICAL OMP over a whole batch. Recovers measurements[k] with
// orthogonal_matching_pursuit(A, rows, cols, measurements[k], sparsity, tol) for every
// k, in parallel, returning the recoveries in input order. Deterministic: element k is
// bit-identical to calling orthogonal_matching_pursuit serially on measurements[k].
[[nodiscard]] auto parallel_omp_batch(std::span<const double> A, std::size_t rows,
                                      std::size_t cols,
                                      std::span<const std::vector<double>> measurements,
                                      std::size_t sparsity, double tol = 1e-9,
                                      std::size_t grain = 1)
    -> std::vector<Result<std::vector<double>>>;

// EXACT (over Q) basis pursuit over a whole batch. Recovers measurements[k] with
// basis_pursuit(A, measurements[k]) for every k, in parallel, in input order.
// Deterministic: element k equals basis_pursuit(A, measurements[k]) run serially.
[[nodiscard]] auto parallel_basis_pursuit_batch(
    const std::vector<std::vector<Rational>>& A,
    std::span<const std::vector<Rational>> measurements, std::size_t grain = 1)
    -> std::vector<Result<std::vector<Rational>>>;

// ===========================================================================
// DISTRIBUTED shard entrypoints (stateless pure; a driver concatenates by index).
// ===========================================================================
// Shard `shard_index` (of `num_shards`) recovers EXACTLY the measurement vectors whose
// original index i satisfies  i % num_shards == shard_index, returned in ASCENDING
// original-index order. Pure and stateless: it reads only A and its assigned columns,
// so shards may run on different threads, processes, or machines. domain_error if
// num_shards == 0 or shard_index >= num_shards. Inside the shard the assigned
// recoveries themselves run in parallel (deterministically).

[[nodiscard]] auto recover_batch_shard(std::span<const double> A, std::size_t rows,
                                       std::size_t cols,
                                       std::span<const std::vector<double>> measurements,
                                       std::size_t shard_index, std::size_t num_shards,
                                       std::size_t sparsity, double tol = 1e-9,
                                       std::size_t grain = 1)
    -> Result<std::vector<Result<std::vector<double>>>>;

[[nodiscard]] auto basis_pursuit_batch_shard(
    const std::vector<std::vector<Rational>>& A,
    std::span<const std::vector<Rational>> measurements, std::size_t shard_index,
    std::size_t num_shards, std::size_t grain = 1)
    -> Result<std::vector<Result<std::vector<Rational>>>>;

// Reassemble `num_shards` shard outputs (as produced by *_batch_shard, i.e. shard s is
// the ascending subsequence of indices i with i % num_shards == s) back into the full
// `total`-length batch in ORIGINAL input order: out[i] = shards[i % num_shards] at its
// (i / num_shards)-th position. This is the DRIVER-side reduction; it is a pure,
// deterministic gather with NO recomputation, so out == the whole-batch result ==
// the serial result. domain_error if num_shards == 0, shards.size() != num_shards, or
// any shard's length disagrees with the count of indices assigned to it.
template <typename T>
[[nodiscard]] auto concat_shards(std::size_t total, std::size_t num_shards,
                                 std::vector<std::vector<Result<std::vector<T>>>> shards)
    -> Result<std::vector<Result<std::vector<T>>>> {
    using Batch = std::vector<Result<std::vector<T>>>;
    if (num_shards == 0 || shards.size() != num_shards) {
        return make_error<Batch>(MathError::domain_error);
    }
    // Each shard must hold exactly its assigned count: |{ i < total : i % num_shards == s }|.
    for (std::size_t s = 0; s < num_shards; ++s) {
        std::size_t assigned = 0;
        for (std::size_t i = s; i < total; i += num_shards) {
            ++assigned;
        }
        if (shards[s].size() != assigned) {
            return make_error<Batch>(MathError::domain_error);
        }
    }
    Batch out;
    out.reserve(total);
    std::vector<std::size_t> cursor(num_shards, 0);  // next unread position per shard
    for (std::size_t i = 0; i < total; ++i) {
        const std::size_t s = i % num_shards;
        out.push_back(std::move(shards[s][cursor[s]]));
        ++cursor[s];
    }
    return out;
}

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// --- exact-rational helpers ------------------------------------------------------

// A canonical Rational has den > 0, so the whole fraction's sign is its numerator's.
[[nodiscard]] auto is_negative(const Rational& r) noexcept -> bool { return r.numerator() < 0; }

// Three-way exact compare of a and b (both denominators positive): returns -1, 0, +1.
// Cross-multiplication is overflow-checked (Rule 32) and surfaces MathError::overflow.
[[nodiscard]] auto compare(const Rational& a, const Rational& b) -> Result<int> {
    std::int64_t lhs = 0;
    std::int64_t rhs = 0;
    if (__builtin_mul_overflow(a.numerator(), b.denominator(), &lhs) ||
        __builtin_mul_overflow(b.numerator(), a.denominator(), &rhs)) {
        return make_error<int>(MathError::overflow);
    }
    if (lhs < rhs) {
        return -1;
    }
    if (lhs > rhs) {
        return 1;
    }
    return 0;
}

// The exact inner product of columns i and j of a rational matrix stored as rows.
[[nodiscard]] auto column_dot(const std::vector<std::vector<Rational>>& A, std::size_t i,
                              std::size_t j) -> Result<Rational> {
    Rational acc;  // 0/1
    for (const auto& row : A) {
        auto term = row[i].multiply(row[j]);
        if (!term) {
            return make_error<Rational>(term.error());
        }
        auto sum = acc.add(*term);
        if (!sum) {
            return make_error<Rational>(sum.error());
        }
        acc = *sum;
    }
    return acc;
}

// --- numeric (double) helpers ----------------------------------------------------

// Column j of a row-major rows x cols matrix dotted with a length-rows vector.
[[nodiscard]] auto col_dot_vec(std::span<const double> A, std::size_t rows, std::size_t cols,
                               std::size_t j, std::span<const double> v) noexcept -> double {
    double acc = 0.0;
    for (std::size_t i = 0; i < rows; ++i) {
        acc = std::fma(A[i * cols + j], v[i], acc);
    }
    return acc;
}

// Euclidean norm of a vector.
[[nodiscard]] auto norm2(std::span<const double> v) noexcept -> double {
    double acc = 0.0;
    for (double e : v) {
        acc = std::fma(e, e, acc);
    }
    return std::sqrt(acc);
}

// A x for a row-major rows x cols matrix and a length-cols vector x.
[[nodiscard]] auto matvec(std::span<const double> A, std::size_t rows, std::size_t cols,
                          std::span<const double> x) -> std::vector<double> {
    std::vector<double> out(rows, 0.0);
    for (std::size_t i = 0; i < rows; ++i) {
        double acc = 0.0;
        for (std::size_t j = 0; j < cols; ++j) {
            acc = std::fma(A[i * cols + j], x[j], acc);
        }
        out[i] = acc;
    }
    return out;
}

// b - A x (the residual of a candidate solution x).
[[nodiscard]] auto residual(std::span<const double> A, std::size_t rows, std::size_t cols,
                            std::span<const double> b, std::span<const double> x)
    -> std::vector<double> {
    std::vector<double> r = matvec(A, rows, cols, x);
    for (std::size_t i = 0; i < rows; ++i) {
        r[i] = b[i] - r[i];
    }
    return r;
}

// Solve the k x k linear system G g = d in place by Gaussian elimination with partial
// pivoting. Returns std::nullopt if the (numerical) pivot collapses — i.e. G is
// singular to working precision. G is row-major k x k; d has length k.
[[nodiscard]] auto solve_dense(std::vector<double> G, std::vector<double> d, std::size_t k)
    -> std::optional<std::vector<double>> {
    for (std::size_t col = 0; col < k; ++col) {
        std::size_t pivot = col;
        double best = std::abs(G[col * k + col]);
        for (std::size_t r = col + 1; r < k; ++r) {
            const double cand = std::abs(G[r * k + col]);
            if (cand > best) {
                best = cand;
                pivot = r;
            }
        }
        if (best <= 1e-300) {
            return std::nullopt;  // singular
        }
        if (pivot != col) {
            for (std::size_t j = 0; j < k; ++j) {
                std::swap(G[col * k + j], G[pivot * k + j]);
            }
            std::swap(d[col], d[pivot]);
        }
        const double diag = G[col * k + col];
        for (std::size_t r = 0; r < k; ++r) {
            if (r == col) {
                continue;
            }
            const double factor = G[r * k + col] / diag;
            if (factor == 0.0) {
                continue;
            }
            for (std::size_t j = col; j < k; ++j) {
                G[r * k + j] -= factor * G[col * k + j];
            }
            d[r] -= factor * d[col];
        }
    }
    std::vector<double> g(k, 0.0);
    for (std::size_t i = 0; i < k; ++i) {
        g[i] = d[i] / G[i * k + i];
    }
    return g;
}

// Least-squares fit of the columns of A indexed by `active` against b, via the normal
// equations (A_S^T A_S) g = A_S^T b. Returns the coefficients aligned with `active`,
// or std::nullopt if the active Gram matrix is singular.
[[nodiscard]] auto least_squares(std::span<const double> A, std::size_t rows, std::size_t cols,
                                 std::span<const double> b, std::span<const std::size_t> active)
    -> std::optional<std::vector<double>> {
    const std::size_t k = active.size();
    if (k == 0) {
        return std::vector<double>{};
    }
    std::vector<double> gram(k * k, 0.0);
    std::vector<double> rhs(k, 0.0);
    for (std::size_t a = 0; a < k; ++a) {
        const std::size_t ca = active[a];
        for (std::size_t bcol = a; bcol < k; ++bcol) {
            const std::size_t cb = active[bcol];
            double acc = 0.0;
            for (std::size_t i = 0; i < rows; ++i) {
                acc = std::fma(A[i * cols + ca], A[i * cols + cb], acc);
            }
            gram[a * k + bcol] = acc;
            gram[bcol * k + a] = acc;  // symmetric
        }
        rhs[a] = col_dot_vec(A, rows, cols, ca, b);
    }
    return solve_dense(std::move(gram), std::move(rhs), k);
}

// The indices of the `keep` largest-magnitude entries of x (or all of them, when keep
// exceeds x.size()).
[[nodiscard]] auto largest_magnitude(std::span<const double> x, std::size_t keep)
    -> std::vector<std::size_t> {
    std::vector<std::size_t> idx(x.size());
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    keep = std::min(keep, x.size());
    // Descending by magnitude; a full sort is cheap here (the candidate sets are small)
    // and matches the ranges-sort idiom used elsewhere in the engine.
    std::ranges::sort(idx, [&](std::size_t a, std::size_t b) {
        return std::abs(x[a]) > std::abs(x[b]);
    });
    idx.resize(keep);
    return idx;
}

}  // namespace

// --- basis pursuit (exact over Q) ------------------------------------------------

auto basis_pursuit(const std::vector<std::vector<Rational>>& A, const std::vector<Rational>& b)
    -> Result<std::vector<Rational>> {
    const std::size_t m = A.size();
    if (m == 0) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    const std::size_t n = A.front().size();
    if (n == 0) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    for (const auto& row : A) {
        if (row.size() != n) {  // ragged
            return make_error<std::vector<Rational>>(MathError::domain_error);
        }
    }
    if (b.size() != m) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }

    // --- Build the exact dual LP:  max b.y  s.t.  |a_j . y| <= 1,  y free -----------
    // maximize() wants y >= 0, so split y = p - q (p, q >= 0) into 2m variables, and
    // wants A z <= rhs with rhs >= 0 — satisfied because every rhs here is 1. Each atom
    // a_j contributes two rows:  a_j.(p - q) <= 1  and  -a_j.(p - q) <= 1.
    const std::size_t vars = 2 * m;
    const Rational one = Rational::from_int(1);

    std::vector<std::vector<Rational>> dual_A;
    dual_A.reserve(2 * n);
    for (std::size_t j = 0; j < n; ++j) {
        std::vector<Rational> pos(vars);
        std::vector<Rational> neg(vars);
        for (std::size_t i = 0; i < m; ++i) {
            const Rational& aij = A[i][j];
            auto minus = aij.negate();  // fails only on INT64_MIN
            if (!minus) {
                return make_error<std::vector<Rational>>(minus.error());
            }
            pos[i] = aij;         // coefficient of p_i
            pos[m + i] = *minus;  // coefficient of q_i  =>  a_j.(p - q)
            neg[i] = *minus;      // -a_j.(p - q)
            neg[m + i] = aij;
        }
        dual_A.push_back(std::move(pos));
        dual_A.push_back(std::move(neg));
    }
    const std::vector<Rational> dual_b(2 * n, one);  // all-ones RHS (>= 0)

    // Dual objective coefficients: [b ; -b] so that the objective is b.(p - q) = b.y.
    std::vector<Rational> dual_c(vars);
    for (std::size_t i = 0; i < m; ++i) {
        auto minus = b[i].negate();
        if (!minus) {
            return make_error<std::vector<Rational>>(minus.error());
        }
        dual_c[i] = b[i];
        dual_c[m + i] = *minus;
    }

    auto solved = maximize(dual_A, dual_b, dual_c);
    if (!solved) {
        return make_error<std::vector<Rational>>(solved.error());
    }
    // An unbounded dual certifies an INFEASIBLE primal: b is not in range(A), so there
    // is no x with A x = b and hence no basis-pursuit solution.
    if (solved->status != LpStatus::optimal) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }

    // Recover y* = p* - q* from the dual optimum.
    std::vector<Rational> y(m);
    for (std::size_t i = 0; i < m; ++i) {
        auto yi = solved->solution[i].subtract(solved->solution[m + i]);
        if (!yi) {
            return make_error<std::vector<Rational>>(yi.error());
        }
        y[i] = *yi;
    }

    // --- Complementary slackness: the support is exactly {j : |a_j . y*| = 1} --------
    auto neg_one = one.negate();
    if (!neg_one) {
        return make_error<std::vector<Rational>>(neg_one.error());
    }
    std::vector<std::size_t> support;
    for (std::size_t j = 0; j < n; ++j) {
        Rational s;  // a_j . y*
        for (std::size_t i = 0; i < m; ++i) {
            auto term = A[i][j].multiply(y[i]);
            if (!term) {
                return make_error<std::vector<Rational>>(term.error());
            }
            auto sum = s.add(*term);
            if (!sum) {
                return make_error<std::vector<Rational>>(sum.error());
            }
            s = *sum;
        }
        if (s == one || s == *neg_one) {  // exact rational tightness test
            support.push_back(j);
        }
    }

    std::vector<Rational> x(n);  // 0/1 everywhere; nonbasic atoms stay exactly zero
    if (support.empty()) {
        // No tight constraint means the optimum is at y* = 0 with value 0, i.e. b = 0
        // and the exact minimiser is the zero vector.
        return x;
    }

    // --- Exact recovery on the support via normal equations through nimblecas.matrix -
    // On the support S, x_S is the exact solution of A_S x_S = b (consistent by
    // construction). Solve the square system (A_S^T A_S) x_S = A_S^T b exactly over Q.
    const std::size_t k = support.size();
    std::vector<std::vector<Rational>> as_rows(m, std::vector<Rational>(k));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t t = 0; t < k; ++t) {
            as_rows[i][t] = A[i][support[t]];
        }
    }
    auto as = Matrix::from_rows(std::move(as_rows));
    if (!as) {
        return make_error<std::vector<Rational>>(as.error());
    }
    auto as_t = as->transpose();
    if (!as_t) {
        return make_error<std::vector<Rational>>(as_t.error());
    }
    auto gram = as_t->multiply(*as);  // k x k
    if (!gram) {
        return make_error<std::vector<Rational>>(gram.error());
    }
    std::vector<std::vector<Rational>> b_rows(m, std::vector<Rational>(1));
    for (std::size_t i = 0; i < m; ++i) {
        b_rows[i][0] = b[i];
    }
    auto b_mat = Matrix::from_rows(std::move(b_rows));
    if (!b_mat) {
        return make_error<std::vector<Rational>>(b_mat.error());
    }
    auto rhs = as_t->multiply(*b_mat);  // k x 1
    if (!rhs) {
        return make_error<std::vector<Rational>>(rhs.error());
    }
    // A singular Gram (dependent tight columns under degeneracy) => domain_error: the
    // exact support cannot be uniquely resolved, and we refuse to guess.
    auto xs = gram->solve(*rhs);
    if (!xs) {
        return make_error<std::vector<Rational>>(xs.error());
    }
    for (std::size_t t = 0; t < k; ++t) {
        x[support[t]] = xs->at(t, 0);
    }
    return x;
}

// --- mutual coherence (exact squared form) ---------------------------------------

auto mutual_coherence_squared(const std::vector<std::vector<Rational>>& A) -> Result<Rational> {
    const std::size_t m = A.size();
    if (m == 0) {
        return make_error<Rational>(MathError::domain_error);
    }
    const std::size_t n = A.front().size();
    if (n < 2) {  // coherence is a pairwise quantity
        return make_error<Rational>(MathError::domain_error);
    }
    for (const auto& row : A) {
        if (row.size() != n) {
            return make_error<Rational>(MathError::domain_error);
        }
    }

    // Precompute the exact squared norms ||a_j||^2; a zero norm is undefined coherence.
    std::vector<Rational> norms(n);
    for (std::size_t j = 0; j < n; ++j) {
        auto nn = column_dot(A, j, j);
        if (!nn) {
            return make_error<Rational>(nn.error());
        }
        if (nn->is_zero()) {
            return make_error<Rational>(MathError::domain_error);  // zero-norm atom
        }
        norms[j] = *nn;
    }

    Rational best;  // 0/1: the running maximum squared coherence
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            auto dot = column_dot(A, i, j);
            if (!dot) {
                return make_error<Rational>(dot.error());
            }
            auto num = dot->multiply(*dot);  // <a_i,a_j>^2 >= 0
            if (!num) {
                return make_error<Rational>(num.error());
            }
            auto den = norms[i].multiply(norms[j]);  // > 0
            if (!den) {
                return make_error<Rational>(den.error());
            }
            auto ratio = num->divide(*den);
            if (!ratio) {
                return make_error<Rational>(ratio.error());
            }
            auto cmp = compare(*ratio, best);
            if (!cmp) {
                return make_error<Rational>(cmp.error());
            }
            if (*cmp > 0) {
                best = *ratio;
            }
        }
    }
    return best;
}

auto coherence_guarantees_recovery(const Rational& mu_squared, std::size_t k) -> Result<bool> {
    if (k == 0) {
        return make_error<bool>(MathError::domain_error);
    }
    if (is_negative(mu_squared)) {  // a squared coherence is never negative
        return make_error<bool>(MathError::domain_error);
    }
    // k < (1 + 1/mu)/2  <=>  2k - 1 < 1/mu  <=>  mu (2k - 1) < 1  <=>  mu^2 (2k-1)^2 < 1
    // (all quantities non-negative). Everything below stays exact over Q.
    const std::int64_t two_k_minus_1 = static_cast<std::int64_t>(2 * k - 1);
    const Rational factor = Rational::from_int(two_k_minus_1);
    auto sq = factor.multiply(factor);  // (2k - 1)^2
    if (!sq) {
        return make_error<bool>(sq.error());
    }
    auto lhs = mu_squared.multiply(*sq);  // mu^2 (2k - 1)^2
    if (!lhs) {
        return make_error<bool>(lhs.error());
    }
    auto cmp = compare(*lhs, Rational::from_int(1));
    if (!cmp) {
        return make_error<bool>(cmp.error());
    }
    return *cmp < 0;
}

// --- OMP (numerical) -------------------------------------------------------------

auto orthogonal_matching_pursuit(std::span<const double> A, std::size_t rows, std::size_t cols,
                                 std::span<const double> b, std::size_t sparsity, double tol)
    -> Result<std::vector<double>> {
    if (A.size() != rows * cols || b.size() != rows) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    std::vector<double> x(cols, 0.0);
    std::vector<double> r(b.begin(), b.end());  // residual, starts at b
    std::vector<std::size_t> active;
    std::vector<char> in_active(cols, 0);

    const std::size_t budget = std::min(sparsity, cols);
    while (active.size() < budget && norm2(r) > tol) {
        // Select the atom most correlated with the residual (largest |<a_j, r>|).
        std::size_t best_j = cols;
        double best_corr = tol;  // ignore atoms that are essentially orthogonal
        for (std::size_t j = 0; j < cols; ++j) {
            if (in_active[j] != 0) {
                continue;
            }
            const double corr = std::abs(col_dot_vec(A, rows, cols, j, r));
            if (corr > best_corr) {
                best_corr = corr;
                best_j = j;
            }
        }
        if (best_j == cols) {
            break;  // no atom improves the fit
        }
        active.push_back(best_j);
        in_active[best_j] = 1;

        // Re-solve the least squares over the whole active set against the ORIGINAL b.
        auto coeffs = least_squares(A, rows, cols, b, active);
        if (!coeffs) {
            return make_error<std::vector<double>>(MathError::domain_error);  // singular
        }
        std::ranges::fill(x, 0.0);
        for (std::size_t t = 0; t < active.size(); ++t) {
            x[active[t]] = (*coeffs)[t];
        }
        r = residual(A, rows, cols, b, x);
    }
    return x;
}

// --- CoSaMP (numerical) ----------------------------------------------------------

auto cosamp(std::span<const double> A, std::size_t rows, std::size_t cols, std::span<const double> b,
            std::size_t sparsity, int max_iter, double tol) -> Result<std::vector<double>> {
    if (A.size() != rows * cols || b.size() != rows) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    const std::size_t k = std::min(sparsity, cols);
    std::vector<double> x(cols, 0.0);
    if (k == 0) {
        return x;
    }
    std::vector<double> r(b.begin(), b.end());

    for (int iter = 0; iter < max_iter && norm2(r) > tol; ++iter) {
        // Signal proxy A^T r, and its 2k largest-magnitude coordinates.
        std::vector<double> proxy(cols, 0.0);
        for (std::size_t j = 0; j < cols; ++j) {
            proxy[j] = col_dot_vec(A, rows, cols, j, r);
        }
        auto add = largest_magnitude(proxy, 2 * k);

        // Merge with the current support.
        std::vector<char> in_set(cols, 0);
        for (std::size_t j = 0; j < cols; ++j) {
            if (x[j] != 0.0) {
                in_set[j] = 1;
            }
        }
        for (std::size_t j : add) {
            in_set[j] = 1;
        }
        std::vector<std::size_t> merged;
        for (std::size_t j = 0; j < cols; ++j) {
            if (in_set[j] != 0) {
                merged.push_back(j);
            }
        }

        // Least-squares on the merged support, then prune back to the k largest.
        auto coeffs = least_squares(A, rows, cols, b, merged);
        if (!coeffs) {
            return make_error<std::vector<double>>(MathError::domain_error);
        }
        std::vector<double> full(cols, 0.0);
        for (std::size_t t = 0; t < merged.size(); ++t) {
            full[merged[t]] = (*coeffs)[t];
        }
        auto keep = largest_magnitude(full, k);
        std::ranges::fill(x, 0.0);
        for (std::size_t j : keep) {
            x[j] = full[j];
        }
        r = residual(A, rows, cols, b, x);
    }
    return x;
}

// --- Iterative Hard Thresholding (numerical) -------------------------------------

auto iterative_hard_thresholding(std::span<const double> A, std::size_t rows, std::size_t cols,
                                 std::span<const double> b, std::size_t sparsity, double step,
                                 int max_iter, double tol) -> Result<std::vector<double>> {
    if (A.size() != rows * cols || b.size() != rows) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    const std::size_t k = std::min(sparsity, cols);
    std::vector<double> x(cols, 0.0);
    if (k == 0) {
        return x;
    }
    for (int iter = 0; iter < max_iter; ++iter) {
        const std::vector<double> r = residual(A, rows, cols, b, x);
        // Gradient step: x + step * A^T r.
        std::vector<double> g(cols, 0.0);
        for (std::size_t j = 0; j < cols; ++j) {
            g[j] = std::fma(step, col_dot_vec(A, rows, cols, j, r), x[j]);
        }
        // Hard threshold to the k largest-magnitude coordinates.
        auto keep = largest_magnitude(g, k);
        std::vector<double> next(cols, 0.0);
        for (std::size_t j : keep) {
            next[j] = g[j];
        }
        // Convergence on the update size.
        double delta = 0.0;
        for (std::size_t j = 0; j < cols; ++j) {
            delta = std::fma(next[j] - x[j], next[j] - x[j], delta);
        }
        x = std::move(next);
        if (std::sqrt(delta) <= tol) {
            break;
        }
    }
    return x;
}

// --- mutual coherence (numerical) ------------------------------------------------

auto mutual_coherence(std::span<const double> A, std::size_t rows, std::size_t cols)
    -> Result<double> {
    if (A.size() != rows * cols) {
        return make_error<double>(MathError::domain_error);
    }
    if (cols < 2) {
        return make_error<double>(MathError::domain_error);
    }
    std::vector<double> norms(cols, 0.0);
    for (std::size_t j = 0; j < cols; ++j) {
        double acc = 0.0;
        for (std::size_t i = 0; i < rows; ++i) {
            const double v = A[i * cols + j];
            acc = std::fma(v, v, acc);
        }
        if (acc <= 0.0) {
            return make_error<double>(MathError::domain_error);  // zero-norm atom
        }
        norms[j] = std::sqrt(acc);
    }
    double best = 0.0;
    for (std::size_t i = 0; i < cols; ++i) {
        for (std::size_t j = i + 1; j < cols; ++j) {
            double dot = 0.0;
            for (std::size_t r = 0; r < rows; ++r) {
                dot = std::fma(A[r * cols + i], A[r * cols + j], dot);
            }
            const double coh = std::abs(dot) / (norms[i] * norms[j]);
            best = std::max(best, coh);
        }
    }
    return best;
}

// --- parallel batch recovery (deterministic; identical to serial) ----------------

auto parallel_omp_batch(std::span<const double> A, std::size_t rows, std::size_t cols,
                        std::span<const std::vector<double>> measurements, std::size_t sparsity,
                        double tol, std::size_t grain)
    -> std::vector<Result<std::vector<double>>> {
    // transform_index is an order-preserving parallel map writing per-index slots; the
    // closure reads only shared const state (A, and its own measurements[i]) and calls
    // the pure orthogonal_matching_pursuit — no data race, no scheduling dependence.
    return parallel::transform_index(
        measurements.size(),
        [A, rows, cols, sparsity, tol, measurements](std::size_t i) -> Result<std::vector<double>> {
            return orthogonal_matching_pursuit(A, rows, cols, measurements[i], sparsity, tol);
        },
        grain);
}

auto parallel_basis_pursuit_batch(const std::vector<std::vector<Rational>>& A,
                                  std::span<const std::vector<Rational>> measurements,
                                  std::size_t grain) -> std::vector<Result<std::vector<Rational>>> {
    // basis_pursuit reads only A and its own b, building purely local LP/matrix state, so
    // distinct columns recover concurrently with no interference. Exact over Q per column.
    return parallel::transform_index(
        measurements.size(),
        [&A, measurements](std::size_t i) -> Result<std::vector<Rational>> {
            return basis_pursuit(A, measurements[i]);
        },
        grain);
}

// --- distributed shard entrypoints (stateless, pure) -----------------------------

namespace {

// The ascending original indices assigned to shard `shard_index`: i with
// i % num_shards == shard_index. Empty when the shard has no work.
[[nodiscard]] auto shard_indices(std::size_t total, std::size_t shard_index,
                                 std::size_t num_shards) -> std::vector<std::size_t> {
    std::vector<std::size_t> idx;
    for (std::size_t i = shard_index; i < total; i += num_shards) {
        idx.push_back(i);
    }
    return idx;
}

}  // namespace

auto recover_batch_shard(std::span<const double> A, std::size_t rows, std::size_t cols,
                         std::span<const std::vector<double>> measurements,
                         std::size_t shard_index, std::size_t num_shards, std::size_t sparsity,
                         double tol, std::size_t grain)
    -> Result<std::vector<Result<std::vector<double>>>> {
    using Batch = std::vector<Result<std::vector<double>>>;
    if (num_shards == 0 || shard_index >= num_shards) {
        return make_error<Batch>(MathError::domain_error);
    }
    const std::vector<std::size_t> assigned =
        shard_indices(measurements.size(), shard_index, num_shards);
    // Recover only this shard's columns, in ascending assigned-index order, in parallel.
    return parallel::transform_index(
        assigned.size(),
        [A, rows, cols, sparsity, tol, measurements, &assigned](std::size_t t)
            -> Result<std::vector<double>> {
            return orthogonal_matching_pursuit(A, rows, cols, measurements[assigned[t]], sparsity,
                                               tol);
        },
        grain);
}

auto basis_pursuit_batch_shard(const std::vector<std::vector<Rational>>& A,
                               std::span<const std::vector<Rational>> measurements,
                               std::size_t shard_index, std::size_t num_shards, std::size_t grain)
    -> Result<std::vector<Result<std::vector<Rational>>>> {
    using Batch = std::vector<Result<std::vector<Rational>>>;
    if (num_shards == 0 || shard_index >= num_shards) {
        return make_error<Batch>(MathError::domain_error);
    }
    const std::vector<std::size_t> assigned =
        shard_indices(measurements.size(), shard_index, num_shards);
    return parallel::transform_index(
        assigned.size(),
        [&A, measurements, &assigned](std::size_t t) -> Result<std::vector<Rational>> {
            return basis_pursuit(A, measurements[assigned[t]]);
        },
        grain);
}

}  // namespace nimblecas
