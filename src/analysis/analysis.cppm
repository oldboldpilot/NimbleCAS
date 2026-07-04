// NimbleCAS numerical / exact analysis: conditioning, convergence, Lyapunov stability.
// @author Olumuyiwa Oluwasanmi
//
// This module sits on top of the exact rational Matrix substrate (nimblecas.matrix)
// and the Routh-Hurwitz stability engine (nimblecas.dynamics). It groups three
// classical analysis facilities and is scrupulously honest about which results are
// EXACT over the rationals Q and which are NUMERICAL (double) estimates:
//
//   CONDITION NUMBER
//     condition_1 / condition_inf : kappa_p(A) = ||A||_p * ||A^{-1}||_p for p = 1 and
//        p = infinity. Both norms (max abs column sum / max abs row sum) and the inverse
//        are computed with exact Rational arithmetic, so kappa_1 and kappa_inf are EXACT.
//     condition_2_estimate         : the spectral condition number sigma_max/sigma_min,
//        obtained by power iteration on A^T A in double precision -- NUMERICAL, an estimate.
//
//   CONVERGENCE TESTS for a series sum a_n
//     ratio_test  : L = lim |a_{n+1}/a_n|. When the ratio is a constant rational (e.g. a
//        geometric series) the limit is detected and returned EXACTLY over Q; otherwise a
//        NUMERICAL estimate of the finite-n ratio is reported (a documented heuristic).
//     root_test / comparison_test / integral_test / alternating_series_test : NUMERICAL
//        (double) verdicts. The alternating (Leibniz) test is exact in its logic but
//        evaluated on sampled doubles.
//
//   LYAPUNOV
//     lyapunov_solve  : the continuous equation A^T P + P A = -Q solved EXACTLY over Q by
//        vectorization -- (I (x) A^T + A^T (x) I) vec(P) = -vec(Q) -- an exact rational
//        linear solve (Matrix::solve on the Kronecker-sum system).
//     stein_solve     : the discrete Stein equation A^T P A - P = -Q, likewise EXACT via
//        (A^T (x) A^T - I) vec(P) = -vec(Q).
//     is_positive_definite : Sylvester's criterion -- all leading principal minors > 0,
//        each an EXACT rational determinant (assumes a symmetric input, as the Lyapunov
//        solution is for symmetric Q).
//     is_stable_lyapunov   : continuous-LTI asymptotic stability, decided EXACTLY -- pick
//        Q = I, solve the Lyapunov equation, and test P > 0 by Sylvester's criterion.
//     stability_cross_check: cross-checks the exact Lyapunov verdict against the exact
//        Routh-Hurwitz verdict (nimblecas.dynamics) on the characteristic polynomial.
//     lyapunov_exponent    : the leading Lyapunov exponent of a Jacobian product,
//        (1/n) log ||J_n ... J_1||, by probe-vector growth in double precision --
//        NUMERICAL, an estimate.
//
// Every fallible step is threaded on the Result railway; dimension violations surface as
// MathError::domain_error and int64 arithmetic boundaries as MathError::overflow.

export module nimblecas.analysis;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.dynamics;

export namespace nimblecas {

// ===========================================================================
// Condition number.
// ===========================================================================

// ||A||_1 = max over columns j of sum_i |a_ij| (max absolute column sum). EXACT over Q.
// Requires a non-empty matrix; a 0x0 / 0-column matrix yields domain_error.
[[nodiscard]] auto matrix_norm_1(const Matrix& a) -> Result<Rational>;

// ||A||_inf = max over rows i of sum_j |a_ij| (max absolute row sum). EXACT over Q.
// Requires a non-empty matrix; a 0-row matrix yields domain_error.
[[nodiscard]] auto matrix_norm_inf(const Matrix& a) -> Result<Rational>;

// kappa_1(A) = ||A||_1 * ||A^{-1}||_1, EXACT over Q. Requires a square, nonsingular
// matrix (a singular or non-square A yields domain_error from the inverse).
[[nodiscard]] auto condition_1(const Matrix& a) -> Result<Rational>;

// kappa_inf(A) = ||A||_inf * ||A^{-1}||_inf, EXACT over Q. Same requirements as above.
[[nodiscard]] auto condition_inf(const Matrix& a) -> Result<Rational>;

// The spectral condition number kappa_2(A) = sigma_max / sigma_min, where the singular
// values sigma are the square roots of the eigenvalues of A^T A. NUMERICAL: computed by
// power iteration on A^T A and on (A^T A)^{-1} in double precision -- an ESTIMATE, not an
// exact rational value. Requires a square, nonsingular matrix (singular => domain_error).
[[nodiscard]] auto condition_2_estimate(const Matrix& a) -> Result<double>;

// ===========================================================================
// Convergence tests for a series sum a_n.
// ===========================================================================

// A real-valued sequence term a_n (n >= 0). Total, never fails by signature.
using RealSequence = std::function<double(std::int64_t)>;

// An exact rational sequence term a_n (n >= 0). Total, never fails by signature.
using RationalSequence = std::function<Rational(std::int64_t)>;

enum class Verdict : std::uint8_t { converges, diverges, inconclusive };

[[nodiscard]] auto to_string_view(Verdict v) noexcept -> std::string_view;

// Result of the ratio (d'Alembert) test.
struct RatioTest {
    Verdict verdict{Verdict::inconclusive};
    bool exact{false};                      // true iff the limit was determined exactly over Q
    std::optional<Rational> exact_limit{};  // L = lim |a_{n+1}/a_n| when `exact`
    double numeric_limit{0.0};              // a double view of the limit / finite-n estimate
};

// Ratio test on the exact rational term a_n. If |a_{n+1}/a_n| is a constant rational over
// the sampled n (as for a geometric series a_n = c r^n, whose ratio is exactly |r|), that
// constant is returned as the EXACT limit L and the verdict follows exactly: L < 1
// converges, L > 1 diverges, L == 1 inconclusive. Otherwise the limit is not pinned down
// exactly and a NUMERICAL finite-n estimate is reported with a heuristic verdict (a
// documented approximation -- the ratio test is genuinely inconclusive when L -> 1).
[[nodiscard]] auto ratio_test(const RationalSequence& a) -> RatioTest;

// Result of a numerical convergence test.
struct NumericTest {
    Verdict verdict{Verdict::inconclusive};
    double numeric_limit{0.0};  // the computed test statistic (NUMERICAL estimate)
};

// Root (Cauchy) test: estimate L = lim |a_n|^{1/n} at the sample index n (NUMERICAL).
// L < 1 converges, L > 1 diverges, L ~ 1 inconclusive. Requires n >= 1.
[[nodiscard]] auto root_test(const RealSequence& a, std::int64_t n = 64) -> NumericTest;

// Direct comparison test (NUMERICAL). Given the known behaviour of a non-negative
// reference b_n, decide a_n by term-wise domination over n = 1..samples:
//   b converges and 0 <= a_n <= b_n for all sampled n  => a converges
//   b diverges  and a_n >= b_n >= 0 for all sampled n   => a diverges
//   otherwise                                           => inconclusive
[[nodiscard]] auto comparison_test(const RealSequence& a, const RealSequence& b,
                                   Verdict b_behaviour, std::int64_t samples = 200) -> Verdict;

// Integral test hook (NUMERICAL). For a positive, decreasing f, sum a_n converges iff the
// improper integral of f converges. The improper integral is estimated by the trapezoidal
// rule on the integer grid; convergence is inferred from the tail increment
// I(2N) - I(N) -> 0. `numeric_limit` carries the partial integral I(upper).
[[nodiscard]] auto integral_test(const RealSequence& f, std::int64_t upper = 4096) -> NumericTest;

// Alternating-series (Leibniz) test (NUMERICAL). `magnitude` supplies b_n = |a_n| >= 0 of
// an alternating series sum (-1)^n b_n. Converges when b_n is monotonically non-increasing
// and b_n -> 0; diverges when b_n does not tend to 0 (nth-term test); otherwise
// inconclusive. Sampled over n = 1..samples (and 2*samples for the limit check).
[[nodiscard]] auto alternating_series_test(const RealSequence& magnitude,
                                           std::int64_t samples = 1000) -> Verdict;

// ===========================================================================
// Lyapunov equations and stability.
// ===========================================================================

// Solve the continuous Lyapunov equation A^T P + P A = -Q for P, EXACTLY over Q, by
// vectorization: (I (x) A^T + A^T (x) I) vec(P) = -vec(Q) with column-major vec. `A` and
// `Q` must both be square and of the same size n >= 1, else domain_error. The Kronecker-
// sum system is singular exactly when two eigenvalues of A sum to zero (e.g. a zero or a
// purely imaginary eigenvalue) -- that surfaces as domain_error from the underlying solve.
[[nodiscard]] auto lyapunov_solve(const Matrix& a, const Matrix& q) -> Result<Matrix>;

// Solve the discrete Stein equation A^T P A - P = -Q for P, EXACTLY over Q, by
// vectorization: (A^T (x) A^T - I) vec(P) = -vec(Q). Same shape requirements as above; the
// system is singular exactly when a product of two eigenvalues of A equals 1.
[[nodiscard]] auto stein_solve(const Matrix& a, const Matrix& q) -> Result<Matrix>;

// Test P > 0 (positive definite) via Sylvester's criterion: all n leading principal minors
// are strictly positive, each an EXACT rational determinant. The criterion presumes a
// SYMMETRIC P (as produced by lyapunov_solve for a symmetric Q); it is not a definiteness
// test for general non-symmetric matrices. Requires a square matrix, else domain_error.
[[nodiscard]] auto is_positive_definite(const Matrix& p) -> Result<bool>;

// Continuous-LTI asymptotic stability of dx/dt = A x, decided EXACTLY over Q: pick Q = I,
// solve A^T P + P A = -I, and return whether P > 0 by Sylvester's criterion. A is
// asymptotically stable iff such a P is positive definite. A singular Kronecker-sum system
// (an eigenvalue pair summing to zero -- necessarily non-Hurwitz) is reported as not stable
// (false). Requires a square matrix with n >= 1, else domain_error.
[[nodiscard]] auto is_stable_lyapunov(const Matrix& a) -> Result<bool>;

// The two exact stability verdicts side by side.
struct StabilityCrossCheck {
    bool lyapunov_stable{false};      // exact: Sylvester on the Lyapunov solution (Q = I)
    bool routh_hurwitz_stable{false}; // exact: nimblecas.dynamics Routh-Hurwitz criterion
    bool agree{false};                // lyapunov_stable == routh_hurwitz_stable
};

// Cross-check the exact Lyapunov/Sylvester stability verdict against the exact Routh-
// Hurwitz verdict on the characteristic polynomial. Both are exact over Q and must agree
// for an asymptotically stable / unstable A. Requires a square matrix n >= 1.
[[nodiscard]] auto stability_cross_check(const Matrix& a) -> Result<StabilityCrossCheck>;

// The leading Lyapunov exponent of a product of Jacobians J_count ... J_1:
// (1/count) * log ||J_count ... J_1|| estimated by the growth rate of a probe vector with
// per-step renormalization, in double precision -- NUMERICAL, an ESTIMATE. `jacobians`
// holds J_1, J_2, ... in application order (J_1 first). Requires a non-empty span of
// square matrices of one common size n >= 1, else domain_error. A product that collapses a
// direction to zero yields -infinity (total contraction).
[[nodiscard]] auto lyapunov_exponent(std::span<const Matrix> jacobians) -> Result<double>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// --- small Rational helpers -------------------------------------------------

// A double view of an exact fraction (for NUMERICAL estimates and reporting only).
[[nodiscard]] auto to_double(const Rational& r) -> double {
    return static_cast<double>(r.numerator()) / static_cast<double>(r.denominator());
}

// |r|, exact. Fails only when the numerator is INT64_MIN (unrepresentable negation).
[[nodiscard]] auto rat_abs(const Rational& r) -> Result<Rational> {
    if (r.numerator() < 0) {
        return r.negate();
    }
    return r;
}

// Compare a NON-NEGATIVE rational L to 1 without any arithmetic: with den > 0 canonical,
// L < 1 iff num < den, L > 1 iff num > den, L == 1 iff num == den.
[[nodiscard]] auto verdict_vs_one(const Rational& l) -> Verdict {
    if (l.numerator() < l.denominator()) {
        return Verdict::converges;
    }
    if (l.numerator() > l.denominator()) {
        return Verdict::diverges;
    }
    return Verdict::inconclusive;
}

[[nodiscard]] auto verdict_from_double(double l, double tol) -> Verdict {
    if (l < 1.0 - tol) {
        return Verdict::converges;
    }
    if (l > 1.0 + tol) {
        return Verdict::diverges;
    }
    return Verdict::inconclusive;
}

// --- double bridges for the NUMERICAL routines ------------------------------

// A row-major double copy of an exact matrix (NUMERICAL bridge only).
[[nodiscard]] auto to_double_buffer(const Matrix& m) -> std::vector<double> {
    const std::size_t rows = m.rows();
    const std::size_t cols = m.cols();
    std::vector<double> out(rows * cols);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            out[i * cols + j] = to_double(m.at(i, j));
        }
    }
    return out;
}

// Dominant eigenvalue magnitude of a symmetric n x n double matrix (row-major) via power
// iteration with a Rayleigh quotient. Returns 0 for the zero matrix. NUMERICAL.
[[nodiscard]] auto dominant_eigenvalue(std::span<const double> m, std::size_t n) -> double {
    if (n == 0) {
        return 0.0;
    }
    std::vector<double> v(n, 1.0 / std::sqrt(static_cast<double>(n)));
    std::vector<double> w(n, 0.0);
    double lambda = 0.0;
    constexpr std::size_t max_iters = 500;
    constexpr double converge_tol = 1e-14;
    for (std::size_t it = 0; it < max_iters; ++it) {
        for (std::size_t i = 0; i < n; ++i) {
            double acc = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                acc += m[i * n + j] * v[j];
            }
            w[i] = acc;
        }
        double norm = 0.0;
        for (const double x : w) {
            norm += x * x;
        }
        norm = std::sqrt(norm);
        if (norm == 0.0) {
            return 0.0;  // the matrix annihilates v -> a zero eigenvalue direction
        }
        // Rayleigh quotient v^T (M v) with the current unit v (before renormalizing).
        double rayleigh = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            rayleigh += v[i] * w[i];
        }
        for (std::size_t i = 0; i < n; ++i) {
            v[i] = w[i] / norm;
        }
        if (std::fabs(rayleigh - lambda) <= converge_tol * (1.0 + std::fabs(rayleigh))) {
            lambda = rayleigh;
            break;
        }
        lambda = rayleigh;
    }
    return lambda;
}

// --- Kronecker machinery for the Lyapunov / Stein solves --------------------

// The Kronecker product B (x) C of two n x n exact matrices: an (n*n) x (n*n) matrix whose
// entry at (p*n + r, q*n + s) is B(p,q) * C(r,s). Overflow in an entry product propagates.
[[nodiscard]] auto kron(const Matrix& b, const Matrix& c) -> Result<Matrix> {
    const std::size_t n = b.rows();
    const std::size_t big = n * n;
    std::vector<std::vector<Rational>> rows(big, std::vector<Rational>(big, Rational{}));
    for (std::size_t p = 0; p < n; ++p) {
        for (std::size_t q = 0; q < n; ++q) {
            const Rational& bpq = b.at(p, q);
            for (std::size_t r = 0; r < n; ++r) {
                for (std::size_t s = 0; s < n; ++s) {
                    auto prod = bpq.multiply(c.at(r, s));
                    if (!prod) {
                        return make_error<Matrix>(prod.error());
                    }
                    rows[p * n + r][q * n + s] = *prod;
                }
            }
        }
    }
    return Matrix::from_rows(std::move(rows));
}

// Column-major vec of -Q as an (n*n) x 1 column: entry (i + j*n, 0) = -Q(i,j).
[[nodiscard]] auto neg_vec(const Matrix& q) -> Result<Matrix> {
    const std::size_t n = q.rows();
    std::vector<std::vector<Rational>> rows(n * n, std::vector<Rational>(1, Rational{}));
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            auto neg = q.at(i, j).negate();
            if (!neg) {
                return make_error<Matrix>(neg.error());
            }
            rows[i + j * n][0] = *neg;
        }
    }
    return Matrix::from_rows(std::move(rows));
}

// Reshape an (n*n) x 1 column-major solution back into the n x n matrix P.
[[nodiscard]] auto unvec(const Matrix& x, std::size_t n) -> Result<Matrix> {
    std::vector<std::vector<Rational>> rows(n, std::vector<Rational>(n, Rational{}));
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            rows[i][j] = x.at(i + j * n, 0);
        }
    }
    return Matrix::from_rows(std::move(rows));
}

}  // namespace

// --- condition number -------------------------------------------------------

auto matrix_norm_1(const Matrix& a) -> Result<Rational> {
    if (a.rows() == 0 || a.cols() == 0) {
        return make_error<Rational>(MathError::domain_error);
    }
    Rational best;  // 0/1; column sums are non-negative
    for (std::size_t j = 0; j < a.cols(); ++j) {
        Rational sum;  // 0/1
        for (std::size_t i = 0; i < a.rows(); ++i) {
            auto abs = rat_abs(a.at(i, j));
            if (!abs) {
                return make_error<Rational>(abs.error());
            }
            auto next = sum.add(*abs);
            if (!next) {
                return make_error<Rational>(next.error());
            }
            sum = *next;
        }
        // best = max(best, sum): sum - best > 0 ?
        auto diff = sum.subtract(best);
        if (!diff) {
            return make_error<Rational>(diff.error());
        }
        if (diff->numerator() > 0) {
            best = sum;
        }
    }
    return best;
}

auto matrix_norm_inf(const Matrix& a) -> Result<Rational> {
    if (a.rows() == 0 || a.cols() == 0) {
        return make_error<Rational>(MathError::domain_error);
    }
    Rational best;  // 0/1
    for (std::size_t i = 0; i < a.rows(); ++i) {
        Rational sum;  // 0/1
        for (std::size_t j = 0; j < a.cols(); ++j) {
            auto abs = rat_abs(a.at(i, j));
            if (!abs) {
                return make_error<Rational>(abs.error());
            }
            auto next = sum.add(*abs);
            if (!next) {
                return make_error<Rational>(next.error());
            }
            sum = *next;
        }
        auto diff = sum.subtract(best);
        if (!diff) {
            return make_error<Rational>(diff.error());
        }
        if (diff->numerator() > 0) {
            best = sum;
        }
    }
    return best;
}

auto condition_1(const Matrix& a) -> Result<Rational> {
    if (!a.is_square()) {
        return make_error<Rational>(MathError::domain_error);
    }
    auto inv = a.inverse();  // singular / non-square => domain_error
    if (!inv) {
        return make_error<Rational>(inv.error());
    }
    auto na = matrix_norm_1(a);
    if (!na) {
        return make_error<Rational>(na.error());
    }
    auto ni = matrix_norm_1(*inv);
    if (!ni) {
        return make_error<Rational>(ni.error());
    }
    return na->multiply(*ni);
}

auto condition_inf(const Matrix& a) -> Result<Rational> {
    if (!a.is_square()) {
        return make_error<Rational>(MathError::domain_error);
    }
    auto inv = a.inverse();
    if (!inv) {
        return make_error<Rational>(inv.error());
    }
    auto na = matrix_norm_inf(a);
    if (!na) {
        return make_error<Rational>(na.error());
    }
    auto ni = matrix_norm_inf(*inv);
    if (!ni) {
        return make_error<Rational>(ni.error());
    }
    return na->multiply(*ni);
}

auto condition_2_estimate(const Matrix& a) -> Result<double> {
    if (!a.is_square()) {
        return make_error<double>(MathError::domain_error);
    }
    const std::size_t n = a.rows();
    if (n == 0) {
        return make_error<double>(MathError::domain_error);
    }
    // M = A^T A is symmetric positive semidefinite; its eigenvalues are the squared
    // singular values of A. Built exactly, then bridged to double for power iteration.
    auto at = a.transpose();
    if (!at) {
        return make_error<double>(at.error());
    }
    auto m = at->multiply(a);
    if (!m) {
        return make_error<double>(m.error());
    }
    // (A^T A)^{-1} exists iff A is nonsingular; a singular A means kappa_2 = infinity.
    auto minv = m->inverse();
    if (!minv) {
        return make_error<double>(minv.error());  // domain_error: singular => ill-conditioned
    }
    const std::vector<double> md = to_double_buffer(*m);
    const std::vector<double> mid = to_double_buffer(*minv);
    const double lambda_max = dominant_eigenvalue(md, n);            // sigma_max^2
    const double lambda_max_inv = dominant_eigenvalue(mid, n);      // 1 / sigma_min^2
    const double ratio = lambda_max * lambda_max_inv;               // (sigma_max/sigma_min)^2
    if (ratio <= 0.0) {
        return make_error<double>(MathError::domain_error);
    }
    return std::sqrt(ratio);
}

// --- convergence tests ------------------------------------------------------

auto to_string_view(Verdict v) noexcept -> std::string_view {
    switch (v) {
        case Verdict::converges:    return "converges";
        case Verdict::diverges:     return "diverges";
        case Verdict::inconclusive: return "inconclusive";
    }
    return "inconclusive";
}

auto ratio_test(const RationalSequence& a) -> RatioTest {
    // Sample |a_{n+1}/a_n| at a few small indices (small keeps geometric r^n inside int64
    // before it overflows). If every computable ratio is the SAME exact rational, that
    // constant is the exact limit L; otherwise fall back to a numerical estimate.
    std::optional<Rational> constant{};
    bool constant_holds = true;
    std::size_t samples = 0;
    double last_numeric = 0.0;
    bool have_numeric = false;

    for (std::int64_t k = 1; k <= 4; ++k) {
        const Rational ak = a(k);
        if (ak.is_zero()) {
            continue;
        }
        const Rational ak1 = a(k + 1);
        auto ratio = ak1.divide(ak);
        if (!ratio) {
            continue;  // overflow in the exact ratio at this k -> skip this sample
        }
        auto abs = rat_abs(*ratio);
        if (!abs) {
            continue;
        }
        last_numeric = to_double(*abs);
        have_numeric = true;
        ++samples;
        if (!constant.has_value()) {
            constant = *abs;
        } else if (!(*constant == *abs)) {
            constant_holds = false;
        }
    }

    RatioTest result;
    if (samples >= 2 && constant_holds && constant.has_value()) {
        result.exact = true;
        result.exact_limit = *constant;
        result.numeric_limit = to_double(*constant);
        result.verdict = verdict_vs_one(*constant);
        return result;
    }

    // NUMERICAL fallback: the finite-n ratio at the largest sampled index (heuristic).
    result.exact = false;
    result.numeric_limit = have_numeric ? last_numeric : 0.0;
    result.verdict = have_numeric ? verdict_from_double(last_numeric, 1e-9)
                                  : Verdict::inconclusive;
    return result;
}

auto root_test(const RealSequence& a, std::int64_t n) -> NumericTest {
    NumericTest result;
    if (n < 1) {
        return result;  // inconclusive, statistic 0
    }
    const double an = std::fabs(a(n));
    const double l = (an == 0.0) ? 0.0 : std::pow(an, 1.0 / static_cast<double>(n));
    result.numeric_limit = l;
    result.verdict = verdict_from_double(l, 1e-9);
    return result;
}

auto comparison_test(const RealSequence& a, const RealSequence& b, Verdict b_behaviour,
                     std::int64_t samples) -> Verdict {
    if (samples < 1) {
        return Verdict::inconclusive;
    }
    if (b_behaviour == Verdict::converges) {
        for (std::int64_t n = 1; n <= samples; ++n) {
            const double an = a(n);
            const double bn = b(n);
            if (!(an >= 0.0 && an <= bn)) {
                return Verdict::inconclusive;
            }
        }
        return Verdict::converges;
    }
    if (b_behaviour == Verdict::diverges) {
        for (std::int64_t n = 1; n <= samples; ++n) {
            const double an = a(n);
            const double bn = b(n);
            if (!(bn >= 0.0 && an >= bn)) {
                return Verdict::inconclusive;
            }
        }
        return Verdict::diverges;
    }
    return Verdict::inconclusive;
}

auto integral_test(const RealSequence& f, std::int64_t upper) -> NumericTest {
    NumericTest result;
    if (upper < 2) {
        return result;
    }
    // Trapezoidal partial integral of f on the integer grid [1, limit].
    const auto partial_integral = [&f](std::int64_t limit) -> double {
        double sum = 0.0;
        for (std::int64_t x = 1; x < limit; ++x) {
            sum += 0.5 * (f(x) + f(x + 1));
        }
        return sum;
    };
    const double i_n = partial_integral(upper);
    const double i_2n = partial_integral(2 * upper);
    const double tail = i_2n - i_n;  // contribution of [upper, 2*upper]
    result.numeric_limit = i_n;
    // A convergent improper integral has a vanishing tail increment; a divergent one
    // (e.g. 1/x, whose decade increments stay ~ln 2) does not.
    if (tail < 1e-3) {
        result.verdict = Verdict::converges;
    } else {
        result.verdict = Verdict::diverges;
    }
    return result;
}

auto alternating_series_test(const RealSequence& magnitude, std::int64_t samples) -> Verdict {
    if (samples < 2) {
        return Verdict::inconclusive;
    }
    constexpr double eps = 1e-12;
    bool monotone = true;
    for (std::int64_t n = 1; n < samples; ++n) {
        if (magnitude(n + 1) > magnitude(n) + eps) {
            monotone = false;
            break;
        }
    }
    const double b1 = std::fabs(magnitude(1));
    const double l1 = std::fabs(magnitude(samples));
    const double l2 = std::fabs(magnitude(2 * samples));
    if (!monotone) {
        return Verdict::inconclusive;
    }
    // b_n monotonically decreasing to 0 -> Leibniz convergence.
    if (l2 <= l1 && l2 < 1e-6 + 1e-3 * b1) {
        return Verdict::converges;
    }
    // b_n not tending to 0 (nth-term test fails) -> divergence.
    if (l1 > 1e-3 && l2 >= l1 - eps) {
        return Verdict::diverges;
    }
    return Verdict::inconclusive;
}

// --- Lyapunov equations -----------------------------------------------------

auto lyapunov_solve(const Matrix& a, const Matrix& q) -> Result<Matrix> {
    if (!a.is_square() || a.rows() < 1) {
        return make_error<Matrix>(MathError::domain_error);
    }
    if (q.rows() != a.rows() || q.cols() != a.cols()) {
        return make_error<Matrix>(MathError::domain_error);
    }
    const std::size_t n = a.rows();
    const Matrix id = Matrix::identity(n);
    auto at = a.transpose();
    if (!at) {
        return make_error<Matrix>(at.error());
    }
    // K = I (x) A^T + A^T (x) I  (column-major vec convention).
    auto k_left = kron(id, *at);
    if (!k_left) {
        return make_error<Matrix>(k_left.error());
    }
    auto k_right = kron(*at, id);
    if (!k_right) {
        return make_error<Matrix>(k_right.error());
    }
    auto k = k_left->add(*k_right);
    if (!k) {
        return make_error<Matrix>(k.error());
    }
    auto rhs = neg_vec(q);  // -vec(Q)
    if (!rhs) {
        return make_error<Matrix>(rhs.error());
    }
    auto x = k->solve(*rhs);  // singular Kronecker sum => domain_error
    if (!x) {
        return make_error<Matrix>(x.error());
    }
    return unvec(*x, n);
}

auto stein_solve(const Matrix& a, const Matrix& q) -> Result<Matrix> {
    if (!a.is_square() || a.rows() < 1) {
        return make_error<Matrix>(MathError::domain_error);
    }
    if (q.rows() != a.rows() || q.cols() != a.cols()) {
        return make_error<Matrix>(MathError::domain_error);
    }
    const std::size_t n = a.rows();
    auto at = a.transpose();
    if (!at) {
        return make_error<Matrix>(at.error());
    }
    // K = A^T (x) A^T - I_{n^2}.
    auto kk = kron(*at, *at);
    if (!kk) {
        return make_error<Matrix>(kk.error());
    }
    const Matrix big_id = Matrix::identity(n * n);
    auto k = kk->subtract(big_id);
    if (!k) {
        return make_error<Matrix>(k.error());
    }
    auto rhs = neg_vec(q);  // -vec(Q)
    if (!rhs) {
        return make_error<Matrix>(rhs.error());
    }
    auto x = k->solve(*rhs);
    if (!x) {
        return make_error<Matrix>(x.error());
    }
    return unvec(*x, n);
}

auto is_positive_definite(const Matrix& p) -> Result<bool> {
    if (!p.is_square()) {
        return make_error<bool>(MathError::domain_error);
    }
    const std::size_t n = p.rows();
    // Sylvester's criterion: every leading principal minor must be strictly positive.
    for (std::size_t k = 1; k <= n; ++k) {
        std::vector<std::vector<Rational>> block(k, std::vector<Rational>(k, Rational{}));
        for (std::size_t i = 0; i < k; ++i) {
            for (std::size_t j = 0; j < k; ++j) {
                block[i][j] = p.at(i, j);
            }
        }
        auto sub = Matrix::from_rows(std::move(block));
        if (!sub) {
            return make_error<bool>(sub.error());
        }
        auto det = sub->determinant();
        if (!det) {
            return make_error<bool>(det.error());
        }
        if (det->numerator() <= 0) {  // den > 0, so the sign is the numerator's sign
            return false;
        }
    }
    return true;
}

auto is_stable_lyapunov(const Matrix& a) -> Result<bool> {
    if (!a.is_square() || a.rows() < 1) {
        return make_error<bool>(MathError::domain_error);
    }
    const Matrix q = Matrix::identity(a.rows());  // Q = I (positive definite)
    auto p = lyapunov_solve(a, q);
    if (!p) {
        // A singular Kronecker sum (an eigenvalue pair summing to zero) cannot occur for a
        // Hurwitz A, so a domain_error here means A is NOT asymptotically stable.
        if (p.error() == MathError::domain_error) {
            return false;
        }
        return make_error<bool>(p.error());
    }
    return is_positive_definite(*p);
}

auto stability_cross_check(const Matrix& a) -> Result<StabilityCrossCheck> {
    if (!a.is_square() || a.rows() < 1) {
        return make_error<StabilityCrossCheck>(MathError::domain_error);
    }
    auto lyap = is_stable_lyapunov(a);
    if (!lyap) {
        return make_error<StabilityCrossCheck>(lyap.error());
    }
    auto routh = is_asymptotically_stable(a);  // nimblecas.dynamics, exact Routh-Hurwitz
    if (!routh) {
        return make_error<StabilityCrossCheck>(routh.error());
    }
    StabilityCrossCheck out;
    out.lyapunov_stable = *lyap;
    out.routh_hurwitz_stable = *routh;
    out.agree = (*lyap == *routh);
    return out;
}

auto lyapunov_exponent(std::span<const Matrix> jacobians) -> Result<double> {
    if (jacobians.empty()) {
        return make_error<double>(MathError::domain_error);
    }
    const std::size_t n = jacobians.front().rows();
    if (n < 1) {
        return make_error<double>(MathError::domain_error);
    }
    for (const Matrix& j : jacobians) {
        if (!j.is_square() || j.rows() != n) {
            return make_error<double>(MathError::domain_error);
        }
    }
    // Probe-vector growth: v_0 a unit vector; for each J, v <- J v with the log of the
    // per-step norm accumulated, then renormalize. The mean log-growth is the leading
    // Lyapunov exponent. NUMERICAL (double) -- an estimate.
    std::vector<double> v(n, 1.0 / std::sqrt(static_cast<double>(n)));
    std::vector<double> w(n, 0.0);
    double sum_log = 0.0;
    for (const Matrix& j : jacobians) {
        const std::vector<double> jd = to_double_buffer(j);
        for (std::size_t i = 0; i < n; ++i) {
            double acc = 0.0;
            for (std::size_t c = 0; c < n; ++c) {
                acc += jd[i * n + c] * v[c];
            }
            w[i] = acc;
        }
        double norm = 0.0;
        for (const double x : w) {
            norm += x * x;
        }
        norm = std::sqrt(norm);
        if (norm == 0.0) {
            return -std::numeric_limits<double>::infinity();  // total contraction
        }
        sum_log += std::log(norm);
        for (std::size_t i = 0; i < n; ++i) {
            v[i] = w[i] / norm;
        }
    }
    return sum_log / static_cast<double>(jacobians.size());
}

}  // namespace nimblecas
