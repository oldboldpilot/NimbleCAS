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
    // `exact` == true means the decisive statistic was a CONSTANT RATIONAL across the
    // sampled indices, and that constant is taken as the exact rational limit in
    // `exact_limit`. This is a sampling determination, not a proof of constancy: it is
    // correct for the intended inputs (geometric / bounded-degree rational-function ratios),
    // but a pathological sequence engineered to match the statistic at every sampled index
    // yet differ elsewhere could defeat it. The samples are spread widely enough that only a
    // very-high-degree crafted sequence could do so; realistic inputs cannot.
    bool exact{false};
    std::optional<Rational> exact_limit{};  // the constant rational limit when `exact`
    double numeric_limit{0.0};              // a double view of the limit / finite-n estimate
};

// Ratio test on the exact rational term a_n. If |a_{n+1}/a_n| is a CONSTANT RATIONAL across
// the widely-spread sampled indices (as for a geometric series a_n = c r^n, whose ratio is
// exactly |r| at every n), that constant is taken as the limit L and the verdict follows
// from it: L < 1 converges, L > 1 diverges, L == 1 inconclusive. Otherwise the limit is not
// pinned down and a NUMERICAL finite-n estimate is reported with an inconclusive verdict --
// a few finite samples cannot distinguish a genuine L < 1 from L -> 1 (the harmonic series
// has ratios n/(n+1) -> 1 yet diverges). See RatioTest::exact for the sampling caveat.
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

// ---------------------------------------------------------------------------
// Extended classical battery (Raabe, Kummer, Gauss, limit-comparison, Cauchy
// condensation, Dirichlet, Abel, and the p-series / Bertrand threshold wrappers).
// ---------------------------------------------------------------------------
//
// Each is as honest about exactness as ratio_test: when the test's decisive statistic is a
// CONSTANT RATIONAL across the widely-spread sampled indices it is TAKEN as the exact
// rational limit (`exact` set, `exact_limit` filled) and the verdict follows from it --
// including the boundary cases a test is designed to resolve; otherwise a NUMERICAL finite-n
// estimate is reported and the verdict is inconclusive whenever the estimate is too close to
// the test's threshold to certify a strict inequality. The constant-across-samples inference
// is a sampling determination (see RatioTest::exact): correct for the intended bounded-degree
// rational statistics, though a pathological high-degree crafted sequence could defeat it.
// Nothing here ever reports converges/diverges on a genuinely inconclusive (boundary) limit.

// Raabe's test: l = lim n (a_n/a_{n+1} - 1). l > 1 converges, l < 1 diverges, l == 1
// INCONCLUSIVE (the boundary Raabe cannot resolve). Reuses the RatioTest carrier: `exact` /
// `exact_limit` hold l when it is a constant rational over the samples (e.g. a_n =
// 1/(n(n+1)) gives l = 2 exactly => converges; a_n = 1/n gives l = 1 exactly =>
// inconclusive), otherwise `numeric_limit` holds a finite-n estimate and the verdict is
// inconclusive within a band of 1. NOTE: the RatioTest verdict here is compared to 1 with
// Raabe's (reversed) direction, not the ratio test's L-vs-1 direction.
[[nodiscard]] auto raabe_test(const RationalSequence& a) -> RatioTest;

// Kummer's general test with a positive auxiliary sequence b_n: l = lim (b_n a_n/a_{n+1} -
// b_{n+1}). l > 0 converges. l < 0 diverges ONLY when Sum 1/b_n diverges -- a side condition
// the caller certifies via `one_over_b_diverges`; without it a negative l is reported
// inconclusive (honest: divergence is not established). l == 0 is inconclusive. RatioTest
// carrier: exact l over Q when it is constant on the samples (e.g. a_n = 1/(n(n+1)), b_n = n
// gives l = 1 exactly => converges), otherwise a numeric estimate with an inconclusive band
// about 0. (b_n == 1 recovers the d'Alembert ratio test in the form l = lim(a_n/a_{n+1}-1).)
[[nodiscard]] auto kummer_test(const RationalSequence& a, const RationalSequence& b,
                               bool one_over_b_diverges = false) -> RatioTest;

// Gauss's test. ASSUMES the ratio has the asymptotic form a_n/a_{n+1} = 1 + h/n +
// O(1/n^{1+r}) with r > 0 (an unverifiable precondition on the sequence); it extracts
// h = lim n (a_n/a_{n+1} - 1) and concludes converges iff h > 1, diverges iff h <= 1.
// Unlike Raabe it RESOLVES the boundary h == 1 (to diverges) -- but only on the EXACT path,
// where h is a constant rational (e.g. the harmonic a_n = 1/n gives h = 1 exactly =>
// diverges). On the numeric path a near-1 estimate cannot certify the boundary and is
// reported inconclusive. Shares the RatioTest carrier and the h extraction with raabe_test.
[[nodiscard]] auto gauss_test(const RationalSequence& a) -> RatioTest;

// Limit-comparison test (NUMERICAL): l = lim a_n/b_n. When 0 < l < infinity the series
// Sum a_n shares the behaviour of the reference Sum b_n, so the verdict echoes `b_behaviour`.
// The hypothesis 0 < l < infinity must be EVIDENCED: the ratio is probed at geometrically
// spaced indices and only a settled positive plateau (successive probe ratios near 1) echoes
// b_behaviour. A ratio drifting to 0 or infinity -- which stays finite and positive over any
// window (e.g. q_n = 1/(ln n)^2) -- is NOT accepted and yields inconclusive, as does a
// non-finite/non-positive sampled ratio or samples < 8. `numeric_limit` carries the finite-n
// estimate of l. Sampled over n = 1..samples plus the spaced probes.
[[nodiscard]] auto limit_comparison_test(const RealSequence& a, const RealSequence& b,
                                         Verdict b_behaviour, std::int64_t samples = 200)
    -> NumericTest;

// Cauchy condensation (NUMERICAL): for a_n >= 0 monotonically non-increasing, Sum a_n
// converges iff the condensed series Sum 2^k a_{2^k} converges. The successive condensed-term
// ratios are examined over an asymptotic window: clear geometric decay => converges; terms
// not tending to 0 (flat/growing) => diverges (nth-term test); a SUB-GEOMETRIC decay that is
// neither -- e.g. a logarithmic / Bertrand-rate condensed series like 1/(n (ln n)^p), whose
// condensed terms decay like 1/k^p -- is genuinely undecidable on the sampled window and is
// reported INCONCLUSIVE (never a wrong definite verdict). The monotonicity / non-negativity
// precondition is CHECKED on [1, samples]; a violation there yields domain_error (it is
// assumed, not checked, beyond the sampled range).
[[nodiscard]] auto cauchy_condensation_test(const RealSequence& a, std::int64_t samples = 1000)
    -> Result<Verdict>;

// Dirichlet's test for Sum a_n b_n (NUMERICAL). Certifies CONVERGENCE from two checkable
// hypotheses on the sampled range: the partial sums A_N = Sum_{n<=N} a_n stay bounded, and
// b_n is monotone with b_n -> 0. Both hold => converges; otherwise inconclusive -- the test
// never certifies divergence. Sampled over n = 1..samples.
[[nodiscard]] auto dirichlet_test(const RealSequence& a, const RealSequence& b,
                                  std::int64_t samples = 1000) -> Verdict;

// Abel's test for Sum a_n b_n (NUMERICAL). Certifies CONVERGENCE when Sum a_n converges
// (checkable via a vanishing partial-sum tail) and b_n is monotone and bounded (convergent);
// both hold => converges, otherwise inconclusive. Sampled over n = 1..2*samples.
[[nodiscard]] auto abel_test(const RealSequence& a, const RealSequence& b,
                             std::int64_t samples = 1000) -> Verdict;

// p-series threshold (EXACT over Q): Sum 1/n^p converges iff p > 1, else diverges. A sharp,
// decidable threshold -- never inconclusive.
[[nodiscard]] auto p_series_test(const Rational& p) -> Verdict;

// Bertrand-series threshold (EXACT over Q): Sum_{n>=2} 1/(n (ln n)^p) converges iff p > 1,
// else diverges. The known logarithmic-scale threshold -- sharp and decidable.
[[nodiscard]] auto bertrand_test(const Rational& p) -> Verdict;

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
    // Sample |a_{n+1}/a_n| at WIDELY-SPREAD small indices (kept small so a geometric r^n
    // stays inside int64; a sample that overflows the exact ratio is skipped). If every
    // computable ratio is the SAME exact rational, that constant is taken as the limit L;
    // otherwise fall back to a numerical estimate. Spreading the indices (rather than
    // clustering them at 1..4) means a non-constant statistic that happens to agree on a few
    // adjacent points is still caught: agreement across this spread pins a bounded-degree
    // rational ratio, though a pathological high-degree sequence engineered to match at every
    // sampled index could still defeat the constancy inference (documented on RatioTest).
    std::optional<Rational> constant{};
    bool constant_holds = true;
    std::size_t samples = 0;
    double last_numeric = 0.0;
    bool have_numeric = false;

    for (const std::int64_t k : {1, 2, 3, 4, 6, 8, 12, 16}) {
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
    // Require at least three agreeing spread samples before taking the ratio as a constant.
    if (samples >= 3 && constant_holds && constant.has_value()) {
        result.exact = true;
        result.exact_limit = *constant;
        result.numeric_limit = to_double(*constant);
        result.verdict = verdict_vs_one(*constant);
        return result;
    }

    // NUMERICAL fallback: the ratios are NOT a single constant, so this is not a geometric
    // sequence and a few finite-n samples cannot certify the limit L — in particular they
    // cannot distinguish a genuine L < 1 from L -> 1 (e.g. the harmonic series has ratios
    // n/(n+1) -> 1 yet DIVERGES). Honestly report `inconclusive` rather than a confident,
    // possibly-wrong verdict; only the exact constant-ratio (geometric) path above concludes.
    // The finite-sample estimate is still surfaced in numeric_limit for the caller's inspection.
    result.exact = false;
    result.numeric_limit = have_numeric ? last_numeric : 0.0;
    result.verdict = Verdict::inconclusive;
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
    // Single-sample estimate of the root L = lim |a_n|^{1/n}. Because the root test is
    // genuinely INCONCLUSIVE at L = 1, and a slowly-varying divergent series (e.g. 1/n, whose
    // root n^{-1/n} ~ 0.94 at n=64) sits just below 1, only conclude when the estimate is
    // CLEARLY bounded away from 1 (band 0.1); otherwise report inconclusive rather than a
    // confident, possibly-wrong verdict.
    constexpr double band = 0.1;
    result.verdict = (l < 1.0 - band)   ? Verdict::converges
                     : (l > 1.0 + band) ? Verdict::diverges
                                        : Verdict::inconclusive;
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

// --- extended battery: shared exact/numeric limit extraction ----------------

namespace {

// The outcome of trying to pin down a series-test limit L from an exact rational term(n).
// `exact` means term(n) was a CONSTANT RATIONAL across the widely-spread sample set (as for
// a ratio 1 + c/n whose Raabe/Kummer combination is constant), and that constant is taken as
// the rational limit -- a sampling determination that is correct for the intended bounded-
// degree rational statistics but that a pathological high-degree crafted sequence could
// defeat. Otherwise a NUMERICAL estimate at the largest feasible index is reported.
struct SeqLimit {
    bool exact{false};
    Rational exact_value{};    // the constant rational limit, valid iff `exact`
    double numeric{0.0};       // a double view of L (constant value or finite-n estimate)
    bool have_numeric{false};  // false only when every sample overflowed / was skipped
};

// term: std::int64_t -> Result<Rational>, the exact test statistic at index n. A sample
// that overflows or is undefined (an error) is dropped rather than failing the whole test.
template <typename Term>
[[nodiscard]] auto extract_limit(Term&& term) -> SeqLimit {
    SeqLimit out;
    std::optional<Rational> constant{};
    bool constant_holds = true;
    int samples = 0;
    // Widely-spread indices: agreement here rejects a statistic that merely coincides on a
    // few adjacent points (e.g. h_n = 1 + (n-2)(n-3)(n-4)(n-5)(n-6)/D(n), which equals 1 only
    // on 2..6 -- it differs at 8, 12, 16, 24 and so is correctly NOT taken as constant).
    for (const std::int64_t n : {2, 3, 4, 5, 6, 8, 12, 16, 24}) {
        auto g = term(n);
        if (!g) {
            continue;
        }
        ++samples;
        if (!constant.has_value()) {
            constant = *g;
        } else if (!(*constant == *g)) {
            constant_holds = false;
        }
    }
    // Four or more agreeing spread samples are taken as constant (correct for a bounded-degree
    // rational statistic; a crafted sequence matching at all nine indices could still fool it).
    if (samples >= 4 && constant_holds && constant.has_value()) {
        out.exact = true;
        out.exact_value = *constant;
        out.numeric = to_double(*constant);
        out.have_numeric = true;
        return out;
    }
    // NUMERICAL estimate: the largest index whose exact statistic stays inside int64.
    for (const std::int64_t n : {256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4}) {
        auto g = term(n);
        if (g) {
            out.numeric = to_double(*g);
            out.have_numeric = true;
            break;
        }
    }
    return out;
}

// n (a_n/a_{n+1} - 1), the Raabe / Gauss statistic h_n, exact over Q. Undefined (error)
// when a_{n+1} == 0 or an int64 boundary is crossed.
[[nodiscard]] auto raabe_statistic(const RationalSequence& a, std::int64_t n) -> Result<Rational> {
    const Rational an = a(n);
    const Rational an1 = a(n + 1);
    if (an1.is_zero()) {
        return make_error<Rational>(MathError::domain_error);
    }
    auto ratio = an.divide(an1);  // a_n / a_{n+1}
    if (!ratio) {
        return make_error<Rational>(ratio.error());
    }
    auto shifted = ratio->subtract(Rational::from_int(1));  // - 1
    if (!shifted) {
        return make_error<Rational>(shifted.error());
    }
    return Rational::from_int(n).multiply(*shifted);  // n * (...)
}

// Verdict for a limit compared to 1 with Raabe's direction: l > 1 converges, l < 1
// diverges, l == 1 inconclusive (den > 0 canonical, so the test is num vs den).
[[nodiscard]] auto verdict_raabe(const Rational& l) -> Verdict {
    if (l.numerator() > l.denominator()) {
        return Verdict::converges;
    }
    if (l.numerator() < l.denominator()) {
        return Verdict::diverges;
    }
    return Verdict::inconclusive;
}

// Numeric verdict for a statistic compared to `threshold` with a symmetric dead-band:
// clearly above => `above`, clearly below => `below`, within the band => inconclusive.
[[nodiscard]] auto verdict_banded(double value, double threshold, double band, Verdict above,
                                  Verdict below) -> Verdict {
    if (value > threshold + band) {
        return above;
    }
    if (value < threshold - band) {
        return below;
    }
    return Verdict::inconclusive;
}

}  // namespace

auto raabe_test(const RationalSequence& a) -> RatioTest {
    const SeqLimit lim =
        extract_limit([&a](std::int64_t n) { return raabe_statistic(a, n); });
    RatioTest result;
    if (lim.exact) {
        result.exact = true;
        result.exact_limit = lim.exact_value;
        result.numeric_limit = lim.numeric;
        result.verdict = verdict_raabe(lim.exact_value);  // l>1 conv, l<1 div, l==1 inconc
        return result;
    }
    result.numeric_limit = lim.have_numeric ? lim.numeric : 0.0;
    // A non-constant statistic cannot certify the l == 1 boundary; conclude only when the
    // estimate is clearly on one side of 1 (band 0.1), else honestly inconclusive.
    result.verdict = lim.have_numeric
                         ? verdict_banded(lim.numeric, 1.0, 0.1, Verdict::converges,
                                          Verdict::diverges)
                         : Verdict::inconclusive;
    return result;
}

auto gauss_test(const RationalSequence& a) -> RatioTest {
    const SeqLimit lim =
        extract_limit([&a](std::int64_t n) { return raabe_statistic(a, n); });
    RatioTest result;
    if (lim.exact) {
        result.exact = true;
        result.exact_limit = lim.exact_value;
        result.numeric_limit = lim.numeric;
        // Gauss RESOLVES the boundary on the exact path: converges iff h > 1, else diverges
        // (h == 1 and h < 1 both diverge -- the O(1/n^{1+r}) hypothesis decides h == 1).
        const Rational& h = lim.exact_value;
        result.verdict =
            (h.numerator() > h.denominator()) ? Verdict::converges : Verdict::diverges;
        return result;
    }
    result.numeric_limit = lim.have_numeric ? lim.numeric : 0.0;
    // Numeric path: the h == 1 boundary is not certifiable, so a near-1 estimate stays
    // inconclusive (only a clearly-off-1 estimate concludes).
    result.verdict = lim.have_numeric
                         ? verdict_banded(lim.numeric, 1.0, 0.1, Verdict::converges,
                                          Verdict::diverges)
                         : Verdict::inconclusive;
    return result;
}

auto kummer_test(const RationalSequence& a, const RationalSequence& b,
                 bool one_over_b_diverges) -> RatioTest {
    // k_n = b_n (a_n/a_{n+1}) - b_{n+1}, exact over Q.
    const auto kummer_statistic = [&a, &b](std::int64_t n) -> Result<Rational> {
        const Rational an = a(n);
        const Rational an1 = a(n + 1);
        if (an1.is_zero()) {
            return make_error<Rational>(MathError::domain_error);
        }
        auto ratio = an.divide(an1);  // a_n / a_{n+1}
        if (!ratio) {
            return make_error<Rational>(ratio.error());
        }
        auto scaled = b(n).multiply(*ratio);  // b_n * a_n/a_{n+1}
        if (!scaled) {
            return make_error<Rational>(scaled.error());
        }
        return scaled->subtract(b(n + 1));  // - b_{n+1}
    };
    const SeqLimit lim = extract_limit(kummer_statistic);
    RatioTest result;
    if (lim.exact) {
        result.exact = true;
        result.exact_limit = lim.exact_value;
        result.numeric_limit = lim.numeric;
        const std::int64_t sign = lim.exact_value.numerator();  // den > 0, so sign(l) = sign(num)
        if (sign > 0) {
            result.verdict = Verdict::converges;  // l > 0
        } else if (sign < 0 && one_over_b_diverges) {
            result.verdict = Verdict::diverges;  // l < 0 AND Sum 1/b_n diverges
        } else {
            result.verdict = Verdict::inconclusive;  // l == 0, or l < 0 without the side condition
        }
        return result;
    }
    result.numeric_limit = lim.have_numeric ? lim.numeric : 0.0;
    constexpr double band = 1e-3;
    if (lim.have_numeric && lim.numeric > band) {
        result.verdict = Verdict::converges;
    } else if (lim.have_numeric && lim.numeric < -band && one_over_b_diverges) {
        result.verdict = Verdict::diverges;
    } else {
        result.verdict = Verdict::inconclusive;
    }
    return result;
}

auto limit_comparison_test(const RealSequence& a, const RealSequence& b, Verdict b_behaviour,
                           std::int64_t samples) -> NumericTest {
    NumericTest result;  // inconclusive, statistic 0
    if (samples < 1) {
        return result;
    }
    // The exact ratio q_n = a_n/b_n at index n, provided it is finite and strictly positive.
    const auto ratio_at = [&a, &b](std::int64_t n) -> std::optional<double> {
        const double bn = b(n);
        if (bn == 0.0) {
            return std::nullopt;
        }
        const double q = a(n) / bn;
        if (!std::isfinite(q) || q <= 0.0) {
            return std::nullopt;
        }
        return q;
    };
    // Scan n = 1..samples: any non-finite/non-positive ratio breaks the 0 < l < infinity
    // hypothesis outright, and the last valid ratio is the reported finite-n estimate of l.
    double last = 0.0;
    bool have = false;
    for (std::int64_t n = 1; n <= samples; ++n) {
        const double bn = b(n);
        if (bn == 0.0) {
            continue;
        }
        const double q = a(n) / bn;
        if (!std::isfinite(q) || q <= 0.0) {
            result.numeric_limit = std::isfinite(q) ? q : 0.0;
            result.verdict = Verdict::inconclusive;
            return result;
        }
        last = q;
        have = true;
    }
    if (!have) {
        return result;  // inconclusive, statistic 0
    }
    result.numeric_limit = last;  // estimate of l at the largest sampled index
    // 0 < l < infinity must be EVIDENCED, not merely assumed from bounded spread: a ratio
    // slowly decaying to 0 (e.g. q_n = 1/(ln n)^2) or growing to infinity stays finite and
    // positive over any window, yet its limit is NOT in (0, infinity). Probe q_n at
    // geometrically spaced indices and require the successive probe ratios to PLATEAU near 1
    // (a settled positive limit); a persistent multiplicative trend (probe ratios staying off
    // 1) signals q_n -> 0 or infinity, so the hypothesis fails and the verdict is inconclusive.
    bool settled = false;
    if (samples >= 8) {
        const std::optional<double> p0 = ratio_at(samples);
        const std::optional<double> p1 = ratio_at(samples / 2);
        const std::optional<double> p2 = ratio_at(samples / 4);
        const std::optional<double> p3 = ratio_at(samples / 8);
        if (p0 && p1 && p2 && p3 && *p1 > 0.0 && *p2 > 0.0 && *p3 > 0.0) {
            constexpr double plateau_lo = 0.85;
            constexpr double plateau_hi = 1.15;
            const double r0 = *p0 / *p1;
            const double r1 = *p1 / *p2;
            const double r2 = *p2 / *p3;
            settled = r0 > plateau_lo && r0 < plateau_hi && r1 > plateau_lo && r1 < plateau_hi &&
                      r2 > plateau_lo && r2 < plateau_hi;
        }
    }
    if (settled && (b_behaviour == Verdict::converges || b_behaviour == Verdict::diverges)) {
        result.verdict = b_behaviour;  // 0 < l < infinity evidenced => mirror the reference
    } else {
        result.verdict = Verdict::inconclusive;
    }
    return result;
}

auto cauchy_condensation_test(const RealSequence& a, std::int64_t samples) -> Result<Verdict> {
    if (samples < 2) {
        return make_error<Verdict>(MathError::domain_error);
    }
    // Precondition: a_n >= 0 and non-increasing on the sampled range [1, samples].
    constexpr double mono_eps = 1e-12;
    double prev = a(1);
    if (!(prev >= 0.0)) {
        return make_error<Verdict>(MathError::domain_error);
    }
    for (std::int64_t n = 2; n <= samples; ++n) {
        const double cur = a(n);
        if (!(cur >= 0.0)) {
            return make_error<Verdict>(MathError::domain_error);  // negative term
        }
        if (cur > prev + mono_eps) {
            return make_error<Verdict>(MathError::domain_error);  // not non-increasing
        }
        prev = cur;
    }
    // Condensed series c_k = 2^k a(2^k); Sum a_n and Sum c_k share convergence. A finite
    // sample can only certify the two DECISIVE regimes -- so we examine the successive
    // condensed-term ratios r_k = c_{k+1}/c_k across an asymptotic window:
    //
    //   * every r_k <= conv_ratio (clear geometric decay, as for 1/n^p, p>1, whose condensed
    //     ratio is the constant 2^{1-p} < 1)                                => converges;
    //   * every r_k >= div_ratio  (condensed terms do NOT shrink to 0, as for the harmonic
    //     1/n whose condensed terms are all 1, or a growing 1/n^p, p<1)     => diverges
    //     (nth-term test on the condensed series);
    //   * otherwise -- a SUB-GEOMETRIC decay (r_k -> 1 from below, as for the Bertrand series
    //     1/(n (ln n)^p) whose condensed terms decay like 1/k^p) -- the sampled window cannot
    //     certify either regime                                             => inconclusive.
    //
    // The window is asymptotic (k in [k_lo, k_hi]) so small-index transients don't mislead;
    // 2^k stays inside int64 for k <= 50. This is a NUMERICAL (sampled) determination.
    const auto condensed_term = [&a](std::int64_t k) -> double {
        const std::int64_t idx = static_cast<std::int64_t>(1) << k;  // 2^k
        return static_cast<double>(idx) * a(idx);
    };
    constexpr std::int64_t k_lo = 16;
    constexpr std::int64_t k_hi = 50;
    constexpr double conv_ratio = 0.85;       // per-step decay this strong => geometric-like
    constexpr double div_ratio = 1.0 - 1e-6;  // terms not shrinking => nth-term divergence
    bool all_geometric = true;   // every window ratio <= conv_ratio (clear geometric decay)
    bool none_shrinking = true;  // every window ratio >= div_ratio (terms do not tend to 0)
    bool saw_ratio = false;
    double prev_c = condensed_term(k_lo);
    for (std::int64_t k = k_lo + 1; k <= k_hi; ++k) {
        if (prev_c <= 0.0) {
            // Condensed terms have vanished (rapid decay underflowing to 0) => summable.
            return Verdict::converges;
        }
        const double c = condensed_term(k);
        const double r = c / prev_c;
        saw_ratio = true;
        if (r > conv_ratio) {
            all_geometric = false;
        }
        if (r < div_ratio) {
            none_shrinking = false;
        }
        prev_c = c;
    }
    if (!saw_ratio) {
        return Verdict::inconclusive;
    }
    if (all_geometric) {
        return Verdict::converges;  // condensed terms decay at least geometrically
    }
    if (none_shrinking) {
        return Verdict::diverges;  // condensed terms do not tend to 0 (nth-term test)
    }
    return Verdict::inconclusive;  // sub-geometric decay: undecidable on the sampled window
}

auto dirichlet_test(const RealSequence& a, const RealSequence& b, std::int64_t samples)
    -> Verdict {
    if (samples < 4) {
        return Verdict::inconclusive;
    }
    // Hypothesis 1: partial sums A_N = Sum_{n<=N} a_n stay bounded. Read as the second-half
    // running max not overshooting the first-half max (growing partials => not bounded).
    double partial = 0.0;
    double max_first = 0.0;
    double max_second = 0.0;
    const std::int64_t half = samples / 2;
    for (std::int64_t n = 1; n <= samples; ++n) {
        partial += a(n);
        const double m = std::fabs(partial);
        if (n <= half) {
            max_first = std::max(max_first, m);
        } else {
            max_second = std::max(max_second, m);
        }
    }
    // A bounded partial sum has comparable first-/second-half maxima (ratio ~ 1); linear
    // growth (e.g. a_n = 1, A_N = N) doubles the max across halves. A 1.5x slack rejects the
    // latter while tolerating a bounded sum that approaches its sup late.
    const bool bounded = max_second <= 1.5 * max_first + 1e-9;
    // Hypothesis 2: b_n monotone and b_n -> 0.
    constexpr double eps = 1e-12;
    const double dir = b(2) - b(1);
    bool monotone = true;
    for (std::int64_t n = 1; n < samples; ++n) {
        const double d = b(n + 1) - b(n);
        if (dir >= 0.0 ? (d < -eps) : (d > eps)) {
            monotone = false;
            break;
        }
    }
    const bool to_zero = std::fabs(b(samples)) < 1e-3 * (1.0 + std::fabs(b(1)));
    if (bounded && monotone && to_zero) {
        return Verdict::converges;
    }
    return Verdict::inconclusive;  // hypotheses not certifiable; never certifies divergence
}

auto abel_test(const RealSequence& a, const RealSequence& b, std::int64_t samples) -> Verdict {
    if (samples < 4) {
        return Verdict::inconclusive;
    }
    // Hypothesis 1: Sum a_n converges (Cauchy: the tail A_{2N} - A_N -> 0).
    double a_n = 0.0;
    for (std::int64_t n = 1; n <= samples; ++n) {
        a_n += a(n);
    }
    double a_2n = a_n;
    for (std::int64_t n = samples + 1; n <= 2 * samples; ++n) {
        a_2n += a(n);
    }
    const bool sum_converges = std::fabs(a_2n - a_n) < 1e-3 * (1.0 + std::fabs(a_n));
    // Hypothesis 2: b_n monotone and bounded (convergent) over the sampled range.
    constexpr double eps = 1e-12;
    const double dir = b(2) - b(1);
    bool monotone = true;
    for (std::int64_t n = 1; n < 2 * samples; ++n) {
        const double d = b(n + 1) - b(n);
        if (dir >= 0.0 ? (d < -eps) : (d > eps)) {
            monotone = false;
            break;
        }
    }
    const bool bounded = std::fabs(b(2 * samples) - b(samples)) < 1e-3 * (1.0 + std::fabs(b(1)));
    if (sum_converges && monotone && bounded) {
        return Verdict::converges;
    }
    return Verdict::inconclusive;
}

auto p_series_test(const Rational& p) -> Verdict {
    // Sum 1/n^p converges iff p > 1 (den > 0 canonical, so p > 1 iff num > den), else
    // diverges. Sharp threshold -- exact over Q, never inconclusive.
    return (p.numerator() > p.denominator()) ? Verdict::converges : Verdict::diverges;
}

auto bertrand_test(const Rational& p) -> Verdict {
    // Sum_{n>=2} 1/(n (ln n)^p) converges iff p > 1, else diverges (including p == 1, where
    // Sum 1/(n ln n) diverges). The known logarithmic-scale threshold -- exact over Q.
    return (p.numerator() > p.denominator()) ? Verdict::converges : Verdict::diverges;
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
