// NimbleCAS sequence acceleration / extrapolation (ROADMAP §7).
// @author Olumuyiwa Oluwasanmi
//
// Conforms to config/cpp_details.txt: C++23 modules, `import std`, trailing return
// types, no owning raw pointers, std::expected error handling (no exceptions),
// [[nodiscard]] everywhere.
//
// Richardson extrapolation, Romberg integration, Aitken's Δ² process and Wynn's
// epsilon algorithm are all RATIONAL LINEAR COMBINATIONS of their inputs. The key
// consequence for a CAS is HONESTY about exactness:
//
// ── HONESTY BOUNDARY (documented and enforced by the API split) ─────────────────────
//   * On EXACT rational inputs (a `Rational` sequence, or a rational integrand sampled
//     at rational abscissae) every routine here is EXACT over Q — the outputs are pure
//     reduced fractions produced by add/sub/mul/div of rationals, with NO rounding.
//     These are the `*_exact` / `Rational`-overload paths.
//   * On `double`-valued functions the same algebra runs in IEEE-754 and is therefore
//     NUMERICAL. Extrapolation ACCELERATES convergence ONLY when the assumed asymptotic
//     error expansion actually holds (i.e. the data is smooth); for rough or noisy data
//     it can AMPLIFY error rather than remove it. No universal-acceleration claim is made.
//   * A stalled sequence makes the Aitken/epsilon denominator (Δ², or an ε difference)
//     ZERO. That is reported as MathError::domain_error on the railway — never a wrong
//     result, never a silent NaN.
//
// Every failure travels the railway (Result<T> / MathError); nothing throws. All exact
// paths are overflow-checked through Rational's own guarded arithmetic (Rule 32).

export module nimblecas.extrapolation;

import std;
import nimblecas.core;
import nimblecas.ratpoly;   // Rational — exact reduced int64 fraction

export namespace nimblecas {

// ---------------------------------------------------------------------------
// Function-sampling callbacks.
// ---------------------------------------------------------------------------
// Numerical scalar function R -> R (IEEE-754).
using RealFunction = std::function<double(double)>;
// Exact scalar function Q -> Q, on the railway (so an integrand undefined at a point,
// or one that overflows int64, can report it rather than fabricate a value).
using ExactFunction = std::function<Result<Rational>(const Rational&)>;

// ---------------------------------------------------------------------------
// Extrapolation tableaux (lower-triangular). table[i] has length i+1; column j is
// the j-th extrapolation level and `best` is the fully-extrapolated corner
// table[n-1][n-1].
// ---------------------------------------------------------------------------
struct RationalTableau {
    std::vector<std::vector<Rational>> table;  // EXACT over Q
    Rational best;
};
struct DoubleTableau {
    std::vector<std::vector<double>> table;    // NUMERICAL (IEEE-754)
    double best{};
};

// ===========================================================================
// RICHARDSON EXTRAPOLATION.
// ===========================================================================
// Given A(h) = A + c_1 h^p + c_2 h^{2p} + ... a single step combines A(h) and the
// refined A(h/r) to cancel the leading h^p term:
//     A_new = (r^p A(h/r) - A(h)) / (r^p - 1).
// EXACT over Q (r, p integers -> r^p integer -> a rational linear combination). The
// refinement ratio r must be >= 2 and the leading order p >= 1 (else domain_error).
[[nodiscard]] auto richardson_step(const Rational& a_h, const Rational& a_hr,
                                   std::int64_t r, std::int64_t p) -> Result<Rational>;
// NUMERICAL double form; r may be any real > 1 and p any real > 0 (uses std::pow).
[[nodiscard]] auto richardson_step(double a_h, double a_hr, double r, double p) -> Result<double>;

// Full iterated Richardson/Neville tableau over a sequence of approximations
// a = [A(h_0), A(h_0/r), A(h_0/r^2), ...] (successive refinement by the factor r),
// with successive error orders p, 2p, 3p, ... The recurrence per column j >= 1 is
//     T[i][j] = T[i][j-1] + (T[i][j-1] - T[i-1][j-1]) / (r^{p j} - 1).
// EXACT over Q. Fails with domain_error on an empty sequence, r < 2, or p < 1, and
// propagates Rational overflow.
[[nodiscard]] auto richardson_tableau(std::span<const Rational> a, std::int64_t r,
                                      std::int64_t p) -> Result<RationalTableau>;
// NUMERICAL double form; requires r > 1 and p > 0.
[[nodiscard]] auto richardson_tableau(std::span<const double> a, double r, double p)
    -> Result<DoubleTableau>;

// Richardson-accelerated numerical derivative. Central differences
// D(h) = (f(x+h) - f(x-h)) / (2h) have an error expansion in EVEN powers h^2, h^4, ...
// so the tableau is built with refinement r = 2 and leading order p = 2 on
// D(h_0), D(h_0/2), ..., D(h_0/2^levels). `best` is the extrapolated estimate.
// NUMERICAL: accelerates for smooth f, but very small h eventually loses accuracy to
// cancellation — extrapolating from a moderate h_0 is the point of the method.
[[nodiscard]] auto richardson_derivative(const RealFunction& f, double x, double h0,
                                         std::size_t levels) -> Result<DoubleTableau>;
// EXACT form for a rational function sampled at rational points: for a polynomial (or
// any rational f whose central-difference error expansion terminates) enough levels
// recover the derivative value EXACTLY over Q.
[[nodiscard]] auto richardson_derivative_exact(const ExactFunction& f, const Rational& x,
                                               const Rational& h0, std::size_t levels)
    -> Result<RationalTableau>;

// ===========================================================================
// ROMBERG INTEGRATION — Richardson applied to the composite trapezoidal rule.
// ===========================================================================
// Build R[i][0] = T(h_i), the composite trapezoid on 2^i sub-intervals of [a,b]
// (successive halving, reusing the previous level's points), then Richardson columns
// with r = 2, p = 2 (the Euler–Maclaurin error is even in h). `best` = R[levels][levels].
// The double form is NUMERICAL. Fails with domain_error on b < a.
[[nodiscard]] auto romberg(const RealFunction& f, double a, double b, std::size_t levels)
    -> Result<DoubleTableau>;
// EXACT form: when the integrand is rational-valued at the (rational) trapezoid
// abscissae the whole tableau is EXACT over Q — e.g. ∫_0^1 x^2 dx resolves to 1/3 after
// the first Richardson column (that column is Simpson's rule, exact for cubics). Fails
// with domain_error on b < a and propagates Rational overflow / any error from f.
[[nodiscard]] auto romberg_exact(const ExactFunction& f, const Rational& a, const Rational& b,
                                 std::size_t levels) -> Result<RationalTableau>;

// ===========================================================================
// AITKEN'S Δ² PROCESS — accelerate a linearly-convergent sequence.
// ===========================================================================
// x_n' = x_n - (Δx_n)^2 / (Δ² x_n),  Δx_n = x_{n+1}-x_n,  Δ² x_n = x_{n+2}-2x_{n+1}+x_n.
// Returns the accelerated sequence (length n-2). EXACT over Q. Fails with domain_error
// if the input has fewer than 3 terms, or if any Δ² x_n == 0 (a stalled window — the
// denominator would vanish; reported, never divided).
[[nodiscard]] auto aitken(std::span<const Rational> x) -> Result<std::vector<Rational>>;
// NUMERICAL double form (guards an exact-zero Δ²; a merely tiny Δ² is the noise-
// amplification regime the honesty note warns about, not an error).
[[nodiscard]] auto aitken(std::span<const double> x) -> Result<std::vector<double>>;

// ===========================================================================
// SHANKS TRANSFORMATION / WYNN'S EPSILON ALGORITHM.
// ===========================================================================
// The epsilon table on a sequence of partial sums s = [S_0, S_1, ..., S_{N-1}]:
//   ε_{-1}^{(n)} = 0,  ε_0^{(n)} = S_n,
//   ε_{k+1}^{(n)} = ε_{k-1}^{(n+1)} + 1 / (ε_k^{(n+1)} - ε_k^{(n)}).
// The even-order columns hold the accelerated estimates (ε_2 is the Shanks transform,
// generalising Aitken). Returns the deepest even-column entry — the most accelerated
// value. EXACT over Q. Fails with domain_error if N < 3, or if any ε difference in the
// recurrence is 0 (a stalled sequence — division by zero, reported not performed).
[[nodiscard]] auto wynn_epsilon(std::span<const Rational> s) -> Result<Rational>;
// NUMERICAL double form (guards an exact-zero ε difference).
[[nodiscard]] auto wynn_epsilon(std::span<const double> s) -> Result<double>;

// ===========================================================================
// LEVIN TRANSFORMS (u-, t-, v-).
// ===========================================================================
// Nonlinear sequence transforms built from the partial sums s_n and an explicit
// remainder estimate ω_n. With the sequence viewed as 1-indexed (first partial sum is
// s_1, term Δs_n = s_n - s_{n-1}) the three classic remainder estimates are:
//   * t:  ω_n = Δs_n                              (Levin t-transform),
//   * u:  ω_n = n · Δs_n                          (Levin u-transform),
//   * v:  ω_n = Δs_n · Δs_{n+1} / (Δs_n - Δs_{n+1})   (Levin v-transform).
// Each returns the top-order transform 𝓛_{k}^{(0)} = Σ_j w_j s_j/ω_j / Σ_j w_j/ω_j with
// w_j = (-1)^j C(k,j) (j+1)^{k-1} — a pure RATIONAL LINEAR-FRACTIONAL combination, hence
// EXACT over Q on rational inputs. These are among the most powerful accelerators for
// BOTH linearly and logarithmically convergent series, but like every nonlinear
// transform they carry NO universal-convergence guarantee: on data that violates the
// implied remainder model they may fail to accelerate (or, in double, amplify rounding).
// Fails with domain_error on too-short input (t/u need >= 2 terms, v needs >= 3), a zero
// remainder estimate ω_n, or a vanishing final denominator; propagates Rational overflow.
[[nodiscard]] auto levin_t(std::span<const Rational> s) -> Result<Rational>;
[[nodiscard]] auto levin_u(std::span<const Rational> s) -> Result<Rational>;
[[nodiscard]] auto levin_v(std::span<const Rational> s) -> Result<Rational>;
// NUMERICAL double forms (same contract; the model-violation caveat above applies).
[[nodiscard]] auto levin_t(std::span<const double> s) -> Result<double>;
[[nodiscard]] auto levin_u(std::span<const double> s) -> Result<double>;
[[nodiscard]] auto levin_v(std::span<const double> s) -> Result<double>;

// ===========================================================================
// EULER TRANSFORMATION of an alternating series.
// ===========================================================================
// For an alternating series Σ_{k>=0} (-1)^k a_k with magnitudes a = [a_0, ..., a_{N-1}],
// the Euler transform reweights by iterated forward differences:
//   E = Σ_{k=0}^{N-1} D_k / 2^{k+1},   D_k = Σ_{i=0}^{k} (-1)^i C(k,i) a_i.
// EXACT over Q (only halvings and integer-binomial-weighted rational sums — no division
// by a data-dependent quantity, so it never stalls). Fails with domain_error only on an
// empty input; propagates Rational overflow (large binomials). NUMERICAL in the double
// form. Accelerates alternating series well; it is not meant for same-sign series (use
// van_wijngaarden first).
[[nodiscard]] auto euler_transform(std::span<const Rational> a) -> Result<Rational>;
[[nodiscard]] auto euler_transform(std::span<const double> a) -> Result<double>;

// ===========================================================================
// THETA ALGORITHM (Brezinski).
// ===========================================================================
// θ_{-1}^{(n)} = 0, θ_0^{(n)} = s_n, and
//   θ_{2k+1}^{(n)} = θ_{2k-1}^{(n+1)} + 1 / Δθ_{2k}^{(n)},
//   θ_{2k+2}^{(n)} = θ_{2k}^{(n+1)} + (Δθ_{2k}^{(n+1)} · Δθ_{2k+1}^{(n+1)}) / Δ²θ_{2k+1}^{(n)},
// where Δ acts on the upper index. The even columns hold the accelerated estimates;
// θ returns the deepest even-column entry. More robust than Wynn ε for some sequences,
// but still a nonlinear transform with NO universal-convergence guarantee. EXACT over Q.
// Needs >= 4 terms for θ_2 (else domain_error). A vanishing denominator STOPS the table
// early and returns the best even column already formed; it is a domain_error only if no
// acceleration column (beyond θ_0) could be formed at all.
[[nodiscard]] auto theta(std::span<const Rational> s) -> Result<Rational>;
[[nodiscard]] auto theta(std::span<const double> s) -> Result<double>;

// ===========================================================================
// RHO ALGORITHM (Wynn), for LOGARITHMICALLY convergent sequences.
// ===========================================================================
// ρ_{-1}^{(n)} = 0, ρ_0^{(n)} = s_n, and
//   ρ_{k+1}^{(n)} = ρ_{k-1}^{(n+1)} + (x_{n+k+1} - x_n) / (ρ_k^{(n+1)} - ρ_k^{(n)}),
// with the auxiliary abscissae taken as the index positions x_n = n by default, so the
// step x_{n+k+1} - x_n reduces to the integer (k+1). The even columns hold the estimates;
// ρ returns the deepest even-column entry. Designed for logarithmic convergence (where
// ε/Aitken struggle); still nonlinear, no universal guarantee. EXACT over Q. Needs >= 3
// terms (else domain_error). Same stall policy as theta: a zero denominator stops the
// table and returns the best even column, domain_error only if none was formed.
[[nodiscard]] auto rho(std::span<const Rational> s) -> Result<Rational>;
[[nodiscard]] auto rho(std::span<const double> s) -> Result<double>;

// ===========================================================================
// VAN WIJNGAARDEN TRANSFORMATION.
// ===========================================================================
// Condenses a slowly-convergent series of POSITIVE terms Σ_{n>=1} a_n (input a = [a_1,
// ..., a_N], 1-indexed) into an alternating series Σ_{k>=1} (-1)^{k-1} b_k with the SAME
// sum, where b_k = Σ_{j>=0} 2^j a_{2^j k} (each b_k truncated to the terms available in
// the input). Returns the magnitudes b = [b_1, ..., b_N]; feed them straight into
// euler_transform (whose Σ (-1)^k b_{k+1} matches the (-1)^{k-1} sign pattern). EXACT over
// Q (only powers of two and rational sums — never divides). Fails with domain_error on an
// empty input; propagates overflow if an index 2^j k exceeds int64.
[[nodiscard]] auto van_wijngaarden(std::span<const Rational> a) -> Result<std::vector<Rational>>;
[[nodiscard]] auto van_wijngaarden(std::span<const double> a) -> Result<std::vector<double>>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// base^exp as an EXACT Rational (exp >= 0), overflow-checked through Rational::multiply.
[[nodiscard]] auto rat_int_pow(std::int64_t base, std::int64_t exp) -> Result<Rational> {
    Rational acc = Rational::from_int(1);
    const Rational b = Rational::from_int(base);
    for (std::int64_t i = 0; i < exp; ++i) {
        auto m = acc.multiply(b);
        if (!m) {
            return make_error<Rational>(m.error());
        }
        acc = *m;
    }
    return acc;
}

// Build the Richardson/Neville columns from a column-0 vector, refinement r, order p.
// Shared by richardson_tableau, richardson_derivative_exact and romberg_exact.
[[nodiscard]] auto build_columns_exact(std::vector<Rational> col0, std::int64_t r,
                                       std::int64_t p) -> Result<RationalTableau> {
    const std::size_t n = col0.size();
    if (n == 0 || r < 2 || p < 1) {
        return make_error<RationalTableau>(MathError::domain_error);
    }
    std::vector<std::vector<Rational>> t(n);
    for (std::size_t i = 0; i < n; ++i) {
        t[i].resize(i + 1);
        t[i][0] = col0[i];
    }
    const Rational one = Rational::from_int(1);
    for (std::size_t j = 1; j < n; ++j) {
        // denom = r^{p*j} - 1  (>= 3 for r>=2, p>=1, j>=1; never zero, but guarded).
        const std::int64_t exp = p * static_cast<std::int64_t>(j);
        auto powr = rat_int_pow(r, exp);
        if (!powr) {
            return make_error<RationalTableau>(powr.error());
        }
        auto denom = powr->subtract(one);
        if (!denom) {
            return make_error<RationalTableau>(denom.error());
        }
        if (denom->is_zero()) {
            return make_error<RationalTableau>(MathError::domain_error);
        }
        for (std::size_t i = j; i < n; ++i) {
            auto diff = t[i][j - 1].subtract(t[i - 1][j - 1]);
            if (!diff) {
                return make_error<RationalTableau>(diff.error());
            }
            auto corr = diff->divide(*denom);
            if (!corr) {
                return make_error<RationalTableau>(corr.error());
            }
            auto val = t[i][j - 1].add(*corr);
            if (!val) {
                return make_error<RationalTableau>(val.error());
            }
            t[i][j] = *val;
        }
    }
    Rational best = t[n - 1][n - 1];
    return RationalTableau{.table = std::move(t), .best = best};
}

// Numerical (double) counterpart of build_columns_exact.
[[nodiscard]] auto build_columns_double(std::vector<double> col0, double r, double p)
    -> Result<DoubleTableau> {
    const std::size_t n = col0.size();
    if (n == 0 || !(r > 1.0) || !(p > 0.0)) {
        return make_error<DoubleTableau>(MathError::domain_error);
    }
    std::vector<std::vector<double>> t(n);
    for (std::size_t i = 0; i < n; ++i) {
        t[i].assign(i + 1, 0.0);
        t[i][0] = col0[i];
    }
    for (std::size_t j = 1; j < n; ++j) {
        const double denom = std::pow(r, p * static_cast<double>(j)) - 1.0;
        if (denom == 0.0) {
            return make_error<DoubleTableau>(MathError::domain_error);
        }
        for (std::size_t i = j; i < n; ++i) {
            t[i][j] = t[i][j - 1] + (t[i][j - 1] - t[i - 1][j - 1]) / denom;
        }
    }
    const double best = t[n - 1][n - 1];
    return DoubleTableau{.table = std::move(t), .best = best};
}

}  // namespace

// --- Richardson single step -------------------------------------------------

auto richardson_step(const Rational& a_h, const Rational& a_hr, std::int64_t r,
                     std::int64_t p) -> Result<Rational> {
    if (r < 2 || p < 1) {
        return make_error<Rational>(MathError::domain_error);
    }
    auto factor = rat_int_pow(r, p);  // r^p
    if (!factor) {
        return make_error<Rational>(factor.error());
    }
    // num = r^p * A(h/r) - A(h);  denom = r^p - 1.
    auto scaled = factor->multiply(a_hr);
    if (!scaled) {
        return make_error<Rational>(scaled.error());
    }
    auto num = scaled->subtract(a_h);
    if (!num) {
        return make_error<Rational>(num.error());
    }
    auto denom = factor->subtract(Rational::from_int(1));
    if (!denom) {
        return make_error<Rational>(denom.error());
    }
    if (denom->is_zero()) {
        return make_error<Rational>(MathError::domain_error);
    }
    return num->divide(*denom);
}

auto richardson_step(double a_h, double a_hr, double r, double p) -> Result<double> {
    if (!(r > 1.0) || !(p > 0.0)) {
        return make_error<double>(MathError::domain_error);
    }
    const double factor = std::pow(r, p);
    const double denom = factor - 1.0;
    if (denom == 0.0) {
        return make_error<double>(MathError::domain_error);
    }
    return (factor * a_hr - a_h) / denom;
}

// --- Richardson tableau -----------------------------------------------------

auto richardson_tableau(std::span<const Rational> a, std::int64_t r, std::int64_t p)
    -> Result<RationalTableau> {
    if (a.empty()) {
        return make_error<RationalTableau>(MathError::domain_error);
    }
    return build_columns_exact(std::vector<Rational>(a.begin(), a.end()), r, p);
}

auto richardson_tableau(std::span<const double> a, double r, double p) -> Result<DoubleTableau> {
    if (a.empty()) {
        return make_error<DoubleTableau>(MathError::domain_error);
    }
    return build_columns_double(std::vector<double>(a.begin(), a.end()), r, p);
}

// --- Richardson-accelerated derivative --------------------------------------

auto richardson_derivative(const RealFunction& f, double x, double h0, std::size_t levels)
    -> Result<DoubleTableau> {
    if (h0 == 0.0) {
        return make_error<DoubleTableau>(MathError::domain_error);
    }
    std::vector<double> col0(levels + 1);
    double h = h0;
    for (std::size_t i = 0; i <= levels; ++i) {
        col0[i] = (f(x + h) - f(x - h)) / (2.0 * h);  // central difference D(h)
        h *= 0.5;                                      // refine h -> h/2
    }
    return build_columns_double(std::move(col0), 2.0, 2.0);
}

auto richardson_derivative_exact(const ExactFunction& f, const Rational& x, const Rational& h0,
                                 std::size_t levels) -> Result<RationalTableau> {
    if (h0.is_zero()) {
        return make_error<RationalTableau>(MathError::domain_error);
    }
    if (levels > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        return make_error<RationalTableau>(MathError::overflow);
    }
    std::vector<Rational> col0(levels + 1);
    const Rational two = Rational::from_int(2);
    for (std::size_t i = 0; i <= levels; ++i) {
        // h_i = h0 / 2^i.
        auto pow2 = rat_int_pow(2, static_cast<std::int64_t>(i));
        if (!pow2) {
            return make_error<RationalTableau>(pow2.error());
        }
        auto h = h0.divide(*pow2);
        if (!h) {
            return make_error<RationalTableau>(h.error());
        }
        auto xp = x.add(*h);
        auto xm = x.subtract(*h);
        if (!xp || !xm) {
            return make_error<RationalTableau>(!xp ? xp.error() : xm.error());
        }
        auto fp = f(*xp);
        auto fm = f(*xm);
        if (!fp) {
            return make_error<RationalTableau>(fp.error());
        }
        if (!fm) {
            return make_error<RationalTableau>(fm.error());
        }
        auto num = fp->subtract(*fm);            // f(x+h) - f(x-h)
        if (!num) {
            return make_error<RationalTableau>(num.error());
        }
        auto den = two.multiply(*h);             // 2h
        if (!den) {
            return make_error<RationalTableau>(den.error());
        }
        auto d = num->divide(*den);              // D(h_i)
        if (!d) {
            return make_error<RationalTableau>(d.error());
        }
        col0[i] = *d;
    }
    return build_columns_exact(std::move(col0), 2, 2);
}

// --- Romberg integration ----------------------------------------------------

auto romberg(const RealFunction& f, double a, double b, std::size_t levels)
    -> Result<DoubleTableau> {
    if (b < a) {
        return make_error<DoubleTableau>(MathError::domain_error);
    }
    std::vector<double> col0(levels + 1);
    const double span = b - a;
    col0[0] = 0.5 * span * (f(a) + f(b));  // one-interval trapezoid T(h_0)
    for (std::size_t i = 1; i <= levels; ++i) {
        if (i >= 63) {  // guard BOTH shifts below: 1<<(i-1) and 1<<i (the latter needs i<=62)
            return make_error<DoubleTableau>(MathError::overflow);
        }
        const std::int64_t new_pts = std::int64_t{1} << (i - 1);  // new midpoints added
        const double h = span / static_cast<double>(std::int64_t{1} << i);  // h_i = span/2^i
        double s = 0.0;
        for (std::int64_t k = 1; k <= new_pts; ++k) {
            s += f(a + static_cast<double>(2 * k - 1) * h);
        }
        col0[i] = 0.5 * col0[i - 1] + h * s;  // trapezoid refinement (reuses prior points)
    }
    return build_columns_double(std::move(col0), 2.0, 2.0);
}

auto romberg_exact(const ExactFunction& f, const Rational& a, const Rational& b,
                   std::size_t levels) -> Result<RationalTableau> {
    // b < a is a domain error; compute (b - a) and check its sign.
    auto span_r = b.subtract(a);
    if (!span_r) {
        return make_error<RationalTableau>(span_r.error());
    }
    if (span_r->numerator() < 0) {
        return make_error<RationalTableau>(MathError::domain_error);
    }
    const Rational span = *span_r;
    const Rational two = Rational::from_int(2);

    auto fa = f(a);
    auto fb = f(b);
    if (!fa) {
        return make_error<RationalTableau>(fa.error());
    }
    if (!fb) {
        return make_error<RationalTableau>(fb.error());
    }

    std::vector<Rational> col0(levels + 1);
    {
        // T(h_0) = (span / 2) * (f(a) + f(b)).
        auto fsum = fa->add(*fb);
        if (!fsum) {
            return make_error<RationalTableau>(fsum.error());
        }
        auto half_span = span.divide(two);
        if (!half_span) {
            return make_error<RationalTableau>(half_span.error());
        }
        auto t0 = half_span->multiply(*fsum);
        if (!t0) {
            return make_error<RationalTableau>(t0.error());
        }
        col0[0] = *t0;
    }

    for (std::size_t i = 1; i <= levels; ++i) {
        if (i - 1 >= 62) {  // 2*new_pts-1 must stay within int64
            return make_error<RationalTableau>(MathError::overflow);
        }
        const std::int64_t new_pts = std::int64_t{1} << (i - 1);
        // h_i = span / 2^i.
        auto pow2 = rat_int_pow(2, static_cast<std::int64_t>(i));
        if (!pow2) {
            return make_error<RationalTableau>(pow2.error());
        }
        auto h = span.divide(*pow2);
        if (!h) {
            return make_error<RationalTableau>(h.error());
        }
        // Sum f over the new midpoints x_k = a + (2k-1) h_i, k = 1..2^{i-1}.
        Rational s{};  // 0/1
        for (std::int64_t k = 1; k <= new_pts; ++k) {
            auto coef = Rational::from_int(2 * k - 1);
            auto off = coef.multiply(*h);
            if (!off) {
                return make_error<RationalTableau>(off.error());
            }
            auto xk = a.add(*off);
            if (!xk) {
                return make_error<RationalTableau>(xk.error());
            }
            auto fx = f(*xk);
            if (!fx) {
                return make_error<RationalTableau>(fx.error());
            }
            auto acc = s.add(*fx);
            if (!acc) {
                return make_error<RationalTableau>(acc.error());
            }
            s = *acc;
        }
        // T(h_i) = T(h_{i-1})/2 + h_i * S.
        auto half_prev = col0[i - 1].divide(two);
        if (!half_prev) {
            return make_error<RationalTableau>(half_prev.error());
        }
        auto hs = h->multiply(s);
        if (!hs) {
            return make_error<RationalTableau>(hs.error());
        }
        auto ti = half_prev->add(*hs);
        if (!ti) {
            return make_error<RationalTableau>(ti.error());
        }
        col0[i] = *ti;
    }
    return build_columns_exact(std::move(col0), 2, 2);
}

// --- Aitken's Δ² ------------------------------------------------------------

auto aitken(std::span<const Rational> x) -> Result<std::vector<Rational>> {
    if (x.size() < 3) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    std::vector<Rational> out;
    out.reserve(x.size() - 2);
    for (std::size_t n = 0; n + 2 < x.size(); ++n) {
        auto dx = x[n + 1].subtract(x[n]);          // Δx_n
        if (!dx) {
            return make_error<std::vector<Rational>>(dx.error());
        }
        auto dx1 = x[n + 2].subtract(x[n + 1]);      // Δx_{n+1}
        if (!dx1) {
            return make_error<std::vector<Rational>>(dx1.error());
        }
        auto d2 = dx1->subtract(*dx);                // Δ² x_n = Δx_{n+1} - Δx_n
        if (!d2) {
            return make_error<std::vector<Rational>>(d2.error());
        }
        if (d2->is_zero()) {  // stalled window: denominator vanishes
            return make_error<std::vector<Rational>>(MathError::domain_error);
        }
        auto dx2 = dx->multiply(*dx);                // (Δx_n)^2
        if (!dx2) {
            return make_error<std::vector<Rational>>(dx2.error());
        }
        auto term = dx2->divide(*d2);                // (Δx_n)^2 / Δ² x_n
        if (!term) {
            return make_error<std::vector<Rational>>(term.error());
        }
        auto val = x[n].subtract(*term);             // x_n - term
        if (!val) {
            return make_error<std::vector<Rational>>(val.error());
        }
        out.push_back(*val);
    }
    return out;
}

auto aitken(std::span<const double> x) -> Result<std::vector<double>> {
    if (x.size() < 3) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    std::vector<double> out;
    out.reserve(x.size() - 2);
    for (std::size_t n = 0; n + 2 < x.size(); ++n) {
        const double dx = x[n + 1] - x[n];
        const double d2 = (x[n + 2] - x[n + 1]) - dx;
        if (d2 == 0.0) {
            return make_error<std::vector<double>>(MathError::domain_error);
        }
        out.push_back(x[n] - (dx * dx) / d2);
    }
    return out;
}

// --- Wynn's epsilon algorithm -----------------------------------------------

auto wynn_epsilon(std::span<const Rational> s) -> Result<Rational> {
    const std::size_t N = s.size();
    if (N < 3) {
        return make_error<Rational>(MathError::domain_error);
    }
    // cols[k][n] = ε_k^{(n)}; column k has length N - k. ε_{-1} is the zero column.
    std::vector<std::vector<Rational>> cols;
    cols.reserve(N);
    cols.emplace_back(s.begin(), s.end());  // ε_0 = S
    for (std::size_t k = 1; k < N; ++k) {
        std::vector<Rational> col(N - k);
        for (std::size_t n = 0; n < N - k; ++n) {
            auto denom = cols[k - 1][n + 1].subtract(cols[k - 1][n]);
            if (!denom) {
                return make_error<Rational>(denom.error());
            }
            if (denom->is_zero()) {  // stalled sequence: 1/0 in the recurrence
                return make_error<Rational>(MathError::domain_error);
            }
            auto inv = Rational::from_int(1).divide(*denom);
            if (!inv) {
                return make_error<Rational>(inv.error());
            }
            // ε_{-1} term is 0 for k == 1; otherwise ε_{k-2}^{(n+1)}.
            const Rational prev2 = (k >= 2) ? cols[k - 2][n + 1] : Rational{};
            auto val = prev2.add(*inv);
            if (!val) {
                return make_error<Rational>(val.error());
            }
            col[n] = *val;
        }
        cols.push_back(std::move(col));
    }
    // Deepest EVEN column holds the most-accelerated estimate; entry [0].
    const std::size_t k_best = ((N - 1) % 2 == 0) ? (N - 1) : (N - 2);
    return cols[k_best][0];
}

auto wynn_epsilon(std::span<const double> s) -> Result<double> {
    const std::size_t N = s.size();
    if (N < 3) {
        return make_error<double>(MathError::domain_error);
    }
    std::vector<std::vector<double>> cols;
    cols.reserve(N);
    cols.emplace_back(s.begin(), s.end());
    for (std::size_t k = 1; k < N; ++k) {
        std::vector<double> col(N - k);
        for (std::size_t n = 0; n < N - k; ++n) {
            const double denom = cols[k - 1][n + 1] - cols[k - 1][n];
            if (denom == 0.0) {
                return make_error<double>(MathError::domain_error);
            }
            const double prev2 = (k >= 2) ? cols[k - 2][n + 1] : 0.0;
            col[n] = prev2 + 1.0 / denom;
        }
        cols.push_back(std::move(col));
    }
    const std::size_t k_best = ((N - 1) % 2 == 0) ? (N - 1) : (N - 2);
    return cols[k_best][0];
}

// ===========================================================================
// ADDITIONAL transforms: Levin (u/t/v), Euler, theta, rho, van Wijngaarden.
// ===========================================================================
namespace {

// Exact binomial coefficient C(k, j) as int64, overflow-checked (Rule 32).
[[nodiscard]] auto binom_i64(std::size_t k, std::size_t j) -> Result<std::int64_t> {
    if (j > k) {
        return std::int64_t{0};
    }
    const std::size_t jj = std::min(j, k - j);
    std::int64_t result = 1;
    for (std::size_t i = 0; i < jj; ++i) {
        std::int64_t prod = 0;
        const std::int64_t mul = static_cast<std::int64_t>(k - jj + 1 + i);
        if (__builtin_mul_overflow(result, mul, &prod)) {
            return make_error<std::int64_t>(MathError::overflow);
        }
        result = prod / static_cast<std::int64_t>(i + 1);  // exact: running binomial
    }
    return result;
}

// Numerical binomial (no overflow railway — used only on the double paths).
[[nodiscard]] auto binom_double(std::size_t k, std::size_t j) -> double {
    if (j > k) {
        return 0.0;
    }
    const std::size_t jj = std::min(j, k - j);
    double result = 1.0;
    for (std::size_t i = 0; i < jj; ++i) {
        result = result * static_cast<double>(k - jj + 1 + i) / static_cast<double>(i + 1);
    }
    return result;
}

// Levin core: 𝓛_{L-1}^{(0)} = Σ_j w_j s_j/ω_j / Σ_j w_j/ω_j, w_j = (-1)^j C(k,j)(j+1)^{k-1}.
// EXACT over Q. s and ω must share length L >= 2.
[[nodiscard]] auto levin_core(std::span<const Rational> s, std::span<const Rational> omega)
    -> Result<Rational> {
    const std::size_t L = s.size();
    if (L < 2 || omega.size() != L) {
        return make_error<Rational>(MathError::domain_error);
    }
    const std::size_t k = L - 1;
    Rational num{};
    Rational den{};
    for (std::size_t j = 0; j <= k; ++j) {
        if (omega[j].is_zero()) {  // remainder estimate must be non-zero
            return make_error<Rational>(MathError::domain_error);
        }
        auto bin = binom_i64(k, j);
        if (!bin) {
            return make_error<Rational>(bin.error());
        }
        auto powp = rat_int_pow(static_cast<std::int64_t>(j + 1),
                                static_cast<std::int64_t>(k) - 1);  // (j+1)^{k-1}
        if (!powp) {
            return make_error<Rational>(powp.error());
        }
        auto w = Rational::from_int(*bin).multiply(*powp);
        if (!w) {
            return make_error<Rational>(w.error());
        }
        Rational weight = *w;
        if (j % 2 == 1) {
            auto neg = weight.negate();
            if (!neg) {
                return make_error<Rational>(neg.error());
            }
            weight = *neg;
        }
        auto coeff = weight.divide(omega[j]);  // w_j / ω_j
        if (!coeff) {
            return make_error<Rational>(coeff.error());
        }
        auto ns = coeff->multiply(s[j]);
        if (!ns) {
            return make_error<Rational>(ns.error());
        }
        auto nn = num.add(*ns);
        if (!nn) {
            return make_error<Rational>(nn.error());
        }
        num = *nn;
        auto dd = den.add(*coeff);
        if (!dd) {
            return make_error<Rational>(dd.error());
        }
        den = *dd;
    }
    if (den.is_zero()) {  // degenerate weighting
        return make_error<Rational>(MathError::domain_error);
    }
    return num.divide(den);
}

// Numerical Levin core.
[[nodiscard]] auto levin_core_d(std::span<const double> s, std::span<const double> omega)
    -> Result<double> {
    const std::size_t L = s.size();
    if (L < 2 || omega.size() != L) {
        return make_error<double>(MathError::domain_error);
    }
    const std::size_t k = L - 1;
    double num = 0.0;
    double den = 0.0;
    for (std::size_t j = 0; j <= k; ++j) {
        if (omega[j] == 0.0) {
            return make_error<double>(MathError::domain_error);
        }
        double weight = binom_double(k, j) * std::pow(static_cast<double>(j + 1),
                                                      static_cast<double>(k) - 1.0);
        if (j % 2 == 1) {
            weight = -weight;
        }
        const double coeff = weight / omega[j];
        num += coeff * s[j];
        den += coeff;
    }
    if (den == 0.0) {
        return make_error<double>(MathError::domain_error);
    }
    return num / den;
}

// Forward differences a = [a_0 = s_0, a_i = s_i - s_{i-1}] over the partial sums (exact).
[[nodiscard]] auto exact_terms(std::span<const Rational> s) -> Result<std::vector<Rational>> {
    std::vector<Rational> a(s.size());
    if (!s.empty()) {
        a[0] = s[0];
    }
    for (std::size_t i = 1; i < s.size(); ++i) {
        auto d = s[i].subtract(s[i - 1]);
        if (!d) {
            return make_error<std::vector<Rational>>(d.error());
        }
        a[i] = *d;
    }
    return a;
}

}  // namespace

// --- Levin transforms -------------------------------------------------------

auto levin_t(std::span<const Rational> s) -> Result<Rational> {
    if (s.size() < 2) {
        return make_error<Rational>(MathError::domain_error);
    }
    auto a = exact_terms(s);  // ω_n = Δs_n
    if (!a) {
        return make_error<Rational>(a.error());
    }
    return levin_core(s, *a);
}

auto levin_u(std::span<const Rational> s) -> Result<Rational> {
    if (s.size() < 2) {
        return make_error<Rational>(MathError::domain_error);
    }
    auto a = exact_terms(s);
    if (!a) {
        return make_error<Rational>(a.error());
    }
    std::vector<Rational> omega(s.size());  // ω_n = n · Δs_n (1-indexed) = (i+1) · a_i
    for (std::size_t i = 0; i < s.size(); ++i) {
        auto w = Rational::from_int(static_cast<std::int64_t>(i + 1)).multiply((*a)[i]);
        if (!w) {
            return make_error<Rational>(w.error());
        }
        omega[i] = *w;
    }
    return levin_core(s, omega);
}

auto levin_v(std::span<const Rational> s) -> Result<Rational> {
    const std::size_t N = s.size();
    if (N < 3) {
        return make_error<Rational>(MathError::domain_error);
    }
    auto a = exact_terms(s);
    if (!a) {
        return make_error<Rational>(a.error());
    }
    // ω_i = a_i · a_{i+1} / (a_i - a_{i+1}), i = 0..N-2; paired with s_0..s_{N-2}.
    std::vector<Rational> omega(N - 1);
    for (std::size_t i = 0; i + 1 < N; ++i) {
        auto diff = (*a)[i].subtract((*a)[i + 1]);
        if (!diff) {
            return make_error<Rational>(diff.error());
        }
        if (diff->is_zero()) {
            return make_error<Rational>(MathError::domain_error);
        }
        auto prod = (*a)[i].multiply((*a)[i + 1]);
        if (!prod) {
            return make_error<Rational>(prod.error());
        }
        auto w = prod->divide(*diff);
        if (!w) {
            return make_error<Rational>(w.error());
        }
        omega[i] = *w;
    }
    return levin_core(s.first(N - 1), omega);
}

auto levin_t(std::span<const double> s) -> Result<double> {
    const std::size_t N = s.size();
    if (N < 2) {
        return make_error<double>(MathError::domain_error);
    }
    std::vector<double> a(N);
    a[0] = s[0];
    for (std::size_t i = 1; i < N; ++i) {
        a[i] = s[i] - s[i - 1];
    }
    return levin_core_d(s, a);
}

auto levin_u(std::span<const double> s) -> Result<double> {
    const std::size_t N = s.size();
    if (N < 2) {
        return make_error<double>(MathError::domain_error);
    }
    std::vector<double> omega(N);
    omega[0] = s[0];  // (0+1) · a_0
    for (std::size_t i = 1; i < N; ++i) {
        omega[i] = static_cast<double>(i + 1) * (s[i] - s[i - 1]);
    }
    return levin_core_d(s, omega);
}

auto levin_v(std::span<const double> s) -> Result<double> {
    const std::size_t N = s.size();
    if (N < 3) {
        return make_error<double>(MathError::domain_error);
    }
    std::vector<double> a(N);
    a[0] = s[0];
    for (std::size_t i = 1; i < N; ++i) {
        a[i] = s[i] - s[i - 1];
    }
    std::vector<double> omega(N - 1);
    for (std::size_t i = 0; i + 1 < N; ++i) {
        const double diff = a[i] - a[i + 1];
        if (diff == 0.0) {
            return make_error<double>(MathError::domain_error);
        }
        omega[i] = a[i] * a[i + 1] / diff;
    }
    return levin_core_d(s.first(N - 1), omega);
}

// --- Euler transformation ---------------------------------------------------

auto euler_transform(std::span<const Rational> a) -> Result<Rational> {
    const std::size_t N = a.size();
    if (N == 0) {
        return make_error<Rational>(MathError::domain_error);
    }
    Rational total{};
    for (std::size_t k = 0; k < N; ++k) {
        // D_k = Σ_{i=0}^{k} (-1)^i C(k,i) a_i.
        Rational dk{};
        for (std::size_t i = 0; i <= k; ++i) {
            auto bin = binom_i64(k, i);
            if (!bin) {
                return make_error<Rational>(bin.error());
            }
            auto term = Rational::from_int(*bin).multiply(a[i]);
            if (!term) {
                return make_error<Rational>(term.error());
            }
            Rational t = *term;
            if (i % 2 == 1) {
                auto neg = t.negate();
                if (!neg) {
                    return make_error<Rational>(neg.error());
                }
                t = *neg;
            }
            auto acc = dk.add(t);
            if (!acc) {
                return make_error<Rational>(acc.error());
            }
            dk = *acc;
        }
        // contribution D_k / 2^{k+1}.
        auto pow2 = rat_int_pow(2, static_cast<std::int64_t>(k) + 1);
        if (!pow2) {
            return make_error<Rational>(pow2.error());
        }
        auto contrib = dk.divide(*pow2);
        if (!contrib) {
            return make_error<Rational>(contrib.error());
        }
        auto nt = total.add(*contrib);
        if (!nt) {
            return make_error<Rational>(nt.error());
        }
        total = *nt;
    }
    return total;
}

auto euler_transform(std::span<const double> a) -> Result<double> {
    const std::size_t N = a.size();
    if (N == 0) {
        return make_error<double>(MathError::domain_error);
    }
    double total = 0.0;
    for (std::size_t k = 0; k < N; ++k) {
        double dk = 0.0;
        for (std::size_t i = 0; i <= k; ++i) {
            const double term = binom_double(k, i) * a[i];
            dk += (i % 2 == 1) ? -term : term;
        }
        total += dk / std::pow(2.0, static_cast<double>(k) + 1.0);
    }
    return total;
}

// --- Theta algorithm (Brezinski) --------------------------------------------

auto theta(std::span<const Rational> s) -> Result<Rational> {
    const std::size_t N = s.size();
    if (N < 4) {  // need θ_2, which consumes 4 partial sums
        return make_error<Rational>(MathError::domain_error);
    }
    std::vector<std::vector<Rational>> cols;
    cols.emplace_back(s.begin(), s.end());  // θ_0, length N
    const Rational two = Rational::from_int(2);
    bool stalled = false;
    for (std::size_t j = 1; !stalled; ++j) {
        const std::size_t l_prev = cols[j - 1].size();
        const std::size_t l_prev2 =
            (j >= 2) ? cols[j - 2].size() : std::numeric_limits<std::size_t>::max();
        const std::size_t base = std::min(l_prev, l_prev2);
        const std::size_t shrink = (j % 2 == 1) ? 1U : 2U;
        if (base <= shrink) {
            break;  // no room for another column
        }
        const std::size_t l_new = base - shrink;
        std::vector<Rational> col(l_new);
        for (std::size_t n = 0; n < l_new; ++n) {
            if (j % 2 == 1) {
                // θ_j^{(n)} = θ_{j-2}^{(n+1)} + 1 / (θ_{j-1}^{(n+1)} - θ_{j-1}^{(n)}).
                auto denom = cols[j - 1][n + 1].subtract(cols[j - 1][n]);
                if (!denom) {
                    return make_error<Rational>(denom.error());
                }
                if (denom->is_zero()) {
                    stalled = true;
                    break;
                }
                auto inv = Rational::from_int(1).divide(*denom);
                if (!inv) {
                    return make_error<Rational>(inv.error());
                }
                const Rational prev2 = (j >= 2) ? cols[j - 2][n + 1] : Rational{};
                auto val = prev2.add(*inv);
                if (!val) {
                    return make_error<Rational>(val.error());
                }
                col[n] = *val;
            } else {
                // θ_j^{(n)} = θ_{j-2}^{(n+1)}
                //   + (Δθ_{j-2}^{(n+1)} · Δθ_{j-1}^{(n+1)}) / Δ²θ_{j-1}^{(n)}.
                auto dprev2 = cols[j - 2][n + 2].subtract(cols[j - 2][n + 1]);
                if (!dprev2) {
                    return make_error<Rational>(dprev2.error());
                }
                auto dprev1 = cols[j - 1][n + 2].subtract(cols[j - 1][n + 1]);
                if (!dprev1) {
                    return make_error<Rational>(dprev1.error());
                }
                auto two_mid = two.multiply(cols[j - 1][n + 1]);
                if (!two_mid) {
                    return make_error<Rational>(two_mid.error());
                }
                auto tmp = cols[j - 1][n + 2].subtract(*two_mid);
                if (!tmp) {
                    return make_error<Rational>(tmp.error());
                }
                auto d2 = tmp->add(cols[j - 1][n]);
                if (!d2) {
                    return make_error<Rational>(d2.error());
                }
                if (d2->is_zero()) {
                    stalled = true;
                    break;
                }
                auto prod = dprev2->multiply(*dprev1);
                if (!prod) {
                    return make_error<Rational>(prod.error());
                }
                auto frac = prod->divide(*d2);
                if (!frac) {
                    return make_error<Rational>(frac.error());
                }
                auto val = cols[j - 2][n + 1].add(*frac);
                if (!val) {
                    return make_error<Rational>(val.error());
                }
                col[n] = *val;
            }
        }
        if (stalled) {
            break;  // discard the partial column; keep those already formed
        }
        cols.push_back(std::move(col));
    }
    // Deepest even column that was fully formed; domain_error if only θ_0 exists.
    std::size_t best = 0;
    for (std::size_t c = 0; c < cols.size(); ++c) {
        if (c % 2 == 0 && !cols[c].empty()) {
            best = c;
        }
    }
    if (best == 0) {
        return make_error<Rational>(MathError::domain_error);
    }
    return cols[best][0];
}

auto theta(std::span<const double> s) -> Result<double> {
    const std::size_t N = s.size();
    if (N < 4) {
        return make_error<double>(MathError::domain_error);
    }
    std::vector<std::vector<double>> cols;
    cols.emplace_back(s.begin(), s.end());
    bool stalled = false;
    for (std::size_t j = 1; !stalled; ++j) {
        const std::size_t l_prev = cols[j - 1].size();
        const std::size_t l_prev2 =
            (j >= 2) ? cols[j - 2].size() : std::numeric_limits<std::size_t>::max();
        const std::size_t base = std::min(l_prev, l_prev2);
        const std::size_t shrink = (j % 2 == 1) ? 1U : 2U;
        if (base <= shrink) {
            break;
        }
        const std::size_t l_new = base - shrink;
        std::vector<double> col(l_new);
        for (std::size_t n = 0; n < l_new; ++n) {
            if (j % 2 == 1) {
                const double denom = cols[j - 1][n + 1] - cols[j - 1][n];
                if (denom == 0.0) {
                    stalled = true;
                    break;
                }
                const double prev2 = (j >= 2) ? cols[j - 2][n + 1] : 0.0;
                col[n] = prev2 + 1.0 / denom;
            } else {
                const double dprev2 = cols[j - 2][n + 2] - cols[j - 2][n + 1];
                const double dprev1 = cols[j - 1][n + 2] - cols[j - 1][n + 1];
                const double d2 =
                    cols[j - 1][n + 2] - 2.0 * cols[j - 1][n + 1] + cols[j - 1][n];
                if (d2 == 0.0) {
                    stalled = true;
                    break;
                }
                col[n] = cols[j - 2][n + 1] + (dprev2 * dprev1) / d2;
            }
        }
        if (stalled) {
            break;
        }
        cols.push_back(std::move(col));
    }
    std::size_t best = 0;
    for (std::size_t c = 0; c < cols.size(); ++c) {
        if (c % 2 == 0 && !cols[c].empty()) {
            best = c;
        }
    }
    if (best == 0) {
        return make_error<double>(MathError::domain_error);
    }
    return cols[best][0];
}

// --- Rho algorithm (Wynn) ---------------------------------------------------

auto rho(std::span<const Rational> s) -> Result<Rational> {
    const std::size_t N = s.size();
    if (N < 3) {  // need ρ_2
        return make_error<Rational>(MathError::domain_error);
    }
    std::vector<std::vector<Rational>> cols;
    cols.emplace_back(s.begin(), s.end());  // ρ_0, length N
    bool stalled = false;
    for (std::size_t j = 1; j < N && !stalled; ++j) {
        std::vector<Rational> col(N - j);
        const Rational step = Rational::from_int(static_cast<std::int64_t>(j));  // x_{n+j}-x_n
        for (std::size_t n = 0; n + j < N; ++n) {
            auto denom = cols[j - 1][n + 1].subtract(cols[j - 1][n]);
            if (!denom) {
                return make_error<Rational>(denom.error());
            }
            if (denom->is_zero()) {
                stalled = true;
                break;
            }
            auto frac = step.divide(*denom);
            if (!frac) {
                return make_error<Rational>(frac.error());
            }
            const Rational prev2 = (j >= 2) ? cols[j - 2][n + 1] : Rational{};
            auto val = prev2.add(*frac);
            if (!val) {
                return make_error<Rational>(val.error());
            }
            col[n] = *val;
        }
        if (stalled) {
            break;
        }
        cols.push_back(std::move(col));
    }
    std::size_t best = 0;
    for (std::size_t c = 0; c < cols.size(); ++c) {
        if (c % 2 == 0 && !cols[c].empty()) {
            best = c;
        }
    }
    if (best == 0) {
        return make_error<Rational>(MathError::domain_error);
    }
    return cols[best][0];
}

auto rho(std::span<const double> s) -> Result<double> {
    const std::size_t N = s.size();
    if (N < 3) {
        return make_error<double>(MathError::domain_error);
    }
    std::vector<std::vector<double>> cols;
    cols.emplace_back(s.begin(), s.end());
    bool stalled = false;
    for (std::size_t j = 1; j < N && !stalled; ++j) {
        std::vector<double> col(N - j);
        const double step = static_cast<double>(j);
        for (std::size_t n = 0; n + j < N; ++n) {
            const double denom = cols[j - 1][n + 1] - cols[j - 1][n];
            if (denom == 0.0) {
                stalled = true;
                break;
            }
            const double prev2 = (j >= 2) ? cols[j - 2][n + 1] : 0.0;
            col[n] = prev2 + step / denom;
        }
        if (stalled) {
            break;
        }
        cols.push_back(std::move(col));
    }
    std::size_t best = 0;
    for (std::size_t c = 0; c < cols.size(); ++c) {
        if (c % 2 == 0 && !cols[c].empty()) {
            best = c;
        }
    }
    if (best == 0) {
        return make_error<double>(MathError::domain_error);
    }
    return cols[best][0];
}

// --- Van Wijngaarden transformation -----------------------------------------

auto van_wijngaarden(std::span<const Rational> a) -> Result<std::vector<Rational>> {
    const std::size_t N = a.size();
    if (N == 0) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    constexpr std::int64_t i64_max = std::numeric_limits<std::int64_t>::max();
    const std::int64_t n = static_cast<std::int64_t>(N);
    std::vector<Rational> b(N);
    for (std::int64_t k = 1; k <= n; ++k) {
        // b_k = Σ_{j>=0, 2^j k <= N} 2^j a_{2^j k}.
        Rational bk{};
        std::int64_t p2 = 1;      // 2^j
        std::int64_t idx = k;     // 2^j k
        while (idx <= n) {
            auto term = Rational::from_int(p2).multiply(a[static_cast<std::size_t>(idx) - 1]);
            if (!term) {
                return make_error<std::vector<Rational>>(term.error());
            }
            auto acc = bk.add(*term);
            if (!acc) {
                return make_error<std::vector<Rational>>(acc.error());
            }
            bk = *acc;
            if (p2 > i64_max / 2) {
                break;  // next power of two would overflow
            }
            p2 *= 2;
            if (p2 > i64_max / k) {
                break;  // next index 2^j k would overflow
            }
            idx = p2 * k;
        }
        b[static_cast<std::size_t>(k) - 1] = bk;
    }
    return b;
}

auto van_wijngaarden(std::span<const double> a) -> Result<std::vector<double>> {
    const std::size_t N = a.size();
    if (N == 0) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    constexpr std::int64_t i64_max = std::numeric_limits<std::int64_t>::max();
    const std::int64_t n = static_cast<std::int64_t>(N);
    std::vector<double> b(N);
    for (std::int64_t k = 1; k <= n; ++k) {
        double bk = 0.0;
        std::int64_t p2 = 1;
        std::int64_t idx = k;
        while (idx <= n) {
            bk += static_cast<double>(p2) * a[static_cast<std::size_t>(idx) - 1];
            if (p2 > i64_max / 2) {
                break;
            }
            p2 *= 2;
            if (p2 > i64_max / k) {
                break;
            }
            idx = p2 * k;
        }
        b[static_cast<std::size_t>(k) - 1] = bk;
    }
    return b;
}

}  // namespace nimblecas
