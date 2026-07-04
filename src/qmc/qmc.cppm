// NimbleCAS Quasi-Monte Carlo: low-discrepancy sequences & adaptive integration (ROADMAP 7.8).
// @author Olumuyiwa Oluwasanmi
//
// The low-discrepancy counterpart of nimblecas.montecarlo. Where plain Monte Carlo scatters
// pseudorandom points (RNG from nimblecas.rng) and converges at the dimension-independent
// O(1/sqrt(N)) rate, QMC lays down a *deterministic* low-discrepancy point set whose extra
// uniformity gives, for smooth low-dimensional integrands, the faster O((log N)^d / N) rate.
//
// ── HONESTY BOUNDARY (documented and enforced by the API split) ──────────────────────────
//   * The low-discrepancy POINTS themselves are EXACT. The radical-inverse (Van der Corput)
//     of an integer in base b is an exact reduced fraction; Halton/Hammersley stack those
//     per dimension; Sobol' direction numbers are dyadic so its points are exact multiples of
//     2^-32; a rank-1 lattice point n·z/N mod 1 is an exact fraction with denominator N. Each
//     of those is offered on an EXACT `Rational` path (`*_rational`) plus a `double` view.
//   * QMC INTEGRATION of a general f is NUMERICAL: the equal-weight average of f over the
//     points is a floating-point estimate. When f is itself rational-valued on rational points
//     the average is an exact `Rational` — that path is offered separately (`qmc_integrate_exact`).
//   * The RQMC error is a STATISTICAL (variance) estimate from independent randomizations, NOT
//     a deterministic error bound. The deterministic Koksma–Hlawka bound |error| ≤ V(f)·D*(P)
//     needs the Hardy–Krause total variation V(f) of the integrand, which this module does not
//     compute — so we report the empirical RQMC standard error and say so.
//   * QMC beats plain MC ONLY for integrands that are smooth and of low effective dimension;
//     for rough/high-dimensional f the (log N)^d factor dominates and MC can be competitive or
//     better. No universal-superiority claim is made.
//
// Every failure travels the railway (Result<T> / MathError) — nothing throws. Point routines
// are stateless pure functions of their index (mirroring counter_u64), so any partition of the
// index range reproduces the whole set; the RQMC randomizations are seeded from nimblecas.rng
// so equal seeds reproduce equal results.

export module nimblecas.qmc;

import std;
import nimblecas.core;
import nimblecas.ratpoly;   // Rational — exact reduced int64 fraction
import nimblecas.rng;       // counter-based RNG for RQMC randomization

export namespace nimblecas {

// ---------------------------------------------------------------------------
// Utility: exact Rational -> double view (num/den in IEEE-754). Pure.
// ---------------------------------------------------------------------------
[[nodiscard]] auto rational_to_double(const Rational& r) noexcept -> double;

// ===========================================================================
// LOW-DISCREPANCY POINTS — exact by construction.
// ===========================================================================

// Radical inverse (Van der Corput) of index n in base b, as an EXACT reduced fraction in
// [0,1). phi_b(n) reflects the base-b digits of n about the radix point. Canonical examples
// (base 2): phi(0)=0, phi(1)=1/2, phi(2)=1/4, phi(3)=3/4, phi(4)=1/8, … Fails with
// domain_error if base < 2, and overflow if the denominator b^(#digits) exceeds int64 (use the
// `double` view below for indices whose exact denominator would not fit).
[[nodiscard]] auto van_der_corput_rational(std::uint64_t n, std::uint64_t base) -> Result<Rational>;

// Numerical double view of the same radical inverse. Works for every index (never overflows,
// since it accumulates the fraction directly in floating point). Fails only if base < 2.
[[nodiscard]] auto van_der_corput(std::uint64_t n, std::uint64_t base) -> Result<double>;

// Halton point of index n in `dimension` dimensions, using the first `dimension` primes as the
// per-dimension (pairwise-coprime) bases: coordinate j is phi_{p_j}(n). EXACT rationals. Fails
// with domain_error if dimension == 0, or overflow if any coordinate's denominator exceeds int64.
[[nodiscard]] auto halton_point_rational(std::uint64_t n, std::size_t dimension)
    -> Result<std::vector<Rational>>;
[[nodiscard]] auto halton_point(std::uint64_t n, std::size_t dimension) -> Result<std::vector<double>>;

// Hammersley point i (of a set of size total_n) in `dimension` dimensions: coordinate 0 is the
// EXACT fraction i/total_n, and coordinates 1..d-1 are the Halton coordinates phi_{p_k}(i) with
// the first d-1 primes. Because the first coordinate needs the set size, Hammersley is a finite
// point set (unlike the extensible Halton). Fails with domain_error if dimension == 0 or
// total_n == 0, or overflow on an int64 denominator boundary.
[[nodiscard]] auto hammersley_point_rational(std::uint64_t i, std::uint64_t total_n,
                                             std::size_t dimension) -> Result<std::vector<Rational>>;
[[nodiscard]] auto hammersley_point(std::uint64_t i, std::uint64_t total_n, std::size_t dimension)
    -> Result<std::vector<double>>;

// Sobol' point n (a base-2 digital net built from integer direction numbers) in `dimension`
// dimensions. Points are DYADIC-EXACT: each coordinate is an exact multiple of 2^-32. The
// built-in Joe–Kuo/Bratley–Fox direction-number table covers dimensions 1..8; a larger
// dimension yields domain_error (honest: we do not fabricate direction numbers we cannot
// vouch for). n must be < 2^32 (the 32-bit resolution of the net); a larger n yields
// domain_error. n == 0 maps to the origin.
[[nodiscard]] auto sobol_point_rational(std::uint64_t n, std::size_t dimension)
    -> Result<std::vector<Rational>>;
[[nodiscard]] auto sobol_point(std::uint64_t n, std::size_t dimension) -> Result<std::vector<double>>;

// Rank-1 lattice point n of a set of size total_n with generating vector z (one entry per
// dimension): coordinate j is the EXACT fraction ((n·z_j) mod total_n)/total_n. Fails with
// domain_error if total_n == 0, z.size() != dimension implied (z.size() is the dimension), or
// overflow if total_n > 2^32 (beyond which the modular product would exceed 64 bits).
[[nodiscard]] auto lattice_point_rational(std::uint64_t n, std::uint64_t total_n,
                                          std::span<const std::uint64_t> z)
    -> Result<std::vector<Rational>>;
[[nodiscard]] auto lattice_point(std::uint64_t n, std::uint64_t total_n,
                                 std::span<const std::uint64_t> z) -> Result<std::vector<double>>;

// ===========================================================================
// QMC INTEGRATION.
// ===========================================================================

// Numerical integrand: maps a point in [0,1]^d to a real value.
using ScalarField = std::function<double(std::span<const double>)>;
// Exact integrand: maps an exact rational point to a Result<Rational> (so an integrand that is
// undefined at a point can report it on the railway).
using RationalField = std::function<Result<Rational>(std::span<const Rational>)>;

// Estimate ∫_{[0,1]^d} f as the equal-weight average of f over the first N Halton points
// (indices 1..N; index 0 = origin is skipped). NUMERICAL. Fails with domain_error if
// dimension == 0 or N == 0, and propagates any error from generating a Halton coordinate.
[[nodiscard]] auto qmc_integrate(const ScalarField& f, std::size_t dimension, std::uint64_t N)
    -> Result<double>;

// EXACT variant: when f is rational-valued on the exact Halton points, the equal-weight average
// is an exact Rational (∫ is still only approximated by the finite average — "exact" refers to
// the arithmetic, not to the integral being resolved). Fails with domain_error on dimension/N
// zero, and propagates overflow from the exact rational sum or from f.
[[nodiscard]] auto qmc_integrate_exact(const RationalField& f, std::size_t dimension,
                                       std::uint64_t N) -> Result<Rational>;

// ===========================================================================
// RANDOMIZED QMC (RQMC).
// ===========================================================================

// Result of a randomized-QMC integration: the mean over `replications` independent
// randomizations, an empirical standard error (statistical, not a deterministic bound), and the
// number of integrand evaluations spent.
struct RqmcResult {
    double        estimate;        // mean of the per-replication QMC averages (UNBIASED for ∫ f)
    double        error_estimate;  // standard error of that mean (sqrt(var / replications))
    std::uint64_t points_used;     // total integrand evaluations = N * replications
    std::uint64_t replications;    // number of independent randomizations
};

// Randomized QMC by Cranley–Patterson rotation: for each replication a uniform random shift
// s ∈ [0,1)^d (drawn from a nimblecas.rng child stream) is added mod 1 to every Halton point,
// then f is averaged. Each replication is an UNBIASED estimator of ∫ f, so the mean over
// replications is unbiased and their spread yields the standard error. Fails with domain_error
// if dimension == 0, N == 0, or replications < 2 (two are needed for a variance).
[[nodiscard]] auto rqmc_integrate(const ScalarField& f, std::size_t dimension, std::uint64_t N,
                                  std::uint64_t replications, std::uint64_t seed)
    -> Result<RqmcResult>;

// ===========================================================================
// ITERATIVE / ADAPTIVE QMC.
// ===========================================================================

// Result of an adaptive refinement: the running estimate, the final RQMC standard error, the
// total integrand evaluations spent, and whether the error tolerance was met before the budget.
struct AdaptiveResult {
    double        estimate;
    double        error_estimate;
    std::uint64_t points_used;   // cumulative integrand evaluations across all refinement levels
    bool          converged;     // true iff error_estimate <= tol before the budget ran out
};

// Progressively refine an RQMC estimate by DOUBLING the Halton point count N (the Halton
// sequence is extensible, so each level's points extend the previous level's — earlier points
// are conceptually reused). At each level `replications` randomizations give a running estimate
// and an RQMC standard-error estimate; refinement stops as soon as the error drops to `tol` or
// the cumulative evaluation budget `max_points` is reached. ALWAYS terminates (N doubles and the
// budget is a hard cap). Fails with domain_error if dimension == 0, tol < 0, max_points == 0, or
// replications < 2.
[[nodiscard]] auto adaptive_qmc(const ScalarField& f, std::size_t dimension, double tol,
                                std::uint64_t max_points, std::uint64_t replications,
                                std::uint64_t seed) -> Result<AdaptiveResult>;

// ===========================================================================
// DISCREPANCY DIAGNOSTIC (Warnock L2 star discrepancy).
// ===========================================================================

// L2 star discrepancy of a point set via Warnock's closed form, NUMERICAL (double). A smaller
// value means a more uniform set. Each point must have exactly `dimension` coordinates in [0,1].
// Fails with domain_error on an empty set, dimension == 0, or a point of the wrong size.
[[nodiscard]] auto l2_star_discrepancy(std::span<const std::vector<double>> points,
                                       std::size_t dimension) -> Result<double>;

// EXACT squared L2 star discrepancy of a set of exact rational points (Warnock's formula is a
// finite sum of products of rationals, hence exact). The square root is generally irrational, so
// we return the exact SQUARED value; take rational_to_double(...) then std::sqrt for the norm.
// Fails with domain_error on an empty set / dimension mismatch, or overflow (the products grow
// as denom^(2·dimension) and quickly exceed int64 — the numerical version has no such limit).
[[nodiscard]] auto l2_star_discrepancy_squared_exact(std::span<const std::vector<Rational>> points,
                                                     std::size_t dimension) -> Result<Rational>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

namespace {

// --- small exact helpers ----------------------------------------------------

// base^exp in int64 with overflow guard (base > 0 in all uses here).
[[nodiscard]] auto int_pow_i64(std::int64_t base, std::uint32_t exp) -> Result<std::int64_t> {
    std::int64_t r = 1;
    for (std::uint32_t i = 0; i < exp; ++i) {
        if (r > std::numeric_limits<std::int64_t>::max() / base) {
            return make_error<std::int64_t>(MathError::overflow);
        }
        r *= base;
    }
    return r;
}

// a * b in int64 with overflow guard (both non-negative here).
[[nodiscard]] auto mul_i64(std::int64_t a, std::int64_t b) -> Result<std::int64_t> {
    if (a != 0 && b > std::numeric_limits<std::int64_t>::max() / a) {
        return make_error<std::int64_t>(MathError::overflow);
    }
    return a * b;
}

// max of two rationals on the railway (subtract can overflow).
[[nodiscard]] auto rat_max(const Rational& a, const Rational& b) -> Result<Rational> {
    auto d = a.subtract(b);
    if (!d) {
        return make_error<Rational>(d.error());
    }
    return d->numerator() >= 0 ? a : b;
}

// First `k` primes by trial division (k is small — one per integration dimension).
[[nodiscard]] auto first_primes(std::size_t k) -> std::vector<std::uint64_t> {
    std::vector<std::uint64_t> primes;
    primes.reserve(k);
    for (std::uint64_t cand = 2; primes.size() < k; ++cand) {
        bool prime = true;
        for (const std::uint64_t p : primes) {
            if (p * p > cand) {
                break;
            }
            if (cand % p == 0) {
                prime = false;
                break;
            }
        }
        if (prime) {
            primes.push_back(cand);
        }
    }
    return primes;
}

// --- Sobol' direction numbers (Joe–Kuo / Bratley–Fox), dimensions 1..8 ------
// Each non-trivial dimension carries a primitive-polynomial degree s, the coefficient word a
// (bits a_1..a_{s-1}, MSB-first), and the s initial direction-number multipliers m_1..m_s.
struct SobolSpec {
    std::uint32_t s;                  // polynomial degree
    std::uint32_t a;                  // coefficient bits a_1..a_{s-1}
    std::array<std::uint32_t, 6> m;   // initial m_1..m_s (only first s used)
};

inline constexpr std::uint32_t sobol_word_bits = 32;

// Table entries for dimensions 2..8 (dimension 1 is the pure bit-reversal, handled specially).
inline constexpr std::array<SobolSpec, 7> sobol_table{{
    {1, 0, {1, 0, 0, 0, 0, 0}},   // dim 2
    {2, 1, {1, 3, 0, 0, 0, 0}},   // dim 3
    {3, 1, {1, 3, 1, 0, 0, 0}},   // dim 4
    {3, 2, {1, 1, 1, 0, 0, 0}},   // dim 5
    {4, 1, {1, 1, 3, 3, 0, 0}},   // dim 6
    {4, 4, {1, 3, 5, 13, 0, 0}},  // dim 7
    {5, 2, {1, 1, 5, 5, 17, 0}},  // dim 8
}};

inline constexpr std::size_t sobol_max_dim = 1 + sobol_table.size();  // 8

// Build the 32 direction integers V[1..32] for a 1-indexed dimension.
[[nodiscard]] auto sobol_directions(std::size_t dim) -> Result<std::array<std::uint32_t, 33>> {
    if (dim == 0 || dim > sobol_max_dim) {
        return make_error<std::array<std::uint32_t, 33>>(MathError::domain_error);
    }
    std::array<std::uint32_t, 33> v{};  // index 1..32 used
    if (dim == 1) {
        for (std::uint32_t k = 1; k <= sobol_word_bits; ++k) {
            v[k] = 1U << (sobol_word_bits - k);
        }
        return v;
    }
    const SobolSpec& spec = sobol_table[dim - 2];
    const std::uint32_t s = spec.s;
    for (std::uint32_t k = 1; k <= s; ++k) {
        v[k] = spec.m[k - 1] << (sobol_word_bits - k);
    }
    for (std::uint32_t k = s + 1; k <= sobol_word_bits; ++k) {
        std::uint32_t val = v[k - s] ^ (v[k - s] >> s);
        for (std::uint32_t i = 1; i + 1 <= s; ++i) {  // i = 1..s-1
            const std::uint32_t bit = (spec.a >> (s - 1 - i)) & 1U;
            val ^= bit * v[k - i];
        }
        v[k] = val;
    }
    return v;
}

// The integer Sobol' coordinate (in [0, 2^32)) for index n via the Gray-code construction.
[[nodiscard]] auto sobol_coordinate_u32(std::uint64_t n, std::size_t dim) -> Result<std::uint32_t> {
    auto v = sobol_directions(dim);
    if (!v) {
        return make_error<std::uint32_t>(v.error());
    }
    std::uint32_t gray = static_cast<std::uint32_t>(n ^ (n >> 1));
    std::uint32_t x = 0;
    std::uint32_t bit = 1;
    while (gray != 0) {
        if ((gray & 1U) != 0) {
            x ^= (*v)[bit];
        }
        gray >>= 1;
        ++bit;
    }
    return x;
}

inline constexpr std::int64_t sobol_denom = 4294967296LL;  // 2^32
inline constexpr double sobol_scale = 1.0 / 4294967296.0;  // 2^-32

}  // namespace

// --- Rational -> double -----------------------------------------------------

auto rational_to_double(const Rational& r) noexcept -> double {
    return static_cast<double>(r.numerator()) / static_cast<double>(r.denominator());
}

// --- Van der Corput ---------------------------------------------------------

auto van_der_corput_rational(std::uint64_t n, std::uint64_t base) -> Result<Rational> {
    if (base < 2) {
        return make_error<Rational>(MathError::domain_error);
    }
    constexpr std::uint64_t cap = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    std::uint64_t num = 0;
    std::uint64_t den = 1;
    std::uint64_t m = n;
    while (m > 0) {
        const std::uint64_t digit = m % base;
        m /= base;
        // Guard den *= base and num = num*base + digit against exceeding int64.
        if (den > cap / base) {
            return make_error<Rational>(MathError::overflow);
        }
        den *= base;
        if (num > (cap - digit) / base) {
            return make_error<Rational>(MathError::overflow);
        }
        num = num * base + digit;
    }
    return Rational::make(static_cast<std::int64_t>(num), static_cast<std::int64_t>(den));
}

auto van_der_corput(std::uint64_t n, std::uint64_t base) -> Result<double> {
    if (base < 2) {
        return make_error<double>(MathError::domain_error);
    }
    const double inv_base = 1.0 / static_cast<double>(base);
    double f = 0.0;
    double q = inv_base;
    std::uint64_t m = n;
    while (m > 0) {
        const std::uint64_t digit = m % base;
        m /= base;
        f += static_cast<double>(digit) * q;
        q *= inv_base;
    }
    return f;
}

// --- Halton -----------------------------------------------------------------

auto halton_point_rational(std::uint64_t n, std::size_t dimension) -> Result<std::vector<Rational>> {
    if (dimension == 0) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    const std::vector<std::uint64_t> bases = first_primes(dimension);
    std::vector<Rational> point;
    point.reserve(dimension);
    for (const std::uint64_t base : bases) {
        auto c = van_der_corput_rational(n, base);
        if (!c) {
            return make_error<std::vector<Rational>>(c.error());
        }
        point.push_back(*c);
    }
    return point;
}

auto halton_point(std::uint64_t n, std::size_t dimension) -> Result<std::vector<double>> {
    if (dimension == 0) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    const std::vector<std::uint64_t> bases = first_primes(dimension);
    std::vector<double> point;
    point.reserve(dimension);
    for (const std::uint64_t base : bases) {
        auto c = van_der_corput(n, base);
        if (!c) {
            return make_error<std::vector<double>>(c.error());
        }
        point.push_back(*c);
    }
    return point;
}

// --- Hammersley -------------------------------------------------------------

auto hammersley_point_rational(std::uint64_t i, std::uint64_t total_n, std::size_t dimension)
    -> Result<std::vector<Rational>> {
    if (dimension == 0 || total_n == 0) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    if (i > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
        total_n > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return make_error<std::vector<Rational>>(MathError::overflow);
    }
    std::vector<Rational> point;
    point.reserve(dimension);
    auto first = Rational::make(static_cast<std::int64_t>(i), static_cast<std::int64_t>(total_n));
    if (!first) {
        return make_error<std::vector<Rational>>(first.error());
    }
    point.push_back(*first);
    if (dimension > 1) {
        const std::vector<std::uint64_t> bases = first_primes(dimension - 1);
        for (const std::uint64_t base : bases) {
            auto c = van_der_corput_rational(i, base);
            if (!c) {
                return make_error<std::vector<Rational>>(c.error());
            }
            point.push_back(*c);
        }
    }
    return point;
}

auto hammersley_point(std::uint64_t i, std::uint64_t total_n, std::size_t dimension)
    -> Result<std::vector<double>> {
    if (dimension == 0 || total_n == 0) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    std::vector<double> point;
    point.reserve(dimension);
    point.push_back(static_cast<double>(i) / static_cast<double>(total_n));
    if (dimension > 1) {
        const std::vector<std::uint64_t> bases = first_primes(dimension - 1);
        for (const std::uint64_t base : bases) {
            auto c = van_der_corput(i, base);
            if (!c) {
                return make_error<std::vector<double>>(c.error());
            }
            point.push_back(*c);
        }
    }
    return point;
}

// --- Sobol' -----------------------------------------------------------------

auto sobol_point_rational(std::uint64_t n, std::size_t dimension) -> Result<std::vector<Rational>> {
    if (dimension == 0) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    if (n > 0xFFFFFFFFULL) {  // 32-bit resolution
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    std::vector<Rational> point;
    point.reserve(dimension);
    for (std::size_t d = 1; d <= dimension; ++d) {
        auto x = sobol_coordinate_u32(n, d);
        if (!x) {
            return make_error<std::vector<Rational>>(x.error());
        }
        auto r = Rational::make(static_cast<std::int64_t>(*x), sobol_denom);
        if (!r) {
            return make_error<std::vector<Rational>>(r.error());
        }
        point.push_back(*r);
    }
    return point;
}

auto sobol_point(std::uint64_t n, std::size_t dimension) -> Result<std::vector<double>> {
    if (dimension == 0) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    if (n > 0xFFFFFFFFULL) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    std::vector<double> point;
    point.reserve(dimension);
    for (std::size_t d = 1; d <= dimension; ++d) {
        auto x = sobol_coordinate_u32(n, d);
        if (!x) {
            return make_error<std::vector<double>>(x.error());
        }
        point.push_back(static_cast<double>(*x) * sobol_scale);
    }
    return point;
}

// --- Rank-1 lattice ---------------------------------------------------------

auto lattice_point_rational(std::uint64_t n, std::uint64_t total_n, std::span<const std::uint64_t> z)
    -> Result<std::vector<Rational>> {
    if (total_n == 0 || z.empty()) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    if (total_n > 0x100000000ULL) {  // > 2^32: modular product would exceed 64 bits
        return make_error<std::vector<Rational>>(MathError::overflow);
    }
    std::vector<Rational> point;
    point.reserve(z.size());
    const std::uint64_t n_mod = n % total_n;
    for (const std::uint64_t zj : z) {
        // n_mod < 2^32 and (zj % total_n) < 2^32, so the product fits in uint64.
        const std::uint64_t residue = (n_mod * (zj % total_n)) % total_n;
        auto r = Rational::make(static_cast<std::int64_t>(residue),
                                static_cast<std::int64_t>(total_n));
        if (!r) {
            return make_error<std::vector<Rational>>(r.error());
        }
        point.push_back(*r);
    }
    return point;
}

auto lattice_point(std::uint64_t n, std::uint64_t total_n, std::span<const std::uint64_t> z)
    -> Result<std::vector<double>> {
    auto exact = lattice_point_rational(n, total_n, z);
    if (!exact) {
        return make_error<std::vector<double>>(exact.error());
    }
    std::vector<double> point;
    point.reserve(exact->size());
    for (const Rational& r : *exact) {
        point.push_back(rational_to_double(r));
    }
    return point;
}

// --- QMC integration --------------------------------------------------------

auto qmc_integrate(const ScalarField& f, std::size_t dimension, std::uint64_t N) -> Result<double> {
    if (dimension == 0 || N == 0) {
        return make_error<double>(MathError::domain_error);
    }
    double sum = 0.0;
    for (std::uint64_t n = 1; n <= N; ++n) {  // skip index 0 (origin)
        auto pt = halton_point(n, dimension);
        if (!pt) {
            return make_error<double>(pt.error());
        }
        sum += f(std::span<const double>{*pt});
    }
    return sum / static_cast<double>(N);
}

auto qmc_integrate_exact(const RationalField& f, std::size_t dimension, std::uint64_t N)
    -> Result<Rational> {
    if (dimension == 0 || N == 0) {
        return make_error<Rational>(MathError::domain_error);
    }
    if (N > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return make_error<Rational>(MathError::overflow);
    }
    Rational sum{};  // 0/1
    for (std::uint64_t n = 1; n <= N; ++n) {
        auto pt = halton_point_rational(n, dimension);
        if (!pt) {
            return make_error<Rational>(pt.error());
        }
        auto fv = f(std::span<const Rational>{*pt});
        if (!fv) {
            return make_error<Rational>(fv.error());
        }
        auto s = sum.add(*fv);
        if (!s) {
            return make_error<Rational>(s.error());
        }
        sum = *s;
    }
    return sum.divide(Rational::from_int(static_cast<std::int64_t>(N)));
}

// --- RQMC -------------------------------------------------------------------

auto rqmc_integrate(const ScalarField& f, std::size_t dimension, std::uint64_t N,
                    std::uint64_t replications, std::uint64_t seed) -> Result<RqmcResult> {
    if (dimension == 0 || N == 0 || replications < 2) {
        return make_error<RqmcResult>(MathError::domain_error);
    }

    const Rng base = Rng::seeded(seed);
    std::vector<double> estimates;
    estimates.reserve(replications);

    for (std::uint64_t r = 0; r < replications; ++r) {
        // Independent Cranley–Patterson shift for this replication, from a child RNG stream.
        Rng sub = base.split(r);
        std::vector<double> shift(dimension);
        for (std::size_t j = 0; j < dimension; ++j) {
            shift[j] = sub.next_unit();
        }

        double sum = 0.0;
        std::vector<double> x(dimension);
        for (std::uint64_t n = 1; n <= N; ++n) {
            auto pt = halton_point(n, dimension);
            if (!pt) {
                return make_error<RqmcResult>(pt.error());
            }
            for (std::size_t j = 0; j < dimension; ++j) {
                const double v = (*pt)[j] + shift[j];   // in [0, 2)
                x[j] = v >= 1.0 ? v - 1.0 : v;          // wrap mod 1 into [0, 1)
            }
            sum += f(std::span<const double>{x});
        }
        estimates.push_back(sum / static_cast<double>(N));
    }

    // Mean and standard error of the independent per-replication estimates.
    double mean = 0.0;
    for (const double e : estimates) {
        mean += e;
    }
    mean /= static_cast<double>(replications);

    double sq = 0.0;
    for (const double e : estimates) {
        const double d = e - mean;
        sq += d * d;
    }
    const double variance = sq / static_cast<double>(replications - 1);  // Bessel-corrected
    const double std_error = std::sqrt(variance / static_cast<double>(replications));

    return RqmcResult{mean, std_error, N * replications, replications};
}

// --- Adaptive / iterative ---------------------------------------------------

auto adaptive_qmc(const ScalarField& f, std::size_t dimension, double tol, std::uint64_t max_points,
                  std::uint64_t replications, std::uint64_t seed) -> Result<AdaptiveResult> {
    if (dimension == 0 || tol < 0.0 || max_points == 0 || replications < 2) {
        return make_error<AdaptiveResult>(MathError::domain_error);
    }

    std::uint64_t n = 64;                 // starting point count per replication
    std::uint64_t cumulative = 0;         // total integrand evaluations spent
    double estimate = 0.0;
    double error = std::numeric_limits<double>::infinity();
    bool converged = false;
    std::uint64_t level = 0;

    while (true) {
        // Decorrelate the shifts used at different refinement levels by folding the level into
        // the seed; within a level, rqmc_integrate splits per replication.
        auto r = rqmc_integrate(f, dimension, n, replications, seed + level);
        if (!r) {
            return make_error<AdaptiveResult>(r.error());
        }
        estimate = r->estimate;
        error = r->error_estimate;
        cumulative += r->points_used;

        if (error <= tol) {
            converged = true;
            break;
        }
        if (cumulative >= max_points) {
            break;
        }
        // Guard the doubling against overflow; if we cannot grow further, stop.
        if (n > (std::numeric_limits<std::uint64_t>::max() >> 1)) {
            break;
        }
        n *= 2;
        ++level;
    }

    return AdaptiveResult{estimate, error, cumulative, converged};
}

// --- Discrepancy ------------------------------------------------------------

auto l2_star_discrepancy(std::span<const std::vector<double>> points, std::size_t dimension)
    -> Result<double> {
    const std::size_t N = points.size();
    if (N == 0 || dimension == 0) {
        return make_error<double>(MathError::domain_error);
    }
    for (const std::vector<double>& p : points) {
        if (p.size() != dimension) {
            return make_error<double>(MathError::domain_error);
        }
    }

    // Warnock: D2*^2 = 3^-d - (2^{1-d}/N) Σ_i Π_k (1 - x_ik^2)
    //                        + (1/N^2) Σ_i Σ_j Π_k (1 - max(x_ik, x_jk)).
    double sum1 = 0.0;
    for (const std::vector<double>& p : points) {
        double prod = 1.0;
        for (const double xk : p) {
            prod *= (1.0 - xk * xk);
        }
        sum1 += prod;
    }

    double sum2 = 0.0;
    for (const std::vector<double>& pi : points) {
        for (const std::vector<double>& pj : points) {
            double prod = 1.0;
            for (std::size_t k = 0; k < dimension; ++k) {
                prod *= (1.0 - std::max(pi[k], pj[k]));
            }
            sum2 += prod;
        }
    }

    const double nd = static_cast<double>(N);
    const double d2 = std::pow(3.0, -static_cast<double>(dimension)) -
                      (std::pow(2.0, 1.0 - static_cast<double>(dimension)) / nd) * sum1 +
                      sum2 / (nd * nd);
    return std::sqrt(std::max(d2, 0.0));
}

auto l2_star_discrepancy_squared_exact(std::span<const std::vector<Rational>> points,
                                       std::size_t dimension) -> Result<Rational> {
    const std::size_t N = points.size();
    if (N == 0 || dimension == 0) {
        return make_error<Rational>(MathError::domain_error);
    }
    for (const std::vector<Rational>& p : points) {
        if (p.size() != dimension) {
            return make_error<Rational>(MathError::domain_error);
        }
    }

    const Rational one = Rational::from_int(1);

    // s1 = Σ_i Π_k (1 - x_ik^2)
    Rational s1{};
    for (const std::vector<Rational>& p : points) {
        Rational prod = one;
        for (const Rational& xk : p) {
            auto x2 = xk.multiply(xk);
            if (!x2) {
                return make_error<Rational>(x2.error());
            }
            auto term = one.subtract(*x2);
            if (!term) {
                return make_error<Rational>(term.error());
            }
            auto pr = prod.multiply(*term);
            if (!pr) {
                return make_error<Rational>(pr.error());
            }
            prod = *pr;
        }
        auto acc = s1.add(prod);
        if (!acc) {
            return make_error<Rational>(acc.error());
        }
        s1 = *acc;
    }

    // s2 = Σ_i Σ_j Π_k (1 - max(x_ik, x_jk))
    Rational s2{};
    for (const std::vector<Rational>& pi : points) {
        for (const std::vector<Rational>& pj : points) {
            Rational prod = one;
            for (std::size_t k = 0; k < dimension; ++k) {
                auto mx = rat_max(pi[k], pj[k]);
                if (!mx) {
                    return make_error<Rational>(mx.error());
                }
                auto term = one.subtract(*mx);
                if (!term) {
                    return make_error<Rational>(term.error());
                }
                auto pr = prod.multiply(*term);
                if (!pr) {
                    return make_error<Rational>(pr.error());
                }
                prod = *pr;
            }
            auto acc = s2.add(prod);
            if (!acc) {
                return make_error<Rational>(acc.error());
            }
            s2 = *acc;
        }
    }

    const std::uint32_t d = static_cast<std::uint32_t>(dimension);
    const std::int64_t n_i64 = static_cast<std::int64_t>(N);

    // 3^-d
    auto three_pow = int_pow_i64(3, d);
    if (!three_pow) {
        return make_error<Rational>(three_pow.error());
    }
    auto inv3d = Rational::make(1, *three_pow);
    if (!inv3d) {
        return make_error<Rational>(inv3d.error());
    }

    // coefficient 2^{1-d}/N = 2 / (2^d · N)
    auto two_pow = int_pow_i64(2, d);
    if (!two_pow) {
        return make_error<Rational>(two_pow.error());
    }
    auto denom2 = mul_i64(*two_pow, n_i64);
    if (!denom2) {
        return make_error<Rational>(denom2.error());
    }
    auto coeff2 = Rational::make(2, *denom2);
    if (!coeff2) {
        return make_error<Rational>(coeff2.error());
    }
    auto term2 = coeff2->multiply(s1);
    if (!term2) {
        return make_error<Rational>(term2.error());
    }

    // 1/N^2
    auto n2 = mul_i64(n_i64, n_i64);
    if (!n2) {
        return make_error<Rational>(n2.error());
    }
    auto invn2 = Rational::make(1, *n2);
    if (!invn2) {
        return make_error<Rational>(invn2.error());
    }
    auto term3 = invn2->multiply(s2);
    if (!term3) {
        return make_error<Rational>(term3.error());
    }

    // 3^-d - term2 + term3
    auto d2a = inv3d->subtract(*term2);
    if (!d2a) {
        return make_error<Rational>(d2a.error());
    }
    return d2a->add(*term3);
}

}  // namespace nimblecas
