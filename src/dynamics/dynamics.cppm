// NimbleCAS dynamical-systems stability (ROADMAP 7.10).
// @author Olumuyiwa Oluwasanmi
//
// Exact stability analysis of linear/affine autonomous systems dx/dt = A x (+ b),
// built entirely on top of the rational eigen substrate. Three entry points:
//
//   fixed_point_affine(A, b)     -- the equilibrium of x |-> A x + b, i.e. the exact
//                                   solution of (I - A) x = b over Q.
//   is_asymptotically_stable(A)  -- true iff every eigenvalue of A has strictly negative
//                                   real part, decided EXACTLY from the characteristic
//                                   polynomial via the Routh-Hurwitz criterion. Because it
//                                   works on coefficients (no root finding), it settles the
//                                   irrational/complex spectra that rational-root testing
//                                   over Q cannot -- e.g. the pure imaginary pair of a
//                                   rotation. That is precisely why Routh-Hurwitz is used.
//   classify_equilibrium(A)      -- a stable, human-readable classification string.
//   classify_phase_portrait(A)   -- the full 2x2 trace-determinant phase-portrait verdict
//                                   (node/saddle/center/spiral/star/degenerate), decided
//                                   EXACTLY from the signs of the rational trace T, det D
//                                   and discriminant delta = T^2 - 4D, plus whether delta
//                                   is a perfect rational square. Eigenvalues are reported
//                                   exactly when rational and otherwise left honestly
//                                   symbolic -- never decimalised.
//   classify_linear_stability(A) -- a coarser nD (n >= 1) verdict (sink/source/saddle/
//                                   borderline) from the Routh-Hurwitz first-column
//                                   sign-change count. It does NOT claim a full spiral/node
//                                   taxonomy in nD -- that needs eigenvalues -- and reports
//                                   the imaginary-axis case honestly as `borderline`.
//
// Everything stays inside exact Rational arithmetic; every fallible step is threaded on
// the Result railway and dimension violations surface as MathError::domain_error.

export module nimblecas.dynamics;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.eigen;

export namespace nimblecas {

// The equilibrium (fixed point) of the affine map x |-> A x + b, i.e. the exact solution
// of (I - A) x = b. `A` must be square (n x n) and `b` an n x 1 column matrix, otherwise
// domain_error. A non-isolated equilibrium -- where (I - A) is singular -- propagates as
// domain_error from the underlying solve. On success the returned n x 1 column is x.
[[nodiscard]] auto fixed_point_affine(const Matrix& a, const Matrix& b) -> Result<Matrix>;

// Whether the continuous-time system dx/dt = A x is asymptotically stable: true iff every
// eigenvalue of A has strictly negative real part (Hurwitz). Decided EXACTLY from the
// characteristic polynomial of A via the Routh-Hurwitz criterion -- the Routh array is
// built with exact Rational arithmetic and the system is Hurwitz-stable iff every entry
// of the array's first column is nonzero and shares the leading term's (positive) sign.
// No root finding is involved, so this decides irrational/complex spectra too.
//
// Edge cases, both of which mean at least one root lies on or right of the imaginary axis
// and so are reported as NOT asymptotically stable (false): a zero in the first column
// (whether or not the rest of that row is nonzero), and a fully-zero row (imaginary-axis
// roots). Requires a square matrix with n >= 1, else domain_error.
[[nodiscard]] auto is_asymptotically_stable(const Matrix& a) -> Result<bool>;

// A human-readable classification of the equilibrium at the origin of dx/dt = A x.
//
// When ALL eigenvalues are rational (their multiplicities from rational_eigenvalues sum to
// n) the sign pattern gives an exact verdict. A positive eigenvalue means instability
// regardless of the rest, so positive dominates a zero (marginal) eigenvalue:
//   - both a positive and a negative eigenvalue -> "saddle"
//   - all non-negative, some positive, no zero  -> "unstable node"
//   - all non-negative, some positive, a zero   -> "unstable (with marginal direction)"
//   - all non-positive, some negative, no zero  -> "stable node"
//   - all non-positive, some negative, a zero   -> "marginally stable"
//   - all eigenvalues zero (or n == 0)          -> "degenerate/marginal"
//
// Otherwise part of the spectrum is irrational/complex and cannot be seen over Q, so the
// verdict falls back to the Routh-Hurwitz result:
//   - asymptotically stable            -> "asymptotically stable (spiral/node)"
//   - not asymptotically stable        -> "unstable or marginal (non-rational spectrum)"
// (Rational eigenvalues cannot detect complex conjugate pairs, so "center"/"focus" is
// never claimed from them.) Requires a square matrix, else domain_error.
[[nodiscard]] auto classify_equilibrium(const Matrix& a) -> Result<std::string>;

// ---------------------------------------------------------------------------
// Exact phase-portrait classification of the origin of dx/dt = A x.
// ---------------------------------------------------------------------------

// Whether the rational r is the exact square of a rational: r >= 0 and, in lowest terms
// r = p/q (q > 0), both p and q are perfect squares of integers (then sqrt(r) = sqrt(p)/
// sqrt(q) is rational). EXACT over Q: an internal integer sqrt is only a search seed --
// every candidate is confirmed by exact overflow-guarded multiplication -- so no floating
// point ever decides the verdict. A negative r is never a (real) square.
[[nodiscard]] auto is_perfect_square(const Rational& r) -> bool;

// The qualitative type of a 2x2 equilibrium at the origin, read off the trace-determinant
// plane. Let T = tr(A), D = det(A), delta = T^2 - 4D (all exact rationals):
enum class PhaseType {
    saddle,           // D < 0: real eigenvalues of opposite sign
    node,             // D > 0, delta > 0: real distinct eigenvalues of one common sign
    spiral,           // D > 0, delta < 0, T != 0: complex pair with nonzero real part (focus)
    center,           // D > 0, delta < 0, T == 0: purely imaginary pair
    star,             // D > 0, delta == 0, A = lambda*I: repeated eigenvalue, diagonalizable
    degenerate_node,  // D > 0, delta == 0, A defective: repeated eigenvalue, one eigenvector
    non_isolated,     // D == 0: a line (or plane) of equilibria -- not isolated
};

// The stability verdict attached to a classification.
enum class Stability {
    stable,            // asymptotically stable: every eigenvalue has negative real part
    unstable,          // at least one eigenvalue has positive real part
    neutrally_stable,  // a center: purely imaginary pair, orbits neither grow nor decay
    marginal,          // non-isolated / borderline: bounded but not asymptotically stable
};

// The exact 2x2 phase-portrait verdict. T, D and delta are exact rationals. Eigenvalues
// are reported exactly ONLY when they are rational; otherwise the corresponding optionals
// are left empty rather than decimalised (the honesty boundary of Rule 32):
//   - real eigenvalues (delta >= 0): lambda1 = (T - sqrt(delta))/2, lambda2 = (T + sqrt(
//     delta))/2 are filled iff delta is a perfect square (lambda1 == lambda2 when delta==0).
//   - complex eigenvalues (delta < 0): the pair is real_part +/- imag_part*i; real_part =
//     T/2 is always exact, while imag_part = sqrt(-delta)/2 is filled iff -delta is a
//     perfect square. `eigenvalues_rational` is true exactly when the filled data is complete.
struct PhasePortrait {
    PhaseType type{};
    Stability stability{};
    Rational trace{};          // T = tr(A)
    Rational determinant{};    // D = det(A)
    Rational discriminant{};   // delta = T^2 - 4D
    bool complex_eigenvalues{};   // delta < 0
    bool repeated_eigenvalue{};   // delta == 0
    bool eigenvalues_rational{};  // exact rational eigenvalue data is present below
    std::optional<Rational> lambda1{};    // real case: (T - sqrt(delta))/2
    std::optional<Rational> lambda2{};    // real case: (T + sqrt(delta))/2
    std::optional<Rational> real_part{};  // complex case: T/2 (always exact)
    std::optional<Rational> imag_part{};  // complex case: sqrt(-delta)/2 when rational
    std::string description{};            // a human-readable one-liner
};

// The full 2x2 phase-portrait classification of dx/dt = A x at the origin. Requires a
// 2x2 matrix (square with n == 2), else domain_error. Everything is decided from the exact
// signs of T, D, delta and from is_perfect_square(delta); no root finding, no floating point.
[[nodiscard]] auto classify_phase_portrait(const Matrix& a) -> Result<PhasePortrait>;

// The coarse nD (n >= 1) stability class of dx/dt = A x, from the characteristic
// polynomial's Routh-Hurwitz first-column sign-change count.
enum class LinearStability {
    sink,        // every eigenvalue has Re < 0 (asymptotically stable)
    source,      // every eigenvalue has Re > 0
    saddle,      // mixed: some Re > 0 and some Re < 0, none on the imaginary axis
    borderline,  // an eigenvalue on (or a root configuration symmetric about) the imaginary
                 // axis: the Routh array degenerates and the precise sink/source/saddle
                 // verdict is NOT decidable from the array alone -- resolving it needs the
                 // actual (possibly irrational/complex) eigenvalues.
};

struct StabilityClassification {
    std::size_t dimension{};
    LinearStability verdict{};
    std::int64_t rhp_count{};      // # eigenvalues with Re > 0 when known, else -1 (borderline)
    bool asymptotically_stable{};  // verdict == sink
};

// Classify dx/dt = A x coarsely for any n >= 1. Requires a square matrix with n >= 1, else
// domain_error. This is EXACT for the hyperbolic (regular Routh array) case: sink/source/
// saddle are decided from the exact sign-change count. When the array degenerates -- the
// imaginary-axis / symmetric-root case -- it reports `borderline` honestly rather than
// guessing a center-vs-defective-vs-unstable verdict that needs the actual spectrum. It
// does NOT attempt the full spiral/node/star taxonomy in nD; use classify_phase_portrait
// for the detailed 2x2 verdict.
[[nodiscard]] auto classify_linear_stability(const Matrix& a) -> Result<StabilityClassification>;

// Human-readable names for the classification enums.
[[nodiscard]] auto to_string(PhaseType t) -> std::string_view;
[[nodiscard]] auto to_string(Stability s) -> std::string_view;
[[nodiscard]] auto to_string(LinearStability s) -> std::string_view;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// -1, 0, +1 for a Rational's sign (denominator is kept positive in canonical form, so the
// numerator's sign is the value's sign).
[[nodiscard]] auto sign_of(const Rational& r) -> int {
    if (r.numerator() > 0) {
        return 1;
    }
    if (r.numerator() < 0) {
        return -1;
    }
    return 0;
}

// One Routh-array entry: (pivot * up_next - up0 * cur_next) / pivot, all exact. `pivot` is
// the first-column entry of the row above (known nonzero at the call site). Any Rational
// overflow/division error is propagated.
[[nodiscard]] auto routh_entry(const Rational& pivot, const Rational& up_next,
                               const Rational& up0, const Rational& cur_next)
    -> Result<Rational> {
    auto lhs = pivot.multiply(up_next);
    if (!lhs) {
        return make_error<Rational>(lhs.error());
    }
    auto rhs = up0.multiply(cur_next);
    if (!rhs) {
        return make_error<Rational>(rhs.error());
    }
    auto num = lhs->subtract(*rhs);
    if (!num) {
        return make_error<Rational>(num.error());
    }
    return num->divide(pivot);
}

}  // namespace

auto fixed_point_affine(const Matrix& a, const Matrix& b) -> Result<Matrix> {
    if (!a.is_square()) {
        return make_error<Matrix>(MathError::domain_error);
    }
    if (b.rows() != a.rows() || b.cols() != 1) {
        return make_error<Matrix>(MathError::domain_error);
    }

    // Solve (I - A) x = b. A singular (I - A) -- a non-isolated equilibrium -- surfaces as
    // domain_error from solve, which we simply propagate.
    const Matrix id = Matrix::identity(a.rows());
    auto shifted = id.subtract(a);
    if (!shifted) {
        return make_error<Matrix>(shifted.error());
    }
    return shifted->solve(b);
}

auto is_asymptotically_stable(const Matrix& a) -> Result<bool> {
    if (!a.is_square() || a.rows() < 1) {
        return make_error<bool>(MathError::domain_error);
    }

    auto poly = characteristic_polynomial(a);
    if (!poly) {
        return make_error<bool>(poly.error());
    }
    const std::size_t n = a.rows();  // characteristic polynomial is monic of degree n

    // Coefficients in DESCENDING order: a_desc[i] is the coefficient of lambda^(n-i), so
    // a_desc[0] is the (monic, = 1) leading coefficient and a_desc[n] the constant term.
    std::vector<Rational> a_desc(n + 1, Rational::from_int(0));
    for (std::size_t i = 0; i <= n; ++i) {
        a_desc[i] = poly->coefficient(n - i);
    }

    // Routh array: n + 1 rows, each `cols` wide (padded with zeros). Row 0 holds the even-
    // indexed coefficients, row 1 the odd-indexed ones.
    const std::size_t cols = n / 2 + 1;
    std::vector<std::vector<Rational>> table(n + 1,
                                             std::vector<Rational>(cols, Rational::from_int(0)));
    for (std::size_t j = 0; j < cols; ++j) {
        if (const std::size_t e = 2 * j; e <= n) {
            table[0][j] = a_desc[e];
        }
        if (const std::size_t o = 2 * j + 1; o <= n) {
            table[1][j] = a_desc[o];
        }
    }

    // Fill rows 2..n. A zero pivot (first-column entry of the row above) means either a
    // zero in the first column or a fully-zero row -- both put a root on or right of the
    // imaginary axis, so the system is not asymptotically stable.
    for (std::size_t r = 2; r <= n; ++r) {
        const Rational pivot = table[r - 1][0];
        if (pivot.is_zero()) {
            return false;
        }
        for (std::size_t j = 0; j < cols; ++j) {
            const Rational up_next = (j + 1 < cols) ? table[r - 2][j + 1] : Rational::from_int(0);
            const Rational cur_next = (j + 1 < cols) ? table[r - 1][j + 1] : Rational::from_int(0);
            auto entry = routh_entry(pivot, up_next, table[r - 2][0], cur_next);
            if (!entry) {
                return make_error<bool>(entry.error());
            }
            table[r][j] = *entry;
        }
    }

    // Hurwitz iff every first-column entry is nonzero and shares the leading (positive)
    // sign. A zero here (e.g. a fully-zero final row that was never a pivot) is a failure.
    const int reference = sign_of(table[0][0]);  // +1, since the polynomial is monic
    for (std::size_t r = 0; r <= n; ++r) {
        const int s = sign_of(table[r][0]);
        if (s == 0 || s != reference) {
            return false;
        }
    }
    return true;
}

auto classify_equilibrium(const Matrix& a) -> Result<std::string> {
    if (!a.is_square()) {
        return make_error<std::string>(MathError::domain_error);
    }
    const std::size_t n = a.rows();

    auto eigenvalues = rational_eigenvalues(a);
    if (!eigenvalues) {
        return make_error<std::string>(eigenvalues.error());
    }

    // Do the rational eigenvalues account for the whole spectrum (multiplicities sum to n)?
    std::int64_t total_multiplicity = 0;
    bool has_negative = false;
    bool has_positive = false;
    bool has_zero = false;
    for (const auto& [value, multiplicity] : *eigenvalues) {
        total_multiplicity += multiplicity;
        switch (sign_of(value)) {
            case -1: has_negative = true; break;
            case 1:  has_positive = true; break;
            default: has_zero = true; break;
        }
    }

    if (total_multiplicity == static_cast<std::int64_t>(n)) {
        // Fully rational spectrum: classify by sign pattern. Any eigenvalue with positive
        // real part makes the origin UNSTABLE regardless of the others, so the positive
        // cases are decided before the zero (marginal) case — otherwise diag(0, +1), which
        // grows like e^t, would be mislabelled "marginal".
        if (has_positive && has_negative) {
            return std::string("saddle");  // stable and unstable directions coexist
        }
        if (has_positive) {
            // All non-negative with at least one positive: unstable, possibly with a
            // marginal (zero-eigenvalue) direction alongside the growing one.
            return has_zero ? std::string("unstable (with marginal direction)")
                            : std::string("unstable node");
        }
        if (has_negative) {
            // All non-positive with at least one negative: a zero eigenvalue makes it only
            // marginally (not asymptotically) stable; otherwise a genuine stable node.
            return has_zero ? std::string("marginally stable")
                            : std::string("stable node");
        }
        // No positive and no negative eigenvalue: the spectrum is all zeros (or empty for
        // n == 0) — no exponential motion, a fully degenerate/marginal equilibrium.
        return std::string("degenerate/marginal");
    }

    // Irrational/complex part of the spectrum: defer to the exact Routh-Hurwitz verdict.
    auto stable = is_asymptotically_stable(a);
    if (!stable) {
        return make_error<std::string>(stable.error());
    }
    return *stable ? std::string("asymptotically stable (spiral/node)")
                   : std::string("unstable or marginal (non-rational spectrum)");
}

// ---------------------------------------------------------------------------
// Phase-portrait classification internals.
// ---------------------------------------------------------------------------
namespace {

// Integer square root that yields the exact root ONLY when n (>= 0) is a perfect square.
// A double sqrt seeds the search; every candidate is confirmed by exact overflow-guarded
// multiplication, so the +/-2 window around the seed makes the verdict exact regardless of
// the double's rounding.
[[nodiscard]] auto isqrt_exact(std::int64_t n) -> std::optional<std::int64_t> {
    if (n < 0) {
        return std::nullopt;
    }
    if (n == 0) {
        return std::int64_t{0};
    }
    const auto seed = static_cast<std::int64_t>(std::sqrt(static_cast<double>(n)));
    const std::int64_t lo = seed > 2 ? seed - 2 : 0;
    for (std::int64_t c = lo; c <= seed + 2; ++c) {
        std::int64_t sq = 0;
        if (__builtin_mul_overflow(c, c, &sq)) {
            continue;
        }
        if (sq == n) {
            return c;
        }
    }
    return std::nullopt;
}

// The exact square root of a rational, when it is itself rational. r = p/q in lowest terms
// (q > 0) is a rational square iff p >= 0 and both p and q are perfect squares; then
// sqrt(r) = sqrt(p)/sqrt(q). Returns nullopt otherwise (r < 0, or an irrational root).
[[nodiscard]] auto exact_sqrt(const Rational& r) -> std::optional<Rational> {
    if (r.numerator() < 0) {
        return std::nullopt;
    }
    const auto sn = isqrt_exact(r.numerator());
    const auto sd = isqrt_exact(r.denominator());
    if (!sn || !sd) {
        return std::nullopt;
    }
    auto q = Rational::make(*sn, *sd);  // *sd >= 1, so never division_by_zero
    if (!q) {
        return std::nullopt;  // a (physically unreachable) overflow in re-normalisation
    }
    return *q;
}

// The number of characteristic roots with strictly positive real part, from the Routh-
// Hurwitz first-column sign-change count. `regular` is false when the array degenerates
// (a zero in the first column, or a fully-zero row) -- the imaginary-axis / symmetric-root
// case the plain array cannot resolve -- in which case `rhp` carries no meaning.
struct RouthCount {
    bool regular;
    std::int64_t rhp;
};

[[nodiscard]] auto routh_rhp_count(const Matrix& a) -> Result<RouthCount> {
    auto poly = characteristic_polynomial(a);
    if (!poly) {
        return make_error<RouthCount>(poly.error());
    }
    const std::size_t n = a.rows();  // characteristic polynomial is monic of degree n

    // Coefficients in descending order: a_desc[i] is the coefficient of lambda^(n-i).
    std::vector<Rational> a_desc(n + 1, Rational::from_int(0));
    for (std::size_t i = 0; i <= n; ++i) {
        a_desc[i] = poly->coefficient(n - i);
    }

    const std::size_t cols = n / 2 + 1;
    std::vector<std::vector<Rational>> table(n + 1,
                                             std::vector<Rational>(cols, Rational::from_int(0)));
    for (std::size_t j = 0; j < cols; ++j) {
        if (const std::size_t e = 2 * j; e <= n) {
            table[0][j] = a_desc[e];
        }
        if (const std::size_t o = 2 * j + 1; o <= n) {
            table[1][j] = a_desc[o];
        }
    }

    // A zero pivot (first-column entry of the row above) is a degeneracy: the array cannot
    // be continued by the ordinary recurrence, and the imaginary axis is in play.
    for (std::size_t r = 2; r <= n; ++r) {
        const Rational pivot = table[r - 1][0];
        if (pivot.is_zero()) {
            return RouthCount{.regular = false, .rhp = 0};
        }
        for (std::size_t j = 0; j < cols; ++j) {
            const Rational up_next = (j + 1 < cols) ? table[r - 2][j + 1] : Rational::from_int(0);
            const Rational cur_next = (j + 1 < cols) ? table[r - 1][j + 1] : Rational::from_int(0);
            auto entry = routh_entry(pivot, up_next, table[r - 2][0], cur_next);
            if (!entry) {
                return make_error<RouthCount>(entry.error());
            }
            table[r][j] = *entry;
        }
    }

    // Any zero remaining in the first column (e.g. a fully-zero final row) is also a
    // degeneracy -- an imaginary-axis root -- so the exact count is not available.
    for (std::size_t r = 0; r <= n; ++r) {
        if (table[r][0].is_zero()) {
            return RouthCount{.regular = false, .rhp = 0};
        }
    }

    // Regular array: #(roots with Re > 0) == #(sign changes down the first column).
    std::int64_t changes = 0;
    int prev = sign_of(table[0][0]);
    for (std::size_t r = 1; r <= n; ++r) {
        const int s = sign_of(table[r][0]);
        if (s != prev) {
            ++changes;
        }
        prev = s;
    }
    return RouthCount{.regular = true, .rhp = changes};
}

// The bare phase-portrait-type name (no stability adjective).
[[nodiscard]] auto phase_type_name(PhaseType t) -> std::string_view {
    switch (t) {
        case PhaseType::saddle:          return "saddle";
        case PhaseType::node:            return "node";
        case PhaseType::spiral:          return "spiral/focus";
        case PhaseType::center:          return "center";
        case PhaseType::star:            return "star";
        case PhaseType::degenerate_node: return "degenerate node";
        case PhaseType::non_isolated:    return "non-isolated equilibrium";
    }
    return "unknown";
}

}  // namespace

auto is_perfect_square(const Rational& r) -> bool {
    return exact_sqrt(r).has_value();
}

auto to_string(PhaseType t) -> std::string_view {
    return phase_type_name(t);
}

auto to_string(Stability s) -> std::string_view {
    switch (s) {
        case Stability::stable:           return "stable";
        case Stability::unstable:         return "unstable";
        case Stability::neutrally_stable: return "neutrally stable";
        case Stability::marginal:         return "marginal";
    }
    return "unknown";
}

auto to_string(LinearStability s) -> std::string_view {
    switch (s) {
        case LinearStability::sink:       return "sink (asymptotically stable)";
        case LinearStability::source:     return "source (unstable)";
        case LinearStability::saddle:     return "saddle (unstable)";
        case LinearStability::borderline:
            return "borderline (imaginary-axis / undecidable by Routh-Hurwitz alone)";
    }
    return "unknown";
}

auto classify_phase_portrait(const Matrix& a) -> Result<PhasePortrait> {
    if (!a.is_square() || a.rows() != 2) {
        return make_error<PhasePortrait>(MathError::domain_error);
    }

    auto tr = a.trace();
    if (!tr) {
        return make_error<PhasePortrait>(tr.error());
    }
    auto det = a.determinant();
    if (!det) {
        return make_error<PhasePortrait>(det.error());
    }
    const Rational t = *tr;
    const Rational d = *det;

    // delta = t^2 - 4 d, all exact over Q.
    auto t2 = t.multiply(t);
    if (!t2) {
        return make_error<PhasePortrait>(t2.error());
    }
    auto four_d = d.multiply(Rational::from_int(4));
    if (!four_d) {
        return make_error<PhasePortrait>(four_d.error());
    }
    auto delta_r = t2->subtract(*four_d);
    if (!delta_r) {
        return make_error<PhasePortrait>(delta_r.error());
    }
    const Rational delta = *delta_r;

    const int s_t = sign_of(t);
    const int s_d = sign_of(d);
    const int s_delta = sign_of(delta);

    PhasePortrait p{};
    p.trace = t;
    p.determinant = d;
    p.discriminant = delta;
    p.complex_eigenvalues = (s_delta < 0);
    p.repeated_eigenvalue = (s_delta == 0);
    p.eigenvalues_rational = false;

    const Rational two = Rational::from_int(2);
    if (s_delta >= 0) {
        // Real eigenvalues (t +/- sqrt(delta)) / 2 -- exact only when delta is a square.
        if (auto root = exact_sqrt(delta)) {
            auto lo_num = t.subtract(*root);
            if (!lo_num) {
                return make_error<PhasePortrait>(lo_num.error());
            }
            auto hi_num = t.add(*root);
            if (!hi_num) {
                return make_error<PhasePortrait>(hi_num.error());
            }
            auto l1 = lo_num->divide(two);
            if (!l1) {
                return make_error<PhasePortrait>(l1.error());
            }
            auto l2 = hi_num->divide(two);
            if (!l2) {
                return make_error<PhasePortrait>(l2.error());
            }
            p.lambda1 = *l1;
            p.lambda2 = *l2;
            p.eigenvalues_rational = true;
        }
    } else {
        // Complex pair t/2 +/- sqrt(-delta)/2 * i. The real part is always exact; the
        // imaginary magnitude is exact only when -delta is a perfect square.
        auto rp = t.divide(two);
        if (!rp) {
            return make_error<PhasePortrait>(rp.error());
        }
        p.real_part = *rp;
        auto neg_delta = delta.negate();  // -delta > 0
        if (!neg_delta) {
            return make_error<PhasePortrait>(neg_delta.error());
        }
        if (auto root = exact_sqrt(*neg_delta)) {
            auto ip = root->divide(two);
            if (!ip) {
                return make_error<PhasePortrait>(ip.error());
            }
            p.imag_part = *ip;
            p.eigenvalues_rational = true;
        }
    }

    if (s_d < 0) {
        // det A < 0: eigenvalues real of opposite sign (delta = t^2 - 4d > 0 automatically).
        p.type = PhaseType::saddle;
        p.stability = Stability::unstable;
    } else if (s_d == 0) {
        // det A == 0: a zero eigenvalue, so the equilibrium is non-isolated (a line, or a
        // plane when A == 0). The eigenvalues are 0 and t; the stability of the line splits
        // three ways, all decided exactly over Q:
        //   - t > 0: a genuinely growing direction (the eigenvalue t > 0) -- UNSTABLE.
        //   - t < 0: eigenvalues 0 and t are distinct, so A is diagonalizable; solutions are
        //            bounded and relax onto the line of equilibria -- MARGINAL.
        //   - t == 0: both eigenvalues are 0. If A == 0 every point is a fixed point and the
        //            flow is trivially bounded (MARGINAL). Otherwise A is a defective
        //            nilpotent Jordan block: the flow is x(t) = x0 + t*(A x0), which grows
        //            linearly WITHOUT BOUND for any A x0 != 0, so the origin is Lyapunov
        //            UNSTABLE. Reporting "bounded/marginal" here would be a claim the
        //            classifier cannot justify (Rule 32), so a defective nilpotent is unstable.
        p.type = PhaseType::non_isolated;
        if (s_t > 0) {
            p.stability = Stability::unstable;
        } else if (s_t < 0) {
            p.stability = Stability::marginal;
        } else {
            const bool zero_matrix = a.at(0, 0).is_zero() && a.at(0, 1).is_zero() &&
                                     a.at(1, 0).is_zero() && a.at(1, 1).is_zero();
            p.stability = zero_matrix ? Stability::marginal : Stability::unstable;
        }
    } else if (s_delta > 0) {
        // det A > 0, delta > 0: real distinct eigenvalues of a common sign (that of t,
        // which cannot be zero here since t^2 > 4d > 0).
        p.type = PhaseType::node;
        p.stability = (s_t < 0) ? Stability::stable : Stability::unstable;
    } else if (s_delta < 0) {
        // det A > 0, delta < 0: a complex pair with real part t/2.
        if (s_t == 0) {
            p.type = PhaseType::center;
            p.stability = Stability::neutrally_stable;
        } else {
            p.type = PhaseType::spiral;
            p.stability = (s_t < 0) ? Stability::stable : Stability::unstable;
        }
    } else {
        // det A > 0, delta == 0: a repeated real eigenvalue t/2 (nonzero, since d > 0). A
        // star (proper node) exactly when A is the scalar matrix lambda*I -- both
        // off-diagonals zero and the two diagonal entries equal; otherwise A is defective
        // (one eigenvector) and the origin is an improper/degenerate node.
        const bool scalar = a.at(0, 1).is_zero() && a.at(1, 0).is_zero() &&
                            (a.at(0, 0) == a.at(1, 1));
        p.type = scalar ? PhaseType::star : PhaseType::degenerate_node;
        p.stability = (s_t < 0) ? Stability::stable : Stability::unstable;
    }

    // A human-readable one-liner.
    switch (p.type) {
        case PhaseType::saddle:
            p.description = "saddle (unstable)";
            break;
        case PhaseType::center:
            p.description = "center (neutrally stable)";
            break;
        case PhaseType::non_isolated:
            p.description = (p.stability == Stability::unstable)
                                ? "non-isolated equilibrium (unbounded growth, unstable)"
                                : "non-isolated equilibrium (bounded, marginal)";
            break;
        default:
            p.description = std::string(to_string(p.stability)) + " " +
                            std::string(phase_type_name(p.type));
            break;
    }

    return p;
}

auto classify_linear_stability(const Matrix& a) -> Result<StabilityClassification> {
    if (!a.is_square() || a.rows() < 1) {
        return make_error<StabilityClassification>(MathError::domain_error);
    }
    auto rc = routh_rhp_count(a);
    if (!rc) {
        return make_error<StabilityClassification>(rc.error());
    }

    StabilityClassification out{};
    out.dimension = a.rows();
    if (!rc->regular) {
        out.verdict = LinearStability::borderline;
        out.rhp_count = -1;
        out.asymptotically_stable = false;
        return out;
    }

    const std::int64_t k = rc->rhp;
    const std::int64_t n = static_cast<std::int64_t>(a.rows());
    out.rhp_count = k;
    if (k == 0) {
        out.verdict = LinearStability::sink;
    } else if (k == n) {
        out.verdict = LinearStability::source;
    } else {
        out.verdict = LinearStability::saddle;
    }
    out.asymptotically_stable = (out.verdict == LinearStability::sink);
    return out;
}

}  // namespace nimblecas
