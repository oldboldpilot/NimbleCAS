// NimbleCAS linear control systems (ROADMAP 7 — control-theory layer).
// @author Olumuyiwa Oluwasanmi
//
// Classical linear control built on the exact rational substrate. The algebra of
// transfer functions and state-space models, the conversions between them, the
// Kalman controllability/observability tests, Routh-Hurwitz stability and the
// bilinear (Tustin) transform are ALL EXACT over Q — no floating point ever enters
// them. Only two things are honestly numerical, and they are the two that are
// irreducibly transcendental:
//
//   * poles()/zeros() extract only the RATIONAL roots (via nimblecas.roots); any
//     irrational or complex roots are reported as a count of NOT-fully-extracted
//     poles/zeros rather than silently dropped or approximated.
//   * bode()/nyquist() evaluate H(iω) in double-precision std::complex and take
//     20·log10|H| and atan2 — those are NUMERICAL. The exact counterpart
//     evaluate_exact(), which evaluates H at a Gaussian-rational point and stays in
//     Q + Qi, is provided alongside so callers who need exactness are not forced
//     through the floating path.
//
// Honesty boundary (documented and true):
//   EXACT over Q: TransferFunction/StateSpace algebra, tf_to_ss / ss_to_tf,
//                 controllability_matrix / observability_matrix and their Kalman-rank
//                 predicates, is_stable_continuous (Routh-Hurwitz, reusing
//                 nimblecas.dynamics via a companion matrix), is_stable_discrete
//                 (Schur/Jury decided exactly by mapping the unit disc to the open
//                 left half-plane with an exact Möbius transform and reusing the same
//                 Routh machinery), and bilinear_c2d / bilinear_d2c.
//   EXACT only for the rational part: poles(), zeros() (see above).
//   NUMERICAL: bode(), nyquist() (double-precision H(iω), log/atan2).
//
// Following the rest of the engine, every exact step is threaded on the Result
// railway (Rule 32): an int64 numerator/denominator that would overflow surfaces as
// MathError::overflow, a zero denominator as division_by_zero, and a dimension /
// well-formedness violation as domain_error — never an exception, never a silent wrap.

module;
#include <cassert>  // assert macro (unavailable via `import std`); active when !NDEBUG

export module nimblecas.control;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.dynamics;
import nimblecas.roots;
import nimblecas.complex;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// RootReport — the exact-rational roots of a polynomial together with an honest
// count of the roots that could NOT be extracted over Q (irrational/complex).
// ---------------------------------------------------------------------------
struct RootReport {
    // Distinct rational roots, each with its multiplicity (>= 1).
    std::vector<std::pair<Rational, std::int64_t>> rational;
    // Number of roots (counted with multiplicity) that are irrational or complex and
    // therefore NOT represented here: degree(p) - sum(multiplicities in `rational`).
    // Zero means the spectrum is fully rational and this report is complete.
    std::int64_t unextracted{0};

    [[nodiscard]] auto fully_extracted() const noexcept -> bool { return unextracted == 0; }
};

// A single Bode-plot sample: angular frequency ω (rad/s), magnitude in decibels
// (20·log10|H(iω)|) and phase in degrees (arg H(iω)). NUMERICAL (double precision).
struct BodePoint {
    double omega{0.0};
    double magnitude_db{0.0};
    double phase_deg{0.0};
};

// A single Nyquist-trace sample: the frequency ω and the point H(iω) = re + im·i in
// the complex plane. NUMERICAL (double precision).
struct NyquistPoint {
    double omega{0.0};
    double re{0.0};
    double im{0.0};
};

// ---------------------------------------------------------------------------
// TransferFunction — a SISO rational transfer function num(s)/den(s) over Q.
// ---------------------------------------------------------------------------
// Both numerator and denominator are exact RationalPoly, so every interconnection
// (series/parallel/feedback) is exact rational polynomial arithmetic. The variable
// is `s` (continuous) or `z` (discrete) depending on how the object is used; the
// algebra is identical, so the class carries no domain tag.
class TransferFunction {
public:
    // Build num/den. The denominator must be a nonzero polynomial, else domain_error.
    [[nodiscard]] static auto make(RationalPoly num, RationalPoly den)
        -> Result<TransferFunction>;

    [[nodiscard]] auto numerator() const noexcept -> const RationalPoly& { return num_; }
    [[nodiscard]] auto denominator() const noexcept -> const RationalPoly& { return den_; }

    // deg(num) <= deg(den): realisable without pure differentiators.
    [[nodiscard]] auto is_proper() const noexcept -> bool {
        return num_.degree() <= den_.degree();
    }
    // deg(num) < deg(den): no direct feedthrough term.
    [[nodiscard]] auto is_strictly_proper() const noexcept -> bool {
        return num_.degree() < den_.degree();
    }
    // deg(den) - deg(num). Positive for a strictly proper plant.
    [[nodiscard]] auto relative_degree() const noexcept -> std::int64_t {
        return den_.degree() - num_.degree();
    }

    // Series (cascade) connection G1·G2 = (n1 n2)/(d1 d2).
    [[nodiscard]] auto series(const TransferFunction& o) const -> Result<TransferFunction>;
    // Parallel connection G1 + G2 = (n1 d2 + n2 d1)/(d1 d2).
    [[nodiscard]] auto parallel(const TransferFunction& o) const -> Result<TransferFunction>;
    // Feedback interconnection of forward plant *this (= G) with feedback element `h`
    // (= H): the closed loop G/(1 ± G H). With negative_feedback (the default) the
    // denominator is d1 d2 + n1 n2; with positive feedback it is d1 d2 - n1 n2. Fails
    // with domain_error if the resulting denominator is the zero polynomial.
    [[nodiscard]] auto feedback(const TransferFunction& h, bool negative_feedback = true) const
        -> Result<TransferFunction>;
    // Unity-feedback special case (H = 1): G/(1 ± G).
    [[nodiscard]] auto unity_feedback(bool negative_feedback = true) const
        -> Result<TransferFunction>;

    // Poles: the roots of the denominator. EXACT for rational roots; irrational/complex
    // poles are counted in RootReport::unextracted. A constant denominator has no poles.
    [[nodiscard]] auto poles() const -> Result<RootReport>;
    // Zeros: the roots of the numerator, same exact/rational semantics as poles(). The
    // zero transfer function (num == 0) has no well-defined finite zeros: domain_error.
    [[nodiscard]] auto zeros() const -> Result<RootReport>;

    // DC gain H(0) = num(0)/den(0), EXACT. Fails with division_by_zero when the plant has
    // a pole at the origin (den(0) == 0).
    [[nodiscard]] auto dc_gain() const -> Result<Rational>;

    // Are *this and `o` the SAME rational function? Compared exactly by cross-
    // multiplication (n1 d2 == n2 d1), so scaled representations compare equal.
    [[nodiscard]] auto equivalent(const TransferFunction& o) const -> Result<bool>;

    [[nodiscard]] auto to_string(std::string_view var = "s") const -> std::string;

private:
    TransferFunction(RationalPoly num, RationalPoly den)
        : num_(std::move(num)), den_(std::move(den)) {}
    RationalPoly num_;
    RationalPoly den_;
};

// ---------------------------------------------------------------------------
// StateSpace — an exact-rational linear system  x' = A x + B u,  y = C x + D u.
// ---------------------------------------------------------------------------
// A is n x n, B is n x m, C is p x n, D is p x m, all over Q. Every observer below is
// exact; the controllability/observability ranks use Matrix::rank (exact over Q).
class StateSpace {
public:
    // Assemble (A, B, C, D) with consistent dimensions (A square; B rows == n;
    // C cols == n; D rows == C rows and D cols == B cols), else domain_error.
    [[nodiscard]] static auto make(Matrix a, Matrix b, Matrix c, Matrix d)
        -> Result<StateSpace>;

    [[nodiscard]] auto a() const noexcept -> const Matrix& { return a_; }
    [[nodiscard]] auto b() const noexcept -> const Matrix& { return b_; }
    [[nodiscard]] auto c() const noexcept -> const Matrix& { return c_; }
    [[nodiscard]] auto d() const noexcept -> const Matrix& { return d_; }
    [[nodiscard]] auto n_states() const noexcept -> std::size_t { return a_.rows(); }
    [[nodiscard]] auto n_inputs() const noexcept -> std::size_t { return b_.cols(); }
    [[nodiscard]] auto n_outputs() const noexcept -> std::size_t { return c_.rows(); }

    // The Kalman controllability matrix [B  AB  A^2 B  ...  A^{n-1} B] (n x n·m), EXACT.
    [[nodiscard]] auto controllability_matrix() const -> Result<Matrix>;
    // The Kalman observability matrix [C; CA; CA^2; ...; CA^{n-1}] (n·p x n), EXACT.
    [[nodiscard]] auto observability_matrix() const -> Result<Matrix>;

    // Controllable iff the controllability matrix has full rank n (exact rational rank).
    [[nodiscard]] auto is_controllable() const -> Result<bool>;
    // Observable iff the observability matrix has full rank n (exact rational rank).
    [[nodiscard]] auto is_observable() const -> Result<bool>;

    // Continuous-time asymptotic stability of x' = A x: every eigenvalue of A in the open
    // left half-plane, decided EXACTLY via Routh-Hurwitz (nimblecas.dynamics). A system
    // with zero states is vacuously stable.
    [[nodiscard]] auto is_asymptotically_stable() const -> Result<bool>;

    // Convert to a SISO transfer function C(sI - A)^{-1} B + D (exact). Requires a single
    // input and a single output, else domain_error.
    [[nodiscard]] auto to_transfer_function() const -> Result<TransferFunction>;

private:
    StateSpace(Matrix a, Matrix b, Matrix c, Matrix d)
        : a_(std::move(a)), b_(std::move(b)), c_(std::move(c)), d_(std::move(d)) {}
    Matrix a_;
    Matrix b_;
    Matrix c_;
    Matrix d_;
};

// --- TF <-> SS conversions (both EXACT over Q) ------------------------------

// Realise a proper SISO transfer function in controllable canonical form. The
// denominator is made monic and the numerator scaled to match; the feedthrough term D
// captures any direct part (deg num == deg den). Fails with domain_error when the plant
// is improper (deg num > deg den) or the denominator is zero.
[[nodiscard]] auto tf_to_ss(const TransferFunction& tf) -> Result<StateSpace>;

// Recover the SISO transfer function of a state-space model via the Faddeev-LeVerrier
// resolvent: C·adj(sI - A)·B / det(sI - A) + D, all exact over Q. Requires a single
// input and single output, else domain_error.
[[nodiscard]] auto ss_to_tf(const StateSpace& ss) -> Result<TransferFunction>;

// --- stability (EXACT over Q) -----------------------------------------------

// Continuous-time stability of the plant: every pole in the open left half-plane.
// Decided EXACTLY by Routh-Hurwitz on the denominator — built as a companion matrix and
// handed to nimblecas.dynamics::is_asymptotically_stable, so irrational/complex poles are
// settled without ever finding a root. A constant denominator (no poles) is stable.
[[nodiscard]] auto is_stable_continuous(const TransferFunction& tf) -> Result<bool>;

// Discrete-time (Schur/Jury) stability of the plant: every pole strictly inside the unit
// circle. Decided EXACTLY by mapping the open unit disc onto the open left half-plane with
// the Möbius transform z = (1 + w)/(1 - w) and testing the transformed denominator for the
// Hurwitz property — the same exact Routh machinery. A boundary pole at z = -1 (which the
// transform sends to infinity, dropping the degree) is correctly reported as not stable.
[[nodiscard]] auto is_stable_discrete(const TransferFunction& tf) -> Result<bool>;

// --- discrete transforms (EXACT over Q) -------------------------------------

// Continuous -> discrete via the bilinear (Tustin) transform s = (2/T)·(z - 1)/(z + 1),
// with sample time T > 0 (a Rational). Exact rational substitution into numerator and
// denominator. The z-transform relationship: an s-domain pole p maps to the z-domain pole
// (1 + pT/2)/(1 - pT/2), so the open left half-plane maps to the open unit disc — Tustin
// preserves stability. Fails with division_by_zero when T == 0.
[[nodiscard]] auto bilinear_c2d(const TransferFunction& tf, const Rational& sample_time)
    -> Result<TransferFunction>;

// Discrete -> continuous, the inverse Tustin map z = (1 + (T/2) s)/(1 - (T/2) s). Exact.
// Fails with division_by_zero when T == 0.
[[nodiscard]] auto bilinear_d2c(const TransferFunction& tf, const Rational& sample_time)
    -> Result<TransferFunction>;

// --- frequency response ------------------------------------------------------

// EXACT evaluation of H at a Gaussian-rational point s (e.g. s = iω with ω rational),
// staying entirely inside Q + Qi. Fails with division_by_zero when den(s) == 0.
[[nodiscard]] auto evaluate_exact(const TransferFunction& tf, const Complex& s)
    -> Result<Complex>;

// NUMERICAL Bode data: one BodePoint per supplied angular frequency ω (rad/s), with
// magnitude_db = 20·log10|H(iω)| and phase_deg = arg H(iω) in degrees. Uses double-
// precision std::complex arithmetic (this is the honestly numerical part of the module).
[[nodiscard]] auto bode(const TransferFunction& tf, std::span<const double> omegas)
    -> std::vector<BodePoint>;

// NUMERICAL Nyquist trace: H(iω) = re + im·i sampled at each supplied ω.
[[nodiscard]] auto nyquist(const TransferFunction& tf, std::span<const double> omegas)
    -> std::vector<NyquistPoint>;

// Convenience: `count` logarithmically spaced frequencies in [w_start, w_end] (both must
// be > 0). Returns a single point when count <= 1 and an empty vector when count == 0.
[[nodiscard]] auto logspace(double w_start, double w_end, std::size_t count)
    -> std::vector<double>;

// ===========================================================================
// Additional stability criteria.
// ===========================================================================

// --- Hurwitz determinant test (EXACT over Q) --------------------------------

// The n x n Hurwitz matrix of a polynomial p of degree n >= 1. The polynomial is first
// made monic (leading coefficient normalised to 1 > 0, which does not move its roots), so
// with a_0 = 1 > 0 the Routh-Hurwitz criterion reduces to the positivity of the leading
// principal minors of this matrix. Entry (i, j) (0-based) is a_{2j - i + 1} with
// a_k = coefficient of s^{n-k} and a_k = 0 outside 0..n. Fails with domain_error on the
// zero polynomial; a degree-0 polynomial yields the empty 0 x 0 matrix.
[[nodiscard]] auto hurwitz_matrix(const RationalPoly& char_poly) -> Result<Matrix>;

// The sequence of leading principal minors Δ_1, ..., Δ_n of the Hurwitz matrix, computed
// EXACTLY over Q. The polynomial is Hurwitz-stable iff every Δ_k > 0. A degree-0
// polynomial (no roots) yields an empty sequence.
[[nodiscard]] auto hurwitz_minors(const RationalPoly& char_poly) -> Result<std::vector<Rational>>;

// Hurwitz stability decided by the DETERMINANT test: true iff all leading principal minors
// of the Hurwitz matrix are strictly positive. EXACT over Q and, by the Routh-Hurwitz
// theorem, agrees with is_stable_continuous on the same denominator (the tests cross-check
// this). A constant (degree-0, root-free) polynomial is vacuously stable.
[[nodiscard]] auto is_hurwitz_stable(const RationalPoly& char_poly) -> Result<bool>;

// --- Kharitonov robust (interval) stability (EXACT over Q) ------------------

// The four Kharitonov polynomials of an interval polynomial whose coefficient of s^i lies
// in [lower[i], upper[i]] (ascending order; both vectors the same length). By the sign
// pattern (-,-,+,+),(+,+,-,-),(+,-,-,+),(-,+,+,-) repeating on i mod 4 (- = lower bound,
// + = upper bound). Fails with domain_error if the two vectors differ in length or are
// empty, or if some lower[i] > upper[i].
[[nodiscard]] auto kharitonov_polynomials(std::span<const Rational> lower,
                                          std::span<const Rational> upper)
    -> Result<std::array<RationalPoly, 4>>;

// Kharitonov's theorem: the whole interval family is robustly Hurwitz-stable iff all four
// Kharitonov polynomials are Hurwitz-stable. Decided EXACTLY over Q (is_hurwitz_stable on
// each rational-endpoint polynomial). Same well-formedness requirements as above.
[[nodiscard]] auto is_robustly_stable(std::span<const Rational> lower,
                                      std::span<const Rational> upper) -> Result<bool>;

// --- Lyapunov stability, state-space side (EXACT over Q) --------------------

// EXACT positive-definiteness test via Sylvester's criterion: a symmetric matrix P is
// positive definite iff every leading principal minor is strictly positive (computed with
// exact rational determinants). Requires a square matrix; the 0 x 0 matrix is vacuously
// positive definite.
[[nodiscard]] auto is_positive_definite(const Matrix& p) -> Result<bool>;

// The unique solution P of the continuous Lyapunov equation Aᵀ P + P A = -I, obtained
// EXACTLY over Q by assembling the Kronecker-sum system (I ⊗ Aᵀ + Aᵀ ⊗ I) vec(P) = -vec(I)
// and solving it with exact rational elimination. Requires a square A; fails with
// domain_error when the Kronecker-sum operator is singular (some eigenvalue pair
// λ_i + λ_j = 0, e.g. a purely imaginary spectrum — which already means A is not
// asymptotically stable).
[[nodiscard]] auto lyapunov_solve(const Matrix& a) -> Result<Matrix>;

// Lyapunov's theorem: A is asymptotically stable iff Aᵀ P + P A = -I has a positive-
// definite solution P. EXACT over Q; agrees with is_asymptotically_stable /
// is_stable_continuous (the tests cross-check this). A singular Kronecker-sum operator
// (no unique P) means A is not asymptotically stable, reported as false.
[[nodiscard]] auto is_stable_lyapunov(const Matrix& a) -> Result<bool>;

// --- Nyquist stability criterion (P exact; N and the trace NUMERICAL) -------

// The outcome of the Nyquist test applied to an open-loop transfer function L(s) = G·H.
struct NyquistResult {
    // N: clockwise encirclements of the -1 point by the Nyquist plot of L. NUMERICAL —
    // a winding number computed from the double-precision Nyquist trace.
    std::int64_t encirclements{0};
    // P: number of open-loop poles of L in the open right half-plane, counted from the
    // RATIONAL poles (see p_exact). This term is exact when the pole set is fully rational.
    std::int64_t open_loop_rhp_poles{0};
    // Z = N + P: the implied number of closed-loop poles in the right half-plane.
    std::int64_t closed_loop_rhp_poles{0};
    // The closed loop is stable iff Z == 0.
    bool closed_loop_stable{false};
    // Whether P (and hence Z) is exact: true when L has no irrational/complex poles that
    // poles() could not extract. When false, treat the P/Z counts as a rational-only lower
    // read and prefer an explicit RHP-pole count.
    bool p_exact{false};
};

// Apply the Nyquist criterion to the open-loop L over the supplied POSITIVE frequency grid
// (ascending; strictly positive). The encirclement count N is a NUMERICAL winding number of
// the full mirror-completed contour about -1; P is counted from L.poles() (exact for
// rational poles). Fails with domain_error on an empty grid or when L's poles cannot be
// obtained. NOTE: an open-loop integrator (pole at the origin) is not represented by a
// finite grid and is outside this numerical routine's scope.
[[nodiscard]] auto nyquist_criterion(const TransferFunction& open_loop,
                                     std::span<const double> omegas) -> Result<NyquistResult>;

// --- gain & phase margins (NUMERICAL) ---------------------------------------

// Classical stability margins read off the open-loop frequency response.
struct StabilityMargins {
    // Gain margin (linear) = 1/|L(iω_pc)| at the phase-crossover frequency, and in dB.
    double gain_margin{0.0};
    double gain_margin_db{0.0};
    // Phase-crossover frequency ω_pc (rad/s), where arg L = -180°.
    double phase_crossover{0.0};
    // Phase margin (degrees) = 180° + arg L(iω_gc) at the gain-crossover frequency.
    double phase_margin{0.0};
    // Gain-crossover frequency ω_gc (rad/s), where |L| = 1.
    double gain_crossover{0.0};
    // Whether a phase crossover / gain crossover was actually found on the supplied grid.
    bool has_gain_margin{false};
    bool has_phase_margin{false};
};

// Gain and phase margins of the open-loop L, found NUMERICALLY by locating the crossover
// frequencies on the (double-precision) Bode data over the supplied positive grid and
// linearly interpolating. The phase is unwrapped before locating the -180° crossing.
[[nodiscard]] auto stability_margins(const TransferFunction& open_loop,
                                     std::span<const double> omegas) -> StabilityMargins;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// A monic linear form's coefficient helpers reused below.
[[nodiscard]] auto one_rat() -> Rational { return Rational::from_int(1); }

// Multiply two RationalPolys, propagating overflow on the railway.
[[nodiscard]] auto pmul(const RationalPoly& a, const RationalPoly& b) -> Result<RationalPoly> {
    return a.multiply(b);
}
[[nodiscard]] auto padd(const RationalPoly& a, const RationalPoly& b) -> Result<RationalPoly> {
    return a.add(b);
}

// Homogenised Möbius substitution: given P(x) of degree d <= dmax and the map
// x = (a·w + b)/(g·w + d), return the exact polynomial
//     Q(w) = (g·w + d)^dmax · P(x) = Σ_i p_i · (a·w + b)^i · (g·w + d)^{dmax - i}.
// All terms have total degree dmax, so Q is an ordinary polynomial. When the SAME dmax is
// used for a numerator and a denominator, the common (g·w + d)^dmax factor cancels in the
// ratio, giving the exact transformed transfer function. Requires dmax >= deg(P).
[[nodiscard]] auto mobius_homogenize(const RationalPoly& p, std::size_t dmax, const Rational& a,
                                     const Rational& b, const Rational& g, const Rational& d)
    -> Result<RationalPoly> {
    const RationalPoly num_lin = RationalPoly::from_coeffs({b, a});  // a·w + b
    const RationalPoly den_lin = RationalPoly::from_coeffs({d, g});  // g·w + d

    std::vector<RationalPoly> num_pows(dmax + 1);
    std::vector<RationalPoly> den_pows(dmax + 1);
    num_pows[0] = RationalPoly::constant(one_rat());
    den_pows[0] = RationalPoly::constant(one_rat());
    for (std::size_t i = 1; i <= dmax; ++i) {
        auto np = pmul(num_pows[i - 1], num_lin);
        if (!np) {
            return make_error<RationalPoly>(np.error());
        }
        num_pows[i] = std::move(*np);
        auto dp = pmul(den_pows[i - 1], den_lin);
        if (!dp) {
            return make_error<RationalPoly>(dp.error());
        }
        den_pows[i] = std::move(*dp);
    }

    RationalPoly result;  // zero polynomial
    const std::span<const Rational> coeffs = p.coefficients();
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        if (coeffs[i].is_zero()) {
            continue;
        }
        assert(i <= dmax && "mobius_homogenize: dmax must be at least deg(P)");
        auto prod = pmul(num_pows[i], den_pows[dmax - i]);
        if (!prod) {
            return make_error<RationalPoly>(prod.error());
        }
        auto term = prod->scale(coeffs[i]);
        if (!term) {
            return make_error<RationalPoly>(term.error());
        }
        auto sum = padd(result, *term);
        if (!sum) {
            return make_error<RationalPoly>(sum.error());
        }
        result = std::move(*sum);
    }
    return result;
}

// Controllable-canonical-form (CCF) companion matrix of a MONIC polynomial
// s^n + a_{n-1} s^{n-1} + ... + a_0: super-diagonal ones and a bottom row of negated
// coefficients. Its characteristic polynomial is exactly that monic polynomial. n >= 1.
[[nodiscard]] auto companion_ccf(const RationalPoly& monic_den) -> Result<Matrix> {
    const std::int64_t deg = monic_den.degree();
    assert(deg >= 1 && "companion_ccf requires degree >= 1");
    const std::size_t n = static_cast<std::size_t>(deg);

    std::vector<std::vector<Rational>> rows(n, std::vector<Rational>(n, Rational::from_int(0)));
    for (std::size_t i = 0; i + 1 < n; ++i) {
        rows[i][i + 1] = Rational::from_int(1);  // super-diagonal
    }
    for (std::size_t j = 0; j < n; ++j) {
        auto neg = monic_den.coefficient(j).negate();  // -a_j on the bottom row
        if (!neg) {
            return make_error<Matrix>(neg.error());
        }
        rows[n - 1][j] = *neg;
    }
    return Matrix::from_rows(std::move(rows));
}

// Hurwitz test on a (possibly non-monic) denominator polynomial: build the companion
// matrix of its monic form and defer to the exact Routh-Hurwitz criterion in
// nimblecas.dynamics. A constant (degree 0) denominator has no roots and is stable.
[[nodiscard]] auto denominator_is_hurwitz(const RationalPoly& den) -> Result<bool> {
    if (den.is_zero()) {
        return make_error<bool>(MathError::domain_error);
    }
    if (den.degree() == 0) {
        return true;  // no poles at all
    }
    auto monic = den.monic();
    if (!monic) {
        return make_error<bool>(monic.error());
    }
    auto comp = companion_ccf(*monic);
    if (!comp) {
        return make_error<bool>(comp.error());
    }
    return is_asymptotically_stable(*comp);  // exact Routh-Hurwitz (nimblecas.dynamics)
}

// Horizontally concatenate same-height blocks: [M0 | M1 | ... ].
[[nodiscard]] auto hstack(std::span<const Matrix> blocks) -> Result<Matrix> {
    if (blocks.empty()) {
        return Matrix::zero(0, 0);
    }
    const std::size_t rows = blocks.front().rows();
    std::size_t total_cols = 0;
    for (const Matrix& m : blocks) {
        if (m.rows() != rows) {
            return make_error<Matrix>(MathError::domain_error);
        }
        total_cols += m.cols();
    }
    if (rows == 0) {
        return Matrix::zero(0, total_cols);
    }
    std::vector<std::vector<Rational>> out(rows, std::vector<Rational>(total_cols));
    for (std::size_t i = 0; i < rows; ++i) {
        std::size_t off = 0;
        for (const Matrix& m : blocks) {
            for (std::size_t j = 0; j < m.cols(); ++j) {
                out[i][off + j] = m.at(i, j);
            }
            off += m.cols();
        }
    }
    return Matrix::from_rows(std::move(out));
}

// Vertically stack same-width blocks: [M0; M1; ...].
[[nodiscard]] auto vstack(std::span<const Matrix> blocks) -> Result<Matrix> {
    if (blocks.empty()) {
        return Matrix::zero(0, 0);
    }
    const std::size_t cols = blocks.front().cols();
    std::size_t total_rows = 0;
    for (const Matrix& m : blocks) {
        if (m.cols() != cols) {
            return make_error<Matrix>(MathError::domain_error);
        }
        total_rows += m.rows();
    }
    std::vector<std::vector<Rational>> out;
    out.reserve(total_rows);
    for (const Matrix& m : blocks) {
        for (std::size_t i = 0; i < m.rows(); ++i) {
            std::vector<Rational> row(cols);
            for (std::size_t j = 0; j < cols; ++j) {
                row[j] = m.at(i, j);
            }
            out.push_back(std::move(row));
        }
    }
    return Matrix::from_rows(std::move(out));
}

// Extract the exact rational roots of p and count the remainder as not-fully-extracted.
[[nodiscard]] auto extract_roots(const RationalPoly& p) -> Result<RootReport> {
    if (p.is_zero()) {
        return make_error<RootReport>(MathError::domain_error);  // every value is a root
    }
    RootReport rep;
    if (p.degree() == 0) {
        return rep;  // nonzero constant: no roots, nothing unextracted
    }
    auto rr = rational_roots(p);
    if (!rr) {
        return make_error<RootReport>(rr.error());
    }
    std::int64_t accounted = 0;
    for (const auto& [value, mult] : *rr) {
        accounted += mult;
    }
    rep.rational = std::move(*rr);
    rep.unextracted = p.degree() - accounted;
    return rep;
}

// Evaluate a RationalPoly at an exact Gaussian-rational point by Horner's method.
[[nodiscard]] auto eval_poly_exact(const RationalPoly& p, const Complex& s) -> Result<Complex> {
    Complex acc;  // 0 + 0i
    const std::span<const Rational> coeffs = p.coefficients();
    for (std::size_t i = coeffs.size(); i-- > 0;) {
        auto scaled = acc.multiply(s);
        if (!scaled) {
            return make_error<Complex>(scaled.error());
        }
        auto sum = scaled->add(Complex::from_real(coeffs[i]));
        if (!sum) {
            return make_error<Complex>(sum.error());
        }
        acc = *sum;
    }
    return acc;
}

// Evaluate a RationalPoly at a double-precision complex point by Horner's method
// (the numerical frequency-response path).
[[nodiscard]] auto eval_poly_cd(const RationalPoly& p, std::complex<double> s)
    -> std::complex<double> {
    std::complex<double> acc{0.0, 0.0};
    const std::span<const Rational> coeffs = p.coefficients();
    for (std::size_t i = coeffs.size(); i-- > 0;) {
        const double v = static_cast<double>(coeffs[i].numerator()) /
                         static_cast<double>(coeffs[i].denominator());
        acc = acc * s + std::complex<double>(v, 0.0);
    }
    return acc;
}

// The exact determinant of the top-left k x k block of a dense Rational matrix given as
// rows. Reuses Matrix's exact rational determinant.
[[nodiscard]] auto leading_minor(const std::vector<std::vector<Rational>>& m, std::size_t k)
    -> Result<Rational> {
    std::vector<std::vector<Rational>> block(k, std::vector<Rational>(k));
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
            block[i][j] = m[i][j];
        }
    }
    auto mat = Matrix::from_rows(std::move(block));
    if (!mat) {
        return make_error<Rational>(mat.error());
    }
    return mat->determinant();
}

// Build the Hurwitz matrix rows of a MONIC polynomial of degree n >= 1 (a_0 = 1). With
// a_k = coefficient of s^{n-k}, entry (i, j) = a_{2j - i + 1}, zero outside 0..n.
[[nodiscard]] auto hurwitz_rows(const RationalPoly& monic, std::size_t n)
    -> std::vector<std::vector<Rational>> {
    std::vector<std::vector<Rational>> rows(n, std::vector<Rational>(n, Rational::from_int(0)));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const std::int64_t idx =
                2 * static_cast<std::int64_t>(j) - static_cast<std::int64_t>(i) + 1;
            if (idx < 0 || idx > static_cast<std::int64_t>(n)) {
                continue;  // a_k = 0 outside 0..n
            }
            // a_k = coefficient of s^{n-k}.
            rows[i][j] = monic.coefficient(n - static_cast<std::size_t>(idx));
        }
    }
    return rows;
}

// Count the sign of a Rational (denominator is kept positive, so the numerator's sign is
// the value's sign): -1, 0, +1.
[[nodiscard]] auto rsign_of(const Rational& r) -> int {
    if (r.numerator() > 0) {
        return 1;
    }
    if (r.numerator() < 0) {
        return -1;
    }
    return 0;
}

// One Kharitonov sign pattern: for polynomial index i, choose the lower (-) or upper (+)
// endpoint. `pattern` gives the four i-mod-4 choices as booleans (true => upper bound).
[[nodiscard]] auto kharitonov_one(std::span<const Rational> lower, std::span<const Rational> upper,
                                  const std::array<bool, 4>& pattern) -> RationalPoly {
    std::vector<Rational> c(lower.size());
    for (std::size_t i = 0; i < lower.size(); ++i) {
        c[i] = pattern[i % 4] ? upper[i] : lower[i];
    }
    return RationalPoly::from_coeffs(std::move(c));
}

}  // namespace

// --- TransferFunction --------------------------------------------------------

auto TransferFunction::make(RationalPoly num, RationalPoly den) -> Result<TransferFunction> {
    if (den.is_zero()) {
        return make_error<TransferFunction>(MathError::domain_error);
    }
    return TransferFunction{std::move(num), std::move(den)};
}

auto TransferFunction::series(const TransferFunction& o) const -> Result<TransferFunction> {
    auto num = pmul(num_, o.num_);
    if (!num) {
        return make_error<TransferFunction>(num.error());
    }
    auto den = pmul(den_, o.den_);
    if (!den) {
        return make_error<TransferFunction>(den.error());
    }
    return make(std::move(*num), std::move(*den));
}

auto TransferFunction::parallel(const TransferFunction& o) const -> Result<TransferFunction> {
    auto n1d2 = pmul(num_, o.den_);
    if (!n1d2) {
        return make_error<TransferFunction>(n1d2.error());
    }
    auto n2d1 = pmul(o.num_, den_);
    if (!n2d1) {
        return make_error<TransferFunction>(n2d1.error());
    }
    auto num = padd(*n1d2, *n2d1);
    if (!num) {
        return make_error<TransferFunction>(num.error());
    }
    auto den = pmul(den_, o.den_);
    if (!den) {
        return make_error<TransferFunction>(den.error());
    }
    return make(std::move(*num), std::move(*den));
}

auto TransferFunction::feedback(const TransferFunction& h, bool negative_feedback) const
    -> Result<TransferFunction> {
    // Closed loop G/(1 ± G H) = (n1 d2) / (d1 d2 ± n1 n2).
    auto num = pmul(num_, h.den_);  // n1 d2
    if (!num) {
        return make_error<TransferFunction>(num.error());
    }
    auto d1d2 = pmul(den_, h.den_);
    if (!d1d2) {
        return make_error<TransferFunction>(d1d2.error());
    }
    auto n1n2 = pmul(num_, h.num_);
    if (!n1n2) {
        return make_error<TransferFunction>(n1n2.error());
    }
    auto den = negative_feedback ? d1d2->add(*n1n2) : d1d2->subtract(*n1n2);
    if (!den) {
        return make_error<TransferFunction>(den.error());
    }
    return make(std::move(*num), std::move(*den));
}

auto TransferFunction::unity_feedback(bool negative_feedback) const -> Result<TransferFunction> {
    const auto unity = make(RationalPoly::constant(one_rat()), RationalPoly::constant(one_rat()));
    if (!unity) {
        return make_error<TransferFunction>(unity.error());
    }
    return feedback(*unity, negative_feedback);
}

auto TransferFunction::poles() const -> Result<RootReport> {
    return extract_roots(den_);
}

auto TransferFunction::zeros() const -> Result<RootReport> {
    return extract_roots(num_);
}

auto TransferFunction::dc_gain() const -> Result<Rational> {
    return num_.coefficient(0).divide(den_.coefficient(0));  // division_by_zero if pole at 0
}

auto TransferFunction::equivalent(const TransferFunction& o) const -> Result<bool> {
    auto lhs = pmul(num_, o.den_);
    if (!lhs) {
        return make_error<bool>(lhs.error());
    }
    auto rhs = pmul(o.num_, den_);
    if (!rhs) {
        return make_error<bool>(rhs.error());
    }
    return lhs->is_equal(*rhs);
}

auto TransferFunction::to_string(std::string_view var) const -> std::string {
    return std::format("({}) / ({})", num_.to_string(var), den_.to_string(var));
}

// --- StateSpace --------------------------------------------------------------

auto StateSpace::make(Matrix a, Matrix b, Matrix c, Matrix d) -> Result<StateSpace> {
    if (!a.is_square()) {
        return make_error<StateSpace>(MathError::domain_error);
    }
    const std::size_t n = a.rows();
    if (b.rows() != n || c.cols() != n) {
        return make_error<StateSpace>(MathError::domain_error);
    }
    if (d.rows() != c.rows() || d.cols() != b.cols()) {
        return make_error<StateSpace>(MathError::domain_error);
    }
    return StateSpace{std::move(a), std::move(b), std::move(c), std::move(d)};
}

auto StateSpace::controllability_matrix() const -> Result<Matrix> {
    const std::size_t n = a_.rows();
    if (n == 0) {
        return Matrix::zero(0, 0);
    }
    std::vector<Matrix> blocks;
    blocks.reserve(n);
    Matrix cur = b_;  // A^0 B
    blocks.push_back(cur);
    for (std::size_t k = 1; k < n; ++k) {
        auto next = a_.multiply(cur);  // A^k B
        if (!next) {
            return make_error<Matrix>(next.error());
        }
        cur = std::move(*next);
        blocks.push_back(cur);
    }
    return hstack(blocks);
}

auto StateSpace::observability_matrix() const -> Result<Matrix> {
    const std::size_t n = a_.rows();
    if (n == 0) {
        return Matrix::zero(0, 0);
    }
    std::vector<Matrix> blocks;
    blocks.reserve(n);
    Matrix cur = c_;  // C A^0
    blocks.push_back(cur);
    for (std::size_t k = 1; k < n; ++k) {
        auto next = cur.multiply(a_);  // C A^k
        if (!next) {
            return make_error<Matrix>(next.error());
        }
        cur = std::move(*next);
        blocks.push_back(cur);
    }
    return vstack(blocks);
}

auto StateSpace::is_controllable() const -> Result<bool> {
    auto mc = controllability_matrix();
    if (!mc) {
        return make_error<bool>(mc.error());
    }
    return mc->rank() == static_cast<std::int64_t>(a_.rows());
}

auto StateSpace::is_observable() const -> Result<bool> {
    auto mo = observability_matrix();
    if (!mo) {
        return make_error<bool>(mo.error());
    }
    return mo->rank() == static_cast<std::int64_t>(a_.rows());
}

auto StateSpace::is_asymptotically_stable() const -> Result<bool> {
    if (a_.rows() == 0) {
        return true;  // no dynamics: vacuously stable
    }
    return nimblecas::is_asymptotically_stable(a_);  // exact Routh-Hurwitz
}

auto StateSpace::to_transfer_function() const -> Result<TransferFunction> {
    return ss_to_tf(*this);
}

// --- TF -> SS (controllable canonical form) ---------------------------------

auto tf_to_ss(const TransferFunction& tf) -> Result<StateSpace> {
    if (!tf.is_proper()) {
        return make_error<StateSpace>(MathError::domain_error);  // improper: needs differentiators
    }
    const RationalPoly& den = tf.denominator();
    const RationalPoly& num = tf.numerator();
    if (den.is_zero()) {
        return make_error<StateSpace>(MathError::domain_error);
    }
    const std::int64_t deg = den.degree();

    // Degree-0 denominator: a pure static gain H = num0/den0. Model it with zero states
    // and a single feedthrough entry.
    if (deg == 0) {
        auto gain = num.coefficient(0).divide(den.coefficient(0));
        if (!gain) {
            return make_error<StateSpace>(gain.error());
        }
        auto a = Matrix::zero(0, 0);
        auto b = Matrix::zero(0, 1);
        auto c = Matrix::zero(1, 0);
        auto dmat = Matrix::from_rows({{*gain}});
        if (!dmat) {
            return make_error<StateSpace>(dmat.error());
        }
        return StateSpace::make(std::move(a), std::move(b), std::move(c), std::move(*dmat));
    }

    const std::size_t n = static_cast<std::size_t>(deg);

    // Make the denominator monic and scale the numerator by the same factor so the ratio
    // is unchanged: work with  num_scaled / monic_den.
    auto monic_den = den.monic();
    if (!monic_den) {
        return make_error<StateSpace>(monic_den.error());
    }
    auto inv_lc = one_rat().divide(den.leading_coefficient());  // 1/lc(den), lc != 0
    if (!inv_lc) {
        return make_error<StateSpace>(inv_lc.error());
    }
    auto num_scaled = num.scale(*inv_lc);
    if (!num_scaled) {
        return make_error<StateSpace>(num_scaled.error());
    }

    // Split off the direct feedthrough: num_scaled = quotient·monic_den + remainder, with
    // quotient a constant (properness) giving D, and remainder (deg < n) giving C.
    auto dm = num_scaled->divide(*monic_den);
    if (!dm) {
        return make_error<StateSpace>(dm.error());
    }
    const Rational feedthrough = dm->quotient.coefficient(0);  // 0 when strictly proper
    const RationalPoly& remainder = dm->remainder;

    // A: super-diagonal ones, bottom row = -a_j (a_j = coefficient j of the monic denom).
    std::vector<std::vector<Rational>> arows(n, std::vector<Rational>(n, Rational::from_int(0)));
    for (std::size_t i = 0; i + 1 < n; ++i) {
        arows[i][i + 1] = Rational::from_int(1);
    }
    for (std::size_t j = 0; j < n; ++j) {
        auto neg = monic_den->coefficient(j).negate();
        if (!neg) {
            return make_error<StateSpace>(neg.error());
        }
        arows[n - 1][j] = *neg;
    }
    auto a = Matrix::from_rows(std::move(arows));
    if (!a) {
        return make_error<StateSpace>(a.error());
    }

    // B = [0, ..., 0, 1]^T.
    std::vector<std::vector<Rational>> brows(n, std::vector<Rational>(1, Rational::from_int(0)));
    brows[n - 1][0] = Rational::from_int(1);
    auto b = Matrix::from_rows(std::move(brows));
    if (!b) {
        return make_error<StateSpace>(b.error());
    }

    // C = [r_0, r_1, ..., r_{n-1}] from the (strictly proper) remainder.
    std::vector<Rational> crow(n, Rational::from_int(0));
    for (std::size_t j = 0; j < n; ++j) {
        crow[j] = remainder.coefficient(j);
    }
    auto c = Matrix::from_rows({std::move(crow)});
    if (!c) {
        return make_error<StateSpace>(c.error());
    }

    // D = [feedthrough].
    auto dmat = Matrix::from_rows({{feedthrough}});
    if (!dmat) {
        return make_error<StateSpace>(dmat.error());
    }

    return StateSpace::make(std::move(*a), std::move(*b), std::move(*c), std::move(*dmat));
}

// --- SS -> TF (Faddeev-LeVerrier resolvent) ---------------------------------

auto ss_to_tf(const StateSpace& ss) -> Result<TransferFunction> {
    if (ss.n_inputs() != 1 || ss.n_outputs() != 1) {
        return make_error<TransferFunction>(MathError::domain_error);  // SISO only
    }
    const Matrix& a = ss.a();
    const Matrix& b = ss.b();
    const Matrix& c = ss.c();
    const Rational d = ss.d().at(0, 0);
    const std::size_t n = a.rows();

    // Zero states: H(s) = D (a constant).
    if (n == 0) {
        return TransferFunction::make(RationalPoly::constant(d), RationalPoly::constant(one_rat()));
    }

    // Faddeev-LeVerrier: M_1 = I; c_k = tr(A M_k)/k; M_{k+1} = A M_k - c_k I. The resolvent
    // is (sI - A)^{-1} = (1/p(s)) Σ_{k=1}^n M_k s^{n-k}, and the monic characteristic
    // polynomial is p(λ) = λ^n - Σ_{k=1}^n c_k λ^{n-k}.
    std::vector<Matrix> m_of(n + 1);       // M_1 .. M_n (index 0 unused)
    std::vector<Rational> cval(n + 1, Rational::from_int(0));
    Matrix m = Matrix::identity(n);
    m_of[1] = m;
    for (std::size_t k = 1; k <= n; ++k) {
        auto am = a.multiply(m);  // A M_k
        if (!am) {
            return make_error<TransferFunction>(am.error());
        }
        auto tr = am->trace();
        if (!tr) {
            return make_error<TransferFunction>(tr.error());
        }
        auto ck = tr->divide(Rational::from_int(static_cast<std::int64_t>(k)));
        if (!ck) {
            return make_error<TransferFunction>(ck.error());
        }
        cval[k] = *ck;
        if (k < n) {
            auto shift = Matrix::identity(n).scale(*ck);
            if (!shift) {
                return make_error<TransferFunction>(shift.error());
            }
            auto next = am->subtract(*shift);  // M_{k+1} = A M_k - c_k I
            if (!next) {
                return make_error<TransferFunction>(next.error());
            }
            m = std::move(*next);
            m_of[k + 1] = m;
        }
    }

    // Denominator p(λ): p[n] = 1, p[n-k] = -c_k.
    std::vector<Rational> pc(n + 1, Rational::from_int(0));
    pc[n] = Rational::from_int(1);
    for (std::size_t k = 1; k <= n; ++k) {
        auto neg = cval[k].negate();
        if (!neg) {
            return make_error<TransferFunction>(neg.error());
        }
        pc[n - k] = *neg;
    }
    const RationalPoly denom = RationalPoly::from_coeffs(std::move(pc));

    // Numerator N(s) = Σ_{k=1}^n (C M_k B) s^{n-k} + D·p(s), each C M_k B a scalar.
    RationalPoly numer;  // zero
    for (std::size_t k = 1; k <= n; ++k) {
        auto cm = c.multiply(m_of[k]);  // 1 x n
        if (!cm) {
            return make_error<TransferFunction>(cm.error());
        }
        auto cmb = cm->multiply(b);  // 1 x 1
        if (!cmb) {
            return make_error<TransferFunction>(cmb.error());
        }
        const Rational scalar = cmb->at(0, 0);
        const RationalPoly term = RationalPoly::monomial(scalar, n - k);  // scalar · s^{n-k}
        auto sum = padd(numer, term);
        if (!sum) {
            return make_error<TransferFunction>(sum.error());
        }
        numer = std::move(*sum);
    }
    auto dp = denom.scale(d);  // D·p(s)
    if (!dp) {
        return make_error<TransferFunction>(dp.error());
    }
    auto full_num = padd(numer, *dp);
    if (!full_num) {
        return make_error<TransferFunction>(full_num.error());
    }

    return TransferFunction::make(std::move(*full_num), denom);
}

// --- stability ---------------------------------------------------------------

auto is_stable_continuous(const TransferFunction& tf) -> Result<bool> {
    return denominator_is_hurwitz(tf.denominator());
}

auto is_stable_discrete(const TransferFunction& tf) -> Result<bool> {
    const RationalPoly& den = tf.denominator();
    if (den.is_zero()) {
        return make_error<bool>(MathError::domain_error);
    }
    const std::int64_t deg = den.degree();
    if (deg == 0) {
        return true;  // no poles
    }
    const std::size_t n = static_cast<std::size_t>(deg);

    // Map the open unit disc |z| < 1 onto the open left half-plane Re(w) < 0 via
    // z = (1 + w)/(1 - w); the transformed denominator is Schur-stable iff Hurwitz.
    const Rational one = one_rat();
    auto minus_one = one.negate();
    if (!minus_one) {
        return make_error<bool>(minus_one.error());
    }
    auto d_hat = mobius_homogenize(den, n, one, one, *minus_one, one);
    if (!d_hat) {
        return make_error<bool>(d_hat.error());
    }
    // A degree drop signals a boundary pole at z = -1 (mapped to infinity): not strictly
    // inside the unit circle.
    if (d_hat->degree() < deg) {
        return false;
    }
    return denominator_is_hurwitz(*d_hat);
}

// --- discrete transforms -----------------------------------------------------

auto bilinear_c2d(const TransferFunction& tf, const Rational& sample_time)
    -> Result<TransferFunction> {
    // s = (2/T)·(z - 1)/(z + 1):  a = 2/T, b = -2/T, g = 1, d = 1.
    auto k = Rational::from_int(2).divide(sample_time);  // division_by_zero when T == 0
    if (!k) {
        return make_error<TransferFunction>(k.error());
    }
    auto neg_k = k->negate();
    if (!neg_k) {
        return make_error<TransferFunction>(neg_k.error());
    }
    const RationalPoly& num = tf.numerator();
    const RationalPoly& den = tf.denominator();
    const std::size_t dmax =
        static_cast<std::size_t>(std::max<std::int64_t>({num.degree(), den.degree(), 0}));
    const Rational one = one_rat();
    auto num_z = mobius_homogenize(num, dmax, *k, *neg_k, one, one);
    if (!num_z) {
        return make_error<TransferFunction>(num_z.error());
    }
    auto den_z = mobius_homogenize(den, dmax, *k, *neg_k, one, one);
    if (!den_z) {
        return make_error<TransferFunction>(den_z.error());
    }
    return TransferFunction::make(std::move(*num_z), std::move(*den_z));
}

auto bilinear_d2c(const TransferFunction& tf, const Rational& sample_time)
    -> Result<TransferFunction> {
    // Inverse Tustin z = (1 + (T/2) s)/(1 - (T/2) s):  a = T/2, b = 1, g = -T/2, d = 1.
    auto half_t = sample_time.divide(Rational::from_int(2));
    if (!half_t) {
        return make_error<TransferFunction>(half_t.error());
    }
    auto neg_half_t = half_t->negate();
    if (!neg_half_t) {
        return make_error<TransferFunction>(neg_half_t.error());
    }
    const RationalPoly& num = tf.numerator();
    const RationalPoly& den = tf.denominator();
    const std::size_t dmax =
        static_cast<std::size_t>(std::max<std::int64_t>({num.degree(), den.degree(), 0}));
    const Rational one = one_rat();
    auto num_s = mobius_homogenize(num, dmax, *half_t, one, *neg_half_t, one);
    if (!num_s) {
        return make_error<TransferFunction>(num_s.error());
    }
    auto den_s = mobius_homogenize(den, dmax, *half_t, one, *neg_half_t, one);
    if (!den_s) {
        return make_error<TransferFunction>(den_s.error());
    }
    return TransferFunction::make(std::move(*num_s), std::move(*den_s));
}

// --- frequency response ------------------------------------------------------

auto evaluate_exact(const TransferFunction& tf, const Complex& s) -> Result<Complex> {
    auto num = eval_poly_exact(tf.numerator(), s);
    if (!num) {
        return make_error<Complex>(num.error());
    }
    auto den = eval_poly_exact(tf.denominator(), s);
    if (!den) {
        return make_error<Complex>(den.error());
    }
    return num->divide(*den);  // division_by_zero when den(s) == 0
}

auto bode(const TransferFunction& tf, std::span<const double> omegas) -> std::vector<BodePoint> {
    constexpr double rad_to_deg = 180.0 / std::numbers::pi;
    std::vector<BodePoint> out;
    out.reserve(omegas.size());
    for (const double w : omegas) {
        const std::complex<double> s{0.0, w};
        const std::complex<double> num = eval_poly_cd(tf.numerator(), s);
        const std::complex<double> den = eval_poly_cd(tf.denominator(), s);
        const std::complex<double> h = num / den;  // may be inf/nan at an imaginary-axis pole
        const double mag = std::abs(h);
        out.push_back(BodePoint{.omega = w,
                                .magnitude_db = 20.0 * std::log10(mag),
                                .phase_deg = std::arg(h) * rad_to_deg});
    }
    return out;
}

auto nyquist(const TransferFunction& tf, std::span<const double> omegas)
    -> std::vector<NyquistPoint> {
    std::vector<NyquistPoint> out;
    out.reserve(omegas.size());
    for (const double w : omegas) {
        const std::complex<double> s{0.0, w};
        const std::complex<double> num = eval_poly_cd(tf.numerator(), s);
        const std::complex<double> den = eval_poly_cd(tf.denominator(), s);
        const std::complex<double> h = num / den;
        out.push_back(NyquistPoint{.omega = w, .re = h.real(), .im = h.imag()});
    }
    return out;
}

auto logspace(double w_start, double w_end, std::size_t count) -> std::vector<double> {
    std::vector<double> out;
    if (count == 0) {
        return out;
    }
    out.reserve(count);
    if (count == 1) {
        out.push_back(w_start);
        return out;
    }
    const double log_lo = std::log10(w_start);
    const double log_hi = std::log10(w_end);
    const double step = (log_hi - log_lo) / static_cast<double>(count - 1);
    for (std::size_t i = 0; i < count; ++i) {
        out.push_back(std::pow(10.0, log_lo + step * static_cast<double>(i)));
    }
    return out;
}

// --- Hurwitz determinant test ------------------------------------------------

auto hurwitz_matrix(const RationalPoly& char_poly) -> Result<Matrix> {
    if (char_poly.is_zero()) {
        return make_error<Matrix>(MathError::domain_error);
    }
    if (char_poly.degree() == 0) {
        return Matrix::zero(0, 0);  // no roots, empty Hurwitz matrix
    }
    auto monic = char_poly.monic();  // normalise a_0 = 1 > 0 (roots unchanged)
    if (!monic) {
        return make_error<Matrix>(monic.error());
    }
    const std::size_t n = static_cast<std::size_t>(char_poly.degree());
    return Matrix::from_rows(hurwitz_rows(*monic, n));
}

auto hurwitz_minors(const RationalPoly& char_poly) -> Result<std::vector<Rational>> {
    if (char_poly.is_zero()) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    if (char_poly.degree() == 0) {
        return std::vector<Rational>{};  // no roots, no minors
    }
    auto monic = char_poly.monic();
    if (!monic) {
        return make_error<std::vector<Rational>>(monic.error());
    }
    const std::size_t n = static_cast<std::size_t>(char_poly.degree());
    const std::vector<std::vector<Rational>> rows = hurwitz_rows(*monic, n);

    std::vector<Rational> minors;
    minors.reserve(n);
    for (std::size_t k = 1; k <= n; ++k) {
        auto delta = leading_minor(rows, k);
        if (!delta) {
            return make_error<std::vector<Rational>>(delta.error());
        }
        minors.push_back(*delta);
    }
    return minors;
}

auto is_hurwitz_stable(const RationalPoly& char_poly) -> Result<bool> {
    auto minors = hurwitz_minors(char_poly);
    if (!minors) {
        return make_error<bool>(minors.error());
    }
    for (const Rational& delta : *minors) {
        if (rsign_of(delta) <= 0) {
            return false;  // a non-positive leading minor breaks Hurwitz positivity
        }
    }
    return true;  // includes the vacuous degree-0 (no minors) case
}

// --- Kharitonov robust stability --------------------------------------------

auto kharitonov_polynomials(std::span<const Rational> lower, std::span<const Rational> upper)
    -> Result<std::array<RationalPoly, 4>> {
    if (lower.empty() || lower.size() != upper.size()) {
        return make_error<std::array<RationalPoly, 4>>(MathError::domain_error);
    }
    for (std::size_t i = 0; i < lower.size(); ++i) {
        auto diff = upper[i].subtract(lower[i]);
        if (!diff) {
            return make_error<std::array<RationalPoly, 4>>(diff.error());  // surface overflow, not lo>hi
        }
        if (rsign_of(*diff) < 0) {
            return make_error<std::array<RationalPoly, 4>>(MathError::domain_error);  // lo > hi
        }
    }
    // Sign patterns per i mod 4 (true = upper bound):
    //   K1 (-,-,+,+)  K2 (+,+,-,-)  K3 (+,-,-,+)  K4 (-,+,+,-)
    constexpr std::array<bool, 4> k1{false, false, true, true};
    constexpr std::array<bool, 4> k2{true, true, false, false};
    constexpr std::array<bool, 4> k3{true, false, false, true};
    constexpr std::array<bool, 4> k4{false, true, true, false};
    return std::array<RationalPoly, 4>{kharitonov_one(lower, upper, k1),
                                       kharitonov_one(lower, upper, k2),
                                       kharitonov_one(lower, upper, k3),
                                       kharitonov_one(lower, upper, k4)};
}

auto is_robustly_stable(std::span<const Rational> lower, std::span<const Rational> upper)
    -> Result<bool> {
    auto polys = kharitonov_polynomials(lower, upper);
    if (!polys) {
        return make_error<bool>(polys.error());
    }
    for (const RationalPoly& k : *polys) {
        auto stable = is_hurwitz_stable(k);  // EXACT determinant test on rational endpoints
        if (!stable) {
            return make_error<bool>(stable.error());
        }
        if (!*stable) {
            return false;  // one unstable vertex => the family is not robustly stable
        }
    }
    return true;
}

// --- Lyapunov stability ------------------------------------------------------

auto is_positive_definite(const Matrix& p) -> Result<bool> {
    if (!p.is_square()) {
        return make_error<bool>(MathError::domain_error);
    }
    const std::size_t n = p.rows();
    // Sylvester's leading-principal-minor criterion certifies positive-definiteness only for a
    // SYMMETRIC matrix (for a non-symmetric P the minors say nothing about x^T P x, whose sign
    // is governed by the symmetric part). Enforce the documented precondition rather than
    // returning a wrong verdict for non-symmetric input (internal Lyapunov callers pass P=P^T).
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (!(p.at(i, j) == p.at(j, i))) {
                return make_error<bool>(MathError::domain_error);
            }
        }
    }
    std::vector<std::vector<Rational>> rows(n, std::vector<Rational>(n));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            rows[i][j] = p.at(i, j);
        }
    }
    for (std::size_t k = 1; k <= n; ++k) {  // Sylvester's criterion
        auto delta = leading_minor(rows, k);
        if (!delta) {
            return make_error<bool>(delta.error());
        }
        if (rsign_of(*delta) <= 0) {
            return false;
        }
    }
    return true;  // 0 x 0 is vacuously positive definite
}

auto lyapunov_solve(const Matrix& a) -> Result<Matrix> {
    if (!a.is_square()) {
        return make_error<Matrix>(MathError::domain_error);
    }
    const std::size_t n = a.rows();
    if (n == 0) {
        return Matrix::zero(0, 0);
    }
    const std::size_t nn = n * n;

    // Assemble the Kronecker-sum operator M = (I ⊗ Aᵀ + Aᵀ ⊗ I) row by row: the (r, c)
    // equation of Aᵀ P + P A = -I reads
    //   Σ_k A[k][r] P[k][c] + Σ_k P[r][k] A[k][c] = -δ_{rc},
    // with the unknown P[u][v] carried at index u*n + v (row-major vec). Overflow-checked.
    std::vector<std::vector<Rational>> m(nn, std::vector<Rational>(nn, Rational::from_int(0)));
    std::vector<std::vector<Rational>> rhs(nn, std::vector<Rational>(1, Rational::from_int(0)));
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t col = 0; col < n; ++col) {
            const std::size_t eq = r * n + col;
            for (std::size_t k = 0; k < n; ++k) {
                // term Σ_k A[k][r] P[k][col] -> unknown (k, col)
                auto acc1 = m[eq][k * n + col].add(a.at(k, r));
                if (!acc1) {
                    return make_error<Matrix>(acc1.error());
                }
                m[eq][k * n + col] = *acc1;
                // term Σ_k P[r][k] A[k][col] -> unknown (r, k)
                auto acc2 = m[eq][r * n + k].add(a.at(k, col));
                if (!acc2) {
                    return make_error<Matrix>(acc2.error());
                }
                m[eq][r * n + k] = *acc2;
            }
            if (r == col) {
                rhs[eq][0] = Rational::from_int(-1);
            }
        }
    }

    auto m_mat = Matrix::from_rows(std::move(m));
    if (!m_mat) {
        return make_error<Matrix>(m_mat.error());
    }
    auto rhs_mat = Matrix::from_rows(std::move(rhs));
    if (!rhs_mat) {
        return make_error<Matrix>(rhs_mat.error());
    }
    auto x = m_mat->solve(*rhs_mat);  // domain_error if the Kronecker sum is singular
    if (!x) {
        return make_error<Matrix>(x.error());
    }

    // Reshape vec(P) back into the n x n matrix P[u][v] = x[u*n + v].
    std::vector<std::vector<Rational>> p(n, std::vector<Rational>(n));
    for (std::size_t u = 0; u < n; ++u) {
        for (std::size_t v = 0; v < n; ++v) {
            p[u][v] = x->at(u * n + v, 0);
        }
    }
    return Matrix::from_rows(std::move(p));
}

auto is_stable_lyapunov(const Matrix& a) -> Result<bool> {
    if (!a.is_square()) {
        return make_error<bool>(MathError::domain_error);
    }
    if (a.rows() == 0) {
        return true;  // vacuously stable
    }
    auto p = lyapunov_solve(a);
    if (!p) {
        // A singular Kronecker sum (no unique P) means some λ_i + λ_j = 0 — the system is
        // not asymptotically stable. Any other error (e.g. overflow) is propagated.
        if (p.error() == MathError::domain_error) {
            return false;
        }
        return make_error<bool>(p.error());
    }
    return is_positive_definite(*p);
}

// --- Nyquist stability criterion --------------------------------------------

auto nyquist_criterion(const TransferFunction& open_loop, std::span<const double> omegas)
    -> Result<NyquistResult> {
    if (omegas.empty()) {
        return make_error<NyquistResult>(MathError::domain_error);
    }

    // P: open-loop RHP poles, counted from the rational poles (exact when fully extracted).
    auto pls = open_loop.poles();
    if (!pls) {
        return make_error<NyquistResult>(pls.error());
    }
    std::int64_t p_count = 0;
    for (const auto& [value, mult] : pls->rational) {
        if (rsign_of(value) > 0) {  // real rational pole with Re > 0
            p_count += mult;
        }
    }

    // Numerical Nyquist trace L(iω) over the positive grid.
    const auto trace = nyquist(open_loop, omegas);

    // Build the full mirror-completed contour ω : -∞ → +∞ as the conjugates in reverse
    // followed by the positive-ω points, and accumulate the signed angle swept about the
    // -1 point. The counter-clockwise winding number is (total angle)/2π; Nyquist counts
    // CLOCKWISE encirclements, so N = -winding_ccw.
    std::vector<std::complex<double>> contour;
    contour.reserve(2 * trace.size());
    for (std::size_t i = trace.size(); i-- > 0;) {
        contour.emplace_back(trace[i].re, -trace[i].im);  // ω < 0: conjugate
    }
    for (const NyquistPoint& pt : trace) {
        contour.emplace_back(pt.re, pt.im);  // ω > 0
    }

    double swept = 0.0;
    const std::complex<double> center{-1.0, 0.0};
    for (std::size_t i = 0; i + 1 < contour.size(); ++i) {
        const std::complex<double> v0 = contour[i] - center;
        const std::complex<double> v1 = contour[i + 1] - center;
        if (v0 == std::complex<double>{0.0, 0.0} || v1 == std::complex<double>{0.0, 0.0}) {
            continue;  // curve passes through -1: encirclement count is ill-defined here
        }
        swept += std::arg(v1 / v0);  // signed angle increment in (-π, π]
    }
    const double winding_ccw = swept / (2.0 * std::numbers::pi);
    const std::int64_t n_clockwise = -static_cast<std::int64_t>(std::llround(winding_ccw));

    NyquistResult result;
    result.encirclements = n_clockwise;
    result.open_loop_rhp_poles = p_count;
    result.closed_loop_rhp_poles = n_clockwise + p_count;
    result.closed_loop_stable = (result.closed_loop_rhp_poles == 0);
    result.p_exact = pls->fully_extracted();
    return result;
}

// --- gain & phase margins ----------------------------------------------------

auto stability_margins(const TransferFunction& open_loop, std::span<const double> omegas)
    -> StabilityMargins {
    StabilityMargins margins;
    const auto data = bode(open_loop, omegas);
    if (data.size() < 2) {
        return margins;
    }

    // Unwrap the phase so the -180° crossing is detectable across the atan2 branch cut.
    std::vector<double> phase(data.size());
    phase[0] = data[0].phase_deg;
    for (std::size_t i = 1; i < data.size(); ++i) {
        double delta = data[i].phase_deg - data[i - 1].phase_deg;
        while (delta > 180.0) {
            delta -= 360.0;
        }
        while (delta < -180.0) {
            delta += 360.0;
        }
        phase[i] = phase[i - 1] + delta;
    }

    // Gain crossover: first segment where the magnitude (dB) crosses zero. Linearly
    // interpolate ω_gc and the phase there; phase margin = 180° + phase.
    for (std::size_t i = 0; i + 1 < data.size(); ++i) {
        const double m0 = data[i].magnitude_db;
        const double m1 = data[i + 1].magnitude_db;
        if ((m0 >= 0.0 && m1 < 0.0) || (m0 <= 0.0 && m1 > 0.0)) {
            const double t = m0 / (m0 - m1);  // fraction to the zero crossing
            margins.gain_crossover =
                data[i].omega + t * (data[i + 1].omega - data[i].omega);
            const double ph = phase[i] + t * (phase[i + 1] - phase[i]);
            margins.phase_margin = 180.0 + ph;
            margins.has_phase_margin = true;
            break;
        }
    }

    // Phase crossover: first segment where the unwrapped phase crosses -180°. Interpolate
    // ω_pc and the magnitude there; gain margin (dB) = -magnitude_db, linear = 1/|L|.
    for (std::size_t i = 0; i + 1 < data.size(); ++i) {
        const double p0 = phase[i];
        const double p1 = phase[i + 1];
        if ((p0 >= -180.0 && p1 < -180.0) || (p0 <= -180.0 && p1 > -180.0)) {
            const double t = (p0 - (-180.0)) / (p0 - p1);
            margins.phase_crossover =
                data[i].omega + t * (data[i + 1].omega - data[i].omega);
            const double mag_db = data[i].magnitude_db +
                                  t * (data[i + 1].magnitude_db - data[i].magnitude_db);
            margins.gain_margin_db = -mag_db;
            margins.gain_margin = std::pow(10.0, margins.gain_margin_db / 20.0);
            margins.has_gain_margin = true;
            break;
        }
    }

    return margins;
}

}  // namespace nimblecas
