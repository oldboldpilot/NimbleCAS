// NimbleCAS stochastic-process analysis (ROADMAP §7.8 stationary-process slice).
// @author Olumuyiwa Oluwasanmi
//
// This module is STOCHASTIC-PROCESS ANALYSIS: the *structure* of a stationary process
// -- its stationary law, its correlation in time, and its spectrum. It is deliberately
// distinct from the numerical sampling modules (nimblecas.sde simulates SDE paths,
// nimblecas.montecarlo / nimblecas.mcmc draw samples). Nothing here draws a random
// number; everything is an exact algebraic statement about the process (or, where the
// mathematics leaves the field Q, an honestly-labelled numerical evaluation).
//
// It covers three families:
//
//   1. FINITE-STATE MARKOV CHAINS. A discrete-time chain is a row-stochastic rational
//      transition matrix P (rows sum to 1, entries >= 0, checked EXACTLY over Q). The
//      stationary distribution pi (pi P = pi, sum pi = 1) is the left Perron eigenvector,
//      obtained by an EXACT rational linear solve of (P^T - I) augmented with the
//      normalisation row -- not a power iteration. n-step transitions P^n, irreducibility
//      and aperiodicity (graph reachability / primitivity over Q), and mean first-passage
//      / recurrence times via the fundamental matrix Z = (I - P + W)^{-1} are likewise
//      EXACT over Q. A continuous-time chain is a rational generator Q (rows sum to 0,
//      off-diagonals >= 0); its stationary pi solves pi Q = 0, again an EXACT solve.
//
//   2. WIDE-SENSE-STATIONARY (WSS) SAMPLE ANALYSIS. For a finite record of rational
//      samples the mean, the biased/unbiased autocovariance R(tau) and autocorrelation
//      rho(tau), and the cross-covariance of two records are EXACT over Q. is_wss() is a
//      finite-record DIAGNOSTIC: it splits the record and checks that the mean and the
//      lagged autocovariances agree between halves. HONESTY: a finite sample can only
//      DIAGNOSE wide-sense stationarity of the record in hand -- it can never certify the
//      stationarity of the underlying process law, which is a statement about an infinite
//      ensemble. A normalised cross-CORRELATION coefficient needs sqrt(var_x var_y) and
//      so is NOT exact over Q; the exact deliverable here is the cross-COVARIANCE.
//
//   3. AR / MA / ARMA PROCESSES. The Yule-Walker equations for an AR(p) process are an
//      EXACT rational linear system, so the autocorrelation sequence rho(k), and thence
//      the theoretical autocovariance gamma(k) and the process variance gamma(0), are
//      EXACT over Q. The MA(q) autocovariance is an EXACT finite convolution of the MA
//      coefficients. Stationarity (AR) and invertibility (MA) are decided from the roots
//      of the characteristic polynomial: RATIONAL roots are found exactly (nimblecas.roots)
//      and tested against the unit circle exactly; when irrational/complex roots remain
//      unenumerated the verdict is honestly reported as `indeterminate`.
//
//   4. POWER SPECTRAL DENSITY (Wiener-Khinchin). S(f) is the Fourier transform of the
//      autocovariance. This is the one place the mathematics leaves Q: the transform is a
//      trigonometric sum, so power_spectral_density() and arma_spectral_density() are
//      NUMERICAL (double). They return (frequency, value) arrays over a one-sided grid
//      [0, 1/2]. The AR/MA closed form S(f) = sigma^2 |Theta(e^{-iw})|^2 / |Phi(e^{-iw})|^2
//      is a rational function of e^{-iw} evaluated numerically.
//
//   5. ERGODICITY. For an irreducible, aperiodic finite chain the ergodic theorem states
//      the time average (1/T) sum_t f(X_t) converges almost surely to the space average
//      sum_i pi_i f(i). ergodic_mean() returns that stationary (space) average EXACTLY
//      over Q; the convergence of the time average to it is the ergodic theorem, which we
//      state but (being a limit statement) do not and cannot certify from a finite path.
//
// HONESTY BOUNDARY (documented and true). EXACT over Q: is_stochastic, the Markov
// stationary distribution, P^n, irreducibility/aperiodicity, mean first-passage /
// fundamental-matrix quantities, the CTMC stationary distribution, the autocovariance /
// autocorrelation / cross-covariance of rational records, the Yule-Walker
// autocorrelation and AR/MA autocovariance, and the unit-circle test of RATIONAL
// characteristic roots. NUMERICAL (double): every power-spectral-density / frequency-
// domain value. DIAGNOSTIC only: is_wss on a finite record. We state this and do not
// overclaim.
//
// Conforms to config/cpp_details.txt: C++23 modules, `import std`, trailing return types,
// no owning raw pointers, std::expected railway error handling via make_error (no
// exceptions), [[nodiscard]] throughout, exact overflow-checked Rational arithmetic.

export module nimblecas.stochastic;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.stats;
import nimblecas.roots;

export namespace nimblecas {

// ===========================================================================
// Discrete-time finite-state Markov chains (EXACT over Q).
// ===========================================================================

// True iff P is a row-stochastic matrix over Q: square, every entry >= 0, and every row
// summing EXACTLY to 1. A non-square matrix is simply not row-stochastic and yields
// false (it is not an error to ask). Propagates overflow from the exact row sums.
[[nodiscard]] auto is_stochastic(const Matrix& p) -> Result<bool>;

// The stationary distribution pi of a discrete-time chain: the unique probability row
// vector with pi P = pi and sum_i pi_i = 1. Solved EXACTLY as the left null space of
// (P^T - I): the last row of that system is replaced by the normalisation 1^T pi = 1 and
// the resulting n x n system is solved over Q (Matrix::solve). Fails with domain_error
// when P is not row-stochastic or the system is singular (no unique stationary law, e.g.
// a reducible chain with several stationary distributions).
[[nodiscard]] auto stationary_distribution(const Matrix& p) -> Result<std::vector<Rational>>;

// The n-step transition matrix P^n, formed by exact binary exponentiation. P^0 is the
// identity. Requires a square P (domain_error otherwise); P need not be stochastic for
// the raw matrix power, but the probabilistic reading assumes it is. Propagates overflow.
[[nodiscard]] auto n_step_transition(const Matrix& p, std::size_t n) -> Result<Matrix>;

// True iff the chain is irreducible: every state reaches every other state (the directed
// graph of positive transitions is strongly connected). Computed by transitive closure
// over Q (an entry "exists" iff it is nonzero -- exact, no tolerance). Requires a
// non-empty square P (domain_error otherwise).
[[nodiscard]] auto is_irreducible(const Matrix& p) -> Result<bool>;

// True iff an IRREDUCIBLE chain is aperiodic (period 1). Decided by primitivity: an
// irreducible chain is aperiodic iff some boolean power of its adjacency matrix is
// strictly positive, which by Wielandt's bound occurs within (n-1)^2 + 1 steps if at all.
// Requires an irreducible chain; a reducible chain has per-class periods and yields
// domain_error ("aperiodicity where tractable").
[[nodiscard]] auto is_aperiodic(const Matrix& p) -> Result<bool>;

// The fundamental matrix Z = (I - P + W)^{-1} of an ergodic chain, where W has every row
// equal to the stationary distribution pi. EXACT over Q. Fails with domain_error when pi
// does not exist (see stationary_distribution) or I - P + W is singular.
[[nodiscard]] auto fundamental_matrix(const Matrix& p) -> Result<Matrix>;

// The matrix M of mean first-passage times: M(i, j) is the expected number of steps to
// first reach state j starting from state i (i != j), and the diagonal M(j, j) = 1/pi_j
// is the mean recurrence time of state j. From the fundamental matrix,
// M(i, j) = (Z(j, j) - Z(i, j)) / pi_j. EXACT over Q. Requires an ergodic chain with a
// stationary distribution (domain_error otherwise).
[[nodiscard]] auto mean_first_passage_times(const Matrix& p) -> Result<Matrix>;

// The stationary (space) average sum_i pi_i f(i) of a state function f, given as one
// rational value per state. By the ergodic theorem this is the almost-sure limit of the
// time average (1/T) sum_t f(X_t) for an irreducible aperiodic chain. EXACT over Q.
// `values` must have one entry per state (domain_error otherwise), and pi must exist.
[[nodiscard]] auto ergodic_mean(const Matrix& p, std::span<const Rational> values)
    -> Result<Rational>;

// ===========================================================================
// Continuous-time Markov chains (EXACT over Q).
// ===========================================================================

// True iff Q is a valid CTMC generator over Q: square, every off-diagonal entry >= 0, and
// every row summing EXACTLY to 0 (so each diagonal entry is minus the off-diagonal row
// sum). A non-square matrix yields false. Propagates overflow from the exact row sums.
[[nodiscard]] auto is_generator(const Matrix& q) -> Result<bool>;

// The stationary distribution pi of a continuous-time chain: the probability row vector
// with pi Q = 0 and sum_i pi_i = 1. Solved EXACTLY as the left null space of Q with the
// last equation replaced by the normalisation, over Q. Fails with domain_error when Q is
// not a generator or the system is singular.
[[nodiscard]] auto ctmc_stationary_distribution(const Matrix& q) -> Result<std::vector<Rational>>;

// ===========================================================================
// Absorbing chains, random walks, the resolvent, birth-death (EXACT over Q).
// ===========================================================================

// The exact canonical-form analysis of an absorbing Markov chain. A state i is ABSORBING
// iff P(i, i) = 1 (row-stochasticity then forces the rest of that row to 0); every other
// state is TRANSIENT. `absorbing` and `transient` list the original state indices in each
// class (ascending). Relabelling the transient states first exposes the canonical blocks
// P = [[Q, R], [0, I]]: `fundamental` is N = (I - Q)^{-1} (t x t, rows/cols indexed by
// `transient`), `expected_steps[a]` = (N 1)_a is the expected number of steps to
// absorption starting from transient state `transient[a]`, and `absorption_probabilities`
// is B = N R (t x r, rows indexed by `transient`, columns by `absorbing`) whose (a, c)
// entry is the probability of being absorbed in `absorbing[c]` starting from
// `transient[a]`. All EXACT over Q.
struct AbsorbingChain {
    std::vector<std::size_t> transient;    // original indices of the transient states
    std::vector<std::size_t> absorbing;    // original indices of the absorbing states
    Matrix fundamental;                    // N = (I - Q)^{-1}
    std::vector<Rational> expected_steps;  // t = N 1
    Matrix absorption_probabilities;       // B = N R
};

// The absorbing-chain analysis of a row-stochastic P (see AbsorbingChain). EXACT over Q via
// Matrix::inverse. Fails with domain_error when P is not row-stochastic, has no absorbing
// state (not an absorbing chain), or when I - Q is singular (a transient class that never
// reaches absorption -- no finite expected absorption time).
[[nodiscard]] auto absorbing_analysis(const Matrix& p) -> Result<AbsorbingChain>;

// The resolvent (s I - Q)^{-1} evaluated EXACTLY at a given rational s. Q is any square
// rational matrix (a CTMC generator, or the transient block of an absorbing chain). Fails
// with domain_error when Q is not square, or when s I - Q is singular (s is an eigenvalue
// of Q -- the resolvent has a pole there). This is the numeric resolvent at a point; the
// fully symbolic (s I - Q)^{-1} as a matrix of rational functions of s is not formed here.
[[nodiscard]] auto resolvent(const Matrix& q, const Rational& s) -> Result<Matrix>;

// The EXACT ruin probability of a simple biased random walk on the states {0, 1, ..., n}
// with up-probability p (down-probability 1 - p): starting from state k, the probability of
// reaching the absorbing barrier 0 before the absorbing barrier n. For p != 1/2 with
// r = (1 - p)/p, it is (r^k - r^n)/(1 - r^n); for p = 1/2 it is the limit (n - k)/n. EXACT
// over Q. Boundaries: k = 0 gives 1, k = n gives 0. Fails with domain_error when n = 0,
// k > n, or p is not strictly inside (0, 1).
[[nodiscard]] auto gamblers_ruin_probability(std::size_t n, std::size_t k, const Rational& p)
    -> Result<Rational>;

// The EXACT expected duration (mean number of steps to absorption at 0 or n) of the same
// biased random walk started from state k. For p = 1/2 it is k(n - k); for p != 1/2 with
// r = (1 - p)/p and q = 1 - p it is (1/(q - p)) [k - n (1 - r^k)/(1 - r^n)]. EXACT over Q.
// Boundaries k = 0 and k = n give 0. Fails with domain_error as gamblers_ruin_probability.
[[nodiscard]] auto gamblers_ruin_duration(std::size_t n, std::size_t k, const Rational& p)
    -> Result<Rational>;

// The EXACT stationary distribution of a birth-death chain on states {0, ..., n-1} from its
// birth rates lambda_0..lambda_{n-2} (`birth[i]` = rate i -> i+1) and death rates
// mu_1..mu_{n-1} (`death[i]` = rate (i+1) -> i). By detailed balance pi_i is proportional to
// prod_{j=1}^{i} lambda_{j-1}/mu_j (pi_0 proportional to 1), normalised to sum 1. `birth`
// and `death` must have equal length L (= n - 1); an empty pair is the single state
// pi = [1]. EXACT over Q. Fails with domain_error on unequal lengths or a negative rate, and
// with division_by_zero if an interior death rate is 0.
[[nodiscard]] auto birth_death_stationary(std::span<const Rational> birth,
                                          std::span<const Rational> death)
    -> Result<std::vector<Rational>>;

// ===========================================================================
// Wide-sense-stationary (WSS) sample analysis (EXACT over Q; is_wss is diagnostic).
// ===========================================================================

// The exact autocovariance at a single lag of a finite rational record:
// R(tau) = (1/D) sum_{t} (x_t - xbar)(x_{t+tau} - xbar), where the sum runs over the
// N - tau overlapping pairs and D = N (biased) or D = N - tau (unbiased). R(0) with the
// biased normalisation is exactly the population variance. Fails with domain_error when
// the record is empty or lag >= N (too short for even one lagged product).
[[nodiscard]] auto autocovariance_at(std::span<const Rational> x, std::size_t lag, bool biased)
    -> Result<Rational>;

// The autocovariance sequence R(0), R(1), ..., R(max_lag) (exact). See autocovariance_at
// for the normalisation; fails if max_lag >= N.
[[nodiscard]] auto autocovariance(std::span<const Rational> x, std::size_t max_lag, bool biased)
    -> Result<std::vector<Rational>>;

// The autocorrelation sequence rho(tau) = R(tau) / R(0), tau = 0..max_lag (exact); always
// rho(0) = 1. Fails with division_by_zero for a constant record (R(0) = 0), or
// domain_error if max_lag >= N.
[[nodiscard]] auto autocorrelation(std::span<const Rational> x, std::size_t max_lag, bool biased)
    -> Result<std::vector<Rational>>;

// The exact cross-covariance at a single (signed) lag of two equal-length records:
// R_xy(lag) = (1/D) sum_t (x_t - xbar)(y_{t+lag} - ybar), with the sum over the |N - lag|
// overlapping pairs and D = N (biased) or D = N - |lag| (unbiased). Positive lag leads y;
// negative lag leads x. Fails with domain_error on unequal lengths, an empty record, or
// |lag| >= N. (A normalised cross-correlation coefficient needs sqrt(var_x var_y) and is
// therefore NOT exact over Q; this exact routine returns the cross-COVARIANCE.)
[[nodiscard]] auto cross_covariance_at(std::span<const Rational> x, std::span<const Rational> y,
                                       std::int64_t lag, bool biased) -> Result<Rational>;

// The cross-covariance sequence R_xy(0..max_lag) (exact, non-negative lags). Fails as
// cross_covariance_at.
[[nodiscard]] auto cross_covariance(std::span<const Rational> x, std::span<const Rational> y,
                                    std::size_t max_lag, bool biased)
    -> Result<std::vector<Rational>>;

// A finite-record WSS DIAGNOSTIC: split the record into two equal halves and return true
// iff their means agree EXACTLY and their biased autocovariances agree EXACTLY at every
// lag 0..max_lag. This checks the two defining features of wide-sense stationarity -- a
// constant mean and a lag-only autocovariance -- on the record in hand. It is a
// diagnostic of the SAMPLE, NOT a proof of the process law (an infinite-ensemble
// statement a finite record cannot certify). Fails with domain_error when a half is empty
// or max_lag is too large for a half.
[[nodiscard]] auto is_wss(std::span<const Rational> x, std::size_t max_lag) -> Result<bool>;

// ===========================================================================
// AR / MA / ARMA processes (autocorrelation & autocovariance EXACT over Q).
// ===========================================================================

// The verdict of a root-location test against the unit circle, with an honest third state
// for the case where irrational/complex roots cannot be enumerated exactly over Q.
enum class StabilityCertificate : std::uint8_t {
    stable,        // every root proven strictly outside the unit circle
    unstable,      // a rational root found on or inside the unit circle
    indeterminate  // no inside/on-circle rational root found, but roots remain unenumerated
};

// The autocorrelation sequence rho(0..max_lag) of an AR(p) process x_t = sum_k phi_k
// x_{t-k} + w_t, from the Yule-Walker equations. rho(1..p) solve the EXACT p x p rational
// linear system rho(k) = sum_j phi_j rho(|k-j|) (with rho(0) = 1), and rho(k) for k > p is
// continued by the same recursion. EXACT over Q; rho(0) = 1. `ar` holds phi_1..phi_p (an
// empty span is white noise: rho = [1, 0, 0, ...]). Fails with domain_error if the
// Yule-Walker system is singular (a non-stationary parameterisation).
[[nodiscard]] auto yule_walker_autocorrelation(std::span<const Rational> ar, std::size_t max_lag)
    -> Result<std::vector<Rational>>;

// The theoretical autocovariance gamma(0..max_lag) of an AR(p) process with white-noise
// variance sigma^2. gamma(0) = sigma^2 / (1 - sum_j phi_j rho(j)) is the process variance,
// and gamma(k) = gamma(0) rho(k). EXACT over Q. Fails as yule_walker_autocorrelation, or
// with division_by_zero if the variance denominator vanishes.
[[nodiscard]] auto ar_autocovariance(std::span<const Rational> ar, const Rational& sigma2,
                                     std::size_t max_lag) -> Result<std::vector<Rational>>;

// The theoretical autocovariance gamma(0..max_lag) of an MA(q) process
// x_t = w_t + sum_{k=1}^q theta_k w_{t-k} with white-noise variance sigma^2:
// gamma(tau) = sigma^2 sum_{k=0}^{q-tau} theta_k theta_{k+tau} (theta_0 = 1) for
// 0 <= tau <= q, and gamma(tau) = 0 for tau > q. EXACT over Q. `ma` holds theta_1..theta_q.
[[nodiscard]] auto ma_autocovariance(std::span<const Rational> ma, const Rational& sigma2,
                                     std::size_t max_lag) -> Result<std::vector<Rational>>;

// Stationarity of an AR(p) process, decided from the roots of Phi(z) = 1 - sum_k phi_k z^k:
// the process is stationary iff every root lies strictly OUTSIDE the unit circle. Rational
// roots are found exactly and tested (|num| vs den); if every root of Phi is rational the
// verdict is definitive (stable/unstable), otherwise unenumerated irrational/complex roots
// make it `indeterminate` (unless a rational root already proves it unstable). `ar` holds
// phi_1..phi_p (empty => white noise => stable).
[[nodiscard]] auto ar_is_stationary(std::span<const Rational> ar) -> Result<StabilityCertificate>;

// Invertibility of an MA(q) process, decided from the roots of Theta(z) = 1 + sum_k
// theta_k z^k against the unit circle, with the same exact-rational / indeterminate
// contract as ar_is_stationary. `ma` holds theta_1..theta_q.
[[nodiscard]] auto ma_is_invertible(std::span<const Rational> ma) -> Result<StabilityCertificate>;

// ===========================================================================
// ARIMA differencing & partial autocorrelation (EXACT over Q).
// ===========================================================================

// The d-th backshift difference nabla^d x of a rational series (EXACT). One difference maps
// x_0..x_{N-1} to (x_1 - x_0), ..., (x_{N-1} - x_{N-2}) (length N - 1); nabla^d applies this
// d times, shortening the series by d. nabla^0 x is a copy. Fails with domain_error if a
// difference is requested of an empty series (d > N).
[[nodiscard]] auto difference(std::span<const Rational> x, std::size_t d)
    -> Result<std::vector<Rational>>;

// The EXACT inverse of nabla^d: integrate (cumulatively sum) the d-th difference `y` back to
// the original series, using the d constants of integration `initial` = [x_0, (nabla x)_0,
// ..., (nabla^{d-1} x)_0] -- the leading value dropped at each differencing level, outermost
// first. Reconstructs a series of length y.size() + d. With initial = difference-level
// leading values, integrate(difference(x, d), initial) == x. An empty `initial` (d = 0) is a
// copy of y. Fails only with overflow from the running sums.
[[nodiscard]] auto integrate(std::span<const Rational> y, std::span<const Rational> initial)
    -> Result<std::vector<Rational>>;

// The EXACT partial autocorrelation sequence (PACF) phi_{1,1}, ..., phi_{m,m} from the
// autocovariances gamma(0..m) via the Durbin-Levinson recursion (m = autocov.size() - 1).
// phi_{k,k} is the correlation between x_t and x_{t+k} with the intervening k-1 lags
// regressed out; each is a rational function of the (rational) autocovariances, hence EXACT
// over Q. Because the recursion uses only ratios, feeding a rational autocorrelation
// sequence (gamma(0) = 1, e.g. from yule_walker_autocorrelation) yields the same PACF. An
// AR(p) process has phi_{k,k} = 0 for k > p. Returns an empty sequence when only gamma(0) is
// supplied. Fails with domain_error on an empty input, and with division_by_zero when
// gamma(0) = 0 or a prediction-error variance vanishes (a deterministic partial correlation
// of magnitude 1).
[[nodiscard]] auto partial_autocorrelation(std::span<const Rational> autocov)
    -> Result<std::vector<Rational>>;

// ===========================================================================
// Power spectral density (NUMERICAL, double -- the Fourier boundary).
// ===========================================================================

// A sampled spectrum: value[k] is the spectral density at frequency[k] (cycles/sample,
// one-sided over [0, 1/2]).
struct Spectrum {
    std::vector<double> frequencies;
    std::vector<double> values;
};

// Convert exact rational values to double (num/den). The point at which exact analysis
// hands off to the NUMERICAL spectral routines.
[[nodiscard]] auto to_doubles(std::span<const Rational> xs) -> std::vector<double>;

// The power spectral density by Wiener-Khinchin from a one-sided autocovariance
// R(0..m): S(f) = R(0) + 2 sum_{tau=1}^m R(tau) cos(2 pi f tau), sampled at num_points
// frequencies on the one-sided grid f_k = (1/2) k/(num_points - 1). NUMERICAL. For white
// noise (R = [sigma^2, 0, ...]) S is flat at sigma^2. Fails with domain_error if the
// autocovariance is empty or num_points < 2.
[[nodiscard]] auto power_spectral_density(std::span<const double> autocov_onesided,
                                          std::size_t num_points) -> Result<Spectrum>;

// Convenience overload taking an exact rational autocovariance and evaluating S(f)
// numerically (see to_doubles / the double overload).
[[nodiscard]] auto power_spectral_density(std::span<const Rational> autocov_onesided,
                                          std::size_t num_points) -> Result<Spectrum>;

// The closed-form ARMA spectral density, evaluated NUMERICALLY:
// S(f) = sigma^2 |Theta(e^{-iw})|^2 / |Phi(e^{-iw})|^2, w = 2 pi f, with
// Phi(x) = 1 - sum_k phi_k x^k (AR) and Theta(x) = 1 + sum_k theta_k x^k (MA). Sampled on
// the one-sided grid as power_spectral_density. `ar`/`ma` may be empty (pure MA / pure AR
// / white noise). Fails with domain_error if num_points < 2, or not_implemented if Phi
// vanishes on the grid (a root on the unit circle -- a non-stationary AR part).
[[nodiscard]] auto arma_spectral_density(std::span<const Rational> ar, std::span<const Rational> ma,
                                         double sigma2, std::size_t num_points) -> Result<Spectrum>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// --- small exact helpers ----------------------------------------------------

[[nodiscard]] auto one_rat() -> Rational { return Rational::from_int(1); }

// |v| as a non-negative int64; v is a canonical Rational numerator so it is never
// INT64_MIN (Rational::make rejects that), making the negation safe.
[[nodiscard]] auto abs_i64(std::int64_t v) -> std::int64_t { return v < 0 ? -v : v; }

// Convert a single Rational to double.
[[nodiscard]] auto rat_to_double(const Rational& r) -> double {
    return static_cast<double>(r.numerator()) / static_cast<double>(r.denominator());
}

// The exact sum of one row of P (used by the stochastic / generator checks).
[[nodiscard]] auto row_sum(const Matrix& p, std::size_t i) -> Result<Rational> {
    Rational acc;  // 0/1
    for (std::size_t j = 0; j < p.cols(); ++j) {
        auto next = acc.add(p.at(i, j));
        if (!next) {
            return make_error<Rational>(next.error());
        }
        acc = *next;
    }
    return acc;
}

// Solve the "left null space + normalisation" system for a stationary distribution.
// `a` is the n x n coefficient grid whose left null space we seek (A = P^T - I for a
// DTMC, A = Q^T for a CTMC); only rows 0..n-2 are read -- the last row is overwritten with
// the all-ones normalisation and the RHS is e_{n-1}. Returns pi as a length-n vector.
// Fails with domain_error if the system is singular.
[[nodiscard]] auto solve_stationary(std::vector<std::vector<Rational>> a)
    -> Result<std::vector<Rational>> {
    const std::size_t n = a.size();
    if (n == 0) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    const Rational one = one_rat();
    for (std::size_t j = 0; j < n; ++j) {
        a[n - 1][j] = one;  // normalisation row: sum_j pi_j = 1
    }
    auto amat = Matrix::from_rows(std::move(a));
    if (!amat) {
        return make_error<std::vector<Rational>>(amat.error());
    }
    std::vector<std::vector<Rational>> brows(n, std::vector<Rational>(1));
    brows[n - 1][0] = one;  // RHS e_{n-1}
    auto bmat = Matrix::from_rows(std::move(brows));
    if (!bmat) {
        return make_error<std::vector<Rational>>(bmat.error());
    }
    auto x = amat->solve(*bmat);  // singular => domain_error
    if (!x) {
        return make_error<std::vector<Rational>>(x.error());
    }
    std::vector<Rational> pi(n);
    for (std::size_t i = 0; i < n; ++i) {
        pi[i] = x->at(i, 0);
    }
    return pi;
}

// --- boolean reachability helpers (exact graph structure) -------------------

using BMat = std::vector<std::vector<bool>>;

// Adjacency of P's positive transitions: adj[i][j] iff P(i, j) != 0 (exact, no tolerance).
[[nodiscard]] auto adjacency(const Matrix& p) -> BMat {
    const std::size_t n = p.rows();
    BMat adj(n, std::vector<bool>(n, false));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            adj[i][j] = !p.at(i, j).is_zero();
        }
    }
    return adj;
}

// Boolean matrix product c = a * b (OR of ANDs).
[[nodiscard]] auto bool_mul(const BMat& a, const BMat& b) -> BMat {
    const std::size_t n = a.size();
    BMat c(n, std::vector<bool>(n, false));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            if (!a[i][k]) {
                continue;
            }
            for (std::size_t j = 0; j < n; ++j) {
                if (b[k][j]) {
                    c[i][j] = true;
                }
            }
        }
    }
    return c;
}

[[nodiscard]] auto all_true(const BMat& m) -> bool {
    for (const auto& row : m) {
        for (bool v : row) {
            if (!v) {
                return false;
            }
        }
    }
    return true;
}

// --- small exact helpers for the random-walk / birth-death / resolvent additions ---

// base^exp over Q by exact binary exponentiation (exp == 0 => 1). Propagates overflow.
[[nodiscard]] auto rat_pow(const Rational& base, std::size_t exp) -> Result<Rational> {
    Rational result = one_rat();
    Rational b = base;
    std::size_t e = exp;
    while (e > 0) {
        if ((e & 1U) != 0U) {
            auto prod = result.multiply(b);
            if (!prod) {
                return prod;
            }
            result = *prod;
        }
        e >>= 1U;
        if (e > 0) {
            auto sq = b.multiply(b);
            if (!sq) {
                return sq;
            }
            b = *sq;
        }
    }
    return result;
}

// A std::size_t as an exact Rational; overflow if it exceeds INT64_MAX (the state counts /
// walk lengths here are tiny in practice, so this only guards a pathological input).
[[nodiscard]] auto size_to_rat(std::size_t v) -> Result<Rational> {
    if (v > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        return make_error<Rational>(MathError::overflow);
    }
    return Rational::from_int(static_cast<std::int64_t>(v));
}

}  // namespace

// --- discrete-time Markov chains --------------------------------------------

auto is_stochastic(const Matrix& p) -> Result<bool> {
    if (!p.is_square() || p.rows() == 0) {
        return false;
    }
    const Rational one = one_rat();
    for (std::size_t i = 0; i < p.rows(); ++i) {
        for (std::size_t j = 0; j < p.cols(); ++j) {
            if (p.at(i, j).numerator() < 0) {  // canonical: den > 0, so sign is the numerator's
                return false;
            }
        }
        auto s = row_sum(p, i);
        if (!s) {
            return make_error<bool>(s.error());
        }
        if (!(*s == one)) {
            return false;
        }
    }
    return true;
}

auto stationary_distribution(const Matrix& p) -> Result<std::vector<Rational>> {
    auto stoch = is_stochastic(p);
    if (!stoch) {
        return make_error<std::vector<Rational>>(stoch.error());
    }
    if (!*stoch) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    const std::size_t n = p.rows();
    // A = P^T - I:  A(i, j) = P(j, i) - [i == j].  (Last row overwritten by solve_stationary.)
    std::vector<std::vector<Rational>> a(n, std::vector<Rational>(n));
    for (std::size_t i = 0; i + 1 < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) {
                auto d = p.at(j, i).subtract(one_rat());
                if (!d) {
                    return make_error<std::vector<Rational>>(d.error());
                }
                a[i][j] = *d;
            } else {
                a[i][j] = p.at(j, i);
            }
        }
    }
    return solve_stationary(std::move(a));
}

auto n_step_transition(const Matrix& p, std::size_t n) -> Result<Matrix> {
    if (!p.is_square()) {
        return make_error<Matrix>(MathError::domain_error);
    }
    Matrix result = Matrix::identity(p.rows());  // P^0
    Matrix base = p;
    std::size_t e = n;
    while (e > 0) {
        if ((e & 1U) != 0U) {
            auto prod = result.multiply(base);
            if (!prod) {
                return prod;
            }
            result = std::move(*prod);
        }
        e >>= 1U;
        if (e > 0) {
            auto sq = base.multiply(base);
            if (!sq) {
                return sq;
            }
            base = std::move(*sq);
        }
    }
    return result;
}

auto is_irreducible(const Matrix& p) -> Result<bool> {
    if (!p.is_square() || p.rows() == 0) {
        return make_error<bool>(MathError::domain_error);
    }
    const std::size_t n = p.rows();
    BMat reach = adjacency(p);
    for (std::size_t i = 0; i < n; ++i) {
        reach[i][i] = true;  // reflexive: a state reaches itself in zero steps
    }
    // Warshall transitive closure.
    for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t i = 0; i < n; ++i) {
            if (!reach[i][k]) {
                continue;
            }
            for (std::size_t j = 0; j < n; ++j) {
                if (reach[k][j]) {
                    reach[i][j] = true;
                }
            }
        }
    }
    return all_true(reach);
}

auto is_aperiodic(const Matrix& p) -> Result<bool> {
    auto irr = is_irreducible(p);
    if (!irr) {
        return make_error<bool>(irr.error());
    }
    if (!*irr) {
        // Aperiodicity is a per-communicating-class property; undefined for the whole
        // reducible chain. Report honestly rather than guess.
        return make_error<bool>(MathError::domain_error);
    }
    const std::size_t n = p.rows();
    const BMat base = adjacency(p);  // no reflexive edges: genuine one-step moves
    // Primitivity (irreducible + aperiodic) <=> some power is strictly positive, reached
    // within Wielandt's bound (n-1)^2 + 1 if at all.
    const std::size_t bound = (n - 1) * (n - 1) + 1;
    BMat power = base;
    for (std::size_t k = 1; k <= bound; ++k) {
        if (all_true(power)) {
            return true;
        }
        power = bool_mul(power, base);
    }
    return false;
}

auto fundamental_matrix(const Matrix& p) -> Result<Matrix> {
    auto pi = stationary_distribution(p);
    if (!pi) {
        return make_error<Matrix>(pi.error());
    }
    const std::size_t n = p.rows();
    // W: every row equals pi.
    std::vector<std::vector<Rational>> wrows(n, *pi);
    auto w = Matrix::from_rows(std::move(wrows));
    if (!w) {
        return w;
    }
    auto i_minus_p = Matrix::identity(n).subtract(p);
    if (!i_minus_p) {
        return i_minus_p;
    }
    auto core = i_minus_p->add(*w);  // I - P + W
    if (!core) {
        return core;
    }
    return core->inverse();  // singular => domain_error
}

auto mean_first_passage_times(const Matrix& p) -> Result<Matrix> {
    auto pi = stationary_distribution(p);
    if (!pi) {
        return make_error<Matrix>(pi.error());
    }
    auto z = fundamental_matrix(p);
    if (!z) {
        return z;
    }
    const std::size_t n = p.rows();
    std::vector<std::vector<Rational>> m(n, std::vector<Rational>(n));
    for (std::size_t j = 0; j < n; ++j) {
        // Diagonal: mean recurrence time m(j, j) = 1 / pi_j.
        auto rec = one_rat().divide((*pi)[j]);  // pi_j == 0 => division_by_zero
        if (!rec) {
            return make_error<Matrix>(rec.error());
        }
        m[j][j] = *rec;
        for (std::size_t i = 0; i < n; ++i) {
            if (i == j) {
                continue;
            }
            // m(i, j) = (Z(j, j) - Z(i, j)) / pi_j.
            auto num = z->at(j, j).subtract(z->at(i, j));
            if (!num) {
                return make_error<Matrix>(num.error());
            }
            auto val = num->divide((*pi)[j]);
            if (!val) {
                return make_error<Matrix>(val.error());
            }
            m[i][j] = *val;
        }
    }
    return Matrix::from_rows(std::move(m));
}

auto ergodic_mean(const Matrix& p, std::span<const Rational> values) -> Result<Rational> {
    auto pi = stationary_distribution(p);
    if (!pi) {
        return make_error<Rational>(pi.error());
    }
    if (values.size() != pi->size()) {
        return make_error<Rational>(MathError::domain_error);
    }
    Rational acc;  // 0/1
    for (std::size_t i = 0; i < values.size(); ++i) {
        auto term = (*pi)[i].multiply(values[i]);
        if (!term) {
            return make_error<Rational>(term.error());
        }
        auto next = acc.add(*term);
        if (!next) {
            return make_error<Rational>(next.error());
        }
        acc = *next;
    }
    return acc;
}

// --- continuous-time Markov chains ------------------------------------------

auto is_generator(const Matrix& q) -> Result<bool> {
    if (!q.is_square() || q.rows() == 0) {
        return false;
    }
    const Rational zero;  // 0/1
    for (std::size_t i = 0; i < q.rows(); ++i) {
        for (std::size_t j = 0; j < q.cols(); ++j) {
            if (i != j && q.at(i, j).numerator() < 0) {  // off-diagonal rates are >= 0
                return false;
            }
        }
        auto s = row_sum(q, i);
        if (!s) {
            return make_error<bool>(s.error());
        }
        if (!(*s == zero)) {  // each row sums to 0
            return false;
        }
    }
    return true;
}

auto ctmc_stationary_distribution(const Matrix& q) -> Result<std::vector<Rational>> {
    auto gen = is_generator(q);
    if (!gen) {
        return make_error<std::vector<Rational>>(gen.error());
    }
    if (!*gen) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    const std::size_t n = q.rows();
    // A = Q^T:  A(i, j) = Q(j, i).  (pi Q = 0  <=>  Q^T pi^T = 0.  Last row overwritten below.)
    std::vector<std::vector<Rational>> a(n, std::vector<Rational>(n));
    for (std::size_t i = 0; i + 1 < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            a[i][j] = q.at(j, i);
        }
    }
    return solve_stationary(std::move(a));
}

// --- absorbing chains / random walks / resolvent / birth-death --------------

auto absorbing_analysis(const Matrix& p) -> Result<AbsorbingChain> {
    auto stoch = is_stochastic(p);
    if (!stoch) {
        return make_error<AbsorbingChain>(stoch.error());
    }
    if (!*stoch) {
        return make_error<AbsorbingChain>(MathError::domain_error);
    }
    const std::size_t n = p.rows();
    const Rational one = one_rat();
    AbsorbingChain out;
    for (std::size_t i = 0; i < n; ++i) {
        if (p.at(i, i) == one) {
            out.absorbing.push_back(i);  // P(i, i) = 1 => row is e_i => absorbing
        } else {
            out.transient.push_back(i);
        }
    }
    if (out.absorbing.empty()) {
        return make_error<AbsorbingChain>(MathError::domain_error);  // not an absorbing chain
    }
    const std::size_t t = out.transient.size();
    const std::size_t r = out.absorbing.size();
    // Build I - Q directly:  (I - Q)(a, b) = [a == b] - P(transient[a], transient[b]).
    std::vector<std::vector<Rational>> iq(t, std::vector<Rational>(t));
    for (std::size_t a = 0; a < t; ++a) {
        for (std::size_t b = 0; b < t; ++b) {
            const Rational& qab = p.at(out.transient[a], out.transient[b]);
            if (a == b) {
                auto d = one.subtract(qab);
                if (!d) {
                    return make_error<AbsorbingChain>(d.error());
                }
                iq[a][b] = *d;
            } else {
                auto neg = qab.negate();
                if (!neg) {
                    return make_error<AbsorbingChain>(neg.error());
                }
                iq[a][b] = *neg;
            }
        }
    }
    auto iqmat = Matrix::from_rows(std::move(iq));
    if (!iqmat) {
        return make_error<AbsorbingChain>(iqmat.error());
    }
    auto nmat = iqmat->inverse();  // singular => domain_error (no finite absorption)
    if (!nmat) {
        return make_error<AbsorbingChain>(nmat.error());
    }
    // expected steps to absorption t = N 1 (row sums of N).
    out.expected_steps.assign(t, Rational{});
    for (std::size_t a = 0; a < t; ++a) {
        Rational acc;  // 0/1
        for (std::size_t b = 0; b < t; ++b) {
            auto next = acc.add(nmat->at(a, b));
            if (!next) {
                return make_error<AbsorbingChain>(next.error());
            }
            acc = *next;
        }
        out.expected_steps[a] = acc;
    }
    // R block (t x r):  R(a, c) = P(transient[a], absorbing[c]).
    std::vector<std::vector<Rational>> rrows(t, std::vector<Rational>(r));
    for (std::size_t a = 0; a < t; ++a) {
        for (std::size_t c = 0; c < r; ++c) {
            rrows[a][c] = p.at(out.transient[a], out.absorbing[c]);
        }
    }
    auto rmat = Matrix::from_rows(std::move(rrows));
    if (!rmat) {
        return make_error<AbsorbingChain>(rmat.error());
    }
    auto bmat = nmat->multiply(*rmat);  // B = N R
    if (!bmat) {
        return make_error<AbsorbingChain>(bmat.error());
    }
    out.fundamental = std::move(*nmat);
    out.absorption_probabilities = std::move(*bmat);
    return out;
}

auto resolvent(const Matrix& q, const Rational& s) -> Result<Matrix> {
    if (!q.is_square()) {
        return make_error<Matrix>(MathError::domain_error);
    }
    auto si = Matrix::identity(q.rows()).scale(s);  // s I
    if (!si) {
        return si;
    }
    auto m = si->subtract(q);  // s I - Q
    if (!m) {
        return m;
    }
    return m->inverse();  // s an eigenvalue of Q => singular => domain_error
}

namespace {

// Validate the random-walk parameters shared by the two gambler's-ruin routines and, on
// success, hand back q = 1 - p. Requires n >= 1, k <= n, and p strictly inside (0, 1).
[[nodiscard]] auto ruin_setup(std::size_t n, std::size_t k, const Rational& p) -> Result<Rational> {
    if (n == 0 || k > n) {
        return make_error<Rational>(MathError::domain_error);
    }
    if (p.numerator() <= 0) {  // canonical den > 0, so this is p <= 0
        return make_error<Rational>(MathError::domain_error);
    }
    auto pm1 = p.subtract(one_rat());
    if (!pm1) {
        return make_error<Rational>(pm1.error());
    }
    if (pm1->numerator() >= 0) {  // p >= 1
        return make_error<Rational>(MathError::domain_error);
    }
    return one_rat().subtract(p);  // q = 1 - p
}

}  // namespace

auto gamblers_ruin_probability(std::size_t n, std::size_t k, const Rational& p) -> Result<Rational> {
    auto q = ruin_setup(n, k, p);
    if (!q) {
        return q;
    }
    if (k == 0) {
        return one_rat();  // already at the ruin barrier
    }
    if (k == n) {
        return Rational{};  // already at the winning barrier
    }
    if (p == *q) {  // symmetric p = 1/2: ruin probability (n - k)/n
        auto num = size_to_rat(n - k);
        if (!num) {
            return num;
        }
        auto den = size_to_rat(n);
        if (!den) {
            return den;
        }
        return num->divide(*den);
    }
    // p != 1/2:  (r^k - r^n)/(1 - r^n),  r = q/p.
    auto r = q->divide(p);
    if (!r) {
        return r;
    }
    auto rk = rat_pow(*r, k);
    if (!rk) {
        return rk;
    }
    auto rn = rat_pow(*r, n);
    if (!rn) {
        return rn;
    }
    auto num = rk->subtract(*rn);
    if (!num) {
        return num;
    }
    auto den = one_rat().subtract(*rn);
    if (!den) {
        return den;
    }
    return num->divide(*den);  // den != 0 since r != 1 (p != 1/2)
}

auto gamblers_ruin_duration(std::size_t n, std::size_t k, const Rational& p) -> Result<Rational> {
    auto q = ruin_setup(n, k, p);
    if (!q) {
        return q;
    }
    if (k == 0 || k == n) {
        return Rational{};  // already absorbed
    }
    auto kr = size_to_rat(k);
    if (!kr) {
        return kr;
    }
    auto nr = size_to_rat(n);
    if (!nr) {
        return nr;
    }
    if (p == *q) {  // symmetric p = 1/2: duration k(n - k)
        auto nmk = nr->subtract(*kr);
        if (!nmk) {
            return nmk;
        }
        return kr->multiply(*nmk);
    }
    // p != 1/2:  (1/(q - p)) [ k - n (1 - r^k)/(1 - r^n) ],  r = q/p.
    auto qmp = q->subtract(p);
    if (!qmp) {
        return qmp;
    }
    auto r = q->divide(p);
    if (!r) {
        return r;
    }
    auto rk = rat_pow(*r, k);
    if (!rk) {
        return rk;
    }
    auto rn = rat_pow(*r, n);
    if (!rn) {
        return rn;
    }
    auto one_minus_rk = one_rat().subtract(*rk);
    if (!one_minus_rk) {
        return one_minus_rk;
    }
    auto one_minus_rn = one_rat().subtract(*rn);
    if (!one_minus_rn) {
        return one_minus_rn;
    }
    auto ratio = one_minus_rk->divide(*one_minus_rn);  // (1 - r^k)/(1 - r^n)
    if (!ratio) {
        return ratio;
    }
    auto scaled = nr->multiply(*ratio);  // n (1 - r^k)/(1 - r^n)
    if (!scaled) {
        return scaled;
    }
    auto inner = kr->subtract(*scaled);  // k - n(...)
    if (!inner) {
        return inner;
    }
    return inner->divide(*qmp);  // qmp != 0 since p != 1/2
}

auto birth_death_stationary(std::span<const Rational> birth, std::span<const Rational> death)
    -> Result<std::vector<Rational>> {
    if (birth.size() != death.size()) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    for (const Rational& b : birth) {
        if (b.numerator() < 0) {  // canonical den > 0
            return make_error<std::vector<Rational>>(MathError::domain_error);
        }
    }
    for (const Rational& d : death) {
        if (d.numerator() < 0) {
            return make_error<std::vector<Rational>>(MathError::domain_error);
        }
    }
    const std::size_t l = birth.size();
    std::vector<Rational> pi(l + 1);
    pi[0] = one_rat();  // unnormalised pi_0 = 1
    for (std::size_t i = 1; i <= l; ++i) {
        auto ratio = birth[i - 1].divide(death[i - 1]);  // lambda_{i-1}/mu_i; mu_i = 0 => div0
        if (!ratio) {
            return make_error<std::vector<Rational>>(ratio.error());
        }
        auto next = pi[i - 1].multiply(*ratio);
        if (!next) {
            return make_error<std::vector<Rational>>(next.error());
        }
        pi[i] = *next;
    }
    // Normalise by the total (>= pi_0 = 1 > 0, since every term is non-negative).
    Rational total;  // 0/1
    for (const Rational& v : pi) {
        auto next = total.add(v);
        if (!next) {
            return make_error<std::vector<Rational>>(next.error());
        }
        total = *next;
    }
    for (Rational& v : pi) {
        auto norm = v.divide(total);
        if (!norm) {
            return make_error<std::vector<Rational>>(norm.error());
        }
        v = *norm;
    }
    return pi;
}

// --- WSS sample analysis ----------------------------------------------------

auto autocovariance_at(std::span<const Rational> x, std::size_t lag, bool biased)
    -> Result<Rational> {
    const std::size_t n = x.size();
    if (n == 0 || lag >= n) {
        return make_error<Rational>(MathError::domain_error);
    }
    auto xbar = mean(x);  // nimblecas.stats: domain_error only on empty (excluded above)
    if (!xbar) {
        return xbar;
    }
    Rational acc;  // 0/1
    for (std::size_t t = 0; t + lag < n; ++t) {
        auto dt = x[t].subtract(*xbar);
        if (!dt) {
            return dt;
        }
        auto dtl = x[t + lag].subtract(*xbar);
        if (!dtl) {
            return dtl;
        }
        auto prod = dt->multiply(*dtl);
        if (!prod) {
            return prod;
        }
        auto next = acc.add(*prod);
        if (!next) {
            return next;
        }
        acc = *next;
    }
    const std::int64_t denom = biased ? static_cast<std::int64_t>(n)
                                      : static_cast<std::int64_t>(n - lag);
    return acc.divide(Rational::from_int(denom));
}

auto autocovariance(std::span<const Rational> x, std::size_t max_lag, bool biased)
    -> Result<std::vector<Rational>> {
    std::vector<Rational> out(max_lag + 1);
    for (std::size_t tau = 0; tau <= max_lag; ++tau) {
        auto r = autocovariance_at(x, tau, biased);
        if (!r) {
            return make_error<std::vector<Rational>>(r.error());
        }
        out[tau] = *r;
    }
    return out;
}

auto autocorrelation(std::span<const Rational> x, std::size_t max_lag, bool biased)
    -> Result<std::vector<Rational>> {
    auto cov = autocovariance(x, max_lag, biased);
    if (!cov) {
        return cov;
    }
    const Rational r0 = (*cov)[0];
    if (r0.is_zero()) {
        return make_error<std::vector<Rational>>(MathError::division_by_zero);  // constant record
    }
    std::vector<Rational> out(max_lag + 1);
    for (std::size_t tau = 0; tau <= max_lag; ++tau) {
        auto rho = (*cov)[tau].divide(r0);
        if (!rho) {
            return make_error<std::vector<Rational>>(rho.error());
        }
        out[tau] = *rho;
    }
    return out;
}

auto cross_covariance_at(std::span<const Rational> x, std::span<const Rational> y,
                         std::int64_t lag, bool biased) -> Result<Rational> {
    const std::size_t n = x.size();
    if (n == 0 || y.size() != n) {
        return make_error<Rational>(MathError::domain_error);
    }
    // `lag` is raw user input (unlike the canonical-Rational-numerator callers of abs_i64), so
    // guard INT64_MIN before negating — its magnitude is unrepresentable and always >= n anyway.
    if (lag == std::numeric_limits<std::int64_t>::min()) {
        return make_error<Rational>(MathError::domain_error);
    }
    const std::size_t alag = static_cast<std::size_t>(abs_i64(lag));
    if (alag >= n) {
        return make_error<Rational>(MathError::domain_error);
    }
    auto xbar = mean(x);
    if (!xbar) {
        return xbar;
    }
    auto ybar = mean(y);
    if (!ybar) {
        return ybar;
    }
    Rational acc;  // 0/1
    // Positive lag: pair x[t] with y[t+lag]. Negative lag: pair x[t+|lag|] with y[t].
    for (std::size_t t = 0; t + alag < n; ++t) {
        const std::size_t xi = lag >= 0 ? t : t + alag;
        const std::size_t yi = lag >= 0 ? t + alag : t;
        auto dx = x[xi].subtract(*xbar);
        if (!dx) {
            return dx;
        }
        auto dy = y[yi].subtract(*ybar);
        if (!dy) {
            return dy;
        }
        auto prod = dx->multiply(*dy);
        if (!prod) {
            return prod;
        }
        auto next = acc.add(*prod);
        if (!next) {
            return next;
        }
        acc = *next;
    }
    const std::int64_t denom = biased ? static_cast<std::int64_t>(n)
                                      : static_cast<std::int64_t>(n - alag);
    return acc.divide(Rational::from_int(denom));
}

auto cross_covariance(std::span<const Rational> x, std::span<const Rational> y,
                      std::size_t max_lag, bool biased) -> Result<std::vector<Rational>> {
    std::vector<Rational> out(max_lag + 1);
    for (std::size_t tau = 0; tau <= max_lag; ++tau) {
        auto r = cross_covariance_at(x, y, static_cast<std::int64_t>(tau), biased);
        if (!r) {
            return make_error<std::vector<Rational>>(r.error());
        }
        out[tau] = *r;
    }
    return out;
}

auto is_wss(std::span<const Rational> x, std::size_t max_lag) -> Result<bool> {
    const std::size_t half = x.size() / 2;
    if (half == 0 || max_lag >= half) {
        return make_error<bool>(MathError::domain_error);  // record too short to diagnose
    }
    const std::span<const Rational> h1 = x.subspan(0, half);
    const std::span<const Rational> h2 = x.subspan(half, half);
    auto m1 = mean(h1);
    auto m2 = mean(h2);
    if (!m1 || !m2) {
        return make_error<bool>(m1 ? m2.error() : m1.error());
    }
    if (!(*m1 == *m2)) {
        return false;  // mean is not constant across the record
    }
    for (std::size_t tau = 0; tau <= max_lag; ++tau) {
        auto c1 = autocovariance_at(h1, tau, true);
        auto c2 = autocovariance_at(h2, tau, true);
        if (!c1 || !c2) {
            return make_error<bool>(c1 ? c2.error() : c1.error());
        }
        if (!(*c1 == *c2)) {
            return false;  // autocovariance is not a function of lag alone
        }
    }
    return true;
}

// --- AR / MA / ARMA ---------------------------------------------------------

auto yule_walker_autocorrelation(std::span<const Rational> ar, std::size_t max_lag)
    -> Result<std::vector<Rational>> {
    const std::size_t p = ar.size();
    std::vector<Rational> rho(max_lag + 1);
    rho[0] = one_rat();
    if (p == 0) {
        return rho;  // white noise: rho(k) = 0 for k >= 1 (already default-zero)
    }
    // Build the p x p Yule-Walker system for u = [rho(1), ..., rho(p)].
    // Equation k (1..p):  rho(k) - sum_j phi_j rho(|k-j|) = 0, with rho(0) = 1 moved to RHS.
    std::vector<std::vector<Rational>> a(p, std::vector<Rational>(p));
    std::vector<Rational> b(p);
    for (std::size_t k = 1; k <= p; ++k) {
        const std::size_t eq = k - 1;
        a[eq][k - 1] = one_rat();  // + rho(k) on the LHS
        for (std::size_t j = 1; j <= p; ++j) {
            const std::size_t idx = k >= j ? k - j : j - k;  // |k - j|
            const Rational& phij = ar[j - 1];
            if (idx == 0) {
                // -phi_j * rho(0) = -phi_j moves to RHS as +phi_j.
                auto nb = b[eq].add(phij);
                if (!nb) {
                    return make_error<std::vector<Rational>>(nb.error());
                }
                b[eq] = *nb;
            } else {
                auto na = a[eq][idx - 1].subtract(phij);
                if (!na) {
                    return make_error<std::vector<Rational>>(na.error());
                }
                a[eq][idx - 1] = *na;
            }
        }
    }
    auto amat = Matrix::from_rows(std::move(a));
    if (!amat) {
        return make_error<std::vector<Rational>>(amat.error());
    }
    std::vector<std::vector<Rational>> brows(p, std::vector<Rational>(1));
    for (std::size_t i = 0; i < p; ++i) {
        brows[i][0] = b[i];
    }
    auto bmat = Matrix::from_rows(std::move(brows));
    if (!bmat) {
        return make_error<std::vector<Rational>>(bmat.error());
    }
    auto sol = amat->solve(*bmat);  // singular => domain_error (non-stationary parameters)
    if (!sol) {
        return make_error<std::vector<Rational>>(sol.error());
    }
    for (std::size_t k = 1; k <= p && k <= max_lag; ++k) {
        rho[k] = sol->at(k - 1, 0);
    }
    // Continue rho(k) for k > p by the AR recursion rho(k) = sum_j phi_j rho(k-j).
    for (std::size_t k = p + 1; k <= max_lag; ++k) {
        Rational acc;  // 0/1
        for (std::size_t j = 1; j <= p; ++j) {
            auto term = ar[j - 1].multiply(rho[k - j]);
            if (!term) {
                return make_error<std::vector<Rational>>(term.error());
            }
            auto next = acc.add(*term);
            if (!next) {
                return make_error<std::vector<Rational>>(next.error());
            }
            acc = *next;
        }
        rho[k] = acc;
    }
    return rho;
}

auto ar_autocovariance(std::span<const Rational> ar, const Rational& sigma2, std::size_t max_lag)
    -> Result<std::vector<Rational>> {
    const std::size_t p = ar.size();
    // Need rho up to max(max_lag, p) to form the variance denominator.
    const std::size_t need = std::max(max_lag, p);
    auto rho = yule_walker_autocorrelation(ar, need);
    if (!rho) {
        return rho;
    }
    // gamma(0) = sigma^2 / (1 - sum_j phi_j rho(j)).
    Rational s;  // 0/1
    for (std::size_t j = 1; j <= p; ++j) {
        auto term = ar[j - 1].multiply((*rho)[j]);
        if (!term) {
            return make_error<std::vector<Rational>>(term.error());
        }
        auto next = s.add(*term);
        if (!next) {
            return make_error<std::vector<Rational>>(next.error());
        }
        s = *next;
    }
    auto denom = one_rat().subtract(s);
    if (!denom) {
        return make_error<std::vector<Rational>>(denom.error());
    }
    auto gamma0 = sigma2.divide(*denom);  // denom == 0 => division_by_zero
    if (!gamma0) {
        return make_error<std::vector<Rational>>(gamma0.error());
    }
    std::vector<Rational> gamma(max_lag + 1);
    for (std::size_t k = 0; k <= max_lag; ++k) {
        auto g = gamma0->multiply((*rho)[k]);
        if (!g) {
            return make_error<std::vector<Rational>>(g.error());
        }
        gamma[k] = *g;
    }
    return gamma;
}

auto ma_autocovariance(std::span<const Rational> ma, const Rational& sigma2, std::size_t max_lag)
    -> Result<std::vector<Rational>> {
    const std::size_t q = ma.size();
    // theta_0 = 1, theta_1..theta_q = ma.
    std::vector<Rational> theta(q + 1);
    theta[0] = one_rat();
    for (std::size_t k = 0; k < q; ++k) {
        theta[k + 1] = ma[k];
    }
    std::vector<Rational> gamma(max_lag + 1);  // default 0 (gamma(tau) = 0 for tau > q)
    for (std::size_t tau = 0; tau <= max_lag && tau <= q; ++tau) {
        Rational acc;  // 0/1
        for (std::size_t k = 0; k + tau <= q; ++k) {
            auto prod = theta[k].multiply(theta[k + tau]);
            if (!prod) {
                return make_error<std::vector<Rational>>(prod.error());
            }
            auto next = acc.add(*prod);
            if (!next) {
                return make_error<std::vector<Rational>>(next.error());
            }
            acc = *next;
        }
        auto g = acc.multiply(sigma2);
        if (!g) {
            return make_error<std::vector<Rational>>(g.error());
        }
        gamma[tau] = *g;
    }
    return gamma;
}

namespace {

// Decide a StabilityCertificate for a polynomial whose stability requires every root to
// lie STRICTLY outside the unit circle. Rational roots are enumerated exactly and tested
// (|num| <= den means on/inside the circle => unstable); if the rational roots (counted
// with multiplicity) account for the whole degree the verdict is definitive, otherwise
// unenumerated roots make it indeterminate.
[[nodiscard]] auto certify_outside_unit_circle(const RationalPoly& poly)
    -> Result<StabilityCertificate> {
    const std::int64_t deg = poly.degree();
    if (deg <= 0) {
        return StabilityCertificate::stable;  // constant: no roots to violate stability
    }
    auto roots = rational_roots(poly);
    if (!roots) {
        return make_error<StabilityCertificate>(roots.error());
    }
    std::int64_t total_mult = 0;
    bool found_unstable = false;
    for (const auto& [r, m] : *roots) {
        total_mult += m;
        // |r| = |num| / den (den > 0). On or inside the unit circle iff |num| <= den.
        if (abs_i64(r.numerator()) <= r.denominator()) {
            found_unstable = true;
        }
    }
    if (found_unstable) {
        return StabilityCertificate::unstable;
    }
    if (total_mult == deg) {
        return StabilityCertificate::stable;  // all roots rational and strictly outside
    }
    return StabilityCertificate::indeterminate;  // irrational/complex roots unenumerated
}

}  // namespace

auto ar_is_stationary(std::span<const Rational> ar) -> Result<StabilityCertificate> {
    // Phi(z) = 1 - sum_k phi_k z^k  (ascending coeffs: c0 = 1, c_k = -phi_k).
    std::vector<Rational> c(ar.size() + 1);
    c[0] = one_rat();
    for (std::size_t k = 0; k < ar.size(); ++k) {
        auto neg = ar[k].negate();
        if (!neg) {
            return make_error<StabilityCertificate>(neg.error());
        }
        c[k + 1] = *neg;
    }
    return certify_outside_unit_circle(RationalPoly::from_coeffs(std::move(c)));
}

auto ma_is_invertible(std::span<const Rational> ma) -> Result<StabilityCertificate> {
    // Theta(z) = 1 + sum_k theta_k z^k.
    std::vector<Rational> c(ma.size() + 1);
    c[0] = one_rat();
    for (std::size_t k = 0; k < ma.size(); ++k) {
        c[k + 1] = ma[k];
    }
    return certify_outside_unit_circle(RationalPoly::from_coeffs(std::move(c)));
}

// --- ARIMA differencing & partial autocorrelation ---------------------------

auto difference(std::span<const Rational> x, std::size_t d) -> Result<std::vector<Rational>> {
    std::vector<Rational> cur(x.begin(), x.end());
    for (std::size_t step = 0; step < d; ++step) {
        if (cur.empty()) {
            return make_error<std::vector<Rational>>(MathError::domain_error);  // d > N
        }
        std::vector<Rational> next(cur.size() - 1);
        for (std::size_t t = 1; t < cur.size(); ++t) {
            auto diff = cur[t].subtract(cur[t - 1]);
            if (!diff) {
                return make_error<std::vector<Rational>>(diff.error());
            }
            next[t - 1] = *diff;
        }
        cur = std::move(next);
    }
    return cur;
}

auto integrate(std::span<const Rational> y, std::span<const Rational> initial)
    -> Result<std::vector<Rational>> {
    std::vector<Rational> cur(y.begin(), y.end());
    // Integrate from the innermost difference outward: level d-1 (constant initial[d-1])
    // reconstructs nabla^{d-1} x from nabla^d x, ..., level 0 (initial[0]) reconstructs x.
    const std::size_t d = initial.size();
    for (std::size_t lvl = d; lvl-- > 0;) {
        std::vector<Rational> next(cur.size() + 1);
        next[0] = initial[lvl];  // constant of integration = dropped leading value
        for (std::size_t t = 0; t < cur.size(); ++t) {
            auto sum = next[t].add(cur[t]);  // running cumulative sum
            if (!sum) {
                return make_error<std::vector<Rational>>(sum.error());
            }
            next[t + 1] = *sum;
        }
        cur = std::move(next);
    }
    return cur;
}

auto partial_autocorrelation(std::span<const Rational> autocov)
    -> Result<std::vector<Rational>> {
    if (autocov.empty()) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    const std::size_t m = autocov.size() - 1;  // number of lags 1..m
    std::vector<Rational> pacf(m);
    if (m == 0) {
        return pacf;  // only gamma(0): no partial autocorrelations to report
    }
    // Durbin-Levinson recursion. v is the prediction-error variance v_{k-1}; phi holds the
    // order-(k-1) coefficients phi_{k-1,1..k-1}.
    Rational v = autocov[0];  // v_0 = gamma(0)
    if (v.is_zero()) {
        return make_error<std::vector<Rational>>(MathError::division_by_zero);  // constant series
    }
    auto phi11 = autocov[1].divide(v);  // phi_{1,1} = gamma(1)/gamma(0)
    if (!phi11) {
        return make_error<std::vector<Rational>>(phi11.error());
    }
    pacf[0] = *phi11;
    std::vector<Rational> phi{*phi11};
    for (std::size_t k = 2; k <= m; ++k) {
        // v_{k-1} = v_{k-2} (1 - phi_{k-1,k-1}^2).
        auto sq = phi.back().multiply(phi.back());
        if (!sq) {
            return make_error<std::vector<Rational>>(sq.error());
        }
        auto factor = one_rat().subtract(*sq);
        if (!factor) {
            return make_error<std::vector<Rational>>(factor.error());
        }
        auto vnext = v.multiply(*factor);
        if (!vnext) {
            return make_error<std::vector<Rational>>(vnext.error());
        }
        v = *vnext;
        if (v.is_zero()) {
            return make_error<std::vector<Rational>>(MathError::division_by_zero);  // deterministic
        }
        // numerator = gamma(k) - sum_{j=1}^{k-1} phi_{k-1,j} gamma(k-j).
        Rational acc;  // 0/1
        for (std::size_t j = 1; j < k; ++j) {
            auto term = phi[j - 1].multiply(autocov[k - j]);
            if (!term) {
                return make_error<std::vector<Rational>>(term.error());
            }
            auto next = acc.add(*term);
            if (!next) {
                return make_error<std::vector<Rational>>(next.error());
            }
            acc = *next;
        }
        auto num = autocov[k].subtract(acc);
        if (!num) {
            return make_error<std::vector<Rational>>(num.error());
        }
        auto phikk = num->divide(v);  // v = v_{k-1} != 0 (checked above)
        if (!phikk) {
            return make_error<std::vector<Rational>>(phikk.error());
        }
        // phi_{k,j} = phi_{k-1,j} - phi_{k,k} phi_{k-1,k-j}, j = 1..k-1; phi_{k,k} last.
        std::vector<Rational> newphi(k);
        for (std::size_t j = 1; j < k; ++j) {
            auto cross = phikk->multiply(phi[k - 1 - j]);
            if (!cross) {
                return make_error<std::vector<Rational>>(cross.error());
            }
            auto upd = phi[j - 1].subtract(*cross);
            if (!upd) {
                return make_error<std::vector<Rational>>(upd.error());
            }
            newphi[j - 1] = *upd;
        }
        newphi[k - 1] = *phikk;
        pacf[k - 1] = *phikk;
        phi = std::move(newphi);
    }
    return pacf;
}

// --- power spectral density (numerical) -------------------------------------

auto to_doubles(std::span<const Rational> xs) -> std::vector<double> {
    std::vector<double> out(xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i) {
        out[i] = rat_to_double(xs[i]);
    }
    return out;
}

auto power_spectral_density(std::span<const double> autocov_onesided, std::size_t num_points)
    -> Result<Spectrum> {
    if (autocov_onesided.empty() || num_points < 2) {
        return make_error<Spectrum>(MathError::domain_error);
    }
    Spectrum spec;
    spec.frequencies.resize(num_points);
    spec.values.resize(num_points);
    const double two_pi = 2.0 * std::numbers::pi;
    for (std::size_t k = 0; k < num_points; ++k) {
        const double f = 0.5 * static_cast<double>(k) / static_cast<double>(num_points - 1);
        double s = autocov_onesided[0];  // R(0)
        for (std::size_t tau = 1; tau < autocov_onesided.size(); ++tau) {
            s += 2.0 * autocov_onesided[tau] * std::cos(two_pi * f * static_cast<double>(tau));
        }
        spec.frequencies[k] = f;
        spec.values[k] = s;
    }
    return spec;
}

auto power_spectral_density(std::span<const Rational> autocov_onesided, std::size_t num_points)
    -> Result<Spectrum> {
    const std::vector<double> r = to_doubles(autocov_onesided);
    return power_spectral_density(std::span<const double>{r}, num_points);
}

auto arma_spectral_density(std::span<const Rational> ar, std::span<const Rational> ma,
                           double sigma2, std::size_t num_points) -> Result<Spectrum> {
    if (num_points < 2) {
        return make_error<Spectrum>(MathError::domain_error);
    }
    const std::vector<double> phi = to_doubles(ar);
    const std::vector<double> theta = to_doubles(ma);
    Spectrum spec;
    spec.frequencies.resize(num_points);
    spec.values.resize(num_points);
    const double two_pi = 2.0 * std::numbers::pi;
    using cd = std::complex<double>;
    for (std::size_t k = 0; k < num_points; ++k) {
        const double f = 0.5 * static_cast<double>(k) / static_cast<double>(num_points - 1);
        const double w = two_pi * f;
        const cd x = std::exp(cd{0.0, -w});  // e^{-i w}
        // Phi(x) = 1 - sum phi_k x^k ; Theta(x) = 1 + sum theta_k x^k.
        cd phi_val{1.0, 0.0};
        cd xpow{1.0, 0.0};
        for (double c : phi) {
            xpow *= x;
            phi_val -= c * xpow;
        }
        cd theta_val{1.0, 0.0};
        xpow = cd{1.0, 0.0};
        for (double c : theta) {
            xpow *= x;
            theta_val += c * xpow;
        }
        const double den = std::norm(phi_val);  // |Phi|^2
        if (den == 0.0) {
            return make_error<Spectrum>(MathError::not_implemented);  // AR root on unit circle
        }
        spec.frequencies[k] = f;
        spec.values[k] = sigma2 * std::norm(theta_val) / den;
    }
    return spec;
}

}  // namespace nimblecas
