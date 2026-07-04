// NimbleCAS discrete hidden Markov models, exact over Q (ROADMAP §7.8 slice).
// @author Olumuyiwa Oluwasanmi
//
// This module is the DISCRETE-EMISSION hidden Markov model (HMM) core. A model has N
// hidden states and M observation symbols and is specified by three EXACT rational
// objects: an initial state distribution pi (length N), a row-stochastic transition
// matrix A (N x N, A(i, j) = P(state j at t+1 | state i at t)), and a row-stochastic
// emission matrix B (N x M, B(i, k) = P(symbol k | state i)). Validity -- correct
// dimensions and rows summing EXACTLY to 1 with non-negative entries -- is checked over Q
// (no tolerance). An observation sequence is a list of symbol indices in [0, M).
//
// It complements nimblecas.stochastic (the *structure* of a stationary Markov chain: its
// stationary law, spectrum, hitting times). Here the chain is HIDDEN: we never see the
// state, only a symbol emitted from it, and the deliverables are the classical HMM
// inference algorithms.
//
//   FORWARD.   alpha_t(i) = P(o_0..o_t, X_t = i) by the sum-product recursion, and the
//              observation likelihood P(O | model) = sum_i alpha_{T-1}(i). EXACT over Q
//              (a sum of products of rationals). For long sequences the exact numerators
//              underflow nothing -- there is no rounding -- but the int64 magnitudes can
//              overflow; forward_scaled() is the NUMERICAL underflow-safe alternative,
//              carrying double alpha-hat rescaled to sum 1 each step and returning
//              log P(O) as the sum of the log scaling constants.
//   BACKWARD.  beta_t(i) = P(o_{t+1}..o_{T-1} | X_t = i). EXACT over Q. The two agree:
//              P(O) = sum_i alpha_t(i) beta_t(i) for every t.
//   POSTERIORS (smoothing). gamma_t(i) = P(X_t = i | O) = alpha_t(i) beta_t(i) / P(O) and
//              xi_t(i, j) = P(X_t = i, X_{t+1} = j | O). EXACT rational ratios. P(O) = 0
//              (an impossible observation under the model) is guarded as domain_error
//              rather than divided by zero.
//   VITERBI.   The single most likely state path by max-product dynamic programming with
//              backpointers. EXACT over Q -- the argmax is an exact rational comparison,
//              so NO floating point is needed at all -- returning the path and its exact
//              rational probability. Ties are broken deterministically toward the SMALLEST
//              state index.
//   BAUM-WELCH (EM). Re-estimate pi, A, B from one or more observation sequences by the
//              expected-count ratios (Rabiner's formulae). Each M-step is EXACT over Q:
//              the re-estimated rows are exact rational ratios that sum to 1 by
//              construction, so the new model is exactly stochastic. HONESTY: EM only
//              converges to a LOCAL optimum and the result depends on the initial model;
//              each individual step is exact, but iterating and *stopping* is a numerical
//              decision. Here we iterate a fixed number of steps and report the exact
//              per-iteration likelihood, which is provably non-decreasing.
//
// HONESTY BOUNDARY (documented and true). EXACT over Q: model validity, forward alpha and
// P(O), backward beta, the smoothing gamma/xi posteriors, Viterbi (path AND probability --
// no floating point), one Baum-Welch M-step, and path_probability. NUMERICAL (double):
// forward_scaled / log_likelihood, the underflow-safe log-domain forward for long
// sequences. LOCAL-OPTIMUM / THRESHOLD (documented): Baum-Welch iteration converges only
// to a local optimum and its termination (iteration count) is a numerical choice.
// OUT OF SCOPE: continuous-emission HMMs (e.g. Gaussian mixtures) are not modelled here --
// this is the discrete-emission exact core. P(O) = 0 is a domain_error, never a divide.
//
// Conforms to config/cpp_details.txt: C++23 modules, `import std`, trailing return types,
// no owning raw pointers, std::expected railway error handling via make_error (no
// exceptions), [[nodiscard]] throughout, exact overflow-checked Rational arithmetic.

export module nimblecas.hmm;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// HiddenMarkovModel — N hidden states, M observation symbols, all exact over Q.
// ---------------------------------------------------------------------------
// Constructed only through make(), which performs the validity check (dimensions +
// row-stochasticity over Q). The stored pi/A/B are therefore always a valid model.
class HiddenMarkovModel {
public:
    // Build and VALIDATE a model. `initial` is pi (length N >= 1), `transition` is A
    // (N x N), `emission` is B (N x M, M >= 1). Fails with domain_error on a dimension
    // mismatch, a non-probability pi (an entry < 0 or a sum != 1), or a non-row-stochastic
    // A or B. Propagates overflow from the exact row sums.
    [[nodiscard]] static auto make(std::vector<Rational> initial, Matrix transition,
                                   Matrix emission) -> Result<HiddenMarkovModel>;

    [[nodiscard]] auto num_states() const noexcept -> std::size_t { return pi_.size(); }
    [[nodiscard]] auto num_symbols() const noexcept -> std::size_t { return b_.cols(); }
    [[nodiscard]] auto initial() const noexcept -> std::span<const Rational> { return pi_; }
    [[nodiscard]] auto transition() const noexcept -> const Matrix& { return a_; }
    [[nodiscard]] auto emission() const noexcept -> const Matrix& { return b_; }

private:
    HiddenMarkovModel(std::vector<Rational> pi, Matrix a, Matrix b)
        : pi_(std::move(pi)), a_(std::move(a)), b_(std::move(b)) {}

    std::vector<Rational> pi_;  // length N
    Matrix a_;                  // N x N, row-stochastic
    Matrix b_;                  // N x M, row-stochastic
};

// The forward table (EXACT). `alpha[t][i]` = P(o_0..o_t, X_t = i); `likelihood` = P(O).
struct ForwardTable {
    std::vector<std::vector<Rational>> alpha;  // T rows, N columns
    Rational likelihood;                       // sum_i alpha[T-1][i]
};

// The smoothing posteriors (EXACT). `gamma[t][i]` = P(X_t = i | O) (T x N); `xi[t][i][j]`
// = P(X_t = i, X_{t+1} = j | O) for t = 0..T-2 ((T-1) x N x N).
struct Posteriors {
    std::vector<std::vector<Rational>> gamma;
    std::vector<std::vector<std::vector<Rational>>> xi;
};

// The Viterbi decoding (EXACT). `states` is the most likely hidden path (length T) and
// `probability` its exact joint probability P(path, O). Ties favour the smallest index.
struct ViterbiPath {
    std::vector<std::size_t> states;
    Rational probability;
};

// The numerical, underflow-safe log-domain forward. `alpha_hat[t]` is the forward vector
// at time t rescaled to sum to 1; `log_likelihood` = log P(O) = sum_t log(scale_t).
struct ScaledForward {
    std::vector<std::vector<double>> alpha_hat;  // T rows, N columns (each row sums to 1)
    double log_likelihood{};
};

// ---------------------------------------------------------------------------
// Inference (EXACT over Q unless noted).
// ---------------------------------------------------------------------------

// The forward algorithm: the full alpha table and the observation likelihood P(O), EXACT
// over Q. Fails with domain_error on an empty sequence or a symbol index >= M; propagates
// overflow from the exact sum-product.
[[nodiscard]] auto forward(const HiddenMarkovModel& model, std::span<const std::size_t> obs)
    -> Result<ForwardTable>;

// The observation likelihood P(O | model) = sum_i alpha_{T-1}(i), EXACT over Q (a
// convenience wrapper over forward()).
[[nodiscard]] auto observation_likelihood(const HiddenMarkovModel& model,
                                          std::span<const std::size_t> obs) -> Result<Rational>;

// The backward algorithm: beta_t(i) = P(o_{t+1}..o_{T-1} | X_t = i), with beta_{T-1} = 1,
// EXACT over Q. Same domain/overflow contract as forward().
[[nodiscard]] auto backward(const HiddenMarkovModel& model, std::span<const std::size_t> obs)
    -> Result<std::vector<std::vector<Rational>>>;

// The smoothing posteriors gamma and xi, EXACT over Q. Fails with domain_error when P(O)
// = 0 (an impossible observation -- the ratios would divide by zero), and otherwise as
// forward()/backward().
[[nodiscard]] auto posteriors(const HiddenMarkovModel& model, std::span<const std::size_t> obs)
    -> Result<Posteriors>;

// Viterbi decoding by exact max-product DP with backpointer traceback: the single most
// likely hidden state path and its EXACT rational probability. No floating point is used;
// the argmax is an exact rational comparison and ties break toward the smallest state
// index. Same domain/overflow contract as forward().
[[nodiscard]] auto viterbi(const HiddenMarkovModel& model, std::span<const std::size_t> obs)
    -> Result<ViterbiPath>;

// The exact joint probability P(states, O) of a specific state path and observation
// sequence: pi_{s0} B(s0, o0) prod_t A(s_{t-1}, s_t) B(s_t, o_t). Fails with domain_error
// on length mismatch, an empty sequence, or an out-of-range state / symbol index.
[[nodiscard]] auto path_probability(const HiddenMarkovModel& model,
                                    std::span<const std::size_t> states,
                                    std::span<const std::size_t> obs) -> Result<Rational>;

// The re-estimated model and per-iteration likelihoods from Baum-Welch (EM).
struct BaumWelchResult {
    HiddenMarkovModel model;
    std::vector<Rational> likelihoods;  // size max_iterations + 1, non-decreasing (EXACT)
};

// Baum-Welch (EM) parameter re-estimation from one or more observation sequences. Runs
// `max_iterations` EXACT M-steps; each re-estimates pi/A/B by expected-count ratios that
// sum to 1 by construction, so every intermediate model is exactly row-stochastic.
// `likelihoods[k]` is the EXACT joint likelihood (product over the independent sequences)
// of the model at the start of iteration k, with a final entry after the last update, so
// the sequence has max_iterations + 1 entries and is provably non-decreasing. HONESTY: EM
// converges only to a LOCAL optimum and the result depends on the initial model; the fixed
// iteration count is a numerical/threshold decision even though each step is exact. Fails
// with domain_error on an empty sequence set or an impossible observation (P(O) = 0), and
// with division_by_zero when a state accrues zero expected visits (its row is
// unidentifiable from the data). Propagates overflow.
[[nodiscard]] auto baum_welch(const HiddenMarkovModel& model,
                              std::span<const std::vector<std::size_t>> sequences,
                              std::size_t max_iterations) -> Result<BaumWelchResult>;

// The numerical, underflow-safe log-domain forward (double). Rescales the forward vector
// to sum to 1 at each step and returns the rescaled table together with log P(O). This is
// the NUMERICAL alternative to the exact forward() for long sequences. Fails with
// domain_error on an empty sequence, a symbol index >= M, or an impossible observation
// (a zero scaling constant, i.e. P(O) = 0).
[[nodiscard]] auto forward_scaled(const HiddenMarkovModel& model,
                                  std::span<const std::size_t> obs) -> Result<ScaledForward>;

// The log-likelihood log P(O | model), computed by the underflow-safe scaled forward
// (NUMERICAL, double). Same failure contract as forward_scaled().
[[nodiscard]] auto log_likelihood(const HiddenMarkovModel& model,
                                  std::span<const std::size_t> obs) -> Result<double>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

[[nodiscard]] auto one_rat() -> Rational { return Rational::from_int(1); }

[[nodiscard]] auto rat_to_double(const Rational& r) -> double {
    return static_cast<double>(r.numerator()) / static_cast<double>(r.denominator());
}

// acc + a*b, exact and overflow-checked.
[[nodiscard]] auto add_prod(const Rational& acc, const Rational& a, const Rational& b)
    -> Result<Rational> {
    auto prod = a.multiply(b);
    if (!prod) {
        return prod;
    }
    return acc.add(*prod);
}

// Strict order over Q: true iff a > b. The canonical form (den > 0) makes the sign of
// (a - b) exactly the sign of its numerator. Propagates overflow from the difference.
[[nodiscard]] auto rat_greater(const Rational& a, const Rational& b) -> Result<bool> {
    auto diff = a.subtract(b);
    if (!diff) {
        return make_error<bool>(diff.error());
    }
    return diff->numerator() > 0;
}

// True iff every entry is >= 0 and every row sums EXACTLY to 1 (a row-stochastic matrix
// over Q). Unlike the square stochastic check, an N x M emission matrix is allowed.
// Propagates overflow from the exact row sums.
[[nodiscard]] auto is_row_stochastic(const Matrix& m) -> Result<bool> {
    if (m.rows() == 0 || m.cols() == 0) {
        return false;
    }
    const Rational one = one_rat();
    for (std::size_t i = 0; i < m.rows(); ++i) {
        Rational acc;  // 0/1
        for (std::size_t j = 0; j < m.cols(); ++j) {
            if (m.at(i, j).numerator() < 0) {  // canonical: den > 0, sign is the numerator's
                return false;
            }
            auto next = acc.add(m.at(i, j));
            if (!next) {
                return make_error<bool>(next.error());
            }
            acc = *next;
        }
        if (!(acc == one)) {
            return false;
        }
    }
    return true;
}

// True iff v is a probability vector over Q: non-empty, every entry >= 0, sum EXACTLY 1.
[[nodiscard]] auto is_prob_vector(std::span<const Rational> v) -> Result<bool> {
    if (v.empty()) {
        return false;
    }
    Rational acc;  // 0/1
    for (const Rational& e : v) {
        if (e.numerator() < 0) {
            return false;
        }
        auto next = acc.add(e);
        if (!next) {
            return make_error<bool>(next.error());
        }
        acc = *next;
    }
    return acc == one_rat();
}

// Validate an observation sequence against a model: non-empty and every symbol < M.
[[nodiscard]] auto check_obs(const HiddenMarkovModel& model, std::span<const std::size_t> obs)
    -> Result<void> {
    if (obs.empty()) {
        return make_error<void>(MathError::domain_error);
    }
    const std::size_t m = model.num_symbols();
    for (const std::size_t o : obs) {
        if (o >= m) {
            return make_error<void>(MathError::domain_error);
        }
    }
    return {};
}

}  // namespace

// --- construction -----------------------------------------------------------

auto HiddenMarkovModel::make(std::vector<Rational> initial, Matrix transition, Matrix emission)
    -> Result<HiddenMarkovModel> {
    const std::size_t n = initial.size();
    if (n == 0) {
        return make_error<HiddenMarkovModel>(MathError::domain_error);
    }
    // A must be square N x N; B must be N x M with M >= 1.
    if (!transition.is_square() || transition.rows() != n) {
        return make_error<HiddenMarkovModel>(MathError::domain_error);
    }
    if (emission.rows() != n || emission.cols() == 0) {
        return make_error<HiddenMarkovModel>(MathError::domain_error);
    }
    auto pv = is_prob_vector(initial);
    if (!pv) {
        return make_error<HiddenMarkovModel>(pv.error());
    }
    if (!*pv) {
        return make_error<HiddenMarkovModel>(MathError::domain_error);
    }
    auto a_stoch = is_row_stochastic(transition);
    if (!a_stoch) {
        return make_error<HiddenMarkovModel>(a_stoch.error());
    }
    if (!*a_stoch) {
        return make_error<HiddenMarkovModel>(MathError::domain_error);
    }
    auto b_stoch = is_row_stochastic(emission);
    if (!b_stoch) {
        return make_error<HiddenMarkovModel>(b_stoch.error());
    }
    if (!*b_stoch) {
        return make_error<HiddenMarkovModel>(MathError::domain_error);
    }
    return HiddenMarkovModel{std::move(initial), std::move(transition), std::move(emission)};
}

// --- forward ----------------------------------------------------------------

auto forward(const HiddenMarkovModel& model, std::span<const std::size_t> obs)
    -> Result<ForwardTable> {
    auto chk = check_obs(model, obs);
    if (!chk) {
        return make_error<ForwardTable>(chk.error());
    }
    const std::size_t n = model.num_states();
    const std::size_t tlen = obs.size();
    const Matrix& a = model.transition();
    const Matrix& b = model.emission();
    const std::span<const Rational> pi = model.initial();

    std::vector<std::vector<Rational>> alpha(tlen, std::vector<Rational>(n));
    // t = 0: alpha_0(i) = pi_i * B(i, o_0).
    for (std::size_t i = 0; i < n; ++i) {
        auto v = pi[i].multiply(b.at(i, obs[0]));
        if (!v) {
            return make_error<ForwardTable>(v.error());
        }
        alpha[0][i] = *v;
    }
    // t >= 1: alpha_t(j) = (sum_i alpha_{t-1}(i) A(i, j)) * B(j, o_t).
    for (std::size_t t = 1; t < tlen; ++t) {
        for (std::size_t j = 0; j < n; ++j) {
            Rational acc;  // 0/1
            for (std::size_t i = 0; i < n; ++i) {
                auto next = add_prod(acc, alpha[t - 1][i], a.at(i, j));
                if (!next) {
                    return make_error<ForwardTable>(next.error());
                }
                acc = *next;
            }
            auto v = acc.multiply(b.at(j, obs[t]));
            if (!v) {
                return make_error<ForwardTable>(v.error());
            }
            alpha[t][j] = *v;
        }
    }
    // P(O) = sum_i alpha_{T-1}(i).
    Rational lik;  // 0/1
    for (std::size_t i = 0; i < n; ++i) {
        auto next = lik.add(alpha[tlen - 1][i]);
        if (!next) {
            return make_error<ForwardTable>(next.error());
        }
        lik = *next;
    }
    return ForwardTable{.alpha = std::move(alpha), .likelihood = lik};
}

auto observation_likelihood(const HiddenMarkovModel& model, std::span<const std::size_t> obs)
    -> Result<Rational> {
    auto fwd = forward(model, obs);
    if (!fwd) {
        return make_error<Rational>(fwd.error());
    }
    return fwd->likelihood;
}

// --- backward ---------------------------------------------------------------

auto backward(const HiddenMarkovModel& model, std::span<const std::size_t> obs)
    -> Result<std::vector<std::vector<Rational>>> {
    auto chk = check_obs(model, obs);
    if (!chk) {
        return make_error<std::vector<std::vector<Rational>>>(chk.error());
    }
    const std::size_t n = model.num_states();
    const std::size_t tlen = obs.size();
    const Matrix& a = model.transition();
    const Matrix& b = model.emission();

    std::vector<std::vector<Rational>> beta(tlen, std::vector<Rational>(n));
    const Rational one = one_rat();
    for (std::size_t i = 0; i < n; ++i) {
        beta[tlen - 1][i] = one;  // beta_{T-1}(i) = 1
    }
    // t = T-2 .. 0: beta_t(i) = sum_j A(i, j) B(j, o_{t+1}) beta_{t+1}(j).
    for (std::size_t t = tlen - 1; t-- > 0;) {
        for (std::size_t i = 0; i < n; ++i) {
            Rational acc;  // 0/1
            for (std::size_t j = 0; j < n; ++j) {
                auto ab = a.at(i, j).multiply(b.at(j, obs[t + 1]));
                if (!ab) {
                    return make_error<std::vector<std::vector<Rational>>>(ab.error());
                }
                auto abc = ab->multiply(beta[t + 1][j]);
                if (!abc) {
                    return make_error<std::vector<std::vector<Rational>>>(abc.error());
                }
                auto next = acc.add(*abc);
                if (!next) {
                    return make_error<std::vector<std::vector<Rational>>>(next.error());
                }
                acc = *next;
            }
            beta[t][i] = acc;
        }
    }
    return beta;
}

// --- posteriors (smoothing) -------------------------------------------------

auto posteriors(const HiddenMarkovModel& model, std::span<const std::size_t> obs)
    -> Result<Posteriors> {
    auto fwd = forward(model, obs);
    if (!fwd) {
        return make_error<Posteriors>(fwd.error());
    }
    auto bwd = backward(model, obs);
    if (!bwd) {
        return make_error<Posteriors>(bwd.error());
    }
    const Rational& prob = fwd->likelihood;
    if (prob.is_zero()) {
        return make_error<Posteriors>(MathError::domain_error);  // impossible observation
    }
    const std::size_t n = model.num_states();
    const std::size_t tlen = obs.size();
    const Matrix& a = model.transition();
    const Matrix& b = model.emission();
    const std::vector<std::vector<Rational>>& alpha = fwd->alpha;
    const std::vector<std::vector<Rational>>& beta = *bwd;

    // gamma_t(i) = alpha_t(i) beta_t(i) / P(O).
    std::vector<std::vector<Rational>> gamma(tlen, std::vector<Rational>(n));
    for (std::size_t t = 0; t < tlen; ++t) {
        for (std::size_t i = 0; i < n; ++i) {
            auto ab = alpha[t][i].multiply(beta[t][i]);
            if (!ab) {
                return make_error<Posteriors>(ab.error());
            }
            auto g = ab->divide(prob);  // prob != 0 (guarded above)
            if (!g) {
                return make_error<Posteriors>(g.error());
            }
            gamma[t][i] = *g;
        }
    }
    // xi_t(i, j) = alpha_t(i) A(i, j) B(j, o_{t+1}) beta_{t+1}(j) / P(O), t = 0..T-2.
    std::vector<std::vector<std::vector<Rational>>> xi;
    if (tlen >= 2) {
        xi.assign(tlen - 1, std::vector<std::vector<Rational>>(n, std::vector<Rational>(n)));
        for (std::size_t t = 0; t + 1 < tlen; ++t) {
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    auto v = alpha[t][i].multiply(a.at(i, j));
                    if (!v) {
                        return make_error<Posteriors>(v.error());
                    }
                    auto v2 = v->multiply(b.at(j, obs[t + 1]));
                    if (!v2) {
                        return make_error<Posteriors>(v2.error());
                    }
                    auto v3 = v2->multiply(beta[t + 1][j]);
                    if (!v3) {
                        return make_error<Posteriors>(v3.error());
                    }
                    auto v4 = v3->divide(prob);
                    if (!v4) {
                        return make_error<Posteriors>(v4.error());
                    }
                    xi[t][i][j] = *v4;
                }
            }
        }
    }
    return Posteriors{.gamma = std::move(gamma), .xi = std::move(xi)};
}

// --- Viterbi ----------------------------------------------------------------

auto viterbi(const HiddenMarkovModel& model, std::span<const std::size_t> obs)
    -> Result<ViterbiPath> {
    auto chk = check_obs(model, obs);
    if (!chk) {
        return make_error<ViterbiPath>(chk.error());
    }
    const std::size_t n = model.num_states();
    const std::size_t tlen = obs.size();
    const Matrix& a = model.transition();
    const Matrix& b = model.emission();
    const std::span<const Rational> pi = model.initial();

    // delta holds the current-time best path scores; psi[t][j] the backpointer.
    std::vector<Rational> delta(n);
    std::vector<std::vector<std::size_t>> psi(tlen, std::vector<std::size_t>(n, 0));
    for (std::size_t i = 0; i < n; ++i) {
        auto v = pi[i].multiply(b.at(i, obs[0]));
        if (!v) {
            return make_error<ViterbiPath>(v.error());
        }
        delta[i] = *v;
    }
    for (std::size_t t = 1; t < tlen; ++t) {
        std::vector<Rational> ndelta(n);
        for (std::size_t j = 0; j < n; ++j) {
            // best_i = argmax_i delta_{t-1}(i) A(i, j); ties -> smallest i.
            Rational best;
            std::size_t best_i = 0;
            for (std::size_t i = 0; i < n; ++i) {
                auto cand = delta[i].multiply(a.at(i, j));
                if (!cand) {
                    return make_error<ViterbiPath>(cand.error());
                }
                if (i == 0) {
                    best = *cand;
                    best_i = 0;
                } else {
                    auto gt = rat_greater(*cand, best);  // strict: keeps the first (smallest i)
                    if (!gt) {
                        return make_error<ViterbiPath>(gt.error());
                    }
                    if (*gt) {
                        best = *cand;
                        best_i = i;
                    }
                }
            }
            psi[t][j] = best_i;
            auto v = best.multiply(b.at(j, obs[t]));  // delta_t(j) = best * B(j, o_t)
            if (!v) {
                return make_error<ViterbiPath>(v.error());
            }
            ndelta[j] = *v;
        }
        delta = std::move(ndelta);
    }
    // Termination: best final state (ties -> smallest index) and its exact probability.
    std::size_t last = 0;
    Rational best_prob = delta[0];
    for (std::size_t i = 1; i < n; ++i) {
        auto gt = rat_greater(delta[i], best_prob);
        if (!gt) {
            return make_error<ViterbiPath>(gt.error());
        }
        if (*gt) {
            best_prob = delta[i];
            last = i;
        }
    }
    // Backpointer traceback.
    std::vector<std::size_t> path(tlen);
    path[tlen - 1] = last;
    for (std::size_t t = tlen - 1; t > 0; --t) {
        path[t - 1] = psi[t][path[t]];
    }
    return ViterbiPath{.states = std::move(path), .probability = best_prob};
}

auto path_probability(const HiddenMarkovModel& model, std::span<const std::size_t> states,
                      std::span<const std::size_t> obs) -> Result<Rational> {
    if (states.empty() || states.size() != obs.size()) {
        return make_error<Rational>(MathError::domain_error);
    }
    const std::size_t n = model.num_states();
    const std::size_t m = model.num_symbols();
    for (const std::size_t s : states) {
        if (s >= n) {
            return make_error<Rational>(MathError::domain_error);
        }
    }
    for (const std::size_t o : obs) {
        if (o >= m) {
            return make_error<Rational>(MathError::domain_error);
        }
    }
    const Matrix& a = model.transition();
    const Matrix& b = model.emission();
    const std::span<const Rational> pi = model.initial();

    // pi_{s0} * B(s0, o0).
    auto acc = pi[states[0]].multiply(b.at(states[0], obs[0]));
    if (!acc) {
        return acc;
    }
    Rational prob = *acc;
    for (std::size_t t = 1; t < states.size(); ++t) {
        auto step = a.at(states[t - 1], states[t]).multiply(b.at(states[t], obs[t]));
        if (!step) {
            return step;
        }
        auto next = prob.multiply(*step);
        if (!next) {
            return next;
        }
        prob = *next;
    }
    return prob;
}

// --- Baum-Welch (EM) --------------------------------------------------------

namespace {

// One EXACT M-step: re-estimate pi/A/B from expected counts accumulated across sequences.
[[nodiscard]] auto baum_welch_step(const HiddenMarkovModel& model,
                                   std::span<const std::vector<std::size_t>> sequences)
    -> Result<HiddenMarkovModel> {
    const std::size_t n = model.num_states();
    const std::size_t m = model.num_symbols();

    std::vector<Rational> pi_acc(n);                                  // sum_r gamma^r_0(i)
    std::vector<std::vector<Rational>> a_num(n, std::vector<Rational>(n));  // sum xi
    std::vector<Rational> a_den(n);                                   // sum_{t<T-1} gamma
    std::vector<std::vector<Rational>> b_num(n, std::vector<Rational>(m));  // sum gamma[o=k]
    std::vector<Rational> b_den(n);                                   // sum_{t} gamma

    for (const std::vector<std::size_t>& seq : sequences) {
        auto post = posteriors(model, std::span<const std::size_t>{seq});
        if (!post) {
            return make_error<HiddenMarkovModel>(post.error());
        }
        const std::vector<std::vector<Rational>>& gamma = post->gamma;
        const std::vector<std::vector<std::vector<Rational>>>& xi = post->xi;
        const std::size_t tlen = seq.size();

        for (std::size_t i = 0; i < n; ++i) {
            auto pa = pi_acc[i].add(gamma[0][i]);
            if (!pa) {
                return make_error<HiddenMarkovModel>(pa.error());
            }
            pi_acc[i] = *pa;
            // Transition expected counts: t = 0 .. T-2.
            for (std::size_t t = 0; t + 1 < tlen; ++t) {
                auto ad = a_den[i].add(gamma[t][i]);
                if (!ad) {
                    return make_error<HiddenMarkovModel>(ad.error());
                }
                a_den[i] = *ad;
                for (std::size_t j = 0; j < n; ++j) {
                    auto an = a_num[i][j].add(xi[t][i][j]);
                    if (!an) {
                        return make_error<HiddenMarkovModel>(an.error());
                    }
                    a_num[i][j] = *an;
                }
            }
            // Emission expected counts: t = 0 .. T-1.
            for (std::size_t t = 0; t < tlen; ++t) {
                auto bd = b_den[i].add(gamma[t][i]);
                if (!bd) {
                    return make_error<HiddenMarkovModel>(bd.error());
                }
                b_den[i] = *bd;
                auto bn = b_num[i][seq[t]].add(gamma[t][i]);
                if (!bn) {
                    return make_error<HiddenMarkovModel>(bn.error());
                }
                b_num[i][seq[t]] = *bn;
            }
        }
    }

    // pi_i' = (1/R) sum_r gamma^r_0(i).
    const Rational r_count = Rational::from_int(static_cast<std::int64_t>(sequences.size()));
    std::vector<Rational> pi_new(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto v = pi_acc[i].divide(r_count);  // R >= 1
        if (!v) {
            return make_error<HiddenMarkovModel>(v.error());
        }
        pi_new[i] = *v;
    }
    // A(i, j)' = a_num[i][j] / a_den[i]; B(i, k)' = b_num[i][k] / b_den[i].
    std::vector<std::vector<Rational>> a_rows(n, std::vector<Rational>(n));
    std::vector<std::vector<Rational>> b_rows(n, std::vector<Rational>(m));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            auto v = a_num[i][j].divide(a_den[i]);  // a_den[i] == 0 => division_by_zero
            if (!v) {
                return make_error<HiddenMarkovModel>(v.error());
            }
            a_rows[i][j] = *v;
        }
        for (std::size_t k = 0; k < m; ++k) {
            auto v = b_num[i][k].divide(b_den[i]);  // b_den[i] == 0 => division_by_zero
            if (!v) {
                return make_error<HiddenMarkovModel>(v.error());
            }
            b_rows[i][k] = *v;
        }
    }
    auto a_mat = Matrix::from_rows(std::move(a_rows));
    if (!a_mat) {
        return make_error<HiddenMarkovModel>(a_mat.error());
    }
    auto b_mat = Matrix::from_rows(std::move(b_rows));
    if (!b_mat) {
        return make_error<HiddenMarkovModel>(b_mat.error());
    }
    // make() re-validates; the ratios sum to 1 exactly, so this succeeds by construction.
    return HiddenMarkovModel::make(std::move(pi_new), std::move(*a_mat), std::move(*b_mat));
}

// The EXACT joint likelihood over the independent sequences: prod_r P(O^r | model).
[[nodiscard]] auto joint_likelihood(const HiddenMarkovModel& model,
                                    std::span<const std::vector<std::size_t>> sequences)
    -> Result<Rational> {
    Rational total = one_rat();
    for (const std::vector<std::size_t>& seq : sequences) {
        auto lik = observation_likelihood(model, std::span<const std::size_t>{seq});
        if (!lik) {
            return lik;
        }
        auto next = total.multiply(*lik);
        if (!next) {
            return next;
        }
        total = *next;
    }
    return total;
}

}  // namespace

auto baum_welch(const HiddenMarkovModel& model,
                std::span<const std::vector<std::size_t>> sequences, std::size_t max_iterations)
    -> Result<BaumWelchResult> {
    if (sequences.empty()) {
        return make_error<BaumWelchResult>(MathError::domain_error);
    }
    HiddenMarkovModel current = model;
    std::vector<Rational> likelihoods;
    likelihoods.reserve(max_iterations + 1);
    for (std::size_t it = 0; it <= max_iterations; ++it) {
        auto lik = joint_likelihood(current, sequences);  // likelihood of the current model
        if (!lik) {
            return make_error<BaumWelchResult>(lik.error());
        }
        likelihoods.push_back(*lik);
        if (it == max_iterations) {
            break;  // recorded the final likelihood; no further update
        }
        auto next = baum_welch_step(current, sequences);
        if (!next) {
            return make_error<BaumWelchResult>(next.error());
        }
        current = std::move(*next);
    }
    return BaumWelchResult{.model = std::move(current), .likelihoods = std::move(likelihoods)};
}

// --- numerical scaled forward -----------------------------------------------

auto forward_scaled(const HiddenMarkovModel& model, std::span<const std::size_t> obs)
    -> Result<ScaledForward> {
    auto chk = check_obs(model, obs);
    if (!chk) {
        return make_error<ScaledForward>(chk.error());
    }
    const std::size_t n = model.num_states();
    const std::size_t tlen = obs.size();
    const Matrix& a = model.transition();
    const Matrix& b = model.emission();
    const std::span<const Rational> pi = model.initial();

    std::vector<std::vector<double>> alpha_hat(tlen, std::vector<double>(n, 0.0));
    double loglik = 0.0;
    // t = 0.
    double scale = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double v = rat_to_double(pi[i]) * rat_to_double(b.at(i, obs[0]));
        alpha_hat[0][i] = v;
        scale += v;
    }
    if (scale == 0.0) {
        return make_error<ScaledForward>(MathError::domain_error);  // impossible observation
    }
    loglik += std::log(scale);
    for (std::size_t i = 0; i < n; ++i) {
        alpha_hat[0][i] /= scale;
    }
    // t >= 1.
    for (std::size_t t = 1; t < tlen; ++t) {
        scale = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                s += alpha_hat[t - 1][i] * rat_to_double(a.at(i, j));
            }
            const double v = s * rat_to_double(b.at(j, obs[t]));
            alpha_hat[t][j] = v;
            scale += v;
        }
        if (scale == 0.0) {
            return make_error<ScaledForward>(MathError::domain_error);
        }
        loglik += std::log(scale);
        for (std::size_t j = 0; j < n; ++j) {
            alpha_hat[t][j] /= scale;
        }
    }
    return ScaledForward{.alpha_hat = std::move(alpha_hat), .log_likelihood = loglik};
}

auto log_likelihood(const HiddenMarkovModel& model, std::span<const std::size_t> obs)
    -> Result<double> {
    auto sf = forward_scaled(model, obs);
    if (!sf) {
        return make_error<double>(sf.error());
    }
    return sf->log_likelihood;
}

}  // namespace nimblecas
