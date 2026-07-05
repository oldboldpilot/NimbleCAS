// NimbleCAS the probabilistic method: non-constructive existence proofs (ROADMAP
// §7.18, "The Probabilistic Method"; Alon & Spencer).
// @author Olumuyiwa Oluwasanmi
//
// This module turns the classical existence arguments of the probabilistic method
// into EXACT, SOUND decision procedures over the rationals Q. Each entry point takes
// the exact numeric ingredients of an argument (an expectation, a list of bad-event
// probabilities, a variance, a dependency degree, ...) and returns a VERDICT together
// with the exact rational quantity the verdict rests on. It builds ON TOP of the tail
// inequalities of nimblecas.probdist (Markov / Chebyshev), which supply the analytic
// justification; this module supplies the existence layer — the step from "the bound
// is < 1" to "therefore a good configuration EXISTS".
//
// Honesty boundary (Rule 32 — nothing throws; a fallible step returns a MathError,
// and a positive existence certificate is NEVER a plausible-but-wrong verdict):
//
//   * first_moment / union_bound / second_moment / asymmetric-LLL / Ramsey are pure
//     EXACT rational (or exact BigInt) inequality checks. When the sufficient
//     hypothesis holds the verdict `exists` is a genuine proof; when it does not, the
//     verdict is `not_certified` — an honest "this argument does not apply", NEVER a
//     claim that no good configuration exists. The reported quantity (the summed
//     probability, the Chebyshev ratio, the per-event slack, the Ramsey threshold) is
//     the exact value, so the caller can audit the verdict independently.
//
//   * SYMMETRIC LOVÁSZ LOCAL LEMMA — the honesty crux. The symmetric LLL certifies
//     simultaneous avoidability of all bad events when  e · p · (d+1) ≤ 1, where e is
//     Euler's number, which is TRANSCENDENTAL and therefore NOT representable in Q.
//     We never compare against a fabricated rational "e". Instead we carry a rigorous
//     rational ENCLOSURE  e_lo < e < e_hi  (proven from the decimal expansion of e)
//     and decide three ways on the exact rational constraint c = p·(d+1):
//       - if  c · e_hi ≤ 1  then  e·c < e_hi·c ≤ 1, so the LLL hypothesis holds with
//         room to spare  ->  verdict `exists` (SOUND: e·p·(d+1) < 1 genuinely).
//       - if  c · e_lo > 1  then  e·c > e_lo·c > 1, so the hypothesis provably FAILS
//         ->  verdict `condition_fails` (no existence claim; the lemma is silent).
//       - otherwise c lies in the thin rational sliver around 1/e that the enclosure
//         cannot resolve  ->  verdict `indeterminate` (reported honestly, never guessed).
//     Because 1/e is irrational no rational c can hit the boundary exactly, so the
//     `indeterminate` band is only as wide as (e_hi − e_lo); tighten the enclosure to
//     shrink it. A positive certificate is thus ALWAYS sound.
//
// All comparisons are exact: Rationals are compared by __int128 cross-multiplication
// (both denominators are canonically positive), so no int64 product is ever formed and
// truncated. Where an exact quantity would leave int64 (a Rational product/quotient),
// the underlying overflow-checked Rational arithmetic surfaces MathError::overflow and
// this module propagates it rather than wrapping.

export module nimblecas.probmethod;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.bigint;
import nimblecas.bigcombinatorics;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// Verdicts.
// ---------------------------------------------------------------------------

// The two-state verdict shared by every exact-rational existence argument. `exists`
// is a SOUND non-constructive proof that a configuration with the desired property is
// present; `not_certified` means the sufficient hypothesis of THIS argument is not
// met — it is NOT a proof that no such configuration exists.
enum class Existence : std::uint8_t {
    exists,
    not_certified,
};

// The three-state verdict of the symmetric Lovász Local Lemma. Distinct from
// `Existence` because the transcendental constant e forces an honest middle state:
//   exists          — e·p·(d+1) < 1 proven via the rational enclosure of e (avoidable).
//   condition_fails — e·p·(d+1) > 1 proven; the LLL sufficient condition does not hold.
//   indeterminate   — p·(d+1) sits in the thin gap around 1/e the enclosure can't decide.
enum class LovaszVerdict : std::uint8_t {
    exists,
    condition_fails,
    indeterminate,
};

[[nodiscard]] constexpr auto to_string_view(Existence v) noexcept -> std::string_view {
    switch (v) {
        case Existence::exists:        return "exists";
        case Existence::not_certified: return "not_certified";
    }
    return "unknown";
}

[[nodiscard]] constexpr auto to_string_view(LovaszVerdict v) noexcept -> std::string_view {
    switch (v) {
        case LovaszVerdict::exists:          return "exists";
        case LovaszVerdict::condition_fails: return "condition_fails";
        case LovaszVerdict::indeterminate:   return "indeterminate";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// Result records — each carries the verdict and the exact quantity it rests on.
// ---------------------------------------------------------------------------

// first_moment_exists: for a non-negative INTEGER counting variable X (the number of
// "bad" events that occur), if E[X] < 1 then P(X = 0) > 0, so an outcome with X = 0 —
// a configuration avoiding every bad event — EXISTS. `attains_at_least_mean` and
// `attains_at_most_mean` record the (unconditional) averaging argument: some outcome
// has X ≥ E[X] and some outcome has X ≤ E[X]; both are always true for any variable
// with a finite mean, independent of the E[X] < 1 test.
struct FirstMoment {
    Existence verdict;               // exists  iff  expected_value < 1
    Rational expected_value;         // E[X], echoed back
    bool attains_at_least_mean;      // ∃ω: X(ω) ≥ E[X]  (averaging; always true)
    bool attains_at_most_mean;       // ∃ω: X(ω) ≤ E[X]  (averaging; always true)
};

// union_bound_exists: if the summed probability of the bad events is < 1 then, with
// positive probability, none of them occurs, so a good configuration EXISTS.
struct UnionBound {
    Existence verdict;               // exists  iff  total_probability < 1
    Rational total_probability;      // Σ_i P(bad_i), exact
};

// second_moment_positive: by Chebyshev, P(X = 0) ≤ Var(X)/E[X]^2. If that ratio is
// < 1 (equivalently Var(X) < E[X]^2) then P(X > 0) > 0, so an outcome with X > 0 —
// a configuration realising at least one desired object — EXISTS.
struct SecondMoment {
    Existence verdict;               // exists  iff  prob_zero_bound < 1
    Rational prob_zero_bound;        // the exact bound  P(X = 0) ≤ Var/E[X]^2
};

// lovasz_symmetric: the symmetric LLL. `constraint` is the exact rational c = p·(d+1);
// the lemma certifies simultaneous avoidability when e·c ≤ 1. See the honesty note at
// the top of this file for how the transcendental e is handled soundly.
struct LovaszSymmetric {
    LovaszVerdict verdict;
    Rational constraint;             // p·(d+1); the LLL asks whether e·constraint ≤ 1
};

// lovasz_asymmetric: the general (asymmetric) LLL with per-event weights x_i ∈ (0,1).
// The weights certify simultaneous avoidability iff, for every event i,
//   P(A_i) ≤ x_i · Π_{j ~ i} (1 − x_j).
// `slack[i]` is the exact margin  x_i·Π(1−x_j) − P(A_i)  (≥ 0 exactly where event i's
// inequality holds); `first_violation` is the index of the first event that fails the
// inequality, or probs.size() when every event passes.
struct LovaszAsymmetric {
    Existence verdict;               // exists  iff  every per-event inequality holds
    std::vector<Rational> slack;     // per-event  x_i·Π(1−x_j) − P(A_i)
    std::size_t first_violation;     // first failing index, or probs.size() if none
};

// ramsey_lower_bound: the first-moment Ramsey bound. A uniformly random 2-colouring of
// the edges of K_n has expected number of monochromatic K_k equal to
//   C(n,k) · 2^{1 − C(k,2)}.
// When that is < 1 — equivalently  C(n,k) < 2^{C(k,2) − 1}  — some colouring has none,
// so R(k,k) > n. `largest_n` is the greatest n for which this holds; `threshold` is the
// exact BigInt 2^{C(k,2) − 1} the binomial is compared against.
struct RamseyLowerBound {
    std::int64_t k;
    std::int64_t largest_n;          // R(k,k) > largest_n
    BigInt threshold;                // 2^{C(k,2) − 1}
};

// ---------------------------------------------------------------------------
// Rigorous rational enclosure of Euler's number e = 2.718281828459045235...
// ---------------------------------------------------------------------------
// e_lower_bound() < e < e_upper_bound(), both proven from the decimal expansion of e
// (itself a rigorous truncation of the convergent series e = Σ 1/n!). Exposed so the
// symmetric-LLL verdict can be audited: a certified `exists` always has p·(d+1) ≤
// 1/e_upper_bound() < 1/e, hence e·p·(d+1) < 1 genuinely.
[[nodiscard]] auto e_lower_bound() -> Rational;   // 2718281/1000000
[[nodiscard]] auto e_upper_bound() -> Rational;   // 2718282/1000000

// ---------------------------------------------------------------------------
// The existence procedures.
// ---------------------------------------------------------------------------

// First-moment / expectation argument. `expected_value` = E[X] for a non-negative
// integer counting variable; it must be ≥ 0 (domain_error otherwise).
[[nodiscard]] auto first_moment_exists(const Rational& expected_value) -> Result<FirstMoment>;

// Union bound over bad-event probabilities. Each probability must lie in [0, 1]
// (domain_error otherwise); an empty span sums to 0 (vacuously `exists`). Propagates
// MathError::overflow if the exact rational sum leaves int64.
[[nodiscard]] auto union_bound_exists(std::span<const Rational> probs) -> Result<UnionBound>;

// Second-moment method via Chebyshev. `mean` = E[X] must be > 0 (domain_error for
// mean ≤ 0, which also guards the E[X] = 0 division); `variance` = Var(X) must be ≥ 0.
// Propagates overflow from the exact Var / E[X]^2 arithmetic.
[[nodiscard]] auto second_moment_positive(const Rational& mean, const Rational& variance)
    -> Result<SecondMoment>;

// Symmetric Lovász Local Lemma. `p` (each bad event's probability, in [0, 1]) and
// `d` (the dependency degree, ≥ 0) with e·p·(d+1) ≤ 1 ⇒ all bad events are
// simultaneously avoidable. domain_error for p ∉ [0,1] or d < 0; overflow if p·(d+1)
// leaves int64. See the honesty note for the sound handling of e.
[[nodiscard]] auto lovasz_symmetric(const Rational& p, std::int64_t d)
    -> Result<LovaszSymmetric>;

// General / asymmetric Lovász Local Lemma. `probs[i]` = P(A_i) ∈ [0,1]; `dependency[i]`
// lists the neighbours j ~ i (indices in [0, n), j ≠ i) that A_i depends on;
// `x_weights[i]` = x_i ∈ (0,1) strictly. Returns whether the weights certify
// simultaneous avoidability, with the exact per-event slack. domain_error on a size
// mismatch, an out-of-range neighbour, a probability outside [0,1], or a weight outside
// the open interval (0,1); overflow from the exact product/difference arithmetic.
[[nodiscard]] auto lovasz_asymmetric(std::span<const Rational> probs,
                                     const std::vector<std::vector<int>>& dependency,
                                     std::span<const Rational> x_weights)
    -> Result<LovaszAsymmetric>;

// Does the first-moment Ramsey argument certify R(k,k) > n for this particular (n, k)?
// True iff C(n,k) < 2^{C(k,2) − 1} (exact BigInt comparison). domain_error for k < 2
// or n < 0.
[[nodiscard]] auto ramsey_certifies(std::int64_t n, std::int64_t k) -> Result<bool>;

// The largest n for which the first-moment argument certifies R(k,k) > n. Requires
// k ≥ 2 (domain_error otherwise). Uses exact BigInt binomials and 2^{C(k,2)−1}, so it
// never overflows; the search cost grows with k (the certified n grows exponentially).
[[nodiscard]] auto ramsey_lower_bound(std::int64_t k) -> Result<RamseyLowerBound>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// Construct a Rational from a known-valid integer fraction (e-enclosure constants and
// the literal 1). The fractions used here can never fail Rational::make (non-zero
// denominators, no INT64_MIN operands), so the unreachable error branch falls back to
// 0/1 rather than throwing — keeping the no-exception invariant (Rule 32).
[[nodiscard]] auto rat_const(std::int64_t num, std::int64_t den) -> Rational {
    auto r = Rational::make(num, den);
    return r.has_value() ? *r : Rational{};
}

// Exact three-way comparison of two canonical Rationals (denominators > 0), by
// __int128 cross-multiplication: sign of (a.num · b.den) − (b.num · a.den). No int64
// product is formed, so nothing is truncated; the operands' numerator·denominator
// products are ≤ ~8.5e37 in magnitude, well within the __int128 range.
[[nodiscard]] auto rat_compare(const Rational& a, const Rational& b) -> int {
    const __int128 lhs = static_cast<__int128>(a.numerator()) * b.denominator();
    const __int128 rhs = static_cast<__int128>(b.numerator()) * a.denominator();
    if (lhs < rhs) {
        return -1;
    }
    if (lhs > rhs) {
        return 1;
    }
    return 0;
}

[[nodiscard]] auto one() -> Rational { return Rational::from_int(1); }
[[nodiscard]] auto zero() -> Rational { return Rational{}; }

// a < b, a ≤ b, a > b as exact rational predicates.
[[nodiscard]] auto lt(const Rational& a, const Rational& b) -> bool { return rat_compare(a, b) < 0; }
[[nodiscard]] auto le(const Rational& a, const Rational& b) -> bool { return rat_compare(a, b) <= 0; }
[[nodiscard]] auto gt(const Rational& a, const Rational& b) -> bool { return rat_compare(a, b) > 0; }

}  // namespace

// --- e enclosure ----------------------------------------------------------

auto e_lower_bound() -> Rational { return rat_const(2718281, 1000000); }  // 2.718281 < e
auto e_upper_bound() -> Rational { return rat_const(2718282, 1000000); }  // 2.718282 > e

// --- first moment ---------------------------------------------------------

auto first_moment_exists(const Rational& expected_value) -> Result<FirstMoment> {
    // E[X] of a non-negative counting variable must be ≥ 0.
    if (lt(expected_value, zero())) {
        return make_error<FirstMoment>(MathError::domain_error);
    }
    const bool exists = lt(expected_value, one());  // E[X] < 1  ⇒  P(X = 0) > 0
    return FirstMoment{
        .verdict = exists ? Existence::exists : Existence::not_certified,
        .expected_value = expected_value,
        // The averaging argument holds unconditionally: the mean is a weighted average
        // of the attained values, so some outcome is ≥ the mean and some is ≤ it.
        .attains_at_least_mean = true,
        .attains_at_most_mean = true,
    };
}

// --- union bound ----------------------------------------------------------

auto union_bound_exists(std::span<const Rational> probs) -> Result<UnionBound> {
    Rational total = zero();
    for (const Rational& p : probs) {
        // Each entry must be a genuine probability in [0, 1].
        if (lt(p, zero()) || gt(p, one())) {
            return make_error<UnionBound>(MathError::domain_error);
        }
        auto next = total.add(p);
        if (!next) {
            return make_error<UnionBound>(next.error());  // exact sum left int64
        }
        total = *next;
    }
    const bool exists = lt(total, one());  // Σ P(bad_i) < 1  ⇒  P(no bad event) > 0
    return UnionBound{
        .verdict = exists ? Existence::exists : Existence::not_certified,
        .total_probability = total,
    };
}

// --- second moment --------------------------------------------------------

auto second_moment_positive(const Rational& mean, const Rational& variance)
    -> Result<SecondMoment> {
    // Var(X) ≥ 0, and E[X] > 0 (the latter also guards the E[X] = 0 division).
    if (lt(variance, zero())) {
        return make_error<SecondMoment>(MathError::domain_error);
    }
    if (le(mean, zero())) {
        return make_error<SecondMoment>(MathError::domain_error);
    }
    auto mean_sq = mean.multiply(mean);          // E[X]^2
    if (!mean_sq) {
        return make_error<SecondMoment>(mean_sq.error());
    }
    auto bound = variance.divide(*mean_sq);       // Var / E[X]^2  (mean_sq > 0)
    if (!bound) {
        return make_error<SecondMoment>(bound.error());
    }
    const bool exists = lt(*bound, one());        // bound < 1  ⇔  Var < E[X]^2  ⇒  P(X>0)>0
    return SecondMoment{
        .verdict = exists ? Existence::exists : Existence::not_certified,
        .prob_zero_bound = *bound,
    };
}

// --- symmetric Lovász Local Lemma -----------------------------------------

auto lovasz_symmetric(const Rational& p, std::int64_t d) -> Result<LovaszSymmetric> {
    // p is a probability in [0, 1]; d is a non-negative dependency degree.
    if (lt(p, zero()) || gt(p, one())) {
        return make_error<LovaszSymmetric>(MathError::domain_error);
    }
    if (d < 0) {
        return make_error<LovaszSymmetric>(MathError::domain_error);
    }
    if (d == std::numeric_limits<std::int64_t>::max()) {
        return make_error<LovaszSymmetric>(MathError::overflow);  // d + 1 would overflow
    }
    // c = p · (d + 1), exact.
    auto constraint = p.multiply(Rational::from_int(d + 1));
    if (!constraint) {
        return make_error<LovaszSymmetric>(constraint.error());
    }
    const Rational c = *constraint;

    // Decide e·c ≤ 1 three ways against the rigorous enclosure e_lo < e < e_hi, using
    // the reciprocals so no product with c is formed:  c ≤ 1/e_hi ⇔ c·e_hi ≤ 1  and
    // c > 1/e_lo ⇔ c·e_lo > 1. Both reciprocals are exact rationals.
    const Rational recip_e_hi = rat_const(1000000, 2718282);  // 1/e_hi
    const Rational recip_e_lo = rat_const(1000000, 2718281);  // 1/e_lo

    LovaszVerdict verdict = LovaszVerdict::indeterminate;
    if (le(c, recip_e_hi)) {
        // c·e_hi ≤ 1 and e < e_hi  ⇒  e·c < 1: SOUND certificate.
        verdict = LovaszVerdict::exists;
    } else if (gt(c, recip_e_lo)) {
        // c·e_lo > 1 and e > e_lo  ⇒  e·c > 1: the LLL condition provably fails.
        verdict = LovaszVerdict::condition_fails;
    }
    // Otherwise 1/e_hi < c ≤ 1/e_lo: the enclosure cannot resolve e·c vs 1 → indeterminate.
    return LovaszSymmetric{.verdict = verdict, .constraint = c};
}

// --- asymmetric Lovász Local Lemma ----------------------------------------

auto lovasz_asymmetric(std::span<const Rational> probs,
                       const std::vector<std::vector<int>>& dependency,
                       std::span<const Rational> x_weights) -> Result<LovaszAsymmetric> {
    const std::size_t n = probs.size();
    if (x_weights.size() != n || dependency.size() != n) {
        return make_error<LovaszAsymmetric>(MathError::domain_error);
    }
    // Validate inputs: probabilities in [0,1], weights strictly inside (0,1), and every
    // declared neighbour a valid event index.
    for (std::size_t i = 0; i < n; ++i) {
        if (lt(probs[i], zero()) || gt(probs[i], one())) {
            return make_error<LovaszAsymmetric>(MathError::domain_error);
        }
        if (le(x_weights[i], zero()) || rat_compare(x_weights[i], one()) >= 0) {
            return make_error<LovaszAsymmetric>(MathError::domain_error);  // x_i ∉ (0,1)
        }
        for (const int j : dependency[i]) {
            if (j < 0 || static_cast<std::size_t>(j) >= n) {
                return make_error<LovaszAsymmetric>(MathError::domain_error);
            }
        }
    }

    std::vector<Rational> slack;
    slack.reserve(n);
    std::size_t first_violation = n;
    for (std::size_t i = 0; i < n; ++i) {
        // rhs_i = x_i · Π_{j ~ i} (1 − x_j), exact.
        Rational rhs = x_weights[i];
        for (const int j : dependency[i]) {
            auto factor = one().subtract(x_weights[static_cast<std::size_t>(j)]);  // 1 − x_j
            if (!factor) {
                return make_error<LovaszAsymmetric>(factor.error());
            }
            auto prod = rhs.multiply(*factor);
            if (!prod) {
                return make_error<LovaszAsymmetric>(prod.error());
            }
            rhs = *prod;
        }
        auto margin = rhs.subtract(probs[i]);  // slack_i = rhs_i − P(A_i)
        if (!margin) {
            return make_error<LovaszAsymmetric>(margin.error());
        }
        slack.push_back(*margin);
        if (first_violation == n && gt(probs[i], rhs)) {  // P(A_i) > rhs_i ⇒ violation
            first_violation = i;
        }
    }
    const bool exists = (first_violation == n);
    return LovaszAsymmetric{
        .verdict = exists ? Existence::exists : Existence::not_certified,
        .slack = std::move(slack),
        .first_violation = first_violation,
    };
}

// --- Ramsey lower bounds --------------------------------------------------

namespace {

// The exact threshold 2^{C(k,2) − 1} as a BigInt, for k ≥ 2. Returns the exponent's
// value through a BigInt::pow; C(k,2) = k(k-1)/2 is formed in __int128 so the exponent
// itself cannot overflow for any representable k.
[[nodiscard]] auto ramsey_threshold(std::int64_t k) -> Result<BigInt> {
    if (k < 2) {
        return make_error<BigInt>(MathError::domain_error);
    }
    const __int128 c_k_2 = static_cast<__int128>(k) * (k - 1) / 2;  // C(k, 2), exact
    const __int128 exponent = c_k_2 - 1;                            // ≥ 0 since k ≥ 2
    if (exponent > static_cast<__int128>(std::numeric_limits<std::uint64_t>::max())) {
        return make_error<BigInt>(MathError::overflow);             // absurdly large k
    }
    return BigInt::from_u64(2).pow(static_cast<std::uint64_t>(exponent));
}

}  // namespace

auto ramsey_certifies(std::int64_t n, std::int64_t k) -> Result<bool> {
    if (k < 2 || n < 0) {
        return make_error<bool>(MathError::domain_error);
    }
    auto threshold = ramsey_threshold(k);
    if (!threshold) {
        return make_error<bool>(threshold.error());
    }
    auto choose = binomial(n, k);  // BigInt C(n, k); 0 when k > n
    if (!choose) {
        return make_error<bool>(choose.error());
    }
    return *choose < *threshold;  // C(n,k) < 2^{C(k,2)-1}
}

auto ramsey_lower_bound(std::int64_t k) -> Result<RamseyLowerBound> {
    if (k < 2) {
        return make_error<RamseyLowerBound>(MathError::domain_error);
    }
    auto threshold = ramsey_threshold(k);
    if (!threshold) {
        return make_error<RamseyLowerBound>(threshold.error());
    }
    // C(n,k) is 0 for n < k and strictly increasing to infinity for n ≥ k, so it crosses
    // the threshold exactly once; scan upward and keep the last n that stays below it.
    std::int64_t largest = -1;
    for (std::int64_t n = 0;; ++n) {
        auto choose = binomial(n, k);
        if (!choose) {
            return make_error<RamseyLowerBound>(choose.error());
        }
        if (*choose < *threshold) {
            largest = n;
        } else {
            break;  // C(n,k) ≥ threshold: no larger n can satisfy it either.
        }
    }
    return RamseyLowerBound{.k = k, .largest_n = largest, .threshold = std::move(*threshold)};
}

}  // namespace nimblecas
