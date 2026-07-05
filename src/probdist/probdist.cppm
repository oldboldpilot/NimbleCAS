// NimbleCAS exact symbolic probability-distribution catalog (ROADMAP §7.7.3 /
// §7.7.4 / §7.7.6): generating functions, moments, cumulants, and tail bounds.
// @author Olumuyiwa Oluwasanmi
//
// This module is an EXACT SYMBOLIC catalog. For each standard distribution it
// returns the moment generating function M_X(t) — and, for the lattice/discrete
// families, the probability generating function G_X(z) — together with the mean
// and variance, all as `Expr` trees in the parameter symbols and a transform
// variable ("t" for the MGF, "z" for the PGF). Nothing is numeric: a parameter
// such as p, lambda, mu, or sigma^2 is a symbol (or any caller-supplied `Expr`),
// and every result is the expression it mathematically IS.
//
// Honesty boundary (Rule 32 — nothing throws; fallible steps return a MathError):
//   * Generating functions, mean, variance: exact closed-form `Expr`, built by
//     hand from the standard definitions. These never fail. Where a family's MGF
//     (or PGF) has no elementary closed form or does not exist at all, the field
//     is std::nullopt — an honest not-available signal, never a fabricated form.
//   * Integral transforms (characteristic function, factorial moment, Laplace-
//     Stieltjes) are formal substitutions / derivatives of a supplied generating
//     function; factorial_moment may return an exact-but-unsimplified `Expr`.
//   * raw_moment / cumulant: the k-th raw moment is [d^k/dt^k M_X(t)]_{t=0} and
//     the k-th cumulant is [d^k/dt^k log M_X(t)]_{t=0}, computed by iterating
//     `nimblecas.diff::differentiate` and substituting t = 0, then running
//     `simplify`. The RESULT IS EXACT symbolically. It may not be maximally
//     simplified — if the engine cannot fold it to the textbook closed form the
//     (correct) unsimplified derivative `Expr` is returned; a plausible-but-wrong
//     "cleaned up" answer is NEVER fabricated. A failure inside differentiation
//     or simplification (e.g. overflow) is propagated as its MathError.
//   * Tail bounds: each function returns the exact bounding `Expr` (the RHS of a
//     classical inequality). It is an exact inequality under the hypotheses
//     stated in docs/reference/probdist.md — not a claim of tightness. The
//     Chernoff helper returns the PRE-optimization bound e^{-t*alpha}*M_X(t); the
//     infimum over t > 0 is the caller's to take.
//
// Division is represented as multiplication by a reciprocal power,
// x^(-1) = Expr::power(x, Expr::integer(-1)), matching the rest of the engine
// (see nimblecas.diff); no `Expr::rational` is required by this module.

export module nimblecas.probdist;

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.diff;
import nimblecas.simplify;

export namespace nimblecas {

// Names of the transform variables. The MGF is a function of "t"; the PGF is a
// function of "z". raw_moment / cumulant differentiate with respect to "t".
inline constexpr std::string_view mgf_variable = "t";
inline constexpr std::string_view pgf_variable = "z";
// The Laplace–Stieltjes transform variable: LST(s) = M_X(-s), a function of "s".
inline constexpr std::string_view laplace_variable = "s";

// The exact symbolic transform data of one distribution. `pgf` is present only
// for the discrete (integer-supported) families; it is std::nullopt otherwise.
// `mgf` is std::nullopt for the families whose moment generating function has no
// elementary closed form (Hypergeometric, Beta), or does not exist as a function
// on a neighbourhood of t = 0 at all (Student-t, Pareto, Log-normal, Weibull) —
// see the honesty boundary below. A not-available MGF is signalled by the empty
// optional; a plausible-but-wrong closed form is NEVER fabricated in its place.
struct DistInfo {
    std::optional<Expr> mgf;     // moment generating function  M_X(t) = E[e^{tX}]
    std::optional<Expr> pgf;     // probability generating fn    G_X(z) = E[z^X]
    Expr mean;                   // E[X]
    Expr variance;               // Var(X)
};

// --- Distribution catalog -------------------------------------------------
// Every parameter is a caller-supplied symbolic `Expr` (typically a symbol such
// as Expr::symbol("p")). None of these constructors can fail.

// Bernoulli(p):  MGF 1 - p + p e^t ;  PGF 1 - p + p z ;  mean p ;  var p(1-p).
[[nodiscard]] auto bernoulli(const Expr& p) -> DistInfo;

// Binomial(n, p): MGF (1 - p + p e^t)^n ; PGF (1 - p + p z)^n ;
//                 mean n p ; var n p (1-p).
[[nodiscard]] auto binomial(const Expr& n, const Expr& p) -> DistInfo;

// Poisson(lambda): MGF exp(lambda(e^t - 1)) ; PGF exp(lambda(z - 1)) ;
//                  mean lambda ; var lambda.
[[nodiscard]] auto poisson(const Expr& lambda) -> DistInfo;

// Geometric(p), support k = 1, 2, ... (number of trials up to first success):
//   PGF  p z / (1 - (1-p) z) ;  MGF  p e^t / (1 - (1-p) e^t) ;
//   mean 1/p ;  var (1-p)/p^2.
[[nodiscard]] auto geometric(const Expr& p) -> DistInfo;

// Exponential(lambda) (continuous): MGF lambda/(lambda - t) ;
//   mean 1/lambda ; var 1/lambda^2 ; no PGF.
[[nodiscard]] auto exponential(const Expr& lambda) -> DistInfo;

// Normal(mu, sigma2) where sigma2 = sigma^2 is the variance (continuous):
//   MGF exp(mu t + sigma2 t^2 / 2) ; mean mu ; var sigma2 ; no PGF.
[[nodiscard]] auto normal(const Expr& mu, const Expr& sigma2) -> DistInfo;

// Gamma(alpha, theta) with shape alpha and scale theta (continuous):
//   MGF (1 - theta t)^(-alpha) ; mean alpha theta ; var alpha theta^2 ; no PGF.
[[nodiscard]] auto gamma(const Expr& alpha, const Expr& theta) -> DistInfo;

// Discrete Uniform on the integers {a, a+1, ..., b}, a <= b, with n = b - a + 1
// equally likely values:
//   PGF  (z^a - z^(b+1)) / (n (1 - z)) ;  MGF  (e^(a t) - e^((b+1) t)) / (n (1 - e^t)) ;
//   mean (a + b)/2 ;  var (n^2 - 1)/12.
[[nodiscard]] auto discrete_uniform(const Expr& a, const Expr& b) -> DistInfo;

// Negative Binomial(r, p): number of failures X in {0, 1, 2, ...} before the
// r-th success (Pascal convention):
//   PGF  (p / (1 - (1-p) z))^r ;  MGF  (p / (1 - (1-p) e^t))^r ;
//   mean r(1-p)/p ;  var r(1-p)/p^2.
[[nodiscard]] auto negative_binomial(const Expr& r, const Expr& p) -> DistInfo;

// Hypergeometric(N, K, n): draw n without replacement from N items, K of which
// are successes. The MGF/PGF are Gauss hypergeometric ( ~ 2F1 ) forms with NO
// elementary closed form, so both are std::nullopt (honest not-available signal):
//   mean n K / N ;  var n (K/N)((N-K)/N)((N-n)/(N-1)).
[[nodiscard]] auto hypergeometric(const Expr& N, const Expr& K, const Expr& n) -> DistInfo;

// Continuous Uniform(a, b), a < b (continuous):
//   MGF (e^(t b) - e^(t a)) / (t (b - a)) ; mean (a+b)/2 ; var (b-a)^2/12 ; no PGF.
[[nodiscard]] auto continuous_uniform(const Expr& a, const Expr& b) -> DistInfo;

// Chi-squared(k) with k degrees of freedom (continuous):
//   MGF (1 - 2 t)^(-k/2) ; mean k ; var 2 k ; no PGF.
[[nodiscard]] auto chi_squared(const Expr& k) -> DistInfo;

// Student-t(nu) with nu degrees of freedom (continuous). The MGF DOES NOT EXIST
// (the integral E[e^{tX}] diverges for every t != 0), so mgf is std::nullopt:
//   mean 0 (defined for nu > 1) ; var nu/(nu - 2) (defined for nu > 2) ; no PGF.
[[nodiscard]] auto student_t(const Expr& nu) -> DistInfo;

// Beta(alpha, beta) on (0, 1) (continuous). The MGF is a confluent hypergeometric
// ( ~ 1F1 ) function with no elementary closed form, so mgf is std::nullopt:
//   mean alpha/(alpha+beta) ; var alpha beta / ((alpha+beta)^2 (alpha+beta+1)) ; no PGF.
[[nodiscard]] auto beta(const Expr& alpha, const Expr& beta) -> DistInfo;

// Weibull(k, lambda) with shape k and scale lambda (continuous). The MGF has no
// elementary closed form (an infinite series of Gamma values), so mgf is
// std::nullopt; the mean/variance use the Gamma function apply("gamma", .):
//   mean lambda * Gamma(1 + 1/k) ;
//   var  lambda^2 * ( Gamma(1 + 2/k) - Gamma(1 + 1/k)^2 ) ; no PGF.
[[nodiscard]] auto weibull(const Expr& k, const Expr& lambda) -> DistInfo;

// Pareto(xm, alpha) with scale xm > 0 and shape alpha > 0 (continuous). The MGF
// does not exist for any t > 0, so mgf is std::nullopt:
//   mean alpha xm/(alpha - 1) (defined for alpha > 1) ;
//   var  xm^2 alpha / ((alpha - 1)^2 (alpha - 2)) (defined for alpha > 2) ; no PGF.
[[nodiscard]] auto pareto(const Expr& xm, const Expr& alpha) -> DistInfo;

// Log-normal(mu, sigma2): X = e^Y with Y ~ Normal(mu, sigma2) and sigma2 = the
// variance of the underlying normal. The MGF DOES NOT EXIST (diverges for every
// t > 0), so mgf is std::nullopt, but the moments still have closed forms:
//   mean exp(mu + sigma2/2) ;
//   var  (exp(sigma2) - 1) exp(2 mu + sigma2) ; no PGF.
[[nodiscard]] auto lognormal(const Expr& mu, const Expr& sigma2) -> DistInfo;

// --- Moments and cumulants from the MGF -----------------------------------

// k-th raw moment  E[X^k] = [ d^k/dt^k M_X(t) ]_{t=0}. Differentiates `mgf`
// k times with respect to "t" (nimblecas.diff), substitutes t = 0, and
// simplifies. k = 0 yields M_X(0) = 1. Exact; may be unsimplified. Propagates a
// MathError raised by differentiation or simplification.
[[nodiscard]] auto raw_moment(const Expr& mgf, std::size_t k) -> Result<Expr>;

// k-th cumulant  kappa_k = [ d^k/dt^k log M_X(t) ]_{t=0}, with the cumulant
// generating function taken as log M_X(t) = ln(M_X(t)). kappa_1 is the mean and
// kappa_2 is the variance. Exact; may be unsimplified. Same error propagation.
[[nodiscard]] auto cumulant(const Expr& mgf, std::size_t k) -> Result<Expr>;

// --- Integral transforms of the MGF/PGF -----------------------------------

// Characteristic function  phi_X(t) = E[e^{i t X}] = M_X(i t), formed by the
// symbolic substitution t -> i*t where the imaginary unit i is represented as
// (-1)^(1/2) = Expr::power(Expr::integer(-1), Expr::power(Expr::integer(2),
// Expr::integer(-1))). No simplification of the resulting i^2 = -1 is performed;
// the result is the exact substituted `Expr`. Unlike the MGF, phi_X always
// exists, but this helper only performs the formal substitution on a supplied
// MGF expression (pass one only when the MGF is available).
[[nodiscard]] auto characteristic_function(const Expr& mgf) -> Expr;

// k-th factorial moment  E[X (X-1) ... (X-k+1)] = [ d^k/dz^k G_X(z) ]_{z=1}.
// Differentiates the PGF `pgf` k times with respect to "z" (nimblecas.diff),
// substitutes z = 1, and simplifies. k = 0 yields G_X(1) = 1. Exact; may be
// unsimplified. Propagates a MathError raised by differentiation or simplify.
[[nodiscard]] auto factorial_moment(const Expr& pgf, std::size_t k) -> Result<Expr>;

// Laplace–Stieltjes transform  LST_X(s) = E[e^{-s X}] = M_X(-s), formed by the
// symbolic substitution t -> -s. Exact substituted `Expr` (no simplification).
[[nodiscard]] auto laplace_stieltjes(const Expr& mgf) -> Expr;

// --- Tail inequalities (exact bounding expressions) -----------------------
// Each returns the RHS of a classical bound; the hypotheses are in the doc.

// Markov:    P(X >= alpha) <= E[X] / alpha   (X >= 0, alpha > 0).
[[nodiscard]] auto markov_bound(const Expr& mean, const Expr& alpha) -> Expr;

// Chebyshev: P(|X - mu| >= k) <= sigma^2 / k^2   (k > 0).
[[nodiscard]] auto chebyshev_bound(const Expr& variance, const Expr& k) -> Expr;

// Cantelli (one-sided): P(X - mu >= k) <= sigma^2 / (sigma^2 + k^2)   (k > 0).
[[nodiscard]] auto cantelli_bound(const Expr& variance, const Expr& k) -> Expr;

// Chernoff (generic, pre-optimization): P(X >= alpha) <= e^{-t alpha} M_X(t) for
// every t > 0; the tightest bound is the infimum over t > 0, left to the caller.
[[nodiscard]] auto chernoff_bound(const Expr& mgf, const Expr& alpha) -> Expr;

// --- Concentration inequalities for sums / functions of independent vars ---
// Each returns the RHS bound as an `Expr`; `widths`/`diffs` are the per-variable
// range or bounded-difference constants, and the sum of their squares is built
// internally. The vectors must be non-empty; hypotheses are in the doc.

// Hoeffding: for S = sum of independent X_i with X_i in [a_i, b_i], with
//   t > 0:  P(S - E[S] >= t) <= exp( -2 t^2 / sum_i (b_i - a_i)^2 ).
// `widths` holds the range widths (b_i - a_i).
[[nodiscard]] auto hoeffding_bound(const Expr& t, const std::vector<Expr>& widths) -> Expr;

// Bernstein (variance-aware): for a sum of independent mean-zero X_i with
//   |X_i| <= M and total variance v = sum_i Var(X_i), for t > 0:
//   P(S >= t) <= exp( - t^2 / ( 2 (v + M t / 3) ) ).
[[nodiscard]] auto bernstein_bound(const Expr& t, const Expr& variance, const Expr& bound)
    -> Expr;

// McDiarmid (bounded differences): for f with bounded differences c_i and t > 0,
//   P(f - E[f] >= t) <= exp( -2 t^2 / sum_i c_i^2 ).
// `diffs` holds the bounded-difference constants c_i.
[[nodiscard]] auto mcdiarmid_bound(const Expr& t, const std::vector<Expr>& diffs) -> Expr;

// Azuma–Hoeffding (martingale): for a martingale with |X_k - X_{k-1}| <= c_k and
//   t > 0:  P(X_n - X_0 >= t) <= exp( - t^2 / ( 2 sum_i c_i^2 ) ).
// `diffs` holds the martingale-difference bounds c_i.
[[nodiscard]] auto azuma_bound(const Expr& t, const std::vector<Expr>& diffs) -> Expr;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// --- small exact builders (module linkage; not exported) ------------------
[[nodiscard]] auto int_(std::int64_t v) -> Expr { return Expr::integer(v); }

// -a  =  (-1) * a.
[[nodiscard]] auto neg(const Expr& a) -> Expr { return Expr::product({int_(-1), a}); }

// a^(-1)  — reciprocal, the engine's representation of division.
[[nodiscard]] auto recip(const Expr& a) -> Expr { return Expr::power(a, int_(-1)); }

// a - b  =  a + (-b).
[[nodiscard]] auto sub(const Expr& a, const Expr& b) -> Expr {
    return Expr::sum({a, neg(b)});
}

// a / b  =  a * b^(-1).
[[nodiscard]] auto ratio(const Expr& a, const Expr& b) -> Expr {
    return Expr::product({a, recip(b)});
}

[[nodiscard]] auto expf(const Expr& a) -> Expr { return Expr::apply("exp", {a}); }

// Natural log — the name the differentiation table recognises ("ln", not "log"),
// so d/dt of the cumulant generating function actually reduces via ln' = 1/u.
[[nodiscard]] auto lnf(const Expr& a) -> Expr { return Expr::apply("ln", {a}); }

// 1 - p.
[[nodiscard]] auto one_minus(const Expr& p) -> Expr { return sub(int_(1), p); }

// 1 - p + p*x  — the shared Bernoulli/Binomial affine core (x = e^t or x = z).
[[nodiscard]] auto affine_p(const Expr& p, const Expr& x) -> Expr {
    return Expr::sum({int_(1), neg(p), Expr::product({p, x})});
}

// 1 - (1-p)*x  — the shared Geometric denominator (x = z or x = e^t).
[[nodiscard]] auto geom_denom(const Expr& p, const Expr& x) -> Expr {
    return sub(int_(1), Expr::product({one_minus(p), x}));
}

[[nodiscard]] auto mgf_sym() -> Expr { return Expr::symbol(std::string(mgf_variable)); }
[[nodiscard]] auto pgf_sym() -> Expr { return Expr::symbol(std::string(pgf_variable)); }

// a + b.
[[nodiscard]] auto add(const Expr& a, const Expr& b) -> Expr { return Expr::sum({a, b}); }

// a^2.
[[nodiscard]] auto square(const Expr& a) -> Expr { return Expr::power(a, int_(2)); }

// The imaginary unit i, represented as (-1)^(1/2) with 1/2 written as 2^(-1),
// matching the module's reciprocal-power convention (no Expr::rational needed).
[[nodiscard]] auto imag_unit() -> Expr { return Expr::power(int_(-1), recip(int_(2))); }

// sum_i x_i^2 — the shared denominator core of the concentration bounds. The
// caller supplies a non-empty vector.
[[nodiscard]] auto sum_of_squares(const std::vector<Expr>& xs) -> Expr {
    std::vector<Expr> terms;
    terms.reserve(xs.size());
    for (const Expr& x : xs) {
        terms.push_back(square(x));
    }
    return Expr::sum(std::move(terms));
}

// The Gamma special function Gamma(u), the name the differentiation table uses.
[[nodiscard]] auto gammaf(const Expr& u) -> Expr { return Expr::apply("gamma", {u}); }

// Differentiate `expr` `k` times with respect to `mgf_variable`, then evaluate
// at t = 0 and simplify. Shared engine for raw_moment (expr = M) and cumulant
// (expr = ln M). Exact; the returned Expr may not be maximally simplified.
[[nodiscard]] auto derivative_at_zero(Expr expr, std::size_t k) -> Result<Expr> {
    for (std::size_t i = 0; i < k; ++i) {
        Result<Expr> d = differentiate(expr, mgf_variable);
        if (!d) {
            return make_error<Expr>(d.error());
        }
        expr = *d;
    }
    const Expr at_zero = substitute(expr, mgf_sym(), int_(0));
    return simplify(at_zero);
}

}  // namespace

// --- catalog --------------------------------------------------------------

auto bernoulli(const Expr& p) -> DistInfo {
    const Expr t = mgf_sym();
    const Expr z = pgf_sym();
    return DistInfo{
        .mgf = affine_p(p, expf(t)),
        .pgf = affine_p(p, z),
        .mean = p,
        .variance = Expr::product({p, one_minus(p)}),
    };
}

auto binomial(const Expr& n, const Expr& p) -> DistInfo {
    const Expr t = mgf_sym();
    const Expr z = pgf_sym();
    return DistInfo{
        .mgf = Expr::power(affine_p(p, expf(t)), n),
        .pgf = Expr::power(affine_p(p, z), n),
        .mean = Expr::product({n, p}),
        .variance = Expr::product({n, p, one_minus(p)}),
    };
}

auto poisson(const Expr& lambda) -> DistInfo {
    const Expr t = mgf_sym();
    const Expr z = pgf_sym();
    return DistInfo{
        .mgf = expf(Expr::product({lambda, sub(expf(t), int_(1))})),
        .pgf = expf(Expr::product({lambda, sub(z, int_(1))})),
        .mean = lambda,
        .variance = lambda,
    };
}

auto geometric(const Expr& p) -> DistInfo {
    const Expr t = mgf_sym();
    const Expr z = pgf_sym();
    return DistInfo{
        .mgf = ratio(Expr::product({p, expf(t)}), geom_denom(p, expf(t))),
        .pgf = ratio(Expr::product({p, z}), geom_denom(p, z)),
        .mean = recip(p),                                              // 1/p
        .variance = Expr::product({one_minus(p), Expr::power(p, int_(-2))}),  // (1-p)/p^2
    };
}

auto exponential(const Expr& lambda) -> DistInfo {
    const Expr t = mgf_sym();
    return DistInfo{
        .mgf = ratio(lambda, sub(lambda, t)),   // lambda / (lambda - t)
        .pgf = std::nullopt,
        .mean = recip(lambda),                  // 1/lambda
        .variance = Expr::power(lambda, int_(-2)),  // 1/lambda^2
    };
}

auto normal(const Expr& mu, const Expr& sigma2) -> DistInfo {
    const Expr t = mgf_sym();
    // exp( mu*t + sigma2 * t^2 * (1/2) ), with 1/2 written as 2^(-1).
    const Expr exponent = Expr::sum({
        Expr::product({mu, t}),
        Expr::product({sigma2, Expr::power(t, int_(2)), Expr::power(int_(2), int_(-1))}),
    });
    return DistInfo{
        .mgf = expf(exponent),
        .pgf = std::nullopt,
        .mean = mu,
        .variance = sigma2,
    };
}

auto gamma(const Expr& alpha, const Expr& theta) -> DistInfo {
    const Expr t = mgf_sym();
    return DistInfo{
        .mgf = Expr::power(sub(int_(1), Expr::product({theta, t})), neg(alpha)),  // (1 - theta t)^(-alpha)
        .pgf = std::nullopt,
        .mean = Expr::product({alpha, theta}),                    // alpha theta
        .variance = Expr::product({alpha, Expr::power(theta, int_(2))}),  // alpha theta^2
    };
}

auto discrete_uniform(const Expr& a, const Expr& b) -> DistInfo {
    const Expr t = mgf_sym();
    const Expr z = pgf_sym();
    const Expr n = add(sub(b, a), int_(1));      // b - a + 1
    const Expr b1 = add(b, int_(1));             // b + 1
    return DistInfo{
        // (e^(a t) - e^((b+1) t)) / (n (1 - e^t))
        .mgf = ratio(sub(expf(Expr::product({a, t})), expf(Expr::product({b1, t}))),
                     Expr::product({n, sub(int_(1), expf(t))})),
        // (z^a - z^(b+1)) / (n (1 - z))
        .pgf = ratio(sub(Expr::power(z, a), Expr::power(z, b1)),
                     Expr::product({n, sub(int_(1), z)})),
        .mean = ratio(add(a, b), int_(2)),                       // (a + b)/2
        .variance = ratio(sub(square(n), int_(1)), int_(12)),    // (n^2 - 1)/12
    };
}

auto negative_binomial(const Expr& r, const Expr& p) -> DistInfo {
    const Expr t = mgf_sym();
    const Expr z = pgf_sym();
    // (p / (1 - (1-p) x))^r  with x = e^t (MGF) or x = z (PGF).
    return DistInfo{
        .mgf = Expr::power(ratio(p, geom_denom(p, expf(t))), r),
        .pgf = Expr::power(ratio(p, geom_denom(p, z)), r),
        .mean = Expr::product({r, one_minus(p), recip(p)}),               // r(1-p)/p
        .variance = Expr::product({r, one_minus(p), Expr::power(p, int_(-2))}),  // r(1-p)/p^2
    };
}

auto hypergeometric(const Expr& N, const Expr& K, const Expr& n) -> DistInfo {
    // MGF/PGF are Gauss-hypergeometric with no elementary closed form: omitted
    // honestly as std::nullopt rather than fabricated.
    return DistInfo{
        .mgf = std::nullopt,
        .pgf = std::nullopt,
        .mean = Expr::product({n, K, recip(N)}),                 // n K / N
        // n * K * (N-K) * (N-n) / ( N^2 (N-1) )
        .variance = Expr::product({n, K, sub(N, K), sub(N, n),
                                   recip(Expr::product({square(N), sub(N, int_(1))}))}),
    };
}

auto continuous_uniform(const Expr& a, const Expr& b) -> DistInfo {
    const Expr t = mgf_sym();
    return DistInfo{
        // (e^(t b) - e^(t a)) / (t (b - a))
        .mgf = ratio(sub(expf(Expr::product({t, b})), expf(Expr::product({t, a}))),
                     Expr::product({t, sub(b, a)})),
        .pgf = std::nullopt,
        .mean = ratio(add(a, b), int_(2)),                       // (a + b)/2
        .variance = ratio(square(sub(b, a)), int_(12)),          // (b - a)^2 / 12
    };
}

auto chi_squared(const Expr& k) -> DistInfo {
    const Expr t = mgf_sym();
    return DistInfo{
        // (1 - 2 t)^(-k/2)
        .mgf = Expr::power(sub(int_(1), Expr::product({int_(2), t})),
                           neg(ratio(k, int_(2)))),
        .pgf = std::nullopt,
        .mean = k,                                // k
        .variance = Expr::product({int_(2), k}),  // 2 k
    };
}

auto student_t(const Expr& nu) -> DistInfo {
    // The MGF does not exist (diverges for all t != 0): std::nullopt, not faked.
    return DistInfo{
        .mgf = std::nullopt,
        .pgf = std::nullopt,
        .mean = int_(0),                        // 0 (defined for nu > 1)
        .variance = ratio(nu, sub(nu, int_(2))),  // nu/(nu - 2) (defined for nu > 2)
    };
}

auto beta(const Expr& alpha, const Expr& beta) -> DistInfo {
    // MGF is a confluent hypergeometric function, no elementary closed form.
    const Expr apb = add(alpha, beta);  // alpha + beta
    return DistInfo{
        .mgf = std::nullopt,
        .pgf = std::nullopt,
        .mean = ratio(alpha, apb),      // alpha/(alpha+beta)
        // alpha beta / ( (alpha+beta)^2 (alpha+beta+1) )
        .variance = Expr::product({alpha, beta,
                                   recip(Expr::product({square(apb), add(apb, int_(1))}))}),
    };
}

auto weibull(const Expr& k, const Expr& lambda) -> DistInfo {
    // MGF has no elementary closed form (an infinite series of Gamma values).
    const Expr one_plus_1k = add(int_(1), recip(k));               // 1 + 1/k
    const Expr one_plus_2k = add(int_(1), Expr::product({int_(2), recip(k)}));  // 1 + 2/k
    return DistInfo{
        .mgf = std::nullopt,
        .pgf = std::nullopt,
        .mean = Expr::product({lambda, gammaf(one_plus_1k)}),  // lambda Gamma(1 + 1/k)
        // lambda^2 ( Gamma(1 + 2/k) - Gamma(1 + 1/k)^2 )
        .variance = Expr::product({square(lambda),
                                   sub(gammaf(one_plus_2k), square(gammaf(one_plus_1k)))}),
    };
}

auto pareto(const Expr& xm, const Expr& alpha) -> DistInfo {
    // MGF does not exist for any t > 0.
    const Expr am1 = sub(alpha, int_(1));  // alpha - 1
    return DistInfo{
        .mgf = std::nullopt,
        .pgf = std::nullopt,
        .mean = Expr::product({alpha, xm, recip(am1)}),  // alpha xm/(alpha - 1)
        // xm^2 alpha / ( (alpha-1)^2 (alpha-2) )
        .variance = Expr::product({square(xm), alpha,
                                   recip(Expr::product({square(am1), sub(alpha, int_(2))}))}),
    };
}

auto lognormal(const Expr& mu, const Expr& sigma2) -> DistInfo {
    // MGF does not exist (diverges for every t > 0); the moments are still closed.
    return DistInfo{
        .mgf = std::nullopt,
        .pgf = std::nullopt,
        // exp(mu + sigma2/2)
        .mean = expf(add(mu, ratio(sigma2, int_(2)))),
        // (exp(sigma2) - 1) exp(2 mu + sigma2)
        .variance = Expr::product({sub(expf(sigma2), int_(1)),
                                   expf(add(Expr::product({int_(2), mu}), sigma2))}),
    };
}

// --- moments / cumulants --------------------------------------------------

auto raw_moment(const Expr& mgf, std::size_t k) -> Result<Expr> {
    return derivative_at_zero(mgf, k);
}

auto cumulant(const Expr& mgf, std::size_t k) -> Result<Expr> {
    return derivative_at_zero(lnf(mgf), k);
}

// --- integral transforms --------------------------------------------------

auto characteristic_function(const Expr& mgf) -> Expr {
    // phi_X(t) = M_X(i t): substitute t -> i*t.
    return substitute(mgf, mgf_sym(), Expr::product({imag_unit(), mgf_sym()}));
}

auto factorial_moment(const Expr& pgf, std::size_t k) -> Result<Expr> {
    Expr expr = pgf;
    for (std::size_t i = 0; i < k; ++i) {
        Result<Expr> d = differentiate(expr, pgf_variable);
        if (!d) {
            return make_error<Expr>(d.error());
        }
        expr = *d;
    }
    const Expr at_one = substitute(expr, pgf_sym(), int_(1));
    return simplify(at_one);
}

auto laplace_stieltjes(const Expr& mgf) -> Expr {
    // LST_X(s) = M_X(-s): substitute t -> -s.
    const Expr s = Expr::symbol(std::string(laplace_variable));
    return substitute(mgf, mgf_sym(), neg(s));
}

// --- tail bounds ----------------------------------------------------------

auto markov_bound(const Expr& mean, const Expr& alpha) -> Expr {
    return ratio(mean, alpha);  // E[X] / alpha
}

auto chebyshev_bound(const Expr& variance, const Expr& k) -> Expr {
    return Expr::product({variance, Expr::power(k, int_(-2))});  // sigma^2 / k^2
}

auto cantelli_bound(const Expr& variance, const Expr& k) -> Expr {
    // sigma^2 / (sigma^2 + k^2)
    return ratio(variance, Expr::sum({variance, Expr::power(k, int_(2))}));
}

auto chernoff_bound(const Expr& mgf, const Expr& alpha) -> Expr {
    const Expr t = mgf_sym();
    // e^{-t alpha} * M_X(t)
    return Expr::product({expf(neg(Expr::product({t, alpha}))), mgf});
}

// --- concentration inequalities -------------------------------------------

auto hoeffding_bound(const Expr& t, const std::vector<Expr>& widths) -> Expr {
    // exp( -2 t^2 / sum_i (b_i - a_i)^2 )
    return expf(Expr::product({int_(-2), square(t), recip(sum_of_squares(widths))}));
}

auto bernstein_bound(const Expr& t, const Expr& variance, const Expr& bound) -> Expr {
    // exp( - t^2 / ( 2 (v + M t / 3) ) )
    const Expr mt3 = Expr::product({bound, t, recip(int_(3))});  // M t / 3
    const Expr denom = Expr::product({int_(2), add(variance, mt3)});
    return expf(Expr::product({int_(-1), square(t), recip(denom)}));
}

auto mcdiarmid_bound(const Expr& t, const std::vector<Expr>& diffs) -> Expr {
    // exp( -2 t^2 / sum_i c_i^2 )
    return expf(Expr::product({int_(-2), square(t), recip(sum_of_squares(diffs))}));
}

auto azuma_bound(const Expr& t, const std::vector<Expr>& diffs) -> Expr {
    // exp( - t^2 / ( 2 sum_i c_i^2 ) )
    const Expr denom = Expr::product({int_(2), sum_of_squares(diffs)});
    return expf(Expr::product({int_(-1), square(t), recip(denom)}));
}

}  // namespace nimblecas
