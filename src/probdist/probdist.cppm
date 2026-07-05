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
//     hand from the standard definitions. These never fail.
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

// The exact symbolic transform data of one distribution. `pgf` is present only
// for the discrete (integer-supported) families; it is std::nullopt otherwise.
struct DistInfo {
    Expr mgf;                    // moment generating function  M_X(t) = E[e^{tX}]
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

// --- moments / cumulants --------------------------------------------------

auto raw_moment(const Expr& mgf, std::size_t k) -> Result<Expr> {
    return derivative_at_zero(mgf, k);
}

auto cumulant(const Expr& mgf, std::size_t k) -> Result<Expr> {
    return derivative_at_zero(lnf(mgf), k);
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

}  // namespace nimblecas
