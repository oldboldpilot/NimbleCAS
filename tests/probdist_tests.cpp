// Tests for nimblecas.probdist: exact symbolic generating functions, moments,
// cumulants, and tail bounds for the standard distribution catalog.
// @author Olumuyiwa Oluwasanmi
//
// Every expectation is hand-derived and exact. The catalog forms are checked by
// structural equality (is_equivalent_to) against independently hand-built Exprs.
// The moment machinery (differentiate at t=0, then simplify) is verified on the
// cases whose t=0 evaluation reduces to a clean closed form WITHOUT relying on
// transcendental-value folding, which the engine deliberately does not do
// (simplify leaves exp(0) as exp(0)):
//   * raw moments of the rational/polynomial MGFs (Exponential, Gamma), and
//   * cumulants of exp-based MGFs where exp(A)^(-1)*exp(A) cancels (Normal).

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.diff;
import nimblecas.simplify;
import nimblecas.probdist;
import nimblecas.testing;

using nimblecas::Expr;
using nimblecas::simplify;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// --- exact builders mirroring the module's construction -------------------
[[nodiscard]] auto i(std::int64_t v) -> Expr { return Expr::integer(v); }
[[nodiscard]] auto sym(std::string_view s) -> Expr { return Expr::symbol(std::string(s)); }
[[nodiscard]] auto neg(const Expr& a) -> Expr { return Expr::product({i(-1), a}); }
[[nodiscard]] auto recip(const Expr& a) -> Expr { return Expr::power(a, i(-1)); }
[[nodiscard]] auto sub(const Expr& a, const Expr& b) -> Expr { return Expr::sum({a, neg(b)}); }
[[nodiscard]] auto ratio(const Expr& a, const Expr& b) -> Expr {
    return Expr::product({a, recip(b)});
}
[[nodiscard]] auto expf(const Expr& a) -> Expr { return Expr::apply("exp", {a}); }
[[nodiscard]] auto one_minus(const Expr& p) -> Expr { return sub(i(1), p); }
[[nodiscard]] auto affine_p(const Expr& p, const Expr& x) -> Expr {
    return Expr::sum({i(1), neg(p), Expr::product({p, x})});
}
[[nodiscard]] auto geom_denom(const Expr& p, const Expr& x) -> Expr {
    return sub(i(1), Expr::product({one_minus(p), x}));
}
[[nodiscard]] auto add(const Expr& a, const Expr& b) -> Expr { return Expr::sum({a, b}); }
[[nodiscard]] auto square(const Expr& a) -> Expr { return Expr::power(a, i(2)); }
[[nodiscard]] auto imag_unit() -> Expr { return Expr::power(i(-1), recip(i(2))); }
[[nodiscard]] auto gammaf(const Expr& u) -> Expr { return Expr::apply("gamma", {u}); }
[[nodiscard]] auto sum_of_squares(const std::vector<Expr>& xs) -> Expr {
    std::vector<Expr> terms;
    for (const Expr& x : xs) {
        terms.push_back(square(x));
    }
    return Expr::sum(std::move(terms));
}

// Canonical (simplified) form of an expected Expr — moment results come back
// simplified, so comparisons are made against the simplified expectation.
[[nodiscard]] auto simp(const Expr& e) -> Expr { return simplify(e).value(); }

// Differentiate `e` once w.r.t. `var`, evaluate at `var = value`, and simplify.
[[nodiscard]] auto d_at(const Expr& e, std::string_view var, std::int64_t value) -> Expr {
    const Expr d = nimblecas::differentiate(e, var).value();
    return simp(nimblecas::substitute(d, sym(var), i(value)));
}

}  // namespace

auto main() -> int {
    const Expr t = sym("t");
    const Expr z = sym("z");
    const Expr p = sym("p");
    const Expr n = sym("n");
    const Expr lambda = sym("lambda");
    const Expr mu = sym("mu");
    const Expr sigma2 = sym("sigma2");
    const Expr alpha = sym("alpha");
    const Expr theta = sym("theta");

    return TestSuite("nimblecas.probdist")
        // ---- catalog: generating functions, mean, variance (structural) ----
        .test("bernoulli_catalog",
              [&](TestContext& tc) {
                  const auto d = nimblecas::bernoulli(p);
                  tc.expect(d.mgf.has_value() && d.mgf->is_equivalent_to(affine_p(p, expf(t))),
                            "Bernoulli MGF = 1 - p + p e^t");
                  tc.expect(d.pgf.has_value() && d.pgf->is_equivalent_to(affine_p(p, z)),
                            "Bernoulli PGF = 1 - p + p z");
                  tc.expect(d.mean.is_equivalent_to(p), "Bernoulli mean = p");
                  tc.expect(d.variance.is_equivalent_to(Expr::product({p, one_minus(p)})),
                            "Bernoulli var = p(1-p)");
              })
        .test("binomial_catalog",
              [&](TestContext& tc) {
                  const auto d = nimblecas::binomial(n, p);
                  tc.expect(d.mgf.has_value() && d.mgf->is_equivalent_to(Expr::power(affine_p(p, expf(t)), n)),
                            "Binomial MGF = (1 - p + p e^t)^n");
                  tc.expect(d.pgf.has_value() &&
                                d.pgf->is_equivalent_to(Expr::power(affine_p(p, z), n)),
                            "Binomial PGF = (1 - p + p z)^n");
                  tc.expect(d.mean.is_equivalent_to(Expr::product({n, p})), "Binomial mean = n p");
                  tc.expect(d.variance.is_equivalent_to(Expr::product({n, p, one_minus(p)})),
                            "Binomial var = n p (1-p)");
              })
        .test("poisson_catalog",
              [&](TestContext& tc) {
                  const auto d = nimblecas::poisson(lambda);
                  tc.expect(d.mgf.has_value() && d.mgf->is_equivalent_to(expf(Expr::product({lambda, sub(expf(t), i(1))}))),
                            "Poisson MGF = exp(lambda(e^t - 1))");
                  tc.expect(d.pgf.has_value() &&
                                d.pgf->is_equivalent_to(expf(Expr::product({lambda, sub(z, i(1))}))),
                            "Poisson PGF = exp(lambda(z - 1))");
                  tc.expect(d.mean.is_equivalent_to(lambda), "Poisson mean = lambda");
                  tc.expect(d.variance.is_equivalent_to(lambda), "Poisson var = lambda");
              })
        .test("geometric_catalog",
              [&](TestContext& tc) {
                  const auto d = nimblecas::geometric(p);
                  tc.expect(d.pgf.has_value() &&
                                d.pgf->is_equivalent_to(
                                    ratio(Expr::product({p, z}), geom_denom(p, z))),
                            "Geometric PGF = p z / (1 - (1-p) z)");
                  tc.expect(d.mgf.has_value() && d.mgf->is_equivalent_to(
                                ratio(Expr::product({p, expf(t)}), geom_denom(p, expf(t)))),
                            "Geometric MGF = p e^t / (1 - (1-p) e^t)");
                  tc.expect(d.mean.is_equivalent_to(recip(p)), "Geometric mean = 1/p");
                  tc.expect(d.variance.is_equivalent_to(
                                Expr::product({one_minus(p), Expr::power(p, i(-2))})),
                            "Geometric var = (1-p)/p^2");
              })
        .test("exponential_catalog",
              [&](TestContext& tc) {
                  const auto d = nimblecas::exponential(lambda);
                  tc.expect(d.mgf.has_value() && d.mgf->is_equivalent_to(ratio(lambda, sub(lambda, t))),
                            "Exponential MGF = lambda/(lambda - t)");
                  tc.expect(!d.pgf.has_value(), "Exponential has no PGF (continuous)");
                  tc.expect(d.mean.is_equivalent_to(recip(lambda)), "Exponential mean = 1/lambda");
                  tc.expect(d.variance.is_equivalent_to(Expr::power(lambda, i(-2))),
                            "Exponential var = 1/lambda^2");
              })
        .test("normal_catalog",
              [&](TestContext& tc) {
                  const auto d = nimblecas::normal(mu, sigma2);
                  const Expr expected_mgf = expf(Expr::sum({
                      Expr::product({mu, t}),
                      Expr::product({sigma2, Expr::power(t, i(2)), Expr::power(i(2), i(-1))}),
                  }));
                  tc.expect(d.mgf.has_value() && d.mgf->is_equivalent_to(expected_mgf),
                            "Normal MGF = exp(mu t + sigma^2 t^2 / 2)");
                  tc.expect(!d.pgf.has_value(), "Normal has no PGF (continuous)");
                  tc.expect(d.mean.is_equivalent_to(mu), "Normal mean = mu");
                  tc.expect(d.variance.is_equivalent_to(sigma2), "Normal var = sigma^2");
              })
        .test("gamma_catalog",
              [&](TestContext& tc) {
                  const auto d = nimblecas::gamma(alpha, theta);
                  tc.expect(d.mgf.has_value() && d.mgf->is_equivalent_to(
                                Expr::power(sub(i(1), Expr::product({theta, t})), neg(alpha))),
                            "Gamma MGF = (1 - theta t)^(-alpha)");
                  tc.expect(!d.pgf.has_value(), "Gamma has no PGF (continuous)");
                  tc.expect(d.mean.is_equivalent_to(Expr::product({alpha, theta})),
                            "Gamma mean = alpha theta");
                  tc.expect(d.variance.is_equivalent_to(
                                Expr::product({alpha, Expr::power(theta, i(2))})),
                            "Gamma var = alpha theta^2");
              })
        // ---- moment machinery: exact raw moments from rational MGFs ----
        .test("raw_moment_exponential",
              [&](TestContext& tc) {
                  const auto d = nimblecas::exponential(lambda);
                  const auto m0 = nimblecas::raw_moment(*d.mgf, 0);
                  tc.expect(m0.has_value() && m0->is_equivalent_to(i(1)),
                            "M_X(0) = 1 (0th raw moment)");
                  const auto m1 = nimblecas::raw_moment(*d.mgf, 1);
                  tc.expect(m1.has_value() && m1->is_equivalent_to(simp(recip(lambda))),
                            "E[X] = 1/lambda (d/dt M at 0)");
                  const auto m2 = nimblecas::raw_moment(*d.mgf, 2);
                  tc.expect(m2.has_value() &&
                                m2->is_equivalent_to(simp(Expr::product({i(2), Expr::power(lambda, i(-2))}))),
                            "E[X^2] = 2/lambda^2 (d^2/dt^2 M at 0)");
              })
        .test("raw_moment_gamma",
              [&](TestContext& tc) {
                  const auto d = nimblecas::gamma(alpha, theta);
                  const auto m0 = nimblecas::raw_moment(*d.mgf, 0);
                  tc.expect(m0.has_value() && m0->is_equivalent_to(i(1)), "M_X(0) = 1");
                  const auto m1 = nimblecas::raw_moment(*d.mgf, 1);
                  tc.expect(m1.has_value() && m1->is_equivalent_to(simp(Expr::product({alpha, theta}))),
                            "Gamma E[X] = alpha theta (matches closed-form mean)");
                  // The mean recovered from the MGF equals the catalog mean.
                  tc.expect(m1.has_value() && m1->is_equivalent_to(simp(d.mean)),
                            "raw_moment(1) == catalog mean");
              })
        // ---- cumulant machinery: exact cumulants from exp-based MGFs ----
        .test("cumulant_normal",
              [&](TestContext& tc) {
                  const auto d = nimblecas::normal(mu, sigma2);
                  const auto k1 = nimblecas::cumulant(*d.mgf, 1);
                  tc.expect(k1.has_value() && k1->is_equivalent_to(mu),
                            "kappa_1 = mu (= mean)");
                  const auto k2 = nimblecas::cumulant(*d.mgf, 2);
                  tc.expect(k2.has_value() && k2->is_equivalent_to(sigma2),
                            "kappa_2 = sigma^2 (= variance)");
                  tc.expect(k1.has_value() && k1->is_equivalent_to(d.mean),
                            "kappa_1 == catalog mean");
                  tc.expect(k2.has_value() && k2->is_equivalent_to(d.variance),
                            "kappa_2 == catalog variance");
              })
        .test("cumulant_poisson_all_equal",
              [&](TestContext& tc) {
                  // Every Poisson cumulant equals lambda; the derivative form of
                  // K'(t) = lambda e^t is stable under further differentiation, so
                  // kappa_1 and kappa_2 are the same exact expression (the engine
                  // leaves the residual exp(0) unevaluated — still exact).
                  const auto d = nimblecas::poisson(lambda);
                  const auto k1 = nimblecas::cumulant(*d.mgf, 1);
                  const auto k2 = nimblecas::cumulant(*d.mgf, 2);
                  tc.expect(k1.has_value() && k2.has_value(), "Poisson cumulants computed");
                  tc.expect(k1.has_value() && k2.has_value() && k1->is_equivalent_to(*k2),
                            "kappa_1 == kappa_2 for Poisson (all cumulants coincide)");
              })
        // ---- generating-function sanity: G(1) = 1, G'(1) = mean ----
        .test("pgf_normalization_and_mean",
              [&](TestContext& tc) {
                  const auto b = nimblecas::bernoulli(p);
                  const Expr g1 = simp(nimblecas::substitute(*b.pgf, z, i(1)));
                  tc.expect(g1.is_equivalent_to(i(1)), "Bernoulli G(1) = 1");
                  // G'(1) = E[X] = p.
                  tc.expect(d_at(*b.pgf, "z", 1).is_equivalent_to(p), "Bernoulli G'(1) = mean = p");

                  const auto bin = nimblecas::binomial(n, p);
                  const Expr bin_g1 = simp(nimblecas::substitute(*bin.pgf, z, i(1)));
                  tc.expect(bin_g1.is_equivalent_to(i(1)), "Binomial G(1) = 1");
                  tc.expect(d_at(*bin.pgf, "z", 1).is_equivalent_to(simp(Expr::product({n, p}))),
                            "Binomial G'(1) = mean = n p");
              })
        // ---- tail inequalities (exact bounding expressions) ----
        .test("tail_bounds",
              [&](TestContext& tc) {
                  const Expr a = sym("a");  // threshold alpha
                  const Expr k = sym("k");
                  const Expr mean = sym("m");
                  const Expr var = sym("s2");
                  const auto d = nimblecas::poisson(lambda);

                  tc.expect(nimblecas::markov_bound(mean, a).is_equivalent_to(ratio(mean, a)),
                            "Markov bound = E[X]/alpha");
                  tc.expect(nimblecas::chebyshev_bound(var, k).is_equivalent_to(
                                Expr::product({var, Expr::power(k, i(-2))})),
                            "Chebyshev bound = sigma^2/k^2");
                  tc.expect(nimblecas::cantelli_bound(var, k).is_equivalent_to(
                                ratio(var, Expr::sum({var, Expr::power(k, i(2))}))),
                            "Cantelli bound = sigma^2/(sigma^2 + k^2)");
                  tc.expect(nimblecas::chernoff_bound(*d.mgf, a).is_equivalent_to(
                                Expr::product({expf(neg(Expr::product({t, a}))), *d.mgf})),
                            "Chernoff bound = e^{-t alpha} M_X(t)");
              })
        // ---- extended catalog: discrete families with closed-form MGF/PGF ----
        .test("discrete_uniform_catalog",
              [&](TestContext& tc) {
                  const Expr a = sym("a");
                  const Expr b = sym("b");
                  const auto d = nimblecas::discrete_uniform(a, b);
                  const Expr nn = add(sub(b, a), i(1));  // b - a + 1
                  const Expr b1 = add(b, i(1));          // b + 1
                  tc.expect(
                      d.mgf.has_value() &&
                          d.mgf->is_equivalent_to(ratio(
                              sub(expf(Expr::product({a, t})), expf(Expr::product({b1, t}))),
                              Expr::product({nn, sub(i(1), expf(t))}))),
                      "DiscreteUniform MGF = (e^{at} - e^{(b+1)t})/(n(1 - e^t))");
                  tc.expect(d.pgf.has_value() &&
                                d.pgf->is_equivalent_to(
                                    ratio(sub(Expr::power(z, a), Expr::power(z, b1)),
                                          Expr::product({nn, sub(i(1), z)}))),
                            "DiscreteUniform PGF = (z^a - z^{b+1})/(n(1 - z))");
                  tc.expect(d.mean.is_equivalent_to(ratio(add(a, b), i(2))),
                            "DiscreteUniform mean = (a+b)/2");
                  tc.expect(d.variance.is_equivalent_to(ratio(sub(square(nn), i(1)), i(12))),
                            "DiscreteUniform var = (n^2 - 1)/12");
              })
        .test("negative_binomial_catalog",
              [&](TestContext& tc) {
                  const Expr r = sym("r");
                  const auto d = nimblecas::negative_binomial(r, p);
                  tc.expect(d.mgf.has_value() &&
                                d.mgf->is_equivalent_to(
                                    Expr::power(ratio(p, geom_denom(p, expf(t))), r)),
                            "NegBinomial MGF = (p/(1-(1-p)e^t))^r");
                  tc.expect(d.pgf.has_value() &&
                                d.pgf->is_equivalent_to(
                                    Expr::power(ratio(p, geom_denom(p, z)), r)),
                            "NegBinomial PGF = (p/(1-(1-p)z))^r");
                  tc.expect(d.mean.is_equivalent_to(Expr::product({r, one_minus(p), recip(p)})),
                            "NegBinomial mean = r(1-p)/p");
                  tc.expect(d.variance.is_equivalent_to(
                                Expr::product({r, one_minus(p), Expr::power(p, i(-2))})),
                            "NegBinomial var = r(1-p)/p^2");
              })
        // ---- extended catalog: continuous families with closed-form MGF ----
        .test("continuous_uniform_catalog",
              [&](TestContext& tc) {
                  const Expr a = sym("a");
                  const Expr b = sym("b");
                  const auto d = nimblecas::continuous_uniform(a, b);
                  tc.expect(
                      d.mgf.has_value() &&
                          d.mgf->is_equivalent_to(ratio(
                              sub(expf(Expr::product({t, b})), expf(Expr::product({t, a}))),
                              Expr::product({t, sub(b, a)}))),
                      "ContinuousUniform MGF = (e^{tb} - e^{ta})/(t(b-a))");
                  tc.expect(!d.pgf.has_value(), "ContinuousUniform has no PGF");
                  tc.expect(d.mean.is_equivalent_to(ratio(add(a, b), i(2))),
                            "ContinuousUniform mean = (a+b)/2");
                  tc.expect(d.variance.is_equivalent_to(ratio(square(sub(b, a)), i(12))),
                            "ContinuousUniform var = (b-a)^2/12");
              })
        .test("chi_squared_catalog",
              [&](TestContext& tc) {
                  const Expr k = sym("k");
                  const auto d = nimblecas::chi_squared(k);
                  // Hand-verified: MGF of chi^2_k is (1 - 2t)^{-k/2}.
                  tc.expect(d.mgf.has_value() &&
                                d.mgf->is_equivalent_to(Expr::power(
                                    sub(i(1), Expr::product({i(2), t})), neg(ratio(k, i(2))))),
                            "ChiSquared MGF = (1 - 2t)^{-k/2}");
                  tc.expect(!d.pgf.has_value(), "ChiSquared has no PGF");
                  tc.expect(d.mean.is_equivalent_to(k), "ChiSquared mean = k");
                  tc.expect(d.variance.is_equivalent_to(Expr::product({i(2), k})),
                            "ChiSquared var = 2k");
              })
        // ---- extended catalog: families with NO elementary MGF (honest nullopt) ----
        .test("student_t_no_mgf",
              [&](TestContext& tc) {
                  const Expr nu = sym("nu");
                  const auto d = nimblecas::student_t(nu);
                  tc.expect(!d.mgf.has_value(),
                            "Student-t MGF does not exist -> reported as std::nullopt");
                  tc.expect(!d.pgf.has_value(), "Student-t has no PGF");
                  tc.expect(d.mean.is_equivalent_to(i(0)), "Student-t mean = 0 (nu > 1)");
                  tc.expect(d.variance.is_equivalent_to(ratio(nu, sub(nu, i(2)))),
                            "Student-t var = nu/(nu-2) (nu > 2)");
              })
        .test("hypergeometric_no_mgf",
              [&](TestContext& tc) {
                  const Expr N = sym("N");
                  const Expr K = sym("K");
                  const auto d = nimblecas::hypergeometric(N, K, n);
                  tc.expect(!d.mgf.has_value(),
                            "Hypergeometric MGF has no elementary form -> std::nullopt");
                  tc.expect(!d.pgf.has_value(), "Hypergeometric PGF has no elementary form");
                  tc.expect(d.mean.is_equivalent_to(Expr::product({n, K, recip(N)})),
                            "Hypergeometric mean = nK/N");
                  tc.expect(
                      d.variance.is_equivalent_to(Expr::product(
                          {n, K, sub(N, K), sub(N, n),
                           recip(Expr::product({square(N), sub(N, i(1))}))})),
                      "Hypergeometric var = n(K/N)((N-K)/N)((N-n)/(N-1))");
              })
        .test("beta_no_mgf",
              [&](TestContext& tc) {
                  const Expr beta = sym("beta");
                  const auto d = nimblecas::beta(alpha, beta);
                  const Expr apb = add(alpha, beta);
                  tc.expect(!d.mgf.has_value(),
                            "Beta MGF is confluent-hypergeometric -> std::nullopt");
                  tc.expect(!d.pgf.has_value(), "Beta has no PGF");
                  tc.expect(d.mean.is_equivalent_to(ratio(alpha, apb)),
                            "Beta mean = alpha/(alpha+beta)");
                  tc.expect(d.variance.is_equivalent_to(Expr::product(
                                {alpha, beta,
                                 recip(Expr::product({square(apb), add(apb, i(1))}))})),
                            "Beta var = alpha beta/((alpha+beta)^2 (alpha+beta+1))");
              })
        .test("weibull_no_mgf",
              [&](TestContext& tc) {
                  const Expr k = sym("k");
                  const auto d = nimblecas::weibull(k, lambda);
                  const Expr g1 = gammaf(add(i(1), recip(k)));                       // Gamma(1+1/k)
                  const Expr g2 = gammaf(add(i(1), Expr::product({i(2), recip(k)})));  // Gamma(1+2/k)
                  tc.expect(!d.mgf.has_value(),
                            "Weibull MGF has no elementary form -> std::nullopt");
                  tc.expect(!d.pgf.has_value(), "Weibull has no PGF");
                  tc.expect(d.mean.is_equivalent_to(Expr::product({lambda, g1})),
                            "Weibull mean = lambda Gamma(1+1/k)");
                  tc.expect(d.variance.is_equivalent_to(
                                Expr::product({square(lambda), sub(g2, square(g1))})),
                            "Weibull var = lambda^2 (Gamma(1+2/k) - Gamma(1+1/k)^2)");
              })
        .test("pareto_no_mgf",
              [&](TestContext& tc) {
                  const Expr xm = sym("xm");
                  const auto d = nimblecas::pareto(xm, alpha);
                  const Expr am1 = sub(alpha, i(1));
                  tc.expect(!d.mgf.has_value(),
                            "Pareto MGF does not exist for t>0 -> std::nullopt");
                  tc.expect(!d.pgf.has_value(), "Pareto has no PGF");
                  tc.expect(d.mean.is_equivalent_to(Expr::product({alpha, xm, recip(am1)})),
                            "Pareto mean = alpha xm/(alpha-1)");
                  tc.expect(d.variance.is_equivalent_to(Expr::product(
                                {square(xm), alpha,
                                 recip(Expr::product({square(am1), sub(alpha, i(2))}))})),
                            "Pareto var = xm^2 alpha/((alpha-1)^2 (alpha-2))");
              })
        .test("lognormal_no_mgf",
              [&](TestContext& tc) {
                  const auto d = nimblecas::lognormal(mu, sigma2);
                  tc.expect(!d.mgf.has_value(),
                            "Log-normal MGF diverges -> std::nullopt");
                  tc.expect(!d.pgf.has_value(), "Log-normal has no PGF");
                  tc.expect(d.mean.is_equivalent_to(expf(add(mu, ratio(sigma2, i(2))))),
                            "Log-normal mean = exp(mu + sigma2/2)");
                  tc.expect(d.variance.is_equivalent_to(Expr::product(
                                {sub(expf(sigma2), i(1)),
                                 expf(add(Expr::product({i(2), mu}), sigma2))})),
                            "Log-normal var = (exp(sigma2)-1) exp(2mu+sigma2)");
              })
        // ---- integral transforms of the generating functions ----
        .test("characteristic_function_normal",
              [&](TestContext& tc) {
                  // phi_X(t) = M_X(i t): the Normal MGF with t -> i*t, i = (-1)^{1/2}.
                  const auto d = nimblecas::normal(mu, sigma2);
                  const Expr cf = nimblecas::characteristic_function(*d.mgf);
                  const Expr R = Expr::product({imag_unit(), t});  // i*t
                  const Expr expected = expf(Expr::sum({
                      Expr::product({mu, R}),
                      Expr::product({sigma2, Expr::power(R, i(2)), Expr::power(i(2), i(-1))}),
                  }));
                  tc.expect(cf.is_equivalent_to(expected),
                            "phi_Normal(t) = exp(mu (it) + sigma2 (it)^2 / 2)");
              })
        .test("factorial_moment_binomial_and_poisson",
              [&](TestContext& tc) {
                  // Binomial PGF is polynomial: the factorial moments reduce fully.
                  const auto bin = nimblecas::binomial(n, p);
                  const auto f0 = nimblecas::factorial_moment(*bin.pgf, 0);
                  tc.expect(f0.has_value() && f0->is_equivalent_to(i(1)),
                            "G(1) = 1 (0th factorial moment)");
                  const auto f1 = nimblecas::factorial_moment(*bin.pgf, 1);
                  tc.expect(f1.has_value() && f1->is_equivalent_to(simp(Expr::product({n, p}))),
                            "Binomial 1st factorial moment = E[X] = n p");
                  // Poisson PGF exp(lambda(z-1)): k-th factorial moment = lambda^k, returned
                  // exactly as lambda^k * exp(0) (the simplifier leaves exp(0) unevaluated).
                  const auto d = nimblecas::poisson(lambda);
                  const auto p1 = nimblecas::factorial_moment(*d.pgf, 1);
                  tc.expect(p1.has_value() &&
                                p1->is_equivalent_to(simp(Expr::product({lambda, expf(i(0))}))),
                            "Poisson 1st factorial moment = lambda (as lambda*exp(0))");
                  const auto p2 = nimblecas::factorial_moment(*d.pgf, 2);
                  tc.expect(p2.has_value() &&
                                p2->is_equivalent_to(
                                    simp(Expr::product({square(lambda), expf(i(0))}))),
                            "Poisson 2nd factorial moment = lambda^2 (as lambda^2*exp(0))");
              })
        .test("laplace_stieltjes_exponential",
              [&](TestContext& tc) {
                  // LST_X(s) = M_X(-s): Exponential MGF with t -> -s, i.e. lambda/(lambda + s),
                  // returned unsimplified as lambda/(lambda - (-s)).
                  const auto d = nimblecas::exponential(lambda);
                  const Expr lst = nimblecas::laplace_stieltjes(*d.mgf);
                  const Expr s = sym("s");
                  tc.expect(lst.is_equivalent_to(ratio(lambda, sub(lambda, neg(s)))),
                            "LST_Exponential(s) = lambda/(lambda - (-s)) = lambda/(lambda + s)");
              })
        // ---- concentration inequalities (exact bounding expressions) ----
        .test("concentration_bounds",
              [&](TestContext& tc) {
                  const Expr tt = sym("t");
                  const Expr c1 = sym("c1");
                  const Expr c2 = sym("c2");
                  const Expr v = sym("v");
                  const Expr M = sym("M");
                  const std::vector<Expr> cs{c1, c2};
                  const Expr ss = sum_of_squares(cs);  // c1^2 + c2^2

                  tc.expect(nimblecas::hoeffding_bound(tt, cs).is_equivalent_to(
                                expf(Expr::product({i(-2), square(tt), recip(ss)}))),
                            "Hoeffding = exp(-2 t^2 / sum (b_i-a_i)^2)");
                  tc.expect(nimblecas::mcdiarmid_bound(tt, cs).is_equivalent_to(
                                expf(Expr::product({i(-2), square(tt), recip(ss)}))),
                            "McDiarmid = exp(-2 t^2 / sum c_i^2)");
                  tc.expect(nimblecas::azuma_bound(tt, cs).is_equivalent_to(expf(Expr::product(
                                {i(-1), square(tt), recip(Expr::product({i(2), ss}))}))),
                            "Azuma = exp(-t^2 / (2 sum c_i^2))");
                  const Expr mt3 = Expr::product({M, tt, recip(i(3))});  // M t / 3
                  tc.expect(
                      nimblecas::bernstein_bound(tt, v, M).is_equivalent_to(expf(Expr::product(
                          {i(-1), square(tt),
                           recip(Expr::product({i(2), add(v, mt3)}))}))),
                      "Bernstein = exp(-t^2 / (2(v + M t/3)))");
              })
        .run();
}
