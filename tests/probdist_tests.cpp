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
                  tc.expect(d.mgf.is_equivalent_to(affine_p(p, expf(t))),
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
                  tc.expect(d.mgf.is_equivalent_to(Expr::power(affine_p(p, expf(t)), n)),
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
                  tc.expect(d.mgf.is_equivalent_to(expf(Expr::product({lambda, sub(expf(t), i(1))}))),
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
                  tc.expect(d.mgf.is_equivalent_to(
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
                  tc.expect(d.mgf.is_equivalent_to(ratio(lambda, sub(lambda, t))),
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
                  tc.expect(d.mgf.is_equivalent_to(expected_mgf),
                            "Normal MGF = exp(mu t + sigma^2 t^2 / 2)");
                  tc.expect(!d.pgf.has_value(), "Normal has no PGF (continuous)");
                  tc.expect(d.mean.is_equivalent_to(mu), "Normal mean = mu");
                  tc.expect(d.variance.is_equivalent_to(sigma2), "Normal var = sigma^2");
              })
        .test("gamma_catalog",
              [&](TestContext& tc) {
                  const auto d = nimblecas::gamma(alpha, theta);
                  tc.expect(d.mgf.is_equivalent_to(
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
                  const auto m0 = nimblecas::raw_moment(d.mgf, 0);
                  tc.expect(m0.has_value() && m0->is_equivalent_to(i(1)),
                            "M_X(0) = 1 (0th raw moment)");
                  const auto m1 = nimblecas::raw_moment(d.mgf, 1);
                  tc.expect(m1.has_value() && m1->is_equivalent_to(simp(recip(lambda))),
                            "E[X] = 1/lambda (d/dt M at 0)");
                  const auto m2 = nimblecas::raw_moment(d.mgf, 2);
                  tc.expect(m2.has_value() &&
                                m2->is_equivalent_to(simp(Expr::product({i(2), Expr::power(lambda, i(-2))}))),
                            "E[X^2] = 2/lambda^2 (d^2/dt^2 M at 0)");
              })
        .test("raw_moment_gamma",
              [&](TestContext& tc) {
                  const auto d = nimblecas::gamma(alpha, theta);
                  const auto m0 = nimblecas::raw_moment(d.mgf, 0);
                  tc.expect(m0.has_value() && m0->is_equivalent_to(i(1)), "M_X(0) = 1");
                  const auto m1 = nimblecas::raw_moment(d.mgf, 1);
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
                  const auto k1 = nimblecas::cumulant(d.mgf, 1);
                  tc.expect(k1.has_value() && k1->is_equivalent_to(mu),
                            "kappa_1 = mu (= mean)");
                  const auto k2 = nimblecas::cumulant(d.mgf, 2);
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
                  const auto k1 = nimblecas::cumulant(d.mgf, 1);
                  const auto k2 = nimblecas::cumulant(d.mgf, 2);
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
                  tc.expect(nimblecas::chernoff_bound(d.mgf, a).is_equivalent_to(
                                Expr::product({expf(neg(Expr::product({t, a}))), d.mgf})),
                            "Chernoff bound = e^{-t alpha} M_X(t)");
              })
        .run();
}
