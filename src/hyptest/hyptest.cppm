// NimbleCAS exact hypothesis-test statistics & maximum-likelihood estimation
// (ROADMAP §7.7.7). @author Olumuyiwa Oluwasanmi
//
// Two exact deliverables, both over the rationals and the symbolic engine:
//
//  (1) TEST STATISTICS.  A test statistic formed from rational data by the four
//      arithmetic operations is itself an exact Rational, and this module returns
//      it as the fraction it mathematically IS — never a double. The catalog
//      covers the one-/two-sample and paired t families, the one-sample z, the
//      chi-squared goodness-of-fit and contingency-table (independence) statistics,
//      the variance-ratio F, and the one-way ANOVA F, each with its exact rational
//      value and integer degrees of freedom.
//
//  (2) MAXIMUM-LIKELIHOOD ESTIMATION.  For the standard one-parameter families
//      (Bernoulli, Poisson, Exponential, Normal mean with known variance, Geometric)
//      this module builds the log-likelihood ell(theta) as an `Expr` in the
//      sufficient-statistic symbols, differentiates it with `nimblecas.diff` to get
//      the score U(theta) = d ell / d theta, records the closed-form MLE theta-hat,
//      and the per-observation Fisher information I(theta) as a closed-form `Expr`.
//      The point estimate from actual rational data (p-hat = xbar, lambda-hat = 1/xbar,
//      ...) is returned as an exact Rational via `nimblecas.stats`.
//
// Honesty boundary (Rule 32; the convention shared with nimblecas.stats /
// statinfer / probdist). Everything that is rational is returned exactly; nothing
// that would require an irrational or a transcendental is faked.
//
//  * A t-statistic and a z-statistic carry a 1/sqrt(var) factor, so t and z are in
//    general IRRATIONAL. Following the stats.cppm "*_squared" convention, this
//    module returns the exact rational SQUARE t^2 / z^2 (which is what is compared
//    against a chi-squared / F critical value anyway, since t^2 ~ F(1, df) and
//    z^2 ~ chi^2(1)). The sign of t is recoverable exactly as sign(xbar - mu0).
//  * The chi-squared (goodness-of-fit and independence), variance-ratio F, and
//    ANOVA F statistics are RATIONAL as computed and are returned directly.
//  * A p-VALUE is the tail integral of a t / chi^2 / F / normal density — a
//    transcendental number NOT representable over Q. This module therefore returns
//    NO p-values. Instead it returns the exact statistic and its degrees of freedom,
//    and offers `exceeds(statistic, critical)`, an exact rational comparison against
//    a caller-supplied (rational) critical value — the honest decision rule when the
//    critical value is known. For a t^2 / z^2 statistic the caller supplies the
//    SQUARE of the critical value.
//  * The likelihood-ratio statistic G^2 = 2( ell(theta-hat) - ell(theta0) ) is a
//    difference of logarithms and is transcendental in general; `log_likelihood_ratio`
//    therefore returns it as an exact SYMBOLIC `Expr` (it may contain `ln`), not a
//    fabricated rational. The Wald and score statistics, whose closed forms ARE
//    rational for the Bernoulli and Poisson families, are returned as exact Rationals.
//
// Arithmetic is exact and overflow-checked throughout: an int64 numerator/denominator
// that would overflow surfaces as MathError::overflow rather than wrapping. Empty
// data, mismatched lengths, a degenerate (zero-variance / zero-count) configuration,
// and a too-small sample surface as MathError::domain_error.

export module nimblecas.hyptest;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.stats;
import nimblecas.statinfer;
import nimblecas.probdist;
import nimblecas.symbolic;
import nimblecas.diff;
import nimblecas.simplify;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// Test statistics.
// ---------------------------------------------------------------------------

// An exact test statistic with its integer degrees of freedom. `value` is the
// exact rational statistic (the SQUARE t^2 / z^2 for the t / z families — see the
// module honesty boundary — and the statistic itself for chi^2 / F / ANOVA).
// `df1` is the (numerator) degrees of freedom; `df2` is the denominator degrees of
// freedom for the F family and is std::nullopt for the single-df statistics.
struct TestStatistic {
    Rational value;
    std::int64_t df1;
    std::optional<std::int64_t> df2;
};

// One-sample t^2 = n (xbar - mu0)^2 / s^2, with s^2 the sample variance and
// df1 = n - 1. Returns the exact rational SQUARE of the t-statistic (t itself carries
// a 1/sqrt(s^2/n) factor and is generally irrational); recover sign(t) = sign(xbar - mu0).
// Needs n >= 2 and a non-zero sample variance; otherwise domain_error.
[[nodiscard]] auto one_sample_t_squared(std::span<const Rational> data, const Rational& mu0)
    -> Result<TestStatistic>;

// Two-sample pooled t^2 = (xbar - ybar)^2 / ( s_p^2 (1/n1 + 1/n2) ), with the pooled
// variance s_p^2 = ((n1-1)s1^2 + (n2-1)s2^2)/(n1+n2-2) and df1 = n1 + n2 - 2. Exact
// rational square of the pooled two-sample t. Needs n1, n2 >= 2 and a non-zero pooled
// variance; otherwise domain_error.
[[nodiscard]] auto two_sample_t_squared(std::span<const Rational> x, std::span<const Rational> y)
    -> Result<TestStatistic>;

// Paired t^2: the one-sample t^2 of the differences d_i = x_i - y_i against mu0 = 0,
// df1 = n - 1. x and y must have the same length n >= 2 and the differences must have
// a non-zero sample variance; otherwise domain_error.
[[nodiscard]] auto paired_t_squared(std::span<const Rational> x, std::span<const Rational> y)
    -> Result<TestStatistic>;

// One-sample z^2 = n (xbar - mu0)^2 / sigma2 for a KNOWN population variance sigma2 > 0.
// Exact rational square of the z-statistic (z^2 ~ chi^2 with 1 degree of freedom, so
// df1 = 1). Needs n >= 1 and sigma2 > 0; otherwise domain_error.
[[nodiscard]] auto z_squared(std::span<const Rational> data, const Rational& mu0,
                             const Rational& pop_variance) -> Result<TestStatistic>;

// Chi-squared goodness-of-fit statistic sum_i (O_i - E_i)^2 / E_i, EXACT rational,
// df1 = k - 1 (k = number of categories). observed and expected must have equal length
// k >= 2 and every expected count E_i must be strictly positive; otherwise domain_error.
[[nodiscard]] auto chi_squared_goodness_of_fit(std::span<const Rational> observed,
                                               std::span<const Rational> expected)
    -> Result<TestStatistic>;

// Chi-squared test of independence on an r x c contingency table of observed counts:
// with row totals R_i, column totals C_j and grand total N, expected E_ij = R_i C_j / N
// and the statistic is sum_{ij} (O_ij - E_ij)^2 / E_ij, EXACT rational, with
// df1 = (r - 1)(c - 1). Needs r, c >= 2, a positive grand total, and every row/column
// total strictly positive (so every E_ij > 0); otherwise domain_error.
[[nodiscard]] auto chi_squared_independence(const Matrix& table) -> Result<TestStatistic>;

// Variance-ratio F = s_x^2 / s_y^2 (sample variances), EXACT rational, with
// df1 = n_x - 1 (numerator) and df2 = n_y - 1 (denominator). Needs n_x, n_y >= 2 and a
// non-zero denominator variance s_y^2; otherwise domain_error.
[[nodiscard]] auto variance_ratio_f(std::span<const Rational> x, std::span<const Rational> y)
    -> Result<TestStatistic>;

// One-way ANOVA F = (SS_between / (k-1)) / (SS_within / (N-k)), EXACT rational, where
// SS_between = sum_g n_g (xbar_g - xbar)^2 and SS_within = sum_g sum_i (x_gi - xbar_g)^2,
// N = sum_g n_g. df1 = k - 1 (numerator), df2 = N - k (denominator). Needs k >= 2 groups,
// every group non-empty, N > k, and a non-zero SS_within; otherwise domain_error.
[[nodiscard]] auto one_way_anova_f(const std::vector<std::span<const Rational>>& groups)
    -> Result<TestStatistic>;

// Exact decision rule: true iff `statistic` > `critical`, by exact rational (cross-
// multiplied, overflow-checked) comparison. This is the honest replacement for a
// p-value when a rational critical value is known; for a t^2 / z^2 statistic pass the
// SQUARE of the critical value. Overflow in the comparison surfaces as overflow.
[[nodiscard]] auto exceeds(const Rational& statistic, const Rational& critical) -> Result<bool>;

// ---------------------------------------------------------------------------
// Maximum-likelihood estimation.
// ---------------------------------------------------------------------------

// The exact symbolic MLE data of a one-parameter family. Symbols used: `parameter`
// (the family parameter theta), "n" (sample size), "m" (the sample mean xbar), and for
// the Normal "v" (the mean of squares, (1/n) sum x_i^2) and "sigma2" (the known
// variance). Every field is exact:
//   * log_likelihood — ell(theta), the sample log-likelihood as an `Expr` in the
//     sufficient-statistic symbols (additive constants independent of theta are dropped,
//     as they do not affect the score or the MLE).
//   * score          — U(theta) = d ell / d theta, computed by `differentiate`; hence
//     substituting theta = mle and simplifying yields 0 (the defining property).
//   * mle            — theta-hat, the closed-form estimator as an `Expr` in the summary
//     symbols (e.g. Bernoulli p-hat = m, Exponential lambda-hat = m^(-1)).
//   * fisher_information — the PER-OBSERVATION Fisher information i(theta) as a closed-
//     form `Expr`; the total sample information is n * i(theta).
struct MleModel {
    std::string parameter;
    Expr log_likelihood;
    Expr score;
    Expr mle;
    Expr fisher_information;
};

// Bernoulli(p): ell = n*m*ln(p) + n*(1-m)*ln(1-p); U = n*m/p - n*(1-m)/(1-p);
// p-hat = m; i(p) = 1/(p(1-p)). Fails only if differentiation overflows.
[[nodiscard]] auto bernoulli_mle_model() -> Result<MleModel>;

// Poisson(lambda): ell = n*m*ln(lambda) - n*lambda; U = n*m/lambda - n;
// lambda-hat = m; i(lambda) = 1/lambda.
[[nodiscard]] auto poisson_mle_model() -> Result<MleModel>;

// Exponential(lambda): ell = n*ln(lambda) - n*m*lambda; U = n/lambda - n*m;
// lambda-hat = m^(-1); i(lambda) = 1/lambda^2.
[[nodiscard]] auto exponential_mle_model() -> Result<MleModel>;

// Normal(mu, sigma2) with KNOWN variance sigma2, estimating mu:
// ell = -(1/(2*sigma2)) * ( n*v - 2*n*m*mu + n*mu^2 ); U = (n*m - n*mu)/sigma2;
// mu-hat = m; i(mu) = 1/sigma2.
[[nodiscard]] auto normal_mean_mle_model() -> Result<MleModel>;

// Geometric(p), support k = 1, 2, ... (trials to first success):
// ell = n*ln(p) + n*(m-1)*ln(1-p); U = n/p - n*(m-1)/(1-p); p-hat = m^(-1);
// i(p) = 1/(p^2 (1-p)).
[[nodiscard]] auto geometric_mle_model() -> Result<MleModel>;

// --- exact rational point estimates from data (consume nimblecas.stats) ---

// Bernoulli / Poisson / Normal-mean MLE = the sample mean xbar, exact rational.
// Empty data fails domain_error.
[[nodiscard]] auto bernoulli_mle(std::span<const Rational> data) -> Result<Rational>;
[[nodiscard]] auto poisson_mle(std::span<const Rational> data) -> Result<Rational>;
[[nodiscard]] auto normal_mean_mle(std::span<const Rational> data) -> Result<Rational>;

// Exponential / Geometric MLE = 1/xbar, exact rational. Empty data or a zero sample
// mean (reciprocal undefined) fails domain_error.
[[nodiscard]] auto exponential_mle(std::span<const Rational> data) -> Result<Rational>;
[[nodiscard]] auto geometric_mle(std::span<const Rational> data) -> Result<Rational>;

// Normal variance MLE = the POPULATION variance (1/n) sum (x_i - xbar)^2, exact
// rational (the biased / divide-by-n form that maximises the likelihood). Empty data
// fails domain_error.
[[nodiscard]] auto normal_variance_mle(std::span<const Rational> data) -> Result<Rational>;

// --- exact rational Wald / score statistics (closed forms rational) ---

// Bernoulli Wald statistic W = n (p-hat - p0)^2 / (p-hat (1 - p-hat)) with p-hat = xbar.
// Exact rational. Empty data, or an estimate p-hat in {0, 1} (zero observed Fisher
// information), fails domain_error.
[[nodiscard]] auto bernoulli_wald_statistic(std::span<const Rational> data, const Rational& p0)
    -> Result<Rational>;

// Bernoulli score (Rao) statistic S = n (xbar - p0)^2 / (p0 (1 - p0)). Exact rational.
// Empty data, or a null p0 in {0, 1}, fails domain_error.
[[nodiscard]] auto bernoulli_score_statistic(std::span<const Rational> data, const Rational& p0)
    -> Result<Rational>;

// Poisson Wald statistic W = n (lambda-hat - lambda0)^2 / lambda-hat with lambda-hat = xbar.
// Exact rational. Empty data or a zero estimate fails domain_error.
[[nodiscard]] auto poisson_wald_statistic(std::span<const Rational> data, const Rational& lambda0)
    -> Result<Rational>;

// Poisson score (Rao) statistic S = n (xbar - lambda0)^2 / lambda0. Exact rational.
// Empty data or a zero null lambda0 fails domain_error.
[[nodiscard]] auto poisson_score_statistic(std::span<const Rational> data, const Rational& lambda0)
    -> Result<Rational>;

// The likelihood-ratio statistic G^2 = 2( ell(theta-hat) - ell(theta0) ), as an EXACT
// SYMBOLIC `Expr` obtained by substituting the two supplied parameter values into the
// model's log-likelihood and simplifying. It is transcendental in general (a difference
// of logarithms) and is therefore NOT returned as a rational; the caller evaluates it
// numerically at whatever precision is wanted. Propagates any simplify overflow.
[[nodiscard]] auto log_likelihood_ratio(const MleModel& model, const Expr& theta_hat,
                                        const Expr& theta0) -> Result<Expr>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// --- exact rational helpers ------------------------------------------------

[[nodiscard]] auto ri(std::size_t n) -> Rational {
    return Rational::from_int(static_cast<std::int64_t>(n));
}

[[nodiscard]] auto square(const Rational& a) -> Result<Rational> {
    return a.multiply(a);
}

// sum_i (a_i - c)^2 over a span (the deviation sum of squares about a scalar c).
[[nodiscard]] auto sum_sq_dev(std::span<const Rational> a, const Rational& c) -> Result<Rational> {
    Rational acc;  // 0/1
    for (const auto& e : a) {
        auto d = e.subtract(c);
        if (!d) {
            return make_error<Rational>(d.error());
        }
        auto sq = square(*d);
        if (!sq) {
            return make_error<Rational>(sq.error());
        }
        auto next = acc.add(*sq);
        if (!next) {
            return make_error<Rational>(next.error());
        }
        acc = *next;
    }
    return acc;
}

// Build a t^2-style statistic: numerator n*(mean - mu0)^2 over the variance factor,
// shared by the one-sample and paired forms. `var_factor` must be non-zero (checked).
[[nodiscard]] auto t_squared_from(std::span<const Rational> data, const Rational& mu0,
                                  const Rational& var_factor, std::int64_t df)
    -> Result<TestStatistic> {
    if (var_factor.is_zero()) {
        return make_error<TestStatistic>(MathError::domain_error);
    }
    auto m = mean(data);
    if (!m) {
        return make_error<TestStatistic>(m.error());
    }
    auto diff = m->subtract(mu0);
    if (!diff) {
        return make_error<TestStatistic>(diff.error());
    }
    auto dsq = square(*diff);
    if (!dsq) {
        return make_error<TestStatistic>(dsq.error());
    }
    auto num = ri(data.size()).multiply(*dsq);
    if (!num) {
        return make_error<TestStatistic>(num.error());
    }
    auto t2 = num->divide(var_factor);
    if (!t2) {
        return make_error<TestStatistic>(t2.error());
    }
    return TestStatistic{.value = *t2, .df1 = df, .df2 = std::nullopt};
}

// --- symbolic Expr helpers -------------------------------------------------

[[nodiscard]] auto sym(std::string name) -> Expr {
    return Expr::symbol(std::move(name));
}
[[nodiscard]] auto intg(std::int64_t v) -> Expr {
    return Expr::integer(v);
}
[[nodiscard]] auto ln(const Expr& e) -> Expr {
    return Expr::apply("ln", {e});
}
[[nodiscard]] auto negate_expr(const Expr& e) -> Expr {
    return Expr::product({Expr::integer(-1), e});
}
// e^(-1) — reciprocal as a power (matches the diff/probdist representation).
[[nodiscard]] auto inv(const Expr& e) -> Expr {
    return Expr::power(e, Expr::integer(-1));
}
// (1 - e).
[[nodiscard]] auto one_minus(const Expr& e) -> Expr {
    return Expr::sum({Expr::integer(1), negate_expr(e)});
}

// Assemble an MleModel from its symbolic pieces, differentiating the log-likelihood to
// obtain the score (so the score IS the exact derivative, never a hand-typed guess).
[[nodiscard]] auto make_model(std::string parameter, Expr log_likelihood, Expr mle,
                              Expr fisher) -> Result<MleModel> {
    auto score = differentiate(log_likelihood, parameter);
    if (!score) {
        return make_error<MleModel>(score.error());
    }
    return MleModel{.parameter = std::move(parameter),
                    .log_likelihood = std::move(log_likelihood),
                    .score = std::move(*score),
                    .mle = std::move(mle),
                    .fisher_information = std::move(fisher)};
}

// Reciprocal point estimate 1/xbar shared by the Exponential and Geometric MLEs.
[[nodiscard]] auto reciprocal_mean(std::span<const Rational> data) -> Result<Rational> {
    auto m = mean(data);  // domain_error on empty data
    if (!m) {
        return m;
    }
    if (m->is_zero()) {  // 1/xbar undefined
        return make_error<Rational>(MathError::domain_error);
    }
    return Rational::from_int(1).divide(*m);
}

}  // namespace

// --- test statistics -------------------------------------------------------

auto one_sample_t_squared(std::span<const Rational> data, const Rational& mu0)
    -> Result<TestStatistic> {
    if (data.size() < 2) {
        return make_error<TestStatistic>(MathError::domain_error);
    }
    auto s2 = variance(data, true);  // sample variance, needs n >= 2
    if (!s2) {
        return make_error<TestStatistic>(s2.error());
    }
    return t_squared_from(data, mu0, *s2, static_cast<std::int64_t>(data.size()) - 1);
}

auto two_sample_t_squared(std::span<const Rational> x, std::span<const Rational> y)
    -> Result<TestStatistic> {
    if (x.size() < 2 || y.size() < 2) {
        return make_error<TestStatistic>(MathError::domain_error);
    }
    const std::int64_t n1 = static_cast<std::int64_t>(x.size());
    const std::int64_t n2 = static_cast<std::int64_t>(y.size());
    auto s1 = variance(x, true);
    if (!s1) {
        return make_error<TestStatistic>(s1.error());
    }
    auto s2 = variance(y, true);
    if (!s2) {
        return make_error<TestStatistic>(s2.error());
    }
    // Pooled variance s_p^2 = ((n1-1)s1^2 + (n2-1)s2^2) / (n1+n2-2).
    auto a = Rational::from_int(n1 - 1).multiply(*s1);
    if (!a) {
        return make_error<TestStatistic>(a.error());
    }
    auto b = Rational::from_int(n2 - 1).multiply(*s2);
    if (!b) {
        return make_error<TestStatistic>(b.error());
    }
    auto pooled_num = a->add(*b);
    if (!pooled_num) {
        return make_error<TestStatistic>(pooled_num.error());
    }
    auto sp2 = pooled_num->divide(Rational::from_int(n1 + n2 - 2));
    if (!sp2) {
        return make_error<TestStatistic>(sp2.error());
    }
    if (sp2->is_zero()) {
        return make_error<TestStatistic>(MathError::domain_error);
    }
    // Standard-error factor s_p^2 (1/n1 + 1/n2).
    auto inv1 = Rational::from_int(1).divide(Rational::from_int(n1));
    if (!inv1) {
        return make_error<TestStatistic>(inv1.error());
    }
    auto inv2 = Rational::from_int(1).divide(Rational::from_int(n2));
    if (!inv2) {
        return make_error<TestStatistic>(inv2.error());
    }
    auto inv_sum = inv1->add(*inv2);
    if (!inv_sum) {
        return make_error<TestStatistic>(inv_sum.error());
    }
    auto se2 = sp2->multiply(*inv_sum);
    if (!se2) {
        return make_error<TestStatistic>(se2.error());
    }
    // t^2 = (xbar - ybar)^2 / se2.
    auto xbar = mean(x);
    if (!xbar) {
        return make_error<TestStatistic>(xbar.error());
    }
    auto ybar = mean(y);
    if (!ybar) {
        return make_error<TestStatistic>(ybar.error());
    }
    auto diff = xbar->subtract(*ybar);
    if (!diff) {
        return make_error<TestStatistic>(diff.error());
    }
    auto dsq = square(*diff);
    if (!dsq) {
        return make_error<TestStatistic>(dsq.error());
    }
    auto t2 = dsq->divide(*se2);
    if (!t2) {
        return make_error<TestStatistic>(t2.error());
    }
    return TestStatistic{.value = *t2, .df1 = n1 + n2 - 2, .df2 = std::nullopt};
}

auto paired_t_squared(std::span<const Rational> x, std::span<const Rational> y)
    -> Result<TestStatistic> {
    if (x.size() != y.size() || x.size() < 2) {
        return make_error<TestStatistic>(MathError::domain_error);
    }
    std::vector<Rational> d(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        auto di = x[i].subtract(y[i]);
        if (!di) {
            return make_error<TestStatistic>(di.error());
        }
        d[i] = *di;
    }
    return one_sample_t_squared(std::span<const Rational>{d}, Rational{});
}

auto z_squared(std::span<const Rational> data, const Rational& mu0, const Rational& pop_variance)
    -> Result<TestStatistic> {
    if (data.empty()) {
        return make_error<TestStatistic>(MathError::domain_error);
    }
    if (pop_variance.numerator() <= 0) {  // variance must be strictly positive
        return make_error<TestStatistic>(MathError::domain_error);
    }
    // z^2 ~ chi^2(1): df1 = 1. Reuses the t^2 numerator machinery over the known variance.
    return t_squared_from(data, mu0, pop_variance, 1);
}

auto chi_squared_goodness_of_fit(std::span<const Rational> observed,
                                 std::span<const Rational> expected) -> Result<TestStatistic> {
    if (observed.size() != expected.size() || observed.size() < 2) {
        return make_error<TestStatistic>(MathError::domain_error);
    }
    Rational acc;  // 0/1
    for (std::size_t i = 0; i < observed.size(); ++i) {
        if (expected[i].numerator() <= 0) {  // every expected count must be > 0
            return make_error<TestStatistic>(MathError::domain_error);
        }
        auto diff = observed[i].subtract(expected[i]);
        if (!diff) {
            return make_error<TestStatistic>(diff.error());
        }
        auto sq = square(*diff);
        if (!sq) {
            return make_error<TestStatistic>(sq.error());
        }
        auto term = sq->divide(expected[i]);
        if (!term) {
            return make_error<TestStatistic>(term.error());
        }
        auto next = acc.add(*term);
        if (!next) {
            return make_error<TestStatistic>(next.error());
        }
        acc = *next;
    }
    return TestStatistic{.value = acc,
                         .df1 = static_cast<std::int64_t>(observed.size()) - 1,
                         .df2 = std::nullopt};
}

auto chi_squared_independence(const Matrix& table) -> Result<TestStatistic> {
    const std::size_t r = table.rows();
    const std::size_t c = table.cols();
    if (r < 2 || c < 2) {
        return make_error<TestStatistic>(MathError::domain_error);
    }
    // Row totals, column totals, and grand total.
    std::vector<Rational> row_tot(r);
    std::vector<Rational> col_tot(c);
    Rational grand;  // 0/1
    for (std::size_t i = 0; i < r; ++i) {
        for (std::size_t j = 0; j < c; ++j) {
            auto rt = row_tot[i].add(table.at(i, j));
            if (!rt) {
                return make_error<TestStatistic>(rt.error());
            }
            row_tot[i] = *rt;
            auto ct = col_tot[j].add(table.at(i, j));
            if (!ct) {
                return make_error<TestStatistic>(ct.error());
            }
            col_tot[j] = *ct;
            auto g = grand.add(table.at(i, j));
            if (!g) {
                return make_error<TestStatistic>(g.error());
            }
            grand = *g;
        }
    }
    if (grand.numerator() <= 0) {  // need a positive grand total for E_ij = R_i C_j / N
        return make_error<TestStatistic>(MathError::domain_error);
    }
    for (const auto& rt : row_tot) {
        if (rt.numerator() <= 0) {  // a zero row total makes E_ij = 0 (undefined term)
            return make_error<TestStatistic>(MathError::domain_error);
        }
    }
    for (const auto& ct : col_tot) {
        if (ct.numerator() <= 0) {
            return make_error<TestStatistic>(MathError::domain_error);
        }
    }
    Rational acc;  // 0/1
    for (std::size_t i = 0; i < r; ++i) {
        for (std::size_t j = 0; j < c; ++j) {
            // Expected E_ij = R_i * C_j / N.
            auto rc = row_tot[i].multiply(col_tot[j]);
            if (!rc) {
                return make_error<TestStatistic>(rc.error());
            }
            auto e = rc->divide(grand);
            if (!e) {
                return make_error<TestStatistic>(e.error());
            }
            auto diff = table.at(i, j).subtract(*e);
            if (!diff) {
                return make_error<TestStatistic>(diff.error());
            }
            auto sq = square(*diff);
            if (!sq) {
                return make_error<TestStatistic>(sq.error());
            }
            auto term = sq->divide(*e);
            if (!term) {
                return make_error<TestStatistic>(term.error());
            }
            auto next = acc.add(*term);
            if (!next) {
                return make_error<TestStatistic>(next.error());
            }
            acc = *next;
        }
    }
    const std::int64_t df = (static_cast<std::int64_t>(r) - 1) * (static_cast<std::int64_t>(c) - 1);
    return TestStatistic{.value = acc, .df1 = df, .df2 = std::nullopt};
}

auto variance_ratio_f(std::span<const Rational> x, std::span<const Rational> y)
    -> Result<TestStatistic> {
    if (x.size() < 2 || y.size() < 2) {
        return make_error<TestStatistic>(MathError::domain_error);
    }
    auto vx = variance(x, true);
    if (!vx) {
        return make_error<TestStatistic>(vx.error());
    }
    auto vy = variance(y, true);
    if (!vy) {
        return make_error<TestStatistic>(vy.error());
    }
    if (vy->is_zero()) {  // F undefined with a zero denominator variance
        return make_error<TestStatistic>(MathError::domain_error);
    }
    auto f = vx->divide(*vy);
    if (!f) {
        return make_error<TestStatistic>(f.error());
    }
    return TestStatistic{.value = *f,
                         .df1 = static_cast<std::int64_t>(x.size()) - 1,
                         .df2 = static_cast<std::int64_t>(y.size()) - 1};
}

auto one_way_anova_f(const std::vector<std::span<const Rational>>& groups)
    -> Result<TestStatistic> {
    if (groups.size() < 2) {
        return make_error<TestStatistic>(MathError::domain_error);
    }
    std::size_t total = 0;
    for (const auto& g : groups) {
        if (g.empty()) {
            return make_error<TestStatistic>(MathError::domain_error);
        }
        total += g.size();
    }
    const std::int64_t k = static_cast<std::int64_t>(groups.size());
    if (static_cast<std::int64_t>(total) <= k) {  // need N > k for a positive within-df
        return make_error<TestStatistic>(MathError::domain_error);
    }
    // Grand mean = (sum over all data) / N. Also collect per-group means.
    Rational grand_sum;  // 0/1
    std::vector<Rational> group_means(groups.size());
    for (std::size_t gi = 0; gi < groups.size(); ++gi) {
        auto gm = mean(groups[gi]);
        if (!gm) {
            return make_error<TestStatistic>(gm.error());
        }
        group_means[gi] = *gm;
        for (const auto& v : groups[gi]) {
            auto s = grand_sum.add(v);
            if (!s) {
                return make_error<TestStatistic>(s.error());
            }
            grand_sum = *s;
        }
    }
    auto grand_mean = grand_sum.divide(ri(total));
    if (!grand_mean) {
        return make_error<TestStatistic>(grand_mean.error());
    }
    // SS_between = sum_g n_g (xbar_g - grand_mean)^2 ; SS_within = sum_g sum_i (x_gi - xbar_g)^2.
    Rational ss_between;  // 0/1
    Rational ss_within;   // 0/1
    for (std::size_t gi = 0; gi < groups.size(); ++gi) {
        auto dev = group_means[gi].subtract(*grand_mean);
        if (!dev) {
            return make_error<TestStatistic>(dev.error());
        }
        auto devsq = square(*dev);
        if (!devsq) {
            return make_error<TestStatistic>(devsq.error());
        }
        auto contrib = ri(groups[gi].size()).multiply(*devsq);
        if (!contrib) {
            return make_error<TestStatistic>(contrib.error());
        }
        auto b = ss_between.add(*contrib);
        if (!b) {
            return make_error<TestStatistic>(b.error());
        }
        ss_between = *b;
        auto within = sum_sq_dev(groups[gi], group_means[gi]);
        if (!within) {
            return make_error<TestStatistic>(within.error());
        }
        auto w = ss_within.add(*within);
        if (!w) {
            return make_error<TestStatistic>(w.error());
        }
        ss_within = *w;
    }
    if (ss_within.is_zero()) {  // F undefined (zero within-group variation)
        return make_error<TestStatistic>(MathError::domain_error);
    }
    // Mean squares and their ratio.
    auto ms_between = ss_between.divide(Rational::from_int(k - 1));
    if (!ms_between) {
        return make_error<TestStatistic>(ms_between.error());
    }
    auto ms_within = ss_within.divide(Rational::from_int(static_cast<std::int64_t>(total) - k));
    if (!ms_within) {
        return make_error<TestStatistic>(ms_within.error());
    }
    auto f = ms_between->divide(*ms_within);
    if (!f) {
        return make_error<TestStatistic>(f.error());
    }
    return TestStatistic{.value = *f,
                         .df1 = k - 1,
                         .df2 = static_cast<std::int64_t>(total) - k};
}

auto exceeds(const Rational& statistic, const Rational& critical) -> Result<bool> {
    // Denominators are positive in canonical form, so statistic > critical iff
    // statistic.num * critical.den > critical.num * statistic.den. Overflow-checked.
    std::int64_t lhs = 0;
    std::int64_t rhs = 0;
    if (__builtin_mul_overflow(statistic.numerator(), critical.denominator(), &lhs) ||
        __builtin_mul_overflow(critical.numerator(), statistic.denominator(), &rhs)) {
        return make_error<bool>(MathError::overflow);
    }
    return lhs > rhs;
}

// --- MLE models ------------------------------------------------------------

auto bernoulli_mle_model() -> Result<MleModel> {
    const Expr n = sym("n");
    const Expr m = sym("m");
    const Expr p = sym("p");
    // ell = n*m*ln(p) + n*(1-m)*ln(1-p). The (1-m) coefficient is kept in FACTORED
    // form so that substituting p = m folds the second term to n via the like-base
    // rule (1-m)^1 * (1-m)^(-1) -> (1-m)^0 = 1 (single-pass Cohen simplification).
    Expr ll = Expr::sum({Expr::product({n, m, ln(p)}),
                         Expr::product({n, one_minus(m), ln(one_minus(p))})});
    // i(p) = 1/(p(1-p)), written as the product of reciprocals p^(-1) (1-p)^(-1) so that
    // the defining identity i(p) * p * (1-p) = 1 folds by like-base cancellation.
    Expr fisher = Expr::product({inv(p), inv(one_minus(p))});
    return make_model("p", std::move(ll), m, std::move(fisher));
}

auto poisson_mle_model() -> Result<MleModel> {
    const Expr n = sym("n");
    const Expr m = sym("m");
    const Expr lambda = sym("lambda");
    // ell = n*m*ln(lambda) - n*lambda (the -sum ln(x_i!) constant is dropped).
    Expr ll = Expr::sum({Expr::product({n, m, ln(lambda)}),
                         negate_expr(Expr::product({n, lambda}))});
    Expr fisher = inv(lambda);  // 1/lambda
    return make_model("lambda", std::move(ll), m, std::move(fisher));
}

auto exponential_mle_model() -> Result<MleModel> {
    const Expr n = sym("n");
    const Expr m = sym("m");
    const Expr lambda = sym("lambda");
    // ell = n*ln(lambda) - n*m*lambda.
    Expr ll = Expr::sum({Expr::product({n, ln(lambda)}),
                         negate_expr(Expr::product({n, m, lambda}))});
    Expr mle = inv(m);          // lambda-hat = 1/m
    Expr fisher = inv(Expr::power(lambda, intg(2)));  // 1/lambda^2
    return make_model("lambda", std::move(ll), std::move(mle), std::move(fisher));
}

auto normal_mean_mle_model() -> Result<MleModel> {
    const Expr n = sym("n");
    const Expr m = sym("m");
    const Expr v = sym("v");            // mean of squares (1/n) sum x_i^2
    const Expr mu = sym("mu");
    const Expr sigma2 = sym("sigma2");  // known variance
    // sum (x_i - mu)^2 = n*v - 2*n*m*mu + n*mu^2 ; ell = -(1/(2*sigma2)) * that.
    Expr inner = Expr::sum({Expr::product({n, v}),
                            negate_expr(Expr::product({intg(2), n, m, mu})),
                            Expr::product({n, Expr::power(mu, intg(2))})});
    Expr prefactor = negate_expr(inv(Expr::product({intg(2), sigma2})));  // -(2*sigma2)^(-1)
    Expr ll = Expr::product({prefactor, inner});
    Expr fisher = inv(sigma2);  // 1/sigma2
    return make_model("mu", std::move(ll), m, std::move(fisher));
}

auto geometric_mle_model() -> Result<MleModel> {
    const Expr n = sym("n");
    const Expr m = sym("m");
    const Expr p = sym("p");
    // ell = n*ln(p) + n*(m-1)*ln(1-p) for support k = 1, 2, ...
    Expr m_minus_1 = Expr::sum({m, negate_expr(intg(1))});
    Expr ll = Expr::sum({Expr::product({n, ln(p)}),
                         Expr::product({n, m_minus_1, ln(one_minus(p))})});
    Expr mle = inv(m);  // p-hat = 1/m
    // i(p) = 1/(p^2 (1-p)) as p^(-2) (1-p)^(-1).
    Expr fisher = Expr::product({Expr::power(p, intg(-2)), inv(one_minus(p))});
    return make_model("p", std::move(ll), std::move(mle), std::move(fisher));
}

// --- exact rational point estimates ----------------------------------------

auto bernoulli_mle(std::span<const Rational> data) -> Result<Rational> {
    return mean(data);
}
auto poisson_mle(std::span<const Rational> data) -> Result<Rational> {
    return mean(data);
}
auto normal_mean_mle(std::span<const Rational> data) -> Result<Rational> {
    return mean(data);
}
auto exponential_mle(std::span<const Rational> data) -> Result<Rational> {
    return reciprocal_mean(data);
}
auto geometric_mle(std::span<const Rational> data) -> Result<Rational> {
    return reciprocal_mean(data);
}
auto normal_variance_mle(std::span<const Rational> data) -> Result<Rational> {
    // The MLE of the Normal variance is the POPULATION variance (divide by n).
    return variance(data, false);
}

// --- Wald / score statistics -----------------------------------------------

auto bernoulli_wald_statistic(std::span<const Rational> data, const Rational& p0)
    -> Result<Rational> {
    auto phat = mean(data);  // domain_error on empty data
    if (!phat) {
        return phat;
    }
    // Observed information factor p-hat(1 - p-hat); zero at p-hat in {0, 1}.
    auto one_minus_phat = Rational::from_int(1).subtract(*phat);
    if (!one_minus_phat) {
        return one_minus_phat;
    }
    auto info_den = phat->multiply(*one_minus_phat);
    if (!info_den) {
        return info_den;
    }
    if (info_den->is_zero()) {
        return make_error<Rational>(MathError::domain_error);
    }
    auto diff = phat->subtract(p0);
    if (!diff) {
        return diff;
    }
    auto dsq = square(*diff);
    if (!dsq) {
        return dsq;
    }
    auto num = ri(data.size()).multiply(*dsq);  // n (p-hat - p0)^2
    if (!num) {
        return num;
    }
    return num->divide(*info_den);
}

auto bernoulli_score_statistic(std::span<const Rational> data, const Rational& p0)
    -> Result<Rational> {
    if (data.empty()) {
        return make_error<Rational>(MathError::domain_error);
    }
    auto one_minus_p0 = Rational::from_int(1).subtract(p0);
    if (!one_minus_p0) {
        return one_minus_p0;
    }
    auto info_den = p0.multiply(*one_minus_p0);  // p0(1 - p0), zero at p0 in {0, 1}
    if (!info_den) {
        return info_den;
    }
    if (info_den->is_zero()) {
        return make_error<Rational>(MathError::domain_error);
    }
    auto xbar = mean(data);
    if (!xbar) {
        return xbar;
    }
    auto diff = xbar->subtract(p0);
    if (!diff) {
        return diff;
    }
    auto dsq = square(*diff);
    if (!dsq) {
        return dsq;
    }
    auto num = ri(data.size()).multiply(*dsq);  // n (xbar - p0)^2
    if (!num) {
        return num;
    }
    return num->divide(*info_den);
}

auto poisson_wald_statistic(std::span<const Rational> data, const Rational& lambda0)
    -> Result<Rational> {
    auto lhat = mean(data);  // domain_error on empty data
    if (!lhat) {
        return lhat;
    }
    if (lhat->is_zero()) {  // observed information n/lambda-hat undefined
        return make_error<Rational>(MathError::domain_error);
    }
    auto diff = lhat->subtract(lambda0);
    if (!diff) {
        return diff;
    }
    auto dsq = square(*diff);
    if (!dsq) {
        return dsq;
    }
    auto num = ri(data.size()).multiply(*dsq);  // n (lambda-hat - lambda0)^2
    if (!num) {
        return num;
    }
    return num->divide(*lhat);  // W = n (lambda-hat - lambda0)^2 / lambda-hat
}

auto poisson_score_statistic(std::span<const Rational> data, const Rational& lambda0)
    -> Result<Rational> {
    if (data.empty()) {
        return make_error<Rational>(MathError::domain_error);
    }
    if (lambda0.is_zero()) {  // expected information n/lambda0 undefined
        return make_error<Rational>(MathError::domain_error);
    }
    auto xbar = mean(data);
    if (!xbar) {
        return xbar;
    }
    auto diff = xbar->subtract(lambda0);
    if (!diff) {
        return diff;
    }
    auto dsq = square(*diff);
    if (!dsq) {
        return dsq;
    }
    auto num = ri(data.size()).multiply(*dsq);  // n (xbar - lambda0)^2
    if (!num) {
        return num;
    }
    return num->divide(lambda0);  // S = n (xbar - lambda0)^2 / lambda0
}

auto log_likelihood_ratio(const MleModel& model, const Expr& theta_hat, const Expr& theta0)
    -> Result<Expr> {
    const Expr theta = Expr::symbol(model.parameter);
    Expr ll_hat = substitute(model.log_likelihood, theta, theta_hat);
    Expr ll_0 = substitute(model.log_likelihood, theta, theta0);
    // G^2 = 2 ( ell(theta-hat) - ell(theta0) ), left symbolic (contains ln in general).
    Expr g = Expr::product({intg(2), Expr::sum({ll_hat, negate_expr(ll_0)})});
    return simplify(g);
}

}  // namespace nimblecas
