// NimbleCAS exact descriptive statistics over the rationals (ROADMAP 7.7).
// @author Olumuyiwa Oluwasanmi
//
// Descriptive statistics done *exactly*. Every datum is a Rational (a reduced int64
// fraction), so a mean, variance, or covariance is the fraction it mathematically is,
// never a double that happens to be close. The headline deliverable is the covariance
// matrix Sigma: for d variables measured over a common sample of length n, Sigma_{jk}
// is the covariance of variables j and k, a symmetric d x d matrix whose diagonal holds
// each variable's variance. It is returned as a nimblecas.matrix Matrix so it composes
// directly with the exact linear algebra (determinant, inverse, rank).
//
// Following the rest of the engine, arithmetic is exact and overflow-checked (Rule 32):
// every step flows through Rational's checked add/subtract/multiply/divide, so an int64
// numerator or denominator that would overflow surfaces as MathError::overflow rather
// than silently wrapping. Empty data, mismatched lengths, and too-few-points-for-a-
// sample-statistic surface as MathError::domain_error.
//
// Computation is a stable two-pass scheme: the mean(s) are formed first, then the
// (co)moment sum of exact deviations from those means. No floating point is ever used,
// so there is no catastrophic cancellation to guard against — the naive two-pass sum is
// already exact.

export module nimblecas.stats;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;

export namespace nimblecas {

// The arithmetic mean (1/n) * sum_i data_i. Empty data has no mean and fails with
// domain_error.
[[nodiscard]] auto mean(std::span<const Rational> data) -> Result<Rational>;

// The variance: (1/(n-1)) * sum_i (data_i - mean)^2 for the sample variance, or
// (1/n) * sum_i (data_i - mean)^2 for the population variance. The sample variance
// needs at least two points (n >= 2) since it divides by n - 1; the population
// variance needs at least one (n >= 1). A shortfall fails with domain_error.
[[nodiscard]] auto variance(std::span<const Rational> data, bool sample) -> Result<Rational>;

// The covariance (1/(n-1)) * sum_i (x_i - xbar)(y_i - ybar) for the sample covariance,
// or (1/n) * sum_i ... for the population covariance. x and y must have equal length,
// with the same n >= 2 (sample) / n >= 1 (population) constraint as variance; a length
// mismatch or a shortfall fails with domain_error.
[[nodiscard]] auto covariance(std::span<const Rational> x, std::span<const Rational> y,
                              bool sample) -> Result<Rational>;

// The symmetric d x d covariance matrix Sigma with Sigma_{jk} = covariance(variables[j],
// variables[k], sample). variables[j] is the sample vector of the j-th variable; every
// variable must share the common sample length n. An empty variable list, ragged
// variables (unequal lengths), or a sample length below the (co)variance constraint fails
// with domain_error. The diagonal Sigma_{jj} is the variance of variable j.
[[nodiscard]] auto covariance_matrix(const std::vector<std::span<const Rational>>& variables,
                                     bool sample) -> Result<Matrix>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// Sum a span of rationals exactly, propagating any overflow.
[[nodiscard]] auto sum(std::span<const Rational> data) -> Result<Rational> {
    Rational acc;  // 0/1
    for (const auto& v : data) {
        auto next = acc.add(v);
        if (!next) {
            return make_error<Rational>(next.error());
        }
        acc = *next;
    }
    return acc;
}

// The divisor for a (co)variance over n points: n - 1 for a sample statistic (requires
// n >= 2), n for a population statistic (requires n >= 1). Fails with domain_error when
// n is below the relevant threshold.
[[nodiscard]] auto degrees_of_freedom(std::size_t n, bool sample) -> Result<Rational> {
    if (sample) {
        if (n < 2) {
            return make_error<Rational>(MathError::domain_error);
        }
        return Rational::from_int(static_cast<std::int64_t>(n - 1));
    }
    if (n < 1) {
        return make_error<Rational>(MathError::domain_error);
    }
    return Rational::from_int(static_cast<std::int64_t>(n));
}

// The exact co-moment sum_i (x_i - xbar)(y_i - ybar), given precomputed means. The
// caller guarantees x and y have equal length.
[[nodiscard]] auto comoment(std::span<const Rational> x, std::span<const Rational> y,
                            const Rational& xbar, const Rational& ybar) -> Result<Rational> {
    Rational acc;  // 0/1
    for (std::size_t i = 0; i < x.size(); ++i) {
        auto dx = x[i].subtract(xbar);
        if (!dx) {
            return make_error<Rational>(dx.error());
        }
        auto dy = y[i].subtract(ybar);
        if (!dy) {
            return make_error<Rational>(dy.error());
        }
        auto prod = dx->multiply(*dy);
        if (!prod) {
            return make_error<Rational>(prod.error());
        }
        auto next = acc.add(*prod);
        if (!next) {
            return make_error<Rational>(next.error());
        }
        acc = *next;
    }
    return acc;
}

}  // namespace

auto mean(std::span<const Rational> data) -> Result<Rational> {
    if (data.empty()) {
        return make_error<Rational>(MathError::domain_error);
    }
    auto total = sum(data);
    if (!total) {
        return total;
    }
    return total->divide(Rational::from_int(static_cast<std::int64_t>(data.size())));
}

auto variance(std::span<const Rational> data, bool sample) -> Result<Rational> {
    // The dof check also rejects empty data (n == 0 is below both thresholds).
    auto dof = degrees_of_freedom(data.size(), sample);
    if (!dof) {
        return make_error<Rational>(dof.error());
    }
    auto m = mean(data);
    if (!m) {
        return m;
    }
    auto ss = comoment(data, data, *m, *m);  // sum of squared deviations
    if (!ss) {
        return ss;
    }
    return ss->divide(*dof);
}

auto covariance(std::span<const Rational> x, std::span<const Rational> y, bool sample)
    -> Result<Rational> {
    if (x.size() != y.size()) {
        return make_error<Rational>(MathError::domain_error);
    }
    auto dof = degrees_of_freedom(x.size(), sample);
    if (!dof) {
        return make_error<Rational>(dof.error());
    }
    auto xbar = mean(x);
    if (!xbar) {
        return xbar;
    }
    auto ybar = mean(y);
    if (!ybar) {
        return ybar;
    }
    auto cm = comoment(x, y, *xbar, *ybar);
    if (!cm) {
        return cm;
    }
    return cm->divide(*dof);
}

auto covariance_matrix(const std::vector<std::span<const Rational>>& variables, bool sample)
    -> Result<Matrix> {
    if (variables.empty()) {
        return make_error<Matrix>(MathError::domain_error);
    }
    // Every variable must share the common sample length n; a mismatch is ragged input.
    const std::size_t n = variables.front().size();
    for (const auto& v : variables) {
        if (v.size() != n) {
            return make_error<Matrix>(MathError::domain_error);
        }
    }
    // Reject a sample length below the (co)variance constraint up front (this also
    // rejects zero-length variables), so covariance() below never fails on dof.
    auto dof = degrees_of_freedom(n, sample);
    if (!dof) {
        return make_error<Matrix>(dof.error());
    }
    const std::size_t d = variables.size();
    // Precompute each variable's mean once (two-pass), then fill the symmetric matrix.
    std::vector<Rational> means(d);
    for (std::size_t j = 0; j < d; ++j) {
        auto m = mean(variables[j]);
        if (!m) {
            return make_error<Matrix>(m.error());
        }
        means[j] = *m;
    }
    std::vector<std::vector<Rational>> rows(d, std::vector<Rational>(d));
    for (std::size_t j = 0; j < d; ++j) {
        for (std::size_t k = j; k < d; ++k) {
            auto cm = comoment(variables[j], variables[k], means[j], means[k]);
            if (!cm) {
                return make_error<Matrix>(cm.error());
            }
            auto sigma = cm->divide(*dof);
            if (!sigma) {
                return make_error<Matrix>(sigma.error());
            }
            rows[j][k] = *sigma;
            rows[k][j] = *sigma;  // covariance is symmetric
        }
    }
    return Matrix::from_rows(std::move(rows));
}

}  // namespace nimblecas
