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

// The weighted mean (sum_i w_i x_i) / (sum_i w_i), exact. data and weights must share the
// same non-empty length; a length mismatch, empty input, or a zero total weight (sum_i w_i
// == 0, which would leave the mean undefined) fails with domain_error. Weights may be any
// rationals, including negatives, provided their total is non-zero.
[[nodiscard]] auto weighted_mean(std::span<const Rational> data,
                                 std::span<const Rational> weights) -> Result<Rational>;

// The k-th raw (about-zero) moment (1/n) * sum_i data_i^k, exact. k == 0 yields 1 (the
// count-normalised sum of ones). Empty data has no moment and fails with domain_error.
[[nodiscard]] auto raw_moment(std::span<const Rational> data, unsigned k) -> Result<Rational>;

// The k-th central moment (1/n) * sum_i (data_i - mean)^k, exact (population form, divisor
// n). m_0 == 1, m_1 == 0, and m_2 is the population variance. Empty data fails with
// domain_error.
[[nodiscard]] auto central_moment(std::span<const Rational> data, unsigned k)
    -> Result<Rational>;

// The exact squared skewness m_3^2 / m_2^3 (population moments), a RATIONAL. The textbook
// skewness m_3 / m_2^(3/2) is generally irrational (an odd root), so it is not returned;
// this exact square is, and the sign is recoverable exactly as sign(skewness) ==
// sign(central_moment(data, 3)). Requires m_2 != 0 (non-constant data); zero variance or
// empty data fails with domain_error.
[[nodiscard]] auto skewness_squared(std::span<const Rational> data) -> Result<Rational>;

// The exact excess kurtosis m_4 / m_2^2 - 3 (population moments), a RATIONAL (no root is
// involved, so unlike skewness this is returned directly). Requires m_2 != 0; zero
// variance or empty data fails with domain_error.
[[nodiscard]] auto excess_kurtosis(std::span<const Rational> data) -> Result<Rational>;

// The exact median: for odd n the middle order statistic, for even n the exact rational
// average of the two middle order statistics. Computed by sorting a copy of the data with
// an exact cross-multiplied rational comparison. Empty data fails with domain_error.
[[nodiscard]] auto median(std::span<const Rational> data) -> Result<Rational>;

// The exact type-7 (linear-interpolation) quantile at probability p in [0, 1]. With the
// data sorted ascending as x_0 <= ... <= x_{n-1}, let h = (n - 1) * p, lo = floor(h): the
// quantile is x_lo + (h - lo) * (x_{lo+1} - x_lo), exact over the rationals. p outside
// [0, 1], or empty data, fails with domain_error.
[[nodiscard]] auto quantile(std::span<const Rational> data, const Rational& p)
    -> Result<Rational>;

// The exact range max - min. Empty data fails with domain_error.
[[nodiscard]] auto range(std::span<const Rational> data) -> Result<Rational>;

// The exact interquartile range Q3 - Q1 == quantile(data, 3/4) - quantile(data, 1/4),
// using the type-7 quantile above. Empty data fails with domain_error.
[[nodiscard]] auto iqr(std::span<const Rational> data) -> Result<Rational>;

// The mode: the most frequently occurring value. Ties are broken toward the smallest
// value (documented, deterministic). Empty data fails with domain_error. See modes() for
// the full tied set.
[[nodiscard]] auto mode(std::span<const Rational> data) -> Result<Rational>;

// Every value attaining the maximum frequency, returned in ascending order (a single-
// element vector when the mode is unique). Empty data fails with domain_error.
[[nodiscard]] auto modes(std::span<const Rational> data) -> Result<std::vector<Rational>>;

// The exact squared Pearson correlation r^2 = cov(x, y)^2 / (var(x) * var(y)), a RATIONAL
// (the divisor n or n-1 cancels, so the flag is immaterial and none is taken). The
// correlation r itself is the square root of this and is generally irrational, so it is
// omitted; r^2 in [0, 1] is exact, equal to 1 for perfectly (anti-)correlated data. x and
// y must share a non-empty length; a mismatch, empty input, or a zero variance (r
// undefined) fails with domain_error.
[[nodiscard]] auto pearson_correlation_squared(std::span<const Rational> x,
                                               std::span<const Rational> y) -> Result<Rational>;

// The symmetric d x d matrix R^2 with R^2_{jk} = pearson_correlation_squared(variables[j],
// variables[k]); the diagonal is exactly 1. Same rational-square honesty as the scalar
// form (the signed r is omitted). An empty variable list, ragged variables, a sample
// length below the variance constraint, or any zero-variance variable fails with
// domain_error.
[[nodiscard]] auto correlation_squared_matrix(
    const std::vector<std::span<const Rational>>& variables) -> Result<Matrix>;

// The exact squared coefficient of variation var(data, sample) / mean(data)^2, a RATIONAL.
// The coefficient of variation itself (std / mean) is its square root and generally
// irrational, so it is omitted. Requires a non-zero mean; a zero mean (cv undefined) or a
// data set too small for the chosen variance fails with domain_error.
[[nodiscard]] auto coefficient_of_variation_squared(std::span<const Rational> data, bool sample)
    -> Result<Rational>;

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

// Exact ordering of two rationals by cross-multiplication: with both denominators positive
// (Rational's canonical form), a < b iff a.num * b.den < b.num * a.den. The two int64
// products are overflow-checked so a comparison that cannot be represented surfaces as
// MathError::overflow rather than wrapping.
[[nodiscard]] auto rat_less(const Rational& a, const Rational& b) -> Result<bool> {
    std::int64_t lhs = 0;  // a.num * b.den
    std::int64_t rhs = 0;  // b.num * a.den
    if (__builtin_mul_overflow(a.numerator(), b.denominator(), &lhs) ||
        __builtin_mul_overflow(b.numerator(), a.denominator(), &rhs)) {
        return make_error<bool>(MathError::overflow);
    }
    return lhs < rhs;
}

// A copy of the data sorted ascending under rat_less. Insertion sort keeps the code simple
// and, crucially, propagates an overflow from any single comparison instead of swallowing
// it (a fallible comparator cannot be handed to std::sort). Order statistics (median,
// quantile, range, mode) all build on this.
[[nodiscard]] auto sorted_copy(std::span<const Rational> data) -> Result<std::vector<Rational>> {
    std::vector<Rational> v(data.begin(), data.end());
    for (std::size_t i = 1; i < v.size(); ++i) {
        const Rational key = v[i];
        std::size_t j = i;
        while (j > 0) {
            auto lt = rat_less(key, v[j - 1]);
            if (!lt) {
                return make_error<std::vector<Rational>>(lt.error());
            }
            if (!*lt) {
                break;
            }
            v[j] = v[j - 1];
            --j;
        }
        v[j] = key;
    }
    return v;
}

// Exact integer power base^k by repeated squaring, k == 0 -> 1. Every multiply is checked,
// so an overflowing power surfaces as MathError::overflow.
[[nodiscard]] auto rat_pow(const Rational& base, unsigned k) -> Result<Rational> {
    Rational result = Rational::from_int(1);
    Rational b = base;
    for (unsigned e = k; e > 0; e >>= 1U) {
        if ((e & 1U) != 0U) {
            auto r = result.multiply(b);
            if (!r) {
                return make_error<Rational>(r.error());
            }
            result = *r;
        }
        if (e > 1U) {
            auto s = b.multiply(b);
            if (!s) {
                return make_error<Rational>(s.error());
            }
            b = *s;
        }
    }
    return result;
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

auto weighted_mean(std::span<const Rational> data, std::span<const Rational> weights)
    -> Result<Rational> {
    if (data.empty() || data.size() != weights.size()) {
        return make_error<Rational>(MathError::domain_error);
    }
    Rational num;  // sum_i w_i x_i, 0/1
    Rational den;  // sum_i w_i, 0/1
    for (std::size_t i = 0; i < data.size(); ++i) {
        auto term = data[i].multiply(weights[i]);
        if (!term) {
            return make_error<Rational>(term.error());
        }
        auto nn = num.add(*term);
        if (!nn) {
            return make_error<Rational>(nn.error());
        }
        num = *nn;
        auto dd = den.add(weights[i]);
        if (!dd) {
            return make_error<Rational>(dd.error());
        }
        den = *dd;
    }
    // A zero total weight leaves the weighted mean undefined; report it as a domain error
    // rather than leaking a division_by_zero from the field arithmetic.
    if (den.is_zero()) {
        return make_error<Rational>(MathError::domain_error);
    }
    return num.divide(den);
}

auto raw_moment(std::span<const Rational> data, unsigned k) -> Result<Rational> {
    if (data.empty()) {
        return make_error<Rational>(MathError::domain_error);
    }
    Rational acc;  // 0/1
    for (const auto& x : data) {
        auto p = rat_pow(x, k);
        if (!p) {
            return make_error<Rational>(p.error());
        }
        auto next = acc.add(*p);
        if (!next) {
            return make_error<Rational>(next.error());
        }
        acc = *next;
    }
    return acc.divide(Rational::from_int(static_cast<std::int64_t>(data.size())));
}

auto central_moment(std::span<const Rational> data, unsigned k) -> Result<Rational> {
    if (data.empty()) {
        return make_error<Rational>(MathError::domain_error);
    }
    auto m = mean(data);
    if (!m) {
        return m;
    }
    Rational acc;  // 0/1
    for (const auto& x : data) {
        auto dx = x.subtract(*m);
        if (!dx) {
            return make_error<Rational>(dx.error());
        }
        auto p = rat_pow(*dx, k);
        if (!p) {
            return make_error<Rational>(p.error());
        }
        auto next = acc.add(*p);
        if (!next) {
            return make_error<Rational>(next.error());
        }
        acc = *next;
    }
    return acc.divide(Rational::from_int(static_cast<std::int64_t>(data.size())));
}

auto skewness_squared(std::span<const Rational> data) -> Result<Rational> {
    auto m2 = central_moment(data, 2);
    if (!m2) {
        return m2;
    }
    // Zero variance leaves the (standardised) skewness undefined.
    if (m2->is_zero()) {
        return make_error<Rational>(MathError::domain_error);
    }
    auto m3 = central_moment(data, 3);
    if (!m3) {
        return m3;
    }
    auto m3sq = m3->multiply(*m3);
    if (!m3sq) {
        return m3sq;
    }
    auto m2cubed = rat_pow(*m2, 3);
    if (!m2cubed) {
        return m2cubed;
    }
    return m3sq->divide(*m2cubed);  // m_3^2 / m_2^3
}

auto excess_kurtosis(std::span<const Rational> data) -> Result<Rational> {
    auto m2 = central_moment(data, 2);
    if (!m2) {
        return m2;
    }
    if (m2->is_zero()) {
        return make_error<Rational>(MathError::domain_error);
    }
    auto m4 = central_moment(data, 4);
    if (!m4) {
        return m4;
    }
    auto m2sq = m2->multiply(*m2);
    if (!m2sq) {
        return m2sq;
    }
    auto ratio = m4->divide(*m2sq);  // m_4 / m_2^2 (the raw kurtosis)
    if (!ratio) {
        return ratio;
    }
    return ratio->subtract(Rational::from_int(3));  // excess kurtosis
}

auto median(std::span<const Rational> data) -> Result<Rational> {
    if (data.empty()) {
        return make_error<Rational>(MathError::domain_error);
    }
    auto sorted = sorted_copy(data);
    if (!sorted) {
        return make_error<Rational>(sorted.error());
    }
    const std::size_t n = sorted->size();
    if ((n & 1U) != 0U) {
        return (*sorted)[n / 2];  // odd: the middle order statistic
    }
    // even: the exact rational average of the two middle order statistics.
    auto s = (*sorted)[n / 2 - 1].add((*sorted)[n / 2]);
    if (!s) {
        return s;
    }
    return s->divide(Rational::from_int(2));
}

auto quantile(std::span<const Rational> data, const Rational& p) -> Result<Rational> {
    if (data.empty()) {
        return make_error<Rational>(MathError::domain_error);
    }
    // Validate p in [0, 1] with exact comparisons.
    const Rational zero;  // 0/1
    const Rational one = Rational::from_int(1);
    auto below = rat_less(p, zero);
    if (!below) {
        return make_error<Rational>(below.error());
    }
    auto above = rat_less(one, p);
    if (!above) {
        return make_error<Rational>(above.error());
    }
    if (*below || *above) {
        return make_error<Rational>(MathError::domain_error);
    }
    auto sorted = sorted_copy(data);
    if (!sorted) {
        return make_error<Rational>(sorted.error());
    }
    const std::size_t n = sorted->size();
    if (n == 1) {
        return (*sorted)[0];  // a single point is its own every quantile
    }
    // Type-7: h = (n - 1) * p, lo = floor(h); interpolate between x_lo and x_{lo+1}.
    auto h = Rational::from_int(static_cast<std::int64_t>(n - 1)).multiply(p);
    if (!h) {
        return h;
    }
    // h >= 0 with a positive denominator, so truncating the numerator is exactly floor(h).
    const std::int64_t lo = h->numerator() / h->denominator();
    const auto lo_idx = static_cast<std::size_t>(lo);
    if (lo_idx >= n - 1) {
        return (*sorted)[n - 1];  // h == n - 1 (p == 1): the maximum
    }
    auto frac = h->subtract(Rational::from_int(lo));  // h - floor(h) in [0, 1)
    if (!frac) {
        return frac;
    }
    auto gap = (*sorted)[lo_idx + 1].subtract((*sorted)[lo_idx]);
    if (!gap) {
        return gap;
    }
    auto step = frac->multiply(*gap);
    if (!step) {
        return step;
    }
    return (*sorted)[lo_idx].add(*step);
}

auto range(std::span<const Rational> data) -> Result<Rational> {
    if (data.empty()) {
        return make_error<Rational>(MathError::domain_error);
    }
    auto sorted = sorted_copy(data);
    if (!sorted) {
        return make_error<Rational>(sorted.error());
    }
    return sorted->back().subtract(sorted->front());  // max - min
}

auto iqr(std::span<const Rational> data) -> Result<Rational> {
    auto q1 = quantile(data, Rational::make(1, 4).value());
    if (!q1) {
        return q1;
    }
    auto q3 = quantile(data, Rational::make(3, 4).value());
    if (!q3) {
        return q3;
    }
    return q3->subtract(*q1);
}

auto modes(std::span<const Rational> data) -> Result<std::vector<Rational>> {
    if (data.empty()) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    auto sorted = sorted_copy(data);
    if (!sorted) {
        return sorted;
    }
    // Equal values are adjacent after sorting, so a single linear pass counts each run.
    std::size_t best = 0;  // largest run length seen
    std::vector<Rational> out;
    std::size_t i = 0;
    const std::size_t n = sorted->size();
    while (i < n) {
        std::size_t j = i + 1;
        while (j < n && (*sorted)[j] == (*sorted)[i]) {
            ++j;
        }
        const std::size_t count = j - i;
        if (count > best) {
            best = count;
            out.clear();
            out.push_back((*sorted)[i]);
        } else if (count == best) {
            out.push_back((*sorted)[i]);  // tie: keep, still ascending
        }
        i = j;
    }
    return out;
}

auto mode(std::span<const Rational> data) -> Result<Rational> {
    auto all = modes(data);
    if (!all) {
        return make_error<Rational>(all.error());
    }
    // modes() returns the tied set ascending, so the first element is the smallest-valued
    // most-frequent datum — the documented tie-break.
    return all->front();
}

auto pearson_correlation_squared(std::span<const Rational> x, std::span<const Rational> y)
    -> Result<Rational> {
    if (x.empty() || x.size() != y.size()) {
        return make_error<Rational>(MathError::domain_error);
    }
    // The 1/n (or 1/(n-1)) divisor cancels in cov^2 / (var*var); use the population form so
    // a single point is still admissible (its zero variance is caught below as domain_error).
    auto cov = covariance(x, y, false);
    if (!cov) {
        return cov;
    }
    auto vx = variance(x, false);
    if (!vx) {
        return vx;
    }
    auto vy = variance(y, false);
    if (!vy) {
        return vy;
    }
    auto denom = vx->multiply(*vy);
    if (!denom) {
        return denom;
    }
    // A zero-variance variable leaves the correlation undefined.
    if (denom->is_zero()) {
        return make_error<Rational>(MathError::domain_error);
    }
    auto numer = cov->multiply(*cov);
    if (!numer) {
        return numer;
    }
    return numer->divide(*denom);  // r^2 = cov^2 / (var_x * var_y)
}

auto correlation_squared_matrix(const std::vector<std::span<const Rational>>& variables)
    -> Result<Matrix> {
    if (variables.empty()) {
        return make_error<Matrix>(MathError::domain_error);
    }
    const std::size_t n = variables.front().size();
    for (const auto& v : variables) {
        if (v.size() != n) {
            return make_error<Matrix>(MathError::domain_error);
        }
    }
    const std::size_t d = variables.size();
    std::vector<std::vector<Rational>> rows(d, std::vector<Rational>(d));
    for (std::size_t j = 0; j < d; ++j) {
        for (std::size_t k = j; k < d; ++k) {
            auto r2 = pearson_correlation_squared(variables[j], variables[k]);
            if (!r2) {
                return make_error<Matrix>(r2.error());
            }
            rows[j][k] = *r2;
            rows[k][j] = *r2;  // r^2 is symmetric
        }
    }
    return Matrix::from_rows(std::move(rows));
}

auto coefficient_of_variation_squared(std::span<const Rational> data, bool sample)
    -> Result<Rational> {
    auto m = mean(data);
    if (!m) {
        return m;
    }
    // A zero mean leaves the coefficient of variation undefined.
    if (m->is_zero()) {
        return make_error<Rational>(MathError::domain_error);
    }
    auto v = variance(data, sample);
    if (!v) {
        return v;
    }
    auto msq = m->multiply(*m);
    if (!msq) {
        return msq;
    }
    return v->divide(*msq);  // (cv)^2 = var / mean^2
}

}  // namespace nimblecas
