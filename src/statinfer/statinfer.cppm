// NimbleCAS exact inferential statistics & regression over the rationals (ROADMAP 7.7.5).
// @author Olumuyiwa Oluwasanmi
//
// Linear regression done *exactly*. Given a design matrix X (m observations x n features,
// the intercept being a leading column of ones) and a response y of length m, the
// ordinary-least-squares coefficient vector beta is the solution of the normal equations
//
//     (X^T X) beta = X^T y.
//
// Every entry of X and y is a Rational (a reduced int64 fraction), and the normal system
// is solved over the field Q by exact Gauss-Jordan elimination (nimblecas.matrix), so
// beta is the fraction it mathematically *is* — never a double that happens to be close.
// Ridge regression adds a rational penalty ((X^T X + lambda I) beta = X^T y), weighted
// least squares a diagonal weight matrix ((X^T W X) beta = X^T W y). The coefficient of
// determination R^2 = 1 - SS_res / SS_tot is likewise an exact rational.
//
// Honesty boundary (Rule 32; the convention shared with nimblecas.stats). Everything that
// is rational is returned exactly; nothing that would require an irrational is faked.
// A coefficient standard error is sqrt(diag(Cov(beta))), and a square root of a rational
// is in general irrational — so no standard errors, t-statistics, or p-values are offered.
// Instead coefficient_covariance returns the *exact rational* covariance matrix of beta,
// sigma^2 (X^T X)^{-1}, from which a caller may take square roots at whatever precision it
// chooses. A singular X^T X (rank-deficient design) surfaces as MathError::domain_error —
// an honest failure, never a bogus beta.
//
// Following the rest of the engine, arithmetic is exact and overflow-checked: any int64
// numerator or denominator that would overflow surfaces as MathError::overflow rather than
// silently wrapping. Dimension mismatches, a negative ridge penalty or weight, and a
// degenerate goodness-of-fit (constant response, SS_tot = 0) surface as domain_error.

export module nimblecas.statinfer;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.stats;

export namespace nimblecas {

// Ordinary least squares. X is m x n (m observations in rows, n features in columns; the
// intercept, if wanted, is a leading all-ones column supplied by the caller). y is the
// length-m response. Returns the length-n coefficient vector beta solving the normal
// equations (X^T X) beta = X^T y exactly over Q. A response whose length does not match
// X.rows(), or a rank-deficient design (singular X^T X, which includes every m < n case),
// fails with domain_error.
[[nodiscard]] auto ols(const Matrix& X, std::span<const Rational> y)
    -> Result<std::vector<Rational>>;

// Ridge (Tikhonov) regression: solve (X^T X + lambda I) beta = X^T y for a rational
// penalty lambda >= 0. lambda = 0 reduces to ols. A negative lambda, or a length
// mismatch, fails with domain_error; the penalised system is singular only in degenerate
// cases (e.g. lambda = 0 with a rank-deficient X), which also surface as domain_error.
[[nodiscard]] auto ridge(const Matrix& X, std::span<const Rational> y, const Rational& lambda)
    -> Result<std::vector<Rational>>;

// Weighted least squares: solve (X^T W X) beta = X^T W y for the diagonal weight matrix
// W = diag(weights). weights has length m and every weight must be >= 0 (a zero weight
// drops that observation). A length mismatch, a negative weight, or a singular X^T W X
// fails with domain_error.
[[nodiscard]] auto weighted_ols(const Matrix& X, std::span<const Rational> y,
                                std::span<const Rational> weights)
    -> Result<std::vector<Rational>>;

// Fitted values yhat = X beta (length m). beta must have length X.cols(), else
// domain_error.
[[nodiscard]] auto predict(const Matrix& X, std::span<const Rational> beta)
    -> Result<std::vector<Rational>>;

// Coefficient of determination R^2 = 1 - SS_res / SS_tot, both exact rationals, where
// SS_res = sum_i (y_i - yhat_i)^2 with yhat = X beta, and SS_tot = sum_i (y_i - ybar)^2.
// A perfect fit gives R^2 = 1; a mean-only model gives R^2 = 0. A constant response
// (SS_tot = 0) leaves R^2 undefined and fails with domain_error, as does a length
// mismatch (y vs X.rows(), or beta vs X.cols()).
[[nodiscard]] auto r_squared(const Matrix& X, std::span<const Rational> y,
                             std::span<const Rational> beta) -> Result<Rational>;

// The exact rational covariance matrix of the OLS estimator, Cov(beta) = sigma^2 (X^T X)^{-1},
// with sigma^2 = SS_res / (m - n) the unbiased residual variance (n = X.cols()). This is
// the honest rational deliverable in place of standard errors: the standard error of beta_j
// is sqrt(Cov(beta)_{jj}), whose square root is omitted because it is in general irrational
// (see the module honesty boundary). Requires m > n (a positive residual degrees of freedom)
// and a nonsingular X^T X; otherwise domain_error. A length mismatch also fails domain_error.
[[nodiscard]] auto coefficient_covariance(const Matrix& X, std::span<const Rational> y)
    -> Result<Matrix>;

// Method of moments for a one-parameter family whose population mean is a linear-fractional
// (Mobius) function of the parameter theta:
//
//     E[X](theta) = (a*theta + b) / (c*theta + d).
//
// Given the sample mean mbar, this sets mbar = E[X](theta) and solves the resulting linear
// equation exactly: theta = (b - mbar*d) / (mbar*c - a). This modest supported form covers
// the common exact cases — Bernoulli/Poisson (a=1,b=0,c=0,d=1 => theta = mbar), Binomial(N,p)
// with known N (a=N,b=0,c=0,d=1 => p = mbar/N), Uniform(0,theta) (a=1/2,b=0,c=0,d=1 =>
// theta = 2*mbar), Exponential rate (a=0,b=1,c=1,d=0 => theta = 1/mbar). A general method-of-
// moments engine is out of scope. Fails with domain_error when mbar*c - a = 0 (no unique
// theta) or when the recovered theta makes the denominator c*theta + d vanish (E[X]
// undefined there).
[[nodiscard]] auto method_of_moments(const Rational& sample_mean, const Rational& a,
                                     const Rational& b, const Rational& c, const Rational& d)
    -> Result<Rational>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// A single column matrix (v.size() x 1) holding the span's entries in order. An empty span
// yields the 0x0 matrix (from_rows of an empty list); callers guard lengths beforehand.
[[nodiscard]] auto to_column(std::span<const Rational> v) -> Result<Matrix> {
    std::vector<std::vector<Rational>> rows;
    rows.reserve(v.size());
    for (const auto& e : v) {
        rows.push_back(std::vector<Rational>{e});
    }
    return Matrix::from_rows(std::move(rows));
}

// The first column of a matrix as a flat vector (caller guarantees at least one column).
[[nodiscard]] auto column_to_vector(const Matrix& col) -> std::vector<Rational> {
    std::vector<Rational> out;
    out.reserve(col.rows());
    for (std::size_t i = 0; i < col.rows(); ++i) {
        out.push_back(col.at(i, 0));
    }
    return out;
}

// diag(weights) * X: X with row i scaled by weights[i]. Caller guarantees weights.size()
// == X.rows(). Propagates any overflow from the entry products.
[[nodiscard]] auto scale_rows(const Matrix& X, std::span<const Rational> weights)
    -> Result<Matrix> {
    std::vector<std::vector<Rational>> rows(X.rows(), std::vector<Rational>(X.cols()));
    for (std::size_t i = 0; i < X.rows(); ++i) {
        for (std::size_t j = 0; j < X.cols(); ++j) {
            auto prod = weights[i].multiply(X.at(i, j));
            if (!prod) {
                return make_error<Matrix>(prod.error());
            }
            rows[i][j] = *prod;
        }
    }
    return Matrix::from_rows(std::move(rows));
}

// Solve the n x n normal system L beta = r (r is n x 1) and return beta as a vector.
// A singular L surfaces as domain_error via Matrix::solve.
[[nodiscard]] auto solve_normal(const Matrix& L, const Matrix& r)
    -> Result<std::vector<Rational>> {
    auto beta_col = L.solve(r);
    if (!beta_col) {
        return make_error<std::vector<Rational>>(beta_col.error());
    }
    return column_to_vector(*beta_col);
}

// sum_i (a_i - b_i)^2 over equal-length spans (caller guarantees equal length).
[[nodiscard]] auto sum_sq_diff(std::span<const Rational> a, std::span<const Rational> b)
    -> Result<Rational> {
    Rational acc;  // 0/1
    for (std::size_t i = 0; i < a.size(); ++i) {
        auto d = a[i].subtract(b[i]);
        if (!d) {
            return make_error<Rational>(d.error());
        }
        auto sq = d->multiply(*d);
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

// sum_i (a_i - c)^2 for a scalar c (the deviation sum used by SS_tot).
[[nodiscard]] auto sum_sq_dev(std::span<const Rational> a, const Rational& c) -> Result<Rational> {
    Rational acc;  // 0/1
    for (const auto& e : a) {
        auto d = e.subtract(c);
        if (!d) {
            return make_error<Rational>(d.error());
        }
        auto sq = d->multiply(*d);
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

// The n x n Gram matrix X^T X and the n x 1 right-hand side X^T y, shared by ols/ridge.
struct Normal {
    Matrix gram;  // X^T X
    Matrix rhs;   // X^T y
};

[[nodiscard]] auto normal_equations(const Matrix& X, std::span<const Rational> y)
    -> Result<Normal> {
    auto Xt = X.transpose();  // never fails, but returns Result for a uniform surface
    if (!Xt) {
        return make_error<Normal>(Xt.error());
    }
    auto gram = Xt->multiply(X);
    if (!gram) {
        return make_error<Normal>(gram.error());
    }
    auto ycol = to_column(y);
    if (!ycol) {
        return make_error<Normal>(ycol.error());
    }
    auto rhs = Xt->multiply(*ycol);
    if (!rhs) {
        return make_error<Normal>(rhs.error());
    }
    return Normal{std::move(*gram), std::move(*rhs)};
}

}  // namespace

auto ols(const Matrix& X, std::span<const Rational> y) -> Result<std::vector<Rational>> {
    if (y.size() != X.rows()) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    auto ne = normal_equations(X, y);
    if (!ne) {
        return make_error<std::vector<Rational>>(ne.error());
    }
    return solve_normal(ne->gram, ne->rhs);
}

auto ridge(const Matrix& X, std::span<const Rational> y, const Rational& lambda)
    -> Result<std::vector<Rational>> {
    if (y.size() != X.rows()) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    if (lambda.numerator() < 0) {  // denominator is always positive in canonical form
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    auto ne = normal_equations(X, y);
    if (!ne) {
        return make_error<std::vector<Rational>>(ne.error());
    }
    // Penalised Gram matrix X^T X + lambda I.
    auto lam_i = Matrix::identity(X.cols()).scale(lambda);
    if (!lam_i) {
        return make_error<std::vector<Rational>>(lam_i.error());
    }
    auto penalised = ne->gram.add(*lam_i);
    if (!penalised) {
        return make_error<std::vector<Rational>>(penalised.error());
    }
    return solve_normal(*penalised, ne->rhs);
}

auto weighted_ols(const Matrix& X, std::span<const Rational> y, std::span<const Rational> weights)
    -> Result<std::vector<Rational>> {
    if (y.size() != X.rows() || weights.size() != X.rows()) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    for (const auto& w : weights) {
        if (w.numerator() < 0) {  // negative weights are not a valid weighted-LS problem
            return make_error<std::vector<Rational>>(MathError::domain_error);
        }
    }
    auto Xt = X.transpose();
    if (!Xt) {
        return make_error<std::vector<Rational>>(Xt.error());
    }
    // W X (rows scaled), then X^T (W X) = X^T W X since W is diagonal (hence symmetric).
    auto wx = scale_rows(X, weights);
    if (!wx) {
        return make_error<std::vector<Rational>>(wx.error());
    }
    auto xtwx = Xt->multiply(*wx);
    if (!xtwx) {
        return make_error<std::vector<Rational>>(xtwx.error());
    }
    // W y (each response scaled), then X^T (W y) = X^T W y.
    std::vector<Rational> wy(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        auto prod = weights[i].multiply(y[i]);
        if (!prod) {
            return make_error<std::vector<Rational>>(prod.error());
        }
        wy[i] = *prod;
    }
    auto wy_col = to_column(std::span<const Rational>{wy});
    if (!wy_col) {
        return make_error<std::vector<Rational>>(wy_col.error());
    }
    auto xtwy = Xt->multiply(*wy_col);
    if (!xtwy) {
        return make_error<std::vector<Rational>>(xtwy.error());
    }
    return solve_normal(*xtwx, *xtwy);
}

auto predict(const Matrix& X, std::span<const Rational> beta) -> Result<std::vector<Rational>> {
    if (beta.size() != X.cols()) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    auto bcol = to_column(beta);
    if (!bcol) {
        return make_error<std::vector<Rational>>(bcol.error());
    }
    auto yhat = X.multiply(*bcol);
    if (!yhat) {
        return make_error<std::vector<Rational>>(yhat.error());
    }
    return column_to_vector(*yhat);
}

auto r_squared(const Matrix& X, std::span<const Rational> y, std::span<const Rational> beta)
    -> Result<Rational> {
    if (y.size() != X.rows()) {
        return make_error<Rational>(MathError::domain_error);
    }
    auto yhat = predict(X, beta);  // also validates beta.size() == X.cols()
    if (!yhat) {
        return make_error<Rational>(yhat.error());
    }
    auto ss_res = sum_sq_diff(y, std::span<const Rational>{*yhat});
    if (!ss_res) {
        return make_error<Rational>(ss_res.error());
    }
    auto ybar = mean(y);  // nimblecas.stats; domain_error on empty y
    if (!ybar) {
        return make_error<Rational>(ybar.error());
    }
    auto ss_tot = sum_sq_dev(y, *ybar);
    if (!ss_tot) {
        return make_error<Rational>(ss_tot.error());
    }
    if (ss_tot->is_zero()) {  // constant response: R^2 is undefined (0/0)
        return make_error<Rational>(MathError::domain_error);
    }
    auto ratio = ss_res->divide(*ss_tot);
    if (!ratio) {
        return make_error<Rational>(ratio.error());
    }
    return Rational::from_int(1).subtract(*ratio);
}

auto coefficient_covariance(const Matrix& X, std::span<const Rational> y) -> Result<Matrix> {
    if (y.size() != X.rows()) {
        return make_error<Matrix>(MathError::domain_error);
    }
    const std::size_t m = X.rows();
    const std::size_t n = X.cols();
    if (m <= n) {  // need a positive residual degrees of freedom m - n
        return make_error<Matrix>(MathError::domain_error);
    }
    auto beta = ols(X, y);  // singular X^T X surfaces here as domain_error
    if (!beta) {
        return make_error<Matrix>(beta.error());
    }
    auto yhat = predict(X, std::span<const Rational>{*beta});
    if (!yhat) {
        return make_error<Matrix>(yhat.error());
    }
    auto ss_res = sum_sq_diff(y, std::span<const Rational>{*yhat});
    if (!ss_res) {
        return make_error<Matrix>(ss_res.error());
    }
    // Unbiased residual variance sigma^2 = SS_res / (m - n).
    auto sigma2 = ss_res->divide(Rational::from_int(static_cast<std::int64_t>(m - n)));
    if (!sigma2) {
        return make_error<Matrix>(sigma2.error());
    }
    auto Xt = X.transpose();
    if (!Xt) {
        return make_error<Matrix>(Xt.error());
    }
    auto gram = Xt->multiply(X);
    if (!gram) {
        return make_error<Matrix>(gram.error());
    }
    auto inv = gram->inverse();  // singular => domain_error (never a bogus covariance)
    if (!inv) {
        return make_error<Matrix>(inv.error());
    }
    return inv->scale(*sigma2);
}

auto method_of_moments(const Rational& sample_mean, const Rational& a, const Rational& b,
                       const Rational& c, const Rational& d) -> Result<Rational> {
    // Solve mbar = (a*theta + b)/(c*theta + d) => theta*(mbar*c - a) = b - mbar*d.
    auto mc = sample_mean.multiply(c);
    if (!mc) {
        return make_error<Rational>(mc.error());
    }
    auto denom = mc->subtract(a);  // mbar*c - a
    if (!denom) {
        return make_error<Rational>(denom.error());
    }
    if (denom->is_zero()) {  // no unique theta
        return make_error<Rational>(MathError::domain_error);
    }
    auto md = sample_mean.multiply(d);
    if (!md) {
        return make_error<Rational>(md.error());
    }
    auto numer = b.subtract(*md);  // b - mbar*d
    if (!numer) {
        return make_error<Rational>(numer.error());
    }
    auto theta = numer->divide(*denom);
    if (!theta) {
        return make_error<Rational>(theta.error());
    }
    // Guard the model itself: E[X](theta) requires c*theta + d != 0.
    auto ct = c.multiply(*theta);
    if (!ct) {
        return make_error<Rational>(ct.error());
    }
    auto model_denom = ct->add(d);
    if (!model_denom) {
        return make_error<Rational>(model_denom.error());
    }
    if (model_denom->is_zero()) {
        return make_error<Rational>(MathError::domain_error);
    }
    return *theta;
}

}  // namespace nimblecas
