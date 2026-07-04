// NimbleCAS big-backed truncated power series Q[[x]]/(x^N) over BigRational.
// @author Olumuyiwa Oluwasanmi
//
// A BigPowerSeries is an element of Q[[x]] / (x^N): it retains a fixed number of
// coefficients `order = N`, representing the terms c_0 + c_1 x + ... + c_{N-1} x^{N-1}
// with everything of degree >= N discarded (an implicit O(x^N) tail). The coefficient
// vector always holds exactly N BigRationals. Every binary operation requires the two
// operands to share the same order (otherwise MathError::domain_error); this keeps the
// truncation ring unambiguous.
//
// This is the EXACT, UNBOUNDED counterpart of nimblecas.powerseries. That module carries
// its coefficients as the int64-backed Rational and therefore reports MathError::overflow
// once a numerator or denominator no longer fits int64 — which happens quickly for the
// transcendental series (e.g. the 1/n! coefficients of exp overflow the int64 denominator
// at n >= 21). BigPowerSeries removes that ceiling: coefficients are BigRational over the
// arbitrary-precision BigInt, so NOTHING overflows and the whole truncated series is exact
// over Q. Because BigRational add/subtract/multiply/negate are INFALLIBLE (arbitrary
// precision cannot overflow), the coefficient combining here never needs an overflow
// railway; only divide/inverse/exp/log (which can divide by zero) and the domain guards
// propagate a MathError (Rule 32).
//
// HONESTY: this is the slow-but-exact tier. Each coefficient op heap-allocates BigInt
// limbs, so it is materially slower than the int64 powerseries. Small, low-order work that
// comfortably fits int64 should use nimblecas.powerseries (faster, but bounded — it returns
// MathError::overflow at high order); reach for BigPowerSeries precisely when the exact
// unbounded range is required.

export module nimblecas.bigpowerseries;

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;

export namespace nimblecas {

// Truncated formal power series over Q with exact unbounded BigRational coefficients,
// working modulo x^N where N == order().
class BigPowerSeries {
public:
    // --- Factories -------------------------------------------------------------
    // Build from an explicit coefficient list, padded with zeros or truncated so the
    // result holds exactly `order` coefficients (c_0..c_{order-1}). order == 0 is a
    // domain_error.
    [[nodiscard]] static auto from_coeffs(std::vector<BigRational> coeffs, std::size_t order)
        -> Result<BigPowerSeries>;
    // The constant series c + O(x^N). order == 0 is a domain_error.
    [[nodiscard]] static auto constant(const BigRational& c, std::size_t order)
        -> Result<BigPowerSeries>;
    // The series x (c_1 = 1). For order == 1 this truncates to 0; order == 0 is a
    // domain_error.
    [[nodiscard]] static auto variable(std::size_t order) -> Result<BigPowerSeries>;
    // The additive identity 0 + O(x^N). order == 0 is a domain_error.
    [[nodiscard]] static auto zero(std::size_t order) -> Result<BigPowerSeries>;
    // The multiplicative identity 1 + O(x^N). order == 0 is a domain_error.
    [[nodiscard]] static auto one(std::size_t order) -> Result<BigPowerSeries>;

    // --- Accessors -------------------------------------------------------------
    // Number of retained coefficients N (terms x^0..x^{N-1}).
    [[nodiscard]] auto order() const noexcept -> std::size_t { return coeffs_.size(); }
    // Coefficient of x^k; 0 for k >= order().
    [[nodiscard]] auto coefficient(std::size_t k) const -> BigRational;
    // Read-only view of all N coefficients (index i is the coefficient of x^i).
    [[nodiscard]] auto coefficients() const noexcept -> std::span<const BigRational> {
        return coeffs_;
    }
    // Human-readable form, e.g. "1 + 2*x + x^2 + O(x^3)".
    [[nodiscard]] auto to_string(std::string_view var = "x") const -> std::string;
    // Same order and identical coefficients.
    [[nodiscard]] auto is_equal(const BigPowerSeries& o) const -> bool;
    [[nodiscard]] auto operator==(const BigPowerSeries& o) const -> bool { return is_equal(o); }

    // --- Arithmetic ------------------------------------------------------------
    // Coefficient-wise sum / difference (requires equal order).
    [[nodiscard]] auto add(const BigPowerSeries& o) const -> Result<BigPowerSeries>;
    [[nodiscard]] auto subtract(const BigPowerSeries& o) const -> Result<BigPowerSeries>;
    // Multiply every coefficient by the scalar s.
    [[nodiscard]] auto scale(const BigRational& s) const -> Result<BigPowerSeries>;
    // Truncated Cauchy product: (a*b)_k = sum_{i+j=k} a_i b_j for k < N (requires equal
    // order).
    [[nodiscard]] auto multiply(const BigPowerSeries& o) const -> Result<BigPowerSeries>;
    // Multiplicative inverse. Requires c_0 != 0 (else domain_error). Uses the standard
    // recurrence b_0 = 1/a_0, b_k = -(1/a_0) sum_{i=1..k} a_i b_{k-i}.
    [[nodiscard]] auto inverse() const -> Result<BigPowerSeries>;
    // Division = multiply by the inverse of o (requires equal order; o_0 != 0).
    [[nodiscard]] auto divide(const BigPowerSeries& o) const -> Result<BigPowerSeries>;

    // --- Calculus --------------------------------------------------------------
    // Formal derivative. Order convention: the result KEEPS the same order N, with the
    // top coefficient set to 0 (since the x^{N-1} term would come from the truncated
    // x^N coefficient). result_k = (k+1) c_{k+1} for k < N-1, result_{N-1} = 0.
    [[nodiscard]] auto derivative() const -> Result<BigPowerSeries>;
    // Formal integral with zero constant of integration. result_0 = 0,
    // result_k = c_{k-1}/k for 1 <= k < N (the old top term c_{N-1} is truncated away).
    [[nodiscard]] auto integrate() const -> Result<BigPowerSeries>;

    // --- Composition -----------------------------------------------------------
    // Compose this with g, i.e. (this o g)(x) = this(g(x)), evaluated by Horner over
    // the truncated ring. Requires g.coefficient(0) == 0 (else domain_error) so the
    // result is a well-defined power series, and equal order.
    [[nodiscard]] auto compose(const BigPowerSeries& g) const -> Result<BigPowerSeries>;

    // --- Transcendental (exact coefficient recurrences over Q) ------------------
    // exp(f). Requires c_0 == 0 (else domain_error) so the constant term e^{c_0} = 1
    // stays rational and the result is a genuine power series. Recurrence from
    // (exp f)' = f' exp f: e_0 = 1, e_k = (1/k) sum_{i=1..k} i f_i e_{k-i}.
    [[nodiscard]] auto exp() const -> Result<BigPowerSeries>;
    // log(f). Requires c_0 == 1 (else domain_error). Computed as integrate(f'/f), which
    // gives l_0 = 0 automatically.
    [[nodiscard]] auto log() const -> Result<BigPowerSeries>;

private:
    explicit BigPowerSeries(std::vector<BigRational> coeffs) : coeffs_(std::move(coeffs)) {}

    std::vector<BigRational> coeffs_;  // exactly order() entries; coeffs_[i] is c_i
};

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

auto BigPowerSeries::from_coeffs(std::vector<BigRational> coeffs, std::size_t order)
    -> Result<BigPowerSeries> {
    if (order == 0) {
        return make_error<BigPowerSeries>(MathError::domain_error);
    }
    coeffs.resize(order);  // pads with BigRational{} (= 0) or truncates to exactly `order`
    return BigPowerSeries(std::move(coeffs));
}

auto BigPowerSeries::constant(const BigRational& c, std::size_t order)
    -> Result<BigPowerSeries> {
    if (order == 0) {
        return make_error<BigPowerSeries>(MathError::domain_error);
    }
    std::vector<BigRational> coeffs(order);  // all zero
    coeffs[0] = c;
    return BigPowerSeries(std::move(coeffs));
}

auto BigPowerSeries::variable(std::size_t order) -> Result<BigPowerSeries> {
    if (order == 0) {
        return make_error<BigPowerSeries>(MathError::domain_error);
    }
    std::vector<BigRational> coeffs(order);  // all zero
    if (order > 1) {
        coeffs[1] = BigRational::from_int(1);
    }
    return BigPowerSeries(std::move(coeffs));
}

auto BigPowerSeries::zero(std::size_t order) -> Result<BigPowerSeries> {
    if (order == 0) {
        return make_error<BigPowerSeries>(MathError::domain_error);
    }
    return BigPowerSeries(std::vector<BigRational>(order));
}

auto BigPowerSeries::one(std::size_t order) -> Result<BigPowerSeries> {
    return constant(BigRational::from_int(1), order);
}

auto BigPowerSeries::coefficient(std::size_t k) const -> BigRational {
    if (k >= coeffs_.size()) {
        return BigRational::from_int(0);
    }
    return coeffs_[k];
}

auto BigPowerSeries::is_equal(const BigPowerSeries& o) const -> bool {
    if (coeffs_.size() != o.coeffs_.size()) {
        return false;
    }
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        if (!(coeffs_[i] == o.coeffs_[i])) {
            return false;
        }
    }
    return true;
}

auto BigPowerSeries::to_string(std::string_view var) const -> std::string {
    std::string out;
    bool first = true;
    for (std::size_t k = 0; k < coeffs_.size(); ++k) {
        if (coeffs_[k].is_zero()) {
            continue;
        }
        if (!first) {
            out += " + ";
        }
        first = false;
        if (k == 0) {
            out += coeffs_[k].to_string();
        } else {
            if (!(coeffs_[k] == BigRational::from_int(1))) {
                out += coeffs_[k].to_string();
                out += "*";
            }
            out += std::string(var);
            if (k > 1) {
                out += "^";
                out += std::to_string(k);
            }
        }
    }
    if (first) {
        out += "0";
    }
    out += " + O(";
    out += std::string(var);
    out += "^";
    out += std::to_string(coeffs_.size());
    out += ")";
    return out;
}

auto BigPowerSeries::add(const BigPowerSeries& o) const -> Result<BigPowerSeries> {
    if (coeffs_.size() != o.coeffs_.size()) {
        return make_error<BigPowerSeries>(MathError::domain_error);
    }
    std::vector<BigRational> result(coeffs_.size());
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        // BigRational::add is infallible (arbitrary precision cannot overflow).
        result[i] = coeffs_[i].add(o.coeffs_[i]);
    }
    return BigPowerSeries(std::move(result));
}

auto BigPowerSeries::subtract(const BigPowerSeries& o) const -> Result<BigPowerSeries> {
    if (coeffs_.size() != o.coeffs_.size()) {
        return make_error<BigPowerSeries>(MathError::domain_error);
    }
    std::vector<BigRational> result(coeffs_.size());
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        result[i] = coeffs_[i].subtract(o.coeffs_[i]);
    }
    return BigPowerSeries(std::move(result));
}

auto BigPowerSeries::scale(const BigRational& s) const -> Result<BigPowerSeries> {
    std::vector<BigRational> result(coeffs_.size());
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        result[i] = coeffs_[i].multiply(s);
    }
    return BigPowerSeries(std::move(result));
}

auto BigPowerSeries::multiply(const BigPowerSeries& o) const -> Result<BigPowerSeries> {
    if (coeffs_.size() != o.coeffs_.size()) {
        return make_error<BigPowerSeries>(MathError::domain_error);
    }
    const std::size_t n = coeffs_.size();
    std::vector<BigRational> result(n);  // all zero
    for (std::size_t k = 0; k < n; ++k) {
        BigRational acc;  // 0
        for (std::size_t i = 0; i <= k; ++i) {
            acc = acc.add(coeffs_[i].multiply(o.coeffs_[k - i]));
        }
        result[k] = acc;
    }
    return BigPowerSeries(std::move(result));
}

auto BigPowerSeries::inverse() const -> Result<BigPowerSeries> {
    const std::size_t n = coeffs_.size();
    if (coeffs_[0].is_zero()) {
        return make_error<BigPowerSeries>(MathError::domain_error);
    }
    std::vector<BigRational> b(n);
    // b_0 = 1 / a_0.
    auto b0 = BigRational::from_int(1).divide(coeffs_[0]);
    if (!b0) {
        return make_error<BigPowerSeries>(b0.error());
    }
    b[0] = *b0;
    for (std::size_t k = 1; k < n; ++k) {
        // s = sum_{i=1..k} a_i b_{k-i}.
        BigRational s;  // 0
        for (std::size_t i = 1; i <= k; ++i) {
            s = s.add(coeffs_[i].multiply(b[k - i]));
        }
        // b_k = -(1/a_0) * s = -(b_0 * s).
        b[k] = s.multiply(b[0]).negate();
    }
    return BigPowerSeries(std::move(b));
}

auto BigPowerSeries::divide(const BigPowerSeries& o) const -> Result<BigPowerSeries> {
    if (coeffs_.size() != o.coeffs_.size()) {
        return make_error<BigPowerSeries>(MathError::domain_error);
    }
    auto inv = o.inverse();
    if (!inv) {
        return make_error<BigPowerSeries>(inv.error());
    }
    return multiply(*inv);
}

auto BigPowerSeries::derivative() const -> Result<BigPowerSeries> {
    const std::size_t n = coeffs_.size();
    std::vector<BigRational> d(n);  // all zero; d_{n-1} stays 0 by convention
    for (std::size_t k = 0; k + 1 < n; ++k) {
        auto factor = BigRational::from_int(static_cast<std::int64_t>(k + 1));
        d[k] = coeffs_[k + 1].multiply(factor);
    }
    return BigPowerSeries(std::move(d));
}

auto BigPowerSeries::integrate() const -> Result<BigPowerSeries> {
    const std::size_t n = coeffs_.size();
    std::vector<BigRational> g(n);  // g_0 = 0
    for (std::size_t k = 1; k < n; ++k) {
        auto denom = BigRational::from_int(static_cast<std::int64_t>(k));
        auto q = coeffs_[k - 1].divide(denom);
        if (!q) {
            return make_error<BigPowerSeries>(q.error());
        }
        g[k] = *q;
    }
    return BigPowerSeries(std::move(g));
}

auto BigPowerSeries::compose(const BigPowerSeries& g) const -> Result<BigPowerSeries> {
    const std::size_t n = coeffs_.size();
    if (n != g.coeffs_.size()) {
        return make_error<BigPowerSeries>(MathError::domain_error);
    }
    if (!g.coeffs_[0].is_zero()) {
        return make_error<BigPowerSeries>(MathError::domain_error);
    }
    // Horner over the truncated ring: acc = a_{n-1}; then repeatedly acc = acc*g + a_i.
    auto acc_init = constant(coeffs_[n - 1], n);
    if (!acc_init) {
        return make_error<BigPowerSeries>(acc_init.error());
    }
    BigPowerSeries acc = *acc_init;
    for (std::size_t idx = n - 1; idx-- > 0;) {
        auto prod = acc.multiply(g);
        if (!prod) {
            return make_error<BigPowerSeries>(prod.error());
        }
        auto term = constant(coeffs_[idx], n);
        if (!term) {
            return make_error<BigPowerSeries>(term.error());
        }
        auto sum = prod->add(*term);
        if (!sum) {
            return make_error<BigPowerSeries>(sum.error());
        }
        acc = *sum;
    }
    return acc;
}

auto BigPowerSeries::exp() const -> Result<BigPowerSeries> {
    const std::size_t n = coeffs_.size();
    if (!coeffs_[0].is_zero()) {
        return make_error<BigPowerSeries>(MathError::domain_error);
    }
    std::vector<BigRational> e(n);
    e[0] = BigRational::from_int(1);
    for (std::size_t k = 1; k < n; ++k) {
        // s = sum_{i=1..k} i * f_i * e_{k-i}.
        BigRational s;  // 0
        for (std::size_t i = 1; i <= k; ++i) {
            auto scaled = coeffs_[i].multiply(BigRational::from_int(static_cast<std::int64_t>(i)));
            s = s.add(scaled.multiply(e[k - i]));
        }
        // e_k = s / k.
        auto ek = s.divide(BigRational::from_int(static_cast<std::int64_t>(k)));
        if (!ek) {
            return make_error<BigPowerSeries>(ek.error());
        }
        e[k] = *ek;
    }
    return BigPowerSeries(std::move(e));
}

auto BigPowerSeries::log() const -> Result<BigPowerSeries> {
    if (!(coeffs_[0] == BigRational::from_int(1))) {
        return make_error<BigPowerSeries>(MathError::domain_error);
    }
    // log(f) = integrate(f' / f); the constant term of the integral is 0.
    auto deriv = derivative();
    if (!deriv) {
        return make_error<BigPowerSeries>(deriv.error());
    }
    auto inv = inverse();  // c_0 == 1 != 0, so this cannot fail on the domain test
    if (!inv) {
        return make_error<BigPowerSeries>(inv.error());
    }
    auto quotient = deriv->multiply(*inv);
    if (!quotient) {
        return make_error<BigPowerSeries>(quotient.error());
    }
    return quotient->integrate();
}

}  // namespace nimblecas
