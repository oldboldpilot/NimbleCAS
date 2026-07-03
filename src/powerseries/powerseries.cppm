// NimbleCAS truncated formal power series over Q (ROADMAP 7.4/7.5 dependency).
// @author Olumuyiwa Oluwasanmi
//
// A PowerSeries is an element of Q[[x]] / (x^N): it retains a fixed number of
// coefficients `order = N`, representing the terms c_0 + c_1 x + ... + c_{N-1} x^{N-1}
// with everything of degree >= N discarded (an implicit O(x^N) tail). The coefficient
// vector always holds exactly N Rationals. Every binary operation requires the two
// operands to share the same order (otherwise MathError::domain_error); this keeps the
// truncation ring unambiguous. All arithmetic flows through the overflow-checked
// Rational operations and propagates their errors along the railway (Rule 32) — no raw
// int64 products that could silently wrap.

export module nimblecas.powerseries;

import std;
import nimblecas.core;
import nimblecas.ratpoly;

export namespace nimblecas {

// Truncated formal power series over Q, working modulo x^N where N == order().
class PowerSeries {
public:
    // --- Factories -------------------------------------------------------------
    // Build from an explicit coefficient list, padded with zeros or truncated so the
    // result holds exactly `order` coefficients (c_0..c_{order-1}). order == 0 is a
    // domain_error.
    [[nodiscard]] static auto from_coeffs(std::vector<Rational> coeffs, std::size_t order)
        -> Result<PowerSeries>;
    // The constant series c + O(x^N). order == 0 is a domain_error.
    [[nodiscard]] static auto constant(const Rational& c, std::size_t order)
        -> Result<PowerSeries>;
    // The series x (c_1 = 1). For order == 1 this truncates to 0; order == 0 is a
    // domain_error.
    [[nodiscard]] static auto variable(std::size_t order) -> Result<PowerSeries>;
    // The additive identity 0 + O(x^N). order == 0 is a domain_error.
    [[nodiscard]] static auto zero(std::size_t order) -> Result<PowerSeries>;
    // The multiplicative identity 1 + O(x^N). order == 0 is a domain_error.
    [[nodiscard]] static auto one(std::size_t order) -> Result<PowerSeries>;

    // --- Accessors -------------------------------------------------------------
    // Number of retained coefficients N (terms x^0..x^{N-1}).
    [[nodiscard]] auto order() const noexcept -> std::size_t { return coeffs_.size(); }
    // Coefficient of x^k; 0 for k >= order().
    [[nodiscard]] auto coefficient(std::size_t k) const -> Rational;
    // Read-only view of all N coefficients (index i is the coefficient of x^i).
    [[nodiscard]] auto coefficients() const noexcept -> std::span<const Rational> {
        return coeffs_;
    }
    // Human-readable form, e.g. "1 + 2*x + x^2 + O(x^3)".
    [[nodiscard]] auto to_string(std::string_view var = "x") const -> std::string;
    // Same order and identical coefficients.
    [[nodiscard]] auto is_equal(const PowerSeries& o) const -> bool;

    // --- Arithmetic ------------------------------------------------------------
    // Coefficient-wise sum / difference (requires equal order).
    [[nodiscard]] auto add(const PowerSeries& o) const -> Result<PowerSeries>;
    [[nodiscard]] auto subtract(const PowerSeries& o) const -> Result<PowerSeries>;
    // Multiply every coefficient by the scalar s.
    [[nodiscard]] auto scale(const Rational& s) const -> Result<PowerSeries>;
    // Truncated Cauchy product: (a*b)_k = sum_{i+j=k} a_i b_j for k < N (requires equal
    // order).
    [[nodiscard]] auto multiply(const PowerSeries& o) const -> Result<PowerSeries>;
    // Multiplicative inverse. Requires c_0 != 0 (else domain_error). Uses the standard
    // recurrence b_0 = 1/a_0, b_k = -(1/a_0) sum_{i=1..k} a_i b_{k-i}.
    [[nodiscard]] auto inverse() const -> Result<PowerSeries>;
    // Division = multiply by the inverse of o (requires equal order; o_0 != 0).
    [[nodiscard]] auto divide(const PowerSeries& o) const -> Result<PowerSeries>;

    // --- Calculus --------------------------------------------------------------
    // Formal derivative. Order convention: the result KEEPS the same order N, with the
    // top coefficient set to 0 (since the x^{N-1} term would come from the truncated
    // x^N coefficient). result_k = (k+1) c_{k+1} for k < N-1, result_{N-1} = 0.
    [[nodiscard]] auto derivative() const -> Result<PowerSeries>;
    // Formal integral with zero constant of integration. result_0 = 0,
    // result_k = c_{k-1}/k for 1 <= k < N (the old top term c_{N-1} is truncated away).
    [[nodiscard]] auto integrate() const -> Result<PowerSeries>;

    // --- Composition -----------------------------------------------------------
    // Compose this with g, i.e. (this o g)(x) = this(g(x)), evaluated by Horner over
    // the truncated ring. Requires g.coefficient(0) == 0 (else domain_error) so the
    // result is a well-defined power series, and equal order.
    [[nodiscard]] auto compose(const PowerSeries& g) const -> Result<PowerSeries>;

    // --- Transcendental (exact coefficient recurrences over Q) ------------------
    // exp(f). Requires c_0 == 0 (else domain_error) so the constant term e^{c_0} = 1
    // stays rational and the result is a genuine power series. Recurrence from
    // (exp f)' = f' exp f: e_0 = 1, e_k = (1/k) sum_{i=1..k} i f_i e_{k-i}.
    [[nodiscard]] auto exp() const -> Result<PowerSeries>;
    // log(f). Requires c_0 == 1 (else domain_error). Computed as integrate(f'/f), which
    // gives l_0 = 0 automatically.
    [[nodiscard]] auto log() const -> Result<PowerSeries>;

private:
    explicit PowerSeries(std::vector<Rational> coeffs) : coeffs_(std::move(coeffs)) {}

    std::vector<Rational> coeffs_;  // exactly order() entries; coeffs_[i] is c_i
};

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

auto PowerSeries::from_coeffs(std::vector<Rational> coeffs, std::size_t order)
    -> Result<PowerSeries> {
    if (order == 0) {
        return make_error<PowerSeries>(MathError::domain_error);
    }
    coeffs.resize(order);  // pads with Rational{} (= 0) or truncates to exactly `order`
    return PowerSeries(std::move(coeffs));
}

auto PowerSeries::constant(const Rational& c, std::size_t order) -> Result<PowerSeries> {
    if (order == 0) {
        return make_error<PowerSeries>(MathError::domain_error);
    }
    std::vector<Rational> coeffs(order);  // all zero
    coeffs[0] = c;
    return PowerSeries(std::move(coeffs));
}

auto PowerSeries::variable(std::size_t order) -> Result<PowerSeries> {
    if (order == 0) {
        return make_error<PowerSeries>(MathError::domain_error);
    }
    std::vector<Rational> coeffs(order);  // all zero
    if (order > 1) {
        coeffs[1] = Rational::from_int(1);
    }
    return PowerSeries(std::move(coeffs));
}

auto PowerSeries::zero(std::size_t order) -> Result<PowerSeries> {
    if (order == 0) {
        return make_error<PowerSeries>(MathError::domain_error);
    }
    return PowerSeries(std::vector<Rational>(order));
}

auto PowerSeries::one(std::size_t order) -> Result<PowerSeries> {
    return constant(Rational::from_int(1), order);
}

auto PowerSeries::coefficient(std::size_t k) const -> Rational {
    if (k >= coeffs_.size()) {
        return Rational::from_int(0);
    }
    return coeffs_[k];
}

auto PowerSeries::is_equal(const PowerSeries& o) const -> bool {
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

auto PowerSeries::to_string(std::string_view var) const -> std::string {
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
            if (!(coeffs_[k] == Rational::from_int(1))) {
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

auto PowerSeries::add(const PowerSeries& o) const -> Result<PowerSeries> {
    if (coeffs_.size() != o.coeffs_.size()) {
        return make_error<PowerSeries>(MathError::domain_error);
    }
    std::vector<Rational> result(coeffs_.size());
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        auto s = coeffs_[i].add(o.coeffs_[i]);
        if (!s) {
            return make_error<PowerSeries>(s.error());
        }
        result[i] = *s;
    }
    return PowerSeries(std::move(result));
}

auto PowerSeries::subtract(const PowerSeries& o) const -> Result<PowerSeries> {
    if (coeffs_.size() != o.coeffs_.size()) {
        return make_error<PowerSeries>(MathError::domain_error);
    }
    std::vector<Rational> result(coeffs_.size());
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        auto d = coeffs_[i].subtract(o.coeffs_[i]);
        if (!d) {
            return make_error<PowerSeries>(d.error());
        }
        result[i] = *d;
    }
    return PowerSeries(std::move(result));
}

auto PowerSeries::scale(const Rational& s) const -> Result<PowerSeries> {
    std::vector<Rational> result(coeffs_.size());
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        auto p = coeffs_[i].multiply(s);
        if (!p) {
            return make_error<PowerSeries>(p.error());
        }
        result[i] = *p;
    }
    return PowerSeries(std::move(result));
}

auto PowerSeries::multiply(const PowerSeries& o) const -> Result<PowerSeries> {
    if (coeffs_.size() != o.coeffs_.size()) {
        return make_error<PowerSeries>(MathError::domain_error);
    }
    const std::size_t n = coeffs_.size();
    std::vector<Rational> result(n);  // all zero
    for (std::size_t k = 0; k < n; ++k) {
        Rational acc;  // 0
        for (std::size_t i = 0; i <= k; ++i) {
            auto prod = coeffs_[i].multiply(o.coeffs_[k - i]);
            if (!prod) {
                return make_error<PowerSeries>(prod.error());
            }
            auto sum = acc.add(*prod);
            if (!sum) {
                return make_error<PowerSeries>(sum.error());
            }
            acc = *sum;
        }
        result[k] = acc;
    }
    return PowerSeries(std::move(result));
}

auto PowerSeries::inverse() const -> Result<PowerSeries> {
    const std::size_t n = coeffs_.size();
    if (coeffs_[0].is_zero()) {
        return make_error<PowerSeries>(MathError::domain_error);
    }
    std::vector<Rational> b(n);
    // b_0 = 1 / a_0.
    auto b0 = Rational::from_int(1).divide(coeffs_[0]);
    if (!b0) {
        return make_error<PowerSeries>(b0.error());
    }
    b[0] = *b0;
    for (std::size_t k = 1; k < n; ++k) {
        // s = sum_{i=1..k} a_i b_{k-i}.
        Rational s;  // 0
        for (std::size_t i = 1; i <= k; ++i) {
            auto prod = coeffs_[i].multiply(b[k - i]);
            if (!prod) {
                return make_error<PowerSeries>(prod.error());
            }
            auto sum = s.add(*prod);
            if (!sum) {
                return make_error<PowerSeries>(sum.error());
            }
            s = *sum;
        }
        // b_k = -(1/a_0) * s = -(b_0 * s).
        auto scaled = s.multiply(b[0]);
        if (!scaled) {
            return make_error<PowerSeries>(scaled.error());
        }
        auto neg = scaled->negate();
        if (!neg) {
            return make_error<PowerSeries>(neg.error());
        }
        b[k] = *neg;
    }
    return PowerSeries(std::move(b));
}

auto PowerSeries::divide(const PowerSeries& o) const -> Result<PowerSeries> {
    if (coeffs_.size() != o.coeffs_.size()) {
        return make_error<PowerSeries>(MathError::domain_error);
    }
    auto inv = o.inverse();
    if (!inv) {
        return make_error<PowerSeries>(inv.error());
    }
    return multiply(*inv);
}

auto PowerSeries::derivative() const -> Result<PowerSeries> {
    const std::size_t n = coeffs_.size();
    std::vector<Rational> d(n);  // all zero; d_{n-1} stays 0 by convention
    for (std::size_t k = 0; k + 1 < n; ++k) {
        auto factor = Rational::from_int(static_cast<std::int64_t>(k + 1));
        auto prod = coeffs_[k + 1].multiply(factor);
        if (!prod) {
            return make_error<PowerSeries>(prod.error());
        }
        d[k] = *prod;
    }
    return PowerSeries(std::move(d));
}

auto PowerSeries::integrate() const -> Result<PowerSeries> {
    const std::size_t n = coeffs_.size();
    std::vector<Rational> g(n);  // g_0 = 0
    for (std::size_t k = 1; k < n; ++k) {
        auto denom = Rational::from_int(static_cast<std::int64_t>(k));
        auto q = coeffs_[k - 1].divide(denom);
        if (!q) {
            return make_error<PowerSeries>(q.error());
        }
        g[k] = *q;
    }
    return PowerSeries(std::move(g));
}

auto PowerSeries::compose(const PowerSeries& g) const -> Result<PowerSeries> {
    const std::size_t n = coeffs_.size();
    if (n != g.coeffs_.size()) {
        return make_error<PowerSeries>(MathError::domain_error);
    }
    if (!g.coeffs_[0].is_zero()) {
        return make_error<PowerSeries>(MathError::domain_error);
    }
    // Horner over the truncated ring: acc = a_{n-1}; then repeatedly acc = acc*g + a_i.
    auto acc_init = constant(coeffs_[n - 1], n);
    if (!acc_init) {
        return make_error<PowerSeries>(acc_init.error());
    }
    PowerSeries acc = *acc_init;
    for (std::size_t idx = n - 1; idx-- > 0;) {
        auto prod = acc.multiply(g);
        if (!prod) {
            return make_error<PowerSeries>(prod.error());
        }
        auto term = constant(coeffs_[idx], n);
        if (!term) {
            return make_error<PowerSeries>(term.error());
        }
        auto sum = prod->add(*term);
        if (!sum) {
            return make_error<PowerSeries>(sum.error());
        }
        acc = *sum;
    }
    return acc;
}

auto PowerSeries::exp() const -> Result<PowerSeries> {
    const std::size_t n = coeffs_.size();
    if (!coeffs_[0].is_zero()) {
        return make_error<PowerSeries>(MathError::domain_error);
    }
    std::vector<Rational> e(n);
    e[0] = Rational::from_int(1);
    for (std::size_t k = 1; k < n; ++k) {
        // s = sum_{i=1..k} i * f_i * e_{k-i}.
        Rational s;  // 0
        for (std::size_t i = 1; i <= k; ++i) {
            auto scaled = coeffs_[i].multiply(Rational::from_int(static_cast<std::int64_t>(i)));
            if (!scaled) {
                return make_error<PowerSeries>(scaled.error());
            }
            auto term = scaled->multiply(e[k - i]);
            if (!term) {
                return make_error<PowerSeries>(term.error());
            }
            auto sum = s.add(*term);
            if (!sum) {
                return make_error<PowerSeries>(sum.error());
            }
            s = *sum;
        }
        // e_k = s / k.
        auto ek = s.divide(Rational::from_int(static_cast<std::int64_t>(k)));
        if (!ek) {
            return make_error<PowerSeries>(ek.error());
        }
        e[k] = *ek;
    }
    return PowerSeries(std::move(e));
}

auto PowerSeries::log() const -> Result<PowerSeries> {
    if (!(coeffs_[0] == Rational::from_int(1))) {
        return make_error<PowerSeries>(MathError::domain_error);
    }
    // log(f) = integrate(f' / f); the constant term of the integral is 0.
    auto deriv = derivative();
    if (!deriv) {
        return make_error<PowerSeries>(deriv.error());
    }
    auto inv = inverse();  // c_0 == 1 != 0, so this cannot fail on the domain test
    if (!inv) {
        return make_error<PowerSeries>(inv.error());
    }
    auto quotient = deriv->multiply(*inv);
    if (!quotient) {
        return make_error<PowerSeries>(quotient.error());
    }
    return quotient->integrate();
}

}  // namespace nimblecas
