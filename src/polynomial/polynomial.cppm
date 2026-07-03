// NimbleCAS dense univariate polynomial arithmetic (ROADMAP 7.20).
// @author Olumuyiwa Oluwasanmi
//
// A dense univariate polynomial over exact int64 coefficients: coeffs[i] is the
// coefficient of x^i, stored trimmed (no trailing zeros) so the degree is
// unambiguous and equality is a vector compare. Ring operations are overflow-checked
// and return Result (Rule 32). evaluate_batch is the numeric fast path — it evaluates
// the polynomial at many points at once through the SIMD Horner kernel.

module;
#include <cassert>

export module nimblecas.polynomial;

import std;
import nimblecas.core;
import nimblecas.simd;

export namespace nimblecas {

class Polynomial {
public:
    Polynomial() = default;  // the zero polynomial
    explicit Polynomial(std::vector<std::int64_t> coefficients)
        : coeffs_(trim(std::move(coefficients))) {}

    // c (degree 0), or coeff * x^degree.
    [[nodiscard]] static auto constant(std::int64_t c) -> Polynomial;
    [[nodiscard]] static auto monomial(std::int64_t coeff, std::size_t degree) -> Polynomial;

    [[nodiscard]] auto is_zero() const noexcept -> bool { return coeffs_.empty(); }

    // Degree of the polynomial; -1 for the zero polynomial (conventional).
    [[nodiscard]] auto degree() const noexcept -> std::int64_t {
        return coeffs_.empty() ? -1 : static_cast<std::int64_t>(coeffs_.size()) - 1;
    }
    // Coefficient of x^i (0 beyond the stored degree).
    [[nodiscard]] auto coefficient(std::size_t i) const noexcept -> std::int64_t {
        return i < coeffs_.size() ? coeffs_[i] : 0;
    }
    [[nodiscard]] auto leading_coefficient() const noexcept -> std::int64_t {
        return coeffs_.empty() ? 0 : coeffs_.back();
    }
    [[nodiscard]] auto coefficients() const noexcept -> std::span<const std::int64_t> {
        return coeffs_;
    }

    // Ring operations (overflow-checked, exact).
    [[nodiscard]] auto add(const Polynomial& other) const -> Result<Polynomial>;
    [[nodiscard]] auto subtract(const Polynomial& other) const -> Result<Polynomial>;
    [[nodiscard]] auto scale(std::int64_t s) const -> Result<Polynomial>;
    [[nodiscard]] auto multiply(const Polynomial& other) const -> Result<Polynomial>;

    // Exact evaluation at an integer point (Horner, overflow-checked).
    [[nodiscard]] auto evaluate(std::int64_t x) const -> Result<std::int64_t>;

    // Evaluate as floating point at many points at once, via the SIMD engine
    // (vectorised Horner). Coefficients are taken as float — the numeric path (NFR-1).
    [[nodiscard]] auto evaluate_batch(std::span<const float> xs) const -> std::vector<float>;

    [[nodiscard]] auto is_equal(const Polynomial& other) const noexcept -> bool {
        return coeffs_ == other.coeffs_;
    }
    [[nodiscard]] auto to_string(std::string_view var = "x") const -> std::string;

private:
    std::vector<std::int64_t> coeffs_;  // trimmed: back() != 0, or empty for zero

    [[nodiscard]] static auto trim(std::vector<std::int64_t> c) -> std::vector<std::int64_t> {
        while (!c.empty() && c.back() == 0) {
            c.pop_back();
        }
        return c;
    }
};

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

[[nodiscard]] auto add_ov(std::int64_t a, std::int64_t b, std::int64_t& out) -> bool {
    return __builtin_add_overflow(a, b, &out);
}
[[nodiscard]] auto sub_ov(std::int64_t a, std::int64_t b, std::int64_t& out) -> bool {
    return __builtin_sub_overflow(a, b, &out);
}
[[nodiscard]] auto mul_ov(std::int64_t a, std::int64_t b, std::int64_t& out) -> bool {
    return __builtin_mul_overflow(a, b, &out);
}

}  // namespace

auto Polynomial::constant(std::int64_t c) -> Polynomial {
    if (c == 0) {
        return Polynomial{};
    }
    return Polynomial{std::vector<std::int64_t>{c}};
}

auto Polynomial::monomial(std::int64_t coeff, std::size_t degree) -> Polynomial {
    if (coeff == 0) {
        return Polynomial{};
    }
    std::vector<std::int64_t> c(degree + 1, 0);
    c[degree] = coeff;
    return Polynomial{std::move(c)};
}

auto Polynomial::add(const Polynomial& other) const -> Result<Polynomial> {
    std::vector<std::int64_t> r(std::max(coeffs_.size(), other.coeffs_.size()), 0);
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        r[i] = coeffs_[i];
    }
    for (std::size_t i = 0; i < other.coeffs_.size(); ++i) {
        if (add_ov(r[i], other.coeffs_[i], r[i])) {
            return make_error<Polynomial>(MathError::overflow);
        }
    }
    return Polynomial{std::move(r)};
}

auto Polynomial::subtract(const Polynomial& other) const -> Result<Polynomial> {
    std::vector<std::int64_t> r(std::max(coeffs_.size(), other.coeffs_.size()), 0);
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        r[i] = coeffs_[i];
    }
    for (std::size_t i = 0; i < other.coeffs_.size(); ++i) {
        if (sub_ov(r[i], other.coeffs_[i], r[i])) {
            return make_error<Polynomial>(MathError::overflow);
        }
    }
    return Polynomial{std::move(r)};
}

auto Polynomial::scale(std::int64_t s) const -> Result<Polynomial> {
    if (s == 0) {
        return Polynomial{};
    }
    std::vector<std::int64_t> r(coeffs_.size(), 0);
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        if (mul_ov(coeffs_[i], s, r[i])) {
            return make_error<Polynomial>(MathError::overflow);
        }
    }
    return Polynomial{std::move(r)};
}

auto Polynomial::multiply(const Polynomial& other) const -> Result<Polynomial> {
    if (is_zero() || other.is_zero()) {
        return Polynomial{};
    }
    std::vector<std::int64_t> r(coeffs_.size() + other.coeffs_.size() - 1, 0);
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        for (std::size_t j = 0; j < other.coeffs_.size(); ++j) {
            std::int64_t product = 0;
            if (mul_ov(coeffs_[i], other.coeffs_[j], product) ||
                add_ov(r[i + j], product, r[i + j])) {
                return make_error<Polynomial>(MathError::overflow);
            }
        }
    }
    return Polynomial{std::move(r)};
}

auto Polynomial::evaluate(std::int64_t x) const -> Result<std::int64_t> {
    std::int64_t acc = 0;
    for (std::size_t k = coeffs_.size(); k-- > 0;) {  // high degree -> low
        if (mul_ov(acc, x, acc) || add_ov(acc, coeffs_[k], acc)) {
            return make_error<std::int64_t>(MathError::overflow);
        }
    }
    return acc;
}

auto Polynomial::evaluate_batch(std::span<const float> xs) const -> std::vector<float> {
    std::vector<float> acc(xs.size(), 0.0f);
    if (coeffs_.empty()) {
        return acc;  // zero polynomial
    }
    // Horner: seed with the leading coefficient, then fold in the rest high -> low.
    std::ranges::fill(acc, static_cast<float>(coeffs_.back()));
    for (std::size_t k = coeffs_.size() - 1; k-- > 0;) {
        simd::horner_step(acc, xs, static_cast<float>(coeffs_[k]));  // acc = acc*x + c_k
    }
    return acc;
}

auto Polynomial::to_string(std::string_view var) const -> std::string {
    if (coeffs_.empty()) {
        return "0";
    }
    std::string out;
    bool first = true;
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        if (coeffs_[i] == 0) {
            continue;
        }
        if (!first) {
            out += " + ";
        }
        first = false;
        if (i == 0) {
            out += std::format("{}", coeffs_[i]);
        } else if (i == 1) {
            out += std::format("{}*{}", coeffs_[i], var);
        } else {
            out += std::format("{}*{}^{}", coeffs_[i], var, i);
        }
    }
    return out;
}

}  // namespace nimblecas
