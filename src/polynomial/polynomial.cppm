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

    // Content = gcd of the coefficients (>= 0; 0 for the zero polynomial).
    [[nodiscard]] auto content() const -> Result<std::int64_t>;
    // Primitive part = this / content, sign-normalised to a positive leading coeff.
    [[nodiscard]] auto primitive_part() const -> Result<Polynomial>;
    // Pseudo-remainder R: lc(divisor)^(deg-diff+1) * this = divisor*Q + R, staying in
    // Z[x] (no fractions). Fails on a zero divisor or int64 overflow.
    [[nodiscard]] auto pseudo_remainder(const Polynomial& divisor) const -> Result<Polynomial>;
    // Greatest common divisor in Z[x] via the primitive Euclidean PRS.
    [[nodiscard]] auto gcd(const Polynomial& other) const -> Result<Polynomial>;
    // Formal derivative d/dx.
    [[nodiscard]] auto derivative() const -> Result<Polynomial>;
    // Exact quotient this / divisor over Z[x]; fails (domain_error) if the division
    // is not exact, or division_by_zero on a zero divisor.
    [[nodiscard]] auto divide_exact(const Polynomial& divisor) const -> Result<Polynomial>;
    // Yun square-free factorization: returns (factor, multiplicity) pairs such that
    // the primitive part of this equals the product of factor^multiplicity, each
    // factor square-free and pairwise coprime.
    [[nodiscard]] auto square_free_factorization() const
        -> Result<std::vector<std::pair<Polynomial, std::int64_t>>>;

    // Exact evaluation at an integer point (Horner, overflow-checked).
    [[nodiscard]] auto evaluate(std::int64_t x) const -> Result<std::int64_t>;

    // Evaluate as floating point at many points at once, via the SIMD engine
    // (vectorised Horner). Coefficients are taken as float — the numeric path (NFR-1).
    [[nodiscard]] auto evaluate_batch(std::span<const float> xs) const -> std::vector<float>;

    // Allocation-free batch evaluation: writes p(xs[i]) into out[i] for each i < xs.size(),
    // returning false (a no-op) when out is smaller than xs. This is the fast path —
    // evaluate_batch() is a thin wrapper that allocates the result. Profiling (perf) shows
    // the returning form is dominated by the per-call output allocation on large sweeps;
    // reusing one caller buffer across calls removes that cost, leaving only the SIMD Horner
    // and DRAM bandwidth.
    [[nodiscard]] auto evaluate_batch_into(std::span<const float> xs, std::span<float> out) const
        -> bool;

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

// std::gcd is UB if |m| or |n| is unrepresentable (i.e. INT64_MIN); guard it.
[[nodiscard]] auto checked_gcd(std::int64_t a, std::int64_t b) -> std::optional<std::int64_t> {
    constexpr std::int64_t int64_min = std::numeric_limits<std::int64_t>::min();
    if (a == int64_min || b == int64_min) {
        return std::nullopt;
    }
    return std::gcd(a, b);
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
    // Guard the degree+1 wrap (degree == SIZE_MAX would index an empty vector). Any
    // such degree is unrepresentable anyway (the allocation would dwarf memory).
    assert(degree < std::numeric_limits<std::size_t>::max() && "monomial degree too large");
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

auto Polynomial::content() const -> Result<std::int64_t> {
    std::int64_t g = 0;
    for (const std::int64_t c : coeffs_) {
        auto next = checked_gcd(g, c);
        if (!next) {
            return make_error<std::int64_t>(MathError::overflow);
        }
        g = *next;  // gcd(0, c) = |c|, so g stays >= 0
    }
    return g;
}

auto Polynomial::primitive_part() const -> Result<Polynomial> {
    if (is_zero()) {
        return Polynomial{};
    }
    auto c = content();
    if (!c) {
        return make_error<Polynomial>(c.error());
    }
    const std::int64_t g = *c;  // > 0 for a non-zero polynomial
    std::vector<std::int64_t> r(coeffs_.size(), 0);
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        r[i] = coeffs_[i] / g;  // exact: content divides every coefficient
    }
    Polynomial p{std::move(r)};
    if (p.leading_coefficient() < 0) {
        return p.scale(-1);  // sign-normalise to a positive leading coefficient
    }
    return p;
}

auto Polynomial::pseudo_remainder(const Polynomial& divisor) const -> Result<Polynomial> {
    if (divisor.is_zero()) {
        return make_error<Polynomial>(MathError::division_by_zero);
    }
    if (degree() < divisor.degree()) {
        return *this;  // already the remainder (scaling power is d^0 = 1)
    }
    const std::int64_t d = divisor.leading_coefficient();
    const std::int64_t n = divisor.degree();
    std::int64_t e = degree() - n + 1;  // total scaling exponent lc(divisor)^e
    Polynomial r = *this;
    while (!r.is_zero() && r.degree() >= n) {
        // r <- d*r - lc(r) * x^(deg r - n) * divisor
        auto s = Polynomial::monomial(r.leading_coefficient(),
                                      static_cast<std::size_t>(r.degree() - n));
        auto dr = r.scale(d);
        if (!dr) {
            return dr;
        }
        auto sb = s.multiply(divisor);
        if (!sb) {
            return sb;
        }
        auto next = dr->subtract(*sb);
        if (!next) {
            return next;
        }
        r = *next;
        --e;
    }
    for (; e > 0; --e) {  // apply the remaining lc(divisor)^e factor
        auto scaled = r.scale(d);
        if (!scaled) {
            return scaled;
        }
        r = *scaled;
    }
    return r;
}

auto Polynomial::gcd(const Polynomial& other) const -> Result<Polynomial> {
    auto normalized = [](const Polynomial& p) -> Result<Polynomial> {
        return p.leading_coefficient() < 0 ? p.scale(-1) : Result<Polynomial>{p};
    };
    if (is_zero()) {
        return normalized(other);
    }
    if (other.is_zero()) {
        return normalized(*this);
    }
    // gcd of contents times the primitive gcd (primitive Euclidean PRS).
    auto ca = content();
    auto cb = other.content();
    if (!ca || !cb) {
        return make_error<Polynomial>(MathError::overflow);
    }
    auto d = checked_gcd(*ca, *cb);
    if (!d) {
        return make_error<Polynomial>(MathError::overflow);
    }
    auto a = primitive_part();
    auto b = other.primitive_part();
    if (!a || !b) {
        return make_error<Polynomial>(a ? b.error() : a.error());
    }
    Polynomial pa = *a;
    Polynomial pb = *b;
    while (!pb.is_zero()) {
        auto r = pa.pseudo_remainder(pb);
        if (!r) {
            return r;
        }
        pa = pb;
        auto prim = r->primitive_part();
        if (!prim) {
            return prim;
        }
        pb = *prim;
    }
    return pa.scale(*d);  // primitive gcd scaled by the content gcd
}

auto Polynomial::derivative() const -> Result<Polynomial> {
    if (coeffs_.size() <= 1) {
        return Polynomial{};  // derivative of a constant / zero is 0
    }
    std::vector<std::int64_t> r(coeffs_.size() - 1, 0);
    for (std::size_t i = 1; i < coeffs_.size(); ++i) {
        if (mul_ov(coeffs_[i], static_cast<std::int64_t>(i), r[i - 1])) {
            return make_error<Polynomial>(MathError::overflow);
        }
    }
    return Polynomial{std::move(r)};
}

auto Polynomial::divide_exact(const Polynomial& divisor) const -> Result<Polynomial> {
    if (divisor.is_zero()) {
        return make_error<Polynomial>(MathError::division_by_zero);
    }
    if (is_zero()) {
        return Polynomial{};
    }
    if (degree() < divisor.degree()) {
        return make_error<Polynomial>(MathError::domain_error);  // non-zero, lower degree
    }
    const std::int64_t lc_b = divisor.leading_coefficient();
    const std::int64_t nb = divisor.degree();
    std::vector<std::int64_t> quotient(static_cast<std::size_t>(degree() - nb) + 1, 0);
    Polynomial r = *this;
    while (!r.is_zero() && r.degree() >= nb) {
        const std::int64_t lc_r = r.leading_coefficient();
        if (lc_r % lc_b != 0) {
            return make_error<Polynomial>(MathError::domain_error);  // not exact over Z
        }
        const std::int64_t q = lc_r / lc_b;
        const std::size_t k = static_cast<std::size_t>(r.degree() - nb);
        quotient[k] = q;
        auto qb = Polynomial::monomial(q, k).multiply(divisor);
        if (!qb) {
            return qb;
        }
        auto next = r.subtract(*qb);
        if (!next) {
            return next;
        }
        r = *next;
    }
    if (!r.is_zero()) {
        return make_error<Polynomial>(MathError::domain_error);  // remainder != 0
    }
    return Polynomial{std::move(quotient)};
}

auto Polynomial::square_free_factorization() const
    -> Result<std::vector<std::pair<Polynomial, std::int64_t>>> {
    std::vector<std::pair<Polynomial, std::int64_t>> factors;
    if (degree() <= 0) {
        return factors;  // constants have no square-free factors
    }
    auto prim = primitive_part();
    if (!prim) {
        return make_error<std::vector<std::pair<Polynomial, std::int64_t>>>(prim.error());
    }
    const Polynomial f = *prim;

    // Yun: a0 = gcd(f, f'); b1 = f/a0; c1 = f'/a0; d1 = c1 - b1'.
    auto fp = f.derivative();
    if (!fp) {
        return make_error<std::vector<std::pair<Polynomial, std::int64_t>>>(fp.error());
    }
    auto a0 = f.gcd(*fp);
    if (!a0) {
        return make_error<std::vector<std::pair<Polynomial, std::int64_t>>>(a0.error());
    }
    auto b = f.divide_exact(*a0);
    auto c = fp->divide_exact(*a0);
    if (!b || !c) {
        return make_error<std::vector<std::pair<Polynomial, std::int64_t>>>(
            b ? c.error() : b.error());
    }
    auto bd = b->derivative();
    if (!bd) {
        return make_error<std::vector<std::pair<Polynomial, std::int64_t>>>(bd.error());
    }
    auto d = c->subtract(*bd);
    if (!d) {
        return make_error<std::vector<std::pair<Polynomial, std::int64_t>>>(d.error());
    }

    Polynomial bi = *b;
    Polynomial di = *d;
    for (std::int64_t i = 1; bi.degree() > 0; ++i) {
        auto ai = bi.gcd(di);  // product of the factors of multiplicity exactly i
        if (!ai) {
            return make_error<std::vector<std::pair<Polynomial, std::int64_t>>>(ai.error());
        }
        if (ai->degree() > 0) {
            factors.emplace_back(*ai, i);
        }
        auto bn = bi.divide_exact(*ai);
        auto cn = di.divide_exact(*ai);
        if (!bn || !cn) {
            return make_error<std::vector<std::pair<Polynomial, std::int64_t>>>(
                bn ? cn.error() : bn.error());
        }
        auto bnd = bn->derivative();
        if (!bnd) {
            return make_error<std::vector<std::pair<Polynomial, std::int64_t>>>(bnd.error());
        }
        auto dn = cn->subtract(*bnd);
        if (!dn) {
            return make_error<std::vector<std::pair<Polynomial, std::int64_t>>>(dn.error());
        }
        bi = *bn;
        di = *dn;
    }
    return factors;
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

auto Polynomial::evaluate_batch_into(std::span<const float> xs, std::span<float> out) const
    -> bool {
    if (out.size() < xs.size()) {
        return false;
    }
    const std::span<float> acc = out.subspan(0, xs.size());
    if (coeffs_.empty()) {
        std::ranges::fill(acc, 0.0f);  // zero polynomial
        return true;
    }
    // Horner: seed with the leading coefficient, then fold in the rest high -> low.
    std::ranges::fill(acc, static_cast<float>(coeffs_.back()));
    for (std::size_t k = coeffs_.size() - 1; k-- > 0;) {
        simd::horner_step(acc, xs, static_cast<float>(coeffs_[k]));  // acc = acc*x + c_k
    }
    return true;
}

auto Polynomial::evaluate_batch(std::span<const float> xs) const -> std::vector<float> {
    std::vector<float> out(xs.size());
    static_cast<void>(evaluate_batch_into(xs, out));  // size always sufficient here
    return out;
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
