// NimbleCAS truncated Laurent series over Q (ROADMAP 7.4/7.5, extends powerseries).
// @author Olumuyiwa Oluwasanmi
//
// A Laurent series is an element of Q((x)): a *finite* principal part of negative-power
// terms plus a truncated Taylor tail. Where nimblecas.powerseries models Q[[x]]/(x^N)
// (coefficients c_0..c_{N-1} with an implicit O(x^N) tail), Laurent lifts the lower
// bound to an arbitrary integer valuation: it stores coefficients c_k for exponents k in
// [order_min, truncation_order), i.e. a leading (lowest) exponent order_min (which MAY be
// negative) and a dense vector of Rational coefficients, with everything of exponent
// >= truncation_order discarded behind an implicit O(x^{truncation_order}) tail.
//
// Honesty boundary (Rule 32), mirroring powerseries:
//   * The UPPER end is truncated: coefficients at exponents >= truncation_order() are
//     UNKNOWN. A result is only valid up to its tracked truncation order.
//   * The LOWER end is EXACT: the principal part is finite by definition, so every
//     coefficient at an exponent < order_min() is a genuine zero, not a truncation.
// Binary operations combine the two operands' tracked ranges honestly (Cauchy products
// shift the valuation and take the smaller relative precision), so the result's tracked
// order never over-claims. All arithmetic flows through the overflow-checked Rational
// operations and propagates their errors along the railway -- no raw int64 products that
// could silently wrap; an int64 numerator/denominator that would exceed range surfaces
// as MathError::overflow.

export module nimblecas.laurent;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.powerseries;

export namespace nimblecas {

// Truncated Laurent series over Q: coefficients c_k for k in
// [order_min(), truncation_order()), with an implicit O(x^{truncation_order()}) tail and
// a genuine zero below order_min().
class Laurent {
public:
    // --- Factories -------------------------------------------------------------
    // Build from an explicit coefficient list whose first entry is the coefficient of
    // x^{order_min}; coeffs[i] is the coefficient of x^{order_min + i}. The vector may
    // NOT be empty (a series must track at least one term) -> domain_error. Fails with
    // overflow if order_min + coeffs.size() would exceed the int64 exponent range. Note:
    // leading/trailing zeros are retained as written (they carry precision information);
    // is_equal compares this stored representation structurally.
    [[nodiscard]] static auto from_coeffs(std::int64_t order_min, std::vector<Rational> coeffs)
        -> Result<Laurent>;
    // The zero series tracked over [order_min, order_min + size). size == 0 -> domain_error.
    [[nodiscard]] static auto zero(std::int64_t order_min, std::size_t size) -> Result<Laurent>;
    // The constant series c + O(x^size) (order_min == 0). size == 0 -> domain_error.
    [[nodiscard]] static auto constant(const Rational& c, std::size_t size) -> Result<Laurent>;
    // The multiplicative identity 1 + O(x^size) (order_min == 0). size == 0 -> domain_error.
    [[nodiscard]] static auto one(std::size_t size) -> Result<Laurent>;
    // The single term c*x^exponent tracked over [order_min, order_min + size). Requires
    // order_min <= exponent < order_min + size (else domain_error: the term would fall
    // outside the tracked window and be silently lost). size == 0 -> domain_error.
    [[nodiscard]] static auto monomial(const Rational& c, std::int64_t exponent,
                                       std::int64_t order_min, std::size_t size)
        -> Result<Laurent>;
    // Lift a truncated power series c_0..c_{N-1} + O(x^N) into a Laurent series with
    // order_min == 0 and truncation_order == N.
    [[nodiscard]] static auto from_power_series(const PowerSeries& p) -> Result<Laurent>;

    // Laurent-expand the rational function num/den about `point`, in powers of
    // (x - point), keeping `order` coefficients of relative precision. The valuation is
    // discovered exactly: when `point` is a pole of num/den the result carries a genuine
    // negative principal part. The returned series has order_min == v (the valuation, the
    // order of the pole negated) and truncation_order == v + order, i.e. exactly `order`
    // tracked coefficients c_v..c_{v+order-1}. den must not be the zero polynomial
    // (division_by_zero); order == 0 -> domain_error. If num vanishes identically the
    // result is the zero series 0 + O(x^order) at order_min 0.
    [[nodiscard]] static auto from_rational_function(const RationalPoly& num,
                                                     const RationalPoly& den,
                                                     const Rational& point, std::size_t order)
        -> Result<Laurent>;

    // --- Accessors -------------------------------------------------------------
    // Lowest tracked exponent (the coefficient of x^{order_min()} is coefficients()[0]).
    // May be negative. Exponents strictly below this are genuine zeros.
    [[nodiscard]] auto order_min() const noexcept -> std::int64_t { return order_min_; }
    // The exponent of the O() tail: coefficients at exponents >= this are UNKNOWN
    // (truncated). Equals order_min() + size().
    [[nodiscard]] auto truncation_order() const noexcept -> std::int64_t {
        return order_min_ + static_cast<std::int64_t>(coeffs_.size());
    }
    // Number of tracked coefficients (the relative precision).
    [[nodiscard]] auto size() const noexcept -> std::size_t { return coeffs_.size(); }
    // Coefficient of x^k. Returns 0 for k < order_min() (a genuine zero). Also returns 0
    // for k >= truncation_order(), but there the 0 is the *truncation*, not a known value
    // -- query truncation_order() to tell the two apart (mirrors PowerSeries::coefficient).
    [[nodiscard]] auto coefficient(std::int64_t k) const -> Rational;
    // Read-only view of the tracked coefficients (index i is the coefficient of
    // x^{order_min() + i}).
    [[nodiscard]] auto coefficients() const noexcept -> std::span<const Rational> {
        return coeffs_;
    }
    // Human-readable form, e.g. "x^-1 + 1 + x + O(x^3)".
    [[nodiscard]] auto to_string(std::string_view var = "x") const -> std::string;
    // Structural equality: same order_min and identical coefficient vector.
    [[nodiscard]] auto is_equal(const Laurent& o) const -> bool;

    // --- Structure -------------------------------------------------------------
    // Exponent of the first nonzero tracked coefficient. domain_error when the tracked
    // part is entirely zero (the valuation cannot be determined -- it is >= truncation
    // order, or the series is the zero series).
    [[nodiscard]] auto valuation() const -> Result<std::int64_t>;
    // The finite principal part: the terms of exponent < 0 (returned as a Laurent whose
    // truncation order is min(0, truncation_order())). The zero series 0 + O(x) when there
    // are no negative-power terms.
    [[nodiscard]] auto principal_part() const -> Result<Laurent>;
    // The regular (Taylor) part: the terms of exponent >= 0, carrying the original
    // O(x^{truncation_order()}) tail. The zero series 0 + O(x) when nothing of exponent
    // >= 0 is tracked.
    [[nodiscard]] auto regular_part() const -> Result<Laurent>;
    // The residue: the coefficient c_{-1}. A genuine 0 when order_min() > -1. domain_error
    // when x^{-1} lies at or beyond the truncation order (its value is unknown).
    [[nodiscard]] auto residue() const -> Result<Rational>;

    // --- Arithmetic ------------------------------------------------------------
    // Sum / difference over the union of tracked ranges: order_min becomes the smaller of
    // the two, and the truncation order the smaller of the two (a result is honest only up
    // to where BOTH inputs are known).
    [[nodiscard]] auto add(const Laurent& o) const -> Result<Laurent>;
    [[nodiscard]] auto subtract(const Laurent& o) const -> Result<Laurent>;
    // Multiply every coefficient by the scalar s (order_min and truncation unchanged).
    [[nodiscard]] auto scale(const Rational& s) const -> Result<Laurent>;
    // Cauchy product with valuation shift: order_min adds (order_min(a)+order_min(b)) and
    // the relative precision is min(size(a), size(b)) -- the honest number of coefficients
    // the truncated inputs jointly determine.
    [[nodiscard]] auto multiply(const Laurent& o) const -> Result<Laurent>;
    // Multiplicative inverse. Factors out the valuation v (Laurent = x^v * unit, unit a
    // power series with nonzero constant term) and inverts the unit via the PowerSeries
    // recurrence; the result has valuation -v. domain_error when the tracked part is
    // entirely zero (no invertible leading term).
    [[nodiscard]] auto inverse() const -> Result<Laurent>;
    // Division: this * o.inverse() (o's tracked part must have a nonzero leading term).
    [[nodiscard]] auto divide(const Laurent& o) const -> Result<Laurent>;

private:
    Laurent(std::int64_t order_min, std::vector<Rational> coeffs)
        : order_min_(order_min), coeffs_(std::move(coeffs)) {}

    std::int64_t order_min_{0};      // exponent of coeffs_[0]; may be negative
    std::vector<Rational> coeffs_;   // coeffs_[i] is the coefficient of x^{order_min_ + i}
};

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

constexpr std::int64_t int64_max = std::numeric_limits<std::int64_t>::max();
constexpr std::int64_t int64_min = std::numeric_limits<std::int64_t>::min();

// order_min + size must stay inside the int64 exponent range so truncation_order() and
// the exponent arithmetic below can never wrap.
[[nodiscard]] auto range_ok(std::int64_t order_min, std::size_t size) -> bool {
    if (size > static_cast<std::uint64_t>(int64_max)) {
        return false;
    }
    const auto n = static_cast<std::int64_t>(size);
    return !(order_min > int64_max - n);  // order_min + n <= int64_max
}

// Index of the first nonzero coefficient of a polynomial read as a series (its
// "trailing-zero" valuation about t = 0); nullopt for the zero polynomial.
[[nodiscard]] auto poly_valuation(const RationalPoly& p) -> std::optional<std::size_t> {
    const auto c = p.coefficients();
    for (std::size_t i = 0; i < c.size(); ++i) {
        if (!c[i].is_zero()) {
            return i;
        }
    }
    return std::nullopt;
}

// Taylor shift: return p(point + t) as a polynomial in t, exactly over Q. Evaluated by
// Horner in the linear polynomial (t + point): acc <- acc*(t + point) + p_i.
[[nodiscard]] auto taylor_shift(const RationalPoly& p, const Rational& point)
    -> Result<RationalPoly> {
    if (p.is_zero()) {
        return RationalPoly{};
    }
    const auto d = static_cast<std::size_t>(p.degree());  // degree() >= 0 here
    const RationalPoly t_plus_point =
        RationalPoly::from_coeffs({point, Rational::from_int(1)});  // point + t
    RationalPoly acc = RationalPoly::constant(p.coefficient(d));
    for (std::size_t step = 0; step < d; ++step) {
        const std::size_t i = d - 1 - step;
        auto mul = acc.multiply(t_plus_point);
        if (!mul) {
            return make_error<RationalPoly>(mul.error());
        }
        auto sum = mul->add(RationalPoly::constant(p.coefficient(i)));
        if (!sum) {
            return make_error<RationalPoly>(sum.error());
        }
        acc = std::move(*sum);
    }
    return acc;
}

}  // namespace

// --- Factories --------------------------------------------------------------

auto Laurent::from_coeffs(std::int64_t order_min, std::vector<Rational> coeffs)
    -> Result<Laurent> {
    if (coeffs.empty()) {
        return make_error<Laurent>(MathError::domain_error);
    }
    if (!range_ok(order_min, coeffs.size())) {
        return make_error<Laurent>(MathError::overflow);
    }
    return Laurent(order_min, std::move(coeffs));
}

auto Laurent::zero(std::int64_t order_min, std::size_t size) -> Result<Laurent> {
    if (size == 0) {
        return make_error<Laurent>(MathError::domain_error);
    }
    if (!range_ok(order_min, size)) {
        return make_error<Laurent>(MathError::overflow);
    }
    return Laurent(order_min, std::vector<Rational>(size));
}

auto Laurent::constant(const Rational& c, std::size_t size) -> Result<Laurent> {
    if (size == 0) {
        return make_error<Laurent>(MathError::domain_error);
    }
    std::vector<Rational> coeffs(size);
    coeffs[0] = c;
    return Laurent(0, std::move(coeffs));
}

auto Laurent::one(std::size_t size) -> Result<Laurent> {
    return constant(Rational::from_int(1), size);
}

auto Laurent::monomial(const Rational& c, std::int64_t exponent, std::int64_t order_min,
                       std::size_t size) -> Result<Laurent> {
    if (size == 0) {
        return make_error<Laurent>(MathError::domain_error);
    }
    if (!range_ok(order_min, size)) {
        return make_error<Laurent>(MathError::overflow);
    }
    const std::int64_t trunc = order_min + static_cast<std::int64_t>(size);
    if (exponent < order_min || exponent >= trunc) {
        return make_error<Laurent>(MathError::domain_error);
    }
    std::vector<Rational> coeffs(size);
    coeffs[static_cast<std::size_t>(exponent - order_min)] = c;
    return Laurent(order_min, std::move(coeffs));
}

auto Laurent::from_power_series(const PowerSeries& p) -> Result<Laurent> {
    const auto c = p.coefficients();
    return from_coeffs(0, std::vector<Rational>(c.begin(), c.end()));
}

auto Laurent::from_rational_function(const RationalPoly& num, const RationalPoly& den,
                                     const Rational& point, std::size_t order)
    -> Result<Laurent> {
    if (order == 0) {
        return make_error<Laurent>(MathError::domain_error);
    }
    if (den.is_zero()) {
        return make_error<Laurent>(MathError::division_by_zero);
    }
    auto ps = taylor_shift(num, point);  // num(point + t)
    if (!ps) {
        return make_error<Laurent>(ps.error());
    }
    auto qs = taylor_shift(den, point);  // den(point + t) (nonzero: den nonzero)
    if (!qs) {
        return make_error<Laurent>(qs.error());
    }
    const auto vq_opt = poly_valuation(*qs);
    if (!vq_opt) {
        // den(point + t) identically zero is impossible for a nonzero den, but guard.
        return make_error<Laurent>(MathError::division_by_zero);
    }
    const auto vp_opt = poly_valuation(*ps);
    if (!vp_opt) {
        // Numerator vanishes identically -> the quotient is the zero series.
        return zero(0, order);
    }
    const std::size_t vp = *vp_opt;
    const std::size_t vq = *vq_opt;

    // Factor out the t^vp / t^vq units: N = ps / t^vp, D = qs / t^vq, each with nonzero
    // constant term. Build them as PowerSeries of relative precision `order`.
    const auto pc = ps->coefficients();
    const auto qc = qs->coefficients();
    std::vector<Rational> n_terms(order);
    std::vector<Rational> d_terms(order);
    for (std::size_t i = 0; i < order; ++i) {
        if (vp + i < pc.size()) {
            n_terms[i] = pc[vp + i];
        }
        if (vq + i < qc.size()) {
            d_terms[i] = qc[vq + i];
        }
    }
    auto n_ps = PowerSeries::from_coeffs(std::move(n_terms), order);
    if (!n_ps) {
        return make_error<Laurent>(n_ps.error());
    }
    auto d_ps = PowerSeries::from_coeffs(std::move(d_terms), order);
    if (!d_ps) {
        return make_error<Laurent>(d_ps.error());
    }
    auto unit = n_ps->divide(*d_ps);  // N / D, a power series to `order` terms (D_0 != 0)
    if (!unit) {
        return make_error<Laurent>(unit.error());
    }
    // Valuation of the quotient: v = vp - vq. Both are small polynomial multiplicities.
    const std::int64_t v = static_cast<std::int64_t>(vp) - static_cast<std::int64_t>(vq);
    const auto uc = unit->coefficients();
    return from_coeffs(v, std::vector<Rational>(uc.begin(), uc.end()));
}

// --- Accessors --------------------------------------------------------------

auto Laurent::coefficient(std::int64_t k) const -> Rational {
    if (k < order_min_ || k >= truncation_order()) {
        return Rational::from_int(0);
    }
    return coeffs_[static_cast<std::size_t>(k - order_min_)];
}

auto Laurent::is_equal(const Laurent& o) const -> bool {
    if (order_min_ != o.order_min_ || coeffs_.size() != o.coeffs_.size()) {
        return false;
    }
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        if (!(coeffs_[i] == o.coeffs_[i])) {
            return false;
        }
    }
    return true;
}

auto Laurent::to_string(std::string_view var) const -> std::string {
    std::string out;
    bool first = true;
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        if (coeffs_[i].is_zero()) {
            continue;
        }
        const std::int64_t e = order_min_ + static_cast<std::int64_t>(i);
        if (!first) {
            out += " + ";
        }
        first = false;
        if (e == 0) {
            out += coeffs_[i].to_string();
        } else {
            if (!(coeffs_[i] == Rational::from_int(1))) {
                out += coeffs_[i].to_string();
                out += "*";
            }
            out += std::string(var);
            if (e != 1) {
                out += "^";
                out += std::to_string(e);
            }
        }
    }
    if (first) {
        out += "0";
    }
    out += " + O(";
    out += std::string(var);
    out += "^";
    out += std::to_string(truncation_order());
    out += ")";
    return out;
}

// --- Structure --------------------------------------------------------------

auto Laurent::valuation() const -> Result<std::int64_t> {
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        if (!coeffs_[i].is_zero()) {
            return order_min_ + static_cast<std::int64_t>(i);
        }
    }
    return make_error<std::int64_t>(MathError::domain_error);
}

auto Laurent::principal_part() const -> Result<Laurent> {
    const std::int64_t hi = std::min<std::int64_t>(0, truncation_order());  // exclusive
    if (order_min_ >= hi) {
        return zero(0, 1);  // no negative-power terms tracked
    }
    const auto len = static_cast<std::size_t>(hi - order_min_);
    std::vector<Rational> coeffs(coeffs_.begin(), coeffs_.begin() + static_cast<std::ptrdiff_t>(len));
    return from_coeffs(order_min_, std::move(coeffs));
}

auto Laurent::regular_part() const -> Result<Laurent> {
    const std::int64_t trunc = truncation_order();
    if (trunc <= 0) {
        return zero(0, 1);  // nothing of exponent >= 0 is tracked
    }
    const std::int64_t lo = std::max<std::int64_t>(order_min_, 0);
    const auto start = static_cast<std::size_t>(lo - order_min_);
    std::vector<Rational> coeffs(coeffs_.begin() + static_cast<std::ptrdiff_t>(start), coeffs_.end());
    return from_coeffs(lo, std::move(coeffs));
}

auto Laurent::residue() const -> Result<Rational> {
    constexpr std::int64_t k = -1;
    if (k < order_min_) {
        return Rational::from_int(0);  // genuine zero: below the (finite) principal part
    }
    if (k >= truncation_order()) {
        return make_error<Rational>(MathError::domain_error);  // x^{-1} beyond tracked order
    }
    return coeffs_[static_cast<std::size_t>(k - order_min_)];
}

// --- Arithmetic -------------------------------------------------------------

auto Laurent::add(const Laurent& o) const -> Result<Laurent> {
    const std::int64_t lo = std::min(order_min_, o.order_min_);
    const std::int64_t hi = std::min(truncation_order(), o.truncation_order());
    // hi > lo always: hi = min(T_a, T_b) and T_x > order_min_x >= lo.
    const auto len = static_cast<std::size_t>(hi - lo);
    std::vector<Rational> r(len);
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        const std::int64_t e = order_min_ + static_cast<std::int64_t>(i);
        if (e >= hi) {
            continue;
        }
        r[static_cast<std::size_t>(e - lo)] = coeffs_[i];
    }
    for (std::size_t i = 0; i < o.coeffs_.size(); ++i) {
        const std::int64_t e = o.order_min_ + static_cast<std::int64_t>(i);
        if (e >= hi) {
            continue;
        }
        auto sum = r[static_cast<std::size_t>(e - lo)].add(o.coeffs_[i]);
        if (!sum) {
            return make_error<Laurent>(sum.error());
        }
        r[static_cast<std::size_t>(e - lo)] = *sum;
    }
    return Laurent(lo, std::move(r));
}

auto Laurent::subtract(const Laurent& o) const -> Result<Laurent> {
    const std::int64_t lo = std::min(order_min_, o.order_min_);
    const std::int64_t hi = std::min(truncation_order(), o.truncation_order());
    const auto len = static_cast<std::size_t>(hi - lo);
    std::vector<Rational> r(len);
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        const std::int64_t e = order_min_ + static_cast<std::int64_t>(i);
        if (e >= hi) {
            continue;
        }
        r[static_cast<std::size_t>(e - lo)] = coeffs_[i];
    }
    for (std::size_t i = 0; i < o.coeffs_.size(); ++i) {
        const std::int64_t e = o.order_min_ + static_cast<std::int64_t>(i);
        if (e >= hi) {
            continue;
        }
        auto diff = r[static_cast<std::size_t>(e - lo)].subtract(o.coeffs_[i]);
        if (!diff) {
            return make_error<Laurent>(diff.error());
        }
        r[static_cast<std::size_t>(e - lo)] = *diff;
    }
    return Laurent(lo, std::move(r));
}

auto Laurent::scale(const Rational& s) const -> Result<Laurent> {
    std::vector<Rational> r(coeffs_.size());
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        auto p = coeffs_[i].multiply(s);
        if (!p) {
            return make_error<Laurent>(p.error());
        }
        r[i] = *p;
    }
    return Laurent(order_min_, std::move(r));
}

auto Laurent::multiply(const Laurent& o) const -> Result<Laurent> {
    // Valuation shift: the product starts at order_min_ + o.order_min_. Guard that sum.
    if ((o.order_min_ > 0 && order_min_ > int64_max - o.order_min_) ||
        (o.order_min_ < 0 && order_min_ < int64_min - o.order_min_)) {
        return make_error<Laurent>(MathError::overflow);
    }
    const std::int64_t lo = order_min_ + o.order_min_;
    // Relative precision of a truncated Cauchy product is the smaller of the two.
    const std::size_t len = std::min(coeffs_.size(), o.coeffs_.size());
    if (!range_ok(lo, len)) {
        return make_error<Laurent>(MathError::overflow);
    }
    std::vector<Rational> r(len);  // all zero
    for (std::size_t i = 0; i < coeffs_.size(); ++i) {
        for (std::size_t j = 0; j < o.coeffs_.size(); ++j) {
            const std::size_t m = i + j;
            if (m >= len) {
                continue;
            }
            auto prod = coeffs_[i].multiply(o.coeffs_[j]);
            if (!prod) {
                return make_error<Laurent>(prod.error());
            }
            auto sum = r[m].add(*prod);
            if (!sum) {
                return make_error<Laurent>(sum.error());
            }
            r[m] = *sum;
        }
    }
    return Laurent(lo, std::move(r));
}

auto Laurent::inverse() const -> Result<Laurent> {
    // Find the valuation index: Laurent = x^v * unit, unit a power series with nonzero
    // constant term. The inverse is x^{-v} * unit^{-1}.
    std::size_t j = 0;
    while (j < coeffs_.size() && coeffs_[j].is_zero()) {
        ++j;
    }
    if (j == coeffs_.size()) {
        return make_error<Laurent>(MathError::domain_error);  // no invertible leading term
    }
    const std::int64_t v = order_min_ + static_cast<std::int64_t>(j);
    if (v == int64_min) {
        return make_error<Laurent>(MathError::overflow);  // -v would wrap
    }
    const std::size_t m = coeffs_.size() - j;  // relative precision of the unit (>= 1)
    std::vector<Rational> unit_coeffs(coeffs_.begin() + static_cast<std::ptrdiff_t>(j), coeffs_.end());
    auto unit = PowerSeries::from_coeffs(std::move(unit_coeffs), m);
    if (!unit) {
        return make_error<Laurent>(unit.error());
    }
    auto inv = unit->inverse();  // unit c_0 != 0, so the domain test passes
    if (!inv) {
        return make_error<Laurent>(inv.error());
    }
    const auto ic = inv->coefficients();
    return from_coeffs(-v, std::vector<Rational>(ic.begin(), ic.end()));
}

auto Laurent::divide(const Laurent& o) const -> Result<Laurent> {
    auto inv = o.inverse();
    if (!inv) {
        return make_error<Laurent>(inv.error());
    }
    return multiply(*inv);
}

}  // namespace nimblecas
