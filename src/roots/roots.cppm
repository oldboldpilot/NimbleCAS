// NimbleCAS analytical equation solving: rational roots of a polynomial (ROADMAP 7.21).
// @author Olumuyiwa Oluwasanmi
//
// A first slice of analytical equation solving. Given p in Q[x], rational_roots()
// returns every RATIONAL root of p together with its multiplicity, via the rational
// root theorem: clear denominators to an integer polynomial a_n x^n + ... + a_0, and
// note that any rational root written p/q in lowest terms must have p dividing the
// lowest nonzero coefficient and q dividing the leading coefficient a_n. Each candidate
// +/-(divisor of a_0)/(divisor of a_n) is tested by exact evaluation (Horner), and a
// confirmed root r is deflated out by dividing (x - r) repeatedly to recover its
// multiplicity.
//
// SCOPE: only RATIONAL roots are returned. Irrational and complex roots — radical /
// Cardano (cubic) / Ferrari (quartic) closed forms, and a symbolic RootOf for the
// unsolvable-in-radicals case — are a planned extension and are not produced here. A
// polynomial such as x^2 + 1 or x^2 - 2 therefore reports no roots.
//
// Following the rest of the engine, arithmetic is exact and overflow-checked: clearing
// denominators and forming candidate magnitudes use __builtin_mul_overflow and reject an
// int64 boundary (such as INT64_MIN, whose magnitude is unrepresentable) with
// MathError::overflow rather than wrapping (Rule 32).

module;
#include <cassert>

export module nimblecas.roots;

import std;
import nimblecas.core;
import nimblecas.ratpoly;

export namespace nimblecas {

// Evaluate p at x by Horner's method (the zero polynomial evaluates to 0). Fails only
// on overflow of the exact int64 rational arithmetic.
[[nodiscard]] auto evaluate(const RationalPoly& p, const Rational& x) -> Result<Rational>;

// All distinct rational roots of p, each paired with its multiplicity (>= 1). The zero
// polynomial has every value as a root and so is rejected with MathError::domain_error;
// a nonzero constant has no roots and yields an empty vector. Roots are returned in no
// particular order. Only rational roots are found (see the module header on scope).
[[nodiscard]] auto rational_roots(const RationalPoly& p)
    -> Result<std::vector<std::pair<Rational, std::int64_t>>>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

constexpr std::int64_t int64_min = std::numeric_limits<std::int64_t>::min();

[[nodiscard]] auto mul_ov(std::int64_t a, std::int64_t b, std::int64_t& out) -> bool {
    return __builtin_mul_overflow(a, b, &out);
}

// |v| as a non-negative int64, or nullopt when v == INT64_MIN (whose magnitude is
// unrepresentable — the same guard ratpoly's checked_gcd applies).
[[nodiscard]] auto magnitude(std::int64_t v) -> std::optional<std::int64_t> {
    if (v == int64_min) {
        return std::nullopt;
    }
    return v < 0 ? -v : v;
}

// Clear denominators: return integer coefficients (low degree first) of the polynomial
// obtained by multiplying p by the LCM of its denominators. p must be non-empty. Fails
// with overflow if the LCM or a scaled numerator exceeds int64.
[[nodiscard]] auto clear_denominators(const RationalPoly& p) -> Result<std::vector<std::int64_t>> {
    const std::span<const Rational> coeffs = p.coefficients();
    // lcm of all (positive) denominators.
    std::int64_t lcm = 1;
    for (const Rational& c : coeffs) {
        const std::int64_t d = c.denominator();  // canonical: d > 0
        const std::int64_t g = std::gcd(lcm, d);  // both positive => no INT64_MIN
        std::int64_t next = 0;
        if (mul_ov(lcm / g, d, next)) {
            return make_error<std::vector<std::int64_t>>(MathError::overflow);
        }
        lcm = next;
    }
    std::vector<std::int64_t> out(coeffs.size());
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        const std::int64_t factor = lcm / coeffs[i].denominator();  // exact (denominator | lcm)
        std::int64_t v = 0;
        if (mul_ov(coeffs[i].numerator(), factor, v)) {
            return make_error<std::vector<std::int64_t>>(MathError::overflow);
        }
        out[i] = v;
    }
    return out;
}

// Positive divisors of n (n > 0), unsorted, no duplicates.
[[nodiscard]] auto divisors(std::int64_t n) -> std::vector<std::int64_t> {
    assert(n > 0 && "divisors requires a positive argument");
    std::vector<std::int64_t> ds;
    for (std::int64_t i = 1; i * i <= n; ++i) {
        if (n % i == 0) {
            ds.push_back(i);
            if (i != n / i) {
                ds.push_back(n / i);
            }
        }
    }
    return ds;
}

}  // namespace

auto evaluate(const RationalPoly& p, const Rational& x) -> Result<Rational> {
    const std::span<const Rational> coeffs = p.coefficients();
    Rational acc{};  // 0/1
    // Horner from the highest-degree coefficient down: acc = acc*x + c_i.
    for (std::size_t i = coeffs.size(); i-- > 0;) {
        auto scaled = acc.multiply(x);
        if (!scaled) {
            return scaled;
        }
        auto sum = scaled->add(coeffs[i]);
        if (!sum) {
            return sum;
        }
        acc = *sum;
    }
    return acc;
}

auto rational_roots(const RationalPoly& p)
    -> Result<std::vector<std::pair<Rational, std::int64_t>>> {
    using Roots = std::vector<std::pair<Rational, std::int64_t>>;
    if (p.is_zero()) {
        return make_error<Roots>(MathError::domain_error);  // every value is a root
    }
    if (p.degree() == 0) {
        return Roots{};  // nonzero constant: no roots
    }

    // Integer coefficients a_0 .. a_n; a_n != 0 (p is trimmed), so we can read the
    // leading coefficient and the lowest nonzero coefficient off this vector.
    auto ints = clear_denominators(p);
    if (!ints) {
        return make_error<Roots>(ints.error());
    }
    const std::vector<std::int64_t>& a = *ints;

    // Lowest nonzero coefficient: its magnitude supplies the numerator candidates. A zero
    // constant term (a[0] == 0) means 0 is a root; add it as an explicit candidate.
    std::size_t low = 0;
    while (a[low] == 0) {
        ++low;  // guaranteed to stop: a.back() != 0
    }
    const auto num_mag = magnitude(a[low]);
    const auto den_mag = magnitude(a.back());
    if (!num_mag || !den_mag) {
        return make_error<Roots>(MathError::overflow);
    }

    std::vector<Rational> candidates;
    if (low > 0) {
        candidates.push_back(Rational{});  // 0 is a root of x^k * (...)
    }
    const std::vector<std::int64_t> ps = divisors(*num_mag);
    const std::vector<std::int64_t> qs = divisors(*den_mag);
    for (const std::int64_t pn : ps) {
        for (const std::int64_t qd : qs) {
            auto pos = Rational::make(pn, qd);
            auto neg = Rational::make(-pn, qd);
            if (!pos || !neg) {
                return make_error<Roots>(pos ? neg.error() : pos.error());
            }
            candidates.push_back(*pos);
            candidates.push_back(*neg);
        }
    }

    Roots roots;
    RationalPoly work = p;
    const Rational one = Rational::from_int(1);
    for (const Rational& r : candidates) {
        // Test by exact evaluation before deflating. A candidate whose root was already
        // removed (e.g. a non-reduced duplicate) no longer evaluates to zero here.
        auto val = evaluate(work, r);
        if (!val) {
            return make_error<Roots>(val.error());
        }
        if (!val->is_zero()) {
            continue;
        }
        // Deflate (x - r) out repeatedly to recover the multiplicity.
        auto neg_r = r.negate();
        if (!neg_r) {
            return make_error<Roots>(neg_r.error());
        }
        const RationalPoly factor = RationalPoly::from_coeffs({*neg_r, one});  // x - r
        std::int64_t mult = 0;
        while (true) {
            auto dm = work.divide(factor);
            if (!dm) {
                return make_error<Roots>(dm.error());
            }
            if (!dm->remainder.is_zero()) {
                break;
            }
            ++mult;
            work = std::move(dm->quotient);
        }
        assert(mult >= 1 && "evaluation confirmed the root, so it must deflate at least once");
        roots.emplace_back(r, mult);
    }
    return roots;
}

}  // namespace nimblecas
