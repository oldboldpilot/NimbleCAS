// NimbleCAS Rothstein-Trager logarithmic integration over Q(x) (ROADMAP 7.19).
// @author Olumuyiwa Oluwasanmi
//
// The Rothstein-Trager theorem computes the logarithmic part of the integral of a
// rational function A(x)/D(x) with D square-free and deg A < deg D:
//
//     int A/D dx = sum_{c : R(c) = 0}  c * log( gcd_x(A - c*D', D) ),
//
// where R(t) = res_x(D, A - t*D') is the Rothstein-Trager resultant. Its distinct roots
// c are the residues (the constant multipliers of the logarithms), and each residue's
// logarithm argument is the gcd of D with A - c*D'.
//
// This module computes R(t) exactly over Q by evaluating the scalar resultant
// (nimblecas.resultant) at deg D + 1 integer points and Lagrange-interpolating, then
// finds R's rational roots by the rational-root theorem. When every residue is rational
// the logarithmic part is returned in full; when a residue is irrational or complex
// (R has a non-rational factor) the answer needs an algebraic extension field, which is
// not yet implemented, so log_part returns MathError::not_implemented. Every rational-case
// operation is exact and overflow-checked (Result / MathError, Rule 32).

export module nimblecas.rothstein;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.resultant;

export namespace nimblecas {

// One logarithmic summand coefficient * log(argument), with argument monic of degree >= 1.
struct LogTerm {
    Rational coefficient;    // c — a rational residue
    RationalPoly argument;   // gcd(A - c*D', D), monic, deg >= 1
};

// int A/D dx = sum over terms of coefficient * log(argument).
struct LogarithmicPart {
    std::vector<LogTerm> terms;
};

// Rothstein-Trager logarithmic integration of A/D over Q, with D square-free and
// deg A < deg D. Fails with division_by_zero (D == 0), not_implemented (an improper
// input, or a residue that is not rational), or overflow (an int64 coefficient limit).
[[nodiscard]] auto log_part(const RationalPoly& numerator, const RationalPoly& denominator)
    -> Result<LogarithmicPart>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

[[nodiscard]] auto one_poly() -> RationalPoly {
    return RationalPoly::constant(Rational::from_int(1));
}

[[nodiscard]] auto exact_quotient(const RationalPoly& a, const RationalPoly& b)
    -> Result<RationalPoly> {
    auto dm = a.divide(b);
    if (!dm) {
        return make_error<RationalPoly>(dm.error());
    }
    return std::move(dm->quotient);
}

// The linear polynomial (t - c).
[[nodiscard]] auto linear_factor(const Rational& c) -> Result<RationalPoly> {
    auto neg = c.negate();
    if (!neg) {
        return make_error<RationalPoly>(neg.error());
    }
    return RationalPoly::from_coeffs({*neg, Rational::from_int(1)});
}

// Evaluate p(x) by Horner's method.
[[nodiscard]] auto poly_eval(const RationalPoly& p, const Rational& x) -> Result<Rational> {
    const auto coeffs = p.coefficients();
    Rational acc;  // 0
    for (std::size_t i = coeffs.size(); i-- > 0;) {
        auto mul = acc.multiply(x);
        if (!mul) {
            return mul;
        }
        auto sum = mul->add(coeffs[i]);
        if (!sum) {
            return sum;
        }
        acc = *sum;
    }
    return acc;
}

// Lagrange interpolation through the points (xs[i], ys[i]) with distinct xs.
[[nodiscard]] auto interpolate(const std::vector<Rational>& xs, const std::vector<Rational>& ys)
    -> Result<RationalPoly> {
    RationalPoly result;  // zero
    const std::size_t n = xs.size();
    for (std::size_t i = 0; i < n; ++i) {
        RationalPoly term = RationalPoly::constant(ys[i]);
        Rational denom = Rational::from_int(1);
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            auto lf = linear_factor(xs[j]);  // (t - xs[j])
            if (!lf) {
                return make_error<RationalPoly>(lf.error());
            }
            auto prod = term.multiply(*lf);
            if (!prod) {
                return prod;
            }
            term = std::move(*prod);
            auto diff = xs[i].subtract(xs[j]);  // xs[i] - xs[j] (nonzero: distinct)
            if (!diff) {
                return make_error<RationalPoly>(diff.error());
            }
            auto d = denom.multiply(*diff);
            if (!d) {
                return make_error<RationalPoly>(d.error());
            }
            denom = *d;
        }
        auto inv = Rational::from_int(1).divide(denom);
        if (!inv) {
            return make_error<RationalPoly>(inv.error());
        }
        auto scaled = term.scale(*inv);
        if (!scaled) {
            return scaled;
        }
        auto sum = result.add(*scaled);
        if (!sum) {
            return sum;
        }
        result = std::move(*sum);
    }
    return result;
}

constexpr std::int64_t int64_min = std::numeric_limits<std::int64_t>::min();

// std::lcm is UB near INT64_MIN and can overflow; do it checked. Denominators are always
// positive (Rational's invariant), so the result is positive unless it overflows.
[[nodiscard]] auto checked_lcm(std::int64_t a, std::int64_t b) -> std::optional<std::int64_t> {
    if (a == 0 || b == 0) {
        return std::int64_t{0};
    }
    const std::int64_t g = std::gcd(a, b);
    std::int64_t r = 0;
    if (__builtin_mul_overflow(a / g, b, &r) || r == int64_min) {
        return std::nullopt;  // r == INT64_MIN would make |r| unrepresentable
    }
    return r < 0 ? -r : r;
}

// Clear denominators: return integer coefficients proportional to p's (same roots).
[[nodiscard]] auto to_integer_coeffs(const RationalPoly& p) -> Result<std::vector<std::int64_t>> {
    const auto coeffs = p.coefficients();
    std::int64_t l = 1;
    for (const Rational& c : coeffs) {
        auto next = checked_lcm(l, c.denominator());
        if (!next) {
            return make_error<std::vector<std::int64_t>>(MathError::overflow);
        }
        l = *next;
    }
    std::vector<std::int64_t> out(coeffs.size());
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        std::int64_t v = 0;
        // coeff * l = numerator * (l / denominator), exact since denominator | l.
        if (__builtin_mul_overflow(coeffs[i].numerator(), l / coeffs[i].denominator(), &v) ||
            v == int64_min) {
            return make_error<std::vector<std::int64_t>>(MathError::overflow);  // |v| must fit
        }
        out[i] = v;
    }
    return out;
}

// Positive divisors of |m| (m != 0).
[[nodiscard]] auto divisors(std::int64_t m) -> std::vector<std::int64_t> {
    const std::int64_t a = m < 0 ? -m : m;
    std::vector<std::int64_t> out;
    for (std::int64_t d = 1; d * d <= a; ++d) {
        if (a % d == 0) {
            out.push_back(d);
            if (d != a / d) {
                out.push_back(a / d);
            }
        }
    }
    return out;
}

// Distinct rational roots of R via the rational-root theorem, testing each candidate.
[[nodiscard]] auto rational_roots(const RationalPoly& r) -> Result<std::vector<Rational>> {
    auto ints = to_integer_coeffs(r);
    if (!ints) {
        return make_error<std::vector<Rational>>(ints.error());
    }
    std::vector<Rational> roots;
    const std::int64_t an = ints->back();  // leading coeff (nonzero: R is trimmed)
    // If the constant term is zero, 0 is a root (it contributes nothing to the integral,
    // but is recorded so the completeness check accounts for it).
    if (ints->front() == 0) {
        roots.push_back(Rational{});
    }
    // A nonzero rational root p/q (lowest terms) has p | a_lo, the lowest-degree nonzero
    // coefficient (== the constant term when that is nonzero), and q | a_n.
    std::int64_t a_lo = an;
    for (const std::int64_t v : *ints) {
        if (v != 0) {
            a_lo = v;
            break;
        }
    }
    const std::vector<std::int64_t> ps = divisors(a_lo);
    const std::vector<std::int64_t> qs = divisors(an);
    for (const std::int64_t p : ps) {
        for (const std::int64_t q : qs) {
            for (const std::int64_t sign : {std::int64_t{1}, std::int64_t{-1}}) {
                auto cand = Rational::make(sign * p, q);
                if (!cand) {
                    continue;  // p or q at an int64 boundary: skip this candidate
                }
                auto val = poly_eval(r, *cand);
                if (!val) {
                    return make_error<std::vector<Rational>>(val.error());
                }
                if (val->is_zero() &&
                    std::find(roots.begin(), roots.end(), *cand) == roots.end()) {
                    roots.push_back(*cand);
                }
            }
        }
    }
    return roots;
}

}  // namespace

auto log_part(const RationalPoly& numerator, const RationalPoly& denominator)
    -> Result<LogarithmicPart> {
    if (denominator.is_zero()) {
        return make_error<LogarithmicPart>(MathError::division_by_zero);
    }
    if (numerator.is_zero()) {
        return LogarithmicPart{};  // integral of 0 has no logarithmic part
    }
    // Reduce A/D to lowest terms so gcd(A, D) == 1 (D stays square-free as a divisor).
    auto g = numerator.gcd(denominator);
    if (!g) {
        return make_error<LogarithmicPart>(g.error());
    }
    RationalPoly a = numerator;
    RationalPoly d = denominator;
    if (g->degree() >= 1) {
        auto ar = exact_quotient(numerator, *g);
        if (!ar) {
            return make_error<LogarithmicPart>(ar.error());
        }
        auto dr = exact_quotient(denominator, *g);
        if (!dr) {
            return make_error<LogarithmicPart>(dr.error());
        }
        a = std::move(*ar);
        d = std::move(*dr);
    }
    // Make D monic, folding its leading constant into A (value A/D unchanged).
    const Rational lc = d.leading_coefficient();
    auto dm = d.monic();
    if (!dm) {
        return make_error<LogarithmicPart>(dm.error());
    }
    auto inv_lc = Rational::from_int(1).divide(lc);
    if (!inv_lc) {
        return make_error<LogarithmicPart>(inv_lc.error());
    }
    auto am = a.scale(*inv_lc);
    if (!am) {
        return make_error<LogarithmicPart>(am.error());
    }
    a = std::move(*am);
    if (a.degree() >= dm->degree()) {
        return make_error<LogarithmicPart>(MathError::not_implemented);  // must be proper
    }

    auto dprime = dm->derivative();
    if (!dprime) {
        return make_error<LogarithmicPart>(dprime.error());
    }
    const std::int64_t n = dm->degree();
    // R(t) = res_x(D, A - t*D'), degree <= n in t: sample at t = 0, 1, ..., n and interpolate.
    std::vector<Rational> xs;
    std::vector<Rational> ys;
    xs.reserve(static_cast<std::size_t>(n) + 1);
    ys.reserve(static_cast<std::size_t>(n) + 1);
    for (std::int64_t k = 0; k <= n; ++k) {
        const Rational tk = Rational::from_int(k);
        auto scaled = dprime->scale(tk);
        if (!scaled) {
            return make_error<LogarithmicPart>(scaled.error());
        }
        auto bk = a.subtract(*scaled);  // A - tk*D'
        if (!bk) {
            return make_error<LogarithmicPart>(bk.error());
        }
        auto rk = resultant(*dm, *bk);
        if (!rk) {
            return make_error<LogarithmicPart>(rk.error());
        }
        xs.push_back(tk);
        ys.push_back(*rk);
    }
    auto rpoly = interpolate(xs, ys);
    if (!rpoly) {
        return make_error<LogarithmicPart>(rpoly.error());
    }

    // Distinct rational roots of R(t) are the rational residues.
    auto roots = rational_roots(*rpoly);
    if (!roots) {
        return make_error<LogarithmicPart>(roots.error());
    }
    // Completeness: strip every (t - c) factor; a non-constant remainder means R has a
    // non-rational root, i.e. an irrational/complex residue this pass cannot express.
    RationalPoly remaining = *rpoly;
    for (const Rational& c : *roots) {
        auto lf = linear_factor(c);
        if (!lf) {
            return make_error<LogarithmicPart>(lf.error());
        }
        while (remaining.degree() >= 1) {
            auto div = remaining.divide(*lf);
            if (!div) {
                return make_error<LogarithmicPart>(div.error());
            }
            if (!div->remainder.is_zero()) {
                break;
            }
            remaining = std::move(div->quotient);
        }
    }
    if (remaining.degree() >= 1) {
        return make_error<LogarithmicPart>(MathError::not_implemented);  // irrational residue
    }

    // Assemble the logarithmic part: c * log(gcd(A - c*D', D)) for each nonzero residue.
    LogarithmicPart result;
    for (const Rational& c : *roots) {
        if (c.is_zero()) {
            continue;  // residue 0 contributes 0 * log(...) = 0
        }
        auto cd = dprime->scale(c);
        if (!cd) {
            return make_error<LogarithmicPart>(cd.error());
        }
        auto shifted = a.subtract(*cd);  // A - c*D'
        if (!shifted) {
            return make_error<LogarithmicPart>(shifted.error());
        }
        auto arg = shifted->gcd(*dm);  // monic gcd, the logarithm argument
        if (!arg) {
            return make_error<LogarithmicPart>(arg.error());
        }
        if (arg->degree() >= 1) {
            result.terms.push_back({.coefficient = c, .argument = std::move(*arg)});
        }
    }
    return result;
}

}  // namespace nimblecas
