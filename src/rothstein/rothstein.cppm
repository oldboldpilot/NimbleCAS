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
// (R has a non-rational factor) log_part returns MathError::not_implemented, since it
// answers purely in terms of rational LogTerms.
//
// log_part_extended lifts that restriction: it factors R(t) completely over Q
// (nimblecas.factor) and, for each irreducible factor of degree >= 2, builds the number
// field K = Q(alpha) that irreducible factor generates (nimblecas.algnum) and computes the
// logarithm argument as a gcd of A - alpha*D' and D over K[x] (nimblecas.algpoly). The
// result is a conjugate-sum block: alpha ranges implicitly over every root (conjugate) of
// that irreducible factor, alpha's canonical representative being field.generator(). A
// completeness identity (sum of field-degree * argument-degree over every term equals
// deg D) is checked before returning, so log_part_extended never emits a partial answer.
//
// Every operation is exact and overflow-checked (Result / MathError, Rule 32).

export module nimblecas.rothstein;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.resultant;
import nimblecas.factor;
import nimblecas.algnum;
import nimblecas.algpoly;

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

// A conjugate-sum block of the logarithmic part contributed by an irreducible factor of
// the Rothstein-Trager resultant R(t) of degree >= 2: the residues are the FULL set of
// conjugates beta of `residue` (the roots of field.modulus()), and the block stands for
//     sum over conjugates beta of R_i  of  beta * log( argument with alpha -> beta ),
// where `argument` is a single monic polynomial in x over field == Q(alpha), deg >= 1.
// `residue` is always field.generator() (alpha itself, the canonical representative).
struct AlgebraicLogTerm {
    NumberField field;
    AlgebraicNumber residue;
    AlgebraicPoly argument;
};

// The logarithmic part with irrational/complex residues expressed exactly via algebraic
// extension fields instead of being punted to not_implemented: a rational-residue term
// for every degree-1 irreducible factor of R(t), and a conjugate-sum block for every
// irreducible factor of degree >= 2.
struct ExtendedLogarithmicPart {
    std::vector<LogTerm> rational_terms;
    std::vector<AlgebraicLogTerm> algebraic_terms;
};

// Rothstein-Trager logarithmic integration of A/D over Q, with D square-free and
// deg A < deg D. Fails with division_by_zero (D == 0), not_implemented (an improper
// input, or a residue that is not rational), or overflow (an int64 coefficient limit).
[[nodiscard]] auto log_part(const RationalPoly& numerator, const RationalPoly& denominator)
    -> Result<LogarithmicPart>;

// Rothstein-Trager logarithmic integration of A/D over Q, expressing every residue
// exactly: rational residues become `rational_terms` (identical to log_part's output when
// every residue happens to be rational) and irrational/complex residues become conjugate-
// sum blocks in `algebraic_terms` over the number field each one generates. Fails with
// division_by_zero (D == 0), not_implemented (an improper input, or the internal
// factorization exhausting its search budget), domain_error (an internal completeness
// check failed — never returned alongside a partial result), or overflow.
[[nodiscard]] auto log_part_extended(const RationalPoly& numerator,
                                      const RationalPoly& denominator)
    -> Result<ExtendedLogarithmicPart>;

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

// The shared reduction + Rothstein-Trager resultant construction, factored out so both
// log_part and log_part_extended run it identically instead of duplicating the
// resultant/interpolation pipeline. Reduces A/D to lowest terms, makes D monic (folding
// its leading constant into A), verifies the input is proper, and builds
// R(t) = res_x(D, A - t*D') by evaluating the resultant at t = 0..deg(D) and
// Lagrange-interpolating.
struct ReducedProblem {
    RationalPoly a;       // numerator, reduced, adjusted so D is monic
    RationalPoly d;       // denominator, monic
    RationalPoly dprime;  // D'
    RationalPoly r;        // R(t) = res_x(D, A - t*D'), the Rothstein-Trager resultant
};

[[nodiscard]] auto build_reduced_problem(const RationalPoly& numerator,
                                          const RationalPoly& denominator)
    -> Result<ReducedProblem> {
    // Reduce A/D to lowest terms so gcd(A, D) == 1 (D stays square-free as a divisor).
    auto g = numerator.gcd(denominator);
    if (!g) {
        return make_error<ReducedProblem>(g.error());
    }
    RationalPoly a = numerator;
    RationalPoly d = denominator;
    if (g->degree() >= 1) {
        auto ar = exact_quotient(numerator, *g);
        if (!ar) {
            return make_error<ReducedProblem>(ar.error());
        }
        auto dr = exact_quotient(denominator, *g);
        if (!dr) {
            return make_error<ReducedProblem>(dr.error());
        }
        a = std::move(*ar);
        d = std::move(*dr);
    }
    // Make D monic, folding its leading constant into A (value A/D unchanged).
    const Rational lc = d.leading_coefficient();
    auto dm = d.monic();
    if (!dm) {
        return make_error<ReducedProblem>(dm.error());
    }
    auto inv_lc = Rational::from_int(1).divide(lc);
    if (!inv_lc) {
        return make_error<ReducedProblem>(inv_lc.error());
    }
    auto am = a.scale(*inv_lc);
    if (!am) {
        return make_error<ReducedProblem>(am.error());
    }
    a = std::move(*am);
    if (a.degree() >= dm->degree()) {
        return make_error<ReducedProblem>(MathError::not_implemented);  // must be proper
    }
    d = std::move(*dm);

    auto dprime = d.derivative();
    if (!dprime) {
        return make_error<ReducedProblem>(dprime.error());
    }
    const std::int64_t n = d.degree();
    // R(t) = res_x(D, A - t*D'), degree <= n in t: sample at t = 0, 1, ..., n and interpolate.
    std::vector<Rational> xs;
    std::vector<Rational> ys;
    xs.reserve(static_cast<std::size_t>(n) + 1);
    ys.reserve(static_cast<std::size_t>(n) + 1);
    for (std::int64_t k = 0; k <= n; ++k) {
        const Rational tk = Rational::from_int(k);
        auto scaled = dprime->scale(tk);
        if (!scaled) {
            return make_error<ReducedProblem>(scaled.error());
        }
        auto bk = a.subtract(*scaled);  // A - tk*D'
        if (!bk) {
            return make_error<ReducedProblem>(bk.error());
        }
        auto rk = resultant(d, *bk);
        if (!rk) {
            return make_error<ReducedProblem>(rk.error());
        }
        xs.push_back(tk);
        ys.push_back(*rk);
    }
    auto rpoly = interpolate(xs, ys);
    if (!rpoly) {
        return make_error<ReducedProblem>(rpoly.error());
    }
    return ReducedProblem{.a = std::move(a), .d = std::move(d), .dprime = std::move(*dprime),
                          .r = std::move(*rpoly)};
}

// The per-residue rational log term c * log(gcd(A - c*D', D)), shared by log_part's final
// assembly and log_part_extended's degree-1 (rational-residue) branch. A zero residue
// contributes 0 * log(...) == 0, and a trivial (degree-0) gcd argument contributes nothing
// either — both cases return std::nullopt rather than a term, matching log_part exactly.
[[nodiscard]] auto rational_log_term(const RationalPoly& a, const RationalPoly& d,
                                      const RationalPoly& dprime, const Rational& c)
    -> Result<std::optional<LogTerm>> {
    if (c.is_zero()) {
        return std::optional<LogTerm>{std::nullopt};
    }
    auto cd = dprime.scale(c);
    if (!cd) {
        return make_error<std::optional<LogTerm>>(cd.error());
    }
    auto shifted = a.subtract(*cd);  // A - c*D'
    if (!shifted) {
        return make_error<std::optional<LogTerm>>(shifted.error());
    }
    auto arg = shifted->gcd(d);  // monic gcd, the logarithm argument
    if (!arg) {
        return make_error<std::optional<LogTerm>>(arg.error());
    }
    if (arg->degree() < 1) {
        return std::optional<LogTerm>{std::nullopt};
    }
    return std::optional<LogTerm>{LogTerm{.coefficient = c, .argument = std::move(*arg)}};
}

// Deterministic total order over Rationals (NOT numeric magnitude — just stable and
// overflow-free), used only to make factor_over_Q's no-particular-order output
// reproducible across runs.
[[nodiscard]] auto rational_less(const Rational& x, const Rational& y) -> bool {
    if (x.numerator() != y.numerator()) {
        return x.numerator() < y.numerator();
    }
    return x.denominator() < y.denominator();
}

// Deterministic total order over RationalPolys: degree ascending, then lexicographic by
// coefficient (constant term first).
[[nodiscard]] auto poly_less(const RationalPoly& x, const RationalPoly& y) -> bool {
    if (x.degree() != y.degree()) {
        return x.degree() < y.degree();
    }
    const std::span<const Rational> xc = x.coefficients();
    const std::span<const Rational> yc = y.coefficients();
    for (std::size_t i = 0; i < xc.size(); ++i) {
        if (!(xc[i] == yc[i])) {
            return rational_less(xc[i], yc[i]);
        }
    }
    return false;
}

// The rational root p/q of a degree-1 irreducible factor q*t - p returned by
// factor_over_Q (integer coefficients, q > 0). Overflow-checked (Rational::make).
[[nodiscard]] auto root_of_linear_factor(const RationalPoly& factor) -> Result<Rational> {
    const Rational& q = factor.coefficient(1);          // == q, denominator 1
    const Rational& neg_p = factor.coefficient(0);       // == -p, denominator 1
    std::int64_t p = 0;
    if (__builtin_sub_overflow(std::int64_t{0}, neg_p.numerator(), &p)) {
        return make_error<Rational>(MathError::overflow);
    }
    return Rational::make(p, q.numerator());
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
    auto rp = build_reduced_problem(numerator, denominator);
    if (!rp) {
        return make_error<LogarithmicPart>(rp.error());
    }
    const RationalPoly& a = rp->a;
    const RationalPoly& dm = rp->d;
    const RationalPoly& dprime = rp->dprime;
    const RationalPoly& rpoly = rp->r;

    // Distinct rational roots of R(t) are the rational residues.
    auto roots = rational_roots(rpoly);
    if (!roots) {
        return make_error<LogarithmicPart>(roots.error());
    }
    // Completeness: strip every (t - c) factor; a non-constant remainder means R has a
    // non-rational root, i.e. an irrational/complex residue this pass cannot express.
    RationalPoly remaining = rpoly;
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
        auto term = rational_log_term(a, dm, dprime, c);
        if (!term) {
            return make_error<LogarithmicPart>(term.error());
        }
        if (*term) {
            result.terms.push_back(std::move(**term));
        }
    }
    return result;
}

auto log_part_extended(const RationalPoly& numerator, const RationalPoly& denominator)
    -> Result<ExtendedLogarithmicPart> {
    if (denominator.is_zero()) {
        return make_error<ExtendedLogarithmicPart>(MathError::division_by_zero);
    }
    if (numerator.is_zero()) {
        return ExtendedLogarithmicPart{};  // integral of 0 has no logarithmic part
    }
    auto rp = build_reduced_problem(numerator, denominator);
    if (!rp) {
        return make_error<ExtendedLogarithmicPart>(rp.error());
    }
    const RationalPoly& a = rp->a;
    const RationalPoly& d = rp->d;
    const RationalPoly& dprime = rp->dprime;
    const RationalPoly& rpoly = rp->r;

    // Factor R(t) completely over Q; a budget exhaustion here is an honest
    // not_implemented, never silently treated as "no algebraic residues".
    auto factors = factor_over_Q(rpoly);
    if (!factors) {
        return make_error<ExtendedLogarithmicPart>(factors.error());
    }
    // factor_over_Q makes no ordering guarantee; sort for a reproducible term order.
    std::vector<std::pair<RationalPoly, std::int64_t>> sorted = std::move(*factors);
    std::ranges::sort(sorted, [](const auto& x, const auto& y) {
        return poly_less(x.first, y.first);
    });

    ExtendedLogarithmicPart result;
    std::int64_t degree_sum = 0;  // completeness tripwire: must equal deg(D) at the end
    for (const auto& [factor, multiplicity] : sorted) {
        (void)multiplicity;  // a multiplicity m means x-degree m in the argument, not a
                              // repeated term: each DISTINCT irreducible factor contributes once
        if (factor.degree() == 1) {
            auto c = root_of_linear_factor(factor);
            if (!c) {
                return make_error<ExtendedLogarithmicPart>(c.error());
            }
            auto term = rational_log_term(a, d, dprime, *c);
            if (!term) {
                return make_error<ExtendedLogarithmicPart>(term.error());
            }
            if (*term) {
                degree_sum += (*term)->argument.degree();
                result.rational_terms.push_back(std::move(**term));
            }
            continue;
        }

        // deg >= 2: the residue lives in the extension field Q[t]/(factor).
        auto field_r = NumberField::create(factor);
        if (!field_r) {
            return make_error<ExtendedLogarithmicPart>(field_r.error());
        }
        const NumberField& field = *field_r;
        auto alpha_r = field.generator();
        if (!alpha_r) {
            return make_error<ExtendedLogarithmicPart>(alpha_r.error());
        }
        const AlgebraicNumber& alpha = *alpha_r;

        const AlgebraicPoly a_k = AlgebraicPoly::embed(field, a);
        const AlgebraicPoly dprime_k = AlgebraicPoly::embed(field, dprime);
        auto scaled = dprime_k.scale(alpha);  // alpha * D'
        if (!scaled) {
            return make_error<ExtendedLogarithmicPart>(scaled.error());
        }
        auto shifted = a_k.subtract(*scaled);  // A - alpha*D', over K
        if (!shifted) {
            return make_error<ExtendedLogarithmicPart>(shifted.error());
        }
        const AlgebraicPoly d_k = AlgebraicPoly::embed(field, d);
        auto v = shifted->gcd(d_k);  // monic (AlgebraicPoly::gcd always normalises)
        if (!v) {
            return make_error<ExtendedLogarithmicPart>(v.error());
        }
        if (v->degree() < 1) {
            // A trivial argument here would silently drop x-degree from the completeness
            // identity with no way to account for it: an honest domain_error, not a skip.
            return make_error<ExtendedLogarithmicPart>(MathError::domain_error);
        }
        degree_sum += field.degree() * v->degree();
        result.algebraic_terms.push_back(
            AlgebraicLogTerm{.field = field, .residue = alpha, .argument = std::move(*v)});
    }

    if (degree_sum != d.degree()) {
        // Rule 32 tripwire: the R-T degree identity failed. Never emit a partial result.
        return make_error<ExtendedLogarithmicPart>(MathError::domain_error);
    }
    return result;
}

}  // namespace nimblecas
