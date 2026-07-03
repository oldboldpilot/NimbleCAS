// NimbleCAS partial-fraction decomposition over the rationals Q[x] (ROADMAP 7.17).
// @author Olumuyiwa Oluwasanmi
//
// Given a rational function A(x)/B(x) over Q, partial_fractions produces the exact
// decomposition
//
//     A(x)/B(x) = P(x) + sum_i sum_{j=1}^{e_i}  C_{i,j}(x) / b_i(x)^j ,
//                 deg C_{i,j} < deg b_i,
//
// where B = prod_i b_i^{e_i} is the square-free factorization of the (monic-normalised)
// denominator, the b_i are monic, square-free and pairwise coprime, and P is the
// polynomial part (zero iff A/B is already proper). The pipeline is three exact,
// overflow-checked stages built on nimblecas.ratpoly:
//
//   1. Yun's algorithm gives the square-free factorization of B over Q (§7.17).
//   2. A Bezout / extended-Euclid "distinct-factor split" writes the proper part
//      R/B = sum_i N_i / b_i^{e_i} with deg N_i < deg(b_i^{e_i}) — each prime power
//      is separated by solving s*p + t*(B/p) = 1 for the coprime pair (p, B/p).
//   3. A base-b expansion of each N_i (repeated division by b_i) splits it across the
//      ascending powers b_i^1 .. b_i^{e_i}, yielding the numerators C_{i,j}.
//
// Everything returns Result (Rule 32): a zero denominator is MathError::division_by_zero
// and an int64 coefficient overflow surfaces as MathError::overflow, exactly as in the
// underlying Q[x] arithmetic.

export module nimblecas.pfd;

import std;
import nimblecas.core;
import nimblecas.ratpoly;

export namespace nimblecas {

// One summand C(x) / factor(x)^power of the proper part, with `factor` monic and
// square-free and deg C < deg factor.
struct PartialFractionTerm {
    RationalPoly factor;     // b_i — a monic, square-free base
    std::int64_t power;      // j >= 1 — the power of b_i in this summand's denominator
    RationalPoly numerator;  // C_{i,j} — deg < deg factor
};

// A(x)/B(x) = polynomial_part + sum over terms.
struct PartialFraction {
    RationalPoly polynomial_part;            // P(x); zero when A/B is proper
    std::vector<PartialFractionTerm> terms;  // the proper part, split per prime power
};

// One square-free "tower": numerator(x) / base(x)^exponent, with base monic and
// square-free and deg numerator < exponent * deg base. This is the proper part grouped
// by square-free factor -- one summand per distinct multiplicity -- BEFORE spreading
// each numerator across the individual powers base^1 .. base^exponent. It is the form
// Hermite reduction consumes (nimblecas.ratint) and the intermediate partial_fractions
// itself expands.
struct SquareFreeTerm {
    RationalPoly base;        // b_i — a monic, square-free base
    std::int64_t exponent;    // e_i >= 1 — the multiplicity of b_i in the denominator
    RationalPoly numerator;   // N_i — deg < e_i * deg b_i
};

// A(x)/B(x) = polynomial_part + sum_i numerator_i / base_i^{exponent_i}.
struct SquareFreePartialFraction {
    RationalPoly polynomial_part;
    std::vector<SquareFreeTerm> towers;  // one per pairwise-coprime prime power
};

// Square-free factorization of f over Q via Yun's algorithm. Returns pairs (a_i, i)
// with each a_i monic, square-free, of degree >= 1, the a_i pairwise coprime, and
//     leading_coefficient(f) * prod_i a_i^i == f.
// A zero or constant f has no factors of positive degree and yields an empty list.
[[nodiscard]] auto square_free_factorization(const RationalPoly& f)
    -> Result<std::vector<std::pair<RationalPoly, std::int64_t>>>;

// Square-free (per-prime-power) partial-fraction decomposition of numerator/denominator
// over Q[x]: the proper part is grouped by square-free factor, one tower per distinct
// multiplicity. Fails with division_by_zero (denominator == 0) or overflow.
[[nodiscard]] auto square_free_partial_fractions(const RationalPoly& numerator,
                                                 const RationalPoly& denominator)
    -> Result<SquareFreePartialFraction>;

// Partial-fraction decomposition of numerator/denominator over Q[x], spread across the
// individual powers. Fails with division_by_zero (denominator == 0) or overflow.
[[nodiscard]] auto partial_fractions(const RationalPoly& numerator,
                                     const RationalPoly& denominator)
    -> Result<PartialFraction>;

// Human-readable rendering, e.g. "x + (1/2)/(x - 1) + (-1)/(x)^2" — for diagnostics.
[[nodiscard]] auto to_string(const PartialFraction& pf, std::string_view var = "x")
    -> std::string;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

[[nodiscard]] auto one_poly() -> RationalPoly {
    return RationalPoly::constant(Rational::from_int(1));
}

// Exact quotient a / b, discarding the remainder. Every caller divides where the
// division is mathematically exact (Yun's cofactors, prime-power cofactors), so the
// remainder is provably zero; only overflow or a zero divisor can fail.
[[nodiscard]] auto exact_quotient(const RationalPoly& a, const RationalPoly& b)
    -> Result<RationalPoly> {
    auto dm = a.divide(b);
    if (!dm) {
        return make_error<RationalPoly>(dm.error());
    }
    return std::move(dm->quotient);
}

// b raised to a non-negative integer power via repeated multiplication.
[[nodiscard]] auto pow_poly(const RationalPoly& b, std::int64_t n) -> Result<RationalPoly> {
    RationalPoly acc = one_poly();
    for (std::int64_t k = 0; k < n; ++k) {
        auto next = acc.multiply(b);
        if (!next) {
            return next;
        }
        acc = std::move(*next);
    }
    return acc;
}

// --- Yun's square-free factorization ---------------------------------------

auto square_free_impl(const RationalPoly& f)
    -> Result<std::vector<std::pair<RationalPoly, std::int64_t>>> {
    using Factors = std::vector<std::pair<RationalPoly, std::int64_t>>;
    Factors out;
    if (f.is_zero() || f.degree() == 0) {
        return out;  // no factors of positive degree
    }
    auto fm = f.monic();  // work monic; the leading constant is recorded by the caller
    if (!fm) {
        return make_error<Factors>(fm.error());
    }
    auto d = fm->derivative();
    if (!d) {
        return make_error<Factors>(d.error());
    }
    auto a0 = fm->gcd(*d);  // monic gcd of f and f'
    if (!a0) {
        return make_error<Factors>(a0.error());
    }
    // b = f / a0 (the square-free part scaffold); c = f' / a0.
    auto b = exact_quotient(*fm, *a0);
    if (!b) {
        return make_error<Factors>(b.error());
    }
    auto c = exact_quotient(*d, *a0);
    if (!c) {
        return make_error<Factors>(c.error());
    }
    // dd = c - b'.
    auto bprime = b->derivative();
    if (!bprime) {
        return make_error<Factors>(bprime.error());
    }
    auto dd = c->subtract(*bprime);
    if (!dd) {
        return make_error<Factors>(dd.error());
    }
    std::int64_t i = 1;
    while (b->degree() > 0) {
        auto ai = b->gcd(*dd);  // monic; the multiplicity-i square-free factor
        if (!ai) {
            return make_error<Factors>(ai.error());
        }
        if (ai->degree() >= 1) {
            out.emplace_back(*ai, i);
        }
        auto b_next = exact_quotient(*b, *ai);
        if (!b_next) {
            return make_error<Factors>(b_next.error());
        }
        auto c_next = exact_quotient(*dd, *ai);
        if (!c_next) {
            return make_error<Factors>(c_next.error());
        }
        auto bn_prime = b_next->derivative();
        if (!bn_prime) {
            return make_error<Factors>(bn_prime.error());
        }
        auto dd_next = c_next->subtract(*bn_prime);
        if (!dd_next) {
            return make_error<Factors>(dd_next.error());
        }
        b = std::move(b_next);
        dd = std::move(dd_next);
        ++i;
    }
    return out;
}

// --- Extended Euclid / Bezout over Q[x] ------------------------------------

struct ExtGcd {
    RationalPoly g;  // gcd(a, b) (not necessarily monic)
    RationalPoly s;  // s*a + t*b == g
    RationalPoly t;
};

auto ext_gcd(const RationalPoly& a, const RationalPoly& b) -> Result<ExtGcd> {
    RationalPoly old_r = a;
    RationalPoly r = b;
    RationalPoly old_s = one_poly();
    RationalPoly s;  // zero
    RationalPoly old_t;  // zero
    RationalPoly t = one_poly();
    while (!r.is_zero()) {
        auto dm = old_r.divide(r);
        if (!dm) {
            return make_error<ExtGcd>(dm.error());
        }
        const RationalPoly& q = dm->quotient;
        // (old_r, r) = (r, old_r - q*r) — the remainder is exactly old_r - q*r.
        old_r = std::move(r);
        r = std::move(dm->remainder);
        // (old_s, s) = (s, old_s - q*s); likewise for t.
        auto qs = q.multiply(s);
        if (!qs) {
            return make_error<ExtGcd>(qs.error());
        }
        auto ns = old_s.subtract(*qs);
        if (!ns) {
            return make_error<ExtGcd>(ns.error());
        }
        old_s = std::move(s);
        s = std::move(*ns);
        auto qt = q.multiply(t);
        if (!qt) {
            return make_error<ExtGcd>(qt.error());
        }
        auto nt = old_t.subtract(*qt);
        if (!nt) {
            return make_error<ExtGcd>(nt.error());
        }
        old_t = std::move(t);
        t = std::move(*nt);
    }
    return ExtGcd{.g = std::move(old_r), .s = std::move(old_s), .t = std::move(old_t)};
}

// Solve s*p + t*q = 1 for a coprime pair (p, q). Returns (s, t) reduced so the identity
// equals exactly 1; a non-constant gcd means the inputs were not coprime (domain_error).
struct Bezout {
    RationalPoly s;
    RationalPoly t;
};

auto bezout_one(const RationalPoly& p, const RationalPoly& q) -> Result<Bezout> {
    auto eg = ext_gcd(p, q);
    if (!eg) {
        return make_error<Bezout>(eg.error());
    }
    if (eg->g.degree() != 0) {
        return make_error<Bezout>(MathError::domain_error);  // not coprime
    }
    // g is a non-zero constant; scale s, t by 1/g so that s*p + t*q == 1.
    auto inv = Rational::from_int(1).divide(eg->g.coefficient(0));
    if (!inv) {
        return make_error<Bezout>(inv.error());
    }
    auto s = eg->s.scale(*inv);
    if (!s) {
        return make_error<Bezout>(s.error());
    }
    auto t = eg->t.scale(*inv);
    if (!t) {
        return make_error<Bezout>(t.error());
    }
    return Bezout{.s = std::move(*s), .t = std::move(*t)};
}

// --- Distinct-factor split --------------------------------------------------

// Given a proper R (deg R < deg prod pk) and pairwise-coprime prime powers pk, return
// numerators N with deg N[i] < deg pk[i] and R / prod pk == sum_i N[i] / pk[i].
auto split_distinct(const RationalPoly& r, std::span<const RationalPoly> pk)
    -> Result<std::vector<RationalPoly>> {
    using Nums = std::vector<RationalPoly>;
    Nums out;
    const std::size_t m = pk.size();
    if (m == 0) {
        return out;  // R is the zero polynomial here
    }
    // suffix[i] = product of pk[i+1 ..]; suffix[m-1] = 1.
    std::vector<RationalPoly> suffix(m);
    suffix[m - 1] = one_poly();
    for (std::size_t i = m - 1; i-- > 0;) {
        auto prod = pk[i + 1].multiply(suffix[i + 1]);
        if (!prod) {
            return make_error<Nums>(prod.error());
        }
        suffix[i] = std::move(*prod);
    }
    RationalPoly cur = r;
    for (std::size_t i = 0; i + 1 < m; ++i) {
        const RationalPoly& p = pk[i];
        const RationalPoly& qrest = suffix[i];  // product of the remaining factors
        auto bz = bezout_one(p, qrest);         // s*p + t*qrest == 1
        if (!bz) {
            return make_error<Nums>(bz.error());
        }
        // R/(p*qrest) = (R*t)/p + (R*s)/qrest, reduced to proper numerators:
        //   N_i = (cur * t) mod p,  cur' = (cur * s) mod qrest.
        auto rt = cur.multiply(bz->t);
        if (!rt) {
            return make_error<Nums>(rt.error());
        }
        auto ni = rt->divide(p);
        if (!ni) {
            return make_error<Nums>(ni.error());
        }
        auto rs = cur.multiply(bz->s);
        if (!rs) {
            return make_error<Nums>(rs.error());
        }
        auto next = rs->divide(qrest);
        if (!next) {
            return make_error<Nums>(next.error());
        }
        out.push_back(std::move(ni->remainder));
        cur = std::move(next->remainder);
    }
    out.push_back(std::move(cur));  // last factor: deg cur < deg pk[m-1] already
    return out;
}

// --- Base-b power expansion -------------------------------------------------

// Expand N / b^e = sum_{j=1}^{e} C_j / b^j with deg C_j < deg b. Returns C indexed by
// j-1: repeatedly divide by b to read off the base-b "digits" D_t (low order first),
// where C_{e-t} = D_t. deg N < e*deg b guarantees exactly e digits (the final quotient
// is zero).
auto power_expand(const RationalPoly& n, const RationalPoly& b, std::int64_t e)
    -> Result<std::vector<RationalPoly>> {
    using Digits = std::vector<RationalPoly>;
    Digits c(static_cast<std::size_t>(e));  // default-constructed zero polynomials
    RationalPoly r = n;
    for (std::int64_t t = 0; t < e; ++t) {
        auto dm = r.divide(b);
        if (!dm) {
            return make_error<Digits>(dm.error());
        }
        // D_t = remainder is the numerator over b^{e-t}, i.e. C_{e-t}.
        c[static_cast<std::size_t>(e - 1 - t)] = std::move(dm->remainder);
        r = std::move(dm->quotient);
    }
    return c;
}

}  // namespace

auto square_free_factorization(const RationalPoly& f)
    -> Result<std::vector<std::pair<RationalPoly, std::int64_t>>> {
    return square_free_impl(f);
}

auto square_free_partial_fractions(const RationalPoly& numerator,
                                   const RationalPoly& denominator)
    -> Result<SquareFreePartialFraction> {
    if (denominator.is_zero()) {
        return make_error<SquareFreePartialFraction>(MathError::division_by_zero);
    }
    // Normalise the denominator to monic, folding the leading constant into the
    // numerator so the rational function's value is unchanged: A/B == (A/lc)/(B/lc).
    const Rational lc = denominator.leading_coefficient();
    auto bm = denominator.monic();
    if (!bm) {
        return make_error<SquareFreePartialFraction>(bm.error());
    }
    auto inv_lc = Rational::from_int(1).divide(lc);  // lc != 0 (B is non-zero)
    if (!inv_lc) {
        return make_error<SquareFreePartialFraction>(inv_lc.error());
    }
    auto am = numerator.scale(*inv_lc);
    if (!am) {
        return make_error<SquareFreePartialFraction>(am.error());
    }
    // Split off the polynomial part: Am = P*Bm + R, deg R < deg Bm.
    auto dm = am->divide(*bm);
    if (!dm) {
        return make_error<SquareFreePartialFraction>(dm.error());
    }
    SquareFreePartialFraction result;
    result.polynomial_part = std::move(dm->quotient);
    const RationalPoly remainder = std::move(dm->remainder);

    // Square-free factorization of the monic denominator: Bm = prod bases[k]^exps[k].
    auto sqf = square_free_impl(*bm);
    if (!sqf) {
        return make_error<SquareFreePartialFraction>(sqf.error());
    }
    // Assemble the pairwise-coprime prime powers pk[k] = bases[k]^exps[k].
    std::vector<RationalPoly> bases;
    std::vector<std::int64_t> exps;
    std::vector<RationalPoly> prime_powers;
    bases.reserve(sqf->size());
    exps.reserve(sqf->size());
    prime_powers.reserve(sqf->size());
    for (const auto& [base, e] : *sqf) {
        auto pk = pow_poly(base, e);
        if (!pk) {
            return make_error<SquareFreePartialFraction>(pk.error());
        }
        bases.push_back(base);
        exps.push_back(e);
        prime_powers.push_back(std::move(*pk));
    }
    // Distinct-factor split of the proper part: R/Bm = sum_k N[k]/prime_powers[k].
    auto numerators = split_distinct(remainder, prime_powers);
    if (!numerators) {
        return make_error<SquareFreePartialFraction>(numerators.error());
    }
    result.towers.reserve(bases.size());
    for (std::size_t k = 0; k < bases.size(); ++k) {
        result.towers.push_back({.base = std::move(bases[k]),
                                 .exponent = exps[k],
                                 .numerator = std::move((*numerators)[k])});
    }
    return result;
}

auto partial_fractions(const RationalPoly& numerator, const RationalPoly& denominator)
    -> Result<PartialFraction> {
    auto sf = square_free_partial_fractions(numerator, denominator);
    if (!sf) {
        return make_error<PartialFraction>(sf.error());
    }
    PartialFraction result;
    result.polynomial_part = std::move(sf->polynomial_part);
    // Expand each tower N / base^e across the ascending powers base^1 .. base^e.
    for (auto& tower : sf->towers) {
        auto digits = power_expand(tower.numerator, tower.base, tower.exponent);
        if (!digits) {
            return make_error<PartialFraction>(digits.error());
        }
        for (std::int64_t j = 1; j <= tower.exponent; ++j) {
            RationalPoly& cj = (*digits)[static_cast<std::size_t>(j - 1)];
            if (cj.is_zero()) {
                continue;  // omit vanishing numerators
            }
            result.terms.push_back(
                {.factor = tower.base, .power = j, .numerator = std::move(cj)});
        }
    }
    return result;
}

auto to_string(const PartialFraction& pf, std::string_view var) -> std::string {
    std::string out;
    bool first = true;
    if (!pf.polynomial_part.is_zero()) {
        out += pf.polynomial_part.to_string(var);
        first = false;
    }
    for (const auto& term : pf.terms) {
        if (!first) {
            out += " + ";
        }
        first = false;
        if (term.power == 1) {
            out += std::format("({})/({})", term.numerator.to_string(var),
                               term.factor.to_string(var));
        } else {
            out += std::format("({})/({})^{}", term.numerator.to_string(var),
                               term.factor.to_string(var), term.power);
        }
    }
    return out.empty() ? "0" : out;
}

}  // namespace nimblecas
