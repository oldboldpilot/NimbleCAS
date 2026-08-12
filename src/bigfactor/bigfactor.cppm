// NimbleCAS exact polynomial factorization over Q on the UNBOUNDED BigRational.
// @author Olumuyiwa Oluwasanmi
//
// This is the arbitrary-precision analogue of nimblecas.factor. Where factor_over_Q there
// runs the Yun -> Kronecker pipeline on the int64 RationalPoly and reports overflow the
// moment a numerator or denominator saturates the int64 ring, this module runs the SAME
// pipeline on BigRationalPoly and has no such ceiling: the exact arithmetic NEVER produces
// MathError::overflow. It is the factorization step a bignum Trager splitting-field path
// needs, so it can factor large-coefficient field norms that overrun the int64 tier.
//
// PIPELINE (mirrors nimblecas.factor). Given p in Q[x]:
//   1. clear denominators to an integer (BigInt-coefficient) polynomial;
//   2. take its primitive part (divide by the integer content, positive leading coeff);
//   3. run Yun's square-free factorization -> (square-free primitive factor, multiplicity);
//   4. factor EACH square-free factor into irreducibles via Kronecker's algorithm;
//   5. combine multiplicities.
//
// KRONECKER. On a square-free primitive integer polynomial f of degree n >= 1, a proper
// factor of degree <= floor(n/2) exists iff f is reducible. For each target degree s, pick
// s+1 distinct integer nodes; an integer root peels a linear factor; otherwise every
// candidate factor g satisfies g(node) | f(node), so enumerating one divisor of f(node) per
// node and Lagrange-interpolating a degree-<= s polynomial through those values yields every
// candidate g. A candidate is accepted when integral and its primitive part divides f
// exactly; the search then recurses on that factor and its cofactor.
//
// HONESTY (Rule 32). Every fallible operation returns Result<T>; nothing throws. The
// arithmetic is exact and unbounded, so MathError::overflow is NEVER produced here. Two
// steps remain genuinely bounded: (a) enumerating the divisors of an integer VALUE f(node)
// needs that value's prime factorization, done by trial division up to kTrialDivisionBound
// with a cap of kMaxDivisorsPerValue on the divisor count; (b) the Cartesian product of the
// per-node divisor choices is bounded by kDivisorTupleBudget. Exceeding either bound returns
// MathError::not_implemented -- an honest "could not factor within budget" -- rather than a
// wrong or partial factorization. The zero polynomial is MathError::domain_error; a nonzero
// constant returns an empty list. factor_over_Q either returns a fully irreducible
// factorization or an error, never a partial result presented as complete.

module;
#include <cassert>

export module nimblecas.bigfactor;

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.bigratpoly;

export namespace nimblecas {

// Factor p into irreducible factors over Q. Returns (irreducible_factor, multiplicity)
// pairs whose product, each factor raised to its multiplicity, equals the PRIMITIVE PART
// of the integer polynomial obtained from p by clearing denominators -- i.e. p up to a
// nonzero rational constant. Each returned factor is a primitive integer polynomial
// (content 1) with a positive leading coefficient, lifted into Q[x], and irreducible over
// Q. Factors are returned in no particular order.
//
// The zero polynomial is MathError::domain_error (every value is a root); a nonzero constant
// yields an empty list (no non-unit factors). An input whose factorization would exceed the
// internal trial-division or divisor-tuple budget returns MathError::not_implemented. The
// exact arithmetic never overflows, so MathError::overflow is never produced.
[[nodiscard]] auto factor_over_Q(const BigRationalPoly& p)
    -> Result<std::vector<std::pair<BigRationalPoly, std::int64_t>>>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// Maximum number of Lagrange divisor-tuples the Kronecker search may consider across an
// entire factorization. The Cartesian product of per-node divisor counts can blow up for
// polynomials with many large-magnitude sample values; when the running total would exceed
// this cap the search stops and reports MathError::not_implemented rather than looping.
constexpr std::int64_t kDivisorTupleBudget = 1'000'000;

// Trial-division ceiling for factoring an integer VALUE f(node) into primes. Trial divisors
// range over 2..kTrialDivisionBound; a residual cofactor above this bound that is not proven
// prime (by d*d exceeding it) means the value cannot be fully factored within budget. Kept
// modest so d*d stays well within int64 and the trial loop stays fast; a value whose
// second-largest prime factor exceeds it yields MathError::not_implemented.
constexpr std::int64_t kTrialDivisionBound = 2'000'000;

// Cap on the number of positive divisors enumerated for any single value; a value with more
// divisors than this yields MathError::not_implemented.
constexpr std::int64_t kMaxDivisorsPerValue = 1'000'000;

// A dense univariate polynomial over the exact integers, coeffs[i] the coefficient of x^i,
// stored trimmed (back() != 0, or empty for the zero polynomial).
using IntPoly = std::vector<BigInt>;

// Exact BigInt quotient a / b for a divisor known to be non-zero (asserts otherwise). Used
// where the division is mathematically exact; BigInt::divide only fails on a zero divisor,
// which the callers rule out. Mirrors bigrational.cppm's divide_checked pattern.
[[nodiscard]] auto divide_checked(const BigInt& a, const BigInt& b) -> BigInt {
    auto q = a.divide(b);
    assert(q.has_value() && "divide_checked requires a non-zero divisor");
    return *q;
}

// Exact BigInt (quotient, remainder) for a non-zero divisor (asserts otherwise).
[[nodiscard]] auto divmod_checked(const BigInt& a, const BigInt& b)
    -> std::pair<BigInt, BigInt> {
    auto dm = a.divmod(b);
    assert(dm.has_value() && "divmod_checked requires a non-zero divisor");
    return *dm;
}

[[nodiscard]] auto trim_int(IntPoly p) -> IntPoly {
    while (!p.empty() && p.back().is_zero()) {
        p.pop_back();
    }
    return p;
}

[[nodiscard]] auto ipoly_degree(const IntPoly& p) -> std::int64_t {
    return p.empty() ? -1 : static_cast<std::int64_t>(p.size()) - 1;
}

// Content = gcd of the coefficients (>= 0; 0 for the zero polynomial). BigInt::gcd is
// sign-independent and non-negative.
[[nodiscard]] auto ipoly_content(const IntPoly& p) -> BigInt {
    BigInt g;  // canonical zero
    for (const BigInt& c : p) {
        g = BigInt::gcd(g, c);
    }
    return g;
}

// Primitive part = this / content, sign-normalised to a positive leading coefficient.
[[nodiscard]] auto ipoly_primitive_part(const IntPoly& p) -> IntPoly {
    if (p.empty()) {
        return IntPoly{};
    }
    const BigInt g = ipoly_content(p);  // > 0 for a non-zero polynomial
    IntPoly r(p.size());
    for (std::size_t i = 0; i < p.size(); ++i) {
        r[i] = divide_checked(p[i], g);  // exact: g divides every coefficient
    }
    if (r.back().sign() < 0) {  // normalise to a positive leading coefficient
        for (BigInt& c : r) {
            c = c.negate();
        }
    }
    return r;  // leading coefficient stays non-zero, so still trimmed
}

// Exact quotient a / b over Z[x], or nullopt when the division is not exact in Z[x]. b must
// be non-zero. Over the exact integers this never overflows; a non-integral step (the
// leading coefficient of the running remainder not divisible by lc(b)) or a non-zero final
// remainder means b does not divide a, reported as nullopt.
[[nodiscard]] auto ipoly_divide_exact(const IntPoly& a, const IntPoly& b)
    -> std::optional<IntPoly> {
    assert(!b.empty() && "ipoly_divide_exact requires a non-zero divisor");
    if (a.empty()) {
        return IntPoly{};  // 0 / b == 0 exactly
    }
    if (a.size() < b.size()) {
        return std::nullopt;  // deg a < deg b with a != 0 -> not exact
    }
    const BigInt lc_b = b.back();
    IntPoly r = a;
    IntPoly q(a.size() - b.size() + 1);  // BigInt default is the canonical zero
    while (!r.empty() && r.size() >= b.size()) {
        const auto [quot, rem] = divmod_checked(r.back(), lc_b);
        if (!rem.is_zero()) {
            return std::nullopt;  // not divisible in Z -> b does not divide a
        }
        const std::size_t k = r.size() - b.size();
        q[k] = quot;
        for (std::size_t i = 0; i < b.size(); ++i) {  // r -= quot * x^k * b
            r[k + i] = r[k + i].subtract(quot.multiply(b[i]));
        }
        r = trim_int(std::move(r));  // the leading term cancels, so the degree strictly drops
    }
    if (!r.empty()) {
        return std::nullopt;  // non-zero remainder -> not exact
    }
    return trim_int(std::move(q));
}

// Horner evaluation of an integer polynomial at an integer point.
[[nodiscard]] auto ipoly_evaluate(const IntPoly& p, const BigInt& x) -> BigInt {
    BigInt acc;  // canonical zero
    for (std::size_t i = p.size(); i-- > 0;) {
        acc = acc.multiply(x).add(p[i]);
    }
    return acc;
}

// Clear denominators: the integer polynomial LCM(denominators) * p. Exact and unbounded --
// the arbitrary-precision arithmetic cannot overflow.
[[nodiscard]] auto to_integer_polynomial(const BigRationalPoly& p) -> IntPoly {
    const std::span<const BigRational> coeffs = p.coefficients();
    BigInt lcm = BigInt::from_i64(1);
    for (const BigRational& c : coeffs) {
        const BigInt& d = c.denominator();     // canonical: d > 0
        const BigInt g = BigInt::gcd(lcm, d);  // both positive
        lcm = divide_checked(lcm, g).multiply(d);
    }
    IntPoly out(coeffs.size());
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        const BigInt factor = divide_checked(lcm, coeffs[i].denominator());  // exact
        out[i] = coeffs[i].numerator().multiply(factor);
    }
    return trim_int(std::move(out));
}

// Lift an integer polynomial into Q[x].
[[nodiscard]] auto ipoly_to_bigratpoly(const IntPoly& p) -> BigRationalPoly {
    std::vector<BigRational> c;
    c.reserve(p.size());
    for (const BigInt& v : p) {
        c.push_back(BigRational::from_bigint(v));
    }
    return BigRationalPoly::from_coeffs(std::move(c));
}

// Narrow a BigRationalPoly to an integer polynomial, or nullopt when any coefficient is not
// integral (denominator != 1).
[[nodiscard]] auto bigratpoly_to_intpoly(const BigRationalPoly& p) -> std::optional<IntPoly> {
    IntPoly out;
    out.reserve(p.coefficients().size());
    for (const BigRational& c : p.coefficients()) {
        if (!c.is_integer()) {
            return std::nullopt;
        }
        out.push_back(c.numerator());
    }
    return trim_int(std::move(out));  // p is trimmed, so this preserves the degree
}

// Exact quotient of two BigRationalPolys known to divide exactly (divisor non-zero). Over
// the field the division always yields a zero remainder here, so the quotient is taken
// directly; BigRationalPoly::divide only fails on a zero divisor, ruled out by the callers.
[[nodiscard]] auto ratpoly_quotient_checked(const BigRationalPoly& a, const BigRationalPoly& b)
    -> BigRationalPoly {
    auto dm = a.divide(b);
    assert(dm.has_value() && "ratpoly_quotient_checked requires a non-zero divisor");
    return dm->quotient;
}

// Yun square-free factorization of an integer polynomial: (square-free primitive factor,
// multiplicity) pairs whose product of factor^multiplicity equals the primitive part of the
// input, each factor square-free and pairwise coprime, positive leading coefficient. Runs
// over Q (BigRationalPoly gcd/divide/derivative) on the monic image, then reduces each monic
// factor to its primitive integer associate.
[[nodiscard]] auto square_free_factorization(const IntPoly& ip)
    -> Result<std::vector<std::pair<IntPoly, std::int64_t>>> {
    using SF = std::vector<std::pair<IntPoly, std::int64_t>>;
    const IntPoly prim = ipoly_primitive_part(ip);
    if (ipoly_degree(prim) <= 0) {
        return SF{};  // a constant has no non-unit square-free factors
    }

    const BigRationalPoly f = ipoly_to_bigratpoly(prim);
    const BigRationalPoly fp = f.derivative();
    auto a0 = f.gcd(fp);  // monic gcd; f non-constant so this is non-zero
    if (!a0) {
        return make_error<SF>(a0.error());
    }
    BigRationalPoly b = ratpoly_quotient_checked(f, *a0);   // b_1 = f / gcd(f, f')
    BigRationalPoly c = ratpoly_quotient_checked(fp, *a0);  // c_1 = f' / gcd(f, f')
    BigRationalPoly d = c.subtract(b.derivative());         // d_1 = c_1 - b_1'

    SF result;
    std::int64_t i = 1;
    while (b.degree() >= 1) {  // until b becomes constant (i past the maximum multiplicity)
        auto ai = b.gcd(d);    // the i-th square-free factor (monic), possibly the unit 1
        if (!ai) {
            return make_error<SF>(ai.error());
        }
        BigRationalPoly b_next = ratpoly_quotient_checked(b, *ai);
        BigRationalPoly c_next = ratpoly_quotient_checked(d, *ai);
        BigRationalPoly d_next = c_next.subtract(b_next.derivative());
        if (ai->degree() >= 1) {  // skip the trivial unit factor (no factor of this mult)
            result.emplace_back(ipoly_primitive_part(to_integer_polynomial(*ai)), i);
        }
        b = std::move(b_next);
        d = std::move(d_next);
        ++i;
    }
    return result;
}

// Prime factorization of n (n >= 1) by trial division, as (prime, exponent) pairs, or
// nullopt when a residual cofactor exceeds the trial-division budget without being proven
// prime. The final cofactor, once all primes up to the bound are stripped, is prime whenever
// d*d exceeds it -- so the LARGEST prime factor is free; only the SECOND-largest must lie
// within the bound for a full factorization.
[[nodiscard]] auto factor_magnitude(BigInt n)
    -> std::optional<std::vector<std::pair<BigInt, int>>> {
    std::vector<std::pair<BigInt, int>> fac;
    const BigInt one = BigInt::from_i64(1);
    for (std::int64_t d = 2;; ++d) {
        const std::int64_t dd = d * d;  // d <= bound keeps d*d well within int64
        if (BigInt::from_i64(dd) > n) {
            break;  // remaining n is 1 or a single prime > d
        }
        if (d > kTrialDivisionBound) {
            return std::nullopt;  // cofactor too large to factor within budget
        }
        const BigInt bd = BigInt::from_i64(d);
        int exp = 0;
        while (true) {
            const auto [q, r] = divmod_checked(n, bd);
            if (!r.is_zero()) {
                break;
            }
            n = q;
            ++exp;
        }
        if (exp > 0) {
            fac.emplace_back(bd, exp);
        }
    }
    if (n > one) {
        fac.emplace_back(std::move(n), 1);  // the leftover prime cofactor
    }
    return fac;
}

// All positive divisors of a non-zero value, or nullopt when it cannot be factored within
// budget or has more than kMaxDivisorsPerValue divisors.
[[nodiscard]] auto divisors_of(const BigInt& value) -> std::optional<std::vector<BigInt>> {
    auto fac = factor_magnitude(value.abs());
    if (!fac) {
        return std::nullopt;
    }
    std::vector<BigInt> divs{BigInt::from_i64(1)};
    for (const auto& [p, e] : *fac) {
        std::vector<BigInt> next;
        BigInt pk = BigInt::from_i64(1);
        for (int k = 0; k <= e; ++k) {
            for (const BigInt& dv : divs) {
                next.push_back(dv.multiply(pk));
            }
            pk = pk.multiply(p);
        }
        if (static_cast<std::int64_t>(next.size()) > kMaxDivisorsPerValue) {
            return std::nullopt;
        }
        divs = std::move(next);
    }
    return divs;
}

// Cardinal Lagrange basis for the given distinct integer nodes: basis[i] is the unique
// polynomial of degree <= nodes.size()-1 that is 1 at nodes[i] and 0 at every other node.
// Over BigRational the arithmetic is exact and unbounded; the only fallible step is the
// reciprocal of the node-difference product, which is non-zero for distinct nodes.
[[nodiscard]] auto lagrange_basis(const std::vector<BigInt>& nodes)
    -> Result<std::vector<BigRationalPoly>> {
    using Basis = std::vector<BigRationalPoly>;
    const std::size_t m = nodes.size();
    Basis basis;
    basis.reserve(m);
    for (std::size_t i = 0; i < m; ++i) {
        BigRationalPoly num = BigRationalPoly::constant(BigRational::from_int(1));
        BigInt denom = BigInt::from_i64(1);
        for (std::size_t j = 0; j < m; ++j) {
            if (j == i) {
                continue;
            }
            const BigRationalPoly lin = BigRationalPoly::from_coeffs(  // x - nodes[j]
                {BigRational::from_bigint(nodes[j].negate()), BigRational::from_int(1)});
            num = num.multiply(lin);
            denom = denom.multiply(nodes[i].subtract(nodes[j]));
        }
        auto scale = BigRational::from_bigint(denom).reciprocal();  // denom != 0
        if (!scale) {
            return make_error<Basis>(scale.error());
        }
        basis.push_back(num.scale(*scale));
    }
    return basis;
}

// Kronecker factorization of a square-free primitive integer polynomial into irreducible
// primitive factors (positive leading coefficient). `budget` is the shared, monotonically
// decreasing divisor-tuple allowance; exhausting it yields not_implemented.
[[nodiscard]] auto factor_square_free(const IntPoly& f, std::int64_t& budget)
    -> Result<std::vector<IntPoly>> {
    using Factors = std::vector<IntPoly>;

    const IntPoly pf = ipoly_primitive_part(f);  // content 1, positive leading coefficient
    const std::int64_t n = ipoly_degree(pf);
    if (n <= 0) {
        return Factors{};    // a constant contributes no non-unit factor
    }
    if (n == 1) {
        return Factors{pf};  // a linear polynomial is irreducible
    }

    const std::int64_t half = n / 2;
    for (std::int64_t s = 1; s <= half; ++s) {
        // s + 1 distinct small integer nodes: 0, 1, -1, 2, -2, ...
        std::vector<BigInt> nodes;
        nodes.reserve(static_cast<std::size_t>(s) + 1);
        nodes.push_back(BigInt::from_i64(0));
        for (std::int64_t k = 1; static_cast<std::int64_t>(nodes.size()) < s + 1; ++k) {
            nodes.push_back(BigInt::from_i64(k));
            if (static_cast<std::int64_t>(nodes.size()) < s + 1) {
                nodes.push_back(BigInt::from_i64(-k));
            }
        }

        // Evaluate at each node. An integer root peels a linear factor immediately.
        std::vector<BigInt> values;
        values.reserve(nodes.size());
        for (const BigInt& a : nodes) {
            BigInt v = ipoly_evaluate(pf, a);
            if (v.is_zero()) {
                const IntPoly lin{a.negate(), BigInt::from_i64(1)};  // x - a
                auto q = ipoly_divide_exact(pf, lin);
                assert(q.has_value() && "an integer root peels an exact linear factor");
                auto left = factor_square_free(lin, budget);
                if (!left) {
                    return left;
                }
                auto right = factor_square_free(*q, budget);
                if (!right) {
                    return right;
                }
                left->insert(left->end(), right->begin(), right->end());
                return left;
            }
            values.push_back(std::move(v));
        }

        // Candidate values per node: all +/- divisors of f(node).
        std::vector<std::vector<BigInt>> cands;
        cands.reserve(values.size());
        for (const BigInt& v : values) {
            auto ds = divisors_of(v);
            if (!ds) {
                return make_error<Factors>(MathError::not_implemented);
            }
            std::vector<BigInt> cs;
            cs.reserve(ds->size() * 2);
            for (const BigInt& d : *ds) {
                cs.push_back(d);
                cs.push_back(d.negate());
            }
            cands.push_back(std::move(cs));
        }

        // Charge the full Cartesian-product size to the shared budget (saturating so the
        // product itself cannot overflow). Exceeding the budget is an honest not_implemented.
        std::int64_t count = 1;
        for (const auto& cs : cands) {
            const std::int64_t k = static_cast<std::int64_t>(cs.size());
            if (k != 0 && count > (kDivisorTupleBudget + 1) / k) {
                count = kDivisorTupleBudget + 1;
                break;
            }
            count *= k;
        }
        if (count > budget) {
            return make_error<Factors>(MathError::not_implemented);
        }
        budget -= count;

        auto basis = lagrange_basis(nodes);
        if (!basis) {
            return make_error<Factors>(basis.error());
        }

        // Enumerate the Cartesian product of divisor choices via a mixed-radix counter.
        std::vector<std::size_t> idx(nodes.size(), 0);
        while (true) {
            BigRationalPoly g_rat = BigRationalPoly::zero();  // sum_i cands[i][idx[i]]*basis[i]
            for (std::size_t i = 0; i < nodes.size(); ++i) {
                const BigRationalPoly term =
                    (*basis)[i].scale(BigRational::from_bigint(cands[i][idx[i]]));
                g_rat = g_rat.add(term);
            }

            auto gp = bigratpoly_to_intpoly(g_rat);  // integral iff every coefficient is
            if (gp && ipoly_degree(*gp) >= 1) {
                const IntPoly gpp = ipoly_primitive_part(*gp);
                auto q = ipoly_divide_exact(pf, gpp);
                if (q) {  // gpp divides f exactly -> a genuine factor; recurse on both parts
                    auto left = factor_square_free(gpp, budget);
                    if (!left) {
                        return left;
                    }
                    auto right = factor_square_free(*q, budget);
                    if (!right) {
                        return right;
                    }
                    left->insert(left->end(), right->begin(), right->end());
                    return left;
                }
                // gpp does not divide f exactly -> keep searching.
            }

            std::size_t pos = 0;
            for (; pos < nodes.size(); ++pos) {
                if (++idx[pos] < cands[pos].size()) {
                    break;
                }
                idx[pos] = 0;
            }
            if (pos == nodes.size()) {
                break;  // exhausted the product for this target degree s
            }
        }
    }
    return Factors{pf};  // no factor of degree <= n/2 exists -> f is irreducible
}

}  // namespace

auto factor_over_Q(const BigRationalPoly& p)
    -> Result<std::vector<std::pair<BigRationalPoly, std::int64_t>>> {
    using Out = std::vector<std::pair<BigRationalPoly, std::int64_t>>;
    if (p.is_zero()) {
        return make_error<Out>(MathError::domain_error);  // every value is a root
    }

    const IntPoly ip = to_integer_polynomial(p);  // exact, unbounded (never overflows)
    if (ipoly_degree(ip) <= 0) {
        return Out{};  // nonzero constant: no non-unit factors
    }

    auto sqf = square_free_factorization(ip);  // Yun; product of factor^mult = prim(ip)
    if (!sqf) {
        return make_error<Out>(sqf.error());
    }

    std::int64_t budget = kDivisorTupleBudget;
    Out out;
    for (const auto& [sf, mult] : *sqf) {
        auto irr = factor_square_free(sf, budget);
        if (!irr) {
            return make_error<Out>(irr.error());
        }
        for (const IntPoly& g : *irr) {
            BigRationalPoly rg = ipoly_to_bigratpoly(g);
            // Distinct square-free factors are pairwise coprime, so an irreducible cannot
            // recur across them; the merge is a defensive combine of equal factors.
            auto it = std::ranges::find_if(
                out, [&](const auto& e) { return e.first.is_equal(rg); });
            if (it != out.end()) {
                it->second += mult;
            } else {
                out.emplace_back(std::move(rg), mult);
            }
        }
    }
    return out;
}

}  // namespace nimblecas
