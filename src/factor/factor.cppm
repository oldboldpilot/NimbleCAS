// NimbleCAS exact polynomial factorization over the rationals (ROADMAP 7.21).
// @author Olumuyiwa Oluwasanmi
//
// Where nimblecas.roots recovers only the RATIONAL roots of a polynomial (linear
// factors), and Yun's square_free_factorization splits off repeated factors, NEITHER
// breaks a square-free polynomial into its irreducible pieces. A reducible high-degree
// polynomial with no rational roots — e.g. (x^2 - 2)(x^3 - 2) = x^5 - 2x^3 - 2x^2 + 4,
// already square-free — is exactly the input that stalls a radical solver: it has no
// rational root to peel, yet it is NOT irreducible. This module supplies the missing
// step, factoring p in Q[x] all the way to irreducibles, so a downstream solver can then
// dispatch each irreducible factor (a quadratic and a cubic, above) to its radical form
// instead of falling back to numerics.
//
// PIPELINE. Given p in Q[x]:
//   1. clear denominators to an integer polynomial (multiply by the LCM of denominators);
//   2. take its primitive part (divide by the integer content, normalise to a positive
//      leading coefficient);
//   3. run Yun's square_free_factorization -> (square-free primitive factor, multiplicity)
//      pairs, pairwise coprime;
//   4. factor EACH square-free factor into irreducibles via Kronecker's algorithm;
//   5. combine multiplicities.
//
// KRONECKER. On a square-free primitive integer polynomial f of degree n >= 1, a proper
// factor of degree <= floor(n/2) exists iff f is reducible. For each target degree s from
// 1 to floor(n/2), pick s+1 distinct integer nodes; f(node) = 0 peels a linear factor
// (x - node) at once; otherwise every candidate factor g satisfies g(node) | f(node), so
// enumerating one integer divisor of f(node) per node and Lagrange-interpolating the unique
// degree-<= s polynomial through those values yields every candidate g. A candidate is
// accepted when its coefficients are integral and its primitive part divides f exactly;
// the algorithm then recurses on that factor and its cofactor. If no s yields a factor,
// f is irreducible.
//
// HONESTY (Rule 32). Every operation returns Result<T>; nothing throws. Arithmetic is
// exact and overflow-checked and surfaces MathError::overflow on an int64 boundary. The
// divisor-tuple search is bounded by a fixed budget (kDivisorTupleBudget); a pathological
// input that would exceed it returns MathError::not_implemented — an honest "could not
// factor within budget" — rather than looping unboundedly or emitting a wrong partial
// factorization. The zero polynomial (every value a root) is MathError::domain_error; a
// nonzero constant has no non-unit factors and returns an empty list. A returned partial
// result is never presented as complete: factor_over_Q either returns a fully irreducible
// factorization or an error.

module;
#include <cassert>

export module nimblecas.factor;

import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.ratpoly;

export namespace nimblecas {

// Factor p into irreducible factors over Q. Returns (irreducible_factor, multiplicity)
// pairs whose product, each factor raised to its multiplicity, equals the PRIMITIVE PART
// of the integer polynomial obtained from p by clearing denominators — i.e. p up to a
// nonzero rational constant (the content and the leading rational scale are folded out).
// Each returned factor is a primitive integer polynomial (content 1) with a positive
// leading coefficient, lifted into Q[x], and irreducible over Q. Factors are returned in
// no particular order.
//
// The zero polynomial is MathError::domain_error (every value is a root); a nonzero
// constant yields an empty list (no non-unit factors). An input whose factorization would
// exceed the internal divisor-tuple budget returns MathError::not_implemented; an int64
// overflow in the exact arithmetic returns MathError::overflow.
[[nodiscard]] auto factor_over_Q(const RationalPoly& p)
    -> Result<std::vector<std::pair<RationalPoly, std::int64_t>>>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

constexpr std::int64_t int64_min = std::numeric_limits<std::int64_t>::min();

// Maximum number of Lagrange divisor-tuples the Kronecker search may consider across an
// entire factorization. The Cartesian product of per-node divisor counts can blow up for
// polynomials with many large-magnitude sample values; when the running total would exceed
// this cap the search stops and reports MathError::not_implemented rather than looping.
constexpr std::int64_t kDivisorTupleBudget = 1'000'000;

[[nodiscard]] auto mul_ov(std::int64_t a, std::int64_t b, std::int64_t& out) -> bool {
    return __builtin_mul_overflow(a, b, &out);
}
[[nodiscard]] auto sub_ov(std::int64_t a, std::int64_t b, std::int64_t& out) -> bool {
    return __builtin_sub_overflow(a, b, &out);
}

// |v| as a non-negative int64, or nullopt when v == INT64_MIN (magnitude unrepresentable).
[[nodiscard]] auto magnitude(std::int64_t v) -> std::optional<std::int64_t> {
    if (v == int64_min) {
        return std::nullopt;
    }
    return v < 0 ? -v : v;
}

// Positive divisors of n (n > 0), unsorted, no duplicates. The division-based bound
// (i <= n/i) avoids the i*i overflow that a near-INT64_MAX argument would trigger.
[[nodiscard]] auto divisors(std::int64_t n) -> std::vector<std::int64_t> {
    assert(n > 0 && "divisors requires a positive argument");
    std::vector<std::int64_t> ds;
    for (std::int64_t i = 1; i <= n / i; ++i) {
        if (n % i == 0) {
            ds.push_back(i);
            if (i != n / i) {
                ds.push_back(n / i);
            }
        }
    }
    return ds;
}

// Clear denominators: the integer polynomial LCM(denominators) * p. Fails with overflow
// if the LCM or a scaled numerator exceeds int64.
[[nodiscard]] auto to_integer_polynomial(const RationalPoly& p) -> Result<Polynomial> {
    const std::span<const Rational> coeffs = p.coefficients();
    std::int64_t lcm = 1;
    for (const Rational& c : coeffs) {
        const std::int64_t d = c.denominator();     // canonical: d > 0
        const std::int64_t g = std::gcd(lcm, d);    // both positive => no INT64_MIN
        std::int64_t next = 0;
        if (mul_ov(lcm / g, d, next)) {
            return make_error<Polynomial>(MathError::overflow);
        }
        lcm = next;
    }
    std::vector<std::int64_t> out(coeffs.size());
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        const std::int64_t factor = lcm / coeffs[i].denominator();  // exact (denominator | lcm)
        std::int64_t v = 0;
        if (mul_ov(coeffs[i].numerator(), factor, v)) {
            return make_error<Polynomial>(MathError::overflow);
        }
        out[i] = v;
    }
    return Polynomial{std::move(out)};
}

// Cardinal Lagrange basis for the given distinct integer nodes: basis[i] is the unique
// polynomial of degree <= nodes.size()-1 that is 1 at nodes[i] and 0 at every other node.
// Interpolating values (y_0..y_s) is then sum_i y_i * basis[i]. Fails with overflow if the
// basis-numerator product or a node-difference product exceeds int64.
[[nodiscard]] auto lagrange_basis(std::span<const std::int64_t> nodes)
    -> Result<std::vector<RationalPoly>> {
    using Basis = std::vector<RationalPoly>;
    const std::size_t m = nodes.size();
    Basis basis;
    basis.reserve(m);
    for (std::size_t i = 0; i < m; ++i) {
        Polynomial num = Polynomial::constant(1);  // prod_{j != i} (x - nodes[j])
        std::int64_t denom = 1;                     // prod_{j != i} (nodes[i] - nodes[j])
        for (std::size_t j = 0; j < m; ++j) {
            if (j == i) {
                continue;
            }
            auto next = num.multiply(Polynomial{std::vector<std::int64_t>{-nodes[j], 1}});
            if (!next) {
                return make_error<Basis>(next.error());
            }
            num = *next;
            std::int64_t diff = 0;
            if (sub_ov(nodes[i], nodes[j], diff)) {
                return make_error<Basis>(MathError::overflow);
            }
            std::int64_t nd = 0;
            if (mul_ov(denom, diff, nd)) {
                return make_error<Basis>(MathError::overflow);
            }
            denom = nd;
        }
        auto scale = Rational::make(1, denom);  // denom != 0 (distinct nodes)
        if (!scale) {
            return make_error<Basis>(scale.error());
        }
        auto b = RationalPoly::from_polynomial(num).scale(*scale);
        if (!b) {
            return make_error<Basis>(b.error());
        }
        basis.push_back(std::move(*b));
    }
    return basis;
}

// Kronecker factorization of a square-free primitive integer polynomial into irreducible
// primitive factors (positive leading coefficient). `budget` is the shared, monotonically
// decreasing divisor-tuple allowance; exhausting it yields not_implemented.
[[nodiscard]] auto factor_square_free(const Polynomial& f, std::int64_t& budget)
    -> Result<std::vector<Polynomial>> {
    using Factors = std::vector<Polynomial>;

    auto prim = f.primitive_part();  // normalise: content 1, positive leading coefficient
    if (!prim) {
        return make_error<Factors>(prim.error());
    }
    const Polynomial pf = *prim;
    const std::int64_t n = pf.degree();
    if (n <= 0) {
        return Factors{};       // a constant contributes no non-unit factor
    }
    if (n == 1) {
        return Factors{pf};     // a linear polynomial is irreducible
    }

    const std::int64_t half = n / 2;
    for (std::int64_t s = 1; s <= half; ++s) {
        // s + 1 distinct small integer nodes: 0, 1, -1, 2, -2, ...
        std::vector<std::int64_t> nodes;
        nodes.reserve(static_cast<std::size_t>(s) + 1);
        nodes.push_back(0);
        for (std::int64_t k = 1; static_cast<std::int64_t>(nodes.size()) < s + 1; ++k) {
            nodes.push_back(k);
            if (static_cast<std::int64_t>(nodes.size()) < s + 1) {
                nodes.push_back(-k);
            }
        }

        // Evaluate at each node. An integer root peels a linear factor immediately.
        std::vector<std::int64_t> values;
        values.reserve(nodes.size());
        for (const std::int64_t a : nodes) {
            auto v = pf.evaluate(a);
            if (!v) {
                return make_error<Factors>(v.error());
            }
            if (*v == 0) {
                const Polynomial lin{std::vector<std::int64_t>{-a, 1}};  // x - a
                auto q = pf.divide_exact(lin);
                if (!q) {
                    return make_error<Factors>(q.error());
                }
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
            values.push_back(*v);
        }

        // Candidate values per node: all +/- integer divisors of f(node).
        std::vector<std::vector<std::int64_t>> cands;
        cands.reserve(values.size());
        for (const std::int64_t v : values) {
            auto mag = magnitude(v);
            if (!mag) {
                return make_error<Factors>(MathError::overflow);
            }
            std::vector<std::int64_t> cs;
            for (const std::int64_t d : divisors(*mag)) {
                cs.push_back(d);
                cs.push_back(-d);
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
            RationalPoly g_rat;  // interpolant = sum_i cands[i][idx[i]] * basis[i]
            bool arith_error = false;
            MathError err = MathError::domain_error;
            for (std::size_t i = 0; i < nodes.size(); ++i) {
                auto term = (*basis)[i].scale(Rational::from_int(cands[i][idx[i]]));
                if (!term) {
                    arith_error = true;
                    err = term.error();
                    break;
                }
                auto sum = g_rat.add(*term);
                if (!sum) {
                    arith_error = true;
                    err = sum.error();
                    break;
                }
                g_rat = *sum;
            }
            if (arith_error) {
                return make_error<Factors>(err);
            }

            auto gp = g_rat.to_polynomial();  // succeeds iff every coefficient is integral
            if (gp && gp->degree() >= 1) {
                auto gpp = gp->primitive_part();
                if (!gpp) {
                    return make_error<Factors>(gpp.error());
                }
                auto q = pf.divide_exact(*gpp);
                if (q) {
                    auto left = factor_square_free(*gpp, budget);
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
                if (q.error() == MathError::overflow) {
                    return make_error<Factors>(MathError::overflow);
                }
                // domain_error: gpp does not divide f exactly -> keep searching.
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

auto factor_over_Q(const RationalPoly& p)
    -> Result<std::vector<std::pair<RationalPoly, std::int64_t>>> {
    using Out = std::vector<std::pair<RationalPoly, std::int64_t>>;
    if (p.is_zero()) {
        return make_error<Out>(MathError::domain_error);  // every value is a root
    }

    auto ip = to_integer_polynomial(p);
    if (!ip) {
        return make_error<Out>(ip.error());
    }
    if (ip->degree() <= 0) {
        return Out{};  // nonzero constant: no non-unit factors
    }

    auto sqf = ip->square_free_factorization();  // Yun; product of factor^mult = prim(ip)
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
        for (const Polynomial& g : *irr) {
            RationalPoly rg = RationalPoly::from_polynomial(g);
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
