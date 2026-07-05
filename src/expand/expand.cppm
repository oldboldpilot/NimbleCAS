// NimbleCAS algebraic expansion (distributes products over sums; the multinomial
// expansion of integer powers of sums).
// @author Olumuyiwa Oluwasanmi
//
// expand(u) is the counterpart to simplify(u): where Cohen's automatic
// simplification deliberately never distributes (x*(y+z) stays factored, so like
// terms can be grouped by structural identity), expand(u) multiplies everything
// out into a flat sum of monomials and then runs simplify to collect like terms.
//
//   (a + b)*(c + d)  ->  a*c + a*d + b*c + b*d
//   (a + b)^n        ->  sum_k C(n,k) a^k b^(n-k)         (binomial / multinomial)
//
// Honesty (Rule 32): expansion is exact — it only ever rewrites an expression into
// an equal one. A power is expanded ONLY when its exponent is a literal non-negative
// integer no greater than `max_expand_exponent`; anything else (symbolic, negative,
// rational, or over the cap) is left intact — that is not an error, it is the honest
// unexpanded-but-exact answer. Blow-up is bounded two ways: the exponent cap, and a
// term cap (`max_expand_terms`) on how many monomials a single distribution/power may
// produce. Exceeding either leaves the relevant subtree undistributed (still exact).
// The only error channels are the ones simplify already owns (integer overflow while
// folding a genuinely huge coefficient/constant, an undefined form such as 0^0).

export module nimblecas.expand;

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.combinatorics;

export namespace nimblecas {

// Distribute products over sums and expand integer non-negative powers of sums, then
// simplify. Exact and total: an unexpandable power is returned intact rather than as
// an error (see the file header for the honesty boundary).
[[nodiscard]] auto expand(const Expr& u) -> Result<Expr>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

template <typename...>
inline constexpr bool always_false = false;

// --- expansion caps --------------------------------------------------------
//
// A power whose literal integer exponent exceeds this is left unexpanded. 64 is
// chosen so that every binomial coefficient of a two-term base stays inside int64:
// the central C(64,32) is about 1.8e18 < INT64_MAX (9.2e18), so a binomial expansion
// never overflows its coefficients under the cap.
inline constexpr std::int64_t max_expand_exponent = 64;

// Upper bound on the number of monomials a single distribution (product of two sums)
// or power expansion (estimated via the multinomial count C(n+m-1, m-1)) may produce.
// Exceeding it leaves that node undistributed (still exact), so expansion can never
// balloon without bound.
inline constexpr std::int64_t max_expand_terms = 4096;

// --- small helpers ---------------------------------------------------------

// The additive operands of e, FLATTENED: the terms of a sum (recursively splicing in
// the terms of any nested sum), or {e} for anything else. Flattening is essential —
// expand_raw rebuilds sums whose terms may themselves be sums (Expr::sum is the raw
// constructor and does not flatten), and expand_product relies on every operand here
// being a non-sum monomial. Without the recursion a term like (t^2 + x^3 t^2) would be
// multiplied as a single opaque factor product{c, (t^2 + x^3 t^2)} and left
// undistributed, so a genuinely-zero difference would fail to collapse to 0.
// Copies the operands so callers can slice/rebuild freely (Expr copies are O(1) COW bumps).
[[nodiscard]] auto as_terms(const Expr& e) -> std::vector<Expr> {
    if (auto a = as<AddNode>(e.node().value)) {
        std::vector<Expr> out;
        out.reserve((*a)->terms.size());
        for (const Expr& t : (*a)->terms) {
            // as_terms(t) is {t} when t is not itself a sum, so this both appends plain
            // terms and recursively splices in the terms of any nested sum.
            std::vector<Expr> sub = as_terms(t);
            out.insert(out.end(), std::make_move_iterator(sub.begin()),
                       std::make_move_iterator(sub.end()));
        }
        return out;
    }
    return {e};
}

// A literal non-negative integer exponent, if e is exactly that: a ConstantNode
// holding an int64 >= 0, or a reduced rational with denominator 1 and numerator >= 0.
// Everything else (double, negative, symbolic, compound) yields nullopt so the power
// is left unexpanded.
[[nodiscard]] auto as_nonneg_int(const Expr& e) -> std::optional<std::int64_t> {
    auto c = as<ConstantNode>(e.node().value);
    if (!c) {
        return std::nullopt;
    }
    return std::visit(
        []<typename V>(const V& v) -> std::optional<std::int64_t> {
            if constexpr (std::is_same_v<V, std::int64_t>) {
                return v >= 0 ? std::optional<std::int64_t>{v} : std::nullopt;
            } else if constexpr (std::is_same_v<V, std::pair<std::int64_t, std::int64_t>>) {
                return (v.second == 1 && v.first >= 0) ? std::optional<std::int64_t>{v.first}
                                                       : std::nullopt;
            } else {  // double
                return std::nullopt;
            }
        },
        (*c)->value);
}

// base^e as a raw node, collapsing the trivial exponents so the tree stays small
// (simplify would do the same, but avoiding the extra node keeps intermediates lean).
[[nodiscard]] auto make_pow(const Expr& base, std::int64_t e) -> Expr {
    if (e == 0) {
        return Expr::integer(1);
    }
    if (e == 1) {
        return base;
    }
    return Expr::power(base, Expr::integer(e));
}

// --- forward declarations of the mutually-recursive expanders --------------
[[nodiscard]] auto expand_raw(const Expr& u) -> Result<Expr>;
[[nodiscard]] auto expand_power(const Expr& base, std::int64_t n) -> Result<Expr>;

// Distribute the product a*b over any sums in either factor, yielding a flat sum of
// pairwise products. Never fallible: if the cross product would exceed the term cap,
// the factors are left multiplied but undistributed (product{a, b}) — still exact.
// a and b are assumed already expanded.
[[nodiscard]] auto expand_product(const Expr& a, const Expr& b) -> Expr {
    std::vector<Expr> ta = as_terms(a);
    std::vector<Expr> tb = as_terms(b);
    // Sizes are bounded by prior caps, so this int64 product cannot overflow.
    if (static_cast<std::int64_t>(ta.size()) * static_cast<std::int64_t>(tb.size()) >
        max_expand_terms) {
        return Expr::product({a, b});  // leave undistributed: exact, just not multiplied out
    }
    std::vector<Expr> out;
    out.reserve(ta.size() * tb.size());
    for (const Expr& x : ta) {
        for (const Expr& y : tb) {
            out.push_back(Expr::product({x, y}));
        }
    }
    if (out.size() == 1) {
        return std::move(out.front());
    }
    return Expr::sum(std::move(out));
}

// (sum of terms)^n via iterated binomial split: with f = first term and r = the rest,
// (f + r)^n = sum_{k=0}^{n} C(n,k) f^k r^(n-k), and r^(n-k) is expanded recursively
// (so an m-term base unfolds into its full multinomial). base is already expanded and
// n >= 0; the caller has bounded n and the resulting term count.
auto expand_power(const Expr& base, std::int64_t n) -> Result<Expr> {
    if (n == 0) {
        return Expr::integer(1);
    }
    if (n == 1) {
        return base;
    }
    std::vector<Expr> terms = as_terms(base);
    if (terms.size() < 2) {
        return make_pow(base, n);  // not a sum: nothing to distribute
    }
    const Expr f = terms.front();
    const Expr r = terms.size() == 2
                       ? terms[1]
                       : Expr::sum(std::vector<Expr>(terms.begin() + 1, terms.end()));

    std::vector<Expr> out;
    out.reserve(static_cast<std::size_t>(n) + 1);
    for (std::int64_t k = 0; k <= n; ++k) {
        auto coeff = binomial(n, k);
        if (!coeff) {
            return make_error<Expr>(coeff.error());
        }
        const Expr fk = make_pow(f, k);
        auto rnk = expand_power(r, n - k);
        if (!rnk) {
            return rnk;
        }
        Expr term = expand_product(fk, *rnk);
        if (*coeff != 1) {
            term = expand_product(Expr::integer(*coeff), term);
        }
        out.push_back(std::move(term));
    }
    return Expr::sum(std::move(out));
}

// Recursively expand a node: distribute products, expand qualifying powers of sums,
// and rebuild sums / function applications from expanded operands. Returns the raw
// (pre-simplify) expanded tree; expand() runs simplify once at the end.
auto expand_raw(const Expr& u) -> Result<Expr> {
    return std::visit(
        [&u]<typename T>(const T& n) -> Result<Expr> {
            if constexpr (std::is_same_v<T, SymbolNode> || std::is_same_v<T, ConstantNode>) {
                return u;
            } else if constexpr (std::is_same_v<T, AddNode>) {
                std::vector<Expr> terms;
                terms.reserve(n.terms.size());
                for (const Expr& t : n.terms) {
                    auto e = expand_raw(t);
                    if (!e) {
                        return e;
                    }
                    terms.push_back(std::move(*e));
                }
                return Expr::sum(std::move(terms));
            } else if constexpr (std::is_same_v<T, MulNode>) {
                if (n.factors.empty()) {
                    return Expr::integer(1);
                }
                auto acc = expand_raw(n.factors.front());
                if (!acc) {
                    return acc;
                }
                Expr result = std::move(*acc);
                for (std::size_t i = 1; i < n.factors.size(); ++i) {
                    auto next = expand_raw(n.factors[i]);
                    if (!next) {
                        return next;
                    }
                    result = expand_product(result, *next);
                }
                return result;
            } else if constexpr (std::is_same_v<T, PowerNode>) {
                auto base = expand_raw(n.base);
                if (!base) {
                    return base;
                }
                auto exponent = expand_raw(n.exponent);
                if (!exponent) {
                    return exponent;
                }
                if (auto ni = as_nonneg_int(*exponent)) {
                    const std::int64_t nn = *ni;
                    if (nn > max_expand_exponent) {
                        return Expr::power(*base, *exponent);  // over cap: leave intact
                    }
                    const std::vector<Expr> parts = as_terms(*base);
                    if (parts.size() < 2) {
                        return Expr::power(*base, *exponent);  // base not a sum: nothing to do
                    }
                    // Estimate the expanded monomial count C(n + m - 1, m - 1); bail if it
                    // overflows int64 or exceeds the term cap.
                    const std::int64_t m = static_cast<std::int64_t>(parts.size());
                    auto estimate = binomial(nn + m - 1, m - 1);
                    if (!estimate || *estimate > max_expand_terms) {
                        return Expr::power(*base, *exponent);
                    }
                    return expand_power(*base, nn);
                }
                return Expr::power(*base, *exponent);  // symbolic / negative / rational exponent
            } else if constexpr (std::is_same_v<T, FunctionNode>) {
                std::vector<Expr> args;
                args.reserve(n.args.size());
                for (const Expr& a : n.args) {
                    auto e = expand_raw(a);
                    if (!e) {
                        return e;
                    }
                    args.push_back(std::move(*e));
                }
                return Expr::apply(n.name, std::move(args));
            } else {
                static_assert(always_false<T>, "expand_raw: unhandled ExprNode kind");
            }
        },
        u.node().value);
}

}  // namespace

auto expand(const Expr& u) -> Result<Expr> {
    auto expanded = expand_raw(u);
    if (!expanded) {
        return expanded;
    }
    return simplify(*expanded);  // collect like terms in the distributed form
}

}  // namespace nimblecas
