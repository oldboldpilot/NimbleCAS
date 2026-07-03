// NimbleCAS automatic simplification (Cohen ASAE).
// @author Olumuyiwa Oluwasanmi
//
// simplify(u) transforms an expression into an Automatically Simplified Algebraic
// Expression: constants are folded with exact (overflow-checked) rational
// arithmetic, algebraic identities are applied (u+0->u, u*1->u, u*0->0, u^0->1,
// u^1->u, ...), nested sums/products are flattened, operands are put in a canonical
// order (so x+y and y+x agree), and like terms / like bases are combined
// (n*x + m*x -> (n+m)*x, x^a * x^b -> x^(a+b)).
//
// Numeric domain: integers and exact rationals are folded precisely with overflow
// detection (returning MathError::overflow); any double constant degrades that
// group to double arithmetic. Undefined forms (0^0, 0^negative) return a MathError
// via the railway model (Rule 32).

export module nimblecas.simplify;

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.parallel;

export namespace nimblecas {

[[nodiscard]] auto simplify(const Expr& u) -> Result<Expr>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

template <typename...>
inline constexpr bool always_false = false;

// --- numeric constant view -------------------------------------------------

// A constant is either an exact rational num/den (den > 0) or an IEEE double.
struct ConstVal {
    bool exact;
    std::int64_t num;
    std::int64_t den;
    double dbl;
    [[nodiscard]] auto as_double() const -> double {
        return exact ? static_cast<double>(num) / static_cast<double>(den) : dbl;
    }
};

[[nodiscard]] auto as_constant(const Expr& e) -> std::optional<ConstVal> {
    const auto* c = std::get_if<ConstantNode>(&e.node().value);
    if (c == nullptr) {
        return std::nullopt;
    }
    return std::visit(
        []<typename V>(const V& v) -> ConstVal {
            if constexpr (std::is_same_v<V, std::int64_t>) {
                return ConstVal{.exact = true, .num = v, .den = 1, .dbl = 0.0};
            } else if constexpr (std::is_same_v<V, double>) {
                return ConstVal{.exact = false, .num = 0, .den = 1, .dbl = v};
            } else {  // std::pair<int64,int64>
                return ConstVal{.exact = true, .num = v.first, .den = v.second, .dbl = 0.0};
            }
        },
        c->value);
}

[[nodiscard]] auto is_constant(const Expr& e) -> bool {
    return std::holds_alternative<ConstantNode>(e.node().value);
}

[[nodiscard]] auto is_zero(const Expr& e) -> bool {
    auto c = as_constant(e);
    return c && (c->exact ? c->num == 0 : c->dbl == 0.0);
}

[[nodiscard]] auto is_one(const Expr& e) -> bool {
    auto c = as_constant(e);
    return c && (c->exact ? (c->num == 1 && c->den == 1) : c->dbl == 1.0);
}

// Build a canonical constant Expr from an exact rational: integers collapse to
// ConstantNode integers, everything else is a reduced fraction. Guards INT64_MIN
// and zero denominators (Rule 32).
[[nodiscard]] auto make_number(std::int64_t num, std::int64_t den) -> Result<Expr> {
    if (den == 0) {
        return make_error<Expr>(MathError::division_by_zero);
    }
    constexpr std::int64_t int64_min = std::numeric_limits<std::int64_t>::min();
    if (num == int64_min || den == int64_min) {
        return make_error<Expr>(MathError::overflow);
    }
    if (den < 0) {
        num = -num;
        den = -den;
    }
    const std::int64_t g = std::gcd(num, den);
    if (g > 1) {
        num /= g;
        den /= g;
    }
    if (den == 1) {
        return Expr::integer(num);
    }
    return Expr::rational(num, den);
}

// Overflow-checked int64 helpers (true == overflow).
[[nodiscard]] auto mul_ov(std::int64_t a, std::int64_t b, std::int64_t& out) -> bool {
    return __builtin_mul_overflow(a, b, &out);
}
[[nodiscard]] auto add_ov(std::int64_t a, std::int64_t b, std::int64_t& out) -> bool {
    return __builtin_add_overflow(a, b, &out);
}

[[nodiscard]] auto add_constants(const ConstVal& a, const ConstVal& b) -> Result<Expr> {
    if (!a.exact || !b.exact) {
        return Expr::real(a.as_double() + b.as_double());
    }
    // a/b + c/d = (a*d + c*b) / (b*d)
    std::int64_t ad = 0;
    std::int64_t cb = 0;
    std::int64_t bd = 0;
    std::int64_t numerator = 0;
    if (mul_ov(a.num, b.den, ad) || mul_ov(b.num, a.den, cb) ||
        mul_ov(a.den, b.den, bd) || add_ov(ad, cb, numerator)) {
        return make_error<Expr>(MathError::overflow);
    }
    return make_number(numerator, bd);
}

[[nodiscard]] auto mul_constants(const ConstVal& a, const ConstVal& b) -> Result<Expr> {
    if (!a.exact || !b.exact) {
        return Expr::real(a.as_double() * b.as_double());
    }
    std::int64_t numerator = 0;
    std::int64_t denominator = 0;
    if (mul_ov(a.num, b.num, numerator) || mul_ov(a.den, b.den, denominator)) {
        return make_error<Expr>(MathError::overflow);
    }
    return make_number(numerator, denominator);
}

// base^exp for exp >= 0 via exponentiation by squaring (O(log exp), not O(exp), so
// crafted large exponents cannot hang the simplifier). Returns true on overflow.
[[nodiscard]] auto ipow_checked(std::int64_t base, std::int64_t exp, std::int64_t& out) -> bool {
    std::int64_t result = 1;
    std::int64_t b = base;
    while (exp > 0) {
        if ((exp & 1) != 0 && mul_ov(result, b, result)) {
            return true;
        }
        exp >>= 1;
        if (exp > 0 && mul_ov(b, b, b)) {
            return true;
        }
    }
    out = result;
    return false;
}

// base^exponent for an exact rational base and integer exponent, overflow-checked.
[[nodiscard]] auto pow_constant(const ConstVal& base, std::int64_t exponent) -> Result<Expr> {
    if (!base.exact) {
        return Expr::real(std::pow(base.dbl, static_cast<double>(exponent)));
    }
    // Negating INT64_MIN is signed-overflow UB; reject before the sign flip below.
    if (exponent == std::numeric_limits<std::int64_t>::min()) {
        return make_error<Expr>(MathError::overflow);
    }
    std::int64_t num = base.num;
    std::int64_t den = base.den;
    if (exponent < 0) {
        if (num == 0) {
            return make_error<Expr>(MathError::division_by_zero);
        }
        std::swap(num, den);
        exponent = -exponent;  // safe: exponent != INT64_MIN (guarded above)
    }
    std::int64_t rnum = 0;
    std::int64_t rden = 0;
    if (ipow_checked(num, exponent, rnum) || ipow_checked(den, exponent, rden)) {
        return make_error<Expr>(MathError::overflow);
    }
    return make_number(rnum, rden);
}

// --- forward declarations of the mutually-recursive simplifiers ------------
[[nodiscard]] auto simplify_impl(const Expr& u) -> Result<Expr>;
[[nodiscard]] auto simplify_power(const Expr& base, const Expr& exponent) -> Result<Expr>;
[[nodiscard]] auto simplify_sum(std::vector<Expr> terms) -> Result<Expr>;
[[nodiscard]] auto simplify_product(std::vector<Expr> factors) -> Result<Expr>;

// Split a sum term into (constant coefficient, remaining factor product). A leading
// constant factor of a product is the coefficient; otherwise the coefficient is 1.
[[nodiscard]] auto split_coefficient(const Expr& term) -> std::pair<Expr, Expr> {
    if (const auto* m = std::get_if<MulNode>(&term.node().value)) {
        if (!m->factors.empty() && is_constant(m->factors.front())) {
            Expr coeff = m->factors.front();
            if (m->factors.size() == 2) {
                return {coeff, m->factors[1]};
            }
            std::vector<Expr> rest(m->factors.begin() + 1, m->factors.end());
            return {coeff, Expr::product(std::move(rest))};
        }
    }
    return {Expr::integer(1), term};
}

// Split a product factor into (base, exponent).
[[nodiscard]] auto split_base_exponent(const Expr& factor) -> std::pair<Expr, Expr> {
    if (const auto* p = std::get_if<PowerNode>(&factor.node().value)) {
        return {p->base, p->exponent};
    }
    return {factor, Expr::integer(1)};
}

// Flatten nested Add terms / Mul factors into a single level. Uses a separate
// worklist stack: splicing children back into the source vector while iterating it
// would alias the container across a reallocation (UB). Order is preserved.
[[nodiscard]] auto flatten_terms(std::vector<Expr> items) -> std::vector<Expr> {
    std::vector<Expr> out;
    std::vector<Expr> stack(std::make_move_iterator(items.rbegin()),
                            std::make_move_iterator(items.rend()));
    while (!stack.empty()) {
        Expr e = std::move(stack.back());
        stack.pop_back();
        if (const auto* a = std::get_if<AddNode>(&e.node().value)) {
            for (auto it = a->terms.rbegin(); it != a->terms.rend(); ++it) {
                stack.push_back(*it);
            }
        } else {
            out.push_back(std::move(e));
        }
    }
    return out;
}

[[nodiscard]] auto flatten_factors(std::vector<Expr> items) -> std::vector<Expr> {
    std::vector<Expr> out;
    std::vector<Expr> stack(std::make_move_iterator(items.rbegin()),
                            std::make_move_iterator(items.rend()));
    while (!stack.empty()) {
        Expr e = std::move(stack.back());
        stack.pop_back();
        if (const auto* m = std::get_if<MulNode>(&e.node().value)) {
            for (auto it = m->factors.rbegin(); it != m->factors.rend(); ++it) {
                stack.push_back(*it);
            }
        } else {
            out.push_back(std::move(e));
        }
    }
    return out;
}

auto simplify_power(const Expr& base, const Expr& exponent) -> Result<Expr> {
    const auto exp_c = as_constant(exponent);
    const auto base_c = as_constant(base);

    if (exp_c) {
        if (is_zero(exponent)) {
            if (is_zero(base)) {
                return make_error<Expr>(MathError::undefined_value);  // 0^0
            }
            return Expr::integer(1);
        }
        if (is_one(exponent)) {
            return base;
        }
    }
    if (is_one(base)) {
        return Expr::integer(1);
    }
    if (is_zero(base) && exp_c) {
        if (exp_c->exact && exp_c->den == 1 && exp_c->num > 0) {
            return Expr::integer(0);
        }
        if (exp_c->as_double() > 0.0) {
            return Expr::integer(0);
        }
        return make_error<Expr>(MathError::division_by_zero);  // 0^(<=0 already handled 0)
    }

    // constant ^ integer-constant  ->  fold
    if (base_c && exp_c && exp_c->exact && exp_c->den == 1) {
        return pow_constant(*base_c, exp_c->num);
    }

    // (v^w)^n = v^(w*n)  when n is an integer constant
    if (exp_c && exp_c->exact && exp_c->den == 1) {
        if (const auto* p = std::get_if<PowerNode>(&base.node().value)) {
            auto inner = simplify_product({p->exponent, exponent});
            if (!inner) {
                return inner;
            }
            return simplify_power(p->base, *inner);
        }
    }

    return Expr::power(base, exponent);
}

auto simplify_sum(std::vector<Expr> terms) -> Result<Expr> {
    terms = flatten_terms(std::move(terms));  // self-contained: never rely on caller flattening
    Expr constant_sum = Expr::integer(0);
    // Groups of (coefficient, rest). Matched by structural equality — NOT by a
    // to_string key, which is not injective (e.g. a symbol named "f(x)" would
    // collide with the call f(x) and wrongly fuse).
    std::vector<std::pair<Expr, Expr>> groups;

    for (const Expr& term : terms) {
        if (is_constant(term)) {
            auto folded = add_constants(*as_constant(constant_sum), *as_constant(term));
            if (!folded) {
                return folded;
            }
            constant_sum = *folded;
            continue;
        }
        auto [coeff, rest] = split_coefficient(term);
        auto it = std::ranges::find_if(
            groups, [&rest](const auto& g) { return g.second.is_equivalent_to(rest); });
        if (it == groups.end()) {
            groups.emplace_back(std::move(coeff), std::move(rest));
        } else {
            auto combined = add_constants(*as_constant(it->first), *as_constant(coeff));
            if (!combined) {
                return combined;
            }
            it->first = *combined;
        }
    }

    std::vector<Expr> result;
    for (auto& [coeff, rest] : groups) {
        if (is_zero(coeff)) {
            continue;
        }
        if (is_one(coeff)) {
            result.push_back(rest);
        } else {
            auto term = simplify_product({coeff, rest});
            if (!term) {
                return term;
            }
            result.push_back(*term);
        }
    }
    if (!is_zero(constant_sum)) {
        result.push_back(constant_sum);
    }

    if (result.empty()) {
        return Expr::integer(0);
    }
    if (result.size() == 1) {
        return result.front();
    }
    std::ranges::sort(result, {}, [](const Expr& e) { return e.to_string(); });
    return Expr::sum(std::move(result));
}

auto simplify_product(std::vector<Expr> factors) -> Result<Expr> {
    factors = flatten_factors(std::move(factors));  // self-contained: flatten nested products
    Expr constant_product = Expr::integer(1);
    // Groups of (base, exponent), matched by structural equality (see simplify_sum).
    std::vector<std::pair<Expr, Expr>> groups;

    for (const Expr& factor : factors) {
        if (is_zero(factor)) {
            return Expr::integer(0);  // absorbing element
        }
        if (is_constant(factor)) {
            auto folded = mul_constants(*as_constant(constant_product), *as_constant(factor));
            if (!folded) {
                return folded;
            }
            constant_product = *folded;
            continue;
        }
        auto [base, exponent] = split_base_exponent(factor);
        auto it = std::ranges::find_if(
            groups, [&base](const auto& g) { return g.first.is_equivalent_to(base); });
        if (it == groups.end()) {
            groups.emplace_back(std::move(base), std::move(exponent));
        } else {
            auto combined = simplify_sum({it->second, exponent});
            if (!combined) {
                return combined;
            }
            it->second = *combined;
        }
    }

    std::vector<Expr> result;
    for (auto& [base, exponent] : groups) {
        auto factor = simplify_power(base, exponent);
        if (!factor) {
            return factor;
        }
        if (is_one(*factor)) {
            continue;
        }
        result.push_back(*factor);
    }

    if (is_zero(constant_product)) {
        return Expr::integer(0);
    }
    if (result.empty()) {
        return constant_product;
    }
    if (!is_one(constant_product)) {
        result.push_back(constant_product);
    }
    if (result.size() == 1) {
        return result.front();
    }
    std::ranges::sort(result, {}, [](const Expr& e) { return e.to_string(); });
    return Expr::product(std::move(result));
}

// Recursively simplify operands, flattening nested sums/products, then apply the
// operator-specific simplification.
auto simplify_impl(const Expr& u) -> Result<Expr> {
    return std::visit(
        [&u]<typename T>(const T& n) -> Result<Expr> {
            if constexpr (std::is_same_v<T, SymbolNode>) {
                return u;
            } else if constexpr (std::is_same_v<T, ConstantNode>) {
                // normalise pair-form integers (e.g. 4/1 -> 4, 0/1 -> 0)
                if (const auto* pr = std::get_if<std::pair<std::int64_t, std::int64_t>>(&n.value)) {
                    if (pr->second == 1) {
                        return Expr::integer(pr->first);
                    }
                }
                return u;
            } else if constexpr (std::is_same_v<T, PowerNode>) {
                auto base = simplify_impl(n.base);
                if (!base) {
                    return base;
                }
                auto exponent = simplify_impl(n.exponent);
                if (!exponent) {
                    return exponent;
                }
                return simplify_power(*base, *exponent);
            } else if constexpr (std::is_same_v<T, AddNode>) {
                // Simplify terms concurrently for large subtrees; independent because
                // the tree is immutable (COW). Errors resolved in index order after.
                const bool par = u.size() >= parallel::parallel_cost_threshold;
                auto results = parallel::transform_index_if(
                    par, n.terms.size(), [&](std::size_t i) { return simplify_impl(n.terms[i]); });
                std::vector<Expr> terms;
                terms.reserve(results.size());
                for (auto& s : results) {
                    if (!s) {
                        return s;
                    }
                    if (const auto* inner = std::get_if<AddNode>(&s->node().value)) {
                        terms.insert(terms.end(), inner->terms.begin(), inner->terms.end());
                    } else {
                        terms.push_back(std::move(*s));
                    }
                }
                return simplify_sum(std::move(terms));
            } else if constexpr (std::is_same_v<T, MulNode>) {
                const bool par = u.size() >= parallel::parallel_cost_threshold;
                auto results = parallel::transform_index_if(
                    par, n.factors.size(),
                    [&](std::size_t i) { return simplify_impl(n.factors[i]); });
                std::vector<Expr> factors;
                factors.reserve(results.size());
                for (auto& s : results) {
                    if (!s) {
                        return s;
                    }
                    if (const auto* inner = std::get_if<MulNode>(&s->node().value)) {
                        factors.insert(factors.end(), inner->factors.begin(),
                                       inner->factors.end());
                    } else {
                        factors.push_back(std::move(*s));
                    }
                }
                return simplify_product(std::move(factors));
            } else if constexpr (std::is_same_v<T, FunctionNode>) {
                const bool par = u.size() >= parallel::parallel_cost_threshold;
                auto results = parallel::transform_index_if(
                    par, n.args.size(), [&](std::size_t i) { return simplify_impl(n.args[i]); });
                std::vector<Expr> args;
                args.reserve(results.size());
                for (auto& s : results) {
                    if (!s) {
                        return s;
                    }
                    args.push_back(std::move(*s));
                }
                return Expr::apply(n.name, std::move(args));
            } else {
                static_assert(always_false<T>, "simplify: unhandled ExprNode kind");
            }
        },
        u.node().value);
}

}  // namespace

auto simplify(const Expr& u) -> Result<Expr> { return simplify_impl(u); }

}  // namespace nimblecas
