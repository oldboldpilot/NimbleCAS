// NimbleCAS table-driven symbolic Laplace transform (ROADMAP 7.6).
// @author Olumuyiwa Oluwasanmi
//
// laplace_transform(f, t, s) computes F(s) = L{f(t)} by LINEARITY over a small
// table of elementary forms in the variable t:
//
//   L{c}        = c / s                    (c free of t)
//   L{t^n}      = n! / s^(n+1)             (n a non-negative integer)
//   L{e^(a t)}  = 1 / (s - a)              (a free of t)
//   L{sin(a t)} = a / (s^2 + a^2)          (a a constant)
//   L{cos(a t)} = s / (s^2 + a^2)          (a a constant)
//   L{f + g}    = L{f} + L{g}              (linearity)
//   L{c g}      = c L{g}                   (c a constant factor, free of t)
//
// The transform is assembled unevaluated from Expr factories and then reduced by
// simplify() to a canonical form. Anything outside the table yields
// MathError::not_implemented; a factorial overflow yields MathError::overflow, so
// the operation is total (Rule 32 — no exceptions).

export module nimblecas.laplace;

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.simplify;

export namespace nimblecas {

// F(s) = L{f(t)}. The transform variable is `t` (the argument name in f) and the
// image variable is `s`. Elementary forms are recognised by structure; unmatched
// input returns MathError::not_implemented.
[[nodiscard]] auto laplace_transform(const Expr& f, std::string_view t, std::string_view s)
    -> Result<Expr>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// n! computed in int64, overflowing gracefully (20! fits, 21! does not).
[[nodiscard]] auto factorial(std::int64_t n) -> Result<std::int64_t> {
    std::int64_t result = 1;
    for (std::int64_t i = 2; i <= n; ++i) {
        if (__builtin_mul_overflow(result, i, &result)) {
            return make_error<std::int64_t>(MathError::overflow);
        }
    }
    return result;
}

// Extract a plain integer from a constant leaf (int64 directly, or an n/1 rational).
[[nodiscard]] auto integer_value(const Expr& e) -> std::optional<std::int64_t> {
    auto c = as<ConstantNode>(e.node().value);
    if (!c) {
        return std::nullopt;
    }
    if (auto i = as<std::int64_t>((*c)->value)) {
        return **i;
    }
    if (auto pair = as<std::pair<std::int64_t, std::int64_t>>((*c)->value)) {
        if ((*pair)->second == 1) {
            return (*pair)->first;
        }
    }
    return std::nullopt;
}

// L{t^n} = n! / s^(n+1) = n! * s^(-(n+1)), assembled unevaluated. n >= 0.
[[nodiscard]] auto transform_power_of_t(std::int64_t n, const Expr& s) -> Result<Expr> {
    auto fact = factorial(n);
    if (!fact) {
        return make_error<Expr>(fact.error());
    }
    std::int64_t exponent = 0;
    if (__builtin_add_overflow(n, std::int64_t{1}, &exponent)) {
        return make_error<Expr>(MathError::overflow);
    }
    return Expr::product({Expr::integer(*fact), Expr::power(s, Expr::integer(-exponent))});
}

// If arg has the linear form a*t with a free of t, return a (a == 1 for a bare t);
// otherwise nullopt. Matches a lone t or a product with exactly one bare t factor.
[[nodiscard]] auto linear_coefficient(const Expr& arg, const Expr& t) -> std::optional<Expr> {
    if (arg.is_equivalent_to(t)) {
        return Expr::integer(1);
    }
    if (auto mul = as<MulNode>(arg.node().value)) {
        std::vector<Expr> rest;
        bool found_t = false;
        for (const Expr& factor : (*mul)->factors) {
            if (!found_t && factor.is_equivalent_to(t)) {
                found_t = true;
            } else if (free_of(factor, t)) {
                rest.push_back(factor);
            } else {
                return std::nullopt;  // a non-t factor still depends on t: not linear
            }
        }
        if (!found_t) {
            return std::nullopt;
        }
        if (rest.empty()) {
            return Expr::integer(1);
        }
        if (rest.size() == 1) {
            return rest.front();
        }
        return Expr::product(std::move(rest));
    }
    return std::nullopt;
}

// Table entries for exp / sin / cos, each requiring an argument of the form a*t.
[[nodiscard]] auto transform_function(const FunctionNode& fn, const Expr& t, const Expr& s)
    -> Result<Expr> {
    if (fn.args.size() != 1) {
        return make_error<Expr>(MathError::not_implemented);
    }
    auto a = linear_coefficient(fn.args.front(), t);
    if (!a) {
        return make_error<Expr>(MathError::not_implemented);
    }
    if (fn.name == "exp") {
        // 1 / (s - a)
        Expr denom = Expr::sum({s, Expr::product({Expr::integer(-1), *a})});
        return Expr::power(std::move(denom), Expr::integer(-1));
    }
    if (fn.name == "sin" || fn.name == "cos") {
        // denominator s^2 + a^2, shared by both
        Expr denom = Expr::sum({Expr::power(s, Expr::integer(2)),
                                Expr::power(*a, Expr::integer(2))});
        Expr recip = Expr::power(std::move(denom), Expr::integer(-1));
        // L{sin(a t)} = a / (s^2 + a^2) ; L{cos(a t)} = s / (s^2 + a^2)
        Expr numerator = fn.name == "sin" ? *a : s;
        return Expr::product({std::move(numerator), std::move(recip)});
    }
    return make_error<Expr>(MathError::not_implemented);
}

// Assemble the (unsimplified) transform by structural dispatch over the table.
[[nodiscard]] auto transform(const Expr& f, const Expr& t, const Expr& s) -> Result<Expr> {
    // L{c} = c / s for any c free of t (constants, other symbols, or subexpressions).
    if (free_of(f, t)) {
        return Expr::product({f, Expr::power(s, Expr::integer(-1))});
    }

    const ExprNode& node = f.node();

    // A bare, t-dependent symbol can only be t itself (t^1).
    if (as<SymbolNode>(node.value)) {
        return transform_power_of_t(1, s);
    }

    // t^n: base == t, exponent a positive integer.
    if (auto power = as<PowerNode>(node.value)) {
        if ((*power)->base.is_equivalent_to(t)) {
            if (auto n = integer_value((*power)->exponent); n && *n >= 1) {
                return transform_power_of_t(*n, s);
            }
        }
        return make_error<Expr>(MathError::not_implemented);
    }

    // exp / sin / cos.
    if (auto fn = as<FunctionNode>(node.value)) {
        return transform_function(**fn, t, s);
    }

    // Linearity: transform each term and sum.
    if (auto add = as<AddNode>(node.value)) {
        std::vector<Expr> terms;
        terms.reserve((*add)->terms.size());
        for (const Expr& term : (*add)->terms) {
            auto lt = transform(term, t, s);
            if (!lt) {
                return lt;
            }
            terms.push_back(std::move(*lt));
        }
        return Expr::sum(std::move(terms));
    }

    // c * g: pull out the factors free of t (constants) and transform the single
    // remaining t-dependent factor. More than one t-dependent factor is unhandled.
    if (auto mul = as<MulNode>(node.value)) {
        std::vector<Expr> constants;
        std::vector<Expr> variables;
        for (const Expr& factor : (*mul)->factors) {
            if (free_of(factor, t)) {
                constants.push_back(factor);
            } else {
                variables.push_back(factor);
            }
        }
        if (variables.size() != 1) {
            return make_error<Expr>(MathError::not_implemented);
        }
        auto lg = transform(variables.front(), t, s);
        if (!lg) {
            return lg;
        }
        if (constants.empty()) {
            return lg;
        }
        std::vector<Expr> factors = std::move(constants);
        factors.push_back(std::move(*lg));
        return Expr::product(std::move(factors));
    }

    return make_error<Expr>(MathError::not_implemented);
}

}  // namespace

auto laplace_transform(const Expr& f, std::string_view t, std::string_view s) -> Result<Expr> {
    const Expr t_sym = Expr::symbol(std::string(t));
    const Expr s_sym = Expr::symbol(std::string(s));
    auto raw = transform(f, t_sym, s_sym);
    if (!raw) {
        return raw;
    }
    return simplify(*raw);
}

}  // namespace nimblecas
