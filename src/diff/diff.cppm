// NimbleCAS symbolic differentiation (ROADMAP 7.19).
// @author Olumuyiwa Oluwasanmi
//
// differentiate(u, x) applies d/dx structurally (sum, product/Leibniz, general
// power, and chain rules) and returns the automatically-simplified result. The
// derivatives of known elementary functions come from a small table; an unknown or
// multi-argument function yields an unevaluated Derivative(u, x) placeholder so the
// operation is total (Rule 32 — no exceptions).

export module nimblecas.diff;

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.simplify;

export namespace nimblecas {

[[nodiscard]] auto differentiate(const Expr& u, std::string_view var) -> Result<Expr>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

template <typename...>
inline constexpr bool always_false = false;

[[nodiscard]] auto derivative_raw(const Expr& u, std::string_view var) -> Expr;

// Small construction helpers for the derivative table.
[[nodiscard]] auto call(std::string name, const Expr& a) -> Expr {
    return Expr::apply(std::move(name), {a});
}
[[nodiscard]] auto neg(const Expr& a) -> Expr {
    return Expr::product({Expr::integer(-1), a});
}
[[nodiscard]] auto square(const Expr& a) -> Expr {
    return Expr::power(a, Expr::integer(2));
}
// (a)^(-1) — reciprocal.
[[nodiscard]] auto recip(const Expr& a) -> Expr {
    return Expr::power(a, Expr::integer(-1));
}
// (a)^(-1/2) — inverse square root.
[[nodiscard]] auto inv_sqrt(const Expr& a) -> Expr {
    return Expr::power(a, Expr::rational(-1, 2).value());
}
[[nodiscard]] auto one_plus(const Expr& a) -> Expr {
    return Expr::sum({Expr::integer(1), a});
}
[[nodiscard]] auto one_minus(const Expr& a) -> Expr {
    return Expr::sum({Expr::integer(1), neg(a)});
}

// Outer derivative f'(arg) for a known unary function f, else nullopt. Covers
// elementary, inverse-trig, hyperbolic, and special functions (ROADMAP 7.1/7.19).
[[nodiscard]] auto known_function_derivative(std::string_view name, const Expr& u)
    -> std::optional<Expr> {
    // --- exponential / logarithm / roots ---
    if (name == "exp") {
        return call("exp", u);
    }
    if (name == "ln") {
        return recip(u);  // 1/u
    }
    if (name == "sqrt") {
        return Expr::product({Expr::rational(1, 2).value(), inv_sqrt(u)});  // (1/2) u^(-1/2)
    }

    // --- trigonometric ---
    if (name == "sin") {
        return call("cos", u);
    }
    if (name == "cos") {
        return neg(call("sin", u));
    }
    if (name == "tan") {
        return one_plus(square(call("tan", u)));  // 1 + tan^2
    }
    if (name == "cot") {
        return neg(one_plus(square(call("cot", u))));  // -(1 + cot^2)
    }
    if (name == "sec") {
        return Expr::product({call("sec", u), call("tan", u)});  // sec*tan
    }
    if (name == "csc") {
        return neg(Expr::product({call("csc", u), call("cot", u)}));  // -csc*cot
    }

    // --- inverse trigonometric ---
    if (name == "asin") {
        return inv_sqrt(one_minus(square(u)));  // (1 - u^2)^(-1/2)
    }
    if (name == "acos") {
        return neg(inv_sqrt(one_minus(square(u))));
    }
    if (name == "atan") {
        return recip(one_plus(square(u)));  // 1/(1 + u^2)
    }

    // --- hyperbolic ---
    if (name == "sinh") {
        return call("cosh", u);
    }
    if (name == "cosh") {
        return call("sinh", u);
    }
    if (name == "tanh") {
        return one_minus(square(call("tanh", u)));  // 1 - tanh^2
    }
    if (name == "asinh") {
        return inv_sqrt(one_plus(square(u)));  // (1 + u^2)^(-1/2)
    }
    if (name == "acosh") {
        return inv_sqrt(Expr::sum({square(u), Expr::integer(-1)}));  // (u^2 - 1)^(-1/2)
    }
    if (name == "atanh") {
        return recip(one_minus(square(u)));  // 1/(1 - u^2)
    }

    // --- special functions (ROADMAP 7.1) ---
    if (name == "erf") {
        // (2/sqrt(pi)) * exp(-u^2)
        return Expr::product({Expr::integer(2), recip(call("sqrt", Expr::symbol("pi"))),
                              call("exp", neg(square(u)))});
    }
    if (name == "erfc") {
        return neg(Expr::product({Expr::integer(2), recip(call("sqrt", Expr::symbol("pi"))),
                                  call("exp", neg(square(u)))}));
    }
    if (name == "gamma") {
        // gamma'(u) = gamma(u) * digamma(u)
        return Expr::product({call("gamma", u), call("digamma", u)});
    }
    if (name == "lambertW") {
        // W'(u) = W(u) / (u * (1 + W(u)))
        Expr w = call("lambertW", u);
        return Expr::product({w, recip(Expr::product({u, one_plus(w)}))});
    }

    return std::nullopt;
}

auto derivative_raw(const Expr& u, std::string_view var) -> Expr {
    return std::visit(
        [&]<typename T>(const T& n) -> Expr {
            if constexpr (std::is_same_v<T, ConstantNode>) {
                return Expr::integer(0);
            } else if constexpr (std::is_same_v<T, SymbolNode>) {
                return Expr::integer(n.name == var ? 1 : 0);
            } else if constexpr (std::is_same_v<T, AddNode>) {
                std::vector<Expr> terms;
                terms.reserve(n.terms.size());
                for (const Expr& t : n.terms) {
                    terms.push_back(derivative_raw(t, var));
                }
                return Expr::sum(std::move(terms));
            } else if constexpr (std::is_same_v<T, MulNode>) {
                // Leibniz: d(prod f_i) = sum_i ( f_i' * prod_{j!=i} f_j )
                std::vector<Expr> terms;
                terms.reserve(n.factors.size());
                for (std::size_t i = 0; i < n.factors.size(); ++i) {
                    std::vector<Expr> parts;
                    parts.reserve(n.factors.size());
                    parts.push_back(derivative_raw(n.factors[i], var));
                    for (std::size_t j = 0; j < n.factors.size(); ++j) {
                        if (j != i) {
                            parts.push_back(n.factors[j]);
                        }
                    }
                    terms.push_back(Expr::product(std::move(parts)));
                }
                return Expr::sum(std::move(terms));
            } else if constexpr (std::is_same_v<T, PowerNode>) {
                // d(f^g) = f^g * ( g'*ln(f) + g * f' / f )
                const Expr& f = n.base;
                const Expr& g = n.exponent;
                Expr df = derivative_raw(f, var);
                Expr dg = derivative_raw(g, var);
                Expr term1 = Expr::product({dg, Expr::apply("ln", {f})});
                Expr term2 = Expr::product({g, df, Expr::power(f, Expr::integer(-1))});
                return Expr::product({Expr::power(f, g), Expr::sum({term1, term2})});
            } else if constexpr (std::is_same_v<T, FunctionNode>) {
                if (n.args.size() == 1) {
                    if (auto outer = known_function_derivative(n.name, n.args.front())) {
                        // chain rule: f'(arg) * arg'
                        return Expr::product({*outer, derivative_raw(n.args.front(), var)});
                    }
                }
                // unknown / multi-argument: leave an unevaluated derivative
                return Expr::apply("Derivative", {u, Expr::symbol(std::string(var))});
            } else {
                static_assert(always_false<T>, "derivative_raw: unhandled ExprNode kind");
            }
        },
        u.node().value);
}

}  // namespace

auto differentiate(const Expr& u, std::string_view var) -> Result<Expr> {
    return simplify(derivative_raw(u, var));
}

}  // namespace nimblecas
