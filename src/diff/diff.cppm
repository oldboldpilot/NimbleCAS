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

// Outer derivative f'(arg) for a known unary function f, else nullopt.
[[nodiscard]] auto known_function_derivative(std::string_view name, const Expr& arg)
    -> std::optional<Expr> {
    if (name == "sin") {
        return Expr::apply("cos", {arg});
    }
    if (name == "cos") {
        return Expr::product({Expr::integer(-1), Expr::apply("sin", {arg})});
    }
    if (name == "exp") {
        return Expr::apply("exp", {arg});
    }
    if (name == "ln") {
        return Expr::power(arg, Expr::integer(-1));  // 1/arg
    }
    if (name == "tan") {
        // 1 + tan(arg)^2
        return Expr::sum({Expr::power(Expr::apply("tan", {arg}), Expr::integer(2)),
                          Expr::integer(1)});
    }
    if (name == "sqrt") {
        // (1/2) * arg^(-1/2)
        return Expr::product({Expr::rational(1, 2).value(),
                              Expr::power(arg, Expr::rational(-1, 2).value())});
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
