// NimbleCAS numerical evaluator: samples a symbolic Expr to an IEEE-754 double at a
// point (or a map of variable bindings).
// @author Olumuyiwa Oluwasanmi
//
// HONESTY BOUNDARY: this module is NUMERICAL, not exact. It is the numeric-sampling
// counterpart to the exact symbolic engine (nimblecas.symbolic) — every leaf and
// intermediate result is an IEEE-754 double, so results carry ordinary floating-point
// rounding error, unlike the exact rational/exact-per-term arithmetic elsewhere in
// this codebase. The symbols "pi" and "e" are given their numeric std::numbers
// mappings here purely for evaluation purposes; that is a rendering-time convenience,
// not a claim that they are exact constants in this module. An unbound symbol or an
// unrecognised function head is never guessed at (never silently 0) — it is an honest
// MathError::domain_error (Rule 32). A non-finite intermediate (NaN/Inf, e.g. from
// log of a negative number or division producing Inf) is likewise never propagated
// silently: it is caught immediately after the node that produced it and converted to
// MathError::domain_error, so a caller never receives a NaN/Inf dressed up as a valid
// double.

export module nimblecas.evalnum;

import std;
import nimblecas.core;
import nimblecas.symbolic;

export namespace nimblecas {

// Evaluate expression `e` to a double, substituting `x` for every occurrence of the
// symbol named `var`. Delegates to the map overload with a single binding so the
// evaluation logic exists in exactly one place.
[[nodiscard]] auto eval_double(const Expr& e, std::string_view var, double x) -> Result<double>;

// Evaluate expression `e` to a double using `bindings` to resolve symbols. Symbols
// "pi" and "e" resolve to std::numbers::pi / std::numbers::e when not shadowed by an
// explicit binding of the same name. Any other unbound symbol, or a function whose
// name is not in the elementary-function whitelist (sin, cos, tan, cot, sec, csc,
// sinh, cosh, tanh, exp, log, sqrt, abs), yields MathError::domain_error. Any
// non-finite intermediate result also yields MathError::domain_error.
[[nodiscard]] auto eval_double(const Expr& e, const std::unordered_map<std::string, double>& bindings)
    -> Result<double>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

namespace {

template <typename...>
inline constexpr bool always_false = false;

// True iff `v` is not a finite IEEE-754 value (NaN or +-Inf).
[[nodiscard]] auto non_finite(double v) -> bool {
    return std::isnan(v) || std::isinf(v);
}

// Numeric value of a ConstantNode: int64 as-is, double as-is, rational num/den
// computed in double.
[[nodiscard]] auto eval_constant(const ConstantNode& c) -> double {
    return std::visit(
        []<typename V>(const V& v) -> double {
            if constexpr (std::is_same_v<V, std::int64_t>) {
                return static_cast<double>(v);
            } else if constexpr (std::is_same_v<V, double>) {
                return v;
            } else {  // std::pair<std::int64_t, std::int64_t>
                return static_cast<double>(v.first) / static_cast<double>(v.second);
            }
        },
        c.value);
}

// The reciprocal-based trig whitelist: cot = cos/sin, sec = 1/cos, csc = 1/sin. All
// other whitelisted names map straight onto the corresponding std function.
[[nodiscard]] auto eval_function(const std::string& name, double arg) -> Result<double> {
    if (name == "sin") return std::sin(arg);
    if (name == "cos") return std::cos(arg);
    if (name == "tan") return std::tan(arg);
    if (name == "cot") return std::cos(arg) / std::sin(arg);
    if (name == "sec") return 1.0 / std::cos(arg);
    if (name == "csc") return 1.0 / std::sin(arg);
    if (name == "sinh") return std::sinh(arg);
    if (name == "cosh") return std::cosh(arg);
    if (name == "tanh") return std::tanh(arg);
    if (name == "exp") return std::exp(arg);
    if (name == "log") return std::log(arg);
    if (name == "sqrt") return std::sqrt(arg);
    if (name == "abs") return std::abs(arg);
    return make_error<double>(MathError::domain_error);
}

[[nodiscard]] auto eval_node(const Expr& e, const std::unordered_map<std::string, double>& bindings)
    -> Result<double> {
    const auto result = std::visit(
        [&](const auto& n) -> Result<double> {
            using T = std::decay_t<decltype(n)>;
            if constexpr (std::is_same_v<T, ConstantNode>) {
                return eval_constant(n);
            } else if constexpr (std::is_same_v<T, SymbolNode>) {
                if (const auto it = bindings.find(n.name); it != bindings.end()) {
                    return it->second;
                }
                if (n.name == "pi") return std::numbers::pi;
                if (n.name == "e") return std::numbers::e;
                return make_error<double>(MathError::domain_error);
            } else if constexpr (std::is_same_v<T, AddNode>) {
                double sum = 0.0;
                for (const Expr& term : n.terms) {
                    const auto v = eval_node(term, bindings);
                    if (!v) return v;
                    sum += *v;
                }
                return sum;
            } else if constexpr (std::is_same_v<T, MulNode>) {
                double product = 1.0;
                for (const Expr& factor : n.factors) {
                    const auto v = eval_node(factor, bindings);
                    if (!v) return v;
                    product *= *v;
                }
                return product;
            } else if constexpr (std::is_same_v<T, PowerNode>) {
                const auto base = eval_node(n.base, bindings);
                if (!base) return base;
                const auto exponent = eval_node(n.exponent, bindings);
                if (!exponent) return exponent;
                return std::pow(*base, *exponent);
            } else if constexpr (std::is_same_v<T, FunctionNode>) {
                if (n.args.size() != 1) {
                    return make_error<double>(MathError::domain_error);
                }
                const auto arg = eval_node(n.args[0], bindings);
                if (!arg) return arg;
                return eval_function(n.name, *arg);
            } else {
                static_assert(always_false<T>, "eval_node: unhandled ExprNode kind");
            }
        },
        e.node().value);
    if (!result) {
        return result;
    }
    if (non_finite(*result)) {
        return make_error<double>(MathError::domain_error);
    }
    return result;
}

}  // namespace

auto eval_double(const Expr& e, const std::unordered_map<std::string, double>& bindings)
    -> Result<double> {
    return eval_node(e, bindings);
}

auto eval_double(const Expr& e, std::string_view var, double x) -> Result<double> {
    const std::unordered_map<std::string, double> bindings{{std::string(var), x}};
    return eval_double(e, bindings);
}

}  // namespace nimblecas
