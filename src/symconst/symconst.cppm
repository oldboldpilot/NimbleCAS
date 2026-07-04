// NimbleCAS symbolic mathematical constants — pi, e, Euler-Mascheroni gamma, and the
// golden ratio phi as expression-tree leaves (the long-open "pi/e as Expr leaves" item).
// @author Olumuyiwa Oluwasanmi
//
// The numeric layer (nimblecas.constants) already computes each of these to arbitrary
// precision as a correctly-rounded BigFloat; what was missing is a way to carry one
// *inside* a symbolic Expr so an expression such as 2*pi + e stays exact and structural
// until the user explicitly asks for a number. This module supplies that symbolic leaf
// plus the bridge back to the numeric layer.
//
// DESIGN — reserved-name symbols, not a new core node kind.
//   A mathematical constant here is simply an ordinary SymbolNode carrying a canonical
//   reserved name ("pi", "e", "gamma", "phi"). We deliberately do NOT add a new
//   alternative to the ExprNode variant: every existing visitor (structural equality,
//   FreeOf, Substitute, differentiate, simplify, to_latex, ...) already handles
//   SymbolNode exhaustively, so a constant leaf inherits all of that machinery for free
//   and the "dependent-false" exhaustiveness guards in those visitors stay satisfied.
//   The reserved spellings were chosen to render well through nimblecas.latex, whose
//   Greek-letter table maps "pi" -> \pi, "gamma" -> \gamma and "phi" -> \phi; "e"
//   renders verbatim as the upright letter e.
//
// WHAT "CONSTANT" MEANS HERE (the honest boundary).
//   * SYMBOLICALLY these leaves are constants only in the sense that they are free of
//     every variable. That is the whole reason differentiation already does the right
//     thing: d/dx of a SymbolNode whose name is not the differentiation variable is 0
//     (see nimblecas.diff), and simplify treats an unknown symbol as an opaque atom, so
//     pi, e, gamma and phi need no special-casing in either engine.
//   * NUMERICALLY their value is a transcendental (pi, e, gamma) or irrational-algebraic
//     (phi) real with NO exact rational representation — evaluate_constant therefore
//     returns a rounded BigFloat, never an exact Rational. This mirrors the discipline
//     documented in nimblecas.constants: a high-precision, correctly-rounded float.
//
// RESERVED-NAME CAVEAT (the deliberate trade-off).
//   Because a constant is just a named symbol, a user variable literally named "pi"
//   (Expr::symbol("pi")) ALIASES the constant: is_named_constant reports it as pi and
//   evaluate_constant will hand back 3.14159.... This collision is the accepted cost of
//   not growing the core variant; the four reserved spellings are documented so callers
//   can avoid them for free variables. (The elementary-function derivative table in
//   nimblecas.diff already uses Expr::symbol("pi") for exactly this constant, so the
//   reservation is consistent with the rest of the engine.)

export module nimblecas.symconst;

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.constants;
import nimblecas.bigfloat;

export namespace nimblecas {

// --- constant-leaf factories ---------------------------------------------------------
// Each returns the constant as an immutable SymbolNode with its canonical reserved name.
// They are free of every variable, so they differentiate to 0 and simplify as atoms via
// the existing engines with no further support code.
[[nodiscard]] auto pi() -> Expr;            // the circle constant, symbol "pi"
[[nodiscard]] auto e() -> Expr;             // Euler's number, symbol "e"
[[nodiscard]] auto euler_gamma() -> Expr;   // Euler-Mascheroni constant, symbol "gamma"
[[nodiscard]] auto golden_ratio() -> Expr;  // (1 + sqrt5)/2, symbol "phi"

// --- recognition ---------------------------------------------------------------------
// True iff u is one of the four reserved constant leaves. Note the aliasing caveat in the
// module header: any symbol spelled "pi"/"e"/"gamma"/"phi" satisfies this predicate.
[[nodiscard]] auto is_named_constant(const Expr& u) -> bool;

// The canonical reserved name of u if it is a named constant, else std::nullopt. The
// returned view points at static storage (not into u's node), so it never dangles.
[[nodiscard]] auto named_constant_name(const Expr& u) -> std::optional<std::string_view>;

// --- numeric bridge ------------------------------------------------------------------
// Evaluate u to a correctly-rounded BigFloat of ~`prec` significant bits.
//   * A single reserved constant leaf dispatches straight to nimblecas.constants.
//   * A compound expression built ONLY from constant leaves, integer/rational/real
//     literals and the operators +, -, *, / and integer powers ^ is evaluated
//     recursively at an elevated working precision and rounded once to `prec`. Being a
//     tree of rounded floats, the compound result is an approximation, not an exact
//     number (consistent with the constants module's precision discipline).
// Returns MathError::domain_error when prec <= 0, and MathError::not_implemented for any
// node this bridge cannot evaluate numerically: a non-reserved (free) symbol, a
// function application, or a non-integer exponent.
[[nodiscard]] auto evaluate_constant(const Expr& u, std::int64_t prec) -> Result<BigFloat>;

// Convenience: evaluate u to a native double. Internally computes at a precision well
// beyond a double's 53-bit mantissa so the returned value is faithful to double rounding.
[[nodiscard]] auto evaluate_constant_double(const Expr& u) -> Result<double>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// Dependent-false helper so a non-exhaustive std::visit chain fails to compile rather
// than run off the end of a non-void lambda (matches the guard used across the engine).
template <typename...>
inline constexpr bool always_false = false;

// The four canonical reserved spellings, in static storage so returned string_views are
// always valid regardless of the lifetime of any particular Expr.
inline constexpr std::string_view name_pi = "pi";
inline constexpr std::string_view name_e = "e";
inline constexpr std::string_view name_gamma = "gamma";
inline constexpr std::string_view name_phi = "phi";

// Guard bits added on top of the caller's requested precision: intermediate BigFloat
// operations (and the leaf constants themselves) are carried at this elevated precision
// and the final value is rounded once to `prec`, so the leading digits are stable.
inline constexpr std::int64_t guard_bits = 64;

// Map a reserved constant name to its numeric provider in nimblecas.constants, evaluated
// at `prec` bits. A name outside the reserved set is not a numeric constant here.
[[nodiscard]] auto named_constant_value(std::string_view name, std::int64_t prec)
    -> Result<BigFloat> {
    if (name == name_pi) {
        return pi(prec);
    }
    if (name == name_e) {
        return e(prec);
    }
    if (name == name_gamma) {
        return euler_mascheroni(prec);
    }
    if (name == name_phi) {
        return golden_ratio(prec);
    }
    return make_error<BigFloat>(MathError::not_implemented);
}

// If `exponent` is an integer literal (an int64 constant, or a canonical rational p/1),
// return that integer; otherwise std::nullopt. Only integer powers are numerically
// tractable here without introducing roots, so anything else stays not_implemented.
[[nodiscard]] auto integer_exponent(const Expr& exponent) -> std::optional<std::int64_t> {
    auto ec = as<ConstantNode>(exponent.node().value);
    if (!ec) {
        return std::nullopt;
    }
    const auto& val = (*ec)->value;
    if (std::holds_alternative<std::int64_t>(val)) {
        return std::get<std::int64_t>(val);
    }
    if (std::holds_alternative<std::pair<std::int64_t, std::int64_t>>(val)) {
        const auto [p, q] = std::get<std::pair<std::int64_t, std::int64_t>>(val);
        if (q == 1) {  // rationals are canonicalised, so q == 1 means the integer p
            return p;
        }
    }
    return std::nullopt;
}

// base^n for an integer n at `work` precision, by exponentiation-by-squaring. A negative
// exponent is |n| positive powers followed by a reciprocal (which surfaces
// division_by_zero when base is 0). INT64_MIN is rejected (overflow) because its
// magnitude is not representable as a positive std::int64_t.
[[nodiscard]] auto integer_power(const BigFloat& base, std::int64_t n, std::int64_t work)
    -> Result<BigFloat> {
    if (n == std::numeric_limits<std::int64_t>::min()) {
        return make_error<BigFloat>(MathError::overflow);
    }
    const bool negative = n < 0;
    auto magnitude = static_cast<std::uint64_t>(negative ? -n : n);

    auto acc = BigFloat::from_i64(1, work);  // n == 0 -> 1
    if (!acc) {
        return acc;
    }
    BigFloat result = *acc;
    BigFloat square = base;
    while (magnitude > 0) {
        if ((magnitude & 1U) != 0U) {
            auto r = result.multiply(square, work);
            if (!r) {
                return r;
            }
            result = *r;
        }
        magnitude >>= 1U;
        if (magnitude > 0) {
            auto s = square.multiply(square, work);
            if (!s) {
                return s;
            }
            square = *s;
        }
    }
    if (!negative) {
        return result;
    }
    auto one = BigFloat::from_i64(1, work);
    if (!one) {
        return one;
    }
    return one->divide(result, work);  // base^(-|n|); base == 0 -> division_by_zero
}

// Recursive numeric evaluator over the constant-only fragment grammar. Forward-declared
// because Add/Mul/Power recurse into it before its definition.
[[nodiscard]] auto eval_node(const Expr& u, std::int64_t work) -> Result<BigFloat>;

// A ConstantNode's stored number (int64, IEEE double, or exact rational) as a BigFloat.
[[nodiscard]] auto eval_constant(const ConstantNode& c, std::int64_t work) -> Result<BigFloat> {
    return std::visit(
        [work]<typename V>(const V& v) -> Result<BigFloat> {
            if constexpr (std::is_same_v<V, std::int64_t>) {
                return BigFloat::from_i64(v, work);
            } else if constexpr (std::is_same_v<V, double>) {
                return BigFloat::from_double(v, work);
            } else {  // std::pair<std::int64_t, std::int64_t> — an exact rational p/q
                auto num = BigFloat::from_i64(v.first, work);
                if (!num) {
                    return num;
                }
                auto den = BigFloat::from_i64(v.second, work);
                if (!den) {
                    return den;
                }
                return num->divide(*den, work);
            }
        },
        c.value);
}

auto eval_node(const Expr& u, std::int64_t work) -> Result<BigFloat> {
    return std::visit(
        [work]<typename T>(const T& n) -> Result<BigFloat> {
            if constexpr (std::is_same_v<T, SymbolNode>) {
                // A reserved constant leaf resolves numerically; any other symbol is a
                // free variable with no numeric value (not_implemented).
                return named_constant_value(n.name, work);
            } else if constexpr (std::is_same_v<T, ConstantNode>) {
                return eval_constant(n, work);
            } else if constexpr (std::is_same_v<T, AddNode>) {
                auto acc = BigFloat::from_i64(0, work);  // empty sum -> 0
                if (!acc) {
                    return acc;
                }
                for (const Expr& term : n.terms) {
                    auto value = eval_node(term, work);
                    if (!value) {
                        return value;
                    }
                    auto sum = acc->add(*value, work);
                    if (!sum) {
                        return sum;
                    }
                    acc = sum;
                }
                return acc;
            } else if constexpr (std::is_same_v<T, MulNode>) {
                auto acc = BigFloat::from_i64(1, work);  // empty product -> 1
                if (!acc) {
                    return acc;
                }
                for (const Expr& factor : n.factors) {
                    auto value = eval_node(factor, work);
                    if (!value) {
                        return value;
                    }
                    auto prod = acc->multiply(*value, work);
                    if (!prod) {
                        return prod;
                    }
                    acc = prod;
                }
                return acc;
            } else if constexpr (std::is_same_v<T, PowerNode>) {
                const auto exp = integer_exponent(n.exponent);
                if (!exp) {
                    return make_error<BigFloat>(MathError::not_implemented);
                }
                auto base = eval_node(n.base, work);
                if (!base) {
                    return base;
                }
                return integer_power(*base, *exp, work);
            } else if constexpr (std::is_same_v<T, FunctionNode>) {
                // Numeric evaluation of function applications (sin, exp, ...) belongs to a
                // later numeric-evaluation layer, not to this constant bridge.
                return make_error<BigFloat>(MathError::not_implemented);
            } else {
                static_assert(always_false<T>, "eval_node: unhandled ExprNode kind");
            }
        },
        u.node().value);
}

}  // namespace

// --- factories -----------------------------------------------------------------------

auto pi() -> Expr { return Expr::symbol(std::string(name_pi)); }
auto e() -> Expr { return Expr::symbol(std::string(name_e)); }
auto euler_gamma() -> Expr { return Expr::symbol(std::string(name_gamma)); }
auto golden_ratio() -> Expr { return Expr::symbol(std::string(name_phi)); }

// --- recognition ---------------------------------------------------------------------

auto is_named_constant(const Expr& u) -> bool { return named_constant_name(u).has_value(); }

auto named_constant_name(const Expr& u) -> std::optional<std::string_view> {
    auto s = as<SymbolNode>(u.node().value);
    if (!s) {
        return std::nullopt;
    }
    const std::string_view name = (*s)->name;
    if (name == name_pi) {
        return name_pi;
    }
    if (name == name_e) {
        return name_e;
    }
    if (name == name_gamma) {
        return name_gamma;
    }
    if (name == name_phi) {
        return name_phi;
    }
    return std::nullopt;
}

// --- numeric bridge ------------------------------------------------------------------

auto evaluate_constant(const Expr& u, std::int64_t prec) -> Result<BigFloat> {
    if (prec <= 0) {
        return make_error<BigFloat>(MathError::domain_error);
    }
    const std::int64_t work = prec + guard_bits;  // elevated working precision
    auto value = eval_node(u, work);
    if (!value) {
        return value;
    }
    return value->with_precision(prec);  // round once to the requested precision
}

auto evaluate_constant_double(const Expr& u) -> Result<double> {
    // 96 bits (~29 decimal digits) comfortably exceeds a double's 53-bit mantissa.
    auto value = evaluate_constant(u, 96);
    if (!value) {
        return make_error<double>(value.error());
    }
    return value->to_double();
}

}  // namespace nimblecas
