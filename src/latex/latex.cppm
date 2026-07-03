// NimbleCAS LaTeX math exporter (ROADMAP 7.12).
// @author Olumuyiwa Oluwasanmi
//
// to_latex(u) walks an immutable symbolic Expr recursively and renders it as a
// LaTeX math string (no surrounding $ delimiters). Parenthesisation is driven by a
// four-level precedence lattice (sum < product < power < atom): a sub-expression is
// wrapped in \left( ... \right) exactly when its outermost operator binds more
// loosely than the position it sits in. The walk is total (Rule 32 — no exceptions):
// every ExprNode alternative is handled and a non-exhaustive visitor fails to compile.

export module nimblecas.latex;

import std;
import nimblecas.core;
import nimblecas.symbolic;

export namespace nimblecas {

// Render an expression as a LaTeX math string (without $...$ delimiters).
[[nodiscard]] auto to_latex(const Expr& u) -> std::string;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

template <typename...>
inline constexpr bool always_false = false;

// Precedence lattice: a fragment is wrapped in parentheses when its precedence is
// strictly below the minimum required by its position. Atoms (symbols, numbers,
// function applications, \frac{}{} and \sqrt{} — all self-delimiting) never wrap.
inline constexpr int prec_sum = 1;      // a + b
inline constexpr int prec_product = 2;  // a b
inline constexpr int prec_power = 3;    // a^{b}
inline constexpr int prec_atom = 4;     // x, 2, \sin(x), \frac{1}{2}

// A rendered fragment: its LaTeX text, the precedence of its outermost operator, and
// whether `text` begins with a peelable leading unary minus. When `neg` is true the
// magnitude (text without the sign) is text.substr(1); AddNode uses this to turn a
// negative term into a " - " join, and constants/products advertise it so a leading
// minus forces sum-level wrapping in tighter contexts.
struct Frag {
    std::string text;
    int prec{prec_atom};
    bool neg{false};
};

// Forward declaration: the renderer is mutually recursive with its per-node helpers.
[[nodiscard]] auto render(const Expr& u) -> Frag;

// Wrap in \left( ... \right) iff the fragment binds more loosely than `min_prec`.
[[nodiscard]] auto wrap(const Frag& f, int min_prec) -> std::string {
    if (f.prec < min_prec) {
        return "\\left(" + f.text + "\\right)";
    }
    return f.text;
}

// The magnitude of a fragment: its text with any leading unary minus removed.
[[nodiscard]] auto magnitude(const Frag& f) -> std::string {
    return f.neg ? f.text.substr(1) : f.text;
}

// Lowercase symbol names that map onto a LaTeX Greek-letter command of the same
// spelling (\alpha, \pi, ...), plus the capitals that have their own command. Any
// other symbol renders verbatim.
[[nodiscard]] auto greek_or_name(const std::string& name) -> std::string {
    static constexpr std::string_view known[] = {
        "alpha", "beta",  "gamma", "delta",   "epsilon", "zeta",  "eta",
        "theta", "iota",  "kappa", "lambda",  "mu",      "nu",    "xi",
        "pi",    "rho",   "sigma", "tau",     "upsilon", "phi",   "chi",
        "psi",   "omega", "Gamma", "Delta",   "Theta",   "Lambda", "Xi",
        "Pi",    "Sigma", "Phi",   "Psi",     "Omega",
    };
    for (std::string_view g : known) {
        if (name == g) {
            return "\\" + std::string(g);
        }
    }
    return name;
}

// LaTeX command for a function name: a known elementary function maps to its control
// word (\sin, \exp, \log, ...); anything else uses \operatorname{name} so it still
// typesets upright and correctly spaced.
[[nodiscard]] auto function_command(const std::string& name) -> std::string {
    static constexpr std::string_view known[] = {
        "sin",  "cos",  "tan",  "cot",    "sec",    "csc",
        "sinh", "cosh", "tanh", "exp",    "log",    "ln",
        "arcsin", "arccos", "arctan", "det", "gcd", "max", "min",
    };
    for (std::string_view k : known) {
        if (name == k) {
            return "\\" + std::string(k);
        }
    }
    return "\\operatorname{" + name + "}";
}

// A rational (p, q) constant with q != 1 renders as \frac{|p|}{q} with the sign kept
// outside; q == 1 collapses to the integer p.
[[nodiscard]] auto render_rational(std::int64_t p, std::int64_t q) -> Frag {
    if (q == 1) {  // canonicalised integer stored as a pair
        if (p < 0) {
            return {std::format("{}", p), prec_sum, true};
        }
        return {std::format("{}", p), prec_atom, false};
    }
    const bool neg = p < 0;
    const std::int64_t ap = neg ? -p : p;  // rational() rejects INT64_MIN, so -p is safe
    const std::string frac = std::format("\\frac{{{}}}{{{}}}", ap, q);
    if (neg) {
        return {"-" + frac, prec_sum, true};
    }
    return {frac, prec_atom, false};
}

[[nodiscard]] auto render_constant(const ConstantNode& c) -> Frag {
    return std::visit(
        []<typename V>(const V& v) -> Frag {
            if constexpr (std::is_same_v<V, std::int64_t>) {
                // std::format handles INT64_MIN; a leading '-' is peelable as the magnitude.
                if (v < 0) {
                    return {std::format("{}", v), prec_sum, true};
                }
                return {std::format("{}", v), prec_atom, false};
            } else if constexpr (std::is_same_v<V, double>) {
                if (v < 0.0) {
                    return {std::format("{}", v), prec_sum, true};
                }
                return {std::format("{}", v), prec_atom, false};
            } else {  // std::pair<std::int64_t, std::int64_t>
                return render_rational(v.first, v.second);
            }
        },
        c.value);
}

// If `f` is base^{n} with a negative integer n, return (base, |n|); else nullopt.
// Such a factor belongs in a product's denominator.
[[nodiscard]] auto negative_integer_power(const Expr& f)
    -> std::optional<std::pair<Expr, std::int64_t>> {
    auto pw = as<PowerNode>(f.node().value);
    if (!pw) {
        return std::nullopt;
    }
    auto ec = as<ConstantNode>((*pw)->exponent.node().value);
    if (!ec) {
        return std::nullopt;
    }
    if (std::holds_alternative<std::int64_t>((*ec)->value)) {
        const std::int64_t n = std::get<std::int64_t>((*ec)->value);
        constexpr std::int64_t int64_min = std::numeric_limits<std::int64_t>::min();
        if (n < 0 && n != int64_min) {  // -n would overflow for INT64_MIN
            return std::pair<Expr, std::int64_t>{(*pw)->base, -n};
        }
    }
    return std::nullopt;
}

// Concatenate numerator parts by juxtaposition, inserting \cdot only between two
// purely-numeric parts so e.g. two integers do not fuse into one number.
[[nodiscard]] auto join_numerator(const std::vector<std::pair<std::string, bool>>& parts)
    -> std::string {
    std::string out;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i != 0) {
            const bool both_numeric = parts[i - 1].second && parts[i].second;
            out += both_numeric ? " \\cdot " : " ";
        }
        out += parts[i].first;
    }
    return out;
}

// Product rendering. Numeric factors fold into an overall sign plus rational
// numerator/denominator; factors of the form base^{-n} move to the denominator; the
// remainder are juxtaposed. A non-empty denominator yields a \frac{...}{...}. Sum
// factors are parenthesised (they bind more loosely than a product).
[[nodiscard]] auto render_product(const std::vector<Expr>& factors) -> Frag {
    int sign = 1;
    std::vector<std::pair<std::string, bool>> num;  // (latex, is_numeric)
    std::vector<std::string> den;
    constexpr std::int64_t int64_min = std::numeric_limits<std::int64_t>::min();

    for (const Expr& f : factors) {
        if (auto cc = as<ConstantNode>(f.node().value)) {
            const auto& val = (*cc)->value;
            if (std::holds_alternative<std::int64_t>(val)) {
                std::int64_t n = std::get<std::int64_t>(val);
                if (n < 0 && n != int64_min) {
                    sign = -sign;
                    n = -n;
                }
                if (n != 1) {  // a coefficient of 1 contributes nothing
                    num.emplace_back(std::format("{}", n), true);
                }
                continue;
            }
            if (std::holds_alternative<std::pair<std::int64_t, std::int64_t>>(val)) {
                auto [p, q] = std::get<std::pair<std::int64_t, std::int64_t>>(val);
                if (p < 0) {  // rational() rejects INT64_MIN numerators
                    sign = -sign;
                    p = -p;
                }
                if (p != 1) {
                    num.emplace_back(std::format("{}", p), true);
                }
                if (q != 1) {
                    den.emplace_back(std::format("{}", q));
                }
                continue;
            }
            // double
            double d = std::get<double>(val);
            if (d < 0.0) {
                sign = -sign;
                d = -d;
            }
            num.emplace_back(std::format("{}", d), true);
            continue;
        }
        if (auto np = negative_integer_power(f)) {
            const auto& [base, mag] = *np;
            if (mag == 1) {
                // \frac groups the whole denominator; wrap only a loose (sum) base so
                // it stays unambiguous alongside further denominator factors.
                den.emplace_back(wrap(render(base), prec_product));
            } else {
                den.emplace_back(wrap(render(base), prec_atom) +
                                 std::format("^{{{}}}", mag));
            }
            continue;
        }
        num.emplace_back(wrap(render(f), prec_product), false);
    }

    std::string num_str = join_numerator(num);
    if (num_str.empty()) {
        num_str = "1";
    }

    std::string body;
    int prec = prec_product;
    if (den.empty()) {
        body = num_str;
    } else {
        std::string den_str;
        for (std::size_t i = 0; i < den.size(); ++i) {
            if (i != 0) {
                den_str += " ";
            }
            den_str += den[i];
        }
        body = "\\frac{" + num_str + "}{" + den_str + "}";
        prec = prec_atom;  // \frac is self-delimiting
    }

    const bool neg = sign < 0;
    if (neg) {
        return {"-" + body, prec_sum, true};
    }
    return {body, prec, false};
}

// Power rendering with the usual typographic special cases:
//   exponent 1/2         -> \sqrt{base}
//   exponent -1          -> \frac{1}{base}
//   exponent -n (n > 1)  -> \frac{1}{base^{n}}
//   otherwise            -> base^{exp}, base parenthesised unless it is an atom.
[[nodiscard]] auto render_power(const Expr& base, const Expr& exponent) -> Frag {
    constexpr std::int64_t int64_min = std::numeric_limits<std::int64_t>::min();
    if (auto ec = as<ConstantNode>(exponent.node().value)) {
        const auto& val = (*ec)->value;
        if (std::holds_alternative<std::pair<std::int64_t, std::int64_t>>(val)) {
            auto [p, q] = std::get<std::pair<std::int64_t, std::int64_t>>(val);
            if (p == 1 && q == 2) {
                return {"\\sqrt{" + render(base).text + "}", prec_atom, false};
            }
        } else if (std::holds_alternative<std::int64_t>(val)) {
            const std::int64_t n = std::get<std::int64_t>(val);
            if (n < 0 && n != int64_min) {
                const std::int64_t mag = -n;
                std::string denom;
                if (mag == 1) {  // \frac groups the base; no exponent means no wrap
                    denom = render(base).text;
                } else {
                    denom = wrap(render(base), prec_atom) + std::format("^{{{}}}", mag);
                }
                return {"\\frac{1}{" + denom + "}", prec_atom, false};
            }
        }
    }
    const std::string base_str = wrap(render(base), prec_atom);
    const std::string exp_str = render(exponent).text;  // braces delimit the exponent
    return {base_str + "^{" + exp_str + "}", prec_power, false};
}

[[nodiscard]] auto render_function(const FunctionNode& fn) -> Frag {
    if (fn.name == "sqrt" && fn.args.size() == 1) {
        return {"\\sqrt{" + render(fn.args[0]).text + "}", prec_atom, false};
    }
    std::string args;
    for (std::size_t i = 0; i < fn.args.size(); ++i) {
        if (i != 0) {
            args += ", ";
        }
        args += render(fn.args[i]).text;  // \left(...\right) delimits the argument list
    }
    return {function_command(fn.name) + "\\left(" + args + "\\right)", prec_atom, false};
}

// Sum rendering: terms joined by " + ", a negative term folded into " - " with its
// magnitude, and a leading negative term keeping its sign (e.g. "-x + y").
[[nodiscard]] auto render_sum(const std::vector<Expr>& terms) -> Frag {
    if (terms.empty()) {
        return {"0", prec_atom, false};
    }
    std::string out;
    for (std::size_t i = 0; i < terms.size(); ++i) {
        const Frag f = render(terms[i]);
        const std::string mag = magnitude(f);
        if (i == 0) {
            out += f.neg ? ("-" + mag) : mag;
        } else {
            out += f.neg ? (" - " + mag) : (" + " + mag);
        }
    }
    return {out, prec_sum, false};
}

auto render(const Expr& u) -> Frag {
    return std::visit(
        []<typename T>(const T& n) -> Frag {
            if constexpr (std::is_same_v<T, SymbolNode>) {
                return {greek_or_name(n.name), prec_atom, false};
            } else if constexpr (std::is_same_v<T, ConstantNode>) {
                return render_constant(n);
            } else if constexpr (std::is_same_v<T, AddNode>) {
                return render_sum(n.terms);
            } else if constexpr (std::is_same_v<T, MulNode>) {
                return render_product(n.factors);
            } else if constexpr (std::is_same_v<T, PowerNode>) {
                return render_power(n.base, n.exponent);
            } else if constexpr (std::is_same_v<T, FunctionNode>) {
                return render_function(n);
            } else {
                static_assert(always_false<T>, "render: unhandled ExprNode kind");
            }
        },
        u.node().value);
}

}  // namespace

auto to_latex(const Expr& u) -> std::string { return render(u).text; }

}  // namespace nimblecas
