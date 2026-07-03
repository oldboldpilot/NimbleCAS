// NimbleCAS symbolic engine: expression trees and the foundational Cohen
// algorithms (structural equality, FreeOf, Substitute).
// @author Olumuyiwa Oluwasanmi
//
// Expressions are immutable trees of variant nodes shared via CowPtr (Rule 22), so
// copies are cheap and safe to read across threads. No inheritance hierarchy is
// used: a node is a type-safe union (std::variant) to keep structures small and
// cache-friendly (ROADMAP 2.2).

export module nimblecas.symbolic;

import std;
import nimblecas.core;

export namespace nimblecas {

// The variant is wrapped in a struct so that Expr can hold CowPtr<ExprNode> while
// ExprNode is still incomplete (a std::variant alias cannot be forward-declared).
struct ExprNode;

// ---------------------------------------------------------------------------
// Expr — a copy-on-write handle to an immutable expression node.
// ---------------------------------------------------------------------------
class Expr {
public:
    // --- leaf factories ---
    [[nodiscard]] static auto symbol(std::string name) -> Expr;
    [[nodiscard]] static auto integer(std::int64_t value) -> Expr;
    [[nodiscard]] static auto real(double value) -> Expr;
    [[nodiscard]] static auto rational(std::int64_t numerator, std::int64_t denominator) -> Expr;

    // --- compound factories ---
    [[nodiscard]] static auto sum(std::vector<Expr> terms) -> Expr;
    [[nodiscard]] static auto product(std::vector<Expr> factors) -> Expr;
    [[nodiscard]] static auto power(Expr base, Expr exponent) -> Expr;
    [[nodiscard]] static auto apply(std::string name, std::vector<Expr> args) -> Expr;

    // --- fluent builders (ROADMAP 8 fluent API) ---
    [[nodiscard]] auto add(const Expr& other) const -> Expr;
    [[nodiscard]] auto mul(const Expr& other) const -> Expr;
    [[nodiscard]] auto pow(const Expr& exponent) const -> Expr;

    [[nodiscard]] auto node() const -> const ExprNode& { return node_.read(); }

    // Structural (syntactic) equality — two expressions are equivalent when their
    // trees are identical. Cohen's automatic simplification is what later maps
    // mathematically-equal expressions to identical trees.
    [[nodiscard]] auto is_equivalent_to(const Expr& other) const -> bool;

    [[nodiscard]] auto to_string() const -> std::string;

private:
    explicit Expr(ExprNode value);
    CowPtr<ExprNode> node_;
};

[[nodiscard]] auto operator==(const Expr& lhs, const Expr& rhs) -> bool {
    return lhs.is_equivalent_to(rhs);
}

// ---------------------------------------------------------------------------
// Node kinds.
// ---------------------------------------------------------------------------
struct SymbolNode {
    std::string name;
};

struct ConstantNode {
    // Integer, IEEE double, or exact rational (numerator, denominator).
    std::variant<std::int64_t, double, std::pair<std::int64_t, std::int64_t>> value;
};

struct AddNode {
    std::vector<Expr> terms;
};

struct MulNode {
    std::vector<Expr> factors;
};

struct PowerNode {
    Expr base;
    Expr exponent;
};

struct FunctionNode {
    std::string name;
    std::vector<Expr> args;
};

struct ExprNode {
    std::variant<SymbolNode, ConstantNode, AddNode, MulNode, PowerNode, FunctionNode> value;
};

// ---------------------------------------------------------------------------
// Cohen algorithms.
// ---------------------------------------------------------------------------

// FreeOf(u, t): true iff expression u does not contain the sub-expression t.
[[nodiscard]] auto free_of(const Expr& u, const Expr& t) -> bool;

// Substitute(u, t, r): replace every occurrence of the sub-expression t in u with r.
[[nodiscard]] auto substitute(const Expr& u, const Expr& t, const Expr& r) -> Expr;

}  // namespace nimblecas

// ===========================================================================
// Implementation (defined once ExprNode is a complete type; not re-exported).
// ===========================================================================
namespace nimblecas {

namespace {

[[nodiscard]] auto vectors_equivalent(const std::vector<Expr>& a, const std::vector<Expr>& b)
    -> bool {
    return a.size() == b.size() &&
           std::ranges::equal(a, b, [](const Expr& x, const Expr& y) {
               return x.is_equivalent_to(y);
           });
}

[[nodiscard]] auto join(const std::vector<Expr>& items, std::string_view separator)
    -> std::string;  // forward decl; needs Expr::to_string

}  // namespace

Expr::Expr(ExprNode value) : node_(CowPtr<ExprNode>::make(std::move(value))) {}

auto Expr::symbol(std::string name) -> Expr {
    return Expr(ExprNode{.value = SymbolNode{.name = std::move(name)}});
}

auto Expr::integer(std::int64_t value) -> Expr {
    return Expr(ExprNode{.value = ConstantNode{.value = value}});
}

auto Expr::real(double value) -> Expr {
    return Expr(ExprNode{.value = ConstantNode{.value = value}});
}

auto Expr::rational(std::int64_t numerator, std::int64_t denominator) -> Expr {
    return Expr(ExprNode{
        .value = ConstantNode{.value = std::pair{numerator, denominator}}});
}

auto Expr::sum(std::vector<Expr> terms) -> Expr {
    return Expr(ExprNode{.value = AddNode{.terms = std::move(terms)}});
}

auto Expr::product(std::vector<Expr> factors) -> Expr {
    return Expr(ExprNode{.value = MulNode{.factors = std::move(factors)}});
}

auto Expr::power(Expr base, Expr exponent) -> Expr {
    return Expr(ExprNode{
        .value = PowerNode{.base = std::move(base), .exponent = std::move(exponent)}});
}

auto Expr::apply(std::string name, std::vector<Expr> args) -> Expr {
    return Expr(ExprNode{
        .value = FunctionNode{.name = std::move(name), .args = std::move(args)}});
}

auto Expr::add(const Expr& other) const -> Expr { return Expr::sum({*this, other}); }
auto Expr::mul(const Expr& other) const -> Expr { return Expr::product({*this, other}); }
auto Expr::pow(const Expr& exponent) const -> Expr { return Expr::power(*this, exponent); }

auto Expr::is_equivalent_to(const Expr& other) const -> bool {
    const ExprNode& a = node_.read();
    const ExprNode& b = other.node_.read();
    if (a.value.index() != b.value.index()) {
        return false;
    }
    return std::visit(
        [&b]<typename T>(const T& lhs) -> bool {
            const T& rhs = std::get<T>(b.value);
            if constexpr (std::is_same_v<T, SymbolNode>) {
                return lhs.name == rhs.name;
            } else if constexpr (std::is_same_v<T, ConstantNode>) {
                return lhs.value == rhs.value;
            } else if constexpr (std::is_same_v<T, AddNode>) {
                return vectors_equivalent(lhs.terms, rhs.terms);
            } else if constexpr (std::is_same_v<T, MulNode>) {
                return vectors_equivalent(lhs.factors, rhs.factors);
            } else if constexpr (std::is_same_v<T, PowerNode>) {
                return lhs.base.is_equivalent_to(rhs.base) &&
                       lhs.exponent.is_equivalent_to(rhs.exponent);
            } else if constexpr (std::is_same_v<T, FunctionNode>) {
                return lhs.name == rhs.name && vectors_equivalent(lhs.args, rhs.args);
            }
        },
        a.value);
}

auto Expr::to_string() const -> std::string {
    return std::visit(
        []<typename T>(const T& n) -> std::string {
            if constexpr (std::is_same_v<T, SymbolNode>) {
                return n.name;
            } else if constexpr (std::is_same_v<T, ConstantNode>) {
                return std::visit(
                    [](const auto& v) -> std::string {
                        using V = std::decay_t<decltype(v)>;
                        if constexpr (std::is_same_v<V, std::pair<std::int64_t, std::int64_t>>) {
                            return std::format("{}/{}", v.first, v.second);
                        } else {
                            return std::format("{}", v);
                        }
                    },
                    n.value);
            } else if constexpr (std::is_same_v<T, AddNode>) {
                return std::format("({})", join(n.terms, " + "));
            } else if constexpr (std::is_same_v<T, MulNode>) {
                return std::format("({})", join(n.factors, " * "));
            } else if constexpr (std::is_same_v<T, PowerNode>) {
                return std::format("{}^{}", n.base.to_string(), n.exponent.to_string());
            } else if constexpr (std::is_same_v<T, FunctionNode>) {
                return std::format("{}({})", n.name, join(n.args, ", "));
            }
        },
        node().value);
}

namespace {

auto join(const std::vector<Expr>& items, std::string_view separator) -> std::string {
    std::string out;
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (i != 0) {
            out += separator;
        }
        out += items[i].to_string();
    }
    return out;
}

}  // namespace

auto free_of(const Expr& u, const Expr& t) -> bool {
    if (u.is_equivalent_to(t)) {
        return false;
    }
    return std::visit(
        [&t]<typename T>(const T& n) -> bool {
            if constexpr (std::is_same_v<T, SymbolNode> || std::is_same_v<T, ConstantNode>) {
                return true;
            } else if constexpr (std::is_same_v<T, AddNode>) {
                return std::ranges::all_of(n.terms,
                                           [&t](const Expr& e) { return free_of(e, t); });
            } else if constexpr (std::is_same_v<T, MulNode>) {
                return std::ranges::all_of(n.factors,
                                           [&t](const Expr& e) { return free_of(e, t); });
            } else if constexpr (std::is_same_v<T, PowerNode>) {
                return free_of(n.base, t) && free_of(n.exponent, t);
            } else if constexpr (std::is_same_v<T, FunctionNode>) {
                return std::ranges::all_of(n.args,
                                           [&t](const Expr& e) { return free_of(e, t); });
            }
        },
        u.node().value);
}

auto substitute(const Expr& u, const Expr& t, const Expr& r) -> Expr {
    if (u.is_equivalent_to(t)) {
        return r;
    }
    return std::visit(
        [&](const auto& n) -> Expr {
            using T = std::decay_t<decltype(n)>;
            if constexpr (std::is_same_v<T, SymbolNode> || std::is_same_v<T, ConstantNode>) {
                return u;  // leaf that is not t itself: unchanged
            } else if constexpr (std::is_same_v<T, AddNode>) {
                std::vector<Expr> out;
                out.reserve(n.terms.size());
                for (const Expr& e : n.terms) {
                    out.push_back(substitute(e, t, r));
                }
                return Expr::sum(std::move(out));
            } else if constexpr (std::is_same_v<T, MulNode>) {
                std::vector<Expr> out;
                out.reserve(n.factors.size());
                for (const Expr& e : n.factors) {
                    out.push_back(substitute(e, t, r));
                }
                return Expr::product(std::move(out));
            } else if constexpr (std::is_same_v<T, PowerNode>) {
                return Expr::power(substitute(n.base, t, r), substitute(n.exponent, t, r));
            } else if constexpr (std::is_same_v<T, FunctionNode>) {
                std::vector<Expr> out;
                out.reserve(n.args.size());
                for (const Expr& e : n.args) {
                    out.push_back(substitute(e, t, r));
                }
                return Expr::apply(n.name, std::move(out));
            }
        },
        u.node().value);
}

}  // namespace nimblecas
