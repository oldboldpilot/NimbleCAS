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
import nimblecas.parallel;

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
    // Exact rational. Fails with MathError::division_by_zero on a zero denominator
    // (Rule 32); the result is canonicalised (sign on the numerator, reduced by gcd)
    // so structurally-equal fractions like 2/4 and 1/2 compare equivalent.
    [[nodiscard]] static auto rational(std::int64_t numerator, std::int64_t denominator)
        -> Result<Expr>;

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

    // Number of nodes in this subtree, memoised at construction (O(1) query). Drives
    // the cost gate for parallel tree recursion (large subtree -> fork children).
    [[nodiscard]] auto size() const noexcept -> std::size_t { return size_; }

private:
    explicit Expr(ExprNode value);
    CowPtr<ExprNode> node_;
    std::size_t size_{1};
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

// Structural hash consistent with is_equivalent_to: a == b implies
// hash_value(a) == hash_value(b). Used for the Python __hash__ and, later, for
// hash-consing (ROADMAP 6.2).
[[nodiscard]] auto hash_value(const Expr& u) -> std::size_t;

}  // namespace nimblecas

// ===========================================================================
// Implementation (defined once ExprNode is a complete type; not re-exported).
// ===========================================================================
namespace nimblecas {

namespace {

// Dependent-false helper so a non-exhaustive std::visit chain fails to compile
// (rather than silently running off the end of a non-void lambda → UB) if a new
// ExprNode alternative is added without updating every visitor.
template <typename...>
inline constexpr bool always_false = false;

[[nodiscard]] auto vectors_equivalent(const std::vector<Expr>& a, const std::vector<Expr>& b)
    -> bool {
    return a.size() == b.size() &&
           std::ranges::equal(a, b, [](const Expr& x, const Expr& y) {
               return x.is_equivalent_to(y);
           });
}

// Structural (syntactic) equality of constants: doubles are compared bitwise so a
// NaN leaf is equivalent to itself and +0.0/-0.0 stay distinct, keeping equality
// syntactic rather than delegating to IEEE float semantics.
[[nodiscard]] auto constant_equal(const ConstantNode& a, const ConstantNode& b) -> bool {
    if (a.value.index() != b.value.index()) {
        return false;
    }
    return std::visit(
        [&b]<typename V>(const V& x) -> bool {
            const V& y = std::get<V>(b.value);
            if constexpr (std::is_same_v<V, double>) {
                return std::bit_cast<std::uint64_t>(x) == std::bit_cast<std::uint64_t>(y);
            } else {
                return x == y;  // std::int64_t or std::pair<int64,int64>
            }
        },
        a.value);
}

[[nodiscard]] auto hash_combine(std::size_t seed, std::size_t value) -> std::size_t {
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

// Hash of a constant, consistent with constant_equal (doubles hashed bitwise).
[[nodiscard]] auto hash_constant(const ConstantNode& c) -> std::size_t {
    const std::size_t kind = std::hash<std::size_t>{}(c.value.index());
    const std::size_t payload = std::visit(
        []<typename V>(const V& v) -> std::size_t {
            if constexpr (std::is_same_v<V, double>) {
                return std::hash<std::uint64_t>{}(std::bit_cast<std::uint64_t>(v));
            } else if constexpr (std::is_same_v<V, std::int64_t>) {
                return std::hash<std::int64_t>{}(v);
            } else {  // std::pair<int64,int64>
                return hash_combine(std::hash<std::int64_t>{}(v.first),
                                    std::hash<std::int64_t>{}(v.second));
            }
        },
        c.value);
    return hash_combine(kind, payload);
}

[[nodiscard]] auto join(const std::vector<Expr>& items, std::string_view separator)
    -> std::string;  // forward decl; needs Expr::to_string

}  // namespace

namespace {

// Subtree node count = 1 + sum of children's (already-memoised) sizes.
[[nodiscard]] auto compute_size(const ExprNode& node) -> std::size_t {
    return std::visit(
        []<typename T>(const T& n) -> std::size_t {
            if constexpr (std::is_same_v<T, SymbolNode> || std::is_same_v<T, ConstantNode>) {
                return 1;
            } else if constexpr (std::is_same_v<T, AddNode>) {
                std::size_t s = 1;
                for (const Expr& e : n.terms) s += e.size();
                return s;
            } else if constexpr (std::is_same_v<T, MulNode>) {
                std::size_t s = 1;
                for (const Expr& e : n.factors) s += e.size();
                return s;
            } else if constexpr (std::is_same_v<T, PowerNode>) {
                return 1 + n.base.size() + n.exponent.size();
            } else if constexpr (std::is_same_v<T, FunctionNode>) {
                std::size_t s = 1;
                for (const Expr& e : n.args) s += e.size();
                return s;
            } else {
                static_assert(always_false<T>, "compute_size: unhandled ExprNode kind");
            }
        },
        node.value);
}

}  // namespace

Expr::Expr(ExprNode value) : node_(CowPtr<ExprNode>::make(std::move(value))) {
    size_ = compute_size(node_.read());
}

auto Expr::symbol(std::string name) -> Expr {
    return Expr(ExprNode{.value = SymbolNode{.name = std::move(name)}});
}

auto Expr::integer(std::int64_t value) -> Expr {
    return Expr(ExprNode{.value = ConstantNode{.value = value}});
}

auto Expr::real(double value) -> Expr {
    return Expr(ExprNode{.value = ConstantNode{.value = value}});
}

auto Expr::rational(std::int64_t numerator, std::int64_t denominator) -> Result<Expr> {
    if (denominator == 0) {
        return make_error<Expr>(MathError::division_by_zero);
    }
    // Negating INT64_MIN is signed-overflow UB, and std::gcd takes absolute values
    // so INT64_MIN would overflow inside it too. Reject both up front (Rule 32).
    constexpr std::int64_t int64_min = std::numeric_limits<std::int64_t>::min();
    if (numerator == int64_min || denominator == int64_min) {
        return make_error<Expr>(MathError::overflow);
    }
    if (denominator < 0) {  // keep the sign on the numerator
        numerator = -numerator;
        denominator = -denominator;
    }
    const std::int64_t divisor = std::gcd(numerator, denominator);
    if (divisor > 1) {
        numerator /= divisor;
        denominator /= divisor;
    }
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
                return constant_equal(lhs, rhs);
            } else if constexpr (std::is_same_v<T, AddNode>) {
                return vectors_equivalent(lhs.terms, rhs.terms);
            } else if constexpr (std::is_same_v<T, MulNode>) {
                return vectors_equivalent(lhs.factors, rhs.factors);
            } else if constexpr (std::is_same_v<T, PowerNode>) {
                return lhs.base.is_equivalent_to(rhs.base) &&
                       lhs.exponent.is_equivalent_to(rhs.exponent);
            } else if constexpr (std::is_same_v<T, FunctionNode>) {
                return lhs.name == rhs.name && vectors_equivalent(lhs.args, rhs.args);
            } else {
                static_assert(always_false<T>, "is_equivalent_to: unhandled ExprNode kind");
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
            } else {
                static_assert(always_false<T>, "to_string: unhandled ExprNode kind");
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
            } else {
                static_assert(always_false<T>, "free_of: unhandled ExprNode kind");
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
                // Children are independent -> cost-gated parallel map (immutable tree).
                const bool par = u.size() >= parallel::parallel_cost_threshold;
                auto out = parallel::transform_index_if(
                    par, n.terms.size(),
                    [&](std::size_t i) { return substitute(n.terms[i], t, r); });
                return Expr::sum(std::move(out));
            } else if constexpr (std::is_same_v<T, MulNode>) {
                const bool par = u.size() >= parallel::parallel_cost_threshold;
                auto out = parallel::transform_index_if(
                    par, n.factors.size(),
                    [&](std::size_t i) { return substitute(n.factors[i], t, r); });
                return Expr::product(std::move(out));
            } else if constexpr (std::is_same_v<T, PowerNode>) {
                return Expr::power(substitute(n.base, t, r), substitute(n.exponent, t, r));
            } else if constexpr (std::is_same_v<T, FunctionNode>) {
                const bool par = u.size() >= parallel::parallel_cost_threshold;
                auto out = parallel::transform_index_if(
                    par, n.args.size(),
                    [&](std::size_t i) { return substitute(n.args[i], t, r); });
                return Expr::apply(n.name, std::move(out));
            } else {
                static_assert(always_false<T>, "substitute: unhandled ExprNode kind");
            }
        },
        u.node().value);
}

auto hash_value(const Expr& u) -> std::size_t {
    const ExprNode& n = u.node();
    std::size_t seed = std::hash<std::size_t>{}(n.value.index());  // node kind
    std::visit(
        [&seed]<typename T>(const T& node) {
            if constexpr (std::is_same_v<T, SymbolNode>) {
                seed = hash_combine(seed, std::hash<std::string>{}(node.name));
            } else if constexpr (std::is_same_v<T, ConstantNode>) {
                seed = hash_combine(seed, hash_constant(node));
            } else if constexpr (std::is_same_v<T, AddNode>) {
                for (const Expr& e : node.terms) {
                    seed = hash_combine(seed, hash_value(e));
                }
            } else if constexpr (std::is_same_v<T, MulNode>) {
                for (const Expr& e : node.factors) {
                    seed = hash_combine(seed, hash_value(e));
                }
            } else if constexpr (std::is_same_v<T, PowerNode>) {
                seed = hash_combine(seed, hash_value(node.base));
                seed = hash_combine(seed, hash_value(node.exponent));
            } else if constexpr (std::is_same_v<T, FunctionNode>) {
                seed = hash_combine(seed, std::hash<std::string>{}(node.name));
                for (const Expr& e : node.args) {
                    seed = hash_combine(seed, hash_value(e));
                }
            } else {
                static_assert(always_false<T>, "hash_value: unhandled ExprNode kind");
            }
        },
        n.value);
    return seed;
}

}  // namespace nimblecas
