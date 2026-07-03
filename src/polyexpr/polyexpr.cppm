// NimbleCAS bridge between symbolic Expr and dense Polynomial (ROADMAP 7.20).
// @author Olumuyiwa Oluwasanmi
//
// to_polynomial extracts a univariate, integer-coefficient Polynomial in `var` from an
// expression, failing (MathError::not_implemented) when the expression is not such a
// polynomial (another symbol, a non-integer constant, a non-constant or negative
// exponent, or a function). from_polynomial rebuilds a sum-of-monomials Expr.

export module nimblecas.polyexpr;

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.polynomial;

export namespace nimblecas {

[[nodiscard]] auto to_polynomial(const Expr& u, std::string_view var) -> Result<Polynomial>;
[[nodiscard]] auto from_polynomial(const Polynomial& p, std::string_view var) -> Expr;

// GCD of two univariate polynomial expressions in `var`, returned as an Expr.
[[nodiscard]] auto polynomial_gcd(const Expr& a, const Expr& b, std::string_view var)
    -> Result<Expr>;

// Square-free factorization of a polynomial expression: (factor Expr, multiplicity).
[[nodiscard]] auto square_free_factor(const Expr& u, std::string_view var)
    -> Result<std::vector<std::pair<Expr, std::int64_t>>>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

template <typename...>
inline constexpr bool always_false = false;

// A non-negative integer value if `e` is exactly an integer constant, else nullopt.
[[nodiscard]] auto as_integer(const Expr& e) -> std::optional<std::int64_t> {
    if (const auto* c = std::get_if<ConstantNode>(&e.node().value)) {
        if (const auto* v = std::get_if<std::int64_t>(&c->value)) {
            return *v;
        }
    }
    return std::nullopt;
}

[[nodiscard]] auto power_via_repeated_multiply(const Polynomial& base, std::int64_t exp)
    -> Result<Polynomial> {
    Polynomial acc = Polynomial::constant(1);
    for (std::int64_t k = 0; k < exp; ++k) {
        auto next = acc.multiply(base);
        if (!next) {
            return next;
        }
        acc = *next;
    }
    return acc;
}

}  // namespace

auto to_polynomial(const Expr& u, std::string_view var) -> Result<Polynomial> {
    return std::visit(
        [&]<typename T>(const T& n) -> Result<Polynomial> {
            if constexpr (std::is_same_v<T, ConstantNode>) {
                if (const auto* v = std::get_if<std::int64_t>(&n.value)) {
                    return Polynomial::constant(*v);
                }
                return make_error<Polynomial>(MathError::not_implemented);  // non-integer
            } else if constexpr (std::is_same_v<T, SymbolNode>) {
                if (n.name == var) {
                    return Polynomial::monomial(1, 1);  // x
                }
                return make_error<Polynomial>(MathError::not_implemented);  // other variable
            } else if constexpr (std::is_same_v<T, AddNode>) {
                Polynomial acc;  // zero
                for (const Expr& term : n.terms) {
                    auto p = to_polynomial(term, var);
                    if (!p) {
                        return p;
                    }
                    auto sum = acc.add(*p);
                    if (!sum) {
                        return sum;
                    }
                    acc = *sum;
                }
                return acc;
            } else if constexpr (std::is_same_v<T, MulNode>) {
                Polynomial acc = Polynomial::constant(1);
                for (const Expr& factor : n.factors) {
                    auto p = to_polynomial(factor, var);
                    if (!p) {
                        return p;
                    }
                    auto prod = acc.multiply(*p);
                    if (!prod) {
                        return prod;
                    }
                    acc = *prod;
                }
                return acc;
            } else if constexpr (std::is_same_v<T, PowerNode>) {
                const auto exp = as_integer(n.exponent);
                if (!exp || *exp < 0) {
                    return make_error<Polynomial>(MathError::not_implemented);
                }
                auto base = to_polynomial(n.base, var);
                if (!base) {
                    return base;
                }
                return power_via_repeated_multiply(*base, *exp);
            } else if constexpr (std::is_same_v<T, FunctionNode>) {
                return make_error<Polynomial>(MathError::not_implemented);  // not a polynomial
            } else {
                static_assert(always_false<T>, "to_polynomial: unhandled ExprNode kind");
            }
        },
        u.node().value);
}

auto from_polynomial(const Polynomial& p, std::string_view var) -> Expr {
    if (p.is_zero()) {
        return Expr::integer(0);
    }
    const Expr x = Expr::symbol(std::string(var));
    const auto coeffs = p.coefficients();
    std::vector<Expr> terms;
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        if (coeffs[i] == 0) {
            continue;
        }
        const Expr c = Expr::integer(coeffs[i]);
        if (i == 0) {
            terms.push_back(c);
        } else if (i == 1) {
            terms.push_back(Expr::product({c, x}));  // c * x
        } else {
            terms.push_back(Expr::product(
                {c, Expr::power(x, Expr::integer(static_cast<std::int64_t>(i)))}));
        }
    }
    if (terms.size() == 1) {
        return terms.front();
    }
    return Expr::sum(std::move(terms));
}

auto polynomial_gcd(const Expr& a, const Expr& b, std::string_view var) -> Result<Expr> {
    auto pa = to_polynomial(a, var);
    if (!pa) {
        return make_error<Expr>(pa.error());
    }
    auto pb = to_polynomial(b, var);
    if (!pb) {
        return make_error<Expr>(pb.error());
    }
    auto g = pa->gcd(*pb);
    if (!g) {
        return make_error<Expr>(g.error());
    }
    return from_polynomial(*g, var);
}

auto square_free_factor(const Expr& u, std::string_view var)
    -> Result<std::vector<std::pair<Expr, std::int64_t>>> {
    using Factors = std::vector<std::pair<Expr, std::int64_t>>;
    auto p = to_polynomial(u, var);
    if (!p) {
        return make_error<Factors>(p.error());
    }
    auto factors = p->square_free_factorization();
    if (!factors) {
        return make_error<Factors>(factors.error());
    }
    Factors out;
    out.reserve(factors->size());
    for (const auto& [poly, multiplicity] : *factors) {
        out.emplace_back(from_polynomial(poly, var), multiplicity);
    }
    return out;
}

}  // namespace nimblecas
