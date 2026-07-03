// NimbleCAS Taylor series expansion (ROADMAP 7.3).
// @author Olumuyiwa Oluwasanmi
//
// taylor_coefficients(f, var, point, order) returns the Taylor coefficients
// c_0..c_order of f about `point`, where c_k = f^{(k)}(point) / k!. The k-th
// coefficient is obtained by repeatedly differentiating (reusing the diff engine),
// substituting the expansion point for the variable, and dividing by the running
// factorial k!; each coefficient is automatically simplified. k! is accumulated in an
// int64 with overflow detection (MathError::overflow) so a large order fails cleanly
// rather than wrapping (Rule 32). taylor_polynomial assembles the truncated series
// sum_{k=0}^{order} c_k (var - point)^k and simplifies it.

export module nimblecas.series;

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.diff;
import nimblecas.simplify;

export namespace nimblecas {

// Taylor coefficients c_0..c_order about `point` (c_k = f^{(k)}(point)/k!), each
// automatically simplified. A negative order is a domain_error; an order large enough
// to overflow int64 k! is an overflow error. Any error from differentiate/simplify/
// rational construction is propagated along the railway.
[[nodiscard]] auto taylor_coefficients(const Expr& f, std::string_view var, const Expr& point,
                                       std::int64_t order) -> Result<std::vector<Expr>>;

// Truncated Taylor polynomial sum_{k=0}^{order} c_k (var - point)^k, simplified.
[[nodiscard]] auto taylor_polynomial(const Expr& f, std::string_view var, const Expr& point,
                                     std::int64_t order) -> Result<Expr>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

auto taylor_coefficients(const Expr& f, std::string_view var, const Expr& point,
                         std::int64_t order) -> Result<std::vector<Expr>> {
    if (order < 0) {
        return make_error<std::vector<Expr>>(MathError::domain_error);
    }

    const Expr x = Expr::symbol(std::string(var));
    std::vector<Expr> coefficients;
    coefficients.reserve(static_cast<std::size_t>(order) + 1);

    Expr derivative = f;              // g_k, starts at g_0 = f
    std::int64_t factorial = 1;       // k! ; 0! = 1

    for (std::int64_t k = 0; k <= order; ++k) {
        // c_k = f^{(k)}(point) / k! = simplify( substitute(g_k, var, point) * (1/k!) ).
        auto reciprocal = Expr::rational(1, factorial);
        if (!reciprocal) {
            return make_error<std::vector<Expr>>(reciprocal.error());
        }
        const Expr at_point = substitute(derivative, x, point);
        auto coefficient = simplify(Expr::product({at_point, *reciprocal}));
        if (!coefficient) {
            return make_error<std::vector<Expr>>(coefficient.error());
        }
        coefficients.push_back(std::move(*coefficient));

        // Advance to g_{k+1} = d/dvar g_k and update the running factorial to (k+1)!.
        // Skipped after the final coefficient (nothing more to compute).
        if (k < order) {
            auto next_derivative = differentiate(derivative, var);
            if (!next_derivative) {
                return make_error<std::vector<Expr>>(next_derivative.error());
            }
            derivative = std::move(*next_derivative);

            std::int64_t next_factorial = 0;
            if (__builtin_mul_overflow(factorial, k + 1, &next_factorial)) {
                return make_error<std::vector<Expr>>(MathError::overflow);
            }
            factorial = next_factorial;
        }
    }

    return coefficients;
}

auto taylor_polynomial(const Expr& f, std::string_view var, const Expr& point,
                       std::int64_t order) -> Result<Expr> {
    auto coefficients = taylor_coefficients(f, var, point, order);
    if (!coefficients) {
        return make_error<Expr>(coefficients.error());
    }

    const Expr x = Expr::symbol(std::string(var));
    // (var - point), represented as var + (-1)*point so simplify can fold it (at
    // point 0 this collapses back to var).
    const Expr shifted = Expr::sum({x, Expr::product({Expr::integer(-1), point})});

    std::vector<Expr> terms;
    terms.reserve(coefficients->size());
    for (std::size_t k = 0; k < coefficients->size(); ++k) {
        // term_k = c_k * (var - point)^k.
        Expr power_term = Expr::power(shifted, Expr::integer(static_cast<std::int64_t>(k)));
        terms.push_back(Expr::product({(*coefficients)[k], std::move(power_term)}));
    }

    return simplify(Expr::sum(std::move(terms)));
}

}  // namespace nimblecas
