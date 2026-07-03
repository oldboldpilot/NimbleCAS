// NimbleCAS vector calculus: gradient, divergence, curl, Laplacian, Jacobian,
// Hessian, and directional / total derivatives (ROADMAP 7.19).
// @author Olumuyiwa Oluwasanmi
//
// These operators are thin, exact compositions over the symbolic differentiation engine
// (nimblecas.diff): a partial derivative is differentiate(f, x_i) with the other symbols
// held fixed, and the vector operators assemble partials into gradients, matrices, and the
// classical field operators. Every result is passed through automatic simplification so
// that, for concrete fields, the calculus identities collapse exactly — e.g. the mixed
// partials of a gradient cancel by Clairaut's theorem, giving curl(grad f) = 0 and
// div(curl F) = 0. Dimension mismatches are reported as MathError::domain_error and any
// differentiation/simplification failure propagates (Result / MathError, Rule 32).

export module nimblecas.vectorcalc;

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.diff;
import nimblecas.simplify;

export namespace nimblecas {

using ExprVector = std::vector<Expr>;
using ExprMatrix = std::vector<std::vector<Expr>>;

// Gradient nabla f = (df/dx_1, ..., df/dx_n), one entry per variable.
[[nodiscard]] auto gradient(const Expr& f, const std::vector<std::string>& vars)
    -> Result<ExprVector>;

// Divergence nabla . F = sum_i dF_i/dx_i. Requires field.size() == vars.size().
[[nodiscard]] auto divergence(const ExprVector& field, const std::vector<std::string>& vars)
    -> Result<Expr>;

// Curl nabla x F in three dimensions. Requires field.size() == vars.size() == 3.
[[nodiscard]] auto curl(const ExprVector& field, const std::vector<std::string>& vars)
    -> Result<ExprVector>;

// Laplacian nabla^2 f = sum_i d^2f/dx_i^2.
[[nodiscard]] auto laplacian(const Expr& f, const std::vector<std::string>& vars)
    -> Result<Expr>;

// Jacobian matrix J_{ij} = dF_i/dx_j (field.size() rows, vars.size() columns).
[[nodiscard]] auto jacobian(const ExprVector& field, const std::vector<std::string>& vars)
    -> Result<ExprMatrix>;

// Hessian matrix H_{ij} = d^2f/(dx_i dx_j), symmetric by Clairaut's theorem.
[[nodiscard]] auto hessian(const Expr& f, const std::vector<std::string>& vars)
    -> Result<ExprMatrix>;

// Directional derivative nabla_u f = grad f . u. Requires direction.size() == vars.size();
// the direction is used as given (not normalised).
[[nodiscard]] auto directional_derivative(const Expr& f, const std::vector<std::string>& vars,
                                          const ExprVector& direction) -> Result<Expr>;

// Total derivative df/dt = df/dt|_explicit + sum_i (df/dx_i)(dx_i/dt) via the
// multivariable chain rule. dep_vars are the x_i(t) and dep_derivs[i] is dx_i/dt.
// Requires dep_vars.size() == dep_derivs.size().
[[nodiscard]] auto total_derivative(const Expr& f, std::string_view indep,
                                    const std::vector<std::string>& dep_vars,
                                    const ExprVector& dep_derivs) -> Result<Expr>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// A single partial derivative df/dx.
[[nodiscard]] auto partial(const Expr& f, std::string_view var) -> Result<Expr> {
    return differentiate(f, var);
}

// Second partial d^2f/(dx_i dx_j).
[[nodiscard]] auto second_partial(const Expr& f, std::string_view vi, std::string_view vj)
    -> Result<Expr> {
    auto first = differentiate(f, vi);
    if (!first) {
        return first;
    }
    return differentiate(*first, vj);
}

// a - b as a symbolic expression.
[[nodiscard]] auto subtract(const Expr& a, const Expr& b) -> Expr {
    return Expr::sum({a, Expr::product({Expr::integer(-1), b})});
}

}  // namespace

auto gradient(const Expr& f, const std::vector<std::string>& vars) -> Result<ExprVector> {
    ExprVector grad;
    grad.reserve(vars.size());
    for (const std::string& var : vars) {
        auto d = partial(f, var);
        if (!d) {
            return make_error<ExprVector>(d.error());
        }
        auto s = simplify(*d);
        if (!s) {
            return make_error<ExprVector>(s.error());
        }
        grad.push_back(std::move(*s));
    }
    return grad;
}

auto divergence(const ExprVector& field, const std::vector<std::string>& vars) -> Result<Expr> {
    if (field.size() != vars.size()) {
        return make_error<Expr>(MathError::domain_error);
    }
    ExprVector terms;
    terms.reserve(field.size());
    for (std::size_t i = 0; i < field.size(); ++i) {
        auto d = partial(field[i], vars[i]);
        if (!d) {
            return d;
        }
        terms.push_back(std::move(*d));
    }
    return simplify(Expr::sum(std::move(terms)));
}

auto curl(const ExprVector& field, const std::vector<std::string>& vars) -> Result<ExprVector> {
    if (field.size() != 3 || vars.size() != 3) {
        return make_error<ExprVector>(MathError::domain_error);  // curl is 3-dimensional
    }
    // component k = d(F_{k+2})/d(x_{k+1}) - d(F_{k+1})/d(x_{k+2})  (indices mod 3)
    ExprVector out;
    out.reserve(3);
    for (std::size_t k = 0; k < 3; ++k) {
        const std::size_t a = (k + 1) % 3;
        const std::size_t b = (k + 2) % 3;
        auto da = partial(field[b], vars[a]);
        if (!da) {
            return make_error<ExprVector>(da.error());
        }
        auto db = partial(field[a], vars[b]);
        if (!db) {
            return make_error<ExprVector>(db.error());
        }
        auto comp = simplify(subtract(*da, *db));
        if (!comp) {
            return make_error<ExprVector>(comp.error());
        }
        out.push_back(std::move(*comp));
    }
    return out;
}

auto laplacian(const Expr& f, const std::vector<std::string>& vars) -> Result<Expr> {
    ExprVector terms;
    terms.reserve(vars.size());
    for (const std::string& var : vars) {
        auto d2 = second_partial(f, var, var);
        if (!d2) {
            return d2;
        }
        terms.push_back(std::move(*d2));
    }
    return simplify(Expr::sum(std::move(terms)));
}

auto jacobian(const ExprVector& field, const std::vector<std::string>& vars) -> Result<ExprMatrix> {
    ExprMatrix rows;
    rows.reserve(field.size());
    for (const Expr& fi : field) {
        auto row = gradient(fi, vars);  // row i is grad F_i
        if (!row) {
            return make_error<ExprMatrix>(row.error());
        }
        rows.push_back(std::move(*row));
    }
    return rows;
}

auto hessian(const Expr& f, const std::vector<std::string>& vars) -> Result<ExprMatrix> {
    ExprMatrix rows;
    rows.reserve(vars.size());
    for (const std::string& vi : vars) {
        ExprVector row;
        row.reserve(vars.size());
        for (const std::string& vj : vars) {
            auto d2 = second_partial(f, vi, vj);
            if (!d2) {
                return make_error<ExprMatrix>(d2.error());
            }
            auto s = simplify(*d2);
            if (!s) {
                return make_error<ExprMatrix>(s.error());
            }
            row.push_back(std::move(*s));
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

auto directional_derivative(const Expr& f, const std::vector<std::string>& vars,
                            const ExprVector& direction) -> Result<Expr> {
    if (direction.size() != vars.size()) {
        return make_error<Expr>(MathError::domain_error);
    }
    ExprVector terms;
    terms.reserve(vars.size());
    for (std::size_t i = 0; i < vars.size(); ++i) {
        auto d = partial(f, vars[i]);
        if (!d) {
            return d;
        }
        terms.push_back(Expr::product({*d, direction[i]}));
    }
    return simplify(Expr::sum(std::move(terms)));
}

auto total_derivative(const Expr& f, std::string_view indep,
                      const std::vector<std::string>& dep_vars, const ExprVector& dep_derivs)
    -> Result<Expr> {
    if (dep_vars.size() != dep_derivs.size()) {
        return make_error<Expr>(MathError::domain_error);
    }
    ExprVector terms;
    terms.reserve(dep_vars.size() + 1);
    // Explicit dependence on the independent variable.
    auto explicit_part = partial(f, indep);
    if (!explicit_part) {
        return explicit_part;
    }
    terms.push_back(std::move(*explicit_part));
    // Chain-rule contributions through each dependent variable.
    for (std::size_t i = 0; i < dep_vars.size(); ++i) {
        auto d = partial(f, dep_vars[i]);
        if (!d) {
            return d;
        }
        terms.push_back(Expr::product({*d, dep_derivs[i]}));
    }
    return simplify(Expr::sum(std::move(terms)));
}

}  // namespace nimblecas
