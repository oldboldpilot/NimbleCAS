// NimbleCAS Expr-level symbolic integration, int f dx (ROADMAP 7.19+).
// @author Olumuyiwa Oluwasanmi
//
// integrate(f, var) computes an indefinite antiderivative (no +C) of an expression
// with respect to a variable, over a deliberately small but *honest* class of
// integrands. Everything outside that class returns MathError::not_implemented — the
// engine never guesses an antiderivative it cannot justify (Rule 32). The guarantee is
// concrete: every antiderivative this module returns differentiates back to the
// integrand under nimblecas.diff, so a caller can always verify a result exactly.
//
// Supported integrand class:
//   * Linearity            int (c*g + h) dx = c*int g dx + int h dx; constant factors
//                          (free_of var) are pulled outside the integral.
//   * Power rule           int (a*x+b)^n dx = (a*x+b)^(n+1) / (a*(n+1)) for constant
//                          n != -1; the n == -1 case is (1/a) ln(a*x+b).
//   * Elementary table     exp/sin/cos/sinh/cosh with a LINEAR inner argument a*x+b,
//                          divided by a (linear u-substitution only).
//   * Inverse-trig forms   int 1/(x^2 + k^2) dx = (1/k) atan(x/k) and
//                          int 1/sqrt(k^2 - x^2) dx = asin(x/k) for integer k.
//   * Rational functions   a ratio of polynomials in var is bridged to the exact
//                          rational integrator (nimblecas.integrate) via
//                          nimblecas.polyexpr, yielding rational + logarithmic parts.
//
// The elementary-function spellings (exp, ln, sin, cos, sinh, cosh, atan, asin) are
// exactly those the nimblecas.diff derivative table inverts, so differentiate() is a
// left inverse of integrate() on the supported class.

export module nimblecas.symint;

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.diff;
import nimblecas.polynomial;
import nimblecas.polyexpr;
import nimblecas.ratpoly;
import nimblecas.rothstein;
import nimblecas.integrate;

export namespace nimblecas {

// Indefinite integral int f dx (no constant of integration). Returns
// MathError::not_implemented when f lies outside the supported integrand class above.
[[nodiscard]] auto integrate(const Expr& f, std::string_view var) -> Result<Expr>;

// Definite integral int_a^b f dx, evaluated as F(b) - F(a) for an antiderivative F.
// Propagates not_implemented from the indefinite step; assumes F is continuous on the
// interval (it does not detect a pole between a and b).
[[nodiscard]] auto integrate_definite(const Expr& f, std::string_view var, const Expr& a,
                                      const Expr& b) -> Result<Expr>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// --- small constructors -----------------------------------------------------

// e^(-1): a reciprocal, as a power node so nimblecas.diff/simplify handle it uniformly.
[[nodiscard]] auto recip(const Expr& e) -> Expr {
    return Expr::power(e, Expr::integer(-1));
}

// -1 * e.
[[nodiscard]] auto neg(const Expr& e) -> Expr {
    return Expr::product({Expr::integer(-1), e});
}

[[nodiscard]] auto is_zero_expr(const Expr& e) -> bool {
    return e.is_equivalent_to(Expr::integer(0));
}

// An exact rational (or integer) coefficient as an Expr. A canonical Rational never
// contains INT64_MIN, so Expr::rational cannot fail here; the guard keeps the helper
// total anyway (Rule 32).
[[nodiscard]] auto rational_to_expr(const Rational& r) -> Expr {
    if (r.is_integer()) {
        return Expr::integer(r.numerator());
    }
    auto e = Expr::rational(r.numerator(), r.denominator());
    return e ? *e : Expr::integer(0);
}

// --- affine (linear) argument detection ------------------------------------

// If g == a*x + b with a, b free of var and a != 0, return (a, b); else nullopt. The
// test is exact: g is affine in x iff dg/dx is free of x (a zero second derivative),
// in which case a = dg/dx and b = g - a*x. Using differentiate() keeps this in lock-step
// with the derivative table that must later invert the antiderivative.
[[nodiscard]] auto affine_form(const Expr& g, std::string_view var)
    -> std::optional<std::pair<Expr, Expr>> {
    const Expr x = Expr::symbol(std::string(var));
    auto da = differentiate(g, var);
    if (!da) {
        return std::nullopt;
    }
    const Expr& a = *da;  // differentiate() returns an already-simplified result
    if (!free_of(a, x)) {
        return std::nullopt;  // slope depends on x -> not linear
    }
    if (is_zero_expr(a)) {
        return std::nullopt;  // constant in x (degenerate; handled by the free_of path)
    }
    auto b = simplify(Expr::sum({g, neg(Expr::product({a, x}))}));
    if (!b || !free_of(*b, x)) {
        return std::nullopt;
    }
    return std::pair{a, *b};
}

// --- power rule (with linear substitution) ---------------------------------

// int (a*x+b)^exp dx for a constant exponent and an affine base. Declines
// (not_implemented) when the exponent is not constant or the base is not affine.
[[nodiscard]] auto integrate_affine_power(const Expr& base, const Expr& exp,
                                          std::string_view var) -> Result<Expr> {
    const Expr x = Expr::symbol(std::string(var));
    if (!free_of(exp, x)) {
        return make_error<Expr>(MathError::not_implemented);  // non-constant exponent
    }
    auto aff = affine_form(base, var);
    if (!aff) {
        return make_error<Expr>(MathError::not_implemented);
    }
    const auto& [a, b] = *aff;
    auto next_exp = simplify(Expr::sum({exp, Expr::integer(1)}));
    if (!next_exp) {
        return make_error<Expr>(next_exp.error());
    }
    if (is_zero_expr(*next_exp)) {
        // exp == -1: int (a*x+b)^-1 dx = (1/a) ln(a*x+b).
        return Expr::product({recip(a), Expr::apply("ln", {base})});
    }
    // int (a*x+b)^exp dx = (a*x+b)^(exp+1) / (a*(exp+1)).
    Expr denom = Expr::product({a, *next_exp});
    return Expr::product({recip(denom), Expr::power(base, *next_exp)});
}

// --- elementary function table (with linear substitution) ------------------

// int f(a*x+b) dx for a known unary f, = (1/a) * F(a*x+b) where F' == f. Only the
// functions whose antiderivative is itself a single table entry that differentiate()
// inverts exactly are included; every other function declines (not_implemented).
[[nodiscard]] auto integrate_function(const FunctionNode& fn, std::string_view var)
    -> Result<Expr> {
    if (fn.args.size() != 1) {
        return make_error<Expr>(MathError::not_implemented);
    }
    const Expr& inner = fn.args.front();
    auto aff = affine_form(inner, var);
    if (!aff) {
        return make_error<Expr>(MathError::not_implemented);
    }
    const auto& [a, b] = *aff;

    std::optional<Expr> antideriv;  // F(inner) with F' == fn
    if (fn.name == "exp") {
        antideriv = Expr::apply("exp", {inner});
    } else if (fn.name == "sin") {
        antideriv = neg(Expr::apply("cos", {inner}));
    } else if (fn.name == "cos") {
        antideriv = Expr::apply("sin", {inner});
    } else if (fn.name == "sinh") {
        antideriv = Expr::apply("cosh", {inner});
    } else if (fn.name == "cosh") {
        antideriv = Expr::apply("sinh", {inner});
    }
    if (!antideriv) {
        return make_error<Expr>(MathError::not_implemented);
    }
    return Expr::product({recip(a), *antideriv});
}

// --- inverse-trig quadratic table ------------------------------------------

// The positive integer square root of v, or nullopt when v is not a positive perfect
// square. Uses a double estimate plus a neighbour scan so it stays O(1) without an
// unbounded loop; the exact integer check is what decides the result.
[[nodiscard]] auto perfect_sqrt(std::int64_t v) -> std::optional<std::int64_t> {
    if (v <= 0) {
        return std::nullopt;
    }
    const auto approx = static_cast<std::int64_t>(std::sqrt(static_cast<double>(v)));
    for (std::int64_t k = approx - 2; k <= approx + 2; ++k) {
        if (k > 0 && k <= 3037000499 && k * k == v) {  // k*k stays within int64
            return k;
        }
    }
    return std::nullopt;
}

// int 1/(x^2 + k^2) dx = (1/k) atan(x/k) and int 1/sqrt(k^2 - x^2) dx = asin(x/k), for
// an integer k. Recognises the base as an integer quadratic c2*x^2 + c0 (no linear term)
// and matches the reciprocal (exp == -1) or inverse-square-root (exp == -1/2) shape.
[[nodiscard]] auto integrate_quadratic_form(const Expr& base, const Expr& exp,
                                            std::string_view var) -> Result<Expr> {
    const Expr x = Expr::symbol(std::string(var));
    auto poly = to_polynomial(base, var);
    if (!poly) {
        return make_error<Expr>(MathError::not_implemented);
    }
    auto coeffs = poly->coefficients();  // low-degree first
    if (coeffs.size() != 3 || coeffs[1] != 0) {
        return make_error<Expr>(MathError::not_implemented);  // must be c2*x^2 + c0
    }
    const std::int64_t c0 = coeffs[0];
    const std::int64_t c2 = coeffs[2];

    auto rational_minus_half = Expr::rational(-1, 2);
    const bool is_recip = exp.is_equivalent_to(Expr::integer(-1));
    const bool is_inv_sqrt =
        rational_minus_half && exp.is_equivalent_to(*rational_minus_half);

    // int 1/(x^2 + k^2) dx = (1/k) atan(x/k).
    if (is_recip && c2 == 1) {
        auto k = perfect_sqrt(c0);
        if (!k) {
            return make_error<Expr>(MathError::not_implemented);
        }
        const Expr k_expr = Expr::integer(*k);
        Expr arg = (*k == 1) ? x : Expr::product({x, recip(k_expr)});  // x/k
        return Expr::product({recip(k_expr), Expr::apply("atan", {arg})});
    }
    // int 1/sqrt(k^2 - x^2) dx = asin(x/k).
    if (is_inv_sqrt && c2 == -1) {
        auto k = perfect_sqrt(c0);
        if (!k) {
            return make_error<Expr>(MathError::not_implemented);
        }
        const Expr k_expr = Expr::integer(*k);
        Expr arg = (*k == 1) ? x : Expr::product({x, recip(k_expr)});  // x/k
        return Expr::apply("asin", {arg});
    }
    return make_error<Expr>(MathError::not_implemented);
}

// --- rational-function bridge ----------------------------------------------

// A ConstantNode holding a strictly negative int64, returned as its positive magnitude
// (denominator exponent), else nullopt. INT64_MIN is rejected (its negation overflows).
[[nodiscard]] auto negative_integer_magnitude(const Expr& exp) -> std::optional<std::int64_t> {
    if (const auto* c = std::get_if<ConstantNode>(&exp.node().value)) {
        if (const auto* v = std::get_if<std::int64_t>(&c->value)) {
            if (*v < 0 && *v != std::numeric_limits<std::int64_t>::min()) {
                return -*v;
            }
        }
    }
    return std::nullopt;
}

// Split f into (numerator, denominator) polynomial-Exprs. A factor g^(-k) (k > 0) moves
// into the denominator as g^k; everything else is numerator. Sums are NOT split over a
// common denominator here — linearity has already peeled them apart before this point.
[[nodiscard]] auto rational_parts(const Expr& f) -> std::pair<Expr, Expr> {
    const auto build = [](std::vector<Expr> parts) -> Expr {
        if (parts.empty()) {
            return Expr::integer(1);
        }
        if (parts.size() == 1) {
            return std::move(parts.front());
        }
        return Expr::product(std::move(parts));
    };

    if (const auto* mul = std::get_if<MulNode>(&f.node().value)) {
        std::vector<Expr> num;
        std::vector<Expr> den;
        for (const Expr& factor : mul->factors) {
            if (const auto* pw = std::get_if<PowerNode>(&factor.node().value)) {
                if (auto k = negative_integer_magnitude(pw->exponent)) {
                    den.push_back(Expr::power(pw->base, Expr::integer(*k)));
                    continue;
                }
            }
            num.push_back(factor);
        }
        return {build(std::move(num)), build(std::move(den))};
    }
    if (const auto* pw = std::get_if<PowerNode>(&f.node().value)) {
        if (auto k = negative_integer_magnitude(pw->exponent)) {
            return {Expr::integer(1), Expr::power(pw->base, Expr::integer(*k))};
        }
    }
    return {f, Expr::integer(1)};
}

// A RationalPoly rebuilt as a sum-of-monomials Expr in var (rational coefficients kept
// exact). from_polynomial in nimblecas.polyexpr only covers integer polynomials, so the
// rational-coefficient case is handled here.
[[nodiscard]] auto ratpoly_to_expr(const RationalPoly& p, std::string_view var) -> Expr {
    if (p.is_zero()) {
        return Expr::integer(0);
    }
    const Expr x = Expr::symbol(std::string(var));
    auto coeffs = p.coefficients();
    std::vector<Expr> terms;
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        if (coeffs[i].is_zero()) {
            continue;
        }
        const Expr c = rational_to_expr(coeffs[i]);
        if (i == 0) {
            terms.push_back(c);
        } else if (i == 1) {
            terms.push_back(Expr::product({c, x}));
        } else {
            terms.push_back(Expr::product(
                {c, Expr::power(x, Expr::integer(static_cast<std::int64_t>(i)))}));
        }
    }
    if (terms.size() == 1) {
        return std::move(terms.front());
    }
    return Expr::sum(std::move(terms));
}

// int numerator/denominator dx via the exact rational integrator, reassembled as an Expr:
//   rational_num/rational_den + sum_i coefficient_i * ln(argument_i).
[[nodiscard]] auto integrate_rational_expr(const Expr& f, std::string_view var)
    -> Result<Expr> {
    auto [num_expr, den_expr] = rational_parts(f);
    auto num_poly = to_polynomial(num_expr, var);
    auto den_poly = to_polynomial(den_expr, var);
    if (!num_poly || !den_poly) {
        return make_error<Expr>(MathError::not_implemented);  // not a ratio of polynomials
    }
    const RationalPoly num_rp = RationalPoly::from_polynomial(*num_poly);
    const RationalPoly den_rp = RationalPoly::from_polynomial(*den_poly);
    auto integral = integrate_rational(num_rp, den_rp);
    if (!integral) {
        return make_error<Expr>(integral.error());  // e.g. not_implemented (complex residue)
    }

    std::vector<Expr> parts;
    if (!integral->rational_num.is_zero()) {
        Expr rat = ratpoly_to_expr(integral->rational_num, var);
        const RationalPoly& rd = integral->rational_den;
        const bool den_is_one = rd.degree() == 0 && rd.coefficient(0) == Rational::from_int(1);
        if (den_is_one) {
            parts.push_back(std::move(rat));
        } else {
            parts.push_back(Expr::product({rat, recip(ratpoly_to_expr(rd, var))}));
        }
    }
    for (const LogTerm& term : integral->log_terms) {
        Expr coeff = rational_to_expr(term.coefficient);
        Expr arg = ratpoly_to_expr(term.argument, var);
        parts.push_back(Expr::product({coeff, Expr::apply("ln", {arg})}));
    }
    if (parts.empty()) {
        return Expr::integer(0);
    }
    if (parts.size() == 1) {
        return std::move(parts.front());
    }
    return Expr::sum(std::move(parts));
}

// --- dispatch --------------------------------------------------------------

// The recursive core, returning an unsimplified antiderivative (the public entry point
// simplifies once at the top). Strategy helpers that decline return not_implemented,
// which is treated as "try the next strategy"; a returned value is always correct.
[[nodiscard]] auto integrate_raw(const Expr& f, std::string_view var) -> Result<Expr> {
    const Expr x = Expr::symbol(std::string(var));

    // int c dx = c*x for a constant integrand (free of var).
    if (free_of(f, x)) {
        return Expr::product({f, x});
    }

    const ExprNode& node = f.node();

    // Linearity: int sum(t_i) dx = sum(int t_i dx).
    if (const auto* add = std::get_if<AddNode>(&node.value)) {
        std::vector<Expr> out;
        out.reserve(add->terms.size());
        for (const Expr& term : add->terms) {
            auto part = integrate_raw(term, var);
            if (!part) {
                return part;
            }
            out.push_back(std::move(*part));
        }
        return Expr::sum(std::move(out));
    }

    // Pull constant factors out of a product; integrate the var-dependent remainder.
    if (const auto* mul = std::get_if<MulNode>(&node.value)) {
        std::vector<Expr> constants;
        std::vector<Expr> variables;
        for (const Expr& factor : mul->factors) {
            (free_of(factor, x) ? constants : variables).push_back(factor);
        }
        if (variables.empty()) {
            return Expr::product({f, x});  // wholly constant (defensive; free_of covers it)
        }
        if (!constants.empty()) {
            Expr inner = variables.size() == 1 ? variables.front()
                                               : Expr::product(std::move(variables));
            auto integrated = integrate_raw(inner, var);
            if (!integrated) {
                return integrated;
            }
            constants.push_back(std::move(*integrated));
            return Expr::product(std::move(constants));
        }
        // No constant factors: a genuine product of var-dependent factors.
        if (variables.size() == 1) {
            return integrate_raw(variables.front(), var);
        }
        return integrate_rational_expr(f, var);  // ratio of polynomials, or not_implemented
    }

    // Power: power rule, then the inverse-trig quadratic table, then the rational bridge.
    if (const auto* pw = std::get_if<PowerNode>(&node.value)) {
        if (auto r = integrate_affine_power(pw->base, pw->exponent, var)) {
            return r;
        }
        if (auto r = integrate_quadratic_form(pw->base, pw->exponent, var)) {
            return r;
        }
        return integrate_rational_expr(f, var);
    }

    // Elementary function with an affine argument.
    if (const auto* fn = std::get_if<FunctionNode>(&node.value)) {
        return integrate_function(*fn, var);
    }

    // A bare symbol equal to var (other symbols are free_of and handled above): int x dx.
    if (std::holds_alternative<SymbolNode>(node.value)) {
        return integrate_affine_power(f, Expr::integer(1), var);
    }

    return make_error<Expr>(MathError::not_implemented);
}

}  // namespace

auto integrate(const Expr& f, std::string_view var) -> Result<Expr> {
    auto raw = integrate_raw(f, var);
    if (!raw) {
        return raw;
    }
    return simplify(*raw);
}

auto integrate_definite(const Expr& f, std::string_view var, const Expr& a, const Expr& b)
    -> Result<Expr> {
    auto antideriv = integrate(f, var);
    if (!antideriv) {
        return antideriv;
    }
    const Expr x = Expr::symbol(std::string(var));
    Expr upper = substitute(*antideriv, x, b);  // F(b)
    Expr lower = substitute(*antideriv, x, a);  // F(a)
    return simplify(Expr::sum({upper, neg(lower)}));
}

}  // namespace nimblecas
