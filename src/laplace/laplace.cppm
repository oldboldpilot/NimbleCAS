// NimbleCAS table-driven symbolic Laplace transform (ROADMAP 7.6).
// @author Olumuyiwa Oluwasanmi
//
// laplace_transform(f, t, s) computes F(s) = L{f(t)} by LINEARITY over a small
// table of elementary forms in the variable t:
//
//   L{c}        = c / s                    (c free of t)
//   L{t^n}      = n! / s^(n+1)             (n a non-negative integer)
//   L{e^(a t)}  = 1 / (s - a)              (a free of t)
//   L{sin(a t)} = a / (s^2 + a^2)          (a a constant)
//   L{cos(a t)} = s / (s^2 + a^2)          (a a constant)
//   L{delta(t - a)} = e^(-a s)             (Dirac delta; L{delta(t)} = 1)
//   L{u(t - a)}     = e^(-a s) / s         (Heaviside step; L{u(t)} = 1/s)
//   L{f + g}    = L{f} + L{g}              (linearity)
//   L{c g}      = c L{g}                   (c a constant factor, free of t)
//
// The transform is assembled unevaluated from Expr factories and then reduced by
// simplify() to a canonical form. Anything outside the table yields
// MathError::not_implemented; a factorial overflow yields MathError::overflow, so
// the operation is total (Rule 32 — no exceptions).
//
// inverse_laplace(F, s, t) inverts the RATIONAL-function class exactly: F is read as
// a ratio of polynomials in s over Q, partial-fraction decomposed via nimblecas.pfd,
// and each term inverted through the standard table (real poles 1/(s-a)^k, irreducible
// quadratics, distinct real quadratics that factor over Q). A denominator factor the
// table does not cover — a higher-degree irreducible, repeated complex poles, or an
// irrational real pole — yields MathError::not_implemented: an honest boundary, never a
// wrong inverse.

export module nimblecas.laplace;

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.ratpoly;  // Rational, RationalPoly — the coefficient field for the inverse
import nimblecas.pfd;      // partial_fractions — the inverse's rational-function decomposition

export namespace nimblecas {

// F(s) = L{f(t)}. The transform variable is `t` (the argument name in f) and the
// image variable is `s`. Elementary forms are recognised by structure; unmatched
// input returns MathError::not_implemented.
[[nodiscard]] auto laplace_transform(const Expr& f, std::string_view t, std::string_view s)
    -> Result<Expr>;

// f(t) = L^{-1}{F(s)} for F in the rational-function class over Q: a ratio of
// polynomials in the image variable `s`. `s` is the image variable (the symbol
// appearing in F) and `t` is the time variable of the result. F is decomposed by
// partial fractions and each term inverted through the elementary table. A polynomial
// part of degree 0 (an improper F with a constant excess) inverts to a Dirac delta
// c·delta(t); a higher-degree polynomial part (delta derivatives), a denominator factor
// of degree > 2, repeated complex poles, or an irrational real pole all yield
// MathError::not_implemented. A non-rational F (an unmatched symbol, an inexact real
// coefficient, or a transcendental subexpression) is likewise not_implemented.
[[nodiscard]] auto inverse_laplace(const Expr& F, std::string_view s, std::string_view t)
    -> Result<Expr>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// n! computed in int64, overflowing gracefully (20! fits, 21! does not).
[[nodiscard]] auto factorial(std::int64_t n) -> Result<std::int64_t> {
    std::int64_t result = 1;
    for (std::int64_t i = 2; i <= n; ++i) {
        if (__builtin_mul_overflow(result, i, &result)) {
            return make_error<std::int64_t>(MathError::overflow);
        }
    }
    return result;
}

// Extract a plain integer from a constant leaf (int64 directly, or an n/1 rational).
[[nodiscard]] auto integer_value(const Expr& e) -> std::optional<std::int64_t> {
    auto c = as<ConstantNode>(e.node().value);
    if (!c) {
        return std::nullopt;
    }
    if (auto i = as<std::int64_t>((*c)->value)) {
        return **i;
    }
    if (auto pair = as<std::pair<std::int64_t, std::int64_t>>((*c)->value)) {
        if ((*pair)->second == 1) {
            return (*pair)->first;
        }
    }
    return std::nullopt;
}

// L{t^n} = n! / s^(n+1) = n! * s^(-(n+1)), assembled unevaluated. n >= 0.
[[nodiscard]] auto transform_power_of_t(std::int64_t n, const Expr& s) -> Result<Expr> {
    auto fact = factorial(n);
    if (!fact) {
        return make_error<Expr>(fact.error());
    }
    std::int64_t exponent = 0;
    if (__builtin_add_overflow(n, std::int64_t{1}, &exponent)) {
        return make_error<Expr>(MathError::overflow);
    }
    return Expr::product({Expr::integer(*fact), Expr::power(s, Expr::integer(-exponent))});
}

// If arg has the linear form a*t with a free of t, return a (a == 1 for a bare t);
// otherwise nullopt. Matches a lone t or a product with exactly one bare t factor.
[[nodiscard]] auto linear_coefficient(const Expr& arg, const Expr& t) -> std::optional<Expr> {
    if (arg.is_equivalent_to(t)) {
        return Expr::integer(1);
    }
    if (auto mul = as<MulNode>(arg.node().value)) {
        std::vector<Expr> rest;
        bool found_t = false;
        for (const Expr& factor : (*mul)->factors) {
            if (!found_t && factor.is_equivalent_to(t)) {
                found_t = true;
            } else if (free_of(factor, t)) {
                rest.push_back(factor);
            } else {
                return std::nullopt;  // a non-t factor still depends on t: not linear
            }
        }
        if (!found_t) {
            return std::nullopt;
        }
        if (rest.empty()) {
            return Expr::integer(1);
        }
        if (rest.size() == 1) {
            return rest.front();
        }
        return Expr::product(std::move(rest));
    }
    return std::nullopt;
}

// If arg has the shifted form t + c with c free of t (a lone t gives c == 0), return c;
// otherwise nullopt. This recognises the delayed argument t - a of a shifted Dirac/step.
[[nodiscard]] auto time_shift(const Expr& arg, const Expr& t) -> std::optional<Expr> {
    if (arg.is_equivalent_to(t)) {
        return Expr::integer(0);
    }
    if (auto add = as<AddNode>(arg.node().value)) {
        std::vector<Expr> rest;
        bool found_t = false;
        for (const Expr& term : (*add)->terms) {
            if (!found_t && term.is_equivalent_to(t)) {
                found_t = true;
            } else if (free_of(term, t)) {
                rest.push_back(term);
            } else {
                return std::nullopt;  // a non-t term still depends on t: not a pure shift
            }
        }
        if (!found_t) {
            return std::nullopt;
        }
        if (rest.empty()) {
            return Expr::integer(0);
        }
        if (rest.size() == 1) {
            return rest.front();
        }
        return Expr::sum(std::move(rest));
    }
    return std::nullopt;
}

// Table entries for exp / sin / cos, each requiring an argument of the form a*t, plus
// the shifted Dirac delta and Heaviside step, each requiring an argument of the form
// t - a (a free of t). The full second-shift theorem L{u(t-a) f(t-a)} = e^(-a s) F(s)
// is the identity behind the step entry; only the f == 1 step (and the bare delta) are
// recognised structurally here — a product u(t-a)·g(t) is left as not_implemented rather
// than guessed at.
[[nodiscard]] auto transform_function(const FunctionNode& fn, const Expr& t, const Expr& s)
    -> Result<Expr> {
    if (fn.args.size() != 1) {
        return make_error<Expr>(MathError::not_implemented);
    }
    // delta(t - a) -> e^(-a s) ; u(t - a) -> e^(-a s)/s. With arg = t + c we have c = -a,
    // so the shift factor is e^(c s) (== 1 when c == 0, i.e. an unshifted t).
    if (fn.name == "dirac" || fn.name == "heaviside") {
        auto c = time_shift(fn.args.front(), t);
        if (!c) {
            return make_error<Expr>(MathError::not_implemented);
        }
        const Expr shift = c->is_equivalent_to(Expr::integer(0))
                               ? Expr::integer(1)
                               : Expr::apply("exp", {Expr::product({*c, s})});
        if (fn.name == "dirac") {
            return shift;
        }
        return Expr::product({shift, Expr::power(s, Expr::integer(-1))});
    }
    auto a = linear_coefficient(fn.args.front(), t);
    if (!a) {
        return make_error<Expr>(MathError::not_implemented);
    }
    if (fn.name == "exp") {
        // 1 / (s - a)
        Expr denom = Expr::sum({s, Expr::product({Expr::integer(-1), *a})});
        return Expr::power(std::move(denom), Expr::integer(-1));
    }
    if (fn.name == "sin" || fn.name == "cos") {
        // denominator s^2 + a^2, shared by both
        Expr denom = Expr::sum({Expr::power(s, Expr::integer(2)),
                                Expr::power(*a, Expr::integer(2))});
        Expr recip = Expr::power(std::move(denom), Expr::integer(-1));
        // L{sin(a t)} = a / (s^2 + a^2) ; L{cos(a t)} = s / (s^2 + a^2)
        Expr numerator = fn.name == "sin" ? *a : s;
        return Expr::product({std::move(numerator), std::move(recip)});
    }
    return make_error<Expr>(MathError::not_implemented);
}

// Assemble the (unsimplified) transform by structural dispatch over the table.
[[nodiscard]] auto transform(const Expr& f, const Expr& t, const Expr& s) -> Result<Expr> {
    // L{c} = c / s for any c free of t (constants, other symbols, or subexpressions).
    if (free_of(f, t)) {
        return Expr::product({f, Expr::power(s, Expr::integer(-1))});
    }

    const ExprNode& node = f.node();

    // A bare, t-dependent symbol can only be t itself (t^1).
    if (as<SymbolNode>(node.value)) {
        return transform_power_of_t(1, s);
    }

    // t^n: base == t, exponent a positive integer.
    if (auto power = as<PowerNode>(node.value)) {
        if ((*power)->base.is_equivalent_to(t)) {
            if (auto n = integer_value((*power)->exponent); n && *n >= 1) {
                return transform_power_of_t(*n, s);
            }
        }
        return make_error<Expr>(MathError::not_implemented);
    }

    // exp / sin / cos.
    if (auto fn = as<FunctionNode>(node.value)) {
        return transform_function(**fn, t, s);
    }

    // Linearity: transform each term and sum.
    if (auto add = as<AddNode>(node.value)) {
        std::vector<Expr> terms;
        terms.reserve((*add)->terms.size());
        for (const Expr& term : (*add)->terms) {
            auto lt = transform(term, t, s);
            if (!lt) {
                return lt;
            }
            terms.push_back(std::move(*lt));
        }
        return Expr::sum(std::move(terms));
    }

    // c * g: pull out the factors free of t (constants) and transform the single
    // remaining t-dependent factor. More than one t-dependent factor is unhandled.
    if (auto mul = as<MulNode>(node.value)) {
        std::vector<Expr> constants;
        std::vector<Expr> variables;
        for (const Expr& factor : (*mul)->factors) {
            if (free_of(factor, t)) {
                constants.push_back(factor);
            } else {
                variables.push_back(factor);
            }
        }
        if (variables.size() != 1) {
            return make_error<Expr>(MathError::not_implemented);
        }
        auto lg = transform(variables.front(), t, s);
        if (!lg) {
            return lg;
        }
        if (constants.empty()) {
            return lg;
        }
        std::vector<Expr> factors = std::move(constants);
        factors.push_back(std::move(*lg));
        return Expr::product(std::move(factors));
    }

    return make_error<Expr>(MathError::not_implemented);
}

// ===========================================================================
// Inverse transform (rational-function class).
// ===========================================================================

// A Rational carried into an Expr leaf: an integer leaf when integral, else an exact
// rational leaf. Expr::rational only fails on INT64_MIN, so this is total for the
// canonical (den > 0, reduced) Rationals produced by the arithmetic below.
[[nodiscard]] auto rational_to_expr(const Rational& r) -> Result<Expr> {
    if (r.is_integer()) {
        return Expr::integer(r.numerator());
    }
    return Expr::rational(r.numerator(), r.denominator());
}

// The exact integer square root of n >= 0 when n is a perfect square, else nullopt.
// The multiply is overflow-guarded so the perfect-square test can never wrap.
[[nodiscard]] auto integer_sqrt_exact(std::int64_t n) -> std::optional<std::int64_t> {
    if (n < 0) {
        return std::nullopt;
    }
    const auto approx = static_cast<std::int64_t>(std::llround(std::sqrt(static_cast<double>(n))));
    for (std::int64_t cand = approx > 1 ? approx - 1 : 0; cand <= approx + 1; ++cand) {
        std::int64_t sq = 0;
        if (!__builtin_mul_overflow(cand, cand, &sq) && sq == n) {
            return cand;
        }
    }
    return std::nullopt;
}

// The exact rational square root of w (assumed > 0, canonical) when both numerator and
// denominator are perfect squares, else nullopt (an irrational root we do not represent
// as an exact Rational — the caller falls back to a symbolic w^(1/2) or refuses).
[[nodiscard]] auto rational_sqrt_exact(const Rational& w) -> std::optional<Rational> {
    auto sn = integer_sqrt_exact(w.numerator());
    auto sd = integer_sqrt_exact(w.denominator());
    if (!sn || !sd) {
        return std::nullopt;
    }
    auto r = Rational::make(*sn, *sd);
    if (!r) {
        return std::nullopt;
    }
    return *r;
}

// b(x)^n via repeated multiplication (n >= 0); overflow surfaces from RationalPoly.
[[nodiscard]] auto poly_pow(const RationalPoly& b, std::int64_t n) -> Result<RationalPoly> {
    RationalPoly acc = RationalPoly::constant(Rational::from_int(1));
    for (std::int64_t k = 0; k < n; ++k) {
        auto next = acc.multiply(b);
        if (!next) {
            return next;
        }
        acc = std::move(*next);
    }
    return acc;
}

// F(s) as an exact ratio num/den of polynomials in the image variable.
struct Ratio {
    RationalPoly num;
    RationalPoly den;
};

// Read an Expr as a rational function of `var` over Q. Constants (integer or rational),
// the variable itself, sums, products and integer powers compose exactly; a foreign
// symbol, an inexact real, a non-integer exponent, or a function subexpression makes the
// expression non-rational (MathError::not_implemented). A 0^negative base is
// division_by_zero.
[[nodiscard]] auto extract_ratio(const Expr& u, std::string_view var) -> Result<Ratio> {
    const RationalPoly one = RationalPoly::constant(Rational::from_int(1));
    const ExprNode& node = u.node();

    if (auto c = as<ConstantNode>(node.value)) {
        if (auto i = as<std::int64_t>((*c)->value)) {
            return Ratio{.num = RationalPoly::constant(Rational::from_int(**i)), .den = one};
        }
        if (auto pr = as<std::pair<std::int64_t, std::int64_t>>((*c)->value)) {
            auto r = Rational::make((*pr)->first, (*pr)->second);
            if (!r) {
                return make_error<Ratio>(r.error());
            }
            return Ratio{.num = RationalPoly::constant(*r), .den = one};
        }
        return make_error<Ratio>(MathError::not_implemented);  // inexact double coefficient
    }

    if (auto sym = as<SymbolNode>(node.value)) {
        if ((*sym)->name == var) {
            return Ratio{.num = RationalPoly::monomial(Rational::from_int(1), 1), .den = one};
        }
        return make_error<Ratio>(MathError::not_implemented);  // a foreign (non-rational) symbol
    }

    if (auto add = as<AddNode>(node.value)) {
        Ratio acc{.num = RationalPoly{}, .den = one};  // 0 / 1
        for (const Expr& term : (*add)->terms) {
            auto r = extract_ratio(term, var);
            if (!r) {
                return r;
            }
            // acc + r = (acc.num*r.den + r.num*acc.den) / (acc.den*r.den).
            auto lhs = acc.num.multiply(r->den);
            if (!lhs) {
                return make_error<Ratio>(lhs.error());
            }
            auto rhs = r->num.multiply(acc.den);
            if (!rhs) {
                return make_error<Ratio>(rhs.error());
            }
            auto sum = lhs->add(*rhs);
            if (!sum) {
                return make_error<Ratio>(sum.error());
            }
            auto den = acc.den.multiply(r->den);
            if (!den) {
                return make_error<Ratio>(den.error());
            }
            acc.num = std::move(*sum);
            acc.den = std::move(*den);
        }
        return acc;
    }

    if (auto mul = as<MulNode>(node.value)) {
        Ratio acc{.num = one, .den = one};
        for (const Expr& factor : (*mul)->factors) {
            auto r = extract_ratio(factor, var);
            if (!r) {
                return r;
            }
            auto num = acc.num.multiply(r->num);
            if (!num) {
                return make_error<Ratio>(num.error());
            }
            auto den = acc.den.multiply(r->den);
            if (!den) {
                return make_error<Ratio>(den.error());
            }
            acc.num = std::move(*num);
            acc.den = std::move(*den);
        }
        return acc;
    }

    if (auto pw = as<PowerNode>(node.value)) {
        auto k = integer_value((*pw)->exponent);
        if (!k) {
            return make_error<Ratio>(MathError::not_implemented);  // non-integer exponent
        }
        auto base = extract_ratio((*pw)->base, var);
        if (!base) {
            return base;
        }
        if (*k >= 0) {
            auto num = poly_pow(base->num, *k);
            if (!num) {
                return make_error<Ratio>(num.error());
            }
            auto den = poly_pow(base->den, *k);
            if (!den) {
                return make_error<Ratio>(den.error());
            }
            return Ratio{.num = std::move(*num), .den = std::move(*den)};
        }
        if (*k == std::numeric_limits<std::int64_t>::min()) {
            return make_error<Ratio>(MathError::overflow);  // -k would overflow
        }
        const std::int64_t m = -*k;
        if (base->num.is_zero()) {
            return make_error<Ratio>(MathError::division_by_zero);  // 0^negative
        }
        auto num = poly_pow(base->den, m);  // (den/num)^m — the reciprocal
        if (!num) {
            return make_error<Ratio>(num.error());
        }
        auto den = poly_pow(base->num, m);
        if (!den) {
            return make_error<Ratio>(den.error());
        }
        return Ratio{.num = std::move(*num), .den = std::move(*den)};
    }

    return make_error<Ratio>(MathError::not_implemented);  // a function is not rational in var
}

// L^{-1}{ C / (s - a)^power } = C · t^(power-1) · e^(a t) / (power-1)! for the monic
// linear factor `factor` = s - a (a == -constant term) and constant numerator C.
[[nodiscard]] auto invert_linear(const RationalPoly& factor, std::int64_t power,
                                 const RationalPoly& numer, const Expr& t) -> Result<Expr> {
    auto a = factor.coefficient(0).negate();  // factor = s + c0, pole a = -c0
    if (!a) {
        return make_error<Expr>(a.error());
    }
    auto fact = factorial(power - 1);  // power >= 1, so power-1 >= 0
    if (!fact) {
        return make_error<Expr>(fact.error());
    }
    auto coeff = numer.coefficient(0).divide(Rational::from_int(*fact));  // C / (power-1)!
    if (!coeff) {
        return make_error<Expr>(coeff.error());
    }
    auto coeff_e = rational_to_expr(*coeff);
    if (!coeff_e) {
        return make_error<Expr>(coeff_e.error());
    }
    std::vector<Expr> factors;
    factors.push_back(std::move(*coeff_e));
    if (power - 1 > 0) {
        factors.push_back(Expr::power(t, Expr::integer(power - 1)));
    }
    if (!a->is_zero()) {  // a == 0 => e^(0) = 1, omit the exponential
        auto a_e = rational_to_expr(*a);
        if (!a_e) {
            return make_error<Expr>(a_e.error());
        }
        factors.push_back(Expr::apply("exp", {Expr::product({std::move(*a_e), t})}));
    }
    return Expr::product(std::move(factors));
}

// L^{-1}{ (c1 s + c0) / ((s+alpha)^2 + w) } for w > 0 (complex-conjugate poles):
//   e^(-alpha t) [ c1 cos(omega t) + ((c0 - c1 alpha)/omega) sin(omega t) ],  omega = sqrt(w).
// omega is exact when w is a perfect rational square, else the symbolic leaf w^(1/2).
[[nodiscard]] auto invert_irreducible_quadratic(const Rational& alpha, const Rational& w,
                                                const Rational& c1, const Rational& c0,
                                                const Expr& t) -> Result<Expr> {
    auto c1a = c1.multiply(alpha);
    if (!c1a) {
        return make_error<Expr>(c1a.error());
    }
    auto coef_b = c0.subtract(*c1a);  // c0 - c1*alpha (numerator of the sin coefficient)
    if (!coef_b) {
        return make_error<Expr>(coef_b.error());
    }

    Expr omega_e = Expr::integer(1);
    Expr sin_coeff_e = Expr::integer(0);
    bool omega_is_one = false;
    if (auto omega_r = rational_sqrt_exact(w)) {
        omega_is_one = *omega_r == Rational::from_int(1);
        auto oe = rational_to_expr(*omega_r);
        if (!oe) {
            return make_error<Expr>(oe.error());
        }
        omega_e = std::move(*oe);
        auto sc = coef_b->divide(*omega_r);  // (c0 - c1 alpha)/omega, exact rational
        if (!sc) {
            return make_error<Expr>(sc.error());
        }
        auto sce = rational_to_expr(*sc);
        if (!sce) {
            return make_error<Expr>(sce.error());
        }
        sin_coeff_e = std::move(*sce);
    } else {
        // Irrational omega: carry it symbolically as w^(1/2); 1/omega is w^(-1/2).
        auto w_e = rational_to_expr(w);
        if (!w_e) {
            return make_error<Expr>(w_e.error());
        }
        auto half = Expr::rational(1, 2);
        auto neg_half = Expr::rational(-1, 2);
        if (!half || !neg_half) {
            return make_error<Expr>(MathError::overflow);
        }
        omega_e = Expr::power(*w_e, *half);
        auto coef_b_e = rational_to_expr(*coef_b);
        if (!coef_b_e) {
            return make_error<Expr>(coef_b_e.error());
        }
        sin_coeff_e = Expr::product({std::move(*coef_b_e), Expr::power(*w_e, *neg_half)});
    }

    const Expr arg = omega_is_one ? t : Expr::product({omega_e, t});  // omega t
    std::vector<Expr> inner;
    if (!c1.is_zero()) {
        auto c1_e = rational_to_expr(c1);
        if (!c1_e) {
            return make_error<Expr>(c1_e.error());
        }
        inner.push_back(Expr::product({std::move(*c1_e), Expr::apply("cos", {arg})}));
    }
    if (!sin_coeff_e.is_equivalent_to(Expr::integer(0))) {
        inner.push_back(Expr::product({sin_coeff_e, Expr::apply("sin", {arg})}));
    }
    if (inner.empty()) {
        inner.push_back(Expr::integer(0));
    }
    Expr trig = Expr::sum(std::move(inner));

    if (alpha.is_zero()) {  // no decay envelope
        return trig;
    }
    auto neg_alpha = alpha.negate();
    if (!neg_alpha) {
        return make_error<Expr>(neg_alpha.error());
    }
    auto nae = rational_to_expr(*neg_alpha);
    if (!nae) {
        return make_error<Expr>(nae.error());
    }
    Expr decay = Expr::apply("exp", {Expr::product({std::move(*nae), t})});
    return Expr::product({std::move(decay), std::move(trig)});
}

// L^{-1}{ (c1 s + c0) / ((s - r1)(s - r2)) } for distinct real rational poles
// r1,2 = -alpha +/- root: A e^(r1 t) + B e^(r2 t) with A = C(r1)/(r1-r2), B = C(r2)/(r2-r1)
// and C(r) = c1 r + c0.
[[nodiscard]] auto invert_distinct_real(const Rational& alpha, const Rational& root,
                                        const Rational& c1, const Rational& c0,
                                        const Expr& t) -> Result<Expr> {
    auto neg_alpha = alpha.negate();
    if (!neg_alpha) {
        return make_error<Expr>(neg_alpha.error());
    }
    auto r1 = neg_alpha->add(root);
    if (!r1) {
        return make_error<Expr>(r1.error());
    }
    auto r2 = neg_alpha->subtract(root);
    if (!r2) {
        return make_error<Expr>(r2.error());
    }
    auto eval_c = [&](const Rational& r) -> Result<Rational> {
        auto m = c1.multiply(r);
        if (!m) {
            return m;
        }
        return m->add(c0);
    };
    auto c_r1 = eval_c(*r1);
    if (!c_r1) {
        return make_error<Expr>(c_r1.error());
    }
    auto c_r2 = eval_c(*r2);
    if (!c_r2) {
        return make_error<Expr>(c_r2.error());
    }
    auto denom = r1->subtract(*r2);  // r1 - r2 = 2*root != 0 (distinct poles)
    if (!denom) {
        return make_error<Expr>(denom.error());
    }
    auto neg_denom = denom->negate();
    if (!neg_denom) {
        return make_error<Expr>(neg_denom.error());
    }
    auto coeff_a = c_r1->divide(*denom);
    if (!coeff_a) {
        return make_error<Expr>(coeff_a.error());
    }
    auto coeff_b = c_r2->divide(*neg_denom);
    if (!coeff_b) {
        return make_error<Expr>(coeff_b.error());
    }
    auto build = [&](const Rational& coeff, const Rational& pole) -> Result<Expr> {
        auto ce = rational_to_expr(coeff);
        if (!ce) {
            return make_error<Expr>(ce.error());
        }
        std::vector<Expr> factors;
        factors.push_back(std::move(*ce));
        if (!pole.is_zero()) {
            auto pe = rational_to_expr(pole);
            if (!pe) {
                return make_error<Expr>(pe.error());
            }
            factors.push_back(Expr::apply("exp", {Expr::product({std::move(*pe), t})}));
        }
        return Expr::product(std::move(factors));
    };
    auto term1 = build(*coeff_a, *r1);
    if (!term1) {
        return term1;
    }
    auto term2 = build(*coeff_b, *r2);
    if (!term2) {
        return term2;
    }
    return Expr::sum({std::move(*term1), std::move(*term2)});
}

// Invert a single power-1 quadratic partial-fraction term. `factor` = s^2 + p s + q is
// monic and square-free; completing the square gives (s+alpha)^2 + w with alpha = p/2,
// w = q - alpha^2. w > 0 is the irreducible (complex-pole) case; w < 0 is two distinct
// real poles (handled only when they are rational — otherwise not_implemented). w == 0
// would be a repeated real root, impossible for a square-free factor.
[[nodiscard]] auto invert_quadratic(const RationalPoly& factor, const RationalPoly& numer,
                                    const Expr& t) -> Result<Expr> {
    const Rational p = factor.coefficient(1);
    const Rational q = factor.coefficient(0);
    const Rational c1 = numer.coefficient(1);  // numer = c1 s + c0, deg < 2
    const Rational c0 = numer.coefficient(0);

    auto alpha = p.divide(Rational::from_int(2));
    if (!alpha) {
        return make_error<Expr>(alpha.error());
    }
    auto a2 = alpha->multiply(*alpha);
    if (!a2) {
        return make_error<Expr>(a2.error());
    }
    auto w = q.subtract(*a2);
    if (!w) {
        return make_error<Expr>(w.error());
    }

    if (w->is_zero()) {
        return make_error<Expr>(MathError::not_implemented);  // repeated root (unreachable)
    }
    if (w->numerator() > 0) {
        return invert_irreducible_quadratic(*alpha, *w, c1, c0, t);
    }
    auto neg_w = w->negate();  // w < 0: real poles at -alpha +/- sqrt(-w)
    if (!neg_w) {
        return make_error<Expr>(neg_w.error());
    }
    auto root = rational_sqrt_exact(*neg_w);
    if (!root) {
        return make_error<Expr>(MathError::not_implemented);  // irrational real poles
    }
    return invert_distinct_real(*alpha, *root, c1, c0, t);
}

// Assemble L^{-1}{num/den} by partial fractions + the elementary table.
[[nodiscard]] auto inverse_transform(const RationalPoly& num, const RationalPoly& den,
                                     const Expr& t) -> Result<Expr> {
    auto pf = partial_fractions(num, den);
    if (!pf) {
        return make_error<Expr>(pf.error());
    }
    std::vector<Expr> terms;

    // Polynomial part: only a constant excess is representable (as a Dirac delta). A
    // higher-degree part would need delta derivatives, which we do not emit.
    if (!pf->polynomial_part.is_zero()) {
        if (pf->polynomial_part.degree() >= 1) {
            return make_error<Expr>(MathError::not_implemented);  // delta' and higher
        }
        auto c = rational_to_expr(pf->polynomial_part.coefficient(0));
        if (!c) {
            return make_error<Expr>(c.error());
        }
        terms.push_back(Expr::product({std::move(*c), Expr::apply("dirac", {t})}));
    }

    for (const auto& term : pf->terms) {
        const std::int64_t deg = term.factor.degree();
        if (deg == 1) {
            auto e = invert_linear(term.factor, term.power, term.numerator, t);
            if (!e) {
                return e;
            }
            terms.push_back(std::move(*e));
        } else if (deg == 2 && term.power == 1) {
            auto e = invert_quadratic(term.factor, term.numerator, t);
            if (!e) {
                return e;
            }
            terms.push_back(std::move(*e));
        } else {
            // Repeated complex poles (deg 2, power > 1) or a higher-degree irreducible
            // factor are outside the table: an honest boundary, not a wrong inverse.
            return make_error<Expr>(MathError::not_implemented);
        }
    }

    if (terms.empty()) {
        return Expr::integer(0);
    }
    return Expr::sum(std::move(terms));
}

}  // namespace

auto laplace_transform(const Expr& f, std::string_view t, std::string_view s) -> Result<Expr> {
    const Expr t_sym = Expr::symbol(std::string(t));
    const Expr s_sym = Expr::symbol(std::string(s));
    auto raw = transform(f, t_sym, s_sym);
    if (!raw) {
        return raw;
    }
    return simplify(*raw);
}

auto inverse_laplace(const Expr& F, std::string_view s, std::string_view t) -> Result<Expr> {
    // Read F as an exact rational function of the image variable s, then invert num/den.
    auto ratio = extract_ratio(F, s);
    if (!ratio) {
        return make_error<Expr>(ratio.error());
    }
    const Expr t_sym = Expr::symbol(std::string(t));
    auto raw = inverse_transform(ratio->num, ratio->den, t_sym);
    if (!raw) {
        return raw;
    }
    return simplify(*raw);
}

}  // namespace nimblecas
