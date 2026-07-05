// NimbleCAS singular-perturbation module: leading-order matched asymptotic expansion
// for the constant-coefficient linear boundary-layer problem.
// @author Olumuyiwa Oluwasanmi
//
// This module builds the classical leading-order MATCHED ASYMPTOTIC EXPANSION of the
// constant-coefficient linear two-point boundary-layer problem
//
//     ε y''(x) + a y'(x) + b y(x) = 0,   0 < x < 1,   y(0) = α,  y(1) = β,
//
// with a > 0, so the boundary layer sits at the LEFT endpoint x = 0. The coefficients
// a, b, α, β are exact Rationals; the three pieces are returned as symbolic `Expr` in
// the physical variable x = symbol("x") and the small parameter ε = symbol("epsilon").
//
// DERIVATION (leading order in ε).
//   * Outer solution. Set ε = 0: a y' + b y = 0 ⇒ y' = −(b/a) y ⇒ y_out = C e^{−(b/a)x}.
//     The outer solution carries the endpoint away from the layer, y_out(1) = β, giving
//         y_out(x) = β · exp((b/a)(1 − x)).
//   * Inner solution. Stretch ξ = x/ε; then ε y'' + a y' + (·) becomes, to leading
//     order, Y'' + a Y' = 0 ⇒ Y(ξ) = A + B e^{−aξ} (decaying as ξ→∞ because a > 0).
//     Matching Y(∞) = y_out(0) fixes A = β·exp(b/a); the layer BC Y(0) = α fixes
//     B = α − β·exp(b/a). Written back in x via ξ = x/ε:
//         y_in(x) = β·exp(b/a) + (α − β·exp(b/a)) · exp(−a x/ε).
//   * Common part (overlap / matching limit). The shared limit
//         y_common = lim_{ξ→∞} y_in = lim_{x→0} y_out = A = β·exp(b/a).
//   * Uniform composite. Additive matching y_unif = y_out + y_in − y_common; the A level
//     inside y_in cancels y_common analytically, leaving the classical closed form
//         y_unif(x) = β·exp((b/a)(1 − x)) + (α − β·exp(b/a)) · exp(−a x/ε).
//     At x = 0 the layer factor is exp(0) and y_unif(0) = α exactly; at x = 1 it equals
//     β up to the exponentially small exp(−a/ε), as leading-order matching guarantees.
//
// HONESTY BOUNDARY (Rule 32). What is returned is a LEADING-ORDER ASYMPTOTIC
// APPROXIMATION valid for small ε — it is NOT an exact closed-form solution of the ODE
// (the exact solution is a combination of two real exponentials with irrational rates
// (−a ± √(a² − 4εb))/(2ε), not representable exactly here). Only this specific class is
// handled: constant coefficients, a > 0 (left boundary layer), leading order. Anything
// else is out of scope and is reported honestly rather than approximated wrongly:
// a ≤ 0 (no decaying left-layer mode under this construction) returns
// MathError::domain_error. No plausible-but-wrong expansion is ever returned.

export module nimblecas.singpert;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.symbolic;

export namespace nimblecas {

// The three pieces of the leading-order matched asymptotic expansion, as symbolic Expr
// in x = symbol("x") and ε = symbol("epsilon"). Exponentials are exp(·) FunctionNodes.
struct MatchedExpansion {
    Expr outer;      // y_out(x)  = β·exp((b/a)(1 − x))
    Expr inner;      // y_in(x)   = β·exp(b/a) + (α − β·exp(b/a))·exp(−a x/ε)   (ξ = x/ε)
    Expr composite;  // y_unif(x) = β·exp((b/a)(1 − x)) + (α − β·exp(b/a))·exp(−a x/ε)
};

// Leading-order matched asymptotic expansion of ε y'' + a y' + b y = 0, y(0)=α, y(1)=β,
// for a > 0 (left boundary layer). See the module header for the derivation and the
// explicit honesty boundary: this is a small-ε ASYMPTOTIC approximation, not an exact
// solution, and only the constant-coefficient a > 0 left-layer leading-order case is in
// scope. a ≤ 0 is rejected with MathError::domain_error; any exact-arithmetic overflow
// in the Rational/Expr coefficient construction is propagated verbatim.
[[nodiscard]] auto matched_asymptotic(Rational a, Rational b, Rational alpha, Rational beta)
    -> Result<MatchedExpansion>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

namespace {

// An exact Rational as an Expr rational constant. A canonicalised Rational has
// denominator > 0 and no INT64_MIN operand, so Expr::rational cannot hit its
// division_by_zero branch here; its Result is still propagated honestly (Rule 32).
[[nodiscard]] auto rat_expr(const Rational& r) -> Result<Expr> {
    return Expr::rational(r.numerator(), r.denominator());
}

}  // namespace

auto matched_asymptotic(Rational a, Rational b, Rational alpha, Rational beta)
    -> Result<MatchedExpansion> {
    // Honesty boundary: only a > 0 (left boundary layer) is constructed. Rational is
    // canonicalised with denominator > 0, so the sign is exactly the numerator's; a ≤ 0
    // has no decaying inner mode exp(−aξ) under this matching and is out of scope.
    if (a.numerator() <= 0) {
        return make_error<MatchedExpansion>(MathError::domain_error);
    }

    // Exact rational combinations the expansion needs (a ≠ 0 is guaranteed above).
    auto ba = b.divide(a);  // b/a
    if (!ba) {
        return make_error<MatchedExpansion>(ba.error());
    }
    auto neg_a = a.negate();  // −a
    if (!neg_a) {
        return make_error<MatchedExpansion>(neg_a.error());
    }
    auto neg_beta = beta.negate();  // −β
    if (!neg_beta) {
        return make_error<MatchedExpansion>(neg_beta.error());
    }

    // Rational-constant Exprs.
    auto ba_e = rat_expr(*ba);
    if (!ba_e) {
        return make_error<MatchedExpansion>(ba_e.error());
    }
    auto alpha_e = rat_expr(alpha);
    if (!alpha_e) {
        return make_error<MatchedExpansion>(alpha_e.error());
    }
    auto beta_e = rat_expr(beta);
    if (!beta_e) {
        return make_error<MatchedExpansion>(beta_e.error());
    }
    auto neg_a_e = rat_expr(*neg_a);
    if (!neg_a_e) {
        return make_error<MatchedExpansion>(neg_a_e.error());
    }
    auto neg_beta_e = rat_expr(*neg_beta);
    if (!neg_beta_e) {
        return make_error<MatchedExpansion>(neg_beta_e.error());
    }

    const Expr x = Expr::symbol("x");
    const Expr eps = Expr::symbol("epsilon");
    const Expr one = Expr::integer(1);
    const Expr neg_one = Expr::integer(-1);

    // exp(b/a): the transcendental level of the boundary layer. Its argument is the
    // pure rational number b/a, kept symbolic as exp(·).
    const Expr exp_ba = Expr::apply("exp", {*ba_e});

    // Common part (overlap) A = lim_{ξ→∞} y_in = lim_{x→0} y_out = β·exp(b/a).
    const Expr common = Expr::product({*beta_e, exp_ba});

    // Inner amplitude B = α − β·exp(b/a) = α + (−β)·exp(b/a) (so y_in(0) = α).
    const Expr amplitude = Expr::sum({*alpha_e, Expr::product({*neg_beta_e, exp_ba})});

    // Boundary-layer factor exp(−a·x/ε) = exp(−a·ξ), ξ = x/ε. The symbolic engine has no
    // divide node, so division by ε is the power ε^(−1).
    const Expr layer =
        Expr::apply("exp", {Expr::product({*neg_a_e, x, Expr::power(eps, neg_one)})});

    // Outer solution y_out(x) = β·exp((b/a)(1 − x)); carries the x = 1 boundary condition.
    const Expr one_minus_x = Expr::sum({one, Expr::product({neg_one, x})});
    const Expr outer = Expr::product(
        {*beta_e, Expr::apply("exp", {Expr::product({*ba_e, one_minus_x})})});

    // Boundary-layer term B·exp(−a x/ε), shared verbatim by the inner solution and the
    // composite (in the composite the outer's A level cancels the common part exactly).
    const Expr layer_term = Expr::product({amplitude, layer});

    // Inner solution in the physical variable via ξ = x/ε, so the composite is a genuine
    // expression identity y_unif = y_out + y_in − y_common.
    const Expr inner = Expr::sum({common, layer_term});

    // Uniform composite: y_out + (y_in − y_common) with the common A level cancelled.
    const Expr composite = Expr::sum({outer, layer_term});

    return MatchedExpansion{.outer = outer, .inner = inner, .composite = composite};
}

}  // namespace nimblecas
