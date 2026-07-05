// Tests for nimblecas.pdeham: the Homotopy Analysis Method (HAM) with the ħ
// convergence-control parameter for the semilinear evolution PDE u_t = F[u].
// @author Olumuyiwa Oluwasanmi
//
// Coverage:
//   * order-1 and order-2 HAM terms against a by-hand computation for u_t = u^2, u(x,0)=x
//     at a general ħ = -1/2 (showing higher HAM orders correct lower t-coefficients);
//   * the ħ = -1 reduction to the ADM/HPM/Cauchy-Kovalevskaya forward series computed by
//     nimblecas.pde, on the shared reaction-diffusion example u_t = u_xx + u^2 and the
//     shared inviscid-Burgers example u_t + u u_x = 0;
//   * the HAM correctness property: differentiating the ħ = -1 truncated series back and
//     substituting into the PDE drives the residual to O(t^{order}) (its t^0..t^{order-1}
//     coefficients vanish exactly).

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.diff;
import nimblecas.expand;
import nimblecas.pde;
import nimblecas.pdeham;
import nimblecas.testing;

using nimblecas::Expr;
using nimblecas::HamPde;
using nimblecas::MathError;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto ri(std::int64_t v) -> Rational { return Rational::from_int(v); }
[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

// Expr constant from a Rational (integer when the denominator is 1).
[[nodiscard]] auto rc(const Rational& r) -> Expr {
    if (r.is_integer()) {
        return Expr::integer(r.numerator());
    }
    return Expr::rational(r.numerator(), r.denominator()).value();
}

[[nodiscard]] auto xsym() -> Expr { return Expr::symbol("x"); }
[[nodiscard]] auto tsym() -> Expr { return Expr::symbol("t"); }

// x^n and t^n.
[[nodiscard]] auto xpow(std::int64_t n) -> Expr { return Expr::power(xsym(), Expr::integer(n)); }
[[nodiscard]] auto tpow(std::int64_t n) -> Expr { return Expr::power(tsym(), Expr::integer(n)); }

// coeff * x^p * t^k as an Expr term.
[[nodiscard]] auto term(const Rational& coeff, std::int64_t p, std::int64_t k) -> Expr {
    std::vector<Expr> f{rc(coeff)};
    if (p != 0) {
        f.push_back(xpow(p));
    }
    if (k != 0) {
        f.push_back(tpow(k));
    }
    return Expr::product(std::move(f));
}

// True iff `e` expands/simplifies to the exact zero expression.
[[nodiscard]] auto is_zero_expr(const Expr& e) -> bool {
    auto r = nimblecas::expand(e);
    return r && r->is_equivalent_to(Expr::integer(0));
}

// True iff a and b are equal as polynomials (their difference expands to 0).
[[nodiscard]] auto equal_expr(const Expr& a, const Expr& b) -> bool {
    return is_zero_expr(Expr::sum({a, Expr::product({Expr::integer(-1), b})}));
}

// Rebuild a RationalPoly in x as an Expr (for comparing against nimblecas.pde output).
[[nodiscard]] auto ratpoly_to_expr(const RationalPoly& p) -> Expr {
    if (p.is_zero()) {
        return Expr::integer(0);
    }
    std::vector<Expr> terms;
    for (std::size_t i = 0; i <= static_cast<std::size_t>(p.degree()); ++i) {
        const Rational c = p.coefficient(i);
        if (c.is_zero()) {
            continue;
        }
        terms.push_back(term(c, static_cast<std::int64_t>(i), 0));
    }
    if (terms.empty()) {
        return Expr::integer(0);
    }
    return Expr::sum(std::move(terms));
}

// Assemble a forward time series c_0..c_order (nimblecas.pde) into Σ_k c_k(x) t^k.
[[nodiscard]] auto forward_to_expr(const std::vector<RationalPoly>& coeffs) -> Expr {
    std::vector<Expr> terms;
    for (std::size_t k = 0; k < coeffs.size(); ++k) {
        Expr ck = ratpoly_to_expr(coeffs[k]);
        terms.push_back(k == 0 ? ck : Expr::product({ck, tpow(static_cast<std::int64_t>(k))}));
    }
    return Expr::sum(std::move(terms));
}

// Apply d/dvar `times` times via nimblecas.diff (each call is a full differentiation pass).
[[nodiscard]] auto diff_n(const Expr& e, std::string_view var, std::size_t times) -> Expr {
    Expr cur = e;
    for (std::size_t i = 0; i < times; ++i) {
        cur = nimblecas::differentiate(cur, var).value();
    }
    return cur;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.pdeham")
        .test("order1_and_order2_terms_match_hand_computation",
              [](TestContext& t) {
                  // u_t = u^2, u(x,0) = x, ħ = -1/2 (pure reaction: diffusivity 0).
                  // By hand: u_1 = -ħ x^2 t, and at HAM order 1 the series is x - ħ x^2 t.
                  // At ħ = -1/2 that is x + (1/2) x^2 t.
                  const std::vector<Rational> f_u2{ri(0), ri(0), ri(1)};  // f(u) = u^2
                  auto o1 = nimblecas::ham_reaction_diffusion(ri(0), f_u2, xsym(), rat(-1, 2), 1,
                                                              "t", "x")
                                .value();
                  Expr exp1 = Expr::sum({xsym(), term(rat(1, 2), 2, 1)});
                  t.expect(equal_expr(o1, exp1), "order-1: x + (1/2) x^2 t");

                  // At HAM order 2 the second term corrects the t^1 coefficient (a hallmark
                  // of HAM: higher orders revise lower t-coefficients away from ħ = -1).
                  // By hand: t^1 -> -2ħ x^2 - ħ^2 x^2, t^2 -> ħ^2 x^3; at ħ = -1/2 that is
                  // t^1 -> (3/4) x^2, t^2 -> (1/4) x^3.
                  auto o2 = nimblecas::ham_reaction_diffusion(ri(0), f_u2, xsym(), rat(-1, 2), 2,
                                                              "t", "x")
                                .value();
                  Expr exp2 = Expr::sum(
                      {xsym(), term(rat(3, 4), 2, 1), term(rat(1, 4), 3, 2)});
                  t.expect(equal_expr(o2, exp2), "order-2: x + (3/4) x^2 t + (1/4) x^3 t^2");

                  // The order-1 and order-2 t^1 coefficients differ (1/2 vs 3/4): general ħ
                  // truncations are NOT nested Taylor partial sums.
                  t.expect(!equal_expr(o1, o2), "HAM orders 1 and 2 differ at ħ = -1/2");
              })
        .test("hbar_minus_one_reduces_to_forward_reaction_diffusion",
              [](TestContext& t) {
                  // Shared example: u_t = u_xx + u^2, u(x,0) = x. ħ = -1 must reproduce the
                  // Cauchy-Kovalevskaya / ADM forward series of nimblecas.pde term for term.
                  const std::size_t order = 3;
                  const std::vector<Rational> f_u2{ri(0), ri(0), ri(1)};
                  auto ham = nimblecas::ham_reaction_diffusion(ri(1), f_u2, xsym(), ri(-1), order,
                                                               "t", "x")
                                 .value();
                  const RationalPoly phi = RationalPoly::monomial(ri(1), 1);  // x
                  auto fwd = nimblecas::reaction_diffusion_quadratic(ri(1), phi, order).value();
                  t.expect(equal_expr(ham, forward_to_expr(fwd)),
                           "HAM(ħ=-1) == forward reaction-diffusion series");

                  // Sanity: at order 2 the closed forward series is x + x^2 t + (1 + x^3) t^2.
                  auto ham2 = nimblecas::ham_reaction_diffusion(ri(1), f_u2, xsym(), ri(-1), 2,
                                                                "t", "x")
                                  .value();
                  Expr exp2 = Expr::sum({xsym(), term(ri(1), 2, 1), term(ri(1), 3, 2),
                                         term(ri(1), 0, 2)});
                  t.expect(equal_expr(ham2, exp2), "order-2 forward value x + x^2 t + (1+x^3) t^2");
              })
        .test("hbar_minus_one_reduces_to_forward_burgers",
              [](TestContext& t) {
                  // Shared example: inviscid Burgers u_t + u u_x = 0, u(x,0) = x (true
                  // solution x/(1+t)). ħ = -1 must match nimblecas.pde::burgers.
                  const std::size_t order = 4;
                  const RationalPoly phi = RationalPoly::monomial(ri(1), 1);  // x
                  auto ham = nimblecas::ham_burgers(ri(0), xsym(), ri(-1), order, "t", "x").value();
                  auto fwd = nimblecas::burgers(ri(0), phi, order).value();
                  t.expect(equal_expr(ham, forward_to_expr(fwd)),
                           "HAM(ħ=-1) == forward inviscid Burgers series");
              })
        .test("hbar_minus_one_residual_is_high_order_in_t",
              [](TestContext& t) {
                  // HAM correctness property at ħ = -1: substituting the truncated series into
                  // u_t - (u_xx + u^2) leaves a residual whose t^0..t^{order-1} coefficients
                  // all vanish (residual = O(t^order)). Differentiation is done by the diff
                  // module; the t^k coefficient is read as (d/dt)^k residual |_{t=0}.
                  const std::size_t order = 3;
                  const std::vector<Rational> f_u2{ri(0), ri(0), ri(1)};
                  auto u = nimblecas::ham_reaction_diffusion(ri(1), f_u2, xsym(), ri(-1), order,
                                                             "t", "x")
                               .value();
                  Expr u_t = nimblecas::differentiate(u, "t").value();
                  Expr u_xx = diff_n(u, "x", 2);
                  Expr u2 = nimblecas::expand(Expr::power(u, Expr::integer(2))).value();
                  Expr rhs = Expr::sum({u_xx, u2});          // u_xx + u^2
                  Expr residual = Expr::sum({u_t, Expr::product({Expr::integer(-1), rhs})});

                  for (std::size_t k = 0; k < order; ++k) {
                      Expr dk = diff_n(residual, "t", k);
                      Expr at0 = nimblecas::substitute(dk, tsym(), Expr::integer(0));
                      t.expect(is_zero_expr(at0),
                               std::format("residual t^{} coefficient vanishes", k));
                  }
              })
        .test("error_paths",
              [](TestContext& t) {
                  const std::vector<Rational> f_u2{ri(0), ri(0), ri(1)};
                  // order == 0 is a domain_error.
                  auto e0 = nimblecas::ham_reaction_diffusion(ri(1), f_u2, xsym(), ri(-1), 0, "t",
                                                              "x");
                  t.expect(!e0 && e0.error() == MathError::domain_error, "order 0 -> domain_error");
                  // t and x the same symbol is a domain_error.
                  auto esame =
                      nimblecas::ham_reaction_diffusion(ri(1), f_u2, xsym(), ri(-1), 2, "x", "x");
                  t.expect(!esame && esame.error() == MathError::domain_error,
                           "tvar == xvar -> domain_error");
                  // An empty operator (no diffusion, convection, or reaction) is not_implemented.
                  HamPde empty{};
                  auto enone = nimblecas::ham_pde_solve(empty, xsym(), ri(-1), 2, "t", "x");
                  t.expect(!enone && enone.error() == MathError::not_implemented,
                           "empty operator -> not_implemented");
              })
        .run();
}
