// Feature/integration tests: symbolic layer (expr, simplify, diff, series, laplace, vectorcalc, latex, polyexpr).
// @author Olumuyiwa Oluwasanmi
//
// These exercise cross-module workflows and exact calculus identities end-to-end,
// rather than re-testing a single module in isolation: differentiate feeds simplify,
// series drives diff, vectorcalc composes diff+simplify, and the Expr<->Polynomial
// bridge round-trips through polynomial arithmetic. Every expression comparison is
// done the way the existing diff/series tests do it — structurally, after routing both
// sides through simplify — and the LaTeX cases pin an exact rendered string.

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.diff;
import nimblecas.series;
import nimblecas.laplace;
import nimblecas.vectorcalc;
import nimblecas.latex;
import nimblecas.polynomial;
import nimblecas.polyexpr;
import nimblecas.testing;

using nimblecas::differentiate;
using nimblecas::Expr;
using nimblecas::from_polynomial;
using nimblecas::laplace_transform;
using nimblecas::Polynomial;
using nimblecas::simplify;
using nimblecas::taylor_coefficients;
using nimblecas::taylor_polynomial;
using nimblecas::to_latex;
using nimblecas::to_polynomial;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// A unary function application, mirroring the helper the diff tests use.
auto fn(std::string name, const Expr& arg) -> Expr {
    return Expr::apply(std::move(name), {arg});
}

// Compare two expressions for mathematical equality the house way: route BOTH sides
// through simplify (giving each a canonical tree) and test structural equality. Any
// simplify failure records a loud failure so the check cannot silently pass.
auto equiv(TestContext& t, const Expr& a, const Expr& b, std::string_view what) -> void {
    auto sa = simplify(a);
    auto sb = simplify(b);
    if (!sa.has_value() || !sb.has_value()) {
        t.expect(false, std::format("{}: unexpected simplify error", what));
        return;
    }
    t.expect(sa->is_equivalent_to(*sb),
             std::format("{}: got {} vs {}", what, sa->to_string(), sb->to_string()));
}

// Like equiv but the actual side is a Result<Expr> produced by a transform (diff,
// laplace, ...); asserts success first.
template <typename R>
auto equiv_result(TestContext& t, const R& actual, const Expr& expected, std::string_view what)
    -> void {
    if (!actual.has_value()) {
        t.expect(false, std::format("{}: producer returned an error", what));
        return;
    }
    equiv(t, actual.value(), expected, what);
}

// Numeric probe: substitute var := value and simplify to a concrete constant. Used to
// verify identities the simplifier will not put over a common denominator (e.g. the
// quotient rule), following the eval-at-points technique from series_tests.
auto eval_at(TestContext& t, const Expr& e, std::string_view var, std::int64_t value,
             std::string_view what) -> Expr {
    const Expr sub = nimblecas::substitute(e, Expr::symbol(std::string(var)), Expr::integer(value));
    auto s = simplify(sub);
    if (!s.has_value()) {
        t.expect(false, std::format("{}: simplify error", what));
        return Expr::integer(0);
    }
    return *s;
}

}  // namespace

auto main() -> int {
    const auto x = Expr::symbol("x");
    const auto y = Expr::symbol("y");
    const auto z = Expr::symbol("z");
    const auto a = Expr::symbol("a");
    const auto s = Expr::symbol("s");
    const auto t_var = Expr::symbol("t");
    const auto zero = Expr::integer(0);
    const auto one = Expr::integer(1);
    const auto two = Expr::integer(2);
    const auto three = Expr::integer(3);
    const auto neg_one = Expr::integer(-1);

    const std::vector<std::string> xy{"x", "y"};
    const std::vector<std::string> xyz{"x", "y", "z"};

    return TestSuite("feature.symbolic")
        // -------------------------------------------------------------------
        // diff ∘ simplify pipeline: derivative + simplify must land on the exact
        // hand-computed rule outcome.
        // -------------------------------------------------------------------
        .test("diff_simplify_power_rule",
              [&](TestContext& t) {
                  // d/dx x^2 = 2x, d/dx x^3 = 3x^2.
                  equiv_result(t, differentiate(x.pow(two), "x"), two.mul(x), "d/dx x^2 = 2x");
                  equiv_result(t, differentiate(x.pow(three), "x"), three.mul(x.pow(two)),
                               "d/dx x^3 = 3x^2");
              })
        .test("diff_simplify_product_rule",
              [&](TestContext& t) {
                  // d/dx [x*(x+1)] = (x+1) + x = 2x + 1.
                  auto u = Expr::product({x, Expr::sum({x, one})});
                  auto expected = Expr::sum({two.mul(x), one});
                  equiv_result(t, differentiate(u, "x"), expected, "d/dx x(x+1) = 2x+1");
              })
        .test("diff_simplify_quotient_rule",
              [&](TestContext& t) {
                  // d/dx [x/(x+1)] = 1/(x+1)^2. simplify will not combine the
                  // derivative over a common denominator, so verify functionally at
                  // several points instead of structurally.
                  auto quotient = Expr::product({x, Expr::power(Expr::sum({x, one}), neg_one)});
                  auto d = differentiate(quotient, "x");
                  t.expect(d.has_value(), "d/dx x/(x+1) succeeds");
                  if (!d) {
                      return;
                  }
                  auto expected = Expr::power(Expr::sum({x, one}), Expr::integer(-2));
                  for (std::int64_t v : {0, 1, 2, 4}) {
                      auto got = eval_at(t, *d, "x", v, std::format("d at x={}", v));
                      auto want = eval_at(t, expected, "x", v, std::format("1/(x+1)^2 at x={}", v));
                      t.expect(got.is_equivalent_to(want),
                               std::format("quotient rule at x={}: {} == {}", v, got.to_string(),
                                           want.to_string()));
                  }
              })
        .test("diff_simplify_chain_rule",
              [&](TestContext& t) {
                  // d/dx exp(x^3) = 3x^2 exp(x^3).
                  auto u = fn("exp", x.pow(three));
                  auto expected = Expr::product({three, x.pow(two), fn("exp", x.pow(three))});
                  equiv_result(t, differentiate(u, "x"), expected,
                               "d/dx exp(x^3) = 3x^2 exp(x^3)");
                  // d/dx sin(x^2) = 2x cos(x^2).
                  auto v = fn("sin", x.pow(two));
                  auto expected2 = Expr::product({two, x, fn("cos", x.pow(two))});
                  equiv_result(t, differentiate(v, "x"), expected2, "d/dx sin(x^2) = 2x cos(x^2)");
              })
        // -------------------------------------------------------------------
        // simplify invariants.
        // -------------------------------------------------------------------
        .test("simplify_algebraic_invariants",
              [&](TestContext& t) {
                  equiv(t, Expr::sum({x, x}), two.mul(x), "x + x = 2x");
                  equiv(t, Expr::product({x, one}), x, "x * 1 = x");
                  equiv(t, Expr::sum({x, zero}), x, "x + 0 = x");
                  equiv(t, Expr::product({zero, x}), zero, "0 * x = 0");
                  equiv(t, Expr::power(x, zero), one, "x^0 = 1");
                  equiv(t, Expr::power(x, one), x, "x^1 = x");
              })
        .test("simplify_idempotent_and_nested_folding",
              [&](TestContext& t) {
                  // Idempotence: simplify(simplify(e)) == simplify(e) on a mixed tree.
                  auto e = Expr::sum({two.mul(x), three.mul(x), x.pow(two), one, two});
                  auto s1 = simplify(e);
                  t.expect(s1.has_value(), "first simplify succeeds");
                  if (s1) {
                      auto s2 = simplify(*s1);
                      t.expect(s2.has_value() && s2->is_equivalent_to(*s1),
                               std::format("simplify is idempotent: {}", s1->to_string()));
                  }
                  // Nested constant folding: (2+3)*x -> 5x, (1+1)^3 -> 8.
                  equiv(t, Expr::product({Expr::sum({two, three}), x}), Expr::integer(5).mul(x),
                        "(2+3)*x = 5x");
                  equiv(t, Expr::power(Expr::sum({one, one}), three), Expr::integer(8),
                        "(1+1)^3 = 8");
              })
        // -------------------------------------------------------------------
        // Taylor series (uses polynomials so every coefficient is exact) crossed
        // with the diff engine.
        // -------------------------------------------------------------------
        .test("taylor_polynomial_coeffs_and_reconstruct",
              [&](TestContext& t) {
                  // f = x^3 + 2x^2 + x about 0: coefficients are the monomial coeffs
                  // [0, 1, 2, 1].
                  auto f = Expr::sum({x.pow(three), two.mul(x.pow(two)), x});
                  auto coeffs = taylor_coefficients(f, "x", zero, 3);
                  t.expect(coeffs.has_value(), "taylor_coefficients succeeds");
                  if (coeffs) {
                      t.expect_eq(coeffs->size(), std::size_t{4}, "4 coefficients");
                      t.expect((*coeffs)[0].is_equivalent_to(zero), "c_0 = 0");
                      t.expect((*coeffs)[1].is_equivalent_to(one), "c_1 = 1");
                      t.expect((*coeffs)[2].is_equivalent_to(two), "c_2 = 2");
                      t.expect((*coeffs)[3].is_equivalent_to(one), "c_3 = 1");
                  }
                  // The truncated polynomial about 0 reconstructs f exactly.
                  equiv_result(t, taylor_polynomial(f, "x", zero, 3), f,
                               "taylor_polynomial reconstructs f");
              })
        .test("taylor_series_of_derivative",
              [&](TestContext& t) {
                  // For f, the series of f' has coefficient d_k = (k+1) c_{k+1}: the
                  // series and the diff engine must agree. f = x^3 + 2x^2 + x.
                  auto f = Expr::sum({x.pow(three), two.mul(x.pow(two)), x});
                  auto fprime = differentiate(f, "x");  // = 3x^2 + 4x + 1
                  t.expect(fprime.has_value(), "differentiate f succeeds");
                  if (!fprime) {
                      return;
                  }
                  auto cf = taylor_coefficients(f, "x", zero, 3);
                  auto cfp = taylor_coefficients(*fprime, "x", zero, 2);
                  t.expect(cf.has_value() && cfp.has_value(), "both coefficient lists succeed");
                  if (!cf || !cfp) {
                      return;
                  }
                  // Direct known values for f' = 3x^2 + 4x + 1: [1, 4, 3].
                  t.expect((*cfp)[0].is_equivalent_to(one), "d_0 = 1");
                  t.expect((*cfp)[1].is_equivalent_to(Expr::integer(4)), "d_1 = 4");
                  t.expect((*cfp)[2].is_equivalent_to(three), "d_2 = 3");
                  // Relationship d_k = (k+1) c_{k+1}.
                  for (std::int64_t k = 0; k <= 2; ++k) {
                      auto rhs = Expr::integer(k + 1).mul((*cf)[static_cast<std::size_t>(k + 1)]);
                      equiv(t, (*cfp)[static_cast<std::size_t>(k)], rhs,
                            std::format("d_{} = {} * c_{}", k, k + 1, k + 1));
                  }
              })
        // -------------------------------------------------------------------
        // Laplace transform: table cases + linearity.
        // -------------------------------------------------------------------
        .test("laplace_table_cases",
              [&](TestContext& t) {
                  // L{1} = 1/s.
                  equiv_result(t, laplace_transform(one, "t", "s"), Expr::power(s, neg_one),
                               "L{1} = 1/s");
                  // L{t} = 1/s^2.
                  equiv_result(t, laplace_transform(t_var, "t", "s"),
                               Expr::power(s, Expr::integer(-2)), "L{t} = 1/s^2");
                  // L{t^2} = 2/s^3.
                  equiv_result(t, laplace_transform(t_var.pow(two), "t", "s"),
                               two.mul(Expr::power(s, Expr::integer(-3))), "L{t^2} = 2/s^3");
                  // L{e^{a t}} = 1/(s - a).
                  auto exp_at = fn("exp", Expr::product({a, t_var}));
                  auto expected = Expr::power(Expr::sum({s, Expr::product({neg_one, a})}), neg_one);
                  equiv_result(t, laplace_transform(exp_at, "t", "s"), expected,
                               "L{e^{a t}} = 1/(s-a)");
              })
        .test("laplace_linearity",
              [&](TestContext& t) {
                  // L{3 t^2 + 5 t} == 3 L{t^2} + 5 L{t}.
                  auto combined = Expr::sum({three.mul(t_var.pow(two)), Expr::integer(5).mul(t_var)});
                  auto lhs = laplace_transform(combined, "t", "s");
                  auto lf = laplace_transform(t_var.pow(two), "t", "s");
                  auto lg = laplace_transform(t_var, "t", "s");
                  t.expect(lhs.has_value() && lf.has_value() && lg.has_value(),
                           "all three transforms succeed");
                  if (!lhs || !lf || !lg) {
                      return;
                  }
                  auto rhs = Expr::sum(
                      {three.mul(*lf), Expr::integer(5).mul(*lg)});
                  equiv(t, *lhs, rhs, "L{3t^2 + 5t} = 3 L{t^2} + 5 L{t}");
              })
        // -------------------------------------------------------------------
        // Vector calculus identities that must collapse EXACTLY for concrete
        // fields (Clairaut cancellation via automatic simplification).
        // -------------------------------------------------------------------
        .test("vectorcalc_curl_of_grad_is_zero",
              [&](TestContext& t) {
                  // f = x^2 y + y^2 z + z^2 x. curl(grad f) = 0 componentwise.
                  auto f = Expr::sum({Expr::product({x.pow(two), y}),
                                      Expr::product({y.pow(two), z}),
                                      Expr::product({z.pow(two), x})});
                  auto grad = nimblecas::gradient(f, xyz);
                  t.expect(grad.has_value(), "gradient succeeds");
                  if (!grad) {
                      return;
                  }
                  auto c = nimblecas::curl(*grad, xyz);
                  t.expect(c.has_value() && c->size() == 3, "curl(grad f) has 3 components");
                  if (c) {
                      for (std::size_t i = 0; i < c->size(); ++i) {
                          t.expect((*c)[i].is_equivalent_to(zero),
                                   std::format("curl(grad f)[{}] = 0", i));
                      }
                  }
              })
        .test("vectorcalc_div_of_curl_is_zero",
              [&](TestContext& t) {
                  // F = (x^2 y, y^2 z, z^2 x). div(curl F) = 0.
                  std::vector<Expr> field{Expr::product({x.pow(two), y}),
                                          Expr::product({y.pow(two), z}),
                                          Expr::product({z.pow(two), x})};
                  auto c = nimblecas::curl(field, xyz);
                  t.expect(c.has_value(), "curl F succeeds");
                  if (!c) {
                      return;
                  }
                  auto d = nimblecas::divergence(*c, xyz);
                  t.expect(d.has_value() && d->is_equivalent_to(zero),
                           std::format("div(curl F) = 0 (got {})",
                                       d.has_value() ? d->to_string() : "error"));
              })
        .test("vectorcalc_laplacian_equals_div_grad",
              [&](TestContext& t) {
                  // laplacian(f) == div(grad f) == 2x + 2y + 2z for f = x^2y + y^2z + z^2x.
                  auto f = Expr::sum({Expr::product({x.pow(two), y}),
                                      Expr::product({y.pow(two), z}),
                                      Expr::product({z.pow(two), x})});
                  auto lap = nimblecas::laplacian(f, xyz);
                  auto grad = nimblecas::gradient(f, xyz);
                  t.expect(lap.has_value() && grad.has_value(), "laplacian and gradient succeed");
                  if (!lap || !grad) {
                      return;
                  }
                  auto div_grad = nimblecas::divergence(*grad, xyz);
                  t.expect(div_grad.has_value(), "div(grad f) succeeds");
                  if (div_grad) {
                      equiv(t, *lap, *div_grad, "laplacian(f) == div(grad f)");
                  }
                  auto expected = Expr::sum({two.mul(x), two.mul(y), two.mul(z)});
                  equiv(t, *lap, expected, "laplacian(f) == 2x + 2y + 2z");
              })
        .test("vectorcalc_gradient_hessian_jacobian_values",
              [&](TestContext& t) {
                  // grad(x^2 y + y^3) = (2xy, x^2 + 3y^2).
                  auto g = Expr::sum({Expr::product({x.pow(two), y}), y.pow(three)});
                  auto grad = nimblecas::gradient(g, xy);
                  t.expect(grad.has_value() && grad->size() == 2, "gradient has 2 components");
                  if (grad) {
                      equiv(t, (*grad)[0], Expr::product({two, x, y}), "df/dx = 2xy");
                      equiv(t, (*grad)[1], Expr::sum({x.pow(two), three.mul(y.pow(two))}),
                            "df/dy = x^2 + 3y^2");
                  }
                  // Hessian(x^2 y) = [[2y, 2x], [2x, 0]] (symmetric).
                  auto h = nimblecas::hessian(Expr::product({x.pow(two), y}), xy);
                  t.expect(h.has_value(), "hessian succeeds");
                  if (h) {
                      equiv(t, (*h)[0][0], two.mul(y), "H[0][0] = 2y");
                      equiv(t, (*h)[0][1], two.mul(x), "H[0][1] = 2x");
                      equiv(t, (*h)[1][0], two.mul(x), "H[1][0] = 2x");
                      t.expect((*h)[1][1].is_equivalent_to(zero), "H[1][1] = 0");
                      t.expect((*h)[0][1].is_equivalent_to((*h)[1][0]), "Hessian is symmetric");
                  }
                  // Jacobian of (x y, x + y) = [[y, x], [1, 1]].
                  std::vector<Expr> field{Expr::product({x, y}), Expr::sum({x, y})};
                  auto j = nimblecas::jacobian(field, xy);
                  t.expect(j.has_value(), "jacobian succeeds");
                  if (j) {
                      t.expect((*j)[0][0].is_equivalent_to(y), "J[0][0] = y");
                      t.expect((*j)[0][1].is_equivalent_to(x), "J[0][1] = x");
                      t.expect((*j)[1][0].is_equivalent_to(one), "J[1][0] = 1");
                      t.expect((*j)[1][1].is_equivalent_to(one), "J[1][1] = 1");
                  }
              })
        // -------------------------------------------------------------------
        // LaTeX exporter: exact rendered strings (built unsimplified so operand
        // order is fully controlled).
        // -------------------------------------------------------------------
        .test("latex_exact_rendering",
              [&](TestContext& t) {
                  t.expect_eq(to_latex(Expr::power(x, two)), std::string{"x^{2}"}, "x^2 -> x^{2}");
                  t.expect_eq(to_latex(Expr::rational(1, 2).value()), std::string{"\\frac{1}{2}"},
                              "1/2 -> \\frac{1}{2}");
                  t.expect_eq(to_latex(Expr::product({x, Expr::power(y, neg_one)})),
                              std::string{"\\frac{x}{y}"}, "x*y^-1 -> \\frac{x}{y}");
                  t.expect_eq(to_latex(fn("sin", x)), std::string{"\\sin\\left(x\\right)"},
                              "sin(x) -> \\sin\\left(x\\right)");
                  t.expect_eq(to_latex(Expr::sum({x.pow(two), two.mul(x)})),
                              std::string{"x^{2} + 2 x"}, "x^2 + 2x -> x^{2} + 2 x");
                  t.expect_eq(to_latex(Expr::sum({x, Expr::product({neg_one, y})})),
                              std::string{"x - y"}, "x + (-1)y -> x - y");
              })
        // -------------------------------------------------------------------
        // polyexpr bridge: Expr <-> Polynomial round-trips and derivative
        // consistency across the two representations.
        // -------------------------------------------------------------------
        .test("polyexpr_roundtrip_and_derivative",
              [&](TestContext& t) {
                  // Polynomial -> Expr -> Polynomial is the identity.
                  const Polynomial p0{std::vector<std::int64_t>{1, 2, 3}};  // 1 + 2x + 3x^2
                  const Expr e = from_polynomial(p0, "x");
                  auto p1 = to_polynomial(e, "x");
                  t.expect(p1.has_value() && p1->is_equal(p0),
                           "Polynomial -> Expr -> Polynomial round-trips");

                  // Expr -> Polynomial reads the monomial coefficients of x^2 + 2x + 1.
                  auto u = Expr::sum({x.pow(two), two.mul(x), one});
                  auto pu = to_polynomial(u, "x");
                  t.expect(pu.has_value(), "to_polynomial(x^2+2x+1) succeeds");
                  if (pu) {
                      t.expect_eq(pu->degree(), std::int64_t{2}, "degree 2");
                      t.expect_eq(pu->coefficient(0), std::int64_t{1}, "coeff x^0 = 1");
                      t.expect_eq(pu->coefficient(1), std::int64_t{2}, "coeff x^1 = 2");
                      t.expect_eq(pu->coefficient(2), std::int64_t{1}, "coeff x^2 = 1");

                      // Differentiating via diff and via the polynomial derivative agree.
                      auto dp = pu->derivative();
                      auto du = differentiate(u, "x");
                      t.expect(dp.has_value() && du.has_value(),
                               "polynomial and symbolic derivatives succeed");
                      if (dp && du) {
                          equiv(t, from_polynomial(*dp, "x"), *du,
                                "d/dx via polynomial == d/dx via diff");
                      }
                  }
              })
        .run();
}
