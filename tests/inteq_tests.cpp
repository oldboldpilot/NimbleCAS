// Tests for nimblecas.inteq: integral-equation solvers (Fredholm / Volterra; linear,
// nonlinear, and the ADM / HPM / HAM decomposition-homotopy family).
// @author Olumuyiwa Oluwasanmi
//
// Every exact case is hand-verified in the comments. The decomposition solvers are
// cross-checked against each other (ADM == HPM, HAM(ħ=-1) == ADM) and against the
// Neumann/Picard series for the linear case (ADM(identity) == Neumann).

import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.ratpoly;
import nimblecas.inteq;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.laplace;
import nimblecas.testing;

using nimblecas::adm_solve;
using nimblecas::adomian_polynomials;
using nimblecas::Expr;
using nimblecas::fredholm1_separable;
using nimblecas::fredholm2_neumann;
using nimblecas::fredholm2_separable;
using nimblecas::ham_solve;
using nimblecas::hammerstein_picard;
using nimblecas::hpm_solve;
using nimblecas::IntegralOperator;
using nimblecas::laplace_transform;
using nimblecas::MathError;
using nimblecas::Nonlinearity;
using nimblecas::Polynomial;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::SeparableKernel;
using nimblecas::simplify;
using nimblecas::volterra2_picard;
using nimblecas::volterra_convolution_laplace;
using nimblecas::volterra_nonlinear_picard;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Integer-coefficient polynomial (each coefficient over denominator 1).
[[nodiscard]] auto ipoly(std::vector<std::int64_t> c) -> RationalPoly {
    return RationalPoly::from_polynomial(Polynomial{std::move(c)});
}
// A rational.
[[nodiscard]] auto R(std::int64_t n, std::int64_t d = 1) -> Rational {
    return Rational::make(n, d).value();
}
// A polynomial from explicit rational coefficients (ascending).
[[nodiscard]] auto rp(std::vector<Rational> c) -> RationalPoly {
    return RationalPoly::from_coeffs(std::move(c));
}
// The single-term separable kernel K(x,t) = g(x) h(t).
[[nodiscard]] auto kernel1(RationalPoly g, RationalPoly h) -> SeparableKernel {
    return SeparableKernel{.g = {std::move(g)}, .h = {std::move(h)}};
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.inteq")
        .test("fredholm2_separable_exact",
              [](TestContext& t) {
                  // K(x,t) = x t on [0,1], f = x, lambda = 1. phi = f + lambda x c with
                  // c = ∫_0^1 t phi = 1/2, so phi = (3/2) x exactly (hand-verified).
                  const auto k = kernel1(ipoly({0, 1}), ipoly({0, 1}));
                  auto sol = fredholm2_separable(ipoly({0, 1}), k, R(1), R(0), R(1)).value();
                  t.expect(sol.phi.is_equal(rp({R(0), R(3, 2)})), "phi == (3/2) x");
                  t.expect(sol.coefficients.size() == 1, "one separable coefficient");
                  t.expect(sol.coefficients[0] == R(1, 2), "moment c == 1/2");
              })
        .test("fredholm2_neumann_matches",
              [](TestContext& t) {
                  // Neumann terms are x/3^n; the order-2 partial sum is x(1 + 1/3 + 1/9)
                  // = (13/9) x, order-3 is (40/27) x, converging to (3/2) x.
                  const auto k = kernel1(ipoly({0, 1}), ipoly({0, 1}));
                  const auto f = ipoly({0, 1});
                  auto n2 = fredholm2_neumann(f, k, R(1), R(0), R(1), 2).value();
                  auto n3 = fredholm2_neumann(f, k, R(1), R(0), R(1), 3).value();
                  t.expect(n2.is_equal(rp({R(0), R(13, 9)})), "order-2 == (13/9) x");
                  t.expect(n3.is_equal(rp({R(0), R(40, 27)})), "order-3 == (40/27) x");
                  // ADM with the identity nonlinearity reproduces the Neumann series exactly.
                  auto op = IntegralOperator::fredholm(k, R(1), R(0), R(1)).value();
                  auto adm = adm_solve(op, f, Nonlinearity::identity(), 2).value();
                  t.expect(adm.is_equal(n2), "ADM(identity) == Neumann order 2");
              })
        .test("fredholm1_separable_illposed",
              [](TestContext& t) {
                  // f = ∫ x t phi dt = x (∫ t phi) => f = c x. For f = 2x the recoverable
                  // moment is c = 2 (phi itself is NON-UNIQUE). f = x^2 is not in span{x}.
                  const auto k = kernel1(ipoly({0, 1}), ipoly({0, 1}));
                  auto sol = fredholm1_separable(ipoly({0, 2}), k, R(0), R(1)).value();
                  t.expect(sol.moments.size() == 1 && sol.moments[0] == R(2), "moment == 2");
                  t.expect(fredholm1_separable(ipoly({0, 0, 1}), k, R(0), R(1)).error() ==
                               MathError::domain_error,
                           "f not in span{g_i} => no solution");
              })
        .test("volterra2_picard_exponential",
              [](TestContext& t) {
                  // Kernel 1, f = 1, lambda = 1, a = 0: phi_n = Σ_{k<=n} x^k/k! (e^x series).
                  const auto k = kernel1(ipoly({1}), ipoly({1}));
                  auto phi = volterra2_picard(ipoly({1}), k, R(1), R(0), 4).value();
                  t.expect(phi.coefficient(0) == R(1), "1");
                  t.expect(phi.coefficient(1) == R(1), "x");
                  t.expect(phi.coefficient(2) == R(1, 2), "x^2/2");
                  t.expect(phi.coefficient(3) == R(1, 6), "x^3/6");
                  t.expect(phi.coefficient(4) == R(1, 24), "x^4/24");
                  // ADM(identity) on the Volterra operator reproduces the same partial sum.
                  auto op = IntegralOperator::volterra(k, R(1), R(0)).value();
                  auto adm = adm_solve(op, ipoly({1}), Nonlinearity::identity(), 4).value();
                  t.expect(adm.is_equal(phi), "ADM(identity) == Volterra Picard sum");
              })
        .test("volterra_convolution_laplace",
              [](TestContext& t) {
                  // K(x,t) = 1 (k = 1), f = 1, lambda = 2. Φ(s) = F/(1 - 2 k̂) with
                  // F = k̂ = 1/s, i.e. Φ = (1/s)/(1 - 2/s) = 1/(s - 2) (inverts to e^{2x}).
                  const Expr one = Expr::integer(1);
                  auto phi = volterra_convolution_laplace(one, one, R(2), "x", "s").value();
                  // Verify the transform-domain identity Φ·(1 - 2 k̂) == F exactly.
                  const Expr F = laplace_transform(one, "x", "s").value();
                  const Expr K = laplace_transform(one, "x", "s").value();
                  const Expr denom =
                      simplify(Expr::sum({Expr::integer(1),
                                          Expr::product({Expr::integer(-1), Expr::integer(2), K})}))
                          .value();
                  const Expr check = simplify(Expr::product({phi, denom})).value();
                  t.expect(check.is_equivalent_to(F), "Phi*(1 - lambda k_hat) == F");
                  // A free term off the Laplace table is not_implemented (honest boundary).
                  const Expr tanx = Expr::apply("tan", {Expr::symbol("x")});
                  t.expect(volterra_convolution_laplace(tanx, one, R(1), "x", "s").error() ==
                               MathError::not_implemented,
                           "off-table free term => not_implemented");
              })
        .test("nonlinear_volterra_picard",
              [](TestContext& t) {
                  // f = x, kernel 1, lambda = 1, a = 0, N(phi) = phi^2.
                  // psi_1 = x + ∫_0^x t^2 dt = x + x^3/3 (hand-verified).
                  const auto k = kernel1(ipoly({1}), ipoly({1}));
                  const auto f = ipoly({0, 1});
                  const auto sq = Nonlinearity::power(2);
                  auto psi1 = volterra_nonlinear_picard(f, k, R(1), R(0), sq, 1).value();
                  t.expect(psi1.is_equal(rp({R(0), R(1), R(0), R(1, 3)})), "psi_1 == x + x^3/3");
              })
        .test("adomian_polynomials_and_adm",
              [](TestContext& t) {
                  // Components phi_0 = x, phi_1 = x^3/3 with N = phi^2:
                  //   A_0 = phi_0^2 = x^2,  A_1 = 2 phi_0 phi_1 = 2 x^4 / 3 (hand-verified).
                  const auto sq = Nonlinearity::power(2);
                  auto polys = adomian_polynomials(sq, {ipoly({0, 1}), rp({R(0), R(0), R(0), R(1, 3)})})
                                   .value();
                  t.expect(polys.size() == 2, "A_0 and A_1");
                  t.expect(polys[0].is_equal(rp({R(0), R(0), R(1)})), "A_0 == x^2");
                  t.expect(polys[1].is_equal(rp({R(0), R(0), R(0), R(0), R(2, 3)})),
                           "A_1 == (2/3) x^4");
                  // ADM order 2 for the nonlinear Volterra equation (f = x, N = phi^2):
                  //   x + x^3/3 + (2/15) x^5.
                  const auto k = kernel1(ipoly({1}), ipoly({1}));
                  auto op = IntegralOperator::volterra(k, R(1), R(0)).value();
                  const auto f = ipoly({0, 1});
                  auto adm = adm_solve(op, f, sq, 2).value();
                  t.expect(adm.is_equal(rp({R(0), R(1), R(0), R(1, 3), R(0), R(2, 15)})),
                           "ADM order 2 == x + x^3/3 + (2/15) x^5");
              })
        .test("adm_hpm_ham_equivalence",
              [](TestContext& t) {
                  // On the same nonlinear Volterra equation: ADM == HPM, and HAM(ħ=-1) == ADM.
                  const auto k = kernel1(ipoly({1}), ipoly({1}));
                  auto op = IntegralOperator::volterra(k, R(1), R(0)).value();
                  const auto f = ipoly({0, 1});
                  const auto sq = Nonlinearity::power(2);
                  auto adm = adm_solve(op, f, sq, 3).value();
                  auto hpm = hpm_solve(op, f, sq, 3).value();
                  auto ham = ham_solve(op, f, sq, R(-1), 3).value();
                  t.expect(adm.is_equal(hpm), "ADM == HPM");
                  t.expect(ham.is_equal(adm), "HAM(hbar=-1) == ADM");
                  // A different hbar yields a different (still exactly rational) member.
                  auto ham2 = ham_solve(op, f, sq, R(-1, 2), 3).value();
                  t.expect(!ham2.is_equal(adm), "HAM(hbar=-1/2) differs from ADM");
              })
        .test("hammerstein_picard_exact",
              [](TestContext& t) {
                  // Fredholm nonlinear: K = 1 on [0,1], f = 0, lambda = 1, N = phi^2.
                  //   psi_1 = ∫_0^1 f^2 dt = 0; psi_2 = ∫_0^1 (psi_1)^2 = 0 (fixed point 0).
                  // Use f = 1 instead: psi_1 = 1 + ∫_0^1 1 dt = 2 (constant iterate).
                  const auto k = kernel1(ipoly({1}), ipoly({1}));
                  const auto sq = Nonlinearity::power(2);
                  auto psi1 = hammerstein_picard(ipoly({1}), k, R(1), R(0), R(1), sq, 1).value();
                  t.expect(psi1.is_equal(ipoly({2})), "psi_1 == 2");
                  // psi_2 = 1 + ∫_0^1 (2)^2 dt = 1 + 4 = 5.
                  auto psi2 = hammerstein_picard(ipoly({1}), k, R(1), R(0), R(1), sq, 2).value();
                  t.expect(psi2.is_equal(ipoly({5})), "psi_2 == 5");
              })
        .test("error_paths",
              [](TestContext& t) {
                  // Ragged kernel (|g| != |h|) => domain_error.
                  const SeparableKernel ragged{.g = {ipoly({0, 1})}, .h = {}};
                  t.expect(fredholm2_separable(ipoly({0, 1}), ragged, R(1), R(0), R(1)).error() ==
                               MathError::domain_error,
                           "ragged kernel => domain_error");
                  // Singular system: with K = x t on [0,1], M = 1/3, so lambda = 3 makes
                  // (I - lambda M) = 0 singular => no unique solution.
                  const auto k = kernel1(ipoly({0, 1}), ipoly({0, 1}));
                  t.expect(fredholm2_separable(ipoly({0, 1}), k, R(3), R(0), R(1)).error() ==
                               MathError::domain_error,
                           "lambda an eigenvalue => singular => domain_error");
              })
        .run();
}
