// Cross-method tests for nimblecas.inteq: take one linear and one nonlinear integral equation
// and cross-validate the exact finite linear-system solve, Neumann/Picard, and the ADM/HPM/HAM
// decomposition-homotopy family against each other and against a hand-manufactured exact answer.
// @author Olumuyiwa Oluwasanmi
//
// This mirrors tests/pde_crossmethod_tests.cpp's "one equation, many methods" structure for
// integral equations, where (unlike nimblecas.pde) the full ADM/HPM/HAM family — INCLUDING a
// real ħ convergence-control parameter — already exists (see inteq.cppm's own header).
//
//   * "FEM-analogue" leg: fredholm2_separable reduces a linear Fredholm-2 equation to a finite
//     exact linear system (the same "reduce to a finite linear algebra problem, solve exactly"
//     idea as Galerkin FEM), verified against a hand-manufactured exact solution.
//   * Linear Volterra leg: phi = 1 + integral(phi) has exact solution e^x; Neumann, Picard,
//     ADM, HPM, and HAM(hbar=-1) must all return the IDENTICAL truncated Taylor series (hand-
//     verified coefficients 1/n!).
//   * Nonlinear Volterra leg: a manufactured quadratic Hammerstein/Volterra equation whose
//     EXACT solution is phi(x) = x. ADM == HPM (bit-identical); HAM(hbar=-1) == ADM; a
//     different hbar gives a genuinely different (still exact) family member. Because a
//     genuine nonlinearity's ADM/HPM/HAM series does not terminate (documented honesty
//     boundary throughout this engine), this leg checks CONVERGENCE toward the exact answer
//     as the truncation order grows, not exact equality at a fixed finite order — and cross-
//     checks the independently-implemented Picard iteration converges to the same target.

import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.ratpoly;
import nimblecas.inteq;
import nimblecas.testing;

using nimblecas::adm_solve;
using nimblecas::fredholm2_separable;
using nimblecas::ham_solve;
using nimblecas::hpm_solve;
using nimblecas::IntegralOperator;
using nimblecas::MathError;
using nimblecas::Nonlinearity;
using nimblecas::Polynomial;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::SeparableKernel;
using nimblecas::volterra2_picard;
using nimblecas::volterra_nonlinear_picard;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto ipoly(std::vector<std::int64_t> c) -> RationalPoly {
    return RationalPoly::from_polynomial(Polynomial{std::move(c)});
}

[[nodiscard]] auto R(std::int64_t n, std::int64_t d = 1) -> Rational {
    return Rational::make(n, d).value();
}

[[nodiscard]] auto rp(std::vector<Rational> c) -> RationalPoly {
    return RationalPoly::from_coeffs(std::move(c));
}

[[nodiscard]] auto kernel1(RationalPoly g, RationalPoly h) -> SeparableKernel {
    return SeparableKernel{.g = {std::move(g)}, .h = {std::move(h)}};
}

// Horner evaluation of a RationalPoly at a Rational point (the test file's own copy of the
// small helper every solver module keeps privately; there is no public RationalPoly::at()).
[[nodiscard]] auto eval_at(const RationalPoly& p, const Rational& x) -> Rational {
    Rational acc;  // 0
    const auto co = p.coefficients();
    for (std::size_t i = co.size(); i-- > 0;) {
        acc = acc.multiply(x).value().add(co[i]).value();
    }
    return acc;
}

// |a - b| as a double, for a purely qualitative "is this closer" comparison in the
// convergence check below (the underlying values themselves remain exact Rational).
[[nodiscard]] auto abs_diff(const Rational& a, const Rational& b) -> double {
    auto d = a.subtract(b).value();
    const double v = static_cast<double>(d.numerator()) / static_cast<double>(d.denominator());
    return v < 0.0 ? -v : v;
}

}  // namespace

auto main() -> int {
    return TestSuite("inteq_crossmethod")
        .test("fredholm_fem_analogue_exact",
              [](TestContext& t) {
                  // K(x,t) = x t, lambda = 1/2. Manufactured so phi(x) = x is exact:
                  // c = integral_0^1 t*t dt = 1/3 (a property of phi = x alone); then
                  // f = phi - lambda*c*g = x - (1/2)(1/3)x = (5/6) x (hand-verified).
                  // Cross-check via the linear system: M = 1/3, d = integral h*f = (5/6)(1/3)
                  // = 5/18; (1 - (1/2)(1/3)) c = 5/18 => (5/6) c = 5/18 => c = 1/3. phi =
                  // f + (1/2)(1/3) x = (5/6)x + (1/6)x = x.
                  const auto f = rp({R(0), R(5, 6)});
                  const auto k = kernel1(ipoly({0, 1}), ipoly({0, 1}));
                  auto sol = fredholm2_separable(f, k, R(1, 2), R(0), R(1));
                  t.expect(sol.has_value(), "fredholm2_separable succeeds");
                  if (!sol) {
                      return;
                  }
                  t.expect(sol->phi.is_equal(ipoly({0, 1})), "phi == x exactly (manufactured)");
                  t.expect(sol->coefficients.size() == 1 && sol->coefficients[0] == R(1, 3),
                           "moment c == 1/3");
              })
        .test("linear_volterra_all_methods_agree_exp",
              [](TestContext& t) {
                  // phi(x) = 1 + integral_0^x phi(t) dt  <=>  phi' = phi, phi(0) = 1  =>
                  // phi = e^x. Neumann/Picard/ADM/HPM/HAM(hbar=-1) must all return the exact
                  // Taylor coefficients 1/n! (the same construction as the ODE "adm_exponential"
                  // test in perturbation_tests.cpp, now via the integral-equation route).
                  const auto f = RationalPoly::constant(R(1));
                  const auto k = kernel1(RationalPoly::constant(R(1)), RationalPoly::constant(R(1)));
                  auto op = IntegralOperator::volterra(k, R(1), R(0));
                  t.expect(op.has_value(), "volterra operator builds");
                  if (!op) {
                      return;
                  }
                  const Nonlinearity id = Nonlinearity::identity();
                  const std::size_t order = 6;

                  auto neumann = volterra2_picard(f, k, R(1), R(0), order);
                  auto adm = adm_solve(*op, f, id, order);
                  auto hpm = hpm_solve(*op, f, id, order);
                  auto ham = ham_solve(*op, f, id, R(-1), order);
                  t.expect(neumann.has_value() && adm.has_value() && hpm.has_value() && ham.has_value(),
                           "all four solves succeed");
                  if (!neumann || !adm || !hpm || !ham) {
                      return;
                  }
                  t.expect(neumann->is_equal(*adm) && adm->is_equal(*hpm) && hpm->is_equal(*ham),
                           "Neumann == ADM == HPM == HAM(hbar=-1), bit-identical");
                  t.expect(adm->is_equal(rp({R(1), R(1), R(1, 2), R(1, 6), R(1, 24), R(1, 120),
                                            R(1, 720)})),
                           "coefficients == 1/n! (exact Taylor series of e^x)");
              })
        .test("nonlinear_volterra_adm_hpm_agree_and_converge",
              [](TestContext& t) {
                  // phi(x) = f(x) + integral_0^x phi(t)^2 dt, K = 1, lambda = 1. Manufactured
                  // so phi(x) = x is the exact fixed point: f(x) = x - integral_0^x t^2 dt
                  // = x - x^3/3 (hand-verified: substituting phi = x into the RHS gives
                  // (x - x^3/3) + x^3/3 = x).
                  const auto f = rp({R(0), R(1), R(0), R(-1, 3)});
                  const auto k = kernel1(RationalPoly::constant(R(1)), RationalPoly::constant(R(1)));
                  auto op = IntegralOperator::volterra(k, R(1), R(0));
                  t.expect(op.has_value(), "volterra operator builds");
                  if (!op) {
                      return;
                  }
                  const Nonlinearity square = Nonlinearity::power(2);
                  const Rational target = R(1, 2);  // the exact phi(1/2) = 1/2

                  double last_diff = 1.0e18;
                  for (std::size_t order = 1; order <= 4; ++order) {
                      auto adm = adm_solve(*op, f, square, order);
                      auto hpm = hpm_solve(*op, f, square, order);
                      auto ham_matches = ham_solve(*op, f, square, R(-1), order);
                      t.expect(adm.has_value() && hpm.has_value() && ham_matches.has_value(),
                               "ADM/HPM/HAM(hbar=-1) all succeed at this order");
                      if (!adm || !hpm || !ham_matches) {
                          continue;
                      }
                      t.expect(adm->is_equal(*hpm), "ADM == HPM (bit-identical) at every order");
                      t.expect(adm->is_equal(*ham_matches), "HAM(hbar=-1) == ADM at every order");

                      if (order >= 2) {
                          auto ham_other = ham_solve(*op, f, square, R(-1, 2), order);
                          t.expect(ham_other.has_value(), "HAM(hbar=-1/2) succeeds");
                          if (ham_other) {
                              t.expect(!ham_other->is_equal(*adm),
                                       "HAM(hbar=-1/2) differs from ADM (a genuinely different, "
                                       "still-exact family member)");
                          }
                      }

                      const double d = abs_diff(eval_at(*adm, target), target);
                      if (order > 1) {
                          t.expect(d < last_diff,
                                   "ADM truncation at x=1/2 converges toward the exact phi(1/2) "
                                   "= 1/2 as the order grows");
                      }
                      last_diff = d;
                  }

                  // Cross-check the INDEPENDENTLY-implemented Picard iteration also converges
                  // toward the same manufactured target (not asserted bit-equal to ADM: Picard's
                  // fixed-point iterate and ADM's graded truncation are different finite
                  // approximations of the same infinite series, so equality is not expected
                  // termwise, only equality in the limit).
                  auto picard_1 = volterra_nonlinear_picard(f, k, R(1), R(0), square, 1);
                  auto picard_4 = volterra_nonlinear_picard(f, k, R(1), R(0), square, 4);
                  t.expect(picard_1.has_value() && picard_4.has_value(), "Picard iterates succeed");
                  if (picard_1 && picard_4) {
                      t.expect(abs_diff(eval_at(*picard_4, target), target) <
                                   abs_diff(eval_at(*picard_1, target), target),
                               "Picard also converges toward the exact phi(1/2) = 1/2");
                  }
              })
        .test("error_paths",
              [](TestContext& t) {
                  const auto f = RationalPoly::constant(R(1));
                  const auto k = kernel1(RationalPoly::constant(R(1)), RationalPoly::constant(R(1)));
                  auto op = IntegralOperator::volterra(k, R(1), R(0));
                  t.expect(op.has_value(), "operator builds");
                  if (!op) {
                      return;
                  }
                  auto zero_order = adm_solve(*op, f, Nonlinearity::identity(), 0);
                  t.expect(!zero_order.has_value() && zero_order.error() == MathError::domain_error,
                           "order == 0 is domain_error");
              })
        .run();
}
