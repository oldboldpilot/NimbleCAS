// Tests for nimblecas.perturbation: ADM, HPM and HAM solvers over Q (exact series).
// @author Olumuyiwa Oluwasanmi
//
// Every case uses exact rational coefficients hand-verified in the comments. The three
// ODE benchmarks (u'=u -> exp, u'=u^2 -> 1/(1-x), u'=1+u^2 -> tan) are checked against
// their known Maclaurin coefficients, ADM/HPM/HAM(hbar=-1) are asserted identical, and
// the Adomian polynomials are checked against explicit hand-built products. No floating
// point anywhere.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.powerseries;
import nimblecas.perturbation;
import nimblecas.testing;

using nimblecas::adm_solve;
using nimblecas::adomian_polynomials;
using nimblecas::ham_solve;
using nimblecas::hpm_solve;
using nimblecas::PowerSeries;
using nimblecas::Rational;
using nimblecas::SeriesOperator;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Small exact rational helpers; inputs are tiny literals so unwrapping is safe.
auto ri(std::int64_t v) -> Rational { return Rational::from_int(v); }
auto rat(std::int64_t num, std::int64_t den) -> Rational { return *Rational::make(num, den); }

// Build a PowerSeries from coefficients at the given order, asserting success.
auto series(TestContext& t, std::vector<Rational> coeffs, std::size_t order,
            std::string_view what) -> PowerSeries {
    auto s = PowerSeries::from_coeffs(std::move(coeffs), order);
    if (!s) {
        t.expect(false, std::format("{}: unexpected series construction error", what));
        return *PowerSeries::zero(1);
    }
    return *s;
}

// Assert a series has exactly the expected coefficients (index i is the coefficient of
// x^i) and the expected order.
auto expect_series(TestContext& t, const PowerSeries& s, const std::vector<Rational>& expected,
                   std::string_view what) -> void {
    t.expect(s.order() == expected.size(),
             std::format("{}: order = {}", what, expected.size()));
    for (std::size_t i = 0; i < expected.size(); ++i) {
        t.expect(s.coefficient(i) == expected[i],
                 std::format("{}: coefficient[{}] = {}", what, i, expected[i].to_string()));
    }
}

// The IVP operators used below (f such that u' = f(u)).
auto f_identity() -> SeriesOperator {
    return [](const PowerSeries& u) -> nimblecas::Result<PowerSeries> { return u; };  // u' = u
}
auto f_square() -> SeriesOperator {
    return [](const PowerSeries& u) { return u.multiply(u); };  // u' = u^2
}
auto f_one_plus_square() -> SeriesOperator {
    return [](const PowerSeries& u) -> nimblecas::Result<PowerSeries> {  // u' = 1 + u^2
        auto one = PowerSeries::one(u.order());
        if (!one) {
            return nimblecas::make_error<PowerSeries>(one.error());
        }
        auto sq = u.multiply(u);
        if (!sq) {
            return nimblecas::make_error<PowerSeries>(sq.error());
        }
        return one->add(*sq);
    };
}
auto f_double() -> SeriesOperator {
    return [](const PowerSeries& u) { return u.scale(Rational::from_int(2)); };  // u' = 2u
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.perturbation")
        .test("adm_exponential",
              [](TestContext& t) {
                  // u' = u, u(0) = 1 -> e^x = sum x^n/n!.
                  auto r = adm_solve(f_identity(), ri(1), 8);
                  t.expect(r.has_value(), "adm(u'=u) succeeds");
                  if (!r) {
                      return;
                  }
                  expect_series(t, *r,
                                {ri(1), ri(1), rat(1, 2), rat(1, 6), rat(1, 24), rat(1, 120),
                                 rat(1, 720), rat(1, 5040)},
                                "exp series");
              })
        .test("adm_geometric",
              [](TestContext& t) {
                  // u' = u^2, u(0) = 1 -> 1/(1-x) = sum x^n (all coefficients 1).
                  auto r = adm_solve(f_square(), ri(1), 8);
                  t.expect(r.has_value(), "adm(u'=u^2) succeeds");
                  if (!r) {
                      return;
                  }
                  expect_series(t, *r,
                                {ri(1), ri(1), ri(1), ri(1), ri(1), ri(1), ri(1), ri(1)},
                                "1/(1-x) series");
              })
        .test("adm_tangent",
              [](TestContext& t) {
                  // u' = 1 + u^2, u(0) = 0 -> tan(x): 0, 1, 0, 1/3, 0, 2/15, 0, 17/315.
                  auto r = adm_solve(f_one_plus_square(), ri(0), 8);
                  t.expect(r.has_value(), "adm(u'=1+u^2) succeeds");
                  if (!r) {
                      return;
                  }
                  expect_series(t, *r,
                                {ri(0), ri(1), ri(0), rat(1, 3), ri(0), rat(2, 15), ri(0),
                                 rat(17, 315)},
                                "tan series");
              })
        .test("adm_exp_two_x",
              [](TestContext& t) {
                  // Autonomous linear scaling: u' = 2u, u(0) = 1 -> e^{2x} = sum 2^n x^n/n!.
                  auto r = adm_solve(f_double(), ri(1), 8);
                  t.expect(r.has_value(), "adm(u'=2u) succeeds");
                  if (!r) {
                      return;
                  }
                  expect_series(t, *r,
                                {ri(1), ri(2), ri(2), rat(4, 3), rat(2, 3), rat(4, 15),
                                 rat(4, 45), rat(8, 315)},
                                "e^{2x} series");
              })
        .test("adm_hpm_ham_agree",
              [](TestContext& t) {
                  // ADM, HPM and HAM(hbar = -1) return the identical series on every case.
                  struct Case {
                      SeriesOperator f;
                      Rational u0;
                      std::string name;
                  };
                  std::vector<Case> cases;
                  cases.push_back({f_identity(), ri(1), "u'=u"});
                  cases.push_back({f_square(), ri(1), "u'=u^2"});
                  cases.push_back({f_one_plus_square(), ri(0), "u'=1+u^2"});
                  cases.push_back({f_double(), ri(1), "u'=2u"});
                  for (const auto& c : cases) {
                      auto a = adm_solve(c.f, c.u0, 8);
                      auto h = hpm_solve(c.f, c.u0, 8);
                      auto m = ham_solve(c.f, c.u0, ri(-1), 8);
                      t.expect(a.has_value() && h.has_value() && m.has_value(),
                               std::format("{}: all three solve", c.name));
                      if (!a || !h || !m) {
                          continue;
                      }
                      t.expect(a->is_equal(*h), std::format("{}: ADM == HPM", c.name));
                      t.expect(a->is_equal(*m), std::format("{}: ADM == HAM(hbar=-1)", c.name));
                  }
              })
        .test("ham_hbar_is_a_free_parameter",
              [](TestContext& t) {
                  // hbar = -1 recovers ADM exactly; a different hbar yields a different
                  // (still exactly rational) series, demonstrating the extra HAM freedom.
                  auto adm = adm_solve(f_identity(), ri(1), 5);
                  auto ham_m1 = ham_solve(f_identity(), ri(1), ri(-1), 5);
                  auto ham_m2 = ham_solve(f_identity(), ri(1), ri(-2), 5);
                  t.expect(adm.has_value() && ham_m1.has_value() && ham_m2.has_value(),
                           "all solves succeed");
                  if (!adm || !ham_m1 || !ham_m2) {
                      return;
                  }
                  t.expect(adm->is_equal(*ham_m1), "HAM(hbar=-1) == ADM");
                  t.expect(!adm->is_equal(*ham_m2), "HAM(hbar=-2) differs from ADM");
              })
        .test("adomian_polynomials_square",
              [](TestContext& t) {
                  // N(u) = u^2 with homogeneous components u_0 = 3, u_1 = 5x, u_2 = 7x^2.
                  // Expected: A_0 = u_0^2 = 9; A_1 = 2 u_0 u_1 = 30x;
                  //           A_2 = 2 u_0 u_2 + u_1^2 = 42x^2 + 25x^2 = 67x^2.
                  const std::size_t order = 6;
                  auto u0 = series(t, {ri(3)}, order, "u0");
                  auto u1 = series(t, {ri(0), ri(5)}, order, "u1");
                  auto u2 = series(t, {ri(0), ri(0), ri(7)}, order, "u2");
                  auto r = adomian_polynomials(f_square(), {u0, u1, u2});
                  t.expect(r.has_value(), "adomian_polynomials succeeds");
                  if (!r) {
                      return;
                  }
                  t.expect(r->size() == 3, "three Adomian polynomials A_0..A_2");
                  if (r->size() != 3) {
                      return;
                  }
                  // Hand-built expected polynomials via exact series products.
                  auto a0_expected = u0.multiply(u0);  // u_0^2
                  auto two_u0_u1 = u0.multiply(u1);
                  auto two_u0_u2 = u0.multiply(u2);
                  auto u1_sq = u1.multiply(u1);
                  t.expect(a0_expected.has_value() && two_u0_u1.has_value() &&
                               two_u0_u2.has_value() && u1_sq.has_value(),
                           "expected products build");
                  if (!a0_expected || !two_u0_u1 || !two_u0_u2 || !u1_sq) {
                      return;
                  }
                  auto a1_expected = two_u0_u1->scale(ri(2));
                  auto two_u0_u2_scaled = two_u0_u2->scale(ri(2));
                  t.expect(a1_expected.has_value() && two_u0_u2_scaled.has_value(),
                           "A_1/A_2 partial products scale");
                  if (!a1_expected || !two_u0_u2_scaled) {
                      return;
                  }
                  auto a2_expected = two_u0_u2_scaled->add(*u1_sq);
                  t.expect(a2_expected.has_value(), "A_2 expected assembles");
                  if (!a2_expected) {
                      return;
                  }
                  t.expect((*r)[0].is_equal(*a0_expected), "A_0 == u_0^2 (= 9)");
                  t.expect((*r)[1].is_equal(*a1_expected), "A_1 == 2 u_0 u_1 (= 30x)");
                  t.expect((*r)[2].is_equal(*a2_expected),
                           "A_2 == 2 u_0 u_2 + u_1^2 (= 67x^2)");
                  // Spot-check the concrete coefficients too.
                  expect_series(t, (*r)[0], {ri(9), ri(0), ri(0), ri(0), ri(0), ri(0)}, "A_0");
                  expect_series(t, (*r)[1], {ri(0), ri(30), ri(0), ri(0), ri(0), ri(0)}, "A_1");
                  expect_series(t, (*r)[2], {ri(0), ri(0), ri(67), ri(0), ri(0), ri(0)}, "A_2");
              })
        .test("degenerate_arguments_are_domain_errors",
              [](TestContext& t) {
                  auto bad_order = adm_solve(f_identity(), ri(1), 0);
                  t.expect(!bad_order.has_value(), "order 0 is rejected");
                  t.expect(bad_order.error() == nimblecas::MathError::domain_error,
                           "order 0 yields domain_error");

                  auto empty = adomian_polynomials(f_square(), {});
                  t.expect(!empty.has_value(), "empty component list is rejected");
                  t.expect(empty.error() == nimblecas::MathError::domain_error,
                           "empty components yield domain_error");

                  auto bad_ham = ham_solve(f_identity(), ri(1), ri(-1), 0);
                  t.expect(!bad_ham.has_value(), "ham order 0 is rejected");
                  t.expect(bad_ham.error() == nimblecas::MathError::domain_error,
                           "ham order 0 yields domain_error");
              })
        .run();
}
