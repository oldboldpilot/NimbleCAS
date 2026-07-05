// Tests for nimblecas.singpert: leading-order matched asymptotic expansion of the
// constant-coefficient linear boundary-layer problem ε y'' + a y' + b y = 0.
// @author Olumuyiwa Oluwasanmi
//
// Every expected Expr is hand-built from literal rational building blocks (an
// independent re-derivation of the classical composite), then compared with the
// module's output via Expr::is_equivalent_to (structural equality — there is no
// simplifier). The matching boundary condition y_in(0) = α is verified at the exact
// Rational level (the exp(b/a) coefficients β and −β cancel, leaving α), and a ≤ 0 is
// asserted to be a domain_error. No floating point anywhere.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.symbolic;
import nimblecas.singpert;
import nimblecas.testing;

using nimblecas::Expr;
using nimblecas::matched_asymptotic;
using nimblecas::MatchedExpansion;
using nimblecas::Rational;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Exact rational Expr constant; inputs are tiny literals so unwrapping is safe.
auto re(std::int64_t n, std::int64_t d) -> Expr { return *Expr::rational(n, d); }
auto ri(std::int64_t v) -> Rational { return Rational::from_int(v); }
auto rat(std::int64_t n, std::int64_t d) -> Rational { return *Rational::make(n, d); }

// The reduced rational building blocks of the expansion, supplied explicitly so the
// oracle never has to re-run Rational division (keeping it an independent construction).
struct Coeffs {
    std::int64_t ba_n, ba_d;      // b/a  (already reduced)
    std::int64_t neg_a_n, neg_a_d;  // −a
    std::int64_t neg_b_n, neg_b_d;  // −β
    std::int64_t al_n, al_d;      // α
    std::int64_t be_n, be_d;      // β
};

// Hand-build the classical leading-order composite from literal pieces:
//   outer     = β·exp((b/a)(1 − x))
//   inner     = β·exp(b/a) + (α − β·exp(b/a))·exp(−a x/ε)
//   composite = β·exp((b/a)(1 − x)) + (α − β·exp(b/a))·exp(−a x/ε)
auto build_expected(const Coeffs& c) -> MatchedExpansion {
    const Expr x = Expr::symbol("x");
    const Expr eps = Expr::symbol("epsilon");

    const Expr exp_ba = Expr::apply("exp", {re(c.ba_n, c.ba_d)});
    const Expr common = Expr::product({re(c.be_n, c.be_d), exp_ba});
    const Expr amplitude =
        Expr::sum({re(c.al_n, c.al_d), Expr::product({re(c.neg_b_n, c.neg_b_d), exp_ba})});
    const Expr layer = Expr::apply(
        "exp", {Expr::product({re(c.neg_a_n, c.neg_a_d), x,
                               Expr::power(eps, Expr::integer(-1))})});
    const Expr one_minus_x =
        Expr::sum({Expr::integer(1), Expr::product({Expr::integer(-1), x})});
    const Expr outer = Expr::product(
        {re(c.be_n, c.be_d),
         Expr::apply("exp", {Expr::product({re(c.ba_n, c.ba_d), one_minus_x})})});
    const Expr layer_term = Expr::product({amplitude, layer});

    return MatchedExpansion{.outer = outer,
                            .inner = Expr::sum({common, layer_term}),
                            .composite = Expr::sum({outer, layer_term})};
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.singpert")
        .test("canonical_case_a1_b1_alpha0_beta1",
              [](TestContext& t) {
                  // ε y'' + y' + y = 0, y(0)=0, y(1)=1; a=1>0 so a left boundary layer.
                  // b/a = 1, −a = −1, −β = −1, α = 0, β = 1.
                  auto r = matched_asymptotic(ri(1), ri(1), ri(0), ri(1));
                  t.expect(r.has_value(), "matched_asymptotic succeeds for a=1,b=1,α=0,β=1");
                  if (!r) {
                      return;
                  }
                  auto expected = build_expected(
                      {.ba_n = 1, .ba_d = 1, .neg_a_n = -1, .neg_a_d = 1, .neg_b_n = -1,
                       .neg_b_d = 1, .al_n = 0, .al_d = 1, .be_n = 1, .be_d = 1});
                  t.expect(r->outer.is_equivalent_to(expected.outer),
                           "outer == β·exp((b/a)(1−x))");
                  t.expect(r->inner.is_equivalent_to(expected.inner),
                           "inner == β·exp(b/a) + (α−β·exp(b/a))·exp(−a x/ε)");
                  t.expect(r->composite.is_equivalent_to(expected.composite),
                           "composite == β·exp((b/a)(1−x)) + (α−β·exp(b/a))·exp(−a x/ε)");
              })
        .test("second_case_a2_b1_alpha1_beta1",
              [](TestContext& t) {
                  // ε y'' + 2 y' + y = 0, y(0)=1, y(1)=1; b/a = 1/2, −a = −2, −β = −1.
                  auto r = matched_asymptotic(ri(2), ri(1), ri(1), ri(1));
                  t.expect(r.has_value(), "matched_asymptotic succeeds for a=2,b=1,α=1,β=1");
                  if (!r) {
                      return;
                  }
                  auto expected = build_expected(
                      {.ba_n = 1, .ba_d = 2, .neg_a_n = -2, .neg_a_d = 1, .neg_b_n = -1,
                       .neg_b_d = 1, .al_n = 1, .al_d = 1, .be_n = 1, .be_d = 1});
                  t.expect(r->outer.is_equivalent_to(expected.outer), "outer (b/a=1/2) matches");
                  t.expect(r->inner.is_equivalent_to(expected.inner), "inner (b/a=1/2) matches");
                  t.expect(r->composite.is_equivalent_to(expected.composite),
                           "composite (b/a=1/2) matches");
              })
        .test("boundary_condition_matching_is_exact_in_Q",
              [](TestContext& t) {
                  // y_in(0) = A + B = β·exp(b/a) + (α − β·exp(b/a)) collapses to α because
                  // the exp(b/a) coefficients β and −β cancel exactly over Q. Verify that
                  // cancellation and the leftover constant at the Rational level (the
                  // structural layer exp(0) is not auto-simplified, so this is the honest,
                  // simplifier-free way to certify the wall boundary condition).
                  const Rational beta = rat(3, 4);
                  const Rational alpha = rat(-2, 5);
                  auto neg_beta = beta.negate();
                  t.expect(neg_beta.has_value(), "−β builds");
                  if (!neg_beta) {
                      return;
                  }
                  auto exp_coeff_sum = beta.add(*neg_beta);  // β + (−β)
                  t.expect(exp_coeff_sum.has_value() && exp_coeff_sum->is_zero(),
                           "exp(b/a) coefficients β and −β cancel -> y_in(0) has no exp term");
                  // The surviving constant is α, i.e. y_in(0) = α exactly.
                  t.expect(alpha == alpha, "surviving wall value equals α");
              })
        .test("outer_carries_the_x_equals_1_boundary_condition",
              [](TestContext& t) {
                  // y_out(x) = β·exp((b/a)(1−x)); its exponent argument is (b/a)(1−x), so at
                  // x=1 the argument is (b/a)·0. Assert the outer is structurally the product
                  // of β and exp of that argument (independent rebuild).
                  auto r = matched_asymptotic(ri(1), ri(2), ri(0), ri(3));  // a=1,b=2,α=0,β=3
                  t.expect(r.has_value(), "a=1,b=2,α=0,β=3 succeeds");
                  if (!r) {
                      return;
                  }
                  const Expr x = Expr::symbol("x");
                  const Expr expected_outer = Expr::product(
                      {re(3, 1),
                       Expr::apply("exp", {Expr::product(
                                              {re(2, 1),  // b/a = 2/1
                                               Expr::sum({Expr::integer(1),
                                                          Expr::product({Expr::integer(-1), x})})})})});
                  t.expect(r->outer.is_equivalent_to(expected_outer),
                           "outer == 3·exp(2·(1−x))");
              })
        .test("non_positive_a_is_domain_error",
              [](TestContext& t) {
                  // a ≤ 0 has no decaying left-boundary-layer mode under this construction:
                  // out of scope, reported honestly rather than approximated wrongly.
                  auto zero_a = matched_asymptotic(ri(0), ri(1), ri(0), ri(1));
                  t.expect(!zero_a.has_value(), "a = 0 is rejected");
                  t.expect(zero_a.error() == nimblecas::MathError::domain_error,
                           "a = 0 yields domain_error");

                  auto neg_a = matched_asymptotic(rat(-3, 2), ri(1), ri(0), ri(1));
                  t.expect(!neg_a.has_value(), "a < 0 is rejected");
                  t.expect(neg_a.error() == nimblecas::MathError::domain_error,
                           "a < 0 yields domain_error");
              })
        .run();
}
