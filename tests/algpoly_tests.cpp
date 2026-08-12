// Tests for nimblecas.algpoly: dense univariate polynomials over Q(alpha).
// @author Olumuyiwa Oluwasanmi
//
// Oracles are hand-verified over Q(i) = Q[t]/(t^2 + 1) (alpha^2 = -1) and, for the
// field-mismatch guard, Q(sqrt2) = Q[t]/(t^2 - 2). The load-bearing identity is
// (x - alpha)(x + alpha) = x^2 - alpha^2 = x^2 + 1 over Q(i).

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.algnum;
import nimblecas.algpoly;
import nimblecas.testing;

using nimblecas::AlgebraicNumber;
using nimblecas::AlgebraicPoly;
using nimblecas::MathError;
using nimblecas::NumberField;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// x^2 + 1 as a RationalPoly (constant, linear, quadratic coefficients in index order).
auto poly_x2_plus_1() -> RationalPoly {
    return RationalPoly::from_coeffs(
        {Rational::from_int(1), Rational::from_int(0), Rational::from_int(1)});
}
auto poly_x2_minus_2() -> RationalPoly {
    return RationalPoly::from_coeffs(
        {Rational::from_int(-2), Rational::from_int(0), Rational::from_int(1)});
}

}  // namespace

auto main() -> int {
    auto qi_r = NumberField::create(poly_x2_plus_1());
    if (!qi_r) {
        std::println("FATAL: could not construct Q(i)");
        return 1;
    }
    const NumberField qi = *qi_r;

    auto q2_r = NumberField::create(poly_x2_minus_2());
    if (!q2_r) {
        std::println("FATAL: could not construct Q(sqrt2)");
        return 1;
    }
    const NumberField q2 = *q2_r;

    // Helper: build (x - alpha) and (x + alpha) over Q(i). Returns std::nullopt on any
    // arithmetic failure so a test can report it rather than crash.
    auto linear_pair = [](const NumberField& f)
        -> std::optional<std::pair<AlgebraicPoly, AlgebraicPoly>> {
        auto alpha = f.generator();
        if (!alpha) return std::nullopt;
        auto neg_alpha = alpha->negate();
        if (!neg_alpha) return std::nullopt;
        auto minus = AlgebraicPoly::from_coeffs(f, {*neg_alpha, f.one()});  // x - alpha
        auto plus = AlgebraicPoly::from_coeffs(f, {*alpha, f.one()});       // x + alpha
        if (!minus || !plus) return std::nullopt;
        return std::pair{*minus, *plus};
    };

    return TestSuite("nimblecas.algpoly")
        .test("zero and embed report the right degree",
              [&](TestContext& t) {
                  const AlgebraicPoly z = AlgebraicPoly::zero(qi);
                  t.expect(z.is_zero() && z.degree() == -1, "zero polynomial has degree -1");
                  const AlgebraicPoly e = AlgebraicPoly::embed(qi, poly_x2_plus_1());
                  t.expect(e.degree() == 2, "embed(x^2+1) has degree 2");
                  t.expect(!e.leading_coefficient().is_zero(), "leading coeff nonzero");
              })
        .test("(x - alpha)(x + alpha) == x^2 + 1 over Q(i)",
              [&](TestContext& t) {
                  auto pr = linear_pair(qi);
                  if (!t.expect(pr.has_value(), "built linear factors")) return;
                  auto prod = pr->first.multiply(pr->second);
                  if (!t.expect(prod.has_value(), "multiply succeeded")) return;
                  const AlgebraicPoly target = AlgebraicPoly::embed(qi, poly_x2_plus_1());
                  t.expect(prod->is_equal(target), "product equals x^2 + 1");
              })
        .test("divide: (x^2+1) / (x - alpha) == (x + alpha) rem 0",
              [&](TestContext& t) {
                  auto pr = linear_pair(qi);
                  if (!t.expect(pr.has_value(), "built linear factors")) return;
                  const AlgebraicPoly target = AlgebraicPoly::embed(qi, poly_x2_plus_1());
                  auto dm = target.divide(pr->first);
                  if (!t.expect(dm.has_value(), "divide succeeded")) return;
                  t.expect(dm->remainder.is_zero(), "remainder is zero");
                  t.expect(dm->quotient.is_equal(pr->second), "quotient is x + alpha");
              })
        .test("gcd(x^2+1, x - alpha) == x - alpha (monic) over Q(i)",
              [&](TestContext& t) {
                  auto pr = linear_pair(qi);
                  if (!t.expect(pr.has_value(), "built linear factors")) return;
                  const AlgebraicPoly target = AlgebraicPoly::embed(qi, poly_x2_plus_1());
                  auto g = target.gcd(pr->first);
                  if (!t.expect(g.has_value(), "gcd succeeded")) return;
                  // x - alpha is already monic, so the monic gcd equals it exactly.
                  t.expect(g->is_equal(pr->first), "gcd equals x - alpha");
              })
        .test("evaluate at the root alpha is zero",
              [&](TestContext& t) {
                  auto alpha = qi.generator();
                  if (!t.expect(alpha.has_value(), "generator")) return;
                  auto pr = linear_pair(qi);
                  if (!t.expect(pr.has_value(), "built linear factors")) return;
                  auto v1 = pr->first.evaluate(*alpha);  // (x - alpha) at alpha
                  if (!t.expect(v1.has_value(), "evaluate x-alpha")) return;
                  t.expect(v1->is_zero(), "(x - alpha)(alpha) == 0");
                  const AlgebraicPoly target = AlgebraicPoly::embed(qi, poly_x2_plus_1());
                  auto v2 = target.evaluate(*alpha);  // alpha^2 + 1 == 0
                  if (!t.expect(v2.has_value(), "evaluate x^2+1")) return;
                  t.expect(v2->is_zero(), "alpha^2 + 1 == 0");
              })
        .test("derivative of x^2 + 1 is 2x",
              [&](TestContext& t) {
                  const AlgebraicPoly target = AlgebraicPoly::embed(qi, poly_x2_plus_1());
                  auto d = target.derivative();
                  if (!t.expect(d.has_value(), "derivative succeeded")) return;
                  t.expect(d->degree() == 1, "derivative has degree 1");
                  const AlgebraicNumber two = qi.from_rational(Rational::from_int(2));
                  t.expect(d->coefficient(1).is_equal(two), "coeff of x is 2");
                  t.expect(d->coefficient(0).is_zero(), "constant term is 0");
              })
        .test("monic normalises 3(x - alpha) back to x - alpha",
              [&](TestContext& t) {
                  auto pr = linear_pair(qi);
                  if (!t.expect(pr.has_value(), "built linear factors")) return;
                  const AlgebraicNumber three = qi.from_rational(Rational::from_int(3));
                  auto scaled = pr->first.scale(three);  // 3x - 3alpha
                  if (!t.expect(scaled.has_value(), "scale succeeded")) return;
                  t.expect(!scaled->leading_coefficient().is_one(), "scaled is not monic");
                  auto m = scaled->monic();
                  if (!t.expect(m.has_value(), "monic succeeded")) return;
                  t.expect(m->is_equal(pr->first), "monic(3(x-alpha)) == x - alpha");
              })
        .test("field mismatch and divide-by-zero are honest errors",
              [&](TestContext& t) {
                  auto pr = linear_pair(qi);
                  if (!t.expect(pr.has_value(), "built Q(i) factors")) return;
                  auto beta = q2.generator();
                  if (!t.expect(beta.has_value(), "Q(sqrt2) generator")) return;
                  auto other = AlgebraicPoly::from_coeffs(q2, {*beta, q2.one()});  // x + sqrt2
                  if (!t.expect(other.has_value(), "built Q(sqrt2) factor")) return;
                  auto bad = pr->first.add(*other);  // different fields
                  t.expect(!bad.has_value() && bad.error() == MathError::domain_error,
                           "add across fields -> domain_error");
                  auto dz = pr->first.divide(AlgebraicPoly::zero(qi));
                  t.expect(!dz.has_value() && dz.error() == MathError::division_by_zero,
                           "divide by zero -> division_by_zero");
              })
        .run();
}
