// Tests for nimblecas.integrate: the rational-function integration capstone.
// @author Olumuyiwa Oluwasanmi
//
// Golden check: differentiate the whole result back. If int A/B = g + sum c_i log(V_i),
// then d/dx(g) + sum c_i V_i'/V_i must equal A/B, verified exactly by cross-multiplication.

import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.ratpoly;
import nimblecas.integrate;
import nimblecas.testing;

using nimblecas::integrate_rational;
using nimblecas::MathError;
using nimblecas::Polynomial;
using nimblecas::Rational;
using nimblecas::RationalIntegral;
using nimblecas::RationalPoly;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto ipoly(std::vector<std::int64_t> c) -> RationalPoly {
    return RationalPoly::from_polynomial(Polynomial{std::move(c)});
}

// d/dx(g) + sum c_i V_i'/V_i == A/D, checked exactly (num*D == A*den).
[[nodiscard]] auto integrates_to(const RationalPoly& a, const RationalPoly& d,
                                 const RationalIntegral& ri) -> bool {
    // g' = (rn' * rd - rn * rd') / rd^2.
    RationalPoly num = ri.rational_num.derivative().value().multiply(ri.rational_den).value()
                           .subtract(ri.rational_num.multiply(
                               ri.rational_den.derivative().value()).value())
                           .value();
    RationalPoly den = ri.rational_den.multiply(ri.rational_den).value();
    // Add each c_i * V_i'/V_i.
    for (const auto& term : ri.log_terms) {
        const RationalPoly cn = term.argument.derivative().value().scale(term.coefficient).value();
        num = num.multiply(term.argument).value().add(cn.multiply(den).value()).value();
        den = den.multiply(term.argument).value();
    }
    return num.multiply(d).value().is_equal(a.multiply(den).value());
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.integrate")
        .test("pure_rational",
              [](TestContext& t) {
                  // int 1/(x-1)^2 dx = -1/(x-1): rational part only, no logs.
                  auto a = ipoly({1});
                  auto d = ipoly({1, -2, 1});  // (x - 1)^2
                  auto ri = integrate_rational(a, d).value();
                  t.expect(ri.log_terms.empty(), "no logarithmic terms");
                  t.expect(!ri.rational_num.is_zero(), "a rational part is present");
                  t.expect(integrates_to(a, d, ri), "d/dx of the result == A/D");
              })
        .test("pure_logarithmic",
              [](TestContext& t) {
                  // int 1/(x^2 - 1) dx = (1/2)log(x-1) - (1/2)log(x+1): logs only.
                  auto a = ipoly({1});
                  auto d = ipoly({-1, 0, 1});
                  auto ri = integrate_rational(a, d).value();
                  t.expect(ri.rational_num.is_zero(), "no rational part");
                  t.expect(ri.log_terms.size() == 2, "two logarithmic terms");
                  t.expect(integrates_to(a, d, ri), "d/dx of the result == A/D");
              })
        .test("improper_rational_plus_log",
              [](TestContext& t) {
                  // int x^3/(x^2 - 1) dx = x^2/2 + (1/2)log(x^2 - 1): polynomial-driven
                  // rational part plus a single merged logarithm.
                  auto a = ipoly({0, 0, 0, 1});
                  auto d = ipoly({-1, 0, 1});
                  auto ri = integrate_rational(a, d).value();
                  t.expect(!ri.rational_num.is_zero(), "rational part present");
                  t.expect(ri.log_terms.size() == 1, "one merged logarithm");
                  t.expect(integrates_to(a, d, ri), "d/dx of the result == A/D");
              })
        .test("proper_rational_plus_logs",
              [](TestContext& t) {
                  // int 1/(x^2 (x-1)) dx = 1/x - log(x) + log(x-1): a double pole gives a
                  // rational part; the simple poles give two logs.
                  auto a = ipoly({1});
                  auto d = ipoly({0, 0, -1, 1});  // x^2 (x - 1)
                  auto ri = integrate_rational(a, d).value();
                  t.expect(!ri.rational_num.is_zero(), "rational part present");
                  t.expect(ri.log_terms.size() == 2, "two logarithmic terms");
                  t.expect(integrates_to(a, d, ri), "d/dx of the result == A/D");
              })
        .test("purely_polynomial",
              [](TestContext& t) {
                  // int (2x)/1 dx = x^2: constant denominator, no poles.
                  auto a = ipoly({0, 2});
                  auto d = ipoly({1});
                  auto ri = integrate_rational(a, d).value();
                  t.expect(ri.log_terms.empty(), "no logarithmic terms");
                  t.expect(ri.rational_num.is_equal(ipoly({0, 0, 1})), "antiderivative is x^2");
                  t.expect(integrates_to(a, d, ri), "d/dx of the result == A/D");
              })
        .test("not_implemented_and_errors",
              [](TestContext& t) {
                  // int 1/(x^2+1)^2 dx has rational part x/(2(x^2+1)) but a logarithmic part
                  // with complex residues, so the whole integral is not_implemented.
                  t.expect(integrate_rational(ipoly({1}), ipoly({1, 0, 2, 0, 1})).error() ==
                               MathError::not_implemented,
                           "complex-residue log part propagates not_implemented");
                  // int 1/(x^2 - 2) dx: irrational residues.
                  t.expect(integrate_rational(ipoly({1}), ipoly({-2, 0, 1})).error() ==
                               MathError::not_implemented,
                           "irrational-residue log part is not_implemented");
                  // Zero denominator.
                  t.expect(integrate_rational(ipoly({1}), RationalPoly{}).error() ==
                               MathError::division_by_zero,
                           "zero denominator fails");
              })
        .run();
}
