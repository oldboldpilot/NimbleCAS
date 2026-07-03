// Tests for nimblecas.rothstein: Rothstein-Trager logarithmic integration over Q(x).
// @author Olumuyiwa Oluwasanmi
//
// Golden check: differentiate the log part back. If int A/D = sum c_i log(V_i), then
// sum c_i V_i'/V_i must equal A/D, verified exactly by cross-multiplication.

import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.ratpoly;
import nimblecas.rothstein;
import nimblecas.testing;

using nimblecas::LogarithmicPart;
using nimblecas::log_part;
using nimblecas::MathError;
using nimblecas::Polynomial;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto ipoly(std::vector<std::int64_t> c) -> RationalPoly {
    return RationalPoly::from_polynomial(Polynomial{std::move(c)});
}

[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

// sum c_i V_i'/V_i as a single reduced fraction (num, den).
[[nodiscard]] auto derivative_sum(const LogarithmicPart& lp)
    -> std::pair<RationalPoly, RationalPoly> {
    RationalPoly num;                 // 0
    RationalPoly den = ipoly({1});    // 1
    for (const auto& term : lp.terms) {
        const RationalPoly vp = term.argument.derivative().value().scale(term.coefficient).value();
        // num/den + vp/argument = (num*argument + vp*den) / (den*argument)
        num = num.multiply(term.argument).value().add(vp.multiply(den).value()).value();
        den = den.multiply(term.argument).value();
    }
    return {num, den};
}

// d/dx(sum c_i log V_i) == A/D, checked exactly: num*D == A*den.
[[nodiscard]] auto integrates_to(const RationalPoly& a, const RationalPoly& d,
                                 const LogarithmicPart& lp) -> bool {
    auto [num, den] = derivative_sum(lp);
    return num.multiply(d).value().is_equal(a.multiply(den).value());
}

// Does lp contain a term coefficient*log(argument)?
[[nodiscard]] auto has_term(const LogarithmicPart& lp, const Rational& c,
                            const RationalPoly& arg) -> bool {
    for (const auto& term : lp.terms) {
        if (term.coefficient == c && term.argument.is_equal(arg)) {
            return true;
        }
    }
    return false;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.rothstein")
        .test("two_rational_residues",
              [](TestContext& t) {
                  // int 1/(x^2 - 1) dx = (1/2) log(x - 1) - (1/2) log(x + 1).
                  auto a = ipoly({1});
                  auto d = ipoly({-1, 0, 1});  // x^2 - 1
                  auto lp = log_part(a, d).value();
                  t.expect(lp.terms.size() == 2, "two logarithmic terms");
                  t.expect(has_term(lp, rat(1, 2), ipoly({-1, 1})), "(1/2) log(x - 1)");
                  t.expect(has_term(lp, rat(-1, 2), ipoly({1, 1})), "(-1/2) log(x + 1)");
                  t.expect(integrates_to(a, d, lp), "d/dx(sum c log V) == A/D");
              })
        .test("composite_squarefree_denominator",
              [](TestContext& t) {
                  // int 1/(x^2 - x) dx = -log(x) + log(x - 1).
                  auto a = ipoly({1});
                  auto d = ipoly({0, -1, 1});  // x^2 - x = x(x - 1)
                  auto lp = log_part(a, d).value();
                  t.expect(has_term(lp, Rational::from_int(-1), ipoly({0, 1})), "-log(x)");
                  t.expect(has_term(lp, Rational::from_int(1), ipoly({-1, 1})), "log(x - 1)");
                  t.expect(integrates_to(a, d, lp), "d/dx(sum c log V) == A/D");
              })
        .test("derivative_over_self_single_log",
              [](TestContext& t) {
                  // int 2x/(x^2 - 1) dx = log(x^2 - 1): a single log of the whole denominator
                  // (a repeated root of the Rothstein-Trager resultant, residue 1).
                  auto a = ipoly({0, 2});
                  auto d = ipoly({-1, 0, 1});
                  auto lp = log_part(a, d).value();
                  t.expect(lp.terms.size() == 1, "one logarithmic term");
                  t.expect(has_term(lp, Rational::from_int(1), ipoly({-1, 0, 1})),
                           "log(x^2 - 1)");
                  t.expect(integrates_to(a, d, lp), "d/dx(log(x^2-1)) == 2x/(x^2-1)");
              })
        .test("three_distinct_residues",
              [](TestContext& t) {
                  // int 1/(x(x-1)(x-4)) dx: three DISTINCT rational residues.
                  // D'(0)=4, D'(1)=-3, D'(4)=12 -> residues 1/4, -1/3, 1/12.
                  auto d = ipoly({0, 1}).multiply(ipoly({-1, 1})).value()
                               .multiply(ipoly({-4, 1})).value();  // x^3 - 5x^2 + 4x
                  auto a = ipoly({1});
                  auto lp = log_part(a, d).value();
                  t.expect(lp.terms.size() == 3, "three logarithmic terms");
                  t.expect(has_term(lp, rat(1, 4), ipoly({0, 1})), "(1/4) log(x)");
                  t.expect(has_term(lp, rat(-1, 3), ipoly({-1, 1})), "(-1/3) log(x - 1)");
                  t.expect(has_term(lp, rat(1, 12), ipoly({-4, 1})), "(1/12) log(x - 4)");
                  t.expect(integrates_to(a, d, lp), "d/dx(sum c log V) == A/D");
              })
        .test("coinciding_residues_merge",
              [](TestContext& t) {
                  // int 1/((x-1)(x-2)(x-3)) dx: residues 1/D'(1)=1/2, 1/D'(2)=-1,
                  // 1/D'(3)=1/2. The shared residue 1/2 MERGES into a single log of the
                  // combined factor -- Rothstein-Trager emits the minimal set of logs.
                  auto d = ipoly({-1, 1}).multiply(ipoly({-2, 1})).value()
                               .multiply(ipoly({-3, 1})).value();  // x^3 - 6x^2 + 11x - 6
                  auto a = ipoly({1});
                  auto lp = log_part(a, d).value();
                  t.expect(lp.terms.size() == 2, "two terms: the residue 1/2 is shared");
                  // (1/2) log((x-1)(x-3)) = (1/2) log(x^2 - 4x + 3).
                  t.expect(has_term(lp, rat(1, 2), ipoly({3, -4, 1})),
                           "(1/2) log((x-1)(x-3))");
                  t.expect(has_term(lp, Rational::from_int(-1), ipoly({-2, 1})), "-log(x - 2)");
                  t.expect(integrates_to(a, d, lp), "d/dx(sum c log V) == A/D");
              })
        .test("non_reduced_input",
              [](TestContext& t) {
                  // A shared factor cancels: int (x)/(x(x-1)) = int 1/(x-1) = log(x - 1).
                  auto a = ipoly({0, 1});                        // x
                  auto d = ipoly({0, -1, 1});                    // x^2 - x = x(x-1)
                  auto lp = log_part(a, d).value();
                  t.expect(lp.terms.size() == 1, "the cancelled pole drops out");
                  t.expect(has_term(lp, Rational::from_int(1), ipoly({-1, 1})), "log(x - 1)");
                  t.expect(integrates_to(a, d, lp), "d/dx(log(x-1)) == x/(x^2-x)");
              })
        .test("irrational_and_error_paths",
              [](TestContext& t) {
                  // int 1/(x^2 + 1) dx = arctan(x): the residues are +-i/2, not rational.
                  t.expect(log_part(ipoly({1}), ipoly({1, 0, 1})).error() ==
                               MathError::not_implemented,
                           "complex residues are not_implemented");
                  // int 1/(x^2 - 2) dx has residues +-1/(2 sqrt 2), irrational.
                  t.expect(log_part(ipoly({1}), ipoly({-2, 0, 1})).error() ==
                               MathError::not_implemented,
                           "irrational residues are not_implemented");
                  // Zero denominator.
                  t.expect(log_part(ipoly({1}), RationalPoly{}).error() ==
                               MathError::division_by_zero,
                           "zero denominator fails");
                  // Zero numerator integrates to no logarithmic part.
                  t.expect(log_part(RationalPoly{}, ipoly({-1, 0, 1})).value().terms.empty(),
                           "zero numerator has no log part");
              })
        .run();
}
