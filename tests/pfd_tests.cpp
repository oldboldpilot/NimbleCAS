// Tests for nimblecas.pfd: Yun square-free factorization and partial fractions over Q[x].
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.ratpoly;
import nimblecas.pfd;
import nimblecas.testing;

using nimblecas::MathError;
using nimblecas::PartialFraction;
using nimblecas::partial_fractions;
using nimblecas::Polynomial;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::square_free_factorization;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Build a RationalPoly from integer coefficients (low degree first).
[[nodiscard]] auto ipoly(std::vector<std::int64_t> c) -> RationalPoly {
    return RationalPoly::from_polynomial(Polynomial{std::move(c)});
}

[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

// Constant RationalPoly holding the rational n/d.
[[nodiscard]] auto cpoly(std::int64_t n, std::int64_t d) -> RationalPoly {
    return RationalPoly::constant(rat(n, d));
}

// Repeated multiplication (test-side b^n), for reconstruction checks.
[[nodiscard]] auto ipow(const RationalPoly& b, std::int64_t n) -> RationalPoly {
    RationalPoly acc = RationalPoly::constant(Rational::from_int(1));
    for (std::int64_t k = 0; k < n; ++k) {
        acc = acc.multiply(b).value();
    }
    return acc;
}

// Find the numerator over factor^power in a decomposition, if present.
[[nodiscard]] auto find_num(const PartialFraction& pf, const RationalPoly& factor,
                            std::int64_t power) -> std::optional<RationalPoly> {
    for (const auto& term : pf.terms) {
        if (term.power == power && term.factor.is_equal(factor)) {
            return term.numerator;
        }
    }
    return std::nullopt;
}

// Reconstruct A/B == P + sum terms over the common (monic) denominator Bm and compare
// against Am = A / lc(B): Am == P*Bm + sum_k C_k * (Bm / factor_k^power_k).
[[nodiscard]] auto reconstructs(const RationalPoly& a, const RationalPoly& b,
                                const PartialFraction& pf) -> bool {
    const Rational lc = b.leading_coefficient();
    const RationalPoly bm = b.monic().value();
    const RationalPoly am = a.scale(Rational::from_int(1).divide(lc).value()).value();
    RationalPoly acc = pf.polynomial_part.multiply(bm).value();
    for (const auto& term : pf.terms) {
        const RationalPoly cofactor = bm.divide(ipow(term.factor, term.power)).value().quotient;
        acc = acc.add(term.numerator.multiply(cofactor).value()).value();
    }
    return acc.is_equal(am);
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.pfd")
        .test("square_free_distinct_multiplicities",
              [](TestContext& t) {
                  // x^2 (x - 1) = x^3 - x^2  ->  (x - 1)^1 and (x)^2.
                  auto f = square_free_factorization(ipoly({0, 0, -1, 1})).value();
                  t.expect(f.size() == 2, "two square-free factors");
                  // Yun emits factors in ascending multiplicity: (x-1, 1) then (x, 2).
                  t.expect(f[0].second == 1 && f[0].first.is_equal(ipoly({-1, 1})),
                           "multiplicity-1 factor is x - 1");
                  t.expect(f[1].second == 2 && f[1].first.is_equal(ipoly({0, 1})),
                           "multiplicity-2 factor is x");
              })
        .test("square_free_already_square_free",
              [](TestContext& t) {
                  // x^2 - 1 is square-free -> a single factor of multiplicity 1.
                  auto f = square_free_factorization(ipoly({-1, 0, 1})).value();
                  t.expect(f.size() == 1, "one factor");
                  t.expect(f[0].second == 1 && f[0].first.is_equal(ipoly({-1, 0, 1})),
                           "the factor is x^2 - 1 itself");
                  // A constant has no positive-degree factors.
                  t.expect(square_free_factorization(ipoly({5})).value().empty(),
                           "constant has no square-free factors");
              })
        .test("square_free_high_multiplicity",
              [](TestContext& t) {
                  // (x - 1)^3 = x^3 - 3x^2 + 3x - 1  ->  (x - 1) with multiplicity 3.
                  auto f = square_free_factorization(ipoly({-1, 3, -3, 1})).value();
                  t.expect(f.size() == 1, "one distinct factor");
                  t.expect(f[0].second == 3 && f[0].first.is_equal(ipoly({-1, 1})),
                           "(x - 1) with multiplicity 3");
              })
        .test("coprime_same_multiplicity_merge",
              [](TestContext& t) {
                  // 1 / (x^2 - x) = 1/(x(x-1)). Both roots have multiplicity 1, so the
                  // square-free denominator is the composite x^2 - x itself -- the two
                  // coprime linear factors are NOT separated (that needs factoring).
                  auto a = ipoly({1});
                  auto b = ipoly({0, -1, 1});  // x^2 - x = x (x - 1)
                  auto pf = partial_fractions(a, b).value();
                  t.expect(pf.polynomial_part.is_zero(), "proper: no polynomial part");
                  t.expect(pf.terms.size() == 1, "one merged square-free term");
                  auto whole = find_num(pf, ipoly({0, -1, 1}), 1);
                  t.expect(whole && whole->is_equal(cpoly(1, 1)), "numerator over x^2-x is 1");
                  t.expect(reconstructs(a, b, pf), "decomposition reconstructs A/B");
              })
        .test("repeated_root",
              [](TestContext& t) {
                  // (x + 1) / x^2 = 1/x + 1/x^2.
                  auto a = ipoly({1, 1});
                  auto b = ipoly({0, 0, 1});  // x^2
                  auto pf = partial_fractions(a, b).value();
                  t.expect(pf.polynomial_part.is_zero(), "proper: no polynomial part");
                  auto c1 = find_num(pf, ipoly({0, 1}), 1);
                  auto c2 = find_num(pf, ipoly({0, 1}), 2);
                  t.expect(c1 && c1->is_equal(cpoly(1, 1)), "1/x coefficient is 1");
                  t.expect(c2 && c2->is_equal(cpoly(1, 1)), "1/x^2 coefficient is 1");
                  t.expect(reconstructs(a, b, pf), "decomposition reconstructs A/B");
              })
        .test("improper_has_polynomial_part",
              [](TestContext& t) {
                  // x^3 / (x^2 - 1) = x + x/(x^2 - 1). Long division peels off the
                  // polynomial part x; the square-free remainder stays over x^2 - 1.
                  auto a = ipoly({0, 0, 0, 1});
                  auto b = ipoly({-1, 0, 1});  // x^2 - 1
                  auto pf = partial_fractions(a, b).value();
                  t.expect(pf.polynomial_part.is_equal(ipoly({0, 1})), "polynomial part is x");
                  t.expect(pf.terms.size() == 1, "one square-free proper term");
                  auto n = find_num(pf, ipoly({-1, 0, 1}), 1);  // over x^2 - 1
                  t.expect(n && n->is_equal(ipoly({0, 1})), "numerator over x^2-1 is x");
                  t.expect(reconstructs(a, b, pf), "decomposition reconstructs A/B");
              })
        .test("non_monic_denominator",
              [](TestContext& t) {
                  // 1 / (2x - 2) = (1/2)/(x - 1): the leading 2 folds into the numerator.
                  auto a = ipoly({1});
                  auto b = ipoly({-2, 2});  // 2x - 2
                  auto pf = partial_fractions(a, b).value();
                  t.expect(pf.polynomial_part.is_zero(), "proper: no polynomial part");
                  auto n = find_num(pf, ipoly({-1, 1}), 1);
                  t.expect(n && n->is_equal(cpoly(1, 2)), "numerator over (x-1) is 1/2");
                  t.expect(reconstructs(a, b, pf), "decomposition reconstructs A/B");
              })
        .test("mixed_multiplicity",
              [](TestContext& t) {
                  // 1 / (x^2 (x - 1)) = -1/x - 1/x^2 + 1/(x - 1).
                  auto a = ipoly({1});
                  auto b = ipoly({0, 0, -1, 1});  // x^3 - x^2 = x^2 (x - 1)
                  auto pf = partial_fractions(a, b).value();
                  t.expect(pf.polynomial_part.is_zero(), "proper: no polynomial part");
                  auto x1 = find_num(pf, ipoly({0, 1}), 1);
                  auto x2 = find_num(pf, ipoly({0, 1}), 2);
                  auto xm1 = find_num(pf, ipoly({-1, 1}), 1);
                  t.expect(x1 && x1->is_equal(cpoly(-1, 1)), "1/x coefficient is -1");
                  t.expect(x2 && x2->is_equal(cpoly(-1, 1)), "1/x^2 coefficient is -1");
                  t.expect(xm1 && xm1->is_equal(cpoly(1, 1)), "1/(x-1) coefficient is 1");
                  t.expect(reconstructs(a, b, pf), "decomposition reconstructs A/B");
              })
        .test("square_free_is_not_irreducible",
              [](TestContext& t) {
                  // The decomposition is SQUARE-FREE, not irreducible (ROADMAP 7.17): an
                  // already square-free denominator stays whole even when it is composite.
                  // (3x + 1) / (x^3 + x) is proper and square-free -> a single term equal
                  // to the input; x and x^2+1 are NOT separated (that needs factoring).
                  auto a = ipoly({1, 3});
                  auto b = ipoly({0, 1, 0, 1});  // x^3 + x = x (x^2 + 1), square-free
                  auto pf = partial_fractions(a, b).value();
                  t.expect(pf.polynomial_part.is_zero(), "proper: no polynomial part");
                  t.expect(pf.terms.size() == 1, "one square-free term (not split further)");
                  auto whole = find_num(pf, ipoly({0, 1, 0, 1}), 1);
                  t.expect(whole && whole->is_equal(ipoly({1, 3})),
                           "numerator over x^3+x is the original 3x + 1");
                  t.expect(reconstructs(a, b, pf), "decomposition reconstructs A/B");
              })
        .test("repeated_quadratic_factor",
              [](TestContext& t) {
                  // A repeated COMPOSITE square-free factor with linear numerators:
                  // x^3 / (x^2 + 1)^2 = x/(x^2 + 1) - x/(x^2 + 1)^2.
                  auto a = ipoly({0, 0, 0, 1});
                  auto b = ipoly({1, 0, 2, 0, 1});  // (x^2 + 1)^2 = x^4 + 2x^2 + 1
                  auto pf = partial_fractions(a, b).value();
                  t.expect(pf.polynomial_part.is_zero(), "proper: no polynomial part");
                  auto c1 = find_num(pf, ipoly({1, 0, 1}), 1);  // over (x^2 + 1)
                  auto c2 = find_num(pf, ipoly({1, 0, 1}), 2);  // over (x^2 + 1)^2
                  t.expect(c1 && c1->is_equal(ipoly({0, 1})), "numerator over (x^2+1) is x");
                  t.expect(c2 && c2->is_equal(ipoly({0, -1})), "numerator over (x^2+1)^2 is -x");
                  t.expect(reconstructs(a, b, pf), "decomposition reconstructs A/B");
              })
        .test("three_distinct_prime_powers",
              [](TestContext& t) {
                  // Three coprime prime powers of distinct multiplicity exercise the
                  // suffix-product Bezout chain (m = 3): B = x (x-1)^2 (x+1)^3.
                  // x (x-1)^2 = x^3 - 2x^2 + x; (x+1)^3 = x^3 + 3x^2 + 3x + 1.
                  auto a = ipoly({1});
                  auto b = ipoly({0, 1, -2, 1}).multiply(ipoly({1, 3, 3, 1})).value();
                  auto pf = partial_fractions(a, b).value();
                  t.expect(pf.polynomial_part.is_zero(), "proper: no polynomial part");
                  // The split must reach the highest power of each repeated base.
                  t.expect(find_num(pf, ipoly({-1, 1}), 2).has_value(),
                           "term over (x-1)^2 is present");
                  t.expect(find_num(pf, ipoly({1, 1}), 3).has_value(),
                           "term over (x+1)^3 is present");
                  t.expect(find_num(pf, ipoly({0, 1}), 1).has_value(),
                           "term over x is present");
                  t.expect(reconstructs(a, b, pf), "3-factor decomposition reconstructs A/B");
              })
        .test("high_power_expansion",
              [](TestContext& t) {
                  // Multiplicity 3 exercises power_expand past the e=2 digit boundary:
                  // x^2 / (x - 1)^3 = 1/(x-1) + 2/(x-1)^2 + 1/(x-1)^3.
                  auto a = ipoly({0, 0, 1});
                  auto b = ipoly({-1, 3, -3, 1});  // (x - 1)^3
                  auto pf = partial_fractions(a, b).value();
                  t.expect(pf.polynomial_part.is_zero(), "proper: no polynomial part");
                  auto c1 = find_num(pf, ipoly({-1, 1}), 1);
                  auto c2 = find_num(pf, ipoly({-1, 1}), 2);
                  auto c3 = find_num(pf, ipoly({-1, 1}), 3);
                  t.expect(c1 && c1->is_equal(cpoly(1, 1)), "1/(x-1) coefficient is 1");
                  t.expect(c2 && c2->is_equal(cpoly(2, 1)), "1/(x-1)^2 coefficient is 2");
                  t.expect(c3 && c3->is_equal(cpoly(1, 1)), "1/(x-1)^3 coefficient is 1");
                  t.expect(reconstructs(a, b, pf), "decomposition reconstructs A/B");
              })
        .test("errors_and_exact_division",
              [](TestContext& t) {
                  // Zero denominator is a division-by-zero error.
                  t.expect(partial_fractions(ipoly({1}), RationalPoly{}).error() ==
                               MathError::division_by_zero,
                           "zero denominator fails");
                  // A/B with B a constant is purely polynomial: (2x)/2 = x.
                  auto pf = partial_fractions(ipoly({0, 2}), ipoly({2})).value();
                  t.expect(pf.terms.empty(), "constant denominator: no proper terms");
                  t.expect(pf.polynomial_part.is_equal(ipoly({0, 1})), "polynomial part is x");
                  // Non-reduced input still decomposes correctly: x / x^2 = 1/x.
                  auto pf2 = partial_fractions(ipoly({0, 1}), ipoly({0, 0, 1})).value();
                  auto c1 = find_num(pf2, ipoly({0, 1}), 1);
                  t.expect(c1 && c1->is_equal(cpoly(1, 1)), "x/x^2 reduces to 1/x");
                  t.expect(reconstructs(ipoly({0, 1}), ipoly({0, 0, 1}), pf2),
                           "non-reduced input reconstructs");
              })
        .run();
}
