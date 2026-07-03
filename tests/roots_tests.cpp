// Tests for nimblecas.roots: rational roots via the rational root theorem.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.ratpoly;
import nimblecas.roots;
import nimblecas.testing;

using nimblecas::MathError;
using nimblecas::Polynomial;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::rational_roots;
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

// Order of the returned roots is unspecified, so search for an expected pair.
[[nodiscard]] auto has_root(const std::vector<std::pair<Rational, std::int64_t>>& roots,
                            const Rational& r, std::int64_t mult) -> bool {
    return std::ranges::any_of(roots, [&](const auto& e) {
        return e.first == r && e.second == mult;
    });
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.roots")
        .test("three_simple_roots",
              [](TestContext& t) {
                  // (x-1)(x-2)(x+3) = x^3 - 7x + 6
                  auto roots = rational_roots(ipoly({6, -7, 0, 1})).value();
                  t.expect(roots.size() == 3, "three distinct roots");
                  t.expect(has_root(roots, Rational::from_int(1), 1), "root 1 (mult 1)");
                  t.expect(has_root(roots, Rational::from_int(2), 1), "root 2 (mult 1)");
                  t.expect(has_root(roots, Rational::from_int(-3), 1), "root -3 (mult 1)");
              })
        .test("repeated_root_multiplicity",
              [](TestContext& t) {
                  // (x-1)^2 (x+2) = x^3 - 3x + 2
                  auto roots = rational_roots(ipoly({2, 0, -3, 1})).value();
                  t.expect(roots.size() == 2, "two distinct roots");
                  t.expect(has_root(roots, Rational::from_int(1), 2), "root 1 has multiplicity 2");
                  t.expect(has_root(roots, Rational::from_int(-2), 1), "root -2 (mult 1)");
              })
        .test("irrational_has_no_rational_roots",
              [](TestContext& t) {
                  // x^2 + 1: complex roots, none rational
                  auto roots = rational_roots(ipoly({1, 0, 1})).value();
                  t.expect(roots.empty(), "x^2 + 1 has no rational roots");
              })
        .test("common_factor_cleared",
              [](TestContext& t) {
                  // 2x^2 - 2 = 2(x-1)(x+1)
                  auto roots = rational_roots(ipoly({-2, 0, 2})).value();
                  t.expect(roots.size() == 2, "two roots after clearing the factor 2");
                  t.expect(has_root(roots, Rational::from_int(1), 1), "root 1 (mult 1)");
                  t.expect(has_root(roots, Rational::from_int(-1), 1), "root -1 (mult 1)");
              })
        .test("fractional_root",
              [](TestContext& t) {
                  // 2x - 1: root 1/2
                  auto roots = rational_roots(ipoly({-1, 2})).value();
                  t.expect(roots.size() == 1, "one root");
                  t.expect(has_root(roots, rat(1, 2), 1), "root 1/2 (mult 1)");
              })
        .test("zero_root_with_multiplicity",
              [](TestContext& t) {
                  // x^2 (x - 1): roots 0 (mult 2) and 1
                  auto roots = rational_roots(ipoly({0, 0, -1, 1})).value();
                  t.expect(roots.size() == 2, "two distinct roots");
                  t.expect(has_root(roots, Rational{}, 2), "root 0 has multiplicity 2");
                  t.expect(has_root(roots, Rational::from_int(1), 1), "root 1 (mult 1)");
              })
        .test("degenerate_inputs",
              [](TestContext& t) {
                  // zero polynomial: every value is a root -> domain_error
                  t.expect(rational_roots(RationalPoly{}).error() == MathError::domain_error,
                           "zero polynomial is a domain error");
                  // nonzero constant: no roots
                  auto roots = rational_roots(ipoly({5})).value();
                  t.expect(roots.empty(), "nonzero constant has no roots");
              })
        .run();
}
