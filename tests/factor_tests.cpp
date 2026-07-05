// Tests for nimblecas.factor: exact factorization of a polynomial over Q into
// irreducibles via the Yun -> Kronecker pipeline.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.ratpoly;
import nimblecas.factor;
import nimblecas.testing;

using nimblecas::factor_over_Q;
using nimblecas::MathError;
using nimblecas::Polynomial;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

using Factorization = std::vector<std::pair<RationalPoly, std::int64_t>>;

// Build a RationalPoly from integer coefficients (low degree first).
[[nodiscard]] auto ipoly(std::vector<std::int64_t> c) -> RationalPoly {
    return RationalPoly::from_polynomial(Polynomial{std::move(c)});
}

// The primitive part of an integer polynomial, as a RationalPoly — the value the returned
// factorization must reproduce when its factors are multiplied out.
[[nodiscard]] auto primitive(std::vector<std::int64_t> c) -> RationalPoly {
    return RationalPoly::from_polynomial(Polynomial{std::move(c)}.primitive_part().value());
}

// Multiply every factor raised to its multiplicity back into a single polynomial.
[[nodiscard]] auto product_of(const Factorization& fs) -> RationalPoly {
    RationalPoly prod = RationalPoly::constant(Rational::from_int(1));
    for (const auto& [f, mult] : fs) {
        for (std::int64_t k = 0; k < mult; ++k) {
            prod = prod.multiply(f).value();
        }
    }
    return prod;
}

// Is there a factor structurally equal to `g` with the given multiplicity?
[[nodiscard]] auto has_factor(const Factorization& fs, const RationalPoly& g,
                              std::int64_t mult) -> bool {
    return std::ranges::any_of(fs, [&](const auto& e) {
        return e.first.is_equal(g) && e.second == mult;
    });
}

// How many factors have the given degree?
[[nodiscard]] auto count_degree(const Factorization& fs, std::int64_t deg) -> std::size_t {
    return static_cast<std::size_t>(std::ranges::count_if(
        fs, [&](const auto& e) { return e.first.degree() == deg; }));
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.factor")
        .test("difference_of_squares_two_linear",
              [](TestContext& t) {
                  // x^2 - 1 = (x - 1)(x + 1)
                  auto f = factor_over_Q(ipoly({-1, 0, 1})).value();
                  t.expect(f.size() == 2, "two irreducible factors");
                  t.expect(has_factor(f, ipoly({-1, 1}), 1), "(x - 1) with multiplicity 1");
                  t.expect(has_factor(f, ipoly({1, 1}), 1), "(x + 1) with multiplicity 1");
                  t.expect(product_of(f).is_equal(primitive({-1, 0, 1})), "product == input");
              })
        .test("irreducible_quadratic",
              [](TestContext& t) {
                  // x^2 - 2: no rational roots, irreducible over Q
                  auto f = factor_over_Q(ipoly({-2, 0, 1})).value();
                  t.expect(f.size() == 1, "a single irreducible factor");
                  t.expect(has_factor(f, ipoly({-2, 0, 1}), 1), "{x^2 - 2} itself");
                  t.expect(product_of(f).is_equal(primitive({-2, 0, 1})), "product == input");
              })
        .test("reducible_quintic_no_rational_roots",
              [](TestContext& t) {
                  // (x^2 - 2)(x^3 - 2) = x^5 - 2x^3 - 2x^2 + 4  (the headline case):
                  // no rational roots, yet splits into a quadratic and a cubic.
                  auto f = factor_over_Q(ipoly({4, 0, -2, -2, 0, 1})).value();
                  t.expect(f.size() == 2, "two irreducible factors");
                  t.expect(count_degree(f, 2) == 1, "one degree-2 factor");
                  t.expect(count_degree(f, 3) == 1, "one degree-3 factor");
                  t.expect(has_factor(f, ipoly({-2, 0, 1}), 1), "(x^2 - 2)");
                  t.expect(has_factor(f, ipoly({-2, 0, 0, 1}), 1), "(x^3 - 2)");
                  t.expect(product_of(f).is_equal(primitive({4, 0, -2, -2, 0, 1})),
                           "product == input");
              })
        .test("repeated_linear_multiplicity",
              [](TestContext& t) {
                  // (x - 1)^2 (x + 2) = x^3 - 3x + 2
                  auto f = factor_over_Q(ipoly({2, -3, 0, 1})).value();
                  t.expect(f.size() == 2, "two distinct irreducible factors");
                  t.expect(has_factor(f, ipoly({-1, 1}), 2), "(x - 1) with multiplicity 2");
                  t.expect(has_factor(f, ipoly({2, 1}), 1), "(x + 2) with multiplicity 1");
                  t.expect(product_of(f).is_equal(primitive({2, -3, 0, 1})), "product == input");
              })
        .test("irreducible_quartic",
              [](TestContext& t) {
                  // x^4 + 1: irreducible over Q
                  auto f = factor_over_Q(ipoly({1, 0, 0, 0, 1})).value();
                  t.expect(f.size() == 1, "a single irreducible factor");
                  t.expect(has_factor(f, ipoly({1, 0, 0, 0, 1}), 1), "{x^4 + 1} itself");
                  t.expect(product_of(f).is_equal(primitive({1, 0, 0, 0, 1})), "product == input");
              })
        .test("quartic_three_irreducibles",
              [](TestContext& t) {
                  // x^4 - 1 = (x - 1)(x + 1)(x^2 + 1)
                  auto f = factor_over_Q(ipoly({-1, 0, 0, 0, 1})).value();
                  t.expect(f.size() == 3, "three irreducible factors");
                  t.expect(count_degree(f, 1) == 2, "two linear factors");
                  t.expect(count_degree(f, 2) == 1, "one quadratic factor");
                  t.expect(has_factor(f, ipoly({-1, 1}), 1), "(x - 1)");
                  t.expect(has_factor(f, ipoly({1, 1}), 1), "(x + 1)");
                  t.expect(has_factor(f, ipoly({1, 0, 1}), 1), "(x^2 + 1)");
                  t.expect(product_of(f).is_equal(primitive({-1, 0, 0, 0, 1})), "product == input");
              })
        .test("reducible_sextic_three_quadratics",
              [](TestContext& t) {
                  // (x^2 + 1)(x^2 + x + 1)(x^2 - 2)
                  //   = x^6 + x^5 - x^3 - 3x^2 - 2x - 2
                  auto f = factor_over_Q(ipoly({-2, -2, -3, -1, 0, 1, 1})).value();
                  t.expect(f.size() == 3, "three irreducible factors");
                  t.expect(count_degree(f, 2) == 3, "all three factors are degree 2");
                  t.expect(has_factor(f, ipoly({1, 0, 1}), 1), "(x^2 + 1)");
                  t.expect(has_factor(f, ipoly({1, 1, 1}), 1), "(x^2 + x + 1)");
                  t.expect(has_factor(f, ipoly({-2, 0, 1}), 1), "(x^2 - 2)");
                  t.expect(product_of(f).is_equal(primitive({-2, -2, -3, -1, 0, 1, 1})),
                           "product == input");
              })
        .test("degenerate_inputs",
              [](TestContext& t) {
                  // zero polynomial: every value is a root -> domain_error
                  t.expect(factor_over_Q(RationalPoly{}).error() == MathError::domain_error,
                           "zero polynomial is a domain error");
                  // nonzero constant: no non-unit factors -> empty list
                  auto f = factor_over_Q(ipoly({5})).value();
                  t.expect(f.empty(), "nonzero constant factors into an empty list");
              })
        .run();
}
