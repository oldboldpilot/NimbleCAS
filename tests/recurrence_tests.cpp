// Tests for nimblecas.recurrence: characteristic polynomials and rational roots of
// linear homogeneous constant-coefficient recurrences.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.ratpoly;
import nimblecas.recurrence;
import nimblecas.testing;

using nimblecas::MathError;
using nimblecas::Polynomial;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Build a RationalPoly from integer coefficients (low degree first).
[[nodiscard]] auto ipoly(std::vector<std::int64_t> c) -> RationalPoly {
    return RationalPoly::from_polynomial(Polynomial{std::move(c)});
}

// Coefficient list {c_0, ..., c_{k-1}} as a vector of integer-valued Rationals.
[[nodiscard]] auto coeffs(std::vector<std::int64_t> c) -> std::vector<Rational> {
    std::vector<Rational> out;
    out.reserve(c.size());
    for (const std::int64_t v : c) {
        out.push_back(Rational::from_int(v));
    }
    return out;
}

// Whether `roots` contains exactly the given (root, multiplicity) set, order-independent.
[[nodiscard]] auto roots_equal(const std::vector<std::pair<Rational, std::int64_t>>& roots,
                               std::vector<std::pair<Rational, std::int64_t>> expected)
    -> bool {
    if (roots.size() != expected.size()) {
        return false;
    }
    for (const auto& [r, m] : roots) {
        const auto it = std::ranges::find_if(
            expected, [&](const auto& e) { return e.first == r && e.second == m; });
        if (it == expected.end()) {
            return false;
        }
        expected.erase(it);
    }
    return expected.empty();
}

[[nodiscard]] auto ratroot(std::int64_t v, std::int64_t mult)
    -> std::pair<Rational, std::int64_t> {
    return {Rational::from_int(v), mult};
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.recurrence")
        .test("distinct_rational_roots",
              [](TestContext& t) {
                  // a_n = 5 a_{n-1} - 6 a_{n-2}  =>  x^2 - 5x + 6 = (x-2)(x-3).
                  const auto c = coeffs({5, -6});
                  const std::span<const Rational> s{c};
                  auto poly = nimblecas::characteristic_polynomial(s).value();
                  t.expect(poly.is_equal(ipoly({6, -5, 1})), "char poly is x^2 - 5x + 6");

                  auto roots = nimblecas::characteristic_roots(s).value();
                  t.expect(roots_equal(roots, {ratroot(2, 1), ratroot(3, 1)}),
                           "characteristic roots are {2:1, 3:1}");
                  t.expect(nimblecas::all_roots_rational(s).value(),
                           "splits completely over Q");
              })
        .test("repeated_rational_root",
              [](TestContext& t) {
                  // a_n = 2 a_{n-1} - a_{n-2}  =>  x^2 - 2x + 1 = (x-1)^2.
                  const auto c = coeffs({2, -1});
                  const std::span<const Rational> s{c};
                  auto poly = nimblecas::characteristic_polynomial(s).value();
                  t.expect(poly.is_equal(ipoly({1, -2, 1})), "char poly is (x-1)^2");

                  auto roots = nimblecas::characteristic_roots(s).value();
                  t.expect(roots_equal(roots, {ratroot(1, 2)}),
                           "characteristic root is {1:2}");
                  t.expect(nimblecas::all_roots_rational(s).value(),
                           "double root still splits over Q");
              })
        .test("fibonacci_irrational_roots",
              [](TestContext& t) {
                  // Fibonacci a_n = a_{n-1} + a_{n-2}  =>  x^2 - x - 1 (golden ratio roots).
                  const auto c = coeffs({1, 1});
                  const std::span<const Rational> s{c};
                  auto poly = nimblecas::characteristic_polynomial(s).value();
                  t.expect(poly.is_equal(ipoly({-1, -1, 1})), "char poly is x^2 - x - 1");

                  auto roots = nimblecas::characteristic_roots(s).value();
                  t.expect(roots.empty(), "no rational characteristic roots");
                  t.expect(!nimblecas::all_roots_rational(s).value(),
                           "does not split over Q (irrational roots)");
              })
        .test("cubic_three_distinct_roots",
              [](TestContext& t) {
                  // a_n = 6 a_{n-1} - 11 a_{n-2} + 6 a_{n-3}
                  //   =>  x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3).
                  const auto c = coeffs({6, -11, 6});
                  const std::span<const Rational> s{c};
                  auto poly = nimblecas::characteristic_polynomial(s).value();
                  t.expect(poly.is_equal(ipoly({-6, 11, -6, 1})),
                           "char poly is x^3 - 6x^2 + 11x - 6");

                  auto roots = nimblecas::characteristic_roots(s).value();
                  t.expect(roots_equal(roots, {ratroot(1, 1), ratroot(2, 1), ratroot(3, 1)}),
                           "characteristic roots are {1:1, 2:1, 3:1}");
                  t.expect(nimblecas::all_roots_rational(s).value(),
                           "cubic splits completely over Q");
              })
        .test("empty_coeffs_is_domain_error",
              [](TestContext& t) {
                  const std::vector<Rational> empty;
                  const std::span<const Rational> s{empty};
                  t.expect(nimblecas::characteristic_polynomial(s).error() ==
                               MathError::domain_error,
                           "empty coeffs -> domain_error (polynomial)");
                  t.expect(nimblecas::characteristic_roots(s).error() ==
                               MathError::domain_error,
                           "empty coeffs -> domain_error (roots)");
                  t.expect(nimblecas::all_roots_rational(s).error() ==
                               MathError::domain_error,
                           "empty coeffs -> domain_error (all_roots_rational)");
              })
        .run();
}
