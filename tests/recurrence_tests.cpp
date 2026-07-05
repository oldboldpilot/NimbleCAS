// Tests for nimblecas.recurrence: characteristic polynomials and rational roots of
// linear homogeneous constant-coefficient recurrences.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.ratpoly;
import nimblecas.recurrence;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.testing;

using nimblecas::Expr;
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

// Evaluate a closed-form Expr a(n) at an integer n: substitute n and fold the resulting
// all-constant tree to a single number via simplify. With integer inputs the result is an
// integer leaf, so it compares equal to Expr::integer(expected).
[[nodiscard]] auto value_at(const Expr& closed, std::int64_t n) -> Expr {
    const Expr subst = nimblecas::substitute(closed, Expr::symbol("n"), Expr::integer(n));
    return nimblecas::simplify(subst).value();
}

// Whether a(n) reproduces the given first terms (values[i] == a(i), integer-valued).
[[nodiscard]] auto reproduces(TestContext& t, const Expr& closed,
                              std::vector<std::int64_t> values) -> void {
    for (std::size_t n = 0; n < values.size(); ++n) {
        const Expr got = value_at(closed, static_cast<std::int64_t>(n));
        t.expect(got.is_equivalent_to(Expr::integer(values[n])),
                 std::format("a({}) == {} (got {})", n, values[n], got.to_string()));
    }
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
        .test("closed_form_distinct_roots",
              [](TestContext& t) {
                  // a_n = 5 a_{n-1} - 6 a_{n-2}, a_0 = 0, a_1 = 1 => x^2 - 5x + 6.
                  //   general A*2^n + B*3^n; A = -1, B = 1  =>  a_n = 3^n - 2^n.
                  const auto c = coeffs({5, -6});
                  const auto init = coeffs({0, 1});
                  auto cf = nimblecas::closed_form(c, init).value();
                  reproduces(t, cf, {0, 1, 5, 19, 65, 211});
              })
        .test("closed_form_negative_root",
              [](TestContext& t) {
                  // a_n = a_{n-1} + 2 a_{n-2}, a_0 = 2, a_1 = 1 => x^2 - x - 2 = (x-2)(x+1).
                  //   A*2^n + B*(-1)^n with A = 1, B = 1  =>  a_n = 2^n + (-1)^n.
                  const auto c = coeffs({1, 2});
                  const auto init = coeffs({2, 1});
                  auto cf = nimblecas::closed_form(c, init).value();
                  reproduces(t, cf, {2, 1, 5, 7, 17, 31});
              })
        .test("closed_form_repeated_root",
              [](TestContext& t) {
                  // a_n = 4 a_{n-1} - 4 a_{n-2}, a_0 = 1, a_1 = 4 => x^2 - 4x + 4 = (x-2)^2.
                  //   (A + B n) 2^n with A = 1, B = 1  =>  a_n = (n + 1) 2^n.
                  const auto c = coeffs({4, -4});
                  const auto init = coeffs({1, 4});
                  auto cf = nimblecas::closed_form(c, init).value();
                  reproduces(t, cf, {1, 4, 12, 32, 80, 192});
              })
        .test("closed_form_cubic_three_roots",
              [](TestContext& t) {
                  // a_n = 6 a_{n-1} - 11 a_{n-2} + 6 a_{n-3} => (x-1)(x-2)(x-3).
                  //   a_0 = 0, a_1 = 1, a_2 = 5 selects a_n = A + B*2^n + C*3^n; verify by
                  //   matching the sequence the recurrence itself generates.
                  const auto c = coeffs({6, -11, 6});
                  const auto init = coeffs({0, 1, 5});
                  auto cf = nimblecas::closed_form(c, init).value();
                  // 0,1,5 then a_n = 6 a_{n-1} - 11 a_{n-2} + 6 a_{n-3}:
                  //   a_3 = 6*5 - 11*1 + 6*0 = 19; a_4 = 6*19 - 11*5 + 6*1 = 65;
                  //   a_5 = 6*65 - 11*19 + 6*5 = 211.
                  reproduces(t, cf, {0, 1, 5, 19, 65, 211});
              })
        .test("closed_form_fibonacci_not_implemented",
              [](TestContext& t) {
                  // Fibonacci x^2 - x - 1 does not split over Q: no rational closed form.
                  const auto c = coeffs({1, 1});
                  const auto init = coeffs({0, 1});
                  t.expect(nimblecas::closed_form(c, init).error() ==
                               MathError::not_implemented,
                           "irrational roots -> not_implemented (no wrong closed form)");
              })
        .test("closed_form_length_mismatch_is_domain_error",
              [](TestContext& t) {
                  const auto c = coeffs({5, -6});
                  const auto init = coeffs({1});  // needs 2 initial conditions
                  t.expect(nimblecas::closed_form(c, init).error() == MathError::domain_error,
                           "wrong number of initial conditions -> domain_error");
                  const std::vector<Rational> none;
                  t.expect(nimblecas::closed_form(std::span<const Rational>{},
                                                  std::span<const Rational>{none})
                                   .error() == MathError::domain_error,
                           "empty coeffs -> domain_error");
              })
        .test("generating_function_fibonacci",
              [](TestContext& t) {
                  // Fibonacci GF is x / (1 - x - x^2) — exact even though the roots are
                  // irrational (its advantage over the closed form).
                  const auto c = coeffs({1, 1});
                  const auto init = coeffs({0, 1});
                  auto gf = nimblecas::generating_function(c, init).value();
                  t.expect(gf.numerator.is_equal(ipoly({0, 1})), "numerator is x");
                  t.expect(gf.denominator.is_equal(ipoly({1, -1, -1})),
                           "denominator is 1 - x - x^2");
              })
        .test("generating_function_distinct_roots",
              [](TestContext& t) {
                  // a_n = 5 a_{n-1} - 6 a_{n-2}, a_0 = 0, a_1 = 1: G = x / (1 - 5x + 6x^2)
                  //   (= 1/(1-3x) - 1/(1-2x), the GF of 3^n - 2^n).
                  const auto c = coeffs({5, -6});
                  const auto init = coeffs({0, 1});
                  auto gf = nimblecas::generating_function(c, init).value();
                  t.expect(gf.numerator.is_equal(ipoly({0, 1})), "numerator is x");
                  t.expect(gf.denominator.is_equal(ipoly({1, -5, 6})),
                           "denominator is 1 - 5x + 6x^2");
              })
        .test("generating_function_domain_errors",
              [](TestContext& t) {
                  const auto c = coeffs({5, -6});
                  const auto init = coeffs({1});
                  t.expect(nimblecas::generating_function(c, init).error() ==
                               MathError::domain_error,
                           "mismatched initial length -> domain_error");
                  const std::vector<Rational> none;
                  t.expect(nimblecas::generating_function(std::span<const Rational>{},
                                                          std::span<const Rational>{none})
                                   .error() == MathError::domain_error,
                           "empty coeffs -> domain_error");
              })
        .run();
}
