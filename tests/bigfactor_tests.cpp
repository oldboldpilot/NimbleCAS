// Tests for nimblecas.bigfactor: exact factorization of a polynomial over Q into
// irreducibles via the Yun -> Kronecker pipeline, on the UNBOUNDED BigRational.
// @author Olumuyiwa Oluwasanmi
//
// Every oracle below is hand-verified. The headline bignum case is
// (x - 2^32)(x + 2^32) = x^2 - 2^64, whose constant term 2^64 = 18446744073709551616
// exceeds the int64 ceiling (INT64_MAX = 9223372036854775807) yet factors cleanly over
// BigRational -- the whole point of the bignum tier. Kronecker's degree-1 search factors
// f(0) = -2^64 = -(2^64) and f(1) = 1 - 2^64 = -(2^64 - 1); the divisors it needs,
// 2^32 | 2^64 and (2^32 - 1) | (2^64 - 1), both exist, so the interpolant x - 2^32 is
// recovered exactly.

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.bigratpoly;
import nimblecas.bigfactor;
import nimblecas.testing;

using nimblecas::BigInt;
using nimblecas::BigRational;
using nimblecas::BigRationalPoly;
using nimblecas::factor_over_Q;
using nimblecas::MathError;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

using Factorization = std::vector<std::pair<BigRationalPoly, std::int64_t>>;

// A BigInt from a signed decimal string (test literals are always well-formed).
[[nodiscard]] auto bi(std::string_view s) -> BigInt {
    return BigInt::from_string(s).value();
}

// A BigRationalPoly from signed decimal integer coefficients, low degree first.
[[nodiscard]] auto bpoly(std::initializer_list<std::string_view> cs) -> BigRationalPoly {
    std::vector<BigRational> v;
    v.reserve(cs.size());
    for (const std::string_view s : cs) {
        v.push_back(BigRational::from_bigint(bi(s)));
    }
    return BigRationalPoly::from_coeffs(std::move(v));
}

// Multiply every factor raised to its multiplicity back into a single polynomial.
[[nodiscard]] auto product_of(const Factorization& fs) -> BigRationalPoly {
    BigRationalPoly prod = BigRationalPoly::constant(BigRational::from_int(1));
    for (const auto& [f, mult] : fs) {
        for (std::int64_t k = 0; k < mult; ++k) {
            prod = prod.multiply(f);
        }
    }
    return prod;
}

// Is there a factor structurally equal to `g` with the given multiplicity?
[[nodiscard]] auto has_factor(const Factorization& fs, const BigRationalPoly& g,
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
    return TestSuite("nimblecas.bigfactor")
        .test("difference_of_squares_two_linear",
              [](TestContext& t) {
                  // x^2 - 1 = (x - 1)(x + 1)
                  auto r = factor_over_Q(bpoly({"-1", "0", "1"}));
                  const bool ok = r.has_value();
                  t.expect(ok, "factored x^2 - 1");
                  if (!ok) {
                      return;
                  }
                  const Factorization& f = *r;
                  t.expect(f.size() == 2, "two irreducible factors");
                  t.expect(has_factor(f, bpoly({"-1", "1"}), 1), "(x - 1) with multiplicity 1");
                  t.expect(has_factor(f, bpoly({"1", "1"}), 1), "(x + 1) with multiplicity 1");
                  t.expect(product_of(f).is_equal(bpoly({"-1", "0", "1"})), "product == input");
              })
        .test("reducible_quintic_no_rational_roots",
              [](TestContext& t) {
                  // (x^2 - 2)(x^3 - 2) = x^5 - 2x^3 - 2x^2 + 4: square-free, no rational
                  // roots, yet splits into a quadratic and a cubic.
                  auto r = factor_over_Q(bpoly({"4", "0", "-2", "-2", "0", "1"}));
                  const bool ok = r.has_value();
                  t.expect(ok, "factored the quintic");
                  if (!ok) {
                      return;
                  }
                  const Factorization& f = *r;
                  t.expect(f.size() == 2, "two irreducible factors");
                  t.expect(count_degree(f, 2) == 1, "one degree-2 factor");
                  t.expect(count_degree(f, 3) == 1, "one degree-3 factor");
                  t.expect(has_factor(f, bpoly({"-2", "0", "1"}), 1), "(x^2 - 2)");
                  t.expect(has_factor(f, bpoly({"-2", "0", "0", "1"}), 1), "(x^3 - 2)");
                  t.expect(product_of(f).is_equal(bpoly({"4", "0", "-2", "-2", "0", "1"})),
                           "product == input");
              })
        .test("irreducible_quadratic_stays_whole",
              [](TestContext& t) {
                  // x^2 + 1: no rational roots, irreducible over Q -> returned whole.
                  auto r = factor_over_Q(bpoly({"1", "0", "1"}));
                  const bool ok = r.has_value();
                  t.expect(ok, "factored x^2 + 1");
                  if (!ok) {
                      return;
                  }
                  const Factorization& f = *r;
                  t.expect(f.size() == 1, "a single irreducible factor");
                  t.expect(has_factor(f, bpoly({"1", "0", "1"}), 1), "{x^2 + 1} itself");
                  t.expect(product_of(f).is_equal(bpoly({"1", "0", "1"})), "product == input");
              })
        .test("repeated_linear_multiplicity",
              [](TestContext& t) {
                  // (x - 1)^2 = x^2 - 2x + 1: one factor (x - 1) with multiplicity 2.
                  auto r = factor_over_Q(bpoly({"1", "-2", "1"}));
                  const bool ok = r.has_value();
                  t.expect(ok, "factored (x - 1)^2");
                  if (!ok) {
                      return;
                  }
                  const Factorization& f = *r;
                  t.expect(f.size() == 1, "a single distinct irreducible factor");
                  t.expect(has_factor(f, bpoly({"-1", "1"}), 2), "(x - 1) with multiplicity 2");
                  t.expect(product_of(f).is_equal(bpoly({"1", "-2", "1"})), "product == input");
              })
        .test("bignum_constant_exceeds_int64",
              [](TestContext& t) {
                  // (x - 2^32)(x + 2^32) = x^2 - 2^64. The constant term 2^64 =
                  // 18446744073709551616 overflows int64 (max 9223372036854775807), yet the
                  // exact bignum factorization recovers x - 4294967296 and x + 4294967296.
                  auto r = factor_over_Q(bpoly({"-18446744073709551616", "0", "1"}));
                  const bool ok = r.has_value();
                  t.expect(ok, "factored x^2 - 2^64 in bignum");
                  if (!ok) {
                      return;
                  }
                  const Factorization& f = *r;
                  t.expect(f.size() == 2, "two irreducible factors");
                  t.expect(count_degree(f, 1) == 2, "both factors are linear");
                  t.expect(has_factor(f, bpoly({"-4294967296", "1"}), 1),
                           "(x - 4294967296) with multiplicity 1");
                  t.expect(has_factor(f, bpoly({"4294967296", "1"}), 1),
                           "(x + 4294967296) with multiplicity 1");
                  t.expect(
                      product_of(f).is_equal(bpoly({"-18446744073709551616", "0", "1"})),
                      "product == x^2 - 2^64");
              })
        .test("degenerate_inputs",
              [](TestContext& t) {
                  // zero polynomial: every value is a root -> domain_error
                  auto z = factor_over_Q(BigRationalPoly::zero());
                  t.expect(!z.has_value() && z.error() == MathError::domain_error,
                           "zero polynomial is a domain error");
                  // nonzero constant: no non-unit factors -> empty list
                  auto c = factor_over_Q(bpoly({"5"}));
                  const bool ok = c.has_value();
                  t.expect(ok, "constant factored");
                  if (!ok) {
                      return;
                  }
                  t.expect(c->empty(), "nonzero constant factors into an empty list");
              })
        .run();
}
