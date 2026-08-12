// Tests for nimblecas.bigfrobenius: unbounded invariant factors / minimal polynomial over Q.
// @author Olumuyiwa Oluwasanmi
//
// Oracles are hand-verified over Q and mirror the int64 nimblecas.frobenius tests. The
// load-bearing case is a diagonal matrix diag(B, B) with B = 10^20: its minimal polynomial
// (x - B) has the coefficient -10^20, which dwarfs the int64 ceiling (~9.22e18). The int64
// frobenius tier could not represent that coefficient at all; the bignum tier returns it
// exactly, which is the whole reason this module exists.

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.bigratpoly;
import nimblecas.bigmatrix;
import nimblecas.bigfrobenius;
import nimblecas.testing;

using nimblecas::BigMatrix;
using nimblecas::BigRational;
using nimblecas::BigRationalPoly;
using nimblecas::MathError;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// A BigRational from a decimal/fraction literal (all literals below are valid by
// construction, so the dereference never faults).
auto big(std::string_view s) -> BigRational {
    auto r = BigRational::from_string(s);
    return *r;
}

auto bi(std::int64_t v) -> BigRational {
    return BigRational::from_int(v);
}

// A BigRationalPoly from low-degree-first coefficients.
auto poly(std::vector<BigRational> c) -> BigRationalPoly {
    return BigRationalPoly::from_coeffs(std::move(c));
}

// A BigMatrix from rows (every oracle matrix below is well-formed, so the dereference is
// safe).
auto mat(std::vector<std::vector<BigRational>> rows) -> BigMatrix {
    auto m = BigMatrix::from_rows(std::move(rows));
    return *m;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.bigfrobenius")
        .test("minimal_polynomial of diag(2,2,3) is (x-2)(x-3) = x^2 - 5x + 6",
              [](TestContext& t) {
                  // A diagonalizable matrix's minimal polynomial is the product of its
                  // DISTINCT eigenvalue factors: (x - 2)(x - 3), not (x - 2)^2 (x - 3).
                  const BigMatrix a = mat({{bi(2), bi(0), bi(0)},
                                           {bi(0), bi(2), bi(0)},
                                           {bi(0), bi(0), bi(3)}});
                  auto r = nimblecas::minimal_polynomial(a);
                  const bool ok = r.has_value();
                  t.expect(ok, "minimal_polynomial of diag(2,2,3) succeeded");
                  if (!ok) return;
                  t.expect(r->is_equal(poly({bi(6), bi(-5), bi(1)})),
                           "min poly is x^2 - 5x + 6");
              })
        .test("minimal_polynomial of the Jordan block [[5,1],[0,5]] is (x-5)^2",
              [](TestContext& t) {
                  // A single 2x2 Jordan block is non-diagonalizable: its minimal polynomial
                  // is the full (x - 5)^2 = x^2 - 10x + 25.
                  const BigMatrix a = mat({{bi(5), bi(1)}, {bi(0), bi(5)}});
                  auto r = nimblecas::minimal_polynomial(a);
                  const bool ok = r.has_value();
                  t.expect(ok, "minimal_polynomial of the Jordan block succeeded");
                  if (!ok) return;
                  t.expect(r->is_equal(poly({bi(25), bi(-10), bi(1)})),
                           "min poly is x^2 - 10x + 25");
              })
        .test("invariant_factors of the 2x2 identity are (x-1), (x-1)",
              [](TestContext& t) {
                  // x*I - I = diag(x - 1, x - 1) is already in Smith form: both diagonal
                  // entries are the non-constant (x - 1), so both are invariant factors.
                  const BigMatrix a = mat({{bi(1), bi(0)}, {bi(0), bi(1)}});
                  auto r = nimblecas::invariant_factors(a);
                  const bool ok = r.has_value();
                  t.expect(ok, "invariant_factors of the identity succeeded");
                  if (!ok) return;
                  t.expect(r->size() == 2, "the 2x2 identity has two invariant factors");
                  if (r->size() != 2) return;
                  t.expect((*r)[0].is_equal(poly({bi(-1), bi(1)})), "first factor is x - 1");
                  t.expect((*r)[1].is_equal(poly({bi(-1), bi(1)})), "second factor is x - 1");
              })
        .test("minimal_polynomial past the int64 ceiling stays exact (bignum advantage)",
              [](TestContext& t) {
                  // diag(B, B) with B = 10^20 is the scalar matrix B*I; its minimal
                  // polynomial is (x - B). The coefficient -10^20 is far beyond the int64
                  // ceiling (~9.22e18), so the int64 frobenius tier could not hold it; the
                  // bignum tier returns it exactly.
                  const BigRational B = big("100000000000000000000");  // 10^20
                  const BigMatrix a = mat({{B, bi(0)}, {bi(0), B}});
                  auto r = nimblecas::minimal_polynomial(a);
                  const bool ok = r.has_value();
                  t.expect(ok, "minimal_polynomial of diag(1e20, 1e20) succeeded");
                  if (!ok) return;
                  // x - 10^20, coefficient beyond the int64 tier.
                  t.expect(r->is_equal(poly({big("-100000000000000000000"), bi(1)})),
                           "min poly is x - 10^20 computed exactly");
                  t.expect(r->coefficient(0) == big("-100000000000000000000"),
                           "constant coefficient is exactly -1e20");
              })
        .test("a non-square matrix is a domain_error",
              [](TestContext& t) {
                  const BigMatrix a = mat({{bi(1), bi(2), bi(3)}, {bi(4), bi(5), bi(6)}});
                  auto factors = nimblecas::invariant_factors(a);
                  t.expect(!factors.has_value() && factors.error() == MathError::domain_error,
                           "invariant_factors of a 2x3 matrix -> domain_error");
                  auto mp = nimblecas::minimal_polynomial(a);
                  t.expect(!mp.has_value() && mp.error() == MathError::domain_error,
                           "minimal_polynomial of a 2x3 matrix -> domain_error");
              })
        .run();
}
