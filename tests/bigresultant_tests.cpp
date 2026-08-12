// Tests for nimblecas.bigresultant: resultant and discriminant over Q[x] on BigRational.
// @author Olumuyiwa Oluwasanmi
//
// Oracles are hand-verified. The load-bearing case is Res(x^2 - a, x^2 - b) = (a - b)^2
// with a, b of magnitude ~1e10: the exact result 1e20 overflows int64, so the int64
// nimblecas.resultant would report MathError::overflow where the bignum tier returns the
// exact value. Small textbook resultants, a discriminant (Res(f, f') form), and the
// zero-polynomial / constant conventions round out the coverage.

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.bigratpoly;
import nimblecas.bigresultant;
import nimblecas.testing;

using nimblecas::BigRational;
using nimblecas::BigRationalPoly;
using nimblecas::MathError;
using nimblecas::discriminant;
using nimblecas::resultant;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Build a BigRationalPoly from integer coefficients in index order (constant first).
auto bigpoly(std::initializer_list<std::int64_t> cs) -> BigRationalPoly {
    std::vector<BigRational> v;
    v.reserve(cs.size());
    for (const std::int64_t c : cs) {
        v.push_back(BigRational::from_int(c));
    }
    return BigRationalPoly::from_coeffs(std::move(v));
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.bigresultant")
        .test("Res(x - 1, x - 2) == -1",
              [](TestContext& t) {
                  auto r = resultant(bigpoly({-1, 1}), bigpoly({-2, 1}));
                  t.expect(r.has_value(), "resultant computed");
                  if (!r) return;
                  t.expect(*r == BigRational::from_int(-1), "Res(x-1, x-2) == -1");
              })
        .test("Res(x^2 - 1, x - 1) == 0 (shared factor x - 1)",
              [](TestContext& t) {
                  auto r = resultant(bigpoly({-1, 0, 1}), bigpoly({-1, 1}));
                  t.expect(r.has_value(), "resultant computed");
                  if (!r) return;
                  t.expect(r->is_zero(), "common factor => resultant 0");
              })
        .test("Res(x^2 - a, x^2 - b) == (a - b)^2 with int64-overflowing oracle",
              [](TestContext& t) {
                  // a = 1e10, b = 2e10 (both fit int64); a - b = -1e10, so (a-b)^2 = 1e20,
                  // which exceeds int64 max (~9.22e18) — the bignum tier returns it exactly.
                  constexpr std::int64_t a = 10000000000LL;
                  constexpr std::int64_t b = 20000000000LL;
                  auto r = resultant(bigpoly({-a, 0, 1}), bigpoly({-b, 0, 1}));
                  t.expect(r.has_value(), "resultant computed");
                  if (!r) return;
                  auto oracle = BigRational::from_string("100000000000000000000");  // 1e20
                  t.expect(oracle.has_value(), "oracle parsed");
                  if (!oracle) return;
                  t.expect(*r == *oracle, "Res == (a - b)^2 == 1e20 (exact, > int64)");
              })
        .test("Res(f, f') via discriminant: disc(x^2 - 1) == 4",
              [](TestContext& t) {
                  // disc(x^2 + bx + c) = b^2 - 4c; for x^2 - 1 that is 0 - 4(-1) = 4. This
                  // exercises resultant(a, a') = res(x^2 - 1, 2x) = -4 internally.
                  auto d = discriminant(bigpoly({-1, 0, 1}));
                  t.expect(d.has_value(), "discriminant computed");
                  if (!d) return;
                  t.expect(*d == BigRational::from_int(4), "disc(x^2 - 1) == 4");
              })
        .test("small Sylvester oracle: Res(x^2 - x - 2, 2x - 1) == -9",
              [](TestContext& t) {
                  // A = x^2 - x - 2 = (x-2)(x+1), B = 2x - 1 (root 1/2). Res(A,B) =
                  // lc(A)^{deg B} * prod_{A(r)=0} ... equivalently lc(B)^{deg A} * A(1/2) with
                  // sign: A(1/2) = 1/4 - 1/2 - 2 = -9/4; lc(B)^{deg A} = 2^2 = 4; 4*(-9/4) = -9.
                  // Hand Sylvester det of [[1,-1,-2],[2,-1,0],[0,2,-1]] = -9.
                  auto r = resultant(bigpoly({-2, -1, 1}), bigpoly({-1, 2}));
                  t.expect(r.has_value(), "resultant computed");
                  if (!r) return;
                  t.expect(*r == BigRational::from_int(-9), "Res(x^2-x-2, 2x-1) == -9");
              })
        .test("degenerate: zero polynomial gives resultant 0, constants give 1",
              [](TestContext& t) {
                  auto rz = resultant(BigRationalPoly::zero(), bigpoly({-1, 1}));
                  t.expect(rz.has_value(), "resultant with zero poly computed");
                  if (!rz) return;
                  t.expect(rz->is_zero(), "Res(0, x - 1) == 0");
                  // res(constant, constant) is the empty product 1.
                  auto rc = resultant(bigpoly({5}), bigpoly({7}));
                  t.expect(rc.has_value(), "resultant of constants computed");
                  if (!rc) return;
                  t.expect(*rc == BigRational::from_int(1), "Res(5, 7) == 1");
              })
        .run();
}
