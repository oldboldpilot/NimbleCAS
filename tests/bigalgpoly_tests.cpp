// Tests for nimblecas.bigalgpoly: dense univariate polynomials over the algebraic number
// field Q(alpha) on BigRational — the unbounded mirror of nimblecas.algpoly.
// @author Olumuyiwa Oluwasanmi
//
// Oracles are hand-verified over Q(i) = Q[t]/(t^2 + 1) (alpha^2 = -1) and Q(sqrt2) =
// Q[t]/(t^2 - 2) (alpha^2 = 2). The int64-mirroring identities cover Q(i): add, the
// factorization (x + i)(x - i) = x^2 + 1, the exact division and monic gcd it induces,
// derivative and Horner evaluation. The load-bearing point of this tier is that coefficient
// arithmetic which would OVERFLOW the int64 Rational succeeds exactly here: with B = 10^10,
// (x + B*sqrt2)(x - B*sqrt2) = x^2 - 2*B^2 = x^2 - 2e20, whose constant coefficient (-2e20)
// far exceeds int64's ~9.22e18 ceiling yet is reproduced (and divided back out) exactly. The
// honest-refusal cases cover divide-by-zero-poly, monic-of-zero-poly, and cross-field ops.

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.bigratpoly;
import nimblecas.bigalgnum;
import nimblecas.bigalgpoly;
import nimblecas.testing;

using nimblecas::BigAlgebraicNumber;
using nimblecas::BigAlgebraicPoly;
using nimblecas::BigNumberField;
using nimblecas::BigRational;
using nimblecas::BigRationalPoly;
using nimblecas::make_error;
using nimblecas::MathError;
using nimblecas::Result;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// BigRational from an int64.
auto q(std::int64_t v) -> BigRational { return BigRational::from_int(v); }

// BigRational from a decimal string (for magnitudes beyond int64). Falls back to canonical
// zero on a malformed literal, which makes any test relying on it fail visibly.
auto qs(std::string_view s) -> BigRational {
    auto r = BigRational::from_string(s);
    return r ? *r : BigRational{};
}

// BigRationalPoly from int64 coefficients in index order (constant first).
auto poly(std::vector<std::int64_t> cs) -> BigRationalPoly {
    std::vector<BigRational> v;
    v.reserve(cs.size());
    for (std::int64_t c : cs) {
        v.push_back(q(c));
    }
    return BigRationalPoly::from_coeffs(std::move(v));
}

// x^2 + 1 (Q(i)) and x^2 - 2 (Q(sqrt2)).
auto poly_x2_plus_1() -> BigRationalPoly { return poly({1, 0, 1}); }
auto poly_x2_minus_2() -> BigRationalPoly { return poly({-2, 0, 1}); }

// Build a BigAlgebraicPoly whose x^i coefficient is the field element (residue) `residues[i]`.
// Each residue is reduced mod m via from_poly, then the coefficient vector is trimmed by
// from_coeffs. Returns Result so callers can guard on failure.
auto apoly(const BigNumberField& f, std::vector<BigRationalPoly> residues)
    -> Result<BigAlgebraicPoly> {
    std::vector<BigAlgebraicNumber> cs;
    cs.reserve(residues.size());
    for (const BigRationalPoly& r : residues) {
        auto e = f.from_poly(r);
        if (!e) {
            return make_error<BigAlgebraicPoly>(e.error());
        }
        cs.push_back(std::move(*e));
    }
    return BigAlgebraicPoly::from_coeffs(f, std::move(cs));
}

// The decimal literal "10000000000" is 10^10; -2e20 = -2*(10^10)^2 overflows int64.
constexpr std::string_view kB = "10000000000";            // 10^10
constexpr std::string_view kNeg2B2 = "-200000000000000000000";  // -2*10^20

}  // namespace

auto main() -> int {
    auto qi_r = BigNumberField::create(poly_x2_plus_1());
    if (!qi_r) {
        std::println("FATAL: could not construct Q(i)");
        return 1;
    }
    const BigNumberField qi = *qi_r;

    auto q2_r = BigNumberField::create(poly_x2_minus_2());
    if (!q2_r) {
        std::println("FATAL: could not construct Q(sqrt2)");
        return 1;
    }
    const BigNumberField q2 = *q2_r;

    return TestSuite("nimblecas.bigalgpoly")
        .test("zero polynomial has empty coeffs and degree -1",
              [&](TestContext& t) {
                  const BigAlgebraicPoly z = BigAlgebraicPoly::zero(qi);
                  t.expect(z.is_zero(), "zero().is_zero()");
                  t.expect(z.degree() == -1, "degree of zero poly is -1");
                  t.expect(z.leading_coefficient().is_zero(), "leading coeff of zero is 0");
              })
        .test("from_coeffs trims trailing zeros over Q(i)",
              [&](TestContext& t) {
                  // (1 + i) + 0*x  -> degree 0 after trimming the trailing zero coefficient.
                  auto p = apoly(qi, {poly({1, 1}), poly({0})});
                  const bool ok = p.has_value();
                  t.expect(ok, "built (1+i) + 0*x");
                  if (!ok) return;
                  t.expect(p->degree() == 0, "trailing zero trimmed -> degree 0");
              })
        .test("add over Q(i)",
              [&](TestContext& t) {
                  // P = (1+i) + 2x, Q = i + (3+i)x, P+Q = (1+2i) + (5+i)x.
                  auto p = apoly(qi, {poly({1, 1}), poly({2})});
                  auto r = apoly(qi, {poly({0, 1}), poly({3, 1})});
                  auto expect = apoly(qi, {poly({1, 2}), poly({5, 1})});
                  const bool ok = p.has_value() && r.has_value() && expect.has_value();
                  t.expect(ok, "built operands and oracle");
                  if (!ok) return;
                  auto sum = p->add(*r);
                  const bool oks = sum.has_value();
                  t.expect(oks, "add succeeded");
                  if (!oks) return;
                  t.expect(sum->is_equal(*expect), "(1+i)+2x plus i+(3+i)x == (1+2i)+(5+i)x");
              })
        .test("multiply over Q(i): (x + i)(x - i) == x^2 + 1",
              [&](TestContext& t) {
                  auto p = apoly(qi, {poly({0, 1}), poly({1})});   // i + x
                  auto r = apoly(qi, {poly({0, -1}), poly({1})});  // -i + x
                  auto expect = apoly(qi, {poly({1}), poly({0}), poly({1})});  // 1 + x^2
                  const bool ok = p.has_value() && r.has_value() && expect.has_value();
                  t.expect(ok, "built operands and oracle");
                  if (!ok) return;
                  auto prod = p->multiply(*r);
                  const bool okp = prod.has_value();
                  t.expect(okp, "multiply succeeded");
                  if (!okp) return;
                  t.expect(prod->degree() == 2, "product degree 2");
                  t.expect(prod->is_equal(*expect), "(x+i)(x-i) == x^2 + 1");
              })
        .test("divide over Q(i): (x^2 + 1) / (x + i) == x - i, remainder 0",
              [&](TestContext& t) {
                  auto num = apoly(qi, {poly({1}), poly({0}), poly({1})});  // x^2 + 1
                  auto den = apoly(qi, {poly({0, 1}), poly({1})});          // i + x
                  auto quo = apoly(qi, {poly({0, -1}), poly({1})});         // -i + x
                  const bool ok = num.has_value() && den.has_value() && quo.has_value();
                  t.expect(ok, "built operands and oracle");
                  if (!ok) return;
                  auto dm = num->divide(*den);
                  const bool okd = dm.has_value();
                  t.expect(okd, "divide succeeded");
                  if (!okd) return;
                  t.expect(dm->remainder.is_zero(), "remainder is zero");
                  t.expect(dm->quotient.is_equal(*quo), "quotient == x - i");
              })
        .test("gcd over Q(i): gcd(x^2 + 1, x + i) == x + i (monic)",
              [&](TestContext& t) {
                  auto a = apoly(qi, {poly({1}), poly({0}), poly({1})});  // x^2 + 1
                  auto b = apoly(qi, {poly({0, 1}), poly({1})});          // i + x (monic)
                  const bool ok = a.has_value() && b.has_value();
                  t.expect(ok, "built operands");
                  if (!ok) return;
                  auto g = a->gcd(*b);
                  const bool okg = g.has_value();
                  t.expect(okg, "gcd succeeded");
                  if (!okg) return;
                  t.expect(g->is_equal(*b), "gcd == x + i");
                  t.expect(g->leading_coefficient().is_one(), "gcd is monic");
              })
        .test("derivative over Q(i): d/dx[(1+i) + 3x + x^2] == 3 + 2x",
              [&](TestContext& t) {
                  auto p = apoly(qi, {poly({1, 1}), poly({3}), poly({1})});
                  auto expect = apoly(qi, {poly({3}), poly({2})});
                  const bool ok = p.has_value() && expect.has_value();
                  t.expect(ok, "built poly and oracle");
                  if (!ok) return;
                  auto d = p->derivative();
                  const bool okd = d.has_value();
                  t.expect(okd, "derivative succeeded");
                  if (!okd) return;
                  t.expect(d->is_equal(*expect), "derivative == 3 + 2x");
              })
        .test("evaluate over Q(i): (x^2 + 1) at x = i is 0",
              [&](TestContext& t) {
                  auto p = apoly(qi, {poly({1}), poly({0}), poly({1})});  // x^2 + 1
                  auto x = qi.generator();  // i
                  const bool ok = p.has_value() && x.has_value();
                  t.expect(ok, "built poly and point");
                  if (!ok) return;
                  auto val = p->evaluate(*x);
                  const bool okv = val.has_value();
                  t.expect(okv, "evaluate succeeded");
                  if (!okv) return;
                  t.expect(val->is_zero(), "i^2 + 1 == 0");
              })
        // --- BIGNUM ADVANTAGE: coefficients that overflow int64 ---------------------
        .test("multiply over Q(sqrt2): (x + B*sqrt2)(x - B*sqrt2) == x^2 - 2e20",
              [&](TestContext& t) {
                  // B = 10^10; (B sqrt2)^2 = 2*B^2 = 2e20, so the product's constant term is
                  // -2e20, far beyond int64's ~9.22e18 ceiling — exact only in the bignum tier.
                  auto p = apoly(q2, {BigRationalPoly::monomial(qs(kB), 1),
                                      BigRationalPoly::constant(q(1))});  // B*sqrt2 + x
                  auto r = apoly(q2, {BigRationalPoly::monomial(qs(kB).negate(), 1),
                                      BigRationalPoly::constant(q(1))});  // -B*sqrt2 + x
                  auto expect = apoly(q2, {BigRationalPoly::constant(qs(kNeg2B2)),
                                           BigRationalPoly::constant(q(0)),
                                           BigRationalPoly::constant(q(1))});  // -2e20 + x^2
                  const bool ok = p.has_value() && r.has_value() && expect.has_value();
                  t.expect(ok, "built operands and oracle");
                  if (!ok) return;
                  auto prod = p->multiply(*r);
                  const bool okp = prod.has_value();
                  t.expect(okp, "multiply without overflow");
                  if (!okp) return;
                  t.expect(prod->is_equal(*expect), "product == x^2 - 2e20 exactly");
                  t.expect(prod->coefficient(0).is_equal(q2.from_bigrational(qs(kNeg2B2))),
                           "constant coefficient == -2e20 (overflows int64)");
              })
        .test("divide over Q(sqrt2): (x^2 - 2e20) / (x + B*sqrt2) == x - B*sqrt2",
              [&](TestContext& t) {
                  // The bignum-scale factorization divides back out exactly, remainder 0.
                  auto num = apoly(q2, {BigRationalPoly::constant(qs(kNeg2B2)),
                                        BigRationalPoly::constant(q(0)),
                                        BigRationalPoly::constant(q(1))});  // x^2 - 2e20
                  auto den = apoly(q2, {BigRationalPoly::monomial(qs(kB), 1),
                                        BigRationalPoly::constant(q(1))});  // B*sqrt2 + x
                  auto quo = apoly(q2, {BigRationalPoly::monomial(qs(kB).negate(), 1),
                                        BigRationalPoly::constant(q(1))});  // -B*sqrt2 + x
                  const bool ok = num.has_value() && den.has_value() && quo.has_value();
                  t.expect(ok, "built operands and oracle");
                  if (!ok) return;
                  auto dm = num->divide(*den);
                  const bool okd = dm.has_value();
                  t.expect(okd, "divide without overflow");
                  if (!okd) return;
                  t.expect(dm->remainder.is_zero(), "remainder is zero");
                  t.expect(dm->quotient.is_equal(*quo), "quotient == x - B*sqrt2");
              })
        // --- honest refusals --------------------------------------------------------
        .test("divide by the zero polynomial is division_by_zero",
              [&](TestContext& t) {
                  auto p = apoly(qi, {poly({1}), poly({1})});  // 1 + x
                  const bool ok = p.has_value();
                  t.expect(ok, "built dividend");
                  if (!ok) return;
                  auto dm = p->divide(BigAlgebraicPoly::zero(qi));
                  t.expect(!dm.has_value() && dm.error() == MathError::division_by_zero,
                           "divide by zero poly -> division_by_zero");
              })
        .test("monic of the zero polynomial is division_by_zero",
              [&](TestContext& t) {
                  auto m = BigAlgebraicPoly::zero(qi).monic();
                  t.expect(!m.has_value() && m.error() == MathError::division_by_zero,
                           "monic of zero poly -> division_by_zero");
              })
        .test("cross-field operation is domain_error",
              [&](TestContext& t) {
                  auto p = apoly(qi, {poly({0, 1}), poly({1})});  // over Q(i)
                  auto r = apoly(q2, {poly({0, 1}), poly({1})});  // over Q(sqrt2)
                  const bool ok = p.has_value() && r.has_value();
                  t.expect(ok, "built cross-field operands");
                  if (!ok) return;
                  auto bad = p->add(*r);
                  t.expect(!bad.has_value() && bad.error() == MathError::domain_error,
                           "add across fields -> domain_error");
              })
        .run();
}
