// Tests for nimblecas.bigalgnum: the algebraic number field Q(alpha) over BigRational.
// @author Olumuyiwa Oluwasanmi
//
// Oracles are hand-verified over Q(i) = Q[t]/(t^2 + 1) (alpha^2 = -1) and Q(sqrt2) =
// Q[t]/(t^2 - 2) (alpha^2 = 2). The load-bearing point of this tier is that residue
// arithmetic which would OVERFLOW the int64 Rational succeeds exactly here: with
// B = 10^10, (B*alpha)^2 = 2*B^2 = 2e20 and norm(B*alpha) = -2e20 both exceed the int64
// ceiling (~9.22e18), yet are reproduced exactly. The int64-mirroring identities cover
// Q(i): (a+bi)(c+di), i*i = -1, inverse of (1+i) = (1-i)/2, is_equal, norm/trace, and the
// honest inverse-of-zero error.

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.bigratpoly;
import nimblecas.bigalgnum;
import nimblecas.testing;

using nimblecas::BigAlgebraicNumber;
using nimblecas::BigNumberField;
using nimblecas::BigRational;
using nimblecas::BigRationalPoly;
using nimblecas::MathError;
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

// The decimal literal "10000000000" is 10^10; its square 10^20 and 2*10^20 overflow int64.
constexpr std::string_view kB = "10000000000";           // 10^10
constexpr std::string_view kB2 = "100000000000000000000";   // 10^20
constexpr std::string_view k2B2 = "200000000000000000000";  // 2*10^20
constexpr std::string_view kNeg2B2 = "-200000000000000000000";

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

    return TestSuite("nimblecas.bigalgnum")
        .test("field reports monic modulus and degree",
              [&](TestContext& t) {
                  t.expect(qi.degree() == 2, "deg Q(i) == 2");
                  t.expect(qi.modulus().is_equal(poly_x2_plus_1()), "modulus is x^2 + 1");
                  t.expect(qi.one().is_one() && qi.zero().is_zero(), "one()/zero() honest");
              })
        .test("create rejects zero and constant minimal polynomials",
              [&](TestContext& t) {
                  auto z = BigNumberField::create(BigRationalPoly::zero());
                  t.expect(!z.has_value() && z.error() == MathError::domain_error,
                           "zero modulus -> domain_error");
                  auto c = BigNumberField::create(BigRationalPoly::constant(q(5)));
                  t.expect(!c.has_value() && c.error() == MathError::domain_error,
                           "constant modulus -> domain_error");
              })
        .test("non-monic minimal is normalised to monic",
              [&](TestContext& t) {
                  // 3x^2 + 3 normalises to x^2 + 1, the same field as Q(i).
                  auto f = BigNumberField::create(poly({3, 0, 3}));
                  t.expect(f.has_value(), "create(3x^2+3)");
                  if (!f) return;
                  t.expect(f->modulus().is_equal(poly_x2_plus_1()), "monic modulus x^2 + 1");
                  t.expect(f->is_same(qi), "same field as Q(i)");
              })
        .test("i*i == -1 over Q(i)",
              [&](TestContext& t) {
                  auto alpha = qi.generator();
                  t.expect(alpha.has_value(), "generator");
                  if (!alpha) return;
                  auto sq = alpha->multiply(*alpha);
                  t.expect(sq.has_value(), "alpha*alpha");
                  if (!sq) return;
                  t.expect(sq->is_equal(qi.from_bigrational(q(-1))), "alpha^2 == -1");
              })
        .test("(1+2i)(3+4i) == -5+10i over Q(i)",
              [&](TestContext& t) {
                  auto a = qi.from_poly(poly({1, 2}));  // 1 + 2i
                  auto b = qi.from_poly(poly({3, 4}));  // 3 + 4i
                  t.expect(a.has_value() && b.has_value(), "built operands");
                  if (!a || !b) return;
                  auto prod = a->multiply(*b);
                  t.expect(prod.has_value(), "multiply");
                  if (!prod) return;
                  auto expect = qi.from_poly(poly({-5, 10}));  // -5 + 10i
                  t.expect(expect.has_value(), "built oracle");
                  if (!expect) return;
                  t.expect(prod->is_equal(*expect), "(1+2i)(3+4i) == -5+10i");
              })
        .test("inverse of (1+i) is (1-i)/2 over Q(i)",
              [&](TestContext& t) {
                  auto a = qi.from_poly(poly({1, 1}));  // 1 + i
                  t.expect(a.has_value(), "built 1+i");
                  if (!a) return;
                  auto inv = a->inverse();
                  t.expect(inv.has_value(), "inverse succeeded");
                  if (!inv) return;
                  // (1 - i)/2 has residue 1/2 - (1/2) i.
                  std::vector<BigRational> half{*q(1).divide(q(2)),
                                                *q(-1).divide(q(2))};
                  auto expect = qi.from_poly(BigRationalPoly::from_coeffs(std::move(half)));
                  t.expect(expect.has_value(), "built oracle (1-i)/2");
                  if (!expect) return;
                  t.expect(inv->is_equal(*expect), "inverse == (1-i)/2");
                  auto prod = a->multiply(*inv);
                  t.expect(prod.has_value(), "(1+i)*inverse");
                  if (!prod) return;
                  t.expect(prod->is_one(), "(1+i)*(1+i)^-1 == 1");
              })
        .test("norm and trace of 1+2i over Q(i)",
              [&](TestContext& t) {
                  auto a = qi.from_poly(poly({1, 2}));  // 1 + 2i
                  t.expect(a.has_value(), "built 1+2i");
                  if (!a) return;
                  auto n = a->norm();
                  auto tr = a->trace();
                  t.expect(n.has_value() && tr.has_value(), "norm/trace computed");
                  if (!n || !tr) return;
                  t.expect(*n == q(5), "N(1+2i) == 1^2 + 2^2 == 5");
                  t.expect(*tr == q(2), "Tr(1+2i) == 2*1 == 2");
              })
        .test("is_equal and inverse-of-zero honesty",
              [&](TestContext& t) {
                  auto a = qi.from_poly(poly({1, 1}));
                  auto b = qi.from_poly(poly({1, 1}));
                  t.expect(a.has_value() && b.has_value(), "built duplicates");
                  if (!a || !b) return;
                  t.expect(a->is_equal(*b), "equal residues are equal");
                  t.expect(!a->is_equal(qi.one()), "1+i != 1");
                  auto dz = qi.zero().inverse();
                  t.expect(!dz.has_value() && dz.error() == MathError::division_by_zero,
                           "inverse(0) -> division_by_zero");
              })
        .test("field mismatch across Q(i) and Q(sqrt2) is domain_error",
              [&](TestContext& t) {
                  auto a = qi.generator();
                  auto b = q2.generator();
                  t.expect(a.has_value() && b.has_value(), "built generators");
                  if (!a || !b) return;
                  auto bad = a->add(*b);
                  t.expect(!bad.has_value() && bad.error() == MathError::domain_error,
                           "add across fields -> domain_error");
              })
        // --- BIGNUM ADVANTAGE: residue coefficients that overflow int64 -------------
        .test("(B*alpha)^2 == 2*B^2 == 2e20 in Q(sqrt2) (overflows int64)",
              [&](TestContext& t) {
                  // g = 10^10 * sqrt2. g^2 = (10^10)^2 * 2 = 2e20, a residue coefficient far
                  // beyond int64's ~9.22e18 ceiling — exact only in the bignum tier.
                  auto g = q2.from_poly(BigRationalPoly::monomial(qs(kB), 1));
                  t.expect(g.has_value(), "built 10^10 * sqrt2");
                  if (!g) return;
                  auto sq = g->multiply(*g);
                  t.expect(sq.has_value(), "squared without overflow");
                  if (!sq) return;
                  t.expect(sq->is_equal(q2.from_bigrational(qs(k2B2))),
                           "(B*sqrt2)^2 == 2e20 exactly");
              })
        .test("(B*alpha + B)(B*alpha - B) == B^2 == 1e20 in Q(sqrt2)",
              [&](TestContext& t) {
                  // (B sqrt2 + B)(B sqrt2 - B) = (B sqrt2)^2 - B^2 = 2B^2 - B^2 = B^2 = 1e20.
                  // The intermediate B^2*x^2 term (1e20) overflows int64 before reduction.
                  auto u = q2.from_poly(BigRationalPoly::from_coeffs({qs(kB), qs(kB)}));
                  std::vector<BigRational> vc{qs(kB).negate(), qs(kB)};
                  auto v = q2.from_poly(BigRationalPoly::from_coeffs(std::move(vc)));
                  t.expect(u.has_value() && v.has_value(), "built operands");
                  if (!u || !v) return;
                  auto prod = u->multiply(*v);
                  t.expect(prod.has_value(), "product without overflow");
                  if (!prod) return;
                  t.expect(prod->is_equal(q2.from_bigrational(qs(kB2))),
                           "product == 1e20 exactly");
              })
        .test("inverse of B*alpha in Q(sqrt2) round-trips to one",
              [&](TestContext& t) {
                  auto g = q2.from_poly(BigRationalPoly::monomial(qs(kB), 1));  // 10^10*sqrt2
                  t.expect(g.has_value(), "built B*sqrt2");
                  if (!g) return;
                  auto inv = g->inverse();
                  t.expect(inv.has_value(), "inverse succeeded");
                  if (!inv) return;
                  auto prod = g->multiply(*inv);
                  t.expect(prod.has_value(), "g * g^-1");
                  if (!prod) return;
                  t.expect(prod->is_one(), "(B*sqrt2)*(B*sqrt2)^-1 == 1");
              })
        .test("norm(B*alpha) == -2*B^2 == -2e20 in Q(sqrt2) (overflows int64)",
              [&](TestContext& t) {
                  // N(c*sqrt2) = (c sqrt2)(-c sqrt2) = -2c^2; for c=10^10 that is -2e20, and
                  // the determinant is computed over BigRational without overflow.
                  auto g = q2.from_poly(BigRationalPoly::monomial(qs(kB), 1));
                  t.expect(g.has_value(), "built B*sqrt2");
                  if (!g) return;
                  auto n = g->norm();
                  t.expect(n.has_value(), "norm computed");
                  if (!n) return;
                  t.expect(*n == qs(kNeg2B2), "norm == -2e20 exactly");
              })
        .test("reducible modulus: a zero-divisor residue reports division_by_zero",
              [&](TestContext& t) {
                  // x^2 - 1 = (x-1)(x+1) is reducible; create() accepts it (no factoring at
                  // this tier). The residue (x - 1) shares the factor (x-1) with the modulus,
                  // so gcd is non-unit and it has no inverse -> honest division_by_zero.
                  auto f = BigNumberField::create(poly({-1, 0, 1}));  // x^2 - 1
                  t.expect(f.has_value(), "create(x^2-1) accepted (unproven irreducibility)");
                  if (!f) return;
                  auto a = f->from_poly(poly({-1, 1}));  // x - 1
                  t.expect(a.has_value(), "built x - 1");
                  if (!a) return;
                  auto inv = a->inverse();
                  t.expect(!inv.has_value() && inv.error() == MathError::division_by_zero,
                           "zero divisor has no inverse -> division_by_zero");
              })
        .run();
}
