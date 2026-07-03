// Feature/integration tests: exact algebra core (Rational, Q[x], Z[x], Gaussian rationals).
// @author Olumuyiwa Oluwasanmi
//
// These are cross-module workflow and algebraic-law tests, not per-module unit tests:
// field/ring axioms on concrete values, the evaluation homomorphism, Euclidean structure
// of Q[x], the Z[x] <-> Q[x] bridge, and the Gaussian-rational conjugate/norm identities.
// Every assertion pins a concrete exact value or an exact identity; overflow and
// division-by-zero must surface on the railway, never as undefined behaviour.

import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.ratpoly;
import nimblecas.complex;
import nimblecas.testing;

using nimblecas::Complex;
using nimblecas::MathError;
using nimblecas::Polynomial;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// --- concise builders -------------------------------------------------------

[[nodiscard]] auto ri(std::int64_t v) -> Rational {
    return Rational::from_int(v);
}

// A reduced fraction n/d; used only where n/d is known valid (non-zero, non-INT64_MIN).
[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

// A Q[x] element from integer coefficients (coeffs[i] is the coefficient of x^i).
[[nodiscard]] auto rpi(std::vector<std::int64_t> cs) -> RationalPoly {
    std::vector<Rational> r;
    r.reserve(cs.size());
    for (const std::int64_t v : cs) {
        r.push_back(Rational::from_int(v));
    }
    return RationalPoly::from_coeffs(std::move(r));
}

// A Q[x] element from explicit Rational coefficients.
[[nodiscard]] auto rpq(std::vector<Rational> cs) -> RationalPoly {
    return RationalPoly::from_coeffs(std::move(cs));
}

// A Z[x] element from integer coefficients.
[[nodiscard]] auto pz(std::vector<std::int64_t> cs) -> Polynomial {
    return Polynomial{std::move(cs)};
}

// A Gaussian rational (a + b i) with integer parts.
[[nodiscard]] auto cxi(std::int64_t a, std::int64_t b) -> Complex {
    return Complex::make(Rational::from_int(a), Rational::from_int(b));
}

}  // namespace

auto main() -> int {
    return TestSuite("feature.algebra")
        // ----------------------------------------------------------------- Q field
        .test("rational_add_assoc_commute",
              [](TestContext& t) {
                  const Rational a = rat(1, 2);
                  const Rational b = rat(1, 3);
                  const Rational c = rat(1, 4);
                  // (a+b)+c == a+(b+c), and both equal the hand value 13/12.
                  const Rational ab = a.add(b).value();
                  const Rational bc = b.add(c).value();
                  const Rational left = ab.add(c).value();
                  const Rational right = a.add(bc).value();
                  t.expect(left == right, "addition is associative");
                  t.expect(left == rat(13, 12), "(1/2+1/3)+1/4 == 13/12");
                  // a+b == b+a, a*b == b*a.
                  t.expect(a.add(b).value() == b.add(a).value(), "addition commutes");
                  const Rational prod = a.multiply(b).value();
                  t.expect(prod == b.multiply(a).value(), "multiplication commutes");
                  t.expect(prod == rat(1, 6), "1/2 * 1/3 == 1/6");
              })
        .test("rational_distributive_both_sides",
              [](TestContext& t) {
                  const Rational a = rat(2, 3);
                  const Rational b = rat(1, 5);
                  const Rational c = rat(3, 7);
                  const Rational bc = b.add(c).value();
                  // Left distributivity a*(b+c) == a*b + a*c.
                  const Rational lhs = a.multiply(bc).value();
                  const Rational rhs = a.multiply(b).value().add(a.multiply(c).value()).value();
                  t.expect(lhs == rhs, "a*(b+c) == a*b + a*c");
                  // Right distributivity (b+c)*a == b*a + c*a (exercises the other operand order).
                  const Rational lhs2 = bc.multiply(a).value();
                  const Rational rhs2 = b.multiply(a).value().add(c.multiply(a).value()).value();
                  t.expect(lhs2 == rhs2, "(b+c)*a == b*a + c*a");
                  // Concrete value: (1/5 + 3/7) = 22/35, times 2/3 = 44/105.
                  t.expect(lhs == rat(44, 105), "2/3 * (1/5 + 3/7) == 44/105");
              })
        .test("rational_additive_and_multiplicative_inverse",
              [](TestContext& t) {
                  const Rational a = rat(3, 7);
                  // a + (-a) == 0/1.
                  t.expect(a.add(a.negate().value()).value() == ri(0),
                           "a + (-a) == 0");
                  // a * (1/a) == 1 for a != 0, via reciprocal and via self-division.
                  const Rational inv = ri(1).divide(a).value();
                  t.expect(inv == rat(7, 3), "1 / (3/7) == 7/3");
                  t.expect(a.multiply(inv).value() == ri(1), "a * (1/a) == 1");
                  t.expect(a.divide(a).value() == ri(1), "a / a == 1");
              })
        .test("rational_canonical_form",
              [](TestContext& t) {
                  // Sign is carried on the numerator; the fraction is reduced.
                  const Rational neg = rat(2, -4);
                  t.expect(neg == rat(-1, 2), "make(2,-4) == make(-1,2)");
                  t.expect(neg.numerator() == -1 && neg.denominator() == 2,
                           "canonical -1/2: den kept positive, reduced");
                  t.expect(rat(6, 4) == rat(3, 2), "6/4 reduces to 3/2");
                  t.expect(rat(-3, -6) == rat(1, 2), "-3/-6 reduces to 1/2");
                  // Zero canonicalises to 0/1 regardless of the given denominator.
                  const Rational z = rat(0, 5);
                  t.expect(z.numerator() == 0 && z.denominator() == 1, "0/5 == 0/1");
                  t.expect(z == ri(0), "0/5 == 0");
              })
        .test("rational_division_by_zero_is_railway",
              [](TestContext& t) {
                  // Dividing by the zero rational returns the error, never UB.
                  const auto q = rat(3, 4).divide(ri(0));
                  t.expect(!q.has_value(), "x / 0 fails on the railway");
                  t.expect(q.error() == MathError::division_by_zero, "x / 0 => division_by_zero");
                  // Constructing n/0 is likewise a railway error.
                  const auto bad = Rational::make(1, 0);
                  t.expect(!bad.has_value() && bad.error() == MathError::division_by_zero,
                           "make(1,0) => division_by_zero");
              })
        .test("rational_overflow_reported_not_UB",
              [](TestContext& t) {
                  // sqrt(INT64_MAX) ~= 3.037e9; the square of 3037000500 exceeds INT64_MAX.
                  const Rational big = ri(3037000500);
                  const auto prod = big.multiply(big);
                  t.expect(!prod.has_value(), "near-max numerator product overflows");
                  t.expect(prod.error() == MathError::overflow, "multiply overflow => overflow");
                  // Addition overflows when the denominator product wraps int64.
                  const auto sum = rat(1, 3037000500).add(rat(1, 3037000501));
                  t.expect(!sum.has_value(), "near-max denominator sum overflows");
                  t.expect(sum.error() == MathError::overflow, "add overflow => overflow");
              })
        // ------------------------------------------------------------ Z[x] ring laws
        .test("polynomial_eval_homomorphism_add",
              [](TestContext& t) {
                  // (p+q)(a) == p(a) + q(a) at several integer points.
                  const Polynomial p = pz({1, 3, 2});      // 2x^2 + 3x + 1
                  const Polynomial q = pz({5, -1, 0, 1});  // x^3 - x + 5
                  const Polynomial s = p.add(q).value();
                  t.expect(s.is_equal(pz({6, 2, 2, 1})), "p+q == x^3+2x^2+2x+6");
                  bool all_ok = true;
                  for (const std::int64_t a : {-1, 0, 1, 2}) {
                      const std::int64_t lhs = s.evaluate(a).value();
                      const std::int64_t rhs = p.evaluate(a).value() + q.evaluate(a).value();
                      all_ok = all_ok && (lhs == rhs);
                  }
                  t.expect(all_ok, "(p+q)(a) == p(a)+q(a) at a in {-1,0,1,2}");
                  t.expect(s.evaluate(2).value() == 26, "(p+q)(2) == 26");
              })
        .test("polynomial_eval_homomorphism_mul_and_degree",
              [](TestContext& t) {
                  const Polynomial p = pz({1, 2});   // 2x + 1
                  const Polynomial q = pz({-4, 3});  // 3x - 4
                  const Polynomial pq = p.multiply(q).value();
                  t.expect(pq.is_equal(pz({-4, -5, 6})), "product == 6x^2 - 5x - 4");
                  // Degree of a product is the sum of degrees.
                  t.expect(pq.degree() == p.degree() + q.degree(), "deg(p*q) == deg p + deg q");
                  t.expect(pq.degree() == 2, "product degree is 2");
                  bool all_ok = true;
                  for (const std::int64_t a : {-2, 0, 3, 5}) {
                      const std::int64_t lhs = pq.evaluate(a).value();
                      const std::int64_t rhs = p.evaluate(a).value() * q.evaluate(a).value();
                      all_ok = all_ok && (lhs == rhs);
                  }
                  t.expect(all_ok, "(p*q)(a) == p(a)*q(a) at a in {-2,0,3,5}");
              })
        .test("polynomial_horner_hand_value",
              [](TestContext& t) {
                  const Polynomial p = pz({5, 0, -2, 3});  // 3x^3 - 2x^2 + 5
                  t.expect(p.evaluate(2).value() == 21, "p(2) == 24 - 8 + 5 == 21");
                  t.expect(p.evaluate(0).value() == 5, "p(0) == constant term 5");
                  t.expect(p.evaluate(-1).value() == 0, "p(-1) == -3 - 2 + 5 == 0 (root)");
              })
        .test("polynomial_product_rule_derivative",
              [](TestContext& t) {
                  const Polynomial p = pz({1, 0, 1});  // x^2 + 1
                  const Polynomial q = pz({-3, 2});    // 2x - 3
                  const Polynomial lhs = p.multiply(q).value().derivative().value();
                  // p'*q + p*q'
                  const Polynomial pd_q = p.derivative().value().multiply(q).value();
                  const Polynomial p_qd = p.multiply(q.derivative().value()).value();
                  const Polynomial rhs = pd_q.add(p_qd).value();
                  t.expect(lhs.is_equal(rhs), "(p*q)' == p'*q + p*q'");
                  t.expect(lhs.is_equal(pz({2, -6, 6})), "(p*q)' == 6x^2 - 6x + 2");
              })
        // --------------------------------------------------- Q[x] Euclidean structure
        .test("ratpoly_euclidean_division_identity",
              [](TestContext& t) {
                  // x^2 / (2x + 1): the quotient/remainder must land in Q, not Z.
                  const RationalPoly p = rpi({0, 0, 1});  // x^2
                  const RationalPoly d = rpi({1, 2});     // 2x + 1
                  const auto dm = p.divide(d);
                  t.expect(dm.has_value(), "division over Q succeeds");
                  const RationalPoly& q = dm->quotient;
                  const RationalPoly& r = dm->remainder;
                  // Exact rational quotient (1/2)x - 1/4 and remainder 1/4.
                  t.expect(q.is_equal(rpq({rat(-1, 4), rat(1, 2)})), "quotient == (1/2)x - 1/4");
                  t.expect(r.is_equal(RationalPoly::constant(rat(1, 4))), "remainder == 1/4");
                  // p == q*d + r exactly, and deg(r) < deg(d).
                  const RationalPoly recon = q.multiply(d).value().add(r).value();
                  t.expect(recon.is_equal(p), "q*d + r == p");
                  t.expect(r.degree() < d.degree(), "deg(remainder) < deg(divisor)");
              })
        .test("ratpoly_gcd_divides_both",
              [](TestContext& t) {
                  const RationalPoly p = rpi({2, -3, 1});   // (x-1)(x-2) = x^2 - 3x + 2
                  const RationalPoly q = rpi({-3, 2, 1});   // (x-1)(x+3) = x^2 + 2x - 3
                  const auto g = p.gcd(q);
                  t.expect(g.has_value(), "gcd computes");
                  // Monic common factor x - 1.
                  t.expect(g->is_equal(rpi({-1, 1})), "gcd((x-1)(x-2),(x-1)(x+3)) == x - 1");
                  // gcd divides both with zero remainder.
                  t.expect(p.divide(*g).value().remainder.is_zero(), "gcd | p exactly");
                  t.expect(q.divide(*g).value().remainder.is_zero(), "gcd | q exactly");
              })
        .test("ratpoly_gcd_scaling_up_to_unit",
              [](TestContext& t) {
                  const RationalPoly p = rpi({-1, 1});  // x - 1
                  const RationalPoly q = rpi({-2, 1});  // x - 2  (coprime to p)
                  const RationalPoly g = rpi({5, 1});   // x + 5
                  const RationalPoly pg = p.multiply(g).value();
                  const RationalPoly qg = q.multiply(g).value();
                  // gcd(p,q) is a unit -> monic 1; gcd(p*g, q*g) is the monic form of g.
                  t.expect(p.gcd(q).value().is_equal(rpi({1})), "gcd(x-1, x-2) == 1");
                  t.expect(pg.gcd(qg).value().is_equal(rpi({5, 1})),
                           "gcd(p*g, q*g) == g*gcd(p,q) up to a unit == x + 5");
              })
        .test("ratpoly_monic_normalization",
              [](TestContext& t) {
                  const RationalPoly p = rpi({6, 4, 2});  // 2x^2 + 4x + 6
                  const auto m = p.monic();
                  t.expect(m.has_value(), "monic computes");
                  t.expect(m->is_equal(rpi({3, 2, 1})), "monic(2x^2+4x+6) == x^2 + 2x + 3");
                  t.expect(m->leading_coefficient() == ri(1), "leading coefficient is 1");
                  // Already-monic input is unchanged.
                  const RationalPoly already = rpi({3, 2, 1});
                  t.expect(already.monic().value().is_equal(already), "monic of a monic is itself");
              })
        // ------------------------------------------------------- Z[x] <-> Q[x] bridge
        .test("bridge_zx_qx_roundtrip",
              [](TestContext& t) {
                  const Polynomial zp = pz({6, -4, 2});  // 2x^2 - 4x + 6
                  const RationalPoly qp = RationalPoly::from_polynomial(zp);
                  t.expect(qp.coefficient(0) == ri(6) && qp.leading_coefficient() == ri(2),
                           "lift preserves integer coefficients");
                  // Round-trip Z[x] -> Q[x] -> Z[x] is the identity.
                  const auto back = qp.to_polynomial();
                  t.expect(back.has_value(), "integral Q[x] returns to Z[x]");
                  t.expect(back->is_equal(zp), "to_polynomial(from_polynomial(zp)) == zp");
                  // Content/primitive-part factorisation in Z[x]: zp == content * primitive.
                  t.expect(zp.content().value() == 2, "content(2x^2-4x+6) == 2");
                  t.expect(zp.primitive_part().value().is_equal(pz({3, -2, 1})),
                           "primitive part == x^2 - 2x + 3");
                  // A genuinely fractional Q[x] element cannot cross back into Z[x].
                  const auto frac = rpq({rat(1, 2)}).to_polynomial();
                  t.expect(!frac.has_value() && frac.error() == MathError::domain_error,
                           "non-integral coefficient => domain_error");
              })
        // --------------------------------------------------- Gaussian rationals Q + Qi
        .test("complex_multiplication_formula",
              [](TestContext& t) {
                  const Complex z = cxi(2, 3);   // 2 + 3i
                  const Complex w = cxi(1, -4);  // 1 - 4i
                  // (ac - bd) + (ad + bc)i = (2 + 12) + (-8 + 3)i = 14 - 5i.
                  const Complex zw = z.multiply(w).value();
                  t.expect(zw == cxi(14, -5), "(2+3i)(1-4i) == 14 - 5i");
                  t.expect(zw.real() == ri(14) && zw.imag() == ri(-5), "parts are 14 and -5");
              })
        .test("complex_conjugate_involution_and_distributes",
              [](TestContext& t) {
                  const Complex z = cxi(2, 3);
                  const Complex w = cxi(1, -4);
                  // conj(conj z) == z.
                  t.expect(z.conjugate().value().conjugate().value() == z,
                           "conjugation is an involution");
                  // conj(z*w) == conj(z) * conj(w) == 14 + 5i.
                  const Complex lhs = z.multiply(w).value().conjugate().value();
                  const Complex rhs = z.conjugate().value().multiply(w.conjugate().value()).value();
                  t.expect(lhs == rhs, "conj(z*w) == conj(z)*conj(w)");
                  t.expect(lhs == cxi(14, 5), "conj((2+3i)(1-4i)) == 14 + 5i");
              })
        .test("complex_norm_is_real_and_multiplicative",
              [](TestContext& t) {
                  const Complex z = cxi(2, 3);   // N = 13
                  const Complex w = cxi(1, -4);  // N = 17
                  // z * conj(z) is real and equals norm_squared(z).
                  const Complex zz = z.multiply(z.conjugate().value()).value();
                  t.expect(zz.is_real(), "z*conj(z) is real");
                  t.expect(zz.real() == z.norm_squared().value() && zz.real() == ri(13),
                           "z*conj(z) == N(z) == 13");
                  // N(z*w) == N(z) * N(w).
                  const Rational nz = z.norm_squared().value();
                  const Rational nw = w.norm_squared().value();
                  const Rational nzw = z.multiply(w).value().norm_squared().value();
                  t.expect(nzw == nz.multiply(nw).value(), "N(z*w) == N(z)*N(w)");
                  t.expect(nzw == ri(221), "N((2+3i)(1-4i)) == 13*17 == 221");
              })
        .test("complex_real_embedding_and_division",
              [](TestContext& t) {
                  // Embedding a real Rational has zero imaginary part and is self-conjugate.
                  const Complex r = Complex::from_real(rat(3, 5));
                  t.expect(r.is_real() && r.imag() == ri(0), "from_real has zero imaginary part");
                  t.expect(r.conjugate().value() == r, "a real number equals its own conjugate");
                  // Division inverts multiplication: (z/w)*w == z for w != 0.
                  const Complex z = cxi(2, 3);
                  const Complex w = cxi(1, -4);
                  const Complex zq = z.divide(w).value();
                  t.expect(zq.multiply(w).value() == z, "(z/w)*w == z");
                  // w * (1/w) == 1.
                  t.expect(w.multiply(w.reciprocal().value()).value() == Complex::from_int(1),
                           "w * (1/w) == 1");
                  // Dividing by zero stays on the railway.
                  const auto bad = z.divide(Complex{});
                  t.expect(!bad.has_value() && bad.error() == MathError::division_by_zero,
                           "z / 0 => division_by_zero");
              })
        .run();
}
