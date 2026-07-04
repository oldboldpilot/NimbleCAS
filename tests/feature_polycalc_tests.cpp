// Feature/integration tests: polynomial algebra & calculus over Q[x] (resultant, pfd, integrate, roots, recurrence, orthopoly, combinatorics).
// @author Olumuyiwa Oluwasanmi
//
// These are cross-module workflows and exact identities over Q, not a rehash of the
// per-module unit tests: partial-fraction round-trips recombined over a common
// denominator, antiderivatives differentiated back through the rational/logarithmic
// split, resultant/discriminant identities, root reconstruction, characteristic-root
// handling, orthogonal-polynomial three-term recurrences, and combinatorial identities.

import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.ratpoly;
import nimblecas.resultant;
import nimblecas.pfd;
import nimblecas.ratint;
import nimblecas.rothstein;
import nimblecas.integrate;
import nimblecas.roots;
import nimblecas.recurrence;
import nimblecas.orthopoly;
import nimblecas.combinatorics;
import nimblecas.testing;

using namespace nimblecas;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// --- construction shorthands ------------------------------------------------

[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}
[[nodiscard]] auto ri(std::int64_t v) -> Rational { return Rational::from_int(v); }

// RationalPoly from integer coefficients, low degree first.
[[nodiscard]] auto ipoly(std::vector<std::int64_t> c) -> RationalPoly {
    return RationalPoly::from_polynomial(Polynomial{std::move(c)});
}
// RationalPoly from rational coefficients, low degree first.
[[nodiscard]] auto rpoly(std::vector<Rational> c) -> RationalPoly {
    return RationalPoly::from_coeffs(std::move(c));
}
// Constant polynomial n/d.
[[nodiscard]] auto cpoly(std::int64_t n, std::int64_t d) -> RationalPoly {
    return RationalPoly::constant(rat(n, d));
}
[[nodiscard]] auto one_rp() -> RationalPoly { return RationalPoly::constant(ri(1)); }
[[nodiscard]] auto x_rp() -> RationalPoly { return RationalPoly::monomial(ri(1), 1); }

// Test-side b^n by repeated multiplication.
[[nodiscard]] auto ipow(const RationalPoly& b, std::int64_t n) -> RationalPoly {
    RationalPoly acc = one_rp();
    for (std::int64_t k = 0; k < n; ++k) {
        acc = acc.multiply(b).value();
    }
    return acc;
}

// --- rational-function fraction algebra (test side) -------------------------

struct Frac {
    RationalPoly num;
    RationalPoly den;  // never zero
};

// x + y = (x.num*y.den + y.num*x.den) / (x.den*y.den) — no reduction needed for
// the exact cross-multiplication identities below.
[[nodiscard]] auto frac_add(const Frac& x, const Frac& y) -> Frac {
    auto ad = x.num.multiply(y.den).value();
    auto cb = y.num.multiply(x.den).value();
    auto num = ad.add(cb).value();
    auto den = x.den.multiply(y.den).value();
    return Frac{.num = std::move(num), .den = std::move(den)};
}

// Recombine a partial-fraction decomposition P + sum C_i/factor_i^power_i into a single
// fraction over the common denominator.
[[nodiscard]] auto recombine(const PartialFraction& pf) -> Frac {
    Frac acc{.num = pf.polynomial_part, .den = one_rp()};
    for (const auto& term : pf.terms) {
        const RationalPoly fp = ipow(term.factor, term.power);
        acc = frac_add(acc, Frac{.num = term.numerator, .den = fp});
    }
    return acc;
}

// Exact rational-function equality f == a/b via cross-multiplication f.num*b == a*f.den.
[[nodiscard]] auto frac_equals(const Frac& f, const RationalPoly& a, const RationalPoly& b)
    -> bool {
    return f.num.multiply(b).value().is_equal(a.multiply(f.den).value());
}

// d/dx(rn/rd) == A/B, verified exactly: (rn'*rd - rn*rd')*B == A*(rd*rd).
[[nodiscard]] auto derivative_matches(const RationalPoly& rn, const RationalPoly& rd,
                                      const RationalPoly& a, const RationalPoly& b) -> bool {
    const auto rnp = rn.derivative().value();
    const auto rdp = rd.derivative().value();
    const auto num =
        rnp.multiply(rd).value().subtract(rn.multiply(rdp).value()).value();
    const auto den = rd.multiply(rd).value();
    return num.multiply(b).value().is_equal(a.multiply(den).value());
}

// --- lookup helpers ---------------------------------------------------------

[[nodiscard]] auto find_num(const PartialFraction& pf, const RationalPoly& factor,
                            std::int64_t power) -> std::optional<RationalPoly> {
    for (const auto& term : pf.terms) {
        if (term.power == power && term.factor.is_equal(factor)) {
            return term.numerator;
        }
    }
    return std::nullopt;
}

[[nodiscard]] auto find_root(const std::vector<std::pair<Rational, std::int64_t>>& roots,
                             const Rational& r) -> std::optional<std::int64_t> {
    for (const auto& [root, mult] : roots) {
        if (root == r) {
            return mult;
        }
    }
    return std::nullopt;
}

[[nodiscard]] auto log_coeff(std::span<const LogTerm> terms, const RationalPoly& argument)
    -> std::optional<Rational> {
    for (const auto& term : terms) {
        if (term.argument.is_equal(argument)) {
            return term.coefficient;
        }
    }
    return std::nullopt;
}

// Rebuild lc * prod (x - r_i)^{m_i} from a root/multiplicity list.
[[nodiscard]] auto rebuild_from_roots(
    const std::vector<std::pair<Rational, std::int64_t>>& roots, const Rational& lc)
    -> RationalPoly {
    RationalPoly acc = RationalPoly::constant(lc);
    for (const auto& [r, mult] : roots) {
        const RationalPoly linear = rpoly({r.negate().value(), ri(1)});  // x - r
        for (std::int64_t i = 0; i < mult; ++i) {
            acc = acc.multiply(linear).value();
        }
    }
    return acc;
}

// p evaluated at x, via the exported roots-module Horner routine.
[[nodiscard]] auto eval_at(const RationalPoly& p, std::int64_t x) -> Rational {
    return evaluate(p, ri(x)).value();
}

}  // namespace

auto main() -> int {
    return TestSuite("feature.polycalc")
        // === PARTIAL FRACTIONS: common-denominator round-trip ===============
        .test("pfd_roundtrip_distinct_factors",
              [](TestContext& t) {
                  // 1 / ((x-1)(x+2)) with a square-free composite denominator: pfd keeps
                  // it as ONE square-free term, and recombining must return 1/(x^2+x-2).
                  const auto a = ipoly({1});
                  const auto b = ipoly({-2, 1, 1});  // (x-1)(x+2) = x^2 + x - 2
                  const auto pf = partial_fractions(a, b).value();
                  t.expect(pf.polynomial_part.is_zero(), "proper: no polynomial part");
                  t.expect(pf.terms.size() == 1, "square-free composite stays one term");
                  const auto whole = find_num(pf, ipoly({-2, 1, 1}), 1);
                  t.expect(whole && whole->is_equal(cpoly(1, 1)),
                           "numerator over x^2+x-2 is 1");
                  t.expect(frac_equals(recombine(pf), a, b),
                           "recombined terms equal 1/((x-1)(x+2)) exactly");
              })
        .test("pfd_roundtrip_repeated_root",
              [](TestContext& t) {
                  // x / (x-1)^2 = 1/(x-1) + 1/(x-1)^2: multiplicity 2 splits across powers.
                  const auto a = ipoly({0, 1});
                  const auto b = ipoly({1, -2, 1});  // (x-1)^2
                  const auto pf = partial_fractions(a, b).value();
                  t.expect(pf.polynomial_part.is_zero(), "proper: no polynomial part");
                  const auto c1 = find_num(pf, ipoly({-1, 1}), 1);
                  const auto c2 = find_num(pf, ipoly({-1, 1}), 2);
                  t.expect(c1 && c1->is_equal(cpoly(1, 1)), "1/(x-1) coefficient is 1");
                  t.expect(c2 && c2->is_equal(cpoly(1, 1)), "1/(x-1)^2 coefficient is 1");
                  t.expect(frac_equals(recombine(pf), a, b),
                           "recombined terms equal x/(x-1)^2 exactly");
              })
        // === RATIONAL INTEGRATION: rational part differentiates back ========
        .test("integrate_purely_rational_antiderivative",
              [](TestContext& t) {
                  // int 1/x^2 dx = -1/x: Hermite absorbs everything into the rational
                  // part, leaving no logarithmic remainder; d/dx(-1/x) == 1/x^2.
                  const auto a = ipoly({1});
                  const auto b = ipoly({0, 0, 1});  // x^2
                  const auto res = integrate_rational(a, b).value();
                  t.expect(res.log_terms.empty(), "no logarithmic part");
                  t.expect(res.rational_num.is_equal(cpoly(-1, 1)), "rational numerator is -1");
                  t.expect(res.rational_den.is_equal(x_rp()), "rational denominator is x");
                  t.expect(derivative_matches(res.rational_num, res.rational_den, a, b),
                           "d/dx(rational part) == 1/x^2");
              })
        .test("integrate_logarithmic_split",
              [](TestContext& t) {
                  // int 2x/(x^2+1) dx = log(x^2+1): denominator already square-free, so the
                  // rational part vanishes and the answer is a single log with residue 1.
                  const auto a = ipoly({0, 2});
                  const auto b = ipoly({1, 0, 1});  // x^2 + 1
                  const auto res = integrate_rational(a, b).value();
                  t.expect(res.rational_num.is_zero(), "no rational part");
                  t.expect(res.log_terms.size() == 1, "one logarithmic term");
                  const auto c = log_coeff(res.log_terms, ipoly({1, 0, 1}));
                  t.expect(c && *c == ri(1), "residue of log(x^2+1) is 1");
              })
        // === ROTHSTEIN-TRAGER: resultant residues on a concrete example =====
        .test("rothstein_log_part_residues",
              [](TestContext& t) {
                  // int 1/(x^2-1) dx = (1/2)log(x-1) - (1/2)log(x+1). The Rothstein-Trager
                  // resultant R(t) = res_x(D, A - t D') has rational roots +-1/2, the two
                  // residues, each attached to its linear log argument.
                  const auto lp = log_part(ipoly({1}), ipoly({-1, 0, 1})).value();
                  t.expect(lp.terms.size() == 2, "two logarithmic terms");
                  const auto plus = log_coeff(lp.terms, ipoly({-1, 1}));   // over (x-1)
                  const auto minus = log_coeff(lp.terms, ipoly({1, 1}));   // over (x+1)
                  t.expect(plus && *plus == rat(1, 2), "residue at x=1 is 1/2");
                  t.expect(minus && *minus == rat(-1, 2), "residue at x=-1 is -1/2");
              })
        // === RESULTANT / DISCRIMINANT ======================================
        .test("resultant_shared_root_vs_coprime",
              [](TestContext& t) {
                  const auto p = ipoly({2, -3, 1});  // (x-1)(x-2)
                  const auto shared = ipoly({6, -5, 1});    // (x-2)(x-3): common root x=2
                  const auto coprime = ipoly({12, -7, 1});  // (x-3)(x-4)
                  t.expect(resultant(p, shared).value() == ri(0),
                           "res == 0 when a root is shared");
                  t.expect(resultant(p, coprime).value() == ri(12),
                           "res((x-1)(x-2),(x-3)(x-4)) == q(1)q(2) == 12");
              })
        .test("resultant_is_multiplicative",
              [](TestContext& t) {
                  // res(p, q*r) == res(p, q) * res(p, r).
                  const auto p = ipoly({1, 0, 1});  // x^2 + 1
                  const auto q = ipoly({-1, 1});    // x - 1
                  const auto r = ipoly({-2, 1});    // x - 2
                  const auto qr = q.multiply(r).value();
                  const auto lhs = resultant(p, qr).value();
                  const auto rq = resultant(p, q).value();
                  const auto rr = resultant(p, r).value();
                  const auto rhs = rq.multiply(rr).value();
                  t.expect(lhs == rhs, "res(p,q*r) == res(p,q)*res(p,r)");
                  t.expect_ne(lhs, ri(0), "coprime factors give nonzero resultant");
              })
        .test("discriminant_equals_b2_minus_4ac",
              [](TestContext& t) {
                  // disc(a x^2 + b x + c) == b^2 - 4ac (coeffs low-order first: {c, b, a}).
                  struct Q {
                      std::int64_t a, b, c;
                  };
                  for (const Q q : {Q{2, 3, 1}, Q{1, 0, -4}, Q{1, -2, 1}, Q{3, 2, -1}}) {
                      const auto poly = ipoly({q.c, q.b, q.a});
                      const auto expected = rat(q.b * q.b - 4 * q.a * q.c, 1);
                      t.expect(discriminant(poly).value() == expected,
                               "discriminant matches b^2 - 4ac");
                  }
                  // A repeated root gives a vanishing discriminant.
                  t.expect(discriminant(ipoly({1, -2, 1})).value() == ri(0),
                           "disc((x-1)^2) == 0");
              })
        // === ROOTS ==========================================================
        .test("roots_factored_reconstruct",
              [](TestContext& t) {
                  // (x-1)(x-2)(2x-3) = 2x^3 - 9x^2 + 13x - 6: three simple rational roots.
                  const auto p = ipoly({-6, 13, -9, 2});
                  const auto roots = rational_roots(p).value();
                  t.expect(roots.size() == 3, "three distinct rational roots");
                  const auto m1 = find_root(roots, ri(1));
                  const auto m2 = find_root(roots, ri(2));
                  const auto m3 = find_root(roots, rat(3, 2));
                  t.expect(m1 && *m1 == 1, "root 1 has multiplicity 1");
                  t.expect(m2 && *m2 == 1, "root 2 has multiplicity 1");
                  t.expect(m3 && *m3 == 1, "root 3/2 has multiplicity 1");
                  // lc * prod (x - r_i) rebuilds the polynomial exactly (lc = 2).
                  t.expect(rebuild_from_roots(roots, ri(2)).is_equal(p),
                           "product of (x - r_i) reconstructs the polynomial");
              })
        .test("roots_with_multiplicity",
              [](TestContext& t) {
                  // (x-1)^2 (x-2) = x^3 - 4x^2 + 5x - 2: mixed multiplicities.
                  const auto p = ipoly({-2, 5, -4, 1});
                  const auto roots = rational_roots(p).value();
                  t.expect(roots.size() == 2, "two distinct roots");
                  const auto m1 = find_root(roots, ri(1));
                  const auto m2 = find_root(roots, ri(2));
                  t.expect(m1 && *m1 == 2, "root 1 has multiplicity 2");
                  t.expect(m2 && *m2 == 1, "root 2 has multiplicity 1");
                  t.expect(rebuild_from_roots(roots, ri(1)).is_equal(p),
                           "multiplicities reconstruct the polynomial");
              })
        .test("roots_none_rational",
              [](TestContext& t) {
                  // x^2 + 1 and x^2 - 2 have no rational roots (see module scope note).
                  t.expect(rational_roots(ipoly({1, 0, 1})).value().empty(),
                           "x^2 + 1 has no rational roots");
                  t.expect(rational_roots(ipoly({-2, 0, 1})).value().empty(),
                           "x^2 - 2 has no rational roots");
              })
        // === RECURRENCE =====================================================
        .test("recurrence_fibonacci_characteristic",
              [](TestContext& t) {
                  // a_n = a_{n-1} + a_{n-2}: characteristic x^2 - x - 1, irrational roots.
                  const std::vector<Rational> coeffs = {ri(1), ri(1)};
                  const auto poly = characteristic_polynomial(coeffs).value();
                  t.expect(poly.is_equal(ipoly({-1, -1, 1})),
                           "characteristic polynomial is x^2 - x - 1");
                  t.expect(all_roots_rational(coeffs).value() == false,
                           "golden-ratio roots are not rational");
                  t.expect(characteristic_roots(coeffs).value().empty(),
                           "no rational characteristic roots");
                  // Cross-check the sequence itself against hand-iterated a_0..a_10.
                  const std::array<std::int64_t, 11> expected = {0,  1,  1,  2,  3, 5,
                                                                 8, 13, 21, 34, 55};
                  for (std::int64_t n = 0; n <= 10; ++n) {
                      t.expect(fibonacci(n).value() == expected[static_cast<std::size_t>(n)],
                               "fibonacci(n) matches hand-iterated value");
                  }
                  for (std::int64_t n = 2; n <= 10; ++n) {
                      t.expect(fibonacci(n).value() ==
                                   fibonacci(n - 1).value() + fibonacci(n - 2).value(),
                               "F_n == F_{n-1} + F_{n-2}");
                  }
              })
        .test("recurrence_rational_characteristic_roots",
              [](TestContext& t) {
                  // a_n = 5 a_{n-1} - 6 a_{n-2}: characteristic x^2 - 5x + 6 = (x-2)(x-3),
                  // a case that splits completely over Q.
                  const std::vector<Rational> coeffs = {ri(5), ri(-6)};
                  const auto poly = characteristic_polynomial(coeffs).value();
                  t.expect(poly.is_equal(ipoly({6, -5, 1})),
                           "characteristic polynomial is x^2 - 5x + 6");
                  t.expect(all_roots_rational(coeffs).value() == true,
                           "characteristic polynomial splits over Q");
                  const auto roots = characteristic_roots(coeffs).value();
                  t.expect(roots.size() == 2, "two rational characteristic roots");
                  const auto m2 = find_root(roots, ri(2));
                  const auto m3 = find_root(roots, ri(3));
                  t.expect(m2 && *m2 == 1 && m3 && *m3 == 1, "roots are 2 and 3, simple");
                  t.expect(eval_at(poly, 2) == ri(0) && eval_at(poly, 3) == ri(0),
                           "both roots satisfy the characteristic polynomial");
              })
        // === ORTHOGONAL POLYNOMIALS ========================================
        .test("orthopoly_legendre_values_and_normalization",
              [](TestContext& t) {
                  t.expect(legendre(0).value().is_equal(ipoly({1})), "P_0 = 1");
                  t.expect(legendre(1).value().is_equal(ipoly({0, 1})), "P_1 = x");
                  t.expect(legendre(2).value().is_equal(rpoly({rat(-1, 2), ri(0), rat(3, 2)})),
                           "P_2 = (3x^2 - 1)/2");
                  t.expect(legendre(3).value().is_equal(
                               rpoly({ri(0), rat(-3, 2), ri(0), rat(5, 2)})),
                           "P_3 = (5x^3 - 3x)/2");
                  // Normalization P_n(1) == 1 for every order.
                  for (std::int64_t n = 0; n <= 5; ++n) {
                      t.expect(eval_at(legendre(n).value(), 1) == ri(1), "P_n(1) == 1");
                  }
              })
        .test("orthopoly_chebyshev_recurrence",
              [](TestContext& t) {
                  t.expect(chebyshev_t(0).value().is_equal(ipoly({1})), "T_0 = 1");
                  t.expect(chebyshev_t(1).value().is_equal(ipoly({0, 1})), "T_1 = x");
                  t.expect(chebyshev_t(2).value().is_equal(ipoly({-1, 0, 2})), "T_2 = 2x^2 - 1");
                  t.expect(chebyshev_t(3).value().is_equal(ipoly({0, -3, 0, 4})),
                           "T_3 = 4x^3 - 3x");
                  // Three-term recurrence T_4 == 2x*T_3 - T_2, built on the test side.
                  const auto t3 = chebyshev_t(3).value();
                  const auto t2 = chebyshev_t(2).value();
                  const auto recon =
                      t3.multiply(x_rp()).value().scale(ri(2)).value().subtract(t2).value();
                  t.expect(chebyshev_t(4).value().is_equal(recon),
                           "T_4 == 2x*T_3 - T_2");
                  t.expect(recon.is_equal(ipoly({1, 0, -8, 0, 8})), "T_4 = 8x^4 - 8x^2 + 1");
                  for (std::int64_t n = 0; n <= 5; ++n) {
                      t.expect(eval_at(chebyshev_t(n).value(), 1) == ri(1), "T_n(1) == 1");
                  }
              })
        .test("orthopoly_families_low_order",
              [](TestContext& t) {
                  t.expect(hermite_prob(2).value().is_equal(ipoly({-1, 0, 1})),
                           "He_2 = x^2 - 1");
                  t.expect(hermite_phys(2).value().is_equal(ipoly({-2, 0, 4})),
                           "H_2 = 4x^2 - 2");
                  t.expect(chebyshev_u(2).value().is_equal(ipoly({-1, 0, 4})),
                           "U_2 = 4x^2 - 1");
                  t.expect(laguerre(2).value().is_equal(rpoly({ri(1), ri(-2), rat(1, 2)})),
                           "L_2 = (x^2 - 4x + 2)/2");
              })
        // === COMBINATORICS =================================================
        .test("combinatorics_binomial_identities",
              [](TestContext& t) {
                  t.expect(binomial(10, 3).value() == 120 && binomial(10, 7).value() == 120,
                           "C(10,3) == C(10,7) == 120 (symmetry)");
                  t.expect(binomial(6, 2).value() == 15, "C(6,2) == 15");
                  // Pascal: C(10,4) == C(9,3) + C(9,4).
                  t.expect(binomial(10, 4).value() ==
                               binomial(9, 3).value() + binomial(9, 4).value(),
                           "C(n,k) == C(n-1,k-1) + C(n-1,k)");
                  // Row sum: sum_k C(n,k) == 2^n.
                  for (std::int64_t n = 0; n <= 10; ++n) {
                      std::int64_t sum = 0;
                      for (std::int64_t k = 0; k <= n; ++k) {
                          sum += binomial(n, k).value();
                      }
                      t.expect(sum == (std::int64_t{1} << n), "sum_k C(n,k) == 2^n");
                  }
              })
        .test("combinatorics_factorial_sequences",
              [](TestContext& t) {
                  t.expect(factorial(0).value() == 1 && factorial(1).value() == 1,
                           "0! == 1! == 1");
                  t.expect(factorial(5).value() == 120 && factorial(10).value() == 3628800,
                           "5! == 120, 10! == 3628800");
                  // Falling factorial identity P(n,k) == C(n,k) * k!.
                  for (const auto [n, k] : {std::pair{5, 2}, std::pair{7, 3}, std::pair{9, 4}}) {
                      t.expect(permutations(n, k).value() ==
                                   binomial(n, k).value() * factorial(k).value(),
                               "P(n,k) == C(n,k) * k!");
                  }
                  // Catalan numbers C_0..C_5.
                  const std::array<std::int64_t, 6> catalans = {1, 1, 2, 5, 14, 42};
                  for (std::int64_t n = 0; n <= 5; ++n) {
                      t.expect(catalan(n).value() == catalans[static_cast<std::size_t>(n)],
                               "Catalan number matches");
                  }
              })
        .test("combinatorics_bernoulli_and_stirling",
              [](TestContext& t) {
                  // Bernoulli numbers, first convention B_1 = -1/2.
                  t.expect(bernoulli(0).value() == ri(1), "B_0 == 1");
                  t.expect(bernoulli(1).value() == rat(-1, 2), "B_1 == -1/2");
                  t.expect(bernoulli(2).value() == rat(1, 6), "B_2 == 1/6");
                  t.expect(bernoulli(3).value() == ri(0), "B_3 == 0");
                  t.expect(bernoulli(4).value() == rat(-1, 30), "B_4 == -1/30");
                  t.expect(bernoulli(6).value() == rat(1, 42), "B_6 == 1/42");
                  // Stirling second kind: row sum is the Bell number B_5 == 52.
                  t.expect(stirling_second(4, 2).value() == 7, "S(4,2) == 7");
                  std::int64_t bell = 0;
                  for (std::int64_t k = 0; k <= 5; ++k) {
                      bell += stirling_second(5, k).value();
                  }
                  t.expect(bell == 52, "sum_k S(5,k) == Bell(5) == 52");
                  // Unsigned Stirling first kind: row sum is n! == 120 for n = 5.
                  t.expect(stirling_first(4, 2).value() == 11, "c(4,2) == 11");
                  std::int64_t row = 0;
                  for (std::int64_t k = 0; k <= 5; ++k) {
                      row += stirling_first(5, k).value();
                  }
                  t.expect(row == factorial(5).value(), "sum_k c(5,k) == 5! == 120");
              })
        .run();
}
