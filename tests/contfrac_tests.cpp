// Tests for nimblecas.contfrac: simple continued fractions of rationals and their
// convergents/reconstruction round-trip, the periodic CF of quadratic irrationals
// (with the Pell identity for sqrt(2)'s convergents), and the Viskovatov series-to-
// C-fraction expansion (matching Pade, with an honest breakdown case).
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.contfrac;
import nimblecas.testing;

using nimblecas::MathError;
using nimblecas::PeriodicCF;
using nimblecas::Rational;
using nimblecas::SeriesCF;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

[[nodiscard]] auto ri(std::int64_t v) -> Rational {
    return Rational::from_int(v);
}

[[nodiscard]] auto rats(std::vector<std::int64_t> ns, std::int64_t d) -> std::vector<Rational> {
    std::vector<Rational> out;
    out.reserve(ns.size());
    for (const std::int64_t n : ns) {
        out.push_back(Rational::make(n, d).value());
    }
    return out;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.contfrac")
        .test("from_rational_355_over_113_round_trips",
              [](TestContext& t) {
                  // 355/113 = [3; 7, 16] (Euclid: 355=3*113+16, 113=7*16+1, 16=16*1).
                  const auto cf = nimblecas::from_rational(rat(355, 113)).value();
                  t.expect(cf == std::vector<std::int64_t>{3, 7, 16}, "355/113 = [3; 7, 16]");
                  // reconstruct must invert from_rational exactly.
                  t.expect(nimblecas::reconstruct(std::span<const std::int64_t>{cf}).value() ==
                               rat(355, 113),
                           "reconstruct([3;7,16]) = 355/113");
                  // Canonical last quotient >= 2 for a non-integer.
                  t.expect(cf.back() >= 2, "canonical form: last quotient >= 2");
              })
        .test("from_rational_integer_and_negative",
              [](TestContext& t) {
                  t.expect(nimblecas::from_rational(ri(5)).value() ==
                               std::vector<std::int64_t>{5},
                           "integer 5 = [5]");
                  t.expect(nimblecas::from_rational(ri(0)).value() ==
                               std::vector<std::int64_t>{0},
                           "0 = [0]");
                  // -7/3 = [-3; 1, 2]  ( -7/3 = -3 + 2/3, 1/(2/3)=3/2=[1;2] ). Round-trips.
                  const auto cf = nimblecas::from_rational(rat(-7, 3)).value();
                  t.expect(cf == std::vector<std::int64_t>{-3, 1, 2}, "-7/3 = [-3; 1, 2]");
                  t.expect(nimblecas::reconstruct(std::span<const std::int64_t>{cf}).value() ==
                               rat(-7, 3),
                           "reconstruct([-3;1,2]) = -7/3");
              })
        .test("convergents_of_355_over_113",
              [](TestContext& t) {
                  const std::vector<std::int64_t> cf{3, 7, 16};
                  const auto conv = nimblecas::convergents(std::span<const std::int64_t>{cf}).value();
                  t.expect(conv.size() == 3, "three convergents");
                  t.expect(conv[0] == ri(3), "p0/q0 = 3/1");
                  t.expect(conv[1] == rat(22, 7), "p1/q1 = 22/7");
                  t.expect(conv[2] == rat(355, 113), "p2/q2 = 355/113");
                  // Last convergent equals reconstruct.
                  t.expect(conv.back() ==
                               nimblecas::reconstruct(std::span<const std::int64_t>{cf}).value(),
                           "last convergent == reconstruct");
                  // Empty list has no convergents.
                  t.expect(nimblecas::convergents(std::span<const std::int64_t>{}).error() ==
                               MathError::domain_error,
                           "convergents([]) = domain_error");
              })
        .test("sqrt2_periodic_cf_and_pell_identity",
              [](TestContext& t) {
                  // sqrt(2) = [1; (2)]: prefix [1], period [2] (length 1).
                  const auto q = nimblecas::quadratic_irrational_cf(2).value();
                  t.expect(q.prefix == std::vector<std::int64_t>{1}, "sqrt(2) prefix = [1]");
                  t.expect(q.period == std::vector<std::int64_t>{2}, "sqrt(2) period = (2)");

                  // Unroll [1; 2, 2, 2, ...] and check the Pell identity p_k^2 - 2 q_k^2 = +-1
                  // for every convergent p_k/q_k (so p_k/q_k -> sqrt(2)).
                  std::vector<std::int64_t> cf{1};
                  for (int i = 0; i < 8; ++i) {
                      cf.push_back(2);
                  }
                  const auto conv = nimblecas::convergents(std::span<const std::int64_t>{cf}).value();
                  bool pell_ok = true;
                  for (const Rational& c : conv) {
                      const std::int64_t p = c.numerator();
                      const std::int64_t qd = c.denominator();  // coprime => these are p_k, q_k
                      const std::int64_t lhs = p * p - 2 * qd * qd;
                      if (lhs != 1 && lhs != -1) {
                          pell_ok = false;
                      }
                  }
                  t.expect(pell_ok, "every sqrt(2) convergent satisfies p^2 - 2 q^2 = +-1");
                  // Convergent 7/5 appears (third term) as a concrete anchor.
                  t.expect(conv[2] == rat(7, 5), "third convergent of sqrt(2) is 7/5");
              })
        .test("sqrt23_known_period",
              [](TestContext& t) {
                  // sqrt(23) = [4; (1, 3, 1, 8)], period length 4.
                  const auto q = nimblecas::quadratic_irrational_cf(23).value();
                  t.expect(q.prefix == std::vector<std::int64_t>{4}, "sqrt(23) prefix = [4]");
                  t.expect(q.period == std::vector<std::int64_t>{1, 3, 1, 8},
                           "sqrt(23) period = (1, 3, 1, 8)");
                  // sqrt(7) = [2; (1, 1, 1, 4)] as a second independent check.
                  const auto q7 = nimblecas::quadratic_irrational_cf(7).value();
                  t.expect(q7.prefix == std::vector<std::int64_t>{2}, "sqrt(7) prefix = [2]");
                  t.expect(q7.period == std::vector<std::int64_t>{1, 1, 1, 4},
                           "sqrt(7) period = (1, 1, 1, 4)");
              })
        .test("quadratic_irrational_domain_errors",
              [](TestContext& t) {
                  t.expect(nimblecas::quadratic_irrational_cf(9).error() == MathError::domain_error,
                           "perfect square 9 => domain_error");
                  t.expect(nimblecas::quadratic_irrational_cf(1).error() == MathError::domain_error,
                           "perfect square 1 => domain_error");
                  t.expect(nimblecas::quadratic_irrational_cf(0).error() == MathError::domain_error,
                           "D = 0 => domain_error");
                  t.expect(nimblecas::quadratic_irrational_cf(-5).error() == MathError::domain_error,
                           "negative D => domain_error");
              })
        .test("viskovatov_geometric_series_terminates",
              [](TestContext& t) {
                  // 1/(1-x) = 1 + x + x^2 + x^3 + ...  =>  (b0; a) = (1; 1, -1), since
                  // 1 + (1*x)/(1 + (-1*x)/1) = 1 + x/(1-x) = 1/(1-x) exactly.
                  const auto c = rats({1, 1, 1, 1}, 1);
                  const auto cf = nimblecas::viskovatov(std::span<const Rational>{c}).value();
                  t.expect(cf.b0 == ri(1), "b0 = 1");
                  t.expect(cf.a.size() == 2, "C-fraction terminates after 2 partial numerators");
                  t.expect(cf.a[0] == ri(1), "a1 = 1");
                  t.expect(cf.a[1] == ri(-1), "a2 = -1");
              })
        .test("viskovatov_exp_matches_pade_staircase",
              [](TestContext& t) {
                  // exp: c = {1, 1, 1/2, 1/6}  =>  (b0; a) = (1; 1, -1/2, 1/6). The two-term
                  // convergent 1 + x/(1 - x/2) = (1 + x/2)/(1 - x/2) is exactly the [1/1] Pade
                  // approximant of exp (see docs/reference/pade.md).
                  std::vector<Rational> c{ri(1), ri(1), rat(1, 2), rat(1, 6)};
                  const auto cf = nimblecas::viskovatov(std::span<const Rational>{c}).value();
                  t.expect(cf.b0 == ri(1), "b0 = 1");
                  t.expect(cf.a.size() == 3, "three partial numerators from four coefficients");
                  t.expect(cf.a[0] == ri(1), "a1 = 1");
                  t.expect(cf.a[1] == rat(-1, 2), "a2 = -1/2");
                  t.expect(cf.a[2] == rat(1, 6), "a3 = 1/6");
              })
        .test("viskovatov_breakdown_and_edge_cases",
              [](TestContext& t) {
                  // f = 1 + x^2: first non-constant term is x^2, so the regular C-fraction
                  // (partial numerator a1*x) cannot represent it: honest not_implemented.
                  std::vector<Rational> bad{ri(1), ri(0), ri(1)};
                  t.expect(nimblecas::viskovatov(std::span<const Rational>{bad}).error() ==
                               MathError::not_implemented,
                           "zero pivot (1 + x^2) => not_implemented, never a wrong CF");
                  // A pure constant series {5, 0, 0}: b0 = 5, no partial numerators.
                  std::vector<Rational> konst{ri(5), ri(0), ri(0)};
                  const auto ck = nimblecas::viskovatov(std::span<const Rational>{konst}).value();
                  t.expect(ck.b0 == ri(5) && ck.a.empty(), "constant series => (5; )");
                  // Empty coefficient list is a domain_error.
                  t.expect(nimblecas::viskovatov(std::span<const Rational>{}).error() ==
                               MathError::domain_error,
                           "viskovatov([]) => domain_error");
              })
        .run();
}
