// Tests for nimblecas.probmethod: exact, SOUND existence proofs by the probabilistic
// method — first/second moment, union bound, the symmetric and asymmetric Lovász Local
// Lemma (with the rigorous rational enclosure of e), and first-moment Ramsey bounds.
// Every expected value is hand-verified; the LLL tests additionally check that every
// certified "exists" is genuinely sound (e·p·(d+1) ≤ 1 for real).
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.bigint;
import nimblecas.probmethod;
import nimblecas.testing;

using nimblecas::Existence;
using nimblecas::LovaszVerdict;
using nimblecas::MathError;
using nimblecas::Rational;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}
[[nodiscard]] auto ri(std::int64_t v) -> Rational { return Rational::from_int(v); }

// Exact Rational comparison for the tests, by __int128 cross-multiplication (canonical
// denominators are positive). Mirrors the module's internal comparator so we can assert
// inequalities on the exact quantities the verdicts report.
[[nodiscard]] auto rcmp(const Rational& a, const Rational& b) -> int {
    const __int128 lhs = static_cast<__int128>(a.numerator()) * b.denominator();
    const __int128 rhs = static_cast<__int128>(b.numerator()) * a.denominator();
    return lhs < rhs ? -1 : (lhs > rhs ? 1 : 0);
}
[[nodiscard]] auto rle(const Rational& a, const Rational& b) -> bool { return rcmp(a, b) <= 0; }

[[nodiscard]] auto sp(const std::vector<Rational>& v) -> std::span<const Rational> {
    return std::span<const Rational>{v};
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.probmethod")
        .test("first_moment_expectation_argument",
              [](TestContext& t) {
                  // E[X] = 1/2 < 1  =>  P(X=0) > 0  =>  a bad-event-free config EXISTS.
                  auto r = nimblecas::first_moment_exists(rat(1, 2)).value();
                  t.expect(r.verdict == Existence::exists, "E[X]=1/2 < 1 => exists");
                  t.expect(r.expected_value == rat(1, 2), "echoes E[X] = 1/2");
                  // The averaging dual always holds, independent of the < 1 test.
                  t.expect(r.attains_at_least_mean && r.attains_at_most_mean,
                           "averaging: outcomes above and below the mean both exist");
                  // E[X] = 2 >= 1  =>  the first-moment argument does not apply.
                  auto r2 = nimblecas::first_moment_exists(ri(2)).value();
                  t.expect(r2.verdict == Existence::not_certified, "E[X]=2 => not_certified");
                  // Exactly E[X] = 1 is not < 1, so not certified (boundary).
                  t.expect(nimblecas::first_moment_exists(ri(1)).value().verdict ==
                               Existence::not_certified,
                           "E[X]=1 is the boundary => not_certified");
                  // A negative expectation is impossible for a non-negative counter.
                  t.expect(nimblecas::first_moment_exists(rat(-1, 3)).error() ==
                               MathError::domain_error,
                           "negative E[X] => domain_error");
              })
        .test("union_bound_over_bad_events",
              [](TestContext& t) {
                  // Probabilities summing to 3/4 < 1  =>  some outcome avoids all => exists.
                  const std::vector<Rational> good{rat(1, 4), rat(1, 4), rat(1, 4)};
                  auto r = nimblecas::union_bound_exists(sp(good)).value();
                  t.expect(r.verdict == Existence::exists, "sum 3/4 < 1 => exists");
                  t.expect(r.total_probability == rat(3, 4), "exact total probability = 3/4");
                  // Probabilities summing to 5/4 >= 1  =>  union bound is vacuous.
                  const std::vector<Rational> bad{rat(1, 2), rat(1, 2), rat(1, 4)};
                  auto r2 = nimblecas::union_bound_exists(sp(bad)).value();
                  t.expect(r2.verdict == Existence::not_certified, "sum 5/4 => not_certified");
                  t.expect(r2.total_probability == rat(5, 4), "exact total probability = 5/4");
                  // Empty list: no bad events, so any configuration is good (vacuous exists).
                  const std::vector<Rational> none{};
                  auto r3 = nimblecas::union_bound_exists(sp(none)).value();
                  t.expect(r3.verdict == Existence::exists && r3.total_probability == ri(0),
                           "empty => total 0 => exists");
                  // An entry outside [0,1] is not a probability.
                  const std::vector<Rational> illegal{rat(3, 2)};
                  t.expect(nimblecas::union_bound_exists(sp(illegal)).error() ==
                               MathError::domain_error,
                           "probability > 1 => domain_error");
              })
        .test("second_moment_chebyshev_bound",
              [](TestContext& t) {
                  // mean = 10, var = 5:  P(X=0) <= Var/E^2 = 5/100 = 1/20 < 1  =>  X>0 exists.
                  auto r = nimblecas::second_moment_positive(ri(10), ri(5)).value();
                  t.expect(r.prob_zero_bound == rat(1, 20), "P(X=0) <= 1/20 exactly");
                  t.expect(r.verdict == Existence::exists, "1/20 < 1 => exists");
                  // Var = E^2 gives bound 1 (not < 1): the argument does not certify.
                  auto r2 = nimblecas::second_moment_positive(ri(3), ri(9)).value();
                  t.expect(r2.prob_zero_bound == ri(1), "Var=E^2 => bound = 1");
                  t.expect(r2.verdict == Existence::not_certified, "bound 1 => not_certified");
                  // E[X] = 0 is guarded (division), as is a negative variance.
                  t.expect(nimblecas::second_moment_positive(ri(0), ri(5)).error() ==
                               MathError::domain_error,
                           "E[X]=0 => domain_error");
                  t.expect(nimblecas::second_moment_positive(ri(10), rat(-1, 2)).error() ==
                               MathError::domain_error,
                           "negative variance => domain_error");
              })
        .test("lovasz_symmetric_e_enclosure_is_sound",
              [](TestContext& t) {
                  // p = 1/100, d = 3:  c = p(d+1) = 1/25.  e*c = 4e/100 ~ 0.1087 <= 1 => exists.
                  auto r = nimblecas::lovasz_symmetric(rat(1, 100), 3).value();
                  t.expect(r.verdict == LovaszVerdict::exists, "1/25 well below 1/e => exists");
                  t.expect(r.constraint == rat(1, 25), "constraint p(d+1) = 1/25 exactly");
                  // SOUNDNESS: a certified exists must have c * e_hi <= 1 (hence e*c < 1 truly),
                  // computed against the rigorous rational UPPER bound of e.
                  auto witness = r.constraint.multiply(nimblecas::e_upper_bound()).value();
                  t.expect(rle(witness, ri(1)),
                           "certificate is sound: p(d+1)*e_upper <= 1, so e*p*(d+1) < 1");

                  // p = 1/2, d = 3:  c = 2.  e*c = 2e ~ 5.4 > 1  =>  condition provably fails.
                  auto r2 = nimblecas::lovasz_symmetric(rat(1, 2), 3).value();
                  t.expect(r2.verdict == LovaszVerdict::condition_fails, "2e > 1 => condition_fails");
                  t.expect(r2.constraint == ri(2), "constraint = 2");
                  // And its failure is real: c * e_lo > 1 against the rational LOWER bound of e.
                  auto witness2 = r2.constraint.multiply(nimblecas::e_lower_bound()).value();
                  t.expect(!rle(witness2, ri(1)),
                           "failure is real: p(d+1)*e_lower > 1, so e*p*(d+1) > 1");

                  // A constraint inside the thin rational sliver around 1/e that the enclosure
                  // cannot resolve: c = 3678795/10000000 lies in (1/e_hi, 1/e_lo].
                  auto r3 = nimblecas::lovasz_symmetric(rat(3678795, 10000000), 0).value();
                  t.expect(r3.verdict == LovaszVerdict::indeterminate,
                           "c in the e-enclosure gap => indeterminate (honest, not guessed)");

                  // Domain guards: p outside [0,1], negative d.
                  t.expect(nimblecas::lovasz_symmetric(rat(3, 2), 1).error() ==
                               MathError::domain_error,
                           "p > 1 => domain_error");
                  t.expect(nimblecas::lovasz_symmetric(rat(1, 10), -1).error() ==
                               MathError::domain_error,
                           "negative d => domain_error");
              })
        .test("lovasz_asymmetric_per_event_check",
              [](TestContext& t) {
                  // Two mutually dependent events; weights x_1 = x_2 = 1/4.
                  // rhs_i = x_i (1 - x_j) = (1/4)(3/4) = 3/16.
                  const std::vector<std::vector<int>> dep{{1}, {0}};
                  const std::vector<Rational> x{rat(1, 4), rat(1, 4)};
                  // P(A_i) = 1/8 = 2/16 <= 3/16  =>  certified, slack 1/16 each.
                  const std::vector<Rational> p_ok{rat(1, 8), rat(1, 8)};
                  auto r = nimblecas::lovasz_asymmetric(sp(p_ok), dep, sp(x)).value();
                  t.expect(r.verdict == Existence::exists, "1/8 <= 3/16 for both => certified");
                  t.expect(r.first_violation == 2, "no violation (index == size)");
                  t.expect(r.slack.size() == 2 && r.slack[0] == rat(1, 16) &&
                               r.slack[1] == rat(1, 16),
                           "exact per-event slack = 3/16 - 1/8 = 1/16");
                  // P(A_0) = 1/4 = 4/16 > 3/16  =>  event 0 violates; not certified.
                  const std::vector<Rational> p_bad{rat(1, 4), rat(1, 8)};
                  auto r2 = nimblecas::lovasz_asymmetric(sp(p_bad), dep, sp(x)).value();
                  t.expect(r2.verdict == Existence::not_certified, "1/4 > 3/16 => not certified");
                  t.expect(r2.first_violation == 0, "first violation at event 0");
                  t.expect(r2.slack[0] == rat(-1, 16), "negative slack 3/16 - 1/4 = -1/16");
                  // A weight on the boundary (x = 1) is outside the required open interval.
                  const std::vector<Rational> x_bad{ri(1), rat(1, 4)};
                  t.expect(nimblecas::lovasz_asymmetric(sp(p_ok), dep, sp(x_bad)).error() ==
                               MathError::domain_error,
                           "x_i = 1 (not in (0,1)) => domain_error");
                  // An out-of-range neighbour index.
                  const std::vector<std::vector<int>> dep_bad{{5}, {0}};
                  t.expect(nimblecas::lovasz_asymmetric(sp(p_ok), dep_bad, sp(x)).error() ==
                               MathError::domain_error,
                           "neighbour index out of range => domain_error");
              })
        .test("ramsey_first_moment_lower_bounds",
              [](TestContext& t) {
                  // R(3,3): threshold 2^{C(3,2)-1} = 2^2 = 4. Largest n with C(n,3) < 4:
                  //   C(3,3)=1<4 (ok), C(4,3)=4 (not < 4) => largest n = 3, so R(3,3) > 3.
                  auto r3 = nimblecas::ramsey_lower_bound(3).value();
                  t.expect(r3.largest_n == 3, "R(3,3) > 3 (largest certified n)");
                  t.expect(r3.threshold == nimblecas::BigInt::from_u64(4), "threshold 2^2 = 4");
                  t.expect(nimblecas::ramsey_certifies(3, 3).value(), "C(3,3)=1 < 4 certifies");
                  t.expect(!nimblecas::ramsey_certifies(4, 3).value(),
                           "C(4,3)=4 is not < 4 (boundary, not certified)");

                  // R(4,4): threshold 2^{C(4,2)-1} = 2^5 = 32. Largest n with C(n,4) < 32:
                  //   C(6,4)=15<32 (ok), C(7,4)=35 (not < 32) => largest n = 6, so R(4,4) > 6.
                  auto r4 = nimblecas::ramsey_lower_bound(4).value();
                  t.expect(r4.largest_n == 6, "R(4,4) > 6 (largest certified n)");
                  t.expect(r4.threshold == nimblecas::BigInt::from_u64(32), "threshold 2^5 = 32");
                  t.expect(nimblecas::ramsey_certifies(6, 4).value(), "C(6,4)=15 < 32 certifies");
                  t.expect(!nimblecas::ramsey_certifies(7, 4).value(), "C(7,4)=35 not < 32");

                  // Domain guards: k < 2 has no C(k,2)-based threshold; n < 0 is undefined.
                  t.expect(nimblecas::ramsey_lower_bound(1).error() == MathError::domain_error,
                           "k < 2 => domain_error");
                  t.expect(nimblecas::ramsey_certifies(-1, 3).error() == MathError::domain_error,
                           "n < 0 => domain_error");
              })
        .run();
}
