// Tests for nimblecas.lp: the exact-rational Simplex method.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.lp;
import nimblecas.testing;

using nimblecas::LpStatus;
using nimblecas::MathError;
using nimblecas::maximize;
using nimblecas::Rational;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

// An integer-valued Rational, spelled to match the exact-arithmetic intent.
[[nodiscard]] auto ri(std::int64_t v) -> Rational {
    return Rational::from_int(v);
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.lp")
        .test("box_optimum",
              [](TestContext& t) {
                  // max x + y  s.t.  x <= 4, y <= 3  =>  optimum 7 at (4, 3).
                  const std::vector<std::vector<Rational>> A{{ri(1), ri(0)}, {ri(0), ri(1)}};
                  const std::vector<Rational> b{ri(4), ri(3)};
                  const std::vector<Rational> c{ri(1), ri(1)};
                  auto sol = maximize(A, b, c).value();
                  t.expect(sol.status == LpStatus::optimal, "status is optimal");
                  t.expect(sol.value == ri(7), "objective value is 7");
                  t.expect(sol.solution.size() == 2, "two original variables reported");
                  t.expect(sol.solution[0] == ri(4), "x = 4");
                  t.expect(sol.solution[1] == ri(3), "y = 3");
              })
        .test("vertex_optimum",
              [](TestContext& t) {
                  // max 3x + 2y  s.t.  x + y <= 4, x + 3y <= 6  =>  optimum 12 at (4, 0).
                  const std::vector<std::vector<Rational>> A{{ri(1), ri(1)}, {ri(1), ri(3)}};
                  const std::vector<Rational> b{ri(4), ri(6)};
                  const std::vector<Rational> c{ri(3), ri(2)};
                  auto sol = maximize(A, b, c).value();
                  t.expect(sol.status == LpStatus::optimal, "status is optimal");
                  t.expect(sol.value == ri(12), "objective value is 12");
                  t.expect(sol.solution[0] == ri(4), "x = 4");
                  t.expect(sol.solution[1] == ri(0), "y = 0");
              })
        .test("fractional_optimum",
              [](TestContext& t) {
                  // max x + y  s.t.  2x + y <= 3, x + 2y <= 3  =>  optimum 2 at (1, 1).
                  // The intermediate pivots pass through 3/2 values; exact rationals must
                  // land precisely on the integer vertex, not a rounded neighbour.
                  const std::vector<std::vector<Rational>> A{{ri(2), ri(1)}, {ri(1), ri(2)}};
                  const std::vector<Rational> b{ri(3), ri(3)};
                  const std::vector<Rational> c{ri(1), ri(1)};
                  auto sol = maximize(A, b, c).value();
                  t.expect(sol.status == LpStatus::optimal, "status is optimal");
                  t.expect(sol.value == ri(2), "objective value is exactly 2");
                  t.expect(sol.solution[0] == ri(1), "x = 1 exactly");
                  t.expect(sol.solution[1] == ri(1), "y = 1 exactly");
              })
        .test("genuine_fraction",
              [](TestContext& t) {
                  // max x + y  s.t.  2x + y <= 4, x + 2y <= 4  =>  optimum 8/3 at (4/3, 4/3).
                  // A truly non-integral vertex, to confirm the fractions survive exactly.
                  const std::vector<std::vector<Rational>> A{{ri(2), ri(1)}, {ri(1), ri(2)}};
                  const std::vector<Rational> b{ri(4), ri(4)};
                  const std::vector<Rational> c{ri(1), ri(1)};
                  auto sol = maximize(A, b, c).value();
                  t.expect(sol.status == LpStatus::optimal, "status is optimal");
                  t.expect(sol.value == rat(8, 3), "objective value is exactly 8/3");
                  t.expect(sol.solution[0] == rat(4, 3), "x = 4/3 exactly");
                  t.expect(sol.solution[1] == rat(4, 3), "y = 4/3 exactly");
              })
        .test("unbounded",
              [](TestContext& t) {
                  // max x  s.t.  x - y <= 1  =>  x can grow without bound (increase y too).
                  const std::vector<std::vector<Rational>> A{{ri(1), ri(-1)}};
                  const std::vector<Rational> b{ri(1)};
                  const std::vector<Rational> c{ri(1), ri(0)};
                  auto sol = maximize(A, b, c).value();
                  t.expect(sol.status == LpStatus::unbounded, "status is unbounded");
                  t.expect(sol.solution.empty(), "no solution vector when unbounded");
              })
        .test("domain_errors",
              [](TestContext& t) {
                  // A negative right-hand side violates b >= 0.
                  const std::vector<std::vector<Rational>> A1{{ri(1), ri(0)}, {ri(0), ri(1)}};
                  const std::vector<Rational> b_neg{ri(4), ri(-1)};
                  const std::vector<Rational> c1{ri(1), ri(1)};
                  auto r1 = maximize(A1, b_neg, c1);
                  t.expect(!r1.has_value() && r1.error() == MathError::domain_error,
                           "negative b entry is a domain error");

                  // A ragged A: the second row has a different width than c.
                  const std::vector<std::vector<Rational>> A_ragged{{ri(1), ri(0)}, {ri(1)}};
                  const std::vector<Rational> b2{ri(4), ri(3)};
                  auto r2 = maximize(A_ragged, b2, c1);
                  t.expect(!r2.has_value() && r2.error() == MathError::domain_error,
                           "ragged A is a domain error");

                  // b length disagreeing with the number of constraints.
                  const std::vector<Rational> b_short{ri(4)};
                  auto r3 = maximize(A1, b_short, c1);
                  t.expect(!r3.has_value() && r3.error() == MathError::domain_error,
                           "b length mismatch is a domain error");
              })
        .run();
}
