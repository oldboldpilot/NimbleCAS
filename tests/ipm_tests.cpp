// Tests for nimblecas.ipm: the numeric primal-dual interior-point LP solver.
// @author Olumuyiwa Oluwasanmi
//
// Unlike the exact-rational Simplex tests (tests/lp_tests.cpp), these compare against
// KNOWN exact optima only to within a tolerance: an interior-point method converges to
// the central-path limit in floating point, so equality is replaced by |got - exp| < tol.

import std;
import nimblecas.core;
import nimblecas.ipm;
import nimblecas.testing;

using nimblecas::IpmStatus;
using nimblecas::MathError;
using nimblecas::solve_ipm;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Numeric closeness for the accuracy asserts; the solver targets ~1e-9, so 1e-6 is a
// comfortable, non-flaky tolerance for the reported optimum.
[[nodiscard]] auto approx(double a, double b, double tol) -> bool {
    return std::fabs(a - b) < tol;
}

inline constexpr double kTol = 1e-6;

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.ipm")
        .test("classic_max_2x_3y",
              [](TestContext& t) {
                  // max 2x + 3y  s.t.  x + y <= 4, x + 2y <= 5, x,y >= 0.
                  // Convert to standard form by adding slacks s1, s2 and negating the
                  // objective (max -> min):  min -2x - 3y  s.t.
                  //   x +  y + s1      = 4
                  //   x + 2y      + s2 = 5,   all vars >= 0.
                  // Known optimum: x = 3, y = 1, s1 = s2 = 0, max = 9  (min = -9).
                  const std::vector<std::vector<double>> A{{1, 1, 1, 0}, {1, 2, 0, 1}};
                  const std::vector<double> b{4, 5};
                  const std::vector<double> c{-2, -3, 0, 0};
                  auto r = solve_ipm(A, b, c);
                  t.expect(r.has_value(), "solve returned a value");
                  if (r.has_value()) {
                      t.expect(r->status == IpmStatus::optimal, "status is optimal");
                      t.expect(approx(r->objective, -9.0, kTol), "objective is -9 (max 9)");
                      t.expect(approx(r->x[0], 3.0, kTol), "x = 3");
                      t.expect(approx(r->x[1], 1.0, kTol), "y = 1");
                      t.expect(approx(r->x[2], 0.0, kTol), "slack s1 = 0 (binding)");
                      t.expect(approx(r->x[3], 0.0, kTol), "slack s2 = 0 (binding)");
                  }
              })
        .test("box_optimum",
              [](TestContext& t) {
                  // max x + y  s.t.  x <= 4, y <= 3  ->  min -x - y with slacks.
                  //   x + s1      = 4
                  //   y      + s2 = 3.   Known optimum: x = 4, y = 3, max = 7.
                  const std::vector<std::vector<double>> A{{1, 0, 1, 0}, {0, 1, 0, 1}};
                  const std::vector<double> b{4, 3};
                  const std::vector<double> c{-1, -1, 0, 0};
                  auto r = solve_ipm(A, b, c);
                  t.expect(r.has_value(), "solve returned a value");
                  if (r.has_value()) {
                      t.expect(r->status == IpmStatus::optimal, "status is optimal");
                      t.expect(approx(r->objective, -7.0, kTol), "objective is -7 (max 7)");
                      t.expect(approx(r->x[0], 4.0, kTol), "x = 4");
                      t.expect(approx(r->x[1], 3.0, kTol), "y = 3");
                  }
              })
        .test("pure_standard_form_equality",
              [](TestContext& t) {
                  // A native standard-form LP (no slack conversion needed):
                  //   min x1 + 2 x2  s.t.  x1 + x2 = 1,  x >= 0.
                  // Known optimum: x1 = 1, x2 = 0, objective = 1. The dual is the shadow
                  // price y1 of the single equality: y1 = 1 (= reduced cost balance).
                  const std::vector<std::vector<double>> A{{1, 1}};
                  const std::vector<double> b{1};
                  const std::vector<double> c{1, 2};
                  auto r = solve_ipm(A, b, c);
                  t.expect(r.has_value(), "solve returned a value");
                  if (r.has_value()) {
                      t.expect(r->status == IpmStatus::optimal, "status is optimal");
                      t.expect(approx(r->objective, 1.0, kTol), "objective is 1");
                      t.expect(approx(r->x[0], 1.0, kTol), "x1 = 1");
                      t.expect(approx(r->x[1], 0.0, kTol), "x2 = 0");
                      t.expect(r->y.size() == 1 && approx(r->y[0], 1.0, kTol),
                               "dual/shadow price y1 = 1");
                  }
              })
        .test("degenerate_vertex",
              [](TestContext& t) {
                  // max x + y  s.t.  2x + y <= 4, x + 2y <= 4  ->  optimum 8/3 at
                  // (4/3, 4/3).  min -x - y with slacks; a genuinely fractional vertex,
                  // recovered to tolerance rather than as an exact rational.
                  const std::vector<std::vector<double>> A{{2, 1, 1, 0}, {1, 2, 0, 1}};
                  const std::vector<double> b{4, 4};
                  const std::vector<double> c{-1, -1, 0, 0};
                  auto r = solve_ipm(A, b, c);
                  t.expect(r.has_value(), "solve returned a value");
                  if (r.has_value()) {
                      t.expect(r->status == IpmStatus::optimal, "status is optimal");
                      t.expect(approx(r->objective, -8.0 / 3.0, kTol), "objective is -8/3");
                      t.expect(approx(r->x[0], 4.0 / 3.0, kTol), "x = 4/3");
                      t.expect(approx(r->x[1], 4.0 / 3.0, kTol), "y = 4/3");
                  }
              })
        .test("unbounded",
              [](TestContext& t) {
                  // min -x1  s.t.  x1 - x2 = 0,  x >= 0.  x1 = x2 may grow without bound
                  // while the objective -x1 decreases without bound: primal unbounded.
                  const std::vector<std::vector<double>> A{{1, -1}};
                  const std::vector<double> b{0};
                  const std::vector<double> c{-1, 0};
                  auto r = solve_ipm(A, b, c);
                  // Honesty check: the solver must NOT report a (false) optimum. The
                  // primal iterate diverges, so it is classified unbounded.
                  t.expect(!r.has_value() || r->status != IpmStatus::optimal,
                           "unbounded LP is never reported optimal");
                  // Stronger: the solver actually detects and classifies this as unbounded
                  // (not merely a non-optimal/not_implemented fallthrough).
                  t.expect(r.has_value() && r->status == IpmStatus::unbounded,
                           "status is detected as unbounded");
              })
        .test("infeasible",
              [](TestContext& t) {
                  // x1 + x2 = -1 with x >= 0 is impossible (a non-negative sum cannot be
                  // negative); x3 = 1 keeps the second row full-rank so the normal matrix
                  // stays factorable. Primal infeasible -> dual unbounded.
                  const std::vector<std::vector<double>> A{{1, 1, 0}, {0, 0, 1}};
                  const std::vector<double> b{-1, 1};
                  const std::vector<double> c{1, 1, 1};
                  auto r = solve_ipm(A, b, c);
                  // Honesty check: never a fabricated optimum. Detection classifies this
                  // as infeasible; an honest MathError is also acceptable, but "optimal"
                  // is not.
                  t.expect(!r.has_value() || r->status != IpmStatus::optimal,
                           "infeasible LP is never reported optimal");
                  // Stronger: the solver actually detects and classifies this as infeasible
                  // (not merely a non-optimal/not_implemented fallthrough).
                  t.expect(r.has_value() && r->status == IpmStatus::infeasible,
                           "status is detected as infeasible");
              })
        .test("domain_errors",
              [](TestContext& t) {
                  const std::vector<std::vector<double>> A{{1, 1, 1, 0}, {1, 2, 0, 1}};
                  const std::vector<double> c{-2, -3, 0, 0};

                  // b length disagreeing with the number of constraints m.
                  auto r1 = solve_ipm(A, std::vector<double>{4}, c);
                  t.expect(!r1.has_value() && r1.error() == MathError::domain_error,
                           "b length mismatch is a domain error");

                  // Ragged A: the second row's width != n = c.size().
                  const std::vector<std::vector<double>> A_ragged{{1, 1, 1, 0}, {1, 2}};
                  auto r2 = solve_ipm(A_ragged, std::vector<double>{4, 5}, c);
                  t.expect(!r2.has_value() && r2.error() == MathError::domain_error,
                           "ragged A is a domain error");

                  // Empty problem (m == 0).
                  auto r3 = solve_ipm({}, {}, c);
                  t.expect(!r3.has_value() && r3.error() == MathError::domain_error,
                           "empty A is a domain error");

                  // Non-finite datum.
                  const std::vector<double> b_nan{std::numeric_limits<double>::quiet_NaN(), 5};
                  auto r4 = solve_ipm(A, b_nan, c);
                  t.expect(!r4.has_value() && r4.error() == MathError::domain_error,
                           "non-finite b is a domain error");
              })
        .run();
}
