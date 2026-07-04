// Tests for nimblecas.dae: exact linear index-1 differential-algebraic equations.
// @author Olumuyiwa Oluwasanmi
//
// Every case uses exact rational coefficients hand-verified in the comments, with no
// floating point anywhere. Benchmarks: the scalar x'=x with constraint y=x (both the exp
// series), a 2-differential + 1-algebraic harmonic DAE whose algebraic variable is the
// exact sin+cos combination, a case driven by nonzero polynomial forcings p(t) and q(t)
// through a nonzero coupling B, and the domain_error guards (a singular D reported as
// higher-index, a shape mismatch, and order 0).

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.powerseries;
import nimblecas.dae;
import nimblecas.testing;

using nimblecas::DaeSolution;
using nimblecas::Matrix;
using nimblecas::PowerSeries;
using nimblecas::Rational;
using nimblecas::solve_linear_index1_dae;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Small exact rational helpers; inputs are tiny literals so unwrapping is safe.
auto ri(std::int64_t v) -> Rational { return Rational::from_int(v); }
auto rat(std::int64_t num, std::int64_t den) -> Rational { return *Rational::make(num, den); }

// Build a matrix from explicit rational rows (inputs are hand-written and well-formed).
auto mat(std::vector<std::vector<Rational>> rows) -> Matrix { return *Matrix::from_rows(std::move(rows)); }

// Build a power series of the given order from an explicit coefficient list (zero-padded).
auto ser(std::vector<Rational> coeffs, std::size_t order) -> PowerSeries {
    return *PowerSeries::from_coeffs(std::move(coeffs), order);
}

// Assert a series has exactly the expected coefficients (index i is the coefficient of x^i)
// and the expected order.
auto expect_series(TestContext& t, const PowerSeries& s, const std::vector<Rational>& expected,
                   std::string_view what) -> void {
    t.expect(s.order() == expected.size(), std::format("{}: order = {}", what, expected.size()));
    for (std::size_t i = 0; i < expected.size(); ++i) {
        t.expect(s.coefficient(i) == expected[i],
                 std::format("{}: coefficient[{}] = {}", what, i, expected[i].to_string()));
    }
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.dae")
        .test("scalar_exponential_with_identity_constraint",
              [](TestContext& t) {
                  // x' = x, 0 = x - y, so A=[1], B=[0], C=[1], D=[-1], p=q=0. Then D is
                  // invertible (index 1), y = -D^{-1}(C x) = x, and x' = x with x(0)=1 gives
                  // x = e^t. Hence both x and y are the exp series.
                  const std::size_t order = 7;
                  auto sol = solve_linear_index1_dae(
                      mat({{ri(1)}}), mat({{ri(0)}}), mat({{ri(1)}}), mat({{ri(-1)}}),
                      {ser({ri(0)}, order)}, {ser({ri(0)}, order)}, {ri(1)}, order);
                  t.expect(sol.has_value(), "scalar DAE solves");
                  if (!sol) {
                      return;
                  }
                  t.expect(sol->x.size() == 1 && sol->y.size() == 1, "one x and one y component");
                  if (sol->x.size() != 1 || sol->y.size() != 1) {
                      return;
                  }
                  // e^t = 1 + x + x^2/2 + x^3/6 + x^4/24 + x^5/120 + x^6/720.
                  const std::vector<Rational> exp_series = {ri(1),      ri(1),      rat(1, 2),
                                                            rat(1, 6),  rat(1, 24), rat(1, 120),
                                                            rat(1, 720)};
                  expect_series(t, sol->x[0], exp_series, "x = e^t");
                  expect_series(t, sol->y[0], exp_series, "y = x = e^t");
                  t.expect(sol->x[0].is_equal(sol->y[0]), "y equals x exactly");
              })
        .test("harmonic_two_differential_one_algebraic",
              [](TestContext& t) {
                  // x1' = x2, x2' = -x1 (so A = [[0,1],[-1,0]], B = 0), with the algebraic
                  // constraint 0 = x1 + x2 - y (C = [[1,1]], D = [[-1]], q = 0), so
                  // y = -D^{-1}(C x) = x1 + x2. With x0 = [0,1], x1 = sin(x), x2 = cos(x) and
                  // y = sin(x) + cos(x). Hand-verified to order 8.
                  const std::size_t order = 8;
                  auto zero_p = std::vector<PowerSeries>{ser({ri(0)}, order), ser({ri(0)}, order)};
                  auto zero_q = std::vector<PowerSeries>{ser({ri(0)}, order)};
                  auto sol = solve_linear_index1_dae(
                      mat({{ri(0), ri(1)}, {ri(-1), ri(0)}}),  // A
                      mat({{ri(0)}, {ri(0)}}),                 // B (2 x 1, zero)
                      mat({{ri(1), ri(1)}}),                   // C (1 x 2)
                      mat({{ri(-1)}}),                         // D (1 x 1)
                      zero_p, zero_q, {ri(0), ri(1)}, order);
                  t.expect(sol.has_value(), "harmonic DAE solves");
                  if (!sol) {
                      return;
                  }
                  t.expect(sol->x.size() == 2 && sol->y.size() == 1, "two x, one y");
                  if (sol->x.size() != 2 || sol->y.size() != 1) {
                      return;
                  }
                  // sin(x) = x - x^3/6 + x^5/120 - x^7/5040.
                  expect_series(t, sol->x[0],
                                {ri(0), ri(1), ri(0), rat(-1, 6), ri(0), rat(1, 120), ri(0),
                                 rat(-1, 5040)},
                                "x1 = sin");
                  // cos(x) = 1 - x^2/2 + x^4/24 - x^6/720.
                  expect_series(t, sol->x[1],
                                {ri(1), ri(0), rat(-1, 2), ri(0), rat(1, 24), ri(0), rat(-1, 720),
                                 ri(0)},
                                "x2 = cos");
                  // y = sin + cos = 1 + x - x^2/2 - x^3/6 + x^4/24 + x^5/120 - x^6/720 - x^7/5040.
                  expect_series(t, sol->y[0],
                                {ri(1), ri(1), rat(-1, 2), rat(-1, 6), rat(1, 24), rat(1, 120),
                                 rat(-1, 720), rat(-1, 5040)},
                                "y = sin + cos");
              })
        .test("polynomial_forcing_p_and_q",
              [](TestContext& t) {
                  // A=[0], B=[1], C=[0], D=[1], with forcings p(t) = 1 and q(t) = t:
                  //     x' = y + 1,     0 = y + t   =>   y = -t.
                  // Then x' = -t + 1 = 1 - t with x(0) = 0, so x = t - t^2/2. The reduction
                  // gives M = A - B D^{-1} C = 0 and r = p - B D^{-1} q = 1 - t, matching.
                  const std::size_t order = 6;
                  auto sol = solve_linear_index1_dae(
                      mat({{ri(0)}}), mat({{ri(1)}}), mat({{ri(0)}}), mat({{ri(1)}}),
                      {ser({ri(1)}, order)},          // p(t) = 1
                      {ser({ri(0), ri(1)}, order)},   // q(t) = t
                      {ri(0)}, order);
                  t.expect(sol.has_value(), "forced DAE solves");
                  if (!sol) {
                      return;
                  }
                  t.expect(sol->x.size() == 1 && sol->y.size() == 1, "one x, one y");
                  if (sol->x.size() != 1 || sol->y.size() != 1) {
                      return;
                  }
                  // x = t - t^2/2.
                  expect_series(t, sol->x[0], {ri(0), ri(1), rat(-1, 2), ri(0), ri(0), ri(0)},
                                "x = t - t^2/2");
                  // y = -t.
                  expect_series(t, sol->y[0], {ri(0), ri(-1), ri(0), ri(0), ri(0), ri(0)},
                                "y = -t");
              })
        .test("shorter_forcing_series_is_zero_padded",
              [](TestContext& t) {
                  // Same problem as above but the forcings are supplied at a LOWER order than
                  // requested; the solver must retruncate (zero-pad) them to the working order
                  // and produce the identical answer. p = 1 (order 1), q = t (order 2).
                  const std::size_t order = 6;
                  auto sol = solve_linear_index1_dae(
                      mat({{ri(0)}}), mat({{ri(1)}}), mat({{ri(0)}}), mat({{ri(1)}}),
                      {ser({ri(1)}, 1)},           // p(t) = 1, order 1
                      {ser({ri(0), ri(1)}, 2)},    // q(t) = t, order 2
                      {ri(0)}, order);
                  t.expect(sol.has_value(), "DAE with short forcings solves");
                  if (!sol) {
                      return;
                  }
                  expect_series(t, sol->x[0], {ri(0), ri(1), rat(-1, 2), ri(0), ri(0), ri(0)},
                                "x = t - t^2/2 (padded forcing)");
                  expect_series(t, sol->y[0], {ri(0), ri(-1), ri(0), ri(0), ri(0), ri(0)},
                                "y = -t (padded forcing)");
              })
        .test("singular_D_is_higher_index_domain_error",
              [](TestContext& t) {
                  // D = [[0]] is not invertible: the constraint 0 = x + 0*y + 0 does not pin
                  // down y, so the system has index >= 2 and is rejected as domain_error.
                  const std::size_t order = 6;
                  auto sol = solve_linear_index1_dae(
                      mat({{ri(1)}}), mat({{ri(0)}}), mat({{ri(1)}}), mat({{ri(0)}}),
                      {ser({ri(0)}, order)}, {ser({ri(0)}, order)}, {ri(0)}, order);
                  t.expect(!sol.has_value(), "singular D is rejected");
                  t.expect(sol.error() == nimblecas::MathError::domain_error,
                           "singular D yields domain_error (higher index)");
              })
        .test("dimension_mismatch_is_domain_error",
              [](TestContext& t) {
                  // A is 1 x 1 (nd = 1) and D is 1 x 1 (na = 1), so B must be 1 x 1; supplying
                  // a 2 x 1 B is a shape violation.
                  const std::size_t order = 6;
                  auto sol = solve_linear_index1_dae(
                      mat({{ri(1)}}),                  // A: 1 x 1
                      mat({{ri(0)}, {ri(0)}}),         // B: 2 x 1 (wrong; expected 1 x 1)
                      mat({{ri(1)}}),                  // C: 1 x 1
                      mat({{ri(1)}}),                  // D: 1 x 1
                      {ser({ri(0)}, order)}, {ser({ri(0)}, order)}, {ri(0)}, order);
                  t.expect(!sol.has_value(), "shape mismatch is rejected");
                  t.expect(sol.error() == nimblecas::MathError::domain_error,
                           "shape mismatch yields domain_error");
              })
        .test("order_zero_is_domain_error",
              [](TestContext& t) {
                  // order 0 has no truncation ring and is rejected.
                  auto sol = solve_linear_index1_dae(
                      mat({{ri(1)}}), mat({{ri(0)}}), mat({{ri(1)}}), mat({{ri(-1)}}),
                      {ser({ri(0)}, 1)}, {ser({ri(0)}, 1)}, {ri(1)}, 0);
                  t.expect(!sol.has_value(), "order 0 is rejected");
                  t.expect(sol.error() == nimblecas::MathError::domain_error,
                           "order 0 yields domain_error");
              })
        .run();
}
