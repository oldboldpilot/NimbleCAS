// Feature/integration tests: the differential-equations track (ode, dde, dae, perturbation, sde).
// @author Olumuyiwa Oluwasanmi
//
// These are cross-module workflow and mathematical-identity tests, NOT per-module unit tests.
// Four of the five modules here are EXACT over Q (the truncated power-series ring Q[[x]]/(x^N)
// of nimblecas.powerseries): ode, dde, dae and perturbation all return exact rational Taylor
// coefficients, so every assertion on them is an EXACT equality against a hand-computed
// closed-form series — no tolerances. We check known-solution correctness (y'=y -> exp with
// coefficient k == 1/k!; y''=-y -> sin/cos; coupled linear systems -> cosh/sinh and the
// rotation cos/sin; the method-of-steps DDE against its piecewise polynomials with continuity
// at every knot; the index-1 and a genuinely index-2 nilpotent DAE against their exact
// trajectories with the differentiation index and consistency classification verified; and the
// ADM == HPM == HAM(hbar=-1) equivalences with the Adomian series matched term by term) AND
// cross-method consistency (the ode exp series == the perturbation exp series == the raw
// powerseries exp; the DAE's underlying-ODE reduction == a direct ode solve on the equivalent
// system). Only nimblecas.sde is numerical (IEEE-754, Itô/Stratonovich Monte Carlo): its value
// checks use generous, never-flaky statistical tolerances on a fixed seed, while its EXACT
// guarantees — bit-for-bit reproducibility, (seed, path) purity / partition independence, and
// the legacy-bool driver == the Scheme driver — are checked with hard equality. Everything is
// seeded and deterministic; nothing here reads a clock or global state.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.powerseries;
import nimblecas.ode;
import nimblecas.dde;
import nimblecas.dae;
import nimblecas.perturbation;
import nimblecas.sde;
import nimblecas.rng;
import nimblecas.testing;

using namespace nimblecas;                 // core / ratpoly / matrix / powerseries / de modules
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

using SeriesVec = std::vector<PowerSeries>;

// --- small exact-arithmetic helpers -----------------------------------------

// The reduced fraction n/d (n/d known valid: non-zero d, no INT64_MIN edge here).
[[nodiscard]] auto req(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

// k! as an int64 (k small: used only up to 8! = 40320 in these tests, far below overflow).
[[nodiscard]] auto fact(std::int64_t k) -> std::int64_t {
    std::int64_t f = 1;
    for (std::int64_t i = 2; i <= k; ++i) {
        f *= i;
    }
    return f;
}

// b^e as an int64 (small exponents only).
[[nodiscard]] auto ipow(std::int64_t b, std::int64_t e) -> std::int64_t {
    std::int64_t r = 1;
    for (std::int64_t i = 0; i < e; ++i) {
        r *= b;
    }
    return r;
}

// (-1)^m.
[[nodiscard]] auto sgn(std::int64_t m) -> std::int64_t { return (m % 2 == 0) ? 1 : -1; }

// A truncated power series over Q from integer coefficients (coeffs[i] multiplies x^i).
[[nodiscard]] auto psi(std::vector<std::int64_t> cs, std::size_t order) -> PowerSeries {
    std::vector<Rational> r;
    r.reserve(cs.size());
    for (const std::int64_t v : cs) {
        r.push_back(Rational::from_int(v));
    }
    return PowerSeries::from_coeffs(std::move(r), order).value();
}

// Exact coefficient-for-coefficient match of a series against an explicit expected coefficient
// list (which must be exactly the series' order long).
[[nodiscard]] auto series_matches(const PowerSeries& s, const std::vector<Rational>& want) -> bool {
    if (s.order() != want.size()) {
        return false;
    }
    for (std::size_t k = 0; k < want.size(); ++k) {
        if (!(s.coefficient(k) == want[k])) {
            return false;
        }
    }
    return true;
}

// The exp(x) Maclaurin coefficients 1/k! to `order` terms.
[[nodiscard]] auto exp_want(std::size_t order) -> std::vector<Rational> {
    std::vector<Rational> w;
    w.reserve(order);
    for (std::size_t k = 0; k < order; ++k) {
        w.push_back(req(1, fact(static_cast<std::int64_t>(k))));
    }
    return w;
}

}  // namespace

auto main() -> int {
    return TestSuite("feature.diffeq")
        // ===================================================================
        // ODE: exact power-series solutions of known IVPs (over Q).
        // ===================================================================
        .test("ode_y_prime_equals_y_is_exp_coefficient_is_reciprocal_factorial",
              [](TestContext& t) {
                  // y' = y, y(0) = 1 as a first-order system: f(u) = u. Component 0 must be the
                  // exp series, coefficient k == 1/k! EXACTLY over Q.
                  constexpr std::size_t order = 9;
                  SystemOperator id = [](const SeriesVec& u) -> Result<SeriesVec> {
                      return SeriesVec{u[0]};
                  };
                  auto sol = solve_first_order_system(id, {Rational::from_int(1)}, order);
                  t.expect(sol.has_value(), "solve_first_order_system(y'=y) succeeds");
                  if (sol) {
                      t.expect(sol->size() == 1, "one solution component");
                      t.expect(series_matches(sol->front(), exp_want(order)),
                               "y'=y solution has coefficient k == 1/k! exactly (exp series)");
                  }
              })
        .test("ode_second_order_harmonic_gives_sin_and_cos",
              [](TestContext& t) {
                  // y'' = -y via companion reduction. [y(0), y'(0)] = [0, 1] -> sin;
                  // [1, 0] -> cos. Exact rational Taylor coefficients.
                  constexpr std::size_t order = 9;
                  HigherOrderOperator f = [](const SeriesVec& y) -> Result<PowerSeries> {
                      return y[0].scale(Rational::from_int(-1));  // u'' = -u
                  };
                  auto s = solve_higher_order(f, {Rational::from_int(0), Rational::from_int(1)},
                                              order);
                  auto c = solve_higher_order(f, {Rational::from_int(1), Rational::from_int(0)},
                                              order);
                  t.expect(s.has_value() && c.has_value(), "sin and cos IVPs solve");

                  std::vector<Rational> sin_want(order, Rational::from_int(0));
                  std::vector<Rational> cos_want(order, Rational::from_int(0));
                  for (std::size_t k = 0; k < order; ++k) {
                      const std::int64_t kk = static_cast<std::int64_t>(k);
                      if (k % 2 == 1) {
                          sin_want[k] = req(sgn((kk - 1) / 2), fact(kk));  // (-1)^((k-1)/2)/k!
                      } else {
                          cos_want[k] = req(sgn(kk / 2), fact(kk));  // (-1)^(k/2)/k!
                      }
                  }
                  if (s) {
                      t.expect(series_matches(*s, sin_want),
                               "y''=-y, [0,1] -> sin series exactly over Q");
                  }
                  if (c) {
                      t.expect(series_matches(*c, cos_want),
                               "y''=-y, [1,0] -> cos series exactly over Q");
                  }
              })
        .test("ode_linear_systems_rotation_and_hyperbolic_closed_forms",
              [](TestContext& t) {
                  constexpr std::size_t order = 9;
                  // Rotation u1' = -u2, u2' = u1, u(0) = (1, 0): u1 -> cos, u2 -> sin.
                  // (Note u1'=u2, u2'=-u1 would instead give u2 -> -sin; this standard
                  // orientation is the one matching the cos/sin expectations below.)
                  SystemOperator rot = [](const SeriesVec& u) -> Result<SeriesVec> {
                      auto neg = u[1].scale(Rational::from_int(-1));  // -u2
                      if (!neg) {
                          return make_error<SeriesVec>(neg.error());
                      }
                      return SeriesVec{*neg, u[0]};  // u1' = -u2, u2' = u1
                  };
                  auto rsol = solve_first_order_system(
                      rot, {Rational::from_int(1), Rational::from_int(0)}, order);
                  // Hyperbolic u1' = u2, u2' = u1, u(0) = (1, 0): u1 -> cosh, u2 -> sinh.
                  SystemOperator hyp = [](const SeriesVec& u) -> Result<SeriesVec> {
                      return SeriesVec{u[1], u[0]};
                  };
                  auto hsol = solve_first_order_system(
                      hyp, {Rational::from_int(1), Rational::from_int(0)}, order);
                  t.expect(rsol.has_value() && hsol.has_value(),
                           "rotation and hyperbolic systems solve");

                  std::vector<Rational> cos_want(order, Rational::from_int(0));
                  std::vector<Rational> sin_want(order, Rational::from_int(0));
                  std::vector<Rational> cosh_want(order, Rational::from_int(0));
                  std::vector<Rational> sinh_want(order, Rational::from_int(0));
                  for (std::size_t k = 0; k < order; ++k) {
                      const std::int64_t kk = static_cast<std::int64_t>(k);
                      if (k % 2 == 0) {
                          cos_want[k] = req(sgn(kk / 2), fact(kk));
                          cosh_want[k] = req(1, fact(kk));
                      } else {
                          sin_want[k] = req(sgn((kk - 1) / 2), fact(kk));
                          sinh_want[k] = req(1, fact(kk));
                      }
                  }
                  if (rsol) {
                      t.expect(series_matches((*rsol)[0], cos_want),
                               "rotation u1 == cos exactly over Q");
                      t.expect(series_matches((*rsol)[1], sin_want),
                               "rotation u2 == sin exactly over Q");
                  }
                  if (hsol) {
                      t.expect(series_matches((*hsol)[0], cosh_want),
                               "hyperbolic u1 == cosh exactly over Q");
                      t.expect(series_matches((*hsol)[1], sinh_want),
                               "hyperbolic u2 == sinh exactly over Q");
                  }

                  // CROSS-METHOD: the rotation system's components must equal the companion
                  // second-order cos/sin solutions bit-for-bit (two routes to the same series).
                  HigherOrderOperator f = [](const SeriesVec& y) -> Result<PowerSeries> {
                      return y[0].scale(Rational::from_int(-1));
                  };
                  auto sin2 = solve_higher_order(f, {Rational::from_int(0), Rational::from_int(1)},
                                                 order);
                  auto cos2 = solve_higher_order(f, {Rational::from_int(1), Rational::from_int(0)},
                                                 order);
                  if (rsol && sin2 && cos2) {
                      t.expect((*rsol)[0].is_equal(*cos2) && (*rsol)[1].is_equal(*sin2),
                               "first-order rotation == second-order companion cos/sin");
                  }
              })
        .test("ode_exp_series_agrees_with_perturbation_and_raw_powerseries",
              [](TestContext& t) {
                  // CROSS-MODULE: the exp series must be identical whether obtained from the ODE
                  // engine (y'=y), the perturbation ADM solver (u'=u), or the powerseries exp of x.
                  constexpr std::size_t order = 8;
                  SystemOperator id_sys = [](const SeriesVec& u) -> Result<SeriesVec> {
                      return SeriesVec{u[0]};
                  };
                  SeriesOperator id_scalar = [](const PowerSeries& u) -> Result<PowerSeries> {
                      return u;
                  };
                  auto ode_sol = solve_first_order_system(id_sys, {Rational::from_int(1)}, order);
                  auto adm = adm_solve(id_scalar, Rational::from_int(1), order);
                  auto raw = PowerSeries::variable(order).value().exp();
                  t.expect(ode_sol.has_value() && adm.has_value() && raw.has_value(),
                           "all three exp constructions succeed");
                  if (ode_sol && adm && raw) {
                      t.expect(ode_sol->front().is_equal(*adm),
                               "ode y'=y == perturbation ADM u'=u (exp), coefficientwise");
                      t.expect(adm->is_equal(*raw),
                               "perturbation exp == raw powerseries exp(x), coefficientwise");
                  }
              })
        // ===================================================================
        // DDE: method of steps, exact piecewise polynomials and knot continuity.
        // ===================================================================
        .test("dde_y_prime_equals_delayed_y_matches_piecewise_polynomials",
              [](TestContext& t) {
                  // u'(t) = u(t-1), history u == 1 on [-1,0], tau = 1. Hand solution:
                  //   [0,1]: 1 + s ; [1,2]: 2 + s + s^2/2 ; [2,3]: 7/2 + 2s + s^2/2 + s^3/6.
                  constexpr std::size_t order = 6;
                  DdeOperator f =
                      [](const PowerSeries&, const PowerSeries& ud, const PowerSeries&)
                      -> Result<PowerSeries> { return ud; };
                  auto history = PowerSeries::constant(Rational::from_int(1), order).value();
                  auto sol = solve_method_of_steps(f, history, Rational::from_int(1), 3, order);
                  t.expect(sol.has_value(), "method of steps (u'=u(t-1)) succeeds");
                  if (sol) {
                      t.expect(sol->pieces.size() == 3, "three interval pieces");
                      t.expect(series_matches(sol->pieces[0], {req(1, 1), req(1, 1), req(0, 1),
                                                               req(0, 1), req(0, 1), req(0, 1)}),
                               "piece 0 == 1 + s exactly");
                      t.expect(series_matches(sol->pieces[1], {req(2, 1), req(1, 1), req(1, 2),
                                                               req(0, 1), req(0, 1), req(0, 1)}),
                               "piece 1 == 2 + s + s^2/2 exactly");
                      t.expect(series_matches(sol->pieces[2], {req(7, 2), req(2, 1), req(1, 2),
                                                               req(1, 6), req(0, 1), req(0, 1)}),
                               "piece 2 == 7/2 + 2s + s^2/2 + s^3/6 exactly");

                      // Continuity at the interior knots: previous piece at s = tau equals next
                      // piece at s = 0.
                      auto end0 = evaluate_series(sol->pieces[0], Rational::from_int(1));
                      auto end1 = evaluate_series(sol->pieces[1], Rational::from_int(1));
                      t.expect(end0.has_value() && *end0 == sol->pieces[1].coefficient(0),
                               "continuity at t=1: piece0(tau) == piece1(0) == 2");
                      t.expect(end1.has_value() && *end1 == sol->pieces[2].coefficient(0),
                               "continuity at t=2: piece1(tau) == piece2(0) == 7/2");

                      // Whole-solution evaluation at rational times (exact over Q).
                      auto v0 = evaluate(*sol, Rational::from_int(0));
                      auto vk = evaluate(*sol, req(3, 2));       // interval 1, s = 1/2
                      auto vend = evaluate(*sol, Rational::from_int(3));  // right endpoint
                      t.expect(v0.has_value() && *v0 == Rational::from_int(1),
                               "evaluate(sol, 0) == 1");
                      t.expect(vk.has_value() && *vk == req(21, 8),
                               "evaluate(sol, 3/2) == 21/8");
                      t.expect(vend.has_value() && *vend == req(37, 6),
                               "evaluate(sol, 3) == 37/6 (endpoint served by last piece)");
                  }
              })
        .test("dde_y_prime_equals_negative_delayed_y_matches_hand_solution",
              [](TestContext& t) {
                  // u'(t) = -u(t-1), history 1, tau = 1: [0,1]: 1 - s ; [1,2]: -s + s^2/2.
                  constexpr std::size_t order = 5;
                  DdeOperator f =
                      [](const PowerSeries&, const PowerSeries& ud, const PowerSeries&)
                      -> Result<PowerSeries> { return ud.scale(Rational::from_int(-1)); };
                  auto history = PowerSeries::constant(Rational::from_int(1), order).value();
                  auto sol = solve_method_of_steps(f, history, Rational::from_int(1), 2, order);
                  t.expect(sol.has_value(), "method of steps (u'=-u(t-1)) succeeds");
                  if (sol) {
                      t.expect(series_matches(sol->pieces[0],
                                              {req(1, 1), req(-1, 1), req(0, 1), req(0, 1),
                                               req(0, 1)}),
                               "piece 0 == 1 - s exactly");
                      t.expect(series_matches(sol->pieces[1],
                                              {req(0, 1), req(-1, 1), req(1, 2), req(0, 1),
                                               req(0, 1)}),
                               "piece 1 == -s + s^2/2 exactly");
                      auto knot = evaluate(*sol, Rational::from_int(1));
                      t.expect(knot.has_value() && *knot == Rational::from_int(0),
                               "continuity at t=1: value == 0");
                  }
              })
        // ===================================================================
        // DAE: index-1 semi-explicit, index-2 nilpotent, index, consistency.
        // ===================================================================
        .test("dae_index1_semiexplicit_exact_solution_and_ode_reduction",
              [](TestContext& t) {
                  // x' = x + y, 0 = x - y, x(0) = 1. D = [-1] invertible (index 1). Then
                  // y = x and the reduced ODE is x' = 2x, so x = y = sum 2^k/k! x^k exactly.
                  constexpr std::size_t order = 8;
                  const Matrix A = Matrix::from_rows({{Rational::from_int(1)}}).value();
                  const Matrix B = Matrix::from_rows({{Rational::from_int(1)}}).value();
                  const Matrix C = Matrix::from_rows({{Rational::from_int(1)}}).value();
                  const Matrix D = Matrix::from_rows({{Rational::from_int(-1)}}).value();
                  const SeriesVec p{PowerSeries::zero(order).value()};
                  const SeriesVec q{PowerSeries::zero(order).value()};
                  auto sol = solve_linear_index1_dae(A, B, C, D, p, q, {Rational::from_int(1)},
                                                     order);
                  t.expect(sol.has_value(), "solve_linear_index1_dae succeeds (D invertible)");

                  std::vector<Rational> e2_want;  // e^{2t}: 2^k/k!
                  e2_want.reserve(order);
                  for (std::size_t k = 0; k < order; ++k) {
                      const std::int64_t kk = static_cast<std::int64_t>(k);
                      e2_want.push_back(req(ipow(2, kk), fact(kk)));
                  }
                  if (sol) {
                      t.expect(sol->x.size() == 1 && sol->y.size() == 1,
                               "one differential and one algebraic component");
                      t.expect(series_matches(sol->x[0], e2_want),
                               "x == e^{2t} series (coefficient k == 2^k/k!) exactly");
                      t.expect(sol->y[0].is_equal(sol->x[0]),
                               "algebraic y == x exactly (consistent y = -D^{-1}(Cx+q))");

                      // CROSS: the DAE's underlying reduction x' = 2x must match a DIRECT ode
                      // solve on that equivalent scalar system.
                      SystemOperator twice = [](const SeriesVec& u) -> Result<SeriesVec> {
                          auto d = u[0].scale(Rational::from_int(2));
                          if (!d) {
                              return make_error<SeriesVec>(d.error());
                          }
                          return SeriesVec{*d};
                      };
                      auto direct = solve_first_order_system(twice, {Rational::from_int(1)}, order);
                      t.expect(direct.has_value() && direct->front().is_equal(sol->x[0]),
                               "DAE reduction == direct ode solve of x' = 2x");
                  }

                  // The same problem via the higher-index-capable semi-explicit driver must agree
                  // and report differentiation index 1 with a consistent guess y0 = 1.
                  auto se = solve_semiexplicit_dae(A, B, C, D, p, q, {Rational::from_int(1)},
                                                   {Rational::from_int(1)}, order,
                                                   ConsistencyPolicy::project);
                  t.expect(se.has_value(), "solve_semiexplicit_dae succeeds");
                  if (se && sol) {
                      t.expect(se->index == 1, "embedded semi-explicit pencil has index 1");
                      t.expect(se->consistent, "supplied y0 = 1 is consistent");
                      t.expect(se->x[0].is_equal(sol->x[0]) && se->y[0].is_equal(sol->y[0]),
                               "semi-explicit driver == index-1 driver on x and y");
                  }
              })
        .test("dae_index2_nilpotent_exact_solution_and_reported_index",
              [](TestContext& t) {
                  // Nilpotent index-2 pencil E x' = A x + f with E = [[0,1],[0,0]], A = I,
                  // f = [0, -t]. Then x2 = -f2 = t, x1 = -f1 - f2' = 1. Fully determined
                  // (no free initial data); differentiation index 2.
                  constexpr std::size_t order = 5;
                  const Matrix E = Matrix::from_rows({{Rational::from_int(0), Rational::from_int(1)},
                                                      {Rational::from_int(0), Rational::from_int(0)}})
                                       .value();
                  const Matrix Amat =
                      Matrix::from_rows({{Rational::from_int(1), Rational::from_int(0)},
                                         {Rational::from_int(0), Rational::from_int(1)}})
                          .value();
                  const SeriesVec f{PowerSeries::zero(order).value(), psi({0, -1}, order)};

                  auto idx = linear_dae_index(E, Amat);
                  t.expect(idx.has_value() && *idx == 2,
                           "differentiation index of the nilpotent pencil == 2");

                  // Consistent initial vector: x1(0) = 1, x2(0) = 0.
                  const std::vector<Rational> x0_ok{Rational::from_int(1), Rational::from_int(0)};
                  auto cons = linear_dae_is_consistent(E, Amat, f, x0_ok, order);
                  t.expect(cons.has_value() && *cons == true,
                           "x0 = (1,0) is classified consistent");

                  auto sol = solve_linear_dae(E, Amat, f, x0_ok, order,
                                              ConsistencyPolicy::require);
                  t.expect(sol.has_value(), "solve_linear_dae succeeds on consistent x0");
                  if (sol) {
                      t.expect(sol->index == 2, "reported index == 2");
                      t.expect(sol->consistent, "reported consistent == true");
                      // x1 == 1 (constant), x2 == t.
                      std::vector<Rational> x1_want(order, Rational::from_int(0));
                      x1_want[0] = Rational::from_int(1);
                      std::vector<Rational> x2_want(order, Rational::from_int(0));
                      x2_want[1] = Rational::from_int(1);
                      t.expect(series_matches(sol->x[0], x1_want), "x1 == 1 exactly");
                      t.expect(series_matches(sol->x[1], x2_want), "x2 == t exactly");
                  }
              })
        .test("dae_consistency_classification_and_projection",
              [](TestContext& t) {
                  // Same nilpotent index-2 pencil. The constraint manifold is the single point
                  // (1, 0): x0 = (2, 5) is inconsistent. `require` must reject it; `project`
                  // must snap it back to (1,0) and recover the exact trajectory, flagging that
                  // the ORIGINAL guess was inconsistent.
                  constexpr std::size_t order = 5;
                  const Matrix E = Matrix::from_rows({{Rational::from_int(0), Rational::from_int(1)},
                                                      {Rational::from_int(0), Rational::from_int(0)}})
                                       .value();
                  const Matrix Amat =
                      Matrix::from_rows({{Rational::from_int(1), Rational::from_int(0)},
                                         {Rational::from_int(0), Rational::from_int(1)}})
                          .value();
                  const SeriesVec f{PowerSeries::zero(order).value(), psi({0, -1}, order)};
                  const std::vector<Rational> x0_bad{Rational::from_int(2), Rational::from_int(5)};

                  auto cons = linear_dae_is_consistent(E, Amat, f, x0_bad, order);
                  t.expect(cons.has_value() && *cons == false,
                           "x0 = (2,5) is classified inconsistent");

                  auto rejected = solve_linear_dae(E, Amat, f, x0_bad, order,
                                                   ConsistencyPolicy::require);
                  t.expect(!rejected.has_value() &&
                               rejected.error() == MathError::domain_error,
                           "ConsistencyPolicy::require rejects the inconsistent x0 (domain_error)");

                  auto proj_x0 = project_to_consistent(E, Amat, f, x0_bad, order);
                  t.expect(proj_x0.has_value() && proj_x0->size() == 2 &&
                               (*proj_x0)[0] == Rational::from_int(1) &&
                               (*proj_x0)[1] == Rational::from_int(0),
                           "projection snaps (2,5) onto the single consistent point (1,0)");

                  auto projected = solve_linear_dae(E, Amat, f, x0_bad, order,
                                                    ConsistencyPolicy::project);
                  t.expect(projected.has_value(), "ConsistencyPolicy::project solves");
                  if (projected) {
                      t.expect(!projected->consistent,
                               "reported consistent == false (original guess off-manifold)");
                      std::vector<Rational> x1_want(order, Rational::from_int(0));
                      x1_want[0] = Rational::from_int(1);
                      std::vector<Rational> x2_want(order, Rational::from_int(0));
                      x2_want[1] = Rational::from_int(1);
                      t.expect(series_matches(projected->x[0], x1_want) &&
                                   series_matches(projected->x[1], x2_want),
                               "projected solution == exact trajectory (x1=1, x2=t)");
                  }
              })
        .test("dae_index_of_ode_and_plain_index1_pencils",
              [](TestContext& t) {
                  // Sanity anchors for the index computation: an invertible E is index 0; a
                  // semi-explicit E = diag(1,0) with A = I is index 1.
                  const Matrix I2 =
                      Matrix::from_rows({{Rational::from_int(1), Rational::from_int(0)},
                                         {Rational::from_int(0), Rational::from_int(1)}})
                          .value();
                  const Matrix Adiag =
                      Matrix::from_rows({{Rational::from_int(-1), Rational::from_int(0)},
                                         {Rational::from_int(0), Rational::from_int(-2)}})
                          .value();
                  auto i0 = linear_dae_index(I2, Adiag);
                  t.expect(i0.has_value() && *i0 == 0, "invertible E gives index 0 (already ODE)");

                  const Matrix Esing =
                      Matrix::from_rows({{Rational::from_int(1), Rational::from_int(0)},
                                         {Rational::from_int(0), Rational::from_int(0)}})
                          .value();
                  auto i1 = linear_dae_index(Esing, I2);
                  t.expect(i1.has_value() && *i1 == 1,
                           "E = diag(1,0), A = I is a plain index-1 pencil");
              })
        // ===================================================================
        // Perturbation: ADM/HPM/HAM equivalence and exact Adomian series.
        // ===================================================================
        .test("perturbation_adm_hpm_ham_agree_on_nonlinear_riccati",
              [](TestContext& t) {
                  // u' = u^2, u(0) = 1. Exact solution 1/(1-t) = 1 + t + t^2 + ... (all-ones).
                  // ADM == HPM identically; HAM at hbar = -1 recovers them.
                  constexpr std::size_t order = 8;
                  SeriesOperator sq = [](const PowerSeries& u) -> Result<PowerSeries> {
                      return u.multiply(u);
                  };
                  auto adm = adm_solve(sq, Rational::from_int(1), order);
                  auto hpm = hpm_solve(sq, Rational::from_int(1), order);
                  auto ham = ham_solve(sq, Rational::from_int(1), Rational::from_int(-1), order);
                  t.expect(adm.has_value() && hpm.has_value() && ham.has_value(),
                           "ADM, HPM, HAM all solve u'=u^2");
                  if (adm) {
                      std::vector<Rational> ones(order, Rational::from_int(1));
                      t.expect(series_matches(*adm, ones),
                               "ADM u'=u^2 == 1/(1-t) (every coefficient == 1) exactly");
                  }
                  if (adm && hpm) {
                      t.expect(adm->is_equal(*hpm), "ADM == HPM coefficientwise (same recursion)");
                  }
                  if (adm && ham) {
                      t.expect(ham->is_equal(*adm), "HAM(hbar=-1) recovers ADM exactly");
                  }
              })
        .test("perturbation_exp_series_and_ham_equivalence",
              [](TestContext& t) {
                  // u' = u, u(0) = 1 -> exp. ADM matches 1/k! and HAM(hbar=-1) matches ADM.
                  constexpr std::size_t order = 8;
                  SeriesOperator id = [](const PowerSeries& u) -> Result<PowerSeries> {
                      return u;
                  };
                  auto adm = adm_solve(id, Rational::from_int(1), order);
                  auto ham = ham_solve(id, Rational::from_int(1), Rational::from_int(-1), order);
                  t.expect(adm.has_value() && ham.has_value(), "ADM and HAM solve u'=u");
                  if (adm) {
                      t.expect(series_matches(*adm, exp_want(order)),
                               "ADM u'=u == exp series (coefficient k == 1/k!) exactly");
                  }
                  if (adm && ham) {
                      t.expect(ham->is_equal(*adm), "HAM(hbar=-1) == ADM on the exp problem");
                  }
              })
        .test("perturbation_adomian_polynomials_are_degree_graded",
              [](TestContext& t) {
                  // For N(u) = u^2 with homogeneous components u_0 = 1, u_1 = t, u_2 = t^2, the
                  // Adomian polynomials are the graded projections of (u_0+u_1+u_2)^2 =
                  // 1 + 2t + 3t^2 + ...  => A_0 = 1, A_1 = 2t, A_2 = 3t^2. Each A_m must be
                  // HOMOGENEOUS of degree m (only its x^m coefficient nonzero).
                  constexpr std::size_t order = 6;
                  SeriesOperator sq = [](const PowerSeries& u) -> Result<PowerSeries> {
                      return u.multiply(u);
                  };
                  const SeriesVec comps{PowerSeries::constant(Rational::from_int(1), order).value(),
                                        psi({0, 1}, order), psi({0, 0, 1}, order)};
                  auto polys = adomian_polynomials(sq, comps);
                  t.expect(polys.has_value() && polys->size() == 3,
                           "three Adomian polynomials A_0..A_2");
                  if (polys) {
                      const std::array<std::int64_t, 3> lead{1, 2, 3};  // A_m coefficient at x^m
                      bool graded = true;
                      for (std::size_t m = 0; m < 3; ++m) {
                          const PowerSeries& a = (*polys)[m];
                          for (std::size_t j = 0; j < order; ++j) {
                              const Rational want = (j == m) ? Rational::from_int(lead[m])
                                                             : Rational::from_int(0);
                              if (!(a.coefficient(j) == want)) {
                                  graded = false;
                              }
                          }
                      }
                      t.expect(graded,
                               "A_m is homogeneous of degree m with A_0=1, A_1=2t, A_2=3t^2");
                  }
              })
        // ===================================================================
        // SDE: seeded Monte Carlo on geometric Brownian motion; exact contracts.
        // ===================================================================
        .test("sde_gbm_ito_schemes_recover_x0_exp_muT",
              [](TestContext& t) {
                  // GBM dX = mu X dt + sigma X dW, x0 = 1, mu = 0, sigma = 1/2, T = 1. The Itô
                  // schemes (Euler-Maruyama, Milstein, SRK, Tamed) have E[X_T] = x0 e^{muT} = 1.
                  const double mu = 0.0;
                  const double sigma = 0.5;
                  const double x0 = 1.0;
                  const double T = 1.0;
                  const std::uint64_t steps = 50;
                  const std::uint64_t paths = 20000;
                  const std::uint64_t seed = 20260703;
                  const double ito_mean = x0 * std::exp(mu * T);  // == 1
                  auto a = [mu](double x) { return mu * x; };
                  auto b = [sigma](double x) { return sigma * x; };
                  auto bp = [sigma](double) { return sigma; };

                  auto em = terminal_moments_scheme(a, b, {}, x0, T, steps, paths, seed,
                                                    Scheme::euler_maruyama);
                  auto mi = terminal_moments_scheme(a, b, bp, x0, T, steps, paths, seed,
                                                    Scheme::milstein);
                  auto sr = terminal_moments_scheme(a, b, {}, x0, T, steps, paths, seed,
                                                    Scheme::srk);
                  auto ta = terminal_moments_scheme(a, b, {}, x0, T, steps, paths, seed,
                                                    Scheme::tamed_euler);
                  t.expect(em.has_value() && mi.has_value() && sr.has_value() && ta.has_value(),
                           "all four Itô-scheme ensembles run");
                  const double tol = 0.02;
                  if (em) {
                      t.expect(std::abs(em->first - ito_mean) < tol,
                               std::format("Euler-Maruyama E[X_T] ~ e^(muT) (got {})", em->first));
                  }
                  if (mi) {
                      t.expect(std::abs(mi->first - ito_mean) < tol,
                               std::format("Milstein E[X_T] ~ e^(muT) (got {})", mi->first));
                  }
                  if (sr) {
                      t.expect(std::abs(sr->first - ito_mean) < tol,
                               std::format("SRK E[X_T] ~ e^(muT) (got {})", sr->first));
                  }
                  if (ta) {
                      t.expect(std::abs(ta->first - ito_mean) < tol,
                               std::format("Tamed Euler E[X_T] ~ e^(muT) (got {})", ta->first));
                  }
              })
        .test("sde_gbm_stratonovich_heun_recovers_the_extra_half_sigma2_drift",
              [](TestContext& t) {
                  // Stochastic Heun re-uses dW in both stages => Stratonovich limit: for GBM its
                  // mean is x0 e^{(mu + 1/2 sigma^2)T}, distinctly ABOVE the Itô schemes' e^{muT}.
                  const double mu = 0.0;
                  const double sigma = 0.5;
                  const double x0 = 1.0;
                  const double T = 1.0;
                  const std::uint64_t steps = 50;
                  const std::uint64_t paths = 20000;
                  const std::uint64_t seed = 20260703;
                  const double ito_mean = x0 * std::exp(mu * T);                        // 1
                  const double strat_mean = x0 * std::exp((mu + 0.5 * sigma * sigma) * T);  // e^0.125
                  auto a = [mu](double x) { return mu * x; };
                  auto b = [sigma](double x) { return sigma * x; };

                  auto he = terminal_moments_scheme(a, b, {}, x0, T, steps, paths, seed,
                                                    Scheme::stochastic_heun);
                  auto em = terminal_moments_scheme(a, b, {}, x0, T, steps, paths, seed,
                                                    Scheme::euler_maruyama);
                  t.expect(he.has_value() && em.has_value(), "Heun and Euler ensembles run");
                  if (he) {
                      t.expect(std::abs(he->first - strat_mean) < 0.03,
                               std::format("Heun E[X_T] ~ e^((mu+sigma^2/2)T) (got {})", he->first));
                  }
                  if (he && em) {
                      // The Stratonovich drift correction must be clearly visible: Heun's mean
                      // sits well above the Itô mean (the two conventions genuinely differ).
                      t.expect(he->first - em->first > 0.05,
                               std::format("Heun mean exceeds Euler mean by the Stratonovich drift "
                                           "(Heun {}, Euler {})", he->first, em->first));
                  }
              })
        .test("sde_paths_are_bit_reproducible_and_ito_schemes_coincide_without_noise",
              [](TestContext& t) {
                  // Determinism: equal (seed, path) reproduce a bit-identical trajectory.
                  auto a = [](double x) { return -x; };
                  auto b = [](double x) { return 0.3 * x; };
                  auto bp = [](double) { return 0.3; };
                  auto p1 = euler_maruyama(a, b, 1.0, 1.0, 64, 4242);
                  auto p2 = euler_maruyama(a, b, 1.0, 1.0, 64, 4242);
                  t.expect(p1.has_value() && p2.has_value(), "two seeded Euler paths run");
                  if (p1 && p2) {
                      t.expect(p1->values == p2->values && p1->times == p2->times,
                               "equal seed+path -> bit-identical trajectory");
                  }

                  // With b == 0 every Itô scheme reduces to Euler-Maruyama bit-for-bit (the
                  // diffusion corrections vanish identically); they share one Brownian stream.
                  auto zero_b = [](double) { return 0.0; };
                  auto e0 = euler_maruyama(a, zero_b, 1.0, 1.0, 64, 99);
                  auto s0 = srk(a, zero_b, 1.0, 1.0, 64, 99);
                  auto m0 = milstein(a, zero_b, zero_b, 1.0, 1.0, 64, 99);
                  t.expect(e0.has_value() && s0.has_value() && m0.has_value(),
                           "Euler, SRK, Milstein all run with b == 0");
                  if (e0 && s0 && m0) {
                      t.expect(e0->values == s0->values,
                               "SRK == Euler-Maruyama bit-for-bit when b == 0");
                      t.expect(e0->values == m0->values,
                               "Milstein == Euler-Maruyama bit-for-bit when b == 0");
                  }
              })
        .test("sde_ensemble_is_partition_independent_and_seed_path_pure",
              [](TestContext& t) {
                  // Each terminal value is a pure function of (seed, path index p): path p is
                  // seeded splitmix64(seed ^ p). So the ensemble vector must equal, element for
                  // element, standalone single-path runs at those derived seeds — regardless of
                  // how 0..paths-1 would be split across workers.
                  auto a = [](double x) { return 0.1 * x; };
                  auto b = [](double x) { return 0.2 * x; };
                  const double x0 = 1.0;
                  const double T = 1.0;
                  const std::uint64_t steps = 32;
                  const std::uint64_t paths = 8;
                  const std::uint64_t seed = 13572468;

                  auto ens = simulate_terminal_scheme(a, b, {}, x0, T, steps, paths, seed,
                                                      Scheme::euler_maruyama);
                  t.expect(ens.has_value() && ens->size() == paths, "ensemble of `paths` terminals");
                  if (ens) {
                      bool all_match = true;
                      for (std::uint64_t p = 0; p < paths; ++p) {
                          const std::uint64_t path_seed = splitmix64(seed ^ p);
                          auto solo = euler_maruyama(a, b, x0, T, steps, path_seed);
                          if (!solo || solo->values.back() != (*ens)[p]) {
                              all_match = false;
                          }
                      }
                      t.expect(all_match,
                               "ensemble[p] == standalone run at splitmix64(seed ^ p), bit-for-bit");
                  }
              })
        .test("sde_legacy_bool_driver_equals_scheme_driver_bit_for_bit",
              [](TestContext& t) {
                  // The legacy use_milstein bool driver must equal the Scheme-enum driver exactly
                  // for both euler_maruyama (false) and milstein (true) — same seeding, same
                  // Brownian stream, same reduction — not merely statistically.
                  auto a = [](double x) { return 0.15 * x; };
                  auto b = [](double x) { return 0.25 * x; };
                  auto bp = [](double) { return 0.25; };
                  const double x0 = 1.0;
                  const double T = 1.0;
                  const std::uint64_t steps = 40;
                  const std::uint64_t paths = 500;
                  const std::uint64_t seed = 777001;

                  auto legacy_em = simulate_terminal(a, b, {}, x0, T, steps, paths, seed, false);
                  auto scheme_em = simulate_terminal_scheme(a, b, {}, x0, T, steps, paths, seed,
                                                            Scheme::euler_maruyama);
                  auto legacy_mi = simulate_terminal(a, b, bp, x0, T, steps, paths, seed, true);
                  auto scheme_mi = simulate_terminal_scheme(a, b, bp, x0, T, steps, paths, seed,
                                                            Scheme::milstein);
                  t.expect(legacy_em.has_value() && scheme_em.has_value() &&
                               legacy_mi.has_value() && scheme_mi.has_value(),
                           "all four driver invocations run");
                  if (legacy_em && scheme_em) {
                      t.expect(*legacy_em == *scheme_em,
                               "legacy(use_milstein=false) == Scheme::euler_maruyama bit-for-bit");
                  }
                  if (legacy_mi && scheme_mi) {
                      t.expect(*legacy_mi == *scheme_mi,
                               "legacy(use_milstein=true) == Scheme::milstein bit-for-bit");
                  }
              })
        .run();
}
