// Tests for nimblecas.calcvar: Euler-Lagrange equations, the Beltrami first integral,
// holonomic / non-holonomic (Pfaffian) constraints, Lagrange multipliers, and Pfaffian
// constraint classification.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.calcvar;
import nimblecas.testing;

using nimblecas::beltrami_identity;
using nimblecas::classify_pfaffian;
using nimblecas::constrained_euler_lagrange;
using nimblecas::ConstraintClass;
using nimblecas::Coordinate;
using nimblecas::euler_lagrange;
using nimblecas::Expr;
using nimblecas::free_of;
using nimblecas::lagrange_equations;
using nimblecas::lagrange_multipliers;
using nimblecas::MathError;
using nimblecas::PfaffianConstraint;
using nimblecas::simplify;
using nimblecas::substitute;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto sym(std::string n) -> Expr { return Expr::symbol(std::move(n)); }
[[nodiscard]] auto I(std::int64_t n) -> Expr { return Expr::integer(n); }
[[nodiscard]] auto R(std::int64_t n, std::int64_t d) -> Expr {
    return Expr::rational(n, d).value();
}
[[nodiscard]] auto S(const Expr& e) -> Expr { return simplify(e).value(); }  // canonical form

}  // namespace

auto main() -> int {
    const Expr x = sym("x");
    const Expr y = sym("y");
    const Expr z = sym("z");
    const Expr yp = sym("yp");    // y'
    const Expr ypp = sym("ypp");  // y''

    return TestSuite("nimblecas.calcvar")
        .test("shortest_path_straight_line",
              [&](TestContext& t) {
                  // F = sqrt(1 + y'^2): the geodesic integrand in the plane. Since F is free
                  // of y, dF/dy = 0 and the whole EL equation is proportional to y'', so a
                  // straight line (y'' = 0) is an exact extremal.
                  const Expr F = Expr::power(Expr::sum({I(1), Expr::power(yp, I(2))}), R(1, 2));
                  auto el = euler_lagrange(F, "y", "yp", "ypp", "x").value();
                  t.expect(free_of(el, y), "EL is free of y (F independent of y)");
                  t.expect(S(substitute(el, ypp, I(0))) == I(0),
                           "y'' = 0 (straight line) satisfies the EL equation exactly");
              })
        .test("brachistochrone_beltrami_first_integral",
              [&](TestContext& t) {
                  // Brachistochrone integrand F = sqrt((1 + y'^2)/y): no explicit x, so the
                  // Beltrami identity yields a first-order first integral.
                  const Expr F = Expr::power(
                      Expr::product({Expr::sum({I(1), Expr::power(yp, I(2))}),
                                     Expr::power(y, I(-1))}),
                      R(1, 2));
                  auto b = beltrami_identity(F, "y", "yp", "x").value();
                  t.expect(free_of(b, x), "first integral has no explicit x");
                  t.expect(free_of(b, ypp), "first integral is first order (no y'')");

                  // A first integral of an F with explicit x is not a Beltrami case.
                  const Expr Fx =
                      Expr::product({x, Expr::power(Expr::sum({I(1), Expr::power(yp, I(2))}),
                                                    R(1, 2))});
                  t.expect(beltrami_identity(Fx, "y", "yp", "x").error() == MathError::domain_error,
                           "Beltrami rejects an integrand with explicit x");
              })
        .test("harmonic_oscillator",
              [&](TestContext& t) {
                  // L = 1/2 q'^2 - 1/2 w^2 q^2  ->  EL gives q'' + w^2 q = 0.
                  const Expr q = sym("q");
                  const Expr qd = sym("qd");
                  const Expr qdd = sym("qdd");
                  const Expr w = sym("omega");
                  const Expr L = Expr::sum(
                      {Expr::product({R(1, 2), Expr::power(qd, I(2))}),
                       Expr::product({R(-1, 2), Expr::power(w, I(2)), Expr::power(q, I(2))})});
                  auto eqs = lagrange_equations(
                                 L, "t", {Coordinate{.value = "q", .velocity = "qd",
                                                     .acceleration = "qdd"}})
                                 .value();
                  t.expect(eqs.size() == 1, "one generalized coordinate");
                  t.expect(eqs[0] == S(Expr::sum({qdd, Expr::product({Expr::power(w, I(2)), q})})),
                           "q'' + w^2 q = 0");
              })
        .test("constrained_pendulum_multiplier",
              [&](TestContext& t) {
                  // Cartesian pendulum: L = 1/2 m (vx^2 + vy^2) - m g y, holonomic constraint
                  // g = x^2 + y^2 - l^2 = 0, multiplier lambda.
                  const Expr m = sym("m");
                  const Expr grav = sym("grav");
                  const Expr ell = sym("ell");
                  const Expr lam = sym("lam");
                  const Expr vx = sym("vx");
                  const Expr vy = sym("vy");
                  const Expr ax = sym("ax");
                  const Expr ay = sym("ay");
                  const std::vector<Coordinate> coords = {
                      Coordinate{.value = "x", .velocity = "vx", .acceleration = "ax"},
                      Coordinate{.value = "y", .velocity = "vy", .acceleration = "ay"}};
                  const Expr L = Expr::sum(
                      {Expr::product({R(1, 2), m, Expr::power(vx, I(2))}),
                       Expr::product({R(1, 2), m, Expr::power(vy, I(2))}),
                       Expr::product({I(-1), m, grav, y})});
                  const Expr g = Expr::sum(
                      {Expr::power(x, I(2)), Expr::power(y, I(2)),
                       Expr::product({I(-1), Expr::power(ell, I(2))})});
                  auto sys = constrained_euler_lagrange(L, "t", coords, {g}, {"lam"}).value();
                  t.expect(sys.equations.size() == 2, "one EL equation per coordinate");
                  // EL_x : m x'' - 2 lambda x = 0.
                  t.expect(sys.equations[0] ==
                               S(Expr::sum({Expr::product({m, ax}),
                                            Expr::product({I(-2), lam, x})})),
                           "EL_x = m x'' - 2 lambda x");
                  // EL_y : m y'' + m g - 2 lambda y = 0.
                  t.expect(sys.equations[1] ==
                               S(Expr::sum({Expr::product({m, ay}), Expr::product({m, grav}),
                                            Expr::product({I(-2), lam, y})})),
                           "EL_y = m y'' + m g - 2 lambda y");
                  t.expect(sys.constraints.size() == 1 && sys.constraints[0] == S(g),
                           "constraint g = x^2 + y^2 - l^2 echoed");
              })
        .test("pfaffian_constrained_free_particle",
              [&](TestContext& t) {
                  // Free particle L = 1/2 m (vx^2 + vy^2) with the non-holonomic (as posed)
                  // Pfaffian constraint  vx - vy = 0, i.e. dx - dy = 0, multiplier lambda.
                  const Expr m = sym("m");
                  const Expr lam = sym("lam");
                  const Expr vx = sym("vx");
                  const Expr vy = sym("vy");
                  const Expr ax = sym("ax");
                  const Expr ay = sym("ay");
                  const std::vector<Coordinate> coords = {
                      Coordinate{.value = "x", .velocity = "vx", .acceleration = "ax"},
                      Coordinate{.value = "y", .velocity = "vy", .acceleration = "ay"}};
                  const Expr L = Expr::sum(
                      {Expr::product({R(1, 2), m, Expr::power(vx, I(2))}),
                       Expr::product({R(1, 2), m, Expr::power(vy, I(2))})});
                  const PfaffianConstraint pf{.coeffs = {I(1), I(-1)}, .time_coeff = I(0)};
                  auto sys =
                      constrained_euler_lagrange(L, "t", coords, {pf}, {"lam"}).value();
                  t.expect(sys.equations.size() == 2, "one EL equation per coordinate");
                  // EL_x : m x'' - lambda * 1 = m x'' - lambda.
                  t.expect(sys.equations[0] ==
                               S(Expr::sum({Expr::product({m, ax}), Expr::product({I(-1), lam})})),
                           "EL_x = m x'' - lambda");
                  // EL_y : m y'' - lambda * (-1) = m y'' + lambda.
                  t.expect(sys.equations[1] == S(Expr::sum({Expr::product({m, ay}), lam})),
                           "EL_y = m y'' + lambda");
                  t.expect(sys.constraints.size() == 1 &&
                               sys.constraints[0] ==
                                   S(Expr::sum({vx, Expr::product({I(-1), vy})})),
                           "velocity-form constraint vx - vy = 0");
              })
        .test("finite_lagrange_multipliers",
              [&](TestContext& t) {
                  // Extremize f = x + y subject to g = x^2 + y^2 - 1 = 0.
                  const Expr lam = sym("lam");
                  const Expr f = Expr::sum({x, y});
                  const Expr g = Expr::sum({Expr::power(x, I(2)), Expr::power(y, I(2)), I(-1)});
                  auto sys = lagrange_multipliers(f, {"x", "y"}, {g}, {"lam"}).value();
                  t.expect(sys.stationarity.size() == 2, "one stationarity equation per variable");
                  // df/dx - lambda dg/dx = 1 - 2 lambda x.
                  t.expect(sys.stationarity[0] ==
                               S(Expr::sum({I(1), Expr::product({I(-2), lam, x})})),
                           "1 - 2 lambda x = 0");
                  t.expect(sys.stationarity[1] ==
                               S(Expr::sum({I(1), Expr::product({I(-2), lam, y})})),
                           "1 - 2 lambda y = 0");
                  t.expect(sys.constraints.size() == 1 && sys.constraints[0] == S(g),
                           "constraint x^2 + y^2 - 1 echoed");
              })
        .test("classify_holonomic_vs_nonholonomic",
              [&](TestContext& t) {
                  // dz - y dx = 0  is the textbook non-holonomic (Frobenius-obstructed) form.
                  const std::vector<Expr> nonholo = {Expr::product({I(-1), y}), I(0), I(1)};
                  t.expect(classify_pfaffian(nonholo, {"x", "y", "z"}).value() ==
                               ConstraintClass::non_holonomic,
                           "dz - y dx is non-holonomic");
                  // y dx + x dy = d(x y): exact, hence holonomic.
                  t.expect(classify_pfaffian({y, x}, {"x", "y"}).value() ==
                               ConstraintClass::holonomic,
                           "y dx + x dy is exact (holonomic)");
                  // y z dx + x z dy + x y dz = d(x y z): exact in three variables.
                  const std::vector<Expr> exact3 = {Expr::product({y, z}), Expr::product({x, z}),
                                                    Expr::product({x, y})};
                  t.expect(classify_pfaffian(exact3, {"x", "y", "z"}).value() ==
                               ConstraintClass::holonomic,
                           "d(x y z) is exact (holonomic)");
                  // Dimension mismatch is a domain error.
                  t.expect(classify_pfaffian({y, x}, {"x", "y", "z"}).error() ==
                               MathError::domain_error,
                           "coeff/var count mismatch fails");
              })
        .run();
}
