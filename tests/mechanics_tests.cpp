// Tests for nimblecas.mechanics: Legendre transform, Hamilton's canonical equations,
// Poisson brackets, conserved quantities, phase space, and the 1-DOF action integrand.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.mechanics;
import nimblecas.testing;

using nimblecas::action_integral;
using nimblecas::ActionIntegral;
using nimblecas::Coordinates;
using nimblecas::cyclic_coordinates;
using nimblecas::Expr;
using nimblecas::hamilton_equations;
using nimblecas::Hamiltonian;
using nimblecas::HamiltonSystem;
using nimblecas::is_constant_of_motion;
using nimblecas::legendre_transform;
using nimblecas::MathError;
using nimblecas::phase_curve;
using nimblecas::phase_portrait_field;
using nimblecas::poisson_bracket;
using nimblecas::simplify;
using nimblecas::angle_variable;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto sym(std::string n) -> Expr { return Expr::symbol(std::move(n)); }
[[nodiscard]] auto I(std::int64_t n) -> Expr { return Expr::integer(n); }
[[nodiscard]] auto S(const Expr& e) -> Expr { return simplify(e).value(); }  // canonical form
[[nodiscard]] auto half() -> Expr { return Expr::rational(1, 2).value(); }
[[nodiscard]] auto neg(const Expr& a) -> Expr { return Expr::product({I(-1), a}); }
[[nodiscard]] auto sqr(const Expr& a) -> Expr { return Expr::power(a, I(2)); }
[[nodiscard]] auto recip(const Expr& a) -> Expr { return Expr::power(a, I(-1)); }

}  // namespace

auto main() -> int {
    const Expr q = sym("q");
    const Expr qd = sym("qd");
    const Expr p = sym("p");
    const Expr w = sym("w");   // angular frequency omega
    const Expr m = sym("m");   // mass
    const Expr E = sym("E");   // energy

    // One-DOF canonical coordinate system (q, q̇ = qd, p).
    const Coordinates one{.q = {"q"}, .qdot = {"qd"}, .p = {"p"}, .time = "t"};

    return TestSuite("nimblecas.mechanics")
        .test("legendre_harmonic_oscillator",
              [&](TestContext& t) {
                  // L = 1/2 q̇^2 − 1/2 ω^2 q^2.
                  const Expr L = Expr::sum(
                      {Expr::product({half(), sqr(qd)}),
                       neg(Expr::product({half(), sqr(w), sqr(q)}))});
                  auto ham = legendre_transform(L, one).value();
                  // p = ∂L/∂q̇ = q̇.
                  t.expect(ham.momenta[0] == qd, "conjugate momentum p = q̇");
                  // Inverted momentum map q̇(p) = p.
                  t.expect(ham.velocities[0] == p, "velocity q̇ = p");
                  // H = 1/2 p^2 + 1/2 ω^2 q^2.
                  const Expr H_expected = Expr::sum(
                      {Expr::product({half(), sqr(p)}),
                       Expr::product({half(), sqr(w), sqr(q)})});
                  t.expect(ham.H == S(H_expected), "H = 1/2 p^2 + 1/2 ω^2 q^2");
              })
        .test("legendre_free_particle",
              [&](TestContext& t) {
                  // L = 1/2 m q̇^2  ->  p = m q̇,  H = p^2 / (2m).
                  const Expr L = Expr::product({half(), m, sqr(qd)});
                  auto ham = legendre_transform(L, one).value();
                  t.expect(ham.momenta[0] == S(Expr::product({m, qd})), "p = m q̇");
                  t.expect(ham.velocities[0] == S(Expr::product({p, recip(m)})), "q̇ = p/m");
                  const Expr H_expected = Expr::product({half(), sqr(p), recip(m)});
                  t.expect(ham.H == S(H_expected), "H = p^2/(2m)");
              })
        .test("legendre_non_quadratic_not_implemented",
              [&](TestContext& t) {
                  // L = q̇^3 is cubic in the velocity: the momentum map is not linearly
                  // invertible here.
                  const Expr L = Expr::power(qd, I(3));
                  t.expect(legendre_transform(L, one).error() == MathError::not_implemented,
                           "cubic-in-q̇ Lagrangian is not_implemented");
              })
        .test("hamilton_equations_sho",
              [&](TestContext& t) {
                  // H = 1/2 p^2 + 1/2 ω^2 q^2  ->  q̇ = p,  ṗ = −ω^2 q.
                  const Expr H = Expr::sum(
                      {Expr::product({half(), sqr(p)}),
                       Expr::product({half(), sqr(w), sqr(q)})});
                  auto sys = hamilton_equations(H, one).value();
                  t.expect(sys.qdot[0] == p, "q̇ = ∂H/∂p = p");
                  t.expect(sys.pdot[0] == S(neg(Expr::product({sqr(w), q}))),
                           "ṗ = −∂H/∂q = −ω^2 q");
              })
        .test("poisson_brackets_and_energy_conservation",
              [&](TestContext& t) {
                  const Expr H = Expr::sum(
                      {Expr::product({half(), sqr(p)}),
                       Expr::product({half(), sqr(w), sqr(q)})});
                  // Canonical relations {q,p}=1, {q,q}=0, {p,p}=0.
                  t.expect(poisson_bracket(q, p, one).value() == I(1), "{q,p} = 1");
                  t.expect(poisson_bracket(q, q, one).value() == I(0), "{q,q} = 0");
                  t.expect(poisson_bracket(p, p, one).value() == I(0), "{p,p} = 0");
                  // {H,H} = 0 and energy is conserved (df/dt = {H,H} + ∂H/∂t = 0).
                  t.expect(poisson_bracket(H, H, one).value() == I(0), "{H,H} = 0");
                  t.expect(is_constant_of_motion(H, H, one).value(), "energy is conserved");
              })
        .test("cyclic_coordinate_conserved_momentum",
              [&](TestContext& t) {
                  // Free rotor H = 1/2 p_θ^2: θ is cyclic, so p_θ is conserved.
                  const Coordinates rot{
                      .q = {"theta"}, .qdot = {"thetadot"}, .p = {"ptheta"}, .time = "t"};
                  const Expr H = Expr::product({half(), sqr(sym("ptheta"))});
                  auto cyc = cyclic_coordinates(H, rot).value();
                  t.expect(cyc.size() == 1 && cyc[0] == 0, "theta is cyclic");
                  t.expect(is_constant_of_motion(sym("ptheta"), H, rot).value(),
                           "p_θ is a constant of motion");
                  // The coordinate itself is NOT conserved: {θ,H} = p_θ != 0.
                  t.expect(!is_constant_of_motion(sym("theta"), H, rot).value(),
                           "theta itself is not conserved");
              })
        .test("phase_curve_and_field",
              [&](TestContext& t) {
                  const Expr H = Expr::sum(
                      {Expr::product({half(), sqr(p)}),
                       Expr::product({half(), sqr(w), sqr(q)})});
                  // Level set H − E defines the closed phase curve at energy E.
                  t.expect(phase_curve(H, E).value() == S(Expr::sum({H, neg(E)})),
                           "phase curve is H − E");
                  // Vector field (q̇,ṗ) = (p, −ω^2 q).
                  auto field = phase_portrait_field(H, one).value();
                  t.expect(field.qdot[0] == p, "field q̇ = p");
                  t.expect(field.pdot[0] == S(neg(Expr::product({sqr(w), q}))),
                           "field ṗ = −ω^2 q");
              })
        .test("action_integrand_sho",
              [&](TestContext& t) {
                  // H = 1/2 p^2 + 1/2 ω^2 q^2.  Solving H = E gives p = sqrt(2E − ω^2 q^2),
                  // the ∮ p dq integrand whose closed loop integral is J = E/ω (non-elementary
                  // here, so the closed form is left absent).
                  const Expr H = Expr::sum(
                      {Expr::product({half(), sqr(p)}),
                       Expr::product({half(), sqr(w), sqr(q)})});
                  auto act = action_integral(H, one, E).value();
                  const Expr integrand_expected =
                      Expr::apply("sqrt", {Expr::sum({Expr::product({I(2), E}),
                                                      neg(Expr::product({sqr(w), sqr(q)}))})});
                  t.expect(act.integrand == S(integrand_expected),
                           "action integrand p = sqrt(2E − ω^2 q^2)");
                  t.expect(act.coordinate == "q", "integration variable is q");
                  t.expect(!act.closed_form.has_value(),
                           "closed-form action is non-elementary here (honest not_implemented)");
                  // The angle variable follows from a closed-form action; not available here.
                  t.expect(angle_variable(H, one, E).error() == MathError::not_implemented,
                           "angle variable is not_implemented without a closed-form action");
              })
        .test("dimension_mismatch_is_domain_error",
              [&](TestContext& t) {
                  // Inconsistent coordinate lists are a domain error.
                  const Coordinates bad{.q = {"q"}, .qdot = {"qd", "qd2"}, .p = {"p"}};
                  t.expect(hamilton_equations(sym("q"), bad).error() == MathError::domain_error,
                           "mismatched coordinate sizes fail");
                  // Action-angle requires exactly one degree of freedom.
                  const Coordinates two{.q = {"x", "y"},
                                        .qdot = {"xd", "yd"},
                                        .p = {"px", "py"}};
                  t.expect(action_integral(sym("px"), two, E).error() == MathError::domain_error,
                           "action integral requires 1-DOF");
              })
        .run();
}
