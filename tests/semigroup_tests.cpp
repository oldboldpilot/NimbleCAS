// Tests for nimblecas.semigroup: operator theory & C0-semigroups (finite-dim realization).
// @author Olumuyiwa Oluwasanmi
//
// Every case is pinned to an exact rational answer: the resolvent and its spectral
// domain_error, the rational spectrum of a triangular matrix, the semigroup property and
// Cauchy problem on a nilpotent generator (where the truncated series is genuinely exact),
// exact dissipativity/Hurwitz verdicts, an exact Sylvester/Lyapunov reconstruction, and
// variation of constants on nilpotent generators with constant and linear polynomial forcing.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.matexp;
import nimblecas.semigroup;
import nimblecas.testing;

using nimblecas::adjoint;
using nimblecas::cauchy_solution;
using nimblecas::is_contraction_generator;
using nimblecas::is_dissipative;
using nimblecas::is_hurwitz;
using nimblecas::lyapunov_solve;
using nimblecas::MathError;
using nimblecas::Matrix;
using nimblecas::operator_norm_1;
using nimblecas::operator_norm_inf;
using nimblecas::pde_semigroup_solution;
using nimblecas::Rational;
using nimblecas::resolvent;
using nimblecas::semigroup;
using nimblecas::spectral_radius;
using nimblecas::spectrum;
using nimblecas::sylvester_solve;
using nimblecas::variation_of_constants;
using nimblecas::verify_semigroup_property;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto ri(std::int64_t v) -> Rational {
    return Rational::from_int(v);
}

// Build a Matrix from integer rows (low-index row first).
[[nodiscard]] auto mat(std::vector<std::vector<std::int64_t>> rows) -> Matrix {
    std::vector<std::vector<Rational>> r;
    r.reserve(rows.size());
    for (const auto& row : rows) {
        std::vector<Rational> rr_row;
        rr_row.reserve(row.size());
        for (const std::int64_t v : row) {
            rr_row.push_back(Rational::from_int(v));
        }
        r.push_back(std::move(rr_row));
    }
    return Matrix::from_rows(std::move(r)).value();
}

// Build an n x 1 column vector from integer entries.
[[nodiscard]] auto col(std::vector<std::int64_t> entries) -> Matrix {
    std::vector<std::vector<std::int64_t>> rows;
    rows.reserve(entries.size());
    for (const std::int64_t v : entries) {
        rows.push_back({v});
    }
    return mat(std::move(rows));
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.semigroup")
        .test("resolvent_exact_off_spectrum",
              [](TestContext& t) {
                  // A = [[1,2],[0,3]] (eigenvalues 1, 3). At lambda = 2, (2I - A) = [[1,-2],[0,-1]]
                  // is invertible and its own inverse, so R(2, A) = [[1,-2],[0,-1]] exactly.
                  const Matrix a = mat({{1, 2}, {0, 3}});
                  auto r = resolvent(a, ri(2));
                  t.expect(r.has_value(), "resolvent off the spectrum succeeds");
                  if (r) {
                      t.expect(r->at(0, 0) == ri(1), "R(2,A)(0,0) = 1");
                      t.expect(r->at(0, 1) == ri(-2), "R(2,A)(0,1) = -2");
                      t.expect(r->at(1, 0) == ri(0), "R(2,A)(1,0) = 0");
                      t.expect(r->at(1, 1) == ri(-1), "R(2,A)(1,1) = -1");
                      // Sanity: (2I - A) * R = I.
                      const Matrix shifted = mat({{1, -2}, {0, -1}});
                      auto prod = shifted.multiply(*r);
                      t.expect(prod && prod->is_equal(Matrix::identity(2)),
                               "(2I - A) R(2,A) = I");
                  }
              })
        .test("resolvent_domain_error_at_eigenvalue",
              [](TestContext& t) {
                  // lambda = 1 is an eigenvalue of A = [[1,2],[0,3]]; (I - A) is singular.
                  const Matrix a = mat({{1, 2}, {0, 3}});
                  auto r = resolvent(a, ri(1));
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "resolvent at an eigenvalue => domain_error");
                  auto r3 = resolvent(a, ri(3));
                  t.expect(!r3.has_value() && r3.error() == MathError::domain_error,
                           "resolvent at the other eigenvalue => domain_error");
              })
        .test("spectrum_of_triangular",
              [](TestContext& t) {
                  // Upper-triangular A = [[1,2],[0,3]]: spectrum {1, 3}, fully rational.
                  const Matrix a = mat({{1, 2}, {0, 3}});
                  auto s = spectrum(a);
                  t.expect(s.has_value(), "spectrum of a triangular matrix succeeds");
                  if (s) {
                      t.expect(s->fully_extracted, "triangular spectrum is fully rational");
                      t.expect(s->rational_count == 2, "two eigenvalues counted");
                      t.expect(s->dimension == 2, "dimension 2");
                  }
                  auto rho = spectral_radius(a);
                  t.expect(rho.has_value(), "spectral radius succeeds");
                  t.expect(rho && rho->exact, "spectral radius is exact (rational spectrum)");
                  t.expect(rho && rho->value == ri(3), "rho(A) = max(|1|,|3|) = 3");
              })
        .test("spectral_radius_bound_when_not_rational",
              [](TestContext& t) {
                  // Rotation A = [[0,-1],[1,0]] has eigenvalues +/- i (no rational part), so
                  // spectral_radius reports a rational upper bound = min(||A||_1, ||A||_inf) = 1.
                  const Matrix a = mat({{0, -1}, {1, 0}});
                  auto s = spectrum(a);
                  t.expect(s && !s->fully_extracted, "rotation spectrum is not fully rational");
                  auto rho = spectral_radius(a);
                  t.expect(rho.has_value(), "spectral radius (bound) succeeds");
                  t.expect(rho && !rho->exact, "reported as a bound, not exact");
                  t.expect(rho && rho->value == ri(1), "bound = min induced norm = 1");
              })
        .test("operator_norms_and_adjoint",
              [](TestContext& t) {
                  const Matrix a = mat({{1, -2}, {3, 4}});
                  auto n1 = operator_norm_1(a);
                  t.expect(n1 && *n1 == ri(6), "||A||_1 = max col |sum| = max(4,6) = 6");
                  auto ninf = operator_norm_inf(a);
                  t.expect(ninf && *ninf == ri(7), "||A||_inf = max row |sum| = max(3,7) = 7");
                  auto adj = adjoint(mat({{1, 2}, {3, 4}}));
                  t.expect(adj && adj->is_equal(mat({{1, 3}, {2, 4}})),
                           "adjoint = transpose (real operator)");
              })
        .test("semigroup_identity_at_zero",
              [](TestContext& t) {
                  // T(0) = e^0 = I exactly for any generator.
                  const Matrix n = mat({{0, 1}, {0, 0}});
                  auto t0 = semigroup(n, ri(0), 3);
                  t.expect(t0.has_value(), "T(0) succeeds");
                  t.expect(t0 && t0->is_equal(Matrix::identity(2)), "T(0) = I");
              })
        .test("semigroup_property_nilpotent_exact",
              [](TestContext& t) {
                  // Nilpotent N = [[0,1],[0,0]]: T(t) = I + tN exactly, so T(s+t) = T(s)T(t).
                  const Matrix n = mat({{0, 1}, {0, 0}});
                  auto prop = verify_semigroup_property(n, ri(1), ri(2), 4);
                  t.expect(prop.has_value() && prop.value_or(false),
                           "T(1+2) = T(1)T(2) exactly for the nilpotent generator");
                  // Direct pin: T(3) = [[1,3],[0,1]].
                  auto t3 = semigroup(n, ri(3), 4);
                  t.expect(t3 && t3->is_equal(mat({{1, 3}, {0, 1}})), "T(3) = I + 3N");
              })
        .test("cauchy_problem_solves_ode",
              [](TestContext& t) {
                  // u' = N u, u0 = [0,1]^T, N = [[0,1],[0,0]]. Then T(t) = [[1,t],[0,1]] and
                  // u(t) = [t, 1]^T. u is affine in t, so (u(1) - u(0)) = du/dt = N u0 = [1,0]^T.
                  const Matrix n = mat({{0, 1}, {0, 0}});
                  const Matrix u0 = col({0, 1});
                  auto u0_sol = cauchy_solution(n, u0, ri(0), 3);
                  t.expect(u0_sol && u0_sol->is_equal(u0), "u(0) = u0");
                  auto u1 = cauchy_solution(n, u0, ri(1), 3);
                  t.expect(u1 && u1->is_equal(col({1, 1})), "u(1) = [1,1]");
                  if (u0_sol && u1) {
                      auto dq = u1->subtract(*u0_sol);  // difference quotient over unit step
                      auto nu0 = n.multiply(u0);        // N u0 = du/dt at t=0
                      t.expect(dq && nu0 && dq->is_equal(*nu0),
                               "du/dt|_0 = N u0 = [1,0] (Cauchy solution solves u' = N u)");
                  }
                  // The abstract-PDE helper is the same Cauchy solution on a discretized generator.
                  auto pde = pde_semigroup_solution(n, u0, ri(1), 3);
                  t.expect(pde && u1 && pde->is_equal(*u1), "pde_semigroup_solution == T(t)u0");
              })
        .test("dissipative_and_contraction_generators",
              [](TestContext& t) {
                  // diag(-1,-2): symmetric part diag(-2,-4) is negative semidefinite -> dissipative;
                  // eigenvalues -1,-2 -> Hurwitz. Contraction generator (Lumer-Phillips).
                  const Matrix stable = mat({{-1, 0}, {0, -2}});
                  auto d1 = is_dissipative(stable);
                  t.expect(d1 && d1.value_or(false), "diag(-1,-2) is dissipative");
                  auto c1 = is_contraction_generator(stable);
                  t.expect(c1 && c1.value_or(false), "diag(-1,-2) generates a contraction");
                  auto h1 = is_hurwitz(stable);
                  t.expect(h1 && h1.value_or(false), "diag(-1,-2) is Hurwitz");

                  // Skew J = [[0,1],[-1,0]]: A + A^T = 0 -> dissipative (contraction), but the
                  // spectrum is {+/- i} on the imaginary axis, so NOT Hurwitz. Distinguishes the two.
                  const Matrix skew = mat({{0, 1}, {-1, 0}});
                  auto d2 = is_dissipative(skew);
                  t.expect(d2 && d2.value_or(false), "skew-symmetric is dissipative");
                  auto c2 = is_contraction_generator(skew);
                  t.expect(c2 && c2.value_or(false), "skew-symmetric generates a contraction");
                  auto h2 = is_hurwitz(skew);
                  t.expect(h2.has_value() && !h2.value_or(true),
                           "skew-symmetric is NOT Hurwitz (imaginary-axis spectrum)");

                  // Identity: symmetric part diag(2,2) is positive definite -> NOT dissipative.
                  const Matrix id = Matrix::identity(2);
                  auto d3 = is_dissipative(id);
                  t.expect(d3.has_value() && !d3.value_or(true), "identity is not dissipative");
                  auto c3 = is_contraction_generator(id);
                  t.expect(c3.has_value() && !c3.value_or(true),
                           "identity does not generate a contraction");
              })
        .test("sylvester_reconstructs_c",
              [](TestContext& t) {
                  // A X + X B = C with A=diag(1,2), B=diag(3,4), X=[[1,2],[3,4]] gives
                  // C = [[4,10],[15,24]]; the solver must recover X exactly.
                  const Matrix a = mat({{1, 0}, {0, 2}});
                  const Matrix b = mat({{3, 0}, {0, 4}});
                  const Matrix x = mat({{1, 2}, {3, 4}});
                  const Matrix c = mat({{4, 10}, {15, 24}});
                  auto sol = sylvester_solve(a, b, c);
                  t.expect(sol.has_value(), "Sylvester solve succeeds");
                  t.expect(sol && sol->is_equal(x), "solve(A,B,C) = X exactly");
                  if (sol) {
                      // Reconstruct C = A X + X B from the solution.
                      auto ax = a.multiply(*sol);
                      auto xb = sol->multiply(b);
                      if (ax && xb) {
                          auto recon = ax->add(*xb);
                          t.expect(recon && recon->is_equal(c), "A X + X B reconstructs C");
                      }
                  }
              })
        .test("lyapunov_special_case",
              [](TestContext& t) {
                  // A X + X A^T = C, A = diag(-1,-2), X = [[2,1],[1,3]] -> C = [[-4,-3],[-3,-12]].
                  const Matrix a = mat({{-1, 0}, {0, -2}});
                  const Matrix x = mat({{2, 1}, {1, 3}});
                  const Matrix c = mat({{-4, -3}, {-3, -12}});
                  auto sol = lyapunov_solve(a, c);
                  t.expect(sol.has_value(), "Lyapunov solve succeeds");
                  t.expect(sol && sol->is_equal(x), "lyapunov_solve(A,C) = X exactly");
              })
        .test("variation_of_constants_constant_forcing",
              [](TestContext& t) {
                  // u' = N u + f, N = [[0,1],[0,0]], u0 = 0, constant f = [1,0]^T.
                  // Exact solution u(t) = [t, 0]^T (verified: u' = [1,0] = N u + f).
                  const Matrix n = mat({{0, 1}, {0, 0}});
                  const Matrix u0 = col({0, 0});
                  const std::vector<Matrix> f = {col({1, 0})};
                  auto u1 = variation_of_constants(n, u0, f, ri(1), 3);
                  t.expect(u1 && u1->is_equal(col({1, 0})), "u(1) = [1,0]");
                  auto u2 = variation_of_constants(n, u0, f, ri(2), 3);
                  t.expect(u2 && u2->is_equal(col({2, 0})), "u(2) = [2,0]");
                  // u affine => du/dt = u(2) - u(1) = [1,0]; check it equals N u(1) + f.
                  if (u1 && u2) {
                      auto dq = u2->subtract(*u1);
                      auto nu = n.multiply(*u1);
                      if (dq && nu) {
                          auto rhs = nu->add(f.front());
                          t.expect(rhs && dq->is_equal(*rhs),
                                   "du/dt = N u + f (variation of constants solves the ODE)");
                      }
                  }
              })
        .test("variation_of_constants_linear_forcing",
              [](TestContext& t) {
                  // u' = N u + f(s), N = [[0,1],[0,0]], u0 = 0, f(s) = [s,0]^T (c0=0, c1=[1,0]).
                  // Direct solve: u2' = 0 -> u2 = 0; u1' = u2 + s = s -> u1 = t^2/2.
                  // So u(t) = [t^2/2, 0]^T; at t=2, u = [2, 0]^T.
                  const Matrix n = mat({{0, 1}, {0, 0}});
                  const Matrix u0 = col({0, 0});
                  const std::vector<Matrix> f = {col({0, 0}), col({1, 0})};  // f(s)=c0+c1 s
                  auto u = variation_of_constants(n, u0, f, ri(2), 4);
                  t.expect(u.has_value(), "variation of constants with linear forcing succeeds");
                  t.expect(u && u->is_equal(col({2, 0})), "u(2) = [t^2/2,0]|_{t=2} = [2,0]");
                  // Homogeneous (empty forcing) must reduce to the plain Cauchy solution.
                  auto homo = variation_of_constants(n, col({0, 1}), {}, ri(1), 3);
                  auto cauchy = cauchy_solution(n, col({0, 1}), ri(1), 3);
                  t.expect(homo && cauchy && homo->is_equal(*cauchy),
                           "empty forcing == homogeneous Cauchy solution");
              })
        .test("domain_errors",
              [](TestContext& t) {
                  const Matrix ns = mat({{1, 2, 3}, {4, 5, 6}});  // non-square
                  auto r = resolvent(ns, ri(1));
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "resolvent of non-square => domain_error");
                  auto s = spectrum(ns);
                  t.expect(!s.has_value() && s.error() == MathError::domain_error,
                           "spectrum of non-square => domain_error");
                  auto sg = semigroup(Matrix::identity(2), ri(1), 0);
                  t.expect(!sg.has_value() && sg.error() == MathError::domain_error,
                           "semigroup terms < 1 => domain_error");
                  auto sg2 = semigroup(ns, ri(1), 3);
                  t.expect(!sg2.has_value() && sg2.error() == MathError::domain_error,
                           "semigroup of non-square => domain_error");
                  // Cauchy with a mismatched initial vector.
                  auto cp = cauchy_solution(Matrix::identity(2), col({1, 2, 3}), ri(1), 3);
                  t.expect(!cp.has_value() && cp.error() == MathError::domain_error,
                           "cauchy_solution with wrong u0 shape => domain_error");
                  // Sylvester with a C of the wrong shape (expects m x n = 2 x 2).
                  auto sy = sylvester_solve(mat({{1, 0}, {0, 1}}), mat({{1, 0}, {0, 1}}),
                                            mat({{1, 2, 3}, {4, 5, 6}}));
                  t.expect(!sy.has_value() && sy.error() == MathError::domain_error,
                           "sylvester_solve with wrong C shape => domain_error");
                  // Dissipativity of a non-square matrix.
                  auto d = is_dissipative(ns);
                  t.expect(!d.has_value() && d.error() == MathError::domain_error,
                           "is_dissipative of non-square => domain_error");
              })
        .run();
}
