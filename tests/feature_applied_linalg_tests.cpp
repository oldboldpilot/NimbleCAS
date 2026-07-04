// Feature/integration tests: applied exact linear algebra across the structured, Krylov,
// Lie, operator-semigroup, control and spectral modules.
// @author Olumuyiwa Oluwasanmi
//
// These are cross-module *feature* tests: every case wires two or more of the exact
// rational modules together and asserts a concrete value or a mathematical identity that
// only holds because every intermediate step is a Rational (never a float). Nothing here is
// approximate — L D L^T reconstructs A on the nose, exact CG lands on the exact Gaussian
// solution, and two independent Lyapunov solvers return the identical fraction matrix.
//
// ---------------------------------------------------------------------------------------
// NOTE: this suite co-links nimblecas.analysis + nimblecas.semigroup + nimblecas.control.
// ---------------------------------------------------------------------------------------
// An earlier version could NOT: all three modules exported, into the same `namespace
// nimblecas`, non-inline functions with identical signatures (lyapunov_solve(Matrix,Matrix)
// in analysis+semigroup; is_positive_definite / is_stable_lyapunov in analysis+control),
// which are duplicate strong symbols -> an ODR violation and a duplicate-symbol link error
// on co-linking. That latent defect was fixed by renaming the semigroup/control copies
// (semigroup: lyapunov_solve -> lyapunov_equation; control: is_positive_definite -> is_spd,
// is_stable_lyapunov -> is_lyapunov_stable), leaving analysis as the canonical owner. This
// suite now performs the full FOUR-way exact Lyapunov cross-check (control, semigroup's
// lyapunov_equation, semigroup's sylvester_solve, and analysis) — its successful link is
// itself the regression test that the collision is gone.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.eigen;
import nimblecas.matdecomp;
import nimblecas.matexp;
import nimblecas.matstruct;
import nimblecas.lie;
import nimblecas.krylov;
import nimblecas.dynamics;
import nimblecas.semigroup;
import nimblecas.control;
import nimblecas.analysis;
import nimblecas.testing;

using nimblecas::characteristic_polynomial;
using nimblecas::conjugate_gradient;
using nimblecas::conjugate_gradient_steps;
using nimblecas::hessenberg_form;
using nimblecas::is_asymptotically_stable;
using nimblecas::is_nilpotent;
using nimblecas::is_symmetric_positive_definite;
using nimblecas::is_upper_hessenberg;
using nimblecas::killing_form;
using nimblecas::ldlt_decompose;
using nimblecas::lie_bracket;
using nimblecas::Matrix;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::rational_eigenvalues;
using nimblecas::structure_constants;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// --- Rational / Matrix builders (mirroring the sibling feature suites) -------

[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

[[nodiscard]] auto ri(std::int64_t v) -> Rational {
    return Rational::from_int(v);
}

// Build a Matrix from integer rows (low-index row first).
[[nodiscard]] auto mat(std::vector<std::vector<std::int64_t>> rows) -> Matrix {
    std::vector<std::vector<Rational>> r;
    r.reserve(rows.size());
    for (const auto& row : rows) {
        std::vector<Rational> rr;
        rr.reserve(row.size());
        for (const std::int64_t v : row) {
            rr.push_back(Rational::from_int(v));
        }
        r.push_back(std::move(rr));
    }
    return Matrix::from_rows(std::move(r)).value();
}

// A single-column Matrix from a list of Rationals.
[[nodiscard]] auto col(std::vector<Rational> entries) -> Matrix {
    std::vector<std::vector<Rational>> rows;
    rows.reserve(entries.size());
    for (auto& e : entries) {
        rows.push_back({std::move(e)});
    }
    return Matrix::from_rows(std::move(rows)).value();
}

// The n x n diagonal Matrix with `entries` on the main diagonal (0 off-diagonal). Used to
// name the expected D of an LDL^T factorization exactly.
[[nodiscard]] auto diag(std::vector<Rational> entries) -> Matrix {
    const std::size_t n = entries.size();
    std::vector<std::vector<Rational>> rows(n, std::vector<Rational>(n, ri(0)));
    for (std::size_t i = 0; i < n; ++i) {
        rows[i][i] = entries[i];
    }
    return Matrix::from_rows(std::move(rows)).value();
}

// -lambda*I ... the scaled negative identity, for building right-hand sides like -I.
[[nodiscard]] auto neg_identity(std::size_t n) -> Matrix {
    return Matrix::identity(n).scale(ri(-1)).value();
}

// Reconstruct A = L * D * L^T exactly from an LDL^T result.
[[nodiscard]] auto reconstruct_ldlt(const Matrix& l, const Matrix& d) -> Matrix {
    const Matrix lt = l.transpose().value();
    const Matrix ld = l.multiply(d).value();
    return ld.multiply(lt).value();
}

}  // namespace

auto main() -> int {
    return TestSuite("feature.applied_linalg")
        // === MATSTRUCT: exact LDL^T reconstruction and positive-definiteness ==========
        .test("ldlt_reconstructs_spd_and_pivot_signs_decide_definiteness",
              [](TestContext& t) {
                  // Classic SPD example; its exact LDL^T has L unit-lower and D = diag(4,1,9),
                  // all perfect-square pivots (so it is genuinely positive definite over Q).
                  auto a = mat({{4, 12, -16}, {12, 37, -43}, {-16, -43, 98}});
                  auto ldlt = ldlt_decompose(a).value();

                  auto expected_l = mat({{1, 0, 0}, {3, 1, 0}, {-4, 5, 1}});
                  auto expected_d = diag({ri(4), ri(1), ri(9)});
                  t.expect(ldlt.l == expected_l, "L is the exact unit-lower factor");
                  t.expect(ldlt.d == expected_d, "D holds the exact pivots diag(4,1,9)");
                  t.expect(reconstruct_ldlt(ldlt.l, ldlt.d) == a, "L * D * L^T == A exactly");

                  // is_symmetric_positive_definite must AGREE with 'every pivot D_ii > 0'.
                  bool all_pivots_positive = true;
                  for (std::size_t i = 0; i < a.rows(); ++i) {
                      if (ldlt.d.at(i, i).numerator() <= 0) {
                          all_pivots_positive = false;
                      }
                  }
                  t.expect(all_pivots_positive, "all LDL^T pivots are strictly positive");
                  t.expect(is_symmetric_positive_definite(a) == all_pivots_positive,
                           "is_symmetric_positive_definite agrees with positive-pivot test");
                  t.expect(is_symmetric_positive_definite(a), "SPD matrix certified positive definite");

                  // An indefinite symmetric matrix: LDL^T exists (no zero pivot) but a pivot is
                  // negative, and the predicate agrees by returning false.
                  auto indef = mat({{1, 2}, {2, 1}});  // eigenvalues 3, -1
                  auto ldlt2 = ldlt_decompose(indef).value();
                  t.expect(ldlt2.d.at(0, 0) == ri(1) && ldlt2.d.at(1, 1) == ri(-3),
                           "indefinite pivots are (1, -3)");
                  bool all_pos2 = ldlt2.d.at(0, 0).numerator() > 0 && ldlt2.d.at(1, 1).numerator() > 0;
                  t.expect(!all_pos2, "a pivot is non-positive for the indefinite matrix");
                  t.expect(is_symmetric_positive_definite(indef) == all_pos2,
                           "predicate agrees: indefinite matrix is not positive definite");
                  t.expect(reconstruct_ldlt(ldlt2.l, ldlt2.d) == indef,
                           "L * D * L^T == A even for the indefinite matrix");
              })
        // === KRYLOV exact CG == MATRIX::solve on an SPD rational system ================
        .test("exact_cg_matches_dense_solve_and_terminates_within_n_steps",
              [](TestContext& t) {
                  // The SPD 1D Laplacian tridiag(-1,2,-1), n = 3, rhs = ones.
                  auto a = mat({{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}});
                  auto b = col({ri(1), ri(1), ri(1)});

                  auto cg = conjugate_gradient_steps(a, b).value();
                  auto dense = a.solve(b).value();

                  // The exact rational solution is (3/2, 2, 3/2).
                  auto expected = col({rat(3, 2), ri(2), rat(3, 2)});
                  t.expect(cg.solution == expected, "CG lands on the exact solution (3/2, 2, 3/2)");
                  t.expect(cg.solution == dense, "exact CG == Matrix::solve, bit-for-bit over Q");
                  t.expect(a.multiply(cg.solution).value() == b, "A * x_cg == b exactly");
                  t.expect(cg.steps <= a.rows(), "CG on an SPD system terminates in <= n steps");

                  // The convenience wrapper returns the same exact vector.
                  t.expect(conjugate_gradient(a, b).value() == dense,
                           "conjugate_gradient == conjugate_gradient_steps solution");
              })
        // === SEMIGROUP vs CONTROL: two independent exact Lyapunov solvers agree =======
        .test("lyapunov_solution_agrees_across_semigroup_and_control_and_is_pd",
              [](TestContext& t) {
                  // A Hurwitz companion matrix (eigenvalues -1, -2). The unique P of
                  // A^T P + P A = -I is exactly [[5/4, 1/4], [1/4, 1/4]] (positive definite).
                  auto a = mat({{0, 1}, {-2, -3}});
                  auto at = a.transpose().value();
                  auto neg_i = neg_identity(2);

                  // control::lyapunov_solve(A) solves A^T P + P A = -I directly (1-arg overload).
                  auto p_control = nimblecas::lyapunov_solve(a).value();
                  // semigroup::lyapunov_equation(a, c) solves a X + X a^T = c; with a = A^T, c = -I
                  // this is A^T X + X A = -I — the SAME equation, via an independent Kronecker
                  // vectorization / Sylvester code path.
                  auto p_semigroup = nimblecas::lyapunov_equation(at, neg_i).value();
                  // A third exact route: semigroup::sylvester_solve(A^T, A, -I) is A^T X + X A = -I.
                  auto p_sylvester = nimblecas::sylvester_solve(at, a, neg_i).value();
                  // A fourth: analysis owns the canonical lyapunov_solve(A, Q) solving
                  // A^T P + P A = -Q; with Q = I it matches the others. (This co-links analysis
                  // with semigroup and control — previously impossible due to a duplicate-symbol
                  // collision on these names, now resolved by renaming the semigroup/control copies.)
                  auto p_analysis = nimblecas::lyapunov_solve(a, Matrix::identity(2)).value();

                  auto expected_p = Matrix::from_rows({{rat(5, 4), rat(1, 4)},
                                                       {rat(1, 4), rat(1, 4)}})
                                        .value();
                  t.expect(p_control == expected_p, "control Lyapunov P == [[5/4,1/4],[1/4,1/4]]");
                  t.expect(p_control == p_semigroup, "control P == semigroup P exactly over Q");
                  t.expect(p_semigroup == p_sylvester, "semigroup Lyapunov == its Sylvester special case");
                  t.expect(p_control == p_analysis, "control P == analysis P exactly over Q (4-way agreement)");

                  // Residual check: A^T P + P A must equal -I on the nose.
                  auto residual = at.multiply(p_control).value().add(p_control.multiply(a).value()).value();
                  t.expect(residual == neg_i, "A^T P + P A == -I exactly");

                  // P is positive definite, cross-certified by two independent predicates.
                  t.expect(is_symmetric_positive_definite(p_control),
                           "matstruct certifies P positive definite");
                  t.expect(nimblecas::is_spd(p_control).value(),
                           "control (Sylvester's criterion) certifies P positive definite");
              })
        .test("stability_verdicts_agree_across_control_semigroup_and_dynamics",
              [](TestContext& t) {
                  // Stable (Hurwitz) A: every exact stability route must return true.
                  auto stable = mat({{0, 1}, {-2, -3}});
                  bool s_lyap = nimblecas::is_lyapunov_stable(stable).value();   // control, Lyapunov/Sylvester
                  bool s_hurwitz = nimblecas::is_hurwitz(stable).value();         // semigroup, Routh-Hurwitz
                  bool s_routh = is_asymptotically_stable(stable).value();        // dynamics, Routh-Hurwitz
                  t.expect(s_lyap && s_hurwitz && s_routh, "stable A: all three verdicts are true");
                  t.expect(s_lyap == s_hurwitz && s_hurwitz == s_routh,
                           "stable A: Lyapunov and Routh-Hurwitz verdicts agree");

                  // Unstable A (eigenvalues +1, +2 in the right half-plane).
                  auto unstable = mat({{1, 0}, {0, 2}});
                  bool u_lyap = nimblecas::is_lyapunov_stable(unstable).value();
                  bool u_hurwitz = nimblecas::is_hurwitz(unstable).value();
                  bool u_routh = is_asymptotically_stable(unstable).value();
                  t.expect(!u_lyap && !u_hurwitz && !u_routh, "unstable A: all three verdicts are false");
                  t.expect(u_lyap == u_hurwitz && u_hurwitz == u_routh,
                           "unstable A: Lyapunov and Routh-Hurwitz verdicts agree");

                  // The unstable Lyapunov solution exists but is NOT positive definite, and both
                  // definiteness predicates agree on that.
                  auto p_u = nimblecas::lyapunov_solve(unstable).value();
                  t.expect(!is_symmetric_positive_definite(p_u),
                           "matstruct: unstable Lyapunov P is not positive definite");
                  t.expect(!nimblecas::is_spd(p_u).value(),
                           "control: unstable Lyapunov P is not positive definite");
              })
        // === LIE: so(3) structure constants, Killing form, Jacobi identity ============
        .test("so3_brackets_structure_constants_killing_form_and_jacobi",
              [](TestContext& t) {
                  // The standard antisymmetric so(3) generators (rotation generators).
                  auto lx = mat({{0, 0, 0}, {0, 0, -1}, {0, 1, 0}});
                  auto ly = mat({{0, 0, 1}, {0, 0, 0}, {-1, 0, 0}});
                  auto lz = mat({{0, -1, 0}, {1, 0, 0}, {0, 0, 0}});

                  // Cyclic bracket relations [L_x,L_y] = L_z, [L_y,L_z] = L_x, [L_z,L_x] = L_y.
                  t.expect(lie_bracket(lx, ly).value() == lz, "[L_x, L_y] == L_z");
                  t.expect(lie_bracket(ly, lz).value() == lx, "[L_y, L_z] == L_x");
                  t.expect(lie_bracket(lz, lx).value() == ly, "[L_z, L_x] == L_y");

                  // Structure constants c^k_ij with [X_i,X_j] = sum_k c^k_ij X_k, basis order
                  // {L_x, L_y, L_z} = {0,1,2}. These must be the Levi-Civita symbol epsilon_ijk.
                  std::vector<Matrix> basis{lx, ly, lz};
                  auto sc = structure_constants(basis).value();
                  t.expect(sc.dimension() == 3, "so(3) has a 3-dimensional basis");
                  t.expect(sc.at(0, 1, 2) == ri(1), "c^z_{xy} == +1 (epsilon_{012})");
                  t.expect(sc.at(1, 2, 0) == ri(1), "c^x_{yz} == +1 (epsilon_{120})");
                  t.expect(sc.at(2, 0, 1) == ri(1), "c^y_{zx} == +1 (epsilon_{201})");
                  t.expect(sc.at(1, 0, 2) == ri(-1), "c^z_{yx} == -1 (antisymmetry)");
                  t.expect(sc.at(0, 1, 2) == sc.at(1, 0, 2).negate().value(),
                           "structure constants are antisymmetric in i,j");
                  t.expect(sc.at(0, 0, 0) == ri(0) && sc.at(1, 1, 2) == ri(0),
                           "diagonal / self-brackets contribute nothing");

                  // Killing form K(X,Y) = trace(ad_X ad_Y). For so(3) it is exactly -2 * delta_ij.
                  t.expect(killing_form(lx, lx, basis).value() == ri(-2), "K(L_x, L_x) == -2");
                  t.expect(killing_form(ly, ly, basis).value() == ri(-2), "K(L_y, L_y) == -2");
                  t.expect(killing_form(lx, ly, basis).value() == ri(0), "K(L_x, L_y) == 0 (orthogonal)");
                  t.expect(killing_form(ly, lz, basis).value() == ri(0), "K(L_y, L_z) == 0 (orthogonal)");

                  // Jacobi identity [X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] == 0 exactly.
                  auto j1 = lie_bracket(lx, lie_bracket(ly, lz).value()).value();
                  auto j2 = lie_bracket(ly, lie_bracket(lz, lx).value()).value();
                  auto j3 = lie_bracket(lz, lie_bracket(lx, ly).value()).value();
                  auto jac = j1.add(j2).value().add(j3).value();
                  t.expect(jac == Matrix::zero(3, 3), "Jacobi identity holds identically over Q");
              })
        // === SEMIGROUP: resolvent, spectrum vs eigen, and nilpotent semigroup laws =====
        .test("resolvent_spectrum_and_nilpotent_semigroup_identities",
              [](TestContext& t) {
                  auto a = mat({{0, 1}, {-2, -3}});  // eigenvalues -1, -2

                  // Resolvent R(lambda,A) = (lambda I - A)^{-1}; 5 is not in the spectrum.
                  auto r = nimblecas::resolvent(a, ri(5)).value();
                  auto shifted = Matrix::identity(2).scale(ri(5)).value().subtract(a).value();
                  t.expect(r == shifted.inverse().value(), "resolvent == (lambda I - A)^{-1}");
                  t.expect(shifted.multiply(r).value() == Matrix::identity(2),
                           "(lambda I - A) * R == I exactly");
                  t.expect(r.multiply(shifted).value() == Matrix::identity(2),
                           "R * (lambda I - A) == I exactly");

                  // spectrum() must reproduce eigen's rational eigenvalues exactly.
                  auto diag_a = mat({{2, 0}, {0, 3}});  // rational spectrum {2, 3}
                  auto spec = nimblecas::spectrum(diag_a).value();
                  auto eig = rational_eigenvalues(diag_a).value();
                  t.expect(spec.fully_extracted, "spectrum of a rational matrix is fully extracted");
                  t.expect(spec.rational_count == 2, "two rational eigenvalues counted");
                  t.expect(spec.rational_values == eig, "semigroup spectrum == eigen rational_eigenvalues");

                  // Nilpotent generator: T(0) = I and T(s+t) = T(s) T(t) hold EXACTLY.
                  auto n = mat({{0, 1, 0}, {0, 0, 1}, {0, 0, 0}});  // nilpotency index 3
                  t.expect(is_nilpotent(n).value(), "N is nilpotent");
                  t.expect(nimblecas::semigroup(n, ri(0), 4).value() == Matrix::identity(3),
                           "T(0) == I exactly");
                  // e^{1*N} = I + N + N^2/2 (the tail N^3, N^4, ... vanish).
                  auto expected_t1 = Matrix::from_rows({{ri(1), ri(1), rat(1, 2)},
                                                        {ri(0), ri(1), ri(1)},
                                                        {ri(0), ri(0), ri(1)}})
                                         .value();
                  t.expect(nimblecas::semigroup(n, ri(1), 4).value() == expected_t1,
                           "T(1) == I + N + N^2/2 exactly for the nilpotent generator");
                  t.expect(nimblecas::verify_semigroup_property(n, ri(1), ri(2), 4).value(),
                           "T(s+t) == T(s) T(t) exactly (nilpotent, enough terms)");
              })
        // === MATSTRUCT Hessenberg is a similarity: same characteristic polynomial =====
        .test("hessenberg_form_preserves_characteristic_polynomial",
              [](TestContext& t) {
                  // A general (non-Hessenberg) 4x4 with entries below the subdiagonal.
                  auto a = mat({{4, 1, -2, 2}, {1, 2, 0, 1}, {-2, 0, 3, -2}, {2, 1, -2, -1}});
                  auto h = hessenberg_form(a).value();

                  t.expect(is_upper_hessenberg(h), "reduced matrix is upper-Hessenberg");

                  // Similarity invariant: identical characteristic polynomial (hence identical
                  // eigenvalues, determinant and trace), computed via eigen's Faddeev-LeVerrier.
                  auto pa = characteristic_polynomial(a).value();
                  auto ph = characteristic_polynomial(h).value();
                  t.expect(pa.is_equal(ph), "char poly of H == char poly of A (similarity invariant)");
                  t.expect(a.determinant().value() == h.determinant().value(),
                           "det(H) == det(A) (similarity invariant)");
                  t.expect(a.trace().value() == h.trace().value(),
                           "trace(H) == trace(A) (similarity invariant)");
                  t.expect(rational_eigenvalues(a).value() == rational_eigenvalues(h).value(),
                           "rational eigenvalues of H match those of A");
              })
        .run();
}
