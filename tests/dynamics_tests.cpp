// Tests for nimblecas.dynamics: stability of linear/affine autonomous systems (ROADMAP 7.10).
// @author Olumuyiwa Oluwasanmi
//
// Exact integer matrices are used throughout so every equilibrium, Routh array, and
// eigenvalue sign is exact and deterministic.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.dynamics;
import nimblecas.testing;

using nimblecas::classify_equilibrium;
using nimblecas::classify_linear_stability;
using nimblecas::classify_phase_portrait;
using nimblecas::fixed_point_affine;
using nimblecas::is_asymptotically_stable;
using nimblecas::is_perfect_square;
using nimblecas::LinearStability;
using nimblecas::MathError;
using nimblecas::Matrix;
using nimblecas::PhaseType;
using nimblecas::Rational;
using nimblecas::Stability;
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
        std::vector<Rational> rr;
        rr.reserve(row.size());
        for (const std::int64_t v : row) {
            rr.push_back(Rational::from_int(v));
        }
        r.push_back(std::move(rr));
    }
    return Matrix::from_rows(std::move(r)).value();
}

// An n x 1 column matrix from integer entries.
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
    return TestSuite("nimblecas.dynamics")
        .test("fixed_point_affine_zero_A",
              [](TestContext& t) {
                  // A = 0 => (I - A) = I, so x = b.
                  auto x = fixed_point_affine(mat({{0, 0}, {0, 0}}), col({3, 5}));
                  t.expect(x.has_value(), "solve succeeds for A = 0");
                  if (x) {
                      t.expect(x->rows() == 2 && x->cols() == 1, "equilibrium is a 2x1 column");
                      t.expect(x->at(0, 0) == ri(3) && x->at(1, 0) == ri(5), "x = [3, 5]");
                  }
              })
        .test("fixed_point_affine_scaled",
              [](TestContext& t) {
                  // A = 2I => (I - A) = -I, so -x = b => x = -b = [-1, -1].
                  auto x = fixed_point_affine(mat({{2, 0}, {0, 2}}), col({1, 1}));
                  t.expect(x.has_value(), "solve succeeds for A = 2I");
                  if (x) {
                      t.expect(x->at(0, 0) == ri(-1) && x->at(1, 0) == ri(-1), "x = [-1, -1]");
                  }
              })
        .test("fixed_point_affine_singular",
              [](TestContext& t) {
                  // A = I => (I - A) = 0, a non-isolated equilibrium => domain_error.
                  auto x = fixed_point_affine(mat({{1, 0}, {0, 1}}), col({1, 1}));
                  t.expect(!x.has_value(), "singular (I - A) is rejected");
                  t.expect(x.error() == MathError::domain_error, "singular => domain_error");
              })
        .test("fixed_point_affine_shape_errors",
              [](TestContext& t) {
                  // Non-square A.
                  auto ns = fixed_point_affine(mat({{1, 2, 3}, {4, 5, 6}}), col({1, 1}));
                  t.expect(!ns.has_value() && ns.error() == MathError::domain_error,
                           "non-square A => domain_error");
                  // b with the wrong number of rows.
                  auto badrows = fixed_point_affine(mat({{0, 0}, {0, 0}}), col({1, 2, 3}));
                  t.expect(!badrows.has_value() && badrows.error() == MathError::domain_error,
                           "b with wrong row count => domain_error");
                  // b that is not a single column.
                  auto badcols = fixed_point_affine(mat({{0, 0}, {0, 0}}), mat({{1, 2}, {3, 4}}));
                  t.expect(!badcols.has_value() && badcols.error() == MathError::domain_error,
                           "b that is not a column => domain_error");
              })
        .test("asymptotically_stable_diagonal_negative",
              [](TestContext& t) {
                  // Eigenvalues -1, -2 (char poly lambda^2 + 3 lambda + 2): Hurwitz.
                  auto s = is_asymptotically_stable(mat({{-1, 0}, {0, -2}}));
                  t.expect(s.has_value(), "Routh array builds");
                  t.expect(s.value_or(false), "diag(-1, -2) is asymptotically stable");
              })
        .test("asymptotically_stable_diagonal_positive",
              [](TestContext& t) {
                  // Eigenvalues 1, 1 (char poly lambda^2 - 2 lambda + 1): not Hurwitz.
                  auto s = is_asymptotically_stable(mat({{1, 0}, {0, 1}}));
                  t.expect(s.has_value(), "Routh array builds");
                  t.expect(!s.value_or(true), "diag(1, 1) is not asymptotically stable");
              })
        .test("asymptotically_stable_rotation_imaginary",
              [](TestContext& t) {
                  // Eigenvalues +/- i (char poly lambda^2 + 1): marginal. A fully-zero second
                  // Routh row -- the imaginary-axis case that rational root testing over Q
                  // could NOT decide -- must resolve to "not asymptotically stable".
                  auto s = is_asymptotically_stable(mat({{0, -1}, {1, 0}}));
                  t.expect(s.has_value(), "Routh array builds for the rotation");
                  t.expect(!s.value_or(true), "rotation (eigenvalues +/- i) is not asymptotically stable");
              })
        .test("asymptotically_stable_defective_negative",
              [](TestContext& t) {
                  // Jordan block with eigenvalue -1 (char poly lambda^2 + 2 lambda + 1): Hurwitz.
                  auto s = is_asymptotically_stable(mat({{-1, 1}, {0, -1}}));
                  t.expect(s.has_value(), "Routh array builds");
                  t.expect(s.value_or(false), "[[-1,1],[0,-1]] is asymptotically stable");
              })
        .test("asymptotically_stable_shape_error",
              [](TestContext& t) {
                  auto ns = is_asymptotically_stable(mat({{1, 2, 3}, {4, 5, 6}}));
                  t.expect(!ns.has_value() && ns.error() == MathError::domain_error,
                           "non-square A => domain_error");
              })
        .test("classify_stable_node",
              [](TestContext& t) {
                  auto c = classify_equilibrium(mat({{-1, 0}, {0, -2}}));
                  t.expect(c.has_value(), "classify succeeds");
                  t.expect(c.value_or("") == "stable node", "diag(-1, -2) => stable node");
              })
        .test("classify_unstable_node",
              [](TestContext& t) {
                  auto c = classify_equilibrium(mat({{1, 0}, {0, 2}}));
                  t.expect(c.has_value(), "classify succeeds");
                  t.expect(c.value_or("") == "unstable node", "diag(1, 2) => unstable node");
              })
        .test("classify_saddle",
              [](TestContext& t) {
                  auto c = classify_equilibrium(mat({{-1, 0}, {0, 2}}));
                  t.expect(c.has_value(), "classify succeeds");
                  t.expect(c.value_or("") == "saddle", "diag(-1, 2) => saddle");
              })
        .test("classify_zero_with_positive_is_unstable",
              [](TestContext& t) {
                  // diag(0, 1) grows like e^t: a positive eigenvalue makes it unstable, and
                  // must NOT be reported as merely marginal just because a zero is present.
                  auto c = classify_equilibrium(mat({{0, 0}, {0, 1}}));
                  t.expect(c.has_value(), "classify succeeds");
                  t.expect(c.value_or("") == "unstable (with marginal direction)",
                           "diag(0, 1) => unstable (with marginal direction)");
              })
        .test("classify_zero_with_negative_is_marginally_stable",
              [](TestContext& t) {
                  // diag(0, -1): bounded but not asymptotically stable (the zero direction
                  // neither grows nor decays) => marginally stable, not a stable node.
                  auto c = classify_equilibrium(mat({{0, 0}, {0, -1}}));
                  t.expect(c.has_value(), "classify succeeds");
                  t.expect(c.value_or("") == "marginally stable",
                           "diag(0, -1) => marginally stable");
              })
        .test("classify_non_rational_fallback",
              [](TestContext& t) {
                  // Rotation: spectrum +/- i is not rational, so the Routh-Hurwitz fallback
                  // applies -- and it is not asymptotically stable.
                  auto c = classify_equilibrium(mat({{0, -1}, {1, 0}}));
                  t.expect(c.has_value(), "classify succeeds");
                  t.expect(c.value_or("") == "unstable or marginal (non-rational spectrum)",
                           "rotation => non-rational marginal fallback string");
              })
        .test("classify_shape_error",
              [](TestContext& t) {
                  auto c = classify_equilibrium(mat({{1, 2, 3}, {4, 5, 6}}));
                  t.expect(!c.has_value() && c.error() == MathError::domain_error,
                           "non-square A => domain_error");
              })
        // ---- is_perfect_square: exact rational perfect-square test --------------
        .test("is_perfect_square_exact",
              [](TestContext& t) {
                  t.expect(is_perfect_square(ri(0)), "0 is a perfect square");
                  t.expect(is_perfect_square(ri(4)), "4 is a perfect square");
                  t.expect(is_perfect_square(ri(9)), "9 is a perfect square");
                  t.expect(!is_perfect_square(ri(2)), "2 is not a perfect square");
                  t.expect(!is_perfect_square(ri(-4)), "-4 is not a (real) perfect square");
                  // 4/9 = (2/3)^2 is a rational square; 2/9 is not.
                  t.expect(is_perfect_square(Rational::make(4, 9).value()), "4/9 is a square");
                  t.expect(!is_perfect_square(Rational::make(2, 9).value()), "2/9 is not a square");
                  t.expect(!is_perfect_square(Rational::make(4, 5).value()), "4/5 is not a square");
              })
        // ---- classify_phase_portrait: the exact 2x2 trace-determinant plane -----
        .test("phase_center",
              [](TestContext& t) {
                  // [[0,-1],[1,0]]: T=0, D=1, delta=-4 => purely imaginary +/- i => CENTER.
                  auto p = classify_phase_portrait(mat({{0, -1}, {1, 0}}));
                  t.expect(p.has_value(), "classify succeeds");
                  if (p) {
                      t.expect(p->type == PhaseType::center, "center");
                      t.expect(p->stability == Stability::neutrally_stable, "neutrally stable");
                      t.expect(p->trace == ri(0) && p->determinant == ri(1) &&
                                   p->discriminant == ri(-4),
                               "T=0, D=1, delta=-4");
                      t.expect(p->complex_eigenvalues, "complex eigenvalues");
                      t.expect(p->real_part.has_value() && p->real_part.value() == ri(0),
                               "real part 0");
                      t.expect(p->imag_part.has_value() && p->imag_part.value() == ri(1),
                               "imag part 1 (eigenvalues +/- i)");
                      t.expect(p->eigenvalues_rational, "+/- i is Gaussian-rational");
                  }
              })
        .test("phase_stable_node",
              [](TestContext& t) {
                  // [[-1,0],[0,-2]]: T=-3, D=2, delta=1 => real -1,-2, both negative.
                  auto p = classify_phase_portrait(mat({{-1, 0}, {0, -2}}));
                  t.expect(p.has_value(), "classify succeeds");
                  if (p) {
                      t.expect(p->type == PhaseType::node, "node");
                      t.expect(p->stability == Stability::stable, "stable");
                      t.expect(p->trace == ri(-3) && p->determinant == ri(2) &&
                                   p->discriminant == ri(1),
                               "T=-3, D=2, delta=1");
                      t.expect(!p->complex_eigenvalues, "real eigenvalues");
                      t.expect(p->lambda1.has_value() && p->lambda1.value() == ri(-2),
                               "lambda1 = -2");
                      t.expect(p->lambda2.has_value() && p->lambda2.value() == ri(-1),
                               "lambda2 = -1");
                      t.expect(p->eigenvalues_rational, "rational eigenvalues");
                  }
              })
        .test("phase_unstable_node",
              [](TestContext& t) {
                  // [[1,0],[0,2]]: T=3, D=2, delta=1 => real 1,2, both positive.
                  auto p = classify_phase_portrait(mat({{1, 0}, {0, 2}}));
                  t.expect(p.has_value(), "classify succeeds");
                  if (p) {
                      t.expect(p->type == PhaseType::node, "node");
                      t.expect(p->stability == Stability::unstable, "unstable");
                      t.expect(p->lambda1.has_value() && p->lambda1.value() == ri(1) &&
                                   p->lambda2.has_value() && p->lambda2.value() == ri(2),
                               "lambda1=1, lambda2=2");
                  }
              })
        .test("phase_unstable_spiral",
              [](TestContext& t) {
                  // [[1,-1],[1,1]]: T=2, D=2, delta=-4 => complex 1 +/- i, Re>0.
                  auto p = classify_phase_portrait(mat({{1, -1}, {1, 1}}));
                  t.expect(p.has_value(), "classify succeeds");
                  if (p) {
                      t.expect(p->type == PhaseType::spiral, "spiral/focus");
                      t.expect(p->stability == Stability::unstable, "unstable");
                      t.expect(p->trace == ri(2) && p->determinant == ri(2) &&
                                   p->discriminant == ri(-4),
                               "T=2, D=2, delta=-4");
                      t.expect(p->real_part.has_value() && p->real_part.value() == ri(1),
                               "real part 1");
                      t.expect(p->imag_part.has_value() && p->imag_part.value() == ri(1),
                               "imag part 1 (1 +/- i)");
                  }
              })
        .test("phase_stable_spiral",
              [](TestContext& t) {
                  // [[-1,-1],[1,-1]]: T=-2, D=2, delta=-4 => complex -1 +/- i, Re<0.
                  auto p = classify_phase_portrait(mat({{-1, -1}, {1, -1}}));
                  t.expect(p.has_value(), "classify succeeds");
                  if (p) {
                      t.expect(p->type == PhaseType::spiral, "spiral/focus");
                      t.expect(p->stability == Stability::stable, "stable");
                      t.expect(p->real_part.has_value() && p->real_part.value() == ri(-1),
                               "real part -1");
                  }
              })
        .test("phase_saddle",
              [](TestContext& t) {
                  // [[1,0],[0,-1]]: T=0, D=-1 => SADDLE (unstable), eigenvalues +/- 1.
                  auto p = classify_phase_portrait(mat({{1, 0}, {0, -1}}));
                  t.expect(p.has_value(), "classify succeeds");
                  if (p) {
                      t.expect(p->type == PhaseType::saddle, "saddle");
                      t.expect(p->stability == Stability::unstable, "unstable");
                      t.expect(p->determinant == ri(-1), "D=-1");
                      t.expect(!p->complex_eigenvalues, "real eigenvalues");
                      t.expect(p->lambda1.has_value() && p->lambda1.value() == ri(-1) &&
                                   p->lambda2.has_value() && p->lambda2.value() == ri(1),
                               "eigenvalues -1 and 1");
                  }
              })
        .test("phase_unstable_star",
              [](TestContext& t) {
                  // [[2,0],[0,2]] = 2I: T=4, D=4, delta=0, scalar => STAR (proper node).
                  auto p = classify_phase_portrait(mat({{2, 0}, {0, 2}}));
                  t.expect(p.has_value(), "classify succeeds");
                  if (p) {
                      t.expect(p->type == PhaseType::star, "star");
                      t.expect(p->stability == Stability::unstable, "unstable");
                      t.expect(p->discriminant == ri(0), "delta=0");
                      t.expect(p->repeated_eigenvalue, "repeated eigenvalue");
                      t.expect(p->lambda1.has_value() && p->lambda1.value() == ri(2) &&
                                   p->lambda2.has_value() && p->lambda2.value() == ri(2),
                               "repeated eigenvalue 2");
                  }
              })
        .test("phase_degenerate_node",
              [](TestContext& t) {
                  // [[1,1],[0,1]]: T=2, D=1, delta=0, defective (not scalar) => improper node.
                  auto p = classify_phase_portrait(mat({{1, 1}, {0, 1}}));
                  t.expect(p.has_value(), "classify succeeds");
                  if (p) {
                      t.expect(p->type == PhaseType::degenerate_node, "degenerate/improper node");
                      t.expect(p->stability == Stability::unstable, "unstable");
                      t.expect(p->discriminant == ri(0), "delta=0");
                  }
              })
        .test("phase_non_isolated",
              [](TestContext& t) {
                  // [[1,0],[0,0]]: D=0 => non-isolated (a line of equilibria); T=1>0 unstable.
                  auto p = classify_phase_portrait(mat({{1, 0}, {0, 0}}));
                  t.expect(p.has_value(), "classify succeeds");
                  if (p) {
                      t.expect(p->type == PhaseType::non_isolated, "non-isolated");
                      t.expect(p->stability == Stability::unstable, "unstable (growing direction)");
                      t.expect(p->determinant == ri(0), "D=0");
                  }
              })
        .test("phase_irrational_node_no_decimal",
              [](TestContext& t) {
                  // [[0,1],[1,1]]: T=1, D=-1 => saddle with delta=5 (not a perfect square).
                  // Eigenvalues (1 +/- sqrt 5)/2 are irrational: they must NOT be filled in.
                  auto p = classify_phase_portrait(mat({{0, 1}, {1, 1}}));
                  t.expect(p.has_value(), "classify succeeds");
                  if (p) {
                      t.expect(p->type == PhaseType::saddle, "saddle");
                      t.expect(p->discriminant == ri(5), "delta=5");
                      t.expect(!p->eigenvalues_rational, "eigenvalues are irrational");
                      t.expect(!p->lambda1.has_value() && !p->lambda2.has_value(),
                               "irrational eigenvalues left symbolic (no decimal)");
                  }
              })
        .test("phase_shape_error",
              [](TestContext& t) {
                  auto e = classify_phase_portrait(mat({{1, 2, 3}, {4, 5, 6}}));
                  t.expect(!e.has_value() && e.error() == MathError::domain_error,
                           "non-2x2 => domain_error");
                  auto e3 = classify_phase_portrait(mat({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}));
                  t.expect(!e3.has_value() && e3.error() == MathError::domain_error,
                           "3x3 => domain_error (phase portrait is 2x2 only)");
              })
        // ---- classify_linear_stability: coarse nD Routh-Hurwitz verdict ---------
        .test("nd_sink_3x3",
              [](TestContext& t) {
                  // diag(-1,-2,-3): char poly lambda^3 + 6 lambda^2 + 11 lambda + 6, Hurwitz.
                  auto c = classify_linear_stability(mat({{-1, 0, 0}, {0, -2, 0}, {0, 0, -3}}));
                  t.expect(c.has_value(), "classify succeeds");
                  if (c) {
                      t.expect(c->verdict == LinearStability::sink, "sink");
                      t.expect(c->asymptotically_stable, "asymptotically stable");
                      t.expect(c->rhp_count == 0, "no RHP eigenvalues");
                      t.expect(c->dimension == 3, "dimension 3");
                  }
              })
        .test("nd_source_3x3",
              [](TestContext& t) {
                  // diag(1,2,3): all eigenvalues Re>0 => source.
                  auto c = classify_linear_stability(mat({{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}));
                  t.expect(c.has_value(), "classify succeeds");
                  if (c) {
                      t.expect(c->verdict == LinearStability::source, "source");
                      t.expect(!c->asymptotically_stable, "not asymptotically stable");
                      t.expect(c->rhp_count == 3, "three RHP eigenvalues");
                  }
              })
        .test("nd_saddle_3x3",
              [](TestContext& t) {
                  // diag(1,-2,-3): one Re>0, two Re<0, none on the axis => saddle.
                  auto c = classify_linear_stability(mat({{1, 0, 0}, {0, -2, 0}, {0, 0, -3}}));
                  t.expect(c.has_value(), "classify succeeds");
                  if (c) {
                      t.expect(c->verdict == LinearStability::saddle, "saddle");
                      t.expect(!c->asymptotically_stable, "not asymptotically stable");
                      t.expect(c->rhp_count == 1, "one RHP eigenvalue");
                  }
              })
        .test("nd_borderline_imaginary_3x3",
              [](TestContext& t) {
                  // Block-diagonal rotation + decay: eigenvalues +/- i and -1. The +/- i pair
                  // lands on the imaginary axis, so the Routh array degenerates => borderline.
                  auto c = classify_linear_stability(mat({{0, -1, 0}, {1, 0, 0}, {0, 0, -1}}));
                  t.expect(c.has_value(), "classify succeeds");
                  if (c) {
                      t.expect(c->verdict == LinearStability::borderline, "borderline");
                      t.expect(!c->asymptotically_stable, "not asymptotically stable");
                      t.expect(c->rhp_count == -1, "RHP count unknown at the boundary");
                  }
              })
        .test("nd_sink_agrees_with_is_asymptotically_stable",
              [](TestContext& t) {
                  // Consistency: sink iff is_asymptotically_stable, on a defective Hurwitz 2x2.
                  auto m = mat({{-1, 1}, {0, -1}});
                  auto c = classify_linear_stability(m);
                  auto s = is_asymptotically_stable(m);
                  t.expect(c.has_value() && s.has_value(), "both succeed");
                  if (c && s) {
                      t.expect((c->verdict == LinearStability::sink) == s.value(),
                               "sink verdict agrees with is_asymptotically_stable");
                  }
              })
        .test("nd_shape_error",
              [](TestContext& t) {
                  auto e = classify_linear_stability(mat({{1, 2, 3}, {4, 5, 6}}));
                  t.expect(!e.has_value() && e.error() == MathError::domain_error,
                           "non-square => domain_error");
              })
        .run();
}
