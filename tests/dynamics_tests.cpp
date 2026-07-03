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
using nimblecas::fixed_point_affine;
using nimblecas::is_asymptotically_stable;
using nimblecas::MathError;
using nimblecas::Matrix;
using nimblecas::Rational;
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
        .run();
}
