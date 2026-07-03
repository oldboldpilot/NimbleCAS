// Tests for nimblecas.eigen: characteristic polynomial, rational eigenvalues, and
// eigenspace bases.
// @author Olumuyiwa Oluwasanmi
//
// Every test uses exact integer/rational matrices so the characteristic polynomial,
// its rational roots, and the eigenvectors are all deterministic. Eigenvector checks
// verify the defining property (A - lambda*I)*v = 0 (equivalently A*v = lambda*v) with
// exact Matrix arithmetic, since a null-space basis vector is only unique up to scaling.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.eigen;
import nimblecas.testing;

using nimblecas::Matrix;
using nimblecas::Rational;
using nimblecas::characteristic_polynomial;
using nimblecas::eigenvectors_for;
using nimblecas::rational_eigenvalues;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Rational from a plain integer.
auto r(std::int64_t v) -> Rational { return Rational::from_int(v); }

// Build a matrix from integer rows, recording a failure and returning a 0x0 matrix on the
// (unexpected) error branch so a later comparison also fails loudly.
auto mat(TestContext& t, std::vector<std::vector<std::int64_t>> rows, std::string_view what)
    -> Matrix {
    std::vector<std::vector<Rational>> converted;
    converted.reserve(rows.size());
    for (const auto& row : rows) {
        std::vector<Rational> out;
        out.reserve(row.size());
        for (std::int64_t v : row) {
            out.push_back(r(v));
        }
        converted.push_back(std::move(out));
    }
    auto m = Matrix::from_rows(std::move(converted));
    if (!m) {
        t.expect(false, std::format("{}: unexpected from_rows error", what));
        return Matrix::zero(0, 0);
    }
    return *m;
}

// A*v == lambda*v, checked exactly by treating v as an n x 1 column matrix.
auto verify_eigenpair(TestContext& t, const Matrix& a, const Rational& lambda,
                      const std::vector<Rational>& v, std::string_view what) -> void {
    std::vector<std::vector<Rational>> col;
    col.reserve(v.size());
    for (const auto& e : v) {
        col.push_back(std::vector<Rational>{e});
    }
    auto cm = Matrix::from_rows(std::move(col));
    if (!cm) {
        t.expect(false, std::format("{}: build column failed", what));
        return;
    }
    auto av = a.multiply(*cm);
    if (!av) {
        t.expect(false, std::format("{}: A*v failed", what));
        return;
    }
    auto lv = cm->scale(lambda);
    if (!lv) {
        t.expect(false, std::format("{}: lambda*v failed", what));
        return;
    }
    t.expect(*av == *lv, std::format("{}: A*v == lambda*v", what));
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.eigen")
        .test("char_poly_diagonal_2x2",
              [&](TestContext& t) {
                  // [[2,0],[0,3]] -> (lambda-2)(lambda-3) = lambda^2 - 5*lambda + 6.
                  auto a = mat(t, {{2, 0}, {0, 3}}, "diag(2,3)");
                  auto p = characteristic_polynomial(a);
                  t.expect(p.has_value(), "characteristic_polynomial succeeds");
                  if (!p) {
                      return;
                  }
                  t.expect_eq(p->degree(), std::int64_t{2}, "degree 2");
                  t.expect(p->coefficient(0) == r(6), "c_0 = 6 (det)");
                  t.expect(p->coefficient(1) == r(-5), "c_1 = -5 (-trace)");
                  t.expect(p->coefficient(2) == r(1), "c_2 = 1 (monic)");
              })
        .test("char_poly_general_2x2_trace_and_det",
              [&](TestContext& t) {
                  // [[1,2],[3,4]]: trace = 5, det = -2, so c_1 = -trace = -5, c_0 = det = -2.
                  auto a = mat(t, {{1, 2}, {3, 4}}, "[[1,2],[3,4]]");
                  auto p = characteristic_polynomial(a);
                  t.expect(p.has_value(), "characteristic_polynomial succeeds");
                  if (!p) {
                      return;
                  }
                  t.expect(p->coefficient(2) == r(1), "monic leading coefficient");
                  t.expect(p->coefficient(1) == r(-5), "c_{n-1} = -trace = -5");
                  t.expect(p->coefficient(0) == r(-2), "c_0 = det = -2");
              })
        .test("eigenvalues_diagonal_are_diagonal_entries",
              [&](TestContext& t) {
                  // diag(2,3): eigenvalues 2 and 3, each multiplicity 1.
                  auto a = mat(t, {{2, 0}, {0, 3}}, "diag(2,3)");
                  auto evs = rational_eigenvalues(a);
                  t.expect(evs.has_value(), "rational_eigenvalues succeeds");
                  if (!evs) {
                      return;
                  }
                  t.expect_eq(evs->size(), std::size_t{2}, "two distinct eigenvalues");
                  bool has2 = false;
                  bool has3 = false;
                  for (const auto& [val, mult] : *evs) {
                      if (val == r(2)) {
                          has2 = true;
                          t.expect_eq(mult, std::int64_t{1}, "eigenvalue 2 has multiplicity 1");
                      }
                      if (val == r(3)) {
                          has3 = true;
                          t.expect_eq(mult, std::int64_t{1}, "eigenvalue 3 has multiplicity 1");
                      }
                  }
                  t.expect(has2 && has3, "eigenvalues are exactly {2, 3}");
              })
        .test("eigenvalue_repeated_multiplicity",
              [&](TestContext& t) {
                  // [[2,1],[0,2]] is a Jordan block: eigenvalue 2 with multiplicity 2.
                  auto a = mat(t, {{2, 1}, {0, 2}}, "jordan(2)");
                  auto evs = rational_eigenvalues(a);
                  t.expect(evs.has_value(), "rational_eigenvalues succeeds");
                  if (!evs) {
                      return;
                  }
                  t.expect_eq(evs->size(), std::size_t{1}, "single distinct eigenvalue");
                  if (!evs->empty()) {
                      t.expect((*evs)[0].first == r(2), "eigenvalue is 2");
                      t.expect_eq((*evs)[0].second, std::int64_t{2}, "multiplicity 2");
                  }
              })
        .test("eigenvectors_axis_vectors_for_diagonal",
              [&](TestContext& t) {
                  // diag(2,3): eigenspace of 2 is the x-axis, of 3 is the y-axis. Basis
                  // vectors may be scaled, so check the eigenpair property directly.
                  auto a = mat(t, {{2, 0}, {0, 3}}, "diag(2,3)");

                  auto v2 = eigenvectors_for(a, r(2));
                  t.expect(v2.has_value(), "eigenvectors_for(2) succeeds");
                  if (v2) {
                      t.expect_eq(v2->size(), std::size_t{1}, "1-dimensional eigenspace for 2");
                      if (!v2->empty()) {
                          t.expect_eq((*v2)[0].size(), std::size_t{2}, "vector has length n=2");
                          verify_eigenpair(t, a, r(2), (*v2)[0], "eigenvector of 2");
                          // Lives on the x-axis: second component is 0.
                          t.expect((*v2)[0][1] == r(0), "eigenvector of 2 is along x-axis");
                      }
                  }

                  auto v3 = eigenvectors_for(a, r(3));
                  t.expect(v3.has_value(), "eigenvectors_for(3) succeeds");
                  if (v3) {
                      t.expect_eq(v3->size(), std::size_t{1}, "1-dimensional eigenspace for 3");
                      if (!v3->empty()) {
                          verify_eigenpair(t, a, r(3), (*v3)[0], "eigenvector of 3");
                          // Lives on the y-axis: first component is 0.
                          t.expect((*v3)[0][0] == r(0), "eigenvector of 3 is along y-axis");
                      }
                  }
              })
        .test("eigenvectors_nullspace_property_3x3",
              [&](TestContext& t) {
                  // A symmetric integer matrix with rational eigenvalues. diag-plus-off:
                  // [[3,0,0],[0,5,0],[0,0,5]] has eigenvalue 5 with a 2-dimensional
                  // eigenspace; verify every returned basis vector satisfies A*v = 5*v.
                  auto a = mat(t, {{3, 0, 0}, {0, 5, 0}, {0, 0, 5}}, "diag(3,5,5)");
                  auto vecs = eigenvectors_for(a, r(5));
                  t.expect(vecs.has_value(), "eigenvectors_for(5) succeeds");
                  if (!vecs) {
                      return;
                  }
                  t.expect_eq(vecs->size(), std::size_t{2}, "eigenvalue 5 is 2-dimensional");
                  for (std::size_t i = 0; i < vecs->size(); ++i) {
                      t.expect_eq((*vecs)[i].size(), std::size_t{3}, "each vector has length 3");
                      verify_eigenpair(t, a, r(5), (*vecs)[i],
                                       std::format("basis vector {} of eigenvalue 5", i));
                  }

                  // The eigenvalue 3 is 1-dimensional (the x-axis).
                  auto v3 = eigenvectors_for(a, r(3));
                  t.expect(v3.has_value(), "eigenvectors_for(3) succeeds");
                  if (v3 && !v3->empty()) {
                      t.expect_eq(v3->size(), std::size_t{1}, "eigenvalue 3 is 1-dimensional");
                      verify_eigenpair(t, a, r(3), (*v3)[0], "eigenvector of 3");
                  }
              })
        .test("non_eigenvalue_has_trivial_kernel",
              [&](TestContext& t) {
                  // 4 is not an eigenvalue of diag(2,3): (A - 4I) is invertible, so the
                  // kernel is trivial and no basis vectors are returned.
                  auto a = mat(t, {{2, 0}, {0, 3}}, "diag(2,3)");
                  auto vecs = eigenvectors_for(a, r(4));
                  t.expect(vecs.has_value(), "eigenvectors_for(4) succeeds");
                  if (vecs) {
                      t.expect(vecs->empty(), "no eigenvectors for a non-eigenvalue");
                  }
              })
        .test("non_square_is_domain_error",
              [&](TestContext& t) {
                  // A 2x3 matrix has no characteristic polynomial.
                  auto a = mat(t, {{1, 2, 3}, {4, 5, 6}}, "2x3");
                  auto p = characteristic_polynomial(a);
                  t.expect(!p.has_value(), "characteristic_polynomial rejects non-square");
                  t.expect(!p && p.error() == nimblecas::MathError::domain_error,
                           "non-square yields domain_error");

                  auto evs = rational_eigenvalues(a);
                  t.expect(!evs.has_value(), "rational_eigenvalues rejects non-square");
                  t.expect(!evs && evs.error() == nimblecas::MathError::domain_error,
                           "rational_eigenvalues propagates domain_error");
              })
        .run();
}
