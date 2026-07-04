// Tests for nimblecas.lie: matrix Lie algebras & Lie transforms over Q (ROADMAP 7.x).
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.matrix;
import nimblecas.ratpoly;
import nimblecas.lie;
import nimblecas.testing;

using nimblecas::adjoint_action_series;
using nimblecas::adjoint_matrix;
using nimblecas::killing_form;
using nimblecas::lie_bracket;
using nimblecas::lie_series;
using nimblecas::MathError;
using nimblecas::Matrix;
using nimblecas::Rational;
using nimblecas::structure_constants;
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

// The standard so(3) ~ su(2) generators L_x, L_y, L_z (real antisymmetric 3x3).
[[nodiscard]] auto so3_x() -> Matrix { return mat({{0, 0, 0}, {0, 0, -1}, {0, 1, 0}}); }
[[nodiscard]] auto so3_y() -> Matrix { return mat({{0, 0, 1}, {0, 0, 0}, {-1, 0, 0}}); }
[[nodiscard]] auto so3_z() -> Matrix { return mat({{0, -1, 0}, {1, 0, 0}, {0, 0, 0}}); }

[[nodiscard]] auto so3_basis() -> std::vector<Matrix> { return {so3_x(), so3_y(), so3_z()}; }

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.lie")
        .test("bracket_basic_and_domain",
              [](TestContext& t) {
                  // [A,A] = 0 for any square A.
                  auto a = mat({{1, 2}, {3, 4}});
                  t.expect(lie_bracket(a, a).value() == Matrix::zero(2, 2), "[A,A] = 0");

                  // Anything commuting with the identity brackets to zero: [A,I] = 0.
                  t.expect(lie_bracket(a, Matrix::identity(2)).value() == Matrix::zero(2, 2),
                           "[A,I] = 0");

                  // Non-square or mismatched-order operands -> domain_error.
                  t.expect(lie_bracket(mat({{1, 2, 3}}), a).error() == MathError::domain_error,
                           "non-square bracket -> domain_error");
                  t.expect(lie_bracket(a, Matrix::identity(3)).error() == MathError::domain_error,
                           "mismatched-order bracket -> domain_error");
              })
        .test("so3_commutation_relations",
              [](TestContext& t) {
                  const auto lx = so3_x();
                  const auto ly = so3_y();
                  const auto lz = so3_z();
                  // Cyclic: [L_x,L_y] = L_z, [L_y,L_z] = L_x, [L_z,L_x] = L_y.
                  t.expect(lie_bracket(lx, ly).value() == lz, "[L_x,L_y] = L_z");
                  t.expect(lie_bracket(ly, lz).value() == lx, "[L_y,L_z] = L_x");
                  t.expect(lie_bracket(lz, lx).value() == ly, "[L_z,L_x] = L_y");
              })
        .test("antisymmetry",
              [](TestContext& t) {
                  auto a = mat({{1, 2}, {3, 4}});
                  auto b = mat({{0, -1}, {5, 2}});
                  auto ab = lie_bracket(a, b).value();
                  auto ba = lie_bracket(b, a).value();
                  // [A,B] = -[B,A]  <=>  [A,B] + [B,A] = 0.
                  t.expect(ab.add(ba).value() == Matrix::zero(2, 2), "[A,B] = -[B,A]");
              })
        .test("bilinearity",
              [](TestContext& t) {
                  // [aA + bB, C] = a[A,C] + b[B,C], with rational scalars a, b.
                  auto a = mat({{1, 0}, {2, -1}});
                  auto b = mat({{0, 3}, {1, 1}});
                  auto c = mat({{2, 1}, {0, 4}});
                  const Rational sa = Rational::make(2, 1).value();
                  const Rational sb = Rational::make(-3, 1).value();

                  auto lhs_arg = a.scale(sa).value().add(b.scale(sb).value()).value();
                  auto lhs = lie_bracket(lhs_arg, c).value();

                  auto rhs = lie_bracket(a, c).value().scale(sa).value()
                                 .add(lie_bracket(b, c).value().scale(sb).value())
                                 .value();
                  t.expect(lhs == rhs, "[aA+bB, C] = a[A,C] + b[B,C]");
              })
        .test("jacobi_identity",
              [](TestContext& t) {
                  // [A,[B,C]] + [B,[C,A]] + [C,[A,B]] = 0 on several fixed small matrices.
                  const std::vector<std::array<Matrix, 3>> triples{
                      {mat({{1, 2}, {3, 4}}), mat({{0, -1}, {2, 1}}), mat({{5, 0}, {1, -2}})},
                      {mat({{2, 0, 1}, {0, 1, 0}, {3, 1, 0}}),
                       mat({{0, 1, 0}, {1, 0, 2}, {0, 0, 1}}),
                       mat({{1, 1, 1}, {0, 2, 0}, {1, 0, 3}})}};
                  for (const auto& [a, b, c] : triples) {
                      auto bc = lie_bracket(b, c).value();
                      auto ca = lie_bracket(c, a).value();
                      auto ab = lie_bracket(a, b).value();
                      auto t1 = lie_bracket(a, bc).value();
                      auto t2 = lie_bracket(b, ca).value();
                      auto t3 = lie_bracket(c, ab).value();
                      auto sum = t1.add(t2).value().add(t3).value();
                      t.expect(sum == Matrix::zero(a.rows(), a.cols()), "Jacobi identity holds");
                  }
              })
        .test("structure_constants_levi_civita",
              [](TestContext& t) {
                  auto sc = structure_constants(so3_basis());
                  t.expect(sc.has_value(), "so(3) basis is closed -> structure constants exist");
                  const auto& c = sc.value();
                  t.expect(c.dimension() == 3, "so(3) has dimension 3");

                  // c^k_ij must equal the Levi-Civita symbol epsilon_{ijk}.
                  auto epsilon = [](std::size_t i, std::size_t j, std::size_t k) -> std::int64_t {
                      if (i == j || j == k || i == k) {
                          return 0;
                      }
                      // Sign of the permutation (i,j,k) of (0,1,2).
                      const bool even = (i == 0 && j == 1 && k == 2) ||
                                        (i == 1 && j == 2 && k == 0) ||
                                        (i == 2 && j == 0 && k == 1);
                      return even ? 1 : -1;
                  };
                  bool all_match = true;
                  for (std::size_t i = 0; i < 3; ++i) {
                      for (std::size_t j = 0; j < 3; ++j) {
                          for (std::size_t k = 0; k < 3; ++k) {
                              if (!(c.at(i, j, k) == ri(epsilon(i, j, k)))) {
                                  all_match = false;
                              }
                          }
                      }
                  }
                  t.expect(all_match, "c^k_ij = epsilon_{ijk} for so(3)");

                  // Spot checks of the named relations.
                  t.expect(c.at(0, 1, 2) == ri(1), "c^z_xy = 1");
                  t.expect(c.at(1, 0, 2) == ri(-1), "c^z_yx = -1 (antisymmetry)");
                  t.expect(c.at(0, 1, 0) == ri(0) && c.at(0, 1, 1) == ri(0),
                           "[L_x,L_y] has no L_x or L_y component");
              })
        .test("structure_constants_not_closed",
              [](TestContext& t) {
                  // {L_x, L_y} alone is NOT closed: [L_x,L_y] = L_z leaves the span.
                  std::vector<Matrix> partial{so3_x(), so3_y()};
                  t.expect(structure_constants(partial).error() == MathError::domain_error,
                           "non-closed basis -> domain_error");
              })
        .test("adjoint_operator",
              [](TestContext& t) {
                  const auto basis = so3_basis();
                  // For so(3) the adjoint representation coincides with the defining rep:
                  // ad_{L_x} (as a matrix in this basis) equals L_x itself, etc.
                  t.expect(adjoint_matrix(so3_x(), basis).value() == so3_x(),
                           "ad_{L_x} = L_x in the so(3) basis");
                  t.expect(adjoint_matrix(so3_y(), basis).value() == so3_y(),
                           "ad_{L_y} = L_y in the so(3) basis");
                  t.expect(adjoint_matrix(so3_z(), basis).value() == so3_z(),
                           "ad_{L_z} = L_z in the so(3) basis");

                  // ad_X acts as a linear operator: (ad_{L_x}) applied to the coordinate
                  // vector of L_y (= e_1) yields the coordinates of [L_x,L_y] = L_z (= e_2).
                  auto ad_x = adjoint_matrix(so3_x(), basis).value();
                  auto e1 = mat({{0}, {1}, {0}});
                  auto image = ad_x.multiply(e1).value();
                  t.expect(image == mat({{0}, {0}, {1}}), "ad_{L_x}(L_y) = L_z in coordinates");
              })
        .test("killing_form_so3",
              [](TestContext& t) {
                  const auto basis = so3_basis();
                  // so(3) is compact simple: its Killing form is K(L_i,L_j) = -2 delta_ij.
                  t.expect(killing_form(so3_x(), so3_x(), basis).value() == ri(-2),
                           "K(L_x,L_x) = -2");
                  t.expect(killing_form(so3_y(), so3_y(), basis).value() == ri(-2),
                           "K(L_y,L_y) = -2");
                  t.expect(killing_form(so3_z(), so3_z(), basis).value() == ri(-2),
                           "K(L_z,L_z) = -2");
                  // Off-diagonal entries vanish.
                  t.expect(killing_form(so3_x(), so3_y(), basis).value() == ri(0),
                           "K(L_x,L_y) = 0");
                  t.expect(killing_form(so3_y(), so3_z(), basis).value() == ri(0),
                           "K(L_y,L_z) = 0");
              })
        .test("lie_series_low_order_transform",
              [](TestContext& t) {
                  // L is nilpotent (L^2 = 0), so ad_L is nilpotent and the Lie series
                  // terminates. With L = [[0,1],[0,0]], f = [[0,0],[1,0]], t = 1:
                  //   f + [L,f] + (1/2)[L,[L,f]] = [[1,-1],[1,-1]]  (third bracket vanishes).
                  auto l = mat({{0, 1}, {0, 0}});
                  auto f = mat({{0, 0}, {1, 0}});
                  auto expected = mat({{1, -1}, {1, -1}});

                  // Order 2 already captures every non-zero term.
                  t.expect(lie_series(l, f, ri(1), 2).value() == expected,
                           "truncated Lie series (order 2) matches hand computation");
                  // Higher order adds only zero terms — the result is unchanged.
                  t.expect(lie_series(l, f, ri(1), 5).value() == expected,
                           "series is stable past the nilpotency index");

                  // Order 0 is just f; order 1 is f + t[L,f].
                  t.expect(lie_series(l, f, ri(1), 0).value() == f, "order 0 Lie series = f");
                  t.expect(lie_series(l, f, ri(1), 1).value() ==
                               f.add(lie_bracket(l, f).value()).value(),
                           "order 1 Lie series = f + [L,f]");
              })
        .test("adjoint_action_equals_conjugation",
              [](TestContext& t) {
                  // Ad_{exp L} f = exp(L) f exp(-L). For nilpotent L (L^2 = 0), exp(L) = I+L
                  // and exp(-L) = I-L exactly, and the truncated adjoint-action series equals
                  // (I+L) f (I-L) once `order` clears ad_L's nilpotency index.
                  auto l = mat({{0, 1}, {0, 0}});
                  auto f = mat({{1, 2}, {3, 4}});
                  auto exp_l = Matrix::identity(2).add(l).value();          // I + L
                  auto exp_neg = Matrix::identity(2).subtract(l).value();   // I - L
                  auto conj = exp_l.multiply(f).value().multiply(exp_neg).value();

                  t.expect(adjoint_action_series(l, f, 6).value() == conj,
                           "Ad_{exp L} f series = exp(L) f exp(-L)");
              })
        .test("domain_errors",
              [](TestContext& t) {
                  auto a = mat({{1, 2}, {3, 4}});
                  // Lie series with mismatched orders / negative order -> domain_error.
                  t.expect(lie_series(a, Matrix::identity(3), ri(1), 2).error() ==
                               MathError::domain_error,
                           "lie_series mismatched order -> domain_error");
                  t.expect(lie_series(a, a, ri(1), -1).error() == MathError::domain_error,
                           "lie_series negative order -> domain_error");
              })
        .run();
}
