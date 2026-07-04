// Tests for nimblecas.tensor: metric determinant/inverse, Christoffel symbols, Riemann/Ricci
// curvature, scalar curvature, Einstein tensor, and the curvilinear Laplace-Beltrami operator.
// @author Olumuyiwa Oluwasanmi
//
// Every expected value is hand-verified in the comments. The two hallmark checks are the
// flat metric (all curvature vanishes) and the round 2-sphere (scalar curvature R = 2/a^2).

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.diff;
import nimblecas.simplify;
import nimblecas.tensor;
import nimblecas.testing;

using nimblecas::christoffel;
using nimblecas::covariant_derivative_vector;
using nimblecas::differentiate;
using nimblecas::einstein_tensor;
using nimblecas::Expr;
using nimblecas::ExprMatrix;
using nimblecas::inverse_metric;
using nimblecas::laplace_beltrami;
using nimblecas::make_metric;
using nimblecas::MathError;
using nimblecas::metric_determinant;
using nimblecas::metric_gradient;
using nimblecas::ricci_tensor;
using nimblecas::riemann_tensor;
using nimblecas::scalar_curvature;
using nimblecas::simplify;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto sym(std::string n) -> Expr { return Expr::symbol(std::move(n)); }
[[nodiscard]] auto I(std::int64_t n) -> Expr { return Expr::integer(n); }

// Canonicalise to the same fixed point the module drives its results to, so structural
// comparison (is_equivalent_to via operator==) is meaningful.
[[nodiscard]] auto S(const Expr& e) -> Expr {
    Expr current = e;
    for (int pass = 0; pass < 8; ++pass) {
        Expr next = simplify(current).value();
        if (next == current) {
            return next;
        }
        current = next;
    }
    return current;
}

[[nodiscard]] auto fn(std::string name, const Expr& a) -> Expr {
    return Expr::apply(std::move(name), {a});
}

// An n x n Kronecker-delta (identity) component matrix.
[[nodiscard]] auto identity_components(std::size_t n) -> ExprMatrix {
    ExprMatrix g(n, std::vector<Expr>(n, I(0)));
    for (std::size_t i = 0; i < n; ++i) {
        g[i][i] = I(1);
    }
    return g;
}

}  // namespace

auto main() -> int {
    const Expr a = sym("a");
    const Expr r = sym("r");
    const Expr theta = sym("theta");
    const Expr sin_t = fn("sin", theta);
    const Expr cos_t = fn("cos", theta);

    return TestSuite("nimblecas.tensor")
        .test("flat_metric_is_curvature_free",
              [&](TestContext& t) {
                  // Euclidean 3-space in Cartesian coordinates: g = I_3.
                  auto m = make_metric({"x", "y", "z"}, identity_components(3)).value();
                  t.expect(metric_determinant(m).value() == I(1), "det(I) = 1");

                  // All Christoffel symbols vanish (metric components are constant).
                  auto gamma = christoffel(m).value();
                  bool gamma_zero = true;
                  for (const auto& gk : gamma)
                      for (const auto& gki : gk)
                          for (const auto& e : gki) gamma_zero = gamma_zero && (e == I(0));
                  t.expect(gamma_zero, "all Gamma^k_{ij} = 0 for flat space");

                  // Riemann, Ricci, scalar curvature all vanish.
                  auto riem = riemann_tensor(m).value();
                  bool riem_zero = true;
                  for (const auto& r0 : riem)
                      for (const auto& r1 : r0)
                          for (const auto& r2 : r1)
                              for (const auto& e : r2) riem_zero = riem_zero && (e == I(0));
                  t.expect(riem_zero, "Riemann tensor is identically zero");

                  auto ric = ricci_tensor(m).value();
                  bool ricci_zero = true;
                  for (const auto& row : ric)
                      for (const auto& e : row) ricci_zero = ricci_zero && (e == I(0));
                  t.expect(ricci_zero, "Ricci tensor is identically zero");

                  t.expect(scalar_curvature(m).value() == I(0), "scalar curvature is zero");
              })
        .test("sphere_christoffel_symbols",
              [&](TestContext& t) {
                  // Round 2-sphere of radius a: coords (theta, phi),
                  // g = diag(a^2, a^2 sin(theta)^2).
                  ExprMatrix g = {{Expr::power(a, I(2)), I(0)},
                                  {I(0), Expr::product({Expr::power(a, I(2)),
                                                        Expr::power(sin_t, I(2))})}};
                  auto m = make_metric({"theta", "phi"}, g).value();
                  auto gamma = christoffel(m).value();  // gamma[k][i][j]

                  // Gamma^theta_{phi phi} = -sin(theta) cos(theta).
                  t.expect(gamma[0][1][1] == S(Expr::product({I(-1), sin_t, cos_t})),
                           "Gamma^theta_{phi phi} = -sin(theta)cos(theta)");

                  // Gamma^phi_{theta phi} = cot(theta) = cos(theta)/sin(theta).
                  const Expr cot = Expr::product({cos_t, Expr::power(sin_t, I(-1))});
                  t.expect(gamma[1][0][1] == S(cot),
                           "Gamma^phi_{theta phi} = cos(theta)/sin(theta)");
                  t.expect(gamma[1][1][0] == S(cot), "Christoffel symmetric in lower indices");

                  // Gamma^theta_{theta theta} = 0 (theta-theta block is a-only).
                  t.expect(gamma[0][0][0] == I(0), "Gamma^theta_{theta theta} = 0");
              })
        .test("sphere_curvature_two_over_a_squared",
              [&](TestContext& t) {
                  ExprMatrix g = {{Expr::power(a, I(2)), I(0)},
                                  {I(0), Expr::product({Expr::power(a, I(2)),
                                                        Expr::power(sin_t, I(2))})}};
                  auto m = make_metric({"theta", "phi"}, g).value();

                  // det(g) = a^4 sin(theta)^2.
                  t.expect(metric_determinant(m).value() ==
                               S(Expr::product({Expr::power(a, I(4)), Expr::power(sin_t, I(2))})),
                           "det(g) = a^4 sin(theta)^2");

                  // Inverse metric g^{phi phi} = a^{-2} sin(theta)^{-2}.
                  auto ginv = inverse_metric(m).value();
                  t.expect(ginv[1][1] ==
                               S(Expr::product({Expr::power(a, I(-2)), Expr::power(sin_t, I(-2))})),
                           "g^{phi phi} = 1/(a^2 sin^2)");

                  // Ricci tensor R_{theta theta} = 1 for the 2-sphere.
                  auto ric = ricci_tensor(m).value();
                  t.expect(ric[0][0] == I(1), "R_{theta theta} = 1");
                  // R_{phi phi} = sin(theta)^2.
                  t.expect(ric[1][1] == S(Expr::power(sin_t, I(2))), "R_{phi phi} = sin(theta)^2");

                  // Scalar curvature R = 2/a^2 (the hallmark result).
                  const Expr expected_R = Expr::product({I(2), Expr::power(a, I(-2))});
                  t.expect(scalar_curvature(m).value().is_equivalent_to(S(expected_R)),
                           "scalar curvature R = 2/a^2");

                  // In two dimensions the Einstein tensor vanishes identically.
                  auto ein = einstein_tensor(m).value();
                  bool einstein_zero = true;
                  for (const auto& row : ein)
                      for (const auto& e : row) einstein_zero = einstein_zero && (e == I(0));
                  t.expect(einstein_zero, "Einstein tensor G_{mu nu} = 0 in 2D");
              })
        .test("polar_laplace_beltrami_reproduces_polar_laplacian",
              [&](TestContext& t) {
                  // Flat plane in polar coordinates (r, theta): g = diag(1, r^2).
                  ExprMatrix g = {{I(1), I(0)}, {I(0), Expr::power(r, I(2))}};
                  auto m = make_metric({"r", "theta"}, g).value();

                  // Test function f = r^2 cos(theta).
                  const Expr f = Expr::product({Expr::power(r, I(2)), cos_t});

                  // Hand computation of the polar Laplacian f_rr + (1/r) f_r + (1/r^2) f_thetatheta:
                  //   f_r = 2r cos,  f_rr = 2 cos,  f_theta = -r^2 sin,  f_thetatheta = -r^2 cos
                  //   => 2cos + (1/r)(2r cos) + (1/r^2)(-r^2 cos) = 2cos + 2cos - cos = 3 cos(theta).
                  const Expr expected = Expr::product({I(3), cos_t});
                  t.expect(laplace_beltrami(m, f).value() == S(expected),
                           "Laplace-Beltrami of r^2 cos(theta) is 3 cos(theta)");

                  // Cross-check against the polar Laplacian assembled directly from partials.
                  auto d = [](const Expr& e, std::string_view v) {
                      return differentiate(e, v).value();
                  };
                  const Expr f_r = d(f, "r");
                  const Expr f_rr = d(f_r, "r");
                  const Expr f_tt = d(d(f, "theta"), "theta");
                  const Expr polar = S(Expr::sum(
                      {f_rr, Expr::product({Expr::power(r, I(-1)), f_r}),
                       Expr::product({Expr::power(r, I(-2)), f_tt})}));
                  t.expect(laplace_beltrami(m, f).value() == polar,
                           "matches f_rr + (1/r) f_r + (1/r^2) f_thetatheta");

                  // Sanity: raised gradient grad^theta f = (1/r^2) d_theta f = -sin(theta).
                  auto grad = metric_gradient(m, f).value();
                  t.expect(grad[1] == S(Expr::product({I(-1), sin_t})),
                           "grad^theta f = -sin(theta)");
              })
        .test("error_and_boundary_cases",
              [&](TestContext& t) {
                  // Non-square component matrix is a domain error.
                  t.expect(make_metric({"x", "y"}, {{I(1), I(0)}}).error() ==
                               MathError::domain_error,
                           "non-square metric rejected");

                  // Singular metric (det = 0) has no inverse.
                  auto singular = make_metric({"u", "v"}, {{I(1), I(1)}, {I(1), I(1)}}).value();
                  t.expect(inverse_metric(singular).error() == MathError::domain_error,
                           "singular metric -> domain_error");
                  t.expect(scalar_curvature(singular).error() == MathError::domain_error,
                           "curvature of singular metric fails on the railway");

                  // Dimension beyond the cofactor cap (n > 5) is not implemented.
                  auto big = make_metric({"a", "b", "c", "d", "e", "f"},
                                         identity_components(6))
                                 .value();
                  t.expect(inverse_metric(big).error() == MathError::not_implemented,
                           "n > 5 -> not_implemented");
                  t.expect(metric_determinant(big).error() == MathError::not_implemented,
                           "det for n > 5 -> not_implemented");

                  // Vector-length mismatch in the covariant derivative is a domain error.
                  auto flat2 = make_metric({"x", "y"}, identity_components(2)).value();
                  t.expect(covariant_derivative_vector(flat2, {I(1)}).error() ==
                               MathError::domain_error,
                           "covariant derivative rejects wrong-length field");
              })
        .run();
}
