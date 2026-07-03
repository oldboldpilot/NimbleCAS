// Tests for nimblecas.vectorcalc: gradient, divergence, curl, Laplacian, Jacobian,
// Hessian, directional and total derivatives.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.vectorcalc;
import nimblecas.testing;

using nimblecas::curl;
using nimblecas::directional_derivative;
using nimblecas::divergence;
using nimblecas::Expr;
using nimblecas::gradient;
using nimblecas::hessian;
using nimblecas::jacobian;
using nimblecas::laplacian;
using nimblecas::MathError;
using nimblecas::simplify;
using nimblecas::total_derivative;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto sym(std::string n) -> Expr { return Expr::symbol(std::move(n)); }
[[nodiscard]] auto I(std::int64_t n) -> Expr { return Expr::integer(n); }
[[nodiscard]] auto S(const Expr& e) -> Expr { return simplify(e).value(); }  // canonical form

}  // namespace

auto main() -> int {
    const Expr x = sym("x");
    const Expr y = sym("y");
    const Expr z = sym("z");

    return TestSuite("nimblecas.vectorcalc")
        .test("gradient",
              [&](TestContext& t) {
                  // f = x^2 y + z  ->  grad = (2xy, x^2, 1).
                  const Expr f = Expr::sum({Expr::product({Expr::power(x, I(2)), y}), z});
                  auto g = gradient(f, {"x", "y", "z"}).value();
                  t.expect(g.size() == 3, "three components");
                  t.expect(g[0] == S(Expr::product({I(2), x, y})), "df/dx = 2xy");
                  t.expect(g[1] == S(Expr::power(x, I(2))), "df/dy = x^2");
                  t.expect(g[2] == I(1), "df/dz = 1");
              })
        .test("divergence_and_laplacian",
              [&](TestContext& t) {
                  // div(x^2, y^2, z^2) = 2x + 2y + 2z.
                  std::vector<Expr> field = {Expr::power(x, I(2)), Expr::power(y, I(2)),
                                             Expr::power(z, I(2))};
                  auto d = divergence(field, {"x", "y", "z"}).value();
                  t.expect(d == S(Expr::sum({Expr::product({I(2), x}), Expr::product({I(2), y}),
                                             Expr::product({I(2), z})})),
                           "divergence is 2x + 2y + 2z");
                  // laplacian(x^2 + y^2 + z^2) = 6.
                  const Expr f = Expr::sum({Expr::power(x, I(2)), Expr::power(y, I(2)),
                                            Expr::power(z, I(2))});
                  t.expect(laplacian(f, {"x", "y", "z"}).value() == I(6), "laplacian is 6");
              })
        .test("curl",
              [&](TestContext& t) {
                  // curl(-y, x, 0) = (0, 0, 2).
                  std::vector<Expr> field = {Expr::product({I(-1), y}), x, I(0)};
                  auto c = curl(field, {"x", "y", "z"}).value();
                  t.expect(c[0] == I(0) && c[1] == I(0), "first two components vanish");
                  t.expect(c[2] == I(2), "third component is 2");
                  // Wrong dimension is a domain error.
                  t.expect(curl({x, y}, {"x", "y"}).error() == MathError::domain_error,
                           "curl requires three dimensions");
              })
        .test("curl_of_gradient_is_zero",
              [&](TestContext& t) {
                  // curl(grad f) = 0 for any scalar field (Clairaut cancellation).
                  const Expr f = Expr::sum({Expr::product({Expr::power(x, I(2)), y}),
                                            Expr::product({Expr::power(y, I(2)), z}),
                                            Expr::product({Expr::power(z, I(2)), x})});
                  auto g = gradient(f, {"x", "y", "z"}).value();
                  auto c = curl(g, {"x", "y", "z"}).value();
                  t.expect(c[0] == I(0) && c[1] == I(0) && c[2] == I(0),
                           "curl(grad f) == (0, 0, 0)");
              })
        .test("div_of_curl_is_zero",
              [&](TestContext& t) {
                  // div(curl F) = 0 for any vector field.
                  std::vector<Expr> field = {Expr::product({Expr::power(x, I(2)), y}),
                                             Expr::product({Expr::power(y, I(2)), z}),
                                             Expr::product({Expr::power(z, I(2)), x})};
                  auto c = curl(field, {"x", "y", "z"}).value();
                  t.expect(divergence(c, {"x", "y", "z"}).value() == I(0),
                           "div(curl F) == 0");
              })
        .test("jacobian_and_hessian",
              [&](TestContext& t) {
                  // J of (xy, yz) wrt (x,y,z) = [[y, x, 0], [0, z, y]].
                  std::vector<Expr> field = {Expr::product({x, y}), Expr::product({y, z})};
                  auto j = jacobian(field, {"x", "y", "z"}).value();
                  t.expect(j.size() == 2 && j[0].size() == 3, "2x3 Jacobian");
                  t.expect(j[0][0] == y && j[0][1] == x && j[0][2] == I(0), "row 1 = (y, x, 0)");
                  t.expect(j[1][0] == I(0) && j[1][1] == z && j[1][2] == y, "row 2 = (0, z, y)");
                  // Hessian of x^2 y = [[2y, 2x], [2x, 0]], symmetric.
                  const Expr f = Expr::product({Expr::power(x, I(2)), y});
                  auto h = hessian(f, {"x", "y"}).value();
                  t.expect(h[0][0] == S(Expr::product({I(2), y})), "H_xx = 2y");
                  t.expect(h[0][1] == S(Expr::product({I(2), x})), "H_xy = 2x");
                  t.expect(h[0][1] == h[1][0], "Hessian is symmetric (Clairaut)");
                  t.expect(h[1][1] == I(0), "H_yy = 0");
              })
        .test("directional_and_total_derivative",
              [&](TestContext& t) {
                  // Directional derivative of x^2 + y^2 along (1, 0) is 2x.
                  const Expr f = Expr::sum({Expr::power(x, I(2)), Expr::power(y, I(2))});
                  auto dd = directional_derivative(f, {"x", "y"}, {I(1), I(0)}).value();
                  t.expect(dd == S(Expr::product({I(2), x})), "directional derivative is 2x");
                  // Total derivative of f = x y with x=x(t), y=y(t): df/dt = y x' + x y'.
                  const Expr u = sym("u");  // x'(t)
                  const Expr v = sym("v");  // y'(t)
                  auto td = total_derivative(Expr::product({x, y}), "t", {"x", "y"}, {u, v})
                                .value();
                  t.expect(td == S(Expr::sum({Expr::product({y, u}), Expr::product({x, v})})),
                           "df/dt = y*x' + x*y'");
                  // Size mismatch is a domain error.
                  t.expect(total_derivative(f, "t", {"x", "y"}, {u}).error() ==
                               MathError::domain_error,
                           "mismatched dependent-variable counts fail");
              })
        .run();
}
