// Tests for nimblecas.forms: differential forms / exterior calculus — wedge product,
// exterior derivative (with d^2 = 0), Euclidean Hodge star, interior product, and the
// closedness/exactness predicates.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.forms;
import nimblecas.testing;

using nimblecas::DifferentialForm;
using nimblecas::exterior_derivative;
using nimblecas::Expr;
using nimblecas::hodge_star;
using nimblecas::hodge_star_euclidean;
using nimblecas::interior_product;
using nimblecas::is_closed;
using nimblecas::is_exact;
using nimblecas::MathError;
using nimblecas::simplify;
using nimblecas::wedge;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto sym(std::string n) -> Expr { return Expr::symbol(std::move(n)); }
[[nodiscard]] auto I(std::int64_t n) -> Expr { return Expr::integer(n); }
[[nodiscard]] auto S(const Expr& e) -> Expr { return simplify(e).value(); }  // canonical form

using Coords = std::vector<std::string>;
using Idx = std::vector<std::size_t>;
using Term = std::pair<Idx, Expr>;

// Convenience: build a single basis component c * dx_{indices}.
[[nodiscard]] auto basis(const Coords& c, Idx indices, Expr coeff) -> DifferentialForm {
    return DifferentialForm::basis(c, std::move(indices), std::move(coeff)).value();
}

}  // namespace

auto main() -> int {
    const Coords coords{"x", "y", "z"};  // index 0=x, 1=y, 2=z
    const Expr x = sym("x");
    const Expr y = sym("y");
    const Expr z = sym("z");

    // Identity (Euclidean) metric.
    const std::vector<std::vector<Expr>> euclid{
        {I(1), I(0), I(0)}, {I(0), I(1), I(0)}, {I(0), I(0), I(1)}};

    return TestSuite("nimblecas.forms")
        .test("wedge_antisymmetry",
              [&](TestContext& t) {
                  const DifferentialForm dx = basis(coords, {0}, I(1));
                  const DifferentialForm dy = basis(coords, {1}, I(1));
                  // dx ^ dy = dx^dy ; dy ^ dx = -dx^dy.
                  t.expect(wedge(dx, dy).value() == basis(coords, {0, 1}, I(1)),
                           "dx ^ dy = dx^dy");
                  t.expect(wedge(dy, dx).value() == basis(coords, {0, 1}, I(-1)),
                           "dy ^ dx = -dx^dy");
                  // dx ^ dx = 0.
                  t.expect(wedge(dx, dx).value().is_zero(), "dx ^ dx = 0");
              })
        .test("wedge_graded_anticommutativity",
              [&](TestContext& t) {
                  // a is a 1-form, b is a 2-form: a ^ b = (-1)^{1*2} b ^ a = b ^ a.
                  const DifferentialForm a = basis(coords, {0}, I(1));       // dx
                  const DifferentialForm b = basis(coords, {1, 2}, I(1));    // dy^dz
                  const auto ab = wedge(a, b).value();
                  const auto ba = wedge(b, a).value();
                  t.expect(ab == ba, "1-form ^ 2-form commutes ((-1)^2 = 1)");
                  t.expect(ab == basis(coords, {0, 1, 2}, I(1)), "dx ^ dy^dz = dx^dy^dz");
                  // dimension mismatch is a domain error.
                  const DifferentialForm other =
                      DifferentialForm::basis(Coords{"x", "y"}, {0}, I(1)).value();
                  t.expect(wedge(a, other).error() == MathError::domain_error,
                           "wedge of forms over different coordinates fails");
              })
        .test("wedge_degree_overflow_is_zero",
              [&](TestContext& t) {
                  // 1-form ^ 3-form over 3 coordinates -> degree 4 > n -> zero form.
                  const DifferentialForm a = basis(coords, {0}, I(1));
                  const DifferentialForm b = basis(coords, {0, 1, 2}, I(1));
                  const auto w = wedge(a, b).value();
                  t.expect(w.is_zero(), "degree-4 form is identically zero");
                  t.expect(w.degree() == 4, "result degree is p + q = 4");
              })
        .test("exterior_derivative_of_scalar_is_gradient",
              [&](TestContext& t) {
                  // f = x^2 y + z  ->  df = 2xy dx + x^2 dy + 1 dz.
                  const Expr f = Expr::sum({Expr::product({Expr::power(x, I(2)), y}), z});
                  const auto f0 = DifferentialForm::scalar(coords, f).value();
                  const auto df = exterior_derivative(f0).value();
                  t.expect(df.degree() == 1, "df is a 1-form");
                  t.expect(df.component({0}).value() == S(Expr::product({I(2), x, y})),
                           "df/dx = 2xy");
                  t.expect(df.component({1}).value() == S(Expr::power(x, I(2))), "df/dy = x^2");
                  t.expect(df.component({2}).value() == I(1), "df/dz = 1");
              })
        .test("exterior_derivative_squared_is_zero",
              [&](TestContext& t) {
                  // A generic 1-form with polynomial coefficients: d(d w) = 0.
                  const Expr P = Expr::product({Expr::power(x, I(2)), y});
                  const Expr Q = Expr::product({y, z});
                  const Expr R = Expr::power(z, I(3));
                  std::vector<Term> terms;
                  terms.push_back({Idx{0}, P});
                  terms.push_back({Idx{1}, Q});
                  terms.push_back({Idx{2}, R});
                  const auto w = DifferentialForm::from_components(coords, 1, std::move(terms)).value();
                  const auto dw = exterior_derivative(w).value();
                  t.expect(!dw.is_zero(), "dw is a non-trivial 2-form");
                  const auto ddw = exterior_derivative(dw).value();
                  t.expect(ddw.is_zero(), "d(d w) = 0");
              })
        .test("hodge_star_euclidean_identities",
              [&](TestContext& t) {
                  const DifferentialForm dx = basis(coords, {0}, I(1));
                  const DifferentialForm dy = basis(coords, {1}, I(1));
                  const DifferentialForm dxdy = basis(coords, {0, 1}, I(1));
                  // *(dx) = dy^dz.
                  t.expect(hodge_star(dx, euclid).value() == basis(coords, {1, 2}, I(1)),
                           "*(dx) = dy^dz");
                  // *(dy) = dz^dx = -dx^dz.
                  t.expect(hodge_star(dy, euclid).value() == basis(coords, {0, 2}, I(-1)),
                           "*(dy) = -dx^dz");
                  // *(dx^dy) = dz.
                  t.expect(hodge_star(dxdy, euclid).value() == basis(coords, {2}, I(1)),
                           "*(dx^dy) = dz");
              })
        .test("hodge_double_star_sign",
              [&](TestContext& t) {
                  // In R^3 on a 1-form (p=1): **w = (-1)^{p(n-p)} w = (-1)^2 w = w.
                  const DifferentialForm dx = basis(coords, {0}, I(1));
                  const auto once = hodge_star_euclidean(dx).value();
                  const auto twice = hodge_star_euclidean(once).value();
                  t.expect(twice == dx, "**(dx) = dx");
              })
        .test("hodge_star_general_metric_not_implemented",
              [&](TestContext& t) {
                  const DifferentialForm dx = basis(coords, {0}, I(1));
                  // A non-identity (here, scaled) metric is honestly rejected.
                  std::vector<std::vector<Expr>> scaled = euclid;
                  scaled[0][0] = I(2);
                  t.expect(hodge_star(dx, scaled).error() == MathError::not_implemented,
                           "non-Euclidean metric returns not_implemented, never a wrong dual");
                  // Wrong-sized metric is a domain error.
                  std::vector<std::vector<Expr>> small{{I(1), I(0)}, {I(0), I(1)}};
                  t.expect(hodge_star(dx, small).error() == MathError::domain_error,
                           "metric of wrong dimension fails");
              })
        .test("interior_product",
              [&](TestContext& t) {
                  // ι_V(dx^dy) with V = (x, y, z) = V^0 dy - V^1 dx = x dy - y dx.
                  const DifferentialForm dxdy = basis(coords, {0, 1}, I(1));
                  const auto iv = interior_product(dxdy, {x, y, z}).value();
                  t.expect(iv.degree() == 1, "contraction lowers degree to 1");
                  t.expect(iv.component({0}).value() == S(Expr::product({I(-1), y})),
                           "dx component is -y");
                  t.expect(iv.component({1}).value() == x, "dy component is x");
                  // Wrong-length vector field is a domain error.
                  t.expect(interior_product(dxdy, {x, y}).error() == MathError::domain_error,
                           "vector field of wrong length fails");
              })
        .test("is_closed_and_is_exact",
              [&](TestContext& t) {
                  // d(anything) is closed: d(df) = 0.
                  const Expr f = Expr::product({x, y, z});
                  const auto df = exterior_derivative(DifferentialForm::scalar(coords, f).value())
                                      .value();
                  t.expect(is_closed(df).value(), "d f is closed");
                  // x dy is not closed: d(x dy) = dx^dy != 0.
                  const DifferentialForm w_open = basis(coords, {1}, x);
                  t.expect(!is_closed(w_open).value(), "x dy is not closed");
                  // is_exact is sound-but-partial: true only for the zero form.
                  const DifferentialForm zero1 = DifferentialForm::zero(coords, 1);
                  t.expect(is_exact(zero1).value(), "the zero form is exact");
                  t.expect(is_exact(w_open).error() == MathError::not_implemented,
                           "general exactness is honestly not decided");
              })
        .test("component_canonicalization_and_bounds",
              [&](TestContext& t) {
                  const DifferentialForm w = basis(coords, {0, 1}, x);  // x dx^dy
                  t.expect(w.component({0, 1}).value() == x, "component(0,1) = x");
                  t.expect(w.component({1, 0}).value() == S(Expr::product({I(-1), x})),
                           "component(1,0) = -x (antisymmetry)");
                  t.expect(w.component({0, 0}).value() == I(0), "repeated index gives 0");
                  t.expect(w.component({0, 3}).error() == MathError::domain_error,
                           "index out of range is a domain error");
                  t.expect(w.component({0}).error() == MathError::domain_error,
                           "wrong-length index tuple is a domain error");
              })
        .run();
}
