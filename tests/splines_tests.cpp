// Tests for nimblecas.splines: exact cubic splines, Hermite/PCHIP, Bezier, B-splines,
// and rational-weight NURBS over Q.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.splines;
import nimblecas.testing;

using nimblecas::BezierCurve;
using nimblecas::BSpline;
using nimblecas::CubicSpline;
using nimblecas::HermiteSpline;
using nimblecas::MathError;
using nimblecas::NurbsCurve;
using nimblecas::Point2;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}
[[nodiscard]] auto ri(std::int64_t v) -> Rational { return Rational::from_int(v); }

[[nodiscard]] auto rats(std::vector<std::int64_t> xs) -> std::vector<Rational> {
    std::vector<Rational> r;
    r.reserve(xs.size());
    for (std::int64_t v : xs) {
        r.push_back(Rational::from_int(v));
    }
    return r;
}

// Horner evaluation of a RationalPoly at a rational point (test-local).
[[nodiscard]] auto peval(const RationalPoly& p, const Rational& x) -> Rational {
    const std::span<const Rational> c = p.coefficients();
    Rational acc{};
    for (std::size_t k = c.size(); k-- > 0;) {
        acc = acc.multiply(x).value().add(c[k]).value();
    }
    return acc;
}

// Second derivative of a RationalPoly (test-local).
[[nodiscard]] auto psecond(const RationalPoly& p) -> RationalPoly {
    return p.derivative().value().derivative().value();
}
[[nodiscard]] auto pfirst(const RationalPoly& p) -> RationalPoly {
    return p.derivative().value();
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.splines")
        .test("natural_spline_through_collinear_is_the_line",
              [](TestContext& t) {
                  // Points on y = 3x - 1: natural cubic spline is that line exactly.
                  auto xs = rats({0, 1, 2, 3});
                  auto ys = rats({-1, 2, 5, 8});
                  auto sp = CubicSpline::natural(xs, ys).value();
                  // All moments zero (S'' == 0 everywhere for a line).
                  for (const auto& m : sp.moments()) {
                      t.expect(m == ri(0), "natural line has zero moments");
                  }
                  // Exact line values at several probe points.
                  const std::vector<Rational> probes{ri(0), rat(1, 2), rat(3, 2), rat(5, 2), ri(3)};
                  for (const auto& x : probes) {
                      auto expected = x.multiply(ri(3)).value().subtract(ri(1)).value();  // 3x - 1
                      t.expect(sp.evaluate(x).value() == expected, "spline == line at probe");
                  }
                  // Each piece is literally the line 3x - 1.
                  for (std::size_t i = 0; i < sp.piece_count(); ++i) {
                      t.expect(sp.piece(i).is_equal(RationalPoly::from_coeffs({ri(-1), ri(3)})),
                               "each natural piece is the line");
                  }
              })
        .test("clamped_spline_reproduces_a_cubic_and_is_C2",
              [](TestContext& t) {
                  // q(x) = 2x^3 - x + 5; sample at knots and clamp with the true end slopes.
                  const RationalPoly q = RationalPoly::from_coeffs({ri(5), ri(-1), ri(0), ri(2)});
                  auto xs = rats({-1, 0, 1, 2});
                  std::vector<Rational> ys;
                  for (const auto& x : xs) {
                      ys.push_back(peval(q, x));
                  }
                  const RationalPoly qp = pfirst(q);   // 6x^2 - 1
                  const RationalPoly qpp = psecond(q); // 12x
                  auto sp = CubicSpline::clamped(xs, ys, peval(qp, xs.front()),
                                                 peval(qp, xs.back()))
                                .value();
                  // Every piece equals q exactly (a cubic is its own clamped spline).
                  for (std::size_t i = 0; i < sp.piece_count(); ++i) {
                      t.expect(sp.piece(i).is_equal(q), "clamped piece reproduces the cubic");
                  }
                  // Moments equal q''(x_i).
                  for (std::size_t i = 0; i < xs.size(); ++i) {
                      t.expect(sp.moments()[i] == peval(qpp, xs[i]),
                               "moment equals second derivative of cubic");
                  }
                  // C^2 continuity at interior knots (left and right pieces agree in S'').
                  for (std::size_t i = 1; i + 1 < xs.size(); ++i) {
                      const Rational left = peval(psecond(sp.piece(i - 1)), xs[i]);
                      const Rational right = peval(psecond(sp.piece(i)), xs[i]);
                      t.expect(left == right && left == sp.moments()[i],
                               "second derivative continuous at interior knot");
                  }
              })
        .test("periodic_spline_is_C2_and_wraps",
              [](TestContext& t) {
                  // y_0 == y_n required. Periodic spline matches S, S', S'' at the ends.
                  auto xs = rats({0, 1, 2, 3});
                  auto ys = rats({0, 2, -1, 0});  // y_0 == y_3 == 0
                  auto sp = CubicSpline::periodic(xs, ys).value();
                  // Passes through the data.
                  for (std::size_t i = 0; i < xs.size(); ++i) {
                      t.expect(sp.evaluate(xs[i]).value() == ys[i], "periodic passes through node");
                  }
                  // S'' wraps: M_0 == M_n.
                  t.expect(sp.moments().front() == sp.moments().back(), "periodic S'' wraps");
                  // C^2 at interior knots.
                  for (std::size_t i = 1; i + 1 < xs.size(); ++i) {
                      const Rational left = peval(psecond(sp.piece(i - 1)), xs[i]);
                      const Rational right = peval(psecond(sp.piece(i)), xs[i]);
                      t.expect(left == right, "periodic S'' continuous at interior knot");
                  }
                  // C^1 across the wrap: S'(x_0) == S'(x_n).
                  const std::size_t last = sp.piece_count() - 1;
                  const Rational slope0 = peval(pfirst(sp.piece(0)), xs.front());
                  const Rational slopeN = peval(pfirst(sp.piece(last)), xs.back());
                  t.expect(slope0 == slopeN, "periodic S' wraps");
              })
        .test("hermite_from_slopes_and_pchip",
              [](TestContext& t) {
                  // from_slopes(q) reproduces a cubic q exactly on each interval.
                  const RationalPoly q = RationalPoly::from_coeffs({ri(1), ri(0), ri(-1), ri(1)});
                  auto xs = rats({0, 1, 2});
                  std::vector<Rational> ys;
                  std::vector<Rational> slopes;
                  const RationalPoly qp = pfirst(q);
                  for (const auto& x : xs) {
                      ys.push_back(peval(q, x));
                      slopes.push_back(peval(qp, x));
                  }
                  auto hs = HermiteSpline::from_slopes(xs, ys, slopes).value();
                  for (std::size_t i = 0; i < xs.size(); ++i) {
                      t.expect(hs.evaluate(xs[i]).value() == ys[i], "Hermite hits value");
                      const std::size_t seg = (i == xs.size() - 1) ? hs.piece_count() - 1 : i;
                      t.expect(peval(pfirst(hs.piece(seg)), xs[i]) == slopes[i],
                               "Hermite hits slope");
                  }

                  // PCHIP through data with an interior local maximum: slope there is 0, and it
                  // passes through every value exactly (all over Q).
                  auto px = rats({0, 1, 2});
                  auto py = rats({0, 1, 0});  // peak at x = 1
                  auto pc = HermiteSpline::pchip(px, py).value();
                  for (std::size_t i = 0; i < px.size(); ++i) {
                      t.expect(pc.evaluate(px[i]).value() == py[i], "PCHIP passes through node");
                  }
                  t.expect(pc.slopes()[1] == ri(0), "PCHIP zeroes slope at a local extremum");
              })
        .test("bezier_de_casteljau_bernstein_and_subdivision",
              [](TestContext& t) {
                  auto bez = BezierCurve::make(rats({1, 3, 2, 5})).value();
                  const Rational half = rat(1, 2);
                  // De Casteljau == Bernstein-sum == 21/8 at t = 1/2.
                  const Rational v = bez.evaluate(half).value();
                  t.expect(v == rat(21, 8), "de Casteljau value is 21/8");
                  t.expect(bez.evaluate_bernstein(half).value() == v,
                           "Bernstein-sum == de Casteljau");
                  // Subdivision midpoint: left end and right start meet at the curve point.
                  auto split = nimblecas::bezier_subdivide(bez, half).value();
                  t.expect(split.left.control_points().back() == v,
                           "left half ends at the split point");
                  t.expect(split.right.control_points().front() == v,
                           "right half starts at the split point");
                  t.expect(split.left.control_points().front() == ri(1),
                           "left half keeps the original start");
                  t.expect(split.right.control_points().back() == ri(5),
                           "right half keeps the original end");
              })
        .test("bezier_degree_one_is_lerp",
              [](TestContext& t) {
                  auto bez = BezierCurve::make(rats({2, 8})).value();
                  // (1-t)*2 + t*8 at t = 1/3 is 4.
                  t.expect(bez.evaluate(rat(1, 3)).value() == ri(4), "degree-1 Bezier is the lerp");
                  t.expect(bez.evaluate(ri(0)).value() == ri(2), "lerp at 0 is P0");
                  t.expect(bez.evaluate(ri(1)).value() == ri(8), "lerp at 1 is P1");
              })
        .test("bezier_power_basis_roundtrip",
              [](TestContext& t) {
                  auto bez = BezierCurve::make(rats({1, 3, 2, 5})).value();
                  auto power = bez.to_power_basis().value();
                  auto back = BezierCurve::from_power_basis(power, bez.degree()).value();
                  const auto orig = bez.control_points();
                  const auto rec = back.control_points();
                  t.expect(orig.size() == rec.size(), "roundtrip preserves control count");
                  for (std::size_t i = 0; i < orig.size(); ++i) {
                      t.expect(orig[i] == rec[i], "roundtrip preserves control point");
                  }
                  // Power basis agrees with de Casteljau at a rational parameter.
                  const Rational tt = rat(2, 5);
                  t.expect(peval(power, tt) == bez.evaluate(tt).value(),
                           "power basis == de Casteljau");
              })
        .test("bezier2_two_dimensional",
              [](TestContext& t) {
                  std::vector<Point2> cps{Point2{.x = ri(0), .y = ri(0)},
                                          Point2{.x = ri(2), .y = ri(4)}};
                  auto bez = nimblecas::BezierCurve2::make(cps).value();
                  auto p = bez.evaluate(rat(1, 2)).value();
                  t.expect(p == (Point2{.x = ri(1), .y = ri(2)}), "2-D Bezier midpoint exact");
                  auto elevated = bez.elevate().value();
                  t.expect(elevated.degree() == 2, "elevation raises degree");
                  t.expect(elevated.evaluate(rat(1, 3)).value() == bez.evaluate(rat(1, 3)).value(),
                           "elevation preserves the curve");
              })
        .test("bspline_partition_of_unity_and_endpoint_interpolation",
              [](TestContext& t) {
                  // Clamped quadratic knot vector; 5 control points.
                  auto knots = rats({0, 0, 0, 1, 2, 3, 3, 3});
                  auto ctrl = rats({1, 2, 0, 3, 1});
                  auto bs = BSpline::make(knots, ctrl, 2).value();
                  // Partition of unity at an interior rational u: sum_i N_{i,2}(u) == 1.
                  const Rational u = rat(3, 2);
                  Rational sum{};
                  for (std::size_t i = 0; i < ctrl.size(); ++i) {
                      sum = sum.add(bs.basis_function(i, u).value()).value();
                  }
                  t.expect(sum == ri(1), "B-spline basis is a partition of unity");
                  // Clamped ends interpolate the first / last control points.
                  t.expect(bs.evaluate(bs.domain_min()).value() == ctrl.front(),
                           "clamped B-spline interpolates P_0");
                  t.expect(bs.evaluate(bs.domain_max()).value() == ctrl.back(),
                           "clamped B-spline interpolates P_last");
              })
        .test("bspline_open_knot_vector_endpoint_is_not_clamped",
              [](TestContext& t) {
                  // OPEN (non-clamped) uniform knot vector U={0,1,2,3,4}, degree 1,
                  // 3 control points (5 == 3 + 1 + 1). Regression for the endpoint special
                  // case that used to short-circuit N to 1 at U[0]/U[m] even when the end is
                  // not clamped. Here N_{0,1}(0) is exactly 0, NOT 1: the (u-U0)/(U1-U0)
                  // factor vanishes at u=U[0]=0.
                  auto knots = rats({0, 1, 2, 3, 4});
                  auto ctrl = rats({1, 2, 3});
                  auto bs = BSpline::make(knots, ctrl, 1).value();
                  t.expect(bs.basis_function(0, ri(0)).value() == ri(0),
                           "open-vector N_{0,1}(0) is 0, not a clamped 1");
                  t.expect(bs.basis_function(2, ri(4)).value() == ri(0),
                           "open-vector N_{last,1}(U[m]) is 0, not a clamped 1");
                  // The recursion is still a partition of unity at an interior rational.
                  const Rational u = rat(3, 2);
                  Rational sum{};
                  for (std::size_t i = 0; i < ctrl.size(); ++i) {
                      sum = sum.add(bs.basis_function(i, u).value()).value();
                  }
                  t.expect(sum == ri(1), "open B-spline basis is a partition of unity interior");
              })
        .test("nurbs_rational_weights_exact",
              [](TestContext& t) {
                  // Degree-1 clamped NURBS with weights {1, 3}: C(1/2) = 3/2 exactly.
                  auto knots = rats({0, 0, 1, 1});
                  auto ctrl = rats({0, 2});
                  auto weights = rats({1, 3});
                  auto nb = NurbsCurve::make(knots, ctrl, weights, 1).value();
                  t.expect(nb.evaluate(rat(1, 2)).value() == rat(3, 2),
                           "rational-weight NURBS evaluates exactly");
                  // Endpoints are the control points regardless of weight.
                  t.expect(nb.evaluate(ri(0)).value() == ri(0), "NURBS interpolates P_0");
                  t.expect(nb.evaluate(ri(1)).value() == ri(2), "NURBS interpolates P_1");
              })
        .test("domain_errors",
              [](TestContext& t) {
                  // Unsorted knots for a cubic spline.
                  auto badx = rats({0, 2, 1});
                  auto goody = rats({0, 1, 2});
                  t.expect(CubicSpline::natural(badx, goody).error() == MathError::domain_error,
                           "natural spline rejects unsorted knots");
                  // Size mismatch.
                  auto x3 = rats({0, 1, 2});
                  auto y2 = rats({0, 1});
                  t.expect(CubicSpline::natural(x3, y2).error() == MathError::domain_error,
                           "natural spline rejects size mismatch");
                  // Periodic without y_0 == y_n.
                  auto px = rats({0, 1, 2, 3});
                  auto py = rats({0, 1, 2, 9});
                  t.expect(CubicSpline::periodic(px, py).error() == MathError::domain_error,
                           "periodic spline requires y_0 == y_n");
                  // Bezier t outside [0, 1].
                  auto bez = BezierCurve::make(rats({1, 2, 3})).value();
                  t.expect(bez.evaluate(ri(2)).error() == MathError::domain_error,
                           "Bezier rejects t > 1");
                  t.expect(bez.evaluate(ri(-1)).error() == MathError::domain_error,
                           "Bezier rejects t < 0");
                  // Empty Bezier.
                  t.expect(BezierCurve::make({}).error() == MathError::domain_error,
                           "Bezier rejects empty control set");
                  // B-spline knot/size-relation violation.
                  auto knots = rats({0, 0, 1, 1});
                  auto ctrl = rats({0, 1, 2});  // wrong count for degree 1 (needs 2)
                  t.expect(BSpline::make(knots, ctrl, 1).error() == MathError::domain_error,
                           "B-spline rejects a bad size relation");
                  // NURBS weight/control mismatch.
                  auto nk = rats({0, 0, 1, 1});
                  auto nc = rats({0, 2});
                  auto nw = rats({1});  // wrong weight count
                  t.expect(NurbsCurve::make(nk, nc, nw, 1).error() == MathError::domain_error,
                           "NURBS rejects weight/control mismatch");
              })
        .run();
}
