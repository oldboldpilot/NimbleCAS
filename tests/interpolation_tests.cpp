// Tests for nimblecas.interpolation: exact polynomial interpolation over Q
// (Lagrange / Newton / barycentric / Neville / Hermite).
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.ratpoly;
import nimblecas.interpolation;
import nimblecas.testing;

using nimblecas::BarycentricInterpolant;
using nimblecas::MathError;
using nimblecas::NewtonInterpolant;
using nimblecas::Polynomial;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// A RationalPoly from integer coefficients (low degree first).
[[nodiscard]] auto ipoly(std::vector<std::int64_t> c) -> RationalPoly {
    return RationalPoly::from_polynomial(Polynomial{std::move(c)});
}

[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

[[nodiscard]] auto ri(std::int64_t v) -> Rational { return Rational::from_int(v); }

// Integer node/value vectors -> Rational vectors, for terse test data.
[[nodiscard]] auto rats(std::vector<std::int64_t> xs) -> std::vector<Rational> {
    std::vector<Rational> r;
    r.reserve(xs.size());
    for (std::int64_t v : xs) {
        r.push_back(Rational::from_int(v));
    }
    return r;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.interpolation")
        .test("recovers_known_polynomial",
              [](TestContext& t) {
                  // x^2 sampled at x = 0,1,2 -> (0,0),(1,1),(2,4). The unique degree<=2
                  // interpolant is exactly x^2.
                  auto nodes = rats({0, 1, 2});
                  auto values = rats({0, 1, 4});
                  auto lag = nimblecas::lagrange_polynomial(nodes, values).value();
                  t.expect(lag.is_equal(ipoly({0, 0, 1})), "Lagrange recovers x^2 exactly");
                  auto newt = nimblecas::newton_polynomial(nodes, values).value();
                  t.expect(newt.is_equal(ipoly({0, 0, 1})), "Newton recovers x^2 exactly");

                  // A cubic 2x^3 - x + 5 sampled at 4 points is recovered exactly.
                  auto cubic = ipoly({5, -1, 0, 2});
                  auto xs = rats({-1, 0, 1, 2});
                  std::vector<Rational> ys;
                  for (const auto& x : xs) {
                      ys.push_back(nimblecas::poly_evaluate(cubic, x).value());
                  }
                  auto rec = nimblecas::lagrange_polynomial(xs, ys).value();
                  t.expect(rec.is_equal(cubic), "cubic recovered exactly from 4 samples");
              })
        .test("methods_agree",
              [](TestContext& t) {
                  // Several datasets: Lagrange == Newton as polynomials, and
                  // barycentric/Neville evaluations match the polynomial everywhere tested.
                  const std::vector<std::pair<std::vector<Rational>, std::vector<Rational>>> data{
                      {rats({0, 1, 2, 3}), rats({1, 3, 7, 13})},   // x^2 + x + 1 on 0..3
                      {rats({-2, -1, 1, 4}), rats({5, 0, -2, 30})},
                      {rats({0, 2, 5}), rats({7, -3, 11})},
                  };
                  for (const auto& [nodes, values] : data) {
                      auto lag = nimblecas::lagrange_polynomial(nodes, values).value();
                      auto newt = nimblecas::newton_polynomial(nodes, values).value();
                      t.expect(lag.is_equal(newt), "Lagrange == Newton polynomial");

                      auto bary = BarycentricInterpolant::make(nodes, values).value();
                      // Evaluate at a spread of rational probe points (nodes and between).
                      const std::vector<Rational> probes{ri(-3), rat(1, 2), ri(3),
                                                          rat(7, 3), ri(10)};
                      for (const auto& x : probes) {
                          auto pv = nimblecas::poly_evaluate(lag, x).value();
                          auto bv = bary.evaluate(x).value();
                          auto nv = nimblecas::neville_evaluate(nodes, values, x).value();
                          t.expect(bv == pv, "barycentric == polynomial value");
                          t.expect(nv == pv, "Neville == polynomial value");
                      }
                  }
              })
        .test("passes_through_nodes",
              [](TestContext& t) {
                  auto nodes = rats({-2, -1, 1, 3, 4});
                  auto values = rats({9, -1, 5, 2, 17});
                  auto poly = nimblecas::newton_polynomial(nodes, values).value();
                  auto bary = BarycentricInterpolant::make(nodes, values).value();
                  for (std::size_t i = 0; i < nodes.size(); ++i) {
                      t.expect(nimblecas::poly_evaluate(poly, nodes[i]).value() == values[i],
                               "polynomial passes through node i");
                      t.expect(bary.evaluate(nodes[i]).value() == values[i],
                               "barycentric returns y_i at node i");
                      t.expect(nimblecas::neville_evaluate(nodes, values, nodes[i]).value() ==
                                   values[i],
                               "Neville returns y_i at node i");
                  }
              })
        .test("newton_incremental",
              [](TestContext& t) {
                  // Building point-by-point yields the same polynomial as building at once.
                  auto nodes = rats({0, 1, 2, 3});
                  auto values = rats({1, 2, 5, 10});  // x^2 + 1
                  auto batch = NewtonInterpolant::from_points(nodes, values).value();

                  NewtonInterpolant inc;  // empty
                  for (std::size_t i = 0; i < nodes.size(); ++i) {
                      inc = inc.with_point(nodes[i], values[i]).value();
                  }
                  t.expect(inc.size() == 4, "incremental interpolant has 4 points");
                  auto pb = batch.polynomial().value();
                  auto pi = inc.polynomial().value();
                  t.expect(pb.is_equal(pi), "incremental == batch Newton polynomial");
                  t.expect(pb.is_equal(ipoly({1, 0, 1})), "polynomial is x^2 + 1");
                  // Nested-Newton evaluate matches the assembled polynomial.
                  t.expect(inc.evaluate(rat(5, 2)).value() ==
                               nimblecas::poly_evaluate(pi, rat(5, 2)).value(),
                           "Newton evaluate == polynomial value");

                  // Adding a duplicate node is a domain_error.
                  t.expect(inc.with_point(ri(1), ri(99)).error() == MathError::domain_error,
                           "duplicate node in with_point is domain_error");
              })
        .test("fractional_stays_exact",
              [](TestContext& t) {
                  // Fractional nodes and values: the interpolant is exact in Q, no floats.
                  // Two points (1/2, 1/3), (3/2, 5) -> a line; check value at the nodes and
                  // that all four methods agree at an off-node rational point.
                  std::vector<Rational> nodes{rat(1, 2), rat(3, 2)};
                  std::vector<Rational> values{rat(1, 3), ri(5)};
                  auto lag = nimblecas::lagrange_polynomial(nodes, values).value();
                  t.expect(nimblecas::poly_evaluate(lag, rat(1, 2)).value() == rat(1, 3),
                           "line hits (1/2, 1/3) exactly");
                  t.expect(nimblecas::poly_evaluate(lag, rat(3, 2)).value() == ri(5),
                           "line hits (3/2, 5) exactly");

                  auto newt = nimblecas::newton_polynomial(nodes, values).value();
                  t.expect(lag.is_equal(newt), "Lagrange == Newton on fractional data");

                  auto bary = BarycentricInterpolant::make(nodes, values).value();
                  const Rational probe = rat(1, 1);  // x = 1, the midpoint
                  auto pv = nimblecas::poly_evaluate(lag, probe).value();
                  // Exact midpoint value: (1/3 + 5)/2 = 8/3.
                  t.expect(pv == rat(8, 3), "midpoint value is exactly 8/3");
                  t.expect(bary.evaluate(probe).value() == pv, "barycentric agrees, exact");
                  t.expect(nimblecas::neville_evaluate(nodes, values, probe).value() == pv,
                           "Neville agrees, exact");
              })
        .test("hermite_value_and_derivative",
              [](TestContext& t) {
                  // Match value + first derivative at each node: (0,0,slope 1),(1,1,slope 2).
                  std::vector<Rational> nodes{ri(0), ri(1)};
                  std::vector<Rational> values{ri(0), ri(1)};
                  std::vector<Rational> slopes{ri(1), ri(2)};
                  auto h = nimblecas::hermite_polynomial(nodes, values, slopes).value();
                  auto hp = h.derivative().value();
                  for (std::size_t i = 0; i < nodes.size(); ++i) {
                      t.expect(nimblecas::poly_evaluate(h, nodes[i]).value() == values[i],
                               "Hermite matches value at node");
                      t.expect(nimblecas::poly_evaluate(hp, nodes[i]).value() == slopes[i],
                               "Hermite matches derivative at node");
                  }
                  t.expect(h.degree() <= 3, "Hermite degree <= 2n-1 = 3");

                  // Single node (a, v, slope d) -> the exact line v + d(x-a).
                  std::vector<Rational> n1{ri(2)};
                  std::vector<Rational> v1{ri(5)};
                  std::vector<Rational> d1{ri(3)};
                  auto line = nimblecas::hermite_polynomial(n1, v1, d1).value();
                  t.expect(line.is_equal(ipoly({-1, 3})), "one-point Hermite is 3x - 1");

                  // Fractional Hermite stays exact: node 1/2, value 1/4, slope 1.
                  std::vector<Rational> nf{rat(1, 2)};
                  std::vector<Rational> vf{rat(1, 4)};
                  std::vector<Rational> df{ri(1)};
                  auto hf = nimblecas::hermite_polynomial(nf, vf, df).value();
                  t.expect(nimblecas::poly_evaluate(hf, rat(1, 2)).value() == rat(1, 4),
                           "fractional Hermite hits value exactly");
                  t.expect(nimblecas::poly_evaluate(hf.derivative().value(), rat(1, 2)).value() ==
                               ri(1),
                           "fractional Hermite hits slope exactly");
              })
        .test("domain_errors",
              [](TestContext& t) {
                  // Duplicate node.
                  auto dupN = rats({0, 1, 1});
                  auto dupV = rats({0, 1, 2});
                  t.expect(nimblecas::lagrange_polynomial(dupN, dupV).error() ==
                               MathError::domain_error,
                           "Lagrange rejects duplicate nodes");
                  t.expect(nimblecas::newton_polynomial(dupN, dupV).error() ==
                               MathError::domain_error,
                           "Newton rejects duplicate nodes");
                  t.expect(nimblecas::neville_evaluate(dupN, dupV, ri(0)).error() ==
                               MathError::domain_error,
                           "Neville rejects duplicate nodes");
                  t.expect(nimblecas::barycentric_weights(dupN).error() ==
                               MathError::domain_error,
                           "barycentric weights reject duplicate nodes");

                  // Size mismatch.
                  auto n2 = rats({0, 1, 2});
                  auto v2 = rats({0, 1});
                  t.expect(nimblecas::lagrange_polynomial(n2, v2).error() ==
                               MathError::domain_error,
                           "Lagrange rejects size mismatch");
                  t.expect(nimblecas::newton_polynomial(n2, v2).error() ==
                               MathError::domain_error,
                           "Newton rejects size mismatch");

                  // Empty input.
                  std::vector<Rational> empty;
                  t.expect(nimblecas::lagrange_polynomial(empty, empty).error() ==
                               MathError::domain_error,
                           "Lagrange rejects empty input");

                  // Hermite: derivatives size mismatch.
                  auto hn = rats({0, 1});
                  auto hv = rats({0, 1});
                  auto hd = rats({1});  // wrong length
                  t.expect(nimblecas::hermite_polynomial(hn, hv, hd).error() ==
                               MathError::domain_error,
                           "Hermite rejects derivative size mismatch");
                  // Hermite: duplicate node.
                  auto hdupN = rats({2, 2});
                  auto hdupV = rats({1, 1});
                  auto hdupD = rats({0, 0});
                  t.expect(nimblecas::hermite_polynomial(hdupN, hdupV, hdupD).error() ==
                               MathError::domain_error,
                           "Hermite rejects duplicate nodes");
              })
        .run();
}
