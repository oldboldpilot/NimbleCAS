// Tests for nimblecas.algnum: exact arithmetic in a simple algebraic number field
// Q(alpha) = Q[x]/(m(x)). Every expected value is hand-derived and exact.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.factor;
import nimblecas.matrix;
import nimblecas.algnum;
import nimblecas.testing;

using nimblecas::AlgebraicNumber;
using nimblecas::MathError;
using nimblecas::NumberField;
using nimblecas::Rational;
using nimblecas::RationalPoly;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

[[nodiscard]] auto ri(std::int64_t v) -> Rational {
    return Rational::from_int(v);
}

// A RationalPoly from integer coefficients (low degree first).
[[nodiscard]] auto ipoly(std::vector<std::int64_t> c) -> RationalPoly {
    std::vector<Rational> rs;
    rs.reserve(c.size());
    for (const std::int64_t v : c) {
        rs.push_back(Rational::from_int(v));
    }
    return RationalPoly::from_coeffs(std::move(rs));
}

// The element whose residue is the constant/linear/quadratic poly with the given rational
// coefficients (low degree first), built via from_poly (reduced mod m).
[[nodiscard]] auto elem(const NumberField& f, std::vector<Rational> coeffs) -> AlgebraicNumber {
    return f.from_poly(RationalPoly::from_coeffs(std::move(coeffs))).value();
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.algnum")
        .test("field_construction_rejects_reducible_and_degenerate",
              [](TestContext& t) {
                  // x^2 - 1 = (x-1)(x+1) is reducible -> domain_error.
                  t.expect(NumberField::create(ipoly({-1, 0, 1})).error() ==
                               MathError::domain_error,
                           "reducible m (x^2-1) is domain_error");
                  // A perfect square (x-1)^2 = x^2 - 2x + 1 is reducible -> domain_error.
                  t.expect(NumberField::create(ipoly({1, -2, 1})).error() ==
                               MathError::domain_error,
                           "square factor m is domain_error");
                  // A nonzero constant defines no extension -> domain_error.
                  t.expect(NumberField::create(ipoly({3})).error() == MathError::domain_error,
                           "constant m is domain_error");
                  // The zero polynomial -> domain_error.
                  t.expect(NumberField::create(RationalPoly{}).error() ==
                               MathError::domain_error,
                           "zero m is domain_error");
                  // x^2 - 2 is irreducible -> a field of degree 2.
                  auto f = NumberField::create(ipoly({-2, 0, 1}));
                  t.expect(f.has_value(), "x^2-2 constructs a field");
                  t.expect(f->degree() == 2, "degree of Q(sqrt2) is 2");
                  // A non-monic multiple is normalised: 2x^2 - 4 defines the same field.
                  auto f2 = NumberField::create(ipoly({-4, 0, 2}));
                  t.expect(f2.has_value() && f2->modulus().is_equal(ipoly({-2, 0, 1})),
                           "non-monic 2x^2-4 normalises to monic x^2-2");
              })
        .test("sqrt2_arithmetic",
              [](TestContext& t) {
                  const auto f = NumberField::create(ipoly({-2, 0, 1})).value();  // x^2 - 2
                  const auto a = f.generator().value();                            // sqrt2
                  const auto one = f.one();
                  // sqrt2 * sqrt2 = 2.
                  auto a2 = a.multiply(a).value();
                  t.expect(a2.is_equal(f.from_rational(ri(2))), "sqrt2 * sqrt2 = 2");
                  // (1 + sqrt2)(1 - sqrt2) = 1 - 2 = -1.
                  const auto onePlus = one.add(a).value();       // 1 + sqrt2
                  const auto oneMinus = one.subtract(a).value();  // 1 - sqrt2
                  auto prod = onePlus.multiply(oneMinus).value();
                  t.expect(prod.is_equal(f.from_rational(ri(-1))), "(1+sqrt2)(1-sqrt2) = -1");
                  // (sqrt2)^{-1} = (1/2) sqrt2.
                  auto ainv = a.inverse().value();
                  t.expect(ainv.is_equal(elem(f, {ri(0), rat(1, 2)})),
                           "sqrt2^{-1} = (1/2) alpha");
                  // Cross-check: sqrt2 * sqrt2^{-1} = 1.
                  t.expect(a.multiply(ainv).value().is_one(), "sqrt2 * sqrt2^{-1} = 1");
                  // (1 + sqrt2)^{-1} = sqrt2 - 1.
                  auto invPlus = onePlus.inverse().value();
                  t.expect(invPlus.is_equal(elem(f, {ri(-1), ri(1)})),
                           "(1+sqrt2)^{-1} = alpha - 1");
                  t.expect(onePlus.multiply(invPlus).value().is_one(),
                           "(1+sqrt2)(1+sqrt2)^{-1} = 1");
                  // pow: sqrt2^2 = 2, sqrt2^3 = 2 sqrt2, sqrt2^0 = 1.
                  t.expect(a.pow(2).value().is_equal(f.from_rational(ri(2))), "sqrt2^2 = 2");
                  t.expect(a.pow(3).value().is_equal(elem(f, {ri(0), ri(2)})),
                           "sqrt2^3 = 2 alpha");
                  t.expect(a.pow(0).value().is_one(), "sqrt2^0 = 1");
                  t.expect(a.pow(-1).error() == MathError::domain_error,
                           "negative power is domain_error");
              })
        .test("sqrt2_norm_and_trace",
              [](TestContext& t) {
                  const auto f = NumberField::create(ipoly({-2, 0, 1})).value();
                  const auto a = f.generator().value();
                  const auto onePlus = f.one().add(a).value();  // 1 + sqrt2
                  // N(1+sqrt2) = (1+sqrt2)(1-sqrt2) = -1; Tr(1+sqrt2) = (1+sqrt2)+(1-sqrt2) = 2.
                  t.expect(onePlus.norm().value() == ri(-1), "norm(1+sqrt2) = -1");
                  t.expect(onePlus.trace().value() == ri(2), "trace(1+sqrt2) = 2");
                  // N(sqrt2) = -2 (det [[0,2],[1,0]]); Tr(sqrt2) = 0.
                  t.expect(a.norm().value() == ri(-2), "norm(sqrt2) = -2");
                  t.expect(a.trace().value() == ri(0), "trace(sqrt2) = 0");
              })
        .test("gaussian_i_arithmetic",
              [](TestContext& t) {
                  const auto f = NumberField::create(ipoly({1, 0, 1})).value();  // x^2 + 1
                  const auto i = f.generator().value();
                  // i * i = -1.
                  t.expect(i.multiply(i).value().is_equal(f.from_rational(ri(-1))),
                           "i * i = -1");
                  // i^{-1} = -i.
                  auto iinv = i.inverse().value();
                  t.expect(iinv.is_equal(elem(f, {ri(0), ri(-1)})), "i^{-1} = -i");
                  // (1 + i)^{-1} = (1 - i)/2 = 1/2 - (1/2) i.
                  const auto onePlusI = f.one().add(i).value();
                  auto inv = onePlusI.inverse().value();
                  t.expect(inv.is_equal(elem(f, {rat(1, 2), rat(-1, 2)})),
                           "(1+i)^{-1} = 1/2 - (1/2) i");
                  t.expect(onePlusI.multiply(inv).value().is_one(), "(1+i)(1+i)^{-1} = 1");
                  // N(1+i) = 2, Tr(1+i) = 2.
                  t.expect(onePlusI.norm().value() == ri(2), "norm(1+i) = 2");
                  t.expect(onePlusI.trace().value() == ri(2), "trace(1+i) = 2");
              })
        .test("cbrt2_cubic_field",
              [](TestContext& t) {
                  const auto f = NumberField::create(ipoly({-2, 0, 0, 1})).value();  // x^3 - 2
                  t.expect(f.degree() == 3, "degree of Q(cbrt2) is 3");
                  const auto a = f.generator().value();  // cbrt2
                  // alpha^3 = 2.
                  t.expect(a.pow(3).value().is_equal(f.from_rational(ri(2))), "alpha^3 = 2");
                  // alpha^{-1} = (1/2) alpha^2.
                  auto ainv = a.inverse().value();
                  t.expect(ainv.is_equal(elem(f, {ri(0), ri(0), rat(1, 2)})),
                           "alpha^{-1} = (1/2) alpha^2");
                  t.expect(a.multiply(ainv).value().is_one(), "alpha * alpha^{-1} = 1");
                  // N(alpha) = 2 (det of [[0,0,2],[1,0,0],[0,1,0]] = 2); Tr(alpha) = 0.
                  t.expect(a.norm().value() == ri(2), "norm(cbrt2) = 2");
                  t.expect(a.trace().value() == ri(0), "trace(cbrt2) = 0");
              })
        .test("error_paths",
              [](TestContext& t) {
                  const auto f = NumberField::create(ipoly({-2, 0, 1})).value();  // Q(sqrt2)
                  // Inverse / divide by zero.
                  t.expect(f.zero().inverse().error() == MathError::division_by_zero,
                           "inverse of 0 is division_by_zero");
                  t.expect(f.one().divide(f.zero()).error() == MathError::division_by_zero,
                           "divide by 0 is division_by_zero");
                  // Mixing elements of different fields is a domain_error.
                  const auto g = NumberField::create(ipoly({1, 0, 1})).value();  // Q(i)
                  t.expect(f.one().add(g.one()).error() == MathError::domain_error,
                           "cross-field add is domain_error");
                  t.expect(f.generator().value().multiply(g.generator().value()).error() ==
                               MathError::domain_error,
                           "cross-field multiply is domain_error");
              })
        .run();
}
