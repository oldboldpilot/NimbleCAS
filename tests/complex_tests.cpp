// Tests for nimblecas.complex: exact Gaussian-rational arithmetic.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.complex;
import nimblecas.testing;

using nimblecas::Complex;
using nimblecas::MathError;
using nimblecas::Rational;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto ri(std::int64_t v) -> Rational { return Rational::from_int(v); }

[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Rational {
    return Rational::make(n, d).value();
}

// Build the complex number re + im*i from integer parts.
[[nodiscard]] auto cx(std::int64_t re, std::int64_t im) -> Complex {
    return Complex::make(ri(re), ri(im));
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.complex")
        .test("addition_and_subtraction",
              [](TestContext& t) {
                  // (1 + 2i) + (3 - i) = 4 + i
                  t.expect(cx(1, 2).add(cx(3, -1)).value() == cx(4, 1), "(1+2i)+(3-i) = 4+i");
                  // (4 + i) - (3 - i) = 1 + 2i  (inverse of the above)
                  t.expect(cx(4, 1).subtract(cx(3, -1)).value() == cx(1, 2), "(4+i)-(3-i) = 1+2i");
              })
        .test("multiplication",
              [](TestContext& t) {
                  // (1 + i)(1 - i) = 2, a real result
                  auto p = cx(1, 1).multiply(cx(1, -1)).value();
                  t.expect(p == cx(2, 0), "(1+i)(1-i) = 2");
                  t.expect(p.is_real(), "the product is real");
                  // i * i = -1
                  t.expect(Complex::i().multiply(Complex::i()).value() == Complex::from_int(-1),
                           "i*i = -1");
              })
        .test("conjugate_and_norm_squared",
              [](TestContext& t) {
                  // conj(3 + 4i) = 3 - 4i
                  t.expect(cx(3, 4).conjugate().value() == cx(3, -4), "conj(3+4i) = 3-4i");
                  // |3 + 4i|^2 = 9 + 16 = 25 (exact)
                  t.expect(cx(3, 4).norm_squared().value() == ri(25), "norm_squared(3+4i) = 25");
              })
        .test("division",
              [](TestContext& t) {
                  // (1 + i)/(1 - i) = i
                  t.expect(cx(1, 1).divide(cx(1, -1)).value() == Complex::i(), "(1+i)/(1-i) = i");
                  // dividing by the zero complex number is reported, not undefined
                  t.expect(cx(1, 1).divide(Complex{}).error() == MathError::division_by_zero,
                           "divide by zero is division_by_zero");
              })
        .test("reciprocal",
              [](TestContext& t) {
                  // 1/i = -i
                  auto r = Complex::i().reciprocal().value();
                  t.expect(r == cx(0, -1), "reciprocal(i) = -i");
                  t.expect(r == Complex::i().negate().value(), "reciprocal(i) = -i (via negate)");
                  // reciprocal of zero fails
                  t.expect(Complex{}.reciprocal().error() == MathError::division_by_zero,
                           "reciprocal(0) is division_by_zero");
              })
        .test("predicates",
              [](TestContext& t) {
                  t.expect(Complex::from_real(rat(3, 2)).is_real(), "3/2 + 0i is real");
                  t.expect(!Complex::from_real(rat(3, 2)).is_imaginary(), "3/2 is not imaginary");
                  t.expect(Complex::i().is_imaginary(), "i is imaginary (real part zero)");
                  t.expect(!Complex::i().is_real(), "i is not real");
                  t.expect(Complex{}.is_zero(), "default Complex is zero");
              })
        .test("fractional_arithmetic",
              [](TestContext& t) {
                  // z = 1/2 + 1/2 i
                  const auto half = rat(1, 2);
                  const auto z = Complex::make(half, half);
                  // z + conj(z) = 2*Re(z) = 1
                  auto sum = z.add(z.conjugate().value()).value();
                  t.expect(sum == Complex::from_int(1), "z + conj(z) = 1");
                  // z * conj(z) = |z|^2 = 1/4 + 1/4 = 1/2
                  auto prod = z.multiply(z.conjugate().value()).value();
                  t.expect(prod == Complex::from_real(rat(1, 2)), "z * conj(z) = 1/2");
                  t.expect(z.norm_squared().value() == rat(1, 2), "norm_squared(z) = 1/2");
                  // (1/2 + 1/2 i) / (1/2 + 1/2 i) = 1
                  t.expect(z.divide(z).value() == Complex::from_int(1), "z / z = 1");
              })
        .test("to_string",
              [](TestContext& t) {
                  t.expect(cx(3, -4).to_string() == "3 - 4i", "3 - 4i");
                  t.expect(cx(3, 4).to_string() == "3 + 4i", "3 + 4i");
                  t.expect(cx(2, 0).to_string() == "2", "purely real prints without i");
                  t.expect(Complex::make(ri(0), ri(5)).to_string() == "5i", "purely imaginary 5i");
                  t.expect(Complex::i().to_string() == "i", "the unit imaginary prints as i");
                  t.expect(cx(0, -1).to_string() == "-i", "negative unit imaginary prints as -i");
                  t.expect(Complex{}.to_string() == "0", "zero prints as 0");
              })
        .run();
}
