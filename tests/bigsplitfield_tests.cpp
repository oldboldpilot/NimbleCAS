// Tests for nimblecas.bigsplitfield: splitting-field construction over Q on the UNBOUNDED
// rationals — the bignum mirror of nimblecas.splitfield.
// @author Olumuyiwa Oluwasanmi
//
// Oracles are hand-verified. The int64-mirroring cases cover the degree-2 splitting fields of
// x^2 + 1 (roots +/- i) and x^2 - 2 (roots +/- sqrt2), plus Trager factorization of x^2 + 1
// embedded into Q(i) into (x - i)(x + i). The LOAD-BEARING case is the degree-6 splitting field
// of x^3 - 2: its intermediate Trager norms and primitive-element multiplication matrices
// overflow the int64 splitfield tier, but on BigRational they are exact, so the construction
// completes — a degree-6 field containing all three cube roots of 2 (each r with r^3 == 2). The
// honest-refusal cases cover max_degree too small (not_implemented) and a non-positive-degree
// input (domain_error).

import std;
import nimblecas.core;
import nimblecas.bigrational;
import nimblecas.bigratpoly;
import nimblecas.bigalgnum;
import nimblecas.bigalgpoly;
import nimblecas.bigsplitfield;
import nimblecas.testing;

using nimblecas::BigAlgebraicNumber;
using nimblecas::BigAlgebraicPoly;
using nimblecas::BigNumberField;
using nimblecas::BigRational;
using nimblecas::BigRationalPoly;
using nimblecas::MathError;
using nimblecas::Result;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// BigRational from an int64.
auto q(std::int64_t v) -> BigRational { return BigRational::from_int(v); }

// BigRationalPoly from int64 coefficients in index order (constant first).
auto poly(std::vector<std::int64_t> cs) -> BigRationalPoly {
    std::vector<BigRational> v;
    v.reserve(cs.size());
    for (std::int64_t c : cs) {
        v.push_back(q(c));
    }
    return BigRationalPoly::from_coeffs(std::move(v));
}

// x^2 + 1, x^2 - 2, and x^3 - 2.
auto poly_x2_plus_1() -> BigRationalPoly { return poly({1, 0, 1}); }
auto poly_x2_minus_2() -> BigRationalPoly { return poly({-2, 0, 1}); }
auto poly_x3_minus_2() -> BigRationalPoly { return poly({-2, 0, 0, 1}); }

// Evaluate the Q[x] polynomial p at the field element x by embedding p into x's field and
// running Horner's scheme; the result is 0 exactly when x is a root of p.
auto eval_poly_at(const BigNumberField& f, const BigRationalPoly& p, const BigAlgebraicNumber& x)
    -> Result<BigAlgebraicNumber> {
    const BigAlgebraicPoly ap = BigAlgebraicPoly::embed(f, p);
    return ap.evaluate(x);
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.bigsplitfield")
        .test("splitting field of x^2 + 1 over Q: degree 2, roots +/- i vanish",
              [&](TestContext& t) {
                  const std::vector<BigRationalPoly> in{poly_x2_plus_1()};
                  auto sf = nimblecas::splitting_field(in, 6);
                  const bool ok = sf.has_value();
                  t.expect(ok, "splitting_field(x^2+1) succeeded");
                  if (!ok) return;
                  t.expect(sf->field.degree() == 2, "splitting field degree == 2");
                  t.expect(sf->roots.size() == 1, "one input polynomial reported back");
                  if (sf->roots.size() != 1) return;
                  const auto& [p, rts] = sf->roots[0];
                  t.expect(p.is_equal(poly_x2_plus_1()), "reported polynomial is x^2 + 1");
                  t.expect(rts.size() == 2, "x^2 + 1 contributes exactly 2 roots");
                  if (rts.size() != 2) return;
                  for (const BigAlgebraicNumber& r : rts) {
                      auto val = eval_poly_at(sf->field, poly_x2_plus_1(), r);
                      const bool okv = val.has_value();
                      t.expect(okv, "evaluated x^2 + 1 at a root");
                      if (!okv) return;
                      t.expect(val->is_zero(), "x^2 + 1 vanishes at the root");
                  }
                  t.expect(!rts[0].is_equal(rts[1]), "the two roots +/- i are distinct");
              })
        .test("splitting field of x^2 - 2 over Q: degree 2, roots +/- sqrt2 vanish",
              [&](TestContext& t) {
                  const std::vector<BigRationalPoly> in{poly_x2_minus_2()};
                  auto sf = nimblecas::splitting_field(in, 6);
                  const bool ok = sf.has_value();
                  t.expect(ok, "splitting_field(x^2-2) succeeded");
                  if (!ok) return;
                  t.expect(sf->field.degree() == 2, "splitting field degree == 2");
                  if (sf->roots.size() != 1) return;
                  const auto& rts = sf->roots[0].second;
                  t.expect(rts.size() == 2, "x^2 - 2 contributes exactly 2 roots");
                  if (rts.size() != 2) return;
                  for (const BigAlgebraicNumber& r : rts) {
                      auto val = eval_poly_at(sf->field, poly_x2_minus_2(), r);
                      const bool okv = val.has_value();
                      t.expect(okv, "evaluated x^2 - 2 at a root");
                      if (!okv) return;
                      t.expect(val->is_zero(), "x^2 - 2 vanishes at the root");
                  }
                  t.expect(!rts[0].is_equal(rts[1]), "the two roots +/- sqrt2 are distinct");
              })
        .test("factor_over_field: x^2 + 1 splits into (x - i)(x + i) over Q(i)",
              [&](TestContext& t) {
                  auto qi_r = BigNumberField::create(poly_x2_plus_1());
                  const bool okf = qi_r.has_value();
                  t.expect(okf, "constructed Q(i)");
                  if (!okf) return;
                  const BigNumberField qi = *qi_r;
                  const BigAlgebraicPoly embedded = BigAlgebraicPoly::embed(qi, poly_x2_plus_1());

                  auto factors = nimblecas::factor_over_field(qi, embedded);
                  const bool okc = factors.has_value();
                  t.expect(okc, "factor_over_field succeeded");
                  if (!okc) return;
                  t.expect(factors->size() == 2, "x^2 + 1 factors into two pieces over Q(i)");
                  if (factors->size() != 2) return;
                  t.expect((*factors)[0].degree() == 1 && (*factors)[1].degree() == 1,
                           "both factors are linear");

                  auto roots = nimblecas::roots_in_field(qi, embedded);
                  const bool okr = roots.has_value();
                  t.expect(okr, "roots_in_field succeeded");
                  if (!okr) return;
                  t.expect(roots->size() == 2, "two roots of x^2 + 1 lie in Q(i)");
                  if (roots->size() != 2) return;
                  for (const BigAlgebraicNumber& r : *roots) {
                      auto val = eval_poly_at(qi, poly_x2_plus_1(), r);
                      const bool okv = val.has_value();
                      t.expect(okv, "evaluated x^2 + 1 at a root in Q(i)");
                      if (!okv) return;
                      t.expect(val->is_zero(), "x^2 + 1 vanishes at the root in Q(i)");
                  }
                  // The two roots must be i and -i: negatives of each other, and distinct.
                  auto neg0 = (*roots)[0].negate();
                  const bool okn = neg0.has_value();
                  t.expect(okn, "negated the first root");
                  if (!okn) return;
                  t.expect(neg0->is_equal((*roots)[1]), "the roots are i and -i (negatives)");
              })
        // --- HEADLINE: the case the int64 tier overflows on ------------------------------
        .test("splitting field of x^3 - 2: degree 6, three cube roots of 2",
              [&](TestContext& t) {
                  const std::vector<BigRationalPoly> in{poly_x3_minus_2()};
                  auto sf = nimblecas::splitting_field(in, 6);
                  const bool ok = sf.has_value();
                  t.expect(ok, "splitting_field(x^3-2, max_degree=6) succeeded");
                  if (!ok) return;
                  t.expect(sf->field.degree() == 6, "splitting field of x^3 - 2 has degree 6");
                  if (sf->roots.size() != 1) return;
                  const auto& rts = sf->roots[0].second;
                  t.expect(rts.size() == 3, "x^3 - 2 contributes exactly 3 roots");
                  if (rts.size() != 3) return;

                  const BigAlgebraicNumber two = sf->field.from_bigrational(q(2));
                  for (const BigAlgebraicNumber& r : rts) {
                      auto cube = r.pow(3);
                      const bool okc = cube.has_value();
                      t.expect(okc, "cubed a root in the splitting field");
                      if (!okc) return;
                      t.expect(cube->is_equal(two), "r^3 == 2 in the splitting field");
                      auto val = eval_poly_at(sf->field, poly_x3_minus_2(), r);
                      const bool okv = val.has_value();
                      t.expect(okv, "evaluated x^3 - 2 at a root");
                      if (!okv) return;
                      t.expect(val->is_zero(), "x^3 - 2 vanishes at the root");
                  }
                  // All three cube roots are pairwise distinct.
                  t.expect(!rts[0].is_equal(rts[1]) && !rts[0].is_equal(rts[2]) &&
                               !rts[1].is_equal(rts[2]),
                           "the three cube roots of 2 are pairwise distinct");
              })
        // --- honest refusals -------------------------------------------------------------
        .test("splitting field of x^3 - 2 with max_degree too small is not_implemented",
              [&](TestContext& t) {
                  const std::vector<BigRationalPoly> in{poly_x3_minus_2()};
                  auto sf = nimblecas::splitting_field(in, 2);  // needs degree 6
                  t.expect(!sf.has_value() && sf.error() == MathError::not_implemented,
                           "max_degree = 2 for x^3 - 2 -> not_implemented");
              })
        .test("splitting field with a non-positive-degree input is domain_error",
              [&](TestContext& t) {
                  const std::vector<BigRationalPoly> in{poly({5})};  // constant: degree 0
                  auto sf = nimblecas::splitting_field(in, 6);
                  t.expect(!sf.has_value() && sf.error() == MathError::domain_error,
                           "constant input polynomial -> domain_error");
              })
        .run();
}
