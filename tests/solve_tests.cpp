// Tests for nimblecas.solve: exact closed-form roots (<= quartic) plus numeric
// companion-matrix eigenvalues for the non-radical degree >= 5 remainder.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.ratpoly;
import nimblecas.symbolic;
import nimblecas.solve;
import nimblecas.testing;

using nimblecas::AddNode;
using nimblecas::ConstantNode;
using nimblecas::Expr;
using nimblecas::MathError;
using nimblecas::MulNode;
using nimblecas::Polynomial;
using nimblecas::PowerNode;
using nimblecas::RationalPoly;
using nimblecas::Root;
using nimblecas::solve_poly;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Build a RationalPoly from integer coefficients (low degree first).
[[nodiscard]] auto ipoly(std::vector<std::int64_t> c) -> RationalPoly {
    return RationalPoly::from_polynomial(Polynomial{std::move(c)});
}

[[nodiscard]] auto mul(const RationalPoly& a, const RationalPoly& b) -> RationalPoly {
    return a.multiply(b).value();
}

// Numerically evaluate an Expr, treating power(base, exp) as the principal complex power.
// This lets any produced root — rational, radical, or numeric leaf — be substituted back
// into its polynomial to confirm it is genuinely a root.
[[nodiscard]] auto eval_expr(const Expr& e) -> std::complex<double> {
    const auto& v = e.node().value;
    if (const auto* c = std::get_if<ConstantNode>(&v)) {
        return std::visit(
            [](const auto& x) -> std::complex<double> {
                using X = std::decay_t<decltype(x)>;
                if constexpr (std::is_same_v<X, std::pair<std::int64_t, std::int64_t>>) {
                    return {static_cast<double>(x.first) / static_cast<double>(x.second), 0.0};
                } else {
                    return {static_cast<double>(x), 0.0};
                }
            },
            c->value);
    }
    if (const auto* a = std::get_if<AddNode>(&v)) {
        std::complex<double> s{0.0, 0.0};
        for (const Expr& t : a->terms) {
            s += eval_expr(t);
        }
        return s;
    }
    if (const auto* m = std::get_if<MulNode>(&v)) {
        std::complex<double> s{1.0, 0.0};
        for (const Expr& f : m->factors) {
            s *= eval_expr(f);
        }
        return s;
    }
    if (const auto* pw = std::get_if<PowerNode>(&v)) {
        return std::pow(eval_expr(pw->base), eval_expr(pw->exponent));
    }
    return {std::numeric_limits<double>::quiet_NaN(), 0.0};  // symbol/function: unexpected here
}

// |p(z)| for the polynomial p and a complex z (Horner over the rational coefficients).
[[nodiscard]] auto residual(const RationalPoly& p, std::complex<double> z) -> double {
    const std::span<const nimblecas::Rational> c = p.coefficients();
    std::complex<double> acc{0.0, 0.0};
    for (std::size_t i = c.size(); i-- > 0;) {
        const double coeff =
            static_cast<double>(c[i].numerator()) / static_cast<double>(c[i].denominator());
        acc = acc * z + coeff;
    }
    return std::abs(acc);
}

// Every returned root, substituted back, must satisfy p ~ 0.
[[nodiscard]] auto all_roots_vanish(const RationalPoly& p, const std::vector<Root>& roots,
                                    double tol) -> bool {
    return std::ranges::all_of(
        roots, [&](const Root& r) { return residual(p, eval_expr(r.value)) < tol; });
}

// Is there a root structurally equal to `want` (used for the clean radical/rational forms)?
[[nodiscard]] auto has_value(const std::vector<Root>& roots, const Expr& want, bool exact)
    -> bool {
    return std::ranges::any_of(roots, [&](const Root& r) {
        return r.exact == exact && r.value.is_equivalent_to(want);
    });
}

[[nodiscard]] auto rexpr(std::int64_t n, std::int64_t d) -> Expr {
    return Expr::rational(n, d).value();
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.solve")
        .test("quadratic_irrational_radicals",
              [](TestContext& t) {
                  // x^2 - 2 -> +/- sqrt(2), kept as power(2, 1/2). No rational roots.
                  auto roots = solve_poly(ipoly({-2, 0, 1})).value();
                  t.expect(roots.size() == 2, "two roots");
                  const Expr sqrt2 = Expr::power(Expr::integer(2), rexpr(1, 2));
                  t.expect(has_value(roots, sqrt2, true), "+sqrt(2) as power(2,1/2), exact");
                  t.expect(has_value(roots, Expr::product({Expr::integer(-1), sqrt2}), true),
                           "-sqrt(2), exact");
                  t.expect(all_roots_vanish(ipoly({-2, 0, 1}), roots, 1e-9), "both satisfy x^2-2");
              })
        .test("quadratic_imaginary_radical",
              [](TestContext& t) {
                  // x^2 + 1 -> +/- i, kept as the imaginary radical power(-1, 1/2).
                  auto roots = solve_poly(ipoly({1, 0, 1})).value();
                  t.expect(roots.size() == 2, "two roots");
                  const Expr i_unit = Expr::power(Expr::integer(-1), rexpr(1, 2));
                  t.expect(has_value(roots, i_unit, true), "power(-1,1/2), exact");
                  t.expect(has_value(roots, Expr::product({Expr::integer(-1), i_unit}), true),
                           "its negation, exact");
                  t.expect(all_roots_vanish(ipoly({1, 0, 1}), roots, 1e-9), "both satisfy x^2+1");
              })
        .test("quadratic_rational_exact",
              [](TestContext& t) {
                  // x^2 - 5x + 6 -> {2, 3}, exact rationals with multiplicity 1.
                  auto roots = solve_poly(ipoly({6, -5, 1})).value();
                  t.expect(roots.size() == 2, "two roots");
                  t.expect(has_value(roots, Expr::integer(2), true), "root 2 exact");
                  t.expect(has_value(roots, Expr::integer(3), true), "root 3 exact");
                  t.expect(std::ranges::all_of(roots,
                                               [](const Root& r) {
                                                   return r.multiplicity == std::uint64_t{1};
                                               }),
                           "rational roots carry multiplicity 1");
              })
        .test("linear_rational_root",
              [](TestContext& t) {
                  // 2x - 1 -> 1/2 (rational, exact).
                  auto roots = solve_poly(ipoly({-1, 2})).value();
                  t.expect(roots.size() == 1, "one root");
                  t.expect(has_value(roots, rexpr(1, 2), true), "root 1/2 exact");
              })
        .test("cubic_rational_plus_quadratic",
              [](TestContext& t) {
                  // (x-1)(x^2-2) = x^3 - x^2 - 2x + 2. Peel rational 1, then quadratic radicals.
                  const RationalPoly p = ipoly({2, -2, -1, 1});
                  auto roots = solve_poly(p).value();
                  t.expect(roots.size() == 3, "three roots");
                  t.expect(has_value(roots, Expr::integer(1), true), "rational root 1 exact");
                  t.expect(has_value(roots, Expr::power(Expr::integer(2), rexpr(1, 2)), true),
                           "sqrt(2) from the quadratic factor, exact");
                  t.expect(all_roots_vanish(p, roots, 1e-9), "all three satisfy the cubic");
              })
        .test("depressed_cubic_cube_roots",
              [](TestContext& t) {
                  // x^3 - 2 -> cube roots of 2; the real one is power(2, 1/3).
                  const RationalPoly p = ipoly({-2, 0, 0, 1});
                  auto roots = solve_poly(p).value();
                  t.expect(roots.size() == 3, "three roots");
                  t.expect(has_value(roots, Expr::power(Expr::integer(2), rexpr(1, 3)), true),
                           "real cube root power(2,1/3), exact");
                  t.expect(all_roots_vanish(p, roots, 1e-9), "all three cube roots satisfy x^3-2");
              })
        .test("general_cubic_cardano",
              [](TestContext& t) {
                  // x^3 - 3x - 1 (irreducible, casus irreducibilis): three real irrational roots
                  // kept as nested radicals. Verify via back-substitution.
                  const RationalPoly p = ipoly({-1, -3, 0, 1});
                  auto roots = solve_poly(p).value();
                  t.expect(roots.size() == 3, "three roots");
                  t.expect(std::ranges::all_of(roots, [](const Root& r) { return r.exact; }),
                           "all exact (radical) roots");
                  t.expect(all_roots_vanish(p, roots, 1e-6), "Cardano radicals satisfy the cubic");
              })
        .test("quartic_pure_fourth_roots",
              [](TestContext& t) {
                  // x^4 - 2 -> real quart root power(2, 1/4) plus i-multiples.
                  const RationalPoly p = ipoly({-2, 0, 0, 0, 1});
                  auto roots = solve_poly(p).value();
                  t.expect(roots.size() == 4, "four roots");
                  t.expect(has_value(roots, Expr::power(Expr::integer(2), rexpr(1, 4)), true),
                           "real fourth root power(2,1/4), exact");
                  t.expect(all_roots_vanish(p, roots, 1e-9), "all four satisfy x^4-2");
              })
        .test("quartic_biquadratic_factors",
              [](TestContext& t) {
                  // x^4 - 5x^2 + 6 = (x^2-2)(x^2-3): four irrational roots, no rational roots.
                  const RationalPoly p = ipoly({6, 0, -5, 0, 1});
                  auto roots = solve_poly(p).value();
                  t.expect(roots.size() == 4, "four roots");
                  t.expect(std::ranges::all_of(roots, [](const Root& r) { return r.exact; }),
                           "all exact (radical) roots");
                  t.expect(all_roots_vanish(p, roots, 1e-9), "all four satisfy the biquadratic");
              })
        .test("quartic_general_ferrari",
              [](TestContext& t) {
                  // x^4 - x^3 - x - 1: q != 0 after depression -> full Ferrari via resolvent cubic.
                  const RationalPoly p = ipoly({-1, -1, 0, -1, 1});  // x^4 - x^3 - x - 1
                  auto roots = solve_poly(p).value();
                  t.expect(roots.size() == 4, "four roots");
                  t.expect(std::ranges::all_of(roots, [](const Root& r) { return r.exact; }),
                           "all exact (radical) roots");
                  t.expect(all_roots_vanish(p, roots, 1e-6), "Ferrari roots satisfy the quartic");
              })
        .test("factorable_quintic_all_exact",
              [](TestContext& t) {
                  // (x-1)(x-2)(x^3-2): rational 1,2 exact + cubic radicals exact; no numeric path.
                  const RationalPoly p = mul(mul(ipoly({-1, 1}), ipoly({-2, 1})), ipoly({-2, 0, 0, 1}));
                  auto roots = solve_poly(p).value();
                  t.expect(roots.size() == 5, "five roots");
                  t.expect(std::ranges::all_of(roots, [](const Root& r) { return r.exact; }),
                           "every root is exact (factors into <= cubic pieces)");
                  t.expect(has_value(roots, Expr::integer(1), true), "rational root 1");
                  t.expect(has_value(roots, Expr::integer(2), true), "rational root 2");
                  t.expect(has_value(roots, Expr::power(Expr::integer(2), rexpr(1, 3)), true),
                           "cube root power(2,1/3)");
                  t.expect(all_roots_vanish(p, roots, 1e-6), "all five satisfy the quintic");
              })
        .test("irreducible_quintic_numeric_eigenvalues",
              [](TestContext& t) {
                  // x^5 - x - 1: irreducible, not solvable in radicals -> five NUMERIC roots via
                  // companion-matrix eigenvalues (1 real ~ 1.1673, two conjugate pairs).
                  const RationalPoly p = ipoly({-1, -1, 0, 0, 0, 1});
                  auto roots = solve_poly(p).value();
                  t.expect(roots.size() == 5, "five roots");
                  t.expect(std::ranges::none_of(roots, [](const Root& r) { return r.exact; }),
                           "every degree-5 root is flagged numeric (exact == false)");
                  // exactly one real root (imag ~ 0), approximately 1.1673.
                  std::size_t reals = 0;
                  bool found_real = false;
                  for (const Root& r : roots) {
                      const std::complex<double> z = eval_expr(r.value);
                      if (std::abs(z.imag()) < 1e-9) {
                          ++reals;
                          if (std::abs(z.real() - 1.1673) < 1e-3) {
                              found_real = true;
                          }
                      }
                  }
                  t.expect(reals == 1, "one real eigenvalue");
                  t.expect(found_real, "real root ~ 1.1673");
                  t.expect(all_roots_vanish(p, roots, 1e-6),
                           "every numeric eigenvalue back-substitutes to ~0");
              })
        .test("mixed_exact_and_numeric",
              [](TestContext& t) {
                  // (x-2)(x^5 - x - 1): rational 2 exact + five numeric eigenvalues.
                  const RationalPoly p = mul(ipoly({-2, 1}), ipoly({-1, -1, 0, 0, 0, 1}));
                  auto roots = solve_poly(p).value();
                  t.expect(roots.size() == 6, "six roots");
                  t.expect(has_value(roots, Expr::integer(2), true), "rational root 2 exact");
                  const std::size_t numeric =
                      static_cast<std::size_t>(std::ranges::count_if(
                          roots, [](const Root& r) { return !r.exact; }));
                  t.expect(numeric == 5, "five numeric eigenvalue roots");
                  t.expect(all_roots_vanish(p, roots, 1e-6), "all six back-substitute to ~0");
              })
        .test("degenerate_inputs",
              [](TestContext& t) {
                  // zero polynomial: every value is a root -> domain_error.
                  t.expect(solve_poly(RationalPoly{}).error() == MathError::domain_error,
                           "zero polynomial is a domain error");
                  // nonzero constant: no roots.
                  t.expect(solve_poly(ipoly({5})).value().empty(), "nonzero constant has no roots");
              })
        .run();
}
