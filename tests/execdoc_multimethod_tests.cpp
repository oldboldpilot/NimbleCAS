// Executable-document demonstrations of the two multi-method cross-checks in
// tests/pde_crossmethod_tests.cpp and tests/inteq_crossmethod_tests.cpp, run through
// nimblecas.execdoc (Live Notebooks) to exercise the executable-document feature itself.
// @author Olumuyiwa Oluwasanmi
//
// execdoc's grammar is deliberately small (arithmetic, diff(expr, symbol), simplify(expr));
// it has no way to call the FEM/FDM/ADM/HAM C++ solvers directly. So these two documents do
// NOT re-run those solvers -- they present the solvers' EXACT, hand-verified results as literal
// expressions and use diff/simplify to verify genuine, checkable algebraic facts about them:
//
//   * pde-multimethod: the closed-form linear-diffusion solution u = phi - 2t satisfies the PDE
//     residual u_t - u_xx == 0 exactly (diff is linear here, no power-of-a-sum expansion is
//     needed, so this is safely within simplify's single-pass Cohen ASAE scope).
//   * inteq-multimethod: the manufactured forcing term f = x - x^3/3 satisfies
//     f + x^3/3 - x == 0 (the algebraic identity behind "phi(x) = x solves the integral
//     equation"), and the degree-4 truncated e^x series' residual phi_4' - phi_4 is shown to be
//     EXACTLY the next discarded term, -x^4/24 -- a precise, not hand-wavy, truncation error.
//
// Every cell's rendered value is checked both at the Session level (is_equivalent_to, robust to
// canonical-form ordering) and in run_document()'s rendered HTML (the literal "\(...\)" MathJax
// fragment), so this suite also proves the executable-document feature actually emits LaTeX.

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.latex;
import nimblecas.execdoc;
import nimblecas.testing;

using nimblecas::Expr;
using nimblecas::simplify;
using nimblecas::to_latex;
using nimblecas::execdoc::CellResult;
using nimblecas::execdoc::run_document;
using nimblecas::execdoc::Session;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto canon(const Expr& e) -> Expr {
    auto s = simplify(e);
    return s.has_value() ? *s : e;
}

[[nodiscard]] auto contains(std::string_view haystack, std::string_view needle) -> bool {
    return haystack.find(needle) != std::string_view::npos;
}

[[nodiscard]] auto latex_fragment(const Expr& e) -> std::string {
    return "\\(" + to_latex(e) + "\\)";
}

const std::string pde_doc =
    "# Reaction-diffusion: one PDE, four methods\n\n"
    "We solve u_t = u_xx + u^2 on x in [0,1], with initial datum phi(x) = x - x^2.\n\n"
    "## The linear diffusion piece has an exact closed form\n\n"
    "Since phi is quadratic, phi'' = -2 (a constant), the Cauchy-Kovalevskaya series\n"
    "u(x,t) = phi(x) - 2t terminates exactly: u_t = u_xx. We verify the PDE residual is\n"
    "exactly zero for this closed form.\n\n"
    "```nimblecas\n"
    "u = x - x^2 - 2*t\n"
    "ut = diff(u, t)\n"
    "ux = diff(u, x)\n"
    "uxx = diff(ux, x)\n"
    "residual = simplify(ut - uxx)\n"
    "```\n\n"
    "Finite Elements (fem_p1_solve) independently reproduces the same spatial profile:\n"
    "solving -w'' = 2 with w(0) = w(1) = 0 on a uniform 4-interval mesh gives nodal values\n"
    "3/16, 1/4, 3/16 at x = 1/4, 1/2, 3/4 -- exactly x - x^2 there. The Finite Difference\n"
    "discrete Laplacian applied to those same nodal values gives exactly -2 at every\n"
    "interior node, matching phi'' above (see tests/pde_crossmethod_tests.cpp for the full\n"
    "exact cross-check).\n\n"
    "## The full nonlinear PDE: ADM and HPM agree\n\n"
    "nimblecas.pde's ADM (reaction_diffusion_quadratic) and HPM\n"
    "(solve_nonlinear_evolution_pde_hpm) both solve u_t = u_xx + u^2 and return\n"
    "bit-identical truncated series (the ADM == HPM homotopy equivalence). The\n"
    "hand-verified first-order coefficient is:\n\n"
    "```nimblecas\n"
    "c1 = x^4 - 2*x^3 + x^2 - 2\n"
    "```\n\n"
    "HAM (the convergence-control deformation) is NOT YET implemented for genuine PDEs in\n"
    "nimblecas.pde -- only for ODEs (nimblecas.perturbation) and integral equations\n"
    "(nimblecas.inteq). This is an honest, documented gap, not a silent omission.\n";

const std::string inteq_doc =
    "# A nonlinear Volterra equation: Neumann, ADM, HPM, HAM\n\n"
    "phi(x) = f(x) + integral_0^x phi(t)^2 dt has phi(x) = x as its exact fixed point when\n"
    "f(x) = x - x^3/3 (since integral_0^x t^2 dt = x^3/3). We verify this algebraic\n"
    "identity directly.\n\n"
    "```nimblecas\n"
    "f = x - x^3/3\n"
    "check = simplify(f + x^3/3 - x)\n"
    "```\n\n"
    "For the LINEAR case phi(x) = 1 + integral_0^x phi(t) dt (i.e. phi' = phi, phi(0) = 1),\n"
    "Neumann/Picard/ADM/HPM/HAM(hbar=-1) all return the exact Taylor series of e^x. The\n"
    "degree-4 truncation phi_4 satisfies phi_4' = phi_4 only up to the missing next-order\n"
    "term -- the residual is exactly that term, -x^4/24, a precise measure of the\n"
    "truncation error rather than a vague \"approximately\":\n\n"
    "```nimblecas\n"
    "phi4 = 1 + x + x^2/2 + x^3/6 + x^4/24\n"
    "phi4prime = diff(phi4, x)\n"
    "residual4 = simplify(phi4prime + (-1 - x - x^2/2 - x^3/6 - x^4/24))\n"
    "```\n\n"
    "The full Neumann == ADM == HPM == HAM(hbar=-1) cross-check (bit-identical rational\n"
    "coefficients) and the nonlinear ADM/HPM/HAM convergence toward the exact phi(x) = x are\n"
    "exact-rational tests in tests/inteq_crossmethod_tests.cpp; this document exercises the\n"
    "same mathematics through the executable-document engine's symbolic (diff / simplify)\n"
    "surface.\n";

}  // namespace

auto main() -> int {
    const Expr x = Expr::symbol("x");

    return TestSuite("execdoc_multimethod")
        .test("pde_residual_cell_is_exactly_zero",
              [&](TestContext& t) {
                  Session s;
                  auto r = s.execute_cell("u = x - x^2 - 2*t\n"
                                          "ut = diff(u, t)\n"
                                          "ux = diff(u, x)\n"
                                          "uxx = diff(ux, x)\n"
                                          "residual = simplify(ut - uxx)");
                  t.expect(r.has_value(), "cell executes without hard failure");
                  if (!r) {
                      return;
                  }
                  t.expect(!r->error.has_value(), "no cell error");
                  t.expect(r->value.has_value() && r->value->is_equivalent_to(Expr::integer(0)),
                           "u_t - u_xx == 0 exactly for the closed-form solution");
              })
        .test("pde_document_renders_and_contains_latex",
              [&](TestContext& t) {
                  auto html = run_document(pde_doc);
                  t.expect(html.has_value(), "pde-multimethod document runs");
                  if (!html) {
                      return;
                  }
                  t.expect(!contains(*html, "ncas-error"), "no cell in the document errors");
                  t.expect(contains(*html, latex_fragment(Expr::integer(0))),
                           "the zero PDE residual is rendered as LaTeX in the document");
                  Session s;
                  auto c1 = s.execute_cell("c1 = x^4 - 2*x^3 + x^2 - 2");
                  t.expect(c1.has_value() && c1->value.has_value(), "c1 cell has a value");
                  if (c1 && c1->value) {
                      t.expect(contains(*html, latex_fragment(*c1->value)),
                               "the ADM/HPM first-order coefficient is rendered as LaTeX");
                  }
              })
        .test("inteq_manufactured_identity_cell_is_exactly_zero",
              [&](TestContext& t) {
                  Session s;
                  auto r = s.execute_cell("f = x - x^3/3\ncheck = simplify(f + x^3/3 - x)");
                  t.expect(r.has_value(), "cell executes without hard failure");
                  if (!r) {
                      return;
                  }
                  t.expect(!r->error.has_value(), "no cell error");
                  t.expect(r->value.has_value() && r->value->is_equivalent_to(Expr::integer(0)),
                           "f(x) + x^3/3 - x == 0: f is exactly the manufactured forcing term");
              })
        .test("inteq_truncation_residual_is_exactly_next_term",
              [&](TestContext& t) {
                  Session s;
                  auto r = s.execute_cell("phi4 = 1 + x + x^2/2 + x^3/6 + x^4/24\n"
                                          "phi4prime = diff(phi4, x)\n"
                                          "residual4 = simplify(phi4prime + (-1 - x - x^2/2 - x^3/6 - x^4/24))");
                  t.expect(r.has_value(), "cell executes without hard failure");
                  if (!r) {
                      return;
                  }
                  t.expect(!r->error.has_value(), "no cell error");
                  auto minus_x4_over_24_r = Expr::rational(-1, 24);
                  t.expect(minus_x4_over_24_r.has_value(), "-1/24 constructs");
                  if (!minus_x4_over_24_r || !r->value) {
                      return;
                  }
                  const Expr expected = canon(Expr::product({*minus_x4_over_24_r, x.pow(Expr::integer(4))}));
                  t.expect(r->value->is_equivalent_to(expected),
                           "phi_4' - phi_4 == -x^4/24 exactly (the missing truncated term)");
              })
        .test("inteq_document_renders_and_contains_latex",
              [&](TestContext& t) {
                  auto html = run_document(inteq_doc);
                  t.expect(html.has_value(), "inteq-multimethod document runs");
                  if (!html) {
                      return;
                  }
                  t.expect(!contains(*html, "ncas-error"), "no cell in the document errors");
                  t.expect(contains(*html, latex_fragment(Expr::integer(0))),
                           "the manufactured-identity zero is rendered as LaTeX in the document");
                  auto minus_x4_over_24_r = Expr::rational(-1, 24);
                  t.expect(minus_x4_over_24_r.has_value(), "-1/24 constructs");
                  if (minus_x4_over_24_r) {
                      const Expr expected =
                          canon(Expr::product({*minus_x4_over_24_r, x.pow(Expr::integer(4))}));
                      t.expect(contains(*html, latex_fragment(expected)),
                               "the exact truncation-error residual -x^4/24 is rendered as LaTeX");
                  }
              })
        .run();
}
