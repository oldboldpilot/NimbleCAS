// Tests for nimblecas.execdoc: the executable document engine (Live Notebooks).
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.latex;
import nimblecas.execdoc;
import nimblecas.testing;

using nimblecas::Expr;
using nimblecas::MathError;
using nimblecas::simplify;
using nimblecas::to_latex;
using nimblecas::execdoc::CellResult;
using nimblecas::execdoc::DocumentCache;
using nimblecas::execdoc::run_cells;
using nimblecas::execdoc::run_document;
using nimblecas::execdoc::Session;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// The canonical (simplified) form of an expected expression, for structural compare.
[[nodiscard]] auto canon(const Expr& e) -> Expr {
    auto s = simplify(e);
    return s.has_value() ? *s : e;
}

[[nodiscard]] auto contains(std::string_view haystack, std::string_view needle) -> bool {
    return haystack.find(needle) != std::string_view::npos;
}

}  // namespace

auto main() -> int {
    const Expr x = Expr::symbol("x");

    return TestSuite("nimblecas.execdoc")
        .test("single_cell_value_and_html",
              [&](TestContext& t) {
                  // A cell that binds x = 2 then evaluates x^2 + 1 -> 5.
                  Session s;
                  auto r1 = s.execute_cell("x = 2");
                  auto r2 = s.execute_cell("x^2 + 1");
                  t.expect(r1.has_value() && r2.has_value(), "cells execute without hard failure");
                  if (r2.has_value()) {
                      t.expect(!r2->error.has_value(), "x^2 + 1 has no cell error");
                      t.expect(r2->value.has_value() &&
                                   r2->value->is_equivalent_to(Expr::integer(5)),
                               "x^2 + 1 with x=2 -> 5");
                  }
                  // The rendered document contains the result's LaTeX in \( ... \).
                  const std::string doc =
                      "# Title\n\n```nimblecas\nx = 2\nx^2 + 1\n```\n";
                  auto html = run_document(doc);
                  t.expect(html.has_value(), "run_document succeeds");
                  if (html.has_value()) {
                      const std::string frag = "\\(" + to_latex(Expr::integer(5)) + "\\)";
                      t.expect(contains(*html, frag),
                               std::format("html contains {}", frag));
                  }
              })
        .test("cross_cell_bindings_persist",
              [&](TestContext& t) {
                  Session s;
                  auto r1 = s.execute_cell("a = 3");
                  auto r2 = s.execute_cell("a + 4");
                  t.expect(r1.has_value() && r2.has_value(), "cells execute");
                  if (r2.has_value()) {
                      t.expect(r2->value.has_value() &&
                                   r2->value->is_equivalent_to(Expr::integer(7)),
                               "a=3 in one cell, a+4 in a later cell -> 7");
                  }
                  t.expect(s.binding("a").has_value(), "binding 'a' is visible in the session");
              })
        .test("builtin_diff",
              [&](TestContext& t) {
                  Session s;
                  auto r = s.execute_cell("diff(x^3, x)");
                  t.expect(r.has_value(), "diff cell executes");
                  if (r.has_value()) {
                      const Expr expected =
                          canon(Expr::product({Expr::integer(3), x.pow(Expr::integer(2))}));
                      t.expect(!r->error.has_value(), "diff has no cell error");
                      t.expect(r->value.has_value() && r->value->is_equivalent_to(expected),
                               std::format("diff(x^3, x) -> 3 x^2 (got {})",
                                           r->value ? r->value->to_string() : "<none>"));
                  }
              })
        .test("builtin_wrong_arity_is_captured",
              [&](TestContext& t) {
                  // diff needs a differentiation variable; a single argument is an error
                  // that is captured, not thrown.
                  Session s;
                  auto r = s.execute_cell("diff(x^3)");
                  t.expect(r.has_value(), "cell returns a value-carrying result");
                  if (r.has_value()) {
                      t.expect(r->error.has_value(), "diff(x^3) records a cell error");
                  }
              })
        .test("incremental_caching_reuses_and_invalidates",
              [&](TestContext& t) {
                  const std::string doc =
                      "```nimblecas\na = 3\n```\n"
                      "```nimblecas\nb = a + 4\n```\n"
                      "```nimblecas\nc = 10\n```\n";
                  DocumentCache cache;

                  auto first = run_cells(doc, cache);
                  t.expect(first.has_value() && first->size() == 3, "3 cells on first run");
                  if (first.has_value()) {
                      const bool none_cached = std::ranges::none_of(
                          *first, [](const CellResult& r) { return r.from_cache; });
                      t.expect(none_cached, "nothing is from_cache on the first run");
                  }

                  auto second = run_cells(doc, cache);
                  t.expect(second.has_value(), "second run succeeds");
                  if (second.has_value()) {
                      const bool all_cached = std::ranges::all_of(
                          *second, [](const CellResult& r) { return r.from_cache; });
                      t.expect(all_cached, "every cell is from_cache on an identical re-run");
                  }

                  // Change the first cell: it and the downstream cell that reads `a` must
                  // re-execute; the independent `c = 10` stays cached.
                  const std::string doc2 =
                      "```nimblecas\na = 5\n```\n"
                      "```nimblecas\nb = a + 4\n```\n"
                      "```nimblecas\nc = 10\n```\n";
                  auto third = run_cells(doc2, cache);
                  t.expect(third.has_value() && third->size() == 3, "3 cells on the changed run");
                  if (third.has_value()) {
                      const auto& cells = *third;
                      t.expect(!cells[0].from_cache, "changed cell 'a' re-executes");
                      t.expect(cells[0].value.has_value() &&
                                   cells[0].value->is_equivalent_to(Expr::integer(5)),
                               "a -> 5");
                      t.expect(!cells[1].from_cache, "downstream cell 'b' re-executes");
                      t.expect(cells[1].value.has_value() &&
                                   cells[1].value->is_equivalent_to(Expr::integer(9)),
                               "b = a + 4 -> 9");
                      t.expect(cells[2].from_cache, "independent cell 'c' stays cached");
                      t.expect(cells[2].value.has_value() &&
                                   cells[2].value->is_equivalent_to(Expr::integer(10)),
                               "c -> 10");
                  }
              })
        .test("cache_invalidates_on_transitive_symbol_change",
              [&](TestContext& t) {
                  // Regression (cache-soundness): `f` is stored as the unevaluated tree x+1
                  // (x unbound at definition), so a later cell whose source is just `f`
                  // TRANSITIVELY depends on x even though x never appears in that source. A
                  // change to x upstream must invalidate the cached `f` cell, not serve a stale
                  // result.
                  const std::string doc1 =
                      "```nimblecas\nf = x + 1\n```\n"
                      "```nimblecas\nx = 5\n```\n"
                      "```nimblecas\nf\n```\n";
                  DocumentCache cache;
                  auto first = run_cells(doc1, cache);
                  t.expect(first.has_value() && first->size() == 3, "3 cells first run");
                  if (first.has_value()) {
                      t.expect((*first)[2].value.has_value() &&
                                   (*first)[2].value->is_equivalent_to(Expr::integer(6)),
                               "f resolves transitively to 6");
                  }
                  // Change ONLY x; the `f` cell's source is unchanged but it must re-execute.
                  const std::string doc2 =
                      "```nimblecas\nf = x + 1\n```\n"
                      "```nimblecas\nx = 9\n```\n"
                      "```nimblecas\nf\n```\n";
                  auto second = run_cells(doc2, cache);
                  t.expect(second.has_value() && second->size() == 3, "3 cells second run");
                  if (second.has_value()) {
                      t.expect(!(*second)[2].from_cache,
                               "transitively-dependent cell re-executes (no stale hit)");
                      t.expect((*second)[2].value.has_value() &&
                                   (*second)[2].value->is_equivalent_to(Expr::integer(10)),
                               "f -> 10 after x changes to 9 (not the stale 6)");
                  }
              })
        .test("cell_error_does_not_abort_document",
              [&](TestContext& t) {
                  // A malformed middle cell must be captured; the cells around it still
                  // execute and the document still renders.
                  const std::string doc =
                      "```nimblecas\nx = 2\n```\n"
                      "```nimblecas\n)(\n```\n"       // reader rejects this
                      "```nimblecas\nx + 1\n```\n";
                  auto cells = run_cells(doc);
                  t.expect(cells.has_value() && cells->size() == 3, "3 cells parsed");
                  if (cells.has_value()) {
                      const auto& c = *cells;
                      t.expect(!c[0].error.has_value(), "first cell fine");
                      t.expect(c[1].error.has_value(), "malformed cell records an error");
                      t.expect(!c[2].error.has_value() && c[2].value.has_value() &&
                                   c[2].value->is_equivalent_to(Expr::integer(3)),
                               "cell after the error still evaluates x + 1 -> 3");
                  }
                  // The whole document still renders to a string (no crash / no abort).
                  auto html = run_document(doc);
                  t.expect(html.has_value() && contains(*html, "ncas-error"),
                           "document renders with an inline error box");
              })
        .test("prose_passthrough_and_html_escaping",
              [&](TestContext& t) {
                  const std::string doc =
                      "Compare 3 < 5 & 4 > 2 in <b>prose</b>\n\n"
                      "```nimblecas\nx = 1\n```\n";
                  auto html = run_document(doc);
                  t.expect(html.has_value(), "run_document succeeds");
                  if (html.has_value()) {
                      t.expect(contains(*html, "&lt;") && contains(*html, "&amp;") &&
                                   contains(*html, "&gt;"),
                               "prose metacharacters are HTML-escaped");
                      t.expect(!contains(*html, "<b>prose</b>"),
                               "raw prose HTML is not passed through unescaped");
                  }
              })
        .run();
}
