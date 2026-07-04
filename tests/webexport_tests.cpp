// Tests for nimblecas.webexport: deterministic JSON export of CAS plot data and
// documents.
// @author Olumuyiwa Oluwasanmi
//
// The suite pins the fixed JSON contract: the PlotSpec object keys/shape, that raw
// data points serialize losslessly (a known line yields an exact "points" substring —
// no pixel mapping, unlike svgplot), that JSON string escaping handles quotes and
// backslashes, that non-finite input is rejected with domain_error, that function
// sampling drops non-finite results, and that a Document assembles prose+math+plot
// blocks with all three kinds present. Because numbers are formatted with a fixed
// 3-decimal precision, the expected strings are exact and locale-independent.

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.webexport;
import nimblecas.testing;

using nimblecas::Block;
using nimblecas::document_to_json;
using nimblecas::Expr;
using nimblecas::math_block;
using nimblecas::MathError;
using nimblecas::plot_block;
using nimblecas::plot_spec_function;
using nimblecas::plot_spec_line;
using nimblecas::plot_spec_multi;
using nimblecas::plot_spec_scatter;
using nimblecas::PlotSpecOptions;
using nimblecas::prose_block;
using nimblecas::Series;
using nimblecas::SeriesKind;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Substring presence.
[[nodiscard]] auto has(const std::string& s, std::string_view sub) -> bool {
    return s.find(sub) != std::string::npos;
}

// Count non-overlapping occurrences of `sub` in `s`.
[[nodiscard]] auto count(const std::string& s, std::string_view sub) -> std::size_t {
    if (sub.empty()) {
        return 0;
    }
    std::size_t n = 0;
    for (std::size_t pos = s.find(sub); pos != std::string::npos;
         pos = s.find(sub, pos + sub.size())) {
        ++n;
    }
    return n;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.webexport")
        .test("line_points_serialize_losslessly",
              [](TestContext& t) {
                  // Points are the raw doubles the CAS produced, fixed 3-decimals, NOT
                  // pixel-mapped: xs={0,1,2}, ys={0,1,0} -> the exact pairs below.
                  std::vector<double> xs{0.0, 1.0, 2.0};
                  std::vector<double> ys{0.0, 1.0, 0.0};
                  auto r = plot_spec_line(xs, ys, PlotSpecOptions{});
                  t.expect(r.has_value(), "plot_spec_line succeeds on valid series");
                  const std::string& s = r.value();
                  t.expect(
                      has(s, "\"points\":[[0.000,0.000],[1.000,1.000],[2.000,0.000]]"),
                      "points carry the exact raw data coordinates");
              })
        .test("plotspec_shape_and_keys",
              [](TestContext& t) {
                  std::vector<double> xs{0.0, 1.0};
                  std::vector<double> ys{0.0, 1.0};
                  PlotSpecOptions o;
                  o.title = "T";
                  o.xLabel = "X";
                  o.yLabel = "Y";
                  auto r = plot_spec_line(xs, ys, o);
                  t.expect(r.has_value(), "plot_spec_line succeeds");
                  const std::string& s = r.value();
                  t.expect(s.starts_with("{"), "output is a JSON object");
                  t.expect(s.ends_with("}"), "output object is closed");
                  t.expect(has(s, "\"type\":\"plot\""), "type is plot");
                  t.expect(has(s, "\"title\":\"T\""), "title present");
                  t.expect(has(s, "\"xLabel\":\"X\""), "xLabel present");
                  t.expect(has(s, "\"yLabel\":\"Y\""), "yLabel present");
                  t.expect(has(s, "\"xRange\":null"), "unset xRange serializes as null");
                  t.expect(has(s, "\"yRange\":null"), "unset yRange serializes as null");
                  t.expect(has(s, "\"series\":["), "series array present");
                  t.expect(has(s, "\"kind\":\"line\""), "series kind is line");
                  t.expect(has(s, "\"color\":\"#1f77b4\""), "default color emitted");
              })
        .test("ranges_serialize_when_set",
              [](TestContext& t) {
                  std::vector<double> xs{0.0, 1.0};
                  std::vector<double> ys{0.0, 1.0};
                  PlotSpecOptions o;
                  o.xRange = std::pair{0.0, 10.0};
                  o.yRange = std::pair{-1.0, 1.0};
                  auto r = plot_spec_line(xs, ys, o);
                  t.expect(r.has_value(), "plot_spec_line succeeds with ranges");
                  const std::string& s = r.value();
                  t.expect(has(s, "\"xRange\":[0.000,10.000]"), "xRange as [lo,hi]");
                  t.expect(has(s, "\"yRange\":[-1.000,1.000]"), "yRange as [lo,hi]");
              })
        .test("scatter_kind",
              [](TestContext& t) {
                  std::vector<double> xs{0.0, 1.0};
                  std::vector<double> ys{0.0, 1.0};
                  auto r = plot_spec_scatter(xs, ys, PlotSpecOptions{});
                  t.expect(r.has_value(), "plot_spec_scatter succeeds");
                  const std::string& s = r.value();
                  t.expect(has(s, "\"kind\":\"scatter\""), "series kind is scatter");
                  t.expect(!has(s, "\"kind\":\"line\""), "no line kind present");
              })
        .test("function_samples_expected_points",
              [](TestContext& t) {
                  // f(x)=x on [0,4], 5 samples -> 5 exact points.
                  auto r = plot_spec_function([](double x) { return x; }, 0.0, 4.0, 5,
                                              PlotSpecOptions{});
                  t.expect(r.has_value(), "plot_spec_function succeeds");
                  const std::string& s = r.value();
                  t.expect(has(s, "\"points\":[[0.000,0.000],[1.000,1.000],[2.000,2.000],"
                                  "[3.000,3.000],[4.000,4.000]]"),
                           "sampled points are exact");
              })
        .test("function_drops_non_finite_samples",
              [](TestContext& t) {
                  // f(x)=1/x on {-2,-1,0,1,2}: x=0 yields inf and is dropped -> 4 points.
                  auto r = plot_spec_function([](double x) { return 1.0 / x; }, -2.0, 2.0, 5,
                                              PlotSpecOptions{});
                  t.expect(r.has_value(), "succeeds after dropping inf");
                  const std::string& s = r.value();
                  // Each surviving point is one "[x,y]" pair -> 4 opening "[[" ... count
                  // the inner "],[" separators instead: 4 points -> 3 separators.
                  t.expect_eq(count(s, "],["), std::size_t{3},
                              "4 finite points -> 3 separators");
                  t.expect(!has(s, "inf") && !has(s, "nan"),
                           "no non-finite token leaks into the JSON");
              })
        .test("function_all_dropped_is_domain_error",
              [](TestContext& t) {
                  const double nan = std::numeric_limits<double>::quiet_NaN();
                  auto r = plot_spec_function([nan](double) { return nan; }, 0.0, 1.0, 4,
                                              PlotSpecOptions{});
                  t.expect(!r.has_value(), "all-non-finite samples fail");
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "reported as domain_error");
              })
        .test("function_bad_samples_and_domain",
              [](TestContext& t) {
                  auto few = plot_spec_function([](double x) { return x; }, 0.0, 1.0, 1,
                                                PlotSpecOptions{});
                  t.expect(!few.has_value() && few.error() == MathError::domain_error,
                           "samples < 2 -> domain_error");
                  auto inv = plot_spec_function([](double x) { return x; }, 1.0, 1.0, 5,
                                                PlotSpecOptions{});
                  t.expect(!inv.has_value() && inv.error() == MathError::domain_error,
                           "xmin >= xmax -> domain_error");
              })
        .test("string_escaping",
              [](TestContext& t) {
                  // A title with a quote and a backslash must be JSON-escaped.
                  PlotSpecOptions o;
                  o.title = R"(a "q" \ b)";
                  std::vector<double> xs{0.0, 1.0};
                  std::vector<double> ys{0.0, 1.0};
                  auto r = plot_spec_line(xs, ys, o);
                  t.expect(r.has_value(), "titled spec succeeds");
                  const std::string& s = r.value();
                  t.expect(has(s, R"(\")"), "double-quote escaped as backslash-quote");
                  t.expect(has(s, R"(\\)"), "backslash escaped as double-backslash");
                  // The exact escaped title fragment.
                  t.expect(has(s, R"("title":"a \"q\" \\ b")"),
                           "title escaped in full");
              })
        .test("domain_error_length_mismatch",
              [](TestContext& t) {
                  std::vector<double> xs{0.0, 1.0, 2.0};
                  std::vector<double> ys{0.0, 1.0};
                  auto r = plot_spec_line(xs, ys, PlotSpecOptions{});
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "length mismatch -> domain_error");
              })
        .test("domain_error_empty_data",
              [](TestContext& t) {
                  std::vector<double> xs{};
                  std::vector<double> ys{};
                  auto r = plot_spec_line(xs, ys, PlotSpecOptions{});
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "empty data -> domain_error");
              })
        .test("domain_error_non_finite_direct_input",
              [](TestContext& t) {
                  const double nan = std::numeric_limits<double>::quiet_NaN();
                  const double inf = std::numeric_limits<double>::infinity();
                  std::vector<double> xs{0.0, nan};
                  std::vector<double> ys{0.0, 1.0};
                  auto r = plot_spec_line(xs, ys, PlotSpecOptions{});
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "non-finite x -> domain_error");
                  std::vector<double> xs2{0.0, 1.0};
                  std::vector<double> ys2{0.0, inf};
                  auto r2 = plot_spec_scatter(xs2, ys2, PlotSpecOptions{});
                  t.expect(!r2.has_value() && r2.error() == MathError::domain_error,
                           "non-finite y -> domain_error");
              })
        .test("multi_series_builder",
              [](TestContext& t) {
                  Series a;
                  a.kind = SeriesKind::line;
                  a.label = "sin";
                  a.color = "#ff0000";
                  a.xs = {0.0, 1.0};
                  a.ys = {0.0, 1.0};
                  Series b;
                  b.kind = SeriesKind::scatter;
                  b.label = "pts";
                  b.color = "#00ff00";
                  b.xs = {0.0, 1.0};
                  b.ys = {1.0, 0.0};
                  std::array<Series, 2> both{a, b};
                  auto r = plot_spec_multi(both, PlotSpecOptions{});
                  t.expect(r.has_value(), "plot_spec_multi succeeds");
                  const std::string& s = r.value();
                  t.expect_eq(count(s, "\"points\":["), std::size_t{2},
                              "two series each with a points array");
                  t.expect(has(s, "\"label\":\"sin\"") && has(s, "\"label\":\"pts\""),
                           "both series labels present");
                  t.expect(has(s, "\"color\":\"#ff0000\"") &&
                               has(s, "\"color\":\"#00ff00\""),
                           "each series keeps its own colour");
              })
        .test("multi_series_empty_is_domain_error",
              [](TestContext& t) {
                  std::array<Series, 0> none{};
                  auto r = plot_spec_multi(none, PlotSpecOptions{});
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "no series -> domain_error");
              })
        .test("document_assembles_all_three_block_kinds",
              [](TestContext& t) {
                  std::vector<double> xs{0.0, 1.0};
                  std::vector<double> ys{0.0, 1.0};
                  auto spec = plot_spec_line(xs, ys, PlotSpecOptions{});
                  t.expect(spec.has_value(), "inner plot spec built");

                  // A math block sourced from an Expr via to_latex (wires nimblecas.latex).
                  auto x = Expr::symbol("x");
                  auto y = Expr::symbol("y");
                  std::vector<Block> blocks;
                  blocks.push_back(prose_block("Hello, world."));
                  blocks.push_back(math_block(x.add(y)));
                  blocks.push_back(plot_block(spec.value()));

                  auto doc = document_to_json("My Doc", blocks);
                  t.expect(doc.has_value(), "document assembles");
                  const std::string& s = doc.value();
                  t.expect(has(s, "\"type\":\"document\""), "document type");
                  t.expect(has(s, "\"title\":\"My Doc\""), "document title");
                  t.expect(has(s, "\"kind\":\"prose\""), "prose block present");
                  t.expect(has(s, "\"text\":\"Hello, world.\""), "prose text carried");
                  t.expect(has(s, "\"kind\":\"math\""), "math block present");
                  t.expect(has(s, "\"latex\":\"x + y\""),
                           "math latex from to_latex(x + y)");
                  t.expect(has(s, "\"kind\":\"plot\""), "plot block present");
                  // The embedded PlotSpec keeps its own "type":"plot".
                  t.expect(has(s, "\"plot\":{\"type\":\"plot\""),
                           "plot block embeds the PlotSpec object");
              })
        .test("math_block_from_raw_latex",
              [](TestContext& t) {
                  std::vector<Block> blocks;
                  blocks.push_back(math_block(std::string_view{"\\frac{1}{2}"}));
                  auto doc = document_to_json("D", blocks);
                  t.expect(doc.has_value(), "document with raw-latex math block");
                  const std::string& s = doc.value();
                  // Backslashes in the LaTeX are JSON-escaped.
                  t.expect(has(s, R"("latex":"\\frac{1}{2}")"),
                           "raw latex backslashes escaped");
              })
        .test("document_rejects_malformed_plot_block",
              [](TestContext& t) {
                  std::vector<Block> blocks;
                  blocks.push_back(plot_block("not-json"));
                  auto doc = document_to_json("D", blocks);
                  t.expect(!doc.has_value() && doc.error() == MathError::domain_error,
                           "non-object plot payload -> domain_error");
              })
        .run();
}
