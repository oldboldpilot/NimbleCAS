// Tests for nimblecas.svgplot: deterministic standalone SVG-string generation.
// @author Olumuyiwa Oluwasanmi
//
// The suite pins the exact data->pixel coordinate mapping (hand-computed on a tiny
// 100x100 canvas), the structural shape of the emitted document (<svg ...> ...
// </svg>, xmlns, width/height, <polyline>/<circle>, axis elements), the flat-series
// mid-height guard, the function-sampling drop of non-finite values, XML escaping of
// titles, and every domain_error path. Because coordinates are formatted with a fixed
// 3-decimal precision, the expected strings are exact and locale-independent.

import std;
import nimblecas.core;
import nimblecas.svgplot;
import nimblecas.testing;

using nimblecas::MathError;
using nimblecas::PlotOptions;
using nimblecas::plot_function;
using nimblecas::plot_line;
using nimblecas::plot_scatter;
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

// A tiny 100x100 canvas with a 10px margin: the mapping sends x in [0,2] and y in
// [0,1] onto integer pixel gridlines, so every expected coordinate is exact.
[[nodiscard]] auto tiny() -> PlotOptions {
    PlotOptions o;
    o.width = 100;
    o.height = 100;
    o.margin = 10;
    return o;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.svgplot")
        .test("line_coordinate_mapping_is_exact",
              [](TestContext& t) {
                  // xs={0,1,2}, ys={0,1,0}: xmin=0,xmax=2,ymin=0,ymax=1.
                  //   x=0 -> px=10, x=1 -> px=50, x=2 -> px=90.
                  //   y=0 -> py=90 (bottom, flipped), y=1 -> py=10 (top).
                  std::vector<double> xs{0.0, 1.0, 2.0};
                  std::vector<double> ys{0.0, 1.0, 0.0};
                  auto r = plot_line(xs, ys, tiny());
                  t.expect(r.has_value(), "plot_line succeeds on valid tiny series");
                  const std::string& s = r.value();
                  t.expect(has(s, "points=\"10.000,90.000 50.000,10.000 90.000,90.000\""),
                           "polyline points carry the exact hand-computed pixels");
              })
        .test("document_shape_and_attributes",
              [](TestContext& t) {
                  std::vector<double> xs{0.0, 1.0, 2.0};
                  std::vector<double> ys{0.0, 1.0, 0.0};
                  auto r = plot_line(xs, ys, tiny());
                  t.expect(r.has_value(), "plot_line succeeds");
                  const std::string& s = r.value();
                  t.expect(s.starts_with("<svg"), "output starts with <svg");
                  t.expect(s.ends_with("</svg>"), "output ends with </svg>");
                  t.expect(has(s, "xmlns=\"http://www.w3.org/2000/svg\""),
                           "carries the SVG namespace");
                  t.expect(has(s, "width=\"100\""), "canvas width attribute");
                  t.expect(has(s, "height=\"100\""), "canvas height attribute");
                  t.expect(has(s, "viewBox=\"0 0 100 100\""), "viewBox matches canvas");
                  t.expect(has(s, "<polyline"), "contains a polyline element");
              })
        .test("axes_on_emit_frame_and_lines",
              [](TestContext& t) {
                  std::vector<double> xs{0.0, 1.0, 2.0};
                  std::vector<double> ys{0.0, 1.0, 0.0};
                  auto r = plot_line(xs, ys, tiny());  // axes default = true
                  t.expect(r.has_value(), "plot_line succeeds");
                  const std::string& s = r.value();
                  t.expect(has(s, "<rect"), "axes on -> framing rect present");
                  t.expect_eq(count(s, "<line"), std::size_t{2},
                              "axes on -> two axis lines (x and y)");
              })
        .test("axes_off_suppresses_axis_elements",
              [](TestContext& t) {
                  PlotOptions o = tiny();
                  o.axes = false;
                  std::vector<double> xs{0.0, 1.0, 2.0};
                  std::vector<double> ys{0.0, 1.0, 0.0};
                  auto r = plot_line(xs, ys, o);
                  t.expect(r.has_value(), "plot_line succeeds with axes off");
                  const std::string& s = r.value();
                  t.expect(!has(s, "<rect"), "axes off -> no framing rect");
                  t.expect(!has(s, "<line"), "axes off -> no axis lines");
                  t.expect(has(s, "<polyline"), "the data polyline is still emitted");
              })
        .test("horizontal_line_renders_at_mid_height",
              [](TestContext& t) {
                  // All ys equal -> ymin==ymax guard pads symmetrically so the flat
                  // series lands at the vertical centre (py=50 on a 100px/10-margin
                  // canvas), never dividing by zero.
                  std::vector<double> xs{0.0, 1.0, 2.0};
                  std::vector<double> ys{5.0, 5.0, 5.0};
                  auto r = plot_line(xs, ys, tiny());
                  t.expect(r.has_value(), "flat series is a valid plot");
                  const std::string& s = r.value();
                  t.expect(has(s, "points=\"10.000,50.000 50.000,50.000 90.000,50.000\""),
                           "flat line maps to mid-height py=50 at every point");
                  t.expect(!has(s, "nan") && !has(s, "inf"),
                           "no non-finite coordinates leak into the output");
              })
        .test("scatter_emits_one_circle_per_point",
              [](TestContext& t) {
                  std::vector<double> xs{0.0, 1.0, 2.0};
                  std::vector<double> ys{0.0, 1.0, 0.0};
                  auto r = plot_scatter(xs, ys, tiny());
                  t.expect(r.has_value(), "plot_scatter succeeds");
                  const std::string& s = r.value();
                  t.expect_eq(count(s, "<circle"), std::size_t{3},
                              "one <circle> per data point");
                  // Centres reuse the same mapping as the line plot.
                  t.expect(has(s, "cx=\"10.000\" cy=\"90.000\""),
                           "first circle centred at the mapped (0,0)");
                  t.expect(has(s, "cx=\"90.000\" cy=\"90.000\""),
                           "last circle centred at the mapped (2,0)");
              })
        .test("function_samples_expected_count",
              [](TestContext& t) {
                  // f(x)=x on [0,4], 5 samples -> 5 vertices -> 5 "px,py" pairs, so 5
                  // commas in the points list.
                  auto r = plot_function([](double x) { return x; }, 0.0, 4.0, 5, tiny());
                  t.expect(r.has_value(), "plot_function succeeds");
                  const std::string& s = r.value();
                  t.expect(has(s, "<polyline"), "delegates to a polyline");
                  t.expect_eq(count(s, ","), std::size_t{5},
                              "5 samples -> 5 coordinate pairs");
              })
        .test("function_drops_non_finite_samples",
              [](TestContext& t) {
                  // f(x)=1/x sampled at {-2,-1,0,1,2}: x=0 yields inf and is dropped,
                  // leaving 4 finite vertices.
                  auto r = plot_function([](double x) { return 1.0 / x; }, -2.0, 2.0, 5,
                                         tiny());
                  t.expect(r.has_value(), "plot_function succeeds after dropping inf");
                  const std::string& s = r.value();
                  t.expect_eq(count(s, ","), std::size_t{4},
                              "the non-finite sample at x=0 is dropped");
                  t.expect(!has(s, "inf") && !has(s, "nan"),
                           "no non-finite value survives into the SVG");
              })
        .test("function_all_dropped_is_domain_error",
              [](TestContext& t) {
                  // Every sample non-finite -> empty data -> plot_line domain_error.
                  const double nan = std::numeric_limits<double>::quiet_NaN();
                  auto r = plot_function([nan](double) { return nan; }, 0.0, 1.0, 4, tiny());
                  t.expect(!r.has_value(), "all-non-finite samples fail");
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "reported as domain_error");
              })
        .test("title_is_xml_escaped",
              [](TestContext& t) {
                  PlotOptions o = tiny();
                  o.title = "a & b < c > \"d\"";
                  std::vector<double> xs{0.0, 1.0};
                  std::vector<double> ys{0.0, 1.0};
                  auto r = plot_line(xs, ys, o);
                  t.expect(r.has_value(), "titled plot succeeds");
                  const std::string& s = r.value();
                  t.expect(has(s, "&amp;"), "ampersand escaped");
                  t.expect(has(s, "&lt;"), "less-than escaped");
                  t.expect(has(s, "&gt;"), "greater-than escaped");
                  t.expect(has(s, "&quot;"), "double-quote escaped");
                  t.expect(has(s, "<title>"), "accessible <title> element present");
                  // The raw unescaped ampersand text must not appear.
                  t.expect(!has(s, "a & b"), "no raw unescaped ampersand leaks through");
              })
        .test("domain_error_length_mismatch",
              [](TestContext& t) {
                  std::vector<double> xs{0.0, 1.0, 2.0};
                  std::vector<double> ys{0.0, 1.0};
                  auto r = plot_line(xs, ys, tiny());
                  t.expect(!r.has_value(), "mismatched xs/ys lengths fail");
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "length mismatch -> domain_error");
              })
        .test("domain_error_empty_data",
              [](TestContext& t) {
                  std::vector<double> xs{};
                  std::vector<double> ys{};
                  auto r = plot_line(xs, ys, tiny());
                  t.expect(!r.has_value(), "empty data fails");
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "empty data -> domain_error");
              })
        .test("domain_error_non_positive_dimensions",
              [](TestContext& t) {
                  std::vector<double> xs{0.0, 1.0};
                  std::vector<double> ys{0.0, 1.0};
                  PlotOptions w0 = tiny();
                  w0.width = 0;
                  auto r0 = plot_line(xs, ys, w0);
                  t.expect(!r0.has_value() && r0.error() == MathError::domain_error,
                           "width <= 0 -> domain_error");
                  PlotOptions h0 = tiny();
                  h0.height = -5;
                  auto r1 = plot_line(xs, ys, h0);
                  t.expect(!r1.has_value() && r1.error() == MathError::domain_error,
                           "height <= 0 -> domain_error");
              })
        .test("domain_error_margin_leaves_no_plot_area",
              [](TestContext& t) {
                  std::vector<double> xs{0.0, 1.0};
                  std::vector<double> ys{0.0, 1.0};
                  PlotOptions o = tiny();
                  o.margin = 60;  // 2*60 >= 100 -> no inner rectangle
                  auto r = plot_line(xs, ys, o);
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "oversized margin -> domain_error");
              })
        .test("domain_error_function_bad_samples_and_domain",
              [](TestContext& t) {
                  auto few = plot_function([](double x) { return x; }, 0.0, 1.0, 1, tiny());
                  t.expect(!few.has_value() && few.error() == MathError::domain_error,
                           "samples < 2 -> domain_error");
                  auto inv = plot_function([](double x) { return x; }, 1.0, 1.0, 5, tiny());
                  t.expect(!inv.has_value() && inv.error() == MathError::domain_error,
                           "xmin >= xmax -> domain_error");
              })
        .run();
}
