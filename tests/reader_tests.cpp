// Tests for nimblecas.reader: the text -> Expr parser (eval surface).
// @author Olumuyiwa Oluwasanmi
//
// The key correctness property is the ROUND-TRIP: for a hand-built Expr `e`,
// parse(e.to_string()) must be structurally equivalent to `e`. This proves the parser
// accepts the printer's own language. Reals are excluded from the round-trip set: the
// reader is exact and never emits Expr::real, so a real leaf's decimal rendering
// re-parses to an exact rational instead (documented in reader.cppm).

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.reader;
import nimblecas.testing;

using nimblecas::Expr;
using nimblecas::MathError;
using nimblecas::parse;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Assert parse(text) succeeds and is structurally equivalent to `expected`.
auto parses_to(TestContext& t, std::string_view text, const Expr& expected) -> void {
    auto got = parse(text);
    if (!got.has_value()) {
        t.expect(false, std::format("parse(\"{}\"): unexpected error", text));
        return;
    }
    t.expect(got->is_equivalent_to(expected),
             std::format("parse(\"{}\") -> {} (expected {})", text, got->to_string(),
                         expected.to_string()));
}

// Assert parse(text) fails with the given MathError.
auto rejects(TestContext& t, std::string_view text, MathError err) -> void {
    auto got = parse(text);
    if (got.has_value()) {
        t.expect(false, std::format("parse(\"{}\"): expected error, got {}", text,
                                    got->to_string()));
        return;
    }
    t.expect(got.error() == err, std::format("parse(\"{}\") -> wrong MathError", text));
}

// Assert parse(e.to_string()) is structurally equivalent to e.
auto round_trips(TestContext& t, const Expr& e) -> void {
    const std::string printed = e.to_string();
    auto got = parse(printed);
    if (!got.has_value()) {
        t.expect(false, std::format("round-trip parse(\"{}\") errored", printed));
        return;
    }
    t.expect(got->is_equivalent_to(e),
             std::format("round-trip: \"{}\" -> {} (expected {})", printed, got->to_string(),
                         e.to_string()));
}

// Helper: (-1) * e, the reader's lowering of a non-literal unary minus / `a - b`.
auto neg(const Expr& e) -> Expr { return Expr::product({Expr::integer(-1), e}); }

}  // namespace

auto main() -> int {
    const auto x = Expr::symbol("x");
    const auto y = Expr::symbol("y");
    const auto a = Expr::symbol("a");
    const auto b = Expr::symbol("b");
    const auto c = Expr::symbol("c");
    const auto i2 = Expr::integer(2);
    const auto i3 = Expr::integer(3);

    return TestSuite("nimblecas.reader")
        .test("literals_and_symbols",
              [&](TestContext& t) {
                  parses_to(t, "42", Expr::integer(42));
                  parses_to(t, "x", x);
                  parses_to(t, "foo_bar1", Expr::symbol("foo_bar1"));
                  parses_to(t, "3/4", Expr::rational(3, 4).value());
                  parses_to(t, "6/4", Expr::rational(3, 2).value());  // canonicalised
                  parses_to(t, "1.5", Expr::rational(3, 2).value());  // decimal -> exact
                  parses_to(t, "0.25", Expr::rational(1, 4).value());
                  parses_to(t, "-5", Expr::integer(-5));              // literal fold
                  parses_to(t, "-3/4", Expr::rational(-3, 4).value());
              })
        .test("precedence_and_associativity",
              [&](TestContext& t) {
                  // + is lowest, * binds tighter: 1 + (2*3)
                  parses_to(t, "1 + 2*3", Expr::sum({Expr::integer(1), Expr::product({i2, i3})}));
                  // ^ is right associative: 2^(3^2)
                  parses_to(t, "2^3^2", Expr::power(i2, Expr::power(i3, i2)));
                  // unary minus is looser than ^: -(x^2)
                  parses_to(t, "-x^2", neg(Expr::power(x, i2)));
                  // binary minus is left associative: (a-b)-c
                  parses_to(t, "a - b - c",
                            Expr::sum({Expr::sum({a, neg(b)}), neg(c)}));
                  // division of non-literals: x * 2^(-1)
                  parses_to(t, "x/2", Expr::product({x, Expr::power(i2, Expr::integer(-1))}));
              })
        .test("grouping",
              [&](TestContext& t) {
                  parses_to(t, "(1+2)*3",
                            Expr::product({Expr::sum({Expr::integer(1), i2}), i3}));
                  parses_to(t, "((x))", x);
                  parses_to(t, "2 * (x + y)", Expr::product({i2, Expr::sum({x, y})}));
              })
        .test("function_calls",
              [&](TestContext& t) {
                  parses_to(t, "sin(x) + cos(x)",
                            Expr::sum({Expr::apply("sin", {x}), Expr::apply("cos", {x})}));
                  parses_to(t, "atan2(y, x)", Expr::apply("atan2", {y, x}));
                  parses_to(t, "f(g(x))", Expr::apply("f", {Expr::apply("g", {x})}));
                  parses_to(t, "log(x*y)", Expr::apply("log", {Expr::product({x, y})}));
              })
        .test("round_trip_printer_language",
              [&](TestContext& t) {
                  // Reals are still excluded (the reader is exact and never emits Expr::real).
                  // The former printer caveats (rational/negative/power bases under `^`, and
                  // rational exponents) are now handled by explicit parenthesization in
                  // Expr::to_string and are exercised in the dedicated test below.
                  round_trips(t, x);
                  round_trips(t, Expr::integer(42));
                  round_trips(t, Expr::integer(-7));
                  round_trips(t, Expr::rational(3, 4).value());
                  round_trips(t, Expr::rational(-3, 4).value());
                  round_trips(t, Expr::sum({x, y}));
                  round_trips(t, Expr::sum({x, Expr::product({i2, y})}));
                  round_trips(t, Expr::product({Expr::integer(-1), x}));       // "(-1 * x)"
                  round_trips(t, Expr::power(x, i2));                          // "x^2"
                  round_trips(t, Expr::power(x, Expr::integer(-2)));          // "x^-2"
                  round_trips(t, Expr::power(Expr::sum({x, y}), i2));         // "(x + y)^2"
                  round_trips(t, Expr::product({x, Expr::power(y, Expr::integer(-1))}));  // x/y
                  round_trips(t, Expr::apply("sin", {x}));
                  round_trips(t, Expr::apply("atan2", {y, x}));
                  round_trips(t, Expr::apply("f", {Expr::apply("g", {x})}));  // nested calls
                  round_trips(t, Expr::sum({x, Expr::product({Expr::integer(-1), y})}));  // a - b
              })
        .test("round_trip_power_operands_now_parenthesized",
              [&](TestContext& t) {
                  // Regression for the printer fix (task #16): these forms previously printed
                  // ambiguously and re-parsed to a DIFFERENT tree; to_string now parenthesizes
                  // the power operands so they round-trip.
                  round_trips(t, Expr::power(Expr::rational(1, 2).value(), x));   // (1/2)^x
                  round_trips(t, Expr::power(Expr::integer(-2), x));             // (-2)^x
                  round_trips(t, Expr::power(Expr::power(x, y), Expr::symbol("z")));  // (x^y)^z
                  round_trips(t, Expr::power(x, Expr::rational(1, 2).value()));   // x^(1/2)
                  round_trips(t, Expr::power(x, Expr::rational(-3, 2).value()));  // x^(-3/2)
              })
        .test("malformed_inputs_rejected",
              [&](TestContext& t) {
                  rejects(t, "", MathError::syntax_error);        // empty
                  rejects(t, "   ", MathError::syntax_error);     // whitespace only
                  rejects(t, "1 +", MathError::syntax_error);     // missing operand
                  rejects(t, "(1+2", MathError::syntax_error);    // unbalanced '('
                  rejects(t, "1+2)", MathError::syntax_error);    // unbalanced ')'
                  rejects(t, "1 2", MathError::syntax_error);     // trailing garbage
                  rejects(t, "sin(", MathError::syntax_error);    // unterminated call
                  rejects(t, "sin(x", MathError::syntax_error);   // unterminated call
                  rejects(t, "@", MathError::syntax_error);       // unknown character
                  rejects(t, "*x", MathError::syntax_error);      // leading binary op
                  rejects(t, "1.", MathError::syntax_error);      // trailing dot
                  rejects(t, "f(x,)", MathError::syntax_error);   // dangling comma
              })
        .test("division_by_zero_is_honest",
              [&](TestContext& t) {
                  // Two integer literals form an exact rational; a zero denominator is
                  // reported rather than silently turned into a 0^(-1) tree.
                  rejects(t, "3/0", MathError::division_by_zero);
              })
        .test("integer_overflow_is_honest",
              [&](TestContext& t) {
                  rejects(t, "99999999999999999999999999", MathError::overflow);
              })
        .test("decimal_trailing_zeros_do_not_overflow",
              [&](TestContext& t) {
                  // Regression: 19 fractional digits denote exactly 1/2; the reduced value
                  // fits in int64, so trailing zeros must be stripped rather than overflow
                  // at 10^19.
                  parses_to(t, "0.5000000000000000000", Expr::rational(1, 2).value());
                  parses_to(t, "2.500", Expr::rational(5, 2).value());
                  parses_to(t, "3.000", Expr::integer(3));
              })
        .test("zero_decimal_denominator_is_division_by_zero",
              [&](TestContext& t) {
                  // Regression: a zero denominator in ANY numeric form is honest, not a
                  // silent 0^(-1) tree — matching "3/0".
                  rejects(t, "3/0.0", MathError::division_by_zero);
                  rejects(t, "x/0.0", MathError::division_by_zero);
                  rejects(t, "5/0.00", MathError::division_by_zero);
              })
        .test("pathological_nesting_is_rejected_not_crashed",
              [&](TestContext& t) {
                  // Regression: unbounded recursion through nested parens / unary minus would
                  // overflow the native stack. It must return a syntax_error, never crash.
                  const std::string deep_parens =
                      std::string(2000, '(') + "1" + std::string(2000, ')');
                  rejects(t, deep_parens, MathError::syntax_error);
                  const std::string deep_minus = std::string(2000, '-') + "1";
                  rejects(t, deep_minus, MathError::syntax_error);
              })
        .run();
}
