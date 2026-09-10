// Tests for nimblecas.logic_parser: the Prolog reader and writer.
// @author Olumuyiwa Oluwasanmi
//
// The reader is checked by asserting the STRUCTURE it produces (via nimblecas::to_string's
// canonical rendering), never by eyeballing source text. The writer is checked by the only
// property worth claiming for it: what it writes must READ BACK as the same term.

import std;
import nimblecas.core;
import nimblecas.logic;
import nimblecas.logic_parser;
import nimblecas.testing;

using nimblecas::apply_substitution;
using nimblecas::atom_of;
using nimblecas::Clause;
using nimblecas::compound_of;
using nimblecas::int_of;
using nimblecas::is_atom;
using nimblecas::is_compound;
using nimblecas::is_int;
using nimblecas::is_var;
using nimblecas::make_atom;
using nimblecas::make_compound;
using nimblecas::make_int;
using nimblecas::make_list;
using nimblecas::var_of;
using nimblecas::make_var;
using nimblecas::MathError;
using nimblecas::Program;
using nimblecas::solve;
using nimblecas::Substitution;
using nimblecas::Term;
using nimblecas::to_string;
using nimblecas::logic_parser::atom_needs_quotes;
using nimblecas::logic_parser::OperatorTable;
using nimblecas::logic_parser::OpType;
using nimblecas::logic_parser::parse_clause;
using nimblecas::logic_parser::parse_program;
using nimblecas::logic_parser::parse_query;
using nimblecas::logic_parser::parse_term;
using nimblecas::logic_parser::quote_atom;
using nimblecas::logic_parser::to_source;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Parses `src` and writes it straight back out. A round-trip assertion is written as
// `roundtrip(s) == expected_source`; the stronger structural form is `reparses_same(s)`.
[[nodiscard]] auto roundtrip(std::string_view src) -> std::string {
    auto t = parse_term(src);
    return t ? to_source(*t) : std::string("<parse-error>");
}

// The property that actually matters: writing a term and reading it back yields the SAME term.
// Comparing structurally rather than comparing strings is what makes this a real check — the
// writer is free to choose spacing and parentheses, it is not free to change the term.
[[nodiscard]] auto reparses_same(std::string_view src) -> bool {
    auto first = parse_term(src);
    if (!first) {
        return false;
    }
    auto second = parse_term(to_source(*first));
    return second.has_value() && *second == *first;
}

// The canonical rendering of a parsed term, or a marker naming the error. Returning a marker
// rather than an empty string keeps a failed parse visible in the assertion message instead of
// silently comparing equal to nothing.
[[nodiscard]] auto shape(std::string_view src) -> std::string {
    auto t = parse_term(src);
    if (!t) {
        return std::format("<error:{}>", static_cast<int>(t.error()));
    }
    return to_string(*t);
}

// The value bound to query variable `name` by answer `s`, rendered.
[[nodiscard]] auto bound(const Substitution& s, std::string_view name) -> std::string {
    return to_string(apply_substitution(s, make_var(std::string(name))));
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.logic_parser")
.test("parse_term_reads_plain_atoms_variables_and_positive_integers", [](TestContext& t) {
    auto r_atom = parse_term("foo");
    t.expect(r_atom.has_value(), "parse_term on plain atom foo succeeds");
    if (r_atom) {
        t.expect(is_atom(*r_atom), "term is an atom");
        t.expect(atom_of(*r_atom).name == "foo", "atom name is foo");
    }

    auto r_var = parse_term("Var_1");
    t.expect(r_var.has_value(), "parse_term on variable Var_1 succeeds");
    if (r_var) {
        t.expect(is_var(*r_var), "term is a variable");
        t.expect(var_of(*r_var).name == "Var_1", "variable name is Var_1");
    }

    auto r_anon = parse_term("_");
    t.expect(r_anon.has_value(), "parse_term on anonymous variable succeeds");
    if (r_anon) {
        t.expect(is_var(*r_anon), "anonymous variable is a variable");
    }

    auto r_int = parse_term("12345");
    t.expect(r_int.has_value(), "parse_term on positive integer succeeds");
    if (r_int) {
        t.expect(is_int(*r_int), "term is an integer");
        t.expect(int_of(*r_int).value == 12345, "integer value is 12345");
    }
})
.test("parse_term_folds_negative_integer_literal_into_signed_int", [](TestContext& t) {
    auto r = parse_term("-42");
    t.expect(r.has_value(), "parse_term on -42 succeeds");
    if (r) {
        t.expect(is_int(*r), "-42 parses as an integer term");
        t.expect(int_of(*r).value == -42, "integer value is exactly -42");
    }
})
.test("parse_term_reads_quoted_atoms_with_spaces_and_escape_sequences", [](TestContext& t) {
    auto r1 = parse_term("'hello world\\n\\t'");
    t.expect(r1.has_value(), "quoted atom with space, newline, and tab parses");
    if (r1) {
        t.expect(is_atom(*r1), "term is an atom");
        t.expect(atom_of(*r1).name == "hello world\n\t", "escapes and spaces are correctly decoded");
    }

    auto r2 = parse_term("'can''t'");
    t.expect(r2.has_value(), "quoted atom with doubled single-quote parses");
    if (r2) {
        t.expect(is_atom(*r2), "term is an atom");
        t.expect(atom_of(*r2).name == "can't", "doubled quote is decoded to single quote");
    }
})
.test("parse_term_supports_radix_and_character_code_integer_literals", [](TestContext& t) {
    auto r_char = parse_term("0'a");
    t.expect(r_char.has_value(), "character code 0'a parses");
    if (r_char) {
        t.expect(is_int(*r_char), "0'a is an integer term");
        t.expect(int_of(*r_char).value == 97, "0'a evaluates to ASCII code 97");
    }

    auto r_hex = parse_term("0x1F");
    t.expect(r_hex.has_value(), "hexadecimal 0x1F parses");
    if (r_hex) {
        t.expect(is_int(*r_hex), "0x1F is an integer term");
        t.expect(int_of(*r_hex).value == 31, "0x1F evaluates to decimal 31");
    }

    auto r_oct = parse_term("0o17");
    t.expect(r_oct.has_value(), "octal 0o17 parses");
    if (r_oct) {
        t.expect(is_int(*r_oct), "0o17 is an integer term");
        t.expect(int_of(*r_oct).value == 15, "0o17 evaluates to decimal 15");
    }

    auto r_bin = parse_term("0b101");
    t.expect(r_bin.has_value(), "binary 0b101 parses");
    if (r_bin) {
        t.expect(is_int(*r_bin), "0b101 is an integer term");
        t.expect(int_of(*r_bin).value == 5, "0b101 evaluates to decimal 5");
    }

    auto r_quote = parse_term("0'''");
    t.expect(r_quote.has_value(), "doubled-quote character code 0''' parses");
    if (r_quote) {
        t.expect(is_int(*r_quote), "0''' is an integer term");
        t.expect(int_of(*r_quote).value == 39, "0''' evaluates to ASCII code 39");
    }
})
.test("parser_rejects_floating_point_literals_as_not_implemented", [](TestContext& t) {
    auto r = parse_term("3.14");
    t.expect(!r.has_value(), "float literal 3.14 must not parse successfully");
    if (!r) {
        t.expect(r.error() == MathError::not_implemented, "float literal returns MathError::not_implemented");
    }

    auto r_clause = parse_clause("p(3.14).");
    t.expect(!r_clause.has_value(), "clause containing float literal 3.14 must not parse");
    if (!r_clause) {
        t.expect(r_clause.error() == MathError::not_implemented, "clause containing float returns MathError::not_implemented");
    }
})
.test("parser_reports_overflow_for_integers_exceeding_64_bits", [](TestContext& t) {
    auto r = parse_term("99999999999999999999999999999999999999999999999999");
    t.expect(!r.has_value(), "integer exceeding 64 bits must fail to parse");
    if (!r) {
        t.expect(r.error() == MathError::overflow, "overlarge integer yields MathError::overflow");
    }
})
.test("parser_accepts_int64_min_literal_but_rejects_positive_counterpart", [](TestContext& t) {
    auto r_min = parse_term("-9223372036854775808");
    t.expect(r_min.has_value(), "minimum signed 64-bit integer literal parses");
    if (r_min) {
        t.expect(is_int(*r_min), "parsed term is an integer");
        t.expect(int_of(*r_min).value == std::numeric_limits<std::int64_t>::min(), "value equals INT64_MIN");
    }

    auto r_pos = parse_term("9223372036854775808");
    t.expect(!r_pos.has_value(), "same digits without minus sign cannot fit in int64 and must fail");
    if (!r_pos) {
        t.expect(r_pos.error() == MathError::overflow, "positive 9223372036854775808 returns MathError::overflow");
    }
})
.test("parser_skips_line_and_block_comments_and_fails_on_unterminated_block", [](TestContext& t) {
    auto r_line = parse_term("% this is a line comment\nfoo");
    t.expect(r_line.has_value(), "term after line comment parses");
    if (r_line) {
        t.expect(is_atom(*r_line) && atom_of(*r_line).name == "foo", "parsed term is atom foo");
    }

    auto r_block = parse_term("/* this is a block comment */ bar");
    t.expect(r_block.has_value(), "term after block comment parses");
    if (r_block) {
        t.expect(is_atom(*r_block) && atom_of(*r_block).name == "bar", "parsed term is atom bar");
    }

    auto r_unterminated = parse_term("/* unterminated block comment");
    t.expect(!r_unterminated.has_value(), "unterminated block comment must fail parsing");
    if (!r_unterminated) {
        t.expect(r_unterminated.error() == MathError::syntax_error, "unterminated block comment returns syntax_error");
    }
})
.test("parser_rejects_unterminated_quoted_atoms_with_syntax_error", [](TestContext& t) {
    auto r1 = parse_term("'unterminated quoted atom");
    t.expect(!r1.has_value(), "unterminated quoted atom fails to parse");
    if (!r1) {
        t.expect(r1.error() == MathError::syntax_error, "unterminated quoted atom returns syntax_error");
    }

    auto r2 = parse_term("'raw newline\ninside quoted atom'");
    t.expect(!r2.has_value(), "quoted atom with raw newline fails to parse");
    if (!r2) {
        t.expect(r2.error() == MathError::syntax_error, "quoted atom with raw newline returns syntax_error");
    }
})
.test("operator_precedence_binds_multiplication_tighter_than_addition", [](TestContext& t) {
    auto r1 = parse_term("1+2*3");
    t.expect(r1.has_value(), "1+2*3 parses successfully");
    if (r1) {
        t.expect(to_string(*r1) == "+(1, *(2, 3))", "canonical structure of 1+2*3 is +(1, *(2, 3))");
        t.expect(*r1 == make_compound("+", {make_int(1), make_compound("*", {make_int(2), make_int(3)})}), "structural equality for +(1, *(2, 3))");
    }

    auto r2 = parse_term("(1+2)*3");
    t.expect(r2.has_value(), "(1+2)*3 parses successfully");
    if (r2) {
        t.expect(to_string(*r2) == "*(+(1, 2), 3)", "canonical structure of (1+2)*3 is *(+(1, 2), 3)");
        t.expect(*r2 == make_compound("*", {make_compound("+", {make_int(1), make_int(2)}), make_int(3)}), "structural equality for *(+(1, 2), 3)");
    }
})
.test("yfx_infix_operator_associates_to_the_left", [](TestContext& t) {
    auto r = parse_term("1-2-3");
    t.expect(r.has_value(), "1-2-3 parses successfully");
    if (r) {
        t.expect(to_string(*r) == "-(-(1, 2), 3)", "1-2-3 canonical structure is -(-(1, 2), 3)");
        t.expect(*r == make_compound("-", {make_compound("-", {make_int(1), make_int(2)}), make_int(3)}), "1-2-3 structural equality");
    }
})
.test("xfy_infix_operator_associates_to_the_right", [](TestContext& t) {
    auto r = parse_term("1^2^3");
    t.expect(r.has_value(), "1^2^3 parses successfully");
    if (r) {
        t.expect(to_string(*r) == "^(1, ^(2, 3))", "1^2^3 canonical structure is ^(1, ^(2, 3))");
        t.expect(*r == make_compound("^", {make_int(1), make_compound("^", {make_int(2), make_int(3)})}), "1^2^3 structural equality");
    }
})
.test("rule_operator_binds_head_and_conjunction_body", [](TestContext& t) {
    auto r = parse_term("a :- b, c");
    t.expect(r.has_value(), "a :- b, c parses successfully");
    if (r) {
        auto const expected = make_compound(":-", {make_atom("a"), make_compound(",", {make_atom("b"), make_atom("c")})});
        t.expect(*r == expected, "a :- b, c structurally matches :-(a, ','(b, c))");
        auto s = to_string(*r);
        t.expect(s == ":-(a, ,(b, c))" || s == ":-(a, ','(b, c))", "to_string matches canonical rule representation");
    }
})
.test("parser_reads_is_expression_and_prefix_negation_compound", [](TestContext& t) {
    auto r1 = parse_term("X is 1+2");
    t.expect(r1.has_value(), "X is 1+2 parses successfully");
    if (r1) {
        auto const expected1 = make_compound("is", {make_var("X"), make_compound("+", {make_int(1), make_int(2)})});
        t.expect(*r1 == expected1, "X is 1+2 matches is(X, +(1, 2))");
        t.expect(to_string(*r1) == "is(X, +(1, 2))", "canonical string for is(X, +(1, 2))");
    }

    auto r2 = parse_term("\\+ a");
    t.expect(r2.has_value(), "\\+ a parses successfully");
    if (r2) {
        auto const expected2 = make_compound("\\+", {make_atom("a")});
        t.expect(*r2 == expected2, "\\+ a matches \\+(a)");
        t.expect(to_string(*r2) == "\\+(a)", "canonical string for \\+(a)");
    }
})
.test("xfx_operator_cannot_nest_at_its_own_priority", [](TestContext& t) {
    auto r = parse_term("a = b = c");
    t.expect(!r.has_value(), "a = b = c must fail parsing due to non-associative operator");
    if (!r) {
        t.expect(r.error() == MathError::syntax_error, "xfx operator at same priority returns syntax_error");
    }
})
.test("parser_disambiguates_prefix_minus_from_negative_int_and_atom", [](TestContext& t) {
    auto r1 = parse_term("- 1");
    t.expect(r1.has_value(), "- 1 parses successfully");
    if (r1) {
        t.expect(is_int(*r1), "- 1 parses as an integer term");
        t.expect(int_of(*r1).value == -1, "- 1 integer value is -1");
    }

    auto r2 = parse_term("-(1)");
    t.expect(r2.has_value(), "-(1) parses successfully");
    if (r2) {
        t.expect(is_compound(*r2), "-(1) parses as a compound");
        t.expect(*r2 == make_compound("-", {make_int(1)}), "-(1) structurally matches compound -(1)");
        t.expect(to_string(*r2) == "-(1)", "-(1) canonical string is -(1)");
    }

    auto r3 = parse_term("- a");
    t.expect(r3.has_value(), "- a parses successfully");
    if (r3) {
        t.expect(is_compound(*r3), "- a parses as a compound");
        t.expect(*r3 == make_compound("-", {make_atom("a")}), "- a structurally matches compound -(a)");
        t.expect(to_string(*r3) == "-(a)", "- a canonical string is -(a)");
    }

    auto r4 = parse_term("f(-, a)");
    t.expect(r4.has_value(), "f(-, a) parses successfully");
    if (r4) {
        t.expect(is_compound(*r4), "f(-, a) is a compound");
        if (is_compound(*r4)) {
            const auto& c = compound_of(*r4);
            t.expect(c.functor == "f", "functor is f");
            t.expect(c.args.size() == 2, "arity is 2");
            if (c.args.size() == 2) {
                t.expect(is_atom(c.args[0]), "first argument is an atom");
                t.expect(atom_of(c.args[0]).name == "-", "first argument atom name is '-'");
                t.expect(c.args[1] == make_atom("a"), "second argument is atom a");
            }
        }
    }
})
.test("parenthesised_comma_in_arguments_preserves_compound_arity", [](TestContext& t) {
    auto r = parse_term("f(a, (b,c))");
    t.expect(r.has_value(), "f(a, (b,c)) parses successfully");
    if (r) {
        t.expect(is_compound(*r), "term is a compound");
        if (is_compound(*r)) {
            const auto& c = compound_of(*r);
            t.expect(c.functor == "f", "functor is f");
            t.expect(c.args.size() == 2, "compound arity is 2");
            if (c.args.size() == 2) {
                t.expect(c.args[0] == make_atom("a"), "first argument is a");
                t.expect(c.args[1] == make_compound(",", {make_atom("b"), make_atom("c")}), "second argument is compound ','(b, c)");
            }
        }
    }
})
.test("parser_handles_proper_improper_empty_and_nested_lists", [](TestContext& t) {
    auto r1 = parse_term("[a,b]");
    t.expect(r1.has_value(), "[a,b] parses successfully");
    if (r1) {
        t.expect(to_string(*r1) == "[a, b]", "canonical string of [a,b] is [a, b]");
        t.expect(*r1 == make_list({make_atom("a"), make_atom("b")}), "structural equality for [a,b]");
    }

    auto r2 = parse_term("[a,b|T]");
    t.expect(r2.has_value(), "[a,b|T] parses successfully");
    if (r2) {
        t.expect(to_string(*r2) == "[a, b | T]", "canonical string of [a,b|T] is [a, b | T]");
        auto const expected2 = make_compound(".", {make_atom("a"), make_compound(".", {make_atom("b"), make_var("T")})});
        t.expect(*r2 == expected2, "structural equality for [a,b|T]");
    }

    auto r3 = parse_term("[]");
    t.expect(r3.has_value(), "[] parses successfully");
    if (r3) {
        t.expect(is_atom(*r3), "[] parses as an atom");
        t.expect(atom_of(*r3).name == "[]", "atom name is []");
        t.expect(is_nil(*r3), "is_nil returns true for []");
    }

    auto r4 = parse_term("[[1, 2], [3]]");
    t.expect(r4.has_value(), "nested list [[1, 2], [3]] parses successfully");
    if (r4) {
        t.expect(to_string(*r4) == "[[1, 2], [3]]", "nested list canonical string is [[1, 2], [3]]");
    }
})
.test("parser_handles_curly_compound_and_empty_curly_atom", [](TestContext& t) {
    auto r1 = parse_term("{a}");
    t.expect(r1.has_value(), "{a} parses successfully");
    if (r1) {
        t.expect(is_compound(*r1), "{a} parses as a compound");
        if (is_compound(*r1)) {
            const auto& c = compound_of(*r1);
            t.expect(c.functor == "{}", "{a} functor is {}");
            t.expect(c.args.size() == 1, "{a} arity is 1");
            if (c.args.size() == 1) {
                t.expect(c.args[0] == make_atom("a"), "{a} argument is atom a");
            }
        }
        t.expect(to_string(*r1) == "{}(a)", "{a} canonical string is {}(a)");
    }

    auto r2 = parse_term("{}");
    t.expect(r2.has_value(), "{} parses successfully");
    if (r2) {
        t.expect(is_atom(*r2), "{} parses as an atom");
        t.expect(atom_of(*r2).name == "{}", "{} atom name is {}");
    }
})
.test("parse_clause_splits_rule_into_head_and_conjunction_body_goals", [](TestContext& t) {
    auto r = parse_clause("p(X) :- q(X), r(X).");
    t.expect(r.has_value(), "parse_clause on rule succeeds");
    if (r) {
        t.expect(is_compound(r->head), "rule head is a compound");
        if (is_compound(r->head)) {
            t.expect(compound_of(r->head).functor == "p", "rule head functor is p");
            t.expect(compound_of(r->head).args.size() == 1, "rule head arity is 1");
        }
        t.expect(r->body.size() == 2, "rule body has exactly two goals");
        if (r->body.size() == 2) {
            t.expect(is_compound(r->body[0]) && compound_of(r->body[0]).functor == "q", "first body goal is q(X)");
            t.expect(is_compound(r->body[1]) && compound_of(r->body[1]).functor == "r", "second body goal is r(X)");
        }
    }
})
.test("parse_clause_produces_empty_body_for_a_fact", [](TestContext& t) {
    auto r = parse_clause("p(a).");
    t.expect(r.has_value(), "parse_clause on fact succeeds");
    if (r) {
        t.expect(is_compound(r->head), "fact head is a compound");
        if (is_compound(r->head)) {
            t.expect(compound_of(r->head).functor == "p", "fact head functor is p");
            t.expect(compound_of(r->head).args.size() == 1, "fact head arity is 1");
            if (compound_of(r->head).args.size() == 1) {
                t.expect(compound_of(r->head).args[0] == make_atom("a"), "fact argument is atom a");
            }
        }
        t.expect(r->body.empty(), "fact has an empty body goal list");
    }
})
.test("parse_program_preserves_clause_source_order", [](TestContext& t) {
    auto r = parse_program("p(1).\np(2).\np(3).\n");
    t.expect(r.has_value(), "parse_program succeeds");
    if (r) {
        t.expect(r->size() == 3, "program contains exactly 3 clauses");
        if (r->size() == 3) {
            t.expect((*r)[0].head == make_compound("p", {make_int(1)}), "clause 0 head is p(1)");
            t.expect((*r)[1].head == make_compound("p", {make_int(2)}), "clause 1 head is p(2)");
            t.expect((*r)[2].head == make_compound("p", {make_int(3)}), "clause 2 head is p(3)");
            t.expect((*r)[0].body.empty(), "clause 0 has empty body");
            t.expect((*r)[1].body.empty(), "clause 1 has empty body");
            t.expect((*r)[2].body.empty(), "clause 2 has empty body");
        }
    }
})
.test("parse_query_accepts_and_strips_leading_query_prompt", [](TestContext& t) {
    auto r_prompt = parse_query("?- p(X), q(X).");
    t.expect(r_prompt.has_value(), "parse_query with leading ?- succeeds");
    if (r_prompt) {
        t.expect(r_prompt->size() == 2, "query has exactly 2 goals");
        if (r_prompt->size() == 2) {
            t.expect(is_compound((*r_prompt)[0]) && compound_of((*r_prompt)[0]).functor == "p", "first goal is p");
            t.expect(is_compound((*r_prompt)[1]) && compound_of((*r_prompt)[1]).functor == "q", "second goal is q");
        }
    }

    auto r_bare = parse_query("p(X), q(X).");
    t.expect(r_bare.has_value(), "parse_query without leading ?- succeeds");
    if (r_bare) {
        t.expect(r_bare->size() == 2, "bare query has exactly 2 goals");
        if (r_bare->size() == 2) {
            t.expect(is_compound((*r_bare)[0]) && compound_of((*r_bare)[0]).functor == "p", "first goal is p");
            t.expect(is_compound((*r_bare)[1]) && compound_of((*r_bare)[1]).functor == "q", "second goal is q");
        }
    }
})
.test("parse_term_rejects_trailing_unconsumed_text_as_syntax_error", [](TestContext& t) {
    auto r = parse_term("foo bar");
    t.expect(!r.has_value(), "parse_term on foo bar must fail due to trailing tokens");
    if (!r) {
        t.expect(r.error() == MathError::syntax_error, "trailing text returns MathError::syntax_error");
    }
})
.test("op3_directive_in_program_defines_custom_infix_operator", [](TestContext& t) {
    auto r = parse_program(":- op(500, xfx, myop).\np(X) :- X myop 42.\n");
    t.expect(r.has_value(), "program containing op/3 directive and custom operator clause parses");
    if (r) {
        t.expect(r->size() == 1, "program has 1 clause after directive processing");
        if (r->size() == 1) {
            const auto& cl = (*r)[0];
            t.expect(cl.body.size() == 1, "clause body has 1 goal");
            if (cl.body.size() == 1) {
                t.expect(is_compound(cl.body[0]), "body goal is a compound");
                if (is_compound(cl.body[0])) {
                    t.expect(compound_of(cl.body[0]).functor == "myop", "goal functor matches custom operator myop");
                    t.expect(compound_of(cl.body[0]).args.size() == 2, "goal arity is 2");
                    if (compound_of(cl.body[0]).args.size() == 2) {
                        t.expect(is_var(compound_of(cl.body[0]).args[0]), "lhs of custom operator is variable");
                        t.expect(is_int(compound_of(cl.body[0]).args[1]), "rhs of custom operator is integer");
                        if (is_int(compound_of(cl.body[0]).args[1])) {
                            t.expect(int_of(compound_of(cl.body[0]).args[1]).value == 42, "rhs integer value is 42");
                        }
                    }
                }
            }
        }
    }
})
.test("program_directive_other_than_op3_is_refused_as_not_implemented", [](TestContext& t) {
    auto r = parse_program(":- dynamic(p/1).\np(1).\n");
    t.expect(!r.has_value(), "non-op directive in program must be refused");
    if (!r) {
        t.expect(r.error() == MathError::not_implemented, "unsupported directive returns MathError::not_implemented");
    }
})
.test("empty_operator_table_forces_canonical_compound_form", [](TestContext& t) {
    auto const empty_ops = OperatorTable::empty();
    auto r_canon = parse_term("+(1, 2)", empty_ops);
    t.expect(r_canon.has_value(), "canonical form +(1, 2) parses with empty OperatorTable");
    if (r_canon) {
        t.expect(is_compound(*r_canon), "+(1, 2) parses as a compound");
        t.expect(*r_canon == make_compound("+", {make_int(1), make_int(2)}), "+(1, 2) structural match");
    }

    auto const r_infix = parse_term("1 + 2", empty_ops);
    t.expect(!r_infix.has_value(), "infix expression 1 + 2 fails to parse as operator with empty OperatorTable");
})
.test("roundtrip_preserves_operator_precedence_and_associativity_parentheses", [](TestContext& t) {
    t.expect(roundtrip("a+b*c") == "a+b*c", "a+b*c roundtrips without redundant parentheses");
    auto r1 = parse_term("a+b*c");
    if (r1) {
        auto r1_rt = parse_term(to_source(*r1));
        t.expect(r1_rt.has_value() && *r1_rt == *r1, "a+b*c roundtrip is structurally identical");
    }

    t.expect(roundtrip("(a+b)*c") == "(a+b)*c", "(a+b)*c preserves necessary grouping parentheses");
    auto r2 = parse_term("(a+b)*c");
    if (r2) {
        auto r2_rt = parse_term(to_source(*r2));
        t.expect(r2_rt.has_value() && *r2_rt == *r2, "(a+b)*c roundtrip is structurally identical");
    }

    t.expect(roundtrip("a-b-c") == "a-b-c", "a-b-c preserves left-associative tree without parentheses");
    auto r3 = parse_term("a-b-c");
    if (r3) {
        auto r3_rt = parse_term(to_source(*r3));
        t.expect(r3_rt.has_value() && *r3_rt == *r3, "a-b-c roundtrip is structurally identical");
    }

    t.expect(roundtrip("a-(b-c)") == "a-(b-c)", "a-(b-c) preserves non-default right associativity with parentheses");
    auto r4 = parse_term("a-(b-c)");
    if (r4) {
        auto r4_rt = parse_term(to_source(*r4));
        t.expect(r4_rt.has_value() && *r4_rt == *r4, "a-(b-c) roundtrip is structurally identical");
    }
})
.test("to_source_writes_unary_minus_compound_with_parentheses_not_as_integer", [](TestContext& t) {
    Term compound_minus = make_compound("-", {make_int(2)});
    std::string src = to_source(compound_minus);
    t.expect(src != "-2", "compound -(2) is not rendered as the bare integer literal -2");

    auto r = parse_term(src);
    t.expect(r.has_value(), "source of compound -(2) parses back successfully");
    if (r) {
        t.expect(is_compound(*r), "re-parsed term is a compound, not an integer");
        t.expect(*r == compound_minus, "re-parsed term is structurally identical to -(2)");
    }
})
.test("negative_integer_as_operator_operand_is_parenthesised_to_roundtrip", [](TestContext& t) {
    Term sub_neg = make_compound("-", {make_int(1), make_int(-2)});
    std::string src = to_source(sub_neg);
    auto r = parse_term(src);
    t.expect(r.has_value(), "source of 1 - (-2) parses back successfully");
    if (r) {
        t.expect(*r == sub_neg, "re-parsed term is structurally -(1, -2)");
        if (is_compound(*r)) {
            const auto& c = compound_of(*r);
            t.expect(c.args.size() == 2, "subtraction compound has arity 2");
            if (c.args.size() == 2) {
                t.expect(is_int(c.args[0]) && int_of(c.args[0]).value == 1, "lhs is integer 1");
                t.expect(is_int(c.args[1]) && int_of(c.args[1]).value == -2, "rhs is integer -2");
            }
        }
    }
})
.test("conjunction_compound_as_argument_is_parenthesised_to_preserve_arity", [](TestContext& t) {
    Term comp_conj = make_compound("f", {make_compound(",", {make_atom("a"), make_atom("b")})});
    std::string src = to_source(comp_conj);
    auto r = parse_term(src);
    t.expect(r.has_value(), "rendered compound with conjunction argument parses back");
    if (r) {
        t.expect(*r == comp_conj, "re-parsed compound structurally equals f((a, b))");
        if (is_compound(*r)) {
            t.expect(compound_of(*r).functor == "f", "functor is f");
            t.expect(compound_of(*r).args.size() == 1, "compound arity is preserved as 1, not flattened into 2");
        }
    }
})
.test("high_priority_operator_atom_roundtrips_in_list_and_compound", [](TestContext& t) {
    Term list_op = make_list({make_atom(":-")});
    std::string list_src = to_source(list_op);
    auto r_list = parse_term(list_src);
    t.expect(r_list.has_value(), "list [:-] parses back successfully");
    if (r_list) {
        t.expect(*r_list == list_op, "re-parsed list matches original list [:-]");
    }

    Term comp_op = make_compound("f", {make_atom(":-")});
    std::string comp_src = to_source(comp_op);
    auto r_comp = parse_term(comp_src);
    t.expect(r_comp.has_value(), "compound f((:-)) parses back successfully");
    if (r_comp) {
        t.expect(*r_comp == comp_op, "re-parsed compound matches original compound f(:-)");
    }
})
.test("quoted_atoms_with_special_characters_roundtrip_identically", [](TestContext& t) {
    Term a_space = make_atom("hello world");
    auto r_space = parse_term(to_source(a_space));
    t.expect(r_space.has_value() && *r_space == a_space, "atom with space roundtrips");

    Term a_empty = make_atom("");
    std::string empty_src = to_source(a_empty);
    t.expect(empty_src == "''", "empty atom is written as ''");
    auto r_empty = parse_term(empty_src);
    t.expect(r_empty.has_value() && *r_empty == a_empty, "empty atom roundtrips");

    Term a_quote = make_atom("it's");
    auto r_quote = parse_term(to_source(a_quote));
    t.expect(r_quote.has_value() && *r_quote == a_quote, "atom containing single quote roundtrips");

    Term a_newline = make_atom("line1\nline2");
    auto r_newline = parse_term(to_source(a_newline));
    t.expect(r_newline.has_value() && *r_newline == a_newline, "atom containing newline roundtrips");
})
.test("improper_lists_curly_terms_and_deeply_nested_compounds_roundtrip", [](TestContext& t) {
    Term improper_list = make_compound(".", {make_atom("a"), make_compound(".", {make_atom("b"), make_var("T")})});
    auto r1 = parse_term(to_source(improper_list));
    t.expect(r1.has_value() && *r1 == improper_list, "improper list [a,b|T] roundtrips structurally");

    Term curly_term = make_compound("{}", {make_compound(",", {make_atom("a"), make_atom("b")})});
    auto r2 = parse_term(to_source(curly_term));
    t.expect(r2.has_value() && *r2 == curly_term, "curly term {a,b} roundtrips structurally");

    Term deep = make_compound("f", {make_compound("f", {make_compound("f", {make_atom("a")})})});
    auto r3 = parse_term(to_source(deep));
    t.expect(r3.has_value() && *r3 == deep, "deeply nested compound f(f(f(a))) roundtrips structurally");
})
.test("atom_needs_quotes_distinguishes_bare_atoms_from_quoted_ones", [](TestContext& t) {
    t.expect(!atom_needs_quotes("=.."), "=.. is a symbolic atom and does not need quotes");
    t.expect(!atom_needs_quotes("foo"), "plain lowercase atom foo does not need quotes");
    t.expect(!atom_needs_quotes("!"), "solo atom ! does not need quotes");
    t.expect(!atom_needs_quotes(";"), "solo atom ; does not need quotes");
    t.expect(!atom_needs_quotes("[]"), "solo atom [] does not need quotes");
    t.expect(!atom_needs_quotes("{}"), "solo atom {} does not need quotes");

    t.expect(atom_needs_quotes("Foo"), "atom starting with uppercase letter requires quotes");
    t.expect(atom_needs_quotes(""), "empty atom requires quotes");
    t.expect(atom_needs_quotes("hello world"), "atom containing space requires quotes");
    t.expect(atom_needs_quotes("_bar"), "atom starting with underscore requires quotes");

    t.expect(quote_atom("foo") == "foo", "quote_atom leaves plain lowercase atom unquoted");
    t.expect(quote_atom("=..") == "=..", "quote_atom leaves symbolic operator atom unquoted");
    t.expect(quote_atom("Foo") == "'Foo'", "quote_atom wraps uppercase-starting atom in quotes");
    t.expect(quote_atom("") == "''", "quote_atom produces '' for empty atom");
    t.expect(quote_atom("hello world") == "'hello world'", "quote_atom wraps atom with spaces in quotes");
})
.test("integration_reader_and_solver_collaborate_on_list_append", [](TestContext& t) {
    auto prog_res = parse_program("append([], L, L).\nappend([H|T], L, [H|R]) :- append(T, L, R).\n");
    t.expect(prog_res.has_value(), "append/3 program parses from source");
    if (!prog_res) {
        return;
    }

    auto q1_res = parse_query("?- append([1, 2], [3, 4], X).");
    t.expect(q1_res.has_value(), "forward append query parses from source");
    if (q1_res) {
        auto sol1 = solve(*prog_res, *q1_res, 0);
        t.expect(sol1.has_value(), "solve succeeds on append([1,2],[3,4],X)");
        if (sol1) {
            t.expect(sol1->size() == 1, "forward append query produces exactly 1 solution");
            if (sol1->size() == 1) {
                auto const x_bind = apply_substitution((*sol1)[0], make_var("X"));
                t.expect(to_string(x_bind) == "[1, 2, 3, 4]", "forward append binds X to [1, 2, 3, 4]");
            }
        }
    }

    auto q2_res = parse_query("?- append(A, B, [1, 2]).");
    t.expect(q2_res.has_value(), "backtracking append decomposition query parses");
    if (q2_res) {
        auto sol2 = solve(*prog_res, *q2_res, 0);
        t.expect(sol2.has_value(), "solve succeeds on append(A,B,[1,2])");
        if (sol2) {
            t.expect(sol2->size() == 3, "append(A,B,[1,2]) produces exactly 3 solutions");
            if (sol2->size() == 3) {
                auto const a0 = apply_substitution((*sol2)[0], make_var("A"));
                auto const b0 = apply_substitution((*sol2)[0], make_var("B"));
                t.expect(to_string(a0) == "[]", "solution 0 binds A to []");
                t.expect(to_string(b0) == "[1, 2]", "solution 0 binds B to [1, 2]");

                auto const a1 = apply_substitution((*sol2)[1], make_var("A"));
                auto const b1 = apply_substitution((*sol2)[1], make_var("B"));
                t.expect(to_string(a1) == "[1]", "solution 1 binds A to [1]");
                t.expect(to_string(b1) == "[2]", "solution 1 binds B to [2]");

                auto const a2 = apply_substitution((*sol2)[2], make_var("A"));
                auto const b2 = apply_substitution((*sol2)[2], make_var("B"));
                t.expect(to_string(a2) == "[1, 2]", "solution 2 binds A to [1, 2]");
                t.expect(to_string(b2) == "[]", "solution 2 binds B to []");
            }
        }
    }
})
.test("integration_reader_and_solver_collaborate_on_recursive_ancestor_relation", [](TestContext& t) {
    std::string src =
        "parent(pam, bob).\n"
        "parent(tom, bob).\n"
        "parent(tom, liz).\n"
        "parent(bob, ann).\n"
        "parent(bob, pat).\n"
        "ancestor(X, Y) :- parent(X, Y).\n"
        "ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).\n";

    auto prog_res = parse_program(src);
    t.expect(prog_res.has_value(), "ancestor program parses from source");
    if (!prog_res) {
        return;
    }

    auto q_yes = parse_query("?- ancestor(tom, ann).");
    t.expect(q_yes.has_value(), "ancestor(tom, ann) query parses");
    if (q_yes) {
        auto sol_yes = solve(*prog_res, *q_yes, 0);
        t.expect(sol_yes.has_value(), "solve succeeds on true ancestor query");
        if (sol_yes) {
            t.expect(sol_yes->size() == 1, "ancestor(tom, ann) succeeds with 1 answer");
        }
    }

    auto q_no = parse_query("?- ancestor(liz, ann).");
    t.expect(q_no.has_value(), "ancestor(liz, ann) query parses");
    if (q_no) {
        auto sol_no = solve(*prog_res, *q_no, 0);
        t.expect(sol_no.has_value(), "solve succeeds on false ancestor query");
        if (sol_no) {
            t.expect(sol_no->empty(), "ancestor(liz, ann) fails honestly with 0 answers");
        }
    }

    auto q_all = parse_query("?- ancestor(bob, X).");
    t.expect(q_all.has_value(), "ancestor(bob, X) query parses");
    if (q_all) {
        auto sol_all = solve(*prog_res, *q_all, 0);
        t.expect(sol_all.has_value(), "solve succeeds on ancestor(bob, X)");
        if (sol_all) {
            t.expect(sol_all->size() == 2, "bob has exactly 2 descendants: ann and pat");
            if (sol_all->size() == 2) {
                auto const x0 = apply_substitution((*sol_all)[0], make_var("X"));
                auto const x1 = apply_substitution((*sol_all)[1], make_var("X"));
                t.expect(to_string(x0) == "ann", "first descendant of bob is ann");
                t.expect(to_string(x1) == "pat", "second descendant of bob is pat");
            }
        }
    }
})
        .run();
}
