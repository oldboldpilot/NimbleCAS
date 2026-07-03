// Tests for nimblecas.logic: unification with occurs-check and SLD resolution over Horn
// clauses (a small Prolog core), including the OR-parallel solver.
// @author Olumuyiwa Oluwasanmi
//
// Every case is deterministic and verifies bindings by APPLYING the answer substitution to the
// query variables and comparing the resulting ground terms structurally.

import std;
import nimblecas.core;
import nimblecas.logic;
import nimblecas.testing;

using nimblecas::apply_substitution;
using nimblecas::Clause;
using nimblecas::make_atom;
using nimblecas::make_compound;
using nimblecas::make_int;
using nimblecas::make_list;
using nimblecas::make_nil;
using nimblecas::make_var;
using nimblecas::MathError;
using nimblecas::Program;
using nimblecas::solve;
using nimblecas::solve_first;
using nimblecas::solve_or_parallel;
using nimblecas::Substitution;
using nimblecas::Term;
using nimblecas::to_string;
using nimblecas::unify;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// parent/2 facts plus grandparent/2 and (recursive) ancestor/2 rules.
[[nodiscard]] auto family_program() -> Program {
    Program p;
    p.push_back(Clause{make_compound("parent", {make_atom("tom"), make_atom("bob")}), {}});
    p.push_back(Clause{make_compound("parent", {make_atom("bob"), make_atom("ann")}), {}});
    p.push_back(Clause{make_compound("parent", {make_atom("bob"), make_atom("pat")}), {}});

    const Term x = make_var("X");
    const Term y = make_var("Y");
    const Term z = make_var("Z");

    // grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
    p.push_back(Clause{make_compound("grandparent", {x, z}),
                       {make_compound("parent", {x, y}), make_compound("parent", {y, z})}});
    // ancestor(X, Y) :- parent(X, Y).
    p.push_back(Clause{make_compound("ancestor", {x, y}), {make_compound("parent", {x, y})}});
    // ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
    p.push_back(Clause{make_compound("ancestor", {x, y}),
                       {make_compound("parent", {x, z}), make_compound("ancestor", {z, y})}});
    return p;
}

// append([], L, L).  append([H|T], L, [H|R]) :- append(T, L, R).
[[nodiscard]] auto append_program() -> Program {
    Program p;
    const Term h = make_var("H");
    const Term t = make_var("T");
    const Term l = make_var("L");
    const Term r = make_var("R");

    p.push_back(Clause{make_compound("append", {make_nil(), l, l}), {}});
    p.push_back(Clause{make_compound("append", {make_compound(".", {h, t}), l,
                                                make_compound(".", {h, r})}),
                       {make_compound("append", {t, l, r})}});
    return p;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.logic")
        .test("unify_occurs_check_fails",
              [](TestContext& t) {
                  // X = f(X) must fail the occurs-check: a value (no error) whose optional is
                  // empty — NOT a MathError.
                  const Term x = make_var("X");
                  const Term fx = make_compound("f", {x});
                  auto u = unify(x, fx, {});
                  t.expect(u.has_value(), "unify returns a value branch (not an error)");
                  if (u) {
                      t.expect(!u->has_value(), "X = f(X) is rejected by occurs-check (nullopt)");
                  }
              })
        .test("unify_simple_binds_both",
              [](TestContext& t) {
                  // f(X, b) with f(a, Y) => X = a, Y = b.
                  const Term x = make_var("X");
                  const Term y = make_var("Y");
                  auto u = unify(make_compound("f", {x, make_atom("b")}),
                                 make_compound("f", {make_atom("a"), y}), {});
                  t.expect(u.has_value() && u->has_value(), "f(X,b) unifies with f(a,Y)");
                  if (u && u->has_value()) {
                      t.expect(apply_substitution(**u, x) == make_atom("a"), "X = a");
                      t.expect(apply_substitution(**u, y) == make_atom("b"), "Y = b");
                  }
              })
        .test("unify_atom_vs_int_fails",
              [](TestContext& t) {
                  // An atom and an integer never unify (distinct constant kinds).
                  auto u = unify(make_atom("1"), make_int(1), {});
                  t.expect(u.has_value(), "no error");
                  t.expect(u.has_value() && !u->has_value(), "atom `1` != integer 1");
              })
        .test("grandparent_enumerates_ann_then_pat",
              [](TestContext& t) {
                  const Program p = family_program();
                  const Term w = make_var("W");
                  const std::vector<Term> goals = {
                      make_compound("grandparent", {make_atom("tom"), w})};
                  auto r = solve(p, goals, 0);
                  t.expect(r.has_value(), "solve succeeds");
                  if (r) {
                      t.expect(r->size() == 2, "grandparent(tom, W) has two solutions");
                      if (r->size() == 2) {
                          t.expect(apply_substitution((*r)[0], w) == make_atom("ann"),
                                   "first W = ann");
                          t.expect(apply_substitution((*r)[1], w) == make_atom("pat"),
                                   "second W = pat");
                      }
                  }
              })
        .test("ancestor_enumerates_bob_ann_pat",
              [](TestContext& t) {
                  const Program p = family_program();
                  const Term w = make_var("W");
                  const std::vector<Term> goals = {
                      make_compound("ancestor", {make_atom("tom"), w})};
                  auto r = solve(p, goals, 0);
                  t.expect(r.has_value(), "solve succeeds");
                  if (r) {
                      t.expect(r->size() == 3, "ancestor(tom, W) has three solutions in budget");
                      if (r->size() == 3) {
                          t.expect(apply_substitution((*r)[0], w) == make_atom("bob"),
                                   "first ancestor = bob");
                          t.expect(apply_substitution((*r)[1], w) == make_atom("ann"),
                                   "second ancestor = ann");
                          t.expect(apply_substitution((*r)[2], w) == make_atom("pat"),
                                   "third ancestor = pat");
                      }
                  }
              })
        .test("no_solution_query_is_empty_not_error",
              [](TestContext& t) {
                  const Program p = family_program();
                  const Term w = make_var("W");
                  // bob has children but no grandchildren.
                  auto r = solve(p, {make_compound("grandparent", {make_atom("bob"), w})}, 0);
                  t.expect(r.has_value(), "a query with no solutions is NOT an error");
                  t.expect(r.has_value() && r->empty(), "grandparent(bob, W) => empty vector");
              })
        .test("max_solutions_caps_enumeration",
              [](TestContext& t) {
                  const Program p = family_program();
                  const Term w = make_var("W");
                  auto r = solve(p, {make_compound("ancestor", {make_atom("tom"), w})}, 1);
                  t.expect(r.has_value(), "solve succeeds");
                  t.expect(r.has_value() && r->size() == 1, "max_solutions = 1 caps to one answer");
                  if (r && r->size() == 1) {
                      t.expect(apply_substitution((*r)[0], w) == make_atom("bob"),
                               "the single answer is the first: bob");
                  }
              })
        .test("solve_first_returns_first_then_nullopt",
              [](TestContext& t) {
                  const Program p = family_program();
                  const Term w = make_var("W");
                  auto f = solve_first(p, {make_compound("grandparent", {make_atom("tom"), w})});
                  t.expect(f.has_value(), "solve_first succeeds");
                  if (f) {
                      t.expect(f->has_value(), "a first solution exists");
                      if (f->has_value()) {
                          t.expect(apply_substitution(**f, w) == make_atom("ann"),
                                   "first grandparent answer is ann");
                      }
                  }
                  auto none = solve_first(p, {make_compound("grandparent", {make_atom("bob"), w})});
                  t.expect(none.has_value(), "no-solution solve_first is not an error");
                  t.expect(none.has_value() && !none->has_value(),
                           "no solution => nullopt (not an error)");
              })
        .test("append_small_lists",
              [](TestContext& t) {
                  const Program p = append_program();
                  const Term r = make_var("R");
                  // append([1], [2], R) => R = [1, 2].
                  const std::vector<Term> goals = {make_compound(
                      "append", {make_list({make_int(1)}), make_list({make_int(2)}), r})};
                  auto sols = solve(p, goals, 0);
                  t.expect(sols.has_value(), "append solve succeeds");
                  if (sols) {
                      t.expect(sols->size() == 1, "exactly one append result");
                      if (sols->size() == 1) {
                          const Term expected = make_list({make_int(1), make_int(2)});
                          t.expect(apply_substitution((*sols)[0], r) == expected, "R = [1, 2]");
                      }
                  }
              })
        .test("list_to_string_renders_bracket_form",
              [](TestContext& t) {
                  // Sanity check on the list pretty-printer used above.
                  t.expect(to_string(make_list({make_int(1), make_int(2)})) == "[1, 2]",
                           "[1, 2] renders in list syntax");
              })
        .test("or_parallel_matches_serial_grandparent",
              [](TestContext& t) {
                  const Program p = family_program();
                  const Term w = make_var("W");
                  const std::vector<Term> goals = {
                      make_compound("grandparent", {make_atom("tom"), w})};
                  auto serial = solve(p, goals, 0);
                  auto parallel = solve_or_parallel(p, goals, 0);
                  t.expect(serial.has_value() && parallel.has_value(), "both solvers succeed");
                  if (serial && parallel) {
                      t.expect(*serial == *parallel,
                               "OR-parallel result is identical to serial (grandparent)");
                  }
              })
        .test("or_parallel_matches_serial_ancestor",
              [](TestContext& t) {
                  const Program p = family_program();
                  const Term w = make_var("W");
                  const std::vector<Term> goals = {
                      make_compound("ancestor", {make_atom("tom"), w})};
                  auto serial = solve(p, goals, 0);
                  auto parallel = solve_or_parallel(p, goals, 0);
                  t.expect(serial.has_value() && parallel.has_value(), "both solvers succeed");
                  if (serial && parallel) {
                      t.expect(serial->size() == 3, "serial finds three ancestors");
                      t.expect(*serial == *parallel,
                               "OR-parallel result is identical to serial (ancestor)");
                  }
              })
        .test("malformed_program_and_query_are_domain_errors",
              [](TestContext& t) {
                  const Program p = family_program();
                  // A bare integer is not a callable goal.
                  auto bad_goal = solve(p, {make_int(5)}, 0);
                  t.expect(!bad_goal.has_value() && bad_goal.error() == MathError::domain_error,
                           "integer goal => domain_error");
                  // A bare variable is not a callable goal.
                  auto var_goal = solve(p, {make_var("G")}, 0);
                  t.expect(!var_goal.has_value() && var_goal.error() == MathError::domain_error,
                           "variable goal => domain_error");
                  // A clause whose head is a variable is malformed.
                  Program bad;
                  bad.push_back(Clause{make_var("X"), {}});
                  auto bad_prog = solve(bad, {make_atom("q")}, 0);
                  t.expect(!bad_prog.has_value() && bad_prog.error() == MathError::domain_error,
                           "variable clause head => domain_error");
              })
        .run();
}
