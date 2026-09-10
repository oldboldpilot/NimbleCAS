// Tests for nimblecas.logic: unification with occurs-check and SLD resolution over Horn
// clauses (a small Prolog core), including the OR-parallel solver.
// @author Olumuyiwa Oluwasanmi
//
// Every case is deterministic and verifies bindings by APPLYING the answer substitution to the
// query variables and comparing the resulting ground terms structurally.

import std;
import nimblecas.core;
import nimblecas.logic;
import nimblecas.logic_parser;
import nimblecas.testing;

using nimblecas::apply_substitution;
using nimblecas::Clause;
using nimblecas::make_atom;
using nimblecas::make_compound;
using nimblecas::make_int;
using nimblecas::make_list;
using nimblecas::make_nil;
using nimblecas::make_not;
using nimblecas::make_var;
using nimblecas::MathError;
using nimblecas::Program;
using nimblecas::solve;
using nimblecas::solve_first;
using nimblecas::solve_or_parallel;
using nimblecas::Substitution;
using nimblecas::Term;
using nimblecas::compare_terms;
using nimblecas::compound_of;
using nimblecas::Database;
using nimblecas::is_compound;
using nimblecas::is_var;
using nimblecas::Machine;
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

// man/1 facts, one married/1 fact, and the textbook negation rule
//   bachelor(X) :- man(X), \+ married(X).
// Note the goal ORDER: man(X) grounds X before the negation is called, which is the only
// arrangement under which `\+ married(X)` is sound.
[[nodiscard]] auto bachelor_program() -> Program {
    Program p;
    p.push_back(Clause{make_compound("man", {make_atom("john")}), {}});
    p.push_back(Clause{make_compound("man", {make_atom("paul")}), {}});
    p.push_back(Clause{make_compound("man", {make_atom("tom")}), {}});
    p.push_back(Clause{make_compound("married", {make_atom("paul")}), {}});

    const Term x = make_var("X");
    p.push_back(Clause{make_compound("bachelor", {x}),
                       {make_compound("man", {x}), make_not(make_compound("married", {x}))}});
    return p;
}

// The value bound to `v` by answer `s`, rendered — the idiom the rest of this file uses.
[[nodiscard]] auto bound(const Substitution& s, const Term& v) -> std::string {
    return to_string(apply_substitution(s, v));
}

// The same, naming the query variable — the idiom the later cases use.
[[nodiscard]] auto bound(const Substitution& s, std::string_view name) -> std::string {
    return to_string(apply_substitution(s, nimblecas::make_var(std::string(name))));
}

// Programs and queries written as PROLOG SOURCE TEXT.
//
// A test that states its program in the language under test is far easier to check against the
// behaviour it claims than the same program spelled out in factory calls, and it exercises the
// reader on every run. A malformed literal yields an empty program, which FAILS the test that
// uses it rather than quietly passing it.
[[nodiscard]] auto prog(std::string_view src) -> Program {
    auto p = nimblecas::logic_parser::parse_program(src);
    return p ? *p : Program{};
}

[[nodiscard]] auto query(std::string_view src) -> std::vector<Term> {
    auto g = nimblecas::logic_parser::parse_query(src);
    return g ? *g : std::vector<Term>{};
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

        // ---- Negation as failure ---------------------------------------------------------

        .test("negation_as_failure_proves_the_bachelors",
              [](TestContext& t) {
                  const Program p = bachelor_program();
                  const Term x = make_var("X");
                  auto r = solve(p, {make_compound("bachelor", {x})}, 0);
                  t.expect(r.has_value(), "bachelor/1 solves");
                  if (!r.has_value()) {
                      return;
                  }
                  t.expect(r->size() == 2, "exactly two bachelors (paul is married)");
                  if (r->size() == 2) {
                      t.expect(bound((*r)[0], x) == "john", "first bachelor is john");
                      t.expect(bound((*r)[1], x) == "tom", "second bachelor is tom");
                  }
                  // The negative half, stated directly: paul is excluded ONLY by the negation.
                  auto paul = solve(p, {make_compound("bachelor", {make_atom("paul")})}, 0);
                  t.expect(paul.has_value() && paul->empty(),
                           "bachelor(paul) has no solutions — married(paul) is provable");
              })

        .test("negation_succeeds_on_unprovable_and_fails_on_provable",
              [](TestContext& t) {
                  const Program p = bachelor_program();
                  // `\+ married(john)`: not provable, so the negation SUCCEEDS.
                  auto ok = solve(p, {make_not(make_compound("married", {make_atom("john")}))}, 0);
                  t.expect(ok.has_value() && ok->size() == 1,
                           "\\+ married(john) succeeds exactly once");
                  // `\+ married(paul)`: provable, so the negation FAILS — and failing is an
                  // empty answer list, NOT an error.
                  auto no = solve(p, {make_not(make_compound("married", {make_atom("paul")}))}, 0);
                  t.expect(no.has_value(), "a failing negation is not an error");
                  t.expect(no.has_value() && no->empty(), "\\+ married(paul) fails");
              })

        .test("negation_binds_nothing",
              [](TestContext& t) {
                  // `\+ G` is a test, not a generator: it must leave the query variable free.
                  const Program p = bachelor_program();
                  const Term x = make_var("X");
                  auto r = solve(p, {make_compound("man", {x}),
                                     make_not(make_compound("married", {make_atom("john")}))},
                                 0);
                  t.expect(r.has_value() && r->size() == 3,
                           "the negation removes no man/1 solution");
                  if (r.has_value() && r->size() == 3) {
                      t.expect(bound((*r)[0], x) == "john" && bound((*r)[1], x) == "paul" &&
                                   bound((*r)[2], x) == "tom",
                               "and contributes no binding of its own");
                  }
              })

        .test("floundering_negation_is_refused_rather_than_answered_unsoundly",
              [](TestContext& t) {
                  // num(1). num(2). p(1).
                  Program p;
                  p.push_back(Clause{make_compound("num", {make_int(1)}), {}});
                  p.push_back(Clause{make_compound("num", {make_int(2)}), {}});
                  p.push_back(Clause{make_compound("p", {make_int(1)}), {}});
                  const Term x = make_var("X");

                  // SAFE order: num(X) grounds X first, so each negation is called ground.
                  auto safe = solve(p, {make_compound("num", {x}),
                                        make_not(make_compound("p", {x}))},
                                    0);
                  t.expect(safe.has_value(), "the ground-at-call order solves");
                  t.expect(safe.has_value() && safe->size() == 1, "exactly one answer");
                  if (safe.has_value() && safe->size() == 1) {
                      t.expect(bound((*safe)[0], x) == "2", "X = 2 — the number that is not p");
                  }

                  // UNSAFE order: the SAME conjunction, reordered. Standard Prolog answers
                  // this with plain failure, which is wrong (X = 2 witnesses the existential)
                  // and disagrees with the run above. Refusing is the honest outcome.
                  auto unsafe = solve(p, {make_not(make_compound("p", {x})),
                                          make_compound("num", {x})},
                                      0);
                  t.expect(!unsafe.has_value(), "the floundering order is NOT answered");
                  t.expect(!unsafe.has_value() && unsafe.error() == MathError::domain_error,
                           "a non-ground negated goal => domain_error");

                  // And the bare case, with nothing to ground X at all.
                  auto bare = solve(p, {make_not(make_compound("p", {x}))}, 0);
                  t.expect(!bare.has_value() && bare.error() == MathError::domain_error,
                           "\\+ p(X) alone => domain_error");

                  // Non-groundness is not always visible at the call site. Both of these are
                  // caught only because the check runs on the RESOLVED goal.
                  Program ap;
                  ap.push_back(Clause{make_compound("eq", {make_var("Z"), make_var("Z")}), {}});
                  ap.push_back(Clause{make_compound("q", {make_atom("a")}), {}});
                  const Term y = make_var("Y");

                  // ALIASED: eq(X, Y) binds X to Y and leaves both free, so `\+ q(X)` resolves
                  // to a variable that is not the one written.
                  auto aliased = solve(
                      ap, {make_compound("eq", {x, y}), make_not(make_compound("q", {x}))}, 0);
                  t.expect(!aliased.has_value() && aliased.error() == MathError::domain_error,
                           "a merely ALIASED variable still flounders");

                  // PARTIALLY ground: a ground first argument does not make the goal ground.
                  auto partial = solve(
                      ap,
                      {make_not(make_compound("q", {make_compound("f", {make_atom("a"), y})}))},
                      0);
                  t.expect(!partial.has_value() && partial.error() == MathError::domain_error,
                           "a PARTIALLY ground negated goal flounders too");
              })

        .test("negation_under_budget_exhaustion_is_not_converged_not_success",
              [](TestContext& t) {
                  // loop :- loop.  The inner search can never terminate, so it is cut off by
                  // the depth budget without ever proving or refuting `loop`.
                  Program p;
                  p.push_back(Clause{make_atom("loop"), {make_atom("loop")}});

                  // A POSITIVE query under the same truncation honestly under-reports: it
                  // returns no answers, and that is not an error.
                  auto positive = solve(p, {make_atom("loop")}, 0);
                  t.expect(positive.has_value(), "a truncated positive search is not an error");
                  t.expect(positive.has_value() && positive->empty(),
                           "and yields no answers");

                  // The NEGATIVE query over the very same truncation must NOT convert "I ran
                  // out of budget" into "not provable". That would be an answer about the
                  // budget, not about the program.
                  auto negative = solve(p, {make_not(make_atom("loop"))}, 0);
                  t.expect(!negative.has_value(),
                           "\\+ loop does NOT silently succeed on budget exhaustion");
                  t.expect(!negative.has_value() &&
                               negative.error() == MathError::not_converged,
                           "truncated negation => not_converged");
              })

        .test("double_negation_succeeds_without_binding",
              [](TestContext& t) {
                  const Program p = bachelor_program();
                  // `\+ \+ married(paul)`: the inner negation fails, so the outer succeeds.
                  auto dd = solve(p, {make_not(make_not(
                                         make_compound("married", {make_atom("paul")})))},
                                  0);
                  t.expect(dd.has_value() && dd->size() == 1,
                           "\\+ \\+ married(paul) succeeds once");
                  // And the reverse: `\+ \+ married(john)` fails, since `\+ married(john)`
                  // succeeds.
                  auto dj = solve(p, {make_not(make_not(
                                         make_compound("married", {make_atom("john")})))},
                                  0);
                  t.expect(dj.has_value() && dj->empty(), "\\+ \\+ married(john) fails");
              })

        .test("malformed_negation_forms_are_domain_errors",
              [](TestContext& t) {
                  const Program p = bachelor_program();
                  // `\+ 3` — an integer can never be a goal.
                  auto ni = solve(p, {make_not(make_int(3))}, 0);
                  t.expect(!ni.has_value() && ni.error() == MathError::domain_error,
                           "\\+ 3 => domain_error");
                  // `\+`/2 is not negation.
                  auto n2 = solve(
                      p, {make_compound("\\+", {make_atom("a"), make_atom("b")})}, 0);
                  t.expect(!n2.has_value() && n2.error() == MathError::domain_error,
                           "\\+/2 => domain_error");
                  // A program may not define clauses for `\+`: two meanings for one goal.
                  Program redefined = bachelor_program();
                  redefined.push_back(Clause{make_compound("\\+", {make_atom("anything")}), {}});
                  auto rd = solve(redefined, {make_atom("man")}, 0);
                  t.expect(!rd.has_value() && rd.error() == MathError::domain_error,
                           "a user clause for \\+/1 => domain_error");

                  // `\+ X` cannot be judged statically — X may yet be bound to a callable
                  // term — so a variable that RESOLVES to a non-goal must be caught at the
                  // moment of the call instead.
                  Program vp;
                  vp.push_back(Clause{make_compound("v", {make_int(3)}), {}});
                  const Term xv = make_var("X");
                  auto rv = solve(vp, {make_compound("v", {xv}), make_not(xv)}, 0);
                  t.expect(!rv.has_value() && rv.error() == MathError::domain_error,
                           "\\+ X where X resolves to an integer => domain_error");
              })

        .test("or_parallel_matches_serial_in_the_presence_of_negation",
              [](TestContext& t) {
                  const Program p = bachelor_program();
                  const Term x = make_var("X");

                  // Negation nested in a clause body, reached through the parallel fan-out.
                  auto serial = solve(p, {make_compound("bachelor", {x})}, 0);
                  auto par = solve_or_parallel(p, {make_compound("bachelor", {x})}, 0);
                  t.expect(serial.has_value() && par.has_value(), "both solve");
                  t.expect(serial.has_value() && par.has_value() && *serial == *par,
                           "OR-parallel is identical to serial with negation in a body");

                  // A LEADING negation has no clause alternatives to fan out over; it must
                  // still agree with serial rather than silently enumerate nothing.
                  const Term g = make_not(make_compound("married", {make_atom("john")}));
                  auto s2 = solve(p, {g}, 0);
                  auto p2 = solve_or_parallel(p, {g}, 0);
                  t.expect(s2.has_value() && p2.has_value() && *s2 == *p2,
                           "OR-parallel matches serial for a leading negation");
                  t.expect(p2.has_value() && p2->size() == 1,
                           "and actually succeeds once rather than returning nothing");

                  // Errors must cross the parallel boundary too.
                  auto pe = solve_or_parallel(p, {make_compound("man", {x}),
                                                  make_not(make_compound("married", {x}))},
                                              0);
                  auto se = solve(p, {make_compound("man", {x}),
                                      make_not(make_compound("married", {x}))},
                                  0);
                  t.expect(se.has_value() && pe.has_value(),
                           "man(X) grounds X, so this order is sound in both solvers");
                  t.expect(se.has_value() && pe.has_value() && *se == *pe,
                           "and agrees");
              })

        // Serial stops the moment the cap is met and NEVER TRIES the later clauses, so a
        // clause serial would not have reached must not be able to fail the parallel query.
        // Every branch is evaluated speculatively, but a speculative failure is not a result.
        .test("a_capped_query_is_not_failed_by_a_clause_serial_never_reaches",
              [](TestContext& t) {
                  Program p;
                  p.push_back(Clause{make_compound("p", {make_int(1)}), {}});  // p(1).
                  // p(2) :- \+ q(X).  <-- flounders if it is ever reached
                  p.push_back(Clause{make_compound("p", {make_int(2)}),
                                     {make_not(make_compound("q", {make_var("X")}))}});
                  const Term y = make_var("Y");
                  const std::vector<Term> goal{make_compound("p", {y})};

                  auto s1 = solve(p, goal, 1);
                  auto p1 = solve_or_parallel(p, goal, 1);
                  t.expect(s1.has_value() && s1->size() == 1,
                           "serial meets the cap on clause 0 and stops");
                  t.expect(p1.has_value(),
                           "parallel must NOT surface an error from a clause serial never tried");
                  t.expect(s1.has_value() && p1.has_value() && *s1 == *p1,
                           "capped: the two solvers agree exactly");

                  // Uncapped, BOTH reach the floundering clause, so both must fail — and with
                  // the same error. The cap is the only thing that hides it.
                  auto s0 = solve(p, goal, 0);
                  auto p0 = solve_or_parallel(p, goal, 0);
                  t.expect(!s0.has_value() && !p0.has_value(),
                           "uncapped: both reach the bad clause and both fail");
                  if (!s0.has_value() && !p0.has_value()) {
                      t.expect(s0.error() == p0.error(), "with the same error");
                  }
              })

        // The truncation flag must be judged per sub-search. Here a SIBLING derivation for the
        // same goal truncates first; the negation that follows is decidable on its own and
        // must still answer.
        .test("a_siblings_truncation_is_not_charged_to_the_negation",
              [](TestContext& t) {
                  Program p;
                  p.push_back(Clause{make_atom("loop"), {make_atom("loop")}});  // loop :- loop.
                  p.push_back(Clause{make_atom("t"), {make_atom("loop")}});     // t :- loop.
                  p.push_back(Clause{make_atom("t"), {}});                      // t.

                  // Clause 2 for t diverges and is cut off; clause 3 proves t outright. The
                  // following `\+ absent` is a scan of a predicate with no clauses at all and
                  // cannot truncate on its own.
                  auto r = solve(p, {make_atom("t"), make_not(make_atom("absent"))}, 0);
                  t.expect(r.has_value(),
                           "the negation is judged on ITS OWN sub-search, not a sibling's");
                  t.expect(r.has_value() && r->size() == 1, "exactly one answer");

                  // Control: asked alone the same negation plainly succeeds.
                  auto alone = solve(p, {make_not(make_atom("absent"))}, 0);
                  t.expect(alone.has_value() && alone->size() == 1,
                           "and it succeeds when asked alone too");
              })

        // The reserved functor at the wrong arity is a domain_error when written literally;
        // it must stay one when it arrives at the call through a VARIABLE. A `\+`/2 term is a
        // compound, so an is_callable test alone would pass it through as an ordinary
        // predicate, match no clause, and make the negation quietly succeed.
        .test("reserved_functor_at_wrong_arity_is_caught_even_via_a_variable",
              [](TestContext& t) {
                  Program p;
                  p.push_back(Clause{
                      make_compound("v", {make_compound("\\+", {make_atom("a"), make_atom("b")})}),
                      {}});
                  const Term x = make_var("X");
                  auto r = solve(p, {make_compound("v", {x}), make_not(x)}, 0);
                  t.expect(!r.has_value(), "\\+ X where X resolves to \\+/2 does not succeed");
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "it is a domain_error, exactly as the literal form is");
              })

        // Negation must not be able to defeat termination. `p :- \+ p.` has no fixed point;
        // the shared depth budget is what stops it, so it must end in not_converged rather
        // than recursing until the native stack dies.
        .test("self_negating_clause_terminates_instead_of_recursing_forever",
              [](TestContext& t) {
                  Program p;
                  p.push_back(Clause{make_atom("p"), {make_not(make_atom("p"))}});
                  auto r = solve(p, {make_atom("p")}, 0);
                  t.expect(!r.has_value(), "p :- \\+ p does not fabricate an answer");
                  t.expect(!r.has_value() && r.error() == MathError::not_converged,
                           "it terminates with not_converged");
              })
.test("cut_in_clause_discards_remaining_clauses_of_its_own_call", [](TestContext& t) {
    const auto p = prog("p(1). p(2). p(3). q(X) :- p(X), !.");
    const auto q = query("q(X).");
    auto r = solve(p, q, 0);
    t.expect(r.has_value(), "cut query solves without error");
    if (r.has_value()) {
        t.expect(r->size() == 1, "cut discards remaining clauses leaving exactly one answer");
        if (r->size() >= 1) {
            t.expect(bound((*r)[0], "X") == "1", "X is bound to the first solution 1");
        }
    }
})
.test("cut_prunes_backtracking_but_preserves_forward_execution", [](TestContext& t) {
    const auto p = prog("p(1). p(2). r(X, Y) :- p(X), !, p(Y).");
    const auto q = query("r(X, Y).");
    auto r = solve(p, q, 0);
    t.expect(r.has_value(), "forward execution query solves without error");
    if (r.has_value()) {
        t.expect(r->size() == 2, "cut prunes alternatives for X but allows full enumeration of Y");
        if (r->size() >= 1) {
            t.expect(bound((*r)[0], "X") == "1", "first answer binds X to 1");
            t.expect(bound((*r)[0], "Y") == "1", "first answer binds Y to 1");
        }
        if (r->size() >= 2) {
            t.expect(bound((*r)[1], "X") == "1", "second answer binds X to 1");
            t.expect(bound((*r)[1], "Y") == "2", "second answer binds Y to 2");
        }
    }
})
.test("cut_is_local_to_its_clause_and_does_not_prune_caller_alternatives", [](TestContext& t) {
    const auto p = prog("cand(1). cand(2). inner(a) :- !. inner(b). top(X, Y) :- cand(X), inner(Y).");
    const auto q = query("top(X, Y).");
    auto r = solve(p, q, 0);
    t.expect(r.has_value(), "caller query solves without error");
    if (r.has_value()) {
        t.expect(r->size() == 2, "inner cut does not prune caller cand alternatives, yielding two answers");
        if (r->size() >= 1) {
            t.expect(bound((*r)[0], "X") == "1", "first answer binds X to 1");
            t.expect(bound((*r)[0], "Y") == "a", "first answer binds Y to a");
        }
        if (r->size() >= 2) {
            t.expect(bound((*r)[1], "X") == "2", "second answer binds X to 2");
            t.expect(bound((*r)[1], "Y") == "a", "second answer binds Y to a");
        }
    }
})
.test("cut_in_query_goal_list_prunes_query_alternatives", [](TestContext& t) {
    const auto p = prog("p(1). p(2). p(3).");
    const auto q = query("p(X), !.");
    auto r = solve(p, q, 0);
    t.expect(r.has_value(), "query with cut solves without error");
    if (r.has_value()) {
        t.expect(r->size() == 1, "cut at query level prunes subsequent query alternatives to exactly one");
        if (r->size() >= 1) {
            t.expect(bound((*r)[0], "X") == "1", "query cut commits to first answer X = 1");
        }
    }
})
.test("cut_in_query_goal_list_prunes_preceding_goals_but_allows_following_goals", [](TestContext& t) {
    const auto p = prog("p(1). p(2). q(a). q(b).");
    const auto q = query("p(X), !, q(Y).");
    auto r = solve(p, q, 0);
    t.expect(r.has_value(), "query solves without error");
    if (r.has_value()) {
        t.expect(r->size() == 2, "cut in query prunes p(X) alternatives while exploring all q(Y)");
        if (r->size() >= 1) {
            t.expect(bound((*r)[0], "X") == "1", "first answer has X = 1");
            t.expect(bound((*r)[0], "Y") == "a", "first answer has Y = a");
        }
        if (r->size() >= 2) {
            t.expect(bound((*r)[1], "X") == "1", "second answer has X = 1");
            t.expect(bound((*r)[1], "Y") == "b", "second answer has Y = b");
        }
    }
})
.test("cut_inside_call_is_opaque_and_does_not_prune_outer_goals", [](TestContext& t) {
    const auto p = prog("p(1). p(2).");
    const auto q = query("call((p(X), !)), p(Y).");
    auto r = solve(p, q, 0);
    t.expect(r.has_value(), "call with cut solves without error");
    if (r.has_value()) {
        t.expect(r->size() == 2, "cut inside call/1 is opaque so p(Y) is not pruned, producing 2 answers");
        if (r->size() >= 1) {
            t.expect(bound((*r)[0], "X") == "1", "first answer binds X to 1");
            t.expect(bound((*r)[0], "Y") == "1", "first answer binds Y to 1");
        }
        if (r->size() >= 2) {
            t.expect(bound((*r)[1], "X") == "1", "second answer binds X to 1");
            t.expect(bound((*r)[1], "Y") == "2", "second answer binds Y to 2");
        }
    }
})
.test("cut_inside_negation_is_opaque_and_negation_binds_nothing", [](TestContext& t) {
    const auto p = prog("p(1). p(2).");
    const auto q = query("p(X), \\+ (!, fail).");
    auto r = solve(p, q, 0);
    t.expect(r.has_value(), "negation with cut solves without error");
    if (r.has_value()) {
        t.expect(r->size() == 2, "cut inside negation is opaque so backtracking on p(X) produces 2 answers");
        if (r->size() >= 1) {
            t.expect(bound((*r)[0], "X") == "1", "first answer retains X = 1 from generator");
        }
        if (r->size() >= 2) {
            t.expect(bound((*r)[1], "X") == "2", "second answer retains X = 2 from generator");
        }
    }
})
.test("negation_of_ground_goal_with_cut_fails_when_goal_succeeds", [](TestContext& t) {
    const auto p = prog("p(1). p(2).");
    const auto q = query("X = 1, \\+ (p(1), !).");
    auto r = solve(p, q, 0);
    t.expect(r.has_value(), "ground negation solves without error");
    if (r.has_value()) {
        t.expect(r->empty(), "negation of provable ground goal fails cleanly with no solutions");
    }
})
.test("max_predicate_with_cut_selects_first_clause_when_first_argument_greater", [](TestContext& t) {
    const auto p = prog("max(X, Y, X) :- X >= Y, !. max(_, Y, Y).");
    const auto q = query("max(3, 2, M).");
    auto r = solve(p, q, 0);
    t.expect(r.has_value(), "max(3, 2, M) solves without error");
    if (r.has_value()) {
        t.expect(r->size() == 1, "max(3, 2, M) yields exactly one solution");
        if (r->size() >= 1) {
            t.expect(bound((*r)[0], "M") == "3", "max(3, 2, M) binds M to 3");
        }
    }
})
.test("max_predicate_with_cut_falls_through_to_second_clause_when_first_argument_smaller", [](TestContext& t) {
    const auto p = prog("max(X, Y, X) :- X >= Y, !. max(_, Y, Y).");
    const auto q = query("max(2, 3, M).");
    auto r = solve(p, q, 0);
    t.expect(r.has_value(), "max(2, 3, M) solves without error");
    if (r.has_value()) {
        t.expect(r->size() == 1, "max(2, 3, M) yields exactly one solution");
        if (r->size() >= 1) {
            t.expect(bound((*r)[0], "M") == "3", "max(2, 3, M) binds M to 3");
        }
    }
})
.test("cut_in_first_clause_prevents_trying_second_clause_even_on_subsequent_failure", [](TestContext& t) {
    const auto p = prog("p :- !, fail. p.");
    const auto q = query("p.");
    auto r = solve(p, q, 0);
    t.expect(r.has_value(), "solve terminates without error");
    if (r.has_value()) {
        t.expect(r->empty(), "cut in first clause prunes second clause, so failure produces no answers");
    }
})
.test("explicit_comma_functor_behaves_identically_to_goal_conjunction", [](TestContext& t) {
    const auto p = prog("p(1). p(2). q(a). q(b).");
    const auto q = query("','(p(X), q(Y)).");
    auto r = solve(p, q, 0);
    t.expect(r.has_value(), "explicit comma goal solves without error");
    if (r.has_value()) {
        t.expect(r->size() == 4, "explicit comma unrolls both goals to yield exactly 4 answers");
        if (r->size() >= 1) {
            t.expect(bound((*r)[0], "X") == "1", "answer 0 has X = 1");
            t.expect(bound((*r)[0], "Y") == "a", "answer 0 has Y = a");
        }
        if (r->size() >= 2) {
            t.expect(bound((*r)[1], "X") == "1", "answer 1 has X = 1");
            t.expect(bound((*r)[1], "Y") == "b", "answer 1 has Y = b");
        }
        if (r->size() >= 3) {
            t.expect(bound((*r)[2], "X") == "2", "answer 2 has X = 2");
            t.expect(bound((*r)[2], "Y") == "a", "answer 2 has Y = a");
        }
        if (r->size() >= 4) {
            t.expect(bound((*r)[3], "X") == "2", "answer 3 has X = 2");
            t.expect(bound((*r)[3], "Y") == "b", "answer 3 has Y = b");
        }
    }
})
.test("disjunction_tries_left_branch_fully_then_right_branch_in_order", [](TestContext& t) {
    const auto p = prog("p(1). p(2). q(3). q(4).");
    const auto q = query("(p(X) ; q(X)).");
    auto r = solve(p, q, 0);
    t.expect(r.has_value(), "disjunction solves without error");
    if (r.has_value()) {
        t.expect(r->size() == 4, "disjunction yields all solutions of left branch followed by right branch");
        if (r->size() >= 1) {
            t.expect(bound((*r)[0], "X") == "1", "answer 0 comes from left branch with X = 1");
        }
        if (r->size() >= 2) {
            t.expect(bound((*r)[1], "X") == "2", "answer 1 comes from left branch with X = 2");
        }
        if (r->size() >= 3) {
            t.expect(bound((*r)[2], "X") == "3", "answer 2 comes from right branch with X = 3");
        }
        if (r->size() >= 4) {
            t.expect(bound((*r)[3], "X") == "4", "answer 3 comes from right branch with X = 4");
        }
    }
})
.test("cut_in_left_branch_of_disjunction_cuts_enclosing_clause_and_skips_right_branch", [](TestContext& t) {
    const auto p = prog("choice(X) :- (X = 1, ! ; X = 2). choice(3).");
    const auto q = query("choice(X).");
    auto r = solve(p, q, 0);
    t.expect(r.has_value(), "query solves without error");
    if (r.has_value()) {
        t.expect(r->size() == 1, "cut in left branch prunes both right branch and outer choice(3) clause");
        if (r->size() >= 1) {
            t.expect(bound((*r)[0], "X") == "1", "X is bound to 1 from the cut branch");
        }
    }
})
.test("cut_followed_by_fail_in_left_branch_of_disjunction_fails_predicate_without_trying_right_branch", [](TestContext& t) {
    const auto p = prog("p(X) :- (X = 1, !, fail ; X = 2). p(3).");
    const auto q = query("p(X).");
    auto r = solve(p, q, 0);
    t.expect(r.has_value(), "query solves without error");
    if (r.has_value()) {
        t.expect(r->empty(), "cut before fail prunes right branch and subsequent clause, producing 0 answers");
    }
})
.test("cut_in_right_branch_of_disjunction_cuts_enclosing_clause_alternatives", [](TestContext& t) {
    const auto p = prog("choice(X) :- (X = 1 ; X = 2, !). choice(3).");
    const auto q = query("choice(X).");
    auto r = solve(p, q, 0);
    t.expect(r.has_value(), "query solves without error");
    if (r.has_value()) {
        t.expect(r->size() == 2, "left branch produces 1 then right branch cuts, pruning choice(3)");
        if (r->size() >= 1) {
            t.expect(bound((*r)[0], "X") == "1", "first answer has X = 1 from left branch");
        }
        if (r->size() >= 2) {
            t.expect(bound((*r)[1], "X") == "2", "second answer has X = 2 from right branch");
        }
    }
})
.test("if_then_else_commits_to_first_solution_of_condition", [](TestContext& t) {
    const auto p = prog("p(1). p(2).");
    const auto q = query("(p(X) -> true ; true).");
    auto r = solve(p, q, 0);
    t.expect(r.has_value(), "if-then-else query solves without error");
    if (r.has_value()) {
        t.expect(r->size() == 1, "if-then-else commits to the first solution of condition p(X)");
        if (r->size() >= 1) {
            t.expect(bound((*r)[0], "X") == "1", "X is bound to the first condition solution 1");
        }
    }
})
.test("if_then_else_takes_else_branch_when_condition_fails_and_discards_condition_bindings", [](TestContext& t) {
    const auto p = prog("");
    const auto q = query("(X = 1, fail -> Y = 10 ; Y = 20).");
    auto r = solve(p, q, 0);
    t.expect(r.has_value(), "if-then-else query solves without error");
    if (r.has_value()) {
        t.expect(r->size() == 1, "failed condition takes else branch, yielding exactly one answer");
        if (r->size() >= 1) {
            t.expect(bound((*r)[0], "X") == "X", "condition bindings do not escape to else branch, X remains unbound");
            t.expect(bound((*r)[0], "Y") == "20", "else branch binds Y to 20");
        }
    }
})
.test("bare_if_then_without_else_fails_when_condition_fails", [](TestContext& t) {
    const auto p = prog("p(1).");
    const auto q = query("(p(99) -> X = 1).");
    auto r = solve(p, q, 0);
    t.expect(r.has_value(), "bare if-then query solves without error");
    if (r.has_value()) {
        t.expect(r->empty(), "bare if-then with failing condition fails with no solutions");
    }
})
.test("soft_cut_runs_then_branch_for_every_solution_of_condition", [](TestContext& t) {
    const auto p = prog("p(1). p(2).");
    const auto q = query("(p(X) *-> true ; true).");
    auto r = solve(p, q, 0);
    t.expect(r.has_value(), "soft cut query solves without error");
    if (r.has_value()) {
        t.expect(r->size() == 2, "soft cut executes then branch for every solution of condition");
        if (r->size() >= 1) {
            t.expect(bound((*r)[0], "X") == "1", "first answer has X = 1");
        }
        if (r->size() >= 2) {
            t.expect(bound((*r)[1], "X") == "2", "second answer has X = 2");
        }
    }
})
.test("soft_cut_takes_else_branch_when_condition_has_no_solutions", [](TestContext& t) {
    const auto p = prog("p(1). p(2).");
    const auto q = query("(p(99) *-> X = 1 ; X = 2).");
    auto r = solve(p, q, 0);
    t.expect(r.has_value(), "soft cut query solves without error");
    if (r.has_value()) {
        t.expect(r->size() == 1, "soft cut with no condition solutions takes else branch");
        if (r->size() >= 1) {
            t.expect(bound((*r)[0], "X") == "2", "else branch binds X to 2");
        }
    }
})
.test("true_succeeds_once_and_binds_nothing", [](TestContext& t) {
    const auto p = prog("");
    const auto q1 = query("true.");
    auto r1 = solve(p, q1, 0);
    t.expect(r1.has_value(), "true goal solves without error");
    if (r1.has_value()) {
        t.expect(r1->size() == 1, "true succeeds exactly once");
    }
    const auto q2 = query("X = 42, true.");
    auto r2 = solve(p, q2, 0);
    t.expect(r2.has_value(), "conjunction with true solves without error");
    if (r2.has_value()) {
        t.expect(r2->size() == 1, "conjunction with true produces one answer");
        if (r2->size() >= 1) {
            t.expect(bound((*r2)[0], "X") == "42", "true preserves binding of X = 42");
        }
    }
})
.test("fail_and_false_produce_no_answers_and_no_error", [](TestContext& t) {
    const auto p = prog("");
    const auto q_fail = query("fail.");
    auto r_fail = solve(p, q_fail, 0);
    t.expect(r_fail.has_value(), "fail goal returns successful Result");
    if (r_fail.has_value()) {
        t.expect(r_fail->empty(), "fail produces exactly zero answers");
    }
    const auto q_false = query("false.");
    auto r_false = solve(p, q_false, 0);
    t.expect(r_false.has_value(), "false goal returns successful Result");
    if (r_false.has_value()) {
        t.expect(r_false->empty(), "false produces exactly zero answers");
    }
})
.test("call_with_runtime_bound_compound_goal_executes_correctly", [](TestContext& t) {
    const auto p = prog("foo(1, a). foo(2, b).");
    const auto q1 = query("G = foo(1, Y), call(G).");
    auto r1 = solve(p, q1, 0);
    t.expect(r1.has_value(), "call with bound goal solves without error");
    if (r1.has_value()) {
        t.expect(r1->size() == 1, "call(foo(1, Y)) produces exactly one answer");
        if (r1->size() >= 1) {
            t.expect(bound((*r1)[0], "G") == "foo(1, a)", "G is bound to foo(1, a)");
            t.expect(bound((*r1)[0], "Y") == "a", "Y is bound to a");
        }
    }
    const auto q2 = query("G = foo(X, Y), call(G).");
    auto r2 = solve(p, q2, 0);
    t.expect(r2.has_value(), "call with multiple solutions solves without error");
    if (r2.has_value()) {
        t.expect(r2->size() == 2, "call(foo(X, Y)) produces both solutions");
        if (r2->size() >= 1) {
            t.expect(bound((*r2)[0], "G") == "foo(1, a)", "answer 0 has G = foo(1, a)");
            t.expect(bound((*r2)[0], "X") == "1", "answer 0 has X = 1");
            t.expect(bound((*r2)[0], "Y") == "a", "answer 0 has Y = a");
        }
        if (r2->size() >= 2) {
            t.expect(bound((*r2)[1], "G") == "foo(2, b)", "answer 1 has G = foo(2, b)");
            t.expect(bound((*r2)[1], "X") == "2", "answer 1 has X = 2");
            t.expect(bound((*r2)[1], "Y") == "b", "answer 1 has Y = b");
        }
    }
})
.test("call_with_extra_arguments_appends_arguments_to_functor", [](TestContext& t) {
    const auto p = prog("foo(a). foo(a, b).");
    const auto q1 = query("call(foo, X).");
    auto r1 = solve(p, q1, 0);
    t.expect(r1.has_value(), "call(foo, X) solves without error");
    if (r1.has_value()) {
        t.expect(r1->size() == 1, "call(foo, X) invokes foo/1 producing one answer");
        if (r1->size() >= 1) {
            t.expect(bound((*r1)[0], "X") == "a", "X is bound to a");
        }
    }
    const auto q2 = query("call(foo(a), Y).");
    auto r2 = solve(p, q2, 0);
    t.expect(r2.has_value(), "call(foo(a), Y) solves without error");
    if (r2.has_value()) {
        t.expect(r2->size() == 1, "call(foo(a), Y) invokes foo/2 producing one answer");
        if (r2->size() >= 1) {
            t.expect(bound((*r2)[0], "Y") == "b", "Y is bound to b");
        }
    }
    const auto q3 = query("call(foo, a).");
    auto r3 = solve(p, q3, 0);
    t.expect(r3.has_value(), "call(foo, a) solves without error");
    if (r3.has_value()) {
        t.expect(r3->size() == 1, "call(foo, a) succeeds once");
    }
    const auto q4 = query("call(foo(a), b).");
    auto r4 = solve(p, q4, 0);
    t.expect(r4.has_value(), "call(foo(a), b) solves without error");
    if (r4.has_value()) {
        t.expect(r4->size() == 1, "call(foo(a), b) succeeds once");
    }
})
.test("call_with_unbound_variable_returns_domain_error", [](TestContext& t) {
    const auto p = prog("");
    const auto q = query("call(X).");
    auto r = solve(p, q, 0);
    t.expect(!r.has_value(), "call on unbound variable must fail");
    if (!r.has_value()) {
        t.expect(r.error() == MathError::domain_error, "error is MathError::domain_error");
    }
})
.test("call_with_integer_returns_domain_error", [](TestContext& t) {
    const auto p = prog("");
    const auto q = query("call(3).");
    auto r = solve(p, q, 0);
    t.expect(!r.has_value(), "call on integer must fail");
    if (!r.has_value()) {
        t.expect(r.error() == MathError::domain_error, "error is MathError::domain_error");
    }
})
.test("solve_or_parallel_agrees_with_solve_on_program_containing_cut", [](TestContext& t) {
    const auto p = prog("p(1). p(2). p(3). q(X) :- p(X), !.");
    const auto q = query("q(X).");
    auto r_serial = solve(p, q, 0);
    auto r_parallel = solve_or_parallel(p, q, 0);
    t.expect(r_serial.has_value(), "serial solve succeeds");
    t.expect(r_parallel.has_value(), "parallel solve succeeds");
    if (r_serial.has_value() && r_parallel.has_value()) {
        t.expect(r_serial->size() == 1, "serial solve has 1 solution");
        t.expect(r_parallel->size() == 1, "parallel solve has 1 solution");
        t.expect(r_parallel->size() == r_serial->size(), "parallel and serial agree on solution count");
        if (r_serial->size() >= 1 && r_parallel->size() >= 1) {
            t.expect(bound((*r_parallel)[0], "X") == bound((*r_serial)[0], "X"), "parallel and serial agree on binding of X");
            t.expect(bound((*r_parallel)[0], "X") == "1", "binding of X is 1");
        }
    }
    const auto p2 = prog("p(1). p(2). r(X, Y) :- p(X), !, p(Y).");
    const auto q2 = query("r(X, Y).");
    auto r2_serial = solve(p2, q2, 0);
    auto r2_parallel = solve_or_parallel(p2, q2, 0);
    t.expect(r2_serial.has_value(), "serial solve on multi-solution cut succeeds");
    t.expect(r2_parallel.has_value(), "parallel solve on multi-solution cut succeeds");
    if (r2_serial.has_value() && r2_parallel.has_value()) {
        t.expect(r2_parallel->size() == r2_serial->size(), "parallel and serial agree on size of 2 answers");
        t.expect(r2_serial->size() == 2, "serial solve has 2 solutions");
        if (r2_serial->size() >= 2 && r2_parallel->size() >= 2) {
            t.expect(bound((*r2_parallel)[0], "X") == bound((*r2_serial)[0], "X"), "answer 0 agrees on X");
            t.expect(bound((*r2_parallel)[0], "Y") == bound((*r2_serial)[0], "Y"), "answer 0 agrees on Y");
            t.expect(bound((*r2_parallel)[1], "X") == bound((*r2_serial)[1], "X"), "answer 1 agrees on X");
            t.expect(bound((*r2_parallel)[1], "Y") == bound((*r2_serial)[1], "Y"), "answer 1 agrees on Y");
            t.expect(bound((*r2_parallel)[0], "X") == "1", "answer 0 has X = 1");
            t.expect(bound((*r2_parallel)[0], "Y") == "1", "answer 0 has Y = 1");
            t.expect(bound((*r2_parallel)[1], "X") == "1", "answer 1 has X = 1");
            t.expect(bound((*r2_parallel)[1], "Y") == "2", "answer 1 has Y = 2");
        }
    }
})
.test("is_evaluates_arithmetic_with_precedence_and_parentheses",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("X is 2+3*4."), 0);
          t.expect(r1.has_value(), "solve succeeded for 2+3*4");
          if (!r1.has_value()) return;
          t.expect(r1->size() == 1, "exactly one solution for 2+3*4");
          if (r1->size() < 1) return;
          t.expect(bound((*r1)[0], "X") == "14", "2+3*4 evaluates to 14 by operator precedence");

          auto r2 = solve(prog(""), query("X is (2+3)*4."), 0);
          t.expect(r2.has_value(), "solve succeeded for (2+3)*4");
          if (!r2.has_value()) return;
          t.expect(r2->size() == 1, "exactly one solution for (2+3)*4");
          if (r2->size() < 1) return;
          t.expect(bound((*r2)[0], "X") == "20", "(2+3)*4 evaluates to 20 with parentheses");
      })
.test("is_evaluates_addition_subtraction_multiplication_and_unary_minus",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("X is 10 + 5."), 0);
          t.expect(r1.has_value(), "solve succeeded for 10 + 5");
          if (!r1.has_value()) return;
          t.expect(r1->size() == 1, "exactly one solution for 10 + 5");
          if (r1->size() < 1) return;
          t.expect(bound((*r1)[0], "X") == "15", "10 + 5 evaluates to 15");

          auto r2 = solve(prog(""), query("X is 10 - 3."), 0);
          t.expect(r2.has_value(), "solve succeeded for 10 - 3");
          if (!r2.has_value()) return;
          t.expect(r2->size() == 1, "exactly one solution for 10 - 3");
          if (r2->size() < 1) return;
          t.expect(bound((*r2)[0], "X") == "7", "10 - 3 evaluates to 7");

          auto r3 = solve(prog(""), query("X is 6 * 7."), 0);
          t.expect(r3.has_value(), "solve succeeded for 6 * 7");
          if (!r3.has_value()) return;
          t.expect(r3->size() == 1, "exactly one solution for 6 * 7");
          if (r3->size() < 1) return;
          t.expect(bound((*r3)[0], "X") == "42", "6 * 7 evaluates to 42");

          auto r4 = solve(prog(""), query("X is - 7."), 0);
          t.expect(r4.has_value(), "solve succeeded for unary minus");
          if (!r4.has_value()) return;
          t.expect(r4->size() == 1, "exactly one solution for unary minus");
          if (r4->size() < 1) return;
          t.expect(bound((*r4)[0], "X") == "-7", "unary minus of 7 evaluates to -7");
      })
.test("is_evaluates_integer_division_and_modulo_truncating_and_flooring",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("X is -7 // 2."), 0);
          t.expect(r1.has_value(), "solve succeeded for -7 // 2");
          if (!r1.has_value()) return;
          t.expect(r1->size() == 1, "exactly one solution for -7 // 2");
          if (r1->size() < 1) return;
          t.expect(bound((*r1)[0], "X") == "-3", "-7 // 2 truncates toward zero to -3");

          auto r2 = solve(prog(""), query("X is 7 // 2."), 0);
          t.expect(r2.has_value(), "solve succeeded for 7 // 2");
          if (!r2.has_value()) return;
          t.expect(r2->size() == 1, "exactly one solution for 7 // 2");
          if (r2->size() < 1) return;
          t.expect(bound((*r2)[0], "X") == "3", "7 // 2 truncates toward zero to 3");

          auto r3 = solve(prog(""), query("X is -7 div 2."), 0);
          t.expect(r3.has_value(), "solve succeeded for -7 div 2");
          if (!r3.has_value()) return;
          t.expect(r3->size() == 1, "exactly one solution for -7 div 2");
          if (r3->size() < 1) return;
          t.expect(bound((*r3)[0], "X") == "-4", "-7 div 2 floors toward negative infinity to -4");

          auto r4 = solve(prog(""), query("X is 7 div 2."), 0);
          t.expect(r4.has_value(), "solve succeeded for 7 div 2");
          if (!r4.has_value()) return;
          t.expect(r4->size() == 1, "exactly one solution for 7 div 2");
          if (r4->size() < 1) return;
          t.expect(bound((*r4)[0], "X") == "3", "7 div 2 floors toward negative infinity to 3");

          auto r5 = solve(prog(""), query("X is -7 mod 2."), 0);
          t.expect(r5.has_value(), "solve succeeded for -7 mod 2");
          if (!r5.has_value()) return;
          t.expect(r5->size() == 1, "exactly one solution for -7 mod 2");
          if (r5->size() < 1) return;
          t.expect(bound((*r5)[0], "X") == "1", "-7 mod 2 takes sign of divisor 2 and gives 1");

          auto r6 = solve(prog(""), query("X is 7 mod -2."), 0);
          t.expect(r6.has_value(), "solve succeeded for 7 mod -2");
          if (!r6.has_value()) return;
          t.expect(r6->size() == 1, "exactly one solution for 7 mod -2");
          if (r6->size() < 1) return;
          t.expect(bound((*r6)[0], "X") == "-1", "7 mod -2 takes sign of divisor -2 and gives -1");

          auto r7 = solve(prog(""), query("X is -7 rem 2."), 0);
          t.expect(r7.has_value(), "solve succeeded for -7 rem 2");
          if (!r7.has_value()) return;
          t.expect(r7->size() == 1, "exactly one solution for -7 rem 2");
          if (r7->size() < 1) return;
          t.expect(bound((*r7)[0], "X") == "-1", "-7 rem 2 takes sign of dividend -7 and gives -1");

          auto r8 = solve(prog(""), query("X is 7 rem -2."), 0);
          t.expect(r8.has_value(), "solve succeeded for 7 rem -2");
          if (!r8.has_value()) return;
          t.expect(r8->size() == 1, "exactly one solution for 7 rem -2");
          if (r8->size() < 1) return;
          t.expect(bound((*r8)[0], "X") == "1", "7 rem -2 takes sign of dividend 7 and gives 1");
      })
.test("is_evaluates_min_max_abs_sign_and_gcd",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("X is min(8, 3)."), 0);
          t.expect(r1.has_value(), "solve succeeded for min(8, 3)");
          if (!r1.has_value()) return;
          t.expect(r1->size() == 1, "exactly one solution for min(8, 3)");
          if (r1->size() < 1) return;
          t.expect(bound((*r1)[0], "X") == "3", "min(8, 3) evaluates to 3");

          auto r2 = solve(prog(""), query("X is max(8, 3)."), 0);
          t.expect(r2.has_value(), "solve succeeded for max(8, 3)");
          if (!r2.has_value()) return;
          t.expect(r2->size() == 1, "exactly one solution for max(8, 3)");
          if (r2->size() < 1) return;
          t.expect(bound((*r2)[0], "X") == "8", "max(8, 3) evaluates to 8");

          auto r3 = solve(prog(""), query("X is abs(-42)."), 0);
          t.expect(r3.has_value(), "solve succeeded for abs(-42)");
          if (!r3.has_value()) return;
          t.expect(r3->size() == 1, "exactly one solution for abs(-42)");
          if (r3->size() < 1) return;
          t.expect(bound((*r3)[0], "X") == "42", "abs(-42) evaluates to 42");

          auto r4 = solve(prog(""), query("X is sign(-42)."), 0);
          t.expect(r4.has_value(), "solve succeeded for sign(-42)");
          if (!r4.has_value()) return;
          t.expect(r4->size() == 1, "exactly one solution for sign(-42)");
          if (r4->size() < 1) return;
          t.expect(bound((*r4)[0], "X") == "-1", "sign(-42) evaluates to -1");

          auto r5 = solve(prog(""), query("X is sign(42)."), 0);
          t.expect(r5.has_value(), "solve succeeded for sign(42)");
          if (!r5.has_value()) return;
          t.expect(r5->size() == 1, "exactly one solution for sign(42)");
          if (r5->size() < 1) return;
          t.expect(bound((*r5)[0], "X") == "1", "sign(42) evaluates to 1");

          auto r6 = solve(prog(""), query("X is sign(0)."), 0);
          t.expect(r6.has_value(), "solve succeeded for sign(0)");
          if (!r6.has_value()) return;
          t.expect(r6->size() == 1, "exactly one solution for sign(0)");
          if (r6->size() < 1) return;
          t.expect(bound((*r6)[0], "X") == "0", "sign(0) evaluates to 0");

          auto r7 = solve(prog(""), query("X is gcd(12, 18)."), 0);
          t.expect(r7.has_value(), "solve succeeded for gcd(12, 18)");
          if (!r7.has_value()) return;
          t.expect(r7->size() == 1, "exactly one solution for gcd(12, 18)");
          if (r7->size() < 1) return;
          t.expect(bound((*r7)[0], "X") == "6", "gcd(12, 18) evaluates to 6");
      })
.test("is_evaluates_bitwise_and_shift_operators",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("X is 16 >> 2."), 0);
          t.expect(r1.has_value(), "solve succeeded for 16 >> 2");
          if (!r1.has_value()) return;
          t.expect(r1->size() == 1, "exactly one solution for 16 >> 2");
          if (r1->size() < 1) return;
          t.expect(bound((*r1)[0], "X") == "4", "16 >> 2 evaluates to 4");

          auto r2 = solve(prog(""), query("X is 3 << 3."), 0);
          t.expect(r2.has_value(), "solve succeeded for 3 << 3");
          if (!r2.has_value()) return;
          t.expect(r2->size() == 1, "exactly one solution for 3 << 3");
          if (r2->size() < 1) return;
          t.expect(bound((*r2)[0], "X") == "24", "3 << 3 evaluates to 24");

          auto r3 = solve(prog(""), query("X is 12 /\\ 10."), 0);
          t.expect(r3.has_value(), "solve succeeded for 12 /\\ 10");
          if (!r3.has_value()) return;
          t.expect(r3->size() == 1, "exactly one solution for 12 /\\ 10");
          if (r3->size() < 1) return;
          t.expect(bound((*r3)[0], "X") == "8", "12 /\\ 10 evaluates to 8");

          auto r4 = solve(prog(""), query("X is 12 \\/ 10."), 0);
          t.expect(r4.has_value(), "solve succeeded for 12 \\/ 10");
          if (!r4.has_value()) return;
          t.expect(r4->size() == 1, "exactly one solution for 12 \\/ 10");
          if (r4->size() < 1) return;
          t.expect(bound((*r4)[0], "X") == "14", "12 \\/ 10 evaluates to 14");

          auto r5 = solve(prog(""), query("X is 12 xor 10."), 0);
          t.expect(r5.has_value(), "solve succeeded for 12 xor 10");
          if (!r5.has_value()) return;
          t.expect(r5->size() == 1, "exactly one solution for 12 xor 10");
          if (r5->size() < 1) return;
          t.expect(bound((*r5)[0], "X") == "6", "12 xor 10 evaluates to 6");

          auto r6 = solve(prog(""), query("X is \\ 0."), 0);
          t.expect(r6.has_value(), "solve succeeded for \\ 0");
          if (!r6.has_value()) return;
          t.expect(r6->size() == 1, "exactly one solution for \\ 0");
          if (r6->size() < 1) return;
          t.expect(bound((*r6)[0], "X") == "-1", "\\ 0 evaluates to bitwise NOT of 0 (-1)");
      })
.test("is_division_requires_exact_integer_result",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("X is 6/2."), 0);
          t.expect(r1.has_value(), "exact division 6/2 succeeds");
          if (!r1.has_value()) return;
          t.expect(r1->size() == 1, "6/2 yields exactly one solution");
          if (r1->size() < 1) return;
          t.expect(bound((*r1)[0], "X") == "3", "6/2 evaluates to 3");

          auto r2 = solve(prog(""), query("X is 7/2."), 0);
          t.expect(!r2.has_value(), "inexact division 7/2 fails because float results do not exist");
          if (!r2.has_value()) {
              t.expect(r2.error() == MathError::inexact, "7/2 reports MathError::inexact");
          }
      })
.test("is_division_by_zero_fails_with_division_by_zero_error",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("X is 1/0."), 0);
          t.expect(!r1.has_value(), "division by zero returns an error");
          if (!r1.has_value()) {
              t.expect(r1.error() == MathError::division_by_zero, "1/0 reports MathError::division_by_zero");
          }

          auto r2 = solve(prog(""), query("X is 1 mod 0."), 0);
          t.expect(!r2.has_value(), "modulo by zero returns an error");
          if (!r2.has_value()) {
              t.expect(r2.error() == MathError::division_by_zero, "1 mod 0 reports MathError::division_by_zero");
          }
      })
.test("is_overflow_checked_on_limits_negation_and_exponentiation",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("X is max_integer + 1."), 0);
          t.expect(!r1.has_value(), "max_integer + 1 returns an error");
          if (!r1.has_value()) {
              t.expect(r1.error() == MathError::overflow, "max_integer + 1 reports MathError::overflow");
          }

          auto r2 = solve(prog(""), query("X is -min_integer."), 0);
          t.expect(!r2.has_value(), "-min_integer returns an error");
          if (!r2.has_value()) {
              t.expect(r2.error() == MathError::overflow, "-min_integer reports MathError::overflow");
          }

          auto r3 = solve(prog(""), query("X is 2^63."), 0);
          t.expect(!r3.has_value(), "2^63 returns an error");
          if (!r3.has_value()) {
              t.expect(r3.error() == MathError::overflow, "2^63 reports MathError::overflow");
          }
      })
.test("is_exponentiation_computes_powers_and_enforces_domain",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("X is 2^10."), 0);
          t.expect(r1.has_value(), "2^10 succeeds");
          if (!r1.has_value()) return;
          t.expect(r1->size() == 1, "2^10 returns one solution");
          if (r1->size() < 1) return;
          t.expect(bound((*r1)[0], "X") == "1024", "2^10 evaluates to 1024");

          auto r2 = solve(prog(""), query("X is 2^(-1)."), 0);
          t.expect(!r2.has_value(), "2^(-1) returns an error");
          if (!r2.has_value()) {
              t.expect(r2.error() == MathError::domain_error, "2^(-1) reports MathError::domain_error");
          }

          auto r3 = solve(prog(""), query("X is (-1)^(-3)."), 0);
          t.expect(r3.has_value(), "(-1)^(-3) succeeds");
          if (!r3.has_value()) return;
          t.expect(r3->size() == 1, "(-1)^(-3) returns one solution");
          if (r3->size() < 1) return;
          t.expect(bound((*r3)[0], "X") == "-1", "(-1)^(-3) evaluates to -1");
      })
.test("is_unbound_variable_operand_yields_domain_error",
      [](TestContext& t) {
          auto r = solve(prog(""), query("X is Y."), 0);
          t.expect(!r.has_value(), "X is Y returns an error when Y is uninstantiated");
          if (!r.has_value()) {
              t.expect(r.error() == MathError::domain_error, "uninstantiated arithmetic variable reports domain_error");
          }
      })
.test("is_float_functions_report_not_implemented",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("X is sqrt(4)."), 0);
          t.expect(!r1.has_value(), "sqrt(4) returns an error");
          if (!r1.has_value()) {
              t.expect(r1.error() == MathError::not_implemented, "sqrt reports MathError::not_implemented");
          }

          auto r2 = solve(prog(""), query("X is pi."), 0);
          t.expect(!r2.has_value(), "pi returns an error");
          if (!r2.has_value()) {
              t.expect(r2.error() == MathError::not_implemented, "pi reports MathError::not_implemented");
          }

          auto r3 = solve(prog(""), query("X is sin(0)."), 0);
          t.expect(!r3.has_value(), "sin(0) returns an error");
          if (!r3.has_value()) {
              t.expect(r3.error() == MathError::not_implemented, "sin reports MathError::not_implemented");
          }
      })
.test("is_unknown_functors_and_atoms_yield_domain_error",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("X is foo."), 0);
          t.expect(!r1.has_value(), "unknown atom in is/2 returns an error");
          if (!r1.has_value()) {
              t.expect(r1.error() == MathError::domain_error, "unknown atom reports MathError::domain_error");
          }

          auto r2 = solve(prog(""), query("X is foo(1)."), 0);
          t.expect(!r2.has_value(), "unknown compound in is/2 returns an error");
          if (!r2.has_value()) {
              t.expect(r2.error() == MathError::domain_error, "unknown compound reports MathError::domain_error");
          }
      })
.test("arithmetic_equality_evaluates_expressions_and_distinguishes_failure_from_error",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("1+1 =:= 2."), 0);
          t.expect(r1.has_value(), "1+1 =:= 2 query succeeds without error");
          if (r1.has_value()) {
              t.expect(r1->size() == 1, "1+1 =:= 2 yields exactly 1 answer");
          }

          auto r2 = solve(prog(""), query("1 =:= 2."), 0);
          t.expect(r2.has_value(), "1 =:= 2 succeeds as a query evaluation without error");
          if (r2.has_value()) {
              t.expect(r2->empty(), "1 =:= 2 fails honestly with zero answers");
          }
      })
.test("arithmetic_equality_is_not_structural_equality",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("1+1 =:= 2."), 0);
          t.expect(r1.has_value(), "arithmetic equality evaluation succeeds");
          if (r1.has_value()) {
              t.expect(r1->size() == 1, "1+1 =:= 2 succeeds arithmetically");
          }

          auto r2 = solve(prog(""), query("1+1 == 2."), 0);
          t.expect(r2.has_value(), "structural equality evaluation succeeds without error");
          if (r2.has_value()) {
              t.expect(r2->empty(), "1+1 == 2 fails structurally");
          }
      })
.test("arithmetic_inequality_and_ordering_comparisons",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("1+1 =\\= 3."), 0);
          t.expect(r1.has_value(), "1+1 =\\= 3 evaluation succeeds");
          if (r1.has_value()) {
              t.expect(r1->size() == 1, "2 =\\= 3 succeeds");
          }

          auto r2 = solve(prog(""), query("1+1 =\\= 2."), 0);
          t.expect(r2.has_value(), "1+1 =\\= 2 evaluation succeeds without error");
          if (r2.has_value()) {
              t.expect(r2->empty(), "2 =\\= 2 fails");
          }

          auto r3 = solve(prog(""), query("2 < 3."), 0);
          t.expect(r3.has_value(), "2 < 3 evaluation succeeds");
          if (r3.has_value()) {
              t.expect(r3->size() == 1, "2 < 3 succeeds");
          }

          auto r4 = solve(prog(""), query("3 < 2."), 0);
          t.expect(r4.has_value(), "3 < 2 evaluation succeeds");
          if (r4.has_value()) {
              t.expect(r4->empty(), "3 < 2 fails");
          }

          auto r5 = solve(prog(""), query("3 > 2."), 0);
          t.expect(r5.has_value(), "3 > 2 evaluation succeeds");
          if (r5.has_value()) {
              t.expect(r5->size() == 1, "3 > 2 succeeds");
          }

          auto r6 = solve(prog(""), query("2 > 3."), 0);
          t.expect(r6.has_value(), "2 > 3 evaluation succeeds");
          if (r6.has_value()) {
              t.expect(r6->empty(), "2 > 3 fails");
          }

          auto r7 = solve(prog(""), query("2 =< 2, 2 =< 3."), 0);
          t.expect(r7.has_value(), "2 =< 2 and 2 =< 3 evaluation succeeds");
          if (r7.has_value()) {
              t.expect(r7->size() == 1, "2 =< 2 and 2 =< 3 both hold");
          }

          auto r8 = solve(prog(""), query("3 =< 2."), 0);
          t.expect(r8.has_value(), "3 =< 2 evaluation succeeds");
          if (r8.has_value()) {
              t.expect(r8->empty(), "3 =< 2 fails");
          }

          auto r9 = solve(prog(""), query("2 >= 2, 3 >= 2."), 0);
          t.expect(r9.has_value(), "2 >= 2 and 3 >= 2 evaluation succeeds");
          if (r9.has_value()) {
              t.expect(r9->size() == 1, "2 >= 2 and 3 >= 2 both hold");
          }

          auto r10 = solve(prog(""), query("2 >= 3."), 0);
          t.expect(r10.has_value(), "2 >= 3 evaluation succeeds");
          if (r10.has_value()) {
              t.expect(r10->empty(), "2 >= 3 fails");
          }
      })
.test("unification_binds_terms_while_not_unifiable_tests_non_unifiability_without_binding",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("X = foo."), 0);
          t.expect(r1.has_value(), "X = foo succeeds");
          if (!r1.has_value()) return;
          t.expect(r1->size() == 1, "X = foo yields one answer");
          if (r1->size() < 1) return;
          t.expect(bound((*r1)[0], "X") == "foo", "X is bound to foo");

          auto r2 = solve(prog(""), query("foo(X, b) = foo(a, Y)."), 0);
          t.expect(r2.has_value(), "compound unification succeeds");
          if (!r2.has_value()) return;
          t.expect(r2->size() == 1, "compound unification yields one answer");
          if (r2->size() < 1) return;
          t.expect(bound((*r2)[0], "X") == "a", "X is bound to a");
          t.expect(bound((*r2)[0], "Y") == "b", "Y is bound to b");

          auto r3 = solve(prog(""), query("a \\= b."), 0);
          t.expect(r3.has_value(), "a \\= b evaluation succeeds");
          if (r3.has_value()) {
              t.expect(r3->size() == 1, "a \\= b succeeds because distinct atoms cannot unify");
          }

          auto r4 = solve(prog(""), query("a \\= a."), 0);
          t.expect(r4.has_value(), "a \\= a evaluation succeeds");
          if (r4.has_value()) {
              t.expect(r4->empty(), "a \\= a fails because a unifies with a");
          }

          auto r5 = solve(prog(""), query("X \\= a."), 0);
          t.expect(r5.has_value(), "X \\= a evaluation succeeds");
          if (r5.has_value()) {
              t.expect(r5->empty(), "X \\= a fails because variable X can unify with a");
          }
      })
.test("structural_equality_compares_variables_without_binding",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("X == Y."), 0);
          t.expect(r1.has_value(), "X == Y evaluation succeeds");
          if (r1.has_value()) {
              t.expect(r1->empty(), "distinct unbound variables fail structural equality");
          }

          auto r2 = solve(prog(""), query("X == X."), 0);
          t.expect(r2.has_value(), "X == X evaluation succeeds");
          if (r2.has_value()) {
              t.expect(r2->size() == 1, "identical variable is structurally equal to itself");
          }

          auto r3 = solve(prog(""), query("X = Y, X == Y."), 0);
          t.expect(r3.has_value(), "aliasing evaluation succeeds");
          if (r3.has_value()) {
              t.expect(r3->size() == 1, "aliased variables are structurally equal");
          }

          auto r4 = solve(prog(""), query("X \\== Y."), 0);
          t.expect(r4.has_value(), "X \\== Y evaluation succeeds");
          if (r4.has_value()) {
              t.expect(r4->size() == 1, "distinct unbound variables satisfy \\==");
          }

          auto r5 = solve(prog(""), query("X = Y, X \\== Y."), 0);
          t.expect(r5.has_value(), "aliased \\== evaluation succeeds");
          if (r5.has_value()) {
              t.expect(r5->empty(), "aliased variables fail \\==");
          }
      })
.test("standard_order_ranks_variable_then_integer_then_atom_then_compound",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("1 @< a."), 0);
          t.expect(r1.has_value(), "1 @< a evaluation succeeds");
          if (r1.has_value()) {
              t.expect(r1->size() == 1, "Integer ranks before Atom in standard order");
          }

          auto r2 = solve(prog(""), query("a @< f(x)."), 0);
          t.expect(r2.has_value(), "a @< f(x) evaluation succeeds");
          if (r2.has_value()) {
              t.expect(r2->size() == 1, "Atom ranks before Compound in standard order");
          }

          auto r3 = solve(prog(""), query("f(a, b) @> g(a)."), 0);
          t.expect(r3.has_value(), "f(a, b) @> g(a) evaluation succeeds");
          if (r3.has_value()) {
              t.expect(r3->size() == 1, "Arity 2 dominates Arity 1 regardless of functor name");
          }

          auto r4 = solve(prog(""), query("1 @=< 1, 1 @=< a."), 0);
          t.expect(r4.has_value(), "1 @=< 1 and 1 @=< a evaluation succeeds");
          if (r4.has_value()) {
              t.expect(r4->size() == 1, "1 @=< 1 and 1 @=< a both hold");
          }

          auto r5 = solve(prog(""), query("a @>= 1, a @>= a."), 0);
          t.expect(r5.has_value(), "a @>= 1 and a @>= a evaluation succeeds");
          if (r5.has_value()) {
              t.expect(r5->size() == 1, "a @>= 1 and a @>= a both hold");
          }
      })
.test("compare_predicate_binds_order_atom",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("compare(Order, a, b)."), 0);
          t.expect(r1.has_value(), "compare(Order, a, b) succeeds");
          if (!r1.has_value()) return;
          t.expect(r1->size() == 1, "compare(Order, a, b) yields one answer");
          if (r1->size() < 1) return;
          t.expect(bound((*r1)[0], "Order") == "<", "Order bound to atom '<'");
          t.expect(apply_substitution((*r1)[0], make_var("Order")) == make_atom("<"), "Order is atom '<'");

          auto r2 = solve(prog(""), query("compare(Order, b, a)."), 0);
          t.expect(r2.has_value(), "compare(Order, b, a) succeeds");
          if (!r2.has_value()) return;
          t.expect(r2->size() == 1, "compare(Order, b, a) yields one answer");
          if (r2->size() < 1) return;
          t.expect(bound((*r2)[0], "Order") == ">", "Order bound to atom '>'");
          t.expect(apply_substitution((*r2)[0], make_var("Order")) == make_atom(">"), "Order is atom '>'");

          auto r3 = solve(prog(""), query("compare(Order, a, a)."), 0);
          t.expect(r3.has_value(), "compare(Order, a, a) succeeds");
          if (!r3.has_value()) return;
          t.expect(r3->size() == 1, "compare(Order, a, a) yields one answer");
          if (r3->size() < 1) return;
          t.expect(bound((*r3)[0], "Order") == "=", "Order bound to atom '='");
          t.expect(apply_substitution((*r3)[0], make_var("Order")) == make_atom("="), "Order is atom '='");
      })
.test("direct_compare_terms_checks_order_ranks_and_compound_arity",
      [](TestContext& t) {
          t.expect(compare_terms(make_int(1), make_atom("a")) < 0,
                   "Integer ranks before Atom in standard order");
          t.expect(compare_terms(make_compound("f", {make_atom("a"), make_atom("b")}),
                                 make_compound("g", {make_atom("a")})) > 0,
                   "Compound of arity 2 ranks after Compound of arity 1");
      })
.test("var_distinguishes_unbound_variable_from_bound_term",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("var(X)."), 0);
          t.expect(r1.has_value(), "var(X) evaluation succeeds");
          if (r1.has_value()) {
              t.expect(r1->size() == 1, "var(X) succeeds for unbound variable");
          }

          auto r2 = solve(prog(""), query("var(a)."), 0);
          t.expect(r2.has_value(), "var(a) evaluation succeeds");
          if (r2.has_value()) {
              t.expect(r2->empty(), "var(a) fails for atom");
          }

          auto r3 = solve(prog(""), query("var(1)."), 0);
          t.expect(r3.has_value(), "var(1) evaluation succeeds");
          if (r3.has_value()) {
              t.expect(r3->empty(), "var(1) fails for integer");
          }
      })
.test("nonvar_distinguishes_instantiated_term_from_unbound_variable",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("nonvar(a)."), 0);
          t.expect(r1.has_value(), "nonvar(a) evaluation succeeds");
          if (r1.has_value()) {
              t.expect(r1->size() == 1, "nonvar(a) succeeds for atom");
          }

          auto r2 = solve(prog(""), query("nonvar(1)."), 0);
          t.expect(r2.has_value(), "nonvar(1) evaluation succeeds");
          if (r2.has_value()) {
              t.expect(r2->size() == 1, "nonvar(1) succeeds for integer");
          }

          auto r3 = solve(prog(""), query("nonvar(X)."), 0);
          t.expect(r3.has_value(), "nonvar(X) evaluation succeeds");
          if (r3.has_value()) {
              t.expect(r3->empty(), "nonvar(X) fails for unbound variable");
          }
      })
.test("atom_identifies_atoms_and_rejects_integers_compounds_and_variables",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("atom(foo)."), 0);
          t.expect(r1.has_value(), "atom(foo) evaluation succeeds");
          if (r1.has_value()) {
              t.expect(r1->size() == 1, "atom(foo) succeeds for atom");
          }

          auto r2 = solve(prog(""), query("atom([])."), 0);
          t.expect(r2.has_value(), "atom([]) evaluation succeeds");
          if (r2.has_value()) {
              t.expect(r2->size() == 1, "atom([]) succeeds for empty list atom");
          }

          auto r3 = solve(prog(""), query("atom(123)."), 0);
          t.expect(r3.has_value(), "atom(123) evaluation succeeds");
          if (r3.has_value()) {
              t.expect(r3->empty(), "atom(123) fails for integer");
          }

          auto r4 = solve(prog(""), query("atom(f(a))."), 0);
          t.expect(r4.has_value(), "atom(f(a)) evaluation succeeds");
          if (r4.has_value()) {
              t.expect(r4->empty(), "atom(f(a)) fails for compound");
          }

          auto r5 = solve(prog(""), query("atom(X)."), 0);
          t.expect(r5.has_value(), "atom(X) evaluation succeeds");
          if (r5.has_value()) {
              t.expect(r5->empty(), "atom(X) fails for unbound variable");
          }
      })
.test("number_and_integer_identify_integers_and_reject_atoms_and_variables",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("number(42), integer(42)."), 0);
          t.expect(r1.has_value(), "number and integer on 42 succeed");
          if (r1.has_value()) {
              t.expect(r1->size() == 1, "number(42) and integer(42) succeed");
          }

          auto r2 = solve(prog(""), query("number(-7), integer(-7)."), 0);
          t.expect(r2.has_value(), "number and integer on -7 succeed");
          if (r2.has_value()) {
              t.expect(r2->size() == 1, "number(-7) and integer(-7) succeed");
          }

          auto r3 = solve(prog(""), query("number(foo)."), 0);
          t.expect(r3.has_value(), "number(foo) evaluation succeeds");
          if (r3.has_value()) {
              t.expect(r3->empty(), "number(foo) fails for atom");
          }

          auto r4 = solve(prog(""), query("integer(foo)."), 0);
          t.expect(r4.has_value(), "integer(foo) evaluation succeeds");
          if (r4.has_value()) {
              t.expect(r4->empty(), "integer(foo) fails for atom");
          }

          auto r5 = solve(prog(""), query("number(X)."), 0);
          t.expect(r5.has_value(), "number(X) evaluation succeeds");
          if (r5.has_value()) {
              t.expect(r5->empty(), "number(X) fails for variable");
          }

          auto r6 = solve(prog(""), query("integer(X)."), 0);
          t.expect(r6.has_value(), "integer(X) evaluation succeeds");
          if (r6.has_value()) {
              t.expect(r6->empty(), "integer(X) fails for variable");
          }
      })
.test("float_always_fails_because_the_engine_has_no_float_type",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("float(1)."), 0);
          t.expect(r1.has_value(), "float(1) evaluation succeeds without error");
          if (r1.has_value()) {
              t.expect(r1->empty(), "float(1) fails because engine has no floats");
          }

          auto r2 = solve(prog(""), query("float(X)."), 0);
          t.expect(r2.has_value(), "float(X) evaluation succeeds without error");
          if (r2.has_value()) {
              t.expect(r2->empty(), "float(X) fails for unbound variable");
          }

          auto r3 = solve(prog(""), query("float(foo)."), 0);
          t.expect(r3.has_value(), "float(foo) evaluation succeeds without error");
          if (r3.has_value()) {
              t.expect(r3->empty(), "float(foo) fails for atom");
          }
      })
.test("atomic_accepts_atoms_and_integers_and_rejects_compounds_and_variables",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("atomic(foo)."), 0);
          t.expect(r1.has_value(), "atomic(foo) evaluation succeeds");
          if (r1.has_value()) {
              t.expect(r1->size() == 1, "atomic(foo) succeeds for atom");
          }

          auto r2 = solve(prog(""), query("atomic(42)."), 0);
          t.expect(r2.has_value(), "atomic(42) evaluation succeeds");
          if (r2.has_value()) {
              t.expect(r2->size() == 1, "atomic(42) succeeds for integer");
          }

          auto r3 = solve(prog(""), query("atomic(f(a))."), 0);
          t.expect(r3.has_value(), "atomic(f(a)) evaluation succeeds");
          if (r3.has_value()) {
              t.expect(r3->empty(), "atomic(f(a)) fails for compound");
          }

          auto r4 = solve(prog(""), query("atomic(X)."), 0);
          t.expect(r4.has_value(), "atomic(X) evaluation succeeds");
          if (r4.has_value()) {
              t.expect(r4->empty(), "atomic(X) fails for unbound variable");
          }
      })
.test("compound_identifies_compounds_and_rejects_atoms_integers_and_variables",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("compound(f(a))."), 0);
          t.expect(r1.has_value(), "compound(f(a)) evaluation succeeds");
          if (r1.has_value()) {
              t.expect(r1->size() == 1, "compound(f(a)) succeeds for compound");
          }

          auto r2 = solve(prog(""), query("compound([1, 2])."), 0);
          t.expect(r2.has_value(), "compound([1, 2]) evaluation succeeds");
          if (r2.has_value()) {
              t.expect(r2->size() == 1, "compound([1, 2]) succeeds for non-empty list");
          }

          auto r3 = solve(prog(""), query("compound(foo)."), 0);
          t.expect(r3.has_value(), "compound(foo) evaluation succeeds");
          if (r3.has_value()) {
              t.expect(r3->empty(), "compound(foo) fails for atom");
          }

          auto r4 = solve(prog(""), query("compound(42)."), 0);
          t.expect(r4.has_value(), "compound(42) evaluation succeeds");
          if (r4.has_value()) {
              t.expect(r4->empty(), "compound(42) fails for integer");
          }

          auto r5 = solve(prog(""), query("compound(X)."), 0);
          t.expect(r5.has_value(), "compound(X) evaluation succeeds");
          if (r5.has_value()) {
              t.expect(r5->empty(), "compound(X) fails for unbound variable");
          }
      })
.test("callable_accepts_atoms_and_compounds_and_rejects_integers_and_variables",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("callable(foo)."), 0);
          t.expect(r1.has_value(), "callable(foo) evaluation succeeds");
          if (r1.has_value()) {
              t.expect(r1->size() == 1, "callable(foo) succeeds for atom");
          }

          auto r2 = solve(prog(""), query("callable(f(a))."), 0);
          t.expect(r2.has_value(), "callable(f(a)) evaluation succeeds");
          if (r2.has_value()) {
              t.expect(r2->size() == 1, "callable(f(a)) succeeds for compound");
          }

          auto r3 = solve(prog(""), query("callable(123)."), 0);
          t.expect(r3.has_value(), "callable(123) evaluation succeeds");
          if (r3.has_value()) {
              t.expect(r3->empty(), "callable(123) fails for integer");
          }

          auto r4 = solve(prog(""), query("callable(X)."), 0);
          t.expect(r4.has_value(), "callable(X) evaluation succeeds");
          if (r4.has_value()) {
              t.expect(r4->empty(), "callable(X) fails for unbound variable");
          }
      })
.test("is_list_accepts_proper_lists_and_rejects_partial_lists_and_non_lists",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("is_list([a, b, c])."), 0);
          t.expect(r1.has_value(), "is_list([a, b, c]) evaluation succeeds");
          if (r1.has_value()) {
              t.expect(r1->size() == 1, "is_list([a, b, c]) succeeds for proper list");
          }

          auto r2 = solve(prog(""), query("is_list([])."), 0);
          t.expect(r2.has_value(), "is_list([]) evaluation succeeds");
          if (r2.has_value()) {
              t.expect(r2->size() == 1, "is_list([]) succeeds for empty list");
          }

          auto r3 = solve(prog(""), query("is_list(foo)."), 0);
          t.expect(r3.has_value(), "is_list(foo) evaluation succeeds");
          if (r3.has_value()) {
              t.expect(r3->empty(), "is_list(foo) fails for non-list atom");
          }

          auto r4 = solve(prog(""), query("is_list([a | X])."), 0);
          t.expect(r4.has_value(), "is_list([a | X]) evaluation succeeds");
          if (r4.has_value()) {
              t.expect(r4->empty(), "is_list([a | X]) fails for partial list with unbound tail");
          }
      })
.test("ground_detects_terms_free_of_unbound_variables",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("ground(f(a, 1))."), 0);
          t.expect(r1.has_value(), "ground(f(a, 1)) evaluation succeeds");
          if (r1.has_value()) {
              t.expect(r1->size() == 1, "ground(f(a, 1)) succeeds for term without variables");
          }

          auto r2 = solve(prog(""), query("ground([])."), 0);
          t.expect(r2.has_value(), "ground([]) evaluation succeeds");
          if (r2.has_value()) {
              t.expect(r2->size() == 1, "ground([]) succeeds for empty list");
          }

          auto r3 = solve(prog(""), query("ground(f(a, X))."), 0);
          t.expect(r3.has_value(), "ground(f(a, X)) evaluation succeeds");
          if (r3.has_value()) {
              t.expect(r3->empty(), "ground(f(a, X)) fails when containing unbound variable");
          }

          auto r4 = solve(prog(""), query("ground(X)."), 0);
          t.expect(r4.has_value(), "ground(X) evaluation succeeds");
          if (r4.has_value()) {
              t.expect(r4->empty(), "ground(X) fails for unbound variable");
          }
      })
.test("functor_inspects_compound_functor_and_arity",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("functor(foo(a, b), N, A)."), 0);
          t.expect(r1.has_value(), "functor inspection on compound succeeds");
          if (!r1.has_value()) return;
          t.expect(r1->size() == 1, "functor on compound yields one answer");
          if (r1->size() < 1) return;
          t.expect(bound((*r1)[0], "N") == "foo", "compound functor name is foo");
          t.expect(bound((*r1)[0], "A") == "2", "compound functor arity is 2");

          auto r2 = solve(prog(""), query("functor(foo, N, A)."), 0);
          t.expect(r2.has_value(), "functor inspection on atom succeeds");
          if (!r2.has_value()) return;
          t.expect(r2->size() == 1, "functor on atom yields one answer");
          if (r2->size() < 1) return;
          t.expect(bound((*r2)[0], "N") == "foo", "atom functor name is foo");
          t.expect(bound((*r2)[0], "A") == "0", "atom functor arity is 0");
      })
.test("functor_constructs_compound_with_fresh_variables",
      [](TestContext& t) {
          auto r = solve(prog(""), query("functor(T, foo, 2)."), 0);
          t.expect(r.has_value(), "functor construction succeeds");
          if (!r.has_value()) return;
          t.expect(r->size() == 1, "functor construction yields one answer");
          if (r->size() < 1) return;
          Term t_term = apply_substitution((*r)[0], make_var("T"));
          t.expect(is_compound(t_term), "constructed term is compound");
          if (is_compound(t_term)) {
              t.expect(compound_of(t_term).functor == "foo", "functor is foo");
              t.expect(compound_of(t_term).args.size() == 2, "arity is 2");
              t.expect(is_var(compound_of(t_term).args[0]), "arg 0 is a variable");
              t.expect(is_var(compound_of(t_term).args[1]), "arg 1 is a variable");
              t.expect(compound_of(t_term).args[0] != compound_of(t_term).args[1], "args 0 and 1 are distinct fresh variables");
          }
          t.expect(to_string(t_term) == "foo(_G0, _G1)", "to_string formats as foo(_G0, _G1)");
      })
.test("univ_deconstructs_and_reconstructs_compound_terms",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("foo(a, b) =.. L."), 0);
          t.expect(r1.has_value(), "univ deconstruction succeeds");
          if (!r1.has_value()) return;
          t.expect(r1->size() == 1, "univ deconstruction yields one answer");
          if (r1->size() < 1) return;
          t.expect(bound((*r1)[0], "L") == "[foo, a, b]", "foo(a, b) =.. [foo, a, b]");

          auto r2 = solve(prog(""), query("T =.. [foo, a, b]."), 0);
          t.expect(r2.has_value(), "univ reconstruction succeeds");
          if (!r2.has_value()) return;
          t.expect(r2->size() == 1, "univ reconstruction yields one answer");
          if (r2->size() < 1) return;
          t.expect(bound((*r2)[0], "T") == "foo(a, b)", "T =.. [foo, a, b] builds foo(a, b)");
      })
.test("arg_extracts_specified_argument_or_enumerates_all_arguments_on_backtracking",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("arg(2, foo(a, b, c), X)."), 0);
          t.expect(r1.has_value(), "arg with bound index succeeds");
          if (!r1.has_value()) return;
          t.expect(r1->size() == 1, "arg with bound index yields one answer");
          if (r1->size() < 1) return;
          t.expect(bound((*r1)[0], "X") == "b", "second argument of foo(a, b, c) is b");

          auto r2 = solve(prog(""), query("arg(N, foo(a, b), X)."), 0);
          t.expect(r2.has_value(), "arg with unbound index succeeds");
          if (!r2.has_value()) return;
          t.expect(r2->size() == 2, "arg with unbound index yields exactly 2 answers");
          if (r2->size() < 2) return;
          t.expect(bound((*r2)[0], "N") == "1", "first answer has N = 1");
          t.expect(bound((*r2)[0], "X") == "a", "first answer has X = a");
          t.expect(bound((*r2)[1], "N") == "2", "second answer has N = 2");
          t.expect(bound((*r2)[1], "X") == "b", "second answer has X = b");
      })
.test("copy_term_produces_fresh_variables_preserving_identity_structure",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("copy_term(f(X, X, Y), C), C == f(X, X, Y)."), 0);
          t.expect(r1.has_value(), "copy_term structural check query succeeds");
          if (r1.has_value()) {
              t.expect(r1->empty(), "copy is not identical to original under ==/2");
          }

          auto r2 = solve(prog(""), query("copy_term(f(X, X, Y), C), C = f(A, B, D), A == B, A \\== D."), 0);
          t.expect(r2.has_value(), "copy_term shape query succeeds");
          if (r2.has_value()) {
              t.expect(r2->size() == 1, "copy preserves internal variable sharing A == B and distinctness A \\== D");
          }
      })
.test("term_variables_extracts_unique_variables_in_order_of_first_appearance",
      [](TestContext& t) {
          auto r = solve(prog(""), query("term_variables(f(X, Y, X), L)."), 0);
          t.expect(r.has_value(), "term_variables succeeds");
          if (!r.has_value()) return;
          t.expect(r->size() == 1, "term_variables yields exactly 1 answer");
          if (r->size() < 1) return;
          t.expect(bound((*r)[0], "L") == "[X, Y]", "L contains [X, Y]");
          Term l_term = apply_substitution((*r)[0], make_var("L"));
          t.expect(is_cons(l_term), "L is a cons cell");
          if (is_cons(l_term)) {
              const auto& c1 = compound_of(l_term);
              t.expect(is_var(c1.args[0]), "first element is variable X");
              t.expect(is_cons(c1.args[1]), "tail is cons cell");
              if (is_cons(c1.args[1])) {
                  const auto& c2 = compound_of(c1.args[1]);
                  t.expect(is_var(c2.args[0]), "second element is variable Y");
                  t.expect(is_nil(c2.args[1]), "list terminates with []");
                  t.expect(c1.args[0] != c2.args[0], "variables X and Y are distinct");
              }
          }
      })
.test("succ_computes_successor_and_predecessor_enforcing_natural_numbers",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("succ(3, X)."), 0);
          t.expect(r1.has_value(), "succ(3, X) succeeds");
          if (!r1.has_value()) return;
          t.expect(r1->size() == 1, "succ(3, X) yields one answer");
          if (r1->size() < 1) return;
          t.expect(bound((*r1)[0], "X") == "4", "successor of 3 is 4");

          auto r2 = solve(prog(""), query("succ(X, 4)."), 0);
          t.expect(r2.has_value(), "succ(X, 4) succeeds");
          if (!r2.has_value()) return;
          t.expect(r2->size() == 1, "succ(X, 4) yields one answer");
          if (r2->size() < 1) return;
          t.expect(bound((*r2)[0], "X") == "3", "predecessor of 4 is 3");

          auto r3 = solve(prog(""), query("succ(X, 0)."), 0);
          t.expect(r3.has_value(), "succ(X, 0) evaluation succeeds");
          if (r3.has_value()) {
              t.expect(r3->empty(), "succ(X, 0) fails honestly with no natural predecessor");
          }

          auto r4 = solve(prog(""), query("succ(-1, X)."), 0);
          t.expect(!r4.has_value(), "succ(-1, X) returns error on negative input");
          if (!r4.has_value()) {
              t.expect(r4.error() == MathError::domain_error, "negative argument to succ/2 reports domain_error");
          }
      })
.test("plus_solves_for_any_single_missing_addend_or_sum",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("plus(1, 2, X)."), 0);
          t.expect(r1.has_value(), "plus(1, 2, X) succeeds");
          if (!r1.has_value()) return;
          t.expect(r1->size() == 1, "plus(1, 2, X) yields one answer");
          if (r1->size() < 1) return;
          t.expect(bound((*r1)[0], "X") == "3", "plus(1, 2, X) binds X to 3");

          auto r2 = solve(prog(""), query("plus(1, X, 3)."), 0);
          t.expect(r2.has_value(), "plus(1, X, 3) succeeds");
          if (!r2.has_value()) return;
          t.expect(r2->size() == 1, "plus(1, X, 3) yields one answer");
          if (r2->size() < 1) return;
          t.expect(bound((*r2)[0], "X") == "2", "plus(1, X, 3) binds X to 2");

          auto r3 = solve(prog(""), query("plus(X, 2, 3)."), 0);
          t.expect(r3.has_value(), "plus(X, 2, 3) succeeds");
          if (!r3.has_value()) return;
          t.expect(r3->size() == 1, "plus(X, 2, 3) yields one answer");
          if (r3->size() < 1) return;
          t.expect(bound((*r3)[0], "X") == "1", "plus(X, 2, 3) binds X to 1");
      })
.test("between_generates_integers_in_range_in_order_and_tests_membership",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("between(1, 3, X)."), 0);
          t.expect(r1.has_value(), "between(1, 3, X) succeeds");
          if (!r1.has_value()) return;
          t.expect(r1->size() == 3, "between(1, 3, X) yields exactly 3 answers");
          if (r1->size() < 3) return;
          t.expect(bound((*r1)[0], "X") == "1", "first answer is 1");
          t.expect(bound((*r1)[1], "X") == "2", "second answer is 2");
          t.expect(bound((*r1)[2], "X") == "3", "third answer is 3");

          auto r2 = solve(prog(""), query("between(1, 3, 2)."), 0);
          t.expect(r2.has_value(), "between(1, 3, 2) evaluation succeeds");
          if (r2.has_value()) {
              t.expect(r2->size() == 1, "between(1, 3, 2) succeeds once");
          }

          auto r3 = solve(prog(""), query("between(1, 3, 7)."), 0);
          t.expect(r3.has_value(), "between(1, 3, 7) evaluation succeeds");
          if (r3.has_value()) {
              t.expect(r3->empty(), "between(1, 3, 7) fails");
          }
      })
.test("length_computes_list_length_and_constructs_list_of_specified_length",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("length([a, b, c], N)."), 0);
          t.expect(r1.has_value(), "length on [a, b, c] succeeds");
          if (!r1.has_value()) return;
          t.expect(r1->size() == 1, "length on list yields one answer");
          if (r1->size() < 1) return;
          t.expect(bound((*r1)[0], "N") == "3", "length of [a, b, c] is 3");

          auto r2 = solve(prog(""), query("length(L, 2)."), 0);
          t.expect(r2.has_value(), "length(L, 2) construction succeeds");
          if (!r2.has_value()) return;
          t.expect(r2->size() == 1, "length(L, 2) yields one answer");
          if (r2->size() < 1) return;
          t.expect(bound((*r2)[0], "L") == "[_G0, _G1]", "L is a 2-element list [_G0, _G1]");
          Term l_term = apply_substitution((*r2)[0], make_var("L"));
          t.expect(is_cons(l_term), "L is cons");
          if (is_cons(l_term)) {
              const auto& c1 = compound_of(l_term);
              t.expect(is_var(c1.args[0]), "first element is variable");
              t.expect(is_cons(c1.args[1]), "tail is cons");
              if (is_cons(c1.args[1])) {
                  const auto& c2 = compound_of(c1.args[1]);
                  t.expect(is_var(c2.args[0]), "second element is variable");
                  t.expect(is_nil(c2.args[1]), "tail terminates with nil");
                  t.expect(c1.args[0] != c2.args[0], "two variables in list are distinct");
              }
          }
      })
.test("numlist_generates_inclusive_integer_sequence_as_list",
      [](TestContext& t) {
          auto r = solve(prog(""), query("numlist(1, 5, L)."), 0);
          t.expect(r.has_value(), "numlist(1, 5, L) succeeds");
          if (!r.has_value()) return;
          t.expect(r->size() == 1, "numlist yields one answer");
          if (r->size() < 1) return;
          t.expect(bound((*r)[0], "L") == "[1, 2, 3, 4, 5]", "numlist(1, 5, L) binds L to [1, 2, 3, 4, 5]");
      })
.test("msort_preserves_duplicates_while_sort_removes_them",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("msort([b, a, b], L)."), 0);
          t.expect(r1.has_value(), "msort succeeds");
          if (!r1.has_value()) return;
          t.expect(r1->size() == 1, "msort yields one answer");
          if (r1->size() < 1) return;
          t.expect(bound((*r1)[0], "L") == "[a, b, b]", "msort retains duplicates yielding [a, b, b]");

          auto r2 = solve(prog(""), query("sort([b, a, b], L)."), 0);
          t.expect(r2.has_value(), "sort succeeds");
          if (!r2.has_value()) return;
          t.expect(r2->size() == 1, "sort yields one answer");
          if (r2->size() < 1) return;
          t.expect(bound((*r2)[0], "L") == "[a, b]", "sort removes duplicates yielding [a, b]");
      })
.test("keysort_sorts_pairs_by_key_stably",
      [](TestContext& t) {
          auto r = solve(prog(""), query("keysort([b-1, a-2, b-3], L)."), 0);
          t.expect(r.has_value(), "keysort succeeds");
          if (!r.has_value()) return;
          t.expect(r->size() == 1, "keysort yields one answer");
          if (r->size() < 1) return;
          t.expect(bound((*r)[0], "L") == "[-(a, 2), -(b, 1), -(b, 3)]",
                   "keysort stably sorts pairs by key yielding [-(a, 2), -(b, 1), -(b, 3)]");
          Term l_term = apply_substitution((*r)[0], make_var("L"));
          t.expect(is_cons(l_term), "L is cons");
          if (is_cons(l_term)) {
              const auto& c1 = compound_of(l_term);
              t.expect(c1.args[0] == make_compound("-", {make_atom("a"), make_int(2)}), "first pair is a-2");
              t.expect(is_cons(c1.args[1]), "tail 1 is cons");
              if (is_cons(c1.args[1])) {
                  const auto& c2 = compound_of(c1.args[1]);
                  t.expect(c2.args[0] == make_compound("-", {make_atom("b"), make_int(1)}), "second pair is b-1");
                  t.expect(is_cons(c2.args[1]), "tail 2 is cons");
                  if (is_cons(c2.args[1])) {
                      const auto& c3 = compound_of(c2.args[1]);
                      t.expect(c3.args[0] == make_compound("-", {make_atom("b"), make_int(3)}), "third pair is b-3");
                      t.expect(is_nil(c3.args[1]), "tail terminates with nil");
                  }
              }
          }
      })
.test("sort_four_orders_descending_by_standard_order",
      [](TestContext& t) {
          auto r = solve(prog(""), query("sort(0, @>=, [1, 3, 2], L)."), 0);
          t.expect(r.has_value(), "sort/4 with @>= succeeds");
          if (!r.has_value()) return;
          t.expect(r->size() == 1, "sort/4 yields one answer");
          if (r->size() < 1) return;
          t.expect(bound((*r)[0], "L") == "[3, 2, 1]", "sort/4 descending yields [3, 2, 1]");
      })
.test("atom_length_codes_chars_and_char_code_inspect_atom_content",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("atom_length(hello, N)."), 0);
          t.expect(r1.has_value(), "atom_length succeeds");
          if (!r1.has_value()) return;
          t.expect(r1->size() == 1, "atom_length yields one answer");
          if (r1->size() < 1) return;
          t.expect(bound((*r1)[0], "N") == "5", "atom_length of hello is 5");

          auto r2 = solve(prog(""), query("atom_codes(ab, L)."), 0);
          t.expect(r2.has_value(), "atom_codes succeeds");
          if (!r2.has_value()) return;
          t.expect(r2->size() == 1, "atom_codes yields one answer");
          if (r2->size() < 1) return;
          t.expect(bound((*r2)[0], "L") == "[97, 98]", "atom_codes of ab is [97, 98]");

          auto r3 = solve(prog(""), query("atom_chars(ab, L)."), 0);
          t.expect(r3.has_value(), "atom_chars succeeds");
          if (!r3.has_value()) return;
          t.expect(r3->size() == 1, "atom_chars yields one answer");
          if (r3->size() < 1) return;
          t.expect(bound((*r3)[0], "L") == "[a, b]", "atom_chars of ab is [a, b]");

          auto r4 = solve(prog(""), query("char_code(a, X)."), 0);
          t.expect(r4.has_value(), "char_code succeeds");
          if (!r4.has_value()) return;
          t.expect(r4->size() == 1, "char_code yields one answer");
          if (r4->size() < 1) return;
          t.expect(bound((*r4)[0], "X") == "97", "char_code of a is 97");
      })
.test("atom_number_converts_numeric_atom_and_fails_on_non_numeric_atom",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("atom_number('42', N)."), 0);
          t.expect(r1.has_value(), "atom_number on numeric atom succeeds");
          if (!r1.has_value()) return;
          t.expect(r1->size() == 1, "atom_number yields one answer");
          if (r1->size() < 1) return;
          t.expect(bound((*r1)[0], "N") == "42", "atom_number converts '42' to 42");

          auto r2 = solve(prog(""), query("atom_number(foo, N)."), 0);
          t.expect(r2.has_value(), "atom_number on non-numeric atom succeeds without error");
          if (r2.has_value()) {
              t.expect(r2->empty(), "atom_number on non-numeric atom fails with zero answers");
          }
      })
.test("upcase_atom_converts_ascii_characters_to_uppercase",
      [](TestContext& t) {
          auto r = solve(prog(""), query("upcase_atom(aBc, X)."), 0);
          t.expect(r.has_value(), "upcase_atom succeeds");
          if (!r.has_value()) return;
          t.expect(r->size() == 1, "upcase_atom yields one answer");
          if (r->size() < 1) return;
          t.expect(bound((*r)[0], "X") == "ABC", "upcase_atom(aBc, X) yields ABC");
      })
.test("atom_concat_joins_atoms_and_splits_an_atom_into_all_decompositions",
      [](TestContext& t) {
          auto r1 = solve(prog(""), query("atom_concat(ab, cd, X)."), 0);
          t.expect(r1.has_value(), "atom_concat join succeeds");
          if (!r1.has_value()) return;
          t.expect(r1->size() == 1, "atom_concat join yields one answer");
          if (r1->size() < 1) return;
          t.expect(bound((*r1)[0], "X") == "abcd", "ab + cd yields abcd");

          auto r2 = solve(prog(""), query("atom_concat(X, Y, abc)."), 0);
          t.expect(r2.has_value(), "atom_concat split succeeds");
          if (!r2.has_value()) return;
          t.expect(r2->size() == 4, "atom_concat on abc yields exactly 4 decompositions");
          if (r2->size() < 4) return;
          t.expect(apply_substitution((*r2)[0], make_var("X")) == make_atom(""), "ans 0: X is empty atom");
          t.expect(apply_substitution((*r2)[0], make_var("Y")) == make_atom("abc"), "ans 0: Y is abc");

          t.expect(apply_substitution((*r2)[1], make_var("X")) == make_atom("a"), "ans 1: X is a");
          t.expect(apply_substitution((*r2)[1], make_var("Y")) == make_atom("bc"), "ans 1: Y is bc");

          t.expect(apply_substitution((*r2)[2], make_var("X")) == make_atom("ab"), "ans 2: X is ab");
          t.expect(apply_substitution((*r2)[2], make_var("Y")) == make_atom("c"), "ans 2: Y is c");

          t.expect(apply_substitution((*r2)[3], make_var("X")) == make_atom("abc"), "ans 3: X is abc");
          t.expect(apply_substitution((*r2)[3], make_var("Y")) == make_atom(""), "ans 3: Y is empty atom");
      })
.test("sub_atom_extracts_substring_by_offset_and_length",
      [](TestContext& t) {
          auto r = solve(prog(""), query("sub_atom(abc, 1, 1, _, S)."), 0);
          t.expect(r.has_value(), "sub_atom succeeds");
          if (!r.has_value()) return;
          t.expect(r->size() == 1, "sub_atom yields one answer");
          if (r->size() < 1) return;
          t.expect(bound((*r)[0], "S") == "b", "extracted substring is b");
          t.expect(apply_substitution((*r)[0], make_var("S")) == make_atom("b"), "S is atom 'b'");
      })
.test("findall_collects_all_solutions_in_clause_order", [](TestContext& t) {
    const Program p = prog("p(1). p(2). p(3).");
    const auto r = solve(p, query("findall(X, p(X), L)."), 0);
    t.expect(r.has_value(), "findall query succeeds");
    if (!r.has_value()) return;
    t.expect(r->size() == 1, "findall yields exactly one solution");
    if (r->size() < 1) return;
    t.expect(bound((*r)[0], "L") == "[1, 2, 3]", "collected list matches elements in clause order");
})
.test("findall_succeeds_with_empty_list_when_goal_fails", [](TestContext& t) {
    const Program p = prog("p(1).");
    const auto r_find = solve(p, query("findall(X, nosuch(X), L)."), 0);
    t.expect(r_find.has_value(), "findall query succeeds");
    if (r_find.has_value()) {
        t.expect(r_find->size() == 1, "findall succeeds with exactly one answer");
        if (r_find->size() >= 1) {
            t.expect(bound((*r_find)[0], "L") == "[]", "findall yields the empty list");
        }
    }
    const auto r_bag = solve(p, query("bagof(X, nosuch(X), L)."), 0);
    t.expect(r_bag.has_value(), "bagof query does not error");
    if (r_bag.has_value()) {
        t.expect(r_bag->empty(), "bagof fails on no solutions");
    }
    const auto r_set = solve(p, query("setof(X, nosuch(X), L)."), 0);
    t.expect(r_set.has_value(), "setof query does not error");
    if (r_set.has_value()) {
        t.expect(r_set->empty(), "setof fails on no solutions");
    }
})
.test("findall_supports_compound_template", [](TestContext& t) {
    const Program p = prog("q(1, a). q(2, b).");
    const auto r = solve(p, query("findall(f(X, Y), q(X, Y), L)."), 0);
    t.expect(r.has_value(), "findall with compound template succeeds");
    if (!r.has_value()) return;
    t.expect(r->size() == 1, "findall returns exactly one answer");
    if (r->size() < 1) return;
    t.expect(bound((*r)[0], "L") == "[f(1, a), f(2, b)]", "elements in list instantiate compound template");
})
.test("findall_does_not_bind_goal_variables_in_outer_answer", [](TestContext& t) {
    const Program p = prog("p(1). p(2).");
    const auto r = solve(p, query("findall(X, p(X), L)."), 0);
    t.expect(r.has_value(), "findall succeeds");
    if (!r.has_value()) return;
    t.expect(r->size() == 1, "exactly one answer");
    if (r->size() < 1) return;
    t.expect(bound((*r)[0], "X") == "X", "query variable X remains unbound in outer substitution");
    t.expect(bound((*r)[0], "L") == "[1, 2]", "list L contains the collected solutions");
})
.test("findall_is_opaque_to_cut_inside_goal", [](TestContext& t) {
    const Program p = prog(
        "choice(1).\n"
        "choice(2).\n"
        "test_cut(one, L) :- findall(Y, (choice(Y), !), L).\n"
        "test_cut(two, []).\n"
    );
    const auto r = solve(p, query("test_cut(Name, Res)."), 0);
    t.expect(r.has_value(), "test_cut query succeeds");
    if (!r.has_value()) return;
    t.expect(r->size() == 2, "cut inside findall does not prune second clause of test_cut");
    if (r->size() < 2) return;
    t.expect(bound((*r)[0], "Name") == "one", "first clause matches Name=one");
    t.expect(bound((*r)[0], "Res") == "[1]", "cut inside findall limited choices to [1]");
    t.expect(bound((*r)[1], "Name") == "two", "second clause matches Name=two");
    t.expect(bound((*r)[1], "Res") == "[]", "second clause returns empty list");
})
.test("findall_4_appends_solutions_to_given_tail", [](TestContext& t) {
    const Program p = prog("p(1). p(2). p(3).");
    const auto r = solve(p, query("findall(X, p(X), L, [4, 5])."), 0);
    t.expect(r.has_value(), "findall/4 succeeds");
    if (!r.has_value()) return;
    t.expect(r->size() == 1, "exactly one answer");
    if (r->size() < 1) return;
    t.expect(bound((*r)[0], "L") == "[1, 2, 3, 4, 5]", "findall/4 appends to non-empty tail");
})
.test("bagof_preserves_order_and_duplicates", [](TestContext& t) {
    const Program p = prog("p(1). p(2). p(3).");
    const auto r = solve(p, query("bagof(X, p(X), L)."), 0);
    t.expect(r.has_value(), "bagof query succeeds");
    if (!r.has_value()) return;
    t.expect(r->size() == 1, "bagof yields one answer when no free variables exist");
    if (r->size() < 1) return;
    t.expect(bound((*r)[0], "L") == "[1, 2, 3]", "bagof preserves element order [1, 2, 3]");
})
.test("bagof_and_setof_fail_when_goal_has_no_solutions", [](TestContext& t) {
    const Program p = prog("p(1).");
    const auto r_bag = solve(p, query("bagof(X, nosuch(X), L)."), 0);
    t.expect(r_bag.has_value(), "bagof on failure returns a valid Result without error");
    if (r_bag.has_value()) {
        t.expect(r_bag->empty(), "bagof produces an empty answer vector on failure");
    }
    const auto r_set = solve(p, query("setof(X, nosuch(X), L)."), 0);
    t.expect(r_set.has_value(), "setof on failure returns a valid Result without error");
    if (r_set.has_value()) {
        t.expect(r_set->empty(), "setof produces an empty answer vector on failure");
    }
})
.test("setof_sorts_by_standard_order_and_removes_duplicates", [](TestContext& t) {
    const Program p = prog("p(3). p(1). p(2). p(1). p(3).");
    const auto r = solve(p, query("setof(X, p(X), L)."), 0);
    t.expect(r.has_value(), "setof query succeeds");
    if (!r.has_value()) return;
    t.expect(r->size() == 1, "setof yields one answer");
    if (r->size() < 1) return;
    t.expect(bound((*r)[0], "L") == "[1, 2, 3]", "setof removes duplicates and sorts ascending");
})
.test("setof_sorts_mixed_types_according_to_standard_order", [](TestContext& t) {
    const Program p = prog("p(beta). p(2). p(alpha). p(1). p(2).");
    const auto r = solve(p, query("setof(X, p(X), L)."), 0);
    t.expect(r.has_value(), "setof query succeeds");
    if (!r.has_value()) return;
    t.expect(r->size() == 1, "setof yields one answer");
    if (r->size() < 1) return;
    t.expect(bound((*r)[0], "L") == "[1, 2, alpha, beta]", "integers precede atoms in standard order");
})
.test("bagof_groups_by_free_variables_in_standard_order_on_backtracking", [](TestContext& t) {
    const Program p = prog(
        "age(peter, 7).\n"
        "age(ann, 11).\n"
        "age(pat, 8).\n"
        "age(tom, 5).\n"
    );
    const auto r = solve(p, query("bagof(C, age(C, A), L)."), 0);
    t.expect(r.has_value(), "bagof with free variable succeeds");
    if (!r.has_value()) return;
    t.expect(r->size() == 4, "bagof yields 4 distinct answers for the 4 ages");
    if (r->size() < 4) return;
    t.expect(bound((*r)[0], "A") == "5", "first answer has age A = 5");
    t.expect(bound((*r)[0], "L") == "[tom]", "first answer holds child tom");
    t.expect(bound((*r)[1], "A") == "7", "second answer has age A = 7");
    t.expect(bound((*r)[1], "L") == "[peter]", "second answer holds child peter");
    t.expect(bound((*r)[2], "A") == "8", "third answer has age A = 8");
    t.expect(bound((*r)[2], "L") == "[pat]", "third answer holds child pat");
    t.expect(bound((*r)[3], "A") == "11", "fourth answer has age A = 11");
    t.expect(bound((*r)[3], "L") == "[ann]", "fourth answer holds child ann");
})
.test("bagof_with_existential_marker_gives_single_lumped_answer", [](TestContext& t) {
    const Program p = prog(
        "age(peter, 7).\n"
        "age(ann, 11).\n"
        "age(pat, 8).\n"
        "age(tom, 5).\n"
    );
    const auto r = solve(p, query("bagof(C, A^age(C, A), L)."), 0);
    t.expect(r.has_value(), "bagof with existential marker succeeds");
    if (!r.has_value()) return;
    t.expect(r->size() == 1, "existential quantification yields exactly one answer");
    if (r->size() < 1) return;
    t.expect(bound((*r)[0], "L") == "[peter, ann, pat, tom]", "list contains every child in clause order");
    t.expect(bound((*r)[0], "A") == "A", "existentially quantified variable A remains unbound");
})
.test("forall_succeeds_when_condition_holds_for_all_generator_solutions", [](TestContext& t) {
    const Program p = prog(
        "num(1). num(2). num(3).\n"
        "pos(1). pos(2). pos(3).\n"
    );
    const auto r = solve(p, query("forall(num(X), pos(X))."), 0);
    t.expect(r.has_value(), "forall query succeeds");
    if (!r.has_value()) return;
    t.expect(r->size() == 1, "forall returns one solution when all generator solutions satisfy condition");
})
.test("forall_fails_when_condition_fails_for_any_generator_solution", [](TestContext& t) {
    const Program p = prog(
        "num(1). num(2). num(3).\n"
        "even(2).\n"
    );
    const auto r = solve(p, query("forall(num(X), even(X))."), 0);
    t.expect(r.has_value(), "forall returns result without engine error");
    if (!r.has_value()) return;
    t.expect(r->empty(), "forall fails because 1 and 3 are not even");
})
.test("forall_succeeds_vacuously_when_generator_has_no_solutions", [](TestContext& t) {
    const Program p = prog("p(1).");
    const auto r = solve(p, query("forall(nosuch(X), even(X))."), 0);
    t.expect(r.has_value(), "forall query succeeds");
    if (!r.has_value()) return;
    t.expect(r->size() == 1, "forall succeeds vacuously on empty generator");
})
.test("assertz_clause_is_immediately_visible_within_same_query", [](TestContext& t) {
    const Program p = prog("");
    const auto r = solve(p, query("assertz(p(1)), p(X)."), 0);
    t.expect(r.has_value(), "assertz and query succeed in single query");
    if (!r.has_value()) return;
    t.expect(r->size() == 1, "asserted clause was found");
    if (r->size() < 1) return;
    t.expect(bound((*r)[0], "X") == "1", "X is bound to 1 from asserted clause");
})
.test("asserta_puts_clause_first_and_assertz_puts_clause_last", [](TestContext& t) {
    const Program p = prog("p(2).");
    const auto r = solve(p, query("asserta(p(1)), assertz(p(3)), p(X)."), 0);
    t.expect(r.has_value(), "query with asserta and assertz succeeds");
    if (!r.has_value()) return;
    t.expect(r->size() == 3, "three clauses found in database");
    if (r->size() < 3) return;
    t.expect(bound((*r)[0], "X") == "1", "first answer is 1 from asserta");
    t.expect(bound((*r)[1], "X") == "2", "second answer is 2 from existing program");
    t.expect(bound((*r)[2], "X") == "3", "third answer is 3 from assertz");
})
.test("free_solve_uses_isolated_database_copy_so_asserts_do_not_persist", [](TestContext& t) {
    const Program p = prog("base(1).");
    const auto r1 = solve(p, query("assertz(base(2)), base(X)."), 0);
    t.expect(r1.has_value(), "first solve succeeds");
    if (r1.has_value()) {
        t.expect(r1->size() == 2, "first query sees base(1) and asserted base(2)");
    }
    const auto r2 = solve(p, query("base(X)."), 0);
    t.expect(r2.has_value(), "second solve succeeds");
    if (!r2.has_value()) return;
    t.expect(r2->size() == 1, "second query against same Program does not see asserted clause");
    if (r2->size() < 1) return;
    t.expect(bound((*r2)[0], "X") == "1", "only original clause base(1) remains");
})
.test("machine_persists_asserted_clauses_across_separate_queries", [](TestContext& t) {
    Machine m = Machine::from(prog("p(1)."));
    t.expect(m.database().size() == 1, "machine database initially has 1 clause");
    const auto r1 = m.solve(query("assertz(p(2))."), 0);
    t.expect(r1.has_value(), "assert query succeeds in Machine");
    t.expect(m.database().size() == 2, "machine database size increased to 2 after assertz");
    const auto r2 = m.solve(query("p(X)."), 0);
    t.expect(r2.has_value(), "second query succeeds in Machine");
    if (!r2.has_value()) return;
    t.expect(r2->size() == 2, "second query in Machine sees both persisted clauses");
    if (r2->size() < 2) return;
    t.expect(bound((*r2)[0], "X") == "1", "first answer is 1");
    t.expect(bound((*r2)[1], "X") == "2", "second answer is 2");
})
.test("retract_is_nondeterministic_and_removes_one_clause_per_solution", [](TestContext& t) {
    // ISO `retract/1` is RE-SATISFIABLE: backtracking into it retracts the next matching
    // clause, and the removals are not undone. Asking for every solution therefore removes
    // every match, which is what makes a `retract` loop terminate.
    Machine m = Machine::from(prog("p(1). p(2). p(1)."));
    t.expect(m.database().size() == 3, "initial size is 3 clauses");
    const auto r1 = m.solve(query("retract(p(1))."), 0);
    t.expect(r1.has_value(), "retract query succeeds");
    if (r1.has_value()) {
        t.expect(r1->size() == 2, "one solution per matching clause, so two");
    }
    t.expect(m.database().size() == 1, "both matching clauses were removed");
    const auto r2 = m.solve(query("p(X)."), 0);
    t.expect(r2.has_value(), "query after retract succeeds");
    if (!r2.has_value()) return;
    t.expect(r2->size() == 1, "only p(2) remains");
    if (r2->size() >= 1) {
        t.expect(bound((*r2)[0], "X") == "2", "the surviving clause is p(2)");
    }
})
.test("retract_stopped_at_the_first_solution_removes_exactly_one_clause", [](TestContext& t) {
    // Capping the query at one solution is the semi-deterministic use of retract, and it must
    // remove exactly one clause — the FIRST match in clause order — leaving the rest alone.
    Machine m = Machine::from(prog("p(1). p(2). p(1)."));
    const auto r1 = m.solve(query("retract(p(1))."), 1);
    t.expect(r1.has_value() && r1->size() == 1, "exactly one solution is taken");
    t.expect(m.database().size() == 2, "exactly one clause was removed");
    const auto r2 = m.solve(query("p(X)."), 0);
    t.expect(r2.has_value(), "query after retract succeeds");
    if (!r2.has_value() || r2->size() < 2) return;
    t.expect(r2->size() == 2, "two clauses remain for p(X)");
    t.expect(bound((*r2)[0], "X") == "2", "the first remaining clause is p(2)");
    t.expect(bound((*r2)[1], "X") == "1", "the second remaining clause is the later p(1)");
})
.test("retract_of_non_matching_clause_fails_without_error", [](TestContext& t) {
    const Program p = prog("p(1). p(2).");
    const auto r = solve(p, query("retract(p(3))."), 0);
    t.expect(r.has_value(), "retract of non-matching clause does not error");
    if (r.has_value()) {
        t.expect(r->empty(), "retract of non-matching clause fails with empty answers");
    }
})
.test("retractall_removes_every_matching_clause_leaving_others_intact", [](TestContext& t) {
    Machine m = Machine::from(prog("p(1). p(2). q(3)."));
    const auto r1 = m.solve(query("retractall(p(_))."), 0);
    t.expect(r1.has_value(), "retractall succeeds");
    if (r1.has_value()) {
        t.expect(r1->size() == 1, "retractall returns one successful derivation");
    }
    const auto r_p = m.solve(query("p(X)."), 0);
    t.expect(r_p.has_value(), "p(X) query succeeds");
    if (r_p.has_value()) {
        t.expect(r_p->empty(), "all clauses matching p(_) have been removed");
    }
    const auto r_q = m.solve(query("q(X)."), 0);
    t.expect(r_q.has_value(), "q(X) query succeeds");
    if (!r_q.has_value()) return;
    t.expect(r_q->size() == 1, "q(3) was untouched by retractall");
    if (r_q->size() < 1) return;
    t.expect(bound((*r_q)[0], "X") == "3", "q(X) yields X = 3");
})
.test("retractall_succeeds_even_when_nothing_matched", [](TestContext& t) {
    Machine m = Machine::from(prog("q(1)."));
    const auto r = m.solve(query("retractall(p(_))."), 0);
    t.expect(r.has_value(), "retractall succeeds without error when no clauses match");
    if (r.has_value()) {
        t.expect(r->size() == 1, "retractall succeeds with one answer even when zero clauses matched");
    }
    t.expect(m.database().size() == 1, "database retains original clause");
})
.test("abolish_removes_specified_predicate_arities_leaving_other_arities_alone", [](TestContext& t) {
    Machine m = Machine::from(prog("p(1). p(2). p(a, b). p(c, d)."));
    t.expect(m.database().is_defined("p/1"), "p/1 initially defined");
    t.expect(m.database().is_defined("p/2"), "p/2 initially defined");
    const auto r1 = m.solve(query("abolish(p/2)."), 0);
    t.expect(r1.has_value(), "abolish(p/2) succeeds");
    t.expect(!m.database().is_defined("p/2"), "p/2 is no longer defined");
    t.expect(m.database().is_defined("p/1"), "p/1 remains defined");
    const auto r_p2 = m.solve(query("p(X, Y)."), 0);
    t.expect(r_p2.has_value(), "querying abolished p/2 does not error");
    if (r_p2.has_value()) {
        t.expect(r_p2->empty(), "abolished p/2 returns no solutions");
    }
    const auto r_p1 = m.solve(query("p(X)."), 0);
    t.expect(r_p1.has_value(), "querying p/1 succeeds");
    if (!r_p1.has_value()) return;
    t.expect(r_p1->size() == 2, "both p/1 clauses remain");
    if (r_p1->size() < 2) return;
    t.expect(bound((*r_p1)[0], "X") == "1", "first p/1 solution is 1");
    t.expect(bound((*r_p1)[1], "X") == "2", "second p/1 solution is 2");
})
.test("assertz_of_clause_with_builtin_head_fails_with_domain_error", [](TestContext& t) {
    const Program p = prog("");
    const auto r = solve(p, query("assertz(length(a, b))."), 0);
    t.expect(!r.has_value(), "asserting a clause for builtin length/2 must fail");
    if (!r.has_value()) {
        t.expect(r.error() == MathError::domain_error, "error is domain_error");
    }
    Database db = Database::from(Program{});
    const auto res = db.assertz(Clause{make_compound("is", {make_var("X"), make_int(1)}), {}});
    t.expect(!res.has_value(), "Database::assertz on builtin is/2 fails");
    if (!res.has_value()) {
        t.expect(res.error() == MathError::domain_error, "Database::assertz returns domain_error");
    }
})
.test("clause_enumerates_matching_clauses_and_represents_fact_body_as_true", [](TestContext& t) {
    const Program p = prog(
        "p(1).\n"
        "p(2) :- q(2).\n"
    );
    const auto r = solve(p, query("clause(p(X), Body)."), 0);
    t.expect(r.has_value(), "clause/2 query succeeds");
    if (!r.has_value()) return;
    t.expect(r->size() == 2, "enumerates both clauses of p");
    if (r->size() < 2) return;
    t.expect(bound((*r)[0], "X") == "1", "first clause matches X = 1");
    t.expect(bound((*r)[0], "Body") == "true", "fact body is represented as atom 'true'");
    t.expect(bound((*r)[1], "X") == "2", "second clause matches X = 2");
    t.expect(bound((*r)[1], "Body") == "q(2)", "rule body is represented as q(2)");
})
.test("first_arg_indexing_with_bound_first_arg_returns_matching_clauses_in_program_order", [](TestContext& t) {
    const Program p = prog(
        "item(tag_a, 10).\n"
        "item(tag_b, 20).\n"
        "item(tag_a, 30).\n"
        "item(tag_c, 40).\n"
        "item(tag_a, 50).\n"
        "item(tag_b, 60).\n"
    );
    const auto r = solve(p, query("item(tag_a, Val)."), 0);
    t.expect(r.has_value(), "call with bound first arg succeeds");
    if (!r.has_value()) return;
    t.expect(r->size() == 3, "returns exactly the 3 matching clauses");
    if (r->size() < 3) return;
    t.expect(bound((*r)[0], "Val") == "10", "first match in program order is 10");
    t.expect(bound((*r)[1], "Val") == "30", "second match in program order is 30");
    t.expect(bound((*r)[2], "Val") == "50", "third match in program order is 50");
})
.test("first_arg_indexing_with_unbound_first_arg_returns_all_clauses_in_program_order", [](TestContext& t) {
    const Program p = prog(
        "item(tag_a, 10).\n"
        "item(tag_b, 20).\n"
        "item(tag_a, 30).\n"
        "item(tag_c, 40).\n"
        "item(tag_a, 50).\n"
        "item(tag_b, 60).\n"
    );
    const auto r = solve(p, query("item(Tag, Val)."), 0);
    t.expect(r.has_value(), "call with unbound first arg succeeds");
    if (!r.has_value()) return;
    t.expect(r->size() == 6, "returns all 6 clauses in program order");
    if (r->size() < 6) return;
    t.expect(bound((*r)[0], "Tag") == "tag_a" && bound((*r)[0], "Val") == "10", "clause 0: tag_a, 10");
    t.expect(bound((*r)[1], "Tag") == "tag_b" && bound((*r)[1], "Val") == "20", "clause 1: tag_b, 20");
    t.expect(bound((*r)[2], "Tag") == "tag_a" && bound((*r)[2], "Val") == "30", "clause 2: tag_a, 30");
    t.expect(bound((*r)[3], "Tag") == "tag_c" && bound((*r)[3], "Val") == "40", "clause 3: tag_c, 40");
    t.expect(bound((*r)[4], "Tag") == "tag_a" && bound((*r)[4], "Val") == "50", "clause 4: tag_a, 50");
    t.expect(bound((*r)[5], "Tag") == "tag_b" && bound((*r)[5], "Val") == "60", "clause 5: tag_b, 60");
})
.test("first_arg_indexing_includes_variable_headed_clause_in_program_order_for_every_call", [](TestContext& t) {
    const Program p = prog(
        "mix(alpha, 1).\n"
        "mix(g(1), 2).\n"
        "mix(V, 3).\n"
        "mix(beta, 4).\n"
    );
    const auto r1 = solve(p, query("mix(alpha, Y)."), 0);
    t.expect(r1.has_value(), "call for alpha succeeds");
    if (r1.has_value()) {
        t.expect(r1->size() == 2, "alpha matches specific clause and variable clause");
        if (r1->size() >= 2) {
            t.expect(bound((*r1)[0], "Y") == "1", "specific clause 1 is first");
            t.expect(bound((*r1)[1], "Y") == "3", "variable clause 3 is second in program order");
        }
    }
    const auto r2 = solve(p, query("mix(g(1), Y)."), 0);
    t.expect(r2.has_value(), "call for compound g(1) succeeds");
    if (r2.has_value()) {
        t.expect(r2->size() == 2, "g(1) matches compound clause and variable clause");
        if (r2->size() >= 2) {
            t.expect(bound((*r2)[0], "Y") == "2", "compound clause 2 is first");
            t.expect(bound((*r2)[1], "Y") == "3", "variable clause 3 is second in program order");
        }
    }
    const auto r3 = solve(p, query("mix(beta, Y)."), 0);
    t.expect(r3.has_value(), "call for beta succeeds");
    if (r3.has_value()) {
        t.expect(r3->size() == 2, "beta matches variable clause and specific clause");
        if (r3->size() >= 2) {
            t.expect(bound((*r3)[0], "Y") == "3", "variable clause 3 precedes beta in program position");
            t.expect(bound((*r3)[1], "Y") == "4", "specific clause 4 follows variable clause");
        }
    }
    const auto r4 = solve(p, query("mix(gamma, Y)."), 0);
    t.expect(r4.has_value(), "call for unseen gamma succeeds");
    if (r4.has_value()) {
        t.expect(r4->size() == 1, "gamma matches only the variable clause");
        if (r4->size() >= 1) {
            t.expect(bound((*r4)[0], "Y") == "3", "answer is 3 from variable clause");
        }
    }
})
.test("first_arg_indexing_distinguishes_integer_and_atom_with_same_text", [](TestContext& t) {
    const Program p = prog(
        "disc(1, int_one).\n"
        "disc('1', atom_one).\n"
    );
    const auto r_int = solve(p, query("disc(1, T)."), 0);
    t.expect(r_int.has_value(), "query with integer 1 succeeds");
    if (r_int.has_value()) {
        t.expect(r_int->size() == 1, "integer 1 unifies with exactly 1 clause");
        if (r_int->size() >= 1) {
            t.expect(bound((*r_int)[0], "T") == "int_one", "integer matches int_one fact");
        }
    }
    const auto r_atom = solve(p, query("disc('1', T)."), 0);
    t.expect(r_atom.has_value(), "query with atom '1' succeeds");
    if (r_atom.has_value()) {
        t.expect(r_atom->size() == 1, "atom '1' unifies with exactly 1 clause");
        if (r_atom->size() >= 1) {
            t.expect(bound((*r_atom)[0], "T") == "atom_one", "atom matches atom_one fact");
        }
    }
})
.test("database_is_defined_reports_indicator_presence", [](TestContext& t) {
    const Database db = Database::from(prog("p(1, 2). p(3, 4). q(1)."));
    t.expect(db.is_defined("p/2"), "p/2 is defined");
    t.expect(db.is_defined("q/1"), "q/1 is defined");
    t.expect(!db.is_defined("p/1"), "p/1 is not defined");
    t.expect(!db.is_defined("p/3"), "p/3 is not defined");
    t.expect(!db.is_defined("missing/0"), "missing/0 is not defined");
})
.test("database_with_clause_is_fluent_and_leaves_receiver_unmodified", [](TestContext& t) {
    const Database orig = Database::from(prog("p(1). p(2)."));
    t.expect(orig.size() == 2, "original database size is 2");
    const Clause c{make_compound("p", {make_int(3)}), {}};
    const Database next = orig.with_clause(c);
    t.expect(orig.size() == 2, "receiver database size remains 2 after with_clause");
    t.expect(next.size() == 3, "new database size is 3");
    t.expect(next.clauses().size() == 3, "new database contains 3 clauses");
})
.test("undefined_predicate_fails_with_empty_answers_without_error", [](TestContext& t) {
    const Program p = prog("p(1).");
    const auto r = solve(p, query("never_defined(X, Y)."), 0);
    t.expect(r.has_value(), "calling an undefined predicate returns Result without error");
    if (!r.has_value()) return;
    t.expect(r->empty(), "undefined predicate call produces an empty answer vector");
})
.test("findall_over_divergent_goal_terminates_with_not_converged", [](TestContext& t) {
    const Program p = prog("loop :- loop.");
    const auto r = solve(p, query("findall(X, loop, L)."), 0);
    t.expect(!r.has_value(), "divergent goal inside findall fails rather than returning partial list");
    if (!r.has_value()) {
        t.expect(r.error() == MathError::not_converged, "failure error is MathError::not_converged");
    }
})
.test("solve_or_parallel_equals_serial_solve_on_program_that_asserts", [](TestContext& t) {
    const Program p = prog(
        "add_fact(X) :- assertz(fact(X)).\n"
        "fact(10).\n"
    );
    const auto goals = query("add_fact(20), fact(V).");
    const auto serial_res = solve(p, goals, 0);
    const auto par_res = solve_or_parallel(p, goals, 0);
    t.expect(serial_res.has_value(), "serial solve succeeds on program that asserts");
    t.expect(par_res.has_value(), "solve_or_parallel succeeds on program that asserts");
    if (!serial_res.has_value() || !par_res.has_value()) return;
    t.expect(*serial_res == *par_res, "solve_or_parallel produces identical solutions to serial solve");
    t.expect(serial_res->size() == 2, "exactly two solutions found: 10 and 20");
    if (serial_res->size() < 2) return;
    t.expect(bound((*serial_res)[0], "V") == "10", "first serial answer is 10");
    t.expect(bound((*serial_res)[1], "V") == "20", "second serial answer is 20");
    t.expect(bound((*par_res)[0], "V") == "10", "first parallel answer is 10");
    t.expect(bound((*par_res)[1], "V") == "20", "second parallel answer is 20");
})
        .run();
}
