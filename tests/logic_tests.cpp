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
using nimblecas::make_not;
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
        .run();
}
