// Tests for nimblecas.planning: SAS+ tasks, delete-relaxation heuristics, and heuristic search.
// @author Olumuyiwa Oluwasanmi
//
// The heuristics are tested against their DEFINING PROPERTIES rather than against remembered
// numbers: h_max must never exceed the true optimal cost (that is what admissible means, and it is
// what makes A* optimal), h_add may exceed it, and h_ff must count a shared action once where
// h_add counts it repeatedly. Those are checkable claims on small tasks whose optimal cost A*
// itself supplies.
//
// The searches are tested on hand-built tasks whose optimal plans are known by construction, plus
// a logistics domain large enough that the difference between the heuristics actually shows.

import std;
import nimblecas.core;
import nimblecas.planning;
import nimblecas.testing;

using nimblecas::MathError;
using nimblecas::planning::applicable;
using nimblecas::planning::apply;
using nimblecas::planning::astar_plan;
using nimblecas::planning::Condition;
using nimblecas::planning::decode_state;
using nimblecas::planning::encode_state;
using nimblecas::planning::FactPair;
using nimblecas::planning::gbfs_plan;
using nimblecas::planning::gbfs_plan_hadd;
using nimblecas::planning::h_add;
using nimblecas::planning::h_ff;
using nimblecas::planning::h_max;
using nimblecas::planning::hash_state;
using nimblecas::planning::holds;
using nimblecas::planning::Operator;
using nimblecas::planning::Plan;
using nimblecas::planning::State;
using nimblecas::planning::Task;
using nimblecas::planning::validate;
using nimblecas::planning::validate_plan;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// A corridor of `n` rooms. One variable holds the position; moving costs one. The optimal plan
// from room 0 to room n-1 is exactly n-1 steps, which makes it the simplest task whose answer is
// known without solving it.
[[nodiscard]] auto corridor(std::int32_t n) -> Task {
    Task t;
    t.domain_size = {n};
    t.initial = {0};
    t.goal = {FactPair{.var = 0, .value = n - 1}};
    for (std::int32_t i = 0; i + 1 < n; ++i) {
        t.operators.push_back(Operator{.name = std::format("fwd{}", i),
                                       .preconditions = {FactPair{.var = 0, .value = i}},
                                       .effects = {FactPair{.var = 0, .value = i + 1}},
                                       .cost = 1});
        t.operators.push_back(Operator{.name = std::format("back{}", i),
                                       .preconditions = {FactPair{.var = 0, .value = i + 1}},
                                       .effects = {FactPair{.var = 0, .value = i}},
                                       .cost = 1});
    }
    return t;
}

// A gripper-like task: `k` independent switches, each flipped by its own operator. The optimal
// plan is exactly k steps in any order -- and it is the task where h_add's double counting is
// harmless and h_ff agrees with it, since nothing is shared.
[[nodiscard]] auto switches(std::int32_t k) -> Task {
    Task t;
    t.domain_size.assign(static_cast<std::size_t>(k), 2);
    t.initial.assign(static_cast<std::size_t>(k), 0);
    for (std::int32_t i = 0; i < k; ++i) {
        t.goal.push_back(FactPair{.var = static_cast<std::size_t>(i), .value = 1});
        t.operators.push_back(
            Operator{.name = std::format("flip{}", i),
                     .preconditions = {FactPair{.var = static_cast<std::size_t>(i), .value = 0}},
                     .effects = {FactPair{.var = static_cast<std::size_t>(i), .value = 1}},
                     .cost = 1});
    }
    return t;
}

// A task with a SHARED PREFIX: one setup action enables `k` goals. The optimal plan is 1 + k.
// h_add counts the setup once per goal and so returns about 2k; h_ff extracts a plan and counts
// it once, so it returns exactly 1 + k. This is the task that separates the two.
[[nodiscard]] auto shared_prefix(std::int32_t k) -> Task {
    Task t;
    // var 0: the setup flag. vars 1..k: the goals.
    t.domain_size.push_back(2);
    t.domain_size.insert(t.domain_size.end(), static_cast<std::size_t>(k), 2);
    t.initial.assign(static_cast<std::size_t>(k) + 1, 0);
    t.operators.push_back(Operator{.name = "setup",
                                   .preconditions = {FactPair{.var = 0, .value = 0}},
                                   .effects = {FactPair{.var = 0, .value = 1}},
                                   .cost = 1});
    for (std::int32_t i = 1; i <= k; ++i) {
        const auto v = static_cast<std::size_t>(i);
        t.goal.push_back(FactPair{.var = v, .value = 1});
        t.operators.push_back(
            Operator{.name = std::format("do{}", i),
                     .preconditions = {FactPair{.var = 0, .value = 1}, FactPair{.var = v, .value = 0}},
                     .effects = {FactPair{.var = v, .value = 1}},
                     .cost = 1});
    }
    return t;
}

// A small logistics task: a truck moving between `loc` locations carrying `pkg` packages.
// var 0 is the truck's location; var i is package i's location, with value `loc` meaning
// "in the truck".
[[nodiscard]] auto logistics(std::int32_t loc, std::int32_t pkg) -> Task {
    Task t;
    t.domain_size.push_back(loc);
    t.domain_size.insert(t.domain_size.end(), static_cast<std::size_t>(pkg), loc + 1);
    t.initial.push_back(0);
    for (std::int32_t p = 0; p < pkg; ++p) {
        t.initial.push_back(0);  // every package starts at location 0
        t.goal.push_back(FactPair{.var = static_cast<std::size_t>(p) + 1, .value = loc - 1});
    }
    for (std::int32_t a = 0; a < loc; ++a) {
        for (std::int32_t b = 0; b < loc; ++b) {
            if (a == b) {
                continue;
            }
            t.operators.push_back(Operator{.name = std::format("drive{}_{}", a, b),
                                           .preconditions = {FactPair{.var = 0, .value = a}},
                                           .effects = {FactPair{.var = 0, .value = b}},
                                           .cost = 1});
        }
    }
    for (std::int32_t p = 0; p < pkg; ++p) {
        const auto v = static_cast<std::size_t>(p) + 1;
        for (std::int32_t a = 0; a < loc; ++a) {
            Condition load_pre{FactPair{.var = 0, .value = a}, FactPair{.var = v, .value = a}};
            std::ranges::sort(load_pre);
            t.operators.push_back(Operator{.name = std::format("load{}_{}", p, a),
                                           .preconditions = load_pre,
                                           .effects = {FactPair{.var = v, .value = loc}},
                                           .cost = 1});
            Condition drop_pre{FactPair{.var = 0, .value = a}, FactPair{.var = v, .value = loc}};
            std::ranges::sort(drop_pre);
            t.operators.push_back(Operator{.name = std::format("unload{}_{}", p, a),
                                           .preconditions = drop_pre,
                                           .effects = {FactPair{.var = v, .value = a}},
                                           .cost = 1});
        }
    }
    return t;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.planning")
        .test("validate_accepts_a_well_formed_task",
              [](TestContext& t) {
                  t.expect(validate(corridor(5)).has_value(), "a corridor task validates");
                  t.expect(validate(switches(3)).has_value(), "a switches task validates");
                  t.expect(validate(logistics(3, 2)).has_value(), "a logistics task validates");
              })
        .test("validate_rejects_the_ways_a_task_can_be_malformed",
              [](TestContext& t) {
                  Task empty;
                  auto a = validate(empty);
                  t.expect(!a.has_value() && a.error() == MathError::domain_error,
                           "a task with no variables is a domain_error");

                  Task bad_domain = corridor(4);
                  bad_domain.domain_size[0] = 0;
                  auto b = validate(bad_domain);
                  t.expect(!b.has_value() && b.error() == MathError::domain_error,
                           "an empty domain is a domain_error");

                  Task short_state = corridor(4);
                  short_state.initial.clear();
                  auto c = validate(short_state);
                  t.expect(!c.has_value() && c.error() == MathError::domain_error,
                           "an initial state missing a variable is a domain_error");

                  Task out_of_range = corridor(4);
                  out_of_range.initial[0] = 99;
                  auto d = validate(out_of_range);
                  t.expect(!d.has_value() && d.error() == MathError::domain_error,
                           "an initial value outside its domain is a domain_error");

                  Task negative_cost = corridor(4);
                  negative_cost.operators[0].cost = -1;
                  auto e = validate(negative_cost);
                  t.expect(!e.has_value() && e.error() == MathError::domain_error,
                           "a negative operator cost is a domain_error");

                  Task unsorted = corridor(4);
                  unsorted.operators[0].preconditions = {FactPair{.var = 0, .value = 0},
                                                         FactPair{.var = 0, .value = 1}};
                  auto f = validate(unsorted);
                  t.expect(!f.has_value() && f.error() == MathError::domain_error,
                           "a condition naming one variable twice is a domain_error -- no state "
                           "could satisfy it, and it would otherwise read as merely unreachable");
              })
        .test("apply_and_applicable_follow_the_operator_semantics",
              [](TestContext& t) {
                  const Task task = corridor(4);
                  const State s0 = task.initial;
                  t.expect(applicable(task, s0, 0), "the forward move from room 0 applies");
                  t.expect(!applicable(task, s0, 1), "the move back from room 1 does not");
                  auto s1 = apply(task, s0, 0);
                  t.expect(s1.has_value(), "applying it succeeds");
                  if (!s1.has_value()) {
                      return;
                  }
                  const State expected{1};
                  t.expect(*s1 == expected, "the truck is now in room 1");
                  auto bad = apply(task, s0, 1);
                  t.expect(!bad.has_value() && bad.error() == MathError::domain_error,
                           "applying an inapplicable operator is a domain_error, never a state "
                           "that quietly ignored a precondition");
                  t.expect(!applicable(task, s0, 999),
                           "an operator index past the end is simply not applicable");
              })
        .test("holds_tests_a_partial_assignment",
              [](TestContext& t) {
                  const State s{1, 0, 2};
                  t.expect(holds(s, Condition{}), "an empty condition holds everywhere");
                  t.expect(holds(s, Condition{FactPair{.var = 0, .value = 1}}),
                           "a satisfied pair holds");
                  t.expect(!holds(s, Condition{FactPair{.var = 0, .value = 0}}),
                           "an unsatisfied pair does not");
                  t.expect(holds(s, Condition{FactPair{.var = 0, .value = 1},
                                              FactPair{.var = 2, .value = 2}}),
                           "several satisfied pairs hold together");
              })
        .test("h_max_is_admissible_on_every_reachable_state",
              [](TestContext& t) {
                  // THE property that makes A* optimal. The optimal cost comes from A* itself, so
                  // this is a genuine check rather than a comparison against a remembered number.
                  for (const Task& task : {corridor(6), switches(3), shared_prefix(3),
                                           logistics(3, 1)}) {
                      auto optimal = astar_plan(task, 200000);
                      t.expect(optimal.has_value(), "A* solves the task");
                      if (!optimal.has_value()) {
                          continue;
                      }
                      const auto h = h_max(task, task.initial);
                      t.expect(h.has_value(), "h_max is defined on a solvable task");
                      if (!h.has_value()) {
                          continue;
                      }
                      t.expect(*h <= optimal->plan.cost,
                               "h_max never exceeds the true optimal cost");
                  }
              })
        .test("h_max_and_h_add_and_h_ff_are_zero_at_the_goal",
              [](TestContext& t) {
                  Task task = corridor(4);
                  const State at_goal{3};
                  t.expect(holds(at_goal, task.goal), "the state satisfies the goal");
                  const auto hm = h_max(task, at_goal);
                  const auto ha = h_add(task, at_goal);
                  const auto hf = h_ff(task, at_goal);
                  t.expect(hm.has_value() && *hm == 0, "h_max is exactly 0 at the goal");
                  t.expect(ha.has_value() && *ha == 0, "h_add is exactly 0 at the goal");
                  t.expect(hf.value.has_value() && *hf.value == 0, "h_ff is exactly 0");
                  t.expect(hf.preferred.empty(),
                           "and there are no helpful actions, since nothing needs doing");
              })
        .test("h_max_counts_the_corridor_exactly",
              [](TestContext& t) {
                  // A corridor has no shared subplans and one goal, so all three heuristics are
                  // exact here -- which makes it the right task to pin the arithmetic on.
                  const Task task = corridor(7);
                  const auto hm = h_max(task, task.initial);
                  const auto ha = h_add(task, task.initial);
                  const auto hf = h_ff(task, task.initial);
                  t.expect(hm.has_value() && *hm == 6, "h_max is exactly 6 for a seven-room walk");
                  t.expect(ha.has_value() && *ha == 6, "so is h_add, with a single goal");
                  t.expect(hf.value.has_value() && *hf.value == 6, "and so is h_ff");
              })
        .test("h_add_overestimates_a_shared_prefix_and_h_ff_does_not",
              [](TestContext& t) {
                  // The defining difference between the two. One setup action enables four goals:
                  // the optimal plan is 1 + 4 = 5. h_add charges the setup to each goal and so
                  // reports 8; h_ff extracts a plan and counts the setup once, reporting 5.
                  const Task task = shared_prefix(4);
                  auto optimal = astar_plan(task, 200000);
                  t.expect(optimal.has_value(), "A* solves it");
                  if (!optimal.has_value()) {
                      return;
                  }
                  t.expect(optimal->plan.cost == 5, "the optimal plan costs exactly 5");

                  const auto ha = h_add(task, task.initial);
                  const auto hf = h_ff(task, task.initial);
                  const auto hm = h_max(task, task.initial);
                  t.expect(ha.has_value() && hf.value.has_value() && hm.has_value(),
                           "all three heuristics are defined");
                  if (!ha.has_value() || !hf.value.has_value() || !hm.has_value()) {
                      return;
                  }
                  t.expect(*ha == 8,
                           "h_add reports 8 -- the setup counted once per goal, which is exactly "
                           "the double counting that makes it inadmissible");
                  t.expect(*ha > optimal->plan.cost,
                           "and it therefore exceeds the true optimum");
                  t.expect(*hf.value == 5,
                           "h_ff reports 5 -- it counts the shared setup once, because it counts "
                           "an actual relaxed plan");
                  t.expect(*hm <= optimal->plan.cost, "h_max stays admissible, as it must");
                  t.expect(*hm == 2, "and is weak here: the dearest single goal costs 2");
              })
        .test("h_ff_reports_helpful_actions_that_are_applicable_now",
              [](TestContext& t) {
                  const Task task = shared_prefix(3);
                  const auto ff = h_ff(task, task.initial);
                  t.expect(ff.value.has_value(), "the heuristic is defined");
                  t.expect(!ff.preferred.empty(), "some action is helpful in the initial state");
                  for (const std::size_t o : ff.preferred) {
                      t.expect(o < task.operators.size(), "each preferred index names an operator");
                      if (o < task.operators.size()) {
                          t.expect(applicable(task, task.initial, o),
                                   "and every preferred operator is applicable right now -- an "
                                   "unapplicable one would be no help at all");
                      }
                  }
                  t.expect(std::ranges::is_sorted(ff.preferred),
                           "preferred operators come back in ascending index order, so the "
                           "ordering is the task's rather than the extraction's");
              })
        .test("the_heuristics_report_an_unreachable_goal_as_such",
              [](TestContext& t) {
                  // A goal no operator can achieve. Under the delete relaxation it is still
                  // unreachable, and a relaxation proving unreachability proves it outright.
                  Task task;
                  task.domain_size = {3};
                  task.initial = {0};
                  task.goal = {FactPair{.var = 0, .value = 2}};
                  task.operators.push_back(Operator{.name = "only",
                                                    .preconditions = {FactPair{.var = 0, .value = 0}},
                                                    .effects = {FactPair{.var = 0, .value = 1}},
                                                    .cost = 1});
                  t.expect(!h_max(task, task.initial).has_value(),
                           "h_max reports no value for an unreachable goal");
                  t.expect(!h_add(task, task.initial).has_value(), "so does h_add");
                  t.expect(!h_ff(task, task.initial).value.has_value(), "and so does h_ff");
                  auto r = astar_plan(task, 100000);
                  t.expect(!r.has_value() && r.error() == MathError::undefined_value,
                           "and A* reports undefined_value -- no plan exists, which is a "
                           "different answer from running out of budget");
              })
        .test("astar_finds_the_optimal_corridor_plan",
              [](TestContext& t) {
                  const Task task = corridor(8);
                  auto r = astar_plan(task, 100000);
                  t.expect(r.has_value(), "A* solves the corridor");
                  if (!r.has_value()) {
                      return;
                  }
                  t.expect(r->plan.cost == 7, "the optimal cost is exactly 7");
                  t.expect(r->plan.length() == 7, "and the plan is exactly seven steps");
                  const auto end = validate_plan(task, r->plan);
                  t.expect(end.has_value(), "the plan replays cleanly to the goal");
              })
        .test("astar_finds_the_optimal_plan_where_greedy_search_need_not",
              [](TestContext& t) {
                  const Task task = shared_prefix(4);
                  auto opt = astar_plan(task, 200000);
                  auto greedy = gbfs_plan(task, 200000);
                  t.expect(opt.has_value() && greedy.has_value(), "both searches find a plan");
                  if (!opt.has_value() || !greedy.has_value()) {
                      return;
                  }
                  t.expect(opt->plan.cost == 5, "A* returns the optimal cost");
                  t.expect(greedy->plan.cost >= opt->plan.cost,
                           "greedy search never beats the optimum -- if it did, A* would be wrong");
                  t.expect(validate_plan(task, greedy->plan).has_value(),
                           "and its plan is valid, which is all it promises");
              })
        .test("every_returned_plan_replays_to_the_goal",
              [](TestContext& t) {
                  // The honesty guarantee: a search that produced an invalid plan is a bug, and
                  // both entry points check before returning rather than trusting themselves.
                  for (const Task& task : {corridor(5), switches(4), shared_prefix(3),
                                           logistics(3, 2)}) {
                      auto a = astar_plan(task, 200000);
                      t.expect(a.has_value(), "A* solves the task");
                      if (a.has_value()) {
                          auto end = validate_plan(task, a->plan);
                          t.expect(end.has_value() && holds(*end, task.goal),
                                   "the A* plan really reaches the goal");
                      }
                      auto g = gbfs_plan(task, 200000);
                      t.expect(g.has_value(), "greedy search solves the task");
                      if (g.has_value()) {
                          auto end = validate_plan(task, g->plan);
                          t.expect(end.has_value() && holds(*end, task.goal),
                                   "the greedy plan really reaches the goal");
                      }
                  }
              })
        .test("validate_plan_rejects_a_plan_that_does_not_reach_the_goal",
              [](TestContext& t) {
                  const Task task = corridor(5);
                  const Plan too_short{.steps = {0, 2}, .cost = 2};
                  auto r = validate_plan(task, too_short);
                  t.expect(!r.has_value() && r.error() == MathError::undefined_value,
                           "a plan that stops short is undefined_value");
                  const Plan inapplicable{.steps = {1}, .cost = 1};
                  auto s = validate_plan(task, inapplicable);
                  t.expect(!s.has_value() && s.error() == MathError::domain_error,
                           "a plan whose first step does not apply is a domain_error");
                  const Plan wrong_cost{.steps = {0, 2, 4, 6}, .cost = 99};
                  auto u = validate_plan(task, wrong_cost);
                  t.expect(!u.has_value() && u.error() == MathError::domain_error,
                           "a plan whose stated cost disagrees with its steps is a domain_error, "
                           "because a plan with a fictional cost is worse than none");
              })
        .test("a_budget_that_runs_out_is_not_converged_and_not_unsolvable",
              [](TestContext& t) {
                  // The two must never be confused: one says no plan exists, the other says we
                  // stopped looking. Reporting the first when the second is true would be a
                  // confident wrong answer.
                  const Task task = logistics(4, 3);
                  auto starved = astar_plan(task, 1);
                  t.expect(!starved.has_value() && starved.error() == MathError::not_converged,
                           "a one-expansion budget is not_converged");
                  const auto ample = astar_plan(task, 500000);
                  t.expect(ample.has_value(),
                           "and the same task solves with a sufficient budget, so the task was "
                           "never the problem");
              })
        .test("preferred_operators_cut_the_search_on_a_logistics_task",
              [](TestContext& t) {
                  // The claim behind gbfs_plan's design, measured rather than asserted. h_ff with
                  // preferred operators against h_add without: same task, same tie-breaks, and the
                  // only difference is the heuristic and the second open list.
                  const Task task = logistics(5, 3);
                  auto with_ff = gbfs_plan(task, 500000);
                  auto with_add = gbfs_plan_hadd(task, 500000);
                  t.expect(with_ff.has_value() && with_add.has_value(),
                           "both configurations solve the task");
                  if (!with_ff.has_value() || !with_add.has_value()) {
                      return;
                  }
                  t.expect(validate_plan(task, with_ff->plan).has_value(),
                           "the h_ff plan is valid");
                  t.expect(validate_plan(task, with_add->plan).has_value(),
                           "the h_add plan is valid");
                  t.expect(with_ff->stats.expanded <= with_add->stats.expanded,
                           "h_ff with preferred operators expands no more states than h_add "
                           "without them on this task");
              })
        .test("greedy_search_scales_past_what_optimal_search_reaches",
              [](TestContext& t) {
                  // The reason satisficing planners exist. A* proves optimality and pays for it;
                  // greedy search gives up the proof and solves instances A* cannot touch within
                  // the same budget.
                  const Task big = logistics(6, 4);
                  auto greedy = gbfs_plan(big, 2000000);
                  t.expect(greedy.has_value(), "greedy search solves the larger task");
                  if (!greedy.has_value()) {
                      return;
                  }
                  t.expect(validate_plan(big, greedy->plan).has_value(),
                           "and the plan it returns is valid");
                  t.expect(greedy->plan.length() >= 4,
                           "four packages need at least four unload actions");
              })
        .test("search_is_deterministic",
              [](TestContext& t) {
                  // A planner whose plan changed between runs would be far harder to trust, and
                  // the tie-breaks exist precisely to prevent it.
                  const Task task = logistics(4, 2);
                  auto a1 = astar_plan(task, 500000);
                  auto a2 = astar_plan(task, 500000);
                  t.expect(a1.has_value() && a2.has_value(), "A* solves it twice");
                  if (a1.has_value() && a2.has_value()) {
                      t.expect(a1->plan.steps == a2->plan.steps,
                               "and returns exactly the same plan both times");
                      t.expect(a1->stats.expanded == a2->stats.expanded,
                               "having expanded exactly the same states");
                  }
                  auto g1 = gbfs_plan(task, 500000);
                  auto g2 = gbfs_plan(task, 500000);
                  t.expect(g1.has_value() && g2.has_value(), "greedy search solves it twice");
                  if (g1.has_value() && g2.has_value()) {
                      t.expect(g1->plan.steps == g2->plan.steps, "with the same plan both times");
                  }
              })
        .test("a_task_already_at_its_goal_yields_the_empty_plan",
              [](TestContext& t) {
                  Task task = corridor(4);
                  task.initial = {3};
                  auto r = astar_plan(task, 1000);
                  t.expect(r.has_value(), "A* succeeds");
                  if (!r.has_value()) {
                      return;
                  }
                  t.expect(r->plan.steps.empty(), "the plan is empty");
                  t.expect(r->plan.cost == 0, "and costs nothing");
                  auto g = gbfs_plan(task, 1000);
                  t.expect(g.has_value() && g->plan.steps.empty(),
                           "greedy search agrees there is nothing to do");
              })
        .test("operator_costs_other_than_one_are_respected",
              [](TestContext& t) {
                  // Two routes: three cheap steps or one dear one. The optimum is whichever costs
                  // less, and A* must find it rather than the shorter one.
                  Task task;
                  task.domain_size = {4};
                  task.initial = {0};
                  task.goal = {FactPair{.var = 0, .value = 3}};
                  task.operators.push_back(Operator{.name = "a",
                                                    .preconditions = {FactPair{.var = 0, .value = 0}},
                                                    .effects = {FactPair{.var = 0, .value = 1}},
                                                    .cost = 1});
                  task.operators.push_back(Operator{.name = "b",
                                                    .preconditions = {FactPair{.var = 0, .value = 1}},
                                                    .effects = {FactPair{.var = 0, .value = 2}},
                                                    .cost = 1});
                  task.operators.push_back(Operator{.name = "c",
                                                    .preconditions = {FactPair{.var = 0, .value = 2}},
                                                    .effects = {FactPair{.var = 0, .value = 3}},
                                                    .cost = 1});
                  task.operators.push_back(Operator{.name = "jump",
                                                    .preconditions = {FactPair{.var = 0, .value = 0}},
                                                    .effects = {FactPair{.var = 0, .value = 3}},
                                                    .cost = 10});
                  auto r = astar_plan(task, 100000);
                  t.expect(r.has_value(), "A* solves it");
                  if (!r.has_value()) {
                      return;
                  }
                  t.expect(r->plan.cost == 3,
                           "three cheap steps beat one dear jump, so the optimum is 3");
                  t.expect(r->plan.length() == 3, "and the plan takes the long way round");

                  task.operators[3].cost = 2;
                  auto r2 = astar_plan(task, 100000);
                  t.expect(r2.has_value(), "A* solves the reweighted task");
                  if (!r2.has_value()) {
                      return;
                  }
                  t.expect(r2->plan.cost == 2, "with the jump now cheaper, the optimum is 2");
                  t.expect(r2->plan.length() == 1, "and the plan is the single jump");
              })
        .test("zero_cost_operators_do_not_break_the_search",
              [](TestContext& t) {
                  Task task = corridor(4);
                  for (Operator& op : task.operators) {
                      op.cost = 0;
                  }
                  auto r = astar_plan(task, 100000);
                  t.expect(r.has_value(), "A* terminates with zero-cost operators");
                  if (!r.has_value()) {
                      return;
                  }
                  t.expect(r->plan.cost == 0, "every plan costs nothing");
                  t.expect(validate_plan(task, r->plan).has_value(), "and the plan is valid");
              })
        .test("state_encoding_round_trips",
              [](TestContext& t) {
                  const State s{0, 7, 3, 100, 1};
                  const auto bytes = encode_state(s);
                  t.expect(bytes.size() == s.size() * 4, "four bytes per variable");
                  auto back = decode_state(bytes, s.size());
                  t.expect(back.has_value(), "the bytes decode");
                  if (back.has_value()) {
                      t.expect(*back == s, "and give back exactly the same state");
                  }
                  auto wrong = decode_state(bytes, s.size() + 1);
                  t.expect(!wrong.has_value() && wrong.error() == MathError::syntax_error,
                           "a length that does not match the variable count is a syntax_error");
              })
        .test("state_hashing_is_stable_and_discriminating",
              [](TestContext& t) {
                  // A distributed search partitions states by this value, so it must depend only
                  // on the state -- not on the platform, the build or the run.
                  const State a{1, 2, 3};
                  const State b{1, 2, 3};
                  const State c{1, 2, 4};
                  t.expect(hash_state(a) == hash_state(b), "equal states hash equally");
                  t.expect(hash_state(a) != hash_state(c),
                           "a state differing in one value hashes differently");
                  // Held in a variable first: `hash_state(a) == hash_state(a)` is one call
                  // compared with itself, which says nothing about stability across calls.
                  const std::uint64_t first_hash = hash_state(a);
                  t.expect(hash_state(a) == first_hash, "and the hash is stable across calls");
                  const State empty;
                  t.expect(hash_state(empty) == hash_state(State{}),
                           "the empty state hashes consistently too");
              })
        .run();
}
