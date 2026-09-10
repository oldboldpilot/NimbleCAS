// Tests for nimblecas.planning_dist: SAS+ planning as task graphs over parallel f-layer expansion.
// @author Olumuyiwa Oluwasanmi
//
// The contract under test is deliberately narrower than "the same answer", and the tests are
// written to that narrower claim rather than to a convenient one. `distributed_astar_plan`
// promises a plan of the same COST as `astar_plan` -- optimality -- and does NOT promise the same
// plan, because `astar_plan` breaks ties toward the state discovered earlier and expanding a whole
// f-layer at once has no such order. So cost equivalence is asserted against the serial planner on
// every task, plan identity is not, and determinism is asserted separately: the same executor and
// task must give the same plan every time, at any shard count. Every returned plan is additionally
// replayed from the initial state, so a plan that does not reach the goal is a failure and not a
// result.

import std;
import nimblecas.core;
import nimblecas.planning;
import nimblecas.planning_dist;
import nimblecas.taskdag;
import nimblecas.taskdag_sgee;
import nimblecas.testing;

using nimblecas::Executor;
using nimblecas::FakeBrokerPort;
using nimblecas::InMemoryResultChannel;
using nimblecas::local_parallel_executor;
using nimblecas::MathError;
using nimblecas::serial_executor;
using nimblecas::SgeeDistributedExecutor;
using nimblecas::SgeeExecutorConfig;
using nimblecas::TaskRegistry;
using nimblecas::planning::astar_plan;
using nimblecas::planning::FactPair;
using nimblecas::planning::gbfs_plan;
using nimblecas::planning::h_max;
using nimblecas::planning::Operator;
using nimblecas::planning::State;
using nimblecas::planning::Task;
using nimblecas::planning::validate_plan;
using nimblecas::planning_dist::decode_successors;
using nimblecas::planning_dist::decode_task;
using nimblecas::planning_dist::distributed_astar_plan;
using nimblecas::planning_dist::distributed_evaluate;
using nimblecas::planning_dist::distributed_gbfs_plan;
using nimblecas::planning_dist::encode_states;
using nimblecas::planning_dist::encode_task;
using nimblecas::planning_dist::Heuristic;
using nimblecas::planning_dist::max_layer_states;
using nimblecas::planning_dist::register_ops;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// A corridor of `n` cells: the agent starts at 0 and must reach n-1, one step at a time.
[[nodiscard]] auto corridor(std::int32_t n) -> Task {
    Task t;
    t.domain_size.push_back(n);
    t.initial.push_back(0);
    t.goal.push_back(FactPair{.var = 0, .value = n - 1});
    for (std::int32_t i = 0; i + 1 < n; ++i) {
        t.operators.push_back(Operator{.name = std::format("step{}", i),
                                       .preconditions = {FactPair{.var = 0, .value = i}},
                                       .effects = {FactPair{.var = 0, .value = i + 1}},
                                       .cost = 1});
    }
    return t;
}

// `k` independent switches, each flipped by its own operator. The optimal plan is k steps, and
// every ordering of them is optimal -- which is exactly the shape that distinguishes "same cost"
// from "same plan".
[[nodiscard]] auto switches(std::int32_t k) -> Task {
    Task t;
    t.domain_size.assign(static_cast<std::size_t>(k), 2);
    t.initial.assign(static_cast<std::size_t>(k), 0);
    for (std::int32_t i = 0; i < k; ++i) {
        const auto v = static_cast<std::size_t>(i);
        t.goal.push_back(FactPair{.var = v, .value = 1});
        t.operators.push_back(Operator{.name = std::format("flip{}", i),
                                       .preconditions = {FactPair{.var = v, .value = 0}},
                                       .effects = {FactPair{.var = v, .value = 1}},
                                       .cost = 1});
    }
    return t;
}

// One setup action that every goal action needs: the shape on which h_add double-counts.
[[nodiscard]] auto shared_prefix(std::int32_t k) -> Task {
    Task t;
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
        t.operators.push_back(Operator{
            .name = std::format("do{}", i),
            .preconditions = {FactPair{.var = 0, .value = 1}, FactPair{.var = v, .value = 0}},
            .effects = {FactPair{.var = v, .value = 1}},
            .cost = 1});
    }
    return t;
}

// A goal that cannot be reached: the only operator needs a precondition nothing establishes.
[[nodiscard]] auto unsolvable() -> Task {
    Task t;
    t.domain_size = {2, 2};
    t.initial = {0, 0};
    t.goal = {FactPair{.var = 1, .value = 1}};
    t.operators.push_back(Operator{.name = "needs_impossible",
                                   .preconditions = {FactPair{.var = 0, .value = 1}},
                                   .effects = {FactPair{.var = 1, .value = 1}},
                                   .cost = 1});
    return t;
}

[[nodiscard]] auto tasks() -> std::vector<Task> {
    return {corridor(6), corridor(9), switches(3), switches(4), shared_prefix(3),
            shared_prefix(4)};
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.planning_dist")
        .test("the_distributed_plan_costs_exactly_what_the_serial_optimal_plan_costs",
              [](TestContext& t) {
                  // Optimality is the whole claim of this entry point, and the serial planner is
                  // the oracle it is checked against. Cost, not plan: see the header.
                  auto ser = serial_executor();
                  auto par = local_parallel_executor();
                  for (const Task& task : tasks()) {
                      auto want = astar_plan(task, 200000);
                      t.expect(want.has_value(), "the serial planner solves the task");
                      if (!want) {
                          continue;
                      }
                      for (const std::size_t shards : {std::size_t{1}, std::size_t{4}}) {
                          auto a = distributed_astar_plan(task, 200000, shards, *ser);
                          auto b = distributed_astar_plan(task, 200000, shards, *par);
                          t.expect(a.has_value() && b.has_value(), "both distributed runs solve it");
                          if (!a || !b) {
                              continue;
                          }
                          t.expect(a->plan.cost == want->plan.cost,
                                   "serial executor: the plan is optimal");
                          t.expect(b->plan.cost == want->plan.cost,
                                   "parallel executor: the plan is optimal, at any shard count");
                      }
                  }
              })
        .test("randomised_tasks_agree_with_the_serial_optimum_every_time",
              [](TestContext& t) {
                  // The optimality claim is a claim about EVERY task, not about the handful that
                  // happen to be written above -- and hand-picked fixtures are exactly where a
                  // subtly wrong search looks right. So: pseudo-random tasks, a fixed seed so a
                  // failure is reproducible, and the serial planner as the oracle. A single
                  // disagreement in cost would falsify the f-layer argument outright.
                  auto exec = local_parallel_executor();
                  std::mt19937_64 rng(0xA5A5C0FFEEULL);
                  int solved = 0;
                  int agreed = 0;
                  for (int trial = 0; trial < 60; ++trial) {
                      const auto nvars = static_cast<std::size_t>(2 + (rng() % 3));
                      Task task;
                      for (std::size_t v = 0; v < nvars; ++v) {
                          task.domain_size.push_back(static_cast<std::int32_t>(2 + (rng() % 3)));
                      }
                      task.initial.assign(nvars, 0);
                      // A goal on a random subset of the variables.
                      for (std::size_t v = 0; v < nvars; ++v) {
                          if ((rng() % 2) == 0) {
                              const auto d = task.domain_size[v];
                              task.goal.push_back(FactPair{
                                  .var = v,
                                  .value = static_cast<std::int32_t>(rng() % static_cast<
                                      std::uint64_t>(d))});
                          }
                      }
                      std::ranges::sort(task.goal);
                      // Operators with one precondition and one effect, and varying costs, so
                      // the search cannot be right by accident on unit costs alone.
                      const int ops = static_cast<int>(3 + (rng() % 6));
                      for (int o = 0; o < ops; ++o) {
                          const auto pv = static_cast<std::size_t>(rng() % nvars);
                          const auto ev = static_cast<std::size_t>(rng() % nvars);
                          const auto pd = static_cast<std::uint64_t>(task.domain_size[pv]);
                          const auto ed = static_cast<std::uint64_t>(task.domain_size[ev]);
                          task.operators.push_back(Operator{
                              .name = std::format("op{}", o),
                              .preconditions = {FactPair{
                                  .var = pv,
                                  .value = static_cast<std::int32_t>(rng() % pd)}},
                              .effects = {FactPair{
                                  .var = ev,
                                  .value = static_cast<std::int32_t>(rng() % ed)}},
                              .cost = static_cast<std::int64_t>(1 + (rng() % 3))});
                      }

                      auto want = astar_plan(task, 100000);
                      auto got = distributed_astar_plan(task, 100000, 3, *exec);
                      // Both must reach the SAME verdict, whatever it is.
                      if (!want.has_value()) {
                          t.expect(!got.has_value() && got.error() == want.error(),
                                   "an unsolvable or budget-starved task gives the identical "
                                   "error from both planners");
                          continue;
                      }
                      ++solved;
                      t.expect(got.has_value(), "the distributed planner solves what A* solves");
                      if (!got) {
                          continue;
                      }
                      if (got->plan.cost == want->plan.cost) {
                          ++agreed;
                      }
                      t.expect(validate_plan(task, got->plan).has_value(),
                               "and its plan really replays to the goal");
                  }
                  t.expect(solved > 10, "the generator produced a decent number of solvable tasks");
                  t.expect(agreed == solved,
                           "EVERY solvable random task came back at exactly the optimal cost");
              })
        .test("every_returned_plan_actually_reaches_the_goal",
              [](TestContext& t) {
                  // A cost is only meaningful if the plan is real. Each one is replayed from the
                  // initial state rather than trusted.
                  auto exec = local_parallel_executor();
                  for (const Task& task : tasks()) {
                      auto r = distributed_astar_plan(task, 200000, 4, *exec);
                      t.expect(r.has_value(), "the task is solved");
                      if (!r) {
                          continue;
                      }
                      auto const replay = validate_plan(task, r->plan);
                      t.expect(replay.has_value(),
                               "the plan replays from the initial state and reaches the goal");
                      t.expect(r->plan.cost ==
                                   static_cast<std::int64_t>(r->plan.steps.size()),
                               "and its stated cost matches its steps, these tasks being unit-cost");
                  }
              })
        .test("the_answer_does_not_depend_on_the_shard_count_or_the_executor",
              [](TestContext& t) {
                  // Determinism is the property that makes a distributed result reportable at
                  // all: the same task must give the same plan, not merely the same cost.
                  auto ser = serial_executor();
                  auto par = local_parallel_executor();
                  const Task task = shared_prefix(4);
                  auto baseline = distributed_astar_plan(task, 200000, 1, *ser);
                  t.expect(baseline.has_value(), "the baseline run solves the task");
                  if (!baseline) {
                      return;
                  }
                  for (const std::size_t shards : {std::size_t{1}, std::size_t{2}, std::size_t{3},
                                                   std::size_t{8}}) {
                      auto a = distributed_astar_plan(task, 200000, shards, *ser);
                      auto b = distributed_astar_plan(task, 200000, shards, *par);
                      t.expect(a.has_value() && a->plan.steps == baseline->plan.steps,
                               "the identical plan comes back at every shard count");
                      t.expect(b.has_value() && b->plan.steps == baseline->plan.steps,
                               "and from the parallel executor too");
                  }
              })
        .test("a_task_already_at_its_goal_yields_the_empty_plan",
              [](TestContext& t) {
                  // The degenerate case, and the one a layered search can most easily get wrong:
                  // the root is the goal, so the answer is a plan of no steps and cost zero, not
                  // an error and not one wasted expansion.
                  Task done;
                  done.domain_size = {3, 3};
                  done.initial = {1, 2};
                  done.goal = {FactPair{.var = 0, .value = 1}, FactPair{.var = 1, .value = 2}};
                  done.operators.push_back(Operator{.name = "noop_ish",
                                                    .preconditions = {FactPair{.var = 0, .value = 0}},
                                                    .effects = {FactPair{.var = 0, .value = 1}},
                                                    .cost = 1});
                  auto exec = local_parallel_executor();
                  auto r = distributed_astar_plan(done, 1000, 4, *exec);
                  t.expect(r.has_value(), "a task already at its goal is solved, not refused");
                  t.expect(r.has_value() && r->plan.steps.empty(), "with a plan of no steps");
                  t.expect(r.has_value() && r->plan.cost == 0, "and a cost of zero");
                  t.expect(r.has_value() && r->stats.expanded == 0,
                           "having expanded nothing at all");
                  auto want = astar_plan(done, 1000);
                  t.expect(want.has_value() && r.has_value() &&
                               r->plan.cost == want->plan.cost,
                           "which is what the serial planner says too");

                  auto g = distributed_gbfs_plan(done, 1000, 4, 4, *exec);
                  t.expect(g.has_value() && g->plan.steps.empty(),
                           "and the satisficing search agrees");
              })
        .test("a_goal_already_in_hand_is_returned_even_at_a_budget_of_zero",
              [](TestContext& t) {
                  // Found by adversarial review. The budget was checked BEFORE the goal test,
                  // so a task whose goal is already satisfied came back not_converged at
                  // max_expansions = 0 while the serial planner returned the empty plan --
                  // a divergence at every budget boundary, and precisely the equivalence this
                  // module exists to provide.
                  Task done;
                  done.domain_size = {2, 2};
                  done.initial = {1, 1};
                  done.goal = {FactPair{.var = 0, .value = 1}};
                  done.operators.push_back(Operator{.name = "irrelevant",
                                                    .preconditions = {FactPair{.var = 0, .value = 0}},
                                                    .effects = {FactPair{.var = 0, .value = 1}},
                                                    .cost = 1});
                  auto exec = local_parallel_executor();
                  auto want = astar_plan(done, 0);
                  auto got = distributed_astar_plan(done, 0, 4, *exec);
                  t.expect(want.has_value(),
                           "the serial planner returns the empty plan at a budget of zero");
                  t.expect(got.has_value(),
                           "and so does the distributed one -- not not_converged");
                  t.expect(got.has_value() && want.has_value() &&
                               got->plan.cost == want->plan.cost,
                           "at the identical cost");
                  auto g = distributed_gbfs_plan(done, 0, 4, 4, *exec);
                  t.expect(g.has_value() && g->plan.steps.empty(),
                           "the satisficing search behaves the same way");
              })
        .test("costs_near_the_top_of_the_range_saturate_instead_of_overflowing",
              [](TestContext& t) {
                  // Also found by review. `g + step_cost` was a plain signed addition on
                  // caller-supplied operator costs, so a path of two enormous steps overflowed
                  // to a NEGATIVE g, which sorts to the front of the f-order and makes the
                  // search return a nonsense "optimum". The serial planner saturates at exactly
                  // these sums; anything else here is a silent divergence.
                  constexpr std::int64_t huge = std::numeric_limits<std::int64_t>::max() / 2;
                  Task t2;
                  t2.domain_size = {3};
                  t2.initial = {0};
                  t2.goal = {FactPair{.var = 0, .value = 2}};
                  t2.operators.push_back(Operator{.name = "big1",
                                                  .preconditions = {FactPair{.var = 0, .value = 0}},
                                                  .effects = {FactPair{.var = 0, .value = 1}},
                                                  .cost = huge});
                  t2.operators.push_back(Operator{.name = "big2",
                                                  .preconditions = {FactPair{.var = 0, .value = 1}},
                                                  .effects = {FactPair{.var = 0, .value = 2}},
                                                  .cost = huge});
                  auto exec = local_parallel_executor();
                  auto want = astar_plan(t2, 100000);
                  auto got = distributed_astar_plan(t2, 100000, 2, *exec);
                  t.expect(want.has_value() && got.has_value(),
                           "both planners solve a task with enormous operator costs");
                  if (want && got) {
                      t.expect(got->plan.cost == want->plan.cost,
                               "and agree on the cost rather than diverging through an overflow");
                      t.expect(got->plan.cost > 0,
                               "which is positive -- an overflowed sum would have gone negative");
                      t.expect(validate_plan(t2, got->plan).has_value(),
                               "and the plan still replays to the goal");
                  }
              })
        .test("a_state_re_reached_more_cheaply_is_not_expanded_twice",
              [](TestContext& t) {
                  // Found by adversarial review. When a state is re-reached more cheaply the old
                  // node stays in `open` carrying the dearer g. Expanding it is harmless -- its
                  // successors all lose to the cheaper path -- but harmless work still costs a
                  // round trip per successor, which is the compute this module exists to spend
                  // well. The shape below reaches var 0's value 2 by a dear direct step and by a
                  // cheap two-step route, so the dear node is superseded before its layer comes
                  // up.
                  Task t2;
                  t2.domain_size = {4};
                  t2.initial = {0};
                  t2.goal = {FactPair{.var = 0, .value = 3}};
                  t2.operators.push_back(Operator{.name = "direct_dear",
                                                  .preconditions = {FactPair{.var = 0, .value = 0}},
                                                  .effects = {FactPair{.var = 0, .value = 2}},
                                                  .cost = 5});
                  t2.operators.push_back(Operator{.name = "hop1",
                                                  .preconditions = {FactPair{.var = 0, .value = 0}},
                                                  .effects = {FactPair{.var = 0, .value = 1}},
                                                  .cost = 1});
                  t2.operators.push_back(Operator{.name = "hop2",
                                                  .preconditions = {FactPair{.var = 0, .value = 1}},
                                                  .effects = {FactPair{.var = 0, .value = 2}},
                                                  .cost = 1});
                  t2.operators.push_back(Operator{.name = "finish",
                                                  .preconditions = {FactPair{.var = 0, .value = 2}},
                                                  .effects = {FactPair{.var = 0, .value = 3}},
                                                  .cost = 1});
                  auto exec = local_parallel_executor();
                  auto want = astar_plan(t2, 100000);
                  auto got = distributed_astar_plan(t2, 100000, 2, *exec);
                  t.expect(want.has_value() && got.has_value(), "both planners solve it");
                  if (!want || !got) {
                      return;
                  }
                  t.expect(got->plan.cost == want->plan.cost,
                           "the cheap route wins, at the serial planner's cost");
                  t.expect(got->plan.cost == 3, "which is 1 + 1 + 1, not the dear 5 + 1");
                  // The point of the filter: a superseded node is never handed to a worker, so
                  // the expansion count cannot exceed the number of distinct states.
                  t.expect(got->stats.expanded <= 4,
                           "no more expansions than there are states -- a superseded node is "
                           "skipped rather than re-expanded");
                  t.expect(validate_plan(t2, got->plan).has_value(), "and the plan replays");
              })
        .test("an_unsolvable_task_is_proved_unsolvable_not_merely_unfinished",
              [](TestContext& t) {
                  // undefined_value means the reachable state space was exhausted -- a proof.
                  // not_converged means the budget ran out, which proves nothing. Confusing the
                  // two would be the dishonest failure mode for a planner.
                  auto exec = local_parallel_executor();
                  const Task task = unsolvable();
                  auto r = distributed_astar_plan(task, 200000, 4, *exec);
                  t.expect(!r.has_value() && r.error() == MathError::undefined_value,
                           "exhausting the state space is undefined_value, a proof of no plan");

                  auto starved = distributed_astar_plan(corridor(40), 2, 4, *exec);
                  t.expect(!starved.has_value() && starved.error() == MathError::not_converged,
                           "running out of expansions is not_converged, which proves nothing");
              })
        .test("satisficing_search_returns_a_valid_plan_and_claims_nothing_about_its_length",
              [](TestContext& t) {
                  auto exec = local_parallel_executor();
                  for (const Task& task : tasks()) {
                      auto want = astar_plan(task, 200000);
                      auto r = distributed_gbfs_plan(task, 200000, 4, 4, *exec);
                      t.expect(r.has_value(), "greedy search solves the task");
                      if (!r || !want) {
                          continue;
                      }
                      auto const replay = validate_plan(task, r->plan);
                      t.expect(replay.has_value(), "and its plan really reaches the goal");
                      // The only claim available: no plan can be CHEAPER than the optimum.
                      t.expect(r->plan.cost >= want->plan.cost,
                               "a satisficing plan is never cheaper than the optimal one");
                  }
              })
        .test("the_beam_width_changes_the_work_but_not_the_validity",
              [](TestContext& t) {
                  auto exec = local_parallel_executor();
                  const Task task = corridor(9);
                  for (const std::size_t beam : {std::size_t{1}, std::size_t{2}, std::size_t{8}}) {
                      auto r = distributed_gbfs_plan(task, 200000, beam, 4, *exec);
                      t.expect(r.has_value(), "every beam width solves the task");
                      if (r) {
                          t.expect(validate_plan(task, r->plan).has_value(),
                                   "and returns a plan that replays to the goal");
                      }
                  }
                  auto zero = distributed_gbfs_plan(task, 200000, 0, 4, *exec);
                  t.expect(!zero.has_value() && zero.error() == MathError::domain_error,
                           "a zero beam is a domain_error, not a silently-serial search");
              })
        .test("batch_evaluation_agrees_with_the_in_process_heuristic",
              [](TestContext& t) {
                  // The parallel primitive the searches rest on. If it disagreed with the
                  // in-process heuristic, every search above it would be searching a different
                  // landscape than the one documented.
                  auto exec = local_parallel_executor();
                  const Task task = shared_prefix(4);
                  std::vector<State> states;
                  states.push_back(task.initial);
                  State s1 = task.initial;
                  s1[0] = 1;  // setup done
                  states.push_back(s1);
                  State s2 = s1;
                  s2[1] = 1;  // one goal achieved
                  states.push_back(s2);

                  auto got = distributed_evaluate(task, states, Heuristic::h_max, 3, *exec);
                  t.expect(got.has_value(), "batch evaluation succeeds");
                  if (!got) {
                      return;
                  }
                  t.expect(got->size() == states.size(), "one value per state, in order");
                  for (const auto i : std::views::iota(std::size_t{0}, states.size())) {
                      const auto want = h_max(task, states[i]);
                      t.expect((*got)[i] == want,
                               "each distributed value equals the in-process h_max exactly");
                  }
              })
        .test("the_wire_format_round_trips_and_rejects_corruption",
              [](TestContext& t) {
                  const Task task = shared_prefix(3);
                  auto bytes = encode_task(task);
                  t.expect(bytes.has_value(), "the task encodes");
                  if (!bytes) {
                      return;
                  }
                  auto back = decode_task(*bytes);
                  t.expect(back.has_value(), "and decodes");
                  if (back) {
                      t.expect(back->domain_size == task.domain_size, "domains intact");
                      t.expect(back->initial == task.initial, "initial state intact");
                      t.expect(back->goal == task.goal, "goal intact");
                      t.expect(back->operators.size() == task.operators.size(),
                               "every operator intact");
                      t.expect(!back->operators.empty() &&
                                   back->operators[0].name == task.operators[0].name,
                               "including its name, which travels as bytes");
                  }

                  auto truncated = *bytes;
                  truncated.resize(truncated.size() / 2);
                  auto bad = decode_task(truncated);
                  t.expect(!bad.has_value() && bad.error() == MathError::syntax_error,
                           "a truncated payload is refused");

                  auto wrong = *bytes;
                  wrong[0] = static_cast<std::byte>(0xFF);
                  auto bm = decode_task(wrong);
                  t.expect(!bm.has_value() && bm.error() == MathError::syntax_error,
                           "so is a payload that is not one of ours");

                  auto const empty_batch = encode_states(std::span<const State>{}, Heuristic::h_max);
                  t.expect(!empty_batch.has_value(), "an empty state batch is refused");
              })
        .test("shape_faults_are_refused_before_any_task_is_built",
              [](TestContext& t) {
                  auto exec = serial_executor();
                  const Task task = corridor(5);
                  auto zero = distributed_astar_plan(task, 1000, 0, *exec);
                  t.expect(!zero.has_value() && zero.error() == MathError::domain_error,
                           "a zero shard count is a domain_error");

                  Task malformed = task;
                  malformed.goal = {FactPair{.var = 9, .value = 0}};  // no such variable
                  auto bad = distributed_astar_plan(malformed, 1000, 2, *exec);
                  t.expect(!bad.has_value() && bad.error() == MathError::domain_error,
                           "a malformed task is refused rather than distributed");
                  t.expect(max_layer_states > 0, "the layer cap is a real published constant");
              })
        .test("the_same_plan_comes_back_over_the_sgee_distributed_executor",
              [](TestContext& t) {
                  // The Executor seam alone is not enough: this drives the real distributed
                  // executor, so the task and every state batch go through Payload encoding, a
                  // broker queue and the worker's own registry lookup of the op id. The transport
                  // is an in-process broker, so what this establishes is the executor, the wire
                  // format and the registry contract -- not a cluster under partition or latency.
                  TaskRegistry reg;
                  t.expect(register_ops(reg).has_value(),
                           "both operations register into the executor's registry");
                  FakeBrokerPort port;
                  InMemoryResultChannel results;
                  SgeeExecutorConfig cfg;
                  cfg.registry = &reg;
                  cfg.num_workers = 4;
                  cfg.poll_interval_ms = 1;
                  SgeeDistributedExecutor sgee(cfg, port, results);
                  t.expect(sgee.name() == "sgee_distributed",
                           "the executor under test really is the distributed one");

                  const Task task = shared_prefix(3);
                  auto ser = serial_executor();
                  auto a = distributed_astar_plan(task, 200000, 3, *ser);
                  auto b = distributed_astar_plan(task, 200000, 3, sgee);
                  t.expect(a.has_value() && b.has_value(), "both runs solve the task");
                  if (a && b) {
                      t.expect(a->plan.steps == b->plan.steps,
                               "and the cluster returns the identical plan");
                      auto want = astar_plan(task, 200000);
                      t.expect(want.has_value() && b->plan.cost == want->plan.cost,
                               "which is optimal, checked against the serial planner");
                  }

                  auto no_plan = distributed_astar_plan(unsolvable(), 200000, 2, sgee);
                  t.expect(!no_plan.has_value() &&
                               no_plan.error() == MathError::undefined_value,
                           "and an unsolvability proof survives the round trip through the broker");
              })
        .run();
}
