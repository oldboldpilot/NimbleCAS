// Tests for nimblecas.taskdag_sched: cost-aware task scheduling, deterministic
// wavefront ordering, and cost-ordered local executor.
// Every expected value is hand-derived and exact.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.parallel;
import nimblecas.taskdag;
import nimblecas.taskdag_sched;
import nimblecas.testing;

using nimblecas::Affinity;
using nimblecas::AffinityTable;
using nimblecas::CostHint;
using nimblecas::CostTable;
using nimblecas::Executor;
using nimblecas::MathError;
using nimblecas::OpId;
using nimblecas::Payload;
using nimblecas::Result;
using nimblecas::ScheduleParams;
using nimblecas::TaskFn;
using nimblecas::TaskGraph;
using nimblecas::TaskId;
using nimblecas::TaskRegistry;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// --- Payload <-> int64 encode/decode helpers ---

[[nodiscard]] auto encode_i64(std::int64_t v) -> Payload {
    const auto bytes = std::bit_cast<std::array<std::byte, sizeof(std::int64_t)>>(v);
    return Payload(bytes.begin(), bytes.end());
}

[[nodiscard]] auto decode_i64(std::span<const std::byte> p) -> std::int64_t {
    std::array<std::byte, sizeof(std::int64_t)> bytes{};
    std::ranges::copy(p, bytes.begin());
    return std::bit_cast<std::int64_t>(bytes);
}

// A no-dependency task that succeeds with constant `v`.
[[nodiscard]] auto const_task(std::int64_t v) -> TaskFn {
    return [v](std::span<const Payload>) -> Result<Payload> { return encode_i64(v); };
}

// A task that fails with given error regardless of inputs.
[[nodiscard]] auto failing_task(MathError err) -> TaskFn {
    return [err](std::span<const Payload>) -> Result<Payload> {
        return nimblecas::make_error<Payload>(err);
    };
}

// Element-wise comparison of Result<Payload>.
[[nodiscard]] auto results_equal(const Result<Payload>& a, const Result<Payload>& b) -> bool {
    if (a.has_value() != b.has_value()) {
        return false;
    }
    return a.has_value() ? (*a == *b) : (a.error() == b.error());
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.taskdag_sched")
        .test("sanitize_hostile_and_edge_case_inputs",
              [](TestContext& t) {
                  constexpr double qnan = std::numeric_limits<double>::quiet_NaN();
                  constexpr double inf = std::numeric_limits<double>::infinity();

                  // Default / zero is unknown.
                  const auto s_def = nimblecas::sanitize(CostHint{0.0, 0.0});
                  t.expect(s_def.mean_seconds == 0.0 && s_def.variance == 0.0, "default {0, 0} is unknown");

                  // Negative / negative zero mean becomes unknown.
                  const auto s_neg = nimblecas::sanitize(CostHint{-1.0, 10.0});
                  t.expect(s_neg.mean_seconds == 0.0 && s_neg.variance == 0.0, "negative mean -> unknown");

                  const auto s_negz = nimblecas::sanitize(CostHint{-0.0, 0.0});
                  t.expect(s_negz.mean_seconds == 0.0 && s_negz.variance == 0.0, "-0.0 mean -> unknown");

                  // Non-finite mean becomes unknown.
                  const auto s_nan = nimblecas::sanitize(CostHint{qnan, 10.0});
                  t.expect(s_nan.mean_seconds == 0.0 && s_nan.variance == 0.0, "NaN mean -> unknown");

                  const auto s_pinf = nimblecas::sanitize(CostHint{inf, 10.0});
                  t.expect(s_pinf.mean_seconds == 0.0 && s_pinf.variance == 0.0, "+inf mean -> unknown");

                  const auto s_ninf = nimblecas::sanitize(CostHint{-inf, 10.0});
                  t.expect(s_ninf.mean_seconds == 0.0 && s_ninf.variance == 0.0, "-inf mean -> unknown");

                  // Enormous values clamp safely.
                  const auto s_huge = nimblecas::sanitize(CostHint{1e300, 1e300});
                  t.expect(s_huge.mean_seconds == 1e9, "enormous mean clamped to 1e9");
                  t.expect(s_huge.variance == 1e18, "enormous variance clamped to 1e18");

                  // Tiny positive values clamp to lower bound.
                  const auto s_tiny = nimblecas::sanitize(CostHint{1e-12, 0.0});
                  t.expect(s_tiny.mean_seconds == 1e-9, "tiny positive mean clamped to 1e-9");
                  t.expect(s_tiny.variance == 0.0, "zero variance remains 0.0");

                  // Hostile / negative / non-finite variance resets to 0.0.
                  const auto s_negvar = nimblecas::sanitize(CostHint{5.0, -10.0});
                  t.expect(s_negvar.mean_seconds == 5.0 && s_negvar.variance == 0.0, "negative var -> 0.0");

                  const auto s_nanvar = nimblecas::sanitize(CostHint{5.0, qnan});
                  t.expect(s_nanvar.mean_seconds == 5.0 && s_nanvar.variance == 0.0, "NaN var -> 0.0");

                  const auto s_infvar = nimblecas::sanitize(CostHint{5.0, inf});
                  t.expect(s_infvar.mean_seconds == 5.0 && s_infvar.variance == 0.0, "+inf var -> 0.0");

                  // Valid finite values preserved.
                  const auto s_valid = nimblecas::sanitize(CostHint{3.5, 1.25});
                  t.expect(s_valid.mean_seconds == 3.5 && s_valid.variance == 1.25, "valid values preserved");
              })
        .test("effective_cost_three_tier_resolution",
              [](TestContext& t) {
                  CostTable table;
                  table["test.add/v1"] = CostHint{20.0, 9.0};
                  table["test.mul/v1"] = CostHint{10.0, 0.0};

                  // Tier 1: per-task hint takes precedence when known.
                  const CostHint task_hint{10.0, 4.0};
                  const double cost_t1_l0 = nimblecas::effective_cost(
                      task_hint, "test.add/v1", &table, ScheduleParams{.risk_lambda = 0.0});
                  t.expect(cost_t1_l0 == 10.0, "Tier 1: task hint overrides table entry (mean 10.0)");

                  // Variance risk margin: mean + lambda * sqrt(variance) = 10.0 + 1.5 * 2.0 = 13.0
                  const double cost_t1_l15 = nimblecas::effective_cost(
                      task_hint, "test.add/v1", &table, ScheduleParams{.risk_lambda = 1.5});
                  t.expect(cost_t1_l15 == 13.0, "Tier 1: risk margin 10.0 + 1.5*2.0 == 13.0");

                  // Tier 2: unknown task hint falls back to CostTable by OpId.
                  const CostHint unk_hint{0.0, 0.0};
                  const double cost_t2_l0 = nimblecas::effective_cost(
                      unk_hint, "test.add/v1", &table, ScheduleParams{.risk_lambda = 0.0});
                  t.expect(cost_t2_l0 == 20.0, "Tier 2: unknown task hint uses CostTable mean 20.0");

                  // Tier 2 with risk margin: 20.0 + 2.0 * sqrt(9.0) = 20.0 + 6.0 = 26.0
                  const double cost_t2_l2 = nimblecas::effective_cost(
                      unk_hint, "test.add/v1", &table, ScheduleParams{.risk_lambda = 2.0});
                  t.expect(cost_t2_l2 == 26.0, "Tier 2: table risk margin 20.0 + 2.0*3.0 == 26.0");

                  // Hostile task hint sanitized to unknown also falls back to CostTable.
                  const CostHint hostile_hint{-5.0, 0.0};
                  const double cost_t2_hostile = nimblecas::effective_cost(
                      hostile_hint, "test.add/v1", &table, ScheduleParams{.risk_lambda = 0.0});
                  t.expect(cost_t2_hostile == 20.0, "Tier 2: hostile task hint sanitized and uses table");

                  // Tier 3: unknown task hint and missing from CostTable -> 0.0.
                  const double cost_t3_missing = nimblecas::effective_cost(
                      unk_hint, "test.unknown/v1", &table, ScheduleParams{.risk_lambda = 1.0});
                  t.expect(cost_t3_missing == 0.0, "Tier 3: missing table entry -> 0.0");

                  // Tier 3: nullptr table -> 0.0.
                  const double cost_t3_notable = nimblecas::effective_cost(
                      unk_hint, "test.add/v1", nullptr, ScheduleParams{.risk_lambda = 1.0});
                  t.expect(cost_t3_notable == 0.0, "Tier 3: nullptr table -> 0.0");

                  // Closure task (empty op_id) never consults table.
                  const double cost_t3_closure = nimblecas::effective_cost(
                      unk_hint, "", &table, ScheduleParams{.risk_lambda = 1.0});
                  t.expect(cost_t3_closure == 0.0, "Tier 3: closure task with empty op_id -> 0.0");

                  // Hostile lambda is sanitized to 0.0.
                  const double cost_hostile_lambda = nimblecas::effective_cost(
                      task_hint, "test.add/v1", &table,
                      ScheduleParams{.risk_lambda = std::numeric_limits<double>::quiet_NaN()});
                  t.expect(cost_hostile_lambda == 10.0, "NaN risk_lambda sanitized to 0.0");
              })
        .test("schedule_determinism_t1",
              [](TestContext& t) {
                  // M6_SPEC T1: Hints {id0: 1.0, id1: 5.0, id2: 3.0, id3: 5.0, id4: unknown}
                  // Expected order: [1, 3, 2, 0, 4]
                  // Hand-computed rationale:
                  // - 5.0 tie between id1 and id3 broken by TaskId 1 < 3 -> [1, 3]
                  // - 3.0 for id2 -> [2]
                  // - 1.0 for id0 -> [0]
                  // - unknown (0.0) for id4 -> [4]

                  auto build_t1_graph = []() -> TaskGraph {
                      TaskGraph g;
                      (void)g.add_task(const_task(0), {}, CostHint{1.0, 0.0});  // id 0
                      (void)g.add_task(const_task(1), {}, CostHint{5.0, 0.0});  // id 1
                      (void)g.add_task(const_task(2), {}, CostHint{3.0, 0.0});  // id 2
                      (void)g.add_task(const_task(3), {}, CostHint{5.0, 0.0});  // id 3
                      (void)g.add_task(const_task(4), {}, CostHint{0.0, 0.0});  // id 4 (unknown)
                      return g;
                  };

                  const TaskGraph g1 = build_t1_graph();
                  const auto raw_level1 = g1.level(0);
                  const ScheduleParams params{};

                  // Run twice on the same graph:
                  const auto order_1a = nimblecas::level_order(g1, raw_level1, nullptr, params);
                  const auto order_1b = nimblecas::level_order(g1, raw_level1, nullptr, params);

                  // Run once on a freshly rebuilt identical graph:
                  const TaskGraph g2 = build_t1_graph();
                  const auto raw_level2 = g2.level(0);
                  const auto order_2 = nimblecas::level_order(g2, raw_level2, nullptr, params);

                  const std::vector<TaskId> expected{TaskId{1}, TaskId{3}, TaskId{2}, TaskId{0}, TaskId{4}};

                  t.expect(order_1a == expected, "T1: first run matches hand-derived [1, 3, 2, 0, 4]");
                  t.expect(order_1b == expected, "T1: second run on same graph is identical");
                  t.expect(order_2 == expected, "T1: run on freshly rebuilt graph is identical");
              })
        .test("policy_hand_computed_skewed_level_t4",
              [](TestContext& t) {
                  // M6_SPEC T4: Tasks id0..id4 with means {7, 1, 4, 4, 9}
                  // Case A: lambda = 0.0 -> costs are {7, 1, 4, 4, 9}
                  //   Sorted: 9(id4), 7(id0), 4(id2, id3 tie -> 2 < 3), 1(id1)
                  //   Expected order: [4, 0, 2, 3, 1]
                  // Case B: lambda = 1.0, variances {0, 0, 9, 0, 0}
                  //   Effective costs: id0=7, id1=1, id2=4+1*3=7, id3=4, id4=9
                  //   id0 and id2 tie with cost 7 -> tie broken by TaskId 0 < 2 -> [4, 0, 2, 3, 1]
                  // Case C: lambda = 2.0, variances {0, 0, 9, 0, 0}
                  //   Effective costs: id0=7, id1=1, id2=4+2*3=10, id3=4, id4=9
                  //   Sorted: 10(id2), 9(id4), 7(id0), 4(id3), 1(id1)
                  //   Expected order: [2, 4, 0, 3, 1]

                  TaskGraph g;
                  (void)g.add_task(const_task(0), {}, CostHint{7.0, 0.0});  // id 0
                  (void)g.add_task(const_task(1), {}, CostHint{1.0, 0.0});  // id 1
                  (void)g.add_task(const_task(2), {}, CostHint{4.0, 9.0});  // id 2 (var=9.0)
                  (void)g.add_task(const_task(3), {}, CostHint{4.0, 0.0});  // id 3
                  (void)g.add_task(const_task(4), {}, CostHint{9.0, 0.0});  // id 4

                  const auto level = g.level(0);

                  // Case A (lambda = 0):
                  const auto order_a = nimblecas::level_order(
                      g, level, nullptr, ScheduleParams{.risk_lambda = 0.0});
                  const std::vector<TaskId> expected_a{TaskId{4}, TaskId{0}, TaskId{2}, TaskId{3}, TaskId{1}};
                  t.expect(order_a == expected_a, "T4 lambda=0: order matches [4, 0, 2, 3, 1]");

                  // Case B (lambda = 1):
                  const auto order_b = nimblecas::level_order(
                      g, level, nullptr, ScheduleParams{.risk_lambda = 1.0});
                  const std::vector<TaskId> expected_b{TaskId{4}, TaskId{0}, TaskId{2}, TaskId{3}, TaskId{1}};
                  t.expect(order_b == expected_b, "T4 lambda=1: id0/id2 tie at 7 resolved to 0 < 2 -> [4, 0, 2, 3, 1]");

                  // Case C (lambda = 2):
                  const auto order_c = nimblecas::level_order(
                      g, level, nullptr, ScheduleParams{.risk_lambda = 2.0});
                  const std::vector<TaskId> expected_c{TaskId{2}, TaskId{4}, TaskId{0}, TaskId{3}, TaskId{1}};
                  t.expect(order_c == expected_c, "T4 lambda=2: id2 cost 10 exceeds id4 cost 9 -> [2, 4, 0, 3, 1]");
              })
        .test("hostile_hints_and_extreme_values_t6",
              [](TestContext& t) {
                  // M6_SPEC T6(i): 5-task level with hints {-1.0, 0.0, NaN, +inf, 1e300}
                  // First four are all sanitized to unknown (cost 0.0);
                  // 1e300 is clamped to 1e9 (known, largest cost).
                  // Expected order: [4, 0, 1, 2, 3] (id4 first, then unknown tasks in ascending TaskId order).

                  constexpr double qnan = std::numeric_limits<double>::quiet_NaN();
                  constexpr double inf = std::numeric_limits<double>::infinity();

                  auto build_t6_graph = [&]() -> TaskGraph {
                      TaskGraph g;
                      (void)g.add_task(const_task(0), {}, CostHint{-1.0, 0.0});   // id 0: negative
                      (void)g.add_task(const_task(1), {}, CostHint{0.0, 0.0});    // id 1: zero
                      (void)g.add_task(const_task(2), {}, CostHint{qnan, 0.0});   // id 2: NaN
                      (void)g.add_task(const_task(3), {}, CostHint{inf, 0.0});    // id 3: +inf
                      (void)g.add_task(const_task(4), {}, CostHint{1e300, 0.0});  // id 4: 1e300
                      return g;
                  };

                  const TaskGraph g = build_t6_graph();
                  const auto level = g.level(0);
                  const ScheduleParams params{};

                  const auto order1 = nimblecas::level_order(g, level, nullptr, params);
                  const auto order2 = nimblecas::level_order(g, level, nullptr, params);

                  const std::vector<TaskId> expected{TaskId{4}, TaskId{0}, TaskId{1}, TaskId{2}, TaskId{3}};

                  t.expect(order1 == expected, "T6: clamped 1e300 first, unknown tasks in issuance order [4, 0, 1, 2, 3]");
                  t.expect(order2 == expected, "T6: second run is deterministic");
              })
        .test("hint_free_graph_strict_no_op_t7",
              [](TestContext& t) {
                  // M6_SPEC T7: On a hint-free graph, level_order is a strict no-op
                  // (preserves ascending TaskId issuance order).
                  TaskGraph g;
                  std::vector<TaskId> expected;
                  for (std::size_t i = 0; i < 8; ++i) {
                      expected.push_back(g.add_task(const_task(static_cast<std::int64_t>(i))).value());
                  }

                  const auto level = g.level(0);

                  // With nullptr CostTable:
                  const auto order_notable = nimblecas::level_order(
                      g, level, nullptr, ScheduleParams{});
                  t.expect(order_notable == expected, "hint-free with nullptr table -> strict no-op");

                  // With empty CostTable:
                  const CostTable empty_table;
                  const auto order_empty = nimblecas::level_order(
                      g, level, &empty_table, ScheduleParams{});
                  t.expect(order_empty == expected, "hint-free with empty table -> strict no-op");

                  // With CostTable containing unrelated ops:
                  CostTable unrelated_table;
                  unrelated_table["other.op/v1"] = CostHint{100.0, 0.0};
                  const auto order_unrelated = nimblecas::level_order(
                      g, level, &unrelated_table, ScheduleParams{});
                  t.expect(order_unrelated == expected, "hint-free with unrelated table -> strict no-op");
              })
        .test("bit_identity_to_serial_executor_and_error_poisoning",
              [](TestContext& t) {
                  // M6_SPEC T3a: Outputs from serial_executor() == cost_ordered_local_executor()
                  // byte-for-byte on a multi-level diamond graph with error poisoning and mixed costs.
                  //
                  // Topology:
                  // Level 0:
                  //   a = 10 (hint 1.0)
                  //   b = failing_task(domain_error) (hint 50.0) -> FAILS, origin=b
                  //   c = 30 (hint 5.0)
                  //   d = 40 (hint 2.0)
                  // Level 1:
                  //   e = a * 2 = 20 (deps: {a}, hint 3.0) -> SUCCEEDS
                  //   f = b + 1 (deps: {b}, hint 100.0) -> POISONED by b
                  //   g = b + c (deps: {b, c}, hint 0.5) -> POISONED by b (b is failing ancestor)
                  //   h = c + d = 70 (deps: {c, d}, hint 8.0) -> SUCCEEDS
                  // Level 2:
                  //   i = e + h = 20 + 70 = 90 (deps: {e, h}) -> SUCCEEDS
                  //   j = g + h (deps: {g, h}) -> POISONED by b (propagated through g)

                  TaskGraph g;
                  const auto a = g.add_task(const_task(10), {}, CostHint{1.0, 0.0}).value();
                  const auto b = g.add_task(failing_task(MathError::domain_error), {}, CostHint{50.0, 0.0}).value();
                  const auto c = g.add_task(const_task(30), {}, CostHint{5.0, 0.0}).value();
                  const auto d = g.add_task(const_task(40), {}, CostHint{2.0, 0.0}).value();

                  const auto e = g.add_task(
                      [](std::span<const Payload> ps) -> Result<Payload> {
                          return encode_i64(decode_i64(ps[0]) * 2);
                      },
                      std::vector<TaskId>{a}, CostHint{3.0, 0.0}).value();

                  const auto f = g.add_task(
                      [](std::span<const Payload> ps) -> Result<Payload> {
                          return encode_i64(decode_i64(ps[0]) + 1);
                      },
                      std::vector<TaskId>{b}, CostHint{100.0, 0.0}).value();

                  const auto g_task = g.add_task(
                      [](std::span<const Payload> ps) -> Result<Payload> {
                          return encode_i64(decode_i64(ps[0]) + decode_i64(ps[1]));
                      },
                      std::vector<TaskId>{b, c}, CostHint{0.5, 0.0}).value();

                  const auto h = g.add_task(
                      [](std::span<const Payload> ps) -> Result<Payload> {
                          return encode_i64(decode_i64(ps[0]) + decode_i64(ps[1]));
                      },
                      std::vector<TaskId>{c, d}, CostHint{8.0, 0.0}).value();

                  const auto i_task = g.add_task(
                      [](std::span<const Payload> ps) -> Result<Payload> {
                          return encode_i64(decode_i64(ps[0]) + decode_i64(ps[1]));
                      },
                      std::vector<TaskId>{e, h}).value();

                  const auto j = g.add_task(
                      [](std::span<const Payload> ps) -> Result<Payload> {
                          return encode_i64(decode_i64(ps[0]) + decode_i64(ps[1]));
                      },
                      std::vector<TaskId>{g_task, h}).value();

                  const auto ser_exec = nimblecas::serial_executor();
                  const auto ord_exec1 = nimblecas::cost_ordered_local_executor(
                      nullptr, ScheduleParams{.risk_lambda = 0.0}, /*grain=*/1);
                  const auto ord_exec2 = nimblecas::cost_ordered_local_executor(
                      nullptr, ScheduleParams{.risk_lambda = 1.0}, /*grain=*/2);

                  t.expect(ord_exec1->name() == "cost_ordered_local", "executor name is cost_ordered_local");

                  const auto ser_res = ser_exec->run(g).value();
                  const auto ord_res1 = ord_exec1->run(g).value();
                  const auto ord_res2 = ord_exec2->run(g).value();

                  t.expect(ser_res.outputs.size() == g.size(), "serial output count == graph size");
                  t.expect(ord_res1.outputs.size() == g.size(), "ordered output count == graph size");
                  t.expect(ord_res2.outputs.size() == g.size(), "ordered (grain 2) output count == graph size");

                  // Bit-identity check between serial and cost-ordered local:
                  bool identical1 = true;
                  bool identical2 = true;
                  for (std::size_t k = 0; k < g.size(); ++k) {
                      if (!results_equal(ser_res.outputs[k], ord_res1.outputs[k])) {
                          identical1 = false;
                      }
                      if (!results_equal(ser_res.outputs[k], ord_res2.outputs[k])) {
                          identical2 = false;
                      }
                  }
                  t.expect(identical1, "outputs bit-identical between serial and cost_ordered_local (grain 1)");
                  t.expect(identical2, "outputs bit-identical between serial and cost_ordered_local (grain 2)");

                  // Executed count: a, b (ran & failed), c, d, e, h, i_task = 7 executed; f, g, j poisoned (not executed)
                  t.expect(ser_res.executed == 7, "serial executed count == 7");
                  t.expect(ord_res1.executed == 7, "cost_ordered executed count == 7");
                  t.expect(ord_res2.executed == 7, "cost_ordered (grain 2) executed count == 7");

                  // Verify specific outputs:
                  t.expect(ser_res.outputs[a.value].has_value() && decode_i64(ser_res.outputs[a.value].value()) == 10, "a == 10");
                  t.expect(!ser_res.outputs[b.value].has_value() && ser_res.outputs[b.value].error() == MathError::domain_error, "b failed with domain_error");
                  t.expect(ser_res.outputs[c.value].has_value() && decode_i64(ser_res.outputs[c.value].value()) == 30, "c == 30");
                  t.expect(ser_res.outputs[d.value].has_value() && decode_i64(ser_res.outputs[d.value].value()) == 40, "d == 40");
                  t.expect(ser_res.outputs[e.value].has_value() && decode_i64(ser_res.outputs[e.value].value()) == 20, "e == 20");
                  t.expect(!ser_res.outputs[f.value].has_value() && ser_res.outputs[f.value].error() == MathError::domain_error, "f poisoned");
                  t.expect(!ser_res.outputs[g_task.value].has_value() && ser_res.outputs[g_task.value].error() == MathError::domain_error, "g poisoned");
                  t.expect(ser_res.outputs[h.value].has_value() && decode_i64(ser_res.outputs[h.value].value()) == 70, "h == 70");
                  t.expect(ser_res.outputs[i_task.value].has_value() && decode_i64(ser_res.outputs[i_task.value].value()) == 90, "i == 90");
                  t.expect(!ser_res.outputs[j.value].has_value() && ser_res.outputs[j.value].error() == MathError::domain_error, "j poisoned");
              })
        .test("affinity_mapping_and_table",
              [](TestContext& t) {
                  // SGEE wire placement values: cpu = 1, gpu = 2
                  t.expect(nimblecas::to_placement(Affinity::cpu_only) == 1, "cpu_only -> 1 (SgeePlacement::cpu)");
                  t.expect(nimblecas::to_placement(Affinity::gpu_only) == 2, "gpu_only -> 2 (SgeePlacement::gpu)");
                  t.expect(nimblecas::to_placement(Affinity::hybrid) == 1, "hybrid -> 1 (SgeePlacement::cpu)");

                  AffinityTable aff_table;
                  aff_table["gpu.gemm/v1"] = Affinity::gpu_only;
                  aff_table["cpu.sort/v1"] = Affinity::cpu_only;
                  aff_table["hybrid.eval/v1"] = Affinity::hybrid;

                  t.expect(aff_table.find("gpu.gemm/v1")->second == Affinity::gpu_only, "AffinityTable lookup gpu");
                  t.expect(aff_table.find("cpu.sort/v1")->second == Affinity::cpu_only, "AffinityTable lookup cpu");
                  t.expect(aff_table.find("hybrid.eval/v1")->second == Affinity::hybrid, "AffinityTable lookup hybrid");
              })
        .run();
}
