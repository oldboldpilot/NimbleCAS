// Tests for nimblecas.taskdag: deterministic single-node wavefront task-DAG scheduling.
// Every expected value is hand-derived and exact (int64 arithmetic encoded into Payload).
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.taskdag;
import nimblecas.testing;

using nimblecas::Executor;
using nimblecas::MathError;
using nimblecas::Payload;
using nimblecas::Result;
using nimblecas::TaskFn;
using nimblecas::TaskGraph;
using nimblecas::TaskId;
using nimblecas::TaskRegistry;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// --- Payload <-> int64 encode/decode (the "serializable by construction" contract). ------

[[nodiscard]] auto encode_i64(std::int64_t v) -> Payload {
    const auto bytes = std::bit_cast<std::array<std::byte, sizeof(std::int64_t)>>(v);
    return Payload(bytes.begin(), bytes.end());
}

[[nodiscard]] auto decode_i64(std::span<const std::byte> p) -> std::int64_t {
    std::array<std::byte, sizeof(std::int64_t)> bytes{};
    std::ranges::copy(p, bytes.begin());
    return std::bit_cast<std::int64_t>(bytes);
}

// A no-dependency task that always succeeds with the constant `v`.
[[nodiscard]] auto const_task(std::int64_t v) -> TaskFn {
    return [v](std::span<const Payload>) -> Result<Payload> { return encode_i64(v); };
}

// A task that always fails, regardless of its (possibly empty) inputs.
[[nodiscard]] auto failing_task(MathError err) -> TaskFn {
    return [err](std::span<const Payload>) -> Result<Payload> { return nimblecas::make_error<Payload>(err); };
}

// Element-wise comparison of two Result<Payload>: both must agree on success/failure, and
// then on the payload bytes or the error code respectively.
[[nodiscard]] auto results_equal(const Result<Payload>& a, const Result<Payload>& b) -> bool {
    if (a.has_value() != b.has_value()) {
        return false;
    }
    return a.has_value() ? (*a == *b) : (a.error() == b.error());
}

// A graph with a wide (500-leaf, well above nimblecas.parallel's default grain of 256)
// fan-out feeding a diamond, used by the determinism test to exercise the parallel path.
[[nodiscard]] auto build_mixed_graph() -> TaskGraph {
    TaskGraph g;
    std::vector<TaskId> leaves;
    leaves.reserve(500);
    for (std::size_t i = 0; i < 500; ++i) {
        leaves.push_back(g.add_task(const_task(static_cast<std::int64_t>(i))).value());
    }
    auto sum = g.add_task(
                    [](std::span<const Payload> ps) -> Result<Payload> {
                        std::int64_t total = 0;
                        for (const Payload& p : ps) {
                            total += decode_i64(p);
                        }
                        return encode_i64(total);
                    },
                    leaves)
                   .value();
    auto doubled = g.add_task(
                        [](std::span<const Payload> ps) -> Result<Payload> {
                            return encode_i64(decode_i64(ps[0]) * 2);
                        },
                        std::vector<TaskId>{sum})
                       .value();
    (void)doubled;
    return g;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.taskdag")
        .test("diamond_dependency_order_and_value",
              [](TestContext& t) {
                  // A=7, B=A*2, C=A+3, D=B+C. Hand value: B=14, C=10, D=24.
                  TaskGraph g;
                  const auto a = g.add_task(const_task(7)).value();
                  const auto b = g.add_task(
                                      [](std::span<const Payload> ps) -> Result<Payload> {
                                          return encode_i64(decode_i64(ps[0]) * 2);
                                      },
                                      std::vector<TaskId>{a})
                                     .value();
                  const auto c = g.add_task(
                                      [](std::span<const Payload> ps) -> Result<Payload> {
                                          return encode_i64(decode_i64(ps[0]) + 3);
                                      },
                                      std::vector<TaskId>{a})
                                     .value();
                  const auto d = g.add_task(
                                      [](std::span<const Payload> ps) -> Result<Payload> {
                                          return encode_i64(decode_i64(ps[0]) + decode_i64(ps[1]));
                                      },
                                      std::vector<TaskId>{b, c})
                                     .value();
                  // An order-sensitive probe (ps[0]*1000 + ps[1]) distinguishes deps given as
                  // {b, c} from {c, b}: 14*1000+10 = 14010 only if B arrived at ps[0].
                  const auto order_probe =
                      g.add_task(
                           [](std::span<const Payload> ps) -> Result<Payload> {
                               return encode_i64(decode_i64(ps[0]) * 1000 + decode_i64(ps[1]));
                           },
                           std::vector<TaskId>{b, c})
                          .value();

                  const auto exec = nimblecas::serial_executor();
                  const auto res = exec->run(g).value();

                  t.expect(res.outputs.size() == g.size(), "one output slot per task");
                  t.expect(res.outputs[a.value].has_value() && decode_i64(res.outputs[a.value].value()) == 7,
                           "A == 7");
                  t.expect(res.outputs[b.value].has_value() && decode_i64(res.outputs[b.value].value()) == 14,
                           "B saw A's output: B == A*2 == 14");
                  t.expect(res.outputs[c.value].has_value() && decode_i64(res.outputs[c.value].value()) == 10,
                           "C saw A's output: C == A+3 == 10");
                  t.expect(res.outputs[d.value].has_value() && decode_i64(res.outputs[d.value].value()) == 24,
                           "D == B+C == 24");
                  t.expect(res.outputs[order_probe.value].has_value() &&
                               decode_i64(res.outputs[order_probe.value].value()) == 14010,
                           "D saw B and C in the exact dep order {b, c}");
                  t.expect(res.executed == g.size(), "every task in the diamond actually ran");
              })
        .test("fan_out_and_aggregate",
              [](TestContext& t) {
                  TaskGraph g;
                  std::vector<TaskId> leaves;
                  leaves.reserve(64);
                  for (std::size_t i = 0; i < 64; ++i) {
                      leaves.push_back(g.add_task(const_task(static_cast<std::int64_t>(i))).value());
                  }
                  const auto agg =
                      g.add_task(
                           [](std::span<const Payload> ps) -> Result<Payload> {
                               std::int64_t total = 0;
                               for (const Payload& p : ps) {
                                   total += decode_i64(p);
                               }
                               return encode_i64(total);
                           },
                           leaves)
                          .value();

                  const auto exec = nimblecas::serial_executor();
                  const auto res = exec->run(g).value();

                  // sum_{i=0}^{63} i = 63*64/2 = 2016.
                  t.expect(res.outputs[agg.value].has_value() &&
                               decode_i64(res.outputs[agg.value].value()) == 2016,
                           "aggregate == closed-form sum 0..63 == 2016");
                  t.expect(g.size() == 65, "64 leaves + 1 aggregate == 65 tasks");
                  t.expect(res.executed == 65, "every task (65) actually ran");
              })
        .test("serial_and_local_parallel_are_bit_identical",
              [](TestContext& t) {
                  const TaskGraph g = build_mixed_graph();

                  const std::unique_ptr<Executor> ser = nimblecas::serial_executor();
                  const std::unique_ptr<Executor> par = nimblecas::local_parallel_executor();
                  t.expect(ser->name() == "serial", "serial_executor reports name \"serial\"");
                  t.expect(par->name() == "local_parallel",
                           "local_parallel_executor reports name \"local_parallel\"");

                  const auto ser_res = ser->run(g).value();
                  const auto par_res = par->run(g).value();

                  t.expect(ser_res.outputs.size() == par_res.outputs.size(), "same output count");
                  bool identical = ser_res.outputs.size() == par_res.outputs.size();
                  for (std::size_t i = 0; identical && i < ser_res.outputs.size(); ++i) {
                      identical = results_equal(ser_res.outputs[i], par_res.outputs[i]);
                  }
                  t.expect(identical, "outputs are bit-identical between serial and local_parallel");
                  t.expect(ser_res.executed == par_res.executed,
                           "executed counts agree between serial and local_parallel");
                  t.expect(ser_res.executed == g.size(), "every task in the mixed graph succeeded");
              })
        .test("error_poisoning_propagates_and_spares_independent_branches",
              [](TestContext& t) {
                  // A -> B (fails) -> D (also depends on C, which is fine).
                  // E -> F is a fully independent branch that must still complete.
                  TaskGraph g;
                  const auto a = g.add_task(const_task(1)).value();
                  const auto b =
                      g.add_task(failing_task(MathError::domain_error), std::vector<TaskId>{a}).value();
                  const auto c = g.add_task(const_task(2)).value();
                  const auto d = g.add_task(
                                      [](std::span<const Payload> ps) -> Result<Payload> {
                                          return encode_i64(decode_i64(ps[0]) + decode_i64(ps[1]));
                                      },
                                      std::vector<TaskId>{b, c})
                                     .value();
                  const auto e = g.add_task(const_task(5)).value();
                  const auto f = g.add_task(
                                      [](std::span<const Payload> ps) -> Result<Payload> {
                                          return encode_i64(decode_i64(ps[0]) * 10);
                                      },
                                      std::vector<TaskId>{e})
                                     .value();

                  const auto exec = nimblecas::serial_executor();
                  const auto res = exec->run(g).value();

                  t.expect(res.outputs[a.value].has_value(), "A (no deps) succeeds");
                  t.expect(!res.outputs[b.value].has_value() &&
                               res.outputs[b.value].error() == MathError::domain_error,
                           "B ran and returned its own domain_error");
                  t.expect(res.outputs[c.value].has_value() && decode_i64(res.outputs[c.value].value()) == 2,
                           "C (independent of B) succeeds");
                  t.expect(!res.outputs[d.value].has_value() &&
                               res.outputs[d.value].error() == MathError::domain_error,
                           "D is poisoned: carries B's domain_error, never ran its own fn");
                  t.expect(res.outputs[e.value].has_value() && decode_i64(res.outputs[e.value].value()) == 5,
                           "E, in the independent branch, succeeds");
                  t.expect(res.outputs[f.value].has_value() && decode_i64(res.outputs[f.value].value()) == 50,
                           "F, downstream of E only, completes: the independent branch runs to completion");
                  // executed = A, B (ran and failed), C, E, F = 5; D is poisoned and not run.
                  t.expect(res.executed == 5, "executed counts A,B,C,E,F but not the poisoned D");
                  t.expect(g.size() == 6, "six tasks were issued in total");
              })
        .test("bound_literals_and_args_ordering_contract",
              [](TestContext& t) {
                  TaskRegistry reg;
                  (void)reg.register_op("test.add_literals/v1",
                                        [](std::span<const Payload> ps) -> Result<Payload> {
                                            if (ps.size() != 2) {
                                                return nimblecas::make_error<Payload>(MathError::domain_error);
                                            }
                                            return encode_i64(decode_i64(ps[0]) + decode_i64(ps[1]));
                                        });
                  (void)reg.register_op("test.count_args/v1",
                                        [](std::span<const Payload> ps) -> Result<Payload> {
                                            return encode_i64(static_cast<std::int64_t>(ps.size()));
                                        });
                  (void)reg.register_op("test.mixed_order_probe/v1",
                                        [](std::span<const Payload> ps) -> Result<Payload> {
                                            // Contract: literals first, then parent outputs in declared order.
                                            // ps[0] = lit0, ps[1] = lit1, ps[2] = dep0, ps[3] = dep1.
                                            if (ps.size() != 4) {
                                                return nimblecas::make_error<Payload>(MathError::domain_error);
                                            }
                                            const auto l0 = decode_i64(ps[0]);
                                            const auto l1 = decode_i64(ps[1]);
                                            const auto d0 = decode_i64(ps[2]);
                                            const auto d1 = decode_i64(ps[3]);
                                            return encode_i64((l0 * 1000 + l1) * 1000000 + (d0 * 1000 + d1));
                                        });

                  TaskGraph g;
                  // Level-0 named task with bound literals (data reaching worker with zero deps)
                  const auto l0_task =
                      g.add_named_task(reg, "test.add_literals/v1",
                                       std::vector<Payload>{encode_i64(10), encode_i64(25)})
                          .value();

                  // Accessor check on literals:
                  t.expect(g.literals(l0_task).size() == 2, "literals accessor returns 2 payloads");
                  t.expect(decode_i64(g.literals(l0_task)[0]) == 10, "literal[0] == 10");
                  t.expect(decode_i64(g.literals(l0_task)[1]) == 25, "literal[1] == 25");

                  // Default empty literals on tasks without explicit literals
                  const auto default_lit_task = g.add_named_task(reg, "test.count_args/v1").value();
                  t.expect(g.literals(default_lit_task).empty(),
                           "literals accessor returns empty span for default task");
                  t.expect(g.literals(default_lit_task).size() == 0,
                           "literals accessor size is 0 for default task");

                  // Plain closure task also has empty literals
                  const auto closure_task =
                      g.add_task([](std::span<const Payload> ps) -> Result<Payload> {
                          return encode_i64(static_cast<std::int64_t>(ps.size()));
                      }).value();
                  t.expect(g.literals(closure_task).empty(),
                           "literals accessor returns empty span for closure task");

                  // Parents for mixed task
                  const auto p0 = g.add_task(const_task(30)).value();
                  const auto p1 = g.add_task(const_task(40)).value();

                  // Task with BOTH bound literals and dependencies:
                  // literals = {1, 2}, deps = {p0, p1} (outputs 30, 40)
                  const auto mixed_task =
                      g.add_named_task(reg, "test.mixed_order_probe/v1",
                                       std::vector<Payload>{encode_i64(1), encode_i64(2)},
                                       std::vector<TaskId>{p0, p1})
                          .value();

                  t.expect(g.literals(mixed_task).size() == 2, "mixed task literals size == 2");
                  t.expect(g.deps(mixed_task).size() == 2, "mixed task deps size == 2");

                  const auto exec = nimblecas::serial_executor();
                  const auto res = exec->run(g).value();

                  t.expect(res.outputs[l0_task.value].has_value() &&
                               decode_i64(res.outputs[l0_task.value].value()) == 35,
                           "level-0 named task received bound literals: 10 + 25 == 35");
                  t.expect(res.outputs[default_lit_task.value].has_value() &&
                               decode_i64(res.outputs[default_lit_task.value].value()) == 0,
                           "task without literals saw 0 args");
                  t.expect(res.outputs[closure_task.value].has_value() &&
                               decode_i64(res.outputs[closure_task.value].value()) == 0,
                           "closure task without literals saw 0 args");
                  t.expect(res.outputs[mixed_task.value].has_value() &&
                               decode_i64(res.outputs[mixed_task.value].value()) == 1002030040,
                           "mixed task received literals first, then parent outputs in declared order");

                  // Error cases with literals:
                  const auto bad_op =
                      g.add_named_task(reg, "no.such/v1", std::vector<Payload>{encode_i64(1)});
                  t.expect(!bad_op.has_value() && bad_op.error() == MathError::domain_error,
                           "add_named_task with literals rejects unregistered op with domain_error");

                  const auto bad_dep =
                      g.add_named_task(reg, "test.add_literals/v1",
                                       std::vector<Payload>{encode_i64(1), encode_i64(2)},
                                       std::vector<TaskId>{TaskId{999}});
                  t.expect(!bad_dep.has_value() && bad_dep.error() == MathError::domain_error,
                           "add_named_task with literals rejects invalid dep with domain_error");
              })
        .test("bound_literals_serial_and_local_parallel_parity",
              [](TestContext& t) {
                  TaskRegistry reg;
                  (void)reg.register_op("test.scale/v1",
                                        [](std::span<const Payload> ps) -> Result<Payload> {
                                            // ps[0] = factor literal, ps[1] = input dep
                                            if (ps.size() != 2) {
                                                return nimblecas::make_error<Payload>(MathError::domain_error);
                                            }
                                            return encode_i64(decode_i64(ps[0]) * decode_i64(ps[1]));
                                        });
                  (void)reg.register_op("test.const_lit/v1",
                                        [](std::span<const Payload> ps) -> Result<Payload> {
                                            if (ps.size() != 1) {
                                                return nimblecas::make_error<Payload>(MathError::domain_error);
                                            }
                                            return ps[0];
                                        });

                  TaskGraph g;
                  // Level-0 tasks with bound literals
                  std::vector<TaskId> leaves;
                  leaves.reserve(64);
                  for (std::int64_t i = 0; i < 64; ++i) {
                      leaves.push_back(
                          g.add_named_task(reg, "test.const_lit/v1",
                                           std::vector<Payload>{encode_i64(i)})
                              .value());
                  }

                  // Level-1 tasks scaling each leaf by literal factor 3
                  std::vector<TaskId> scaled;
                  scaled.reserve(64);
                  for (std::size_t i = 0; i < 64; ++i) {
                      scaled.push_back(
                          g.add_named_task(reg, "test.scale/v1",
                                           std::vector<Payload>{encode_i64(3)},
                                           std::vector<TaskId>{leaves[i]})
                              .value());
                  }

                  const auto ser = nimblecas::serial_executor();
                  const auto par = nimblecas::local_parallel_executor();

                  const auto ser_res = ser->run(g).value();
                  const auto par_res = par->run(g).value();

                  t.expect(ser_res.outputs.size() == par_res.outputs.size(), "same output count");
                  bool identical = ser_res.outputs.size() == par_res.outputs.size();
                  for (std::size_t i = 0; identical && i < ser_res.outputs.size(); ++i) {
                      identical = results_equal(ser_res.outputs[i], par_res.outputs[i]);
                  }
                  t.expect(identical, "outputs bit-identical between serial and local_parallel");
                  t.expect(ser_res.executed == par_res.executed, "executed counts agree");
                  t.expect(ser_res.executed == g.size(), "every task executed successfully");
                  t.expect(decode_i64(ser_res.outputs[scaled[10].value].value()) == 30,
                           "scaled[10] == 10 * 3 == 30");
              })
        .run();
}
