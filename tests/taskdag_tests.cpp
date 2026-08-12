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
        .run();
}
