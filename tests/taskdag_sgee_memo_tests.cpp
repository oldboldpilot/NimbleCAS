// Tests for nimblecas.taskdag_sgee: content-addressed memoization and task deduplication integration.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.taskdag;
import nimblecas.taskdag_sgee;
import nimblecas.memo_dist;
import nimblecas.testing;

using nimblecas::ContentKey;
using nimblecas::content_key;
using nimblecas::DistributedMemo;
using nimblecas::Executor;
using nimblecas::FakeBrokerPort;
using nimblecas::InMemoryResultChannel;
using nimblecas::InProcessMemo;
using nimblecas::is_memoizable_status;
using nimblecas::MathError;
using nimblecas::MemoStats;
using nimblecas::Payload;
using nimblecas::Result;
using nimblecas::SgeeDistributedExecutor;
using nimblecas::SgeeExecutorConfig;
using nimblecas::TaskGraph;
using nimblecas::TaskId;
using nimblecas::TaskRegistry;
using nimblecas::TaskRunResult;
using nimblecas::sgee_bridge::decode_result;
using nimblecas::sgee_bridge::encode_result;
using nimblecas::sgee_bridge::encode_task;
using nimblecas::sgee_bridge::ResultEnvelope;
using nimblecas::sgee_bridge::TaskEnvelope;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto encode_i64(std::int64_t v) -> Payload {
    const auto bytes = std::bit_cast<std::array<std::byte, sizeof(std::int64_t)>>(v);
    return Payload(bytes.begin(), bytes.end());
}

[[nodiscard]] auto decode_i64(std::span<const std::byte> p) -> std::int64_t {
    std::array<std::byte, sizeof(std::int64_t)> bytes{};
    std::ranges::copy(p, bytes.begin());
    return std::bit_cast<std::int64_t>(bytes);
}

[[nodiscard]] auto results_equal(const Result<Payload>& a, const Result<Payload>& b) -> bool {
    if (a.has_value() != b.has_value()) {
        return false;
    }
    return a.has_value() ? (*a == *b) : (a.error() == b.error());
}

[[nodiscard]] auto outputs_bit_identical(std::span<const Result<Payload>> a,
                                         std::span<const Result<Payload>> b) -> bool {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!results_equal(a[i], b[i])) {
            return false;
        }
    }
    return true;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.taskdag_sgee_memo")
        // -------------------------------------------------------------------
        // 1. memo_off_by_default_changes_nothing
        // -------------------------------------------------------------------
        .test("memo_off_by_default_changes_nothing",
              [](TestContext& t) {
                  TaskRegistry reg;
                  (void)reg.register_op("op.c10/v1", [](auto) -> Result<Payload> { return encode_i64(10); });
                  (void)reg.register_op("op.mul2/v1", [](auto ps) -> Result<Payload> {
                      return encode_i64(decode_i64(ps[0]) * 2);
                  });
                  (void)reg.register_op("op.add/v1", [](auto ps) -> Result<Payload> {
                      return encode_i64(decode_i64(ps[0]) + decode_i64(ps[1]));
                  });

                  TaskGraph g;
                  const auto a1 = g.add_named_task(reg, "op.c10/v1").value();
                  const auto a2 = g.add_named_task(reg, "op.c10/v1").value();
                  const auto b1 = g.add_named_task(reg, "op.mul2/v1", std::vector<TaskId>{a1}).value();
                  const auto b2 = g.add_named_task(reg, "op.mul2/v1", std::vector<TaskId>{a1}).value();
                  const auto c = g.add_named_task(reg, "op.add/v1", std::vector<TaskId>{b1, b2}).value();

                  // Reference run on serial_executor
                  const auto ser_exec = nimblecas::serial_executor();
                  const auto ser_res = ser_exec->run(g).value();
                  t.expect_eq(ser_res.executed, std::size_t{5}, "serial executor runs all 5 tasks");

                  // SGEE distributed run with default configuration
                  FakeBrokerPort port;
                  InMemoryResultChannel results;
                  SgeeExecutorConfig cfg;
                  cfg.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1);

                  t.expect(cfg.memo == nullptr, "default config memo is nullptr");
                  t.expect(!cfg.dedup_identical_tasks, "default config dedup_identical_tasks is false");

                  SgeeDistributedExecutor dist_exec(cfg, port, results);
                  const auto dist_res = dist_exec.run(g).value();

                  t.expect(outputs_bit_identical(dist_res.outputs, ser_res.outputs),
                           "outputs are BIT-IDENTICAL between default distributed executor and serial reference");
                  t.expect_eq(dist_res.executed, std::size_t{5},
                              "executed count equals total number of non-poisoned tasks (nothing skipped)");
                  t.expect_eq(dist_res.executed, ser_res.executed, "executed count matches serial reference (5)");

                  t.expect_eq(decode_i64(dist_res.outputs[a1.value].value()), std::int64_t{10}, "a1 == 10");
                  t.expect_eq(decode_i64(dist_res.outputs[a2.value].value()), std::int64_t{10}, "a2 == 10");
                  t.expect_eq(decode_i64(dist_res.outputs[b1.value].value()), std::int64_t{20}, "b1 == 20");
                  t.expect_eq(decode_i64(dist_res.outputs[b2.value].value()), std::int64_t{20}, "b2 == 20");
                  t.expect_eq(decode_i64(dist_res.outputs[c.value].value()), std::int64_t{40}, "c == 40");
              })

        // -------------------------------------------------------------------
        // 2. dedup_collapses_identical_tasks_same_level
        // -------------------------------------------------------------------
        .test("dedup_collapses_identical_tasks_same_level",
              [](TestContext& t) {
                  TaskRegistry reg;
                  (void)reg.register_op("op.c42/v1", [](auto) -> Result<Payload> { return encode_i64(42); });
                  (void)reg.register_op("op.c100/v1", [](auto) -> Result<Payload> { return encode_i64(100); });
                  (void)reg.register_op("op.add/v1", [](auto ps) -> Result<Payload> {
                      return encode_i64(decode_i64(ps[0]) + decode_i64(ps[1]));
                  });

                  // Level 0: 4 identical tasks (t0..t3) + 1 distinct task (t_other)
                  // Level 1: 3 identical tasks (u0..u2)
                  // Total tasks = 8, distinct computations = 3 (t0, t_other, u0)
                  TaskGraph g;
                  const auto t0 = g.add_named_task(reg, "op.c42/v1").value();
                  const auto t1 = g.add_named_task(reg, "op.c42/v1").value();
                  const auto t2 = g.add_named_task(reg, "op.c42/v1").value();
                  const auto t3 = g.add_named_task(reg, "op.c42/v1").value();
                  const auto t_other = g.add_named_task(reg, "op.c100/v1").value();

                  const auto u0 = g.add_named_task(reg, "op.add/v1", std::vector<TaskId>{t0, t_other}).value();
                  const auto u1 = g.add_named_task(reg, "op.add/v1", std::vector<TaskId>{t0, t_other}).value();
                  const auto u2 = g.add_named_task(reg, "op.add/v1", std::vector<TaskId>{t0, t_other}).value();

                  // (a) Plain run (dedup off)
                  FakeBrokerPort port_plain;
                  InMemoryResultChannel chan_plain;
                  SgeeExecutorConfig cfg_plain;
                  cfg_plain.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1);
                  SgeeDistributedExecutor exec_plain(cfg_plain, port_plain, chan_plain);
                  const auto res_plain = exec_plain.run(g).value();
                  t.expect_eq(res_plain.executed, std::size_t{8}, "plain run executes all 8 tasks");

                  // (b) Dedup on
                  FakeBrokerPort port_dedup;
                  InMemoryResultChannel chan_dedup;
                  SgeeExecutorConfig cfg_dedup;
                  cfg_dedup.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1)
                           .with_dedup_identical_tasks(true);
                  SgeeDistributedExecutor exec_dedup(cfg_dedup, port_dedup, chan_dedup);
                  const auto res_dedup = exec_dedup.run(g).value();

                  // (c) Fresh InProcessMemo attached
                  InProcessMemo memo(8, 1000, 1024 * 1024);
                  FakeBrokerPort port_memo;
                  InMemoryResultChannel chan_memo;
                  SgeeExecutorConfig cfg_memo;
                  cfg_memo.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1)
                          .with_memo(memo);
                  SgeeDistributedExecutor exec_memo(cfg_memo, port_memo, chan_memo);
                  const auto res_memo = exec_memo.run(g).value();

                  // Central invariant: BIT-IDENTICAL outputs across all three modes
                  t.expect(outputs_bit_identical(res_dedup.outputs, res_plain.outputs),
                           "dedup outputs are BIT-IDENTICAL to plain run");
                  t.expect(outputs_bit_identical(res_memo.outputs, res_plain.outputs),
                           "memo outputs are BIT-IDENTICAL to plain run");

                  // In-flight alias path collapses duplicates within each level
                  t.expect_eq(res_dedup.executed, std::size_t{3},
                              "dedup executed count drops strictly to 3 distinct tasks");
                  t.expect(res_dedup.executed < res_plain.executed, "executed count strictly decreases");

                  // Exact value assertions
                  t.expect_eq(decode_i64(res_dedup.outputs[t0.value].value()), std::int64_t{42}, "t0 == 42");
                  t.expect_eq(decode_i64(res_dedup.outputs[t1.value].value()), std::int64_t{42}, "t1 == 42");
                  t.expect_eq(decode_i64(res_dedup.outputs[t2.value].value()), std::int64_t{42}, "t2 == 42");
                  t.expect_eq(decode_i64(res_dedup.outputs[t3.value].value()), std::int64_t{42}, "t3 == 42");
                  t.expect_eq(decode_i64(res_dedup.outputs[t_other.value].value()), std::int64_t{100}, "t_other == 100");
                  t.expect_eq(decode_i64(res_dedup.outputs[u0.value].value()), std::int64_t{142}, "u0 == 142");
                  t.expect_eq(decode_i64(res_dedup.outputs[u1.value].value()), std::int64_t{142}, "u1 == 142");
                  t.expect_eq(decode_i64(res_dedup.outputs[u2.value].value()), std::int64_t{142}, "u2 == 142");
              })

        // -------------------------------------------------------------------
        // 3. dedup_collapses_identical_tasks_across_levels
        // -------------------------------------------------------------------
        .test("dedup_collapses_identical_tasks_across_levels",
              [](TestContext& t) {
                  TaskRegistry reg;
                  (void)reg.register_op("op.c7/v1", [](auto) -> Result<Payload> { return encode_i64(7); });
                  (void)reg.register_op("op.c100/v1", [](auto) -> Result<Payload> { return encode_i64(100); });
                  (void)reg.register_op("op.echo7/v1", [](auto) -> Result<Payload> { return encode_i64(7); });
                  (void)reg.register_op("op.mul2/v1", [](auto ps) -> Result<Payload> {
                      return encode_i64(decode_i64(ps[0]) * 2);
                  });
                  (void)reg.register_op("op.add/v1", [](auto ps) -> Result<Payload> {
                      return encode_i64(decode_i64(ps[0]) + decode_i64(ps[1]));
                  });

                  // Level 0: t_base (returns 7), t_seed (returns 100)
                  // Level 1: t_early (mul2 with dep t_base -> receives 7, returns 14)
                  //          t_echo7 (echo7 with dep t_seed -> receives 100, returns 7)
                  // Level 2: t_late (mul2 with dep t_echo7 -> receives 7, returns 14)
                  //          t_late has identical op_id and input arg (7) as t_early from level 1!
                  // Level 3: t_comb (add with deps t_early, t_late -> returns 28)
                  // Total tasks = 6. Distinct computations = 5.
                  TaskGraph g;
                  const auto t_base = g.add_named_task(reg, "op.c7/v1").value();
                  const auto t_seed = g.add_named_task(reg, "op.c100/v1").value();
                  const auto t_early = g.add_named_task(reg, "op.mul2/v1", std::vector<TaskId>{t_base}).value();
                  const auto t_echo7 = g.add_named_task(reg, "op.echo7/v1", std::vector<TaskId>{t_seed}).value();
                  const auto t_late = g.add_named_task(reg, "op.mul2/v1", std::vector<TaskId>{t_echo7}).value();
                  const auto t_comb = g.add_named_task(reg, "op.add/v1", std::vector<TaskId>{t_early, t_late}).value();

                  t.expect_eq(g.depth(t_early), std::size_t{1}, "t_early is in level 1");
                  t.expect_eq(g.depth(t_late), std::size_t{2}, "t_late is in level 2 (across levels)");

                  // (a) Plain run
                  FakeBrokerPort port_plain;
                  InMemoryResultChannel chan_plain;
                  SgeeExecutorConfig cfg_plain;
                  cfg_plain.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1);
                  SgeeDistributedExecutor exec_plain(cfg_plain, port_plain, chan_plain);
                  const auto res_plain = exec_plain.run(g).value();
                  t.expect_eq(res_plain.executed, std::size_t{6}, "plain run executes all 6 tasks");

                  // (b) Dedup on
                  FakeBrokerPort port_dedup;
                  InMemoryResultChannel chan_dedup;
                  SgeeExecutorConfig cfg_dedup;
                  cfg_dedup.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1)
                           .with_dedup_identical_tasks(true);
                  SgeeDistributedExecutor exec_dedup(cfg_dedup, port_dedup, chan_dedup);
                  const auto res_dedup = exec_dedup.run(g).value();

                  // (c) Fresh InProcessMemo attached
                  InProcessMemo memo(8, 1000, 1024 * 1024);
                  FakeBrokerPort port_memo;
                  InMemoryResultChannel chan_memo;
                  SgeeExecutorConfig cfg_memo;
                  cfg_memo.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1)
                          .with_memo(memo);
                  SgeeDistributedExecutor exec_memo(cfg_memo, port_memo, chan_memo);
                  const auto res_memo = exec_memo.run(g).value();

                  // Central invariant: BIT-IDENTICAL outputs across all three modes
                  t.expect(outputs_bit_identical(res_dedup.outputs, res_plain.outputs),
                           "dedup across levels outputs are BIT-IDENTICAL to plain run");
                  t.expect(outputs_bit_identical(res_memo.outputs, res_plain.outputs),
                           "memo outputs are BIT-IDENTICAL to plain run");

                  // Settled alias path collapses t_late against drained level 1 twin t_early
                  t.expect_eq(res_dedup.executed, std::size_t{5},
                              "dedup executed count drops by 1 across levels (executed == 5)");
                  t.expect(res_dedup.executed < res_plain.executed, "executed count strictly decreases");

                  t.expect_eq(decode_i64(res_dedup.outputs[t_early.value].value()), std::int64_t{14}, "t_early == 14");
                  t.expect_eq(decode_i64(res_dedup.outputs[t_late.value].value()), std::int64_t{14}, "t_late == 14");
                  t.expect_eq(decode_i64(res_dedup.outputs[t_comb.value].value()), std::int64_t{28}, "t_comb == 28");
              })

        // -------------------------------------------------------------------
        // 4. dedup_preserves_distinct_tasks_that_differ_by_one_byte
        // -------------------------------------------------------------------
        .test("dedup_preserves_distinct_tasks_that_differ_by_one_byte",
              [](TestContext& t) {
                  TaskRegistry reg;
                  (void)reg.register_op("op.echo_lit/v1", [](std::span<const Payload> ps) -> Result<Payload> {
                      if (ps.empty()) {
                          return nimblecas::make_error<Payload>(MathError::domain_error);
                      }
                      return ps[0];
                  });
                  (void)reg.register_op("op.concat/v1", [](std::span<const Payload> ps) -> Result<Payload> {
                      if (ps.size() != 2) {
                          return nimblecas::make_error<Payload>(MathError::domain_error);
                      }
                      Payload out = ps[0];
                      out.insert(out.end(), ps[1].begin(), ps[1].end());
                      return out;
                  });

                  // Two tasks with literal arguments differing by exactly one byte
                  const Payload lit1{std::byte{0x41}};
                  const Payload lit2{std::byte{0x42}};

                  TaskGraph g;
                  const auto t1 = g.add_named_task(reg, "op.echo_lit/v1", std::vector<Payload>{lit1}).value();
                  const auto t2 = g.add_named_task(reg, "op.echo_lit/v1", std::vector<Payload>{lit2}).value();
                  const auto t3 = g.add_named_task(reg, "op.concat/v1", std::vector<TaskId>{t1, t2}).value();

                  // Plain serial reference
                  const auto ser_res = nimblecas::serial_executor()->run(g).value();
                  t.expect_eq(ser_res.executed, std::size_t{3}, "serial executes 3 tasks");

                  // Dedup on
                  FakeBrokerPort port;
                  InMemoryResultChannel chan;
                  SgeeExecutorConfig cfg;
                  cfg.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1)
                     .with_dedup_identical_tasks(true);
                  SgeeDistributedExecutor exec(cfg, port, chan);
                  const auto res = exec.run(g).value();

                  t.expect(outputs_bit_identical(res.outputs, ser_res.outputs),
                           "outputs are BIT-IDENTICAL to serial reference");
                  t.expect_eq(res.executed, std::size_t{3},
                              "executed count did NOT drop (both distinct tasks ran)");
                  t.expect(res.outputs[t1.value].value() != res.outputs[t2.value].value(),
                           "outputs of t1 and t2 differ");
                  t.expect_eq(res.outputs[t1.value].value().size(), std::size_t{1}, "t1 output size is 1");
                  t.expect_eq(res.outputs[t2.value].value().size(), std::size_t{1}, "t2 output size is 1");
                  t.expect_eq(static_cast<std::uint8_t>(res.outputs[t1.value].value()[0]), std::uint8_t{0x41},
                              "t1 output byte is 0x41");
                  t.expect_eq(static_cast<std::uint8_t>(res.outputs[t2.value].value()[0]), std::uint8_t{0x42},
                              "t2 output byte is 0x42");
                  const Payload expected_concat{std::byte{0x41}, std::byte{0x42}};
                  t.expect(res.outputs[t3.value].value() == expected_concat, "t3 concat output matches {0x41, 0x42}");
              })

        // -------------------------------------------------------------------
        // 5. memo_hit_across_runs_is_bit_identical
        // -------------------------------------------------------------------
        .test("memo_hit_across_runs_is_bit_identical",
              [](TestContext& t) {
                  TaskRegistry reg;
                  (void)reg.register_op("op.const7/v1", [](auto) -> Result<Payload> { return encode_i64(7); });
                  (void)reg.register_op("op.mul2/v1", [](auto ps) -> Result<Payload> {
                      return encode_i64(decode_i64(ps[0]) * 2);
                  });
                  (void)reg.register_op("op.add3/v1", [](auto ps) -> Result<Payload> {
                      return encode_i64(decode_i64(ps[0]) + 3);
                  });
                  (void)reg.register_op("op.add/v1", [](auto ps) -> Result<Payload> {
                      return encode_i64(decode_i64(ps[0]) + decode_i64(ps[1]));
                  });
                  (void)reg.register_op("op.probe/v1", [](auto ps) -> Result<Payload> {
                      return encode_i64(decode_i64(ps[0]) * 1000 + decode_i64(ps[1]));
                  });

                  TaskGraph g;
                  const auto a = g.add_named_task(reg, "op.const7/v1").value();
                  const auto b = g.add_named_task(reg, "op.mul2/v1", std::vector<TaskId>{a}).value();
                  const auto c = g.add_named_task(reg, "op.add3/v1", std::vector<TaskId>{a}).value();
                  const auto d = g.add_named_task(reg, "op.add/v1", std::vector<TaskId>{b, c}).value();
                  const auto probe = g.add_named_task(reg, "op.probe/v1", std::vector<TaskId>{b, c}).value();

                  InProcessMemo memo(8, 1000, 1024 * 1024);

                  // Run 1: cold cache -> executes all tasks and populates memo
                  FakeBrokerPort port1;
                  InMemoryResultChannel chan1;
                  SgeeExecutorConfig cfg1;
                  cfg1.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1)
                      .with_memo(memo);
                  SgeeDistributedExecutor exec1(cfg1, port1, chan1);
                  const auto res1 = exec1.run(g).value();

                  t.expect_eq(res1.executed, std::size_t{5}, "run 1 executes all 5 tasks");
                  const auto stats1 = memo.stats();
                  t.expect_eq(stats1.hits, std::uint64_t{0}, "run 1: 0 hits");
                  t.expect_eq(stats1.misses, std::uint64_t{5}, "run 1: 5 misses");
                  t.expect_eq(stats1.publishes, std::uint64_t{5}, "run 1: 5 publishes");
                  t.expect_eq(stats1.hits + stats1.misses, std::uint64_t{5}, "total lookups in run 1 == 5");

                  // Run 2: warm cache -> identical outputs, 0 tasks executed
                  FakeBrokerPort port2;
                  InMemoryResultChannel chan2;
                  SgeeExecutorConfig cfg2;
                  cfg2.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1)
                      .with_memo(memo);
                  SgeeDistributedExecutor exec2(cfg2, port2, chan2);
                  const auto res2 = exec2.run(g).value();

                  t.expect(outputs_bit_identical(res2.outputs, res1.outputs),
                           "run 2 outputs are BIT-IDENTICAL to run 1");
                  t.expect_eq(res2.executed, std::size_t{0},
                              "run 2 executed count is ZERO (everything served from table)");

                  const auto stats2 = memo.stats();
                  t.expect_eq(stats2.hits, std::uint64_t{5}, "run 2: hits grew by 5 (total hits == 5)");
                  t.expect_eq(stats2.misses, std::uint64_t{5}, "run 2: misses unchanged (total misses == 5)");
                  t.expect_eq(stats2.hits + stats2.misses, std::uint64_t{10},
                              "stats.hits + stats.misses equals total lookups made (10)");
                  t.expect_eq(stats2.publishes, std::uint64_t{5}, "publishes unchanged at 5");
                  t.expect_eq(stats2.key_mismatches, std::uint64_t{0}, "zero key mismatches");

                  t.expect_eq(decode_i64(res2.outputs[a.value].value()), std::int64_t{7}, "a == 7");
                  t.expect_eq(decode_i64(res2.outputs[b.value].value()), std::int64_t{14}, "b == 14");
                  t.expect_eq(decode_i64(res2.outputs[c.value].value()), std::int64_t{10}, "c == 10");
                  t.expect_eq(decode_i64(res2.outputs[d.value].value()), std::int64_t{24}, "d == 24");
                  t.expect_eq(decode_i64(res2.outputs[probe.value].value()), std::int64_t{14010}, "probe == 14010");
              })

        // -------------------------------------------------------------------
        // 6. memo_does_not_cache_a_failure_that_is_not_the_inputs_fault
        // -------------------------------------------------------------------
        .test("memo_does_not_cache_a_failure_that_is_not_the_inputs_fault",
              [](TestContext& t) {
                  TaskRegistry reg;
                  (void)reg.register_op("op.fail_div0/v1", [](auto) -> Result<Payload> {
                      return nimblecas::make_error<Payload>(MathError::division_by_zero);
                  });
                  (void)reg.register_op("op.c5/v1", [](auto) -> Result<Payload> { return encode_i64(5); });

                  TaskGraph g;
                  const auto t_err = g.add_named_task(reg, "op.fail_div0/v1").value();
                  const auto t_ok = g.add_named_task(reg, "op.c5/v1").value();

                  InProcessMemo memo(8, 1000, 1024 * 1024);

                  // Run 1: executes deterministic failure and success, caches both
                  FakeBrokerPort port1;
                  InMemoryResultChannel chan1;
                  SgeeExecutorConfig cfg1;
                  cfg1.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1).with_memo(memo);
                  SgeeDistributedExecutor exec1(cfg1, port1, chan1);
                  const auto res1 = exec1.run(g).value();

                  t.expect_eq(res1.executed, std::size_t{2}, "both tasks executed in run 1");
                  t.expect(!res1.outputs[t_err.value].has_value() &&
                           res1.outputs[t_err.value].error() == MathError::division_by_zero,
                           "t_err produced division_by_zero in run 1");
                  t.expect(res1.outputs[t_ok.value].has_value() &&
                           decode_i64(res1.outputs[t_ok.value].value()) == 5,
                           "t_ok produced 5 in run 1");

                  // Run 2: cached deterministic math_error is reproduced identically from memo
                  FakeBrokerPort port2;
                  InMemoryResultChannel chan2;
                  SgeeExecutorConfig cfg2;
                  cfg2.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1).with_memo(memo);
                  SgeeDistributedExecutor exec2(cfg2, port2, chan2);
                  const auto res2 = exec2.run(g).value();

                  t.expect(outputs_bit_identical(res2.outputs, res1.outputs),
                           "run 2 outputs are BIT-IDENTICAL to run 1 (including cached math_error)");
                  t.expect_eq(res2.executed, std::size_t{0}, "run 2 executed count is 0");
                  t.expect(!res2.outputs[t_err.value].has_value() &&
                           res2.outputs[t_err.value].error() == MathError::division_by_zero,
                           "t_err error reproduced identically in slot 0");
                  t.expect_eq(memo.stats().hits, std::uint64_t{2}, "2 memo hits recorded on run 2");

                  // Directly assert via is_memoizable_status that status 2 (bridge_error) is refused
                  t.expect(is_memoizable_status(0), "status 0 (ok) is memoizable");
                  t.expect(is_memoizable_status(1), "status 1 (math_error) is memoizable");
                  t.expect(!is_memoizable_status(2), "status 2 (bridge_error) is REFUSED by is_memoizable_status");
                  t.expect(!is_memoizable_status(3), "status 3 is refused");
                  t.expect(!is_memoizable_status(-1), "status -1 is refused");
              })

        // -------------------------------------------------------------------
        // 7. poisoning_is_unchanged_by_memoization
        // -------------------------------------------------------------------
        .test("poisoning_is_unchanged_by_memoization",
              [](TestContext& t) {
                  TaskRegistry reg;
                  (void)reg.register_op("op.c10/v1", [](auto) -> Result<Payload> { return encode_i64(10); });
                  (void)reg.register_op("op.fail_dom/v1", [](auto) -> Result<Payload> {
                      return nimblecas::make_error<Payload>(MathError::domain_error);
                  });
                  (void)reg.register_op("op.mul2/v1", [](auto ps) -> Result<Payload> {
                      return encode_i64(decode_i64(ps[0]) * 2);
                  });
                  (void)reg.register_op("op.add/v1", [](auto ps) -> Result<Payload> {
                      return encode_i64(decode_i64(ps[0]) + decode_i64(ps[1]));
                  });
                  (void)reg.register_op("op.c5/v1", [](auto) -> Result<Payload> { return encode_i64(5); });
                  (void)reg.register_op("op.mul10/v1", [](auto ps) -> Result<Payload> {
                      return encode_i64(decode_i64(ps[0]) * 10);
                  });

                  // Graph with failing tasks, duplicates, and poisoned descendants:
                  // a_ok: const 10
                  // b_fail: fail_dom (dep a_ok)
                  // b_fail_dup: fail_dom (dep a_ok) [duplicate of b_fail]
                  // c_poison1: mul2 (dep b_fail) [poisoned]
                  // c_poison2: mul2 (dep b_fail) [poisoned duplicate]
                  // c_poison3: add (deps b_fail, a_ok) [poisoned]
                  // d_ind: const 5 (independent)
                  // e_ind: mul10 (dep d_ind -> returns 50)
                  // f_poison: add (deps c_poison1, e_ind) [transitively poisoned]
                  TaskGraph g;
                  const auto a_ok = g.add_named_task(reg, "op.c10/v1").value();
                  const auto b_fail = g.add_named_task(reg, "op.fail_dom/v1", std::vector<TaskId>{a_ok}).value();
                  const auto b_fail_dup = g.add_named_task(reg, "op.fail_dom/v1", std::vector<TaskId>{a_ok}).value();
                  const auto c_poison1 = g.add_named_task(reg, "op.mul2/v1", std::vector<TaskId>{b_fail}).value();
                  const auto c_poison2 = g.add_named_task(reg, "op.mul2/v1", std::vector<TaskId>{b_fail}).value();
                  const auto c_poison3 = g.add_named_task(reg, "op.add/v1", std::vector<TaskId>{b_fail, a_ok}).value();
                  const auto d_ind = g.add_named_task(reg, "op.c5/v1").value();
                  const auto e_ind = g.add_named_task(reg, "op.mul10/v1", std::vector<TaskId>{d_ind}).value();
                  const auto f_poison = g.add_named_task(reg, "op.add/v1", std::vector<TaskId>{c_poison1, e_ind}).value();

                  // Reference serial run
                  const auto ser_res = nimblecas::serial_executor()->run(g).value();

                  // Mode (a): plain distributed run
                  FakeBrokerPort port_plain;
                  InMemoryResultChannel chan_plain;
                  SgeeExecutorConfig cfg_plain;
                  cfg_plain.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1);
                  SgeeDistributedExecutor exec_plain(cfg_plain, port_plain, chan_plain);
                  const auto res_plain = exec_plain.run(g).value();

                  // Mode (b): dedup on
                  FakeBrokerPort port_dedup;
                  InMemoryResultChannel chan_dedup;
                  SgeeExecutorConfig cfg_dedup;
                  cfg_dedup.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1)
                           .with_dedup_identical_tasks(true);
                  SgeeDistributedExecutor exec_dedup(cfg_dedup, port_dedup, chan_dedup);
                  const auto res_dedup = exec_dedup.run(g).value();

                  // Mode (c): fresh InProcessMemo attached
                  InProcessMemo memo(8, 1000, 1024 * 1024);
                  FakeBrokerPort port_memo;
                  InMemoryResultChannel chan_memo;
                  SgeeExecutorConfig cfg_memo;
                  cfg_memo.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1)
                          .with_memo(memo);
                  SgeeDistributedExecutor exec_memo(cfg_memo, port_memo, chan_memo);
                  const auto res_memo = exec_memo.run(g).value();

                  // All three modes + serial must produce BIT-IDENTICAL outputs and identical error poisoning
                  t.expect(outputs_bit_identical(res_plain.outputs, ser_res.outputs),
                           "plain distributed outputs are BIT-IDENTICAL to serial");
                  t.expect(outputs_bit_identical(res_dedup.outputs, ser_res.outputs),
                           "dedup outputs are BIT-IDENTICAL to serial");
                  t.expect(outputs_bit_identical(res_memo.outputs, ser_res.outputs),
                           "memo outputs are BIT-IDENTICAL to serial");

                  // Exact error assertions
                  t.expect(!res_dedup.outputs[b_fail.value].has_value() &&
                           res_dedup.outputs[b_fail.value].error() == MathError::domain_error,
                           "b_fail produced domain_error");
                  t.expect(!res_dedup.outputs[b_fail_dup.value].has_value() &&
                           res_dedup.outputs[b_fail_dup.value].error() == MathError::domain_error,
                           "b_fail_dup produced domain_error");
                  t.expect(!res_dedup.outputs[c_poison1.value].has_value() &&
                           res_dedup.outputs[c_poison1.value].error() == MathError::domain_error,
                           "c_poison1 poisoned with domain_error");
                  t.expect(!res_dedup.outputs[c_poison2.value].has_value() &&
                           res_dedup.outputs[c_poison2.value].error() == MathError::domain_error,
                           "c_poison2 poisoned with domain_error");
                  t.expect(!res_dedup.outputs[c_poison3.value].has_value() &&
                           res_dedup.outputs[c_poison3.value].error() == MathError::domain_error,
                           "c_poison3 poisoned with domain_error");
                  t.expect(!res_dedup.outputs[f_poison.value].has_value() &&
                           res_dedup.outputs[f_poison.value].error() == MathError::domain_error,
                           "f_poison transitively poisoned with domain_error");

                  // Independent branch ran and succeeded
                  t.expect(res_dedup.outputs[a_ok.value].has_value() &&
                           decode_i64(res_dedup.outputs[a_ok.value].value()) == 10, "a_ok == 10");
                  t.expect(res_dedup.outputs[d_ind.value].has_value() &&
                           decode_i64(res_dedup.outputs[d_ind.value].value()) == 5, "d_ind == 5");
                  t.expect(res_dedup.outputs[e_ind.value].has_value() &&
                           decode_i64(res_dedup.outputs[e_ind.value].value()) == 50, "e_ind == 50");

                  // Execution counts:
                  // Plain runs 5 tasks: a_ok, b_fail, b_fail_dup, d_ind, e_ind (poisoned tasks not executed)
                  t.expect_eq(res_plain.executed, std::size_t{5}, "plain executed count is 5");
                  // Dedup collapses b_fail_dup onto b_fail, so executed is 4
                  t.expect_eq(res_dedup.executed, std::size_t{4},
                              "dedup executed count drops to 4 (collapsed duplicate failing task)");
                  t.expect(res_dedup.executed < res_plain.executed, "dedup executed count < plain executed count");
              })

        // -------------------------------------------------------------------
        // 8. memo_with_forged_collision_never_serves_the_wrong_answer
        // -------------------------------------------------------------------
        .test("memo_with_forged_collision_never_serves_the_wrong_answer",
              [](TestContext& t) {
                  TaskRegistry reg;
                  (void)reg.register_op("op.c7/v1", [](auto) -> Result<Payload> { return encode_i64(7); });

                  TaskGraph g;
                  const auto t0 = g.add_named_task(reg, "op.c7/v1").value();

                  // Determine exact TaskEnvelope bytes that the coordinator will encode for t0
                  const TaskEnvelope real_env{
                      .registry_fp = reg.fingerprint(),
                      .op_id = "op.c7/v1",
                      .args = {}
                  };
                  const Payload real_key_bytes = encode_task(real_env).value();
                  const ContentKey real_ck = content_key(real_key_bytes);

                  // Pre-poison the memo under the EXACT ContentKey of t0, but with a DIFFERENT full key
                  Payload forged_key_bytes = real_key_bytes;
                  forged_key_bytes.push_back(std::byte{0xFF});  // append a byte

                  const ResultEnvelope bogus_res_env{
                      .status = ResultEnvelope::Status::ok,
                      .math_err = MathError::division_by_zero,
                      .seconds = 0.0,
                      .bytes = encode_i64(999999)  // bogus value
                  };
                  const Payload bogus_val_bytes = encode_result(bogus_res_env).value();

                  InProcessMemo memo(8, 1000, 1024 * 1024);
                  const auto pub_res = memo.publish(real_ck, forged_key_bytes, bogus_val_bytes);
                  t.expect(pub_res.has_value(), "pre-poison publish succeeds");
                  t.expect_eq(memo.stats().publishes, std::uint64_t{1}, "1 publish in memo before run");
                  t.expect_eq(memo.stats().key_mismatches, std::uint64_t{0}, "0 key mismatches before run");

                  // Run the graph with the pre-poisoned memo attached
                  FakeBrokerPort port;
                  InMemoryResultChannel chan;
                  SgeeExecutorConfig cfg;
                  cfg.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1).with_memo(memo);
                  SgeeDistributedExecutor exec(cfg, port, chan);
                  const auto res = exec.run(g).value();

                  // Full-key verification MUST miss the forged entry and execute t0 for real
                  t.expect_eq(res.outputs.size(), std::size_t{1}, "1 output produced");
                  t.expect(res.outputs[t0.value].has_value(), "t0 succeeded");
                  t.expect_eq(decode_i64(res.outputs[t0.value].value()), std::int64_t{7},
                              "t0 output is CORRECT (7, NEVER the forged 999999)");
                  t.expect_eq(res.executed, std::size_t{1}, "task was executed for real after cache miss");

                  // Assert telemetry: key_mismatches grew, misses grew, no false hit
                  const auto stats = memo.stats();
                  t.expect(stats.key_mismatches >= 1, "stats().key_mismatches grew (evidence that exactness rule fired)");
                  t.expect_eq(stats.hits, std::uint64_t{0}, "stats().hits is ZERO (never served forged answer)");
                  t.expect_eq(stats.misses, std::uint64_t{1}, "stats().misses recorded the collision as a miss");
                  t.expect_eq(stats.publishes, std::uint64_t{2},
                              "stats().publishes is 2 (forged entry + coordinator publication of real result)");
              })

        // -------------------------------------------------------------------
        // 9. empty_output_task_is_not_confused_with_an_unresolved_slot
        // -------------------------------------------------------------------
        .test("empty_output_task_is_not_confused_with_an_unresolved_slot",
              [](TestContext& t) {
                  TaskRegistry reg;
                  // Op that legitimately returns a ZERO-BYTE payload
                  (void)reg.register_op("op.empty/v1", [](auto) -> Result<Payload> {
                      return Payload{};
                  });
                  // Op that inspects if the input payload is genuinely empty
                  (void)reg.register_op("op.check_empty/v1", [](std::span<const Payload> ps) -> Result<Payload> {
                      if (ps.size() != 1) {
                          return nimblecas::make_error<Payload>(MathError::domain_error);
                      }
                      return ps[0].empty() ? encode_i64(1) : encode_i64(0);
                  });

                  // Level 0: 3 duplicate tasks legitimately returning empty payloads
                  // Level 1: 3 tasks consuming the respective empty outputs
                  TaskGraph g;
                  const auto e0 = g.add_named_task(reg, "op.empty/v1").value();
                  const auto e1 = g.add_named_task(reg, "op.empty/v1").value();
                  const auto e2 = g.add_named_task(reg, "op.empty/v1").value();

                  const auto c0 = g.add_named_task(reg, "op.check_empty/v1", std::vector<TaskId>{e0}).value();
                  const auto c1 = g.add_named_task(reg, "op.check_empty/v1", std::vector<TaskId>{e1}).value();
                  const auto c2 = g.add_named_task(reg, "op.check_empty/v1", std::vector<TaskId>{e2}).value();

                  // Serial reference run
                  const auto ser_res = nimblecas::serial_executor()->run(g).value();
                  t.expect_eq(ser_res.executed, std::size_t{6}, "serial runs all 6 tasks");

                  // Dedup on
                  FakeBrokerPort port_dedup;
                  InMemoryResultChannel chan_dedup;
                  SgeeExecutorConfig cfg_dedup;
                  cfg_dedup.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1)
                           .with_dedup_identical_tasks(true);
                  SgeeDistributedExecutor exec_dedup(cfg_dedup, port_dedup, chan_dedup);
                  const auto res_dedup = exec_dedup.run(g).value();

                  // Memo run
                  InProcessMemo memo(8, 1000, 1024 * 1024);
                  FakeBrokerPort port_memo;
                  InMemoryResultChannel chan_memo;
                  SgeeExecutorConfig cfg_memo;
                  cfg_memo.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1)
                          .with_memo(memo);
                  SgeeDistributedExecutor exec_memo(cfg_memo, port_memo, chan_memo);
                  const auto res_memo = exec_memo.run(g).value();

                  // Assert BIT-IDENTICAL outputs across all runs
                  t.expect(outputs_bit_identical(res_dedup.outputs, ser_res.outputs),
                           "dedup outputs are BIT-IDENTICAL to serial reference");
                  t.expect(outputs_bit_identical(res_memo.outputs, ser_res.outputs),
                           "memo outputs are BIT-IDENTICAL to serial reference");

                  // Check that empty payloads are valid and distinct from unresolved slots
                  t.expect(res_dedup.outputs[e0.value].has_value(), "e0 output has value");
                  t.expect(res_dedup.outputs[e0.value].value().empty(), "e0 returned genuine empty payload");
                  t.expect(res_dedup.outputs[e1.value].has_value(), "e1 output has value");
                  t.expect(res_dedup.outputs[e1.value].value().empty(), "e1 alias received genuine empty payload");
                  t.expect(res_dedup.outputs[e2.value].has_value(), "e2 output has value");
                  t.expect(res_dedup.outputs[e2.value].value().empty(), "e2 alias received genuine empty payload");

                  t.expect_eq(decode_i64(res_dedup.outputs[c0.value].value()), std::int64_t{1},
                              "c0 verified empty input (1)");
                  t.expect_eq(decode_i64(res_dedup.outputs[c1.value].value()), std::int64_t{1},
                              "c1 verified empty input from alias (1)");
                  t.expect_eq(decode_i64(res_dedup.outputs[c2.value].value()), std::int64_t{1},
                              "c2 verified empty input from alias (1)");

                  // With dedup: e0 runs once (e1, e2 aliased), c0 runs once (c1, c2 aliased) -> executed == 2
                  t.expect_eq(res_dedup.executed, std::size_t{2},
                              "dedup executed count is 2 (e0 and c0 executed once each)");
              })
        .run();
}
