// Tests for nimblecas.taskdag_sgee: SGEE distributed task-DAG executor scaffolding & codec.
// @author Olumuyiwa Oluwasanmi

import std;
import nimblecas.core;
import nimblecas.taskdag;
import nimblecas.taskdag_sgee;
import nimblecas.testing;

using nimblecas::BrokerPort;
#ifdef NIMBLECAS_SGEE
using nimblecas::CapiBrokerPort;
#endif
using nimblecas::CostHint;
using nimblecas::Executor;
using nimblecas::FakeBrokerPort;
using nimblecas::InMemoryResultChannel;
using nimblecas::MathError;
using nimblecas::Payload;
using nimblecas::Result;
using nimblecas::ResultChannel;
using nimblecas::run_worker_pump;
using nimblecas::SgeeDistributedExecutor;
using nimblecas::SgeeExecutorConfig;
using nimblecas::SgeePlacement;
using nimblecas::TaskFn;
using nimblecas::WorkerPumpConfig;
using nimblecas::TaskGraph;
using nimblecas::TaskId;
using nimblecas::TaskRegistry;
using nimblecas::sgee_bridge::TaskEnvelope;
using nimblecas::sgee_bridge::ResultEnvelope;
using nimblecas::sgee_bridge::encode_task;
using nimblecas::sgee_bridge::decode_task;
using nimblecas::sgee_bridge::encode_result;
using nimblecas::sgee_bridge::decode_result;
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

// Polls pred() every 1 ms until true or `budget` elapses; returns pred()'s final value.
// Assertions are made on OUTCOMES after the wait, never on timing itself.
[[nodiscard]] auto wait_until(const std::function<bool()>& pred,
                              std::chrono::milliseconds budget = std::chrono::seconds(10)) -> bool {
    const auto deadline = std::chrono::steady_clock::now() + budget;
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return pred();
}

// Accepts put() (returns success) but silently discards the first `n` payloads: the
// "worker committed via complete(), but the channel lost the bytes" fault.
class SwallowingResultChannel final : public ResultChannel {
public:
    explicit SwallowingResultChannel(std::size_t swallow_first_n) : remaining_(swallow_first_n) {}
    [[nodiscard]] auto put(std::uint64_t qid, Payload p) -> Result<void> override {
        std::lock_guard lock(mutex_);
        if (remaining_ > 0) { --remaining_; return {}; }  // pretend success, drop bytes
        return inner_.put(qid, std::move(p));
    }
    [[nodiscard]] auto get(std::uint64_t qid) const -> Result<Payload> override {
        return inner_.get(qid);
    }
private:
    mutable std::mutex mutex_;
    std::size_t remaining_;
    nimblecas::InMemoryResultChannel inner_;
};

// put() stores a COPY with byte 0 flipped (corrupt result envelope on the wire).
class CorruptingResultChannel final : public ResultChannel {
public:
    [[nodiscard]] auto put(std::uint64_t qid, Payload p) -> Result<void> override {
        std::lock_guard lock(mutex_);
        if (!p.empty()) { p[0] = p[0] ^ std::byte{0xFF}; }
        return inner_.put(qid, std::move(p));
    }
    [[nodiscard]] auto get(std::uint64_t qid) const -> Result<Payload> override {
        return inner_.get(qid);
    }
private:
    mutable std::mutex mutex_;
    nimblecas::InMemoryResultChannel inner_;
};

// put() always fails with distributed_error (drives the pump's fail() -> retry -> DLQ path).
class RefusingResultChannel final : public ResultChannel {
public:
    [[nodiscard]] auto put(std::uint64_t, Payload) -> Result<void> override {
        return nimblecas::make_error<void>(MathError::distributed_error);
    }
    [[nodiscard]] auto get(std::uint64_t) const -> Result<Payload> override {
        return nimblecas::make_error<Payload>(MathError::distributed_error);
    }
};

// Registers the diamond ops into reg and builds the diamond DAG into g. Returns the
// five task ids. A=7, B=A*2=14, C=A+3=10, D=B+C=24, probe=B*1000+C=14010.
struct DiamondIds { TaskId a, b, c, d, probe; };
[[nodiscard]] auto build_diamond(TaskRegistry& reg, TaskGraph& g) -> DiamondIds {
    (void)reg.register_op("test.const7/v1", [](auto) -> Result<Payload> { return encode_i64(7); });
    (void)reg.register_op("test.mul2/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) * 2); });
    (void)reg.register_op("test.add3/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) + 3); });
    (void)reg.register_op("test.add/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) + decode_i64(ps[1])); });
    (void)reg.register_op("test.probe/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) * 1000 + decode_i64(ps[1])); });
    DiamondIds ids;
    ids.a = g.add_named_task(reg, "test.const7/v1").value();
    ids.b = g.add_named_task(reg, "test.mul2/v1", std::vector<TaskId>{ids.a}).value();
    ids.c = g.add_named_task(reg, "test.add3/v1", std::vector<TaskId>{ids.a}).value();
    ids.d = g.add_named_task(reg, "test.add/v1", std::vector<TaskId>{ids.b, ids.c}).value();
    ids.probe = g.add_named_task(reg, "test.probe/v1", std::vector<TaskId>{ids.b, ids.c}).value();
    return ids;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.taskdag_sgee")
        .test("task_envelope_codec_roundtrip_and_layout",
              [](TestContext& t) {
                  TaskEnvelope env1{
                      .registry_fp = 0x123456789ABCDEF0ULL,
                      .op_id = "nimblecas.poly.eval/v1",
                      .args = {encode_i64(42), encode_i64(100)}
                  };

                  const auto enc1 = encode_task(env1);
                  t.expect(enc1.has_value(), "encode_task succeeds for valid envelope");
                  const auto dec1 = decode_task(*enc1);
                  t.expect(dec1.has_value(), "decode_task succeeds for encoded envelope");
                  t.expect(dec1->registry_fp == env1.registry_fp, "registry_fp roundtrip");
                  t.expect(dec1->op_id == env1.op_id, "op_id roundtrip");
                  t.expect(dec1->args.size() == 2, "args count roundtrip");
                  t.expect(decode_i64(dec1->args[0]) == 42 && decode_i64(dec1->args[1]) == 100,
                           "args content roundtrip");

                  // Hand-assert exact magic & header layout bytes
                  TaskEnvelope env_small{
                      .registry_fp = 0x0102030405060708ULL,
                      .op_id = "a.b/v1",
                      .args = {Payload{std::byte{0x77}}}
                  };
                  const auto enc_small = encode_task(env_small).value();
                  t.expect(static_cast<char>(enc_small[0]) == 'T' &&
                           static_cast<char>(enc_small[1]) == 'D' &&
                           static_cast<char>(enc_small[2]) == 'C' &&
                           static_cast<char>(enc_small[3]) == 'N',
                           "magic bytes match NCDT in LE");
                  t.expect(static_cast<std::uint8_t>(enc_small[4]) == 1, "version == 1");
                  t.expect(static_cast<std::uint8_t>(enc_small[16]) == 6, "op_len == 6");
                  t.expect(std::string_view(reinterpret_cast<const char*>(enc_small.data() + 20), 6) == "a.b/v1",
                           "op_id bytes match");
              })
        .test("result_envelope_codec_roundtrip",
              [](TestContext& t) {
                  ResultEnvelope env_ok{
                      .status = ResultEnvelope::Status::ok,
                      .math_err = MathError::division_by_zero,
                      .seconds = 1.25,
                      .bytes = encode_i64(99)
                  };
                  const auto enc_ok = encode_result(env_ok).value();
                  const auto dec_ok = decode_result(enc_ok).value();
                  t.expect(dec_ok.status == ResultEnvelope::Status::ok, "status ok");
                  t.expect(dec_ok.seconds == 1.25, "seconds roundtrip");
                  t.expect(decode_i64(dec_ok.bytes) == 99, "bytes payload roundtrip");

                  ResultEnvelope env_err{
                      .status = ResultEnvelope::Status::math_error,
                      .math_err = MathError::domain_error,
                      .seconds = 0.05,
                      .bytes = {}
                  };
                  const auto enc_err = encode_result(env_err).value();
                  const auto dec_err = decode_result(enc_err).value();
                  t.expect(dec_err.status == ResultEnvelope::Status::math_error, "status math_error");
                  t.expect(dec_err.math_err == MathError::domain_error, "math_err enum roundtrip");
              })
        .test("codec_defensive_rejection",
              [](TestContext& t) {
                  // Bad magic
                  Payload bad_magic = encode_task(TaskEnvelope{.op_id = "test/v1"}).value();
                  bad_magic[0] = std::byte{'X'};
                  t.expect(!decode_task(bad_magic).has_value() &&
                           decode_task(bad_magic).error() == MathError::syntax_error,
                           "decode_task rejects bad magic with syntax_error");

                  // Truncation
                  Payload truncated(bad_magic.begin(), bad_magic.begin() + 10);
                  t.expect(!decode_task(truncated).has_value() &&
                           decode_task(truncated).error() == MathError::syntax_error,
                           "decode_task rejects truncated bytes with syntax_error");

                  // Trailing garbage
                  Payload extra = encode_task(TaskEnvelope{.op_id = "test/v1"}).value();
                  extra.push_back(std::byte{0xFF});
                  t.expect(!decode_task(extra).has_value() &&
                           decode_task(extra).error() == MathError::syntax_error,
                           "decode_task rejects trailing garbage with syntax_error");

                  // Oversize encode
                  TaskEnvelope env_huge{.op_id = "test/v1", .args = {Payload(1024, std::byte{0})}};
                  const auto enc_cap = encode_task(env_huge, 100);
                  t.expect(!enc_cap.has_value() && enc_cap.error() == MathError::overflow,
                           "encode_task rejects payload exceeding max_bytes with overflow");

                  // Arg-length overflow (the OOB exploit): two declared lengths summing to 2^64 must
                  // be rejected, never reach subspan() as an out-of-contract count. Encode two empty
                  // args (op "a.b/v1": header 20 + op 6 + n_args 4 => length table at bytes[30..45]),
                  // then patch length[0]=100, length[1]=2^64-100 so the naive sum wraps to 0.
                  Payload oob = encode_task(TaskEnvelope{.op_id = "a.b/v1",
                                                         .args = {Payload{}, Payload{}}}).value();
                  oob[30] = std::byte{100};  // length[0] = 100 (LE, remaining bytes zero)
                  for (int k = 0; k < 8; ++k) oob[38 + k] = std::byte{0xFF};
                  oob[38] = std::byte{0x9C};  // length[1] = 0xFFFFFFFFFFFFFF9C = 2^64 - 100
                  t.expect(!decode_task(oob).has_value() &&
                           decode_task(oob).error() == MathError::syntax_error,
                           "decode_task rejects arg lengths that overflow (no OOB read)");

                  // decode_result must reject a math_err byte outside the enum (transport corruption),
                  // never smuggle a non-existent MathError in as task data.
                  Payload bad_err = encode_result(ResultEnvelope{
                      .status = ResultEnvelope::Status::math_error,
                      .math_err = MathError::domain_error, .seconds = 0.0, .bytes = {}}).value();
                  bad_err[7] = std::byte{200};  // not a valid MathError enumerator
                  t.expect(!decode_result(bad_err).has_value() &&
                           decode_result(bad_err).error() == MathError::syntax_error,
                           "decode_result rejects out-of-range MathError byte");
              })
        .test("task_registry_validation_and_fingerprint",
              [](TestContext& t) {
                  TaskRegistry reg;
                  t.expect(reg.register_op("nimblecas.eval/v1", [](auto) -> Result<Payload> { return encode_i64(1); }).has_value(),
                           "valid op grammar succeeds");

                  t.expect(!reg.register_op("invalid_grammar", [](auto) -> Result<Payload> { return encode_i64(1); }).has_value(),
                           "grammar violation without /v<N> fails");

                  t.expect(!reg.register_op("nodot/v1", [](auto) -> Result<Payload> { return encode_i64(1); }).has_value(),
                           "grammar violation without a <domain>. separator fails");

                  t.expect(!reg.register_op("nimblecas.eval/v1", [](auto) -> Result<Payload> { return encode_i64(1); }).has_value(),
                           "duplicate registration fails loudly with domain_error");

                  t.expect(reg.find("nimblecas.eval/v1") != nullptr, "registered op lookup succeeds");
                  t.expect(reg.find("absent/v1") == nullptr, "absent op lookup returns nullptr");

                  // Fingerprint determinism and order-independence
                  TaskRegistry regA;
                  TaskRegistry regB;
                  const auto fn = [](std::span<const Payload>) -> Result<Payload> { return encode_i64(0); };
                  (void)regA.register_op("a.op/v1", fn);
                  (void)regA.register_op("b.op/v1", fn);

                  (void)regB.register_op("b.op/v1", fn);
                  (void)regB.register_op("a.op/v1", fn);

                  t.expect(regA.fingerprint() == regB.fingerprint(),
                           "fingerprint is order-independent across registration order");

                  TaskRegistry regC = regA;
                  (void)regC.register_op("c.op/v1", fn);
                  t.expect(regC.fingerprint() != regA.fingerprint(),
                           "fingerprint changes when a new op is added");
              })
        .test("add_named_task_and_local_executor_parity",
              [](TestContext& t) {
                  TaskRegistry reg;
                  (void)reg.register_op("op.c7/v1", [](auto) -> Result<Payload> { return encode_i64(7); });
                  (void)reg.register_op("op.m2/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) * 2); });

                  TaskGraph g_named;
                  const auto a_n = g_named.add_named_task(reg, "op.c7/v1").value();
                  const auto b_n = g_named.add_named_task(reg, "op.m2/v1", std::vector<TaskId>{a_n}).value();

                  t.expect(g_named.op_id(a_n) == "op.c7/v1", "op_id accessor returns registered name");

                  TaskGraph g_plain;
                  const auto a_p = g_plain.add_task([](auto) -> Result<Payload> { return encode_i64(7); }).value();
                  const auto b_p = g_plain.add_task([](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) * 2); }, std::vector<TaskId>{a_p}).value();
                  t.expect(g_plain.op_id(a_p).empty(), "op_id accessor returns empty string for plain closure");

                  const auto ser = nimblecas::serial_executor();
                  const auto res_named = ser->run(g_named).value();
                  const auto res_plain = ser->run(g_plain).value();

                  t.expect(res_named.outputs.size() == res_plain.outputs.size(), "same output count");
                  t.expect(decode_i64(res_named.outputs[b_n.value].value()) == decode_i64(res_plain.outputs[b_p.value].value()),
                           "add_named_task graph output is identical to add_task closure graph on serial_executor");

                  // add_named_task honesty (§6.4 row 12): an unregistered op and an out-of-range
                  // dependency both fail with domain_error BEFORE the task is issued (no mutation).
                  const auto unknown_op = g_named.add_named_task(reg, "no.such/v1");
                  t.expect(!unknown_op.has_value() && unknown_op.error() == MathError::domain_error,
                           "add_named_task rejects an unregistered op with domain_error");
                  const auto bad_dep =
                      g_named.add_named_task(reg, "op.c7/v1", std::vector<TaskId>{TaskId{999}});
                  t.expect(!bad_dep.has_value() && bad_dep.error() == MathError::domain_error,
                           "add_named_task rejects an out-of-range dependency with domain_error");
              })
        .test("acceptance_diamond_dag_sgee_distributed_vs_serial_bit_identity",
              [](TestContext& t) {
                  // Build TaskRegistry with named ops
                  TaskRegistry reg;
                  (void)reg.register_op("test.const7/v1", [](auto) -> Result<Payload> { return encode_i64(7); });
                  (void)reg.register_op("test.mul2/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) * 2); });
                  (void)reg.register_op("test.add3/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) + 3); });
                  (void)reg.register_op("test.add/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) + decode_i64(ps[1])); });
                  (void)reg.register_op("test.probe/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) * 1000 + decode_i64(ps[1])); });

                  // Diamond DAG: A=7, B=A*2=14, C=A+3=10, D=B+C=24, probe=B*1000+C=14010
                  TaskGraph g;
                  const auto a = g.add_named_task(reg, "test.const7/v1").value();
                  const auto b = g.add_named_task(reg, "test.mul2/v1", std::vector<TaskId>{a}).value();
                  const auto c = g.add_named_task(reg, "test.add3/v1", std::vector<TaskId>{a}).value();
                  const auto d = g.add_named_task(reg, "test.add/v1", std::vector<TaskId>{b, c}).value();
                  const auto probe = g.add_named_task(reg, "test.probe/v1", std::vector<TaskId>{b, c}).value();

                  // 1. Run on reference serial_executor
                  const auto ser_exec = nimblecas::serial_executor();
                  const auto ser_res = ser_exec->run(g).value();

                  // 2. Run on SgeeDistributedExecutor over FakeBrokerPort + InMemoryResultChannel
                  FakeBrokerPort port;
                  InMemoryResultChannel results;
                  SgeeExecutorConfig cfg;
                  cfg.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1);

                  SgeeDistributedExecutor dist_exec(cfg, port, results);
                  t.expect(dist_exec.name() == "sgee_distributed", "executor name is sgee_distributed");
                  const auto dist_res = dist_exec.run(g).value();

                  // ACCEPTANCE CRITERIA:
                  // Outputs are elementwise BIT-IDENTICAL to serial_executor
                  t.expect(dist_res.outputs.size() == ser_res.outputs.size(), "outputs size match");
                  bool bit_identical = (dist_res.outputs.size() == ser_res.outputs.size());
                  for (std::size_t i = 0; bit_identical && i < dist_res.outputs.size(); ++i) {
                      bit_identical = results_equal(dist_res.outputs[i], ser_res.outputs[i]);
                  }
                  t.expect(bit_identical, "THE ACCEPTANCE TEST: SgeeDistributedExecutor outputs are BIT-IDENTICAL to serial_executor");

                  t.expect(decode_i64(dist_res.outputs[a.value].value()) == 7, "A == 7");
                  t.expect(decode_i64(dist_res.outputs[b.value].value()) == 14, "B == 14");
                  t.expect(decode_i64(dist_res.outputs[c.value].value()) == 10, "C == 10");
                  t.expect(decode_i64(dist_res.outputs[d.value].value()) == 24, "D == 24");
                  t.expect(decode_i64(dist_res.outputs[probe.value].value()) == 14010, "probe == 14010");
                  t.expect(dist_res.executed == ser_res.executed, "executed counts match (5)");
                  t.expect(dist_res.executed == 5, "executed == 5");
              })
        .test("bound_literals_distributed_execution_and_ordering_contract",
              [](TestContext& t) {
                  TaskRegistry reg;
                  // Level-0 op consuming 2 bound literals:
                  (void)reg.register_op("test.add_lits/v1",
                                        [](std::span<const Payload> ps) -> Result<Payload> {
                                            if (ps.size() != 2) {
                                                return nimblecas::make_error<Payload>(MathError::domain_error);
                                            }
                                            return encode_i64(decode_i64(ps[0]) + decode_i64(ps[1]));
                                        });
                  // Unary op consuming 1 literal:
                  (void)reg.register_op("test.echo_lit/v1",
                                        [](std::span<const Payload> ps) -> Result<Payload> {
                                            if (ps.size() != 1) {
                                                return nimblecas::make_error<Payload>(MathError::domain_error);
                                            }
                                            return ps[0];
                                        });
                  // Mixed op consuming 2 bound literals and 2 parent deps:
                  // Asserts ordering: literals first, then parent outputs in declared order
                  (void)reg.register_op("test.mixed_lits_deps/v1",
                                        [](std::span<const Payload> ps) -> Result<Payload> {
                                            if (ps.size() != 4) {
                                                return nimblecas::make_error<Payload>(MathError::domain_error);
                                            }
                                            const auto l0 = decode_i64(ps[0]);
                                            const auto l1 = decode_i64(ps[1]);
                                            const auto d0 = decode_i64(ps[2]);
                                            const auto d1 = decode_i64(ps[3]);
                                            return encode_i64((l0 * 1000 + l1) * 1000000 + (d0 * 1000 + d1));
                                        });
                  // Op with empty args (verifying empty-literals default):
                  (void)reg.register_op("test.zero_args/v1",
                                        [](std::span<const Payload> ps) -> Result<Payload> {
                                            return encode_i64(static_cast<std::int64_t>(ps.size()));
                                        });

                  TaskGraph g;
                  // 1. Level-0 named tasks with bound literals:
                  const auto a0 = g.add_named_task(reg, "test.add_lits/v1",
                                                   std::vector<Payload>{encode_i64(10), encode_i64(25)})
                                       .value();
                  const auto a1 = g.add_named_task(reg, "test.echo_lit/v1",
                                                   std::vector<Payload>{encode_i64(40)})
                                       .value();
                  const auto a_empty = g.add_named_task(reg, "test.zero_args/v1").value();

                  // Verify literals accessor
                  t.expect(g.literals(a0).size() == 2, "a0 literals size == 2");
                  t.expect(decode_i64(g.literals(a0)[0]) == 10 && decode_i64(g.literals(a0)[1]) == 25,
                           "a0 literals content matches");
                  t.expect(g.literals(a1).size() == 1 && decode_i64(g.literals(a1)[0]) == 40,
                           "a1 literals content matches");
                  t.expect(g.literals(a_empty).empty(), "a_empty has empty literals");

                  // 2. Downstream task with BOTH bound literals and parent outputs:
                  // literals = {5, 6}, deps = {a0, a1} (outputs should be 35 and 40)
                  const auto b_mixed = g.add_named_task(reg, "test.mixed_lits_deps/v1",
                                                       std::vector<Payload>{encode_i64(5), encode_i64(6)},
                                                       std::vector<TaskId>{a0, a1})
                                           .value();

                  t.expect(g.literals(b_mixed).size() == 2, "b_mixed literals size == 2");
                  t.expect(g.deps(b_mixed).size() == 2, "b_mixed deps size == 2");

                  // Run on serial reference executor
                  const auto ser = nimblecas::serial_executor();
                  const auto ser_res = ser->run(g).value();

                  // Run on local_parallel executor
                  const auto par = nimblecas::local_parallel_executor();
                  const auto par_res = par->run(g).value();

                  // Run on SgeeDistributedExecutor
                  FakeBrokerPort port;
                  InMemoryResultChannel results;
                  SgeeExecutorConfig cfg;
                  cfg.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1);
                  SgeeDistributedExecutor dist_exec(cfg, port, results);
                  const auto dist_res = dist_exec.run(g).value();

                  // Check bit-identity between serial, parallel, and distributed
                  t.expect(dist_res.outputs.size() == ser_res.outputs.size(), "output sizes match");
                  t.expect(par_res.outputs.size() == ser_res.outputs.size(), "parallel output sizes match");

                  bool dist_match = (dist_res.outputs.size() == ser_res.outputs.size());
                  for (std::size_t i = 0; dist_match && i < dist_res.outputs.size(); ++i) {
                      dist_match = results_equal(dist_res.outputs[i], ser_res.outputs[i]);
                  }
                  t.expect(dist_match, "distributed outputs are BIT-IDENTICAL to serial_executor");

                  bool par_match = (par_res.outputs.size() == ser_res.outputs.size());
                  for (std::size_t i = 0; par_match && i < par_res.outputs.size(); ++i) {
                      par_match = results_equal(par_res.outputs[i], ser_res.outputs[i]);
                  }
                  t.expect(par_match, "local_parallel outputs are BIT-IDENTICAL to serial_executor");

                  // Hand-check values
                  t.expect(decode_i64(dist_res.outputs[a0.value].value()) == 35,
                           "level-0 task a0 received bound literals: 10 + 25 = 35");
                  t.expect(decode_i64(dist_res.outputs[a1.value].value()) == 40,
                           "level-0 task a1 received bound literal: 40");
                  t.expect(decode_i64(dist_res.outputs[a_empty.value].value()) == 0,
                           "task without literals received 0 args");
                  // Hand value for b_mixed: (5 * 1000 + 6) * 1000000 + (35 * 1000 + 40) = 5006035040
                  t.expect(decode_i64(dist_res.outputs[b_mixed.value].value()) == 5006035040LL,
                           "distributed task received literals first, then parent outputs in declared order");
                  t.expect(dist_res.executed == 4, "all 4 tasks executed");
              })
        .test("error_poisoning_parity_under_distributed_execution",
              [](TestContext& t) {
                  TaskRegistry reg;
                  (void)reg.register_op("op.c1/v1", [](auto) -> Result<Payload> { return encode_i64(1); });
                  (void)reg.register_op("op.fail_dom/v1", [](auto) -> Result<Payload> { return nimblecas::make_error<Payload>(MathError::domain_error); });
                  (void)reg.register_op("op.c2/v1", [](auto) -> Result<Payload> { return encode_i64(2); });
                  (void)reg.register_op("op.add/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) + decode_i64(ps[1])); });
                  (void)reg.register_op("op.c5/v1", [](auto) -> Result<Payload> { return encode_i64(5); });
                  (void)reg.register_op("op.mul10/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) * 10); });

                  TaskGraph g;
                  const auto a = g.add_named_task(reg, "op.c1/v1").value();
                  const auto b = g.add_named_task(reg, "op.fail_dom/v1", std::vector<TaskId>{a}).value();
                  const auto c = g.add_named_task(reg, "op.c2/v1").value();
                  const auto d = g.add_named_task(reg, "op.add/v1", std::vector<TaskId>{b, c}).value();
                  const auto e = g.add_named_task(reg, "op.c5/v1").value();
                  const auto f = g.add_named_task(reg, "op.mul10/v1", std::vector<TaskId>{e}).value();

                  const auto ser_res = nimblecas::serial_executor()->run(g).value();

                  FakeBrokerPort port;
                  InMemoryResultChannel results;
                  SgeeExecutorConfig cfg;
                  cfg.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1);
                  SgeeDistributedExecutor dist_exec(cfg, port, results);
                  const auto dist_res = dist_exec.run(g).value();

                  t.expect(dist_res.outputs.size() == ser_res.outputs.size(), "outputs size match");
                  bool parity = (dist_res.outputs.size() == ser_res.outputs.size());
                  for (std::size_t i = 0; parity && i < dist_res.outputs.size(); ++i) {
                      parity = results_equal(dist_res.outputs[i], ser_res.outputs[i]);
                  }
                  t.expect(parity, "error poisoning outputs are BIT-IDENTICAL between distributed and serial");
                  t.expect(!dist_res.outputs[b.value].has_value() && dist_res.outputs[b.value].error() == MathError::domain_error,
                           "B executed and returned domain_error");
                  t.expect(!dist_res.outputs[d.value].has_value() && dist_res.outputs[d.value].error() == MathError::domain_error,
                           "D poisoned: carries B's domain_error");
                  t.expect(dist_res.outputs[f.value].has_value() && decode_i64(dist_res.outputs[f.value].value()) == 50,
                           "independent branch E->F completed to 50");
                  t.expect(dist_res.executed == ser_res.executed && dist_res.executed == 5,
                           "executed count == 5 (poisoned D not run)");
              })
        .test("honest_transport_failures_return_distributed_error",
              [](TestContext& t) {
                  TaskRegistry reg;
                  (void)reg.register_op("op.c1/v1", [](auto) -> Result<Payload> { return encode_i64(1); });

                  // 1. Unnamed task in graph -> not_implemented
                  TaskGraph g_unnamed;
                  (void)g_unnamed.add_task([](auto) -> Result<Payload> { return encode_i64(1); });
                  FakeBrokerPort port1;
                  InMemoryResultChannel results1;
                  SgeeExecutorConfig cfg1;
                  cfg1.with_registry(reg).with_num_workers(1).with_poll_interval_ms(1);
                  SgeeDistributedExecutor exec1(cfg1, port1, results1);
                  const auto res1 = exec1.run(g_unnamed);
                  t.expect(!res1.has_value() && res1.error() == MathError::not_implemented,
                           "unnamed task in graph yields not_implemented before enqueue");

                  // 2. Unregistered op -> domain_error
                  TaskGraph g_unreg;
                  TaskRegistry empty_reg;
                  (void)empty_reg.register_op("op.dummy/v1", [](auto) -> Result<Payload> { return encode_i64(1); });
                  // Add to g_unreg using empty_reg, but execute with reg (which lacks op.dummy/v1)
                  (void)g_unreg.add_named_task(empty_reg, "op.dummy/v1");
                  FakeBrokerPort port2;
                  InMemoryResultChannel results2;
                  SgeeExecutorConfig cfg2;
                  cfg2.with_registry(reg).with_num_workers(1).with_poll_interval_ms(1);
                  SgeeDistributedExecutor exec2(cfg2, port2, results2);
                  const auto res2 = exec2.run(g_unreg);
                  t.expect(!res2.has_value() && res2.error() == MathError::domain_error,
                           "unregistered op in graph yields domain_error before enqueue");

                  // 3. Broker task goes Dead -> distributed_error
                  TaskGraph g_dead;
                  (void)g_dead.add_named_task(reg, "op.c1/v1");
                  FakeBrokerPort port3;
                  InMemoryResultChannel results3;
                  // num_workers = 0 so task sits in pending until forced to dead
                  SgeeExecutorConfig cfg3;
                  cfg3.with_registry(reg).with_num_workers(0).with_poll_interval_ms(1);
                  SgeeDistributedExecutor exec3(cfg3, port3, results3);

                  // We force task 1 state to dead asynchronously
                  std::thread stopper([&port3]() {
                      std::this_thread::sleep_for(std::chrono::milliseconds(5));
                      port3.force_state(1, BrokerPort::QState::dead);
                  });
                  const auto res3 = exec3.run(g_dead);
                  if (stopper.joinable()) stopper.join();

                  t.expect(!res3.has_value() && res3.error() == MathError::distributed_error,
                           "Dead queue state surfaces as distributed_error (never a partial output)");
              })
        // ===================================================================
        // M2 hardening: honesty-path robustness suite (DEFAULT OFF build).
        // ===================================================================
        .test("lost_result_completed_but_channel_empty_aborts",  // T1 — gap (a)
              [](TestContext& t) {
                  TaskRegistry reg;
                  (void)reg.register_op("op.c7/v1", [](auto) -> Result<Payload> { return encode_i64(7); });
                  TaskGraph g;
                  (void)g.add_named_task(reg, "op.c7/v1");

                  FakeBrokerPort port;
                  SwallowingResultChannel results{1};  // worker's put() "succeeds" but drops the bytes
                  SgeeExecutorConfig cfg;
                  cfg.with_registry(reg).with_num_workers(1).with_poll_interval_ms(1);
                  SgeeDistributedExecutor exec(cfg, port, results);

                  const auto res = exec.run(g);
                  t.expect(!res.has_value() && res.error() == MathError::distributed_error,
                           "completed task with a lost result aborts the whole run (never fabricated)");
                  t.expect(port.state(1).has_value() && *port.state(1) == BrokerPort::QState::completed,
                           "the abort came from the missing result, not the queue (state == completed)");
                  t.expect(!results.get(1).has_value(), "the channel really has nothing for qid 1");
                  t.expect(port.attempts(1) == 1, "a lost result after commit is NOT retried");
              })
        .test("corrupt_result_envelope_aborts",  // T1b — result undecodable
              [](TestContext& t) {
                  TaskRegistry reg;
                  (void)reg.register_op("op.c7/v1", [](auto) -> Result<Payload> { return encode_i64(7); });
                  TaskGraph g;
                  (void)g.add_named_task(reg, "op.c7/v1");

                  FakeBrokerPort port;
                  CorruptingResultChannel results;  // flips byte 0 -> decode_result syntax_error
                  SgeeExecutorConfig cfg;
                  cfg.with_registry(reg).with_num_workers(1).with_poll_interval_ms(1);
                  SgeeDistributedExecutor exec(cfg, port, results);

                  const auto res = exec.run(g);
                  t.expect(!res.has_value() && res.error() == MathError::distributed_error,
                           "an undecodable result envelope aborts with distributed_error (codec syntax_error not leaked)");
                  t.expect(port.state(1).has_value() && *port.state(1) == BrokerPort::QState::completed,
                           "the worker completed; the coordinator refused the corrupt bytes");
              })
        .test("worker_publishes_bridge_error_for_poison_payloads",  // T2 — gap (b), pump level
              [](TestContext& t) {
                  TaskRegistry reg;
                  (void)reg.register_op("op.c7/v1", [](auto) -> Result<Payload> { return encode_i64(7); });

                  FakeBrokerPort port;
                  InMemoryResultChannel results;

                  // 1) garbage bytes -> worker decode failure
                  (void)port.enqueue(std::vector<std::byte>{std::byte{0xDE}, std::byte{0xAD}},
                                     SgeePlacement::cpu, 3);
                  // 2) valid frame, wrong fingerprint -> fp mismatch
                  const auto p2 = encode_task(TaskEnvelope{
                      .registry_fp = reg.fingerprint() ^ 1ULL, .op_id = "op.c7/v1", .args = {}}).value();
                  (void)port.enqueue(p2, SgeePlacement::cpu, 3);
                  // 3) valid frame, honest fingerprint, unknown op -> find() == nullptr
                  const auto p3 = encode_task(TaskEnvelope{
                      .registry_fp = reg.fingerprint(), .op_id = "no.such/v1", .args = {}}).value();
                  (void)port.enqueue(p3, SgeePlacement::cpu, 3);

                  std::jthread pump([&](std::stop_token st) {
                      run_worker_pump(port, reg, results,
                                      WorkerPumpConfig{.worker_id = 1, .lease_timeout_ms = 100,
                                                       .idle_backoff_ms = 1, .heartbeat_every_ms = 0}, st);
                  });
                  const bool all_done = wait_until([&] {
                      return port.state(1).has_value() && *port.state(1) == BrokerPort::QState::completed &&
                             port.state(2).has_value() && *port.state(2) == BrokerPort::QState::completed &&
                             port.state(3).has_value() && *port.state(3) == BrokerPort::QState::completed;
                  });
                  pump.request_stop();
                  pump.join();

                  t.expect(all_done, "all three poison payloads reach completed");
                  for (std::uint64_t qid = 1; qid <= 3; ++qid) {
                      const auto bytes = results.get(qid);
                      t.expect(bytes.has_value(), "a result envelope was published for the poison payload");
                      if (bytes.has_value()) {
                          const auto dec = decode_result(*bytes);
                          t.expect(dec.has_value() && dec->status == ResultEnvelope::Status::bridge_error,
                                   "worker publishes a bridge_error envelope (not a fabricated result)");
                      }
                      t.expect(port.attempts(qid) == 1, "a deterministic worker failure is never retried");
                  }
              })
        .test("coordinator_aborts_whole_run_on_bridge_error",  // T3 — gap (b), executor level
              [](TestContext& t) {
                  TaskRegistry reg_coord;
                  (void)reg_coord.register_op("op.c7/v1", [](auto) -> Result<Payload> { return encode_i64(7); });
                  TaskRegistry reg_worker;
                  (void)reg_worker.register_op("op.c7/v1", [](auto) -> Result<Payload> { return encode_i64(7); });
                  (void)reg_worker.register_op("extra.op/v1", [](auto) -> Result<Payload> { return encode_i64(0); });
                  t.expect(reg_coord.fingerprint() != reg_worker.fingerprint(),
                           "registry skew makes the fingerprints differ");

                  TaskGraph g;
                  (void)g.add_named_task(reg_coord, "op.c7/v1");

                  FakeBrokerPort port;
                  InMemoryResultChannel results;
                  SgeeExecutorConfig cfg;
                  cfg.with_registry(reg_coord).with_num_workers(0).with_poll_interval_ms(1);
                  SgeeDistributedExecutor exec(cfg, port, results);

                  std::jthread pump([&](std::stop_token st) {
                      run_worker_pump(port, reg_worker, results,
                                      WorkerPumpConfig{.worker_id = 1, .lease_timeout_ms = 100'000,
                                                       .idle_backoff_ms = 1, .heartbeat_every_ms = 0}, st);
                  });
                  const auto res = exec.run(g);
                  pump.request_stop();
                  pump.join();

                  t.expect(!res.has_value() && res.error() == MathError::distributed_error,
                           "registry-skew bridge error aborts the whole run");
                  t.expect(port.state(1).has_value() && *port.state(1) == BrokerPort::QState::completed,
                           "the worker completed honestly with a bridge error; the coordinator refused");
              })
        .test("organic_lease_expiry_retry_is_bit_identical",  // T4 — gap (c)
              [](TestContext& t) {
                  // Serial reference on a clean diamond.
                  TaskRegistry reg_s;
                  TaskGraph g_s;
                  const auto ids_s = build_diamond(reg_s, g_s);
                  const auto ser_res = nimblecas::serial_executor()->run(g_s).value();

                  // Distributed diamond whose const7 is instrumented to count physical executions.
                  std::atomic<int> exec_count{0};
                  TaskRegistry reg;
                  (void)reg.register_op("test.const7/v1", [&exec_count](auto) -> Result<Payload> {
                      exec_count.fetch_add(1); return encode_i64(7); });
                  (void)reg.register_op("test.mul2/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) * 2); });
                  (void)reg.register_op("test.add3/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) + 3); });
                  (void)reg.register_op("test.add/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) + decode_i64(ps[1])); });
                  (void)reg.register_op("test.probe/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) * 1000 + decode_i64(ps[1])); });
                  TaskGraph g;
                  const auto a = g.add_named_task(reg, "test.const7/v1").value();
                  const auto b = g.add_named_task(reg, "test.mul2/v1", std::vector<TaskId>{a}).value();
                  const auto c = g.add_named_task(reg, "test.add3/v1", std::vector<TaskId>{a}).value();
                  const auto d = g.add_named_task(reg, "test.add/v1", std::vector<TaskId>{b, c}).value();
                  const auto probe = g.add_named_task(reg, "test.probe/v1", std::vector<TaskId>{b, c}).value();

                  FakeBrokerPort port;                 // hand clock starts at 0
                  InMemoryResultChannel results;
                  SgeeExecutorConfig cfg;
                  cfg.with_registry(reg).with_num_workers(0).with_poll_interval_ms(1)
                     .with_visibility_timeout_ms(100);

                  SgeeDistributedExecutor exec(cfg, port, results);
                  Result<nimblecas::TaskRunResult> run_res = nimblecas::make_error<nimblecas::TaskRunResult>(MathError::distributed_error);
                  std::jthread runner([&] { run_res = exec.run(g); });

                  // A (qid 1) is enqueued; the test plays the doomed worker.
                  t.expect(wait_until([&] { return port.state(1).has_value(); }), "A enqueued as qid 1");
                  auto l1_res = port.lease(99, 100);
                  t.expect(l1_res.has_value() && l1_res->has_value(), "doomed worker leases A");
                  const auto l1 = l1_res->value();
                  t.expect(l1.qid == 1 && l1.attempt == 1, "first lease: qid 1, attempt 1");

                  // Execute attempt 1, publish the result, then CRASH before complete().
                  const auto env1 = decode_task(l1.payload).value();
                  const TaskFn* fn1 = reg.find(env1.op_id);
                  t.expect(fn1 != nullptr, "doomed worker resolves the op");
                  const auto r1 = (*fn1)(std::span<const Payload>(env1.args));
                  const auto renv1 = encode_result(ResultEnvelope{
                      .status = ResultEnvelope::Status::ok, .math_err = MathError::division_by_zero,
                      .seconds = 0.0, .bytes = r1.value()}).value();
                  (void)results.put(1, renv1);           // published, but never complete()d

                  // Push past the visibility deadline; the coordinator's own sweep re-pends A.
                  port.advance_time_ms(101);
                  t.expect(wait_until([&] {
                      return port.state(1).has_value() && *port.state(1) == BrokerPort::QState::pending &&
                             port.attempts(1) == 1;
                  }), "A organically re-pended by the coordinator sweep (attempt still 1)");

                  // Now a real pump re-leases and re-executes with a huge lease timeout (no further expiry).
                  std::jthread pump([&](std::stop_token st) {
                      run_worker_pump(port, reg, results,
                                      WorkerPumpConfig{.worker_id = 1, .lease_timeout_ms = 100'000,
                                                       .idle_backoff_ms = 1, .heartbeat_every_ms = 0}, st);
                  });
                  runner.join();
                  pump.request_stop();
                  pump.join();

                  t.expect(run_res.has_value(), "the run completes after the organic retry");
                  if (run_res.has_value()) {
                      bool bit_identical = (run_res->outputs.size() == ser_res.outputs.size());
                      for (std::size_t i = 0; bit_identical && i < run_res->outputs.size(); ++i) {
                          bit_identical = results_equal(run_res->outputs[i], ser_res.outputs[i]);
                      }
                      t.expect(bit_identical, "outputs are BIT-IDENTICAL to serial despite the retry");
                      t.expect(run_res->executed == 5, "executed == 5 (once per task, regardless of physical attempts)");
                      t.expect(decode_i64(run_res->outputs[a.value].value()) == 7, "A == 7");
                      t.expect(decode_i64(run_res->outputs[b.value].value()) == 14, "B == 14");
                      t.expect(decode_i64(run_res->outputs[c.value].value()) == 10, "C == 10");
                      t.expect(decode_i64(run_res->outputs[d.value].value()) == 24, "D == 24");
                      t.expect(decode_i64(run_res->outputs[probe.value].value()) == 14010, "probe == 14010");
                  }
                  t.expect(port.attempts(1) == 2, "the retry really happened (attempt 2)");
                  t.expect(exec_count.load() == 2, "A's fn physically ran twice; purity made it invisible");
              })
        .test("stale_fencing_token_is_dropped",  // T5 — gap (d), pure broker level
              [](TestContext& t) {
                  FakeBrokerPort port;
                  const auto p = encode_task(TaskEnvelope{.registry_fp = 0, .op_id = "op.c7/v1", .args = {}}).value();
                  (void)port.enqueue(p, SgeePlacement::cpu, 3);

                  const auto l1 = port.lease(1, 100).value().value();  // token t1, attempt 1
                  port.advance_time_ms(101);
                  const auto swept = port.sweep_expired(port.now_ms());
                  t.expect(swept.has_value() && *swept == 1, "the first lease expires and re-pends");
                  t.expect(port.state(1).has_value() && *port.state(1) == BrokerPort::QState::pending, "state is pending");
                  const auto l2 = port.lease(2, 100).value().value();  // token t2 != t1, attempt 2
                  t.expect(l2.token != l1.token, "the retry gets a fresh fencing token");

                  t.expect(!port.complete(1, l1.token).has_value(), "stale complete() is dropped");
                  t.expect(port.state(1).has_value() && *port.state(1) == BrokerPort::QState::leased,
                           "the retry's lease is intact after the stale commit");
                  t.expect(!port.fail(1, l1.token).has_value(), "stale fail() is dropped");
                  t.expect(!port.heartbeat(1, l1.token, 0).has_value(), "a zombie worker cannot extend the retry's lease");
                  t.expect(port.complete(1, l2.token).has_value(), "the retry's own token commits");
                  t.expect(port.state(1).has_value() && *port.state(1) == BrokerPort::QState::completed, "the retry's result wins");
                  t.expect(port.attempts(1) == 2, "exactly two attempts");
              })
        .test("poisoned_task_payload_never_enqueued",  // T6 — gap (e)
              [](TestContext& t) {
                  TaskRegistry reg;
                  (void)reg.register_op("op.c1/v1", [](auto) -> Result<Payload> { return encode_i64(1); });
                  (void)reg.register_op("op.fail_dom/v1", [](auto) -> Result<Payload> { return nimblecas::make_error<Payload>(MathError::domain_error); });
                  (void)reg.register_op("op.c2/v1", [](auto) -> Result<Payload> { return encode_i64(2); });
                  (void)reg.register_op("op.add/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) + decode_i64(ps[1])); });
                  (void)reg.register_op("op.c5/v1", [](auto) -> Result<Payload> { return encode_i64(5); });
                  (void)reg.register_op("op.mul10/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) * 10); });

                  TaskGraph g;
                  const auto a = g.add_named_task(reg, "op.c1/v1").value();
                  const auto b = g.add_named_task(reg, "op.fail_dom/v1", std::vector<TaskId>{a}).value();
                  const auto c = g.add_named_task(reg, "op.c2/v1").value();
                  const auto d = g.add_named_task(reg, "op.add/v1", std::vector<TaskId>{b, c}).value();
                  const auto e = g.add_named_task(reg, "op.c5/v1").value();
                  const auto f = g.add_named_task(reg, "op.mul10/v1", std::vector<TaskId>{e}).value();
                  (void)d; (void)f;

                  FakeBrokerPort port;
                  InMemoryResultChannel results;
                  SgeeExecutorConfig cfg;
                  cfg.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1);
                  SgeeDistributedExecutor exec(cfg, port, results);
                  const auto res = exec.run(g).value();

                  t.expect(!res.outputs[d.value].has_value() && res.outputs[d.value].error() == MathError::domain_error,
                           "D is poisoned by B's domain_error");
                  t.expect(port.enqueue_count() == 5, "exactly the five non-poisoned tasks bought qids");
                  for (std::uint64_t q = 1; q <= 5; ++q) {
                      t.expect(port.state(q).has_value() && *port.state(q) == BrokerPort::QState::completed,
                               "each of the five enqueued tasks reached completed");
                  }
                  t.expect(!port.state(6).has_value(), "no sixth task ever existed (D's bytes never travelled)");
              })
        .test("multi_worker_pumps_serial_identical",  // T7 — gap (f)
              [](TestContext& t) {
                  TaskRegistry reg;
                  for (int i = 0; i < 8; ++i) {
                      reg.register_op("op.leaf" + std::to_string(i) + "/v1",
                                      [i](auto) -> Result<Payload> { return encode_i64(i); }).value();
                  }
                  (void)reg.register_op("op.add/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) + decode_i64(ps[1])); });

                  TaskGraph g;
                  std::vector<TaskId> leaves;
                  for (int i = 0; i < 8; ++i) {
                      leaves.push_back(g.add_named_task(reg, "op.leaf" + std::to_string(i) + "/v1").value());
                  }
                  const auto n0 = g.add_named_task(reg, "op.add/v1", std::vector<TaskId>{leaves[0], leaves[1]}).value();
                  const auto n1 = g.add_named_task(reg, "op.add/v1", std::vector<TaskId>{leaves[2], leaves[3]}).value();
                  const auto n2 = g.add_named_task(reg, "op.add/v1", std::vector<TaskId>{leaves[4], leaves[5]}).value();
                  const auto n3 = g.add_named_task(reg, "op.add/v1", std::vector<TaskId>{leaves[6], leaves[7]}).value();
                  const auto m0 = g.add_named_task(reg, "op.add/v1", std::vector<TaskId>{n0, n1}).value();
                  const auto m1 = g.add_named_task(reg, "op.add/v1", std::vector<TaskId>{n2, n3}).value();
                  const auto root = g.add_named_task(reg, "op.add/v1", std::vector<TaskId>{m0, m1}).value();

                  const auto ser_res = nimblecas::serial_executor()->run(g).value();
                  t.expect(decode_i64(ser_res.outputs[root.value].value()) == 28, "serial reduction root == 28");

                  for (const std::size_t wc : {std::size_t{1}, std::size_t{2}, std::size_t{4}}) {
                      FakeBrokerPort port;
                      InMemoryResultChannel results;
                      SgeeExecutorConfig cfg;
                      cfg.with_registry(reg).with_num_workers(wc).with_poll_interval_ms(1);
                      SgeeDistributedExecutor exec(cfg, port, results);
                      const auto res = exec.run(g).value();

                      bool identical = (res.outputs.size() == ser_res.outputs.size());
                      for (std::size_t i = 0; identical && i < res.outputs.size(); ++i) {
                          identical = results_equal(res.outputs[i], ser_res.outputs[i]);
                      }
                      t.expect(identical, "outputs are BIT-IDENTICAL to serial at every worker count");
                      t.expect(res.executed == 15, "all 15 tasks executed exactly once");
                      t.expect(port.enqueue_count() == 15, "every task enqueued exactly once (no double-submit under contention)");
                  }
              })
        .test("dead_letter_after_max_attempts_aborts_run",  // T8 — gap (g), organic DLQ
              [](TestContext& t) {
                  TaskRegistry reg;
                  (void)reg.register_op("op.c7/v1", [](auto) -> Result<Payload> { return encode_i64(7); });
                  TaskGraph g;
                  (void)g.add_named_task(reg, "op.c7/v1");

                  FakeBrokerPort port;
                  RefusingResultChannel results;  // every put() fails -> fail() -> retry -> DLQ
                  SgeeExecutorConfig cfg;
                  cfg.with_registry(reg).with_num_workers(1).with_poll_interval_ms(1).with_max_attempts(2);
                  SgeeDistributedExecutor exec(cfg, port, results);

                  const auto res = exec.run(g);
                  t.expect(!res.has_value() && res.error() == MathError::distributed_error,
                           "a task that exhausts its attempts aborts the run");
                  t.expect(port.state(1).has_value() && *port.state(1) == BrokerPort::QState::dead, "qid 1 is dead");
                  t.expect(port.attempts(1) == 2, "exactly max_attempts attempts, no over-retry");
              })
        .test("heartbeat_extends_visibility_under_sweep",  // T9 — Part D
              [](TestContext& t) {
                  std::binary_semaphore sem{0};
                  TaskRegistry reg;
                  (void)reg.register_op("test.block/v1", [sem = &sem](auto) -> Result<Payload> {
                      sem->acquire(); return encode_i64(9); });

                  FakeBrokerPort port;  // clock at 0
                  InMemoryResultChannel results;
                  const auto p = encode_task(TaskEnvelope{
                      .registry_fp = reg.fingerprint(), .op_id = "test.block/v1", .args = {}}).value();
                  (void)port.enqueue(p, SgeePlacement::cpu, 3);

                  std::jthread pump([&](std::stop_token st) {
                      run_worker_pump(port, reg, results,
                                      WorkerPumpConfig{.worker_id = 1, .lease_timeout_ms = 100,
                                                       .idle_backoff_ms = 1, .heartbeat_every_ms = 1}, st);
                  });

                  t.expect(wait_until([&] {
                      return port.state(1).has_value() && *port.state(1) == BrokerPort::QState::leased;
                  }), "the task is leased (initial deadline 0 + 100)");
                  t.expect(wait_until([&] { return port.heartbeat_count(1) >= 1; }),
                           "a heartbeat landed (deadline now 0 + 10000)");

                  port.advance_time_ms(5'000);  // now = 5000, inside 10000
                  const auto s1 = port.sweep_expired(port.now_ms());
                  t.expect(s1.has_value() && *s1 == 0, "sweep at t=5000 does not expire the heartbeated lease");
                  t.expect(port.state(1).has_value() && *port.state(1) == BrokerPort::QState::leased, "still leased");

                  const auto hb = port.heartbeat_count(1);
                  t.expect(wait_until([&] { return port.heartbeat_count(1) > hb; }), "another heartbeat lands at t=5000");
                  t.expect(port.visibility_deadline_ms(1) >= 15'000,
                           "the extension is real: deadline pushed strictly beyond the earlier 10000");

                  port.advance_time_ms(5'000);  // now = 10000, past the step-2 deadline
                  const auto s2 = port.sweep_expired(port.now_ms());
                  t.expect(s2.has_value() && *s2 == 0, "the guard kept the lease alive past the un-extended deadline");

                  sem.release();
                  t.expect(wait_until([&] {
                      return port.state(1).has_value() && *port.state(1) == BrokerPort::QState::completed;
                  }), "the task completes once released");
                  pump.request_stop();
                  pump.join();
                  const auto bytes = results.get(1);
                  t.expect(bytes.has_value() && decode_result(*bytes).value().status == ResultEnvelope::Status::ok &&
                           decode_i64(decode_result(*bytes).value().bytes) == 9, "result is 9");
                  t.expect(port.attempts(1) == 1, "no spurious retry, ever");
              })
        .test("coordinator_transport_faults_abort_honestly",  // T10 (i)-(iii)
              [](TestContext& t) {
                  const auto make_single = [](TaskRegistry& reg, TaskGraph& g) {
                      (void)reg.register_op("op.c7/v1", [](auto) -> Result<Payload> { return encode_i64(7); });
                      (void)g.add_named_task(reg, "op.c7/v1");
                  };

                  // (i) enqueue fault -> no qid consumed
                  {
                      TaskRegistry reg; TaskGraph g; make_single(reg, g);
                      FakeBrokerPort port; InMemoryResultChannel results;
                      port.inject_fault(FakeBrokerPort::FaultOp::enqueue);
                      SgeeExecutorConfig cfg;
                      cfg.with_registry(reg).with_num_workers(1).with_poll_interval_ms(1);
                      SgeeDistributedExecutor exec(cfg, port, results);
                      const auto res = exec.run(g);
                      t.expect(!res.has_value() && res.error() == MathError::distributed_error, "(i) enqueue fault aborts");
                      t.expect(port.enqueue_count() == 0, "(i) the fault consumed no qid");
                  }
                  // (ii) sweep fault on the first await tick
                  {
                      TaskRegistry reg; TaskGraph g; make_single(reg, g);
                      FakeBrokerPort port; InMemoryResultChannel results;
                      port.inject_fault(FakeBrokerPort::FaultOp::sweep);
                      SgeeExecutorConfig cfg;
                      cfg.with_registry(reg).with_num_workers(1).with_poll_interval_ms(1);
                      SgeeDistributedExecutor exec(cfg, port, results);
                      const auto res = exec.run(g);
                      t.expect(!res.has_value() && res.error() == MathError::distributed_error, "(ii) sweep fault aborts");
                  }
                  // (iii) state fault on the per-qid poll
                  {
                      TaskRegistry reg; TaskGraph g; make_single(reg, g);
                      FakeBrokerPort port; InMemoryResultChannel results;
                      port.inject_fault(FakeBrokerPort::FaultOp::state);
                      SgeeExecutorConfig cfg;
                      cfg.with_registry(reg).with_num_workers(1).with_poll_interval_ms(1);
                      SgeeDistributedExecutor exec(cfg, port, results);
                      const auto res = exec.run(g);
                      t.expect(!res.has_value() && res.error() == MathError::distributed_error, "(iii) state fault aborts");
                  }
                  // (iv) per-run transport factory failure & null-transport (the OFF-build path that
                  // keeps the per-run lifecycle honest without SGEE).
                  {
                      TaskRegistry reg; TaskGraph g; make_single(reg, g);
                      SgeeExecutorConfig cfg;
                      cfg.with_registry(reg).with_num_workers(1).with_poll_interval_ms(1);
                      SgeeDistributedExecutor exec(cfg, nimblecas::RunTransportFactory{
                          []() -> Result<nimblecas::RunTransport> {
                              return nimblecas::make_error<nimblecas::RunTransport>(MathError::distributed_error);
                          }});
                      const auto res = exec.run(g);
                      t.expect(!res.has_value() && res.error() == MathError::distributed_error,
                               "(iv) a failing transport factory aborts run() before any enqueue");
                  }
                  {
                      TaskRegistry reg; TaskGraph g; make_single(reg, g);
                      SgeeExecutorConfig cfg;
                      cfg.with_registry(reg).with_num_workers(1).with_poll_interval_ms(1);
                      SgeeDistributedExecutor exec(cfg, nimblecas::RunTransportFactory{
                          []() -> Result<nimblecas::RunTransport> {
                              return nimblecas::RunTransport{};  // null port/channel views
                          }});
                      const auto res = exec.run(g);
                      t.expect(!res.has_value() && res.error() == MathError::distributed_error,
                               "(iv) a transport with a null port aborts run() honestly");
                  }
              })
        .test("oversize_fan_in_payload_aborts_run",  // T12 — run-level oversize (kept large; near the end)
              [](TestContext& t) {
                  constexpr std::size_t k_max = nimblecas::sgee_bridge::k_max_task_payload_bytes;
                  TaskRegistry reg;
                  (void)reg.register_op("op.big/v1", [](auto) -> Result<Payload> {
                      return Payload(k_max - 8, std::byte{0}); });
                  (void)reg.register_op("op.len/v1", [](auto ps) -> Result<Payload> {
                      return encode_i64(static_cast<std::int64_t>(ps[0].size())); });

                  TaskGraph g;
                  const auto big = g.add_named_task(reg, "op.big/v1").value();
                  (void)g.add_named_task(reg, "op.len/v1", std::vector<TaskId>{big}).value();

                  FakeBrokerPort port;
                  InMemoryResultChannel results;
                  SgeeExecutorConfig cfg;
                  cfg.with_registry(reg).with_num_workers(1).with_poll_interval_ms(1);
                  SgeeDistributedExecutor exec(cfg, port, results);

                  const auto res = exec.run(g);
                  t.expect(!res.has_value() && res.error() == MathError::distributed_error,
                           "an oversize child task envelope aborts (codec overflow not leaked)");
                  t.expect(port.enqueue_count() == 1, "the producer ran; the oversize child was never enqueued");
              })
        .test("run_deadline_exceeded_aborts",  // T13 — whole-run liveness deadline
              [](TestContext& t) {
                  TaskRegistry reg;
                  (void)reg.register_op("op.c7/v1", [](auto) -> Result<Payload> { return encode_i64(7); });
                  TaskGraph g;
                  (void)g.add_named_task(reg, "op.c7/v1");

                  FakeBrokerPort port;
                  InMemoryResultChannel results;
                  SgeeExecutorConfig cfg;
                  cfg.with_registry(reg).with_num_workers(0).with_poll_interval_ms(1).with_run_deadline_ms(50);
                  SgeeDistributedExecutor exec(cfg, port, results);

                  const auto res = exec.run(g);
                  t.expect(!res.has_value() && res.error() == MathError::distributed_error,
                           "an unserved task past the run deadline aborts honestly");
                  t.expect(port.state(1).has_value() && *port.state(1) == BrokerPort::QState::pending,
                           "the task was enqueued and honestly abandoned, not fabricated");
              })
#ifndef NIMBLECAS_SGEE
        .test("stub_factory_is_honest",  // T11 — OFF build only
              [](TestContext& t) {
                  SgeeExecutorConfig bad_cfg;  // default: null registry etc.
                  const auto bad = nimblecas::sgee_distributed_executor(bad_cfg);
                  t.expect(!bad.has_value() && bad.error() == MathError::domain_error,
                           "stub factory rejects invalid config with domain_error");

                  TaskRegistry reg;
                  (void)reg.register_op("op.c7/v1", [](auto) -> Result<Payload> { return encode_i64(7); });
                  SgeeExecutorConfig good;
                  good.with_registry(reg)
                      .with_wal_dir(std::filesystem::temp_directory_path())
                      .with_num_workers(1).with_max_attempts(3).with_visibility_timeout_ms(30'000);
                  const auto res = nimblecas::sgee_distributed_executor(good);
                  t.expect(!res.has_value() && res.error() == MathError::not_implemented,
                           "stub factory NEVER pretends a backend exists (not_implemented for valid config)");
              })
#endif
#ifdef NIMBLECAS_SGEE
        .test("acceptance_diamond_dag_real_capi_broker_port",
              [](TestContext& t) {
                  // Build TaskRegistry with named ops
                  TaskRegistry reg;
                  (void)reg.register_op("test.const7/v1", [](auto) -> Result<Payload> { return encode_i64(7); });
                  (void)reg.register_op("test.mul2/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) * 2); });
                  (void)reg.register_op("test.add3/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) + 3); });
                  (void)reg.register_op("test.add/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) + decode_i64(ps[1])); });
                  (void)reg.register_op("test.probe/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) * 1000 + decode_i64(ps[1])); });

                  // Diamond DAG: A=7, B=A*2=14, C=A+3=10, D=B+C=24, probe=B*1000+C=14010
                  TaskGraph g;
                  const auto a = g.add_named_task(reg, "test.const7/v1").value();
                  const auto b = g.add_named_task(reg, "test.mul2/v1", std::vector<TaskId>{a}).value();
                  const auto c = g.add_named_task(reg, "test.add3/v1", std::vector<TaskId>{a}).value();
                  const auto d = g.add_named_task(reg, "test.add/v1", std::vector<TaskId>{b, c}).value();
                  const auto probe = g.add_named_task(reg, "test.probe/v1", std::vector<TaskId>{b, c}).value();

                  // 1. Reference serial_executor
                  const auto ser_exec = nimblecas::serial_executor();
                  const auto ser_res = ser_exec->run(g).value();

                  // 2. Real CapiBrokerPort over libsgee_capi C-ABI with unique WAL path
                  static std::atomic<std::uint64_t> test_seq{0};
                  const auto now_ms = static_cast<std::uint64_t>(
                      std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch()).count());
                  const auto tmp_dir = std::filesystem::temp_directory_path();
                  const auto wal_filename = "ncsgee_test_real_capi_" + std::to_string(now_ms) + "_" + std::to_string(test_seq.fetch_add(1)) + ".wal";
                  const auto wal_path = tmp_dir / wal_filename;

                  auto port_res = CapiBrokerPort::create(wal_path, 30'000, 3);
                  t.expect(port_res.has_value(), "CapiBrokerPort opens real WAL broker successfully");
                  if (!port_res.has_value()) {
                      return;
                  }
                  auto& port = **port_res;
                  InMemoryResultChannel results;

                  SgeeExecutorConfig cfg;
                  cfg.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1);

                  SgeeDistributedExecutor dist_exec(cfg, port, results);
                  t.expect(dist_exec.name() == "sgee_distributed", "executor name is sgee_distributed");
                  const auto dist_res_res = dist_exec.run(g);
                  t.expect(dist_res_res.has_value(), "SgeeDistributedExecutor run succeeds over real CapiBrokerPort");
                  if (!dist_res_res.has_value()) {
                      std::error_code ec;
                      std::filesystem::remove(wal_path, ec);
                      return;
                  }
                  const auto& dist_res = *dist_res_res;

                  // Assert elementwise bit-identity
                  t.expect(dist_res.outputs.size() == ser_res.outputs.size(), "outputs size match");
                  bool bit_identical = (dist_res.outputs.size() == ser_res.outputs.size());
                  for (std::size_t i = 0; bit_identical && i < dist_res.outputs.size(); ++i) {
                      bit_identical = results_equal(dist_res.outputs[i], ser_res.outputs[i]);
                  }
                  t.expect(bit_identical, "REAL CAPI ACCEPTANCE: SgeeDistributedExecutor outputs are BIT-IDENTICAL to serial_executor");

                  t.expect(decode_i64(dist_res.outputs[a.value].value()) == 7, "A == 7");
                  t.expect(decode_i64(dist_res.outputs[b.value].value()) == 14, "B == 14");
                  t.expect(decode_i64(dist_res.outputs[c.value].value()) == 10, "C == 10");
                  t.expect(decode_i64(dist_res.outputs[d.value].value()) == 24, "D == 24");
                  t.expect(decode_i64(dist_res.outputs[probe.value].value()) == 14010, "probe == 14010");
                  t.expect(dist_res.executed == ser_res.executed, "executed counts match (5)");
                  t.expect(dist_res.executed == 5, "executed == 5");

                  // Also test math error task poisoning identically over real broker
                  TaskRegistry err_reg;
                  (void)err_reg.register_op("op.c1/v1", [](auto) -> Result<Payload> { return encode_i64(1); });
                  (void)err_reg.register_op("op.fail_dom/v1", [](auto) -> Result<Payload> { return nimblecas::make_error<Payload>(MathError::domain_error); });
                  (void)err_reg.register_op("op.c2/v1", [](auto) -> Result<Payload> { return encode_i64(2); });
                  (void)err_reg.register_op("op.add/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) + decode_i64(ps[1])); });

                  TaskGraph g_err;
                  const auto ea = g_err.add_named_task(err_reg, "op.c1/v1").value();
                  const auto eb = g_err.add_named_task(err_reg, "op.fail_dom/v1", std::vector<TaskId>{ea}).value();
                  const auto ec_id = g_err.add_named_task(err_reg, "op.c2/v1").value();
                  const auto ed = g_err.add_named_task(err_reg, "op.add/v1", std::vector<TaskId>{eb, ec_id}).value();

                  const auto ser_err_res = ser_exec->run(g_err).value();

                  const auto wal_filename2 = "ncsgee_test_real_capi_err_" + std::to_string(now_ms) + "_" + std::to_string(test_seq.fetch_add(1)) + ".wal";
                  const auto wal_path2 = tmp_dir / wal_filename2;
                  auto port_err_res = CapiBrokerPort::create(wal_path2, 30'000, 3);
                  t.expect(port_err_res.has_value(), "CapiBrokerPort for err test opens successfully");
                  if (port_err_res.has_value()) {
                      InMemoryResultChannel results_err;
                      SgeeExecutorConfig cfg_err;
                      cfg_err.with_registry(err_reg).with_num_workers(2).with_poll_interval_ms(1);
                      SgeeDistributedExecutor dist_err_exec(cfg_err, **port_err_res, results_err);
                      const auto dist_err_res_res = dist_err_exec.run(g_err);
                      t.expect(dist_err_res_res.has_value(), "err-graph run succeeds over real CapiBrokerPort");
                      if (!dist_err_res_res.has_value()) {
                          std::error_code ec2;
                          std::filesystem::remove(wal_path2, ec2);
                          std::filesystem::remove(wal_path, ec2);
                          return;
                      }
                      const auto& dist_err_res = *dist_err_res_res;

                      t.expect(dist_err_res.outputs.size() == ser_err_res.outputs.size(), "err outputs size match");
                      bool err_parity = (dist_err_res.outputs.size() == ser_err_res.outputs.size());
                      for (std::size_t i = 0; err_parity && i < dist_err_res.outputs.size(); ++i) {
                          err_parity = results_equal(dist_err_res.outputs[i], ser_err_res.outputs[i]);
                      }
                      t.expect(err_parity, "error poisoning outputs are BIT-IDENTICAL over real CapiBrokerPort");
                      t.expect(!dist_err_res.outputs[eb.value].has_value() && dist_err_res.outputs[eb.value].error() == MathError::domain_error,
                               "B executed and returned domain_error on real broker");
                      t.expect(!dist_err_res.outputs[ed.value].has_value() && dist_err_res.outputs[ed.value].error() == MathError::domain_error,
                               "D poisoned on real broker");

                      std::error_code ec;
                      std::filesystem::remove(wal_path2, ec);
                  }

                  // Cleanup main test WAL file
                  std::error_code ec;
                  std::filesystem::remove(wal_path, ec);
              })
        .test("real_factory_sgee_distributed_executor_and_config_validation",
              [](TestContext& t) {
                  // Invalid config is rejected with domain_error before any broker is opened.
                  SgeeExecutorConfig bad_cfg;  // null registry + empty wal_dir + zero workers
                  auto bad_res = nimblecas::sgee_distributed_executor(bad_cfg);
                  t.expect(!bad_res.has_value() && bad_res.error() == MathError::domain_error,
                           "factory rejects invalid config with domain_error");

                  // Valid config: the factory opens its OWN broker on a unique WAL and reaps it on
                  // destruction. Drive the diamond DAG through it and assert bit-identity to serial.
                  TaskRegistry reg;
                  (void)reg.register_op("test.const7/v1", [](auto) -> Result<Payload> { return encode_i64(7); });
                  (void)reg.register_op("test.mul2/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) * 2); });
                  (void)reg.register_op("test.add3/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) + 3); });
                  (void)reg.register_op("test.add/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) + decode_i64(ps[1])); });

                  TaskGraph g;
                  const auto a = g.add_named_task(reg, "test.const7/v1").value();
                  const auto b = g.add_named_task(reg, "test.mul2/v1", std::vector<TaskId>{a}).value();
                  const auto c = g.add_named_task(reg, "test.add3/v1", std::vector<TaskId>{a}).value();
                  const auto d = g.add_named_task(reg, "test.add/v1", std::vector<TaskId>{b, c}).value();

                  const auto ser_exec = nimblecas::serial_executor();
                  const auto ser_res = ser_exec->run(g).value();

                  SgeeExecutorConfig cfg;
                  cfg.with_registry(reg)
                      .with_wal_dir(std::filesystem::temp_directory_path())
                      .with_num_workers(2)
                      .with_poll_interval_ms(1)
                      .with_max_attempts(3)
                      .with_visibility_timeout_ms(30'000);
                  auto exec_res = nimblecas::sgee_distributed_executor(cfg);
                  t.expect(exec_res.has_value(), "factory builds a live executor for valid config");
                  if (exec_res.has_value()) {
                      const auto dist_res = (*exec_res)->run(g);
                      t.expect(dist_res.has_value(), "factory-built executor runs the graph");
                      if (dist_res.has_value()) {
                          bool ok = (dist_res->outputs.size() == ser_res.outputs.size());
                          for (std::size_t i = 0; ok && i < dist_res->outputs.size(); ++i) {
                              ok = results_equal(dist_res->outputs[i], ser_res.outputs[i]);
                          }
                          t.expect(ok, "factory executor outputs are BIT-IDENTICAL to serial_executor");
                          t.expect(dist_res->executed == ser_res.executed, "factory executor executed count matches");
                      }
                  }
                  // Each successful run() reaps its own per-run WAL; nothing to clean up here.
              })
        .test("wal_reaped_on_success_retained_on_failure",  // T14 — per-run WAL lifecycle (ON only)
              [](TestContext& t) {
                  static std::atomic<std::uint64_t> t14_seq{0};
                  const auto now_ms = static_cast<std::uint64_t>(
                      std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch()).count());
                  const auto wal_dir = std::filesystem::temp_directory_path() /
                      ("ncsgee_m2_" + std::to_string(now_ms) + "_" + std::to_string(t14_seq.fetch_add(1)));
                  std::error_code ec;
                  std::filesystem::create_directories(wal_dir, ec);

                  const auto count_wals = [&]() -> std::size_t {
                      std::size_t count = 0;
                      std::error_code lec;
                      for (const auto& e : std::filesystem::directory_iterator(wal_dir, lec)) {
                          const auto fname = e.path().filename().string();
                          if (fname.rfind("ncsgee-", 0) == 0 && e.path().extension() == ".wal") { ++count; }
                      }
                      return count;
                  };

                  // Success half: the diamond succeeds; its per-run WAL is reaped.
                  TaskRegistry reg; TaskGraph g;
                  const auto ids = build_diamond(reg, g);
                  (void)ids;
                  const auto ser_res = nimblecas::serial_executor()->run(g).value();

                  SgeeExecutorConfig cfg;
                  cfg.with_registry(reg).with_wal_dir(wal_dir).with_num_workers(2)
                     .with_poll_interval_ms(1).with_max_attempts(3).with_visibility_timeout_ms(30'000);
                  auto exec_res = nimblecas::sgee_distributed_executor(cfg);
                  t.expect(exec_res.has_value(), "factory builds a per-run executor");
                  if (exec_res.has_value()) {
                      const auto r = (*exec_res)->run(g);
                      t.expect(r.has_value(), "the diamond run succeeds");
                      if (r.has_value()) {
                          bool ok = (r->outputs.size() == ser_res.outputs.size());
                          for (std::size_t i = 0; ok && i < r->outputs.size(); ++i) {
                              ok = results_equal(r->outputs[i], ser_res.outputs[i]);
                          }
                          t.expect(ok, "outputs bit-identical to serial");
                      }
                      t.expect(count_wals() == 0, "the WAL is reaped on success (zero *.wal remain)");

                      // Failure half: a run that aborts on its deadline retains exactly one WAL.
                      TaskRegistry sreg;
                      (void)sreg.register_op("test.sleep/v1", [](auto) -> Result<Payload> {
                          std::this_thread::sleep_for(std::chrono::milliseconds(50));
                          return encode_i64(1); });
                      TaskGraph sg;
                      (void)sg.add_named_task(sreg, "test.sleep/v1");
                      SgeeExecutorConfig scfg;
                      scfg.with_registry(sreg).with_wal_dir(wal_dir).with_num_workers(1)
                          .with_poll_interval_ms(1).with_max_attempts(3)
                          .with_visibility_timeout_ms(30'000).with_run_deadline_ms(1);
                      auto sexec_res = nimblecas::sgee_distributed_executor(scfg);
                      t.expect(sexec_res.has_value(), "factory builds the sleeper executor");
                      if (sexec_res.has_value()) {
                          const auto sr = (*sexec_res)->run(sg);
                          t.expect(!sr.has_value() && sr.error() == MathError::distributed_error,
                                   "the sleeper run aborts on its deadline");
                          t.expect(count_wals() == 1, "exactly one WAL retained for post-mortem");
                      }

                      // Reuse the first executor after a failed run -> succeeds (fresh per-run broker,
                      // no stale-task contamination; the very defect per-run brokers remove).
                      const auto r2 = (*exec_res)->run(g);
                      t.expect(r2.has_value(), "the per-run executor is reusable after a failed run");
                  }
                  std::filesystem::remove_all(wal_dir, ec);
              })
#endif
        .run();
}
