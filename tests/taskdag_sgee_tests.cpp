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
using nimblecas::SgeeDistributedExecutor;
using nimblecas::SgeeExecutorConfig;
using nimblecas::TaskFn;
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
                  // The executor's destructor reaps its own WAL; nothing to clean up here.
              })
#endif
        .run();
}
