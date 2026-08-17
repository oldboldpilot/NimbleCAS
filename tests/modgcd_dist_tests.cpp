// Tests for nimblecas.modgcd_dist: distributed modular polynomial GCD driver.
// @author Olumuyiwa Oluwasanmi
//
// Test cases D1..D5 from M5_SPEC.md section 8.2 covering bit-identity across execution
// engines (serial, local_parallel, SgeeDistributedExecutor over FakeBrokerPort, modular_gcd,
// and Polynomial::gcd), WU1 bound-literal conformance, math-error as data propagation,
// transport aborts and organic lease-expiry recovery, and complete-but-absent recovery bounds.

import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.bigint;
import nimblecas.numbertheory;
import nimblecas.modgcd;
import nimblecas.taskdag;
import nimblecas.taskdag_sgee;
import nimblecas.modgcd_dist;
import nimblecas.testing;

using nimblecas::BigInt;
using nimblecas::BrokerPort;
using nimblecas::Candidate;
using nimblecas::CoprimeProven;
using nimblecas::CostHint;
using nimblecas::decode_image_request;
using nimblecas::decode_image_result;
using nimblecas::decode_polynomial;
using nimblecas::encode_image_request;
using nimblecas::encode_image_result;
using nimblecas::encode_polynomial;
using nimblecas::Executor;
using nimblecas::FakeBrokerPort;
using nimblecas::gcd_image_mod_p;
using nimblecas::ImageRequest;
using nimblecas::ImageResult;
using nimblecas::InMemoryResultChannel;
using nimblecas::local_parallel_executor;
using nimblecas::MathError;
using nimblecas::modular_gcd;
using nimblecas::modular_gcd_with;
using nimblecas::Payload;
using nimblecas::Polynomial;
using nimblecas::register_modgcd_ops;
using nimblecas::Result;
using nimblecas::ResultChannel;
using nimblecas::run_worker_pump;
using nimblecas::serial_executor;
using nimblecas::SgeeDistributedExecutor;
using nimblecas::SgeeExecutorConfig;
using nimblecas::TaskGraph;
using nimblecas::TaskId;
using nimblecas::TaskRegistry;
using nimblecas::WorkerPumpConfig;
using nimblecas::sgee_bridge::TaskEnvelope;
using nimblecas::sgee_bridge::encode_task;
using nimblecas::sgee_bridge::decode_task;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

auto poly(std::vector<std::int64_t> coeffs) -> Polynomial {
    return Polynomial{std::move(coeffs)};
}

[[nodiscard]] auto encode_i64(std::int64_t v) -> Payload {
    const auto bytes = std::bit_cast<std::array<std::byte, sizeof(std::int64_t)>>(v);
    return Payload(bytes.begin(), bytes.end());
}

[[nodiscard]] auto decode_i64(std::span<const std::byte> p) -> std::int64_t {
    std::array<std::byte, sizeof(std::int64_t)> bytes{};
    std::ranges::copy_n(p.begin(), std::min<std::size_t>(p.size(), sizeof(std::int64_t)),
                        bytes.begin());
    return std::bit_cast<std::int64_t>(bytes);
}

[[nodiscard]] auto results_equal(const Result<Payload>& a, const Result<Payload>& b) -> bool {
    if (a.has_value() != b.has_value()) {
        return false;
    }
    return a.has_value() ? (*a == *b) : (a.error() == b.error());
}

class SwallowingResultChannel final : public ResultChannel {
public:
    explicit SwallowingResultChannel(std::size_t swallow_first_n) : remaining_(swallow_first_n) {}

    [[nodiscard]] auto put(std::uint64_t qid, Payload p) -> Result<void> override {
        std::lock_guard lock(mutex_);
        if (remaining_ > 0) {
            --remaining_;
            return {};
        }
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

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.modgcd_dist")
        // -------------------------------------------------------------------
        // D1: bit-identity ladder across all executors
        // -------------------------------------------------------------------
        .test("D1_bit_identity_ladder", [](TestContext& t) {
            TaskRegistry reg;
            auto reg_res = register_modgcd_ops(reg);
            t.expect(reg_res.has_value(), "register_modgcd_ops succeeds");

            // Test inputs corresponding to T1..T5
            std::vector<std::pair<Polynomial, Polynomial>> cases = {
                // T1: known gcd x + 1
                {poly({-2, -1, 1}), poly({3, 4, 1})},
                // T2: coprime inputs, gcd = 1
                {poly({1, 0, 1}), poly({-2, 0, 1})},
                // T3: content handling, gcd = 2x - 2
                {poly({-6, 0, 6}), poly({4, -8, 4})},
                // T4: nontrivial gamma, deg-3 gcd = 3x^3 + 2x^2 + x + 5
                {poly({35, 7, 19, 22, 2, 3}), poly({55, 6, 26, 32, -1, 3})},
                // T5: unlucky first prime discard, gcd = 1
                {poly({-1, 1}), poly({-1073741828, 1})},
            };

            // Seeded sweep cases (T8 inputs)
            std::uint64_t state = 0xABCDEF0123456789ULL;
            auto next_rand = [&]() -> std::int64_t {
                state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                return static_cast<std::int64_t>((state >> 60) & 0xF) - 7;
            };

            for (std::size_t trial = 0; trial < 15; ++trial) {
                const std::size_t deg_g = (trial % 2) + 1;
                std::vector<std::int64_t> g_coeffs(deg_g + 1);
                for (auto& c : g_coeffs) { c = next_rand(); }
                if (g_coeffs.back() == 0) { g_coeffs.back() = 1; }
                const auto g = poly(g_coeffs);

                const std::size_t deg_u = ((trial + 1) % 2) + 1;
                std::vector<std::int64_t> u_coeffs(deg_u + 1);
                for (auto& c : u_coeffs) { c = next_rand(); }
                if (u_coeffs.back() == 0) { u_coeffs.back() = 1; }
                const auto u = poly(u_coeffs);

                const std::size_t deg_v = ((trial + 2) % 2) + 1;
                std::vector<std::int64_t> v_coeffs(deg_v + 1);
                for (auto& c : v_coeffs) { c = next_rand(); }
                if (v_coeffs.back() == 0) { v_coeffs.back() = 1; }
                const auto v = poly(v_coeffs);

                auto a_res = g.multiply(u);
                auto b_res = g.multiply(v);
                if (a_res && b_res && !a_res->is_zero() && !b_res->is_zero()) {
                    cases.emplace_back(std::move(*a_res), std::move(*b_res));
                }
            }

            for (std::size_t idx = 0; idx < cases.size(); ++idx) {
                const auto& [a, b] = cases[idx];

                // 1. Serial executor
                const auto ser_exec = serial_executor();
                auto res_ser = modular_gcd_with(a, b, *ser_exec, reg);
                t.expect(res_ser.has_value(), std::format("case {} serial modular_gcd_with succeeds", idx));

                // 2. Local parallel executor
                const auto par_exec = local_parallel_executor();
                auto res_par = modular_gcd_with(a, b, *par_exec, reg);
                t.expect(res_par.has_value(), std::format("case {} local_parallel modular_gcd_with succeeds", idx));

                // 3. SgeeDistributedExecutor over FakeBrokerPort
                FakeBrokerPort port;
                InMemoryResultChannel results;
                SgeeExecutorConfig cfg;
                cfg.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1);
                SgeeDistributedExecutor dist_exec(cfg, port, results);
                auto res_dist = modular_gcd_with(a, b, dist_exec, reg);
                t.expect(res_dist.has_value(), std::format("case {} distributed modular_gcd_with succeeds", idx));

                // 4. In-process modular_gcd
                auto res_mod = modular_gcd(a, b);
                t.expect(res_mod.has_value(), std::format("case {} in-process modular_gcd succeeds", idx));

                // 5. PRS exact reference
                auto res_prs = a.gcd(b);
                t.expect(res_prs.has_value(), std::format("case {} Polynomial::gcd succeeds", idx));

                if (res_ser && res_par && res_dist && res_mod && res_prs) {
                    t.expect(res_ser->is_equal(*res_mod),
                             std::format("case {} serial matches modular_gcd bit-for-bit", idx));
                    t.expect(res_par->is_equal(*res_mod),
                             std::format("case {} local_parallel matches modular_gcd bit-for-bit", idx));
                    t.expect(res_dist->is_equal(*res_mod),
                             std::format("case {} distributed matches modular_gcd bit-for-bit", idx));
                    t.expect(res_mod->is_equal(*res_prs),
                             std::format("case {} modular_gcd matches Polynomial::gcd bit-for-bit", idx));
                }
            }
        })

        // -------------------------------------------------------------------
        // D2: WU1 conformance — bound literal arguments
        // -------------------------------------------------------------------
        .test("D2_wu1_literal_conformance", [](TestContext& t) {
            // 1. Envelope round-trip preserves literals byte-exactly
            TaskEnvelope env{
                .registry_fp = 0x1234567890ABCDEFULL,
                .op_id = "test.custom/v1",
                .args = {encode_i64(100), encode_i64(200), encode_i64(300)},
            };
            auto enc_env = encode_task(env);
            t.expect(enc_env.has_value(), "encode_task succeeds");
            if (enc_env) {
                auto dec_env = decode_task(*enc_env);
                t.expect(dec_env.has_value(), "decode_task succeeds");
                if (dec_env) {
                    t.expect(dec_env->registry_fp == env.registry_fp, "fp matches");
                    t.expect(dec_env->op_id == env.op_id, "op_id matches");
                    t.expect(dec_env->args.size() == 3, "args count matches (3)");
                    if (dec_env->args.size() == 3) {
                        t.expect(decode_i64(dec_env->args[0]) == 100, "arg[0] == 100");
                        t.expect(decode_i64(dec_env->args[1]) == 200, "arg[1] == 200");
                        t.expect(decode_i64(dec_env->args[2]) == 300, "arg[2] == 300");
                    }
                }
            }

            // 2. Args arrive as literals-then-deps across all three executors
            TaskRegistry reg;
            (void)reg.register_op("test.const7/v1", [](auto) -> Result<Payload> {
                return encode_i64(7);
            });
            (void)reg.register_op("test.lit_and_dep/v1", [](std::span<const Payload> args) -> Result<Payload> {
                if (args.size() != 3) {
                    return make_error<Payload>(MathError::syntax_error);
                }
                const std::int64_t lit1 = decode_i64(args[0]);
                const std::int64_t lit2 = decode_i64(args[1]);
                const std::int64_t dep1 = decode_i64(args[2]);
                // lit1 * 100 + lit2 * 10 + dep1
                return encode_i64(lit1 * 100 + lit2 * 10 + dep1);
            });

            TaskGraph g;
            const auto dep_id = g.add_named_task(reg, "test.const7/v1").value();
            const auto task_id = g.add_named_task(reg, "test.lit_and_dep/v1",
                                                  {encode_i64(3), encode_i64(5)},
                                                  std::vector<TaskId>{dep_id}).value();

            // Expected result: 3 * 100 + 5 * 10 + 7 = 357
            const auto ser_exec = serial_executor();
            const auto ser_res = ser_exec->run(g).value();
            t.expect(decode_i64(ser_res.outputs[task_id.value].value()) == 357,
                     "serial executor produces 357");

            const auto par_exec = local_parallel_executor();
            const auto par_res = par_exec->run(g).value();
            t.expect(decode_i64(par_res.outputs[task_id.value].value()) == 357,
                     "local_parallel executor produces 357");

            FakeBrokerPort port;
            InMemoryResultChannel results;
            SgeeExecutorConfig cfg;
            cfg.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1);
            SgeeDistributedExecutor dist_exec(cfg, port, results);
            const auto dist_res = dist_exec.run(g).value();
            t.expect(decode_i64(dist_res.outputs[task_id.value].value()) == 357,
                     "SgeeDistributedExecutor produces 357");

            // Bit-identity across all three executors
            bool bit_id_par = (par_res.outputs.size() == ser_res.outputs.size());
            bool bit_id_dist = (dist_res.outputs.size() == ser_res.outputs.size());
            for (std::size_t i = 0; i < ser_res.outputs.size(); ++i) {
                if (!results_equal(par_res.outputs[i], ser_res.outputs[i])) { bit_id_par = false; }
                if (!results_equal(dist_res.outputs[i], ser_res.outputs[i])) { bit_id_dist = false; }
            }
            t.expect(bit_id_par, "local_parallel is bit-identical to serial");
            t.expect(bit_id_dist, "SgeeDistributedExecutor is bit-identical to serial");
        })

        // -------------------------------------------------------------------
        // D3: math error is data
        // -------------------------------------------------------------------
        .test("D3_math_error_is_data", [](TestContext& t) {
            TaskRegistry reg;
            auto reg_res = register_modgcd_ops(reg);
            t.expect(reg_res.has_value(), "register_modgcd_ops succeeds");

            // Hand-built graph with one corrupted ImageRequest literal
            TaskGraph g;
            Payload corrupt_req = {std::byte{0xDE}, std::byte{0xAD}, std::byte{0xBE}, std::byte{0xEF}};
            const auto t_corrupt = g.add_named_task(reg, "nimblecas.modgcd.image/v1",
                                                    {corrupt_req}).value();

            // Serial
            const auto ser_exec = serial_executor();
            const auto ser_run = ser_exec->run(g);
            t.expect(ser_run.has_value(), "serial run succeeds at transport level");
            if (ser_run) {
                t.expect(!ser_run->outputs[t_corrupt.value].has_value(), "task output is an error");
                t.expect(ser_run->outputs[t_corrupt.value].error() == MathError::syntax_error,
                         "task output error is MathError::syntax_error");
            }

            // Local parallel
            const auto par_exec = local_parallel_executor();
            const auto par_run = par_exec->run(g);
            t.expect(par_run.has_value(), "local_parallel run succeeds at transport level");
            if (par_run) {
                t.expect(!par_run->outputs[t_corrupt.value].has_value(), "task output is an error");
                t.expect(par_run->outputs[t_corrupt.value].error() == MathError::syntax_error,
                         "task output error is MathError::syntax_error");
            }

            // SgeeDistributedExecutor
            FakeBrokerPort port;
            InMemoryResultChannel results;
            SgeeExecutorConfig cfg;
            cfg.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1);
            SgeeDistributedExecutor dist_exec(cfg, port, results);
            const auto dist_run = dist_exec.run(g);
            t.expect(dist_run.has_value(), "distributed run succeeds at transport level");
            if (dist_run) {
                t.expect(!dist_run->outputs[t_corrupt.value].has_value(), "task output is an error");
                t.expect(dist_run->outputs[t_corrupt.value].error() == MathError::syntax_error,
                         "task output error is MathError::syntax_error");
            }

            // Verify bit-identity of error outcomes across all three
            if (ser_run && par_run && dist_run) {
                t.expect(results_equal(par_run->outputs[0], ser_run->outputs[0]),
                         "local_parallel error matches serial");
                t.expect(results_equal(dist_run->outputs[0], ser_run->outputs[0]),
                         "distributed error matches serial");
            }
        })

        // -------------------------------------------------------------------
        // D4: transport aborts and organic lease-expiry recovery
        // -------------------------------------------------------------------
        .test("D4_transport_aborts_and_recovery", [](TestContext& t) {
            TaskRegistry reg;
            (void)register_modgcd_ops(reg);
            const auto a = poly({-2, -1, 1});
            const auto b = poly({3, 4, 1});

            // 1. Enqueue fault aborts whole run with distributed_error
            {
                FakeBrokerPort port;
                InMemoryResultChannel results;
                port.inject_fault(FakeBrokerPort::FaultOp::enqueue);
                SgeeExecutorConfig cfg;
                cfg.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1);
                SgeeDistributedExecutor dist_exec(cfg, port, results);
                auto res = modular_gcd_with(a, b, dist_exec, reg);
                t.expect(!res.has_value() && res.error() == MathError::distributed_error,
                         "enqueue fault aborts with distributed_error");
            }

            // 2. Dead queue state aborts with distributed_error
            {
                FakeBrokerPort port;
                InMemoryResultChannel results;
                SgeeExecutorConfig cfg;
                cfg.with_registry(reg).with_num_workers(0).with_poll_interval_ms(1);
                SgeeDistributedExecutor dist_exec(cfg, port, results);

                std::thread stopper([&port]() {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    port.force_state(1, BrokerPort::QState::dead);
                });
                auto res = modular_gcd_with(a, b, dist_exec, reg);
                if (stopper.joinable()) { stopper.join(); }

                t.expect(!res.has_value() && res.error() == MathError::distributed_error,
                         "dead queue state aborts with distributed_error");
            }

            // 3. Lease-expiry organic retry converges to exact gcd
            {
                FakeBrokerPort port;
                InMemoryResultChannel results;
                SgeeExecutorConfig cfg;
                cfg.with_registry(reg).with_num_workers(0).with_poll_interval_ms(1);
                SgeeDistributedExecutor dist_exec(cfg, port, results);

                Result<Polynomial> dist_res;
                std::thread coord([&]() {
                    dist_res = modular_gcd_with(a, b, dist_exec, reg);
                });

                // Wait until tasks are enqueued
                while (port.enqueue_count() == 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                // Worker 1 leases task 1 with 100ms timeout
                auto l1 = port.lease(1, 100);
                t.expect(l1.has_value() && l1->has_value(), "worker 1 acquired lease on task 1");

                // Advance clock past expiration to expire worker 1's lease
                port.advance_time_ms(150);

                // Start worker pump that picks up the expired task and executes to completion
                std::jthread pump([&](std::stop_token st) {
                    run_worker_pump(port, reg, results,
                                    WorkerPumpConfig{.worker_id = 2, .lease_timeout_ms = 100'000,
                                                     .idle_backoff_ms = 1, .heartbeat_every_ms = 0}, st);
                });

                coord.join();
                pump.request_stop();
                pump.join();

                t.expect(dist_res.has_value(), "run converges to exact gcd after lease-expiry retry");
                if (dist_res.has_value()) {
                    t.expect(dist_res->is_equal(poly({1, 1})),
                             "recovered gcd is x + 1 exactly");
                }
            }
        })

        // -------------------------------------------------------------------
        // D5: complete-but-absent recovery bound
        // -------------------------------------------------------------------
        .test("D5_recovery_bound", [](TestContext& t) {
            TaskRegistry reg;
            (void)register_modgcd_ops(reg);
            const auto a = poly({-2, -1, 1});
            const auto b = poly({3, 4, 1});

            // 1. max_result_recoveries(0) aborts with distributed_error on lost result
            {
                FakeBrokerPort port;
                SwallowingResultChannel results{1};  // drops 1st result
                SgeeExecutorConfig cfg;
                cfg.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1)
                   .with_max_result_recoveries(0);
                SgeeDistributedExecutor dist_exec(cfg, port, results);
                auto res = modular_gcd_with(a, b, dist_exec, reg);
                t.expect(!res.has_value() && res.error() == MathError::distributed_error,
                         "lost result with max_result_recoveries(0) aborts with distributed_error");
            }

            // 2. max_result_recoveries(3) recovers and succeeds with exact gcd
            {
                FakeBrokerPort port;
                SwallowingResultChannel results{1};  // drops 1st result
                SgeeExecutorConfig cfg;
                cfg.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1)
                   .with_max_result_recoveries(3);
                SgeeDistributedExecutor dist_exec(cfg, port, results);
                auto res = modular_gcd_with(a, b, dist_exec, reg);
                t.expect(res.has_value(), "lost result with max_result_recoveries(3) recovers and succeeds");
                if (res.has_value()) {
                    t.expect(res->is_equal(poly({1, 1})),
                             "recovered gcd is x + 1 exactly");
                }
            }
        })

        .run();
}
