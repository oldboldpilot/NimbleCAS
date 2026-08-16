// Cross-process gRPC worker for the SGEE distributed integration test.
// @author Olumuyiwa Oluwasanmi
//
// A standalone worker process: connects a GrpcBrokerPort + GrpcResultChannel to running
// sgee_queue_node(s) and runs run_worker_pump until SIGTERM/SIGINT. It is a TEST fixture, not a
// general-purpose production worker (which is out of M3b/M4 scope) — it registers the diamond
// ops and an optional gate op when requested.
//
//   usage: taskdag_sgee_grpc_worker --endpoint HOST:PORT [--endpoint HOST2:PORT2 ...]
//                                   [--endpoints HOST1:PORT1,HOST2:PORT2 ...]
//                                   [--worker-id N] [--idle-backoff-ms 2] [--lease-timeout-ms 0]
//                                   [--ca-cert PATH] [--cert PATH] [--key PATH]
//                                   [--target-name-override NAME] [--gate-file PATH]

#include <csignal>

import std;
import nimblecas.core;
import nimblecas.taskdag;
import nimblecas.taskdag_sgee;

#include "taskdag_sgee_diamond_ops.h"

using nimblecas::GrpcBrokerPort;
using nimblecas::GrpcResultChannel;
using nimblecas::MathError;
using nimblecas::Payload;
using nimblecas::Result;
using nimblecas::run_worker_pump;
using nimblecas::SgeeGrpcTlsOptions;
using nimblecas::TaskRegistry;
using nimblecas::WorkerPumpConfig;

namespace {

// Set from the signal handler only (atomic<bool> is lock-free => async-signal-safe to store). A
// watcher thread polls it and drives std::stop_source, keeping the non-signal-safe request_stop()
// off the handler itself.
std::atomic<bool> g_stop{false};

extern "C" void on_signal(int) { g_stop.store(true, std::memory_order_relaxed); }

[[nodiscard]] auto arg_value(int argc, char** argv, std::string_view flag) -> std::optional<std::string> {
    for (int i = 1; i + 1 < argc; ++i) {
        if (flag == argv[i]) {
            return std::string(argv[i + 1]);
        }
    }
    return std::nullopt;
}

[[nodiscard]] auto collect_endpoints(int argc, char** argv) -> std::vector<std::string> {
    std::vector<std::string> endpoints;
    for (int i = 1; i + 1 < argc; ++i) {
        std::string_view flag{argv[i]};
        if (flag == "--endpoint" || flag == "--endpoints") {
            std::string_view val{argv[i + 1]};
            std::size_t start = 0;
            while (start < val.size()) {
                const auto comma = val.find(',', start);
                const auto sub = (comma == std::string_view::npos) ? val.substr(start) : val.substr(start, comma - start);
                if (!sub.empty()) {
                    endpoints.emplace_back(sub);
                }
                if (comma == std::string_view::npos) {
                    break;
                }
                start = comma + 1;
            }
        }
    }
    return endpoints;
}

}  // namespace

auto main(int argc, char** argv) -> int {
    const auto endpoints = collect_endpoints(argc, argv);
    if (endpoints.empty()) {
        std::cerr << "worker: --endpoint HOST:PORT is required\n";
        return 2;
    }
    std::uint64_t worker_id = 1;
    if (const auto w = arg_value(argc, argv, "--worker-id")) {
        worker_id = std::strtoull(w->c_str(), nullptr, 10);
    }
    std::uint64_t idle_backoff_ms = 2;
    if (const auto v = arg_value(argc, argv, "--idle-backoff-ms")) {
        idle_backoff_ms = std::strtoull(v->c_str(), nullptr, 10);
    }
    std::uint64_t lease_timeout_ms = 0;
    if (const auto v = arg_value(argc, argv, "--lease-timeout-ms")) {
        lease_timeout_ms = std::strtoull(v->c_str(), nullptr, 10);
    }

    SgeeGrpcTlsOptions tls{};
    if (const auto v = arg_value(argc, argv, "--ca-cert")) {
        tls.ca_cert_path = *v;
    }
    if (const auto v = arg_value(argc, argv, "--cert")) {
        tls.cert_path = *v;
    }
    if (const auto v = arg_value(argc, argv, "--key")) {
        tls.key_path = *v;
    }
    if (const auto v = arg_value(argc, argv, "--target-name-override")) {
        tls.target_name_override = *v;
    }

    TaskRegistry reg;
    nimblecas::diamond_ops::register_diamond_ops(reg);

    const auto gate_entered_file = arg_value(argc, argv, "--gate-entered-file");
    if (const auto gate_file = arg_value(argc, argv, "--gate-file")) {
        (void)reg.register_op("test.gate/v1", [gate_path = *gate_file, entered_path = gate_entered_file ? *gate_entered_file : ""](std::span<const Payload> ps) -> Result<Payload> {
            if (!entered_path.empty()) {
                std::ofstream f(entered_path);
                f << "entered\n";
            }
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
            while (!std::filesystem::exists(gate_path)) {
                if (std::chrono::steady_clock::now() >= deadline) {
                    return nimblecas::make_error<Payload>(MathError::distributed_error);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            return ps.empty() ? nimblecas::diamond_ops::encode_i64(7) : Payload(ps[0]);
        });
    }

    auto port = GrpcBrokerPort::connect(endpoints, /*auth_token=*/"", /*rpc_deadline_ms=*/0, tls);
    if (!port) {
        std::cerr << "worker " << worker_id << ": failed to connect port\n";
        return 3;
    }
    auto channel = std::make_unique<GrpcResultChannel>((*port)->ring(), /*consume_on_get=*/false);
    if (!channel) {
        std::cerr << "worker " << worker_id << ": failed to connect result channel\n";
        return 4;
    }

    std::signal(SIGTERM, on_signal);
    std::signal(SIGINT, on_signal);

    std::stop_source stop_src;
    std::thread watcher([&stop_src]() {
        while (!g_stop.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        stop_src.request_stop();
    });

    WorkerPumpConfig cfg;
    cfg.worker_id = worker_id;
    cfg.idle_backoff_ms = idle_backoff_ms;
    cfg.lease_timeout_ms = lease_timeout_ms;

    std::cerr << "worker " << worker_id << ": pumping against " << endpoints.size() << " endpoint(s)\n";
    run_worker_pump(**port, reg, *channel, cfg, stop_src.get_token());

    g_stop.store(true, std::memory_order_relaxed);
    if (watcher.joinable()) {
        watcher.join();
    }
    std::cerr << "worker " << worker_id << ": clean stop\n";
    return 0;
}
