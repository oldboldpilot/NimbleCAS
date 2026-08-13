// Cross-process gRPC worker for the SGEE distributed integration test.
// @author Olumuyiwa Oluwasanmi
//
// A standalone worker process: connects a GrpcBrokerPort + GrpcResultChannel to a running
// sgee_queue_node and runs run_worker_pump until SIGTERM/SIGINT. It is a TEST fixture, not a
// general-purpose production worker (which is out of M3b scope) — it registers exactly the diamond
// ops the integration test enqueues.
//
//   usage: taskdag_sgee_grpc_worker --endpoint HOST:PORT --worker-id N
//                                   [--idle-backoff-ms 2] [--lease-timeout-ms 0]

#include <csignal>

import std;
import nimblecas.core;
import nimblecas.taskdag;
import nimblecas.taskdag_sgee;

#include "taskdag_sgee_diamond_ops.h"

using nimblecas::GrpcBrokerPort;
using nimblecas::GrpcResultChannel;
using nimblecas::run_worker_pump;
using nimblecas::TaskRegistry;
using nimblecas::WorkerPumpConfig;

namespace {

// Set from the signal handler only (atomic<bool> is lock-free ⇒ async-signal-safe to store). A
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

}  // namespace

auto main(int argc, char** argv) -> int {
    const auto endpoint = arg_value(argc, argv, "--endpoint");
    if (!endpoint) {
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

    TaskRegistry reg;
    nimblecas::diamond_ops::register_diamond_ops(reg);

    auto port = GrpcBrokerPort::connect(*endpoint);
    if (!port) {
        std::cerr << "worker " << worker_id << ": failed to connect port to " << *endpoint << "\n";
        return 3;
    }
    auto channel = GrpcResultChannel::connect(*endpoint, /*consume_on_get=*/false);
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

    std::cerr << "worker " << worker_id << ": pumping against " << *endpoint << "\n";
    run_worker_pump(**port, reg, **channel, cfg, stop_src.get_token());

    g_stop.store(true, std::memory_order_relaxed);
    if (watcher.joinable()) {
        watcher.join();
    }
    std::cerr << "worker " << worker_id << ": clean stop\n";
    return 0;
}
