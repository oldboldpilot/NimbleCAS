// SGEE distributed taskdag backend implementation (NIMBLECAS_SGEE=ON).
// @author Olumuyiwa Oluwasanmi

module;
#include "sgee_capi.h"

// Weak-symbol declaration for graceful fallback if linked against an older runtime library.
// The filtered lease arrived in SGEE main @ 6ec13bfc; a libsgee_capi.so predating it exports
// every other symbol this file uses. Declaring this one weak keeps NimbleCAS linkable and fully
// functional against that older .so, refusing ONLY the filtered path (see lease_filtered below)
// instead of failing to load at all.
extern "C" {
__attribute__((weak)) sgee_error_t sgee_broker_lease_filtered(
    sgee_task_broker_t* broker,
    uint64_t worker_id,
    uint64_t timeout_ms,
    uint64_t* out_task_id,
    uint64_t* out_token,
    int* out_placement,
    uint32_t* out_attempt,
    void** out_payload,
    size_t* out_payload_len,
    const void* payload_filter,
    size_t payload_filter_len);
}

module nimblecas.taskdag_sgee;

import std;
import nimblecas.core;
import nimblecas.taskdag;

namespace nimblecas {

auto CapiBrokerPort::create(const std::filesystem::path& wal_path,
                           std::uint64_t vis_timeout_ms,
                           std::uint32_t max_attempts)
    -> Result<std::unique_ptr<CapiBrokerPort>> {
    const std::string path_str = wal_path.string();
    sgee_task_broker_t* broker = nullptr;
    const int rc = sgee_broker_open(path_str.c_str(), vis_timeout_ms, max_attempts, &broker);
    if (rc != SGEE_OK || !broker) {
        return make_error<std::unique_ptr<CapiBrokerPort>>(MathError::distributed_error);
    }
    return std::make_unique<CapiBrokerPort>(broker);
}

CapiBrokerPort::CapiBrokerPort(sgee_task_broker_t* broker)
    : broker_(broker) {}

CapiBrokerPort::CapiBrokerPort(const std::filesystem::path& wal_path,
                               std::uint64_t vis_timeout_ms,
                               std::uint32_t max_attempts) {
    const std::string path_str = wal_path.string();
    const int rc = sgee_broker_open(path_str.c_str(), vis_timeout_ms, max_attempts, &broker_);
    if (rc != SGEE_OK) {
        broker_ = nullptr;
    }
}

CapiBrokerPort::~CapiBrokerPort() {
    if (broker_) {
        sgee_broker_destroy(broker_);
        broker_ = nullptr;
    }
}

auto CapiBrokerPort::is_open() const noexcept -> bool {
    const std::lock_guard lock(mutex_);
    return broker_ != nullptr;
}

auto CapiBrokerPort::enqueue(std::span<const std::byte> payload,
                            SgeePlacement placement, std::uint32_t max_attempts)
    -> Result<std::uint64_t> {
    const std::lock_guard lock(mutex_);
    if (!broker_) {
        return make_error<std::uint64_t>(MathError::distributed_error);
    }
    std::uint64_t out_id = 0;
    const int rc = sgee_broker_enqueue(broker_, payload.data(), payload.size(),
                                       static_cast<int>(placement), max_attempts, &out_id);
    if (rc != SGEE_OK) {
        return make_error<std::uint64_t>(MathError::distributed_error);
    }
    return out_id;
}

auto CapiBrokerPort::lease_filtered(std::uint64_t worker_id, std::uint64_t timeout_ms,
                                    std::span<const std::byte> payload_filter)
    -> Result<std::optional<Lease>> {
    const std::lock_guard lock(mutex_);
    if (!broker_) {
        return make_error<std::optional<Lease>>(MathError::distributed_error);
    }
    // An older libsgee_capi.so has no filtered entry point. REFUSE rather than fall back to the
    // unfiltered lease: silently handing this worker whatever sits at the head is precisely the
    // wrong answer the filter exists to prevent, and it would be indistinguishable from success.
    // Rule 32 -- an honest not_implemented beats a plausible-looking wrong task.
    if (!payload_filter.empty() && &sgee_broker_lease_filtered == nullptr) {
        return make_error<std::optional<Lease>>(MathError::not_implemented);
    }
    std::uint64_t qid = 0;
    std::uint64_t token = 0;
    int placement = 0;
    std::uint32_t attempt = 0;
    void* out_payload = nullptr;
    std::size_t out_payload_len = 0;

    // An empty filter routes to the ORIGINAL symbol, not to the filtered one with an empty span.
    // Upstream those are bit-identical, but routing this way means an unfiltered run against an
    // older .so takes a code path that has always existed.
    const int rc =
        payload_filter.empty()
            ? sgee_broker_lease(broker_, worker_id, timeout_ms, &qid, &token,
                                &placement, &attempt, &out_payload, &out_payload_len)
            : sgee_broker_lease_filtered(broker_, worker_id, timeout_ms, &qid, &token,
                                         &placement, &attempt, &out_payload, &out_payload_len,
                                         payload_filter.data(), payload_filter.size());
    if (rc == SGEE_ERR_QUEUE_EMPTY) {
        // Defensive: the header only populates out-params on success, but free unconditionally so
        // correctness never rests on a shim internal (this is the hottest path — polled every tick).
        if (out_payload) {
            sgee_free_buffer(out_payload);
        }
        return std::optional<Lease>{std::nullopt};
    }
    if (rc != SGEE_OK) {
        if (out_payload) {
            sgee_free_buffer(out_payload);
        }
        return make_error<std::optional<Lease>>(MathError::distributed_error);
    }

    Payload payload_bytes;
    if (out_payload != nullptr && out_payload_len > 0) {
        const auto* ptr = static_cast<const std::byte*>(out_payload);
        payload_bytes.assign(ptr, ptr + out_payload_len);
    }
    if (out_payload != nullptr) {
        sgee_free_buffer(out_payload);
        out_payload = nullptr;
    }

    return std::optional<Lease>{Lease{
        .qid = qid,
        .token = token,
        .attempt = attempt,
        .payload = std::move(payload_bytes)
    }};
}

auto CapiBrokerPort::complete(std::uint64_t qid, std::uint64_t token)
    -> Result<void> {
    const std::lock_guard lock(mutex_);
    if (!broker_) {
        return make_error<void>(MathError::distributed_error);
    }
    const int rc = sgee_broker_complete(broker_, qid, token);
    if (rc != SGEE_OK) {
        return make_error<void>(MathError::distributed_error);
    }
    return {};
}

auto CapiBrokerPort::fail(std::uint64_t qid, std::uint64_t token)
    -> Result<void> {
    const std::lock_guard lock(mutex_);
    if (!broker_) {
        return make_error<void>(MathError::distributed_error);
    }
    const int rc = sgee_broker_fail(broker_, qid, token);
    if (rc != SGEE_OK) {
        return make_error<void>(MathError::distributed_error);
    }
    return {};
}

auto CapiBrokerPort::heartbeat(std::uint64_t qid, std::uint64_t token,
                              std::uint64_t extend_by_ms) -> Result<void> {
    const std::lock_guard lock(mutex_);
    if (!broker_) {
        return make_error<void>(MathError::distributed_error);
    }
    const int rc = sgee_broker_heartbeat(broker_, qid, token, extend_by_ms);
    if (rc != SGEE_OK) {
        return make_error<void>(MathError::distributed_error);
    }
    return {};
}

auto CapiBrokerPort::sweep_expired(std::uint64_t now_ms)
    -> Result<std::size_t> {
    const std::lock_guard lock(mutex_);
    if (!broker_) {
        return make_error<std::size_t>(MathError::distributed_error);
    }
    std::size_t count = 0;
    const int rc = sgee_broker_sweep_expired(broker_, now_ms, &count);
    if (rc != SGEE_OK) {
        return make_error<std::size_t>(MathError::distributed_error);
    }
    return count;
}

auto CapiBrokerPort::state(std::uint64_t qid) -> Result<QState> {
    const std::lock_guard lock(mutex_);
    if (!broker_) {
        return make_error<QState>(MathError::distributed_error);
    }
    int st = 0;
    std::uint32_t attempt = 0;
    const int rc = sgee_broker_task_state(broker_, qid, &st, &attempt);
    if (rc != SGEE_OK) {
        return make_error<QState>(MathError::distributed_error);
    }
    switch (st) {
        case SGEE_STATE_PENDING: return QState::pending;
        case SGEE_STATE_LEASED: return QState::leased;
        case SGEE_STATE_COMPLETED: return QState::completed;
        case SGEE_STATE_DEAD: return QState::dead;
        default: return make_error<QState>(MathError::distributed_error);
    }
}

auto CapiBrokerPort::now_ms() const -> std::uint64_t {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

namespace {

// A per-run WAL path, unique across PROCESSES, not just within one: a reused WAL is durably
// recovered, so two processes sharing wal_dir that collide on a name would replay each other's
// tasks (or two single-writer brokers would corrupt one file). Timestamp + a per-process random
// nonce + a within-process atomic sequence makes a collision astronomical. Called once per run().
[[nodiscard]] auto make_unique_wal_path(const std::filesystem::path& wal_dir)
    -> std::filesystem::path {
    static std::atomic<std::uint64_t> seq{0};
    static const std::uint64_t proc_nonce = [] {
        std::random_device rd;
        return (static_cast<std::uint64_t>(rd()) << 32U) ^ static_cast<std::uint64_t>(rd());
    }();
    const auto now_ms = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    return wal_dir / ("ncsgee-" + std::to_string(now_ms) + "-" +
                      std::to_string(proc_nonce) + "-" +
                      std::to_string(seq.fetch_add(1)) + ".wal");
}

}  // namespace

auto sgee_distributed_executor(SgeeExecutorConfig cfg) -> Result<std::unique_ptr<Executor>> {
    if (cfg.registry == nullptr || cfg.wal_dir.empty() || cfg.num_workers == 0 ||
        cfg.max_attempts == 0 || cfg.visibility_timeout_ms == 0) {
        return make_error<std::unique_ptr<Executor>>(MathError::domain_error);
    }

    // Per-run transport: every run() gets a FRESH broker on a unique WAL, deleted on success and
    // retained on failure (the reaper in SgeeDistributedExecutor::run). Consecutive runs no longer
    // share a queue, so an aborted run can never leave stale tasks for the next one. Captured by
    // value so the functor stays valid for the executor's lifetime and never dangles into `cfg`
    // after the move below. Behavior change vs. the old single-broker handle: broker-open failure
    // now surfaces at run() time (distributed_error), not at factory time — the factory fails only
    // on invalid config.
    RunTransportFactory make_transport =
        [wal_dir = cfg.wal_dir, vis = cfg.visibility_timeout_ms,
         attempts = cfg.max_attempts] -> Result<RunTransport> {
            std::error_code ec;
            std::filesystem::create_directories(wal_dir, ec);  // idempotent, per-run
            if (ec) {
                return make_error<RunTransport>(MathError::distributed_error);
            }
            const std::filesystem::path wal_path = make_unique_wal_path(wal_dir);
            auto port = std::make_unique<CapiBrokerPort>(wal_path, vis, attempts);
            if (!port->is_open()) {
                return make_error<RunTransport>(MathError::distributed_error);
            }
            RunTransport tr;
            tr.owned_port = std::move(port);
            tr.port = tr.owned_port.get();
            tr.owned_channel = std::make_unique<InMemoryResultChannel>();
            tr.channel = tr.owned_channel.get();
            tr.wal_path = wal_path;
            return tr;
        };
    return std::make_unique<SgeeDistributedExecutor>(std::move(cfg), std::move(make_transport));
}

}  // namespace nimblecas
