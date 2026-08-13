// SGEE cross-process gRPC distributed taskdag backend implementation (NIMBLECAS_SGEE_GRPC=ON).
// @author Olumuyiwa Oluwasanmi

module;
#include "sgee_capi_grpc.h"

module nimblecas.taskdag_sgee;

import std;
import nimblecas.core;
import nimblecas.taskdag;

namespace {

// RAII free for a malloc'd buffer handed back across the C ABI. Guarantees the C-heap buffer is
// released on EVERY exit path — including if copying it into a Payload throws std::bad_alloc,
// which a manual free-after-copy would leak.
struct GrpcBufferGuard {
    void* p{nullptr};
    explicit GrpcBufferGuard(void* ptr) noexcept : p(ptr) {}
    ~GrpcBufferGuard() {
        if (p != nullptr) {
            sgee_grpc_free_buffer(p);
        }
    }
    GrpcBufferGuard(const GrpcBufferGuard&) = delete;
    auto operator=(const GrpcBufferGuard&) -> GrpcBufferGuard& = delete;
};

}  // namespace

namespace nimblecas {

// ---------------------------------------------------------------------------
// GrpcBrokerPort
// ---------------------------------------------------------------------------

auto GrpcBrokerPort::connect(std::string endpoint, std::string auth_token,
                             std::uint64_t rpc_deadline_ms)
    -> Result<std::unique_ptr<GrpcBrokerPort>> {
    sgee_grpc_client_t* client = nullptr;
    const int rc = sgee_grpc_client_open(endpoint.c_str(),
                                         auth_token.empty() ? nullptr : auth_token.c_str(),
                                         rpc_deadline_ms, &client);
    if (rc != SGEE_OK || client == nullptr) {
        return make_error<std::unique_ptr<GrpcBrokerPort>>(MathError::distributed_error);
    }
    return std::make_unique<GrpcBrokerPort>(client);
}

GrpcBrokerPort::GrpcBrokerPort(sgee_grpc_client_t* client) : client_(client) {}

GrpcBrokerPort::~GrpcBrokerPort() {
    if (client_) {
        sgee_grpc_client_destroy(client_);
        client_ = nullptr;
    }
}

auto GrpcBrokerPort::is_open() const noexcept -> bool {
    std::lock_guard lock(mutex_);
    return client_ != nullptr;
}

auto GrpcBrokerPort::enqueue(std::span<const std::byte> payload,
                             SgeePlacement placement, std::uint32_t max_attempts)
    -> Result<std::uint64_t> {
    std::lock_guard lock(mutex_);
    if (!client_) {
        return make_error<std::uint64_t>(MathError::distributed_error);
    }
    std::uint64_t out_id = 0;
    const int rc = sgee_grpc_enqueue(client_, payload.data(), payload.size(),
                                     static_cast<int>(placement), max_attempts, &out_id);
    if (rc != SGEE_OK) {
        return make_error<std::uint64_t>(MathError::distributed_error);
    }
    return out_id;
}

auto GrpcBrokerPort::lease(std::uint64_t worker_id, std::uint64_t timeout_ms)
    -> Result<std::optional<Lease>> {
    std::lock_guard lock(mutex_);
    if (!client_) {
        return make_error<std::optional<Lease>>(MathError::distributed_error);
    }
    std::uint64_t qid = 0;
    std::uint64_t token_local = 0;
    std::uint64_t token_term = 0;
    std::uint64_t token_index = 0;
    int placement = 0;
    std::uint32_t attempt = 0;
    void* out_payload = nullptr;
    std::size_t out_payload_len = 0;

    const int rc = sgee_grpc_lease(client_, worker_id, timeout_ms, &qid, &token_local,
                                   &token_term, &token_index, &placement, &attempt,
                                   &out_payload, &out_payload_len);
    // Frees out_payload on every path below, including a throwing assign() (defensive on null).
    GrpcBufferGuard payload_guard{out_payload};
    if (rc == SGEE_ERR_QUEUE_EMPTY) {
        return std::optional<Lease>{std::nullopt};
    }
    if (rc != SGEE_OK) {
        return make_error<std::optional<Lease>>(MathError::distributed_error);
    }

    Payload payload_bytes;
    if (out_payload != nullptr && out_payload_len > 0) {
        const auto* ptr = static_cast<const std::byte*>(out_payload);
        payload_bytes.assign(ptr, ptr + out_payload_len);
    }

    // Stash the full wire triple so complete/fail/heartbeat can re-attach term/index. A re-lease
    // after expiry mints a NEW token, so overwrite any prior entry for this qid.
    leases_[qid] = TokenTriple{.local = token_local, .term = token_term, .index = token_index};

    return std::optional<Lease>{Lease{
        .qid = qid,
        .token = token_local,
        .attempt = attempt,
        .payload = std::move(payload_bytes)
    }};
}

auto GrpcBrokerPort::complete(std::uint64_t qid, std::uint64_t token)
    -> Result<void> {
    std::lock_guard lock(mutex_);
    if (!client_) {
        return make_error<void>(MathError::distributed_error);
    }
    const auto it = leases_.find(qid);
    if (it == leases_.end() || it->second.local != token) {
        // Stale/unknown caller: this port never took (or already released) this lease. Fail
        // without an RPC — exactly the fencing a bare CapiBrokerPort delegates to the broker.
        return make_error<void>(MathError::distributed_error);
    }
    const int rc = sgee_grpc_complete(client_, qid, it->second.local, it->second.term,
                                      it->second.index);
    // Erase regardless of outcome: this lease is over either way (a failed complete is not
    // retried by the pump), so keeping the entry would only leak on a long-lived worker.
    leases_.erase(it);
    if (rc != SGEE_OK) {
        return make_error<void>(MathError::distributed_error);
    }
    return {};
}

auto GrpcBrokerPort::fail(std::uint64_t qid, std::uint64_t token)
    -> Result<void> {
    std::lock_guard lock(mutex_);
    if (!client_) {
        return make_error<void>(MathError::distributed_error);
    }
    const auto it = leases_.find(qid);
    if (it == leases_.end() || it->second.local != token) {
        return make_error<void>(MathError::distributed_error);
    }
    const int rc = sgee_grpc_fail(client_, qid, it->second.local, it->second.term,
                                  it->second.index);
    // Erase regardless of outcome (see complete()): the lease is over, never revisit this qid.
    leases_.erase(it);
    if (rc != SGEE_OK) {
        return make_error<void>(MathError::distributed_error);
    }
    return {};
}

auto GrpcBrokerPort::heartbeat(std::uint64_t qid, std::uint64_t token,
                               std::uint64_t extend_by_ms) -> Result<void> {
    std::lock_guard lock(mutex_);
    if (!client_) {
        return make_error<void>(MathError::distributed_error);
    }
    const auto it = leases_.find(qid);
    if (it == leases_.end() || it->second.local != token) {
        return make_error<void>(MathError::distributed_error);
    }
    // Retain the entry: the lease lives on after a heartbeat.
    const int rc = sgee_grpc_heartbeat(client_, qid, it->second.local, it->second.term,
                                       it->second.index, extend_by_ms);
    if (rc != SGEE_OK) {
        return make_error<void>(MathError::distributed_error);
    }
    return {};
}

auto GrpcBrokerPort::sweep_expired(std::uint64_t /*now_ms*/)
    -> Result<std::size_t> {
    // No-op by design: sgee_queue_node's driver background thread sweeps expired leases
    // autonomously. There is no client-side sweep RPC and none is needed; the coordinator's
    // per-tick call becomes a free success. now_ms is ignored (see now_ms()).
    return std::size_t{0};
}

auto GrpcBrokerPort::state(std::uint64_t qid) -> Result<QState> {
    std::lock_guard lock(mutex_);
    if (!client_) {
        return make_error<QState>(MathError::distributed_error);
    }
    int st = 0;
    std::uint32_t attempt = 0;
    const int rc = sgee_grpc_task_state(client_, qid, &st, &attempt);
    if (rc != SGEE_OK) {
        // UNKNOWN_TASK included: the coordinator enqueued this qid, so "unknown" means the node
        // compacted it away or we are pointed at the wrong endpoint — an honest whole-run abort.
        return make_error<QState>(MathError::distributed_error);
    }
    switch (st) {
        case SGEE_STATE_PENDING:   return QState::pending;
        case SGEE_STATE_LEASED:    return QState::leased;
        case SGEE_STATE_COMPLETED: return QState::completed;
        case SGEE_STATE_DEAD:      return QState::dead;
        default:                   return make_error<QState>(MathError::distributed_error);
    }
}

auto GrpcBrokerPort::now_ms() const -> std::uint64_t {
    // Inert for this port: lease deadlines live on the server's clock and the only coordinator
    // consumer of now_ms() is the argument to sweep_expired(), which is a no-op here. Returned as
    // wall-clock ms only to honor the interface's "same domain as the deadlines this port sets"
    // spirit and to keep any future diagnostics sane.
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

// ---------------------------------------------------------------------------
// GrpcResultChannel
// ---------------------------------------------------------------------------

auto GrpcResultChannel::connect(std::string endpoint, bool consume_on_get,
                                std::string auth_token, std::uint64_t rpc_deadline_ms)
    -> Result<std::unique_ptr<GrpcResultChannel>> {
    sgee_grpc_client_t* client = nullptr;
    const int rc = sgee_grpc_client_open(endpoint.c_str(),
                                         auth_token.empty() ? nullptr : auth_token.c_str(),
                                         rpc_deadline_ms, &client);
    if (rc != SGEE_OK || client == nullptr) {
        return make_error<std::unique_ptr<GrpcResultChannel>>(MathError::distributed_error);
    }
    return std::make_unique<GrpcResultChannel>(client, consume_on_get);
}

GrpcResultChannel::GrpcResultChannel(sgee_grpc_client_t* client, bool consume_on_get)
    : client_(client), consume_on_get_(consume_on_get) {}

GrpcResultChannel::~GrpcResultChannel() {
    if (client_) {
        sgee_grpc_client_destroy(client_);
        client_ = nullptr;
    }
}

auto GrpcResultChannel::is_open() const noexcept -> bool {
    std::lock_guard lock(mutex_);
    return client_ != nullptr;
}

auto GrpcResultChannel::put(std::uint64_t qid, Payload result_envelope) -> Result<void> {
    std::lock_guard lock(mutex_);
    if (!client_) {
        return make_error<void>(MathError::distributed_error);
    }
    // Token zeros: the server's result store does not enforce fencing (see the proto note); the
    // reader is gated by the broker's fenced Complete plus op determinism.
    const int rc = sgee_grpc_put_result(client_, qid, 0, 0, 0,
                                        result_envelope.data(), result_envelope.size());
    if (rc != SGEE_OK) {
        return make_error<void>(MathError::distributed_error);
    }
    return {};
}

auto GrpcResultChannel::get(std::uint64_t qid) const -> Result<Payload> {
    std::lock_guard lock(mutex_);
    if (!client_) {
        return make_error<Payload>(MathError::distributed_error);
    }
    int found = 0;
    void* out_bytes = nullptr;
    std::size_t out_len = 0;
    const int rc = sgee_grpc_get_result(client_, qid, consume_on_get_ ? 1 : 0, &found,
                                        &out_bytes, &out_len);
    // Frees out_bytes on every path below, including a throwing assign() (defensive on null).
    GrpcBufferGuard bytes_guard{out_bytes};
    if (rc != SGEE_OK) {
        return make_error<Payload>(MathError::distributed_error);
    }
    if (!found) {
        // Absent here is a real fault: the coordinator only calls get() after state()==completed,
        // so a missing result means the node lost its in-memory store (restart) — honest abort.
        // Matches InMemoryResultChannel::get's contract.
        return make_error<Payload>(MathError::distributed_error);
    }
    Payload bytes;
    if (out_bytes != nullptr && out_len > 0) {
        const auto* ptr = static_cast<const std::byte*>(out_bytes);
        bytes.assign(ptr, ptr + out_len);
    }
    return bytes;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

auto sgee_grpc_distributed_executor(SgeeExecutorConfig cfg, SgeeGrpcExecutorOptions opts)
    -> Result<std::unique_ptr<Executor>> {
    // wal_dir IGNORED (the queue is server-side); num_workers == 0 is VALID (pure remote workers).
    if (cfg.registry == nullptr || cfg.max_attempts == 0 || cfg.visibility_timeout_ms == 0 ||
        opts.endpoint.empty()) {
        return make_error<std::unique_ptr<Executor>>(MathError::domain_error);
    }

    // Per-run transport: every run() opens a FRESH port + channel pair (independent connections)
    // to the endpoint. Captured by value so the functor never dangles into `cfg`/`opts` after the
    // moves below. No local WAL, so RunTransport.wal_path stays empty and the reaper deletes
    // nothing. Broker-unreachable surfaces at the first RPC in run(), not here (lazy connect).
    RunTransportFactory make_transport =
        [endpoint = opts.endpoint, auth = opts.auth_token,
         deadline = opts.rpc_deadline_ms]() -> Result<RunTransport> {
            auto port = GrpcBrokerPort::connect(endpoint, auth, deadline);
            if (!port) {
                return make_error<RunTransport>(port.error());
            }
            auto channel = GrpcResultChannel::connect(endpoint, /*consume_on_get=*/true, auth,
                                                      deadline);
            if (!channel) {
                return make_error<RunTransport>(channel.error());
            }
            RunTransport tr;
            tr.owned_port = std::move(*port);
            tr.port = tr.owned_port.get();
            tr.owned_channel = std::move(*channel);
            tr.channel = tr.owned_channel.get();
            // tr.wal_path stays empty: nothing local to reap.
            return tr;
        };
    return std::make_unique<SgeeDistributedExecutor>(std::move(cfg), std::move(make_transport));
}

}  // namespace nimblecas
