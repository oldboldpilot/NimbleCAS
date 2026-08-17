// SGEE cross-process gRPC distributed taskdag backend implementation (NIMBLECAS_SGEE_GRPC=ON).
// @author Olumuyiwa Oluwasanmi

module;
#include "sgee_capi_grpc.h"

// Weak-symbol declarations for graceful fallback if linked against an older runtime library
extern "C" {
__attribute__((weak)) sgee_error_t sgee_grpc_client_open_tls(
    const char* endpoint,
    const char* auth_token,
    uint64_t rpc_deadline_ms,
    const sgee_grpc_tls_options_t* tls,
    sgee_grpc_client_t** out_client);

__attribute__((weak)) uint64_t sgee_grpc_last_leader_hint(void);
}

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

// Validates options and normalizes endpoint list.
[[nodiscard]] auto validate_grpc_executor_options(
    const nimblecas::SgeeExecutorConfig& cfg,
    const nimblecas::SgeeGrpcExecutorOptions& opts) -> nimblecas::Result<std::vector<std::string>> {
    if (cfg.registry == nullptr || cfg.max_attempts == 0 || cfg.visibility_timeout_ms == 0) {
        return nimblecas::make_error<std::vector<std::string>>(nimblecas::MathError::domain_error);
    }

    // TLS validation: all-or-nothing (ca, cert, key must all be present if any TLS field is set)
    const bool has_ca = !opts.tls.ca_cert_path.empty();
    const bool has_cert = !opts.tls.cert_path.empty();
    const bool has_key = !opts.tls.key_path.empty();
    const bool has_override = !opts.tls.target_name_override.empty();
    const bool any_tls = has_ca || has_cert || has_key || has_override;
    const bool all_tls = has_ca && has_cert && has_key;
    if (any_tls && !all_tls) {
        return nimblecas::make_error<std::vector<std::string>>(nimblecas::MathError::domain_error);
    }

    // Endpoints validation
    std::vector<std::string> effective_endpoints;
    if (!opts.endpoints.empty()) {
        for (const auto& ep : opts.endpoints) {
            if (ep.empty()) {
                return nimblecas::make_error<std::vector<std::string>>(nimblecas::MathError::domain_error);
            }
            effective_endpoints.push_back(ep);
        }
    } else if (!opts.endpoint.empty()) {
        effective_endpoints.push_back(opts.endpoint);
    } else {
        return nimblecas::make_error<std::vector<std::string>>(nimblecas::MathError::domain_error);
    }

    return effective_endpoints;
}

// Opens clients for each endpoint and builds a GrpcEndpointRing.
[[nodiscard]] auto open_grpc_endpoint_ring(
    const std::vector<std::string>& endpoints,
    const std::string& auth_token,
    std::uint64_t rpc_deadline_ms,
    const nimblecas::SgeeGrpcTlsOptions& tls) -> nimblecas::Result<std::shared_ptr<nimblecas::GrpcEndpointRing>> {
    if (endpoints.empty()) {
        return nimblecas::make_error<std::shared_ptr<nimblecas::GrpcEndpointRing>>(
            nimblecas::MathError::domain_error);
    }

    const bool use_tls = !tls.ca_cert_path.empty();
    sgee_grpc_tls_options_t tls_c{
        .ca_cert_pem_path = tls.ca_cert_path.c_str(),
        .cert_pem_path = tls.cert_path.c_str(),
        .key_pem_path = tls.key_path.c_str(),
        .target_name_override = tls.target_name_override.empty() ? nullptr : tls.target_name_override.c_str()
    };

    auto ring = std::make_shared<nimblecas::GrpcEndpointRing>();
    ring->nodes.reserve(endpoints.size());

    for (const auto& ep : endpoints) {
        if (ep.empty()) {
            return nimblecas::make_error<std::shared_ptr<nimblecas::GrpcEndpointRing>>(
                nimblecas::MathError::domain_error);
        }
        sgee_grpc_client_t* client = nullptr;
        int rc = SGEE_OK;
        if (use_tls) {
            if (&sgee_grpc_client_open_tls != nullptr) {
                rc = sgee_grpc_client_open_tls(
                    ep.c_str(),
                    auth_token.empty() ? nullptr : auth_token.c_str(),
                    rpc_deadline_ms,
                    &tls_c,
                    &client);
            } else {
                return nimblecas::make_error<std::shared_ptr<nimblecas::GrpcEndpointRing>>(
                    nimblecas::MathError::not_implemented);
            }
        } else {
            rc = sgee_grpc_client_open(
                ep.c_str(),
                auth_token.empty() ? nullptr : auth_token.c_str(),
                rpc_deadline_ms,
                &client);
        }
        if (rc != SGEE_OK || client == nullptr) {
            return nimblecas::make_error<std::shared_ptr<nimblecas::GrpcEndpointRing>>(
                nimblecas::MathError::distributed_error);
        }
        ring->nodes.push_back({.endpoint = ep, .client = client});
    }

    return ring;
}

// How long a transport call keeps trying to find a leader before giving up honestly.
//
// This MUST be able to span a full Raft election, and the previous count-based budget could not:
// 3*n attempts with a 100ms pause every n gave ~200ms on a 3-node ring, while an election at the
// 80ms base timeout used by the cluster tests routinely runs longer under load. A leader is simply
// ABSENT for the duration of an election, so a budget shorter than one turns a routine, survivable
// failover into an aborted run — a spurious distributed_error for a cluster that was working.
// SGEE's own server sizes its await budget as 4*(2*election + heartbeat) for exactly this reason.
//
// Deadline-based rather than attempt-based: what matters is wall-clock coverage of the election,
// not a number of RPCs, since attempts against a dead node fail fast and would burn a count budget
// in milliseconds. On expiry the call still fails honestly with distributed_error — this widens the
// window in which recovery is possible, it never fabricates or papers over a result.
inline constexpr auto k_transport_retry_budget = std::chrono::seconds(5);
inline constexpr auto k_retry_sweep_pause = std::chrono::milliseconds(50);

// SGEE C ABI transport retry constants:
// - SGEE_ERR_CONSENSUS_BASE (9): follower refusing a write (NotLeader), defined in sgee_capi.h
// - SGEE_ERR_GRPC_TRANSPORT (-20): RPC never completed (unreachable / deadline / cancelled), defined in sgee_capi_grpc.h
//
// These static assertions pin the expected ABI error codes so that any future header change
// breaks the build loudly at compile time instead of silently degrading leader-following / failover at runtime.
static_assert(SGEE_ERR_CONSENSUS_BASE == 9,
              "SGEE_ERR_CONSENSUS_BASE value drifted; expected 9 for Raft follower NotLeader rejection");
static_assert(SGEE_ERR_GRPC_TRANSPORT == -20,
              "SGEE_ERR_GRPC_TRANSPORT value drifted; expected -20 for gRPC transport failure");

// Returns true if the return code indicates a transient / retryable transport condition:
// - SGEE_ERR_CONSENSUS_BASE: follower refusal (NotLeader) -> rotate to new leader
// - SGEE_ERR_GRPC_TRANSPORT: transport failure (unreachable, deadline, cancelled) -> retry / failover
[[nodiscard]] constexpr auto is_retryable_transport_rc(int rc) noexcept -> bool {
    return rc == SGEE_ERR_CONSENSUS_BASE || rc == SGEE_ERR_GRPC_TRANSPORT;
}

// Advances the ring active_index on a retryable error (SGEE_ERR_CONSENSUS_BASE or SGEE_ERR_GRPC_TRANSPORT).
auto rotate_ring_on_error(nimblecas::GrpcEndpointRing& ring, std::size_t current_idx) -> void {
    const std::size_t n = ring.nodes.size();
    if (n <= 1) {
        return;
    }
    std::size_t next_idx = (current_idx + 1) % n;
    if (&sgee_grpc_last_leader_hint != nullptr) {
        const std::uint64_t hint = sgee_grpc_last_leader_hint();
        if (hint >= 1 && hint <= n) {
            next_idx = static_cast<std::size_t>(hint - 1);
            if (next_idx == current_idx) {
                next_idx = (current_idx + 1) % n;
            }
        }
    }
    std::size_t expected = current_idx;
    if (ring.active_index.compare_exchange_strong(expected, next_idx)) {
        ring.failover_rotations.fetch_add(1, std::memory_order_relaxed);
    }
}

}  // namespace

namespace nimblecas {

// ---------------------------------------------------------------------------
// GrpcEndpointRing
// ---------------------------------------------------------------------------

GrpcEndpointRing::~GrpcEndpointRing() {
    for (auto& node : nodes) {
        if (node.client != nullptr) {
            sgee_grpc_client_destroy(node.client);
            node.client = nullptr;
        }
    }
}

// ---------------------------------------------------------------------------
// GrpcBrokerPort
// ---------------------------------------------------------------------------

auto GrpcBrokerPort::connect(std::string endpoint, std::string auth_token,
                             std::uint64_t rpc_deadline_ms)
    -> Result<std::unique_ptr<GrpcBrokerPort>> {
    if (endpoint.empty()) {
        return make_error<std::unique_ptr<GrpcBrokerPort>>(MathError::domain_error);
    }
    return connect(std::vector<std::string>{std::move(endpoint)}, std::move(auth_token),
                   rpc_deadline_ms, SgeeGrpcTlsOptions{});
}

auto GrpcBrokerPort::connect(std::vector<std::string> endpoints,
                             std::string auth_token,
                             std::uint64_t rpc_deadline_ms,
                             const SgeeGrpcTlsOptions& tls)
    -> Result<std::unique_ptr<GrpcBrokerPort>> {
    auto ring_res = open_grpc_endpoint_ring(endpoints, auth_token, rpc_deadline_ms, tls);
    if (!ring_res.has_value()) {
        return make_error<std::unique_ptr<GrpcBrokerPort>>(ring_res.error());
    }
    return std::make_unique<GrpcBrokerPort>(std::move(*ring_res));
}

GrpcBrokerPort::GrpcBrokerPort(sgee_grpc_client_t* client) {
    if (client != nullptr) {
        ring_ = std::make_shared<GrpcEndpointRing>();
        ring_->nodes.push_back({.endpoint = "", .client = client});
    }
}

GrpcBrokerPort::GrpcBrokerPort(std::shared_ptr<GrpcEndpointRing> ring)
    : ring_(std::move(ring)) {}

GrpcBrokerPort::~GrpcBrokerPort() = default;

auto GrpcBrokerPort::is_open() const noexcept -> bool {
    std::lock_guard lock(mutex_);
    return ring_ != nullptr && !ring_->nodes.empty() && ring_->nodes[0].client != nullptr;
}

auto GrpcBrokerPort::failover_rotations() const noexcept -> std::uint64_t {
    std::lock_guard lock(mutex_);
    return ring_ ? ring_->failover_rotations.load(std::memory_order_relaxed) : 0;
}

auto GrpcBrokerPort::enqueue(std::span<const std::byte> payload,
                             SgeePlacement placement, std::uint32_t max_attempts)
    -> Result<std::uint64_t> {
    std::lock_guard lock(mutex_);
    if (!ring_ || ring_->nodes.empty()) {
        return make_error<std::uint64_t>(MathError::distributed_error);
    }
    const std::size_t n = ring_->nodes.size();
    const auto retry_deadline = std::chrono::steady_clock::now() + k_transport_retry_budget;

    for (std::size_t attempt = 0;; ++attempt) {
        const std::size_t idx = ring_->active_index.load(std::memory_order_relaxed) % n;
        sgee_grpc_client_t* client = ring_->nodes[idx].client;
        if (!client) {
            return make_error<std::uint64_t>(MathError::distributed_error);
        }
        std::uint64_t out_id = 0;
        const int rc = sgee_grpc_enqueue(client, payload.data(), payload.size(),
                                         static_cast<int>(placement), max_attempts, &out_id);
        if (rc == SGEE_OK) {
            return out_id;
        }
        if (!is_retryable_transport_rc(rc)) {
            return make_error<std::uint64_t>(MathError::distributed_error);
        }
        // Retryable
        rotate_ring_on_error(*ring_, idx);
        if (std::chrono::steady_clock::now() >= retry_deadline) {
            break;
        }
        // Pause once per full sweep of the ring, so a wedged cluster is not hot-spun.
        if ((attempt + 1) % n == 0) {
            std::this_thread::sleep_for(k_retry_sweep_pause);
        }
    }
    return make_error<std::uint64_t>(MathError::distributed_error);
}

auto GrpcBrokerPort::lease(std::uint64_t worker_id, std::uint64_t timeout_ms)
    -> Result<std::optional<Lease>> {
    std::lock_guard lock(mutex_);
    if (!ring_ || ring_->nodes.empty()) {
        return make_error<std::optional<Lease>>(MathError::distributed_error);
    }
    const std::size_t n = ring_->nodes.size();
    const auto retry_deadline = std::chrono::steady_clock::now() + k_transport_retry_budget;

    for (std::size_t attempt = 0;; ++attempt) {
        const std::size_t idx = ring_->active_index.load(std::memory_order_relaxed) % n;
        sgee_grpc_client_t* client = ring_->nodes[idx].client;
        if (!client) {
            return make_error<std::optional<Lease>>(MathError::distributed_error);
        }

        std::uint64_t qid = 0;
        std::uint64_t token_local = 0;
        std::uint64_t token_term = 0;
        std::uint64_t token_index = 0;
        int placement = 0;
        std::uint32_t attempt_num = 0;
        void* out_payload = nullptr;
        std::size_t out_payload_len = 0;

        const int rc = sgee_grpc_lease(client, worker_id, timeout_ms, &qid, &token_local,
                                       &token_term, &token_index, &placement, &attempt_num,
                                       &out_payload, &out_payload_len);
        GrpcBufferGuard payload_guard{out_payload};

        if (rc == SGEE_ERR_QUEUE_EMPTY) {
            return std::optional<Lease>{std::nullopt};
        }
        if (rc == SGEE_OK) {
            Payload payload_bytes;
            if (out_payload != nullptr && out_payload_len > 0) {
                const auto* ptr = static_cast<const std::byte*>(out_payload);
                payload_bytes.assign(ptr, ptr + out_payload_len);
            }
            leases_[qid] = TokenTriple{
                .local = token_local,
                .term = token_term,
                .index = token_index
            };
            return std::optional<Lease>{Lease{
                .qid = qid,
                .token = token_local,
                .attempt = attempt_num,
                .payload = std::move(payload_bytes)
            }};
        }
        if (!is_retryable_transport_rc(rc)) {
            return make_error<std::optional<Lease>>(MathError::distributed_error);
        }
        // Retryable
        rotate_ring_on_error(*ring_, idx);
        if (std::chrono::steady_clock::now() >= retry_deadline) {
            break;
        }
        // Pause once per full sweep of the ring, so a wedged cluster is not hot-spun.
        if ((attempt + 1) % n == 0) {
            std::this_thread::sleep_for(k_retry_sweep_pause);
        }
    }
    return make_error<std::optional<Lease>>(MathError::distributed_error);
}

auto GrpcBrokerPort::complete(std::uint64_t qid, std::uint64_t token)
    -> Result<void> {
    std::lock_guard lock(mutex_);
    if (!ring_ || ring_->nodes.empty()) {
        return make_error<void>(MathError::distributed_error);
    }
    const auto it = leases_.find(qid);
    if (it == leases_.end() || it->second.local != token) {
        return make_error<void>(MathError::distributed_error);
    }
    const auto triple = it->second;
    const std::size_t n = ring_->nodes.size();
    const auto retry_deadline = std::chrono::steady_clock::now() + k_transport_retry_budget;

    for (std::size_t attempt = 0;; ++attempt) {
        const std::size_t idx = ring_->active_index.load(std::memory_order_relaxed) % n;
        sgee_grpc_client_t* client = ring_->nodes[idx].client;
        if (!client) {
            leases_.erase(qid);
            return make_error<void>(MathError::distributed_error);
        }
        const int rc = sgee_grpc_complete(client, qid, triple.local, triple.term, triple.index);
        if (rc == SGEE_OK) {
            leases_.erase(qid);
            return {};
        }
        if (!is_retryable_transport_rc(rc)) {
            leases_.erase(qid);
            return make_error<void>(MathError::distributed_error);
        }
        // Retryable
        rotate_ring_on_error(*ring_, idx);
        if (std::chrono::steady_clock::now() >= retry_deadline) {
            break;
        }
        // Pause once per full sweep of the ring, so a wedged cluster is not hot-spun.
        if ((attempt + 1) % n == 0) {
            std::this_thread::sleep_for(k_retry_sweep_pause);
        }
    }
    leases_.erase(qid);
    return make_error<void>(MathError::distributed_error);
}

auto GrpcBrokerPort::fail(std::uint64_t qid, std::uint64_t token)
    -> Result<void> {
    std::lock_guard lock(mutex_);
    if (!ring_ || ring_->nodes.empty()) {
        return make_error<void>(MathError::distributed_error);
    }
    const auto it = leases_.find(qid);
    if (it == leases_.end() || it->second.local != token) {
        return make_error<void>(MathError::distributed_error);
    }
    const auto triple = it->second;
    const std::size_t n = ring_->nodes.size();
    const auto retry_deadline = std::chrono::steady_clock::now() + k_transport_retry_budget;

    for (std::size_t attempt = 0;; ++attempt) {
        const std::size_t idx = ring_->active_index.load(std::memory_order_relaxed) % n;
        sgee_grpc_client_t* client = ring_->nodes[idx].client;
        if (!client) {
            leases_.erase(qid);
            return make_error<void>(MathError::distributed_error);
        }
        const int rc = sgee_grpc_fail(client, qid, triple.local, triple.term, triple.index);
        if (rc == SGEE_OK) {
            leases_.erase(qid);
            return {};
        }
        if (!is_retryable_transport_rc(rc)) {
            leases_.erase(qid);
            return make_error<void>(MathError::distributed_error);
        }
        // Retryable
        rotate_ring_on_error(*ring_, idx);
        if (std::chrono::steady_clock::now() >= retry_deadline) {
            break;
        }
        // Pause once per full sweep of the ring, so a wedged cluster is not hot-spun.
        if ((attempt + 1) % n == 0) {
            std::this_thread::sleep_for(k_retry_sweep_pause);
        }
    }
    leases_.erase(qid);
    return make_error<void>(MathError::distributed_error);
}

auto GrpcBrokerPort::heartbeat(std::uint64_t qid, std::uint64_t token,
                               std::uint64_t extend_by_ms) -> Result<void> {
    std::lock_guard lock(mutex_);
    if (!ring_ || ring_->nodes.empty()) {
        return make_error<void>(MathError::distributed_error);
    }
    const auto it = leases_.find(qid);
    if (it == leases_.end() || it->second.local != token) {
        return make_error<void>(MathError::distributed_error);
    }
    const auto triple = it->second;
    const std::size_t n = ring_->nodes.size();
    const auto retry_deadline = std::chrono::steady_clock::now() + k_transport_retry_budget;

    for (std::size_t attempt = 0;; ++attempt) {
        const std::size_t idx = ring_->active_index.load(std::memory_order_relaxed) % n;
        sgee_grpc_client_t* client = ring_->nodes[idx].client;
        if (!client) {
            return make_error<void>(MathError::distributed_error);
        }
        const int rc = sgee_grpc_heartbeat(client, qid, triple.local, triple.term,
                                           triple.index, extend_by_ms);
        if (rc == SGEE_OK) {
            return {};
        }
        if (!is_retryable_transport_rc(rc)) {
            return make_error<void>(MathError::distributed_error);
        }
        // Retryable
        rotate_ring_on_error(*ring_, idx);
        if (std::chrono::steady_clock::now() >= retry_deadline) {
            break;
        }
        // Pause once per full sweep of the ring, so a wedged cluster is not hot-spun.
        if ((attempt + 1) % n == 0) {
            std::this_thread::sleep_for(k_retry_sweep_pause);
        }
    }
    return make_error<void>(MathError::distributed_error);
}

auto GrpcBrokerPort::sweep_expired(std::uint64_t /*now_ms*/)
    -> Result<std::size_t> {
    return std::size_t{0};
}

auto GrpcBrokerPort::state(std::uint64_t qid) -> Result<QState> {
    std::lock_guard lock(mutex_);
    if (!ring_ || ring_->nodes.empty()) {
        return make_error<QState>(MathError::distributed_error);
    }
    const std::size_t n = ring_->nodes.size();
    const auto retry_deadline = std::chrono::steady_clock::now() + k_transport_retry_budget;

    for (std::size_t attempt = 0;; ++attempt) {
        const std::size_t idx = ring_->active_index.load(std::memory_order_relaxed) % n;
        sgee_grpc_client_t* client = ring_->nodes[idx].client;
        if (!client) {
            return make_error<QState>(MathError::distributed_error);
        }
        int st = 0;
        std::uint32_t attempt_num = 0;
        const int rc = sgee_grpc_task_state(client, qid, &st, &attempt_num);
        if (rc == SGEE_OK) {
            switch (st) {
                case SGEE_STATE_PENDING:   return QState::pending;
                case SGEE_STATE_LEASED:    return QState::leased;
                case SGEE_STATE_COMPLETED: return QState::completed;
                case SGEE_STATE_DEAD:      return QState::dead;
                default:                   return make_error<QState>(MathError::distributed_error);
            }
        }
        if (!is_retryable_transport_rc(rc)) {
            return make_error<QState>(MathError::distributed_error);
        }
        // Retryable
        rotate_ring_on_error(*ring_, idx);
        if (std::chrono::steady_clock::now() >= retry_deadline) {
            break;
        }
        // Pause once per full sweep of the ring, so a wedged cluster is not hot-spun.
        if ((attempt + 1) % n == 0) {
            std::this_thread::sleep_for(k_retry_sweep_pause);
        }
    }
    return make_error<QState>(MathError::distributed_error);
}

auto GrpcBrokerPort::now_ms() const -> std::uint64_t {
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
    if (endpoint.empty()) {
        return make_error<std::unique_ptr<GrpcResultChannel>>(MathError::domain_error);
    }
    return connect(std::vector<std::string>{std::move(endpoint)}, consume_on_get,
                   std::move(auth_token), rpc_deadline_ms, SgeeGrpcTlsOptions{});
}

auto GrpcResultChannel::connect(std::vector<std::string> endpoints, bool consume_on_get,
                                std::string auth_token, std::uint64_t rpc_deadline_ms,
                                const SgeeGrpcTlsOptions& tls)
    -> Result<std::unique_ptr<GrpcResultChannel>> {
    auto ring_res = open_grpc_endpoint_ring(endpoints, auth_token, rpc_deadline_ms, tls);
    if (!ring_res.has_value()) {
        return make_error<std::unique_ptr<GrpcResultChannel>>(ring_res.error());
    }
    return std::make_unique<GrpcResultChannel>(std::move(*ring_res), consume_on_get);
}

GrpcResultChannel::GrpcResultChannel(sgee_grpc_client_t* client, bool consume_on_get)
    : consume_on_get_(consume_on_get) {
    if (client != nullptr) {
        ring_ = std::make_shared<GrpcEndpointRing>();
        ring_->nodes.push_back({.endpoint = "", .client = client});
    }
}

GrpcResultChannel::GrpcResultChannel(std::shared_ptr<GrpcEndpointRing> ring, bool consume_on_get)
    : ring_(std::move(ring)), consume_on_get_(consume_on_get) {}

GrpcResultChannel::~GrpcResultChannel() = default;

auto GrpcResultChannel::is_open() const noexcept -> bool {
    std::lock_guard lock(mutex_);
    return ring_ != nullptr && !ring_->nodes.empty() && ring_->nodes[0].client != nullptr;
}

auto GrpcResultChannel::put(std::uint64_t qid, Payload result_envelope) -> Result<void> {
    std::lock_guard lock(mutex_);
    if (!ring_ || ring_->nodes.empty()) {
        return make_error<void>(MathError::distributed_error);
    }
    const std::size_t n = ring_->nodes.size();
    const auto retry_deadline = std::chrono::steady_clock::now() + k_transport_retry_budget;

    for (std::size_t attempt = 0;; ++attempt) {
        const std::size_t idx = ring_->active_index.load(std::memory_order_relaxed) % n;
        sgee_grpc_client_t* client = ring_->nodes[idx].client;
        if (!client) {
            return make_error<void>(MathError::distributed_error);
        }
        const int rc = sgee_grpc_put_result(client, qid, 0, 0, 0,
                                            result_envelope.data(), result_envelope.size());
        if (rc == SGEE_OK) {
            return {};
        }
        if (!is_retryable_transport_rc(rc)) {
            return make_error<void>(MathError::distributed_error);
        }
        // Retryable
        rotate_ring_on_error(*ring_, idx);
        if (std::chrono::steady_clock::now() >= retry_deadline) {
            break;
        }
        // Pause once per full sweep of the ring, so a wedged cluster is not hot-spun.
        if ((attempt + 1) % n == 0) {
            std::this_thread::sleep_for(k_retry_sweep_pause);
        }
    }
    return make_error<void>(MathError::distributed_error);
}

auto GrpcResultChannel::try_get(std::uint64_t qid) const -> Result<std::optional<Payload>> {
    std::lock_guard lock(mutex_);
    if (!ring_ || ring_->nodes.empty()) {
        return make_error<std::optional<Payload>>(MathError::distributed_error);
    }
    const std::size_t n = ring_->nodes.size();
    const auto retry_deadline = std::chrono::steady_clock::now() + k_transport_retry_budget;

    for (std::size_t attempt = 0;; ++attempt) {
        const std::size_t idx = ring_->active_index.load(std::memory_order_relaxed) % n;
        sgee_grpc_client_t* client = ring_->nodes[idx].client;
        if (!client) {
            return make_error<std::optional<Payload>>(MathError::distributed_error);
        }
        int found = 0;
        void* out_bytes = nullptr;
        std::size_t out_len = 0;
        const int rc = sgee_grpc_get_result(client, qid, consume_on_get_ ? 1 : 0,
                                            &found, &out_bytes, &out_len);
        GrpcBufferGuard bytes_guard{out_bytes};
        if (rc == SGEE_OK) {
            if (!found) {
                return std::optional<Payload>{std::nullopt};
            }
            Payload bytes;
            if (out_bytes != nullptr && out_len > 0) {
                const auto* ptr = static_cast<const std::byte*>(out_bytes);
                bytes.assign(ptr, ptr + out_len);
            }
            return std::optional<Payload>{std::move(bytes)};
        }
        if (!is_retryable_transport_rc(rc)) {
            return make_error<std::optional<Payload>>(MathError::distributed_error);
        }
        // Retryable
        rotate_ring_on_error(*ring_, idx);
        if (std::chrono::steady_clock::now() >= retry_deadline) {
            break;
        }
        // Pause once per full sweep of the ring, so a wedged cluster is not hot-spun.
        if ((attempt + 1) % n == 0) {
            std::this_thread::sleep_for(k_retry_sweep_pause);
        }
    }
    return make_error<std::optional<Payload>>(MathError::distributed_error);
}

auto GrpcResultChannel::get(std::uint64_t qid) const -> Result<Payload> {
    auto res = try_get(qid);
    if (!res.has_value()) {
        return make_error<Payload>(res.error());
    }
    if (!res->has_value()) {
        return make_error<Payload>(MathError::distributed_error);
    }
    return std::move(**res);
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

auto sgee_grpc_distributed_executor(SgeeExecutorConfig cfg, SgeeGrpcExecutorOptions opts)
    -> Result<std::unique_ptr<Executor>> {
    auto endpoints_res = validate_grpc_executor_options(cfg, opts);
    if (!endpoints_res) {
        return make_error<std::unique_ptr<Executor>>(endpoints_res.error());
    }
    auto effective_endpoints = std::move(*endpoints_res);
    const bool multi_endpoint = effective_endpoints.size() > 1;
    if (multi_endpoint && cfg.max_result_recoveries == 0) {
        cfg.max_result_recoveries = 1;
    }
    const bool consume_on_get = !multi_endpoint;

    // Per-run transport: every run() opens a FRESH ring of ports + channels shared per run.
    RunTransportFactory make_transport =
        [endpoints = std::move(effective_endpoints), auth = opts.auth_token,
         deadline = opts.rpc_deadline_ms, tls = opts.tls, consume_on_get]() -> Result<RunTransport> {
            auto ring_res = open_grpc_endpoint_ring(endpoints, auth, deadline, tls);
            if (!ring_res) {
                return make_error<RunTransport>(ring_res.error());
            }
            auto ring = std::move(*ring_res);
            auto port = std::make_unique<GrpcBrokerPort>(ring);
            auto channel = std::make_unique<GrpcResultChannel>(ring, consume_on_get);

            RunTransport tr;
            tr.owned_port = std::move(port);
            tr.port = tr.owned_port.get();
            tr.owned_channel = std::move(channel);
            tr.channel = tr.owned_channel.get();
            return tr;
        };
    return std::make_unique<SgeeDistributedExecutor>(std::move(cfg), std::move(make_transport));
}

}  // namespace nimblecas
