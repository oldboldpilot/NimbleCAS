// NimbleCAS SGEE distributed taskdag backend module.
// @author Olumuyiwa Oluwasanmi

module;
#include <cassert>
#ifdef NIMBLECAS_SGEE
// Pull the SGEE C ABI into the GLOBAL module fragment (not the purview) so `sgee_task_broker_t`
// is the header's global-module type. A forward declaration inside the module purview would
// instead attach a DISTINCT module-owned `struct sgee_task_broker`, which the C functions in the
// .cpp (that see the real global type) cannot accept — "cannot convert ... incomplete type".
#include "sgee_capi.h"
#endif
#ifdef NIMBLECAS_SGEE_GRPC
// The cross-process gRPC C ABI, same GLOBAL-module-fragment rationale as above for
// `sgee_grpc_client_t`. Independent of NIMBLECAS_SGEE: this header brings the gRPC-client
// surface only, and includes sgee_capi.h itself (guarded) for the shared enums/helpers.
#include "sgee_capi_grpc.h"
#endif

export module nimblecas.taskdag_sgee;

import std;
import nimblecas.core;
import nimblecas.taskdag;
import nimblecas.taskdag_sched;
import nimblecas.memo_dist;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// Placement target values (mirrors SGEE PlacementTarget)
// ---------------------------------------------------------------------------
enum class SgeePlacement : std::uint8_t {
    direct = 0,
    cpu = 1,
    gpu = 2,
    docker = 3,
    distributed = 4,
    cloud = 5
};

}  // namespace nimblecas

export namespace nimblecas::sgee_bridge {

// Maximum task payload over the wire (64 MiB).
inline constexpr std::size_t k_max_task_payload_bytes = 64u * 1024u * 1024u;

// Envelope for task arguments sent from coordinator to worker.
struct TaskEnvelope {
    std::uint64_t registry_fp{0};
    std::string op_id;
    std::vector<Payload> args;

    [[nodiscard]] auto operator==(const TaskEnvelope&) const noexcept -> bool = default;
};

// Envelope for task execution results returned from worker to coordinator.
struct ResultEnvelope {
    enum class Status : std::uint8_t { ok = 0, math_error = 1, bridge_error = 2 };
    Status status{Status::ok};
    MathError math_err{MathError::division_by_zero};
    double seconds{0.0};
    Payload bytes;

    [[nodiscard]] auto operator==(const ResultEnvelope&) const noexcept -> bool = default;
};

// Codec API
[[nodiscard]] auto encode_task(const TaskEnvelope& env,
                               std::size_t max_bytes = k_max_task_payload_bytes)
    -> Result<Payload>;

[[nodiscard]] auto decode_task(std::span<const std::byte> bytes) -> Result<TaskEnvelope>;

[[nodiscard]] auto encode_result(const ResultEnvelope& env) -> Result<Payload>;

[[nodiscard]] auto decode_result(std::span<const std::byte> bytes) -> Result<ResultEnvelope>;

}  // namespace nimblecas::sgee_bridge

export namespace nimblecas {

// ---------------------------------------------------------------------------
// Result Channel interface & InMemory implementation
// ---------------------------------------------------------------------------
// Out-of-band result channel. NOTE: put() is NOT fencing-token-guarded — a stale worker whose
// lease expired can still overwrite a committed envelope for the same qid. This is safe only
// because ops are deterministic (the retry republishes identical bytes; the broker's fencing
// blocks the stale *commit*, so the coordinator only reads a qid it saw reach `completed`). The
// one field that can differ between attempts is `seconds` (wall-time). Token-fenced puts are an
// M3 concern.
class ResultChannel {
public:
    virtual ~ResultChannel() = default;

    [[nodiscard]] virtual auto put(std::uint64_t qid, Payload result_envelope)
        -> Result<void> = 0;

    [[nodiscard]] virtual auto get(std::uint64_t qid) const -> Result<Payload> = 0;

    [[nodiscard]] virtual auto try_get(std::uint64_t qid) const -> Result<std::optional<Payload>> {
        auto res = get(qid);
        if (res.has_value()) {
            return std::optional<Payload>{std::move(*res)};
        }
        return std::optional<Payload>{std::nullopt};
    }
};

class InMemoryResultChannel final : public ResultChannel {
public:
    InMemoryResultChannel() = default;

    [[nodiscard]] auto put(std::uint64_t qid, Payload result_envelope)
        -> Result<void> override {
        std::lock_guard lock(mutex_);
        storage_[qid] = std::move(result_envelope);
        return {};
    }

    [[nodiscard]] auto get(std::uint64_t qid) const -> Result<Payload> override {
        std::lock_guard lock(mutex_);
        const auto it = storage_.find(qid);
        if (it == storage_.end()) {
            return make_error<Payload>(MathError::distributed_error);
        }
        return it->second;
    }

    [[nodiscard]] auto try_get(std::uint64_t qid) const -> Result<std::optional<Payload>> override {
        std::lock_guard lock(mutex_);
        const auto it = storage_.find(qid);
        if (it == storage_.end()) {
            return std::optional<Payload>{std::nullopt};
        }
        return std::optional<Payload>{it->second};
    }

private:
    mutable std::mutex mutex_{};
    std::unordered_map<std::uint64_t, Payload> storage_{};
};

// ---------------------------------------------------------------------------
// BrokerPort interface
// ---------------------------------------------------------------------------
class BrokerPort {
public:
    virtual ~BrokerPort() = default;

    struct Lease {
        std::uint64_t qid{0};
        std::uint64_t token{0};
        std::uint32_t attempt{0};
        Payload payload;
    };

    enum class QState : std::uint8_t { pending, leased, completed, dead };

    [[nodiscard]] virtual auto enqueue(std::span<const std::byte> payload,
                                       SgeePlacement placement, std::uint32_t max_attempts)
        -> Result<std::uint64_t> = 0;

    [[nodiscard]] virtual auto lease(std::uint64_t worker_id, std::uint64_t timeout_ms)
        -> Result<std::optional<Lease>> = 0;

    [[nodiscard]] virtual auto complete(std::uint64_t qid, std::uint64_t token)
        -> Result<void> = 0;

    [[nodiscard]] virtual auto fail(std::uint64_t qid, std::uint64_t token)
        -> Result<void> = 0;

    [[nodiscard]] virtual auto heartbeat(std::uint64_t qid, std::uint64_t token,
                                         std::uint64_t extend_by_ms) -> Result<void> = 0;

    [[nodiscard]] virtual auto sweep_expired(std::uint64_t now_ms)
        -> Result<std::size_t> = 0;

    [[nodiscard]] virtual auto state(std::uint64_t qid) -> Result<QState> = 0;

    // "Now" in the port's OWN clock domain. The coordinator MUST source sweep time from here,
    // never from a wall clock: lease deadlines are set against this clock, so mixing a wall clock
    // into sweep_expired() would spuriously expire every in-flight lease (the fake clock starts at
    // 0; a real Capi port would return the same wall clock it sets deadlines from).
    [[nodiscard]] virtual auto now_ms() const -> std::uint64_t = 0;
};

// Deterministic in-memory FakeBrokerPort for testing without SGEE.
class FakeBrokerPort final : public BrokerPort {
public:
    using ClockFn = std::function<std::uint64_t()>;

    explicit FakeBrokerPort(ClockFn clock = nullptr)
        : clock_(clock ? std::move(clock) : [this]() { return current_time_ms_.load(); }) {}

    // current_time_ms_ is atomic so the default clock lambda can read it lock-free from any thread
    // (including now_ms() called inside a mutex-held lease()/heartbeat()) without racing the writers
    // below, and without the self-deadlock a mutex-taking clock would create.
    void set_time_ms(std::uint64_t now_ms) { current_time_ms_.store(now_ms); }

    void advance_time_ms(std::uint64_t delta_ms) { current_time_ms_.fetch_add(delta_ms); }

    [[nodiscard]] auto now_ms() const -> std::uint64_t override { return clock_(); }

    // -----------------------------------------------------------------------
    // M2 hardening hooks (additive; happy path bit-identical when no fault is
    // armed and no accessor is called). All take the existing mutex_.
    // -----------------------------------------------------------------------

    // Which BrokerPort entry point to fault. Values index fault_counters_.
    enum class FaultOp : std::uint8_t {
        enqueue = 0, lease = 1, complete = 2, fail = 3, heartbeat = 4, sweep = 5, state = 6
    };

    // Arms the port so the next `count` calls of `op` return
    // make_error(MathError::distributed_error) WITHOUT touching queue state (an
    // enqueue fault consumes no qid; a sweep fault expires nothing). Counts are
    // consumed under mutex_, so concurrent callers observe exactly `count`
    // failures in total. Additive: all counters default 0 => today's behavior.
    void inject_fault(FaultOp op, std::size_t count = 1) {
        std::lock_guard lock(mutex_);
        fault_counters_[static_cast<std::size_t>(op)] += count;
    }

    // Total tasks ever enqueued on this port. next_qid_ pre-increments below, so
    // qids are 1-based/sequential and next_qid_ IS the enqueue count.
    [[nodiscard]] auto enqueue_count() const -> std::uint64_t {
        std::lock_guard lock(mutex_);
        return next_qid_;
    }

    // Lease attempts consumed by qid so far (0 for an unknown qid).
    [[nodiscard]] auto attempts(std::uint64_t qid) const -> std::uint32_t {
        std::lock_guard lock(mutex_);
        const auto it = tasks_.find(qid);
        return it == tasks_.end() ? 0u : it->second.attempt;
    }

    // Number of successful heartbeat() calls for qid (0 if unknown/none).
    [[nodiscard]] auto heartbeat_count(std::uint64_t qid) const -> std::uint64_t {
        std::lock_guard lock(mutex_);
        const auto it = tasks_.find(qid);
        return it == tasks_.end() ? 0u : it->second.heartbeat_count;
    }

    // Current visibility deadline for qid in the port clock domain (0 if unknown
    // / never leased). Lets the heartbeat test assert extension without racing
    // the sweep.
    [[nodiscard]] auto visibility_deadline_ms(std::uint64_t qid) const -> std::uint64_t {
        std::lock_guard lock(mutex_);
        const auto it = tasks_.find(qid);
        return it == tasks_.end() ? 0u : it->second.visibility_deadline_ms;
    }

    [[nodiscard]] auto enqueue(std::span<const std::byte> payload,
                               SgeePlacement placement, std::uint32_t max_attempts)
        -> Result<std::uint64_t> override {
        std::lock_guard lock(mutex_);
        if (take_fault(FaultOp::enqueue)) {
            return make_error<std::uint64_t>(MathError::distributed_error);
        }
        const std::uint64_t qid = ++next_qid_;
        TaskEntry entry{
            .qid = qid,
            .payload = Payload(payload.begin(), payload.end()),
            .placement = placement,
            .attempt = 0,
            .max_attempts = max_attempts ? max_attempts : 3,
            .state = QState::pending
        };
        tasks_[qid] = std::move(entry);
        fifo_order_.push_back(qid);
        return qid;
    }

    [[nodiscard]] auto lease(std::uint64_t worker_id, std::uint64_t timeout_ms)
        -> Result<std::optional<Lease>> override {
        std::lock_guard lock(mutex_);
        // A lease fault is the ERROR branch, not nullopt: queue-empty is not a fault.
        if (take_fault(FaultOp::lease)) {
            return make_error<std::optional<Lease>>(MathError::distributed_error);
        }
        for (const std::uint64_t qid : fifo_order_) {
            auto it = tasks_.find(qid);
            if (it != tasks_.end() && it->second.state == QState::pending) {
                TaskEntry& entry = it->second;
                entry.attempt++;
                entry.state = QState::leased;
                entry.worker_id = worker_id;
                entry.fencing_token = ++next_token_;
                const std::uint64_t vis_timeout = timeout_ms ? timeout_ms : 30'000;
                entry.visibility_deadline_ms = now_ms() + vis_timeout;
                return std::optional<Lease>{Lease{
                    .qid = entry.qid,
                    .token = entry.fencing_token,
                    .attempt = entry.attempt,
                    .payload = entry.payload
                }};
            }
        }
        return std::optional<Lease>{std::nullopt};
    }

    [[nodiscard]] auto complete(std::uint64_t qid, std::uint64_t token)
        -> Result<void> override {
        std::lock_guard lock(mutex_);
        if (take_fault(FaultOp::complete)) {
            return make_error<void>(MathError::distributed_error);
        }
        auto it = tasks_.find(qid);
        if (it == tasks_.end()) {
            return make_error<void>(MathError::distributed_error);
        }
        if (it->second.state != QState::leased || it->second.fencing_token != token) {
            return make_error<void>(MathError::distributed_error);
        }
        it->second.state = QState::completed;
        return {};
    }

    [[nodiscard]] auto fail(std::uint64_t qid, std::uint64_t token)
        -> Result<void> override {
        std::lock_guard lock(mutex_);
        if (take_fault(FaultOp::fail)) {
            return make_error<void>(MathError::distributed_error);
        }
        auto it = tasks_.find(qid);
        if (it == tasks_.end()) {
            return make_error<void>(MathError::distributed_error);
        }
        if (it->second.state != QState::leased || it->second.fencing_token != token) {
            return make_error<void>(MathError::distributed_error);
        }
        if (it->second.attempt >= it->second.max_attempts) {
            it->second.state = QState::dead;
        } else {
            it->second.state = QState::pending;
        }
        return {};
    }

    [[nodiscard]] auto heartbeat(std::uint64_t qid, std::uint64_t token,
                                 std::uint64_t extend_by_ms) -> Result<void> override {
        std::lock_guard lock(mutex_);
        if (take_fault(FaultOp::heartbeat)) {
            return make_error<void>(MathError::distributed_error);
        }
        auto it = tasks_.find(qid);
        if (it == tasks_.end()) {
            return make_error<void>(MathError::distributed_error);
        }
        if (it->second.state != QState::leased || it->second.fencing_token != token) {
            return make_error<void>(MathError::distributed_error);
        }
        const std::uint64_t extend = extend_by_ms ? extend_by_ms : 10'000;
        it->second.visibility_deadline_ms = now_ms() + extend;
        ++it->second.heartbeat_count;
        return {};
    }

    [[nodiscard]] auto sweep_expired(std::uint64_t now_ms)
        -> Result<std::size_t> override {
        std::lock_guard lock(mutex_);
        if (take_fault(FaultOp::sweep)) {
            return make_error<std::size_t>(MathError::distributed_error);
        }
        std::size_t swept = 0;
        for (auto& [qid, entry] : tasks_) {
            if (entry.state == QState::leased && now_ms >= entry.visibility_deadline_ms) {
                if (entry.attempt >= entry.max_attempts) {
                    entry.state = QState::dead;
                } else {
                    entry.state = QState::pending;
                }
                swept++;
            }
        }
        return swept;
    }

    [[nodiscard]] auto state(std::uint64_t qid) -> Result<QState> override {
        std::lock_guard lock(mutex_);
        if (take_fault(FaultOp::state)) {
            return make_error<QState>(MathError::distributed_error);
        }
        auto it = tasks_.find(qid);
        if (it == tasks_.end()) {
            return make_error<QState>(MathError::distributed_error);
        }
        return it->second.state;
    }

    // Testing hook: forces a task's state directly
    void force_state(std::uint64_t qid, QState new_state) {
        std::lock_guard lock(mutex_);
        auto it = tasks_.find(qid);
        if (it != tasks_.end()) {
            it->second.state = new_state;
        }
    }

private:
    struct TaskEntry {
        std::uint64_t qid{0};
        Payload payload;
        SgeePlacement placement{SgeePlacement::cpu};
        std::uint32_t attempt{0};
        std::uint32_t max_attempts{3};
        QState state{QState::pending};
        std::uint64_t worker_id{0};
        std::uint64_t fencing_token{0};
        std::uint64_t visibility_deadline_ms{0};
        std::uint64_t heartbeat_count{0};
    };

    // Decrement-and-return-true when `op`'s counter is nonzero. MUST be called
    // with mutex_ held (every caller is inside a lock_guard).
    [[nodiscard]] auto take_fault(FaultOp op) -> bool {
        std::size_t& c = fault_counters_[static_cast<std::size_t>(op)];
        if (c == 0) { return false; }
        --c;
        return true;
    }

    ClockFn clock_{nullptr};
    std::atomic<std::uint64_t> current_time_ms_{0};
    mutable std::mutex mutex_{};
    std::uint64_t next_qid_{0};
    std::uint64_t next_token_{0};
    std::unordered_map<std::uint64_t, TaskEntry> tasks_{};
    std::vector<std::uint64_t> fifo_order_{};
    std::array<std::size_t, 7> fault_counters_{};
};

// ---------------------------------------------------------------------------
// Real SGEE C-ABI BrokerPort (NIMBLECAS_SGEE=ON)
// ---------------------------------------------------------------------------
#ifdef NIMBLECAS_SGEE
// sgee_task_broker_t comes from sgee_capi.h, included in the global module fragment above.
class CapiBrokerPort final : public BrokerPort {
public:
    [[nodiscard]] static auto create(const std::filesystem::path& wal_path,
                       std::uint64_t vis_timeout_ms = 30'000,
                       std::uint32_t max_attempts = 3)
        -> Result<std::unique_ptr<CapiBrokerPort>>;

    explicit CapiBrokerPort(sgee_task_broker_t* broker);
    CapiBrokerPort(const std::filesystem::path& wal_path,
                   std::uint64_t vis_timeout_ms = 30'000,
                   std::uint32_t max_attempts = 3);
    ~CapiBrokerPort() override;

    // Non-copyable AND non-movable: the port is only ever held by unique_ptr (create()/factory),
    // and worker/coordinator threads hold a live BrokerPort& into it. A move would either dangle
    // those references or leave a moved-from port silently answering every call with a transport
    // error, so it is forbidden outright rather than made safe.
    CapiBrokerPort(const CapiBrokerPort&) = delete;
    auto operator=(const CapiBrokerPort&) -> CapiBrokerPort& = delete;
    CapiBrokerPort(CapiBrokerPort&&) = delete;
    auto operator=(CapiBrokerPort&&) -> CapiBrokerPort& = delete;

    [[nodiscard]] auto is_open() const noexcept -> bool;

    [[nodiscard]] auto enqueue(std::span<const std::byte> payload,
                               SgeePlacement placement, std::uint32_t max_attempts)
        -> Result<std::uint64_t> override;

    [[nodiscard]] auto lease(std::uint64_t worker_id, std::uint64_t timeout_ms)
        -> Result<std::optional<Lease>> override;

    [[nodiscard]] auto complete(std::uint64_t qid, std::uint64_t token)
        -> Result<void> override;

    [[nodiscard]] auto fail(std::uint64_t qid, std::uint64_t token)
        -> Result<void> override;

    [[nodiscard]] auto heartbeat(std::uint64_t qid, std::uint64_t token,
                                 std::uint64_t extend_by_ms) -> Result<void> override;

    [[nodiscard]] auto sweep_expired(std::uint64_t now_ms)
        -> Result<std::size_t> override;

    [[nodiscard]] auto state(std::uint64_t qid) -> Result<QState> override;

    [[nodiscard]] auto now_ms() const -> std::uint64_t override;

private:
    sgee_task_broker_t* broker_{nullptr};
    mutable std::mutex mutex_{};
};
#endif

// ---------------------------------------------------------------------------
// Cross-process gRPC options and classes
// ---------------------------------------------------------------------------
struct SgeeGrpcTlsOptions {           // all-or-nothing
    std::string ca_cert_path{};
    std::string cert_path{};
    std::string key_path{};
    std::string target_name_override{};  // optional
};

struct SgeeGrpcExecutorOptions {
    std::string endpoint{};                 // UNCHANGED, still works alone
    std::string auth_token{};
    std::uint64_t rpc_deadline_ms{0};
    std::vector<std::string> endpoints{};   // NEW: when non-empty, supersedes `endpoint`
    SgeeGrpcTlsOptions tls{};               // NEW: empty == plaintext
};

#ifdef NIMBLECAS_SGEE_GRPC
// Shared endpoint ring holding one open sgee_grpc_client_t* per endpoint plus an atomic active index.
struct GrpcEndpointRing {
    struct Node {
        std::string endpoint;
        sgee_grpc_client_t* client{nullptr};
    };
    std::vector<Node> nodes{};
    std::atomic<std::size_t> active_index{0};
    std::atomic<std::uint64_t> failover_rotations{0};

    GrpcEndpointRing() = default;
    ~GrpcEndpointRing();

    GrpcEndpointRing(const GrpcEndpointRing&) = delete;
    auto operator=(const GrpcEndpointRing&) -> GrpcEndpointRing& = delete;
    GrpcEndpointRing(GrpcEndpointRing&&) = delete;
    auto operator=(GrpcEndpointRing&&) -> GrpcEndpointRing& = delete;
};

// sgee_grpc_client_t comes from sgee_capi_grpc.h, included in the global module fragment above.
// A BrokerPort backed by a remote sgee_queue_node over gRPC — the TRUE cross-process transport.
// Unlike CapiBrokerPort there is no local WAL: the queue lives in the server. sweep_expired is a
// no-op (the node's driver sweeps expired leases autonomously), and now_ms is inert (its only
// coordinator use is the sweep argument). The single-u64 BrokerPort token is the wire token's
// local half; the (term, index) halves are stashed per-qid and re-attached on
// complete/fail/heartbeat — safe because those are only ever called by the process that leased.
//
// SERIALIZATION: mutex_ is held across the whole RPC (a network call, up to rpc_deadline_ms). In
// the intended topology this is free — the coordinator port is driven by one thread and each
// worker PROCESS has its own port. The in-process-pump config (num_workers > 0) instead shares one
// port across N pump threads + their heartbeat threads, which then fully serialize their transport
// on this mutex; it is a convenience configuration, not a throughput path (use separate worker
// processes for parallelism). Correctness is unaffected either way (ops are deterministic, so a
// lease swept for slow-heartbeat starvation simply re-executes bit-identically).
class GrpcBrokerPort final : public BrokerPort {
public:
    // Connect to a single TaskQueue endpoint ("host:port"). NEVER blocks on the network (the gRPC
    // channel connects lazily); a wrong endpoint surfaces as distributed_error on the first RPC.
    [[nodiscard]] static auto connect(std::string endpoint, std::string auth_token = {},
                                      std::uint64_t rpc_deadline_ms = 0)
        -> Result<std::unique_ptr<GrpcBrokerPort>>;

    // Connect to multiple TaskQueue endpoints forming a Raft cluster ring with optional TLS.
    [[nodiscard]] static auto connect(std::vector<std::string> endpoints,
                                      std::string auth_token = {},
                                      std::uint64_t rpc_deadline_ms = 0,
                                      const SgeeGrpcTlsOptions& tls = {})
        -> Result<std::unique_ptr<GrpcBrokerPort>>;

    explicit GrpcBrokerPort(sgee_grpc_client_t* client);
    explicit GrpcBrokerPort(std::shared_ptr<GrpcEndpointRing> ring);
    ~GrpcBrokerPort() override;

    // Non-copyable AND non-movable — same rationale as CapiBrokerPort.
    GrpcBrokerPort(const GrpcBrokerPort&) = delete;
    auto operator=(const GrpcBrokerPort&) -> GrpcBrokerPort& = delete;
    GrpcBrokerPort(GrpcBrokerPort&&) = delete;
    auto operator=(GrpcBrokerPort&&) -> GrpcBrokerPort& = delete;

    [[nodiscard]] auto is_open() const noexcept -> bool;
    [[nodiscard]] auto failover_rotations() const noexcept -> std::uint64_t;

    [[nodiscard]] auto enqueue(std::span<const std::byte> payload,
                               SgeePlacement placement, std::uint32_t max_attempts)
        -> Result<std::uint64_t> override;

    [[nodiscard]] auto lease(std::uint64_t worker_id, std::uint64_t timeout_ms)
        -> Result<std::optional<Lease>> override;

    [[nodiscard]] auto complete(std::uint64_t qid, std::uint64_t token)
        -> Result<void> override;

    [[nodiscard]] auto fail(std::uint64_t qid, std::uint64_t token)
        -> Result<void> override;

    [[nodiscard]] auto heartbeat(std::uint64_t qid, std::uint64_t token,
                                 std::uint64_t extend_by_ms) -> Result<void> override;

    [[nodiscard]] auto sweep_expired(std::uint64_t now_ms)
        -> Result<std::size_t> override;

    [[nodiscard]] auto state(std::uint64_t qid) -> Result<QState> override;

    [[nodiscard]] auto now_ms() const -> std::uint64_t override;

    [[nodiscard]] auto ring() const noexcept -> std::shared_ptr<GrpcEndpointRing> {
        return ring_;
    }

private:
    struct TokenTriple {
        std::uint64_t local{0};
        std::uint64_t term{0};
        std::uint64_t index{0};
    };
    std::shared_ptr<GrpcEndpointRing> ring_{nullptr};
    mutable std::mutex mutex_{};
    // qid -> the wire token triple for the lease THIS port took (see the class comment).
    std::unordered_map<std::uint64_t, TokenTriple> leases_{};
};

// A ResultChannel over the same remote node's out-of-band result store. Holds its OWN client
// handle or shared ring. Coordinator side reads with consume_on_get (each qid once when
// single endpoint; consume=false when multi-node); worker side only ever puts.
class GrpcResultChannel final : public ResultChannel {
public:
    [[nodiscard]] static auto connect(std::string endpoint, bool consume_on_get,
                                      std::string auth_token = {},
                                      std::uint64_t rpc_deadline_ms = 0)
        -> Result<std::unique_ptr<GrpcResultChannel>>;

    [[nodiscard]] static auto connect(std::vector<std::string> endpoints, bool consume_on_get,
                                      std::string auth_token = {},
                                      std::uint64_t rpc_deadline_ms = 0,
                                      const SgeeGrpcTlsOptions& tls = {})
        -> Result<std::unique_ptr<GrpcResultChannel>>;

    GrpcResultChannel(sgee_grpc_client_t* client, bool consume_on_get);
    GrpcResultChannel(std::shared_ptr<GrpcEndpointRing> ring, bool consume_on_get);
    ~GrpcResultChannel() override;

    GrpcResultChannel(const GrpcResultChannel&) = delete;
    auto operator=(const GrpcResultChannel&) -> GrpcResultChannel& = delete;
    GrpcResultChannel(GrpcResultChannel&&) = delete;
    auto operator=(GrpcResultChannel&&) -> GrpcResultChannel& = delete;

    [[nodiscard]] auto is_open() const noexcept -> bool;

    [[nodiscard]] auto put(std::uint64_t qid, Payload result_envelope)
        -> Result<void> override;

    [[nodiscard]] auto get(std::uint64_t qid) const -> Result<Payload> override;

    [[nodiscard]] auto try_get(std::uint64_t qid) const
        -> Result<std::optional<Payload>> override;

    [[nodiscard]] auto ring() const noexcept -> std::shared_ptr<GrpcEndpointRing> {
        return ring_;
    }

private:
    std::shared_ptr<GrpcEndpointRing> ring_{nullptr};
    bool consume_on_get_{true};
    mutable std::mutex mutex_{};
};
#endif

// ---------------------------------------------------------------------------
// Worker pump
// ---------------------------------------------------------------------------
struct WorkerPumpConfig {
    std::uint64_t worker_id{1};
    std::uint64_t lease_timeout_ms{0};   // 0 = default (30000)
    std::uint64_t idle_backoff_ms{2};    // sleep when QueueEmpty
    std::uint64_t heartbeat_every_ms{0}; // 0 = visibility / 3
};

auto run_worker_pump(BrokerPort& port, const TaskRegistry& reg, ResultChannel& results,
                     WorkerPumpConfig cfg, std::stop_token stop) -> void;

// ---------------------------------------------------------------------------
// SgeeDistributedExecutor
// ---------------------------------------------------------------------------
struct SgeeExecutorConfig {
    std::filesystem::path wal_dir{};
    const TaskRegistry* registry{nullptr};
    std::uint64_t visibility_timeout_ms{30'000};
    std::uint32_t max_attempts{3};
    std::size_t num_workers{1};
    std::uint64_t poll_interval_ms{2};
    std::uint64_t run_deadline_ms{0};
    std::size_t max_result_recoveries{0};
    std::function<SgeePlacement(const TaskGraph&, TaskId)> placement{};
    bool cost_ordering{false};
    ScheduleParams schedule_params{};
    const CostTable* cost_table{nullptr};

    // ── Content-addressed memoization (ROADMAP §6.2 item 2) ──────────────────────────
    // Both default OFF, so an executor configured as before behaves exactly as before.
    //
    // `memo`, when set, is consulted before every enqueue and published to after every
    // result: a task whose canonical bytes are already known is never dispatched. The
    // table outlives the run, so it memoizes ACROSS runs and (for a shared implementation)
    // across the cluster. The caller owns it; it must outlive the executor.
    //
    // `dedup_identical_tasks` needs no EXTERNAL table: it collapses tasks that are
    // byte-identical to one ALREADY SEEN IN THIS RUN, dispatching the first and copying its
    // output to the rest. That is the redundancy a DAG creates on its own. It is not free,
    // though -- it keeps a run-wide index holding the full encoded bytes of every distinct
    // dispatched task until the run ends, and those bytes embed complete parent outputs. On a
    // graph with large intermediates that is real memory the plain path releases at each level
    // boundary, and unlike InProcessMemo the index is unbounded.
    //
    // Both preserve `outputs` bit-for-bit -- a skipped task is a pure function of bytes
    // that were computed anyway. Two things are NOT preserved, and must not be: `executed`
    // counts tasks actually dispatched, which is precisely the quantity these options reduce,
    // and `measured_seconds` is 0.0 for every hit and every collapsed duplicate, because
    // nothing ran and a borrowed duration would be fictional work in a timing field.
    //
    // Neither can fail a run. A memo lookup or publish that reports failure, and an entry that
    // will not decode, are all treated as misses.
    DistributedMemo* memo{nullptr};
    bool dedup_identical_tasks{false};

    auto with_wal_dir(std::filesystem::path p) -> SgeeExecutorConfig& {
        wal_dir = std::move(p);
        return *this;
    }

    auto with_registry(const TaskRegistry& r) -> SgeeExecutorConfig& {
        registry = &r;
        return *this;
    }

    auto with_memo(DistributedMemo& m) -> SgeeExecutorConfig& {
        memo = &m;
        return *this;
    }

    auto with_dedup_identical_tasks(bool on) -> SgeeExecutorConfig& {
        dedup_identical_tasks = on;
        return *this;
    }

    auto with_visibility_timeout_ms(std::uint64_t ms) -> SgeeExecutorConfig& {
        visibility_timeout_ms = ms;
        return *this;
    }

    auto with_max_attempts(std::uint32_t a) -> SgeeExecutorConfig& {
        max_attempts = a;
        return *this;
    }

    auto with_num_workers(std::size_t n) -> SgeeExecutorConfig& {
        num_workers = n;
        return *this;
    }

    auto with_poll_interval_ms(std::uint64_t ms) -> SgeeExecutorConfig& {
        poll_interval_ms = ms;
        return *this;
    }

    auto with_run_deadline_ms(std::uint64_t ms) -> SgeeExecutorConfig& {
        run_deadline_ms = ms;
        return *this;
    }

    auto with_max_result_recoveries(std::size_t n) -> SgeeExecutorConfig& {
        max_result_recoveries = n;
        return *this;
    }

    auto with_placement(std::function<SgeePlacement(const TaskGraph&, TaskId)> f)
        -> SgeeExecutorConfig& {
        placement = std::move(f);
        return *this;
    }

    auto with_cost_ordering(bool enable) -> SgeeExecutorConfig& {
        cost_ordering = enable;
        return *this;
    }

    auto with_schedule_params(ScheduleParams p) -> SgeeExecutorConfig& {
        schedule_params = p;
        return *this;
    }

    auto with_cost_table(const CostTable* t) -> SgeeExecutorConfig& {
        cost_table = t;
        return *this;
    }
};

// One run()'s transport: the broker and result channel the coordinator and its pump
// threads use for exactly one execution, plus the WAL file to reap on success.
//
// `port`/`channel` are the non-owning views run() actually uses; `owned_*` keep
// per-run instances alive (a borrowed transport -- e.g. a test's FakeBrokerPort that
// must outlive the run for inspection -- leaves them null). `wal_path` empty means
// "nothing on disk to reap" (every Fake-backed transport). Raw view pointers stay
// valid across a move because moving a unique_ptr never relocates the pointee.
struct RunTransport {
    BrokerPort* port{nullptr};
    ResultChannel* channel{nullptr};
    std::unique_ptr<BrokerPort> owned_port{};
    std::unique_ptr<ResultChannel> owned_channel{};
    std::filesystem::path wal_path{};
};

// Invoked once per run(). Returning an error aborts the run with that error before any
// task is enqueued. A factory that opens durable state (a WAL) MUST return a FRESH
// instance per call -- reusing a WAL would durably replay a previous run's tasks into
// this one (broker recovery aimed at the wrong target).
using RunTransportFactory = std::function<Result<RunTransport>()>;

class SgeeDistributedExecutor final : public Executor {
public:
    // Borrowed-transport ctor (unchanged signature & caller contract: `port` and
    // `results` must outlive this executor). Every run() reuses the same borrowed pair
    // -- the caller owns queue hygiene across runs. This is the ctor the deterministic
    // test suite uses so it can inspect/fault the port directly.
    SgeeDistributedExecutor(SgeeExecutorConfig cfg, BrokerPort& port, ResultChannel& results)
        : cfg_(std::move(cfg)),
          make_transport_([&port, &results]() -> Result<RunTransport> {
              return RunTransport{.port = &port, .channel = &results};
          }) {}

    // Per-run-transport ctor: `make_transport` is called at the top of every run(); the
    // transport lives exactly as long as that run. On run success a non-empty wal_path
    // is deleted; on failure it is retained for post-mortem.
    SgeeDistributedExecutor(SgeeExecutorConfig cfg, RunTransportFactory make_transport)
        : cfg_(std::move(cfg)), make_transport_(std::move(make_transport)) {}

    [[nodiscard]] auto name() const noexcept -> std::string_view override {
        return "sgee_distributed";
    }

    [[nodiscard]] auto run(const TaskGraph& g) -> Result<TaskRunResult> override;

private:
    SgeeExecutorConfig cfg_;
    RunTransportFactory make_transport_;
};

// ---------------------------------------------------------------------------
// Affinity -> placement (ROADMAP §6.1 "Hardware Affinities")
// ---------------------------------------------------------------------------
//
// Builds the `SgeeExecutorConfig::placement` function from a frozen per-op AffinityTable:
// `cpu_only` -> cpu, `gpu_only` -> gpu, `hybrid` -> cpu. A task whose op id is absent from the
// table, and any UNNAMED (closure) task -- which has no op id to look up -- takes `fallback`.
//
// PRECONDITION: every TaskId passed to the returned function must have been ISSUED BY the
// TaskGraph passed alongside it. The function forwards to TaskGraph::op_id, which is guarded
// by an assert only, and the canonical flags define NDEBUG -- so in a shipping build an
// out-of-range id, or an id belonging to a different graph, is an out-of-bounds read yielding
// a string_view over garbage and hence a silently wrong label. The executor's own call sites
// always pass its own graph; this matters because the returned function is public API and is
// designed to outlive the expression that built it.
//
// The table is COPIED into the returned closure, not referenced. A placement function commonly
// outlives the expression that built it, and a dangling AffinityTable would be a silent wrong
// routing rather than a crash. Affinity tables are small; the copy is the cheap side of that trade.
//
// ── HONESTY BOUNDARY (Rule 32) ───────────────────────────────────────────────────────────
// THIS IS HALF A MECHANISM, AND TURNING IT ON CHANGES NOTHING TODAY. It labels every task
// correctly and the ABI carries one `int placement` per task from enqueue through to lease. But the
// label never actually reaches a worker: `CapiBrokerPort::lease` and `GrpcBrokerPort::lease`
// both DISCARD `out_placement`, and `BrokerPort::Lease` has no field to hold it, so it dies
// inside the port. Nor could a worker use it if it had it: `sgee_broker_lease`
// hands out the earliest pending task and accepts no placement filter, and the CAPI port drops
// the `out_placement` it is given. So a GPU-only task is still leased by whichever worker asks
// first.
//
// It is built and tested now because it is the piece that lives in this repo, and because
// `Affinity`/`to_placement` were otherwise dead code reachable only from unit tests. It becomes
// real routing the moment the lease side can filter -- see docs/technical/sgee-locality-abi.md
// for exactly what that requires. Until then, do not read a placement label as a guarantee about
// where a task ran.
[[nodiscard]] auto affinity_placement(const AffinityTable& table,
                                      SgeePlacement fallback = SgeePlacement::cpu)
    -> std::function<SgeePlacement(const TaskGraph&, TaskId)>;

// Factory (NIMBLECAS_SGEE=ON). Validates config (invalid -> domain_error) and returns an executor
// whose every run() opens a FRESH per-run CapiBrokerPort on a process-unique WAL, deleted on success
// and retained on failure. The factory does NOT open a broker itself: broker-open failure surfaces
// at run() time as distributed_error, and an executor is safe to reuse after a failed run. With
// NIMBLECAS_SGEE=OFF the stub honestly returns not_implemented for a valid config.
[[nodiscard]] auto sgee_distributed_executor(SgeeExecutorConfig cfg)
    -> Result<std::unique_ptr<Executor>>;

// ---------------------------------------------------------------------------
// Cross-process gRPC executor factory (NIMBLECAS_SGEE_GRPC=ON)
// ---------------------------------------------------------------------------
// Factory for the TRUE cross-process backend: every run() opens a FRESH GrpcBrokerPort +
// GrpcResultChannel pair (independent connections) to `opts.endpoint`. Validates cfg the same
// way as sgee_distributed_executor EXCEPT: `wal_dir` is IGNORED (the queue lives in the server's
// SGEE_DATA_DIR) and `num_workers == 0` is VALID — it means "no in-process pumps; work is done by
// separate worker processes attached to the same endpoint". num_workers > 0 also runs in-process
// pumps against the remote broker (a useful halfway configuration). The RunTransport carries no
// local WAL, so nothing is reaped on success. With NIMBLECAS_SGEE_GRPC=OFF the stub honestly
// returns not_implemented for a valid config.
[[nodiscard]] auto sgee_grpc_distributed_executor(SgeeExecutorConfig cfg,
                                                  SgeeGrpcExecutorOptions opts)
    -> Result<std::unique_ptr<Executor>>;

}  // namespace nimblecas

// ===========================================================================
// Implementation of Codec, Worker Pump, and SgeeDistributedExecutor::run
// ===========================================================================
namespace nimblecas {

namespace {

inline auto write_u16_le(std::uint16_t val, std::vector<std::byte>& out) -> void {
    out.push_back(static_cast<std::byte>(val & 0xFF));
    out.push_back(static_cast<std::byte>((val >> 8) & 0xFF));
}

inline auto write_u32_le(std::uint32_t val, std::vector<std::byte>& out) -> void {
    out.push_back(static_cast<std::byte>(val & 0xFF));
    out.push_back(static_cast<std::byte>((val >> 8) & 0xFF));
    out.push_back(static_cast<std::byte>((val >> 16) & 0xFF));
    out.push_back(static_cast<std::byte>((val >> 24) & 0xFF));
}

inline auto write_u64_le(std::uint64_t val, std::vector<std::byte>& out) -> void {
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<std::byte>((val >> (i * 8)) & 0xFF));
    }
}

inline auto read_u16_le(std::span<const std::byte> bytes, std::size_t offset) -> std::uint16_t {
    return static_cast<std::uint16_t>(bytes[offset]) |
           (static_cast<std::uint16_t>(bytes[offset + 1]) << 8);
}

inline auto read_u32_le(std::span<const std::byte> bytes, std::size_t offset) -> std::uint32_t {
    return static_cast<std::uint32_t>(bytes[offset]) |
           (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
           (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

inline auto read_u64_le(std::span<const std::byte> bytes, std::size_t offset) -> std::uint64_t {
    std::uint64_t val = 0;
    for (int i = 0; i < 8; ++i) {
        val |= (static_cast<std::uint64_t>(bytes[offset + i]) << (i * 8));
    }
    return val;
}

inline auto write_f64_le(double val, std::vector<std::byte>& out) -> void {
    const auto u = std::bit_cast<std::uint64_t>(val);
    write_u64_le(u, out);
}

inline auto read_f64_le(std::span<const std::byte> bytes, std::size_t offset) -> double {
    const auto u = read_u64_le(bytes, offset);
    return std::bit_cast<double>(u);
}

constexpr std::uint32_t k_task_magic = 0x4E434454;   // "NCDT" in LE
constexpr std::uint32_t k_result_magic = 0x4E435254; // "NCRT" in LE

}  // namespace

auto affinity_placement(const AffinityTable& table, SgeePlacement fallback)
    -> std::function<SgeePlacement(const TaskGraph&, TaskId)> {
    return [table, fallback](const TaskGraph& g, TaskId id) -> SgeePlacement {
        const std::string_view op = g.op_id(id);
        if (op.empty()) {
            // An unnamed closure task carries no op id, so there is nothing to look up.
            return fallback;
        }
        const auto it = table.find(op);
        if (it == table.end()) {
            return fallback;
        }
        // to_placement returns a raw uint8_t so that taskdag_sched need not import this module
        // (the dependency runs the other way). Map it back here, and treat any code this enum
        // does not name as unroutable rather than casting blindly into an invalid enumerator.
        switch (to_placement(it->second)) {
            case 0: return SgeePlacement::direct;
            case 1: return SgeePlacement::cpu;
            case 2: return SgeePlacement::gpu;
            case 3: return SgeePlacement::docker;
            case 4: return SgeePlacement::distributed;
            case 5: return SgeePlacement::cloud;
            default: return fallback;
        }
    };
}

namespace sgee_bridge {

auto encode_task(const TaskEnvelope& env, std::size_t max_bytes) -> Result<Payload> {
    if (env.op_id.empty() || env.op_id.size() > 256) {
        return make_error<Payload>(MathError::syntax_error);
    }
    // Accumulate the framed size with no size_t wrap: compare each addend against the headroom
    // left under max_bytes rather than summing everything first and checking after (a wrapped total
    // could otherwise slip under the cap and mis-size the reserve).
    std::size_t total_size = 4 + 2 + 2 + 8 + 4 + env.op_id.size() + 4;  // header + op + n_args
    if (total_size > max_bytes || env.args.size() > (max_bytes - total_size) / 8) {
        return make_error<Payload>(MathError::overflow);
    }
    total_size += env.args.size() * 8;  // the per-arg length table
    for (const auto& arg : env.args) {
        if (arg.size() > max_bytes - total_size) {
            return make_error<Payload>(MathError::overflow);
        }
        total_size += arg.size();
    }

    Payload out;
    out.reserve(total_size);

    write_u32_le(k_task_magic, out);
    write_u16_le(1, out); // version 1
    write_u16_le(0, out); // reserved
    write_u64_le(env.registry_fp, out);
    write_u32_le(static_cast<std::uint32_t>(env.op_id.size()), out);

    for (const char c : env.op_id) {
        out.push_back(static_cast<std::byte>(c));
    }

    write_u32_le(static_cast<std::uint32_t>(env.args.size()), out);
    for (const auto& arg : env.args) {
        write_u64_le(static_cast<std::uint64_t>(arg.size()), out);
    }
    for (const auto& arg : env.args) {
        out.insert(out.end(), arg.begin(), arg.end());
    }

    return out;
}

auto decode_task(std::span<const std::byte> bytes) -> Result<TaskEnvelope> {
    if (bytes.size() < 20 || bytes.size() > k_max_task_payload_bytes) {
        return make_error<TaskEnvelope>(MathError::syntax_error);
    }
    const std::uint32_t magic = read_u32_le(bytes, 0);
    const std::uint16_t version = read_u16_le(bytes, 4);
    if (magic != k_task_magic || version != 1) {
        return make_error<TaskEnvelope>(MathError::syntax_error);
    }

    const std::uint64_t fp = read_u64_le(bytes, 8);
    const std::uint32_t op_len = read_u32_le(bytes, 16);

    if (op_len < 1 || op_len > 256 || bytes.size() < 20 + op_len + 4) {
        return make_error<TaskEnvelope>(MathError::syntax_error);
    }

    std::string op_id;
    op_id.reserve(op_len);
    for (std::size_t i = 0; i < op_len; ++i) {
        const char c = static_cast<char>(bytes[20 + i]);
        if (static_cast<unsigned char>(c) >= 128 || c == '\0') {
            return make_error<TaskEnvelope>(MathError::syntax_error);
        }
        op_id.push_back(c);
    }

    std::size_t offset = 20 + op_len;
    const std::uint32_t n_args = read_u32_le(bytes, offset);
    offset += 4;

    if (bytes.size() < offset + static_cast<std::size_t>(n_args) * 8) {
        return make_error<TaskEnvelope>(MathError::syntax_error);
    }

    std::vector<std::uint64_t> arg_lens(n_args);
    for (std::uint32_t i = 0; i < n_args; ++i) {
        arg_lens[i] = read_u64_le(bytes, offset + static_cast<std::size_t>(i) * 8);
    }
    offset += static_cast<std::size_t>(n_args) * 8;  // 64-bit; the table was bounds-checked above

    // Bounds-check the declared arg lengths against the bytes still unconsumed, with NO summation
    // that could wrap size_t: a crafted length (or a pair summing to 2^64) must be rejected, never
    // allowed to reach subspan() as an out-of-contract count -> OOB read. `remaining - consumed`
    // never underflows because we only advance `consumed` when the length fits.
    const std::size_t remaining = bytes.size() - offset;
    std::size_t consumed = 0;
    for (std::uint32_t i = 0; i < n_args; ++i) {
        if (arg_lens[i] > remaining - consumed) {
            return make_error<TaskEnvelope>(MathError::syntax_error);
        }
        consumed += static_cast<std::size_t>(arg_lens[i]);
    }
    if (consumed != remaining) {
        return make_error<TaskEnvelope>(MathError::syntax_error);
    }

    std::vector<Payload> args;
    args.reserve(n_args);
    for (std::uint32_t i = 0; i < n_args; ++i) {
        const std::size_t len = static_cast<std::size_t>(arg_lens[i]);
        args.emplace_back(bytes.subspan(offset, len).begin(), bytes.subspan(offset, len).end());
        offset += len;
    }

    return TaskEnvelope{
        .registry_fp = fp,
        .op_id = std::move(op_id),
        .args = std::move(args)
    };
}

auto encode_result(const ResultEnvelope& env) -> Result<Payload> {
    Payload out;
    const std::size_t len = (env.status == ResultEnvelope::Status::ok) ? env.bytes.size() : 0;
    out.reserve(24 + len);

    write_u32_le(k_result_magic, out);
    write_u16_le(1, out); // version 1
    out.push_back(static_cast<std::byte>(env.status));
    out.push_back(static_cast<std::byte>((env.status == ResultEnvelope::Status::math_error)
                                             ? env.math_err
                                             : MathError::division_by_zero));
    write_f64_le(env.seconds, out);
    write_u64_le(static_cast<std::uint64_t>(len), out);

    if (len > 0) {
        out.insert(out.end(), env.bytes.begin(),
                   std::next(env.bytes.begin(), static_cast<std::ptrdiff_t>(len)));
    }
    return out;
}

auto decode_result(std::span<const std::byte> bytes) -> Result<ResultEnvelope> {
    if (bytes.size() < 24) {
        return make_error<ResultEnvelope>(MathError::syntax_error);
    }
    const std::uint32_t magic = read_u32_le(bytes, 0);
    const std::uint16_t version = read_u16_le(bytes, 4);
    if (magic != k_result_magic || version != 1) {
        return make_error<ResultEnvelope>(MathError::syntax_error);
    }

    const auto status_raw = static_cast<std::uint8_t>(bytes[6]);
    if (status_raw > 2) {
        return make_error<ResultEnvelope>(MathError::syntax_error);
    }
    const auto math_err_raw = static_cast<std::uint8_t>(bytes[7]);
    // A corrupt result must be rejected as transport corruption, never smuggled in as a math error
    // that does not exist: reject a math_err byte outside the enum, and (for a non-ok status) a
    // spurious payload length. distributed_error is the last enumerator.
    if (status_raw == 1 &&
        math_err_raw > static_cast<std::uint8_t>(MathError::distributed_error)) {
        return make_error<ResultEnvelope>(MathError::syntax_error);
    }
    const double seconds = read_f64_le(bytes, 8);
    const std::uint64_t len = read_u64_le(bytes, 16);

    // encode_result writes len == 0 for EVERY non-ok status, so a non-ok envelope carrying
    // payload bytes was not produced by this codec. The comment above always claimed this
    // check; it now exists, because a memo makes foreign-authored envelopes a reachable input
    // to this decoder rather than only ever bytes a trusted worker just wrote.
    if (status_raw != 0 && len != 0) {
        return make_error<ResultEnvelope>(MathError::syntax_error);
    }

    // No-wrap length check: bytes.size() >= 24 is established above, so bytes.size() - 24 is safe.
    if (len != bytes.size() - 24) {
        return make_error<ResultEnvelope>(MathError::syntax_error);
    }

    Payload payload;
    if (len > 0) {
        payload.assign(bytes.begin() + 24, bytes.end());
    }

    return ResultEnvelope{
        .status = static_cast<ResultEnvelope::Status>(status_raw),
        .math_err = static_cast<MathError>(math_err_raw),
        .seconds = seconds,
        .bytes = std::move(payload)
    };
}

}  // namespace sgee_bridge

auto run_worker_pump(BrokerPort& port, const TaskRegistry& reg, ResultChannel& results,
                     WorkerPumpConfig cfg, std::stop_token stop) -> void {
    while (!stop.stop_requested()) {
        auto lease_res = port.lease(cfg.worker_id, cfg.lease_timeout_ms);
        if (!lease_res.has_value()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(cfg.idle_backoff_ms));
            continue;
        }
        auto lease_opt = std::move(*lease_res);
        if (!lease_opt.has_value()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(cfg.idle_backoff_ms));
            continue;
        }
        const auto& lease = *lease_opt;

        std::uint64_t hb_ms = cfg.heartbeat_every_ms;
        if (hb_ms == 0) {
            const std::uint64_t timeout = cfg.lease_timeout_ms ? cfg.lease_timeout_ms : 30'000;
            hb_ms = std::max<std::uint64_t>(1000, timeout / 3);
        }

        struct HeartbeatGuard {
            BrokerPort& port;
            std::uint64_t qid;
            std::uint64_t token;
            std::uint64_t interval_ms;
            std::mutex mtx;
            std::condition_variable cv;
            bool active{true};
            std::thread thread;

            HeartbeatGuard(BrokerPort& p, std::uint64_t q, std::uint64_t tok, std::uint64_t interval)
                : port(p), qid(q), token(tok), interval_ms(interval) {
                thread = std::thread([this]() {
                    std::unique_lock lock(mtx);
                    // INTERRUPTIBLE wait: wake the instant the destructor clears `active` instead of
                    // sleeping out the whole interval. An uninterruptible sleep_for here would stall
                    // the pump for a full heartbeat interval (~10s at default timeouts) after every
                    // task, throttling throughput to ~1 task/interval and blowing the run deadline on
                    // deeper graphs. Heartbeat only on a genuine interval timeout.
                    while (active) {
                        if (cv.wait_for(lock, std::chrono::milliseconds(interval_ms),
                                        [this] { return !active; })) {
                            break;  // active cleared -> stop promptly
                        }
                        lock.unlock();
                        (void)port.heartbeat(qid, token, 0);
                        lock.lock();
                    }
                });
            }

            ~HeartbeatGuard() {
                {
                    std::lock_guard lock(mtx);
                    active = false;
                }
                cv.notify_all();
                if (thread.joinable()) {
                    thread.join();
                }
            }
        };

        HeartbeatGuard hb(port, lease.qid, lease.token, hb_ms);

        auto task_env_res = sgee_bridge::decode_task(lease.payload);
        if (!task_env_res.has_value()) {
            sgee_bridge::ResultEnvelope err_env{
                .status = sgee_bridge::ResultEnvelope::Status::bridge_error,
                .math_err = MathError::division_by_zero,
                .seconds = 0.0,
                .bytes = {}
            };
            auto enc_res = sgee_bridge::encode_result(err_env);
            if (enc_res.has_value()) {
                (void)results.put(lease.qid, std::move(*enc_res));
            }
            (void)port.complete(lease.qid, lease.token);
            continue;
        }

        const auto& env = *task_env_res;

        if (env.registry_fp != reg.fingerprint()) {
            sgee_bridge::ResultEnvelope err_env{
                .status = sgee_bridge::ResultEnvelope::Status::bridge_error,
                .math_err = MathError::division_by_zero,
                .seconds = 0.0,
                .bytes = {}
            };
            auto enc_res = sgee_bridge::encode_result(err_env);
            if (enc_res.has_value()) {
                (void)results.put(lease.qid, std::move(*enc_res));
            }
            (void)port.complete(lease.qid, lease.token);
            continue;
        }

        const TaskFn* fn = reg.find(env.op_id);
        if (fn == nullptr) {
            sgee_bridge::ResultEnvelope err_env{
                .status = sgee_bridge::ResultEnvelope::Status::bridge_error,
                .math_err = MathError::division_by_zero,
                .seconds = 0.0,
                .bytes = {}
            };
            auto enc_res = sgee_bridge::encode_result(err_env);
            if (enc_res.has_value()) {
                (void)results.put(lease.qid, std::move(*enc_res));
            }
            (void)port.complete(lease.qid, lease.token);
            continue;
        }

        const auto t0 = std::chrono::steady_clock::now();
        auto r = (*fn)(std::span<const Payload>(env.args));
        const auto t1 = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(t1 - t0).count();

        sgee_bridge::ResultEnvelope res_env{};
        if (r.has_value()) {
            res_env.status = sgee_bridge::ResultEnvelope::Status::ok;
            res_env.seconds = seconds;
            res_env.bytes = std::move(*r);
        } else {
            res_env.status = sgee_bridge::ResultEnvelope::Status::math_error;
            res_env.math_err = r.error();
            res_env.seconds = seconds;
        }

        auto enc_res = sgee_bridge::encode_result(res_env);
        if (!enc_res.has_value()) {
            (void)port.fail(lease.qid, lease.token);
            continue;
        }

        auto put_res = results.put(lease.qid, std::move(*enc_res));
        if (!put_res.has_value()) {
            (void)port.fail(lease.qid, lease.token);
            continue;
        }

        (void)port.complete(lease.qid, lease.token);
    }
}

auto SgeeDistributedExecutor::run(const TaskGraph& g) -> Result<TaskRunResult> {
    if (cfg_.registry == nullptr) {
        return make_error<TaskRunResult>(MathError::domain_error);
    }
    const std::size_t n = g.size();
    if (n == 0) {
        return TaskRunResult{};
    }

    // Pre-flight check: every task must be named & registered
    for (std::size_t i = 0; i < n; ++i) {
        const TaskId id{i};
        const std::string_view op = g.op_id(id);
        if (op.empty()) {
            return make_error<TaskRunResult>(MathError::not_implemented);
        }
        if (cfg_.registry->find(op) == nullptr) {
            return make_error<TaskRunResult>(MathError::domain_error);
        }
    }

    // Acquire this run's transport: a fresh broker + channel + WAL for a per-run
    // factory, or the borrowed pair for the reference ctor. Done AFTER the pre-flight
    // so an invalid graph never opens a broker or creates a WAL. An empty factory (a
    // misuse of the public per-run ctor) aborts honestly rather than throwing
    // std::bad_function_call out of this Result-returning API (Rule 32).
    if (!make_transport_) {
        return make_error<TaskRunResult>(MathError::distributed_error);
    }
    auto transport_res = make_transport_();
    if (!transport_res.has_value()) {
        return make_error<TaskRunResult>(transport_res.error());
    }
    RunTransport transport = std::move(*transport_res);
    if (transport.port == nullptr || transport.channel == nullptr) {
        return make_error<TaskRunResult>(MathError::distributed_error);
    }
    BrokerPort& port = *transport.port;
    ResultChannel& results = *transport.channel;

    // RAII teardown; declaration order is load-bearing. `reaper` is declared FIRST so
    // it is destroyed LAST: the pump threads (joined by PumpGuard) hold BrokerPort&/
    // ResultChannel& into the transport and MUST stop before the reaper releases the
    // owned broker/channel. This replaces the per-return-site cleanup() calls -- a
    // future early return can no longer forget to join a pump or leak a WAL decision.
    struct TransportReaper {
        RunTransport& tr;
        bool succeeded{false};
        ~TransportReaper() {
            // Only an OWNED transport (a per-run factory) has a WAL to reap; a borrowed
            // transport (owned_port == null) must never have its caller's file removed,
            // even if a contract-violating factory left wal_path non-empty. Latch before reset.
            const bool owned = tr.owned_port != nullptr;
            tr.owned_port.reset();      // closes the broker => releases the WAL file
            tr.owned_channel.reset();
            if (succeeded && owned && !tr.wal_path.empty()) {
                std::error_code ec;
                std::filesystem::remove(tr.wal_path, ec);  // best-effort on success
            }                            // failure: file retained for post-mortem
        }
    };
    struct PumpGuard {
        std::stop_source stop{};
        std::vector<std::thread> threads{};
        ~PumpGuard() {
            stop.request_stop();
            for (auto& t : threads) {
                if (t.joinable()) { t.join(); }
            }
        }
    };
    TransportReaper reaper{transport};
    PumpGuard pumps;
    if (cfg_.num_workers > 0) {
        pumps.threads.reserve(cfg_.num_workers);
        for (std::size_t w = 0; w < cfg_.num_workers; ++w) {
            WorkerPumpConfig pump_cfg{
                .worker_id = w + 1,
                .lease_timeout_ms = cfg_.visibility_timeout_ms,
                .idle_backoff_ms = cfg_.poll_interval_ms,
                .heartbeat_every_ms = 0
            };
            pumps.threads.emplace_back(
                [&port, &results, reg = cfg_.registry, pump_cfg,
                 token = pumps.stop.get_token()]() {
                    run_worker_pump(port, *reg, results, pump_cfg, token);
                });
        }
    }

    std::vector<Result<Payload>> outputs(n);
    std::vector<std::optional<std::size_t>> origins(n);
    std::vector<double> seconds(n, 0.0);
    std::size_t executed = 0;

    std::uint64_t run_deadline_ms = cfg_.run_deadline_ms;
    if (run_deadline_ms == 0) {
        double total_cost = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            total_cost += std::max(0.0, g.hint(TaskId{i}).mean_seconds);
        }
        // Clamp BEFORE the cast. total_cost is caller-supplied via CostHint, so an absurd or
        // infinite hint would otherwise make this double->uint64 conversion undefined — a bad
        // estimate must at worst give a bad deadline, never UB.
        constexpr double k_max_deadline_ms = 30.0 * 24.0 * 60.0 * 60.0 * 1000.0;  // 30 days
        const double derived_ms = 10.0 * total_cost * 1000.0;
        const double clamped_ms = (std::isfinite(derived_ms) && derived_ms > 0.0)
                                      ? std::min(derived_ms, k_max_deadline_ms)
                                      : 0.0;
        run_deadline_ms = std::max<std::uint64_t>(60'000, static_cast<std::uint64_t>(clamped_ms));
    }

    // Run-wide index of tasks already dispatched, keyed by ContentKey and holding EVERY
    // candidate under that fingerprint together with its full encoded bytes. The vector is
    // not paranoia: two different tasks may share a ContentKey, and collapsing them would
    // return one task's answer for another. A candidate counts only on a full byte match.
    std::map<ContentKey, std::vector<std::pair<TaskId, Payload>>> run_index;

    // Resolve a byte-identical task already dispatched in this run, or nullopt.
    const auto find_identical_task =
        [&run_index](const ContentKey& key, const Payload& encoded) -> std::optional<TaskId> {
        const auto it = run_index.find(key);
        if (it == run_index.end()) {
            return std::nullopt;
        }
        for (const auto& [tid, bytes] : it->second) {
            if (bytes == encoded) {
                return tid;
            }
        }
        return std::nullopt;
    };

    const auto start_time = std::chrono::steady_clock::now();

    for (std::size_t lvl = 0; lvl < g.num_levels(); ++lvl) {
        const std::span<const TaskId> raw_level = g.level(lvl);
        const std::vector<TaskId> ordered_tasks =
            cfg_.cost_ordering
                ? level_order(g, raw_level, cfg_.cost_table, cfg_.schedule_params)
                : std::vector<TaskId>{};
        const std::span<const TaskId> level_tasks =
            cfg_.cost_ordering ? std::span<const TaskId>(ordered_tasks) : raw_level;
        struct PendingTaskInfo {
            TaskId id;
            Payload encoded_payload;
            std::size_t recoveries{0};
        };
        std::unordered_map<std::uint64_t, PendingTaskInfo> pending_tasks;

        // Tasks collapsed onto a twin still IN FLIGHT this level: their outputs cannot be
        // copied until the level drains, so record them and settle up afterwards.
        std::vector<std::pair<TaskId, TaskId>> pending_aliases;

        // Which tasks this level has dispatched but not yet collected. This CANNOT be
        // inferred from `outputs`: Result<Payload> default-constructs to a VALID empty
        // payload, so an untouched slot and a task that legitimately returned no bytes are
        // indistinguishable there. Copying an unresolved slot would hand a twin an empty
        // answer and call it a result. Levels below this one are always fully drained, so
        // membership here is the exact test for "still in flight".
        std::unordered_set<std::size_t> dispatched_this_level;

        for (const TaskId id : level_tasks) {
            // Check poisoning
            std::optional<std::size_t> best_origin;
            for (const TaskId d : g.deps(id)) {
                if (!outputs[d.value].has_value()) {
                    const std::size_t origin = origins[d.value].value();
                    if (!best_origin.has_value() || origin < *best_origin) {
                        best_origin = origin;
                    }
                }
            }

            if (best_origin.has_value()) {
                outputs[id.value] = make_error<Payload>(outputs[*best_origin].error());
                origins[id.value] = best_origin;
                seconds[id.value] = 0.0;
                continue;
            }

            // Gather inputs (bound literals first, then parent outputs in declared dependency order)
            const auto lits = g.literals(id);
            const auto task_deps = g.deps(id);
            std::vector<Payload> args;
            args.reserve(lits.size() + task_deps.size());
            for (const Payload& lit : lits) {
                args.push_back(lit);
            }
            for (const TaskId d : task_deps) {
                args.push_back(outputs[d.value].value());
            }

            sgee_bridge::TaskEnvelope env{
                .registry_fp = cfg_.registry->fingerprint(),
                .op_id = std::string(g.op_id(id)),
                .args = std::move(args)
            };

            auto enc_res = sgee_bridge::encode_task(env);
            if (!enc_res.has_value()) {
                return make_error<TaskRunResult>(MathError::distributed_error);
            }

            // ── Content-addressed short-circuits, in cost order: the run index is a map
            // lookup, the memo may be a network round trip, and dispatch is the whole
            // cluster. Try them in that order and never pay for a later one needlessly.
            //
            // The guard is not a micro-optimisation. content_key is two multiplies and two
            // XORs per byte, serially dependent and unvectorizable, over an encoded task that
            // may reach k_max_task_payload_bytes. Computing it unconditionally would tax every
            // existing default-configured run -- which is meant to behave exactly as before in
            // cost, not only in output -- to produce a value nothing would read.
            const bool content_addressing_on =
                cfg_.dedup_identical_tasks || cfg_.memo != nullptr;
            const ContentKey key =
                content_addressing_on ? content_key(*enc_res) : ContentKey{};

            if (cfg_.dedup_identical_tasks) {
                if (const auto twin = find_identical_task(key, *enc_res)) {
                    // Byte-identical to a task already dispatched this run. If its output
                    // is already settled, take it now; otherwise it is in flight in THIS
                    // level, so defer until the level drains.
                    if (!dispatched_this_level.contains(twin->value)) {
                        outputs[id.value] = outputs[twin->value];
                        // A task that fails IS its own failure origin -- that is what a local
                        // executor records for the same computation, and matching it keeps
                        // descendants' poisoning identical to the single-node run.
                        origins[id.value] =
                            outputs[id.value].has_value() ? std::nullopt
                                                          : std::optional<std::size_t>{id.value};
                    } else {
                        pending_aliases.emplace_back(id, *twin);
                    }
                    seconds[id.value] = 0.0;
                    continue;
                }
            }

            if (cfg_.memo != nullptr) {
                // A memo that reports its own failure is treated as a miss, exactly like a
                // corrupt entry below: turning ON a CACHE must never turn a run that would
                // have succeeded into one that fails. A network-backed table having a bad
                // moment would otherwise abort a whole distributed run, and a bad write by
                // another process into a shared table would abort every run after it.
                auto hit_res = cfg_.memo->lookup(key, *enc_res);

                // An entry that will not decode, or that carries a status that is never
                // published (only ok and math_error are -- see is_memoizable_status), means the
                // TABLE is wrong, not that the task failed. Treat it as a miss and dispatch.
                //
                // Deliberately NOT an abort. A miss is by construction observationally
                // identical to a hit, so recomputing is exactly as honest and strictly more
                // available: aborting would let one malformed entry -- which any writer to a
                // shared table can create -- permanently fail every future run of every graph
                // containing that task, with no way to purge it. An accelerator must not be able
                // to do that. The recomputation shows up in `executed`.
                bool served_from_memo = false;
                if (hit_res.has_value() && hit_res->has_value()) {
                    const auto dec_res = sgee_bridge::decode_result(**hit_res);
                    if (dec_res.has_value()) {
                        const auto& memo_env = *dec_res;
                        if (memo_env.status == sgee_bridge::ResultEnvelope::Status::ok) {
                            outputs[id.value] = memo_env.bytes;
                            origins[id.value] = std::nullopt;
                            served_from_memo = true;
                        } else if (memo_env.status ==
                                   sgee_bridge::ResultEnvelope::Status::math_error) {
                            outputs[id.value] = make_error<Payload>(memo_env.math_err);
                            origins[id.value] = id.value;
                            served_from_memo = true;
                        }
                    }
                }

                if (served_from_memo) {
                    // 0.0, not the stored duration: nothing ran here, and reporting a borrowed
                    // time would put fictional work into measured_seconds.
                    seconds[id.value] = 0.0;
                    // Settled without dispatch: index it (but NOT as in-flight) so a later twin
                    // costs a map lookup rather than a second trip to the memo.
                    if (cfg_.dedup_identical_tasks) {
                        run_index[key].emplace_back(id, *enc_res);
                    }
                    continue;
                }
            }

            if (cfg_.dedup_identical_tasks) {
                run_index[key].emplace_back(id, *enc_res);
                dispatched_this_level.insert(id.value);
            }

            const SgeePlacement placement = cfg_.placement ? cfg_.placement(g, id) : SgeePlacement::cpu;
            auto enqueue_res = port.enqueue(*enc_res, placement, cfg_.max_attempts);
            if (!enqueue_res.has_value()) {
                return make_error<TaskRunResult>(MathError::distributed_error);
            }

            pending_tasks[*enqueue_res] = PendingTaskInfo{
                .id = id,
                .encoded_payload = std::move(*enc_res),
                .recoveries = 0
            };
        }

        // Await resolution of this level
        while (!pending_tasks.empty()) {
            const auto elapsed_ms = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start_time).count());
            if (elapsed_ms > run_deadline_ms) {
                return make_error<TaskRunResult>(MathError::distributed_error);
            }

            // Sweep in the broker's OWN clock domain (never a wall clock): lease deadlines were set
            // against port.now_ms(), so sourcing "now" from anywhere else would spuriously expire
            // every in-flight lease. A sweep failure is a transport fault -> honest whole-run abort.
            const auto sweep_res = port.sweep_expired(port.now_ms());
            if (!sweep_res.has_value()) {
                return make_error<TaskRunResult>(MathError::distributed_error);
            }

            std::vector<std::uint64_t> resolved_qids;
            std::vector<std::pair<std::uint64_t, PendingTaskInfo>> re_enqueued_tasks;

            for (const auto& [qid, info] : pending_tasks) {
                auto st_res = port.state(qid);
                if (!st_res.has_value()) {
                    return make_error<TaskRunResult>(MathError::distributed_error);
                }
                const BrokerPort::QState st = *st_res;
                if (st == BrokerPort::QState::completed) {
                    auto res_bytes_res = results.try_get(qid);
                    if (!res_bytes_res.has_value()) {
                        return make_error<TaskRunResult>(MathError::distributed_error);
                    }
                    if (!res_bytes_res->has_value()) {
                        // Genuinely absent result (found == false)!
                        if (info.recoveries < cfg_.max_result_recoveries) {
                            const SgeePlacement placement =
                                cfg_.placement ? cfg_.placement(g, info.id) : SgeePlacement::cpu;
                            auto re_enqueue_res =
                                port.enqueue(info.encoded_payload, placement, cfg_.max_attempts);
                            if (!re_enqueue_res.has_value()) {
                                return make_error<TaskRunResult>(MathError::distributed_error);
                            }
                            resolved_qids.push_back(qid);
                            re_enqueued_tasks.emplace_back(
                                *re_enqueue_res,
                                PendingTaskInfo{
                                    .id = info.id,
                                    .encoded_payload = info.encoded_payload,
                                    .recoveries = info.recoveries + 1
                                });
                            continue;
                        } else {
                            // Recoveries exhausted or bound == 0: honest abort!
                            return make_error<TaskRunResult>(MathError::distributed_error);
                        }
                    }

                    const Payload& res_bytes = **res_bytes_res;
                    auto dec_res = sgee_bridge::decode_result(res_bytes);
                    if (!dec_res.has_value()) {
                        return make_error<TaskRunResult>(MathError::distributed_error);
                    }
                    const auto& res_env = *dec_res;

                    // Publish before applying. Only a result that is a deterministic FUNCTION
                    // OF THE KEY may be cached: is_memoizable_status admits ok and math_error,
                    // and refuses bridge_error, which reports the state of the cluster at one
                    // moment rather than a property of the input. Caching that would make one
                    // transient fault permanent for every future lookup of these bytes.
                    if (cfg_.memo != nullptr &&
                        is_memoizable_status(static_cast<int>(res_env.status))) {
                        const ContentKey pub_key = content_key(info.encoded_payload);
                        // Discarded deliberately. Publishing is an optimisation for some LATER
                        // run; a table that refuses the write must not throw away results this
                        // run has already collected.
                        (void)cfg_.memo->publish(pub_key, info.encoded_payload, res_bytes);
                    }

                    if (res_env.status == sgee_bridge::ResultEnvelope::Status::ok) {
                        outputs[info.id.value] = res_env.bytes;
                        origins[info.id.value] = std::nullopt;
                        seconds[info.id.value] = res_env.seconds;
                        ++executed;
                        resolved_qids.push_back(qid);
                    } else if (res_env.status == sgee_bridge::ResultEnvelope::Status::math_error) {
                        outputs[info.id.value] = make_error<Payload>(res_env.math_err);
                        origins[info.id.value] = info.id.value;
                        seconds[info.id.value] = res_env.seconds;
                        ++executed;
                        resolved_qids.push_back(qid);
                    } else {
                        return make_error<TaskRunResult>(MathError::distributed_error);
                    }
                } else if (st == BrokerPort::QState::dead) {
                    return make_error<TaskRunResult>(MathError::distributed_error);
                }
            }

            for (const std::uint64_t qid : resolved_qids) {
                pending_tasks.erase(qid);
            }
            for (auto& [new_qid, new_info] : re_enqueued_tasks) {
                pending_tasks[new_qid] = std::move(new_info);
            }

            if (!pending_tasks.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(cfg_.poll_interval_ms));
            }
        }

        // The level has drained, so every twin is settled: copy each collapsed task's answer
        // across. Iterating `pending_aliases` in issuance order keeps this deterministic.
        for (const auto& [alias, twin] : pending_aliases) {
            outputs[alias.value] = outputs[twin.value];
            origins[alias.value] = outputs[alias.value].has_value()
                                       ? std::nullopt
                                       : std::optional<std::size_t>{alias.value};
            seconds[alias.value] = 0.0;
        }
    }

    reaper.succeeded = true;             // ONLY the full-result path sets this
    return TaskRunResult{
        .outputs = std::move(outputs),
        .measured_seconds = std::move(seconds),
        .executed = executed
    };
}

}  // namespace nimblecas
