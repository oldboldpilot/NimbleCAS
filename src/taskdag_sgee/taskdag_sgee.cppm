// NimbleCAS SGEE distributed taskdag backend module.
// @author Olumuyiwa Oluwasanmi

module;
#include <cassert>

export module nimblecas.taskdag_sgee;

import std;
import nimblecas.core;
import nimblecas.taskdag;

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
class ResultChannel {
public:
    virtual ~ResultChannel() = default;

    [[nodiscard]] virtual auto put(std::uint64_t qid, Payload result_envelope)
        -> Result<void> = 0;

    [[nodiscard]] virtual auto get(std::uint64_t qid) const -> Result<Payload> = 0;
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
};

// Deterministic in-memory FakeBrokerPort for testing without SGEE.
class FakeBrokerPort final : public BrokerPort {
public:
    using ClockFn = std::function<std::uint64_t()>;

    explicit FakeBrokerPort(ClockFn clock = nullptr)
        : clock_(clock ? std::move(clock) : [this]() { return current_time_ms_; }) {}

    void set_time_ms(std::uint64_t now_ms) {
        std::lock_guard lock(mutex_);
        current_time_ms_ = now_ms;
    }

    void advance_time_ms(std::uint64_t delta_ms) {
        std::lock_guard lock(mutex_);
        current_time_ms_ += delta_ms;
    }

    [[nodiscard]] auto now_ms() const -> std::uint64_t {
        if (clock_) return clock_();
        std::lock_guard lock(mutex_);
        return current_time_ms_;
    }

    [[nodiscard]] auto enqueue(std::span<const std::byte> payload,
                               SgeePlacement placement, std::uint32_t max_attempts)
        -> Result<std::uint64_t> override {
        std::lock_guard lock(mutex_);
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
        auto it = tasks_.find(qid);
        if (it == tasks_.end()) {
            return make_error<void>(MathError::distributed_error);
        }
        if (it->second.state != QState::leased || it->second.fencing_token != token) {
            return make_error<void>(MathError::distributed_error);
        }
        const std::uint64_t extend = extend_by_ms ? extend_by_ms : 10'000;
        it->second.visibility_deadline_ms = now_ms() + extend;
        return {};
    }

    [[nodiscard]] auto sweep_expired(std::uint64_t now_ms)
        -> Result<std::size_t> override {
        std::lock_guard lock(mutex_);
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
    };

    ClockFn clock_{nullptr};
    std::uint64_t current_time_ms_{0};
    mutable std::mutex mutex_{};
    std::uint64_t next_qid_{0};
    std::uint64_t next_token_{0};
    std::unordered_map<std::uint64_t, TaskEntry> tasks_{};
    std::vector<std::uint64_t> fifo_order_{};
};

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
    std::function<SgeePlacement(const TaskGraph&, TaskId)> placement{};

    auto with_wal_dir(std::filesystem::path p) -> SgeeExecutorConfig& {
        wal_dir = std::move(p);
        return *this;
    }

    auto with_registry(const TaskRegistry& r) -> SgeeExecutorConfig& {
        registry = &r;
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

    auto with_placement(std::function<SgeePlacement(const TaskGraph&, TaskId)> f)
        -> SgeeExecutorConfig& {
        placement = std::move(f);
        return *this;
    }
};

class SgeeDistributedExecutor final : public Executor {
public:
    SgeeDistributedExecutor(SgeeExecutorConfig cfg, BrokerPort& port, ResultChannel& results)
        : cfg_(std::move(cfg)), port_(port), results_(results) {}

    [[nodiscard]] auto name() const noexcept -> std::string_view override {
        return "sgee_distributed";
    }

    [[nodiscard]] auto run(const TaskGraph& g) -> Result<TaskRunResult> override;

private:
    SgeeExecutorConfig cfg_;
    BrokerPort& port_;
    ResultChannel& results_;
};

// Factory function
[[nodiscard]] auto sgee_distributed_executor(SgeeExecutorConfig cfg)
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

namespace sgee_bridge {

auto encode_task(const TaskEnvelope& env, std::size_t max_bytes) -> Result<Payload> {
    if (env.op_id.empty() || env.op_id.size() > 256) {
        return make_error<Payload>(MathError::syntax_error);
    }
    std::size_t total_size = 4 + 2 + 2 + 8 + 4 + env.op_id.size() + 4 + env.args.size() * 8;
    for (const auto& arg : env.args) {
        total_size += arg.size();
    }
    if (total_size > max_bytes) {
        return make_error<Payload>(MathError::overflow);
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
    if (bytes.size() < 20) {
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
    std::uint64_t total_arg_bytes = 0;
    for (std::uint32_t i = 0; i < n_args; ++i) {
        arg_lens[i] = read_u64_le(bytes, offset + i * 8);
        total_arg_bytes += arg_lens[i];
    }
    offset += n_args * 8;

    if (bytes.size() != offset + total_arg_bytes) {
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
        out.insert(out.end(), env.bytes.begin(), env.bytes.begin() + len);
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
    const double seconds = read_f64_le(bytes, 8);
    const std::uint64_t len = read_u64_le(bytes, 16);

    if (bytes.size() != 24 + len) {
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
            std::atomic<bool> active{true};
            std::thread thread;

            HeartbeatGuard(BrokerPort& p, std::uint64_t q, std::uint64_t tok, std::uint64_t interval)
                : port(p), qid(q), token(tok), interval_ms(interval) {
                thread = std::thread([this]() {
                    while (active.load()) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
                        if (active.load()) {
                            (void)port.heartbeat(qid, token, 0);
                        }
                    }
                });
            }

            ~HeartbeatGuard() {
                active.store(false);
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

    // Worker threads management
    std::stop_source stop_source;
    std::vector<std::thread> workers;
    if (cfg_.num_workers > 0) {
        workers.reserve(cfg_.num_workers);
        for (std::size_t w = 0; w < cfg_.num_workers; ++w) {
            WorkerPumpConfig pump_cfg{
                .worker_id = w + 1,
                .lease_timeout_ms = cfg_.visibility_timeout_ms,
                .idle_backoff_ms = cfg_.poll_interval_ms,
                .heartbeat_every_ms = 0
            };
            workers.emplace_back([this, pump_cfg, token = stop_source.get_token()]() {
                run_worker_pump(port_, *cfg_.registry, results_, pump_cfg, token);
            });
        }
    }

    auto cleanup = [&]() {
        stop_source.request_stop();
        for (auto& w : workers) {
            if (w.joinable()) {
                w.join();
            }
        }
    };

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
        run_deadline_ms = std::max<std::uint64_t>(60'000, static_cast<std::uint64_t>(10.0 * total_cost * 1000.0));
    }

    const auto start_time = std::chrono::steady_clock::now();

    for (std::size_t lvl = 0; lvl < g.num_levels(); ++lvl) {
        const std::span<const TaskId> level_tasks = g.level(lvl);
        std::unordered_map<std::uint64_t, TaskId> pending_qids;

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

            // Gather inputs
            const auto task_deps = g.deps(id);
            std::vector<Payload> args;
            args.reserve(task_deps.size());
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
                cleanup();
                return make_error<TaskRunResult>(MathError::distributed_error);
            }

            const SgeePlacement placement = cfg_.placement ? cfg_.placement(g, id) : SgeePlacement::cpu;
            auto enqueue_res = port_.enqueue(*enc_res, placement, cfg_.max_attempts);
            if (!enqueue_res.has_value()) {
                cleanup();
                return make_error<TaskRunResult>(MathError::distributed_error);
            }

            pending_qids[*enqueue_res] = id;
        }

        // Await resolution of this level
        while (!pending_qids.empty()) {
            const auto elapsed_ms = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start_time).count());
            if (elapsed_ms > run_deadline_ms) {
                cleanup();
                return make_error<TaskRunResult>(MathError::distributed_error);
            }

            const auto now_ms = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count());
            (void)port_.sweep_expired(now_ms);

            std::vector<std::uint64_t> resolved_qids;

            for (const auto& [qid, id] : pending_qids) {
                auto st_res = port_.state(qid);
                if (!st_res.has_value()) {
                    cleanup();
                    return make_error<TaskRunResult>(MathError::distributed_error);
                }
                const BrokerPort::QState st = *st_res;
                if (st == BrokerPort::QState::completed) {
                    auto res_bytes_res = results_.get(qid);
                    if (!res_bytes_res.has_value()) {
                        cleanup();
                        return make_error<TaskRunResult>(MathError::distributed_error);
                    }
                    auto dec_res = sgee_bridge::decode_result(*res_bytes_res);
                    if (!dec_res.has_value()) {
                        cleanup();
                        return make_error<TaskRunResult>(MathError::distributed_error);
                    }
                    const auto& res_env = *dec_res;
                    if (res_env.status == sgee_bridge::ResultEnvelope::Status::ok) {
                        outputs[id.value] = res_env.bytes;
                        origins[id.value] = std::nullopt;
                        seconds[id.value] = res_env.seconds;
                        ++executed;
                        resolved_qids.push_back(qid);
                    } else if (res_env.status == sgee_bridge::ResultEnvelope::Status::math_error) {
                        outputs[id.value] = make_error<Payload>(res_env.math_err);
                        origins[id.value] = id.value;
                        seconds[id.value] = res_env.seconds;
                        ++executed;
                        resolved_qids.push_back(qid);
                    } else {
                        cleanup();
                        return make_error<TaskRunResult>(MathError::distributed_error);
                    }
                } else if (st == BrokerPort::QState::dead) {
                    cleanup();
                    return make_error<TaskRunResult>(MathError::distributed_error);
                }
            }

            for (const std::uint64_t qid : resolved_qids) {
                pending_qids.erase(qid);
            }

            if (!pending_qids.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(cfg_.poll_interval_ms));
            }
        }
    }

    cleanup();
    return TaskRunResult{
        .outputs = std::move(outputs),
        .measured_seconds = std::move(seconds),
        .executed = executed
    };
}

}  // namespace nimblecas
