// NimbleCAS PostgreSQL-backed distributed memoization.
// @author Olumuyiwa Oluwasanmi
//
// A `DistributedMemo` whose table lives in PostgreSQL, so a memo survives not merely the
// coordinator process (which `FileMemo` already achieves) but the MACHINE -- and, more to the
// point, is shared by executors that have no filesystem in common. That is the gap this closes:
// `InProcessMemo` is one process, `FileMemo` is one filesystem, and a cluster is neither.
//
// OPT-IN, AND WHY. This is the only module in the repository with a third-party runtime
// dependency: it links libpq. The whole build is otherwise dependency-free by policy, so this
// module is gated behind the `NIMBLECAS_POSTGRES` CMake option and is not built unless asked for.
// Nothing else in the repository imports it.
//
// EXACTNESS IS PRESERVED ACROSS THE WIRE. `DistributedMemo` requires that a fingerprint collision
// whose stored full key differs from the caller's return a MISS, never the colliding value, and
// that requirement does not weaken because the table is remote. The query selects by fingerprint
// and the full key is then compared byte for byte here, so a colliding row is rejected on the
// spot. The 128-bit fingerprint is an INDEX, never an identity: it narrows the scan and decides
// nothing. Putting the full key in the WHERE clause instead would be equally correct and would
// cost one thing worth keeping -- a collision and an absent row would become the same empty
// result set, and `key_mismatches` could then only ever be a guess.
//
// WHAT A FAILURE MEANS HERE. Every libpq failure -- a refused connection, a dropped socket, a
// query error, a server that went away mid-transaction -- is `MathError::distributed_error`, the
// transport-class code, and never a miss. That distinction is the honesty boundary of this module
// and it runs the opposite way to the obvious convenience: reporting a broken connection as a
// cache miss would let a run continue and silently recompute everything, which looks like it
// worked. A miss means "the table does not have this"; an error means "the table could not be
// asked". A caller that wants the former behaviour can choose it; this module will not choose it
// for them.
//
// THREAD SAFETY. `DistributedMemo` implementations must be thread-safe, and a `PGconn` is not: it
// carries one connection's protocol state and two threads sharing it corrupt each other's results.
// A small POOL of connections is therefore opened at construction and handed out under a mutex.
// The pool size is the concurrency ceiling, not a hint: a thread that finds the pool empty waits.
//
// WHAT IS NOT CLAIMED. Nothing here is transactional across more than one statement, because
// nothing needs to be: a publish is a single idempotent INSERT and a lookup is a single SELECT.
// There is no eviction, no TTL and no size cap on the table -- the table grows until somebody
// deletes from it, and that is a deployment decision rather than one this module should make
// silently. No performance claim is made anywhere: the round trip to a database is obviously
// dearer than a hash lookup, and whether memoizing is worth it depends entirely on what the task
// costs to recompute.

module;

#include <libpq-fe.h>

export module nimblecas.memo_postgres;

import std;
import nimblecas.core;
import nimblecas.memo_dist;
import nimblecas.taskdag;

export namespace nimblecas {

// A `DistributedMemo` backed by a PostgreSQL table.
//
// The table is created if it does not exist, together with the unique index that makes `publish`
// idempotent. Two coordinators pointed at the same table share their results; that is the point.
class PostgresMemo final : public DistributedMemo {
  public:
    // Connects and ensures the schema. `create()` rather than a throwing constructor, for the
    // same reason as `FileMemo`: an unreachable host, bad credentials, or a database the user
    // cannot create a table in are ordinary runtime conditions, and all of them return an honest
    // error instead of a half-built object.
    //
    // `conninfo` is a libpq connection string ("host=... dbname=... user=..." or a postgresql://
    // URI). Callers are strongly advised to include `connect_timeout`, because libpq's default is
    // to wait indefinitely and a memo that hangs is worse than one that fails.
    //
    // `table` must be a plain identifier -- letters, digits and underscores, not starting with a
    // digit. It is validated rather than escaped, because it is interpolated into the DDL and the
    // only safe interpolation is one that cannot carry anything but an identifier. Every VALUE is
    // sent as a bound parameter and never interpolated at all.
    //
    // `pool_size` connections are opened; it is the ceiling on concurrent database work.
    [[nodiscard]] static auto create(std::string conninfo, std::string table = "nimblecas_memo",
                                     std::size_t pool_size = 4,
                                     std::size_t max_value_bytes = std::size_t{16} * 1024 * 1024)
        -> Result<std::unique_ptr<PostgresMemo>>;

    ~PostgresMemo() override;
    PostgresMemo(const PostgresMemo&) = delete;
    auto operator=(const PostgresMemo&) -> PostgresMemo& = delete;
    PostgresMemo(PostgresMemo&&) = delete;
    auto operator=(PostgresMemo&&) -> PostgresMemo& = delete;

    [[nodiscard]] auto name() const -> std::string_view override;

    // A hit requires the full key to match exactly; a fingerprint collision is a miss. A
    // transport failure is `distributed_error` and is NOT reported as a miss.
    [[nodiscard]] auto lookup(const ContentKey& key, std::span<const std::byte> full_key)
        -> Result<std::optional<Payload>> override;

    // Idempotent: publishing the same key and full key twice keeps the first value and is not an
    // error. `domain_error` if the value exceeds `max_value_bytes`.
    [[nodiscard]] auto publish(const ContentKey& key, std::span<const std::byte> full_key,
                               std::span<const std::byte> value) -> Result<void> override;

    [[nodiscard]] auto stats() const -> MemoStats override;

    // The number of rows currently in the table. Exposed for operational use -- there is no
    // eviction, so this is how a deployment finds out how large the memo has become.
    [[nodiscard]] auto row_count() -> Result<std::uint64_t>;

    // Deletes every row. Provided because a test needs a clean table and an operator needs a way
    // to reclaim the space; it is deliberately explicit rather than any kind of automatic policy.
    [[nodiscard]] auto clear() -> Result<void>;

    // Whether `table` is a plain SQL identifier. Exposed because the check is the whole reason
    // interpolating the table name is safe, and a caller may want to apply it before calling.
    [[nodiscard]] static auto is_plain_identifier(std::string_view table) noexcept -> bool;

  private:
    PostgresMemo(std::string conninfo, std::string table, std::size_t max_value_bytes);

    // One pooled connection, returned to the pool by the guard's destructor.
    class Lease;

    [[nodiscard]] auto acquire() -> Lease;

    std::string conninfo_;
    std::string table_;
    std::size_t max_value_bytes_{0};

    mutable std::mutex mutex_;
    std::condition_variable available_;
    std::vector<PGconn*> free_;
    std::vector<PGconn*> owned_;

    mutable std::atomic<std::uint64_t> hits_{0};
    mutable std::atomic<std::uint64_t> misses_{0};
    mutable std::atomic<std::uint64_t> publishes_{0};
    mutable std::atomic<std::uint64_t> key_mismatches_{0};
    mutable std::atomic<std::uint64_t> rejected_{0};
};

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

class PostgresMemo::Lease {
  public:
    Lease(PostgresMemo* owner, PGconn* conn) : owner_(owner), conn_(conn) {}
    // Returning a connection to the pool must not throw: this is a destructor, and an escaping
    // exception here terminates the process. `free_` is reserved to the pool's full capacity at
    // construction and can never hold more connections than that, so this push_back writes into
    // memory that already exists and cannot allocate.
    // `free_` is reserved to the pool size at construction and can never hold more connections
    // than that, so the push_back below writes into memory that already exists: it cannot
    // allocate and cannot reach the length check the analyser is following. The diagnostic is
    // attached to the destructor, so the suppression belongs on the line before it.
    // NOLINTNEXTLINE(bugprone-exception-escape)
    ~Lease() {
        if (owner_ != nullptr && conn_ != nullptr) {
            {
                const std::lock_guard<std::mutex> lock(owner_->mutex_);
                owner_->free_.push_back(conn_);
            }
            owner_->available_.notify_one();
        }
    }
    Lease(const Lease&) = delete;
    auto operator=(const Lease&) -> Lease& = delete;
    Lease(Lease&& other) noexcept : owner_(other.owner_), conn_(other.conn_) {
        other.owner_ = nullptr;
        other.conn_ = nullptr;
    }
    auto operator=(Lease&&) -> Lease& = delete;

    [[nodiscard]] auto get() const noexcept -> PGconn* { return conn_; }

  private:
    PostgresMemo* owner_{nullptr};
    PGconn* conn_{nullptr};
};

namespace {

// A libpq result, always cleared. PGresult is a C allocation and every early return below would
// otherwise leak it.
class ResultGuard {
  public:
    explicit ResultGuard(PGresult* r) noexcept : r_(r) {}
    ~ResultGuard() {
        if (r_ != nullptr) {
            PQclear(r_);
        }
    }
    ResultGuard(const ResultGuard&) = delete;
    auto operator=(const ResultGuard&) -> ResultGuard& = delete;
    ResultGuard(ResultGuard&&) = delete;
    auto operator=(ResultGuard&&) -> ResultGuard& = delete;

    [[nodiscard]] auto get() const noexcept -> PGresult* { return r_; }

  private:
    PGresult* r_{nullptr};
};

// The two halves of a ContentKey as BIGINTs. Postgres has no unsigned 64-bit type, so each half
// travels as its two's-complement bit pattern reinterpreted as a signed value -- a total,
// reversible mapping, and the only one that does not lose a bit. The column is an index, never
// something to do arithmetic on, so the sign it appears to have does not matter.
[[nodiscard]] auto as_signed(std::uint64_t v) noexcept -> std::int64_t {
    return static_cast<std::int64_t>(v);
}

}  // namespace

auto PostgresMemo::is_plain_identifier(std::string_view table) noexcept -> bool {
    if (table.empty() || table.size() > 63) {  // Postgres truncates identifiers past 63 bytes
        return false;
    }
    const char first = table.front();
    const bool first_ok = (first >= 'a' && first <= 'z') || (first >= 'A' && first <= 'Z') ||
                          first == '_';
    if (!first_ok) {
        return false;
    }
    return std::ranges::all_of(table, [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
               c == '_';
    });
}

PostgresMemo::PostgresMemo(std::string conninfo, std::string table, std::size_t max_value_bytes)
    : conninfo_(std::move(conninfo)), table_(std::move(table)), max_value_bytes_(max_value_bytes) {}

PostgresMemo::~PostgresMemo() {
    for (PGconn* c : owned_) {
        if (c != nullptr) {
            PQfinish(c);
        }
    }
}

auto PostgresMemo::acquire() -> Lease {
    std::unique_lock<std::mutex> lock(mutex_);
    available_.wait(lock, [this] { return !free_.empty(); });
    PGconn* c = free_.back();
    free_.pop_back();
    return Lease{this, c};
}

auto PostgresMemo::create(std::string conninfo, std::string table, std::size_t pool_size,
                          std::size_t max_value_bytes)
    -> Result<std::unique_ptr<PostgresMemo>> {
    using Ret = std::unique_ptr<PostgresMemo>;
    if (conninfo.empty() || pool_size == 0 || max_value_bytes == 0) {
        return make_error<Ret>(MathError::domain_error);
    }
    if (!is_plain_identifier(table)) {
        // Refused rather than quoted. A table name that needs escaping is a name this module
        // will not interpolate, and interpolation is unavoidable for DDL.
        return make_error<Ret>(MathError::domain_error);
    }

    std::unique_ptr<PostgresMemo> memo(
        new PostgresMemo(std::move(conninfo), std::move(table), max_value_bytes));

    // Reserved BEFORE anything is opened, and load-bearing rather than an optimisation: the
    // Lease destructor returns a connection by pushing here, and a destructor that allocates is
    // a destructor that can terminate the process.
    memo->free_.reserve(pool_size);
    memo->owned_.reserve(pool_size);

    for ([[maybe_unused]] const auto i : std::views::iota(std::size_t{0}, pool_size)) {
        PGconn* c = PQconnectdb(memo->conninfo_.c_str());
        if (c == nullptr) {
            return make_error<Ret>(MathError::distributed_error);
        }
        if (PQstatus(c) != CONNECTION_OK) {
            PQfinish(c);
            // Connections already opened are closed by the destructor when `memo` goes out of
            // scope, so a partial pool never leaks.
            return make_error<Ret>(MathError::distributed_error);
        }
        memo->owned_.push_back(c);
        memo->free_.push_back(c);
    }

    // The schema. `IF NOT EXISTS` throughout, so several coordinators starting at once all
    // succeed rather than racing. The unique index is what makes `publish` idempotent: it is on
    // the fingerprint plus a DIGEST of the full key rather than the full key itself, because a
    // btree entry has a size limit and a full key is arbitrarily long.
    const std::string ddl = std::format(
        "CREATE TABLE IF NOT EXISTS {} ("
        "  key_hi BIGINT NOT NULL,"
        "  key_lo BIGINT NOT NULL,"
        "  full_key BYTEA NOT NULL,"
        "  value BYTEA NOT NULL,"
        "  created TIMESTAMPTZ NOT NULL DEFAULT now()"
        "); "
        "CREATE INDEX IF NOT EXISTS {}_key_idx ON {} (key_hi, key_lo); "
        "CREATE UNIQUE INDEX IF NOT EXISTS {}_uniq_idx ON {} (key_hi, key_lo, md5(full_key));",
        memo->table_, memo->table_, memo->table_, memo->table_, memo->table_);

    const ResultGuard res(PQexec(memo->owned_.front(), ddl.c_str()));
    if (res.get() == nullptr || PQresultStatus(res.get()) != PGRES_COMMAND_OK) {
        return make_error<Ret>(MathError::distributed_error);
    }
    return memo;
}

auto PostgresMemo::name() const -> std::string_view { return "postgres_memo"; }

auto PostgresMemo::lookup(const ContentKey& key, std::span<const std::byte> full_key)
    -> Result<std::optional<Payload>> {
    using Ret = std::optional<Payload>;
    const auto lease = acquire();

    const std::string hi = std::to_string(as_signed(key.hi));
    const std::string lo = std::to_string(as_signed(key.lo));
    // Selected by FINGERPRINT, then verified here byte for byte.
    //
    // Matching the full key in the WHERE clause would also be correct -- a colliding row would
    // not match and the result would be a miss -- but it would make a collision and an absent
    // row the same empty result set, and `key_mismatches` could then only ever be a guess. The
    // fingerprint is an index; the full key decides; and doing the deciding here is what lets
    // the counter mean what the other backends make it mean. A fingerprint normally selects
    // exactly one row, so this fetches no more than the other formulation would.
    const std::string sql =
        std::format("SELECT full_key, value FROM {} WHERE key_hi = $1 AND key_lo = $2", table_);

    const std::array<const char*, 2> values{hi.c_str(), lo.c_str()};
    const std::array<int, 2> lengths{0, 0};
    const std::array<int, 2> formats{0, 0};

    const ResultGuard res(PQexecParams(lease.get(), sql.c_str(), 2, nullptr, values.data(),
                                       lengths.data(), formats.data(), 1 /* binary results */));
    if (res.get() == nullptr || PQresultStatus(res.get()) != PGRES_TUPLES_OK) {
        // A failed query is NOT a miss. Reporting it as one would let a run continue and quietly
        // recompute everything while looking healthy.
        return make_error<Ret>(MathError::distributed_error);
    }
    const int rows = PQntuples(res.get());
    bool saw_fingerprint = false;
    for (const int r : std::views::iota(0, rows)) {
        const int key_len = PQgetlength(res.get(), r, 0);
        const char* key_data = PQgetvalue(res.get(), r, 0);
        if (key_data == nullptr || key_len < 0) {
            return make_error<Ret>(MathError::distributed_error);
        }
        saw_fingerprint = true;
        if (static_cast<std::size_t>(key_len) != full_key.size() ||
            (full_key.size() != 0 &&
             std::memcmp(key_data, full_key.data(), full_key.size()) != 0)) {
            continue;  // the fingerprint collided; this row is somebody else's
        }
        const int len = PQgetlength(res.get(), r, 1);
        const char* data = PQgetvalue(res.get(), r, 1);
        if (data == nullptr || len < 0) {
            return make_error<Ret>(MathError::distributed_error);
        }
        Payload out;
        out.resize(static_cast<std::size_t>(len));
        if (len > 0) {
            std::memcpy(out.data(), data, static_cast<std::size_t>(len));
        }
        hits_.fetch_add(1, std::memory_order_relaxed);
        return Ret{std::move(out)};
    }
    // Nothing matched. If rows existed under this fingerprint, every one of them belonged to a
    // different key -- which is the collision case, and it is still a MISS.
    if (saw_fingerprint) {
        key_mismatches_.fetch_add(1, std::memory_order_relaxed);
    }
    misses_.fetch_add(1, std::memory_order_relaxed);
    return Ret{std::nullopt};
}

auto PostgresMemo::publish(const ContentKey& key, std::span<const std::byte> full_key,
                           std::span<const std::byte> value) -> Result<void> {
    if (value.size() > max_value_bytes_) {
        rejected_.fetch_add(1, std::memory_order_relaxed);
        return make_error<void>(MathError::domain_error);
    }
    const auto lease = acquire();

    const std::string hi = std::to_string(as_signed(key.hi));
    const std::string lo = std::to_string(as_signed(key.lo));
    // ON CONFLICT DO NOTHING against the unique index: publishing the same result twice, which
    // two workers racing on the same task will do, keeps the first row and is not an error.
    const std::string sql = std::format(
        "INSERT INTO {} (key_hi, key_lo, full_key, value) VALUES ($1, $2, $3, $4) "
        "ON CONFLICT DO NOTHING",
        table_);

    const std::array<const char*, 4> values{
        hi.c_str(), lo.c_str(),
        full_key.empty() ? "" : reinterpret_cast<const char*>(full_key.data()),
        value.empty() ? "" : reinterpret_cast<const char*>(value.data())};
    const std::array<int, 4> lengths{0, 0, static_cast<int>(full_key.size()),
                                     static_cast<int>(value.size())};
    const std::array<int, 4> formats{0, 0, 1, 1};

    const ResultGuard res(PQexecParams(lease.get(), sql.c_str(), 4, nullptr, values.data(),
                                       lengths.data(), formats.data(), 0));
    if (res.get() == nullptr || PQresultStatus(res.get()) != PGRES_COMMAND_OK) {
        return make_error<void>(MathError::distributed_error);
    }
    // A row that was not inserted means an entry for this fingerprint AND this full key already
    // existed: two workers raced on the same task and both published. That is the idempotent
    // case and is counted as an ordinary publish -- it is emphatically not a key mismatch, since
    // a genuine collision carries a different full key, hence a different digest, and inserts a
    // row of its own.
    publishes_.fetch_add(1, std::memory_order_relaxed);
    return {};
}

auto PostgresMemo::stats() const -> MemoStats {
    return MemoStats{.hits = hits_.load(std::memory_order_relaxed),
                     .misses = misses_.load(std::memory_order_relaxed),
                     .publishes = publishes_.load(std::memory_order_relaxed),
                     .key_mismatches = key_mismatches_.load(std::memory_order_relaxed),
                     .rejected = rejected_.load(std::memory_order_relaxed)};
}

auto PostgresMemo::row_count() -> Result<std::uint64_t> {
    const auto lease = acquire();
    const std::string sql = std::format("SELECT count(*) FROM {}", table_);
    const ResultGuard res(PQexec(lease.get(), sql.c_str()));
    if (res.get() == nullptr || PQresultStatus(res.get()) != PGRES_TUPLES_OK ||
        PQntuples(res.get()) != 1) {
        return make_error<std::uint64_t>(MathError::distributed_error);
    }
    const char* text = PQgetvalue(res.get(), 0, 0);
    if (text == nullptr) {
        return make_error<std::uint64_t>(MathError::distributed_error);
    }
    std::uint64_t out = 0;
    const std::string_view sv(text);
    const auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), out);
    // The standard defines from_chars success as a VALUE-INITIALISED error code, and std::errc
    // has no zero-valued enumerator to name, so testing for it must construct one. That is the
    // correct idiom rather than an accidental default-initialised enum.
    // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization)
    const bool parsed_ok = ec == std::errc{};
    if (!parsed_ok || ptr != sv.data() + sv.size()) {
        return make_error<std::uint64_t>(MathError::distributed_error);
    }
    return out;
}

auto PostgresMemo::clear() -> Result<void> {
    const auto lease = acquire();
    const std::string sql = std::format("DELETE FROM {}", table_);
    const ResultGuard res(PQexec(lease.get(), sql.c_str()));
    if (res.get() == nullptr || PQresultStatus(res.get()) != PGRES_COMMAND_OK) {
        return make_error<void>(MathError::distributed_error);
    }
    return {};
}

}  // namespace nimblecas
