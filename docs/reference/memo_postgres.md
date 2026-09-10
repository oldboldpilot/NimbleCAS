# `nimblecas.memo_postgres` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/memo_postgres/memo_postgres.cppm`

A PostgreSQL-backed implementation of [`nimblecas.memo_dist::DistributedMemo`](memo_dist.md),
providing persistent, content-addressed memoisation for distributed task DAG execution across a
network. While [`InProcessMemo`](memo_dist.md) is confined to a single process address space and
[`FileMemo`](memo_dist.md) is confined to a single local filesystem, an executor cluster has neither.
`PostgresMemo` closes that architectural gap: it stores task inputs, fingerprints, and outputs in a
central PostgreSQL table, enabling multiple independent coordinators and worker nodes on distinct
machines to share cached results without a shared filesystem.

The module is governed by a strict **honesty boundary** (Code Policy Rule 32):

- **Opt-in module with third-party dependency:** this is the **only module in the entire repository**
  with a third-party runtime dependency, linking against the PostgreSQL C client library (`libpq`).
  By policy, the NimbleCAS codebase is strictly self-contained and dependency-free. Consequently, this
  module is opt-in, gated behind the `NIMBLECAS_POSTGRES` CMake option, and is neither built nor linked
  unless explicitly requested. Nothing else in the repository imports it.
- **The honesty boundary — transport failures are errors, never misses:** every `libpq` failure —
  an unreachable server, a refused connection, a dropped socket, a query execution fault, or a
  connection termination mid-statement — returns `MathError::distributed_error` (the transport-class
  error code) and **never** a cache miss (`std::nullopt`). This distinction is critical and runs
  counter to the superficial convenience of masking network faults:
  - A **miss** means that the table was successfully queried and does not contain an entry for the task.
  - An **error** means that the table could not be consulted.
  Reporting an unavailable or broken database connection as a cache miss would allow a distributed run
  to continue execution, silently recomputing every task from scratch while masquerading as a healthy
  cache. A caller desiring fallback behaviour must explicitly handle the error; `PostgresMemo` will not
  make that choice silently.
- **Exactness across the wire and fingerprint collision semantics:** `DistributedMemo` mandates that a
  fingerprint collision whose stored full key differs from the caller's query key must return a
  **miss**, never the colliding value. This requirement does not weaken because the table is remote.
  In `PostgresMemo`, the SQL query selects rows solely by the 128-bit fingerprint (`key_hi` and
  `key_lo`), and the client code performs a byte-for-byte verification of `full_key` against the
  returned data using `std::memcmp`. A colliding row is rejected on the spot, returning `std::nullopt`
  and incrementing `MemoStats::key_mismatches`. Matching the full key in the SQL `WHERE` clause would
  be equally sound, but it would cause a collision and an absent entry to yield the identical empty
  result set; in that case, `key_mismatches` could only ever be an unverified estimate.
- **SQL injection prevention via identifier validation:** because PostgreSQL DDL does not permit
  parameterised table identifiers, the table name must be interpolated into schema statements
  (`CREATE TABLE`, `CREATE INDEX`). To ensure absolute safety, the table name is strictly validated
  via `is_plain_identifier` (1 to 63 ASCII alphanumeric characters or underscores, not beginning with
  a digit). Any name failing this check is rejected with `MathError::domain_error` rather than quoted
  or escaped. In contrast, all data values in DML operations (`INSERT`, `SELECT`) are bound binary
  parameters transmitted via `PQexecParams` and are never interpolated.
- **Thread safety via connection pooling:** `DistributedMemo` implementations must be thread-safe
  under concurrent evaluation. Because a `libpq` connection handle (`PGconn`) encapsulates mutable
  protocol state and cannot be accessed concurrently without corrupting streams, `PostgresMemo`
  maintains a connection pool protected by a mutex and condition variable. The configured `pool_size`
  acts as a strict concurrency ceiling, not an advisory hint: threads seeking a connection when the
  pool is exhausted block until a connection is released by an RAII `Lease`.
- **Idempotent publish and index sizing:** multiple workers racing on identical tasks will attempt to
  publish identical results concurrently. `publish` executes `INSERT ... ON CONFLICT DO NOTHING`
  against a unique index defined on `(key_hi, key_lo, md5(full_key))`. Hashing the full key via
  `md5()` inside the index avoids PostgreSQL B-tree index row-size limits (which reject indexed
  columns exceeding approximately 2700 bytes), whilst supporting arbitrarily long canonical task keys.
  The first published result is preserved; subsequent writes for the same key succeed idempotently and
  increment `MemoStats::publishes`.
- **What is not claimed:**
  - Operations do not span multi-statement transactions; lookups are single `SELECT` statements and
    publishes are single `INSERT` statements.
  - No cache eviction, time-to-live (TTL), or automatic table pruning is implemented. The database table
    grows until an operator or test suite explicitly issues `clear()` or truncates the table.
  - Absolutely no performance, latency, or throughput claims are made. Network database round-trips are
    inherently costlier than in-memory hash lookups. Whether distributed memoisation provides an overall
    benefit depends entirely on whether the computation time of a task exceeds the database round-trip
    cost.
- **Operational timeouts:** callers should always provide a `connect_timeout` parameter in the
  connection string (`conninfo`). By default, `libpq` blocks indefinitely when attempting to reach an
  unresponsive host, converting an offline database into an indefinite process hang.
- **Test suite gating:** test suites split into offline and live suites. Offline tests verify identifier
  syntax validation and error reporting on unreachable ports without requiring a database. Live tests
  require an active PostgreSQL instance and are gated on the `NIMBLECAS_POSTGRES_DSN` environment
  variable; if unset, live tests loudly report `SKIPPED` rather than silently succeeding.

```cpp
import nimblecas.memo_postgres;
```

Depends on [`core`](core.md) (`Result`, `MathError`, `Payload`), [`memo_dist`](memo_dist.md)
(`DistributedMemo`, `ContentKey`, `MemoStats`, `content_key`), [`taskdag`](taskdag.md), and
`libpq` (`<libpq-fe.h>`).

## Class definition and API

```cpp
export namespace nimblecas {

class PostgresMemo final : public DistributedMemo {
  public:
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

    [[nodiscard]] auto lookup(const ContentKey& key, std::span<const std::byte> full_key)
        -> Result<std::optional<Payload>> override;

    [[nodiscard]] auto publish(const ContentKey& key, std::span<const std::byte> full_key,
                               std::span<const std::byte> value) -> Result<void> override;

    [[nodiscard]] auto stats() const -> MemoStats override;

    [[nodiscard]] auto row_count() -> Result<std::uint64_t>;

    [[nodiscard]] auto clear() -> Result<void>;

    [[nodiscard]] static auto is_plain_identifier(std::string_view table) noexcept -> bool;
};

}  // namespace nimblecas
```

| Method / Member | Role | Contract and Invariants |
| :--- | :--- | :--- |
| `create` | Factory function connecting to PostgreSQL, initialising the connection pool, and creating the schema. | Returns `MathError::domain_error` on empty `conninfo`, zero `pool_size`, zero `max_value_bytes`, or invalid `table`. Returns `MathError::distributed_error` if connection or DDL execution fails. |
| `name` | Identifies the memo implementation. | Always returns `"postgres_memo"`. |
| `lookup` | Retrieves a cached value for a task key. | Returns `std::optional<Payload>` containing the value on hit, `std::nullopt` on miss or fingerprint collision. Returns `MathError::distributed_error` on any database failure. |
| `publish` | Persists a task output payload under its content key and full key bytes. | Idempotent via `ON CONFLICT DO NOTHING`. Returns `MathError::domain_error` and increments `stats.rejected` if `value.size() > max_value_bytes`. |
| `stats` | Returns a snapshot of operational telemetry counters. | Returns a `MemoStats` struct containing `hits`, `misses`, `publishes`, `key_mismatches`, and `rejected`. |
| `row_count` | Queries `SELECT count(*) FROM {table}` to determine current table size. | Returns total row count as `std::uint64_t`, or `MathError::distributed_error` on database query failure. |
| `clear` | Reclaims space by issuing `DELETE FROM {table}`. | Clears all rows in the memo table; returns `MathError::distributed_error` on failure. |
| `is_plain_identifier` | Validates whether a candidate table name is safe for DDL interpolation. | Returns `true` if `table` has length 1–63, starts with `[a-zA-Z_]`, and contains only `[a-zA-Z0-9_]`. |

## Database schema and index structure

Upon initialisation, `PostgresMemo::create` establishes the required schema using `IF NOT EXISTS`
qualifiers, allowing concurrent coordinators to initialise the database without race conditions:

```sql
CREATE TABLE IF NOT EXISTS {table} (
  key_hi BIGINT NOT NULL,
  key_lo BIGINT NOT NULL,
  full_key BYTEA NOT NULL,
  value BYTEA NOT NULL,
  created TIMESTAMPTZ NOT NULL DEFAULT now()
); 
CREATE INDEX IF NOT EXISTS {table}_key_idx ON {table} (key_hi, key_lo); 
CREATE UNIQUE INDEX IF NOT EXISTS {table}_uniq_idx ON {table} (key_hi, key_lo, md5(full_key));
```

Because PostgreSQL lacks an unsigned 64-bit integer type, the 64-bit components of `ContentKey`
(`hi` and `lo`) are mapped to `BIGINT` using their signed two's-complement bit patterns. This
transformation is bidirectional, lossless, and preserves indexing capabilities.

## Error model

All operations report status through `Result<T>`. Network and database faults are classified
strictly under the transport error category.

| Operation | Trigger Condition | Outcome |
| :--- | :--- | :--- |
| `create` | `conninfo.empty()`, `pool_size == 0`, or `max_value_bytes == 0` | `MathError::domain_error` |
| `create` | `table` fails `is_plain_identifier` | `MathError::domain_error` |
| `create` | `PQconnectdb` fails or returns `CONNECTION_BAD` | `MathError::distributed_error` |
| `create` | Schema DDL execution (`PQexec`) fails | `MathError::distributed_error` |
| `lookup` | Database unreachable, connection dropped, or query failure | `MathError::distributed_error` |
| `lookup` | Key not present in table | Success with `std::nullopt` (`misses++`) |
| `lookup` | Fingerprint found but `full_key` byte comparison fails | Success with `std::nullopt` (`misses++`, `key_mismatches++`) |
| `lookup` | Fingerprint and `full_key` match | Success with `Payload` (`hits++`) |
| `publish` | `value.size() > max_value_bytes` | `MathError::domain_error` (`rejected++`) |
| `publish` | Database unreachable or `INSERT` query fails | `MathError::distributed_error` |
| `publish` | Successful insertion or idempotent conflict ignore | Success with `void` (`publishes++`) |
| `row_count` | Query failure or non-integer count response | `MathError::distributed_error` |
| `clear` | `DELETE` query execution fails | `MathError::distributed_error` |

## Worked examples

### Initialising the memo and executing basic operations

This example connects to a PostgreSQL instance, publishes a computed result payload, and reads it
back. Callers should always provide `connect_timeout` in `conninfo`.

```cpp
import std;
import nimblecas.core;
import nimblecas.memo_dist;
import nimblecas.memo_postgres;

using namespace nimblecas;

// Convert string views to byte buffers
auto to_bytes = [](std::string_view s) -> std::vector<std::byte> {
    std::vector<std::byte> out;
    out.reserve(s.size());
    for (const char c : s) {
        out.push_back(static_cast<std::byte>(static_cast<unsigned char>(c)));
    }
    return out;
};

// Connection string with explicit connection timeout (in seconds)
const std::string conninfo =
    "host=localhost port=5432 user=postgres dbname=nimblecas connect_timeout=5";

// Initialise memo with a pool of 4 connections and a custom table name
auto memo_res = PostgresMemo::create(conninfo, "task_cache", /*pool_size=*/4);
if (!memo_res) {
    // Database connection failed (MathError::distributed_error)
    return;
}
std::unique_ptr<PostgresMemo> memo = std::move(*memo_res);

const auto full_key = to_bytes("task_canonical_representation_v1");
const auto key = content_key(full_key);
const auto value = to_bytes("task_computed_output_bytes");

// 1. Initial lookup results in a miss (std::nullopt)
auto before = memo->lookup(key, full_key);
if (before && !before->has_value()) {
    // Cache miss: task must be computed
}

// 2. Publish computed output
auto pub_res = memo->publish(key, full_key, value);
if (pub_res) {
    // Successfully published
}

// 3. Subsequent lookup retrieves the cached payload
auto after = memo->lookup(key, full_key);
if (after && after->has_value()) {
    const Payload& payload = **after;
    // payload matches value byte for byte
}

// Inspect telemetry
MemoStats stats = memo->stats();
// stats.misses == 1, stats.publishes == 1, stats.hits == 1
```

### Transport failure verification (honesty boundary)

An unreachable host or closed port produces `MathError::distributed_error` rather than a cache miss:

```cpp
import nimblecas.core;
import nimblecas.memo_dist;
import nimblecas.memo_postgres;

using namespace nimblecas;

// Loopback port 1 has no service listening; connect_timeout bounds the connection wait
const std::string dead_conninfo =
    "host=127.0.0.1 port=1 dbname=nimblecas connect_timeout=2";

auto memo = PostgresMemo::create(dead_conninfo, "test_table", /*pool_size=*/1);

if (!memo) {
    // Fails with MathError::distributed_error
    // The failure is never reported as a cache miss or silent bypass
    bool is_transport_error = (memo.error() == MathError::distributed_error);
    // is_transport_error is true
}
```

### Multithreaded concurrent evaluation

Because `PostgresMemo` manages an internal connection pool, multiple worker threads may share a
single instance concurrently:

```cpp
import std;
import nimblecas.core;
import nimblecas.memo_dist;
import nimblecas.memo_postgres;

using namespace nimblecas;

auto run_concurrent_memo(PostgresMemo& memo) -> void {
    constexpr int worker_count = 8;
    constexpr int tasks_per_worker = 10;
    std::atomic<int> completed{0};

    std::vector<std::jthread> workers;
    workers.reserve(worker_count);

    for (int w = 0; w < worker_count; ++w) {
        workers.emplace_back([&memo, &completed, w]() {
            for (int i = 0; i < tasks_per_worker; ++i) {
                std::string key_str = std::format("worker_{}_task_{}", w, i);
                std::string val_str = std::format("result_{}_{}", w, i);

                std::vector<std::byte> full_key(key_str.size());
                std::memcpy(full_key.data(), key_str.data(), key_str.size());

                std::vector<std::byte> value(val_str.size());
                std::memcpy(value.data(), val_str.data(), val_str.size());

                const auto key = content_key(full_key);

                if (memo.publish(key, full_key, value).has_value()) {
                    auto got = memo.lookup(key, full_key);
                    if (got.has_value() && got->has_value() && **got == value) {
                        completed.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        });
    }

    // All worker threads join automatically via jthread destructor
}
```

## See also

- [`nimblecas.memo_dist`](memo_dist.md) — content-addressed memoisation interfaces, `ContentKey`,
  and `InProcessMemo`.
- [`nimblecas.taskdag`](taskdag.md) — task DAG definition, registries, and local executors.
- [`nimblecas.taskdag_sgee`](taskdag_sgee.md) — distributed task DAG execution over SGEE brokers.
- [`nimblecas.core`](core.md) — `Result<T>`, `MathError`, and fundamental type definitions.
- [Documentation hub](../Index.md)
