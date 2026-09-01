// NimbleCAS content-addressed distributed memoization module (ROADMAP §6.2).
// @author Olumuyiwa Oluwasanmi
//
// WHAT THIS IS. A deterministic, content-addressed memoization layer for task-DAG execution
// and distributed evaluation. The task-DAG codec emits a canonical, fixed-width little-endian
// byte encoding for every task (registry fingerprint, OpId, and arguments). Two tasks represent
// the identical mathematical computation if and only if those canonical byte representations
// are equal AND the same implementations sit behind those op ids. This module provides a fast
// 128-bit ContentKey fingerprint for sharding and bucket indexing, coupled with full-key
// verification to ensure zero false hits.
//
// THAT PROVISO IS NOT A FORMALITY. TaskRegistry::fingerprint() is FNV-1a over the sorted op-ID
// STRINGS -- it hashes names, not bodies. Editing an op's implementation without changing its
// id moves nothing, so every existing entry stays live and keeps serving the old code's
// results. A per-call cache never sees this because it dies with the call; a memo that outlives
// the run, which is the whole point here, turns it into a wrong answer that persists. Any
// semantic change to a registered op REQUIRES a version bump (<domain>.<op>/vN) or a discarded
// memo. Nothing enforces that -- the /vN grammar is checked syntactically, never semantically.
//
// It deliberately does NOT key on Expr::structural_hash(): that is built from std::hash, whose
// values are implementation-defined and therefore differ across standard libraries, library
// versions, and size_t widths. A fingerprint that changes when the process changes cannot
// address content across a cluster. ContentKey is defined by arithmetic on byte values alone.
//
// DETERMINISM & EXACTNESS CONTRACT. A memo hit MUST be observationally indistinguishable from
// executing the underlying pure TaskFn. Because a 128-bit hash is not a proof of equality,
// lookup always performs byte-for-byte comparison of the full key bytes against stored entries.
// A fingerprint collision results in a key_mismatches count and returns nullopt (treated as a
// miss) — it NEVER returns an incorrect cached result. key_mismatches counts only a TRUE
// 128-bit collision: sharing a bucket is not one, because a bucket is addressed by
// (hi % shard_count, lo) and its occupants may differ in hi entirely.
//
// VALUE SHIPPABILITY & DETERMINISTIC ERROR CACHING. Only results that are pure, deterministic
// functions of their input arguments may be cached. Successful evaluations (ok) and domain/math
// failures (math_error) are deterministic and memoizable. Ephemeral coordination or transport
// failures (bridge_error) reflect transient cluster state rather than input semantics and MUST
// NOT be memoized, preventing temporary transport blips from poisoning future lookups.
//
// ── HONESTY BOUNDARY (Rule 32) ───────────────────────────────────────────────────────────
// The memoization layer is strictly EXACT: all lookups are verified against full canonical key
// bytes. Fingerprints are used solely for O(1) routing and collision-bucket indexing, never as
// probabilistic proofs of equality. The in-process memo operates with bounded capacity and NO
// eviction: once capacity or value-size bounds are reached, publishing requests are safely
// rejected rather than evicting entries, preserving bit-identical reproducibility across runs.

module;
#include <cassert>

export module nimblecas.memo_dist;

import std;
import nimblecas.core;
import nimblecas.taskdag;

export namespace nimblecas {

// 128-bit content fingerprint computed strictly from canonical byte representations.
// Stable across processes, architectures, and standard library implementations.
struct ContentKey {
    std::uint64_t hi{0};
    std::uint64_t lo{0};

    [[nodiscard]] friend auto operator==(const ContentKey&, const ContentKey&) noexcept -> bool = default;
    [[nodiscard]] friend auto operator<=>(const ContentKey&, const ContentKey&) noexcept = default;
};

// Computes a deterministic 128-bit ContentKey over an arbitrary byte sequence: two 64-bit
// FNV-1a lanes with distinct offset bases, the second folding each byte XORed with 0xA5 so the
// lanes consume different streams.
//
// This is a FINGERPRINT, not a collision-resistant hash, and nothing here claims otherwise: two
// FNV-1a lanes sharing one prime are not an independent 128-bit function, and FNV is not
// cryptographic. It is chosen for what this module actually needs — determinism across processes,
// architectures, and standard libraries, at one multiply per byte. Correctness never rests on it;
// every hit is confirmed by a full-key byte comparison (see the exactness rule in lookup).
[[nodiscard]] auto content_key(std::span<const std::byte> bytes) noexcept -> ContentKey;

// Formats a ContentKey as 32 lowercase hexadecimal characters (hi followed by lo).
[[nodiscard]] auto to_hex(const ContentKey& key) -> std::string;

// Telemetry counters for memo lookups, publications, collisions, and capacity rejections.
struct MemoStats {
    std::uint64_t hits{0};
    std::uint64_t misses{0};
    std::uint64_t publishes{0};
    std::uint64_t key_mismatches{0};  // fingerprint collided, full key differed -> treated as MISS
    std::uint64_t rejected{0};        // publish refused (over capacity / oversize value)

    [[nodiscard]] friend auto operator==(const MemoStats&, const MemoStats&) noexcept -> bool = default;
};

// Abstract interface for content-addressed task memoization.
class DistributedMemo {
public:
    DistributedMemo() = default;
    virtual ~DistributedMemo() = default;
    DistributedMemo(const DistributedMemo&) = delete;
    auto operator=(const DistributedMemo&) -> DistributedMemo& = delete;
    DistributedMemo(DistributedMemo&&) = delete;
    auto operator=(DistributedMemo&&) -> DistributedMemo& = delete;

    // Implementations MUST be thread-safe: the advertised deployment is one table shared by
    // several executors, and nothing above this interface serialises access to it.
    [[nodiscard]] virtual auto name() const -> std::string_view = 0;

    // Returns the stored value, or nullopt for a miss. A fingerprint collision whose
    // stored full key differs from `full_key` MUST return nullopt (a miss), never the
    // colliding value.
    [[nodiscard]] virtual auto lookup(const ContentKey& key, std::span<const std::byte> full_key)
        -> Result<std::optional<Payload>> = 0;

    // Publishes a computed result under `key` with its verifying `full_key`.
    [[nodiscard]] virtual auto publish(const ContentKey& key, std::span<const std::byte> full_key,
                                       std::span<const std::byte> value) -> Result<void> = 0;

    // A snapshot, not an atomic one: the counters are read independently, so a caller reading
    // stats() WHILE other threads are using the table may observe hits + misses momentarily
    // disagreeing with the number of lookups. The identity holds once the traffic stops.
    [[nodiscard]] virtual auto stats() const -> MemoStats = 0;
};

// Thread-safe, bounded-capacity, in-process implementation of DistributedMemo.
// Uses a two-level sharded bucket table with full-key verification and zero eviction.
class InProcessMemo final : public DistributedMemo {
public:
    explicit InProcessMemo(std::size_t shard_count = 32,
                           std::size_t max_entries = 100'000,
                           std::size_t max_value_bytes = std::size_t{16} * 1024 * 1024);

    [[nodiscard]] auto name() const -> std::string_view override;

    [[nodiscard]] auto lookup(const ContentKey& key, std::span<const std::byte> full_key)
        -> Result<std::optional<Payload>> override;

    [[nodiscard]] auto publish(const ContentKey& key, std::span<const std::byte> full_key,
                               std::span<const std::byte> value) -> Result<void> override;

    [[nodiscard]] auto stats() const -> MemoStats override;

    [[nodiscard]] auto size() const noexcept -> std::size_t;

private:
    struct Entry {
        // The full fingerprint, not just the bytes. The bucket is addressed by
        // (hi % shard_count, lo), so `hi` is otherwise never compared in full and two
        // entries can share a bucket while their 128-bit keys differ. Keeping the key here
        // lets `key_mismatches` mean what it claims: a genuine ContentKey collision.
        ContentKey fp{};
        Payload key;
        Payload value;
    };

    struct Shard {
        mutable std::mutex m;
        std::unordered_map<std::uint64_t, std::vector<Entry>> buckets;
    };

    std::size_t max_entries_{100'000};
    std::size_t max_value_bytes_{std::size_t{16} * 1024 * 1024};
    std::vector<std::unique_ptr<Shard>> shards_{};
    std::atomic<std::size_t> total_entries_{0};

    mutable std::atomic<std::uint64_t> hits_{0};
    mutable std::atomic<std::uint64_t> misses_{0};
    mutable std::atomic<std::uint64_t> publishes_{0};
    mutable std::atomic<std::uint64_t> mismatches_{0};
    mutable std::atomic<std::uint64_t> rejected_{0};
};


// ---------------------------------------------------------------------------
// FileMemo — the same table, made DURABLE (ROADMAP §6.2)
// ---------------------------------------------------------------------------
//
// InProcessMemo dies with the process, so a coordinator restart re-executes everything it
// had already computed. FileMemo outlives the process: an APPEND-ONLY log on disk plus an
// in-memory index of where each record lives.
//
// Append-only suits the data: a memo entry is immutable by construction, because the value is
// a pure function of the key, so nothing ever needs rewriting. The one place the writer does
// NOT simply append is when it repairs a damaged tail, and that case is handled by truncating
// at open (below) rather than by seeking past unverified bytes.
//
// ── HONESTY BOUNDARY (Rule 32) ───────────────────────────────────────────────────────────
//
//   * A hit is confirmed by a FULL-KEY byte comparison, so a ContentKey collision costs an
//     entry, never an answer.
//   * Every record carries its own checksum. A truncated record is caught by the length bound
//     and a damaged one by the checksum; either way it is a MISS and the task is recomputed.
//     A damaged store can cost work; it cannot produce a wrong value.
//   * Loading stops at the FIRST unreadable record rather than skipping it, and the log is then
//     TRUNCATED to the end of the last good record. Scanning past a bad record for ones that
//     merely look right is how a corrupt log starts serving plausible garbage -- and leaving
//     the bytes in place is worse, because a value's own bytes can contain something that
//     parses as a record, and after a short repair those bytes would land on a record boundary.
//   * A genuine I/O ERROR while loading is NOT treated as end-of-log. Refusing to open is the
//     only safe response: a short read taken for EOF would truncate a healthy log at the point
//     the error happened, destroying durable records.
//
// DURABILITY IS BOUNDED, AND THE BOUND IS STATED RATHER THAN OVERSOLD. Every publish issues a
// write and flushes it out of the library's buffers, so records survive the PROCESS dying --
// a crash, a kill, an exit. They are NOT proof against POWER LOSS: reaching stable storage
// needs fsync/fdatasync/FlushFileBuffers, and std::fstream exposes no file descriptor to call
// it on. There is deliberately no "sync" option, because through this API it could only close
// and reopen the stream, which pushes the library's buffers into the OS page cache -- exactly
// what flush() already did -- and would charge two syscalls per publish for no added
// guarantee. Power-loss durability needs the native file API and is not implemented; it is
// better to say so than to ship a flag whose name promises it.
//
// SINGLE WRITER PER PATH. Nothing locks the file. Two FileMemo objects over one path each keep
// their own append offset and will overwrite each other's records. A reader rejects the
// wreckage rather than serving it, so this cannot produce a wrong ANSWER, but it can destroy a
// store. One process, one FileMemo per path. Many threads through ONE FileMemo are fine.
//
// UNBOUNDED, unlike InProcessMemo. There is no max_entries analogue and nothing is ever
// evicted: the log grows until the disk does not accept it, after which publishes are refused
// and the store keeps serving what it already holds. InProcessMemo's entry bound exists to keep
// behaviour reproducible; here the bound is the filesystem.
class FileMemo final : public DistributedMemo {
  public:
    // Opens (creating if absent) the log at `path` and indexes what is already there, then
    // truncates any damaged tail. `create()` rather than a throwing constructor: a bad path, an
    // unreadable file, or a missing directory are ordinary runtime conditions, and so is an I/O
    // error mid-log -- all of them return an honest error instead of a half-built object.
    [[nodiscard]] static auto create(std::filesystem::path path,
                                     std::size_t max_value_bytes = std::size_t{16} * 1024 * 1024)
        -> Result<std::unique_ptr<FileMemo>>;

    ~FileMemo() override = default;

    [[nodiscard]] auto name() const -> std::string_view override;

    [[nodiscard]] auto lookup(const ContentKey& key, std::span<const std::byte> full_key)
        -> Result<std::optional<Payload>> override;

    [[nodiscard]] auto publish(const ContentKey& key, std::span<const std::byte> full_key,
                               std::span<const std::byte> value) -> Result<void> override;

    [[nodiscard]] auto stats() const -> MemoStats override;

    // Records successfully indexed from the log.
    [[nodiscard]] auto size() const -> std::size_t;

    // Whether the log carried a damaged or truncated tail when it was opened, which was then
    // cut away. A boolean, not a count: loading stops at the first bad record, so how many
    // records were lost beyond it is exactly what cannot be known. Worth surfacing because it
    // is otherwise invisible -- the store simply behaves as though it holds less.
    [[nodiscard]] auto log_was_damaged() const -> bool;

    // Byte length of the log's verified prefix, which is also the offset the next record is
    // appended at. After open this is the whole file, because a damaged tail is truncated.
    [[nodiscard]] auto good_prefix_bytes() const -> std::uintmax_t;

  private:
    // Private-tag construction so make_unique works without exposing the constructor, and
    // without the raw `new` the code policy bans.
    struct PrivateTag {
        explicit PrivateTag() = default;
    };

  public:
    explicit FileMemo(PrivateTag) {}

  private:
    struct Located {
        ContentKey fp{};
        std::uintmax_t offset{0};
    };

    enum class ReadOutcome : std::uint8_t {
        ok,        // a complete, checksum-verified record
        end,       // cleanly out of records: EOF or a short/damaged tail
        io_error,  // the stream itself failed; the log's contents are UNKNOWN past here
    };

    [[nodiscard]] auto read_record_at(std::uintmax_t offset, ContentKey& out_fp, Payload& out_key,
                                      Payload& out_value) -> ReadOutcome;

    std::filesystem::path path_{};
    std::size_t max_value_bytes_{std::size_t{16} * 1024 * 1024};
    std::uintmax_t good_prefix_{0};
    bool damaged_{false};

    // Cached because read_record_at consults it to bound an allocation, and that runs once per
    // record on the dispatch path -- a stat() per lookup against a 17 us/task coordinator floor
    // is not free. Only this class writes the file, and always under m_, so it cannot drift.
    std::uintmax_t file_size_{0};

    // One mutex for the whole store. FileMemo is bounded by disk, not by lock contention, and
    // sharding would buy nothing while making the append order -- which IS the file format --
    // harder to reason about.
    mutable std::mutex m_;
    std::map<ContentKey, std::vector<Located>> index_;
    std::fstream file_;

    mutable std::atomic<std::uint64_t> hits_{0};
    mutable std::atomic<std::uint64_t> misses_{0};
    mutable std::atomic<std::uint64_t> publishes_{0};
    mutable std::atomic<std::uint64_t> mismatches_{0};
    mutable std::atomic<std::uint64_t> rejected_{0};
};

// A result may be memoized only if it is a deterministic FUNCTION OF THE KEY.
// ok (0) and math_error (1) qualify: a pure op given identical bytes always produces them.
// bridge_error (2) does NOT: it reports a transport/coordination failure, a property of the
// cluster at one moment, not of the input. Caching it would make a transient network
// fault permanent and would be a wrong answer on every later lookup.
[[nodiscard]] auto is_memoizable_status(int status) noexcept -> bool;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

auto content_key(std::span<const std::byte> bytes) noexcept -> ContentKey {
    constexpr std::uint64_t fnv_prime = 0x00000100000001b3ULL;
    constexpr std::uint64_t hi_offset_basis = 0xcbf29ce484222325ULL;
    constexpr std::uint64_t lo_offset_basis = 0x9e3779b97f4a7c15ULL;

    std::uint64_t hi = hi_offset_basis;
    std::uint64_t lo = lo_offset_basis;

    for (const std::byte b : bytes) {
        const auto byte_val = static_cast<std::uint64_t>(static_cast<std::uint8_t>(b));
        hi ^= byte_val;
        hi *= fnv_prime;

        lo ^= (byte_val ^ 0xA5ULL);
        lo *= fnv_prime;
    }

    return ContentKey{.hi = hi, .lo = lo};
}

auto to_hex(const ContentKey& key) -> std::string {
    return std::format("{:016x}{:016x}", key.hi, key.lo);
}

auto is_memoizable_status(int status) noexcept -> bool {
    // 0 == ok, 1 == math_error: both are deterministic functions of the input.
    // 2 == bridge_error (and any unknown status): transient transport/system failure,
    // not deterministic, MUST NOT be memoized.
    return status == 0 || status == 1;
}

InProcessMemo::InProcessMemo(std::size_t shard_count, std::size_t max_entries,
                             std::size_t max_value_bytes)
    : max_entries_(max_entries), max_value_bytes_(max_value_bytes) {
    // Clamp rather than assert: a caller asking for zero shards wants a working table, not a
    // crash, and an assert placed after the clamp could never fire anyway.
    if (shard_count == 0) {
        shard_count = 1;
    }
    shards_.reserve(shard_count);
    for (std::size_t i = 0; i < shard_count; ++i) {
        shards_.push_back(std::make_unique<Shard>());
    }
}

auto InProcessMemo::name() const -> std::string_view {
    return "in_process";
}

auto InProcessMemo::lookup(const ContentKey& key, std::span<const std::byte> full_key)
    -> Result<std::optional<Payload>> {
    assert(!shards_.empty() && "InProcessMemo has no shards");
    const std::size_t shard_idx = static_cast<std::size_t>(key.hi % shards_.size());
    Shard& shard = *shards_[shard_idx];

    const std::lock_guard<std::mutex> lock(shard.m);
    const auto it = shard.buckets.find(key.lo);
    if (it == shard.buckets.end()) {
        misses_.fetch_add(1, std::memory_order_relaxed);
        return std::optional<Payload>{std::nullopt};
    }

    for (const Entry& entry : it->second) {
        // ── EXACTNESS RULE (Code Policy Rule 32) ───────────────────────────
        // A 128-bit ContentKey fingerprint is not a mathematical proof of byte
        // equality. To guarantee that a cache hit is observationally identical to
        // executing the underlying pure computation, we verify the full key bytes
        // byte-for-byte. A fingerprint collision must cost a lost cache entry
        // (treated as a miss), NEVER a wrong answer.
        if (entry.key.size() == full_key.size() &&
            std::equal(entry.key.begin(), entry.key.end(), full_key.begin())) {
            hits_.fetch_add(1, std::memory_order_relaxed);
            return std::optional<Payload>{entry.value};
        }
    }

    // No entry matched full_key byte-for-byte. `key_mismatches` counts only a GENUINE
    // fingerprint collision — a non-empty bucket whose occupants all disagree with
    // full_key — because that counter is the evidence that the exactness rule fired.
    // An empty bucket is an ordinary miss and must NOT be reported as a collision, or
    // the one statistic that proves the rule works becomes noise.
    //
    // A mismatch is ALSO a miss: `hits + misses` must equal the number of lookups, or
    // measured hit rates silently under-count their denominator.
    // Count a mismatch only for an occupant whose FULL 128-bit fingerprint equals `key`
    // while its bytes differ. Sharing a bucket is not a collision: the bucket is addressed
    // by (hi % shard_count, lo), so occupants may differ in `hi` entirely. Counting those
    // would let ordinary traffic inflate the one statistic offered as evidence that the
    // exactness rule fired.
    //
    // A mismatch is ALSO a miss: `hits + misses` must equal the number of lookups, or
    // measured hit rates silently under-count their denominator.
    const bool true_collision =
        std::ranges::any_of(it->second, [&key](const Entry& e) { return e.fp == key; });
    if (true_collision) {
        mismatches_.fetch_add(1, std::memory_order_relaxed);
    }
    misses_.fetch_add(1, std::memory_order_relaxed);
    return std::optional<Payload>{std::nullopt};
}

auto InProcessMemo::publish(const ContentKey& key, std::span<const std::byte> full_key,
                            std::span<const std::byte> value) -> Result<void> {
    assert(!shards_.empty() && "InProcessMemo has no shards");
    const std::size_t shard_idx = static_cast<std::size_t>(key.hi % shards_.size());
    Shard& shard = *shards_[shard_idx];

    const std::lock_guard<std::mutex> lock(shard.m);

    // Look the bucket up with find(), NEVER with operator[]. operator[] would
    // default-construct an EMPTY bucket, and the capacity check below can then return
    // before anything is inserted — leaving that empty bucket behind for good. A later
    // lookup would find it, match none of its (zero) occupants, and report a fingerprint
    // collision that never happened, corrupting the very counter that evidences the
    // exactness rule.
    const auto it = shard.buckets.find(key.lo);

    if (it != shard.buckets.end()) {
        for (const Entry& entry : it->second) {
            // EXACTNESS RULE: byte-for-byte verification.
            if (entry.key.size() == full_key.size() &&
                std::equal(entry.key.begin(), entry.key.end(), full_key.begin())) {
                // First writer wins: pure operations guarantee identical results.
                // Counts as a successful publish.
                publishes_.fetch_add(1, std::memory_order_relaxed);
                return {};
            }
        }
    }

    // Oversize values are refused, but only AFTER the duplicate scan above: re-publishing a
    // key already held is a no-op, and reporting it as a capacity rejection would misattribute
    // it. Refusing to cache is not an error either way -- the caller simply recomputes.
    if (value.size() > max_value_bytes_) {
        rejected_.fetch_add(1, std::memory_order_relaxed);
        return {};
    }

    // Check table entry capacity.
    // Bounded-capacity, no-eviction table: when capacity is reached, new entries are
    // rejected without eviction to ensure deterministic and reproducible execution.
    // The bound is advisory under concurrency (two threads on different shards may both
    // observe headroom and both insert); it caps growth, it is not a hard invariant.
    if (total_entries_.load(std::memory_order_relaxed) >= max_entries_) {
        rejected_.fetch_add(1, std::memory_order_relaxed);
        return {};
    }

    // Build the entry BEFORE touching the map. If either copy throws, the map is untouched
    // and no empty bucket is left behind for a later lookup to find.
    Entry entry{.fp = key,
                .key = Payload(full_key.begin(), full_key.end()),
                .value = Payload(value.begin(), value.end())};

    std::vector<Entry>& bucket =
        (it != shard.buckets.end()) ? it->second : shard.buckets[key.lo];
    bucket.push_back(std::move(entry));
    total_entries_.fetch_add(1, std::memory_order_relaxed);
    publishes_.fetch_add(1, std::memory_order_relaxed);
    return {};
}

auto InProcessMemo::stats() const -> MemoStats {
    return MemoStats{
        .hits = hits_.load(std::memory_order_relaxed),
        .misses = misses_.load(std::memory_order_relaxed),
        .publishes = publishes_.load(std::memory_order_relaxed),
        .key_mismatches = mismatches_.load(std::memory_order_relaxed),
        .rejected = rejected_.load(std::memory_order_relaxed)
    };
}

auto InProcessMemo::size() const noexcept -> std::size_t {
    return total_entries_.load(std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// FileMemo implementation
// ---------------------------------------------------------------------------
namespace filememo_detail {

// Record framing, little-endian throughout so a store written on one machine is readable on
// another — the same reason ContentKey is defined by byte arithmetic rather than std::hash.
//
//   u32 magic | u16 version=1 | u16 reserved=0 | u64 fp_hi | u64 fp_lo
//   u32 key_len | u32 value_len | key bytes | value bytes | u64 checksum
//
// The checksum covers the header, the key and the value — everything before itself — so a
// record boundary cannot be shifted, nor the key/value split moved, without failing it.
//
// The ContentKey is STORED, not recomputed on load. `publish` indexes under the key the caller
// supplied, and the interface nowhere requires that to equal content_key(full_key); recomputing
// it at load would silently re-key such an entry and make it unreachable after a restart, so
// FileMemo would honour a different contract than InProcessMemo across exactly one event.
inline constexpr std::uint32_t k_record_magic = 0x524D434E;  // bytes 'N','C','M','R' on disk
inline constexpr std::uint16_t k_record_version = 1;
inline constexpr std::size_t k_header_bytes = 32;
inline constexpr std::size_t k_checksum_bytes = 8;
inline constexpr std::size_t k_min_record_bytes = k_header_bytes + k_checksum_bytes;

inline auto put_u16(std::uint16_t v, Payload& out) -> void {
    out.push_back(static_cast<std::byte>(v & 0xFFu));
    out.push_back(static_cast<std::byte>((v >> 8) & 0xFFu));
}

inline auto put_u32(std::uint32_t v, Payload& out) -> void {
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xFFu));
    }
}

inline auto put_u64(std::uint64_t v, Payload& out) -> void {
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xFFu));
    }
}

[[nodiscard]] inline auto get_u16(std::span<const std::byte> b, std::size_t off) -> std::uint16_t {
    return static_cast<std::uint16_t>(static_cast<std::uint16_t>(b[off]) |
                                      (static_cast<std::uint16_t>(b[off + 1]) << 8));
}

[[nodiscard]] inline auto get_u32(std::span<const std::byte> b, std::size_t off) -> std::uint32_t {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
        v |= static_cast<std::uint32_t>(b[off + static_cast<std::size_t>(i)]) << (8 * i);
    }
    return v;
}

[[nodiscard]] inline auto get_u64(std::span<const std::byte> b, std::size_t off) -> std::uint64_t {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<std::uint64_t>(b[off + static_cast<std::size_t>(i)]) << (8 * i);
    }
    return v;
}

// Reuse the module's own fingerprint as the record checksum. It is not cryptographic and does
// not need to be: it guards against a torn write and bit rot, not an adversary with write
// access, and a record that passes it is STILL verified by a full-key comparison before its
// value is served.
[[nodiscard]] inline auto checksum(std::span<const std::byte> bytes) noexcept -> std::uint64_t {
    return content_key(bytes).hi;
}

}  // namespace filememo_detail

auto FileMemo::create(std::filesystem::path path, std::size_t max_value_bytes)
    -> Result<std::unique_ptr<FileMemo>> {
    namespace fd = filememo_detail;

    auto memo = std::make_unique<FileMemo>(PrivateTag{});
    memo->path_ = std::move(path);
    memo->max_value_bytes_ = max_value_bytes;

    std::error_code ec;
    if (!std::filesystem::exists(memo->path_, ec)) {
        // Create, then reopen for update: opening a missing file with in|out fails rather than
        // creating it, which is why this is two steps.
        std::ofstream make(memo->path_, std::ios::binary | std::ios::app);
        if (!make) {
            return make_error<std::unique_ptr<FileMemo>>(MathError::domain_error);
        }
    }

    memo->file_.open(memo->path_, std::ios::binary | std::ios::in | std::ios::out);
    if (!memo->file_) {
        return make_error<std::unique_ptr<FileMemo>>(MathError::domain_error);
    }

    {
        std::error_code size_ec;
        const auto initial = std::filesystem::file_size(memo->path_, size_ec);
        if (size_ec) {
            return make_error<std::unique_ptr<FileMemo>>(MathError::domain_error);
        }
        memo->file_size_ = initial;
    }

    // Index what is already on disk, stopping at the FIRST record that does not verify.
    std::uintmax_t offset = 0;
    for (;;) {
        ContentKey fp{};
        Payload key;
        Payload value;
        const ReadOutcome outcome = memo->read_record_at(offset, fp, key, value);
        if (outcome == ReadOutcome::io_error) {
            // NOT end-of-log. Everything past here is unknown, and taking it for EOF would let
            // the next publish overwrite healthy durable records at this offset. Refuse.
            return make_error<std::unique_ptr<FileMemo>>(MathError::domain_error);
        }
        if (outcome == ReadOutcome::end) {
            break;
        }
        auto& bucket = memo->index_[fp];
        const bool already = std::ranges::any_of(bucket, [&](const Located& loc) {
            ContentKey k_fp{};
            Payload k;
            Payload v;
            return memo->read_record_at(loc.offset, k_fp, k, v) == ReadOutcome::ok && k == key;
        });
        // First writer wins on disk exactly as in memory: the ops are pure, so a later record
        // for the same key carries the same value, and preferring the earlier one keeps the
        // store's meaning independent of append order.
        if (!already) {
            bucket.push_back(Located{.fp = fp, .offset = offset});
        }
        offset += static_cast<std::uintmax_t>(fd::k_header_bytes + key.size() + value.size() +
                                              fd::k_checksum_bytes);
    }
    memo->good_prefix_ = offset;

    if (memo->file_size_ > offset) {
        // A damaged or truncated tail. CUT IT, rather than leaving it and appending over it.
        // Leaving it would keep the store permanently flagged as damaged even after a repair,
        // and worse: a repair record shorter than the remnant leaves the remnant's middle
        // sitting on a record boundary, where bytes an op's own RESULT controlled become parse
        // candidates. Truncation removes that class entirely.
        memo->damaged_ = true;
        std::error_code trunc_ec;
        std::filesystem::resize_file(memo->path_, offset, trunc_ec);
        if (trunc_ec) {
            return make_error<std::unique_ptr<FileMemo>>(MathError::domain_error);
        }
        memo->file_size_ = offset;
        // The stream cached a size from before the truncation; reopen so later reads and the
        // append cursor agree with the file that now exists.
        memo->file_.close();
        memo->file_.open(memo->path_, std::ios::binary | std::ios::in | std::ios::out);
        if (!memo->file_) {
            return make_error<std::unique_ptr<FileMemo>>(MathError::domain_error);
        }
    }

    memo->file_.clear();
    memo->file_.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!memo->file_) {
        return make_error<std::unique_ptr<FileMemo>>(MathError::domain_error);
    }

    return memo;
}

auto FileMemo::read_record_at(std::uintmax_t offset, ContentKey& out_fp, Payload& out_key,
                              Payload& out_value) -> ReadOutcome {
    namespace fd = filememo_detail;

    // A record cannot even fit: cleanly out of records, not an error.
    if (offset > file_size_ || file_size_ - offset < fd::k_min_record_bytes) {
        return ReadOutcome::end;
    }

    file_.clear();
    file_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (file_.bad()) {
        return ReadOutcome::io_error;
    }
    if (!file_) {
        return ReadOutcome::end;
    }

    std::array<char, fd::k_header_bytes> header_buf{};
    file_.read(header_buf.data(), static_cast<std::streamsize>(header_buf.size()));
    if (file_.bad()) {
        return ReadOutcome::io_error;
    }
    if (file_.gcount() != static_cast<std::streamsize>(header_buf.size())) {
        return ReadOutcome::end;
    }
    const auto header = std::as_bytes(std::span<const char>(header_buf));

    if (fd::get_u32(header, 0) != fd::k_record_magic ||
        fd::get_u16(header, 4) != fd::k_record_version) {
        return ReadOutcome::end;
    }
    // `reserved` must be zero. Accepting a non-zero value would forfeit the field for any
    // future use, because old readers would already have tolerated records that set it.
    if (fd::get_u16(header, 6) != 0) {
        return ReadOutcome::end;
    }

    const ContentKey fp{.hi = fd::get_u64(header, 8), .lo = fd::get_u64(header, 16)};
    const std::uint32_t key_len = fd::get_u32(header, 24);
    const std::uint32_t value_len = fd::get_u32(header, 28);

    // Bound the allocation BEFORE reserving anything. key_len and value_len are u32, so `need`
    // cannot overflow uintmax_t, and `offset` never exceeds the file size. The configured value
    // bound applies on the way IN as well as on the way out, so a foreign or hand-written file
    // cannot make this allocate more than this store was configured to hold.
    const std::uintmax_t need = static_cast<std::uintmax_t>(fd::k_header_bytes) + key_len +
                                value_len + fd::k_checksum_bytes;
    if (need > file_size_ - offset || value_len > max_value_bytes_) {
        return ReadOutcome::end;
    }

    Payload body(static_cast<std::size_t>(key_len) + value_len);
    if (!body.empty()) {
        file_.read(reinterpret_cast<char*>(body.data()),
                   static_cast<std::streamsize>(body.size()));
        if (file_.bad()) {
            return ReadOutcome::io_error;
        }
        if (file_.gcount() != static_cast<std::streamsize>(body.size())) {
            return ReadOutcome::end;
        }
    }

    std::array<char, fd::k_checksum_bytes> sum_buf{};
    file_.read(sum_buf.data(), static_cast<std::streamsize>(sum_buf.size()));
    if (file_.bad()) {
        return ReadOutcome::io_error;
    }
    if (file_.gcount() != static_cast<std::streamsize>(sum_buf.size())) {
        return ReadOutcome::end;
    }
    const std::uint64_t stored = fd::get_u64(std::as_bytes(std::span<const char>(sum_buf)), 0);

    Payload covered;
    covered.reserve(fd::k_header_bytes + body.size());
    covered.insert(covered.end(), header.begin(), header.end());
    covered.insert(covered.end(), body.begin(), body.end());
    if (fd::checksum(covered) != stored) {
        return ReadOutcome::end;
    }

    out_fp = fp;
    out_key.assign(body.begin(), body.begin() + static_cast<std::ptrdiff_t>(key_len));
    out_value.assign(body.begin() + static_cast<std::ptrdiff_t>(key_len), body.end());
    return ReadOutcome::ok;
}

auto FileMemo::name() const -> std::string_view {
    return "file";
}

auto FileMemo::lookup(const ContentKey& key, std::span<const std::byte> full_key)
    -> Result<std::optional<Payload>> {
    const std::lock_guard<std::mutex> lock(m_);

    const auto it = index_.find(key);
    if (it == index_.end()) {
        misses_.fetch_add(1, std::memory_order_relaxed);
        return std::optional<Payload>{std::nullopt};
    }

    bool true_collision = false;
    for (const Located& loc : it->second) {
        ContentKey stored_fp{};
        Payload stored_key;
        Payload stored_value;
        if (read_record_at(loc.offset, stored_fp, stored_key, stored_value) != ReadOutcome::ok) {
            // Indexed but no longer readable. A miss, and the task is recomputed -- never a
            // guess at what the record used to hold.
            continue;
        }
        // ── EXACTNESS RULE (Code Policy Rule 32) ───────────────────────────
        // The checksum proves the record is INTACT; it does not prove it is THIS key. Only a
        // full byte comparison does, and a fingerprint collision must cost an entry rather than
        // yield another task's answer.
        if (stored_key.size() == full_key.size() &&
            std::equal(stored_key.begin(), stored_key.end(), full_key.begin())) {
            hits_.fetch_add(1, std::memory_order_relaxed);
            return std::optional<Payload>{std::move(stored_value)};
        }
        // Count a mismatch only for a record whose FULL 128-bit fingerprint equals `key` while
        // its bytes differ -- the same strict definition InProcessMemo uses, so the counter
        // means the same thing in both and cannot be inflated by ordinary traffic.
        if (stored_fp == key) {
            true_collision = true;
        }
    }

    if (true_collision) {
        mismatches_.fetch_add(1, std::memory_order_relaxed);
    }
    misses_.fetch_add(1, std::memory_order_relaxed);
    return std::optional<Payload>{std::nullopt};
}

auto FileMemo::publish(const ContentKey& key, std::span<const std::byte> full_key,
                       std::span<const std::byte> value) -> Result<void> {
    namespace fd = filememo_detail;

    const std::lock_guard<std::mutex> lock(m_);

    // Already held? First writer wins, and re-appending would grow the log without adding
    // information. Counted as a publish, matching InProcessMemo.
    if (const auto it = index_.find(key); it != index_.end()) {
        for (const Located& loc : it->second) {
            ContentKey k_fp{};
            Payload k;
            Payload v;
            if (read_record_at(loc.offset, k_fp, k, v) == ReadOutcome::ok &&
                k.size() == full_key.size() && std::equal(k.begin(), k.end(), full_key.begin())) {
                publishes_.fetch_add(1, std::memory_order_relaxed);
                return {};
            }
        }
    }

    if (value.size() > max_value_bytes_ ||
        full_key.size() > std::numeric_limits<std::uint32_t>::max() ||
        value.size() > std::numeric_limits<std::uint32_t>::max()) {
        rejected_.fetch_add(1, std::memory_order_relaxed);
        return {};
    }

    Payload record;
    record.reserve(fd::k_header_bytes + full_key.size() + value.size() + fd::k_checksum_bytes);
    fd::put_u32(fd::k_record_magic, record);
    fd::put_u16(fd::k_record_version, record);
    fd::put_u16(0, record);  // reserved
    fd::put_u64(key.hi, record);
    fd::put_u64(key.lo, record);
    fd::put_u32(static_cast<std::uint32_t>(full_key.size()), record);
    fd::put_u32(static_cast<std::uint32_t>(value.size()), record);
    record.insert(record.end(), full_key.begin(), full_key.end());
    record.insert(record.end(), value.begin(), value.end());
    fd::put_u64(fd::checksum(record), record);

    const std::uintmax_t offset = good_prefix_;

    file_.clear();
    file_.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
    file_.write(reinterpret_cast<const char*>(record.data()),
                static_cast<std::streamsize>(record.size()));
    // Flush before checking: a buffered failure surfaces here rather than being swallowed.
    file_.flush();
    if (!file_) {
        // A write that did not land must not be indexed -- indexing it would promise a record
        // the file does not contain. `good_prefix_` is left where it was, so the next publish
        // overwrites whatever partial bytes reached the disk.
        file_.clear();
        std::error_code resync_ec;
        if (const auto actual = std::filesystem::file_size(path_, resync_ec); !resync_ec) {
            file_size_ = std::max(file_size_, actual);
        }
        rejected_.fetch_add(1, std::memory_order_relaxed);
        return {};
    }

    good_prefix_ = offset + record.size();
    file_size_ = std::max(file_size_, good_prefix_);
    index_[key].push_back(Located{.fp = key, .offset = offset});
    publishes_.fetch_add(1, std::memory_order_relaxed);
    return {};
}

auto FileMemo::stats() const -> MemoStats {
    return MemoStats{.hits = hits_.load(std::memory_order_relaxed),
                     .misses = misses_.load(std::memory_order_relaxed),
                     .publishes = publishes_.load(std::memory_order_relaxed),
                     .key_mismatches = mismatches_.load(std::memory_order_relaxed),
                     .rejected = rejected_.load(std::memory_order_relaxed)};
}

auto FileMemo::size() const -> std::size_t {
    const std::lock_guard<std::mutex> lock(m_);
    std::size_t n = 0;
    for (const auto& [fp, bucket] : index_) {
        n += bucket.size();
    }
    return n;
}

auto FileMemo::log_was_damaged() const -> bool {
    const std::lock_guard<std::mutex> lock(m_);
    return damaged_;
}

auto FileMemo::good_prefix_bytes() const -> std::uintmax_t {
    const std::lock_guard<std::mutex> lock(m_);
    return good_prefix_;
}

}  // namespace nimblecas
