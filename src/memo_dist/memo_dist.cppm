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

}  // namespace nimblecas
