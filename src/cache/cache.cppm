// NimbleCAS concurrent hash-consing / memoization (ROADMAP 6.2; see
// docs/architecture/parallel-tree-computation.md section 5).
// @author Olumuyiwa Oluwasanmi
//
// A thread-safe memo for pure Expr -> Result<Expr> transforms (e.g. simplify,
// differentiate). Keyed by structural identity — the O(1) memoised structural_hash
// picks a shard and bucket, is_equivalent_to resolves collisions. Because Expr is
// immutable (COW), a memoised result is safe to share across threads, so each unique
// subexpression is transformed once even under parallel recursion. Sharding keeps
// concurrent lookups on distinct keys from contending.

module;
#include <cassert>

export module nimblecas.cache;

import std;
import nimblecas.core;
import nimblecas.symbolic;

export namespace nimblecas {

class ExprMemo {
public:
    explicit ExprMemo(std::size_t shard_count = 64);

    // Returns the memoised result for `key`, computing it via `compute` on first
    // sight and caching it. `compute` MUST be a pure function of `key` (its result
    // may be reused for any structurally-equal expression). Safe for concurrent calls
    // — `compute` runs WITHOUT holding any shard lock, so it may recurse into
    // get_or_compute on other keys without risk of deadlock.
    [[nodiscard]] auto get_or_compute(const Expr& key,
                                      const std::function<Result<Expr>()>& compute)
        -> Result<Expr>;

    // Number of distinct entries cached (for tests / introspection).
    [[nodiscard]] auto size() const -> std::size_t;

private:
    struct Shard {
        mutable std::mutex mtx;
        // structural_hash -> collision bucket of (key, memoised result)
        std::unordered_map<std::size_t, std::vector<std::pair<Expr, Result<Expr>>>> buckets;
    };
    std::vector<std::unique_ptr<Shard>> shards_;

    [[nodiscard]] auto shard_for(std::size_t hash) const -> Shard& {
        return *shards_[hash % shards_.size()];
    }
};

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

ExprMemo::ExprMemo(std::size_t shard_count) {
    assert(shard_count > 0 && "ExprMemo needs at least one shard");
    shards_.reserve(shard_count);
    for (std::size_t i = 0; i < shard_count; ++i) {
        shards_.push_back(std::make_unique<Shard>());
    }
}

auto ExprMemo::get_or_compute(const Expr& key, const std::function<Result<Expr>()>& compute)
    -> Result<Expr> {
    const std::size_t h = key.structural_hash();
    Shard& shard = shard_for(h);

    // 1. Fast path: look up under the shard lock.
    {
        const std::lock_guard<std::mutex> lock(shard.mtx);
        if (auto it = shard.buckets.find(h); it != shard.buckets.end()) {
            for (const auto& [cached_key, cached_value] : it->second) {
                if (cached_key.is_equivalent_to(key)) {
                    return cached_value;
                }
            }
        }
    }

    // 2. Compute WITHOUT the lock — `compute` may recurse into get_or_compute on
    //    other keys (other shards); holding this shard's lock would risk deadlock.
    Result<Expr> result = compute();

    // 3. Insert, double-checking in case another thread computed the same key.
    {
        const std::lock_guard<std::mutex> lock(shard.mtx);
        auto& bucket = shard.buckets[h];
        for (const auto& [cached_key, cached_value] : bucket) {
            if (cached_key.is_equivalent_to(key)) {
                return cached_value;
            }
        }
        bucket.emplace_back(key, result);
    }
    return result;
}

auto ExprMemo::size() const -> std::size_t {
    std::size_t total = 0;
    for (const auto& shard : shards_) {
        const std::lock_guard<std::mutex> lock(shard->mtx);
        for (const auto& [hash, bucket] : shard->buckets) {
            total += bucket.size();
        }
    }
    return total;
}

}  // namespace nimblecas
