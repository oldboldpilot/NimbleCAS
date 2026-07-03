// NimbleCAS parallel runtime & tree combinators (ROADMAP 3; see
// docs/architecture/parallel-tree-computation.md).
// @author Olumuyiwa Oluwasanmi
//
// A thin, deterministic fork-join layer over Intel oneTBB (Linux/macOS) and Microsoft
// PPL (Windows), with a serial fallback below a grain-size cutoff. Because NimbleCAS
// expressions are immutable (CowPtr), independent subtrees can be transformed
// concurrently with no locking; these combinators express that safely.
//
// Module note: the TBB/PPL headers live in the global module fragment and are NOT
// visible to importers, so all runtime calls are confined to the concrete, non-template
// functions (for_ranges / invoke2). The exported generic combinators are thin templates
// that call those — they reference only `std` and the concrete functions, both visible
// across the module boundary.

module;
#ifdef _WIN32
#include <ppl.h>
#else
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#endif

export module nimblecas.parallel;

import std;

export namespace nimblecas::parallel {

// Work below this many items is done serially (forking would cost more than it saves).
// Tunable; to be profiled with perf on representative large expressions (Rule 43/58).
inline constexpr std::size_t default_grain = 256;

// Runs `body` over a partition of disjoint sub-ranges [begin, end) covering [0, n),
// in parallel (work-stealing) when n >= grain, else serially. Per-index work must be
// independent; the union of the ranges is exactly [0, n).
auto for_ranges(std::size_t n, std::size_t grain,
                const std::function<void(std::size_t, std::size_t)>& body) -> void;

// Runs two independent tasks, possibly concurrently (fork-join).
auto invoke2(const std::function<void()>& f, const std::function<void()>& g) -> void;

// Worker threads available to the runtime (informational; >= 1).
[[nodiscard]] auto max_concurrency() noexcept -> unsigned;

// Order-preserving parallel map by index: result[i] = fn(i) for i in [0, n).
// Deterministic (result depends only on fn, not on scheduling). Serial below grain.
template <typename Fn>
[[nodiscard]] auto transform_index(std::size_t n, Fn fn, std::size_t grain = default_grain)
    -> std::vector<std::invoke_result_t<Fn&, std::size_t>> {
    using R = std::invoke_result_t<Fn&, std::size_t>;
    // optional slots so R need not be default-constructible (Expr is not); each index
    // writes its own disjoint slot, so concurrent fills never race.
    std::vector<std::optional<R>> slots(n);
    for_ranges(n, grain, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            slots[i].emplace(fn(i));
        }
    });
    std::vector<R> out;
    out.reserve(n);
    for (auto& slot : slots) {
        out.push_back(std::move(*slot));
    }
    return out;
}

// Order-preserving parallel map over a range: result[i] = fn(in[i]).
template <typename T, typename Fn>
[[nodiscard]] auto transform(std::span<const T> in, Fn fn, std::size_t grain = default_grain)
    -> std::vector<std::invoke_result_t<Fn&, const T&>> {
    return transform_index(
        in.size(), [&](std::size_t i) { return fn(in[i]); }, grain);
}

}  // namespace nimblecas::parallel

// ===========================================================================
// Implementation (TBB/PPL confined here, where the global-fragment headers are visible).
// ===========================================================================
namespace nimblecas::parallel {

auto for_ranges(std::size_t n, std::size_t grain,
                const std::function<void(std::size_t, std::size_t)>& body) -> void {
    if (n == 0) {
        return;
    }
    if (n < grain) {
        body(0, n);
        return;
    }
#ifdef _WIN32
    const std::size_t block_count = (n + grain - 1) / grain;
    concurrency::parallel_for(std::size_t{0}, block_count, [&](std::size_t block) {
        const std::size_t begin = block * grain;
        body(begin, std::min(n, begin + grain));
    });
#else
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, n, grain),
                      [&](const tbb::blocked_range<std::size_t>& r) { body(r.begin(), r.end()); });
#endif
}

auto invoke2(const std::function<void()>& f, const std::function<void()>& g) -> void {
#ifdef _WIN32
    concurrency::parallel_invoke(f, g);
#else
    tbb::parallel_invoke(f, g);
#endif
}

auto max_concurrency() noexcept -> unsigned {
    const unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1U : hw;
}

}  // namespace nimblecas::parallel
