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
// Backend selection (Code Policy: PPL on Windows, TBB elsewhere):
//   Windows      -> Microsoft PPL (Concurrency Runtime, part of the MSVC STL runtime)
//   Linux/macOS  -> Intel oneTBB   (when its headers are available)
//   otherwise    -> serial fallback (identical results, no parallelism)
#if defined(_WIN32)
#  include <ppl.h>
#  define NIMBLECAS_PARALLEL_BACKEND_PPL 1
#elif __has_include(<tbb/parallel_for.h>)
#  include <tbb/blocked_range.h>
#  include <tbb/parallel_for.h>
#  include <tbb/parallel_invoke.h>
#  define NIMBLECAS_PARALLEL_BACKEND_TBB 1
#else
#  define NIMBLECAS_PARALLEL_BACKEND_SERIAL 1
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

// Name of the active backend: "ppl" (Windows), "tbb" (Linux/macOS), or "serial".
[[nodiscard]] auto backend() noexcept -> std::string_view;

// Order-preserving parallel map by index: result[i] = fn(i) for i in [0, n).
// Deterministic (result depends only on fn, not on scheduling). Serial below grain.
// PRECONDITION: fn must be safe to invoke concurrently for distinct indices — i.e.
// stateless or synchronising internally. In NimbleCAS this holds because the tree is
// immutable (CowPtr), so fn only reads shared subexpressions and produces new ones.
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
#if defined(NIMBLECAS_PARALLEL_BACKEND_PPL)
    const std::size_t block_count = (n + grain - 1) / grain;
    concurrency::parallel_for(std::size_t{0}, block_count, [&](std::size_t block) {
        const std::size_t begin = block * grain;
        body(begin, std::min(n, begin + grain));
    });
#elif defined(NIMBLECAS_PARALLEL_BACKEND_TBB)
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, n, grain),
                      [&](const tbb::blocked_range<std::size_t>& r) { body(r.begin(), r.end()); });
#else
    body(0, n);  // serial fallback
#endif
}

auto invoke2(const std::function<void()>& f, const std::function<void()>& g) -> void {
#if defined(NIMBLECAS_PARALLEL_BACKEND_PPL)
    concurrency::parallel_invoke(f, g);
#elif defined(NIMBLECAS_PARALLEL_BACKEND_TBB)
    tbb::parallel_invoke(f, g);
#else
    f();
    g();
#endif
}

auto backend() noexcept -> std::string_view {
#if defined(NIMBLECAS_PARALLEL_BACKEND_PPL)
    return "ppl";
#elif defined(NIMBLECAS_PARALLEL_BACKEND_TBB)
    return "tbb";
#else
    return "serial";
#endif
}

auto max_concurrency() noexcept -> unsigned {
    const unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1U : hw;
}

}  // namespace nimblecas::parallel
