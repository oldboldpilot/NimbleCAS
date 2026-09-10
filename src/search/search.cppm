// NimbleCAS graph & tree search and dynamic programming (parallel algorithmics).
// @author Olumuyiwa Oluwasanmi
//
// A deterministic toolkit of graph/tree traversal, shortest-path, metaheuristic, and
// dynamic-programming routines. Graphs are described abstractly: a node is a
// std::int64_t id and a `SuccessorFn` returns each node's out-neighbours in the exact
// order they should be expanded. Goals are predicates and edge weights are functions,
// so no adjacency structure is ever materialised by these routines.
//
// DETERMINISM CONTRACT: every function returns the SAME result regardless of how many
// worker threads the parallel backend uses. Frontiers, priority queues, and tie-breaks
// are all ordered (lowest node id / first-in-expansion-order wins), and the one parallel
// traversal here (`parallel_bfs_levels`) merges its per-node results through an ordered
// set before returning, so it reproduces a serial level-BFS bit-for-bit.
//
// The dynamic-programming section fills dense tables of exact integers. Those tables --
// and the anti-diagonal wavefront used by `edit_distance_parallel` -- are the regular,
// rectangular, data-parallel shapes a GPU backend would offload: each independent cell of
// an anti-diagonal is a pure function of already-settled cells, so a whole diagonal maps
// to one `parallel::transform_index` launch. There is no CUDA here; the structure is
// arranged so a device kernel could drop in unchanged. All integer accumulations are
// overflow-guarded and every failure travels the railway (Result<T> / MathError); nothing
// throws.

export module nimblecas.search;

import std;
import nimblecas.core;
import nimblecas.parallel;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// Graph description callbacks. A node is an integer id; the successor function
// defines the expansion order; the goal predicate defines acceptance; the cost
// function returns a NON-NEGATIVE edge weight; the heuristic is an (admissible)
// lower bound on the remaining cost to a goal.
// ---------------------------------------------------------------------------
using SuccessorFn = std::function<std::vector<std::int64_t>(std::int64_t)>;
using GoalFn = std::function<bool(std::int64_t)>;
using CostFn = std::function<std::int64_t(std::int64_t, std::int64_t)>;
using HeuristicFn = std::function<std::int64_t(std::int64_t)>;

// Fewest-edges path from `start` to the first node satisfying `goal`, found by an
// iterative FIFO breadth-first search with a visited set. The returned path includes
// both `start` and the goal node. If `start` already satisfies `goal` the path is just
// {start}. Returns undefined_value when no goal node is reachable from `start`.
[[nodiscard]] auto bfs(std::int64_t start, const GoalFn& goal, const SuccessorFn& successors)
    -> Result<std::vector<std::int64_t>>;

// Depth-first search to the first goal within `max_depth` edges of `start`, expressed
// recursively. Depth of `start` is 0; a node at depth `max_depth` is tested for the goal
// but its successors are not expanded. Successors are explored in returned order and a
// persistent visited set prevents re-expansion. Returns the path (incl. start & goal),
// or undefined_value if no goal is found within the bound. domain_error if max_depth < 0.
[[nodiscard]] auto dfs_recursive(std::int64_t start, GoalFn goal, SuccessorFn successors,
                                 std::int64_t max_depth) -> Result<std::vector<std::int64_t>>;

// Iterative (explicit-stack) form of `dfs_recursive`. It pushes each node's successors in
// reverse so the leftmost is popped first and gates acceptance/expansion at pop time,
// which reproduces the recursive pre-order exactly: for any fixed expansion order the two
// return the identical first-goal path. Same bound semantics and errors as above.
[[nodiscard]] auto dfs_iterative(std::int64_t start, const GoalFn& goal,
                                 const SuccessorFn& successors, std::int64_t max_depth)
    -> Result<std::vector<std::int64_t>>;

// Iterative-deepening DFS: repeated depth-limited searches with the bound growing from 0
// up to `max_depth`, returning the SHALLOWEST goal (fewest edges), matching a BFS-style
// depth while keeping DFS's linear memory. Cycles are broken with an on-path (ancestor)
// check rather than a global visited set, so a goal reachable within the bound is never
// missed. Path includes start & goal; undefined_value if unreachable within the bound;
// domain_error if max_depth < 0.
[[nodiscard]] auto iterative_deepening_dfs(std::int64_t start, GoalFn goal,
                                           SuccessorFn successors, std::int64_t max_depth)
    -> Result<std::vector<std::int64_t>>;

// Minimum-cost path from `start` to the first goal under NON-NEGATIVE edge weights,
// returned as (path, total_cost). A binary-heap priority queue ordered by (distance,
// node id) makes the search deterministic; on equal tentative distances the predecessor
// with the lower node id wins. Returns domain_error if any traversed edge weight is
// negative, overflow if a cost sum exceeds std::int64_t, and undefined_value if no goal
// is reachable.
[[nodiscard]] auto dijkstra(std::int64_t start, const GoalFn& goal, const SuccessorFn& successors,
                            const CostFn& cost)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>>;

// A* search: Dijkstra guided by an admissible `heuristic` (a lower bound on the remaining
// cost to a goal). Nodes are ordered by f = g + h with ties broken by g then node id, and
// tentative costs are relaxed lazily so an admissible-but-inconsistent heuristic still
// yields the optimum. The returned total cost is the true path cost g and EQUALS the cost
// `dijkstra` reports for the same problem. Same error semantics as `dijkstra`.
[[nodiscard]] auto a_star(std::int64_t start, const GoalFn& goal, const SuccessorFn& successors,
                          const CostFn& cost, const HeuristicFn& heuristic)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>>;

// Breadth-first search that returns nodes grouped by level (level 0 is {start}), stopping
// after at most `max_levels` levels or when the frontier empties. Each frontier is
// expanded IN PARALLEL: `parallel::transform_index` maps every frontier node to its
// successor list concurrently, after which the newly discovered nodes are merged through
// an ordered set (each level sorted ascending). The result is therefore identical to a
// serial level-BFS regardless of thread count. Returns domain_error if max_levels < 0;
// an empty result when max_levels == 0.
[[nodiscard]] auto parallel_bfs_levels(std::int64_t start, SuccessorFn successors,
                                       std::int64_t max_levels)
    -> Result<std::vector<std::vector<std::int64_t>>>;

// ---------------------------------------------------------------------------
// Tabu search (metaheuristic local search).
// ---------------------------------------------------------------------------
using TabuState = std::vector<std::int64_t>;
using NeighborFn = std::function<std::vector<TabuState>(const TabuState&)>;
using ObjectiveFn = std::function<std::int64_t(const TabuState&)>;

// Minimise `objective` over the neighbourhood graph induced by `neighbors`, starting from
// `initial`. Each iteration moves to the best admissible neighbour -- the lowest objective
// value, ties broken by the lexicographically smallest state -- where a neighbour is
// forbidden while it sits in the tabu list (a recency memory of length `tabu_tenure`),
// unless it beats the incumbent best (aspiration). The overall best state and value ever
// seen are returned. Fully deterministic; terminates after at most `max_iters` iterations
// (or earlier if no admissible move exists). domain_error if tabu_tenure or max_iters < 0.
[[nodiscard]] auto tabu_search(TabuState initial, const NeighborFn& neighbors,
                               const ObjectiveFn& objective, std::int64_t tabu_tenure,
                               std::int64_t max_iters)
    -> Result<std::pair<TabuState, std::int64_t>>;


// ---------------------------------------------------------------------------
// A* variants and the best-first family.
//
// Each states its OPTIMALITY plainly, because that is the property a caller reasons with and
// the one most easily assumed: `ida_star` is optimal, `weighted_a_star` is optimal to within
// its stated factor, and the last two are neither optimal nor (for beam) even complete.
// ---------------------------------------------------------------------------

// Iterative-deepening A*: depth-first passes bounded by a rising f = g + h threshold. OPTIMAL
// for an admissible heuristic — the same cost `a_star` returns — but in memory linear in the
// DEPTH rather than in the number of expanded nodes, which is the whole reason to use it.
// Exhausting `max_iterations` threshold-raising passes is not_converged, never a best-so-far.
[[nodiscard]] auto ida_star(std::int64_t start, GoalFn goal, SuccessorFn successors,
                            CostFn cost, HeuristicFn heuristic, std::int64_t max_iterations)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>>;

// Weighted A*: orders by f = g + w*h, with w given as the exact rational
// `weight_numerator / weight_denominator` so no floating point enters the ordering.
// BOUNDED SUBOPTIMAL — the returned path costs at most w times the optimum. w = 1 behaves
// exactly like `a_star`. A denominator of 0, or a ratio below 1, is a domain_error.
[[nodiscard]] auto weighted_a_star(std::int64_t start, const GoalFn& goal,
                                   const SuccessorFn& successors, const CostFn& cost,
                                   HeuristicFn heuristic, std::int64_t weight_numerator,
                                   std::int64_t weight_denominator)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>>;

// Greedy best-first: expands the smallest h and ignores g entirely. NOT OPTIMAL — it finds a
// path quickly and makes no claim about its cost. Exhausting `max_expansions` is not_converged.
[[nodiscard]] auto greedy_best_first(std::int64_t start, const GoalFn& goal,
                                     const SuccessorFn& successors, const HeuristicFn& heuristic,
                                     std::int64_t max_expansions)
    -> Result<std::vector<std::int64_t>>;

// Beam search: a level sweep keeping only the `beam_width` best nodes per level. NOT OPTIMAL
// and INCOMPLETE — it can miss a reachable goal entirely, which is a stronger caveat than
// suboptimality and the reason to reach for it only when the alternative is not finishing.
// A width below 1 is a domain_error.
[[nodiscard]] auto beam_search(std::int64_t start, const GoalFn& goal,
                               const SuccessorFn& successors, const HeuristicFn& heuristic,
                               std::int64_t beam_width, std::int64_t max_levels)
    -> Result<std::vector<std::int64_t>>;

// Bidirectional Dijkstra: forward from `start`, backward from `goal_node`, meeting in the
// middle. Needs an explicit goal NODE and a PREDECESSOR function, because a backward search
// cannot be driven by a goal predicate alone — which is why this is a separate entry point
// rather than an option on `dijkstra`. Returns the same (path, cost) `dijkstra` does.
[[nodiscard]] auto bidirectional_dijkstra(std::int64_t start, std::int64_t goal_node,
                                          const SuccessorFn& successors, const SuccessorFn& predecessors,
                                          const CostFn& cost)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>>;

// ---------------------------------------------------------------------------
// Shared-memory parallel search.
//
// Every one of these returns EXACTLY what its serial counterpart returns — the same path and
// the same cost, on any thread count. The parallel part is a pure map (expand these nodes,
// score these states) and every DECISION stays in deterministic serial code consuming the
// map's index-ordered output. A search whose answer depended on scheduling could not be
// tested against anything, which would cost more than the speed is worth.
//
// The callbacks are invoked CONCURRENTLY and must therefore be pure: a heuristic that
// memoises into a shared cache would race.
// ---------------------------------------------------------------------------

// A* with each expansion wave evaluated concurrently. Identical results to `a_star`.
[[nodiscard]] auto parallel_a_star(std::int64_t start, const GoalFn& goal, SuccessorFn successors,
                                   CostFn cost, HeuristicFn heuristic)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>>;

// Dijkstra with the same wave structure. Identical results to `dijkstra`.
[[nodiscard]] auto parallel_dijkstra(std::int64_t start, const GoalFn& goal, SuccessorFn successors,
                                     CostFn cost)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>>;

// Multi-start tabu search: independent runs from each start, concurrently, best result wins
// (lowest objective, ties by the lexicographically smallest state). Independence is what makes
// this the honest way to parallelise a metaheuristic — a shared incumbent would explore
// differently on every run and could not be compared with anything. Empty starts is a
// domain_error; a single start equals `tabu_search` from it.
[[nodiscard]] auto parallel_multistart_tabu(std::vector<TabuState> starts, NeighborFn neighbors,
                                            ObjectiveFn objective, std::int64_t tabu_tenure,
                                            std::int64_t max_iters)
    -> Result<std::pair<TabuState, std::int64_t>>;

// Scores every neighbour of `state` concurrently, returned in neighbour order — the building
// block for a caller's own parallel local search, without re-deriving the determinism argument.
[[nodiscard]] auto parallel_neighbourhood_scan(const TabuState& state, const NeighborFn& neighbors,
                                               ObjectiveFn objective)
    -> Result<std::vector<std::pair<TabuState, std::int64_t>>>;
// ---------------------------------------------------------------------------
// Dynamic programming (exact integers; every accumulation overflow-guarded).
// Each problem is offered in a tabulated (bottom-up) and a memoised (top-down) form that
// return the identical value; `edit_distance` additionally has a parallel anti-diagonal
// form. Tabulated tables are the dense, rectangular, GPU-offloadable shape referenced in
// the module header.
// ---------------------------------------------------------------------------

// Levenshtein edit distance (insert/delete/substitute, unit cost) between `a` and `b`,
// bottom-up with a rolling two-row table. e.g. edit_distance("kitten","sitting") == 3.
[[nodiscard]] auto edit_distance(std::string_view a, std::string_view b) -> Result<std::int64_t>;

// Recursive memoised form of `edit_distance`; equal to the tabulated value.
[[nodiscard]] auto edit_distance_memo(std::string_view a, std::string_view b)
    -> Result<std::int64_t>;

// Anti-diagonal parallel form of `edit_distance`: the (n+1)x(m+1) table is filled one
// anti-diagonal at a time, and the independent interior cells of each diagonal are
// computed with `parallel::transform_index` (a device kernel would map one launch per
// diagonal). Equals `edit_distance` exactly for every input.
[[nodiscard]] auto edit_distance_parallel(std::string_view a, std::string_view b)
    -> Result<std::int64_t>;

// Length of the longest common subsequence of integer sequences `a` and `b` (bottom-up,
// rolling two-row table).
[[nodiscard]] auto longest_common_subsequence(std::span<const std::int64_t> a,
                                              std::span<const std::int64_t> b)
    -> Result<std::int64_t>;

// Recursive memoised form of `longest_common_subsequence`; equal to the tabulated length.
[[nodiscard]] auto longest_common_subsequence_memo(std::span<const std::int64_t> a,
                                                   std::span<const std::int64_t> b)
    -> Result<std::int64_t>;

// Maximum total value of a subset of items whose weights fit within `capacity`, each item
// used at most once (0/1 knapsack), computed with the standard 1-D capacity table.
// domain_error if the weight/value counts differ, capacity is negative, or a weight is
// negative; overflow if a value sum exceeds std::int64_t.
[[nodiscard]] auto knapsack_01(std::span<const std::int64_t> weights,
                               std::span<const std::int64_t> values, std::int64_t capacity)
    -> Result<std::int64_t>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

namespace {

// Checked signed addition: writes a + b to `out` and returns false, or returns true
// (leaving `out` untouched) when the mathematical sum falls outside std::int64_t. Used
// to keep every DP/path cost accumulation on the railway (MathError::overflow).
[[nodiscard]] auto add_overflow(std::int64_t a, std::int64_t b, std::int64_t& out) noexcept
    -> bool {
    constexpr std::int64_t lo = std::numeric_limits<std::int64_t>::min();
    constexpr std::int64_t hi = std::numeric_limits<std::int64_t>::max();
    if ((b > 0 && a > hi - b) || (b < 0 && a < lo - b)) {
        return true;
    }
    out = a + b;
    return false;
}

// Walk predecessor links from `goal` back to `start` and return the path in forward order
// (start .. goal).
//
// BOUNDED, and that bound is load-bearing rather than defensive dressing. A predecessor chain
// visits each node at most once, so a walk longer than the map has entries is a CYCLE, and a
// cycle here would spin forever. The relaxations that build `pred` only ever tie-break across a
// strictly positive edge precisely so no cycle can form; this bound is what turns a violation of
// that invariant into an honest error instead of a hang.
//
// `std::nullopt` means no chain from `goal` back to `start` -- a missing link or a cycle. Every
// caller maps it to a MathError rather than returning the partial path it walked, because a
// truncated path presented as a path is exactly the plausible-looking wrong answer the honesty
// invariant forbids.
[[nodiscard]] auto reconstruct(const std::unordered_map<std::int64_t, std::int64_t>& pred,
                               std::int64_t start, std::int64_t goal)
    -> std::optional<std::vector<std::int64_t>> {
    std::vector<std::int64_t> path;
    std::int64_t cur = goal;
    path.push_back(cur);
    while (cur != start) {
        if (path.size() > pred.size() + 1) {
            return std::nullopt;
        }
        const auto it = pred.find(cur);
        if (it == pred.end()) {
            return std::nullopt;
        }
        cur = it->second;
        path.push_back(cur);
    }
    std::ranges::reverse(path);
    return path;
}


// S1 — A* variants and best-first family for nimblecas.search
// @author Olumuyiwa Oluwasanmi
//
// Extensions to nimblecas.search: ida_star, weighted_a_star, greedy_best_first,
// beam_search, and bidirectional_dijkstra. Every operation obeys the honesty invariant,
// overflow guarding, and strict determinism.

// Checked multiplication: writes a * b to `out` and returns false, or returns true
// (overflow) if the mathematical product exceeds std::int64_t limits.
[[nodiscard]] auto sx_mul_overflow(std::int64_t a, std::int64_t b, std::int64_t& out) noexcept
    -> bool {
    if (a == 0 || b == 0) {
        out = 0;
        return false;
    }
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_mul_overflow(a, b, &out);
#else
    constexpr std::int64_t hi = std::numeric_limits<std::int64_t>::max();
    constexpr std::int64_t lo = std::numeric_limits<std::int64_t>::min();
    if (a > 0) {
        if (b > 0) {
            if (a > hi / b) {
                return true;
            }
        } else {
            if (b < lo / a) {
                return true;
            }
        }
    } else {
        if (b > 0) {
            if (a < lo / b) {
                return true;
            }
        } else {
            if (a != 0 && b < hi / a) {
                return true;
            }
        }
    }
    out = a * b;
    return false;
#endif
}

// Iterative-deepening A*: depth-first search bounded by an f = g + h threshold.
// OPTIMALITY: Optimal for an admissible heuristic, matching a_star cost. Memory is linear
// in search depth O(d) rather than exponential in the number of generated nodes.
// TIE-BREAK: Successors are expanded in the exact order returned by `successors(u)`. If multiple
// optimal paths exist with the same minimal cost, the first goal discovered in depth-first
// order is returned.
// ERRORS:
// - domain_error if max_iterations < 0 or heuristic returns a negative estimate.
// - domain_error if edge cost is negative.
// - overflow if path cost g or f = g + h exceeds std::int64_t.
// - undefined_value if reachable graph is fully exhausted without finding a goal.
// - not_converged if max_iterations threshold passes are exhausted without finding a goal.


// Parallel A*: identical answers to `a_star`, with each expansion wave evaluated concurrently.
// Successor and heuristic callbacks run concurrently across the wave and MUST be pure; any
// unsynchronised mutable state in callbacks would induce a data race.

}  // namespace

auto bfs(std::int64_t start, const GoalFn& goal, const SuccessorFn& successors)
    -> Result<std::vector<std::int64_t>> {
    if (goal(start)) {
        return std::vector<std::int64_t>{start};
    }

    std::queue<std::int64_t> frontier;
    std::unordered_set<std::int64_t> visited;
    std::unordered_map<std::int64_t, std::int64_t> pred;

    frontier.push(start);
    visited.insert(start);

    while (!frontier.empty()) {
        const std::int64_t u = frontier.front();
        frontier.pop();
        for (const std::int64_t v : successors(u)) {
            if (visited.contains(v)) {
                continue;
            }
            visited.insert(v);
            pred[v] = u;
            if (goal(v)) {
                auto path = reconstruct(pred, start, v);
                if (!path.has_value()) {
                    return make_error<std::vector<std::int64_t>>(MathError::not_converged);
                }
                return std::move(*path);
            }
            frontier.push(v);
        }
    }
    return make_error<std::vector<std::int64_t>>(MathError::undefined_value);
}

auto dfs_recursive(std::int64_t start, GoalFn goal, SuccessorFn successors,
                   std::int64_t max_depth) -> Result<std::vector<std::int64_t>> {
    if (max_depth < 0) {
        return make_error<std::vector<std::int64_t>>(MathError::domain_error);
    }

    std::unordered_set<std::int64_t> visited;
    std::vector<std::int64_t> path;

    // Pre-order DFS: push the node, accept immediately if it is a goal, otherwise mark it
    // and recurse into not-yet-visited successors in order; unwind (pop) on failure.
    const std::function<bool(std::int64_t, std::int64_t)> go =
        [&](std::int64_t node, std::int64_t depth) -> bool {
        path.push_back(node);
        if (goal(node)) {
            return true;
        }
        visited.insert(node);
        if (depth < max_depth) {
            for (const std::int64_t next : successors(node)) {
                if (!visited.contains(next) && go(next, depth + 1)) {
                    return true;
                }
            }
        }
        path.pop_back();
        return false;
    };

    if (go(start, 0)) {
        return path;
    }
    return make_error<std::vector<std::int64_t>>(MathError::undefined_value);
}

auto dfs_iterative(std::int64_t start, const GoalFn& goal, const SuccessorFn& successors,
                   std::int64_t max_depth) -> Result<std::vector<std::int64_t>> {
    if (max_depth < 0) {
        return make_error<std::vector<std::int64_t>>(MathError::domain_error);
    }

    std::unordered_set<std::int64_t> visited;
    // Each stack frame carries the full path to its node so reconstruction is trivial and
    // the deepest path to any node is the one popped first (LIFO), reproducing the
    // recursive pre-order. Successors are pushed reversed so the leftmost pops first, and
    // acceptance/expansion is gated at pop time via the visited set.
    std::vector<std::vector<std::int64_t>> stack;
    stack.push_back({start});

    while (!stack.empty()) {
        std::vector<std::int64_t> path = std::move(stack.back());
        stack.pop_back();
        const std::int64_t node = path.back();
        if (visited.contains(node)) {
            continue;
        }
        if (goal(node)) {
            return path;
        }
        visited.insert(node);
        const std::int64_t depth = static_cast<std::int64_t>(path.size()) - 1;
        if (depth < max_depth) {
            const std::vector<std::int64_t> succ = successors(node);
            for (auto it = succ.rbegin(); it != succ.rend(); ++it) {
                if (!visited.contains(*it)) {
                    std::vector<std::int64_t> next = path;
                    next.push_back(*it);
                    stack.push_back(std::move(next));
                }
            }
        }
    }
    return make_error<std::vector<std::int64_t>>(MathError::undefined_value);
}

auto iterative_deepening_dfs(std::int64_t start, GoalFn goal, SuccessorFn successors,
                             std::int64_t max_depth) -> Result<std::vector<std::int64_t>> {
    if (max_depth < 0) {
        return make_error<std::vector<std::int64_t>>(MathError::domain_error);
    }

    std::vector<std::int64_t> path;
    std::unordered_set<std::int64_t> on_path;

    // Depth-limited search using an ANCESTOR-only visited set (nodes currently on the path)
    // so every node within `limit` edges is reachable regardless of branch ordering; this
    // is what lets the outer loop return the shallowest goal.
    const std::function<bool(std::int64_t, std::int64_t, std::int64_t)> dls =
        [&](std::int64_t node, std::int64_t depth, std::int64_t limit) -> bool {
        path.push_back(node);
        on_path.insert(node);
        if (goal(node)) {
            return true;
        }
        if (depth < limit) {
            for (const std::int64_t next : successors(node)) {
                if (!on_path.contains(next) && dls(next, depth + 1, limit)) {
                    return true;
                }
            }
        }
        path.pop_back();
        on_path.erase(node);
        return false;
    };

    for (std::int64_t limit = 0; limit <= max_depth; ++limit) {
        path.clear();
        on_path.clear();
        if (dls(start, 0, limit)) {
            return path;
        }
    }
    return make_error<std::vector<std::int64_t>>(MathError::undefined_value);
}

auto dijkstra(std::int64_t start, const GoalFn& goal, const SuccessorFn& successors,
              const CostFn& cost)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>> {
    using PathCost = std::pair<std::vector<std::int64_t>, std::int64_t>;

    std::unordered_map<std::int64_t, std::int64_t> dist;
    std::unordered_map<std::int64_t, std::int64_t> pred;
    // Min-heap keyed by (distance, node id): equal distances pop lowest node id first.
    using Entry = std::pair<std::int64_t, std::int64_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;

    dist[start] = 0;
    pq.push({0, start});

    while (!pq.empty()) {
        const auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u]) {
            continue;  // stale queue entry superseded by a shorter relaxation
        }
        if (goal(u)) {
            auto path = reconstruct(pred, start, u);
            if (!path.has_value()) {
                return make_error<PathCost>(MathError::not_converged);
            }
            return PathCost{std::move(*path), d};
        }
        for (const std::int64_t v : successors(u)) {
            const std::int64_t w = cost(u, v);
            if (w < 0) {
                return make_error<PathCost>(MathError::domain_error);
            }
            std::int64_t nd = 0;
            if (add_overflow(d, w, nd)) {
                return make_error<PathCost>(MathError::overflow);
            }
            const auto it = dist.find(v);
            bool better = false;
            if (it == dist.end() || nd < it->second) {
                better = true;
            } else if (nd == it->second) {
                // Deterministic tie-break: keep the predecessor with the lower node id.
                const auto pit = pred.find(v);
                // Only on a STRICTLY positive edge. At equal distance the tie-break could otherwise make a
                // node its own ancestor -- a zero-cost self-loop or cycle lets u and v share a distance --
                // and a predecessor cycle is a path reconstruction that never terminates.
                if (w > 0 && pit != pred.end() && u < pit->second) {
                    better = true;
                }
            }
            if (better) {
                dist[v] = nd;
                pred[v] = u;
                pq.push({nd, v});
            }
        }
    }
    return make_error<PathCost>(MathError::undefined_value);
}

auto a_star(std::int64_t start, const GoalFn& goal, const SuccessorFn& successors,
            const CostFn& cost, const HeuristicFn& heuristic)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>> {
    using PathCost = std::pair<std::vector<std::int64_t>, std::int64_t>;

    std::unordered_map<std::int64_t, std::int64_t> dist;  // best known true cost g
    std::unordered_map<std::int64_t, std::int64_t> pred;
    // Ordered by (f, g, node): f = g + h breaks toward smaller estimate, then smaller g,
    // then lowest node id -- fully deterministic. g is stored in the entry so a stale pop
    // (one whose g exceeds the current best) can be discarded without recomputation.
    using Entry = std::tuple<std::int64_t, std::int64_t, std::int64_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;

    std::int64_t f0 = 0;
    if (add_overflow(0, heuristic(start), f0)) {
        return make_error<PathCost>(MathError::overflow);
    }
    dist[start] = 0;
    pq.push(Entry{f0, 0, start});

    while (!pq.empty()) {
        const Entry top = pq.top();
        pq.pop();
        const std::int64_t g = std::get<1>(top);  // true cost g settled for this entry
        const std::int64_t u = std::get<2>(top);
        if (g > dist[u]) {
            continue;  // stale entry
        }
        if (goal(u)) {
            auto path = reconstruct(pred, start, u);
            if (!path.has_value()) {
                return make_error<PathCost>(MathError::not_converged);
            }
            return PathCost{std::move(*path), g};
        }
        for (const std::int64_t v : successors(u)) {
            const std::int64_t w = cost(u, v);
            if (w < 0) {
                return make_error<PathCost>(MathError::domain_error);
            }
            std::int64_t ng = 0;
            if (add_overflow(g, w, ng)) {
                return make_error<PathCost>(MathError::overflow);
            }
            const auto it = dist.find(v);
            bool better = false;
            if (it == dist.end() || ng < it->second) {
                better = true;
            } else if (ng == it->second) {
                const auto pit = pred.find(v);
                // Only on a STRICTLY positive edge. At equal distance the tie-break could otherwise make a
                // node its own ancestor -- a zero-cost self-loop or cycle lets u and v share a distance --
                // and a predecessor cycle is a path reconstruction that never terminates.
                if (w > 0 && pit != pred.end() && u < pit->second) {
                    better = true;
                }
            }
            if (better) {
                std::int64_t nf = 0;
                if (add_overflow(ng, heuristic(v), nf)) {
                    return make_error<PathCost>(MathError::overflow);
                }
                dist[v] = ng;
                pred[v] = u;
                pq.push(Entry{nf, ng, v});
            }
        }
    }
    return make_error<PathCost>(MathError::undefined_value);
}

auto parallel_bfs_levels(std::int64_t start, SuccessorFn successors, std::int64_t max_levels)
    -> Result<std::vector<std::vector<std::int64_t>>> {
    if (max_levels < 0) {
        return make_error<std::vector<std::vector<std::int64_t>>>(MathError::domain_error);
    }

    std::vector<std::vector<std::int64_t>> levels;
    if (max_levels == 0) {
        return levels;
    }

    std::unordered_set<std::int64_t> visited;
    std::vector<std::int64_t> frontier{start};
    visited.insert(start);
    levels.push_back(frontier);  // level 0

    while (static_cast<std::int64_t>(levels.size()) < max_levels && !frontier.empty()) {
        // Expand the whole frontier in parallel: index i -> successor list of frontier[i].
        // Each index writes an independent slot, so the map is race-free and its result
        // depends only on the (pure) successor function, not on scheduling.
        const std::vector<std::vector<std::int64_t>> expanded =
            parallel::transform_index(frontier.size(), [&](std::size_t i) {
                return successors(frontier[i]);
            });

        // Serial, ordered merge of newly discovered nodes: std::set gives ascending order
        // and de-duplication, so the level is identical to a single-threaded level-BFS.
        std::set<std::int64_t> next;
        for (const auto& list : expanded) {
            for (const std::int64_t v : list) {
                if (!visited.contains(v)) {
                    next.insert(v);
                }
            }
        }
        if (next.empty()) {
            break;
        }

        std::vector<std::int64_t> level(next.begin(), next.end());
        for (const std::int64_t v : level) {
            visited.insert(v);
        }
        levels.push_back(level);
        frontier = std::move(level);
    }
    return levels;
}

auto tabu_search(TabuState initial, const NeighborFn& neighbors, const ObjectiveFn& objective,
                 std::int64_t tabu_tenure, std::int64_t max_iters)
    -> Result<std::pair<TabuState, std::int64_t>> {
    using Best = std::pair<TabuState, std::int64_t>;
    if (tabu_tenure < 0 || max_iters < 0) {
        return make_error<Best>(MathError::domain_error);
    }

    TabuState current = std::move(initial);
    TabuState best = current;
    std::int64_t best_val = objective(current);

    // Recency memory: state -> remaining tenure (a state is forbidden while tenure > 0).
    // std::map keeps it ordered for reproducibility (iteration order is deterministic).
    std::map<TabuState, std::int64_t> tabu;

    for (std::int64_t iter = 0; iter < max_iters; ++iter) {
        const std::vector<TabuState> candidates = neighbors(current);

        bool found = false;
        TabuState chosen;
        std::int64_t chosen_val = 0;
        for (const TabuState& cand : candidates) {
            const std::int64_t v = objective(cand);
            const auto it = tabu.find(cand);
            const bool is_tabu = (it != tabu.end() && it->second > 0);
            const bool aspires = v < best_val;  // aspiration overrides the tabu ban
            if (is_tabu && !aspires) {
                continue;
            }
            // Best admissible: lowest value, ties broken by lexicographically least state.
            if (!found || v < chosen_val || (v == chosen_val && cand < chosen)) {
                found = true;
                chosen = cand;
                chosen_val = v;
            }
        }
        if (!found) {
            break;  // no admissible move remains
        }

        // Age the memory (the move we are about to make gets a fresh, undecremented tenure).
        for (auto& [state, tenure] : tabu) {
            if (tenure > 0) {
                --tenure;
            }
        }
        current = chosen;
        tabu[current] = tabu_tenure;
        if (chosen_val < best_val) {
            best = current;
            best_val = chosen_val;
        }
    }

    return Best{best, best_val};
}

auto edit_distance(std::string_view a, std::string_view b) -> Result<std::int64_t> {
    const std::size_t n = a.size();
    const std::size_t m = b.size();

    // Rolling two-row table: prev = row i-1, cur = row i.
    std::vector<std::int64_t> prev(m + 1);
    std::vector<std::int64_t> cur(m + 1);
    for (std::size_t j = 0; j <= m; ++j) {
        prev[j] = static_cast<std::int64_t>(j);
    }

    for (std::size_t i = 1; i <= n; ++i) {
        cur[0] = static_cast<std::int64_t>(i);
        for (std::size_t j = 1; j <= m; ++j) {
            const std::int64_t sub_cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            std::int64_t del = 0;
            std::int64_t ins = 0;
            std::int64_t sub = 0;
            if (add_overflow(prev[j], 1, del) || add_overflow(cur[j - 1], 1, ins) ||
                add_overflow(prev[j - 1], sub_cost, sub)) {
                return make_error<std::int64_t>(MathError::overflow);
            }
            cur[j] = std::min({del, ins, sub});
        }
        std::swap(prev, cur);
    }
    return prev[m];
}

auto edit_distance_memo(std::string_view a, std::string_view b) -> Result<std::int64_t> {
    const std::size_t n = a.size();
    const std::size_t m = b.size();
    std::vector<std::vector<std::optional<std::int64_t>>> memo(
        n + 1, std::vector<std::optional<std::int64_t>>(m + 1));

    // solve(i, j) = edit distance between the prefixes a[0..i) and b[0..j).
    const std::function<Result<std::int64_t>(std::size_t, std::size_t)> solve =
        [&](std::size_t i, std::size_t j) -> Result<std::int64_t> {
        if (i == 0) {
            return static_cast<std::int64_t>(j);
        }
        if (j == 0) {
            return static_cast<std::int64_t>(i);
        }
        if (memo[i][j].has_value()) {
            return *memo[i][j];
        }
        const std::int64_t sub_cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
        const auto r_del = solve(i - 1, j);
        if (!r_del) {
            return r_del;
        }
        const auto r_ins = solve(i, j - 1);
        if (!r_ins) {
            return r_ins;
        }
        const auto r_sub = solve(i - 1, j - 1);
        if (!r_sub) {
            return r_sub;
        }
        std::int64_t del = 0;
        std::int64_t ins = 0;
        std::int64_t sub = 0;
        if (add_overflow(*r_del, 1, del) || add_overflow(*r_ins, 1, ins) ||
            add_overflow(*r_sub, sub_cost, sub)) {
            return make_error<std::int64_t>(MathError::overflow);
        }
        const std::int64_t best = std::min({del, ins, sub});
        memo[i][j] = best;
        return best;
    };

    return solve(n, m);
}

auto edit_distance_parallel(std::string_view a, std::string_view b) -> Result<std::int64_t> {
    const std::size_t n = a.size();
    const std::size_t m = b.size();
    const std::size_t rows = n + 1;
    const std::size_t cols = m + 1;

    // Guard the flat table dimension against std::size_t overflow before allocating.
    if (cols != 0 && rows > std::numeric_limits<std::size_t>::max() / cols) {
        return make_error<std::int64_t>(MathError::overflow);
    }
    std::vector<std::int64_t> table(rows * cols);
    const auto at = [cols](std::size_t i, std::size_t j) { return i * cols + j; };

    // Boundary rows/columns (the d = 0 and d = 1 anti-diagonals).
    for (std::size_t i = 0; i < rows; ++i) {
        table[at(i, 0)] = static_cast<std::int64_t>(i);
    }
    for (std::size_t j = 0; j < cols; ++j) {
        table[at(0, j)] = static_cast<std::int64_t>(j);
    }

    // Fill interior cells one anti-diagonal d = i + j at a time. Every cell on diagonal d
    // depends only on diagonals d-1 and d-2 (already settled), so the cells of a single
    // diagonal are mutually independent -- exactly the data-parallel wavefront a GPU would
    // launch as one kernel. Cell values are bounded by n + m, so no accumulation can
    // overflow and the numeric result matches the serial table cell-for-cell.
    for (std::size_t d = 2; d <= n + m; ++d) {
        const std::size_t i_lo = (d > m) ? d - m : 1;
        const std::size_t i_hi = std::min(n, d - 1);
        if (i_lo > i_hi) {
            continue;
        }
        std::vector<std::pair<std::size_t, std::size_t>> cells;
        cells.reserve(i_hi - i_lo + 1);
        for (std::size_t i = i_lo; i <= i_hi; ++i) {
            cells.push_back({i, d - i});
        }

        const std::vector<std::int64_t> vals =
            parallel::transform_index(cells.size(), [&](std::size_t k) -> std::int64_t {
                const auto [i, j] = cells[k];
                const std::int64_t sub_cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
                const std::int64_t del = table[at(i - 1, j)] + 1;
                const std::int64_t ins = table[at(i, j - 1)] + 1;
                const std::int64_t sub = table[at(i - 1, j - 1)] + sub_cost;
                return std::min({del, ins, sub});
            });
        for (std::size_t k = 0; k < cells.size(); ++k) {
            const auto [i, j] = cells[k];
            table[at(i, j)] = vals[k];
        }
    }
    return table[at(n, m)];
}

auto longest_common_subsequence(std::span<const std::int64_t> a,
                                std::span<const std::int64_t> b) -> Result<std::int64_t> {
    const std::size_t n = a.size();
    const std::size_t m = b.size();

    std::vector<std::int64_t> prev(m + 1, 0);
    std::vector<std::int64_t> cur(m + 1, 0);

    for (std::size_t i = 1; i <= n; ++i) {
        cur[0] = 0;
        for (std::size_t j = 1; j <= m; ++j) {
            if (a[i - 1] == b[j - 1]) {
                std::int64_t v = 0;
                if (add_overflow(prev[j - 1], 1, v)) {
                    return make_error<std::int64_t>(MathError::overflow);
                }
                cur[j] = v;
            } else {
                cur[j] = std::max(prev[j], cur[j - 1]);
            }
        }
        std::swap(prev, cur);
    }
    return prev[m];
}

auto longest_common_subsequence_memo(std::span<const std::int64_t> a,
                                     std::span<const std::int64_t> b) -> Result<std::int64_t> {
    const std::size_t n = a.size();
    const std::size_t m = b.size();
    std::vector<std::vector<std::optional<std::int64_t>>> memo(
        n + 1, std::vector<std::optional<std::int64_t>>(m + 1));

    // solve(i, j) = LCS length of prefixes a[0..i) and b[0..j).
    const std::function<Result<std::int64_t>(std::size_t, std::size_t)> solve =
        [&](std::size_t i, std::size_t j) -> Result<std::int64_t> {
        if (i == 0 || j == 0) {
            return std::int64_t{0};
        }
        if (memo[i][j].has_value()) {
            return *memo[i][j];
        }
        std::int64_t result = 0;
        if (a[i - 1] == b[j - 1]) {
            const auto r = solve(i - 1, j - 1);
            if (!r) {
                return r;
            }
            if (add_overflow(*r, 1, result)) {
                return make_error<std::int64_t>(MathError::overflow);
            }
        } else {
            const auto r_drop_a = solve(i - 1, j);
            if (!r_drop_a) {
                return r_drop_a;
            }
            const auto r_drop_b = solve(i, j - 1);
            if (!r_drop_b) {
                return r_drop_b;
            }
            result = std::max(*r_drop_a, *r_drop_b);
        }
        memo[i][j] = result;
        return result;
    };

    return solve(n, m);
}

auto knapsack_01(std::span<const std::int64_t> weights, std::span<const std::int64_t> values,
                 std::int64_t capacity) -> Result<std::int64_t> {
    if (weights.size() != values.size() || capacity < 0) {
        return make_error<std::int64_t>(MathError::domain_error);
    }
    for (const std::int64_t w : weights) {
        if (w < 0) {
            return make_error<std::int64_t>(MathError::domain_error);
        }
    }

    const std::size_t cap = static_cast<std::size_t>(capacity);
    // dp[c] = best value achievable with a capacity budget of exactly c, considering the
    // items processed so far. Iterating capacity downward keeps each item single-use.
    std::vector<std::int64_t> dp(cap + 1, 0);

    for (std::size_t i = 0; i < weights.size(); ++i) {
        const std::int64_t wi = weights[i];
        const std::int64_t vi = values[i];
        if (wi > capacity) {
            continue;  // item cannot fit under any budget
        }
        for (std::int64_t c = capacity; c >= wi; --c) {
            const std::size_t below = static_cast<std::size_t>(c - wi);
            std::int64_t cand = 0;
            if (add_overflow(dp[below], vi, cand)) {
                return make_error<std::int64_t>(MathError::overflow);
            }
            if (cand > dp[static_cast<std::size_t>(c)]) {
                dp[static_cast<std::size_t>(c)] = cand;
            }
        }
    }
    return dp[cap];
}

// ---------------------------------------------------------------------------
// A* variants and the best-first family.
// ---------------------------------------------------------------------------

[[nodiscard]] auto ida_star(std::int64_t start, GoalFn goal, SuccessorFn successors,
                            CostFn cost, HeuristicFn heuristic, std::int64_t max_iterations)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>> {
    using PathCost = std::pair<std::vector<std::int64_t>, std::int64_t>;

    if (max_iterations < 0) {
        return make_error<PathCost>(MathError::domain_error);
    }
    if (goal(start)) {
        return PathCost{std::vector<std::int64_t>{start}, 0};
    }
    if (max_iterations == 0) {
        return make_error<PathCost>(MathError::not_converged);
    }

    const std::int64_t h0 = heuristic(start);
    if (h0 < 0) {
        return make_error<PathCost>(MathError::domain_error);
    }

    std::int64_t threshold = h0;
    std::vector<std::int64_t> path;
    std::unordered_set<std::int64_t> on_path;
    std::int64_t min_exceeded = std::numeric_limits<std::int64_t>::max();
    std::int64_t final_cost = 0;
    std::optional<MathError> error_occurred;

    const std::function<bool(std::int64_t, std::int64_t)> dfs =
        [&](std::int64_t u, std::int64_t g) -> bool {
        const std::int64_t hu = heuristic(u);
        if (hu < 0) {
            error_occurred = MathError::domain_error;
            return false;
        }
        std::int64_t f = 0;
        if (add_overflow(g, hu, f)) {
            error_occurred = MathError::overflow;
            return false;
        }
        if (f > threshold) {
            if (f < min_exceeded) {
                min_exceeded = f;
            }
            return false;
        }
        if (goal(u)) {
            final_cost = g;
            return true;
        }
        for (const std::int64_t v : successors(u)) {
            if (on_path.contains(v)) {
                continue;
            }
            const std::int64_t w = cost(u, v);
            if (w < 0) {
                error_occurred = MathError::domain_error;
                return false;
            }
            std::int64_t ng = 0;
            if (add_overflow(g, w, ng)) {
                error_occurred = MathError::overflow;
                return false;
            }
            path.push_back(v);
            on_path.insert(v);
            if (dfs(v, ng)) {
                return true;
            }
            if (error_occurred.has_value()) {
                return false;
            }
            path.pop_back();
            on_path.erase(v);
        }
        return false;
    };

    for (std::int64_t iter = 0; iter < max_iterations; ++iter) {
        min_exceeded = std::numeric_limits<std::int64_t>::max();
        error_occurred.reset();
        path.clear();
        path.push_back(start);
        on_path.clear();
        on_path.insert(start);

        if (dfs(start, 0)) {
            return PathCost{std::move(path), final_cost};
        }
        if (error_occurred.has_value()) {
            return make_error<PathCost>(*error_occurred);
        }
        if (min_exceeded == std::numeric_limits<std::int64_t>::max()) {
            return make_error<PathCost>(MathError::undefined_value);
        }
        threshold = min_exceeded;
    }

    return make_error<PathCost>(MathError::not_converged);
}

// Weighted A*: best-first search guided by f = g + w*h where w >= 1 is an exact rational.
// OPTIMALITY / BOUND: BOUNDED SUBOPTIMAL — the returned path costs at most w = (weight_numerator /
// weight_denominator) times the optimum. When w == 1, it behaves identically to a_star and is
// strictly optimal. A caller assuming optimality for w > 1 will be wrong.
// TIE-BREAK: Ordered by scaled f = weight_denominator*g + weight_numerator*h. Ties are broken
// by true cost g, then by lowest node id. Edge relaxation breaks ties toward the predecessor
// with the lower node id.
// ERRORS:
// - domain_error if weight_denominator == 0 or w < 1 (weight_numerator < weight_denominator).
// - domain_error if any edge cost is negative or heuristic is negative.
// - overflow if integer multiplication or cost addition exceeds std::int64_t.
// - undefined_value if no goal is reachable.
[[nodiscard]] auto weighted_a_star(std::int64_t start, const GoalFn& goal,
                                   const SuccessorFn& successors, const CostFn& cost,
                                   HeuristicFn heuristic, std::int64_t weight_numerator,
                                   std::int64_t weight_denominator)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>> {
    using PathCost = std::pair<std::vector<std::int64_t>, std::int64_t>;

    if (weight_denominator == 0) {
        return make_error<PathCost>(MathError::domain_error);
    }
    if (weight_denominator < 0) {
        if (weight_numerator == std::numeric_limits<std::int64_t>::min() ||
            weight_denominator == std::numeric_limits<std::int64_t>::min()) {
            return make_error<PathCost>(MathError::overflow);
        }
        weight_numerator = -weight_numerator;
        weight_denominator = -weight_denominator;
    }
    if (weight_numerator < weight_denominator) {
        return make_error<PathCost>(MathError::domain_error);
    }

    const std::int64_t g_cd = std::gcd(weight_numerator, weight_denominator);
    if (g_cd > 0) {
        weight_numerator /= g_cd;
        weight_denominator /= g_cd;
    }

    const std::int64_t h0 = heuristic(start);
    if (h0 < 0) {
        return make_error<PathCost>(MathError::domain_error);
    }

    const auto compute_f_scaled =
        [&](std::int64_t g, std::int64_t node, std::int64_t& out_f) -> Result<void> {
        const std::int64_t h = heuristic(node);
        if (h < 0) {
            return make_error<void>(MathError::domain_error);
        }
        if (weight_numerator == 1 && weight_denominator == 1) {
            if (add_overflow(g, h, out_f)) {
                return make_error<void>(MathError::overflow);
            }
            return {};
        }
        std::int64_t term_g = 0;
        if (sx_mul_overflow(weight_denominator, g, term_g)) {
            return make_error<void>(MathError::overflow);
        }
        std::int64_t term_h = 0;
        if (sx_mul_overflow(weight_numerator, h, term_h)) {
            return make_error<void>(MathError::overflow);
        }
        if (add_overflow(term_g, term_h, out_f)) {
            return make_error<void>(MathError::overflow);
        }
        return {};
    };

    std::unordered_map<std::int64_t, std::int64_t> dist;
    std::unordered_map<std::int64_t, std::int64_t> pred;

    // Ordered by (f_scaled, g, node_id) with min-heap
    using Entry = std::tuple<std::int64_t, std::int64_t, std::int64_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;

    std::int64_t f0 = 0;
    const auto r0 = compute_f_scaled(0, start, f0);
    if (!r0) {
        return make_error<PathCost>(r0.error());
    }

    dist[start] = 0;
    pq.push(Entry{f0, 0, start});

    while (!pq.empty()) {
        const Entry top = pq.top();
        pq.pop();
        const std::int64_t g = std::get<1>(top);
        const std::int64_t u = std::get<2>(top);
        if (g > dist[u]) {
            continue;  // Stale entry
        }
        if (goal(u)) {
            auto path = reconstruct(pred, start, u);
            if (!path.has_value()) {
                return make_error<PathCost>(MathError::not_converged);
            }
            return PathCost{std::move(*path), g};
        }
        for (const std::int64_t v : successors(u)) {
            const std::int64_t w = cost(u, v);
            if (w < 0) {
                return make_error<PathCost>(MathError::domain_error);
            }
            std::int64_t ng = 0;
            if (add_overflow(g, w, ng)) {
                return make_error<PathCost>(MathError::overflow);
            }
            const auto it = dist.find(v);
            bool better = false;
            if (it == dist.end() || ng < it->second) {
                better = true;
            } else if (ng == it->second) {
                const auto pit = pred.find(v);
                // Only on a STRICTLY positive edge. At equal distance the tie-break could otherwise make a
                // node its own ancestor -- a zero-cost self-loop or cycle lets u and v share a distance --
                // and a predecessor cycle is a path reconstruction that never terminates.
                if (w > 0 && pit != pred.end() && u < pit->second) {
                    better = true;
                }
            }
            if (better) {
                std::int64_t nf_scaled = 0;
                const auto rf = compute_f_scaled(ng, v, nf_scaled);
                if (!rf) {
                    return make_error<PathCost>(rf.error());
                }
                dist[v] = ng;
                pred[v] = u;
                pq.push(Entry{nf_scaled, ng, v});
            }
        }
    }
    return make_error<PathCost>(MathError::undefined_value);
}

// Greedy best-first search: prioritises expansion strictly by heuristic estimate h.
// OPTIMALITY: FAST and NOT OPTIMAL. Ignores the true accumulated path cost g entirely,
// expanding whichever reachable node appears closest to a goal. It may return a heavily
// suboptimal path.
// TIE-BREAK: Minimum h wins. Ties are broken by lowest node id. Predecessor ties are broken
// by keeping the predecessor with the lower node id.
// ERRORS:
// - domain_error if max_expansions < 0 or heuristic returns a negative value.
// - not_converged if max_expansions node expansions are performed without finding a goal.
// - undefined_value if the reachable frontier is exhausted without finding a goal.
[[nodiscard]] auto greedy_best_first(std::int64_t start, const GoalFn& goal,
                                     const SuccessorFn& successors, const HeuristicFn& heuristic,
                                     std::int64_t max_expansions)
    -> Result<std::vector<std::int64_t>> {
    using Path = std::vector<std::int64_t>;

    if (max_expansions < 0) {
        return make_error<Path>(MathError::domain_error);
    }
    if (goal(start)) {
        return Path{start};
    }
    if (max_expansions == 0) {
        return make_error<Path>(MathError::not_converged);
    }

    const std::int64_t h0 = heuristic(start);
    if (h0 < 0) {
        return make_error<Path>(MathError::domain_error);
    }

    // Min-heap ordered by (h, node_id)
    using Entry = std::pair<std::int64_t, std::int64_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;

    std::unordered_set<std::int64_t> visited;
    std::unordered_map<std::int64_t, std::int64_t> pred;

    pq.push(Entry{h0, start});
    std::int64_t expansions = 0;

    while (!pq.empty()) {
        const auto [h, u] = pq.top();
        pq.pop();
        if (visited.contains(u)) {
            continue;
        }
        if (goal(u)) {
            auto path = reconstruct(pred, start, u);
            if (!path.has_value()) {
                return make_error<Path>(MathError::not_converged);
            }
            return std::move(*path);
        }
        if (expansions >= max_expansions) {
            return make_error<Path>(MathError::not_converged);
        }
        ++expansions;
        visited.insert(u);

        for (const std::int64_t v : successors(u)) {
            if (visited.contains(v)) {
                continue;
            }
            const std::int64_t hv = heuristic(v);
            if (hv < 0) {
                return make_error<Path>(MathError::domain_error);
            }
            const auto pit = pred.find(v);
            if (pit == pred.end()) {
                pred[v] = u;
                pq.push(Entry{hv, v});
            } else if (u < pit->second) {
                pred[v] = u;
            }
        }
    }

    return make_error<Path>(MathError::undefined_value);
}

// Beam search: breadth-first search maintaining at most `beam_width` candidate nodes at each level.
// OPTIMALITY / COMPLETENESS: NOT OPTIMAL and INCOMPLETE. By pruning nodes with larger heuristic
// estimates at each depth level, it may discard the only path to a reachable goal.
// TIE-BREAK: Candidates at each level are ranked by (heuristic h, node_id) ascending.
// The best `beam_width` nodes are retained. When multiple parents generate the same child,
// the parent with the lower node id is retained. If multiple retained nodes satisfy the goal,
// the one with the lowest (h, node_id) rank is chosen.
// ERRORS:
// - domain_error if beam_width < 1 or max_levels < 0 or heuristic returns a negative value.
// - not_converged if max_levels levels are explored without finding a goal.
// - undefined_value if the beam becomes empty with no goal found.
[[nodiscard]] auto beam_search(std::int64_t start, const GoalFn& goal,
                               const SuccessorFn& successors, const HeuristicFn& heuristic,
                               std::int64_t beam_width, std::int64_t max_levels)
    -> Result<std::vector<std::int64_t>> {
    using Path = std::vector<std::int64_t>;

    if (beam_width < 1 || max_levels < 0) {
        return make_error<Path>(MathError::domain_error);
    }
    if (goal(start)) {
        return Path{start};
    }
    if (max_levels == 0) {
        return make_error<Path>(MathError::not_converged);
    }

    const std::int64_t h0 = heuristic(start);
    if (h0 < 0) {
        return make_error<Path>(MathError::domain_error);
    }

    std::vector<std::int64_t> current_beam{start};
    std::unordered_set<std::int64_t> visited{start};
    std::unordered_map<std::int64_t, std::int64_t> pred;

    for (std::int64_t level = 0; level < max_levels; ++level) {
        if (current_beam.empty()) {
            return make_error<Path>(MathError::undefined_value);
        }

        // Map candidate successor node -> lowest parent id
        std::map<std::int64_t, std::int64_t> cand_parents;
        for (const std::int64_t u : current_beam) {
            for (const std::int64_t v : successors(u)) {
                if (visited.contains(v)) {
                    continue;
                }
                // No parent yet, or a lower-numbered one: either way this is the parent.
                // Ties break toward the smaller id, which is what makes the beam
                // deterministic regardless of the order successors are produced in.
                const auto it = cand_parents.find(v);
                if (it == cand_parents.end() || u < it->second) {
                    cand_parents[v] = u;
                }
            }
        }

        if (cand_parents.empty()) {
            return make_error<Path>(MathError::undefined_value);
        }

        struct Candidate {
            std::int64_t h;
            std::int64_t node;
            std::int64_t parent;
            auto operator<=>(const Candidate& other) const = default;
        };
        std::vector<Candidate> candidates;
        candidates.reserve(cand_parents.size());

        for (const auto& [v, parent] : cand_parents) {
            const std::int64_t hv = heuristic(v);
            if (hv < 0) {
                return make_error<Path>(MathError::domain_error);
            }
            candidates.push_back(Candidate{hv, v, parent});
        }

        std::sort(candidates.begin(), candidates.end());

        const std::size_t keep_count =
            std::min(candidates.size(), static_cast<std::size_t>(beam_width));

        // Test goal acceptance on retained beam nodes in priority order
        for (std::size_t i = 0; i < keep_count; ++i) {
            const auto& cand = candidates[i];
            if (goal(cand.node)) {
                pred[cand.node] = cand.parent;
                auto path = reconstruct(pred, start, cand.node);
                if (!path.has_value()) {
                    return make_error<Path>(MathError::not_converged);
                }
                return std::move(*path);
            }
        }

        std::vector<std::int64_t> next_beam;
        next_beam.reserve(keep_count);
        for (std::size_t i = 0; i < keep_count; ++i) {
            const auto& cand = candidates[i];
            pred[cand.node] = cand.parent;
            visited.insert(cand.node);
            next_beam.push_back(cand.node);
        }
        current_beam = std::move(next_beam);
    }

    return make_error<Path>(MathError::not_converged);
}

// Bidirectional Dijkstra: simultaneous forward search from `start` and backward search
// from `goal_node` with exact meeting condition.
// OPTIMALITY: Strictly OPTIMAL for non-negative edge weights, matching dijkstra cost exactly.
// STOPPING RULE: Stops when the sum of the two frontier minima is at least the best meeting
// cost found so far (min_f + min_b >= mu). Does NOT terminate on first meeting, guaranteeing
// optimal cost.
// TIE-BREAK: Each frontier prioritises smallest tentative distance, broken by lower node id.
// Frontier expansion alternates by expanding the side with the smaller frontier minimum
// (min_f <= min_b breaks toward forward). Relaxation ties keep the predecessor/successor
// with the lower node id. Equal meeting costs tie-break to the lower meeting node id.
// ERRORS:
// - domain_error if any forward or backward edge cost is negative.
// - overflow if path cost addition exceeds std::int64_t.
// - undefined_value if no path connects start to goal_node.
[[nodiscard]] auto bidirectional_dijkstra(std::int64_t start, std::int64_t goal_node,
                                          const SuccessorFn& successors, const SuccessorFn& predecessors,
                                          const CostFn& cost)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>> {
    using PathCost = std::pair<std::vector<std::int64_t>, std::int64_t>;

    if (start == goal_node) {
        return PathCost{std::vector<std::int64_t>{start}, 0};
    }

    using Entry = std::pair<std::int64_t, std::int64_t>;  // (distance, node_id)
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq_f;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq_b;

    std::unordered_map<std::int64_t, std::int64_t> dist_f;
    std::unordered_map<std::int64_t, std::int64_t> dist_b;

    std::unordered_map<std::int64_t, std::int64_t> pred_f;
    std::unordered_map<std::int64_t, std::int64_t> succ_b;

    dist_f[start] = 0;
    pq_f.push(Entry{0, start});

    dist_b[goal_node] = 0;
    pq_b.push(Entry{0, goal_node});

    std::int64_t mu = std::numeric_limits<std::int64_t>::max();
    std::int64_t best_meeting_node = -1;

    while (!pq_f.empty() || !pq_b.empty()) {
        while (!pq_f.empty() && pq_f.top().first > dist_f[pq_f.top().second]) {
            pq_f.pop();
        }
        while (!pq_b.empty() && pq_b.top().first > dist_b[pq_b.top().second]) {
            pq_b.pop();
        }

        if (pq_f.empty() || pq_b.empty()) {
            break;
        }

        const std::int64_t min_f = pq_f.top().first;
        const std::int64_t min_b = pq_b.top().first;

        std::int64_t sum_min = 0;
        const bool sum_overflow = add_overflow(min_f, min_b, sum_min);
        if (sum_overflow) {
            if (mu != std::numeric_limits<std::int64_t>::max()) {
                break;
            }
        } else if (sum_min >= mu) {
            break;
        }

        if (min_f <= min_b) {
            const auto [d, u] = pq_f.top();
            pq_f.pop();

            for (const std::int64_t v : successors(u)) {
                const std::int64_t w = cost(u, v);
                if (w < 0) {
                    return make_error<PathCost>(MathError::domain_error);
                }
                std::int64_t nd = 0;
                if (add_overflow(d, w, nd)) {
                    return make_error<PathCost>(MathError::overflow);
                }
                const auto it = dist_f.find(v);
                bool better = false;
                if (it == dist_f.end() || nd < it->second) {
                    better = true;
                } else if (nd == it->second) {
                    const auto pit = pred_f.find(v);
                    // Only on a STRICTLY positive edge. At equal distance the tie-break could otherwise make a
                    // node its own ancestor -- a zero-cost self-loop or cycle lets u and v share a distance --
                    // and a predecessor cycle is a path reconstruction that never terminates.
                    if (w > 0 && pit != pred_f.end() && u < pit->second) {
                        better = true;
                    }
                }
                if (better) {
                    dist_f[v] = nd;
                    pred_f[v] = u;
                    pq_f.push(Entry{nd, v});

                    const auto bit = dist_b.find(v);
                    if (bit != dist_b.end()) {
                        std::int64_t cand = 0;
                        if (!add_overflow(nd, bit->second, cand)) {
                            if (cand < mu) {
                                mu = cand;
                                best_meeting_node = v;
                            } else if (cand == mu &&
                                       (best_meeting_node == -1 || v < best_meeting_node)) {
                                best_meeting_node = v;
                            }
                        }
                    }
                }
            }
        } else {
            const auto [d, v] = pq_b.top();
            pq_b.pop();

            for (const std::int64_t p : predecessors(v)) {
                // Directed edge is p -> v, so weight is cost(p, v)
                const std::int64_t w = cost(p, v);
                if (w < 0) {
                    return make_error<PathCost>(MathError::domain_error);
                }
                std::int64_t nd = 0;
                if (add_overflow(d, w, nd)) {
                    return make_error<PathCost>(MathError::overflow);
                }
                const auto it = dist_b.find(p);
                bool better = false;
                if (it == dist_b.end() || nd < it->second) {
                    better = true;
                } else if (nd == it->second) {
                    const auto sit = succ_b.find(p);
                    // Only on a STRICTLY positive edge. At equal distance the tie-break could otherwise make a
                    // node its own ancestor -- a zero-cost self-loop or cycle lets u and v share a distance --
                    // and a predecessor cycle is a path reconstruction that never terminates.
                    if (w > 0 && sit != succ_b.end() && v < sit->second) {
                        better = true;
                    }
                }
                if (better) {
                    dist_b[p] = nd;
                    succ_b[p] = v;
                    pq_b.push(Entry{nd, p});

                    const auto fit = dist_f.find(p);
                    if (fit != dist_f.end()) {
                        std::int64_t cand = 0;
                        if (!add_overflow(fit->second, nd, cand)) {
                            if (cand < mu) {
                                mu = cand;
                                best_meeting_node = p;
                            } else if (cand == mu &&
                                       (best_meeting_node == -1 || p < best_meeting_node)) {
                                best_meeting_node = p;
                            }
                        }
                    }
                }
            }
        }
    }

    if (mu == std::numeric_limits<std::int64_t>::max() || best_meeting_node == -1) {
        return make_error<PathCost>(MathError::undefined_value);
    }

    // Reconstruct path: forward from start to best_meeting_node.
    //
    // Both walks below are BOUNDED by the size of the link map for the same reason `reconstruct`
    // is: a link cycle would spin forever, and an honest error beats a hang.
    std::vector<std::int64_t> path;
    std::int64_t cur = best_meeting_node;
    path.push_back(cur);
    while (cur != start) {
        if (path.size() > pred_f.size() + 1) {
            return make_error<PathCost>(MathError::not_converged);
        }
        const auto it = pred_f.find(cur);
        if (it == pred_f.end()) {
            return make_error<PathCost>(MathError::undefined_value);
        }
        cur = it->second;
        path.push_back(cur);
    }
    std::reverse(path.begin(), path.end());

    // Append backward path: forward from best_meeting_node to goal_node
    cur = best_meeting_node;
    const std::size_t forward_len = path.size();
    while (cur != goal_node) {
        if (path.size() > forward_len + succ_b.size() + 1) {
            return make_error<PathCost>(MathError::not_converged);
        }
        const auto it = succ_b.find(cur);
        if (it == succ_b.end()) {
            return make_error<PathCost>(MathError::undefined_value);
        }
        cur = it->second;
        path.push_back(cur);
    }

    return PathCost{std::move(path), mu};
}

// ---------------------------------------------------------------------------
// Shared-memory parallel search.
// ---------------------------------------------------------------------------

[[nodiscard]] auto parallel_a_star(std::int64_t start, const GoalFn& goal, SuccessorFn successors,
                                   CostFn cost, HeuristicFn heuristic)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>> {
    using PathCost = std::pair<std::vector<std::int64_t>, std::int64_t>;

    std::unordered_map<std::int64_t, std::int64_t> dist;
    std::unordered_map<std::int64_t, std::int64_t> pred;
    // Min-heap ordered by (f, g, node) with ties broken by g then lowest node id.
    using Entry = std::tuple<std::int64_t, std::int64_t, std::int64_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;

    std::int64_t f0 = 0;
    if (add_overflow(0, heuristic(start), f0)) {
        return make_error<PathCost>(MathError::overflow);
    }
    dist[start] = 0;
    pq.push(Entry{f0, 0, start});

    while (!pq.empty()) {
        // Discard stale entries superseded by earlier shorter relaxations.
        while (!pq.empty()) {
            const Entry& top = pq.top();
            const std::int64_t g = std::get<1>(top);
            const std::int64_t u = std::get<2>(top);
            const auto it = dist.find(u);
            if (it != dist.end() && g > it->second) {
                pq.pop();
                continue;
            }
            break;
        }
        if (pq.empty()) {
            break;
        }

        // The top of the priority queue is the definitive lowest (f, g, u).
        const Entry top = pq.top();
        const std::int64_t min_f = std::get<0>(top);
        const std::int64_t g_top = std::get<1>(top);
        const std::int64_t u_top = std::get<2>(top);
        if (goal(u_top)) {
            auto path = reconstruct(pred, start, u_top);
            if (!path.has_value()) {
                return make_error<PathCost>(MathError::not_converged);
            }
            return PathCost{std::move(*path), g_top};
        }

        // Form an expansion wave of all open nodes sharing the current minimum f value.
        struct WaveNode {
            std::int64_t u;
            std::int64_t g;
        };
        std::vector<WaveNode> wave;
        std::unordered_set<std::int64_t> in_wave;

        while (!pq.empty()) {
            const Entry& cur = pq.top();
            if (std::get<0>(cur) != min_f) {
                break;
            }
            const std::int64_t g = std::get<1>(cur);
            const std::int64_t u = std::get<2>(cur);
            pq.pop();

            const auto it = dist.find(u);
            if (it != dist.end() && g > it->second) {
                continue;
            }
            // A goal node must never be expanded; leave it in pq to be settled at queue top.
            if (goal(u)) {
                pq.push(Entry{min_f, g, u});
                break;
            }
            if (!in_wave.contains(u)) {
                in_wave.insert(u);
                wave.push_back(WaveNode{u, g});
            }
        }

        struct SuccRecord {
            std::int64_t v;
            std::int64_t cost;
            std::int64_t h;
        };
        std::vector<std::vector<SuccRecord>> expansions;
        const bool run_parallel = wave.size() >= parallel::parallel_cost_threshold;

        if (run_parallel) {
            // Successors and heuristic evaluations run concurrently across the wave.
            expansions = parallel::transform_index(wave.size(), [&](std::size_t i) -> std::vector<SuccRecord> {
                const std::int64_t u = wave[i].u;
                const std::vector<std::int64_t> succs = successors(u);
                std::vector<SuccRecord> recs;
                recs.reserve(succs.size());
                for (const std::int64_t v : succs) {
                    const std::int64_t w = cost(u, v);
                    const std::int64_t h = heuristic(v);
                    recs.push_back(SuccRecord{v, w, h});
                }
                return recs;
            });
        } else {
            // Below the cost threshold, threading coordination overhead exceeds the evaluation work.
            expansions.reserve(wave.size());
            for (std::size_t i = 0; i < wave.size(); ++i) {
                const std::int64_t u = wave[i].u;
                const std::vector<std::int64_t> succs = successors(u);
                std::vector<SuccRecord> recs;
                recs.reserve(succs.size());
                for (const std::int64_t v : succs) {
                    const std::int64_t w = cost(u, v);
                    const std::int64_t h = heuristic(v);
                    recs.push_back(SuccRecord{v, w, h});
                }
                expansions.push_back(std::move(recs));
            }
        }

        // Serial relaxation in deterministic wave index order ensures exact identity with serial A*.
        for (std::size_t i = 0; i < wave.size(); ++i) {
            const std::int64_t u = wave[i].u;
            const std::int64_t g = wave[i].g;
            for (const auto& rec : expansions[i]) {
                const std::int64_t v = rec.v;
                const std::int64_t w = rec.cost;
                if (w < 0) {
                    return make_error<PathCost>(MathError::domain_error);
                }
                std::int64_t ng = 0;
                if (add_overflow(g, w, ng)) {
                    return make_error<PathCost>(MathError::overflow);
                }
                const auto it = dist.find(v);
                bool better = false;
                if (it == dist.end() || ng < it->second) {
                    better = true;
                } else if (ng == it->second) {
                    const auto pit = pred.find(v);
                    // Only on a STRICTLY positive edge. At equal distance the tie-break could otherwise make a
                    // node its own ancestor -- a zero-cost self-loop or cycle lets u and v share a distance --
                    // and a predecessor cycle is a path reconstruction that never terminates.
                    if (w > 0 && pit != pred.end() && u < pit->second) {
                        better = true;
                    }
                }
                if (better) {
                    std::int64_t nf = 0;
                    if (add_overflow(ng, rec.h, nf)) {
                        return make_error<PathCost>(MathError::overflow);
                    }
                    dist[v] = ng;
                    pred[v] = u;
                    pq.push(Entry{nf, ng, v});
                }
            }
        }
    }
    return make_error<PathCost>(MathError::undefined_value);
}

// Parallel Dijkstra: wave expansion with zero heuristic, matching serial dijkstra exactly.
// Successor callback is executed concurrently across each wave and MUST be pure.
[[nodiscard]] auto parallel_dijkstra(std::int64_t start, const GoalFn& goal, SuccessorFn successors,
                                     CostFn cost)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>> {
    using PathCost = std::pair<std::vector<std::int64_t>, std::int64_t>;

    std::unordered_map<std::int64_t, std::int64_t> dist;
    std::unordered_map<std::int64_t, std::int64_t> pred;
    using Entry = std::pair<std::int64_t, std::int64_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;

    dist[start] = 0;
    pq.push(Entry{0, start});

    while (!pq.empty()) {
        while (!pq.empty()) {
            const Entry& top = pq.top();
            const std::int64_t d = top.first;
            const std::int64_t u = top.second;
            const auto it = dist.find(u);
            if (it != dist.end() && d > it->second) {
                pq.pop();
                continue;
            }
            break;
        }
        if (pq.empty()) {
            break;
        }

        const Entry top = pq.top();
        const std::int64_t min_d = top.first;
        const std::int64_t u_top = top.second;
        if (goal(u_top)) {
            auto path = reconstruct(pred, start, u_top);
            if (!path.has_value()) {
                return make_error<PathCost>(MathError::not_converged);
            }
            return PathCost{std::move(*path), min_d};
        }

        struct WaveNode {
            std::int64_t u;
            std::int64_t d;
        };
        std::vector<WaveNode> wave;
        std::unordered_set<std::int64_t> in_wave;

        while (!pq.empty()) {
            const Entry& cur = pq.top();
            if (cur.first != min_d) {
                break;
            }
            const std::int64_t d = cur.first;
            const std::int64_t u = cur.second;
            pq.pop();

            const auto it = dist.find(u);
            if (it != dist.end() && d > it->second) {
                continue;
            }
            if (goal(u)) {
                pq.push(Entry{min_d, u});
                break;
            }
            if (!in_wave.contains(u)) {
                in_wave.insert(u);
                wave.push_back(WaveNode{u, d});
            }
        }

        struct SuccRecord {
            std::int64_t v;
            std::int64_t cost;
        };
        std::vector<std::vector<SuccRecord>> expansions;
        const bool run_parallel = wave.size() >= parallel::parallel_cost_threshold;

        if (run_parallel) {
            // Concurrently query successors and evaluate edge weights across the wave.
            expansions = parallel::transform_index(wave.size(), [&](std::size_t i) -> std::vector<SuccRecord> {
                const std::int64_t u = wave[i].u;
                const std::vector<std::int64_t> succs = successors(u);
                std::vector<SuccRecord> recs;
                recs.reserve(succs.size());
                for (const std::int64_t v : succs) {
                    const std::int64_t w = cost(u, v);
                    recs.push_back(SuccRecord{v, w});
                }
                return recs;
            });
        } else {
            expansions.reserve(wave.size());
            for (std::size_t i = 0; i < wave.size(); ++i) {
                const std::int64_t u = wave[i].u;
                const std::vector<std::int64_t> succs = successors(u);
                std::vector<SuccRecord> recs;
                recs.reserve(succs.size());
                for (const std::int64_t v : succs) {
                    const std::int64_t w = cost(u, v);
                    recs.push_back(SuccRecord{v, w});
                }
                expansions.push_back(std::move(recs));
            }
        }

        // Serial relaxation preserves exact deterministic predecessor choice and queue insertion order.
        for (std::size_t i = 0; i < wave.size(); ++i) {
            const std::int64_t u = wave[i].u;
            const std::int64_t d = wave[i].d;
            for (const auto& rec : expansions[i]) {
                const std::int64_t v = rec.v;
                const std::int64_t w = rec.cost;
                if (w < 0) {
                    return make_error<PathCost>(MathError::domain_error);
                }
                std::int64_t nd = 0;
                if (add_overflow(d, w, nd)) {
                    return make_error<PathCost>(MathError::overflow);
                }
                const auto it = dist.find(v);
                bool better = false;
                if (it == dist.end() || nd < it->second) {
                    better = true;
                } else if (nd == it->second) {
                    const auto pit = pred.find(v);
                    // Only on a STRICTLY positive edge. At equal distance the tie-break could otherwise make a
                    // node its own ancestor -- a zero-cost self-loop or cycle lets u and v share a distance --
                    // and a predecessor cycle is a path reconstruction that never terminates.
                    if (w > 0 && pit != pred.end() && u < pit->second) {
                        better = true;
                    }
                }
                if (better) {
                    dist[v] = nd;
                    pred[v] = u;
                    pq.push(Entry{nd, v});
                }
            }
        }
    }
    return make_error<PathCost>(MathError::undefined_value);
}

// Multi-start tabu search: evaluates independent local search runs concurrently.
// Returns the global best state and objective value, breaking ties lexicographically by state.
// Neighbor and objective callbacks are invoked across threads and MUST be pure.
[[nodiscard]] auto parallel_multistart_tabu(std::vector<TabuState> starts, NeighborFn neighbors,
                                            ObjectiveFn objective, std::int64_t tabu_tenure,
                                            std::int64_t max_iters)
    -> Result<std::pair<TabuState, std::int64_t>> {
    using Best = std::pair<TabuState, std::int64_t>;
    if (starts.empty() || tabu_tenure < 0 || max_iters < 0) {
        return make_error<Best>(MathError::domain_error);
    }

    // Runs are completely independent, so no cross-thread synchronization or shared incumbent exists.
    const std::vector<Result<Best>> results =
        parallel::transform_index(starts.size(), [&](std::size_t i) -> Result<Best> {
            return tabu_search(starts[i], neighbors, objective, tabu_tenure, max_iters);
        });

    TabuState best_state;
    std::int64_t best_val = 0;
    bool has_best = false;

    for (const auto& res : results) {
        if (!res) {
            return res;
        }
        const auto& [state, val] = *res;
        // Deterministic reduction: lowest objective wins, ties broken by lexicographically smaller state.
        if (!has_best || val < best_val || (val == best_val && state < best_state)) {
            has_best = true;
            best_state = state;
            best_val = val;
        }
    }
    return Best{best_state, best_val};
}

// Evaluates objective over every neighbour of state concurrently and returns them in neighbour order.
// The objective callback is invoked across threads and MUST be pure.
[[nodiscard]] auto parallel_neighbourhood_scan(const TabuState& state, const NeighborFn& neighbors,
                                               ObjectiveFn objective)
    -> Result<std::vector<std::pair<TabuState, std::int64_t>>> {
    using ScannedNeighbors = std::vector<std::pair<TabuState, std::int64_t>>;

    const std::vector<TabuState> nbrs = neighbors(state);
    if (nbrs.empty()) {
        return ScannedNeighbors{};
    }

    // Evaluate the objective concurrently for each candidate neighbour.
    const std::vector<std::int64_t> vals =
        parallel::transform_index(nbrs.size(), [&](std::size_t i) -> std::int64_t {
            return objective(nbrs[i]);
        });

    ScannedNeighbors out;
    out.reserve(nbrs.size());
    for (std::size_t i = 0; i < nbrs.size(); ++i) {
        out.emplace_back(nbrs[i], vals[i]);
    }
    return out;
}
}  // namespace nimblecas
