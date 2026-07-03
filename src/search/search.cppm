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
[[nodiscard]] auto bfs(std::int64_t start, GoalFn goal, SuccessorFn successors)
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
[[nodiscard]] auto dfs_iterative(std::int64_t start, GoalFn goal, SuccessorFn successors,
                                 std::int64_t max_depth) -> Result<std::vector<std::int64_t>>;

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
[[nodiscard]] auto dijkstra(std::int64_t start, GoalFn goal, SuccessorFn successors,
                            CostFn cost)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>>;

// A* search: Dijkstra guided by an admissible `heuristic` (a lower bound on the remaining
// cost to a goal). Nodes are ordered by f = g + h with ties broken by g then node id, and
// tentative costs are relaxed lazily so an admissible-but-inconsistent heuristic still
// yields the optimum. The returned total cost is the true path cost g and EQUALS the cost
// `dijkstra` reports for the same problem. Same error semantics as `dijkstra`.
[[nodiscard]] auto a_star(std::int64_t start, GoalFn goal, SuccessorFn successors,
                          CostFn cost, HeuristicFn heuristic)
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
[[nodiscard]] auto tabu_search(TabuState initial, NeighborFn neighbors,
                               ObjectiveFn objective, std::int64_t tabu_tenure,
                               std::int64_t max_iters)
    -> Result<std::pair<TabuState, std::int64_t>>;

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
// (start .. goal). `pred` must contain a link for every node on the chain except `start`.
[[nodiscard]] auto reconstruct(const std::unordered_map<std::int64_t, std::int64_t>& pred,
                               std::int64_t start, std::int64_t goal)
    -> std::vector<std::int64_t> {
    std::vector<std::int64_t> path;
    std::int64_t cur = goal;
    path.push_back(cur);
    while (cur != start) {
        cur = pred.at(cur);
        path.push_back(cur);
    }
    std::reverse(path.begin(), path.end());
    return path;
}

}  // namespace

auto bfs(std::int64_t start, GoalFn goal, SuccessorFn successors)
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
                return reconstruct(pred, start, v);
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

auto dfs_iterative(std::int64_t start, GoalFn goal, SuccessorFn successors,
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

auto dijkstra(std::int64_t start, GoalFn goal, SuccessorFn successors, CostFn cost)
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
            return PathCost{reconstruct(pred, start, u), d};
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
                if (pit != pred.end() && u < pit->second) {
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

auto a_star(std::int64_t start, GoalFn goal, SuccessorFn successors, CostFn cost,
            HeuristicFn heuristic)
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
            return PathCost{reconstruct(pred, start, u), g};
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
                if (pit != pred.end() && u < pit->second) {
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

auto tabu_search(TabuState initial, NeighborFn neighbors, ObjectiveFn objective,
                 std::int64_t tabu_tenure, std::int64_t max_iters)
    -> Result<std::pair<TabuState, std::int64_t>> {
    using Best = std::pair<TabuState, std::int64_t>;
    if (tabu_tenure < 0 || max_iters < 0) {
        return make_error<Best>(MathError::domain_error);
    }

    TabuState current = initial;
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

}  // namespace nimblecas
