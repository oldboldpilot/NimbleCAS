# `nimblecas.search` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/search/search.cppm`

A deterministic toolkit of graph/tree traversal, shortest-path, metaheuristic,
and dynamic-programming routines — the combinatorial **algorithmics layer** that
sits above the [`parallel`](parallel.md) fork–join substrate and alongside the
exact numeric modules. Graphs are never materialised: a node is a
`std::int64_t` id, a `SuccessorFn` returns each node's out-neighbours in the
exact order they should be expanded, goals are predicates, and edge weights are
functions. Everything is **exact integer** work — there is no floating point and
no numerical tolerance anywhere in this module.

Two honesty boundaries govern the layer. **Overflow:** every path-cost and
almost every DP-table accumulation is `int64` and overflow-guarded through a
checked `add_overflow`; a sum that would leave `std::int64_t` surfaces as
`MathError::overflow` rather than wrapping (Rule 32). The one exception is
`edit_distance_parallel`, whose cells are provably bounded by `n + m` and so
cannot overflow the accumulation; it instead guards its flat table dimension
`rows * cols` against `std::size_t` overflow before allocating. **Determinism:** every
function returns the *same* result regardless of how many worker threads the
parallel backend uses. Frontiers, priority queues and tie-breaks are all ordered
(lowest node id / first-in-expansion-order wins), and the one parallel traversal
here (`parallel_bfs_levels`) merges its per-node results through an ordered set,
so it reproduces a serial level-BFS bit-for-bit. The tabulated DP tables — and
the anti-diagonal wavefront of `edit_distance_parallel` — are the dense,
rectangular, data-parallel shapes a GPU backend could offload unchanged; there
is no CUDA here, only that structure. Nothing throws: every failure travels the
`Result<T>` / `MathError` railway.

```cpp
import nimblecas.search;
```

Depends on [`core`](core.md) and [`parallel`](parallel.md).

## Graph description callbacks

A search problem is described entirely by these callbacks; no adjacency
structure is ever built. The cost function must return a **non-negative** edge
weight, and the heuristic must be an **admissible** lower bound on the remaining
cost to a goal.

| Alias | Definition | Role |
| :--- | :--- | :--- |
| `SuccessorFn` | `std::function<std::vector<std::int64_t>(std::int64_t)>` | Out-neighbours of a node, in expansion order. |
| `GoalFn` | `std::function<bool(std::int64_t)>` | Acceptance predicate. |
| `CostFn` | `std::function<std::int64_t(std::int64_t, std::int64_t)>` | Non-negative weight of edge `(u, v)`. |
| `HeuristicFn` | `std::function<std::int64_t(std::int64_t)>` | Admissible lower bound on remaining cost to a goal. |

## Graph & tree search

Every traversal returns a path that includes both `start` and the goal node. If
`start` already satisfies `goal` the path is just `{start}`. Unreachable goals
yield `MathError::undefined_value`.

```cpp
[[nodiscard]] auto bfs(std::int64_t start, GoalFn goal, SuccessorFn successors)
    -> Result<std::vector<std::int64_t>>;

[[nodiscard]] auto dfs_recursive(std::int64_t start, GoalFn goal, SuccessorFn successors,
                                 std::int64_t max_depth) -> Result<std::vector<std::int64_t>>;

[[nodiscard]] auto dfs_iterative(std::int64_t start, GoalFn goal, SuccessorFn successors,
                                 std::int64_t max_depth) -> Result<std::vector<std::int64_t>>;

[[nodiscard]] auto iterative_deepening_dfs(std::int64_t start, GoalFn goal,
                                           SuccessorFn successors, std::int64_t max_depth)
    -> Result<std::vector<std::int64_t>>;

[[nodiscard]] auto dijkstra(std::int64_t start, GoalFn goal, SuccessorFn successors,
                            CostFn cost)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>>;

[[nodiscard]] auto a_star(std::int64_t start, GoalFn goal, SuccessorFn successors,
                          CostFn cost, HeuristicFn heuristic)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>>;

[[nodiscard]] auto parallel_bfs_levels(std::int64_t start, SuccessorFn successors,
                                       std::int64_t max_levels)
    -> Result<std::vector<std::vector<std::int64_t>>>;
```

| Function | Behavior |
| :--- | :--- |
| `bfs` | Fewest-**edges** path from `start` to the first node satisfying `goal`, by an iterative FIFO breadth-first search with a visited set. `undefined_value` if no goal is reachable. |
| `dfs_recursive` | Recursive pre-order depth-first search to the first goal within `max_depth` edges. `start` is at depth 0; a node at depth `max_depth` is tested for the goal but not expanded. A persistent visited set prevents re-expansion. `domain_error` if `max_depth < 0`; `undefined_value` if no goal within the bound. |
| `dfs_iterative` | Explicit-stack form of `dfs_recursive`. Successors are pushed reversed so the leftmost pops first, and acceptance/expansion is gated at pop time, reproducing the recursive pre-order **exactly**: for any fixed expansion order the two return the identical first-goal path. Same bound semantics and errors. |
| `iterative_deepening_dfs` | Repeated depth-limited searches with the bound growing from 0 up to `max_depth`, returning the **shallowest** goal (fewest edges, like BFS) while keeping DFS's linear memory. Cycles are broken by an on-path (ancestor) check rather than a global visited set, so a goal reachable within the bound is never missed. Same errors as the DFS forms. |
| `dijkstra` | Minimum-cost path under non-negative weights, returned as `(path, total_cost)`. A binary-heap priority queue ordered by `(distance, node id)` makes it deterministic; on equal tentative distances the predecessor with the lower node id wins. `domain_error` on a negative edge weight, `overflow` on a cost sum leaving `int64`, `undefined_value` if no goal is reachable. |
| `a_star` | Dijkstra guided by an admissible `heuristic`. Nodes are ordered by `f = g + h`, ties broken by `g` then node id; tentative costs are relaxed lazily so an admissible-but-**inconsistent** heuristic still yields the optimum. The returned total cost is the true path cost `g` and **equals** the cost `dijkstra` reports for the same problem. Same error semantics as `dijkstra`. |
| `parallel_bfs_levels` | Breadth-first search returning nodes grouped by level (level 0 is `{start}`), stopping after at most `max_levels` levels or when the frontier empties. Each frontier is expanded in parallel via `parallel::transform_index`, then the newly discovered nodes are merged through an ordered set (each level sorted ascending), so the result is identical to a serial level-BFS regardless of thread count. `domain_error` if `max_levels < 0`; an **empty** result when `max_levels == 0`. |

## Tabu search (metaheuristic local search)

| Alias | Definition | Role |
| :--- | :--- | :--- |
| `TabuState` | `std::vector<std::int64_t>` | A candidate solution. |
| `NeighborFn` | `std::function<std::vector<TabuState>(const TabuState&)>` | Neighbourhood of a state. |
| `ObjectiveFn` | `std::function<std::int64_t(const TabuState&)>` | Value to minimise. |

```cpp
[[nodiscard]] auto tabu_search(TabuState initial, NeighborFn neighbors,
                               ObjectiveFn objective, std::int64_t tabu_tenure,
                               std::int64_t max_iters)
    -> Result<std::pair<TabuState, std::int64_t>>;
```

| Function | Behavior |
| :--- | :--- |
| `tabu_search` | Minimise `objective` over the neighbourhood graph induced by `neighbors`, starting from `initial`. Each iteration moves to the best admissible neighbour — lowest objective value, ties broken by the lexicographically smallest state — where a neighbour is forbidden while it sits in the tabu list (a recency memory of length `tabu_tenure`), unless it beats the incumbent best (**aspiration**). The overall best `(state, value)` ever seen is returned. Fully deterministic; terminates after at most `max_iters` iterations, or earlier if no admissible move exists. `domain_error` if `tabu_tenure < 0` or `max_iters < 0`. |

Tabu search is a heuristic, not an exact solver: it returns the best state its
bounded walk encountered, which is the global optimum only when the
neighbourhood and iteration budget permit reaching it.

## Dynamic programming (exact integers, overflow-guarded)

Each problem is offered in a tabulated (bottom-up) and a memoised (top-down)
form that return the **identical** value; `edit_distance` additionally has a
parallel anti-diagonal form. Every accumulation is overflow-guarded.

```cpp
[[nodiscard]] auto edit_distance(std::string_view a, std::string_view b) -> Result<std::int64_t>;

[[nodiscard]] auto edit_distance_memo(std::string_view a, std::string_view b)
    -> Result<std::int64_t>;

[[nodiscard]] auto edit_distance_parallel(std::string_view a, std::string_view b)
    -> Result<std::int64_t>;

[[nodiscard]] auto longest_common_subsequence(std::span<const std::int64_t> a,
                                              std::span<const std::int64_t> b)
    -> Result<std::int64_t>;

[[nodiscard]] auto longest_common_subsequence_memo(std::span<const std::int64_t> a,
                                                   std::span<const std::int64_t> b)
    -> Result<std::int64_t>;

[[nodiscard]] auto knapsack_01(std::span<const std::int64_t> weights,
                               std::span<const std::int64_t> values, std::int64_t capacity)
    -> Result<std::int64_t>;
```

| Function | Behavior |
| :--- | :--- |
| `edit_distance` | Levenshtein edit distance (insert/delete/substitute, unit cost) between `a` and `b`, bottom-up with a rolling two-row table. e.g. `edit_distance("kitten", "sitting") == 3`. |
| `edit_distance_memo` | Recursive memoised form; equal to the tabulated value. |
| `edit_distance_parallel` | Anti-diagonal parallel form: the `(n+1)×(m+1)` table is filled one anti-diagonal `d = i + j` at a time, and the mutually independent interior cells of each diagonal are computed with `parallel::transform_index` (one launch per diagonal — the wavefront a GPU kernel would map). Cell values are bounded by `n + m`, so it equals `edit_distance` cell-for-cell for every input. Returns `overflow` only if the flat table dimension `rows * cols` would exceed `std::size_t`. |
| `longest_common_subsequence` | Length of the longest common subsequence of integer sequences `a` and `b`, bottom-up with a rolling two-row table. |
| `longest_common_subsequence_memo` | Recursive memoised form; equal to the tabulated length. |
| `knapsack_01` | Maximum total value of a subset of items whose weights fit within `capacity`, each item used at most once (0/1 knapsack), via the standard 1-D capacity table. `domain_error` if the weight/value counts differ, `capacity` is negative, or a weight is negative; `overflow` if a value sum exceeds `std::int64_t`. |

## Error model

| Condition | Error |
| :--- | :--- |
| Goal unreachable (`bfs`, `dfs_*`, `iterative_deepening_dfs`, `dijkstra`, `a_star`) | `MathError::undefined_value` |
| `max_depth < 0` (`dfs_recursive`, `dfs_iterative`, `iterative_deepening_dfs`) | `MathError::domain_error` |
| `max_levels < 0` (`parallel_bfs_levels`) | `MathError::domain_error` |
| Negative edge weight traversed (`dijkstra`, `a_star`) | `MathError::domain_error` |
| Path-cost sum leaves `std::int64_t` (`dijkstra`, `a_star`) | `MathError::overflow` |
| `tabu_tenure < 0` or `max_iters < 0` (`tabu_search`) | `MathError::domain_error` |
| `knapsack_01`: weight/value counts differ, negative capacity, or negative weight | `MathError::domain_error` |
| DP accumulation leaves `std::int64_t` (`edit_distance`, `edit_distance_memo`, `longest_common_subsequence*`, `knapsack_01`) | `MathError::overflow` |
| Flat table dimension `rows * cols` exceeds `std::size_t` (`edit_distance_parallel`) | `MathError::overflow` |

`parallel_bfs_levels` with `max_levels == 0` is **not** an error — it succeeds
with an empty level list. `bfs` when `start` is already a goal succeeds with
`{start}`.

## Worked examples

```cpp
import nimblecas.search;
import nimblecas.core;
using namespace nimblecas;

// A small fixed directed graph (successor order is significant):
//   0 -> [1, 2]   1 -> [3]   2 -> [6]   3 -> [4]   4 -> [6]   5 -> [6]
//   6, 7 are sinks; 5 and 7 are unreachable from 0.
auto successors = [](std::int64_t n) -> std::vector<std::int64_t> {
    switch (n) {
        case 0: return {1, 2};
        case 1: return {3};
        case 2: return {6};
        case 3: return {4};
        case 4: return {6};
        case 5: return {6};
        default: return {};
    }
};
auto is = [](std::int64_t target) -> GoalFn {
    return [target](std::int64_t n) { return n == target; };
};

// BFS finds the fewest-EDGES path; DFS descends the first branch fully.
bfs(0, is(6), successors).value();                 // [0, 2, 6]
dfs_recursive(0, is(6), successors, 10).value();   // [0, 1, 3, 4, 6]
dfs_iterative(0, is(6), successors, 10).value();   // [0, 1, 3, 4, 6]  (== recursive)
iterative_deepening_dfs(0, is(6), successors, 10).value();  // [0, 2, 6] (shallowest)

// Unreachable / bound violations travel the railway.
bfs(0, is(7), successors).error();                 // MathError::undefined_value
dfs_iterative(0, is(6), successors, 1).error();    // undefined_value (goal beyond depth 1)
dfs_iterative(0, is(6), successors, -1).error();   // MathError::domain_error

// Weighted shortest paths. The cheap route is the LONG branch (cost 4),
// beating the 2-edge [0,2,6] whose 0->2 weight is 5.
auto cost = [](std::int64_t u, std::int64_t v) -> std::int64_t {
    if (u == 0 && v == 2) return 5;
    return 1;  // every other edge costs 1
};
auto h = [](std::int64_t n) -> std::int64_t {  // admissible lower bound to node 6
    switch (n) { case 0: return 2; case 1: return 2; case 2: return 1;
                 case 3: return 1; case 4: return 1; default: return 0; }
};
auto d = dijkstra(0, is(6), successors, cost).value();
d.first;                                            // [0, 1, 3, 4, 6]
d.second;                                           // 4
auto a = a_star(0, is(6), successors, cost, h).value();
a.first == d.first && a.second == d.second;         // true — A* matches Dijkstra

// Parallel level-BFS is identical to a serial level-BFS (thread-count independent).
parallel_bfs_levels(0, successors, 10).value();     // [[0], [1, 2], [3, 6], [4]]
parallel_bfs_levels(0, successors, 2).value();      // [[0], [1, 2]]
parallel_bfs_levels(0, successors, 0).value();      // []  (success, not an error)

// Tabu search: minimise (x-3)^2 + (y+2)^2 over the integer grid [-10,10]^2.
auto objective = [](const TabuState& s) -> std::int64_t {
    const std::int64_t dx = s[0] - 3, dy = s[1] + 2;
    return dx * dx + dy * dy;
};
auto neighbors = [](const TabuState& s) -> std::vector<TabuState> {
    std::vector<TabuState> out;
    for (auto [dx, dy] : {std::pair{-1, 0}, {1, 0}, {0, -1}, {0, 1}}) {
        const std::int64_t nx = s[0] + dx, ny = s[1] + dy;
        if (nx >= -10 && nx <= 10 && ny >= -10 && ny <= 10) out.push_back({nx, ny});
    }
    return out;
};
auto best = tabu_search(TabuState{0, 0}, neighbors, objective, 3, 100).value();
best.first;                                         // {3, -2}
best.second;                                        // 0

// Dynamic programming: three edit-distance forms agree exactly.
edit_distance("kitten", "sitting").value();         // 3
edit_distance_memo("kitten", "sitting").value();    // 3
edit_distance_parallel("kitten", "sitting").value();// 3
edit_distance("abc", "abc").value();                // 0

// LCS of [1,2,3,4,5] and [2,4,5] is [2,4,5], length 3.
const std::vector<std::int64_t> x{1, 2, 3, 4, 5}, y{2, 4, 5};
longest_common_subsequence(x, y).value();           // 3
longest_common_subsequence_memo(x, y).value();      // 3

// 0/1 knapsack: items (w,v) = (1,1),(3,4),(4,5),(5,7), capacity 7 => 3+4 = 9.
const std::vector<std::int64_t> w{1, 3, 4, 5}, v{1, 4, 5, 7};
knapsack_01(w, v, 7).value();                       // 9
knapsack_01(w, {1}, 7).error();                     // domain_error (size mismatch)
knapsack_01(w, v, -1).error();                      // domain_error (negative capacity)
```

## See also

- [`nimblecas.parallel`](parallel.md) — the deterministic fork–join layer whose
  `transform_index` drives `parallel_bfs_levels` and `edit_distance_parallel`.
- [`nimblecas.core`](core.md) — the `Result<T>` / `MathError` railway every
  routine reports through.
- [`nimblecas.csp`](csp.md), [`nimblecas.bitcsp`](bitcsp.md), and
  [`nimblecas.lp`](lp.md) — sibling combinatorial / constraint-solving modules in
  the algorithmics layer.
- [Documentation hub](../Index.md)
```
