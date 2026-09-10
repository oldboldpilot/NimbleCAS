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
(lowest node id / first-in-expansion-order wins), and parallel traversals merge
their results deterministically.

Crucially, **the predecessor tie-break applies only across strictly positive edges**,
because at equal distance it could otherwise make a node its own ancestor (for example,
via a zero-cost self-loop or cycle where $u$ and $v$ share a distance); a predecessor cycle
would cause path reconstruction to loop infinitely and never terminate.

The tabulated DP tables — and the anti-diagonal wavefront of `edit_distance_parallel` —
are the dense, rectangular, data-parallel shapes a GPU backend could offload unchanged;
there is no CUDA here, only that structure. Nothing throws: every failure travels the
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
| `SuccessorFn` | `std::function<std::vector<std::int64_t>(std::int64_t)>` | Out-neighbours of a node, in expansion order. In bidirectional search, also used for in-neighbours (predecessors). |
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

| Function | Behaviour |
| :--- | :--- |
| `bfs` | Fewest-**edges** path from `start` to the first node satisfying `goal`, by an iterative FIFO breadth-first search with a visited set. `undefined_value` if no goal is reachable. |
| `dfs_recursive` | Recursive pre-order depth-first search to the first goal within `max_depth` edges. `start` is at depth 0; a node at depth `max_depth` is tested for the goal but not expanded. A persistent visited set prevents re-expansion. `domain_error` if `max_depth < 0`; `undefined_value` if no goal within the bound. |
| `dfs_iterative` | Explicit-stack form of `dfs_recursive`. Successors are pushed reversed so the leftmost pops first, and acceptance/expansion is gated at pop time, reproducing the recursive pre-order **exactly**: for any fixed expansion order the two return the identical first-goal path. Same bound semantics and errors. |
| `iterative_deepening_dfs` | Repeated depth-limited searches with the bound growing from 0 up to `max_depth`, returning the **shallowest** goal (fewest edges, like BFS) while keeping DFS's linear memory. Cycles are broken by an on-path (ancestor) check rather than a global visited set, so a goal reachable within the bound is never missed. Same errors as the DFS forms. |
| `dijkstra` | Minimum-cost path under non-negative weights, returned as `(path, total_cost)`. A binary-heap priority queue ordered by `(distance, node id)` makes it deterministic; on equal tentative distances the predecessor with the lower node id wins across strictly positive edges. `domain_error` on a negative edge weight, `overflow` on a cost sum leaving `int64`, `undefined_value` if no goal is reachable. |
| `a_star` | Dijkstra guided by an admissible `heuristic`. Nodes are ordered by `f = g + h`, ties broken by `g` then node id; tentative costs are relaxed lazily so an admissible-but-**inconsistent** heuristic still yields the optimum. The returned total cost is the true path cost `g` and **equals** the cost `dijkstra` reports for the same problem. Same error semantics as `dijkstra`. |
| `parallel_bfs_levels` | Breadth-first search returning nodes grouped by level (level 0 is `{start}`), stopping after at most `max_levels` levels or when the frontier empties. Each frontier is expanded in parallel via `parallel::transform_index`, then the newly discovered nodes are merged through an ordered set (each level sorted ascending), so the result is identical to a serial level-BFS regardless of thread count. `domain_error` if `max_levels < 0`; an **empty** result when `max_levels == 0`. |

## A* variants and best-first heuristic search

Each algorithm in this family states its optimality guarantees explicitly. Where an
algorithm is suboptimal or incomplete, that boundary is stated plainly.

```cpp
[[nodiscard]] auto ida_star(std::int64_t start, GoalFn goal, SuccessorFn successors,
                            CostFn cost, HeuristicFn heuristic, std::int64_t max_iterations)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>>;

[[nodiscard]] auto weighted_a_star(std::int64_t start, GoalFn goal, SuccessorFn successors,
                                   CostFn cost, HeuristicFn heuristic,
                                   std::int64_t weight_numerator,
                                   std::int64_t weight_denominator)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>>;

[[nodiscard]] auto greedy_best_first(std::int64_t start, GoalFn goal, SuccessorFn successors,
                                     HeuristicFn heuristic, std::int64_t max_expansions)
    -> Result<std::vector<std::int64_t>>;

[[nodiscard]] auto beam_search(std::int64_t start, GoalFn goal, SuccessorFn successors,
                               HeuristicFn heuristic, std::int64_t beam_width,
                               std::int64_t max_levels)
    -> Result<std::vector<std::int64_t>>;

[[nodiscard]] auto bidirectional_dijkstra(std::int64_t start, std::int64_t goal_node,
                                          SuccessorFn successors, SuccessorFn predecessors,
                                          CostFn cost)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>>;
```

| Function | Behaviour |
| :--- | :--- |
| `ida_star` | Iterative-deepening A\*: depth-first passes bounded by a monotonically increasing $f = g + h$ threshold. **Strictly optimal** for an admissible heuristic, returning the exact path cost `a_star` computes, but requiring memory **linear in path depth** rather than proportional to the number of generated nodes. Cycles are prevented using an on-path ancestor set. Exhausting `max_iterations` threshold passes returns `MathError::not_converged`. |
| `weighted_a_star` | Best-first search ordering nodes by $f = g + w \cdot h$, where the weight $w \ge 1$ is specified as an exact rational `weight_numerator / weight_denominator` (no floating-point arithmetic is used). **Bounded suboptimal**: the returned path cost is guaranteed to be $\le w \times \text{optimum}$, and never claims strict optimality when $w > 1$. When $w = 1$, it behaves identically to `a_star`. Refuses $w < 1$ or zero denominator with `MathError::domain_error`. |
| `greedy_best_first` | Greedily expands the node with the smallest heuristic estimate $h$, ignoring accumulated cost $g$ entirely. **Not optimal**: prioritises finding a path rapidly without bounding cost. Returns path only (no cost). Exhausting `max_expansions` without reaching a goal returns `MathError::not_converged`. |
| `beam_search` | Breadth-first level search retaining only the `beam_width` most promising candidate nodes per level, ranked by $(h, \text{node\_id})$ ascending. **Not optimal and incomplete**: pruning nodes at each depth level means a narrow beam can discard the sole route to a reachable goal. Terminating after `max_levels` without reaching a goal returns `MathError::not_converged`. `beam_width < 1` yields `MathError::domain_error`. |
| `bidirectional_dijkstra` | Simultaneous forward search from `start` and backward search from `goal_node` using `predecessors`. Meets with exact termination condition: halts only when $\min f + \min b \ge \mu$ (where $\mu$ is the best meeting path cost seen), rather than terminating on first contact. **Strictly optimal**: returns the identical optimal `(path, total_cost)` as `dijkstra`. |

## Shared-memory parallel search

Parallel routines evaluate pure callbacks concurrently across threads using
[`nimblecas.parallel`](parallel.md) while maintaining bit-identical results with their
serial counterparts.

```cpp
[[nodiscard]] auto parallel_a_star(std::int64_t start, GoalFn goal, SuccessorFn successors,
                                   CostFn cost, HeuristicFn heuristic)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>>;

[[nodiscard]] auto parallel_dijkstra(std::int64_t start, GoalFn goal, SuccessorFn successors,
                                     CostFn cost)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>>;

[[nodiscard]] auto parallel_multistart_tabu(std::vector<TabuState> starts, NeighborFn neighbors,
                                            ObjectiveFn objective, std::int64_t tabu_tenure,
                                            std::int64_t max_iters)
    -> Result<std::pair<TabuState, std::int64_t>>;

[[nodiscard]] auto parallel_neighbourhood_scan(const TabuState& state, NeighborFn neighbors,
                                               ObjectiveFn objective)
    -> Result<std::vector<std::pair<TabuState, std::int64_t>>>;
```

| Function | Behaviour |
| :--- | :--- |
| `parallel_a_star` | Evaluates node expansions across each wavefront concurrently across threads via `parallel::transform_index` when the wave size meets `parallel_cost_threshold`. Node relaxations and priority queue updates are committed serially in deterministic wave order. Returns the **identical path and cost** as serial `a_star`. Callbacks must be pure. |
| `parallel_dijkstra` | Wave-based concurrent expansion matching serial `dijkstra` bit for bit on all thread counts. |
| `parallel_multistart_tabu` | Runs independent tabu search walks concurrently from each state in `starts`. Completely decoupled across threads; the overall best `(state, value)` is reduced deterministically (lowest objective value wins, ties broken by lexicographically smaller state). Returns `MathError::domain_error` if `starts` is empty. |
| `parallel_neighbourhood_scan` | Evaluates `objective` across all neighbours of `state` concurrently, returning neighbour-objective pairs in original neighbour order. Serves as a building block for custom parallel local search algorithms. |

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

| Function | Behaviour |
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

| Function | Behaviour |
| :--- | :--- |
| `edit_distance` | Levenshtein edit distance (insert/delete/substitute, unit cost) between `a` and `b`, bottom-up with a rolling two-row table. e.g. `edit_distance("kitten", "sitting") == 3`. |
| `edit_distance_memo` | Recursive memoised form; equal to the tabulated value. |
| `edit_distance_parallel` | Anti-diagonal parallel form: the `(n+1)×(m+1)` table is filled one anti-diagonal $d = i + j$ at a time, and the mutually independent interior cells of each diagonal are computed with `parallel::transform_index` (one launch per diagonal — the wavefront a GPU kernel would map). Cell values are bounded by $n + m$, so it equals `edit_distance` cell-for-cell for every input. Returns `overflow` only if the flat table dimension `rows * cols` would exceed `std::size_t`. |
| `longest_common_subsequence` | Length of the longest common subsequence of integer sequences `a` and `b`, bottom-up with a rolling two-row table. |
| `longest_common_subsequence_memo` | Recursive memoised form; equal to the tabulated length. |
| `knapsack_01` | Maximum total value of a subset of items whose weights fit within `capacity`, each item used at most once (0/1 knapsack), via the standard 1-D capacity table. `domain_error` if the weight/value counts differ, `capacity` is negative, or a weight is negative; `overflow` if a value sum exceeds `std::int64_t`. |

## Error model

| Condition | Error |
| :--- | :--- |
| Goal unreachable (`bfs`, `dfs_*`, `iterative_deepening_dfs`, `dijkstra`, `a_star`, `ida_star`, `weighted_a_star`, `greedy_best_first`, `beam_search`, `bidirectional_dijkstra`, `parallel_*`) | `MathError::undefined_value` |
| `max_depth < 0` (`dfs_recursive`, `dfs_iterative`, `iterative_deepening_dfs`) | `MathError::domain_error` |
| `max_levels < 0` (`parallel_bfs_levels`, `beam_search`) | `MathError::domain_error` |
| `max_iterations < 0` (`ida_star`) or `max_expansions < 0` (`greedy_best_first`) | `MathError::domain_error` |
| `beam_width < 1` (`beam_search`) | `MathError::domain_error` |
| `weight_denominator <= 0` or `weight_numerator < weight_denominator` (`weighted_a_star`) | `MathError::domain_error` |
| Iteration or expansion budget exhausted (`ida_star`, `greedy_best_first`, `beam_search`) | `MathError::not_converged` |
| Negative edge weight traversed (`dijkstra`, `a_star`, `ida_star`, `weighted_a_star`, `bidirectional_dijkstra`, `parallel_*`) | `MathError::domain_error` |
| Negative heuristic value returned (`a_star`, `ida_star`, `weighted_a_star`, `greedy_best_first`, `beam_search`, `parallel_a_star`) | `MathError::domain_error` |
| Path-cost sum leaves `std::int64_t` (`dijkstra`, `a_star`, `ida_star`, `weighted_a_star`, `bidirectional_dijkstra`, `parallel_*`) | `MathError::overflow` |
| `tabu_tenure < 0` or `max_iters < 0` (`tabu_search`, `parallel_multistart_tabu`) | `MathError::domain_error` |
| `starts` vector is empty (`parallel_multistart_tabu`) | `MathError::domain_error` |
| `knapsack_01`: weight/value counts differ, negative capacity, or negative weight | `MathError::domain_error` |
| DP accumulation leaves `std::int64_t` (`edit_distance`, `edit_distance_memo`, `longest_common_subsequence*`, `knapsack_01`) | `MathError::overflow` |
| Flat table dimension `rows * cols` exceeds `std::size_t` (`edit_distance_parallel`) | `MathError::overflow` |

`parallel_bfs_levels` with `max_levels == 0` is **not** an error — it succeeds
with an empty level list. `bfs` and other traversals where `start` is already a goal
succeed with `{start}` (and cost 0 where applicable).

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

// In-neighbours for bidirectional search:
auto predecessors = [](std::int64_t n) -> std::vector<std::int64_t> {
    switch (n) {
        case 1: return {0};
        case 2: return {0};
        case 3: return {1};
        case 4: return {3};
        case 6: return {2, 4, 5};
        default: return {};
    }
};

// Bidirectional Dijkstra matches dijkstra and a_star optimal cost exactly:
auto bi = bidirectional_dijkstra(0, 6, successors, predecessors, cost).value();
bi.first;                                           // [0, 1, 3, 4, 6]
bi.second;                                          // 4

// IDA* achieves the same optimal cost in memory linear in search depth:
auto ida = ida_star(0, is(6), successors, cost, h, 20).value();
ida.first == d.first && ida.second == d.second;     // true — cost 4

// Weighted A* (exact rational w = 3/2, bounded suboptimal: cost <= 1.5 * 4 = 6):
auto wa = weighted_a_star(0, is(6), successors, cost, h, 3, 2).value();
wa.second <= 6;                                     // true (finds cost 4 on this graph)

// Greedy best-first (fast heuristic expansion, not cost-optimal):
auto gbf = greedy_best_first(0, is(6), successors, h, 50).value();

// Beam search (beam width 2, max 10 levels; incomplete):
auto bm = beam_search(0, is(6), successors, h, 2, 10).value();

// Parallel Dijkstra and Parallel A* reproduce serial results bit for bit:
auto pd = parallel_dijkstra(0, is(6), successors, cost).value();
auto pa = parallel_a_star(0, is(6), successors, cost, h).value();
pd.first == d.first && pd.second == d.second;       // true
pa.first == a.first && pa.second == a.second;       // true

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

// Parallel multi-start tabu search: independent walks from multiple seeds:
std::vector<TabuState> starts = {{-10, -10}, {0, 0}, {10, 10}};
auto pbest = parallel_multistart_tabu(starts, neighbors, objective, 3, 100).value();
pbest.first;                                        // {3, -2}
pbest.second;                                       // 0

// Parallel neighbourhood scan: scores candidate neighbours concurrently:
auto scanned = parallel_neighbourhood_scan(TabuState{3, -2}, neighbors, objective).value();

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

- [`nimblecas.search_dist`](search_dist.md) — distributed graph searches over `taskdag`.
- [`nimblecas.parallel`](parallel.md) — deterministic fork–join substrate.
- [`nimblecas.core`](core.md) — `Result<T>`, `MathError`, and checked integer arithmetic.
- [`nimblecas.csp`](csp.md) and [`nimblecas.bitcsp`](bitcsp.md) — constraint satisfaction solvers.
- [Documentation hub](../Index.md)
