# `nimblecas.search_dist` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/search_dist/search_dist.cppm`

A comprehensive distributed graph algorithmics toolkit built on [`nimblecas.taskdag`](taskdag.md).
While [`nimblecas.search`](search.md) executes graph traversals, shortest-path routines, and
metaheuristics across shared-memory threads on a single node, `nimblecas.search_dist` decomposes
these computations into per-round task graphs that can be distributed across processes and
cluster nodes via any conforming `Executor` (including [`nimblecas.taskdag_sgee`](taskdag_sgee.md)).

The module is governed by a rigorous **honesty boundary** (Code Policy Rule 32):

- **The graph must be data (`WireGraph`), not callbacks:** the in-process search API
  specifies graphs via closures (`SuccessorFn`, `CostFn`, `HeuristicFn`). Closures cannot
  be serialised across an ABI boundary, and a `TaskFn` receives only opaque bytes (`Payload`).
  Distributed execution therefore requires the graph to be concrete data: dense integer node
  identities `0 .. n-1`, explicit edge structures, and optional heuristic or landscape vectors.
  Adapter functions (`successors_of`, `cost_of`, `heuristic_of`) project a `WireGraph` back
  into the in-process callback signatures so that serial and distributed solvers can be tested
  on identical graph instances.
- **Round-based relaxation vs Dijkstra:** Dijkstra's algorithm sequentially settles one
  minimum-key node at a time. This strict sequential dependency cannot be distributed.
  Instead, `distributed_sssp` and `distributed_shortest_path` employ **round-based frontier
  relaxation**: in each round, every node whose tentative distance improved in the prior round
  relaxes its outgoing edges in parallel. A shortest path of $k$ edges is guaranteed to be fully
  relaxed after $k$ rounds.
- **Costs are exact; paths match where optimal paths are unique:** round-based relaxation
  over non-negative edge weights provably computes the **exact minimum path cost**, matching
  `dijkstra` and `a_star` without numeric tolerance. When multiple paths share the optimal cost,
  ties are broken toward the lowest parent node id; however, because round relaxation discovers
  relaxations in a different order than Dijkstra's priority queue, the selected parent may differ.
  Path equality is therefore guaranteed **only when the optimum is unique**.
- **No `distributed_dfs` on purpose:** depth-first search is defined entirely by its sequential
  visit order (exploring the leftmost branch to completion before touching a sibling). This order
  has no independent sub-problems to distribute. True parallel DFS relies on dynamic work
  stealing, which requires workers to enqueue discovered work; a `TaskFn` has no broker handle
  and cannot spawn tasks dynamically. What survives distribution is DFS's **reachability
  answer**: [`distributed_reachable_set`](#connectivity-ordering-and-structure) computes the
  complete set of reachable nodes via parallel frontier rounds, matching the reachability of
  `dfs_iterative` without pretending to preserve DFS traversal order.
- **`distributed_strongly_connected_components` is quadratic:** strongly connected components
  are computed from all-pairs reachability derived from `distributed_floyd_warshall`. This
  imposes an $O(V^2)$ cost in both memory and communication. It is appropriate for dense or
  moderate graphs, but callers should be aware of this quadratic scaling before sizing large
  sparse workloads.
- **Cost of the wire:** each round re-transmits each shard's slice of adjacency data. A search
  requiring $R$ rounds transmits approximately $R$ times the graph's size, trading network
  bandwidth for embarrassingly parallel edge relaxation.
- **Bounded suboptimality of weighted A\*:** `distributed_weighted_a_star` scales the heuristic
  by an exact rational weight $w \ge 1$. It guarantees only that the returned path cost is
  $\le w \times \text{optimum}$; it never claims strict optimality for $w > 1$.

```cpp
import nimblecas.search_dist;
```

Depends on [`core`](core.md) (`Result`, `MathError`, `Payload`), [`search`](search.md)
(`CostFn`, `GoalFn`, `HeuristicFn`, `SuccessorFn`, `TabuState`), and [`taskdag`](taskdag.md)
(`TaskGraph`, `TaskRegistry`, `Executor`).

## Graph representation and the semiring abstraction

```cpp
enum class Semiring : std::uint8_t { min_plus, max_min };

struct Edge {
    std::int64_t target{0};
    std::int64_t cost{0};

    [[nodiscard]] auto operator==(const Edge&) const noexcept -> bool = default;
};

struct WireGraph {
    std::vector<std::vector<Edge>> adjacency;
    std::vector<std::int64_t> heuristic;
    std::vector<std::int64_t> landscape;

    [[nodiscard]] auto node_count() const noexcept -> std::size_t { return adjacency.size(); }
};
```

### The semiring abstraction

The round relaxation engine is parameterised by an algebraic semiring:

- `Semiring::min_plus`: path cost is accumulated via addition ($\sum c_i$), and the optimal
  path minimises total cost. Used for shortest-path algorithms (`distributed_sssp`,
  `distributed_shortest_path`, `distributed_a_star`).
- `Semiring::max_min`: path capacity is the minimum edge weight along the route ($\min c_i$),
  and the optimal path maximises this bottleneck. Used for widest paths (`distributed_widest_path`).

Both problems share the identical sharding, frontier communication, and proposal merging loops.

### Validation and in-process adapters

```cpp
[[nodiscard]] auto validate(const WireGraph& g) -> Result<void>;

[[nodiscard]] auto successors_of(const WireGraph& g) -> SuccessorFn;
[[nodiscard]] auto cost_of(const WireGraph& g) -> CostFn;
[[nodiscard]] auto heuristic_of(const WireGraph& g) -> HeuristicFn;
```

- `validate` verifies that all edge targets are within `[0, node_count())`, all edge costs are
  non-negative, and that `heuristic` and `landscape` vectors are either empty or sized exactly to
  `node_count()` with non-negative heuristic values.
- `successors_of`, `cost_of`, and `heuristic_of` produce standard `std::function` callbacks
  matching the `nimblecas.search` API, enabling direct cross-validation.

## Wire protocol and task registration

```cpp
inline constexpr std::string_view relax_op_id = "nimblecas.search.relax_shard/v1";
inline constexpr std::string_view tabu_op_id  = "nimblecas.search.tabu_run/v1";
inline constexpr std::string_view dls_op_id   = "nimblecas.search.depth_limited/v1";
inline constexpr std::string_view fw_op_id    = "nimblecas.search.floyd_block/v1";
inline constexpr std::string_view tri_op_id   = "nimblecas.search.triangle_shard/v1";
inline constexpr std::string_view mst_op_id   = "nimblecas.search.boruvka_shard/v1";

struct FrontierEntry {
    std::int64_t node{0};
    std::int64_t distance{0};
    [[nodiscard]] auto operator==(const FrontierEntry&) const noexcept -> bool = default;
};

struct Proposal {
    std::int64_t target{0};
    std::int64_t distance{0};
    std::int64_t parent{0};
    [[nodiscard]] auto operator==(const Proposal&) const noexcept -> bool = default;
};

[[nodiscard]] auto register_ops(TaskRegistry& reg) -> Result<void>;

[[nodiscard]] auto encode_slice(const WireGraph& g, std::size_t first, std::size_t count)
    -> Result<Payload>;
[[nodiscard]] auto encode_frontier(std::span<const FrontierEntry> frontier, std::int64_t bound)
    -> Result<Payload>;
[[nodiscard]] auto decode_proposals(std::span<const std::byte> bytes)
    -> Result<std::vector<Proposal>>;

[[nodiscard]] auto build_round_graph(const TaskRegistry& reg, const WireGraph& g,
                                     std::span<const FrontierEntry> frontier, std::int64_t bound,
                                     std::size_t shard_count) -> Result<TaskGraph>;
```

`register_ops` must be invoked on the coordinator and on every remote worker before execution.
Wire payloads use little-endian binary encodings with distinct magic headers (`NCS_SLC1`,
`NCS_FRN1`, `NCS_PRP1`, etc.).

## Algorithm reference

### Shortest paths and traversal

```cpp
[[nodiscard]] auto distributed_sssp(const WireGraph& g, std::int64_t start,
                                    std::size_t shard_count, Executor& exec)
    -> Result<std::vector<std::optional<std::int64_t>>>;

[[nodiscard]] auto distributed_shortest_path(const WireGraph& g, std::int64_t start,
                                             std::int64_t goal_node, std::size_t shard_count,
                                             Executor& exec)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>>;

[[nodiscard]] auto distributed_a_star(const WireGraph& g, std::int64_t start,
                                      std::int64_t goal_node, std::size_t shard_count,
                                      Executor& exec)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>>;

[[nodiscard]] auto distributed_weighted_a_star(const WireGraph& g, std::int64_t start,
                                               std::int64_t goal_node, std::int64_t weight_num,
                                               std::int64_t weight_den, std::size_t shard_count,
                                               Executor& exec)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>>;

[[nodiscard]] auto distributed_bfs(const WireGraph& g, std::int64_t start,
                                   std::span<const std::int64_t> goals, std::size_t shard_count,
                                   Executor& exec) -> Result<std::vector<std::int64_t>>;

[[nodiscard]] auto distributed_iterative_deepening(const WireGraph& g, std::int64_t start,
                                                   std::span<const std::int64_t> goals,
                                                   std::int64_t max_depth,
                                                   std::size_t shard_count, Executor& exec)
    -> Result<std::vector<std::int64_t>>;

[[nodiscard]] auto distributed_floyd_warshall(const WireGraph& g, std::size_t shard_count,
                                              Executor& exec)
    -> Result<std::vector<std::vector<std::optional<std::int64_t>>>>;
```

### Connectivity, ordering, and structure

```cpp
[[nodiscard]] auto distributed_reachable_set(const WireGraph& g, std::int64_t start,
                                             std::size_t shard_count, Executor& exec)
    -> Result<std::vector<std::int64_t>>;

[[nodiscard]] auto distributed_connected_components(const WireGraph& g, std::size_t shard_count,
                                                    Executor& exec)
    -> Result<std::vector<std::int64_t>>;

[[nodiscard]] auto distributed_strongly_connected_components(const WireGraph& g,
                                                             std::size_t shard_count,
                                                             Executor& exec)
    -> Result<std::vector<std::int64_t>>;

[[nodiscard]] auto distributed_topological_order(const WireGraph& g, std::size_t shard_count,
                                                 Executor& exec)
    -> Result<std::vector<std::int64_t>>;

[[nodiscard]] auto distributed_k_core(const WireGraph& g, std::int64_t k, std::size_t shard_count,
                                      Executor& exec) -> Result<std::vector<std::int64_t>>;

[[nodiscard]] auto distributed_triangle_counts(const WireGraph& g, std::size_t shard_count,
                                               Executor& exec)
    -> Result<std::vector<std::int64_t>>;

[[nodiscard]] auto distributed_widest_path(const WireGraph& g, std::int64_t start,
                                           std::int64_t goal_node, std::size_t shard_count,
                                           Executor& exec)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>>;

struct MstEdge {
    std::int64_t u{0};
    std::int64_t v{0};
    std::int64_t weight{0};
    [[nodiscard]] auto operator==(const MstEdge&) const noexcept -> bool = default;
};

[[nodiscard]] auto distributed_minimum_spanning_forest(const WireGraph& g, std::size_t shard_count,
                                                       Executor& exec)
    -> Result<std::pair<std::vector<MstEdge>, std::int64_t>>;

[[nodiscard]] auto distributed_multistart_tabu(const WireGraph& g,
                                               std::span<const std::int64_t> starts,
                                               std::int64_t tabu_tenure, std::int64_t max_iters,
                                               Executor& exec)
    -> Result<std::pair<TabuState, std::int64_t>>;
```

## Guarantees and error conditions

The following table details the mathematical guarantee and honest error conditions for every
distributed routine:

| Algorithm | Mathematical guarantee | Error conditions |
| :--- | :--- | :--- |
| `distributed_sssp` | Exact minimum path costs for all reachable nodes; matches `dijkstra`. | `domain_error` on malformed graph, invalid `start`, or `shard_count == 0`. `overflow` if path cost exceeds `int64`. `not_converged` if round budget ($|V|$) is exhausted. |
| `distributed_shortest_path` | Exact path cost matching `dijkstra`. Path matches `dijkstra` when the optimal path is unique. | Same as `distributed_sssp`. `undefined_value` if `goal_node` is unreachable. |
| `distributed_a_star` | Exact path cost matching `a_star`. Heuristic prunes non-competitive proposals. Does not return distance table. | Same as `distributed_shortest_path`. Inadmissible heuristic yields incorrect paths (caller responsibility). |
| `distributed_weighted_a_star` | Bounded suboptimal: returned cost $\le \frac{\text{weight\_num}}{\text{weight\_den}} \times \text{optimum}$. | `domain_error` if `weight_den <= 0` or ratio $< 1$. Same errors as `distributed_a_star`. |
| `distributed_bfs` | Fewest-edges path to first reachable goal in `goals`. Ties broken by lowest parent node id. | `domain_error` on invalid inputs. `undefined_value` if no goal is reachable. |
| `distributed_reachable_set` | Exact set of all reachable nodes in ascending order; matches `dfs_iterative` reachable set. | `domain_error` on malformed graph or invalid `start`. |
| `distributed_iterative_deepening` | Fewest-edges hop count matching `iterative_deepening_dfs`. 1-level fan-out across root branches. | `undefined_value` if reachable graph exhausted without finding goal; `not_converged` if `max_depth` reached. |
| `distributed_floyd_warshall` | Exact all-pairs shortest distances via block-partitioned pivot rounds; matches `dijkstra`. | `domain_error` on malformed graph or zero shards. `overflow` if any accumulated path cost exceeds `int64`. |
| `distributed_connected_components` | Exact connected component labels on undirected view (smallest node id in component). | `domain_error` on malformed graph or zero shards. |
| `distributed_strongly_connected_components` | Exact strongly connected component labels on directed graph via all-pairs reachability ($O(V^2)$). | Same as `distributed_floyd_warshall`. |
| `distributed_topological_order` | Valid topological ordering of a DAG via level-synchronous in-degree peeling. | `undefined_value` if the graph contains a directed cycle (honest refusal; partial order is not emitted). |
| `distributed_k_core` | Exact maximal subgraph whose nodes have degree $\ge k$ in the simple undirected view. Empty if none. | `domain_error` if $k < 0$, malformed graph, or zero shards. |
| `distributed_triangle_counts` | Exact count of triangles incident to each node in the undirected view. | `domain_error` on malformed graph or zero shards. |
| `distributed_widest_path` | Exact widest bottleneck path and bottleneck edge capacity under `max_min` semiring. | `undefined_value` if `goal_node` unreachable. Cost is `INT64_MAX` when `start == goal_node`. |
| `distributed_minimum_spanning_forest` | Exact minimum spanning forest on undirected view via Boruvka's algorithm. Total order tie-break. | `overflow` if sum of forest edge weights exceeds `int64`. `domain_error` on malformed graph. |
| `distributed_multistart_tabu` | Independent tabu runs per start; best objective wins (ties broken lexicographically). | `domain_error` if `starts` is empty, graph has no landscape, or tenure/iters negative. |

## Worked examples

```cpp
import nimblecas.search_dist;
import nimblecas.taskdag;
import nimblecas.core;

using namespace nimblecas;
using namespace nimblecas::search_dist;

// 1. Build a WireGraph
//
//   0 -1-> 1 -2-> 2 -1-> 5   (route cost = 4)
//   0 -7-> 3 -1-> 4 -1-> 5   (route cost = 9)
WireGraph g;
g.adjacency = {
    {Edge{.target = 1, .cost = 1}, Edge{.target = 3, .cost = 7}},  // 0
    {Edge{.target = 2, .cost = 2}},                                // 1
    {Edge{.target = 5, .cost = 1}},                                // 2
    {Edge{.target = 4, .cost = 1}},                                // 3
    {Edge{.target = 5, .cost = 1}},                                // 4
    {}                                                             // 5 (sink)
};
g.heuristic = {3, 2, 1, 2, 1, 0};  // Admissible lower bounds to node 5

validate(g).value();

// 2. Initialise task registry and executor
TaskRegistry registry;
register_ops(registry).value();
auto exec = local_parallel_executor();

// 3. Distributed shortest path with unique optimum
auto res = distributed_shortest_path(g, 0, 5, 2, exec).value();
// res.first == {0, 1, 2, 5}
// res.second == 4 (matches dijkstra cost exactly)

// 4. Distributed A* pruning
auto astar_res = distributed_a_star(g, 0, 5, 2, exec).value();
// astar_res.first == {0, 1, 2, 5}
// astar_res.second == 4

// 5. Distributed connected components
auto comp = distributed_connected_components(g, 2, exec).value();
// All connected nodes receive component label 0

// 6. Topological ordering of DAG
auto topo = distributed_topological_order(g, 2, exec).value();
// topo == {0, 1, 3, 2, 4, 5} (valid topological order)

// 7. Widest bottleneck path
auto widest = distributed_widest_path(g, 0, 5, 2, exec).value();
// Path through 0 -> 1 -> 2 -> 5 has bottleneck min(1, 2, 1) = 1
// Path through 0 -> 3 -> 4 -> 5 has bottleneck min(7, 1, 1) = 1
```

## See also

- [`nimblecas.search`](search.md) — the in-process serial and parallel search algorithms.
- [`nimblecas.taskdag`](taskdag.md) — the task-DAG engine driving distributed rounds.
- [`nimblecas.taskdag_sgee`](taskdag_sgee.md) — SGEE-backed multi-node distributed executor.
- [`nimblecas.logic_dist`](logic_dist.md) — companion distributed logic programming module.
- [Documentation hub](../Index.md)
