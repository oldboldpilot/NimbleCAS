// NimbleCAS distributed graph search — shortest paths and tabu search as task graphs over SGEE.
// @author Olumuyiwa Oluwasanmi
//
// `nimblecas.search` fans A*, Dijkstra and tabu search across THREADS. This module fans the same
// searches across PROCESSES, by expressing each round as a `TaskGraph` of `nimblecas.taskdag`
// tasks that any `Executor` can run: the serial reference, the local parallel one, or
// `nimblecas.taskdag_sgee`'s distributed executor over a broker.
//
// WHY THE GRAPH HAS TO BE DATA.
//
// The in-process search API takes the graph as CALLBACKS — `SuccessorFn`, `CostFn`,
// `HeuristicFn`. A `std::function` cannot be serialised, and a `TaskFn` receives only bytes, so a
// callback-defined graph cannot cross a process boundary at all. Distributing search therefore
// requires the graph itself to have a wire representation, and that is what `WireGraph` is: a
// finite directed graph over dense node ids with explicit edge costs and an explicit per-node
// heuristic. This is not a limitation invented here; it is the shape a stored graph already has
// once it is loaded from a file, a database or a message.
//
// `successors_of` / `cost_of` / `heuristic_of` turn a `WireGraph` back into the callbacks the
// in-process algorithms take, so the SAME graph can drive both the distributed search and the
// serial oracle it must agree with. That is what makes the equivalence claims below testable
// rather than aspirational.
//
// HOW A ROUND IS DISTRIBUTED.
//
// Nodes are partitioned into contiguous SLICES, one per shard. A round hands each shard its own
// slice's adjacency together with the frontier nodes that fall in that slice, and the shard
// relaxes every out-edge of those nodes, returning PROPOSALS — (target, tentative distance,
// parent) triples. The coordinator merges the proposals, keeps the improvements, and the nodes
// that improved become the next round's frontier. The run ends when a round proposes no
// improvement.
//
// Each round is a fresh, static `TaskGraph` with no dependencies between its tasks — shards
// relax disjoint node sets, so they cannot interfere. This is the coordinator-loop-over-static-
// graphs shape that `nimblecas.logic_dist` describes as future work for SLD resolution; here it
// is the natural fit, because a search frontier is DATA that the coordinator already holds,
// whereas a suspended derivation is a continuation that would have to be captured first.
//
// WHY ROUND-BASED RELAXATION AND NOT DISTRIBUTED DIJKSTRA.
//
// Dijkstra settles one minimum-key node at a time; that serialisation is the whole algorithm and
// there is nothing in it to distribute. Relaxing by rounds — every node whose distance improved
// last round relaxes its out-edges this round — computes the SAME distances for non-negative
// edge weights, because a shortest path of k edges is fully relaxed after k rounds. It trades a
// larger total number of edge relaxations for rounds that are embarrassingly parallel, which is
// the right trade exactly when the graph is large enough to be worth distributing and the wrong
// one when it is not. `distributed_sssp` on a two-node graph is slower than `dijkstra`, and that
// is expected rather than a defect.
//
// WHAT IS GUARANTEED, AND WHAT IS NOT.
//
//   * COSTS ARE EXACT. `distributed_sssp` returns the same distances as `dijkstra`, and
//     `distributed_shortest_path` and `distributed_a_star` return the same total cost as
//     `dijkstra` and `a_star`. This is a provable consequence of round-based relaxation over
//     non-negative weights, not a hope, and the tests assert it against those functions directly.
//   * PATHS ARE EQUAL WHERE THE OPTIMUM IS UNIQUE. When several optimal paths exist, this module
//     breaks ties toward the LOWEST parent node id, which is the same rule `dijkstra` applies at
//     relaxation time — but the serial search visits nodes in settle order and this one visits
//     them in rounds, so the two can select different optimal parents. Rather than claim an
//     equivalence that has not been proved, the guarantee stated is cost equality always and path
//     equality when the optimal path is unique.
//   * THE HEURISTIC ONLY PRUNES. `distributed_a_star` uses an admissible heuristic to discard
//     proposals that cannot beat the best goal distance found so far. Pruning never changes the
//     goal's distance, but it does leave OTHER nodes' distances non-final, which is why
//     `distributed_a_star` returns a path and a cost and never a distance table.
//
// COST OF THE WIRE. Each round re-ships every shard's slice of the adjacency, so a run of R
// rounds moves roughly R times the graph. A locality-aware executor that kept a shard's slice
// resident between rounds would remove that, and doing so needs two additions to the SGEE C ABI
// rather than any change here. Until then the honest characterisation is that this module pays
// bandwidth to buy parallelism, and is worth it when expanding a node costs more than shipping
// its edges.

export module nimblecas.search_dist;

import std;
import nimblecas.core;
import nimblecas.search;
import nimblecas.taskdag;

export namespace nimblecas::search_dist {

// The registered operations. Versioned, because a wire format is part of the contract between a
// coordinator and a worker that may be running a different build: bumping the version is how an
// incompatible change announces itself instead of being discovered as a mis-parsed payload.
inline constexpr std::string_view relax_op_id = "nimblecas.search.relax_shard/v1";
inline constexpr std::string_view tabu_op_id = "nimblecas.search.tabu_run/v1";
// One depth-limited probe of a single root branch, for iterative deepening.
inline constexpr std::string_view dls_op_id = "nimblecas.search.depth_limited/v1";
// One pivot round's update of one row block, for Floyd-Warshall.
inline constexpr std::string_view fw_op_id = "nimblecas.search.floyd_block/v1";
// Triangles closed at the nodes of one shard.
inline constexpr std::string_view tri_op_id = "nimblecas.search.triangle_shard/v1";
// The cheapest edge leaving each component a shard touches, for Boruvka.
inline constexpr std::string_view mst_op_id = "nimblecas.search.boruvka_shard/v1";

// How a round combines a path value with an edge, as a SEMIRING -- which is what these searches
// actually have in common once the arithmetic is named honestly.
//
//   min_plus: value = sum of edge costs, best = smallest. Shortest paths.
//   max_min:  value = smallest edge on the route, best = largest. Widest (bottleneck) paths.
//
// The round loop, the sharding, the frontier and the merge are identical for both; only the two
// operations change. Writing it once and parameterising by semiring is not generality for its own
// sake -- it is why widest paths needed no new round machinery, and why a bug fixed in the loop is
// fixed for every search built on it.
enum class Semiring : std::uint8_t { min_plus, max_min };

// One directed, weighted edge. `cost` must be non-negative.
struct Edge {
    std::int64_t target{0};
    std::int64_t cost{0};

    [[nodiscard]] auto operator==(const Edge&) const noexcept -> bool = default;
};

// A finite directed graph over dense node ids 0 .. adjacency.size()-1.
//
// `heuristic` is either empty (meaning uniformly zero, so A* degenerates to Dijkstra) or exactly
// one non-negative value per node. `landscape` is the objective value used by the tabu search
// entry points and is likewise either empty or one value per node; it is separate from
// `heuristic` because a search landscape and a distance estimate are different quantities that
// happen to share a shape.
struct WireGraph {
    std::vector<std::vector<Edge>> adjacency;
    std::vector<std::int64_t> heuristic;
    std::vector<std::int64_t> landscape;

    [[nodiscard]] auto node_count() const noexcept -> std::size_t { return adjacency.size(); }
};

// ---------------------------------------------------------------------------
// Validation and adaptation.
// ---------------------------------------------------------------------------

// Checks the invariants every entry point relies on: every edge target in range, every edge cost
// non-negative, and `heuristic` / `landscape` either empty or one entry per node with no negative
// heuristic value. `domain_error` names the first violation found.
//
// Checking HERE means a malformed graph is refused by the coordinator rather than discovered by a
// worker that can only fail on arrival.
[[nodiscard]] auto validate(const WireGraph& g) -> Result<void>;

// The in-process callbacks for this graph, so the serial algorithms in `nimblecas.search` can be
// run over the very same graph as the distributed ones. A successor query for a node outside the
// graph yields no successors; a cost query for an edge that does not exist yields 0.
[[nodiscard]] auto successors_of(const WireGraph& g) -> SuccessorFn;
[[nodiscard]] auto cost_of(const WireGraph& g) -> CostFn;
[[nodiscard]] auto heuristic_of(const WireGraph& g) -> HeuristicFn;

// ---------------------------------------------------------------------------
// Wire format.
// ---------------------------------------------------------------------------
//
// Little-endian fixed-width integers throughout, matching the convention `nimblecas.taskdag_sgee`
// already uses for envelope framing. Signed values are carried as their two's-complement bit
// pattern, which C++20 onward defines exactly, so INT64_MIN survives a round trip unchanged.

// Encodes the slice of `g` covering nodes [first, first + count), together with the graph's node
// count and the heuristic values for that slice. `domain_error` if the slice is out of range.
[[nodiscard]] auto encode_slice(const WireGraph& g, std::size_t first, std::size_t count)
    -> Result<Payload>;

// One node of a frontier: the node and the distance it was reached at.
struct FrontierEntry {
    std::int64_t node{0};
    std::int64_t distance{0};

    [[nodiscard]] auto operator==(const FrontierEntry&) const noexcept -> bool = default;
};

// Encodes a frontier together with the pruning bound. `bound` is the best goal distance known so
// far, or INT64_MAX for "no bound yet"; a shard discards a proposal whose distance plus the
// target's heuristic exceeds it.
[[nodiscard]] auto encode_frontier(std::span<const FrontierEntry> frontier, std::int64_t bound)
    -> Result<Payload>;

// One relaxation result: reaching `target` at `distance` by way of `parent`.
struct Proposal {
    std::int64_t target{0};
    std::int64_t distance{0};
    std::int64_t parent{0};

    [[nodiscard]] auto operator==(const Proposal&) const noexcept -> bool = default;
};

[[nodiscard]] auto decode_proposals(std::span<const std::byte> bytes) -> Result<std::vector<Proposal>>;

// ---------------------------------------------------------------------------
// Running.
// ---------------------------------------------------------------------------

// Registers both operations. BOTH the coordinator and every worker must call this on their own
// registry at startup: the distributed executor ships an op id and arguments, never code, so a
// worker that has not registered an op cannot run the task it is handed.
[[nodiscard]] auto register_ops(TaskRegistry& reg) -> Result<void>;

// Builds one round's task graph: one task per shard, each carrying its slice and its share of the
// frontier as bound literals, with no dependencies between them.
//
// Exposed because a round is the unit a scheduler or a profiler wants to see. `domain_error` if
// `shard_count` is zero or the graph is malformed.
[[nodiscard]] auto build_round_graph(const TaskRegistry& reg, const WireGraph& g,
                                     std::span<const FrontierEntry> frontier, std::int64_t bound,
                                     std::size_t shard_count) -> Result<TaskGraph>;

// Single-source shortest distances from `start`, computed by distributed round-based relaxation.
//
// Returns one entry per node: the exact distance, or `std::nullopt` for a node unreachable from
// `start`. These are the SAME distances `dijkstra` computes.
//
// `domain_error` for a malformed graph, a `start` out of range or a zero `shard_count`;
// `overflow` if a path cost exceeds std::int64_t; `not_converged` if the round budget (one round
// per node, which a correct run can never exhaust) is used up — a guard that says so honestly
// rather than assuming the bound holds.
[[nodiscard]] auto distributed_sssp(const WireGraph& g, std::int64_t start,
                                    std::size_t shard_count, Executor& exec)
    -> Result<std::vector<std::optional<std::int64_t>>>;

// Minimum-cost path from `start` to `goal_node` and its total cost, both endpoints included, and
// just `{start}` with cost 0 when they are the same node.
//
// The cost equals `dijkstra`'s cost exactly. The path equals `dijkstra`'s path when the optimal
// path is unique; where several optima exist, ties break toward the lowest parent node id.
// `undefined_value` when `goal_node` is not reachable.
[[nodiscard]] auto distributed_shortest_path(const WireGraph& g, std::int64_t start,
                                             std::int64_t goal_node, std::size_t shard_count,
                                             Executor& exec)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>>;

// As `distributed_shortest_path`, but the graph's heuristic prunes relaxations that cannot beat
// the best goal distance found so far. The heuristic must be admissible — never overestimating
// the true remaining distance — which `validate` cannot check and the caller must guarantee; an
// inadmissible heuristic makes the result wrong rather than merely slow.
//
// Returns the same cost as `a_star` over the same graph. No distance table is returned, because
// pruning leaves non-goal distances non-final.
[[nodiscard]] auto distributed_a_star(const WireGraph& g, std::int64_t start,
                                      std::int64_t goal_node, std::size_t shard_count,
                                      Executor& exec)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>>;

// Breadth-first search from `start` to the lowest-numbered goal in `goals`, level by level.
//
// This is the round loop with every edge weighted 1, so the round index IS the BFS level and no
// separate mechanism is needed. The returned path has the FEWEST EDGES, which is the property
// `bfs` guarantees; where several shortest-hop paths exist, ties break toward the lowest parent
// node id, so the path matches `bfs`'s where the shallowest path is unique.
//
// `undefined_value` when no goal is reachable; `domain_error` for an out-of-range node or a zero
// `shard_count`.
[[nodiscard]] auto distributed_bfs(const WireGraph& g, std::int64_t start,
                                   std::span<const std::int64_t> goals, std::size_t shard_count,
                                   Executor& exec) -> Result<std::vector<std::int64_t>>;

// Every node reachable from `start`, in ascending node order.
//
// THIS IS THE DISTRIBUTED FORM OF DEPTH-FIRST SEARCH, and the name says what is honestly on
// offer. DFS is defined by its VISIT ORDER: the leftmost branch is explored to exhaustion before
// its sibling, and that order is a strict sequence with no independent pieces to hand out. A
// worker cannot be given "the next node in depth-first order" without first being told the answer
// to the whole search. Parallel DFS in the literature relies on WORK STEALING, where an idle
// worker takes an unexplored branch from a busy one -- which needs a worker to enqueue work it
// discovers, and a `TaskFn` has no broker handle to do it with (the same limit
// `nimblecas.logic_dist` documents).
//
// What survives distribution is DFS's ANSWER rather than its route: the set of reachable nodes is
// the same whichever order they are visited in. That set is what this computes, by frontier
// rounds, and it equals the set of nodes `dfs_iterative` would visit given an unbounded depth.
// Offering it under a name that promises reachability rather than depth-first order is the honest
// way to ship this; a `distributed_dfs` returning nodes in some other order would be a different
// algorithm wearing DFS's name.
[[nodiscard]] auto distributed_reachable_set(const WireGraph& g, std::int64_t start,
                                             std::size_t shard_count, Executor& exec)
    -> Result<std::vector<std::int64_t>>;

// Iterative-deepening search for the goal reachable in the FEWEST EDGES, distributed by fanning
// each depth limit's ROOT BRANCHES across tasks: one task per successor of `start`, each running
// a bounded depth-first probe of its own subtree.
//
// The decomposition is the one `nimblecas.logic_dist` uses for OR-branches, and it inherits the
// same ONE-LEVEL FAN-OUT: parallelism is bounded by the root's out-degree, however deep the
// search below it runs. A root with a single successor distributes to exactly one worker.
//
// Returns the same HOP COUNT as `iterative_deepening_dfs` over the same graph, which is the
// property that algorithm guarantees. `not_converged` when `max_depth` is reached without a goal
// but deeper nodes remain; `undefined_value` when the reachable graph is exhausted first.
[[nodiscard]] auto distributed_iterative_deepening(const WireGraph& g, std::int64_t start,
                                                   std::span<const std::int64_t> goals,
                                                   std::int64_t max_depth,
                                                   std::size_t shard_count, Executor& exec)
    -> Result<std::vector<std::int64_t>>;

// Weighted A*: the graph's heuristic is scaled by the exact rational `weight_num / weight_den`
// before it prunes, so no floating point enters the decision.
//
// BOUNDED SUBOPTIMAL, and the bound is the only promise: the returned cost is at most w times the
// optimum. At w = 1 it is `distributed_a_star` and therefore optimal. A zero denominator, or a
// ratio below 1, is a `domain_error` -- a weight below 1 would prune paths that are actually
// better, which is not a speed/quality trade but a wrong answer.
[[nodiscard]] auto distributed_weighted_a_star(const WireGraph& g, std::int64_t start,
                                               std::int64_t goal_node, std::int64_t weight_num,
                                               std::int64_t weight_den, std::size_t shard_count,
                                               Executor& exec)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>>;

// All-pairs shortest distances by a distributed Floyd-Warshall.
//
// Row-block partitioned in the classical way: the algorithm's k-th iteration updates every entry
// from row k and column k alone, so with the matrix split into row blocks each block can be
// updated INDEPENDENTLY once row k is known. Every pivot k is therefore one round of a static
// task graph -- n rounds of `shard_count` independent tasks -- and this is the one algorithm here
// whose textbook form is already a wavefront rather than something reshaped to become one.
//
// `distances[i][j]` is the exact shortest distance, or `std::nullopt` when j is unreachable from
// i. `distances[i][i]` is 0 even where no self-loop exists. The values equal what `dijkstra`
// returns from each source, which is what the tests assert.
//
// `overflow` if any path cost exceeds std::int64_t; `domain_error` for a malformed graph or a
// zero `shard_count`.
[[nodiscard]] auto distributed_floyd_warshall(const WireGraph& g, std::size_t shard_count,
                                              Executor& exec)
    -> Result<std::vector<std::vector<std::optional<std::int64_t>>>>;

// ---------------------------------------------------------------------------
// Connectivity, ordering, structure.
// ---------------------------------------------------------------------------

// Connected components of the UNDIRECTED view, as one label per node: the smallest node id in
// that node's component. Two nodes are in the same component exactly when their labels match.
//
// This is min-label propagation, which is the standard parallel connectivity algorithm and needs
// no machinery of its own here: over a zero-weighted undirected graph, seeding every node with its
// own id and running the min_plus rounds propagates the smallest id through each component. Rounds
// are bounded by the graph's diameter.
//
// `domain_error` for a malformed graph or a zero `shard_count`.
[[nodiscard]] auto distributed_connected_components(const WireGraph& g, std::size_t shard_count,
                                                    Executor& exec)
    -> Result<std::vector<std::int64_t>>;

// Strongly connected components of the DIRECTED graph, as one label per node: the smallest node id
// in that node's component.
//
// u and v share a component exactly when each reaches the other, so this is computed from the
// all-pairs reachability `distributed_floyd_warshall` already produces. That is honest about what
// it costs: QUADRATIC in the number of nodes, in both memory and wire traffic, where a
// forward-backward decomposition would be near-linear on sparse graphs. The trade is deliberate --
// it reuses an algorithm that is already distributed and already tested rather than adding a
// second, subtler one -- but on a large sparse graph it is the wrong tool, and a caller sizing a
// run should know that before measuring it.
[[nodiscard]] auto distributed_strongly_connected_components(const WireGraph& g,
                                                             std::size_t shard_count,
                                                             Executor& exec)
    -> Result<std::vector<std::int64_t>>;

// A topological order of the directed graph: every edge runs from an earlier node to a later one.
//
// Kahn's algorithm by LEVELS, which is its naturally parallel form -- every node whose remaining
// in-degree has reached zero can be emitted at the same time, and one round per level counts the
// edges leaving that level. Within a level nodes are emitted in ascending id order, so the result
// is one specific topological order rather than an arbitrary one.
//
// `undefined_value` when the graph has a cycle, because then no topological order exists. That is
// a different thing from an error in the input, and a partial order covering only the acyclic part
// would be a plausible-looking wrong answer.
[[nodiscard]] auto distributed_topological_order(const WireGraph& g, std::size_t shard_count,
                                                 Executor& exec)
    -> Result<std::vector<std::int64_t>>;

// The k-core: the largest subgraph of the UNDIRECTED view in which every node has degree at least
// `k`, returned as its nodes in ascending order.
//
// Computed by peeling: every node whose degree has fallen below k is removed, its neighbours lose
// a degree, and the removal cascades. Each peeling wave is one round. An empty result is a real
// answer -- many graphs have no k-core at all for large k -- and not an error.
//
// Degrees are counted on the SIMPLE undirected view: parallel edges count once and self-loops not
// at all, which is the usual definition and the one that makes "degree at least k" mean what a
// reader expects.
//
// `domain_error` for a negative `k`, a malformed graph, or a zero `shard_count`.
[[nodiscard]] auto distributed_k_core(const WireGraph& g, std::int64_t k, std::size_t shard_count,
                                      Executor& exec) -> Result<std::vector<std::int64_t>>;

// The number of triangles through each node of the UNDIRECTED view.
//
// Every triangle is counted once at each of its three corners, which is the per-node definition
// clustering coefficients are built on; the total number of distinct triangles is the sum divided
// by three. Self-loops and parallel edges close nothing and are ignored.
//
// Each shard counts only at its own nodes but receives the whole graph, because closing a triangle
// at u needs the neighbourhoods of u's neighbours and those live anywhere.
[[nodiscard]] auto distributed_triangle_counts(const WireGraph& g, std::size_t shard_count,
                                               Executor& exec)
    -> Result<std::vector<std::int64_t>>;

// The widest (bottleneck) path from `start` to `goal_node`: the route whose NARROWEST edge is as
// wide as possible, returned with that width.
//
// This is the same round loop under the (max, min) semiring instead of (min, +) -- the value of a
// route is its smallest edge rather than its total, and the best route is the largest such value.
// It needed no new round machinery, which is the payoff for naming the semiring.
//
// `undefined_value` when `goal_node` is unreachable. A path from a node to itself is `{start}`
// with width INT64_MAX, since an empty route has no narrow edge to limit it.
[[nodiscard]] auto distributed_widest_path(const WireGraph& g, std::int64_t start,
                                           std::int64_t goal_node, std::size_t shard_count,
                                           Executor& exec)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>>;

// One edge of a minimum spanning forest.
struct MstEdge {
    std::int64_t u{0};
    std::int64_t v{0};
    std::int64_t weight{0};

    [[nodiscard]] auto operator==(const MstEdge&) const noexcept -> bool = default;
};

// A minimum spanning forest of the UNDIRECTED view, as its edges in ascending (u, v) order
// together with the total weight.
//
// Boruvka's algorithm, which is the one classical MST algorithm that is parallel by construction:
// every component finds its own cheapest outgoing edge simultaneously, all those edges are added,
// and the components merge. The number of components at least halves each round, so the whole run
// is a logarithmic number of rounds rather than one round per edge.
//
// A FOREST, not a tree: a disconnected graph has no spanning tree, and returning one component's
// tree while silently dropping the rest would be worse than saying so. The result spans each
// component separately, and `edges.size()` equals the node count minus the number of components.
//
// Ties are broken by the total order (weight, u, v). That is not cosmetic -- Boruvka can close a
// cycle when two components each pick a different edge of equal weight between them, and a strict
// total order on edges is what rules that out. It also makes the forest reproducible.
//
// `overflow` if the total weight exceeds std::int64_t.
[[nodiscard]] auto distributed_minimum_spanning_forest(const WireGraph& g, std::size_t shard_count,
                                                       Executor& exec)
    -> Result<std::pair<std::vector<MstEdge>, std::int64_t>>;

// Multi-start tabu search over the graph's `landscape`, one independent run per start, each
// distributed as its own task. A state is a single node; its neighbours are that node's
// successors; its objective is `landscape[node]`.
//
// Returns the best state and objective across the runs, reduced by the same deterministic rule
// `parallel_multistart_tabu` uses — lowest objective wins, ties broken by lexicographically
// smaller state — so the two agree exactly on the same inputs.
//
// `domain_error` for an empty `starts`, a start out of range, a negative tenure or iteration
// count, or a graph with no landscape.
[[nodiscard]] auto distributed_multistart_tabu(const WireGraph& g,
                                               std::span<const std::int64_t> starts,
                                               std::int64_t tabu_tenure, std::int64_t max_iters,
                                               Executor& exec)
    -> Result<std::pair<TabuState, std::int64_t>>;

}  // namespace nimblecas::search_dist

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::search_dist {

namespace {

constexpr std::uint64_t slice_magic = 0x4e43535f534c4331ULL;     // "NCS_SLC1"
constexpr std::uint64_t frontier_magic = 0x4e43535f46524e31ULL;  // "NCS_FRN1"
constexpr std::uint64_t proposal_magic = 0x4e43535f50525031ULL;  // "NCS_PRP1"
constexpr std::uint64_t tabu_magic = 0x4e43535f54425531ULL;      // "NCS_TBU1"
constexpr std::uint64_t dls_magic = 0x4e43535f444c5331ULL;       // "NCS_DLS1"
constexpr std::uint64_t fw_magic = 0x4e43535f46574b31ULL;        // "NCS_FWK1"
constexpr std::uint64_t tri_magic = 0x4e43535f54524931ULL;       // "NCS_TRI1"
constexpr std::uint64_t mst_magic = 0x4e43535f4d535431ULL;       // "NCS_MST1"

// Distances are never negative, so a negative slot is free to mean "no path" in the
// Floyd-Warshall matrix on the wire. That is cheaper and less error-prone than a parallel array
// of presence flags, and it cannot collide with a real value.
constexpr std::int64_t fw_absent = -1;

// ---------------------------------------------------------------------------
// Little-endian scalar framing.
// ---------------------------------------------------------------------------

auto put_u64(std::vector<std::byte>& out, std::uint64_t v) -> void {
    for (const int shift : std::views::iota(0, 8) | std::views::transform([](int i) { return 8 * i; })) {
        out.push_back(static_cast<std::byte>((v >> shift) & 0xffULL));
    }
}

auto put_i64(std::vector<std::byte>& out, std::int64_t v) -> void {
    put_u64(out, static_cast<std::uint64_t>(v));
}

// Reads a u64 at `offset`, advancing it. Returns nullopt when the span is too short, which is how
// every truncated payload is caught rather than read past the end.
[[nodiscard]] auto take_u64(std::span<const std::byte> bytes, std::size_t& offset)
    -> std::optional<std::uint64_t> {
    if (offset + 8 > bytes.size()) {
        return std::nullopt;
    }
    std::uint64_t v = 0;
    for (const int i : std::views::iota(0, 8)) {
        const auto byte = std::to_integer<unsigned char>(bytes[offset + static_cast<std::size_t>(i)]);
        v |= static_cast<std::uint64_t>(byte) << (8 * i);
    }
    offset += 8;
    return v;
}

[[nodiscard]] auto take_i64(std::span<const std::byte> bytes, std::size_t& offset)
    -> std::optional<std::int64_t> {
    const auto v = take_u64(bytes, offset);
    if (!v.has_value()) {
        return std::nullopt;
    }
    return static_cast<std::int64_t>(*v);
}

// Checked addition of a path cost. `true` on overflow, matching the convention in
// nimblecas.search so the two modules read the same way.
[[nodiscard]] auto add_overflows(std::int64_t a, std::int64_t b, std::int64_t& out) noexcept -> bool {
    return __builtin_add_overflow(a, b, &out);
}

// A size that must fit in std::size_t once it comes off the wire. A corrupt length field is a
// syntax error, never an allocation attempt.
[[nodiscard]] auto count_is_sane(std::uint64_t n, std::size_t remaining_bytes,
                                 std::size_t bytes_per_item) -> bool {
    if (bytes_per_item == 0) {
        return false;
    }
    return n <= remaining_bytes / bytes_per_item;
}

// ---------------------------------------------------------------------------
// Slice payloads.
// ---------------------------------------------------------------------------

// A decoded slice: the adjacency of nodes [first, first + adjacency.size()), plus the graph-wide
// node count and this slice's heuristic values.
struct Slice {
    std::uint64_t node_count{0};
    std::uint64_t first{0};
    std::vector<std::vector<Edge>> adjacency;
    std::vector<std::int64_t> heuristic;  // empty == uniformly zero
};

[[nodiscard]] auto decode_slice(std::span<const std::byte> bytes) -> Result<Slice> {
    std::size_t off = 0;
    const auto magic = take_u64(bytes, off);
    if (!magic.has_value() || *magic != slice_magic) {
        return make_error<Slice>(MathError::syntax_error);
    }
    const auto total = take_u64(bytes, off);
    const auto first = take_u64(bytes, off);
    const auto count = take_u64(bytes, off);
    if (!total.has_value() || !first.has_value() || !count.has_value()) {
        return make_error<Slice>(MathError::syntax_error);
    }
    // Each node contributes at least its degree field.
    if (!count_is_sane(*count, bytes.size() - off, 8)) {
        return make_error<Slice>(MathError::syntax_error);
    }

    Slice s;
    s.node_count = *total;
    s.first = *first;
    s.adjacency.resize(static_cast<std::size_t>(*count));
    for (std::vector<Edge>& row : s.adjacency) {
        const auto deg = take_u64(bytes, off);
        if (!deg.has_value() || !count_is_sane(*deg, bytes.size() - off, 16)) {
            return make_error<Slice>(MathError::syntax_error);
        }
        row.resize(static_cast<std::size_t>(*deg));
        for (Edge& e : row) {
            const auto tgt = take_i64(bytes, off);
            const auto cst = take_i64(bytes, off);
            if (!tgt.has_value() || !cst.has_value()) {
                return make_error<Slice>(MathError::syntax_error);
            }
            e = Edge{.target = *tgt, .cost = *cst};
        }
    }
    const auto hlen = take_u64(bytes, off);
    if (!hlen.has_value() || !count_is_sane(*hlen, bytes.size() - off, 8)) {
        return make_error<Slice>(MathError::syntax_error);
    }
    s.heuristic.resize(static_cast<std::size_t>(*hlen));
    for (std::int64_t& slot : s.heuristic) {
        const auto h = take_i64(bytes, off);
        if (!h.has_value()) {
            return make_error<Slice>(MathError::syntax_error);
        }
        slot = *h;
    }
    if (off != bytes.size()) {
        return make_error<Slice>(MathError::syntax_error);
    }
    return s;
}

// ---------------------------------------------------------------------------
// Frontier payloads.
// ---------------------------------------------------------------------------

struct Frontier {
    std::int64_t bound{std::numeric_limits<std::int64_t>::max()};
    // The heuristic weight as an exact rational. 1/1 is plain A*; a larger ratio prunes harder
    // and buys speed for a bounded loss of optimality. No floating point ever enters the
    // comparison, so the same graph prunes identically on every machine.
    std::int64_t weight_num{1};
    std::int64_t weight_den{1};
    Semiring semiring{Semiring::min_plus};
    std::vector<FrontierEntry> entries;
};

[[nodiscard]] auto decode_frontier(std::span<const std::byte> bytes) -> Result<Frontier> {
    std::size_t off = 0;
    const auto magic = take_u64(bytes, off);
    if (!magic.has_value() || *magic != frontier_magic) {
        return make_error<Frontier>(MathError::syntax_error);
    }
    const auto bound = take_i64(bytes, off);
    const auto wnum = take_i64(bytes, off);
    const auto wden = take_i64(bytes, off);
    const auto ring = take_u64(bytes, off);
    const auto n = take_u64(bytes, off);
    if (!bound.has_value() || !wnum.has_value() || !wden.has_value() || !ring.has_value() ||
        !n.has_value() || *ring > 1 || !count_is_sane(*n, bytes.size() - off, 16)) {
        return make_error<Frontier>(MathError::syntax_error);
    }
    if (*wden <= 0 || *wnum < *wden) {
        return make_error<Frontier>(MathError::syntax_error);
    }
    Frontier f;
    f.bound = *bound;
    f.weight_num = *wnum;
    f.weight_den = *wden;
    f.semiring = *ring == 0 ? Semiring::min_plus : Semiring::max_min;
    f.entries.resize(static_cast<std::size_t>(*n));
    for (FrontierEntry& fe : f.entries) {
        const auto node = take_i64(bytes, off);
        const auto dist = take_i64(bytes, off);
        if (!node.has_value() || !dist.has_value()) {
            return make_error<Frontier>(MathError::syntax_error);
        }
        fe = FrontierEntry{.node = *node, .distance = *dist};
    }
    if (off != bytes.size()) {
        return make_error<Frontier>(MathError::syntax_error);
    }
    return f;
}

[[nodiscard]] auto encode_proposals(std::span<const Proposal> ps) -> Payload {
    Payload out;
    out.reserve(16 + ps.size() * 24);
    put_u64(out, proposal_magic);
    put_u64(out, static_cast<std::uint64_t>(ps.size()));
    for (const Proposal& p : ps) {
        put_i64(out, p.target);
        put_i64(out, p.distance);
        put_i64(out, p.parent);
    }
    return out;
}

// ---------------------------------------------------------------------------
// The relax operation — this is what a worker runs.
// ---------------------------------------------------------------------------

// Relaxes every out-edge of every frontier node that falls in this shard's slice.
//
// Nodes outside the slice are IGNORED rather than rejected: the coordinator partitions the
// frontier by slice, so a stray node means a coordinator bug, and silently dropping it here would
// hide that. It is dropped anyway — the shard has no adjacency for it and inventing one would be
// worse — but the partitioning is asserted coordinator-side where it can be tested.
[[nodiscard]] auto relax_shard(std::span<const Payload> args) -> Result<Payload> {
    if (args.size() != 2) {
        return make_error<Payload>(MathError::domain_error);
    }
    auto slice = decode_slice(args[0]);
    if (!slice) {
        return make_error<Payload>(slice.error());
    }
    auto frontier = decode_frontier(args[1]);
    if (!frontier) {
        return make_error<Payload>(frontier.error());
    }

    const std::uint64_t first = slice->first;
    const std::uint64_t last = first + static_cast<std::uint64_t>(slice->adjacency.size());

    std::vector<Proposal> proposals;
    for (const FrontierEntry& fe : frontier->entries) {
        if (fe.node < 0) {
            continue;
        }
        const auto u = static_cast<std::uint64_t>(fe.node);
        if (u < first || u >= last) {
            continue;
        }
        const std::vector<Edge>& edges = slice->adjacency[static_cast<std::size_t>(u - first)];
        for (const Edge& e : edges) {
            std::int64_t nd = 0;
            if (frontier->semiring == Semiring::min_plus) {
                if (add_overflows(fe.distance, e.cost, nd)) {
                    return make_error<Payload>(MathError::overflow);
                }
            } else {
                // max_min: the value of a route is its NARROWEST edge, so extending a route can
                // only ever narrow it. Nothing is summed, so nothing can overflow.
                nd = std::min(fe.distance, e.cost);
            }
            // Admissible-heuristic pruning. `bound` is the best goal distance the coordinator has
            // seen; a proposal that cannot beat it cannot lie on a better goal path.
            if (frontier->bound != std::numeric_limits<std::int64_t>::max()) {
                std::int64_t h = 0;
                if (e.target >= 0) {
                    const auto t = static_cast<std::uint64_t>(e.target);
                    if (!slice->heuristic.empty() && t >= first && t < last) {
                        h = slice->heuristic[static_cast<std::size_t>(t - first)];
                    }
                }
                // Compared in SCALED integers: den*(g + w*h) > den*bound becomes
                // den*g + num*h > den*bound, with no division and no floating point, so the
                // pruning decision is exact and identical on every machine.
                //
                // If any of those products overflows, the proposal is KEPT. Pruning is an
                // optimisation and never pruning is always correct, so an arithmetic edge case
                // costs time rather than answers -- the one direction in which it is safe to be
                // wrong.
                std::int64_t lhs = 0;
                std::int64_t rhs = 0;
                std::int64_t term_g = 0;
                std::int64_t term_h = 0;
                const bool exact =
                    !__builtin_mul_overflow(frontier->weight_den, nd, &term_g) &&
                    !__builtin_mul_overflow(frontier->weight_num, h, &term_h) &&
                    !add_overflows(term_g, term_h, lhs) &&
                    !__builtin_mul_overflow(frontier->weight_den, frontier->bound, &rhs);
                if (exact && lhs > rhs) {
                    continue;
                }
            }
            proposals.push_back(
                Proposal{.target = e.target, .distance = nd, .parent = fe.node});
        }
    }
    return encode_proposals(proposals);
}

// ---------------------------------------------------------------------------
// The tabu operation.
// ---------------------------------------------------------------------------

[[nodiscard]] auto decode_tabu_args(std::span<const std::byte> bytes)
    -> Result<std::tuple<std::int64_t, std::int64_t, std::int64_t>> {
    using Args = std::tuple<std::int64_t, std::int64_t, std::int64_t>;
    std::size_t off = 0;
    const auto magic = take_u64(bytes, off);
    if (!magic.has_value() || *magic != tabu_magic) {
        return make_error<Args>(MathError::syntax_error);
    }
    const auto start = take_i64(bytes, off);
    const auto tenure = take_i64(bytes, off);
    const auto iters = take_i64(bytes, off);
    if (!start.has_value() || !tenure.has_value() || !iters.has_value() || off != bytes.size()) {
        return make_error<Args>(MathError::syntax_error);
    }
    return Args{*start, *tenure, *iters};
}

// Runs one tabu search over the slice's landscape. The slice for a tabu task is the WHOLE graph:
// a local search walks wherever the landscape leads it, so restricting it to a node range would
// change the answer rather than merely distribute it.
[[nodiscard]] auto tabu_run(std::span<const Payload> args) -> Result<Payload> {
    if (args.size() != 2) {
        return make_error<Payload>(MathError::domain_error);
    }
    auto slice = decode_slice(args[0]);
    if (!slice) {
        return make_error<Payload>(slice.error());
    }
    auto parsed = decode_tabu_args(args[1]);
    if (!parsed) {
        return make_error<Payload>(parsed.error());
    }
    const auto [start, tenure, iters] = *parsed;

    // The landscape rides in the slice's heuristic field: one value per node, and the tabu entry
    // point refuses a graph without one, so an empty vector here is a corrupt payload.
    if (slice->heuristic.size() != slice->adjacency.size()) {
        return make_error<Payload>(MathError::syntax_error);
    }
    const std::vector<std::vector<Edge>>& adj = slice->adjacency;
    const std::vector<std::int64_t>& land = slice->heuristic;

    const auto neighbours = [&adj](const TabuState& s) -> std::vector<TabuState> {
        std::vector<TabuState> out;
        if (s.size() != 1 || s[0] < 0) {
            return out;
        }
        const auto u = static_cast<std::size_t>(s[0]);
        if (u >= adj.size()) {
            return out;
        }
        out.reserve(adj[u].size());
        for (const Edge& e : adj[u]) {
            out.push_back(TabuState{e.target});
        }
        return out;
    };
    const auto objective = [&land](const TabuState& s) -> std::int64_t {
        if (s.size() != 1 || s[0] < 0) {
            return std::numeric_limits<std::int64_t>::max();
        }
        const auto u = static_cast<std::size_t>(s[0]);
        if (u >= land.size()) {
            return std::numeric_limits<std::int64_t>::max();
        }
        return land[u];
    };

    auto r = tabu_search(TabuState{start}, neighbours, objective, tenure, iters);
    if (!r) {
        return make_error<Payload>(r.error());
    }
    Payload out;
    put_u64(out, tabu_magic);
    put_u64(out, static_cast<std::uint64_t>(r->first.size()));
    for (const std::int64_t v : r->first) {
        put_i64(out, v);
    }
    put_i64(out, r->second);
    return out;
}

[[nodiscard]] auto decode_tabu_result(std::span<const std::byte> bytes)
    -> Result<std::pair<TabuState, std::int64_t>> {
    using Best = std::pair<TabuState, std::int64_t>;
    std::size_t off = 0;
    const auto magic = take_u64(bytes, off);
    if (!magic.has_value() || *magic != tabu_magic) {
        return make_error<Best>(MathError::syntax_error);
    }
    const auto n = take_u64(bytes, off);
    if (!n.has_value() || !count_is_sane(*n, bytes.size() - off, 8)) {
        return make_error<Best>(MathError::syntax_error);
    }
    TabuState state(static_cast<std::size_t>(*n));
    for (std::int64_t& slot : state) {
        const auto v = take_i64(bytes, off);
        if (!v.has_value()) {
            return make_error<Best>(MathError::syntax_error);
        }
        slot = *v;
    }
    const auto val = take_i64(bytes, off);
    if (!val.has_value() || off != bytes.size()) {
        return make_error<Best>(MathError::syntax_error);
    }
    return Best{std::move(state), *val};
}

// ---------------------------------------------------------------------------
// Coordinator helpers.
// ---------------------------------------------------------------------------

// Slice boundaries: `shard_count` contiguous ranges over `n` nodes, the first `n % shard_count`
// of them one node longer. Returns start offsets with a trailing sentinel, so slice i is
// [bounds[i], bounds[i+1]).
[[nodiscard]] auto slice_bounds(std::size_t n, std::size_t shard_count) -> std::vector<std::size_t> {
    std::vector<std::size_t> bounds;
    bounds.reserve(shard_count + 1);
    const std::size_t base = n / shard_count;
    const std::size_t extra = n % shard_count;
    std::size_t at = 0;
    for (const std::size_t i : std::views::iota(std::size_t{0}, shard_count)) {
        bounds.push_back(at);
        at += base + (i < extra ? 1 : 0);
    }
    bounds.push_back(at);
    return bounds;
}

// Reconstructs start -> goal from a parent table. `parent[goal]` must be set unless goal == start.
[[nodiscard]] auto rebuild_path(const std::vector<std::optional<std::int64_t>>& parent,
                                std::int64_t start, std::int64_t goal)
    -> Result<std::vector<std::int64_t>> {
    std::vector<std::int64_t> rev;
    std::int64_t at = goal;
    // The parent chain is strictly decreasing in distance, so it cannot cycle; the node-count
    // bound is a guard against a corrupt table rather than an expected outcome.
    for ([[maybe_unused]] const std::size_t hop : std::views::iota(std::size_t{0}, parent.size() + 1)) {
        rev.push_back(at);
        if (at == start) {
            std::ranges::reverse(rev);
            return rev;
        }
        if (at < 0 || static_cast<std::size_t>(at) >= parent.size()) {
            return make_error<std::vector<std::int64_t>>(MathError::undefined_value);
        }
        const std::optional<std::int64_t>& p = parent[static_cast<std::size_t>(at)];
        if (!p.has_value()) {
            return make_error<std::vector<std::int64_t>>(MathError::undefined_value);
        }
        at = *p;
    }
    return make_error<std::vector<std::int64_t>>(MathError::not_converged);
}

// The shared round loop behind every shortest-path entry point.
//
// `goal` is the node whose distance bounds the A* pruning, or nullopt for a plain SSSP run in
// which nothing is pruned and every distance comes out final.
struct RoundOutcome {
    std::vector<std::optional<std::int64_t>> dist;
    std::vector<std::optional<std::int64_t>> parent;
};

[[nodiscard]] auto run_rounds(const TaskRegistry& reg, const WireGraph& g, std::int64_t start,
                              std::optional<std::int64_t> goal, std::int64_t weight_num,
                              std::int64_t weight_den, std::size_t shard_count, Executor& exec)
    -> Result<RoundOutcome>;

// Runs exactly ONE round and hands back the raw proposals.
//
// The peeling algorithms -- topological ordering and k-core -- are not relaxations at all: they
// COUNT how many edges arrive at each node from the current level. A proposal happens to be one
// per traversed edge, which is exactly that count, so they reuse the sharded round rather than
// pretending to be shortest-path searches.
[[nodiscard]] auto run_one_round(const TaskRegistry& reg, const WireGraph& g,
                                 std::span<const FrontierEntry> frontier, std::size_t shard_count,
                                 Executor& exec) -> Result<std::vector<Proposal>>;

}  // namespace

// ---------------------------------------------------------------------------
// Validation and adaptation.
// ---------------------------------------------------------------------------

auto validate(const WireGraph& g) -> Result<void> {
    const std::size_t n = g.adjacency.size();
    for (const std::vector<Edge>& edges : g.adjacency) {
        for (const Edge& e : edges) {
            if (e.cost < 0) {
                return make_error<void>(MathError::domain_error);
            }
            if (e.target < 0 || static_cast<std::size_t>(e.target) >= n) {
                return make_error<void>(MathError::domain_error);
            }
        }
    }
    if (!g.heuristic.empty()) {
        if (g.heuristic.size() != n) {
            return make_error<void>(MathError::domain_error);
        }
        for (const std::int64_t h : g.heuristic) {
            if (h < 0) {
                return make_error<void>(MathError::domain_error);
            }
        }
    }
    if (!g.landscape.empty() && g.landscape.size() != n) {
        return make_error<void>(MathError::domain_error);
    }
    return {};
}

auto successors_of(const WireGraph& g) -> SuccessorFn {
    return [g](std::int64_t u) -> std::vector<std::int64_t> {
        std::vector<std::int64_t> out;
        if (u < 0 || static_cast<std::size_t>(u) >= g.adjacency.size()) {
            return out;
        }
        const std::vector<Edge>& edges = g.adjacency[static_cast<std::size_t>(u)];
        out.reserve(edges.size());
        for (const Edge& e : edges) {
            out.push_back(e.target);
        }
        return out;
    };
}

auto cost_of(const WireGraph& g) -> CostFn {
    return [g](std::int64_t u, std::int64_t v) -> std::int64_t {
        if (u < 0 || static_cast<std::size_t>(u) >= g.adjacency.size()) {
            return 0;
        }
        // PARALLEL EDGES COLLAPSE TO THEIR MINIMUM. A CostFn can express only one cost per
        // (u, v) pair, so a graph holding two edges u -> v of different cost has no faithful
        // callback view -- and the minimum is the only collapse that leaves every shortest path
        // unchanged, because no route would ever choose the dearer of two identical hops.
        // Returning the first-listed cost instead would silently disagree with the distributed
        // search, which relaxes each parallel edge at its own cost.
        const auto& edges = g.adjacency[static_cast<std::size_t>(u)];
        auto matching =
            edges | std::views::filter([v](const Edge& e) { return e.target == v; }) |
            std::views::transform(&Edge::cost);
        const auto cheapest = std::ranges::min_element(matching);
        if (cheapest == std::ranges::end(matching)) {
            return 0;
        }
        return *cheapest;
    };
}

auto heuristic_of(const WireGraph& g) -> HeuristicFn {
    return [g](std::int64_t u) -> std::int64_t {
        if (g.heuristic.empty() || u < 0 ||
            static_cast<std::size_t>(u) >= g.heuristic.size()) {
            return 0;
        }
        return g.heuristic[static_cast<std::size_t>(u)];
    };
}

// ---------------------------------------------------------------------------
// Wire format.
// ---------------------------------------------------------------------------

auto encode_slice(const WireGraph& g, std::size_t first, std::size_t count) -> Result<Payload> {
    if (first > g.adjacency.size() || count > g.adjacency.size() - first) {
        return make_error<Payload>(MathError::domain_error);
    }
    Payload out;
    put_u64(out, slice_magic);
    put_u64(out, static_cast<std::uint64_t>(g.adjacency.size()));
    put_u64(out, static_cast<std::uint64_t>(first));
    put_u64(out, static_cast<std::uint64_t>(count));
    for (const std::vector<Edge>& edges :
         g.adjacency | std::views::drop(first) | std::views::take(count)) {
        put_u64(out, static_cast<std::uint64_t>(edges.size()));
        for (const Edge& e : edges) {
            put_i64(out, e.target);
            put_i64(out, e.cost);
        }
    }
    if (g.heuristic.empty()) {
        put_u64(out, 0);
    } else {
        put_u64(out, static_cast<std::uint64_t>(count));
        for (const std::int64_t h :
             g.heuristic | std::views::drop(first) | std::views::take(count)) {
            put_i64(out, h);
        }
    }
    return out;
}

// The weighted form every round builder uses. `encode_frontier` is the exported 1/1 case: plain
// A*, where the heuristic is taken at face value.
auto encode_frontier_weighted(std::span<const FrontierEntry> frontier, std::int64_t bound,
                              std::int64_t weight_num, std::int64_t weight_den,
                              Semiring ring = Semiring::min_plus) -> Payload {
    Payload out;
    out.reserve(48 + frontier.size() * 16);
    put_u64(out, frontier_magic);
    put_i64(out, bound);
    put_i64(out, weight_num);
    put_i64(out, weight_den);
    put_u64(out, ring == Semiring::min_plus ? 0ULL : 1ULL);
    put_u64(out, static_cast<std::uint64_t>(frontier.size()));
    for (const FrontierEntry& fe : frontier) {
        put_i64(out, fe.node);
        put_i64(out, fe.distance);
    }
    return out;
}

auto encode_frontier(std::span<const FrontierEntry> frontier, std::int64_t bound)
    -> Result<Payload> {
    return encode_frontier_weighted(frontier, bound, 1, 1);
}

auto decode_proposals(std::span<const std::byte> bytes) -> Result<std::vector<Proposal>> {
    using Ps = std::vector<Proposal>;
    std::size_t off = 0;
    const auto magic = take_u64(bytes, off);
    if (!magic.has_value() || *magic != proposal_magic) {
        return make_error<Ps>(MathError::syntax_error);
    }
    const auto n = take_u64(bytes, off);
    if (!n.has_value() || !count_is_sane(*n, bytes.size() - off, 24)) {
        return make_error<Ps>(MathError::syntax_error);
    }
    Ps ps(static_cast<std::size_t>(*n));
    for (Proposal& slot : ps) {
        const auto t = take_i64(bytes, off);
        const auto d = take_i64(bytes, off);
        const auto p = take_i64(bytes, off);
        if (!t.has_value() || !d.has_value() || !p.has_value()) {
            return make_error<Ps>(MathError::syntax_error);
        }
        slot = Proposal{.target = *t, .distance = *d, .parent = *p};
    }
    if (off != bytes.size()) {
        return make_error<Ps>(MathError::syntax_error);
    }
    return ps;
}

// ---------------------------------------------------------------------------
// The depth-limited probe — one root branch of an iterative-deepening pass.
// ---------------------------------------------------------------------------

// What a probe found: the shallowest goal path within its limit, and whether the limit CUT OFF
// unexplored nodes. The cut-off flag is what lets the coordinator tell "no goal exists" from "no
// goal yet, go deeper" -- without it a search that ran out of depth would be indistinguishable
// from one that exhausted the graph, and reporting the wrong one of those is exactly the kind of
// plausible-looking wrong answer the honesty invariant rules out.
struct ProbeResult {
    bool found{false};
    bool cut_off{false};
    std::vector<std::int64_t> path;
};

[[nodiscard]] auto encode_probe(const ProbeResult& r) -> Payload {
    Payload out;
    put_u64(out, dls_magic);
    put_u64(out, r.found ? 1ULL : 0ULL);
    put_u64(out, r.cut_off ? 1ULL : 0ULL);
    put_u64(out, static_cast<std::uint64_t>(r.path.size()));
    for (const std::int64_t v : r.path) {
        put_i64(out, v);
    }
    return out;
}

[[nodiscard]] auto decode_probe(std::span<const std::byte> bytes) -> Result<ProbeResult> {
    std::size_t off = 0;
    const auto magic = take_u64(bytes, off);
    if (!magic.has_value() || *magic != dls_magic) {
        return make_error<ProbeResult>(MathError::syntax_error);
    }
    const auto found = take_u64(bytes, off);
    const auto cut = take_u64(bytes, off);
    const auto n = take_u64(bytes, off);
    if (!found.has_value() || !cut.has_value() || !n.has_value() ||
        !count_is_sane(*n, bytes.size() - off, 8)) {
        return make_error<ProbeResult>(MathError::syntax_error);
    }
    ProbeResult r;
    r.found = *found != 0;
    r.cut_off = *cut != 0;
    r.path.resize(static_cast<std::size_t>(*n));
    for (std::int64_t& slot : r.path) {
        const auto v = take_i64(bytes, off);
        if (!v.has_value()) {
            return make_error<ProbeResult>(MathError::syntax_error);
        }
        slot = *v;
    }
    if (off != bytes.size()) {
        return make_error<ProbeResult>(MathError::syntax_error);
    }
    return r;
}

// Runs one depth-limited probe. Literals are [whole-graph slice, probe arguments].
//
// Cycles are broken with an ON-PATH ancestor check rather than a global visited set, matching
// `iterative_deepening_dfs`: a node reachable by two routes of different length must stay
// reachable by the shorter one, and a global visited set would close it off after the first.
[[nodiscard]] auto depth_limited_probe(std::span<const Payload> args) -> Result<Payload> {
    if (args.size() != 2) {
        return make_error<Payload>(MathError::domain_error);
    }
    auto slice = decode_slice(args[0]);
    if (!slice) {
        return make_error<Payload>(slice.error());
    }
    std::size_t off = 0;
    const auto magic = take_u64(args[1], off);
    const auto root = take_i64(args[1], off);
    const auto limit = take_i64(args[1], off);
    const auto goal_count = take_u64(args[1], off);
    if (!magic.has_value() || *magic != dls_magic || !root.has_value() || !limit.has_value() ||
        !goal_count.has_value() || !count_is_sane(*goal_count, args[1].size() - off, 8)) {
        return make_error<Payload>(MathError::syntax_error);
    }
    std::vector<std::int64_t> goals(static_cast<std::size_t>(*goal_count));
    for (std::int64_t& slot : goals) {
        const auto v = take_i64(args[1], off);
        if (!v.has_value()) {
            return make_error<Payload>(MathError::syntax_error);
        }
        slot = *v;
    }
    if (off != args[1].size()) {
        return make_error<Payload>(MathError::syntax_error);
    }

    const std::vector<std::vector<Edge>>& adj = slice->adjacency;
    const auto is_goal = [&goals](std::int64_t u) -> bool {
        return std::ranges::find(goals, u) != goals.end();
    };

    ProbeResult result;
    std::vector<std::int64_t> on_path;

    // Explicit stack rather than recursion: the probe limit is caller-supplied, and a deep limit
    // must not become a stack overflow inside a worker process.
    struct Frame {
        std::int64_t node{0};
        std::int64_t depth{0};
        std::size_t next_child{0};
        bool entered{false};
    };
    std::vector<Frame> stack;
    stack.push_back(Frame{.node = *root, .depth = 0, .next_child = 0, .entered = false});

    while (!stack.empty()) {
        Frame& f = stack.back();
        if (!f.entered) {
            f.entered = true;
            if (is_goal(f.node)) {
                result.found = true;
                result.path = on_path;
                result.path.push_back(f.node);
                return encode_probe(result);
            }
            if (f.depth >= *limit) {
                // Successors exist that we are not allowed to look at, so "not found" here is
                // provisional and the coordinator has to be told so.
                if (f.node >= 0 && static_cast<std::size_t>(f.node) < adj.size() &&
                    !adj[static_cast<std::size_t>(f.node)].empty()) {
                    result.cut_off = true;
                }
                stack.pop_back();
                continue;
            }
            on_path.push_back(f.node);
        }
        if (f.node < 0 || static_cast<std::size_t>(f.node) >= adj.size() ||
            f.next_child >= adj[static_cast<std::size_t>(f.node)].size()) {
            on_path.pop_back();
            stack.pop_back();
            continue;
        }
        const Edge& e = adj[static_cast<std::size_t>(f.node)][f.next_child];
        ++f.next_child;
        if (std::ranges::find(on_path, e.target) != on_path.end()) {
            continue;  // an ancestor: following it could only lengthen the path
        }
        stack.push_back(
            Frame{.node = e.target, .depth = f.depth + 1, .next_child = 0, .entered = false});
    }
    return encode_probe(result);
}

// ---------------------------------------------------------------------------
// The Floyd-Warshall row-block update.
// ---------------------------------------------------------------------------

// Literals are [block payload, pivot row payload]. The block carries the rows this shard owns;
// the pivot row is row k, which every shard needs and none owns exclusively.
[[nodiscard]] auto floyd_block(std::span<const Payload> args) -> Result<Payload> {
    if (args.size() != 2) {
        return make_error<Payload>(MathError::domain_error);
    }
    std::size_t off = 0;
    const auto magic = take_u64(args[0], off);
    const auto n64 = take_u64(args[0], off);
    const auto first = take_u64(args[0], off);
    const auto rows64 = take_u64(args[0], off);
    const auto pivot = take_i64(args[0], off);
    if (!magic.has_value() || *magic != fw_magic || !n64.has_value() || !first.has_value() ||
        !rows64.has_value() || !pivot.has_value()) {
        return make_error<Payload>(MathError::syntax_error);
    }
    const auto n = static_cast<std::size_t>(*n64);
    const auto rows = static_cast<std::size_t>(*rows64);
    if (n == 0 || rows > n || !count_is_sane(*rows64, args[0].size() - off, n * 8)) {
        return make_error<Payload>(MathError::syntax_error);
    }
    std::vector<std::int64_t> block(rows * n);
    for (std::int64_t& slot : block) {
        const auto v = take_i64(args[0], off);
        if (!v.has_value()) {
            return make_error<Payload>(MathError::syntax_error);
        }
        slot = *v;
    }
    if (off != args[0].size()) {
        return make_error<Payload>(MathError::syntax_error);
    }

    std::size_t poff = 0;
    const auto pmagic = take_u64(args[1], poff);
    const auto plen = take_u64(args[1], poff);
    if (!pmagic.has_value() || *pmagic != fw_magic || !plen.has_value() ||
        static_cast<std::size_t>(*plen) != n) {
        return make_error<Payload>(MathError::syntax_error);
    }
    std::vector<std::int64_t> pivot_row(n);
    for (std::int64_t& slot : pivot_row) {
        const auto v = take_i64(args[1], poff);
        if (!v.has_value()) {
            return make_error<Payload>(MathError::syntax_error);
        }
        slot = *v;
    }
    if (poff != args[1].size()) {
        return make_error<Payload>(MathError::syntax_error);
    }

    const auto k = static_cast<std::size_t>(*pivot);
    if (k >= n) {
        return make_error<Payload>(MathError::syntax_error);
    }
    for (const std::size_t r : std::views::iota(std::size_t{0}, rows)) {
        const std::int64_t to_pivot = block[r * n + k];
        if (to_pivot == fw_absent) {
            continue;  // this row cannot reach the pivot, so the pivot cannot shorten it
        }
        for (const std::size_t c : std::views::iota(std::size_t{0}, n)) {
            const std::int64_t from_pivot = pivot_row[c];
            if (from_pivot == fw_absent) {
                continue;
            }
            std::int64_t through = 0;
            if (add_overflows(to_pivot, from_pivot, through)) {
                return make_error<Payload>(MathError::overflow);
            }
            std::int64_t& cur = block[r * n + c];
            if (cur == fw_absent || through < cur) {
                cur = through;
            }
        }
    }

    Payload out;
    put_u64(out, fw_magic);
    put_u64(out, *first);
    put_u64(out, static_cast<std::uint64_t>(rows));
    for (const std::int64_t v : block) {
        put_i64(out, v);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Triangle counting — one shard counts the triangles at its own nodes.
// ---------------------------------------------------------------------------

// Literals are [whole undirected graph, range payload]. The WHOLE graph travels because closing a
// triangle at u needs the neighbourhood of u's neighbours, which live anywhere; only the nodes
// COUNTED AT are restricted to the shard's range, and that is what makes the shards disjoint.
[[nodiscard]] auto triangle_shard(std::span<const Payload> args) -> Result<Payload> {
    if (args.size() != 2) {
        return make_error<Payload>(MathError::domain_error);
    }
    auto slice = decode_slice(args[0]);
    if (!slice) {
        return make_error<Payload>(slice.error());
    }
    std::size_t off = 0;
    const auto magic = take_u64(args[1], off);
    const auto first64 = take_u64(args[1], off);
    const auto count64 = take_u64(args[1], off);
    if (!magic.has_value() || *magic != tri_magic || !first64.has_value() ||
        !count64.has_value() || off != args[1].size()) {
        return make_error<Payload>(MathError::syntax_error);
    }
    const std::vector<std::vector<Edge>>& adj = slice->adjacency;
    const auto first = static_cast<std::size_t>(*first64);
    const auto count = static_cast<std::size_t>(*count64);
    if (first > adj.size() || count > adj.size() - first) {
        return make_error<Payload>(MathError::syntax_error);
    }

    // The adjacency arrives sorted by target (the undirected view is built from an ordered map),
    // so membership is a binary search rather than a scan.
    const auto adjacent = [&adj](std::size_t a, std::int64_t b) -> bool {
        return std::ranges::binary_search(adj[a], b, {}, &Edge::target);
    };

    std::vector<std::int64_t> counts(count, 0);
    for (const std::size_t i : std::views::iota(std::size_t{0}, count)) {
        const std::size_t u = first + i;
        const std::vector<Edge>& nbrs = adj[u];
        for (const std::size_t a : std::views::iota(std::size_t{0}, nbrs.size())) {
            for (const std::size_t b : std::views::iota(a + 1, nbrs.size())) {
                const std::int64_t v = nbrs[a].target;
                const std::int64_t w = nbrs[b].target;
                if (v == static_cast<std::int64_t>(u) || w == static_cast<std::int64_t>(u)) {
                    continue;  // a self-loop closes nothing
                }
                if (adjacent(static_cast<std::size_t>(v), w)) {
                    ++counts[i];
                }
            }
        }
    }

    Payload out;
    put_u64(out, tri_magic);
    put_u64(out, *first64);
    put_u64(out, static_cast<std::uint64_t>(counts.size()));
    for (const std::int64_t c : counts) {
        put_i64(out, c);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Boruvka — one shard finds the cheapest edge leaving each component it touches.
// ---------------------------------------------------------------------------

// One candidate: the cheapest edge leaving `component`, as (weight, u, v).
struct Candidate {
    std::int64_t component{0};
    std::int64_t weight{0};
    std::int64_t u{0};
    std::int64_t v{0};

    // A STRICT TOTAL ORDER on edges, and it is what makes Boruvka safe here. With ties broken
    // arbitrarily, two components can each pick a different edge of the same weight between them
    // and the merge closes a cycle. Ordering by (weight, u, v) makes the cheapest outgoing edge
    // UNIQUE, so the classic tie-breaking hazard cannot arise -- and it makes the result
    // reproducible, which a distributed run needs anyway.
    [[nodiscard]] auto better_than(const Candidate& other) const noexcept -> bool {
        return std::tie(weight, u, v) < std::tie(other.weight, other.u, other.v);
    }
};

[[nodiscard]] auto boruvka_shard(std::span<const Payload> args) -> Result<Payload> {
    if (args.size() != 2) {
        return make_error<Payload>(MathError::domain_error);
    }
    auto slice = decode_slice(args[0]);
    if (!slice) {
        return make_error<Payload>(slice.error());
    }
    std::size_t off = 0;
    const auto magic = take_u64(args[1], off);
    const auto n64 = take_u64(args[1], off);
    if (!magic.has_value() || *magic != mst_magic || !n64.has_value() ||
        !count_is_sane(*n64, args[1].size() - off, 8)) {
        return make_error<Payload>(MathError::syntax_error);
    }
    std::vector<std::int64_t> component(static_cast<std::size_t>(*n64));
    for (std::int64_t& slot : component) {
        const auto v = take_i64(args[1], off);
        if (!v.has_value()) {
            return make_error<Payload>(MathError::syntax_error);
        }
        slot = *v;
    }
    if (off != args[1].size()) {
        return make_error<Payload>(MathError::syntax_error);
    }

    const auto first = slice->first;
    std::map<std::int64_t, Candidate> best;
    for (const std::size_t i : std::views::iota(std::size_t{0}, slice->adjacency.size())) {
        const auto u = static_cast<std::int64_t>(first + i);
        if (static_cast<std::size_t>(u) >= component.size()) {
            return make_error<Payload>(MathError::syntax_error);
        }
        const std::int64_t cu = component[static_cast<std::size_t>(u)];
        for (const Edge& e : slice->adjacency[i]) {
            if (e.target < 0 || static_cast<std::size_t>(e.target) >= component.size()) {
                return make_error<Payload>(MathError::syntax_error);
            }
            const std::int64_t cv = component[static_cast<std::size_t>(e.target)];
            if (cu == cv) {
                continue;  // inside the component: taking it would close a cycle
            }
            const Candidate cand{
                .component = cu, .weight = e.cost, .u = u, .v = e.target};
            auto it = best.find(cu);
            if (it == best.end() || cand.better_than(it->second)) {
                best[cu] = cand;
            }
        }
    }

    Payload out;
    put_u64(out, mst_magic);
    put_u64(out, static_cast<std::uint64_t>(best.size()));
    for (const auto& [comp, cand] : best) {
        put_i64(out, cand.component);
        put_i64(out, cand.weight);
        put_i64(out, cand.u);
        put_i64(out, cand.v);
    }
    return out;
}

[[nodiscard]] auto decode_candidates(std::span<const std::byte> bytes)
    -> Result<std::vector<Candidate>> {
    using Cs = std::vector<Candidate>;
    std::size_t off = 0;
    const auto magic = take_u64(bytes, off);
    if (!magic.has_value() || *magic != mst_magic) {
        return make_error<Cs>(MathError::syntax_error);
    }
    const auto n = take_u64(bytes, off);
    if (!n.has_value() || !count_is_sane(*n, bytes.size() - off, 32)) {
        return make_error<Cs>(MathError::syntax_error);
    }
    Cs cs(static_cast<std::size_t>(*n));
    for (Candidate& c : cs) {
        const auto comp = take_i64(bytes, off);
        const auto w = take_i64(bytes, off);
        const auto u = take_i64(bytes, off);
        const auto v = take_i64(bytes, off);
        if (!comp.has_value() || !w.has_value() || !u.has_value() || !v.has_value()) {
            return make_error<Cs>(MathError::syntax_error);
        }
        c = Candidate{.component = *comp, .weight = *w, .u = *u, .v = *v};
    }
    if (off != bytes.size()) {
        return make_error<Cs>(MathError::syntax_error);
    }
    return cs;
}

// ---------------------------------------------------------------------------
// Running.
// ---------------------------------------------------------------------------

auto register_ops(TaskRegistry& reg) -> Result<void> {
    // All of them or none: a registry holding half would let a graph build and then fail in a
    // worker, which is the failure this whole registration step exists to move earlier.
    const std::array<std::pair<std::string_view, TaskFn>, 6> ops{{
        {relax_op_id, relax_shard},
        {tabu_op_id, tabu_run},
        {dls_op_id, depth_limited_probe},
        {fw_op_id, floyd_block},
        {tri_op_id, triangle_shard},
        {mst_op_id, boruvka_shard},
    }};
    for (const auto& [id, fn] : ops) {
        auto r = reg.register_op(OpId{id}, fn);
        if (!r) {
            return r;
        }
    }
    return {};
}

auto build_round_graph_weighted(const TaskRegistry& reg, const WireGraph& g,
                                std::span<const FrontierEntry> frontier, std::int64_t bound,
                                std::int64_t weight_num, std::int64_t weight_den, Semiring ring,
                                std::size_t shard_count) -> Result<TaskGraph> {
    if (shard_count == 0) {
        return make_error<TaskGraph>(MathError::domain_error);
    }
    auto ok = validate(g);
    if (!ok) {
        return make_error<TaskGraph>(ok.error());
    }

    const std::vector<std::size_t> bounds = slice_bounds(g.adjacency.size(), shard_count);
    TaskGraph graph;
    for (const std::size_t s : std::views::iota(std::size_t{0}, shard_count)) {
        auto slice = encode_slice(g, bounds[s], bounds[s + 1] - bounds[s]);
        if (!slice) {
            return make_error<TaskGraph>(slice.error());
        }
        // Only the frontier nodes this shard owns are shipped to it. Partitioning here rather
        // than shard-side is what makes the split testable: a node in the wrong shard is a
        // coordinator bug, and a coordinator bug belongs in coordinator tests.
        std::vector<FrontierEntry> mine;
        for (const FrontierEntry& fe : frontier) {
            if (fe.node < 0) {
                continue;
            }
            const auto u = static_cast<std::size_t>(fe.node);
            if (u >= bounds[s] && u < bounds[s + 1]) {
                mine.push_back(fe);
            }
        }
        Payload front = encode_frontier_weighted(mine, bound, weight_num, weight_den, ring);
        auto id = graph.add_named_task(reg, OpId{relax_op_id},
                                       std::vector<Payload>{std::move(*slice), std::move(front)});
        if (!id) {
            return make_error<TaskGraph>(id.error());
        }
    }
    return graph;
}

// The exported unweighted form: plain A* / Dijkstra rounds, the heuristic taken at face value.
auto build_round_graph(const TaskRegistry& reg, const WireGraph& g,
                       std::span<const FrontierEntry> frontier, std::int64_t bound,
                       std::size_t shard_count) -> Result<TaskGraph> {
    return build_round_graph_weighted(reg, g, frontier, bound, 1, 1, Semiring::min_plus,
                                      shard_count);
}

namespace {

// The round loop, seeded with whatever initial values the caller has.
//
// A single-source search seeds one node at 0. LABEL PROPAGATION seeds EVERY node with its own id
// and puts them all in the first frontier, and connected components falls straight out of that:
// over a zero-weighted undirected graph the min_plus semiring propagates the smallest id in each
// component. That is the same loop, not a similar one, which is the point of separating the
// seeding from the iteration.
auto run_rounds_seeded(const TaskRegistry& reg, const WireGraph& g,
                       std::vector<std::optional<std::int64_t>> initial,
                       std::vector<FrontierEntry> frontier, std::optional<std::int64_t> goal,
                       std::int64_t weight_num, std::int64_t weight_den, Semiring ring,
                       std::size_t shard_count, Executor& exec) -> Result<RoundOutcome> {
    const std::size_t n = g.adjacency.size();
    RoundOutcome out;
    out.dist = std::move(initial);
    out.parent.assign(n, std::nullopt);

    std::int64_t bound = std::numeric_limits<std::int64_t>::max();
    if (ring == Semiring::min_plus && goal.has_value()) {
        const auto gi = static_cast<std::size_t>(*goal);
        if (gi < n && out.dist[gi].has_value()) {
            bound = *out.dist[gi];
        }
    }

    // One round per node is the worst case: a shortest path has at most n-1 edges, and each round
    // finalises at least one more edge of it. Exceeding the bound means the invariant broke, and
    // saying not_converged is the honest response to that.
    for ([[maybe_unused]] const std::size_t round : std::views::iota(std::size_t{0}, n + 1)) {
        if (frontier.empty()) {
            return out;
        }
        auto graph = build_round_graph_weighted(reg, g, frontier, bound, weight_num, weight_den,
                                                ring, shard_count);
        if (!graph) {
            return make_error<RoundOutcome>(graph.error());
        }
        auto run = exec.run(*graph);
        if (!run) {
            return make_error<RoundOutcome>(run.error());
        }

        std::vector<FrontierEntry> next;
        // Improvements are collected per node so a node that improves several times in one round
        // enters the next frontier once, at its best distance.
        std::map<std::int64_t, FrontierEntry> improved;
        for (const Result<Payload>& outcome : run->outputs) {
            if (!outcome) {
                return make_error<RoundOutcome>(outcome.error());
            }
            auto ps = decode_proposals(*outcome);
            if (!ps) {
                return make_error<RoundOutcome>(ps.error());
            }
            for (const Proposal& p : *ps) {
                if (p.target < 0 || static_cast<std::size_t>(p.target) >= n) {
                    return make_error<RoundOutcome>(MathError::domain_error);
                }
                const auto t = static_cast<std::size_t>(p.target);
                std::optional<std::int64_t>& d = out.dist[t];
                // "Better" is the semiring's ordering: smaller for min_plus, larger for max_min.
                const bool better = !d.has_value() || (ring == Semiring::min_plus
                                                           ? p.distance < *d
                                                           : p.distance > *d);
                // Equal distance, lower parent id: the same tie-break `dijkstra` applies when it
                // relaxes, so an optimal path that is unique comes out identical.
                const bool same_but_lower_parent =
                    d.has_value() && p.distance == *d && out.parent[t].has_value() &&
                    p.parent < *out.parent[t];
                if (better || same_but_lower_parent) {
                    d = p.distance;
                    out.parent[t] = p.parent;
                    if (better) {
                        improved[p.target] =
                            FrontierEntry{.node = p.target, .distance = p.distance};
                    }
                }
            }
        }
        if (ring == Semiring::min_plus && goal.has_value()) {
            const auto gi = static_cast<std::size_t>(*goal);
            if (out.dist[gi].has_value()) {
                bound = *out.dist[gi];
            }
        }
        next.reserve(improved.size());
        for (const auto& [node, fe] : improved) {
            next.push_back(fe);
        }
        frontier = std::move(next);
    }
    return make_error<RoundOutcome>(MathError::not_converged);
}

auto run_rounds(const TaskRegistry& reg, const WireGraph& g, std::int64_t start,
                std::optional<std::int64_t> goal, std::int64_t weight_num,
                std::int64_t weight_den, std::size_t shard_count, Executor& exec)
    -> Result<RoundOutcome> {
    std::vector<std::optional<std::int64_t>> initial(g.adjacency.size(), std::nullopt);
    initial[static_cast<std::size_t>(start)] = 0;
    std::vector<FrontierEntry> frontier{FrontierEntry{.node = start, .distance = 0}};
    return run_rounds_seeded(reg, g, std::move(initial), std::move(frontier), goal, weight_num,
                             weight_den, Semiring::min_plus, shard_count, exec);
}

auto run_one_round(const TaskRegistry& reg, const WireGraph& g,
                   std::span<const FrontierEntry> frontier, std::size_t shard_count,
                   Executor& exec) -> Result<std::vector<Proposal>> {
    using Ps = std::vector<Proposal>;
    auto graph = build_round_graph_weighted(reg, g, frontier,
                                            std::numeric_limits<std::int64_t>::max(), 1, 1,
                                            Semiring::min_plus, shard_count);
    if (!graph) {
        return make_error<Ps>(graph.error());
    }
    auto run = exec.run(*graph);
    if (!run) {
        return make_error<Ps>(run.error());
    }
    Ps all;
    for (const Result<Payload>& outcome : run->outputs) {
        if (!outcome) {
            return make_error<Ps>(outcome.error());
        }
        auto ps = decode_proposals(*outcome);
        if (!ps) {
            return make_error<Ps>(ps.error());
        }
        all.insert(all.end(), ps->begin(), ps->end());
    }
    return all;
}

// The undirected view: every edge also runs backwards, and parallel edges collapse to the
// cheapest. Connected components, k-core, triangles and the spanning forest are all properties of
// the UNDIRECTED graph, so building this view once -- and saying so -- is better than each of them
// quietly assuming the caller already symmetrised.
[[nodiscard]] auto undirected_view(const WireGraph& g, bool unit_weights) -> WireGraph {
    const std::size_t n = g.adjacency.size();
    std::vector<std::map<std::int64_t, std::int64_t>> best(n);
    for (const std::size_t u : std::views::iota(std::size_t{0}, n)) {
        for (const Edge& e : g.adjacency[u]) {
            const auto v = static_cast<std::size_t>(e.target);
            const std::int64_t w = unit_weights ? 1 : e.cost;
            for (const auto& [from, to] : {std::pair{u, v}, std::pair{v, u}}) {
                auto it = best[from].find(static_cast<std::int64_t>(to));
                if (it == best[from].end() || w < it->second) {
                    best[from][static_cast<std::int64_t>(to)] = w;
                }
            }
        }
    }
    WireGraph out;
    out.adjacency.resize(n);
    for (const std::size_t u : std::views::iota(std::size_t{0}, n)) {
        for (const auto& [v, w] : best[u]) {
            out.adjacency[u].push_back(Edge{.target = v, .cost = w});
        }
    }
    return out;
}

}  // namespace

auto distributed_sssp(const WireGraph& g, std::int64_t start, std::size_t shard_count,
                      Executor& exec) -> Result<std::vector<std::optional<std::int64_t>>> {
    using Dists = std::vector<std::optional<std::int64_t>>;
    auto ok = validate(g);
    if (!ok) {
        return make_error<Dists>(ok.error());
    }
    if (shard_count == 0 || start < 0 || static_cast<std::size_t>(start) >= g.adjacency.size()) {
        return make_error<Dists>(MathError::domain_error);
    }
    TaskRegistry reg;
    auto reg_ok = register_ops(reg);
    if (!reg_ok) {
        return make_error<Dists>(reg_ok.error());
    }
    auto rounds = run_rounds(reg, g, start, std::nullopt, 1, 1, shard_count, exec);
    if (!rounds) {
        return make_error<Dists>(rounds.error());
    }
    return std::move(rounds->dist);
}

namespace {

[[nodiscard]] auto path_from_rounds(const WireGraph& g, std::int64_t start, std::int64_t goal_node,
                                    std::size_t shard_count, Executor& exec, bool use_heuristic,
                                    std::int64_t weight_num = 1, std::int64_t weight_den = 1)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>> {
    using PathCost = std::pair<std::vector<std::int64_t>, std::int64_t>;
    auto ok = validate(g);
    if (!ok) {
        return make_error<PathCost>(ok.error());
    }
    const std::size_t n = g.adjacency.size();
    if (shard_count == 0 || start < 0 || static_cast<std::size_t>(start) >= n ||
        goal_node < 0 || static_cast<std::size_t>(goal_node) >= n) {
        return make_error<PathCost>(MathError::domain_error);
    }
    if (start == goal_node) {
        return PathCost{std::vector<std::int64_t>{start}, 0};
    }
    TaskRegistry reg;
    auto reg_ok = register_ops(reg);
    if (!reg_ok) {
        return make_error<PathCost>(reg_ok.error());
    }
    const std::optional<std::int64_t> goal =
        use_heuristic ? std::optional<std::int64_t>{goal_node} : std::nullopt;
    auto rounds = run_rounds(reg, g, start, goal, weight_num, weight_den, shard_count, exec);
    if (!rounds) {
        return make_error<PathCost>(rounds.error());
    }
    const std::optional<std::int64_t>& d = rounds->dist[static_cast<std::size_t>(goal_node)];
    if (!d.has_value()) {
        return make_error<PathCost>(MathError::undefined_value);
    }
    auto path = rebuild_path(rounds->parent, start, goal_node);
    if (!path) {
        return make_error<PathCost>(path.error());
    }
    return PathCost{std::move(*path), *d};
}

}  // namespace

auto distributed_shortest_path(const WireGraph& g, std::int64_t start, std::int64_t goal_node,
                               std::size_t shard_count, Executor& exec)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>> {
    return path_from_rounds(g, start, goal_node, shard_count, exec, false);
}

auto distributed_a_star(const WireGraph& g, std::int64_t start, std::int64_t goal_node,
                        std::size_t shard_count, Executor& exec)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>> {
    return path_from_rounds(g, start, goal_node, shard_count, exec, true);
}

auto distributed_multistart_tabu(const WireGraph& g, std::span<const std::int64_t> starts,
                                 std::int64_t tabu_tenure, std::int64_t max_iters, Executor& exec)
    -> Result<std::pair<TabuState, std::int64_t>> {
    using Best = std::pair<TabuState, std::int64_t>;
    auto ok = validate(g);
    if (!ok) {
        return make_error<Best>(ok.error());
    }
    const std::size_t n = g.adjacency.size();
    if (starts.empty() || tabu_tenure < 0 || max_iters < 0 || g.landscape.size() != n) {
        return make_error<Best>(MathError::domain_error);
    }
    for (const std::int64_t s : starts) {
        if (s < 0 || static_cast<std::size_t>(s) >= n) {
            return make_error<Best>(MathError::domain_error);
        }
    }

    TaskRegistry reg;
    auto reg_ok = register_ops(reg);
    if (!reg_ok) {
        return make_error<Best>(reg_ok.error());
    }

    // The tabu op reads the landscape out of the slice's heuristic field: one payload shape, and
    // the graph shipped whole because a local search is not confined to a node range.
    WireGraph shipped = g;
    shipped.heuristic = g.landscape;
    auto slice = encode_slice(shipped, 0, n);
    if (!slice) {
        return make_error<Best>(slice.error());
    }

    TaskGraph graph;
    for (const std::int64_t s : starts) {
        Payload args;
        put_u64(args, tabu_magic);
        put_i64(args, s);
        put_i64(args, tabu_tenure);
        put_i64(args, max_iters);
        auto id = graph.add_named_task(reg, OpId{tabu_op_id},
                                       std::vector<Payload>{*slice, std::move(args)});
        if (!id) {
            return make_error<Best>(id.error());
        }
    }
    auto run = exec.run(graph);
    if (!run) {
        return make_error<Best>(run.error());
    }

    TabuState best_state;
    std::int64_t best_val = 0;
    bool has_best = false;
    for (const Result<Payload>& outcome : run->outputs) {
        if (!outcome) {
            return make_error<Best>(outcome.error());
        }
        auto one = decode_tabu_result(*outcome);
        if (!one) {
            return make_error<Best>(one.error());
        }
        const auto& [state, val] = *one;
        // The same deterministic reduction parallel_multistart_tabu uses, so the two agree.
        if (!has_best || val < best_val || (val == best_val && state < best_state)) {
            has_best = true;
            best_state = state;
            best_val = val;
        }
    }
    return Best{best_state, best_val};
}


namespace {

// A copy of `g` with every edge weighted 1. BFS and reachability are the round loop over this
// graph: with unit weights the round index IS the hop count, so no second mechanism is needed and
// the two searches share every line of the one that is already tested.
[[nodiscard]] auto unit_weighted(const WireGraph& g) -> WireGraph {
    WireGraph u = g;
    for (std::vector<Edge>& row : u.adjacency) {
        for (Edge& e : row) {
            e.cost = 1;
        }
    }
    u.heuristic.clear();
    return u;
}

}  // namespace

auto distributed_bfs(const WireGraph& g, std::int64_t start, std::span<const std::int64_t> goals,
                     std::size_t shard_count, Executor& exec) -> Result<std::vector<std::int64_t>> {
    using Path = std::vector<std::int64_t>;
    auto ok = validate(g);
    if (!ok) {
        return make_error<Path>(ok.error());
    }
    const std::size_t n = g.adjacency.size();
    if (shard_count == 0 || start < 0 || static_cast<std::size_t>(start) >= n || goals.empty()) {
        return make_error<Path>(MathError::domain_error);
    }
    for (const std::int64_t gnode : goals) {
        if (gnode < 0 || static_cast<std::size_t>(gnode) >= n) {
            return make_error<Path>(MathError::domain_error);
        }
    }
    if (std::ranges::find(goals, start) != goals.end()) {
        return Path{start};
    }

    TaskRegistry reg;
    auto reg_ok = register_ops(reg);
    if (!reg_ok) {
        return make_error<Path>(reg_ok.error());
    }
    auto rounds = run_rounds(reg, unit_weighted(g), start, std::nullopt, 1, 1, shard_count, exec);
    if (!rounds) {
        return make_error<Path>(rounds.error());
    }

    // Fewest hops wins; among goals tied at that depth the lowest node id wins, which is a rule
    // that does not depend on which worker answered first.
    std::optional<std::int64_t> best_goal;
    std::int64_t best_hops = 0;
    for (const std::int64_t gnode : goals) {
        const std::optional<std::int64_t>& d = rounds->dist[static_cast<std::size_t>(gnode)];
        if (!d.has_value()) {
            continue;
        }
        if (!best_goal.has_value() || *d < best_hops || (*d == best_hops && gnode < *best_goal)) {
            best_goal = gnode;
            best_hops = *d;
        }
    }
    if (!best_goal.has_value()) {
        return make_error<Path>(MathError::undefined_value);
    }
    return rebuild_path(rounds->parent, start, *best_goal);
}

auto distributed_reachable_set(const WireGraph& g, std::int64_t start, std::size_t shard_count,
                               Executor& exec) -> Result<std::vector<std::int64_t>> {
    using Nodes = std::vector<std::int64_t>;
    auto ok = validate(g);
    if (!ok) {
        return make_error<Nodes>(ok.error());
    }
    const std::size_t n = g.adjacency.size();
    if (shard_count == 0 || start < 0 || static_cast<std::size_t>(start) >= n) {
        return make_error<Nodes>(MathError::domain_error);
    }
    TaskRegistry reg;
    auto reg_ok = register_ops(reg);
    if (!reg_ok) {
        return make_error<Nodes>(reg_ok.error());
    }
    auto rounds = run_rounds(reg, unit_weighted(g), start, std::nullopt, 1, 1, shard_count, exec);
    if (!rounds) {
        return make_error<Nodes>(rounds.error());
    }
    Nodes reachable;
    for (const std::size_t v : std::views::iota(std::size_t{0}, n)) {
        if (rounds->dist[v].has_value()) {
            reachable.push_back(static_cast<std::int64_t>(v));
        }
    }
    return reachable;
}

auto distributed_iterative_deepening(const WireGraph& g, std::int64_t start,
                                     std::span<const std::int64_t> goals, std::int64_t max_depth,
                                     std::size_t shard_count, Executor& exec)
    -> Result<std::vector<std::int64_t>> {
    using Path = std::vector<std::int64_t>;
    auto ok = validate(g);
    if (!ok) {
        return make_error<Path>(ok.error());
    }
    const std::size_t n = g.adjacency.size();
    if (shard_count == 0 || max_depth < 0 || start < 0 ||
        static_cast<std::size_t>(start) >= n || goals.empty()) {
        return make_error<Path>(MathError::domain_error);
    }
    for (const std::int64_t gnode : goals) {
        if (gnode < 0 || static_cast<std::size_t>(gnode) >= n) {
            return make_error<Path>(MathError::domain_error);
        }
    }
    if (std::ranges::find(goals, start) != goals.end()) {
        return Path{start};
    }

    TaskRegistry reg;
    auto reg_ok = register_ops(reg);
    if (!reg_ok) {
        return make_error<Path>(reg_ok.error());
    }
    auto whole = encode_slice(g, 0, n);
    if (!whole) {
        return make_error<Path>(whole.error());
    }

    const std::vector<Edge>& roots = g.adjacency[static_cast<std::size_t>(start)];
    if (roots.empty()) {
        // Nothing to fan out and `start` is not a goal, so the reachable graph is exhausted at
        // depth 0. That is genuinely "no such goal", not "not deep enough".
        return make_error<Path>(MathError::undefined_value);
    }

    // Depth limits grow one at a time, and the FIRST limit at which any branch reports a goal
    // gives the fewest-hops answer -- that is the whole reason iterative deepening exists, and
    // stopping at the first success is what preserves it.
    for (const std::int64_t limit : std::views::iota(std::int64_t{1}, max_depth + 1)) {
        TaskGraph graph;
        for (const Edge& e : roots) {
            Payload args;
            put_u64(args, dls_magic);
            put_i64(args, e.target);
            put_i64(args, limit - 1);
            put_u64(args, static_cast<std::uint64_t>(goals.size()));
            for (const std::int64_t gnode : goals) {
                put_i64(args, gnode);
            }
            auto id = graph.add_named_task(reg, OpId{dls_op_id},
                                           std::vector<Payload>{*whole, std::move(args)});
            if (!id) {
                return make_error<Path>(id.error());
            }
        }
        auto run = exec.run(graph);
        if (!run) {
            return make_error<Path>(run.error());
        }

        std::optional<Path> best;
        bool any_cut_off = false;
        for (const Result<Payload>& outcome : run->outputs) {
            if (!outcome) {
                return make_error<Path>(outcome.error());
            }
            auto probe = decode_probe(*outcome);
            if (!probe) {
                return make_error<Path>(probe.error());
            }
            any_cut_off = any_cut_off || probe->cut_off;
            if (!probe->found) {
                continue;
            }
            Path candidate;
            candidate.push_back(start);
            candidate.insert(candidate.end(), probe->path.begin(), probe->path.end());
            // Shortest wins; equal-length ties break to the lexicographically smaller path, so
            // the answer does not depend on the order the branches finished in.
            if (!best.has_value() || candidate.size() < best->size() ||
                (candidate.size() == best->size() && candidate < *best)) {
                best = std::move(candidate);
            }
        }
        if (best.has_value()) {
            return std::move(*best);
        }
        if (!any_cut_off) {
            // Every branch ran to exhaustion inside this limit and found nothing, so no deeper
            // limit can find anything either.
            return make_error<Path>(MathError::undefined_value);
        }
    }
    // The budget ran out with unexplored nodes still below the limit. Saying so is the honest
    // answer; returning the best partial path found so far would be a wrong answer wearing the
    // shape of a right one.
    return make_error<Path>(MathError::not_converged);
}

auto distributed_weighted_a_star(const WireGraph& g, std::int64_t start, std::int64_t goal_node,
                                 std::int64_t weight_num, std::int64_t weight_den,
                                 std::size_t shard_count, Executor& exec)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>> {
    using PathCost = std::pair<std::vector<std::int64_t>, std::int64_t>;
    if (weight_den <= 0 || weight_num < weight_den) {
        return make_error<PathCost>(MathError::domain_error);
    }
    return path_from_rounds(g, start, goal_node, shard_count, exec, true, weight_num, weight_den);
}

auto distributed_floyd_warshall(const WireGraph& g, std::size_t shard_count, Executor& exec)
    -> Result<std::vector<std::vector<std::optional<std::int64_t>>>> {
    using Matrix = std::vector<std::vector<std::optional<std::int64_t>>>;
    auto ok = validate(g);
    if (!ok) {
        return make_error<Matrix>(ok.error());
    }
    const std::size_t n = g.adjacency.size();
    if (shard_count == 0) {
        return make_error<Matrix>(MathError::domain_error);
    }
    if (n == 0) {
        return Matrix{};
    }

    TaskRegistry reg;
    auto reg_ok = register_ops(reg);
    if (!reg_ok) {
        return make_error<Matrix>(reg_ok.error());
    }

    // Row-major, `fw_absent` for "no path known yet". A self-distance is 0 whether or not the
    // graph has a self-loop, and parallel edges collapse to the cheapest, both of which are
    // properties of shortest paths rather than of the adjacency list.
    std::vector<std::int64_t> dist(n * n, fw_absent);
    for (const std::size_t i : std::views::iota(std::size_t{0}, n)) {
        dist[i * n + i] = 0;
    }
    for (const std::size_t u : std::views::iota(std::size_t{0}, n)) {
        for (const Edge& e : g.adjacency[u]) {
            const auto v = static_cast<std::size_t>(e.target);
            std::int64_t& cur = dist[u * n + v];
            if (cur == fw_absent || e.cost < cur) {
                cur = e.cost;
            }
        }
    }

    const std::vector<std::size_t> bounds = slice_bounds(n, shard_count);
    for (const std::size_t k : std::views::iota(std::size_t{0}, n)) {
        Payload pivot_row;
        put_u64(pivot_row, fw_magic);
        put_u64(pivot_row, static_cast<std::uint64_t>(n));
        for (const std::size_t c : std::views::iota(std::size_t{0}, n)) {
            put_i64(pivot_row, dist[k * n + c]);
        }

        TaskGraph graph;
        for (const std::size_t s : std::views::iota(std::size_t{0}, shard_count)) {
            const std::size_t first = bounds[s];
            const std::size_t rows = bounds[s + 1] - first;
            Payload block;
            put_u64(block, fw_magic);
            put_u64(block, static_cast<std::uint64_t>(n));
            put_u64(block, static_cast<std::uint64_t>(first));
            put_u64(block, static_cast<std::uint64_t>(rows));
            put_i64(block, static_cast<std::int64_t>(k));
            for (const std::size_t r : std::views::iota(std::size_t{0}, rows)) {
                for (const std::size_t c : std::views::iota(std::size_t{0}, n)) {
                    put_i64(block, dist[(first + r) * n + c]);
                }
            }
            auto id = graph.add_named_task(reg, OpId{fw_op_id},
                                           std::vector<Payload>{std::move(block), pivot_row});
            if (!id) {
                return make_error<Matrix>(id.error());
            }
        }

        auto run = exec.run(graph);
        if (!run) {
            return make_error<Matrix>(run.error());
        }
        for (const Result<Payload>& outcome : run->outputs) {
            if (!outcome) {
                return make_error<Matrix>(outcome.error());
            }
            std::size_t off = 0;
            const auto magic = take_u64(*outcome, off);
            const auto first64 = take_u64(*outcome, off);
            const auto rows64 = take_u64(*outcome, off);
            if (!magic.has_value() || *magic != fw_magic || !first64.has_value() ||
                !rows64.has_value()) {
                return make_error<Matrix>(MathError::syntax_error);
            }
            const auto first = static_cast<std::size_t>(*first64);
            const auto rows = static_cast<std::size_t>(*rows64);
            if (first > n || rows > n - first) {
                return make_error<Matrix>(MathError::syntax_error);
            }
            for (const std::size_t r : std::views::iota(std::size_t{0}, rows)) {
                for (const std::size_t c : std::views::iota(std::size_t{0}, n)) {
                    const auto v = take_i64(*outcome, off);
                    if (!v.has_value()) {
                        return make_error<Matrix>(MathError::syntax_error);
                    }
                    dist[(first + r) * n + c] = *v;
                }
            }
            if (off != outcome->size()) {
                return make_error<Matrix>(MathError::syntax_error);
            }
        }
    }

    Matrix out(n, std::vector<std::optional<std::int64_t>>(n, std::nullopt));
    for (const std::size_t i : std::views::iota(std::size_t{0}, n)) {
        for (const std::size_t j : std::views::iota(std::size_t{0}, n)) {
            const std::int64_t v = dist[i * n + j];
            if (v != fw_absent) {
                out[i][j] = v;
            }
        }
    }
    return out;
}

auto distributed_connected_components(const WireGraph& g, std::size_t shard_count, Executor& exec)
    -> Result<std::vector<std::int64_t>> {
    using Labels = std::vector<std::int64_t>;
    auto ok = validate(g);
    if (!ok) {
        return make_error<Labels>(ok.error());
    }
    const std::size_t n = g.adjacency.size();
    if (shard_count == 0) {
        return make_error<Labels>(MathError::domain_error);
    }
    if (n == 0) {
        return Labels{};
    }
    TaskRegistry reg;
    auto reg_ok = register_ops(reg);
    if (!reg_ok) {
        return make_error<Labels>(reg_ok.error());
    }

    // Zero-weighted, so a proposal carries the propagating LABEL rather than a distance: under
    // min_plus with cost 0, relaxing u across an edge proposes exactly label(u).
    WireGraph zero = undirected_view(g, false);
    for (std::vector<Edge>& row : zero.adjacency) {
        for (Edge& e : row) {
            e.cost = 0;
        }
    }

    std::vector<std::optional<std::int64_t>> initial(n);
    std::vector<FrontierEntry> frontier;
    frontier.reserve(n);
    for (const std::size_t u : std::views::iota(std::size_t{0}, n)) {
        const auto id = static_cast<std::int64_t>(u);
        initial[u] = id;
        frontier.push_back(FrontierEntry{.node = id, .distance = id});
    }

    auto rounds = run_rounds_seeded(reg, zero, std::move(initial), std::move(frontier),
                                    std::nullopt, 1, 1, Semiring::min_plus, shard_count, exec);
    if (!rounds) {
        return make_error<Labels>(rounds.error());
    }
    Labels labels(n, 0);
    for (const std::size_t u : std::views::iota(std::size_t{0}, n)) {
        if (!rounds->dist[u].has_value()) {
            // Every node was seeded, so an absent label means the loop lost one.
            return make_error<Labels>(MathError::not_converged);
        }
        labels[u] = *rounds->dist[u];
    }
    return labels;
}

auto distributed_strongly_connected_components(const WireGraph& g, std::size_t shard_count,
                                               Executor& exec) -> Result<std::vector<std::int64_t>> {
    using Labels = std::vector<std::int64_t>;
    auto reach = distributed_floyd_warshall(g, shard_count, exec);
    if (!reach) {
        return make_error<Labels>(reach.error());
    }
    const std::size_t n = reach->size();
    Labels labels(n, 0);
    for (const std::size_t u : std::views::iota(std::size_t{0}, n)) {
        std::int64_t smallest = static_cast<std::int64_t>(u);
        for (const std::size_t v : std::views::iota(std::size_t{0}, u)) {
            // Mutual reachability, which is the definition of the relation -- one direction alone
            // says only that v is downstream of u.
            if ((*reach)[u][v].has_value() && (*reach)[v][u].has_value()) {
                smallest = static_cast<std::int64_t>(v);
                break;  // v ascends, so the first match is the smallest
            }
        }
        labels[u] = smallest;
    }
    return labels;
}

auto distributed_topological_order(const WireGraph& g, std::size_t shard_count, Executor& exec)
    -> Result<std::vector<std::int64_t>> {
    using Order = std::vector<std::int64_t>;
    auto ok = validate(g);
    if (!ok) {
        return make_error<Order>(ok.error());
    }
    const std::size_t n = g.adjacency.size();
    if (shard_count == 0) {
        return make_error<Order>(MathError::domain_error);
    }
    if (n == 0) {
        return Order{};
    }
    TaskRegistry reg;
    auto reg_ok = register_ops(reg);
    if (!reg_ok) {
        return make_error<Order>(reg_ok.error());
    }

    // In-degree counts every edge, parallel ones included: two edges u -> v are two dependencies,
    // and the rounds below emit one proposal per edge, so both sides must count the same way.
    std::vector<std::int64_t> indegree(n, 0);
    for (const std::vector<Edge>& row : g.adjacency) {
        for (const Edge& e : row) {
            ++indegree[static_cast<std::size_t>(e.target)];
        }
    }

    Order order;
    order.reserve(n);
    std::vector<FrontierEntry> level;
    for (const std::size_t u : std::views::iota(std::size_t{0}, n)) {
        if (indegree[u] == 0) {
            level.push_back(FrontierEntry{.node = static_cast<std::int64_t>(u), .distance = 0});
        }
    }

    for ([[maybe_unused]] const std::size_t round : std::views::iota(std::size_t{0}, n + 1)) {
        if (level.empty()) {
            break;
        }
        for (const FrontierEntry& fe : level) {
            order.push_back(fe.node);
        }
        auto proposals = run_one_round(reg, g, level, shard_count, exec);
        if (!proposals) {
            return make_error<Order>(proposals.error());
        }
        for (const Proposal& p : *proposals) {
            if (p.target < 0 || static_cast<std::size_t>(p.target) >= n) {
                return make_error<Order>(MathError::domain_error);
            }
            --indegree[static_cast<std::size_t>(p.target)];
        }
        std::vector<FrontierEntry> next;
        for (const Proposal& p : *proposals) {
            const auto t = static_cast<std::size_t>(p.target);
            if (indegree[t] == 0) {
                // Guard against emitting a node twice when several edges into it drop to zero in
                // the same round: only the transition to zero enqueues, so mark it as taken.
                indegree[t] = -1;
                next.push_back(FrontierEntry{.node = p.target, .distance = 0});
            }
        }
        std::ranges::sort(next, {}, &FrontierEntry::node);
        level = std::move(next);
    }

    if (order.size() != n) {
        // Nodes remain with a non-zero in-degree, which happens exactly when they lie on a cycle.
        return make_error<Order>(MathError::undefined_value);
    }
    return order;
}

auto distributed_k_core(const WireGraph& g, std::int64_t k, std::size_t shard_count, Executor& exec)
    -> Result<std::vector<std::int64_t>> {
    using Nodes = std::vector<std::int64_t>;
    auto ok = validate(g);
    if (!ok) {
        return make_error<Nodes>(ok.error());
    }
    const std::size_t n = g.adjacency.size();
    if (shard_count == 0 || k < 0) {
        return make_error<Nodes>(MathError::domain_error);
    }
    if (n == 0) {
        return Nodes{};
    }
    TaskRegistry reg;
    auto reg_ok = register_ops(reg);
    if (!reg_ok) {
        return make_error<Nodes>(reg_ok.error());
    }

    // The simple undirected view: parallel edges once, self-loops not at all, which is what makes
    // "degree at least k" mean what a reader expects.
    WireGraph view = undirected_view(g, true);
    for (const std::size_t u : std::views::iota(std::size_t{0}, n)) {
        std::erase_if(view.adjacency[u],
                      [u](const Edge& e) { return e.target == static_cast<std::int64_t>(u); });
    }

    std::vector<std::int64_t> degree(n, 0);
    for (const std::size_t u : std::views::iota(std::size_t{0}, n)) {
        degree[u] = static_cast<std::int64_t>(view.adjacency[u].size());
    }
    std::vector<bool> removed(n, false);

    std::vector<FrontierEntry> wave;
    for (const std::size_t u : std::views::iota(std::size_t{0}, n)) {
        if (degree[u] < k) {
            removed[u] = true;
            wave.push_back(FrontierEntry{.node = static_cast<std::int64_t>(u), .distance = 0});
        }
    }

    for ([[maybe_unused]] const std::size_t round : std::views::iota(std::size_t{0}, n + 1)) {
        if (wave.empty()) {
            break;
        }
        auto proposals = run_one_round(reg, view, wave, shard_count, exec);
        if (!proposals) {
            return make_error<Nodes>(proposals.error());
        }
        std::vector<FrontierEntry> next;
        for (const Proposal& p : *proposals) {
            if (p.target < 0 || static_cast<std::size_t>(p.target) >= n) {
                return make_error<Nodes>(MathError::domain_error);
            }
            const auto t = static_cast<std::size_t>(p.target);
            if (removed[t]) {
                continue;  // already gone; its own edges were accounted for when it went
            }
            --degree[t];
            if (degree[t] < k) {
                removed[t] = true;
                next.push_back(FrontierEntry{.node = p.target, .distance = 0});
            }
        }
        std::ranges::sort(next, {}, &FrontierEntry::node);
        wave = std::move(next);
    }

    Nodes core;
    for (const std::size_t u : std::views::iota(std::size_t{0}, n)) {
        if (!removed[u]) {
            core.push_back(static_cast<std::int64_t>(u));
        }
    }
    return core;
}

auto distributed_triangle_counts(const WireGraph& g, std::size_t shard_count, Executor& exec)
    -> Result<std::vector<std::int64_t>> {
    using Counts = std::vector<std::int64_t>;
    auto ok = validate(g);
    if (!ok) {
        return make_error<Counts>(ok.error());
    }
    const std::size_t n = g.adjacency.size();
    if (shard_count == 0) {
        return make_error<Counts>(MathError::domain_error);
    }
    if (n == 0) {
        return Counts{};
    }
    TaskRegistry reg;
    auto reg_ok = register_ops(reg);
    if (!reg_ok) {
        return make_error<Counts>(reg_ok.error());
    }

    WireGraph view = undirected_view(g, true);
    for (const std::size_t u : std::views::iota(std::size_t{0}, n)) {
        std::erase_if(view.adjacency[u],
                      [u](const Edge& e) { return e.target == static_cast<std::int64_t>(u); });
    }
    auto whole = encode_slice(view, 0, n);
    if (!whole) {
        return make_error<Counts>(whole.error());
    }

    const std::vector<std::size_t> bounds = slice_bounds(n, shard_count);
    TaskGraph graph;
    for (const std::size_t s : std::views::iota(std::size_t{0}, shard_count)) {
        Payload range;
        put_u64(range, tri_magic);
        put_u64(range, static_cast<std::uint64_t>(bounds[s]));
        put_u64(range, static_cast<std::uint64_t>(bounds[s + 1] - bounds[s]));
        auto id = graph.add_named_task(reg, OpId{tri_op_id},
                                       std::vector<Payload>{*whole, std::move(range)});
        if (!id) {
            return make_error<Counts>(id.error());
        }
    }
    auto run = exec.run(graph);
    if (!run) {
        return make_error<Counts>(run.error());
    }

    Counts counts(n, 0);
    for (const Result<Payload>& outcome : run->outputs) {
        if (!outcome) {
            return make_error<Counts>(outcome.error());
        }
        std::size_t off = 0;
        const auto magic = take_u64(*outcome, off);
        const auto first64 = take_u64(*outcome, off);
        const auto len = take_u64(*outcome, off);
        if (!magic.has_value() || *magic != tri_magic || !first64.has_value() || !len.has_value()) {
            return make_error<Counts>(MathError::syntax_error);
        }
        const auto first = static_cast<std::size_t>(*first64);
        const auto count = static_cast<std::size_t>(*len);
        if (first > n || count > n - first) {
            return make_error<Counts>(MathError::syntax_error);
        }
        for (const std::size_t i : std::views::iota(std::size_t{0}, count)) {
            const auto v = take_i64(*outcome, off);
            if (!v.has_value()) {
                return make_error<Counts>(MathError::syntax_error);
            }
            counts[first + i] = *v;
        }
        if (off != outcome->size()) {
            return make_error<Counts>(MathError::syntax_error);
        }
    }
    return counts;
}

auto distributed_widest_path(const WireGraph& g, std::int64_t start, std::int64_t goal_node,
                             std::size_t shard_count, Executor& exec)
    -> Result<std::pair<std::vector<std::int64_t>, std::int64_t>> {
    using PathCost = std::pair<std::vector<std::int64_t>, std::int64_t>;
    auto ok = validate(g);
    if (!ok) {
        return make_error<PathCost>(ok.error());
    }
    const std::size_t n = g.adjacency.size();
    if (shard_count == 0 || start < 0 || static_cast<std::size_t>(start) >= n || goal_node < 0 ||
        static_cast<std::size_t>(goal_node) >= n) {
        return make_error<PathCost>(MathError::domain_error);
    }
    if (start == goal_node) {
        // An empty route has no narrow edge to limit it.
        return PathCost{std::vector<std::int64_t>{start}, std::numeric_limits<std::int64_t>::max()};
    }
    TaskRegistry reg;
    auto reg_ok = register_ops(reg);
    if (!reg_ok) {
        return make_error<PathCost>(reg_ok.error());
    }

    std::vector<std::optional<std::int64_t>> initial(n, std::nullopt);
    initial[static_cast<std::size_t>(start)] = std::numeric_limits<std::int64_t>::max();
    std::vector<FrontierEntry> frontier{FrontierEntry{
        .node = start, .distance = std::numeric_limits<std::int64_t>::max()}};

    auto rounds = run_rounds_seeded(reg, g, std::move(initial), std::move(frontier), std::nullopt,
                                    1, 1, Semiring::max_min, shard_count, exec);
    if (!rounds) {
        return make_error<PathCost>(rounds.error());
    }
    const std::optional<std::int64_t>& width = rounds->dist[static_cast<std::size_t>(goal_node)];
    if (!width.has_value()) {
        return make_error<PathCost>(MathError::undefined_value);
    }
    auto path = rebuild_path(rounds->parent, start, goal_node);
    if (!path) {
        return make_error<PathCost>(path.error());
    }
    return PathCost{std::move(*path), *width};
}

auto distributed_minimum_spanning_forest(const WireGraph& g, std::size_t shard_count,
                                         Executor& exec)
    -> Result<std::pair<std::vector<MstEdge>, std::int64_t>> {
    using Forest = std::pair<std::vector<MstEdge>, std::int64_t>;
    auto ok = validate(g);
    if (!ok) {
        return make_error<Forest>(ok.error());
    }
    const std::size_t n = g.adjacency.size();
    if (shard_count == 0) {
        return make_error<Forest>(MathError::domain_error);
    }
    if (n == 0) {
        return Forest{std::vector<MstEdge>{}, 0};
    }
    TaskRegistry reg;
    auto reg_ok = register_ops(reg);
    if (!reg_ok) {
        return make_error<Forest>(reg_ok.error());
    }

    const WireGraph view = undirected_view(g, false);

    // Union-find over components, coordinator-side. The shards only ever READ the labels, so the
    // merge stays in one place and cannot race.
    std::vector<std::int64_t> parent(n);
    std::iota(parent.begin(), parent.end(), std::int64_t{0});
    const auto find = [&parent](std::int64_t x) -> std::int64_t {
        while (parent[static_cast<std::size_t>(x)] != x) {
            parent[static_cast<std::size_t>(x)] =
                parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
            x = parent[static_cast<std::size_t>(x)];
        }
        return x;
    };

    std::vector<MstEdge> chosen;
    std::int64_t total = 0;

    // Components at least halve each round, so log2(n) + 1 rounds suffice; the extra guard turns a
    // broken invariant into an error instead of a spin.
    for ([[maybe_unused]] const std::size_t round : std::views::iota(std::size_t{0}, n + 1)) {
        std::vector<std::int64_t> labels(n, 0);
        for (const std::size_t u : std::views::iota(std::size_t{0}, n)) {
            labels[u] = find(static_cast<std::int64_t>(u));
        }

        Payload label_payload;
        put_u64(label_payload, mst_magic);
        put_u64(label_payload, static_cast<std::uint64_t>(n));
        for (const std::int64_t l : labels) {
            put_i64(label_payload, l);
        }

        const std::vector<std::size_t> bounds = slice_bounds(n, shard_count);
        TaskGraph graph;
        for (const std::size_t s : std::views::iota(std::size_t{0}, shard_count)) {
            auto slice = encode_slice(view, bounds[s], bounds[s + 1] - bounds[s]);
            if (!slice) {
                return make_error<Forest>(slice.error());
            }
            auto id = graph.add_named_task(
                reg, OpId{mst_op_id}, std::vector<Payload>{std::move(*slice), label_payload});
            if (!id) {
                return make_error<Forest>(id.error());
            }
        }
        auto run = exec.run(graph);
        if (!run) {
            return make_error<Forest>(run.error());
        }

        std::map<std::int64_t, Candidate> best;
        for (const Result<Payload>& outcome : run->outputs) {
            if (!outcome) {
                return make_error<Forest>(outcome.error());
            }
            auto cands = decode_candidates(*outcome);
            if (!cands) {
                return make_error<Forest>(cands.error());
            }
            for (const Candidate& c : *cands) {
                auto it = best.find(c.component);
                if (it == best.end() || c.better_than(it->second)) {
                    best[c.component] = c;
                }
            }
        }
        if (best.empty()) {
            break;  // no component has an outgoing edge: the forest is complete
        }

        bool merged_any = false;
        for (const auto& [comp, cand] : best) {
            const std::int64_t ru = find(cand.u);
            const std::int64_t rv = find(cand.v);
            if (ru == rv) {
                continue;  // the other end of this edge already merged us this round
            }
            parent[static_cast<std::size_t>(ru)] = rv;
            std::int64_t next_total = 0;
            if (add_overflows(total, cand.weight, next_total)) {
                return make_error<Forest>(MathError::overflow);
            }
            total = next_total;
            chosen.push_back(MstEdge{.u = std::min(cand.u, cand.v),
                                     .v = std::max(cand.u, cand.v),
                                     .weight = cand.weight});
            merged_any = true;
        }
        if (!merged_any) {
            break;
        }
    }

    std::ranges::sort(chosen, [](const MstEdge& a, const MstEdge& b) {
        return std::tie(a.u, a.v) < std::tie(b.u, b.v);
    });
    return Forest{std::move(chosen), total};
}
}  // namespace nimblecas::search_dist
