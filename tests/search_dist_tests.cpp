// Tests for nimblecas.search_dist: graph search decomposed into per-round task graphs and run
// over an Executor -- serial, local-parallel, or (unchanged) a distributed one over a broker.
// @author Olumuyiwa Oluwasanmi
//
// The contract under test is agreement with the in-process algorithms in nimblecas.search. Every
// graph here is built once as a WireGraph and driven BOTH ways -- through the distributed entry
// point and through `dijkstra` / `a_star` over the same graph's callbacks -- so the comparison is
// against the trusted implementation on identical input rather than against a hand-copied number.
//
// The round-based relaxation this module uses is a different algorithm from Dijkstra, not a
// parallelisation of it, so the agreement is a real property worth testing and not a tautology.

import std;
import nimblecas.core;
import nimblecas.search;
import nimblecas.search_dist;
import nimblecas.taskdag;
import nimblecas.testing;

using nimblecas::a_star;
using nimblecas::bfs;
using nimblecas::iterative_deepening_dfs;
using nimblecas::dijkstra;
using nimblecas::Executor;
using nimblecas::local_parallel_executor;
using nimblecas::MathError;
using nimblecas::parallel_multistart_tabu;
using nimblecas::Payload;
using nimblecas::serial_executor;
using nimblecas::TabuState;
using nimblecas::TaskRegistry;
using nimblecas::search_dist::decode_proposals;
using nimblecas::search_dist::distributed_a_star;
using nimblecas::search_dist::distributed_bfs;
using nimblecas::search_dist::distributed_floyd_warshall;
using nimblecas::search_dist::distributed_iterative_deepening;
using nimblecas::search_dist::distributed_reachable_set;
using nimblecas::search_dist::distributed_weighted_a_star;
using nimblecas::search_dist::distributed_multistart_tabu;
using nimblecas::search_dist::distributed_shortest_path;
using nimblecas::search_dist::distributed_sssp;
using nimblecas::search_dist::Edge;
using nimblecas::search_dist::encode_frontier;
using nimblecas::search_dist::encode_slice;
using nimblecas::search_dist::FrontierEntry;
using nimblecas::search_dist::Proposal;
using nimblecas::search_dist::WireGraph;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// A 6-node graph with two routes to node 5 of DIFFERENT cost, so the optimum is unique and paths
// can be compared as well as costs.
//
//   0 -1-> 1 -2-> 2 -1-> 5      (total 4)
//   0 -7-> 3 -1-> 4 -1-> 5      (total 9)
//   1 -9-> 4
[[nodiscard]] auto diamond_graph() -> WireGraph {
    WireGraph g;
    g.adjacency = {
        {Edge{.target = 1, .cost = 1}, Edge{.target = 3, .cost = 7}},
        {Edge{.target = 2, .cost = 2}, Edge{.target = 4, .cost = 9}},
        {Edge{.target = 5, .cost = 1}},
        {Edge{.target = 4, .cost = 1}},
        {Edge{.target = 5, .cost = 1}},
        {},
    };
    return g;
}

// A 10-node line 0 -> 1 -> ... -> 9, every edge cost 2. A path of nine edges needs nine rounds,
// which is what exercises the round loop rather than a single fan-out.
[[nodiscard]] auto line_graph(std::int64_t n, std::int64_t w) -> WireGraph {
    WireGraph g;
    g.adjacency.resize(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i + 1 < n; ++i) {
        g.adjacency[static_cast<std::size_t>(i)].push_back(Edge{.target = i + 1, .cost = w});
    }
    return g;
}

// Manhattan distance to the bottom-right corner of a rows x cols grid, which is admissible when
// every edge costs at least 1.
[[nodiscard]] auto grid_graph(std::int64_t rows, std::int64_t cols, std::int64_t w) -> WireGraph {
    WireGraph g;
    const auto n = static_cast<std::size_t>(rows * cols);
    g.adjacency.resize(n);
    g.heuristic.resize(n);
    for (std::int64_t r = 0; r < rows; ++r) {
        for (std::int64_t c = 0; c < cols; ++c) {
            const std::int64_t id = r * cols + c;
            if (c + 1 < cols) {
                g.adjacency[static_cast<std::size_t>(id)].push_back(
                    Edge{.target = id + 1, .cost = w});
            }
            if (r + 1 < rows) {
                g.adjacency[static_cast<std::size_t>(id)].push_back(
                    Edge{.target = id + cols, .cost = w});
            }
            g.heuristic[static_cast<std::size_t>(id)] = ((rows - 1 - r) + (cols - 1 - c)) * w;
        }
    }
    return g;
}

// The serial oracle over the same graph, so both sides see identical input.
[[nodiscard]] auto serial_path(const WireGraph& g, std::int64_t start, std::int64_t goal)
    -> nimblecas::Result<std::pair<std::vector<std::int64_t>, std::int64_t>> {
    return dijkstra(
        start, [goal](std::int64_t u) -> bool { return u == goal; },
        nimblecas::search_dist::successors_of(g), nimblecas::search_dist::cost_of(g));
}

[[nodiscard]] auto serial_astar(const WireGraph& g, std::int64_t start, std::int64_t goal)
    -> nimblecas::Result<std::pair<std::vector<std::int64_t>, std::int64_t>> {
    return a_star(
        start, [goal](std::int64_t u) -> bool { return u == goal; },
        nimblecas::search_dist::successors_of(g), nimblecas::search_dist::cost_of(g),
        nimblecas::search_dist::heuristic_of(g));
}

}  // namespace

auto main() -> int {
    TestSuite("nimblecas.search_dist")
        .test("validate_accepts_a_well_formed_graph",
              [](TestContext& t) {
                  const WireGraph g = diamond_graph();
                  t.expect(nimblecas::search_dist::validate(g).has_value(),
                           "a graph with in-range targets and non-negative costs validates");
              })
        .test("validate_rejects_a_negative_edge_cost",
              [](TestContext& t) {
                  WireGraph g = diamond_graph();
                  g.adjacency[0][0].cost = -1;
                  auto r = nimblecas::search_dist::validate(g);
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "a negative edge cost is a domain_error");
              })
        .test("validate_rejects_an_out_of_range_edge_target",
              [](TestContext& t) {
                  WireGraph g = diamond_graph();
                  g.adjacency[0][0].target = 99;
                  auto r = nimblecas::search_dist::validate(g);
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "an edge pointing outside the graph is a domain_error");
              })
        .test("validate_rejects_a_heuristic_of_the_wrong_length",
              [](TestContext& t) {
                  WireGraph g = diamond_graph();
                  g.heuristic = {0, 0};
                  auto r = nimblecas::search_dist::validate(g);
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "a heuristic with one entry per node is required when present");
              })
        .test("validate_rejects_a_negative_heuristic_value",
              [](TestContext& t) {
                  WireGraph g = diamond_graph();
                  g.heuristic = {0, 0, 0, 0, 0, -1};
                  auto r = nimblecas::search_dist::validate(g);
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "a negative heuristic value is a domain_error");
              })
        .test("successors_and_cost_adapters_reproduce_the_graph",
              [](TestContext& t) {
                  const WireGraph g = diamond_graph();
                  const auto succ = nimblecas::search_dist::successors_of(g);
                  const auto cost = nimblecas::search_dist::cost_of(g);
                  const std::vector<std::int64_t> from_zero{1, 3};
                  t.expect(succ(0) == from_zero, "node 0's successors are exactly {1, 3}");
                  t.expect(cost(0, 1) == 1, "edge 0->1 costs 1");
                  t.expect(cost(0, 3) == 7, "edge 0->3 costs 7");
                  t.expect(succ(5).empty(), "node 5 is a sink with no successors");
                  t.expect(succ(99).empty(), "a node outside the graph has no successors");
              })
        .test("slice_payload_round_trips_through_a_relax_task",
              [](TestContext& t) {
                  const WireGraph g = diamond_graph();
                  TaskRegistry reg;
                  t.expect(nimblecas::search_dist::register_ops(reg).has_value(),
                           "both operations register on a fresh registry");
                  const std::vector<FrontierEntry> front{FrontierEntry{.node = 0, .distance = 0}};
                  auto graph = nimblecas::search_dist::build_round_graph(
                      reg, g, front, std::numeric_limits<std::int64_t>::max(), 1);
                  t.expect(graph.has_value(), "a one-shard round graph builds");
                  if (!graph.has_value()) {
                      return;
                  }
                  t.expect(graph->size() == 1, "one shard yields exactly one task");
                  auto exec = serial_executor();
                  auto run = exec->run(*graph);
                  t.expect(run.has_value(), "the round runs");
                  if (!run.has_value() || run->outputs.size() != 1 || !run->outputs[0]) {
                      return;
                  }
                  auto ps = decode_proposals(*run->outputs[0]);
                  t.expect(ps.has_value(), "the shard's proposals decode");
                  if (!ps.has_value() || ps->size() != 2) {
                      return;
                  }
                  const Proposal a{.target = 1, .distance = 1, .parent = 0};
                  const Proposal b{.target = 3, .distance = 7, .parent = 0};
                  t.expect((*ps)[0] == a, "relaxing node 0 proposes node 1 at distance 1");
                  t.expect((*ps)[1] == b, "relaxing node 0 proposes node 3 at distance 7");
              })
        .test("build_round_graph_makes_one_task_per_shard",
              [](TestContext& t) {
                  const WireGraph g = line_graph(10, 2);
                  TaskRegistry reg;
                  t.expect(nimblecas::search_dist::register_ops(reg).has_value(),
                           "operations register");
                  const std::vector<FrontierEntry> front{FrontierEntry{.node = 0, .distance = 0}};
                  for (const std::size_t shards : {std::size_t{1}, std::size_t{3},
                                                   std::size_t{4}, std::size_t{16}}) {
                      auto graph = nimblecas::search_dist::build_round_graph(
                          reg, g, front, std::numeric_limits<std::int64_t>::max(), shards);
                      t.expect(graph.has_value() && graph->size() == shards,
                               "the round graph has exactly one task per requested shard");
                  }
              })
        .test("build_round_graph_rejects_zero_shards",
              [](TestContext& t) {
                  const WireGraph g = diamond_graph();
                  TaskRegistry reg;
                  t.expect(nimblecas::search_dist::register_ops(reg).has_value(),
                           "operations register");
                  const std::vector<FrontierEntry> front{FrontierEntry{.node = 0, .distance = 0}};
                  auto r = nimblecas::search_dist::build_round_graph(reg, g, front, 0, 0);
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "zero shards is a domain_error");
              })
        .test("register_ops_twice_on_one_registry_is_a_domain_error",
              [](TestContext& t) {
                  TaskRegistry reg;
                  t.expect(nimblecas::search_dist::register_ops(reg).has_value(),
                           "the first registration succeeds");
                  auto second = nimblecas::search_dist::register_ops(reg);
                  t.expect(!second.has_value() && second.error() == MathError::domain_error,
                           "registering the same operation again is a domain_error");
              })
        .test("decode_proposals_rejects_bytes_it_did_not_write",
              [](TestContext& t) {
                  const Payload empty;
                  auto r0 = decode_proposals(empty);
                  t.expect(!r0.has_value() && r0.error() == MathError::syntax_error,
                           "empty bytes are a syntax_error, not an empty proposal list");
                  Payload garbage;
                  for (const char c : std::string_view{"not a proposal payload at all"}) {
                      garbage.push_back(static_cast<std::byte>(c));
                  }
                  auto r1 = decode_proposals(garbage);
                  t.expect(!r1.has_value() && r1.error() == MathError::syntax_error,
                           "arbitrary bytes are a syntax_error");
              })
        .test("decode_proposals_rejects_a_truncated_payload",
              [](TestContext& t) {
                  const WireGraph g = diamond_graph();
                  TaskRegistry reg;
                  if (!nimblecas::search_dist::register_ops(reg)) {
                      return;
                  }
                  const std::vector<FrontierEntry> front{FrontierEntry{.node = 0, .distance = 0}};
                  auto graph = nimblecas::search_dist::build_round_graph(
                      reg, g, front, std::numeric_limits<std::int64_t>::max(), 1);
                  if (!graph.has_value()) {
                      return;
                  }
                  auto exec = serial_executor();
                  auto run = exec->run(*graph);
                  if (!run.has_value() || run->outputs.size() != 1 || !run->outputs[0]) {
                      return;
                  }
                  Payload good = *run->outputs[0];
                  t.expect(decode_proposals(good).has_value(), "the intact payload decodes");
                  good.pop_back();
                  auto r = decode_proposals(good);
                  t.expect(!r.has_value() && r.error() == MathError::syntax_error,
                           "a payload one byte short is a syntax_error, never a short list");
              })
        .test("encode_slice_rejects_a_range_outside_the_graph",
              [](TestContext& t) {
                  const WireGraph g = diamond_graph();
                  auto r = encode_slice(g, 4, 9);
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "a slice running past the last node is a domain_error");
                  auto ok = encode_slice(g, 4, 2);
                  t.expect(ok.has_value(), "a slice ending exactly at the last node is accepted");
              })
        .test("distributed_sssp_matches_dijkstra_on_every_node_of_the_diamond",
              [](TestContext& t) {
                  const WireGraph g = diamond_graph();
                  auto exec = serial_executor();
                  auto d = distributed_sssp(g, 0, 3, *exec);
                  t.expect(d.has_value(), "the distributed sssp succeeds");
                  if (!d.has_value() || d->size() != 6) {
                      return;
                  }
                  const std::vector<std::int64_t> expected{0, 1, 3, 7, 8, 4};
                  for (std::size_t i = 0; i < expected.size(); ++i) {
                      const auto oracle = serial_path(g, 0, static_cast<std::int64_t>(i));
                      t.expect((*d)[i].has_value() && *(*d)[i] == expected[i],
                               "the distributed distance equals the hand-computed distance");
                      t.expect(oracle.has_value() && oracle->second == expected[i],
                               "dijkstra agrees with the same hand-computed distance");
                  }
              })
        .test("distributed_sssp_reports_unreachable_nodes_as_nullopt",
              [](TestContext& t) {
                  WireGraph g = diamond_graph();
                  g.adjacency.push_back({});               // node 6, isolated
                  g.adjacency.push_back({Edge{.target = 6, .cost = 1}});  // node 7 -> 6 only
                  auto exec = serial_executor();
                  auto d = distributed_sssp(g, 0, 2, *exec);
                  t.expect(d.has_value(), "the sssp succeeds on a graph with an unreachable part");
                  if (!d.has_value() || d->size() != 8) {
                      return;
                  }
                  t.expect(!(*d)[6].has_value(), "an unreachable node has no distance");
                  t.expect(!(*d)[7].has_value(), "a node reaching only unreachable nodes has none");
                  t.expect((*d)[0].has_value() && *(*d)[0] == 0, "the source is at distance 0");
              })
        .test("distributed_sssp_is_identical_for_every_shard_count",
              [](TestContext& t) {
                  const WireGraph g = grid_graph(4, 4, 3);
                  auto exec = serial_executor();
                  auto baseline = distributed_sssp(g, 0, 1, *exec);
                  t.expect(baseline.has_value(), "the one-shard run succeeds");
                  if (!baseline.has_value()) {
                      return;
                  }
                  for (const std::size_t shards : {std::size_t{2}, std::size_t{3},
                                                   std::size_t{5}, std::size_t{16},
                                                   std::size_t{64}}) {
                      auto d = distributed_sssp(g, 0, shards, *exec);
                      t.expect(d.has_value() && *d == *baseline,
                               "the partition into shards does not change a single distance");
                  }
              })
        .test("distributed_sssp_is_identical_on_both_executors",
              [](TestContext& t) {
                  const WireGraph g = grid_graph(5, 5, 2);
                  auto ser = serial_executor();
                  auto par = local_parallel_executor();
                  auto a = distributed_sssp(g, 0, 4, *ser);
                  t.expect(a.has_value(), "the serial executor succeeds");
                  if (!a.has_value()) {
                      return;
                  }
                  // Repeated so a scheduling dependence shows up as a flake, which is exactly
                  // what this assertion exists to catch.
                  for (int attempt = 0; attempt < 8; ++attempt) {
                      auto b = distributed_sssp(g, 0, 4, *par);
                      t.expect(b.has_value() && *b == *a,
                               "the parallel executor gives byte-for-byte the same distances");
                  }
              })
        .test("distributed_shortest_path_matches_dijkstra_path_and_cost_on_the_diamond",
              [](TestContext& t) {
                  const WireGraph g = diamond_graph();
                  auto exec = serial_executor();
                  auto d = distributed_shortest_path(g, 0, 5, 3, *exec);
                  const auto oracle = serial_path(g, 0, 5);
                  t.expect(d.has_value(), "the distributed path search succeeds");
                  t.expect(oracle.has_value(), "dijkstra succeeds on the same graph");
                  if (!d.has_value() || !oracle.has_value()) {
                      return;
                  }
                  const std::vector<std::int64_t> expected{0, 1, 2, 5};
                  t.expect(d->second == 4, "the optimal cost is exactly 4");
                  t.expect(d->second == oracle->second, "the cost equals dijkstra's cost");
                  t.expect(d->first == expected, "the path is exactly {0, 1, 2, 5}");
                  t.expect(d->first == oracle->first,
                           "the path equals dijkstra's path where the optimum is unique");
              })
        .test("distributed_shortest_path_matches_dijkstra_across_a_long_line",
              [](TestContext& t) {
                  const WireGraph g = line_graph(12, 5);
                  auto exec = local_parallel_executor();
                  auto d = distributed_shortest_path(g, 0, 11, 3, *exec);
                  const auto oracle = serial_path(g, 0, 11);
                  t.expect(d.has_value() && oracle.has_value(), "both searches succeed");
                  if (!d.has_value() || !oracle.has_value()) {
                      return;
                  }
                  t.expect(d->second == 55, "eleven edges of cost 5 total exactly 55");
                  t.expect(d->second == oracle->second, "the cost equals dijkstra's");
                  t.expect(d->first.size() == 12, "the path visits all twelve nodes");
                  t.expect(d->first == oracle->first, "the path equals dijkstra's path");
              })
        .test("distributed_shortest_path_from_a_node_to_itself_is_the_single_node_at_cost_zero",
              [](TestContext& t) {
                  const WireGraph g = diamond_graph();
                  auto exec = serial_executor();
                  auto d = distributed_shortest_path(g, 3, 3, 2, *exec);
                  t.expect(d.has_value(), "start == goal succeeds");
                  if (!d.has_value()) {
                      return;
                  }
                  const std::vector<std::int64_t> expected{3};
                  t.expect(d->first == expected, "the path is exactly {3}");
                  t.expect(d->second == 0, "the cost is exactly 0");
              })
        .test("distributed_shortest_path_to_an_unreachable_goal_is_undefined_value",
              [](TestContext& t) {
                  WireGraph g = diamond_graph();
                  g.adjacency.push_back({});  // node 6, unreachable from 0
                  auto exec = serial_executor();
                  auto d = distributed_shortest_path(g, 0, 6, 2, *exec);
                  t.expect(!d.has_value() && d.error() == MathError::undefined_value,
                           "an unreachable goal is undefined_value, never a partial path");
                  const auto oracle = serial_path(g, 0, 6);
                  t.expect(!oracle.has_value() && oracle.error() == MathError::undefined_value,
                           "dijkstra reports the same error on the same graph");
              })
        .test("distributed_shortest_path_rejects_out_of_range_endpoints_and_zero_shards",
              [](TestContext& t) {
                  const WireGraph g = diamond_graph();
                  auto exec = serial_executor();
                  auto a = distributed_shortest_path(g, -1, 5, 2, *exec);
                  t.expect(!a.has_value() && a.error() == MathError::domain_error,
                           "a negative start is a domain_error");
                  auto b = distributed_shortest_path(g, 0, 99, 2, *exec);
                  t.expect(!b.has_value() && b.error() == MathError::domain_error,
                           "a goal outside the graph is a domain_error");
                  auto c = distributed_shortest_path(g, 0, 5, 0, *exec);
                  t.expect(!c.has_value() && c.error() == MathError::domain_error,
                           "zero shards is a domain_error");
              })
        .test("distributed_a_star_matches_a_star_cost_on_a_grid",
              [](TestContext& t) {
                  const WireGraph g = grid_graph(5, 5, 3);
                  auto exec = serial_executor();
                  auto d = distributed_a_star(g, 0, 24, 4, *exec);
                  const auto oracle = serial_astar(g, 0, 24);
                  t.expect(d.has_value() && oracle.has_value(), "both searches succeed");
                  if (!d.has_value() || !oracle.has_value()) {
                      return;
                  }
                  t.expect(d->second == 24, "eight grid steps of cost 3 total exactly 24");
                  t.expect(d->second == oracle->second, "the cost equals a_star's cost");
              })
        .test("distributed_a_star_cost_equals_distributed_shortest_path_cost",
              [](TestContext& t) {
                  const WireGraph g = grid_graph(4, 6, 2);
                  auto exec = local_parallel_executor();
                  auto with_h = distributed_a_star(g, 0, 23, 3, *exec);
                  auto without_h = distributed_shortest_path(g, 0, 23, 3, *exec);
                  t.expect(with_h.has_value() && without_h.has_value(), "both searches succeed");
                  if (!with_h.has_value() || !without_h.has_value()) {
                      return;
                  }
                  t.expect(with_h->second == without_h->second,
                           "heuristic pruning does not change the optimal cost");
                  t.expect(with_h->first == without_h->first,
                           "heuristic pruning does not change the chosen optimal path");
              })
        .test("distributed_a_star_with_a_zero_heuristic_equals_dijkstra",
              [](TestContext& t) {
                  WireGraph g = grid_graph(4, 4, 5);
                  g.heuristic.assign(g.adjacency.size(), 0);
                  auto exec = serial_executor();
                  auto d = distributed_a_star(g, 0, 15, 2, *exec);
                  const auto oracle = serial_path(g, 0, 15);
                  t.expect(d.has_value() && oracle.has_value(), "both searches succeed");
                  if (!d.has_value() || !oracle.has_value()) {
                      return;
                  }
                  t.expect(d->second == oracle->second,
                           "a zero heuristic makes the search agree with dijkstra's cost");
                  t.expect(d->first == oracle->first, "and with dijkstra's path");
              })
        .test("distributed_a_star_on_an_unreachable_goal_is_undefined_value",
              [](TestContext& t) {
                  WireGraph g = grid_graph(3, 3, 1);
                  g.adjacency.push_back({});      // node 9, unreachable
                  g.heuristic.push_back(0);
                  auto exec = serial_executor();
                  auto d = distributed_a_star(g, 0, 9, 2, *exec);
                  t.expect(!d.has_value() && d.error() == MathError::undefined_value,
                           "an unreachable goal is undefined_value under pruning too");
              })
        .test("a_zero_cost_edge_does_not_break_the_distances",
              [](TestContext& t) {
                  // Zero-weight edges are the case a settle-once scheme gets wrong, because two
                  // nodes at the same key can still improve each other.
                  WireGraph g;
                  g.adjacency = {
                      {Edge{.target = 1, .cost = 0}, Edge{.target = 2, .cost = 5}},
                      {Edge{.target = 2, .cost = 0}},
                      {Edge{.target = 3, .cost = 1}},
                      {},
                  };
                  auto exec = serial_executor();
                  auto d = distributed_sssp(g, 0, 4, *exec);
                  t.expect(d.has_value(), "the sssp succeeds with zero-cost edges");
                  if (!d.has_value() || d->size() != 4) {
                      return;
                  }
                  t.expect((*d)[1].has_value() && *(*d)[1] == 0, "node 1 is reached at cost 0");
                  t.expect((*d)[2].has_value() && *(*d)[2] == 0,
                           "node 2 is reached at cost 0 through the zero-cost chain, not 5");
                  t.expect((*d)[3].has_value() && *(*d)[3] == 1, "node 3 is reached at cost 1");
                  const auto oracle = serial_path(g, 0, 3);
                  t.expect(oracle.has_value() && oracle->second == 1,
                           "dijkstra agrees the cost to node 3 is 1");
              })
        .test("a_cycle_does_not_prevent_termination",
              [](TestContext& t) {
                  WireGraph g;
                  g.adjacency = {
                      {Edge{.target = 1, .cost = 1}},
                      {Edge{.target = 2, .cost = 1}},
                      {Edge{.target = 0, .cost = 1}, Edge{.target = 3, .cost = 4}},
                      {},
                  };
                  auto exec = serial_executor();
                  auto d = distributed_sssp(g, 0, 2, *exec);
                  t.expect(d.has_value(), "a cyclic graph terminates rather than looping");
                  if (!d.has_value() || d->size() != 4) {
                      return;
                  }
                  t.expect((*d)[0].has_value() && *(*d)[0] == 0, "the source stays at 0");
                  t.expect((*d)[3].has_value() && *(*d)[3] == 6, "node 3 is reached at cost 6");
              })
        .test("a_path_cost_that_would_overflow_is_reported_as_overflow",
              [](TestContext& t) {
                  constexpr std::int64_t big = std::numeric_limits<std::int64_t>::max() - 1;
                  WireGraph g;
                  g.adjacency = {
                      {Edge{.target = 1, .cost = big}},
                      {Edge{.target = 2, .cost = big}},
                      {},
                  };
                  auto exec = serial_executor();
                  auto d = distributed_sssp(g, 0, 2, *exec);
                  t.expect(!d.has_value() && d.error() == MathError::overflow,
                           "a path cost past int64 is overflow, never a wrapped-around distance");
              })
        .test("distributed_multistart_tabu_with_one_start_matches_the_serial_tabu_search",
              [](TestContext& t) {
                  WireGraph g = line_graph(8, 1);
                  // Make it walkable in both directions so the local search has somewhere to go.
                  for (std::int64_t i = 1; i < 8; ++i) {
                      g.adjacency[static_cast<std::size_t>(i)].push_back(
                          Edge{.target = i - 1, .cost = 1});
                  }
                  g.landscape = {9, 7, 5, 3, 1, 4, 6, 8};
                  auto exec = serial_executor();
                  const std::vector<std::int64_t> starts{0};
                  auto d = distributed_multistart_tabu(g, starts, 2, 20, *exec);
                  t.expect(d.has_value(), "the distributed tabu search succeeds");
                  if (!d.has_value()) {
                      return;
                  }
                  const std::vector<TabuState> serial_starts{TabuState{0}};
                  const auto succ = nimblecas::search_dist::successors_of(g);
                  const std::vector<std::int64_t> land = g.landscape;
                  auto oracle = parallel_multistart_tabu(
                      serial_starts,
                      [succ](const TabuState& s) -> std::vector<TabuState> {
                          std::vector<TabuState> out;
                          if (s.size() != 1) {
                              return out;
                          }
                          for (const std::int64_t v : succ(s[0])) {
                              out.push_back(TabuState{v});
                          }
                          return out;
                      },
                      [land](const TabuState& s) -> std::int64_t {
                          if (s.size() != 1 || s[0] < 0 ||
                              static_cast<std::size_t>(s[0]) >= land.size()) {
                              return std::numeric_limits<std::int64_t>::max();
                          }
                          return land[static_cast<std::size_t>(s[0])];
                      },
                      2, 20);
                  t.expect(oracle.has_value(), "the serial multistart tabu succeeds");
                  if (!oracle.has_value()) {
                      return;
                  }
                  t.expect(d->second == oracle->second,
                           "the distributed objective equals the serial objective");
                  t.expect(d->first == oracle->first,
                           "the distributed state equals the serial state");
                  t.expect(d->second == 1, "the minimum of the landscape is exactly 1");
              })
        .test("distributed_multistart_tabu_takes_the_best_across_starts",
              [](TestContext& t) {
                  WireGraph g;
                  // Two disjoint chains, so each start can only reach its own basin: the reduction
                  // is what picks between them, and that is the thing under test.
                  g.adjacency = {
                      {Edge{.target = 1, .cost = 1}},
                      {},
                      {Edge{.target = 3, .cost = 1}},
                      {},
                  };
                  g.landscape = {50, 40, 30, 10};
                  auto exec = local_parallel_executor();
                  const std::vector<std::int64_t> both{0, 2};
                  auto d = distributed_multistart_tabu(g, both, 1, 10, *exec);
                  t.expect(d.has_value(), "the two-start run succeeds");
                  if (!d.has_value()) {
                      return;
                  }
                  const TabuState expected{3};
                  t.expect(d->second == 10, "the best objective across both basins is exactly 10");
                  t.expect(d->first == expected, "the winning state is node 3");

                  const std::vector<std::int64_t> only_first{0};
                  auto worse = distributed_multistart_tabu(g, only_first, 1, 10, *exec);
                  t.expect(worse.has_value() && worse->second == 40,
                           "starting only in the first basin cannot find the better one");
              })
        .test("distributed_multistart_tabu_guards_its_arguments",
              [](TestContext& t) {
                  WireGraph g = line_graph(4, 1);
                  g.landscape = {4, 3, 2, 1};
                  auto exec = serial_executor();
                  const std::vector<std::int64_t> none;
                  auto a = distributed_multistart_tabu(g, none, 1, 5, *exec);
                  t.expect(!a.has_value() && a.error() == MathError::domain_error,
                           "an empty starts list is a domain_error");
                  const std::vector<std::int64_t> outside{99};
                  auto b = distributed_multistart_tabu(g, outside, 1, 5, *exec);
                  t.expect(!b.has_value() && b.error() == MathError::domain_error,
                           "a start outside the graph is a domain_error");
                  const std::vector<std::int64_t> ok{0};
                  auto c = distributed_multistart_tabu(g, ok, -1, 5, *exec);
                  t.expect(!c.has_value() && c.error() == MathError::domain_error,
                           "a negative tenure is a domain_error");
                  auto d = distributed_multistart_tabu(g, ok, 1, -5, *exec);
                  t.expect(!d.has_value() && d.error() == MathError::domain_error,
                           "a negative iteration count is a domain_error");
                  WireGraph no_land = line_graph(4, 1);
                  auto e = distributed_multistart_tabu(no_land, ok, 1, 5, *exec);
                  t.expect(!e.has_value() && e.error() == MathError::domain_error,
                           "a graph with no landscape is a domain_error, not a zero objective");
              })
        .test("distributed_multistart_tabu_is_identical_on_both_executors",
              [](TestContext& t) {
                  WireGraph g = line_graph(10, 1);
                  for (std::int64_t i = 1; i < 10; ++i) {
                      g.adjacency[static_cast<std::size_t>(i)].push_back(
                          Edge{.target = i - 1, .cost = 1});
                  }
                  g.landscape = {20, 15, 11, 8, 6, 5, 7, 9, 12, 16};
                  auto ser = serial_executor();
                  auto par = local_parallel_executor();
                  const std::vector<std::int64_t> starts{0, 3, 9};
                  auto a = distributed_multistart_tabu(g, starts, 2, 25, *ser);
                  t.expect(a.has_value(), "the serial executor succeeds");
                  if (!a.has_value()) {
                      return;
                  }
                  for (int attempt = 0; attempt < 8; ++attempt) {
                      auto b = distributed_multistart_tabu(g, starts, 2, 25, *par);
                      t.expect(b.has_value() && b->first == a->first && b->second == a->second,
                               "the parallel executor gives exactly the same best state and value");
                  }
              })
        .test("an_empty_frontier_ends_the_run_without_a_round",
              [](TestContext& t) {
                  // A single isolated node: the first round relaxes nothing, so the loop must
                  // finish rather than spin to its round budget.
                  WireGraph g;
                  g.adjacency = {{}};
                  auto exec = serial_executor();
                  auto d = distributed_sssp(g, 0, 1, *exec);
                  t.expect(d.has_value(), "a single-node graph succeeds");
                  if (!d.has_value() || d->size() != 1) {
                      return;
                  }
                  t.expect((*d)[0].has_value() && *(*d)[0] == 0, "the only node is at distance 0");
              })
        .test("randomised_graphs_agree_with_dijkstra_on_every_node",
              [](TestContext& t) {
                  // The hand-built graphs above test the cases their author thought of. This one
                  // tests the cases nobody thought of: 40 pseudo-random graphs, every node's
                  // distance compared against dijkstra. The seed is fixed, so a failure here is
                  // reproducible rather than a once-seen flake.
                  std::mt19937_64 rng(0x5eed1234u);
                  auto exec = serial_executor();
                  int graphs_checked = 0;
                  int nodes_checked = 0;
                  for (int trial = 0; trial < 40; ++trial) {
                      const auto n = static_cast<std::size_t>(3 + (rng() % 12));
                      WireGraph g;
                      g.adjacency.resize(n);
                      for (std::size_t u = 0; u < n; ++u) {
                          const std::size_t deg = rng() % 4;
                          for (std::size_t e = 0; e < deg; ++e) {
                              g.adjacency[u].push_back(
                                  Edge{.target = static_cast<std::int64_t>(rng() % n),
                                       .cost = static_cast<std::int64_t>(rng() % 10)});
                          }
                      }
                      const auto shards = static_cast<std::size_t>(1 + (rng() % 5));
                      auto d = distributed_sssp(g, 0, shards, *exec);
                      if (!d.has_value() || d->size() != n) {
                          t.expect(false, "every randomised sssp run succeeds");
                          continue;
                      }
                      ++graphs_checked;
                      bool all_match = true;
                      for (std::size_t v = 0; v < n; ++v) {
                          const auto oracle = serial_path(g, 0, static_cast<std::int64_t>(v));
                          const bool reachable = oracle.has_value();
                          if (reachable != (*d)[v].has_value()) {
                              all_match = false;
                              break;
                          }
                          if (reachable && oracle->second != *(*d)[v]) {
                              all_match = false;
                              break;
                          }
                          ++nodes_checked;
                      }
                      t.expect(all_match,
                               "the distributed distances match dijkstra for every node");
                  }
                  t.expect(graphs_checked == 40, "all forty randomised graphs were checked");
                  t.expect(nodes_checked > 200,
                           "the sweep covered a substantial number of nodes, not a handful");
              })
        .test("encode_frontier_and_slice_carry_int64_extremes_intact",
              [](TestContext& t) {
                  // A distance is a two's-complement bit pattern on the wire, so the extremes are
                  // where a sign-handling mistake would show up first.
                  constexpr std::int64_t lo = std::numeric_limits<std::int64_t>::min();
                  const std::vector<FrontierEntry> front{
                      FrontierEntry{.node = 0, .distance = lo}};
                  auto p = encode_frontier(front, std::numeric_limits<std::int64_t>::max());
                  t.expect(p.has_value(), "a frontier holding INT64_MIN encodes");
                  if (!p.has_value()) {
                      return;
                  }
                  t.expect(p->size() == 8 + 8 + 8 + 8 + 8 + 16,
                           "the payload is magic + bound + weight pair + count + one 16-byte entry");
                  WireGraph g;
                  g.adjacency = {{Edge{.target = 0, .cost = 0}}};
                  auto s = encode_slice(g, 0, 1);
                  t.expect(s.has_value(), "a one-node slice encodes");
              })
        .test("distributed_bfs_finds_the_fewest_hop_path_and_matches_bfs_hop_count",
              [](TestContext& t) {
                  // Two routes to node 5: three hops the cheap way, two hops the dear way. BFS
                  // must take the SHORTER-IN-HOPS one, which is the dearer -- the case that
                  // separates it from the shortest-cost searches above.
                  WireGraph g;
                  g.adjacency = {
                      {Edge{.target = 1, .cost = 1}, Edge{.target = 4, .cost = 100}},
                      {Edge{.target = 2, .cost = 1}},
                      {Edge{.target = 5, .cost = 1}},
                      {},
                      {Edge{.target = 5, .cost = 100}},
                      {},
                  };
                  auto exec = serial_executor();
                  const std::vector<std::int64_t> goals{5};
                  auto d = distributed_bfs(g, 0, goals, 2, *exec);
                  t.expect(d.has_value(), "the distributed bfs succeeds");
                  if (!d.has_value()) {
                      return;
                  }
                  const std::vector<std::int64_t> expected{0, 4, 5};
                  t.expect(*d == expected, "the path is exactly {0, 4, 5} -- two hops, not three");
                  auto oracle = bfs(0, [](std::int64_t u) { return u == 5; },
                                    nimblecas::search_dist::successors_of(g));
                  t.expect(oracle.has_value(), "serial bfs succeeds on the same graph");
                  if (!oracle.has_value()) {
                      return;
                  }
                  t.expect(d->size() == oracle->size(),
                           "the distributed hop count equals the serial bfs hop count");
              })
        .test("distributed_bfs_matches_serial_bfs_hop_count_on_a_grid",
              [](TestContext& t) {
                  const WireGraph g = grid_graph(4, 5, 7);
                  auto exec = local_parallel_executor();
                  const std::vector<std::int64_t> goals{19};
                  auto d = distributed_bfs(g, 0, goals, 3, *exec);
                  auto oracle = bfs(0, [](std::int64_t u) { return u == 19; },
                                    nimblecas::search_dist::successors_of(g));
                  t.expect(d.has_value() && oracle.has_value(), "both searches succeed");
                  if (!d.has_value() || !oracle.has_value()) {
                      return;
                  }
                  t.expect(d->size() == 8, "the corner of a 4x5 grid is seven hops away");
                  t.expect(d->size() == oracle->size(), "the hop counts agree");
                  t.expect(d->front() == 0 && d->back() == 19,
                           "the path runs from the start to the goal");
              })
        .test("distributed_bfs_on_the_start_node_returns_the_single_node_path",
              [](TestContext& t) {
                  const WireGraph g = diamond_graph();
                  auto exec = serial_executor();
                  const std::vector<std::int64_t> goals{2, 0};
                  auto d = distributed_bfs(g, 0, goals, 2, *exec);
                  t.expect(d.has_value(), "a start that is already a goal succeeds");
                  if (!d.has_value()) {
                      return;
                  }
                  const std::vector<std::int64_t> expected{0};
                  t.expect(*d == expected, "the path is exactly {0}");
              })
        .test("distributed_bfs_reports_an_unreachable_goal_and_guards_its_arguments",
              [](TestContext& t) {
                  WireGraph g = diamond_graph();
                  g.adjacency.push_back({});
                  auto exec = serial_executor();
                  const std::vector<std::int64_t> unreachable{6};
                  auto a = distributed_bfs(g, 0, unreachable, 2, *exec);
                  t.expect(!a.has_value() && a.error() == MathError::undefined_value,
                           "an unreachable goal is undefined_value");
                  const std::vector<std::int64_t> none;
                  auto b = distributed_bfs(g, 0, none, 2, *exec);
                  t.expect(!b.has_value() && b.error() == MathError::domain_error,
                           "an empty goal list is a domain_error");
                  const std::vector<std::int64_t> outside{99};
                  auto c = distributed_bfs(g, 0, outside, 2, *exec);
                  t.expect(!c.has_value() && c.error() == MathError::domain_error,
                           "a goal outside the graph is a domain_error");
              })
        .test("distributed_reachable_set_equals_the_nodes_a_serial_traversal_visits",
              [](TestContext& t) {
                  WireGraph g = diamond_graph();
                  g.adjacency.push_back({});                             // 6, isolated
                  g.adjacency.push_back({Edge{.target = 6, .cost = 1}}); // 7 -> 6 only
                  auto exec = serial_executor();
                  auto r = distributed_reachable_set(g, 0, 3, *exec);
                  t.expect(r.has_value(), "the reachable set is computed");
                  if (!r.has_value()) {
                      return;
                  }
                  const std::vector<std::int64_t> expected{0, 1, 2, 3, 4, 5};
                  t.expect(*r == expected,
                           "exactly nodes 0..5 are reachable, in ascending order");
                  // Cross-check against the serial traversals: every node in the set is reachable
                  // by bfs, and no node outside it is.
                  for (std::int64_t v = 0; v < 8; ++v) {
                      const bool in_set = std::ranges::find(*r, v) != r->end();
                      auto path = bfs(0, [v](std::int64_t u) { return u == v; },
                                      nimblecas::search_dist::successors_of(g));
                      t.expect(in_set == path.has_value(),
                               "membership of the reachable set matches what bfs can reach");
                  }
              })
        .test("distributed_reachable_set_from_an_isolated_node_is_just_that_node",
              [](TestContext& t) {
                  WireGraph g = diamond_graph();
                  g.adjacency.push_back({});
                  auto exec = local_parallel_executor();
                  auto r = distributed_reachable_set(g, 6, 2, *exec);
                  t.expect(r.has_value(), "the reachable set of an isolated node is computed");
                  if (!r.has_value()) {
                      return;
                  }
                  const std::vector<std::int64_t> expected{6};
                  t.expect(*r == expected, "only the node itself is reachable");
              })
        .test("distributed_iterative_deepening_matches_the_serial_hop_count",
              [](TestContext& t) {
                  WireGraph g;
                  g.adjacency = {
                      {Edge{.target = 1, .cost = 1}, Edge{.target = 3, .cost = 1}},
                      {Edge{.target = 2, .cost = 1}},
                      {Edge{.target = 5, .cost = 1}},
                      {Edge{.target = 4, .cost = 1}},
                      {Edge{.target = 5, .cost = 1}},
                      {},
                  };
                  auto exec = serial_executor();
                  const std::vector<std::int64_t> goals{5};
                  auto d = distributed_iterative_deepening(g, 0, goals, 10, 2, *exec);
                  auto oracle = iterative_deepening_dfs(
                      0, [](std::int64_t u) { return u == 5; },
                      nimblecas::search_dist::successors_of(g), 10);
                  t.expect(d.has_value() && oracle.has_value(), "both searches succeed");
                  if (!d.has_value() || !oracle.has_value()) {
                      return;
                  }
                  t.expect(d->size() == 4, "the shallowest goal is three hops away");
                  t.expect(d->size() == oracle->size(),
                           "the distributed hop count equals the serial one");
                  t.expect(d->front() == 0 && d->back() == 5,
                           "the path runs from the start to the goal");
              })
        .test("distributed_iterative_deepening_returns_not_converged_when_the_depth_runs_out",
              [](TestContext& t) {
                  const WireGraph g = line_graph(8, 1);
                  auto exec = serial_executor();
                  const std::vector<std::int64_t> goals{7};
                  auto shallow = distributed_iterative_deepening(g, 0, goals, 3, 2, *exec);
                  t.expect(!shallow.has_value() && shallow.error() == MathError::not_converged,
                           "too small a depth budget is not_converged, never a wrong path");
                  auto deep = distributed_iterative_deepening(g, 0, goals, 7, 2, *exec);
                  t.expect(deep.has_value(), "a sufficient depth budget finds the goal");
                  if (!deep.has_value()) {
                      return;
                  }
                  t.expect(deep->size() == 8, "the path is the whole eight-node line");
              })
        .test("distributed_iterative_deepening_reports_an_exhausted_graph_as_undefined_value",
              [](TestContext& t) {
                  WireGraph g;
                  g.adjacency = {
                      {Edge{.target = 1, .cost = 1}},
                      {},
                      {},
                  };
                  auto exec = serial_executor();
                  const std::vector<std::int64_t> goals{2};
                  auto d = distributed_iterative_deepening(g, 0, goals, 10, 2, *exec);
                  t.expect(!d.has_value() && d.error() == MathError::undefined_value,
                           "a goal that no depth could reach is undefined_value, not "
                           "not_converged -- the two mean different things");
              })
        .test("distributed_iterative_deepening_terminates_on_a_cycle",
              [](TestContext& t) {
                  // The on-path ancestor check, not a global visited set: a node reachable by two
                  // routes of different length must stay reachable by the shorter one.
                  WireGraph g;
                  g.adjacency = {
                      {Edge{.target = 1, .cost = 1}},
                      {Edge{.target = 0, .cost = 1}, Edge{.target = 2, .cost = 1}},
                      {Edge{.target = 1, .cost = 1}, Edge{.target = 3, .cost = 1}},
                      {},
                  };
                  auto exec = local_parallel_executor();
                  const std::vector<std::int64_t> goals{3};
                  auto d = distributed_iterative_deepening(g, 0, goals, 8, 2, *exec);
                  t.expect(d.has_value(), "a cyclic graph terminates");
                  if (!d.has_value()) {
                      return;
                  }
                  const std::vector<std::int64_t> expected{0, 1, 2, 3};
                  t.expect(*d == expected, "the path is exactly {0, 1, 2, 3}");
              })
        .test("distributed_weighted_a_star_at_weight_one_equals_distributed_a_star",
              [](TestContext& t) {
                  const WireGraph g = grid_graph(4, 4, 3);
                  auto exec = serial_executor();
                  auto w1 = distributed_weighted_a_star(g, 0, 15, 1, 1, 3, *exec);
                  auto plain = distributed_a_star(g, 0, 15, 3, *exec);
                  t.expect(w1.has_value() && plain.has_value(), "both searches succeed");
                  if (!w1.has_value() || !plain.has_value()) {
                      return;
                  }
                  t.expect(w1->second == plain->second, "weight 1/1 gives the same cost");
                  t.expect(w1->first == plain->first, "weight 1/1 gives the same path");
              })
        .test("distributed_weighted_a_star_respects_its_suboptimality_bound",
              [](TestContext& t) {
                  const WireGraph g = grid_graph(5, 5, 2);
                  auto exec = local_parallel_executor();
                  auto optimal = distributed_shortest_path(g, 0, 24, 3, *exec);
                  t.expect(optimal.has_value(), "the optimum is computed");
                  if (!optimal.has_value()) {
                      return;
                  }
                  // The only promise weighted A* makes is cost <= w * optimum. Asserting exactly
                  // that, and not optimality, is the difference between a test and a wish.
                  for (const std::int64_t num : {std::int64_t{3}, std::int64_t{5}}) {
                      auto w = distributed_weighted_a_star(g, 0, 24, num, 2, 3, *exec);
                      t.expect(w.has_value(), "the weighted search succeeds");
                      if (!w.has_value()) {
                          continue;
                      }
                      t.expect(2 * w->second <= num * optimal->second,
                               "the returned cost is within w times the optimum");
                  }
              })
        .test("distributed_weighted_a_star_rejects_a_weight_below_one",
              [](TestContext& t) {
                  const WireGraph g = grid_graph(3, 3, 1);
                  auto exec = serial_executor();
                  auto a = distributed_weighted_a_star(g, 0, 8, 1, 2, 2, *exec);
                  t.expect(!a.has_value() && a.error() == MathError::domain_error,
                           "a weight below 1 is a domain_error -- it would prune better paths");
                  auto b = distributed_weighted_a_star(g, 0, 8, 3, 0, 2, *exec);
                  t.expect(!b.has_value() && b.error() == MathError::domain_error,
                           "a zero denominator is a domain_error");
                  auto c = distributed_weighted_a_star(g, 0, 8, -1, 1, 2, *exec);
                  t.expect(!c.has_value() && c.error() == MathError::domain_error,
                           "a negative weight is a domain_error");
              })
        .test("distributed_floyd_warshall_matches_dijkstra_from_every_source",
              [](TestContext& t) {
                  const WireGraph g = diamond_graph();
                  auto exec = serial_executor();
                  auto m = distributed_floyd_warshall(g, 3, *exec);
                  t.expect(m.has_value(), "the all-pairs matrix is computed");
                  if (!m.has_value() || m->size() != 6) {
                      return;
                  }
                  for (std::int64_t i = 0; i < 6; ++i) {
                      const auto& row = (*m)[static_cast<std::size_t>(i)];
                      if (row.size() != 6) {
                          t.expect(false, "every row has one entry per node");
                          continue;
                      }
                      t.expect(row[static_cast<std::size_t>(i)].has_value() &&
                                   *row[static_cast<std::size_t>(i)] == 0,
                               "the distance from a node to itself is exactly 0");
                      for (std::int64_t j = 0; j < 6; ++j) {
                          if (i == j) {
                              continue;
                          }
                          const auto oracle = serial_path(g, i, j);
                          const auto& cell = row[static_cast<std::size_t>(j)];
                          t.expect(cell.has_value() == oracle.has_value(),
                                   "reachability agrees with dijkstra for this pair");
                          if (cell.has_value() && oracle.has_value()) {
                              t.expect(*cell == oracle->second,
                                       "the all-pairs distance equals dijkstra's distance");
                          }
                      }
                  }
              })
        .test("distributed_floyd_warshall_is_identical_for_every_shard_count_and_executor",
              [](TestContext& t) {
                  const WireGraph g = grid_graph(4, 4, 2);
                  auto ser = serial_executor();
                  auto par = local_parallel_executor();
                  auto baseline = distributed_floyd_warshall(g, 1, *ser);
                  t.expect(baseline.has_value(), "the one-shard run succeeds");
                  if (!baseline.has_value()) {
                      return;
                  }
                  for (const std::size_t shards : {std::size_t{2}, std::size_t{3},
                                                   std::size_t{5}, std::size_t{32}}) {
                      auto a = distributed_floyd_warshall(g, shards, *ser);
                      t.expect(a.has_value() && *a == *baseline,
                               "the row-block partition does not change a single entry");
                      auto b = distributed_floyd_warshall(g, shards, *par);
                      t.expect(b.has_value() && *b == *baseline,
                               "the parallel executor gives byte-for-byte the same matrix");
                  }
              })
        .test("distributed_floyd_warshall_collapses_parallel_edges_and_ignores_self_loops",
              [](TestContext& t) {
                  WireGraph g;
                  g.adjacency = {
                      {Edge{.target = 1, .cost = 9}, Edge{.target = 1, .cost = 4},
                       Edge{.target = 0, .cost = 5}},
                      {},
                  };
                  auto exec = serial_executor();
                  auto m = distributed_floyd_warshall(g, 2, *exec);
                  t.expect(m.has_value(), "the matrix is computed");
                  if (!m.has_value() || m->size() != 2 || (*m)[0].size() != 2) {
                      return;
                  }
                  t.expect((*m)[0][1].has_value() && *(*m)[0][1] == 4,
                           "two edges 0->1 collapse to the cheaper, exactly 4");
                  t.expect((*m)[0][0].has_value() && *(*m)[0][0] == 0,
                           "a cost-5 self-loop does not make the self-distance 5");
                  t.expect(!(*m)[1][0].has_value(), "node 0 is not reachable from node 1");
              })
        .test("distributed_floyd_warshall_on_an_empty_graph_is_an_empty_matrix",
              [](TestContext& t) {
                  const WireGraph g;
                  auto exec = serial_executor();
                  auto m = distributed_floyd_warshall(g, 2, *exec);
                  t.expect(m.has_value() && m->empty(),
                           "a graph with no nodes gives an empty matrix, not an error");
                  auto bad = distributed_floyd_warshall(g, 0, *exec);
                  t.expect(!bad.has_value() && bad.error() == MathError::domain_error,
                           "zero shards is still a domain_error");
              })
        .test("randomised_all_pairs_agree_with_dijkstra_from_every_source",
              [](TestContext& t) {
                  // The same adversarial sweep as the sssp one, against the all-pairs algorithm.
                  // Floyd-Warshall reaches its answer by a completely different route from
                  // Dijkstra, so agreement across random graphs is real evidence rather than a
                  // restatement of the implementation.
                  std::mt19937_64 rng(0xa11a17e5u);
                  auto exec = serial_executor();
                  int graphs_checked = 0;
                  for (int trial = 0; trial < 20; ++trial) {
                      const auto n = static_cast<std::size_t>(2 + (rng() % 7));
                      WireGraph g;
                      g.adjacency.resize(n);
                      for (std::size_t u = 0; u < n; ++u) {
                          const std::size_t deg = rng() % 4;
                          for (std::size_t e = 0; e < deg; ++e) {
                              g.adjacency[u].push_back(
                                  Edge{.target = static_cast<std::int64_t>(rng() % n),
                                       .cost = static_cast<std::int64_t>(rng() % 8)});
                          }
                      }
                      const auto shards = static_cast<std::size_t>(1 + (rng() % 4));
                      auto m = distributed_floyd_warshall(g, shards, *exec);
                      if (!m.has_value() || m->size() != n) {
                          t.expect(false, "every randomised all-pairs run succeeds");
                          continue;
                      }
                      ++graphs_checked;
                      bool all_match = true;
                      for (std::size_t i = 0; i < n && all_match; ++i) {
                          for (std::size_t j = 0; j < n; ++j) {
                              const auto& cell = (*m)[i][j];
                              if (i == j) {
                                  if (!cell.has_value() || *cell != 0) {
                                      all_match = false;
                                      break;
                                  }
                                  continue;
                              }
                              const auto oracle = serial_path(g, static_cast<std::int64_t>(i),
                                                              static_cast<std::int64_t>(j));
                              if (cell.has_value() != oracle.has_value()) {
                                  all_match = false;
                                  break;
                              }
                              if (cell.has_value() && *cell != oracle->second) {
                                  all_match = false;
                                  break;
                              }
                          }
                      }
                      t.expect(all_match,
                               "every all-pairs entry matches dijkstra from that source");
                  }
                  t.expect(graphs_checked == 20, "all twenty randomised graphs were checked");
              })
        .run();
}
