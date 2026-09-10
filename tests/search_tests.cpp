// Tests for nimblecas.search: graph/tree search, shortest paths, tabu search, and DP.
// @author Olumuyiwa Oluwasanmi
//
// A single small, fixed directed graph with hand-computed BFS/DFS/IDDFS paths and known
// Dijkstra/A* optimal costs anchors the traversal tests; the DP tests use textbook inputs
// with well-known answers. Every value is exact and deterministic, so results are
// independent of the parallel backend's thread count.

import std;
import nimblecas.core;
import nimblecas.search;
import nimblecas.testing;

using nimblecas::bidirectional_dijkstra;
using nimblecas::parallel_dijkstra;
using nimblecas::a_star;
using nimblecas::bfs;
using nimblecas::dfs_iterative;
using nimblecas::dfs_recursive;
using nimblecas::dijkstra;
using nimblecas::edit_distance;
using nimblecas::edit_distance_memo;
using nimblecas::edit_distance_parallel;
using nimblecas::iterative_deepening_dfs;
using nimblecas::knapsack_01;
using nimblecas::longest_common_subsequence;
using nimblecas::longest_common_subsequence_memo;
using nimblecas::MathError;
using nimblecas::parallel_bfs_levels;
using nimblecas::tabu_search;
using nimblecas::TabuState;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

using Path = std::vector<std::int64_t>;
using Levels = std::vector<std::vector<std::int64_t>>;

// Fixed directed graph (successor order is significant):
//   0 -> [1, 2]     1 -> [3]      2 -> [6]
//   3 -> [4]        4 -> [6]      5 -> [6]   (5 has no in-edges: unreachable from 0)
//   6 -> []         7 -> []       (7 isolated: unreachable from 0)
[[nodiscard]] auto successors(std::int64_t node) -> Path {
    switch (node) {
        case 0: return {1, 2};
        case 1: return {3};
        case 2: return {6};
        case 3: return {4};
        case 4: return {6};
        case 5: return {6};
        default: return {};  // 6, 7 and any other node are sinks
    }
}

// Edge weights used by dijkstra / a_star (non-negative). Only ever queried on real edges.
//   0->1:1  0->2:5  1->3:1  3->4:1  4->6:1  2->6:1  5->6:1
[[nodiscard]] auto edge_cost(std::int64_t u, std::int64_t v) -> std::int64_t {
    if (u == 0 && v == 1) return 1;
    if (u == 0 && v == 2) return 5;
    if (u == 1 && v == 3) return 1;
    if (u == 3 && v == 4) return 1;
    if (u == 4 && v == 6) return 1;
    if (u == 2 && v == 6) return 1;
    if (u == 5 && v == 6) return 1;
    return 1;  // any other (unused) edge
}

// Admissible AND consistent heuristic: a lower bound on the true remaining cost to node 6.
// True min costs to 6: d(6)=0 d(4)=1 d(3)=2 d(2)=1 d(1)=3 d(0)=4; every h below is <=.
[[nodiscard]] auto heuristic_to_6(std::int64_t node) -> std::int64_t {
    switch (node) {
        case 0: return 2;
        case 1: return 2;
        case 2: return 1;
        case 3: return 1;
        case 4: return 1;
        default: return 0;  // 5, 6, 7
    }
}

[[nodiscard]] auto is(std::int64_t target) -> nimblecas::GoalFn {
    return [target](std::int64_t n) { return n == target; };
}

// A reference serial level-BFS to compare parallel_bfs_levels against.
[[nodiscard]] auto serial_bfs_levels(std::int64_t start, std::int64_t max_levels) -> Levels {
    Levels levels;
    if (max_levels <= 0) {
        return levels;
    }
    std::set<std::int64_t> visited{start};
    std::vector<std::int64_t> frontier{start};
    levels.push_back(frontier);
    while (static_cast<std::int64_t>(levels.size()) < max_levels && !frontier.empty()) {
        std::set<std::int64_t> next;
        for (const std::int64_t u : frontier) {
            for (const std::int64_t v : successors(u)) {
                if (!visited.contains(v)) {
                    next.insert(v);
                }
            }
        }
        if (next.empty()) {
            break;
        }
        for (const std::int64_t v : next) {
            visited.insert(v);
        }
        std::vector<std::int64_t> level(next.begin(), next.end());
        levels.push_back(level);
        frontier = std::move(level);
    }
    return levels;
}

[[nodiscard]] auto seq(std::vector<std::int64_t> v) -> std::vector<std::int64_t> {
    return v;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.search")
        .test("bfs_shortest_edges",
              [](TestContext& t) {
                  // Fewest edges 0 -> 6 is via node 2: [0, 2, 6].
                  const auto p = bfs(0, is(6), successors);
                  t.expect(p.has_value(), "bfs finds a path");
                  t.expect(p.value_or(Path{}) == Path{0, 2, 6}, "bfs path is [0, 2, 6]");
              })
        .test("bfs_start_is_goal",
              [](TestContext& t) {
                  const auto p = bfs(0, is(0), successors);
                  t.expect(p.has_value() && p.value_or(Path{}) == Path{0},
                           "start already a goal => [0]");
              })
        .test("bfs_unreachable",
              [](TestContext& t) {
                  // Node 7 is isolated: no path from 0.
                  auto p = bfs(0, is(7), successors);
                  t.expect(!p.has_value() && p.error() == MathError::undefined_value,
                           "unreachable goal => undefined_value");
              })
        .test("dfs_recursive_deep_path",
              [](TestContext& t) {
                  // DFS descends the first branch fully: 0->1->3->4->6.
                  const auto p = dfs_recursive(0, is(6), successors, 10);
                  t.expect(p.has_value(), "dfs_recursive finds a path");
                  t.expect(p.value_or(Path{}) == Path{0, 1, 3, 4, 6},
                           "dfs path is [0, 1, 3, 4, 6]");
              })
        .test("dfs_iterative_matches_recursive",
              [](TestContext& t) {
                  const auto pr = dfs_recursive(0, is(6), successors, 10);
                  const auto pi = dfs_iterative(0, is(6), successors, 10);
                  t.expect(pr.has_value() && pi.has_value(), "both dfs forms find a path");
                  t.expect(pr.value_or(Path{}) == pi.value_or(seq({1})),
                           "iterative dfs == recursive dfs (same first-goal path)");
                  t.expect(pi.value_or(Path{}) == Path{0, 1, 3, 4, 6},
                           "iterative dfs path is [0, 1, 3, 4, 6]");
              })
        .test("dfs_depth_limit_blocks",
              [](TestContext& t) {
                  // Node 6 is at least 2 edges from 0, so a depth limit of 1 reaches
                  // neither branch's goal.
                  auto p = dfs_recursive(0, is(6), successors, 1);
                  t.expect(!p.has_value() && p.error() == MathError::undefined_value,
                           "goal beyond max_depth => undefined_value");
                  auto q = dfs_iterative(0, is(6), successors, 1);
                  t.expect(!q.has_value() && q.error() == MathError::undefined_value,
                           "iterative dfs respects the same bound");
              })
        .test("dfs_negative_depth_error",
              [](TestContext& t) {
                  auto p = dfs_iterative(0, is(6), successors, -1);
                  t.expect(!p.has_value() && p.error() == MathError::domain_error,
                           "max_depth < 0 => domain_error");
              })
        .test("iddfs_finds_shallowest",
              [](TestContext& t) {
                  // IDDFS returns the shallowest goal: [0, 2, 6] (2 edges), like BFS.
                  const auto p = iterative_deepening_dfs(0, is(6), successors, 10);
                  t.expect(p.has_value(), "iddfs finds a path");
                  t.expect(p.value_or(Path{}) == Path{0, 2, 6}, "iddfs path is [0, 2, 6]");
              })
        .test("iddfs_unreachable",
              [](TestContext& t) {
                  auto p = iterative_deepening_dfs(0, is(5), successors, 10);
                  t.expect(!p.has_value() && p.error() == MathError::undefined_value,
                           "unreachable goal => undefined_value");
              })
        .test("dijkstra_min_cost_path",
              [](TestContext& t) {
                  // Cheapest 0 -> 6 is the long branch: 1+1+1+1 = 4 via [0,1,3,4,6],
                  // beating the 2-edge [0,2,6] which costs 5+1 = 6.
                  auto r = dijkstra(0, is(6), successors, edge_cost);
                  t.expect(r.has_value(), "dijkstra finds a path");
                  if (r) {
                      t.expect(r->first == Path{0, 1, 3, 4, 6}, "dijkstra path is [0,1,3,4,6]");
                      t.expect(r->second == 4, "dijkstra optimal cost is 4");
                  }
              })
        .test("dijkstra_unreachable",
              [](TestContext& t) {
                  auto r = dijkstra(0, is(7), successors, edge_cost);
                  t.expect(!r.has_value() && r.error() == MathError::undefined_value,
                           "unreachable goal => undefined_value");
              })
        .test("a_star_equals_dijkstra",
              [](TestContext& t) {
                  auto d = dijkstra(0, is(6), successors, edge_cost);
                  auto a = a_star(0, is(6), successors, edge_cost, heuristic_to_6);
                  t.expect(d.has_value() && a.has_value(), "both searches find a path");
                  if (d && a) {
                      t.expect(a->second == d->second, "A* cost == dijkstra cost (== 4)");
                      t.expect(a->second == 4, "A* optimal cost is 4");
                      t.expect(a->first == d->first, "A* path == dijkstra path");
                  }
              })
        .test("parallel_bfs_levels_matches_serial",
              [](TestContext& t) {
                  const auto p = parallel_bfs_levels(0, successors, 10);
                  t.expect(p.has_value(), "parallel_bfs_levels succeeds");
                  const Levels expected{{0}, {1, 2}, {3, 6}, {4}};
                  t.expect(p.value_or(Levels{}) == expected,
                           "levels are [[0],[1,2],[3,6],[4]]");
                  t.expect(p.value_or(Levels{}) == serial_bfs_levels(0, 10),
                           "parallel levels == serial level-BFS");
              })
        .test("parallel_bfs_levels_truncates_and_guards",
              [](TestContext& t) {
                  const auto two = parallel_bfs_levels(0, successors, 2);
                  const Levels expected2{{0}, {1, 2}};
                  t.expect(two.value_or(Levels{}) == expected2, "max_levels=2 => first 2 levels");
                  const auto zero = parallel_bfs_levels(0, successors, 0);
                  t.expect(zero.has_value() && zero.value_or(Levels{{9}}).empty(),
                           "max_levels=0 => empty");
                  auto neg = parallel_bfs_levels(0, successors, -1);
                  t.expect(!neg.has_value() && neg.error() == MathError::domain_error,
                           "max_levels<0 => domain_error");
              })
        .test("tabu_search_minimises_quadratic",
              [](TestContext& t) {
                  // Minimise (x-3)^2 + (y+2)^2 over the integer grid [-10,10]^2 with unit
                  // steps in one coordinate. Unique minimum {3,-2} with value 0.
                  const auto objective = [](const TabuState& s) -> std::int64_t {
                      const std::int64_t dx = s[0] - 3;
                      const std::int64_t dy = s[1] + 2;
                      return dx * dx + dy * dy;
                  };
                  const auto neighbors = [](const TabuState& s) -> std::vector<TabuState> {
                      std::vector<TabuState> out;
                      const std::array<std::pair<int, int>, 4> steps{
                          {{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};
                      for (const auto& [dx, dy] : steps) {
                          const std::int64_t nx = s[0] + dx;
                          const std::int64_t ny = s[1] + dy;
                          if (nx >= -10 && nx <= 10 && ny >= -10 && ny <= 10) {
                              out.push_back(TabuState{nx, ny});
                          }
                      }
                      return out;
                  };
                  auto r = tabu_search(TabuState{0, 0}, neighbors, objective, 3, 100);
                  t.expect(r.has_value(), "tabu_search succeeds");
                  if (r) {
                      t.expect(r->second == 0, "minimum objective value is 0");
                      t.expect(r->first == TabuState{3, -2}, "minimiser is {3, -2}");
                  }
              })
        .test("tabu_search_negative_params_error",
              [](TestContext& t) {
                  const auto objective = [](const TabuState& s) -> std::int64_t { return s[0]; };
                  const auto neighbors = [](const TabuState&) -> std::vector<TabuState> { return {}; };
                  auto r = tabu_search(TabuState{0}, neighbors, objective, -1, 10);
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "negative tenure => domain_error");
              })
        .test("edit_distance_textbook",
              [](TestContext& t) {
                  const auto tab = edit_distance("kitten", "sitting");
                  const auto memo = edit_distance_memo("kitten", "sitting");
                  const auto par = edit_distance_parallel("kitten", "sitting");
                  t.expect(tab.value_or(-1) == 3, "edit_distance(kitten, sitting) == 3");
                  t.expect(memo.value_or(-1) == 3, "memo edit distance == 3");
                  t.expect(par.value_or(-1) == 3, "parallel edit distance == 3");
                  t.expect(tab.value_or(-1) == memo.value_or(-2) &&
                               memo.value_or(-2) == par.value_or(-3),
                           "tabulated == memo == parallel");
              })
        .test("edit_distance_edges",
              [](TestContext& t) {
                  // Empty vs non-empty is the length; equal strings are distance 0.
                  t.expect(edit_distance("", "abc").value_or(-1) == 3, "'' -> abc == 3");
                  t.expect(edit_distance("abc", "").value_or(-1) == 3, "abc -> '' == 3");
                  t.expect(edit_distance("abc", "abc").value_or(-1) == 0, "abc -> abc == 0");
                  t.expect(edit_distance_parallel("", "abc").value_or(-1) == 3,
                           "parallel '' -> abc == 3");
                  t.expect(edit_distance_parallel("flaw", "lawn").value_or(-1) ==
                               edit_distance("flaw", "lawn").value_or(-2),
                           "parallel == tabulated on flaw/lawn");
              })
        .test("lcs_textbook",
              [](TestContext& t) {
                  // LCS of [1,2,3,4,5] and [2,4,5] is [2,4,5], length 3.
                  const std::vector<std::int64_t> a{1, 2, 3, 4, 5};
                  const std::vector<std::int64_t> b{2, 4, 5};
                  const auto tab = longest_common_subsequence(a, b);
                  const auto memo = longest_common_subsequence_memo(a, b);
                  t.expect(tab.value_or(-1) == 3, "LCS length is 3");
                  t.expect(memo.value_or(-1) == 3, "memo LCS length is 3");
                  t.expect(tab.value_or(-1) == memo.value_or(-2), "tabulated == memo");
              })
        .test("lcs_disjoint_is_zero",
              [](TestContext& t) {
                  const std::vector<std::int64_t> a{1, 2, 3};
                  const std::vector<std::int64_t> b{4, 5, 6};
                  t.expect(longest_common_subsequence(a, b).value_or(-1) == 0,
                           "disjoint sequences => LCS 0");
                  t.expect(longest_common_subsequence_memo(a, b).value_or(-1) == 0,
                           "memo disjoint => LCS 0");
              })
        .test("knapsack_textbook",
              [](TestContext& t) {
                  // Items (w,v): (1,1),(3,4),(4,5),(5,7); capacity 7. Optimum = 3+4 => 9.
                  const std::vector<std::int64_t> w{1, 3, 4, 5};
                  const std::vector<std::int64_t> v{1, 4, 5, 7};
                  const auto r = knapsack_01(w, v, 7);
                  t.expect(r.value_or(-1) == 9, "knapsack optimum is 9");
              })
        .test("knapsack_errors",
              [](TestContext& t) {
                  const std::vector<std::int64_t> w{1, 2};
                  const std::vector<std::int64_t> v{1};
                  t.expect(knapsack_01(w, v, 5).error() == MathError::domain_error,
                           "mismatched sizes => domain_error");
                  const std::vector<std::int64_t> w2{1, 2};
                  const std::vector<std::int64_t> v2{1, 2};
                  t.expect(knapsack_01(w2, v2, -1).error() == MathError::domain_error,
                           "negative capacity => domain_error");
                  const std::vector<std::int64_t> wn{-1, 2};
                  t.expect(knapsack_01(wn, v2, 5).error() == MathError::domain_error,
                           "negative weight => domain_error");
              })
        .test("ida_star_cost_equals_a_star_cost_on_weighted_grid",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto successors = [](std::int64_t u) -> std::vector<std::int64_t> {
                      const std::int64_t r = u / 3;
                      const std::int64_t c = u % 3;
                      std::vector<std::int64_t> succs;
                      if (c + 1 < 3) {
                          succs.push_back(r * 3 + (c + 1));
                      }
                      if (r + 1 < 3) {
                          succs.push_back((r + 1) * 3 + c);
                      }
                      return succs;
                  };
                  const auto cost = [](std::int64_t u, std::int64_t v) -> std::int64_t {
                      if (u == 0 && v == 1) return 2;
                      if (u == 1 && v == 2) return 5;
                      if (u == 0 && v == 3) return 3;
                      if (u == 3 && v == 6) return 4;
                      if (u == 1 && v == 4) return 1;
                      if (u == 3 && v == 4) return 2;
                      if (u == 4 && v == 5) return 3;
                      if (u == 2 && v == 5) return 1;
                      if (u == 6 && v == 7) return 2;
                      if (u == 4 && v == 7) return 3;
                      if (u == 7 && v == 8) return 1;
                      if (u == 5 && v == 8) return 4;
                      return 1;
                  };
                  const auto heuristic = [](std::int64_t u) -> std::int64_t {
                      const std::int64_t r = u / 3;
                      const std::int64_t c = u % 3;
                      return (2 - r) + (2 - c);
                  };
                  const auto goal = [](std::int64_t u) -> bool { return u == 8; };

                  auto r_ida = ida_star(0, goal, successors, cost, heuristic, 100);
                  auto r_astar = a_star(0, goal, successors, cost, heuristic);

                  t.expect(r_ida.has_value(), "ida_star succeeds on weighted grid");
                  t.expect(r_astar.has_value(), "a_star succeeds on weighted grid");
                  if (r_ida.has_value() && r_astar.has_value()) {
                      t.expect(r_ida->second == r_astar->second, "ida_star cost matches a_star cost");
                      t.expect(r_ida->second == 7, "ida_star optimal cost is exactly 7");
                      const std::vector<std::int64_t> expected_path{0, 1, 4, 7, 8};
                      t.expect(r_ida->first == expected_path, "ida_star path matches expected optimal path");
                      t.expect(r_astar->first == expected_path, "a_star path matches expected optimal path");
                  }
              })
        .test("ida_star_cost_equals_a_star_cost_on_dag_with_multiple_paths",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto successors = [](std::int64_t u) -> std::vector<std::int64_t> {
                      switch (u) {
                          case 0: return {1, 2};
                          case 1: return {3, 4};
                          case 2: return {3, 4};
                          case 3: return {5};
                          case 4: return {5};
                          default: return {};
                      }
                  };
                  const auto cost = [](std::int64_t u, std::int64_t v) -> std::int64_t {
                      if (u == 0 && v == 1) return 3;
                      if (u == 0 && v == 2) return 1;
                      if (u == 1 && v == 3) return 2;
                      if (u == 1 && v == 4) return 4;
                      if (u == 2 && v == 3) return 4;
                      if (u == 2 && v == 4) return 1;
                      if (u == 3 && v == 5) return 3;
                      if (u == 4 && v == 5) return 6;
                      return 1;
                  };
                  const auto heuristic = [](std::int64_t) -> std::int64_t { return 0; };
                  const auto goal = [](std::int64_t u) -> bool { return u == 5; };

                  auto r_ida = ida_star(0, goal, successors, cost, heuristic, 50);
                  auto r_astar = a_star(0, goal, successors, cost, heuristic);

                  t.expect(r_ida.has_value(), "ida_star finds a path on multi-path DAG");
                  t.expect(r_astar.has_value(), "a_star finds a path on multi-path DAG");
                  if (r_ida.has_value() && r_astar.has_value()) {
                      t.expect(r_ida->second == r_astar->second, "ida_star cost equals a_star cost on DAG");
                      t.expect(r_ida->second == 8, "optimal DAG path cost is exactly 8");
                  }
              })
        .test("parallel_a_star_matches_a_star_cost_and_path_identically",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto successors = [](std::int64_t u) -> std::vector<std::int64_t> {
                      const std::int64_t r = u / 3;
                      const std::int64_t c = u % 3;
                      std::vector<std::int64_t> succs;
                      if (c + 1 < 3) succs.push_back(r * 3 + (c + 1));
                      if (r + 1 < 3) succs.push_back((r + 1) * 3 + c);
                      return succs;
                  };
                  const auto cost = [](std::int64_t u, std::int64_t v) -> std::int64_t {
                      if (u == 0 && v == 1) return 2;
                      if (u == 1 && v == 2) return 5;
                      if (u == 0 && v == 3) return 3;
                      if (u == 3 && v == 6) return 4;
                      if (u == 1 && v == 4) return 1;
                      if (u == 3 && v == 4) return 2;
                      if (u == 4 && v == 5) return 3;
                      if (u == 2 && v == 5) return 1;
                      if (u == 6 && v == 7) return 2;
                      if (u == 4 && v == 7) return 3;
                      if (u == 7 && v == 8) return 1;
                      if (u == 5 && v == 8) return 4;
                      return 1;
                  };
                  const auto heuristic = [](std::int64_t u) -> std::int64_t {
                      const std::int64_t r = u / 3;
                      const std::int64_t c = u % 3;
                      return (2 - r) + (2 - c);
                  };
                  const auto goal = [](std::int64_t u) -> bool { return u == 8; };

                  auto serial_ref = a_star(0, goal, successors, cost, heuristic);
                  t.expect(serial_ref.has_value(), "serial a_star succeeds as reference oracle");
                  if (!serial_ref.has_value()) {
                      return;
                  }

                  for (int run = 0; run < 5; ++run) {
                      auto par_res = parallel_a_star(0, goal, successors, cost, heuristic);
                      t.expect(par_res.has_value(), "parallel_a_star succeeds across repeated runs");
                      if (par_res.has_value()) {
                          t.expect(par_res->second == serial_ref->second, "parallel_a_star cost bit-identical to a_star");
                          t.expect(par_res->first == serial_ref->first, "parallel_a_star path vector bit-identical to a_star");
                      }
                  }
              })
        .test("parallel_dijkstra_matches_dijkstra_cost_and_path_identically",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto successors = [](std::int64_t u) -> std::vector<std::int64_t> {
                      const std::int64_t r = u / 3;
                      const std::int64_t c = u % 3;
                      std::vector<std::int64_t> succs;
                      if (c + 1 < 3) succs.push_back(r * 3 + (c + 1));
                      if (r + 1 < 3) succs.push_back((r + 1) * 3 + c);
                      return succs;
                  };
                  const auto cost = [](std::int64_t u, std::int64_t v) -> std::int64_t {
                      if (u == 0 && v == 1) return 2;
                      if (u == 1 && v == 2) return 5;
                      if (u == 0 && v == 3) return 3;
                      if (u == 3 && v == 6) return 4;
                      if (u == 1 && v == 4) return 1;
                      if (u == 3 && v == 4) return 2;
                      if (u == 4 && v == 5) return 3;
                      if (u == 2 && v == 5) return 1;
                      if (u == 6 && v == 7) return 2;
                      if (u == 4 && v == 7) return 3;
                      if (u == 7 && v == 8) return 1;
                      if (u == 5 && v == 8) return 4;
                      return 1;
                  };
                  const auto goal = [](std::int64_t u) -> bool { return u == 8; };

                  auto serial_ref = dijkstra(0, goal, successors, cost);
                  t.expect(serial_ref.has_value(), "serial dijkstra succeeds as reference oracle");
                  if (!serial_ref.has_value()) {
                      return;
                  }

                  for (int run = 0; run < 5; ++run) {
                      auto par_res = parallel_dijkstra(0, goal, successors, cost);
                      t.expect(par_res.has_value(), "parallel_dijkstra succeeds across repeated runs");
                      if (par_res.has_value()) {
                          t.expect(par_res->second == serial_ref->second, "parallel_dijkstra cost bit-identical to dijkstra");
                          t.expect(par_res->first == serial_ref->first, "parallel_dijkstra path vector bit-identical to dijkstra");
                      }
                  }
              })
        .test("bidirectional_dijkstra_finds_optimal_path_when_first_meeting_is_suboptimal",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto successors = [](std::int64_t u) -> std::vector<std::int64_t> {
                      switch (u) {
                          case 0: return {1, 3};
                          case 1: return {2};
                          case 2: return {4};
                          case 3: return {5};
                          case 4: return {5};
                          default: return {};
                      }
                  };
                  const auto predecessors = [](std::int64_t v) -> std::vector<std::int64_t> {
                      switch (v) {
                          case 1: return {0};
                          case 2: return {1};
                          case 3: return {0};
                          case 4: return {2};
                          case 5: return {3, 4};
                          default: return {};
                      }
                  };
                  const auto cost = [](std::int64_t u, std::int64_t v) -> std::int64_t {
                      if (u == 0 && v == 3) return 10;
                      if (u == 3 && v == 5) return 10;
                      if (u == 0 && v == 1) return 2;
                      if (u == 1 && v == 2) return 2;
                      if (u == 2 && v == 4) return 2;
                      if (u == 4 && v == 5) return 2;
                      return 1;
                  };
                  const auto goal = [](std::int64_t u) -> bool { return u == 5; };

                  auto r_dijk = dijkstra(0, goal, successors, cost);
                  auto r_bidi = bidirectional_dijkstra(0, 5, successors, predecessors, cost);

                  t.expect(r_dijk.has_value(), "dijkstra succeeds");
                  t.expect(r_bidi.has_value(), "bidirectional_dijkstra succeeds");
                  if (r_dijk.has_value() && r_bidi.has_value()) {
                      t.expect(r_bidi->second == r_dijk->second, "bidirectional cost equals dijkstra cost");
                      t.expect(r_bidi->second == 8, "bidirectional finds optimal cost 8 instead of naive meeting cost 20");
                      const std::vector<std::int64_t> expected_path{0, 1, 2, 4, 5};
                      t.expect(r_bidi->first == expected_path, "bidirectional path is exactly {0, 1, 2, 4, 5}");
                  }
              })
        .test("bidirectional_dijkstra_cost_matches_dijkstra_on_line_graph",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto successors = [](std::int64_t u) -> std::vector<std::int64_t> {
                      if (u >= 0 && u < 4) {
                          return {u + 1};
                      }
                      return {};
                  };
                  const auto predecessors = [](std::int64_t v) -> std::vector<std::int64_t> {
                      if (v > 0 && v <= 4) {
                          return {v - 1};
                      }
                      return {};
                  };
                  const auto cost = [](std::int64_t u, std::int64_t v) -> std::int64_t {
                      if (u == 0 && v == 1) return 3;
                      if (u == 1 && v == 2) return 5;
                      if (u == 2 && v == 3) return 2;
                      if (u == 3 && v == 4) return 4;
                      return 1;
                  };
                  const auto goal = [](std::int64_t u) -> bool { return u == 4; };

                  auto r_dijk = dijkstra(0, goal, successors, cost);
                  auto r_bidi = bidirectional_dijkstra(0, 4, successors, predecessors, cost);

                  t.expect(r_dijk.has_value() && r_bidi.has_value(), "both searches find line path");
                  if (r_dijk.has_value() && r_bidi.has_value()) {
                      t.expect(r_bidi->second == r_dijk->second, "bidirectional cost matches dijkstra on line");
                      t.expect(r_bidi->second == 14, "line path total cost is exactly 14");
                      const std::vector<std::int64_t> expected_path{0, 1, 2, 3, 4};
                      t.expect(r_bidi->first == expected_path, "bidirectional path is exactly {0, 1, 2, 3, 4}");
                  }
              })
        .test("weighted_a_star_with_weight_one_matches_a_star_exactly",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto successors = [](std::int64_t u) -> std::vector<std::int64_t> {
                      const std::int64_t r = u / 3;
                      const std::int64_t c = u % 3;
                      std::vector<std::int64_t> succs;
                      if (c + 1 < 3) succs.push_back(r * 3 + (c + 1));
                      if (r + 1 < 3) succs.push_back((r + 1) * 3 + c);
                      return succs;
                  };
                  const auto cost = [](std::int64_t u, std::int64_t v) -> std::int64_t {
                      if (u == 0 && v == 1) return 2;
                      if (u == 1 && v == 2) return 5;
                      if (u == 0 && v == 3) return 3;
                      if (u == 3 && v == 6) return 4;
                      if (u == 1 && v == 4) return 1;
                      if (u == 3 && v == 4) return 2;
                      if (u == 4 && v == 5) return 3;
                      if (u == 2 && v == 5) return 1;
                      if (u == 6 && v == 7) return 2;
                      if (u == 4 && v == 7) return 3;
                      if (u == 7 && v == 8) return 1;
                      if (u == 5 && v == 8) return 4;
                      return 1;
                  };
                  const auto heuristic = [](std::int64_t u) -> std::int64_t {
                      const std::int64_t r = u / 3;
                      const std::int64_t c = u % 3;
                      return (2 - r) + (2 - c);
                  };
                  const auto goal = [](std::int64_t u) -> bool { return u == 8; };

                  auto r_astar = a_star(0, goal, successors, cost, heuristic);
                  auto r_wastar = weighted_a_star(0, goal, successors, cost, heuristic, 1, 1);

                  t.expect(r_astar.has_value(), "a_star succeeds");
                  t.expect(r_wastar.has_value(), "weighted_a_star with w=1/1 succeeds");
                  if (r_astar.has_value() && r_wastar.has_value()) {
                      t.expect(r_wastar->second == r_astar->second, "weighted_a_star w=1/1 cost == a_star cost");
                      t.expect(r_wastar->first == r_astar->first, "weighted_a_star w=1/1 path == a_star path");
                      t.expect(r_wastar->second == 7, "cost is exactly 7");
                  }
              })
        .test("weighted_a_star_with_weight_greater_than_one_respects_suboptimality_bound",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto successors = [](std::int64_t u) -> std::vector<std::int64_t> {
                      const std::int64_t r = u / 3;
                      const std::int64_t c = u % 3;
                      std::vector<std::int64_t> succs;
                      if (c + 1 < 3) succs.push_back(r * 3 + (c + 1));
                      if (r + 1 < 3) succs.push_back((r + 1) * 3 + c);
                      return succs;
                  };
                  const auto cost = [](std::int64_t u, std::int64_t v) -> std::int64_t {
                      if (u == 0 && v == 1) return 2;
                      if (u == 1 && v == 2) return 5;
                      if (u == 0 && v == 3) return 3;
                      if (u == 3 && v == 6) return 4;
                      if (u == 1 && v == 4) return 1;
                      if (u == 3 && v == 4) return 2;
                      if (u == 4 && v == 5) return 3;
                      if (u == 2 && v == 5) return 1;
                      if (u == 6 && v == 7) return 2;
                      if (u == 4 && v == 7) return 3;
                      if (u == 7 && v == 8) return 1;
                      if (u == 5 && v == 8) return 4;
                      return 1;
                  };
                  const auto heuristic = [](std::int64_t u) -> std::int64_t {
                      const std::int64_t r = u / 3;
                      const std::int64_t c = u % 3;
                      return (2 - r) + (2 - c);
                  };
                  const auto goal = [](std::int64_t u) -> bool { return u == 8; };

                  auto opt = a_star(0, goal, successors, cost, heuristic);
                  auto r_w = weighted_a_star(0, goal, successors, cost, heuristic, 3, 2);

                  t.expect(opt.has_value(), "a_star succeeds for baseline optimum");
                  t.expect(r_w.has_value(), "weighted_a_star succeeds with w=3/2");
                  if (opt.has_value() && r_w.has_value()) {
                      t.expect(r_w->second * 2 <= opt->second * 3, "weighted A* cost obeys suboptimality bound (<= 1.5 * opt)");
                  }
              })
        .test("parallel_multistart_tabu_with_single_start_matches_serial_tabu_search",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto objective = [](const TabuState& s) -> std::int64_t {
                      if (s.size() < 2) return 0;
                      const std::int64_t dx = s[0] - 3;
                      const std::int64_t dy = s[1] + 2;
                      return dx * dx + dy * dy;
                  };
                  const auto neighbors = [](const TabuState& s) -> std::vector<TabuState> {
                      if (s.size() < 2) return {};
                      std::vector<TabuState> out;
                      const std::array<std::pair<int, int>, 4> steps{{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};
                      for (const auto& [dx, dy] : steps) {
                          const std::int64_t nx = s[0] + dx;
                          const std::int64_t ny = s[1] + dy;
                          if (nx >= -10 && nx <= 10 && ny >= -10 && ny <= 10) {
                              out.push_back(TabuState{nx, ny});
                          }
                      }
                      return out;
                  };

                  const TabuState start{0, 0};
                  auto r_serial = tabu_search(start, neighbors, objective, 3, 100);
                  auto r_par = parallel_multistart_tabu({start}, neighbors, objective, 3, 100);

                  t.expect(r_serial.has_value(), "serial tabu_search succeeds");
                  t.expect(r_par.has_value(), "parallel_multistart_tabu with 1 start succeeds");
                  if (r_serial.has_value() && r_par.has_value()) {
                      t.expect(r_par->second == r_serial->second, "objective value matches serial tabu exactly");
                      t.expect(r_par->first == r_serial->first, "minimiser state matches serial tabu exactly");
                      t.expect(r_par->second == 0, "optimal quadratic objective is exactly 0");
                      t.expect(r_par->first == TabuState{3, -2}, "minimiser is {3, -2}");
                  }
              })
        .test("parallel_multistart_tabu_with_multiple_starts_picks_the_best_run",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto objective = [](const TabuState& s) -> std::int64_t {
                      if (s.size() < 2) return 0;
                      const std::int64_t dx = s[0] - 2;
                      const std::int64_t dy = s[1] - 4;
                      return dx * dx + dy * dy;
                  };
                  const auto neighbors = [](const TabuState& s) -> std::vector<TabuState> {
                      if (s.size() < 2) return {};
                      std::vector<TabuState> out;
                      const std::array<std::pair<int, int>, 4> steps{{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};
                      for (const auto& [dx, dy] : steps) {
                          const std::int64_t nx = s[0] + dx;
                          const std::int64_t ny = s[1] + dy;
                          if (nx >= -10 && nx <= 10 && ny >= -10 && ny <= 10) {
                              out.push_back(TabuState{nx, ny});
                          }
                      }
                      return out;
                  };

                  const std::vector<TabuState> starts{{0, 0}, {2, 3}};
                  auto r1 = tabu_search(starts[0], neighbors, objective, 2, 5);
                  auto r2 = tabu_search(starts[1], neighbors, objective, 2, 5);
                  auto r_par = parallel_multistart_tabu(starts, neighbors, objective, 2, 5);

                  t.expect(r1.has_value() && r2.has_value(), "individual serial runs succeed");
                  t.expect(r_par.has_value(), "parallel_multistart_tabu succeeds with 2 starts");
                  if (r1.has_value() && r2.has_value() && r_par.has_value()) {
                      const std::int64_t best_serial_obj = std::min(r1->second, r2->second);
                      t.expect(r_par->second == best_serial_obj, "parallel multistart matches best serial run objective");
                  }
              })
        .test("zero_heuristic_makes_all_a_star_variants_agree_with_dijkstra",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto successors = [](std::int64_t u) -> std::vector<std::int64_t> {
                      const std::int64_t r = u / 3;
                      const std::int64_t c = u % 3;
                      std::vector<std::int64_t> succs;
                      if (c + 1 < 3) succs.push_back(r * 3 + (c + 1));
                      if (r + 1 < 3) succs.push_back((r + 1) * 3 + c);
                      return succs;
                  };
                  const auto cost = [](std::int64_t u, std::int64_t v) -> std::int64_t {
                      if (u == 0 && v == 1) return 2;
                      if (u == 1 && v == 2) return 5;
                      if (u == 0 && v == 3) return 3;
                      if (u == 3 && v == 6) return 4;
                      if (u == 1 && v == 4) return 1;
                      if (u == 3 && v == 4) return 2;
                      if (u == 4 && v == 5) return 3;
                      if (u == 2 && v == 5) return 1;
                      if (u == 6 && v == 7) return 2;
                      if (u == 4 && v == 7) return 3;
                      if (u == 7 && v == 8) return 1;
                      if (u == 5 && v == 8) return 4;
                      return 1;
                  };
                  const auto zero_h = [](std::int64_t) -> std::int64_t { return 0; };
                  const auto goal = [](std::int64_t u) -> bool { return u == 8; };

                  auto r_dijk = dijkstra(0, goal, successors, cost);
                  auto r_astar = a_star(0, goal, successors, cost, zero_h);
                  auto r_ida = ida_star(0, goal, successors, cost, zero_h, 100);
                  auto r_wastar = weighted_a_star(0, goal, successors, cost, zero_h, 1, 1);

                  t.expect(r_dijk.has_value() && r_astar.has_value(), "dijkstra and a_star succeed with h=0");
                  t.expect(r_ida.has_value() && r_wastar.has_value(), "ida_star and weighted_a_star succeed with h=0");
                  if (r_dijk.has_value() && r_astar.has_value() && r_ida.has_value() && r_wastar.has_value()) {
                      t.expect(r_dijk->second == 7, "dijkstra optimal cost is 7");
                      t.expect(r_astar->second == 7, "a_star with h=0 cost is 7");
                      t.expect(r_ida->second == 7, "ida_star with h=0 cost is 7");
                      t.expect(r_wastar->second == 7, "weighted_a_star with h=0 cost is 7");
                  }
              })
        .test("start_satisfying_goal_returns_single_node_path_with_cost_zero",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto successors = [](std::int64_t) -> std::vector<std::int64_t> { return {}; };
                  const auto predecessors = [](std::int64_t) -> std::vector<std::int64_t> { return {}; };
                  const auto cost = [](std::int64_t, std::int64_t) -> std::int64_t { return 1; };
                  const auto heuristic = [](std::int64_t) -> std::int64_t { return 0; };
                  const auto goal_at_start = [](std::int64_t u) -> bool { return u == 42; };

                  const std::vector<std::int64_t> expected_path{42};

                  auto r_ida = ida_star(42, goal_at_start, successors, cost, heuristic, 10);
                  t.expect(r_ida.has_value() && r_ida->first == expected_path && r_ida->second == 0,
                           "ida_star start-is-goal returns {42} cost 0");

                  auto r_wastar = weighted_a_star(42, goal_at_start, successors, cost, heuristic, 1, 1);
                  t.expect(r_wastar.has_value() && r_wastar->first == expected_path && r_wastar->second == 0,
                           "weighted_a_star start-is-goal returns {42} cost 0");

                  auto r_gbf = greedy_best_first(42, goal_at_start, successors, heuristic, 10);
                  t.expect(r_gbf.has_value() && *r_gbf == expected_path,
                           "greedy_best_first start-is-goal returns {42}");

                  auto r_beam = beam_search(42, goal_at_start, successors, heuristic, 5, 10);
                  t.expect(r_beam.has_value() && *r_beam == expected_path,
                           "beam_search start-is-goal returns {42}");

                  auto r_bidi = bidirectional_dijkstra(42, 42, successors, predecessors, cost);
                  t.expect(r_bidi.has_value() && r_bidi->first == expected_path && r_bidi->second == 0,
                           "bidirectional_dijkstra start==goal returns {42} cost 0");

                  auto r_par_a = parallel_a_star(42, goal_at_start, successors, cost, heuristic);
                  t.expect(r_par_a.has_value() && r_par_a->first == expected_path && r_par_a->second == 0,
                           "parallel_a_star start-is-goal returns {42} cost 0");

                  auto r_par_d = parallel_dijkstra(42, goal_at_start, successors, cost);
                  t.expect(r_par_d.has_value() && r_par_d->first == expected_path && r_par_d->second == 0,
                           "parallel_dijkstra start-is-goal returns {42} cost 0");
              })
        .test("ida_star_returns_undefined_value_when_goal_is_unreachable",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto successors = [](std::int64_t u) -> std::vector<std::int64_t> {
                      if (u == 0) return {1};
                      if (u == 1) return {2};
                      return {};
                  };
                  const auto cost = [](std::int64_t, std::int64_t) -> std::int64_t { return 1; };
                  const auto heuristic = [](std::int64_t) -> std::int64_t { return 0; };
                  const auto unreachable_goal = [](std::int64_t u) -> bool { return u == 99; };

                  auto r = ida_star(0, unreachable_goal, successors, cost, heuristic, 50);
                  t.expect(!r.has_value() && r.error() == MathError::undefined_value,
                           "unreachable goal returns undefined_value in ida_star");
              })
        .test("bidirectional_dijkstra_returns_undefined_value_when_goal_is_unreachable",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto successors = [](std::int64_t u) -> std::vector<std::int64_t> {
                      if (u == 0) return {1};
                      return {};
                  };
                  const auto predecessors = [](std::int64_t v) -> std::vector<std::int64_t> {
                      if (v == 1) return {0};
                      return {};
                  };
                  const auto cost = [](std::int64_t, std::int64_t) -> std::int64_t { return 1; };

                  auto r = bidirectional_dijkstra(0, 99, successors, predecessors, cost);
                  t.expect(!r.has_value() && r.error() == MathError::undefined_value,
                           "unreachable goal returns undefined_value in bidirectional_dijkstra");
              })
        .test("negative_edge_cost_returns_domain_error_for_all_cost_algorithms",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto successors = [](std::int64_t u) -> std::vector<std::int64_t> {
                      if (u == 0) return {1};
                      return {};
                  };
                  const auto predecessors = [](std::int64_t v) -> std::vector<std::int64_t> {
                      if (v == 1) return {0};
                      return {};
                  };
                  const auto negative_cost = [](std::int64_t, std::int64_t) -> std::int64_t { return -5; };
                  const auto heuristic = [](std::int64_t) -> std::int64_t { return 0; };
                  const auto goal = [](std::int64_t u) -> bool { return u == 1; };

                  auto r_ida = ida_star(0, goal, successors, negative_cost, heuristic, 10);
                  t.expect(!r_ida.has_value() && r_ida.error() == MathError::domain_error,
                           "negative edge cost => domain_error in ida_star");

                  auto r_wastar = weighted_a_star(0, goal, successors, negative_cost, heuristic, 1, 1);
                  t.expect(!r_wastar.has_value() && r_wastar.error() == MathError::domain_error,
                           "negative edge cost => domain_error in weighted_a_star");

                  auto r_bidi = bidirectional_dijkstra(0, 1, successors, predecessors, negative_cost);
                  t.expect(!r_bidi.has_value() && r_bidi.error() == MathError::domain_error,
                           "negative edge cost => domain_error in bidirectional_dijkstra");

                  auto r_par_a = parallel_a_star(0, goal, successors, negative_cost, heuristic);
                  t.expect(!r_par_a.has_value() && r_par_a.error() == MathError::domain_error,
                           "negative edge cost => domain_error in parallel_a_star");

                  auto r_par_d = parallel_dijkstra(0, goal, successors, negative_cost);
                  t.expect(!r_par_d.has_value() && r_par_d.error() == MathError::domain_error,
                           "negative edge cost => domain_error in parallel_dijkstra");
              })
        .test("negative_heuristic_returns_domain_error_for_all_heuristic_algorithms",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto successors = [](std::int64_t u) -> std::vector<std::int64_t> {
                      if (u == 0) return {1};
                      return {};
                  };
                  const auto cost = [](std::int64_t, std::int64_t) -> std::int64_t { return 1; };
                  const auto neg_heuristic = [](std::int64_t) -> std::int64_t { return -1; };
                  const auto goal = [](std::int64_t u) -> bool { return u == 1; };

                  auto r_ida = ida_star(0, goal, successors, cost, neg_heuristic, 10);
                  t.expect(!r_ida.has_value() && r_ida.error() == MathError::domain_error,
                           "negative heuristic => domain_error in ida_star");

                  auto r_wastar = weighted_a_star(0, goal, successors, cost, neg_heuristic, 1, 1);
                  t.expect(!r_wastar.has_value() && r_wastar.error() == MathError::domain_error,
                           "negative heuristic => domain_error in weighted_a_star");

                  auto r_gbf = greedy_best_first(0, goal, successors, neg_heuristic, 10);
                  t.expect(!r_gbf.has_value() && r_gbf.error() == MathError::domain_error,
                           "negative heuristic => domain_error in greedy_best_first");

                  auto r_beam = beam_search(0, goal, successors, neg_heuristic, 5, 10);
                  t.expect(!r_beam.has_value() && r_beam.error() == MathError::domain_error,
                           "negative heuristic => domain_error in beam_search");
              })
        .test("ida_star_exhausting_max_iterations_returns_not_converged",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto successors = [](std::int64_t u) -> std::vector<std::int64_t> {
                      if (u >= 0 && u < 3) return {u + 1};
                      return {};
                  };
                  const auto cost = [](std::int64_t, std::int64_t) -> std::int64_t { return 10; };
                  const auto heuristic = [](std::int64_t) -> std::int64_t { return 0; };
                  const auto goal = [](std::int64_t u) -> bool { return u == 3; };

                  auto r0 = ida_star(0, goal, successors, cost, heuristic, 0);
                  t.expect(!r0.has_value() && r0.error() == MathError::not_converged,
                           "max_iterations=0 => not_converged in ida_star");

                  auto r1 = ida_star(0, goal, successors, cost, heuristic, 1);
                  t.expect(!r1.has_value() && r1.error() == MathError::not_converged,
                           "max_iterations=1 with remaining threshold passes => not_converged in ida_star");
              })
        .test("greedy_best_first_exhausting_max_expansions_returns_not_converged",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto successors = [](std::int64_t u) -> std::vector<std::int64_t> {
                      if (u >= 0 && u < 4) return {u + 1};
                      return {};
                  };
                  const auto heuristic = [](std::int64_t u) -> std::int64_t { return 10 - u; };
                  const auto goal = [](std::int64_t u) -> bool { return u == 4; };

                  auto r0 = greedy_best_first(0, goal, successors, heuristic, 0);
                  t.expect(!r0.has_value() && r0.error() == MathError::not_converged,
                           "max_expansions=0 => not_converged in greedy_best_first");

                  auto r1 = greedy_best_first(0, goal, successors, heuristic, 1);
                  t.expect(!r1.has_value() && r1.error() == MathError::not_converged,
                           "max_expansions=1 on multi-step path => not_converged in greedy_best_first");
              })
        .test("greedy_best_first_finds_path_guided_by_heuristic",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto successors = [](std::int64_t u) -> std::vector<std::int64_t> {
                      switch (u) {
                          case 0: return {1, 2};
                          case 1: return {3};
                          case 2: return {3};
                          default: return {};
                      }
                  };
                  const auto heuristic = [](std::int64_t u) -> std::int64_t {
                      if (u == 1) return 1;
                      if (u == 2) return 50;
                      return 0;
                  };
                  const auto goal = [](std::int64_t u) -> bool { return u == 3; };

                  auto r = greedy_best_first(0, goal, successors, heuristic, 10);
                  t.expect(r.has_value(), "greedy_best_first finds a path");
                  if (r.has_value()) {
                      const std::vector<std::int64_t> expected_path{0, 1, 3};
                      t.expect(*r == expected_path, "greedy_best_first chose branch with lowest heuristic {0, 1, 3}");
                  }
              })
        .test("greedy_best_first_returns_undefined_value_when_frontier_exhausted",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto successors = [](std::int64_t u) -> std::vector<std::int64_t> {
                      if (u == 0) return {1};
                      return {};
                  };
                  const auto heuristic = [](std::int64_t) -> std::int64_t { return 0; };
                  const auto unreachable_goal = [](std::int64_t u) -> bool { return u == 99; };

                  auto r = greedy_best_first(0, unreachable_goal, successors, heuristic, 10);
                  t.expect(!r.has_value() && r.error() == MathError::undefined_value,
                           "unreachable goal returns undefined_value in greedy_best_first");
              })
        .test("beam_search_invalid_arguments_return_domain_error",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto successors = [](std::int64_t) -> std::vector<std::int64_t> { return {}; };
                  const auto heuristic = [](std::int64_t) -> std::int64_t { return 0; };
                  const auto goal = [](std::int64_t u) -> bool { return u == 5; };

                  auto r_w0 = beam_search(0, goal, successors, heuristic, 0, 10);
                  t.expect(!r_w0.has_value() && r_w0.error() == MathError::domain_error,
                           "beam_width=0 => domain_error in beam_search");

                  auto r_wneg = beam_search(0, goal, successors, heuristic, -1, 10);
                  t.expect(!r_wneg.has_value() && r_wneg.error() == MathError::domain_error,
                           "beam_width<0 => domain_error in beam_search");

                  auto r_lneg = beam_search(0, goal, successors, heuristic, 2, -1);
                  t.expect(!r_lneg.has_value() && r_lneg.error() == MathError::domain_error,
                           "max_levels<0 => domain_error in beam_search");
              })
        .test("beam_search_with_width_one_acts_as_greedy_chain",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto successors = [](std::int64_t u) -> std::vector<std::int64_t> {
                      switch (u) {
                          case 0: return {1, 2};
                          case 2: return {3, 4};
                          case 4: return {5};
                          default: return {};
                      }
                  };
                  const auto heuristic = [](std::int64_t u) -> std::int64_t {
                      switch (u) {
                          case 1: return 10;
                          case 2: return 2;
                          case 3: return 5;
                          case 4: return 1;
                          case 5: return 0;
                          default: return 0;
                      }
                  };
                  const auto goal = [](std::int64_t u) -> bool { return u == 5; };

                  auto r = beam_search(0, goal, successors, heuristic, 1, 10);
                  t.expect(r.has_value(), "beam_search with width 1 finds path");
                  if (r.has_value()) {
                      const std::vector<std::int64_t> expected_path{0, 2, 4, 5};
                      t.expect(*r == expected_path, "beam_search width 1 greedily selects {0, 2, 4, 5}");
                  }
              })
        .test("beam_search_narrow_beam_misses_goal_and_returns_undefined_value",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto successors = [](std::int64_t u) -> std::vector<std::int64_t> {
                      switch (u) {
                          case 0: return {1, 2};
                          case 2: return {3};
                          default: return {};
                      }
                  };
                  const auto heuristic = [](std::int64_t u) -> std::int64_t {
                      if (u == 1) return 1;
                      if (u == 2) return 50;
                      return 0;
                  };
                  const auto goal = [](std::int64_t u) -> bool { return u == 3; };

                  auto r = beam_search(0, goal, successors, heuristic, 1, 5);
                  t.expect(!r.has_value() && r.error() == MathError::undefined_value,
                           "narrow beam prunes only reaching branch => returns undefined_value honestly");
              })
        .test("beam_search_exhausting_max_levels_returns_not_converged",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto successors = [](std::int64_t u) -> std::vector<std::int64_t> {
                      if (u >= 0 && u < 4) return {u + 1};
                      return {};
                  };
                  const auto heuristic = [](std::int64_t u) -> std::int64_t { return 10 - u; };
                  const auto goal = [](std::int64_t u) -> bool { return u == 4; };

                  auto r0 = beam_search(0, goal, successors, heuristic, 5, 0);
                  t.expect(!r0.has_value() && r0.error() == MathError::not_converged,
                           "max_levels=0 => not_converged in beam_search");

                  auto r2 = beam_search(0, goal, successors, heuristic, 5, 2);
                  t.expect(!r2.has_value() && r2.error() == MathError::not_converged,
                           "max_levels=2 exhausted before depth 4 => not_converged in beam_search");
              })
        .test("parallel_neighbourhood_scan_returns_ordered_evaluations",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const TabuState state{5, 10};
                  const auto neighbors = [](const TabuState&) -> std::vector<TabuState> {
                      return {TabuState{4, 10}, TabuState{6, 10}, TabuState{5, 9}, TabuState{5, 11}};
                  };
                  const auto objective = [](const TabuState& s) -> std::int64_t {
                      if (s.size() < 2) return 0;
                      return s[0] * 100 + s[1];
                  };

                  auto r = parallel_neighbourhood_scan(state, neighbors, objective);
                  t.expect(r.has_value(), "parallel_neighbourhood_scan succeeds");
                  if (r.has_value()) {
                      t.expect(r->size() == 4, "scan returns exactly 4 evaluations");
                      if (r->size() == 4) {
                          t.expect((*r)[0].first == TabuState{4, 10} && (*r)[0].second == 410,
                                   "first neighbour scored 410 in index order");
                          t.expect((*r)[1].first == TabuState{6, 10} && (*r)[1].second == 610,
                                   "second neighbour scored 610 in index order");
                          t.expect((*r)[2].first == TabuState{5, 9} && (*r)[2].second == 509,
                                   "third neighbour scored 509 in index order");
                          t.expect((*r)[3].first == TabuState{5, 11} && (*r)[3].second == 511,
                                   "fourth neighbour scored 511 in index order");
                      }
                  }
              })
        .test("parallel_neighbourhood_scan_empty_neighbours_returns_empty",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const TabuState state{0};
                  const auto empty_neighbors = [](const TabuState&) -> std::vector<TabuState> { return {}; };
                  const auto objective = [](const TabuState&) -> std::int64_t { return 42; };

                  auto r = parallel_neighbourhood_scan(state, empty_neighbors, objective);
                  t.expect(r.has_value(), "parallel_neighbourhood_scan succeeds with empty neighbours");
                  if (r.has_value()) {
                      t.expect(r->empty(), "result vector is empty when neighbours list is empty");
                  }
              })
        .test("parallel_multistart_tabu_guards_domain_errors",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto neighbors = [](const TabuState&) -> std::vector<TabuState> { return {}; };
                  const auto objective = [](const TabuState&) -> std::int64_t { return 0; };

                  auto r_empty = parallel_multistart_tabu({}, neighbors, objective, 3, 10);
                  t.expect(!r_empty.has_value() && r_empty.error() == MathError::domain_error,
                           "empty starts vector => domain_error in parallel_multistart_tabu");

                  auto r_neg_tenure = parallel_multistart_tabu({{0}}, neighbors, objective, -1, 10);
                  t.expect(!r_neg_tenure.has_value() && r_neg_tenure.error() == MathError::domain_error,
                           "negative tabu_tenure => domain_error in parallel_multistart_tabu");

                  auto r_neg_iters = parallel_multistart_tabu({{0}}, neighbors, objective, 3, -1);
                  t.expect(!r_neg_iters.has_value() && r_neg_iters.error() == MathError::domain_error,
                           "negative max_iters => domain_error in parallel_multistart_tabu");
              })
        .test("weighted_a_star_invalid_weights_return_domain_error",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto successors = [](std::int64_t u) -> std::vector<std::int64_t> {
                      if (u == 0) return {1};
                      return {};
                  };
                  const auto cost = [](std::int64_t, std::int64_t) -> std::int64_t { return 1; };
                  const auto heuristic = [](std::int64_t) -> std::int64_t { return 0; };
                  const auto goal = [](std::int64_t u) -> bool { return u == 1; };

                  auto r_zero_den = weighted_a_star(0, goal, successors, cost, heuristic, 1, 0);
                  t.expect(!r_zero_den.has_value() && r_zero_den.error() == MathError::domain_error,
                           "weight_den=0 => domain_error in weighted_a_star");

                  auto r_sub_one = weighted_a_star(0, goal, successors, cost, heuristic, 1, 2);
                  t.expect(!r_sub_one.has_value() && r_sub_one.error() == MathError::domain_error,
                           "weight < 1 (1/2) => domain_error in weighted_a_star");

                  auto r_neg = weighted_a_star(0, goal, successors, cost, heuristic, -3, 2);
                  t.expect(!r_neg.has_value() && r_neg.error() == MathError::domain_error,
                           "negative weight ratio => domain_error in weighted_a_star");
              })
        .test("ida_star_matches_a_star_on_grid_with_obstacles",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto successors = [](std::int64_t u) -> std::vector<std::int64_t> {
                      const std::int64_t r = u / 4;
                      const std::int64_t c = u % 4;
                      std::vector<std::int64_t> succs;
                      if (c + 1 < 4) {
                          const std::int64_t right = r * 4 + (c + 1);
                          if (right != 5 && right != 6 && right != 9) succs.push_back(right);
                      }
                      if (r + 1 < 4) {
                          const std::int64_t down = (r + 1) * 4 + c;
                          if (down != 5 && down != 6 && down != 9) succs.push_back(down);
                      }
                      return succs;
                  };
                  const auto cost = [](std::int64_t, std::int64_t) -> std::int64_t { return 1; };
                  const auto heuristic = [](std::int64_t u) -> std::int64_t {
                      const std::int64_t r = u / 4;
                      const std::int64_t c = u % 4;
                      return (3 - r) + (3 - c);
                  };
                  const auto goal = [](std::int64_t u) -> bool { return u == 15; };

                  auto r_ida = ida_star(0, goal, successors, cost, heuristic, 100);
                  auto r_astar = a_star(0, goal, successors, cost, heuristic);

                  t.expect(r_ida.has_value(), "ida_star succeeds on obstacle grid");
                  t.expect(r_astar.has_value(), "a_star succeeds on obstacle grid");
                  if (r_ida.has_value() && r_astar.has_value()) {
                      t.expect(r_ida->second == r_astar->second, "ida_star cost matches a_star cost on obstacle grid");
                      t.expect(r_ida->second == 6, "optimal cost avoiding obstacles is exactly 6");
                  }
              })
        .test("parallel_a_star_and_parallel_dijkstra_on_unreachable_goal_return_undefined_value",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  const auto successors = [](std::int64_t u) -> std::vector<std::int64_t> {
                      if (u == 0) return {1};
                      return {};
                  };
                  const auto cost = [](std::int64_t, std::int64_t) -> std::int64_t { return 1; };
                  const auto heuristic = [](std::int64_t) -> std::int64_t { return 0; };
                  const auto unreachable_goal = [](std::int64_t u) -> bool { return u == 99; };

                  auto r_a = parallel_a_star(0, unreachable_goal, successors, cost, heuristic);
                  t.expect(!r_a.has_value() && r_a.error() == MathError::undefined_value,
                           "unreachable goal returns undefined_value in parallel_a_star");

                  auto r_d = parallel_dijkstra(0, unreachable_goal, successors, cost);
                  t.expect(!r_d.has_value() && r_d.error() == MathError::undefined_value,
                           "unreachable goal returns undefined_value in parallel_dijkstra");
              })
        .test("a_zero_cost_self_loop_does_not_make_a_node_its_own_predecessor",
              [](TestContext& t) {
                  // Regression. The equal-distance predecessor tie-break used to fire across a
                  // zero-cost edge, so relaxing a self-loop at node 2 could set pred[2] = 2. Path
                  // reconstruction then walked 2 -> 2 -> 2 forever and the call never returned.
                  // A hang is the one failure a test cannot report, so this pins it directly.
                  const auto successors = [](std::int64_t u) -> std::vector<std::int64_t> {
                      switch (u) {
                          case 0: return {1};
                          case 1: return {2};
                          case 2: return {2, 3};   // zero-cost self-loop, then onward
                          default: return {};
                      }
                  };
                  const auto cost = [](std::int64_t u, std::int64_t v) -> std::int64_t {
                      if (u == 2 && v == 2) {
                          return 0;
                      }
                      return 1;
                  };
                  const auto goal = [](std::int64_t u) -> bool { return u == 3; };
                  auto r = dijkstra(0, goal, successors, cost);
                  t.expect(r.has_value(), "dijkstra returns rather than spinning on the self-loop");
                  if (!r.has_value()) {
                      return;
                  }
                  const std::vector<std::int64_t> expected{0, 1, 2, 3};
                  t.expect(r->first == expected, "the path is exactly {0, 1, 2, 3}");
                  t.expect(r->second == 3, "the cost is exactly 3");
              })
        .test("a_zero_cost_two_cycle_does_not_create_a_predecessor_cycle",
              [](TestContext& t) {
                  // The same hazard without a self-loop: nodes 1 and 2 sit at equal distance via
                  // a zero-cost cycle, so the tie-break could set pred[1] = 2 and pred[2] = 1.
                  const auto successors = [](std::int64_t u) -> std::vector<std::int64_t> {
                      switch (u) {
                          case 0: return {1};
                          case 1: return {2};
                          case 2: return {1, 3};
                          default: return {};
                      }
                  };
                  const auto cost = [](std::int64_t u, std::int64_t v) -> std::int64_t {
                      if ((u == 1 && v == 2) || (u == 2 && v == 1)) {
                          return 0;
                      }
                      return 2;
                  };
                  const auto goal = [](std::int64_t u) -> bool { return u == 3; };
                  auto r = dijkstra(0, goal, successors, cost);
                  t.expect(r.has_value(), "dijkstra returns on a zero-cost two-cycle");
                  if (!r.has_value()) {
                      return;
                  }
                  t.expect(r->second == 4, "the cost is exactly 4");
                  const std::vector<std::int64_t> expected{0, 1, 2, 3};
                  t.expect(r->first == expected, "the path is exactly {0, 1, 2, 3}");
                  auto ra = a_star(0, goal, successors, cost,
                                   [](std::int64_t) -> std::int64_t { return 0; });
                  t.expect(ra.has_value() && ra->second == 4,
                           "a_star returns the same cost on the same graph");
                  auto rp = parallel_dijkstra(0, goal, successors, cost);
                  t.expect(rp.has_value() && rp->second == 4,
                           "parallel_dijkstra returns the same cost on the same graph");
                  auto rb = bidirectional_dijkstra(
                      0, 3, successors,
                      [](std::int64_t u) -> std::vector<std::int64_t> {
                          switch (u) {
                              case 1: return {0, 2};
                              case 2: return {1};
                              case 3: return {2};
                              default: return {};
                          }
                      },
                      cost);
                  t.expect(rb.has_value() && rb->second == 4,
                           "bidirectional_dijkstra returns the same cost on the same graph");
              })
        .run();
}
