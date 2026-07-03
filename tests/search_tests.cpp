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
                  auto p = bfs(0, is(6), successors);
                  t.expect(p.has_value(), "bfs finds a path");
                  t.expect(p.value_or(Path{}) == Path{0, 2, 6}, "bfs path is [0, 2, 6]");
              })
        .test("bfs_start_is_goal",
              [](TestContext& t) {
                  auto p = bfs(0, is(0), successors);
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
                  auto p = dfs_recursive(0, is(6), successors, 10);
                  t.expect(p.has_value(), "dfs_recursive finds a path");
                  t.expect(p.value_or(Path{}) == Path{0, 1, 3, 4, 6},
                           "dfs path is [0, 1, 3, 4, 6]");
              })
        .test("dfs_iterative_matches_recursive",
              [](TestContext& t) {
                  auto pr = dfs_recursive(0, is(6), successors, 10);
                  auto pi = dfs_iterative(0, is(6), successors, 10);
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
                  auto p = iterative_deepening_dfs(0, is(6), successors, 10);
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
                  auto p = parallel_bfs_levels(0, successors, 10);
                  t.expect(p.has_value(), "parallel_bfs_levels succeeds");
                  const Levels expected{{0}, {1, 2}, {3, 6}, {4}};
                  t.expect(p.value_or(Levels{}) == expected,
                           "levels are [[0],[1,2],[3,6],[4]]");
                  t.expect(p.value_or(Levels{}) == serial_bfs_levels(0, 10),
                           "parallel levels == serial level-BFS");
              })
        .test("parallel_bfs_levels_truncates_and_guards",
              [](TestContext& t) {
                  auto two = parallel_bfs_levels(0, successors, 2);
                  const Levels expected2{{0}, {1, 2}};
                  t.expect(two.value_or(Levels{}) == expected2, "max_levels=2 => first 2 levels");
                  auto zero = parallel_bfs_levels(0, successors, 0);
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
                  auto objective = [](const TabuState& s) -> std::int64_t {
                      const std::int64_t dx = s[0] - 3;
                      const std::int64_t dy = s[1] + 2;
                      return dx * dx + dy * dy;
                  };
                  auto neighbors = [](const TabuState& s) -> std::vector<TabuState> {
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
                  auto objective = [](const TabuState& s) -> std::int64_t { return s[0]; };
                  auto neighbors = [](const TabuState&) -> std::vector<TabuState> { return {}; };
                  auto r = tabu_search(TabuState{0}, neighbors, objective, -1, 10);
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "negative tenure => domain_error");
              })
        .test("edit_distance_textbook",
              [](TestContext& t) {
                  auto tab = edit_distance("kitten", "sitting");
                  auto memo = edit_distance_memo("kitten", "sitting");
                  auto par = edit_distance_parallel("kitten", "sitting");
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
                  auto tab = longest_common_subsequence(a, b);
                  auto memo = longest_common_subsequence_memo(a, b);
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
                  auto r = knapsack_01(w, v, 7);
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
        .run();
}
