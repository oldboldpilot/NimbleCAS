// Feature/integration tests: the reasoning / algorithmics layer end to end
// (search, sat, csp, logic, bitset, bitcsp).
// @author Olumuyiwa Oluwasanmi
//
// These are cross-module workflow and invariant tests, NOT per-module unit tests. The theme
// is CROSS-SOLVER CONSISTENCY: wherever two engines can decide the same problem they must
// AGREE, and every documented honesty boundary is respected rather than papered over:
//
//   * search — BFS gives the fewest-edges path; Dijkstra and A* (admissible heuristic) agree
//     on the optimal cost; IDDFS returns the same depth as BFS; recursive/iterative DFS agree
//     bit-for-bit; the DP routines (edit distance / LCS / knapsack) match hand values and their
//     tabulated == memoised == parallel forms coincide; parallel level-BFS == serial.
//   * sat — a SAT formula's returned model actually satisfies every clause (checked by
//     evaluation); the COMPLETE solvers (DPLL/CDCL) prove a tiny UNSAT instance; the STOCHASTIC
//     solvers (WalkSAT/GSAT) find a model but are asserted NEVER to return `unsatisfiable`;
//     solve_portfolio and a manual solve_shard all-reduce agree with DPLL's verdict.
//   * csp — AC-3 prunes a known instance exactly and correctly reports arc-inconsistency, yet is
//     honestly NOT a complete solver (an arc-consistent-but-unsatisfiable CSP); backtracking,
//     forward-checking and the parallel search return the identical lexicographically-first
//     solution; N-queens solution counts (n = 4..8) match the known values.
//   * logic — unification honours the occurs-check; SLD resolution answers ancestor/append
//     queries; OR-parallel == serial within budget.
//   * bitset/bitcsp — word-parallel set ops match a std::set reference; bitcsp bitmask N-queens
//     counts equal the csp counts (two independent implementations cross-checked) for n = 4..8;
//     the parallel count equals the serial count.
//
// Everything here is deterministic: the complete solvers are pure functions of their input, the
// stochastic ones are seeded, and every parallel routine is contractually thread-count invariant.

import std;
import nimblecas.core;
import nimblecas.search;
import nimblecas.sat;
import nimblecas.csp;
import nimblecas.logic;
import nimblecas.bitset;
import nimblecas.bitcsp;
import nimblecas.testing;

using namespace nimblecas;                 // the whole reasoning surface lives in nimblecas::
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// --- shared builders --------------------------------------------------------

// A small weighted DAG on nodes 0..4 used across the search tests.
//   0 ->1 (1)  0 ->2 (4)  1 ->2 (2)  1 ->3 (5)  2 ->3 (1)  2 ->4 (7)  3 ->4 (3)
// Fewest-edges 0->4 is [0,2,4] (2 edges); the minimum-cost 0->4 is [0,1,2,3,4] at cost 7.
[[nodiscard]] auto graph_successors() -> SuccessorFn {
    return [](std::int64_t u) -> std::vector<std::int64_t> {
        switch (u) {
            case 0:
                return {1, 2};
            case 1:
                return {2, 3};
            case 2:
                return {3, 4};
            case 3:
                return {4};
            default:
                return {};
        }
    };
}

[[nodiscard]] auto graph_cost() -> CostFn {
    return [](std::int64_t u, std::int64_t v) -> std::int64_t {
        if (u == 0 && v == 1) return 1;
        if (u == 0 && v == 2) return 4;
        if (u == 1 && v == 2) return 2;
        if (u == 1 && v == 3) return 5;
        if (u == 2 && v == 3) return 1;
        if (u == 2 && v == 4) return 7;
        if (u == 3 && v == 4) return 3;
        return 0;  // no such edge is ever traversed
    };
}

// |a - b| without risking int64 negation overflow.
[[nodiscard]] auto abs_diff(std::int64_t a, std::int64_t b) -> std::int64_t {
    return a >= b ? a - b : b - a;
}

// The n-queens problem as a finite-domain CSP: variable i is the column of the queen on row i,
// domain [0, n), with a binary constraint on each row pair enforcing distinct columns and
// distinct diagonals. Cross-checked against the branchless bitcsp count and the known counts.
[[nodiscard]] auto queens_csp(int n) -> Csp {
    Csp csp;
    std::vector<std::int64_t> dom;
    dom.reserve(static_cast<std::size_t>(n));
    for (int v = 0; v < n; ++v) {
        dom.push_back(v);
    }
    csp.domains.assign(static_cast<std::size_t>(n), dom);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
        for (std::size_t j = i + 1; j < static_cast<std::size_t>(n); ++j) {
            const std::int64_t d = static_cast<std::int64_t>(j - i);
            csp.binary.push_back(BinaryConstraint{
                .x = i,
                .y = j,
                .allowed = [d](std::int64_t a, std::int64_t b) {
                    return a != b && abs_diff(a, b) != d;
                }});
        }
    }
    return csp;
}

// True iff `cols` is a valid non-attacking n-queens placement (one column per row).
[[nodiscard]] auto valid_queens(const std::vector<std::int64_t>& cols) -> bool {
    for (std::size_t i = 0; i < cols.size(); ++i) {
        for (std::size_t j = i + 1; j < cols.size(); ++j) {
            if (cols[i] == cols[j]) {
                return false;
            }
            if (abs_diff(cols[i], cols[j]) == static_cast<std::int64_t>(j - i)) {
                return false;
            }
        }
    }
    return true;
}

// The deterministic associative all-reduce solve_portfolio / solve_shard use: any UNSAT wins;
// else the model from the lowest index; else unknown. Replicated here to check the shard
// contract without depending on the module's private merge.
[[nodiscard]] auto merge_shards(const std::vector<SatResult>& rs) -> SatResult {
    for (const auto& r : rs) {
        if (r.verdict == SatVerdict::unsatisfiable) {
            return SatResult{SatVerdict::unsatisfiable, {}};
        }
    }
    for (const auto& r : rs) {
        if (r.verdict == SatVerdict::satisfiable) {
            return r;
        }
    }
    return SatResult{SatVerdict::unknown, {}};
}

// The binding of query variable `name` in a solved substitution, rendered as a Prolog string.
[[nodiscard]] auto binding_of(const Substitution& s, std::string_view name) -> std::string {
    for (const auto& [k, v] : s) {
        if (k.name == name) {
            return to_string(v);
        }
    }
    return "<unbound>";
}

}  // namespace

auto main() -> int {
    return TestSuite("feature.reasoning")
        // ===================================================================
        // SEARCH: graph traversal, shortest paths, and dynamic programming.
        // ===================================================================
        .test("search_bfs_dijkstra_astar_on_known_graph",
              [](TestContext& t) {
                  const auto succ = graph_successors();
                  const auto cost = graph_cost();
                  const auto is4 = [](std::int64_t x) { return x == 4; };

                  // BFS returns the FEWEST-EDGES path: 0 -> 2 -> 4.
                  auto bp = bfs(0, is4, succ);
                  t.expect(bp.has_value(), "bfs reaches node 4");
                  if (bp) {
                      const std::vector<std::int64_t> want{0, 2, 4};
                      t.expect(*bp == want, "bfs fewest-edges path is [0,2,4]");
                  }

                  // Dijkstra returns the MINIMUM-COST path and cost: [0,1,2,3,4], cost 7.
                  auto dj = dijkstra(0, is4, succ, cost);
                  t.expect(dj.has_value(), "dijkstra reaches node 4");
                  if (dj) {
                      const std::vector<std::int64_t> want{0, 1, 2, 3, 4};
                      t.expect(dj->first == want, "dijkstra optimal path is [0,1,2,3,4]");
                      t.expect(dj->second == 7,
                               std::format("dijkstra optimal cost is 7 (got {})", dj->second));
                  }

                  // A* with an ADMISSIBLE heuristic (a lower bound on the true remaining cost to
                  // node 4: true distances are 7,6,4,3,0 for nodes 0..4) must return the SAME
                  // optimal cost as Dijkstra.
                  const auto heur = [](std::int64_t x) -> std::int64_t {
                      switch (x) {
                          case 0:
                              return 5;  // <= 7
                          case 1:
                              return 4;  // <= 6
                          case 2:
                              return 3;  // <= 4
                          case 3:
                              return 2;  // <= 3
                          default:
                              return 0;
                      }
                  };
                  auto as = a_star(0, is4, succ, cost, heur);
                  t.expect(as.has_value(), "a_star reaches node 4");
                  if (as && dj) {
                      t.expect(as->second == dj->second,
                               std::format("a_star cost == dijkstra cost ({} vs {})",
                                           as->second, dj->second));
                      t.expect(as->first == dj->first,
                               "a_star recovers the same optimal path as dijkstra");
                  }

                  // A ZERO heuristic is trivially admissible, so A* must reduce to Dijkstra.
                  auto as0 = a_star(0, is4, succ, cost, [](std::int64_t) { return 0; });
                  t.expect(as0.has_value() && dj.has_value() && as0->second == dj->second,
                           "a_star with h==0 equals dijkstra cost");
              })
        .test("search_iddfs_depth_matches_bfs_and_dfs_variants_agree",
              [](TestContext& t) {
                  const auto succ = graph_successors();
                  const auto is4 = [](std::int64_t x) { return x == 4; };

                  auto bp = bfs(0, is4, succ);
                  auto id = iterative_deepening_dfs(0, is4, succ, 10);
                  t.expect(bp.has_value() && id.has_value(), "bfs and iddfs both reach node 4");
                  if (bp && id) {
                      // IDDFS returns the SHALLOWEST goal, so its edge count matches BFS's.
                      t.expect(bp->size() == id->size(),
                               std::format("iddfs depth == bfs depth ({} vs {} nodes)",
                                           id->size(), bp->size()));
                  }

                  // Recursive and iterative DFS reproduce the identical pre-order first-goal path.
                  auto dr = dfs_recursive(0, is4, succ, 6);
                  auto di = dfs_iterative(0, is4, succ, 6);
                  t.expect(dr.has_value() && di.has_value(), "both DFS variants reach node 4");
                  if (dr && di) {
                      t.expect(*dr == *di,
                               "dfs_recursive and dfs_iterative return the identical path");
                      const std::vector<std::int64_t> want{0, 1, 2, 3, 4};
                      t.expect(*dr == want, "DFS pre-order path is [0,1,2,3,4]");
                  }

                  // A negative depth bound is a domain_error for the depth-limited searches.
                  auto bad = dfs_recursive(0, is4, succ, -1);
                  t.expect(!bad.has_value() && bad.error() == MathError::domain_error,
                           "dfs_recursive with max_depth < 0 is a domain_error");
              })
        .test("search_error_paths_and_parallel_bfs_levels",
              [](TestContext& t) {
                  const auto succ = graph_successors();

                  // An unreachable goal is undefined_value, not a crash.
                  auto none = bfs(0, [](std::int64_t x) { return x == 99; }, succ);
                  t.expect(!none.has_value() && none.error() == MathError::undefined_value,
                           "bfs to an unreachable goal is undefined_value");

                  // A negative edge weight makes Dijkstra a domain_error.
                  auto neg = dijkstra(0, [](std::int64_t x) { return x == 4; }, succ,
                                      [](std::int64_t, std::int64_t) { return -1; });
                  t.expect(!neg.has_value() && neg.error() == MathError::domain_error,
                           "dijkstra with a negative edge weight is a domain_error");

                  // Parallel level-BFS reproduces a serial level-BFS bit-for-bit (each level is an
                  // ascending, de-duplicated set): from node 0 the levels are [[0],[1,2],[3,4]].
                  auto lv = parallel_bfs_levels(0, succ, 10);
                  t.expect(lv.has_value(), "parallel_bfs_levels succeeds");
                  if (lv) {
                      const std::vector<std::vector<std::int64_t>> want{{0}, {1, 2}, {3, 4}};
                      t.expect(*lv == want,
                               "parallel_bfs_levels == serial level-BFS ([[0],[1,2],[3,4]])");
                      // Node 4 first appears at level 2, matching its BFS edge-distance.
                      t.expect(lv->size() == 3 &&
                                   std::find((*lv)[2].begin(), (*lv)[2].end(), 4) != (*lv)[2].end(),
                               "node 4 sits on level 2 (its BFS distance)");
                  }
              })
        .test("search_dp_matches_hand_values_and_forms_agree",
              [](TestContext& t) {
                  // Edit distance: tabulated == memoised == parallel, all == 3 for kitten/sitting.
                  auto ed = edit_distance("kitten", "sitting");
                  auto em = edit_distance_memo("kitten", "sitting");
                  auto ep = edit_distance_parallel("kitten", "sitting");
                  t.expect(ed.has_value() && em.has_value() && ep.has_value(),
                           "all three edit-distance forms succeed");
                  if (ed && em && ep) {
                      t.expect(*ed == 3, std::format("edit_distance(kitten,sitting)==3 (got {})",
                                                     *ed));
                      t.expect(*ed == *em && *em == *ep,
                               "tabulated == memoised == parallel edit distance");
                  }

                  // Boundary cases: distance to the empty string is the other length; identical
                  // strings have distance 0. Parallel form must agree everywhere.
                  auto e0 = edit_distance("", "abc");
                  auto e0p = edit_distance_parallel("", "abc");
                  auto e1 = edit_distance("abc", "abc");
                  t.expect(e0.has_value() && e0p.has_value() && *e0 == 3 && *e0 == *e0p,
                           "edit_distance('',abc)==3 and parallel agrees");
                  t.expect(e1.has_value() && *e1 == 0, "edit_distance(abc,abc)==0");

                  // LCS of [1,2,3,2,1] and [2,3,1] is [2,3,1] -> length 3; memo == tabulated.
                  const std::array<std::int64_t, 5> a{1, 2, 3, 2, 1};
                  const std::array<std::int64_t, 3> b{2, 3, 1};
                  std::span<const std::int64_t> sa{a};
                  std::span<const std::int64_t> sb{b};
                  auto lc = longest_common_subsequence(sa, sb);
                  auto lm = longest_common_subsequence_memo(sa, sb);
                  t.expect(lc.has_value() && lm.has_value(), "both LCS forms succeed");
                  if (lc && lm) {
                      t.expect(*lc == 3, std::format("LCS length == 3 (got {})", *lc));
                      t.expect(*lc == *lm, "tabulated LCS == memoised LCS");
                  }

                  // 0/1 knapsack: weights {1,3,4,5}, values {1,4,5,7}, capacity 7 -> best 9
                  // (items of weight 3 and 4, value 4 + 5).
                  const std::array<std::int64_t, 4> w{1, 3, 4, 5};
                  const std::array<std::int64_t, 4> v{1, 4, 5, 7};
                  auto ks = knapsack_01(std::span<const std::int64_t>{w},
                                        std::span<const std::int64_t>{v}, 7);
                  t.expect(ks.has_value() && *ks == 9,
                           std::format("knapsack_01 best value == 9 (got {})",
                                       ks ? *ks : -1));

                  // Mismatched weight/value counts and a negative capacity are domain_errors.
                  const std::array<std::int64_t, 2> w2{1, 2};
                  auto mism = knapsack_01(std::span<const std::int64_t>{w2},
                                          std::span<const std::int64_t>{v}, 7);
                  t.expect(!mism.has_value() && mism.error() == MathError::domain_error,
                           "knapsack_01 with mismatched sizes is a domain_error");
                  auto negcap = knapsack_01(std::span<const std::int64_t>{w},
                                            std::span<const std::int64_t>{v}, -1);
                  t.expect(!negcap.has_value() && negcap.error() == MathError::domain_error,
                           "knapsack_01 with negative capacity is a domain_error");
              })
        .test("search_tabu_minimises_simple_objective",
              [](TestContext& t) {
                  // Minimise (x - 3)^2 over the integer line, neighbours x-1 and x+1, from x = 0.
                  // The best value ever seen must reach 0 at state {3}.
                  const auto neighbors = [](const TabuState& s) -> std::vector<TabuState> {
                      const std::int64_t x = s.front();
                      return {TabuState{x - 1}, TabuState{x + 1}};
                  };
                  const auto objective = [](const TabuState& s) -> std::int64_t {
                      const std::int64_t d = s.front() - 3;
                      return d * d;
                  };
                  auto res = tabu_search(TabuState{0}, neighbors, objective, 2, 50);
                  t.expect(res.has_value(), "tabu_search succeeds");
                  if (res) {
                      t.expect(res->second == 0,
                               std::format("tabu best value == 0 (got {})", res->second));
                      t.expect(res->first == TabuState{3}, "tabu best state == {3}");
                  }

                  // Negative tenure / iteration bounds are domain_errors.
                  auto bad = tabu_search(TabuState{0}, neighbors, objective, -1, 10);
                  t.expect(!bad.has_value() && bad.error() == MathError::domain_error,
                           "tabu_search with negative tenure is a domain_error");
              })
        // ===================================================================
        // SAT: complete vs stochastic, self-verifying models, portfolio & shards.
        // ===================================================================
        .test("sat_satisfiable_model_is_verified_by_every_solver",
              [](TestContext& t) {
                  // (x1 v x2) & (-x1 v x3) & (-x2 v -x3): satisfiable (e.g. x1=T, x2=F, x3=T).
                  const Cnf sat{.num_vars = 3, .clauses = {{1, 2}, {-1, 3}, {-2, -3}}};

                  auto dp = dpll(sat);
                  t.expect(dp.has_value() && dp->verdict == SatVerdict::satisfiable,
                           "dpll reports satisfiable");
                  if (dp && dp->verdict == SatVerdict::satisfiable) {
                      t.expect(verify_assignment(sat, dp->model),
                               "dpll's model actually satisfies every clause");
                  }

                  auto cd = cdcl(sat, 100000);
                  t.expect(cd.has_value() && cd->verdict == SatVerdict::satisfiable,
                           "cdcl reports satisfiable within budget");
                  if (cd && cd->verdict == SatVerdict::satisfiable) {
                      t.expect(verify_assignment(sat, cd->model),
                               "cdcl's model actually satisfies every clause");
                  }

                  // Stochastic local search finds a model on this easy instance and, crucially,
                  // NEVER reports unsatisfiable (it cannot prove UNSAT).
                  auto ws = walksat(sat, 100000, 0.5, 12345);
                  t.expect(ws.has_value() && ws->verdict != SatVerdict::unsatisfiable,
                           "walksat never returns unsatisfiable");
                  if (ws && ws->verdict == SatVerdict::satisfiable) {
                      t.expect(verify_assignment(sat, ws->model),
                               "walksat's model satisfies every clause");
                  }
                  auto gs = gsat(sat, 2000, 200, 777);
                  t.expect(gs.has_value() && gs->verdict != SatVerdict::unsatisfiable,
                           "gsat never returns unsatisfiable");
                  if (gs && gs->verdict == SatVerdict::satisfiable) {
                      t.expect(verify_assignment(sat, gs->model),
                               "gsat's model satisfies every clause");
                  }

                  // A hand-built satisfying assignment must pass verify_assignment; a wrong-sized
                  // model must be rejected.
                  const std::vector<bool> good{true, false, true};  // x1=T, x2=F, x3=T
                  t.expect(verify_assignment(sat, good),
                           "verify_assignment accepts a known-good model");
                  t.expect(!verify_assignment(sat, std::vector<bool>{true, false}),
                           "verify_assignment rejects a model of the wrong size");
              })
        .test("sat_complete_solvers_prove_unsat_stochastic_never_do",
              [](TestContext& t) {
                  // All four clauses over two variables are falsified by some assignment, so the
                  // conjunction is UNSAT; the complete solvers must prove it.
                  const Cnf uns{.num_vars = 2,
                                .clauses = {{1, 2}, {1, -2}, {-1, 2}, {-1, -2}}};
                  auto dp = dpll(uns);
                  t.expect(dp.has_value() && dp->verdict == SatVerdict::unsatisfiable,
                           "dpll proves the 2-var contradiction UNSAT");
                  auto cd = cdcl(uns, 100000);
                  t.expect(cd.has_value() && cd->verdict == SatVerdict::unsatisfiable,
                           "cdcl proves the 2-var contradiction UNSAT");

                  // The trivial contradiction (x) & (-x) is also UNSAT for the complete solvers.
                  const Cnf triv{.num_vars = 1, .clauses = {{1}, {-1}}};
                  auto dpt = dpll(triv);
                  auto cdt = cdcl(triv, 1000);
                  t.expect(dpt.has_value() && dpt->verdict == SatVerdict::unsatisfiable &&
                               cdt.has_value() && cdt->verdict == SatVerdict::unsatisfiable,
                           "dpll and cdcl both prove (x)&(-x) UNSAT");

                  // Stochastic search on an UNSAT instance can only give up (unknown) — it MUST
                  // NOT claim unsatisfiable.
                  auto ws = walksat(uns, 20000, 0.5, 42);
                  t.expect(ws.has_value() && ws->verdict == SatVerdict::unknown,
                           "walksat on UNSAT returns unknown (never unsatisfiable)");
                  auto gs = gsat(uns, 500, 50, 42);
                  t.expect(gs.has_value() && gs->verdict == SatVerdict::unknown,
                           "gsat on UNSAT returns unknown (never unsatisfiable)");

                  // A malformed CNF (num_vars == 0) is a domain_error.
                  const Cnf bad{.num_vars = 0, .clauses = {{1}}};
                  auto db = dpll(bad);
                  t.expect(!db.has_value() && db.error() == MathError::domain_error,
                           "dpll on a malformed CNF is a domain_error");
              })
        .test("sat_portfolio_and_shards_agree_with_dpll",
              [](TestContext& t) {
                  const Cnf sat{.num_vars = 3, .clauses = {{1, 2}, {-1, 3}, {-2, -3}}};
                  const Cnf uns{.num_vars = 2,
                                .clauses = {{1, 2}, {1, -2}, {-1, 2}, {-1, -2}}};

                  for (const auto& [cnf, label] :
                       {std::pair{sat, "SAT"}, std::pair{uns, "UNSAT"}}) {
                      auto ref = dpll(cnf);
                      auto port = solve_portfolio(cnf, 2024, 4);
                      t.expect(ref.has_value() && port.has_value(),
                               std::format("dpll and portfolio both run ({})", label));
                      if (ref && port) {
                          t.expect(port->verdict == ref->verdict,
                                   std::format("portfolio verdict == dpll verdict ({})", label));
                          if (port->verdict == SatVerdict::satisfiable) {
                              t.expect(verify_assignment(cnf, port->model),
                                       std::format("portfolio's model verifies ({})", label));
                          }
                      }

                      // solve_shard over 4 shards, merged with the same associative all-reduce,
                      // reduces to the same verdict a single portfolio (and DPLL) reports.
                      constexpr std::size_t shards = 4;
                      std::vector<SatResult> collected;
                      bool all_ran = true;
                      for (std::size_t s = 0; s < shards; ++s) {
                          auto sr = solve_shard(cnf, s, shards, 2024);
                          if (!sr) {
                              all_ran = false;
                          } else {
                              collected.push_back(*sr);
                          }
                      }
                      t.expect(all_ran, std::format("all {} shards ran ({})", shards, label));
                      if (all_ran && ref) {
                          const SatResult merged = merge_shards(collected);
                          t.expect(merged.verdict == ref->verdict,
                                   std::format("merged shard verdict == dpll verdict ({})",
                                               label));
                      }
                  }
              })
        // ===================================================================
        // CSP: AC-3 pruning, honesty (arc-consistent != solved), and the
        // agreement of backtracking / forward-checking / parallel search.
        // ===================================================================
        .test("csp_ac3_prunes_and_reports_inconsistency",
              [](TestContext& t) {
                  const auto less = [](std::int64_t a, std::int64_t b) { return a < b; };

                  // x < y over {0,1,2}: AC-3 prunes to D[x]={0,1}, D[y]={1,2}.
                  Csp c;
                  c.domains = {{0, 1, 2}, {0, 1, 2}};
                  c.binary.push_back(BinaryConstraint{.x = 0, .y = 1, .allowed = less});
                  auto pr = ac3(c);
                  t.expect(pr.has_value() && pr->has_value(), "ac3 succeeds and keeps domains");
                  if (pr && pr->has_value()) {
                      const auto& d = **pr;
                      const std::vector<std::int64_t> dx{0, 1};
                      const std::vector<std::int64_t> dy{1, 2};
                      t.expect(d.size() == 2 && d[0] == dx && d[1] == dy,
                               "ac3 prunes x<y to D[x]={0,1}, D[y]={1,2}");
                  }

                  // x < y with D[x]={3}, D[y]={1,2}: no support, D[x] empties -> arc-inconsistent,
                  // which is a VALID engaged std::nullopt result, not an error.
                  Csp e;
                  e.domains = {{3}, {1, 2}};
                  e.binary.push_back(BinaryConstraint{.x = 0, .y = 1, .allowed = less});
                  auto ei = ac3(e);
                  t.expect(ei.has_value() && !ei->has_value(),
                           "ac3 reports arc-inconsistency as an engaged nullopt");

                  // Shape fault: a variable with an empty domain is a domain_error.
                  Csp bad;
                  bad.domains = {{0, 1}, {}};
                  auto br = ac3(bad);
                  t.expect(!br.has_value() && br.error() == MathError::domain_error,
                           "ac3 on an empty domain is a domain_error");
              })
        .test("csp_ac3_is_not_a_complete_solver",
              [](TestContext& t) {
                  // Three variables over {0,1}, pairwise all-different: arc-consistent (each value
                  // keeps a support) yet UNSATISFIABLE (pigeonhole). AC-3 must NOT prune it away,
                  // but the search must report no solution and the count must be 0.
                  const auto ne = [](std::int64_t a, std::int64_t b) { return a != b; };
                  Csp c;
                  c.domains = {{0, 1}, {0, 1}, {0, 1}};
                  c.binary = {BinaryConstraint{.x = 0, .y = 1, .allowed = ne},
                              BinaryConstraint{.x = 0, .y = 2, .allowed = ne},
                              BinaryConstraint{.x = 1, .y = 2, .allowed = ne}};

                  auto pr = ac3(c);
                  t.expect(pr.has_value() && pr->has_value(),
                           "ac3 leaves the all-different triple arc-consistent (non-empty)");
                  if (pr && pr->has_value()) {
                      bool all_full = true;
                      for (const auto& dom : **pr) {
                          if (dom.size() != 2) {
                              all_full = false;
                          }
                      }
                      t.expect(all_full, "ac3 prunes nothing here (every domain still {0,1})");
                  }

                  auto bt = backtracking_search(c);
                  t.expect(bt.has_value() && !bt->has_value(),
                           "backtracking reports the arc-consistent triple UNSAT");
                  auto cnt = solution_count(c, 0);
                  t.expect(cnt.has_value() && *cnt == 0,
                           "solution_count of the unsatisfiable triple is 0");
              })
        .test("csp_backtracking_fc_parallel_agree_on_queens",
              [](TestContext& t) {
                  const Csp q6 = queens_csp(6);
                  auto bt = backtracking_search(q6);
                  auto fc = backtracking_search_fc(q6);
                  auto pp = parallel_search(q6);
                  t.expect(bt.has_value() && fc.has_value() && pp.has_value(),
                           "all three CSP searches run on 6-queens");
                  if (bt && fc && pp) {
                      t.expect(bt->has_value(), "6-queens has a solution");
                      if (bt->has_value()) {
                          t.expect(valid_queens(**bt),
                                   "backtracking's 6-queens assignment is a valid placement");
                      }
                      // All three return the SAME lexicographically-first solution.
                      t.expect(*bt == *fc,
                               "forward-checking returns the same solution as plain backtracking");
                      t.expect(*bt == *pp,
                               "parallel_search returns the same solution as backtracking");
                  }
              })
        .test("csp_queens_counts_match_known_values",
              [](TestContext& t) {
                  // Known n-queens solution counts.
                  const std::array<std::pair<int, std::uint64_t>, 5> known{
                      {{4, 2}, {5, 10}, {6, 4}, {7, 40}, {8, 92}}};
                  for (const auto& [n, count] : known) {
                      auto c = solution_count(queens_csp(n), 0);
                      t.expect(c.has_value() && *c == count,
                               std::format("csp solution_count({}-queens) == {} (got {})", n,
                                           count, c ? *c : 0));
                  }
              })
        // ===================================================================
        // LOGIC: unification (occurs-check) and SLD resolution.
        // ===================================================================
        .test("logic_unification_honours_occurs_check",
              [](TestContext& t) {
                  const Term X = make_var("X");
                  const Term Y = make_var("Y");
                  const Term a = make_atom("a");
                  const Term b = make_atom("b");

                  // Occurs-check: X does not unify with f(X). This is a normal non-unification
                  // (an engaged Result holding std::nullopt), NOT a MathError.
                  const Term fX = make_compound("f", {X});
                  auto occ = unify(X, fX, Substitution{});
                  t.expect(occ.has_value() && !occ->has_value(),
                           "unify(X, f(X)) fails the occurs-check (nullopt, not an error)");

                  // Structural unification: f(X, a) with f(b, Y) binds X->b, Y->a.
                  const Term t1 = make_compound("f", {X, a});
                  const Term t2 = make_compound("f", {b, Y});
                  auto u = unify(t1, t2, Substitution{});
                  t.expect(u.has_value() && u->has_value(), "f(X,a) unifies with f(b,Y)");
                  if (u && u->has_value()) {
                      const Substitution& s = u->value();
                      t.expect(apply_substitution(s, X) == b, "unifier binds X -> b");
                      t.expect(apply_substitution(s, Y) == a, "unifier binds Y -> a");
                  }

                  // Distinct atoms never unify.
                  auto na = unify(a, b, Substitution{});
                  t.expect(na.has_value() && !na->has_value(), "distinct atoms do not unify");
              })
        .test("logic_sld_answers_ancestor_and_append",
              [](TestContext& t) {
                  const Term tom = make_atom("tom");
                  const Term bob = make_atom("bob");
                  const Term ann = make_atom("ann");
                  const Term pat = make_atom("pat");
                  const Term X = make_var("X");
                  const Term Y = make_var("Y");
                  const Term Z = make_var("Z");

                  const auto parent = [](const Term& p, const Term& c) {
                      return make_compound("parent", {p, c});
                  };
                  const auto ancestor = [](const Term& p, const Term& c) {
                      return make_compound("ancestor", {p, c});
                  };

                  Program prog{
                      Clause{.head = parent(tom, bob), .body = {}},
                      Clause{.head = parent(bob, ann), .body = {}},
                      Clause{.head = parent(bob, pat), .body = {}},
                      Clause{.head = ancestor(X, Y), .body = {parent(X, Y)}},
                      Clause{.head = ancestor(X, Y), .body = {parent(X, Z), ancestor(Z, Y)}},
                  };

                  // ancestor(tom, Who) has exactly three answers: bob, ann, pat.
                  const std::vector<Term> query{ancestor(tom, make_var("Who"))};
                  auto sols = solve(prog, query, 0);
                  t.expect(sols.has_value(), "ancestor query solves");
                  if (sols) {
                      t.expect(sols->size() == 3,
                               std::format("ancestor(tom, Who) has 3 answers (got {})",
                                           sols->size()));
                      std::set<std::string> who;
                      for (const auto& s : *sols) {
                          who.insert(binding_of(s, "Who"));
                      }
                      const std::set<std::string> want{"ann", "bob", "pat"};
                      t.expect(who == want, "ancestor answers are exactly {ann, bob, pat}");
                  }

                  // solve_first returns the first answer, bob.
                  auto first = solve_first(prog, query);
                  t.expect(first.has_value() && first->has_value(), "solve_first finds an answer");
                  if (first && first->has_value()) {
                      t.expect(binding_of(**first, "Who") == "bob",
                               "the first ancestor of tom is bob");
                  }

                  // OR-parallel == serial within budget (byte-for-byte equal answer set).
                  auto par = solve_or_parallel(prog, query, 0);
                  t.expect(par.has_value() && sols.has_value() && *par == *sols,
                           "solve_or_parallel == serial solve within budget");

                  // append([1,2],[3],R) has the single answer R = [1,2,3].
                  const auto cons = [](const Term& h, const Term& tl) {
                      return make_compound(".", {h, tl});
                  };
                  const Term H = make_var("H");
                  const Term T = make_var("T");
                  const Term L = make_var("L");
                  const Term R = make_var("R");
                  const auto append3 = [](const Term& x, const Term& y, const Term& z) {
                      return make_compound("append", {x, y, z});
                  };
                  Program app{
                      Clause{.head = append3(make_nil(), L, L), .body = {}},
                      Clause{.head = append3(cons(H, T), L, cons(H, R)),
                             .body = {append3(T, L, R)}},
                  };
                  const std::vector<Term> aq{append3(make_list({make_int(1), make_int(2)}),
                                                     make_list({make_int(3)}), make_var("Out"))};
                  auto ar = solve(app, aq, 0);
                  t.expect(ar.has_value() && ar->size() == 1,
                           "append([1,2],[3],Out) has exactly one answer");
                  if (ar && ar->size() == 1) {
                      t.expect(binding_of(ar->front(), "Out") == "[1, 2, 3]",
                               "append yields Out = [1, 2, 3]");
                  }

                  // A non-callable goal (a bare integer) is a domain_error.
                  auto bad = solve(prog, {make_int(5)}, 0);
                  t.expect(!bad.has_value() && bad.error() == MathError::domain_error,
                           "a non-callable goal is a domain_error");
              })
        // ===================================================================
        // BITSET / BITCSP: word-parallel set algebra and a cross-checked
        // branchless N-queens counter.
        // ===================================================================
        .test("bitset_word_parallel_ops_match_reference",
              [](TestContext& t) {
                  constexpr std::size_t cap = 128;  // spans two 64-bit words plus a masked tail
                  Bitset a(cap);
                  Bitset b(cap);
                  for (const std::size_t i : {std::size_t{1}, std::size_t{5}, std::size_t{63},
                                              std::size_t{64}, std::size_t{100}}) {
                      a.set(i);
                  }
                  for (const std::size_t i : {std::size_t{5}, std::size_t{64}, std::size_t{101},
                                              std::size_t{127}}) {
                      b.set(i);
                  }

                  t.expect(a.count() == 5 && b.count() == 4,
                           "popcount matches the number of set bits");
                  t.expect(a.first_set() == 1, "first_set(a) == 1");

                  auto un = a.or_with(b);
                  auto in = a.and_with(b);
                  auto xr = a.xor_with(b);
                  auto dn = a.andnot_with(b);
                  t.expect(un.has_value() && in.has_value() && xr.has_value() && dn.has_value(),
                           "the four word-parallel combinators succeed at equal capacity");
                  if (un && in && xr && dn) {
                      const std::vector<std::size_t> want_un{1, 5, 63, 64, 100, 101, 127};
                      const std::vector<std::size_t> want_in{5, 64};
                      const std::vector<std::size_t> want_xr{1, 63, 100, 101, 127};
                      const std::vector<std::size_t> want_dn{1, 63, 100};
                      t.expect(un->set_bits() == want_un, "union matches the reference");
                      t.expect(in->set_bits() == want_in, "intersection matches the reference");
                      t.expect(xr->set_bits() == want_xr,
                               "symmetric difference matches the reference");
                      t.expect(dn->set_bits() == want_dn, "set difference matches the reference");
                      t.expect(in->count() == 2 && un->count() == 7,
                               "combinator popcounts agree with the reference sizes");
                  }

                  // Complement within the capacity via full \ a: count is cap - count(a) and bit 0
                  // (absent from a) is the lowest complement bit.
                  Bitset full(cap);
                  full.set_all();
                  auto comp = full.andnot_with(a);
                  t.expect(comp.has_value(), "complement (full andnot a) succeeds");
                  if (comp) {
                      t.expect(comp->count() == cap - a.count(),
                               "complement popcount == capacity - count(a)");
                      t.expect(comp->first_set() == 0, "lowest complement bit is 0");
                  }

                  // A capacity mismatch is a domain_error on the Result-returning combinators.
                  Bitset small(64);
                  auto mism = a.and_with(small);
                  t.expect(!mism.has_value() && mism.error() == MathError::domain_error,
                           "combining unequal capacities is a domain_error");
              })
        .test("bitcsp_ac3_cross_checks_csp_ac3",
              [](TestContext& t) {
                  // The same x < y instance as the csp AC-3 test, over the value range {0,1,2}.
                  // The branchless bitset propagator must prune to the identical domains.
                  const auto less_i = [](int aa, int bb) { return aa < bb; };
                  BitCsp bc;
                  Bitset dx(3);
                  Bitset dy(3);
                  dx.set_all();
                  dy.set_all();
                  bc.domains = {dx, dy};
                  bc.constraints.push_back(make_bit_constraint(0, 1, 3, 3, less_i));
                  auto rb = ac3_bitset(bc);
                  t.expect(rb.has_value() && rb->has_value(), "ac3_bitset succeeds and prunes");
                  if (rb && rb->has_value()) {
                      const auto& d = **rb;
                      const std::vector<std::size_t> dxs{0, 1};
                      const std::vector<std::size_t> dys{1, 2};
                      t.expect(d.size() == 2 && d[0].set_bits() == dxs && d[1].set_bits() == dys,
                               "ac3_bitset prunes x<y to D[x]={0,1}, D[y]={1,2} (matches csp)");
                  }
              })
        .test("bitcsp_nqueens_counts_match_csp_and_parallel",
              [](TestContext& t) {
                  const std::array<std::pair<int, std::uint64_t>, 5> known{
                      {{4, 2}, {5, 10}, {6, 4}, {7, 40}, {8, 92}}};
                  for (const auto& [n, count] : known) {
                      auto serial = count_nqueens(n, false);
                      auto par = count_nqueens(n, true);
                      auto csp = solution_count(queens_csp(n), 0);
                      t.expect(serial.has_value() && *serial == count,
                               std::format("bitcsp count_nqueens({}) == {} (got {})", n, count,
                                           serial ? *serial : 0));
                      // Parallel subtree fan-out equals the serial count for any worker count.
                      t.expect(par.has_value() && serial.has_value() && *par == *serial,
                               std::format("count_nqueens({}) parallel == serial", n));
                      // Two independent implementations (bitmask vs finite-domain CSP) agree.
                      t.expect(csp.has_value() && serial.has_value() && *csp == *serial,
                               std::format("bitcsp count == csp count for {}-queens", n));
                  }

                  // Out-of-range board sizes are domain_errors.
                  auto zero = count_nqueens(0, false);
                  auto big = count_nqueens(33, false);
                  t.expect(!zero.has_value() && zero.error() == MathError::domain_error,
                           "count_nqueens(0) is a domain_error");
                  t.expect(!big.has_value() && big.error() == MathError::domain_error,
                           "count_nqueens(33) is a domain_error");
              })
        .run();
}
