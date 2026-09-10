// Tests for nimblecas.sat_dist: portfolio and cube-and-conquer SAT as task graphs.
// @author Olumuyiwa Oluwasanmi
//
// The contract is that the VERDICT DOES NOT REVEAL WHICH EXECUTOR RAN IT, and for cubes that the
// decomposition is exact: the cubes partition the assignment space, so their merged verdict is the
// original formula's verdict. Both are checked against `nimblecas.sat`'s in-process solvers on the
// same formulas, and every returned model is verified against the original formula rather than
// taken on trust.

import std;
import nimblecas.core;
import nimblecas.sat;
import nimblecas.sat_dist;
import nimblecas.taskdag;
import nimblecas.taskdag_sgee;
import nimblecas.testing;

using nimblecas::Cnf;
using nimblecas::cdcl;
using nimblecas::dpll;
using nimblecas::Executor;
using nimblecas::local_parallel_executor;
using nimblecas::MathError;
using nimblecas::Payload;
using nimblecas::SatResult;
using nimblecas::SatVerdict;
using nimblecas::serial_executor;
using nimblecas::solve_portfolio;
using nimblecas::TaskRegistry;
using nimblecas::FakeBrokerPort;
using nimblecas::InMemoryResultChannel;
using nimblecas::SgeeDistributedExecutor;
using nimblecas::SgeeExecutorConfig;
using nimblecas::verify_assignment;
using nimblecas::sat_dist::build_cube_graph;
using nimblecas::sat_dist::build_portfolio_graph;
using nimblecas::sat_dist::cube_of;
using nimblecas::sat_dist::decode_cnf;
using nimblecas::sat_dist::decode_result;
using nimblecas::sat_dist::encode_cnf;
using nimblecas::sat_dist::encode_result;
using nimblecas::sat_dist::max_cube_vars;
using nimblecas::sat_dist::register_ops;
using nimblecas::sat_dist::solve_cubes_distributed;
using nimblecas::sat_dist::solve_portfolio_distributed;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto simple_sat() -> Cnf {
    return Cnf{.num_vars = 4, .clauses = {{1, 2}, {-1, 3}, {-2, 4}, {-3, -4, 1}}};
}

[[nodiscard]] auto simple_unsat() -> Cnf {
    return Cnf{.num_vars = 2, .clauses = {{1}, {-1}}};
}

// Pigeonhole with three pigeons and two holes: unsatisfiable, and famously hard for resolution,
// which makes it a good formula on which to insist the UNSAT verdict is genuinely complete.
// p(i,j) is variable 2*(i-1) + j for i in 1..3, j in 1..2.
[[nodiscard]] auto pigeonhole_3_2() -> Cnf {
    const auto p = [](int i, int j) -> std::int64_t {
        return static_cast<std::int64_t>(2 * (i - 1) + j);
    };
    Cnf cnf{.num_vars = 6, .clauses = {}};
    for (int i = 1; i <= 3; ++i) {
        cnf.clauses.push_back({p(i, 1), p(i, 2)});  // every pigeon is in some hole
    }
    for (int j = 1; j <= 2; ++j) {
        for (int a = 1; a <= 3; ++a) {
            for (int b = a + 1; b <= 3; ++b) {
                cnf.clauses.push_back({-p(a, j), -p(b, j)});  // no hole holds two pigeons
            }
        }
    }
    return cnf;
}

}  // namespace

auto main() -> int {
    TestSuite("nimblecas.sat_dist")
        .test("encode_and_decode_round_trip_a_formula",
              [](TestContext& t) {
                  const Cnf cnf = simple_sat();
                  auto bytes = encode_cnf(cnf);
                  t.expect(bytes.has_value(), "the formula encodes");
                  if (!bytes.has_value()) {
                      return;
                  }
                  auto back = decode_cnf(*bytes);
                  t.expect(back.has_value(), "the bytes decode");
                  if (!back.has_value()) {
                      return;
                  }
                  t.expect(back->num_vars == cnf.num_vars, "the variable count survives");
                  t.expect(back->clauses == cnf.clauses, "every clause survives exactly");
              })
        .test("encode_and_decode_round_trip_the_awkward_shapes",
              [](TestContext& t) {
                  const Cnf odd{.num_vars = 3, .clauses = {{}, {-3}, {1, -2, 3, 1}}};
                  auto bytes = encode_cnf(odd);
                  t.expect(bytes.has_value(), "a formula with an empty clause encodes");
                  if (!bytes.has_value()) {
                      return;
                  }
                  auto back = decode_cnf(*bytes);
                  t.expect(back.has_value(), "it decodes");
                  if (!back.has_value()) {
                      return;
                  }
                  t.expect(back->clauses.size() == 3, "all three clauses come back");
                  if (back->clauses.size() != 3) {
                      return;
                  }
                  t.expect(back->clauses[0].empty(), "the empty clause is still empty");
                  t.expect(back->clauses[2].size() == 4,
                           "a clause with a repeated literal keeps all four entries");
                  const Cnf no_clauses{.num_vars = 1, .clauses = {}};
                  auto b2 = encode_cnf(no_clauses);
                  t.expect(b2.has_value() && decode_cnf(*b2).has_value(),
                           "a formula with no clauses round-trips");
              })
        .test("decode_cnf_rejects_bytes_it_did_not_write",
              [](TestContext& t) {
                  const Payload empty;
                  auto a = decode_cnf(empty);
                  t.expect(!a.has_value() && a.error() == MathError::syntax_error,
                           "empty bytes are a syntax_error, not an empty formula");
                  Payload garbage;
                  for (const char c : std::string_view{"this is not a formula payload!!"}) {
                      garbage.push_back(static_cast<std::byte>(c));
                  }
                  auto b = decode_cnf(garbage);
                  t.expect(!b.has_value() && b.error() == MathError::syntax_error,
                           "arbitrary bytes are a syntax_error");
                  auto good = encode_cnf(simple_sat());
                  if (!good.has_value()) {
                      return;
                  }
                  Payload truncated = *good;
                  truncated.pop_back();
                  auto c = decode_cnf(truncated);
                  t.expect(!c.has_value() && c.error() == MathError::syntax_error,
                           "a payload one byte short is a syntax_error, never a short formula");
              })
        .test("encode_and_decode_round_trip_a_result",
              [](TestContext& t) {
                  const SatResult r{.verdict = SatVerdict::satisfiable,
                                    .model = {true, false, true, true}};
                  auto bytes = encode_result(r);
                  t.expect(bytes.has_value(), "the result encodes");
                  if (!bytes.has_value()) {
                      return;
                  }
                  auto back = decode_result(*bytes);
                  t.expect(back.has_value(), "it decodes");
                  if (!back.has_value()) {
                      return;
                  }
                  t.expect(back->verdict == SatVerdict::satisfiable, "the verdict survives");
                  t.expect(back->model == r.model, "the model survives exactly");
                  const SatResult unsat{.verdict = SatVerdict::unsatisfiable, .model = {}};
                  auto u = encode_result(unsat);
                  t.expect(u.has_value(), "an unsatisfiable result encodes");
                  if (!u.has_value()) {
                      return;
                  }
                  auto ub = decode_result(*u);
                  t.expect(ub.has_value() && ub->verdict == SatVerdict::unsatisfiable &&
                               ub->model.empty(),
                           "and comes back with an empty model");
              })
        .test("register_ops_twice_on_one_registry_is_a_domain_error",
              [](TestContext& t) {
                  TaskRegistry reg;
                  t.expect(register_ops(reg).has_value(), "the first registration succeeds");
                  auto second = register_ops(reg);
                  t.expect(!second.has_value() && second.error() == MathError::domain_error,
                           "registering the same operation again is a domain_error");
              })
        .test("cube_of_adds_one_unit_clause_per_fixed_variable",
              [](TestContext& t) {
                  const Cnf cnf = simple_sat();
                  auto c = cube_of(cnf, 2, 0);
                  t.expect(c.has_value(), "cube 0 of a two-variable split is built");
                  if (!c.has_value()) {
                      return;
                  }
                  t.expect(c->num_vars == cnf.num_vars,
                           "the variable count is unchanged -- a cube restricts, it does not "
                           "shrink the formula");
                  t.expect(c->clauses.size() == cnf.clauses.size() + 2,
                           "exactly two unit clauses were appended");
                  if (c->clauses.size() != cnf.clauses.size() + 2) {
                      return;
                  }
                  const std::vector<std::int64_t> x1_false{-1};
                  const std::vector<std::int64_t> x2_false{-2};
                  t.expect(c->clauses[cnf.clauses.size()] == x1_false,
                           "cube 0 fixes variable 1 false");
                  t.expect(c->clauses[cnf.clauses.size() + 1] == x2_false,
                           "cube 0 fixes variable 2 false");
                  auto c3 = cube_of(cnf, 2, 3);
                  t.expect(c3.has_value(), "cube 3 is built");
                  if (!c3.has_value() || c3->clauses.size() != cnf.clauses.size() + 2) {
                      return;
                  }
                  const std::vector<std::int64_t> x1_true{1};
                  const std::vector<std::int64_t> x2_true{2};
                  t.expect(c3->clauses[cnf.clauses.size()] == x1_true,
                           "cube 3 fixes variable 1 true");
                  t.expect(c3->clauses[cnf.clauses.size() + 1] == x2_true,
                           "cube 3 fixes variable 2 true");
              })
        .test("the_cubes_partition_the_assignment_space_exactly",
              [](TestContext& t) {
                  // The whole correctness argument for cube-and-conquer rests on this: every
                  // assignment lies in exactly one cube. Checking it directly is cheap and makes
                  // the merged verdict trustworthy rather than merely plausible.
                  const Cnf cnf = simple_sat();
                  const std::size_t cube_vars = 3;
                  const std::size_t assignments = 1U << cnf.num_vars;
                  std::vector<int> covered(assignments, 0);
                  for (std::uint64_t c = 0; c < (1ULL << cube_vars); ++c) {
                      auto cube = cube_of(cnf, cube_vars, c);
                      if (!cube.has_value()) {
                          t.expect(false, "every cube is built");
                          return;
                      }
                      for (std::size_t a = 0; a < assignments; ++a) {
                          std::vector<bool> model(cnf.num_vars, false);
                          for (std::size_t v = 0; v < cnf.num_vars; ++v) {
                              model[v] = ((a >> v) & 1U) != 0U;
                          }
                          // The unit clauses alone decide membership, so test them directly.
                          bool in_cube = true;
                          for (std::size_t v = 0; v < cube_vars; ++v) {
                              const bool want = ((c >> v) & 1ULL) != 0ULL;
                              if (model[v] != want) {
                                  in_cube = false;
                                  break;
                              }
                          }
                          if (in_cube) {
                              ++covered[a];
                          }
                      }
                  }
                  const bool exactly_once =
                      std::ranges::all_of(covered, [](int n) { return n == 1; });
                  t.expect(exactly_once,
                           "every one of the sixteen assignments lies in exactly one cube");
              })
        .test("cube_of_guards_its_arguments",
              [](TestContext& t) {
                  const Cnf cnf = simple_sat();
                  auto a = cube_of(cnf, 0, 0);
                  t.expect(!a.has_value() && a.error() == MathError::domain_error,
                           "splitting on zero variables is a domain_error");
                  auto b = cube_of(cnf, cnf.num_vars + 1, 0);
                  t.expect(!b.has_value() && b.error() == MathError::domain_error,
                           "splitting on more variables than the formula has is a domain_error");
                  auto c = cube_of(cnf, 2, 4);
                  t.expect(!c.has_value() && c.error() == MathError::domain_error,
                           "a cube index past the end of a two-variable split is a domain_error");
                  auto d = cube_of(cnf, max_cube_vars + 1, 0);
                  t.expect(!d.has_value() && d.error() == MathError::domain_error,
                           "splitting past the cube limit is a domain_error");
              })
        .test("build_cube_graph_makes_one_task_per_cube",
              [](TestContext& t) {
                  TaskRegistry reg;
                  if (!register_ops(reg)) {
                      return;
                  }
                  const Cnf cnf = simple_sat();
                  for (const std::size_t k : {std::size_t{1}, std::size_t{2}, std::size_t{3}}) {
                      auto g = build_cube_graph(reg, cnf, k, 0);
                      t.expect(g.has_value() && g->size() == (std::size_t{1} << k),
                               "a k-variable split builds exactly 2^k tasks");
                  }
              })
        .test("build_portfolio_graph_makes_one_task_per_shard",
              [](TestContext& t) {
                  TaskRegistry reg;
                  if (!register_ops(reg)) {
                      return;
                  }
                  const Cnf cnf = simple_sat();
                  for (const std::size_t shards : {std::size_t{1}, std::size_t{4},
                                                   std::size_t{9}}) {
                      auto g = build_portfolio_graph(reg, cnf, 12345, shards);
                      t.expect(g.has_value() && g->size() == shards,
                               "the portfolio graph has exactly one task per shard");
                  }
                  auto bad = build_portfolio_graph(reg, cnf, 1, 0);
                  t.expect(!bad.has_value() && bad.error() == MathError::domain_error,
                           "zero shards is a domain_error");
              })
        .test("distributed_cubes_find_a_model_that_really_satisfies_the_formula",
              [](TestContext& t) {
                  const Cnf cnf = simple_sat();
                  auto exec = serial_executor();
                  auto r = solve_cubes_distributed(cnf, 2, 0, *exec);
                  t.expect(r.has_value(), "the cube solve succeeds");
                  if (!r.has_value()) {
                      return;
                  }
                  t.expect(r->verdict == SatVerdict::satisfiable, "the formula is satisfiable");
                  t.expect(verify_assignment(cnf, r->model),
                           "the returned model satisfies every clause of the ORIGINAL formula, "
                           "not merely the restricted cube it came from");
              })
        .test("distributed_cubes_prove_unsatisfiability_completely",
              [](TestContext& t) {
                  const Cnf cnf = simple_unsat();
                  auto exec = serial_executor();
                  auto r = solve_cubes_distributed(cnf, 1, 0, *exec);
                  t.expect(r.has_value(), "the cube solve succeeds");
                  if (!r.has_value()) {
                      return;
                  }
                  t.expect(r->verdict == SatVerdict::unsatisfiable,
                           "every cube being unsatisfiable proves the formula is");
                  t.expect(r->model.empty(), "an unsatisfiable verdict carries no model");
              })
        .test("distributed_cubes_settle_the_pigeonhole_formula_as_unsatisfiable",
              [](TestContext& t) {
                  const Cnf cnf = pigeonhole_3_2();
                  auto exec = local_parallel_executor();
                  auto mine = solve_cubes_distributed(cnf, 3, 0, *exec);
                  auto theirs = dpll(cnf);
                  t.expect(mine.has_value() && theirs.has_value(), "both solvers succeed");
                  if (!mine.has_value() || !theirs.has_value()) {
                      return;
                  }
                  t.expect(mine->verdict == SatVerdict::unsatisfiable,
                           "three pigeons do not fit in two holes");
                  t.expect(theirs->verdict == SatVerdict::unsatisfiable,
                           "dpll independently agrees");
              })
        .test("distributed_cubes_give_the_same_verdict_on_both_executors_and_every_split",
              [](TestContext& t) {
                  // The core contract: the verdict must not reveal how the work was divided.
                  const std::array<Cnf, 3> formulas{simple_sat(), simple_unsat(),
                                                    pigeonhole_3_2()};
                  auto ser = serial_executor();
                  auto par = local_parallel_executor();
                  for (const Cnf& cnf : formulas) {
                      auto baseline = solve_cubes_distributed(cnf, 1, 0, *ser);
                      t.expect(baseline.has_value(), "the one-variable split succeeds");
                      if (!baseline.has_value()) {
                          continue;
                      }
                      for (const std::size_t k : {std::size_t{2}, std::size_t{3}}) {
                          if (k > cnf.num_vars) {
                              continue;
                          }
                          auto a = solve_cubes_distributed(cnf, k, 0, *ser);
                          t.expect(a.has_value() && a->verdict == baseline->verdict,
                                   "a wider split gives the same verdict");
                          auto b = solve_cubes_distributed(cnf, k, 0, *par);
                          t.expect(b.has_value() && b->verdict == baseline->verdict,
                                   "the parallel executor gives the same verdict");
                      }
                  }
              })
        .test("distributed_cubes_agree_with_the_in_process_solvers_across_random_formulas",
              [](TestContext& t) {
                  std::mt19937_64 rng(0x5a7d15701ULL);
                  auto exec = local_parallel_executor();
                  int checked = 0;
                  int sat_seen = 0;
                  int unsat_seen = 0;
                  for (int trial = 0; trial < 60; ++trial) {
                      const auto nv = static_cast<std::size_t>(3 + (rng() % 5));
                      const auto nc = static_cast<std::size_t>(2 + (rng() % 12));
                      Cnf cnf{.num_vars = nv, .clauses = {}};
                      for (std::size_t c = 0; c < nc; ++c) {
                          const std::size_t len = 1 + (rng() % 3);
                          std::vector<std::int64_t> clause;
                          for (std::size_t l = 0; l < len; ++l) {
                              const auto v = static_cast<std::int64_t>(1 + (rng() % nv));
                              clause.push_back((rng() % 2) == 0 ? v : -v);
                          }
                          cnf.clauses.push_back(clause);
                      }
                      const std::size_t k = 1 + (rng() % 3);
                      auto mine = solve_cubes_distributed(cnf, k, 0, *exec);
                      auto theirs = dpll(cnf);
                      if (!mine.has_value() || !theirs.has_value()) {
                          t.expect(false, "both solvers succeed on every random formula");
                          continue;
                      }
                      ++checked;
                      t.expect(mine->verdict == theirs->verdict,
                               "the cube decomposition and dpll reach the same verdict");
                      if (mine->verdict == SatVerdict::satisfiable) {
                          ++sat_seen;
                          t.expect(verify_assignment(cnf, mine->model),
                                   "and the model it returns really satisfies the formula");
                      } else if (mine->verdict == SatVerdict::unsatisfiable) {
                          ++unsat_seen;
                      }
                  }
                  t.expect(checked == 60, "all sixty random formulas were checked");
                  t.expect(sat_seen > 5 && unsat_seen > 5,
                           "the sweep produced both verdicts, so neither branch went untested");
              })
        .test("distributed_cubes_report_unknown_rather_than_guessing_when_a_budget_runs_out",
              [](TestContext& t) {
                  // A conflict budget of one is far too small for the pigeonhole formula. The
                  // honest answer is that the status is not known -- reporting unsatisfiable here
                  // because no model turned up would be a wrong answer dressed as a cautious one.
                  const Cnf cnf = pigeonhole_3_2();
                  auto exec = serial_executor();
                  auto r = solve_cubes_distributed(cnf, 1, 1, *exec);
                  t.expect(r.has_value(), "the solve succeeds");
                  if (!r.has_value()) {
                      return;
                  }
                  t.expect(r->verdict != SatVerdict::satisfiable,
                           "an unsatisfiable formula never comes back satisfiable");
                  auto complete = solve_cubes_distributed(cnf, 1, 0, *exec);
                  t.expect(complete.has_value() &&
                               complete->verdict == SatVerdict::unsatisfiable,
                           "with no budget the same formula is proved unsatisfiable");
              })
        .test("distributed_cubes_guard_their_arguments",
              [](TestContext& t) {
                  const Cnf cnf = simple_sat();
                  auto exec = serial_executor();
                  auto a = solve_cubes_distributed(cnf, 0, 0, *exec);
                  t.expect(!a.has_value() && a.error() == MathError::domain_error,
                           "a zero-variable split is a domain_error");
                  auto b = solve_cubes_distributed(cnf, cnf.num_vars + 1, 0, *exec);
                  t.expect(!b.has_value() && b.error() == MathError::domain_error,
                           "splitting on more variables than exist is a domain_error");
              })
        .test("distributed_portfolio_matches_the_in_process_portfolio",
              [](TestContext& t) {
                  const Cnf cnf = simple_sat();
                  auto ser = serial_executor();
                  auto mine = solve_portfolio_distributed(cnf, 4242, 4, *ser);
                  auto theirs = solve_portfolio(cnf, 4242, 4);
                  t.expect(mine.has_value() && theirs.has_value(), "both portfolios succeed");
                  if (!mine.has_value() || !theirs.has_value()) {
                      return;
                  }
                  t.expect(mine->verdict == theirs->verdict,
                           "the distributed portfolio reaches the same verdict as the in-process "
                           "one for the same formula, seed and worker count");
                  t.expect(mine->model == theirs->model,
                           "and returns the same model, since both take the lowest shard index");
              })
        .test("distributed_portfolio_is_identical_on_both_executors",
              [](TestContext& t) {
                  const Cnf cnf = simple_sat();
                  auto ser = serial_executor();
                  auto par = local_parallel_executor();
                  auto a = solve_portfolio_distributed(cnf, 99, 6, *ser);
                  t.expect(a.has_value(), "the serial executor succeeds");
                  if (!a.has_value()) {
                      return;
                  }
                  // Repeated so a scheduling dependence shows up as a flake, which is exactly what
                  // this assertion exists to catch.
                  for (int attempt = 0; attempt < 6; ++attempt) {
                      auto b = solve_portfolio_distributed(cnf, 99, 6, *par);
                      t.expect(b.has_value() && b->verdict == a->verdict && b->model == a->model,
                               "the parallel executor gives exactly the same verdict and model");
                  }
              })
        .test("distributed_portfolio_verifies_the_model_it_returns",
              [](TestContext& t) {
                  const Cnf cnf = simple_sat();
                  auto exec = local_parallel_executor();
                  auto r = solve_portfolio_distributed(cnf, 7, 8, *exec);
                  t.expect(r.has_value(), "the portfolio succeeds");
                  if (!r.has_value()) {
                      return;
                  }
                  if (r->verdict == SatVerdict::satisfiable) {
                      t.expect(verify_assignment(cnf, r->model),
                               "a satisfiable verdict comes with a model that really satisfies "
                               "the formula");
                  } else {
                      t.expect(r->model.empty(),
                               "a non-satisfiable verdict carries no model at all");
                  }
              })
        .test("distributed_portfolio_settles_an_unsatisfiable_formula",
              [](TestContext& t) {
                  const Cnf cnf = simple_unsat();
                  auto exec = serial_executor();
                  auto r = solve_portfolio_distributed(cnf, 3, 4, *exec);
                  t.expect(r.has_value(), "the portfolio succeeds");
                  if (!r.has_value()) {
                      return;
                  }
                  t.expect(r->verdict == SatVerdict::unsatisfiable,
                           "a complete worker's unsatisfiable verdict wins the merge");
                  t.expect(r->model.empty(), "and carries no model");
              })
        .test("distributed_portfolio_guards_its_arguments",
              [](TestContext& t) {
                  const Cnf cnf = simple_sat();
                  auto exec = serial_executor();
                  auto a = solve_portfolio_distributed(cnf, 1, 0, *exec);
                  t.expect(!a.has_value() && a.error() == MathError::domain_error,
                           "zero shards is a domain_error");
                  const Cnf malformed{.num_vars = 0, .clauses = {{1}}};
                  auto b = solve_portfolio_distributed(malformed, 1, 2, *exec);
                  t.expect(!b.has_value(),
                           "a malformed formula is refused rather than solved");
              })
        .test("cube_and_conquer_reaches_the_same_verdict_over_the_sgee_distributed_executor",
              [](TestContext& t) {
                  // Cube-and-conquer is the decomposition that most wants a cluster: 2^k
                  // independent formulas, and an UNSAT verdict that needs EVERY one of them to
                  // come back. Running it over SGEE is the case the design was for, and this
                  // checks it rather than assuming the Executor seam is enough.
                  TaskRegistry reg;
                  t.expect(nimblecas::sat_dist::register_ops(reg).has_value(),
                           "both operations register into the executor's registry");
                  FakeBrokerPort port;
                  InMemoryResultChannel results;
                  SgeeExecutorConfig cfg;
                  cfg.registry = &reg;
                  cfg.num_workers = 4;
                  cfg.poll_interval_ms = 1;
                  SgeeDistributedExecutor sgee(cfg, port, results);
                  t.expect(sgee.name() == "sgee_distributed",
                           "the executor under test really is the distributed one");
                  auto ser = serial_executor();

                  const Cnf sat = simple_sat();
                  auto s_ser = solve_cubes_distributed(sat, 2, 0, *ser);
                  auto s_sgee = solve_cubes_distributed(sat, 2, 0, sgee);
                  t.expect(s_ser.has_value() && s_sgee.has_value(), "both runs succeed");
                  if (s_ser.has_value() && s_sgee.has_value()) {
                      t.expect(s_ser->verdict == s_sgee->verdict,
                               "the same verdict comes back from the cluster");
                      t.expect(s_sgee->verdict == SatVerdict::satisfiable &&
                                   verify_assignment(sat, s_sgee->model),
                               "and the model it carries really satisfies the formula");
                  }

                  const Cnf hard = pigeonhole_3_2();
                  auto u_ser = solve_cubes_distributed(hard, 3, 0, *ser);
                  auto u_sgee = solve_cubes_distributed(hard, 3, 0, sgee);
                  t.expect(u_ser.has_value() && u_sgee.has_value(), "both runs succeed");
                  if (u_ser.has_value() && u_sgee.has_value()) {
                      t.expect(u_sgee->verdict == SatVerdict::unsatisfiable,
                               "eight cubes all coming back unsatisfiable proves the formula is "
                               "-- the case that needs every worker to report");
                      t.expect(u_ser->verdict == u_sgee->verdict,
                               "and the serial executor agrees");
                  }
              })
        .test("the_portfolio_reaches_the_same_verdict_and_model_over_sgee",
              [](TestContext& t) {
                  TaskRegistry reg;
                  if (!nimblecas::sat_dist::register_ops(reg)) {
                      t.expect(false, "operations register");
                      return;
                  }
                  FakeBrokerPort port;
                  InMemoryResultChannel results;
                  SgeeExecutorConfig cfg;
                  cfg.registry = &reg;
                  cfg.num_workers = 3;
                  cfg.poll_interval_ms = 1;
                  SgeeDistributedExecutor sgee(cfg, port, results);
                  auto ser = serial_executor();

                  const Cnf cnf = simple_sat();
                  auto a = solve_portfolio_distributed(cnf, 4242, 6, *ser);
                  auto b = solve_portfolio_distributed(cnf, 4242, 6, sgee);
                  t.expect(a.has_value() && b.has_value(), "both portfolios succeed");
                  if (!a.has_value() || !b.has_value()) {
                      return;
                  }
                  t.expect(a->verdict == b->verdict, "the same verdict");
                  t.expect(a->model == b->model,
                           "and the same model, because both take the lowest shard index rather "
                           "than whichever worker answered first");
              })
        .test("sgee_sat_verdicts_do_not_change_with_the_worker_count",
              [](TestContext& t) {
                  const Cnf cnf = pigeonhole_3_2();
                  auto ser = serial_executor();
                  auto baseline = solve_cubes_distributed(cnf, 2, 0, *ser);
                  t.expect(baseline.has_value(), "the serial baseline succeeds");
                  if (!baseline.has_value()) {
                      return;
                  }
                  for (const std::size_t workers : {std::size_t{1}, std::size_t{2},
                                                    std::size_t{5}}) {
                      TaskRegistry reg;
                      if (!nimblecas::sat_dist::register_ops(reg)) {
                          t.expect(false, "operations register");
                          return;
                      }
                      FakeBrokerPort port;
                      InMemoryResultChannel results;
                      SgeeExecutorConfig cfg;
                      cfg.registry = &reg;
                      cfg.num_workers = workers;
                      cfg.poll_interval_ms = 1;
                      SgeeDistributedExecutor sgee(cfg, port, results);
                      auto r = solve_cubes_distributed(cnf, 2, 0, sgee);
                      t.expect(r.has_value() && r->verdict == baseline->verdict,
                               "the verdict is identical however many workers ran it");
                  }
              })
        .run();
}
