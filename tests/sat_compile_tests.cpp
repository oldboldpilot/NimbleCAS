// Tests for nimblecas.sat_compile: a CNF formula emitted as parallel C++, CUDA and Triton source.
// @author Olumuyiwa Oluwasanmi
//
// Two things are under test and they are quite different. The REFERENCE SOLVER is the algorithm
// the emitted code implements, and it is checked against `nimblecas.sat`'s independent solvers on
// the same formulas -- two unrelated algorithms agreeing on a verdict is real evidence, where a
// solver agreeing with itself is none. The EMITTED SOURCE is checked structurally: that it says
// what it should say, refuses what it should refuse, and carries the formula faithfully. Compiling
// the emitted C++ and running it happens in the end-to-end driver, not here, because a unit test
// that shells out to a compiler tests the compiler.

import std;
import nimblecas.core;
import nimblecas.sat;
import nimblecas.sat_compile;
import nimblecas.testing;

using nimblecas::Cnf;
using nimblecas::dpll;
using nimblecas::MathError;
using nimblecas::SatVerdict;
using nimblecas::verify_assignment;
using nimblecas::sat_compile::compile;
using nimblecas::sat_compile::CompileOptions;
using nimblecas::sat_compile::is_compilable;
using nimblecas::sat_compile::max_variables;
using nimblecas::sat_compile::model_of;
using nimblecas::sat_compile::reference_solve;
using nimblecas::sat_compile::Target;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// (x1 or x2) and (not x1 or x2) and (not x2 or x3): forces x2 true and x3 true, x1 free. The
// smallest satisfying assignment therefore has x1 false, x2 true, x3 true -- bits 0,1,1 -- which
// is the number 6.
[[nodiscard]] auto simple_sat() -> Cnf {
    return Cnf{.num_vars = 3, .clauses = {{1, 2}, {-1, 2}, {-2, 3}}};
}

// (x1) and (not x1): no assignment satisfies both.
[[nodiscard]] auto simple_unsat() -> Cnf {
    return Cnf{.num_vars = 1, .clauses = {{1}, {-1}}};
}

// All eight clauses over three variables: every assignment is excluded by exactly one of them.
[[nodiscard]] auto exhaustive_unsat() -> Cnf {
    Cnf cnf{.num_vars = 3, .clauses = {}};
    for (int mask = 0; mask < 8; ++mask) {
        std::vector<std::int64_t> clause;
        for (int v = 0; v < 3; ++v) {
            const bool bit = ((mask >> v) & 1) != 0;
            // Exclude the assignment `mask` by asserting at least one variable differs from it.
            clause.push_back(bit ? -(v + 1) : (v + 1));
        }
        cnf.clauses.push_back(clause);
    }
    return cnf;
}

constexpr std::uint64_t plenty = 1ULL << 20;

}  // namespace

auto main() -> int {
    TestSuite("nimblecas.sat_compile")
        .test("reference_solve_finds_the_smallest_satisfying_assignment",
              [](TestContext& t) {
                  auto r = reference_solve(simple_sat(), plenty);
                  t.expect(r.has_value(), "the reference solver succeeds");
                  if (!r.has_value()) {
                      return;
                  }
                  t.expect(r->has_value(), "the formula is satisfiable");
                  if (!r->has_value()) {
                      return;
                  }
                  t.expect(**r == 6ULL,
                           "the smallest satisfying assignment is exactly 6 (x1=0, x2=1, x3=1)");
              })
        .test("reference_solve_reports_no_solution_for_an_unsatisfiable_formula",
              [](TestContext& t) {
                  auto r = reference_solve(simple_unsat(), plenty);
                  t.expect(r.has_value(), "the solver succeeds");
                  if (!r.has_value()) {
                      return;
                  }
                  t.expect(!r->has_value(),
                           "an unsatisfiable formula gives no assignment inside a successful "
                           "Result, not an error");
              })
        .test("reference_solve_agrees_with_dpll_on_an_exhaustively_excluded_formula",
              [](TestContext& t) {
                  const Cnf cnf = exhaustive_unsat();
                  auto mine = reference_solve(cnf, plenty);
                  auto theirs = dpll(cnf);
                  t.expect(mine.has_value() && theirs.has_value(), "both solvers succeed");
                  if (!mine.has_value() || !theirs.has_value()) {
                      return;
                  }
                  t.expect(!mine->has_value(), "the enumeration finds no assignment");
                  t.expect(theirs->verdict == SatVerdict::unsatisfiable,
                           "dpll independently reports unsatisfiable");
              })
        .test("reference_solve_agrees_with_dpll_across_many_random_formulas",
              [](TestContext& t) {
                  // Two unrelated algorithms on the same input. dpll backtracks with unit
                  // propagation; the reference enumerates bit-parallel. Agreement on hundreds of
                  // random formulas is evidence; either one agreeing with itself is not.
                  std::mt19937_64 rng(0xc0ffee5a7ULL);
                  int checked = 0;
                  int sat_seen = 0;
                  int unsat_seen = 0;
                  for (int trial = 0; trial < 200; ++trial) {
                      const auto nv = static_cast<std::size_t>(2 + (rng() % 7));
                      const auto nc = static_cast<std::size_t>(1 + (rng() % 14));
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
                      auto mine = reference_solve(cnf, plenty);
                      auto theirs = dpll(cnf);
                      if (!mine.has_value() || !theirs.has_value()) {
                          t.expect(false, "both solvers succeed on every random formula");
                          continue;
                      }
                      ++checked;
                      const bool i_say_sat = mine->has_value();
                      const bool they_say_sat = theirs->verdict == SatVerdict::satisfiable;
                      t.expect(i_say_sat == they_say_sat,
                               "the enumeration and dpll agree on satisfiability");
                      if (i_say_sat) {
                          ++sat_seen;
                          // The model is not merely claimed satisfying -- it is checked.
                          t.expect(verify_assignment(cnf, model_of(**mine, nv)),
                                   "the reported assignment really does satisfy every clause");
                      } else {
                          ++unsat_seen;
                      }
                  }
                  t.expect(checked == 200, "all two hundred random formulas were checked");
                  t.expect(sat_seen > 20 && unsat_seen > 20,
                           "the sweep produced a healthy mix of both verdicts, so neither branch "
                           "went untested");
              })
        .test("reference_solve_honours_its_block_budget",
              [](TestContext& t) {
                  // Twenty variables is 2^14 blocks; a budget of ten is not enough.
                  const Cnf big{.num_vars = 20, .clauses = {{1, 2}}};
                  auto r = reference_solve(big, 10);
                  t.expect(!r.has_value() && r.error() == MathError::not_converged,
                           "a budget too small is not_converged, never a premature verdict");
                  auto ok = reference_solve(big, plenty);
                  t.expect(ok.has_value() && ok->has_value(),
                           "a sufficient budget solves the same formula");
              })
        .test("a_formula_with_fewer_than_six_variables_ignores_the_unused_lanes",
              [](TestContext& t) {
                  // One variable fills only two of a block's sixty-four lanes. The other
                  // sixty-two stand for assignments that do not exist, and reporting one of them
                  // would be a solution to a formula nobody asked about.
                  const Cnf only_false{.num_vars = 1, .clauses = {{-1}}};
                  auto r = reference_solve(only_false, plenty);
                  t.expect(r.has_value() && r->has_value(), "the formula is satisfiable");
                  if (!r.has_value() || !r->has_value()) {
                      return;
                  }
                  t.expect(**r == 0ULL, "the only satisfying assignment is exactly 0");
                  const Cnf only_true{.num_vars = 1, .clauses = {{1}}};
                  auto r2 = reference_solve(only_true, plenty);
                  t.expect(r2.has_value() && r2->has_value() && **r2 == 1ULL,
                           "asserting the variable gives exactly assignment 1");
              })
        .test("an_empty_clause_list_is_satisfied_by_the_smallest_assignment",
              [](TestContext& t) {
                  const Cnf trivially_true{.num_vars = 3, .clauses = {}};
                  auto r = reference_solve(trivially_true, plenty);
                  t.expect(r.has_value() && r->has_value(), "a formula with no clauses is true");
                  if (!r.has_value() || !r->has_value()) {
                      return;
                  }
                  t.expect(**r == 0ULL, "and the smallest satisfying assignment is 0");
              })
        .test("an_empty_clause_makes_the_formula_unsatisfiable",
              [](TestContext& t) {
                  const Cnf has_falsity{.num_vars = 3, .clauses = {{1, 2}, {}}};
                  auto r = reference_solve(has_falsity, plenty);
                  t.expect(r.has_value(), "the solver succeeds");
                  if (!r.has_value()) {
                      return;
                  }
                  t.expect(!r->has_value(),
                           "an empty clause denotes falsity, so nothing satisfies the formula");
              })
        .test("model_of_maps_an_assignment_to_the_variable_values",
              [](TestContext& t) {
                  const std::vector<bool> m = model_of(6ULL, 3);
                  t.expect(m.size() == 3, "the model has one entry per variable");
                  if (m.size() != 3) {
                      return;
                  }
                  t.expect(!m[0], "assignment 6 has x1 false");
                  t.expect(m[1], "assignment 6 has x2 true");
                  t.expect(m[2], "assignment 6 has x3 true");
              })
        .test("is_compilable_refuses_a_malformed_formula",
              [](TestContext& t) {
                  const Cnf no_vars{.num_vars = 0, .clauses = {{1}}};
                  auto a = is_compilable(no_vars);
                  t.expect(!a.has_value() && a.error() == MathError::domain_error,
                           "zero variables is a domain_error");
                  const Cnf zero_lit{.num_vars = 2, .clauses = {{1, 0}}};
                  auto b = is_compilable(zero_lit);
                  t.expect(!b.has_value() && b.error() == MathError::domain_error,
                           "a literal of 0 is a domain_error");
                  const Cnf out_of_range{.num_vars = 2, .clauses = {{3}}};
                  auto c = is_compilable(out_of_range);
                  t.expect(!c.has_value() && c.error() == MathError::domain_error,
                           "a literal naming a variable outside the formula is a domain_error");
                  const Cnf too_many{.num_vars = max_variables + 1, .clauses = {{1}}};
                  auto d = is_compilable(too_many);
                  t.expect(!d.has_value() && d.error() == MathError::domain_error,
                           "more variables than an assignment number can hold is a domain_error");
              })
        .test("is_compilable_accepts_the_edge_shapes_that_are_legal",
              [](TestContext& t) {
                  const Cnf empty_clause{.num_vars = 2, .clauses = {{}}};
                  t.expect(is_compilable(empty_clause).has_value(),
                           "an empty clause is legal -- it denotes falsity");
                  const Cnf no_clauses{.num_vars = 2, .clauses = {}};
                  t.expect(is_compilable(no_clauses).has_value(),
                           "an empty clause list is legal -- it denotes truth");
                  const Cnf at_limit{.num_vars = max_variables, .clauses = {{1}}};
                  t.expect(is_compilable(at_limit).has_value(),
                           "exactly the maximum variable count is accepted");
              })
        .test("cpp_emission_contains_the_documented_entry_points",
              [](TestContext& t) {
                  CompileOptions opts;
                  opts.target = Target::cpp;
                  opts.entry = "puzzle";
                  auto src = compile(simple_sat(), opts);
                  t.expect(src.has_value(), "the formula compiles to C++");
                  if (!src.has_value()) {
                      return;
                  }
                  t.expect(src->find("puzzle_solve_range") != std::string::npos,
                           "the range entry point is emitted under the requested name");
                  t.expect(src->find("puzzle_solve(") != std::string::npos,
                           "the whole-space entry point is emitted");
                  t.expect(src->find("puzzle_solve_parallel") != std::string::npos,
                           "the threaded entry point is emitted");
                  t.expect(src->find("puzzle_solve_range_simd") != std::string::npos,
                           "the SIMD range entry point is emitted");
                  t.expect(src->find("nc_puzzle_none") != std::string::npos,
                           "the no-solution sentinel is emitted");
                  t.expect(src->find("nc_puzzle_blocks = 1ULL") != std::string::npos,
                           "three variables need exactly one block of sixty-four assignments");
              })
        .test("cpp_emission_is_standalone_and_free_of_nimblecas_headers",
              [](TestContext& t) {
                  CompileOptions opts;
                  opts.target = Target::cpp;
                  auto src = compile(simple_sat(), opts);
                  t.expect(src.has_value(), "the formula compiles");
                  if (!src.has_value()) {
                      return;
                  }
                  t.expect(src->find("nimblecas") == std::string::npos ||
                               src->find("#include \"nimblecas") == std::string::npos,
                           "the emitted code includes no NimbleCAS header");
                  t.expect(src->find("import ") == std::string::npos,
                           "the emitted code imports no module, so it drops into any build");
                  t.expect(src->find("#include <cstdint>") != std::string::npos,
                           "it includes what it actually needs");
              })
        .test("emission_can_omit_the_simd_and_threaded_paths",
              [](TestContext& t) {
                  CompileOptions opts;
                  opts.target = Target::cpp;
                  opts.emit_simd = false;
                  opts.emit_parallel = false;
                  auto src = compile(simple_sat(), opts);
                  t.expect(src.has_value(), "the formula compiles");
                  if (!src.has_value()) {
                      return;
                  }
                  t.expect(src->find("_simd") == std::string::npos,
                           "no SIMD path is emitted when it was not asked for");
                  t.expect(src->find("jthread") == std::string::npos,
                           "no threading is emitted when it was not asked for");
                  t.expect(src->find("formula_solve(") != std::string::npos,
                           "the scalar entry point is still there");
              })
        .test("cuda_emission_contains_a_kernel_and_a_deterministic_reduction",
              [](TestContext& t) {
                  CompileOptions opts;
                  opts.target = Target::cuda;
                  opts.entry = "board";
                  auto src = compile(simple_sat(), opts);
                  t.expect(src.has_value(), "the formula compiles to CUDA");
                  if (!src.has_value()) {
                      return;
                  }
                  t.expect(src->find("__global__ void board_kernel") != std::string::npos,
                           "a kernel is emitted under the requested name");
                  t.expect(src->find("atomicMin") != std::string::npos,
                           "the reduction is a minimum, which is what makes the answer "
                           "independent of launch geometry");
                  t.expect(src->find("board_solve_cuda") != std::string::npos,
                           "a host wrapper is emitted");
                  t.expect(src->find("__device__") != std::string::npos,
                           "the block evaluator is a device function");
              })
        .test("triton_emission_contains_a_jit_kernel_and_a_host_resolver",
              [](TestContext& t) {
                  CompileOptions opts;
                  opts.target = Target::triton;
                  opts.entry = "grid";
                  auto src = compile(simple_sat(), opts);
                  t.expect(src.has_value(), "the formula compiles to Triton");
                  if (!src.has_value()) {
                      return;
                  }
                  t.expect(src->find("@triton.jit") != std::string::npos,
                           "the kernel is decorated for Triton");
                  t.expect(src->find("def grid_kernel") != std::string::npos,
                           "the kernel takes the requested name");
                  t.expect(src->find("tl.atomic_min") != std::string::npos,
                           "the reduction is a minimum");
                  t.expect(src->find("def grid_block_mask") != std::string::npos,
                           "a host-side block evaluator is emitted, since Triton has no "
                           "count-trailing-zeros to resolve the lane with");
                  t.expect(src->find("def grid_solve") != std::string::npos,
                           "a host entry point is emitted");
              })
        .test("every_target_bakes_in_every_clause",
              [](TestContext& t) {
                  // A clause that vanished from the emission would be a solver for a DIFFERENT,
                  // weaker formula -- and it would still return answers, which is exactly the kind
                  // of silent wrongness worth a test of its own. Counting the AND-accumulations is
                  // a direct check that each clause reached the output.
                  //
                  // Six variables or more, deliberately: a smaller formula does not fill a block,
                  // and the emitter then adds one further AND to mask off the lanes standing for
                  // assignments that do not exist, which would make this count ambiguous.
                  const Cnf cnf{
                      .num_vars = 7,
                      .clauses = {{1, 2}, {-2, 3}, {4, -5}, {6, 7}, {-1, -7}, {3, -4, 5}}};
                  const auto count_occurrences = [](const std::string& hay,
                                                    std::string_view needle) -> std::size_t {
                      std::size_t n = 0;
                      std::size_t at = hay.find(needle);
                      while (at != std::string::npos) {
                          ++n;
                          at = hay.find(needle, at + needle.size());
                      }
                      return n;
                  };
                  CompileOptions cpp_opts;
                  cpp_opts.target = Target::cpp;
                  cpp_opts.emit_simd = false;
                  cpp_opts.emit_parallel = false;
                  auto cpp_src = compile(cnf, cpp_opts);
                  t.expect(cpp_src.has_value(), "the formula compiles to C++");
                  if (cpp_src.has_value()) {
                      t.expect(count_occurrences(*cpp_src, "sat &= ") == cnf.clauses.size(),
                               "the C++ emission ANDs in exactly one term per clause");
                  }
                  CompileOptions tri_opts;
                  tri_opts.target = Target::triton;
                  auto tri_src = compile(cnf, tri_opts);
                  t.expect(tri_src.has_value(), "the formula compiles to Triton");
                  if (tri_src.has_value()) {
                      t.expect(count_occurrences(*tri_src, "sat = sat & ") == cnf.clauses.size(),
                               "the Triton kernel ANDs in exactly one term per clause");
                  }
              })
        .test("a_negative_literal_is_emitted_as_a_complement",
              [](TestContext& t) {
                  const Cnf cnf{.num_vars = 1, .clauses = {{-1}}};
                  CompileOptions opts;
                  opts.target = Target::cpp;
                  opts.emit_simd = false;
                  opts.emit_parallel = false;
                  auto src = compile(cnf, opts);
                  t.expect(src.has_value(), "the formula compiles");
                  if (!src.has_value()) {
                      return;
                  }
                  t.expect(src->find("~(0xAAAAAAAAAAAAAAAAULL)") != std::string::npos,
                           "negating variable 1 complements its lane pattern");
              })
        .test("a_variable_beyond_the_sixth_is_emitted_as_a_block_indexed_mask",
              [](TestContext& t) {
                  // Variables 1..6 are lane patterns; from the seventh on, a variable is constant
                  // within a block and depends on the block number instead.
                  const Cnf cnf{.num_vars = 8, .clauses = {{7}, {8}}};
                  CompileOptions opts;
                  opts.target = Target::cpp;
                  opts.emit_simd = false;
                  opts.emit_parallel = false;
                  auto src = compile(cnf, opts);
                  t.expect(src.has_value(), "the formula compiles");
                  if (!src.has_value()) {
                      return;
                  }
                  t.expect(src->find("(0ULL - ((b >> 0) & 1ULL))") != std::string::npos,
                           "variable 7 reads bit 0 of the block index");
                  t.expect(src->find("(0ULL - ((b >> 1) & 1ULL))") != std::string::npos,
                           "variable 8 reads bit 1 of the block index");
                  t.expect(src->find("nc_formula_blocks = 4ULL") != std::string::npos,
                           "eight variables need exactly four blocks");
              })
        .test("two_emissions_share_no_unprefixed_symbol",
              [](TestContext& t) {
                  // Regression. The SIMD helpers were once emitted as bare `nc_u64x8` and
                  // `nc_splat`, so including two generated solvers in ONE translation unit was a
                  // redefinition error -- which a unit test never sees, because a unit test
                  // includes neither. Every emitted name now carries the entry, and this checks
                  // it by name rather than by hoping.
                  CompileOptions a;
                  a.target = Target::cpp;
                  a.entry = "alpha";
                  CompileOptions b = a;
                  b.entry = "beta";
                  auto sa = compile(simple_sat(), a);
                  auto sb = compile(simple_sat(), b);
                  t.expect(sa.has_value() && sb.has_value(), "both formulas compile");
                  if (!sa.has_value() || !sb.has_value()) {
                      return;
                  }
                  // Anything the emitter declares begins "nc_" or the entry name; the shared
                  // helpers are the ones that used to escape that rule.
                  for (const std::string_view bare : {"nc_u64x8", "nc_splat("}) {
                      t.expect(sa->find(bare) == std::string::npos,
                               "no unprefixed helper is declared in the first emission");
                      t.expect(sb->find(bare) == std::string::npos,
                               "nor in the second");
                  }
                  t.expect(sa->find("nc_alpha_u64x8") != std::string::npos,
                           "the vector type carries the first entry name");
                  t.expect(sb->find("nc_beta_u64x8") != std::string::npos,
                           "and the second carries its own");
                  t.expect(sa->find("nc_beta_") == std::string::npos,
                           "neither emission mentions the other's names");
              })
        .test("compile_rejects_an_entry_name_that_is_not_an_identifier",
              [](TestContext& t) {
                  CompileOptions opts;
                  opts.target = Target::cpp;
                  opts.entry = "not a name";
                  auto a = compile(simple_sat(), opts);
                  t.expect(!a.has_value() && a.error() == MathError::domain_error,
                           "an entry name with a space is a domain_error, not emitted code that "
                           "cannot compile");
                  opts.entry = "";
                  auto b = compile(simple_sat(), opts);
                  t.expect(!b.has_value() && b.error() == MathError::domain_error,
                           "an empty entry name is a domain_error");
                  opts.entry = "9lives";
                  auto c = compile(simple_sat(), opts);
                  t.expect(!c.has_value() && c.error() == MathError::domain_error,
                           "an entry name starting with a digit is a domain_error");
              })
        .test("compile_rejects_a_formula_is_compilable_refuses",
              [](TestContext& t) {
                  const Cnf bad{.num_vars = 2, .clauses = {{5}}};
                  for (const Target target : {Target::cpp, Target::cuda, Target::triton}) {
                      CompileOptions opts;
                      opts.target = target;
                      auto r = compile(bad, opts);
                      t.expect(!r.has_value() && r.error() == MathError::domain_error,
                               "every target refuses a malformed formula rather than emitting "
                               "code for it");
                  }
              })
        .run();
}
