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
using nimblecas::sat_compile::is_compilable_for;
using nimblecas::sat_compile::reference_walksat;
using nimblecas::sat_compile::SlsVariant;
using nimblecas::sat_compile::Strategy;
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
    for (unsigned mask = 0; mask < 8; ++mask) {
        std::vector<std::int64_t> clause;
        for (unsigned v = 0; v < 3; ++v) {
            const bool bit = ((mask >> v) & 1U) != 0;
            // Exclude the assignment `mask` by asserting at least one variable differs from it.
            //
            // The variable number is built as int64_t BEFORE it is negated. Written as
            // `-(v + 1)` on the unsigned `v` it is unsigned arithmetic: literal -1 came out
            // as 4294967295, so this formula named three variables that do not exist, every
            // solver rejected it as malformed, and the case proved nothing.
            const auto lit = static_cast<std::int64_t>(v) + 1;
            clause.push_back(bit ? -lit : lit);
        }
        cnf.clauses.push_back(clause);
    }
    return cnf;
}

constexpr std::uint64_t plenty = 1ULL << 20U;

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.sat_compile")
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
                  t.expect(src->contains("puzzle_solve_range"),
                           "the range entry point is emitted under the requested name");
                  t.expect(src->contains("puzzle_solve("),
                           "the whole-space entry point is emitted");
                  t.expect(src->contains("puzzle_solve_parallel"),
                           "the threaded entry point is emitted");
                  t.expect(src->contains("puzzle_solve_range_simd"),
                           "the SIMD range entry point is emitted");
                  t.expect(src->contains("nc_puzzle_none"),
                           "the no-solution sentinel is emitted");
                  t.expect(src->contains("nc_puzzle_blocks = 1ULL"),
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
                  t.expect(!src->contains("nimblecas") ||
                               !src->contains("#include \"nimblecas"),
                           "the emitted code includes no NimbleCAS header");
                  t.expect(!src->contains("import "),
                           "the emitted code imports no module, so it drops into any build");
                  t.expect(src->contains("#include <cstdint>"),
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
                  t.expect(!src->contains("_simd"),
                           "no SIMD path is emitted when it was not asked for");
                  t.expect(!src->contains("jthread"),
                           "no threading is emitted when it was not asked for");
                  t.expect(src->contains("formula_solve("),
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
                  t.expect(src->contains("__global__ void board_kernel"),
                           "a kernel is emitted under the requested name");
                  t.expect(src->contains("atomicMin"),
                           "the reduction is a minimum, which is what makes the answer "
                           "independent of launch geometry");
                  t.expect(src->contains("board_solve_cuda"),
                           "a host wrapper is emitted");
                  t.expect(src->contains("__device__"),
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
                  t.expect(src->contains("@triton.jit"),
                           "the kernel is decorated for Triton");
                  t.expect(src->contains("def grid_kernel"),
                           "the kernel takes the requested name");
                  t.expect(src->contains("tl.atomic_min"),
                           "the reduction is a minimum");
                  t.expect(src->contains("def grid_block_mask"),
                           "a host-side block evaluator is emitted, since Triton has no "
                           "count-trailing-zeros to resolve the lane with");
                  t.expect(src->contains("def grid_solve"),
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
                  t.expect(src->contains("~(0xAAAAAAAAAAAAAAAAULL)"),
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
                  t.expect(src->contains("(0ULL - ((b >> 0) & 1ULL))"),
                           "variable 7 reads bit 0 of the block index");
                  t.expect(src->contains("(0ULL - ((b >> 1) & 1ULL))"),
                           "variable 8 reads bit 1 of the block index");
                  t.expect(src->contains("nc_formula_blocks = 4ULL"),
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
                      t.expect(!sa->contains(bare),
                               "no unprefixed helper is declared in the first emission");
                      t.expect(!sb->contains(bare),
                               "nor in the second");
                  }
                  t.expect(sa->contains("nc_alpha_u64x8"),
                           "the vector type carries the first entry name");
                  t.expect(sb->contains("nc_beta_u64x8"),
                           "and the second carries its own");
                  t.expect(!sa->contains("nc_beta_"),
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
        .test("reference_walksat_solves_a_satisfiable_formula_and_verifies_its_model",
              [](TestContext& t) {
                  const Cnf cnf = simple_sat();
                  auto r = reference_walksat(cnf, 16, 1000, 50, 12345);
                  t.expect(r.has_value(), "the walk succeeds");
                  if (!r.has_value()) {
                      return;
                  }
                  t.expect(r->found, "a satisfiable formula is solved");
                  if (!r->found) {
                      return;
                  }
                  t.expect(verify_assignment(cnf, r->model),
                           "the model it returns really satisfies every clause");
              })
        .test("reference_walksat_never_claims_unsatisfiability",
              [](TestContext& t) {
                  // The single most important property of a local search: it CANNOT prove
                  // unsatisfiability, so on an unsatisfiable formula it must come back
                  // empty-handed rather than confident. `found == false` is UNKNOWN, and the
                  // return type has no way to say `unsatisfiable` at all -- which is the design.
                  const Cnf cnf = simple_unsat();
                  auto r = reference_walksat(cnf, 64, 10000, 50, 999);
                  t.expect(r.has_value(), "the walk succeeds as a call");
                  if (!r.has_value()) {
                      return;
                  }
                  t.expect(!r->found, "no model is found, because none exists");
                  t.expect(r->model.empty(), "and no model is offered");
                  t.expect(r->walkers_run == 64,
                           "every walker was run, since none could succeed");
              })
        .test("reference_walksat_is_deterministic_in_its_seed",
              [](TestContext& t) {
                  const Cnf cnf = simple_sat();
                  auto a = reference_walksat(cnf, 32, 5000, 50, 0xABCDEF);
                  auto b = reference_walksat(cnf, 32, 5000, 50, 0xABCDEF);
                  t.expect(a.has_value() && b.has_value(), "both walks succeed");
                  if (!a.has_value() || !b.has_value()) {
                      return;
                  }
                  t.expect(a->found == b->found, "the same seed reaches the same verdict");
                  t.expect(a->walker == b->walker, "and the same winning walker");
                  t.expect(a->model == b->model, "and the same model, bit for bit");
                  const auto c = reference_walksat(cnf, 32, 5000, 50, 0x123456);
                  t.expect(c.has_value(), "a different seed also succeeds");
              })
        .test("reference_walksat_stops_at_the_first_successful_walker",
              [](TestContext& t) {
                  const Cnf cnf = simple_sat();
                  auto r = reference_walksat(cnf, 1000, 5000, 50, 7);
                  t.expect(r.has_value(), "the walk succeeds");
                  if (!r.has_value() || !r->found) {
                      return;
                  }
                  t.expect(r->walkers_run == r->walker + 1,
                           "walkers after the winner are never run -- the count says so rather "
                           "than the caller having to assume it");
              })
        .test("reference_walksat_scales_past_the_enumerator_limit",
              [](TestContext& t) {
                  // The entire reason this strategy exists. Two hundred variables is 2^200
                  // assignments, which the exhaustive path cannot even number, let alone visit.
                  // A planted solution guarantees the formula is satisfiable, so a failure here
                  // would be the solver rather than the instance.
                  std::mt19937_64 rng(0xBEEF1234ULL);
                  constexpr std::size_t nv = 200;
                  std::vector<bool> planted(nv);
                  for (std::size_t v = 0; v < nv; ++v) {
                      planted[v] = (rng() % 2) == 0;
                  }
                  Cnf cnf{.num_vars = nv, .clauses = {}};
                  while (cnf.clauses.size() < nv * 4) {
                      std::vector<std::int64_t> clause;
                      bool satisfied = false;
                      for (int k = 0; k < 3; ++k) {
                          const auto v = static_cast<std::int64_t>(1 + (rng() % nv));
                          const bool pos = (rng() % 2) == 0;
                          clause.push_back(pos ? v : -v);
                          if (pos == planted[static_cast<std::size_t>(v) - 1]) {
                              satisfied = true;
                          }
                      }
                      if (satisfied) {
                          cnf.clauses.push_back(clause);
                      }
                  }
                  t.expect(is_compilable_for(cnf, Strategy::walksat).has_value(),
                           "two hundred variables is compilable for walksat");
                  t.expect(!is_compilable_for(cnf, Strategy::exhaustive).has_value(),
                           "and is refused by the enumerator, which is the whole point");
                  auto r = reference_walksat(cnf, 32, 200000, 50, 4242);
                  t.expect(r.has_value(), "the walk succeeds");
                  if (!r.has_value()) {
                      return;
                  }
                  t.expect(r->found, "a 200-variable planted instance is solved");
                  if (!r->found) {
                      return;
                  }
                  t.expect(verify_assignment(cnf, r->model),
                           "and the model satisfies all eight hundred clauses");
              })
        .test("reference_walksat_solves_a_fifty_thousand_variable_instance",
              [](TestContext& t) {
                  // The instance the reference documentation quotes a timing for. It is here so
                  // that claim is REPRODUCIBLE rather than merely asserted: a reader who doubts
                  // the number can run this test and time it. Nothing here asserts a duration --
                  // a wall-clock bound would be a flaky test on a shared machine, and a flaky
                  // test is worse than no test. What is asserted is the part that must hold on
                  // every machine: the walk finds a model, and the model satisfies the formula.
                  std::mt19937_64 rng(0xBEEF1234ULL);
                  constexpr std::size_t nv = 50000;
                  std::vector<bool> planted(nv);
                  for (std::size_t v = 0; v < nv; ++v) {
                      planted[v] = (rng() % 2) == 0;
                  }
                  Cnf cnf{.num_vars = nv, .clauses = {}};
                  while (cnf.clauses.size() < nv * 42 / 10) {
                      std::vector<std::int64_t> clause;
                      bool satisfied = false;
                      for (int k = 0; k < 3; ++k) {
                          const auto v = static_cast<std::int64_t>(1 + (rng() % nv));
                          const bool pos = (rng() % 2) == 0;
                          clause.push_back(pos ? v : -v);
                          if (pos == planted[static_cast<std::size_t>(v) - 1]) {
                              satisfied = true;
                          }
                      }
                      if (satisfied) {
                          cnf.clauses.push_back(clause);
                      }
                  }
                  t.expect(cnf.clauses.size() == 210000,
                           "the instance is the documented 210,000 clauses");
                  t.expect(is_compilable_for(cnf, Strategy::walksat).has_value(),
                           "fifty thousand variables is compilable for walksat");
                  auto r = reference_walksat(cnf, 8, 2000000, 50, 4242);
                  t.expect(r.has_value(), "the walk runs");
                  if (!r.has_value()) {
                      return;
                  }
                  t.expect(r->found, "and solves a fifty-thousand-variable planted instance");
                  if (!r->found) {
                      return;
                  }
                  t.expect(verify_assignment(cnf, r->model),
                           "with a model that satisfies every one of the 210,000 clauses");
              })
        .test("reference_walksat_agrees_with_dpll_on_satisfiability_where_it_succeeds",
              [](TestContext& t) {
                  // A local search that says SAT must be right, since it exhibits a model; where
                  // it says nothing, dpll is consulted to check it was not simply wrong to give
                  // up on something easy.
                  std::mt19937_64 rng(0x5A7C0DEULL);
                  int solved = 0;
                  int gave_up = 0;
                  for (int trial = 0; trial < 60; ++trial) {
                      const auto nv = static_cast<std::size_t>(4 + (rng() % 8));
                      const auto nc = static_cast<std::size_t>(nv * 3);
                      Cnf cnf{.num_vars = nv, .clauses = {}};
                      for (std::size_t c = 0; c < nc; ++c) {
                          std::vector<std::int64_t> clause;
                          for (int k = 0; k < 3; ++k) {
                              const auto v = static_cast<std::int64_t>(1 + (rng() % nv));
                              clause.push_back((rng() % 2) == 0 ? v : -v);
                          }
                          cnf.clauses.push_back(clause);
                      }
                      auto walk = reference_walksat(cnf, 24, 20000, 50, 31337 + trial);
                      auto truth = dpll(cnf);
                      if (!walk.has_value() || !truth.has_value()) {
                          t.expect(false, "both solvers succeed as calls");
                          continue;
                      }
                      if (walk->found) {
                          ++solved;
                          t.expect(verify_assignment(cnf, walk->model),
                                   "every model the walk reports really satisfies the formula");
                          t.expect(truth->verdict == SatVerdict::satisfiable,
                                   "and dpll agrees the formula is satisfiable");
                      } else {
                          ++gave_up;
                      }
                  }
                  t.expect(solved > 30,
                           "the walk solved most of the satisfiable instances, so the sweep "
                           "actually exercised the success path");
                  t.expect(solved + gave_up == 60, "every trial was accounted for");
              })
        .test("reference_walksat_guards_its_arguments",
              [](TestContext& t) {
                  const Cnf cnf = simple_sat();
                  auto a = reference_walksat(cnf, 0, 100, 50, 1);
                  t.expect(!a.has_value() && a.error() == MathError::domain_error,
                           "zero walkers is a domain_error");
                  auto b = reference_walksat(cnf, 4, 100, 101, 1);
                  t.expect(!b.has_value() && b.error() == MathError::domain_error,
                           "a noise percentage above 100 is a domain_error");
                  const Cnf malformed{.num_vars = 2, .clauses = {{5}}};
                  auto c = reference_walksat(malformed, 4, 100, 50, 1);
                  t.expect(!c.has_value() && c.error() == MathError::domain_error,
                           "a malformed formula is a domain_error");
              })
        .test("zero_flips_still_solves_a_formula_the_random_start_happens_to_satisfy",
              [](TestContext& t) {
                  // An edge worth pinning: with no flips allowed the walk can still succeed, if a
                  // starting assignment is already a model. Enough walkers make that near certain
                  // on a formula this loose, and the check must happen BEFORE the flip loop.
                  const Cnf easy{.num_vars = 3, .clauses = {{1, 2, 3}}};
                  auto r = reference_walksat(easy, 64, 0, 50, 5);
                  t.expect(r.has_value(), "the call succeeds");
                  if (!r.has_value()) {
                      return;
                  }
                  t.expect(r->found,
                           "some random start satisfies a single three-literal clause");
                  if (r->found) {
                      t.expect(verify_assignment(easy, r->model), "and it is a real model");
                  }
              })
        .test("the_variable_cap_applies_to_the_enumerator_and_not_to_the_walker",
              [](TestContext& t) {
                  const Cnf big{.num_vars = max_variables + 50, .clauses = {{1, 2}}};
                  auto ex = is_compilable_for(big, Strategy::exhaustive);
                  t.expect(!ex.has_value() && ex.error() == MathError::domain_error,
                           "the enumerator refuses more variables than it can number");
                  const auto wk = is_compilable_for(big, Strategy::walksat);
                  t.expect(wk.has_value(),
                           "the walker accepts them, because it never numbers an assignment");
              })
        .test("walksat_cpp_emission_contains_the_documented_entry_points",
              [](TestContext& t) {
                  CompileOptions opts;
                  opts.target = Target::cpp;
                  opts.strategy = Strategy::walksat;
                  opts.entry = "wsolve";
                  auto src = compile(simple_sat(), opts);
                  t.expect(src.has_value(), "the formula compiles to a WalkSAT solver");
                  if (!src.has_value()) {
                      return;
                  }
                  t.expect(src->contains("wsolve_solve_range"),
                           "the range entry point is emitted");
                  t.expect(src->contains("wsolve_solve_parallel"),
                           "the threaded entry point is emitted");
                  t.expect(src->contains("struct wsolve_result"),
                           "the result type is emitted");
                  t.expect(src->contains("nc_wsolve_lits"),
                           "the formula is baked into a literal table");
                  t.expect(src->contains("nc_wsolve_occp") &&
                               src->contains("nc_wsolve_occn"),
                           "occurrences are split by sign, which is what makes the break count "
                           "one contiguous scan");
                  t.expect(src->contains("nc_wsolve_break_count"),
                           "the break-count helper is emitted");
              })
        .test("walksat_cuda_emission_uses_managed_tables_and_one_thread_per_walker",
              [](TestContext& t) {
                  CompileOptions opts;
                  opts.target = Target::cuda;
                  opts.strategy = Strategy::walksat;
                  opts.entry = "gwalk";
                  auto src = compile(simple_sat(), opts);
                  t.expect(src.has_value(), "the formula compiles to CUDA");
                  if (!src.has_value()) {
                      return;
                  }
                  t.expect(src->contains("__global__ void gwalk_kernel"),
                           "a kernel is emitted");
                  t.expect(src->contains("__managed__"),
                           "the tables are managed memory -- a real formula overflows the 64 KB "
                           "constant bank, and the host needs to read them too");
                  t.expect(!src->contains("__constant__"),
                           "and specifically not constant memory");
                  t.expect(src->contains("atomicMin"),
                           "the winner is reduced by minimum, so it is the lowest walker rather "
                           "than whichever warp finished first");
                  t.expect(src->contains("gwalk_scratch_words"),
                           "the per-walker scratch size is exposed, since the host allocates it");
              })
        .test("walksat_triton_emission_is_a_scorer_and_says_so",
              [](TestContext& t) {
                  CompileOptions opts;
                  opts.target = Target::triton;
                  opts.strategy = Strategy::walksat;
                  opts.entry = "tscore";
                  auto src = compile(simple_sat(), opts);
                  t.expect(src.has_value(), "the formula compiles to Triton");
                  if (!src.has_value()) {
                      return;
                  }
                  t.expect(src->contains("@triton.jit"), "a kernel is emitted");
                  t.expect(src->contains("tscore_score_kernel"),
                           "it is a scorer");
                  t.expect(src->contains("tscore_sample_and_score"),
                           "with a host entry point that draws and scores samples");
                  t.expect(src->contains("deliberately NOT the walk"),
                           "and the emission says plainly that the walk belongs on CUDA, rather "
                           "than pretending a random walk suits a tensor kernel");
              })
        .test("walksat_compile_guards_its_options",
              [](TestContext& t) {
                  CompileOptions opts;
                  opts.strategy = Strategy::walksat;
                  opts.walkers = 0;
                  auto a = compile(simple_sat(), opts);
                  t.expect(!a.has_value() && a.error() == MathError::domain_error,
                           "zero walkers is a domain_error");
                  opts.walkers = 8;
                  opts.noise_percent = 101;
                  auto b = compile(simple_sat(), opts);
                  t.expect(!b.has_value() && b.error() == MathError::domain_error,
                           "a noise percentage above 100 is a domain_error");
              })
        .test("every_sls_variant_emits_its_own_choice_rule",
              [](TestContext& t) {
                  // The four variants share the whole walk and differ ONLY in which variable they
                  // pick, so what distinguishes them in the emitted source is the choice function
                  // and nothing else. Checking that directly is what stops a variant silently
                  // being a copy of another.
                  struct Case {
                      SlsVariant variant;
                      std::string_view marker;
                  };
                  const std::array<Case, 4> cases{{
                      {SlsVariant::skc, "WalkSAT/SKC"},
                      {SlsVariant::probsat, "probSAT"},
                      {SlsVariant::novelty_plus, "Novelty"},
                      {SlsVariant::adaptive_novelty_plus, "AdaptNovelty+"},
                  }};
                  for (const Case& c : cases) {
                      CompileOptions opts;
                      opts.target = Target::cpp;
                      opts.strategy = Strategy::walksat;
                      opts.variant = c.variant;
                      auto src = compile(simple_sat(), opts);
                      t.expect(src.has_value(), "the variant compiles");
                      if (!src.has_value()) {
                          continue;
                      }
                      t.expect(src->contains(c.marker),
                               "the emitted source names the rule it implements");
                      t.expect(src->contains("_choose("),
                               "and routes the decision through the shared choice function");
                  }
              })
        .test("only_the_adaptive_variant_emits_a_noise_schedule",
              [](TestContext& t) {
                  // AdaptNovelty+ is the only rule that changes its own noise; if the schedule
                  // leaked into the others they would stop being the algorithms they are named
                  // after.
                  for (const SlsVariant v : {SlsVariant::skc, SlsVariant::probsat,
                                             SlsVariant::novelty_plus}) {
                      CompileOptions opts;
                      opts.strategy = Strategy::walksat;
                      opts.variant = v;
                      auto src = compile(simple_sat(), opts);
                      t.expect(src.has_value(), "the variant compiles");
                      if (src.has_value()) {
                          t.expect(!src->contains("AdaptNovelty+"),
                                   "a non-adaptive variant emits no noise schedule");
                          t.expect(src->contains("unsigned noise = noise_percent;"),
                                   "and takes the caller's noise as given");
                      }
                  }
                  CompileOptions adaptive;
                  adaptive.strategy = Strategy::walksat;
                  adaptive.variant = SlsVariant::adaptive_novelty_plus;
                  auto src = compile(simple_sat(), adaptive);
                  t.expect(src.has_value(), "the adaptive variant compiles");
                  if (!src.has_value()) {
                      return;
                  }
                  t.expect(src->contains("unsigned noise = 0u;"),
                           "it starts from pure greed, as the published algorithm does, rather "
                           "than from the caller's setting");
                  t.expect(src->contains("best_unsat = now_unsat;"),
                           "and RESETS its reference count when it adapts -- without that the "
                           "decay is unreachable once the search plateaus and the noise ratchets "
                           "to its ceiling, which measured 0 solves out of 40");
              })
        .test("the_age_based_variants_clear_ages_at_the_start_of_every_walk",
              [](TestContext& t) {
                  // Regression. The age array is allocated once per walker RANGE but the flip
                  // counter restarts at zero for every walk, so an age left from the previous
                  // walker read as being in the future and the youngest test consulted nonsense.
                  // Fixing it took Novelty+ from 3 solves out of 40 to 7.
                  CompileOptions opts;
                  opts.strategy = Strategy::walksat;
                  opts.variant = SlsVariant::novelty_plus;
                  auto src = compile(simple_sat(), opts);
                  t.expect(src.has_value(), "the variant compiles");
                  if (!src.has_value()) {
                      return;
                  }
                  t.expect(src->contains("s->age[v] = 0ULL;"),
                           "ages are cleared inside the walk, not only at allocation");
                  const std::size_t walk_at = src->find("_walk(");
                  const std::size_t reset_at = src->find("s->age[v] = 0ULL;");
                  t.expect(walk_at != std::string::npos && reset_at != std::string::npos &&
                               reset_at > walk_at,
                           "and the clearing sits inside the walk function specifically");
              })
        .test("the_novelty_variants_score_by_make_minus_break",
              [](TestContext& t) {
                  // Scoring on break alone is a different and measurably worse algorithm: a
                  // variable that breaks two clauses but repairs five is a good move, and a rule
                  // blind to the five refuses it.
                  for (const SlsVariant v : {SlsVariant::novelty_plus,
                                             SlsVariant::adaptive_novelty_plus}) {
                      CompileOptions opts;
                      opts.strategy = Strategy::walksat;
                      opts.variant = v;
                      auto src = compile(simple_sat(), opts);
                      t.expect(src.has_value(), "the variant compiles");
                      if (src.has_value()) {
                          t.expect(src->contains("_make_count("),
                                   "a make count is emitted");
                          t.expect(src->contains("score[i] = (int)"),
                                   "and the ranking is a score rather than a raw break count");
                      }
                  }
                  CompileOptions skc;
                  skc.strategy = Strategy::walksat;
                  skc.variant = SlsVariant::skc;
                  auto src = compile(simple_sat(), skc);
                  t.expect(src.has_value() && !src->contains("score[i] = (int)"),
                           "SKC ranks on break alone, which is what SKC is");
              })
        .test("probsat_emits_no_noise_parameter_and_no_freebie_case",
              [](TestContext& t) {
                  // probSAT's whole claim is that it needs neither: a break of zero simply carries
                  // the largest weight, and one draw settles the choice.
                  CompileOptions opts;
                  opts.strategy = Strategy::walksat;
                  opts.variant = SlsVariant::probsat;
                  auto src = compile(simple_sat(), opts);
                  t.expect(src.has_value(), "probSAT compiles");
                  if (!src.has_value()) {
                      return;
                  }
                  t.expect(src->contains("weight[i] = 1ULL <<"),
                           "weights decay with the break count");
                  t.expect(!src->contains("freebie"),
                           "and there is no special case for a free move");
              })
        .test("every_variant_still_refuses_what_it_should",
              [](TestContext& t) {
                  for (const SlsVariant v : {SlsVariant::skc, SlsVariant::probsat,
                                             SlsVariant::novelty_plus,
                                             SlsVariant::adaptive_novelty_plus}) {
                      CompileOptions opts;
                      opts.strategy = Strategy::walksat;
                      opts.variant = v;
                      opts.walkers = 0;
                      auto a = compile(simple_sat(), opts);
                      t.expect(!a.has_value() && a.error() == MathError::domain_error,
                               "zero walkers is a domain_error whichever rule was asked for");
                      opts.walkers = 4;
                      const Cnf bad{.num_vars = 2, .clauses = {{7}}};
                      auto b = compile(bad, opts);
                      t.expect(!b.has_value() && b.error() == MathError::domain_error,
                               "and a malformed formula is refused before any code is emitted");
                  }
              })
        .run();
}
