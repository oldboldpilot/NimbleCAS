// Tests for nimblecas.csp_compile: a finite-domain CSP emitted as C++23, CUDA or Triton source.
// @author Olumuyiwa Oluwasanmi
//
// The emitter cannot compile or run its own output, so "the text looks right" would be no claim
// at all. What is tested instead is threefold. First, the SEMANTICS the emitted code implements,
// through the in-process reference implementations: `reference_exhaustive` must agree with
// `nimblecas.csp`'s own backtracking search on the same problem, assignment for assignment.
// Second, the HONEST REFUSALS -- a problem or option combination the emitter cannot serve must
// come back as the documented MathError rather than as plausible-looking source. Third, the
// structural invariants the module documents about its output: determinism, the entry-point
// name, and that the completeness or incompleteness of the strategy is stated in the text so a
// reader of the generated file cannot mistake one for the other.

import std;
import nimblecas.core;
import nimblecas.csp;
import nimblecas.csp_compile;
import nimblecas.testing;

using nimblecas::as_csp;
using nimblecas::backtracking_search;
using nimblecas::ConstraintKind;
using nimblecas::MathError;
using nimblecas::WireConstraint;
using nimblecas::WireCsp;
using nimblecas::csp_compile::conflict_count;
using nimblecas::csp_compile::emit;
using nimblecas::csp_compile::EmitOptions;
using nimblecas::csp_compile::entry_point_name;
using nimblecas::csp_compile::is_compilable_for;
using nimblecas::csp_compile::max_search_space;
using nimblecas::csp_compile::reference_exhaustive;
using nimblecas::csp_compile::reference_min_conflicts;
using nimblecas::csp_compile::search_space;
using nimblecas::csp_compile::Strategy;
using nimblecas::csp_compile::Target;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto range_domain(std::int64_t k) -> std::vector<std::int64_t> {
    std::vector<std::int64_t> d;
    d.reserve(static_cast<std::size_t>(k));
    for (std::int64_t v = 0; v < k; ++v) {
        d.push_back(v);
    }
    return d;
}

[[nodiscard]] auto wire_queens(std::int64_t n) -> WireCsp {
    WireCsp w;
    const auto un = static_cast<std::size_t>(n);
    for (std::size_t i = 0; i < un; ++i) {
        w.domains.push_back(range_domain(n));
    }
    std::vector<std::size_t> all;
    for (std::size_t i = 0; i < un; ++i) {
        all.push_back(i);
    }
    w.constraints.push_back(WireConstraint{
        .kind = ConstraintKind::all_different, .scope = std::move(all), .params = {}});
    for (std::size_t i = 0; i < un; ++i) {
        for (std::size_t j = i + 1; j < un; ++j) {
            w.constraints.push_back(WireConstraint{
                .kind = ConstraintKind::abs_diff_ne,
                .scope = {i, j},
                .params = {static_cast<std::int64_t>(j - i)}});
        }
    }
    return w;
}

// The SAME n-queens problem, but with the rows written as pairwise not-equals instead of one
// all_different. The solution sets are identical -- all_different IS pairwise not-equal -- but
// the two models give local search very different landscapes, which is what this encoding is
// here to exercise. See the module header on constraint granularity.
[[nodiscard]] auto wire_queens_pairwise(std::int64_t n) -> WireCsp {
    WireCsp w;
    const auto un = static_cast<std::size_t>(n);
    for (std::size_t i = 0; i < un; ++i) {
        w.domains.push_back(range_domain(n));
    }
    for (std::size_t i = 0; i < un; ++i) {
        for (std::size_t j = i + 1; j < un; ++j) {
            w.constraints.push_back(WireConstraint{
                .kind = ConstraintKind::not_equal, .scope = {i, j}, .params = {}});
            w.constraints.push_back(WireConstraint{
                .kind = ConstraintKind::abs_diff_ne,
                .scope = {i, j},
                .params = {static_cast<std::int64_t>(j - i)}});
        }
    }
    return w;
}

// Three variables that must be pairwise distinct over a two-value domain: unsatisfiable, and
// small enough that an exhaustive scan settles it instantly.
[[nodiscard]] auto pigeonhole_3_2() -> WireCsp {
    WireCsp w;
    w.domains = {range_domain(2), range_domain(2), range_domain(2)};
    w.constraints.push_back(WireConstraint{
        .kind = ConstraintKind::all_different, .scope = {0, 1, 2}, .params = {}});
    return w;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.csp_compile")
        .test("the_exhaustive_reference_returns_what_backtracking_search_returns",
              [](TestContext& t) {
                  // The emitted exhaustive scan claims to produce the lexicographically-first
                  // assignment. That is only meaningful if it is the SAME assignment the
                  // in-process solver returns, so the two are compared directly.
                  for (const std::int64_t n : {4, 5, 6}) {
                      const auto w = wire_queens(n);
                      auto csp = as_csp(w);
                      t.expect(csp.has_value(), "the problem converts to a functional Csp");
                      if (!csp) {
                          continue;
                      }
                      auto want = backtracking_search(*csp);
                      auto got = reference_exhaustive(w);
                      t.expect(want.has_value() && got.has_value(), "both searches succeed");
                      t.expect(want.value_or(std::nullopt) == got.value_or(std::nullopt),
                               "the exhaustive scan agrees with backtracking_search exactly");
                  }
              })
        .test("the_exhaustive_scan_settles_unsatisfiability_as_a_proof",
              [](TestContext& t) {
                  // Completeness is the whole reason this strategy exists: its silence has to
                  // mean "there is none", not "I gave up".
                  const auto w = pigeonhole_3_2();
                  auto r = reference_exhaustive(w);
                  t.expect(r.has_value(), "the scan succeeds");
                  t.expect(r.has_value() && !r->has_value(),
                           "and reports no solution, having examined every assignment");
                  auto space = search_space(w);
                  t.expect(space.has_value() && *space == 8,
                           "2 x 2 x 2 = 8 assignments, all of them examined");
              })
        .test("min_conflicts_finds_a_solution_and_never_claims_unsatisfiability",
              [](TestContext& t) {
                  // Local search is incomplete, and the test asserts both halves of that: it
                  // solves a satisfiable instance, and on an UNSATISFIABLE one it returns the
                  // same empty answer it returns when it merely runs out of steps -- which is
                  // exactly why that answer means UNKNOWN and must never be read as a proof.
                  // The PAIRWISE model is used here deliberately. Local search steers by the
                  // number of violated constraints, so one all_different over every column --
                  // violated or not, saying nothing about whether two queens collide or five --
                  // gives the walk no gradient to descend. The pairwise encoding has the
                  // identical solution set and a count that falls as the assignment improves.
                  const auto w = wire_queens_pairwise(8);
                  auto r = reference_min_conflicts(w, 8, 20000, 200, 0x1234ULL);
                  t.expect(r.has_value(), "the search succeeds");
                  t.expect(r.has_value() && r->has_value(),
                           "and finds a solution to 8-queens");
                  if (r.has_value() && r->has_value()) {
                      auto c = conflict_count(w, std::span<const std::int64_t>(**r));
                      t.expect(c.has_value() && *c == 0,
                               "the assignment it returns violates no constraint");
                  }

                  const auto unsat = pigeonhole_3_2();
                  auto u = reference_min_conflicts(unsat, 4, 500, 200, 0x99ULL);
                  t.expect(u.has_value() && !u->has_value(),
                           "on an unsatisfiable problem it returns the empty answer -- which is "
                           "UNKNOWN, indistinguishable from running out of steps, and is why "
                           "this strategy can never be used to prove unsatisfiability");
              })
        .test("min_conflicts_is_reproducible_from_its_seed",
              [](TestContext& t) {
                  const auto w = wire_queens_pairwise(8);
                  auto a = reference_min_conflicts(w, 4, 20000, 200, 0xABCDULL);
                  auto b = reference_min_conflicts(w, 4, 20000, 200, 0xABCDULL);
                  t.expect(a.has_value() && b.has_value(), "both runs succeed");
                  t.expect(a.value_or(std::nullopt) == b.value_or(std::nullopt),
                           "the same seed gives the identical assignment");
              })
        .test("conflict_count_is_zero_exactly_on_a_solution",
              [](TestContext& t) {
                  const auto w = wire_queens(4);
                  auto sol = reference_exhaustive(w);
                  t.expect(sol.has_value() && sol->has_value(), "4-queens has a solution");
                  if (!sol.has_value() || !sol->has_value()) {
                      return;
                  }
                  auto zero = conflict_count(w, std::span<const std::int64_t>(**sol));
                  t.expect(zero.has_value() && *zero == 0, "a solution violates nothing");

                  // Every queen on row 0: the all_different constraint and every diagonal
                  // constraint between adjacent columns are violated.
                  const std::vector<std::int64_t> all_same{0, 0, 0, 0};
                  auto many = conflict_count(w, std::span<const std::int64_t>(all_same));
                  t.expect(many.has_value() && *many > 0,
                           "an assignment with every queen in one row violates constraints");

                  const std::vector<std::int64_t> wrong_size{0, 0};
                  auto bad = conflict_count(w, std::span<const std::int64_t>(wrong_size));
                  t.expect(!bad.has_value() && bad.error() == MathError::domain_error,
                           "an assignment of the wrong length is a domain_error, not a count");
              })
        .test("emission_succeeds_for_every_supported_target_and_strategy",
              [](TestContext& t) {
                  const auto w = wire_queens(5);
                  for (const auto target : {Target::cpp, Target::cuda}) {
                      for (const auto strategy : {Strategy::exhaustive, Strategy::min_conflicts}) {
                          EmitOptions opts;
                          opts.target = target;
                          opts.strategy = strategy;
                          auto src = emit(w, opts);
                          t.expect(src.has_value(), "emission succeeds");
                          t.expect(src.has_value() && !src->empty(), "and returns real text");
                          t.expect(src.has_value() &&
                                       src->find(entry_point_name(opts)) != std::string::npos,
                                   "the text defines the entry point the module names");
                      }
                  }
                  EmitOptions tri;
                  tri.target = Target::triton;
                  auto ts = emit(w, tri);
                  t.expect(ts.has_value(), "the Triton scorer emits");
                  t.expect(ts.has_value() && ts->find("@triton.jit") != std::string::npos,
                           "and is a real Triton kernel");
                  t.expect(ts.has_value() && ts->find("conflicts_ptr") != std::string::npos,
                           "that writes a conflict count rather than a solution");
              })
        .test("the_emitted_text_states_which_guarantee_it_carries",
              [](TestContext& t) {
                  // A reader of the generated file must not have to guess whether a false
                  // return is a proof or an admission of ignorance.
                  const auto w = wire_queens(4);
                  EmitOptions ex;
                  ex.target = Target::cpp;
                  ex.strategy = Strategy::exhaustive;
                  auto a = emit(w, ex);
                  t.expect(a.has_value() && a->find("COMPLETE") != std::string::npos,
                           "the exhaustive output says it is complete");
                  t.expect(a.has_value() && a->find("PROOF") != std::string::npos,
                           "and that a false return is a proof");

                  EmitOptions mc;
                  mc.target = Target::cpp;
                  mc.strategy = Strategy::min_conflicts;
                  auto b = emit(w, mc);
                  t.expect(b.has_value() && b->find("INCOMPLETE") != std::string::npos,
                           "the local-search output says it is incomplete");
                  t.expect(b.has_value() && b->find("UNKNOWN") != std::string::npos,
                           "and that a false return means unknown, not unsatisfiable");
              })
        .test("emission_is_deterministic",
              [](TestContext& t) {
                  const auto w = wire_queens(6);
                  for (const auto target : {Target::cpp, Target::cuda, Target::triton}) {
                      EmitOptions opts;
                      opts.target = target;
                      auto a = emit(w, opts);
                      auto b = emit(w, opts);
                      t.expect(a.has_value() && b.has_value(), "both emissions succeed");
                      t.expect(a.value_or("x") == b.value_or("y"),
                               "the same problem emits byte-identical source");
                  }
              })
        .test("a_search_space_beyond_the_cap_is_refused_rather_than_emitted",
              [](TestContext& t) {
                  // Emitting an exhaustive scan of a space that cannot be walked would be
                  // promising something the generated code could never deliver.
                  WireCsp big;
                  for (int i = 0; i < 20; ++i) {
                      big.domains.push_back(range_domain(10));  // 10^20, past uint64 and the cap
                  }
                  big.constraints.push_back(WireConstraint{
                      .kind = ConstraintKind::not_equal, .scope = {0, 1}, .params = {}});
                  EmitOptions opts;
                  opts.strategy = Strategy::exhaustive;
                  auto r = emit(big, opts);
                  t.expect(!r.has_value() && r.error() == MathError::overflow,
                           "a space past the cap is an honest overflow, not source");

                  // The same problem IS compilable with local search, which has no such ceiling.
                  EmitOptions mc;
                  mc.strategy = Strategy::min_conflicts;
                  t.expect(is_compilable_for(big, mc).has_value(),
                           "local search has no search-space ceiling and accepts it");
                  t.expect(max_search_space > 0, "the cap is a real published constant");
              })
        .test("malformed_problems_and_nonsensical_options_are_refused",
              [](TestContext& t) {
                  WireCsp bad;
                  bad.domains = {range_domain(3), range_domain(3)};
                  bad.constraints.push_back(WireConstraint{
                      .kind = ConstraintKind::not_equal, .scope = {0, 9}, .params = {}});
                  EmitOptions opts;
                  auto r = emit(bad, opts);
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "an out-of-range scope is refused");

                  const auto w = wire_queens(4);
                  EmitOptions zero_shards;
                  zero_shards.strategy = Strategy::exhaustive;
                  zero_shards.shards = 0;
                  t.expect(!is_compilable_for(w, zero_shards).has_value(),
                           "zero shards is refused");

                  EmitOptions zero_walkers;
                  zero_walkers.strategy = Strategy::min_conflicts;
                  zero_walkers.walkers = 0;
                  t.expect(!is_compilable_for(w, zero_walkers).has_value(),
                           "zero walkers is refused");

                  EmitOptions bad_noise;
                  bad_noise.strategy = Strategy::min_conflicts;
                  bad_noise.noise_per_1024 = 2000;
                  t.expect(!is_compilable_for(w, bad_noise).has_value(),
                           "a noise probability above 1024/1024 is refused");

                  EmitOptions no_prefix;
                  no_prefix.name_prefix = "";
                  t.expect(!emit(w, no_prefix).has_value(), "an empty symbol prefix is refused");
              })
        .test("triton_refuses_to_pretend_it_emits_a_solver",
              [](TestContext& t) {
                  // The Triton target produces a scorer. Asking `is_compilable_for` about a
                  // STRATEGY on that target is asking for something it does not produce, and
                  // saying so beats emitting a kernel that quietly is not what was requested.
                  const auto w = wire_queens(4);
                  EmitOptions opts;
                  opts.target = Target::triton;
                  opts.strategy = Strategy::exhaustive;
                  auto r = is_compilable_for(w, opts);
                  t.expect(!r.has_value() && r.error() == MathError::not_implemented,
                           "a strategy on the Triton target is not_implemented");
                  t.expect(entry_point_name(opts) ==
                               std::string("nc_csp_conflict_kernel"),
                           "and the Triton entry point is named as a scorer, not a solver");
              })
        .test("every_constraint_kind_reaches_the_emitted_text",
              [](TestContext& t) {
                  // A kind the emitter silently dropped would produce source that solves a
                  // DIFFERENT, weaker problem -- the worst possible failure for this module.
                  WireCsp w;
                  w.domains = {range_domain(4), range_domain(4), range_domain(4)};
                  w.constraints = {
                      WireConstraint{.kind = ConstraintKind::not_equal, .scope = {0, 1}, .params = {}},
                      WireConstraint{.kind = ConstraintKind::equal, .scope = {1, 2}, .params = {}},
                      WireConstraint{.kind = ConstraintKind::less_equal, .scope = {0, 2}, .params = {1}},
                      WireConstraint{.kind = ConstraintKind::abs_diff_ne, .scope = {0, 2}, .params = {2}},
                      WireConstraint{
                          .kind = ConstraintKind::all_different, .scope = {0, 1, 2}, .params = {}},
                      WireConstraint{
                          .kind = ConstraintKind::linear_eq, .scope = {0, 1}, .params = {1, 1, 3}},
                      WireConstraint{
                          .kind = ConstraintKind::linear_le, .scope = {0, 1}, .params = {1, 1, 5}},
                      WireConstraint{.kind = ConstraintKind::table_allowed,
                                     .scope = {0, 1},
                                     .params = {0, 1, 2, 3}},
                  };
                  EmitOptions opts;
                  auto src = emit(w, opts);
                  t.expect(src.has_value(), "a problem using every kind emits");
                  if (!src) {
                      return;
                  }
                  for (const char* name : {"not_equal", "equal", "less_equal", "abs_diff_ne",
                                           "all_different", "linear_eq", "linear_le",
                                           "table_allowed"}) {
                      t.expect(src->find(name) != std::string::npos,
                               "the emitted source accounts for every constraint kind");
                  }
              })
        .run();
}
