// Tests for nimblecas.csp_dist: finite-domain CSP as task graphs over an exact prefix partition.
// @author Olumuyiwa Oluwasanmi
//
// Two contracts are under test. First, THE ANSWER DOES NOT REVEAL WHICH EXECUTOR RAN IT: the
// distributed solve must return the same lexicographically-first assignment as the in-process
// `backtracking_search`, serially, in parallel, and over SGEE. Second, THE PARTITION IS EXACT:
// the per-prefix solution counts must sum to the whole problem's count, because a decomposition
// that double-counted or dropped an assignment would make both a distributed count and a
// distributed UNSAT verdict worthless. Every returned assignment is verified against the
// ORIGINAL problem rather than the restricted sub-problem the shard actually saw.

import std;
import nimblecas.core;
import nimblecas.csp;
import nimblecas.csp_dist;
import nimblecas.taskdag;
import nimblecas.taskdag_sgee;
import nimblecas.testing;

using nimblecas::as_csp;
using nimblecas::backtracking_search;
using nimblecas::ConstraintKind;
using nimblecas::FakeBrokerPort;
using nimblecas::InMemoryResultChannel;
using nimblecas::local_parallel_executor;
using nimblecas::MathError;
using nimblecas::Payload;
using nimblecas::serial_executor;
using nimblecas::SgeeDistributedExecutor;
using nimblecas::SgeeExecutorConfig;
using nimblecas::solution_count;
using nimblecas::TaskRegistry;
using nimblecas::WireConstraint;
using nimblecas::WireCsp;
using nimblecas::csp_dist::count_distributed;
using nimblecas::csp_dist::CountResult;
using nimblecas::csp_dist::decode_csp;
using nimblecas::csp_dist::decode_solution;
using nimblecas::csp_dist::encode_csp;
using nimblecas::csp_dist::encode_solution;
using nimblecas::csp_dist::max_prefix_tasks;
using nimblecas::csp_dist::register_ops;
using nimblecas::csp_dist::solve_distributed;
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

// N-queens declaratively: rows pairwise distinct, and no shared diagonal, which is exactly
// |row_i - row_j| != (j - i).
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

// Three mutually distinct variables over a two-value domain: arc consistent, yet plainly
// unsatisfiable. The right shape for insisting an UNSAT answer is complete rather than a
// budget having run out.
[[nodiscard]] auto pigeonhole_3_2() -> WireCsp {
    WireCsp w;
    w.domains = {range_domain(2), range_domain(2), range_domain(2)};
    w.constraints.push_back(WireConstraint{
        .kind = ConstraintKind::all_different, .scope = {0, 1, 2}, .params = {}});
    return w;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.csp_dist")
        .test("the_distributed_solution_is_the_one_the_serial_search_returns",
              [](TestContext& t) {
                  const auto w = wire_queens(6);
                  auto reference_csp = as_csp(w);
                  t.expect(reference_csp.has_value(), "the problem converts");
                  if (!reference_csp) {
                      return;
                  }
                  auto const want = backtracking_search(*reference_csp);
                  t.expect(want.has_value(), "the in-process search succeeds");

                  auto ser = serial_executor();
                  auto par = local_parallel_executor();
                  auto const a = solve_distributed(w, 1, *ser);
                  auto const b = solve_distributed(w, 2, *par);
                  t.expect(a.has_value() && b.has_value(), "both distributed runs succeed");
                  t.expect(a.value_or(std::nullopt) == want.value_or(std::nullopt),
                           "serial executor, one fixed variable: the identical assignment");
                  t.expect(b.value_or(std::nullopt) == want.value_or(std::nullopt),
                           "parallel executor, two fixed variables: still the identical "
                           "assignment -- the split does not change the answer");
              })
        .test("an_unsatisfiable_problem_comes_back_complete_not_unknown",
              [](TestContext& t) {
                  // The prefixes partition the space, so "no shard found anything" is a proof
                  // and not an exhausted budget. There is no unknown verdict to report here,
                  // which is exactly what distinguishes this from a portfolio.
                  const auto w = pigeonhole_3_2();
                  auto exec = serial_executor();
                  auto r = solve_distributed(w, 2, *exec);
                  t.expect(r.has_value(), "the solve itself succeeds");
                  t.expect(r.has_value() && !r->has_value(),
                           "and reports no solution, which is a complete answer");
                  auto n = count_distributed(w, 2, 0, *exec);
                  t.expect(n.has_value() && n->count == 0 && n->exact,
                           "the distributed count agrees: exactly zero solutions, exactly");
              })
        .test("the_prefix_counts_sum_to_the_whole_problems_count",
              [](TestContext& t) {
                  // Distributed model counting is only meaningful if the partition is exact.
                  // 6-queens has 4 solutions and 8-queens has 92; both must come back whole
                  // however the prefix is split.
                  auto exec = local_parallel_executor();
                  const auto six = wire_queens(6);
                  auto c6 = count_distributed(six, 2, 0, *exec);
                  t.expect(c6.has_value(), "the 6-queens count succeeds");
                  t.expect(c6.has_value() && c6->count == 4 && c6->exact,
                           "6-queens: exactly 4 solutions, summed over 36 prefixes");

                  const auto eight = wire_queens(8);
                  auto c8_a = count_distributed(eight, 1, 0, *exec);
                  auto c8_b = count_distributed(eight, 2, 0, *exec);
                  t.expect(c8_a.has_value() && c8_a->count == 92,
                           "8-queens over 8 prefixes: 92 solutions");
                  t.expect(c8_b.has_value() && c8_b->count == 92,
                           "8-queens over 64 prefixes: the same 92, so the split is exact");

                  // And it agrees with the in-process counter on the same problem.
                  auto csp = as_csp(eight);
                  auto const serial_n = csp.has_value() ? solution_count(*csp, 0)
                                                  : nimblecas::make_error<std::uint64_t>(
                                                        MathError::domain_error);
                  t.expect(serial_n.has_value() && serial_n.value_or(0) == 92,
                           "which is what the in-process counter says too");
              })
        .test("a_capped_count_says_it_is_a_lower_bound_rather_than_a_total",
              [](TestContext& t) {
                  // Honesty: a shard that stopped at its limit may have had more to find, so
                  // the merged count is a lower bound and must be labelled one.
                  auto exec = serial_executor();
                  const auto w = wire_queens(8);
                  auto capped = count_distributed(w, 1, 1, *exec);
                  t.expect(capped.has_value(), "the capped count succeeds");
                  t.expect(capped.has_value() && !capped->exact,
                           "and reports itself INEXACT because a shard hit its limit");
                  t.expect(capped.has_value() && capped->count <= 92,
                           "the lower bound does not exceed the true total");
                  auto uncapped = count_distributed(w, 1, 0, *exec);
                  t.expect(uncapped.has_value() && uncapped->exact && uncapped->count == 92,
                           "while an uncapped count is exact and complete");
              })
        .test("the_wire_format_round_trips_and_rejects_corruption",
              [](TestContext& t) {
                  const auto w = wire_queens(5);
                  auto bytes = encode_csp(w);
                  t.expect(bytes.has_value(), "the problem encodes");
                  if (!bytes) {
                      return;
                  }
                  auto back = decode_csp(*bytes);
                  t.expect(back.has_value(), "and decodes");
                  t.expect(back.has_value() && back->domains == w.domains,
                           "with the domains intact");
                  t.expect(back.has_value() && back->constraints == w.constraints,
                           "and every constraint intact, kind, scope and parameters");

                  // A truncated payload is a syntax error, never a partial problem.
                  auto truncated = *bytes;
                  truncated.resize(truncated.size() / 2);
                  auto bad = decode_csp(truncated);
                  t.expect(!bad.has_value() && bad.error() == MathError::syntax_error,
                           "a truncated payload is refused");

                  // So is a wrong magic number.
                  auto wrong_magic = *bytes;
                  wrong_magic[0] = static_cast<std::byte>(0xFF);
                  auto bm = decode_csp(wrong_magic);
                  t.expect(!bm.has_value() && bm.error() == MathError::syntax_error,
                           "so is a payload that is not one of ours");

                  // Solutions round-trip too, including the absence of one.
                  const std::optional<std::vector<std::int64_t>> some{{1, 3, 0, 2}};
                  auto se = encode_solution(some);
                  t.expect(se.has_value() && decode_solution(*se).value_or(std::nullopt) == some,
                           "a solution round-trips");
                  const std::optional<std::vector<std::int64_t>> none{};
                  auto ne = encode_solution(none);
                  t.expect(ne.has_value() && !decode_solution(*ne).value_or(some).has_value(),
                           "and so does the absence of one");
              })
        .test("shape_faults_are_refused_before_any_task_is_built",
              [](TestContext& t) {
                  auto exec = serial_executor();
                  const auto w = wire_queens(4);
                  auto zero = solve_distributed(w, 0, *exec);
                  t.expect(!zero.has_value() && zero.error() == MathError::domain_error,
                           "fixing zero variables is a domain_error, not a silent whole-problem "
                           "solve");
                  auto too_many = solve_distributed(w, 9, *exec);
                  t.expect(!too_many.has_value() && too_many.error() == MathError::domain_error,
                           "fixing more variables than exist is a domain_error");

                  WireCsp malformed;
                  malformed.domains = {range_domain(3), range_domain(3)};
                  malformed.constraints.push_back(WireConstraint{
                      .kind = ConstraintKind::not_equal, .scope = {0, 5}, .params = {}});
                  auto const bad = solve_distributed(malformed, 1, *exec);
                  t.expect(!bad.has_value(),
                           "a malformed problem is refused rather than distributed");
              })
        .test("a_split_larger_than_the_task_cap_is_refused_honestly",
              [](TestContext& t) {
                  // The tasks are materialised eagerly, so the cap is real. Exceeding it is an
                  // overflow, not a quietly-truncated partition -- a truncated partition would
                  // silently turn a complete search into an incomplete one.
                  WireCsp wide;
                  for (int i = 0; i < 6; ++i) {
                      wide.domains.push_back(range_domain(50));  // 50^6 prefixes
                  }
                  wide.constraints.push_back(WireConstraint{
                      .kind = ConstraintKind::not_equal, .scope = {0, 1}, .params = {}});
                  auto exec = serial_executor();
                  auto r = solve_distributed(wide, 6, *exec);
                  t.expect(!r.has_value() && r.error() == MathError::overflow,
                           "a prefix product past max_prefix_tasks is an honest overflow");
                  t.expect(max_prefix_tasks > 0, "the cap is a real published constant");
              })
        .test("the_same_answer_comes_back_over_the_sgee_distributed_executor",
              [](TestContext& t) {
                  // The Executor seam alone is not enough: this drives the real distributed
                  // executor, so the tasks go through Payload encoding, a broker queue and the
                  // worker's own registry lookup of the op id. The transport is an in-process
                  // broker, so what this establishes is the executor, the wire format and the
                  // registry contract -- not a real cluster under partition or latency.
                  TaskRegistry reg;
                  t.expect(register_ops(reg).has_value(),
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

                  const auto w = wire_queens(6);
                  auto ser = serial_executor();
                  auto const a = solve_distributed(w, 2, *ser);
                  auto const b = solve_distributed(w, 2, sgee);
                  t.expect(a.has_value() && b.has_value(), "both runs succeed");
                  t.expect(a.value_or(std::nullopt) == b.value_or(std::nullopt),
                           "and the cluster returns the identical assignment");

                  auto const ca = count_distributed(w, 2, 0, *ser);
                  auto const cb = count_distributed(w, 2, 0, sgee);
                  t.expect(ca.has_value() && cb.has_value(), "both counts succeed");
                  t.expect(ca.value_or(CountResult{}) == cb.value_or(CountResult{}),
                           "and the counts agree exactly, flag included");

                  const auto unsat = pigeonhole_3_2();
                  auto u = solve_distributed(unsat, 2, sgee);
                  t.expect(u.has_value() && !u->has_value(),
                           "an UNSAT proof survives the round trip through the broker");
              })
        .run();
}
