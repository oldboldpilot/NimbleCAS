// Tests for nimblecas.bitcsp: the branchless bitset-domain CSP and bitmask N-queens.
// @author Olumuyiwa Oluwasanmi
//
// The branchless bitmask N-queens counts are pinned against the classic values (4->2, 6->4,
// 8->92, 10->724) in BOTH serial and parallel modes, with the two modes asserted equal.
// AC-3 over support masks is checked against a hand-computed pruning and against an emptied
// domain (a valid nullopt, not an error). A cross-check proves count_nqueens(8) agrees with the
// independent value-branching reference solver in nimblecas.csp on the same 8-queens problem.

import std;
import nimblecas.core;
import nimblecas.bitset;
import nimblecas.bitcsp;
import nimblecas.csp;
import nimblecas.testing;

using nimblecas::ac3_bitset;
using nimblecas::BinaryConstraint;
using nimblecas::BitConstraint;
using nimblecas::BitCsp;
using nimblecas::Bitset;
using nimblecas::count_nqueens;
using nimblecas::Csp;
using nimblecas::make_bit_constraint;
using nimblecas::MathError;
using nimblecas::solution_count;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

using Indices = std::vector<std::size_t>;

// A full domain Bitset over [0, k): every value initially live.
[[nodiscard]] auto full_domain(std::size_t k) -> Bitset {
    Bitset b(k);
    b.set_all();
    return b;
}

// The reference N-queens CSP (identical construction to tests/csp_tests.cpp): variable i is the
// row of the queen in column i; each column pair forbids the same row and the two diagonals.
[[nodiscard]] auto queens_csp(std::int64_t n) -> Csp {
    Csp csp;
    const auto sz = static_cast<std::size_t>(n);
    std::vector<std::int64_t> dom;
    dom.reserve(sz);
    for (std::int64_t v = 0; v < n; ++v) {
        dom.push_back(v);
    }
    csp.domains.assign(sz, dom);
    for (std::size_t i = 0; i < sz; ++i) {
        for (std::size_t j = i + 1; j < sz; ++j) {
            const std::int64_t d = static_cast<std::int64_t>(j) - static_cast<std::int64_t>(i);
            csp.binary.push_back(BinaryConstraint{
                i, j, [d](std::int64_t a, std::int64_t b) {
                    return a != b && (a - b != d) && (b - a != d);
                }});
        }
    }
    return csp;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.bitcsp")
        .test("nqueens_counts_serial",
              [](TestContext& t) {
                  const std::pair<int, std::uint64_t> cases[] = {
                      {4, 2}, {6, 4}, {8, 92}, {10, 724}};
                  for (const auto& [n, expected] : cases) {
                      auto c = count_nqueens(n, false);
                      t.expect(c.has_value(), "serial count succeeds");
                      t.expect(c.value_or(0) == expected, "serial N-queens count matches");
                  }
              })
        .test("nqueens_counts_parallel_equal_serial",
              [](TestContext& t) {
                  const std::pair<int, std::uint64_t> cases[] = {
                      {4, 2}, {6, 4}, {8, 92}, {10, 724}};
                  for (const auto& [n, expected] : cases) {
                      auto serial = count_nqueens(n, false);
                      auto parallel = count_nqueens(n, true);
                      t.expect(serial.has_value() && parallel.has_value(), "both modes succeed");
                      t.expect(parallel.value_or(0) == expected, "parallel N-queens count matches");
                      t.expect(serial.value_or(0) == parallel.value_or(1),
                               "parallel count equals serial count");
                  }
              })
        .test("nqueens_small_and_domain_error",
              [](TestContext& t) {
                  auto one = count_nqueens(1, false);
                  t.expect(one.value_or(0) == 1, "1-queens has exactly 1 solution");
                  auto two = count_nqueens(2, false);
                  t.expect(two.value_or(9) == 0, "2-queens has no solution");
                  auto three = count_nqueens(3, true);
                  t.expect(three.value_or(9) == 0, "3-queens has no solution (parallel too)");

                  auto zero = count_nqueens(0, false);
                  t.expect(!zero.has_value() && zero.error() == MathError::domain_error,
                           "n = 0 -> domain_error");
                  auto neg = count_nqueens(-4, false);
                  t.expect(!neg.has_value() && neg.error() == MathError::domain_error,
                           "negative n -> domain_error");
                  auto big = count_nqueens(33, false);
                  t.expect(!big.has_value() && big.error() == MathError::domain_error,
                           "n > 32 -> domain_error");
              })
        .test("ac3_bitset_prunes_hand_computed",
              [](TestContext& t) {
                  // X, Y over values {0,1,2} with the constraint X < Y. Arc consistency removes
                  // value 2 from D[X] (no larger Y) and value 0 from D[Y] (no smaller X), leaving
                  // D[X] = {0,1}, D[Y] = {1,2}.
                  BitCsp csp;
                  csp.domains = {full_domain(3), full_domain(3)};
                  csp.constraints.push_back(make_bit_constraint(
                      0, 1, 3, 3, [](int a, int b) { return a < b; }));
                  auto r = ac3_bitset(csp);
                  t.expect(r.has_value(), "ac3_bitset succeeds");
                  t.expect(r.value_or(std::nullopt).has_value(), "domains stay non-empty");
                  if (r && *r) {
                      const auto& d = **r;
                      t.expect(d[0].set_bits() == Indices({0, 1}), "D[X] pruned to {0,1}");
                      t.expect(d[1].set_bits() == Indices({1, 2}), "D[Y] pruned to {1,2}");
                  }
              })
        .test("ac3_bitset_detects_emptied_domain",
              [](TestContext& t) {
                  // X, Y over the single value {0} with X < Y: value 0 in X has no support, so
                  // D[X] empties. An emptied domain is a valid nullopt, not an error.
                  BitCsp csp;
                  csp.domains = {full_domain(1), full_domain(1)};
                  csp.constraints.push_back(make_bit_constraint(
                      0, 1, 1, 1, [](int a, int b) { return a < b; }));
                  auto r = ac3_bitset(csp);
                  t.expect(r.has_value(), "ac3_bitset returns a value (empty domain is not error)");
                  t.expect(!r.value_or(std::nullopt).has_value(),
                           "arc-inconsistent CSP yields nullopt");
              })
        .test("ac3_bitset_all_different_chain_stays_consistent",
              [](TestContext& t) {
                  // Three variables over {0,1,2} pairwise not-equal. This is arc consistent with
                  // full domains (each value still has two differing partners), so AC-3 prunes
                  // nothing and reports consistency.
                  BitCsp csp;
                  csp.domains = {full_domain(3), full_domain(3), full_domain(3)};
                  const auto ne = [](int a, int b) { return a != b; };
                  csp.constraints.push_back(make_bit_constraint(0, 1, 3, 3, ne));
                  csp.constraints.push_back(make_bit_constraint(0, 2, 3, 3, ne));
                  csp.constraints.push_back(make_bit_constraint(1, 2, 3, 3, ne));
                  auto r = ac3_bitset(csp);
                  t.expect(r.value_or(std::nullopt).has_value(), "stays arc consistent");
                  if (r && *r) {
                      const auto& d = **r;
                      t.expect(d[0].set_bits() == Indices({0, 1, 2}) &&
                                   d[1].set_bits() == Indices({0, 1, 2}) &&
                                   d[2].set_bits() == Indices({0, 1, 2}),
                               "no value pruned (!= is arc consistent over a size-3 domain)");
                  }
              })
        .test("ac3_bitset_domain_error_paths",
              [](TestContext& t) {
                  // Empty variable set.
                  BitCsp empty;
                  auto e = ac3_bitset(empty);
                  t.expect(!e.has_value() && e.error() == MathError::domain_error,
                           "empty variable set -> domain_error");

                  // Capacity-0 domain.
                  BitCsp zero;
                  zero.domains = {full_domain(2), Bitset(0)};
                  auto z = ac3_bitset(zero);
                  t.expect(!z.has_value() && z.error() == MathError::domain_error,
                           "capacity-0 domain -> domain_error");

                  // Out-of-range constraint scope: constraint references variable 2 of 2.
                  BitCsp oob;
                  oob.domains = {full_domain(2), full_domain(2)};
                  BitConstraint bad = make_bit_constraint(
                      0, 1, 2, 2, [](int a, int b) { return a != b; });
                  bad.y = 2;  // now references a non-existent variable
                  oob.constraints.push_back(bad);
                  auto o = ac3_bitset(oob);
                  t.expect(!o.has_value() && o.error() == MathError::domain_error,
                           "out-of-range constraint scope -> domain_error");

                  // Support table shaped to the wrong capacity.
                  BitCsp mis;
                  mis.domains = {full_domain(2), full_domain(2)};
                  BitConstraint wrong = make_bit_constraint(
                      0, 1, 2, 3, [](int a, int b) { return a == b; });  // ky = 3 but D[Y] is 2
                  mis.constraints.push_back(wrong);
                  auto m = ac3_bitset(mis);
                  t.expect(!m.has_value() && m.error() == MathError::domain_error,
                           "mis-sized support table -> domain_error");
              })
        .test("nqueens_cross_checks_reference_csp",
              [](TestContext& t) {
                  // The branchless bitmask engine must agree with the independent value-branching
                  // reference solver (nimblecas.csp) on the number of 8-queens solutions.
                  auto bit8 = count_nqueens(8, false);
                  auto bit8p = count_nqueens(8, true);
                  auto ref8 = solution_count(queens_csp(8), 0);
                  t.expect(bit8.has_value() && ref8.has_value(), "both engines succeed");
                  t.expect(bit8.value_or(0) == 92, "bitmask 8-queens count is 92");
                  t.expect(ref8.value_or(0) == 92, "reference CSP 8-queens count is 92");
                  t.expect(bit8.value_or(0) == ref8.value_or(1),
                           "branchless engine agrees with the reference solver on 8-queens");
                  t.expect(bit8p.value_or(0) == ref8.value_or(1),
                           "parallel branchless engine also agrees with the reference solver");
              })
        .run();
}
