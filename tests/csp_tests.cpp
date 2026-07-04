// Tests for nimblecas.csp: finite-domain constraint satisfaction (AC-3, backtracking,
// forward checking, parallel search).
// @author Olumuyiwa Oluwasanmi
//
// Every returned assignment is checked against the constraints it must satisfy, and each
// solver is exercised so its determinism / first-solution contract is observable. Classic
// benchmarks (N-queens solution counts, Australia map-colouring) pin the numbers exactly.

import std;
import nimblecas.core;
import nimblecas.csp;
import nimblecas.testing;

using nimblecas::ac3;
using nimblecas::backtracking_search;
using nimblecas::backtracking_search_fc;
using nimblecas::BinaryConstraint;
using nimblecas::Constraint;
using nimblecas::Csp;
using nimblecas::MathError;
using nimblecas::parallel_search;
using nimblecas::solution_count;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// A full domain {0, 1, ..., k-1}.
[[nodiscard]] auto range_domain(std::int64_t k) -> std::vector<std::int64_t> {
    std::vector<std::int64_t> d;
    d.reserve(static_cast<std::size_t>(k));
    for (std::int64_t v = 0; v < k; ++v) {
        d.push_back(v);
    }
    return d;
}

// A "not equal" binary constraint between variables i and j.
[[nodiscard]] auto ne(std::size_t i, std::size_t j) -> BinaryConstraint {
    return BinaryConstraint{i, j, [](std::int64_t a, std::int64_t b) { return a != b; }};
}

// N-queens as a CSP: variable i is the row of the queen in column i (so "no two in the same
// column" is structural). For each pair of columns i < j we forbid the same row and the two
// diagonals. Standard solution counts: 4 -> 2, 8 -> 92.
[[nodiscard]] auto queens(std::int64_t n) -> Csp {
    Csp csp;
    const auto sz = static_cast<std::size_t>(n);
    csp.domains.assign(sz, range_domain(n));
    for (std::size_t i = 0; i < sz; ++i) {
        for (std::size_t j = i + 1; j < sz; ++j) {
            const std::int64_t d = static_cast<std::int64_t>(j) - static_cast<std::int64_t>(i);
            csp.binary.push_back(BinaryConstraint{
                i, j, [d](std::int64_t a, std::int64_t b) {
                    return a != b && (a - b != d) && (b - a != d);  // no same row, no diagonal
                }});
        }
    }
    return csp;
}

// The classic Australia map-colouring instance (7 regions, 3 colours). Regions:
// 0=WA 1=NT 2=SA 3=Q 4=NSW 5=V 6=T. T is isolated (no adjacency).
[[nodiscard]] auto australia() -> Csp {
    Csp csp;
    csp.domains.assign(7, range_domain(3));  // colours {0, 1, 2}
    const std::pair<std::size_t, std::size_t> edges[] = {
        {0, 1}, {0, 2}, {1, 2}, {1, 3}, {2, 3}, {2, 4}, {2, 5}, {3, 4}, {4, 5}};
    for (const auto& [a, b] : edges) {
        csp.binary.push_back(ne(a, b));
    }
    return csp;
}

// Confirms an assignment respects every adjacency (used to validate a colouring).
[[nodiscard]] auto colouring_ok(const std::vector<std::int64_t>& c) -> bool {
    const std::pair<std::size_t, std::size_t> edges[] = {
        {0, 1}, {0, 2}, {1, 2}, {1, 3}, {2, 3}, {2, 4}, {2, 5}, {3, 4}, {4, 5}};
    for (const auto& [a, b] : edges) {
        if (c[a] == c[b]) {
            return false;
        }
    }
    return true;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.csp")
        .test("four_queens_solution_count",
              [](TestContext& t) {
                  auto n = solution_count(queens(4), 0);
                  t.expect(n.has_value(), "count succeeds");
                  t.expect(n.value_or(0) == 2, "4-queens has exactly 2 solutions");
              })
        .test("eight_queens_solution_count",
              [](TestContext& t) {
                  auto n = solution_count(queens(8), 0);
                  t.expect(n.has_value(), "count succeeds");
                  t.expect(n.value_or(0) == 92, "8-queens has exactly 92 solutions");
              })
        .test("solution_count_limit_caps",
              [](TestContext& t) {
                  // With a cap of 10 the count stops early at exactly the limit.
                  auto n = solution_count(queens(8), 10);
                  t.expect(n.has_value(), "capped count succeeds");
                  t.expect(n.value_or(0) == 10, "limit caps the count at 10");
              })
        .test("four_queens_first_solution_valid",
              [](TestContext& t) {
                  auto s = backtracking_search(queens(4));
                  t.expect(s.has_value(), "search succeeds");
                  t.expect(s.value_or(std::nullopt).has_value(), "4-queens is satisfiable");
                  if (s && *s) {
                      const auto& a = **s;
                      // Lexicographically-first 4-queens solution is rows [1, 3, 0, 2].
                      t.expect(a == std::vector<std::int64_t>({1, 3, 0, 2}),
                               "returns the lexicographically-first placement [1,3,0,2]");
                  }
              })
        .test("australia_colouring_satisfies_adjacency",
              [](TestContext& t) {
                  auto s = backtracking_search(australia());
                  t.expect(s.has_value(), "search succeeds");
                  t.expect(s.value_or(std::nullopt).has_value(), "Australia is 3-colourable");
                  if (s && *s) {
                      t.expect(colouring_ok(**s), "returned colouring satisfies every adjacency");
                  }
              })
        .test("forward_checking_matches_plain_backtracking",
              [](TestContext& t) {
                  // The FC variant must return the SAME lexicographically-first solution.
                  auto plain = backtracking_search(australia());
                  auto fc = backtracking_search_fc(australia());
                  t.expect(plain.has_value() && fc.has_value(), "both searches succeed");
                  t.expect(plain.value_or(std::nullopt) == fc.value_or(std::nullopt),
                           "forward checking returns the identical first solution");
                  // And on queens too.
                  auto pq = backtracking_search(queens(6));
                  auto fq = backtracking_search_fc(queens(6));
                  t.expect(pq.value_or(std::nullopt) == fq.value_or(std::nullopt),
                           "FC matches plain backtracking on 6-queens");
              })
        .test("ac3_prunes_a_value",
              [](TestContext& t) {
                  // X, Y in {1,2,3} with X < Y: arc consistency removes 3 from X and 1 from Y.
                  Csp csp;
                  csp.domains = {{1, 2, 3}, {1, 2, 3}};
                  csp.binary.push_back(BinaryConstraint{
                      0, 1, [](std::int64_t a, std::int64_t b) { return a < b; }});
                  auto r = ac3(csp);
                  t.expect(r.has_value(), "ac3 succeeds");
                  t.expect(r.value_or(std::nullopt).has_value(), "domains stay non-empty");
                  if (r && *r) {
                      const auto& d = **r;
                      t.expect(d[0] == std::vector<std::int64_t>({1, 2}), "X pruned to {1,2}");
                      t.expect(d[1] == std::vector<std::int64_t>({2, 3}), "Y pruned to {2,3}");
                  }
              })
        .test("ac3_detects_inconsistency",
              [](TestContext& t) {
                  // X, Y in {0} with X != Y: X has no support, a domain empties -> nullopt.
                  Csp csp;
                  csp.domains = {{0}, {0}};
                  csp.binary.push_back(ne(0, 1));
                  auto r = ac3(csp);
                  t.expect(r.has_value(), "ac3 returns a value (empty domain is not an error)");
                  t.expect(!r.value_or(std::nullopt).has_value(),
                           "arc-inconsistent CSP yields nullopt");
              })
        .test("ac3_two_constraints_same_pair_stays_consistent",
              [](TestContext& t) {
                  // Regression: two binary constraints on the SAME ordered pair (0, 1). AC-3's
                  // "skip the reverse arc" re-queue optimisation is sound only PER CONSTRAINT; a
                  // per-variable exclusion wrongly suppresses the reverse arc of the OTHER
                  // constraint and can leave a non-arc-consistent domain.
                  //   X, Y in {0,1,2};  C0: X == Y ;  C1: X != 2.
                  // C1 removes 2 from D[X]; that then removes support for value 2 in D[Y] under
                  // C0 (nothing in {0,1} equals 2). The arc-consistent fixpoint is
                  //   D[X] = {0,1},  D[Y] = {0,1}.
                  // The old per-variable exclusion instead left D[Y] = {0,1,2} (value 2 with no
                  // C0 support) -- arc-inconsistent.
                  Csp csp;
                  csp.domains = {{0, 1, 2}, {0, 1, 2}};
                  csp.binary.push_back(BinaryConstraint{
                      0, 1, [](std::int64_t a, std::int64_t b) { return a == b; }});
                  csp.binary.push_back(BinaryConstraint{
                      0, 1, [](std::int64_t a, std::int64_t /*b*/) { return a != 2; }});
                  auto r = ac3(csp);
                  t.expect(r.has_value() && r.value_or(std::nullopt).has_value(),
                           "ac3 succeeds with non-empty domains");
                  if (r && *r) {
                      const auto& d = **r;
                      t.expect(d[0] == std::vector<std::int64_t>({0, 1}), "D[X] pruned to {0,1}");
                      t.expect(d[1] == std::vector<std::int64_t>({0, 1}),
                               "D[Y] pruned to {0,1} (value 2 has no support -> arc consistent)");
                  }
              })
        .test("unsatisfiable_returns_nullopt",
              [](TestContext& t) {
                  // Three mutually-distinct variables over a two-value domain: no solution.
                  Csp csp;
                  csp.domains = {{0, 1}, {0, 1}, {0, 1}};
                  csp.binary.push_back(ne(0, 1));
                  csp.binary.push_back(ne(0, 2));
                  csp.binary.push_back(ne(1, 2));
                  auto s = backtracking_search(csp);
                  t.expect(s.has_value(), "search succeeds (unsatisfiable is not an error)");
                  t.expect(!s.value_or(std::nullopt).has_value(), "no assignment exists -> nullopt");
                  auto n = solution_count(csp, 0);
                  t.expect(n.value_or(1) == 0, "solution_count is 0 for the unsatisfiable CSP");
                  // AC-3 alone cannot detect this: != stays arc consistent over a size-2 domain.
                  auto r = ac3(csp);
                  t.expect(r.value_or(std::nullopt).has_value(),
                           "AC-3 keeps the (still unsatisfiable) CSP arc consistent");
              })
        .test("general_constraint_all_different",
              [](TestContext& t) {
                  // A single ternary all-different over {0,1} is unsatisfiable (pigeonhole),
                  // exercising the general n-ary constraint path.
                  Csp csp;
                  csp.domains = {{0, 1}, {0, 1}, {0, 1}};
                  csp.general.push_back(Constraint{
                      {0, 1, 2}, [](std::span<const std::int64_t> v) {
                          return v[0] != v[1] && v[0] != v[2] && v[1] != v[2];
                      }});
                  auto s = backtracking_search(csp);
                  t.expect(s.has_value(), "search succeeds");
                  t.expect(!s.value_or(std::nullopt).has_value(),
                           "all-different over {0,1}^3 is unsatisfiable");

                  // Widen the domain to {0,1,2}: now the general constraint is satisfiable and
                  // FC must agree, both returning the valid lexicographic-first [0,1,2].
                  csp.domains = {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}};
                  auto s2 = backtracking_search(csp);
                  auto f2 = backtracking_search_fc(csp);
                  t.expect(s2.value_or(std::nullopt).has_value(), "widened CSP is satisfiable");
                  if (s2 && *s2) {
                      t.expect(**s2 == std::vector<std::int64_t>({0, 1, 2}),
                               "returns lexicographic-first all-different [0,1,2]");
                  }
                  t.expect(s2.value_or(std::nullopt) == f2.value_or(std::nullopt),
                           "FC agrees on the general-constraint CSP");
              })
        .test("parallel_search_agrees_with_backtracking",
              [](TestContext& t) {
                  // Same result regardless of the (thread-count-independent) branch split.
                  auto seq = backtracking_search(australia());
                  auto par = parallel_search(australia());
                  t.expect(seq.has_value() && par.has_value(), "both searches succeed");
                  t.expect(seq.value_or(std::nullopt) == par.value_or(std::nullopt),
                           "parallel_search matches backtracking_search on Australia");

                  auto sq = backtracking_search(queens(6));
                  auto pq = parallel_search(queens(6));
                  t.expect(sq.value_or(std::nullopt) == pq.value_or(std::nullopt),
                           "parallel_search matches backtracking_search on 6-queens");

                  // And it still agrees on an unsatisfiable instance (both nullopt).
                  Csp bad;
                  bad.domains = {{0, 1}, {0, 1}, {0, 1}};
                  bad.binary.push_back(ne(0, 1));
                  bad.binary.push_back(ne(0, 2));
                  bad.binary.push_back(ne(1, 2));
                  auto pbad = parallel_search(bad);
                  t.expect(pbad.has_value() && !pbad.value_or(std::nullopt).has_value(),
                           "parallel_search returns nullopt for the unsatisfiable CSP");
              })
        .test("domain_error_out_of_range_scope",
              [](TestContext& t) {
                  // Binary constraint references variable 2, but only 2 variables exist.
                  Csp csp;
                  csp.domains = {{0, 1}, {0, 1}};
                  csp.binary.push_back(ne(0, 2));
                  auto b = backtracking_search(csp);
                  t.expect(!b.has_value() && b.error() == MathError::domain_error,
                           "out-of-range binary scope -> domain_error");
                  // Same fault surfaces through AC-3.
                  auto a = ac3(csp);
                  t.expect(!a.has_value() && a.error() == MathError::domain_error,
                           "AC-3 also rejects the out-of-range scope");
                  // And through a general constraint scope.
                  Csp csp2;
                  csp2.domains = {{0, 1}, {0, 1}};
                  csp2.general.push_back(Constraint{
                      {0, 5}, [](std::span<const std::int64_t> v) { return v[0] == v[1]; }});
                  auto g = solution_count(csp2, 0);
                  t.expect(!g.has_value() && g.error() == MathError::domain_error,
                           "out-of-range general scope -> domain_error");
              })
        .test("domain_error_empty_domain_and_empty_csp",
              [](TestContext& t) {
                  // An empty initial domain is a shape fault.
                  Csp csp;
                  csp.domains = {{0, 1}, {}};
                  auto b = backtracking_search(csp);
                  t.expect(!b.has_value() && b.error() == MathError::domain_error,
                           "empty initial domain -> domain_error");
                  // An empty variable set is a shape fault.
                  Csp empty;
                  auto e = solution_count(empty, 0);
                  t.expect(!e.has_value() && e.error() == MathError::domain_error,
                           "empty variable set -> domain_error");
                  auto pe = parallel_search(empty);
                  t.expect(!pe.has_value() && pe.error() == MathError::domain_error,
                           "parallel_search rejects the empty CSP too");
              })
        .run();
}
