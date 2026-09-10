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
using nimblecas::as_csp;
using nimblecas::ConstraintKind;
using nimblecas::Csp;
using nimblecas::MathError;
using nimblecas::holds;
using nimblecas::parallel_search;
using nimblecas::prefix_assignment;
using nimblecas::prefix_count;
using nimblecas::restrict_prefix;
using nimblecas::validate;
using nimblecas::WireConstraint;
using nimblecas::WireCsp;
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

// The same n-queens problem as `queens` above, expressed declaratively: the rows are
// pairwise distinct (all_different), and no two queens share a diagonal, which is exactly
// |row_i - row_j| != (j - i) for every pair -- the abs_diff_ne kind.
[[nodiscard]] auto wire_queens(std::int64_t n) -> WireCsp {
    WireCsp w;
    const auto un = static_cast<std::size_t>(n);
    for (std::size_t i = 0; i < un; ++i) {
        w.domains.push_back(range_domain(n));
    }
    std::vector<std::size_t> all;
    all.reserve(un);
    for (std::size_t i = 0; i < un; ++i) {
        all.push_back(i);
    }
    w.constraints.push_back(WireConstraint{
        .kind = ConstraintKind::all_different, .scope = std::move(all), .params = {}});
    for (std::size_t i = 0; i < un; ++i) {
        for (std::size_t j = i + 1; j < un; ++j) {
            w.constraints.push_back(
                WireConstraint{.kind = ConstraintKind::abs_diff_ne,
                               .scope = {i, j},
                               .params = {static_cast<std::int64_t>(j - i)}});
        }
    }
    return w;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.csp")
        .test("four_queens_solution_count",
              [](TestContext& t) {
                  const auto n = solution_count(queens(4), 0);
                  t.expect(n.has_value(), "count succeeds");
                  t.expect(n.value_or(0) == 2, "4-queens has exactly 2 solutions");
              })
        .test("eight_queens_solution_count",
              [](TestContext& t) {
                  const auto n = solution_count(queens(8), 0);
                  t.expect(n.has_value(), "count succeeds");
                  t.expect(n.value_or(0) == 92, "8-queens has exactly 92 solutions");
              })
        .test("solution_count_limit_caps",
              [](TestContext& t) {
                  // With a cap of 10 the count stops early at exactly the limit.
                  const auto n = solution_count(queens(8), 10);
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
                  const auto plain = backtracking_search(australia());
                  const auto fc = backtracking_search_fc(australia());
                  t.expect(plain.has_value() && fc.has_value(), "both searches succeed");
                  t.expect(plain.value_or(std::nullopt) == fc.value_or(std::nullopt),
                           "forward checking returns the identical first solution");
                  // And on queens too.
                  const auto pq = backtracking_search(queens(6));
                  const auto fq = backtracking_search_fc(queens(6));
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
                  const auto r = ac3(csp);
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
                  const auto s = backtracking_search(csp);
                  t.expect(s.has_value(), "search succeeds (unsatisfiable is not an error)");
                  t.expect(!s.value_or(std::nullopt).has_value(), "no assignment exists -> nullopt");
                  const auto n = solution_count(csp, 0);
                  t.expect(n.value_or(1) == 0, "solution_count is 0 for the unsatisfiable CSP");
                  // AC-3 alone cannot detect this: != stays arc consistent over a size-2 domain.
                  const auto r = ac3(csp);
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
                  const auto s = backtracking_search(csp);
                  t.expect(s.has_value(), "search succeeds");
                  t.expect(!s.value_or(std::nullopt).has_value(),
                           "all-different over {0,1}^3 is unsatisfiable");

                  // Widen the domain to {0,1,2}: now the general constraint is satisfiable and
                  // FC must agree, both returning the valid lexicographic-first [0,1,2].
                  csp.domains = {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}};
                  auto s2 = backtracking_search(csp);
                  const auto f2 = backtracking_search_fc(csp);
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
                  const auto seq = backtracking_search(australia());
                  const auto par = parallel_search(australia());
                  t.expect(seq.has_value() && par.has_value(), "both searches succeed");
                  t.expect(seq.value_or(std::nullopt) == par.value_or(std::nullopt),
                           "parallel_search matches backtracking_search on Australia");

                  const auto sq = backtracking_search(queens(6));
                  const auto pq = parallel_search(queens(6));
                  t.expect(sq.value_or(std::nullopt) == pq.value_or(std::nullopt),
                           "parallel_search matches backtracking_search on 6-queens");

                  // And it still agrees on an unsatisfiable instance (both nullopt).
                  Csp bad;
                  bad.domains = {{0, 1}, {0, 1}, {0, 1}};
                  bad.binary.push_back(ne(0, 1));
                  bad.binary.push_back(ne(0, 2));
                  bad.binary.push_back(ne(1, 2));
                  const auto pbad = parallel_search(bad);
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
        .test("wire_queens_denotes_the_same_problem_as_the_functional_queens",
              [](TestContext& t) {
                  // The whole point of the declarative layer: the SAME problem, expressed as
                  // data instead of closures, must have the same solutions. 8-queens is the
                  // right check because its count is a fixed, published number.
                  auto w = wire_queens(8);
                  auto c = as_csp(w);
                  t.expect(c.has_value(), "the wire form converts to a functional Csp");
                  if (!c) {
                      return;
                  }
                  const auto n = solution_count(*c, 0);
                  t.expect(n.has_value(), "count succeeds");
                  t.expect(n.value_or(0) == 92,
                           "the declarative 8-queens has the same 92 solutions");
                  // And the first solution agrees with the functional encoding exactly, not
                  // merely in count -- both contracts promise the lexicographically-first one.
                  const auto a = backtracking_search(*c);
                  const auto b = backtracking_search(queens(8));
                  t.expect(a.has_value() && b.has_value(), "both searches succeed");
                  t.expect(a.value_or(std::nullopt) == b.value_or(std::nullopt),
                           "and return the identical first assignment");
              })
        .test("all_different_expands_to_binary_constraints_so_ac3_can_prune",
              [](TestContext& t) {
                  // all_different IS pairwise not-equal, and AC-3 propagates over binary
                  // constraints only. Leaving it as one general constraint would be
                  // semantically identical and strictly weaker, so the expansion is checked.
                  WireCsp w;
                  w.domains = {range_domain(4), range_domain(4), range_domain(4)};
                  w.constraints.push_back(
                      WireConstraint{.kind = ConstraintKind::all_different,
                                     .scope = {0, 1, 2},
                                     .params = {}});
                  auto c = as_csp(w);
                  t.expect(c.has_value(), "conversion succeeds");
                  if (!c) {
                      return;
                  }
                  t.expect(c->binary.size() == 3, "3 variables give 3 pairwise not-equals");
                  t.expect(c->general.empty(), "and nothing is left in the general list");
              })
        .test("holds_evaluates_each_constraint_kind",
              [](TestContext& t) {
                  const auto ck = [&t](ConstraintKind kind, std::vector<std::size_t> scope,
                                       std::vector<std::int64_t> params,
                                       std::vector<std::int64_t> values, bool want,
                                       const char* why) {
                      const WireConstraint c{
                          .kind = kind, .scope = std::move(scope), .params = std::move(params)};
                      t.expect(holds(c, std::span<const std::int64_t>(values)) == want, why);
                  };
                  ck(ConstraintKind::not_equal, {0, 1}, {}, {1, 2}, true, "1 != 2");
                  ck(ConstraintKind::not_equal, {0, 1}, {}, {2, 2}, false, "2 != 2 is false");
                  ck(ConstraintKind::equal, {0, 1}, {}, {5, 5}, true, "5 == 5");
                  // less_equal is x + k <= y.
                  ck(ConstraintKind::less_equal, {0, 1}, {3}, {1, 4}, true, "1 + 3 <= 4");
                  ck(ConstraintKind::less_equal, {0, 1}, {3}, {2, 4}, false, "2 + 3 <= 4 is false");
                  ck(ConstraintKind::abs_diff_ne, {0, 1}, {2}, {1, 3}, false, "|1-3| != 2 is false");
                  ck(ConstraintKind::abs_diff_ne, {0, 1}, {2}, {1, 4}, true, "|1-4| != 2");
                  ck(ConstraintKind::all_different, {0, 1, 2}, {}, {1, 2, 3}, true, "all distinct");
                  ck(ConstraintKind::all_different, {0, 1, 2}, {}, {1, 2, 1}, false, "a repeat");
                  // 2x + 3y == 12.
                  ck(ConstraintKind::linear_eq, {0, 1}, {2, 3, 12}, {3, 2}, true, "2*3 + 3*2 == 12");
                  ck(ConstraintKind::linear_eq, {0, 1}, {2, 3, 12}, {3, 3}, false, "15 != 12");
                  ck(ConstraintKind::linear_le, {0, 1}, {2, 3, 12}, {1, 1}, true, "5 <= 12");
                  ck(ConstraintKind::table_allowed, {0, 1}, {1, 2, 3, 4}, {3, 4}, true,
                     "the tuple (3,4) is a row of the table");
                  ck(ConstraintKind::table_allowed, {0, 1}, {1, 2, 3, 4}, {1, 4}, false,
                     "the tuple (1,4) is not a row of the table");
              })
        .test("validate_names_every_shape_fault",
              [](TestContext& t) {
                  const auto bad = [&t](const WireCsp& w, MathError want, const char* why) {
                      auto r = validate(w);
                      t.expect(!r.has_value() && r.error() == want, why);
                  };
                  WireCsp base;
                  base.domains = {range_domain(3), range_domain(3)};

                  WireCsp empty_vars;
                  bad(empty_vars, MathError::domain_error, "an empty variable set is refused");

                  WireCsp empty_dom;
                  empty_dom.domains = {{}};
                  bad(empty_dom, MathError::domain_error, "an empty domain is refused");

                  auto out_of_range = base;
                  out_of_range.constraints.push_back(WireConstraint{
                      .kind = ConstraintKind::not_equal, .scope = {0, 7}, .params = {}});
                  bad(out_of_range, MathError::domain_error, "an out-of-range scope is refused");

                  auto dup = base;
                  dup.constraints.push_back(WireConstraint{
                      .kind = ConstraintKind::not_equal, .scope = {1, 1}, .params = {}});
                  bad(dup, MathError::domain_error,
                      "a variable repeated in one scope is refused, not given a reading");

                  auto wrong_arity = base;
                  wrong_arity.constraints.push_back(WireConstraint{
                      .kind = ConstraintKind::not_equal, .scope = {0, 1}, .params = {4}});
                  bad(wrong_arity, MathError::domain_error, "not_equal takes no parameters");

                  auto missing_param = base;
                  missing_param.constraints.push_back(WireConstraint{
                      .kind = ConstraintKind::abs_diff_ne, .scope = {0, 1}, .params = {}});
                  bad(missing_param, MathError::domain_error, "abs_diff_ne needs its one parameter");

                  auto ragged = base;
                  ragged.constraints.push_back(WireConstraint{
                      .kind = ConstraintKind::table_allowed, .scope = {0, 1}, .params = {1, 2, 3}});
                  bad(ragged, MathError::domain_error, "a ragged tuple table is refused");

                  auto short_linear = base;
                  short_linear.constraints.push_back(WireConstraint{
                      .kind = ConstraintKind::linear_eq, .scope = {0, 1}, .params = {1, 1}});
                  bad(short_linear, MathError::domain_error,
                      "a linear constraint needs one coefficient per variable plus a rhs");
              })
        .test("a_difference_constraint_that_could_overflow_is_refused_not_evaluated",
              [](TestContext& t) {
                  // less_equal and abs_diff_ne both evaluate values[0] - values[1], and both
                  // negate. Signed overflow there is undefined behaviour in the evaluator and
                  // silently wrong in emitted code, so the domains are bounded ONCE at
                  // validation rather than checked on every evaluation.
                  constexpr std::int64_t lo = std::numeric_limits<std::int64_t>::min();
                  constexpr std::int64_t hi = std::numeric_limits<std::int64_t>::max();

                  WireCsp wide;
                  wide.domains = {{hi - 1, hi}, {lo, lo + 1}};
                  wide.constraints.push_back(WireConstraint{
                      .kind = ConstraintKind::abs_diff_ne, .scope = {0, 1}, .params = {1}});
                  auto r = validate(wide);
                  t.expect(!r.has_value() && r.error() == MathError::overflow,
                           "a difference that cannot fit in int64 is an honest overflow");

                  // A parameter of INT64_MIN cannot be negated, which less_equal must do.
                  WireCsp bad_param;
                  bad_param.domains = {range_domain(3), range_domain(3)};
                  bad_param.constraints.push_back(WireConstraint{
                      .kind = ConstraintKind::less_equal, .scope = {0, 1}, .params = {lo}});
                  auto p = validate(bad_param);
                  t.expect(!p.has_value() && p.error() == MathError::overflow,
                           "a parameter whose negation is unrepresentable is refused");

                  // The case the first version of this guard missed: a difference of exactly
                  // INT64_MIN, reached with a non-positive upper bound on the second domain,
                  // so the "cannot be negated" test must not depend on that bound being
                  // positive.
                  WireCsp exact_min;
                  exact_min.domains = {{lo, lo + 1}, {0, 1}};
                  exact_min.constraints.push_back(WireConstraint{
                      .kind = ConstraintKind::abs_diff_ne, .scope = {0, 1}, .params = {1}});
                  auto m = validate(exact_min);
                  t.expect(!m.has_value() && m.error() == MathError::overflow,
                           "a difference of exactly INT64_MIN is refused, having no negation");

                  // Ordinary domains are of course accepted, and evaluate correctly.
                  WireCsp ok;
                  ok.domains = {range_domain(5), range_domain(5)};
                  ok.constraints.push_back(WireConstraint{
                      .kind = ConstraintKind::abs_diff_ne, .scope = {0, 1}, .params = {2}});
                  t.expect(validate(ok).has_value(), "a bounded difference constraint is accepted");
              })
        .test("a_linear_constraint_that_could_overflow_is_refused_not_evaluated",
              [](TestContext& t) {
                  // The evaluator adds coef*value with plain int64. Rather than check for
                  // overflow on every evaluation -- in a search inner loop, and in emitted
                  // code with nowhere to report it -- the worst case is bounded once here.
                  // A constraint that could overflow must be REFUSED, never silently wrong.
                  constexpr std::int64_t huge = std::numeric_limits<std::int64_t>::max() / 2;
                  WireCsp w;
                  w.domains = {{0, huge}, {0, huge}};
                  w.constraints.push_back(WireConstraint{.kind = ConstraintKind::linear_eq,
                                                         .scope = {0, 1},
                                                         .params = {3, 3, 0}});
                  auto r = validate(w);
                  t.expect(!r.has_value() && r.error() == MathError::overflow,
                           "a weighted sum that cannot fit in int64 is an honest overflow");

                  // A large right-hand side is NOT a reason to refuse. `holds` compares the
                  // finished sum against it and never forms `sum - rhs`, so the rhs cannot
                  // overflow anything; refusing here would be a false refusal.
                  WireCsp big_rhs;
                  big_rhs.domains = {range_domain(4), range_domain(4)};
                  big_rhs.constraints.push_back(
                      WireConstraint{.kind = ConstraintKind::linear_le,
                                     .scope = {0, 1},
                                     .params = {1, 1, std::numeric_limits<std::int64_t>::max()}});
                  t.expect(validate(big_rhs).has_value(),
                           "a bounded sum against a huge rhs is accepted, not refused");

                  // The same shape with domains that fit is accepted.
                  WireCsp ok;
                  ok.domains = {range_domain(10), range_domain(10)};
                  ok.constraints.push_back(WireConstraint{.kind = ConstraintKind::linear_eq,
                                                          .scope = {0, 1},
                                                          .params = {3, 3, 12}});
                  t.expect(validate(ok).has_value(), "and a bounded one is accepted");
              })
        .test("the_prefix_split_partitions_the_assignment_space_exactly",
              [](TestContext& t) {
                  // This is the property distribution rests on. If the sub-problems did not
                  // partition the space, a distributed solve could double-count a solution or
                  // miss one, and an UNSAT verdict merged from them would be worthless.
                  // Checked by counting: the sub-counts must sum to the whole count exactly.
                  auto w = wire_queens(6);
                  auto whole = as_csp(w);
                  t.expect(whole.has_value(), "the 6-queens wire form converts");
                  if (!whole) {
                      return;
                  }
                  const auto total = solution_count(*whole, 0);
                  t.expect(total.has_value() && total.value_or(0) == 4,
                           "6-queens has exactly 4 solutions");

                  constexpr std::size_t fixed_vars = 2;
                  auto count = prefix_count(w, fixed_vars);
                  t.expect(count.has_value(), "the prefix count is computable");
                  t.expect(count.value_or(0) == 36, "6 x 6 = 36 prefixes over the first 2 columns");
                  if (!count) {
                      return;
                  }
                  std::uint64_t summed = 0;
                  bool all_ok = true;
                  for (std::uint64_t i = 0; i < *count; ++i) {
                      auto prefix = prefix_assignment(w, fixed_vars, i);
                      if (!prefix) {
                          all_ok = false;
                          break;
                      }
                      auto sub = restrict_prefix(w, std::span<const std::int64_t>(*prefix));
                      if (!sub) {
                          all_ok = false;
                          break;
                      }
                      auto sub_csp = as_csp(*sub);
                      if (!sub_csp) {
                          all_ok = false;
                          break;
                      }
                      auto n = solution_count(*sub_csp, 0);
                      if (!n) {
                          all_ok = false;
                          break;
                      }
                      summed += *n;
                  }
                  t.expect(all_ok, "every sub-problem is well formed and solvable");
                  t.expect(summed == total.value_or(0),
                           "the sub-counts sum to the whole count: no solution double-counted, "
                           "none missed");
              })
        .test("prefix_assignment_enumerates_in_ascending_lexicographic_order",
              [](TestContext& t) {
                  // Ascending index must mean ascending assignment, because that is what makes
                  // "the lowest-indexed shard wins" the same answer the serial search gives.
                  WireCsp w;
                  w.domains = {{10, 20}, {5, 6, 7}, range_domain(4)};
                  const auto count = prefix_count(w, 2);
                  t.expect(count.value_or(0) == 6, "2 x 3 = 6 prefixes");
                  std::vector<std::vector<std::int64_t>> seen;
                  for (std::uint64_t i = 0; i < count.value_or(0); ++i) {
                      auto p = prefix_assignment(w, 2, i);
                      t.expect(p.has_value(), "each index yields an assignment");
                      if (p) {
                          seen.push_back(*p);
                      }
                  }
                  const std::vector<std::vector<std::int64_t>> want{
                      {10, 5}, {10, 6}, {10, 7}, {20, 5}, {20, 6}, {20, 7}};
                  t.expect(seen == want, "the enumeration is exactly ascending lexicographic");
                  auto past = prefix_assignment(w, 2, 6);
                  t.expect(!past.has_value() && past.error() == MathError::domain_error,
                           "an index past the end is a domain_error, not a wrapped value");
              })
        .test("restrict_prefix_refuses_a_value_outside_the_domain",
              [](TestContext& t) {
                  WireCsp w;
                  w.domains = {{1, 2, 3}, {4, 5}};
                  const std::vector<std::int64_t> not_in_domain{9};
                  auto r = restrict_prefix(w, std::span<const std::int64_t>(not_in_domain));
                  t.expect(!r.has_value() && r.error() == MathError::domain_error,
                           "fixing a variable to a value it cannot take is a domain_error");
                  const std::vector<std::int64_t> good{2};
                  auto ok = restrict_prefix(w, std::span<const std::int64_t>(good));
                  t.expect(ok.has_value(), "a legal prefix restricts");
                  t.expect(ok.has_value() && ok->domains[0].size() == 1 && ok->domains[0][0] == 2,
                           "and collapses that domain to the single fixed value");
                  t.expect(ok.has_value() && ok->domains[1].size() == 2,
                           "leaving the unfixed domains untouched");
              })
        .run();
}
