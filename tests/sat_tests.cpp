// Tests for nimblecas.sat: complete (DPLL, CDCL) and stochastic-local-search (WalkSAT, GSAT)
// engines, plus the CPU portfolio and stateless distributed shards.
// @author Olumuyiwa Oluwasanmi
//
// Every satisfiable outcome is re-checked with verify_assignment so a bogus model can never
// pass silently; the local-search engines are only ever asserted to return satisfiable-or-
// unknown (never unsatisfiable); and the complete engines are cross-checked for verdict
// agreement on shared instances. All local-search calls use fixed seeds, so the suite is
// deterministic.

import std;
import nimblecas.core;
import nimblecas.sat;
import nimblecas.testing;

using nimblecas::Cnf;
using nimblecas::MathError;
using nimblecas::SatResult;
using nimblecas::SatVerdict;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// A satisfiable 3-SAT-ish instance whose all-true assignment is a model (every clause has a
// positive literal), giving the local-search engines a clear basin to descend into.
[[nodiscard]] auto easy_sat() -> Cnf {
    return Cnf{5, {{1, -2, 3}, {2, -3, 4}, {3, -4, 5}, {1, 4, 5}, {-1, 2, 5}}};
}

// A small satisfiable 3-SAT instance for the complete solvers.
[[nodiscard]] auto small_sat() -> Cnf {
    return Cnf{3, {{1, 2}, {-1, 3}, {-2, -3}}};
}

// (x) ∧ (¬x): the minimal unsatisfiable formula.
[[nodiscard]] auto trivial_unsat() -> Cnf {
    return Cnf{1, {{1}, {-1}}};
}

// Pigeonhole PHP(2 pigeons -> 1 hole): variable v (1,2) is "pigeon v sits in the only hole".
// Each pigeon must be placed ((1), (2)); the two pigeons cannot share the hole ((¬1 ∨ ¬2)).
// Unsatisfiable.
[[nodiscard]] auto php_2_1() -> Cnf {
    return Cnf{2, {{1}, {2}, {-1, -2}}};
}

// Deterministic replica of the portfolio/shard merge rule, for the distributed-shard test.
[[nodiscard]] auto merge(const std::vector<SatResult>& shards) -> SatVerdict {
    for (const auto& s : shards) {
        if (s.verdict == SatVerdict::unsatisfiable) {
            return SatVerdict::unsatisfiable;
        }
    }
    for (const auto& s : shards) {
        if (s.verdict == SatVerdict::satisfiable) {
            return SatVerdict::satisfiable;
        }
    }
    return SatVerdict::unknown;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.sat")
        .test("verify_assignment_true_and_false",
              [](TestContext& t) {
                  const Cnf cnf{2, {{1, 2}, {-1, 2}}};
                  t.expect(nimblecas::verify_assignment(cnf, {true, true}),
                           "[T,T] satisfies (1∨2)∧(¬1∨2)");
                  t.expect(!nimblecas::verify_assignment(cnf, {true, false}),
                           "[T,F] falsifies (¬1∨2)");
                  // An empty clause can never be satisfied.
                  t.expect(!nimblecas::verify_assignment(Cnf{1, {{}}}, {false}),
                           "empty clause is unsatisfiable");
                  // A model of the wrong size cannot be verified.
                  t.expect(!nimblecas::verify_assignment(cnf, {true}),
                           "wrong-size model is rejected");
              })
        .test("dpll_satisfiable_returns_verified_model",
              [](TestContext& t) {
                  const Cnf cnf = small_sat();
                  auto r = nimblecas::dpll(cnf);
                  t.expect(r.has_value(), "dpll succeeds on a well-formed formula");
                  if (r) {
                      t.expect(r->verdict == SatVerdict::satisfiable, "small_sat is satisfiable");
                      t.expect(r->model.size() == cnf.num_vars, "model has num_vars entries");
                      t.expect(nimblecas::verify_assignment(cnf, r->model),
                               "returned model actually satisfies the formula");
                  }
              })
        .test("trivial_unsat_via_dpll_and_cdcl",
              [](TestContext& t) {
                  const Cnf cnf = trivial_unsat();
                  auto d = nimblecas::dpll(cnf);
                  auto c = nimblecas::cdcl(cnf, 100000);
                  t.expect(d.has_value() && d->verdict == SatVerdict::unsatisfiable,
                           "dpll: (x)∧(¬x) is UNSAT");
                  t.expect(c.has_value() && c->verdict == SatVerdict::unsatisfiable,
                           "cdcl: (x)∧(¬x) is UNSAT");
              })
        .test("pigeonhole_php_2_1_unsat",
              [](TestContext& t) {
                  const Cnf cnf = php_2_1();
                  auto d = nimblecas::dpll(cnf);
                  auto c = nimblecas::cdcl(cnf, 100000);
                  t.expect(d.has_value() && d->verdict == SatVerdict::unsatisfiable,
                           "dpll: PHP(2->1) is UNSAT");
                  t.expect(c.has_value() && c->verdict == SatVerdict::unsatisfiable,
                           "cdcl: PHP(2->1) is UNSAT");
              })
        .test("walksat_finds_and_verifies_a_model",
              [](TestContext& t) {
                  const Cnf cnf = easy_sat();
                  auto r = nimblecas::walksat(cnf, 200000, 0.4, 12345);
                  t.expect(r.has_value(), "walksat succeeds");
                  if (r) {
                      t.expect(r->verdict == SatVerdict::satisfiable,
                               "walksat locates a model within the flip budget");
                      t.expect(nimblecas::verify_assignment(cnf, r->model),
                               "walksat model satisfies the formula");
                  }
              })
        .test("walksat_never_reports_unsat",
              [](TestContext& t) {
                  // SLS cannot prove UNSAT: on an unsatisfiable formula it must give up with
                  // `unknown`, never claim `unsatisfiable`.
                  auto r = nimblecas::walksat(php_2_1(), 5000, 0.5, 999);
                  t.expect(r.has_value(), "walksat succeeds on a well-formed UNSAT formula");
                  t.expect(r && r->verdict == SatVerdict::unknown,
                           "walksat returns unknown, not unsatisfiable, on UNSAT input");
              })
        .test("gsat_finds_and_verifies_a_model",
              [](TestContext& t) {
                  const Cnf cnf = easy_sat();
                  auto r = nimblecas::gsat(cnf, 2000, 200, 7);
                  t.expect(r.has_value(), "gsat succeeds");
                  if (r) {
                      t.expect(r->verdict == SatVerdict::satisfiable,
                               "gsat locates a model within its restart budget");
                      t.expect(nimblecas::verify_assignment(cnf, r->model),
                               "gsat model satisfies the formula");
                  }
              })
        .test("dpll_and_cdcl_agree_on_verdict",
              [](TestContext& t) {
                  const std::vector<Cnf> instances{small_sat(), easy_sat(), trivial_unsat(),
                                                    php_2_1(), Cnf{2, {{1, 2}, {-1, -2}}}};
                  for (const auto& cnf : instances) {
                      auto d = nimblecas::dpll(cnf);
                      auto c = nimblecas::cdcl(cnf, 100000);
                      t.expect(d.has_value() && c.has_value(), "both solvers succeed");
                      if (d && c) {
                          t.expect(d->verdict == c->verdict, "dpll and cdcl agree on the verdict");
                      }
                  }
              })
        .test("portfolio_and_shards_agree_with_serial",
              [](TestContext& t) {
                  constexpr std::size_t workers = 4;
                  const std::vector<Cnf> instances{small_sat(), easy_sat(), trivial_unsat(),
                                                    php_2_1()};
                  for (const auto& cnf : instances) {
                      const auto serial = nimblecas::dpll(cnf);
                      const auto portfolio = nimblecas::solve_portfolio(cnf, 2024, workers);
                      t.expect(serial.has_value() && portfolio.has_value(),
                               "serial and portfolio both succeed");
                      if (serial && portfolio) {
                          t.expect(portfolio->verdict == serial->verdict,
                                   "portfolio verdict matches serial dpll");
                      }

                      // Merge the individual distributed shards with the associative rule and
                      // confirm it reproduces the serial verdict.
                      std::vector<SatResult> shards;
                      bool ok = true;
                      for (std::size_t s = 0; s < workers; ++s) {
                          auto sh = nimblecas::solve_shard(cnf, s, workers, 2024);
                          ok = ok && sh.has_value();
                          if (sh) {
                              shards.push_back(*sh);
                          }
                      }
                      t.expect(ok, "every shard succeeds");
                      if (ok && serial) {
                          t.expect(merge(shards) == serial->verdict,
                                   "merged shard verdict matches serial dpll");
                      }
                  }
              })
        .test("malformed_cnf_is_domain_error",
              [](TestContext& t) {
                  // num_vars == 0.
                  auto a = nimblecas::dpll(Cnf{0, {}});
                  t.expect(!a.has_value() && a.error() == MathError::domain_error,
                           "num_vars == 0 => domain_error");
                  // A literal referencing a variable beyond num_vars.
                  auto b = nimblecas::cdcl(Cnf{2, {{1, 3}}}, 1000);
                  t.expect(!b.has_value() && b.error() == MathError::domain_error,
                           "out-of-range literal => domain_error");
                  // The illegal literal 0.
                  auto c = nimblecas::dpll(Cnf{2, {{1, 0}}});
                  t.expect(!c.has_value() && c.error() == MathError::domain_error,
                           "literal 0 => domain_error");
                  // Noise outside [0, 1] for walksat.
                  auto d = nimblecas::walksat(small_sat(), 100, 1.5, 1);
                  t.expect(!d.has_value() && d.error() == MathError::domain_error,
                           "noise > 1 => domain_error");
                  // Zero workers / bad shard index.
                  auto e = nimblecas::solve_portfolio(small_sat(), 1, 0);
                  t.expect(!e.has_value() && e.error() == MathError::domain_error,
                           "workers == 0 => domain_error");
                  auto f = nimblecas::solve_shard(small_sat(), 3, 3, 1);
                  t.expect(!f.has_value() && f.error() == MathError::domain_error,
                           "shard_index >= num_shards => domain_error");
              })
        .run();
}
