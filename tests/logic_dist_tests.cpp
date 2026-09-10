// Tests for nimblecas.logic_dist: SLD resolution decomposed into a task graph and run over an
// Executor -- serial, local-parallel, or (unchanged) a distributed one over a broker.
// @author Olumuyiwa Oluwasanmi
//
// The contract under test is that the ANSWERS DO NOT REVEAL WHICH EXECUTOR RAN THEM: every
// distributed solve is compared against `solve` on the same program and query, for the same
// answers in the same order. The wire-format cases exist for the failures that would otherwise be
// silent -- a variable that loses its rename generation in transit reads back as a DIFFERENT
// variable, and nothing downstream would notice.

import std;
import nimblecas.core;
import nimblecas.logic;
import nimblecas.logic_dist;
import nimblecas.logic_parser;
import nimblecas.taskdag;
import nimblecas.taskdag_sgee;
import nimblecas.testing;

using nimblecas::apply_substitution;
using nimblecas::Clause;
using nimblecas::Executor;
using nimblecas::local_parallel_executor;
using nimblecas::make_atom;
using nimblecas::make_compound;
using nimblecas::make_int;
using nimblecas::make_list;
using nimblecas::make_var;
using nimblecas::MathError;
using nimblecas::Payload;
using nimblecas::Program;
using nimblecas::serial_executor;
using nimblecas::solve;
using nimblecas::solve_clause_branch;
using nimblecas::Substitution;
using nimblecas::TaskRegistry;
using nimblecas::FakeBrokerPort;
using nimblecas::InMemoryResultChannel;
using nimblecas::SgeeDistributedExecutor;
using nimblecas::SgeeExecutorConfig;
using nimblecas::Term;
using nimblecas::to_string;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// A program from Prolog source. A source error yields an EMPTY program rather than a throw, so a
// mistyped fixture shows up as a test that fails on its answers instead of one that aborts.
[[nodiscard]] auto prog(std::string_view src) -> Program {
    auto p = nimblecas::logic_parser::parse_program(src);
    if (!p) {
        return Program{};
    }
    return *p;
}

// A goal list from Prolog source, empty on a source error for the same reason.
[[nodiscard]] auto query(std::string_view src) -> std::vector<Term> {
    auto q = nimblecas::logic_parser::parse_query(src);
    if (!q) {
        return std::vector<Term>{};
    }
    return *q;
}

// The rendered binding of query variable `name` in `answer`, or "<unbound>" when the answer
// leaves it free. Comparing rendered ground terms is how every logic test in this repository
// pins a binding, so these tests read the same way as the ones next to them.
[[nodiscard]] auto bound(const Substitution& answer, std::string_view name) -> std::string {
    const Term v = make_var(std::string(name));
    const Term applied = apply_substitution(answer, v);
    if (nimblecas::is_var(applied) && nimblecas::var_of(applied).name == name) {
        return "<unbound>";
    }
    return to_string(applied);
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.logic_dist")
        .test("encode_decode_shard_roundtrip_basic_types",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  using namespace nimblecas::logic_dist;

                  Program prog_data;
                  // Fact: p(42, -17, min_i64).
                  const Term min_i64 = make_int(std::numeric_limits<std::int64_t>::min());
                  prog_data.push_back(Clause{
                      .head = make_compound("p", {make_int(42), make_int(-17), min_i64}),
                      .body = {},
                  });
                  // Rule: q(X) :- f(g(1, -42), h(a, b)), [1, 2, 3].
                  const Term nested_compound = make_compound(
                      "f", {make_compound("g", {make_int(1), make_int(-42)}),
                            make_compound("h", {make_atom("a"), make_atom("b")})});
                  const Term int_list = make_list({make_int(1), make_int(2), make_int(3)});
                  prog_data.push_back(Clause{
                      .head = make_compound("q", {make_var("X")}),
                      .body = {nested_compound, int_list},
                  });

                  std::vector<Term> goal_data;
                  goal_data.push_back(make_compound("p", {make_atom("query_atom"), make_int(100)}));
                  goal_data.push_back(int_list);

                  const Shard original{
                      .program = prog_data,
                      .goals = goal_data,
                      .clause_index = 1,
                      .max_solutions = 5,
                  };

                  const auto encoded = encode_shard(original);
                  t.expect(encoded.has_value(), "encode_shard succeeds on valid shard with basic types");
                  if (!encoded.has_value()) return;

                  const auto decoded = decode_shard(*encoded);
                  t.expect(decoded.has_value(), "decode_shard round-trips successfully");
                  if (!decoded.has_value()) return;

                  t.expect(decoded->clause_index == 1, "decoded clause_index matches original");
                  t.expect(decoded->max_solutions == 5, "decoded max_solutions matches original");

                  t.expect(decoded->goals.size() == 2, "decoded goals count matches original");
                  if (decoded->goals.size() >= 2) {
                      t.expect(decoded->goals[0] == goal_data[0], "goal 0 matches after round-trip");
                      t.expect(decoded->goals[1] == goal_data[1], "goal 1 matches after round-trip");
                  }

                  t.expect(decoded->program.size() == 2, "decoded program clause count matches original");
                  if (decoded->program.size() >= 2) {
                      t.expect(decoded->program[0].head == prog_data[0].head, "clause 0 head matches");
                      t.expect(decoded->program[0].body.empty(), "clause 0 body is empty fact");
                      t.expect(decoded->program[1].head == prog_data[1].head, "clause 1 head matches");
                      t.expect(decoded->program[1].body.size() == 2, "clause 1 body has 2 goals");
                      if (decoded->program[1].body.size() >= 2) {
                          t.expect(decoded->program[1].body[0] == nested_compound, "clause 1 body goal 0 matches");
                          t.expect(decoded->program[1].body[1] == int_list, "clause 1 body goal 1 matches");
                      }
                  }
              })
        .test("encode_decode_shard_atoms_needing_quotes",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  using namespace nimblecas::logic_dist;

                  const Term space_atom = make_atom("hello world");
                  const Term empty_atom = make_atom("");
                  const Term quote_atom = make_atom("it's");

                  Program prog_data;
                  prog_data.push_back(Clause{
                      .head = make_compound("test_quoted", {space_atom, empty_atom, quote_atom}),
                      .body = {},
                  });

                  std::vector<Term> goal_data;
                  goal_data.push_back(make_compound("test_quoted", {space_atom, empty_atom, quote_atom}));

                  const Shard original{
                      .program = prog_data,
                      .goals = goal_data,
                      .clause_index = 0,
                      .max_solutions = 0,
                  };

                  const auto encoded = encode_shard(original);
                  t.expect(encoded.has_value(), "encode_shard succeeds for atoms requiring quotes");
                  if (!encoded.has_value()) return;

                  const auto decoded = decode_shard(*encoded);
                  t.expect(decoded.has_value(), "decode_shard succeeds for atoms requiring quotes");
                  if (!decoded.has_value()) return;

                  t.expect(decoded->goals.size() == 1, "decoded goals size is 1");
                  if (!decoded->goals.empty()) {
                      t.expect(is_compound(decoded->goals[0]), "decoded goal is compound");
                      if (is_compound(decoded->goals[0])) {
                          const auto& args = compound_of(decoded->goals[0]).args;
                          t.expect(args.size() == 3, "decoded goal has 3 arguments");
                          if (args.size() >= 3) {
                              t.expect(is_atom(args[0]) && atom_of(args[0]).name == "hello world",
                                       "atom with space preserved exactly");
                              t.expect(is_atom(args[1]) && atom_of(args[1]).name.empty(),
                                       "empty atom preserved exactly");
                              t.expect(is_atom(args[2]) && atom_of(args[2]).name == "it's",
                                       "atom with single quote preserved exactly");
                          }
                      }
                  }
              })
        .test("variables_preserve_rename_generation_across_shard_roundtrip",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  using namespace nimblecas::logic_dist;

                  const Term var_x7 = make_var("X", 7);
                  const Term var_y0 = make_var("Y", 0);
                  const Term var_z42 = make_var("Z", 42);

                  Program prog_data;
                  prog_data.push_back(Clause{
                      .head = make_compound("rule_head", {var_x7}),
                      .body = {make_compound("body_goal", {var_y0, var_z42})},
                  });

                  std::vector<Term> goal_data;
                  goal_data.push_back(make_compound("query_goal", {var_x7, var_y0, var_z42}));

                  const Shard original{
                      .program = prog_data,
                      .goals = goal_data,
                      .clause_index = 0,
                      .max_solutions = 0,
                  };

                  const auto encoded = encode_shard(original);
                  t.expect(encoded.has_value(), "encode_shard succeeds for variables with non-zero generation");
                  if (!encoded.has_value()) return;

                  const auto decoded = decode_shard(*encoded);
                  t.expect(decoded.has_value(), "decode_shard succeeds for variables with non-zero generation");
                  if (!decoded.has_value()) return;

                  t.expect(decoded->goals.size() == 1, "decoded goals count is 1");
                  if (!decoded->goals.empty()) {
                      t.expect(is_compound(decoded->goals[0]), "goal is compound");
                      if (is_compound(decoded->goals[0])) {
                          const auto& args = compound_of(decoded->goals[0]).args;
                          t.expect(args.size() == 3, "goal has 3 variable arguments");
                          if (args.size() >= 3) {
                              t.expect(is_var(args[0]), "arg 0 is a variable");
                              if (is_var(args[0])) {
                                  t.expect(var_of(args[0]).name == "X", "variable name is X");
                                  t.expect(var_of(args[0]).generation == 7,
                                           "variable generation 7 preserved — not reset to 0");
                              }
                              t.expect(is_var(args[1]), "arg 1 is a variable");
                              if (is_var(args[1])) {
                                  t.expect(var_of(args[1]).name == "Y", "variable name is Y");
                                  t.expect(var_of(args[1]).generation == 0,
                                           "variable generation 0 preserved");
                              }
                              t.expect(is_var(args[2]), "arg 2 is a variable");
                              if (is_var(args[2])) {
                                  t.expect(var_of(args[2]).name == "Z", "variable name is Z");
                                  t.expect(var_of(args[2]).generation == 42,
                                           "variable generation 42 preserved");
                              }
                          }
                      }
                  }

                  t.expect(decoded->program.size() == 1, "program has 1 clause");
                  if (!decoded->program.empty()) {
                      const auto& cl = decoded->program[0];
                      t.expect(is_compound(cl.head), "clause head is compound");
                      if (is_compound(cl.head) && !compound_of(cl.head).args.empty()) {
                          const Term& head_arg = compound_of(cl.head).args[0];
                          t.expect(is_var(head_arg), "clause head arg is var");
                          if (is_var(head_arg)) {
                              t.expect(var_of(head_arg).name == "X" && var_of(head_arg).generation == 7,
                                       "clause head variable preserves generation 7");
                          }
                      }
                  }
              })
        .test("encode_decode_answers_roundtrip_compound_and_unbound",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  using namespace nimblecas::logic_dist;

                  std::vector<Substitution> answers;

                  // Answer 0: X bound to compound node f(g(1), [a, b])
                  Substitution sub0;
                  const Term compound_val = make_compound(
                      "f", {make_compound("g", {make_int(1)}),
                            make_list({make_atom("a"), make_atom("b")})});
                  sub0.emplace_back(VarKey{.name = "X", .generation = 0}, compound_val);
                  answers.push_back(std::move(sub0));

                  // Answer 1: empty substitution (unbound query variables / ground truth)
                  Substitution sub1;
                  answers.push_back(std::move(sub1));

                  // Answer 2: variable with non-zero generation bound to an integer
                  Substitution sub2;
                  sub2.emplace_back(VarKey{.name = "Ans", .generation = 99}, make_int(42));
                  answers.push_back(std::move(sub2));

                  const auto encoded = encode_answers(answers);
                  t.expect(encoded.has_value(), "encode_answers succeeds");
                  if (!encoded.has_value()) return;

                  const auto decoded = decode_answers(*encoded);
                  t.expect(decoded.has_value(), "decode_answers succeeds");
                  if (!decoded.has_value()) return;

                  t.expect(decoded->size() == 3, "decoded answers count is 3");
                  if (decoded->size() >= 3) {
                      t.expect((*decoded)[0].size() == 1, "answer 0 has 1 binding");
                      if (!(*decoded)[0].empty()) {
                          t.expect((*decoded)[0][0].first.name == "X", "answer 0 variable name is X");
                          t.expect((*decoded)[0][0].first.generation == 0, "answer 0 generation is 0");
                          t.expect((*decoded)[0][0].second == compound_val, "answer 0 bound value matches compound");
                      }

                      t.expect((*decoded)[1].empty(), "answer 1 is empty substitution representing unbound query");

                      t.expect((*decoded)[2].size() == 1, "answer 2 has 1 binding");
                      if (!(*decoded)[2].empty()) {
                          t.expect((*decoded)[2][0].first.name == "Ans", "answer 2 variable name is Ans");
                          t.expect((*decoded)[2][0].first.generation == 99, "answer 2 generation is 99");
                          t.expect((*decoded)[2][0].second == make_int(42), "answer 2 value is 42");
                      }
                  }
              })
        .test("decode_shard_rejects_malformed_or_unexpected_payloads",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  using namespace nimblecas::logic_dist;

                  const auto to_payload = [](std::string_view s) -> Payload {
                      Payload p;
                      p.reserve(s.size());
                      for (const char c : s) {
                          p.push_back(static_cast<std::byte>(static_cast<unsigned char>(c)));
                      }
                      return p;
                  };

                  // 1. Empty payload
                  const auto r_empty = decode_shard(std::span<const std::byte>{});
                  t.expect(!r_empty.has_value() && r_empty.error() == MathError::syntax_error,
                           "decode_shard on empty bytes returns syntax_error");

                  // 2. Unparseable garbage
                  const auto r_garbage = decode_shard(to_payload("### not valid prolog @@@"));
                  t.expect(!r_garbage.has_value() && r_garbage.error() == MathError::syntax_error,
                           "decode_shard on garbage string returns syntax_error");

                  // 3. Well-formed term of wrong shape
                  const auto r_wrong_shape = decode_shard(to_payload("foo(1)."));
                  t.expect(!r_wrong_shape.has_value() && r_wrong_shape.error() == MathError::syntax_error,
                           "decode_shard on wrong term shape foo(1) returns syntax_error");

                  // 4. Shard with wrong arity
                  const auto r_wrong_arity = decode_shard(to_payload("shard(0, 0)."));
                  t.expect(!r_wrong_arity.has_value() && r_wrong_arity.error() == MathError::syntax_error,
                           "decode_shard with wrong arity returns syntax_error");

                  // 5. Shard with negative clause index
                  const auto r_neg_idx = decode_shard(to_payload("shard(-1, 0, [], [])."));
                  t.expect(!r_neg_idx.has_value() && r_neg_idx.error() == MathError::syntax_error,
                           "decode_shard with negative clause index returns syntax_error");

                  // 6. Shard with negative max_solutions
                  const auto r_neg_max = decode_shard(to_payload("shard(0, -1, [], [])."));
                  t.expect(!r_neg_max.has_value() && r_neg_max.error() == MathError::syntax_error,
                           "decode_shard with negative max_solutions returns syntax_error");

                  // 7. Shard with non-integer metadata
                  const auto r_non_int = decode_shard(to_payload("shard(bad, 0, [], [])."));
                  t.expect(!r_non_int.has_value() && r_non_int.error() == MathError::syntax_error,
                           "decode_shard with non-integer clause index returns syntax_error");
              })
        .test("decode_answers_rejects_malformed_payloads",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  using namespace nimblecas::logic_dist;

                  const auto to_payload = [](std::string_view s) -> Payload {
                      Payload p;
                      p.reserve(s.size());
                      for (const char c : s) {
                          p.push_back(static_cast<std::byte>(static_cast<unsigned char>(c)));
                      }
                      return p;
                  };

                  const auto r1 = decode_answers(std::span<const std::byte>{});
                  t.expect(!r1.has_value() && r1.error() == MathError::syntax_error,
                           "decode_answers on empty bytes returns syntax_error");

                  const auto r2 = decode_answers(to_payload("corrupted payload"));
                  t.expect(!r2.has_value() && r2.error() == MathError::syntax_error,
                           "decode_answers on garbage text returns syntax_error");

                  const auto r3 = decode_answers(to_payload("foo(1)."));
                  t.expect(!r3.has_value() && r3.error() == MathError::syntax_error,
                           "decode_answers on wrong term shape returns syntax_error");

                  const auto r4 = decode_answers(to_payload("answers(42)."));
                  t.expect(!r4.has_value() && r4.error() == MathError::syntax_error,
                           "decode_answers with non-list argument returns syntax_error");
              })
        .test("encode_shard_with_empty_goals_returns_domain_error",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  using namespace nimblecas::logic_dist;

                  const Shard sh{
                      .program = prog("p(1)."),
                      .goals = {},
                      .clause_index = 0,
                      .max_solutions = 0,
                  };
                  const auto res = encode_shard(sh);
                  t.expect(!res.has_value(), "encode_shard rejects empty goal list");
                  if (!res.has_value()) {
                      t.expect(res.error() == MathError::domain_error,
                               "empty goals list in encode_shard yields domain_error");
                  }
              })
        .test("solve_distributed_simple_facts_matches_serial_and_parallel_executors",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  using namespace nimblecas::logic_dist;

                  const auto p = prog("p(1). p(2). p(3).");
                  const auto q = query("p(X).");

                  const auto r_solve = solve(p, q, 0);
                  t.expect(r_solve.has_value(), "serial solve succeeds");
                  if (!r_solve.has_value()) return;

                  auto ser_exec = serial_executor();
                  auto par_exec = local_parallel_executor();
                  t.expect(ser_exec != nullptr && par_exec != nullptr, "executors created");
                  if (!ser_exec || !par_exec) return;

                  const auto r_ser = solve_distributed(p, q, 0, *ser_exec);
                  const auto r_par = solve_distributed(p, q, 0, *par_exec);

                  t.expect(r_ser.has_value(), "solve_distributed with serial_executor succeeds");
                  t.expect(r_par.has_value(), "solve_distributed with local_parallel_executor succeeds");
                  if (!r_ser.has_value() || !r_par.has_value()) return;

                  t.expect(r_ser->size() == 3, "serial executor returned exactly 3 answers");
                  t.expect(r_par->size() == 3, "parallel executor returned exactly 3 answers");

                  if (r_ser->size() >= 3 && r_par->size() >= 3) {
                      t.expect(bound((*r_ser)[0], "X") == "1", "serial answer 0 is 1");
                      t.expect(bound((*r_ser)[1], "X") == "2", "serial answer 1 is 2");
                      t.expect(bound((*r_ser)[2], "X") == "3", "serial answer 2 is 3");

                      t.expect(bound((*r_par)[0], "X") == "1", "parallel answer 0 is 1");
                      t.expect(bound((*r_par)[1], "X") == "2", "parallel answer 1 is 2");
                      t.expect(bound((*r_par)[2], "X") == "3", "parallel answer 2 is 3");

                      t.expect(*r_ser == *r_solve, "serial executor answers equal solve identically");
                      t.expect(*r_par == *r_solve, "parallel executor answers equal solve identically");
                      t.expect(*r_ser == *r_par, "serial and parallel executor answers match bit-for-bit");
                  }
              })
        .test("solve_distributed_recursive_ancestor_family_matches_serial_solve",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  using namespace nimblecas::logic_dist;

                  const auto p = prog(
                      "parent(tom, bob).\n"
                      "parent(bob, ann).\n"
                      "parent(bob, pat).\n"
                      "ancestor(X, Y) :- parent(X, Y).\n"
                      "ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).\n");
                  const auto q = query("ancestor(tom, Descendant).");

                  const auto r_solve = solve(p, q, 0);
                  t.expect(r_solve.has_value(), "solve succeeds on recursive ancestor program");
                  if (!r_solve.has_value()) return;

                  auto ser_exec = serial_executor();
                  auto par_exec = local_parallel_executor();
                  if (!ser_exec || !par_exec) return;

                  const auto r_ser = solve_distributed(p, q, 0, *ser_exec);
                  const auto r_par = solve_distributed(p, q, 0, *par_exec);

                  t.expect(r_ser.has_value(), "solve_distributed serial succeeds on ancestor");
                  t.expect(r_par.has_value(), "solve_distributed parallel succeeds on ancestor");
                  if (!r_ser.has_value() || !r_par.has_value()) return;

                  t.expect(r_ser->size() == 3, "found all 3 descendants of tom");
                  t.expect(r_par->size() == 3, "parallel found all 3 descendants of tom");

                  if (r_ser->size() >= 3 && r_par->size() >= 3) {
                      t.expect(bound((*r_ser)[0], "Descendant") == "bob", "descendant 0 is bob");
                      t.expect(bound((*r_ser)[1], "Descendant") == "ann", "descendant 1 is ann");
                      t.expect(bound((*r_ser)[2], "Descendant") == "pat", "descendant 2 is pat");

                      t.expect(bound((*r_par)[0], "Descendant") == "bob", "parallel descendant 0 is bob");
                      t.expect(bound((*r_par)[1], "Descendant") == "ann", "parallel descendant 1 is ann");
                      t.expect(bound((*r_par)[2], "Descendant") == "pat", "parallel descendant 2 is pat");

                      t.expect(*r_ser == *r_solve, "ancestor serial answers equal solve");
                      t.expect(*r_par == *r_solve, "ancestor parallel answers equal solve");
                  }
              })
        .test("solve_distributed_recursive_grandparent_matches_serial_solve",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  using namespace nimblecas::logic_dist;

                  const auto p = prog(
                      "parent(tom, bob).\n"
                      "parent(bob, ann).\n"
                      "parent(bob, pat).\n"
                      "grandparent(X, Z) :- parent(X, Y), parent(Y, Z).\n");
                  const auto q = query("grandparent(tom, Grandchild).");

                  const auto r_solve = solve(p, q, 0);
                  t.expect(r_solve.has_value(), "solve succeeds on grandparent");
                  if (!r_solve.has_value()) return;

                  auto ser_exec = serial_executor();
                  auto par_exec = local_parallel_executor();
                  if (!ser_exec || !par_exec) return;

                  const auto r_ser = solve_distributed(p, q, 0, *ser_exec);
                  const auto r_par = solve_distributed(p, q, 0, *par_exec);

                  t.expect(r_ser.has_value() && r_par.has_value(), "both executors succeed on grandparent");
                  if (!r_ser.has_value() || !r_par.has_value()) return;

                  t.expect(r_ser->size() == 2, "exactly 2 grandchildren found");
                  t.expect(r_par->size() == 2, "parallel found exactly 2 grandchildren");

                  if (r_ser->size() >= 2 && r_par->size() >= 2) {
                      t.expect(bound((*r_ser)[0], "Grandchild") == "ann", "first grandchild is ann");
                      t.expect(bound((*r_ser)[1], "Grandchild") == "pat", "second grandchild is pat");

                      t.expect(bound((*r_par)[0], "Grandchild") == "ann", "parallel grandchild 0 is ann");
                      t.expect(bound((*r_par)[1], "Grandchild") == "pat", "parallel grandchild 1 is pat");

                      t.expect(*r_ser == *r_solve, "grandparent serial distributed matches solve");
                      t.expect(*r_par == *r_solve, "grandparent parallel distributed matches solve");
                  }
              })
        .test("solve_distributed_max_solutions_cap_one_returns_first_answer_only",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  using namespace nimblecas::logic_dist;

                  const auto p = prog("p(1). p(2). p(3).");
                  const auto q = query("p(X).");

                  const auto r_solve = solve(p, q, 1);
                  t.expect(r_solve.has_value(), "solve with cap 1 succeeds");
                  if (!r_solve.has_value()) return;

                  auto ser_exec = serial_executor();
                  auto par_exec = local_parallel_executor();
                  if (!ser_exec || !par_exec) return;

                  const auto r_ser = solve_distributed(p, q, 1, *ser_exec);
                  const auto r_par = solve_distributed(p, q, 1, *par_exec);

                  t.expect(r_ser.has_value(), "solve_distributed cap 1 serial succeeds");
                  t.expect(r_par.has_value(), "solve_distributed cap 1 parallel succeeds");
                  if (!r_ser.has_value() || !r_par.has_value()) return;

                  t.expect(r_ser->size() == 1, "serial executor returned exactly one solution");
                  t.expect(r_par->size() == 1, "parallel executor returned exactly one solution");

                  if (r_ser->size() >= 1 && r_par->size() >= 1) {
                      t.expect(bound((*r_ser)[0], "X") == "1", "returned solution is the FIRST (1)");
                      t.expect(bound((*r_par)[0], "X") == "1", "parallel solution is the FIRST (1)");

                      t.expect(*r_ser == *r_solve, "matches solve with cap 1");
                      t.expect(*r_par == *r_solve, "parallel matches solve with cap 1");
                  }
              })
        .test("solve_distributed_max_solutions_intermediate_cap",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  using namespace nimblecas::logic_dist;

                  const auto p = prog("p(1). p(2). p(3). p(4).");
                  const auto q = query("p(X).");

                  const auto r_solve = solve(p, q, 2);
                  t.expect(r_solve.has_value(), "solve with cap 2 succeeds");
                  if (!r_solve.has_value()) return;

                  auto ser_exec = serial_executor();
                  auto par_exec = local_parallel_executor();
                  if (!ser_exec || !par_exec) return;

                  const auto r_ser = solve_distributed(p, q, 2, *ser_exec);
                  const auto r_par = solve_distributed(p, q, 2, *par_exec);

                  t.expect(r_ser.has_value() && r_par.has_value(), "both executors succeed with cap 2");
                  if (!r_ser.has_value() || !r_par.has_value()) return;

                  t.expect(r_ser->size() == 2, "serial returns exactly 2 answers");
                  t.expect(r_par->size() == 2, "parallel returns exactly 2 answers");

                  if (r_ser->size() >= 2 && r_par->size() >= 2) {
                      t.expect(bound((*r_ser)[0], "X") == "1", "answer 0 is 1");
                      t.expect(bound((*r_ser)[1], "X") == "2", "answer 1 is 2");

                      t.expect(bound((*r_par)[0], "X") == "1", "parallel answer 0 is 1");
                      t.expect(bound((*r_par)[1], "X") == "2", "parallel answer 1 is 2");

                      t.expect(*r_ser == *r_solve, "serial matches solve(2)");
                      t.expect(*r_par == *r_solve, "parallel matches solve(2)");
                  }
              })
        .test("solve_distributed_query_with_no_solutions_returns_empty_vector",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  using namespace nimblecas::logic_dist;

                  const auto p = prog("p(1). p(2).");
                  const auto q = query("p(99).");

                  auto ser_exec = serial_executor();
                  auto par_exec = local_parallel_executor();
                  if (!ser_exec || !par_exec) return;

                  const auto r_ser = solve_distributed(p, q, 0, *ser_exec);
                  const auto r_par = solve_distributed(p, q, 0, *par_exec);

                  t.expect(r_ser.has_value(), "unsolvable query succeeds without error in serial");
                  t.expect(r_par.has_value(), "unsolvable query succeeds without error in parallel");
                  if (r_ser.has_value()) {
                      t.expect(r_ser->empty(), "serial result is an empty vector of substitutions");
                  }
                  if (r_par.has_value()) {
                      t.expect(r_par->empty(), "parallel result is an empty vector of substitutions");
                  }
              })
        .test("solve_distributed_multiple_predicates_filters_correctly_in_program_order",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  using namespace nimblecas::logic_dist;

                  const auto p = prog(
                      "q(10).\n"
                      "p(1).\n"
                      "r(20).\n"
                      "p(2).\n"
                      "q(30).\n"
                      "p(3).\n"
                      "r(40).\n");
                  const auto q = query("p(X).");

                  const auto r_solve = solve(p, q, 0);
                  t.expect(r_solve.has_value(), "solve succeeds on multi-predicate program");
                  if (!r_solve.has_value()) return;

                  auto ser_exec = serial_executor();
                  auto par_exec = local_parallel_executor();
                  if (!ser_exec || !par_exec) return;

                  const auto r_ser = solve_distributed(p, q, 0, *ser_exec);
                  const auto r_par = solve_distributed(p, q, 0, *par_exec);

                  t.expect(r_ser.has_value() && r_par.has_value(), "distributed solves succeed");
                  if (!r_ser.has_value() || !r_par.has_value()) return;

                  t.expect(r_ser->size() == 3, "only the 3 matching p clauses contribute");
                  t.expect(r_par->size() == 3, "parallel receives only the 3 matching p clauses");

                  if (r_ser->size() >= 3 && r_par->size() >= 3) {
                      t.expect(bound((*r_ser)[0], "X") == "1", "first answer is 1");
                      t.expect(bound((*r_ser)[1], "X") == "2", "second answer is 2");
                      t.expect(bound((*r_ser)[2], "X") == "3", "third answer is 3");

                      t.expect(*r_ser == *r_solve, "matches serial solve in exact clause order");
                      t.expect(*r_par == *r_solve, "parallel matches serial solve in exact clause order");
                  }
              })
        .test("solve_distributed_multi_variable_query_preserves_all_bindings",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  using namespace nimblecas::logic_dist;

                  const auto p = prog(
                      "edge(a, b).\n"
                      "edge(b, c).\n"
                      "edge(a, c).\n");
                  const auto q = query("edge(X, Y).");

                  const auto r_solve = solve(p, q, 0);
                  t.expect(r_solve.has_value(), "solve succeeds for multi-variable query");
                  if (!r_solve.has_value()) return;

                  auto ser_exec = serial_executor();
                  auto par_exec = local_parallel_executor();
                  if (!ser_exec || !par_exec) return;

                  const auto r_ser = solve_distributed(p, q, 0, *ser_exec);
                  const auto r_par = solve_distributed(p, q, 0, *par_exec);

                  t.expect(r_ser.has_value() && r_par.has_value(), "both distributed executors succeed");
                  if (!r_ser.has_value() || !r_par.has_value()) return;

                  t.expect(r_ser->size() == 3, "3 edge answers found");
                  t.expect(r_par->size() == 3, "parallel found 3 edge answers");

                  if (r_ser->size() >= 3 && r_par->size() >= 3) {
                      t.expect(bound((*r_ser)[0], "X") == "a" && bound((*r_ser)[0], "Y") == "b",
                               "edge 0 is a -> b");
                      t.expect(bound((*r_ser)[1], "X") == "b" && bound((*r_ser)[1], "Y") == "c",
                               "edge 1 is b -> c");
                      t.expect(bound((*r_ser)[2], "X") == "a" && bound((*r_ser)[2], "Y") == "c",
                               "edge 2 is a -> c");

                      t.expect(*r_ser == *r_solve, "serial matches solve");
                      t.expect(*r_par == *r_solve, "parallel matches solve");
                  }
              })
        .test("solve_clause_branch_commit_to_clause_zero",
              [](TestContext& t) -> void {
                  using namespace nimblecas;

                  const auto p = prog("p(1). p(2).");
                  const auto q = query("p(X).");

                  const auto r = solve_clause_branch(p, q, 0, 0);
                  t.expect(r.has_value(), "solve_clause_branch(0) succeeds");
                  if (!r.has_value()) return;

                  t.expect(r->size() == 1, "committing to clause 0 yields exactly 1 answer");
                  if (!r->empty()) {
                      t.expect(bound((*r)[0], "X") == "1", "clause 0 answer is X = 1");
                  }
              })
        .test("solve_clause_branch_commit_to_clause_one",
              [](TestContext& t) -> void {
                  using namespace nimblecas;

                  const auto p = prog("p(1). p(2).");
                  const auto q = query("p(X).");

                  const auto r = solve_clause_branch(p, q, 1, 0);
                  t.expect(r.has_value(), "solve_clause_branch(1) succeeds");
                  if (!r.has_value()) return;

                  t.expect(r->size() == 1, "committing to clause 1 yields exactly 1 answer");
                  if (!r->empty()) {
                      t.expect(bound((*r)[0], "X") == "2", "clause 1 answer is X = 2");
                  }
              })
        .test("solve_clause_branch_clause_index_past_the_end_returns_empty",
              [](TestContext& t) -> void {
                  using namespace nimblecas;

                  const auto p = prog("p(1). p(2).");
                  const auto q = query("p(X).");

                  const auto r1 = solve_clause_branch(p, q, 2, 0);
                  t.expect(r1.has_value(), "clause index 2 (at boundary) succeeds without error");
                  if (r1.has_value()) {
                      t.expect(r1->empty(), "clause index 2 yields empty answer list");
                  }

                  const auto r2 = solve_clause_branch(p, q, 999, 0);
                  t.expect(r2.has_value(), "clause index 999 (far past end) succeeds without error");
                  if (r2.has_value()) {
                      t.expect(r2->empty(), "clause index 999 yields empty answer list");
                  }
              })
        .test("solve_clause_branch_clause_for_different_predicate_returns_empty",
              [](TestContext& t) -> void {
                  using namespace nimblecas;

                  const auto p = prog(
                      "p(1).\n"
                      "q(2).\n"
                      "p(3).\n");
                  const auto qry = query("p(X).");

                  // Clause index 1 is `q(2)`
                  const auto r = solve_clause_branch(p, qry, 1, 0);
                  t.expect(r.has_value(), "clause index 1 (predicate q) succeeds without error");
                  if (r.has_value()) {
                      t.expect(r->empty(), "clause index for different predicate yields empty answers");
                  }
              })
        .test("solve_clause_branch_concatenating_answers_in_order_equals_solve",
              [](TestContext& t) -> void {
                  using namespace nimblecas;

                  const auto p = prog(
                      "p(1).\n"
                      "other(50).\n"
                      "p(2).\n"
                      "p(3).\n");
                  const auto q = query("p(X).");

                  const auto r_solve = solve(p, q, 0);
                  t.expect(r_solve.has_value(), "solve succeeds");
                  if (!r_solve.has_value()) return;

                  std::vector<Substitution> concatenated;
                  for (std::size_t ci = 0; ci < p.size(); ++ci) {
                      const auto r_branch = solve_clause_branch(p, q, ci, 0);
                      t.expect(r_branch.has_value(), "branch solve succeeds");
                      if (!r_branch.has_value()) return;
                      for (const auto& s : *r_branch) {
                          concatenated.push_back(s);
                      }
                  }

                  t.expect(concatenated.size() == r_solve->size(), "concatenated answers size matches solve");
                  t.expect(concatenated == *r_solve,
                           "concatenating every clause index in order equals solve exactly");
              })
        .test("is_distributable_returns_false_for_empty_goal_list",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  using namespace nimblecas::logic_dist;

                  const auto p = prog("p(1).");
                  t.expect(!is_distributable(p, {}),
                           "is_distributable returns false for empty goal conjunction");
              })
        .test("is_distributable_returns_true_for_regular_horn_clauses",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  using namespace nimblecas::logic_dist;

                  const auto p = prog("p(1). p(2).");
                  const auto q = query("p(X).");
                  t.expect(is_distributable(p, q),
                           "is_distributable returns true for valid Horn clauses and query");
              })
        .test("solve_distributed_delegates_program_with_cut_to_serial_solver",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  using namespace nimblecas::logic_dist;

                  const auto p = prog("p(1). p(2). p(3). q(X) :- p(X), !.");
                  const auto q = query("q(X).");

                  const auto r_solve = solve(p, q, 0);
                  t.expect(r_solve.has_value(), "solve succeeds on program containing cut");
                  if (!r_solve.has_value()) return;

                  auto ser_exec = serial_executor();
                  auto par_exec = local_parallel_executor();
                  if (!ser_exec || !par_exec) return;

                  const auto r_ser = solve_distributed(p, q, 0, *ser_exec);
                  const auto r_par = solve_distributed(p, q, 0, *par_exec);

                  t.expect(r_ser.has_value(), "solve_distributed serial succeeds on program containing cut");
                  t.expect(r_par.has_value(), "solve_distributed parallel succeeds on program containing cut");
                  if (!r_ser.has_value() || !r_par.has_value()) return;

                  t.expect(r_ser->size() == 1, "cut prunes alternatives in serial distributed execution");
                  t.expect(r_par->size() == 1, "cut prunes alternatives in parallel distributed execution");

                  if (!r_ser->empty() && !r_par->empty()) {
                      t.expect(bound((*r_ser)[0], "X") == "1", "answer is 1");
                      t.expect(bound((*r_par)[0], "X") == "1", "parallel answer is 1");
                      t.expect(*r_ser == *r_solve, "serial distributed answers equal solve on cut");
                      t.expect(*r_par == *r_solve, "parallel distributed answers equal solve on cut");
                  }
              })
        .test("solve_distributed_delegates_program_with_assertz_to_serial_solver",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  using namespace nimblecas::logic_dist;

                  const auto p = prog(
                      "add_fact(X) :- assertz(fact(X)).\n"
                      "fact(10).\n");
                  const auto q = query("add_fact(20), fact(V).");

                  const auto r_solve = solve(p, q, 0);
                  t.expect(r_solve.has_value(), "solve succeeds on program with assertz");
                  if (!r_solve.has_value()) return;

                  auto ser_exec = serial_executor();
                  auto par_exec = local_parallel_executor();
                  if (!ser_exec || !par_exec) return;

                  const auto r_ser = solve_distributed(p, q, 0, *ser_exec);
                  const auto r_par = solve_distributed(p, q, 0, *par_exec);

                  t.expect(r_ser.has_value(), "solve_distributed serial succeeds on assertz program");
                  t.expect(r_par.has_value(), "solve_distributed parallel succeeds on assertz program");
                  if (!r_ser.has_value() || !r_par.has_value()) return;

                  t.expect(r_ser->size() == 2, "exactly 2 answers found: 10 and 20");
                  t.expect(r_par->size() == 2, "parallel found exactly 2 answers");

                  if (r_ser->size() >= 2 && r_par->size() >= 2) {
                      t.expect(bound((*r_ser)[0], "V") == "10", "first answer is 10");
                      t.expect(bound((*r_ser)[1], "V") == "20", "second answer is 20");

                      t.expect(bound((*r_par)[0], "V") == "10", "parallel first answer is 10");
                      t.expect(bound((*r_par)[1], "V") == "20", "parallel second answer is 20");

                      t.expect(*r_ser == *r_solve, "serial answers equal solve on assertz program");
                      t.expect(*r_par == *r_solve, "parallel answers equal solve on assertz program");
                  }
              })
        .test("build_shard_graph_with_empty_goals_returns_domain_error",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  using namespace nimblecas::logic_dist;

                  TaskRegistry reg;
                  const auto p = prog("p(1).");
                  const auto r = build_shard_graph(reg, p, {}, 0);
                  t.expect(!r.has_value(), "build_shard_graph rejects empty goal list");
                  if (!r.has_value()) {
                      t.expect(r.error() == MathError::domain_error,
                               "empty goal list in build_shard_graph returns domain_error");
                  }
              })
        .test("build_shard_graph_creates_correct_number_of_independent_tasks",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  using namespace nimblecas::logic_dist;

                  TaskRegistry reg;
                  const auto reg_res = register_ops(reg);
                  t.expect(reg_res.has_value(), "register_ops succeeds");

                  const auto p = prog("p(1). p(2). p(3).");
                  const auto q = query("p(X).");

                  const auto g = build_shard_graph(reg, p, q, 0);
                  t.expect(g.has_value(), "build_shard_graph succeeds for 3-clause program");
                  if (g.has_value()) {
                      t.expect(g->size() == 3, "task graph contains exactly 3 tasks (one per clause)");
                  }
              })
        .test("register_ops_fails_with_domain_error_when_called_twice",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  using namespace nimblecas::logic_dist;

                  TaskRegistry reg;
                  const auto r1 = register_ops(reg);
                  t.expect(r1.has_value(), "first register_ops call on registry succeeds");

                  const auto r2 = register_ops(reg);
                  t.expect(!r2.has_value(), "second register_ops call on registry fails");
                  if (!r2.has_value()) {
                      t.expect(r2.error() == MathError::domain_error,
                               "duplicate op registration returns domain_error");
                  }
              })
        .test("shard_op_id_constant_matches_expected_specification",
              [](TestContext& t) -> void {
                  using namespace nimblecas::logic_dist;

                  t.expect(shard_op_id == "nimblecas.logic.sld_shard/v1",
                           "shard_op_id matches versioned contract string nimblecas.logic.sld_shard/v1");
              })
        .test("solve_distributed_ground_query_success_and_failure",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  using namespace nimblecas::logic_dist;

                  const auto p = prog(
                      "likes(mary, food).\n"
                      "likes(mary, wine).\n"
                      "likes(john, wine).\n");

                  // Ground query that succeeds: likes(mary, wine).
                  const auto q_true = query("likes(mary, wine).");
                  const auto r_solve_true = solve(p, q_true, 0);
                  t.expect(r_solve_true.has_value(), "solve succeeds on true ground query");

                  auto ser_exec = serial_executor();
                  auto par_exec = local_parallel_executor();
                  if (!ser_exec || !par_exec) return;

                  const auto r_ser_true = solve_distributed(p, q_true, 0, *ser_exec);
                  const auto r_par_true = solve_distributed(p, q_true, 0, *par_exec);

                  t.expect(r_ser_true.has_value() && r_par_true.has_value(), "distributed true queries succeed");
                  if (r_ser_true.has_value() && r_par_true.has_value()) {
                      t.expect(r_ser_true->size() == 1, "ground success returns 1 empty substitution (true)");
                      t.expect(r_par_true->size() == 1, "parallel ground success returns 1 empty substitution");
                      t.expect(*r_ser_true == *r_solve_true, "serial matches solve on ground success");
                      t.expect(*r_par_true == *r_solve_true, "parallel matches solve on ground success");
                  }

                  // Ground query that fails: likes(john, food).
                  const auto q_false = query("likes(john, food).");
                  const auto r_solve_false = solve(p, q_false, 0);
                  t.expect(r_solve_false.has_value() && r_solve_false->empty(), "solve returns empty for false query");

                  const auto r_ser_false = solve_distributed(p, q_false, 0, *ser_exec);
                  const auto r_par_false = solve_distributed(p, q_false, 0, *par_exec);

                  t.expect(r_ser_false.has_value() && r_ser_false->empty(), "serial returns empty for false query");
                  t.expect(r_par_false.has_value() && r_par_false->empty(), "parallel returns empty for false query");
              })
        .test("solve_distributed_list_member_recursion_matches_solve",
              [](TestContext& t) -> void {
                  using namespace nimblecas;
                  using namespace nimblecas::logic_dist;

                  const auto p = prog(
                      "member(X, [X|_]).\n"
                      "member(X, [_|T]) :- member(X, T).\n");
                  const auto q = query("member(Elem, [10, 20, 30]).");

                  const auto r_solve = solve(p, q, 0);
                  t.expect(r_solve.has_value(), "solve succeeds on member/2");
                  if (!r_solve.has_value()) return;

                  auto ser_exec = serial_executor();
                  auto par_exec = local_parallel_executor();
                  if (!ser_exec || !par_exec) return;

                  const auto r_ser = solve_distributed(p, q, 0, *ser_exec);
                  const auto r_par = solve_distributed(p, q, 0, *par_exec);

                  t.expect(r_ser.has_value() && r_par.has_value(), "distributed member/2 succeeds");
                  if (!r_ser.has_value() || !r_par.has_value()) return;

                  t.expect(r_ser->size() == 3, "found 3 list elements");
                  t.expect(r_par->size() == 3, "parallel found 3 list elements");

                  if (r_ser->size() >= 3 && r_par->size() >= 3) {
                      t.expect(bound((*r_ser)[0], "Elem") == "10", "first element is 10");
                      t.expect(bound((*r_ser)[1], "Elem") == "20", "second element is 20");
                      t.expect(bound((*r_ser)[2], "Elem") == "30", "third element is 30");

                      t.expect(*r_ser == *r_solve, "serial member/2 matches solve");
                      t.expect(*r_par == *r_solve, "parallel member/2 matches solve");
                  }
              })
        .test("solve_distributed_gives_the_same_answers_over_the_sgee_distributed_executor",
              [](TestContext& t) {
                  // Everything above runs over executors that share this process's memory. SGEE
                  // does not: it ships an op id and bytes to a worker, which looks the op up in
                  // ITS OWN registry and runs it with no access to anything the coordinator
                  // holds. That is why `register_ops` is exported, and until this test existed
                  // "any Executor, including a distributed one" was an intention rather than a
                  // fact.
                  TaskRegistry reg;
                  t.expect(nimblecas::logic_dist::register_ops(reg).has_value(),
                           "the shard operation registers into the executor's registry");
                  FakeBrokerPort port;
                  InMemoryResultChannel results;
                  SgeeExecutorConfig cfg;
                  cfg.registry = &reg;
                  cfg.num_workers = 4;
                  cfg.poll_interval_ms = 1;
                  SgeeDistributedExecutor sgee(cfg, port, results);
                  t.expect(sgee.name() == "sgee_distributed",
                           "the executor under test really is the distributed one");

                  const Program facts = prog("p(1). p(2). p(3).");
                  const std::vector<Term> q = query("p(X).");
                  auto expected = solve(facts, q, 0);
                  auto over_sgee = nimblecas::logic_dist::solve_distributed(facts, q, 0, sgee);
                  t.expect(expected.has_value() && over_sgee.has_value(),
                           "both solves succeed");
                  if (!expected.has_value() || !over_sgee.has_value()) {
                      return;
                  }
                  t.expect(over_sgee->size() == expected->size(),
                           "the same number of answers comes back from the cluster");
                  if (over_sgee->size() != expected->size()) {
                      return;
                  }
                  for (std::size_t i = 0; i < expected->size(); ++i) {
                      t.expect(bound((*over_sgee)[i], "X") == bound((*expected)[i], "X"),
                               "each answer is identical, in the same order");
                  }
              })
        .test("a_recursive_program_solves_identically_over_sgee",
              [](TestContext& t) {
                  TaskRegistry reg;
                  if (!nimblecas::logic_dist::register_ops(reg)) {
                      t.expect(false, "the shard operation registers");
                      return;
                  }
                  FakeBrokerPort port;
                  InMemoryResultChannel results;
                  SgeeExecutorConfig cfg;
                  cfg.registry = &reg;
                  cfg.num_workers = 3;
                  cfg.poll_interval_ms = 1;
                  SgeeDistributedExecutor sgee(cfg, port, results);

                  const Program family = prog(
                      "parent(tom, bob). parent(bob, ann). parent(bob, pat). "
                      "grandparent(X, Z) :- parent(X, Y), parent(Y, Z).");
                  const std::vector<Term> q = query("grandparent(tom, W).");
                  auto expected = solve(family, q, 0);
                  auto over_sgee = nimblecas::logic_dist::solve_distributed(family, q, 0, sgee);
                  t.expect(expected.has_value() && over_sgee.has_value(), "both solves succeed");
                  if (!expected.has_value() || !over_sgee.has_value()) {
                      return;
                  }
                  t.expect(over_sgee->size() == expected->size(),
                           "a recursive program gives the same answer count over the cluster");
                  if (over_sgee->size() != expected->size() || expected->empty()) {
                      return;
                  }
                  for (std::size_t i = 0; i < expected->size(); ++i) {
                      t.expect(bound((*over_sgee)[i], "W") == bound((*expected)[i], "W"),
                               "and each binding is identical, in order");
                  }
              })
        .test("sgee_answers_do_not_change_with_the_worker_count",
              [](TestContext& t) {
                  const Program facts = prog("q(a). q(b). q(c). q(d). q(e).");
                  const std::vector<Term> qy = query("q(X).");
                  auto expected = solve(facts, qy, 0);
                  t.expect(expected.has_value(), "the serial baseline succeeds");
                  if (!expected.has_value()) {
                      return;
                  }
                  for (const std::size_t workers : {std::size_t{1}, std::size_t{2},
                                                    std::size_t{6}}) {
                      TaskRegistry reg;
                      if (!nimblecas::logic_dist::register_ops(reg)) {
                          t.expect(false, "the shard operation registers");
                          return;
                      }
                      FakeBrokerPort port;
                      InMemoryResultChannel results;
                      SgeeExecutorConfig cfg;
                      cfg.registry = &reg;
                      cfg.num_workers = workers;
                      cfg.poll_interval_ms = 1;
                      SgeeDistributedExecutor sgee(cfg, port, results);
                      auto r = nimblecas::logic_dist::solve_distributed(facts, qy, 0, sgee);
                      bool same = r.has_value() && r->size() == expected->size();
                      for (std::size_t i = 0; same && i < expected->size(); ++i) {
                          same = bound((*r)[i], "X") == bound((*expected)[i], "X");
                      }
                      t.expect(same,
                               "the answers are identical however many workers ran them");
                  }
              })
        .run();
}
