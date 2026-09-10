// NimbleCAS distributed logic — SLD resolution as a task graph over SGEE.
// @author Olumuyiwa Oluwasanmi
//
// `nimblecas.logic` already decomposes a query into independent OR-branches: one per candidate
// clause of the first goal, each a pure function of its arguments with its own substitution,
// rename counter, step budget and database copy. `solve_or_parallel` fans those across threads.
// This module fans the SAME branches across PROCESSES, by expressing them as a `TaskGraph` of
// `nimblecas.taskdag` tasks — which any `Executor` can run: the serial reference, the local
// parallel one, or `nimblecas.taskdag_sgee`'s distributed executor over a broker.
//
// The decomposition is not re-derived here. `solve_clause_branch` is the unit of work in both
// cases, so the local and distributed paths cannot drift apart — a bug fixed in one is fixed in
// both, and a test that pins one pins the other.
//
// WHAT CROSSES THE WIRE, and why it is Prolog text.
//
// A `TaskFn` is `Result<Payload>(span<const Payload>)` — opaque bytes in, opaque bytes out. A
// shard therefore has to be serialised, and this module writes it as PROLOG TEXT, read back by
// `nimblecas.logic_parser`. That is a deliberate choice over inventing a binary format:
//
//   * the reader and writer are already implemented, already tested, and already hold a
//     round-trip guarantee — what the writer emits, the reader reads back as the same term;
//   * a shard on the wire is READABLE, so a stuck worker's payload can be inspected by a person
//     rather than hex-dumped;
//   * there is one term representation in the system rather than two that can disagree.
//
// The cost is size, and it is the right trade for a payload whose cost is dominated by the
// search it triggers rather than by its own bytes.
//
// Variables are encoded EXPLICITLY as `v(Name, Generation)` rather than written as source
// variables. Source syntax cannot express a variable's rename generation — `to_source` renders a
// standardised-apart variable as a NAME, which reads back as a different variable with
// generation 0. For ordinary program text that is harmless; for a shard it would silently merge
// variables that the solver had carefully kept apart. Being explicit costs a few bytes and
// removes the whole class of failure.
//
// THE FAN-OUT IS ONE LEVEL DEEP. This is the honest limit of the design and worth stating
// before anyone measures it and is surprised.
//
// A shard runs a COMPLETE sub-derivation in its own process — depth-first, with backtracking,
// to exhaustion or to its budget — and returns every answer it found. The graph is therefore
// STATIC: one task per top-level clause, no dependencies, known before the run starts. That is
// what lets an ordinary `TaskGraph` express it at all, because a `TaskFn` receives only its
// argument bytes and has no handle on the broker: a worker CANNOT enqueue new work, so a
// decomposition that discovered tasks as it went could not be expressed this way.
//
// The consequence is that parallelism is bounded by the number of candidate clauses at the top
// goal. A query whose first goal has two clauses uses two workers however deep the search below
// them runs, and a deeply recursive query over a single clause distributes to exactly one.
//
// Lifting it does NOT need new SGEE plumbing, which is the useful part: a shard that hits its
// budget could return a CONTINUATION — the goal list and substitution it stopped at — and a
// coordinator loop could build a fresh graph from those continuations and run another wave.
// `TaskGraph` is immutable-by-append and a per-wave graph is cheap, so adaptive splitting is a
// coordinator-side loop over successive static graphs rather than a change to the substrate.
// That is future work, and it is deliberately not pretended to here.
//
// WHAT IS REFUSED, and why refusing is right.
//
// A program containing a CUT or a DATABASE UPDATE is not distributable, exactly as it is not
// OR-parallelisable: a cut commits to one clause after the others have already been evaluated,
// and a side effect performed by a shard that serial resolution would never have reached cannot
// be un-performed by the merge. `solve_distributed` runs such a program SERIALLY rather than
// refusing outright, because the contract callers rely on is the ANSWERS — the same ones
// `solve` gives — and delegating keeps that contract while a refusal would break it.

export module nimblecas.logic_dist;

import std;
import nimblecas.core;
import nimblecas.logic;
import nimblecas.logic_parser;
import nimblecas.taskdag;

export namespace nimblecas::logic_dist {

// The registered operation a shard runs under. Versioned, because the wire format is part of the
// contract between a coordinator and a worker that may be running a different build: bumping the
// version is how an incompatible change announces itself instead of being discovered as a
// mis-parsed payload.
inline constexpr std::string_view shard_op_id = "nimblecas.logic.sld_shard/v1";

// One unit of distributed work: prove `goals` against `program`, committed to the clause at
// `clause_index`, for at most `max_solutions` answers (0 = all within budget).
struct Shard {
    Program program;
    std::vector<Term> goals;
    std::size_t clause_index{0};
    std::uint64_t max_solutions{0};
};

// ---------------------------------------------------------------------------
// Wire format.
// ---------------------------------------------------------------------------

// Encodes a shard as Prolog text. Fails with `domain_error` on a malformed program or query —
// the same conditions `solve` refuses, checked HERE so a shard is never sent to a worker that
// can only fail on arrival.
[[nodiscard]] auto encode_shard(const Shard& shard) -> Result<Payload>;

// Reads a shard back. `syntax_error` when the bytes are not the text this module writes.
[[nodiscard]] auto decode_shard(std::span<const std::byte> bytes) -> Result<Shard>;

// Encodes answers — a list of substitutions, each a list of variable bindings.
[[nodiscard]] auto encode_answers(const std::vector<Substitution>& answers) -> Result<Payload>;

[[nodiscard]] auto decode_answers(std::span<const std::byte> bytes)
    -> Result<std::vector<Substitution>>;

// ---------------------------------------------------------------------------
// Running.
// ---------------------------------------------------------------------------

// Registers the shard operation. BOTH the coordinator and every worker must call this on their
// own registry at startup: the distributed executor ships an op id and arguments, never code, so
// a worker that has not registered the op cannot run the shard it is handed.
[[nodiscard]] auto register_ops(TaskRegistry& reg) -> Result<void>;

// Whether `goals` against `program` can be fanned out at all. False for a leading builtin or
// control construct (which resolves against no clause), and for a program containing a cut or a
// database update (whose effects a speculative shard cannot take back).
[[nodiscard]] auto is_distributable(const Program& program, const std::vector<Term>& goals)
    -> bool;

// Builds the task graph: one named task per clause of the first goal's predicate, each carrying
// its shard as a bound literal, with no dependencies between them — OR-branches are independent
// by construction, which is exactly why this decomposition is the one worth distributing.
//
// `domain_error` when the query is not distributable; check with `is_distributable` first if you
// want to choose a different strategy rather than be refused.
[[nodiscard]] auto build_shard_graph(const TaskRegistry& reg, const Program& program,
                                     const std::vector<Term>& goals,
                                     std::uint64_t max_solutions) -> Result<TaskGraph>;

// Solves `goals` by running the shard graph on `exec`, then merging the per-clause answers in
// CLAUSE ORDER.
//
// The result is what `solve` returns, and that is the whole point: `exec` may be the serial
// executor, the local parallel one, or a distributed one over a broker, and the answers must not
// be able to tell which. A query that cannot be distributed is run serially rather than refused.
[[nodiscard]] auto solve_distributed(const Program& program, const std::vector<Term>& goals,
                                     std::uint64_t max_solutions, Executor& exec)
    -> Result<std::vector<Substitution>>;

}  // namespace nimblecas::logic_dist

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::logic_dist {

namespace {

using nimblecas::atom_of;
using nimblecas::compound_of;
using nimblecas::int_of;
using nimblecas::is_atom;
using nimblecas::is_compound;
using nimblecas::is_int;
using nimblecas::is_var;
using nimblecas::var_of;

// The generation counter can exceed what a Prolog integer literal holds in principle; in practice
// it is bounded by the step budget. Encoding it as an integer is therefore exact, and this guard
// says so rather than assuming it.
[[nodiscard]] auto generation_fits(std::uint64_t g) -> bool {
    return g <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
}

// A term as its explicit encoding: v(Name, Gen) | a(Name) | i(Value) | c(Functor, [Args]).
[[nodiscard]] auto encode_term(const Term& t) -> Result<Term> {
    if (is_var(t)) {
        const VarNode& v = var_of(t);
        if (!generation_fits(v.generation)) {
            return make_error<Term>(MathError::overflow);
        }
        return make_compound("v", {make_atom(v.name),
                                   make_int(static_cast<std::int64_t>(v.generation))});
    }
    if (is_atom(t)) {
        return make_compound("a", {make_atom(atom_of(t).name)});
    }
    if (is_int(t)) {
        return make_compound("i", {make_int(int_of(t).value)});
    }
    const CompoundNode& c = compound_of(t);
    std::vector<Term> args;
    args.reserve(c.args.size());
    for (const Term& a : c.args) {
        auto e = encode_term(a);
        if (!e) {
            return e;
        }
        args.push_back(*e);
    }
    return make_compound("c", {make_atom(c.functor), make_list(std::move(args))});
}

// The inverse. Anything that is not one of the four shapes is a `syntax_error`: a payload this
// module did not write is not a term with a lenient reading, it is not a term at all.
[[nodiscard]] auto decode_term(const Term& t) -> Result<Term> {
    if (!is_compound(t)) {
        return make_error<Term>(MathError::syntax_error);
    }
    const CompoundNode& c = compound_of(t);
    if (c.functor == "v" && c.args.size() == 2 && is_atom(c.args[0]) && is_int(c.args[1])) {
        const std::int64_t g = int_of(c.args[1]).value;
        if (g < 0) {
            return make_error<Term>(MathError::syntax_error);
        }
        return make_var(atom_of(c.args[0]).name, static_cast<std::uint64_t>(g));
    }
    if (c.functor == "a" && c.args.size() == 1 && is_atom(c.args[0])) {
        return make_atom(atom_of(c.args[0]).name);
    }
    if (c.functor == "i" && c.args.size() == 1 && is_int(c.args[0])) {
        return make_int(int_of(c.args[0]).value);
    }
    if (c.functor == "c" && c.args.size() == 2 && is_atom(c.args[0])) {
        std::vector<Term> args;
        Term cur = c.args[1];
        while (is_compound(cur) && compound_of(cur).functor == "." &&
               compound_of(cur).args.size() == 2) {
            auto d = decode_term(compound_of(cur).args[0]);
            if (!d) {
                return d;
            }
            args.push_back(*d);
            cur = compound_of(cur).args[1];
        }
        if (!is_atom(cur) || atom_of(cur).name != "[]") {
            return make_error<Term>(MathError::syntax_error);  // improper argument list
        }
        if (args.empty()) {
            // `make_compound` folds an empty argument list to an atom, so a 0-arity compound
            // cannot exist and a payload claiming one is malformed rather than merely unusual.
            return make_error<Term>(MathError::syntax_error);
        }
        return make_compound(atom_of(c.args[0]).name, std::move(args));
    }
    return make_error<Term>(MathError::syntax_error);
}

[[nodiscard]] auto encode_clause(const Clause& cl) -> Result<Term> {
    auto head = encode_term(cl.head);
    if (!head) {
        return head;
    }
    std::vector<Term> body;
    body.reserve(cl.body.size());
    for (const Term& g : cl.body) {
        auto e = encode_term(g);
        if (!e) {
            return e;
        }
        body.push_back(*e);
    }
    return make_compound("cl", {*head, make_list(std::move(body))});
}

// Walks a Prolog list term into a vector, or nullopt when it is not a proper list.
[[nodiscard]] auto list_items(const Term& t) -> std::optional<std::vector<Term>> {
    std::vector<Term> out;
    Term cur = t;
    while (is_compound(cur) && compound_of(cur).functor == "." &&
           compound_of(cur).args.size() == 2) {
        out.push_back(compound_of(cur).args[0]);
        cur = compound_of(cur).args[1];
    }
    if (!is_atom(cur) || atom_of(cur).name != "[]") {
        return std::nullopt;
    }
    return out;
}

[[nodiscard]] auto text_to_payload(std::string_view s) -> Payload {
    Payload p;
    p.reserve(s.size());
    for (const char ch : s) {
        p.push_back(static_cast<std::byte>(static_cast<unsigned char>(ch)));
    }
    return p;
}

[[nodiscard]] auto payload_to_text(std::span<const std::byte> bytes) -> std::string {
    std::string s;
    s.reserve(bytes.size());
    for (const std::byte b : bytes) {
        s.push_back(static_cast<char>(std::to_integer<unsigned char>(b)));
    }
    return s;
}

}  // namespace

auto encode_shard(const Shard& shard) -> Result<Payload> {
    // Validated here rather than on arrival: a shard that can only fail at the worker wastes a
    // round trip and reports the failure from the wrong place.
    if (shard.goals.empty()) {
        return make_error<Payload>(MathError::domain_error);
    }
    if (!generation_fits(shard.max_solutions) ||
        shard.clause_index > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        return make_error<Payload>(MathError::overflow);
    }

    std::vector<Term> goals;
    goals.reserve(shard.goals.size());
    for (const Term& g : shard.goals) {
        auto e = encode_term(g);
        if (!e) {
            return make_error<Payload>(e.error());
        }
        goals.push_back(*e);
    }
    std::vector<Term> clauses;
    clauses.reserve(shard.program.size());
    for (const Clause& cl : shard.program) {
        auto e = encode_clause(cl);
        if (!e) {
            return make_error<Payload>(e.error());
        }
        clauses.push_back(*e);
    }

    const Term doc = make_compound(
        "shard", {make_int(static_cast<std::int64_t>(shard.clause_index)),
                  make_int(static_cast<std::int64_t>(shard.max_solutions)),
                  make_list(std::move(goals)), make_list(std::move(clauses))});
    return text_to_payload(logic_parser::to_source(doc));
}

auto decode_shard(std::span<const std::byte> bytes) -> Result<Shard> {
    auto parsed = logic_parser::parse_term(payload_to_text(bytes));
    if (!parsed) {
        return make_error<Shard>(parsed.error());
    }
    if (!is_compound(*parsed) || compound_of(*parsed).functor != "shard" ||
        compound_of(*parsed).args.size() != 4) {
        return make_error<Shard>(MathError::syntax_error);
    }
    const CompoundNode& c = compound_of(*parsed);
    if (!is_int(c.args[0]) || !is_int(c.args[1])) {
        return make_error<Shard>(MathError::syntax_error);
    }
    const std::int64_t idx = int_of(c.args[0]).value;
    const std::int64_t maxs = int_of(c.args[1]).value;
    if (idx < 0 || maxs < 0) {
        return make_error<Shard>(MathError::syntax_error);
    }

    const auto goal_items = list_items(c.args[2]);
    const auto clause_items = list_items(c.args[3]);
    if (!goal_items || !clause_items) {
        return make_error<Shard>(MathError::syntax_error);
    }

    Shard out;
    out.clause_index = static_cast<std::size_t>(idx);
    out.max_solutions = static_cast<std::uint64_t>(maxs);
    out.goals.reserve(goal_items->size());
    for (const Term& g : *goal_items) {
        auto d = decode_term(g);
        if (!d) {
            return make_error<Shard>(d.error());
        }
        out.goals.push_back(*d);
    }
    out.program.reserve(clause_items->size());
    for (const Term& cl : *clause_items) {
        if (!is_compound(cl) || compound_of(cl).functor != "cl" ||
            compound_of(cl).args.size() != 2) {
            return make_error<Shard>(MathError::syntax_error);
        }
        auto head = decode_term(compound_of(cl).args[0]);
        if (!head) {
            return make_error<Shard>(head.error());
        }
        const auto body_items = list_items(compound_of(cl).args[1]);
        if (!body_items) {
            return make_error<Shard>(MathError::syntax_error);
        }
        std::vector<Term> body;
        body.reserve(body_items->size());
        for (const Term& g : *body_items) {
            auto d = decode_term(g);
            if (!d) {
                return make_error<Shard>(d.error());
            }
            body.push_back(*d);
        }
        out.program.push_back(Clause{.head = *head, .body = std::move(body)});
    }
    return out;
}

auto encode_answers(const std::vector<Substitution>& answers) -> Result<Payload> {
    std::vector<Term> subs;
    subs.reserve(answers.size());
    for (const Substitution& s : answers) {
        std::vector<Term> bindings;
        bindings.reserve(s.size());
        for (const auto& [key, value] : s) {
            if (!generation_fits(key.generation)) {
                return make_error<Payload>(MathError::overflow);
            }
            auto v = encode_term(value);
            if (!v) {
                return make_error<Payload>(v.error());
            }
            bindings.push_back(make_compound(
                "bnd", {make_atom(key.name),
                        make_int(static_cast<std::int64_t>(key.generation)), *v}));
        }
        subs.push_back(make_compound("sub", {make_list(std::move(bindings))}));
    }
    return text_to_payload(
        logic_parser::to_source(make_compound("answers", {make_list(std::move(subs))})));
}

auto decode_answers(std::span<const std::byte> bytes) -> Result<std::vector<Substitution>> {
    auto parsed = logic_parser::parse_term(payload_to_text(bytes));
    if (!parsed) {
        return make_error<std::vector<Substitution>>(parsed.error());
    }
    if (!is_compound(*parsed) || compound_of(*parsed).functor != "answers" ||
        compound_of(*parsed).args.size() != 1) {
        return make_error<std::vector<Substitution>>(MathError::syntax_error);
    }
    const auto subs = list_items(compound_of(*parsed).args[0]);
    if (!subs) {
        return make_error<std::vector<Substitution>>(MathError::syntax_error);
    }

    std::vector<Substitution> out;
    out.reserve(subs->size());
    for (const Term& s : *subs) {
        if (!is_compound(s) || compound_of(s).functor != "sub" ||
            compound_of(s).args.size() != 1) {
            return make_error<std::vector<Substitution>>(MathError::syntax_error);
        }
        const auto bindings = list_items(compound_of(s).args[0]);
        if (!bindings) {
            return make_error<std::vector<Substitution>>(MathError::syntax_error);
        }
        Substitution sub;
        sub.reserve(bindings->size());
        for (const Term& b : *bindings) {
            if (!is_compound(b) || compound_of(b).functor != "bnd" ||
                compound_of(b).args.size() != 3 || !is_atom(compound_of(b).args[0]) ||
                !is_int(compound_of(b).args[1])) {
                return make_error<std::vector<Substitution>>(MathError::syntax_error);
            }
            const std::int64_t gen = int_of(compound_of(b).args[1]).value;
            if (gen < 0) {
                return make_error<std::vector<Substitution>>(MathError::syntax_error);
            }
            auto value = decode_term(compound_of(b).args[2]);
            if (!value) {
                return make_error<std::vector<Substitution>>(value.error());
            }
            sub.emplace_back(VarKey{.name = atom_of(compound_of(b).args[0]).name,
                                    .generation = static_cast<std::uint64_t>(gen)},
                             *value);
        }
        out.push_back(std::move(sub));
    }
    return out;
}

auto register_ops(TaskRegistry& reg) -> Result<void> {
    // The shard body: decode, solve the one branch, encode the answers. It is a pure function of
    // its argument bytes, which is what `TaskFn` requires and what makes it safe to run on a
    // worker that shares nothing with the coordinator.
    return reg.register_op(OpId(shard_op_id), [](std::span<const Payload> args)
                                                  -> Result<Payload> {
        if (args.size() != 1) {
            return make_error<Payload>(MathError::domain_error);
        }
        auto shard = decode_shard(args[0]);
        if (!shard) {
            return make_error<Payload>(shard.error());
        }
        auto answers = solve_clause_branch(shard->program, shard->goals, shard->clause_index,
                                           shard->max_solutions);
        if (!answers) {
            // The branch's own honest failure — a floundering negation, an exhausted budget —
            // travels back as itself rather than as a transport error, so the coordinator can
            // tell "this shard could not be answered" from "this shard never arrived".
            return make_error<Payload>(answers.error());
        }
        return encode_answers(*answers);
    });
}

auto is_distributable(const Program& program, const std::vector<Term>& goals) -> bool {
    if (goals.empty()) {
        return false;
    }
    // `solve_or_parallel` already decides exactly this question for the shared-memory case, and
    // asking it here keeps one answer rather than two that can disagree. It returns the same
    // answers as `solve` when it declines to fan out, so a disagreement would be invisible until
    // it mattered — which is the worst kind.
    const auto probe = solve_or_parallel(program, goals, 1);
    if (!probe) {
        return false;  // a malformed or unsound query is not a distribution candidate
    }
    return true;
}

auto build_shard_graph(const TaskRegistry& reg, const Program& program,
                       const std::vector<Term>& goals, std::uint64_t max_solutions)
    -> Result<TaskGraph> {
    if (goals.empty()) {
        return make_error<TaskGraph>(MathError::domain_error);
    }
    TaskGraph g;
    for (std::size_t ci = 0; ci < program.size(); ++ci) {
        auto payload = encode_shard(Shard{.program = program,
                                          .goals = goals,
                                          .clause_index = ci,
                                          .max_solutions = max_solutions});
        if (!payload) {
            return make_error<TaskGraph>(payload.error());
        }
        std::vector<Payload> literals;
        literals.push_back(std::move(*payload));
        // No dependencies: OR-branches are independent by construction, which is precisely the
        // property that makes this the decomposition worth distributing. Every task lands at
        // depth 0, so a wavefront executor runs the whole graph in one level.
        auto id = g.add_named_task(reg, OpId(shard_op_id), std::move(literals), {});
        if (!id) {
            return make_error<TaskGraph>(id.error());
        }
    }
    return g;
}

auto solve_distributed(const Program& program, const std::vector<Term>& goals,
                       std::uint64_t max_solutions, Executor& exec)
    -> Result<std::vector<Substitution>> {
    if (!is_distributable(program, goals)) {
        // Not a failure: a query with a cut, a database update, or a leading builtin has answers,
        // they simply cannot be reached by fanning out over clauses. Callers rely on the ANSWERS
        // being what `solve` gives; refusing would break that contract to protect a strategy.
        return solve(program, goals, max_solutions);
    }

    TaskRegistry reg;
    auto registered = register_ops(reg);
    if (!registered) {
        return make_error<std::vector<Substitution>>(registered.error());
    }
    auto graph = build_shard_graph(reg, program, goals, max_solutions);
    if (!graph) {
        return make_error<std::vector<Substitution>>(graph.error());
    }
    auto run = exec.run(*graph);
    if (!run) {
        return make_error<std::vector<Substitution>>(run.error());
    }

    // Merge in CLAUSE ORDER — `outputs[i]` is task i's result, and task i is clause i — so the
    // answers do not depend on which worker finished first, or on how many workers there were.
    //
    // The cap is consulted BEFORE a shard's error, for the reason `solve_or_parallel` documents:
    // serial resolution stops the instant it has enough answers and never tries the later
    // clauses, so a shard serial would not have reached must not be able to fail the query. Every
    // shard is evaluated speculatively — that is what makes it distributable — and a speculative
    // failure is not a result.
    std::vector<Substitution> out;
    for (const Result<Payload>& payload : run->outputs) {
        if (max_solutions != 0 && out.size() >= max_solutions) {
            return out;
        }
        if (!payload) {
            return make_error<std::vector<Substitution>>(payload.error());
        }
        auto answers = decode_answers(*payload);
        if (!answers) {
            return make_error<std::vector<Substitution>>(answers.error());
        }
        for (Substitution& s : *answers) {
            if (max_solutions != 0 && out.size() >= max_solutions) {
                return out;
            }
            out.push_back(std::move(s));
        }
    }
    return out;
}

}  // namespace nimblecas::logic_dist
