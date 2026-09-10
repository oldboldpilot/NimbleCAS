// NimbleCAS logic programming — unification and SLD resolution over Horn clauses (OR-parallel).
// @author Olumuyiwa Oluwasanmi
//
// A small Prolog core: first-order terms, Robinson unification WITH occurs-check, and SLD
// resolution (depth-first, backtracking) over definite Horn clauses, plus an OR-parallel
// solver. Everything is deterministic: clauses are tried in program order, subgoals
// left-to-right, and variables are standardised-apart by a rename counter threaded
// explicitly through the search (never a global or random source), so a given
// program+query always enumerates the same answers in the same order.
//
// Terms are immutable values shared via CowPtr (nimblecas.core, Rule 22): copying a Term is
// an O(1) refcount bump and read-only sharing across threads is safe — which is exactly
// what makes the OR-parallel decomposition below sound (branches only READ shared terms and
// produce new ones; no handle is ever mutated via write()).
//
// HONESTY / SCOPE (deliberate limitations, documented rather than hidden):
//  * SLD resolution is only semi-decidable: a left-recursive rule or an infinite derivation
//    can diverge. Two budgets bound the search so it ALWAYS terminates and returns the
//    solutions found within budget — this is NOT a completeness claim. `default_step_budget`
//    caps the total number of resolution attempts, and `max_derivation_depth` caps the
//    length of any single derivation (which also keeps the native recursion bounded, so a
//    runaway program can never overflow the C++ stack). A query with NO solutions is an
//    EMPTY answer list, never an error; only a malformed program/query (a non-callable head
//    or goal — a bare variable or integer where a predicate is required) is a domain_error.
//  * Clause bodies are definite Horn goals plus NEGATION AS FAILURE (`\+ G`, below). Cut,
//    arithmetic evaluation, and the other impure Prolog features are intentionally NOT
//    implemented.
//
// NEGATION AS FAILURE (`\+ G`) — AND THE TWO PLACES REAL PROLOG LIES
//
// `\+ G` succeeds iff G is NOT PROVABLE from the program. That is emphatically not "G is
// false": under the closed-world assumption the two coincide, and outside it they do not.
// The name is the honest one, and the two ways the rule breaks are handled rather than
// papered over — each is a textbook instance of Rule 32 (never return a plausible-looking
// wrong value).
//
//  1. FLOUNDERING (a non-ground negated goal). `\+ G` is only sound when G is GROUND at the
//     moment it is called. With the single fact `p(1)`, standard Prolog answers the query
//     `\+ p(X)` with FAILURE — yet `∃X ¬p(X)` is plainly TRUE (X = 2 witnesses it). Worse,
//     the answer depends on goal order: `\+ p(X), X = 2` fails while `X = 2, \+ p(X)`
//     succeeds, so conjunction stops being commutative. Prolog returns that answer with no
//     indication anything went wrong. This solver instead REFUSES: a negated goal that is
//     still non-ground after the current substitution is applied returns domain_error. A
//     wrong answer that looks right is the one outcome worth failing loudly to avoid.
//
//  2. NEGATION AS *BUDGET EXHAUSTION*. The budgets above make the search always terminate,
//     which means "found no solution" has two very different causes: the space was searched
//     out, or the search was cut off. Only the first licenses concluding "not provable". If
//     the sub-search for G is truncated by the step or depth budget without finding a
//     solution, concluding `\+ G` would silently upgrade "I ran out of time" into "it is not
//     provable" — and, because the budgets are a resource limit rather than a property of the
//     program, the same query could then answer differently on a different day. That case
//     returns not_converged instead. This is the negation-specific counterpart of the
//     semi-decidability caveat above: a truncated POSITIVE search honestly under-reports
//     answers, but a truncated NEGATIVE one would INVENT them.
//
// Both conditions are errors rather than silent failures, so `\+` never contributes an answer
// the solver cannot stand behind. Negation binds nothing: on success the substitution is
// unchanged (there is no witness to bind), which is why `\+` is a test, not a generator.
//
// OR-PARALLELISM: the natural parallel/distributed decomposition of SLD resolution is
// OR-parallelism — the independent clause branches for a goal. `solve_or_parallel` maps the
// first goal's candidate clauses across workers with `parallel::transform_index`; each
// branch is a stateless continuation carrying its OWN copy of the substitution and rename
// counter (no shared mutable state), and the branch results are concatenated in clause order.
// The output is identical to the serial `solve` WHENEVER the enumeration completes within the
// budgets — the normal terminating case. Note the honest caveat: the step/depth budgets are a
// PER-BRANCH safety cap here, not a single shared quota, so a query that actually EXHAUSTS its
// step budget mid-search may return additional solutions in parallel that serial `solve`'s one
// global budget would have cut off. "One branch per clause" is exactly the "one shard per
// clause" shape a distributed engine would use.
// SLD's AND-side backtracking, by contrast, is irregular and data-dependent — a poor fit for
// SIMT/GPU execution — so this solver targets CPU/distributed parallelism only and
// deliberately ships no CUDA path.

export module nimblecas.logic;

import std;
import nimblecas.core;
import nimblecas.parallel;

export namespace nimblecas {

// A Term is one of: a logic Variable, an Atom (symbolic constant), an Integer constant, or a
// Compound (functor + argument terms). The variant is wrapped in a struct so Term can hold a
// CowPtr<TermNode> while TermNode is still incomplete (a variant alias cannot be
// forward-declared).
struct TermNode;

// ---------------------------------------------------------------------------
// Term — a copy-on-write handle to an immutable first-order term node.
// ---------------------------------------------------------------------------
class Term {
public:
    // Wraps a fully-formed node; prefer the make_* free factories below.
    explicit Term(TermNode value);

    [[nodiscard]] auto node() const -> const TermNode& { return node_.read(); }

private:
    CowPtr<TermNode> node_;
};

// ---------------------------------------------------------------------------
// Node kinds.
// ---------------------------------------------------------------------------

// A logic variable is identified by NAME plus a rename GENERATION. Standardising a clause
// apart stamps a fresh generation on every variable, so the same source name reused across
// clause activations yields distinct variables while staying consistent within one clause.
struct VarNode {
    std::string name;
    std::uint64_t generation;
};

// A symbolic constant, e.g. `tom` or the list terminator `[]`.
struct AtomNode {
    std::string name;
};

// An integer constant.
struct IntNode {
    std::int64_t value;
};

// A functor applied to one or more argument terms, e.g. parent(tom, bob). A zero-argument
// compound is normalised to an Atom (see make_compound).
struct CompoundNode {
    std::string functor;
    std::vector<Term> args;
};

struct TermNode {
    std::variant<VarNode, AtomNode, IntNode, CompoundNode> value;
};

// ---------------------------------------------------------------------------
// Kind predicates and typed accessors.
// ---------------------------------------------------------------------------
//
// Exported because every consumer that builds or walks terms needs them — the parser
// module, and any caller inspecting an answer. The accessors have a PRECONDITION: call
// `atom_of(t)` only after `is_atom(t)` returned true.

// Kind predicates and typed accessors over a Term's node variant.
[[nodiscard]] auto is_var(const Term& t) -> bool {
    return std::holds_alternative<VarNode>(t.node().value);
}
[[nodiscard]] auto is_atom(const Term& t) -> bool {
    return std::holds_alternative<AtomNode>(t.node().value);
}
[[nodiscard]] auto is_int(const Term& t) -> bool {
    return std::holds_alternative<IntNode>(t.node().value);
}
[[nodiscard]] auto is_compound(const Term& t) -> bool {
    return std::holds_alternative<CompoundNode>(t.node().value);
}
[[nodiscard]] auto var_of(const Term& t) -> const VarNode& {
    return std::get<VarNode>(t.node().value);
}
[[nodiscard]] auto atom_of(const Term& t) -> const AtomNode& {
    return std::get<AtomNode>(t.node().value);
}
[[nodiscard]] auto int_of(const Term& t) -> const IntNode& {
    return std::get<IntNode>(t.node().value);
}
[[nodiscard]] auto compound_of(const Term& t) -> const CompoundNode& {
    return std::get<CompoundNode>(t.node().value);
}

// A goal or clause head must denote a predicate: an atom (0-arity) or a compound. A bare
// variable or integer is not callable and marks the program/query as malformed.
[[nodiscard]] auto is_callable(const Term& t) -> bool { return is_atom(t) || is_compound(t); }

// A '.'/2 cons cell used to encode lists.
[[nodiscard]] auto is_cons(const Term& t) -> bool {
    if (!is_compound(t)) {
        return false;
    }
    const CompoundNode& c = compound_of(t);
    return c.functor == "." && c.args.size() == 2;
}
[[nodiscard]] auto is_nil(const Term& t) -> bool {
    return is_atom(t) && atom_of(t).name == "[]";
}

// ---------------------------------------------------------------------------
// Term factories (Prolog-style constructors).
// ---------------------------------------------------------------------------

// A logic variable. `generation` defaults to 0 (the "source" generation); standardise-apart
// overwrites it with the solver's rename counter.
[[nodiscard]] auto make_var(std::string name, std::uint64_t generation = 0) -> Term;
// A symbolic constant.
[[nodiscard]] auto make_atom(std::string name) -> Term;
// An integer constant.
[[nodiscard]] auto make_int(std::int64_t value) -> Term;
// A compound term; with empty args it collapses to make_atom(functor) since a 0-arg compound
// is just an atom.
[[nodiscard]] auto make_compound(std::string functor, std::vector<Term> args) -> Term;

// The empty list, i.e. the atom `[]`.
[[nodiscard]] auto make_nil() -> Term;
// A proper list `[e0, e1, ...]` encoded as nested '.'/2 cons cells terminated by make_nil().
[[nodiscard]] auto make_list(std::vector<Term> elements) -> Term;

// The reserved functor of negation as failure. Deliberately `\+` — the ISO operator — because
// it is not a name a user could define a predicate with by accident. `not` is NOT treated as a
// synonym: silently reinterpreting a user's own `not/1` predicate as negation is exactly the
// kind of quiet meaning change Rule 32 forbids.
inline constexpr std::string_view naf_functor = "\\+";

// The negated goal `\+ G`: succeeds iff G is not provable. See the header for the two
// conditions (floundering, budget exhaustion) that make this an ERROR rather than a failure.
[[nodiscard]] auto make_not(Term goal) -> Term;

// Whether `t` is a well-formed negated goal `\+`/1.
[[nodiscard]] auto is_negation(const Term& t) -> bool;

// Whether `t` contains no variables. A negated goal must be ground when called for its answer
// to be sound, so this is the predicate the solver gates `\+` on.
[[nodiscard]] auto is_ground(const Term& t) -> bool;

// Structural (syntactic) equality of terms: identical trees, with variables equal iff both
// name and generation match.
[[nodiscard]] auto operator==(const Term& a, const Term& b) -> bool;
[[nodiscard]] auto operator!=(const Term& a, const Term& b) -> bool;

// Prolog-style rendering: atoms/integers verbatim, compounds as f(a, b), and '.'/2 spines as
// list syntax [a, b] (or [a | Tail] when improper). A variable prints as its name, suffixed
// with _<generation> once standardised apart.
[[nodiscard]] auto to_string(const Term& t) -> std::string;

// ---------------------------------------------------------------------------
// Substitutions, clauses, and programs.
// ---------------------------------------------------------------------------

// The identity of a logic variable: name plus rename generation.
struct VarKey {
    std::string name;
    std::uint64_t generation;
};

[[nodiscard]] auto operator==(const VarKey& a, const VarKey& b) -> bool {
    return a.generation == b.generation && a.name == b.name;
}

// An ordered variable -> term binding set. A std::vector of pairs (not a hash map) keeps
// iteration order deterministic, which the solver relies on for reproducible answers.
// The identity of the variable `t`. Precondition: `is_var(t)`.
[[nodiscard]] auto key_of(const Term& t) -> VarKey {
    const VarNode& v = var_of(t);
    return VarKey{.name = v.name, .generation = v.generation};
}

using Substitution = std::vector<std::pair<VarKey, Term>>;

// A definite Horn clause `head :- body[0], body[1], ...`. A fact has an empty body.
struct Clause {
    Term head;
    std::vector<Term> body;
};

using Program = std::vector<Clause>;

// ---------------------------------------------------------------------------
// Budgets that guarantee termination (see the semi-decidability note in the header).
// ---------------------------------------------------------------------------

// Maximum number of clause-resolution attempts across an entire search. Exhausting it stops
// the search and returns whatever was found so far (not a completeness guarantee).
inline constexpr std::uint64_t default_step_budget = 1'000'000;

// Maximum length of any single SLD derivation. Also bounds the native recursion depth so a
// non-terminating program cannot overflow the C++ stack.
inline constexpr std::uint64_t max_derivation_depth = 1'000;

// ---------------------------------------------------------------------------
// Predicate indicators and the builtin table.
// ---------------------------------------------------------------------------

// A predicate is named by functor and arity: `parent/2`. Rendered as "name/arity", which is
// also the key the clause index is bucketed by.
[[nodiscard]] auto indicator(std::string_view name, std::size_t arity) -> std::string;

// The predicate indicator of a callable term, or nullopt when `t` is not callable.
[[nodiscard]] auto indicator_of(const Term& t) -> std::optional<std::string>;

// Whether `name`/`arity` is a builtin this engine defines itself. A program may not supply
// clauses for one: two meanings for the same goal, chosen by whichever was tried first, is
// exactly the silent ambiguity Rule 32 exists to prevent. `validate_program` rejects such a
// clause, and `assert/1` refuses it at runtime with the same error.
[[nodiscard]] auto is_builtin(std::string_view name, std::size_t arity) -> bool;

// Every builtin indicator this engine implements, sorted, for documentation and for a caller
// that wants to check a program against the reserved set before running it.
[[nodiscard]] auto builtin_indicators() -> std::vector<std::string>;

// ---------------------------------------------------------------------------
// Standard order of terms.
// ---------------------------------------------------------------------------

// Three-way comparison in the ISO standard order: Var < Integer < Atom < Compound, integers
// by value, atoms by name, compounds by ARITY then functor then arguments left to right.
// Variables are ordered by (generation, name), which is total and deterministic — that is what
// lets `sort/2` remove duplicates reproducibly. Returns a negative value, zero, or a positive
// value in the manner of strcmp.
[[nodiscard]] auto compare_terms(const Term& a, const Term& b) -> int;

// ---------------------------------------------------------------------------
// The clause database.
// ---------------------------------------------------------------------------

// An indexed, mutable clause store.
//
// Two things distinguish it from the plain `Program` vector it is built from:
//
//  * FIRST-ARGUMENT INDEXING. Clauses are bucketed by predicate indicator and, within a
//    predicate, by the principal functor of the first head argument. A call with a bound first
//    argument only ever considers clauses whose first argument could unify with it, plus those
//    whose first argument is a variable. This is a pure performance property: candidates are
//    always returned in clause order, so the answers and their order are exactly what an
//    unindexed linear scan produces.
//
//  * STABLE CLAUSE IDENTITY. `retract` marks a clause dead rather than erasing it, and
//    `asserta` orders a clause ahead of the others rather than shifting them. A clause's
//    internal index therefore never moves, which is what makes it safe to mutate the database
//    from inside a running search — an active clause loop is holding indices.
//
// Copying a Database copies its clauses; the type is a value, and two copies are independent.
class Database {
public:
    Database() = default;

    // A database over `p`, keeping program order.
    [[nodiscard]] static auto from(Program p) -> Database;

    // A copy with `c` appended (fluent; the receiver is unchanged).
    [[nodiscard]] auto with_clause(Clause c) const -> Database;

    // Adds `c` as the LAST clause of its predicate. Rejects a clause for a builtin.
    auto assertz(Clause c) -> Result<void>;
    // Adds `c` as the FIRST clause of its predicate. Rejects a clause for a builtin.
    auto asserta(Clause c) -> Result<void>;

    // The live clauses, in clause order.
    [[nodiscard]] auto clauses() const -> Program;
    // How many clauses are live.
    [[nodiscard]] auto size() const -> std::size_t;
    // Whether any clause is defined for this indicator (used to tell an UNDEFINED predicate
    // apart from one that is defined and simply has no matching clause).
    [[nodiscard]] auto is_defined(std::string_view ind) const -> bool;

private:
    friend struct DatabaseAccess;
    struct Entry {
        Clause clause;
        std::int64_t order{};
        bool live{true};
    };
    std::vector<Entry> entries_{};
    std::int64_t next_high_{0};
    std::int64_t next_low_{0};
};

// ---------------------------------------------------------------------------
// Solvers.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Core operations.
// ---------------------------------------------------------------------------

// Robinson unification WITH occurs-check, applying the incoming substitution `in` as it goes.
// Returns the extended most-general-unifier substitution, or std::nullopt when the terms do
// not unify. The occurs-check failure (e.g. unifying X with f(X)) is a normal non-unification
// — it yields nullopt, NOT a MathError. The Result wrapper is reserved for genuine errors and
// is presently always the value branch.
[[nodiscard]] auto unify(const Term& a, const Term& b, const Substitution& in)
    -> Result<std::optional<Substitution>>;

// Resolves every variable in `t` against `s` to a fixed point, rebuilding compounds with
// their arguments resolved. Occurs-check keeps bindings acyclic so this terminates; a depth
// cap is a defensive backstop.
[[nodiscard]] auto apply_substitution(const Substitution& s, const Term& t) -> Term;

// SLD resolution over `program` for the conjunction `goals`, depth-first with backtracking,
// standardising each clause apart per activation via a deterministic rename counter. Answers
// are enumerated in a fixed order (clauses in program order, subgoals left-to-right) up to
// `max_solutions` (0 = all reachable within the budgets). Each returned substitution binds the
// query's variables (restricted to them; any residual internal variables are canonicalised so
// the output does not depend on the internal rename counter). A query with no solutions
// returns an empty vector. A malformed program/query returns domain_error.
//
// The query runs against its OWN copy of the program, so `assert/1` and `retract/1` are visible
// for the remainder of the query and then discarded. Hold a `Machine` for updates that persist
// across queries — a solver handed a `const Program&` cannot honestly claim to have changed it.
[[nodiscard]] auto solve(const Program& program, const std::vector<Term>& goals,
                         std::uint64_t max_solutions) -> Result<std::vector<Substitution>>;

// The first solution to `goals`, or std::nullopt if there is none within budget.
[[nodiscard]] auto solve_first(const Program& program, const std::vector<Term>& goals)
    -> Result<std::optional<Substitution>>;

// As `solve`, but against a database the caller owns, so database updates PERSIST in `db`.
[[nodiscard]] auto solve_in(Database& db, const std::vector<Term>& goals,
                            std::uint64_t max_solutions) -> Result<std::vector<Substitution>>;

// OR-parallel SLD resolution: the first goal's candidate clauses are explored as independent
// branches via parallel::transform_index (one branch per clause, each a stateless continuation
// with its own substitution, rename counter and database copy), and the branch results are
// concatenated in clause order. Byte-for-byte identical to solve() whenever the search completes
// within budget; because each branch carries its own (not a shared) step/depth budget, a query
// that exhausts the step budget mid-search may enumerate further here than serial solve() would.
//
// A query whose program contains a CUT or a DATABASE UPDATE is run serially instead. Both break
// the decomposition: a cut commits to one clause after the others have already been evaluated,
// and a side effect performed by a branch serial resolution would never have reached cannot be
// un-performed by the merge. Delegating keeps the contract callers rely on — the same answers
// as `solve` — rather than trading it for parallelism.
// Solves `goals` committed to ONE clause of the first goal's predicate.
//
// This is the unit of work OR-parallelism fans out over, and — because it is a pure function of
// its arguments with no shared state — it is also the unit a DISTRIBUTED shard can carry to
// another process. `solve_or_parallel` and `nimblecas.logic_dist` both call it, so the local and
// distributed decompositions cannot drift apart.
//
// Answers are restricted to the query's variables exactly as `solve` does, so concatenating
// every clause's answers IN CLAUSE ORDER yields precisely what `solve` returns. A `clause_index`
// out of range, or naming a clause for a different predicate, gives an EMPTY answer list rather
// than an error: not being an alternative is not a failure.
//
// The budget is PER BRANCH here, not a share of one global quota — the same honest caveat
// `solve_or_parallel` documents, and for the same reason: a branch cannot know what the others
// have spent without talking to them, and talking to them is what the decomposition avoids.
[[nodiscard]] auto solve_clause_branch(const Program& program, const std::vector<Term>& goals,
                                       std::size_t clause_index, std::uint64_t max_solutions)
    -> Result<std::vector<Substitution>>;

[[nodiscard]] auto solve_or_parallel(const Program& program, const std::vector<Term>& goals,
                                     std::uint64_t max_solutions)
    -> Result<std::vector<Substitution>>;

// ---------------------------------------------------------------------------
// Machine — a database that persists across queries.
// ---------------------------------------------------------------------------

// Owns a clause database, so `assert/1` and `retract/1` performed by one query are visible to
// the next. This is the only place in the module where a query's effects outlive it, and it is
// a separate type precisely so that the free `solve` functions can keep their pure reading.
class Machine {
public:
    [[nodiscard]] static auto create() -> Machine;
    [[nodiscard]] static auto from(Program p) -> Machine;

    // A copy with `c` appended (fluent; the receiver is unchanged).
    [[nodiscard]] auto with_clause(Clause c) const -> Machine;

    // Adds every clause of `p`, rejecting one that redefines a builtin.
    auto consult(Program p) -> Result<void>;

    // Runs a query against this machine's database, which the query may update.
    [[nodiscard]] auto solve(const std::vector<Term>& goals, std::uint64_t max_solutions)
        -> Result<std::vector<Substitution>>;
    [[nodiscard]] auto solve_first(const std::vector<Term>& goals)
        -> Result<std::optional<Substitution>>;

    [[nodiscard]] auto database() const -> const Database&;

private:
    Database db_{};
};

}  // namespace nimblecas

// ===========================================================================
// Implementation (defined once TermNode is a complete type; not re-exported).
// ===========================================================================
namespace nimblecas {


Term::Term(TermNode value) : node_(CowPtr<TermNode>::make(std::move(value))) {}

auto make_var(std::string name, std::uint64_t generation) -> Term {
    return Term(TermNode{.value = VarNode{.name = std::move(name), .generation = generation}});
}

auto make_atom(std::string name) -> Term {
    return Term(TermNode{.value = AtomNode{.name = std::move(name)}});
}

auto make_int(std::int64_t value) -> Term {
    return Term(TermNode{.value = IntNode{.value = value}});
}

auto make_compound(std::string functor, std::vector<Term> args) -> Term {
    if (args.empty()) {
        return make_atom(std::move(functor));  // a 0-arg compound is just an atom
    }
    return Term(TermNode{
        .value = CompoundNode{.functor = std::move(functor), .args = std::move(args)}});
}

auto make_nil() -> Term { return make_atom("[]"); }

auto make_list(std::vector<Term> elements) -> Term {
    Term acc = make_nil();
    // Fold from the back so element order is preserved in the cons spine.
    for (std::size_t i = elements.size(); i-- > 0;) {
        acc = make_compound(".", {elements[i], std::move(acc)});
    }
    return acc;
}

auto make_not(Term goal) -> Term {
    // Built directly rather than via make_compound so a 0-arg inner goal cannot collapse the
    // negation itself into an atom (make_compound folds empty args to an atom).
    return Term(TermNode{
        .value = CompoundNode{.functor = std::string(naf_functor), .args = {std::move(goal)}}});
}

auto is_negation(const Term& t) -> bool {
    if (!is_compound(t)) {
        return false;
    }
    const CompoundNode& c = compound_of(t);
    return c.functor == naf_functor && c.args.size() == 1;
}

auto is_ground(const Term& t) -> bool {
    if (is_var(t)) {
        return false;
    }
    if (is_compound(t)) {
        return std::ranges::all_of(compound_of(t).args,
                                   [](const Term& a) -> bool { return is_ground(a); });
    }
    return true;  // atoms and integers
}

auto operator==(const Term& a, const Term& b) -> bool {
    if (a.node().value.index() != b.node().value.index()) {
        return false;
    }
    if (is_var(a)) {
        const VarNode& x = var_of(a);
        const VarNode& y = var_of(b);
        return x.name == y.name && x.generation == y.generation;
    }
    if (is_atom(a)) {
        return atom_of(a).name == atom_of(b).name;
    }
    if (is_int(a)) {
        return int_of(a).value == int_of(b).value;
    }
    const CompoundNode& x = compound_of(a);
    const CompoundNode& y = compound_of(b);
    if (x.functor != y.functor || x.args.size() != y.args.size()) {
        return false;
    }
    for (std::size_t i = 0; i < x.args.size(); ++i) {
        if (!(x.args[i] == y.args[i])) {
            return false;
        }
    }
    return true;
}

auto operator!=(const Term& a, const Term& b) -> bool { return !(a == b); }

namespace {

// Renders a '.'/2 spine as [a, b, ...] (proper) or [a, b | Tail] (improper).
[[nodiscard]] auto list_to_string(const Term& t) -> std::string {
    std::string out = "[";
    Term cur = t;
    bool first = true;
    while (is_cons(cur)) {
        const CompoundNode& c = compound_of(cur);
        if (!first) {
            out += ", ";
        }
        out += to_string(c.args[0]);
        first = false;
        cur = c.args[1];
    }
    if (is_nil(cur)) {
        out += ']';
    } else {
        out += " | " + to_string(cur) + "]";
    }
    return out;
}

}  // namespace

auto to_string(const Term& t) -> std::string {
    if (is_var(t)) {
        const VarNode& v = var_of(t);
        return v.generation == 0 ? v.name : std::format("{}_{}", v.name, v.generation);
    }
    if (is_atom(t)) {
        return atom_of(t).name;
    }
    if (is_int(t)) {
        return std::format("{}", int_of(t).value);
    }
    const CompoundNode& c = compound_of(t);
    if (is_cons(t)) {
        return list_to_string(t);
    }
    std::string args;
    for (std::size_t i = 0; i < c.args.size(); ++i) {
        if (i != 0) {
            args += ", ";
        }
        args += to_string(c.args[i]);
    }
    return std::format("{}({})", c.functor, args);
}

namespace {

// The current binding of variable key `k` in `s`, or nullopt. Linear scan over the ordered
// vector — fine for the small substitutions these searches produce, and keeps iteration order
// deterministic.
[[nodiscard]] auto find_binding(const Substitution& s, const VarKey& k) -> std::optional<Term> {
    for (const auto& [key, val] : s) {
        if (key == k) {
            return val;
        }
    }
    return std::nullopt;
}

// Follows variable bindings at the TOP of `t` to a fixed point (triangular substitution): the
// result is either a non-variable or an unbound variable. Occurs-check keeps bindings acyclic,
// so this loop terminates; the guard is a defensive backstop.
[[nodiscard]] auto walk(const Term& t, const Substitution& s) -> Term {
    Term cur = t;
    std::size_t guard = 0;
    while (is_var(cur)) {
        auto bound = find_binding(s, key_of(cur));
        if (!bound) {
            break;
        }
        cur = *bound;
        if (++guard > s.size() + 1) {
            break;  // unreachable given occurs-check; prevents a hang if it were ever violated
        }
    }
    return cur;
}

// occurs(v, t, s): does variable `v` occur in `t` once resolved through `s`? Guards the
// occurs-check that keeps unification sound (no cyclic bindings).
[[nodiscard]] auto occurs(const VarKey& v, const Term& t, const Substitution& s) -> bool {
    const Term r = walk(t, s);
    if (is_var(r)) {
        return key_of(r) == v;
    }
    if (is_compound(r)) {
        const CompoundNode& c = compound_of(r);
        for (const Term& arg : c.args) {
            if (occurs(v, arg, s)) {
                return true;
            }
        }
    }
    return false;
}

// Robinson unification core (occurs-checked). Returns the extended substitution or nullopt.
[[nodiscard]] auto unify_terms(const Term& a0, const Term& b0, const Substitution& s)
    -> std::optional<Substitution> {
    const Term a = walk(a0, s);
    const Term b = walk(b0, s);

    const bool av = is_var(a);
    const bool bv = is_var(b);
    if (av && bv && key_of(a) == key_of(b)) {
        return s;  // same variable — already unified
    }
    if (av) {
        if (occurs(key_of(a), b, s)) {
            return std::nullopt;  // occurs-check failure
        }
        Substitution next = s;
        next.emplace_back(key_of(a), b);
        return next;
    }
    if (bv) {
        if (occurs(key_of(b), a, s)) {
            return std::nullopt;
        }
        Substitution next = s;
        next.emplace_back(key_of(b), a);
        return next;
    }
    // Both are non-variables: they unify only when the same kind and content matches.
    if (is_atom(a) && is_atom(b)) {
        return atom_of(a).name == atom_of(b).name ? std::optional<Substitution>(s)
                                                  : std::nullopt;
    }
    if (is_int(a) && is_int(b)) {
        return int_of(a).value == int_of(b).value ? std::optional<Substitution>(s)
                                                  : std::nullopt;
    }
    if (is_compound(a) && is_compound(b)) {
        const CompoundNode& ca = compound_of(a);
        const CompoundNode& cb = compound_of(b);
        if (ca.functor != cb.functor || ca.args.size() != cb.args.size()) {
            return std::nullopt;
        }
        Substitution cur = s;
        for (std::size_t i = 0; i < ca.args.size(); ++i) {
            auto u = unify_terms(ca.args[i], cb.args[i], cur);
            if (!u) {
                return std::nullopt;
            }
            cur = std::move(*u);
        }
        return cur;
    }
    return std::nullopt;  // mismatched kinds (atom vs int, atom vs compound, ...)
}

// Substitution application. Recursion follows the TERM structure, so its depth is the term's
// depth; the binding chain is followed ITERATIVELY, because a chain is as long as the search
// made bindings and walking it on the native stack would be unbounded.
//
// The SELF-BINDING is the case that matters. `restrict_answer` records an unbound query
// variable as bound to ITSELF, which is how an answer says "no value". Following that
// recursively spins until the depth cap — a hundred thousand frames, some twenty megabytes,
// long past any stack — so `apply_substitution(answer, X)` on an ordinary answer with an
// unbound variable would take the process down. Stopping at a self-binding is not a guard
// against a pathological input; it is the correct resolution of one the engine produces itself.
[[nodiscard]] auto apply_rec(const Substitution& s, const Term& t, std::uint64_t depth) -> Term {
    constexpr std::uint64_t apply_depth_cap = 10'000;
    if (depth > apply_depth_cap) {
        return t;
    }
    if (is_var(t)) {
        Term cur = t;
        std::size_t guard = 0;
        while (is_var(cur)) {
            auto bound = find_binding(s, key_of(cur));
            if (!bound) {
                return cur;  // genuinely unbound
            }
            if (is_var(*bound) && key_of(*bound) == key_of(cur)) {
                return cur;  // bound to itself: an answer's way of saying "unbound"
            }
            cur = *bound;
            if (++guard > s.size() + 1) {
                return cur;  // unreachable given occurs-check; a cycle would otherwise hang
            }
        }
        return apply_rec(s, cur, depth + 1);
    }
    if (is_compound(t)) {
        const CompoundNode& c = compound_of(t);
        std::vector<Term> args;
        args.reserve(c.args.size());
        for (const Term& arg : c.args) {
            args.push_back(apply_rec(s, arg, depth + 1));
        }
        return make_compound(c.functor, std::move(args));
    }
    return t;  // atom or integer
}

// Stamps generation `gen` onto every variable in `t` (standardise-apart). Clause variables are
// scoped by NAME, so overwriting the generation with a fresh value per activation makes a
// clause's variables distinct from every other activation and from the query.
[[nodiscard]] auto rename_term(const Term& t, std::uint64_t gen) -> Term {
    if (is_var(t)) {
        return make_var(var_of(t).name, gen);
    }
    if (is_compound(t)) {
        const CompoundNode& c = compound_of(t);
        std::vector<Term> args;
        args.reserve(c.args.size());
        for (const Term& arg : c.args) {
            args.push_back(rename_term(arg, gen));
        }
        return make_compound(c.functor, std::move(args));
    }
    return t;
}

[[nodiscard]] auto rename_clause(const Clause& c, std::uint64_t gen) -> Clause {
    Clause out{.head = rename_term(c.head, gen), .body = {}};
    out.body.reserve(c.body.size());
    for (const Term& g : c.body) {
        out.body.push_back(rename_term(g, gen));
    }
    return out;
}

// Collects the variable identities appearing in `t` in first-appearance order (deduplicated).
auto collect_vars(const Term& t, std::vector<VarKey>& out) -> void {
    if (is_var(t)) {
        const VarKey k = key_of(t);
        for (const VarKey& seen : out) {
            if (seen == k) {
                return;
            }
        }
        out.push_back(k);
        return;
    }
    if (is_compound(t)) {
        for (const Term& arg : compound_of(t).args) {
            collect_vars(arg, out);
        }
    }
}

[[nodiscard]] auto collect_query_vars(const std::vector<Term>& goals) -> std::vector<VarKey> {
    std::vector<VarKey> out;
    for (const Term& g : goals) {
        collect_vars(g, out);
    }
    return out;
}

// A goal is well-formed if it is callable, or a well-formed negation. The checks that can be
// made STATICALLY are made here; whether a negated goal is GROUND cannot be, because a variable
// may be bound to a ground callable term by the time the goal is reached — that is a runtime
// condition and is enforced in sld_search.
[[nodiscard]] auto validate_goal(const Term& g) -> bool {
    if (is_compound(g) && compound_of(g).functor == naf_functor) {
        const CompoundNode& c = compound_of(g);
        if (c.args.size() != 1) {
            return false;  // `\+` is strictly arity 1; anything else is not negation
        }
        const Term& inner = c.args.front();
        if (is_var(inner)) {
            return true;  // may be bound to a callable term before it is called
        }
        if (!is_callable(inner) && !is_negation(inner)) {
            return false;  // e.g. `\+ 3` — an integer can never become a goal
        }
        return validate_goal(inner);  // nested negation is fine
    }
    return is_callable(g);
}

[[nodiscard]] auto validate_program(const Program& program) -> bool {
    for (const Clause& c : program) {
        if (!is_callable(c.head)) {
            return false;
        }
        // `\+` is solver-defined; letting a program supply clauses for it would mean two
        // different meanings for the same goal depending on which was tried first. The whole
        // NAME is reserved at every arity, so a near-miss like `\+`/2 cannot be defined either.
        if (is_compound(c.head) && compound_of(c.head).functor == naf_functor) {
            return false;
        }
        if (is_atom(c.head) && atom_of(c.head).name == naf_functor) {
            return false;
        }
        // Every other builtin is reserved on the same reasoning: the engine already gives the
        // goal a meaning, and a clause would give it a second one.
        if (is_builtin(is_atom(c.head) ? atom_of(c.head).name : compound_of(c.head).functor,
                       is_compound(c.head) ? compound_of(c.head).args.size() : 0)) {
            return false;
        }
        for (const Term& g : c.body) {
            if (!validate_goal(g)) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] auto validate_goals(const std::vector<Term>& goals) -> bool {
    for (const Term& g : goals) {
        if (!validate_goal(g)) {
            return false;
        }
    }
    return true;
}

// Deterministically renumbers residual (internal, standardised-apart) variables so answers do
// not leak the internal rename counter — this is what lets solve_or_parallel match solve
// byte-for-byte regardless of how generations were assigned. Query variables are seeded to map
// to themselves, so an unbound query variable keeps its own name and only genuinely internal
// variables become _G0, _G1, ... in encounter order.
struct Canonicaliser {
    std::vector<std::pair<VarKey, Term>> mapping;
    std::uint64_t counter = 0;
};

[[nodiscard]] auto canonicalise(Canonicaliser& c, const Term& t) -> Term {
    if (is_var(t)) {
        const VarKey k = key_of(t);
        for (const auto& [key, val] : c.mapping) {
            if (key == k) {
                return val;
            }
        }
        Term fresh = make_var(std::format("_G{}", c.counter), 0);
        ++c.counter;
        c.mapping.emplace_back(k, fresh);
        return fresh;
    }
    if (is_compound(t)) {
        const CompoundNode& cn = compound_of(t);
        std::vector<Term> args;
        args.reserve(cn.args.size());
        for (const Term& arg : cn.args) {
            args.push_back(canonicalise(c, arg));
        }
        return make_compound(cn.functor, std::move(args));
    }
    return t;
}

// Restricts a raw (internally-bound) substitution to the query's variables and canonicalises
// any residual variables in their resolved values.
[[nodiscard]] auto restrict_answer(const Substitution& raw, const std::vector<VarKey>& qvars)
    -> Substitution {
    Canonicaliser c;
    for (const VarKey& q : qvars) {
        c.mapping.emplace_back(q, make_var(q.name, q.generation));  // query vars keep themselves
    }
    Substitution out;
    out.reserve(qvars.size());
    for (const VarKey& q : qvars) {
        const Term resolved = apply_rec(raw, make_var(q.name, q.generation), 0);
        out.emplace_back(q, canonicalise(c, resolved));
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Standard order of terms, list helpers, and sorting.
// ---------------------------------------------------------------------------

[[nodiscard]] auto ord_kind_rank(const Term& t) -> int {
    if (is_var(t)) {
        return 0;
    }
    if (is_int(t)) {
        return 1;
    }
    if (is_atom(t)) {
        return 2;
    }
    if (is_compound(t)) {
        return 3;
    }
    return -1;
}

[[nodiscard]] auto ord_compare_strings(std::string_view s1, std::string_view s2) -> int {
    // Plain byte order compares characters as unsigned char so non-ASCII bytes
    // are consistently ordered across platforms where char may be signed.
    const std::size_t n1 = s1.size();
    const std::size_t n2 = s2.size();
    const std::size_t min_len = std::min(n1, n2);
    for (std::size_t i = 0; i < min_len; ++i) {
        const auto c1 = static_cast<unsigned char>(s1[i]);
        const auto c2 = static_cast<unsigned char>(s2[i]);
        if (c1 < c2) {
            return -1;
        }
        if (c1 > c2) {
            return 1;
        }
    }
    if (n1 < n2) {
        return -1;
    }
    if (n1 > n2) {
        return 1;
    }
    return 0;
}

[[nodiscard]] auto ord_compare_terms_impl(Term a, Term b, std::size_t depth) -> int {
    while (true) {
        // The depth cap serves purely as a defensive backstop for deeply nested non-tail structures,
        // biasing toward 0 at the bottom to guarantee termination without stack overflow.
        if (depth >= 10000) {
            return 0;
        }

        const int rank_a = ord_kind_rank(a);
        const int rank_b = ord_kind_rank(b);
        if (rank_a != rank_b) {
            return rank_a < rank_b ? -1 : 1;
        }

        switch (rank_a) {
            case 0: {
                const auto& va = var_of(a);
                const auto& vb = var_of(b);
                if (va.generation != vb.generation) {
                    return va.generation < vb.generation ? -1 : 1;
                }
                return ord_compare_strings(va.name, vb.name);
            }
            case 1: {
                const auto val_a = int_of(a).value;
                const auto val_b = int_of(b).value;
                if (val_a != val_b) {
                    return val_a < val_b ? -1 : 1;
                }
                return 0;
            }
            case 2: {
                return ord_compare_strings(atom_of(a).name, atom_of(b).name);
            }
            case 3: {
                const auto& ca = compound_of(a);
                const auto& cb = compound_of(b);
                const auto arity_a = ca.args.size();
                const auto arity_b = cb.args.size();
                if (arity_a != arity_b) {
                    return arity_a < arity_b ? -1 : 1;
                }
                const int name_cmp = ord_compare_strings(ca.functor, cb.functor);
                if (name_cmp != 0) {
                    return name_cmp;
                }
                for (std::size_t i = 0; i + 1 < arity_a; ++i) {
                    const int c = ord_compare_terms_impl(ca.args[i], cb.args[i], depth + 1);
                    if (c != 0) {
                        return c;
                    }
                }
                // Prolog list spines and right-branching trees can grow to millions of cells.
                // Re-binding to the final argument and continuing the loop provides tail-call
                // elimination, guaranteeing O(1) recursion depth along spines without hitting the depth cap.
                Term next_a = ca.args.back();
                Term next_b = cb.args.back();
                a = std::move(next_a);
                b = std::move(next_b);
                break;
            }
            default:
                return 0;
        }
    }
}

[[nodiscard]] auto compare_terms(const Term& a, const Term& b) -> int {
    return ord_compare_terms_impl(a, b, 0);
}

[[nodiscard]] auto list_to_vector(const Term& t) -> std::optional<std::vector<Term>> {
    std::vector<Term> elements;
    Term current = t;
    std::size_t count = 0;
    constexpr std::size_t kMaxCells = 10'000'000;

    while (is_cons(current)) {
        if (++count > kMaxCells) {
            return std::nullopt;
        }
        const auto& comp = compound_of(current);
        elements.push_back(comp.args[0]);
        current = comp.args[1];
    }

    if (is_nil(current)) {
        return elements;
    }
    return std::nullopt;
}

[[nodiscard]] auto is_proper_list(const Term& t) -> bool {
    // Walking the spine directly without allocating vector storage enables fast validation
    // while capping traversal at 10,000,000 cells defensively prevents infinite loops on cyclic data.
    Term current = t;
    std::size_t count = 0;
    constexpr std::size_t kMaxCells = 10'000'000;

    while (is_cons(current)) {
        if (++count > kMaxCells) {
            return false;
        }
        current = compound_of(current).args[1];
    }
    return is_nil(current);
}

[[nodiscard]] auto sort_terms_keep_duplicates(std::vector<Term> xs) -> std::vector<Term> {
    // Stable sort preserves the input relative order of equal elements to avoid gratuitous reordering.
    std::stable_sort(xs.begin(), xs.end(), [](const Term& a, const Term& b) -> bool {
        return compare_terms(a, b) < 0;
    });
    return xs;
}

[[nodiscard]] auto sort_terms_unique(std::vector<Term> xs) -> std::vector<Term> {
    std::stable_sort(xs.begin(), xs.end(), [](const Term& a, const Term& b) -> bool {
        return compare_terms(a, b) < 0;
    });
    const auto it = std::unique(xs.begin(), xs.end(), [](const Term& a, const Term& b) -> bool {
        return compare_terms(a, b) == 0;
    });
    xs.erase(it, xs.end());
    return xs;
}

[[nodiscard]] auto keysort_pairs(std::vector<Term> xs) -> std::optional<std::vector<Term>> {
    for (const auto& elem : xs) {
        if (!is_compound(elem)) {
            return std::nullopt;
        }
        const auto& comp = compound_of(elem);
        if (comp.functor != "-" || comp.args.size() != 2) {
            return std::nullopt;
        }
    }

    // Stable sort is required by keysort semantics so pairs with equal keys retain their input order.
    std::stable_sort(xs.begin(), xs.end(), [](const Term& a, const Term& b) -> bool {
        const auto& key_a = compound_of(a).args[0];
        const auto& key_b = compound_of(b).args[0];
        return compare_terms(key_a, key_b) < 0;
    });

    return xs;
}

[[nodiscard]] auto ord_get_key(const Term& t, std::size_t key) -> const Term& {
    if (key == 0) {
        return t;
    }
    return compound_of(t).args[key - 1];
}

[[nodiscard]] auto sort_terms_general(std::vector<Term> xs, std::size_t key,
                                      std::string_view order) -> std::optional<std::vector<Term>> {
    const bool is_asc_unique = (order == "@<");
    const bool is_asc_keep = (order == "@=<");
    const bool is_desc_unique = (order == "@>");
    const bool is_desc_keep = (order == "@>=");

    if (!is_asc_unique && !is_asc_keep && !is_desc_unique && !is_desc_keep) {
        return std::nullopt;
    }

    if (key > 0) {
        for (const auto& elem : xs) {
            if (!is_compound(elem) || compound_of(elem).args.size() < key) {
                return std::nullopt;
            }
        }
    }

    if (is_asc_unique) {
        // Stable sort places equivalent keys consecutively in original encounter order,
        // allowing std::unique to retain the first occurrence as guaranteed by SWI-Prolog sort/4.
        std::stable_sort(xs.begin(), xs.end(), [key](const Term& a, const Term& b) -> bool {
            return compare_terms(ord_get_key(a, key), ord_get_key(b, key)) < 0;
        });
        const auto it = std::unique(xs.begin(), xs.end(), [key](const Term& a, const Term& b) -> bool {
            return compare_terms(ord_get_key(a, key), ord_get_key(b, key)) == 0;
        });
        xs.erase(it, xs.end());
        return xs;
    }

    if (is_asc_keep) {
        std::stable_sort(xs.begin(), xs.end(), [key](const Term& a, const Term& b) -> bool {
            return compare_terms(ord_get_key(a, key), ord_get_key(b, key)) < 0;
        });
        return xs;
    }

    if (is_desc_unique) {
        std::stable_sort(xs.begin(), xs.end(), [key](const Term& a, const Term& b) -> bool {
            return compare_terms(ord_get_key(a, key), ord_get_key(b, key)) > 0;
        });
        const auto it = std::unique(xs.begin(), xs.end(), [key](const Term& a, const Term& b) -> bool {
            return compare_terms(ord_get_key(a, key), ord_get_key(b, key)) == 0;
        });
        xs.erase(it, xs.end());
        return xs;
    }

    // Descending order keeping duplicates must be stable according to the specification.
    std::stable_sort(xs.begin(), xs.end(), [key](const Term& a, const Term& b) -> bool {
        return compare_terms(ord_get_key(a, key), ord_get_key(b, key)) > 0;
    });
    return xs;
}

// ---------------------------------------------------------------------------
// Arithmetic evaluation (integers only).
// ---------------------------------------------------------------------------

[[nodiscard]] auto arith_abs_u64(std::int64_t v) -> std::uint64_t {
    if (v >= 0) {
        return static_cast<std::uint64_t>(v);
    }
    // Negating via unsigned arithmetic avoids signed integer overflow UB on INT64_MIN.
    return 0ULL - static_cast<std::uint64_t>(v);
}

[[nodiscard]] auto arith_msb(std::int64_t x) -> Result<std::int64_t> {
    if (x <= 0) {
        return make_error<std::int64_t>(MathError::domain_error);
    }
    // Software binary search determines the index of the highest set bit without external headers.
    auto u = static_cast<std::uint64_t>(x);
    std::int64_t bit = 0;
    if (u >= (1ULL << 32)) { u >>= 32; bit += 32; }
    if (u >= (1ULL << 16)) { u >>= 16; bit += 16; }
    if (u >= (1ULL << 8))  { u >>= 8;  bit += 8; }
    if (u >= (1ULL << 4))  { u >>= 4;  bit += 4; }
    if (u >= (1ULL << 2))  { u >>= 2;  bit += 2; }
    if (u >= (1ULL << 1))  { u >>= 1;  bit += 1; }
    return bit;
}

[[nodiscard]] auto arith_add(std::int64_t a, std::int64_t b) -> Result<std::int64_t> {
    constexpr auto max_val = std::numeric_limits<std::int64_t>::max();
    constexpr auto min_val = std::numeric_limits<std::int64_t>::min();
    // Compare bounds before addition to prevent invoking undefined behavior under UBSan.
    if (b > 0 && a > max_val - b) {
        return make_error<std::int64_t>(MathError::overflow);
    }
    if (b < 0 && a < min_val - b) {
        return make_error<std::int64_t>(MathError::overflow);
    }
    return a + b;
}

[[nodiscard]] auto arith_sub(std::int64_t a, std::int64_t b) -> Result<std::int64_t> {
    constexpr auto max_val = std::numeric_limits<std::int64_t>::max();
    constexpr auto min_val = std::numeric_limits<std::int64_t>::min();
    // Compare bounds before subtraction to prevent invoking undefined behavior under UBSan.
    if (b < 0 && a > max_val + b) {
        return make_error<std::int64_t>(MathError::overflow);
    }
    if (b > 0 && a < min_val + b) {
        return make_error<std::int64_t>(MathError::overflow);
    }
    return a - b;
}

[[nodiscard]] auto arith_mul(std::int64_t a, std::int64_t b) -> Result<std::int64_t> {
    constexpr auto max_val = std::numeric_limits<std::int64_t>::max();
    constexpr auto min_val = std::numeric_limits<std::int64_t>::min();
    if (a == 0 || b == 0) {
        return std::int64_t{0};
    }
    if (a == 1) return b;
    if (b == 1) return a;
    if (a == -1) {
        if (b == min_val) return make_error<std::int64_t>(MathError::overflow);
        return -b;
    }
    if (b == -1) {
        if (a == min_val) return make_error<std::int64_t>(MathError::overflow);
        return -a;
    }
    if (a == min_val || b == min_val) {
        // Multiplying INT64_MIN by any factor other than 0, 1, or -1 inevitably overflows.
        return make_error<std::int64_t>(MathError::overflow);
    }
    if (a > 0) {
        if (b > 0) {
            if (a > max_val / b) return make_error<std::int64_t>(MathError::overflow);
        } else {
            if (b < min_val / a) return make_error<std::int64_t>(MathError::overflow);
        }
    } else {
        if (b > 0) {
            if (a < min_val / b) return make_error<std::int64_t>(MathError::overflow);
        } else {
            if (a < max_val / b) return make_error<std::int64_t>(MathError::overflow);
        }
    }
    return a * b;
}

[[nodiscard]] auto arith_div(std::int64_t x, std::int64_t y) -> Result<std::int64_t> {
    if (y == 0) {
        return make_error<std::int64_t>(MathError::division_by_zero);
    }
    if (x == std::numeric_limits<std::int64_t>::min() && y == -1) {
        return make_error<std::int64_t>(MathError::overflow);
    }
    auto q = x / y;
    auto r = x % y;
    // Floor division must decrement quotient toward negative infinity when signs differ and remainder is non-zero.
    if (r != 0 && ((x < 0) ^ (y < 0))) {
        --q;
    }
    return q;
}

[[nodiscard]] auto arith_mod(std::int64_t x, std::int64_t y) -> Result<std::int64_t> {
    if (y == 0) {
        return make_error<std::int64_t>(MathError::division_by_zero);
    }
    if (y == 1 || y == -1) {
        return std::int64_t{0};
    }
    auto r = x % y;
    // The modulo paired with floor division takes the sign of the divisor.
    if (r != 0 && ((x < 0) ^ (y < 0))) {
        r += y;
    }
    return r;
}

[[nodiscard]] auto arith_rem(std::int64_t x, std::int64_t y) -> Result<std::int64_t> {
    if (y == 0) {
        return make_error<std::int64_t>(MathError::division_by_zero);
    }
    if (y == 1 || y == -1) {
        // Explicitly intercepting +/-1 avoids hardware overflow on INT64_MIN % -1 on x86 architectures.
        return std::int64_t{0};
    }
    return x % y;
}

[[nodiscard]] auto arith_min(std::int64_t x, std::int64_t y) -> std::int64_t {
    return x < y ? x : y;
}

[[nodiscard]] auto arith_max(std::int64_t x, std::int64_t y) -> std::int64_t {
    return x > y ? x : y;
}

[[nodiscard]] auto arith_gcd(std::int64_t x, std::int64_t y) -> Result<std::int64_t> {
    auto a = arith_abs_u64(x);
    auto b = arith_abs_u64(y);
    while (b != 0) {
        auto t = b;
        b = a % b;
        a = t;
    }
    constexpr auto max_signed = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    if (a > max_signed) {
        // Producing 2^63 (for example gcd(INT64_MIN, 0)) cannot be represented as positive int64.
        return make_error<std::int64_t>(MathError::overflow);
    }
    return static_cast<std::int64_t>(a);
}

[[nodiscard]] auto arith_pow(std::int64_t x, std::int64_t y) -> Result<std::int64_t> {
    if (x == 1) {
        return std::int64_t{1};
    }
    if (x == -1) {
        // Alternates between 1 and -1 based on parity for both positive and negative exponents.
        return (y % 2 == 0) ? std::int64_t{1} : std::int64_t{-1};
    }
    if (y < 0) {
        return make_error<std::int64_t>(MathError::domain_error);
    }
    if (y == 0) {
        return std::int64_t{1};
    }
    if (x == 0) {
        return std::int64_t{0};
    }

    std::int64_t base = x;
    std::int64_t exp = y;
    std::int64_t result = 1;

    while (exp > 0) {
        if (exp % 2 != 0) {
            auto m = arith_mul(result, base);
            if (!m) {
                return m;
            }
            result = *m;
        }
        exp /= 2;
        if (exp > 0) {
            auto m = arith_mul(base, base);
            if (!m) {
                return m;
            }
            base = *m;
        }
    }
    return result;
}

[[nodiscard]] auto arith_shr(std::int64_t x, std::int64_t y) -> Result<std::int64_t> {
    if (y < 0 || y >= 64) {
        return make_error<std::int64_t>(MathError::domain_error);
    }
    if (y == 0) {
        return x;
    }
    return x >> y;
}

[[nodiscard]] auto arith_shl(std::int64_t x, std::int64_t y) -> Result<std::int64_t> {
    if (y < 0 || y >= 64) {
        return make_error<std::int64_t>(MathError::domain_error);
    }
    if (x == 0 || y == 0) {
        return x;
    }
    constexpr auto max_val = std::numeric_limits<std::int64_t>::max();
    constexpr auto min_val = std::numeric_limits<std::int64_t>::min();
    if (x > 0) {
        if (x > (max_val >> y)) {
            return make_error<std::int64_t>(MathError::overflow);
        }
    } else {
        if (x < (min_val >> y)) {
            return make_error<std::int64_t>(MathError::overflow);
        }
    }
    auto u = static_cast<std::uint64_t>(x) << static_cast<std::uint64_t>(y);
    return static_cast<std::int64_t>(u);
}

[[nodiscard]] auto arith_is_unary_float(std::string_view name) -> bool {
    return name == "sqrt" || name == "sin" || name == "cos" || name == "tan" ||
           name == "asin" || name == "acos" || name == "atan" || name == "exp" ||
           name == "log" || name == "float" || name == "integer" ||
           name == "float_integer_part" || name == "float_fractional_part" ||
           name == "round" || name == "truncate" || name == "ceiling" ||
           name == "floor";
}

[[nodiscard]] auto arith_is_binary_float(std::string_view name) -> bool {
    return name == "**" || name == "atan2";
}

[[nodiscard]] auto is_arith_functor(std::string_view name, std::size_t arity) -> bool {
    if (arity == 0) {
        return name == "max_integer" ||
               name == "min_integer" ||
               name == "pi" ||
               name == "e" ||
               name == "inf" ||
               name == "nan" ||
               name == "epsilon" ||
               name == "max_tagged_integer" ||
               name == "random";
    }
    if (arity == 1) {
        return name == "+" ||
               name == "-" ||
               name == "abs" ||
               name == "sign" ||
               name == "msb" ||
               name == "\\" ||
               arith_is_unary_float(name);
    }
    if (arity == 2) {
        return name == "+" ||
               name == "-" ||
               name == "*" ||
               name == "/" ||
               name == "//" ||
               name == "div" ||
               name == "mod" ||
               name == "rem" ||
               name == "min" ||
               name == "max" ||
               name == "gcd" ||
               name == "^" ||
               name == ">>" ||
               name == "<<" ||
               name == "/\\" ||
               name == "\\/" ||
               name == "xor" ||
               arith_is_binary_float(name);
    }
    return false;
}

[[nodiscard]] auto arith_eval_impl(const Term& t, const Substitution& s, std::size_t depth)
    -> Result<std::int64_t> {
    constexpr std::size_t arith_max_depth = 10000;
    if (depth >= arith_max_depth) {
        return make_error<std::int64_t>(MathError::domain_error);
    }

    Term cur = is_var(t) ? apply_substitution(s, t) : t;

    if (is_int(cur)) {
        return int_of(cur).value;
    }

    if (is_var(cur)) {
        return make_error<std::int64_t>(MathError::domain_error);
    }

    if (is_atom(cur)) {
        std::string_view name = atom_of(cur).name;
        if (name == "max_integer") {
            return std::numeric_limits<std::int64_t>::max();
        }
        if (name == "min_integer") {
            return std::numeric_limits<std::int64_t>::min();
        }
        if (name == "pi" || name == "e" || name == "inf" || name == "nan" ||
            name == "epsilon" || name == "max_tagged_integer" || name == "random") {
            return make_error<std::int64_t>(MathError::not_implemented);
        }
        return make_error<std::int64_t>(MathError::domain_error);
    }

    if (is_compound(cur)) {
        const auto& comp = compound_of(cur);
        std::string_view name = comp.functor;
        const auto& args = comp.args;
        const auto arity = args.size();

        if (!is_arith_functor(name, arity)) {
            return make_error<std::int64_t>(MathError::domain_error);
        }

        if (arity == 1) {
            auto arg_res = arith_eval_impl(args[0], s, depth + 1);
            if (!arg_res) {
                return arg_res;
            }
            if (arith_is_unary_float(name)) {
                return make_error<std::int64_t>(MathError::not_implemented);
            }
            auto val = *arg_res;
            if (name == "+") {
                return val;
            }
            if (name == "-") {
                if (val == std::numeric_limits<std::int64_t>::min()) {
                    return make_error<std::int64_t>(MathError::overflow);
                }
                return -val;
            }
            if (name == "abs") {
                if (val == std::numeric_limits<std::int64_t>::min()) {
                    return make_error<std::int64_t>(MathError::overflow);
                }
                return val < 0 ? -val : val;
            }
            if (name == "sign") {
                return val > 0 ? std::int64_t{1} : (val < 0 ? std::int64_t{-1} : std::int64_t{0});
            }
            if (name == "msb") {
                return arith_msb(val);
            }
            if (name == "\\") {
                return ~val;
            }
            return make_error<std::int64_t>(MathError::domain_error);
        }

        if (arity == 2) {
            auto lhs_res = arith_eval_impl(args[0], s, depth + 1);
            if (!lhs_res) {
                return lhs_res;
            }
            auto rhs_res = arith_eval_impl(args[1], s, depth + 1);
            if (!rhs_res) {
                return rhs_res;
            }
            if (arith_is_binary_float(name)) {
                return make_error<std::int64_t>(MathError::not_implemented);
            }
            auto a = *lhs_res;
            auto b = *rhs_res;
            if (name == "+") {
                return arith_add(a, b);
            }
            if (name == "-") {
                return arith_sub(a, b);
            }
            if (name == "*") {
                return arith_mul(a, b);
            }
            if (name == "/") {
                if (b == 0) {
                    return make_error<std::int64_t>(MathError::division_by_zero);
                }
                if (a == std::numeric_limits<std::int64_t>::min() && b == -1) {
                    return make_error<std::int64_t>(MathError::overflow);
                }
                if (a % b != 0) {
                    return make_error<std::int64_t>(MathError::inexact);
                }
                return a / b;
            }
            if (name == "//") {
                if (b == 0) {
                    return make_error<std::int64_t>(MathError::division_by_zero);
                }
                if (a == std::numeric_limits<std::int64_t>::min() && b == -1) {
                    return make_error<std::int64_t>(MathError::overflow);
                }
                return a / b;
            }
            if (name == "div") {
                return arith_div(a, b);
            }
            if (name == "mod") {
                return arith_mod(a, b);
            }
            if (name == "rem") {
                return arith_rem(a, b);
            }
            if (name == "min") {
                return arith_min(a, b);
            }
            if (name == "max") {
                return arith_max(a, b);
            }
            if (name == "gcd") {
                return arith_gcd(a, b);
            }
            if (name == "^") {
                return arith_pow(a, b);
            }
            if (name == ">>") {
                return arith_shr(a, b);
            }
            if (name == "<<") {
                return arith_shl(a, b);
            }
            if (name == "/\\") {
                return a & b;
            }
            if (name == "\\/") {
                return a | b;
            }
            if (name == "xor") {
                return a ^ b;
            }
            return make_error<std::int64_t>(MathError::domain_error);
        }

        return make_error<std::int64_t>(MathError::domain_error);
    }

    return make_error<std::int64_t>(MathError::domain_error);
}

[[nodiscard]] auto eval_arith(const Term& t, const Substitution& s) -> Result<std::int64_t> {
    Term resolved = apply_substitution(s, t);
    return arith_eval_impl(resolved, s, 0);
}

[[nodiscard]] auto eval_arith_compare(std::string_view op, const Term& lhs, const Term& rhs,
                                      const Substitution& s) -> Result<bool> {
    if (op != "=:=" && op != "=\\=" && op != "=\\\\=" &&
        op != "<" && op != ">" && op != "=<" && op != ">=") {
        return make_error<bool>(MathError::domain_error);
    }
    auto lhs_res = eval_arith(lhs, s);
    if (!lhs_res) {
        return make_error<bool>(lhs_res.error());
    }
    auto rhs_res = eval_arith(rhs, s);
    if (!rhs_res) {
        return make_error<bool>(rhs_res.error());
    }
    auto l = *lhs_res;
    auto r = *rhs_res;
    if (op == "=:=") {
        return l == r;
    }
    if (op == "=\\=" || op == "=\\\\=") {
        return l != r;
    }
    if (op == "<") {
        return l < r;
    }
    if (op == ">") {
        return l > r;
    }
    if (op == "=<") {
        return l <= r;
    }
    if (op == ">=") {
        return l >= r;
    }
    return make_error<bool>(MathError::domain_error);
}

// ---------------------------------------------------------------------------
// Predicate indicators and the reserved builtin set.
// ---------------------------------------------------------------------------

auto indicator(std::string_view name, std::size_t arity) -> std::string {
    return std::format("{}/{}", name, arity);
}

auto indicator_of(const Term& t) -> std::optional<std::string> {
    if (is_atom(t)) {
        return indicator(atom_of(t).name, 0);
    }
    if (is_compound(t)) {
        const CompoundNode& c = compound_of(t);
        return indicator(c.functor, c.args.size());
    }
    return std::nullopt;
}

namespace {

// The goal's argument list. A 0-arity goal is an atom and has none, which is why this returns
// a fresh empty vector rather than a reference into the term.
[[nodiscard]] auto args_of(const Term& t) -> std::vector<Term> {
    if (is_compound(t)) {
        return compound_of(t).args;
    }
    return {};
}

// The name part of a callable term.
[[nodiscard]] auto name_of(const Term& t) -> std::string {
    if (is_atom(t)) {
        return atom_of(t).name;
    }
    return compound_of(t).functor;
}

// Builtins whose control flow is bespoke: they are woven into the search loop itself because
// they choose their own continuation, can succeed more than once, or must be opaque to cut.
[[nodiscard]] auto control_builtins() -> const std::unordered_set<std::string>& {
    static const std::unordered_set<std::string> table = {
        "true/0",    "fail/0",       "false/0",   "!/0",         ",/2",
        ";/2",       "->/2",         "*->/2",     "\\+/1",       "call/1",
        "call/2",    "call/3",       "call/4",    "call/5",      "call/6",
        "call/7",    "call/8",       "findall/3", "findall/4",   "bagof/3",
        "setof/3",   "forall/2",     "between/3", "clause/2",    "retract/1",
        "arg/3",     "length/2",     "nth0/3",    "nth1/3",      "atom_concat/3",
        "sub_atom/5",
    };
    return table;
}

// Builtins that either succeed once with an extended substitution or fail: a single
// deterministic step the search loop can treat uniformly.
[[nodiscard]] auto det_builtins() -> const std::unordered_set<std::string>& {
    static const std::unordered_set<std::string> table = {
        "=/2",           "\\=/2",         "==/2",          "\\==/2",
        "@</2",          "@>/2",          "@=</2",         "@>=/2",
        "compare/3",     "var/1",         "nonvar/1",      "atom/1",
        "number/1",      "integer/1",     "float/1",       "atomic/1",
        "compound/1",    "callable/1",    "is_list/1",     "ground/1",
        "functor/3",     "=../2",         "copy_term/2",   "term_variables/2",
        "is/2",          "=:=/2",         "=\\=/2",        "</2",
        ">/2",           "=</2",          ">=/2",          "succ/2",
        "plus/3",        "msort/2",       "sort/2",        "sort/4",
        "keysort/2",     "numlist/3",     "atom_length/2", "atom_codes/2",
        "atom_chars/2",  "char_code/2",   "number_codes/2", "number_chars/2",
        "atom_number/2", "upcase_atom/2", "downcase_atom/2",
        "assert/1",      "asserta/1",     "assertz/1",     "retractall/1",
        "abolish/1",
    };
    return table;
}

}  // namespace

auto is_builtin(std::string_view name, std::size_t arity) -> bool {
    const std::string ind = indicator(name, arity);
    return control_builtins().contains(ind) || det_builtins().contains(ind);
}

auto builtin_indicators() -> std::vector<std::string> {
    std::vector<std::string> out;
    out.reserve(control_builtins().size() + det_builtins().size());
    for (const std::string& s : control_builtins()) {
        out.push_back(s);
    }
    for (const std::string& s : det_builtins()) {
        out.push_back(s);
    }
    std::ranges::sort(out);
    return out;
}

// ---------------------------------------------------------------------------
// The clause database.
// ---------------------------------------------------------------------------

// Grants the search loop access to Database's storage. The index is an implementation detail
// of the module, not of the class, so the accessors live here rather than widening the public
// surface with methods only the solver would ever call.
struct DatabaseAccess {
    // The entry type is private to Database; re-exporting it here is what lets the search
    // loop name it without widening the class's public surface.
    using Entry = Database::Entry;
    [[nodiscard]] static auto entries(const Database& db) -> const std::vector<DatabaseAccess::Entry>& {
        return db.entries_;
    }
    [[nodiscard]] static auto entries_mut(Database& db) -> std::vector<DatabaseAccess::Entry>& {
        return db.entries_;
    }
};

namespace {

// The index key for a first argument: enough to prove two terms CANNOT unify, never enough to
// claim they can. A variable has no key — it unifies with everything, so clauses with a
// variable first argument are candidates for every call.
[[nodiscard]] auto first_arg_key(const Term& t) -> std::optional<std::string> {
    if (is_var(t)) {
        return std::nullopt;
    }
    if (is_atom(t)) {
        return std::format("a{}", atom_of(t).name);
    }
    if (is_int(t)) {
        return std::format("i{}", int_of(t).value);
    }
    const CompoundNode& c = compound_of(t);
    return std::format("c{}/{}", c.functor, c.args.size());
}

// The clauses that could match `goal`, in clause order.
//
// This is the first-argument index at work. When the call's first argument is bound, a clause
// whose own first argument is a distinct constant or a different functor cannot unify, and is
// skipped without paying for the clause rename and the unification attempt. Skipping only
// provably-impossible clauses is what keeps the answers and their ORDER identical to a linear
// scan — the index is a performance property, never a semantic one.
[[nodiscard]] auto candidate_clauses(const Database& db, const Term& goal, const Substitution& sub)
    -> std::vector<std::size_t> {
    const auto ind = indicator_of(goal);
    if (!ind) {
        return {};
    }
    const std::vector<DatabaseAccess::Entry>& entries = DatabaseAccess::entries(db);

    std::optional<std::string> want;
    if (is_compound(goal)) {
        want = first_arg_key(walk(compound_of(goal).args.front(), sub));
    }

    std::vector<std::size_t> picked;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const DatabaseAccess::Entry& e = entries[i];
        if (!e.live) {
            continue;
        }
        const auto head_ind = indicator_of(e.clause.head);
        if (!head_ind || *head_ind != *ind) {
            continue;
        }
        if (want && is_compound(e.clause.head)) {
            const auto have = first_arg_key(compound_of(e.clause.head).args.front());
            if (have && *have != *want) {
                continue;  // two different constants or functors can never unify
            }
        }
        picked.push_back(i);
    }
    // `asserta` gives a clause a lower order than every existing one without moving anybody,
    // so physical position is not clause position and the candidates must be ordered here.
    std::ranges::stable_sort(picked, [&entries](std::size_t a, std::size_t b) -> bool {
        return entries[a].order < entries[b].order;
    });
    return picked;
}

// Whether a clause may be added for this head. A builtin's meaning is fixed by the engine;
// letting a program supply clauses for one would give the same goal two meanings, resolved by
// whichever happened to be tried first.
[[nodiscard]] auto clause_head_is_permitted(const Term& head) -> bool {
    if (!is_callable(head)) {
        return false;
    }
    // The whole `\+` NAME is reserved at every arity, not just the `\+`/1 the engine defines,
    // so a program cannot introduce a near-miss that reads like negation but is not.
    if (is_compound(head) && compound_of(head).functor == naf_functor) {
        return false;
    }
    if (is_atom(head) && atom_of(head).name == naf_functor) {
        return false;
    }
    return !is_builtin(name_of(head), args_of(head).size());
}

}  // namespace

auto Database::from(Program p) -> Database {
    Database db;
    db.entries_.reserve(p.size());
    for (Clause& c : p) {
        db.entries_.push_back(Entry{.clause = std::move(c), .order = db.next_high_, .live = true});
        ++db.next_high_;
    }
    return db;
}

auto Database::with_clause(Clause c) const -> Database {
    Database copy = *this;
    // The fluent builder is for programs under construction, where refusing a clause would be
    // reported far from where it was written; validation happens once, in solve().
    copy.entries_.push_back(
        Entry{.clause = std::move(c), .order = copy.next_high_, .live = true});
    ++copy.next_high_;
    return copy;
}

auto Database::assertz(Clause c) -> Result<void> {
    if (!clause_head_is_permitted(c.head)) {
        return make_error<void>(MathError::domain_error);
    }
    entries_.push_back(Entry{.clause = std::move(c), .order = next_high_, .live = true});
    ++next_high_;
    return {};
}

auto Database::asserta(Clause c) -> Result<void> {
    if (!clause_head_is_permitted(c.head)) {
        return make_error<void>(MathError::domain_error);
    }
    // Ordering ahead of every existing clause rather than shifting them is what keeps a
    // running search's clause indices valid across an assert.
    --next_low_;
    entries_.push_back(Entry{.clause = std::move(c), .order = next_low_, .live = true});
    return {};
}

auto Database::clauses() const -> Program {
    std::vector<std::size_t> live;
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        if (entries_[i].live) {
            live.push_back(i);
        }
    }
    std::ranges::stable_sort(live, [this](std::size_t a, std::size_t b) -> bool {
        return entries_[a].order < entries_[b].order;
    });
    Program out;
    out.reserve(live.size());
    for (std::size_t i : live) {
        out.push_back(entries_[i].clause);
    }
    return out;
}

auto Database::size() const -> std::size_t {
    return static_cast<std::size_t>(std::ranges::count_if(
        entries_, [](const Entry& e) -> bool { return e.live; }));
}

auto Database::is_defined(std::string_view ind) const -> bool {
    return std::ranges::any_of(entries_, [ind](const Entry& e) -> bool {
        if (!e.live) {
            return false;
        }
        const auto head_ind = indicator_of(e.clause.head);
        return head_ind && *head_ind == ind;
    });
}

namespace {

// ---------------------------------------------------------------------------
// Search state.
// ---------------------------------------------------------------------------

// A goal together with the CUT BARRIER it was introduced under: the identity of the predicate
// call whose remaining clauses a `!` in this position must discard. Body goals of a clause
// activated by call C carry barrier C; the control constructs that are transparent to cut
// (`,`, `;`, `->`'s then-branch) pass the barrier through unchanged, while the opaque ones
// (`\+`, `call/N`, `findall` and friends) allocate a fresh one so a cut inside them stays
// inside them.
struct GoalNode;

// The continuation — every goal still to be proved — as an IMMUTABLE SHARED LIST.
//
// The obvious representation is a vector, and it is the wrong one. Resolution replaces the
// first goal with a clause body, so a vector has to be copied at every step: O(n) work per
// step, and, far worse, the copy lives in the native stack frame. At a derivation depth of
// 1000 that alone overflows a 1 MB stack, which would quietly falsify the promise that the
// depth budget keeps the native recursion bounded. A persistent cons list is pushed in O(1)
// and its tail is SHARED with every alternative that still needs it, so a frame carries one
// pointer and backtracking costs nothing.
using GoalList = std::shared_ptr<const GoalNode>;

struct GoalNode {
    Term goal;
    std::uint64_t barrier;  // the cut barrier this goal was introduced under
    GoalList next;
};

[[nodiscard]] auto cons_goal(Term goal, std::uint64_t barrier, GoalList next) -> GoalList {
    return std::make_shared<const GoalNode>(
        GoalNode{.goal = std::move(goal), .barrier = barrier, .next = std::move(next)});
}

// Mutable state threaded through one search.
//
// `truncated` exists for negation and for the all-solutions builtins: a positive search that
// runs out of budget merely under-reports answers, but a NEGATIVE or an ALL-SOLUTIONS one that
// runs out would report a conclusion that is a fact about the budget rather than about the
// program.
//
// `cut_signal` carries a cut in flight. A `!` sets it to its own barrier and returns; every
// frame between there and that barrier stops offering alternatives, and the clause loop that
// owns the barrier consumes it. Because it unwinds by RETURNING rather than by throwing, the
// continuation to the right of the `!` has already been explored by the time the signal is
// raised — which is precisely the semantics of cut: it prunes backtracking, not forward
// execution.
struct SearchCtx {
    std::uint64_t gen = 0;    // standardise-apart rename counter
    std::uint64_t steps = default_step_budget;
    std::uint64_t calls = 0;  // cut-barrier counter; every predicate call takes the next id
    bool truncated = false;
    std::optional<MathError> error;
    std::optional<std::uint64_t> cut_signal;
};

// A fresh variable that cannot collide with a program or query variable, because it takes a
// generation from the same counter that standardises clauses apart.
[[nodiscard]] auto fresh_var(SearchCtx& ctx) -> Term {
    return make_var("_F", ++ctx.gen);
}

// Copies `t`, replacing each DISTINCT variable with a fresh one.
//
// This is not `rename_term`: that one keys off the variable NAME, which is right for a source
// clause (where distinct variables have distinct names) and wrong for a runtime term, where
// `X_3` and `X_7` are different variables sharing a name. Keying off the whole VarKey is what
// makes `copy_term/2`, `findall/3` and `assert/1` copy rather than merge.
[[nodiscard]] auto copy_rec(const Term& t, std::vector<std::pair<VarKey, Term>>& map,
                            std::uint64_t gen) -> Term {
    if (is_var(t)) {
        const VarKey k = key_of(t);
        for (const auto& [key, val] : map) {
            if (key == k) {
                return val;
            }
        }
        Term fresh = make_var(std::format("_C{}", map.size()), gen);
        map.emplace_back(k, fresh);
        return fresh;
    }
    if (is_compound(t)) {
        const CompoundNode& c = compound_of(t);
        std::vector<Term> args;
        args.reserve(c.args.size());
        for (const Term& a : c.args) {
            args.push_back(copy_rec(a, map, gen));
        }
        return make_compound(c.functor, std::move(args));
    }
    return t;
}

[[nodiscard]] auto copy_term_fresh(const Term& t, SearchCtx& ctx) -> Term {
    std::vector<std::pair<VarKey, Term>> map;
    return copy_rec(t, map, ++ctx.gen);
}

// The variables of `t` in first-appearance order, as terms.
[[nodiscard]] auto term_variables_of(const Term& t) -> std::vector<Term> {
    std::vector<VarKey> keys;
    collect_vars(t, keys);
    std::vector<Term> out;
    out.reserve(keys.size());
    for (const VarKey& k : keys) {
        out.push_back(make_var(k.name, k.generation));
    }
    return out;
}

// Splits a clause body on `,/2`. Only the TOP level is split: `(a ; b)` and `(a -> b)` are
// single goals with their own control semantics and must not be flattened into a conjunction.
auto flatten_conjunction(const Term& t, std::vector<Term>& out) -> void {
    if (is_compound(t)) {
        const CompoundNode& c = compound_of(t);
        if (c.functor == "," && c.args.size() == 2) {
            flatten_conjunction(c.args[0], out);
            flatten_conjunction(c.args[1], out);
            return;
        }
    }
    out.push_back(t);
}

// Reads a term as a clause: `Head :- Body` becomes head plus the flattened body, and anything
// else is a fact. `true` as a body is an empty body, which is the same clause.
[[nodiscard]] auto term_to_clause(const Term& t) -> std::optional<Clause> {
    Term head = t;
    std::vector<Term> body;
    if (is_compound(t)) {
        const CompoundNode& c = compound_of(t);
        if (c.functor == ":-" && c.args.size() == 2) {
            head = c.args[0];
            flatten_conjunction(c.args[1], body);
        }
    }
    if (!is_callable(head)) {
        return std::nullopt;
    }
    std::erase_if(body, [](const Term& g) -> bool {
        return is_atom(g) && atom_of(g).name == "true";
    });
    for (const Term& g : body) {
        if (!is_callable(g)) {
            return std::nullopt;
        }
    }
    return Clause{.head = std::move(head), .body = std::move(body)};
}

// ---------------------------------------------------------------------------
// Atom and character-code conversions.
// ---------------------------------------------------------------------------

// The text of an atomic term: an atom's name or an integer's decimal form. A variable or a
// compound has no text, which the callers report as a domain error.
[[nodiscard]] auto atomic_text(const Term& t) -> std::optional<std::string> {
    if (is_atom(t)) {
        return atom_of(t).name;
    }
    if (is_int(t)) {
        return std::format("{}", int_of(t).value);
    }
    return std::nullopt;
}

// `text` as a list of character CODES (integers) or of one-character atoms (chars). Bytes are
// taken one at a time: this engine has no character-encoding model, so a multi-byte UTF-8
// sequence becomes several codes. Saying that plainly is better than pretending otherwise.
[[nodiscard]] auto text_to_code_list(std::string_view text) -> Term {
    std::vector<Term> cs;
    cs.reserve(text.size());
    for (const char ch : text) {
        cs.push_back(make_int(static_cast<std::int64_t>(static_cast<unsigned char>(ch))));
    }
    return make_list(std::move(cs));
}

[[nodiscard]] auto text_to_char_list(std::string_view text) -> Term {
    std::vector<Term> cs;
    cs.reserve(text.size());
    for (const char ch : text) {
        cs.push_back(make_atom(std::string(1, ch)));
    }
    return make_list(std::move(cs));
}

// The inverse: a proper list of codes or of one-character atoms back to text. Anything else
// (an unbound tail, a code outside a byte, a multi-character atom) is nullopt.
[[nodiscard]] auto code_list_to_text(const Term& t) -> std::optional<std::string> {
    const auto items = list_to_vector(t);
    if (!items) {
        return std::nullopt;
    }
    std::string out;
    out.reserve(items->size());
    for (const Term& e : *items) {
        if (is_int(e)) {
            const std::int64_t v = int_of(e).value;
            if (v < 0 || v > 255) {
                return std::nullopt;  // no encoding model, so a non-byte code has no text
            }
            out.push_back(static_cast<char>(static_cast<unsigned char>(v)));
            continue;
        }
        if (is_atom(e) && atom_of(e).name.size() == 1) {
            out.push_back(atom_of(e).name.front());
            continue;
        }
        return std::nullopt;
    }
    return out;
}

// Parses `text` as a Prolog integer, including a leading sign. Returns nullopt when the text
// is not exactly an integer — `atom_number/2` FAILS there rather than erroring, which is the
// ISO behaviour, so the distinction is the caller's to make.
[[nodiscard]] auto text_to_integer(std::string_view text) -> std::optional<std::int64_t> {
    std::string_view s = text;
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) {
        s.remove_prefix(1);
    }
    bool negative = false;
    if (!s.empty() && (s.front() == '+' || s.front() == '-')) {
        negative = s.front() == '-';
        s.remove_prefix(1);
    }
    if (s.empty() || !std::ranges::all_of(s, [](char c) -> bool {
            return c >= '0' && c <= '9';
        })) {
        return std::nullopt;
    }
    // from_chars cannot represent the most negative value as a positive magnitude, so the sign
    // is parsed together with the digits rather than applied afterwards.
    std::string signed_text;
    signed_text.reserve(s.size() + 1);
    if (negative) {
        signed_text.push_back('-');
    }
    signed_text.append(s);
    std::int64_t value = 0;
    const char* const begin = signed_text.data();
    const char* const end = begin + signed_text.size();
    const auto res = std::from_chars(begin, end, value);
    if (res.ec != std::errc{} || res.ptr != end) {
        return std::nullopt;
    }
    return value;
}

}  // namespace

namespace {

// Forward declaration: the builtins and the search loop are mutually recursive, because a
// builtin like `findall/3` runs a sub-derivation and the search loop dispatches builtins.
auto sld_search(Database& db, const GoalList& goals, const Substitution& sub,
                SearchCtx& ctx, std::uint64_t max_solutions, std::uint64_t depth,
                std::vector<Substitution>& out) -> void;

// The outcome of running a goal as an isolated sub-derivation.
struct SubResult {
    std::vector<Substitution> answers;
    bool truncated{};
};

// Runs `goal` on its own, under a FRESH cut barrier, collecting up to `want` answers
// (0 = all). This is the shape every cut-opaque construct needs: `\+`, `call/N`, the condition
// of `->`, and the all-solutions builtins.
//
// The truncation FLAG is saved and restored around the sub-search so a sibling derivation's
// truncation is not misread as this one's. The step and depth BUDGETS are deliberately
// inherited: giving a sub-search a fresh allowance is what makes `p :- \+ p.` never terminate,
// since every level would restart the count. The cost is that the conjunction is not
// commutative in its ERROR behaviour, and the direction of that cost is the safe one — an
// inherited budget can turn an answer into an error, never into a wrong answer.
[[nodiscard]] auto run_sub(Database& db, const Term& goal, const Substitution& sub,
                           SearchCtx& ctx, std::uint64_t want, std::uint64_t depth)
    -> SubResult {
    const std::uint64_t barrier = ++ctx.calls;
    const bool outer_truncated = ctx.truncated;
    ctx.truncated = false;
    SubResult r;
    sld_search(db, cons_goal(goal, barrier, nullptr), sub, ctx, want,
               depth, r.answers);
    r.truncated = ctx.truncated;
    ctx.truncated = outer_truncated || r.truncated;
    // A cut inside the sub-derivation is confined to it: consume the signal at the boundary so
    // it cannot prune the alternatives of whatever called us.
    if (ctx.cut_signal && *ctx.cut_signal >= barrier) {
        ctx.cut_signal.reset();
    }
    return r;
}

// Adds `extra` arguments to a callable term — the `call/N` construction, and also how
// `findall`'s goal argument is completed. `call(foo, a)` is `foo(a)`.
[[nodiscard]] auto add_args(const Term& goal, const std::vector<Term>& extra) -> Term {
    if (extra.empty()) {
        return goal;
    }
    if (!is_callable(goal)) {
        return goal;  // the caller validates; returning unchanged keeps this total
    }
    std::vector<Term> args = args_of(goal);
    args.insert(args.end(), extra.begin(), extra.end());
    return make_compound(name_of(goal), std::move(args));
}

// ---------------------------------------------------------------------------
// Deterministic builtins.
// ---------------------------------------------------------------------------

// Succeeds at most once. `std::nullopt` is an ordinary failure; an error is an honest refusal.
// `db` is mutable because the database-update builtins live here.
[[nodiscard]] auto call_det_builtin(Database& db, const std::string& ind,
                                    const std::vector<Term>& raw_args, const Substitution& sub,
                                    SearchCtx& ctx) -> Result<std::optional<Substitution>> {
    using Answer = std::optional<Substitution>;
    const auto ok = [](Substitution s) -> Result<Answer> { return Answer(std::move(s)); };
    const auto fail = []() -> Result<Answer> { return Answer(std::nullopt); };
    const auto err = [](MathError e) -> Result<Answer> { return make_error<Answer>(e); };

    // Every builtin below works on RESOLVED arguments: a builtin inspects the shape of its
    // argument, and the shape is only meaningful after the current bindings are applied.
    std::vector<Term> a;
    a.reserve(raw_args.size());
    for (const Term& t : raw_args) {
        a.push_back(apply_rec(sub, t, 0));
    }

    const auto bind = [&](const Term& lhs, const Term& rhs) -> Result<Answer> {
        auto u = unify_terms(lhs, rhs, sub);
        if (!u) {
            return fail();
        }
        return ok(std::move(*u));
    };
    const auto boolean = [&](bool cond) -> Result<Answer> {
        return cond ? ok(sub) : fail();
    };

    // ---- unification and term comparison ----
    if (ind == "=/2") {
        return bind(a[0], a[1]);
    }
    if (ind == "\\=/2") {
        return boolean(!unify_terms(a[0], a[1], sub).has_value());
    }
    if (ind == "==/2") {
        return boolean(compare_terms(a[0], a[1]) == 0);
    }
    if (ind == "\\==/2") {
        return boolean(compare_terms(a[0], a[1]) != 0);
    }
    if (ind == "@</2") {
        return boolean(compare_terms(a[0], a[1]) < 0);
    }
    if (ind == "@>/2") {
        return boolean(compare_terms(a[0], a[1]) > 0);
    }
    if (ind == "@=</2") {
        return boolean(compare_terms(a[0], a[1]) <= 0);
    }
    if (ind == "@>=/2") {
        return boolean(compare_terms(a[0], a[1]) >= 0);
    }
    if (ind == "compare/3") {
        const int c = compare_terms(a[1], a[2]);
        return bind(a[0], make_atom(c < 0 ? "<" : (c > 0 ? ">" : "=")));
    }

    // ---- type tests ----
    if (ind == "var/1") {
        return boolean(is_var(a[0]));
    }
    if (ind == "nonvar/1") {
        return boolean(!is_var(a[0]));
    }
    if (ind == "atom/1") {
        return boolean(is_atom(a[0]));
    }
    if (ind == "number/1" || ind == "integer/1") {
        return boolean(is_int(a[0]));
    }
    if (ind == "float/1") {
        // There is no float term kind and none is planned, so this is always false rather than
        // an error: `float(X)` is a legitimate question with the answer "no".
        return fail();
    }
    if (ind == "atomic/1") {
        return boolean(is_atom(a[0]) || is_int(a[0]));
    }
    if (ind == "compound/1") {
        return boolean(is_compound(a[0]));
    }
    if (ind == "callable/1") {
        return boolean(is_callable(a[0]));
    }
    if (ind == "is_list/1") {
        return boolean(is_proper_list(a[0]));
    }
    if (ind == "ground/1") {
        return boolean(is_ground(a[0]));
    }

    // ---- term construction and inspection ----
    if (ind == "functor/3") {
        if (!is_var(a[0])) {
            if (is_compound(a[0])) {
                const CompoundNode& c = compound_of(a[0]);
                auto u = unify_terms(a[1], make_atom(c.functor), sub);
                if (!u) {
                    return fail();
                }
                auto v = unify_terms(a[2], make_int(static_cast<std::int64_t>(c.args.size())), *u);
                if (!v) {
                    return fail();
                }
                return ok(std::move(*v));
            }
            auto u = unify_terms(a[1], a[0], sub);  // an atomic term is its own functor
            if (!u) {
                return fail();
            }
            auto v = unify_terms(a[2], make_int(0), *u);
            if (!v) {
                return fail();
            }
            return ok(std::move(*v));
        }
        // Construction mode needs both the name and the arity; without them there is nothing to
        // build and guessing would be inventing an answer.
        if (!is_int(a[2])) {
            return err(MathError::domain_error);
        }
        const std::int64_t arity = int_of(a[2]).value;
        if (arity < 0) {
            return err(MathError::domain_error);
        }
        if (arity == 0) {
            if (is_var(a[1])) {
                return err(MathError::domain_error);
            }
            return bind(a[0], a[1]);
        }
        if (!is_atom(a[1])) {
            return err(MathError::domain_error);  // only an atom can be a compound's functor
        }
        std::vector<Term> args;
        args.reserve(static_cast<std::size_t>(arity));
        for (std::int64_t i = 0; i < arity; ++i) {
            args.push_back(fresh_var(ctx));
        }
        return bind(a[0], make_compound(atom_of(a[1]).name, std::move(args)));
    }
    if (ind == "=../2") {
        if (!is_var(a[0])) {
            std::vector<Term> parts;
            if (is_compound(a[0])) {
                const CompoundNode& c = compound_of(a[0]);
                parts.push_back(make_atom(c.functor));
                parts.insert(parts.end(), c.args.begin(), c.args.end());
            } else {
                parts.push_back(a[0]);
            }
            return bind(a[1], make_list(std::move(parts)));
        }
        const auto items = list_to_vector(a[1]);
        if (!items || items->empty()) {
            return err(MathError::domain_error);
        }
        if (items->size() == 1) {
            return bind(a[0], items->front());
        }
        if (!is_atom(items->front())) {
            return err(MathError::domain_error);
        }
        std::vector<Term> args(items->begin() + 1, items->end());
        return bind(a[0], make_compound(atom_of(items->front()).name, std::move(args)));
    }
    if (ind == "copy_term/2") {
        return bind(a[1], copy_term_fresh(a[0], ctx));
    }
    if (ind == "term_variables/2") {
        return bind(a[1], make_list(term_variables_of(a[0])));
    }

    // ---- arithmetic ----
    if (ind == "is/2") {
        auto v = eval_arith(a[1], sub);
        if (!v) {
            return err(v.error());
        }
        return bind(a[0], make_int(*v));
    }
    if (ind == "=:=/2" || ind == "=\\=/2" || ind == "</2" || ind == ">/2" || ind == "=</2" ||
        ind == ">=/2") {
        const std::string op = ind.substr(0, ind.size() - 2);
        auto v = eval_arith_compare(op, a[0], a[1], sub);
        if (!v) {
            return err(v.error());
        }
        return boolean(*v);
    }
    if (ind == "succ/2") {
        if (is_int(a[0])) {
            const std::int64_t x = int_of(a[0]).value;
            if (x < 0) {
                return err(MathError::domain_error);  // succ/2 is defined on naturals only
            }
            if (x == std::numeric_limits<std::int64_t>::max()) {
                return err(MathError::overflow);
            }
            return bind(a[1], make_int(x + 1));
        }
        if (is_int(a[1])) {
            const std::int64_t y = int_of(a[1]).value;
            if (y <= 0) {
                return fail();  // no natural predecessor of 0; an honest failure, not an error
            }
            return bind(a[0], make_int(y - 1));
        }
        return err(MathError::domain_error);
    }
    if (ind == "plus/3") {
        const bool k0 = is_int(a[0]);
        const bool k1 = is_int(a[1]);
        const bool k2 = is_int(a[2]);
        if (k0 && k1) {
            auto v = eval_arith(make_compound("+", {a[0], a[1]}), sub);
            if (!v) {
                return err(v.error());
            }
            return bind(a[2], make_int(*v));
        }
        if (k0 && k2) {
            auto v = eval_arith(make_compound("-", {a[2], a[0]}), sub);
            if (!v) {
                return err(v.error());
            }
            return bind(a[1], make_int(*v));
        }
        if (k1 && k2) {
            auto v = eval_arith(make_compound("-", {a[2], a[1]}), sub);
            if (!v) {
                return err(v.error());
            }
            return bind(a[0], make_int(*v));
        }
        return err(MathError::domain_error);
    }

    // ---- sorting ----
    if (ind == "msort/2" || ind == "sort/2") {
        const auto items = list_to_vector(a[0]);
        if (!items) {
            return err(MathError::domain_error);
        }
        auto sorted = ind == "sort/2" ? sort_terms_unique(*items)
                                      : sort_terms_keep_duplicates(*items);
        return bind(a[1], make_list(std::move(sorted)));
    }
    if (ind == "keysort/2") {
        const auto items = list_to_vector(a[0]);
        if (!items) {
            return err(MathError::domain_error);
        }
        auto sorted = keysort_pairs(*items);
        if (!sorted) {
            return err(MathError::domain_error);
        }
        return bind(a[1], make_list(std::move(*sorted)));
    }
    if (ind == "sort/4") {
        if (!is_int(a[0]) || !is_atom(a[1])) {
            return err(MathError::domain_error);
        }
        const std::int64_t key = int_of(a[0]).value;
        if (key < 0) {
            return err(MathError::domain_error);
        }
        const auto items = list_to_vector(a[2]);
        if (!items) {
            return err(MathError::domain_error);
        }
        auto sorted =
            sort_terms_general(*items, static_cast<std::size_t>(key), atom_of(a[1]).name);
        if (!sorted) {
            return err(MathError::domain_error);
        }
        return bind(a[3], make_list(std::move(*sorted)));
    }
    if (ind == "numlist/3") {
        if (!is_int(a[0]) || !is_int(a[1])) {
            return err(MathError::domain_error);
        }
        const std::int64_t lo = int_of(a[0]).value;
        const std::int64_t hi = int_of(a[1]).value;
        if (lo > hi) {
            return fail();
        }
        // The list is materialised, so its length has to fit the step budget rather than
        // silently allocating gigabytes for numlist(1, 10^18, L).
        const std::uint64_t count = static_cast<std::uint64_t>(hi - lo) + 1;
        if (count > ctx.steps) {
            ctx.truncated = true;
            return err(MathError::not_converged);
        }
        std::vector<Term> xs;
        xs.reserve(static_cast<std::size_t>(count));
        for (std::int64_t v = lo; v <= hi; ++v) {
            xs.push_back(make_int(v));
        }
        return bind(a[2], make_list(std::move(xs)));
    }

    // ---- atoms and characters ----
    if (ind == "atom_length/2") {
        const auto text = atomic_text(a[0]);
        if (!text) {
            return err(MathError::domain_error);
        }
        return bind(a[1], make_int(static_cast<std::int64_t>(text->size())));
    }
    if (ind == "atom_codes/2" || ind == "atom_chars/2") {
        const bool codes = ind == "atom_codes/2";
        const auto text = atomic_text(a[0]);
        if (text) {
            return bind(a[1], codes ? text_to_code_list(*text) : text_to_char_list(*text));
        }
        if (!is_var(a[0])) {
            return err(MathError::domain_error);
        }
        const auto back = code_list_to_text(a[1]);
        if (!back) {
            return err(MathError::domain_error);
        }
        return bind(a[0], make_atom(*back));
    }
    if (ind == "char_code/2") {
        if (is_atom(a[0]) && atom_of(a[0]).name.size() == 1) {
            const auto ch = static_cast<unsigned char>(atom_of(a[0]).name.front());
            return bind(a[1], make_int(static_cast<std::int64_t>(ch)));
        }
        if (is_int(a[1])) {
            const std::int64_t v = int_of(a[1]).value;
            if (v < 0 || v > 255) {
                return err(MathError::domain_error);
            }
            return bind(a[0], make_atom(std::string(1, static_cast<char>(v))));
        }
        return err(MathError::domain_error);
    }
    if (ind == "number_codes/2" || ind == "number_chars/2") {
        const bool codes = ind == "number_codes/2";
        if (is_int(a[0])) {
            const std::string text = std::format("{}", int_of(a[0]).value);
            return bind(a[1], codes ? text_to_code_list(text) : text_to_char_list(text));
        }
        const auto back = code_list_to_text(a[1]);
        if (!back) {
            return err(MathError::domain_error);
        }
        const auto n = text_to_integer(*back);
        if (!n) {
            // Unlike atom_number/2 this is a genuine error in ISO: the text was required to
            // denote a number and did not.
            return err(MathError::syntax_error);
        }
        return bind(a[0], make_int(*n));
    }
    if (ind == "atom_number/2") {
        if (is_int(a[1])) {
            return bind(a[0], make_atom(std::format("{}", int_of(a[1]).value)));
        }
        const auto text = atomic_text(a[0]);
        if (!text) {
            return err(MathError::domain_error);
        }
        const auto n = text_to_integer(*text);
        if (!n) {
            return fail();  // atom_number/2 FAILS on non-numeric text; that is its contract
        }
        return bind(a[1], make_int(*n));
    }
    if (ind == "upcase_atom/2" || ind == "downcase_atom/2") {
        const auto text = atomic_text(a[0]);
        if (!text) {
            return err(MathError::domain_error);
        }
        const bool up = ind == "upcase_atom/2";
        std::string out;
        out.reserve(text->size());
        for (const char ch : *text) {
            // ASCII only, deliberately: case mapping outside ASCII is locale- and
            // encoding-dependent, and this engine has no encoding model to do it honestly.
            const auto u = static_cast<unsigned char>(ch);
            if (up && u >= 'a' && u <= 'z') {
                out.push_back(static_cast<char>(u - 32));
            } else if (!up && u >= 'A' && u <= 'Z') {
                out.push_back(static_cast<char>(u + 32));
            } else {
                out.push_back(ch);
            }
        }
        return bind(a[1], make_atom(std::move(out)));
    }

    // ---- database updates ----
    if (ind == "assert/1" || ind == "assertz/1" || ind == "asserta/1") {
        // The clause is COPIED with fresh variables. Asserting the runtime term itself would
        // leave the stored clause sharing variables with the derivation that asserted it, so a
        // later binding here would silently change a clause already in the database.
        auto clause = term_to_clause(copy_term_fresh(a[0], ctx));
        if (!clause) {
            return err(MathError::domain_error);
        }
        const auto added = ind == "asserta/1" ? db.asserta(std::move(*clause))
                                              : db.assertz(std::move(*clause));
        if (!added) {
            return err(added.error());
        }
        return ok(sub);
    }
    if (ind == "retractall/1") {
        if (!is_callable(a[0])) {
            return err(MathError::domain_error);
        }
        std::vector<DatabaseAccess::Entry>& entries = DatabaseAccess::entries_mut(db);
        for (DatabaseAccess::Entry& e : entries) {
            if (!e.live) {
                continue;
            }
            const std::uint64_t g = ++ctx.gen;
            const Clause renamed = rename_clause(e.clause, g);
            if (unify_terms(a[0], renamed.head, sub)) {
                e.live = false;
            }
        }
        return ok(sub);  // retractall always succeeds, even when nothing matched
    }
    if (ind == "abolish/1") {
        // abolish(Name/Arity): the argument is an indicator, not a head.
        if (!is_compound(a[0]) || compound_of(a[0]).functor != "/" ||
            compound_of(a[0]).args.size() != 2) {
            return err(MathError::domain_error);
        }
        const Term& n = compound_of(a[0]).args[0];
        const Term& ar = compound_of(a[0]).args[1];
        if (!is_atom(n) || !is_int(ar) || int_of(ar).value < 0) {
            return err(MathError::domain_error);
        }
        const std::string target =
            indicator(atom_of(n).name, static_cast<std::size_t>(int_of(ar).value));
        std::vector<DatabaseAccess::Entry>& entries = DatabaseAccess::entries_mut(db);
        for (DatabaseAccess::Entry& e : entries) {
            const auto head_ind = indicator_of(e.clause.head);
            if (e.live && head_ind && *head_ind == target) {
                e.live = false;
            }
        }
        return ok(sub);
    }

    return err(MathError::not_implemented);
}

}  // namespace

namespace {

// Continues the conjunction once for each candidate substitution, in order. This is the shape
// every NONDETERMINISTIC builtin needs: it is the builtin's own choice point, and it obeys the
// same three stopping conditions as the clause loop — an honest error, a cut in flight, and
// the answer cap.
auto continue_each(Database& db, const std::vector<Substitution>& candidates,
                   const GoalList& goals, SearchCtx& ctx, std::uint64_t max_solutions,
                   std::uint64_t depth, std::vector<Substitution>& out) -> void {
    const GoalList rest = goals->next;
    for (const Substitution& s : candidates) {
        if (ctx.steps == 0) {
            ctx.truncated = true;
            return;
        }
        --ctx.steps;
        sld_search(db, rest, s, ctx, max_solutions, depth + 1, out);
        if (ctx.error || ctx.cut_signal) {
            return;
        }
        if (max_solutions != 0 && out.size() >= max_solutions) {
            return;
        }
    }
}

// Strips the `^/2` existential markers from a bagof/setof goal, collecting the quantified
// variables. `X^Y^Goal` means "the bag may vary with neither X nor Y".
auto strip_carets(const Term& t, std::vector<VarKey>& quantified) -> Term {
    Term cur = t;
    while (is_compound(cur) && compound_of(cur).functor == "^" &&
           compound_of(cur).args.size() == 2) {
        collect_vars(compound_of(cur).args[0], quantified);
        cur = compound_of(cur).args[1];
    }
    return cur;
}

// The all-solutions core shared by findall, bagof and setof: every answer to `goal`, with
// `tmpl` instantiated and copied out. A truncated sub-search is `not_converged` rather than a
// short list, because a list that is missing answers is not a smaller answer — it is a wrong
// one, and the caller cannot tell the difference.
[[nodiscard]] auto collect_solutions(Database& db, const Term& tmpl, const Term& goal,
                                     const Substitution& sub, SearchCtx& ctx, std::uint64_t depth)
    -> Result<std::vector<Term>> {
    if (!validate_goal(goal)) {
        return make_error<std::vector<Term>>(MathError::domain_error);
    }
    SubResult r = run_sub(db, goal, sub, ctx, 0, depth);
    if (ctx.error) {
        return make_error<std::vector<Term>>(*ctx.error);
    }
    if (r.truncated) {
        return make_error<std::vector<Term>>(MathError::not_converged);
    }
    std::vector<Term> items;
    items.reserve(r.answers.size());
    for (const Substitution& s : r.answers) {
        // Copied, not merely resolved: the collected terms outlive the derivation that produced
        // them, and must not keep sharing its variables.
        items.push_back(copy_term_fresh(apply_rec(s, tmpl, 0), ctx));
    }
    return items;
}

}  // namespace

namespace {

auto sld_search(Database& db, const GoalList& goals, const Substitution& sub,
                SearchCtx& ctx, std::uint64_t max_solutions, std::uint64_t depth,
                std::vector<Substitution>& out) -> void {
    if (ctx.error) {
        return;  // an honest failure already occurred; abandon the whole search
    }
    if (ctx.cut_signal) {
        return;  // a cut is unwinding to its barrier; offer no further alternatives
    }
    if (max_solutions != 0 && out.size() >= max_solutions) {
        return;
    }
    if (ctx.steps == 0 || depth > max_derivation_depth) {
        ctx.truncated = true;  // a budget fired: this search did NOT exhaust the space
        return;
    }
    if (!goals) {
        out.push_back(sub);  // empty conjunction proved: an answer
        return;
    }

    const GoalNode& frame = *goals;
    const std::uint64_t barrier = frame.barrier;
    const Term goal = walk(frame.goal, sub);

    if (is_var(goal)) {
        ctx.error = MathError::domain_error;  // an uninstantiated goal is not callable
        return;
    }
    if (!is_callable(goal)) {
        ctx.error = MathError::domain_error;
        return;
    }
    // A goal that arrived through a variable never passed static validation, so the same rules
    // are applied again here — testing only `is_callable` would let a malformed `\+`/2 through
    // as an ordinary predicate and make the negation quietly succeed.
    if (!validate_goal(goal)) {
        ctx.error = MathError::domain_error;
        return;
    }

    // `is_callable` above means this is an atom or a compound, and both have an indicator.
    // Checking it anyway costs one branch and matches the two guards immediately above,
    // rather than resting a dereference on an invariant established in another function.
    const auto ind_opt = indicator_of(goal);
    if (!ind_opt) {
        ctx.error = MathError::domain_error;
        return;
    }
    const std::string ind = *ind_opt;
    const std::vector<Term> args = args_of(goal);
    const GoalList rest = goals->next;

    // ---- control constructs ---------------------------------------------------------------
    if (ind == "true/0") {
        --ctx.steps;
        sld_search(db, rest, sub, ctx, max_solutions, depth + 1, out);
        return;
    }
    if (ind == "fail/0" || ind == "false/0") {
        --ctx.steps;
        return;
    }
    if (ind == "!/0") {
        --ctx.steps;
        // The continuation runs FIRST and in full: cut prunes backtracking, not forward
        // execution, so everything to the right of the `!` still gets all its solutions.
        sld_search(db, rest, sub, ctx, max_solutions, depth + 1, out);
        if (ctx.error) {
            return;
        }
        // A cut already in flight from the continuation targets an OUTER barrier and therefore
        // prunes strictly more; keeping the smaller one preserves it.
        if (!ctx.cut_signal || *ctx.cut_signal > barrier) {
            ctx.cut_signal = barrier;
        }
        return;
    }
    if (ind == ",/2") {
        --ctx.steps;
        sld_search(db,
                   cons_goal(args[0], barrier, cons_goal(args[1], barrier, goals->next)),
                   sub, ctx, max_solutions, depth + 1, out);
        return;
    }
    if (ind == ";/2") {
        --ctx.steps;
        const Term& lhs = args[0];
        // `(C -> T ; E)` and `(C *-> T ; E)` are one construct, not a disjunction whose left
        // branch happens to be an if-then: the else branch belongs to the condition.
        const bool is_ite = is_compound(lhs) && compound_of(lhs).args.size() == 2 &&
                            (compound_of(lhs).functor == "->" || compound_of(lhs).functor == "*->");
        if (is_ite) {
            const bool soft = compound_of(lhs).functor == "*->";
            const Term cond = compound_of(lhs).args[0];
            const Term then_goal = compound_of(lhs).args[1];
            // The condition is opaque to cut and, for `->`, committed to its FIRST solution.
            SubResult r = run_sub(db, cond, sub, ctx, soft ? 0 : 1, depth + 1);
            if (ctx.error) {
                return;
            }
            if (r.answers.empty()) {
                if (r.truncated) {
                    // "The condition did not finish" is not "the condition is false"; taking the
                    // else branch here would make the answer a fact about the budget.
                    ctx.error = MathError::not_converged;
                    return;
                }
                sld_search(db, cons_goal(args[1], barrier, goals->next),
                           sub, ctx, max_solutions, depth + 1, out);
                return;
            }
            if (soft && r.truncated) {
                ctx.error = MathError::not_converged;
                return;
            }
            // The then-branch is transparent to cut: a `!` in it cuts the enclosing clause.
            const GoalList then_goals =
                cons_goal(then_goal, barrier, goals->next);
            for (const Substitution& s : r.answers) {
                sld_search(db, then_goals, s, ctx, max_solutions, depth + 1, out);
                if (ctx.error || ctx.cut_signal) {
                    return;
                }
                if (max_solutions != 0 && out.size() >= max_solutions) {
                    return;
                }
            }
            return;
        }
        // A plain disjunction is transparent to cut in BOTH branches: `(a, ! ; b)` cuts the
        // clause that contains the disjunction, and having cut, must not try `b`.
        sld_search(db, cons_goal(args[0], barrier, goals->next), sub,
                   ctx, max_solutions, depth + 1, out);
        if (ctx.error || ctx.cut_signal) {
            return;
        }
        if (max_solutions != 0 && out.size() >= max_solutions) {
            return;
        }
        sld_search(db, cons_goal(args[1], barrier, goals->next), sub,
                   ctx, max_solutions, depth + 1, out);
        return;
    }
    if (ind == "->/2" || ind == "*->/2") {
        // A bare if-then with no else: the missing else is `fail`.
        --ctx.steps;
        const bool soft = ind == "*->/2";
        SubResult r = run_sub(db, args[0], sub, ctx, soft ? 0 : 1, depth + 1);
        if (ctx.error) {
            return;
        }
        if (r.truncated && r.answers.empty()) {
            ctx.error = MathError::not_converged;
            return;
        }
        const GoalList then_goals =
            cons_goal(args[1], barrier, goals->next);
        for (const Substitution& s : r.answers) {
            sld_search(db, then_goals, s, ctx, max_solutions, depth + 1, out);
            if (ctx.error || ctx.cut_signal) {
                return;
            }
            if (max_solutions != 0 && out.size() >= max_solutions) {
                return;
            }
        }
        return;
    }
    if (ind == "\\+/1") {
        --ctx.steps;
        const Term inner = apply_rec(sub, args[0], 0);
        if (!validate_goal(inner)) {
            ctx.error = MathError::domain_error;
            return;
        }
        if (!is_ground(inner)) {
            // FLOUNDERING. Prolog answers anyway — unsoundly, and differently depending on goal
            // order. Refusing is the only answer this solver can stand behind.
            ctx.error = MathError::domain_error;
            return;
        }
        SubResult r = run_sub(db, inner, sub, ctx, 1, depth + 1);
        if (ctx.error) {
            return;
        }
        if (!r.answers.empty()) {
            return;  // G is provable, so `\+ G` FAILS — ordinary backtracking, not an error
        }
        if (r.truncated) {
            // No witness, but the space was not searched out. Concluding "not provable" here is
            // negation as BUDGET EXHAUSTION: the same query could answer differently tomorrow.
            ctx.error = MathError::not_converged;
            return;
        }
        // `\+ G` binds NOTHING on success: there is no witness to bind, which is why it is a
        // test and not a generator.
        sld_search(db, rest, sub, ctx, max_solutions, depth + 1, out);
        return;
    }
    if (ind.starts_with("call/")) {
        --ctx.steps;
        const Term target = apply_rec(sub, args[0], 0);
        const std::vector<Term> extra(args.begin() + 1, args.end());
        const Term built = add_args(target, extra);
        if (!is_callable(built) || !validate_goal(built)) {
            ctx.error = MathError::domain_error;
            return;
        }
        // `call/N` is opaque to cut: a `!` inside it is local to the called goal. A fresh
        // barrier is what makes that true, and consuming the signal here is what confines it.
        const std::uint64_t inner_barrier = ++ctx.calls;
        sld_search(db, cons_goal(built, inner_barrier, goals->next), sub,
                   ctx, max_solutions, depth + 1, out);
        if (ctx.cut_signal && *ctx.cut_signal == inner_barrier) {
            ctx.cut_signal.reset();
        }
        return;
    }
    if (ind == "forall/2") {
        --ctx.steps;
        // `forall(C, A)` succeeds iff A is provable for EVERY solution of C.
        //
        // The textbook definition is `\+ (C, \+ A)`, and expanding it that way here would be
        // WRONG. This engine refuses a non-ground negated goal (floundering), and C's variables
        // are exactly the ones C is there to bind — they are bound by the generator, not free,
        // so refusing them would reject every useful forall. Running the two sub-searches
        // directly says what forall means without borrowing negation's groundness rule.
        //
        // Truncation is fatal at BOTH levels, for the usual reason: "the generator did not
        // finish" and "the action did not finish" would each turn a fact about the budget into
        // a claim about the program.
        const SubResult gen = run_sub(db, apply_rec(sub, args[0], 0), sub, ctx, 0, depth + 1);
        if (ctx.error) {
            return;
        }
        if (gen.truncated) {
            ctx.error = MathError::not_converged;
            return;
        }
        for (const Substitution& witness : gen.answers) {
            const SubResult act =
                run_sub(db, apply_rec(witness, args[1], 0), witness, ctx, 1, depth + 1);
            if (ctx.error) {
                return;
            }
            if (act.answers.empty()) {
                if (act.truncated) {
                    ctx.error = MathError::not_converged;
                    return;
                }
                return;  // a counterexample: forall FAILS, which is an ordinary failure
            }
        }
        // Vacuously true when the generator had no solutions, and binding nothing either way:
        // forall is a test over C's solutions, not a generator of them.
        sld_search(db, rest, sub, ctx, max_solutions, depth + 1, out);
        return;
    }
    if (ind == "findall/3" || ind == "findall/4") {
        --ctx.steps;
        auto items = collect_solutions(db, args[0], apply_rec(sub, args[1], 0), sub, ctx, depth + 1);
        if (!items) {
            ctx.error = items.error();
            return;
        }
        // findall/4 appends to a tail rather than closing the list with [].
        Term result = ind == "findall/4" ? apply_rec(sub, args[3], 0) : make_nil();
        for (std::size_t i = items->size(); i-- > 0;) {
            result = make_compound(".", {(*items)[i], std::move(result)});
        }
        auto u = unify_terms(args[2], result, sub);
        if (!u) {
            return;
        }
        sld_search(db, rest, *u, ctx, max_solutions, depth + 1, out);
        return;
    }
    if (ind == "bagof/3" || ind == "setof/3") {
        --ctx.steps;
        const bool want_set = ind == "setof/3";
        std::vector<VarKey> quantified;
        const Term inner_goal = strip_carets(apply_rec(sub, args[1], 0), quantified);
        collect_vars(apply_rec(sub, args[0], 0), quantified);  // template vars are not free

        std::vector<VarKey> goal_vars;
        collect_vars(inner_goal, goal_vars);
        std::vector<Term> free_vars;
        for (const VarKey& k : goal_vars) {
            if (!std::ranges::any_of(quantified,
                                     [&k](const VarKey& q) -> bool { return q == k; })) {
                free_vars.push_back(make_var(k.name, k.generation));
            }
        }

        const Term witness = make_list(free_vars);
        auto pairs = collect_solutions(db, make_compound("-", {witness, args[0]}), inner_goal, sub,
                                       ctx, depth + 1);
        if (!pairs) {
            ctx.error = pairs.error();
            return;
        }
        if (pairs->empty()) {
            return;  // bagof/setof FAIL on no solutions; that is what distinguishes them
        }
        if (free_vars.empty()) {
            std::vector<Term> items;
            items.reserve(pairs->size());
            for (const Term& p : *pairs) {
                items.push_back(compound_of(p).args[1]);
            }
            if (want_set) {
                items = sort_terms_unique(std::move(items));
            }
            auto u = unify_terms(args[2], make_list(std::move(items)), sub);
            if (!u) {
                return;
            }
            sld_search(db, rest, *u, ctx, max_solutions, depth + 1, out);
            return;
        }
        // Group by the witness in the standard order — the ISO rule that bagof BACKTRACKS over
        // distinct bindings of the free variables rather than lumping them into one bag.
        std::vector<Term> keys;
        for (const Term& p : *pairs) {
            const Term& k = compound_of(p).args[0];
            if (!std::ranges::any_of(keys,
                                     [&k](const Term& s) -> bool { return compare_terms(s, k) == 0; })) {
                keys.push_back(k);
            }
        }
        keys = sort_terms_unique(std::move(keys));
        std::vector<Substitution> candidates;
        for (const Term& k : keys) {
            std::vector<Term> items;
            for (const Term& p : *pairs) {
                if (compare_terms(compound_of(p).args[0], k) == 0) {
                    items.push_back(compound_of(p).args[1]);
                }
            }
            if (want_set) {
                items = sort_terms_unique(std::move(items));
            }
            auto u = unify_terms(witness, k, sub);
            if (!u) {
                continue;
            }
            auto v = unify_terms(args[2], make_list(std::move(items)), *u);
            if (!v) {
                continue;
            }
            candidates.push_back(std::move(*v));
        }
        continue_each(db, candidates, goals, ctx, max_solutions, depth, out);
        return;
    }

    // ---- nondeterministic builtins ---------------------------------------------------------
    if (ind == "between/3") {
        --ctx.steps;
        if (!is_int(args[0]) && !is_int(apply_rec(sub, args[0], 0))) {
            ctx.error = MathError::domain_error;
            return;
        }
        const Term lo_t = apply_rec(sub, args[0], 0);
        const Term hi_t = apply_rec(sub, args[1], 0);
        const Term x_t = apply_rec(sub, args[2], 0);
        if (!is_int(lo_t) || !is_int(hi_t)) {
            ctx.error = MathError::domain_error;
            return;
        }
        const std::int64_t lo = int_of(lo_t).value;
        const std::int64_t hi = int_of(hi_t).value;
        if (is_int(x_t)) {
            const std::int64_t x = int_of(x_t).value;
            if (x < lo || x > hi) {
                return;
            }
            sld_search(db, rest, sub, ctx, max_solutions, depth + 1, out);
            return;
        }
        // Enumerating lazily rather than materialising the range is what lets between/3 drive a
        // loop over a large interval without allocating it.
        for (std::int64_t v = lo; v <= hi; ++v) {
            if (ctx.steps == 0) {
                ctx.truncated = true;
                return;
            }
            --ctx.steps;
            auto u = unify_terms(args[2], make_int(v), sub);
            if (u) {
                sld_search(db, rest, *u, ctx, max_solutions, depth + 1, out);
                if (ctx.error || ctx.cut_signal) {
                    return;
                }
                if (max_solutions != 0 && out.size() >= max_solutions) {
                    return;
                }
            }
            if (v == hi) {
                break;  // guards the overflow when hi is the largest representable integer
            }
        }
        return;
    }
    if (ind == "length/2") {
        --ctx.steps;
        const Term list_t = apply_rec(sub, args[0], 0);
        const Term len_t = apply_rec(sub, args[1], 0);
        if (const auto items = list_to_vector(list_t)) {
            auto u = unify_terms(args[1], make_int(static_cast<std::int64_t>(items->size())), sub);
            if (!u) {
                return;
            }
            sld_search(db, rest, *u, ctx, max_solutions, depth + 1, out);
            return;
        }
        if (is_int(len_t)) {
            const std::int64_t n = int_of(len_t).value;
            if (n < 0) {
                return;
            }
            if (static_cast<std::uint64_t>(n) > ctx.steps) {
                ctx.truncated = true;
                ctx.error = MathError::not_converged;
                return;
            }
            std::vector<Term> xs;
            xs.reserve(static_cast<std::size_t>(n));
            for (std::int64_t i = 0; i < n; ++i) {
                xs.push_back(fresh_var(ctx));
            }
            auto u = unify_terms(args[0], make_list(std::move(xs)), sub);
            if (!u) {
                return;
            }
            sld_search(db, rest, *u, ctx, max_solutions, depth + 1, out);
            return;
        }
        // Both ends open: enumerate lengths upward. The step budget is the only thing that ends
        // this, which is honest — `length(L, N)` genuinely has infinitely many solutions.
        for (std::int64_t n = 0;; ++n) {
            if (ctx.steps == 0) {
                ctx.truncated = true;
                return;
            }
            --ctx.steps;
            std::vector<Term> xs;
            xs.reserve(static_cast<std::size_t>(n));
            for (std::int64_t i = 0; i < n; ++i) {
                xs.push_back(fresh_var(ctx));
            }
            auto u = unify_terms(args[0], make_list(std::move(xs)), sub);
            if (u) {
                auto v = unify_terms(args[1], make_int(n), *u);
                if (v) {
                    sld_search(db, rest, *v, ctx, max_solutions, depth + 1, out);
                    if (ctx.error || ctx.cut_signal) {
                        return;
                    }
                    if (max_solutions != 0 && out.size() >= max_solutions) {
                        return;
                    }
                }
            }
        }
    }
    if (ind == "arg/3") {
        --ctx.steps;
        const Term n_t = apply_rec(sub, args[0], 0);
        const Term t_t = apply_rec(sub, args[1], 0);
        if (!is_compound(t_t)) {
            ctx.error = MathError::domain_error;
            return;
        }
        const CompoundNode& c = compound_of(t_t);
        if (is_int(n_t)) {
            const std::int64_t n = int_of(n_t).value;
            if (n < 1 || static_cast<std::size_t>(n) > c.args.size()) {
                return;
            }
            auto u = unify_terms(args[2], c.args[static_cast<std::size_t>(n) - 1], sub);
            if (!u) {
                return;
            }
            sld_search(db, rest, *u, ctx, max_solutions, depth + 1, out);
            return;
        }
        std::vector<Substitution> candidates;
        for (std::size_t i = 0; i < c.args.size(); ++i) {
            auto u = unify_terms(args[0], make_int(static_cast<std::int64_t>(i) + 1), sub);
            if (!u) {
                continue;
            }
            auto v = unify_terms(args[2], c.args[i], *u);
            if (v) {
                candidates.push_back(std::move(*v));
            }
        }
        continue_each(db, candidates, goals, ctx, max_solutions, depth, out);
        return;
    }
    if (ind == "nth0/3" || ind == "nth1/3") {
        --ctx.steps;
        const std::int64_t base = ind == "nth1/3" ? 1 : 0;
        const auto items = list_to_vector(apply_rec(sub, args[1], 0));
        if (!items) {
            ctx.error = MathError::domain_error;
            return;
        }
        std::vector<Substitution> candidates;
        for (std::size_t i = 0; i < items->size(); ++i) {
            auto u = unify_terms(args[0], make_int(static_cast<std::int64_t>(i) + base), sub);
            if (!u) {
                continue;
            }
            auto v = unify_terms(args[2], (*items)[i], *u);
            if (v) {
                candidates.push_back(std::move(*v));
            }
        }
        continue_each(db, candidates, goals, ctx, max_solutions, depth, out);
        return;
    }
    if (ind == "atom_concat/3") {
        --ctx.steps;
        const auto x = atomic_text(apply_rec(sub, args[0], 0));
        const auto y = atomic_text(apply_rec(sub, args[1], 0));
        const auto z = atomic_text(apply_rec(sub, args[2], 0));
        if (x && y) {
            auto u = unify_terms(args[2], make_atom(*x + *y), sub);
            if (!u) {
                return;
            }
            sld_search(db, rest, *u, ctx, max_solutions, depth + 1, out);
            return;
        }
        if (!z) {
            ctx.error = MathError::domain_error;  // nothing to split and nothing to join
            return;
        }
        // The split mode is genuinely nondeterministic: every division of the whole is a
        // solution, enumerated left to right.
        std::vector<Substitution> candidates;
        for (std::size_t i = 0; i <= z->size(); ++i) {
            auto u = unify_terms(args[0], make_atom(z->substr(0, i)), sub);
            if (!u) {
                continue;
            }
            auto v = unify_terms(args[1], make_atom(z->substr(i)), *u);
            if (v) {
                candidates.push_back(std::move(*v));
            }
        }
        continue_each(db, candidates, goals, ctx, max_solutions, depth, out);
        return;
    }
    if (ind == "sub_atom/5") {
        --ctx.steps;
        const auto whole = atomic_text(apply_rec(sub, args[0], 0));
        if (!whole) {
            ctx.error = MathError::domain_error;
            return;
        }
        const std::size_t n = whole->size();
        std::vector<Substitution> candidates;
        for (std::size_t b = 0; b <= n; ++b) {
            for (std::size_t len = 0; b + len <= n; ++len) {
                auto u = unify_terms(args[1], make_int(static_cast<std::int64_t>(b)), sub);
                if (!u) {
                    continue;
                }
                auto v = unify_terms(args[2], make_int(static_cast<std::int64_t>(len)), *u);
                if (!v) {
                    continue;
                }
                auto w = unify_terms(
                    args[3], make_int(static_cast<std::int64_t>(n - b - len)), *v);
                if (!w) {
                    continue;
                }
                auto s = unify_terms(args[4], make_atom(whole->substr(b, len)), *w);
                if (s) {
                    candidates.push_back(std::move(*s));
                }
            }
        }
        continue_each(db, candidates, goals, ctx, max_solutions, depth, out);
        return;
    }
    if (ind == "clause/2") {
        --ctx.steps;
        const Term head = apply_rec(sub, args[0], 0);
        if (!is_callable(head)) {
            ctx.error = MathError::domain_error;
            return;
        }
        std::vector<Substitution> candidates;
        for (const std::size_t ci : candidate_clauses(db, head, sub)) {
            const DatabaseAccess::Entry& e = DatabaseAccess::entries(db)[ci];
            const Clause renamed = rename_clause(e.clause, ++ctx.gen);
            auto u = unify_terms(head, renamed.head, sub);
            if (!u) {
                continue;
            }
            Term body = make_atom("true");
            for (std::size_t i = renamed.body.size(); i-- > 0;) {
                body = i + 1 == renamed.body.size()
                           ? renamed.body[i]
                           : make_compound(",", {renamed.body[i], std::move(body)});
            }
            auto v = unify_terms(args[1], body, *u);
            if (v) {
                candidates.push_back(std::move(*v));
            }
        }
        continue_each(db, candidates, goals, ctx, max_solutions, depth, out);
        return;
    }
    if (ind == "retract/1") {
        --ctx.steps;
        const Term pattern = apply_rec(sub, args[0], 0);
        const auto target = term_to_clause(pattern);
        if (!target) {
            ctx.error = MathError::domain_error;
            return;
        }
        // Retract removes a clause and succeeds; on backtracking it looks for the NEXT match
        // against the database as it now stands. The removal is not undone — that is the ISO
        // behaviour and the reason a retract loop terminates.
        const std::vector<std::size_t> cands = candidate_clauses(db, target->head, sub);
        for (const std::size_t ci : cands) {
            if (ctx.steps == 0) {
                ctx.truncated = true;
                return;
            }
            --ctx.steps;
            std::vector<DatabaseAccess::Entry>& entries = DatabaseAccess::entries_mut(db);
            if (!entries[ci].live) {
                continue;
            }
            const Clause renamed = rename_clause(entries[ci].clause, ++ctx.gen);
            auto u = unify_terms(target->head, renamed.head, sub);
            if (!u) {
                continue;
            }
            if (!target->body.empty()) {
                // A `Head :- Body` pattern must match the stored body too.
                if (renamed.body.size() != target->body.size()) {
                    continue;
                }
                bool matched = true;
                for (std::size_t i = 0; i < target->body.size() && matched; ++i) {
                    auto v = unify_terms(target->body[i], renamed.body[i], *u);
                    if (v) {
                        u = std::move(v);
                    } else {
                        matched = false;
                    }
                }
                if (!matched) {
                    continue;
                }
            }
            entries[ci].live = false;
            sld_search(db, rest, *u, ctx, max_solutions, depth + 1, out);
            if (ctx.error || ctx.cut_signal) {
                return;
            }
            if (max_solutions != 0 && out.size() >= max_solutions) {
                return;
            }
        }
        return;
    }

    // ---- deterministic builtins -------------------------------------------------------------
    if (det_builtins().contains(ind)) {
        --ctx.steps;
        auto r = call_det_builtin(db, ind, args, sub, ctx);
        if (!r) {
            ctx.error = r.error();
            return;
        }
        if (!*r) {
            return;  // an ordinary failure
        }
        sld_search(db, rest, **r, ctx, max_solutions, depth + 1, out);
        return;
    }

    // ---- ordinary clause resolution ----------------------------------------------------------
    // This call owns a cut barrier: a `!` in the body of any clause activated here discards the
    // remaining clauses of THIS call.
    const std::uint64_t my_call = ++ctx.calls;
    const std::vector<std::size_t> cands = candidate_clauses(db, goal, sub);
    for (const std::size_t ci : cands) {
        if (ctx.steps == 0) {
            ctx.truncated = true;
            return;
        }
        --ctx.steps;
        const std::vector<DatabaseAccess::Entry>& entries = DatabaseAccess::entries(db);
        if (ci >= entries.size() || !entries[ci].live) {
            continue;  // retracted by a side effect since the candidates were computed
        }
        const std::uint64_t g = ++ctx.gen;  // fresh generation per clause activation
        const Clause renamed = rename_clause(entries[ci].clause, g);
        auto unified = unify_terms(goal, renamed.head, sub);
        if (!unified) {
            continue;
        }
        // Folded from the BACK so the body reads left to right once consed onto `rest`.
        GoalList next = rest;
        for (std::size_t i = renamed.body.size(); i-- > 0;) {
            next = cons_goal(renamed.body[i], my_call, std::move(next));
        }
        sld_search(db, next, *unified, ctx, max_solutions, depth + 1, out);
        if (ctx.error) {
            return;
        }
        if (ctx.cut_signal) {
            // A signal naming this call is consumed here; one naming an outer call keeps
            // travelling. Either way this call offers no further clauses.
            if (*ctx.cut_signal >= my_call) {
                ctx.cut_signal.reset();
            }
            return;
        }
        if (max_solutions != 0 && out.size() >= max_solutions) {
            return;
        }
    }
}

}  // namespace

namespace {

// Whether a query is safe to fan out across clause branches.
//
// OR-parallelism speculatively evaluates EVERY clause of the first goal. That is sound only
// while a branch cannot observe or affect another. Two things break it:
//
//  * A CUT commits to one clause and discards the rest — but the rest have already run, and
//    their answers are already in the merge.
//  * A DATABASE UPDATE is a side effect. A branch that serial resolution would never have
//    reached can still perform one, and no merge rule can un-perform it.
//
// Both make the parallel solver disagree with the serial one, so a program containing either
// is run serially instead. Delegating rather than refusing keeps the contract — identical
// answers to `solve` — which is the promise callers actually rely on.
[[nodiscard]] auto term_has_unsafe_builtin(const Term& t, std::uint64_t depth) -> bool {
    if (depth > 1'000) {
        return true;  // too deep to certify; assume unsafe rather than guess it is safe
    }
    if (is_atom(t)) {
        return atom_of(t).name == "!";
    }
    if (!is_compound(t)) {
        return false;
    }
    const CompoundNode& c = compound_of(t);
    static const std::unordered_set<std::string> unsafe = {
        "assert/1", "asserta/1", "assertz/1", "retract/1", "retractall/1", "abolish/1",
    };
    if (unsafe.contains(indicator(c.functor, c.args.size()))) {
        return true;
    }
    return std::ranges::any_of(c.args, [depth](const Term& a) -> bool {
        return term_has_unsafe_builtin(a, depth + 1);
    });
}

[[nodiscard]] auto or_parallel_is_safe(const Program& program, const std::vector<Term>& goals)
    -> bool {
    for (const Clause& c : program) {
        if (term_has_unsafe_builtin(c.head, 0)) {
            return false;
        }
        for (const Term& g : c.body) {
            if (term_has_unsafe_builtin(g, 0)) {
                return false;
            }
        }
    }
    return std::ranges::none_of(
        goals, [](const Term& g) -> bool { return term_has_unsafe_builtin(g, 0); });
}

// Runs a validated query against `db`, restricting the raw answers to the query's variables.
[[nodiscard]] auto run_query(Database& db, const std::vector<Term>& goals,
                             std::uint64_t max_solutions) -> Result<std::vector<Substitution>> {
    const std::vector<VarKey> qvars = collect_query_vars(goals);
    // Barrier 0 is the query own: a cut written in the query prunes the query alternatives
    // and nothing outside, since nothing is outside.
    GoalList frames;
    for (std::size_t i = goals.size(); i-- > 0;) {
        frames = cons_goal(goals[i], 0, std::move(frames));
    }
    SearchCtx ctx;
    std::vector<Substitution> raw;
    sld_search(db, frames, Substitution{}, ctx, max_solutions, 0, raw);
    if (ctx.error) {
        return make_error<std::vector<Substitution>>(*ctx.error);
    }
    std::vector<Substitution> answers;
    answers.reserve(raw.size());
    for (const Substitution& s : raw) {
        answers.push_back(restrict_answer(s, qvars));
    }
    return answers;
}

}  // namespace

auto unify(const Term& a, const Term& b, const Substitution& in)
    -> Result<std::optional<Substitution>> {
    // Occurs-check failure is a normal non-unification (nullopt), never a MathError.
    return unify_terms(a, b, in);
}

auto apply_substitution(const Substitution& s, const Term& t) -> Term {
    return apply_rec(s, t, 0);
}

auto solve_in(Database& db, const std::vector<Term>& goals, std::uint64_t max_solutions)
    -> Result<std::vector<Substitution>> {
    if (!validate_program(db.clauses()) || !validate_goals(goals)) {
        return make_error<std::vector<Substitution>>(MathError::domain_error);
    }
    return run_query(db, goals, max_solutions);
}

auto solve(const Program& program, const std::vector<Term>& goals, std::uint64_t max_solutions)
    -> Result<std::vector<Substitution>> {
    // The query gets its OWN database, so `assert/1` and `retract/1` are visible for the rest of
    // this query and then discarded. A caller who wants updates to persist across queries holds
    // a Machine, which owns its database — the difference is deliberate and documented, because
    // a solver taking a `const Program&` cannot honestly claim to have mutated it.
    Database db = Database::from(program);
    return solve_in(db, goals, max_solutions);
}

auto solve_first(const Program& program, const std::vector<Term>& goals)
    -> Result<std::optional<Substitution>> {
    auto r = solve(program, goals, 1);
    if (!r) {
        return make_error<std::optional<Substitution>>(r.error());
    }
    if (r->empty()) {
        return std::optional<Substitution>{};
    }
    return std::optional<Substitution>(r->front());
}

auto solve_clause_branch(const Program& program, const std::vector<Term>& goals,
                         std::size_t clause_index, std::uint64_t max_solutions)
    -> Result<std::vector<Substitution>> {
    if (!validate_program(program) || !validate_goals(goals) || goals.empty()) {
        return make_error<std::vector<Substitution>>(MathError::domain_error);
    }
    const auto first_ind = indicator_of(goals.front());
    if (!first_ind) {
        return make_error<std::vector<Substitution>>(MathError::domain_error);
    }

    Database db = Database::from(program);
    const std::vector<DatabaseAccess::Entry>& entries = DatabaseAccess::entries(db);
    // A clause index past the end, or naming a different predicate, is NOT AN ALTERNATIVE —
    // and not being an alternative is not a failure. Reporting an error here would make a
    // caller that fans out over every clause of a mixed program fail on the first clause
    // belonging to some other predicate.
    if (clause_index >= entries.size()) {
        return std::vector<Substitution>{};
    }

    SearchCtx ctx;
    const std::uint64_t g = ++ctx.gen;
    const Clause renamed = rename_clause(entries[clause_index].clause, g);
    const auto head_ind = indicator_of(renamed.head);
    if (!head_ind || *head_ind != *first_ind) {
        return std::vector<Substitution>{};
    }

    std::vector<Substitution> raw;
    auto unified = unify_terms(goals.front(), renamed.head, Substitution{});
    if (unified) {
        const std::uint64_t my_call = ++ctx.calls;
        GoalList next;
        // Folded from the back so the clause body reads left to right ahead of the remaining
        // query goals. The query's own goals carry barrier 0; the body carries this call's.
        for (std::size_t i = goals.size(); i-- > 1;) {
            next = cons_goal(goals[i], 0, std::move(next));
        }
        for (std::size_t i = renamed.body.size(); i-- > 0;) {
            next = cons_goal(renamed.body[i], my_call, std::move(next));
        }
        // depth = 1 mirrors serial, where the continuation after the first resolution runs one
        // level deep, so the depth budget cuts at exactly the same point.
        sld_search(db, next, *unified, ctx, max_solutions, 1, raw);
    }
    if (ctx.error) {
        return make_error<std::vector<Substitution>>(*ctx.error);
    }

    const std::vector<VarKey> qvars = collect_query_vars(goals);
    std::vector<Substitution> out;
    out.reserve(raw.size());
    for (const Substitution& s : raw) {
        out.push_back(restrict_answer(s, qvars));
    }
    return out;
}

auto solve_or_parallel(const Program& program, const std::vector<Term>& goals,
                       std::uint64_t max_solutions) -> Result<std::vector<Substitution>> {
    if (!validate_program(program) || !validate_goals(goals)) {
        return make_error<std::vector<Substitution>>(MathError::domain_error);
    }

    const std::vector<VarKey> qvars = collect_query_vars(goals);

    // An empty conjunction is trivially true: one (empty) answer, matching serial solve.
    if (goals.empty()) {
        std::vector<Substitution> out;
        out.push_back(restrict_answer(Substitution{}, qvars));
        return out;
    }

    // A leading control construct or builtin has no clause alternatives to fan out over: this
    // decomposition is over the first goal's candidate CLAUSES, and a builtin resolves against
    // none of them. Running one through `branch` would unify it against clause heads and
    // silently enumerate nothing.
    const auto first_ind = indicator_of(goals.front());
    if (!first_ind || is_builtin(name_of(goals.front()), args_of(goals.front()).size()) ||
        is_var(goals.front()) || !or_parallel_is_safe(program, goals)) {
        return solve(program, goals, max_solutions);
    }

    const Term& first = goals.front();
    const std::vector<Term> rest_goals(goals.begin() + 1, goals.end());

    // OR-parallel branch for clause `ci`. The whole branch is `solve_clause_branch`, which is a
    // pure function of its arguments — no shared state, its own rename counter, step budget and
    // database copy — so branches are safe to run concurrently, and the SAME function is what a
    // distributed shard carries to another process. Sharing it is what stops the local and
    // distributed decompositions from drifting apart.
    //
    // The error travels WITH the branch in its Result rather than through a shared slot: branches
    // run concurrently, so one shared error slot would make WHICH failure is reported depend on
    // worker timing.
    const auto branch = [&](std::size_t ci) -> Result<std::vector<Substitution>> {
        return solve_clause_branch(program, goals, ci, max_solutions);
    };

    // grain = 1 so every clause becomes an independent task (the backend auto-chunks);
    // transform_index is order-preserving, so per-clause results come back in clause order.
    const std::vector<Result<std::vector<Substitution>>> per_clause =
        parallel::transform_index(program.size(), branch, 1);

    // Walk the branches in CLAUSE order, interleaving the answer cap with error reporting.
    // BOTH orderings are load-bearing:
    //
    //  * Errors surface in clause order rather than completion order, so which failure a caller
    //    sees never depends on which worker happened to finish first.
    //
    //  * The cap is consulted BEFORE a branch's error is. Serial solve stops the instant it has
    //    enough answers and NEVER TRIES the later clauses, so a branch serial would not have
    //    reached must not be able to fail the query here. Every branch is speculatively
    //    evaluated — that is what makes it parallel — but a speculative failure is not a result.
    std::vector<Substitution> out;
    for (const Result<std::vector<Substitution>>& b : per_clause) {
        if (max_solutions != 0 && out.size() >= max_solutions) {
            return out;  // serial had already stopped; this clause was never reached
        }
        if (!b) {
            return make_error<std::vector<Substitution>>(b.error());
        }
        for (const Substitution& answer : *b) {
            if (max_solutions != 0 && out.size() >= max_solutions) {
                return out;
            }
            out.push_back(answer);  // already restricted by solve_clause_branch
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Machine — a database that persists across queries.
// ---------------------------------------------------------------------------

auto Machine::create() -> Machine { return Machine{}; }

auto Machine::from(Program p) -> Machine {
    Machine m;
    m.db_ = Database::from(std::move(p));
    return m;
}

auto Machine::with_clause(Clause c) const -> Machine {
    Machine copy = *this;
    copy.db_ = copy.db_.with_clause(std::move(c));
    return copy;
}

auto Machine::consult(Program p) -> Result<void> {
    for (Clause& c : p) {
        auto added = db_.assertz(std::move(c));
        if (!added) {
            return added;
        }
    }
    return {};
}

auto Machine::solve(const std::vector<Term>& goals, std::uint64_t max_solutions)
    -> Result<std::vector<Substitution>> {
    return solve_in(db_, goals, max_solutions);
}

auto Machine::solve_first(const std::vector<Term>& goals) -> Result<std::optional<Substitution>> {
    auto r = solve(goals, 1);
    if (!r) {
        return make_error<std::optional<Substitution>>(r.error());
    }
    if (r->empty()) {
        return std::optional<Substitution>{};
    }
    return std::optional<Substitution>(r->front());
}

auto Machine::database() const -> const Database& { return db_; }

}  // namespace nimblecas
