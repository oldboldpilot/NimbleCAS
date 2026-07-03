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
//  * Only definite Horn clauses are supported. Negation-as-failure, cut, arithmetic
//    evaluation, and the other impure Prolog features are intentionally NOT implemented.
//
// OR-PARALLELISM: the natural parallel/distributed decomposition of SLD resolution is
// OR-parallelism — the independent clause branches for a goal. `solve_or_parallel` maps the
// first goal's candidate clauses across workers with `parallel::transform_index`; each
// branch is a stateless continuation carrying its OWN copy of the substitution and rename
// counter (no shared mutable state), and the branch results are concatenated in clause order
// so the output is identical to the serial `solve` (within the shared budgets). "One branch
// per clause" is exactly the "one shard per clause" shape a distributed engine would use.
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
[[nodiscard]] auto solve(const Program& program, const std::vector<Term>& goals,
                         std::uint64_t max_solutions) -> Result<std::vector<Substitution>>;

// The first solution to `goals`, or std::nullopt if there is none within budget.
[[nodiscard]] auto solve_first(const Program& program, const std::vector<Term>& goals)
    -> Result<std::optional<Substitution>>;

// OR-parallel SLD resolution: the first goal's candidate clauses are explored as independent
// branches via parallel::transform_index (one branch per clause, each a stateless continuation
// with its own substitution and rename counter), and the branch results are concatenated in
// clause order. Byte-for-byte identical to solve() within the shared budgets.
[[nodiscard]] auto solve_or_parallel(const Program& program, const std::vector<Term>& goals,
                                     std::uint64_t max_solutions)
    -> Result<std::vector<Substitution>>;

}  // namespace nimblecas

// ===========================================================================
// Implementation (defined once TermNode is a complete type; not re-exported).
// ===========================================================================
namespace nimblecas {

namespace {

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
[[nodiscard]] auto key_of(const Term& t) -> VarKey {
    const VarNode& v = var_of(t);
    return VarKey{.name = v.name, .generation = v.generation};
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

}  // namespace

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
        out += "]";
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

// Recursive substitution application with a depth cap. cap defends against a pathological
// cyclic binding (which occurs-check already forbids).
[[nodiscard]] auto apply_rec(const Substitution& s, const Term& t, std::uint64_t depth) -> Term {
    constexpr std::uint64_t apply_depth_cap = 100'000;
    if (depth > apply_depth_cap) {
        return t;
    }
    if (is_var(t)) {
        auto bound = find_binding(s, key_of(t));
        if (bound) {
            return apply_rec(s, *bound, depth + 1);
        }
        return t;
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

[[nodiscard]] auto validate_program(const Program& program) -> bool {
    for (const Clause& c : program) {
        if (!is_callable(c.head)) {
            return false;
        }
        for (const Term& g : c.body) {
            if (!is_callable(g)) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] auto validate_goals(const std::vector<Term>& goals) -> bool {
    for (const Term& g : goals) {
        if (!is_callable(g)) {
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

// Depth-first SLD search. `gen` (the rename counter) and `steps` (the global attempt budget)
// are threaded by reference; `depth` bounds derivation length. Raw answer substitutions are
// appended to `out`.
auto sld_search(const Program& program, const std::vector<Term>& goals, const Substitution& sub,
                std::uint64_t& gen, std::uint64_t& steps, std::uint64_t max_solutions,
                std::uint64_t depth, std::vector<Substitution>& out) -> void {
    if (max_solutions != 0 && out.size() >= max_solutions) {
        return;
    }
    if (steps == 0 || depth > max_derivation_depth) {
        return;  // budget exhausted or derivation too deep — return what was found
    }
    if (goals.empty()) {
        out.push_back(sub);  // empty conjunction proved: an answer
        return;
    }

    const Term& first = goals.front();
    for (std::size_t ci = 0; ci < program.size(); ++ci) {
        if (steps == 0) {
            return;
        }
        --steps;
        const std::uint64_t g = ++gen;  // fresh generation per clause activation
        const Clause renamed = rename_clause(program[ci], g);
        auto unified = unify_terms(first, renamed.head, sub);
        if (!unified) {
            continue;
        }
        // New goal list: the clause body (left-to-right) followed by the remaining goals.
        std::vector<Term> next;
        next.reserve(renamed.body.size() + goals.size() - 1);
        for (const Term& b : renamed.body) {
            next.push_back(b);
        }
        for (std::size_t i = 1; i < goals.size(); ++i) {
            next.push_back(goals[i]);
        }
        sld_search(program, next, *unified, gen, steps, max_solutions, depth + 1, out);
        if (max_solutions != 0 && out.size() >= max_solutions) {
            return;
        }
    }
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

auto solve(const Program& program, const std::vector<Term>& goals, std::uint64_t max_solutions)
    -> Result<std::vector<Substitution>> {
    if (!validate_program(program) || !validate_goals(goals)) {
        return make_error<std::vector<Substitution>>(MathError::domain_error);
    }

    std::vector<Substitution> raw;
    std::uint64_t gen = 0;
    std::uint64_t steps = default_step_budget;
    sld_search(program, goals, Substitution{}, gen, steps, max_solutions, 0, raw);

    const std::vector<VarKey> qvars = collect_query_vars(goals);
    std::vector<Substitution> out;
    out.reserve(raw.size());
    for (const Substitution& r : raw) {
        out.push_back(restrict_answer(r, qvars));
    }
    return out;
}

auto solve_first(const Program& program, const std::vector<Term>& goals)
    -> Result<std::optional<Substitution>> {
    auto r = solve(program, goals, 1);
    if (!r) {
        return make_error<std::optional<Substitution>>(r.error());
    }
    if (r->empty()) {
        return std::optional<Substitution>{std::nullopt};
    }
    return std::optional<Substitution>{r->front()};
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

    const Term first = goals.front();
    const std::vector<Term> rest(goals.begin() + 1, goals.end());

    // OR-parallel branch for clause `ci`: a stateless continuation with its OWN rename counter
    // and step budget. It resolves the first goal against clause `ci` and runs the serial SLD
    // search on the resulting continuation. Branches share no mutable state and only READ the
    // (immutable) program and goal terms, so they are safe to run concurrently — the CowPtr
    // term representation guarantees no branch mutates a term another branch observes.
    auto branch = [&](std::size_t ci) -> std::vector<Substitution> {
        std::vector<Substitution> local;
        std::uint64_t gen = 0;
        std::uint64_t steps = default_step_budget;
        const std::uint64_t g = ++gen;
        const Clause renamed = rename_clause(program[ci], g);
        auto unified = unify_terms(first, renamed.head, Substitution{});
        if (unified) {
            std::vector<Term> next = renamed.body;
            next.insert(next.end(), rest.begin(), rest.end());
            // depth = 1 mirrors serial, where the continuation after the first resolution runs
            // one level deep, so the depth budget cuts at exactly the same point.
            sld_search(program, next, *unified, gen, steps, max_solutions, 1, local);
        }
        return local;
    };

    // grain = 1 so every clause becomes an independent task (the backend auto-chunks);
    // transform_index is order-preserving, so per-clause results come back in clause order.
    const std::vector<std::vector<Substitution>> per_clause =
        parallel::transform_index(program.size(), branch, 1);

    // Concatenate in clause order, restrict/canonicalise, and truncate to max_solutions — the
    // same sequence serial solve would produce.
    std::vector<Substitution> out;
    for (const std::vector<Substitution>& branch_out : per_clause) {
        for (const Substitution& raw : branch_out) {
            if (max_solutions != 0 && out.size() >= max_solutions) {
                return out;
            }
            out.push_back(restrict_answer(raw, qvars));
        }
    }
    return out;
}

}  // namespace nimblecas
