# `nimblecas.logic` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/logic/logic.cppm`

A small **Prolog core**: first-order terms, Robinson **unification with
occurs-check**, and **SLD resolution** (depth-first, backtracking) over definite
Horn clauses, plus an **OR-parallel** solver. It sits in the engine's
algorithmic/reasoning layer alongside [`csp`](csp.md) and [`lp`](lp.md) — not its
exact-arithmetic core — so the honesty boundary here is **decidability, not
exactness**: there is no floating point and no rational arithmetic to overflow.
The `Integer` term kind carries a plain `std::int64_t` constant that the solver
only ever compares for equality, never computes on, so no `overflow` can arise.
What must be stated plainly instead is that **SLD resolution is only
semi-decidable**: a left-recursive rule or an infinite derivation can diverge,
so two budgets bound the search and guarantee it always terminates —
`default_step_budget` caps the total resolution attempts and
`max_derivation_depth` caps the length of any single derivation (which also keeps
the native C++ recursion bounded, so a runaway program can never overflow the
stack). These budgets are **not** a completeness claim: the solver returns the
solutions found *within budget*. A query with **no** solutions is an empty answer
list, never an error; only a **malformed** program or query (a non-callable head
or goal — a bare variable or integer where a predicate is required) is a
`MathError::domain_error`. Only definite Horn clauses are supported:
negation-as-failure, cut, and arithmetic evaluation are intentionally **not**
implemented.

Everything is **deterministic**: clauses are tried in program order, subgoals
left-to-right, and variables are standardised-apart by a rename counter threaded
explicitly through the search (never a global or random source), so a given
program + query always enumerates the same answers in the same order.

```cpp
import nimblecas.logic;
```

Depends on [`core`](core.md) (`Result`, `MathError`, `CowPtr`) and
[`parallel`](parallel.md) (`transform_index`, used by the OR-parallel solver).

## The term representation

A `Term` is an immutable first-order term — one of a logic **Variable**, an
**Atom** (symbolic constant), an **Integer** constant, or a **Compound** (functor
applied to one or more argument terms). It is a copy-on-write handle
(`CowPtr<TermNode>`, [`core`](core.md) Rule 22): copying a `Term` is an O(1)
refcount bump and read-only sharing across threads is safe — which is exactly
what makes the OR-parallel decomposition sound (branches only *read* shared terms
and produce new ones; no handle is ever mutated).

| Type | Signature | Role |
| :--- | :--- | :--- |
| `Term` | `class Term { explicit Term(TermNode); auto node() const -> const TermNode&; }` | The COW handle; prefer the `make_*` factories over the constructor. |
| `TermNode` | `struct TermNode { std::variant<VarNode, AtomNode, IntNode, CompoundNode> value; }` | The tagged node the handle wraps. |
| `VarNode` | `struct { std::string name; std::uint64_t generation; }` | A logic variable identified by **name plus rename generation** (see below). |
| `AtomNode` | `struct { std::string name; }` | A symbolic constant, e.g. `tom` or the list terminator `[]`. |
| `IntNode` | `struct { std::int64_t value; }` | An integer constant (compared, never arithmetic'd). |
| `CompoundNode` | `struct { std::string functor; std::vector<Term> args; }` | A functor over argument terms, e.g. `parent(tom, bob)`. |

A logic variable's identity is its **name plus a rename `generation`**.
Standardising a clause apart stamps a fresh generation on every variable, so the
same source name reused across clause activations yields distinct variables while
staying consistent within one activation. `generation` `0` is the "source"
generation the factories default to.

### Term factories

Constructing terms never fails — each returns a plain `Term`.

| Factory | Signature | Behavior |
| :--- | :--- | :--- |
| `make_var` | `auto make_var(std::string name, std::uint64_t generation = 0) -> Term` | A logic variable. `generation` defaults to `0`; standardise-apart overwrites it. |
| `make_atom` | `auto make_atom(std::string name) -> Term` | A symbolic constant. |
| `make_int` | `auto make_int(std::int64_t value) -> Term` | An integer constant. |
| `make_compound` | `auto make_compound(std::string functor, std::vector<Term> args) -> Term` | A compound term. With **empty** `args` it collapses to `make_atom(functor)`, since a 0-arg compound is just an atom. |
| `make_nil` | `auto make_nil() -> Term` | The empty list, i.e. the atom `[]`. |
| `make_list` | `auto make_list(std::vector<Term> elements) -> Term` | A proper list `[e0, e1, …]` encoded as nested `'.'/2` cons cells terminated by `make_nil()`. |

### Term equality and rendering

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `operator==` | `auto operator==(const Term& a, const Term& b) -> bool` | **Structural (syntactic)** equality: identical trees, with variables equal iff both `name` and `generation` match. |
| `operator!=` | `auto operator!=(const Term& a, const Term& b) -> bool` | Negation of `operator==`. |
| `to_string` | `auto to_string(const Term& t) -> std::string` | Prolog-style rendering: atoms/integers verbatim, compounds as `f(a, b)`, and `'.'/2` spines as list syntax `[a, b]` (or `[a | Tail]` when improper). A variable prints as its name, suffixed `_<generation>` once standardised apart. |

## Substitutions, clauses, and programs

| Type | Definition | Role |
| :--- | :--- | :--- |
| `VarKey` | `struct { std::string name; std::uint64_t generation; }` | The identity of a logic variable (name + generation). `operator==(const VarKey&, const VarKey&)` compares both fields. |
| `Substitution` | `using Substitution = std::vector<std::pair<VarKey, Term>>` | An **ordered** variable → term binding set. A vector of pairs (not a hash map) keeps iteration order deterministic, which the solver relies on for reproducible answers. |
| `Clause` | `struct { Term head; std::vector<Term> body; }` | A definite Horn clause `head :- body[0], body[1], …`. A **fact** has an empty `body`. |
| `Program` | `using Program = std::vector<Clause>` | An ordered list of clauses; clause order fixes enumeration order. |

## Budgets

Two `inline constexpr` bounds guarantee termination (see the semi-decidability
note above). They are **fixed constants**, not per-call parameters.

| Constant | Type | Value | Purpose |
| :--- | :--- | :--- | :--- |
| `default_step_budget` | `std::uint64_t` | `1'000'000` | Maximum clause-resolution attempts across a whole search. Exhausting it stops the search and returns whatever was found so far (**not** a completeness guarantee). |
| `max_derivation_depth` | `std::uint64_t` | `1'000` | Maximum length of any single SLD derivation. Also bounds native recursion depth, so a non-terminating program cannot overflow the C++ stack. |

## Core operations

### Unification and substitution

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `unify` | `auto unify(const Term& a, const Term& b, const Substitution& in) -> Result<std::optional<Substitution>>` | Robinson unification **with occurs-check**, applying the incoming substitution `in` as it goes. Returns the extended most-general-unifier substitution, or `std::nullopt` when the terms do not unify. An **occurs-check failure** (e.g. unifying `X` with `f(X)`) is a normal non-unification — it yields `nullopt`, **not** a `MathError`. The `Result` wrapper is reserved for genuine errors and is presently **always the value branch**. |
| `apply_substitution` | `auto apply_substitution(const Substitution& s, const Term& t) -> Term` | Resolves every variable in `t` against `s` to a fixed point, rebuilding compounds with their arguments resolved. Occurs-check keeps bindings acyclic so this terminates; a depth cap is a defensive backstop. |

### The solvers

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `solve` | `auto solve(const Program& program, const std::vector<Term>& goals, std::uint64_t max_solutions) -> Result<std::vector<Substitution>>` | SLD resolution over `program` for the conjunction `goals`, depth-first with backtracking, standardising each clause apart per activation via a deterministic rename counter. Answers are enumerated in a fixed order (clauses in program order, subgoals left-to-right) up to `max_solutions` (`0` = all reachable **within the budgets**). Each returned substitution binds the query's variables, restricted to them; any residual internal variables are canonicalised (as `_G0`, `_G1`, …) so the output does not depend on the internal rename counter. A query with **no** solutions returns an **empty** vector. A malformed program/query returns `domain_error`. |
| `solve_first` | `auto solve_first(const Program& program, const std::vector<Term>& goals) -> Result<std::optional<Substitution>>` | The first solution to `goals`, or `std::nullopt` if there is none within budget. Implemented as `solve(program, goals, 1)`; a malformed program/query still returns `domain_error`. |
| `solve_or_parallel` | `auto solve_or_parallel(const Program& program, const std::vector<Term>& goals, std::uint64_t max_solutions) -> Result<std::vector<Substitution>>` | **OR-parallel** SLD resolution: the first goal's candidate clauses are explored as independent branches via `parallel::transform_index` (one branch per clause, each a stateless continuation with its **own** substitution and rename counter), and the branch results are concatenated in clause order. **Byte-for-byte identical to `solve`** whenever the search completes within budget — the normal terminating case. |

### The OR-parallel budget caveat

`solve_or_parallel` gives each branch its **own** step/depth budget rather than a
single shared quota, since "one branch per clause" is exactly the "one shard per
clause" shape a distributed engine would use. The honest consequence: a query
that actually **exhausts** its step budget mid-search may enumerate *more*
solutions in parallel than serial `solve`'s one global budget would have cut off.
Within budget — the ordinary case — the two agree exactly.

SLD's *AND*-side backtracking, by contrast, is irregular and data-dependent — a
poor fit for SIMT/GPU execution — so this solver targets CPU/distributed
parallelism only and deliberately ships **no CUDA path**.

## Error model

| Condition | Result |
| :--- | :--- |
| A goal or clause **head** is not callable (a bare variable or integer where a predicate — atom or compound — is required) | `MathError::domain_error` (from `solve`, `solve_first`, `solve_or_parallel`) |
| A query with **no solutions** | Value branch: an **empty** `std::vector<Substitution>` (`solve` / `solve_or_parallel`) or `std::nullopt` (`solve_first`) — **not** an error |
| Occurs-check failure in `unify` (e.g. `X = f(X)`) | Value branch holding `std::nullopt` — **not** an error |
| Budget exhausted mid-search (`default_step_budget` or `max_derivation_depth`) | Value branch: the solutions found so far (**no** error; not a completeness guarantee) |

There is no `overflow` path: integer terms are compared, never computed on.
`unify`'s `Result` is presently always the value branch; the wrapper is reserved
for future genuinely-fallible variants.

## Worked examples

```cpp
import nimblecas.logic;
using namespace nimblecas;

// --- A family database: parent/2 facts, grandparent/2 and recursive ancestor/2 rules. ---
Program p;
p.push_back(Clause{make_compound("parent", {make_atom("tom"), make_atom("bob")}), {}});
p.push_back(Clause{make_compound("parent", {make_atom("bob"), make_atom("ann")}), {}});
p.push_back(Clause{make_compound("parent", {make_atom("bob"), make_atom("pat")}), {}});

const Term x = make_var("X"), y = make_var("Y"), z = make_var("Z");
// grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
p.push_back(Clause{make_compound("grandparent", {x, z}),
                   {make_compound("parent", {x, y}), make_compound("parent", {y, z})}});
// ancestor(X, Y) :- parent(X, Y).
p.push_back(Clause{make_compound("ancestor", {x, y}), {make_compound("parent", {x, y})}});
// ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
p.push_back(Clause{make_compound("ancestor", {x, y}),
                   {make_compound("parent", {x, z}), make_compound("ancestor", {z, y})}});

const Term w = make_var("W");

// grandparent(tom, W): two solutions, enumerated in program order — ann then pat.
auto gp = solve(p, {make_compound("grandparent", {make_atom("tom"), w})}, 0).value();
apply_substitution(gp[0], w) == make_atom("ann");   // first  W = ann
apply_substitution(gp[1], w) == make_atom("pat");   // second W = pat

// ancestor(tom, W): three solutions within budget — bob, ann, pat.
auto anc = solve(p, {make_compound("ancestor", {make_atom("tom"), w})}, 0).value();
anc.size() == 3;

// max_solutions caps enumeration; the single answer is the first found (bob).
auto one = solve(p, {make_compound("ancestor", {make_atom("tom"), w})}, 1).value();
apply_substitution(one[0], w) == make_atom("bob");

// A query with no solutions is an EMPTY vector, never an error (bob has no grandchildren).
auto none = solve(p, {make_compound("grandparent", {make_atom("bob"), w})}, 0).value();
none.empty();

// solve_first: the first answer, or nullopt when there is none.
solve_first(p, {make_compound("grandparent", {make_atom("tom"), w})}).value().has_value();  // ann
solve_first(p, {make_compound("grandparent", {make_atom("bob"), w})}).value().has_value() == false;

// OR-parallel is byte-for-byte identical to serial within budget.
auto serial   = solve(p, {make_compound("ancestor", {make_atom("tom"), w})}, 0).value();
auto parallel = solve_or_parallel(p, {make_compound("ancestor", {make_atom("tom"), w})}, 0).value();
serial == parallel;

// --- Unification with occurs-check. ---
// X = f(X) is rejected: a VALUE branch whose optional is empty — NOT a MathError.
auto occ = unify(x, make_compound("f", {x}), {});
occ.has_value() && !occ->has_value();                     // value branch, nullopt inside

// f(X, b) unifies with f(a, Y) => X = a, Y = b.
auto u = unify(make_compound("f", {x, make_atom("b")}),
               make_compound("f", {make_atom("a"), y}), {}).value().value();
apply_substitution(u, x) == make_atom("a");
apply_substitution(u, y) == make_atom("b");

// An atom and an integer never unify (distinct constant kinds): value branch, nullopt.
unify(make_atom("1"), make_int(1), {}).value().has_value() == false;

// --- Lists: append([], L, L). append([H|T], L, [H|R]) :- append(T, L, R). ---
Program ap;
const Term h = make_var("H"), t = make_var("T"), l = make_var("L"), r = make_var("R");
ap.push_back(Clause{make_compound("append", {make_nil(), l, l}), {}});
ap.push_back(Clause{make_compound("append", {make_compound(".", {h, t}), l,
                                             make_compound(".", {h, r})}),
                    {make_compound("append", {t, l, r})}});
// append([1], [2], R) => R = [1, 2].
auto sols = solve(ap, {make_compound("append",
                       {make_list({make_int(1)}), make_list({make_int(2)}), r})}, 0).value();
apply_substitution(sols[0], r) == make_list({make_int(1), make_int(2)});
to_string(make_list({make_int(1), make_int(2)})) == "[1, 2]";   // list pretty-printing

// --- Malformed inputs are domain_errors. ---
solve(p, {make_int(5)}, 0).error()   == MathError::domain_error;  // integer goal
solve(p, {make_var("G")}, 0).error() == MathError::domain_error;  // variable goal
Program bad;
bad.push_back(Clause{make_var("X"), {}});                          // variable clause head
solve(bad, {make_atom("q")}, 0).error() == MathError::domain_error;
```

## See also

- [`nimblecas.csp`](csp.md) and [`nimblecas.lp`](lp.md) — the sibling
  reasoning-layer solvers (finite-domain constraint satisfaction and linear
  programming), likewise bounded by decidability/cost rather than exactness.
- [`nimblecas.parallel`](parallel.md) — the deterministic fork–join layer whose
  `transform_index` drives the OR-parallel solver.
- [`nimblecas.core`](core.md) — the `Result` / `MathError` error model and the
  `CowPtr` the immutable `Term` is built on.
- [Documentation hub](../Index.md)
