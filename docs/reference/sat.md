# `nimblecas.sat` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/sat/sat.cppm`

A deterministic, parallel Boolean-satisfiability substrate — part of the
engine's algorithmic/reasoning layer, not its exact-arithmetic core. Conjunctive
normal form (CNF) is the shared representation: a **literal** is a signed 64-bit
integer (variable `v` in `1..num_vars`, negative meaning `¬v`, `0` illegal), a
**clause** is a disjunction of literals, and a **formula** is their conjunction.
Four engines sit on top of it — two **complete** ([`dpll`](#dpll), [`cdcl`](#cdcl))
that can prove *both* satisfiability and unsatisfiability, and two **stochastic
local search** engines ([`walksat`](#walksat), [`gsat`](#gsat)) that can only ever
exhibit a model or give up — plus a CPU [`solve_portfolio`](#solve_portfolio) and a
stateless [`solve_shard`](#solve_shard) entry point that make the module
distributed-ready.

The honesty boundary here is **decidability and cost, not exactness**: there is no
floating point in the answer and no rational arithmetic to overflow — a solve is a
combinatorial search over Boolean assignments. What must be stated plainly instead
is that **SAT is NP-complete**. The complete solvers can take time *exponential* in
`num_vars`: `dpll` is bounded only by its search tree and always returns a
definitive verdict, while `cdcl` is bounded *additionally* by a `max_conflicts`
budget and returns `unknown` when that budget is spent before a verdict. The
local-search solvers are **incomplete**: they can find a model but can **never**
prove unsatisfiability, so they return only `satisfiable` or `unknown` and **must
never** return `unsatisfiable`. Every solver is hard-capped (recursion bounded by
the variable count; the iterative engines by `max_conflicts` / `max_flips` /
`max_restarts`), so each call always **terminates** — but termination is a
guarantee of a *result*, not of a *verdict*.

**Determinism.** The complete solvers are pure functions of the formula (fixed
tie-breaks, no RNG). The local-search solvers are pure functions of
`(formula, seed)`: they draw from the counter-based `nimblecas.rng` substrate, so
equal seeds reproduce equal runs bit-for-bit. There is no time or entropy seeding
anywhere.

**Parallelism, and why there is no GPU.** SAT search is deeply irregular
(data-dependent branching, pointer chasing, dynamic clause learning) and a **poor
fit for GPUs**; there is deliberately no CUDA here. The realistic parallelism is a
CPU **portfolio**: launch several *independent* solver configurations on the same
formula and race them. Because each configuration is a pure function of
`(formula, worker_index, base_seed)` with no shared mutable state, a worker is also
the natural unit of **distributed** execution — `solve_shard` exposes exactly one
such worker so an external orchestrator can place one shard per process/machine and
merge the per-shard verdicts with a trivial associative all-reduce.

```cpp
import nimblecas.sat;
```

Depends on [`core`](core.md) (the `Result<T>` / `MathError` railway),
`nimblecas.rng` (the counter-based generator the local-search engines draw from),
and [`parallel`](parallel.md) (the `transform_index` fan-out behind the portfolio).

All failure travels the railway (`Result<T>` / `MathError`), never an exception. A
malformed CNF — a literal equal to `0`, a variable index exceeding `num_vars`, or
`num_vars == 0` — is a `domain_error`.

## Types

### `struct Cnf` — a formula in conjunctive normal form

```cpp
struct Cnf {
    std::size_t num_vars;
    std::vector<std::vector<std::int64_t>> clauses;
};
```

A CNF formula over variables `1..num_vars`. Each inner vector is a **clause** (a
disjunction of literals); the formula is the conjunction of its clauses. A literal
is a signed variable index: `+v` asserts variable `v`, `−v` asserts `¬v`, and `0`
is never a legal literal. An **empty clause** is permitted and denotes *falsity*
(an unsatisfiable formula); an **empty clause list** denotes *truth*. `num_vars == 0`
is malformed.

### `enum class SatVerdict : std::uint8_t`

```cpp
enum class SatVerdict : std::uint8_t { satisfiable, unsatisfiable, unknown };
```

The three possible outcomes of a solve. `unknown` is only ever produced by a
**resource cap** (CDCL's conflict budget, or a local-search flip/restart budget),
never as a substitute for a verdict a complete solver could have reached given
time.

### `struct SatResult`

```cpp
struct SatResult {
    SatVerdict verdict;
    std::vector<bool> model;
};
```

`model` is 0-indexed and, when present, has size `num_vars` with `model[v-1]` giving
the Boolean value assigned to variable `v`. It is non-empty only when
`verdict == satisfiable` (and is then guaranteed to satisfy every clause — see
[`verify_assignment`](#verify_assignment)); it is **empty** for `unsatisfiable` and
`unknown`.

## Verification

<a id="verify_assignment"></a>

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `verify_assignment` | `[[nodiscard]] auto verify_assignment(const Cnf& cnf, const std::vector<bool>& model) -> bool` | `true` iff `model` satisfies **every** clause of `cnf` — each clause contains at least one literal made true by the assignment (`model[v-1]` is variable `v`; `+v` is satisfied when the variable is true, `−v` when it is false). Returns `false` for a model whose size differs from `num_vars`, for a malformed CNF (literal `0` or an out-of-range variable), and for any clause (including the empty clause) with no satisfied literal. Total predicate: never errors. |

Every solver in this module calls `verify_assignment` on its own candidate before
returning `satisfiable`; a failed self-check is treated as an **internal bug** and
reported as `MathError::undefined_value` rather than a false-positive `satisfiable`
(see [Error model](#error-model)).

## Complete solvers

Both reach a definitive verdict (`dpll` always; `cdcl` within its budget) and can
prove `unsatisfiable`. Both are worst-case exponential in `num_vars`.

<a id="dpll"></a>
<a id="cdcl"></a>

| Solver | Signature | Behavior |
| :--- | :--- | :--- |
| `dpll` | `[[nodiscard]] auto dpll(const Cnf& cnf) -> Result<SatResult>` | Classic **DPLL** — unit propagation + pure-literal elimination + chronological backtracking over the lowest-index unassigned variable (true before false). Always reaches a definitive verdict (**never** `unknown`): `satisfiable` with a verified model, or `unsatisfiable`. **Worst case exponential** in `num_vars`. |
| `cdcl` | `[[nodiscard]] auto cdcl(const Cnf& cnf, std::uint64_t max_conflicts) -> Result<SatResult>` | **CDCL** (conflict-driven clause learning) with 1-UIP learned clauses, non-chronological backjumping, and Luby restarts. Deterministic — no RNG, fixed decision order (lowest-index variable, false first) and tie-breaks. Performs at most `max_conflicts` conflict analyses: `satisfiable` (verified model) or `unsatisfiable` if a verdict is reached within budget, else **`unknown`** if the budget is spent first. A conflict at decision level 0 is **always** UNSAT regardless of budget. |

## Stochastic local search (incomplete)

Both start from a random full assignment and flip variables. They return
`satisfiable` with a verified model if one is found within budget, else `unknown`.
Both are **deterministic given `seed`** and **must never** return `unsatisfiable` —
they cannot prove UNSAT; on an unsatisfiable formula they exhaust their budget and
report `unknown`.

<a id="walksat"></a>
<a id="gsat"></a>

| Solver | Signature | Behavior |
| :--- | :--- | :--- |
| `walksat` | `[[nodiscard]] auto walksat(const Cnf& cnf, std::uint64_t max_flips, double noise, std::uint64_t seed) -> Result<SatResult>` | **WalkSAT.** Repeatedly pick a random currently-unsatisfied clause; with probability `noise` flip a uniformly random variable in it, otherwise flip the variable in it whose flip breaks the fewest currently-satisfied clauses (ties by clause order). Runs at most `max_flips` flips. `noise` must lie in `[0, 1]`. |
| `gsat` | `[[nodiscard]] auto gsat(const Cnf& cnf, std::uint64_t max_flips, std::uint64_t max_restarts, std::uint64_t seed) -> Result<SatResult>` | **GSAT.** Runs `1 + max_restarts` independent tries (an initial random assignment plus `max_restarts` restarts). Within each try, up to `max_flips` greedy flips: flip the variable whose flip most reduces the number of unsatisfied clauses (net gain = clauses made satisfied − clauses broken; ties by lowest variable index), accepting sideways and uphill moves as classic GSAT does. |

## Parallel and distributed

Each worker is a pure function of `(cnf, worker_index, base_seed)` with **no shared
mutable state**. Worker 0 is always complete `dpll`; other indices cycle through
`cdcl` (with an index-varied conflict budget), `walksat`, and `gsat`, each
stochastic worker seeded with `splitmix64(base_seed ^ worker_index)`.

<a id="solve_portfolio"></a>
<a id="solve_shard"></a>

| Entry point | Signature | Behavior |
| :--- | :--- | :--- |
| `solve_portfolio` | `[[nodiscard]] auto solve_portfolio(const Cnf& cnf, std::uint64_t base_seed, std::size_t workers) -> Result<SatResult>` | Runs `workers` **independent** solver configurations concurrently over the same formula (via `parallel::transform_index`) and merges their verdicts. Because worker 0 is always complete DPLL, the portfolio itself **always reaches a definitive verdict**. `workers` must be `> 0`. |
| `solve_shard` | `[[nodiscard]] auto solve_shard(const Cnf& cnf, std::size_t shard_index, std::size_t num_shards, std::uint64_t base_seed) -> Result<SatResult>` | Returns the verdict of **exactly one** portfolio worker — the configuration keyed by `(shard_index, base_seed)` — computed with no shared state. The unit of distributed execution: an orchestrator runs one process/task per shard over `0..num_shards-1` and merges the results with the *same* reduction (below). `num_shards` bounds the shard space only; a shard's computation depends solely on its own index and `base_seed`, never on `num_shards`. Requires `num_shards > 0` and `shard_index < num_shards`. |

**Deterministic merge (an associative all-reduce).** For both the portfolio and a
set of shards, results combine by the same commutative rule: any `unsatisfiable`
from a complete worker is definitive and **wins**; otherwise the `satisfiable` model
from the **lowest** worker/shard index is returned; otherwise `unknown`. (A genuine
per-worker error is propagated from the lowest offending index.) Because each worker
is stateless and keyed only by `(index, seed)`, the merge does **not** depend on
scheduling, worker count, or the order shards are collected in.

## Error model

| Condition | Error |
| :--- | :--- |
| Malformed CNF: `num_vars == 0` | `MathError::domain_error` |
| Malformed CNF: a literal equal to `0` | `MathError::domain_error` |
| Malformed CNF: a literal whose variable magnitude exceeds `num_vars` | `MathError::domain_error` |
| `walksat` called with `noise` outside `[0, 1]` | `MathError::domain_error` |
| `solve_portfolio` called with `workers == 0` | `MathError::domain_error` |
| `solve_shard` called with `num_shards == 0` or `shard_index >= num_shards` | `MathError::domain_error` |
| A solver's own candidate model fails its internal self-check (a bug) | `MathError::undefined_value` |

An **unsatisfiable** formula is **not** an error: it is a valid `SatResult` with
`verdict == unsatisfiable` and an empty `model`. Likewise a budget-exhausted
`unknown` is a valid result, not an error. `verify_assignment` never errors (it
returns `bool`). `undefined_value` is a defensive last resort — no correct solver
should ever produce it.

## Worked examples

```cpp
import nimblecas.sat;
import nimblecas.core;
using namespace nimblecas;

// A small satisfiable 3-SAT instance and the minimal unsatisfiable formula.
const Cnf small_sat{3, {{1, 2}, {-1, 3}, {-2, -3}}};
const Cnf trivial_unsat{1, {{1}, {-1}}};          // (x) ∧ (¬x)
const Cnf php_2_1{2, {{1}, {2}, {-1, -2}}};       // pigeonhole 2→1: UNSAT

// Complete solvers reach a definitive verdict and return a verified model.
auto d = dpll(small_sat).value();
d.verdict == SatVerdict::satisfiable;             // true
verify_assignment(small_sat, d.model);            // true — model has num_vars entries

// Both complete engines prove UNSAT and agree.
dpll(trivial_unsat).value().verdict;              // SatVerdict::unsatisfiable
cdcl(trivial_unsat, 100000).value().verdict;      // SatVerdict::unsatisfiable
cdcl(php_2_1, 100000).value().verdict;            // SatVerdict::unsatisfiable

// A satisfiable formula with a clear basin for local search.
const Cnf easy_sat{5, {{1, -2, 3}, {2, -3, 4}, {3, -4, 5}, {1, 4, 5}, {-1, 2, 5}}};

// WalkSAT / GSAT find and verify a model (deterministic given the seed).
auto w = walksat(easy_sat, 200000, 0.4, 12345).value();
w.verdict == SatVerdict::satisfiable;             // true
verify_assignment(easy_sat, w.model);             // true

auto g = gsat(easy_sat, 2000, 200, 7).value();
g.verdict == SatVerdict::satisfiable;             // true

// Local search CANNOT prove UNSAT: it gives up with `unknown`, never `unsatisfiable`.
walksat(php_2_1, 5000, 0.5, 999).value().verdict; // SatVerdict::unknown

// Portfolio: worker 0 is complete DPLL, so the verdict always matches serial dpll.
solve_portfolio(php_2_1, 2024, 4).value().verdict;    // SatVerdict::unsatisfiable
solve_portfolio(easy_sat, 2024, 4).value().verdict;   // SatVerdict::satisfiable

// Distributed shards: solve each independently, then merge with the associative rule
// (any UNSAT wins; else the model from the lowest index; else unknown).
std::vector<SatResult> shards;
for (std::size_t s = 0; s < 4; ++s) {
    shards.push_back(solve_shard(php_2_1, s, 4, 2024).value());
}
// merged verdict == dpll(php_2_1) == SatVerdict::unsatisfiable

// Malformed input and out-of-range parameters travel the railway.
dpll(Cnf{0, {}}).error();                         // MathError::domain_error (num_vars == 0)
cdcl(Cnf{2, {{1, 3}}}, 1000).error();             // domain_error (variable 3 > num_vars)
dpll(Cnf{2, {{1, 0}}}).error();                   // domain_error (literal 0)
walksat(small_sat, 100, 1.5, 1).error();          // domain_error (noise > 1)
solve_portfolio(small_sat, 1, 0).error();         // domain_error (workers == 0)
solve_shard(small_sat, 3, 3, 1).error();          // domain_error (shard_index >= num_shards)
```

## See also

- [`nimblecas.csp`](csp.md) — the sibling finite-domain constraint-satisfaction
  solver in the same algorithmic/reasoning layer.
- [`nimblecas.lp`](lp.md) — the sibling linear-programming optimisation solver.
- [`nimblecas.parallel`](parallel.md) — the `transform_index` fan-out behind
  `solve_portfolio`.
- [`nimblecas.core`](core.md) — the `Result<T>` / `MathError` railway every entry
  point returns through.
- [Documentation hub](../Index.md)
```