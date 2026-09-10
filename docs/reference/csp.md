# `nimblecas.csp` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/csp/csp.cppm`

A deterministic **finite-domain constraint satisfaction** toolkit — part of the
engine's algorithmic/reasoning layer rather than its exact-arithmetic core.
Variables are indices `0..n-1`, each carrying a finite integer domain
(`std::int64_t` values, supplied ascending and deduplicated **by the caller**).
Constraints come in two flavours: **binary** relations `R(x, y)` and general
**n-ary** relations over a variable scope. `ac3` enforces **arc consistency over
the binary constraints only**; `backtracking_search`, its forward-checking
variant, `solution_count`, and `parallel_search` honour **both** binary and
general constraints.

The honesty boundary here is **decidability and cost, not exactness**: there is
no floating point and no rational arithmetic to overflow — every value is a plain
`std::int64_t` the caller chose, and the solvers never do arithmetic on them,
only feed them to caller-supplied predicates. What must be stated plainly instead
is that **CSP is NP-complete**: backtracking (even with forward checking) is
**worst-case exponential**, so `solution_count` takes a hard `limit` that caps
enumeration and can never run away. `ac3` is a **propagation/preprocessing step,
not a complete solver** — an arc-consistent CSP can still be unsatisfiable (three
mutually distinct variables over a two-value domain stay arc consistent yet have
no solution). An **unsatisfiable problem is not an error**: it is a valid empty
result (`std::nullopt` for the searches, `0` for the count, a disengaged optional
for an emptied `ac3` domain). Only **shape faults** — an empty variable set, an
empty initial domain, or a constraint scope that references an out-of-range
variable — are errors (`MathError::domain_error`).

**Determinism:** every solver returns the **same** answer regardless of thread
count. Search uses a **static** variable order (`0, 1, …, n-1`) and tries each
variable's values in the caller-supplied ascending order, so "the solution" is
always the **lexicographically-first** complete assignment. `ac3`'s worklist is a
FIFO of arcs seeded and re-queued in arc-index order, so its pruned domains do
not depend on scheduling either. CSP search is irregular and branchy, so the
realistic parallel win is **CPU branch-parallelism / distribution** across
independent subtrees — **not the GPU**; there is deliberately no CUDA here.

```cpp
import nimblecas.csp;
```

Depends on [`core`](core.md) and [`parallel`](parallel.md).

## `BinaryConstraint` — a binary relation `R(x, y)`

A constraint on variables `x` and `y`: `allowed(a, b)` is `true` iff assigning
`x = a` and `y = b` is permitted. The relation **need not be symmetric**; the
argument order matches `(x, y)`.

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `x` | `std::size_t` | First variable index. |
| `y` | `std::size_t` | Second variable index. |
| `allowed` | `std::function<bool(std::int64_t, std::int64_t)>` | Predicate; `allowed(a, b)` is `true` iff `x = a, y = b` is permitted. |

## `Constraint` — a general n-ary relation over a scope

A constraint over `scope`: `allowed(values)` is `true` iff the assignment that
binds `scope[k] = values[k]` (values passed **in scope order**) is permitted.

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `scope` | `std::vector<std::size_t>` | The variable indices the relation ranges over. Must be non-empty. |
| `allowed` | `std::function<bool(std::span<const std::int64_t>)>` | Predicate over the tuple of values, indexed in `scope` order. |

## `Csp` — a finite-domain constraint satisfaction problem

Per-variable domains plus binary and general constraints. Variables are the
indices `0..domains.size()-1`; variable `i` owns `domains[i]`.

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `domains` | `std::vector<std::vector<std::int64_t>>` | Per-variable finite domains; each must be non-empty, and callers supply values ascending and deduplicated. |
| `binary` | `std::vector<BinaryConstraint>` | The binary constraints. |
| `general` | `std::vector<Constraint>` | The general n-ary constraints. |

The struct has no invariant-enforcing constructor: shape is validated by each
entry point, and value ordering / deduplication within a domain is the caller's
responsibility (the lexicographic-first contract is stated **relative to** the
order the caller provides).

## Arc consistency

```cpp
[[nodiscard]] auto ac3(const Csp& csp)
    -> Result<std::optional<std::vector<std::vector<std::int64_t>>>>;
```

| Function | Behavior |
| :--- | :--- |
| `ac3` | AC-3 arc consistency over the **binary constraints only** (general constraints are ignored). Returns the pruned per-variable domains; returns an **engaged `Result` holding `std::nullopt`** when a domain is emptied (arc-inconsistent, hence certainly unsatisfiable — not an error). Each binary constraint contributes two directed arcs; the FIFO worklist is seeded and re-queued in **arc-index order**, so the fixed point is independent of scheduling. Shape faults are `domain_error`. |

**Re-queue subtlety (correctness, not an optimisation to skip lightly):** when a
value is dropped from `D[from]`, `ac3` re-queues every arc reading *into* `from`
**except the reverse arc of the same constraint**. The classic "skip `Xj`"
optimisation is sound only **per constraint** — a value dropped for lack of
support under constraint `c` cannot have been the support for any value under
`c`'s reverse arc, but it *can* have supported a value under a **different**
constraint on the same ordered pair. Excluding every arc coming from `to`
(rather than just the same-constraint reverse) would let `ac3` return
non-arc-consistent domains when two binary constraints share a variable pair.

`ac3` is a propagation step. It **cannot** by itself decide satisfiability: an
arc-consistent CSP can still have no solution.

## Search and enumeration

```cpp
[[nodiscard]] auto backtracking_search(const Csp& csp)
    -> Result<std::optional<std::vector<std::int64_t>>>;
[[nodiscard]] auto backtracking_search_fc(const Csp& csp)
    -> Result<std::optional<std::vector<std::int64_t>>>;
[[nodiscard]] auto solution_count(const Csp& csp, std::uint64_t limit)
    -> Result<std::uint64_t>;
[[nodiscard]] auto parallel_search(const Csp& csp)
    -> Result<std::optional<std::vector<std::int64_t>>>;
```

| Function | Behavior |
| :--- | :--- |
| `backtracking_search` | Plain backtracking honouring **both** binary and general constraints. Returns the **lexicographically-first** complete assignment (static variable order, ascending values), or `std::nullopt` if the CSP is unsatisfiable. The returned assignment is **re-verified** against every constraint before being handed back; a violation would be an internal bug and surfaces as `undefined_value`. Worst-case exponential (NP-complete). |
| `backtracking_search_fc` | Backtracking augmented with **forward checking**: after each assignment the domains of the as-yet-unassigned variables are pruned of now-inconsistent values, and a branch is abandoned the instant a future domain empties. It **keeps the static variable order** (it deliberately does *not* use MRV dynamic ordering, which would change *which* solution is found first) and uses forward checking purely for pruning — so the first complete assignment found is **bit-for-bit identical** to `backtracking_search`'s, just reached with less thrashing. Same `undefined_value` self-check on the result. |
| `solution_count` | Counts complete, constraint-satisfying assignments up to `limit`. `limit == 0` means **no cap** (enumerate the whole tree); otherwise the search stops and returns `limit` once that many solutions have been seen (a runaway guard). Returns `0` for an unsatisfiable CSP. Shape faults are `domain_error`. |
| `parallel_search` | Same answer as `backtracking_search` for **any worker count**. The **first variable's** domain is split across workers via `nimblecas::parallel::transform_index`: worker `j` fixes variable `0` to the `j`-th (ascending) domain value, copies the CSP (sharing nothing mutable — the constraint functors are read-only), and solves that subtree independently. Results are combined **in index order**, so the smallest `j` yielding any solution gives the lexicographically-first assignment overall. Same `undefined_value` self-check. |

`parallel_search`'s stateless per-branch decomposition is exactly what makes the
problem distributable: each first-variable value is one self-contained task, so
the same split maps onto an external orchestrator (one shard per value) with no
shared state to reconcile.

## Error model

Every entry point runs the same structural validation first. An
**unsatisfiable-but-well-formed** CSP is **not** a fault — that is a valid empty
result, decided by the solver, not an error.

| Condition | Error |
| :--- | :--- |
| Empty variable set (`domains.empty()`) | `MathError::domain_error` |
| Any variable's initial domain is empty | `MathError::domain_error` |
| A `BinaryConstraint` references `x` or `y` `>= n` | `MathError::domain_error` |
| A `Constraint` has an empty `scope` | `MathError::domain_error` |
| A `Constraint` scope references an index `>= n` | `MathError::domain_error` |
| A returned complete assignment fails re-verification (internal bug) | `MathError::undefined_value` |

There is **no `overflow`**: values are caller-supplied `std::int64_t`s the
solvers never do arithmetic on, only pass to caller predicates. `ac3` emitting a
disengaged optional (a domain emptied), a search returning `std::nullopt`, and
`solution_count` returning `0` are all **successful** results describing an
unsatisfiable CSP, never errors.

## Worked examples

```cpp
import nimblecas.csp;
using namespace nimblecas;

auto range_domain = [](std::int64_t k) {
    std::vector<std::int64_t> d;
    for (std::int64_t v = 0; v < k; ++v) d.push_back(v);
    return d;
};
auto ne = [](std::size_t i, std::size_t j) {
    return BinaryConstraint{i, j,
        [](std::int64_t a, std::int64_t b) { return a != b; }};
};

// N-queens: variable i is the row of the queen in column i.
auto queens = [&](std::int64_t n) {
    Csp csp;
    const auto sz = static_cast<std::size_t>(n);
    csp.domains.assign(sz, range_domain(n));
    for (std::size_t i = 0; i < sz; ++i)
        for (std::size_t j = i + 1; j < sz; ++j) {
            const std::int64_t d =
                static_cast<std::int64_t>(j) - static_cast<std::int64_t>(i);
            csp.binary.push_back(BinaryConstraint{i, j,
                [d](std::int64_t a, std::int64_t b) {
                    return a != b && (a - b != d) && (b - a != d);
                }});
        }
    return csp;
};

// Exact solution counts (limit 0 = enumerate the whole tree).
solution_count(queens(4), 0).value();   // 2
solution_count(queens(8), 0).value();   // 92
solution_count(queens(8), 10).value();  // 10  (capped early by the limit)

// The lexicographically-first 4-queens placement (rows by column).
backtracking_search(queens(4)).value().value();  // {1, 3, 0, 2}

// Forward checking returns the IDENTICAL first solution, just faster.
auto plain = backtracking_search(queens(6)).value();
auto fc    = backtracking_search_fc(queens(6)).value();
// plain == fc

// parallel_search agrees with backtracking_search for any worker count.
parallel_search(queens(6)).value();  // == backtracking_search(queens(6))

// AC-3 prunes: X, Y in {1,2,3} with X < Y removes 3 from X and 1 from Y.
Csp lt;
lt.domains = {{1, 2, 3}, {1, 2, 3}};
lt.binary.push_back(BinaryConstraint{0, 1,
    [](std::int64_t a, std::int64_t b) { return a < b; }});
auto pruned = ac3(lt).value().value();
// pruned[0] == {1, 2},  pruned[1] == {2, 3}

// AC-3 reports arc-inconsistency as an engaged Result holding nullopt.
Csp bad;
bad.domains = {{0}, {0}};
bad.binary.push_back(ne(0, 1));         // X != Y over a single value
ac3(bad).value().has_value();           // false  (a domain emptied; NOT an error)

// Unsatisfiable is a valid empty result, not an error.
Csp three;
three.domains = {{0, 1}, {0, 1}, {0, 1}};
three.binary.push_back(ne(0, 1));
three.binary.push_back(ne(0, 2));
three.binary.push_back(ne(1, 2));
backtracking_search(three).value().has_value();  // false (no solution)
solution_count(three, 0).value();                // 0
ac3(three).value().has_value();                  // true — still arc consistent,
                                                 // yet unsatisfiable

// A general n-ary all-different, exercising the general-constraint path.
Csp alldiff;
alldiff.domains = {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}};
alldiff.general.push_back(Constraint{{0, 1, 2},
    [](std::span<const std::int64_t> v) {
        return v[0] != v[1] && v[0] != v[2] && v[1] != v[2];
    }});
backtracking_search(alldiff).value().value();    // {0, 1, 2}

// Shape faults are domain_error.
Csp oob;
oob.domains = {{0, 1}, {0, 1}};
oob.binary.push_back(ne(0, 2));                  // variable 2 does not exist
backtracking_search(oob).error();                // MathError::domain_error
ac3(oob).error();                                // MathError::domain_error
solution_count(Csp{}, 0).error();                // MathError::domain_error (empty CSP)
```

## The declarative layer — `WireCsp`

Everything above carries its constraints as `std::function`. That is the right
representation for an in-process solver, because any predicate the caller can write is
expressible, but it has a hard consequence: a `std::function` is a closure over this
process's address space. It cannot be sent to a worker, and it cannot be read by a code
generator. Distribution and ahead-of-time emission therefore need constraints as *data*,
and `WireCsp` is that form.

The trade is deliberate and worth stating plainly rather than discovering later: the
declarative form is strictly **less expressive** than the functional one. It covers the
eight constraint kinds below and nothing else. A problem outside that set is still
perfectly solvable in process through `Csp`; it simply cannot be shipped or compiled, and
`WireCsp` will not pretend otherwise.

```cpp
enum class ConstraintKind : std::uint8_t {
    not_equal,      // x != y
    equal,          // x == y
    less_equal,     // x + k <= y
    abs_diff_ne,    // |x - y| != k  -- the diagonal constraint of the n-queens family
    all_different,  // pairwise distinct
    linear_eq,      // sum(coef[i] * x[scope[i]]) == rhs
    linear_le,      // sum(coef[i] * x[scope[i]]) <= rhs
    table_allowed,  // an extensional relation: a flat row-major array of allowed tuples
};

struct WireConstraint {
    ConstraintKind kind{ConstraintKind::not_equal};
    std::vector<std::size_t> scope;
    std::vector<std::int64_t> params;
};

struct WireCsp {
    std::vector<std::vector<std::int64_t>> domains;
    std::vector<WireConstraint> constraints;

    [[nodiscard]] auto num_vars() const noexcept -> std::size_t;
};

[[nodiscard]] auto fixed_arity(ConstraintKind kind) noexcept -> std::optional<std::size_t>;
[[nodiscard]] auto validate(const WireCsp& w) -> Result<void>;
[[nodiscard]] auto holds(const WireConstraint& c, std::span<const std::int64_t> values) -> bool;
[[nodiscard]] auto as_csp(const WireCsp& w) -> Result<Csp>;
```

`as_csp` converts back, so the *same* problem runs through every solver above unchanged.
That is what makes a distributed or emitted answer checkable against a known-good one
rather than merely self-consistent.

Two details in that conversion are load-bearing. `all_different` is expanded into its
pairwise not-equals — which is exactly what it means — because AC-3 propagates over
**binary** constraints only; leaving it in `general` would be identical in meaning and
strictly weaker in pruning. And any constraint of arity two becomes a `BinaryConstraint`
for the same reason.

### Overflow is refused once, not checked forever

`linear_eq` and `linear_le` accumulate a weighted sum, and a sum that overflows
`std::int64_t` would be undefined behaviour in the evaluator and a wrong answer in emitted
code. Rather than test on every evaluation — in the inner loop of a search, and in
generated CUDA where there is nowhere to report it — `validate` computes the worst-case
magnitude **once**, from the coefficients and the domain extremes, and refuses a
constraint that could overflow with `MathError::overflow`. Every constraint that passes
validation can then be evaluated with plain `int64` arithmetic that cannot overflow.

`validate` also refuses a variable repeated within one scope. That is not pedantry: a
repeated variable would make the constraint's reading depend on which occurrence a solver
bound first, and refusing is better than silently picking one of two meanings.

## Partitioning the assignment space

```cpp
[[nodiscard]] auto prefix_count(const WireCsp& w, std::size_t fixed_vars) -> Result<std::uint64_t>;
[[nodiscard]] auto prefix_assignment(const WireCsp& w, std::size_t fixed_vars,
                                     std::uint64_t index) -> Result<std::vector<std::int64_t>>;
[[nodiscard]] auto restrict_prefix(const WireCsp& w, std::span<const std::int64_t> assignment)
    -> Result<WireCsp>;
```

Fixing the first *k* variables to one combination of values from their domains yields an
independent sub-problem. Over the ascending product of those *k* domains the sub-problems
**partition** the assignment space exactly: every complete assignment lies in exactly one
of them, none in two. That is the property [`nimblecas.csp_dist`](csp_dist.md) rests on,
and it is what makes a distributed *unsatisfiability* verdict meaningful rather than
merely an absence of results.

`prefix_assignment` enumerates in ascending lexicographic order — the last fixed variable
varying fastest — which is the same order the serial search visits, and is why "the
lowest-indexed shard wins" reproduces `backtracking_search`'s answer exactly.

```cpp
// The declarative 8-queens: rows pairwise distinct, and no shared diagonal.
WireCsp w;
for (std::size_t i = 0; i < 8; ++i) {
    w.domains.push_back({0, 1, 2, 3, 4, 5, 6, 7});
}
w.constraints.push_back(WireConstraint{
    .kind = ConstraintKind::all_different, .scope = {0, 1, 2, 3, 4, 5, 6, 7}, .params = {}});
for (std::size_t i = 0; i < 8; ++i) {
    for (std::size_t j = i + 1; j < 8; ++j) {
        w.constraints.push_back(WireConstraint{
            .kind = ConstraintKind::abs_diff_ne,
            .scope = {i, j},
            .params = {static_cast<std::int64_t>(j - i)}});
    }
}

const Csp same_problem = as_csp(w).value();
solution_count(same_problem, 0).value();   // 92 — the same 92 the functional encoding gives

// The partition, checked by counting: the sub-counts sum to the whole count.
const std::uint64_t prefixes = prefix_count(w, 2).value();          // 64
for (std::uint64_t i = 0; i < prefixes; ++i) {
    const auto prefix = prefix_assignment(w, 2, i).value();
    const WireCsp sub = restrict_prefix(w, prefix).value();
    // solution_count(as_csp(sub).value(), 0) summed over i == 92
}
```

## See also


- [`nimblecas.parallel`](parallel.md) — the `transform_index` branch-parallel map
  `parallel_search` fans the first-variable split across.
- [`nimblecas.lp`](lp.md) — the sibling optimisation solver in the algorithmic
  layer (linear programming over exact arithmetic).
- [`nimblecas.core`](core.md) — the `Result<T>` / `MathError` railway every entry
  point returns through.
- [`nimblecas.csp_dist`](csp_dist.md) — the same problem distributed over `taskdag`, on the
  exact prefix partition described above.
- [`nimblecas.csp_compile`](csp_compile.md) — the same problem emitted as C++23, CUDA or
  Triton source.
- [Documentation hub](../Index.md)
