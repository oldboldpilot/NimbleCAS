# `nimblecas.planning` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/planning/planning.cppm`

A classical automated planning layer situated above [`nimblecas.search`](search.md).
A planning task comprises a set of finite-domain state variables, a set of operators (actions)
characterised by preconditions and effects, an initial state, and a partial goal condition. A plan is
a sequence of operator applications that transitions the system from the initial state into one satisfying
the goal.

The module is governed by a strict **honesty boundary** (Code Policy Rule 32):

- **SAS+ formulation over STRIPS propositions:** the module models state spaces using SAS+
  (Simplified Action Structures) rather than STRIPS boolean propositions. Each variable has a finite domain,
  and a complete state assigns exactly one value to each variable. This design decision is structural:
  SAS+ makes **mutual exclusion structural**. A package cannot be simultaneously in two locations
  because its corresponding state variable can hold only one value at a time. In STRIPS, mutual exclusion
  must be encoded via explicit negative axioms that every heuristic must rediscover. Planners such as
  Fast Downward adopt SAS+ for this exact reason.
- **Delete-relaxation heuristics and their honesty boundaries:** all three implemented heuristics derive
  from the *delete relaxation*: assuming that once a variable attains a value, it retains that value
  permanently alongside any subsequent assignments. While solving a relaxed task is polynomial-time, the
  relaxation discards negative interactions and undo actions:
  - `h_max` — **Admissible**. Estimates the cost of achieving a set of facts as the cost of its
    *dearest* (maximum-cost) member. Because it never overestimates the true remaining distance, A* search
    guided by `h_max` is guaranteed to return an **optimal** minimum-cost plan. However, `h_max` is
    inherently weak: taking the maximum discards the cost of achieving all other goal facts.
  - `h_add` — **Inadmissible**. Estimates the cost as the direct *sum* over all facts in the set. While
    far more informative than `h_max` in practice, it is systematically too large because it counts
    shared subplans once for each goal fact that depends on them. It must never be used to claim plan
    optimality.
  - `h_ff` — **Inadmissible**, but the standard practical workhorse. Rather than summing fact costs, it
    extracts an explicit *relaxed plan* and computes its total cost. Shared actions serving multiple goals
    are counted exactly once. Furthermore, `h_ff` generates **preferred operators** (Hoffmann's helpful
    actions) — the subset of relaxed plan operators applicable in the current state.
  - *Measured comparison from tests:* on a task with a shared prefix (`shared_prefix(4)`), a single setup
    action enables 4 independent goals, yielding a true optimal plan cost of $1 + 4 = 5$. Because `h_add`
    charges the setup action to each goal independently, it reports a cost of 8. In contrast, `h_ff`
    extracts a unified relaxed plan and counts the setup once, reporting exactly 5, while `h_max` reports 2
    (the dearest single goal). That single measurement illustrates why `h_add` overestimates and why
    `h_ff` behaves far better.
- **Search guarantees:**
  - `astar_plan` with `h_max` returns an **optimal plan**, or reports honestly that none exists.
  - `gbfs_plan` (Greedy Best-First Search on `h_ff` with preferred operators) returns a **valid plan**
    and makes **no claim of optimality**. Dual open lists alternate between preferred operators and
    standard successors; the preferred list accelerates search when the heuristic gradient is accurate,
    while the standard list prevents search starvation when it is misleading.
  - `gbfs_plan_hadd` provides a baseline greedy search using `h_add` without preferred operators to serve
    as an experimental control.
- **Mandatory plan replay and verification:** every plan returned by `astar_plan` or `gbfs_plan` is
  replayed and verified from the initial state using `validate_plan` before being returned to the caller.
  Applicability is verified at every step. If a search ever produces a plan that fails to reach the goal,
  the solve reports `MathError::undefined_value` rather than returning a corrupt plan. If a plan's stated
  cost disagrees with its step costs, it is reported as `MathError::domain_error`.
- **Honest error distinction:** `MathError::undefined_value` indicates that the task is provably
  unsolvable (the search space was exhausted without finding a goal). In contrast, `MathError::not_converged`
  indicates that the expansion budget (`max_expansions`) was exhausted while states remained open. These
  convey fundamentally different facts and are never conflated.
- **Determinism:** tie-breaking rules order states by minimum $h$, then by lowest operator index, and
  finally by state canonical ordering. Searches are strictly deterministic across runs and platforms.
- **No distributed planner:** `nimblecas.planning_dist` does not currently exist. The module provides
  canonical state encoding and stable 64-bit state hashing for in-process use and future wire compatibility,
  but distributed planning is deliberately not implemented.

```cpp
import nimblecas.planning;
```

Depends on [`core`](core.md) (`Result`, `MathError`).

## Data structures

```cpp
struct FactPair {
    std::size_t var{0};
    std::int32_t value{0};

    [[nodiscard]] auto operator==(const FactPair&) const noexcept -> bool = default;
    [[nodiscard]] auto operator<=>(const FactPair&) const noexcept = default;
};

using State = std::vector<std::int32_t>;
using Condition = std::vector<FactPair>; // Kept sorted by var

struct Operator {
    std::string name;
    Condition preconditions;
    Condition effects;
    std::int64_t cost{1};
};

struct Task {
    std::vector<std::int32_t> domain_size;
    std::vector<Operator> operators;
    State initial;
    Condition goal;

    [[nodiscard]] auto num_vars() const noexcept -> std::size_t { return domain_size.size(); }
};

struct Plan {
    std::vector<std::size_t> steps;
    std::int64_t cost{0};

    [[nodiscard]] auto length() const noexcept -> std::size_t { return steps.size(); }
};

struct FfResult {
    std::optional<std::int64_t> value;
    std::vector<std::size_t> preferred;
};

struct SearchStats {
    std::uint64_t expanded{0};
    std::uint64_t generated{0};
    std::uint64_t evaluated{0};
};

struct PlanResult {
    Plan plan;
    SearchStats stats;
};
```

| Type | Role |
| :--- | :--- |
| `FactPair` | Assignment of variable `var` to `value`. Values must be in $0 \le \text{value} < \text{domain\_size}[\text{var}]$. |
| `State` | Vector of size `num_vars()` defining the assignment of every variable. |
| `Condition` | Vector of `FactPair` entries representing partial states (preconditions, goals). Always kept sorted by `var` with no duplicate variables. |
| `Operator` | Transition action. `cost` must be non-negative ($\ge 0$). |
| `Task` | Self-contained SAS+ task specification. |
| `Plan` | Executable sequence of operator indices and accumulated cost. |
| `FfResult` | The relaxed plan cost `value` and the sorted list of helpful action indices `preferred`. |
| `PlanResult` | The verified `Plan` alongside search performance counters (`SearchStats`). |

## Task validation and execution semantics

```cpp
[[nodiscard]] auto validate(const Task& task) -> Result<void>;
[[nodiscard]] auto holds(const State& state, const Condition& condition) -> bool;
[[nodiscard]] auto applicable(const Task& task, const State& state, std::size_t op_index) -> bool;
[[nodiscard]] auto apply(const Task& task, const State& state, std::size_t op_index)
    -> Result<State>;
[[nodiscard]] auto validate_plan(const Task& task, const Plan& plan) -> Result<State>;
```

| Function | Guarantee | Error conditions |
| :--- | :--- | :--- |
| `validate` | Validates task structure: non-empty positive domains, initial state matching variable count, conditions sorted with valid ranges, non-negative operator costs. | `MathError::domain_error` on any malformed element or duplicate condition variable. |
| `holds` | Tests whether `state` satisfies every variable-value assignment in `condition`. | Infallible (`bool`). |
| `applicable` | Returns true if `op_index < operators.size()` and its preconditions hold in `state`. | Infallible (`bool`). |
| `apply` | Applies operator effects to `state`. | `MathError::domain_error` if operator is not applicable. |
| `validate_plan` | Replays `plan` step-by-step from `task.initial`, verifying applicability at every transition, validating that the final state satisfies `task.goal`, and checking cost consistency. | `MathError::domain_error` on inapplicable step or inconsistent cost; `MathError::undefined_value` if plan does not achieve goal. |

## Heuristic evaluation

```cpp
[[nodiscard]] auto h_max(const Task& task, const State& state) -> std::optional<std::int64_t>;
[[nodiscard]] auto h_add(const Task& task, const State& state) -> std::optional<std::int64_t>;
[[nodiscard]] auto h_ff(const Task& task, const State& state) -> FfResult;
```

| Heuristic | Admissibility | Behaviour | Error / Unreachable |
| :--- | :--- | :--- | :--- |
| `h_max` | **Admissible** | Max-cost over goal facts under delete relaxation. Zero at goal. | Returns `std::nullopt` if goal is unreachable even under relaxation (proving real unreachability). |
| `h_add` | **Inadmissible** | Sum of costs over goal facts under delete relaxation. Double-counts shared actions. Zero at goal. | Returns `std::nullopt` if goal is unreachable. |
| `h_ff` | **Inadmissible** | Cost of extracted relaxed plan. Shared actions counted once. Yields preferred operators. | `value` is `std::nullopt` and `preferred` is empty if goal is unreachable. |

## Search algorithms

```cpp
[[nodiscard]] auto astar_plan(const Task& task, std::uint64_t max_expansions)
    -> Result<PlanResult>;

[[nodiscard]] auto gbfs_plan(const Task& task, std::uint64_t max_expansions)
    -> Result<PlanResult>;

[[nodiscard]] auto gbfs_plan_hadd(const Task& task, std::uint64_t max_expansions)
    -> Result<PlanResult>;
```

| Search | Optimality | Methodology | Errors |
| :--- | :--- | :--- | :--- |
| `astar_plan` | **Optimal** | A* search guided by admissible `h_max`. Priority queue ordered by $f = g + h$, breaking ties by lower $h$. Replays and verifies plan before return. | `MathError::domain_error` if task invalid; `undefined_value` if provably unsolvable; `not_converged` if `max_expansions` exceeded. |
| `gbfs_plan` | Satisficing (no length claim) | Greedy best-first search on `h_ff` using dual alternating priority queues (preferred vs standard operators). Replays and verifies plan before return. | Same error semantics as `astar_plan`. |
| `gbfs_plan_hadd` | Satisficing (no length claim) | Greedy best-first search on `h_add` without preferred operators (experimental baseline). | Same error semantics as `astar_plan`. |

## Canonical state serialization and stable hashing

```cpp
[[nodiscard]] auto encode_state(const State& state) -> std::vector<std::byte>;
[[nodiscard]] auto decode_state(std::span<const std::byte> bytes, std::size_t num_vars)
    -> Result<State>;
[[nodiscard]] auto hash_state(const State& state) noexcept -> std::uint64_t;
```

- `encode_state` / `decode_state`: serialises state vectors into canonical byte sequences. Used for
  hash tables, closed sets, and wire representations.
- `hash_state`: computes an architecture-independent, seed-independent 64-bit hash of the state
  variables. Reproducible across builds, OS platforms, and runs.

## Worked examples

### Optimal planning on a corridor navigation problem

```cpp
import std;
import nimblecas.core;
import nimblecas.planning;

using namespace nimblecas;
using namespace nimblecas::planning;

// Construct an n-room corridor (0 -> 1 -> ... -> n-1)
const std::int32_t n = 5;
Task task;
task.domain_size = {n};
task.initial = {0};
task.goal = {FactPair{.var = 0, .value = n - 1}};

for (std::int32_t i = 0; i + 1 < n; ++i) {
    task.operators.push_back(Operator{
        .name = std::format("fwd{}", i),
        .preconditions = {FactPair{.var = 0, .value = i}},
        .effects = {FactPair{.var = 0, .value = i + 1}},
        .cost = 1
    });
    task.operators.push_back(Operator{
        .name = std::format("back{}", i),
        .preconditions = {FactPair{.var = 0, .value = i + 1}},
        .effects = {FactPair{.var = 0, .value = i}},
        .cost = 1
    });
}

// Find optimal plan using A*
auto res = astar_plan(task, 10000);
if (res) {
    // res->plan.cost == 4
    // res->plan.length() == 4
    // res->plan.steps contains [0, 2, 4, 6] corresponding to fwd0, fwd1, fwd2, fwd3
}
```

### Fast satisficing planning on a logistics task

```cpp
import std;
import nimblecas.core;
import nimblecas.planning;

using namespace nimblecas;
using namespace nimblecas::planning;

Task logistics_task = /* ... multi-package logistics task ... */;

// Solve using greedy best-first search with FF heuristic and preferred operators
auto satisficing_res = gbfs_plan(logistics_task, 100000);
if (satisficing_res) {
    std::println("Found plan with {} steps, total cost: {}",
                 satisficing_res->plan.length(), satisficing_res->plan.cost);
    std::println("States expanded: {}, generated: {}",
                 satisficing_res->stats.expanded, satisficing_res->stats.generated);
}
```
