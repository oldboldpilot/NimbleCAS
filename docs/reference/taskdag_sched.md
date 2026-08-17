# `nimblecas.taskdag_sched` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/taskdag_sched/taskdag_sched.cppm`

Cost-aware, deterministic wavefront task scheduling over [`nimblecas.taskdag`](taskdag.md)
and [`nimblecas.parallel`](parallel.md): three-tier cost resolution, sanitized
longest-expected-processing-time-first (LPT) level ordering with variance risk margin,
device affinity mapping, and a cost-ordered local executor.

```cpp
import nimblecas.taskdag_sched;
```

Depends on [`core`](core.md), [`parallel`](parallel.md), and [`taskdag`](taskdag.md).

## Honesty boundary — hints about time, not results

- **A cost model is a HINT about TIME, never about RESULTS.** A wrong, negative,
  infinite, NaN, or absent estimate changes only the *order* in which independent tasks
  within a wavefront level are executed, never what is computed. `TaskRunResult::outputs`
  is guaranteed **bit-identical to `serial_executor()`** under every policy and every cost input.
- **Determinism is strictly preserved.** Given the same `TaskGraph`, `CostTable`, and
  `ScheduleParams`, `level_order()` produces the **identical** order across runs and
  platforms. Ties are broken strictly by **ascending `TaskId`** (issuance order) — never
  by clock, memory address, or thread completion. Costs are frozen offline inputs,
  never feedback from live clocks inside executors.
- **Hostile inputs are sanitized.** Negative or non-finite means (`NaN`, `±inf`) normalize
  to the unknown state (`{0.0, 0.0}`); usable means are clamped to `[1e-9, 1e9]` seconds;
  variances to `[0.0, 1e18]` seconds². Sanitization guarantees a strict weak ordering
  and prevents floating-point comparison undefined behavior.
- **No performance claim is asserted.** Whether cost-aware ordering yields any wall-clock
  reduction on a given workload is **unmeasured pending execution of the pre-registered
  benchmark harness** (ROADMAP §6.1 / M6_SPEC §2.5). On uniform-cost levels (such as
  `modgcd` rounds), single-worker runs, systems where worker count exceeds level size, or
  hint-free graphs, ordering provably changes nothing.
- **Affinity is metadata mapping, not placement optimization.** `Affinity` tags are mapped
  deterministically to wire placement codes (`cpu_only` → 1, `gpu_only` → 2, `hybrid` → 1
  today); placement enforcement and load-balancing across devices are properties of the
  underlying runtime or broker outside this repository.

## Three-tier cost resolution

For any task in a `TaskGraph`, its scheduling cost is resolved in strict order of precedence:

1. **Caller-supplied per-task hint:** if known (`mean_seconds > 0.0` and finite after sanitization),
   the per-task `CostHint` is used directly.
2. **Frozen `CostTable` per-`OpId` entry:** if the per-task hint is unknown and the task is a
   named task (`op_id` is non-empty), looked up in the caller-supplied `CostTable`.
3. **Unknown:** if no hint and no matching table entry exist, effective cost is defined as
   `0.0`, which sorts after all known tasks in ascending `TaskId` order.

Closure tasks (empty `op_id`) never consult the table. On a graph where no task has a known hint
and no `CostTable` is provided, `level_order()` is a **strict no-op** returning exact issuance order.

## Ordering policy

Within each wavefront level, tasks are ordered by effective cost:

$$\text{effective\_cost} = \text{mean} + \lambda \sqrt{\text{variance}} \quad (\lambda \ge 0)$$

where $\lambda$ (`ScheduleParams::risk_lambda`, default `0.0`) provides an optional variance risk
margin to hedge against high-variance tail latency.

Tasks are sorted by `(effective_cost descending, TaskId ascending)`.

## API

All entry points live in namespace `nimblecas`, `[[nodiscard]]`.

| Type / Function | Signature | Behavior |
| :--- | :--- | :--- |
| `CostTable` | `using CostTable = std::map<OpId, CostHint, std::less<>>;` | Frozen map from registered `OpId` strings to calibrated `CostHint` statistics. |
| `ScheduleParams` | `struct ScheduleParams { double risk_lambda{0.0}; };` | Scheduling parameters; `risk_lambda` controls variance risk margin (default 0.0). |
| `Affinity` | `enum class Affinity : std::uint8_t { cpu_only, gpu_only, hybrid };` | Device affinity categories for registered operations. |
| `AffinityTable` | `using AffinityTable = std::map<OpId, Affinity, std::less<>>;` | Frozen map from `OpId` strings to `Affinity` categories. |
| `sanitize` | `auto sanitize(const CostHint& hint) noexcept -> CostHint` | Clamps means to `[1e-9, 1e9]`, variances to `[0, 1e18]`; non-finite/non-positive means normalize to `{0.0, 0.0}`. |
| `effective_cost` | `auto effective_cost(const CostHint& task_hint, std::string_view op, const CostTable* table, const ScheduleParams& params) noexcept -> double` | Three-tier cost resolution: returns `mean + lambda * sqrt(variance)` for known hints, or `0.0` for unknown. |
| `level_order` | `auto level_order(const TaskGraph& g, std::span<const TaskId> level, const CostTable* table, const ScheduleParams& params) -> std::vector<TaskId>` | Computes deterministic task execution order for a wavefront level `(cost desc, TaskId asc)`. |
| `to_placement` | `auto to_placement(Affinity a) noexcept -> std::uint8_t` | Maps `Affinity` to wire placement tag (`cpu_only` → 1, `gpu_only` → 2, `hybrid` → 1). |
| `cost_ordered_local_executor` | `auto cost_ordered_local_executor(const CostTable* table = nullptr, ScheduleParams params = {}, std::size_t grain = 1) -> std::unique_ptr<Executor>` | Wavefront local executor applying cost-based ordering within each level. Produces outputs bit-identical to `serial_executor()`. |

## Error model

This module contains no throwing operations and generates no synthetic errors.
Execution results and error poisoning propagate according to the exact `TaskGraph` error model:
a failing task is executed and records its own `MathError`; downstream dependent tasks are poisoned
and carry the error of their lowest-topological-index failing ancestor.

## Worked example

```cpp
import std;
import nimblecas.core;
import nimblecas.taskdag;
import nimblecas.taskdag_sched;
using namespace nimblecas;

// Setup a level with varying costs:
TaskGraph g;
const auto a = g.add_task(my_fn_a, {}, CostHint{1.0, 0.0}).value();
const auto b = g.add_task(my_fn_b, {}, CostHint{5.0, 0.0}).value();
const auto c = g.add_task(my_fn_c, {}, CostHint{3.0, 0.0}).value();
const auto d = g.add_task(my_fn_d, {}, CostHint{5.0, 0.0}).value();
const auto e = g.add_task(my_fn_e, {}, CostHint{0.0, 0.0}).value();  // unknown

// Compute level order for level 0:
const auto order = level_order(g, g.level(0), nullptr, ScheduleParams{});
// order == [b, d, c, a, e]
// (5.0 tie broken by TaskId b < d; 3.0 next; 1.0 next; unknown e last)

// Execute with cost-ordered local executor:
const auto exec = cost_ordered_local_executor(nullptr, ScheduleParams{}, /*grain=*/1);
const auto res = exec->run(g).value();

// Outputs are bit-identical to serial_executor():
const auto ser_res = serial_executor()->run(g).value();
assert(res.outputs.size() == ser_res.outputs.size());
```

## See also

- [`nimblecas.taskdag`](taskdag.md) — Single-node task-DAG scheduler and `Executor` interface.
- [`nimblecas.parallel`](parallel.md) — Deterministic fork–join runtime (`transform_index`).
- [Documentation hub](../Index.md)
