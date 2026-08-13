# `nimblecas.taskdag` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/taskdag/taskdag.cppm`

A **deterministic, single-node** task-DAG scheduler: wavefront execution of a
dependency graph of pure computation steps over
[`nimblecas.parallel`](parallel.md) (TBB/PPL), with a pluggable `Executor`
seam.

```cpp
import nimblecas.taskdag;
```

Depends on [`core`](core.md) (`Result` / `MathError`) and
[`parallel`](parallel.md) (`transform_index`, the order-preserving fork-join
primitive `local_parallel_executor` fans a wavefront level out over).

## Honesty boundary — single-node core with named-task extension

**This is a single-node, local fork-join scheduler, now featuring an additive
`TaskRegistry` and `add_named_task` path for named operations.** `Payload`
(`std::vector<std::byte>`) is serializable by construction. `TaskFn` closures
cannot be serialized to remote processes. Named tasks identified by a
registered `OpId` (`"<domain>.<op>/v<N>"`) allow a remote/distributed backend
(such as [`nimblecas.taskdag_sgee`](taskdag_sgee.md)) to ship op identifiers and
serialized arguments while executing identical code across nodes. Named tasks
run on local executors bit-identically to plain closure tasks.

## Determinism contract

`serial_executor()` and `local_parallel_executor()` produce **bit-identical**
`TaskRunResult::outputs` for the same `TaskGraph`, for any thread count:

- A `TaskGraph` is built **incrementally**: `add_task(fn, deps, hint)` or
  `add_named_task(reg, op, deps, hint)` appends one task whose dependencies must
  already have been issued `TaskId`s, so the graph is **acyclic by construction** —
  no cycle check is ever needed or possible to fail.
- Each task's `depth = 1 + max(parent depth)` (`0` with no deps) places it in
  a wavefront **level**. An executor runs level `0`, then level `1`, and so
  on; within a level every task is independent of every other task in that
  level (their dependencies all sit at strictly smaller depth), so an
  executor is free to run a level's tasks concurrently.
- Every `TaskFn` is required to be a **pure** function of its parents'
  payloads (no shared mutable state, no visible side effects) — the only
  requirement `TaskFn` places on a caller, since `local_parallel_executor`
  invokes every task in a wavefront level concurrently.
- `local_parallel_executor`'s per-level fan-out goes through
  `nimblecas.parallel::transform_index`, which is **order-preserving**, so
  the accumulation back into `outputs` always happens in a single fixed index
  order — never in thread-completion order. No atomics-driven
  nondeterminism.

## Error poisoning

If task `T`'s own `TaskFn` returns an error, `T` **is executed** (that is how
the error was discovered) and the error becomes `T`'s output; `T` counts
toward `executed`. Any task that depends, directly or transitively, on `T` is
**poisoned**: it is **not** executed (its `TaskFn` is never called), its
output slot carries the error of its **lowest-topological-index failing
ancestor** (`TaskId` order == issuance order == a topological order here, so
"lowest index" is well-defined and independent of scheduling), and this
propagates deterministically to further descendants. An **independent
subgraph with no failing ancestor still runs to completion.**

## API

All entry points live in namespace `nimblecas`, `[[nodiscard]]`.

| Type / Function | Signature | Behavior |
| :--- | :--- | :--- |
| `Payload` | `using Payload = std::vector<std::byte>;` | A task's output (or another task's input): an opaque byte blob, serializable by construction. |
| `TaskFn` | `using TaskFn = std::function<Result<Payload>(std::span<const Payload>)>;` | A task's compute step: parents' outputs (in the exact order given to `add_task`) → this task's output or error. Must be **pure**. |
| `OpId` | `using OpId = std::string;` | Versioned canonical string identifier for an op (`<domain>.<op>/v<N>`). |
| `TaskRegistry` | `class TaskRegistry` | Registry mapping versioned `OpId`s to `TaskFn` bodies. `register_op`, `find`, `size`, `fingerprint` (FNV-1a). |
| `TaskId` | `struct { std::size_t value{0}; }` | Opaque handle; `TaskId` order == issuance order == a valid topological order. |
| `CostHint` | `struct { double mean_seconds{0.0}; double variance{0.0}; }` | A recorded cost estimate, readable via `TaskGraph::hint(id)`. |
| `TaskGraph::add_task` | `auto add_task(TaskFn fn, std::span<const TaskId> deps = {}, CostHint hint = {}) -> Result<TaskId>` | Appends a closure task depending on `deps`. Fails with `domain_error`, **without adding the task**, if any `deps` entry names an unissued `TaskId`. |
| `TaskGraph::add_named_task` | `auto add_named_task(const TaskRegistry& reg, OpId op, std::span<const TaskId> deps = {}, CostHint hint = {}) -> Result<TaskId>` | Appends a named task looking up `op` in `reg`. Fails with `domain_error` if `op` is unregistered or `deps` invalid. |
| `TaskGraph::op_id` | `auto op_id(TaskId id) const noexcept -> std::string_view` | The registered op name for named tasks, or empty for plain closures. |
| `TaskGraph::size` | `auto size() const noexcept -> std::size_t` | Number of tasks issued so far. |
| `TaskGraph::deps` / `depth` / `hint` / `num_levels` / `level` / `invoke` | — | Accessors for `Executor` implementations; not part of the curated graph-building surface. |
| `serial_executor` | `auto serial_executor() -> std::unique_ptr<Executor>` | Runs every task strictly one at a time, wavefront order. The deterministic reference implementation — validate any faster executor's output against it. |
| `local_parallel_executor` | `auto local_parallel_executor() -> std::unique_ptr<Executor>` | Wavefront execution with each level's independent tasks fanned out over `parallel::transform_index`. Bit-identical to `serial_executor()` — purely a wall-clock optimization. |

### `TaskRunResult`

```cpp
struct TaskRunResult {
    std::vector<Result<Payload>> outputs;      // outputs[id.value]: own result, or (if
                                                // poisoned) the lowest failing ancestor's error
    std::vector<double> measured_seconds;      // wall-clock time per task; 0.0 if poisoned
    std::size_t executed{0};                   // tasks actually invoked (poisoned ones excluded;
                                                // a task that ran and returned its OWN error IS counted)
};
```

### `Executor`

```cpp
class Executor {
public:
    virtual ~Executor() = default;
    [[nodiscard]] virtual auto name() const noexcept -> std::string_view = 0;
    [[nodiscard]] virtual auto run(const TaskGraph& g) -> Result<TaskRunResult> = 0;
};
```

Every `Executor` — the two shipped here, and any future one built against
this same interface (a distributed backend included) — must honour the
determinism contract above: for a given `TaskGraph`, `run` must return
outputs bit-identical to every other `Executor`'s.

## Error model

| Condition | Error |
| :--- | :--- |
| `add_task`: a `deps` entry names a `TaskId` this graph never issued | `MathError::domain_error` (task not added) |
| A task's own `TaskFn` returns an error | that error, on that task's output slot; poisons its descendants |
| Anything else | whatever `TaskFn` itself returns — this module never invents an error of its own beyond the two above |

## Worked example — the diamond `A -> {B, C} -> D`

```cpp
import std;
import nimblecas.core;
import nimblecas.taskdag;
using namespace nimblecas;

// Payload <-> int64 encode/decode (the "serializable by construction" contract).
auto encode_i64 = [](std::int64_t v) -> Payload {
    const auto bytes = std::bit_cast<std::array<std::byte, sizeof(std::int64_t)>>(v);
    return Payload(bytes.begin(), bytes.end());
};
auto decode_i64 = [](std::span<const std::byte> p) -> std::int64_t {
    std::array<std::byte, sizeof(std::int64_t)> bytes{};
    std::ranges::copy(p, bytes.begin());
    return std::bit_cast<std::int64_t>(bytes);
};

// A = 7, B = A*2, C = A+3, D = B+C. Hand value: B=14, C=10, D=24.
TaskGraph g;
const auto a = g.add_task([&](std::span<const Payload>) -> Result<Payload> {
                    return encode_i64(7);
                }).value();
const auto b = g.add_task(
    [&](std::span<const Payload> ps) -> Result<Payload> {
        return encode_i64(decode_i64(ps[0]) * 2);
    },
    std::vector<TaskId>{a}).value();
const auto c = g.add_task(
    [&](std::span<const Payload> ps) -> Result<Payload> {
        return encode_i64(decode_i64(ps[0]) + 3);
    },
    std::vector<TaskId>{a}).value();
const auto d = g.add_task(
    [&](std::span<const Payload> ps) -> Result<Payload> {
        return encode_i64(decode_i64(ps[0]) + decode_i64(ps[1]));
    },
    std::vector<TaskId>{b, c}).value();

const auto exec = serial_executor();
const auto res = exec->run(g).value();

decode_i64(res.outputs[a.value].value());  // 7
decode_i64(res.outputs[b.value].value());  // 14  (B saw A's own output)
decode_i64(res.outputs[c.value].value());  // 10  (C saw A's own output)
decode_i64(res.outputs[d.value].value());  // 24  (D == B + C, deps delivered in {b, c} order)
res.executed == g.size();                   // true — every task in the diamond ran

// serial_executor() and local_parallel_executor() agree bit-for-bit on the SAME graph.
const auto par_res = local_parallel_executor()->run(g).value();
// res.outputs and par_res.outputs compare equal element-wise, and both report executed == 4.

// Error poisoning: if B fails, D is poisoned (never invoked) and carries B's own error,
// while any task independent of B still runs to completion.
```

## See also

- [`nimblecas.parallel`](parallel.md) — the deterministic, order-preserving
  fork-join runtime (`transform_index`) `local_parallel_executor` fans each
  wavefront level out over.
- [Documentation hub](../Index.md)
