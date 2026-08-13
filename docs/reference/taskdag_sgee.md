# `nimblecas.taskdag_sgee` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/taskdag_sgee/taskdag_sgee.cppm`

An SGEE-backed distributed `Executor` backend for [`nimblecas.taskdag`](taskdag.md),
implementing coordinator-side level scheduling over SGEE's task broker queue and an
out-of-band result channel.

```cpp
import nimblecas.taskdag_sgee;
```

Depends on [`core`](core.md) (`Result` / `MathError::distributed_error`) and
[`taskdag`](taskdag.md) (`TaskGraph`, `TaskRegistry`, `Executor`).

## Honesty boundary — transport vs math errors (Rule 32)

- **Math errors are data:** If a task's own `TaskFn` returns a `MathError` (e.g. `domain_error`),
  that error is captured as data in the `ResultEnvelope`, transmitted back to the coordinator,
  stored in `TaskRunResult::outputs`, and poisons downstream tasks with identical lowest-index
  origin rules and executed counts as `serial_executor()`.
- **Transport failures abort the whole run:** Any transport-class failure (missing result,
  DLQ/Dead state, payload exceeding the 64 MiB wire cap, envelope decode failure, registry
  fingerprint mismatch, or whole-run liveness deadline) aborts the entire `run()` with
  `MathError::distributed_error`. The executor **never** fabricates, patches, or partially
  returns outputs.
- **Determinism contract:** When `run()` succeeds, `TaskRunResult::outputs` is **bit-identical**
  to `serial_executor()->run(g)` on the same graph with the same registry. The distributed
  executor is a transport, not a recomputation.

## Scope note (Part 1 vs Part 2)

- **Part 1 (SGEE-free Scaffolding):** The complete distributed executor seam (`SgeeDistributedExecutor`),
  envelope codec, worker pump (`run_worker_pump`), out-of-band result channel (`InMemoryResultChannel`),
  and deterministic in-memory `FakeBrokerPort` (with injectable hand clock for tests).
  Builds and tests with zero SGEE dependency.
- **Part 2 (SGEE C-ABI Port):** `CapiBrokerPort` wrapping `libsgee_capi` behind `-DNIMBLECAS_SGEE=ON`.
  When enabled via `-DNIMBLECAS_SGEE=ON -DNIMBLECAS_SGEE_ROOT=/path/to/SGEE`, `sgee_distributed_executor`
  constructs a live `CapiBrokerPort` backed by a real SGEE durable WAL task broker.
  **WAL lifecycle:** the factory opens one broker on a process-unique WAL file
  (`ncsgee-<epoch_ms>-<process_nonce>-<seq>.wal` under `wal_dir`) that is shared by every `run()`
  on the returned executor and best-effort reaped when the executor is destroyed. Because the broker
  is per-executor (not per-run), an executor should not be reused after a `run()` has failed — an
  aborted run leaves stale tasks in its queue. Per-run brokers with success-path WAL deletion are a
  planned refinement.

## Build recipe

Default build (OFF, SGEE-free scaffolding stub):
```bash
cmake -B build -GNinja
cmake --build build
```

Real SGEE-backed backend build (ON, `libsgee_capi` C-ABI):
```bash
cmake -B build -GNinja -DNIMBLECAS_SGEE=ON -DNIMBLECAS_SGEE_ROOT=/path/to/StochasticGraphExecutionEngine
cmake --build build
```
Requires a built SGEE tree where `sgee_capi.h` is present under `${NIMBLECAS_SGEE_ROOT}/bindings/capi` and `libsgee_capi` shared library is present under `${NIMBLECAS_SGEE_ROOT}/build-capi/lib` or `${NIMBLECAS_SGEE_ROOT}/build/lib` or `${NIMBLECAS_SGEE_ROOT}/lib`.

## API

All entry points live in namespace `nimblecas` and `nimblecas::sgee_bridge`, `[[nodiscard]]`.

| Type / Function | Signature / Description |
| :--- | :--- |
| `SgeePlacement` | `enum class { direct=0, cpu=1, gpu=2, docker=3, distributed=4, cloud=5 }` |
| `TaskEnvelope` | `struct { u64 registry_fp; string op_id; vector<Payload> args; }` |
| `ResultEnvelope` | `struct { Status status; MathError math_err; double seconds; Payload bytes; }` |
| `encode_task` / `decode_task` | Binary codec framing `"NCDT"` (version 1 LE), capped at 64 MiB. |
| `encode_result` / `decode_result` | Binary codec framing `"NCRT"` (version 1 LE). |
| `ResultChannel` | Abstract out-of-band result store interface (`put`/`get` keyed by qid). |
| `InMemoryResultChannel` | Mutex + hash map implementation of `ResultChannel`. |
| `BrokerPort` | Abstract queue interface (`enqueue`, `lease`, `complete`, `fail`, `heartbeat`, `sweep_expired`, `state`). |
| `FakeBrokerPort` | Deterministic in-memory `BrokerPort` with injectable hand clock. |
| `CapiBrokerPort` | `class CapiBrokerPort : public BrokerPort` — C-ABI wrapper over `libsgee_capi`'s `sgee_task_broker_t*` (ON only). |
| `WorkerPumpConfig` | `struct { u64 worker_id; u64 lease_timeout_ms; u64 idle_backoff_ms; u64 heartbeat_every_ms; }` |
| `run_worker_pump` | Worker execution loop: lease → decode → registry lookup → invoke → put result → complete. |
| `SgeeExecutorConfig` | Config struct with fluent setters (`with_registry`, `with_num_workers`, `with_poll_interval_ms`, etc.). |
| `SgeeDistributedExecutor` | `class SgeeDistributedExecutor : public Executor` — level-synchronous wavefront coordinator. |
| `sgee_distributed_executor` | `auto sgee_distributed_executor(SgeeExecutorConfig) -> Result<unique_ptr<Executor>>` factory. |

## Error model

| Condition | Error |
| :--- | :--- |
| Factory called when `NIMBLECAS_SGEE=OFF` | `MathError::not_implemented` |
| Factory config invalid (null registry / zero workers / zero attempts) | `MathError::domain_error` |
| Graph contains an unnamed closure task | `MathError::not_implemented` (before enqueue) |
| Graph names an OpId absent from the registry | `MathError::domain_error` (before enqueue) |
| Transport / WAL error, lost result, dead state, decode failure, fp mismatch | `MathError::distributed_error` (aborts whole run) |
| Task's own `TaskFn` returns `MathError E` | `outputs[id] = unexpected(E)` (bit-identical to serial) |

## Worked example — diamond DAG over `FakeBrokerPort`

```cpp
import std;
import nimblecas.core;
import nimblecas.taskdag;
import nimblecas.taskdag_sgee;
using namespace nimblecas;

// 1. Populate TaskRegistry
TaskRegistry reg;
reg.register_op("test.const7/v1", [](auto) -> Result<Payload> { return encode_i64(7); });
reg.register_op("test.mul2/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) * 2); });
reg.register_op("test.add3/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) + 3); });
reg.register_op("test.add/v1", [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) + decode_i64(ps[1])); });

// 2. Build named diamond DAG: A=7, B=14, C=10, D=24
TaskGraph g;
const auto a = g.add_named_task(reg, "test.const7/v1").value();
const auto b = g.add_named_task(reg, "test.mul2/v1", std::vector<TaskId>{a}).value();
const auto c = g.add_named_task(reg, "test.add3/v1", std::vector<TaskId>{a}).value();
const auto d = g.add_named_task(reg, "test.add/v1", std::vector<TaskId>{b, c}).value();

// 3. Wire FakeBrokerPort + InMemoryResultChannel + SgeeDistributedExecutor
FakeBrokerPort port;
InMemoryResultChannel results;
SgeeExecutorConfig cfg;
cfg.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1);

SgeeDistributedExecutor dist_exec(cfg, port, results);
const auto res = dist_exec.run(g).value();

decode_i64(res.outputs[d.value].value()); // 24 — BIT-IDENTICAL to serial_executor()
```

## See also

- [`nimblecas.taskdag`](taskdag.md) — local task-DAG scheduler.
- [`nimblecas.core`](core.md) — error handling model.
- [Documentation hub](../Index.md)
