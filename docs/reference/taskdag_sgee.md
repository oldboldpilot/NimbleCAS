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
- **Envelope argument semantics:** `TaskEnvelope::args` carries **bound literals
  first, followed by parent outputs in declared dependency order**. The worker pump
  passes `env.args` verbatim to the registered `TaskFn`, matching local execution bit-for-bit.

## Scope note (Part 1 vs Part 2)

- **Part 1 (SGEE-free Scaffolding):** The complete distributed executor seam (`SgeeDistributedExecutor`),
  envelope codec, worker pump (`run_worker_pump`), out-of-band result channel (`InMemoryResultChannel`),
  and deterministic in-memory `FakeBrokerPort` (with injectable hand clock for tests).
  Builds and tests with zero SGEE dependency.
- **Part 2 (SGEE C-ABI Port):** `CapiBrokerPort` wrapping `libsgee_capi` behind `-DNIMBLECAS_SGEE=ON`.
  When enabled via `-DNIMBLECAS_SGEE=ON -DNIMBLECAS_SGEE_ROOT=/path/to/SGEE`, `sgee_distributed_executor`
  constructs a live `CapiBrokerPort` backed by a real SGEE durable WAL task broker.
  **WAL lifecycle (per-run brokers):** every `run()` on the returned executor opens a **fresh**
  `CapiBrokerPort` on a process-unique WAL file (`ncsgee-<epoch_ms>-<process_nonce>-<seq>.wal`
  under `wal_dir`). On a **successful** run the WAL is deleted; on a **failed** run it is retained
  for post-mortem inspection. Consecutive runs never share a queue, so an executor is **safe to
  reuse after a `run()` has failed** — the next run starts on a clean broker with no stale tasks.
  Broker-open failure surfaces at `run()` time as `MathError::distributed_error`; the factory
  validates configuration only (invalid config → `domain_error`) and does not itself open a broker.
- **Part 3 (Cross-process gRPC — `-DNIMBLECAS_SGEE_GRPC=ON`):** `GrpcBrokerPort` /
  `GrpcResultChannel` drive a **remote** `sgee_queue_node` over gRPC, so a coordinator process and
  its worker **processes** need not share a machine. This option is **independent** of
  `NIMBLECAS_SGEE` (enable either, both, or neither; the default build links honest stubs for both).
  It links **`libsgee_capi_grpc`**, a shared library that statically embeds gRPC/protobuf — so
  NimbleCAS needs the header and the `.so` only, never the gRPC toolchain.
  `sgee_grpc_distributed_executor(cfg, {endpoint})` opens a fresh `GrpcBrokerPort` + `GrpcResultChannel`
  pair per `run()` to `endpoint` (`"host:port"` of the node's queue port); `cfg.wal_dir` is **ignored**
  (the queue lives in the node's `SGEE_DATA_DIR`) and `num_workers == 0` is **valid** — it means "no
  in-process pumps; work is done by separate worker processes attached to the same endpoint".
  - **`sweep_expired` is a no-op** (returns 0): the queue node's driver sweeps expired leases
    autonomously, so the coordinator's per-tick sweep call is a free success and `now_ms` is inert.
  - **Result transport is the node's out-of-band, in-memory store** (never the broker WAL). It is
    **not durable**: a queue-node restart mid-run loses stored results, which the coordinator observes
    as `Completed`-but-absent and — per the honesty boundary below — turns into an honest whole-run
    `distributed_error` abort, never a fabricated value.
  - **Fencing tokens** cross the wire as the full `(local, term, index)` triple; the single-`u64`
    `BrokerPort` token is the `local` half, with `(term, index)` stashed per-qid and re-attached on
    `complete`/`fail`/`heartbeat` (only ever called by the process that leased).
  - **Serialization:** `GrpcBrokerPort`'s mutex is held across each RPC. The intended topology (one
    coordinator thread; each worker a separate process with its own port) never contends. The
    in-process-pump config (`num_workers > 0`) fully serializes its transport on that mutex and is a
    convenience, not a throughput path — use separate worker processes for parallelism.
- **Part 4 (Raft quorum + mTLS — same `-DNIMBLECAS_SGEE_GRPC=ON`):** the same two classes now
  address a **cluster** of `sgee_queue_node`s instead of a single node, so killing the leader no
  longer kills the run. Part 3's single-endpoint API is unchanged and still works verbatim — the
  ring below simply collapses to one node.
  - **Endpoint ring.** `GrpcBrokerPort::connect(vector<string> endpoints, ...)` opens **one client
    per endpoint up front** and shares them through a `GrpcEndpointRing` (the `GrpcResultChannel`
    for the same run shares the *same* ring, so both follow one another's failovers). An atomic
    `active_index` names the node currently believed to be leader.
  - **Following leadership.** A non-leader answers a write with a `StaleToken` rejection and names
    the leader it knows; the port reads that via the C ABI's `sgee_grpc_last_leader_hint()` and
    rotates `active_index` to it. A hint outside the ring, or a plain transport failure, falls back
    to trying the next endpoint in order. `failover_rotations()` counts rotations, so a test can
    assert a failover **actually happened** rather than assuming one did.
  - **mTLS is opt-in and all-or-nothing.** `SgeeGrpcTlsOptions` carries `(ca_cert_path, cert_path,
    key_path)` plus an optional `target_name_override` for a certificate issued to a name the caller
    cannot resolve. An **empty** struct means plaintext; a **partially** filled one is refused, never
    silently downgraded. The node enforces the client side too — with
    `GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY`, a client presenting no certificate
    is dropped at the transport (under TLS 1.3 this surfaces as a **connection teardown**, not a TLS
    alert, since the rejection is post-handshake).
  - **Weak linkage — honest degradation.** The two C-ABI entry points this needs
    (`sgee_grpc_client_open_tls`, `sgee_grpc_last_leader_hint`) are declared **weak**, so NimbleCAS
    links and runs against an older `libsgee_capi_grpc` that predates them. Asking for TLS against
    such a build returns `MathError::not_implemented` — it does not call through a null symbol, and
    it does not fall back to an unencrypted channel.
  - **Bounded result recovery.** A result stored on a node that then lost leadership can be absent
    from the node now serving reads. `cfg.with_max_result_recoveries(n)` (**default 0 — off**) lets
    the coordinator re-enqueue such a task at most `n` times. The bound is the point: a genuinely
    lost result must end the run with `distributed_error`, not spin. `ResultChannel::try_get`
    exists so "not there" can be asked without being an error.
  - **What is still not durable:** the result store remains in-memory and per-node (Part 3). Raft
    replicates the *queue*, not the result store; recovery above is a best-effort re-execution of a
    task, which is safe precisely because execution is deterministic.

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

Cross-process gRPC backend build (ON, `libsgee_capi_grpc` — independent of `NIMBLECAS_SGEE`):
```bash
cmake -B build -GNinja -DNIMBLECAS_SGEE_GRPC=ON -DNIMBLECAS_SGEE_ROOT=/path/to/StochasticGraphExecutionEngine
cmake --build build
```
Requires a built SGEE **gRPC** tree with `sgee_capi_grpc.h` under `${NIMBLECAS_SGEE_ROOT}/bindings/capi` and `libsgee_capi_grpc` under `${NIMBLECAS_SGEE_ROOT}/build-grpc/lib` (built with `-DSGEE_USE_GRPC=ON`, which also produces the `sgee_queue_node` server). The cross-process integration test `taskdag_sgee_grpc_tests` runs only when `NIMBLECAS_SGEE_QUEUE_NODE` points at a built `sgee_queue_node` (else it CTest-skips, exit 77); it spawns the node + two worker processes and asserts bit-identity to `serial_executor`.

The **cluster** test `taskdag_sgee_grpc_cluster_tests` (three legs: plaintext quorum, mTLS quorum,
leader-failover) additionally needs `NIMBLECAS_GEN_TEST_CERTS_SH` pointing at
`tests/gen_test_certs.sh`, which mints a throwaway CA plus server/client certificates **at run
time** — no certificate or key is ever committed (`*.pem`/`*.key`/`*.crt`/`*.csr`/`*.srl` are
gitignored). CTest wires all three variables automatically; running the binary by hand without them
skips (exit 77) or fails on the mTLS leg. It is marked `RUN_SERIAL` because it binds fixed ports and
counts node processes.

Each node in the quorum needs its own `SGEE_DATA_DIR` and the shared peer list, e.g.
`SGEE_NODE_ID=1`, `SGEE_PEERS=1=[::1]:7001,2=[::1]:7002,3=[::1]:7003`, `SGEE_CONSENSUS_PORT`,
`SGEE_QUEUE_PORT`, and — for TLS — the all-or-nothing triple `SGEE_TLS_CA_CERT` / `SGEE_TLS_CERT` /
`SGEE_TLS_KEY` (a partial triple is rejected by the node with `ConfigError::PartialTls`).

## API

All entry points live in namespace `nimblecas` and `nimblecas::sgee_bridge`, `[[nodiscard]]`.

| Type / Function | Signature / Description |
| :--- | :--- |
| `SgeePlacement` | `enum class { direct=0, cpu=1, gpu=2, docker=3, distributed=4, cloud=5 }` |
| `TaskEnvelope` | `struct { u64 registry_fp; string op_id; vector<Payload> args; }` — `args` carries bound literals prepended before parent outputs in declared dependency order. |
| `ResultEnvelope` | `struct { Status status; MathError math_err; double seconds; Payload bytes; }` |
| `encode_task` / `decode_task` | Binary codec framing `"NCDT"` (version 1 LE), capped at 64 MiB. |
| `encode_result` / `decode_result` | Binary codec framing `"NCRT"` (version 1 LE). |
| `ResultChannel` | Abstract out-of-band result store interface (`put`/`get` keyed by qid), plus `try_get` → `Result<optional<Payload>>` where **absence is not an error**. |
| `InMemoryResultChannel` | Mutex + hash map implementation of `ResultChannel`. |
| `BrokerPort` | Abstract queue interface (`enqueue`, `lease`, `complete`, `fail`, `heartbeat`, `sweep_expired`, `state`). |
| `FakeBrokerPort` | Deterministic in-memory `BrokerPort` with injectable hand clock. |
| `CapiBrokerPort` | `class CapiBrokerPort : public BrokerPort` — C-ABI wrapper over `libsgee_capi`'s `sgee_task_broker_t*` (ON only). |
| `GrpcBrokerPort` | `class GrpcBrokerPort : public BrokerPort` — remote `sgee_queue_node` over gRPC via `libsgee_capi_grpc` (`NIMBLECAS_SGEE_GRPC` only). Two `connect` factories: `connect(endpoint, auth, deadline_ms)` (single node) and `connect(vector<endpoints>, auth, deadline_ms, tls)` (Raft ring). `failover_rotations()` reports how many times it followed leadership. |
| `GrpcResultChannel` | `class GrpcResultChannel : public ResultChannel` — the node's out-of-band result store over gRPC (`NIMBLECAS_SGEE_GRPC` only). Same two `connect` shapes; shares the port's `GrpcEndpointRing`. |
| `GrpcEndpointRing` | One open client per endpoint + atomic active index + rotation counter; shared (`shared_ptr`) by the port and channel of a run so both follow the same leader. |
| `SgeeGrpcTlsOptions` | `struct { string ca_cert_path, cert_path, key_path, target_name_override; }` — **all-or-nothing**: empty ⇒ plaintext, partial ⇒ rejected. |
| `SgeeGrpcExecutorOptions` | `struct { string endpoint; string auth_token; u64 rpc_deadline_ms; vector<string> endpoints; SgeeGrpcTlsOptions tls; }` — a non-empty `endpoints` supersedes `endpoint`. |
| `sgee_grpc_distributed_executor` | `auto sgee_grpc_distributed_executor(SgeeExecutorConfig, SgeeGrpcExecutorOptions) -> Result<unique_ptr<Executor>>` factory (cross-process; `wal_dir` ignored, `num_workers==0` valid). |
| `WorkerPumpConfig` | `struct { u64 worker_id; u64 lease_timeout_ms; u64 idle_backoff_ms; u64 heartbeat_every_ms; }` |
| `run_worker_pump` | Worker execution loop: lease → decode → registry lookup → invoke → put result → complete. |
| `SgeeExecutorConfig` | Config struct with fluent setters (`with_registry`, `with_num_workers`, `with_poll_interval_ms`, `with_max_result_recoveries`, etc.). |
| `SgeeDistributedExecutor` | `class SgeeDistributedExecutor : public Executor` — level-synchronous wavefront coordinator. |
| `sgee_distributed_executor` | `auto sgee_distributed_executor(SgeeExecutorConfig) -> Result<unique_ptr<Executor>>` factory. |

## Error model

| Condition | Error |
| :--- | :--- |
| Factory called when `NIMBLECAS_SGEE=OFF` | `MathError::not_implemented` |
| `sgee_grpc_distributed_executor` called when `NIMBLECAS_SGEE_GRPC=OFF` | `MathError::not_implemented` (valid config) / `domain_error` (invalid) |
| gRPC: `GetResult` returns `found=false` after `state()==completed` (node lost its store) | `MathError::distributed_error` (honest whole-run abort, never a fabricated value) |
| Cluster: result missing after a leadership change, `max_result_recoveries` exhausted | `MathError::distributed_error` (the bound exists so a lost result terminates rather than spins) |
| TLS requested but `SgeeGrpcTlsOptions` only partially filled | `MathError::domain_error` — never a silent downgrade to plaintext |
| TLS requested against a `libsgee_capi_grpc` predating the mTLS C-ABI entry points | `MathError::not_implemented` (weak symbols; no null call, no plaintext fallback) |
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
