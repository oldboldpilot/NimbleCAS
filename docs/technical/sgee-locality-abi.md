# Data-locality scheduling: what it needs from the SGEE C ABI

**Author:** Olumuyiwa Oluwasanmi

ROADMAP §6.2 item 3 asks that "tasks are scheduled on nodes that already hold parent datasets
(e.g. a downstream matrix calculation is scheduled on the specific GPU holding the parent
matrix) to minimize inter-node TCP/IP or PCIe transfers."

**That cannot be built inside NimbleCAS.** This page records exactly why, exactly what the
`StochasticGraphExecutionEngine` C ABI would have to add, and what NimbleCAS would then do with
it — so the work can be picked up without re-deriving any of it.

## The half that exists

The coordinator side is already in place and needs nothing new:

| Piece | Where | State |
| :--- | :--- | :--- |
| Per-task placement hook | `SgeeExecutorConfig::placement` | Consulted at **both** enqueue sites (initial dispatch and result-recovery re-enqueue) |
| Affinity categories | `taskdag_sched::Affinity`, `AffinityTable` | Present |
| Affinity → placement code | `taskdag_sched::to_placement` | Present |
| The connector | `taskdag_sgee::affinity_placement` | **Added in M8** — builds the placement function from an `AffinityTable` |
| ABI transport for the label | `sgee_broker_enqueue(..., int placement, ...)` → `sgee_broker_lease(..., int* out_placement, ...)` | Carries one placement value per task, end to end |

So a task can be labelled, and the label reaches the worker. What it cannot do is *change who
gets the task*.

## Blocker 1 — lease has no filter, so a worker cannot ask for its own work

`sgee_broker_lease` returns the **earliest pending task, unconditionally**:

```c
sgee_error_t sgee_broker_lease(sgee_task_broker_t* broker,
                               uint64_t worker_id, uint64_t timeout_ms,
                               uint64_t* out_task_id, uint64_t* out_token,
                               int* out_placement, uint32_t* out_attempt,
                               void** out_payload, size_t* out_payload_len);
```

There is no filter parameter. `out_placement` is *reported* after the choice is made, never used
to make it — and NimbleCAS's `CapiBrokerPort::lease` discards it entirely. A GPU-only task is
therefore leased by whichever worker asks first.

**Lease-then-hand-back is not a workaround**, for two structural reasons in the queue itself:

1. `lease()` has no skip, and `mark_requeued` re-inserts a task with its **original**
   `enqueue_time_ms` while the pending set orders by `(enqueue_time_ms, id)`. A released task
   goes straight back to the head, so the worker that cannot run it meets it again next poll
   and never reaches its own work.
2. `lease` is the **only** place `attempt` is incremented. Each hand-back spends one of the
   task's lives having executed nothing, so an idle worker polling a queue whose head belongs
   to a busy peer will dead-letter a perfectly good task within seconds.

Filtering has to happen **at the choice**, on the leader, before the task is picked.

### The good news: the core already does this

SGEE commit `28537cc5` ("Partition the lease by a payload filter, chosen on the leader") added
it everywhere **except** the C ABI:

| Layer | Status |
| :--- | :--- |
| `proto/task_queue.proto` — `bytes payload_filter = 3` in `LeaseRequest` | **done** |
| `TaskBroker::lease` / `InMemoryIndex::earliest_pending_matching` | **done** |
| `ReplicatedTaskBroker::lease` / `earliest_leasable_excluding` | **done** |
| `ReplicatedQueueRuntime` + driver | **done** |
| gRPC client `TaskQueueClient::lease` | **done** |
| gRPC server `TaskQueueService::Lease` (both branches) | **done** |
| Raft-replicated command shape | **unchanged, deliberately** — the proposal still names one task id, so an old node simply ignores the new proto field |
| **C ABI** | **done** — SGEE `main` @ `6ec13bfc` |

Semantics, confirmed in the code rather than the commit message: the filter selects the
earliest pending task whose payload **contains** those bytes — a raw unanchored byte-substring
search over the whole payload — applied on the leader before the choice. An **empty** filter
short-circuits to the existing head-of-line behaviour, bit-identical, which is what every
current caller sends. No match returns `QueueEmpty`.

### The change — **landed** (SGEE `main` @ `6ec13bfc`)

Additive only. Three new entry points beside the existing ones — never extra parameters on the
current symbols, which the Python `ctypes` layer and the nanobind bindings both call:

```c
sgee_error_t sgee_broker_lease_filtered(
    /* ...identical to sgee_broker_lease... */,
    const void* payload_filter, size_t payload_filter_len);   /* NULL iff len == 0 */

sgee_error_t sgee_replicated_broker_lease_filtered(/* ...same pattern... */);
sgee_error_t sgee_grpc_lease_filtered(/* ...same pattern... */);
```

Each shim already calls a function that accepts the filter as a defaulted parameter
(`TaskBroker::lease`, `ReplicatedTaskBroker::lease`, `TaskQueueClient::lease`), so the
substantive delta per entry point is **one argument on one call**. Factor each existing lease
body into a static helper taking a span; the old symbol passes `{}`.

Estimated surface: **~130–160 lines across 4 files**, all additive. **Actual: +250/−29 across
those 4 files plus `tests/CMakeLists.txt`**, and the three existing symbols are byte-for-byte
unchanged in behaviour — each now forwards to the shared helper with an empty span.

The caveat above was written down in the headers at the same time: a filtered lease returns
`SGEE_ERR_QUEUE_EMPTY` to mean *"nothing for you"*, not *"the queue is empty"*. Writing it up
also turned up a header claim that did not match the code — `sgee_replicated_broker_lease` was
documented as returning `SGEE_ERR_NOT_LEASED` when nothing is pending, but it maps `NotLeased →
SGEE_ERR_QUEUE_EMPTY`. Corrected in the same commit.

**How it is verified.** `tests/task_queue/capi_lease_filter_test.cpp` drives the new entry point
through the pure C surface only — it includes `sgee_capi.h`, imports no module, and compiles
with no module flags at all, which is the shim's actual contract. Mutating
`sgee_broker_lease_filtered` to ignore its filter — the shape lease-then-hand-back degenerates
to — fails 7 of its 25 checks, two of them the structural ones this page argues from: the
unmatched lease left the head `LEASED` rather than `PENDING`, and spent one of its attempts
having executed nothing. SGEE's suite is 78/78 with the test included.

## Blocker 2 — nothing reports which worker produced a result

This one is not addressed by commit `28537cc5`, and it is the reason a filtered lease alone
still does not give **data** locality.

`worker_id` appears in the C ABI **only as an input to lease**. No function returns it. The
coordinator collects a result by queue id from the result channel and learns nothing about
where it was computed. So when it comes to enqueue the child, it has no idea which node holds
the parent's data, and no filter value it could sensibly attach.

The minimal addition is an out-parameter on the completion/state side — for example
`uint64_t* out_completed_by` on `sgee_broker_task_state`, or a producer id alongside the
out-of-band result (`sgee_grpc_get_result` already returns bytes and would need one more
scalar). Either way the queue must retain the leasing worker's id past completion, which it
holds during the lease but does not currently surface.

**Until both exist, "data locality" in this codebase can only mean affinity labelling.** That
distinction is why `affinity_placement` carries an explicit honesty note saying that turning it
on changes nothing today.

## What NimbleCAS would do once both land

1. ~~Extend `CapiBrokerPort` / `GrpcBrokerPort` to call the `_filtered` lease, with each worker
   passing a routing tag it is willing to serve.~~ **DONE.** `BrokerPort::lease_filtered` is the
   pure virtual; `lease` is now a non-virtual convenience that forwards with an empty span, so
   every existing caller is unchanged by construction. `CapiBrokerPort` and `GrpcBrokerPort`
   route an empty filter to the original symbol and a non-empty one to
   `sgee_broker_lease_filtered` / `sgee_grpc_lease_filtered`, both declared **weak**: linked
   against an older `.so` a filtered lease returns `not_implemented` rather than silently
   degrading to an unfiltered one — the fallback that would hand back exactly the wrong task.
   `FakeBrokerPort` implements the same predicate (`payload_matches_filter`) so the double and
   the real broker cannot drift, and `WorkerPumpConfig::payload_filter` carries a pump's tag.

   One degradation this canNOT catch, recorded because it is invisible: a *current* client shim
   against an *old gRPC server*. proto3 makes `payload_filter` a field an old node simply
   ignores, so the request succeeds and returns an unfiltered task. There is no version
   handshake. A mixed-version cluster is the one configuration where a filtered lease can still
   hand back work the filter excluded.
2. Put that tag in the task payload where a substring match can find it. **This is now the
   blocking step.** The `TaskEnvelope`
   wire format v1 has a **reserved `u16`** at offset 6, but two bytes are too few and too
   collidable for a substring search over arbitrary argument bytes; a v2 framing with an
   explicit, length-prefixed, magic-delimited routing field is the honest option. Note that the
   envelope bytes are also the memo key (see [`memo_dist`](../reference/memo_dist.md)), so a
   framing change must either keep the routing field **out** of the memo key or accept that
   changing a task's routing retires its cache entries.
3. Record the producing worker per completed task, and derive each child's tag from the
   producer of its heaviest parent.
4. Measure it the way M6 and M7 were measured, with a pre-registered rule and a negative
   control — locality wins are highly sensitive to payload size and network cost, and it is
   entirely possible the answer is NEGATIVE on a single machine.

## Status

- **Blocker 1**: **CLEARED.** The filter reached SGEE's core in `28537cc5` and its C ABI in
  `6ec13bfc`; `sgee_broker_lease_filtered`, `sgee_replicated_broker_lease_filtered` and
  `sgee_grpc_lease_filtered` all export unmangled from the built shims.
- **Blocker 2**: **still open, and still not designed anywhere.** Nothing in the C ABI reports
  which worker produced a result, so the coordinator cannot derive a child's routing tag from
  its parent's location.
- **NimbleCAS**: **consumes the filtered lease as of this commit** — step 1 above is done, with
  `payload_matches_filter`, `BrokerPort::lease_filtered`, both real ports, `FakeBrokerPort`, and
  `WorkerPumpConfig::payload_filter`. Verified by 7 tests (48 checks) in
  `tests/taskdag_sgee_tests.cpp`, which run in the DEFAULT build because they drive
  `FakeBrokerPort`; the SGEE=ON configuration passes 37/37 against the real `CapiBrokerPort`
  linked to SGEE `6ec13bfc`, and the full suite is 176/176.

  Those tests were mutation-checked rather than trusted. Making the predicate ignore its filter
  kills 6 of the 7; re-implementing the skip as **lease-then-hand-back** — spend an attempt, mark
  it leased — kills 5, including the end-to-end pump test. The one survivor of the first mutant
  is the empty-filter equivalence test, which is correct: "always match" *is* the right answer
  for an empty filter.

- **Still inert end to end, for a NARROWER reason.** Nothing writes a routing tag into a payload,
  so every caller still passes an empty filter and behaviour is unchanged. `SgeePlacement` rides
  its own `int` parameter, a channel the substring search cannot see. The blocking piece has
  moved from upstream into this repo: it is step 2's framing decision, which is not free because
  the envelope bytes are also the memo key.

**What clearing blocker 1 does and does not buy.** It buys *affinity* routing: a worker can
refuse to be handed work it cannot serve, and a GPU-only task can wait for a GPU worker instead
of being leased by whoever polls first. It does **not** buy *data* locality — that needs blocker
2. `affinity_placement`'s honesty note has been updated rather than removed: it still says the
mechanism changes nothing today, because it still does.
