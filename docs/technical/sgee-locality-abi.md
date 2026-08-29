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
| **C ABI** | **missing** |

Semantics, confirmed in the code rather than the commit message: the filter selects the
earliest pending task whose payload **contains** those bytes — a raw unanchored byte-substring
search over the whole payload — applied on the leader before the choice. An **empty** filter
short-circuits to the existing head-of-line behaviour, bit-identical, which is what every
current caller sends. No match returns `QueueEmpty`.

### The change

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

Estimated surface: **~130–160 lines across 4 files**, all additive.

One documentation caveat worth writing down at the same time: a filtered lease returns
`SGEE_ERR_QUEUE_EMPTY` to mean *"nothing for you"*, not *"the queue is empty"*. The replicated
shim's existing `NotLeased → QUEUE_EMPTY` mapping already conflates these.

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

1. Extend `CapiBrokerPort` / `GrpcBrokerPort` to call the `_filtered` lease, with each worker
   passing a routing tag it is willing to serve.
2. Put that tag in the task payload where a substring match can find it. The `TaskEnvelope`
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

- **Blocker 1**: unblocked upstream in SGEE's core; needs ~150 additive lines in its C ABI.
- **Blocker 2**: not yet designed anywhere; needs the queue to surface the leasing worker id.
- **NimbleCAS**: the coordinator-side half is complete as of M8 and is inert by design until
  the above lands.
