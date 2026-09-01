# `nimblecas.memo_dist` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/memo_dist/memo_dist.cppm`

Content-addressed memoization for distributed task execution (ROADMAP §6.2 item 2,
"Distributed Hash-Consing"): a stable 128-bit `ContentKey` fingerprint over canonical bytes,
a `DistributedMemo` interface, and a thread-safe bounded `InProcessMemo` implementation.

```cpp
import nimblecas.memo_dist;
```

Depends on [`core`](core.md) and [`taskdag`](taskdag.md). Consumed by
[`taskdag_sgee`](taskdag_sgee.md).

## Honesty boundary — an exact table, not a probabilistic one

- **A hit is observationally identical to a miss.** Every hit is confirmed by a
  **byte-for-byte comparison of the full key**, not by the fingerprint. This is the module's
  reason for existing in this form, and it is not negotiable (Code Policy Rule 32).
- **`ContentKey` is a fingerprint, not a proof.** It is two 64-bit FNV-1a lanes with distinct
  offset bases, the second folding each byte XORed with `0xA5`. Two lanes sharing one prime
  are **not** an independent 128-bit function, and FNV is not cryptographic. No claim of
  collision resistance is made or relied upon. A collision costs a **lost cache entry**,
  never a wrong answer, and is counted in `MemoStats::key_mismatches`.
- **It is stable where `std::hash` is not.** The module deliberately does **not** key on
  `Expr::structural_hash()`. That value is derived from `std::hash`, which is
  implementation-defined and therefore differs across standard libraries, library versions,
  and `size_t` widths. A fingerprint that changes when the process changes cannot address
  content across a cluster. `ContentKey` is defined by arithmetic on byte values alone, so
  the same bytes yield the same key on every machine — pinned by golden vectors in
  `tests/memo_dist_tests.cpp`.
- **Only deterministic results are cacheable.** `is_memoizable_status` admits `ok` (0) and
  `math_error` (1), which are functions of the input, and refuses `bridge_error` (2) and every
  unknown status. A transport failure describes the cluster at one moment, not the input;
  caching it would make one transient fault permanent for every later lookup of those bytes.
- **Bounded, and it does not evict.** When `max_entries` or `max_value_bytes` is reached,
  `publish` **succeeds** and counts a `rejected` — refusing to cache is not an error, the
  caller simply recomputes. Nothing is evicted, so a run's behaviour does not depend on what
  some earlier run happened to insert. The entry bound is advisory under concurrency: two
  threads on different shards may both observe headroom and both insert. It caps growth; it
  is not a hard invariant.

## Why the key is the encoded task, not the expression

`sgee_bridge::encode_task` already emits a canonical byte string per task — fixed-width
little-endian fields, no padding, no ordering ambiguity — carrying the registry fingerprint,
the op id, and every argument. Because a `TaskFn` is contractually pure, two tasks are the same
computation **if and only if** those bytes are equal, *given the same implementations behind
those op ids*. That proviso is the important one.

### ⚠ The registry fingerprint does NOT pin the implementation

`TaskRegistry::fingerprint()` is FNV-1a over the **sorted op-ID strings**. It hashes the
*names*, not the *bodies*. So:

- Adding or removing an unrelated op moves the fingerprint and retires **every** entry —
  harmless over-invalidation.
- **Editing an op's implementation without changing its id moves nothing at all.** Every
  existing entry stays live and keeps serving the old implementation's results.

For a per-call cache that gap is invisible, because the table dies with the call. A memo that
outlives the run — which is the entire point of this module — turns it into a wrong answer
that persists. The operational rule is therefore not optional:

> **Any semantic change to a registered op REQUIRES bumping its version (`<domain>.<op>/vN`)
> or discarding the memo.** The `/vN` grammar is enforced syntactically by `validate_op_id`;
> nothing enforces that you bump it when you should.

This is a property of the registry, not of this module, and it is stated here because this
module is what makes it dangerous.

## API

| Name | Purpose |
| :--- | :--- |
| `struct ContentKey { std::uint64_t hi, lo; }` | 128-bit fingerprint; ordered and equality-comparable |
| `content_key(std::span<const std::byte>) -> ContentKey` | `noexcept`, allocation-free, byte-value-only |
| `to_hex(const ContentKey&) -> std::string` | 32 lowercase hex chars, `hi` then `lo` |
| `struct MemoStats` | `hits`, `misses`, `publishes`, `key_mismatches`, `rejected` |
| `class DistributedMemo` | Abstract: `name()`, `lookup()`, `publish()`, `stats()` |
| `class InProcessMemo` | Sharded, mutex-guarded, bounded, no eviction |
| `class FileMemo` | Append-only log on disk plus an in-memory index; survives the process |
| `is_memoizable_status(int) -> bool` | True for `0` and `1` only |

`key_mismatches` counts only a **genuine** collision — a non-empty bucket whose occupants all
disagree with the supplied full key. A mismatch is also counted as a miss, so
`hits + misses` equals the number of lookups and a measured hit rate has an honest denominator.

## Error model

| Situation | Result | Rationale |
| :--- | :--- | :--- |
| `lookup` of an absent key | `Result` **ok**, value `nullopt` | A miss is not an error |
| `lookup` where the fingerprint matches but the bytes differ | `Result` **ok**, value `nullopt`, `key_mismatches++` | The exactness rule; a collision costs an entry, never an answer |
| `publish` of a value over `max_value_bytes` | `Result` **ok**, `rejected++` | Refusing to cache is not an error — the caller recomputes |
| `publish` when `max_entries` is reached | `Result` **ok**, `rejected++` | Bounded, no eviction, so behaviour stays reproducible |
| `publish` of a key already held | `Result` **ok**, `publishes++` | First writer wins; pure ops must agree anyway |
| `InProcessMemo` | never returns an error | Every refusal above is a success with a counter |

A `DistributedMemo` implementation *may* return an error — a network-backed one will. **The
executor treats that as a miss**, along with an entry that will not decode: turning on a cache
must never turn a run that would have succeeded into one that fails. See the error model in
[`taskdag_sgee`](taskdag_sgee.md) for the run-level aborts, none of which the memo adds to.

`stats()` is a snapshot of five independently-read counters, so a caller reading it *while*
other threads use the table may see `hits + misses` momentarily disagree with the number of
lookups. The identity holds once traffic stops.

`is_memoizable_status` takes a plain `int` rather than `ResultEnvelope::Status` so that this
module need not import `taskdag_sgee`. The dependency runs the other way, and keeping the
import graph acyclic is why `taskdag_sched::to_placement` likewise returns a raw `uint8_t`.

## Worked example

```cpp
import nimblecas.memo_dist;
using namespace nimblecas;

InProcessMemo memo;                       // 32 shards, 100k entries, 16 MiB per value

const Payload key   = /* canonical encoded task bytes */;
const Payload value = /* encoded ResultEnvelope      */;

const ContentKey k = content_key(key);

auto hit = memo.lookup(k, key);           // Result<std::optional<Payload>>
if (hit.has_value() && hit->has_value()) {
    use(**hit);                           // verified against the full key, byte for byte
} else {
    const sgee_bridge::ResultEnvelope computed = run_the_task();
    // Publish only what is a function of the key, and only if it encodes.
    if (is_memoizable_status(static_cast<int>(computed.status))) {
        if (const auto enc = sgee_bridge::encode_result(computed); enc.has_value()) {
            (void)memo.publish(k, key, *enc);   // a refusal to cache is not an error
        }
    }
}

const MemoStats s = memo.stats();
// s.hits + s.misses == number of lookups; s.key_mismatches counts real collisions only.
```

## Use from the distributed executor

`SgeeExecutorConfig` exposes two switches, **both default off**, so an executor configured
as before behaves exactly as before:

```cpp
InProcessMemo memo;
auto cfg = SgeeExecutorConfig{}
               .with_registry(reg)
               .with_dedup_identical_tasks(true)   // collapse duplicates within one run
               .with_memo(memo);                   // and remember results across runs
```

`dedup_identical_tasks` needs no table: it collapses tasks byte-identical to one already
dispatched in this run. `memo` is consulted before dispatch and published to after every
collected result, so it memoizes across runs and, given a shared implementation, across the
cluster.

**What changes and what does not.** `outputs` stay bit-for-bit identical either way, failed
tasks included. Two things do change, by design: `executed` counts tasks actually dispatched,
which is precisely what these options reduce, and `measured_seconds` is **0.0** for every hit
and every collapsed duplicate — nothing ran, and reporting a borrowed duration would put
fictional work into a timing field.

**A bad entry cannot fail a run.** A memo value that will not decode, or that carries a status
never published, is treated as a **miss**: the task is dispatched and computed for real. A miss
is by construction observationally identical to a hit, so this is exactly as honest as aborting
and strictly more available — otherwise one malformed entry in a shared table would permanently
fail every future run of every graph containing that task, and there is no purge API. A memo
implementation that reports its own failure from `lookup` or `publish` **does** abort the run:
that is a component saying it is broken, not a bad row.

**Memory.** With `dedup_identical_tasks` on, the run index retains the full encoded payload of
every distinct dispatched task for the **whole run**, and those payloads embed complete parent
outputs. On graphs with large intermediates that is a real cost; without dedup the same bytes
were retained only per level.

See [`taskdag_sgee`](taskdag_sgee.md) for the coordinator-side wiring.

## `FileMemo` — the same table, made durable

`InProcessMemo` dies with the process, so a coordinator restart re-executes everything it had
already computed. `FileMemo` outlives it:

```cpp
auto memo = FileMemo::create("results.memo");        // Result<std::unique_ptr<FileMemo>>
if (!memo.has_value()) { /* honest domain_error: bad path, unwritable, or I/O error mid-log */ }
cfg.with_memo(**memo);
```

An **append-only** log with an in-memory index. Append-only suits the data: a memo entry is
immutable by construction, because the value is a pure function of the key, so nothing ever
needs rewriting. `create()` is a factory rather than a throwing constructor because a missing
directory, an unwritable file, or an I/O error partway through the log are ordinary runtime
conditions.

Framing is little-endian throughout, for the same reason `ContentKey` is defined by byte
arithmetic: a store written on one machine must be readable on another.

```
u32 magic | u16 version | u16 reserved | u64 fp_hi | u64 fp_lo
u32 key_len | u32 value_len | key bytes | value bytes | u64 checksum
```

The first four bytes of the file are literally `N C M R` — asserted by a test, because a format
spec that misdescribes its own bytes is useless to anyone writing an independent reader. The
checksum covers everything before it, so a record boundary cannot be shifted nor the key/value
split moved without failing it.

**The `ContentKey` is stored, not recomputed.** `publish` indexes under the key the *caller*
supplied, and the interface nowhere requires that to equal `content_key(full_key)`. Recomputing
it at load would silently re-key such an entry and make it unreachable — so `FileMemo` would
honour a different contract from `InProcessMemo` across exactly one event, a restart.

### Honesty under a fault model the in-memory table never faces

- **A hit is still confirmed by a full-key byte comparison.** The checksum proves a record is
  *intact*; only the byte comparison proves it is *this key*.
- **A damaged record is a miss.** Truncation is caught by the length bound, in-place damage by
  the checksum. A damaged store costs work; it cannot produce a wrong value.
- **Loading stops at the first unreadable record**, and the log is then **truncated** to the end
  of the last good one. A record *behind* a bad one is deliberately not served, even if intact —
  scanning past a record you cannot verify is how a corrupt log starts yielding plausible
  garbage. Truncating matters too: a value's own bytes can contain something that parses as a
  record, and after a short repair those bytes would land on a record boundary.
- **A genuine I/O error while loading is not treated as end-of-log.** `create()` fails instead.
  A short read mistaken for EOF would truncate a healthy log at the point of the error and
  destroy durable records.
- `log_was_damaged()` reports whether a damaged tail was found and cut. A **boolean, not a
  count** — loading stops at the first bad record, so how many were lost beyond it is precisely
  what cannot be known.

### Durability is bounded, and the bound is stated rather than oversold

Every publish writes and flushes, so records survive the **process** dying — a crash, a kill,
an exit. They are **not** proof against **power loss**: reaching stable storage requires
`fsync`/`fdatasync`/`FlushFileBuffers`, and `std::fstream` exposes no descriptor to call it on.

There is deliberately **no "sync" option**. Through this API it could only close and reopen the
stream, which pushes the library's buffers into the OS page cache — exactly what `flush()`
already did — and would charge two syscalls per publish for no added guarantee. Power-loss
durability needs the native file API and is **not implemented**. Saying so is better than
shipping a flag whose name promises it.

### One writer per path, and no bound

Nothing locks the file. Two `FileMemo` objects over one path each keep their own append offset
and overwrite each other's records. A reader rejects the wreckage rather than serving it, so
this cannot produce a wrong *answer* — but it can destroy a store. **One process, one `FileMemo`
per path.** Many threads through a single `FileMemo` are fine; it is internally locked.

Unlike `InProcessMemo` there is **no `max_entries` analogue and no eviction**: the log grows
until the disk refuses it, after which publishes are refused and the store keeps serving what it
holds. `InProcessMemo`'s entry bound exists to keep behaviour reproducible; here the bound is the
filesystem.

### Error model

| Situation | Result |
| :--- | :--- |
| `create` on a bad path, unwritable file, or **I/O error mid-log** | `MathError::domain_error` |
| `create` on a log with a damaged tail | **succeeds**, tail truncated, `log_was_damaged()` true |
| `lookup` of a damaged or absent record | **ok**, `nullopt` — a miss |
| `publish` of an oversize value | **ok**, `rejected++` |
| `publish` when the write fails (disk full, EIO) | **ok**, `rejected++` — the cache degrades, the run does not fail |

The last row is worth reading twice: a dying disk is observationally identical to routine
oversize refusal. Both raise `rejected`, and neither fails the caller's run — which is the right
trade for a cache, but means `rejected` alone cannot distinguish a full disk from a working store.

### What durability costs

Measured, not assumed. `tools/memo_dist_bench --file-memo <path>` runs the all-hits arm against
a `FileMemo` instead of the in-memory table; everything else is identical, and a fresh store is
written per cell so no cell inherits another's warmth.

| task ms | in-memory arm C (s) | file-backed arm C (s) |
| ---: | ---: | ---: |
| 0.1 | 0.001083 | 0.001088 |
| 1 | 0.001085 | 0.001091 |
| 10 | 0.001112 | 0.001116 |

64 tasks, duplicate ratio 0.5, 7 repetitions, medians. The two are **within 0.5% and well inside
the run-to-run spread** — on this workload durability is not measurably slower.

Read that narrowly. It measures a **warm** store of 32 records served by the OS page cache, on
one machine. It says nothing about a store large enough to miss the page cache, or about the
write path at scale.

### What this closes

ROADMAP §6 recorded that "the result store remains in-memory and per-node … recovery is a
deterministic re-execution rather than durability." With a `FileMemo` attached, a distributed run
repeated after the coordinator restarts dispatches **nothing** and returns outputs bit-identical
to a plain run — asserted in `tests/taskdag_sgee_memo_tests.cpp`. The harness there is the
in-process `FakeBrokerPort`, and the "restart" is destroying and recreating the memo over the
same file; no test drives a `FileMemo` through the gRPC/Raft path.

## Measured

The benefit was measured, not assumed — see
[memo-dist-bench.md](../technical/memo-dist-bench.md) for the harness, the raw 567-row data,
and the caveats. The short version: **how many tasks are removed is exact**, but the fraction
of that saving which becomes wall-clock time depends on **task cost alone**, climbing from
~35% of the theoretical maximum at 0.05 ms per task to ~99% at 50 ms, and is essentially
indifferent to how many duplicates the graph contains. A memoized run that dispatches nothing
still costs a flat **17.1 µs per task** of coordinator work, which is the floor no table can
remove. The verdict was POSITIVE; both switches nonetheless default off.
