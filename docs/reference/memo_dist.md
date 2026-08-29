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
little-endian fields, no padding, no ordering ambiguity — carrying the registry fingerprint
(which pins the code version), the op id, and every argument. Because a `TaskFn` is
contractually pure, two tasks are the same computation **if and only if** those bytes are
equal. So the memo keys on those bytes and inherits version-safety for free: a registry
change moves the fingerprint, which moves every key, which retires the old entries.

## API

| Name | Purpose |
| :--- | :--- |
| `struct ContentKey { std::uint64_t hi, lo; }` | 128-bit fingerprint; ordered and equality-comparable |
| `content_key(std::span<const std::byte>) -> ContentKey` | `noexcept`, allocation-free, byte-value-only |
| `to_hex(const ContentKey&) -> std::string` | 32 lowercase hex chars, `hi` then `lo` |
| `struct MemoStats` | `hits`, `misses`, `publishes`, `key_mismatches`, `rejected` |
| `class DistributedMemo` | Abstract: `name()`, `lookup()`, `publish()`, `stats()` |
| `class InProcessMemo` | Sharded, mutex-guarded, bounded, no eviction |
| `is_memoizable_status(int) -> bool` | True for `0` and `1` only |

`key_mismatches` counts only a **genuine** collision — a non-empty bucket whose occupants all
disagree with the supplied full key. A mismatch is also counted as a miss, so
`hits + misses` equals the number of lookups and a measured hit rate has an honest denominator.

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
    const auto computed = run_the_task();
    // Publish only what is a function of the key.
    if (is_memoizable_status(static_cast<int>(computed.status))) {
        (void)memo.publish(k, key, encode_result(computed));
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
cluster. `outputs` stay bit-for-bit identical either way; `executed` does not, and must not —
it counts tasks actually dispatched, which is precisely what these options reduce.

See [`taskdag_sgee`](taskdag_sgee.md) for the coordinator-side wiring.

## Measured

The benefit was measured, not assumed — see
[memo-dist-bench.md](../technical/memo-dist-bench.md) for the harness, the raw 567-row data,
and the caveats. The short version: **how many tasks are removed is exact**, but the fraction
of that saving which becomes wall-clock time depends on **task cost alone**, climbing from
~36% of the theoretical maximum at 0.05 ms per task to ~99% at 50 ms, and is essentially
indifferent to how many duplicates the graph contains. A memoized run that dispatches nothing
still costs a flat **17.1 µs per task** of coordinator work, which is the floor no table can
remove. The verdict was POSITIVE; both switches nonetheless default off.
