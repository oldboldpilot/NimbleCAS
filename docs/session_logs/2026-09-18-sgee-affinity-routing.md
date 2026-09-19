# Session log — Closing affinity routing: the TaskEnvelope routing trailer

**Date:** 2026-09-18
**Author:** Olumuyiwa Oluwasanmi
**Build/verify host:** `DESKTOP-3SON9JT` (WSL Ubuntu, clang++-23 + libc++)

## Scope

ROADMAP §6.2 item 3, **step 2** of [sgee-locality-abi.md](../technical/sgee-locality-abi.md):
put a routing tag where the leader's lease filter can see it. Step 1 (consuming SGEE's
payload-filtered lease) landed in `afab6c1`; this session closes the chain
*affinity table → routing tag → payload → leader-side filter → pump*.

## The design decision, and why it cost nothing

The blocker recorded in the previous session was that the encoded `TaskEnvelope` bytes are also
the **memo key** ([`memo_dist`](../reference/memo_dist.md)), so adding a routing field looked like
a versioned framing change that would retire every existing cache entry.

It does not, because the tag is **not a member of `TaskEnvelope` at all**. A `TaskEnvelope` is a
task's *identity* — registry fingerprint, op id, argument bytes — and where a computation runs is
not part of what it computes. Keeping routing out of the identity struct gives three properties:

| Property | Mechanism |
| :--- | :--- |
| No cache entry retired | `encode_task` is untouched; an untagged task encodes byte-for-byte as before |
| Routing does not partition the cache | The executor keys memo + dedup on the **untagged** bytes, dispatches the **tagged** bytes |
| A worker sees the same task either way | `decode_task` of a tagged payload yields the identical envelope |

### Wire format

```
... v1 task bytes ... || k_routing_magic (u32 LE) || tag bytes || tag_len (u32 LE)
```

version at offset 4 stamped `2`. The length is written **last** so the trailer is locatable from
the end in constant time — that is what lets `decode_task` strip a suffix and then run the v1
parser completely unchanged, with no v1 offset moved and no v1 parsing rule relaxed to tolerate
trailing bytes.

## API added

- `sgee_bridge::routing_filter(tag)` — the exact trailer bytes. Both the dispatcher and the pump
  go through it, so needle and haystack cannot drift.
- `sgee_bridge::with_routing_tag` / `routing_tag_of` / `without_routing_tag` — fluent tag/untag;
  an empty tag is the identity, and re-tagging **replaces** rather than stacks.
- `placement_tag(SgeePlacement)` — frozen ASCII wire names.
- `affinity_routing_tag(table, fallback)` — built *on top of* `affinity_placement`, so the label
  and the tag cannot disagree.
- `SgeeExecutorConfig::routing_tag` (+ `with_routing_tag`), consulted at **both** enqueue sites.
- `FakeBrokerPort::enqueued_payloads()` — test-double observation hook.

## Verification

- **176/176** default `ctest`; **47/47** `taskdag_sgee_tests` and **15/15**
  `taskdag_sgee_memo_tests` with `NIMBLECAS_SGEE=ON` against the real `CapiBrokerPort`.
- `clang-tidy-23`: zero findings on the changed module and both changed test files. One
  pre-existing `readability-redundant-member-init` on `WorkerPumpConfig::payload_filter`
  (introduced in `afab6c1`) was fixed in passing.
- **11 new tests**, mutation-checked with **six mutants, all killed**:

| Mutant | Killed by |
| :--- | :--- |
| `with_routing_tag` drops the tag | 6 tests |
| Memo keyed on the **dispatched** bytes | **1** — `a_memo_entry_survives_being_routed_somewhere_else` |
| Re-tagging stacks instead of replacing | 1 |
| `placement_tag` collapses gpu→cpu | 2 |
| Routing trailer invisible to readers | 5 |
| `k_routing_magic` changed | **1** — the hand-pinned needle vector |

The two single-kill rows are the ones worth remembering. Both mutants initially **survived the
whole suite**; each is now caught by a test written specifically because it survived.

## Review findings (agy `gemini-3.1-pro-high`, adversarial)

Two real defects, both mine, both fixed:

1. **A self-referential test.** `the_routing_filter_matches_the_tagged_payload_and_only_it` built
   both the haystack (via `with_routing_tag`) and the needle (via `routing_filter`) from the same
   function, so a bug shared by both would pass. Fixed by pinning a hand-derived 11-byte trailer
   vector — and then proved to matter: changing `k_routing_magic`, an internally consistent and
   otherwise invisible wire-contract break, is caught by that assertion and nothing else.
2. **A comment describing the pre-rewrite field order** (`magic || len || tag`) after the layout
   moved the length last. Corrected.

Clean in: parsing safety (no wrap/underflow/OOB), memo-key independence, round-trips, and the
`enqueue_log_` concurrency addition.

## What is still NOT done

- **This is routing by *category* (`"cpu"`, `"gpu"`), not by *node*.** It is affinity routing, not
  data locality. Deriving a child's tag from the node that produced its heaviest parent needs
  **blocker 2** — nothing in the SGEE C ABI reports which worker produced a result — which remains
  open and undesigned.
- **Nothing is measured.** No locality or performance win is claimed anywhere. Step 4 of
  sgee-locality-abi.md still stands: measure with a pre-registered rule and a negative control. A
  NEGATIVE result on a single machine is entirely plausible.
- An accidental filter match remains **improbable, not impossible** — an argument whose bytes
  contain the trailer sequence matches. This is a mis-*routing* risk only, never a wrong answer,
  and is asserted as observed behaviour rather than left as a caveat.
