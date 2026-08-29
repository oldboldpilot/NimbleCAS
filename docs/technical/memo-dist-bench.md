# Distributed memoization — measured (ROADMAP §6.2 item 2 / M7)

**Author:** Olumuyiwa Oluwasanmi

Harness: `tools/memo_dist_bench.cpp`. Raw per-repetition data:
[`memo-dist-bench-results.csv`](memo-dist-bench-results.csv) (567 rows).
Full console output: [`memo-dist-bench-run.txt`](memo-dist-bench-run.txt).

## What was asked, and what could not be assumed

Memoization removes tasks from the cluster. **How many it removes is exact and
deterministic** — a duplicate ratio of 0.75 over 64 tasks leaves 16 to dispatch, every run,
on every machine. That number is a fact about the graph, not a measurement.

**Whether removing them saves wall-clock time is a different question**, and it has an
obvious way to come out badly: if a task is cheaper than the coordination needed to dispatch
it, then skipping it saves almost nothing, and the bookkeeping to decide to skip it costs
something. So task cost is the swept axis. A negative result was an acceptable outcome.

## Method

| | |
| :--- | :--- |
| Machine | `oluwasanmi-tradingbot-server`, 32 cores, load average 1.26 at start |
| Toolchain | `clang++-22` (22.1.8) + libc++, Release, `-O3 -march=x86-64-v3` |
| Graph | 64 tasks in one level, seeded generator (seed 1337), duplicate ratio ∈ {0.25, 0.50, 0.75} |
| Task cost | ∈ {0.05, 0.1, 0.5, 1, 5, 10, 50} ms of calibrated spin |
| Workers | 4 |
| Repetitions | 9 per arm per cell, arms **alternated within** each repetition |
| Calibration | 1.654 × 10⁸ spin rounds/second |

Three arms: **A** plain, **B** `dedup_identical_tasks`, **C** dedup plus a memo **pre-warmed**
by one prior run of the same graph, so C measures the all-hits case.

Arms are alternated inside each repetition (A,B,C,A,B,C…) rather than run in blocks, so a
machine that drifts during the session cannot present that drift as an effect. Every
repetition gates on arms B and C producing `outputs` **bit-identical** to arm A; the harness
exits 3 on any difference, because a speed number from a run that computed something else is
worthless. **No cell violated it.**

## Result — positive, and bounded by a measured constant

`save%` is the wall-clock reduction versus arm A. `ideal%` is the reduction if a removed task
were free — it equals the duplicate ratio. `eff%` is `save/ideal`: the fraction of the
theoretical saving actually realised.

| task ms | dup R | A median (s) | B median (s) | save % | ideal % | **eff %** | C median (s) | save % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.25 | 0.00336 | 0.00303 | 10.0 | 25.0 | **40.1** | 0.00109 | 67.5 |
| 0.05 | 0.50 | 0.00338 | 0.00276 | 18.3 | 50.0 | **36.6** | 0.00109 | 67.8 |
| 0.05 | 0.75 | 0.00341 | 0.00249 | 26.8 | 75.0 | **35.8** | 0.00109 | 67.9 |
| 0.10 | 0.25 | 0.00427 | 0.00372 | 12.8 | 25.0 | **51.4** | 0.00109 | 74.5 |
| 0.10 | 0.50 | 0.00424 | 0.00321 | 24.2 | 50.0 | **48.5** | 0.00109 | 74.2 |
| 0.10 | 0.75 | 0.00424 | 0.00270 | 36.3 | 75.0 | **48.5** | 0.00109 | 74.4 |
| 0.50 | 0.25 | 0.01114 | 0.00890 | 20.1 | 25.0 | **80.5** | 0.00109 | 90.2 |
| 0.50 | 0.50 | 0.01063 | 0.00618 | 41.9 | 50.0 | **83.7** | 0.00108 | 89.8 |
| 0.50 | 0.75 | 0.01009 | 0.00415 | 58.8 | 75.0 | **78.4** | 0.00109 | 89.2 |
| 1.00 | 0.25 | 0.01795 | 0.01459 | 18.7 | 25.0 | **75.0** | 0.00109 | 93.9 |
| 1.00 | 0.50 | 0.01782 | 0.01052 | 41.0 | 50.0 | **82.0** | 0.00109 | 93.9 |
| 1.00 | 0.75 | 0.01784 | 0.00605 | 66.1 | 75.0 | **88.2** | 0.00109 | 93.9 |
| 5.00 | 0.25 | 0.07858 | 0.05963 | 24.1 | 25.0 | **96.4** | 0.00109 | 98.6 |
| 5.00 | 0.50 | 0.07763 | 0.04073 | 47.5 | 50.0 | **95.1** | 0.00109 | 98.6 |
| 5.00 | 0.75 | 0.07851 | 0.02141 | 72.7 | 75.0 | **97.0** | 0.00111 | 98.6 |
| 10.0 | 0.25 | 0.14842 | 0.11650 | 21.5 | 25.0 | **86.0** | 0.00111 | 99.3 |
| 10.0 | 0.50 | 0.15003 | 0.07890 | 47.4 | 50.0 | **94.8** | 0.00111 | 99.3 |
| 10.0 | 0.75 | 0.15054 | 0.04073 | 72.9 | 75.0 | **97.3** | 0.00112 | 99.3 |
| 50.0 | 0.25 | 0.72124 | 0.54247 | 24.8 | 25.0 | **99.1** | 0.00114 | 99.8 |
| 50.0 | 0.50 | 0.72903 | 0.36602 | 49.8 | 50.0 | **99.6** | 0.00114 | 99.8 |
| 50.0 | 0.75 | 0.72248 | 0.18702 | 74.1 | 75.0 | **98.8** | 0.00113 | 99.8 |

**No cell is negative.** Every arm B and arm C median beats its arm A median, and the
min/max spreads in the raw output do not overlap between arms in any cell except the two
cheapest task sizes.

### The finding: efficiency tracks task cost, and is indifferent to duplicate ratio

Read the `eff%` column down, not across. At a fixed task cost the three duplicate ratios
agree closely — 40.1 / 36.6 / 35.8 at 0.05 ms, 96.4 / 95.1 / 97.0 at 5 ms — while across task
costs the value climbs monotonically from ~36% to ~99%.

So **how much of the theoretical saving you actually get is a property of your task size, not
of how much duplication you have.** Duplication decides the size of the prize; task cost
decides what fraction of it you collect. Practically: above roughly 5 ms per task, dedup
collects essentially all of the available saving; below about 0.1 ms, it collects half or
less, and most of what it removes is given straight back to coordination.

### The floor that memoization cannot remove

Arm C dispatches **nothing** — every task is a memo hit — so all of arm C is coordinator
work. Its median is **1.093 ms** (min 1.056, max 1.175) and it is **flat across all 21
cells**, including the ones where a task costs 50 ms.

That constant is the useful number: at 64 tasks it is **17.1 µs per task** of coordinator
cost — encoding the envelope, fingerprinting it, the table lookup, and the bookkeeping — that
no amount of memoization removes. It is also why the cheap-task cells do poorly: at 0.05 ms
per task the coordinator's own 17 µs is a third of the task itself.

## What this does NOT establish

- **A single machine, in-process broker.** These runs use the local broker port, not the
  gRPC Raft quorum. Over a real network the coordination constant is larger, which would move
  the crossover to the right — cheap tasks would do worse, not better.
- **A synthetic duplicate distribution.** Duplicates are generated by a seeded PRNG at a
  chosen ratio. Real symbolic workloads have their own structure, and this says nothing about
  what ratio a real `simplify` workload exhibits.
- **Arm C is the best case by construction.** It measures a fully pre-warmed memo. A cold memo
  pays a publish per task and hits nothing; the honest expectation for a first run is arm B,
  not arm C.
- **Nothing about symbolic hash-consing end-to-end.** Keying works on canonical task bytes.
  Putting `Expr` values into task payloads needs an exact `Expr` byte codec, which does not
  exist yet — the only exact round-trip in the repo today is `to_string`/`parse`, and that is
  lossy for any expression containing a `double`.

## Verdict

**POSITIVE, and shipped off by default.** The measurement supports turning `dedup_identical_tasks`
on when tasks cost more than roughly a millisecond and the graph is known to contain
duplicates. It does not support turning it on universally, and it is not on universally: both
switches default off, so an executor configured as before behaves as before.
