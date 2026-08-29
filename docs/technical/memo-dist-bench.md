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
| Machine | `oluwasanmi-tradingbot-server`, 32 cores, load average 2.37 at start |
| Toolchain | `clang++-22` (22.1.8) + libc++, Release, `-O3 -march=x86-64-v3` |
| Graph | 64 tasks in one level, seeded generator (seed 1337), duplicate ratio ∈ {0.25, 0.50, 0.75} |
| Task cost | ∈ {0.05, 0.1, 0.5, 1, 5, 10, 50} ms of calibrated spin |
| Workers | 4 |
| Repetitions | 9 per arm per cell, arms **alternated within** each repetition |
| Calibration | 1.65 × 10⁸ spin rounds/second |

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
| 0.05 | 0.25 | 0.00341 | 0.00309 | 9.4 | 25 | **37.4** | 0.00108 | 68.1 |
| 0.05 | 0.50 | 0.00336 | 0.00276 | 17.9 | 50 | **35.8** | 0.00108 | 67.8 |
| 0.05 | 0.75 | 0.00334 | 0.00247 | 26.2 | 75 | **35.0** | 0.00109 | 67.5 |
| 0.1 | 0.25 | 0.00421 | 0.00370 | 12.2 | 25 | **48.9** | 0.00110 | 73.9 |
| 0.1 | 0.50 | 0.00416 | 0.00319 | 23.5 | 50 | **46.9** | 0.00109 | 73.9 |
| 0.1 | 0.75 | 0.00414 | 0.00268 | 35.3 | 75 | **47.1** | 0.00109 | 73.8 |
| 0.5 | 0.25 | 0.01154 | 0.00944 | 18.2 | 25 | **72.7** | 0.00109 | 90.6 |
| 0.5 | 0.50 | 0.01152 | 0.00736 | 36.1 | 50 | **72.2** | 0.00109 | 90.5 |
| 0.5 | 0.75 | 0.01153 | 0.00530 | 54.1 | 75 | **72.1** | 0.00109 | 90.6 |
| 1 | 0.25 | 0.01955 | 0.01549 | 20.8 | 25 | **83.0** | 0.00109 | 94.4 |
| 1 | 0.50 | 0.01956 | 0.01138 | 41.8 | 50 | **83.7** | 0.00109 | 94.4 |
| 1 | 0.75 | 0.01955 | 0.00732 | 62.5 | 75 | **83.4** | 0.00109 | 94.4 |
| 5 | 0.25 | 0.08782 | 0.06639 | 24.4 | 25 | **97.6** | 0.00110 | 98.8 |
| 5 | 0.50 | 0.08269 | 0.04353 | 47.4 | 50 | **94.7** | 0.00109 | 98.7 |
| 5 | 0.75 | 0.08479 | 0.02360 | 72.2 | 75 | **96.2** | 0.00109 | 98.7 |
| 10 | 0.25 | 0.16831 | 0.12951 | 23.1 | 25 | **92.2** | 0.00113 | 99.3 |
| 10 | 0.50 | 0.17237 | 0.08789 | 49.0 | 50 | **98.0** | 0.00113 | 99.3 |
| 10 | 0.75 | 0.17329 | 0.04599 | 73.5 | 75 | **97.9** | 0.00111 | 99.4 |
| 50 | 0.25 | 0.81401 | 0.61390 | 24.6 | 25 | **98.3** | 0.00114 | 99.9 |
| 50 | 0.50 | 0.82487 | 0.41217 | 50.0 | 50 | **100.1** | 0.00115 | 99.9 |
| 50 | 0.75 | 0.85372 | 0.21582 | 74.7 | 75 | **99.6** | 0.00114 | 99.9 |

One cell reads `eff% = 100.1`. That is noise, not a result: a saving cannot exceed the ideal,
and the excess is well inside the run-to-run spread. It is left as measured rather than
rounded down, because silently tidying a number is how a table stops being data.

**No cell is negative.** Every arm B and arm C median beats its arm A median, and the
min/max spreads in the raw output do not overlap between arms in any cell except the two
cheapest task sizes.

### The finding: efficiency tracks task cost, and is indifferent to duplicate ratio

Read the `eff%` column down, not across. At a fixed task cost the three duplicate ratios
agree closely — 37.4 / 35.8 / 35.0 at 0.05 ms, 97.6 / 94.7 / 96.2 at 5 ms — while across task
costs the value climbs from ~35% to ~99%.

So **how much of the theoretical saving you actually get is a property of your task size, not
of how much duplication you have.** Duplication decides the size of the prize; task cost
decides what fraction of it you collect. Practically: above roughly 5 ms per task, dedup
collects essentially all of the available saving; below about 0.1 ms, it collects half or
less, and most of what it removes is given straight back to coordination.

### The floor that memoization cannot remove

Arm C dispatches **nothing** — every task is a memo hit — so all of arm C is coordinator
work. Its median is **1.092 ms** (min 1.059, max 1.172) and it is **flat across all 21
cells**, including the ones where a task costs 50 ms.

That constant is the useful number: at 64 tasks it is **17.1 µs per task** of coordinator
cost — encoding the envelope, fingerprinting it, the table lookup, and the bookkeeping — that
no amount of memoization removes. It is also why the cheap-task cells do poorly: at 0.05 ms
per task the coordinator's own 17 µs is a third of the task itself.

## This is a re-measurement, and why

An earlier sweep was run before an adversarial review found that `content_key` was being
computed on **every** task of **every** run, including when both switches are off, and the
value discarded. Arm A — the plain baseline — was therefore paying for a hash it never used,
which flattered arms B and C. That was fixed, and this whole sweep re-run against the code
that actually ships. The numbers here supersede the earlier ones.

The correction moved efficiency by a few points in both directions, mostly **down** on cheap
tasks (0.05 ms: 40.1 → 37.4; 0.1 ms: 51.4 → 48.9; 0.5 ms: 80.5 → 72.7), which is the expected
direction once the baseline stops doing unnecessary work. The conclusion is unchanged and the
coordinator floor is identical to three significant figures.

One caveat on comparing the two sweeps: the machine was busier for this one (load 2.37 versus
1.26), and arm A's absolute times are correspondingly higher in the expensive cells. That is
precisely why the arms alternate **within** each repetition — comparisons *inside* a sweep are
sound, comparisons *between* sweeps are not, and only the former is used to draw a conclusion.

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
