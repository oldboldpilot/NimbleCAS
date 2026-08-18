# Task-DAG Scheduling Benchmark Protocol & Pre-Registered Decision Register

**Author:** Olumuyiwa Oluwasanmi  
**Subsystem:** `nimblecas.taskdag_sched` (M6 milestone / ROADMAP §6.1)  
**Status:** Protocol implemented in `tools/taskdag_sched_bench.cpp`; measurements to be conducted on `mgpu` under `clang++-22`.

---

## 1. Overview & Methodology

This document registers the empirical benchmark protocol, measurement harness specification, and pre-registered decision rules for cost-aware task scheduling in NimbleCAS (Code Policy Rules 43, 58, 59: *no performance claim without evidence*).

The scheduling optimization under test is **Longest Processing Time First (LPT)** ordering of independent tasks within a wavefront level. Under a pull-based FIFO broker (`FakeBrokerPort` / SGEE coordinator), the coordinator chooses the enqueue order. LPT list scheduling provides a theoretical makespan bound $\le (4/3 - 1/(3m)) \times \text{OPT}$ (Graham 1969) versus $\le (2 - 1/m) \times \text{OPT}$ for arbitrary insertion order.

### Workload Definitions

All benchmark tasks execute the deterministic integer operation `nimblecas.bench.spin/v1`:
$$\text{iterate } x_{k+1} = (x_k \cdot a + c) \bmod p \quad \text{for } n \text{ rounds}$$
with $a = 6364136223846793005$, $c = 1442695040888963407$, $p = 2^{64} - 59$. The computation is CPU-bound, exact, allocation-free, and linear in $n$. The heavy task ($n = 32 \times n_{\text{unit}}$) is calibrated to take approximately 100–500 ms (target ~200 ms) to dwarf the 1–2 ms polling interval.

Four single-level graph shapes (33 tasks per level) are evaluated:

| Workload | Structure | Theoretical Purpose |
|:---|:---|:---|
| **SKEW-LAST** | 32 tasks of cost 1, 1 task of cost 32 (added **last**) | Adversarial FIFO baseline; maximum potential makespan reduction |
| **SKEW-PERM** | Same 33 tasks, 5 fixed seeded permutations of insertion order | Average-case skew across arbitrary submission orders |
| **UNIFORM** | 33 tasks of equal cost (cost 4) | **Negative control — must show $\sim 0\%$ delta** |
| **ZIPF** | 33 tasks with cost $\propto 1/\text{rank}$ ($r \in [1, 33]$) | Realistic workload decay; exploration of risk margin $\lambda \in \{0.0, 0.5, 1.0\}$ |

---

## 2. Statistical Protocol

1. **Warmups:** 3 full warmup iterations (baseline and ordered) are executed and discarded prior to data collection.
2. **Interleaved Execution:** Measurements alternate A/B/A/B (baseline, ordered, baseline, ordered) across $N \ge 30$ repetitions. This cancels out systemic thermal throttling and background load drift.
3. **Metrics Reported:**
   - **Median makespan ($M_{\text{base}}, M_{\text{ord}}$)**: robust central tendency.
   - **Interquartile Range (IQR)**: non-parametric spread.
   - **Median paired ratio ($M_r = \text{median}(T_{\text{ord}, i} / T_{\text{base}, i})$)**.
   - **Median reduction % ($ (1 - M_r) \times 100\% $)**.
   - **Two-sided Sign Test p-value**: exact binomial significance test over paired outcomes.

---

## 3. Pre-Registered Decision Rules

The decision rules are pre-registered prior to running the benchmark on the production hardware:

### POSITIVE Decision (Speedup Claim Approved)
- **SKEW-LAST** achieves median makespan reduction $\ge 10\%$ at $W = 4$ workers ($p < 0.01$).
- **SKEW-PERM** achieves median makespan reduction $\ge 5\%$ across permutations at $W = 4$ workers ($p < 0.01$).
- **UNIFORM** negative control shows $|\Delta| < 2\%$ across all worker counts.
- **Sanity Floor** ($W = 33 \ge \text{level size}$) holds: all policies tie within noise ($|\Delta| < 2\%$).

### NEGATIVE Decision (Feature Ships Off-by-Default or Removed)
- **SKEW-LAST** achieves median makespan reduction $< 5\%$ at $W = 4$, OR
- Sign-test significance fails ($p \ge 0.05$).
- *Consequence:* Ordering is documented as ineffective on the tested shapes, kept off-by-default, and the negative result is honestly recorded in `docs/reference/taskdag_sched.md`.

### INVALID Session Failure Modes (Benchmark Self-Check)
1. **Negative Control Failure:** If UNIFORM shows $|\Delta| \ge 2\%$ with $p < 0.05$, the harness is measuring sort/executor overhead rather than real scheduling speedup. The benchmark is flagged `INVALID_BENCHMARK`.
2. **Sanity Floor Violation:** If $W = 33$ ($W \ge \text{level size}$) shows a statistically significant "win", the session is flagged `INVALID_SESSION`.

---

## 4. Execution Procedure on `mgpu`

```bash
# Ensure process pinning to half the cores per Code Policy Rule 53
taskset -c 0-15 ./build/tools/taskdag_sched_bench --reps 30 --warmups 3 --csv docs/technical/taskdag_sched_results.csv
```

---

## 5. Raw Measurement Results Register

*Note: In accordance with Rules 32, 43, and 58, no synthetic or fabricated measurements are recorded below. Real data will be committed from execution on `mgpu` with `clang++-22`.*

```
# Status: Unbuilt and unrun on target machine.
# Target environment: mgpu (192.168.1.155), clang++-22, libc++, -O3 -march=x86-64-v3.
```

### Summary Table Template

| Workload | Config | Policy | N | BaseMed(s) | BaseIQR(s) | OrdMed(s) | OrdIQR(s) | Ratio | Δ(%) | p-value | Verdict |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| *Pending `mgpu` run* | | | | | | | | | | | |

### Raw Repetition CSV Schema
`rep,workload,config,policy,baseline_seconds,ordered_seconds,ratio`

## Measured result — NEGATIVE (2026-08-18)

The pre-registered rule in `M6_SPEC.md` §2.5 was fixed **before this code existed**, and the
harness evaluates it itself. The verdict:

> **DECISION: NEGATIVE — LPT ordering did not achieve pre-registered speedup thresholds.**
> Ordering remains off-by-default.

The rule required a *conjunction*: ≥10% median makespan reduction on SKEW-LAST **and** ≥5% on
SKEW-PERM at 4 workers. SKEW-LAST passed comfortably; SKEW-PERM did not hold up across
permutations, so the conjunction fails and the milestone is recorded as negative.

Host: `mgpu`, 36 cores, all 36 pinned. 30 interleaved repetitions after warmups, median of paired
per-repetition ratios, sign test. Raw per-repetition data:
[`taskdag-sched-bench-results.csv`](taskdag-sched-bench-results.csv); full run log:
[`taskdag-sched-bench-run.txt`](taskdag-sched-bench-run.txt). No number here is absent from that CSV.

| Workload | Config | Δ median | p | Verdict |
| :--- | :--- | ---: | ---: | :--- |
| SKEW-LAST | 2 workers | +25.9% | 1.9e-09 | strong win |
| SKEW-LAST | 4 workers | +14.3% | 5.8e-08 | strong win |
| SKEW-LAST | 8 workers | +10.8% | 5.8e-08 | strong win |
| SKEW-PERM-1 | 4 workers | +7.8% | 1.4e-03 | win |
| SKEW-PERM-2 | 4 workers | +1.8% | 3.6e-01 | **tie** |
| SKEW-PERM-3 | 4 workers | +3.1% | 1.6e-02 | **inconclusive** |
| SKEW-PERM-4 | 4 workers | +14.4% | 1.9e-09 | strong win |
| SKEW-PERM-5 | 4 workers | +8.7% | 6.0e-05 | win |
| UNIFORM | 2 / 4 / 8 workers | +0.4% / +0.1% / +0.1% | ns | tie (control held) |
| ZIPF (λ = 0, 0.5, 1.0) | 4 workers | +0.9% / −0.5% / +0.1% | ns | tie |
| SANITY-FLOOR-SKEW | 33 workers | +1.3% | 5.9e-01 | tie (floor held) |
| SKEW-LAST | local_parallel | +2.4% | 5.2e-03 | inconclusive |

### What this actually says

Ordering **does** help on the case LPT is built for: when the expensive task is enqueued last, a
FIFO queue starts it last and every worker waits on it, and reordering recovers 10–26%. That effect
is real and strongly significant.

It does **not** generalise. Across five seeded permutations of the same task costs the win ranges
from +1.8% (a tie) to +14.4%, so the benefit depends on how adversarial the insertion order happens
to be, not on the skew itself. ZIPF — a realistically skewed level — shows nothing at any risk
margin. Since real callers do not systematically enqueue their heaviest task last, the expected win
is small and unreliable, which is exactly what the ≥5%-on-SKEW-PERM clause was written to detect.

The variance risk margin (λ) never moved anything measurable and stays at its default of 0.

### Why the numbers can be believed

Two guards had to hold, and both did:

- **UNIFORM (negative control)** stayed within ±0.4%. Had it "improved", the harness would have been
  timing its own sort rather than the schedule.
- **The sanity floor** (33 workers ≥ 33 tasks, where ordering provably cannot matter) tied at 1.3%,
  p = 0.59.

**An earlier session was thrown out by that floor.** Pinned to 16 cores, the floor showed a 3.2%
"win" (p = 0.016) and the harness declared `INVALID_SESSION`. The premise of the floor is that every
task starts immediately, which needs enough *cores*, not merely enough workers — with 33 workers on
16 cores the heavy task still queues, so ordering still pays. That session also showed +27.6% and
+17.3% on SKEW-LAST; those numbers were discarded along with the rest of it, because the same run
demonstrated the clock moving where it provably should not.

### Consequence

Per the pre-registered rule, `nimblecas.taskdag_sched` ships **off by default**
(`SgeeExecutorConfig::cost_ordering == false`). It is kept rather than deleted because the
SKEW-LAST result is genuine and reproducible: a caller who knows their heavy tasks are enqueued
late can opt in and measure their own workload. Turning it on by default would trade a reliable
no-op for an unreliable few percent.

## Known issue — the gRPC cluster tests are load-sensitive (not an M6 regression)

Recorded here because it was observed during this milestone's verification, and because the
measured rates matter more than the impression.

`taskdag_sgee_grpc_cluster_tests` and `modgcd_sgee_grpc_cluster_tests` drive a real 3-node Raft
quorum with an 80 ms base election timeout. On a **quiet** machine both suites passed 4/4 and the
modgcd failover leg passed 10/10. On a **shared, loaded** machine (`mgpu`, load average ~7, another
user's CUDA suite running) the pair passed **5/6**, with the failover leg aborting on
`distributed_error` and, once, the mTLS rejection leg failing (that one passed 6/6 on retry).

This is not caused by M6: `cost_ordering` defaults to false and no cluster test enables it, so
M6's only effect on that path is one bool test per level.

Two hypotheses were considered and rejected on evidence rather than plausibility:
- *Mixed toolchain* — NimbleCAS was rebuilt with clang 22.1.8 while `libsgee_capi_grpc.so` was
  built by the previous compiler. But that boundary is a **pure C ABI**, which is exactly what such
  a difference is safe across. (An attempted SGEE rebuild was also a no-op: ninja does not track
  the compiler binary as a dependency, so nothing recompiled.)
- *A new defect in the retry path* — the client retry budget was already fixed this milestone to be
  deadline-based rather than attempt-based, and that took the failover leg from ~25% failures to
  10/10 on a quiet box.

The residual cause is timing: under CPU contention a Raft election can exceed the client's fixed
5 s transport retry budget, and the run aborts honestly rather than incorrectly. The principled fix
is to derive that budget from the cluster's election timeout the way SGEE's server derives its own
await budget (`4*(2*election + heartbeat)`), instead of a constant chosen by hand. That is
deliberately **not** done here: this milestone already demonstrated that tuning a constant until the
symptom disappears produces a number that encodes a misunderstanding. It needs the election timing
plumbed to the client, which is a change to the C ABI's surface and belongs on its own branch.
