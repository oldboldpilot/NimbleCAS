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
