# Session 2026-10-24 — parallel Monte Carlo estimators, roadmap refresh

**Author:** Olumuyiwa Oluwasanmi
**Host:** DESKTOP-3SON9JT (Windows 11 + WSL Ubuntu)

## What changed

**`nimblecas.montecarlo` gained `integrate_parallel` and `estimate_pi_parallel`.** This was the
CPU Monte Carlo parallel-sum item: the module had claimed since it was written that its
estimators were "partition-independent by construction ... the design admits parallel
summation", but nothing ran them in parallel.

That claim was also only half true. The *draws* are partition-independent (sample `i` is a pure
function of `counter_u64(key, i)`), but the *sum* is not: floating-point addition is not
associative, so a different grouping of the same draws gives a different last few bits. The
header said "summing them in any grouping yields the same total", which is false for `double`.

The fix makes the grouping part of the definition:

- `0..samples-1` is cut into blocks of `montecarlo_block` = 65536 samples; each block is summed
  in index order and the partials are added in block order.
- The serial `integrate` / `estimate_pi` now use that same grouping, so the parallel entry
  points (blocks run through `parallel::transform_index`, grain 1) return **bit-identical**
  results at any thread count.
- **Consequence:** serial `integrate`'s output moved by a few ulps relative to the old single
  running sum. No caller pinned its last bit (the tests use tolerances and self-reproducibility),
  and the change is far inside the sampling error. `estimate_pi` is unchanged: an integer hit
  count does not depend on grouping.
- `integrate_parallel` calls `f` concurrently, so `f` must be safe to call concurrently. This is
  stated in the declaration and the reference.

Tests (`montecarlo_tests`, 13 cases, 3 new): bit-identity of serial vs parallel across sample
counts straddling every block edge (1, block-1, block, block+1, a ragged multi-block count,
1000003); a hand-computed blocked sum that `integrate` must match bit-for-bit, so serial and
parallel cannot drift to another grouping *together*; `estimate_pi_parallel` identity; and the
domain errors.

`tools/montecarlo_bench.cpp` (target `montecarlo_bench`) times serial vs parallel and prints
whether the results are bit-identical. Measured numbers are in
[the reference](../reference/montecarlo.md).

## Roadmap corrections

- §0 "Open" list: the GPU item listed qmc/control/wavelets/krylov as unbuilt; CUDA families for
  all four (and batched Levenberg–Marquardt) are in `src/gpu/`. The WebGPU/WASM item listed the
  front-end as unbuilt; `web/` ships it over a 17-module WASM slice. Both entries now say what
  remains.
- §6.2 item 3 still read "routing BLOCKED at the C ABI"; affinity routing closed at
  `2cf106e`. The status now also records the owner's rule that SGEE exposes no C interfaces, so
  the remaining producer-identity addition is a C++ change consumed through SGEE's exported C++
  modules.

## Verification

- Linux (WSL, clang 23.1.5 + libc++): full `ctest` 174/174.
- Windows (clang 23.1.4 + MSVC STL): full `ctest` 174/174.

## Not done

- No adversarial review was run (time-boxed session).
- One mutant was run: summing the block partials in REVERSE in both entry points. Against the
  first version of the pin test (3 blocks + tail) it **survived** — four partials summed either
  way gave the same bits. The test now uses 64 blocks + a 3-sample tail and first asserts that
  this input is order-sensitive (forward != reverse), so it cannot pass vacuously; the mutant is
  killed. No wider mutation sweep.
