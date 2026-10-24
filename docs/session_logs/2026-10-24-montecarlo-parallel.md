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

## Review (after the first push, 9def168)

The feature landed on `main` before review; that is a policy deviation, recorded here. A
read-only `claude -p` review then found, and a follow-up commit fixed:

1. **Block count wrapped** for `samples` within 65535 of `UINT64_MAX`:
   `(samples + block - 1) / block` became 0, so `integrate` returned **0.0 as a success**. Now
   quotient + remainder test; block ends computed as `begin + min(samples - begin, block)`.
2. **Allocation could throw out of the railway**: the serial path reserved one `double` per
   block (the old code allocated nothing); `samples = 2^62` asks for 512 TiB. Serial now keeps a
   running total (same additions, same bits, no allocation); parallel runs rounds of 4096 blocks.
3. **FP environment** is per thread and not inherited by pool threads: documented as a
   precondition of `integrate_parallel` (not enforced).
4. **The bench could print `bit-identical true` for two failed calls** (both left at 0.0): values
   now start as NaN and `same()` rejects NaN.
5. Pre-existing: a NaN bound passed `b < a` and produced a NaN estimate; now `domain_error`
   (tested).

A second review by agy (`gemini-4-argon-high`; the first attempt failed because
`gemini-3.1-pro-high` no longer exists) on the fixed code found, and this commit also fixes:

6. **The second parallel round was never tested** (one round = 4096 blocks = 268M samples; the
   largest test used 65 blocks), so `block_fn(k)` instead of `block_fn(first + k)` survived. New
   test: a constant integrand over 4096 blocks + 5 samples must give exactly the width. That
   mutant was run and is now killed.
7. **No test pinned `estimate_pi`'s counter pairing** (2i, 2i+1) independently of `pi_block`;
   now one does.
8. Pre-existing, now `domain_error`: an empty `std::function` threw `bad_function_call` out of
   the railway (`integrate`, `integrate_parallel`, `rejection_sample`); `estimate_pi` above 2^63
   samples silently replayed earlier darts (`2*i` wraps); `rejection_sample` let NaN bounds and a
   NaN ceiling through, and could still throw from an up-front `reserve` when both `want` and
   `max_trials` were huge (now reserves at most 2^20 and grows); infinite bounds gave a NaN
   estimate. A finite interval whose width overflows (`[-1e308, 1e308]`) now returns `overflow`,
   as `rng::uniform_double` already did.

Not acted on: `parallel::for_ranges` with `grain == 0` divides by zero on PPL (outside this
change; montecarlo always passes 1); `sample_variance` overflows on `{1e308, 1e308}`; montecarlo
does not use the AVX-512 `counter_u64_batch`; the speedups are wall-clock timings, not a perf
profile.

## Not done
- One mutant was run: summing the block partials in REVERSE in both entry points. Against the
  first version of the pin test (3 blocks + tail) it **survived** — four partials summed either
  way gave the same bits. The test now uses 64 blocks + a 3-sample tail and first asserts that
  this input is order-sensitive (forward != reverse), so it cannot pass vacuously; the mutant is
  killed. No wider mutation sweep.
