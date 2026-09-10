# `nimblecas.csp_compile` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/csp_compile/csp_compile.cppm`

Ahead-of-time source code compilation for finite-domain constraint satisfaction problems. This module transforms
a declarative problem into standalone, parallel source code targeted at modern execution environments: C++23
(multi-threaded via `std::jthread`), CUDA (`__global__` device kernels paired with host launchers), or Triton
(`@triton.jit` tensor conflict scorers).

The module adheres strictly to the **honesty boundary** (Code Policy Rule 32):

- **Source emission, not a JIT engine:**
  This module emits source text as a `std::string`. It is not a just-in-time compiler: it never produces PTX or
  machine instructions, never loads dynamic libraries into process memory, and never allocates device buffers or
  launches GPU kernels itself. Because the emitter cannot compile or run its own output, it makes no runtime claims
  for uncompiled text; runtime guarantees are strictly those of the underlying mathematical algorithms.
- **Compilation of declarative `WireCsp`, not functional `Csp`:**
  The compiler accepts the declarative `WireCsp` layer from [`nimblecas.csp`](csp.md), not the functional `Csp`.
  `Csp` expresses constraints as `std::function` closures over host memory, which cannot be inspected, analysed,
  or emitted by an external code generator. The declarative form represents constraints as concrete data (`ConstraintKind`,
  variable scope, and parameters). This representation is strictly less expressive than arbitrary C++ functors,
  supporting only catalogued constraint families; this expressiveness boundary is accepted and preserved in full.
- **Two search strategies with fundamentally different guarantees:**
  - `Strategy::exhaustive`: **complete search**. It enumerates the entire Cartesian product of variable domains
    mapped as a mixed-radix index space. Exhaustive search settles satisfiability in both directions: it returns
    the lexicographically-first solution if one exists, and its termination without a solution is an exact
    **proof of unsatisfiability**. Because assignment spaces grow exponentially with variable count and domain
    cardinality, the strategy is capped by `max_search_space = 1ULL << 32U` ($4{,}294{,}967{,}296$ assignments).
    Refusing larger search spaces with `MathError::overflow` is an honest guard against emitting loops that could
    not finish in practical time.
  - `Strategy::min_conflicts`: **incomplete local search**. From a pseudorandom initial assignment, it iteratively
    re-values variables involved in conflicts to descend the violation landscape, incorporating random noise to
    escape local minima. It scales to large instances that exhaustive search cannot approach. However, it
    **cannot prove unsatisfiability**: exhausting its step budget without finding a solution yields `std::nullopt`,
    which represents `unknown`. The emitted source documents this warning prominently: a false return from local
    search must never be interpreted as a proof of unsatisfiability.
- **Deliberate exclusion of backtracking search:**
  The module intentionally does not emit parallel backtracking kernels. Backtracking's efficiency derives from
  dynamic pruning: partial assignments are abandoned, and the shape of the remaining search tree depends on every
  antecedent decision. This creates a sequential dependency chain rather than a data-parallel workload. An emitted
  "parallel backtracking" GPU kernel would either serialise across shared search bounds or duplicate the entire
  tree per thread. Single-threaded in-process `backtracking_search` on a CPU outperforms such an emitted kernel,
  and [`nimblecas.csp_dist`](csp_dist.md) already provides the principled way to distribute backtracking via prefix
  space partitioning. Emitting an uncompetitive parallel backtracking kernel would be dishonest, so the strategy
  is not offered.
- **Constraint granularity and local search gradients:**
  Local search steers by the number of violated constraints (`conflict_count`). Consequently, the formulation of a
  problem dictates the topography of the search landscape:
  - Formulating mutual distinction over $k$ variables as a single `all_different` constraint means the constraint
    is either satisfied or violated. Whether two variables collide or all $k$ collide, the violation count is 1.
    The resulting search landscape is flat, providing no gradient for local search to descend.
  - Formulating the identical requirement as $k(k-1)/2$ pairwise `not_equal` constraints produces a fine-grained
    violation count that decreases monotonically as collisions are resolved, providing a clear gradient that
    guides local search to a valid assignment.
  
  Both formulations have identical solution sets (exhaustive search is indifferent between them), but their
  local-search behaviour differs radically. This is an intrinsic algorithmic property rather than an engine defect;
  the compiler does not silently alter or re-weight constraints, ensuring that `conflict_count` reflects the exact
  number of violated constraints in the user's model.
- **Triton target emits a scorer, not a search walk:**
  Triton is engineered for dense tensor operations. Local search is a branchy, data-dependent random walk with
  divergent execution paths at every step, which is the antithesis of Triton's programming model. The Triton target
  therefore emits a batch conflict scorer (`@triton.jit`) rather than a search loop. Given a 2D batch of candidate
  assignments, the kernel evaluates the number of violated constraints for each candidate. This represents the
  computationally intensive inner loop of local search and composes cleanly with a higher-level driver in Python.
  Emitting a search walk in Triton and marketing it as a solver would be dishonest.
- **In-process reference implementations:**
  To ensure the emitted code's semantics are verifiable in C++ without requiring external toolchains, the module
  provides in-process reference routines (`reference_exhaustive`, `reference_min_conflicts`, `conflict_count`)
  that mirror the emitted algorithms exactly.

```cpp
import nimblecas.csp_compile;
```

Depends on [`core`](core.md) (`Result`, `MathError`) and [`csp`](csp.md) (`ConstraintKind`, `WireConstraint`,
`WireCsp`, `validate`, `holds`, `as_csp`, `backtracking_search`).

## Compilation targets and strategies

```cpp
enum class Target : std::uint8_t {
    cpp,     // C++23, parallel over std::jthread
    cuda,    // a __global__ kernel plus a host launcher
    triton,  // a @triton.jit conflict-count scorer, not a solver -- see the header
};

enum class Strategy : std::uint8_t {
    exhaustive,     // complete: settles SAT and UNSAT, but is the whole domain product
    min_conflicts,  // incomplete: scales, and can never report unsatisfiable
};

inline constexpr std::uint64_t max_search_space = 1ULL << 32U;
```

| Enumerator / Constant | Role |
| :--- | :--- |
| `Target::cpp` | Emits self-contained C++23 code using standard headers (`<algorithm>`, `<cstdint>`, `<thread>`, `<vector>`) parallelised across `std::jthread`. |
| `Target::cuda` | Emits CUDA device code (`__device__ __constant__` tables, device evaluation functions, and `__global__` kernels) paired with an `inline` host launcher. |
| `Target::triton` | Emits Python source containing a `@triton.jit` block-parallel kernel that evaluates constraint conflict counts across candidate batches. |
| `Strategy::exhaustive` | Complete assignment space scan over a mixed-radix index space. Settles both satisfiability and unsatisfiability. |
| `Strategy::min_conflicts` | Incomplete stochastic local search via min-conflicts heuristic with counter-based pseudo-random noise. |
| `max_search_space` | Maximum domain product ($2^{32} = 4{,}294{,}967{,}296$) permitted for `Strategy::exhaustive`. |

## Configuration options

```cpp
struct EmitOptions {
    Target target{Target::cpp};
    Strategy strategy{Strategy::exhaustive};
    std::size_t shards{8};
    std::size_t walkers{8};
    std::uint64_t max_steps{100000};
    std::uint32_t noise_per_1024{200};
    std::uint64_t seed{0x9E3779B97F4A7C15ULL};
    std::string name_prefix{"nc_csp"};
};
```

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `target` | `Target` | Language and execution runtime to generate. Defaults to `Target::cpp`. |
| `strategy` | `Strategy` | Search algorithm to emit for C++ and CUDA. Defaults to `Strategy::exhaustive`. |
| `shards` | `std::size_t` | Number of index shards into which the search space is divided for exhaustive C++ execution. Purely controls concurrency and does not alter the result. |
| `walkers` | `std::size_t` | Number of independent local-search walkers for `Strategy::min_conflicts`. |
| `max_steps` | `std::uint64_t` | Maximum iterative repair step budget per walker before reporting `unknown`. |
| `noise_per_1024` | `std::uint32_t` | Probability (in parts per 1024) of selecting a random domain value instead of the conflict-minimising value. Integer formulation eliminates floating-point discrepancies across compilers. |
| `seed` | `std::uint64_t` | 64-bit base seed for the deterministic counter-based generator (Splitmix64). |
| `name_prefix` | `std::string` | Symbol prefix prepended to all emitted functions, tables, and kernels to prevent namespace collisions when multiple CSPs are compiled into one translation unit. |

## Preconditions and code emission

```cpp
[[nodiscard]] auto is_compilable_for(const WireCsp& w, const EmitOptions& opts) -> Result<void>;

[[nodiscard]] auto search_space(const WireCsp& w) -> Result<std::uint64_t>;

[[nodiscard]] auto entry_point_name(const EmitOptions& opts) -> std::string;

[[nodiscard]] auto emit(const WireCsp& w, const EmitOptions& opts) -> Result<std::string>;
```

| Function | Guarantee | Error conditions |
| :--- | :--- | :--- |
| `is_compilable_for` | Verifies whether the problem and configuration can be legally compiled under the chosen target and strategy. | `MathError::domain_error` if `w` is malformed, `name_prefix` is empty, `shards == 0`, `walkers == 0`, `max_steps == 0`, or `noise_per_1024 > 1024`; `MathError::overflow` if exhaustive search space exceeds `max_search_space` or `uint64_t`; `MathError::not_implemented` if `target == Target::triton` (Triton produces only a scorer, having no search strategy). |
| `search_space` | Computes the total Cartesian product of all variable domain sizes. | `MathError::domain_error` if `w` is malformed; `MathError::overflow` if the product exceeds `std::numeric_limits<std::uint64_t>::max()`. |
| `entry_point_name` | Computes the mangled function name of the primary entry point generated by the emitter: `<prefix>_conflict_kernel` for Triton, `<prefix>_solve_exhaustive` for exhaustive search, or `<prefix>_solve_min_conflicts` for local search. | Never fails. |
| `emit` | Generates self-contained source text implementing the specified CSP solver or scorer. | Same errors as `is_compilable_for`; for Triton, rejects malformed problems with `MathError::domain_error` or `MathError::overflow` and empty prefix with `MathError::domain_error`. |

### Emitted code characteristics

1. **C++23 (`Target::cpp`):**
   - Emits embedded domain values, offsets, and sizes as `constexpr` flat arrays.
   - Generates inline constraint evaluation (`*_satisfies`, `*_conflicts`, and `*_conflicted`).
   - For `exhaustive`, divides the index space evenly across `shards` using `std::jthread`. Each thread records the lowest satisfying index it finds into a thread-safe slot; the host reduces via `std::min_element` to ensure the overall lowest index wins.
   - For `min_conflicts`, runs `walkers` threads using Splitmix64. The winning assignment is selected from the lowest-indexed walker that succeeded, ensuring results are reproducible regardless of thread scheduling.
2. **CUDA (`Target::cuda`):**
   - Emits device tables in `__device__ __constant__` memory. No claim is made here about what that costs at runtime: nothing in this module has been compiled or measured.
   - Emits duplicate tables with a `_host` suffix for the host launcher, because CUDA `__constant__` memory cannot be accessed directly by host pointers.
   - For `exhaustive`, threads calculate grid strides across the index space and reduce the winning index via `atomicMin`. The launcher retrieves the index and decodes it into output host memory.
   - For `min_conflicts`, each thread runs an independent walker writing into an assigned result buffer.
3. **Triton (`Target::triton`):**
   - Emits Python text defining constants `<prefix>_NUM_VARS` and `<prefix>_NUM_CONSTRAINTS`.
   - Generates a `@triton.jit` kernel function that accepts an `assignments_ptr` tensor (2D layout: `n_rows × num_vars`), a `conflicts_ptr` destination tensor, `n_rows`, and a power-of-two `BLOCK` size.
   - Loads candidate rows vectorially into individual registers per variable, evaluates constraint expressions using bitwise operators (`&`, `|`), and stores accumulated conflict counts.

## Reference implementations

These in-process reference routines run the exact algorithms implemented by the emitted source text. They allow
validation of solver semantics directly within unit tests without invoking external compilers.

```cpp
[[nodiscard]] auto reference_exhaustive(const WireCsp& w)
    -> Result<std::optional<std::vector<std::int64_t>>>;

[[nodiscard]] auto reference_min_conflicts(const WireCsp& w, std::size_t walkers,
                                           std::uint64_t max_steps, std::uint32_t noise_per_1024,
                                           std::uint64_t seed)
    -> Result<std::optional<std::vector<std::int64_t>>>;

[[nodiscard]] auto conflict_count(const WireCsp& w, std::span<const std::int64_t> assignment)
    -> Result<std::uint64_t>;
```

| Function | Guarantee | Error conditions |
| :--- | :--- | :--- |
| `reference_exhaustive` | Exhaustively enumerates all assignments in ascending mixed-radix order. Returns the lexicographically-first solution, matching `backtracking_search(as_csp(w))`. Returns `std::nullopt` if the problem is unsatisfiable. | `MathError::domain_error` if `w` is malformed; `MathError::overflow` if search space exceeds `max_search_space` or `uint64_t`. |
| `reference_min_conflicts` | Executes multi-walker local search using the Splitmix64 counter sequence. Walkers are evaluated in ascending index order; the first successful walker wins. Returns `std::nullopt` on step exhaustion, indicating `unknown` (never unsatisfiable). | `MathError::domain_error` if `w` is malformed, `walkers == 0`, `max_steps == 0`, or `noise_per_1024 > 1024`. |
| `conflict_count` | Evaluates all constraints in `w` against `assignment`, returning the exact number of violated constraints. Returns 0 if and only if the assignment is a valid solution. | `MathError::domain_error` if `w` is malformed or `assignment.size() != w.domains.size()`. |

## Error model

| Condition | Error |
| :--- | :--- |
| Malformed `WireCsp` (empty domain, empty variable set, out-of-range scope, duplicate scope index, ragged table) | `MathError::domain_error` |
| `name_prefix` is empty in `EmitOptions` (`is_compilable_for`, `emit`) | `MathError::domain_error` |
| `shards == 0` for `Strategy::exhaustive` (`is_compilable_for`) | `MathError::domain_error` |
| `walkers == 0` or `max_steps == 0` for `Strategy::min_conflicts` (`is_compilable_for`, `reference_min_conflicts`) | `MathError::domain_error` |
| `noise_per_1024 > 1024` (`is_compilable_for`, `reference_min_conflicts`) | `MathError::domain_error` |
| Assignment length does not match variable count (`conflict_count`) | `MathError::domain_error` |
| Search space exceeds `max_search_space` ($2^{32}$) for exhaustive strategy (`is_compilable_for`, `reference_exhaustive`) | `MathError::overflow` |
| Cartesian product of domains exceeds `std::uint64_t` (`search_space`, `is_compilable_for`) | `MathError::overflow` |
| Linear constraint coefficients and domain extrema produce sums that could overflow `std::int64_t` | `MathError::overflow` |
| Requesting a strategy for `Target::triton` in `is_compilable_for` | `MathError::not_implemented` |

## Worked examples

### Emitting and verifying a complete C++23 solver (Pigeonhole principle)

The 3-pigeon, 2-hole instance cannot be satisfied. Emitting with `Strategy::exhaustive` generates a complete
scan that returns `false`, which constitutes an exact proof:

```cpp
import std;
import nimblecas.core;
import nimblecas.csp;
import nimblecas.csp_compile;

using namespace nimblecas;
using namespace nimblecas::csp_compile;

// Formulate 3 pigeons, 2 holes
WireCsp ph;
ph.domains = {{0, 1}, {0, 1}, {0, 1}};
ph.constraints.push_back(WireConstraint{
    .kind = ConstraintKind::all_different, .scope = {0, 1, 2}, .params = {}});

EmitOptions opts;
opts.target = Target::cpp;
opts.strategy = Strategy::exhaustive;
opts.name_prefix = "ph_3_2";

// Emits C++23 source text containing the entry point `ph_3_2_solve_exhaustive`
auto src = emit(ph, opts);
if (src) {
    // *src contains the generated C++23 code with header comment:
    // "// COMPLETE search: the whole assignment space is scanned, so a false return is a
    //  // PROOF that no solution exists, not a budget having run out."
}

// In-process verification: reference_exhaustive returns std::nullopt (proven UNSAT)
auto ref = reference_exhaustive(ph);
if (ref && !ref->has_value()) {
    // Successfully proved unsatisfiable in-process
}
```

### Local search with pairwise constraints (8-queens)

To provide an effective gradient for local search, mutual row distinction is formulated as pairwise
`not_equal` constraints rather than a monolithic `all_different`:

```cpp
import std;
import nimblecas.core;
import nimblecas.csp;
import nimblecas.csp_compile;

using namespace nimblecas;
using namespace nimblecas::csp_compile;

auto range_domain = [](std::int64_t k) {
    std::vector<std::int64_t> d;
    for (std::int64_t v = 0; v < k; ++v) d.push_back(v);
    return d;
};

// Pairwise 8-queens
WireCsp queens_8;
for (std::size_t i = 0; i < 8; ++i) {
    queens_8.domains.push_back(range_domain(8));
}
for (std::size_t i = 0; i < 8; ++i) {
    for (std::size_t j = i + 1; j < 8; ++j) {
        // Pairwise row difference gives local search a decreasing conflict gradient
        queens_8.constraints.push_back(WireConstraint{
            .kind = ConstraintKind::not_equal, .scope = {i, j}, .params = {}});
        // Diagonal constraints
        queens_8.constraints.push_back(WireConstraint{
            .kind = ConstraintKind::abs_diff_ne,
            .scope = {i, j},
            .params = {static_cast<std::int64_t>(j - i)}});
    }
}

EmitOptions cuda_opts;
cuda_opts.target = Target::cuda;
cuda_opts.strategy = Strategy::min_conflicts;
cuda_opts.walkers = 128;
cuda_opts.max_steps = 10000;
cuda_opts.name_prefix = "queens8";

auto cuda_code = emit(queens_8, cuda_opts);
if (cuda_code) {
    // Generates `__global__ void queens8_kernel(...)`
}

// Run the reference min-conflicts walk in-process
auto sol = reference_min_conflicts(queens_8, /*walkers=*/8, /*max_steps=*/20000,
                                   /*noise_per_1024=*/200, /*seed=*/0x1234ULL);
if (sol && sol->has_value()) {
    auto c = conflict_count(queens_8, std::span<const std::int64_t>(**sol));
    // c.value() == 0, confirming the assignment satisfies all constraints
}
```

### Emitting a Triton batch scorer

```cpp
import std;
import nimblecas.core;
import nimblecas.csp;
import nimblecas.csp_compile;

using namespace nimblecas;
using namespace nimblecas::csp_compile;

WireCsp csp;
// ... populate csp ...

EmitOptions opts;
opts.target = Target::triton;
opts.name_prefix = "benchmark";

auto triton_src = emit(csp, opts);
if (triton_src) {
    // Emits Python source defining:
    // def benchmark_conflict_kernel(assignments_ptr, conflicts_ptr, n_rows, BLOCK: tl.constexpr):
    // Computes conflict counts for a 2D batch of candidate assignments
}
```

## See also

- [`nimblecas.csp`](csp.md) — in-process constraint satisfaction and declarative `WireCsp` definition.
- [`nimblecas.csp_dist`](csp_dist.md) — distributed constraint satisfaction and model counting over task graphs.
- [`nimblecas.core`](core.md) — `Result<T>` and `MathError` error conventions.
- [Documentation hub](../Index.md)
