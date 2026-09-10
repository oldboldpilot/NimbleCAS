# `nimblecas.csp_dist` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/csp_dist/csp_dist.cppm`

Distributed finite-domain constraint satisfaction over [`nimblecas.taskdag`](taskdag.md) task graphs.
While [`nimblecas.csp`](csp.md) provides in-process solvers (`backtracking_search`, `backtracking_search_fc`,
`solution_count`, `parallel_search`) over both functional and declarative representations, this module expresses
search and model counting as explicit task graphs. These graphs can be dispatched across any conforming
`Executor`: `serial_executor`, `local_parallel_executor`, or the cluster-wide `SgeeDistributedExecutor` over an
SGEE broker.

The module adheres strictly to the **honesty boundary** (Code Policy Rule 32):

- **Exact partition decomposition:**
  The decomposition fixes the first $k$ variables to one combination of values from their domains. That yields
  an independent sub-problem. Over the ascending Cartesian product of those $k$ domains, the sub-problems
  **partition the assignment space exactly**: every complete assignment lies in precisely one sub-problem. Nothing
  is evaluated twice and nothing is omitted. Two mathematical consequences follow:
  - The original problem is **satisfiable** if *any* sub-problem is satisfiable.
  - The original problem is **unsatisfiable** only if *every* sub-problem is unsatisfiable.
  
  The second guarantee is what a portfolio of differently-seeded solvers can never give. A portfolio that finds
  no solution has merely exhausted individual search budgets and must honestly answer `unknown`; an exhausted
  partition is a **complete, cluster-wide proof of unsatisfiability**.
- **Distributed model counting:**
  Because the prefix sub-problems partition the space without overlap, the per-prefix solution counts sum directly
  to the total number of solutions for the whole problem. This provides exact distributed model counting rather
  than merely distributed search.
- **Determinism and the equivalence contract:**
  `nimblecas.csp` guarantees that search returns the lexicographically-first assignment. That promise survives
  distribution: `prefix_assignment` enumerates the prefixes in ascending lexicographic order, each worker returns
  the lexicographically-first extension of its own prefix, and the merge step selects the lowest-indexed shard that
  found a solution. The merge is associative and index-ordered; the result is strictly independent of shard completion
  order, arrival order, worker count, or executor choice. Running an instance under `serial_executor`,
  `local_parallel_executor`, or `SgeeDistributedExecutor` yields the identical assignment, bit-for-bit matching
  `backtracking_search(as_csp(w))`.
- **Model verification against the original problem:**
  Every assignment returned by a shard is verified against the **original, unrestricted `WireCsp`** (and its
  `as_csp` conversion), rather than against the restricted sub-problem the shard evaluated. A shard searches a
  problem with collapsed prefix domains; verifying the produced assignment against the uncollapsed original
  problem guarantees that an internal solver discrepancy, serialization flaw, or memory corruption surfaces as
  `MathError::undefined_value` rather than masquerading as a valid assignment.
- **Honest resource bounding and task caps:**
  Finite-domain constraint satisfaction is NP-complete, and partitioning divides the work without reducing its total
  combinatorial volume. The task count is the product of the fixed domain sizes, and every task is materialised
  eagerly in the graph carrying its own encoded sub-problem. The split is therefore capped by
  `max_prefix_tasks = 1U << 16U` (65,536 tasks); past this point graph construction and payload encoding overhead
  dominate the search. Exceeding the cap returns an honest `MathError::overflow`, never a silently truncated partition
  (a truncated partition would quietly transform a complete search into an incomplete one).
- **Honest counting limits:**
  `count_distributed` accepts a `per_shard_limit`. When a shard stops because it reached a non-zero limit, it may have
  had further solutions to find. The shard explicitly flags that it was capped, causing `CountResult::exact` to be
  `false`. The returned count is then reported honestly as a **lower bound** on the true number of solutions, rather
  than passed off as an exact total.
- **Verification scope:**
  The distributed contract is verified in tests over `SgeeDistributedExecutor`, where tasks undergo real `Payload`
  encoding, broker queue transport, and registry lookup of operation identifiers. However, the transport in those
  tests is an in-process broker (`FakeBrokerPort` with `InMemoryResultChannel`), not a physical network. What is
  established is the executor, the wire format, and the registry contract — not the behaviour of a real cluster under
  network partition, packet loss, or latency.

```cpp
import nimblecas.csp_dist;
```

Depends on [`core`](core.md) (`Result`, `MathError`, `Payload`), [`csp`](csp.md) (`Csp`, `WireCsp`, `as_csp`,
`backtracking_search_fc`, `solution_count`, `prefix_count`, `prefix_assignment`, `restrict_prefix`, `holds`), and
[`taskdag`](taskdag.md) (`TaskGraph`, `TaskRegistry`, `Executor`).

## Operation identifiers and constants

```cpp
inline constexpr std::string_view solve_op_id = "nimblecas.csp.prefix_solve/v1";
inline constexpr std::string_view count_op_id = "nimblecas.csp.prefix_count/v1";
inline constexpr std::uint64_t max_prefix_tasks = 1U << 16U;
```

| Identifier / Constant | Role |
| :--- | :--- |
| `solve_op_id` | Operation name `"nimblecas.csp.prefix_solve/v1"` registered in the task registry for prefix solves. |
| `count_op_id` | Operation name `"nimblecas.csp.prefix_count/v1"` registered in the task registry for prefix counting. |
| `max_prefix_tasks` | Maximum number of prefix sub-problems a split may produce ($2^{16} = 65{,}536$). |

## Outcome data structures

```cpp
struct CountResult {
    std::uint64_t count{0};
    bool exact{true};

    [[nodiscard]] auto operator==(const CountResult&) const noexcept -> bool = default;
};
```

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `count` | `std::uint64_t` | The accumulated number of solutions discovered across all shards. |
| `exact` | `bool` | `true` if all shards ran without hitting a limit, guaranteeing `count` is the exact total; `false` if any shard was capped by `per_shard_limit`, in which case `count` is a lower bound. |

## Wire format and task registration

Shards communicate via binary payloads using little-endian fixed-width framing. Signed 64-bit integers travel
in two's-complement representation, ensuring exact round-tripping of all values including `INT64_MIN`. Fixed
64-bit magic numbers guard against protocol and format mismatches.

```cpp
[[nodiscard]] auto encode_csp(const WireCsp& w) -> Result<Payload>;
[[nodiscard]] auto decode_csp(std::span<const std::byte> bytes) -> Result<WireCsp>;
[[nodiscard]] auto encode_solution(const std::optional<std::vector<std::int64_t>>& s)
    -> Result<Payload>;
[[nodiscard]] auto decode_solution(std::span<const std::byte> bytes)
    -> Result<std::optional<std::vector<std::int64_t>>>;

[[nodiscard]] auto register_ops(TaskRegistry& reg) -> Result<void>;
```

| Function | Role |
| :--- | :--- |
| `encode_csp` | Serialises a `WireCsp` problem with magic header `0x4e43535f43535031ULL` (`NCS_CSP1`). Validates the problem structure before encoding. |
| `decode_csp` | Deserialises a `WireCsp` problem. Validates magic header, bounds checks lengths against payload size, and validates problem integrity. Truncated or corrupt inputs return `MathError::syntax_error`. |
| `encode_solution` | Serialises an optional assignment (`std::nullopt` or concrete values) with magic header `0x4e43535f434f4c31ULL` (`NCS_COL1`). |
| `decode_solution` | Deserialises an optional assignment. Rejects malformed or truncated payloads with `MathError::syntax_error`. |
| `register_ops` | Registers both `solve_op_id` and `count_op_id` task handlers into a `TaskRegistry`. Must be invoked on both the coordinator and every remote worker node. |

## Task graph construction

Graphs are constructed eagerly by enumerating the prefix sub-problems. Each task receives an encoded
sub-problem payload and executes independently with no inter-task dependencies.

```cpp
[[nodiscard]] auto build_prefix_graph(const TaskRegistry& reg, const WireCsp& w,
                                      std::size_t fixed_vars) -> Result<TaskGraph>;

[[nodiscard]] auto build_count_graph(const TaskRegistry& reg, const WireCsp& w,
                                     std::size_t fixed_vars, std::uint64_t per_shard_limit)
    -> Result<TaskGraph>;
```

| Function | Guarantee | Error conditions |
| :--- | :--- | :--- |
| `build_prefix_graph` | Constructs an embarrassingly parallel `TaskGraph` containing one task per prefix sub-problem. Each task executes `backtracking_search_fc` on its restricted sub-problem. | `MathError::domain_error` if `fixed_vars == 0`, `fixed_vars > w.domains.size()`, or `w` is malformed; `MathError::overflow` if the prefix product exceeds `max_prefix_tasks` or `uint64_t`. |
| `build_count_graph` | Constructs an embarrassingly parallel `TaskGraph` containing one task per prefix sub-problem. Each task counts solutions up to `per_shard_limit`. | Same error semantics as `build_prefix_graph`. |

## Distributed solving entry points

```cpp
[[nodiscard]] auto solve_distributed(const WireCsp& w, std::size_t fixed_vars, Executor& exec)
    -> Result<std::optional<std::vector<std::int64_t>>>;

[[nodiscard]] auto count_distributed(const WireCsp& w, std::size_t fixed_vars,
                                     std::uint64_t per_shard_limit, Executor& exec)
    -> Result<CountResult>;
```

| Function | Guarantee | Error conditions |
| :--- | :--- | :--- |
| `solve_distributed` | Executes prefix search tasks across `exec`. Merges results in prefix order: returns the verified assignment from the lowest-indexed shard that succeeded, matching `backtracking_search`. Returns an engaged `Result` holding `std::nullopt` when every shard reports no solution (a complete proof of unsatisfiability). | `MathError::domain_error` if `fixed_vars == 0`, `fixed_vars > w.domains.size()`, or `w` is malformed; `MathError::overflow` if prefix product exceeds `max_prefix_tasks`; `MathError::undefined_value` if the returned model fails re-verification against `w`. |
| `count_distributed` | Executes prefix counting tasks across `exec`. Sums discovered counts across all shards. When `per_shard_limit == 0`, the count is exact. If any shard reached a non-zero limit, `exact` is marked `false`. | Same domain and overflow errors as `solve_distributed`; `MathError::overflow` if the summed solution count exceeds `std::numeric_limits<std::uint64_t>::max()`. |

## Error model

| Condition | Error |
| :--- | :--- |
| `fixed_vars == 0` or `fixed_vars > w.domains.size()` (`build_*_graph`, `solve_distributed`, `count_distributed`) | `MathError::domain_error` |
| Malformed `WireCsp` (empty variable set, empty initial domain, out-of-range scope, duplicate scope variable, arity mismatch, ragged table) | `MathError::domain_error` |
| Linear constraint weighted sum or right-hand side comparison could overflow `std::int64_t` (from `validate(w)`) | `MathError::overflow` |
| Prefix sub-problem count exceeds `max_prefix_tasks` ($65{,}536$) or `std::uint64_t` | `MathError::overflow` |
| Summed solution count across shards exceeds `std::numeric_limits<std::uint64_t>::max()` (`count_distributed`) | `MathError::overflow` |
| Payload magic mismatch, truncated buffer, or corrupt length framing (`decode_csp`, `decode_solution`) | `MathError::syntax_error` |
| Returned assignment violates domain bounds or constraints of original `WireCsp` (`solve_distributed`) | `MathError::undefined_value` |

An unsatisfiable problem is **not an error**:
- `solve_distributed` returns an engaged `Result` holding `std::nullopt`. Because the prefix decomposition partitions the space, this is a complete proof of unsatisfiability, not an admission of budget exhaustion.
- `count_distributed` returns an engaged `Result` holding `CountResult{.count = 0, .exact = true}`.

## Worked examples

### Complete UNSAT proof via prefix partition (Pigeonhole principle)

Three pigeons cannot fit into two pigeonholes without sharing a hole. Expressed as a declarative CSP with three
variables over a domain of two values subject to an `all_different` constraint, the problem is arc-consistent yet
provably unsatisfiable. Splitting across two fixed variables partitions the search space into $2^2 = 4$ sub-problems.
Running across a serial or parallel executor confirms unsatisfiability completely:

```cpp
import std;
import nimblecas.core;
import nimblecas.csp;
import nimblecas.csp_dist;
import nimblecas.taskdag;

using namespace nimblecas;
using namespace nimblecas::csp_dist;

// Domain {0, 1} representing two pigeonholes
auto range_domain = [](std::int64_t k) {
    std::vector<std::int64_t> d;
    for (std::int64_t v = 0; v < k; ++v) d.push_back(v);
    return d;
};

// Three variables, each in {0, 1}, required to be pairwise distinct
WireCsp pigeonhole_3_2;
pigeonhole_3_2.domains = {range_domain(2), range_domain(2), range_domain(2)};
pigeonhole_3_2.constraints.push_back(WireConstraint{
    .kind = ConstraintKind::all_different, .scope = {0, 1, 2}, .params = {}});

auto exec = serial_executor();

// Partition over the first 2 variables (2^2 = 4 sub-problems)
auto res = solve_distributed(pigeonhole_3_2, /*fixed_vars=*/2, *exec);
if (res) {
    // res->has_value() is false: std::nullopt returned.
    // Every prefix was exhausted; unsatisfiability is proven across the partition.
}

auto count = count_distributed(pigeonhole_3_2, /*fixed_vars=*/2, /*per_shard_limit=*/0, *exec);
if (count) {
    // count->count == 0 && count->exact == true
}
```

### Exact distributed model counting (8-queens)

The 8-queens problem has exactly 92 solutions. Partitioning over one variable yields 8 shards; partitioning over
two variables yields 64 shards. Both decompositions sum to the exact total:

```cpp
import std;
import nimblecas.core;
import nimblecas.csp;
import nimblecas.csp_dist;
import nimblecas.taskdag;

using namespace nimblecas;
using namespace nimblecas::csp_dist;

auto range_domain = [](std::int64_t k) {
    std::vector<std::int64_t> d;
    for (std::int64_t v = 0; v < k; ++v) d.push_back(v);
    return d;
};

// Declarative 8-queens
WireCsp queens_8;
for (std::size_t i = 0; i < 8; ++i) {
    queens_8.domains.push_back(range_domain(8));
}
std::vector<std::size_t> all_vars{0, 1, 2, 3, 4, 5, 6, 7};
queens_8.constraints.push_back(WireConstraint{
    .kind = ConstraintKind::all_different, .scope = std::move(all_vars), .params = {}});
for (std::size_t i = 0; i < 8; ++i) {
    for (std::size_t j = i + 1; j < 8; ++j) {
        queens_8.constraints.push_back(WireConstraint{
            .kind = ConstraintKind::abs_diff_ne,
            .scope = {i, j},
            .params = {static_cast<std::int64_t>(j - i)}});
    }
}

auto par_exec = local_parallel_executor();

// Exact count across 64 shards (fixed_vars = 2, per_shard_limit = 0 for uncapped)
auto uncapped = count_distributed(queens_8, /*fixed_vars=*/2, /*per_shard_limit=*/0, *par_exec);
if (uncapped) {
    // uncapped->count == 92
    // uncapped->exact == true (complete exact count)
}

// Capped count: limits each shard to at most 1 solution
auto capped = count_distributed(queens_8, /*fixed_vars=*/1, /*per_shard_limit=*/1, *par_exec);
if (capped) {
    // capped->count <= 92
    // capped->exact == false: honest lower bound, reporting limit hit
}
```

### Distributed solving with SGEE

Remote workers evaluate sub-problems by decoding payloads received over the broker. Operation handlers must
be registered on both the coordinator and every worker:

```cpp
import nimblecas.core;
import nimblecas.csp;
import nimblecas.csp_dist;
import nimblecas.taskdag;
import nimblecas.taskdag_sgee;

using namespace nimblecas;
using namespace nimblecas::csp_dist;

TaskRegistry reg;
auto ok = register_ops(reg);

SgeeExecutorConfig cfg;
cfg.registry = &reg;
cfg.num_workers = 4;
cfg.poll_interval_ms = 1;

// Borrowed broker transport (driven here with in-process test fixtures)
FakeBrokerPort port;
InMemoryResultChannel results;
SgeeDistributedExecutor sgee{cfg, port, results};

// Queens instance solved across the SGEE executor
WireCsp queens_6;
// ... initialise queens_6 ...
auto sol = solve_distributed(queens_6, /*fixed_vars=*/2, sgee);
if (sol && sol->has_value()) {
    // The returned assignment is identical to backtracking_search(as_csp(queens_6))
    // and was verified against the uncollapsed queens_6 problem.
}
```

## See also

- [`nimblecas.csp`](csp.md) — in-process constraint satisfaction and declarative `WireCsp` specification.
- [`nimblecas.csp_compile`](csp_compile.md) — ahead-of-time source compilation for finite-domain CSPs.
- [`nimblecas.taskdag`](taskdag.md) — task graph representation, registry, and execution interfaces.
- [`nimblecas.taskdag_sgee`](taskdag_sgee.md) — cluster task execution over an SGEE message broker.
- [`nimblecas.sat_dist`](sat_dist.md) — distributed Boolean satisfiability solving over task graphs.
- [`nimblecas.core`](core.md) — `Result<T>`, `MathError`, and binary `Payload` serialization.
- [Documentation hub](../Index.md)
