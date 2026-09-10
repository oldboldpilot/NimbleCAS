# `nimblecas.sat_dist` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/sat_dist/sat_dist.cppm`

Distributed Boolean satisfiability solving over [`nimblecas.taskdag`](taskdag.md) task graphs.
While [`nimblecas.sat`](sat.md) provides single-node in-process solvers (`cdcl`, `dpll`, `solve_portfolio`)
and single-worker execution via `solve_shard`, this module expresses portfolio search and
cube-and-conquer space partitioning as explicit task graphs. These graphs can be dispatched across any
conforming `Executor`: `serial_executor`, `local_parallel_executor`, or the cluster-wide
`SgeeDistributedExecutor` over an SGEE broker.

The module adheres strictly to the **honesty boundary** (Code Policy Rule 32):

- **Two decompositions answering different questions:**
  - **Portfolio decomposition** (`build_portfolio_graph`, `solve_portfolio_distributed`): dispatches
    differently configured, independently seeded solvers across the whole formula. It is the natural
    choice when an instance is likely satisfiable, as one fortunate configuration will locate a model
    and finish the job while others terminate. However, its fundamental limitation is that it inherits
    each worker's resource bound: if no worker reaches a definitive verdict, the merged outcome is
    `SatVerdict::unknown`, and `unknown` is all it can honestly report.
  - **Cube-and-conquer decomposition** (`build_cube_graph`, `solve_cubes_distributed`): partitions the
    entire assignment space. Selecting $k$ decision variables and fixing them to each of their $2^k$
    truth assignments produces $2^k$ disjoint cubes. Each cube is an independent sub-problem with unit
    clauses appended for the fixed literals. Crucially, the sub-verdicts compose: the original formula is
    **satisfiable** if *any* cube is satisfiable, and **unsatisfiable** only if *every* cube is
    unsatisfiable. This provides what portfolio search cannot: a **complete, cluster-wide proof of
    unsatisfiability**.
- **Exact partition guarantee:** the $2^k$ cubes partition the Boolean hypercube without overlap or
  omission. No candidate assignment is evaluated twice, and none is missed. When all sub-solvers run to
  completion, the merged verdict is the exact verdict of the original formula, not an approximation.
- **Model verification against the original formula:** every returned model is verified by
  `nimblecas::verify_assignment` against the **original CNF formula**, rather than the restricted cube.
  A cube solver evaluates a formula augmented with artificial unit clauses; verifying the produced
  assignment against the unaugmented original ensures that an internal solver discrepancy or corruption
  never surfaces as a valid model. If verification fails, the solve returns `MathError::undefined_value`.
- **Deterministic associative merge reduction:** both portfolio and cube reductions are associative
  and order-independent. When multiple shards or cubes locate satisfying models, the model from the
  **lowest shard or cube index** wins. The result does not depend on thread scheduling, task completion
  order, or worker count.
- **The equivalence contract:** distributed execution guarantees that the answers returned do not reveal
  which executor ran them. Running an instance under `serial_executor`, `local_parallel_executor`, or
  `SgeeDistributedExecutor` yields the identical verdict and model. This contract is verified in tests
  over `SgeeDistributedExecutor` itself, not merely across the in-process `Executor` seam: the tasks
  go through real `Payload` encoding, a broker queue, and the worker's own registry lookup of the
  op id. The transport in those tests is `FakeBrokerPort` with `InMemoryResultChannel` -- an
  in-process broker, not a network -- so what is verified is the executor, the wire format and the
  registry contract, and not the behaviour of a real cluster under partition or latency.
- **Structural cube limits:** `max_cube_vars = 20`. Because $2^k$ tasks are materialised eagerly in the
  task graph, fixing more than 20 variables ($> 1{,}048{,}576$ tasks) causes graph construction and
  memory overhead to dominate the search itself.

```cpp
import nimblecas.sat_dist;
```

Depends on [`core`](core.md) (`Result`, `MathError`, `Payload`), [`sat`](sat.md) (`Cnf`, `SatResult`,
`SatVerdict`, `verify_assignment`, `cdcl`, `solve_shard`), and [`taskdag`](taskdag.md) (`TaskGraph`,
`TaskRegistry`, `Executor`).

## Wire format and registered operations

Shards communicate via binary payloads using little-endian fixed-width framing. Fixed 64-bit magic
numbers guard against format mismatches.

```cpp
inline constexpr std::string_view portfolio_op_id = "nimblecas.sat.portfolio_shard/v1";
inline constexpr std::string_view cube_op_id = "nimblecas.sat.cube_shard/v1";
inline constexpr std::size_t max_cube_vars = 20;

[[nodiscard]] auto encode_cnf(const Cnf& cnf) -> Result<Payload>;
[[nodiscard]] auto decode_cnf(std::span<const std::byte> bytes) -> Result<Cnf>;
[[nodiscard]] auto encode_result(const SatResult& r) -> Result<Payload>;
[[nodiscard]] auto decode_result(std::span<const std::byte> bytes) -> Result<SatResult>;

[[nodiscard]] auto register_ops(TaskRegistry& reg) -> Result<void>;
```

| Identifier / Function | Role |
| :--- | :--- |
| `portfolio_op_id` | Operation name `"nimblecas.sat.portfolio_shard/v1"` registered in the task registry. |
| `cube_op_id` | Operation name `"nimblecas.sat.cube_shard/v1"` registered in the task registry. |
| `max_cube_vars` | Maximum number of variables a cube split may fix (20, yielding up to $2^{20}$ tasks). |
| `encode_cnf` / `decode_cnf` | Binary serialization for `Cnf` using magic header `0x4e43535f434e4631ULL` (`NCS_CNF1`). |
| `encode_result` / `decode_result` | Binary serialization for `SatResult` using magic header `0x4e43535f52455331ULL` (`NCS_RES1`). |
| `register_ops` | Registers both `portfolio_shard` and `cube_shard` task handlers into a `TaskRegistry`. Must be called on both coordinator and remote workers. |

## Decomposition and graph construction

```cpp
[[nodiscard]] auto cube_of(const Cnf& cnf, std::size_t cube_vars, std::uint64_t cube_index)
    -> Result<Cnf>;

[[nodiscard]] auto build_portfolio_graph(const TaskRegistry& reg, const Cnf& cnf,
                                         std::uint64_t base_seed, std::size_t shards)
    -> Result<TaskGraph>;

[[nodiscard]] auto build_cube_graph(const TaskRegistry& reg, const Cnf& cnf, std::size_t cube_vars,
                                    std::uint64_t max_conflicts) -> Result<TaskGraph>;
```

| Function | Guarantee | Error conditions |
| :--- | :--- | :--- |
| `cube_of` | Generates the sub-formula restricted to `cube_index` by appending unit clauses for the first `cube_vars` variables. | `MathError::domain_error` if `cube_vars == 0`, `cube_vars > num_vars`, `cube_vars > max_cube_vars`, or `cube_index >= 2^cube_vars`. |
| `build_portfolio_graph` | Constructs an independent task graph of `shards` tasks. Each task executes `solve_shard` on the entire formula with a deterministic seed. | `MathError::domain_error` if `shards == 0` or formula is malformed. |
| `build_cube_graph` | Constructs an embarrassingly parallel graph of $2^{\text{cube\_vars}}$ tasks. Each task runs CDCL on its restricted formula with conflict bound `max_conflicts`. | `MathError::domain_error` if `cube_vars == 0`, `cube_vars > num_vars`, or `cube_vars > max_cube_vars`. |

## Distributed solving entry points

```cpp
[[nodiscard]] auto solve_portfolio_distributed(const Cnf& cnf, std::uint64_t base_seed,
                                               std::size_t shards, Executor& exec)
    -> Result<SatResult>;

[[nodiscard]] auto solve_cubes_distributed(const Cnf& cnf, std::size_t cube_vars,
                                           std::uint64_t max_conflicts, Executor& exec)
    -> Result<SatResult>;
```

| Function | Guarantee | Error conditions |
| :--- | :--- | :--- |
| `solve_portfolio_distributed` | Runs portfolio shards across `exec`. Merges verdicts: a definitive `unsatisfiable` wins; otherwise the verified model from the lowest shard index; otherwise `unknown`. Matches in-process `solve_portfolio`. | `MathError::domain_error` if `shards == 0` or formula is malformed; `undefined_value` if returned model fails verification against original CNF. |
| `solve_cubes_distributed` | Runs cube tasks across `exec`. Returns `satisfiable` with the lowest-indexed verified model if any cube succeeds. Returns `unsatisfiable` only when **all** cubes report `unsatisfiable`. If no model is found and any cube hit `max_conflicts`, returns `unknown`. | `MathError::domain_error` if `cube_vars == 0` or exceeds bounds; `undefined_value` if model fails verification. |

### The `max_conflicts` parameter in cube solving

In `solve_cubes_distributed`, setting `max_conflicts = 0` denotes an **unlimited conflict budget**.
Each cube task runs CDCL to exhaustion, guaranteeing a complete verdict (`satisfiable` or
`unsatisfiable`) for every cube. This ensures the merged cluster verdict is complete. Setting
`max_conflicts > 0` bounds per-cube runtimes; if difficult cubes exceed this threshold without a model,
the solve returns `SatVerdict::unknown` rather than stalling indefinitely.

## Worked examples

### Complete UNSAT proof via cube-and-conquer (Pigeonhole principle)

The pigeonhole problem with 3 pigeons and 2 holes is unsatisfiable: three pigeons cannot occupy two
holes without sharing one. This instance is tiny and any solver settles it instantly -- it is used here
because it is small enough to read, not because it is hard. Partitioning the search space over 2
variables ($2^2 = 4$ cubes) yields an exact UNSAT proof:

```cpp
import std;
import nimblecas.core;
import nimblecas.sat;
import nimblecas.sat_dist;
import nimblecas.taskdag;

using namespace nimblecas;
using namespace nimblecas::sat_dist;

// Construct pigeonhole formula (6 variables, 3 pigeons, 2 holes)
// p(i, j) = 2*(i - 1) + j
const auto p = [](int i, int j) -> std::int64_t {
    return static_cast<std::int64_t>(2 * (i - 1) + j);
};
Cnf cnf{.num_vars = 6, .clauses = {}};
for (int i = 1; i <= 3; ++i) {
    cnf.clauses.push_back({p(i, 1), p(i, 2)}); // Pigeon i in hole 1 or 2
}
for (int j = 1; j <= 2; ++j) {
    for (int a = 1; a <= 3; ++a) {
        for (int b = a + 1; b <= 3; ++b) {
            cnf.clauses.push_back({-p(a, j), -p(b, j)}); // No two pigeons share hole j
        }
    }
}

// Execute across local worker threads. The public entry point is the factory, which
// returns a std::unique_ptr<Executor>.
auto exec = local_parallel_executor();

// Split across 2 cube variables (4 cubes total), unlimited conflict budget
auto res = solve_cubes_distributed(cnf, /*cube_vars=*/2, /*max_conflicts=*/0, *exec);
if (res) {
    // res->verdict == SatVerdict::unsatisfiable
    // Every cube was proven UNSAT, establishing an exact cluster-wide proof.
}
```

### Distributed portfolio solving with SGEE

```cpp
import nimblecas.core;
import nimblecas.sat;
import nimblecas.sat_dist;
import nimblecas.taskdag;
import nimblecas.taskdag_sgee;

using namespace nimblecas;
using namespace nimblecas::sat_dist;

// The registry must carry the shard ops on the coordinator AND on every remote worker:
// the worker looks the op up by id, so an unregistered op is a task that cannot run.
TaskRegistry reg;
register_ops(reg);

SgeeExecutorConfig cfg;
cfg.registry = &reg;
cfg.num_workers = 8;

// `port` and `results` are borrowed and must outlive the executor. The test suite drives
// this with FakeBrokerPort/InMemoryResultChannel; a real deployment supplies an SGEE broker.
FakeBrokerPort port;
InMemoryResultChannel results;
SgeeDistributedExecutor exec{cfg, port, results};

auto res = solve_portfolio_distributed(cnf, 0x12345ULL, /*shards=*/16, exec);
if (res) {
    // res->verdict is whatever the serial executor would have said, and the model is
    // byte-identical -- that equivalence is the contract, and the tests assert it.
}
```
