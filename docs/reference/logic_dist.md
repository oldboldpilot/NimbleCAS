# `nimblecas.logic_dist` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/logic_dist/logic_dist.cppm`

Distributed SLD resolution for Horn clause logic programs. While [`nimblecas.logic`](logic.md)
fans OR-branches across CPU threads on a single node via `solve_or_parallel`, this module
fans those same branches across **processes and cluster nodes** by expressing the search
as a dependency-free [`nimblecas.taskdag`](taskdag.md) `TaskGraph`. The graph can be
dispatched to any conforming `Executor`: `serial_executor`, `local_parallel_executor`, or
the SGEE-backed broker in [`nimblecas.taskdag_sgee`](taskdag_sgee.md).

The module is governed by a strict **honesty boundary** (Code Policy Rule 32):

- **The equivalence contract:** distributed execution guarantees that the answers
  returned **do not reveal which executor ran them**. Solving a query with `solve_distributed`
  yields the identical set and ordering of substitutions as `nimblecas::logic::solve`.
- **One-level fan-out limit:** each shard task executes a complete, self-contained
  sub-derivation in its own worker process (depth-first with backtracking, up to exhaustion
  or step budget). A `TaskFn` receives only opaque payload bytes and possesses no handle
  to the broker; a worker **cannot enqueue new work dynamically** as it discovers branch points.
  Consequently, parallelism is strictly bounded by the number of candidate clauses at the
  top-level goal. A goal matching two clauses dispatches to two workers, regardless of the
  depth of the subtrees beneath them; a single recursive rule dispatches to one worker.
- **Refuses nothing:** programs with a cut (`!`) or side-effecting database updates
  (`assert`/`retract`) cannot be soundly distributed because speculative workers cannot
  retract side effects or enforce prune barriers across nodes. Rather than failing or
  returning partial answers, `solve_distributed` **delegates such programs to the serial
  solver**. Callers rely on receiving correct answers; delegation preserves that contract
  where an outright refusal would break it.
- **Prolog text wire format and variable preservation:** shards and answers cross the
  wire as Prolog text, serialised and deserialised via [`nimblecas.logic_parser`](logic_parser.md).
  Text payloads are human-readable, testable, and utilise the parser's proven round-trip
  guarantee. Crucially, logic variables are encoded explicitly as `v(Name, Generation)`
  rather than plain variable names: standard Prolog text syntax drops the internal rename
  generation, which would cause distinct variables across activations to falsely unify upon
  deserialisation.

```cpp
import nimblecas.logic_dist;
```

Depends on [`core`](core.md) (`Result`, `MathError`, `Payload`), [`logic`](logic.md)
(`Program`, `Clause`, `Term`, `Substitution`), [`logic_parser`](logic_parser.md)
(`parse_term`, `to_source`), and [`taskdag`](taskdag.md) (`TaskGraph`, `TaskRegistry`,
`Executor`).

## Shards and the wire format

A `Shard` encapsulates a single OR-branch: a program, a goal conjunction, a committed
clause index at the root goal, and an answer limit.

```cpp
inline constexpr std::string_view shard_op_id = "nimblecas.logic.sld_shard/v1";

struct Shard {
    Program program;
    std::vector<Term> goals;
    std::size_t clause_index{0};
    std::uint64_t max_solutions{0};  // 0 = all solutions within budget
};

[[nodiscard]] auto encode_shard(const Shard& shard) -> Result<Payload>;
[[nodiscard]] auto decode_shard(std::span<const std::byte> bytes) -> Result<Shard>;

[[nodiscard]] auto encode_answers(const std::vector<Substitution>& answers) -> Result<Payload>;
[[nodiscard]] auto decode_answers(std::span<const std::byte> bytes)
    -> Result<std::vector<Substitution>>;
```

### Why variables are encoded as `v(Name, Generation)`

In `nimblecas.logic`, standardising a clause apart stamps a monotonic `generation` integer
on each variable to distinguish it from variables of the same name in earlier or later
activations. Plain Prolog syntax renders variables purely by name (e.g. `X_1`), which on
re-parsing creates a variable with generation 0. In a distributed setting, this would
silently collapse distinct execution branches. The wire format therefore serialises
terms with explicit structural tags:

- Variables: `v(Name, Generation)`
- Atoms: `a(Name)`
- Integers: `i(Value)`
- Compounds: `c(Functor, [Args...])`

This ensures that variable identities are preserved bit for bit across process boundaries.

## Distributed execution API

```cpp
[[nodiscard]] auto register_ops(TaskRegistry& reg) -> Result<void>;

[[nodiscard]] auto is_distributable(const Program& program, const std::vector<Term>& goals)
    -> bool;

[[nodiscard]] auto build_shard_graph(const TaskRegistry& reg, const Program& program,
                                     const std::vector<Term>& goals,
                                     std::uint64_t max_solutions) -> Result<TaskGraph>;

[[nodiscard]] auto solve_distributed(const Program& program, const std::vector<Term>& goals,
                                     std::uint64_t max_solutions, Executor& exec)
    -> Result<std::vector<Substitution>>;
```

| Function | Behaviour |
| :--- | :--- |
| `register_ops` | Registers the `shard_op_id` (`"nimblecas.logic.sld_shard/v1"`) handler in the provided `TaskRegistry`. Must be called on both the coordinator node and all remote worker processes. |
| `is_distributable` | Returns `true` if the query can be decomposed across multiple workers. Returns `false` if the leading goal is a builtin/control construct, or if the program contains cuts (`!`) or database updates. |
| `build_shard_graph` | Constructs a static, one-level `TaskGraph` containing one independent named task per clause matching the initial goal. Each task carries its encoded `Shard` as a bound literal payload. Returns `MathError::domain_error` if the query is not distributable. |
| `solve_distributed` | Executes the query across `exec`. If distributable, it constructs the shard graph, runs it across the executor, and merges the per-shard answer substitutions in program clause order. If not distributable, it delegates directly to serial `solve`. |

## Error model

| Condition | Error |
| :--- | :--- |
| Empty goal list or malformed clause in `encode_shard` | `MathError::domain_error` |
| Variable generation exceeds signed 64-bit integer limits | `MathError::overflow` |
| Malformed payload bytes in `decode_shard` or `decode_answers` | `MathError::syntax_error` |
| Attempting to build a shard graph for a non-distributable program via `build_shard_graph` | `MathError::domain_error` |
| Executor failure or node communication abort | Error propagated from `Executor::run` |

## Worked examples

```cpp
import nimblecas.logic_dist;
import nimblecas.logic_parser;
import nimblecas.logic;
import nimblecas.taskdag;
import nimblecas.core;

using namespace nimblecas;
using namespace nimblecas::logic_dist;

// 1. Register distributed operations on the task registry
TaskRegistry registry;
register_ops(registry).value();

// 2. Parse a program with multiple clauses at the top goal
const std::string prog_src = R"(
    color(red).
    color(green).
    color(blue).
    color(yellow).

    valid_pair(C1, C2) :- color(C1), color(C2), C1 \= C2.
)";

auto program = logic_parser::parse_program(prog_src).value();
auto query = logic_parser::parse_query("?- color(X).").value();

// 3. Solve over a local parallel executor (or distributed SGEE executor)
auto exec = local_parallel_executor();
auto answers = solve_distributed(program, query, 0, exec).value();

// Returns 4 substitutions: X -> red, green, blue, yellow
// The answers match serial solve(program, query) exactly in clause order.

// 4. Checking distributability
bool can_dist = is_distributable(program, query);  // true

// A program with a cut is not distributable, but solve_distributed delegates
// honestly to the serial solver rather than failing:
auto cut_prog = logic_parser::parse_program("p(1) :- !. p(2).").value();
auto cut_q = logic_parser::parse_query("?- p(X).").value();

bool cut_dist = is_distributable(cut_prog, cut_q);  // false
auto cut_ans = solve_distributed(cut_prog, cut_q, 0, exec).value();
// Returns single answer X -> 1, computed serially
```

## See also

- [`nimblecas.logic`](logic.md) — serial and OR-parallel SLD resolution.
- [`nimblecas.logic_parser`](logic_parser.md) — reader and writer backing the text wire format.
- [`nimblecas.taskdag`](taskdag.md) — task graphs and executor interfaces.
- [`nimblecas.taskdag_sgee`](taskdag_sgee.md) — multi-node distributed executor over SGEE.
- [Documentation hub](../Index.md)
