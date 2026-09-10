# `nimblecas.logic_index` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/logic_index/logic_index.cppm`

A high-performance first-argument clause indexing kernel for Prolog SLD resolution.
While the AND-side of logic resolution is an irregular, data-dependent backtracking walk
that is poorly suited to SIMD or SIMT vectorisation, **first-argument clause indexing**
is completely regular: deciding which candidate clauses of a predicate could possibly unify
with a goal compares a 64-bit integer key against a contiguous array of clause keys.
`nimblecas.logic_index` implements this batched probe across goals, producing a dense
bitmask of candidate clauses per goal.

The module operates under a rigorous **honesty boundary** (Code Policy Rule 32):

- **The CPU path is authoritative:** the scalar loop defines the mathematical semantics
  of candidate selection. The AVX2 and AVX-512 vectorised paths must agree with the scalar
  loop **bit for bit**, and tests assert exact identity rather than verifying against
  divergent tolerances.
- **Key semantics and zero-as-wildcard:** key `0` is reserved strictly for **unknown**,
  matching everything. A goal whose first argument is an unbound variable receives key `0`
  and must consider every clause; a clause whose head has no arguments (an atom) or whose
  first argument is a variable receives key `0` and must be tested against every goal.
  Two non-zero keys are candidates if and only if they are equal:
  `goal_key == 0 || clause_key == 0 || goal_key == clause_key`.
- **Sound filtering without false negatives:** the filter may only rule out a clause
  when it can definitively prove that head unification is impossible. A false positive
  merely causes an unsuccessful unification attempt, whereas a false negative would
  silently lose an answer. Therefore, any ambiguity resolves in favour of retaining the
  candidate.
- **Hardware-accelerated CUDA mirror:** an optional GPU kernel in [`nimblecas.gpu`](gpu.md)
  mirrors this operation on CUDA hardware. The GPU path is held to the same standard: it is
  verified **bit-identical** to the CPU path. This GPU mirror is gated behind the build flag
  `-DNIMBLECAS_CUDA=ON`.

```cpp
import nimblecas.logic_index;
```

Depends on [`core`](core.md) (`Result`, `MathError`), [`logic`](logic.md) (`Term`, `Program`),
and [`parallel`](parallel.md) (`transform_index`, parallelising across independent goal rows).
Optionally mirrors to [`gpu`](gpu.md) when built with CUDA support.

## Instruction set architecture dispatch

The CPU implementation dynamically inspects host CPU capabilities at run time to dispatch
the widest supported SIMD path via GCC/Clang `[[gnu::target(...)]]` attributes, avoiding
global compiler flags:

| ISA | Enumerator | Width | Operation |
| :--- | :--- | :--- | :--- |
| Scalar | `ProbeIsa::scalar` | 64-bit integer | Reference loop; 1 key comparison per iteration. |
| AVX2 | `ProbeIsa::avx2` | 256-bit vector | `_mm256_cmpeq_epi64` comparing 4 keys concurrently; sign bits extracted to bit positions. |
| AVX-512 | `ProbeIsa::avx512` | 512-bit vector | `_mm512_cmpeq_epu64_mask` comparing 8 keys concurrently directly into an 8-bit mask register. |

```cpp
enum class ProbeIsa : std::uint8_t { scalar, avx2, avx512 };

[[nodiscard]] constexpr auto to_string_view(ProbeIsa isa) noexcept -> std::string_view;
[[nodiscard]] auto detect_probe_isa() noexcept -> ProbeIsa;
```

`detect_probe_isa()` detects the maximum instruction set available on the running processor.

## Key generation and bitmask layout

Terms used as first arguments are hashed into 64-bit non-zero integers using a deterministic,
dependency-free 64-bit FNV-1a hash over their principal functor and arity:

```cpp
[[nodiscard]] auto first_arg_key(const Term& t) -> std::uint64_t;
[[nodiscard]] auto clause_index_keys(const Program& program) -> std::vector<std::uint64_t>;
[[nodiscard]] constexpr auto probe_row_words(std::size_t clause_count) noexcept -> std::size_t;
```

- `first_arg_key(t)` returns `0` if `t` is a logic variable. For any other term kind, it
  computes a non-zero hash. If the hash function produces `0`, it is nudged to `1` to avoid
  spurious wildcard collisions.
- `clause_index_keys(program)` extracts the index key of each clause's head first argument
  in program order. Clauses with atom heads (0 arguments) receive key `0`.
- `probe_row_words(clause_count)` computes `(clause_count + 63) / 64`, the number of
  64-bit words required to represent candidate bitmasks for one goal.

## Batched probing

```cpp
[[nodiscard]] auto index_probe_batch(std::span<const std::uint64_t> clause_keys,
                                     std::span<const std::uint64_t> goal_keys,
                                     std::span<std::uint64_t> out_words) -> Result<void>;

[[nodiscard]] auto index_probe_batch_with(ProbeIsa isa,
                                          std::span<const std::uint64_t> clause_keys,
                                          std::span<const std::uint64_t> goal_keys,
                                          std::span<std::uint64_t> out_words) -> Result<void>;

[[nodiscard]] auto decode_candidates(std::span<const std::uint64_t> row,
                                     std::size_t clause_count) -> std::vector<std::size_t>;
```

| Function | Behaviour |
| :--- | :--- |
| `index_probe_batch` | Evaluates candidate clauses for every goal in `goal_keys` against `clause_keys`. For goal `g` and clause `c`, bit `c` of row `g` is set to `1` if clause `c` is a candidate. Goal rows are evaluated in parallel across threads using `nimblecas.parallel::transform_index`. |
| `index_probe_batch_with` | Executes the probe forcing a specific instruction set (`scalar`, `avx2`, or `avx512`). Allows test suites to verify that vectorised paths reproduce the scalar result bit for bit. Requesting an ISA unsupported by the host processor returns `MathError::domain_error`, never a silent fallback. |
| `decode_candidates` | Extracts candidate clause indices in ascending order from a single goal's bitmask row. |

## The CUDA GPU mirror

When the repository is built with `-DNIMBLECAS_CUDA=ON`, the module [`nimblecas.gpu`](gpu.md)
exports a GPU-accelerated mirror of `index_probe_batch`:

```cpp
// In namespace nimblecas::gpu:
[[nodiscard]] auto index_probe_batch(std::span<const std::uint64_t> clause_keys,
                                     std::span<const std::uint64_t> goal_keys)
    -> Result<std::vector<std::uint64_t>>;
```

The GPU kernel computes the exact same FNV-1a candidate logic across CUDA threads and blocks.
Outputs are verified bit-identical to the CPU scalar implementation. If no CUDA device is
available or driver launch fails, it returns `MathError::gpu_error`. If input spans exceed
signed 32-bit integer limits, it returns `MathError::overflow`. When `-DNIMBLECAS_CUDA=OFF`,
GPU symbols are not compiled.

## Error model

| Condition | Error |
| :--- | :--- |
| `out_words.size() != goal_keys.size() * probe_row_words(clause_keys.size())` | `MathError::domain_error` |
| `index_probe_batch_with` called with an ISA not supported by the host CPU | `MathError::domain_error` |
| GPU kernel invoked when no CUDA device is present or kernel launch fails (`gpu`) | `MathError::gpu_error` |
| Key buffer lengths exceed integer bounds (`gpu`) | `MathError::overflow` |

## Worked examples

```cpp
import nimblecas.logic_index;
import nimblecas.logic_parser;
import nimblecas.logic;
import nimblecas.core;

using namespace nimblecas;
using namespace nimblecas::logic_index;

// Define a program with several clauses indexed on their first argument
const std::string source = R"(
    p(a, 1).
    p(b, 2).
    p(X, 3).
    p(a, 4).
    p(f(z), 5).
)";

auto prog = logic_parser::parse_program(source).value();
std::vector<std::uint64_t> clause_keys = clause_index_keys(prog);

// Prepare goal keys:
// Goal 0: p(a, Ans)  -> non-zero hash for 'a'
// Goal 1: p(c, Ans)  -> non-zero hash for 'c'
// Goal 2: p(Var, Ans)-> 0 (unbound variable wildcard)
std::vector<std::uint64_t> goal_keys = {
    first_arg_key(make_atom("a")),
    first_arg_key(make_atom("c")),
    first_arg_key(make_var("Var"))
};

const std::size_t words_per_row = probe_row_words(clause_keys.size());
std::vector<std::uint64_t> out_words(goal_keys.size() * words_per_row, 0U);

// Run the batched probe (automatically choosing AVX-512, AVX2, or scalar)
index_probe_batch(clause_keys, goal_keys, out_words).value();

// Decode candidates for Goal 0: p(a, Ans)
// Matches clause 0: p(a, 1) [equal 'a']
// Matches clause 2: p(X, 3) [clause head is variable -> wildcard 0]
// Matches clause 3: p(a, 4) [equal 'a']
auto cands_0 = decode_candidates(
    std::span(out_words.data(), words_per_row), clause_keys.size());
// cands_0 contains {0, 2, 3}

// Decode candidates for Goal 1: p(c, Ans)
// Matches only clause 2: p(X, 3) because neither 'a', 'b', nor 'f(z)' match 'c'
auto cands_1 = decode_candidates(
    std::span(out_words.data() + words_per_row, words_per_row), clause_keys.size());
// cands_1 contains {2}

// Decode candidates for Goal 2: p(Var, Ans)
// Wildcard matches all 5 clauses
auto cands_2 = decode_candidates(
    std::span(out_words.data() + 2 * words_per_row, words_per_row), clause_keys.size());
// cands_2 contains {0, 1, 2, 3, 4}
```

## See also

- [`nimblecas.logic`](logic.md) — the Prolog resolution and unification engine.
- [`nimblecas.logic_parser`](logic_parser.md) — parsing programs to clauses and terms.
- [`nimblecas.gpu`](gpu.md) — the CUDA GPU kernel mirror of the batched probe.
- [`nimblecas.parallel`](parallel.md) — the fork–join substrate driving multi-goal batching.
- [Documentation hub](../Index.md)
