# `nimblecas.sat_compile` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/sat_compile/sat_compile.cppm`

An ahead-of-time Boolean satisfiability compiler that translates a fixed CNF formula into
high-performance, branchless source code for **C++23**, **CUDA C++**, and **Triton Python**.
While [`nimblecas.sat`](sat.md) provides dynamic in-process solvers over runtime data structures,
this module bakes the clauses of a formula into source constants and data-parallel bitwise operations
that a production compiler can fold, reassociate, schedule, and vectorise.

The compiler adheres strictly to the **honesty boundary** (Code Policy Rule 32):

- **Emits source text, not machine binary:** `nimblecas.sat_compile` generates complete,
  standalone source code strings. It is **not a JIT** compiler and emits neither PTX nor machine
  instructions directly. The emitted source code contains no hidden dependencies on NimbleCAS
  headers and can be dropped directly into downstream compilation pipelines. Standalone means it
  depends on nothing of NimbleCAS's — not that it depends on nothing: the C++ target opens with
  `import std;`, like the rest of the repository.
- **The emitted C++ follows the same code policy as the emitter:** `config/cpp_details.txt`
  applies to generated files too — `import std` (Rules 11/12/41), fixed-width types (Rule 48),
  `std::span` over the tables rather than raw pointers (Rules 3/24), trailing return types
  (Rule 31), `[[nodiscard]]` (Rule 10), and `std::countr_zero` rather than a compiler builtin.
  The CUDA and Triton targets are deliberately exempt: `nvcc` has no `import std`, and Python is
  not C++.
- **CDCL cannot be emitted as a data-parallel kernel:** Conflict-Driven Clause Learning (CDCL) —
  the algorithm that powers serious general-purpose solvers — cannot be compiled into a data-parallel
  SIMD or GPU kernel. The strength of CDCL stems from dynamic clause learning: each learned clause is
  derived by analysing the conflict that previous learned clauses led to. This constitutes an
  inherently sequential dependency chain with no static wavefront and no independent pieces to
  distribute across execution lanes. Claiming a "GPU CDCL solver" would be claiming something false;
  for hard combinatorial instances, `nimblecas.sat`'s single-core CPU `cdcl` outperforms any kernel
  emitted here.
- **What is genuinely data-parallel:** evaluating many assignments simultaneously is embarrassingly
  parallel. Under a fixed assignment, a clause is an OR of literals, and a formula is an AND of
  clauses. Both are bitwise operations that operate across any word width. Packing 64 assignments into
  a single `uint64_t` evaluates 64 candidate solutions with one bitwise AND. Packing eight such words
  into a 512-bit vector register (AVX-512) evaluates 512 assignments in a single CPU instruction.
  Launching thousands of thread blocks on a GPU evaluates tens of millions of assignments per kernel
  invocation.
- **Two strategies with distinct guarantees and boundaries:**
  - `exhaustive` — **Complete** (proves both satisfiability and unsatisfiability) and branch-free, but
    exponential ($2^n$). It enumerates the entire assignment space without clause learning or
    backtracking. It is excellent to about thirty variables and useless well before sixty, so it is
    the right choice only when a formula is small enough that a definite UNSAT is worth having. The
    representability ceiling `max_variables = 63` is the
    enumerator's limit, arising because assignments are indexed by unsigned 64-bit integers with
    `~0ULL` reserved as the "no solution" sentinel.
  - `walksat` — **Incomplete** stochastic local search (SLS). It cannot prove unsatisfiability: running
    out of flips yields `unknown` (`found == false`), never `unsatisfiable`. In return, it is completely
    unconstrained by the $2^n$ ceiling and scales to tens of thousands of variables and hundreds of
    thousands of clauses. A planted 3-SAT instance of 50,000 variables and 210,000 clauses (ratio 4.2)
    is solved and verified by the in-process oracle on its first walker; the suite covers this instance
    in `reference_walksat_solves_a_fifty_thousand_variable_instance`, and it measured 105-111 ms over
    five runs on the reference machine (`clang++-23`, `-O3 -march=x86-64-v3`, single walker). That
    figure is one machine's measurement, not a portable guarantee. Parallelism is across independent
    walkers, each conducting its own random walk from a deterministic seed.
- **Four published SLS variable-selection variants:** when `Strategy::walksat` is selected, all four
  variants share the outer walk structure (identifying an unsatisfied clause and selecting a variable
  within it to flip) and differ solely in the variable-choice heuristic:
  - `skc` — WalkSAT/SKC (Selman, Kautz, & Cohen, 1994). If any variable in the clause breaks zero
    currently satisfied clauses (a "freebie"), it is chosen immediately. Otherwise, with probability
    $1 - \text{noise}$, it greedily chooses the variable minimising the break count; with probability
    $\text{noise}$, it chooses a variable uniformly at random.
  - `probsat` — probSAT (Balint & Schöning, 2012; winner of the random track of the SAT Competition).
    It eliminates the greedy/random dichotomy: every variable in the unsatisfied clause receives a
    selection probability that decays exponentially with its break count, and a single random draw
    selects the variable.
    * *Integer halving trade-off:* the implementation assigns weights via powers of two
      (`weight = 1ULL << (40 - min(break, 40))`), corresponding to $c_b = 2$. The literature's optimal
      empirical setting for 3-SAT is near $c_b \in [2.3, 2.5]$, which requires floating-point
      computation. Floating-point transcendental functions can evaluate with slight discrepancies across
      architectures and compiler optimisation levels; using exact integer powers of two ensures
      bit-identical reproducibility across all host machines and compilers, trading a few per cent of
      solve rate for absolute determinism.
  - `novelty_plus` — Novelty+ (McAllester, Selman, & Kautz, 1997; Hoos, 1999). Scores variables by
    $\text{make} - \text{break}$ rather than break alone, preventing moves that repair many clauses from
    being discarded. If the highest-scoring variable is also the most recently flipped ("youngest"), the
    second-best variable is selected with probability $\text{noise}$. The `+` incorporates Hoos's
    stagnation fix: a mandatory 2% probability of a uniform random step, rendering the walk
    probabilistically approximately complete.
  - `adaptive_novelty_plus` — AdaptNovelty+ (Hoos, 2002). Dynamically adapts the noise parameter
    during search: noise increases when objective progress stagnates and decreases when the count of
    unsatisfied clauses improves, eliminating manual noise hyperparameter tuning.
- **Smallest assignment and lowest walker determinism:** every target returns the **smallest
  satisfying assignment** (numerically lowest integer assignment) for `exhaustive` search via trailing
  zero count and minimum reduction across blocks. For `walksat`, the result is the **lowest-indexed
  successful walker**. Both are strictly reproducible from the base seed.
- **Triton target role:** a random walk comprises dynamic control flow, data-dependent memory gathers,
  and variable-length unsatisfied clause sets — structures poorly matched to Triton's dense tensor
  model. Emitting a random walk in Triton would be slower than CUDA and unreadable. Instead, Triton
  receives a **Monte Carlo restart scorer**: drawing large batches of random assignments and scoring
  all clauses simultaneously via dense 2D reductions (`samples × clauses`). Portfolios use this dense
  scorer to identify high-quality starting points prior to sequential walking.

```cpp
import nimblecas.sat_compile;
```

Depends on [`core`](core.md) (`Result`, `MathError`) and [`sat`](sat.md) (`Cnf`, `verify_assignment`).

## Compilation options and targets

```cpp
enum class Target : std::uint8_t { cpp, cuda, triton };
enum class Strategy : std::uint8_t { exhaustive, walksat };
enum class SlsVariant : std::uint8_t { skc, probsat, novelty_plus, adaptive_novelty_plus };

inline constexpr std::size_t max_variables = 63;
inline constexpr std::uint64_t assignments_per_block = 64;
inline constexpr std::uint64_t blocks_per_group = 8;

struct CompileOptions {
    Target target{Target::cpp};
    Strategy strategy{Strategy::exhaustive};
    std::string entry{"formula"};
    bool emit_simd{true};
    bool emit_parallel{true};

    // WalkSAT options
    std::uint32_t walkers{64};
    std::uint64_t max_flips{100000};
    std::uint32_t noise_percent{50};
    std::uint64_t base_seed{0x9E3779B97F4A7C15ULL};
    SlsVariant variant{SlsVariant::skc};
};
```

| Field | Target / Strategy | Default | Role |
| :--- | :--- | :--- | :--- |
| `target` | All | `Target::cpp` | Source backend to emit: `cpp` (C++23), `cuda` (CUDA C++), or `triton` (Python/Triton). |
| `strategy` | All | `Strategy::exhaustive` | Solving methodology: complete `exhaustive` enumeration ($2^n$) or incomplete `walksat` local search. |
| `entry` | All | `"formula"` | Identifier prefix used for emitted solver functions, data structures, and kernels. |
| `emit_simd` | C++ (`exhaustive`) | `true` | When true, emits vectorised paths processing 8 blocks (512 assignments) per iteration. |
| `emit_parallel` | C++ (`exhaustive`) | `true` | When true, emits multithreaded partitioning using `std::jthread` reduced by minimum assignment. |
| `walkers` | WalkSAT | `64` | Number of concurrent independent random walks. Must be $> 0$. |
| `max_flips` | WalkSAT | `100000` | Flip budget per walker before aborting with `unknown`. |
| `noise_percent` | WalkSAT (`skc`, `novelty`) | `50` | Random walk probability expressed as an integer percentage in $[0, 100]$. |
| `base_seed` | WalkSAT | `0x9E3779...` | Base seed for the deterministic splitmix64 PRNG sequence per walker. |
| `variant` | WalkSAT | `SlsVariant::skc` | Variable selection rule (`skc`, `probsat`, `novelty_plus`, `adaptive_novelty_plus`). |

## API and in-process oracles

```cpp
[[nodiscard]] auto is_compilable_for(const Cnf& cnf, Strategy strategy) -> Result<void>;
[[nodiscard]] auto is_compilable(const Cnf& cnf) -> Result<void>;
[[nodiscard]] auto compile(const Cnf& cnf, const CompileOptions& opts) -> Result<std::string>;
[[nodiscard]] auto reference_solve(const Cnf& cnf, std::uint64_t max_blocks)
    -> Result<std::optional<std::uint64_t>>;
[[nodiscard]] auto model_of(std::uint64_t assignment, std::size_t num_vars) -> std::vector<bool>;

struct WalkResult {
    bool found{false};
    std::uint32_t walker{0};
    std::vector<bool> model;
    std::uint32_t walkers_run{0};
};

[[nodiscard]] auto reference_walksat(const Cnf& cnf, std::uint32_t walkers,
                                     std::uint64_t max_flips, std::uint32_t noise_percent,
                                     std::uint64_t base_seed) -> Result<WalkResult>;
```

| Function | Guarantee | Error conditions |
| :--- | :--- | :--- |
| `is_compilable_for` | Validates formula structure against the requested strategy. Enforces `num_vars <= 63` only when `strategy == Strategy::exhaustive`. | `MathError::domain_error` if `num_vars == 0`, a literal is `0`, a literal exceeds `num_vars`, or variables exceed 63 under `exhaustive`. |
| `is_compilable` | Shorthand for `is_compilable_for(cnf, Strategy::exhaustive)`. | Same as `is_compilable_for`. |
| `compile` | Emits complete, standalone source text for the requested target language and options. | `MathError::domain_error` if `is_compilable_for` fails, `entry` is not a valid C identifier, `walkers == 0`, or `noise_percent > 100`. |
| `reference_solve` | Exact algorithmic oracle for exhaustive solving. Computes the smallest satisfying assignment or `std::nullopt` (unsatisfiable). | `MathError::not_converged` if formula exceeds `max_blocks`; `domain_error` on malformed CNF. |
| `model_of` | Unpacks an assignment integer into a boolean vector where index `v - 1` corresponds to variable `v`. | Infallible. |
| `reference_walksat` | Exact algorithmic oracle for WalkSAT (SKC rule). Matches the random walk logic, tie-breaks, and walker ordering of emitted code. | `MathError::domain_error` on malformed CNF, zero walkers, or `noise_percent > 100`. |

## CLI driver: `tools/sat_compile.cpp`

The repository supplies a command-line tool to compile DIMACS CNF files directly:

```bash
sat_compile <file.cnf> [cpp|cuda|triton] [entry] [OPTIONS...]
```

Options include:
- `exhaustive` (default) or `walksat`
- `skc`, `probsat`, `novelty`, or `adaptnovelty`
- `nosimd` (disable vector extensions in C++)
- `noparallel` (disable multithreading in C++)
- `walkers=N` (number of stochastic walkers)
- `flips=N` (flip budget per walker)
- `noise=PERCENT` (random-walk probability percentage)

The tool reads the DIMACS formula, outputs the generated source code to `stdout`, and prints a diagnostic
summary (variable count, clause count, and reference solve verification) to `stderr`.

## Verifying the generated code

The test suite checks the emitted TEXT. A string match cannot tell whether that text is a
program, and for a long time nothing ran a compiler over it -- which is how an emitted CUDA
target that had never been compilable survived a green suite.

`scripts/verify-generated.sh` closes that: it emits every variant, compiles each one
(`clang++-23` with libc++ for C++, `nvcc` for CUDA, a Python parse for Triton) and RUNS the C++
ones against the reference answer the emitting tool prints. It is not part of `ctest`, because it
needs a toolchain a test binary has no business assuming. Run it after touching an emitter.


## Worked examples

### Compiling a small formula for exhaustive solving in C++23

```cpp
import std;
import nimblecas.core;
import nimblecas.sat;
import nimblecas.sat_compile;

using namespace nimblecas;
using namespace nimblecas::sat_compile;

// (x1 or x2) and (not x1 or x2) and (not x2 or x3)
// Requires x2=true, x3=true, x1 free. Smallest assignment: x1=0, x2=1, x3=1 (binary 110 = 6).
Cnf cnf{
    .num_vars = 3,
    .clauses = {{1, 2}, {-1, 2}, {-2, 3}}
};

CompileOptions opts{
    .target = Target::cpp,
    .strategy = Strategy::exhaustive,
    .entry = "puzzle"
};

// Check compilability and emit source
if (auto ok = is_compilable(cnf); ok) {
    auto src = compile(cnf, opts);
    if (src) {
        std::println("Generated C++ solver:\n{}", *src);
    }
}

// Compute reference solution directly
auto sol = reference_solve(cnf, 1024);
if (sol && sol->has_value()) {
    std::uint64_t assignment = **sol; // 6ULL
    std::vector<bool> model = model_of(assignment, cnf.num_vars);
    // model[0] == false (x1), model[1] == true (x2), model[2] == true (x3)
}
```

### Compiling large formulas with WalkSAT probSAT

```cpp
import std;
import nimblecas.core;
import nimblecas.sat;
import nimblecas.sat_compile;

using namespace nimblecas;
using namespace nimblecas::sat_compile;

// Large formula with thousands of variables
Cnf large_cnf = /* ... */;

CompileOptions opts{
    .target = Target::cuda,
    .strategy = Strategy::walksat,
    .entry = "benchmark",
    .walkers = 128,
    .max_flips = 500000,
    .base_seed = 0x12345678ULL,
    .variant = SlsVariant::probsat
};

auto cuda_source = compile(large_cnf, opts);
if (cuda_source) {
    // Standalone CUDA source ready to pass to nvcc
}
```

### Emitting a Triton Monte Carlo restart scorer

```cpp
CompileOptions triton_opts{
    .target = Target::triton,
    .strategy = Strategy::walksat,
    .entry = "dense_scorer"
};

auto py_source = compile(large_cnf, triton_opts);
// Emits Python/Triton code implementing batch scoring over sample assignments
```
