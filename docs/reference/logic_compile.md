# `nimblecas.logic_compile` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/logic_compile/logic_compile.cppm`

A Prolog-to-source compiler that translates deterministic, moded Horn clauses over
integers into high-performance source code for **C++23**, **CUDA**, and **Triton**.
While the general interpreter in [`nimblecas.logic`](logic.md) supports full first-order
terms, backtracking, and dynamic databases, many computationally demanding workloads
consist of deterministic numeric predicates invoked millions of times. This module
lowers that subset into native source code that can be compiled directly by production
compilers (`clang++-23`, `nvcc`, or Python/Triton).

The compiler adheres strictly to the **honesty boundary** (Code Policy Rule 32):

- **Emits source text, not binary:** `nimblecas.logic_compile` produces readable source
  code strings; it is **not a JIT** compiler and emits no PTX or machine binary directly.
- **Strict compilable subset:** compilation requires integer, moded, deterministic
  clauses. The caller must state each argument's mode (`input` or `output`) explicitly;
  modes are never guessed. Clause bodies may contain only `is/2` evaluations, arithmetic
  comparisons (`=:=`, `=\=`, `<`, `>`, `=<`, `>=`), `true`, cut `!`, and calls to other
  compilable predicates. Every variable must be bound before it is read.
- **Honest refusals without silent fallback:** any construct outside the compilable
  subset — compound terms, lists, floating-point literals, database updates
  (`assert`/`retract`), higher-order builtins (`findall`), or uninitialised output
  variables — is refused with `MathError::not_implemented` or `MathError::domain_error`.
  A program is never partially compiled, and unsupported constructs are never silently
  coerced.
- **Overflow-checked arithmetic:** in accordance with Rule 32, all emitted code guards
  arithmetic against overflow. Functions return a `bool` status that evaluates to `false`
  upon failure or when an arithmetic operation exceeds the representation limit,
  faithfully mirroring the interpreter's error semantics.

```cpp
import nimblecas.logic_compile;
```

Depends on [`core`](core.md) (`Result`, `MathError`) and [`logic`](logic.md)
(`Program`, `Clause`, `Term`).

## Compilation targets, widths, and styles

```cpp
enum class Target : std::uint8_t { cpp, cuda, triton };
enum class ArgMode : std::uint8_t { input, output };
enum class Width  : std::uint8_t { bits64, bits128, arbitrary };
enum class Style  : std::uint8_t { deterministic, continuation };

struct PredicateSignature {
    std::string name;
    std::vector<ArgMode> modes;

    [[nodiscard]] auto arity() const -> std::size_t { return modes.size(); }
};

struct CompileOptions {
    Target target{Target::cpp};
    Width width{Width::bits64};
    Style style{Style::deterministic};
    bool tail_call_optimise{true};
    bool emit_batch{false};
};
```

### Targets

- `Target::cpp`: emits modern C++23 source code adhering to repository standards
  (trailing return types, `[[nodiscard]]`, no exceptions, overflow-checked). Compatible
  with `clang++-23`.
- `Target::cuda`: emits a `__device__` function and a companion `__global__` batch kernel
  for compilation with `nvcc`.
- `Target::triton`: emits a `@triton.jit` kernel operating elementwise over `int64`
  tensors. Triton kernels are flat block-level operations without a call stack;
  consequently, **recursion and predicate calls are refused** with
  `MathError::not_implemented`.

### Integer widths

- `Width::bits64`: computes using `std::int64_t` with checked arithmetic builtins.
  Supported by all targets.
- `Width::bits128`: computes using `__int128`. In CUDA, 128-bit operations are synthesised
  from 64-bit primitives. Refused by Triton (`MathError::not_implemented`) because Triton's
  type lattice tops out at `tl.int64`.
- `Width::arbitrary`: computes using [`nimblecas.bigint`](bigint.md) (`BigInt`). Addition,
  subtraction, and multiplication never overflow. Supported only by `Target::cpp`.
  Refused by CUDA (`MathError::not_implemented`) because `BigInt` dynamically allocates on
  the host heap, which GPU device functions cannot do.

### Evaluation styles

- `Style::deterministic`: compiles a predicate to a function computing its single
  (first) solution.
- `Style::continuation`: compiles to continuation-passing style (CPS), taking a callback
  invoked once per solution to handle non-deterministic predicates without truncation.
  Supported only on `Target::cpp`; refused on CUDA and Triton (`MathError::not_implemented`)
  because nested lambda continuations cannot be represented on device architectures.

### Optimisation and batching flags

- `tail_call_optimise`: replaces self-tail-recursive calls with an iterative loop. Enabled
  by default.
- `emit_batch`: when targeting C++, emits a multi-input batch runner alongside vectorised
  SIMD arithmetic kernels.

## API functions

```cpp
[[nodiscard]] auto compile(const Program& program, const PredicateSignature& entry,
                           const CompileOptions& options) -> Result<std::string>;

[[nodiscard]] auto compile(const Program& program, const PredicateSignature& entry,
                           Target target) -> Result<std::string>;

[[nodiscard]] auto is_compilable(const Program& program, const PredicateSignature& entry,
                                 Target target) -> Result<void>;
```

| Function | Behaviour |
| :--- | :--- |
| `compile(program, entry, options)` | Translates `entry` and its transitive dependencies from `program` into source code conforming to `options`. Refuses unsupported constructs with an honest `MathError`. |
| `compile(program, entry, target)` | Convenience overload using default `CompileOptions` (64-bit, deterministic, tail-call optimised) for `target`. |
| `is_compilable(program, entry, target)` | Validates whether `entry` can be compiled for `target` without generating source text. Returns `Result<void>` with the identical error `compile` would produce. |

## Command-line driver: `prolog_compile`

The repository includes a command-line tool `tools/prolog_compile.cpp` built into
`build/prolog_compile`:

```bash
prolog_compile <file.pl> <entry-name> <modes:i|o...> [cpp|cuda|triton] [64|128|big] [det|cps] [batch]
```

### Arguments

- `<file.pl>`: path to the input Prolog source file.
- `<entry-name>`: name of the top-level predicate to compile.
- `<modes>`: argument modes specified as a character sequence of `i` (input) and `o` (output).
  For example, `fib(+N, -Result)` is specified as `io`.
- Target (optional): `cpp` (default), `cuda`, or `triton`.
- Width (optional): `64` (default), `128`, or `big`.
- Style (optional): `det` (default) or `cps`.
- Batch (optional): `batch` or `simd` to emit batch evaluation functions.

The driver writes generated code to `stdout` and diagnostic messages to `stderr`.

### Which example goes with which target

`examples/fib.pl` and `examples/countdown.pl` are recursive, so they reach `cpp` and `cuda` but
are refused by `triton`, which has no call stack. `examples/poly.pl` is the straight-line
arithmetic the Triton target accepts:

```bash
prolog_compile examples/fib.pl  fib       io cpp    64   # or cuda
prolog_compile examples/poly.pl eval_poly io triton 64
```

The Triton kernel takes one pointer per argument plus an `ok_ptr`, a length and a `BLOCK`
constexpr, and reports per lane through `ok` whether the predicate succeeded there.

## Error model

| Condition | Error |
| :--- | :--- |
| Non-integer types, floats, lists, compounds, or unsupported builtins | `MathError::not_implemented` |
| Recursive calls or sub-predicate calls when target is `Target::triton` | `MathError::not_implemented` |
| `Width::bits128` or `Width::arbitrary` requested for `Target::triton` | `MathError::not_implemented` |
| `Width::arbitrary` requested for `Target::cuda` | `MathError::not_implemented` |
| `Style::continuation` requested for `Target::cuda` or `Target::triton` | `MathError::not_implemented` |
| Entry predicate not found, arity mismatch, or output variable never bound | `MathError::domain_error` |
| Syntax error while parsing the Prolog source file | `MathError::syntax_error` |

## Verifying the generated code

The test suite checks the emitted TEXT. A string match cannot tell whether that text is a
program, and for a long time nothing ran a compiler over it -- which is how an emitted CUDA
target that had never been compilable survived a green suite.

`scripts/verify-generated.sh` closes that: it emits every variant, compiles each one
(`clang++-23` with libc++ for C++, `nvcc` for CUDA, the Triton JIT for Triton) and RUNS every
target it can -- the C++ emissions and the Triton kernels -- against the reference answer the
emitting tool prints. It is not part of `ctest`, because it needs a toolchain a test binary has
no business assuming. Run it after touching an emitter.

A Triton kernel that parses as Python is not thereby a kernel: `shape = offs.shape` and a bare
`INT64_MAX = 9223372036854775807` at module scope both parse, and each on its own makes the
Triton compiler refuse the kernel -- a tensor shape must be a tuple of compile-time integers,
and a `@triton.jit` function cannot read a global that is not a `tl.constexpr`. Both were in the
emitted output until the script started compiling it. Where there is no Triton or no CUDA device
the script falls back to the parse and says so, rather than reporting a pass for something it
did not test.

## Worked examples

### Compiling a recursive factorial predicate

```cpp
import nimblecas.logic_compile;
import nimblecas.logic_parser;
import nimblecas.logic;
import nimblecas.core;

using namespace nimblecas;
using namespace nimblecas::logic_compile;

// 1. Parse a tail-recursive factorial program
const std::string pl_source = R"(
    fact_acc(0, Acc, Acc).
    fact_acc(N, Acc, Result) :-
        N > 0,
        N1 is N - 1,
        Acc1 is Acc * N,
        fact_acc(N1, Acc1, Result).

    fact(N, Result) :- fact_acc(N, 1, Result).
)";

auto program = logic_parser::parse_program(pl_source).value();

// 2. Define predicate signature: fact(+N, -Result)
PredicateSignature entry{
    .name = "fact",
    .modes = {ArgMode::input, ArgMode::output}
};

// 3. Verify compilability before generating code
auto check = is_compilable(program, entry, Target::cpp);
if (check.has_value()) {
    // 4. Compile to C++23 with tail-call optimisation
    CompileOptions opts{
        .target = Target::cpp,
        .width = Width::bits64,
        .style = Style::deterministic,
        .tail_call_optimise = true,
        .emit_batch = false
    };

    std::string cpp_source = compile(program, entry, opts).value();
    // cpp_source contains generated C++ functions with while-loop tail recursion
}
```

### Compiling to a CUDA batch kernel

```cpp
// Target CUDA for parallel execution across a batch
CompileOptions cuda_opts{
    .target = Target::cuda,
    .width = Width::bits64,
    .style = Style::deterministic,
    .tail_call_optimise = true
};

std::string cuda_source = compile(program, entry, cuda_opts).value();
// cuda_source contains a __device__ function and a __global__ batch kernel
```

## See also

- [`nimblecas.logic`](logic.md) — the Prolog resolution engine.
- [`nimblecas.logic_parser`](logic_parser.md) — parsing Prolog programs and queries.
- [`nimblecas.logic_index`](logic_index.md) — batched first-argument clause indexing.
- [`nimblecas.bigint`](bigint.md) — arbitrary-precision arithmetic backing `Width::arbitrary`.
- [Documentation hub](../Index.md)
