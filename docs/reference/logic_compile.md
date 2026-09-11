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
enum class Width  : std::uint8_t { bits64, bits128, arbitrary, decimal128 };
enum class Style  : std::uint8_t { deterministic, continuation };

struct PredicateSignature {
    std::string name;
    std::vector<ArgMode> modes;

    [[nodiscard]] auto arity() const -> std::size_t { return modes.size(); }
};

struct CompileOptions {
    Target target{Target::cpp};
    Width width{Width::bits64};
    std::int32_t decimal_scale{2};   // digits after the point when width is decimal128
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

### `Width::decimal128` — fixed-point decimal, for money

A 128-bit **fixed-point decimal** at a scale fixed when you compile. A value is an integer
count of **minor units**: at `decimal_scale = 2`, the integer `u` denotes `u / 100` exactly.
Nothing is a binary float, so nothing is approximately `0.1`. Supported by `Target::cpp` and
`Target::cuda`; refused by Triton, whose lattice stops at `tl.int64`.

**Literals in the Prolog source are minor units.** The reader has no decimal literal, so
`Price is 1999` means 19.99 at scale 2 — the source is written in the unit the representation
uses, and the generated file states the scale in `nc_scale_digits` and `nc_scale_v`.

**Exact or refused, never rounded.** `+`, `-`, unary minus and every comparison are the
integer operations unchanged, which is the point of fixed point. Multiplication and division
rescale, and an answer the scale cannot hold exactly is reported rather than rounded:

| Expression at scale 2 | Result |
| :--- | :--- |
| `19.99 * 3.00` | `59.97` |
| `100.00 * 0.10` | `10.00` |
| `0.01 * 0.01` | refused — `0.0001` needs four digits |
| `10.00 // 4.00` | `2.50` |
| `10.00 // 3.00` | refused — not representable at two digits |

There is deliberately **no rounding mode**. A rounding rule chosen inside a code generator is
a wrong answer no caller asked for; when you need one, quantise with
[`nimblecas.currency`](currency.md), whose `convert()` takes a scale and a `Rounding`
explicitly. `//` and `div` are the same function here, because they differ only in how they
round a division that does not come out even and this width refuses those instead.

`mod` and `rem` are **refused** (`MathError::not_implemented`). Their integer meaning does not
survive scaling — the remainder of one amount divided by another differs depending on whether
the quotient is taken as an integer count or as a decimal — and picking one silently is
exactly the failure Rule 32 exists to prevent.

`decimal_scale` must be in `0..18`; anything else is a `MathError::domain_error`. The
generated file also carries `nc_to_string`, which renders a scaled integer as the decimal it
means (`1999` as `"19.99"`, `-1999` as `"-19.99"`, `0` as `"0.00"`). It builds a
`std::string`, so it is emitted for `Target::cpp` only.

A worked example is `examples/money.pl`.

### Evaluation styles

- `Style::deterministic`: compiles a predicate to a function computing its single
  (first) solution.
- `Style::continuation`: compiles to continuation-passing style (CPS), taking a callback
  invoked once per solution to handle non-deterministic predicates without truncation.
  Supported only on `Target::cpp`; refused on CUDA and Triton (`MathError::not_implemented`)
  because nested lambda continuations cannot be represented on device architectures.

  The public entry is a template on the caller's callable, but the body that actually
  enumerates takes an **erased continuation view** (`nc_cont_<n>`, one per output arity) and is
  an ordinary function. That is what lets a RECURSIVE predicate compile at all: when the body
  was a template, each level of the enumeration handed itself a fresh lambda type and demanded
  a fresh instantiation, so `between/3` — the very shape CPS exists for — could not be built.
  The view is a pointer to the callable plus a pointer to a function that invokes it: no
  allocation, and so nothing that can throw.

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
prolog_compile <file.pl> <entry-name> <modes:i|o...> [cpp|cuda|triton] [64|128|big|dec[:N]] [det|cps] [batch]
```

### Arguments

- `<file.pl>`: path to the input Prolog source file.
- `<entry-name>`: name of the top-level predicate to compile.
- `<modes>`: argument modes specified as a character sequence of `i` (input) and `o` (output).
  For example, `fib(+N, -Result)` is specified as `io`.
- Target (optional): `cpp` (default), `cuda`, or `triton`.
- Width (optional): `64` (default), `128`, `big`, or `dec` / `dec:N` for a fixed-point
  decimal with N digits after the point (`dec` is two).
- Style (optional): `det` (default) or `cps`.
- Batch (optional): `batch` or `simd` to emit batch evaluation functions.

The driver writes generated code to `stdout` and diagnostic messages to `stderr`.

### Which example goes with which target

`examples/fib.pl` and `examples/countdown.pl` are recursive, so they reach `cpp` and `cuda` but
are refused by `triton`, which has no call stack to re-enter. A call that is not recursive needs
no stack — the Triton target INLINES it — so `triton` takes any program whose call graph is
acyclic. `examples/poly.pl` is the straight-line arithmetic; `examples/fee.pl` is the call:

```bash
prolog_compile examples/fib.pl      fib       io   cpp    64      # or cuda
prolog_compile examples/poly.pl     eval_poly io   triton 64      # straight-line
prolog_compile examples/fee.pl      total     iiio triton 64      # a call, inlined
prolog_compile examples/divmod.pl   divs      iioooo triton 64    # the four divisions
prolog_compile examples/countdown.pl count     iio  cpp    64      # a self tail call, as a loop
prolog_compile examples/fact.pl     fact_acc  iio  cpp    128     # past 2^63 at 21!
prolog_compile examples/between.pl  between   iio  cpp    64 cps  # nondeterministic
prolog_compile examples/money.pl    settle    iiio cpp    dec     # fixed-point money
```

| Example | What it is there to exercise |
| :--- | :--- |
| `fib.pl` | ordinary recursion, every target and width |
| `countdown.pl` | a self tail call, which the compiler rewrites to a loop |
| `poly.pl` | straight-line arithmetic, on every target |
| `fee.pl` | a call to a multi-clause, partial predicate, which Triton inlines |
| `divmod.pl` | `//`, `div`, `rem` and `mod`, which round two different ways |
| `fact.pl` | the width boundaries: 20!/21! at 64 bits, 33!/34! at 128 |
| `between.pl` | a nondeterministic predicate, for `Style::continuation` |
| `money.pl` | `Width::decimal128`, including the refusals |

The Triton kernel takes one pointer per argument plus an `ok_ptr`, a length and a `BLOCK`
constexpr, and reports per lane through `ok` whether the predicate succeeded there.

### How a call survives having no call stack

Each call site is expanded in place under a fresh `f<n>_` prefix: the callee's clauses are
emitted with the caller's values as their inputs and the caller's variables as their outputs,
and they pick a clause with their own `f<n>_done` mask exactly as the entry predicate does.
The prefix is what keeps the frames apart — a caller and a callee may both write `X`, and a
clause assigns its variables unconditionally because masking happens when outputs are selected,
not when they are computed. Whether the call succeeded comes back as `f<n>_done`, which is
conjoined into the calling clause's guard, so a lane whose call failed cannot be reported as a
success. Inlining is duplication, so the number of call sites expanded is capped; past the cap
the request is refused rather than turned into a kernel that will not finish compiling.

Recursion is the one shape this cannot reach, and it stays a `not_implemented` — direct or
mutual, since the check is for a CYCLE in the call graph and not for a predicate naming
itself.

### The four divisions on Triton

Triton's `//` and `%` truncate toward zero, the way C does and the way Python does not. Prolog's
`//` and `rem` truncate too, so those lower straight through; its `div` and `mod` **floor**, and
the kernel emits the same correction the C++ and CUDA targets apply. The two roundings differ
only when the signs differ and the division is not exact — `-7 div 2` is `-4`, and truncation
says `-3` — which is a wrong answer no string match and no JIT pass can see.
`examples/divmod.pl` is run against an oracle for it.

## Error model

| Condition | Error |
| :--- | :--- |
| Non-integer types, floats, lists, compounds, or unsupported builtins | `MathError::not_implemented` |
| A cycle in the call graph — recursion, direct or mutual — when target is `Target::triton` | `MathError::not_implemented` |
| More inlined call sites than `Target::triton` will expand (256) | `MathError::not_implemented` |
| One predicate reached in two different mode patterns | `MathError::not_implemented` |
| Any width other than `Width::bits64` requested for `Target::triton` | `MathError::not_implemented` |
| `mod` or `rem` in a program compiled at `Width::decimal128` | `MathError::not_implemented` |
| `decimal_scale` outside `0..18` at `Width::decimal128` | `MathError::domain_error` |
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
