// Tests for nimblecas.logic_compile: Prolog clauses compiled to C++, CUDA and Triton source.
// @author Olumuyiwa Oluwasanmi
//
// The Prolog compiler lowers an integer, moded, deterministic subset of Prolog into native
// source text: C++23 (standard and arbitrary precision), CUDA device kernels, and Triton
// JIT elementwise functions. It also provides a continuation-passing transform for predicates
// with multiple solutions.
//
// Because unit tests running under ctest cannot shell out to external native compilers (clang,
// nvcc, triton JIT) without turning unit testing into compiler testing, this test suite verifies
// the emitter's essential properties in-process:
//
//   1. Target & width coverage: code generation succeeds for each supported Target and integer
//      width, producing non-empty source containing documented structural markers.
//   2. Determinism: emitting the same program repeatedly under identical options yields
//      strictly byte-identical source text across every backend.
//   3. Honesty invariants: every unsupported construct, arity mismatch, uninitialised variable,
//      invalid target/width pairing, and recursive Triton kernel request returns an honest
//      MathError rather than emitting broken or misleading code.
//   4. Algorithmic invariants: tail-call optimisation loop rewrites, multi-predicate call graph
//      topological sorting, batch parallel wrappers, and continuation-passing callbacks.

import std;
import nimblecas.core;
import nimblecas.logic;
import nimblecas.logic_compile;
import nimblecas.logic_parser;
import nimblecas.testing;

using nimblecas::Clause;
using nimblecas::make_compound;
using nimblecas::make_int;
using nimblecas::make_var;
using nimblecas::MathError;
using nimblecas::Program;
using nimblecas::logic_compile::ArgMode;
using nimblecas::logic_compile::compile;
using nimblecas::logic_compile::CompileOptions;
using nimblecas::logic_compile::is_compilable;
using nimblecas::logic_compile::PredicateSignature;
using nimblecas::logic_compile::Style;
using nimblecas::logic_compile::Target;
using nimblecas::logic_compile::to_string_view;
using nimblecas::logic_compile::Width;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

[[nodiscard]] auto prog(std::string_view src) -> Program {
    auto p = nimblecas::logic_parser::parse_program(src);
    return p ? *p : Program{};
}

// Flat arithmetic evaluation: computes y = x * x + 3 * x + 5.
[[nodiscard]] auto eval_poly_program() -> Program {
    return prog("eval_poly(X, Y) :- T1 is X * X, T2 is 3 * X, Y is T1 + T2 + 5.\n");
}

// Factorial with an accumulator: tail-recursive when tail_call_optimise is enabled.
[[nodiscard]] auto fact_acc_program() -> Program {
    return prog(
        "fact_acc(0, A, A).\n"
        "fact_acc(N, A, R) :- N > 0, A1 is A * N, N1 is N - 1, fact_acc(N1, A1, R).\n");
}

// Multi-predicate program with callee dependency.
[[nodiscard]] auto multi_pred_program() -> Program {
    return prog(
        "square(X, Y) :- Y is X * X.\n"
        "sum_of_squares(A, B, S) :- square(A, SA), square(B, SB), S is SA + SB.\n");
}

// Nondeterministic between/3 predicate suitable for CPS compilation.
[[nodiscard]] auto between_program() -> Program {
    return prog(
        "between(Low, High, Low) :- Low =< High.\n"
        "between(Low, High, Out) :- Low < High, Next is Low + 1, between(Next, High, Out).\n");
}

// Nondeterministic predicate with cut in CPS form.
[[nodiscard]] auto cut_cps_program() -> Program {
    return prog(
        "between(Low, High, Low) :- Low =< High.\n"
        "between(Low, High, Out) :- Low < High, Next is Low + 1, between(Next, High, Out).\n"
        "first_between(Low, High, Out) :- between(Low, High, Out), !.\n");
}

// All checked arithmetic binary operators.
[[nodiscard]] auto arithmetic_ops_program() -> Program {
    return prog(
        "ops(A, B, Q, F, R, M, Mn, Mx) :-\n"
        "    Q is A // B,\n"
        "    F is A div B,\n"
        "    R is A rem B,\n"
        "    M is A mod B,\n"
        "    Mn is min(A, B),\n"
        "    Mx is max(A, B).\n");
}

// Relational comparisons.
[[nodiscard]] auto comparisons_program() -> Program {
    return prog(
        "check_ranges(A, B) :-\n"
        "    A < B,\n"
        "    B > A,\n"
        "    A =< B,\n"
        "    B >= A,\n"
        "    A =:= A,\n"
        "    A =\\= B.\n");
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.logic_compile")
        .test("target_and_width_to_string_view_return_canonical_identifiers",
              [](TestContext& t) {
                  t.expect(to_string_view(Target::cpp) == "cpp",
                           "Target::cpp string view is 'cpp'");
                  t.expect(to_string_view(Target::cuda) == "cuda",
                           "Target::cuda string view is 'cuda'");
                  t.expect(to_string_view(Target::triton) == "triton",
                           "Target::triton string view is 'triton'");

                  t.expect(to_string_view(Width::bits64) == "bits64",
                           "Width::bits64 string view is 'bits64'");
                  t.expect(to_string_view(Width::bits128) == "bits128",
                           "Width::bits128 string view is 'bits128'");
                  t.expect(to_string_view(Width::arbitrary) == "arbitrary",
                           "Width::arbitrary string view is 'arbitrary'");
              })
        .test("is_compilable_agrees_with_compile_for_valid_predicates",
              [](TestContext& t) {
                  const Program p = eval_poly_program();
                  const PredicateSignature sig{
                      .name = "eval_poly",
                      .modes = {ArgMode::input, ArgMode::output},
                  };

                  for (const Target target : {Target::cpp, Target::cuda, Target::triton}) {
                      const auto comp = is_compilable(p, sig, target);
                      t.expect(comp.has_value(),
                               "is_compilable succeeds for valid predicate on target");
                      auto src = compile(p, sig, target);
                      t.expect(src.has_value() && !src->empty(),
                               "compile produces non-empty source for valid predicate on target");
                  }
              })
        .test("cpp_64bit_emission_contains_checked_arithmetic_preamble_and_entry_signatures",
              [](TestContext& t) {
                  const Program p = eval_poly_program();
                  const PredicateSignature sig{
                      .name = "eval_poly",
                      .modes = {ArgMode::input, ArgMode::output},
                  };
                  CompileOptions opts;
                  opts.target = Target::cpp;
                  opts.width = Width::bits64;

                  auto src = compile(p, sig, opts);
                  t.expect(src.has_value(), "compilation to 64-bit C++ succeeds");
                  if (!src.has_value()) {
                      return;
                  }

                  t.expect(src->contains("using nc_int = std::int64_t;"),
                           "64-bit preamble defines nc_int as std::int64_t");
                  t.expect(src->contains("auto nc_lit(std::int64_t v) -> nc_int"),
                           "literal constructor takes std::int64_t, not long long (Rule 48)");
                  t.expect(src->contains(
                               "[[nodiscard]] constexpr auto nc_add(nc_int a, nc_int b) "
                               "-> std::optional<nc_int>"),
                           "checked addition returns std::optional (Rules 9/32), constexpr "
                           "(Rules 8/42), nodiscard (Rule 10)");
                  t.expect(src->contains(
                               "[[nodiscard]] constexpr auto nc_sub(nc_int a, nc_int b) "
                               "-> std::optional<nc_int>"),
                           "checked subtraction has the same shape");
                  t.expect(src->contains(
                               "[[nodiscard]] constexpr auto nc_mul(nc_int a, nc_int b) "
                               "-> std::optional<nc_int>"),
                           "checked multiplication has the same shape");
                  t.expect(src->contains("std::numeric_limits<nc_int>::max()"),
                           "pre-checked overflow guards check std::numeric_limits max");
                  t.expect(src->contains("[[nodiscard]] inline auto p_eval_poly_2(nc_int a0, nc_int& a1) -> bool"),
                           "entry predicate has nodiscard trailing return type and moded parameters");
                  // Rules 11/41: import std, never the headers. The arbitrary-precision
                  // width always did this; the narrow ones used to emit <cstdint>/<limits>,
                  // so the compiler's own output contradicted the policy depending on which
                  // width you asked for.
                  t.expect(src->contains("import std;"),
                           "the standard library arrives as a module (Rules 11/41)");
                  t.expect(!src->contains("#include <cstdint>") &&
                               !src->contains("#include <limits>"),
                           "and no standard header is included in the C++ target");
                  t.expect(!src->contains("nimblecas.bigint"),
                           "64-bit build does not import bigint module");
                  t.expect(src->contains("// @author Olumuyiwa Oluwasanmi"),
                           "emitted file header preserves repository authorship");
              })
        .test("cpp_target_emits_only_c++23_conforming_source",
              [](TestContext& t) {
                  // config/cpp_details.txt applied to the code this compiler WRITES, not just
                  // to the compiler itself. Checked as properties of the text so that any
                  // future change reintroducing a header, an out-parameter or a `long long`
                  // in the C++ target fails here.
                  const Program p = eval_poly_program();
                  const PredicateSignature sig{
                      .name = "eval_poly",
                      .modes = {ArgMode::input, ArgMode::output},
                  };
                  for (const Width w : {Width::bits64, Width::bits128, Width::arbitrary}) {
                      CompileOptions opts{};
                      opts.target = Target::cpp;
                      opts.width = w;
                      const auto src = compile(p, sig, opts);
                      t.expect(src.has_value(), "compilation succeeds at every width");
                      if (!src.has_value()) {
                          continue;
                      }
                      t.expect(src->contains("import std;"), "Rules 11/41: import std");
                      t.expect(!src->contains("#include <"),
                               "Rules 11/41: no standard header in the C++ target");
                      t.expect(!src->contains("long long"),
                               "Rule 48: fixed-width integer types, never long long");
                      t.expect(!src->contains("nc_int& r) -> bool"),
                               "Rules 9/32: no out-parameter-plus-status arithmetic helper");
                      t.expect(src->contains("-> std::optional<nc_int>"),
                               "Rules 9/32: the checked helpers return std::optional");
                      t.expect(src->contains("[[nodiscard]]"),
                               "Rule 10: results that must not be dropped say so");
                      // Rule 31/46: trailing return types. Checked against the helpers by
                      // name -- a bare `!contains("nc_int nc_")` would also match the
                      // `constexpr nc_int nc_max_v` bounds constant, which is a variable and
                      // has no return type to put anywhere.
                      t.expect(!src->contains("nc_int nc_add(") &&
                                   !src->contains("nc_int nc_lit(") &&
                                   !src->contains("bool nc_"),
                               "Rule 31: trailing return types throughout");
                      t.expect(src->contains("auto nc_add(") && src->contains("auto nc_lit("),
                               "and every helper is declared with auto ... -> T");
                  }
                  // CUDA is deliberately exempt and must NOT be changed to match: nvcc has no
                  // `import std`, and std::optional is not device-callable without libcu++.
                  CompileOptions cuda{};
                  cuda.target = Target::cuda;
                  const auto csrc = compile(p, sig, cuda);
                  t.expect(csrc.has_value(), "CUDA compilation still succeeds");
                  if (csrc.has_value()) {
                      t.expect(csrc->contains("#include <cstdint>"),
                               "CUDA keeps headers, because nvcc has no import std");
                      t.expect(csrc->contains("nc_int& r) -> bool"),
                               "CUDA keeps the out-parameter form, which is device-callable");
                  }
              })
        .test("cpp_128bit_emission_emits_compiler_overflow_builtins_and_sizeof_guard",
              [](TestContext& t) {
                  const Program p = eval_poly_program();
                  const PredicateSignature sig{
                      .name = "eval_poly",
                      .modes = {ArgMode::input, ArgMode::output},
                  };
                  CompileOptions opts;
                  opts.target = Target::cpp;
                  opts.width = Width::bits128;

                  auto src = compile(p, sig, opts);
                  t.expect(src.has_value(), "compilation to 128-bit C++ succeeds");
                  if (!src.has_value()) {
                      return;
                  }

                  t.expect(src->contains("#if !defined(__SIZEOF_INT128__)"),
                           "128-bit preamble guards against compilers lacking __int128");
                  t.expect(src->contains("#error \"generated for 128-bit arithmetic, which this compiler does not provide\""),
                           "unsupported compiler triggers documented compile-time error");
                  t.expect(src->contains("using nc_int = __int128;"),
                           "128-bit nc_int alias is defined as __int128");
                  t.expect(src->contains("if (__builtin_add_overflow(a, b, &out))"),
                           "checked addition uses __builtin_add_overflow");
                  t.expect(src->contains("if (__builtin_sub_overflow(a, b, &out))"),
                           "checked subtraction uses __builtin_sub_overflow");
                  t.expect(src->contains("if (__builtin_mul_overflow(a, b, &out))"),
                           "checked multiplication uses __builtin_mul_overflow");
                  t.expect(src->contains("[[nodiscard]] inline auto p_eval_poly_2("),
                           "entry predicate signature is emitted");
              })
        .test("cpp_arbitrary_width_emission_imports_bigint_module_and_delegates_to_bigint_methods",
              [](TestContext& t) {
                  const Program p = eval_poly_program();
                  const PredicateSignature sig{
                      .name = "eval_poly",
                      .modes = {ArgMode::input, ArgMode::output},
                  };
                  CompileOptions opts;
                  opts.target = Target::cpp;
                  opts.width = Width::arbitrary;

                  auto src = compile(p, sig, opts);
                  t.expect(src.has_value(), "compilation to arbitrary-precision C++ succeeds");
                  if (!src.has_value()) {
                      return;
                  }

                  t.expect(src->contains("import std;\nimport nimblecas.core;\nimport nimblecas.bigint;\n"),
                           "arbitrary-precision translation unit imports nimblecas.bigint module");
                  t.expect(src->contains("using nc_int = nimblecas::BigInt;"),
                           "nc_int is aliased to nimblecas::BigInt");
                  t.expect(src->contains("nimblecas::BigInt::from_i64"),
                           "integer literals are converted via BigInt::from_i64 factory");
                  t.expect(src->contains("return a.add(b);"),
                           "addition delegates to BigInt::add without redundant overflow checks");
                  t.expect(src->contains("return a.subtract(b);"),
                           "subtraction delegates to BigInt::subtract");
                  t.expect(src->contains("return a.multiply(b);"),
                           "multiplication delegates to BigInt::multiply");
                  t.expect(src->contains("return a.negate();"),
                           "negation delegates to BigInt::negate");
                  t.expect(src->contains("a.divmod(b)"),
                           "division delegates to BigInt::divmod");
                  t.expect(!src->contains("#include <cstdint>"),
                           "modular translation unit does not include cstdint header");
              })
        .test("cuda_64bit_emission_emits_device_functions_and_global_batch_kernel",
              [](TestContext& t) {
                  const Program p = eval_poly_program();
                  const PredicateSignature sig{
                      .name = "eval_poly",
                      .modes = {ArgMode::input, ArgMode::output},
                  };
                  CompileOptions opts;
                  opts.target = Target::cuda;
                  opts.width = Width::bits64;

                  auto src = compile(p, sig, opts);
                  t.expect(src.has_value(), "compilation to 64-bit CUDA succeeds");
                  if (!src.has_value()) {
                      return;
                  }

                  t.expect(src->contains("__device__ [[nodiscard]] inline auto nc_add("),
                           "preamble helpers are __device__, inline, and nodiscard -- a "
                           "dropped nc_add result is a dropped overflow check");
                  t.expect(src->contains("__device__ inline auto p_eval_poly_2("),
                           "predicate function is decorated with __device__ inline");
                  t.expect(src->contains("__global__ void p_eval_poly_2_batch(int n, unsigned char* ok, const nc_int* arg0, nc_int* arg1)"),
                           "global batch kernel is emitted with exact parameter qualifiers");
                  t.expect(src->contains("const int i = blockIdx.x * blockDim.x + threadIdx.x;"),
                           "batch kernel computes flat lane index from block and thread dimensions");
                  t.expect(src->contains("if (i >= n) { return; }"),
                           "batch kernel guards against thread indices exceeding batch length");
                  t.expect(src->contains("ok[i] = p_eval_poly_2(arg0[i], arg1[i]) ? 1 : 0;"),
                           "batch kernel invokes device predicate and records success in ok mask");
              })
        .test("cuda_128bit_emission_synthesises_128bit_types_for_device_functions",
              [](TestContext& t) {
                  const Program p = eval_poly_program();
                  const PredicateSignature sig{
                      .name = "eval_poly",
                      .modes = {ArgMode::input, ArgMode::output},
                  };
                  CompileOptions opts;
                  opts.target = Target::cuda;
                  opts.width = Width::bits128;

                  auto src = compile(p, sig, opts);
                  t.expect(src.has_value(), "compilation to 128-bit CUDA succeeds");
                  if (!src.has_value()) {
                      return;
                  }

                  t.expect(src->contains("using nc_int = __int128;"),
                           "CUDA 128-bit defines nc_int as __int128");
                  t.expect(src->contains("__device__ [[nodiscard]] inline auto nc_add"),
                           "CUDA 128-bit helpers are marked __device__ inline");
                  t.expect(src->contains("__global__ void p_eval_poly_2_batch(int n, unsigned char* ok, const nc_int* arg0, nc_int* arg1)"),
                           "CUDA 128-bit batch kernel is emitted");
              })
        .test("triton_emission_emits_jit_decorated_kernel_with_lane_masks_and_elementwise_loads",
              [](TestContext& t) {
                  const Program p = eval_poly_program();
                  const PredicateSignature sig{
                      .name = "eval_poly",
                      .modes = {ArgMode::input, ArgMode::output},
                  };
                  CompileOptions opts;
                  opts.target = Target::triton;
                  opts.width = Width::bits64;

                  auto src = compile(p, sig, opts);
                  t.expect(src.has_value(), "compilation to Triton succeeds");
                  if (!src.has_value()) {
                      return;
                  }

                  t.expect(src->contains("import triton\nimport triton.language as tl\n"),
                           "triton and triton.language modules are imported");
                  // The bounds must be tl.constexpr. A @triton.jit function cannot read a
                  // module global that is not one, so the bare-integer form this assertion
                  // used to pin made every emitted kernel refuse to compile.
                  t.expect(src->contains("INT64_MAX = tl.constexpr(9223372036854775807)"),
                           "INT64_MAX boundary is a tl.constexpr a jitted kernel can read");
                  t.expect(src->contains("INT64_MIN = tl.constexpr(-9223372036854775808)"),
                           "INT64_MIN boundary is a tl.constexpr a jitted kernel can read");
                  // And the tile shape must be a tuple of compile-time integers. Reading it
                  // off another tensor does not survive being stored in a variable.
                  //
                  // The constructs are named exactly rather than banning the word `.shape`
                  // outright: a blanket ban is the assertion that matches its own explanatory
                  // comment the moment someone writes one. What actually keeps this honest is
                  // scripts/verify-generated.sh, which puts the kernel through the JIT.
                  t.expect(!src->contains("tl.zeros(shape") && !src->contains("tl.full(shape"),
                           "no tensor is sized from a shape bound to a variable");
                  t.expect(src->contains("tl.zeros((BLOCK,), dtype=tl.int1)"),
                           "the tile shape is written literally as (BLOCK,)");
                  t.expect(src->contains("@triton.jit"),
                           "kernel function is decorated with @triton.jit");
                  t.expect(src->contains("def p_eval_poly_2_kernel(a0_ptr, a1_ptr, ok_ptr, n, BLOCK: tl.constexpr):"),
                           "kernel function signature matches arguments and block size constexpr");
                  t.expect(src->contains("offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)"),
                           "lane offsets are computed with tl.program_id and tl.arange");
                  t.expect(src->contains("m = offs < n"),
                           "boundary mask m is computed from offsets and batch length");
                  t.expect(src->contains("a0 = tl.load(a0_ptr + offs, mask=m, other=0)"),
                           "input parameter is loaded with mask");
                  t.expect(src->contains("tl.store(a1_ptr + offs, a1, mask=m)"),
                           "output parameter is stored with mask");
                  t.expect(src->contains("tl.store(ok_ptr + offs, done.to(tl.int8), mask=m)"),
                           "success mask is converted to int8 and stored to ok_ptr");
              })
        .test("emission_is_strictly_deterministic_across_all_supported_targets_and_widths",
              [](TestContext& t) {
                  const Program p = eval_poly_program();
                  const PredicateSignature sig{
                      .name = "eval_poly",
                      .modes = {ArgMode::input, ArgMode::output},
                  };

                  struct Config {
                      Target target;
                      Width width;
                  };
                  const std::array<Config, 6> configs{{
                      {Target::cpp, Width::bits64},
                      {Target::cpp, Width::bits128},
                      {Target::cpp, Width::arbitrary},
                      {Target::cuda, Width::bits64},
                      {Target::cuda, Width::bits128},
                      {Target::triton, Width::bits64},
                  }};

                  for (const auto& cfg : configs) {
                      CompileOptions opts;
                      opts.target = cfg.target;
                      opts.width = cfg.width;

                      auto first = compile(p, sig, opts);
                      auto second = compile(p, sig, opts);
                      t.expect(first.has_value() && second.has_value(),
                               "both compilation passes succeed");
                      if (first.has_value() && second.has_value()) {
                          t.expect(*first == *second,
                                   "repeated compilation produces byte-identical source output");
                      }
                  }
              })
        .test("tail_call_optimisation_rewrites_self_tail_recursion_into_a_while_loop",
              [](TestContext& t) {
                  const Program p = fact_acc_program();
                  const PredicateSignature sig{
                      .name = "fact_acc",
                      .modes = {ArgMode::input, ArgMode::input, ArgMode::output},
                  };
                  CompileOptions opts;
                  opts.target = Target::cpp;
                  opts.tail_call_optimise = true;

                  auto src = compile(p, sig, opts);
                  t.expect(src.has_value(), "tail-recursive compilation succeeds");
                  if (!src.has_value()) {
                      return;
                  }

                  t.expect(src->contains("for (;;) {"),
                           "tail-call optimised code wraps clauses in an unbounded for loop");
                  t.expect(src->contains("bool nc_again = false;"),
                           "loop iteration re-entry flag nc_again is declared");
                  t.expect(src->contains("nc_again = true;\n"),
                           "self tail call sets nc_again to true");
                  t.expect(src->contains("if (nc_again) { continue; }"),
                           "nc_again triggers loop continuation rather than call stack recursion");
                  t.expect(src->contains("compiled to a LOOP"),
                           "explanatory comment documents loop transform");
                  t.expect(src->contains("const nc_int n"),
                           "new argument values are evaluated into temporaries before write-back");
              })
        .test("disabling_tail_call_optimisation_preserves_recursive_function_calls",
              [](TestContext& t) {
                  const Program p = fact_acc_program();
                  const PredicateSignature sig{
                      .name = "fact_acc",
                      .modes = {ArgMode::input, ArgMode::input, ArgMode::output},
                  };
                  CompileOptions opts;
                  opts.target = Target::cpp;
                  opts.tail_call_optimise = false;

                  auto src = compile(p, sig, opts);
                  t.expect(src.has_value(), "unoptimised compilation succeeds");
                  if (!src.has_value()) {
                      return;
                  }

                  t.expect(!src->contains("nc_again"),
                           "unoptimised code emits no nc_again loop flag");
                  t.expect(!src->contains("for (;;)"),
                           "unoptimised code emits no outer infinite loop");
                  t.expect(src->contains("p_fact_acc_3_c0("),
                           "clause 0 is emitted as an independent helper function");
                  t.expect(src->contains("p_fact_acc_3_c1("),
                           "clause 1 is emitted as an independent helper function");
                  t.expect(src->contains("if (p_fact_acc_3_c0(a0, a1, a2)) { return true; }"),
                           "dispatcher tries clause 0 first in program order");
                  t.expect(src->contains("p_fact_acc_3("),
                           "recursive call is emitted as a standard function call");
              })
        .test("batch_driver_emission_emits_serial_and_multithreaded_map_wrappers",
              [](TestContext& t) {
                  const Program p = eval_poly_program();
                  const PredicateSignature sig{
                      .name = "eval_poly",
                      .modes = {ArgMode::input, ArgMode::output},
                  };
                  CompileOptions opts;
                  opts.target = Target::cpp;
                  opts.emit_batch = true;

                  auto src = compile(p, sig, opts);
                  t.expect(src.has_value(), "compilation with batch drivers enabled succeeds");
                  if (!src.has_value()) {
                      return;
                  }

                  // std::jthread and std::size_t arrive through `import std`, not headers
                  // (Rules 11/41). Only the x86 intrinsics, which have no module form
                  // anywhere, are still included.
                  t.expect(src->contains("import std;"),
                           "the batch driver's standard facilities come from the std module");
                  t.expect(src->contains("#include <immintrin.h>"),
                           "the SIMD intrinsics, which have no module form, are still included");
                  t.expect(src->contains("inline auto p_eval_poly_2_batch(const nc_int* arg0, nc_int* arg1, unsigned char* ok, std::size_t n) -> void"),
                           "serial batch wrapper is emitted with const input and mutable output pointers");
                  t.expect(src->contains("inline auto p_eval_poly_2_batch_parallel(const nc_int* arg0, nc_int* arg1, unsigned char* ok, std::size_t n,"),
                           "parallel batch wrapper is emitted");
                  t.expect(src->contains("std::jthread"),
                           "parallel batch wrapper employs std::jthread workers");
                  t.expect(src->contains("std::thread::hardware_concurrency()"),
                           "worker count defaults to hardware_concurrency when threads == 0");
                  t.expect(src->contains("constexpr std::size_t serial_below = 4096;"),
                           "small batch sizes below 4096 lanes run serially on the calling thread");
              })
        .test("continuation_passing_style_emits_templated_caller_and_nested_closures",
              [](TestContext& t) {
                  const Program p = between_program();
                  const PredicateSignature sig{
                      .name = "between",
                      .modes = {ArgMode::input, ArgMode::input, ArgMode::output},
                  };
                  CompileOptions opts;
                  opts.target = Target::cpp;
                  opts.style = Style::continuation;

                  auto src = compile(p, sig, opts);
                  t.expect(src.has_value(), "CPS compilation succeeds");
                  if (!src.has_value()) {
                      return;
                  }

                  t.expect(src->contains("template <class K>\n[[nodiscard]] auto p_between_3(nc_int a0, nc_int a1, K&& k) -> bool;"),
                           "forward declaration is templated on continuation callable K");
                  t.expect(src->contains("template <class K>\n[[nodiscard]] auto p_between_3(nc_int a0, nc_int a1, K&& k) -> bool {"),
                           "the public entry stays templated on the caller's callable");
                  t.expect(src->contains("return k(v_Low);"),
                           "base clause invokes continuation with solution value v_Low");
                  t.expect(src->contains("return k(v_Out);"),
                           "recursive clause invokes continuation with solution value v_Out");
                  t.expect(src->contains("[&](nc_int v_Out) -> bool"),
                           "continuation closure binds returned argument and chains the rest of the clause");

                  // THE RECURSIVE CALL MUST NOT BE A TEMPLATE INSTANTIATION. It used to be, and
                  // every level of the enumeration handed itself a fresh lambda type, so each
                  // instantiation demanded another and the compiler never finished -- between/3,
                  // the predicate continuation passing exists for, could not be built at all.
                  // The erased view gives every continuation one type, so there is one body.
                  t.expect(src->contains("class nc_cont_1 {"),
                           "a continuation view with a single output argument is emitted");
                  t.expect(src->contains("return p_between_3_cps(v_Next, v_High, nc_cont_1{"),
                           "the recursive call goes through the erased view, not a new instantiation");
                  t.expect(src->contains("[[nodiscard]] inline auto p_between_3_cps(nc_int a0, nc_int a1, nc_cont_1 k) -> bool {"),
                           "the body that recurses is an ordinary function, so it instantiates once");
                  t.expect(!src->contains("std::function"),
                           "the view allocates nothing, so nothing in the emitted code can throw");
              })
        .test("continuation_passing_style_handles_cut_to_prune_subsequent_clauses",
              [](TestContext& t) {
                  const Program p = cut_cps_program();
                  const PredicateSignature sig{
                      .name = "first_between",
                      .modes = {ArgMode::input, ArgMode::input, ArgMode::output},
                  };
                  CompileOptions opts;
                  opts.target = Target::cpp;
                  opts.style = Style::continuation;

                  auto src = compile(p, sig, opts);
                  t.expect(src.has_value(), "CPS compilation with cut succeeds");
                  if (!src.has_value()) {
                      return;
                  }

                  t.expect(src->contains("bool nc_cut = false;"),
                           "clause dispatcher tracks cut status with nc_cut flag");
                  t.expect(src->contains("nc_cut = true;"),
                           "cut operator in clause body sets nc_cut flag");
                  t.expect(src->contains("if (nc_cut) { return true; }"),
                           "pruning stops searching subsequent clauses once cut succeeds");
              })
        .test("multi_predicate_call_graph_walk_emits_callees_before_callers",
              [](TestContext& t) {
                  const Program p = multi_pred_program();
                  const PredicateSignature sig{
                      .name = "sum_of_squares",
                      .modes = {ArgMode::input, ArgMode::input, ArgMode::output},
                  };
                  CompileOptions opts;
                  opts.target = Target::cpp;

                  auto src = compile(p, sig, opts);
                  t.expect(src.has_value(), "compilation of multi-predicate call graph succeeds");
                  if (!src.has_value()) {
                      return;
                  }

                  const std::size_t decl_square = src->find("auto p_square_2(");
                  const std::size_t decl_sum = src->find("auto p_sum_of_squares_3(");
                  t.expect(decl_square != std::string::npos && decl_sum != std::string::npos,
                           "both predicates are declared in source");

                  const std::size_t def_square = src->find("auto p_square_2(nc_int a0, nc_int& a1) -> bool {");
                  const std::size_t def_sum = src->find("auto p_sum_of_squares_3(nc_int a0, nc_int a1, nc_int& a2) -> bool {");
                  t.expect(def_square != std::string::npos && def_sum != std::string::npos &&
                               def_square < def_sum,
                           "callee definition p_square_2 precedes caller definition p_sum_of_squares_3");

                  // The call passes the HEAD-BOUND variable, not the raw argument: an input
                  // head argument is bound to `v_A` first, and that is what the goal refers to.
                  t.expect(src->contains("if (!p_square_2(v_A, v_SA)) { return false; }"),
                           "caller invokes compiled callee and checks return status");
              })
        .test("all_arithmetic_operators_emit_their_corresponding_checked_helpers",
              [](TestContext& t) {
                  const Program p = arithmetic_ops_program();
                  const PredicateSignature sig{
                      .name = "ops",
                      .modes = {ArgMode::input,  ArgMode::input,  ArgMode::output,
                                ArgMode::output, ArgMode::output, ArgMode::output,
                                ArgMode::output, ArgMode::output},
                  };
                  CompileOptions opts;
                  opts.target = Target::cpp;

                  auto src = compile(p, sig, opts);
                  t.expect(src.has_value(), "compilation of all arithmetic operators succeeds");
                  if (!src.has_value()) {
                      return;
                  }

                  t.expect(src->contains("= nc_quot("),
                           "// operator emits checked nc_quot helper");
                  t.expect(src->contains("= nc_fdiv("),
                           "div operator emits checked nc_fdiv helper");
                  t.expect(src->contains("= nc_rem("),
                           "rem operator emits checked nc_rem helper");
                  t.expect(src->contains("= nc_mod("),
                           "mod operator emits checked nc_mod helper");
                  t.expect(src->contains("= nc_min("),
                           "min operator emits checked nc_min helper");
                  t.expect(src->contains("= nc_max("),
                           "max operator emits checked nc_max helper");
              })
        .test("all_relational_comparisons_emit_their_matching_cpp_operators",
              [](TestContext& t) {
                  const Program p = comparisons_program();
                  const PredicateSignature sig{
                      .name = "check_ranges",
                      .modes = {ArgMode::input, ArgMode::input},
                  };
                  CompileOptions opts;
                  opts.target = Target::cpp;

                  auto src = compile(p, sig, opts);
                  t.expect(src.has_value(), "compilation of comparison operators succeeds");
                  if (!src.has_value()) {
                      return;
                  }

                  t.expect(src->contains(" < "),
                           "< operator emits C++ < comparison");
                  t.expect(src->contains(" > "),
                           "> operator emits C++ > comparison");
                  t.expect(src->contains(" <= "),
                           "=< operator emits C++ <= comparison");
                  t.expect(src->contains(" >= "),
                           ">= operator emits C++ >= comparison");
                  t.expect(src->contains(" == "),
                           "=:=" " operator emits C++ == equality comparison");
                  t.expect(src->contains(" != "),
                           "=\\= operator emits C++ != inequality comparison");
              })
        .test("fact_only_and_zero_arity_predicates_compile_correctly",
              [](TestContext& t) {
                  const Program fact_prog = prog("answer(42).\n");
                  const PredicateSignature fact_sig{
                      .name = "answer",
                      .modes = {ArgMode::output},
                  };
                  auto fact_src = compile(fact_prog, fact_sig, Target::cpp);
                  t.expect(fact_src.has_value(), "fact-only predicate compiles");
                  if (fact_src.has_value()) {
                      t.expect(fact_src->contains("a0 = nc_lit(42);"),
                               "fact assigns literal constant to output argument");
                      t.expect(fact_src->contains("return true;"),
                               "fact returns true");
                  }

                  const Program zero_prog = prog("truth.\n");
                  const PredicateSignature zero_sig{
                      .name = "truth",
                      .modes = {},
                  };
                  auto zero_src = compile(zero_prog, zero_sig, Target::cpp);
                  t.expect(zero_src.has_value(), "zero-arity atom fact predicate compiles");
                  if (zero_src.has_value()) {
                      t.expect(zero_src->contains("[[nodiscard]] inline auto p_truth_0() -> bool"),
                               "zero-arity function signature takes no arguments");
                  }
              })
        .test("repeated_input_variables_are_checked_for_equality",
              [](TestContext& t) {
                  const Program p = prog("same(X, X).\n");
                  const PredicateSignature sig{
                      .name = "same",
                      .modes = {ArgMode::input, ArgMode::input},
                  };
                  auto src = compile(p, sig, Target::cpp);
                  t.expect(src.has_value(), "predicate with repeated input variable compiles");
                  if (!src.has_value()) {
                      return;
                  }

                  t.expect(src->contains("const nc_int v_X = a0;"),
                           "first input occurrence of X binds variable");
                  t.expect(src->contains("if (a1 != v_X) { return false; }"),
                           "subsequent input occurrence checks argument equality against bound variable");
              })
        .test("an_anonymous_variable_in_an_output_position_is_refused",
              [](TestContext& t) {
                  // `_` binds nothing, so an output holding it has no value to hand back.
                  // Skipping it would leave the output argument untouched -- and in
                  // continuation-passing style would reference an identifier that was never
                  // declared, emitting source that cannot compile. Refusing is the only
                  // honest answer.
                  const Program p = prog("unbound_out(X, _) :- X > 0.\n");
                  const PredicateSignature sig{
                      .name = "unbound_out",
                      .modes = {ArgMode::input, ArgMode::output},
                  };
                  auto direct = compile(p, sig, Target::cpp);
                  t.expect(!direct.has_value() && direct.error() == MathError::domain_error,
                           "an anonymous output is a domain_error, not an untouched argument");

                  CompileOptions cps;
                  cps.target = Target::cpp;
                  cps.style = Style::continuation;
                  auto k = compile(p, sig, cps);
                  t.expect(!k.has_value() && k.error() == MathError::domain_error,
                           "and the continuation-passing path refuses it too, rather than "
                           "passing an undeclared identifier to the continuation");
              })
        .test("anonymous_variables_in_head_arguments_are_ignored",
              [](TestContext& t) {
                  const Program p = prog("ignore_first(_, Y, Y).\n");
                  const PredicateSignature sig{
                      .name = "ignore_first",
                      .modes = {ArgMode::input, ArgMode::input, ArgMode::output},
                  };
                  auto src = compile(p, sig, Target::cpp);
                  t.expect(src.has_value(), "predicate with anonymous head variable compiles");
                  if (!src.has_value()) {
                      return;
                  }

                  t.expect(!src->contains("v__"),
                           "anonymous variable _ does not generate an unneeded variable declaration");
                  t.expect(src->contains("const nc_int v_Y = a1;"),
                           "named head variable Y is bound as expected");
                  t.expect(src->contains("a2 = v_Y;"),
                           "output argument is assigned from Y");
              })
        .test("symbolic_predicate_names_are_escaped_cleanly",
              [](TestContext& t) {
                  Program p;
                  p.push_back(Clause{
                      .head = make_compound("plus+one", {make_var("X"), make_var("Y")}),
                      .body = {make_compound("is", {make_var("Y"),
                                                   make_compound("+", {make_var("X"), make_int(1)})})},
                  });
                  const PredicateSignature sig{
                      .name = "plus+one",
                      .modes = {ArgMode::input, ArgMode::output},
                  };
                  auto src = compile(p, sig, Target::cpp);
                  t.expect(src.has_value(), "predicate with symbolic character in name compiles");
                  if (src.has_value()) {
                      t.expect(src->contains("p_plus_x2bone_2"),
                               "plus symbol + is escaped as _x2b in mangled function identifier");
                  }
              })
        .test("triton_arithmetic_emits_safe_division_and_overflow_predicates",
              [](TestContext& t) {
                  const Program p = arithmetic_ops_program();
                  const PredicateSignature sig{
                      .name = "ops",
                      .modes = {ArgMode::input,  ArgMode::input,  ArgMode::output,
                                ArgMode::output, ArgMode::output, ArgMode::output,
                                ArgMode::output, ArgMode::output},
                  };
                  auto src = compile(p, sig, Target::triton);
                  t.expect(src.has_value(), "triton compilation of arithmetic ops succeeds");
                  if (!src.has_value()) {
                      return;
                  }

                  t.expect(src->contains("tl.where"),
                           "triton division employs tl.where guards to prevent division by zero");
                  t.expect(src->contains("// tl.where"),
                           "the quotient is taken against a divisor guarded away from zero");
                  t.expect(src->contains("% tl.where"),
                           "the remainder is taken against a divisor guarded away from zero");

                  // Triton's `//` and `%` TRUNCATE toward zero -- -7 // 2 is -3, not the -4
                  // Python would give. That is Prolog's `//` and `rem` exactly, and it is NOT
                  // Prolog's `div` and `mod`, which floor. Emitting all four as the same two
                  // operators answered -7 div 2 = -3 where every other target says -4, so the
                  // flooring correction has to appear.
                  t.expect(src->contains("_adj = "),
                           "div and mod emit the correction that turns truncation into floor");
                  t.expect(src->contains("_q - 1"),
                           "div steps the truncated quotient down where it rounded toward zero");
              })
        .test("compile_and_is_compilable_refuse_unbound_output_variables",
              [](TestContext& t) {
                  // Output argument Y is never bound in clause body.
                  const Program p = prog("unbound_out(X, Y) :- X > 0.\n");
                  const PredicateSignature sig{
                      .name = "unbound_out",
                      .modes = {ArgMode::input, ArgMode::output},
                  };

                  auto comp = compile(p, sig, Target::cpp);
                  t.expect(!comp.has_value() && comp.error() == MathError::domain_error,
                           "compile refuses unbound output argument as domain_error");

                  auto check = is_compilable(p, sig, Target::cpp);
                  t.expect(!check.has_value() && check.error() == MathError::domain_error,
                           "is_compilable returns domain_error for unbound output argument");
              })
        .test("compile_and_is_compilable_refuse_uninitialised_variable_reads",
              [](TestContext& t) {
                  // Variable Z is read in arithmetic expression without having been bound.
                  const Program p = prog("read_unbound(X, Y) :- Y is X + Z.\n");
                  const PredicateSignature sig{
                      .name = "read_unbound",
                      .modes = {ArgMode::input, ArgMode::output},
                  };

                  auto comp = compile(p, sig, Target::cpp);
                  t.expect(!comp.has_value() && comp.error() == MathError::domain_error,
                           "reading an unbound variable is refused as domain_error");

                  auto check = is_compilable(p, sig, Target::cpp);
                  t.expect(!check.has_value() && check.error() == MathError::domain_error,
                           "is_compilable returns domain_error when reading uninitialised variable");
              })
        .test("compile_and_is_compilable_refuse_compound_terms_in_head_arguments",
              [](TestContext& t) {
                  const Program p = prog("pair_head(pair(X, Y), Z) :- Z is X + Y.\n");
                  const PredicateSignature sig{
                      .name = "pair_head",
                      .modes = {ArgMode::input, ArgMode::output},
                  };

                  auto comp = compile(p, sig, Target::cpp);
                  t.expect(!comp.has_value() && comp.error() == MathError::not_implemented,
                           "compound term in head argument is refused as not_implemented");

                  auto check = is_compilable(p, sig, Target::cpp);
                  t.expect(!check.has_value() && check.error() == MathError::not_implemented,
                           "is_compilable returns not_implemented for compound head term");
              })
        .test("compile_and_is_compilable_refuse_non_integer_constants_in_head_arguments",
              [](TestContext& t) {
                  const Program p = prog("atom_head(hello, 1).\n");
                  const PredicateSignature sig{
                      .name = "atom_head",
                      .modes = {ArgMode::input, ArgMode::output},
                  };

                  auto comp = compile(p, sig, Target::cpp);
                  t.expect(!comp.has_value() && comp.error() == MathError::not_implemented,
                           "atom constant in head argument is refused as not_implemented");

                  auto check = is_compilable(p, sig, Target::cpp);
                  t.expect(!check.has_value() && check.error() == MathError::not_implemented,
                           "is_compilable returns not_implemented for atom constant head argument");
              })
        .test("compile_and_is_compilable_refuse_unsupported_operators_in_expressions",
              [](TestContext& t) {
                  // Power operator ^ is outside the integer compilable subset.
                  const Program p = prog("pow_op(X, Y, Z) :- Z is X ^ Y.\n");
                  const PredicateSignature sig{
                      .name = "pow_op",
                      .modes = {ArgMode::input, ArgMode::input, ArgMode::output},
                  };

                  auto comp = compile(p, sig, Target::cpp);
                  t.expect(!comp.has_value() && comp.error() == MathError::not_implemented,
                           "unsupported operator ^ is refused as not_implemented");

                  auto check = is_compilable(p, sig, Target::cpp);
                  t.expect(!check.has_value() && check.error() == MathError::not_implemented,
                           "is_compilable returns not_implemented for unsupported operator ^");
              })
        .test("compile_and_is_compilable_refuse_empty_entry_names",
              [](TestContext& t) {
                  const Program p = eval_poly_program();
                  const PredicateSignature sig{
                      .name = "",
                      .modes = {ArgMode::input, ArgMode::output},
                  };

                  auto comp = compile(p, sig, Target::cpp);
                  t.expect(!comp.has_value() && comp.error() == MathError::domain_error,
                           "empty entry name is refused as domain_error");

                  auto check = is_compilable(p, sig, Target::cpp);
                  t.expect(!check.has_value() && check.error() == MathError::domain_error,
                           "is_compilable returns domain_error for empty entry name");
              })
        .test("compile_and_is_compilable_refuse_undefined_entry_predicates",
              [](TestContext& t) {
                  const Program p = prog("foo(1).\n");
                  const PredicateSignature sig{
                      .name = "bar",
                      .modes = {ArgMode::output},
                  };

                  auto comp = compile(p, sig, Target::cpp);
                  t.expect(!comp.has_value() && comp.error() == MathError::domain_error,
                           "undefined entry predicate is refused as domain_error");

                  auto check = is_compilable(p, sig, Target::cpp);
                  t.expect(!check.has_value() && check.error() == MathError::domain_error,
                           "is_compilable returns domain_error for undefined entry predicate");
              })
        .test("compile_and_is_compilable_refuse_arity_mismatches",
              [](TestContext& t) {
                  const Program p = eval_poly_program();  // arity 2

                  const PredicateSignature arity_1{
                      .name = "eval_poly",
                      .modes = {ArgMode::input},
                  };
                  auto comp1 = compile(p, arity_1, Target::cpp);
                  t.expect(!comp1.has_value() && comp1.error() == MathError::domain_error,
                           "signature with too few arguments is refused as domain_error");

                  const PredicateSignature arity_3{
                      .name = "eval_poly",
                      .modes = {ArgMode::input, ArgMode::output, ArgMode::output},
                  };
                  auto comp3 = compile(p, arity_3, Target::cpp);
                  t.expect(!comp3.has_value() && comp3.error() == MathError::domain_error,
                           "signature with too many arguments is refused as domain_error");

                  auto check = is_compilable(p, arity_3, Target::cpp);
                  t.expect(!check.has_value() && check.error() == MathError::domain_error,
                           "is_compilable returns domain_error for arity mismatch");
              })
        .test("compile_refuses_incompatible_width_and_target_combinations",
              [](TestContext& t) {
                  const Program p = eval_poly_program();
                  const PredicateSignature sig{
                      .name = "eval_poly",
                      .modes = {ArgMode::input, ArgMode::output},
                  };

                  // Triton supports only 64-bit integers.
                  CompileOptions triton_128;
                  triton_128.target = Target::triton;
                  triton_128.width = Width::bits128;
                  auto res_t128 = compile(p, sig, triton_128);
                  t.expect(!res_t128.has_value() && res_t128.error() == MathError::not_implemented,
                           "Triton refuses bits128 as not_implemented");

                  CompileOptions triton_big;
                  triton_big.target = Target::triton;
                  triton_big.width = Width::arbitrary;
                  auto res_tbig = compile(p, sig, triton_big);
                  t.expect(!res_tbig.has_value() && res_tbig.error() == MathError::not_implemented,
                           "Triton refuses arbitrary precision as not_implemented");

                  // CUDA cannot allocate device memory dynamically for BigInt.
                  CompileOptions cuda_big;
                  cuda_big.target = Target::cuda;
                  cuda_big.width = Width::arbitrary;
                  auto res_cbig = compile(p, sig, cuda_big);
                  t.expect(!res_cbig.has_value() && res_cbig.error() == MathError::not_implemented,
                           "CUDA refuses arbitrary precision BigInt as not_implemented");
              })
        .test("decimal128_emits_a_scaled_fixed_point_type_that_is_exact_or_refuses",
              [](TestContext& t) {
                  // Money at a fixed scale. A value is an integer count of minor units, so the
                  // arithmetic is integer arithmetic plus a rescale on multiply and divide --
                  // and the rescale is exact or the operation reports that it is not.
                  const Program p = prog("price(A, B, C) :- C is A * B.\n");
                  const PredicateSignature sig{
                      .name = "price",
                      .modes = {ArgMode::input, ArgMode::input, ArgMode::output},
                  };
                  CompileOptions opts;
                  opts.width = Width::decimal128;
                  opts.decimal_scale = 2;

                  auto src = compile(p, sig, opts);
                  t.expect(src.has_value(), "compilation at decimal128 succeeds");
                  if (!src.has_value()) {
                      return;
                  }
                  t.expect(src->contains("using nc_int = __int128;"),
                           "the representation is 128 bits wide");
                  t.expect(src->contains("inline constexpr std::int32_t nc_scale_digits = 2;"),
                           "the scale is stated in the generated file, not left to the caller");
                  t.expect(src->contains("inline constexpr nc_int nc_scale_v = "
                                         "static_cast<nc_int>(100LL);"),
                           "and carried as the power of ten the arithmetic divides by");
                  t.expect(src->contains("if (prod % nc_scale_v != 0) { return std::nullopt; }"),
                           "a product the scale cannot hold exactly is refused, not rounded");
                  t.expect(src->contains("nc_to_string"),
                           "a scaled integer can be rendered as the decimal it means");

                  // The scale must reach the output: a different request is a different file.
                  CompileOptions four;
                  four.width = Width::decimal128;
                  four.decimal_scale = 4;
                  auto wider = compile(p, sig, four);
                  t.expect(wider.has_value() &&
                               wider->contains("static_cast<nc_int>(10000LL)"),
                           "a scale of four divides by ten thousand");

                  t.expect(to_string_view(Width::decimal128) == "decimal128",
                           "Width::decimal128 names itself");
              })
        .test("decimal128_refuses_mod_rem_an_out_of_range_scale_and_triton",
              [](TestContext& t) {
                  // `mod` and `rem` have no scale-independent meaning: the remainder of one
                  // amount divided by another differs depending on whether the quotient is an
                  // integer count or a decimal. Refusing is the honest answer; choosing one
                  // silently is the plausible-looking wrong one.
                  const Program modp = prog("leftover(A, B, R) :- R is A mod B.\n");
                  const PredicateSignature msig{
                      .name = "leftover",
                      .modes = {ArgMode::input, ArgMode::input, ArgMode::output},
                  };
                  CompileOptions dec;
                  dec.width = Width::decimal128;
                  auto refused = compile(modp, msig, dec);
                  t.expect(!refused.has_value() &&
                               refused.error() == MathError::not_implemented,
                           "mod is refused at a decimal width");
                  t.expect(compile(modp, msig, CompileOptions{}).has_value(),
                           "and still compiles at the integer width, where it means something");

                  const Program p = prog("price(A, B, C) :- C is A * B.\n");
                  const PredicateSignature sig{
                      .name = "price",
                      .modes = {ArgMode::input, ArgMode::input, ArgMode::output},
                  };
                  CompileOptions too_wide;
                  too_wide.width = Width::decimal128;
                  too_wide.decimal_scale = 19;
                  auto wide = compile(p, sig, too_wide);
                  t.expect(!wide.has_value() && wide.error() == MathError::domain_error,
                           "a scale past eighteen digits is refused as a domain error");

                  CompileOptions negative;
                  negative.width = Width::decimal128;
                  negative.decimal_scale = -1;
                  t.expect(!compile(p, sig, negative).has_value(),
                           "and so is a negative one");

                  CompileOptions tri;
                  tri.target = Target::triton;
                  tri.width = Width::decimal128;
                  auto refused_triton = compile(p, sig, tri);
                  t.expect(!refused_triton.has_value() &&
                               refused_triton.error() == MathError::not_implemented,
                           "Triton refuses a decimal, whose lattice stops at tl.int64");

                  CompileOptions cuda;
                  cuda.target = Target::cuda;
                  cuda.width = Width::decimal128;
                  t.expect(compile(p, sig, cuda).has_value(),
                           "CUDA accepts one, because __int128 is synthesised there");
              })
        .test("a_clause_may_chain_two_predicate_calls",
              [](TestContext& t) {
                  // A VARIABLE BOUND BY A CALL IS BOUND. Mode inference used to seed its known
                  // set from the head's inputs and from earlier is/2 goals only, so the second
                  // call in a clause read its input as unbound, inferred the callee to have two
                  // outputs, and refused the program -- naming the symptom, an output no clause
                  // binds, rather than the cause. Any pipeline of predicates has this shape.
                  const Program p = prog(
                      "inner(X, Y) :- Y is X + 1.\n"
                      "outer_p(X, Y) :- Y is X * 2.\n"
                      "chain(X, Z) :- inner(X, T), outer_p(T, Z).\n");
                  const PredicateSignature sig{
                      .name = "chain",
                      .modes = {ArgMode::input, ArgMode::output},
                  };
                  auto src = compile(p, sig, CompileOptions{});
                  t.expect(src.has_value(), "a clause with two calls in it compiles");
                  if (!src.has_value()) {
                      return;
                  }
                  t.expect(src->contains("p_inner_2") && src->contains("p_outer_p_2"),
                           "and both callees are emitted");

                  // Distinct anonymous variables stay distinct. Each `_` carries its own
                  // generation, so noting one as bound cannot make the next read as an input.
                  const Program anon = prog(
                      "two(A, B) :- A is 1, B is 2.\n"
                      "useit(X) :- two(_, T), two(_, U), X is T + U.\n");
                  const PredicateSignature asig{.name = "useit", .modes = {ArgMode::output}};
                  t.expect(compile(anon, asig, CompileOptions{}).has_value(),
                           "two anonymous variables in one clause still compile");
              })
        .test("a_call_may_not_bind_the_same_variable_in_two_output_positions",
              [](TestContext& t) {
                  // `top(Z) :- pair(Z, Z).` asks in Prolog that pair's two answers UNIFY, and
                  // fails when they differ. This compiler has no unification to express that:
                  // it would emit `p_pair_2(v_Z, v_Z)`, let the second write win and report
                  // success. That is the plausible-looking wrong answer Rule 32 forbids, so the
                  // shape is refused -- as the same shape already is in a head.
                  const Program p = prog(
                      "pair(A, B) :- A is 1, B is 2.\n"
                      "top(Z) :- pair(Z, Z).\n");
                  const PredicateSignature sig{.name = "top", .modes = {ArgMode::output}};
                  auto refused = compile(p, sig, CompileOptions{});
                  t.expect(!refused.has_value() && refused.error() == MathError::domain_error,
                           "a call binding one variable twice is refused as a domain error");

                  // The same variable in two INPUT positions is fine: nothing is being unified,
                  // the callee simply receives the value twice.
                  const Program inputs = prog(
                      "sum(A, B, C) :- C is A + B.\n"
                      "twice(X, Y) :- sum(X, X, Y).\n");
                  const PredicateSignature isig{
                      .name = "twice",
                      .modes = {ArgMode::input, ArgMode::output},
                  };
                  t.expect(compile(inputs, isig, CompileOptions{}).has_value(),
                           "the same variable in two input positions still compiles");
              })
        .test("decimal_rendering_does_not_negate_the_most_negative_value",
              [](TestContext& t) {
                  // `-v` on the most negative 128-bit value overflows a signed integer, which is
                  // undefined behaviour. The magnitude has to be taken in the unsigned type.
                  const Program p = prog("price(A, B, C) :- C is A * B.\n");
                  const PredicateSignature sig{
                      .name = "price",
                      .modes = {ArgMode::input, ArgMode::input, ArgMode::output},
                  };
                  CompileOptions opts;
                  opts.width = Width::decimal128;
                  auto src = compile(p, sig, opts);
                  t.expect(src.has_value(), "the decimal width compiles");
                  if (!src.has_value()) {
                      return;
                  }
                  t.expect(!src->contains("static_cast<unsigned __int128>(negative ? -v : v)"),
                           "the signed value is not negated before the cast");
                  t.expect(src->contains("const auto unsigned_v = static_cast<unsigned __int128>(v);"),
                           "the cast happens first, and the negation is the unsigned one");
              })
        .test("compile_refuses_cps_style_on_non_cpp_targets",
              [](TestContext& t) {
                  const Program p = between_program();
                  const PredicateSignature sig{
                      .name = "between",
                      .modes = {ArgMode::input, ArgMode::input, ArgMode::output},
                  };

                  CompileOptions cuda_cps;
                  cuda_cps.target = Target::cuda;
                  cuda_cps.style = Style::continuation;
                  auto res_cuda = compile(p, sig, cuda_cps);
                  t.expect(!res_cuda.has_value() && res_cuda.error() == MathError::not_implemented,
                           "CUDA refuses continuation style as not_implemented");

                  CompileOptions triton_cps;
                  triton_cps.target = Target::triton;
                  triton_cps.style = Style::continuation;
                  auto res_triton = compile(p, sig, triton_cps);
                  t.expect(!res_triton.has_value() && res_triton.error() == MathError::not_implemented,
                           "Triton refuses continuation style as not_implemented");
              })
        .test("compile_and_is_compilable_refuse_recursion_on_triton",
              [](TestContext& t) {
                  // A Triton kernel executes elementwise and has no call stack. What a stack is
                  // needed FOR is re-entering a predicate, so recursion is what the target
                  // cannot express -- not calling as such, which is inlined.
                  const Program rec_p = fact_acc_program();
                  const PredicateSignature rec_sig{
                      .name = "fact_acc",
                      .modes = {ArgMode::input, ArgMode::input, ArgMode::output},
                  };
                  auto rec_comp = compile(rec_p, rec_sig, Target::triton);
                  t.expect(!rec_comp.has_value() && rec_comp.error() == MathError::not_implemented,
                           "Triton refuses recursion as not_implemented");
                  auto rec_check = is_compilable(rec_p, rec_sig, Target::triton);
                  t.expect(!rec_check.has_value() && rec_check.error() == MathError::not_implemented,
                           "is_compilable returns not_implemented for recursive predicate on Triton");

                  // MUTUAL recursion is the same refusal reached the long way round. Neither
                  // predicate names itself, so a check that looked for self-calls would inline
                  // this pair forever; what is refused is a CYCLE in the call graph.
                  const Program mutual_p = prog(
                      "ping(X, Y) :- X =< 0, Y is 0.\n"
                      "ping(X, Y) :- X > 0, N is X - 1, pong(N, Y).\n"
                      "pong(X, Y) :- ping(X, Y).\n");
                  const PredicateSignature mutual_sig{
                      .name = "ping",
                      .modes = {ArgMode::input, ArgMode::output},
                  };
                  auto mutual_comp = compile(mutual_p, mutual_sig, Target::triton);
                  t.expect(!mutual_comp.has_value() &&
                               mutual_comp.error() == MathError::not_implemented,
                           "Triton refuses mutual recursion as not_implemented");

                  // Repeated head variables on Triton are also refused.
                  const Program rep_p = prog("same(X, X).\n");
                  const PredicateSignature rep_sig{
                      .name = "same",
                      .modes = {ArgMode::input, ArgMode::input},
                  };
                  auto rep_comp = compile(rep_p, rep_sig, Target::triton);
                  t.expect(!rep_comp.has_value() && rep_comp.error() == MathError::not_implemented,
                           "Triton refuses repeated head variable as not_implemented");
              })
        .test("a_callee_reached_in_two_modes_is_refused_rather_than_compiled_as_the_first",
              [](TestContext& t) {
                  // Modes are read off the CALL SITE and one function is emitted per name and
                  // arity, so a predicate reached first as an output and then as an input used
                  // to be compiled as the first at both sites. Here that meant `seven(T)` with
                  // T bound to 3 emitting a call that OVERWROTE T with 7 -- so b/1 answered
                  // Z = 7 where Prolog runs `7 is 7` against a bound 3 and fails. A wrong
                  // answer, reported as a success, on every target.
                  //
                  // The call order matters: a/1 has to be reached first for its (output) modes
                  // to be the ones cached.
                  const Program p = prog(
                      "seven(V) :- V is 7.\n"
                      "a(Z) :- seven(Z).\n"
                      "b(Z) :- T is 3, seven(T), Z is T.\n"
                      "top(P, Q) :- b(Q), a(P).\n");
                  const PredicateSignature sig{
                      .name = "top",
                      .modes = {ArgMode::output, ArgMode::output},
                  };
                  for (const Target target : {Target::cpp, Target::cuda, Target::triton}) {
                      auto refused = compile(p, sig, target);
                      t.expect(!refused.has_value() &&
                                   refused.error() == MathError::not_implemented,
                               "a predicate called in two modes is refused, not mis-compiled");
                  }

                  // One mode, reached from two call sites, is still perfectly ordinary.
                  const Program fine = prog(
                      "sq(X, Y) :- Y is X * X.\n"
                      "both(A, B, S) :- sq(A, P), sq(B, Q), S is P + Q.\n");
                  const PredicateSignature fsig{
                      .name = "both",
                      .modes = {ArgMode::input, ArgMode::input, ArgMode::output},
                  };
                  t.expect(compile(fine, fsig, Target::cpp).has_value(),
                           "two call sites agreeing on the modes still compile");
              })
        .test("triton_inlines_a_non_recursive_call_under_its_own_frame_prefix",
              [](TestContext& t) {
                  // A call needs a stack only to be RE-ENTERED. One that cannot be is pasted
                  // into its caller instead, which is something a flat kernel can hold.
                  const Program p = multi_pred_program();
                  const PredicateSignature sig{
                      .name = "sum_of_squares",
                      .modes = {ArgMode::input, ArgMode::input, ArgMode::output},
                  };
                  auto src = compile(p, sig, Target::triton);
                  t.expect(src.has_value(), "a call to a non-recursive predicate reaches Triton");
                  if (!src.has_value()) {
                      return;
                  }

                  t.expect(src->contains("def p_sum_of_squares_3_kernel("),
                           "the entry predicate is the kernel");
                  t.expect(!src->contains("def p_square_2"),
                           "the callee is pasted in, not emitted as a function a kernel "
                           "would have to call");

                  // The two call sites are separate FRAMES. Both paste in the same clause,
                  // which writes X and Y, and a clause assigns its variables unmasked -- so
                  // without a per-frame prefix the second call would overwrite the first's
                  // value while the caller still had to add it.
                  t.expect(src->contains("f0_v_X") && src->contains("f1_v_X"),
                           "each call site is inlined under its own frame prefix");
                  t.expect(src->contains("f0_done") && src->contains("f1_done"),
                           "each inlined frame carries its own first-match-wins mask");
                  t.expect(src->contains("(f0_done & f1_done)"),
                           "the calling clause succeeds only on lanes where both calls did");

                  // And a callee with more than one clause has to choose between them inside
                  // the frame, exactly as the entry predicate does.
                  const Program guarded = prog(
                      "abs_val(X, Y) :- X < 0, Y is -X.\n"
                      "abs_val(X, Y) :- X >= 0, Y is X.\n"
                      "l1(A, B, S) :- abs_val(A, PA), abs_val(B, PB), S is PA + PB.\n");
                  const PredicateSignature lsig{
                      .name = "l1",
                      .modes = {ArgMode::input, ArgMode::input, ArgMode::output},
                  };
                  auto lsrc = compile(guarded, lsig, Target::triton);
                  t.expect(lsrc.has_value(), "a multi-clause callee is inlined too");
                  if (!lsrc.has_value()) {
                      return;
                  }
                  t.expect(lsrc->contains("f0_g0") && lsrc->contains("f0_g1"),
                           "both clauses of the callee are emitted inside the frame");
                  t.expect(lsrc->contains("f0_g1 = ") && lsrc->contains("& ~f0_done"),
                           "the frame's later clause is masked out where an earlier one won");
              })
        .test("compile_and_is_compilable_refuse_unsupported_body_goals",
              [](TestContext& t) {
                  // Atom goal other than true, !, fail, false.
                  const Program atom_p = prog("bad_goal(X) :- other_atom.\n");
                  const PredicateSignature atom_sig{
                      .name = "bad_goal",
                      .modes = {ArgMode::input},
                  };
                  auto atom_comp = compile(atom_p, atom_sig, Target::cpp);
                  t.expect(!atom_comp.has_value() && atom_comp.error() == MathError::not_implemented,
                           "unsupported atom goal is refused as not_implemented");

                  // Call to predicate not present in program.
                  const Program missing_p = prog("call_missing(X, Y) :- missing(X, Y).\n");
                  const PredicateSignature missing_sig{
                      .name = "call_missing",
                      .modes = {ArgMode::input, ArgMode::output},
                  };
                  auto missing_comp = compile(missing_p, missing_sig, Target::cpp);
                  // A callee that is not in the program is a fault in the INPUT, not a feature
                  // the emitter has yet to support, so it is a domain_error. What matters for
                  // the honesty invariant is that it is refused at all rather than emitted as
                  // source that could never compile.
                  t.expect(!missing_comp.has_value() && missing_comp.error() == MathError::domain_error,
                           "call to undefined callee is refused as domain_error");

                  // The same callee reached in two different modes. `square(X, Y)` asks for an
                  // answer and `square(X, 10)` checks one, and a compiler that emits a single
                  // function per name and arity cannot be both. It is refused as a thing this
                  // emitter does not do, not as a fault in the program -- the program is
                  // ordinary Prolog.
                  const Program const_out_p = prog(
                      "test_call(X, Y) :- square(X, Y), square(X, 10).\n"
                      "square(X, Y) :- Y is X * X.\n");
                  const PredicateSignature const_out_sig{
                      .name = "test_call",
                      .modes = {ArgMode::input, ArgMode::output},
                  };
                  auto const_out_comp = compile(const_out_p, const_out_sig, Target::cpp);
                  t.expect(!const_out_comp.has_value() &&
                               const_out_comp.error() == MathError::not_implemented,
                           "a callee reached in two different modes is refused");
              })
        .test("compile_refuses_output_only_variable_in_multiple_head_positions",
              [](TestContext& t) {
                  // Variable Z appears only in output positions without any goal defining it.
                  const Program p = prog("two_outs(Z, Z).\n");
                  const PredicateSignature sig{
                      .name = "two_outs",
                      .modes = {ArgMode::output, ArgMode::output},
                  };

                  auto comp = compile(p, sig, Target::cpp);
                  t.expect(!comp.has_value() && comp.error() == MathError::domain_error,
                           "unbound output variable appearing in multiple head positions is a domain_error");
              })
        .test("cps_refuses_head_argument_with_compound_term_and_unsupported_goals",
              [](TestContext& t) {
                  const Program bad_head = prog("cps_pair(pair(X, Y), Z) :- Z is X + Y.\n");
                  const PredicateSignature sig_pair{
                      .name = "cps_pair",
                      .modes = {ArgMode::input, ArgMode::output},
                  };
                  CompileOptions opts;
                  opts.target = Target::cpp;
                  opts.style = Style::continuation;

                  auto res_head = compile(bad_head, sig_pair, opts);
                  t.expect(!res_head.has_value() && res_head.error() == MathError::not_implemented,
                           "CPS refuses compound head arguments as not_implemented");

                  const Program bad_atom = prog("cps_atom(X) :- custom_atom.\n");
                  const PredicateSignature sig_atom{
                      .name = "cps_atom",
                      .modes = {ArgMode::input},
                  };
                  auto res_atom = compile(bad_atom, sig_atom, opts);
                  t.expect(!res_atom.has_value() && res_atom.error() == MathError::not_implemented,
                           "CPS refuses unsupported atom goals as not_implemented");
              })
        .run();
}
