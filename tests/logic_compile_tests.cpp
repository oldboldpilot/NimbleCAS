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
                  t.expect(src->contains("auto nc_lit(long long v) -> nc_int"),
                           "literal constructor nc_lit is emitted");
                  t.expect(src->contains("auto nc_add(nc_int a, nc_int b, nc_int& r) -> bool"),
                           "checked addition helper is emitted");
                  t.expect(src->contains("auto nc_sub(nc_int a, nc_int b, nc_int& r) -> bool"),
                           "checked subtraction helper is emitted");
                  t.expect(src->contains("auto nc_mul(nc_int a, nc_int b, nc_int& r) -> bool"),
                           "checked multiplication helper is emitted");
                  t.expect(src->contains("std::numeric_limits<nc_int>::max()"),
                           "pre-checked overflow guards check std::numeric_limits max");
                  t.expect(src->contains("[[nodiscard]] inline auto p_eval_poly_2(nc_int a0, nc_int& a1) -> bool"),
                           "entry predicate has nodiscard trailing return type and moded parameters");
                  t.expect(src->contains("#include <cstdint>"),
                           "cstdint header is included for std::int64_t");
                  t.expect(src->contains("#include <limits>"),
                           "limits header is included for std::numeric_limits");
                  t.expect(!src->contains("nimblecas.bigint"),
                           "64-bit build does not import bigint module");
                  t.expect(src->contains("// @author Olumuyiwa Oluwasanmi"),
                           "emitted file header preserves repository authorship");
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
                  t.expect(src->contains("return !__builtin_add_overflow(a, b, &r);"),
                           "checked addition uses __builtin_add_overflow");
                  t.expect(src->contains("return !__builtin_sub_overflow(a, b, &r);"),
                           "checked subtraction uses __builtin_sub_overflow");
                  t.expect(src->contains("return !__builtin_mul_overflow(a, b, &r);"),
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
                  t.expect(src->contains("r = a.add(b);"),
                           "addition delegates to BigInt::add without redundant overflow checks");
                  t.expect(src->contains("r = a.subtract(b);"),
                           "subtraction delegates to BigInt::subtract");
                  t.expect(src->contains("r = a.multiply(b);"),
                           "multiplication delegates to BigInt::multiply");
                  t.expect(src->contains("r = a.negate();"),
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

                  t.expect(src->contains("__device__ inline auto nc_add("),
                           "preamble helpers are decorated with __device__ inline");
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
                  t.expect(src->contains("__device__ inline auto nc_add"),
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
                  t.expect(src->contains("INT64_MAX = 9223372036854775807"),
                           "INT64_MAX constant boundary is defined for overflow checks");
                  t.expect(src->contains("INT64_MIN = -9223372036854775808"),
                           "INT64_MIN constant boundary is defined for overflow checks");
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

                  t.expect(src->contains("#include <thread>"),
                           "thread header is included for parallel batch processing");
                  t.expect(src->contains("#include <span>"),
                           "span header is included");
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
                           "predicate definition takes continuation argument k");
                  t.expect(src->contains("return k(v_Low);"),
                           "base clause invokes continuation with solution value v_Low");
                  t.expect(src->contains("return k(v_Out);"),
                           "recursive clause invokes continuation with solution value v_Out");
                  t.expect(src->contains("return p_between_3("),
                           "recursive call passes trailing lambda continuation");
                  t.expect(src->contains("[&](nc_int v_Out) -> bool"),
                           "continuation closure binds returned argument and chains the rest of the clause");
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

                  t.expect(src->contains("!nc_quot("),
                           "// operator emits checked nc_quot helper");
                  t.expect(src->contains("!nc_fdiv("),
                           "div operator emits checked nc_fdiv helper");
                  t.expect(src->contains("!nc_rem("),
                           "rem operator emits checked nc_rem helper");
                  t.expect(src->contains("!nc_mod("),
                           "mod operator emits checked nc_mod helper");
                  t.expect(src->contains("!nc_min("),
                           "min operator emits checked nc_min helper");
                  t.expect(src->contains("!nc_max("),
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
                           "integer division emits floor division // with guarded divisor");
                  t.expect(src->contains("% tl.where"),
                           "remainder emits modulo % with guarded divisor");
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
        .test("compile_and_is_compilable_refuse_recursion_and_calls_on_triton",
              [](TestContext& t) {
                  // Triton executes elementwise without a call stack; recursion is refused.
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

                  // Calls to secondary predicates are likewise refused on Triton.
                  const Program multi_p = multi_pred_program();
                  const PredicateSignature multi_sig{
                      .name = "sum_of_squares",
                      .modes = {ArgMode::input, ArgMode::input, ArgMode::output},
                  };
                  auto multi_comp = compile(multi_p, multi_sig, Target::triton);
                  t.expect(!multi_comp.has_value() && multi_comp.error() == MathError::not_implemented,
                           "Triton refuses callee calls as not_implemented");

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

                  // Callee call where output argument is an integer constant rather than a variable.
                  const Program const_out_p = prog(
                      "test_call(X, Y) :- square(X, Y), square(X, 10).\n"
                      "square(X, Y) :- Y is X * X.\n");
                  const PredicateSignature const_out_sig{
                      .name = "test_call",
                      .modes = {ArgMode::input, ArgMode::output},
                  };
                  auto const_out_comp = compile(const_out_p, const_out_sig, Target::cpp);
                  t.expect(!const_out_comp.has_value() &&
                               const_out_comp.error() == MathError::domain_error,
                           "callee call with non-variable output argument is refused as domain_error");
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
