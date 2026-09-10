// NimbleCAS Prolog compiler — Prolog clauses to C++, CUDA and Triton SOURCE.
// @author Olumuyiwa Oluwasanmi
//
// The interpreter in `nimblecas.logic` is general: it runs any Horn program, with backtracking,
// unification and a mutable database. That generality is exactly what makes it slow, and most
// of it is unnecessary for the programs people actually want to run fast — a numeric predicate
// over integers, called a million times.
//
// This module compiles that case to SOURCE TEXT you can hand to a real compiler:
//
//   * `Target::cpp`     — C++23 the repo's own clang++-23 build accepts, so a compiled
//                          predicate links into an ordinary library alongside everything else.
//   * `Target::cuda`    — a `__device__` function plus a `__global__` kernel that maps it over
//                          a batch, for nvcc.
//   * `Target::triton`  — a `@triton.jit` kernel, elementwise over int64 tensors.
//
// WHAT IT WILL AND WILL NOT COMPILE — the honest part, because a compiler that silently
// mis-compiles is worse than one that refuses.
//
// The compilable subset is INTEGER, MODED, DETERMINISTIC Prolog:
//   * every argument is an integer, and the caller states each argument's MODE (input or
//     output) — modes are not guessed, because guessing one wrong changes what the program
//     means rather than how fast it runs;
//   * head arguments are integer constants or distinct variables;
//   * bodies contain only `is/2`, the arithmetic comparisons, `true`, `!`, and calls to other
//     compilable predicates;
//   * every variable is bound before it is read.
// Anything else — a compound term, an unbound output that no goal binds, a call to `findall`,
// a database update, a non-integer constant — is refused with a MathError. It is NOT partially
// compiled and it is NOT quietly handed back to the interpreter.
//
// Additionally, `Target::triton` refuses RECURSION and calls: a Triton kernel is a flat
// elementwise program over a block of lanes, with no call stack to recurse on. Saying so is
// better than emitting something that looks like a kernel and deadlocks.
//
// THE GENERATED CODE IS HELD TO THE SAME STANDARD AS THIS REPO'S OWN. Trailing return types,
// `[[nodiscard]]`, no exceptions, no raw pointers — and, most importantly, Rule 32: the emitted
// arithmetic is OVERFLOW-CHECKED. A compiler that turns a checked interpreter into unchecked
// native code has not made the program faster, it has made it wrong. Every emitted predicate
// returns a bool that is false when it fails OR when its arithmetic could not be represented,
// which is the same distinction the interpreter draws between failure and an honest error.

export module nimblecas.logic_compile;

import std;
import nimblecas.core;
import nimblecas.logic;

export namespace nimblecas::logic_compile {

// The language to emit.
enum class Target : std::uint8_t { cpp, cuda, triton };

[[nodiscard]] constexpr auto to_string_view(Target t) noexcept -> std::string_view {
    switch (t) {
        case Target::cpp:    return "cpp";
        case Target::cuda:   return "cuda";
        case Target::triton: return "triton";
    }
    return "cpp";
}

// Whether an argument is supplied by the caller or computed by the predicate. There is no
// "either" — a moded compiler that has to discover the mode at run time is an interpreter.
enum class ArgMode : std::uint8_t { input, output };

// The predicate to compile, and the mode of each of its arguments.
struct PredicateSignature {
    std::string name;
    std::vector<ArgMode> modes;

    [[nodiscard]] auto arity() const -> std::size_t { return modes.size(); }
};

// The integer width the generated code computes in.
//
// This is a REAL choice, not a knob: a Prolog program that overflows 64 bits does not compute a
// different answer at 128, it computes the SAME answer that 64 bits could not hold. Widening is
// therefore about which programs the compiler can serve, and the honest positions are:
//
//   * `bits64`      — `std::int64_t`, every operation pre-checked. The default.
//   * `bits128`     — `__int128`, checked with the compiler's overflow builtins. Still BOUNDED:
//                     an operation leaving the range is an overflow, not a wrong answer. This is
//                     the same position `nimblecas.int128` takes, in the same words.
//   * `arbitrary`   — `nimblecas::BigInt`, where addition, subtraction and multiplication CANNOT
//                     fail because arbitrary precision cannot overflow, and only division can.
//
// Not every width fits every target, and `compile` refuses the combinations that do not rather
// than emitting something that wraps — see the notes on `CompileOptions::target`.
enum class Width : std::uint8_t { bits64, bits128, arbitrary };

[[nodiscard]] constexpr auto to_string_view(Width w) noexcept -> std::string_view {
    switch (w) {
        case Width::bits64:    return "bits64";
        case Width::bits128:   return "bits128";
        case Width::arbitrary: return "arbitrary";
    }
    return "bits64";
}

// How to compile.
//
// WIDTH AND TARGET DO NOT COMBINE FREELY, and the refusals are deliberate:
//   * Triton accepts `bits64` only. Its type lattice tops out at `tl.int64` — there is no
//     `tl.int128` and no arbitrary-precision tensor dtype — so any wider request is refused
//     rather than silently narrowed.
//   * CUDA accepts `bits64` and `bits128`. `__int128` compiles under nvcc but has no PTX
//     backing, so it is SYNTHESISED from 64-bit pieces; the generated header says so. Arbitrary
//     precision is refused outright: `BigInt` allocates a `std::vector` per operation and
//     device code cannot.
//   * C++ accepts all three. `arbitrary` emits a MODULE-importing translation unit (it imports
//     `nimblecas.bigint`) rather than a standalone one, because BigInt has no header form —
//     that is a real difference in how the output is built, and it is stated in the output.
// How solutions are delivered.
//
// `deterministic` compiles a predicate to a function returning its FIRST solution — the right
// shape for arithmetic, and the wrong one for anything that can succeed more than once.
// `continuation` compiles it to a function taking a callback invoked once per solution, so a
// nondeterministic predicate compiles honestly instead of being truncated to its first answer.
enum class Style : std::uint8_t { deterministic, continuation };

struct CompileOptions {
    Target target{Target::cpp};
    Width width{Width::bits64};
    Style style{Style::deterministic};
    // Compile a self tail call to a loop. On by default; the flag exists so the two forms can
    // be compared, not because recursion is ever preferable.
    bool tail_call_optimise{true};
    // Emit a BATCH entry point applying the predicate across many inputs, plus the SIMD
    // checked-arithmetic kernels it rests on.
    //
    // Lanes are INDEPENDENT — one call of the predicate per input, sharing nothing — so the
    // batch is a pure map and its result cannot depend on how the work was split. That is what
    // makes it safe to run across threads and what makes the parallel form testable against the
    // serial one.
    //
    // Only meaningful for `Target::cpp`: CUDA already emits a batch kernel of its own, and a
    // Triton kernel IS a batch by construction.
    bool emit_batch{false};
};

// Compiles `entry` (and every predicate it calls) out of `program` into source text for
// `target`.
//
// Returns `MathError::not_implemented` when the program is outside the compilable subset —
// with the whole program refused rather than the compilable part emitted, because a header
// containing half a program is a trap. Returns `MathError::domain_error` when the signature
// does not describe the program: a predicate that is not defined, an arity mismatch, or an
// output argument no clause ever binds.
[[nodiscard]] auto compile(const Program& program, const PredicateSignature& entry,
                           const CompileOptions& options) -> Result<std::string>;

// The common case: 64-bit, tail calls optimised.
[[nodiscard]] auto compile(const Program& program, const PredicateSignature& entry, Target target)
    -> Result<std::string>;

// Whether `entry` is compilable for `target`, without producing the source. Useful for
// deciding whether to compile or to interpret, and it returns the SAME error `compile` would,
// so the two can never disagree about what is supported.
[[nodiscard]] auto is_compilable(const Program& program, const PredicateSignature& entry,
                                 Target target) -> Result<void>;

}  // namespace nimblecas::logic_compile

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::logic_compile {

namespace {

using nimblecas::atom_of;
using nimblecas::compound_of;
using nimblecas::int_of;
using nimblecas::is_atom;
using nimblecas::is_compound;
using nimblecas::is_int;
using nimblecas::is_var;
using nimblecas::var_of;

// The arithmetic comparisons, which are goals rather than expressions.
[[nodiscard]] auto comparison_op(std::string_view name) -> std::optional<std::string_view> {
    if (name == "=:=") { return "=="; }
    if (name == "=\\=") { return "!="; }
    if (name == "<") { return "<"; }
    if (name == ">") { return ">"; }
    if (name == "=<") { return "<="; }
    if (name == ">=") { return ">="; }
    return std::nullopt;
}

// The binary arithmetic operators that map directly onto a checked helper.
[[nodiscard]] auto checked_binary(std::string_view name) -> std::optional<std::string_view> {
    if (name == "+") { return "add"; }
    if (name == "-") { return "sub"; }
    if (name == "*") { return "mul"; }
    if (name == "//") { return "quot"; }
    if (name == "div") { return "fdiv"; }
    if (name == "mod") { return "mod"; }
    if (name == "rem") { return "rem"; }
    if (name == "min") { return "min"; }
    if (name == "max") { return "max"; }
    return std::nullopt;
}

[[nodiscard]] auto predicate_key(std::string_view name, std::size_t arity) -> std::string {
    return std::format("{}/{}", name, arity);
}

// A legal identifier for a Prolog name. Prolog atoms can hold characters C++ cannot, so
// anything outside [A-Za-z0-9_] is escaped rather than dropped — dropping would let `a-b` and
// `a+b` compile to the same symbol.
[[nodiscard]] auto mangle(std::string_view name, std::size_t arity) -> std::string {
    std::string out = "p_";
    for (const char c : name) {
        const bool plain = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                           (c >= '0' && c <= '9') || c == '_';
        if (plain) {
            out.push_back(c);
        } else {
            out += std::format("_x{:02x}", static_cast<unsigned>(static_cast<unsigned char>(c)));
        }
    }
    return out + std::format("_{}", arity);
}

// Whether a Prolog variable is the ANONYMOUS one -- `_`, which binds nothing and is read by
// nobody, so a head argument holding it needs no binding emitted at all.
//
// The decision is made on the SOURCE name rather than on the mangled C++ identifier, and that
// distinction is load-bearing in both directions. The reader gives every `_` occurrence a
// distinct fresh name of the form `_$N`, precisely so that two anonymous variables in one
// clause never unify with each other; mangled, that is `v___N`, never the `v__` a bare `_`
// would produce -- so a check against the mangled spelling of `_` misses every anonymous
// variable that came from parsed source. Widening that check to the `v___` prefix instead
// would be worse: a legitimate named variable such as `_Acc` also mangles to `v___Acc` and
// would be silently treated as anonymous. `_$` cannot appear in a user-written variable,
// because `$` is not a legal character in an unquoted Prolog variable name, so testing the
// source name catches exactly the reader's anonymous variables and nothing else.
[[nodiscard]] auto is_anonymous_var(std::string_view name) -> bool {
    return name == "_" || name.starts_with("_$");
}

// A legal identifier for a Prolog variable, kept recognisable so the emitted source reads like
// the program it came from.
[[nodiscard]] auto mangle_var(const VarKey& v) -> std::string {
    std::string out = "v_";
    for (const char c : v.name) {
        const bool plain = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                           (c >= '0' && c <= '9') || c == '_';
        out.push_back(plain ? c : '_');
    }
    return out;
}

// One predicate's clauses, gathered out of the program in clause order.
struct PredicateDef {
    std::string name;
    std::size_t arity{};
    std::vector<Clause> clauses;
};

[[nodiscard]] auto gather(const Program& program, std::string_view name, std::size_t arity)
    -> PredicateDef {
    PredicateDef def{.name = std::string(name), .arity = arity, .clauses = {}};
    for (const Clause& c : program) {
        if ((is_compound(c.head) && compound_of(c.head).functor == name &&
             compound_of(c.head).args.size() == arity) ||
            (arity == 0 && is_atom(c.head) && atom_of(c.head).name == name)) {
            def.clauses.push_back(c);
        }
    }
    return def;
}

// ---------------------------------------------------------------------------
// Expression compilation.
// ---------------------------------------------------------------------------

// Emits the statements that evaluate arithmetic expression `t` into a fresh temporary, whose
// name is returned. `bound` names the variables already in scope; reading one that is not is a
// domain error, because the generated code would otherwise use an uninitialised value.
//
// `fail` is the statement that abandons the current clause — it differs per target, which is
// the only place the three backends disagree about expressions.
struct ExprEmitter {
    std::string* body;
    const std::set<std::string>* bound;
    std::string indent;
    std::string fail_stmt;
    std::uint32_t* temp_counter;
    bool device;  // emit CUDA __device__-callable helper names
};

// Emits a call to one checked arithmetic helper, binding its result to `tmp`.
//
// The two targets differ here and only here. C++ helpers return std::optional (Rules 9/32 of
// config/cpp_details.txt), so the emitted code tests the optional and then binds a const
// nc_int -- the value cannot be read without the check having happened, and the binding is
// const because nothing downstream assigns to a temporary. CUDA helpers take an out-parameter
// and return bool, because nvcc compiles the generated file as C++17 with no libcu++ assumed
// and std::optional is not device-callable there.
auto emit_checked_call(const ExprEmitter& e, std::string_view tmp, std::string_view fn,
                       std::string_view lhs, std::optional<std::string_view> rhs) -> void {
    const std::string args =
        rhs ? std::format("{}, {}", lhs, *rhs) : std::string{lhs};
    if (e.device) {
        *e.body += std::format("{}nc_int {} = nc_lit(0);\n", e.indent, tmp);
        *e.body += std::format("{}if (!{}({}, {})) {{ {} }}\n", e.indent, fn, args, tmp,
                               e.fail_stmt);
        return;
    }
    *e.body += std::format("{}const auto {}_r = {}({});\n", e.indent, tmp, fn, args);
    *e.body += std::format("{}if (!{}_r) {{ {} }}\n", e.indent, tmp, e.fail_stmt);
    *e.body += std::format("{}const nc_int {} = *{}_r;\n", e.indent, tmp, tmp);
}

[[nodiscard]] auto emit_expr(const ExprEmitter& e, const Term& t) -> Result<std::string> {
    if (is_int(t)) {
        std::string tmp = std::format("t{}", (*e.temp_counter)++);
        *e.body += std::format("{}const nc_int {} = nc_lit({});\n", e.indent, tmp,
                               int_of(t).value);
        return tmp;
    }
    if (is_var(t)) {
        std::string name = mangle_var(VarKey{.name = var_of(t).name,
                                             .generation = var_of(t).generation});
        if (!e.bound->contains(name)) {
            return make_error<std::string>(MathError::domain_error);
        }
        return name;
    }
    if (!is_compound(t)) {
        return make_error<std::string>(MathError::not_implemented);
    }
    const CompoundNode& c = compound_of(t);

    if (c.args.size() == 1 && c.functor == "-") {
        auto inner = emit_expr(e, c.args[0]);
        if (!inner) {
            return inner;
        }
        std::string tmp = std::format("t{}", (*e.temp_counter)++);
        emit_checked_call(e, tmp, "nc_neg", *inner, std::nullopt);
        return tmp;
    }
    if (c.args.size() == 2) {
        const auto op = checked_binary(c.functor);
        if (!op) {
            return make_error<std::string>(MathError::not_implemented);
        }
        auto lhs = emit_expr(e, c.args[0]);
        if (!lhs) {
            return lhs;
        }
        auto rhs = emit_expr(e, c.args[1]);
        if (!rhs) {
            return rhs;
        }
        std::string tmp = std::format("t{}", (*e.temp_counter)++);
        // Every operation goes through a checked helper. An overflow abandons the clause the
        // same way a failed comparison does, so a compiled predicate can never return a wrapped
        // value the interpreter would have refused to produce.
        emit_checked_call(e, tmp, std::format("nc_{}", *op), *lhs, *rhs);
        return tmp;
    }
    return make_error<std::string>(MathError::not_implemented);
}

// ---------------------------------------------------------------------------
// The C++ / CUDA preamble: the checked arithmetic the generated code rests on.
// ---------------------------------------------------------------------------

// How a checked operation is spelled in the generated code.
//
// The two targets cannot share one shape. C++ gets `std::optional<nc_int> nc_add(a, b)` --
// Rule 9 of config/cpp_details.txt, and railway-oriented per Rule 32: the absence of a value
// IS the overflow report, so there is no way to read a result without having looked at
// whether there is one. CUDA gets the out-parameter form, because nvcc compiles the emitted
// file as C++17 with no libcu++ assumed, where std::optional is not device-callable.
struct OpStyle {
    bool optional;  // C++: return std::optional<nc_int>. CUDA: bool + nc_int& out-parameter.
    std::string qualifiers;  // "[[nodiscard]] constexpr " / "__device__ inline " / ...
};

// One checked operation, rendered in whichever shape the target needs.
//
// `body` is written once, against two names: `NC_OK(expr)` succeeds with a value and
// `NC_FAIL` abandons. Rendering substitutes the target's spelling for each, so the guards
// themselves are written down once and cannot drift apart between the two backends.
[[nodiscard]] auto render_op(const OpStyle& style, std::string_view name,
                             std::string_view params_by_value, std::string_view params_by_ref,
                             std::string_view body) -> std::string {
    std::string out = style.qualifiers;
    if (style.optional) {
        out += std::format("auto {}({}) -> std::optional<nc_int> {{\n", name, params_by_value);
    } else {
        out += std::format("auto {}({}, nc_int& r) -> bool {{\n", name, params_by_ref);
    }
    std::string rendered{body};
    const std::string ok_open = style.optional ? "return " : "r = ";
    const std::string ok_close = style.optional ? "" : "; return true";
    const std::string fail = style.optional ? "return std::nullopt" : "return false";
    // NC_OK(x) -> `return x` / `r = x; return true`; NC_FAIL -> `return std::nullopt` / false.
    for (std::size_t at = rendered.find("NC_OK("); at != std::string::npos;
         at = rendered.find("NC_OK(", at)) {
        std::size_t depth = 0;
        std::size_t i = at + 5;
        for (; i < rendered.size(); ++i) {
            if (rendered[i] == '(') { ++depth; }
            if (rendered[i] == ')') {
                --depth;
                if (depth == 0) { break; }
            }
        }
        std::string replacement = ok_open;
        replacement += rendered.substr(at + 6, i - at - 6);
        replacement += ok_close;
        rendered.replace(at, i - at + 1, replacement);
    }
    for (std::size_t at = rendered.find("NC_FAIL"); at != std::string::npos;
         at = rendered.find("NC_FAIL", at)) {
        rendered.replace(at, 7, fail);
    }
    out += rendered;
    out += "}\n\n";
    return out;
}

// The representable range, as named constants rather than as calls.
//
// std::numeric_limits<T>::min() is a constexpr __host__ function, and nvcc refuses to call
// one from __device__ code unless the consumer passes --expt-relaxed-constexpr -- which a
// generated file has no way to arrange. Naming the bounds once lets the overflow guards read
// identically in both targets while each gets an initialiser its own compiler accepts. For
// C++ that initialiser is still std::numeric_limits, which is where the value belongs.
// The width-generic checked-arithmetic bodies: guards written against nc_max_v / nc_min_v,
// valid for any signed two's-complement type. The 64-bit width always used these; the 128-bit
// CUDA path needs them too, because nvcc treats __builtin_add_overflow as __host__ and will
// not call it from device code.
struct GenericOps {
    std::string add;
    std::string sub;
    std::string neg;
    std::string mul;
};

[[nodiscard]] auto generic_ops() -> GenericOps {
    return GenericOps{
        .add = "    if (b > 0 && a > nc_max_v - b) { NC_FAIL; }\n"
               "    if (b < 0 && a < nc_min_v - b) { NC_FAIL; }\n"
               "    NC_OK(a + b);\n",
        .sub = "    if (b < 0 && a > nc_max_v + b) { NC_FAIL; }\n"
               "    if (b > 0 && a < nc_min_v + b) { NC_FAIL; }\n"
               "    NC_OK(a - b);\n",
        .neg = "    if (a == nc_min_v) { NC_FAIL; }\n"
               "    NC_OK(-a);\n",
        .mul = "    if (a == 0 || b == 0) { NC_OK(nc_int{0}); }\n"
               "    // Negation is written out rather than delegated to nc_neg: the two\n"
               "    // targets call that helper differently, and a shared body must not\n"
               "    // depend on which one it landed in.\n"
               "    if (a == -1) { if (b == nc_min_v) { NC_FAIL; } NC_OK(-b); }\n"
               "    if (b == -1) { if (a == nc_min_v) { NC_FAIL; } NC_OK(-a); }\n"
               "    if (a > 0 ? (b > 0 ? a > nc_max_v / b : b < nc_min_v / a)\n"
               "              : (b > 0 ? a < nc_min_v / b : a < nc_max_v / b)) {\n"
               "        NC_FAIL;\n    }\n"
               "    NC_OK(a * b);\n",
    };
}

[[nodiscard]] auto bounds_constants(bool device, Width width) -> std::string {
    if (!device) {
        return "inline constexpr nc_int nc_max_v = std::numeric_limits<nc_int>::max();\n"
               "inline constexpr nc_int nc_min_v = std::numeric_limits<nc_int>::min();\n\n";
    }
    if (width == Width::bits64) {
        return "// Literals, not std::numeric_limits: see bounds_constants in the compiler.\n"
               "inline constexpr nc_int nc_max_v = 0x7fffffffffffffffLL;\n"
               "inline constexpr nc_int nc_min_v = -nc_max_v - 1;\n\n";
    }
    // 2^127 - 1, built through the UNSIGNED type: shifting a 1 into a signed sign bit is not
    // something to write down on purpose.
    return "// Literals, not std::numeric_limits: see bounds_constants in the compiler.\n"
           "inline constexpr nc_int nc_max_v =\n"
           "    static_cast<nc_int>((static_cast<unsigned __int128>(1) << 127) - 1);\n"
           "inline constexpr nc_int nc_min_v = -nc_max_v - 1;\n\n";
}

[[nodiscard]] auto checked_preamble(bool device, Width width) -> std::string {
    // Rules 8/10/42: pure arithmetic is constexpr and its result must not be discarded --
    // a dropped nc_add return is a dropped overflow check. The CUDA helpers stay `inline`
    // rather than `constexpr` so nvcc needs no relaxed-constexpr flag from its caller.
    const OpStyle style{.optional = !device,
                        .qualifiers = device ? "__device__ [[nodiscard]] inline "
                                             : "[[nodiscard]] constexpr "};
    // BigInt is not a literal type, so its helpers cannot be constexpr whatever the target.
    const OpStyle big_style{.optional = !device, .qualifiers = "[[nodiscard]] inline "};
    const OpStyle& st = width == Width::arbitrary ? big_style : style;
    const std::string val = width == Width::arbitrary ? "const nc_int& " : "nc_int ";
    const std::string a1 = val + "a";
    const std::string a2 = val + "a, " + val + "b";
    std::string s;

    // `nc_int` is the one place the width lives. Everything the compiler emits is written in
    // terms of it and of `nc_lit`, so widening is a change to this preamble rather than to the
    // code generator — and a generated file always says, in its own text, what it computes in.
    if (width == Width::bits64) {
        s += "// 64-bit arithmetic. Every operation is PRE-checked: signed overflow is undefined\n";
        s += "// behaviour in C++, so a guard spelled `a + b < a` is itself the bug. Reporting\n";
        s += "// the absence of a result rather than wrapping is what keeps the compiled program\n";
        s += "// as honest as the interpreter it came from — it refuses an unrepresentable value,\n";
        s += "// it does not invent one.\n";
        s += "using nc_int = std::int64_t;\n";
        s += bounds_constants(device, width);
        s += st.qualifiers + "auto nc_lit(std::int64_t v) -> nc_int { return v; }\n\n";
        const GenericOps g = generic_ops();
        s += render_op(st, "nc_add", a2, a2, g.add);
        s += render_op(st, "nc_sub", a2, a2, g.sub);
        s += render_op(st, "nc_neg", a1, a1, g.neg);
        s += render_op(st, "nc_mul", a2, a2, g.mul);
    } else if (width == Width::bits128) {
        s += "// 128-bit arithmetic, checked with the compiler's overflow builtins — the same\n";
        s += "// three that nimblecas.int128 uses, for the same reason.\n";
        s += "//\n";
        s += "// STILL BOUNDED. Widening to 128 bits does not make a program that overflows\n";
        s += "// correct; it makes a larger set of programs representable. An operation whose\n";
        s += "// exact result leaves the range is an OVERFLOW, not a wrong answer, and a caller\n";
        s += "// who sees one should move to arbitrary precision rather than assume the value.\n";
        s += "#if !defined(__SIZEOF_INT128__)\n";
        s += "#error \"generated for 128-bit arithmetic, which this compiler does not provide\"\n";
        s += "#endif\n";
        s += "using nc_int = __int128;\n";
        s += bounds_constants(device, width);
        s += st.qualifiers
             + "auto nc_lit(std::int64_t v) -> nc_int { return static_cast<nc_int>(v); }\n\n";
        if (device) {
            // nvcc treats the overflow builtins as __host__ and refuses to call them from
            // device code, so the device path uses the same width-generic guards the 64-bit
            // output has always used. They hold for any signed two's-complement type.
            const GenericOps g = generic_ops();
            s += render_op(st, "nc_add", a2, a2, g.add);
            s += render_op(st, "nc_sub", a2, a2, g.sub);
            s += render_op(st, "nc_neg", a1, a1, g.neg);
            s += render_op(st, "nc_mul", a2, a2, g.mul);
        } else {
            s += render_op(st, "nc_add", a2, a2,
                           "    nc_int out{};\n"
                           "    if (__builtin_add_overflow(a, b, &out)) { NC_FAIL; }\n"
                           "    NC_OK(out);\n");
            s += render_op(st, "nc_sub", a2, a2,
                           "    nc_int out{};\n"
                           "    if (__builtin_sub_overflow(a, b, &out)) { NC_FAIL; }\n"
                           "    NC_OK(out);\n");
            s += render_op(st, "nc_neg", a1, a1,
                           "    nc_int out{};\n"
                           "    if (__builtin_sub_overflow(static_cast<nc_int>(0), a, &out)) { NC_FAIL; }\n"
                           "    NC_OK(out);\n");
            s += render_op(st, "nc_mul", a2, a2,
                           "    nc_int out{};\n"
                           "    if (__builtin_mul_overflow(a, b, &out)) { NC_FAIL; }\n"
                           "    NC_OK(out);\n");
        }
    } else {
        s += "// ARBITRARY-PRECISION arithmetic, on nimblecas::BigInt.\n";
        s += "//\n";
        s += "// Addition, subtraction, multiplication and negation CANNOT FAIL here, and the\n";
        s += "// helpers below always yield a value. That is not laziness: arbitrary precision\n";
        s += "// has no range to leave, so a check would be theatre — and a reader who saw one\n";
        s += "// would reasonably conclude there was a failure mode to worry about.\n";
        s += "// Division still fails, because dividing by zero is undefined at every precision.\n";
        s += "using nc_int = nimblecas::BigInt;\n";
        s += st.qualifiers + "auto nc_lit(std::int64_t v) -> nc_int {\n";
        s += "    return nimblecas::BigInt::from_i64(v);\n}\n\n";
        s += render_op(st, "nc_add", a2, a2, "    NC_OK(a.add(b));\n");
        s += render_op(st, "nc_sub", a2, a2, "    NC_OK(a.subtract(b));\n");
        s += render_op(st, "nc_neg", a1, a1, "    NC_OK(a.negate());\n");
        s += render_op(st, "nc_mul", a2, a2, "    NC_OK(a.multiply(b));\n");
    }

    // Division is where the widths agree again: a zero divisor has no answer, and the four
    // Prolog operators differ only in which way they round.
    if (width == Width::arbitrary) {
        s += "// BigInt::divmod truncates toward zero and gives the remainder the DIVIDEND's\n";
        s += "// sign, so `//` and `rem` are direct while `div` and `mod` need the usual\n";
        s += "// correction when the signs differ and the division is not exact.\n";
        s += render_op(st, "nc_quot", a2, a2,
                       "    const auto d = a.divmod(b);\n    if (!d) { NC_FAIL; }\n"
                       "    NC_OK(d->first);\n");
        s += render_op(st, "nc_rem", a2, a2,
                       "    const auto d = a.divmod(b);\n    if (!d) { NC_FAIL; }\n"
                       "    NC_OK(d->second);\n");
        s += render_op(st, "nc_fdiv", a2, a2,
                       "    const auto d = a.divmod(b);\n    if (!d) { NC_FAIL; }\n"
                       "    nc_int out = d->first;\n"
                       "    if (!d->second.is_zero() && (a.is_negative() != b.is_negative())) {\n"
                       "        out = out.subtract(nc_lit(1));\n    }\n"
                       "    NC_OK(out);\n");
        s += render_op(st, "nc_mod", a2, a2,
                       "    const auto d = a.divmod(b);\n    if (!d) { NC_FAIL; }\n"
                       "    nc_int out = d->second;\n"
                       "    if (!out.is_zero() && (a.is_negative() != b.is_negative())) {\n"
                       "        out = out.add(b);\n    }\n"
                       "    NC_OK(out);\n");
        s += render_op(st, "nc_min", a2, a2, "    NC_OK((a < b) ? a : b);\n");
        s += render_op(st, "nc_max", a2, a2, "    NC_OK((a > b) ? a : b);\n");
        return s;
    }

    s += render_op(st, "nc_quot", a2, a2,
                   "    if (b == 0) { NC_FAIL; }\n"
                   "    if (a == nc_min_v && b == -1) { NC_FAIL; }\n"
                   "    NC_OK(a / b);\n");
    s += render_op(st, "nc_fdiv", a2, a2,
                   "    if (b == 0) { NC_FAIL; }\n"
                   "    if (a == nc_min_v && b == -1) { NC_FAIL; }\n"
                   "    nc_int out = a / b;\n"
                   "    if (a % b != 0 && ((a < 0) != (b < 0))) { --out; }\n"
                   "    NC_OK(out);\n");
    s += render_op(st, "nc_rem", a2, a2,
                   "    if (b == 0) { NC_FAIL; }\n"
                   "    if (b == 1 || b == -1) { NC_OK(nc_int{0}); }\n"
                   "    NC_OK(a % b);\n");
    s += render_op(st, "nc_mod", a2, a2,
                   "    if (b == 0) { NC_FAIL; }\n"
                   "    if (b == 1 || b == -1) { NC_OK(nc_int{0}); }\n"
                   "    nc_int out = a % b;\n"
                   "    if (out != 0 && ((a < 0) != (b < 0))) { out += b; }\n"
                   "    NC_OK(out);\n");
    s += render_op(st, "nc_min", a2, a2, "    NC_OK(a < b ? a : b);\n");
    s += render_op(st, "nc_max", a2, a2, "    NC_OK(a > b ? a : b);\n");
    return s;
}

// ---------------------------------------------------------------------------
// Clause and predicate compilation (C++ and CUDA share everything but a qualifier).
// ---------------------------------------------------------------------------

struct CompileCtx {
    const Program* program;
    std::map<std::string, PredicateSignature> signatures;  // key -> signature, in call order
    std::vector<std::string> order;                        // emission order, callees first
    bool device{};
    bool allow_calls{true};
    bool tail_call_optimise{true};
    Width width{Width::bits64};
};

// The C++ spelling of the generated code's integer type.
[[nodiscard]] auto int_type_of(Width w) -> std::string {
    switch (w) {
        case Width::bits64:    return "std::int64_t";
        case Width::bits128:   return "nc_int";  // an alias, so the #if guard sits in one place
        case Width::arbitrary: return "nimblecas::BigInt";
    }
    return "std::int64_t";
}

// An integer literal in the generated code's type. BigInt has no converting constructor from a
// built-in integer — `from_i64` is the factory — so a literal cannot simply be written out.
[[nodiscard]] auto literal_of(Width w, std::int64_t v) -> std::string {
    if (w == Width::arbitrary) {
        return std::format("nimblecas::BigInt::from_i64({})", v);
    }
    return std::format("{}", v);
}

// Whether the width's helpers can fail. Arbitrary precision CANNOT overflow, so `+`, `-` and
// `*` are total there and emitting a check around them would be theatre.
[[nodiscard]] auto width_can_overflow(Width w) -> bool { return w != Width::arbitrary; }

[[nodiscard]] auto signature_of(const CompileCtx& ctx, std::string_view name, std::size_t arity)
    -> const PredicateSignature* {
    const auto it = ctx.signatures.find(predicate_key(name, arity));
    return it == ctx.signatures.end() ? nullptr : &it->second;
}

// Whether `goal` is the LAST goal of `clause` and is a call to this same predicate whose
// output arguments are exactly the head's output variables.
//
// That is the shape a tail call has to have for the loop rewrite to be sound: the recursive
// call's answer IS this clause's answer, so there is nothing left to do after it returns and
// the frame can be reused rather than stacked. If the outputs differ — the caller post-processes
// the result — it is not a tail call and rewriting it would change the program.
[[nodiscard]] auto is_self_tail_call(const PredicateDef& def, const PredicateSignature& sig,
                                     const Clause& clause, const Term& goal) -> bool {
    if (clause.body.empty() || !is_compound(goal)) {
        return false;
    }
    if (!(&goal == &clause.body.back())) {
        return false;
    }
    const CompoundNode& g = compound_of(goal);
    if (g.functor != def.name || g.args.size() != def.arity) {
        return false;
    }
    const std::vector<Term> head_args =
        is_compound(clause.head) ? compound_of(clause.head).args : std::vector<Term>{};
    if (head_args.size() != sig.arity()) {
        return false;
    }
    for (std::size_t i = 0; i < sig.arity(); ++i) {
        if (sig.modes[i] != ArgMode::output) {
            continue;
        }
        if (!is_var(g.args[i]) || !is_var(head_args[i])) {
            return false;
        }
        if (var_of(g.args[i]).name != var_of(head_args[i]).name ||
            var_of(g.args[i]).generation != var_of(head_args[i]).generation) {
            return false;
        }
    }
    return true;
}

// Emits one clause.
//
// In ORDINARY form the clause becomes its own function returning bool. In TAIL-RECURSIVE form
// (`tro`) it becomes a `do { ... } while (false)` block to be dropped inside the predicate's
// loop: failing the clause `break`s to the next one, and a self tail call assigns the next
// iteration's arguments and asks the loop to go round again instead of recursing.
[[nodiscard]] auto emit_clause(const CompileCtx& ctx, const PredicateDef& def,
                               const PredicateSignature& sig, const Clause& clause,
                               std::size_t index, bool tro, bool& used_tail)
    -> Result<std::string> {
    const std::string fn = std::format("{}_c{}", mangle(def.name, def.arity), index);
    const std::string qualifier = ctx.device ? "__device__ inline " : "inline ";
    used_tail = false;

    std::string params;
    for (std::size_t i = 0; i < sig.arity(); ++i) {
        params += i == 0 ? "" : ", ";
        params += sig.modes[i] == ArgMode::input
                      ? std::format("nc_int a{}", i)
                      : std::format("nc_int& a{}", i);
    }

    std::string body;
    std::set<std::string> bound;
    std::uint32_t temp = 0;
    const std::string indent = tro ? "        " : "    ";
    // Inside the loop form a failed clause falls through to the next one; standalone it just
    // returns.
    const std::string fail = tro ? "break;" : "return false;";

    // --- head matching -------------------------------------------------------------------
    const std::vector<Term> head_args =
        is_compound(clause.head) ? compound_of(clause.head).args : std::vector<Term>{};
    if (head_args.size() != sig.arity()) {
        return make_error<std::string>(MathError::domain_error);
    }
    // Head matching, including a REPEATED variable.
    //
    // `count(0, A, A)` — an accumulator's base clause — repeats `A` across an input and an
    // output, and `p(X, X)` repeats it across two inputs. Neither needs general unification
    // once the modes are known: the first INPUT occurrence of a variable defines it, a later
    // input occurrence is an equality test against that definition, and an output occurrence is
    // an assignment made once the value exists. Refusing these would rule out the single most
    // common shape in the subset this compiler exists to serve.
    //
    // What is still refused is a variable appearing ONLY in output positions, which asks the
    // predicate to make two results equal without ever saying what they are.
    std::set<std::string> head_vars;
    for (std::size_t i = 0; i < head_args.size(); ++i) {
        const Term& h = head_args[i];
        if (is_int(h)) {
            if (sig.modes[i] == ArgMode::input) {
                body += std::format("{}if (a{} != nc_lit({})) {{ {} }}\n", indent, i,
                                    int_of(h).value, fail);
            } else {
                body += std::format("{}a{} = nc_lit({});\n", indent, i, int_of(h).value);
            }
            continue;
        }
        if (!is_var(h)) {
            return make_error<std::string>(MathError::not_implemented);
        }
        const std::string v = mangle_var(
            VarKey{.name = var_of(h).name, .generation = var_of(h).generation});
        if (is_anonymous_var(var_of(h).name)) {
            continue;  // the anonymous variable binds nothing and is read by nobody
        }
        if (sig.modes[i] == ArgMode::input) {
            if (bound.contains(v)) {
                // A second input occurrence: the two arguments must agree.
                body += std::format("{}if (a{} != {}) {{ {} }}\n", indent, i, v, fail);
            } else {
                body += std::format("{}const nc_int {} = a{};\n", indent, v, i);
                bound.insert(v);
            }
        }
        head_vars.insert(v);
        // An output head variable is assigned after the body, by which time either an input
        // occurrence or a goal has given it a value.
    }

    // --- body ----------------------------------------------------------------------------
    const ExprEmitter emitter{.body = &body,
                              .bound = &bound,
                              .indent = indent,
                              .fail_stmt = fail,
                              .temp_counter = &temp,
                              .device = ctx.device};

    for (const Term& goal : clause.body) {
        if (is_atom(goal)) {
            const std::string& g = atom_of(goal).name;
            if (g == "true" || g == "!") {
                // A cut in a deterministic predicate is a no-op: the emitted predicate already
                // commits to the first clause that succeeds.
                continue;
            }
            if (g == "fail" || g == "false") {
                body += std::format("{}return false;\n", indent);
                continue;
            }
            return make_error<std::string>(MathError::not_implemented);
        }
        if (!is_compound(goal)) {
            return make_error<std::string>(MathError::not_implemented);
        }
        const CompoundNode& g = compound_of(goal);

        if (g.functor == "is" && g.args.size() == 2) {
            auto value = emit_expr(emitter, g.args[1]);
            if (!value) {
                return make_error<std::string>(value.error());
            }
            if (!is_var(g.args[0])) {
                // `2 is 1+1` as a TEST rather than an assignment.
                auto lhs = emit_expr(emitter, g.args[0]);
                if (!lhs) {
                    return make_error<std::string>(lhs.error());
                }
                body += std::format("{}if ({} != {}) {{ {} }}\n", indent, *lhs,
                                    *value, fail);
                continue;
            }
            const std::string v = mangle_var(VarKey{.name = var_of(g.args[0]).name,
                                                    .generation = var_of(g.args[0]).generation});
            if (bound.contains(v)) {
                body += std::format("{}if ({} != {}) {{ {} }}\n", indent, v, *value, fail);
            } else {
                body += std::format("{}const nc_int {} = {};\n", indent, v, *value);
                bound.insert(v);
            }
            continue;
        }

        if (const auto cmp = comparison_op(g.functor); cmp && g.args.size() == 2) {
            auto lhs = emit_expr(emitter, g.args[0]);
            if (!lhs) {
                return make_error<std::string>(lhs.error());
            }
            auto rhs = emit_expr(emitter, g.args[1]);
            if (!rhs) {
                return make_error<std::string>(rhs.error());
            }
            body += std::format("{}if (!({} {} {})) {{ {} }}\n", indent, *lhs, *cmp,
                                *rhs, fail);
            continue;
        }

        // --- a SELF TAIL CALL becomes a jump, not a call ---
        //
        // The recursive call's answer is this clause's answer, so nothing remains to be done
        // after it returns and the frame can be reused. New arguments are computed into
        // temporaries FIRST and only then written back, because the expressions read the
        // arguments they are about to replace — `count(N1, A1, R)` with `N1 is N-1` would
        // otherwise compute A1 from the already-overwritten N.
        //
        // This is what turns a recursion bounded by the native stack into a loop bounded by
        // nothing, which for an accumulator predicate is the difference between working and
        // overflowing at a few hundred thousand iterations.
        if (tro && is_self_tail_call(def, sig, clause, goal)) {
            std::vector<std::string> next_values;
            for (std::size_t i = 0; i < g.args.size(); ++i) {
                if (sig.modes[i] != ArgMode::input) {
                    next_values.emplace_back();
                    continue;
                }
                auto a = emit_expr(emitter, g.args[i]);
                if (!a) {
                    return make_error<std::string>(a.error());
                }
                const std::string slot = std::format("n{}", temp++);
                body += std::format("{}const nc_int {} = {};\n", indent, slot, *a);
                next_values.push_back(slot);
            }
            for (std::size_t i = 0; i < g.args.size(); ++i) {
                if (sig.modes[i] == ArgMode::input) {
                    body += std::format("{}a{} = {};\n", indent, i, next_values[i]);
                }
            }
            body += std::format("{}nc_again = true;\n", indent);
            body += std::format("{}break;\n", indent);
            used_tail = true;
            continue;
        }

        // --- a call to another compiled predicate ---
        if (!ctx.allow_calls) {
            return make_error<std::string>(MathError::not_implemented);
        }
        const PredicateSignature* callee = signature_of(ctx, g.functor, g.args.size());
        if (callee == nullptr) {
            return make_error<std::string>(MathError::not_implemented);
        }
        std::string call_args;
        for (std::size_t i = 0; i < g.args.size(); ++i) {
            call_args += i == 0 ? "" : ", ";
            if (callee->modes[i] == ArgMode::input) {
                auto a = emit_expr(emitter, g.args[i]);
                if (!a) {
                    return make_error<std::string>(a.error());
                }
                call_args += *a;
                continue;
            }
            if (!is_var(g.args[i])) {
                return make_error<std::string>(MathError::not_implemented);
            }
            const std::string v =
                mangle_var(VarKey{.name = var_of(g.args[i]).name,
                                  .generation = var_of(g.args[i]).generation});
            if (!bound.contains(v)) {
                body += std::format("{}nc_int {} = nc_lit(0);\n", indent, v);
                bound.insert(v);
            }
            call_args += v;
        }
        body += std::format("{}if (!{}({})) {{ {} }}\n", indent,
                            mangle(g.functor, g.args.size()), call_args, fail);
    }

    // --- bind the outputs ------------------------------------------------------------------
    //
    // Skipped entirely for a clause that ended in a TAIL CALL. Such a clause has already jumped
    // back to the top of the loop, and its outputs are produced by whichever later iteration
    // finally matches a base clause — demanding that they be bound HERE would reject exactly
    // the accumulator predicates the tail-call rewrite exists to serve.
    if (!used_tail) {
        for (std::size_t i = 0; i < head_args.size(); ++i) {
            if (sig.modes[i] != ArgMode::output || !is_var(head_args[i])) {
                continue;
            }
            const std::string v =
                mangle_var(VarKey{.name = var_of(head_args[i]).name,
                                  .generation = var_of(head_args[i]).generation});
            if (is_anonymous_var(var_of(head_args[i]).name)) {
                // An anonymous variable in an OUTPUT position binds nothing, so there is no
                // value to hand back. Skipping it would leave the output argument untouched and
                // skipping it in continuation-passing style would reference an identifier that
                // was never declared. Both are the plausible-looking wrong answer Rule 32
                // forbids, so this is refused instead.
                return make_error<std::string>(MathError::domain_error);
            }
            if (!bound.contains(v)) {
                // No goal ever gives this output a value, so the clause cannot honestly claim
                // to compute it. Emitting a zero here is exactly the plausible-looking wrong
                // answer Rule 32 forbids.
                return make_error<std::string>(MathError::domain_error);
            }
            body += std::format("{}a{} = {};\n", indent, i, v);
        }
        body += std::format("{}return true;\n", indent);
    }

    if (tro) {
        return std::format("        do {{  // clause {}\n{}        }} while (false);\n", index,
                           body);
    }
    return std::format("[[nodiscard]] {}auto {}({}) -> bool {{\n{}}}\n", qualifier, fn, params,
                       body);
}

[[nodiscard]] auto emit_predicate(const CompileCtx& ctx, const PredicateDef& def,
                                  const PredicateSignature& sig) -> Result<std::string> {
    if (def.clauses.empty()) {
        return make_error<std::string>(MathError::domain_error);
    }
    const std::string qualifier = ctx.device ? "__device__ inline " : "inline ";
    const std::string fn = mangle(def.name, def.arity);

    std::string params;
    std::string args;
    for (std::size_t i = 0; i < sig.arity(); ++i) {
        params += i == 0 ? "" : ", ";
        args += i == 0 ? "" : ", ";
        params += sig.modes[i] == ArgMode::input ? std::format("nc_int a{}", i)
                                                 : std::format("nc_int& a{}", i);
        args += std::format("a{}", i);
    }

    // Does any clause end in a call to this predicate whose result IS this clause's result?
    // If so the whole predicate is emitted as a loop instead of as recursion.
    const bool tro = ctx.tail_call_optimise &&
                     std::ranges::any_of(def.clauses, [&](const Clause& c) -> bool {
                         return !c.body.empty() &&
                                is_self_tail_call(def, sig, c, c.body.back());
                     });

    std::string out;
    if (tro) {
        out += std::format(
            "// {}/{}: clauses in program order, first success wins. The self tail call in this\n"
            "// predicate is compiled to a LOOP — `nc_again` re-enters with the next arguments\n"
            "// rather than growing the native stack, so its recursion depth is unbounded.\n",
            def.name, def.arity);
        out += std::format("[[nodiscard]] {}auto {}({}) -> bool {{\n", qualifier, fn, params);
        out += "    for (;;) {\n";
        out += "        bool nc_again = false;\n";
        for (std::size_t i = 0; i < def.clauses.size(); ++i) {
            bool used = false;
            auto c = emit_clause(ctx, def, sig, def.clauses[i], i, true, used);
            if (!c) {
                return c;
            }
            out += *c;
        }
        out += "        if (nc_again) { continue; }\n";
        out += "        return false;\n";
        out += "    }\n}\n";
        return out;
    }

    for (std::size_t i = 0; i < def.clauses.size(); ++i) {
        bool used = false;
        auto c = emit_clause(ctx, def, sig, def.clauses[i], i, false, used);
        if (!c) {
            return c;
        }
        out += *c;
        out += '\n';
    }
    // Clauses are tried in program order and the first success wins — which is the whole of
    // the determinism assumption, made visible.
    out += std::format("// {}/{}: the clauses in program order; the first that succeeds wins.\n",
                       def.name, def.arity);
    out += std::format("[[nodiscard]] {}auto {}({}) -> bool {{\n", qualifier, fn, params);
    for (std::size_t i = 0; i < def.clauses.size(); ++i) {
        out += std::format("    if ({}_c{}({})) {{ return true; }}\n", fn, i, args);
    }
    out += "    return false;\n}\n";
    return out;
}

// ---------------------------------------------------------------------------
// Continuation-passing emission.
// ---------------------------------------------------------------------------
//
// The deterministic emitter above compiles a predicate to a function returning the FIRST
// solution. That is the right shape for arithmetic, and the wrong shape for anything that can
// succeed more than once — which is most of Prolog.
//
// CPS compiles a predicate to a function taking a CONTINUATION, called once per solution with
// the output arguments as values:
//
//     template <class K> auto p_between_3(nc_int lo, nc_int hi, K&& k) -> bool;
//
// The return value says whether the enumeration RAN TO COMPLETION: `false` means the
// continuation asked to stop, and that answer is propagated outward untouched. That single
// convention gives `once/1`, `findall/3` and cut the same implementation — stop, or keep going.
//
// A call to another predicate becomes a NESTED CONTINUATION holding the rest of the clause, so
// backtracking into that call is just its own loop resuming. That is the whole transform, and
// it is why CPS needs no explicit choice-point stack: the C++ call stack is the choice-point
// stack.

// Emits the goals of one clause from `pos` onward, with the remainder of the clause nested
// inside any call's continuation. `on_success` is the statement that reports a solution.
[[nodiscard]] auto emit_cps_goals(const CompileCtx& ctx, const PredicateSignature& sig,
                                  const Clause& clause, std::size_t pos,
                                  std::set<std::string> bound, const std::string& indent,
                                  std::uint32_t& temp, const std::string& on_success)
    -> Result<std::string> {
    if (pos >= clause.body.size()) {
        // Indented HERE rather than at construction: the success statement lands at whatever
        // depth the nested continuations have reached, which the caller cannot know.
        return indent + on_success;
    }
    const Term& goal = clause.body[pos];
    std::string body;
    // Failing a goal abandons THIS branch, not the enumeration: returning true means "no stop
    // was requested", so the caller keeps offering solutions.
    const std::string fail = "return true;";

    if (is_atom(goal)) {
        const std::string& g = atom_of(goal).name;
        if (g == "true") {
            return emit_cps_goals(ctx, sig, clause, pos + 1, std::move(bound), indent, temp,
                                  on_success);
        }
        if (g == "fail" || g == "false") {
            return std::format("{}return true;\n", indent);
        }
        if (g == "!") {
            // Cut: record it, then carry on. The clause loop reads the flag after this clause
            // returns and stops offering later clauses — cut prunes alternatives, it does not
            // stop the solution that reached it.
            auto rest = emit_cps_goals(ctx, sig, clause, pos + 1, std::move(bound), indent, temp,
                                       on_success);
            if (!rest) {
                return rest;
            }
            return std::format("{}nc_cut = true;\n{}", indent, *rest);
        }
        return make_error<std::string>(MathError::not_implemented);
    }
    if (!is_compound(goal)) {
        return make_error<std::string>(MathError::not_implemented);
    }
    const CompoundNode& g = compound_of(goal);

    const ExprEmitter emitter{.body = &body,
                              .bound = &bound,
                              .indent = indent,
                              .fail_stmt = fail,
                              .temp_counter = &temp,
                              .device = ctx.device};

    if (g.functor == "is" && g.args.size() == 2 && is_var(g.args[0])) {
        auto value = emit_expr(emitter, g.args[1]);
        if (!value) {
            return make_error<std::string>(value.error());
        }
        const std::string v = mangle_var(
            VarKey{.name = var_of(g.args[0]).name, .generation = var_of(g.args[0]).generation});
        if (bound.contains(v)) {
            body += std::format("{}if ({} != {}) {{ {} }}\n", indent, v, *value, fail);
        } else {
            body += std::format("{}const nc_int {} = {};\n", indent, v, *value);
            bound.insert(v);
        }
        auto rest = emit_cps_goals(ctx, sig, clause, pos + 1, std::move(bound), indent, temp,
                                   on_success);
        if (!rest) {
            return rest;
        }
        return body + *rest;
    }

    if (const auto cmp = comparison_op(g.functor); cmp && g.args.size() == 2) {
        auto lhs = emit_expr(emitter, g.args[0]);
        if (!lhs) {
            return make_error<std::string>(lhs.error());
        }
        auto rhs = emit_expr(emitter, g.args[1]);
        if (!rhs) {
            return make_error<std::string>(rhs.error());
        }
        body += std::format("{}if (!({} {} {})) {{ {} }}\n", indent, *lhs, *cmp, *rhs, fail);
        auto rest = emit_cps_goals(ctx, sig, clause, pos + 1, std::move(bound), indent, temp,
                                   on_success);
        if (!rest) {
            return rest;
        }
        return body + *rest;
    }

    // --- a call: the rest of the clause becomes its continuation --------------------------
    const PredicateSignature* callee = signature_of(ctx, g.functor, g.args.size());
    if (callee == nullptr) {
        return make_error<std::string>(MathError::not_implemented);
    }
    std::string call_args;
    std::string lambda_params;
    std::set<std::string> inner = bound;
    for (std::size_t i = 0; i < g.args.size(); ++i) {
        if (callee->modes[i] == ArgMode::input) {
            auto a = emit_expr(emitter, g.args[i]);
            if (!a) {
                return make_error<std::string>(a.error());
            }
            call_args += call_args.empty() ? "" : ", ";
            call_args += *a;
            continue;
        }
        if (!is_var(g.args[i])) {
            return make_error<std::string>(MathError::not_implemented);
        }
        const std::string v = mangle_var(
            VarKey{.name = var_of(g.args[i]).name, .generation = var_of(g.args[i]).generation});
        lambda_params += lambda_params.empty() ? "" : ", ";
        lambda_params += std::format("nc_int {}", v);
        inner.insert(v);
    }
    const std::string deeper = indent + "    ";
    auto rest = emit_cps_goals(ctx, sig, clause, pos + 1, std::move(inner), deeper, temp,
                               on_success);
    if (!rest) {
        return rest;
    }
    call_args += call_args.empty() ? "" : ", ";
    body += std::format("{}return {}({}[&]({}) -> bool {{\n{}{}}});\n", indent,
                        mangle(g.functor, g.args.size()), call_args, lambda_params, *rest,
                        indent);
    return body;
}

[[nodiscard]] auto emit_cps_predicate(const CompileCtx& ctx, const PredicateDef& def,
                                      const PredicateSignature& sig) -> Result<std::string> {
    if (def.clauses.empty()) {
        return make_error<std::string>(MathError::domain_error);
    }
    const std::string fn = mangle(def.name, def.arity);

    std::string params;
    for (std::size_t i = 0; i < sig.arity(); ++i) {
        if (sig.modes[i] == ArgMode::input) {
            params += std::format("nc_int a{}, ", i);
        }
    }

    std::string out;
    out += std::format(
        "// {}/{} in continuation-passing form. `k` is called once per solution with the output\n"
        "// arguments; returning false from it stops the enumeration, and that answer travels\n"
        "// back out unchanged. The function itself returns whether the enumeration COMPLETED.\n",
        def.name, def.arity);
    out += std::format("template <class K>\n[[nodiscard]] auto {}({}K&& k) -> bool {{\n", fn,
                       params);
    out += "    bool nc_cut = false;\n";

    std::uint32_t temp = 0;
    for (std::size_t ci = 0; ci < def.clauses.size(); ++ci) {
        const Clause& clause = def.clauses[ci];
        const std::vector<Term> head_args =
            is_compound(clause.head) ? compound_of(clause.head).args : std::vector<Term>{};
        if (head_args.size() != sig.arity()) {
            return make_error<std::string>(MathError::domain_error);
        }
        const std::string indent = "        ";
        std::string head;
        std::set<std::string> bound;
        const std::string fail = "return true;";

        for (std::size_t i = 0; i < head_args.size(); ++i) {
            const Term& h = head_args[i];
            if (is_int(h)) {
                if (sig.modes[i] == ArgMode::input) {
                    head += std::format("{}if (a{} != nc_lit({})) {{ {} }}\n", indent, i,
                                        int_of(h).value, fail);
                }
                continue;
            }
            if (!is_var(h)) {
                return make_error<std::string>(MathError::not_implemented);
            }
            const std::string v =
                mangle_var(VarKey{.name = var_of(h).name, .generation = var_of(h).generation});
            if (is_anonymous_var(var_of(h).name)) {
                continue;
            }
            if (sig.modes[i] == ArgMode::input) {
                if (bound.contains(v)) {
                    head += std::format("{}if (a{} != {}) {{ {} }}\n", indent, i, v, fail);
                } else {
                    head += std::format("{}const nc_int {} = a{};\n", indent, v, i);
                    bound.insert(v);
                }
            }
        }

        // Reporting a solution: hand the continuation the outputs, in head order.
        std::string k_args;
        std::string prelude;
        for (std::size_t i = 0; i < head_args.size(); ++i) {
            if (sig.modes[i] != ArgMode::output) {
                continue;
            }
            k_args += k_args.empty() ? "" : ", ";
            if (is_int(head_args[i])) {
                k_args += std::format("nc_lit({})", int_of(head_args[i]).value);
                continue;
            }
            if (!is_var(head_args[i])) {
                return make_error<std::string>(MathError::not_implemented);
            }
            if (is_anonymous_var(var_of(head_args[i]).name)) {
                // An anonymous variable in an OUTPUT position binds nothing, so there is no
                // value to hand back. Skipping it would leave the output argument untouched and
                // skipping it in continuation-passing style would reference an identifier that
                // was never declared. Both are the plausible-looking wrong answer Rule 32
                // forbids, so this is refused instead.
                return make_error<std::string>(MathError::domain_error);
            }
            k_args += mangle_var(VarKey{.name = var_of(head_args[i]).name,
                                        .generation = var_of(head_args[i]).generation});
        }

        std::uint32_t clause_temp = temp;
        auto goals = emit_cps_goals(ctx, sig, clause, 0, bound, indent, clause_temp,
                                    std::format("return k({});\n", k_args));
        if (!goals) {
            return goals;
        }
        temp = clause_temp;

        out += std::format("    // clause {}\n", ci);
        out += "    if (![&]() -> bool {\n";
        out += head;
        out += *goals;
        out += "    }()) { return false; }\n";
        out += "    if (nc_cut) { return true; }\n";
    }
    out += "    return true;\n}\n";
    return out;
}

// Walks the call graph from `entry`, checking every predicate reached is defined and deriving
// each callee's modes. A callee's modes are taken from the CALL SITE: an argument that is an
// already-bound variable or an expression is an input, an unbound variable is an output. That
// is the only place modes are inferred, and it is inferable because the caller's own modes
// pin it down.
[[nodiscard]] auto collect_predicates(const Program& program, const PredicateSignature& entry,
                                      bool allow_calls, CompileCtx& ctx) -> Result<void> {
    std::vector<PredicateSignature> queue{entry};
    std::set<std::string> seen;

    while (!queue.empty()) {
        const PredicateSignature sig = queue.back();
        queue.pop_back();
        const std::string key = predicate_key(sig.name, sig.arity());
        if (seen.contains(key)) {
            continue;
        }
        seen.insert(key);
        const PredicateDef def = gather(program, sig.name, sig.arity());
        if (def.clauses.empty()) {
            return make_error<void>(MathError::domain_error);  // undefined predicate
        }
        ctx.signatures.emplace(key, sig);
        ctx.order.push_back(key);

        for (const Clause& c : def.clauses) {
            for (const Term& goal : c.body) {
                if (!is_compound(goal)) {
                    continue;
                }
                const CompoundNode& g = compound_of(goal);
                if (g.functor == "is" || comparison_op(g.functor)) {
                    continue;
                }
                if (!allow_calls) {
                    return make_error<void>(MathError::not_implemented);
                }
                // Modes at the call site: the head's input variables plus anything already
                // computed are inputs; a variable first seen here is an output.
                std::set<std::string> known;
                if (is_compound(c.head)) {
                    const auto& hargs = compound_of(c.head).args;
                    const PredicateSignature* hs = signature_of(ctx, sig.name, sig.arity());
                    for (std::size_t i = 0; i < hargs.size() && hs != nullptr; ++i) {
                        if (hs->modes[i] == ArgMode::input && is_var(hargs[i])) {
                            known.insert(mangle_var(
                                VarKey{.name = var_of(hargs[i]).name,
                                       .generation = var_of(hargs[i]).generation}));
                        }
                    }
                }
                for (const Term& earlier : c.body) {
                    if (&earlier == &goal) {
                        break;
                    }
                    if (is_compound(earlier) && compound_of(earlier).functor == "is" &&
                        is_var(compound_of(earlier).args[0])) {
                        const auto& v = var_of(compound_of(earlier).args[0]);
                        known.insert(
                            mangle_var(VarKey{.name = v.name, .generation = v.generation}));
                    }
                }
                PredicateSignature callee{.name = g.functor, .modes = {}};
                for (const Term& a : g.args) {
                    const bool is_out =
                        is_var(a) &&
                        !known.contains(mangle_var(VarKey{.name = var_of(a).name,
                                                          .generation = var_of(a).generation}));
                    callee.modes.push_back(is_out ? ArgMode::output : ArgMode::input);
                }
                queue.push_back(std::move(callee));
            }
        }
    }
    return {};
}

// ---------------------------------------------------------------------------
// Triton emission.
// ---------------------------------------------------------------------------

// A Triton value: the expression naming it, and the mask naming the lanes where computing it
// was VALID. Triton has no traps and no early exit — every lane executes every instruction — so
// "this lane overflowed" has to be carried alongside the value as a mask rather than raised.
// That is what keeps the emitted kernel as honest as the interpreter: a lane whose arithmetic
// could not be represented is reported as failed, never as a wrapped number.
struct TritonValue {
    std::string expr;
    std::string ok;  // a mask expression; "" means unconditionally valid
};

struct TritonEmitter {
    std::string* body;
    const std::set<std::string>* bound;
    std::uint32_t* temp_counter;
};

[[nodiscard]] auto tri_and(const std::string& a, const std::string& b) -> std::string {
    if (a.empty()) {
        return b;
    }
    if (b.empty()) {
        return a;
    }
    return std::format("({} & {})", a, b);
}

[[nodiscard]] auto emit_triton_expr(const TritonEmitter& e, const Term& t)
    -> Result<TritonValue> {
    if (is_int(t)) {
        return TritonValue{.expr = std::format("tl.full(shape, {}, tl.int64)", int_of(t).value),
                           .ok = ""};
    }
    if (is_var(t)) {
        const std::string v =
            mangle_var(VarKey{.name = var_of(t).name, .generation = var_of(t).generation});
        if (!e.bound->contains(v)) {
            return make_error<TritonValue>(MathError::domain_error);
        }
        return TritonValue{.expr = v, .ok = ""};
    }
    if (!is_compound(t)) {
        return make_error<TritonValue>(MathError::not_implemented);
    }
    const CompoundNode& c = compound_of(t);

    if (c.args.size() == 1 && c.functor == "-") {
        auto inner = emit_triton_expr(e, c.args[0]);
        if (!inner) {
            return inner;
        }
        const std::string n = std::format("t{}", (*e.temp_counter)++);
        *e.body += std::format("    {}_ok = ({} != INT64_MIN)\n", n, inner->expr);
        *e.body += std::format("    {} = -{}\n", n, inner->expr);
        return TritonValue{.expr = n, .ok = tri_and(inner->ok, std::format("{}_ok", n))};
    }
    if (c.args.size() != 2) {
        return make_error<TritonValue>(MathError::not_implemented);
    }
    auto lhs = emit_triton_expr(e, c.args[0]);
    if (!lhs) {
        return lhs;
    }
    auto rhs = emit_triton_expr(e, c.args[1]);
    if (!rhs) {
        return rhs;
    }
    const std::string n = std::format("t{}", (*e.temp_counter)++);
    const std::string both = tri_and(lhs->ok, rhs->ok);

    if (c.functor == "+") {
        *e.body += std::format(
            "    {}_ok = ~(({} > 0) & ({} > INT64_MAX - {})) & ~(({} < 0) & ({} < INT64_MIN - {}))\n",
            n, rhs->expr, lhs->expr, rhs->expr, rhs->expr, lhs->expr, rhs->expr);
        *e.body += std::format("    {} = {} + {}\n", n, lhs->expr, rhs->expr);
    } else if (c.functor == "-") {
        *e.body += std::format(
            "    {}_ok = ~(({} < 0) & ({} > INT64_MAX + {})) & ~(({} > 0) & ({} < INT64_MIN + {}))\n",
            n, rhs->expr, lhs->expr, rhs->expr, rhs->expr, lhs->expr, rhs->expr);
        *e.body += std::format("    {} = {} - {}\n", n, lhs->expr, rhs->expr);
    } else if (c.functor == "*") {
        // Division by the operand is the only overflow test that never itself overflows, and a
        // zero operand has to be excluded from it before it is used as a divisor.
        // The round-trip test (a*b)//b == a detects ordinary overflow, but NOT INT64_MIN * -1:
        // the division that would reveal it is itself unrepresentable and wraps back to
        // INT64_MIN, so the check passes for an operation that overflowed. That pair is
        // excluded explicitly.
        *e.body += std::format(
            "    {}_ok = (({} == 0) | ({} == 0) | ((({} * {}) // tl.where({} == 0, 1, {})) == "
            "{})) & (({} != -9223372036854775808) | ({} != -1)) & (({} != "
            "-9223372036854775808) | ({} != -1))\n",
            n, lhs->expr, rhs->expr, lhs->expr, rhs->expr, rhs->expr, rhs->expr, lhs->expr,
            lhs->expr, rhs->expr, rhs->expr, lhs->expr);
        *e.body += std::format("    {} = {} * {}\n", n, lhs->expr, rhs->expr);
    } else if (c.functor == "min" || c.functor == "max") {
        const std::string cmp = c.functor == "min" ? "<" : ">";
        *e.body += std::format("    {}_ok = tl.full(shape, 1, tl.int1)\n", n);
        *e.body += std::format("    {} = tl.where({} {} {}, {}, {})\n", n, lhs->expr, cmp,
                               rhs->expr, lhs->expr, rhs->expr);
    } else if (c.functor == "//" || c.functor == "mod" || c.functor == "rem" ||
               c.functor == "div") {
        // Division by zero is not the only unrepresentable case: INT64_MIN / -1 has no
        // int64 quotient, and on a GPU it wraps silently instead of trapping. Excluding it
        // here keeps the `ok` mask honest rather than letting a wrapped value through under a
        // mask that says the operation succeeded.
        *e.body += std::format(
            "    {}_ok = ({} != 0) & (({} != -9223372036854775808) | ({} != -1))\n", n,
            rhs->expr, lhs->expr, rhs->expr);
        const std::string safe = std::format("tl.where({} == 0, 1, {})", rhs->expr, rhs->expr);
        if (c.functor == "//" || c.functor == "div") {
            *e.body += std::format("    {} = {} // {}\n", n, lhs->expr, safe);
        } else {
            *e.body += std::format("    {} = {} % {}\n", n, lhs->expr, safe);
        }
    } else {
        return make_error<TritonValue>(MathError::not_implemented);
    }
    return TritonValue{.expr = n, .ok = tri_and(both, std::format("{}_ok", n))};
}

[[nodiscard]] auto emit_triton(const Program& program, const PredicateSignature& sig)
    -> Result<std::string> {
    const PredicateDef def = gather(program, sig.name, sig.arity());
    if (def.clauses.empty()) {
        return make_error<std::string>(MathError::domain_error);
    }
    const std::string fn = mangle(sig.name, sig.arity());

    std::string clauses_src;
    std::uint32_t temp = 0;

    for (std::size_t ci = 0; ci < def.clauses.size(); ++ci) {
        const Clause& clause = def.clauses[ci];
        const std::vector<Term> head_args =
            is_compound(clause.head) ? compound_of(clause.head).args : std::vector<Term>{};
        if (head_args.size() != sig.arity()) {
            return make_error<std::string>(MathError::domain_error);
        }
        std::string body;
        std::set<std::string> bound;
        std::string guard;
        std::set<std::string> head_vars;

        body += std::format("    # --- clause {} ---\n", ci);
        for (std::size_t i = 0; i < head_args.size(); ++i) {
            const Term& h = head_args[i];
            if (is_int(h)) {
                if (sig.modes[i] == ArgMode::input) {
                    guard = tri_and(guard, std::format("(a{} == {})", i, int_of(h).value));
                }
                continue;
            }
            if (!is_var(h)) {
                return make_error<std::string>(MathError::not_implemented);
            }
            const std::string v =
                mangle_var(VarKey{.name = var_of(h).name, .generation = var_of(h).generation});
            if (is_anonymous_var(var_of(h).name)) {
                continue;
            }
            if (head_vars.contains(v)) {
                return make_error<std::string>(MathError::not_implemented);
            }
            head_vars.insert(v);
            if (sig.modes[i] == ArgMode::input) {
                body += std::format("    {} = a{}\n", v, i);
                bound.insert(v);
            }
        }

        const TritonEmitter em{.body = &body, .bound = &bound, .temp_counter = &temp};
        for (const Term& goal : clause.body) {
            if (is_atom(goal)) {
                const std::string& g = atom_of(goal).name;
                if (g == "true" || g == "!") {
                    continue;
                }
                return make_error<std::string>(MathError::not_implemented);
            }
            if (!is_compound(goal)) {
                return make_error<std::string>(MathError::not_implemented);
            }
            const CompoundNode& g = compound_of(goal);
            if (g.functor == "is" && g.args.size() == 2 && is_var(g.args[0])) {
                auto value = emit_triton_expr(em, g.args[1]);
                if (!value) {
                    return make_error<std::string>(value.error());
                }
                const std::string v = mangle_var(
                    VarKey{.name = var_of(g.args[0]).name,
                           .generation = var_of(g.args[0]).generation});
                body += std::format("    {} = {}\n", v, value->expr);
                bound.insert(v);
                guard = tri_and(guard, value->ok);
                continue;
            }
            if (const auto cmp = comparison_op(g.functor); cmp && g.args.size() == 2) {
                auto lhs = emit_triton_expr(em, g.args[0]);
                if (!lhs) {
                    return make_error<std::string>(lhs.error());
                }
                auto rhs = emit_triton_expr(em, g.args[1]);
                if (!rhs) {
                    return make_error<std::string>(rhs.error());
                }
                guard = tri_and(guard, tri_and(tri_and(lhs->ok, rhs->ok),
                                               std::format("({} {} {})", lhs->expr, *cmp,
                                                           rhs->expr)));
                continue;
            }
            // A call would need a stack; Triton has none. This is refused earlier, but a
            // predicate reaching here with one is refused rather than mis-compiled.
            return make_error<std::string>(MathError::not_implemented);
        }

        if (guard.empty()) {
            guard = "tl.full(shape, 1, tl.int1)";
        }
        body += std::format("    g{} = {} & ~done\n", ci, guard);
        for (std::size_t i = 0; i < head_args.size(); ++i) {
            if (sig.modes[i] != ArgMode::output) {
                continue;
            }
            if (is_int(head_args[i])) {
                body += std::format("    a{} = tl.where(g{}, {}, a{})\n", i, ci,
                                    int_of(head_args[i]).value, i);
                continue;
            }
            if (!is_var(head_args[i])) {
                return make_error<std::string>(MathError::not_implemented);
            }
            const std::string v =
                mangle_var(VarKey{.name = var_of(head_args[i]).name,
                                  .generation = var_of(head_args[i]).generation});
            if (is_anonymous_var(var_of(head_args[i]).name)) {
                // As on the other targets: an anonymous variable in an OUTPUT position binds
                // nothing, so there is no value to write into the output lane. Leaving the lane
                // untouched would silently hand back whatever was there before.
                return make_error<std::string>(MathError::domain_error);
            }
            if (!bound.contains(v)) {
                return make_error<std::string>(MathError::domain_error);
            }
            body += std::format("    a{} = tl.where(g{}, {}, a{})\n", i, ci, v, i);
        }
        body += std::format("    done = done | g{}\n", ci);
        clauses_src += body;
    }

    std::string ptr_params;
    std::string loads;
    std::string stores;
    for (std::size_t i = 0; i < sig.arity(); ++i) {
        ptr_params += std::format("a{}_ptr, ", i);
        if (sig.modes[i] == ArgMode::input) {
            loads += std::format("    a{} = tl.load(a{}_ptr + offs, mask=m, other=0)\n", i, i);
        } else {
            loads += std::format("    a{} = tl.zeros(shape, dtype=tl.int64)\n", i);
            stores += std::format("    tl.store(a{}_ptr + offs, a{}, mask=m)\n", i, i);
        }
    }

    std::string out;
    out += "# Generated by nimblecas.logic_compile from a Prolog program.\n";
    out += "# @author Olumuyiwa Oluwasanmi\n";
    out += "#\n";
    out += "# DO NOT EDIT. Clauses are tried in program order and the first whose guard holds\n";
    out += "# wins, which `done` enforces: once a lane is done, later clauses are masked out of\n";
    out += "# it. `ok` reports, per lane, whether ANY clause succeeded — a lane that failed, or\n";
    out += "# whose arithmetic could not be represented in 64 bits, is reported rather than\n";
    out += "# left holding a wrapped value. Triton executes every lane through every\n";
    out += "# instruction, so overflow is carried as a MASK; there is no trap to raise.\n";
    out += '\n';
    out += "import triton\n";
    out += "import triton.language as tl\n";
    out += '\n';
    out += "INT64_MAX = 9223372036854775807\n";
    out += "INT64_MIN = -9223372036854775808\n";
    out += '\n';
    out += "@triton.jit\n";
    out += std::format("def {}_kernel({}ok_ptr, n, BLOCK: tl.constexpr):\n", fn, ptr_params);
    out += "    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n";
    out += "    m = offs < n\n";
    out += "    shape = offs.shape\n";
    out += loads;
    out += "    done = tl.zeros(shape, dtype=tl.int1)\n";
    out += clauses_src;
    out += stores;
    out += "    tl.store(ok_ptr + offs, done.to(tl.int8), mask=m)\n";
    return out;
}

}  // namespace

auto is_compilable(const Program& program, const PredicateSignature& entry, Target target)
    -> Result<void> {
    auto src = compile(program, entry, target);
    if (!src) {
        return make_error<void>(src.error());
    }
    return {};
}

auto compile(const Program& program, const PredicateSignature& entry, Target target)
    -> Result<std::string> {
    return compile(program, entry, CompileOptions{.target = target});
}

auto compile(const Program& program, const PredicateSignature& entry,
             const CompileOptions& options) -> Result<std::string> {
    const Target target = options.target;
    if (entry.name.empty()) {
        return make_error<std::string>(MathError::domain_error);
    }

    // Width/target combinations that cannot be honoured are refused HERE, before any source is
    // produced, so a caller never receives a file that computes in a width it did not ask for.
    if (target == Target::triton && options.width != Width::bits64) {
        // Triton's type lattice stops at tl.int64; there is no wider tensor dtype to lower to.
        return make_error<std::string>(MathError::not_implemented);
    }
    if (target == Target::cuda && options.width == Width::arbitrary) {
        // BigInt allocates a std::vector per operation; device code cannot allocate.
        return make_error<std::string>(MathError::not_implemented);
    }
    if (options.style == Style::continuation && target != Target::cpp) {
        // A CPS predicate is a template taking an arbitrary callable, and its continuations
        // nest to the depth of the conjunction. Neither a CUDA device function nor a Triton
        // kernel can express that, so the request is refused rather than silently narrowed to
        // the first solution — which would be a different program, not a slower one.
        return make_error<std::string>(MathError::not_implemented);
    }

    CompileCtx ctx{.program = &program,
                   .signatures = {},
                   .order = {},
                   .device = target == Target::cuda,
                   .allow_calls = target != Target::triton,
                   .tail_call_optimise = options.tail_call_optimise,
                   .width = options.width};
    auto collected = collect_predicates(program, entry, ctx.allow_calls, ctx);
    if (!collected) {
        return make_error<std::string>(collected.error());
    }

    if (target == Target::triton && ctx.order.size() != 1) {
        // A Triton kernel is a flat elementwise program over a block of lanes. It has no call
        // stack, so recursion and calls are not "slow" here, they are unrepresentable. One
        // predicate, straight-line, is what the target can honestly express.
        return make_error<std::string>(MathError::not_implemented);
    }

    // Emit callees before callers so the generated source needs no forward declarations —
    // except for a recursive predicate, which needs one for itself.
    std::string decls;
    std::string defs;
    const bool cps = options.style == Style::continuation;
    for (auto it = ctx.order.rbegin(); it != ctx.order.rend(); ++it) {
        const PredicateSignature& sig = ctx.signatures.at(*it);
        const PredicateDef def = gather(program, sig.name, sig.arity());
        auto body = cps ? emit_cps_predicate(ctx, def, sig) : emit_predicate(ctx, def, sig);
        if (!body) {
            return body;
        }
        if (cps) {
            // A CPS predicate is a TEMPLATE on its continuation, so its forward declaration is
            // one too — which is what lets a recursive predicate name itself.
            std::string params;
            for (std::size_t i = 0; i < sig.arity(); ++i) {
                if (sig.modes[i] == ArgMode::input) {
                    params += std::format("nc_int a{}, ", i);
                }
            }
            decls += std::format("template <class K>\n[[nodiscard]] auto {}({}K&& k) -> bool;\n",
                                 mangle(sig.name, sig.arity()), params);
            defs += *body;
            defs += '\n';
            continue;
        }
        std::string params;
        for (std::size_t i = 0; i < sig.arity(); ++i) {
            params += i == 0 ? "" : ", ";
            params += sig.modes[i] == ArgMode::input ? std::format("nc_int a{}", i)
                                                     : std::format("nc_int& a{}", i);
        }
        decls += std::format("[[nodiscard]] {}auto {}({}) -> bool;\n",
                             ctx.device ? "__device__ inline " : "inline ",
                             mangle(sig.name, sig.arity()), params);
        defs += *body;
        defs += '\n';
    }

    const PredicateSignature& esig = ctx.signatures.at(predicate_key(entry.name, entry.arity()));
    const std::string efn = mangle(entry.name, entry.arity());

    if (target == Target::triton) {
        return emit_triton(program, esig);
    }

    std::string out;
    out += std::format("// Generated by nimblecas.logic_compile from a Prolog program.\n");
    out += "// @author Olumuyiwa Oluwasanmi\n";
    out += "//\n";
    out += "// DO NOT EDIT. Every predicate below is a direct translation of the clauses it was\n";
    out += "// compiled from: clauses are tried in program order and the first that succeeds\n";
    out += "// wins, which is the determinism assumption the compiler checked before emitting.\n";
    out += "// A `false` return means the predicate FAILED or its arithmetic could not be\n";
    out += "// represented — the compiled code refuses an unrepresentable value exactly as the\n";
    out += "// interpreter does.\n";
    out += '\n';
    if (options.width == Width::arbitrary) {
        // BigInt is a NAMED-MODULE entity with no header form anywhere in the repo, so a file
        // that uses it cannot be a standalone translation unit — it has to be built by
        // clang++-23 against the repo's module stack. That is a real difference in how this
        // output is compiled, so the file states it rather than leaving a build failure to.
        out += "// BUILD NOTE: this file IMPORTS a module, so it is NOT standalone. Compile it\n";
        out += "// with clang++-23 alongside the NimbleCAS module build, which is where the BMI\n";
        out += "// for nimblecas.bigint comes from; a plain `clang++ file.cpp` cannot find it.\n";
        out += "// The narrower widths emit a self-contained file with no such requirement.\n\n";
        out += "import std;\nimport nimblecas.core;\nimport nimblecas.bigint;\n\n";
    } else if (target == Target::cuda) {
        // nvcc has no `import std`, so the CUDA output keeps headers. That is a property of
        // the compiler this file is for, not a style choice.
        out += "#include <cstdint>\n#include <limits>\n\n";
    } else {
        // Rules 11 and 41 of config/cpp_details.txt: import std, never the headers. The
        // arbitrary-precision width above has always done this; the narrower ones used to
        // emit <cstdint> and <limits> instead, which made the compiler contradict its own
        // policy depending on which width you asked for.
        out += "// BUILD NOTE: this file uses `import std;`, so it needs a C++23 compiler with\n";
        out += "// the standard library module available (clang++-23 with libc++, which is what\n";
        out += "// NimbleCAS itself is built with). It imports nothing from NimbleCAS, so no\n";
        out += "// other BMI is required.\n\n";
        out += "import std;\n\n";
        if (options.emit_batch && target == Target::cpp) {
            // The SIMD kernels need the CPUID and intrinsic headers, which have no module
            // form anywhere. They are emitted here rather than being assumed present, because
            // the generated file is compiled on its own by somebody who did not write it.
            out += "#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || "
                   "defined(_M_IX86)\n";
            out += "#include <immintrin.h>\n";
            out += "#if defined(_WIN32)\n#include <intrin.h>\n#else\n#include <cpuid.h>\n#endif\n";
            out += "#endif\n\n";
            out += "// The SIMD checked-arithmetic kernels. This header ships with NimbleCAS at\n";
            out += "// src/logic_compile/runtime/simd_batch.inc — put it beside this file, or\n";
            out += "// point the compiler at that directory with -I. It is self-contained: it\n";
            out += "// reads CPUID directly rather than through __builtin_cpu_supports, so it\n";
            out += "// imposes no link dependency on clang's builtins archive.\n";
            out += "#include \"simd_batch.inc\"\n\n";
        }
    }
    out += checked_preamble(ctx.device, options.width);
    out += '\n';
    out += decls;
    out += '\n';
    out += defs;

    if (options.emit_batch && target == Target::cpp) {
        // The CPU batch entry points: one predicate call per lane, lanes independent.
        std::string params;
        std::string call_args;
        for (std::size_t i = 0; i < esig.arity(); ++i) {
            params += std::format("{}nc_int* arg{}, ",
                                  esig.modes[i] == ArgMode::input ? "const " : "", i);
            call_args += call_args.empty() ? "" : ", ";
            call_args += std::format("arg{}[i]", i);
        }
        out += '\n';
        out += "// Applies the predicate to `n` independent inputs.\n";
        out += "//\n";
        out += "// `ok[i]` records whether lane i SUCCEEDED — the predicate proved its goal and\n";
        out += "// its arithmetic was representable. A lane whose ok is 0 leaves its output\n";
        out += "// arguments unspecified and the caller must not read them; writing a plausible\n";
        out += "// value there is exactly the failure the checked arithmetic exists to prevent.\n";
        out += std::format("inline auto {}_batch({}unsigned char* ok, std::size_t n) -> void {{\n",
                           efn, params);
        out += "    for (std::size_t i = 0; i < n; ++i) {\n";
        out += std::format("        ok[i] = {}({}) ? 1 : 0;\n", efn, call_args);
        out += "    }\n}\n\n";

        out += "// The same map, across threads.\n";
        out += "//\n";
        out += "// Lanes share nothing, so this is a pure map: the result is IDENTICAL to\n";
        out += std::format("// `{}_batch` on any thread count, which is what makes the parallel\n",
                           efn);
        out += "// form testable against the serial one rather than merely plausible. The range\n";
        out += "// is split into contiguous chunks, one per thread, so each lane is written by\n";
        out += "// exactly one thread and no synchronisation is needed on the outputs.\n";
        out += "//\n";
        out += "// Below a few thousand lanes the threads cost more than they save, so the work\n";
        out += "// stays on the calling thread; `threads == 0` asks for hardware_concurrency.\n";
        out += std::format(
            "inline auto {}_batch_parallel({}unsigned char* ok, std::size_t n,\n", efn, params);
        out += "                              unsigned threads = 0) -> void {\n";
        out += "    constexpr std::size_t serial_below = 4096;\n";
        out += "    unsigned want = threads;\n";
        out += "    if (want == 0) {\n";
        out += "        want = std::thread::hardware_concurrency();\n";
        out += "    }\n";
        out += "    if (want <= 1 || n < serial_below) {\n";
        out += std::format("        {}_batch({}ok, n);\n", efn, [&] {
            std::string a;
            for (std::size_t i = 0; i < esig.arity(); ++i) {
                a += std::format("arg{}, ", i);
            }
            return a;
        }());
        out += "        return;\n    }\n";
        out += "    const std::size_t chunk = (n + want - 1) / want;\n";
        out += "    std::vector<std::jthread> pool;\n";
        out += "    pool.reserve(want);\n";
        out += "    for (std::size_t begin = 0; begin < n; begin += chunk) {\n";
        out += "        const std::size_t end = begin + chunk < n ? begin + chunk : n;\n";
        out += "        pool.emplace_back([=] {\n";
        out += "            for (std::size_t i = begin; i < end; ++i) {\n";
        out += std::format("                ok[i] = {}({}) ? 1 : 0;\n", efn, call_args);
        out += "            }\n        });\n    }\n";
        out += "    // jthread joins on destruction, so the pool going out of scope is the\n";
        out += "    // barrier; there is no path that returns with a thread still writing.\n";
        out += "}\n";
    }

    if (target == Target::cuda) {
        // The batch kernel: one thread per input row. This is where the GPU target earns its
        // keep — the predicate itself is scalar, and the parallelism is over calls to it.
        std::string kparams;
        std::string kargs;
        for (std::size_t i = 0; i < esig.arity(); ++i) {
            kparams += std::format(", {}nc_int* arg{}",
                                   esig.modes[i] == ArgMode::input ? "const " : "", i);
        }
        for (std::size_t i = 0; i < esig.arity(); ++i) {
            kargs += i == 0 ? "" : ", ";
            kargs += std::format("arg{}[i]", i);
        }
        out += '\n';
        out += "// Batch entry point: one thread per row. `ok[i]` records whether row i\n";
        out += "// succeeded, so a failed or unrepresentable row is reported rather than left\n";
        out += "// as a silent zero in the output.\n";
        out += "//\n";
        out += "// DEVICE STACK: a recursive predicate needs more than the 1 KB per thread that\n";
        out += "// CUDA gives by default, and running past it surfaces as `an illegal memory\n";
        out += "// access was encountered` from cudaDeviceSynchronize -- which reads like a bug\n";
        out += "// in this file and is not one. Raise the limit before launching:\n";
        out += "//     cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024);\n";
        out += std::format("__global__ void {}_batch(int n, unsigned char* ok{})\n", efn, kparams);
        out += "{\n";
        out += "    const int i = blockIdx.x * blockDim.x + threadIdx.x;\n";
        out += "    if (i >= n) { return; }\n";
        out += std::format("    ok[i] = {}({}) ? 1 : 0;\n", efn, kargs);
        out += "}\n";
    }
    return out;
}

}  // namespace nimblecas::logic_compile
