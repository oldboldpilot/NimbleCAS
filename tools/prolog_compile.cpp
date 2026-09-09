// NimbleCAS Prolog compiler driver — reads a Prolog file, writes C++, CUDA or Triton source.
// @author Olumuyiwa Oluwasanmi
//
// Usage:
//   prolog_compile <file.pl> <entry-name> <modes> [cpp|cuda|triton]
//
// `modes` is one character per argument: `i` for an input, `o` for an output. So compiling
// `fib/2` called as fib(+N, -F) is `prolog_compile fib.pl fib io cpp`.
//
// The modes are given rather than inferred because a wrong mode changes what the program MEANS,
// not merely how fast it runs, and a compiler is not entitled to guess that.
//
// The generated source goes to stdout; diagnostics go to stderr, so the output can be piped
// straight into a compiler. A program outside the compilable subset is REFUSED with a message
// naming why — never partially compiled.

import std;
import nimblecas.core;
import nimblecas.logic;
import nimblecas.logic_compile;
import nimblecas.logic_parser;

namespace lc = nimblecas::logic_compile;

namespace {

[[nodiscard]] auto read_file(const std::string& path) -> std::optional<std::string> {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::nullopt;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

[[nodiscard]] auto parse_modes(std::string_view spec) -> std::optional<std::vector<lc::ArgMode>> {
    std::vector<lc::ArgMode> modes;
    for (const char c : spec) {
        if (c == 'i') {
            modes.push_back(lc::ArgMode::input);
        } else if (c == 'o') {
            modes.push_back(lc::ArgMode::output);
        } else {
            return std::nullopt;
        }
    }
    return modes;
}

[[nodiscard]] auto parse_style(std::string_view name) -> std::optional<lc::Style> {
    if (name == "det" || name == "deterministic") {
        return lc::Style::deterministic;
    }
    if (name == "cps" || name == "continuation") {
        return lc::Style::continuation;
    }
    return std::nullopt;
}

[[nodiscard]] auto parse_width(std::string_view name) -> std::optional<lc::Width> {
    if (name == "64") {
        return lc::Width::bits64;
    }
    if (name == "128") {
        return lc::Width::bits128;
    }
    if (name == "big" || name == "arbitrary") {
        return lc::Width::arbitrary;
    }
    return std::nullopt;
}

[[nodiscard]] auto parse_target(std::string_view name) -> std::optional<lc::Target> {
    if (name == "cpp") {
        return lc::Target::cpp;
    }
    if (name == "cuda") {
        return lc::Target::cuda;
    }
    if (name == "triton") {
        return lc::Target::triton;
    }
    return std::nullopt;
}

[[nodiscard]] auto why(nimblecas::MathError e) -> std::string_view {
    switch (e) {
        case nimblecas::MathError::not_implemented:
            return "either the program uses something outside the compilable subset (integers, "
                   "is/2, the arithmetic comparisons, and calls to other compilable "
                   "predicates), or the requested WIDTH is not available on this target — "
                   "Triton is 64-bit only, and CUDA cannot do arbitrary precision because "
                   "device code cannot allocate";
        case nimblecas::MathError::domain_error:
            return "the signature does not describe the program: an undefined predicate, an "
                   "arity mismatch, or an output argument no clause ever binds";
        case nimblecas::MathError::syntax_error:
            return "the Prolog source did not parse";
        default:
            return nimblecas::to_string_view(e);
    }
}

}  // namespace

auto main(int argc, char** argv) -> int {
    const std::vector<std::string> args(argv, argv + argc);
    if (args.size() < 4) {
        std::println(std::cerr,
                     "usage: prolog_compile <file.pl> <entry-name> <modes:i|o...> "
                     "[cpp|cuda|triton] [64|128|big] [det|cps] [batch]");
        return 2;
    }
    const std::string path = args[1];
    const std::string entry_name = args[2];
    const auto modes = parse_modes(args[3]);
    if (!modes) {
        std::println(std::cerr, "error: modes must be a string of 'i' and 'o', got '{}'", args[3]);
        return 2;
    }
    const auto target = parse_target(args.size() > 4 ? args[4] : "cpp");
    if (!target) {
        std::println(std::cerr, "error: target must be cpp, cuda or triton");
        return 2;
    }

    const auto text = read_file(path);
    if (!text) {
        std::println(std::cerr, "error: cannot read {}", path);
        return 2;
    }
    const auto program = nimblecas::logic_parser::parse_program(*text);
    if (!program) {
        std::println(std::cerr, "error: {}", why(program.error()));
        return 1;
    }

    const auto width = parse_width(args.size() > 5 ? args[5] : "64");
    if (!width) {
        std::println(std::cerr, "error: width must be 64, 128 or big");
        return 2;
    }

    const lc::PredicateSignature sig{.name = entry_name, .modes = *modes};
    const bool want_batch = args.size() > 7 && (args[7] == "batch" || args[7] == "simd");
    const auto style = parse_style(args.size() > 6 ? args[6] : "det");
    if (!style) {
        std::println(std::cerr, "error: style must be det or cps");
        return 2;
    }

    const lc::CompileOptions opts{.target = *target,
                                  .width = *width,
                                  .style = *style,
                                  .tail_call_optimise = true,
                                  .emit_batch = want_batch};
    const auto src = lc::compile(*program, sig, opts);
    if (!src) {
        std::println(std::cerr, "error: cannot compile {}/{} for target {} — {}", entry_name,
                     modes->size(), lc::to_string_view(*target), why(src.error()));
        return 1;
    }
    std::print("{}", *src);
    return 0;
}
