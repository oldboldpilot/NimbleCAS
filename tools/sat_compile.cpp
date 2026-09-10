// sat_compile — emit a solver for a DIMACS CNF as C++, CUDA or Triton source.
// @author Olumuyiwa Oluwasanmi
//
// Usage:
//   sat_compile <file.cnf> [cpp|cuda|triton] [entry] [OPTIONS...]
//
// OPTIONS: exhaustive (default) | walksat, nosimd, noparallel,
//          walkers=N, flips=N, noise=PERCENT
//
// Reads DIMACS on the given path (`p cnf <vars> <clauses>`, then clauses of literals terminated
// by 0, with `c` comment lines ignored) and writes the emitted source to standard output. A
// summary — variable count, clause count and the reference solver's answer — goes to standard
// error, so the output can be redirected straight into a file without contaminating it.

import std;
import nimblecas.core;
import nimblecas.sat;
import nimblecas.sat_compile;

using nimblecas::Cnf;
using nimblecas::sat_compile::compile;
using nimblecas::sat_compile::CompileOptions;
using nimblecas::sat_compile::model_of;
using nimblecas::sat_compile::reference_solve;
using nimblecas::sat_compile::reference_walksat;
using nimblecas::sat_compile::SlsVariant;
using nimblecas::sat_compile::Strategy;
using nimblecas::sat_compile::Target;

namespace {

// Reads DIMACS. Returns nullopt on anything malformed rather than a formula that silently differs
// from the file — a solver for the wrong formula still answers, which is the worst failure here.
[[nodiscard]] auto read_dimacs(std::istream& in) -> std::optional<Cnf> {
    Cnf cnf{.num_vars = 0, .clauses = {}};
    std::vector<std::int64_t> current;
    bool seen_header = false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line.front() == 'c' || line.front() == '%') {
            continue;
        }
        std::istringstream ls(line);
        if (line.front() == 'p') {
            std::string p;
            std::string cnf_word;
            std::size_t vars = 0;
            std::size_t clauses = 0;
            if (!(ls >> p >> cnf_word >> vars >> clauses) || cnf_word != "cnf" || vars == 0) {
                return std::nullopt;
            }
            cnf.num_vars = vars;
            cnf.clauses.reserve(clauses);
            seen_header = true;
            continue;
        }
        if (!seen_header) {
            return std::nullopt;
        }
        std::int64_t lit = 0;
        while (ls >> lit) {
            if (lit == 0) {
                cnf.clauses.push_back(current);
                current.clear();
                continue;
            }
            current.push_back(lit);
        }
        if (!ls.eof()) {
            return std::nullopt;  // a token that was not an integer
        }
    }
    if (!seen_header) {
        return std::nullopt;
    }
    if (!current.empty()) {
        cnf.clauses.push_back(current);  // a final clause with no terminating 0
    }
    return cnf;
}

[[nodiscard]] auto variant_of(std::string_view s) -> std::optional<SlsVariant> {
    if (s == "skc") {
        return SlsVariant::skc;
    }
    if (s == "probsat") {
        return SlsVariant::probsat;
    }
    if (s == "novelty") {
        return SlsVariant::novelty_plus;
    }
    if (s == "adaptnovelty") {
        return SlsVariant::adaptive_novelty_plus;
    }
    return std::nullopt;
}

[[nodiscard]] auto strategy_of(std::string_view s) -> std::optional<Strategy> {
    if (s == "exhaustive") {
        return Strategy::exhaustive;
    }
    if (s == "walksat") {
        return Strategy::walksat;
    }
    return std::nullopt;
}

[[nodiscard]] auto target_of(std::string_view s) -> std::optional<Target> {
    if (s == "cpp") {
        return Target::cpp;
    }
    if (s == "cuda") {
        return Target::cuda;
    }
    if (s == "triton") {
        return Target::triton;
    }
    return std::nullopt;
}

}  // namespace

auto main(int argc, char** argv) -> int {
    const std::vector<std::string_view> args(argv, argv + argc);
    if (args.size() < 2) {
        std::println(std::cerr,
                     "usage: sat_compile <file.cnf> [cpp|cuda|triton] [entry] [nosimd] "
                     "[noparallel]");
        return 2;
    }

    std::ifstream file{std::string(args[1])};
    if (!file) {
        std::println(std::cerr, "sat_compile: cannot open {}", args[1]);
        return 2;
    }
    const auto cnf = read_dimacs(file);
    if (!cnf.has_value()) {
        std::println(std::cerr, "sat_compile: {} is not well-formed DIMACS", args[1]);
        return 2;
    }

    CompileOptions opts;
    if (args.size() > 2) {
        const auto t = target_of(args[2]);
        if (!t.has_value()) {
            std::println(std::cerr, "sat_compile: unknown target {}", args[2]);
            return 2;
        }
        opts.target = *t;
    }
    if (args.size() > 3) {
        opts.entry = std::string(args[3]);
    }
    for (const std::string_view a : args | std::views::drop(4)) {
        if (a == "nosimd") {
            opts.emit_simd = false;
        } else if (a == "noparallel") {
            opts.emit_parallel = false;
        } else if (const auto st = strategy_of(a); st.has_value()) {
            opts.strategy = *st;
        } else if (const auto vr = variant_of(a); vr.has_value()) {
            opts.variant = *vr;
        } else if (a.starts_with("walkers=")) {
            opts.walkers = static_cast<std::uint32_t>(std::strtoul(std::string(a.substr(8)).c_str(), nullptr, 10));
        } else if (a.starts_with("flips=")) {
            opts.max_flips = std::strtoull(std::string(a.substr(6)).c_str(), nullptr, 10);
        } else if (a.starts_with("noise=")) {
            opts.noise_percent = static_cast<std::uint32_t>(std::strtoul(std::string(a.substr(6)).c_str(), nullptr, 10));
        }
    }

    auto src = compile(*cnf, opts);
    if (!src) {
        std::println(std::cerr, "sat_compile: refused ({})", static_cast<int>(src.error()));
        return 1;
    }
    std::print("{}", *src);

    std::println(std::cerr, "sat_compile: {} variables, {} clauses", cnf->num_vars,
                 cnf->clauses.size());
    if (opts.strategy == Strategy::walksat) {
        // The reference walk, so a caller can check the emitted walker against it. Same seeding,
        // same move rule, same walker ordering -- so the answer is comparable model for model
        // rather than merely both being satisfying.
        auto walk = reference_walksat(*cnf, opts.walkers, opts.max_flips, opts.noise_percent,
                                      opts.base_seed);
        if (!walk) {
            std::println(std::cerr, "sat_compile: reference walk refused");
            return 1;
        }
        if (walk->found) {
            std::println(std::cerr,
                         "sat_compile: reference walk solved it -- walker {} of {} run",
                         walk->walker, walk->walkers_run);
        } else {
            std::println(std::cerr,
                         "sat_compile: reference walk found nothing in {} walkers x {} flips "
                         "(UNKNOWN, not unsatisfiable)",
                         opts.walkers, opts.max_flips);
        }
        return 0;
    }

    // The reference answer, so a caller can check the emitted solver against it without writing
    // their own oracle. Bounded, because a large formula's enumeration is not worth waiting for
    // just to print a line.
    auto answer = reference_solve(*cnf, 1ULL << 22);
    if (!answer) {
        std::println(std::cerr, "sat_compile: reference solver declined (too large to enumerate)");
        return 0;
    }
    if (answer->has_value()) {
        const std::vector<bool> model = model_of(**answer, cnf->num_vars);
        std::string rendered;
        for (const std::size_t i : std::views::iota(std::size_t{0}, model.size())) {
            rendered += std::format("{}{}", model[i] ? "" : "-", i + 1);
            rendered += ' ';
        }
        std::println(std::cerr, "sat_compile: reference smallest assignment = {} ({})", **answer,
                     rendered);
    } else {
        std::println(std::cerr, "sat_compile: reference says unsatisfiable");
    }
    return 0;
}
