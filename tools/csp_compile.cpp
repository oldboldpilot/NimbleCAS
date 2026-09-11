// csp_compile — emit a solver for an n-queens CSP as C++, CUDA or Triton source.
// @author Olumuyiwa Oluwasanmi
//
// Usage:
//   csp_compile <n> [cpp|cuda|triton] [prefix] [OPTIONS...]
//
// OPTIONS: exhaustive (default) | minconflicts, pairwise,
//          shards=N, walkers=N, steps=N, noise=PER1024, seed=N
//
// The two sibling emitters each have a command-line front end; this one completes the set, and
// it is what lets `scripts/verify-generated.sh` put this module's output through a compiler.
//
// N-QUEENS RATHER THAN A FILE FORMAT, deliberately. A CSP is a graph of typed constraints, not
// a line-oriented format like DIMACS, and inventing one here would mean inventing a parser and
// a specification for it. The queens family instead exercises every part of the emitter that
// matters -- a variadic `all_different`, a parametrised binary `abs_diff_ne`, and a search
// space that grows fast enough to reach the emitter's own limits -- from a single integer.
//
// `pairwise` swaps the single all_different for pairwise not-equals. The solution sets are
// identical, but the constraint granularity is what local search actually walks over, so the
// two encodings emit meaningfully different programs.
//
// The emitted source goes to standard output and everything else to standard error, so the
// output can be redirected straight into a file.

import std;
import nimblecas.core;
import nimblecas.csp;
import nimblecas.csp_compile;

using nimblecas::as_csp;
using nimblecas::backtracking_search;
using nimblecas::ConstraintKind;
using nimblecas::WireConstraint;
using nimblecas::WireCsp;
using nimblecas::csp_compile::emit;
using nimblecas::csp_compile::EmitOptions;
using nimblecas::csp_compile::entry_point_name;
using nimblecas::csp_compile::is_compilable_for;
using nimblecas::csp_compile::search_space;
using nimblecas::csp_compile::Strategy;
using nimblecas::csp_compile::Target;

namespace {

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

[[nodiscard]] auto strategy_of(std::string_view s) -> std::optional<Strategy> {
    if (s == "exhaustive") {
        return Strategy::exhaustive;
    }
    if (s == "minconflicts") {
        return Strategy::min_conflicts;
    }
    return std::nullopt;
}

// The n-queens problem: one variable per row, its value the column, all columns distinct and no
// two queens on a diagonal.
[[nodiscard]] auto queens(std::int64_t n, bool pairwise) -> WireCsp {
    const auto un = static_cast<std::size_t>(n);
    WireCsp w;
    std::vector<std::int64_t> domain;
    domain.reserve(un);
    for (std::int64_t v = 0; v < n; ++v) {
        domain.push_back(v);
    }
    w.domains.assign(un, domain);

    if (!pairwise) {
        std::vector<std::size_t> all;
        all.reserve(un);
        for (std::size_t i = 0; i < un; ++i) {
            all.push_back(i);
        }
        w.constraints.push_back(WireConstraint{
            .kind = ConstraintKind::all_different, .scope = std::move(all), .params = {}});
    }
    for (std::size_t i = 0; i < un; ++i) {
        for (std::size_t j = i + 1; j < un; ++j) {
            if (pairwise) {
                w.constraints.push_back(WireConstraint{
                    .kind = ConstraintKind::not_equal, .scope = {i, j}, .params = {}});
            }
            w.constraints.push_back(
                WireConstraint{.kind = ConstraintKind::abs_diff_ne,
                               .scope = {i, j},
                               .params = {static_cast<std::int64_t>(j - i)}});
        }
    }
    return w;
}

[[nodiscard]] auto number_after(std::string_view arg, std::size_t skip) -> std::uint64_t {
    return std::strtoull(std::string(arg.substr(skip)).c_str(), nullptr, 10);
}

}  // namespace

auto main(int argc, char** argv) -> int {
    const std::vector<std::string_view> args(argv, argv + argc);
    if (args.size() < 2) {
        std::println(std::cerr,
                     "usage: csp_compile <n> [cpp|cuda|triton] [prefix] "
                     "[exhaustive|minconflicts] [pairwise] [shards=N] [walkers=N] [steps=N] "
                     "[noise=PER1024] [seed=N]");
        return 2;
    }

    const auto n = static_cast<std::int64_t>(number_after(args[1], 0));
    if (n < 1) {
        std::println(std::cerr, "csp_compile: n must be at least 1");
        return 2;
    }

    EmitOptions opts;
    if (args.size() > 2) {
        const auto t = target_of(args[2]);
        if (!t.has_value()) {
            std::println(std::cerr, "csp_compile: unknown target {}", args[2]);
            return 2;
        }
        opts.target = *t;
    }
    if (args.size() > 3) {
        opts.name_prefix = std::string(args[3]);
    }
    bool pairwise = false;
    for (const std::string_view a : args | std::views::drop(4)) {
        if (a == "pairwise") {
            pairwise = true;
        } else if (const auto st = strategy_of(a); st.has_value()) {
            opts.strategy = *st;
        } else if (a.starts_with("shards=")) {
            opts.shards = static_cast<std::size_t>(number_after(a, 7));
        } else if (a.starts_with("walkers=")) {
            opts.walkers = static_cast<std::size_t>(number_after(a, 8));
        } else if (a.starts_with("steps=")) {
            opts.max_steps = number_after(a, 6);
        } else if (a.starts_with("noise=")) {
            opts.noise_per_1024 = static_cast<std::uint32_t>(number_after(a, 6));
        } else if (a.starts_with("seed=")) {
            opts.seed = number_after(a, 5);
        } else {
            std::println(std::cerr, "csp_compile: unknown option {}", a);
            return 2;
        }
    }

    const WireCsp w = queens(n, pairwise);
    // `is_compilable_for` answers about a SEARCH, and refuses Triton outright -- correctly, since
    // the Triton target emits a conflict scorer and a strategy is meaningless to it. Asking it
    // about Triton would refuse the one target that has nothing to refuse.
    if (opts.target != Target::triton) {
        if (const auto ok = is_compilable_for(w, opts); !ok.has_value()) {
            // Refusing is the honest answer for a space beyond the emitter's cap; emitting a
            // loop that would not finish would not be.
            std::println(std::cerr, "csp_compile: refused ({})", static_cast<int>(ok.error()));
            return 1;
        }
    }

    const auto src = emit(w, opts);
    if (!src.has_value()) {
        std::println(std::cerr, "csp_compile: emission refused ({})",
                     static_cast<int>(src.error()));
        return 1;
    }
    std::print("{}", *src);

    std::println(std::cerr, "csp_compile: {}-queens, {} variables, {} constraints, entry {}", n,
                 w.domains.size(), w.constraints.size(), entry_point_name(opts));
    if (const auto space = search_space(w); space.has_value()) {
        std::println(std::cerr, "csp_compile: assignment space {}", *space);
    }

    // The reference answer, so a caller can check the emitted solver without writing an oracle.
    // Backtracking search finds the lexicographically first solution, which is exactly what the
    // emitted exhaustive scan reports; the local-search target may legitimately return another.
    const auto csp = as_csp(w);
    if (!csp.has_value()) {
        return 0;
    }
    const auto found = backtracking_search(*csp);
    if (!found.has_value()) {
        std::println(std::cerr, "csp_compile: reference search declined");
        return 0;
    }
    if (found->has_value()) {
        std::string rendered;
        for (const std::int64_t v : **found) {
            rendered += std::format("{} ", v);
        }
        std::println(std::cerr, "csp_compile: reference first solution = {}", rendered);
    } else {
        std::println(std::cerr, "csp_compile: reference says no solution exists");
    }
    return 0;
}
