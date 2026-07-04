// NimbleCAS bitset-domain finite CSP & branchless bitmask N-queens (branchless / bit-parallel).
// @author Olumuyiwa Oluwasanmi
//
// A finite-domain constraint problem in which each variable's live domain is a Bitset over
// its value range [0, k). Everything that matters — constraint propagation and the N-queens
// search — is done with WORD-PARALLEL BITWISE OPERATIONS and no value-by-value branching,
// so the CPU reference here has the exact control-flow shape a GPU SIMT kernel would run.
//
// SUPPORT MASKS (the branchless propagator). A binary constraint between variables x and y
// is precompiled, once, from a bool(int, int) predicate into two tables of Bitsets:
//   support_x_given_y[v] = the set of x-values compatible with y = v, and
//   support_y_given_x[u] = the set of y-values compatible with x = u.
// Revising D[x] against D[y] is then pure bitwise: OR together support_x_given_y[v] for every
// v currently in D[y] (a word-parallel union), then AND that into D[x] (a word-parallel
// intersection). No branch inspects an individual (x, y) value pair at propagation time — the
// per-pair decisions were baked into the masks up front. On a GPU each such union/intersection
// is a warp-coherent bitwise reduction; a value-branching AC-3 would instead diverge per pair.
//
// N-QUEENS (the branchless-parallel showcase). Rows are placed one at a time with three
// bitmasks — occupied columns, and the two diagonal frontiers — so the free squares of a row
// are `~(cols | d1 | d2) & board_mask`; the lowest free square is taken with `free & (-free)`
// (no conditional), and the recursion carries shifted diagonal masks. Fanning the FIRST row's
// placements across nimblecas::parallel::transform_index makes each starting column an
// independent, stateless subtree count — precisely the "one thread per subtree" launch a GPU
// performs — and the parallel total is identical to the serial one for any worker count.
//
// ERRORS (railway, Rule 32): shape faults (no variables, capacity-0 domain, out-of-range
// constraint scope, support tables whose sizes/capacities don't match the domains, an
// out-of-range N) are domain_error. INCONSISTENCY IS NOT AN ERROR: an emptied domain during
// AC-3 is a valid std::nullopt result, mirroring nimblecas.csp.

export module nimblecas.bitcsp;

import std;
import nimblecas.core;
import nimblecas.bitset;
import nimblecas.parallel;

export namespace nimblecas {

// A binary constraint between variables x and y, precompiled into support masks. For a value
// v in y's range, support_x_given_y[v] is the Bitset of x-values compatible with y = v (so its
// capacity is x's range size, and the table has y's range size entries); support_y_given_x is
// the symmetric table for revising y against x. Build with make_bit_constraint.
struct BitConstraint {
    std::size_t x;
    std::size_t y;
    std::vector<Bitset> support_x_given_y;  // indexed by a y-value; each Bitset over x's range
    std::vector<Bitset> support_y_given_x;  // indexed by an x-value; each Bitset over y's range
};

// A bitset-domain CSP: domains[i] is variable i's live Bitset over [0, capacity_i), and
// `constraints` are binary constraints referencing those variables.
struct BitCsp {
    std::vector<Bitset> domains;
    std::vector<BitConstraint> constraints;
};

// Compiles a binary constraint on (x, y) from a predicate allowed(x_value, y_value) into the
// two support-mask tables, given x's range size kx and y's range size ky. This is the ONLY
// place the predicate is evaluated — O(kx * ky) once — after which propagation is pure bitwise.
[[nodiscard]] auto make_bit_constraint(std::size_t x, std::size_t y, std::size_t kx,
                                       std::size_t ky,
                                       const std::function<bool(int, int)>& allowed)
    -> BitConstraint;

// AC-3 arc consistency over the support masks. Returns the pruned per-variable Bitset domains,
// or an engaged Result holding std::nullopt if some domain is emptied (arc-inconsistent — hence
// unsatisfiable, but a VALID result, not an error). Shape faults are domain_error. The worklist
// is a FIFO seeded and re-queued in a fixed arc-index order, so the fixpoint is independent of
// scheduling — deterministic. Each revise step is the branchless OR-of-supports-then-AND above.
[[nodiscard]] auto ac3_bitset(const BitCsp& csp)
    -> Result<std::optional<std::vector<Bitset>>>;

// Counts solutions to the N-queens problem with the classic BRANCHLESS bitmask search. With
// `parallel` true the first row's N placements are fanned across
// nimblecas::parallel::transform_index as independent stateless subtree counts and summed; the
// result is IDENTICAL to the serial count for any worker count. Returns domain_error unless
// 1 <= n <= 32 (the board mask must fit a 64-bit word with head-room for the diagonal shifts).
[[nodiscard]] auto count_nqueens(int n, bool parallel) -> Result<std::uint64_t>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

auto make_bit_constraint(std::size_t x, std::size_t y, std::size_t kx, std::size_t ky,
                         const std::function<bool(int, int)>& allowed) -> BitConstraint {
    BitConstraint c;
    c.x = x;
    c.y = y;
    // support_x_given_y[v] gathers the x-values u with allowed(u, v); support_y_given_x[u]
    // gathers the y-values v with allowed(u, v). Every bit is decided here, once.
    c.support_x_given_y.assign(ky, Bitset(kx));
    c.support_y_given_x.assign(kx, Bitset(ky));
    for (std::size_t u = 0; u < kx; ++u) {
        for (std::size_t v = 0; v < ky; ++v) {
            if (allowed(static_cast<int>(u), static_cast<int>(v))) {
                c.support_x_given_y[v].set(u);
                c.support_y_given_x[u].set(v);
            }
        }
    }
    return c;
}

// Structural validation shared by the propagator. Returns the offending MathError, or nullopt
// when the CSP is well formed. A well-formed CSP may still be unsatisfiable — that is decided
// by AC-3, not here.
[[nodiscard]] auto bitcsp_validate(const BitCsp& csp) -> std::optional<MathError> {
    const std::size_t n = csp.domains.size();
    if (n == 0) {
        return MathError::domain_error;  // empty variable set
    }
    for (const Bitset& d : csp.domains) {
        if (d.capacity() == 0) {
            return MathError::domain_error;  // a variable with no possible values is malformed
        }
    }
    for (const BitConstraint& c : csp.constraints) {
        if (c.x >= n || c.y >= n) {
            return MathError::domain_error;  // constraint references an out-of-range variable
        }
        const std::size_t kx = csp.domains[c.x].capacity();
        const std::size_t ky = csp.domains[c.y].capacity();
        // The support tables must be shaped to the domains they revise, or the branchless
        // OR/AND below would index out of range or mix mismatched capacities.
        if (c.support_x_given_y.size() != ky || c.support_y_given_x.size() != kx) {
            return MathError::domain_error;
        }
        for (const Bitset& s : c.support_x_given_y) {
            if (s.capacity() != kx) {
                return MathError::domain_error;
            }
        }
        for (const Bitset& s : c.support_y_given_x) {
            if (s.capacity() != ky) {
                return MathError::domain_error;
            }
        }
    }
    return std::nullopt;
}

// Branchless revise of D[from] using D[to] and the support table `support` (support[v] = the
// from-values compatible with to = v). Builds new = OR over v in set_bits(D[to]) of support[v]
// (word-parallel union) and intersects it into D[from] (word-parallel AND). Returns true iff
// D[from] lost at least one value.
[[nodiscard]] auto revise_bitset(Bitset& from, const Bitset& to,
                                 const std::vector<Bitset>& support) -> bool {
    Bitset allowed(from.capacity());  // all-clear; will accumulate the union of supports
    for (const std::size_t v : to.set_bits()) {
        allowed.or_inplace(support[v]);
    }
    Bitset next = from;
    next.and_inplace(allowed);
    if (next == from) {
        return false;
    }
    from = std::move(next);
    return true;
}

auto ac3_bitset(const BitCsp& csp) -> Result<std::optional<std::vector<Bitset>>> {
    using Ret = std::optional<std::vector<Bitset>>;
    if (const auto err = bitcsp_validate(csp)) {
        return make_error<Ret>(*err);
    }

    std::vector<Bitset> domains = csp.domains;

    // Each constraint contributes two directed arcs: revise x given y, and revise y given x.
    // `rev` selects which endpoint is being revised (and thus which support table to read).
    struct Arc {
        std::size_t ci;
        bool rev;
        std::size_t from;
        std::size_t to;
    };
    std::vector<Arc> arcs;
    arcs.reserve(csp.constraints.size() * 2);
    for (std::size_t ci = 0; ci < csp.constraints.size(); ++ci) {
        const BitConstraint& c = csp.constraints[ci];
        arcs.push_back(Arc{ci, false, c.x, c.y});  // revise D[x] using D[y]
        arcs.push_back(Arc{ci, true, c.y, c.x});   // revise D[y] using D[x]
    }

    // FIFO worklist seeded in arc-index order; `queued` prevents duplicate enqueues. Both the
    // seed order and the re-queue scan run in ascending arc index, so the reached fixpoint is
    // independent of any scheduling — the determinism guarantee.
    std::deque<std::size_t> work;
    std::vector<char> queued(arcs.size(), 0);
    for (std::size_t i = 0; i < arcs.size(); ++i) {
        work.push_back(i);
        queued[i] = 1;
    }

    while (!work.empty()) {
        const std::size_t idx = work.front();
        work.pop_front();
        queued[idx] = 0;
        const Arc arc = arcs[idx];
        const BitConstraint& c = csp.constraints[arc.ci];
        const std::vector<Bitset>& support =
            arc.rev ? c.support_y_given_x : c.support_x_given_y;

        const bool changed = revise_bitset(domains[arc.from], domains[arc.to], support);
        if (changed) {
            if (domains[arc.from].none()) {
                return Ret{std::nullopt};  // domain emptied: arc-inconsistent (valid, not error)
            }
            // Re-queue every arc that reads INTO `from` (its from-domain may have lost support),
            // except the reverse arc of THIS SAME constraint — a value dropped from D[from]
            // under constraint `arc.ci` cannot have been the support for any value under that
            // same constraint's reverse arc. A DIFFERENT constraint on the same ordered pair
            // must NOT be skipped, so we exclude only the same-constraint reverse (mirrors the
            // reasoning in nimblecas.csp).
            for (std::size_t k = 0; k < arcs.size(); ++k) {
                const bool same_constraint_reverse =
                    arcs[k].ci == arc.ci && arcs[k].from == arc.to;
                if (arcs[k].to == arc.from && !same_constraint_reverse && queued[k] == 0) {
                    work.push_back(k);
                    queued[k] = 1;
                }
            }
        }
    }

    return Ret{std::move(domains)};
}

// Counts completions of a partial N-queens board described by three bitmasks over the columns:
// `cols` = occupied columns, `d1` = "/" diagonal frontier, `d2` = "\" diagonal frontier; `all`
// is the n-bit board mask. BRANCHLESS INNER LOOP: the free squares are computed with a single
// bitwise expression and consumed one lowest-bit-at-a-time via `free & (~free + 1)` (== free &
// -free) with `free ^= bit`; the only branch is the loop test. When every column is filled
// (cols == all) the placement is one complete solution.
[[nodiscard]] auto nqueens_count(std::uint64_t all, std::uint64_t cols, std::uint64_t d1,
                                 std::uint64_t d2) -> std::uint64_t {
    if (cols == all) {
        return 1;  // all n columns occupied: a full, non-attacking placement
    }
    std::uint64_t free = ~(cols | d1 | d2) & all;  // squares open in the current row
    std::uint64_t count = 0;
    while (free != 0) {
        const std::uint64_t bit = free & (~free + 1);  // lowest free square (branchless)
        free ^= bit;                                   // consume it
        // Placing a queen advances the diagonals by one column each; the AND with `all` in the
        // free-square expression above keeps out-of-board diagonal bits harmless.
        count += nqueens_count(all, cols | bit, (d1 | bit) << 1, (d2 | bit) >> 1);
    }
    return count;
}

auto count_nqueens(int n, bool parallel) -> Result<std::uint64_t> {
    if (n < 1 || n > 32) {
        return make_error<std::uint64_t>(MathError::domain_error);
    }
    const std::uint64_t all = (std::uint64_t{1} << n) - 1;  // n-bit board mask (n <= 32)

    if (!parallel) {
        return nqueens_count(all, 0, 0, 0);
    }

    // Fan the first row's n column placements across workers. Each starting column is a pure,
    // stateless subtree count sharing nothing mutable — exactly one GPU-thread-per-subtree
    // task. transform_index preserves index order; summation is associative, so the parallel
    // total equals the serial count regardless of worker count.
    const std::vector<std::uint64_t> subtotals = nimblecas::parallel::transform_index(
        static_cast<std::size_t>(n), [all](std::size_t j) -> std::uint64_t {
            const std::uint64_t bit = std::uint64_t{1} << j;  // queen in column j of row 0
            return nqueens_count(all, bit, bit << 1, bit >> 1);
        });

    std::uint64_t total = 0;
    for (const std::uint64_t s : subtotals) {
        total += s;
    }
    return total;
}

}  // namespace nimblecas
