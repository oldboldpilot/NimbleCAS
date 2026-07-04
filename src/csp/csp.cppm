// NimbleCAS finite-domain constraint satisfaction — AC-3, backtracking, forward checking (parallel).
// @author Olumuyiwa Oluwasanmi
//
// A deterministic finite-domain CSP toolkit. Variables are indices 0..n-1, each with a
// finite integer domain (values ascending, deduplicated by the caller). Constraints come
// in two flavours: binary relations R(x, y) and general n-ary relations over a variable
// scope. AC-3 enforces arc consistency over the BINARY constraints only; backtracking (with
// an optional forward-checking variant) and a solution counter honour BOTH binary and
// general constraints; and a parallel entry point fans the first variable's domain across
// workers.
//
// DETERMINISM: every solver returns the SAME answer regardless of thread count. Search uses
// a STATIC variable order (index 0, 1, ..., n-1) and tries each variable's values in the
// caller-supplied ascending order, so "the solution" is always the lexicographically-first
// complete assignment (compare by variable index, ties broken by ascending domain value).
// AC-3's worklist is a FIFO of arcs seeded and re-queued in arc-index order, so its pruned
// domains do not depend on scheduling either.
//
// HONESTY: backtracking is worst-case exponential (CSP is NP-complete); `solution_count`
// therefore takes a hard `limit` so enumeration can never run away. AC-3 only establishes
// arc consistency over binary constraints — it is a preprocessing/propagation step, NOT a
// complete solver: an arc-consistent CSP can still be unsatisfiable (e.g. three mutually
// distinct variables over a two-value domain stay arc consistent yet have no solution).
// An UNSATISFIABLE problem is not an error; it is a valid empty result (std::nullopt for the
// searches, 0 for the count, a disengaged optional for an emptied AC-3 domain). CSP search
// is irregular and branchy, so the realistic parallel win is CPU branch-parallelism /
// distribution across independent subtrees — NOT the GPU; there is deliberately no CUDA here.

export module nimblecas.csp;

import std;
import nimblecas.core;
import nimblecas.parallel;

export namespace nimblecas {

// A binary constraint on variables x and y: allowed(a, b) is true iff assigning x = a and
// y = b is permitted. The relation need not be symmetric; the value order matches (x, y).
struct BinaryConstraint {
    std::size_t x;
    std::size_t y;
    std::function<bool(std::int64_t, std::int64_t)> allowed;
};

// A general n-ary constraint over `scope`: allowed(values) is true iff the assignment that
// binds scope[k] = values[k] (values passed in scope order) is permitted.
struct Constraint {
    std::vector<std::size_t> scope;
    std::function<bool(std::span<const std::int64_t>)> allowed;
};

// A finite-domain CSP: per-variable domains (index i owns domains[i]) plus binary and
// general constraints. Variables are the indices 0..domains.size()-1.
struct Csp {
    std::vector<std::vector<std::int64_t>> domains;
    std::vector<BinaryConstraint> binary;
    std::vector<Constraint> general;
};

// AC-3 arc consistency over the BINARY constraints only. Returns the pruned per-variable
// domains, or an engaged Result holding std::nullopt if any domain is emptied (the CSP is
// arc-inconsistent — hence certainly unsatisfiable). General constraints are ignored here.
// Shape faults (empty variable set, out-of-range constraint scope, empty initial domain)
// are domain_error. The worklist is processed in a fixed arc-index order, so the pruned
// domains are independent of any scheduling.
[[nodiscard]] auto ac3(const Csp& csp)
    -> Result<std::optional<std::vector<std::vector<std::int64_t>>>>;

// Plain backtracking honouring both binary and general constraints. Returns the
// lexicographically-first complete assignment (static variable order, ascending values), or
// std::nullopt if the CSP is unsatisfiable. A returned assignment is verified to satisfy
// every constraint before it is handed back; a violation would be an internal bug and is
// reported as undefined_value.
[[nodiscard]] auto backtracking_search(const Csp& csp)
    -> Result<std::optional<std::vector<std::int64_t>>>;

// Backtracking augmented with FORWARD CHECKING: after each assignment the domains of the
// as-yet-unassigned variables are pruned of values that can no longer be consistent, and a
// branch is abandoned the instant a future domain empties.
//
// VARIABLE ORDER — a deliberate choice: the classic companion to forward checking is MRV
// (minimum-remaining-values) dynamic ordering. MRV, however, changes WHICH variable is
// branched on first, and therefore which complete assignment is discovered "first" — it
// would no longer be the lexicographically-first one that `backtracking_search` returns.
// Because the contract here is to return the SAME lexicographically-first solution, this
// routine KEEPS THE STATIC variable order and uses forward checking purely for pruning.
// Forward checking only removes values that no solution could use, so the first complete
// assignment found is bit-for-bit identical to `backtracking_search`'s — just reached with
// less thrashing.
[[nodiscard]] auto backtracking_search_fc(const Csp& csp)
    -> Result<std::optional<std::vector<std::int64_t>>>;

// Counts complete, constraint-satisfying assignments up to `limit` (limit == 0 means no cap;
// enumeration then explores the whole tree). The cap is a runaway guard: once `limit`
// solutions have been seen the search stops and returns `limit`. Returns 0 for an
// unsatisfiable CSP. Shape faults are domain_error.
[[nodiscard]] auto solution_count(const Csp& csp, std::uint64_t limit) -> Result<std::uint64_t>;

// Parallel search: the FIRST variable's domain is split across workers via
// nimblecas::parallel::transform_index — worker j fixes variable 0 to the j-th domain value
// and solves that subtree INDEPENDENTLY, as a pure function of the value it was handed (it
// copies the CSP and shares nothing mutable). The lexicographically-first solution overall
// is the one from the smallest j that yields any solution, since variable 0 is the primary
// lexicographic key and its domain is ascending; results are therefore combined in index
// order. The answer is identical to `backtracking_search` for any worker count.
//
// This stateless per-branch decomposition is exactly what makes the problem distributable:
// each first-variable value is one self-contained task, so the same split maps onto an
// external orchestrator (one shard per value) with no shared state to reconcile.
[[nodiscard]] auto parallel_search(const Csp& csp)
    -> Result<std::optional<std::vector<std::int64_t>>>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

// Structural validation shared by every entry point. Returns the offending MathError, or
// std::nullopt when the CSP is well formed. An unsatisfiable-but-well-formed CSP is NOT a
// fault here — that is decided by the solvers.
[[nodiscard]] auto csp_validate(const Csp& csp) -> std::optional<MathError> {
    const std::size_t n = csp.domains.size();
    if (n == 0) {
        return MathError::domain_error;  // empty variable set
    }
    for (const auto& dom : csp.domains) {
        if (dom.empty()) {
            return MathError::domain_error;  // empty initial domain
        }
    }
    for (const auto& bc : csp.binary) {
        if (bc.x >= n || bc.y >= n) {
            return MathError::domain_error;  // binary constraint references an out-of-range variable
        }
    }
    for (const auto& c : csp.general) {
        if (c.scope.empty()) {
            return MathError::domain_error;  // a constraint over no variables is malformed
        }
        for (const std::size_t idx : c.scope) {
            if (idx >= n) {
                return MathError::domain_error;  // general scope references an out-of-range variable
            }
        }
    }
    return std::nullopt;
}

// Consistency check for the constraints that become fully assigned exactly when the
// variables 0..depth are bound and `focus` (<= depth) is their latest addition. A constraint
// is "activated" the moment its highest-index variable is assigned; checking only at
// activation visits each constraint once per relevant node without redundant re-tests.
[[nodiscard]] auto ok_activated(const Csp& csp, const std::vector<std::int64_t>& assign,
                                std::size_t depth, std::size_t focus) -> bool {
    for (const auto& bc : csp.binary) {
        if ((bc.x == focus || bc.y == focus) && bc.x <= depth && bc.y <= depth) {
            if (!bc.allowed(assign[bc.x], assign[bc.y])) {
                return false;
            }
        }
    }
    for (const auto& c : csp.general) {
        bool has_focus = false;
        std::size_t max_idx = 0;
        for (const std::size_t idx : c.scope) {
            has_focus = has_focus || (idx == focus);
            max_idx = std::max(max_idx, idx);
        }
        if (has_focus && max_idx <= depth) {  // whole scope now assigned
            std::vector<std::int64_t> vals;
            vals.reserve(c.scope.size());
            for (const std::size_t idx : c.scope) {
                vals.push_back(assign[idx]);
            }
            if (!c.allowed(std::span<const std::int64_t>(vals))) {
                return false;
            }
        }
    }
    return true;
}

// Verifies that a COMPLETE assignment satisfies every constraint. Used as an internal
// correctness guard on returned solutions (a failure here would be a solver bug).
[[nodiscard]] auto satisfies_all(const Csp& csp, const std::vector<std::int64_t>& assign)
    -> bool {
    for (const auto& bc : csp.binary) {
        if (!bc.allowed(assign[bc.x], assign[bc.y])) {
            return false;
        }
    }
    for (const auto& c : csp.general) {
        std::vector<std::int64_t> vals;
        vals.reserve(c.scope.size());
        for (const std::size_t idx : c.scope) {
            vals.push_back(assign[idx]);
        }
        if (!c.allowed(std::span<const std::int64_t>(vals))) {
            return false;
        }
    }
    return true;
}

// Plain backtracking over the static variable order. Fills `assign` and returns true on the
// first complete, consistent extension of the current prefix.
[[nodiscard]] auto backtrack_rec(const Csp& csp, std::size_t depth,
                                 std::vector<std::int64_t>& assign) -> bool {
    if (depth == csp.domains.size()) {
        return true;
    }
    for (const std::int64_t v : csp.domains[depth]) {  // ascending → lexicographic-first search
        assign[depth] = v;
        if (ok_activated(csp, assign, depth, depth) && backtrack_rec(csp, depth + 1, assign)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] auto search_first_static(const Csp& csp)
    -> std::optional<std::vector<std::int64_t>> {
    std::vector<std::int64_t> assign(csp.domains.size(), 0);
    if (backtrack_rec(csp, 0, assign)) {
        return assign;
    }
    return std::nullopt;
}

// Forward-checking prune: with variables 0..depth assigned (depth just bound to
// assign[depth]), filter every future variable's domain in `domains` down to the values
// still consistent with the assigned prefix. Returns false if any future domain empties.
[[nodiscard]] auto fc_prune(const Csp& csp, const std::vector<std::int64_t>& assign,
                            std::size_t depth,
                            std::vector<std::vector<std::int64_t>>& domains) -> bool {
    const std::size_t n = domains.size();
    for (std::size_t f = depth + 1; f < n; ++f) {
        std::vector<std::int64_t> kept;
        kept.reserve(domains[f].size());
        for (const std::int64_t b : domains[f]) {
            bool ok = true;
            // Binary constraints between f and the just-assigned variable `depth`. Earlier
            // assignments were already reflected in the inherited (pruned) domains, so it
            // suffices to test against `depth`.
            for (const auto& bc : csp.binary) {
                if (bc.x == f && bc.y == depth) {
                    if (!bc.allowed(b, assign[depth])) {
                        ok = false;
                        break;
                    }
                } else if (bc.y == f && bc.x == depth) {
                    if (!bc.allowed(assign[depth], b)) {
                        ok = false;
                        break;
                    }
                }
            }
            // General constraints for which f is the ONLY unassigned variable can also be
            // checked now (every other scope member is already bound).
            if (ok) {
                for (const auto& c : csp.general) {
                    bool has_f = false;
                    bool others_assigned = true;
                    for (const std::size_t idx : c.scope) {
                        if (idx == f) {
                            has_f = true;
                        } else if (idx > depth) {
                            others_assigned = false;
                        }
                    }
                    if (has_f && others_assigned) {
                        std::vector<std::int64_t> vals;
                        vals.reserve(c.scope.size());
                        for (const std::size_t idx : c.scope) {
                            vals.push_back(idx == f ? b : assign[idx]);
                        }
                        if (!c.allowed(std::span<const std::int64_t>(vals))) {
                            ok = false;
                            break;
                        }
                    }
                }
            }
            if (ok) {
                kept.push_back(b);
            }
        }
        if (kept.empty()) {
            return false;  // dead end: a future variable has no legal value
        }
        domains[f] = std::move(kept);
    }
    return true;
}

// Forward-checking backtracking over the static variable order (see the header note on why
// MRV is intentionally not used). `domains` carries the progressively pruned live domains.
[[nodiscard]] auto fc_rec(const Csp& csp, std::size_t depth, std::vector<std::int64_t>& assign,
                          std::vector<std::vector<std::int64_t>>& domains) -> bool {
    if (depth == domains.size()) {
        return true;
    }
    for (const std::int64_t v : domains[depth]) {  // pruned domain, still ascending
        assign[depth] = v;
        if (!ok_activated(csp, assign, depth, depth)) {
            continue;
        }
        std::vector<std::vector<std::int64_t>> next = domains;  // private copy for this branch
        next[depth] = {v};
        if (fc_prune(csp, assign, depth, next) && fc_rec(csp, depth + 1, assign, next)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] auto search_first_fc(const Csp& csp)
    -> std::optional<std::vector<std::int64_t>> {
    std::vector<std::int64_t> assign(csp.domains.size(), 0);
    std::vector<std::vector<std::int64_t>> domains = csp.domains;
    if (fc_rec(csp, 0, assign, domains)) {
        return assign;
    }
    return std::nullopt;
}

// Enumerates consistent complete assignments, incrementing `count`, stopping early once the
// (non-zero) `limit` is reached.
auto count_rec(const Csp& csp, std::size_t depth, std::vector<std::int64_t>& assign,
               std::uint64_t limit, std::uint64_t& count) -> void {
    if (limit != 0 && count >= limit) {
        return;
    }
    if (depth == csp.domains.size()) {
        ++count;
        return;
    }
    for (const std::int64_t v : csp.domains[depth]) {
        assign[depth] = v;
        if (ok_activated(csp, assign, depth, depth)) {
            count_rec(csp, depth + 1, assign, limit, count);
            if (limit != 0 && count >= limit) {
                return;
            }
        }
    }
}

auto ac3(const Csp& csp) -> Result<std::optional<std::vector<std::vector<std::int64_t>>>> {
    using Ret = std::optional<std::vector<std::vector<std::int64_t>>>;
    if (const auto err = csp_validate(csp)) {
        return make_error<Ret>(*err);
    }

    std::vector<std::vector<std::int64_t>> domains = csp.domains;

    // Each binary constraint contributes two directed arcs; `rev` records which endpoint is
    // being revised so allowed(x, y) is evaluated with the values in the right slots.
    struct Arc {
        std::size_t ci;
        bool rev;
        std::size_t from;
        std::size_t to;
    };
    std::vector<Arc> arcs;
    arcs.reserve(csp.binary.size() * 2);
    for (std::size_t ci = 0; ci < csp.binary.size(); ++ci) {
        const auto& bc = csp.binary[ci];
        arcs.push_back(Arc{ci, false, bc.x, bc.y});
        arcs.push_back(Arc{ci, true, bc.y, bc.x});
    }

    // FIFO worklist seeded in arc-index order; `queued` prevents duplicate enqueues. Both the
    // seed order and the re-queue scan below run in ascending arc index, so the fixed point
    // reached is independent of any scheduling — the determinism guarantee.
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
        const auto& bc = csp.binary[arc.ci];

        // Revise D[from]: keep a only if some b in D[to] supports it under the constraint.
        std::vector<std::int64_t> kept;
        kept.reserve(domains[arc.from].size());
        bool changed = false;
        for (const std::int64_t a : domains[arc.from]) {
            bool supported = false;
            for (const std::int64_t b : domains[arc.to]) {
                const bool ok = arc.rev ? bc.allowed(b, a) : bc.allowed(a, b);
                if (ok) {
                    supported = true;
                    break;
                }
            }
            if (supported) {
                kept.push_back(a);
            } else {
                changed = true;
            }
        }

        if (changed) {
            domains[arc.from] = std::move(kept);
            if (domains[arc.from].empty()) {
                return Ret{std::nullopt};  // arc-inconsistent: a valid empty (not an error)
            }
            // Re-queue every arc that reads INTO `from` (its neighbours may lose support),
            // except the reverse arc of THIS SAME constraint. The classic "skip Xj"
            // optimisation is sound only per-constraint: a value was dropped from D[from]
            // because it had no support under constraint `arc.ci`, so it cannot have been the
            // support for any value under that same constraint's reverse arc. It CAN, however,
            // have supported a value under a *different* constraint on the same ordered pair,
            // so we must exclude only the same-constraint reverse (arc.ci, from<->to), not
            // every arc coming from `arc.to` — otherwise AC-3 can return non-arc-consistent
            // domains when two binary constraints share a variable pair.
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

auto backtracking_search(const Csp& csp) -> Result<std::optional<std::vector<std::int64_t>>> {
    using Ret = std::optional<std::vector<std::int64_t>>;
    if (const auto err = csp_validate(csp)) {
        return make_error<Ret>(*err);
    }
    auto sol = search_first_static(csp);
    if (sol && !satisfies_all(csp, *sol)) {
        return make_error<Ret>(MathError::undefined_value);  // a returned solution must be valid
    }
    return sol;
}

auto backtracking_search_fc(const Csp& csp)
    -> Result<std::optional<std::vector<std::int64_t>>> {
    using Ret = std::optional<std::vector<std::int64_t>>;
    if (const auto err = csp_validate(csp)) {
        return make_error<Ret>(*err);
    }
    auto sol = search_first_fc(csp);
    if (sol && !satisfies_all(csp, *sol)) {
        return make_error<Ret>(MathError::undefined_value);
    }
    return sol;
}

auto solution_count(const Csp& csp, std::uint64_t limit) -> Result<std::uint64_t> {
    if (const auto err = csp_validate(csp)) {
        return make_error<std::uint64_t>(*err);
    }
    std::vector<std::int64_t> assign(csp.domains.size(), 0);
    std::uint64_t count = 0;
    count_rec(csp, 0, assign, limit, count);
    return count;
}

auto parallel_search(const Csp& csp) -> Result<std::optional<std::vector<std::int64_t>>> {
    using Ret = std::optional<std::vector<std::int64_t>>;
    if (const auto err = csp_validate(csp)) {
        return make_error<Ret>(*err);
    }

    const std::vector<std::int64_t>& d0 = csp.domains[0];
    const std::size_t m = d0.size();

    // Each worker is a pure function of its first-variable value: it copies the CSP (sharing
    // nothing mutable — the constraint functors are read-only) and solves the subtree with
    // variable 0 fixed. Distinct indices touch disjoint copies, so the map is data-race free.
    std::vector<Ret> results = nimblecas::parallel::transform_index(
        m, [&csp, &d0](std::size_t j) -> Ret {
            Csp sub = csp;
            sub.domains[0] = {d0[j]};
            return search_first_static(sub);
        });

    // Combine in index order: the smallest j with a solution yields the lexicographically-
    // first assignment overall (variable 0 is the primary key, d0 is ascending, and each
    // worker already returned the lexicographically-first extension for its fixed value).
    for (std::size_t j = 0; j < m; ++j) {
        if (results[j]) {
            if (!satisfies_all(csp, *results[j])) {
                return make_error<Ret>(MathError::undefined_value);
            }
            return results[j];
        }
    }
    return Ret{std::nullopt};
}

}  // namespace nimblecas
