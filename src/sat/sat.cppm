// NimbleCAS internal parallel SAT solver — DPLL, CDCL, WalkSAT/GSAT (distributed-ready).
// @author Olumuyiwa Oluwasanmi
//
// A deterministic Boolean-satisfiability substrate for NimbleCAS. Conjunctive normal form
// (CNF) is the shared representation: a literal is a signed 64-bit integer (variable v in
// 1..num_vars, negative meaning ¬v, 0 illegal), a clause is a disjunction of literals, and a
// formula is their conjunction. Four engines are provided — two COMPLETE (DPLL, CDCL) that
// can prove both satisfiability and unsatisfiability, and two STOCHASTIC LOCAL SEARCH engines
// (WalkSAT, GSAT) that can only ever exhibit a model or give up. Above them sits a CPU
// portfolio and a stateless-shard entry point that make the module distributed-ready.
//
// DETERMINISM. The complete solvers are pure functions of the formula (fixed tie-breaks, no
// RNG). The local-search solvers are pure functions of (formula, seed): they draw from the
// counter-based nimblecas.rng substrate, so equal seeds reproduce equal runs bit-for-bit.
// There is no time or entropy seeding anywhere. Every solver is hard-capped (recursion is
// bounded by the variable count; the iterative engines by max_conflicts / max_flips /
// max_restarts), so each call always terminates.
//
// HONESTY — COST AND COMPLETENESS. SAT is NP-complete: the complete solvers can take time
// exponential in the number of variables (DPLL is bounded only by the search tree; CDCL is
// bounded additionally by its max_conflicts budget and returns `unknown` when that budget is
// spent before a verdict). The local-search solvers are INCOMPLETE: they can find a model but
// can NEVER prove unsatisfiability, so they return only `satisfiable` or `unknown` and MUST
// NEVER return `unsatisfiable`. The caps guarantee termination, not a verdict.
//
// HONESTY — PARALLELISM AND GPUs. SAT search is deeply irregular (data-dependent branching,
// pointer chasing, dynamic clause learning) and is a POOR fit for GPUs; there is deliberately
// no CUDA here. The realistic parallelism is a CPU PORTFOLIO: launch several INDEPENDENT
// solver configurations on the same formula and race them. Because each configuration is a
// pure function of (formula, worker_index, base_seed) with no shared mutable state, a worker
// is also the natural unit of DISTRIBUTED execution — `solve_shard` exposes exactly one such
// worker so an external orchestrator can place one shard per process/machine and merge the
// per-shard verdicts with a trivial associative all-reduce (any UNSAT from a complete worker
// wins; else the model from the lowest worker index; else unknown).
//
// All failure travels the railway (Result<T> / MathError), never an exception. A malformed
// CNF — a literal equal to 0, a variable index exceeding num_vars, or num_vars == 0 — is a
// domain_error.

export module nimblecas.sat;

import std;
import nimblecas.core;
import nimblecas.rng;
import nimblecas.parallel;

export namespace nimblecas {

// A CNF formula over variables 1..num_vars. Each inner vector is a clause (a disjunction of
// literals); the formula is the conjunction of its clauses. A literal is a signed variable
// index: +v asserts variable v, −v asserts ¬v, and 0 is never a legal literal. An empty
// clause is permitted and denotes falsity (an unsatisfiable formula); an empty clause list
// denotes truth. num_vars == 0 is malformed.
struct Cnf {
    std::size_t num_vars;
    std::vector<std::vector<std::int64_t>> clauses;
};

// The three possible outcomes of a solve. `unknown` is only ever produced by a resource cap
// (CDCL's conflict budget, or a local-search flip/restart budget), never as a substitute for
// a verdict a complete solver could have reached given time.
enum class SatVerdict : std::uint8_t { satisfiable, unsatisfiable, unknown };

// The result of a solve. `model` is 0-indexed and, when present, has size num_vars with
// model[v-1] giving the Boolean value assigned to variable v. It is non-empty only when
// verdict == satisfiable (and is then guaranteed to satisfy every clause — see
// verify_assignment); it is empty for unsatisfiable and unknown.
struct SatResult {
    SatVerdict verdict;
    std::vector<bool> model;
};

// True iff `model` satisfies EVERY clause of `cnf` — i.e. each clause contains at least one
// literal made true by the assignment (model[v-1] is variable v; +v is satisfied when the
// variable is true, −v when it is false). Returns false for a model whose size differs from
// num_vars, for a malformed CNF (literal 0 or an out-of-range variable), and for any clause
// (including the empty clause) with no satisfied literal. Every solver in this module calls
// this on its own candidate before returning `satisfiable`; a failed self-check is treated as
// an internal bug and reported as undefined_value rather than a false positive.
[[nodiscard]] auto verify_assignment(const Cnf& cnf, const std::vector<bool>& model) -> bool;

// COMPLETE solver: classic DPLL — unit propagation + pure-literal elimination + chronological
// backtracking over the lowest-index unassigned variable (true before false). Always reaches
// a definitive verdict (never `unknown`): `satisfiable` with a verified model, or
// `unsatisfiable`. WORST CASE is exponential in num_vars. Returns domain_error for a malformed
// CNF; undefined_value only if its own model fails the internal self-check (a bug).
[[nodiscard]] auto dpll(const Cnf& cnf) -> Result<SatResult>;

// COMPLETE solver: CDCL (conflict-driven clause learning) with 1-UIP learned clauses,
// non-chronological backjumping, and Luby restarts. Deterministic — no RNG, fixed decision
// order (lowest-index variable, false first) and tie-breaks. It performs at most
// `max_conflicts` conflict analyses: if it reaches a verdict within that budget it returns
// `satisfiable` (verified model) or `unsatisfiable`; if the budget is spent first it returns
// `unknown`. A conflict at decision level 0 is always UNSAT regardless of budget. Returns
// domain_error for a malformed CNF; undefined_value only on a failed self-check.
[[nodiscard]] auto cdcl(const Cnf& cnf, std::uint64_t max_conflicts) -> Result<SatResult>;

// STOCHASTIC LOCAL SEARCH: WalkSAT. From a random full assignment, repeatedly pick a random
// currently-unsatisfied clause; with probability `noise` flip a uniformly random variable in
// it, otherwise flip the variable in it whose flip breaks the fewest currently-satisfied
// clauses (ties broken by clause order). Runs at most `max_flips` flips.
//
// INCOMPLETE — CANNOT PROVE UNSAT: returns `satisfiable` with a verified model if one is found
// within the flip budget, else `unknown`. It MUST NEVER return `unsatisfiable`. Deterministic
// given `seed`. Returns domain_error for a malformed CNF or for noise outside [0, 1].
[[nodiscard]] auto walksat(const Cnf& cnf, std::uint64_t max_flips, double noise,
                           std::uint64_t seed) -> Result<SatResult>;

// STOCHASTIC LOCAL SEARCH: GSAT. Runs 1 + max_restarts independent tries (an initial random
// assignment plus max_restarts restarts). Within each try it performs up to `max_flips`
// greedy flips: at each step it flips the variable whose flip most reduces the number of
// unsatisfied clauses (net gain = clauses made satisfied − clauses broken; ties broken by the
// lowest variable index), accepting sideways and uphill moves as classic GSAT does.
//
// INCOMPLETE — CANNOT PROVE UNSAT: returns `satisfiable` with a verified model if any try
// reaches one, else `unknown`. It MUST NEVER return `unsatisfiable`. Deterministic given
// `seed`. Returns domain_error for a malformed CNF.
[[nodiscard]] auto gsat(const Cnf& cnf, std::uint64_t max_flips, std::uint64_t max_restarts,
                        std::uint64_t seed) -> Result<SatResult>;

// PARALLEL CPU PORTFOLIO. Runs `workers` INDEPENDENT solver configurations concurrently over
// the same formula via nimblecas::parallel::transform_index and merges their verdicts. Each
// worker is a pure function of (cnf, worker_index, base_seed) with NO shared mutable state:
// the configuration is chosen by worker_index (worker 0 is always complete DPLL, guaranteeing
// the portfolio itself always reaches a definitive verdict; other indices cycle through CDCL
// with index-varied conflict budgets, WalkSAT, and GSAT), and each stochastic worker is
// seeded with splitmix64(base_seed ^ worker_index).
//
// DETERMINISTIC MERGE (an associative all-reduce): any `unsatisfiable` from a complete worker
// is definitive and wins; otherwise the `satisfiable` model from the LOWEST worker index is
// returned; otherwise `unknown`. Because each worker is stateless and keyed only by
// (index, seed), the merge does not depend on scheduling or worker count. Returns domain_error
// for a malformed CNF or workers == 0.
[[nodiscard]] auto solve_portfolio(const Cnf& cnf, std::uint64_t base_seed, std::size_t workers)
    -> Result<SatResult>;

// DISTRIBUTED SHARD. Returns the verdict of exactly ONE portfolio worker — the configuration
// keyed by (shard_index, base_seed) — computed with no shared state. This is the unit of
// distributed execution: an external orchestrator runs one process/task per shard, possibly on
// separate machines, over shards 0..num_shards-1, then merges the per-shard SatResults with
// the SAME associative reduction solve_portfolio uses (UNSAT from a complete shard wins; else
// the model from the lowest shard index; else unknown). The reduction is a trivial commutative
// combine, so shards can be collected in any order. `num_shards` bounds the shard space only
// (a shard's computation depends solely on its own index and base_seed, never on num_shards —
// that independence is exactly what makes the design distributable). Returns domain_error for a
// malformed CNF, num_shards == 0, or shard_index >= num_shards.
[[nodiscard]] auto solve_shard(const Cnf& cnf, std::size_t shard_index, std::size_t num_shards,
                               std::uint64_t base_seed) -> Result<SatResult>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

namespace {

// Variable index (1-based) named by a literal. Computed through unsigned arithmetic so that
// even INT64_MIN (whose negation overflows int64) yields a well-defined magnitude rather than
// undefined behaviour; well-formedness (checked separately) guarantees the result is in
// 1..num_vars for real formulas.
[[nodiscard]] auto lit_var(std::int64_t lit) noexcept -> std::size_t {
    const std::uint64_t mag =
        lit < 0 ? (0ULL - static_cast<std::uint64_t>(lit)) : static_cast<std::uint64_t>(lit);
    return static_cast<std::size_t>(mag);
}

// A CNF is well-formed iff it has at least one variable and every literal is non-zero with a
// magnitude within 1..num_vars. Empty clauses are allowed (they denote falsity); an empty
// clause list is allowed (it denotes truth).
[[nodiscard]] auto is_wellformed(const Cnf& cnf) noexcept -> bool {
    if (cnf.num_vars == 0) {
        return false;
    }
    for (const auto& clause : cnf.clauses) {
        for (const std::int64_t lit : clause) {
            if (lit == 0) {
                return false;
            }
            if (lit_var(lit) > cnf.num_vars) {
                return false;
            }
        }
    }
    return true;
}

// Whether `lit` is made true by a full Boolean model (model[v-1] is the value of variable v).
[[nodiscard]] auto lit_true_under(std::int64_t lit, const std::vector<bool>& model) noexcept
    -> bool {
    const bool var_value = model[lit_var(lit) - 1];
    return lit > 0 ? var_value : !var_value;
}

// Whether a clause has at least one satisfied literal under a full model.
[[nodiscard]] auto clause_satisfied(const std::vector<std::int64_t>& clause,
                                    const std::vector<bool>& model) noexcept -> bool {
    for (const std::int64_t lit : clause) {
        if (lit_true_under(lit, model)) {
            return true;
        }
    }
    return false;
}

// Builds the definitive SatResult for a full model, after re-checking it against the formula.
// The self-check is defensive: if a solver ever hands back a model that does not actually
// satisfy the formula, that is a bug and is surfaced as undefined_value rather than a bogus
// `satisfiable`.
[[nodiscard]] auto satisfiable_from(const Cnf& cnf, std::vector<bool> model)
    -> Result<SatResult> {
    if (!verify_assignment(cnf, model)) {
        return make_error<SatResult>(MathError::undefined_value);
    }
    return SatResult{SatVerdict::satisfiable, std::move(model)};
}

// The i-th term (i >= 1) of the Luby restart sequence 1,1,2,1,1,2,4,1,1,2,1,1,2,4,8,…, which
// gives CDCL a schedule of restart intervals with good worst-case guarantees. Recursive and
// pure; the recursion depth is logarithmic in i.
[[nodiscard]] auto luby(std::uint64_t i) -> std::uint64_t {
    for (std::uint64_t k = 1; k < 64; ++k) {
        if (i == (1ULL << k) - 1) {
            return 1ULL << (k - 1);
        }
    }
    std::uint64_t k = 1;
    while ((1ULL << k) - 1 < i) {
        ++k;
    }
    return luby(i - (1ULL << (k - 1)) + 1);
}

// ---------------------------------------------------------------------------
// DPLL search over a tri-state assignment (value[v] ∈ {-1,0,+1}, index 1..num_vars).
// ---------------------------------------------------------------------------

// Truth of a literal under the partial assignment: +1 satisfied, -1 falsified, 0 unassigned.
[[nodiscard]] auto partial_lit(std::int64_t lit, const std::vector<std::int8_t>& value) noexcept
    -> int {
    const std::int8_t s = value[lit_var(lit)];
    if (s == 0) {
        return 0;
    }
    const bool lit_sat = lit > 0 ? (s == 1) : (s == -1);
    return lit_sat ? 1 : -1;
}

enum class ClauseState : std::uint8_t { satisfied, conflict, unit, unresolved };

// Classifies a clause under the partial assignment; on `unit` it writes the single remaining
// unassigned literal to `out_unit`. A clause with no unassigned literal and no satisfied
// literal (including the empty clause) is a `conflict`.
[[nodiscard]] auto classify(const std::vector<std::int64_t>& clause,
                            const std::vector<std::int8_t>& value, std::int64_t& out_unit)
    -> ClauseState {
    int unassigned = 0;
    std::int64_t last = 0;
    for (const std::int64_t lit : clause) {
        const int v = partial_lit(lit, value);
        if (v == 1) {
            return ClauseState::satisfied;
        }
        if (v == 0) {
            ++unassigned;
            last = lit;
        }
    }
    if (unassigned == 0) {
        return ClauseState::conflict;
    }
    if (unassigned == 1) {
        out_unit = last;
        return ClauseState::unit;
    }
    return ClauseState::unresolved;
}

// Recursive DPLL. Drives unit propagation and pure-literal elimination to a fixpoint, then
// branches on the lowest-index unassigned variable (true first). Returns true with `value`
// holding a satisfying (partial) assignment, or false if the subtree is unsatisfiable.
[[nodiscard]] auto dpll_search(const Cnf& cnf, std::vector<std::int8_t>& value) -> bool {
    const std::size_t n = cnf.num_vars;

    for (bool progress = true; progress;) {
        progress = false;

        // Unit propagation to fixpoint: a clause with exactly one unassigned literal forces it.
        for (bool unit_found = true; unit_found;) {
            unit_found = false;
            for (const auto& clause : cnf.clauses) {
                std::int64_t unit_lit = 0;
                switch (classify(clause, value, unit_lit)) {
                    case ClauseState::conflict:
                        return false;
                    case ClauseState::unit: {
                        value[lit_var(unit_lit)] =
                            static_cast<std::int8_t>(unit_lit > 0 ? 1 : -1);
                        unit_found = true;
                        progress = true;
                        break;
                    }
                    default:
                        break;
                }
            }
        }

        // Pure-literal elimination: a variable that appears with only one polarity across the
        // still-unsatisfied clauses can be fixed to that polarity without losing any model.
        std::vector<bool> seen_pos(n + 1, false);
        std::vector<bool> seen_neg(n + 1, false);
        for (const auto& clause : cnf.clauses) {
            std::int64_t ignored = 0;
            if (classify(clause, value, ignored) == ClauseState::satisfied) {
                continue;
            }
            for (const std::int64_t lit : clause) {
                const std::size_t v = lit_var(lit);
                if (value[v] == 0) {
                    (lit > 0 ? seen_pos : seen_neg)[v] = true;
                }
            }
        }
        for (std::size_t v = 1; v <= n; ++v) {
            if (value[v] != 0) {
                continue;
            }
            if (seen_pos[v] && !seen_neg[v]) {
                value[v] = 1;
                progress = true;
            } else if (seen_neg[v] && !seen_pos[v]) {
                value[v] = -1;
                progress = true;
            }
        }
    }

    // Fixpoint reached with no conflict: pick the lowest unassigned variable to branch on. If
    // none remains, every clause is satisfied (an unsatisfied clause would have >= 2 unassigned
    // literals, contradicting full assignment), so the subtree is satisfiable.
    std::size_t branch = 0;
    for (std::size_t v = 1; v <= n; ++v) {
        if (value[v] == 0) {
            branch = v;
            break;
        }
    }
    if (branch == 0) {
        return true;
    }

    for (const std::int8_t choice : {std::int8_t{1}, std::int8_t{-1}}) {
        std::vector<std::int8_t> saved = value;
        value[branch] = choice;
        if (dpll_search(cnf, value)) {
            return true;
        }
        value = std::move(saved);
    }
    return false;
}

// ---------------------------------------------------------------------------
// CDCL engine. Watch-less propagation (clauses are rescanned; ample for NimbleCAS-scale
// formulas and fully deterministic), 1-UIP learning, non-chronological backjump, Luby
// restarts. All state is local to one solve; there is no global mutable state.
// ---------------------------------------------------------------------------
struct CdclEngine {
    std::size_t n;
    std::vector<std::vector<std::int64_t>> clauses;  // original clauses followed by learned ones
    std::vector<std::int8_t> value;                  // -1/0/+1, index 1..n
    std::vector<int> level;                          // decision level at which a var was set
    std::vector<std::int64_t> reason;                // implying clause index, or -1 for decisions
    std::vector<std::int64_t> trail;                 // assigned literals in assignment order
    std::vector<std::size_t> trail_lim;              // trail sizes at each decision boundary

    explicit CdclEngine(const Cnf& cnf)
        : n(cnf.num_vars),
          clauses(cnf.clauses),
          value(cnf.num_vars + 1, 0),
          level(cnf.num_vars + 1, 0),
          reason(cnf.num_vars + 1, -1) {}

    [[nodiscard]] auto decision_level() const noexcept -> int {
        return static_cast<int>(trail_lim.size());
    }

    // Truth of a literal under the current assignment: +1 true, -1 false, 0 unassigned.
    [[nodiscard]] auto lit_value(std::int64_t lit) const noexcept -> int {
        const std::int8_t s = value[lit_var(lit)];
        if (s == 0) {
            return 0;
        }
        const bool lit_true = lit > 0 ? (s == 1) : (s == -1);
        return lit_true ? 1 : -1;
    }

    auto enqueue(std::int64_t lit, std::int64_t reason_idx) -> void {
        const std::size_t v = lit_var(lit);
        value[v] = static_cast<std::int8_t>(lit > 0 ? 1 : -1);
        level[v] = decision_level();
        reason[v] = reason_idx;
        trail.push_back(lit);
    }

    auto new_decision_level() -> void { trail_lim.push_back(trail.size()); }

    // Undo every assignment made above decision level `lvl`, restoring those variables to
    // unassigned; keeps everything at level `lvl` and below (including all learned clauses).
    auto cancel_until(int lvl) -> void {
        if (decision_level() <= lvl) {
            return;
        }
        const std::size_t keep = trail_lim[static_cast<std::size_t>(lvl)];
        for (std::size_t i = trail.size(); i > keep; --i) {
            value[lit_var(trail[i - 1])] = 0;
        }
        trail.resize(keep);
        trail_lim.resize(static_cast<std::size_t>(lvl));
    }

    // Rescans clauses, enqueuing forced units, until a fixpoint or a conflict. Returns the
    // index of a falsified (all-literals-false, including empty) clause, or -1 if none.
    [[nodiscard]] auto propagate() -> std::int64_t {
        for (bool changed = true; changed;) {
            changed = false;
            for (std::size_t ci = 0; ci < clauses.size(); ++ci) {
                const auto& c = clauses[ci];
                int unassigned = 0;
                std::int64_t unit_lit = 0;
                bool satisfied = false;
                for (const std::int64_t lit : c) {
                    const int v = lit_value(lit);
                    if (v == 1) {
                        satisfied = true;
                        break;
                    }
                    if (v == 0) {
                        ++unassigned;
                        unit_lit = lit;
                    }
                }
                if (satisfied) {
                    continue;
                }
                if (unassigned == 0) {
                    return static_cast<std::int64_t>(ci);
                }
                if (unassigned == 1) {
                    enqueue(unit_lit, static_cast<std::int64_t>(ci));
                    changed = true;
                }
            }
        }
        return -1;
    }

    // 1-UIP conflict analysis. Resolves the conflict clause against the reasons of
    // current-level literals until a single current-level literal (the unique implication
    // point) remains, producing an asserting learned clause (all literals false under the
    // current assignment, the asserting literal at index 0) and the level to backjump to.
    auto analyze(std::int64_t confl, std::vector<std::int64_t>& learnt, int& out_btlevel) -> void {
        std::vector<char> seen(n + 1, 0);
        int path_count = 0;
        std::int64_t p = 0;  // 0 == "no pivot yet"
        learnt.push_back(0);  // reserve slot 0 for the asserting literal
        std::size_t index = trail.size();

        do {
            for (const std::int64_t q : clauses[static_cast<std::size_t>(confl)]) {
                const std::size_t vq = lit_var(q);
                if (p != 0 && vq == lit_var(p)) {
                    continue;  // skip the pivot variable resolved on the previous step
                }
                if (!seen[vq] && level[vq] > 0) {
                    seen[vq] = 1;
                    if (level[vq] >= decision_level()) {
                        ++path_count;
                    } else {
                        learnt.push_back(q);
                    }
                }
            }
            // Walk back along the trail to the most recent literal marked seen.
            while (index > 0 && !seen[lit_var(trail[index - 1])]) {
                --index;
            }
            --index;
            p = trail[index];
            seen[lit_var(p)] = 0;
            --path_count;
            confl = reason[lit_var(p)];
        } while (path_count > 0);

        learnt[0] = -p;  // asserting literal: negation of the unique implication point

        if (learnt.size() == 1) {
            out_btlevel = 0;
        } else {
            int maxlvl = 0;
            for (std::size_t i = 1; i < learnt.size(); ++i) {
                maxlvl = std::max(maxlvl, level[lit_var(learnt[i])]);
            }
            out_btlevel = maxlvl;
        }
    }

    // Runs CDCL under a conflict budget. Returns the verdict; on satisfiable, `model` is
    // filled. `unknown` means the budget was spent before a verdict.
    [[nodiscard]] auto solve(std::uint64_t max_conflicts, std::vector<bool>& model)
        -> SatVerdict {
        constexpr std::uint64_t restart_base = 100;
        std::uint64_t conflicts = 0;
        std::uint64_t conflicts_at_restart = 0;
        std::uint64_t restart_count = 1;

        while (true) {
            const std::int64_t confl = propagate();
            if (confl != -1) {
                if (decision_level() == 0) {
                    return SatVerdict::unsatisfiable;  // level-0 conflict is definitive UNSAT
                }
                ++conflicts;
                if (conflicts > max_conflicts) {
                    return SatVerdict::unknown;  // budget spent before a verdict
                }
                std::vector<std::int64_t> learnt;
                int btlevel = 0;
                analyze(confl, learnt, btlevel);
                cancel_until(btlevel);
                clauses.push_back(learnt);
                enqueue(learnt[0], static_cast<std::int64_t>(clauses.size() - 1));

                if (conflicts - conflicts_at_restart >= restart_base * luby(restart_count)) {
                    cancel_until(0);  // learned clauses are kept, so progress is retained
                    conflicts_at_restart = conflicts;
                    ++restart_count;
                }
            } else {
                std::size_t next = 0;
                for (std::size_t v = 1; v <= n; ++v) {
                    if (value[v] == 0) {
                        next = v;
                        break;
                    }
                }
                if (next == 0) {
                    model.assign(n, false);
                    for (std::size_t v = 1; v <= n; ++v) {
                        model[v - 1] = value[v] == 1;
                    }
                    return SatVerdict::satisfiable;
                }
                new_decision_level();
                enqueue(-static_cast<std::int64_t>(next), -1);  // decide false first
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Stochastic-local-search shared helpers.
// ---------------------------------------------------------------------------

// Occurrence lists: occ[v] holds the indices of every clause that mentions variable v (in
// either polarity). Built once per solve to make make/break counts cheap.
[[nodiscard]] auto build_occurrences(const Cnf& cnf)
    -> std::vector<std::vector<std::size_t>> {
    std::vector<std::vector<std::size_t>> occ(cnf.num_vars + 1);
    for (std::size_t ci = 0; ci < cnf.clauses.size(); ++ci) {
        for (const std::int64_t lit : cnf.clauses[ci]) {
            occ[lit_var(lit)].push_back(ci);
        }
    }
    return occ;
}

// The set of clause indices not satisfied by `model`.
[[nodiscard]] auto unsatisfied_clauses(const Cnf& cnf, const std::vector<bool>& model)
    -> std::vector<std::size_t> {
    std::vector<std::size_t> out;
    for (std::size_t ci = 0; ci < cnf.clauses.size(); ++ci) {
        if (!clause_satisfied(cnf.clauses[ci], model)) {
            out.push_back(ci);
        }
    }
    return out;
}

// Number of currently-satisfied clauses that would become unsatisfied if variable v flipped —
// i.e. clauses in which v's literal is the sole satisfying literal.
[[nodiscard]] auto break_count(const Cnf& cnf, const std::vector<bool>& model,
                               const std::vector<std::vector<std::size_t>>& occ, std::size_t v)
    -> std::size_t {
    std::size_t breaks = 0;
    for (const std::size_t ci : occ[v]) {
        const auto& clause = cnf.clauses[ci];
        std::size_t sat = 0;
        bool v_is_the_sat = false;
        for (const std::int64_t lit : clause) {
            if (lit_true_under(lit, model)) {
                ++sat;
                if (lit_var(lit) == v) {
                    v_is_the_sat = true;
                }
            }
        }
        if (sat == 1 && v_is_the_sat) {
            ++breaks;
        }
    }
    return breaks;
}

// A fresh uniformly-random full assignment drawn from the generator's stream.
[[nodiscard]] auto random_model(std::size_t n, Rng& rng) -> std::vector<bool> {
    std::vector<bool> model(n);
    for (std::size_t v = 0; v < n; ++v) {
        model[v] = (rng.next_u64() & 1ULL) != 0;
    }
    return model;
}

// A worker/shard configuration keyed purely by (cnf, index, base_seed) — the stateless unit
// shared by the portfolio and the distributed-shard entry points. Worker 0 is always complete
// DPLL, so any collection of workers that includes index 0 can reach a definitive verdict;
// other indices cycle through CDCL (with an index-varied conflict budget) and the two
// local-search engines (each seeded from the index). Returns a Result so a genuine internal
// error (e.g. a failed self-check) still travels the railway.
[[nodiscard]] auto solve_worker(const Cnf& cnf, std::size_t index, std::uint64_t base_seed)
    -> Result<SatResult> {
    const std::uint64_t seed = splitmix64(base_seed ^ static_cast<std::uint64_t>(index));
    switch (index % 4) {
        case 0:
            return dpll(cnf);
        case 1:
            return cdcl(cnf, 2000 + 1000 * static_cast<std::uint64_t>(index));
        case 2:
            return walksat(cnf, 200000, 0.5, seed);
        default:
            return gsat(cnf, 2000, 200, seed);
    }
}

}  // namespace

auto verify_assignment(const Cnf& cnf, const std::vector<bool>& model) -> bool {
    if (model.size() != cnf.num_vars) {
        return false;
    }
    for (const auto& clause : cnf.clauses) {
        bool satisfied = false;
        for (const std::int64_t lit : clause) {
            if (lit == 0 || lit_var(lit) > cnf.num_vars) {
                return false;  // malformed literal: cannot honestly verify
            }
            if (lit_true_under(lit, model)) {
                satisfied = true;
                break;
            }
        }
        if (!satisfied) {
            return false;  // an unsatisfied (or empty) clause
        }
    }
    return true;
}

auto dpll(const Cnf& cnf) -> Result<SatResult> {
    if (!is_wellformed(cnf)) {
        return make_error<SatResult>(MathError::domain_error);
    }

    std::vector<std::int8_t> value(cnf.num_vars + 1, 0);
    if (!dpll_search(cnf, value)) {
        return SatResult{SatVerdict::unsatisfiable, {}};
    }

    // Free variables (never forced) default to false; the self-check confirms the full model.
    std::vector<bool> model(cnf.num_vars, false);
    for (std::size_t v = 1; v <= cnf.num_vars; ++v) {
        model[v - 1] = value[v] == 1;
    }
    return satisfiable_from(cnf, std::move(model));
}

auto cdcl(const Cnf& cnf, std::uint64_t max_conflicts) -> Result<SatResult> {
    if (!is_wellformed(cnf)) {
        return make_error<SatResult>(MathError::domain_error);
    }

    CdclEngine engine(cnf);
    std::vector<bool> model;
    switch (engine.solve(max_conflicts, model)) {
        case SatVerdict::satisfiable:
            return satisfiable_from(cnf, std::move(model));
        case SatVerdict::unsatisfiable:
            return SatResult{SatVerdict::unsatisfiable, {}};
        default:
            return SatResult{SatVerdict::unknown, {}};
    }
}

auto walksat(const Cnf& cnf, std::uint64_t max_flips, double noise, std::uint64_t seed)
    -> Result<SatResult> {
    if (!is_wellformed(cnf)) {
        return make_error<SatResult>(MathError::domain_error);
    }
    if (noise < 0.0 || noise > 1.0) {
        return make_error<SatResult>(MathError::domain_error);
    }

    const auto occ = build_occurrences(cnf);
    auto rng = Rng::seeded(seed);
    std::vector<bool> model = random_model(cnf.num_vars, rng);

    for (std::uint64_t flip = 0; flip < max_flips; ++flip) {
        const std::vector<std::size_t> unsat = unsatisfied_clauses(cnf, model);
        if (unsat.empty()) {
            return satisfiable_from(cnf, std::move(model));  // never returns UNSAT — SLS cannot
        }

        // Pick a random unsatisfied clause. An empty clause is unsatisfiable by construction
        // and offers no variable to flip, so it is skipped (the run will end in `unknown`).
        const std::size_t pick = unsat[static_cast<std::size_t>(
            rng.next_int(0, static_cast<std::int64_t>(unsat.size()) - 1).value())];
        const auto& clause = cnf.clauses[pick];
        if (clause.empty()) {
            continue;
        }

        std::size_t flip_var = 0;
        if (rng.next_unit() < noise) {
            // Noise step: flip a uniformly random variable from the clause.
            const std::size_t k = static_cast<std::size_t>(
                rng.next_int(0, static_cast<std::int64_t>(clause.size()) - 1).value());
            flip_var = lit_var(clause[k]);
        } else {
            // Greedy step: flip the variable of least break-count (ties by clause order).
            std::size_t best_break = std::numeric_limits<std::size_t>::max();
            for (const std::int64_t lit : clause) {
                const std::size_t v = lit_var(lit);
                const std::size_t b = break_count(cnf, model, occ, v);
                if (b < best_break) {
                    best_break = b;
                    flip_var = v;
                }
            }
        }
        model[flip_var - 1] = !model[flip_var - 1];
    }

    if (unsatisfied_clauses(cnf, model).empty()) {
        return satisfiable_from(cnf, std::move(model));
    }
    return SatResult{SatVerdict::unknown, {}};
}

auto gsat(const Cnf& cnf, std::uint64_t max_flips, std::uint64_t max_restarts, std::uint64_t seed)
    -> Result<SatResult> {
    if (!is_wellformed(cnf)) {
        return make_error<SatResult>(MathError::domain_error);
    }

    const auto occ = build_occurrences(cnf);
    auto rng = Rng::seeded(seed);

    // 1 + max_restarts independent tries (initial assignment plus max_restarts restarts).
    for (std::uint64_t attempt = 0; attempt <= max_restarts; ++attempt) {
        std::vector<bool> model = random_model(cnf.num_vars, rng);

        for (std::uint64_t flip = 0; flip < max_flips; ++flip) {
            const std::vector<std::size_t> unsat = unsatisfied_clauses(cnf, model);
            if (unsat.empty()) {
                return satisfiable_from(cnf, std::move(model));  // never returns UNSAT
            }

            // Greedy: flip the variable of greatest net gain = (clauses made satisfied) −
            // (clauses broken). Every unsatisfied clause containing v is made satisfied by the
            // flip (v's literal there becomes true), so make(v) = #unsatisfied clauses on v.
            // Ties resolve to the lowest variable index. Sideways/uphill moves are permitted.
            std::vector<std::size_t> unsat_uses(cnf.num_vars + 1, 0);
            for (const std::size_t ci : unsat) {
                for (const std::int64_t lit : cnf.clauses[ci]) {
                    ++unsat_uses[lit_var(lit)];
                }
            }

            std::size_t best_var = 1;
            std::int64_t best_gain = std::numeric_limits<std::int64_t>::min();
            for (std::size_t v = 1; v <= cnf.num_vars; ++v) {
                const std::int64_t make = static_cast<std::int64_t>(unsat_uses[v]);
                const std::int64_t brk =
                    static_cast<std::int64_t>(break_count(cnf, model, occ, v));
                const std::int64_t gain = make - brk;
                if (gain > best_gain) {
                    best_gain = gain;
                    best_var = v;
                }
            }
            model[best_var - 1] = !model[best_var - 1];
        }
        // The flip loop tests satisfaction at the TOP of each iteration, so a model reached by
        // the very last flip of this attempt is only visible here — re-check before restarting,
        // exactly as walksat does, so GSAT never discards a model it actually found.
        if (unsatisfied_clauses(cnf, model).empty()) {
            return satisfiable_from(cnf, std::move(model));  // never returns UNSAT
        }
    }

    return SatResult{SatVerdict::unknown, {}};
}

// Deterministic associative merge of a set of worker/shard results: any UNSAT (necessarily
// from a complete worker, since local search never reports it) wins; otherwise the model from
// the lowest index; otherwise unknown. A genuine per-worker error is propagated from the
// lowest offending index.
namespace {

[[nodiscard]] auto merge_results(const std::vector<Result<SatResult>>& results)
    -> Result<SatResult> {
    for (const auto& r : results) {
        if (!r) {
            return make_error<SatResult>(r.error());
        }
    }
    for (const auto& r : results) {
        if (r->verdict == SatVerdict::unsatisfiable) {
            return SatResult{SatVerdict::unsatisfiable, {}};
        }
    }
    for (const auto& r : results) {
        if (r->verdict == SatVerdict::satisfiable) {
            return *r;  // lowest index wins
        }
    }
    return SatResult{SatVerdict::unknown, {}};
}

}  // namespace

auto solve_portfolio(const Cnf& cnf, std::uint64_t base_seed, std::size_t workers)
    -> Result<SatResult> {
    if (!is_wellformed(cnf)) {
        return make_error<SatResult>(MathError::domain_error);
    }
    if (workers == 0) {
        return make_error<SatResult>(MathError::domain_error);
    }

    // Each worker is a pure function of (cnf, index, base_seed): no shared mutable state, so
    // the fan-out is safe and the merged verdict is independent of scheduling (grain 1: one
    // task per worker, the backend auto-chunks).
    auto results = parallel::transform_index(
        workers, [&](std::size_t i) -> Result<SatResult> { return solve_worker(cnf, i, base_seed); },
        std::size_t{1});

    return merge_results(results);
}

auto solve_shard(const Cnf& cnf, std::size_t shard_index, std::size_t num_shards,
                 std::uint64_t base_seed) -> Result<SatResult> {
    if (!is_wellformed(cnf)) {
        return make_error<SatResult>(MathError::domain_error);
    }
    if (num_shards == 0 || shard_index >= num_shards) {
        return make_error<SatResult>(MathError::domain_error);
    }
    // One stateless worker, keyed only by (shard_index, base_seed): an external orchestrator
    // runs shards 0..num_shards-1 on separate tasks/machines and merges them with the same
    // reduction solve_portfolio applies (see merge_results).
    return solve_worker(cnf, shard_index, base_seed);
}

}  // namespace nimblecas
