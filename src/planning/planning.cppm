// NimbleCAS classical planning — SAS+ tasks, delete-relaxation heuristics, and heuristic search.
// @author Olumuyiwa Oluwasanmi
//
// The planning layer above `nimblecas.search`: a task is a set of finite-domain variables, a set
// of operators with preconditions and effects, an initial state and a partial goal, and a PLAN is
// a sequence of operators that carries the initial state into one satisfying the goal.
//
// WHY THIS SHAPE, AND NOT STRIPS PROPOSITIONS.
//
// SAS+ gives each variable a finite domain and lets a state be one value per variable, where
// STRIPS would use a set of independent propositions. The difference is not cosmetic: SAS+ makes
// mutual exclusion STRUCTURAL -- a package cannot be in two places because its variable holds one
// value -- where STRIPS has to encode that in axioms and every heuristic has to rediscover it.
// This is the representation Fast Downward and the planners built on it use, for that reason.
//
// THE HEURISTICS, AND WHAT EACH IS ACTUALLY WORTH.
//
// All three come from the DELETE RELAXATION: pretend a variable, once it has taken a value, keeps
// that value forever alongside its new one. The relaxed task is solvable in polynomial time, and
// the cost of relaxing is that the estimate ignores every way an action can undo progress.
//
//   h_max      ADMISSIBLE. The cost of a set of facts is the cost of its DEAREST member. Never
//              overestimates, so A* with it returns an OPTIMAL plan -- and it is weak, because
//              taking a maximum throws away everything the other facts cost.
//   h_add      INADMISSIBLE. The cost of a set is the SUM over its members. Far more informative
//              than h_max and systematically too large, because it counts shared subplans once
//              per fact that needs them. Use it to search fast, never to claim optimality.
//   h_ff       INADMISSIBLE, and the workhorse. It extracts an actual RELAXED PLAN and counts
//              it, so it never double-counts a shared action the way h_add does, and it yields
//              PREFERRED OPERATORS as a by-product -- the first actions of that relaxed plan,
//              which are the ones worth trying first. Greedy best-first search with h_ff and
//              preferred operators is the configuration that made satisficing planning practical,
//              and it is what `gbfs_plan` runs.
//
// WHAT IS GUARANTEED.
//
//   * `astar_plan` with h_max returns an OPTIMAL plan, or reports honestly that it could not.
//   * `gbfs_plan` returns a VALID plan and makes NO claim about its length. Every plan either
//     entry point returns is validated by replaying it from the initial state before it is
//     handed back -- a plan that does not reach the goal is reported as an error, never returned.
//   * Both are deterministic: ties break by the lowest operator index and then by the state's
//     canonical order, so the same task gives the same plan on every run and every machine.
//
// The distributed entry point is in `nimblecas.planning_dist`; this module is the task, the
// heuristics and the in-process searches.

export module nimblecas.planning;

import std;
import nimblecas.core;

export namespace nimblecas::planning {

// One variable's assignment: variable index and the value it takes.
struct FactPair {
    std::size_t var{0};
    std::int32_t value{0};

    [[nodiscard]] auto operator==(const FactPair&) const noexcept -> bool = default;
    [[nodiscard]] auto operator<=>(const FactPair&) const noexcept = default;
};

// A complete state: one value per variable.
using State = std::vector<std::int32_t>;

// A partial assignment: the conditions an operator needs, or the goal. Kept SORTED by variable, so
// two conditions over the same variables compare and hash identically however they were built.
using Condition = std::vector<FactPair>;

// One operator (an "action"). `cost` must be non-negative; unit costs are the common case and give
// plan length as the plan cost.
struct Operator {
    std::string name;
    Condition preconditions;
    Condition effects;
    std::int64_t cost{1};
};

// A SAS+ planning task.
//
// `domain_size[v]` is the number of values variable v may take, so its legal values are
// 0 .. domain_size[v] - 1.
struct Task {
    std::vector<std::int32_t> domain_size;
    std::vector<Operator> operators;
    State initial;
    Condition goal;

    [[nodiscard]] auto num_vars() const noexcept -> std::size_t { return domain_size.size(); }
};

// A plan: the operator indices to apply, in order, and the total cost.
struct Plan {
    std::vector<std::size_t> steps;
    std::int64_t cost{0};

    [[nodiscard]] auto length() const noexcept -> std::size_t { return steps.size(); }
};

// ---------------------------------------------------------------------------
// Validation and semantics.
// ---------------------------------------------------------------------------

// Checks everything the searches below rely on: domains positive, the initial state complete and
// in range, every precondition and effect naming a real variable and a legal value, conditions
// sorted and free of a variable mentioned twice, and no negative operator cost.
//
// `domain_error` names the first violation. Checking here means a malformed task is refused before
// a search spends time on it and before a heuristic reads past an array.
[[nodiscard]] auto validate(const Task& task) -> Result<void>;

// Whether `state` satisfies every pair in `condition`.
[[nodiscard]] auto holds(const State& state, const Condition& condition) -> bool;

// Whether the operator's preconditions hold in `state`.
[[nodiscard]] auto applicable(const Task& task, const State& state, std::size_t op_index) -> bool;

// `state` with the operator's effects applied. The operator must be applicable; `domain_error`
// otherwise, rather than a state that quietly ignored a precondition.
[[nodiscard]] auto apply(const Task& task, const State& state, std::size_t op_index)
    -> Result<State>;

// Replays `plan` from the initial state and returns the state it reaches, checking applicability
// at every step.
//
// `domain_error` if any step is inapplicable; `undefined_value` if the plan runs to the end without
// reaching the goal. Every plan this module returns has been through here first -- a search that
// produced an invalid plan is a bug, and this is what turns that bug into an error rather than a
// confidently wrong answer.
[[nodiscard]] auto validate_plan(const Task& task, const Plan& plan) -> Result<State>;

// ---------------------------------------------------------------------------
// Delete-relaxation heuristics.
// ---------------------------------------------------------------------------

// Estimated cost to the goal from `state`, taking the DEAREST fact of a set.
//
// ADMISSIBLE: never overestimates, so A* with it returns an optimal plan. Returns
// `std::nullopt` when the goal is unreachable even under the relaxation, which is a proof that it
// is unreachable in the real task -- the one thing a relaxation can prove outright.
[[nodiscard]] auto h_max(const Task& task, const State& state) -> std::optional<std::int64_t>;

// Estimated cost taking the SUM over a set's facts.
//
// INADMISSIBLE: systematically too large, because a subplan shared by several facts is counted
// once for each of them. More informative than h_max in practice, and unusable for optimality.
[[nodiscard]] auto h_add(const Task& task, const State& state) -> std::optional<std::int64_t>;

// The FF heuristic, and the preferred operators that come with it.
struct FfResult {
    // The cost of an extracted relaxed plan. `std::nullopt` when the goal is unreachable even
    // under the relaxation.
    std::optional<std::int64_t> value;
    // The operators of the relaxed plan applicable in `state` RIGHT NOW -- Hoffmann's helpful
    // actions. Ascending by operator index, so the ordering is the task's rather than the
    // extraction's. Empty when the goal is unreachable, and possibly empty when it is not.
    std::vector<std::size_t> preferred;
};

// Extracts a relaxed plan and counts it.
//
// INADMISSIBLE, but far better behaved than h_add: because it counts an actual plan, an action
// serving several goals is counted once. The preferred operators are the reason to call this
// rather than h_add -- they cut the branching factor sharply, and `gbfs_plan` uses them.
[[nodiscard]] auto h_ff(const Task& task, const State& state) -> FfResult;

// ---------------------------------------------------------------------------
// Search.
// ---------------------------------------------------------------------------

// What a search did, beyond the plan itself. Reported because "no plan" and "gave up" are
// different answers and a caller sizing a run needs to tell them apart.
struct SearchStats {
    std::uint64_t expanded{0};
    std::uint64_t generated{0};
    std::uint64_t evaluated{0};
};

struct PlanResult {
    Plan plan;
    SearchStats stats;
};

// OPTIMAL planning: A* with h_max.
//
// Returns a plan of minimum cost, validated before it is returned. `undefined_value` when the task
// is provably unsolvable -- the search exhausted every reachable state -- and `not_converged` when
// `max_expansions` ran out first, which means the answer is unknown rather than that none exists.
// Those two must not be confused, and the error codes keep them apart.
//
// Ties in f break toward the lower h, then toward the state discovered earlier, so the plan is
// reproducible.
[[nodiscard]] auto astar_plan(const Task& task, std::uint64_t max_expansions)
    -> Result<PlanResult>;

// SATISFICING planning: greedy best-first search on h_ff with PREFERRED OPERATORS.
//
// This is the configuration that made satisficing planning practical, and the preferred-operator
// part is most of why. Two open lists are kept -- one holding every successor, one holding only
// the successors reached by a preferred operator -- and expansion alternates between them. The
// preferred list finds plans quickly when the heuristic is pointing the right way; the ordinary
// list is what stops the search being trapped when it is not. Neither alone behaves as well.
//
// NO CLAIM OF OPTIMALITY. The plan is valid and usually far from shortest, and asking this
// function for an optimal plan is a category error -- use `astar_plan`.
//
// `undefined_value` when the reachable state space is exhausted without reaching the goal;
// `not_converged` when `max_expansions` ran out.
[[nodiscard]] auto gbfs_plan(const Task& task, std::uint64_t max_expansions)
    -> Result<PlanResult>;

// GBFS on h_add instead of h_ff, and with no preferred operators.
//
// Kept because it is the honest control for the pair above: h_ff's advantage is a claim, and a
// claim needs something to be measured against. The tests compare expansions between them.
[[nodiscard]] auto gbfs_plan_hadd(const Task& task, std::uint64_t max_expansions)
    -> Result<PlanResult>;

// ---------------------------------------------------------------------------
// Wire representation, for the distributed planner.
// ---------------------------------------------------------------------------

// A state's canonical bytes. Used to hash a state to a shard and to key a closed list, so two
// equal states always agree on both.
[[nodiscard]] auto encode_state(const State& state) -> std::vector<std::byte>;
[[nodiscard]] auto decode_state(std::span<const std::byte> bytes, std::size_t num_vars)
    -> Result<State>;

// A stable 64-bit hash of a state.
//
// Stable in the strong sense: it depends only on the state's values, not on the platform, the
// build or the run. A distributed search that partitioned states by an unstable hash would send
// the same state to different workers on different runs, and its results would stop being
// reproducible -- so this is part of the contract rather than an implementation detail.
[[nodiscard]] auto hash_state(const State& state) noexcept -> std::uint64_t;

}  // namespace nimblecas::planning

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::planning {

namespace {

// A fact as a single dense index, so the relaxation's tables are flat arrays rather than maps.
// `offset[v]` is where variable v's values begin.
struct FactIndex {
    std::vector<std::size_t> offset;
    std::size_t total{0};

    [[nodiscard]] auto of(std::size_t var, std::int32_t value) const noexcept -> std::size_t {
        return offset[var] + static_cast<std::size_t>(value);
    }
};

[[nodiscard]] auto build_index(const Task& task) -> FactIndex {
    FactIndex idx;
    idx.offset.reserve(task.num_vars() + 1);
    std::size_t at = 0;
    for (const std::int32_t d : task.domain_size) {
        idx.offset.push_back(at);
        at += static_cast<std::size_t>(d);
    }
    idx.offset.push_back(at);
    idx.total = at;
    return idx;
}

constexpr std::int64_t unreachable = std::numeric_limits<std::int64_t>::max();

[[nodiscard]] auto add_capped(std::int64_t a, std::int64_t b) noexcept -> std::int64_t {
    if (a == unreachable || b == unreachable) {
        return unreachable;
    }
    std::int64_t out = 0;
    if (__builtin_add_overflow(a, b, &out)) {
        return unreachable;
    }
    return out;
}

// The delete-relaxation fixpoint, shared by all three heuristics.
//
// `sum` selects h_add's aggregation over h_max's. The two differ ONLY in how an operator's
// preconditions are combined -- a maximum or a sum -- which is why one function computes both and
// they cannot drift apart.
struct Relaxation {
    std::vector<std::int64_t> cost;       // per fact
    std::vector<std::int64_t> op_cost;    // per operator: aggregated precondition cost + own cost
    std::vector<std::size_t> supporter;   // per fact: the operator that first achieved it
};

[[nodiscard]] auto relax(const Task& task, const FactIndex& idx, const State& state, bool sum)
    -> Relaxation {
    Relaxation r;
    r.cost.assign(idx.total, unreachable);
    r.op_cost.assign(task.operators.size(), unreachable);
    r.supporter.assign(idx.total, std::numeric_limits<std::size_t>::max());

    for (const std::size_t v : std::views::iota(std::size_t{0}, task.num_vars())) {
        r.cost[idx.of(v, state[v])] = 0;
    }

    // Iterate to a fixpoint. Each pass can only lower a cost, and costs are bounded below by
    // zero, so the loop terminates; the operator-count bound is a guard against a broken
    // invariant rather than the expected exit.
    bool changed = true;
    std::size_t rounds = 0;
    const std::size_t bound = task.operators.size() + idx.total + 2;
    while (changed && rounds < bound) {
        changed = false;
        ++rounds;
        for (const std::size_t o : std::views::iota(std::size_t{0}, task.operators.size())) {
            const Operator& op = task.operators[o];
            std::int64_t pre = 0;
            for (const FactPair& p : op.preconditions) {
                const std::int64_t c = r.cost[idx.of(p.var, p.value)];
                if (c == unreachable) {
                    pre = unreachable;
                    break;
                }
                pre = sum ? add_capped(pre, c) : std::max(pre, c);
            }
            if (pre == unreachable) {
                continue;
            }
            const std::int64_t applied = add_capped(pre, op.cost);
            if (applied < r.op_cost[o]) {
                r.op_cost[o] = applied;
            }
            for (const FactPair& eff : op.effects) {
                const std::size_t f = idx.of(eff.var, eff.value);
                if (applied < r.cost[f]) {
                    r.cost[f] = applied;
                    r.supporter[f] = o;
                    changed = true;
                }
            }
        }
    }
    return r;
}

[[nodiscard]] auto goal_value(const Task& task, const FactIndex& idx, const Relaxation& r, bool sum)
    -> std::optional<std::int64_t> {
    std::int64_t total = 0;
    for (const FactPair& g : task.goal) {
        const std::int64_t c = r.cost[idx.of(g.var, g.value)];
        if (c == unreachable) {
            return std::nullopt;
        }
        total = sum ? add_capped(total, c) : std::max(total, c);
    }
    if (total == unreachable) {
        return std::nullopt;
    }
    return total;
}

// A closed list keyed by the state's canonical bytes.
struct StateKeyHash {
    [[nodiscard]] auto operator()(const State& s) const noexcept -> std::size_t {
        return static_cast<std::size_t>(hash_state(s));
    }
};

// One node of a search tree, kept in a flat vector so a parent is an index rather than a pointer
// and the whole tree stays contiguous.
struct Node {
    State state;
    std::size_t parent{std::numeric_limits<std::size_t>::max()};
    std::size_t op{std::numeric_limits<std::size_t>::max()};
    std::int64_t g{0};
};

[[nodiscard]] auto extract_plan(const std::vector<Node>& nodes, std::size_t goal_node) -> Plan {
    Plan plan;
    std::size_t at = goal_node;
    while (nodes[at].parent != std::numeric_limits<std::size_t>::max()) {
        plan.steps.push_back(nodes[at].op);
        at = nodes[at].parent;
    }
    std::ranges::reverse(plan.steps);
    plan.cost = nodes[goal_node].g;
    return plan;
}

}  // namespace

// ---------------------------------------------------------------------------
// Validation and semantics.
// ---------------------------------------------------------------------------

auto validate(const Task& task) -> Result<void> {
    const std::size_t n = task.num_vars();
    if (n == 0) {
        return make_error<void>(MathError::domain_error);
    }
    for (const std::int32_t d : task.domain_size) {
        if (d <= 0) {
            return make_error<void>(MathError::domain_error);
        }
    }
    if (task.initial.size() != n) {
        return make_error<void>(MathError::domain_error);
    }
    for (const std::size_t v : std::views::iota(std::size_t{0}, n)) {
        if (task.initial[v] < 0 || task.initial[v] >= task.domain_size[v]) {
            return make_error<void>(MathError::domain_error);
        }
    }

    // A condition must be sorted and mention each variable once. Both matter: sorted so equal
    // conditions compare equal however they were built, and once so an operator cannot demand two
    // values of one variable, which no state could satisfy and which would otherwise read as a
    // merely unreachable action.
    const auto check_condition = [&](const Condition& c) -> bool {
        for (const std::size_t i : std::views::iota(std::size_t{0}, c.size())) {
            if (c[i].var >= n || c[i].value < 0 || c[i].value >= task.domain_size[c[i].var]) {
                return false;
            }
            if (i > 0 && !(c[i - 1].var < c[i].var)) {
                return false;
            }
        }
        return true;
    };
    if (!check_condition(task.goal)) {
        return make_error<void>(MathError::domain_error);
    }
    for (const Operator& op : task.operators) {
        if (op.cost < 0 || !check_condition(op.preconditions) || !check_condition(op.effects)) {
            return make_error<void>(MathError::domain_error);
        }
        if (op.effects.empty()) {
            return make_error<void>(MathError::domain_error);
        }
    }
    return {};
}

auto holds(const State& state, const Condition& condition) -> bool {
    return std::ranges::all_of(condition, [&state](const FactPair& p) {
        return p.var < state.size() && state[p.var] == p.value;
    });
}

auto applicable(const Task& task, const State& state, std::size_t op_index) -> bool {
    if (op_index >= task.operators.size()) {
        return false;
    }
    return holds(state, task.operators[op_index].preconditions);
}

auto apply(const Task& task, const State& state, std::size_t op_index) -> Result<State> {
    if (!applicable(task, state, op_index)) {
        return make_error<State>(MathError::domain_error);
    }
    State out = state;
    for (const FactPair& eff : task.operators[op_index].effects) {
        out[eff.var] = eff.value;
    }
    return out;
}

auto validate_plan(const Task& task, const Plan& plan) -> Result<State> {
    auto ok = validate(task);
    if (!ok) {
        return make_error<State>(ok.error());
    }
    State state = task.initial;
    std::int64_t cost = 0;
    for (const std::size_t step : plan.steps) {
        auto next = apply(task, state, step);
        if (!next) {
            return make_error<State>(next.error());
        }
        cost = add_capped(cost, task.operators[step].cost);
        state = std::move(*next);
    }
    if (cost != plan.cost) {
        // The recorded cost disagrees with the plan's own steps, which means the search's
        // bookkeeping is wrong. Better an error than a plan whose stated cost is fiction.
        return make_error<State>(MathError::domain_error);
    }
    if (!holds(state, task.goal)) {
        return make_error<State>(MathError::undefined_value);
    }
    return state;
}

// ---------------------------------------------------------------------------
// Heuristics.
// ---------------------------------------------------------------------------

auto h_max(const Task& task, const State& state) -> std::optional<std::int64_t> {
    const FactIndex idx = build_index(task);
    const Relaxation r = relax(task, idx, state, false);
    return goal_value(task, idx, r, false);
}

auto h_add(const Task& task, const State& state) -> std::optional<std::int64_t> {
    const FactIndex idx = build_index(task);
    const Relaxation r = relax(task, idx, state, true);
    return goal_value(task, idx, r, true);
}

auto h_ff(const Task& task, const State& state) -> FfResult {
    FfResult out;
    const FactIndex idx = build_index(task);
    // The relaxed plan is extracted from the h_add fixpoint, which is Hoffmann's construction:
    // the additive costs order the subgoals sensibly, and the extraction then counts each chosen
    // operator ONCE however many subgoals it serves -- which is precisely the double counting
    // h_add suffers and h_ff does not.
    const Relaxation r = relax(task, idx, state, true);
    if (!goal_value(task, idx, r, true).has_value()) {
        return out;  // unreachable under the relaxation, so unreachable in truth
    }

    std::vector<bool> chosen(task.operators.size(), false);
    std::vector<bool> done(idx.total, false);
    std::vector<std::size_t> open;
    for (const FactPair& g : task.goal) {
        open.push_back(idx.of(g.var, g.value));
    }
    for (const std::size_t v : std::views::iota(std::size_t{0}, task.num_vars())) {
        done[idx.of(v, state[v])] = true;  // already true in the state; nothing to achieve
    }

    while (!open.empty()) {
        const std::size_t fact = open.back();
        open.pop_back();
        if (done[fact]) {
            continue;
        }
        done[fact] = true;
        const std::size_t o = r.supporter[fact];
        if (o == std::numeric_limits<std::size_t>::max()) {
            continue;  // true in the state already
        }
        if (chosen[o]) {
            continue;  // counted once, however many facts it supports -- the point of FF
        }
        chosen[o] = true;
        for (const FactPair& p : task.operators[o].preconditions) {
            const std::size_t pf = idx.of(p.var, p.value);
            if (!done[pf]) {
                open.push_back(pf);
            }
        }
    }

    std::int64_t total = 0;
    for (const std::size_t o : std::views::iota(std::size_t{0}, task.operators.size())) {
        if (chosen[o]) {
            total = add_capped(total, task.operators[o].cost);
            // A chosen operator already applicable in this state is a HELPFUL ACTION: the relaxed
            // plan says it belongs in the solution and nothing stands in its way now.
            if (holds(state, task.operators[o].preconditions)) {
                out.preferred.push_back(o);
            }
        }
    }
    out.value = total;
    return out;
}

// ---------------------------------------------------------------------------
// Search.
// ---------------------------------------------------------------------------

auto astar_plan(const Task& task, std::uint64_t max_expansions) -> Result<PlanResult> {
    auto ok = validate(task);
    if (!ok) {
        return make_error<PlanResult>(ok.error());
    }

    std::vector<Node> nodes;
    std::unordered_map<State, std::size_t, StateKeyHash> seen;
    // (f, h, insertion order, node) -- h before insertion order so that among equal f the node
    // closer to the goal is expanded first, which is the standard tie-break and measurably better
    // than breaking on discovery order alone.
    using Entry = std::tuple<std::int64_t, std::int64_t, std::uint64_t, std::size_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;

    SearchStats stats;
    const auto h0 = h_max(task, task.initial);
    ++stats.evaluated;
    if (!h0.has_value()) {
        return make_error<PlanResult>(MathError::undefined_value);
    }
    nodes.push_back(Node{.state = task.initial, .parent = std::numeric_limits<std::size_t>::max(),
                         .op = std::numeric_limits<std::size_t>::max(), .g = 0});
    seen.emplace(task.initial, std::size_t{0});
    std::uint64_t order = 0;
    open.push(Entry{*h0, *h0, order++, std::size_t{0}});

    while (!open.empty()) {
        const auto [f, h, ins, id] = open.top();
        open.pop();
        if (f > add_capped(nodes[id].g, h)) {
            continue;  // stale entry, superseded by a cheaper route to the same state
        }
        if (holds(nodes[id].state, task.goal)) {
            Plan plan = extract_plan(nodes, id);
            auto check = validate_plan(task, plan);
            if (!check) {
                return make_error<PlanResult>(check.error());
            }
            return PlanResult{.plan = std::move(plan), .stats = stats};
        }
        if (stats.expanded >= max_expansions) {
            return make_error<PlanResult>(MathError::not_converged);
        }
        ++stats.expanded;

        for (const std::size_t o : std::views::iota(std::size_t{0}, task.operators.size())) {
            if (!holds(nodes[id].state, task.operators[o].preconditions)) {
                continue;
            }
            State next = nodes[id].state;
            for (const FactPair& eff : task.operators[o].effects) {
                next[eff.var] = eff.value;
            }
            const std::int64_t g = add_capped(nodes[id].g, task.operators[o].cost);
            ++stats.generated;
            const auto it = seen.find(next);
            if (it != seen.end() && nodes[it->second].g <= g) {
                continue;
            }
            const auto hn = h_max(task, next);
            ++stats.evaluated;
            if (!hn.has_value()) {
                continue;  // provably cannot reach the goal from there
            }
            if (it != seen.end()) {
                nodes[it->second].g = g;
                nodes[it->second].parent = id;
                nodes[it->second].op = o;
                open.push(Entry{add_capped(g, *hn), *hn, order++, it->second});
            } else {
                nodes.push_back(Node{.state = next, .parent = id, .op = o, .g = g});
                const std::size_t nid = nodes.size() - 1;
                seen.emplace(next, nid);
                open.push(Entry{add_capped(g, *hn), *hn, order++, nid});
            }
        }
    }
    return make_error<PlanResult>(MathError::undefined_value);
}

namespace {

// The shared body of the two greedy searches. `use_preferred` switches the second open list on;
// `use_ff` picks h_ff over h_add.
[[nodiscard]] auto greedy(const Task& task, std::uint64_t max_expansions, bool use_ff,
                          bool use_preferred) -> Result<PlanResult> {
    auto ok = validate(task);
    if (!ok) {
        return make_error<PlanResult>(ok.error());
    }

    std::vector<Node> nodes;
    std::unordered_map<State, std::size_t, StateKeyHash> seen;
    using Entry = std::tuple<std::int64_t, std::uint64_t, std::size_t>;  // (h, order, node)
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> plain;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> preferred;

    SearchStats stats;
    const auto evaluate = [&](const State& s) -> std::pair<std::optional<std::int64_t>,
                                                           std::vector<std::size_t>> {
        ++stats.evaluated;
        if (use_ff) {
            FfResult r = h_ff(task, s);
            return {r.value, std::move(r.preferred)};
        }
        return {h_add(task, s), std::vector<std::size_t>{}};
    };

    auto [h0, pref0] = evaluate(task.initial);
    if (!h0.has_value()) {
        return make_error<PlanResult>(MathError::undefined_value);
    }
    nodes.push_back(Node{.state = task.initial, .parent = std::numeric_limits<std::size_t>::max(),
                         .op = std::numeric_limits<std::size_t>::max(), .g = 0});
    seen.emplace(task.initial, std::size_t{0});
    std::uint64_t order = 0;
    plain.push(Entry{*h0, order++, std::size_t{0}});

    // Alternation, not preference. Taking only from the preferred list is faster when the
    // heuristic points the right way and never recovers when it does not; alternating keeps the
    // speed and keeps the completeness of the ordinary list underneath it.
    bool take_preferred = false;
    while (!plain.empty() || !preferred.empty()) {
        take_preferred = use_preferred && !take_preferred;
        auto& queue = (take_preferred && !preferred.empty()) ? preferred : plain;
        if (queue.empty()) {
            continue;
        }
        const auto [h, ins, id] = queue.top();
        queue.pop();

        if (holds(nodes[id].state, task.goal)) {
            Plan plan = extract_plan(nodes, id);
            auto check = validate_plan(task, plan);
            if (!check) {
                return make_error<PlanResult>(check.error());
            }
            return PlanResult{.plan = std::move(plan), .stats = stats};
        }
        if (stats.expanded >= max_expansions) {
            return make_error<PlanResult>(MathError::not_converged);
        }
        ++stats.expanded;

        const auto [hv, pref] = evaluate(nodes[id].state);
        std::vector<bool> is_preferred(task.operators.size(), false);
        for (const std::size_t o : pref) {
            is_preferred[o] = true;
        }

        for (const std::size_t o : std::views::iota(std::size_t{0}, task.operators.size())) {
            if (!holds(nodes[id].state, task.operators[o].preconditions)) {
                continue;
            }
            State next = nodes[id].state;
            for (const FactPair& eff : task.operators[o].effects) {
                next[eff.var] = eff.value;
            }
            ++stats.generated;
            if (seen.contains(next)) {
                continue;  // greedy search closes a state on first sight
            }
            auto [hn, ignored] = evaluate(next);
            if (!hn.has_value()) {
                continue;
            }
            const std::int64_t g = add_capped(nodes[id].g, task.operators[o].cost);
            nodes.push_back(Node{.state = next, .parent = id, .op = o, .g = g});
            const std::size_t nid = nodes.size() - 1;
            seen.emplace(std::move(next), nid);
            plain.push(Entry{*hn, order, nid});
            if (use_preferred && is_preferred[o]) {
                preferred.push(Entry{*hn, order, nid});
            }
            ++order;
        }
    }
    return make_error<PlanResult>(MathError::undefined_value);
}

}  // namespace

auto gbfs_plan(const Task& task, std::uint64_t max_expansions) -> Result<PlanResult> {
    return greedy(task, max_expansions, true, true);
}

auto gbfs_plan_hadd(const Task& task, std::uint64_t max_expansions) -> Result<PlanResult> {
    return greedy(task, max_expansions, false, false);
}

// ---------------------------------------------------------------------------
// Wire representation.
// ---------------------------------------------------------------------------

auto encode_state(const State& state) -> std::vector<std::byte> {
    std::vector<std::byte> out;
    out.reserve(state.size() * 4);
    for (const std::int32_t v : state) {
        const auto u = static_cast<std::uint32_t>(v);
        for (const unsigned int i : std::views::iota(0U, 4U)) {
            out.push_back(static_cast<std::byte>((u >> (8U * i)) & 0xFFU));
        }
    }
    return out;
}

auto decode_state(std::span<const std::byte> bytes, std::size_t num_vars) -> Result<State> {
    if (bytes.size() != num_vars * 4) {
        return make_error<State>(MathError::syntax_error);
    }
    State out(num_vars, 0);
    for (const std::size_t v : std::views::iota(std::size_t{0}, num_vars)) {
        std::uint32_t u = 0;
        for (const std::size_t i : std::views::iota(std::size_t{0}, std::size_t{4})) {
            const auto byte = std::to_integer<unsigned char>(bytes[v * 4 + i]);
            u |= static_cast<std::uint32_t>(byte) << (8U * i);
        }
        out[v] = static_cast<std::int32_t>(u);
    }
    return out;
}

auto hash_state(const State& state) noexcept -> std::uint64_t {
    // FNV-1a over the state's bytes, written out rather than taken from std::hash: a distributed
    // search partitions states by this value, and std::hash is allowed to differ between
    // implementations and even between runs. An unstable hash would send the same state to
    // different workers on different runs and quietly cost reproducibility.
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    constexpr std::uint64_t prime = 0x00000100000001b3ULL;
    for (const std::int32_t v : state) {
        const auto u = static_cast<std::uint32_t>(v);
        for (const unsigned int i : std::views::iota(0U, 4U)) {
            hash ^= static_cast<std::uint64_t>((u >> (8U * i)) & 0xFFU);
            hash *= prime;
        }
    }
    return hash;
}

}  // namespace nimblecas::planning
