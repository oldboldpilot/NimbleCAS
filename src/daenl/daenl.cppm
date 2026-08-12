// NimbleCAS nonlinear/higher-index differential-algebraic equations (DAE).
// @author Olumuyiwa Oluwasanmi
//
// Companion to nimblecas.dae (which is scoped to LINEAR constant-coefficient DAEs).
// This module handles NONLINEAR F(t, x, x') = 0 systems, in three tiers of decreasing
// exactness:
//
//   1. solve_semiexplicit_nonlinear_series — EXACT power series over Q for the
//      semi-explicit form x' = f(x,y,t), 0 = g(x,y,t). RESTRICTED (honestly, not
//      silently): dg/dy must be a CONSTANT rational matrix (i.e. g is affine in y with
//      x/t-independent coefficients — g = A y + h(x,t)); f, g may only be built from
//      +, *, integer powers (positive or negative), exp, and ln. Anything outside that
//      (a non-constant dg/dy, a fractional/symbolic exponent, sin/cos/..., a floating
//      literal) is rejected with MathError::not_implemented rather than guessed at.
//      Under those conditions the reduction y' = -A^{-1}(dg/dx f + dg/dt) is exact, and
//      the result is verified by checking g == 0 to the requested truncation order.
//
//   2. analyze_structure — a structural (Pantelides-style) index/consistency analysis:
//      symbolically differentiate the whole residual system, sweep by sweep, until the
//      bipartite incidence between equations and derivative symbols admits a perfect
//      matching (Kuhn's algorithm), or max_index sweeps are exhausted
//      (MathError::not_implemented). This is a SIMPLER, non-minimal variant of
//      Pantelides (it differentiates the whole most-recent batch rather than selecting
//      the minimal unmatched subset), so it can over-differentiate relative to the
//      textbook algorithm, but every equation/symbol relation it reports is exact
//      symbolic differentiation, and it never claims an index it has not verified by
//      matching.
//
//   3. consistent_initial_values / solve_nonlinear_dae — NUMERICAL. Given a candidate
//      state x (held fixed) and a derivative guess, consistent_initial_values solves
//      the augmented (index-reduced) system from analyze_structure for the derivative
//      levels via Newton (nlsolve), so xdot (and the internal higher derivatives used
//      only to satisfy the chain rule) is exact only in the sense of solving that
//      nonlinear system to nlsolve's tolerance — non-convergence is reported as
//      MathError::not_converged, never as a silently-approximate answer.
//      solve_nonlinear_dae time-steps an implicit Euler / BDF-2 scheme on the ORIGINAL
//      residuals; it is HONEST about its own limits and refuses (not_implemented) any
//      system whose structural_index != 1, because reliably stabilising the trajectory
//      of a higher-index system needs projection/dummy-derivative machinery this module
//      does not implement. Every Newton non-convergence, or post-step residual drift
//      beyond opts.constraint_tol, is reported as MathError::not_converged rather than
//      accepted as a state.

export module nimblecas.daenl;

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.simplify;
import nimblecas.diff;
import nimblecas.evalnum;
import nimblecas.nlsolve;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.powerseries;
import nimblecas.ode;
import nimblecas.dae;

export namespace nimblecas::daenl {

// A nonlinear DAE residual system F(t, x, x') = 0: residuals[i] is an Expr over the
// symbols vars (the state x) and ders (the corresponding x', matched positionally:
// ders[j] is the symbol standing for d(vars[j])/dt) plus the independent variable
// `time`. residuals, vars, and ders must have equal length.
struct DaeSystem {
    std::vector<Expr> residuals;
    std::vector<std::string> vars;
    std::vector<std::string> ders;
    std::string time{"t"};
};

// Result of the structural sweep: structural_index is (differentiation sweeps + 1);
// diff_count[j] is how many times original equation j was differentiated;
// augmented_residuals is the full accumulated equation set (original + all
// differentiated batches) whose incidence against the derivative-symbol columns
// admitted a perfect matching.
struct StructuralInfo {
    std::size_t structural_index{0};
    std::vector<std::size_t> diff_count;
    std::vector<Expr> augmented_residuals;
};

[[nodiscard]] auto analyze_structure(const DaeSystem& sys, std::size_t max_index = 3)
    -> Result<StructuralInfo>;

struct NlDaeOptions {
    std::size_t max_index{3};
    std::size_t bdf_order{2};
    double constraint_tol{1e-8};
    nlsolve::Options newton{};
};

struct NlDaeSolution {
    std::vector<double> times;
    std::vector<std::vector<double>> states;  // states[k] is x at times[k] (length n)
    std::size_t structural_index{0};
    bool initial_guess_consistent{false};
};

// Fix x = x_guess and solve for a consistent derivative (and, internally, any higher
// derivative levels the structural analysis needs) via Newton on the index-reduced
// system. Returns (x_guess unchanged, the corrected xdot).
[[nodiscard]] auto consistent_initial_values(const DaeSystem& sys,
                                             std::span<const double> x_guess,
                                             std::span<const double> xdot_guess, double t0,
                                             const NlDaeOptions& opts = {})
    -> Result<std::pair<std::vector<double>, std::vector<double>>>;

// Time-step F(t, x, x') = 0 from t0 to t1 over `steps` equal steps by implicit Euler
// (bdf_order == 1) or BDF-2 (bdf_order == 2, falling back to Euler for the first step).
// Only structural_index == 1 systems are integrated; anything else is
// MathError::not_implemented (see module header).
[[nodiscard]] auto solve_nonlinear_dae(const DaeSystem& sys, std::span<const double> x0_guess,
                                       std::span<const double> xdot0_guess, double t0,
                                       double t1, std::uint64_t steps,
                                       const NlDaeOptions& opts = {}) -> Result<NlDaeSolution>;

// EXACT power-series solution of the semi-explicit nonlinear DAE x' = f(x,y,t),
// 0 = g(x,y,t), x(0) = x0, y(0) = y0. See the module header for the (honestly stated)
// restrictions on f and g.
[[nodiscard]] auto solve_semiexplicit_nonlinear_series(
    const std::vector<Expr>& f, const std::vector<Expr>& g, const std::vector<std::string>& xvars,
    const std::vector<std::string>& yvars, std::string_view time, const std::vector<Rational>& x0,
    const std::vector<Rational>& y0, std::size_t order) -> Result<DaeSolution>;

}  // namespace nimblecas::daenl

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::daenl {
namespace {

template <typename...>
inline constexpr bool always_false = false;

// True iff `e` contains the symbol named `name` anywhere in its tree.
[[nodiscard]] auto contains(const Expr& e, const std::string& name) -> bool {
    return !free_of(e, Expr::symbol(name));
}

// The symbol standing for the L-th time-derivative of state variable j: level 0 is the
// state itself (vars[j]), level 1 is the supplied derivative symbol (ders[j]), and each
// higher level appends a prime to the level-1 name (a fresh symbol introduced by the
// structural sweep).
[[nodiscard]] auto level_symbol(const DaeSystem& sys, std::size_t j, std::size_t level)
    -> std::string {
    if (level == 0) {
        return sys.vars[j];
    }
    if (level == 1) {
        return sys.ders[j];
    }
    return sys.ders[j] + std::string(level - 1, '\'');
}

// Total time derivative of `e` by the multivariable chain rule, treating every
// level-L symbol's own time-derivative as the level-(L+1) symbol (levels 0..up_to_level
// are assumed to be the ones currently present in `e`). Explicit dependence on
// sys.time is included via a direct partial derivative.
[[nodiscard]] auto total_time_derivative(const DaeSystem& sys, const Expr& e,
                                         std::size_t up_to_level) -> Result<Expr> {
    auto dt = differentiate(e, sys.time);
    if (!dt) {
        return make_error<Expr>(dt.error());
    }
    std::vector<Expr> terms;
    terms.reserve(1 + sys.vars.size() * (up_to_level + 1));
    terms.push_back(std::move(*dt));
    for (std::size_t j = 0; j < sys.vars.size(); ++j) {
        for (std::size_t level = 0; level <= up_to_level; ++level) {
            auto d = differentiate(e, level_symbol(sys, j, level));
            if (!d) {
                return make_error<Expr>(d.error());
            }
            terms.push_back(
                Expr::product({std::move(*d), Expr::symbol(level_symbol(sys, j, level + 1))}));
        }
    }
    return simplify(Expr::sum(std::move(terms)));
}

// Kuhn's augmenting-path bipartite matching: adj[i] lists the right-side indices
// incident to left index i. Returns the size of a maximum matching.
[[nodiscard]] auto max_matching(const std::vector<std::vector<std::size_t>>& adj,
                                std::size_t num_right) -> std::size_t {
    std::vector<std::int64_t> match_right(num_right, -1);
    std::vector<bool> visited;
    std::function<bool(std::size_t)> try_augment = [&](std::size_t u) -> bool {
        for (std::size_t v : adj[u]) {
            if (visited[v]) {
                continue;
            }
            visited[v] = true;
            if (match_right[v] == -1 ||
                try_augment(static_cast<std::size_t>(match_right[v]))) {
                match_right[v] = static_cast<std::int64_t>(u);
                return true;
            }
        }
        return false;
    };
    std::size_t matched = 0;
    for (std::size_t i = 0; i < adj.size(); ++i) {
        visited.assign(num_right, false);
        if (try_augment(i)) {
            ++matched;
        }
    }
    return matched;
}

// Extract an EXACT rational value from a constant Expr (integer or rational leaf); a
// double leaf, or a non-constant Expr, is honestly not_implemented (Tier 1 is exact).
[[nodiscard]] auto expr_to_exact_rational(const Expr& e) -> Result<Rational> {
    auto cn = as<ConstantNode>(e.node().value);
    if (!cn) {
        return make_error<Rational>(MathError::not_implemented);
    }
    if (auto iv = std::get_if<std::int64_t>(&(*cn)->value)) {
        return Rational::from_int(*iv);
    }
    if (auto pv = std::get_if<std::pair<std::int64_t, std::int64_t>>(&(*cn)->value)) {
        return Rational::make(pv->first, pv->second);
    }
    return make_error<Rational>(MathError::not_implemented);  // double leaf: not exact
}

// Evaluate `e` to a PowerSeries by substituting the bound symbols; see the module
// header for the supported subset (+, *, integer powers, exp, ln).
[[nodiscard]] auto eval_expr_series(const Expr& e,
                                    const std::unordered_map<std::string, PowerSeries>& b,
                                    std::size_t order) -> Result<PowerSeries> {
    return std::visit(
        [&]<typename T>(const T& n) -> Result<PowerSeries> {
            if constexpr (std::is_same_v<T, ConstantNode>) {
                if (auto iv = std::get_if<std::int64_t>(&n.value)) {
                    return PowerSeries::constant(Rational::from_int(*iv), order);
                }
                if (auto pv = std::get_if<std::pair<std::int64_t, std::int64_t>>(&n.value)) {
                    auto r = Rational::make(pv->first, pv->second);
                    if (!r) {
                        return make_error<PowerSeries>(r.error());
                    }
                    return PowerSeries::constant(*r, order);
                }
                return make_error<PowerSeries>(MathError::not_implemented);
            } else if constexpr (std::is_same_v<T, SymbolNode>) {
                auto it = b.find(n.name);
                if (it == b.end()) {
                    return make_error<PowerSeries>(MathError::domain_error);
                }
                return it->second;
            } else if constexpr (std::is_same_v<T, AddNode>) {
                auto acc = PowerSeries::zero(order);
                if (!acc) {
                    return acc;
                }
                PowerSeries s = std::move(*acc);
                for (const Expr& t : n.terms) {
                    auto v = eval_expr_series(t, b, order);
                    if (!v) {
                        return v;
                    }
                    auto sum = s.add(*v);
                    if (!sum) {
                        return make_error<PowerSeries>(sum.error());
                    }
                    s = std::move(*sum);
                }
                return s;
            } else if constexpr (std::is_same_v<T, MulNode>) {
                auto acc = PowerSeries::one(order);
                if (!acc) {
                    return acc;
                }
                PowerSeries s = std::move(*acc);
                for (const Expr& t : n.factors) {
                    auto v = eval_expr_series(t, b, order);
                    if (!v) {
                        return v;
                    }
                    auto prod = s.multiply(*v);
                    if (!prod) {
                        return make_error<PowerSeries>(prod.error());
                    }
                    s = std::move(*prod);
                }
                return s;
            } else if constexpr (std::is_same_v<T, PowerNode>) {
                auto base_v = eval_expr_series(n.base, b, order);
                if (!base_v) {
                    return base_v;
                }
                auto exp_c = as<ConstantNode>(n.exponent.node().value);
                if (!exp_c) {
                    return make_error<PowerSeries>(MathError::not_implemented);
                }
                auto iv = std::get_if<std::int64_t>(&(*exp_c)->value);
                if (!iv) {
                    return make_error<PowerSeries>(MathError::not_implemented);
                }
                const std::int64_t p = *iv;
                PowerSeries base_series = std::move(*base_v);
                if (p < 0) {
                    auto inv = base_series.inverse();
                    if (!inv) {
                        return make_error<PowerSeries>(inv.error());
                    }
                    base_series = std::move(*inv);
                }
                const std::int64_t reps = p < 0 ? -p : p;
                auto acc = PowerSeries::one(order);
                if (!acc) {
                    return acc;
                }
                PowerSeries s = std::move(*acc);
                for (std::int64_t c = 0; c < reps; ++c) {
                    auto pr = s.multiply(base_series);
                    if (!pr) {
                        return make_error<PowerSeries>(pr.error());
                    }
                    s = std::move(*pr);
                }
                return s;
            } else if constexpr (std::is_same_v<T, FunctionNode>) {
                if (n.args.size() != 1) {
                    return make_error<PowerSeries>(MathError::not_implemented);
                }
                auto arg = eval_expr_series(n.args.front(), b, order);
                if (!arg) {
                    return arg;
                }
                if (n.name == "exp") {
                    return arg->exp();
                }
                if (n.name == "ln") {
                    return arg->log();
                }
                return make_error<PowerSeries>(MathError::not_implemented);
            } else {
                static_assert(always_false<T>, "eval_expr_series: unhandled ExprNode kind");
            }
        },
        e.node().value);
}

[[nodiscard]] auto all_finite(std::span<const double> v) -> bool {
    return std::ranges::all_of(v, [](double x) { return std::isfinite(x); });
}

}  // namespace

auto analyze_structure(const DaeSystem& sys, std::size_t max_index) -> Result<StructuralInfo> {
    const std::size_t n = sys.vars.size();
    if (n == 0 || sys.residuals.size() != n || sys.ders.size() != n || max_index == 0) {
        return make_error<StructuralInfo>(MathError::domain_error);
    }
    std::vector<Expr> eqs = sys.residuals;
    std::vector<Expr> last_batch = sys.residuals;
    std::vector<std::string> cols;
    cols.reserve(n);
    for (std::size_t j = 0; j < n; ++j) {
        cols.push_back(sys.ders[j]);
    }
    std::size_t sweep = 0;
    std::size_t up_to_level = 1;
    while (true) {
        std::vector<std::vector<std::size_t>> adj(eqs.size());
        for (std::size_t i = 0; i < eqs.size(); ++i) {
            for (std::size_t k = 0; k < cols.size(); ++k) {
                if (contains(eqs[i], cols[k])) {
                    adj[i].push_back(k);
                }
            }
        }
        if (max_matching(adj, cols.size()) == cols.size()) {
            return StructuralInfo{sweep + 1, std::vector<std::size_t>(n, sweep), std::move(eqs)};
        }
        if (sweep >= max_index) {
            return make_error<StructuralInfo>(MathError::not_implemented);
        }
        std::vector<Expr> new_batch;
        new_batch.reserve(n);
        for (const Expr& e : last_batch) {
            auto d = total_time_derivative(sys, e, up_to_level);
            if (!d) {
                return make_error<StructuralInfo>(d.error());
            }
            new_batch.push_back(std::move(*d));
        }
        eqs.insert(eqs.end(), new_batch.begin(), new_batch.end());
        for (std::size_t j = 0; j < n; ++j) {
            cols.push_back(level_symbol(sys, j, up_to_level + 1));
        }
        last_batch = std::move(new_batch);
        ++up_to_level;
        ++sweep;
    }
}

auto consistent_initial_values(const DaeSystem& sys, std::span<const double> x_guess,
                               std::span<const double> xdot_guess, double t0,
                               const NlDaeOptions& opts)
    -> Result<std::pair<std::vector<double>, std::vector<double>>> {
    using Pair = std::pair<std::vector<double>, std::vector<double>>;
    const std::size_t n = sys.vars.size();
    if (n == 0 || sys.ders.size() != n || sys.residuals.size() != n || x_guess.size() != n ||
        xdot_guess.size() != n || !std::isfinite(t0) || !all_finite(x_guess) ||
        !all_finite(xdot_guess)) {
        return make_error<Pair>(MathError::domain_error);
    }

    auto si = analyze_structure(sys, opts.max_index);
    if (!si) {
        return make_error<Pair>(si.error());
    }
    const std::size_t idx = si->structural_index;
    const std::size_t m = n * idx;
    const std::vector<Expr>& eqs = si->augmented_residuals;

    std::vector<std::string> unk;
    unk.reserve(m);
    for (std::size_t level = 1; level <= idx; ++level) {
        for (std::size_t j = 0; j < n; ++j) {
            unk.push_back(level_symbol(sys, j, level));
        }
    }

    std::vector<Expr> jac;
    jac.reserve(m * m);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t c = 0; c < m; ++c) {
            auto d = differentiate(eqs[i], unk[c]);
            if (!d) {
                return make_error<Pair>(d.error());
            }
            jac.push_back(std::move(*d));
        }
    }

    std::vector<double> x_fixed(x_guess.begin(), x_guess.end());
    auto make_bindings = [&](std::span<const double> z) {
        std::unordered_map<std::string, double> b;
        b.reserve(n + m + 1);
        b[sys.time] = t0;
        for (std::size_t j = 0; j < n; ++j) {
            b[sys.vars[j]] = x_fixed[j];
        }
        for (std::size_t c = 0; c < m; ++c) {
            b[unk[c]] = z[c];
        }
        return b;
    };
    nlsolve::ResidualFn F = [&](std::span<const double> z) -> std::vector<double> {
        auto b = make_bindings(z);
        std::vector<double> out;
        out.reserve(m);
        for (const Expr& e : eqs) {
            auto v = eval_double(e, b);
            out.push_back(v ? *v : std::numeric_limits<double>::quiet_NaN());
        }
        return out;
    };
    nlsolve::JacobianFn J = [&](std::span<const double> z) -> std::vector<double> {
        auto b = make_bindings(z);
        std::vector<double> out(m * m);
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t c = 0; c < m; ++c) {
                auto v = eval_double(jac[i * m + c], b);
                out[i * m + c] = v ? *v : std::numeric_limits<double>::quiet_NaN();
            }
        }
        return out;
    };

    std::vector<double> z0(m, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        z0[j] = xdot_guess[j];
    }
    auto res = nlsolve::newton(F, J, std::span<const double>{z0}, opts.newton);
    if (!res) {
        return make_error<Pair>(res.error());
    }
    if (!res->converged) {
        return make_error<Pair>(MathError::not_converged);
    }
    std::vector<double> xdot(res->x.begin(), res->x.begin() + static_cast<std::ptrdiff_t>(n));
    return Pair{std::move(x_fixed), std::move(xdot)};
}

auto solve_nonlinear_dae(const DaeSystem& sys, std::span<const double> x0_guess,
                         std::span<const double> xdot0_guess, double t0, double t1,
                         std::uint64_t steps, const NlDaeOptions& opts) -> Result<NlDaeSolution> {
    const std::size_t n = sys.vars.size();
    if (n == 0 || sys.ders.size() != n || sys.residuals.size() != n || x0_guess.size() != n ||
        xdot0_guess.size() != n || steps == 0 || !std::isfinite(t0) || !std::isfinite(t1) ||
        !(t1 > t0) || (opts.bdf_order != 1 && opts.bdf_order != 2)) {
        return make_error<NlDaeSolution>(MathError::domain_error);
    }

    auto civ = consistent_initial_values(sys, x0_guess, xdot0_guess, t0, opts);
    if (!civ) {
        return make_error<NlDaeSolution>(civ.error());
    }
    auto si = analyze_structure(sys, opts.max_index);
    if (!si) {
        return make_error<NlDaeSolution>(si.error());
    }
    if (si->structural_index != 1) {
        // Honest boundary: reliable time-stepping of a higher-index trajectory needs
        // projection/dummy-derivative machinery this module does not implement.
        return make_error<NlDaeSolution>(MathError::not_implemented);
    }

    std::vector<Expr> dFdx, dFdxdot;
    dFdx.reserve(n * n);
    dFdxdot.reserve(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            auto a = differentiate(sys.residuals[i], sys.vars[j]);
            if (!a) {
                return make_error<NlDaeSolution>(a.error());
            }
            dFdx.push_back(std::move(*a));
            auto c = differentiate(sys.residuals[i], sys.ders[j]);
            if (!c) {
                return make_error<NlDaeSolution>(c.error());
            }
            dFdxdot.push_back(std::move(*c));
        }
    }

    const double dt = (t1 - t0) / static_cast<double>(steps);
    std::vector<double> x_prev2;
    std::vector<double> x_prev = civ->first;

    NlDaeSolution out;
    out.structural_index = si->structural_index;
    out.initial_guess_consistent = true;
    out.times.reserve(steps + 1);
    out.states.reserve(steps + 1);
    out.times.push_back(t0);
    out.states.push_back(x_prev);

    for (std::uint64_t k = 0; k < steps; ++k) {
        const double t_next = t0 + dt * static_cast<double>(k + 1);
        const bool use_bdf2 = opts.bdf_order == 2 && k >= 1;
        const double alpha_over_dt = (use_bdf2 ? 1.5 : 1.0) / dt;

        auto make_bindings = [&](std::span<const double> z) {
            std::unordered_map<std::string, double> b;
            b.reserve(2 * n + 1);
            b[sys.time] = t_next;
            for (std::size_t j = 0; j < n; ++j) {
                b[sys.vars[j]] = z[j];
                b[sys.ders[j]] = use_bdf2
                                     ? (1.5 * z[j] - 2.0 * x_prev[j] + 0.5 * x_prev2[j]) / dt
                                     : (z[j] - x_prev[j]) / dt;
            }
            return b;
        };
        nlsolve::ResidualFn F = [&](std::span<const double> z) -> std::vector<double> {
            auto b = make_bindings(z);
            std::vector<double> res_out;
            res_out.reserve(n);
            for (const Expr& e : sys.residuals) {
                auto v = eval_double(e, b);
                res_out.push_back(v ? *v : std::numeric_limits<double>::quiet_NaN());
            }
            return res_out;
        };
        nlsolve::JacobianFn J = [&](std::span<const double> z) -> std::vector<double> {
            auto b = make_bindings(z);
            std::vector<double> jac_out(n * n);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    auto a = eval_double(dFdx[i * n + j], b);
                    auto c = eval_double(dFdxdot[i * n + j], b);
                    jac_out[i * n + j] = (a && c) ? (*a + alpha_over_dt * (*c))
                                                   : std::numeric_limits<double>::quiet_NaN();
                }
            }
            return jac_out;
        };

        auto res = nlsolve::newton(F, J, std::span<const double>{x_prev}, opts.newton);
        if (!res) {
            return make_error<NlDaeSolution>(res.error());
        }
        if (!res->converged || res->residual_norm > opts.constraint_tol) {
            return make_error<NlDaeSolution>(MathError::not_converged);
        }
        x_prev2 = std::move(x_prev);
        x_prev = res->x;
        out.times.push_back(t_next);
        out.states.push_back(x_prev);
    }
    return out;
}

auto solve_semiexplicit_nonlinear_series(const std::vector<Expr>& f, const std::vector<Expr>& g,
                                         const std::vector<std::string>& xvars,
                                         const std::vector<std::string>& yvars,
                                         std::string_view time, const std::vector<Rational>& x0,
                                         const std::vector<Rational>& y0, std::size_t order)
    -> Result<DaeSolution> {
    const std::size_t nx = xvars.size();
    const std::size_t ny = yvars.size();
    if (order == 0 || nx == 0 || ny == 0 || f.size() != nx || g.size() != ny ||
        x0.size() != nx || y0.size() != ny) {
        return make_error<DaeSolution>(MathError::domain_error);
    }

    // dg/dy must be a CONSTANT rational matrix (Tier-1 restriction, see module header).
    std::vector<std::vector<Rational>> a_rows(ny, std::vector<Rational>(ny));
    for (std::size_t i = 0; i < ny; ++i) {
        for (std::size_t k = 0; k < ny; ++k) {
            auto d = differentiate(g[i], yvars[k]);
            if (!d) {
                return make_error<DaeSolution>(d.error());
            }
            auto val = expr_to_exact_rational(*d);
            if (!val) {
                return make_error<DaeSolution>(val.error());
            }
            a_rows[i][k] = *val;
        }
    }
    auto a_mat = Matrix::from_rows(a_rows);
    if (!a_mat) {
        return make_error<DaeSolution>(a_mat.error());
    }
    auto ainv = a_mat->inverse();
    if (!ainv) {
        return make_error<DaeSolution>(ainv.error());
    }

    // Expr has no default constructor, so these are filled in row-major order by
    // push_back (dgdx[i*nx + k], dgdt[i]) rather than sized-then-assigned.
    std::vector<Expr> dgdx;
    dgdx.reserve(ny * nx);
    std::vector<Expr> dgdt;
    dgdt.reserve(ny);
    for (std::size_t i = 0; i < ny; ++i) {
        auto dt = differentiate(g[i], time);
        if (!dt) {
            return make_error<DaeSolution>(dt.error());
        }
        dgdt.push_back(std::move(*dt));
        for (std::size_t k = 0; k < nx; ++k) {
            auto dx = differentiate(g[i], xvars[k]);
            if (!dx) {
                return make_error<DaeSolution>(dx.error());
            }
            dgdx.push_back(std::move(*dx));
        }
    }

    // y'_i = sum_k -Ainv(i,k) * ( sum_m dgdx(k,m) f[m] + dgdt(k) ).
    std::vector<Expr> y_prime;
    y_prime.reserve(ny);
    for (std::size_t i = 0; i < ny; ++i) {
        std::vector<Expr> terms;
        terms.reserve(ny);
        for (std::size_t k = 0; k < ny; ++k) {
            auto neg = ainv->at(i, k).negate();
            if (!neg) {
                return make_error<DaeSolution>(neg.error());
            }
            auto coeff = Expr::rational(neg->numerator(), neg->denominator());
            if (!coeff) {
                return make_error<DaeSolution>(coeff.error());
            }
            std::vector<Expr> inner;
            inner.reserve(nx + 1);
            for (std::size_t m = 0; m < nx; ++m) {
                inner.push_back(Expr::product({dgdx[k * nx + m], f[m]}));
            }
            inner.push_back(dgdt[k]);
            terms.push_back(Expr::product({*coeff, Expr::sum(std::move(inner))}));
        }
        auto yp = simplify(Expr::sum(std::move(terms)));
        if (!yp) {
            return make_error<DaeSolution>(yp.error());
        }
        y_prime.push_back(std::move(*yp));
    }

    const std::string time_name(time);
    SystemOperator field = [&, order, time_name](const std::vector<PowerSeries>& u)
        -> Result<std::vector<PowerSeries>> {
        if (u.size() != nx + ny) {
            return make_error<std::vector<PowerSeries>>(MathError::domain_error);
        }
        auto tvar = PowerSeries::variable(order);
        if (!tvar) {
            return make_error<std::vector<PowerSeries>>(tvar.error());
        }
        std::unordered_map<std::string, PowerSeries> b;
        b.reserve(nx + ny + 1);
        // PowerSeries has no default constructor, so operator[] is unavailable; bind each
        // symbol with insert_or_assign instead.
        b.insert_or_assign(time_name, std::move(*tvar));
        for (std::size_t m = 0; m < nx; ++m) {
            b.insert_or_assign(xvars[m], u[m]);
        }
        for (std::size_t i = 0; i < ny; ++i) {
            b.insert_or_assign(yvars[i], u[nx + i]);
        }
        std::vector<PowerSeries> rhs;
        rhs.reserve(nx + ny);
        for (const Expr& e : f) {
            auto v = eval_expr_series(e, b, order);
            if (!v) {
                return make_error<std::vector<PowerSeries>>(v.error());
            }
            rhs.push_back(std::move(*v));
        }
        for (const Expr& e : y_prime) {
            auto v = eval_expr_series(e, b, order);
            if (!v) {
                return make_error<std::vector<PowerSeries>>(v.error());
            }
            rhs.push_back(std::move(*v));
        }
        return rhs;
    };

    std::vector<Rational> u0;
    u0.reserve(nx + ny);
    u0.insert(u0.end(), x0.begin(), x0.end());
    u0.insert(u0.end(), y0.begin(), y0.end());

    auto sol = solve_first_order_system(std::move(field), u0, order);
    if (!sol) {
        return make_error<DaeSolution>(sol.error());
    }
    DaeSolution out;
    out.x.assign(sol->begin(), sol->begin() + static_cast<std::ptrdiff_t>(nx));
    out.y.assign(sol->begin() + static_cast<std::ptrdiff_t>(nx), sol->end());

    // Verify g == 0 to truncation order.
    auto tvar_check = PowerSeries::variable(order);
    if (!tvar_check) {
        return make_error<DaeSolution>(tvar_check.error());
    }
    std::unordered_map<std::string, PowerSeries> check_b;
    check_b.reserve(nx + ny + 1);
    check_b.insert_or_assign(time_name, std::move(*tvar_check));
    for (std::size_t m = 0; m < nx; ++m) {
        check_b.insert_or_assign(xvars[m], out.x[m]);
    }
    for (std::size_t i = 0; i < ny; ++i) {
        check_b.insert_or_assign(yvars[i], out.y[i]);
    }
    for (const Expr& e : g) {
        auto v = eval_expr_series(e, check_b, order);
        if (!v) {
            return make_error<DaeSolution>(v.error());
        }
        for (const Rational& c : v->coefficients()) {
            if (!c.is_zero()) {
                return make_error<DaeSolution>(MathError::domain_error);
            }
        }
    }
    return out;
}

}  // namespace nimblecas::daenl
