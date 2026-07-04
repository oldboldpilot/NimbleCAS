// NimbleCAS calculus of variations: Euler-Lagrange equations, the Beltrami first
// integral, holonomic / non-holonomic (Pfaffian) constraints, and Lagrange
// multipliers -- all EXACT symbolic manipulation over the Expr layer (ROADMAP 7.19).
// @author Olumuyiwa Oluwasanmi
//
// Everything here is a thin, exact composition over the symbolic differentiation
// engine (nimblecas.diff) and automatic simplification (nimblecas.simplify). A
// functional J[y] = integral F(x, y, y') dx has stationary paths characterised by the
// Euler-Lagrange equation  dF/dy - d/dx(dF/dy') = 0, where dF/dy and dF/dy' are ordinary
// partials (y, y' treated as independent symbols) and d/dx is the TOTAL derivative that
// chains through the jet  x -> (explicit),  y -> y',  y' -> y''. Mechanics Lagrangians
// L(q, q', t) use the same machinery with the sign convention  d/dt(dL/dq') - dL/dq = 0
// and t as the independent variable. Several dependent variables / generalized
// coordinates are supported: the total derivative chains through EVERY coordinate's jet
// so coupled systems are handled exactly.
//
// HONESTY BOUNDARY. This module produces the GOVERNING equations exactly; it does NOT
// solve the resulting ODEs/PDEs (that is the job of nimblecas.ode / nimblecas.pde).
// Integrability classification of a Pfaffian form is symbolic and best-effort: exactness
// (closedness) and the Frobenius condition are checked by structural zero-testing after
// simplification, which is a sound-but-incomplete decision procedure -- when a Frobenius
// obstruction neither reduces to zero nor is a provably nonzero constant, the form is
// reported as not_determinable rather than guessed. No floating-point heuristics are used.

export module nimblecas.calcvar;

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.diff;
import nimblecas.simplify;

export namespace nimblecas {

using ExprVector = std::vector<Expr>;

// ---------------------------------------------------------------------------
// A generalized coordinate expressed as a jet of independent symbols: the value
// q, its velocity q' (= dq/dt), and its acceleration q'' (= d^2 q/dt^2). The
// three are distinct symbols in the Expr layer; the total-derivative machinery
// links them by the chain rule  q -> q' -> q''.
// ---------------------------------------------------------------------------
struct Coordinate {
    std::string value;         // q  (or y)
    std::string velocity;      // q' (or y')
    std::string acceleration;  // q'' (or y''), introduced by the total d/dt
};

// Convenience: the primed jet of a base name -- {"q", "q'", "q''"}.
[[nodiscard]] auto coordinate(std::string_view base) -> Coordinate;

// Classification of a Pfaffian (differential-form) constraint.
enum class ConstraintClass : std::uint8_t {
    holonomic,        // integrable: equivalent to g(q, t) = const (exact or Frobenius-integrable)
    non_holonomic,    // a provable Frobenius obstruction (a nonzero-constant ω ∧ dω term)
    not_determinable, // an obstruction that could not be proven zero or nonzero here
};

// A Pfaffian 1-form constraint  sum_i coeffs[i] dq_i + time_coeff dt = 0.
// coeffs.size() must equal the number of coordinates; time_coeff is the dt term
// (use Expr::integer(0) when absent).
struct PfaffianConstraint {
    ExprVector coeffs;
    Expr time_coeff;
};

// The augmented equations of a constrained variational / mechanics problem: one
// Euler-Lagrange equation per coordinate (each == 0) plus the constraint equations
// (each == 0).
struct ConstrainedSystem {
    ExprVector equations;
    ExprVector constraints;
};

// The finite-dimensional Lagrange-multiplier stationarity system: one equation per
// variable  df/dx_i - sum_k lambda_k dg_k/dx_i = 0  plus the constraints g_k = 0.
struct MultiplierSystem {
    ExprVector stationarity;
    ExprVector constraints;
};

// ---------------------------------------------------------------------------
// Euler-Lagrange (variational / functional form): dF/dy_i - d/dx(dF/dy_i') for
// each coordinate, one equation (== 0) per dependent variable.
// ---------------------------------------------------------------------------
[[nodiscard]] auto euler_lagrange(const Expr& F, std::string_view indep,
                                  const std::vector<Coordinate>& coords) -> Result<ExprVector>;

// Single dependent-variable convenience overload. `y`, `yp`, `ypp` name the value,
// first-, and second-derivative symbols; `x` is the independent variable. Returns the
// scalar EL expression  dF/dy - d/dx(dF/dyp).
[[nodiscard]] auto euler_lagrange(const Expr& F, std::string_view y, std::string_view yp,
                                  std::string_view ypp, std::string_view x) -> Result<Expr>;

// Mechanics form: d/dt(dL/dq_i') - dL/dq_i for each generalized coordinate. This is the
// variational EL negated, i.e. the Newtonian sign convention (e.g. q'' + w^2 q = 0).
[[nodiscard]] auto lagrange_equations(const Expr& L, std::string_view indep,
                                      const std::vector<Coordinate>& coords) -> Result<ExprVector>;

// The first variation delta-J of J[y] = integral F dx is
//   delta-J = integral sum_i (dF/dy_i - d/dx dF/dy_i') eta_i dx,
// and the extremal condition delta-J = 0 (for arbitrary variations eta_i vanishing at the
// endpoints) is exactly EL_i = 0. This returns the variational derivatives dJ/dy_i (the
// EL integrand factors); the extremal condition is that every entry equals zero.
[[nodiscard]] auto variational_derivative(const Expr& F, std::string_view indep,
                                          const std::vector<Coordinate>& coords)
    -> Result<ExprVector>;

// Beltrami first integral: when F has no EXPLICIT dependence on the independent variable,
// F - sum_i y_i' dF/dy_i' is constant along extremals (a first-order first integral). Fails
// with MathError::domain_error if F depends explicitly on `indep` (Beltrami does not apply).
[[nodiscard]] auto beltrami_identity(const Expr& F, std::string_view indep,
                                     const std::vector<Coordinate>& coords) -> Result<Expr>;

// Single-variable convenience: `y`, `yp` name the value / first-derivative symbols and
// `x` the independent variable.
[[nodiscard]] auto beltrami_identity(const Expr& F, std::string_view y, std::string_view yp,
                                     std::string_view x) -> Result<Expr>;

// ---------------------------------------------------------------------------
// Constrained Euler-Lagrange via Lagrange multipliers.
// ---------------------------------------------------------------------------

// Holonomic constraints g_k(q, t) = 0: forms L* = L + sum_k lambda_k g_k and returns the
// augmented mechanics EL equations  d/dt(dL*/dq_i') - dL*/dq_i = 0  together with the
// echoed constraints g_k = 0. `multipliers` names the lambda_k (size == holonomic.size()).
[[nodiscard]] auto constrained_euler_lagrange(const Expr& L, std::string_view indep,
                                              const std::vector<Coordinate>& coords,
                                              const ExprVector& holonomic,
                                              const std::vector<std::string>& multipliers)
    -> Result<ConstrainedSystem>;

// Non-holonomic (Pfaffian) constraints: each coordinate equation becomes
//   d/dt(dL/dq_i') - dL/dq_i - sum_k lambda_k a_{k,i} = 0,
// and the returned constraint equations are the velocity forms  sum_i a_{k,i} q_i' + a_{k,t} = 0.
[[nodiscard]] auto constrained_euler_lagrange(const Expr& L, std::string_view indep,
                                              const std::vector<Coordinate>& coords,
                                              const std::vector<PfaffianConstraint>& constraints,
                                              const std::vector<std::string>& multipliers)
    -> Result<ConstrainedSystem>;

// Finite-dimensional Lagrange multipliers: stationary points of f subject to g_k = 0 via
// grad f = sum_k lambda_k grad g_k. Returns the multiplier system (stationarity per
// variable + the constraints). `multipliers` names lambda_k (size == constraints.size()).
[[nodiscard]] auto lagrange_multipliers(const Expr& f, const std::vector<std::string>& vars,
                                        const ExprVector& constraints,
                                        const std::vector<std::string>& multipliers)
    -> Result<MultiplierSystem>;

// ---------------------------------------------------------------------------
// Constraint classification.
// ---------------------------------------------------------------------------

// Classify a Pfaffian 1-form  sum_i coeffs[i] d(vars[i]) = 0  as holonomic (integrable) or
// non-holonomic. Exact/closed forms and forms in fewer than three variables are holonomic;
// otherwise the Frobenius condition ω ∧ dω = 0 is tested symbolically. coeffs.size() must
// equal vars.size().
[[nodiscard]] auto classify_pfaffian(const ExprVector& coeffs,
                                     const std::vector<std::string>& vars)
    -> Result<ConstraintClass>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// a - b and -a as unsimplified symbolic expressions (folded by the caller's simplify).
[[nodiscard]] auto negate(const Expr& a) -> Expr {
    return Expr::product({Expr::integer(-1), a});
}
[[nodiscard]] auto subtract(const Expr& a, const Expr& b) -> Expr {
    return Expr::sum({a, negate(b)});
}

// The top-level additive terms of a (simplified) expression; a non-sum is a single term.
[[nodiscard]] auto add_terms(const Expr& e) -> std::vector<Expr> {
    if (auto a = as<AddNode>(e.node().value)) {
        return (*a)->terms;
    }
    return {e};
}

// simplify(pos - neg), distributing the subtraction over neg's additive terms. Cohen's
// automatic simplification does NOT expand -1 * (t_1 + ... + t_n), so a plain
// simplify(subtract(pos, neg)) would leave a sign trapped outside a sum and fail to cancel;
// pushing the negation onto each additive term of neg (which is already flat after simplify)
// lets like terms combine exactly.
[[nodiscard]] auto simplified_difference(const Expr& pos, const Expr& neg) -> Result<Expr> {
    std::vector<Expr> terms = add_terms(pos);
    for (const Expr& t : add_terms(neg)) {
        terms.push_back(negate(t));
    }
    return simplify(Expr::sum(std::move(terms)));
}

// True iff e is structurally the integer 0 (after the caller has simplified it).
[[nodiscard]] auto is_syntactic_zero(const Expr& e) -> bool {
    return e.is_equivalent_to(Expr::integer(0));
}

// True iff e is a numeric constant that is provably nonzero.
[[nodiscard]] auto nonzero_constant(const Expr& e) -> bool {
    auto c = as<ConstantNode>(e.node().value);
    if (!c) {
        return false;
    }
    return std::visit(
        []<typename V>(const V& v) -> bool {
            if constexpr (std::is_same_v<V, std::int64_t>) {
                return v != 0;
            } else if constexpr (std::is_same_v<V, double>) {
                return v != 0.0;
            } else {  // std::pair<int64,int64> exact rational: numerator nonzero
                return v.first != 0;
            }
        },
        (*c)->value);
}

// The chain-rule jet: each coordinate contributes  q -> q'  and  q' -> q''  links, so a
// total d/d(indep) chains through value->velocity and velocity->acceleration for EVERY
// coordinate (required for coupled multi-coordinate systems).
[[nodiscard]] auto build_jet(const std::vector<Coordinate>& coords)
    -> std::vector<std::pair<std::string, Expr>> {
    std::vector<std::pair<std::string, Expr>> jet;
    jet.reserve(coords.size() * 2);
    for (const Coordinate& c : coords) {
        jet.emplace_back(c.value, Expr::symbol(c.velocity));
        jet.emplace_back(c.velocity, Expr::symbol(c.acceleration));
    }
    return jet;
}

// Total derivative d/d(indep) of g along the jet: the explicit partial plus one chain-rule
// contribution (dg/dname) * deriv per jet link.
[[nodiscard]] auto total_derivative_jet(const Expr& g, std::string_view indep,
                                        const std::vector<std::pair<std::string, Expr>>& jet)
    -> Result<Expr> {
    std::vector<Expr> terms;
    terms.reserve(jet.size() + 1);
    auto explicit_part = differentiate(g, indep);
    if (!explicit_part) {
        return explicit_part;
    }
    terms.push_back(std::move(*explicit_part));
    for (const auto& [name, deriv] : jet) {
        auto d = differentiate(g, name);
        if (!d) {
            return d;
        }
        terms.push_back(Expr::product({*d, deriv}));
    }
    return simplify(Expr::sum(std::move(terms)));
}

enum class ELForm : std::uint8_t { functional, mechanics };

// Core Euler-Lagrange assembly shared by the variational and mechanics forms.
[[nodiscard]] auto el_equations(const Expr& F, std::string_view indep,
                                const std::vector<Coordinate>& coords, ELForm form)
    -> Result<ExprVector> {
    if (coords.empty()) {
        return make_error<ExprVector>(MathError::domain_error);
    }
    const auto jet = build_jet(coords);
    ExprVector eqs;
    eqs.reserve(coords.size());
    for (const Coordinate& c : coords) {
        auto dF_dvel = differentiate(F, c.velocity);  // dF/dq_i'
        if (!dF_dvel) {
            return make_error<ExprVector>(dF_dvel.error());
        }
        auto ddt = total_derivative_jet(*dF_dvel, indep, jet);  // d/d(indep)(dF/dq_i')
        if (!ddt) {
            return make_error<ExprVector>(ddt.error());
        }
        auto dF_dval = differentiate(F, c.value);  // dF/dq_i
        if (!dF_dval) {
            return make_error<ExprVector>(dF_dval.error());
        }
        auto eq = (form == ELForm::mechanics) ? simplified_difference(*ddt, *dF_dval)
                                              : simplified_difference(*dF_dval, *ddt);
        if (!eq) {
            return make_error<ExprVector>(eq.error());
        }
        eqs.push_back(std::move(*eq));
    }
    return eqs;
}

// One Frobenius term a_i(d_j a_k - d_k a_j) + a_j(d_k a_i - d_i a_k) + a_k(d_i a_j - d_j a_i),
// the coefficient of the triple (i, j, k) in ω ∧ dω.
[[nodiscard]] auto frobenius_triple(const ExprVector& a, const std::vector<std::string>& vars,
                                    std::size_t i, std::size_t j, std::size_t k) -> Result<Expr> {
    auto dj_ak = differentiate(a[k], vars[j]);
    if (!dj_ak) {
        return dj_ak;
    }
    auto dk_aj = differentiate(a[j], vars[k]);
    if (!dk_aj) {
        return dk_aj;
    }
    auto dk_ai = differentiate(a[i], vars[k]);
    if (!dk_ai) {
        return dk_ai;
    }
    auto di_ak = differentiate(a[k], vars[i]);
    if (!di_ak) {
        return di_ak;
    }
    auto di_aj = differentiate(a[j], vars[i]);
    if (!di_aj) {
        return di_aj;
    }
    auto dj_ai = differentiate(a[i], vars[j]);
    if (!dj_ai) {
        return dj_ai;
    }
    const Expr t1 = Expr::product({a[i], subtract(*dj_ak, *dk_aj)});
    const Expr t2 = Expr::product({a[j], subtract(*dk_ai, *di_ak)});
    const Expr t3 = Expr::product({a[k], subtract(*di_aj, *dj_ai)});
    return Expr::sum({t1, t2, t3});
}

}  // namespace

auto coordinate(std::string_view base) -> Coordinate {
    std::string value{base};
    return Coordinate{.value = value,
                      .velocity = value + "'",
                      .acceleration = value + "''"};
}

auto euler_lagrange(const Expr& F, std::string_view indep,
                    const std::vector<Coordinate>& coords) -> Result<ExprVector> {
    return el_equations(F, indep, coords, ELForm::functional);
}

auto euler_lagrange(const Expr& F, std::string_view y, std::string_view yp,
                    std::string_view ypp, std::string_view x) -> Result<Expr> {
    auto eqs = el_equations(
        F, x,
        {Coordinate{.value = std::string(y), .velocity = std::string(yp),
                    .acceleration = std::string(ypp)}},
        ELForm::functional);
    if (!eqs) {
        return make_error<Expr>(eqs.error());
    }
    return eqs->front();
}

auto lagrange_equations(const Expr& L, std::string_view indep,
                        const std::vector<Coordinate>& coords) -> Result<ExprVector> {
    return el_equations(L, indep, coords, ELForm::mechanics);
}

auto variational_derivative(const Expr& F, std::string_view indep,
                            const std::vector<Coordinate>& coords) -> Result<ExprVector> {
    return el_equations(F, indep, coords, ELForm::functional);
}

auto beltrami_identity(const Expr& F, std::string_view indep,
                       const std::vector<Coordinate>& coords) -> Result<Expr> {
    if (coords.empty()) {
        return make_error<Expr>(MathError::domain_error);
    }
    // Beltrami holds only when F has no explicit dependence on the independent variable.
    if (!free_of(F, Expr::symbol(std::string(indep)))) {
        return make_error<Expr>(MathError::domain_error);
    }
    std::vector<Expr> terms;
    terms.reserve(coords.size() + 1);
    terms.push_back(F);
    for (const Coordinate& c : coords) {
        auto d = differentiate(F, c.velocity);  // dF/dq_i'
        if (!d) {
            return d;
        }
        terms.push_back(negate(Expr::product({Expr::symbol(c.velocity), *d})));  // - q_i' dF/dq_i'
    }
    return simplify(Expr::sum(std::move(terms)));
}

auto beltrami_identity(const Expr& F, std::string_view y, std::string_view yp,
                       std::string_view x) -> Result<Expr> {
    // The acceleration name is unused by Beltrami (velocity terms only); supply a placeholder.
    return beltrami_identity(
        F, x,
        {Coordinate{.value = std::string(y), .velocity = std::string(yp),
                    .acceleration = std::string(yp) + "'"}});
}

auto constrained_euler_lagrange(const Expr& L, std::string_view indep,
                                const std::vector<Coordinate>& coords,
                                const ExprVector& holonomic,
                                const std::vector<std::string>& multipliers)
    -> Result<ConstrainedSystem> {
    if (holonomic.size() != multipliers.size()) {
        return make_error<ConstrainedSystem>(MathError::domain_error);
    }
    // L* = L + sum_k lambda_k g_k, then ordinary mechanics EL on the augmented Lagrangian.
    std::vector<Expr> terms;
    terms.reserve(holonomic.size() + 1);
    terms.push_back(L);
    for (std::size_t k = 0; k < holonomic.size(); ++k) {
        terms.push_back(Expr::product({Expr::symbol(multipliers[k]), holonomic[k]}));
    }
    const Expr augmented = Expr::sum(std::move(terms));

    auto eqs = el_equations(augmented, indep, coords, ELForm::mechanics);
    if (!eqs) {
        return make_error<ConstrainedSystem>(eqs.error());
    }
    ExprVector gs;
    gs.reserve(holonomic.size());
    for (const Expr& g : holonomic) {
        auto s = simplify(g);
        if (!s) {
            return make_error<ConstrainedSystem>(s.error());
        }
        gs.push_back(std::move(*s));
    }
    return ConstrainedSystem{.equations = std::move(*eqs), .constraints = std::move(gs)};
}

auto constrained_euler_lagrange(const Expr& L, std::string_view indep,
                                const std::vector<Coordinate>& coords,
                                const std::vector<PfaffianConstraint>& constraints,
                                const std::vector<std::string>& multipliers)
    -> Result<ConstrainedSystem> {
    if (constraints.size() != multipliers.size()) {
        return make_error<ConstrainedSystem>(MathError::domain_error);
    }
    for (const PfaffianConstraint& pf : constraints) {
        if (pf.coeffs.size() != coords.size()) {
            return make_error<ConstrainedSystem>(MathError::domain_error);
        }
    }
    // Unconstrained mechanics EL, then subtract the multiplier * Pfaffian-coefficient terms.
    auto base = el_equations(L, indep, coords, ELForm::mechanics);
    if (!base) {
        return make_error<ConstrainedSystem>(base.error());
    }
    ExprVector eqs;
    eqs.reserve(coords.size());
    for (std::size_t i = 0; i < coords.size(); ++i) {
        std::vector<Expr> terms;
        terms.reserve(constraints.size() + 1);
        terms.push_back((*base)[i]);
        for (std::size_t k = 0; k < constraints.size(); ++k) {
            terms.push_back(
                negate(Expr::product({Expr::symbol(multipliers[k]), constraints[k].coeffs[i]})));
        }
        auto eq = simplify(Expr::sum(std::move(terms)));
        if (!eq) {
            return make_error<ConstrainedSystem>(eq.error());
        }
        eqs.push_back(std::move(*eq));
    }
    // Velocity form of each Pfaffian constraint: sum_i a_i q_i' + a_t = 0.
    ExprVector cons;
    cons.reserve(constraints.size());
    for (const PfaffianConstraint& pf : constraints) {
        std::vector<Expr> terms;
        terms.reserve(coords.size() + 1);
        for (std::size_t i = 0; i < coords.size(); ++i) {
            terms.push_back(Expr::product({pf.coeffs[i], Expr::symbol(coords[i].velocity)}));
        }
        terms.push_back(pf.time_coeff);
        auto s = simplify(Expr::sum(std::move(terms)));
        if (!s) {
            return make_error<ConstrainedSystem>(s.error());
        }
        cons.push_back(std::move(*s));
    }
    return ConstrainedSystem{.equations = std::move(eqs), .constraints = std::move(cons)};
}

auto lagrange_multipliers(const Expr& f, const std::vector<std::string>& vars,
                          const ExprVector& constraints,
                          const std::vector<std::string>& multipliers)
    -> Result<MultiplierSystem> {
    if (constraints.size() != multipliers.size()) {
        return make_error<MultiplierSystem>(MathError::domain_error);
    }
    ExprVector stationarity;
    stationarity.reserve(vars.size());
    for (const std::string& v : vars) {
        auto df = differentiate(f, v);  // df/dx_i
        if (!df) {
            return make_error<MultiplierSystem>(df.error());
        }
        std::vector<Expr> terms;
        terms.reserve(constraints.size() + 1);
        terms.push_back(*df);
        for (std::size_t k = 0; k < constraints.size(); ++k) {
            auto dg = differentiate(constraints[k], v);  // dg_k/dx_i
            if (!dg) {
                return make_error<MultiplierSystem>(dg.error());
            }
            terms.push_back(negate(Expr::product({Expr::symbol(multipliers[k]), *dg})));
        }
        auto s = simplify(Expr::sum(std::move(terms)));  // df/dx_i - sum_k lambda_k dg_k/dx_i
        if (!s) {
            return make_error<MultiplierSystem>(s.error());
        }
        stationarity.push_back(std::move(*s));
    }
    ExprVector cons;
    cons.reserve(constraints.size());
    for (const Expr& g : constraints) {
        auto s = simplify(g);
        if (!s) {
            return make_error<MultiplierSystem>(s.error());
        }
        cons.push_back(std::move(*s));
    }
    return MultiplierSystem{.stationarity = std::move(stationarity), .constraints = std::move(cons)};
}

auto classify_pfaffian(const ExprVector& coeffs, const std::vector<std::string>& vars)
    -> Result<ConstraintClass> {
    const std::size_t m = coeffs.size();
    if (m != vars.size() || m == 0) {
        return make_error<ConstraintClass>(MathError::domain_error);
    }

    // Closedness (exactness): d a_i / d x_j == d a_j / d x_i for all i < j  =>  the form is
    // dP for a potential P, hence holonomic.
    bool exact = true;
    for (std::size_t i = 0; i < m && exact; ++i) {
        for (std::size_t j = i + 1; j < m; ++j) {
            auto dij = differentiate(coeffs[i], vars[j]);
            if (!dij) {
                return make_error<ConstraintClass>(dij.error());
            }
            auto dji = differentiate(coeffs[j], vars[i]);
            if (!dji) {
                return make_error<ConstraintClass>(dji.error());
            }
            auto diff = simplified_difference(*dij, *dji);
            if (!diff) {
                return make_error<ConstraintClass>(diff.error());
            }
            if (!is_syntactic_zero(*diff)) {
                exact = false;
                break;
            }
        }
    }
    if (exact) {
        return ConstraintClass::holonomic;
    }

    // In fewer than three variables a 1-form is always integrable (ω ∧ dω is a 3-form and
    // vanishes identically), so it admits an integrating factor: holonomic.
    if (m < 3) {
        return ConstraintClass::holonomic;
    }

    // Frobenius integrability: ω ∧ dω = 0 over every variable triple.
    bool undecided = false;
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = i + 1; j < m; ++j) {
            for (std::size_t k = j + 1; k < m; ++k) {
                auto term = frobenius_triple(coeffs, vars, i, j, k);
                if (!term) {
                    return make_error<ConstraintClass>(term.error());
                }
                auto s = simplify(*term);
                if (!s) {
                    return make_error<ConstraintClass>(s.error());
                }
                if (is_syntactic_zero(*s)) {
                    continue;
                }
                if (nonzero_constant(*s)) {
                    return ConstraintClass::non_holonomic;  // provable obstruction
                }
                undecided = true;  // nonzero here, but not provably so in general
            }
        }
    }
    if (undecided) {
        return ConstraintClass::not_determinable;
    }
    return ConstraintClass::holonomic;
}

}  // namespace nimblecas
