// NimbleCAS Hamiltonian mechanics: Legendre transform, canonical equations, Poisson
// brackets, conserved quantities, phase space, and action-angle variables.
// @author Olumuyiwa Oluwasanmi
//
// These operators are exact, symbolic compositions over the differentiation engine
// (nimblecas.diff) and Cohen automatic simplification (nimblecas.simplify). Nothing here
// is numeric: a canonical coordinate system carries the symbol names of the generalized
// coordinates q_i, their velocities q̇_i, and their conjugate momenta p_i, and every
// operator is a partial-derivative composition in which the other symbols are held fixed
// (q and p are treated as independent phase-space coordinates, exactly as Hamilton's
// formalism requires).
//
// The Legendre transform L(q,q̇,t) -> H(q,p,t) inverts the momentum map p_i = ∂L/∂q̇_i to
// recover q̇_i(q,p,t). That inversion is only linear (hence exactly solvable here) when L
// is affine/quadratic in the velocities: the Hessian M_ij = ∂²L/∂q̇_i∂q̇_j must be free of
// the velocities, and it must be non-singular. When M depends on the velocities (L is
// cubic or higher in q̇) or when det M simplifies to zero, the inversion is reported as
// MathError::not_implemented — the honest boundary of this module.
//
// The one-degree-of-freedom action J = (1/2π)∮ p dq is returned as its integrand
// p(q,E) obtained by solving H(q,p)=E for p (elementary only when H is quadratic in p).
// The closed-form ∮ is generally non-elementary (p(q,E) is an irrational function such as
// sqrt(2E - ω²q²), outside the rational domain of nimblecas.integrate), so the closed form
// is left absent and the angle variable θ = ∂W/∂J is reported as not_implemented rather
// than fabricated. Dimension mismatches are MathError::domain_error; any differentiation
// or simplification failure propagates (Result / MathError, Rule 32).

export module nimblecas.mechanics;

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.diff;
import nimblecas.simplify;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// A canonical coordinate system.
//
// q[i], qdot[i], p[i] are the symbol names of the i-th generalized coordinate, its
// velocity q̇_i, and its conjugate momentum p_i; they must all have the same length
// (the number of degrees of freedom). time is the name of the evolution parameter.
// ---------------------------------------------------------------------------
struct Coordinates {
    std::vector<std::string> q;      // generalized coordinates q_i
    std::vector<std::string> qdot;   // velocities q̇_i (same size as q)
    std::vector<std::string> p;      // conjugate momenta p_i (same size as q)
    std::string time{"t"};           // evolution-parameter symbol
};

// The result of a Legendre transform L(q,q̇,t) -> H(q,p,t).
struct Hamiltonian {
    Expr H;                        // H(q,p,t) = Σ p_i q̇_i − L, in phase-space variables
    std::vector<Expr> momenta;     // p_i = ∂L/∂q̇_i, in (q,q̇,t) — the momentum definition
    std::vector<Expr> velocities;  // q̇_i(q,p,t), the inverted momentum map
};

// Hamilton's canonical first-order system (also the phase-space velocity field).
struct HamiltonSystem {
    std::vector<Expr> qdot;  // q̇_i = ∂H/∂p_i
    std::vector<Expr> pdot;  // ṗ_i = −∂H/∂q_i
};

// The 1-DOF action integrand p(q,E) for J = (1/2π)∮ p dq.
struct ActionIntegral {
    Expr integrand;                   // p(q,E): the ∮ p dq integrand at energy E
    std::string coordinate;           // the q variable of integration
    std::optional<Expr> closed_form;  // the elementary ∮ p dq, if one was found (else absent)
};

// Legendre transform L(q,q̇,t) -> H(q,p,t). Defines p_i = ∂L/∂q̇_i, inverts the (linear)
// momentum map to obtain q̇_i(q,p,t), and forms H = Σ p_i q̇_i − L in phase-space
// variables. Fails with domain_error (inconsistent coordinate sizes) or not_implemented
// (L not quadratic in the velocities, or a singular mass matrix).
[[nodiscard]] auto legendre_transform(const Expr& lagrangian, const Coordinates& coords)
    -> Result<Hamiltonian>;

// Hamilton's canonical equations q̇_i = ∂H/∂p_i, ṗ_i = −∂H/∂q_i as a 2n first-order system.
[[nodiscard]] auto hamilton_equations(const Expr& H, const Coordinates& coords)
    -> Result<HamiltonSystem>;

// Poisson bracket {f,g} = Σ_i (∂f/∂q_i ∂g/∂p_i − ∂f/∂p_i ∂g/∂q_i).
[[nodiscard]] auto poisson_bracket(const Expr& f, const Expr& g, const Coordinates& coords)
    -> Result<Expr>;

// Total time derivative along the Hamiltonian flow: df/dt = {f,H} + ∂f/∂t.
[[nodiscard]] auto total_time_derivative(const Expr& f, const Expr& H,
                                         const Coordinates& coords) -> Result<Expr>;

// Whether f is a constant of motion, i.e. df/dt = {f,H} + ∂f/∂t simplifies to 0.
[[nodiscard]] auto is_constant_of_motion(const Expr& f, const Expr& H,
                                         const Coordinates& coords) -> Result<bool>;

// Indices i of cyclic (ignorable) coordinates: H is free of q_i, hence p_i is conserved.
[[nodiscard]] auto cyclic_coordinates(const Expr& H, const Coordinates& coords)
    -> Result<std::vector<std::size_t>>;

// Implicit phase curve of a level set H(q,p)=E: returns the relation H − E (which is 0 on
// the curve).
[[nodiscard]] auto phase_curve(const Expr& H, const Expr& energy) -> Result<Expr>;

// Phase-space vector field (q̇,ṗ) for sampling/plotting — Hamilton's equations by another
// name.
[[nodiscard]] auto phase_portrait_field(const Expr& H, const Coordinates& coords)
    -> Result<HamiltonSystem>;

// 1-DOF action integrand: solve H(q,p)=E for p(q,E), the ∮ p dq integrand. Requires a
// single degree of freedom (domain_error otherwise) and H at most quadratic in p
// (not_implemented otherwise). The closed form is filled in only if it is elementary here.
[[nodiscard]] auto action_integral(const Expr& H, const Coordinates& coords,
                                   const Expr& energy) -> Result<ActionIntegral>;

// Angle variable θ = ∂W/∂J, where W = ∫ p dq is Hamilton's characteristic function.
// Elementary only when the action's closed form exists; otherwise not_implemented (honest:
// the module does not fabricate a closed form for a non-elementary action).
[[nodiscard]] auto angle_variable(const Expr& H, const Coordinates& coords,
                                  const Expr& energy) -> Result<Expr>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// --- small symbolic construction helpers -----------------------------------

[[nodiscard]] auto neg(const Expr& a) -> Expr {
    return Expr::product({Expr::integer(-1), a});
}
[[nodiscard]] auto sub(const Expr& a, const Expr& b) -> Expr {
    return Expr::sum({a, neg(b)});
}
[[nodiscard]] auto recip(const Expr& a) -> Expr {
    return Expr::power(a, Expr::integer(-1));  // a^(-1)
}
[[nodiscard]] auto square(const Expr& a) -> Expr {
    return Expr::power(a, Expr::integer(2));
}

// A partial derivative ∂f/∂v (differentiate already returns a simplified result).
[[nodiscard]] auto partial(const Expr& f, std::string_view v) -> Result<Expr> {
    return differentiate(f, v);
}

// The coordinate system is well-formed when the three symbol lists agree in length.
[[nodiscard]] auto valid(const Coordinates& c) -> bool {
    return c.q.size() == c.qdot.size() && c.q.size() == c.p.size() && !c.q.empty();
}

// Symbolic determinant by Laplace (cofactor) expansion along the first row. The matrix is
// square; entries are arbitrary Exprs. Mechanics mass matrices are small (a handful of
// degrees of freedom) so the factorial cost is irrelevant, and the result stays exact.
[[nodiscard]] auto determinant(const std::vector<std::vector<Expr>>& m) -> Expr {
    const std::size_t n = m.size();
    if (n == 1) {
        return m[0][0];
    }
    if (n == 2) {
        return sub(Expr::product({m[0][0], m[1][1]}), Expr::product({m[0][1], m[1][0]}));
    }
    std::vector<Expr> terms;
    terms.reserve(n);
    for (std::size_t j = 0; j < n; ++j) {
        std::vector<std::vector<Expr>> minor;
        minor.reserve(n - 1);
        for (std::size_t r = 1; r < n; ++r) {
            std::vector<Expr> row;
            row.reserve(n - 1);
            for (std::size_t c = 0; c < n; ++c) {
                if (c != j) {
                    row.push_back(m[r][c]);
                }
            }
            minor.push_back(std::move(row));
        }
        Expr cofactor = Expr::product({m[0][j], determinant(minor)});
        terms.push_back((j % 2 == 0) ? cofactor : neg(cofactor));
    }
    return Expr::sum(std::move(terms));
}

// Solve the linear system M q̇ = rhs for q̇ by Cramer's rule, keeping everything symbolic.
// Fails with not_implemented when det M simplifies to zero (a singular / non-invertible
// mass matrix — the velocities are not recoverable from the momenta).
[[nodiscard]] auto cramer_solve(const std::vector<std::vector<Expr>>& m,
                                const std::vector<Expr>& rhs) -> Result<std::vector<Expr>> {
    const std::size_t n = m.size();
    auto det_m = simplify(determinant(m));
    if (!det_m) {
        return make_error<std::vector<Expr>>(det_m.error());
    }
    if (det_m->is_equivalent_to(Expr::integer(0))) {
        return make_error<std::vector<Expr>>(MathError::not_implemented);
    }
    std::vector<Expr> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<std::vector<Expr>> mi = m;  // copy; replace column i with rhs
        for (std::size_t r = 0; r < n; ++r) {
            mi[r][i] = rhs[r];
        }
        auto det_i = simplify(determinant(mi));
        if (!det_i) {
            return make_error<std::vector<Expr>>(det_i.error());
        }
        auto xi = simplify(Expr::product({*det_i, recip(*det_m)}));
        if (!xi) {
            return make_error<std::vector<Expr>>(xi.error());
        }
        out.push_back(std::move(*xi));
    }
    return out;
}

// True when e contains none of the given velocity symbols (used to test the quadratic-in-q̇
// condition: a velocity-independent mass matrix).
[[nodiscard]] auto free_of_all(const Expr& e, const std::vector<std::string>& syms) -> bool {
    return std::ranges::all_of(
        syms, [&](const std::string& s) { return free_of(e, Expr::symbol(s)); });
}

}  // namespace

auto legendre_transform(const Expr& lagrangian, const Coordinates& coords)
    -> Result<Hamiltonian> {
    if (!valid(coords)) {
        return make_error<Hamiltonian>(MathError::domain_error);
    }
    const std::size_t n = coords.q.size();

    // 1. Conjugate momenta p_i = ∂L/∂q̇_i (in the velocity variables).
    std::vector<Expr> momenta;
    momenta.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto pi = partial(lagrangian, coords.qdot[i]);
        if (!pi) {
            return make_error<Hamiltonian>(pi.error());
        }
        momenta.push_back(std::move(*pi));
    }

    // 2. Mass (Hessian) matrix M_ij = ∂p_i/∂q̇_j = ∂²L/∂q̇_i∂q̇_j. For the momentum map to be
    //    linearly invertible here, every entry must be free of the velocities (L quadratic
    //    in q̇). Otherwise the inversion is nonlinear — the honest not_implemented boundary.
    std::vector<std::vector<Expr>> mass(n, std::vector<Expr>{});
    for (std::size_t i = 0; i < n; ++i) {
        mass[i].reserve(n);
        for (std::size_t j = 0; j < n; ++j) {
            auto mij = partial(momenta[i], coords.qdot[j]);
            if (!mij) {
                return make_error<Hamiltonian>(mij.error());
            }
            if (!free_of_all(*mij, coords.qdot)) {
                return make_error<Hamiltonian>(MathError::not_implemented);
            }
            mass[i].push_back(std::move(*mij));
        }
    }

    // 3. Affine offsets b_i = p_i|_{q̇=0}, so that p_i = Σ_j M_ij q̇_j + b_i. The momentum
    //    equations rearrange to M q̇ = (p_i − b_i), with p_i now the momentum SYMBOLS.
    std::vector<Expr> rhs;
    rhs.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        Expr bi = momenta[i];
        for (const std::string& v : coords.qdot) {
            bi = substitute(bi, Expr::symbol(v), Expr::integer(0));
        }
        auto ri = simplify(sub(Expr::symbol(coords.p[i]), bi));
        if (!ri) {
            return make_error<Hamiltonian>(ri.error());
        }
        rhs.push_back(std::move(*ri));
    }

    // 4. Invert: q̇_i(q,p,t) = (M^{-1} (p − b))_i via Cramer's rule.
    auto velocities = cramer_solve(mass, rhs);
    if (!velocities) {
        return make_error<Hamiltonian>(velocities.error());
    }

    // 5. H = 1/2 Σ_i (p_i − b_i) q̇_i − c, with c = L|_{q̇=0} the velocity-free remainder and
    //    q̇_i the inverted momentum map. This is the exact quadratic-form value of Σ p_i q̇_i − L
    //    for an L quadratic in the velocities (H = 1/2 (p−b)ᵀ M⁻¹ (p−b) − c, and M⁻¹(p−b) = q̇).
    //    Using it — instead of substituting the rational velocities back into L — keeps every
    //    factor linear, so no power-of-a-product (which the simplifier does not expand) is ever
    //    formed, and H collapses to its canonical form.
    Expr c = lagrangian;
    for (const std::string& v : coords.qdot) {
        c = substitute(c, Expr::symbol(v), Expr::integer(0));
    }
    const Expr half = Expr::rational(1, 2).value();
    std::vector<Expr> h_terms;
    h_terms.reserve(n + 1);
    for (std::size_t i = 0; i < n; ++i) {
        h_terms.push_back(Expr::product({half, rhs[i], (*velocities)[i]}));
    }
    h_terms.push_back(neg(c));
    auto hamil = simplify(Expr::sum(std::move(h_terms)));
    if (!hamil) {
        return make_error<Hamiltonian>(hamil.error());
    }

    return Hamiltonian{.H = std::move(*hamil),
                       .momenta = std::move(momenta),
                       .velocities = std::move(*velocities)};
}

auto hamilton_equations(const Expr& H, const Coordinates& coords) -> Result<HamiltonSystem> {
    if (!valid(coords)) {
        return make_error<HamiltonSystem>(MathError::domain_error);
    }
    const std::size_t n = coords.q.size();
    HamiltonSystem sys;
    sys.qdot.reserve(n);
    sys.pdot.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto dq = partial(H, coords.p[i]);  // q̇_i = ∂H/∂p_i
        if (!dq) {
            return make_error<HamiltonSystem>(dq.error());
        }
        sys.qdot.push_back(std::move(*dq));
        auto dp = partial(H, coords.q[i]);  // ṗ_i = −∂H/∂q_i
        if (!dp) {
            return make_error<HamiltonSystem>(dp.error());
        }
        auto npd = simplify(neg(*dp));
        if (!npd) {
            return make_error<HamiltonSystem>(npd.error());
        }
        sys.pdot.push_back(std::move(*npd));
    }
    return sys;
}

auto poisson_bracket(const Expr& f, const Expr& g, const Coordinates& coords) -> Result<Expr> {
    if (!valid(coords)) {
        return make_error<Expr>(MathError::domain_error);
    }
    const std::size_t n = coords.q.size();
    std::vector<Expr> terms;
    terms.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto df_dq = partial(f, coords.q[i]);
        auto dg_dp = partial(g, coords.p[i]);
        auto df_dp = partial(f, coords.p[i]);
        auto dg_dq = partial(g, coords.q[i]);
        if (!df_dq) return df_dq;
        if (!dg_dp) return dg_dp;
        if (!df_dp) return df_dp;
        if (!dg_dq) return dg_dq;
        // ∂f/∂q_i ∂g/∂p_i − ∂f/∂p_i ∂g/∂q_i
        terms.push_back(sub(Expr::product({*df_dq, *dg_dp}),
                            Expr::product({*df_dp, *dg_dq})));
    }
    return simplify(Expr::sum(std::move(terms)));
}

auto total_time_derivative(const Expr& f, const Expr& H, const Coordinates& coords)
    -> Result<Expr> {
    auto bracket = poisson_bracket(f, H, coords);
    if (!bracket) {
        return bracket;
    }
    auto explicit_t = partial(f, coords.time);  // ∂f/∂t
    if (!explicit_t) {
        return explicit_t;
    }
    return simplify(Expr::sum({*bracket, *explicit_t}));
}

auto is_constant_of_motion(const Expr& f, const Expr& H, const Coordinates& coords)
    -> Result<bool> {
    auto ddt = total_time_derivative(f, H, coords);
    if (!ddt) {
        return make_error<bool>(ddt.error());
    }
    return ddt->is_equivalent_to(Expr::integer(0));
}

auto cyclic_coordinates(const Expr& H, const Coordinates& coords)
    -> Result<std::vector<std::size_t>> {
    if (!valid(coords)) {
        return make_error<std::vector<std::size_t>>(MathError::domain_error);
    }
    std::vector<std::size_t> cyclic;
    for (std::size_t i = 0; i < coords.q.size(); ++i) {
        if (free_of(H, Expr::symbol(coords.q[i]))) {  // H independent of q_i => p_i conserved
            cyclic.push_back(i);
        }
    }
    return cyclic;
}

auto phase_curve(const Expr& H, const Expr& energy) -> Result<Expr> {
    return simplify(sub(H, energy));  // H − E == 0 on the level set
}

auto phase_portrait_field(const Expr& H, const Coordinates& coords) -> Result<HamiltonSystem> {
    return hamilton_equations(H, coords);  // the (q̇,ṗ) field IS Hamilton's system
}

namespace {

// Solve H(q,p) = E for p, treating H as (at most) a quadratic polynomial in the momentum
// symbol p: H = a p² + b p + c with a,b,c free of p. Returns the p = [−b + sqrt(b²−4a(c−E))]
// / (2a) branch (or the linear root when a == 0). not_implemented when H is higher than
// quadratic in p, or when the momentum cannot be isolated (a == 0 and b == 0).
[[nodiscard]] auto solve_momentum(const Expr& H, std::string_view p_name, const Expr& energy)
    -> Result<Expr> {
    const Expr p = Expr::symbol(std::string(p_name));

    auto dH = partial(H, p_name);  // 2a p + b
    if (!dH) {
        return dH;
    }
    auto d2H = partial(*dH, p_name);  // 2a
    if (!d2H) {
        return d2H;
    }
    // c = H|_{p=0}; b = ∂H/∂p|_{p=0}; a = (1/2) ∂²H/∂p².
    Expr c = substitute(H, p, Expr::integer(0));
    Expr b = substitute(*dH, p, Expr::integer(0));
    auto a = simplify(Expr::product({Expr::rational(1, 2).value(), *d2H}));
    if (!a) {
        return a;
    }
    // Quadratic requires a, b, c all free of p (H at most degree two in p).
    if (!free_of(*a, p) || !free_of(b, p) || !free_of(c, p)) {
        return make_error<Expr>(MathError::not_implemented);
    }

    if (a->is_equivalent_to(Expr::integer(0))) {
        // Linear in p: b p + c = E  =>  p = (E − c) / b. Needs b != 0.
        if (b.is_equivalent_to(Expr::integer(0))) {
            return make_error<Expr>(MathError::not_implemented);
        }
        return simplify(Expr::product({sub(energy, c), recip(b)}));
    }

    // Quadratic: p = (−b + sqrt(b² − 4a(c − E))) / (2a), the outgoing (p >= 0) branch.
    Expr disc = sub(square(b), Expr::product({Expr::integer(4), *a, sub(c, energy)}));
    Expr root = Expr::apply("sqrt", {disc});
    Expr twice_a = Expr::product({Expr::integer(2), *a});
    return simplify(Expr::product({sub(root, b), recip(twice_a)}));
}

}  // namespace

auto action_integral(const Expr& H, const Coordinates& coords, const Expr& energy)
    -> Result<ActionIntegral> {
    if (!valid(coords) || coords.q.size() != 1) {
        return make_error<ActionIntegral>(MathError::domain_error);  // action-angle is 1-DOF here
    }
    auto integrand = solve_momentum(H, coords.p[0], energy);
    if (!integrand) {
        return make_error<ActionIntegral>(integrand.error());
    }
    // The closed form ∮ p(q,E) dq is generally non-elementary (p is irrational in q, outside
    // the rational domain of nimblecas.integrate). We return the exact integrand and leave the
    // closed form absent rather than fabricate one — the honest boundary for action-angle.
    return ActionIntegral{.integrand = std::move(*integrand),
                          .coordinate = coords.q[0],
                          .closed_form = std::nullopt};
}

auto angle_variable(const Expr& H, const Coordinates& coords, const Expr& energy)
    -> Result<Expr> {
    // θ = ∂W/∂J with W = ∫ p dq. It is elementary only when the action's closed form exists;
    // since action_integral leaves that absent for the non-elementary (irrational) integrand,
    // the angle variable is honestly not_implemented here.
    auto action = action_integral(H, coords, energy);
    if (!action) {
        return make_error<Expr>(action.error());
    }
    if (!action->closed_form) {
        return make_error<Expr>(MathError::not_implemented);
    }
    // A closed-form action W(q,E) and J(E) would give θ = ∂W/∂J; reached only when the
    // integral above is elementary.
    return make_error<Expr>(MathError::not_implemented);
}

}  // namespace nimblecas
