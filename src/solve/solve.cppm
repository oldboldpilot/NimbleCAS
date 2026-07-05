// NimbleCAS analytical equation solving: exact closed-form polynomial roots (ROADMAP 7.21).
// @author Olumuyiwa Oluwasanmi
//
// Where nimblecas.roots returns only the RATIONAL roots of a polynomial, this module
// returns the full set of roots in exact closed form, as symbolic Expr values. It first
// peels the rational roots off (via rational_roots, deflating (x - r) with the required
// multiplicity), then dispatches on the degree of the remaining factor to the classical
// radical solutions:
//
//   - degree 1 (linear)    -b/a.
//   - degree 2 (quadratic) (-b +/- sqrt(b^2 - 4ac)) / (2a).
//   - degree 3 (cubic)     Cardano: depress x = t - b/(3a) to t^3 + p t + q, then combine
//                          cube-root radicals with the primitive cube roots of unity.
//   - degree 4 (quartic)   Ferrari: depress to t^4 + p t^2 + q t + r, solve the resolvent
//                          cubic with the routine above, and split into two quadratics.
//
// Radicals are built as Expr::power(base, Expr::rational(1, n)); no numeric approximation
// is ever produced. A negative quantity under a square root is kept as the exact imaginary
// radical power(negative, 1/2) (so the two roots of x^2 + 1 are power(-1, 1/2) and its
// negation), and the cubic "casus irreducibilis" keeps its nested imaginary radicals
// symbolically rather than switching to trigonometric form — the result stays exact and
// algebraic. This is documented in docs/reference/solve.md.
//
// Degree >= 5 remainder: rather than refuse, the roots of the leftover factor are computed as
// the eigenvalues of its companion matrix, in double precision, via nimblecas.numeigen's
// companion_eigenvalues. These are NUMERICAL approximations, NOT exact algebraic values, and
// every result is tagged accordingly (see Root::exact below) so an approximation is never
// presented as exact. deg <= 4 factors, and any rational or <= quartic factor peeled off a
// higher-degree polynomial, stay EXACT.
//
// Note on the boundary: the numeric path is chosen purely on the DEGREE of the factor left
// after rational-root extraction, NOT on a proof of irreducibility. By Abel-Ruffini a general
// quintic-or-higher has no radical solution, but a PARTICULAR degree >= 5 factor may still be
// reducible and radical-solvable (e.g. (x^2-2)(x^3-2), which has no rational roots yet exact
// radicals). Such a factor is currently returned numerically: the pre-factoring stops at
// rational roots and does NOT attempt square-free / higher-degree factorization to recover
// those radicals. So "returned numerically" means "not extracted exactly here", not "proven
// to have no closed form". Extending the pre-factoring is future work.
//
// HONESTY (Rule 32): every operation returns Result<T>; nothing throws, and no
// plausible-but-wrong value is ever presented as exact — numeric eigenvalue roots carry
// exact = false and are documented as companion-matrix eigenvalues computed iteratively to a
// tolerance. Exact int64 rational arithmetic is overflow-checked and surfaces
// MathError::overflow on a boundary; the zero polynomial (every value is a root) is
// MathError::domain_error.

export module nimblecas.solve;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.roots;
import nimblecas.symbolic;
import nimblecas.numeigen;

export namespace nimblecas {

// One root of a polynomial.
//   value        the root as a symbolic Expr (a rational, a radical, or a numeric leaf).
//   exact        true  -> value is an EXACT algebraic root (rational or a radical closed form);
//                false -> value is a NUMERICAL companion-matrix eigenvalue (deg >= 5 remainder),
//                         an approximation to the stated tolerance, never an exact value.
//   multiplicity the algebraic multiplicity when known (rational roots carry it); nullopt for
//                the flattened radical / numeric roots, which are listed one entry per root.
struct Root {
    Expr value;
    bool exact;
    std::optional<std::uint64_t> multiplicity;
};

// All roots of p. Rational roots (peeled by rational_roots) and the roots of a remaining
// factor of degree <= 4 (linear / quadratic / cubic / quartic) are returned EXACT, as
// symbolic radicals. A remaining factor of degree >= 5 that is not solvable in radicals has
// its roots returned NUMERICALLY as companion-matrix eigenvalues (via nimblecas.numeigen),
// each tagged exact = false. `tol` classifies an eigenvalue as real (|imag| < tol) vs a
// complex conjugate and bounds the iteration; `max_iter` caps it. The zero polynomial is
// MathError::domain_error; a nonzero constant has no roots (empty vector).
[[nodiscard]] auto solve_poly(const RationalPoly& p, double tol = 1e-12,
                              std::size_t max_iter = 1000) -> Result<std::vector<Root>>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================

// Unwrap a Result or propagate its error as make_error<RetType>. Each enclosing function
// defines `using RetType = ...;` before its first use. Variadic so the wrapped expression
// may itself contain commas (e.g. a multi-argument call).
#define TRY(var, ...)                                                    \
    auto var##__r = (__VA_ARGS__);                                       \
    if (!(var##__r)) return make_error<RetType>((var##__r).error());     \
    auto var = std::move(*var##__r)

namespace nimblecas {
namespace {

// -1 * e, used to negate an Expr without a dedicated subtraction node.
[[nodiscard]] auto neg(const Expr& e) -> Expr {
    return Expr::product({Expr::integer(-1), e});
}

// A rational as an Expr: an integer leaf when integral (cleaner trees), else a rational
// leaf. Fails only if Expr::rational rejects an int64 boundary (Rule 32).
[[nodiscard]] auto ex_rat(const Rational& r) -> Result<Expr> {
    if (r.is_integer()) {
        return Expr::integer(r.numerator());
    }
    return Expr::rational(r.numerator(), r.denominator());
}

// Constant symbolic sub-expressions reused across the closed forms.
struct Consts {
    Expr half;     // 1/2
    Expr third;    // 1/3
    Expr quarter;  // 1/4
    Expr omega;    // primitive cube root of unity (-1 + sqrt(-3)) / 2
    Expr omega2;   // its conjugate/square         (-1 - sqrt(-3)) / 2
    Expr i_unit;   // imaginary unit               sqrt(-1) = power(-1, 1/2)
};

[[nodiscard]] auto make_consts() -> Result<Consts> {
    using RetType = Consts;
    TRY(half, Expr::rational(1, 2));
    TRY(third, Expr::rational(1, 3));
    TRY(quarter, Expr::rational(1, 4));
    const Expr negone = Expr::integer(-1);
    const Expr sqrt_m3 = Expr::power(Expr::integer(-3), half);
    const Expr omega = Expr::product({half, Expr::sum({negone, sqrt_m3})});
    const Expr omega2 = Expr::product({half, Expr::sum({negone, neg(sqrt_m3)})});
    const Expr i_unit = Expr::power(Expr::integer(-1), half);
    return Consts{.half = half,
                  .third = third,
                  .quarter = quarter,
                  .omega = omega,
                  .omega2 = omega2,
                  .i_unit = i_unit};
}

// Roots of the monic quadratic t^2 + a1 t + a0 with symbolic (Expr) coefficients, via the
// quadratic formula. Pure tree construction (no fallible arithmetic): used by Ferrari to
// solve the two quadratic factors whose coefficients are themselves radicals.
[[nodiscard]] auto symbolic_quadratic(const Expr& a1, const Expr& a0, const Consts& c)
    -> std::array<Expr, 2> {
    const Expr disc =
        Expr::sum({Expr::power(a1, Expr::integer(2)), Expr::product({Expr::integer(-4), a0})});
    const Expr sq = Expr::power(disc, c.half);
    const Expr nega1 = neg(a1);
    return {Expr::product({Expr::sum({nega1, sq}), c.half}),
            Expr::product({Expr::sum({nega1, neg(sq)}), c.half})};
}

// --- degree-directed closed forms (coefficients low-degree-first, a_k the leading one) ---

// a1 x + a0 = 0 -> {-a0/a1}. (In practice a degree-1 factor always has a rational root and
// is handled by extraction; this path is kept for completeness/robustness.)
[[nodiscard]] auto solve_linear(const Rational& a1, const Rational& a0)
    -> Result<std::vector<Expr>> {
    using RetType = std::vector<Expr>;
    TRY(ratio, a0.divide(a1));
    TRY(root, ratio.negate());
    TRY(e, ex_rat(root));
    return std::vector<Expr>{e};
}

// a x^2 + b x + c = 0. With b == 0 the roots are the clean +/- sqrt(-c/a); otherwise the
// general formula (-b +/- sqrt(b^2 - 4ac)) / (2a). A negative radicand is kept symbolically.
[[nodiscard]] auto solve_quadratic(const Rational& a, const Rational& b, const Rational& c)
    -> Result<std::vector<Expr>> {
    using RetType = std::vector<Expr>;
    TRY(consts, make_consts());
    if (b.is_zero()) {
        TRY(c_over_a, c.divide(a));
        TRY(val, c_over_a.negate());  // -c/a
        TRY(base, ex_rat(val));
        const Expr s = Expr::power(base, consts.half);
        return std::vector<Expr>{s, neg(s)};
    }
    const Rational two = Rational::from_int(2);
    const Rational four = Rational::from_int(4);
    TRY(b2, b.multiply(b));
    TRY(ac, a.multiply(c));
    TRY(fourac, ac.multiply(four));
    TRY(disc, b2.subtract(fourac));
    TRY(negb, b.negate());
    TRY(two_a, a.multiply(two));
    TRY(inv2a, Rational::from_int(1).divide(two_a));
    TRY(disc_e, ex_rat(disc));
    TRY(negb_e, ex_rat(negb));
    TRY(inv2a_e, ex_rat(inv2a));
    const Expr sq = Expr::power(disc_e, consts.half);
    const Expr rplus = Expr::product({Expr::sum({negb_e, sq}), inv2a_e});
    const Expr rminus = Expr::product({Expr::sum({negb_e, neg(sq)}), inv2a_e});
    return std::vector<Expr>{rplus, rminus};
}

// a3 x^3 + a2 x^2 + a1 x + a0 = 0 by Cardano. Depress x = t - B/3 (B = a2/a3) to
// t^3 + p t + q, solve for t, then shift back. p == 0 and q == 0 are handled directly for
// clean radicals; the general branch uses t_k = w^k C + w^{2k} v with v = -p/(3C), which
// pins the cube-root branch so C*v = -p/3 holds automatically (exact for any fixed C).
[[nodiscard]] auto solve_cubic(const Rational& a3, const Rational& a2, const Rational& a1,
                               const Rational& a0) -> Result<std::vector<Expr>> {
    using RetType = std::vector<Expr>;
    TRY(consts, make_consts());
    const Rational two = Rational::from_int(2);
    const Rational three = Rational::from_int(3);
    const Rational four = Rational::from_int(4);
    const Rational twentyseven = Rational::from_int(27);

    TRY(B, a2.divide(a3));
    TRY(C, a1.divide(a3));
    TRY(D, a0.divide(a3));
    TRY(B2, B.multiply(B));
    TRY(B2_3, B2.divide(three));
    TRY(p, C.subtract(B2_3));  // p = C - B^2/3
    TRY(B3, B2.multiply(B));
    TRY(twoB3, B3.multiply(two));
    TRY(twoB3_27, twoB3.divide(twentyseven));
    TRY(BC, B.multiply(C));
    TRY(BC_3, BC.divide(three));
    TRY(q_part, twoB3_27.subtract(BC_3));
    TRY(q, q_part.add(D));  // q = 2B^3/27 - BC/3 + D
    TRY(negB, B.negate());
    TRY(shift, negB.divide(three));  // x = t - B/3

    std::vector<Expr> ts;
    if (p.is_zero()) {
        // t^3 = -q  ->  t_k = cbrt(-q) * w^k.
        TRY(negq, q.negate());
        TRY(cb_base, ex_rat(negq));
        const Expr cb = Expr::power(cb_base, consts.third);
        ts = {cb, Expr::product({consts.omega, cb}), Expr::product({consts.omega2, cb})};
    } else if (q.is_zero()) {
        // t^3 + p t = t (t^2 + p)  ->  t = 0, +/- sqrt(-p).
        TRY(negp, p.negate());
        TRY(negp_e, ex_rat(negp));
        const Expr s = Expr::power(negp_e, consts.half);
        ts = {Expr::integer(0), s, neg(s)};
    } else {
        // C = cbrt(-q/2 + sqrt(q^2/4 + p^3/27));  v = -p/3 * C^-1;  t_k = w^k C + w^{2k} v.
        TRY(q2, q.multiply(q));
        TRY(q2_4, q2.divide(four));
        TRY(p2, p.multiply(p));
        TRY(p3, p2.multiply(p));
        TRY(p3_27, p3.divide(twentyseven));
        TRY(disc_inner, q2_4.add(p3_27));
        TRY(q_2, q.divide(two));
        TRY(negq2, q_2.negate());
        TRY(p_3, p.divide(three));
        TRY(negp3, p_3.negate());
        TRY(disc_e, ex_rat(disc_inner));
        TRY(negq2_e, ex_rat(negq2));
        TRY(negp3_e, ex_rat(negp3));
        const Expr inner_sqrt = Expr::power(disc_e, consts.half);
        const Expr Cexpr = Expr::power(Expr::sum({negq2_e, inner_sqrt}), consts.third);
        const Expr Cinv = Expr::power(Cexpr, Expr::integer(-1));
        const Expr v = Expr::product({negp3_e, Cinv});
        ts = {Expr::sum({Cexpr, v}),
              Expr::sum({Expr::product({consts.omega, Cexpr}), Expr::product({consts.omega2, v})}),
              Expr::sum({Expr::product({consts.omega2, Cexpr}), Expr::product({consts.omega, v})})};
    }

    if (shift.is_zero()) {
        return ts;
    }
    TRY(shift_e, ex_rat(shift));
    std::vector<Expr> out;
    out.reserve(ts.size());
    for (const Expr& t : ts) {
        out.push_back(Expr::sum({t, shift_e}));
    }
    return out;
}

// a4 x^4 + ... + a0 = 0 by Ferrari. Depress x = t - B/4 (B = a3/a4) to t^4 + p t^2 + q t + r.
// q == 0 is biquadratic (t^2 solved as a quadratic; a pure t^4 = -r gives clean fourth
// roots). Otherwise pick a nonzero root y of the resolvent cubic
// y^3 + 2p y^2 + (p^2 - 4r) y - q^2 = 0 and factor into
//   (t^2 - A t + (M - B0)) (t^2 + A t + (M + B0)),  A = sqrt(y), B0 = -q/(2A), M = (p+y)/2,
// an exact identity for every root y (see the reference doc's derivation).
[[nodiscard]] auto solve_quartic(const Rational& a4, const Rational& a3, const Rational& a2,
                                 const Rational& a1, const Rational& a0)
    -> Result<std::vector<Expr>> {
    using RetType = std::vector<Expr>;
    TRY(consts, make_consts());
    const Rational two = Rational::from_int(2);
    const Rational three = Rational::from_int(3);
    const Rational four = Rational::from_int(4);
    const Rational eight = Rational::from_int(8);
    const Rational sixteen = Rational::from_int(16);
    const Rational two_five_six = Rational::from_int(256);

    TRY(B, a3.divide(a4));
    TRY(C, a2.divide(a4));
    TRY(D, a1.divide(a4));
    TRY(E, a0.divide(a4));
    TRY(B2, B.multiply(B));
    TRY(threeB2, B2.multiply(three));
    TRY(threeB2_8, threeB2.divide(eight));
    TRY(p, C.subtract(threeB2_8));  // p = C - 3B^2/8
    TRY(B3, B2.multiply(B));
    TRY(B3_8, B3.divide(eight));
    TRY(BC, B.multiply(C));
    TRY(BC_2, BC.divide(two));
    TRY(D_BC2, D.subtract(BC_2));
    TRY(q, D_BC2.add(B3_8));  // q = D - BC/2 + B^3/8
    TRY(BD, B.multiply(D));
    TRY(BD_4, BD.divide(four));
    TRY(E_BD4, E.subtract(BD_4));
    TRY(B2C, B2.multiply(C));
    TRY(B2C_16, B2C.divide(sixteen));
    TRY(r_part, E_BD4.add(B2C_16));
    TRY(B4, B2.multiply(B2));
    TRY(threeB4, B4.multiply(three));
    TRY(threeB4_256, threeB4.divide(two_five_six));
    TRY(r, r_part.subtract(threeB4_256));  // r = E - BD/4 + B^2 C/16 - 3B^4/256
    TRY(negB, B.negate());
    TRY(shift, negB.divide(four));  // x = t - B/4

    std::vector<Expr> ts;
    if (q.is_zero()) {
        if (p.is_zero()) {
            // t^4 = -r  ->  fourth roots: u * {1, i, -1, -i}.
            TRY(negr, r.negate());
            TRY(negr_e, ex_rat(negr));
            const Expr u = Expr::power(negr_e, consts.quarter);
            ts = {u, Expr::product({consts.i_unit, u}), neg(u),
                  Expr::product({Expr::integer(-1), consts.i_unit, u})};
        } else {
            // t^4 + p t^2 + r = 0:  t^2 = (-p +/- sqrt(p^2 - 4r))/2,  then t = +/- sqrt(t^2).
            TRY(p2, p.multiply(p));
            TRY(fourr, r.multiply(four));
            TRY(disc, p2.subtract(fourr));
            TRY(disc_e, ex_rat(disc));
            TRY(negp, p.negate());
            TRY(negp_e, ex_rat(negp));
            const Expr sq = Expr::power(disc_e, consts.half);
            const Expr t2a = Expr::product({Expr::sum({negp_e, sq}), consts.half});
            const Expr t2b = Expr::product({Expr::sum({negp_e, neg(sq)}), consts.half});
            const Expr ra = Expr::power(t2a, consts.half);
            const Expr rb = Expr::power(t2b, consts.half);
            ts = {ra, neg(ra), rb, neg(rb)};
        }
    } else {
        // General Ferrari via the resolvent cubic.
        TRY(q2, q.multiply(q));
        TRY(negq2, q2.negate());  // resolvent constant term -q^2 (nonzero, so y != 0)
        TRY(p2, p.multiply(p));
        TRY(fourr, r.multiply(four));
        TRY(p2_4r, p2.subtract(fourr));
        TRY(twop, p.multiply(two));
        const Rational one = Rational::from_int(1);
        TRY(ycube, solve_cubic(one, twop, p2_4r, negq2));
        const Expr y = ycube[0];
        const Expr A = Expr::power(y, consts.half);
        TRY(q_2, q.divide(two));
        TRY(negq_2, q_2.negate());
        TRY(negq_2_e, ex_rat(negq_2));
        TRY(p_2, p.divide(two));
        TRY(p_2_e, ex_rat(p_2));
        const Expr B0 = Expr::product({negq_2_e, Expr::power(A, Expr::integer(-1))});  // -q/(2A)
        const Expr M = Expr::sum({p_2_e, Expr::product({consts.half, y})});            // (p+y)/2
        const std::array<Expr, 2> quad1 =
            symbolic_quadratic(neg(A), Expr::sum({M, neg(B0)}), consts);
        const std::array<Expr, 2> quad2 = symbolic_quadratic(A, Expr::sum({M, B0}), consts);
        ts = {quad1[0], quad1[1], quad2[0], quad2[1]};
    }

    if (shift.is_zero()) {
        return ts;
    }
    TRY(shift_e, ex_rat(shift));
    std::vector<Expr> out;
    out.reserve(ts.size());
    for (const Expr& t : ts) {
        out.push_back(Expr::sum({t, shift_e}));
    }
    return out;
}

// Roots of the factor left after rational-root extraction, dispatched on its degree.
[[nodiscard]] auto solve_remaining(const RationalPoly& w) -> Result<std::vector<Expr>> {
    using RetType = std::vector<Expr>;
    switch (w.degree()) {
        case 1:
            return solve_linear(w.coefficient(1), w.coefficient(0));
        case 2:
            return solve_quadratic(w.coefficient(2), w.coefficient(1), w.coefficient(0));
        case 3:
            return solve_cubic(w.coefficient(3), w.coefficient(2), w.coefficient(1),
                               w.coefficient(0));
        case 4:
            return solve_quartic(w.coefficient(4), w.coefficient(3), w.coefficient(2),
                                 w.coefficient(1), w.coefficient(0));
        default:
            return make_error<std::vector<Expr>>(MathError::not_implemented);
    }
}

}  // namespace

auto solve_poly(const RationalPoly& p, double tol, std::size_t max_iter)
    -> Result<std::vector<Root>> {
    using RetType = std::vector<Root>;
    if (p.is_zero()) {
        return make_error<std::vector<Root>>(MathError::domain_error);  // every value is a root
    }
    std::vector<Root> roots;
    if (p.degree() == 0) {
        return roots;  // nonzero constant: no roots
    }

    // Peel off the rational roots (exact), deflating (x - r) with multiplicity.
    const Rational one = Rational::from_int(1);
    TRY(rr, rational_roots(p));
    RationalPoly work = p;
    for (const auto& [r, mult] : rr) {
        TRY(re, ex_rat(r));
        roots.push_back(
            Root{.value = re, .exact = true, .multiplicity = static_cast<std::uint64_t>(mult)});
        TRY(negr, r.negate());
        const RationalPoly factor = RationalPoly::from_coeffs({negr, one});  // x - r
        for (std::int64_t k = 0; k < mult; ++k) {
            TRY(dm, work.divide(factor));
            work = std::move(dm.quotient);
        }
    }

    const std::int64_t d = work.degree();
    if (d <= 0) {  // fully split into rational roots
        return roots;
    }

    if (d <= 4) {  // exact radicals for the remaining linear/quadratic/cubic/quartic factor
        TRY(closed, solve_remaining(work));
        for (Expr& e : closed) {
            roots.push_back(
                Root{.value = std::move(e), .exact = true, .multiplicity = std::nullopt});
        }
        return roots;
    }

    // Degree >= 5 factor (after rational-root peeling): numeric companion-matrix eigenvalues.
    // Chosen on degree, not on a proof of irreducibility (see the header note). Each root is
    // an approximation (exact = false). A near-real eigenvalue (|imag| < tol) becomes a real
    // leaf; otherwise re + im*i in the module's imaginary convention (i = power(-1, 1/2)).
    TRY(consts, make_consts());
    TRY(evals, companion_eigenvalues(work, tol, max_iter));
    for (const std::complex<double>& z : evals) {
        Expr value =
            std::abs(z.imag()) < tol
                ? Expr::real(z.real())
                : Expr::sum({Expr::real(z.real()),
                             Expr::product({Expr::real(z.imag()), consts.i_unit})});
        roots.push_back(
            Root{.value = std::move(value), .exact = false, .multiplicity = std::nullopt});
    }
    return roots;
}

}  // namespace nimblecas

#undef TRY
