// NimbleCAS splitting-field construction over a number field: Trager's norm-based
// factorization of AlgebraicPoly over a NumberField, adjunction of a single root of an
// irreducible extension polynomial via a primitive-element search, and iterated
// adjunction to build the splitting field of a batch of Q-irreducible polynomials
// (§7 algebra substrate; builds on nimblecas.algnum / nimblecas.algpoly).
// @author Olumuyiwa Oluwasanmi
//
// FACTOR_OVER_FIELD (Trager's norm algorithm). Given L = Q(gamma) = Q[t]/(h(t)) and a
// squarefree f in L[x], pick an integer shift s from the deterministic sequence
// 0, 1, -1, 2, -2, ... and set g(x) = f(x - s*gamma). The norm N(x) = Norm_{L/Q}(g) is a
// polynomial in Q[x] of degree [L:Q]*deg(f); since h is monic, Norm_{L/Q}(b) for a field
// element b (represented as a degree-<[L:Q] residue polynomial in t) is exactly the plain
// resultant Res_t(h(t), b(t)) -- no leading-coefficient normalisation is needed. N(x) is
// recovered by evaluation-interpolation: sample g at deg(N)+1 rational points x0 (writing
// g(x0, t) as a Q[t] polynomial via the field elements' own residues), take the resultant
// against h at each sample, and Lagrange-interpolate. If N is not squarefree the shift was
// unlucky; the search retries the next s up to the bound |s| <= 2*[L:Q]*deg(f). Once N is
// squarefree, factor_over_Q(N) gives its Q-irreducible pieces N_i, and the corresponding
// L-factor of g is gcd_L(g, embed_L(N_i)); shifting back by x -> x + s*gamma and
// normalising monic recovers the L-factors of f. The reconstruction f == lc(f) * prod(f_i)
// is re-verified before returning (Rule 32): a mismatch is an embedding bug, not a
// mathematical possibility, and is reported as domain_error rather than presented as fact.
//
// ADJOIN_ROOT (simple-extension primitive-element search). Given g irreducible over L of
// degree e >= 2, the Q-algebra M0 = L[y]/(g(y)) has Q-basis {alpha^i * beta^j : 0<=i<[L:Q],
// 0<=j<e} (alpha = L's generator, beta = the class of y), dimension n = [L:Q]*e. A
// candidate primitive element gamma' = beta + c*alpha (c = 1, -1, 2, -2, ... in a bounded
// search) has its Q-linear multiplication matrix built on that basis; gamma' is primitive
// iff that matrix's minimal polynomial (via nimblecas.frobenius) has degree exactly n. The
// minimal polynomial h' is then automatically irreducible when M0 truly is a field -- and
// conversely, if g were NOT actually irreducible over L, M0 is a product of smaller fields,
// and any degree-n minimal polynomial found there is necessarily REDUCIBLE (Q[z]/(minpoly)
// would be isomorphic to the non-field M0), so NumberField::create(h') honestly rejects it.
// M = Q[z]/(h') is built, then alpha and beta are recovered as Q-polynomials in gamma' by
// solving the (invertible, since gamma' is primitive) linear system expressing the powers
// gamma'^0..gamma'^(n-1) against the targets "alpha" and "beta" in the same coordinates.
// Before returning, both g(root) == 0 and L.modulus()(old_generator) == 0 are indepedently
// re-verified in M (Rule 32): an embedding bug is caught at the source, not downstream.
//
// SPLITTING_FIELD. Input factors are sorted into a canonical order (degree ascending, then
// coefficient-lexicographic) purely to make the field-growth sequence deterministic; results
// are reported back in the CALLER's original order. The starting field is Q[x]/(first
// non-linear factor in canonical order), or the trivial degree-1 field Q[t]/(t) (representing
// Q itself) when every input factor is already linear. Each outer pass re-embeds and
// re-factors EVERY input factor over the current field via factor_over_field (a fresh
// recomputation each pass rather than incremental bookkeeping -- simpler and no less
// correct at the small scale this module targets) and picks the first non-linear
// irreducible piece encountered as the next adjunction target; adjoin_root extends the
// field and the pass repeats. The loop stops when every factor splits into exactly deg_i
// distinct linear pieces, which is re-verified (Rule 32) before the roots are reported.
//
// HONESTY (Rule 32). Nothing throws; nothing returns a plausible-but-wrong field or root.
// [L:Q] exceeding max_degree at any point -- the initial field, or a subsequent
// adjunction -- is MathError::not_implemented (a quartic with Galois group S4 needs degree
// 24; this module refuses rather than guess). Exhausting the shift bound in Trager's
// algorithm (no squarefree norm found) or the primitive-element search bound in adjoin_root
// is MathError::not_implemented. factor_over_Q's own internal budget and any int64 Rational
// overflow propagate as not_implemented / overflow respectively. The practical envelope, in
// line with factor_over_Q and NumberField, is roughly degree <= 6 with small coefficients;
// beyond that, an honest not_implemented is preferred over a wrong answer.

module;
#include <cassert>

export module nimblecas.splitfield;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.matrix;
import nimblecas.algnum;
import nimblecas.algpoly;
import nimblecas.factor;
import nimblecas.resultant;
import nimblecas.frobenius;

export namespace nimblecas {

// Factor a SQUAREFREE f in L[x] into its monic L-irreducible factors, via Trager's norm
// algorithm. f must live in `l` (else domain_error); the zero polynomial is domain_error.
// A nonzero constant f yields an empty list (no non-unit factors); a linear f yields itself
// (monic). Fails with MathError::not_implemented when no shift in the bounded search
// |s| <= 2*[L:Q]*deg(f) yields a squarefree norm, or when factor_over_Q(N) exceeds its own
// budget; with MathError::overflow on an int64 arithmetic boundary. The reconstruction
// f == lc(f) * prod(factors) is verified before returning (domain_error on mismatch).
[[nodiscard]] auto factor_over_field(const NumberField& l, const AlgebraicPoly& f)
    -> Result<std::vector<AlgebraicPoly>>;

// The roots of f that lie in L, read off the degree-1 factors of factor_over_field(l, f)
// as lambda = -c0/c1. Order matches factor_over_field's (unspecified but deterministic)
// factor order. Errors propagate from factor_over_field.
[[nodiscard]] auto roots_in_field(const NumberField& l, const AlgebraicPoly& f)
    -> Result<std::vector<AlgebraicNumber>>;

// A simple extension M >= L containing a root of g, plus the images in M of L's generator
// (`old_generator`, needed to re-embed any other element of L into M) and of the adjoined
// root itself (`root`).
struct AdjoinedRoot {
    NumberField field;
    AlgebraicNumber old_generator;
    AlgebraicNumber root;
};

// Adjoin a root of g (irreducible over L, deg g = e >= 2) to L, building M = Q(gamma') as a
// SIMPLE extension via a primitive-element search gamma' = beta + c*alpha over
// c = 1, -1, 2, -2, .... Fails with domain_error when g does not live in `l` or has degree
// < 2; with MathError::not_implemented when no primitive element is found within the
// bounded search (this is also the honest signal if g was not actually irreducible over L
// -- see the module header); with overflow on an int64 arithmetic boundary; with
// domain_error if the mandatory post-construction verification (g(root) == 0 and
// L.modulus()(old_generator) == 0, both checked in M) fails.
[[nodiscard]] auto adjoin_root(const NumberField& l, const AlgebraicPoly& g)
    -> Result<AdjoinedRoot>;

// The splitting field of a batch of Q-irreducible polynomials, together with each
// polynomial's full set of roots in that field (in the SAME order as `irreducibles`).
struct SplittingField {
    NumberField field;
    std::vector<std::pair<RationalPoly, std::vector<AlgebraicNumber>>> roots;
};

// Build the splitting field of `irreducibles` (each assumed Q-irreducible; a non-positive
// degree entry is domain_error), refusing to build a field of degree exceeding
// `max_degree` at any point (the initial field or any subsequent adjunction) with
// MathError::not_implemented -- the honest boundary, since e.g. a quartic with Galois
// group S4 needs degree 24. Errors from factor_over_field / adjoin_root propagate
// (not_implemented / overflow / domain_error). A defensive completeness guard (every
// factor contributes exactly deg_i pairwise-distinct roots) is re-verified before
// returning; a violation is domain_error (an embedding bug, not a mathematical
// possibility for a genuinely Q-irreducible, hence separable, input).
[[nodiscard]] auto splitting_field(std::span<const RationalPoly> irreducibles,
                                    std::int64_t max_degree) -> Result<SplittingField>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// --- small arithmetic helpers ------------------------------------------------

[[nodiscard]] auto checked_mul_i64(std::int64_t a, std::int64_t b) -> std::optional<std::int64_t> {
    std::int64_t out = 0;
    if (__builtin_mul_overflow(a, b, &out)) {
        return std::nullopt;
    }
    return out;
}

// Exact a < b for canonical Rationals (den > 0 by Rational's own invariant), via 128-bit
// cross-multiplication so no int64 overflow guard is needed. Used only to establish a
// deterministic sort order, never as a computed mathematical result.
[[nodiscard]] auto rational_less(const Rational& a, const Rational& b) -> bool {
    const __int128 lhs = static_cast<__int128>(a.numerator()) * static_cast<__int128>(b.denominator());
    const __int128 rhs = static_cast<__int128>(b.numerator()) * static_cast<__int128>(a.denominator());
    return lhs < rhs;
}

// Canonical order over RationalPoly: degree ascending, then coefficient-lexicographic
// (constant term first). Used only to make the outer field-growth algorithm deterministic.
[[nodiscard]] auto poly_less(const RationalPoly& a, const RationalPoly& b) -> bool {
    if (a.degree() != b.degree()) {
        return a.degree() < b.degree();
    }
    for (std::int64_t i = 0; i <= a.degree(); ++i) {
        const Rational ca = a.coefficient(static_cast<std::size_t>(i));
        const Rational cb = b.coefficient(static_cast<std::size_t>(i));
        if (!(ca == cb)) {
            return rational_less(ca, cb);
        }
    }
    return false;
}

// The deterministic rational sample sequence 0, 1, -1, 2, -2, ..., `count` values.
[[nodiscard]] auto sample_nodes(std::size_t count) -> std::vector<Rational> {
    std::vector<Rational> nodes;
    nodes.reserve(count);
    if (count == 0) {
        return nodes;
    }
    nodes.push_back(Rational::from_int(0));
    std::int64_t k = 1;
    while (nodes.size() < count) {
        nodes.push_back(Rational::from_int(k));
        if (nodes.size() < count) {
            nodes.push_back(Rational::from_int(-k));
        }
        ++k;
    }
    return nodes;
}

// Lagrange-interpolate the unique polynomial of degree < nodes.size() through the given
// (node, value) pairs (nodes assumed distinct). Fails only on overflow.
[[nodiscard]] auto lagrange_interpolate(std::span<const Rational> nodes,
                                        std::span<const Rational> values)
    -> Result<RationalPoly> {
    const std::size_t m = nodes.size();
    RationalPoly acc;  // zero
    for (std::size_t i = 0; i < m; ++i) {
        if (values[i].is_zero()) {
            continue;
        }
        RationalPoly num = RationalPoly::constant(Rational::from_int(1));
        Rational denom = Rational::from_int(1);
        for (std::size_t j = 0; j < m; ++j) {
            if (j == i) {
                continue;
            }
            auto neg_nj = nodes[j].negate();
            if (!neg_nj) {
                return make_error<RationalPoly>(neg_nj.error());
            }
            const RationalPoly lin =
                RationalPoly::from_coeffs(std::vector<Rational>{*neg_nj, Rational::from_int(1)});
            auto next_num = num.multiply(lin);
            if (!next_num) {
                return make_error<RationalPoly>(next_num.error());
            }
            num = std::move(*next_num);
            auto diff = nodes[i].subtract(nodes[j]);
            if (!diff) {
                return make_error<RationalPoly>(diff.error());
            }
            auto nd = denom.multiply(*diff);
            if (!nd) {
                return make_error<RationalPoly>(nd.error());
            }
            denom = *nd;
        }
        auto inv_denom = Rational::from_int(1).divide(denom);  // denom != 0: distinct nodes
        if (!inv_denom) {
            return make_error<RationalPoly>(inv_denom.error());
        }
        auto coeff = values[i].multiply(*inv_denom);
        if (!coeff) {
            return make_error<RationalPoly>(coeff.error());
        }
        auto term = num.scale(*coeff);
        if (!term) {
            return make_error<RationalPoly>(term.error());
        }
        auto sum = acc.add(*term);
        if (!sum) {
            return make_error<RationalPoly>(sum.error());
        }
        acc = std::move(*sum);
    }
    return acc;
}

// --- AlgebraicPoly helpers ----------------------------------------------------

// f(x - a): polynomial composition with the linear shift (x - a), via Horner's scheme.
// a and f must live in the same field.
[[nodiscard]] auto shift_by(const AlgebraicPoly& f, const AlgebraicNumber& a)
    -> Result<AlgebraicPoly> {
    const NumberField field = f.field();
    auto neg_a = a.negate();
    if (!neg_a) {
        return make_error<AlgebraicPoly>(neg_a.error());
    }
    auto linear_res =
        AlgebraicPoly::from_coeffs(field, std::vector<AlgebraicNumber>{*neg_a, field.one()});
    if (!linear_res) {
        return make_error<AlgebraicPoly>(linear_res.error());
    }
    const AlgebraicPoly linear = std::move(*linear_res);
    AlgebraicPoly acc = AlgebraicPoly::zero(field);
    for (std::int64_t i = f.degree(); i >= 0; --i) {
        auto prod = acc.multiply(linear);
        if (!prod) {
            return make_error<AlgebraicPoly>(prod.error());
        }
        auto ci_res = AlgebraicPoly::from_coeffs(
            field, std::vector<AlgebraicNumber>{f.coefficient(static_cast<std::size_t>(i))});
        if (!ci_res) {
            return make_error<AlgebraicPoly>(ci_res.error());
        }
        auto sum = prod->add(*ci_res);
        if (!sum) {
            return make_error<AlgebraicPoly>(sum.error());
        }
        acc = std::move(*sum);
    }
    return acc;
}

// g(x0, t): the AlgebraicPoly g over L = Q[t]/(h(t)) evaluated at x = x0 (a plain
// rational), WITHOUT reducing the field generator further -- i.e. the Q[t] polynomial
// sum_i g_i(t) * x0^i where g_i(t) is the canonical residue of g's i-th coefficient. This
// is exactly the object whose resultant against h gives Norm_{L/Q}(g(x0)).
[[nodiscard]] auto eval_coeffs_at_x(const AlgebraicPoly& g, const Rational& x0)
    -> Result<RationalPoly> {
    RationalPoly acc;  // zero
    Rational power = Rational::from_int(1);
    const std::int64_t deg = g.degree();
    for (std::int64_t i = 0; i <= deg; ++i) {
        const AlgebraicNumber ci = g.coefficient(static_cast<std::size_t>(i));
        auto term = ci.value().scale(power);
        if (!term) {
            return make_error<RationalPoly>(term.error());
        }
        auto sum = acc.add(*term);
        if (!sum) {
            return make_error<RationalPoly>(sum.error());
        }
        acc = std::move(*sum);
        if (i < deg) {
            auto next_power = power.multiply(x0);
            if (!next_power) {
                return make_error<RationalPoly>(next_power.error());
            }
            power = *next_power;
        }
    }
    return acc;
}

// p reduced modulo the (nonzero) divisor g.
[[nodiscard]] auto reduce_mod(const AlgebraicPoly& p, const AlgebraicPoly& g)
    -> Result<AlgebraicPoly> {
    auto dm = p.divide(g);
    if (!dm) {
        return make_error<AlgebraicPoly>(dm.error());
    }
    return std::move(dm->remainder);
}

// Re-embed a field element x (living in some field whose generator maps to
// `image_of_generator` in `new_field`) into `new_field`, by lifting x's canonical residue
// polynomial and evaluating it at the generator's image. This is the ring homomorphism
// induced by generator |-> image_of_generator.
[[nodiscard]] auto reembed(const AlgebraicNumber& x, const NumberField& new_field,
                           const AlgebraicNumber& image_of_generator) -> Result<AlgebraicNumber> {
    const RationalPoly rep = x.value();
    const AlgebraicPoly lifted = AlgebraicPoly::embed(new_field, rep);
    return lifted.evaluate(image_of_generator);
}

// Coordinate map for adjoin_root's Q-algebra M0 = L[y]/(g): an M0 element, represented as
// an AlgebraicPoly p over L of degree < e (coefficient j is p's L-coefficient of y^j), is
// mapped to its n = d*e coordinates on the basis {alpha^i * beta^j}, index i*e + j.
[[nodiscard]] auto to_vector(const AlgebraicPoly& p, std::int64_t d, std::int64_t e)
    -> std::vector<Rational> {
    std::vector<Rational> vec(static_cast<std::size_t>(d) * static_cast<std::size_t>(e));
    for (std::int64_t j = 0; j < e; ++j) {
        const AlgebraicNumber cj = p.coefficient(static_cast<std::size_t>(j));
        const RationalPoly& val = cj.value();
        for (std::int64_t i = 0; i < d; ++i) {
            vec[static_cast<std::size_t>(i * e + j)] = val.coefficient(static_cast<std::size_t>(i));
        }
    }
    return vec;
}

}  // namespace

// --- factor_over_field --------------------------------------------------------

auto factor_over_field(const NumberField& l, const AlgebraicPoly& f)
    -> Result<std::vector<AlgebraicPoly>> {
    using Out = std::vector<AlgebraicPoly>;
    if (!f.field().is_same(l)) {
        return make_error<Out>(MathError::domain_error);
    }
    if (f.is_zero()) {
        return make_error<Out>(MathError::domain_error);
    }
    const std::int64_t deg_f = f.degree();
    if (deg_f == 0) {
        return Out{};  // nonzero constant: no non-unit factors
    }
    if (deg_f == 1) {
        auto fm = f.monic();
        if (!fm) {
            return make_error<Out>(fm.error());
        }
        return Out{std::move(*fm)};
    }

    const std::int64_t d = l.degree();
    auto gamma_res = l.generator();
    if (!gamma_res) {
        return make_error<Out>(gamma_res.error());
    }
    const AlgebraicNumber gamma = std::move(*gamma_res);
    const RationalPoly h = l.modulus();

    auto deg_n_opt = checked_mul_i64(d, deg_f);
    if (!deg_n_opt) {
        return make_error<Out>(MathError::overflow);
    }
    const std::int64_t deg_n = *deg_n_opt;  // deg N(x) = [L:Q] * deg f
    const std::int64_t bound = checked_mul_i64(2, deg_n).value_or(std::numeric_limits<std::int64_t>::max());

    for (std::int64_t k = 0;; ++k) {
        std::int64_t s = 0;
        if (k > 0) {
            s = (k % 2 == 1) ? (k + 1) / 2 : -(k / 2);
        }
        if (s > bound || s < -bound) {
            return make_error<Out>(MathError::not_implemented);
        }

        const AlgebraicNumber s_elem = l.from_rational(Rational::from_int(s));
        auto shift_amt = gamma.multiply(s_elem);
        if (!shift_amt) {
            return make_error<Out>(shift_amt.error());
        }
        auto g_res = shift_by(f, *shift_amt);  // g(x) = f(x - s*gamma)
        if (!g_res) {
            return make_error<Out>(g_res.error());
        }
        const AlgebraicPoly g = std::move(*g_res);

        const std::vector<Rational> nodes = sample_nodes(static_cast<std::size_t>(deg_n) + 1);
        std::vector<Rational> values;
        values.reserve(nodes.size());
        bool arith_error = false;
        MathError err = MathError::domain_error;
        for (const Rational& x0 : nodes) {
            auto gx0_t = eval_coeffs_at_x(g, x0);
            if (!gx0_t) {
                arith_error = true;
                err = gx0_t.error();
                break;
            }
            auto nx0 = resultant(h, *gx0_t);  // h monic => this IS Norm_{L/Q}(g(x0))
            if (!nx0) {
                arith_error = true;
                err = nx0.error();
                break;
            }
            values.push_back(std::move(*nx0));
        }
        if (arith_error) {
            return make_error<Out>(err);
        }

        auto n_res = lagrange_interpolate(nodes, values);
        if (!n_res) {
            return make_error<Out>(n_res.error());
        }
        const RationalPoly N = std::move(*n_res);

        // f's (hence g's) leading coefficient is a nonzero field element, so its norm is
        // nonzero and deg N must be exactly deg_n; a mismatch is an interpolation bug, not
        // a mathematical possibility, and is reported honestly rather than risked further.
        assert(N.degree() == deg_n && "Trager norm degree must equal [L:Q] * deg f");
        if (N.degree() != deg_n) {
            return make_error<Out>(MathError::domain_error);
        }

        auto n_prime = N.derivative();
        if (!n_prime) {
            return make_error<Out>(n_prime.error());
        }
        auto sqfree_gcd = N.gcd(*n_prime);
        if (!sqfree_gcd) {
            return make_error<Out>(sqfree_gcd.error());
        }
        if (sqfree_gcd->degree() > 0) {
            continue;  // N not squarefree for this shift: try the next s
        }

        auto n_factors = factor_over_Q(N);
        if (!n_factors) {
            return make_error<Out>(n_factors.error());  // propagate factor_over_Q's own budget
        }

        auto neg_shift = shift_amt->negate();
        if (!neg_shift) {
            return make_error<Out>(neg_shift.error());
        }

        Out result;
        result.reserve(n_factors->size());
        for (const auto& [ni, mult] : *n_factors) {
            (void)mult;  // N is squarefree here, so every multiplicity is 1
            const AlgebraicPoly ni_l = AlgebraicPoly::embed(l, ni);
            auto fac = g.gcd(ni_l);
            if (!fac) {
                return make_error<Out>(fac.error());
            }
            auto shifted_back = shift_by(*fac, *neg_shift);  // f_i(x) = fac(x + s*gamma)
            if (!shifted_back) {
                return make_error<Out>(shifted_back.error());
            }
            auto fim = shifted_back->monic();
            if (!fim) {
                return make_error<Out>(fim.error());
            }
            result.push_back(std::move(*fim));
        }

        // VERIFY (Rule 32): the recovered monic factors, scaled by f's leading
        // coefficient, must reconstruct f exactly.
        auto product_res = AlgebraicPoly::from_coeffs(l, std::vector<AlgebraicNumber>{l.one()});
        if (!product_res) {
            return make_error<Out>(product_res.error());
        }
        AlgebraicPoly product = std::move(*product_res);
        for (const AlgebraicPoly& fac : result) {
            auto p = product.multiply(fac);
            if (!p) {
                return make_error<Out>(p.error());
            }
            product = std::move(*p);
        }
        auto scaled = product.scale(f.leading_coefficient());
        if (!scaled) {
            return make_error<Out>(scaled.error());
        }
        if (!scaled->is_equal(f)) {
            return make_error<Out>(MathError::domain_error);  // embedding bug, not fabricated
        }
        return result;
    }
}

// --- roots_in_field -------------------------------------------------------------

auto roots_in_field(const NumberField& l, const AlgebraicPoly& f)
    -> Result<std::vector<AlgebraicNumber>> {
    using Out = std::vector<AlgebraicNumber>;
    auto factors = factor_over_field(l, f);
    if (!factors) {
        return make_error<Out>(factors.error());
    }
    Out roots;
    for (const AlgebraicPoly& fac : *factors) {
        if (fac.degree() != 1) {
            continue;
        }
        const AlgebraicNumber c0 = fac.coefficient(0);
        const AlgebraicNumber c1 = fac.coefficient(1);
        auto neg_c0 = c0.negate();
        if (!neg_c0) {
            return make_error<Out>(neg_c0.error());
        }
        auto root = neg_c0->divide(c1);  // c1 != 0: fac has degree exactly 1
        if (!root) {
            return make_error<Out>(root.error());
        }
        roots.push_back(std::move(*root));
    }
    return roots;
}

// --- adjoin_root ------------------------------------------------------------------

auto adjoin_root(const NumberField& l, const AlgebraicPoly& g) -> Result<AdjoinedRoot> {
    if (!g.field().is_same(l)) {
        return make_error<AdjoinedRoot>(MathError::domain_error);
    }
    const std::int64_t e = g.degree();
    if (e < 2) {
        return make_error<AdjoinedRoot>(MathError::domain_error);
    }
    const std::int64_t d = l.degree();
    auto n_opt = checked_mul_i64(d, e);
    if (!n_opt) {
        return make_error<AdjoinedRoot>(MathError::overflow);
    }
    const std::int64_t n = *n_opt;
    const std::size_t un = static_cast<std::size_t>(n);

    auto alpha_res = l.generator();
    if (!alpha_res) {
        return make_error<AdjoinedRoot>(alpha_res.error());
    }
    const AlgebraicNumber alpha = std::move(*alpha_res);

    // target_alpha: the M0 = L[y]/(g) element equal to alpha itself (y^0 coefficient).
    auto target_alpha_res = AlgebraicPoly::from_coeffs(l, std::vector<AlgebraicNumber>{alpha});
    if (!target_alpha_res) {
        return make_error<AdjoinedRoot>(target_alpha_res.error());
    }
    // target_beta: the M0 element equal to y (= beta) itself.
    auto target_beta_res =
        AlgebraicPoly::from_coeffs(l, std::vector<AlgebraicNumber>{l.zero(), l.one()});
    if (!target_beta_res) {
        return make_error<AdjoinedRoot>(target_beta_res.error());
    }
    const std::vector<Rational> target_alpha_vec = to_vector(*target_alpha_res, d, e);
    const std::vector<Rational> target_beta_vec = to_vector(*target_beta_res, d, e);

    constexpr std::int64_t kMaxPrimitiveSearch = 64;
    for (std::int64_t k = 1; k <= kMaxPrimitiveSearch; ++k) {
        const std::int64_t c = (k % 2 == 1) ? (k + 1) / 2 : -(k / 2);  // 1, -1, 2, -2, ...

        const AlgebraicNumber c_elem = l.from_rational(Rational::from_int(c));
        auto c_alpha = alpha.multiply(c_elem);
        if (!c_alpha) {
            return make_error<AdjoinedRoot>(c_alpha.error());
        }
        auto gamma_prime_res =
            AlgebraicPoly::from_coeffs(l, std::vector<AlgebraicNumber>{*c_alpha, l.one()});
        if (!gamma_prime_res) {
            return make_error<AdjoinedRoot>(gamma_prime_res.error());
        }
        const AlgebraicPoly gamma_prime = std::move(*gamma_prime_res);

        // The n x n rational matrix of multiplication-by-gamma' on the Q-basis
        // { alpha^i * beta^j : 0 <= i < d, 0 <= j < e }, column index i*e + j.
        std::vector<std::vector<Rational>> rows(un, std::vector<Rational>(un));
        bool build_error = false;
        MathError berr = MathError::domain_error;
        for (std::int64_t i = 0; i < d && !build_error; ++i) {
            auto a_i = alpha.pow(i);
            if (!a_i) {
                build_error = true;
                berr = a_i.error();
                break;
            }
            for (std::int64_t j = 0; j < e && !build_error; ++j) {
                std::vector<AlgebraicNumber> basis_coeffs(static_cast<std::size_t>(j) + 1, l.zero());
                basis_coeffs[static_cast<std::size_t>(j)] = *a_i;
                auto basis_elem = AlgebraicPoly::from_coeffs(l, std::move(basis_coeffs));
                if (!basis_elem) {
                    build_error = true;
                    berr = basis_elem.error();
                    break;
                }
                auto prod = gamma_prime.multiply(*basis_elem);
                if (!prod) {
                    build_error = true;
                    berr = prod.error();
                    break;
                }
                auto reduced = reduce_mod(*prod, g);
                if (!reduced) {
                    build_error = true;
                    berr = reduced.error();
                    break;
                }
                const std::vector<Rational> col = to_vector(*reduced, d, e);
                const std::size_t col_idx = static_cast<std::size_t>(i * e + j);
                for (std::size_t row = 0; row < un; ++row) {
                    rows[row][col_idx] = col[row];
                }
            }
        }
        if (build_error) {
            return make_error<AdjoinedRoot>(berr);
        }
        auto mmat = Matrix::from_rows(std::move(rows));
        if (!mmat) {
            return make_error<AdjoinedRoot>(mmat.error());
        }

        auto minpoly = minimal_polynomial(*mmat);
        if (!minpoly) {
            return make_error<AdjoinedRoot>(minpoly.error());
        }
        if (minpoly->degree() != n) {
            continue;  // gamma' not primitive: try the next c
        }

        auto field_res = NumberField::create(*minpoly);
        if (!field_res) {
            // A reducible minimal polynomial here means g was not actually irreducible
            // over L (see module header): surface that honestly rather than retry.
            return make_error<AdjoinedRoot>(field_res.error());
        }
        const NumberField m_field = std::move(*field_res);

        // Powers gamma'^0 .. gamma'^(n-1) as M0 elements, each an n-dim Q-vector: the
        // columns of the change-of-basis matrix V.
        std::vector<std::vector<Rational>> vrows(un, std::vector<Rational>(un));
        auto one_res = AlgebraicPoly::from_coeffs(l, std::vector<AlgebraicNumber>{l.one()});
        if (!one_res) {
            return make_error<AdjoinedRoot>(one_res.error());
        }
        AlgebraicPoly power = std::move(*one_res);
        bool pow_error = false;
        MathError perr = MathError::domain_error;
        for (std::int64_t col = 0; col < n; ++col) {
            const std::vector<Rational> pv = to_vector(power, d, e);
            for (std::size_t row = 0; row < un; ++row) {
                vrows[row][static_cast<std::size_t>(col)] = pv[row];
            }
            if (col + 1 < n) {
                auto prod = power.multiply(gamma_prime);
                if (!prod) {
                    pow_error = true;
                    perr = prod.error();
                    break;
                }
                auto reduced = reduce_mod(*prod, g);
                if (!reduced) {
                    pow_error = true;
                    perr = reduced.error();
                    break;
                }
                power = std::move(*reduced);
            }
        }
        if (pow_error) {
            return make_error<AdjoinedRoot>(perr);
        }
        auto vmat = Matrix::from_rows(std::move(vrows));
        if (!vmat) {
            return make_error<AdjoinedRoot>(vmat.error());
        }

        std::vector<std::vector<Rational>> brows(un, std::vector<Rational>(2));
        for (std::size_t row = 0; row < un; ++row) {
            brows[row][0] = target_alpha_vec[row];
            brows[row][1] = target_beta_vec[row];
        }
        auto bmat = Matrix::from_rows(std::move(brows));
        if (!bmat) {
            return make_error<AdjoinedRoot>(bmat.error());
        }
        auto xmat = vmat->solve(*bmat);  // V invertible: gamma' is primitive (checked above)
        if (!xmat) {
            return make_error<AdjoinedRoot>(xmat.error());
        }

        std::vector<Rational> alpha_coeffs(un);
        std::vector<Rational> beta_coeffs(un);
        for (std::size_t row = 0; row < un; ++row) {
            alpha_coeffs[row] = xmat->at(row, 0);
            beta_coeffs[row] = xmat->at(row, 1);
        }
        const RationalPoly p_alpha = RationalPoly::from_coeffs(std::move(alpha_coeffs));
        const RationalPoly p_beta = RationalPoly::from_coeffs(std::move(beta_coeffs));

        auto alpha_in_m = m_field.from_poly(p_alpha);
        if (!alpha_in_m) {
            return make_error<AdjoinedRoot>(alpha_in_m.error());
        }
        auto root_in_m = m_field.from_poly(p_beta);
        if (!root_in_m) {
            return make_error<AdjoinedRoot>(root_in_m.error());
        }

        // VERIFY before returning (mandatory, Rule 32): g's image over M vanishes at
        // `root`, and L's modulus vanishes at the image of L's generator in M.
        std::vector<AlgebraicNumber> g_in_m_coeffs;
        g_in_m_coeffs.reserve(static_cast<std::size_t>(g.degree()) + 1);
        bool embed_error = false;
        MathError eerr = MathError::domain_error;
        for (std::int64_t j = 0; j <= g.degree(); ++j) {
            const AlgebraicNumber cj = g.coefficient(static_cast<std::size_t>(j));
            auto img = reembed(cj, m_field, *alpha_in_m);
            if (!img) {
                embed_error = true;
                eerr = img.error();
                break;
            }
            g_in_m_coeffs.push_back(std::move(*img));
        }
        if (embed_error) {
            return make_error<AdjoinedRoot>(eerr);
        }
        auto g_in_m = AlgebraicPoly::from_coeffs(m_field, std::move(g_in_m_coeffs));
        if (!g_in_m) {
            return make_error<AdjoinedRoot>(g_in_m.error());
        }
        auto g_at_root = g_in_m->evaluate(*root_in_m);
        if (!g_at_root) {
            return make_error<AdjoinedRoot>(g_at_root.error());
        }
        if (!g_at_root->is_zero()) {
            return make_error<AdjoinedRoot>(MathError::domain_error);
        }

        const AlgebraicPoly h_in_m = AlgebraicPoly::embed(m_field, l.modulus());
        auto h_at_alpha = h_in_m.evaluate(*alpha_in_m);
        if (!h_at_alpha) {
            return make_error<AdjoinedRoot>(h_at_alpha.error());
        }
        if (!h_at_alpha->is_zero()) {
            return make_error<AdjoinedRoot>(MathError::domain_error);
        }

        return AdjoinedRoot{.field = m_field, .old_generator = std::move(*alpha_in_m),
                             .root = std::move(*root_in_m)};
    }
    return make_error<AdjoinedRoot>(MathError::not_implemented);
}

// --- splitting_field ----------------------------------------------------------------

auto splitting_field(std::span<const RationalPoly> irreducibles, std::int64_t max_degree)
    -> Result<SplittingField> {
    const std::size_t m = irreducibles.size();
    for (const RationalPoly& p : irreducibles) {
        if (p.degree() < 1) {
            return make_error<SplittingField>(MathError::domain_error);
        }
    }

    // Canonical order (degree ascending, then coefficient-lexicographic): used only to
    // pick a deterministic starting field and adjunction order. Results are reported back
    // in the CALLER's original order (below).
    std::vector<std::size_t> order(m);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::ranges::sort(order, [&](std::size_t a, std::size_t b) {
        return poly_less(irreducibles[a], irreducibles[b]);
    });

    // The trivial degree-1 field Q[t]/(t), representing Q itself, for the all-linear case.
    auto trivial_res = NumberField::create(RationalPoly::monomial(Rational::from_int(1), 1));
    if (!trivial_res) {
        return make_error<SplittingField>(trivial_res.error());
    }
    NumberField field = std::move(*trivial_res);

    std::optional<std::size_t> first_nonlinear;
    for (std::size_t idx : order) {
        if (irreducibles[idx].degree() >= 2) {
            first_nonlinear = idx;
            break;
        }
    }
    if (first_nonlinear) {
        auto f0 = NumberField::create(irreducibles[*first_nonlinear]);
        if (!f0) {
            return make_error<SplittingField>(f0.error());
        }
        field = std::move(*f0);
        if (field.degree() > max_degree) {
            return make_error<SplittingField>(MathError::not_implemented);
        }
    }

    // Bounded outer loop: each adjunction at least doubles [L:Q] (deg g >= 2), so at most
    // O(log2(max_degree)) extensions are ever needed. This cap is a defensive circuit
    // breaker, not the honesty gate itself -- that gate is the max_degree check below.
    const std::int64_t iteration_cap =
        std::max<std::int64_t>(max_degree, 1) + static_cast<std::int64_t>(m) + 8;

    std::vector<std::vector<AlgebraicPoly>> per_factor(m);  // refreshed every pass
    for (std::int64_t iter = 0;; ++iter) {
        if (iter > iteration_cap) {
            return make_error<SplittingField>(MathError::not_implemented);
        }
        std::optional<AlgebraicPoly> pending_nonlinear;
        for (std::size_t idx : order) {
            const AlgebraicPoly embedded = AlgebraicPoly::embed(field, irreducibles[idx]);
            auto fres = factor_over_field(field, embedded);
            if (!fres) {
                return make_error<SplittingField>(fres.error());
            }
            per_factor[idx] = std::move(*fres);
            if (!pending_nonlinear) {
                for (const AlgebraicPoly& piece : per_factor[idx]) {
                    if (piece.degree() >= 2) {
                        pending_nonlinear = piece;
                        break;
                    }
                }
            }
        }
        if (!pending_nonlinear) {
            break;  // every factor splits into linear pieces over `field`
        }
        auto deg_check = checked_mul_i64(field.degree(), pending_nonlinear->degree());
        if (!deg_check || *deg_check > max_degree) {
            return make_error<SplittingField>(MathError::not_implemented);
        }
        auto adjoined = adjoin_root(field, *pending_nonlinear);
        if (!adjoined) {
            return make_error<SplittingField>(adjoined.error());
        }
        field = std::move(adjoined->field);
    }

    // Harvest roots per factor and GUARD: each factor must contribute exactly deg_i
    // pairwise-distinct linear pieces over the final field.
    std::vector<std::pair<RationalPoly, std::vector<AlgebraicNumber>>> roots(m);
    for (std::size_t idx = 0; idx < m; ++idx) {
        std::vector<AlgebraicNumber> factor_roots;
        factor_roots.reserve(per_factor[idx].size());
        for (const AlgebraicPoly& piece : per_factor[idx]) {
            if (piece.degree() != 1) {
                return make_error<SplittingField>(MathError::domain_error);  // unreachable
            }
            const AlgebraicNumber c0 = piece.coefficient(0);
            auto neg = c0.negate();
            if (!neg) {
                return make_error<SplittingField>(neg.error());
            }
            factor_roots.push_back(std::move(*neg));  // piece is monic: root = -c0
        }
        if (static_cast<std::int64_t>(factor_roots.size()) != irreducibles[idx].degree()) {
            return make_error<SplittingField>(MathError::domain_error);
        }
        for (std::size_t a = 0; a < factor_roots.size(); ++a) {
            for (std::size_t b = a + 1; b < factor_roots.size(); ++b) {
                if (factor_roots[a].is_equal(factor_roots[b])) {
                    return make_error<SplittingField>(MathError::domain_error);  // not distinct
                }
            }
        }
        roots[idx] = std::make_pair(irreducibles[idx], std::move(factor_roots));
    }

    return SplittingField{.field = std::move(field), .roots = std::move(roots)};
}

}  // namespace nimblecas
